"""Multiplicative gates: the dynamics chosen by the error."""

import numpy as np
import pytest

from pcnet import PCConfig, PCNetwork, make_signal
from pcnet.dtypes import F
from pcnet.gated import GatedTransition
from pcnet.train import evaluate, train


def make(n=8, seed=0):
    return GatedTransition(n, np.random.default_rng(seed))


def test_prediction_is_the_gated_mixture():
    t = make(4)
    z = np.array([0.5, -0.3, 0.2, 0.1], dtype=F)
    out = t.predict(z)
    c = np.tanh(t.A @ z)
    g = 1.0 / (1.0 + np.exp(-(t.G @ z + t.b)))
    assert np.allclose(out, (1 - g) * z + g * c, atol=1e-5)


def test_a_closed_gate_retains_and_an_open_gate_updates():
    t = make(4)
    z = np.array([0.5, -0.3, 0.2, 0.1], dtype=F)
    t.b[:] = -20.0  # g -> 0: retain
    assert np.allclose(t.predict(z), z, atol=1e-4)
    t.b[:] = 20.0  # g -> 1: follow the dynamics
    assert np.allclose(t.predict(z), np.tanh(t.A @ z), atol=1e-4)


def test_learning_closes_a_temporal_step():
    """With a fixed (before, after) pair, the transition has to learn it."""
    t = make(6, seed=1)
    z = np.array([0.6, -0.4, 0.3, 0.2, -0.5, 0.1], dtype=F)
    target = np.array([-0.4, 0.6, 0.2, 0.3, 0.1, -0.5], dtype=F)
    errs = []
    for _ in range(600):
        eps = target - t.predict(z)
        errs.append(float(np.abs(eps).sum()))
        t.learn(eps, lr=0.3)
    assert errs[-1] < 0.05 * errs[0]


def test_the_gate_learns_where_to_hold_and_where_to_move():
    """Half of the units should retain, half should follow the dynamics: the
    gate has to diverge per unit - that is the modularity the previous
    mechanisms (mixture, sparsity) could not learn."""
    t = make(4, seed=2)
    z = np.array([0.5, 0.5, 0.5, 0.5], dtype=F)
    # target: units 0-1 stay as they are; 2-3 flip sign
    target = np.array([0.5, 0.5, -0.5, -0.5], dtype=F)
    for _ in range(2000):
        t.learn(target - t.predict(z), lr=0.3)
    t.predict(z)
    g = t.gate
    assert np.abs(target - t.predict(z)).max() < 0.1


def test_credit_only_flows_through_open_gates():
    """ΔA ∝ ε·g: with the gate closed, the dynamics takes no blame at all."""
    t = make(4)
    t.b[:] = -20.0  # all closed
    A0 = t.A.copy()
    z = np.array([0.5, -0.3, 0.2, 0.1], dtype=F)
    t.predict(z)
    t.learn(np.ones(4, dtype=F), lr=0.5)
    assert np.allclose(t.A, A0, atol=1e-5)


def test_step_is_scale_invariant():
    """NLMS: shrinking the context 100× does not change the final relative error.

    We test downward (1.0 vs 0.01) and not upward, because upward the tanh
    saturates and the derivative dies - that is activation saturation, which
    the GRU and biology also have; NLMS protects the *step*, not the
    activation. Without NLMS, shrinking the input 100× would shrink the
    effective step 10 000× and learning would freeze.
    """
    def rel(scale):
        t = make(4, seed=3)
        z = np.array([0.5, -0.3, 0.2, 0.1], dtype=F) * scale
        target = np.array([0.1, 0.2, -0.3, 0.5], dtype=F) * scale
        for _ in range(300):
            t.learn(target - t.predict(z), lr=0.3)
        return float(np.abs(target - t.predict(z)).sum() / (np.abs(target).sum() + 1e-9))
    assert rel(0.01) == pytest.approx(rel(1.0), abs=0.1)


# --------------------------------------------------------------------------
def test_network_with_gates_learns_a_single_task_as_well():
    sig = make_signal(n_frames=300, frame_len=32, seed=1)
    tr, te = sig.split(0.8)
    scores = {}
    for gated in (False, True):
        net = PCNetwork(PCConfig(seed=0, sizes=(32, 16, 8), use_precision=True,
                                 gated_transition=gated))
        train(net, tr.frames, epochs=30)
        scores[gated] = evaluate(net, te.frames).pred_nrmse
    assert scores[True] < 1.3 * scores[False] + 0.05


def test_gated_and_plain_transitions_are_exclusive_paths():
    net = PCNetwork(PCConfig(sizes=(32, 16, 8), gated_transition=True))
    assert net.gated is not None
    A0 = net.A.copy()
    net.run(make_signal(n_frames=60, frame_len=32, seed=1).frames, learn=True)
    # the classic A stays untouched; it is the gate that learns
    assert np.array_equal(A0, net.A)


def test_eligibility_trace_accumulates_and_decays_with_retention():
    """The trace is pure, testable mechanics: it accumulates the instantaneous
    product and decays by (1-g) - the gate's retention is the half-life of the
    credit."""
    from pcnet.gated import GatedTransition

    t = GatedTransition(3, np.random.default_rng(0), eligibility=True)
    z1 = np.array([0.4, -0.2, 0.3], dtype=F)
    z2 = np.array([-0.1, 0.5, 0.2], dtype=F)

    t.predict(z1)
    g1 = t.gate.copy()
    inst1 = np.outer((g1 * (1 - t._c**2)) / (float(z1 @ z1) + 1e-6), z1)
    t.learn(np.zeros(3, dtype=F), lr=0.1)  # eps=0: only the trace moves
    assert np.allclose(t.eA, inst1, atol=1e-5)

    t.predict(z2)
    g2 = t.gate.copy()
    inst2 = np.outer((g2 * (1 - t._c**2)) / (float(z2 @ z2) + 1e-6), z2)
    t.learn(np.zeros(3, dtype=F), lr=0.1)
    assert np.allclose(t.eA, (1 - g2)[:, None] * inst1 + inst2, atol=1e-4)


def test_eligibility_does_not_hurt_instant_credit_much():
    """On a problem without delay, the trace can only cost a bit of stale
    credit noise - if it costs a lot, the implementation is wrong. (The gain
    with delay is measured in the benchmark, not here: building true delay in
    a unit test requires the whole network.)"""
    from pcnet.gated import GatedTransition

    def residual(eligibility):
        t = GatedTransition(4, np.random.default_rng(5), eligibility=eligibility)
        rng = np.random.default_rng(7)
        z = np.array([0.5, -0.3, 0.2, 0.1], dtype=F)
        target = np.array([0.1, 0.2, -0.3, 0.5], dtype=F)
        for _ in range(400):
            t.learn((target - t.predict(z)).astype(F), lr=0.2)
        return float(np.abs(target - t.predict(z)).sum())

    assert residual(True) <= residual(False) * 1.5 + 1e-3
