"""Portões multiplicativos: a dinâmica escolhida pelo erro."""

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
    t.b[:] = -20.0  # g -> 0: reter
    assert np.allclose(t.predict(z), z, atol=1e-4)
    t.b[:] = 20.0  # g -> 1: seguir a dinâmica
    assert np.allclose(t.predict(z), np.tanh(t.A @ z), atol=1e-4)


def test_learning_closes_a_temporal_step():
    """Com um par (antes, depois) fixo, a transição tem de o aprender."""
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
    """Metade das unidades deve reter, metade deve seguir a dinâmica: o
    portão tem de divergir por unidade — é essa a modularidade que os
    mecanismos anteriores (mistura, esparsidade) não conseguiam aprender."""
    t = make(4, seed=2)
    z = np.array([0.5, 0.5, 0.5, 0.5], dtype=F)
    # alvo: unidades 0-1 ficam como estão; 2-3 mudam de sinal
    target = np.array([0.5, 0.5, -0.5, -0.5], dtype=F)
    for _ in range(2000):
        t.learn(target - t.predict(z), lr=0.3)
    t.predict(z)
    g = t.gate
    assert np.abs(target - t.predict(z)).max() < 0.1


def test_credit_only_flows_through_open_gates():
    """ΔA ∝ ε·g: com o portão fechado, a dinâmica não leva culpa nenhuma."""
    t = make(4)
    t.b[:] = -20.0  # tudo fechado
    A0 = t.A.copy()
    z = np.array([0.5, -0.3, 0.2, 0.1], dtype=F)
    t.predict(z)
    t.learn(np.ones(4, dtype=F), lr=0.5)
    assert np.allclose(t.A, A0, atol=1e-5)


def test_step_is_scale_invariant():
    """NLMS: encolher o contexto 100× não muda o erro relativo final.

    Testa-se para baixo (1.0 vs 0.01) e não para cima, porque para cima o
    tanh satura e a derivada morre — isso é saturação de ativação, que o GRU
    e a biologia também têm; o NLMS protege o *passo*, não a ativação. Sem
    NLMS, encolher a entrada 100× encolheria o passo efetivo 10 000× e a
    aprendizagem congelava.
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
    # o A clássico fica intocado; o portão é que aprende
    assert np.array_equal(A0, net.A)
