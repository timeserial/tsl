"""Escalas de tempo hierárquicas."""

import numpy as np
import pytest

from pcnet import activations
from pcnet.dtypes import F
from pcnet.temporal import Transition, timescales


def make(kind="dense", n=8, lam=0.5, seed=0):
    return Transition(n, kind, lam, np.random.default_rng(seed),
                      activations.get("tanh"))


def test_timescales_slow_down_going_up():
    lams = timescales(4, base=1.0, ratio=2.0)
    assert lams == (1.0, 0.5, 0.25, 0.125)
    assert all(a > b for a, b in zip(lams, lams[1:]))


def test_timescales_never_exceed_one():
    assert all(v <= 1.0 for v in timescales(3, base=5.0, ratio=2.0))


def test_parameter_cost_diagonal_vs_dense():
    """A versão diagonal é uma constante de tempo por unidade, e mais nada."""
    assert make("diagonal", n=16).n_params == 16
    assert make("dense", n=16).n_params == 256
    assert make("none", n=16).n_params == 0


def test_no_transition_is_pure_retention():
    t = make("none", n=4)
    z = np.array([0.1, -0.2, 0.3, 0.4], dtype=F)
    assert np.array_equal(t.predict(z), z)


def test_small_lambda_means_long_memory():
    """λ→0: o nível guarda o que tinha, seja o que for que B diga."""
    slow = make("dense", n=6, lam=0.02)
    z = np.full(6, 0.5, dtype=F)
    assert np.allclose(slow.predict(z), np.tanh(z), atol=0.05)


def test_learning_closes_a_temporal_pattern():
    """Com um par (antes, depois) fixo, a transição tem de aprender o passo."""
    t = make("dense", n=6, lam=1.0)
    z_prev = np.array([0.5, -0.3, 0.2, 0.1, -0.4, 0.0], dtype=F)
    target = np.array([-0.3, 0.5, 0.1, 0.2, 0.0, -0.4], dtype=F)

    errors = []
    for _ in range(400):
        pred = t.predict(z_prev)
        eps = target - pred
        errors.append(float(np.abs(eps).sum()))
        t.learn(eps, z_prev, lr=0.05)
    assert errors[-1] < 0.1 * errors[0]


def test_diagonal_cannot_permute_but_dense_can():
    """Uma constante de tempo por unidade só escala; não roda nem mistura.

    É por isto que a versão diagonal não serve para sinais oscilatórios: o que
    é preciso prever é um avanço de fase, que é uma rotação.
    """
    z_prev = np.array([1.0, 0.0], dtype=F)
    target = np.array([0.0, 1.0], dtype=F)

    def residual(kind):
        t = Transition(2, kind, 1.0, np.random.default_rng(0),
                       activations.get("identity"))
        for _ in range(500):
            t.learn(target - t.predict(z_prev), z_prev, lr=0.05)
        return float(np.abs(target - t.predict(z_prev)).sum())

    assert residual("diagonal") > 0.5
    assert residual("dense") < 0.05


def test_sigma_max_matches_the_true_singular_value():
    t = make("dense", n=10, lam=0.7)
    A = (1 - t.lam) * np.eye(10, dtype=F) + t.lam * t.B
    assert t.sigma_max() == pytest.approx(
        float(np.linalg.svd(A, compute_uv=False)[0]), rel=1e-4
    )


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match="transição desconhecida"):
        make("quantum")
