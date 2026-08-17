import numpy as np
import pytest

from pcnet.dtypes import F
from pcnet.layer import PCLayer


def make_layer(n_below=6, n_above=4, act="tanh", seed=0):
    return PCLayer(n_below, n_above, act, np.random.default_rng(seed))


def test_predict_matches_definition():
    lay = make_layer()
    z = np.random.default_rng(1).standard_normal(4).astype(F)
    assert np.allclose(lay.predict(z), np.tanh(lay.W @ z), atol=1e-6)


def test_identity_layer_is_linear():
    lay = make_layer(act="identity")
    z = np.random.default_rng(2).standard_normal(4).astype(F)
    assert np.allclose(lay.predict(z), lay.W @ z, atol=1e-6)


def test_everything_is_float32():
    lay = make_layer()
    z = np.ones(4, dtype=F)
    assert lay.W.dtype == F
    assert lay.predict(z).dtype == F
    assert lay.backward(lay.modulated_error(np.ones(6, dtype=F))).dtype == F


def test_backward_is_the_transpose_path():
    lay = make_layer()
    z = np.random.default_rng(3).standard_normal(4).astype(F)
    lay.predict(z)
    eps = np.random.default_rng(4).standard_normal(6).astype(F)
    mod = lay.modulated_error(eps)
    assert np.allclose(mod, eps * (1 - np.tanh(lay.W @ z) ** 2), atol=1e-5)
    assert np.allclose(lay.backward(mod), lay.W.T @ mod, atol=1e-5)


def test_learning_reduces_error_on_a_fixed_pattern():
    """Local Hebbian rule: with z fixed, ΔW ∝ ε·zᵀ has to close the error."""
    lay = make_layer(act="identity")
    z = np.ones(4, dtype=F)
    target = np.linspace(-1, 1, 6).astype(F)

    errors = []
    for _ in range(200):
        zhat = lay.predict(z)
        eps = target - zhat
        errors.append(float(np.abs(eps).sum()))
        lay.learn(lay.modulated_error(eps), z, lr=0.05)

    assert errors[-1] < 1e-3 * errors[0]


def test_learning_uses_only_local_quantities():
    """ΔW depends only on ε (in the layer) and z (immediately above)."""
    lay = make_layer(act="identity")
    z = np.array([1.0, 0.0, 0.0, 0.0], dtype=F)
    W0 = lay.W.copy()
    lay.learn(np.ones(6, dtype=F), z, lr=0.1)
    delta = lay.W - W0
    # z is zero except in column 0 -> only that column can change.
    assert np.allclose(delta[:, 1:], 0.0)
    assert np.all(delta[:, 0] > 0)


def test_grad_clip_bounds_the_update():
    lay = make_layer(act="identity")
    z = np.full(4, 100.0, dtype=F)
    W0 = lay.W.copy()
    lay.learn(np.full(6, 100.0, dtype=F), z, lr=1.0, grad_clip=0.5)
    assert np.max(np.abs(lay.W - W0)) <= 0.5 + 1e-6


def test_mac_accounting_scales_with_sparsity():
    lay = make_layer(n_below=6, n_above=4)
    assert lay.macs_down() == 24
    assert lay.macs_up(0) == 0
    assert lay.macs_up(2) == 8
    assert lay.macs_up(6) == lay.macs_down()


def test_unknown_activation_is_rejected():
    with pytest.raises(ValueError, match="ativação desconhecida"):
        make_layer(act="relu6000")
