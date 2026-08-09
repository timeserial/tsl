"""Precisão: o ganho com que cada erro conta."""

import numpy as np
import pytest

from pcnet.dtypes import F
from pcnet.precision import Precision, UnitPrecision


def test_precision_converges_to_the_inverse_variance():
    """π → 1/⟨ε²⟩. É o ponto fixo da regra e é o que a torna interpretável."""
    rng = np.random.default_rng(0)
    sigma = 0.25
    p = Precision(16, lr=0.05)
    for _ in range(4000):
        p.learn((rng.standard_normal(16) * sigma).astype(F))
    assert p.scalar == pytest.approx(1.0 / sigma**2, rel=0.25)


def test_a_noisy_channel_ends_up_trusted_less():
    rng = np.random.default_rng(1)
    quiet, noisy = Precision(8, lr=0.05), Precision(8, lr=0.05)
    for _ in range(2000):
        quiet.learn((rng.standard_normal(8) * 0.1).astype(F))
        noisy.learn((rng.standard_normal(8) * 1.0).astype(F))
    assert quiet.scalar > 10 * noisy.scalar


def test_per_unit_precision_separates_units():
    rng = np.random.default_rng(2)
    p = Precision(4, per_unit=True, lr=0.05)
    scale = np.array([0.1, 0.1, 2.0, 2.0], dtype=F)
    for _ in range(3000):
        p.learn((rng.standard_normal(4) * scale).astype(F))
    v = p.value
    assert v[0] > 10 * v[2] and v[1] > 10 * v[3]


def test_normalize_puts_errors_in_units_of_sigma():
    p = Precision(4, lr=0.0, init=4.0)
    eps = np.ones(4, dtype=F)
    assert np.allclose(p.normalize(eps), 2.0)
    assert np.allclose(p.weight(eps), 4.0)


def test_rescale_only_shifts_and_keeps_ratios():
    a, b = Precision(4, lr=0.0, init=100.0), Precision(4, lr=0.0, init=10.0)
    before = a.scalar / b.scalar
    a.rescale(np.log(100.0))
    b.rescale(np.log(100.0))
    assert a.scalar == pytest.approx(1.0, rel=1e-4)
    assert a.scalar / b.scalar == pytest.approx(before, rel=1e-4)


def test_precision_is_clipped_at_both_ends():
    p = Precision(4, lr=1.0, lo=0.5, hi=2.0)
    for _ in range(200):
        p.learn(np.zeros(4, dtype=F))  # erro nulo empurra π para cima
    assert p.scalar <= 2.0 + 1e-6
    for _ in range(200):
        p.learn(np.full(4, 100.0, dtype=F))  # erro enorme empurra para baixo
    assert p.scalar >= 0.5 - 1e-6


def test_unit_precision_is_a_no_op():
    u = UnitPrecision()
    eps = np.array([1.0, -2.0], dtype=F)
    assert np.array_equal(u.weight(eps), eps)
    assert np.array_equal(u.normalize(eps), eps)
    assert u.learn(eps) is None
    assert u.rescale(3.0) is None
    assert u.scalar == 1.0
