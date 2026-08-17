"""Precision: how much each error deserves to be believed.

In the free-energy formulation (Friston), each prediction error comes with a
precision π - the inverse of that error's expected variance. The energy is
not ½‖ε‖², it is ½π‖ε‖². A level whose error is habitually large and
unpredictable deserves a low π: its complaints count less.

Friston identifies precision modulation with **attention**, and its
substrate with neuromodulation (acetylcholine, noradrenaline) adjusting the
postsynaptic gain. Here it is the same object: a per-level gain, learned
locally.

This also solves a concrete and not at all theoretical problem the previous
version had: the sensory level (64 units, raw signal) and the top (8 units,
abstract latent) have errors of completely different scales, and they shared
a single threshold θ tuned by hand. With precision, the threshold is applied
to the *normalized* error √π·ε - the same θ means the same thing
everywhere.

Update rule, from the gradient of the free energy with respect to log π:

    ∂F/∂logπ = ½(π·⟨ε²⟩ − 1)   ->   Δlogπ ∝ 1 − π·⟨ε²⟩

whose fixed point is π = 1/⟨ε²⟩. Local, one line, and no divisions on the
critical path if we store log π.
"""

from __future__ import annotations

import numpy as np

from .dtypes import F


class Precision:
    """A level's gain. Scalar per level, or one per unit."""

    __slots__ = ("n", "per_unit", "log_pi", "lr", "lo", "hi")

    def __init__(
        self,
        n: int,
        per_unit: bool = False,
        lr: float = 0.01,
        init: float = 1.0,
        lo: float = 1e-2,
        hi: float = 1e3,
    ) -> None:
        self.n = int(n)
        self.per_unit = bool(per_unit)
        shape = (n,) if per_unit else (1,)
        self.log_pi = np.full(shape, np.log(init), dtype=F)
        self.lr = float(lr)
        self.lo, self.hi = float(lo), float(hi)

    @property
    def value(self) -> np.ndarray:
        return np.exp(self.log_pi).astype(F, copy=False)

    @property
    def scalar(self) -> float:
        """A representative precision for the level (for the stability bound)."""
        return float(np.exp(self.log_pi).max())

    def weight(self, eps: np.ndarray) -> np.ndarray:
        """π·ε - the error as it counts for the energy and for the state."""
        return (self.value * eps).astype(F, copy=False)

    def normalize(self, eps: np.ndarray) -> np.ndarray:
        """√π·ε - the error in standard-deviation units, for the threshold."""
        return (np.sqrt(self.value) * eps).astype(F, copy=False)

    def learn(self, eps: np.ndarray) -> None:
        """Δlogπ ∝ 1 − π·ε². Fixed point at π = 1/⟨ε²⟩."""
        if self.lr <= 0.0:
            return
        e2 = eps * eps
        if not self.per_unit:
            e2 = np.array([float(e2.mean())], dtype=F)
        grad = 1.0 - self.value * e2
        self.log_pi += F(self.lr) * grad.astype(F, copy=False)
        np.clip(self.log_pi, np.log(self.lo), np.log(self.hi), out=self.log_pi)

    def rescale(self, log_offset: float) -> None:
        """Shifts log π by a constant common to the whole network.

        The *absolute* scale of the precisions has no meaning: multiplying
        them all by a constant only rescales the energy, and the minimum is
        the same. What has meaning is the ratios between levels.

        But settling is gradient descent with a fixed step, which is not
        scale invariant: π rising inflates the Hessian, the adaptive step
        shrinks to avoid diverging, and inference starves. That is exactly
        what happened - π reached 10³ and the NRMSE 1.0. Anchoring the
        scale to a reference level solves it, losing nothing.
        """
        self.log_pi -= F(log_offset)
        np.clip(self.log_pi, np.log(self.lo), np.log(self.hi), out=self.log_pi)

    def __repr__(self) -> str:  # pragma: no cover
        v = self.value
        return f"Precision(n={self.n}, π={v.mean():.3g})"


class UnitPrecision:
    """Precision fixed at 1. The no-precision case, with no branches in the code."""

    __slots__ = ()
    value = F(1.0)
    scalar = 1.0

    def weight(self, eps):
        return eps

    def normalize(self, eps):
        return eps

    def learn(self, eps):
        return None

    def rescale(self, log_offset):
        return None
