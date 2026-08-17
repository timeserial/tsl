"""Hierarchy of timescales.

The step 1 version had a single temporal model, at the top. All the levels
below were instantaneous: they knew *what* was happening but had no memory
at all of their own past. The cortex is not like that.

Hasson et al. measured "temporal receptive windows" that grow along the
hierarchy: primary auditory cortex integrates over tens of milliseconds,
frontal areas over tens of seconds. Kiebel, Daunizeau & Friston (2008,
"A hierarchy of time-scales and the brain") show that this falls naturally
out of a hierarchical predictive model where each level has its own,
progressively slower dynamics.

Here, each latent level gains:

    ẑ_l(t) = f( A_l · z_l(t-1) )        with   A_l = (1-λ_l)·I + λ_l·B_l

λ_l = 1/τ_l is the level's rate. τ grows with l, so the top barely moves
between frames (long memory) and the low levels follow the signal (short
memory). The identity part is the leaky integrator; B_l is what gets learned.

`kind="diagonal"` gives each unit its own time constant and nothing more -
it is the cheapest version and the most literally biological (the membrane
time constant). `kind="dense"` lets the levels mix within themselves.
"""

from __future__ import annotations

import numpy as np

from .dtypes import F


class Transition:
    """The prediction a level makes of itself at the next instant."""

    __slots__ = ("n", "kind", "lam", "B", "a", "_f", "_fprime")

    def __init__(
        self,
        n: int,
        kind: str,
        lam: float,
        rng: np.random.Generator,
        activation_pair,
        init_scale: float = 0.1,
    ) -> None:
        if kind not in ("none", "diagonal", "dense"):
            raise ValueError(f"transição desconhecida: {kind!r}")
        self.n = int(n)
        self.kind = kind
        self.lam = float(lam)
        self._f, self._fprime = activation_pair

        if kind == "dense":
            self.B = (
                np.eye(n, dtype=F)
                + (rng.standard_normal((n, n)) * (init_scale / np.sqrt(n))).astype(F)
            )
        elif kind == "diagonal":
            # One time constant per unit. Starts at 1 (keep the state) with
            # a pinch of variability, like real neurons.
            self.B = (1.0 + init_scale * rng.standard_normal(n)).astype(F)
        else:
            self.B = np.zeros(0, dtype=F)

        self.a = np.zeros(n, dtype=F)

    # ------------------------------------------------------------------
    @property
    def n_params(self) -> int:
        return int(self.B.size)

    def predict(self, z_prev: np.ndarray) -> np.ndarray:
        """ẑ(t) = f((1-λ)·z(t-1) + λ·B·z(t-1))."""
        if self.kind == "none":
            self.a[:] = z_prev
            return self.a
        if self.kind == "diagonal":
            mixed = self.B * z_prev
        else:
            mixed = self.B.dot(z_prev)
        self.a[:] = (1.0 - self.lam) * z_prev + self.lam * mixed
        return self._f(self.a).astype(F, copy=False)

    def learn(
        self,
        eps: np.ndarray,
        z_prev: np.ndarray,
        lr: float,
        grad_clip: float = 0.0,
    ) -> None:
        """ΔB ∝ (ε ⊙ f') ⊗ z(t-1). Local: only the error here and the state from before."""
        if self.kind == "none" or lr <= 0.0:
            return
        mod = (eps * self._fprime(self.a) * self.lam).astype(F, copy=False)
        if self.kind == "diagonal":
            dB = mod * z_prev
        else:
            dB = np.outer(mod, z_prev).astype(F, copy=False)
        if grad_clip > 0.0:
            np.clip(dB, -grad_clip, grad_clip, out=dB)
        self.B += F(lr) * dB

    def sigma_max(self) -> float:
        """Largest singular value of A = (1-λ)I + λB, for the stability bound."""
        if self.kind == "none":
            return 1.0
        if self.kind == "diagonal":
            return float(np.max(np.abs((1.0 - self.lam) + self.lam * self.B)))
        A = (1.0 - self.lam) * np.eye(self.n, dtype=F) + self.lam * self.B
        return float(np.linalg.svd(A, compute_uv=False)[0])


def timescales(n_levels: int, base: float, ratio: float) -> tuple[float, ...]:
    """λ_l per level: fast at the bottom, slow at the top.

    `base` is the rate of the lowest latent level, `ratio` how much it slows
    per step. ratio=2 means each level integrates over twice the time of the
    one below - the progression Hasson measures in the cortex.
    """
    return tuple(min(1.0, base / (ratio**l)) for l in range(n_levels))
