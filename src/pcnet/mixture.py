"""Several dynamics, chosen by context.

The diagnosis that led here, and it is worth having it written down: with a
single transition matrix at the top, the network cannot retain several tasks -
not even when it sees them all at once, nor with twice the parameters
(measured: 6848 -> 13824 parameters changes the result from 0.823 to 0.825,
that is, nothing).

The reason is structural, not one of size. A linear matrix has *one* set of
eigenvalues, hence one set of rotation frequencies. Two tasks with different
phase advances pull it toward an average that serves neither. In principle an
8×8 matrix has four pairs of eigenvalues and could host four dynamics in
orthogonal subspaces - but nothing in the local rule encourages that
separation, and so it does not happen.

The solution is to give it *several* dynamics and a mechanism to choose:

    ẑ_L(t) = f( Σ_k g_k · A_k · z_L(t-1) )      g = softmax(sim(z, c_k) / τ)

Each component has its own matrix `A_k` and its own context prototype `c_k`.
The prototype says "this dynamics applies when the world looks like this";
the responsibility `g_k` measures how much it does. It is a switched linear
dynamical system, and it is among the oldest and best studied models for
signals with several regimes.

Everything is still local. `A_k` learns from the same error as always,
weighted by its responsibility - whoever was responsible for the prediction
is who pays for the error. `c_k` follows the mean of the contexts in which it
was used, which is online k-means.

And it fits with episodic memory instead of competing with it: the prototypes
are keys, the responsibility is the retrieval. Knowing *which world you are
in* becomes the same operation as recognizing a context.
"""

from __future__ import annotations

import numpy as np

from .dtypes import F


class TopMixture:
    """Mixture of linear transitions at the top, with a context selector."""

    __slots__ = ("n", "k", "A", "protos", "tau", "_resp", "_z_prev", "a")

    def __init__(self, n: int, k: int, rng: np.random.Generator, tau: float = 0.25):
        self.n = int(n)
        self.k = max(1, int(k))
        self.tau = float(tau)
        # Each component starts near the identity with a different
        # perturbation: identical ones would collapse to the same solution.
        self.A = np.stack([
            np.eye(n, dtype=F)
            + (rng.standard_normal((n, n)) * (0.15 / np.sqrt(n))).astype(F)
            for _ in range(self.k)
        ])
        # Context prototypes, random and normalized.
        p = rng.standard_normal((self.k, n)).astype(F)
        self.protos = p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-8)
        self._resp = np.full(self.k, 1.0 / self.k, dtype=F)
        self._z_prev = np.zeros(n, dtype=F)
        self.a = np.zeros(n, dtype=F)

    # ------------------------------------------------------------------
    @property
    def n_params(self) -> int:
        return int(self.A.size + self.protos.size)

    @property
    def responsibilities(self) -> np.ndarray:
        return self._resp

    def responsibility(self, z_prev: np.ndarray) -> np.ndarray:
        """How much each dynamics applies to the current context."""
        if self.k == 1:
            self._resp[:] = 1.0
            return self._resp
        norm = float(np.linalg.norm(z_prev))
        if norm < 1e-8:
            self._resp[:] = 1.0 / self.k
            return self._resp
        sim = self.protos.dot(z_prev) / norm
        e = np.exp((sim - sim.max()) / max(self.tau, 1e-6))
        self._resp[:] = (e / e.sum()).astype(F)
        return self._resp

    def predict(self, z_prev: np.ndarray) -> np.ndarray:
        """Σ_k g_k · A_k · z(t-1). Returns the pre-activation."""
        g = self.responsibility(z_prev)
        self._z_prev[:] = z_prev
        np.dot(np.tensordot(g, self.A, axes=(0, 0)), z_prev, out=self.a)
        return self.a

    def learn(self, eps_mod: np.ndarray, lr: float, proto_lr: float = 0.02,
              grad_clip: float = 0.0) -> None:
        """Whoever was responsible for the prediction is who pays for the error.

        ΔA_k ∝ g_k · ε ⊗ z(t-1), and the prototype follows the mean of the
        contexts in which its component was used. Step normalized by ‖z‖²,
        for the same reason as always: a delta rule is only stable below
        1/‖x‖².
        """
        z = self._z_prev
        norm = float(np.dot(z, z)) + 1e-6
        outer = np.outer(eps_mod, z).astype(F, copy=False) / norm
        if grad_clip > 0.0:
            np.clip(outer, -grad_clip, grad_clip, out=outer)
        for i in range(self.k):
            g = float(self._resp[i])
            if g > 1e-4:
                self.A[i] += F(lr * g) * outer
                # online k-means: the prototype moves toward the context in
                # which it was used, in proportion to how much it was used.
                self.protos[i] += F(proto_lr * g) * (z - self.protos[i])
        n = np.linalg.norm(self.protos, axis=1, keepdims=True)
        self.protos /= np.maximum(n, 1e-8)

    def sigma_max(self) -> float:
        """Largest singular value of the current effective mixture."""
        mixed = np.tensordot(self._resp, self.A, axes=(0, 0))
        return float(np.linalg.svd(mixed, compute_uv=False)[0])

    def usage(self) -> np.ndarray:
        return self._resp.copy()
