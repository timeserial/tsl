"""Gated temporal transition: the dynamics chosen by the error.

Where this comes from, because it is the most important decomposition we
measured: a GRU forbidden from assigning credit through time (state detached
at every step) scores 0.576 on the three tasks where we score 0.778 and full
BPTT scores 0.490. In other words, most of the wall is not in time - it is in
what the update does with the error of *one* instant. And the only relevant
structural difference in its cell is the **multiplicative gate**: deciding,
as a function of context, how much of each state unit to update, with credit
flowing through that decision.

Our previous attempts at modularity failed precisely for lacking this: the
mixture of dynamics chose by similarity (k-means, no credit), sparsity cut by
magnitude (blind to the error). The GRU's gate is chosen *by the error*. And
multiplication between signals is no engineering trick - shunting inhibition,
dendritic gating, the thalamus gating the cortex: it is among the best
documented mechanisms in neurophysiology.

The top's transition goes from

    ẑ(t) = (1-λ)·z + λ·tanh(A·z)          λ fixed, the same for everything

to the version where λ is learned and depends on the context:

    c = tanh(A·z)                          the candidate (the dynamics)
    g = σ(G·z + b)                         the gate (how much to update)
    ẑ(t) = (1-g)⊙z + g⊙c

Credit is still local. With ε = z_settled − ẑ coming from settling:

    ∂ẑ/∂c = g          ->  ΔA ∝ (ε ⊙ g ⊙ (1-c²)) ⊗ z
    ∂ẑ/∂g = c − z      ->  ΔG ∝ (ε ⊙ (c−z) ⊙ g(1-g)) ⊗ z

Each factor exists in the neuron: the error, the candidate, the gate value.
It is a three-factor rule (pre × post × modulator), which is the current
consensus on how biological plasticity actually works. Steps normalized by
‖z‖² (NLMS) - a lesson learned three times in this session.
"""

from __future__ import annotations

import numpy as np

from .dtypes import F


def _sigmoid(a: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-a))).astype(F, copy=False)


class GatedTransition:
    """ẑ(t) = (1-g)⊙z + g⊙tanh(A·z), with g = σ(G·z + b) learned by the error."""

    __slots__ = ("n", "A", "G", "b", "_z", "_c", "_g", "eA", "eG", "eb",
                 "eligibility", "B", "H", "_u")

    def __init__(self, n: int, rng: np.random.Generator, gate_bias: float = 0.0,
                 eligibility: bool = False, input_dim: int = 0):
        self.n = int(n)
        self.A = (
            np.eye(n, dtype=F)
            + (rng.standard_normal((n, n)) * (0.2 / np.sqrt(n))).astype(F)
        )
        self.G = (rng.standard_normal((n, n)) * (0.3 / np.sqrt(n))).astype(F)
        # initial gate bias: 0 -> g≈0.5, half open, no prejudice.
        self.b = np.full(n, gate_bias, dtype=F)
        self._z = np.zeros(n, dtype=F)
        self._c = np.zeros(n, dtype=F)
        self._g = np.full(n, 0.5, dtype=F)

        # Eligibility traces (e-prop / synaptic tagging): each synapse keeps
        # a trace of what it just did, and the error that arrives while the
        # trace lasts credits it - credit through time, without BPTT. The
        # trace decays at the gate's own retention rate (1-g): a unit that
        # retains a lot propagates credit far; one that always updates is
        # only credited for the instant. The memory of credit and the memory
        # of state are the same thing, as they should be.
        self.eligibility = bool(eligibility)
        self.eA = np.zeros((n, n), dtype=F) if eligibility else None
        self.eG = np.zeros((n, n), dtype=F) if eligibility else None
        self.eb = np.zeros(n, dtype=F) if eligibility else None

        # The thalamic relay: the candidate and the gate also see the
        # previous input, not only the recurrent state - c = tanh(A·z + B·u),
        # g = σ(G·z + H·u + b). It is the last structural difference to the
        # GRU cell, and in cortex it is literal: the input arrives in
        # parallel with the recurrence. Credit still local; u has its own
        # norm (NLMS).
        if input_dim > 0:
            self.B = (rng.standard_normal((n, input_dim))
                      * (0.2 / np.sqrt(input_dim))).astype(F)
            self.H = (rng.standard_normal((n, input_dim))
                      * (0.3 / np.sqrt(input_dim))).astype(F)
        else:
            self.B = None
            self.H = None
        self._u = np.zeros(max(input_dim, 1), dtype=F)

    # ------------------------------------------------------------------
    @property
    def n_params(self) -> int:
        return int(self.A.size + self.G.size + self.b.size)

    @property
    def gate(self) -> np.ndarray:
        """The gate of the last prediction - instrumentation."""
        return self._g

    def predict(self, z_prev: np.ndarray,
                u: np.ndarray | None = None) -> np.ndarray:
        self._z[:] = z_prev
        a_c = self.A.dot(z_prev)
        a_g = self.G.dot(z_prev) + self.b
        if self.B is not None and u is not None:
            self._u[:] = u
            a_c = a_c + self.B.dot(u)
            a_g = a_g + self.H.dot(u)
        self._c[:] = np.tanh(a_c)
        self._g[:] = _sigmoid(a_g)
        return ((1.0 - self._g) * z_prev + self._g * self._c).astype(F, copy=False)

    def learn(self, eps: np.ndarray, lr: float, grad_clip: float = 0.0) -> None:
        """Three factors, all local, step normalized by ‖z‖².

        With eligibility, the instantaneous product is accumulated into a
        per-synapse trace that decays at (1-g), and it is the trace that the
        error credits.
        """
        z, c, g = self._z, self._c, self._g
        norm = float(np.dot(z, z)) + 1e-6

        inst_c = np.outer((g * (1.0 - c * c)) / norm, z).astype(F, copy=False)
        inst_g_vec = ((c - z) * g * (1.0 - g)).astype(F, copy=False)
        inst_g = np.outer(inst_g_vec / norm, z).astype(F, copy=False)

        if self.eligibility:
            retain = (1.0 - g)[:, None]
            self.eA *= retain
            self.eA += inst_c
            self.eG *= retain
            self.eG += inst_g
            self.eb *= (1.0 - g)
            self.eb += inst_g_vec
            dA = (eps[:, None] * self.eA).astype(F, copy=False)
            dG = (eps[:, None] * self.eG).astype(F, copy=False)
            db = (eps * self.eb).astype(F, copy=False)
        else:
            dA = (eps[:, None] * inst_c).astype(F, copy=False)
            dG = (eps[:, None] * inst_g).astype(F, copy=False)
            db = (eps * inst_g_vec).astype(F, copy=False)

        if grad_clip > 0.0:
            np.clip(dA, -grad_clip, grad_clip, out=dA)
            np.clip(dG, -grad_clip, grad_clip, out=dG)
        self.A += F(lr) * dA
        self.G += F(lr) * dG
        self.b += F(lr) * db

        if self.B is not None:
            u = self._u
            un = float(np.dot(u, u)) + 1e-6
            mod_c = (eps * g * (1.0 - c * c)).astype(F, copy=False)
            mod_g = ((eps * (c - z) * g * (1.0 - g))).astype(F, copy=False)
            dB = np.outer(mod_c / un, u).astype(F, copy=False)
            dH = np.outer(mod_g / un, u).astype(F, copy=False)
            if grad_clip > 0.0:
                np.clip(dB, -grad_clip, grad_clip, out=dB)
                np.clip(dH, -grad_clip, grad_clip, out=dH)
            self.B += F(lr) * dB
            self.H += F(lr) * dH

    def sigma_max(self) -> float:
        """Bound for the settling step. The Jacobian of the mixture is
        (1-g)·I + g·diag(1-c²)·A plus gate terms; we bound it by the worst
        case g=1, which is σ_max(A). Conservative and cheap."""
        return float(np.linalg.svd(self.A, compute_uv=False)[0])
