"""The base unit: a predictive layer.

    ẑ_l = f(W_l · z_{l+1})        # the layer above PREDICTS the one below
    ε_l = z_l − ẑ_l               # only the error rises
    Δz_{l+1} ∝ W_lᵀ · (ε_l ⊙ f')  # the state above adjusts itself
    ΔW_l ∝ (ε_l ⊙ f') · z_{l+1}ᵀ  # local Hebbian learning

Nothing here sees the rest of the network: `learn` uses only the error that
is in the layer and the state immediately above - the two quantities that
physically exist at the synapse. That restriction is what makes the C
trivial (no graph, no stored activations) and what allows training on analog
hardware.

Shape convention: W has shape (n_below, n_above), C-contiguous, float32 -
the same memory the crossbar will have, row = neuron below.
"""

from __future__ import annotations

import numpy as np

from . import activations
from .dtypes import F


class PCLayer:
    """Generator of one level: predicts `n_below` from `n_above`."""

    __slots__ = (
        "n_below", "n_above", "W", "W_eff", "device",
        "act_name", "_f", "_fprime", "a", "zhat",
        "sigma_max", "_pi_v", "importance", "Gg", "bg", "_gate", "_base",
    )

    def __init__(
        self,
        n_below: int,
        n_above: int,
        activation: str,
        rng: np.random.Generator,
        init_scale: float = 1.0,
        gated: bool = False,
    ) -> None:
        self.n_below = int(n_below)
        self.n_above = int(n_above)
        self.act_name = activation
        self._f, self._fprime = activations.get(activation)
        self.device = None

        # ~1/sqrt(fan_in): keeps the pre-activation at O(1) and the tanh out
        # of saturation, which is where the derivative dies and local
        # learning stops.
        scale = init_scale / np.sqrt(self.n_above)
        self.W = (rng.standard_normal((self.n_below, self.n_above)) * scale).astype(F)
        # Without a device, W_eff is W itself (same memory, zero cost).
        self.W_eff = self.W

        # Preallocated buffers (the C will have no malloc; neither does the
        # Python).
        self.a = np.zeros(self.n_below, dtype=F)
        self.zhat = np.zeros(self.n_below, dtype=F)

        # Estimate of σ_max(W_eff) by power iteration, so the settling step
        # knows how far it can go without diverging.
        self._pi_v = (rng.standard_normal(self.n_above)).astype(F)
        self._pi_v /= np.linalg.norm(self._pi_v) or 1.0
        self.sigma_max = 0.0
        self._estimate_sigma_max()

        # Importance of each synapse: running mean of the square of its own
        # local gradient. It is the diagonal of the Fisher information,
        # available here at the synapse with no backward pass.
        self.importance = np.zeros_like(self.W)

        # Output gate (optional): decides per unit how much of the generated
        # prediction applies in this context. bias +2 -> g≈0.88: born almost
        # open, to start like the ungated layer and only close where the
        # error justifies it.
        if gated:
            self.Gg = (rng.standard_normal((self.n_below, self.n_above))
                       * (0.3 / np.sqrt(self.n_above))).astype(F)
            self.bg = np.full(self.n_below, 2.0, dtype=F)
        else:
            self.Gg = None
            self.bg = None
        self._gate = np.ones(self.n_below, dtype=F)
        self._base = np.zeros(self.n_below, dtype=F)

    # -- substrate ----------------------------------------------------------
    def attach_device(self, array) -> None:
        """Switches to inferring with the weights the device has, not the requested ones.

        `W` remains the float weight the local rule updates - the
        "shadow weight" of quantization-aware training. `W_eff` is what is
        actually on the crossbar. Without backprop, this separation is
        trivial: there is no gradient to pass through the quantizer.
        """
        self.device = array
        self.refresh_device()

    def detach_device(self) -> None:
        self.device = None
        self.W_eff = self.W
        self._estimate_sigma_max()

    def refresh_device(self) -> None:
        """Reprograms the crossbar from the current float weights."""
        if self.device is None:
            self.W_eff = self.W
        else:
            self.W_eff = self.device.program(self.W)
        self._estimate_sigma_max()

    # -- stability ----------------------------------------------------------
    def _estimate_sigma_max(self, max_iters: int = 32, tol: float = 1e-5) -> None:
        """Power iteration over W_eff, with warm start.

        Iterates until convergence, not a fixed number of times. With warm
        start it is typically 1-2 iterations (W changes slowly between
        updates), but switching crossbars needs more - and an underestimated
        σ_max gives a settling step too large, which diverges.

        Iterating to convergence also makes σ_max a function of W_eff only,
        and not of the call history: two models with the same weights behave
        the same way, whether they come from training or from
        `load_state_dict`. It is two matrix-vector products per iteration -
        on the crossbar, two reads, one in each direction.
        """
        previous = 0.0
        for _ in range(max_iters):
            u = self.W_eff.dot(self._pi_v)
            nu = float(np.linalg.norm(u))
            if nu <= 1e-12:
                self.sigma_max = 0.0
                return
            u /= nu
            v = self.W_eff.T.dot(u)
            nv = float(np.linalg.norm(v))
            if nv <= 1e-12:
                self.sigma_max = 0.0
                return
            self._pi_v = (v / nv).astype(F)
            self.sigma_max = nv
            if abs(nv - previous) <= tol * max(nv, 1e-12):
                return
            previous = nv

    def max_stable_z_lr(self) -> float:
        """z_lr < 2/(1 + σ_max²): the step above which settling diverges.

        The Hessian of the energy with respect to the state above is
        I + WᵀW, so the largest eigenvalue is 1 + σ_max(W)². This is the
        number device variability ruins - not by adding noise to the
        response, but by inflating σ_max until the fixed step stops being
        stable.
        """
        return 2.0 / (1.0 + self.sigma_max**2)

    # -- descending path: the prediction -----------------------------------
    def predict(self, z_above: np.ndarray) -> np.ndarray:
        """ẑ = f(W · z_above). Writes into internal buffers and returns ẑ."""
        np.dot(self.W_eff, z_above, out=self.a)
        if self.device is not None:
            self.a[:] = self.device.read(self.a)
        self._base = self._f(self.a).astype(F, copy=False)
        if self.Gg is not None:
            self._gate = 1.0 / (1.0 + np.exp(-(self.Gg.dot(z_above) + self.bg)))
            self._gate = self._gate.astype(F, copy=False)
            self.zhat = (self._gate * self._base).astype(F, copy=False)
        else:
            self.zhat = self._base
        return self.zhat

    # -- ascending path: the error -----------------------------------------
    def modulated_error(self, eps_below: np.ndarray) -> np.ndarray:
        """ε ⊙ [g] ⊙ f'(a): the error as the synapse sees it.

        With a gate, the credit for W passes through it - a closed gate
        means "this prediction did not apply here", and W takes no blame.
        """
        mod = eps_below * self._fprime(self.a)
        if self.Gg is not None:
            mod = mod * self._gate
        return mod.astype(F, copy=False)

    def gate_error(self, eps_below: np.ndarray) -> np.ndarray:
        """ε ⊙ f(Wz) ⊙ g(1-g): the gate's own credit."""
        assert self.Gg is not None
        return (eps_below * self._base * self._gate * (1.0 - self._gate)).astype(
            F, copy=False
        )

    def backward(self, eps_mod: np.ndarray,
                 eps_raw: np.ndarray | None = None) -> np.ndarray:
        """Wᵀ·(ε⊙[g]⊙f') [+ Gᵀ·(ε⊙f⊙g(1-g))]: the correction that rises.

        With a gate there are two paths to the state above: through the
        prediction and through the choice to apply it. It is the same
        crossbar read backwards, so it suffers the same programming
        deviations and another dose of read noise.
        """
        out = self.W_eff.T.dot(eps_mod).astype(F, copy=False)
        if self.Gg is not None and eps_raw is not None:
            out = out + self.Gg.T.dot(self.gate_error(eps_raw))
        return self.device.read(out) if self.device is not None else out

    # -- plasticity ---------------------------------------------------------
    def learn(
        self,
        eps_mod: np.ndarray,
        z_above: np.ndarray,
        lr: float,
        weight_decay: float = 0.0,
        grad_clip: float = 0.0,
        meta: float = 0.0,
        meta_decay: float = 0.999,
    ) -> None:
        """ΔW = lr · (ε ⊙ f') ⊗ z_above. Purely local, in-place.

        Always updates the float weight; if there is a device, it is
        reprogrammed afterwards (quantization-aware training).
        """
        dW = np.outer(eps_mod, z_above).astype(F, copy=False)
        if grad_clip > 0.0:
            np.clip(dW, -grad_clip, grad_clip, out=dW)
        if weight_decay > 0.0:
            self.W *= F(1.0 - lr * weight_decay)

        if meta > 0.0:
            # The synapse hardens in proportion to what it has already proven
            # to matter.
            self.importance *= F(meta_decay)
            self.importance += F(1.0 - meta_decay) * (dW * dW)
            scale = self.importance / max(float(self.importance.mean()), 1e-12)
            self.W += F(lr) * dW / (1.0 + F(meta) * scale)
        else:
            self.W += F(lr) * dW
        # Always, with or without a device: σ_max grows over training and it
        # is what fixes the stable settling step. Without this the float
        # network kept the initialization's estimate and the adaptive step
        # never got to act.
        self.refresh_device()

    def learn_gate(self, eps_raw: np.ndarray, z_above: np.ndarray,
                   lr: float, grad_clip: float = 0.0) -> None:
        """ΔG ∝ (ε ⊙ f(Wz) ⊙ g(1-g)) ⊗ z. Three factors, local."""
        if self.Gg is None:
            return
        mod = self.gate_error(eps_raw)
        dG = np.outer(mod, z_above).astype(F, copy=False)
        if grad_clip > 0.0:
            np.clip(dG, -grad_clip, grad_clip, out=dG)
        self.Gg += F(lr) * dG
        self.bg += F(lr) * mod

    # -- energy accounting --------------------------------------------------
    def macs_down(self) -> int:
        """MACs of the prediction: always dense (it is the crossbar doing it in analog)."""
        return self.n_below * self.n_above

    def macs_up(self, nnz_eps: int) -> int:
        """MACs of the rising error: proportional to the errors that pass the threshold."""
        return int(nnz_eps) * self.n_above

    def __repr__(self) -> str:  # pragma: no cover - convenience
        return (
            f"PCLayer({self.n_above} -> {self.n_below}, act={self.act_name!r})"
        )
