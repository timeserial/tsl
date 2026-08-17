"""Transparent analog crossbar: a weight matrix as a differential pair of
conductances, with the classic non-idealities from the device literature,
each one independent and with its own magnitude.

The mapping (standard since Burr et al. 2015, "Experimental demonstration ...
using ... PCM"; Yu 2018, Proc. IEEE):

    W_ij = (G+_ij − G−_ij) / k,   G± ∈ [G_min, G_max],   k = (G_max−G_min)/w_max

with G_max normalized to 1 and G_min = 1/on_off. The pair starts at the
midpoint (G± = (G_min+G_max)/2 ± w·k/2): zero weight = balanced pair, and
each update Δw splits into +Δw·k/2 on the positive branch and −Δw·k/2 on the
negative one - the usual differential write scheme (Gokmen & Vlasov 2016).

The non-idealities, each with the canonical model cited at the place where it
is applied:

  1. Write noise (cycle-to-cycle):  ΔG ← ΔG + N(0, (σ_w·ΔG_range)²)
     per pulse and per device (NeuroSim model, Chen et al. 2018).
  2. Write nonlinearity + asymmetry: exponential saturation -
     potentiation compressed near G_max, depression near G_min, with
     different factors α_p/α_d (per-step form of NeuroSim's A_p/A_d model;
     limit α→0 = linear; generalizes the soft-bounds of
     Fusi & Abbott 2007).
  3. Read noise: multiplicative per device, G_read = G·(1+ξ),
     ξ~N(0,σ_r) i.i.d. per read (aggregated RTN/1f; Joshi et al. 2020,
     Nat. Comm.). Propagated analytically to the output - exact in
     distribution, without generating a noise matrix per MAC.
  4. ADC: uniform quantization with saturation at ±full-scale, n_bits,
     applied to reads in both directions.
  5. Conductance drift (PCM): G(t) = G(t₀)·(t/t₀)^(−ν), ν per
     device ~ N(ν̄, σ_ν) (Ielmini et al. 2007; Le Gallo & Sebastian
     2020). It is the spread of ν that prevents the differential pair from
     canceling it.
  6. IR drop (1st order): effective attenuation that grows with the position
     on the line and with the total current of the array,
         a_ij = 1 − γ·p_ij·u,  p_ij = ((i+1)/N + (j+1)/M)/2,
         u = mean(G+ + G−)/(2·G_max)
     with γ = maximum drop at the farthest corner at full load. A
     deliberately simple model: the real drop solves the resistive network
     (see ISAAC, Shafiee et al. 2016); here only the 1st-order term stays -
     position × utilization - folded over the two reads.
  7. Endurance write gate (the project's θ): |Δw| < θ_w generates no
     physical pulse at all. Every device pulse issued and saved is counted -
     the currency of the endurance/accuracy trade-off.

Numerical fidelity: with everything off the object degenerates into a
float32 accumulator bit-identical to float training - it is the sanity gate
the experiment demands (reproduce 0.579±0.004 before turning anything on).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .dtypes import F


@dataclass(frozen=True)
class CrossbarModel:
    """Physics shared by all arrays. Everything at 0 / on_off=inf = ideal."""

    # on/off ratio r = G_max/G_min. G_min = 1/r (G_max ≡ 1). inf → G_min = 0.
    # Literature: filamentary ReRAM ~10-100; PCM ~10²-10³ (Yu 2018).
    on_off: float = math.inf
    # cycle-to-cycle write noise, fraction of the range (G_max−G_min) per pulse.
    # Typical measurements: 0.5-5% of the range; >10% is a bad device (Chen 2018).
    sigma_write: float = 0.0
    # write nonlinearity: ΔG_pot = ΔG·exp(−α_p·x̂),
    # ΔG_dep = ΔG·exp(−α_d·(1−x̂)), x̂ = (G−G_min)/(G_max−G_min) ∈ [0,1].
    # α = ln(ratio between the largest and smallest step across the range):
    # α≈0.5 almost linear, α≈1-2 typical of TaOx/HfOx, α≥5 severe (PCMO).
    alpha_p: float = 0.0
    alpha_d: float = 0.0
    # multiplicative read noise per device, per read.
    # ~1% (calibrated PCM arrays, Joshi 2020) to ~5-10% (strong RTN).
    sigma_read: float = 0.0
    # ADC at the outputs (0 = no quantization). 8 bits is the usual design
    # point (ISAAC); 6 is aggressive; 12 is nearly transparent.
    adc_bits: int = 0
    # PCM drift: mean exponent ν̄ (amorphous ~0.05-0.1, crystalline ~0.005,
    # cell engineered for low drift ~0.02) and relative spread across
    # devices (Le Gallo 2018 measures ~30%).
    drift_nu: float = 0.0
    drift_nu_spread: float = 1.0 / 3.0
    # t/t₀ between programming and reading: 1e4 (t₀=1 s → read ~3 h later).
    drift_t_ratio: float = 1e4
    # IR drop: γ = maximum attenuation (farthest corner, loaded array).
    # With R_wire ~1-5 Ω/cell and R_device ~10-100 kΩ in arrays of 64+:
    # a few % (mild) to tens of % (severe) - Shafiee 2016, Ielmini 2018.
    ir_drop: float = 0.0
    # endurance gate: |Δw| < theta_write ⇒ no physical pulse.
    theta_write: float = 0.0

    @property
    def analog_ideal(self) -> bool:
        """No analog physics at all (the θ_w gate is allowed: it operates
        on the write request, before any device exists)."""
        return (
            math.isinf(self.on_off)
            and not self.sigma_write
            and not self.alpha_p
            and not self.alpha_d
            and not self.sigma_read
            and not self.adc_bits
            and not self.drift_nu
            and not self.ir_drop
        )

    def to_dict(self) -> dict:
        return asdict(self)


class Crossbar:
    """A weight matrix living on a concrete analog array.

    Operations - exactly the three of the TSL and nothing more:
      read(x)    - direct read      y = W·x      (prediction)
      read_T(e)  - transposed read  h = Wᵀ·e     (backprojection)
      update(dW) - rank-1 write (the caller passes the outer product)
    and the two lifecycle ones: initial program(W) and apply_drift() between
    training and evaluation.

    `w_max` is the array's weight full scale (it fixes k); requested weights
    beyond it saturate in the conductances - as in silicon. `adc_range` is
    the full scale of this array's ADC (the periphery is sized per array).
    """

    G_MAX = 1.0

    def __init__(self, shape: tuple[int, int], w_max: float,
                 model: CrossbarModel, seed: int,
                 adc_range: float = 4.0, adc_range_T: float | None = None,
                 use_adc: bool = True, use_ir: bool = True) -> None:
        self.shape = tuple(shape)
        self.model = model
        self.w_max = float(w_max)
        # Distinct ADCs in the two directions: the direct read comes out on
        # the columns, the transposed one on the rows - separate peripheries,
        # full scales sized to each direction's signal (design calibration).
        self.adc_range = float(adc_range)
        self.adc_range_T = float(adc_range_T if adc_range_T is not None
                                 else adc_range)
        # bias/single columns: no ADC or IR (offset added in the digital
        # periphery; a single column has no long line for voltage to drop).
        self.use_adc = bool(use_adc) and model.adc_bits > 0
        self.use_ir = bool(use_ir) and model.ir_drop > 0.0
        self.rng = np.random.default_rng(seed)

        self.g_min = 0.0 if math.isinf(model.on_off) else self.G_MAX / model.on_off
        self.g_range = self.G_MAX - self.g_min
        self.k = self.g_range / self.w_max  # conductance per unit of weight
        self._g0 = 0.5 * (self.G_MAX + self.g_min)  # midpoint (weight 0)

        # p_ij ∈ (0,1]: electrical distance normalized to the drivers (corner
        # (0,0) near, corner (N,M) far) - the position term of the IR drop.
        n, m = self.shape
        self._pos = 0.5 * ((np.arange(1, n + 1)[:, None] / n)
                           + (np.arange(1, m + 1)[None, :] / m))

        # exact mode: float32 accumulator bit-identical to float training.
        self._exact = model.analog_ideal
        self._Gp: np.ndarray | None = None  # conductances (float64)
        self._Gm: np.ndarray | None = None
        self._Wf = np.zeros(self.shape, dtype=F)  # effective weights (read)
        self._S2: np.ndarray | None = None  # (a²·(G+²+G−²))/k², for noise

        # endurance accounting and diagnostics
        self.pulses_issued = 0   # device pulses issued
        self.pulses_gated = 0    # pulses saved by the θ_w gate
        self.updates = 0         # calls to update()
        self.clip_events = 0     # elements that hit the full scale

    # -- initial programming ------------------------------------------------
    def program(self, W: np.ndarray) -> None:
        """Programs the array from W (program-and-verify: the nonlinearity
        does not enter - one iterates until it lands - but the residual
        write noise of the last pulse does)."""
        W = np.asarray(W, dtype=F)
        if self._exact:
            self._Wf = W.copy()  # float32 accumulator: bit-exact
            return
        Wc = np.clip(W.astype(np.float64), -self.w_max, self.w_max)
        self.clip_events += int(np.sum(np.abs(W) > self.w_max))
        half = 0.5 * self.k * Wc
        self._Gp = self._g0 + half
        self._Gm = self._g0 - half
        if self.model.sigma_write > 0.0:
            s = self.model.sigma_write * self.g_range
            self._Gp += self.rng.standard_normal(self.shape) * s
            self._Gm += self.rng.standard_normal(self.shape) * s
        self._clip_rails()
        self._refresh()

    # -- reads --------------------------------------------------------------
    def read(self, x: np.ndarray) -> np.ndarray:
        """y = W·x, with the read physics on top."""
        y = self._Wf.dot(x)
        if self.model.sigma_read > 0.0 and self._S2 is not None:
            # Var(y_i) = σ_r²·Σ_j (G+²+G−²)_ij x_j² / k²  - the exact
            # propagation of the multiplicative i.i.d. per-device noise. This
            # is where the on/off ratio bites: high G_min = large common mode
            # that the subtraction cancels in the signal but not in the noise.
            y = y + self.rng.standard_normal(y.shape) * (
                self.model.sigma_read * np.sqrt(self._S2.dot(x * x)))
        return self._adc(y)

    def read_T(self, e: np.ndarray) -> np.ndarray:
        """h = Wᵀ·e: the same devices, read in the opposite direction -
        hence the same static deviations and another dose of read noise."""
        y = self._Wf.T.dot(e)
        if self.model.sigma_read > 0.0 and self._S2 is not None:
            y = y + self.rng.standard_normal(y.shape) * (
                self.model.sigma_read * np.sqrt(self._S2.T.dot(e * e)))
        return self._adc(y, self.adc_range_T)

    def _adc(self, y: np.ndarray, fs: float | None = None) -> np.ndarray:
        if not self.use_adc:
            return y.astype(F, copy=False)
        # uniform quantization with saturation (mid-tread), signed n_bits.
        fs = self.adc_range if fs is None else fs
        step = 2.0 * fs / ((1 << self.model.adc_bits) - 1)
        return (np.round(np.clip(y, -fs, fs) / step) * step).astype(F)

    # -- write --------------------------------------------------------------
    def update(self, dW: np.ndarray) -> None:
        """Rank-1 write (or any ΔW): differential pulses ±ΔG/2.

        The θ_w gate decides BEFORE the physics: elements below the threshold
        generate no pulse (endurance saved), the rest go through
        nonlinearity → noise → rails, in that order.
        """
        self.updates += 1
        m = self.model
        if m.theta_write > 0.0:
            mask = np.abs(dW) >= m.theta_write
            n_write = int(mask.sum())
            self.pulses_gated += 2 * (dW.size - n_write)
        else:
            mask = None
            n_write = dW.size
        self.pulses_issued += 2 * n_write  # two devices per pair
        if n_write == 0:
            return

        if self._exact:
            # float32 accumulator: same arithmetic as float training (numpy's
            # += over float32), so the sanity gate is bit-exact.
            self._Wf += (dW if mask is None else np.where(mask, dW, F(0.0))).astype(F, copy=False)
            return

        dG = 0.5 * self.k * np.asarray(dW, dtype=np.float64)
        if mask is not None:
            dG = np.where(mask, dG, 0.0)
        pulsed = dG != 0.0

        # positive branch receives +dG, negative −dG; at each device the
        # direction of the pulse decides whether it is potentiation (α_p) or
        # depression (α_d).
        for G, sgn in ((self._Gp, 1.0), (self._Gm, -1.0)):
            step = sgn * dG
            if m.alpha_p > 0.0 or m.alpha_d > 0.0:
                x_hat = np.clip((G - self.g_min) / self.g_range, 0.0, 1.0)
                gain = np.where(
                    step > 0.0,
                    np.exp(-m.alpha_p * x_hat),          # saturation toward G_max
                    np.exp(-m.alpha_d * (1.0 - x_hat)),  # saturation toward G_min
                )
                step = step * gain
            if m.sigma_write > 0.0:
                step = step + pulsed * (
                    self.rng.standard_normal(self.shape)
                    * (m.sigma_write * self.g_range))
            G += step
        self._clip_rails()
        self._refresh()

    # -- drift (between training and evaluation) ----------------------------
    def apply_drift(self) -> None:
        """G(t) = G(t₀)·(t/t₀)^(−ν), ν_ij ~ N(ν̄, σ_ν) per device."""
        m = self.model
        if m.drift_nu <= 0.0 or self._exact:
            return
        s = m.drift_nu * m.drift_nu_spread
        for G in (self._Gp, self._Gm):
            nu = np.clip(self.rng.normal(m.drift_nu, s, self.shape), 0.0, None)
            G *= m.drift_t_ratio ** (-nu)
        # drift lowers the absolute conductance; it can go below G_min
        # (the cell "cools" beyond the programmable state) - only the
        # physical floor of zero is enforced.
        np.clip(self._Gp, 0.0, self.G_MAX, out=self._Gp)
        np.clip(self._Gm, 0.0, self.G_MAX, out=self._Gm)
        self._refresh()

    # -- internals ----------------------------------------------------------
    def _clip_rails(self) -> None:
        np.clip(self._Gp, self.g_min, self.G_MAX, out=self._Gp)
        np.clip(self._Gm, self.g_min, self.G_MAX, out=self._Gm)

    def _refresh(self) -> None:
        """Recomputes the effective operator after any write/drift."""
        if self.use_ir:
            # a_ij = 1 − γ·p_ij·u: position × mean current utilization.
            u = float((self._Gp.mean() + self._Gm.mean()) / (2.0 * self.G_MAX))
            att = 1.0 - self.model.ir_drop * self._pos * u
        else:
            att = 1.0
        self._Wf = ((att * (self._Gp - self._Gm)) / self.k).astype(F)
        if self.model.sigma_read > 0.0:
            self._S2 = (att * att) * (self._Gp**2 + self._Gm**2) / (self.k * self.k)

    # -- inspection ---------------------------------------------------------
    @property
    def W_eff(self) -> np.ndarray:
        """The weights the array actually applies (float32, no noise)."""
        return self._Wf

    def rail_frac(self) -> float:
        """Fraction of devices pinned against the rails (1% of the range)."""
        if self._exact:
            return 0.0
        tol = 0.01 * self.g_range
        at = ((self._Gp <= self.g_min + tol) | (self._Gp >= self.G_MAX - tol)
              | (self._Gm <= self.g_min + tol) | (self._Gm >= self.G_MAX - tol))
        return float(at.mean())

    def stats(self) -> dict:
        return {
            "pulses_issued": self.pulses_issued,
            "pulses_gated": self.pulses_gated,
            "updates": self.updates,
            "rail_frac": round(self.rail_frac(), 4),
            "clip_events": self.clip_events,
        }
