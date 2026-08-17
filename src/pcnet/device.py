"""The substrate: ternary weights and device noise.

Step 2 of the plan. There is a single question: how much does the network
lose when the weights stop being float32 and become imperfect physical
devices?

Three imperfections, and the distinction between them is what matters:

  * **Quantization** - the weight becomes {-1,0,1} with a per-row gain. It is
    deterministic and known.
  * **Programming variability** - each device ends up with a conductance
    slightly different from the requested one, and *stays that way*. It is
    sampled once, at programming time, and cannot be averaged away: it is the
    error that truly tests the architecture's tolerance.
  * **Read noise** - changes on every MAC. This one the settling loop does
    have a chance to dilute across iterations.

Confusing the last two is the easiest way to overstate the noise tolerance
of an analog system, which is why they are kept separate in the model.

ADC quantization enters elsewhere: in `PCNetwork._threshold`, which is
where the error crosses the analog→digital boundary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .dtypes import F


@dataclass(frozen=True)
class DeviceModel:
    """Substrate parameters. Everything at 0/False = ideal float32."""

    # --- quantization -----------------------------------------------------
    ternary: bool = False
    # TWN threshold: Δ = ternary_threshold × mean(|W|). 0.75 is the value of
    # Li & Liu (2016); above that too much is zeroed, below it the sparsity
    # benefit is lost.
    ternary_threshold: float = 0.75
    # Per-row gain (one sense amplifier per output neuron) instead of a
    # single gain per matrix. Physically realizable and considerably better,
    # because the rows have very different scales.
    per_row_scale: bool = True

    # --- programming variability (static) ---------------------------------
    sigma_rel: float = 0.0  # deviation proportional to the programmed weight
    sigma_abs: float = 0.0  # absolute deviation (leakage, offset)
    stuck_frac: float = 0.0  # fraction of devices stuck at zero

    # --- read noise (dynamic, per MAC) ------------------------------------
    read_sigma: float = 0.0

    # --- ADC ---------------------------------------------------------------
    adc_bits: int = 0  # 0 = no quantization
    adc_range: float = 2.0  # full scale, ±adc_range

    # The temporal transition A is 64 weights against the generators' 2688,
    # and in the design it is the fast path - a small SSM that makes sense to
    # keep digital. By default it stays out of the crossbar; setting it to
    # True tests that assumption.
    include_transition: bool = False

    seed: int = 0

    @property
    def is_ideal(self) -> bool:
        return not (
            self.ternary
            or self.sigma_rel
            or self.sigma_abs
            or self.stuck_frac
            or self.read_sigma
            or self.adc_bits
        )

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# quantization
# ---------------------------------------------------------------------------
def ternarize(
    W: np.ndarray, threshold: float = 0.75, per_row: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """W ≈ scale · T, with T in {-1, 0, 1}.

    Returns (T, scale). The scale is per row if `per_row`, otherwise a
    replicated scalar - in either case with shape (n_rows, 1) so it can
    multiply directly.
    """
    W = np.asarray(W, dtype=F)
    absW = np.abs(W)
    axis_mean = absW.mean(axis=1, keepdims=True) if per_row else absW.mean()
    delta = (threshold * axis_mean).astype(F)

    T = np.where(absW >= delta, np.sign(W), 0.0).astype(F)
    kept = T != 0

    if per_row:
        n_kept = kept.sum(axis=1, keepdims=True)
        total = (absW * kept).sum(axis=1, keepdims=True)
        scale = np.divide(total, n_kept, out=np.zeros_like(total), where=n_kept > 0)
    else:
        n_kept = kept.sum()
        s = float((absW * kept).sum() / n_kept) if n_kept else 0.0
        scale = np.full((W.shape[0], 1), s, dtype=F)

    return T, scale.astype(F)


# ---------------------------------------------------------------------------
# a weight matrix as it exists on the substrate
# ---------------------------------------------------------------------------
class AnalogArray:
    """The fixed deviations of a concrete crossbar, sampled a single time.

    `program(W)` takes the weights training wants and returns the ones the
    device actually has. The deviations are not re-sampled on each call -
    that is what makes them programming variability and not noise.
    """

    __slots__ = ("model", "shape", "_rel", "_abs", "_alive", "_rng")

    def __init__(self, shape: tuple[int, int], model: DeviceModel, seed: int) -> None:
        self.model = model
        self.shape = shape
        rng = np.random.default_rng(seed)

        self._rel = (
            (rng.standard_normal(shape) * model.sigma_rel).astype(F)
            if model.sigma_rel
            else None
        )
        self._abs = (
            (rng.standard_normal(shape) * model.sigma_abs).astype(F)
            if model.sigma_abs
            else None
        )
        self._alive = (
            (rng.random(shape) >= model.stuck_frac) if model.stuck_frac else None
        )
        # Separate RNG for read noise: the dynamic one cannot share the
        # sequence with the static one, otherwise reprogramming changes the noise.
        self._rng = np.random.default_rng(seed + 1_000_003)

    def program(self, W: np.ndarray) -> np.ndarray:
        """Requested weights -> weights the device has."""
        m = self.model
        if m.ternary:
            T, scale = ternarize(W, m.ternary_threshold, m.per_row_scale)
            W_eff = (T * scale).astype(F)
        else:
            W_eff = np.asarray(W, dtype=F).copy()

        if self._rel is not None:
            W_eff = (W_eff * (1.0 + self._rel)).astype(F)
        if self._abs is not None:
            W_eff = (W_eff + self._abs).astype(F)
        if self._alive is not None:
            W_eff = np.where(self._alive, W_eff, F(0.0)).astype(F)
        return W_eff

    def read(self, out: np.ndarray) -> np.ndarray:
        """Read noise, re-sampled on every MAC."""
        s = self.model.read_sigma
        if not s:
            return out
        return (out + self._rng.standard_normal(out.shape) * s).astype(F)


def quantize_adc(x: np.ndarray, bits: int, full_scale: float) -> np.ndarray:
    """Uniform quantization with saturation at ±full_scale.

    `bits` includes the sign: 8 bits -> 256 levels over 2·full_scale.
    """
    if bits <= 0:
        return x
    levels = (1 << bits) - 1
    step = (2.0 * full_scale) / levels
    clipped = np.clip(x, -full_scale, full_scale)
    return (np.round(clipped / step) * step).astype(F)
