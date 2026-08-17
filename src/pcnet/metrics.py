"""Instrumentation.

What matters to measure is not the loss - it is the cost. Three numbers per
frame: surprise per settling iteration, fraction of errors silenced by the
threshold, and number of iterations until the early exit fires. From these
comes the accounting the energy argument needs: ADC conversions (the
crossbar's expensive bottleneck) proportional to the surprise and not to the
size of the network.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Why the settling stopped. The distinction matters: "explained" is the
# early exit we want (correct prediction, silence, zero cost); "stalled"
# is the network giving up on something it cannot explain - it also saves
# energy, but it is an admission of ignorance, not a victory; "ceiling" is
# the expensive case.
EXIT_EXPLAINED = "explicado"
EXIT_STALLED = "estagnado"
EXIT_CEILING = "teto"


@dataclass
class StepTrace:
    """Everything a frame spent and revealed."""

    # Surprise = energy ½Σ‖ε_l‖² over all levels, *before* the threshold,
    # one entry per error evaluation. Element 0 is the open-loop surprise:
    # how much the prediction descended from the previous step missed.
    surprise: list[float] = field(default_factory=list)
    # The same thing after the threshold: what actually went up.
    transmitted: list[float] = field(default_factory=list)

    iters: int = 0  # state updates performed
    exit_reason: str = EXIT_CEILING

    eps_total: int = 0  # error components evaluated
    eps_silenced: int = 0  # ... of which stayed below the threshold

    macs_up: int = 0  # MACs on the ascending path (sparse)
    macs_up_dense: int = 0  # ... if nothing were silenced
    macs_down: int = 0  # MACs on the descending path (dense; it is the crossbar)

    # ε_0 in open loop: the frame's prediction error, the "ML" number.
    pred_error: np.ndarray | None = None
    target_rms: float = 0.0

    @property
    def early_exit(self) -> bool:
        """Exited before spending the iteration ceiling (for any reason)."""
        return self.exit_reason != EXIT_CEILING

    @property
    def open_loop_surprise(self) -> float:
        return self.surprise[0] if self.surprise else 0.0

    @property
    def final_surprise(self) -> float:
        return self.surprise[-1] if self.surprise else 0.0

    @property
    def silenced_frac(self) -> float:
        return self.eps_silenced / self.eps_total if self.eps_total else 0.0

    @property
    def adc_conversions(self) -> int:
        """Errors that passed the threshold = ADC conversions needed."""
        return self.eps_total - self.eps_silenced

    @property
    def pred_rmse(self) -> float:
        if self.pred_error is None:
            return 0.0
        return float(np.sqrt(np.mean(np.square(self.pred_error))))

    @property
    def pred_nrmse(self) -> float:
        """RMSE normalized by the target RMS: 1.0 = as good as predicting zero."""
        return self.pred_rmse / self.target_rms if self.target_rms > 0 else 0.0


@dataclass
class RunStats:
    """Aggregate over a sequence of frames."""

    n_steps: int = 0
    mean_iters: float = 0.0
    mean_open_loop_surprise: float = 0.0
    mean_final_surprise: float = 0.0
    silenced_frac: float = 0.0
    early_exit_frac: float = 0.0
    explained_frac: float = 0.0
    stalled_frac: float = 0.0
    mac_up_frac: float = 0.0  # effective ascending MACs / dense
    adc_frac: float = 0.0  # ADC conversions / (dense network, max_iters)
    pred_nrmse: float = 0.0

    def as_row(self) -> dict:
        return {
            "n_steps": self.n_steps,
            "iters": round(self.mean_iters, 3),
            "surprise_open_loop": round(self.mean_open_loop_surprise, 4),
            "surprise_final": round(self.mean_final_surprise, 4),
            "silenced_%": round(100 * self.silenced_frac, 1),
            "early_exit_%": round(100 * self.early_exit_frac, 1),
            "explained_%": round(100 * self.explained_frac, 1),
            "stalled_%": round(100 * self.stalled_frac, 1),
            "macs_up_%": round(100 * self.mac_up_frac, 1),
            "adc_%": round(100 * self.adc_frac, 1),
            "pred_nrmse": round(self.pred_nrmse, 4),
        }


def summarize(traces: list[StepTrace], max_iters: int, eps_per_pass: int) -> RunStats:
    """Reduces a list of frames to one table row.

    `eps_per_pass` is the number of error components of one dense pass; it
    serves as the denominator for the worst-case ADC fraction (a dense
    network always running up to the iteration ceiling).
    """
    if not traces:
        return RunStats()

    n = len(traces)
    dense_adc = max(1, (max_iters + 1) * eps_per_pass * n)
    macs_up_dense = sum(t.macs_up_dense for t in traces)
    eps_total = sum(t.eps_total for t in traces)

    return RunStats(
        n_steps=n,
        mean_iters=sum(t.iters for t in traces) / n,
        mean_open_loop_surprise=sum(t.open_loop_surprise for t in traces) / n,
        mean_final_surprise=sum(t.final_surprise for t in traces) / n,
        silenced_frac=(sum(t.eps_silenced for t in traces) / eps_total)
        if eps_total
        else 0.0,
        early_exit_frac=sum(1 for t in traces if t.early_exit) / n,
        explained_frac=sum(1 for t in traces if t.exit_reason == EXIT_EXPLAINED) / n,
        stalled_frac=sum(1 for t in traces if t.exit_reason == EXIT_STALLED) / n,
        mac_up_frac=(sum(t.macs_up for t in traces) / macs_up_dense)
        if macs_up_dense
        else 0.0,
        adc_frac=sum(t.adc_conversions for t in traces) / dense_adc,
        pred_nrmse=float(np.mean([t.pred_nrmse for t in traces])),
    )
