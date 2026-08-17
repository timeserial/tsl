"""pcnet - sparse hierarchical predictive network (Phase 0, Python prototype).

See `README.md` at the project root. The critical path is:
`config.PCConfig` -> `network.PCNetwork` -> `metrics.StepTrace`.
"""

from __future__ import annotations

from .config import PCConfig
from .layer import PCLayer
from .metrics import RunStats, StepTrace, summarize
from .network import PCNetwork
from .signals import ToySignal, make_signal, persistence_nrmse
from .train import evaluate, evaluate_with_theta, make_net, theta_sweep, train

__all__ = [
    "PCConfig",
    "PCLayer",
    "PCNetwork",
    "StepTrace",
    "RunStats",
    "summarize",
    "ToySignal",
    "make_signal",
    "persistence_nrmse",
    "train",
    "evaluate",
    "evaluate_with_theta",
    "theta_sweep",
    "make_net",
]
