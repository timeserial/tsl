"""pcnet — rede preditiva hierárquica esparsa (Phase 0, protótipo Python).

Ver `README.md` na raiz do projeto. O caminho crítico é:
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
