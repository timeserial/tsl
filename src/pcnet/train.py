"""Training loop.

There are no epochs in the usual sense - the network is online, learns frame
by frame and the top's state crosses time. An "epoch" here is just passing
over the same sequence again, which on a real sensor would simply be more
signal.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .config import PCConfig
from .metrics import RunStats, summarize
from .network import PCNetwork


def evaluate(net: PCNetwork, frames: np.ndarray, reset: bool = True) -> RunStats:
    """Runs the sequence without learning and without leaving a trace in the dynamic state."""
    snapshot = net.snapshot_state()
    try:
        traces = net.run(frames, learn=False, reset=reset)
        return summarize(traces, net.cfg.max_iters, net.eps_per_pass)
    finally:
        net.restore_state(snapshot)


def train(
    net: PCNetwork,
    frames: np.ndarray,
    epochs: int = 60,
    eval_frames: np.ndarray | None = None,
    log_every: int = 0,
    on_epoch=None,
    replay_per_epoch: int = 0,
) -> list[RunStats]:
    """Trains and returns the per-epoch statistics (training ones, or evaluation ones).

    `replay_per_epoch` interleaves replay of old episodes with the new
    learning. The hippocampus does not replay only during sleep - it also
    replays during quiet wakefulness, interleaved with experience. And in
    continual learning it is well known that replaying *during* training
    works and replaying only at the end does not: 64 consolidation steps
    against thousands of new-task steps move nothing.
    """
    history: list[RunStats] = []
    for ep in range(epochs):
        traces = net.run(frames, learn=True)
        if replay_per_epoch and net.memory is not None:
            net.consolidate(n_episodes=replay_per_epoch)
        stats = summarize(traces, net.cfg.max_iters, net.eps_per_pass)
        if eval_frames is not None:
            stats = evaluate(net, eval_frames)
        history.append(stats)
        if on_epoch is not None:
            on_epoch(ep, stats)
        elif log_every and (ep % log_every == 0 or ep == epochs - 1):
            print(f"  época {ep + 1:3d}/{epochs}  {stats.as_row()}")
    return history


def evaluate_with_theta(net: PCNetwork, frames: np.ndarray, theta: float) -> RunStats:
    """Evaluates with a different sparsity threshold, without touching training.

    The threshold is an *inference* knob: changing it does not require
    retraining, it is literally deciding how much noise you accept not
    converting in the ADC.
    """
    original = net.cfg
    try:
        net.cfg = replace(original, theta=theta)
        return evaluate(net, frames)
    finally:
        net.cfg = original


def theta_sweep(
    net: PCNetwork, frames: np.ndarray, thetas: tuple[float, ...]
) -> dict[float, RunStats]:
    return {th: evaluate_with_theta(net, frames, th) for th in thetas}


def make_net(seed: int = 0, **overrides) -> PCNetwork:
    """Shortcut: network with the default config plus the requested overrides."""
    return PCNetwork(PCConfig(seed=seed, **overrides))
