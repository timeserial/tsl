"""Laço de treino.

Não há épocas no sentido habitual — a rede é online, aprende trama a trama e
o estado do topo atravessa o tempo. "Época" aqui é só voltar a passar pela
mesma sequência, o que num sensor real seria simplesmente mais sinal.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .config import PCConfig
from .metrics import RunStats, summarize
from .network import PCNetwork


def evaluate(net: PCNetwork, frames: np.ndarray, reset: bool = True) -> RunStats:
    """Corre a sequência sem aprender e sem deixar rasto no estado dinâmico."""
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
    """Treina e devolve as estatísticas por época (de treino, ou de avaliação).

    `replay_per_epoch` intercala reprodução de episódios antigos com a
    aprendizagem nova. O hipocampo não reproduz só durante o sono — reproduz
    também em repouso acordado, entrecalado com a experiência. E em
    aprendizagem contínua é sabido que reproduzir *durante* o treino funciona
    e reproduzir só no fim não: 64 passos de consolidação contra milhares de
    passos de tarefa nova não movem nada.
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
    """Avalia com outro limiar de esparsidade, sem mexer no treino.

    O limiar é um botão de *inferência*: mudá-lo não requer treinar de novo,
    é literalmente decidir quanto ruído se aceita não converter no ADC.
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
    """Atalho: rede com a config default e as alterações pedidas."""
    return PCNetwork(PCConfig(seed=seed, **overrides))
