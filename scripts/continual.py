#!/usr/bin/env python3
"""Learn something new without destroying what came before.

    .venv/bin/python scripts/continual.py --tasks 5 --epochs 25

Catastrophic forgetting is one of the oldest and most real problems in the
field: teach a network one task, then another, and the first one disappears.
It is not lack of capacity - it is the plasticity/stability dilemma. A network
that learns fast forgets fast, and one that does not forget does not learn.

Transformers have the problem in full, and it is **structural**: they are
trained once and frozen, and everything that looks like learning after that
(context, RAG, fine-tuning) is scaffolding bolted on outside. It is not solved
by scale.

The brain's answer, and the one section 4 of `CONTEXTO.md` proposes, is not to
choose: two memories at different speeds. The hippocampus records the episode
on the spot without touching the weights; sleep replays what repeated and
distills it slowly into the cortex.

The protocol here is the standard one from the continual-learning literature:

  * N tasks, each a signal with different structure;
  * training is **sequential**, one task at a time, never going back;
  * at the end, measure on **all** of them.

Two measures, and the second is the one that matters:

  * **mean final NRMSE** - how well it ended up on everything it learned.
  * **Forgetting** - how much a task got worse between just having learned it
    and reaching the end. This is the number that separates those who
    accumulate from those who replace.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pcnet import PCConfig, PCNetwork, make_signal  # noqa: E402
from pcnet.dtypes import F  # noqa: E402
from pcnet.episodic import EpisodicConfig  # noqa: E402
from pcnet.report import pm, rule, table  # noqa: E402
from pcnet.signals import DEFAULT_FREQS  # noqa: E402
from pcnet.train import train  # noqa: E402


def make_tasks(n_tasks: int, frame_len: int, n_frames: int, seed: int):
    """N signals with distinct spectral content - N different "worlds"."""
    tasks = []
    for k in range(n_tasks):
        scale = 0.6 + 0.35 * k  # each task lives in a different band
        freqs = tuple(min(0.45, f * scale) for f in DEFAULT_FREQS)
        sig = make_signal(n_frames=n_frames, frame_len=frame_len,
                          freqs=freqs, seed=seed + 100 * k)
        cut = int(0.8 * sig.n_frames)
        tasks.append((sig.frames[:cut], sig.frames[cut:]))
    return tasks


def nrmse(pred, target) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2) / np.mean(target**2)))


def evaluate_task(net: PCNetwork, test: np.ndarray, warmup: np.ndarray) -> float:
    """Evaluates without learning and without leaving a trace in the state."""
    snapshot = net.snapshot_state()
    try:
        net.reset()
        for frame in warmup[-16:]:
            net.step(frame, learn=False)
        preds = []
        for frame in test:
            preds.append(net.predict_next())
            net.step(frame, learn=False)
        return nrmse(np.array(preds, dtype=F), test)
    finally:
        net.restore_state(snapshot)


def run_pcnet(tasks, epochs, seed, memory: bool, sleep: int,
              interleave: int = 0, **cfg_kw):
    net = PCNetwork(PCConfig.recommended(seed=seed, **cfg_kw))
    if memory:
        net.attach_memory(EpisodicConfig(n_slots=256))

    matrix = np.zeros((len(tasks), len(tasks)))
    for k, (train_frames, _) in enumerate(tasks):
        train(net, train_frames, epochs=epochs, replay_per_epoch=interleave)
        if sleep and net.memory is not None:
            # Sleep: replays episodes of everything seen so far, not just what
            # was just learned. This is what should prevent forgetting.
            net.consolidate(n_episodes=32, passes=sleep)
        for j, (tr_j, te_j) in enumerate(tasks):
            matrix[k, j] = evaluate_task(net, te_j, tr_j)
    return matrix, (net.memory.stats() if net.memory else {})


def run_torch(tasks, epochs, seed, kind: str, frame_len: int):
    """A network trained with backprop, on the same sequence. The contrast."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    if kind == "mlp":
        net = nn.Sequential(nn.Linear(frame_len, 48), nn.Tanh(), nn.Linear(48, frame_len))
    else:
        class G(nn.Module):
            def __init__(self):
                super().__init__()
                self.rnn = nn.GRU(frame_len, 32, batch_first=True)
                self.head = nn.Linear(32, frame_len)

            def forward(self, x):
                out, _ = self.rnn(x.unsqueeze(0))
                return self.head(out.squeeze(0))
        net = G()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)

    def fwd(seq):
        return net(seq) if kind == "gru" else net(seq)

    def score(test, warmup):
        with torch.no_grad():
            seq = torch.tensor(np.concatenate([warmup[-16:], test]), dtype=torch.float32)
            pred = fwd(seq[:-1])[15:]
        return nrmse(pred.numpy(), test)

    matrix = np.zeros((len(tasks), len(tasks)))
    for k, (train_frames, _) in enumerate(tasks):
        x = torch.tensor(np.asarray(train_frames), dtype=torch.float32)
        for _ in range(epochs * 8):
            loss = ((fwd(x[:-1]) - x[1:]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        for j, (tr_j, te_j) in enumerate(tasks):
            matrix[k, j] = score(te_j, tr_j)
    return matrix, {}


def run_joint(tasks, epochs, seed):
    """The decisive control: train on everything at the same time.

    If the network cannot do the N tasks even then, what the table shows is
    not forgetting - it is lack of capacity, and no episodic memory in the
    world fixes that. It is the same care that avoided concluding the model
    was bad at the HAR gyroscope, when what had no signal was the task.
    """
    net = PCNetwork(PCConfig.recommended(seed=seed))
    # Interleave in short blocks, do not concatenate. Concatenating would be
    # sequential training by another name - the model would always end up on
    # the last task, which is exactly what the first version of this control
    # did, giving a "ceiling" lower than the variants it was supposed to bound.
    block = 16
    chunks = []
    n_blocks = min(len(t[0]) for t in tasks) // block
    for b in range(n_blocks):
        for tr, _ in tasks:
            chunks.append(tr[b * block:(b + 1) * block])
    mixed = np.concatenate(chunks)
    train(net, mixed, epochs=epochs)
    row = np.array([evaluate_task(net, te, tr) for tr, te in tasks])
    return np.tile(row, (len(tasks), 1)), {}


def summarize(matrix: np.ndarray) -> dict:
    n = len(matrix)
    final = matrix[-1]
    # Forgetting: how much each task worsened between just-learned and the end
    forgetting = [float(matrix[-1, j] - matrix[j, j]) for j in range(n - 1)]
    return {
        "final_medio": float(final.mean()),
        "primeira_tarefa_no_fim": float(final[0]),
        "primeira_tarefa_acabada_de_aprender": float(matrix[0, 0]),
        "esquecimento_medio": float(np.mean(forgetting)) if forgetting else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--frame-len", type=int, default=64)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--sleep", type=int, default=2, help="consolidation passes")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    tasks = make_tasks(args.tasks, args.frame_len, args.frames, seed=1)
    print(f"{args.tasks} tasks, {args.epochs} passes each, "
          f"{len(tasks[0][0])} training frames per task")
    print("Sequential training: each task is seen once and never again.\n")

    variants = [
        ("TETO: treino conjunto (não sequencial)",
         lambda s: run_joint(tasks, args.epochs, s)),
        ("rede preditiva (sem memória)", lambda s: run_pcnet(tasks, args.epochs, s, False, 0)),
        ("+ memória episódica", lambda s: run_pcnet(tasks, args.epochs, s, True, 0)),
        ("+ memória + consolidação", lambda s: run_pcnet(tasks, args.epochs, s, True, args.sleep)),
        ("+ memória + replay intercalado", lambda s: run_pcnet(
            tasks, args.epochs, s, True, args.sleep, interleave=32)),
    ]
    try:
        import torch  # noqa: F401
        variants += [
            ("MLP (backprop)", lambda s: run_torch(tasks, args.epochs, s, "mlp", args.frame_len)),
            ("GRU (backprop)", lambda s: run_torch(tasks, args.epochs, s, "gru", args.frame_len)),
        ]
    except ImportError:
        pass

    rule("Learn in sequence, measure on everything")
    rows, results = [], {}
    for name, fn in variants:
        stats, mems = [], {}
        for seed in range(args.seeds):
            matrix, mems = fn(seed)
            stats.append(summarize(matrix))
        agg = {k: (float(np.mean([s[k] for s in stats])),
                   float(np.std([s[k] for s in stats]))) for k in stats[0]}
        results[name] = {"agg": agg, "memoria": mems}
        rows.append({
            "modelo": name,
            "NRMSE final (todas)": pm(*agg["final_medio"]),
            "1ª tarefa: acabada": f"{agg['primeira_tarefa_acabada_de_aprender'][0]:.3f}",
            "1ª tarefa: no fim": f"{agg['primeira_tarefa_no_fim'][0]:.3f}",
            "esquecimento": pm(*agg["esquecimento_medio"]),
        })
        print(f"  … {name}")
    print()
    table(rows)
    print("\n  Forgetting = how much each task got worse between just having learned")
    print("  it and reaching the end. Zero would be accumulating; large positive is")
    print("  replacing.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2, default=float))
        print(f"\n  {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
