#!/usr/bin/env python3
"""Benchmark da neurogénese por novidade no protocolo sequencial de 3 tarefas.

    python3 scripts/neurogenesis_bench.py --variant neuro --sizes deep

Protocolo fixo (o mesmo dos marcos): make_tasks(3, 64, 400, seed=1),
40 épocas por tarefa, treino tarefa a tarefa sem revisitar, avaliação com
evaluate_task no fim de cada tarefa. Config campeã:
PCConfig(seed=s, fast_path=False, use_precision=True, metaplasticity opcional).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from continual import make_tasks, evaluate_task  # noqa: E402
from pcnet import PCConfig, PCNetwork  # noqa: E402
from pcnet.train import train  # noqa: E402
from pcnet.neurogenesis import NeurogenesisConfig, NeurogenesisNetwork  # noqa: E402

SIZES = {"deep": (64, 32, 16, 8), "shallow": (64, 24)}
EPOCHS = 40
TASKS = make_tasks(3, 64, 400, seed=1)


def build(variant: str, seed: int, sizes: tuple[int, ...], rule: str,
          meta_lambda: float = 100.0):
    meta = meta_lambda if variant in ("meta", "neuro+meta") else 0.0
    cfg = PCConfig(seed=seed, fast_path=False, use_precision=True,
                   metaplasticity=meta, sizes=sizes)
    if variant.startswith("neuro"):
        return NeurogenesisNetwork(cfg, NeurogenesisConfig(protect_rule=rule))
    return PCNetwork(cfg)


def run_seq(variant: str, seed: int, sizes, rule: str = "min",
            meta_lambda: float = 100.0) -> dict:
    net = build(variant, seed, sizes, rule, meta_lambda)
    matrix = np.zeros((3, 3))
    recruits_per_task = []
    for k, (tr, _) in enumerate(TASKS):
        before = getattr(net, "n_recruited", 0)
        train(net, tr, epochs=EPOCHS)
        recruits_per_task.append(getattr(net, "n_recruited", 0) - before)
        for j, (tr_j, te_j) in enumerate(TASKS):
            matrix[k, j] = evaluate_task(net, te_j, tr_j)
    return {
        "final_mean": float(matrix[-1].mean()),
        "task0_final": float(matrix[-1, 0]),
        "task0_just_learned": float(matrix[0, 0]),
        "matrix": matrix.tolist(),
        "recruits_per_task": recruits_per_task,
        "recruit_log": [dict(r, per_level={str(k): v for k, v in r["per_level"].items()})
                        for r in getattr(net, "recruit_log", [])],
    }


def run_joint(seed: int, sizes) -> dict:
    """Teto: treino conjunto intercalado em blocos de 16 (como em continual.py)."""
    net = PCNetwork(PCConfig(seed=seed, fast_path=False, use_precision=True,
                             sizes=sizes))
    block = 16
    n_blocks = min(len(t[0]) for t in TASKS) // block
    mixed = np.concatenate([tr[b * block:(b + 1) * block]
                            for b in range(n_blocks) for tr, _ in TASKS])
    train(net, mixed, epochs=EPOCHS)
    row = [evaluate_task(net, te, tr) for tr, te in TASKS]
    return {"final_mean": float(np.mean(row)), "task0_final": float(row[0]),
            "task0_just_learned": float(row[0]), "matrix": [row],
            "recruits_per_task": [], "recruit_log": []}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=["none", "meta", "neuro", "neuro+meta", "joint"])
    ap.add_argument("--sizes", default="deep", choices=list(SIZES))
    ap.add_argument("--rule", default="min", choices=["min", "hyper"])
    ap.add_argument("--meta-lambda", type=float, default=100.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sizes = SIZES[args.sizes]
    per_seed = {}
    for s in args.seeds:
        t0 = time.time()
        r = (run_joint(s, sizes) if args.variant == "joint"
             else run_seq(args.variant, s, sizes, args.rule, args.meta_lambda))
        per_seed[s] = r
        print(f"[{args.variant}/{args.sizes}] seed {s}: "
              f"final={r['final_mean']:.3f} t0={r['task0_final']:.3f} "
              f"recruits={r['recruits_per_task']} ({time.time()-t0:.0f}s)",
              flush=True)

    vals = {k: [per_seed[s][k] for s in args.seeds]
            for k in ("final_mean", "task0_final", "task0_just_learned")}
    summary = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
               for k, v in vals.items()}
    out = {"variant": args.variant, "sizes": args.sizes, "rule": args.rule,
           "meta_lambda": args.meta_lambda, "epochs": EPOCHS,
           "summary": summary, "per_seed": {str(k): v for k, v in per_seed.items()}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({"variant": args.variant, "sizes": args.sizes,
                      **{k: f"{v['mean']:.3f}±{v['std']:.3f}"
                         for k, v in summary.items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
