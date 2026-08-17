#!/usr/bin/env python3
"""Ablations: each idea in isolation, measured in the same place.

    .venv/bin/python scripts/ablate.py --toy
    .venv/bin/python scripts/ablate.py --data ETTm1.csv --column OT
    .venv/bin/python scripts/ablate.py --har "UCI HAR Dataset" --signal total_acc_x

A beautiful idea that is not measured is an opinion. This script runs each
variant of the architecture on the same dataset, with the same seeds, and prints
two columns: how well it predicts, and how many ADC conversions it cost. An idea
only becomes a default if it wins on at least one without hurting the other.

The spread over seeds is always reported. Many of the differences here are
smaller than it, and that has to be visible in the table instead of being
discovered later.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pcnet import PCConfig, PCNetwork  # noqa: E402
from pcnet.datasets import (  # noqa: E402
    load_csv_column, load_toy, load_uci_har_inertial, load_wav,
)
from pcnet.report import pm, rule, table  # noqa: E402
from pcnet.train import train  # noqa: E402

WARMUP = 32

VARIANTS: dict[str, dict] = {
    "passo 1 (base)": {},
    "+ precisão": dict(use_precision=True),
    "+ precisão por unidade": dict(use_precision=True, precision_per_unit=True),
    "+ escalas de tempo (diag)": dict(level_transition="diagonal"),
    "+ escalas de tempo (densa)": dict(level_transition="dense"),
    "+ via rápida": dict(fast_path=True, fast_path_lr=0.05),
    "+ via rápida + precisão": dict(fast_path=True, fast_path_lr=0.05,
                                    use_precision=True),
}


def nrmse(pred, target) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2) / np.mean(target**2)))


def measure(ds, cfg_kw, seeds, epochs, thetas=None):
    """Trains per seed; sweeps θ only at inference (θ is an inference knob)."""
    thetas = thetas or (PCConfig().theta,)
    per_theta = {th: [] for th in thetas}
    n_params = 0
    for seed in seeds:
        net = PCNetwork(PCConfig(seed=seed, sizes=_sizes(ds.frame_len), **cfg_kw))
        train(net, ds.train, epochs=epochs)
        n_params = (
            sum(W.size for W in net.weights)
            + net.A.size
            + sum(t.n_params for t in net.transitions if t is not None)
            + (net.A0.size if net.A0 is not None else 0)
        )
        base = net.cfg
        for th in thetas:
            net.cfg = replace(base, theta=th)
            net.reset()
            for frame in ds.train[-WARMUP:]:
                net.step(frame, learn=False)
            preds, traces = [], []
            for frame in ds.test:
                preds.append(net.predict_next())
                traces.append(net.step(frame, learn=False))
            per_theta[th].append((
                nrmse(np.array(preds), ds.test),
                float(np.mean([t.adc_conversions for t in traces])),
                float(np.mean([t.iters for t in traces])),
            ))
        net.cfg = base
    return per_theta, n_params


def _sizes(frame_len):
    sizes, n = [frame_len], frame_len
    while n > 8 and len(sizes) < 4:
        n //= 2
        sizes.append(n)
    return tuple(sizes)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path)
    ap.add_argument("--column", default="OT")
    ap.add_argument("--wav", type=Path)
    ap.add_argument("--har", type=Path)
    ap.add_argument("--signal", default="total_acc_x")
    ap.add_argument("--toy", action="store_true")
    ap.add_argument("--frame-len", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--thetas", type=float, nargs="*")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    if args.data:
        ds = load_csv_column(args.data, args.column, args.frame_len)
    elif args.wav:
        ds = load_wav(args.wav, args.frame_len)
    elif args.har:
        ds = load_uci_har_inertial(args.har, args.signal, frame_len=args.frame_len)
    else:
        ds = load_toy(frame_len=args.frame_len)

    seeds = tuple(range(args.seeds))
    thetas = tuple(args.thetas) if args.thetas else (PCConfig().theta,)
    print(ds.describe())
    print(f"{args.seeds} seeds, {args.epochs} passes, θ = "
          f"{', '.join(str(t) for t in thetas)}")

    rule("Each idea in isolation")
    rows, results = [], {}
    for name, kw in VARIANTS.items():
        per_theta, n_params = measure(ds, kw, seeds, args.epochs, thetas)
        results[name] = {str(th): v for th, v in per_theta.items()}
        for th, vals in per_theta.items():
            arr = np.array(vals)
            rows.append({
                "variante": name if th == thetas[0] else "",
                "θ": th,
                "NRMSE": pm(float(arr[:, 0].mean()), float(arr[:, 0].std())),
                "ADC/trama": f"{arr[:, 1].mean():,.0f}",
                "iters": f"{arr[:, 2].mean():.1f}",
                "parâmetros": n_params if th == thetas[0] else "",
            })
        print(f"  … {name}")
    print()
    table(rows, ["variante", "θ", "NRMSE", "ADC/trama", "iters", "parâmetros"])
    print("\n  An idea only deserves to stay on by default if it wins in one column")
    print("  without hurting the other, and by a margin larger than the spread.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"dataset": ds.describe(), "variants": results}, indent=2, default=float
        ))
        print(f"\n  {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
