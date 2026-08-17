#!/usr/bin/env python3
"""Step 1 of the plan: validates the prototype and prints the instrumentation.

    python3 scripts/run_phase0.py [--epochs 60] [--out runs/phase0]

Four questions, four answers:

  1. Does the settling dynamics converge?           -> surprise/iteration table
  2. Does local learning work?                      -> NRMSE vs. baseline
  3. Does the sparsity threshold pay off?           -> θ sweep
  4. Is compute proportional to surprise?           -> mundane vs. event
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pcnet import PCConfig, PCNetwork, make_signal, persistence_nrmse  # noqa: E402
from pcnet.export import save_npz, write_c_header, write_golden  # noqa: E402
from pcnet.metrics import EXIT_CEILING, EXIT_EXPLAINED, EXIT_STALLED  # noqa: E402
from pcnet.report import rule, table  # noqa: E402
from pcnet.train import evaluate, theta_sweep, train  # noqa: E402

THETAS = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("runs/phase0"))
    args = ap.parse_args()

    cfg = PCConfig(seed=args.seed)
    signal = make_signal(n_frames=args.frames, frame_len=cfg.sizes[0], seed=args.seed + 1)
    train_sig, test_sig = signal.split(0.8)
    net = PCNetwork(cfg)
    results: dict = {"config": asdict(cfg)}

    print(f"hierarchy    {' -> '.join(str(n) for n in cfg.sizes)}")
    print(f"parameters   {sum(W.size for W in net.weights) + net.A.size}")
    print(f"frames       {train_sig.n_frames} train / {test_sig.n_frames} test"
          f"  ({cfg.sizes[0]} samples each, no overlap)")

    # ------------------------------------------------------------------
    rule("2. Local learning (no backprop)")
    baseline = persistence_nrmse(test_sig.frames)
    before = evaluate(net, test_sig.frames)
    history = train(net, train_sig.frames, epochs=args.epochs, eval_frames=test_sig.frames)
    after = history[-1]

    print(f"  Next-frame prediction NRMSE (1.0 = as good as predicting zero)")
    print(f"    baseline (repeat previous frame)         : {baseline:.3f}")
    print(f"    network before training                  : {before.pred_nrmse:.3f}")
    print(f"    network after {args.epochs} passes    {' ' * 14}: {after.pred_nrmse:.3f}")
    verdict = "learned" if after.pred_nrmse < min(baseline, before.pred_nrmse) else "did NOT learn"
    print(f"    -> {verdict}")
    results["learning"] = {
        "baseline_persistence_nrmse": baseline,
        "nrmse_before": before.pred_nrmse,
        "nrmse_after": after.pred_nrmse,
        "history_nrmse": [h.pred_nrmse for h in history],
    }

    # ------------------------------------------------------------------
    rule("1. Convergence of the settling dynamics")
    traces = net.run(test_sig.frames, learn=False, reset=True)
    max_len = max(len(t.surprise) for t in traces)
    curve = [
        float(np.mean([t.surprise[k] for t in traces if len(t.surprise) > k]))
        for k in range(max_len)
    ]
    rows = [
        {
            "iter": k,
            "surpresa": round(curve[k], 4),
            "vs_iter0": f"{curve[k] / curve[0]:.3f}",
            "tramas_vivas": sum(1 for t in traces if len(t.surprise) > k),
        }
        for k in range(max_len)
    ]
    table(rows, ["iter", "surpresa", "vs_iter0", "tramas_vivas"])
    print("  (each row's mean is over the frames still alive, which are fewer and")
    print("   fewer, so the column can rise without any single frame getting worse.)")
    # Monotonicity has to be checked frame by frame. The last evaluation can
    # rise: it is exactly that rise that makes the gain criterion stop the loop.
    monotone = sum(
        1
        for t in traces
        if all(t.surprise[k + 1] <= t.surprise[k] + 1e-9
               for k in range(len(t.surprise) - 2))
    )
    print(f"  frames with monotonically decreasing energy: {monotone}/{len(traces)}")
    print(f"  final / initial energy (mean): "
          f"{np.mean([t.final_surprise / max(t.open_loop_surprise, 1e-12) for t in traces]):.3f}")
    reasons = {r: sum(1 for t in traces if t.exit_reason == r) for r in
               (EXIT_EXPLAINED, EXIT_STALLED, EXIT_CEILING)}
    print(f"  exits: " + ", ".join(f"{k}={v}" for k, v in reasons.items()))
    results["settling"] = {"surprise_curve": curve, "monotone_frames": monotone,
                           "n_frames": len(traces), "exit_reasons": reasons}

    # ------------------------------------------------------------------
    rule("3. Sparsity threshold: what silence costs")
    sweep = theta_sweep(net, test_sig.frames, THETAS)
    rows = []
    for th, s in sweep.items():
        rows.append({
            "θ": th,
            "NRMSE": round(s.pred_nrmse, 3),
            "silenciados_%": round(100 * s.silenced_frac, 1),
            "ADC_%": round(100 * s.adc_frac, 1),
            "MACs_sobem_%": round(100 * s.mac_up_frac, 1),
            "iters": round(s.mean_iters, 2),
            "explicado_%": round(100 * s.explained_frac, 1),
        })
    table(rows, ["θ", "NRMSE", "silenciados_%", "ADC_%", "MACs_sobem_%", "iters",
                 "explicado_%"])
    print("  ADC_% = conversions made / conversions of a dense network always running")
    print(f"          to the ceiling ({cfg.max_iters + 1} passes × {net.eps_per_pass} units).")
    print("          This is the number that decides the energy budget on the crossbar.")
    results["theta_sweep"] = {str(th): s.as_row() for th, s in sweep.items()}

    # ------------------------------------------------------------------
    rule("4. Compute proportional to surprise")
    ev = make_signal(n_frames=200, frame_len=cfg.sizes[0], seed=args.seed + 99,
                     n_events=12)
    ev_traces = net.run(ev.frames, learn=False, reset=True)
    mask = np.zeros(len(ev_traces), dtype=bool)
    mask[ev.event_frames] = True
    rows = []
    for label, sel in (("mundane", ~mask), ("with transient", mask)):
        sub = [t for t, m in zip(ev_traces, sel) if m]
        rows.append({
            "tramas": label,
            "n": len(sub),
            "surpresa": round(float(np.mean([t.open_loop_surprise for t in sub])), 3),
            "iters": round(float(np.mean([t.iters for t in sub])), 2),
            "conversões_ADC": round(float(np.mean([t.adc_conversions for t in sub])), 1),
        })
    table(rows, ["tramas", "n", "surpresa", "iters", "conversões_ADC"])
    ratio = rows[1]["conversões_ADC"] / max(rows[0]["conversões_ADC"], 1e-9)
    print(f"  a surprising frame costs {ratio:.1f}× a mundane one")
    results["surprise_scaling"] = {"rows": rows, "adc_ratio": ratio}

    # ------------------------------------------------------------------
    rule("Artifacts")
    out = args.out
    save_npz(out / "model.npz", net)
    write_c_header(out / "model.h", net)
    write_golden(out / "golden.h", net, test_sig.frames)
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    for f in ("model.npz", "model.h", "golden.h", "results.json"):
        print(f"  {out / f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
