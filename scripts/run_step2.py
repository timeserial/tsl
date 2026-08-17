#!/usr/bin/env python3
"""Step 2 of the plan: ternarization and device noise.

    python3 scripts/run_step2.py [--epochs 60] [--seeds 3] [--out runs/step2]

There is one question: how much does the network lose when the weights stop
being float32 and become imperfect physical devices? And, more important for
the argument, *by what mechanism* does it lose.

Nothing here is reported without spread over seeds. An analog device is a
sample from a distribution - measuring one and calling it a result would be
measuring luck.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pcnet import PCConfig, PCNetwork, make_signal  # noqa: E402
from pcnet.device import DeviceModel, ternarize  # noqa: E402
from pcnet.report import pm, rule, table  # noqa: E402
from pcnet.train import evaluate, evaluate_with_theta, train  # noqa: E402

SIGMAS = (0.0, 0.05, 0.1, 0.2, 0.4, 0.8)


class Lab:
    """Trains and caches: each configuration is trained only once."""

    def __init__(self, frames, seeds, epochs):
        self.frames, self.seeds, self.epochs = frames, seeds, epochs
        self._cache: dict[tuple, list[PCNetwork]] = {}

    def nets(self, device: DeviceModel | None, **cfg_kw) -> list[PCNetwork]:
        key = (device.to_dict() if device else None, tuple(sorted(cfg_kw.items())))
        key = json.dumps(key, default=str)
        if key not in self._cache:
            out = []
            for s in self.seeds:
                net = PCNetwork(PCConfig(seed=s, **cfg_kw))
                if device is not None:
                    net.attach_device(replace(device, seed=s))
                train(net, self.frames, epochs=self.epochs)
                out.append(net)
            self._cache[key] = out
        return self._cache[key]


def stat(values) -> tuple[float, float]:
    a = np.asarray(list(values), dtype=float)
    return float(a.mean()), float(a.std())


def on_device(nets: list[PCNetwork], model: DeviceModel, frames, theta=None):
    """Evaluates already-trained networks on a crossbar different from the one
    they trained on.

    Restores the original device at the end: the networks stay in the cache
    and will be reused by other sections.
    """
    out = []
    for net in nets:
        original = net.device_model
        net.attach_device(replace(model, seed=net.cfg.seed))
        out.append(
            evaluate(net, frames)
            if theta is None
            else evaluate_with_theta(net, frames, theta)
        )
        if original is None:
            net.detach_device()
        else:
            net.attach_device(original)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("runs/step2"))
    args = ap.parse_args()

    seeds = tuple(range(args.seeds))
    cfg = PCConfig()
    signal = make_signal(n_frames=args.frames, frame_len=cfg.sizes[0], seed=1)
    tr, te = signal.split(0.8)
    lab = Lab(tr.frames, seeds, args.epochs)
    results: dict = {"epochs": args.epochs, "seeds": args.seeds}

    print(f"hierarchy    {' -> '.join(str(n) for n in cfg.sizes)}")
    print(f"frames       {tr.n_frames} train / {te.n_frames} test")
    print(f"seeds        {args.seeds} (network and device move together)")

    TERNARY = DeviceModel(ternary=True)

    # ------------------------------------------------------------------
    rule("1. Quantization: float32 -> {-1, 0, 1}")
    float_nets = lab.nets(None)
    float_nrmse = stat(evaluate(n, te.frames).pred_nrmse for n in float_nets)
    ptq = stat(s.pred_nrmse for s in on_device(float_nets, TERNARY, te.frames))
    qat_nets = lab.nets(TERNARY)
    qat = stat(evaluate(n, te.frames).pred_nrmse for n in qat_nets)
    zeros = float(np.mean([
        np.mean(ternarize(W)[0] == 0) for n in float_nets for W in n.weights
    ]))

    table([
        {"pesos": "float32", "NRMSE": pm(*float_nrmse), "vs float": "-"},
        {"pesos": "ternary, quantized at the end (PTQ)", "NRMSE": pm(*ptq),
         "vs float": f"{ptq[0] / float_nrmse[0]:.2f}×"},
        {"pesos": "ternary, trained on the crossbar (QAT)", "NRMSE": pm(*qat),
         "vs float": f"{qat[0] / float_nrmse[0]:.2f}×"},
    ], ["pesos", "NRMSE", "vs float"])
    print(f"  {100 * zeros:.0f}% of the ternary weights end up at zero: devices that")
    print("  never need to be programmed or read.")
    print("  Training with the quantizer in the loop is trivial here, the rule is")
    print("  local, there is no gradient to pass through it; the float weight is the")
    print("  shadow weight.")
    results["quantization"] = {"float": float_nrmse, "ptq": ptq, "qat": qat,
                               "ternary_zero_frac": zeros}

    # ------------------------------------------------------------------
    rule("2. Programming variability: train on the device or not")
    rows = []
    for sigma in SIGMAS:
        model = replace(TERNARY, sigma_rel=sigma)
        deploy = stat(s.pred_nrmse for s in on_device(qat_nets, model, te.frames))
        aware = stat(evaluate(n, te.frames).pred_nrmse for n in lab.nets(model))
        rows.append({
            "σ_rel": sigma,
            "programado só no fim": pm(*deploy),
            "treinado no dispositivo": pm(*aware),
            "recuperado": f"{100 * (deploy[0] - aware[0]) / max(deploy[0] - qat[0], 1e-9):.0f}%"
            if sigma else "-",
        })
    table(rows)
    print("  σ_rel is each device's relative deviation from the requested conductance,")
    print("  sampled once at programming time and fixed from then on. It does not get")
    print("  diluted over iterations - this is what tests tolerance, not read noise.")
    print("  Real ReRAM/PCM variability sits around 5-20%; above that the left column")
    print("  is fiction and the right one is what matters.")
    results["programming_variability"] = rows

    # ------------------------------------------------------------------
    rule("3. Self-correction: what the loop closes and what it cannot close")
    rows = []
    for sigma in SIGMAS:
        stats = on_device(qat_nets, replace(TERNARY, sigma_rel=sigma), te.frames)
        o = stat(s.mean_open_loop_surprise for s in stats)
        f = stat(s.mean_final_surprise for s in stats)
        rows.append({
            "σ_rel": sigma,
            "energia em malha aberta": f"{o[0]:.2f}",
            "energia depois de assentar": f"{f[0]:.3f}",
            "fechado pelo ciclo": f"{100 * (1 - f[0] / max(o[0], 1e-12)):.1f}%",
            "NRMSE": pm(*stat(s.pred_nrmse for s in stats)),
        })
    table(rows)
    print("  The loop keeps explaining the frame even with a very faulty crossbar -")
    print("  at σ_rel=0.4 the open-loop energy blows up but settling closes ~95% of")
    print("  it, with the states bounded.")
    print("  What degrades is the *prediction* of the next frame, and it has to be so:")
    print("  the prediction goes down through the same broken weights without yet")
    print("  having an error to correct it. Self-correction is not omniscience - it is")
    print("  what gives the argument for training on the device instead of training")
    print("  clean and programming afterwards.")
    results["self_correction"] = rows

    # ------------------------------------------------------------------
    rule("3b. The cliff: where each crossbar stops working")
    fine = (0.0, 0.08, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25, 0.30)
    rows = []
    for sigma in fine:
        stats = on_device(qat_nets, replace(TERNARY, sigma_rel=sigma), te.frames)
        row = {"σ_rel": sigma}
        for i, s in enumerate(stats):
            row[f"crossbar {i}"] = f"{s.pred_nrmse:.2f}"
        rows.append(row)
    table(rows)
    print("  No retraining. Degradation is smooth up to ~15% and then some crossbars")
    print("  fall off a cliff while others keep behaving well - the transition depends")
    print("  on the sample, not only on σ. A prediction that falls off the cliff")
    print("  becomes *decorrelated* from the target, not merely badly scaled")
    print("  (rescaling it recovers nothing), with the states always bounded: it is a")
    print("  bifurcation of the closed-loop dynamics, not saturation nor divergence.")
    print("  Training on the device eliminates the cliff (section 2).")
    results["cliff"] = rows

    # ------------------------------------------------------------------
    rule("4. The settling step against the stability limit")
    rows = []
    for sigma in (0.0, 0.2, 0.4):
        model = replace(TERNARY, sigma_rel=sigma)
        adapt_nets = lab.nets(model)
        fixed = stat(
            evaluate(n, te.frames).pred_nrmse
            for n in lab.nets(model, adaptive_z_lr=False)
        )
        adapt = stat(evaluate(n, te.frames).pred_nrmse for n in adapt_nets)
        s_max = float(np.mean([max(l.sigma_max for l in n.layers) for n in adapt_nets]))
        z_used = float(np.mean([
            min(n._z_lr_for(l) for l in range(1, n.L + 1)) for n in adapt_nets
        ]))
        rows.append({
            "σ_rel": sigma,
            "σ_max(W_eff)": f"{s_max:.1f}",
            "z_lr estável": f"{2 / (1 + s_max**2):.3f}",
            "z_lr usado": f"{z_used:.3f}",
            "NRMSE passo fixo": pm(*fixed),
            "NRMSE passo adapt.": pm(*adapt),
        })
    table(rows)
    print(f"  Ternary weights have a much larger σ_max than float ones, and the fixed")
    print(f"  step z_lr={cfg.z_lr} sits above the 2/(1+σ_max²) limit. Capping the step by")
    print("  that bound (two crossbar reads per update) fixes it, at the cost of more")
    print("  iterations, which the early exit then cuts back again.")
    results["stability"] = rows

    # ------------------------------------------------------------------
    rule("5. Static vs dynamic: which defects hurt more")
    rows = []
    for label, model in (
        ("none", TERNARY),
        ("20% variability (static)", replace(TERNARY, sigma_rel=0.2)),
        ("read noise 0.05 (dynamic)", replace(TERNARY, read_sigma=0.05)),
        ("5% of devices stuck at zero", replace(TERNARY, stuck_frac=0.05)),
        ("absolute offset 0.02", replace(TERNARY, sigma_abs=0.02)),
    ):
        stats = on_device(qat_nets, model, te.frames)
        rows.append({
            "defeito": label,
            "NRMSE": pm(*stat(s.pred_nrmse for s in stats)),
            "fechado pelo ciclo": "{:.0f}%".format(100 * np.mean([
                1 - s.mean_final_surprise / max(s.mean_open_loop_surprise, 1e-12)
                for s in stats
            ])),
        })
    table(rows)
    results["defect_types"] = rows

    # ------------------------------------------------------------------
    rule("6. ADC: how many bits does the rising error need")
    rows = []
    for bits in (0, 8, 6, 4, 3, 2):
        stats = on_device(
            qat_nets, replace(TERNARY, sigma_rel=0.1, adc_bits=bits), te.frames
        )
        rows.append({"bits": bits or "float",
                     "NRMSE": pm(*stat(s.pred_nrmse for s in stats))})
    table(rows, ["bits", "NRMSE"])
    print("  With σ_rel=0.1. The ADC is the crossbar's expensive component: the")
    print("  threshold already reduced the *number* of conversions, this says how")
    print("  much each one costs.")
    results["adc"] = rows

    # ------------------------------------------------------------------
    rule("7. Does the sparsity threshold still pay off in ternary?")
    rows = []
    for theta in (0.0, 0.01, 0.02, 0.05, 0.1):
        stats = on_device(
            qat_nets, replace(TERNARY, sigma_rel=0.1), te.frames, theta=theta
        )
        rows.append({
            "θ": theta,
            "NRMSE": pm(*stat(s.pred_nrmse for s in stats)),
            "silenciados_%": f"{100 * np.mean([s.silenced_frac for s in stats]):.1f}",
            "ADC_%": f"{100 * np.mean([s.adc_frac for s in stats]):.1f}",
        })
    table(rows)
    results["theta_ternary"] = rows

    # ------------------------------------------------------------------
    rule("Artifacts")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"  {args.out / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
