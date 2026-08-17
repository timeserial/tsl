#!/usr/bin/env python3
"""Crossbar study: which analog non-idealities hurt TSL, and by how much.

    .venv/bin/python -u scripts/crossbar_study.py

The cell is the canonical building block from
experiments/profundidade_empilhamento.py (64→24, gated transition, two-stroke
rule, milestone 0.579±0.004 on 3 interleaved worlds, 80 epochs). ALL weight
operations - direct read (prediction), transpose read (h = Wᵀe) and rank-1
write - of W₀, A, G and b go through the crossbar model in
src/pcnet/crossbar.py. The state s and the tanh/σ non-linearities stay
digital (periphery), as does the error threshold θ.

Protocol:
  * sanity gate: the ideal crossbar has to reproduce the milestone - the
    Crossbar's ideal mode is a float32 accumulator bit-identical to float
    training, so equality is exact per seed;
  * ablation: ONE non-ideality at a time, at 3 magnitudes
    (mild/realistic/pessimistic, justified in pcnet/crossbar.py and in the
    tables below), 3 seeds × 80 epochs;
  * "realistic device": everything on at the middle values, 4 seeds;
  * write gate θ_w ∈ {0, 1e-3, 5e-3}: NRMSE and physical pulses
    (the endurance/accuracy trade-off), on the ideal crossbar and on the
    realistic device.

Physical scales fixed by probing float training (seed 0, 80 epochs):
max |W₀|=0.87, |A|=4.1, |G|=1.7, |b|=4.7; direct reads of W₀ ≤ 1.0,
transpose ≤ 12, reads of A/G ≤ 12; median per-element |Δw| from
2e-4 (G) to 4e-3 (b). Hence the w_max values (per-array weight full scale,
with headroom for other seeds) and the ADC full scales per direction.
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# small matrices: multi-threaded BLAS only gets in the way of 7 parallel processes
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
for p in ("src", "scripts", "experiments"):
    sys.path.insert(0, str(ROOT / p))

import profundidade_empilhamento as PE  # noqa: E402  (builds TASKS/MIXED)
from pcnet import PCConfig, PCNetwork  # noqa: E402
from pcnet.crossbar import Crossbar, CrossbarModel  # noqa: E402
from pcnet.dtypes import F  # noqa: E402

EPOCHS = 80
MARK = 0.579  # the building block's published milestone (0.579 ± 0.004, 4 seeds)

# per-array weight full scale (probed + ~30-40% headroom) and ADC full
# scales per array and direction (powers of 2 above the measured signal).
W_MAX = {"W0": 1.5, "A": 5.0, "G": 2.5, "b": 6.0}
ADC_FS = {"W0": (2.0, 16.0), "A": (16.0, 16.0), "G": (16.0, 16.0)}


# ---------------------------------------------------------------------------
# network ↔ crossbars wiring
# ---------------------------------------------------------------------------
class XbarLayer:
    """PCLayer with the three weight operations going through the crossbar.

    Delegates buffers and σ_max to the original PCLayer so that the settling
    path (adaptive step included) is the canonical one.
    """

    def __init__(self, base, xw: Crossbar) -> None:
        self.base = base
        self.xw = xw
        # points the effective operator at the crossbar's WITHOUT re-estimating
        # σ_max: the values are the same and the power iteration's warm start
        # has to follow the same trajectory as float training.
        base.W_eff = xw.W_eff

    def predict(self, z_above):
        b = self.base
        b.a[:] = self.xw.read(z_above)          # direct read (prediction)
        b._base = b._f(b.a).astype(F, copy=False)
        b.zhat = b._base
        return b.zhat

    def modulated_error(self, eps_below):
        return self.base.modulated_error(eps_below)

    def backward(self, eps_mod, eps_raw=None):
        return self.xw.read_T(eps_mod)          # transpose read

    def refresh_device(self):
        b = self.base
        b.W_eff = self.xw.W_eff
        b._estimate_sigma_max()

    @property
    def sigma_max(self):
        return self.base.sigma_max

    @property
    def zhat(self):
        return self.base.zhat

    def macs_down(self):
        return self.base.macs_down()

    def macs_up(self, nnz):
        return self.base.macs_up(nnz)


class XbarGated:
    """GatedTransition with A, G and b on crossbars (b = its own column, read
    with input 1; tanh and σ are digital periphery)."""

    def __init__(self, base, xA: Crossbar, xG: Crossbar, xb: Crossbar) -> None:
        self.base = base
        self.xA, self.xG, self.xb = xA, xG, xb
        self._one = np.ones(1, dtype=F)

    def predict(self, z_prev, u=None):
        a_c = self.xA.read(z_prev)
        a_g = self.xG.read(z_prev) + self.xb.read(self._one)
        c = np.tanh(a_c)
        g = (1.0 / (1.0 + np.exp(-a_g))).astype(F, copy=False)
        return ((1.0 - g) * z_prev + g * c).astype(F, copy=False)


def make_xbar_net(seed: int, model: CrossbarModel) -> PCNetwork:
    net = PCNetwork(PCConfig(seed=seed, sizes=(64, 24), **PE.BASE))
    gt = net.gated
    xw = Crossbar(net.layers[0].W.shape, W_MAX["W0"], model, seed * 1000 + 1,
                  adc_range=ADC_FS["W0"][0], adc_range_T=ADC_FS["W0"][1])
    xA = Crossbar(gt.A.shape, W_MAX["A"], model, seed * 1000 + 2,
                  adc_range=ADC_FS["A"][0], adc_range_T=ADC_FS["A"][1])
    xG = Crossbar(gt.G.shape, W_MAX["G"], model, seed * 1000 + 3,
                  adc_range=ADC_FS["G"][0], adc_range_T=ADC_FS["G"][1])
    xb = Crossbar((gt.b.size, 1), W_MAX["b"], model, seed * 1000 + 4,
                  use_adc=False, use_ir=False)
    xw.program(net.layers[0].W)
    xA.program(gt.A)
    xG.program(gt.G)
    xb.program(gt.b[:, None])
    net.layers[0] = XbarLayer(net.layers[0], xw)
    net.gated = XbarGated(gt, xA, xG, xb)
    return net


def _xbars(net) -> list[Crossbar]:
    g = net.gated
    return [net.layers[0].xw, g.xA, g.xG, g.xb]


def two_time_update_xbar(net, target, s_prev, lr=PE.LR) -> None:
    """The two-stroke rule, an exact mirror of the building block's, with
    every weight operation on the crossbar: 4 reads (A, G, b, W₀ direct),
    1 transpose read (h) and 4 rank-1 writes per frame."""
    gx = net.gated
    xw = net.layers[0].xw
    c = np.tanh(gx.xA.read(s_prev))
    g = PE._sigm(gx.xG.read(s_prev) + gx.xb.read(gx._one))
    prior = ((1.0 - g) * s_prev + g * c).astype(F)

    e = (target - xw.read(prior)).astype(F)
    h = xw.read_T(e)                       # back-projection: pre-update

    xw.update((lr / (float(prior @ prior) + 1e-6))
              * np.outer(e, prior).astype(F))
    # canonical: re-estimate σ_max after touching W₀ (adaptive step)
    net.layers[0].refresh_device()
    ns = float(s_prev @ s_prev) + 1e-6
    mod_g = (h * (c - s_prev) * g * (1.0 - g)).astype(F)
    gx.xA.update((lr / ns) * np.outer((h * g * (1.0 - c * c)).astype(F), s_prev))
    gx.xG.update((lr / ns) * np.outer(mod_g, s_prev))
    gx.xb.update((F(lr) * mod_g)[:, None])


# ---------------------------------------------------------------------------
# one cell of the grid
# ---------------------------------------------------------------------------
def run_cell(job):
    """job = (group, magnitude, seed, CrossbarModel kwargs)."""
    group, mag, seed, kwargs = job
    t0 = time.time()
    model = CrossbarModel(**kwargs)
    net = make_xbar_net(seed, model)
    L = net.L
    for _ in range(EPOCHS):
        for fr in PE.MIXED:
            s_prev = net._z_prev[L].copy()
            net.step(fr, learn=False)          # settling (inference only)
            two_time_update_xbar(net, np.asarray(fr, dtype=F), s_prev)
    if model.drift_nu > 0.0:                   # drift between train and test
        for xb in _xbars(net):
            xb.apply_drift()
        net.layers[0].refresh_device()         # σ_max tracks the drift
    score = float(np.mean([PE.evaluate_task(net, te, tr)
                           for tr, te in PE.TASKS]))
    agg = {"pulses_issued": 0, "pulses_gated": 0, "clip_events": 0,
           "rail_frac_max": 0.0}
    for xb in _xbars(net):
        s = xb.stats()
        agg["pulses_issued"] += s["pulses_issued"]
        agg["pulses_gated"] += s["pulses_gated"]
        agg["clip_events"] += s["clip_events"]
        agg["rail_frac_max"] = max(agg["rail_frac_max"], s["rail_frac"])
    agg["secs"] = round(time.time() - t0, 1)
    return group, mag, seed, score, agg


# ---------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------
# Magnitudes mild/realistic/pessimistic - rationale in the comments of
# pcnet/crossbar.py (literature ranges cited there):
#   σ_write   1% / 5% / 20% of the range  (typical c2c 0.5-5%; 20% = bad)
#   α (p,d)   0.5,0.5 / 1,2 / 2,5         (near linear / TaOx-HfOx / severe)
#   σ_read    1% / 3% / 10%               (calibrated PCM ~1%; strong RTN 10%)
#   ADC       12 / 8 / 6 bits             (transparent / ISAAC / aggressive)
#   ν drift   0.005 / 0.02 / 0.1          (crystalline / good cell / amorphous)
#   IR γ      2% / 10% / 30%              (small arrays / 64+ / poor wire)
#   on/off    ∞ / 50 / 10, with σ_read=1% (alone, the differential pair
#             cancels G_min exactly - only visible against read noise)
ABLATION = [
    ("write-noise", [("1%", dict(sigma_write=0.01)),
                     ("5%", dict(sigma_write=0.05)),
                     ("20%", dict(sigma_write=0.20))]),
    ("write-NL", [("a=.5/.5", dict(alpha_p=0.5, alpha_d=0.5)),
                  ("a=1/2", dict(alpha_p=1.0, alpha_d=2.0)),
                  ("a=2/5", dict(alpha_p=2.0, alpha_d=5.0))]),
    ("read-noise", [("1%", dict(sigma_read=0.01)),
                    ("3%", dict(sigma_read=0.03)),
                    ("10%", dict(sigma_read=0.10))]),
    ("ADC", [("12b", dict(adc_bits=12)),
             ("8b", dict(adc_bits=8)),
             ("6b", dict(adc_bits=6))]),
    ("drift", [("nu=.005", dict(drift_nu=0.005)),
               ("nu=.02", dict(drift_nu=0.02)),
               ("nu=.1", dict(drift_nu=0.1))]),
    ("IR-drop", [("2%", dict(ir_drop=0.02)),
                 ("10%", dict(ir_drop=0.10)),
                 ("30%", dict(ir_drop=0.30))]),
    ("on/off+1%read", [("inf", dict(sigma_read=0.01)),
                       ("50", dict(on_off=50.0, sigma_read=0.01)),
                       ("10", dict(on_off=10.0, sigma_read=0.01))]),
]

# everything at the middle values - the headline number
REALISTIC = dict(on_off=50.0, sigma_write=0.05, alpha_p=1.0, alpha_d=2.0,
                 sigma_read=0.03, adc_bits=8, drift_nu=0.02, ir_drop=0.10)

# endurance gate: θ_w in weight units; median of the probed |Δw|
# ~2e-4-4e-3, so 1e-3 saves ~half the pulses and 5e-3 ~90%.
GATE_THETAS = [("t=1e-3", 1e-3), ("t=5e-3", 5e-3)]

SEEDS = (0, 1, 2)


def build_jobs():
    jobs = [("ideal", "-", s, {}) for s in SEEDS]
    for group, mags in ABLATION:
        for mag, kw in mags:
            jobs += [(group, mag, s, kw) for s in SEEDS]
    jobs += [("realistic", "middle", s, dict(REALISTIC)) for s in (0, 1, 2, 3)]
    for mag, th in GATE_THETAS:
        jobs += [("gate-ideal", mag, s, dict(theta_write=th)) for s in SEEDS]
        jobs += [("gate-realistic", mag, s,
                  dict(REALISTIC, theta_write=th)) for s in SEEDS]
    return jobs


def verdict(mean: float, base: float) -> str:
    d = 100.0 * (mean - base) / base
    if d <= 2.0:
        return "SURVIVES"
    if d <= 25.0:
        return f"DEGRADES {d:+.1f}%"
    return f"BREAKS {d:+.1f}%"


def main() -> int:
    jobs = build_jobs()
    print(f"{len(jobs)} training runs (80 epochs each), 7 processes\n", flush=True)
    results: dict[tuple[str, str], list] = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=7) as ex:
        futs = [ex.submit(run_cell, j) for j in jobs]
        for fut in as_completed(futs):
            group, mag, seed, score, agg = fut.result()
            results.setdefault((group, mag), []).append((seed, score, agg))
            print(f"  [{time.time()-t0:5.0f}s] {group:15s} {mag:8s} "
                  f"seed {seed}: {score:.4f}  "
                  f"(pulses {agg['pulses_issued']/1e6:.0f}M, "
                  f"saved {agg['pulses_gated']/1e6:.0f}M, "
                  f"rails {agg['rail_frac_max']:.2f}, {agg['secs']:.0f}s)",
                  flush=True)

    # ---- final table -----------------------------------------------------
    base_scores = [sc for _, sc, _ in sorted(results[("ideal", "-")])]
    base = float(np.mean(base_scores))
    print("\n" + "=" * 76)
    print(f"ideal crossbar baseline: {base:.4f} ± {np.std(base_scores):.4f} "
          f"(float milestone: {MARK} ± 0.004; bit-exact equality per seed)")
    print("=" * 76)
    header = (f"{'configuration':<28s} {'NRMSE':>15s} {'Δ vs ideal':>10s}  "
              f"{'pulses':>9s}  verdict")
    print(header)
    print("-" * len(header))
    order = [("ideal", "-")]
    order += [(g, m) for g, mags in ABLATION for m, _ in mags]
    order += [("realistic", "middle")]
    order += [(g, m) for m, _ in GATE_THETAS
              for g in ("gate-ideal", "gate-realistic")]
    out = {}
    for key in order:
        rows = sorted(results[key])
        scores = np.array([sc for _, sc, _ in rows])
        pulses = np.mean([a["pulses_issued"] for _, _, a in rows])
        name = f"{key[0]} {key[1]}"
        v = "(baseline)" if key == ("ideal", "-") else verdict(scores.mean(), base)
        print(f"{name:<28s} {scores.mean():7.4f} ± {scores.std():.4f} "
              f"{100*(scores.mean()-base)/base:+9.1f}%  {pulses/1e6:8.1f}M  {v}")
        out[name] = {"scores": scores.tolist(), "mean": float(scores.mean()),
                     "std": float(scores.std()), "pulses_M": float(pulses / 1e6),
                     "stats": [a for _, _, a in rows]}
    dest = ROOT / "runs" / "crossbar_study" / "results.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(
        {"baseline": base, "epochs": EPOCHS, "results": out}, indent=2))
    print(f"\n  {dest}  ({time.time()-t0:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
