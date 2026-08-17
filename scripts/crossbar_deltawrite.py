#!/usr/bin/env python3
"""Delta-write: the missing arm of the crossbar study.

crossbar_study.py showed that continuous analog writing dies on the
realistic device (1.94, +236%): 417M micro-pulses, each with per-pulse
noise, drown the weights. This arm tests the master-shadow architecture
that the fixed-point path already uses in digital (Q20→Q12):

- digital MASTERS accumulate the micro-updates (SRAM: no wear, no noise);
- every N frames, ONE physical write per element programs the accumulated
  delta into the crossbar (a large pulse: the per-pulse noise, which is a
  fraction of the RANGE, amortizes over a delta N times larger);
- ALL reads (prediction, transpose) stay analog, with the weights "stale"
  until the next flush - that is the architecture's real cost.

Grid: the study's realistic device × N ∈ {10, 100, 400}; 3 seeds;
plus the ideal with N=100 (separating the staleness effect from the
physics effect).
"""
import sys, time, json
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
sys.path.insert(0, "experiments")
import crossbar_study as CS
import profundidade_empilhamento as PE
from pcnet.crossbar import CrossbarModel
from pcnet.dtypes import F

EPOCHS = CS.EPOCHS
LR = PE.LR


def run_delta(job):
    group, N, seed, kwargs = job
    t0 = time.time()
    model = CrossbarModel(**kwargs)
    net = CS.make_xbar_net(seed, model)
    gx = net.gated; xw = net.layers[0].xw
    # digital masters: start equal to what was programmed
    acc = {k: 0.0 for k in ("W", "A", "G", "b")}
    accW = np.zeros(xw.shape, dtype=np.float64)
    accA = np.zeros(gx.xA.shape, dtype=np.float64)
    accG = np.zeros(gx.xG.shape, dtype=np.float64)
    accb = np.zeros(gx.xb.shape, dtype=np.float64)
    frames = 0
    for _ in range(EPOCHS):
        for fr in PE.MIXED:
            s_prev = net._z_prev[net.L].copy()
            net.step(fr, learn=False)
            tgt = np.asarray(fr, dtype=F)
            # the rule, with analog reads and updates going to the MASTERS
            c = np.tanh(gx.xA.read(s_prev))
            g = PE._sigm(gx.xG.read(s_prev) + gx.xb.read(gx._one))
            prior = ((1.0 - g) * s_prev + g * c).astype(F)
            e = (tgt - xw.read(prior)).astype(F)
            h = xw.read_T(e)
            npr = float(prior @ prior) + 1e-6
            ns = float(s_prev @ s_prev) + 1e-6
            mod_g = (h * (c - s_prev) * g * (1.0 - g)).astype(F)
            accW += (LR / npr) * np.outer(e, prior)
            accA += (LR / ns) * np.outer(h * g * (1.0 - c * c), s_prev)
            accG += (LR / ns) * np.outer(mod_g, s_prev)
            accb += (LR * mod_g)[:, None]
            frames += 1
            if frames % N == 0:      # flush: ONE physical write of the delta
                xw.update(accW.astype(F)); accW[:] = 0.0
                gx.xA.update(accA.astype(F)); accA[:] = 0.0
                gx.xG.update(accG.astype(F)); accG[:] = 0.0
                gx.xb.update(accb.astype(F)); accb[:] = 0.0
                net.layers[0].refresh_device()
    if model.drift_nu > 0.0:
        for xb in CS._xbars(net):
            xb.apply_drift()
        net.layers[0].refresh_device()
    score = float(np.mean([PE.evaluate_task(net, te, tr)
                           for tr, te in PE.TASKS]))
    pulses = sum(xb.stats()["pulses_issued"] for xb in CS._xbars(net))
    rails = max(xb.stats()["rail_frac"] for xb in CS._xbars(net))
    print(f"  {group:<16} N={N:<4} seed {seed}: {score:.4f}  "
          f"(pulses {pulses/1e6:.0f}M, rails {rails:.2f}, "
          f"{time.time()-t0:.0f}s)", flush=True)
    return group, N, seed, score, pulses


if __name__ == "__main__":
    jobs = []
    for N in (10, 100, 400):
        jobs += [("delta-realistic", N, s, dict(CS.REALISTIC)) for s in (0, 1, 2)]
    jobs += [("delta-ideal", 100, s, {}) for s in (0, 1, 2)]
    print(f"{len(jobs)} delta-write training runs", flush=True)
    with ProcessPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(run_delta, jobs))
    print("\n" + "=" * 68, flush=True)
    print("references: continuous ideal 0.5779±0.0045 | continuous realistic "
          "1.9438±0.1406 (417M pulses)", flush=True)
    out = {}
    for group, N, seed, score, pulses in results:
        out.setdefault((group, N), []).append((score, pulses))
    for (group, N), v in sorted(out.items()):
        sc = np.array([x[0] for x in v]); pu = np.mean([x[1] for x in v])
        print(f"{group:<16} N={N:<4} {sc.mean():.4f} ± {sc.std():.4f}  "
              f"({pu/1e6:.1f}M pulses)", flush=True)
    json.dump({f"{g}_N{n}": [x[0] for x in v]
               for (g, n), v in out.items()},
              open("runs/crossbar_study/deltawrite.json", "w"), indent=1)
