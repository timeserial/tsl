#!/usr/bin/env python3
"""Delta-write: o braço que falta no estudo do crossbar.

O crossbar_study.py mostrou que a escrita analógica contínua morre no
dispositivo realista (1.94, +236%): 417M de micro-impulsos, cada um com
ruído por impulso, afogam os pesos. Este braço testa a arquitetura
mestre-sombra que o ponto fixo já usa em digital (Q20→Q12):

- MESTRES digitais acumulam os micro-updates (SRAM: sem desgaste, sem ruído);
- a cada N tramas, UMA escrita física por elemento programa o delta
  acumulado no crossbar (impulso grande: o ruído por impulso, que é fração
  da GAMA, amortiza-se sobre um delta N vezes maior);
- TODAS as leituras (previsão, transposta) continuam no analógico, com os
  pesos "atrasados" até ao próximo flush — é o custo real da arquitetura.

Grelha: dispositivo realista do estudo × N ∈ {10, 100, 400}; 3 seeds;
mais o ideal com N=100 (separar o efeito do atraso do efeito da física).
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
    # mestres digitais: começam iguais ao que foi programado
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
            # a regra, com leituras analógicas e updates para os MESTRES
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
            if frames % N == 0:      # flush: UMA escrita física do delta
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
          f"(imp {pulses/1e6:.0f}M, rails {rails:.2f}, "
          f"{time.time()-t0:.0f}s)", flush=True)
    return group, N, seed, score, pulses


if __name__ == "__main__":
    jobs = []
    for N in (10, 100, 400):
        jobs += [("delta-realistic", N, s, dict(CS.REALISTIC)) for s in (0, 1, 2)]
    jobs += [("delta-ideal", 100, s, {}) for s in (0, 1, 2)]
    print(f"{len(jobs)} treinos delta-write", flush=True)
    with ProcessPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(run_delta, jobs))
    print("\n" + "=" * 68, flush=True)
    print("referências: ideal contínuo 0.5779±0.0045 | realista contínuo "
          "1.9438±0.1406 (417M impulsos)", flush=True)
    out = {}
    for group, N, seed, score, pulses in results:
        out.setdefault((group, N), []).append((score, pulses))
    for (group, N), v in sorted(out.items()):
        sc = np.array([x[0] for x in v]); pu = np.mean([x[1] for x in v])
        print(f"{group:<16} N={N:<4} {sc.mean():.4f} ± {sc.std():.4f}  "
              f"({pu/1e6:.1f}M impulsos)", flush=True)
    json.dump({f"{g}_N{n}": [x[0] for x in v]
               for (g, n), v in out.items()},
              open("runs/crossbar_study/deltawrite.json", "w"), indent=1)
