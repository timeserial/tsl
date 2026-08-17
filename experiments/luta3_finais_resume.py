#!/usr/bin/env python3
"""FIGHT 3 - resume of the FINALS after the process died midway.

The selection phase ran to completion under the pre-registered protocol
(luta3_dados_virgens.py, commit 87c4bab) and is recorded in
experiments/luta3_finais.txt: winners us top=16 | adv GRU h16
lr=1e-3 | NLMS mu=0.1 | ESN (leak 0.3, rho 0.8). This script runs
ONLY the finals with those winners, changing nothing in the protocol:
20 seeds per side, test set used only here, decision rule bootstrap
95% CI + Welch + Cohen's d, as declared.
"""
import sys
sys.path.insert(0, "experiments")
import numpy as np
from luta3_dados_virgens import (train_ours, eval_ours, train_adv, eval_adv,
                                 eval_persistence, eval_nlms_ar, eval_esn,
                                 stats_block, mix_full, test_pairs, DOM)

TOP, ADV, MU, ESN_CFG = 16, ("gru", 16, 1e-3), 0.1, (0.3, 0.8)

if __name__ == "__main__":
    print("LUTA 3 - FINAIS (retoma; seleção registada em luta3_finais.txt)",
          flush=True)
    print(f"vencedores: nós topo={TOP} | adv {ADV} | NLMS mu={MU} | "
          f"ESN {ESN_CFG}\n", flush=True)
    ours = []
    for sd in range(20):
        ours.append(eval_ours(train_ours(TOP, mix_full, sd, 400), test_pairs))
        print(f"  nós seed={sd}: {ours[-1]:.4f}", flush=True)
    cell, h, lr = ADV
    adv = []
    for sd in range(20):
        adv.append(eval_adv(train_adv(cell, h, lr, mix_full, sd), test_pairs))
        print(f"  adv seed={sd}: {adv[-1]:.4f}", flush=True)
    pers = eval_persistence(test_pairs)
    nlms = eval_nlms_ar(MU, test_pairs)
    esn = [eval_esn(*ESN_CFG, sd, test_pairs) for sd in range(20)]
    print(f"\n  persistência: {pers:.4f}", flush=True)
    print(f"  NLMS-AR (mu={MU}): {nlms:.4f}", flush=True)
    print(f"  ESN: {np.mean(esn):.4f} ± {np.std(esn, ddof=1):.4f}", flush=True)
    print("\n=== VEREDICTOS (regra: IC bootstrap 95% da diferença) ===",
          flush=True)
    stats_block(ours, adv, "NÓS", "ADV")
    stats_block(ours, esn, "NÓS", "ESN")
    print(f"  NÓS vs NLMS-AR ({nlms:.4f}) e persistência ({pers:.4f}): "
          f"determinísticos - comparar com a média e o IC de NÓS.", flush=True)
