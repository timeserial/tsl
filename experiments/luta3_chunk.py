#!/usr/bin/env python3
"""LUTA 3 — executor em fatias (os processos longos morrem nesta máquina).

Mesmo protocolo pré-registado (87c4bab), mesma retoma (6c1ed5b); cada
invocação faz um punhado de seeds e sai, appendando ao ficheiro de
resultados. --verdict lê o ficheiro completo e aplica a regra declarada.

  python -u experiments/luta3_chunk.py ours 11 12 13
  python -u experiments/luta3_chunk.py adv 0 1 ... 19
  python -u experiments/luta3_chunk.py esn 0 1 ... 19
  python -u experiments/luta3_chunk.py det        # persistência + NLMS
  python -u experiments/luta3_chunk.py verdict
"""
import sys, re
sys.path.insert(0, "experiments")
import numpy as np

OUT = "experiments/luta3_finais_resume.txt"
TOP, ADV, MU, ESN_CFG = 16, ("gru", 16, 1e-3), 0.1, (0.3, 0.8)

def log(line):
    print(line, flush=True)
    with open(OUT, "a") as f:
        f.write(line + "\n")

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "verdict":
        txt = open(OUT).read()
        ours = [float(v) for v in re.findall(r"nós seed=\d+: ([0-9.]+)", txt)]
        adv = [float(v) for v in re.findall(r"adv seed=\d+: ([0-9.]+)", txt)]
        esn = [float(v) for v in re.findall(r"esn seed=\d+: ([0-9.]+)", txt)]
        pers = float(re.findall(r"persistência: ([0-9.]+)", txt)[-1])
        nlms = float(re.findall(r"NLMS-AR[^:]*: ([0-9.]+)", txt)[-1])
        assert len(ours) == 20 and len(adv) == 20 and len(esn) == 20, \
            (len(ours), len(adv), len(esn))
        from luta3_dados_virgens import stats_block
        log("\n=== VEREDICTOS (regra: IC bootstrap 95% da diferença) ===")
        import io, contextlib
        for a, b, la, lb in [(ours, adv, "NÓS", "ADV"),
                             (ours, esn, "NÓS", "ESN")]:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                stats_block(a, b, la, lb)
            for ln in buf.getvalue().strip().splitlines():
                log(ln)
        log(f"  NÓS vs NLMS-AR ({nlms:.4f}) e persistência ({pers:.4f}): "
            f"determinísticos — comparar com a média e o IC de NÓS.")
        sys.exit(0)

    from luta3_dados_virgens import (train_ours, eval_ours, train_adv,
                                     eval_adv, eval_persistence,
                                     eval_nlms_ar, eval_esn, mix_full,
                                     test_pairs)
    if mode == "ours":
        for sd in map(int, sys.argv[2:]):
            v = eval_ours(train_ours(TOP, mix_full, sd, 400), test_pairs)
            log(f"  nós seed={sd}: {v:.4f}")
    elif mode == "adv":
        cell, h, lr = ADV
        for sd in map(int, sys.argv[2:]):
            v = eval_adv(train_adv(cell, h, lr, mix_full, sd), test_pairs)
            log(f"  adv seed={sd}: {v:.4f}")
    elif mode == "esn":
        for sd in map(int, sys.argv[2:]):
            v = eval_esn(*ESN_CFG, sd, test_pairs)
            log(f"  esn seed={sd}: {v:.4f}")
    elif mode == "det":
        log(f"  persistência: {eval_persistence(test_pairs):.4f}")
        log(f"  NLMS-AR (mu={MU}): {eval_nlms_ar(MU, test_pairs):.4f}")
