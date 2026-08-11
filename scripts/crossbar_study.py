#!/usr/bin/env python3
"""Estudo de crossbar: que não-idealidades analógicas doem ao TSL, e quanto.

    .venv/bin/python -u scripts/crossbar_study.py

A célula é o tijolo canónico de experiments/profundidade_empilhamento.py
(64→24, transição portada, regra de dois tempos, marco 0.579±0.004 em
3 mundos intercalados, 80 épocas). TODAS as operações de peso — leitura
direta (previsão), leitura transposta (h = Wᵀe) e escrita rank-1 — de W₀, A,
G e b passam pelo modelo de crossbar de src/pcnet/crossbar.py. O estado s e
as não-linearidades tanh/σ ficam digitais (periferia), tal como o limiar θ
dos erros.

Protocolo:
  * portão de sanidade: crossbar ideal tem de reproduzir o marco — o modo
    ideal do Crossbar é um acumulador float32 bit-idêntico ao treino float,
    logo a igualdade é exata por seed;
  * ablação: UMA não-idealidade de cada vez, a 3 magnitudes
    (mild/realistic/pessimistic, justificadas em pcnet/crossbar.py e nas
    tabelas abaixo), 3 seeds × 80 épocas;
  * "dispositivo realista": tudo ligado nos valores do meio, 4 seeds;
  * comporta de escrita θ_w ∈ {0, 1e-3, 5e-3}: NRMSE e impulsos físicos
    (o trade-off endurance/exatidão), sobre crossbar ideal e sobre o
    dispositivo realista.

Escalas físicas fixadas por sondagem do treino float (seed 0, 80 épocas):
máx |W₀|=0.87, |A|=4.1, |G|=1.7, |b|=4.7; leituras diretas de W₀ ≤ 1.0,
transpostas ≤ 12, leituras de A/G ≤ 12; mediana dos |Δw| por elemento
2e-4 (G) a 4e-3 (b). Daí os w_max (fundo de escala de peso por array, com
folga para outras seeds) e os fundos de escala dos ADC por sentido.
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# matrizes pequenas: BLAS multi-thread só atrapalha 7 processos em paralelo
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
for p in ("src", "scripts", "experiments"):
    sys.path.insert(0, str(ROOT / p))

import profundidade_empilhamento as PE  # noqa: E402  (constrói TASKS/MIXED)
from pcnet import PCConfig, PCNetwork  # noqa: E402
from pcnet.crossbar import Crossbar, CrossbarModel  # noqa: E402
from pcnet.dtypes import F  # noqa: E402

EPOCHS = 80
MARK = 0.579  # o marco publicado do tijolo (0.579 ± 0.004, 4 seeds)

# fundo de escala de peso por array (sondado + folga ~30-40%) e fundos de
# escala dos ADC por array e sentido (potências de 2 acima do sinal medido).
W_MAX = {"W0": 1.5, "A": 5.0, "G": 2.5, "b": 6.0}
ADC_FS = {"W0": (2.0, 16.0), "A": (16.0, 16.0), "G": (16.0, 16.0)}


# ---------------------------------------------------------------------------
# ligação rede ↔ crossbars
# ---------------------------------------------------------------------------
class XbarLayer:
    """PCLayer com as três operações de peso a passar pelo crossbar.

    Delega buffers e σ_max no PCLayer original para que o caminho de
    assentamento (passo adaptativo incluído) seja o canónico.
    """

    def __init__(self, base, xw: Crossbar) -> None:
        self.base = base
        self.xw = xw
        # aponta o operador efetivo para o do crossbar SEM re-estimar σ_max:
        # os valores são os mesmos e o warm-start da iteração de potência
        # tem de seguir a mesma trajetória do treino float.
        base.W_eff = xw.W_eff

    def predict(self, z_above):
        b = self.base
        b.a[:] = self.xw.read(z_above)          # leitura direta (previsão)
        b._base = b._f(b.a).astype(F, copy=False)
        b.zhat = b._base
        return b.zhat

    def modulated_error(self, eps_below):
        return self.base.modulated_error(eps_below)

    def backward(self, eps_mod, eps_raw=None):
        return self.xw.read_T(eps_mod)          # leitura transposta

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
    """GatedTransition com A, G e b em crossbars (b = coluna própria, lida
    com entrada 1; tanh e σ são periferia digital)."""

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
    """A regra de dois tempos, espelho exato da do tijolo, com cada operação
    de peso no crossbar: 4 leituras (A, G, b, W₀ direta), 1 leitura
    transposta (h) e 4 escritas rank-1 por trama."""
    gx = net.gated
    xw = net.layers[0].xw
    c = np.tanh(gx.xA.read(s_prev))
    g = PE._sigm(gx.xG.read(s_prev) + gx.xb.read(gx._one))
    prior = ((1.0 - g) * s_prev + g * c).astype(F)

    e = (target - xw.read(prior)).astype(F)
    h = xw.read_T(e)                       # retroprojeção: pré-atualização

    xw.update((lr / (float(prior @ prior) + 1e-6))
              * np.outer(e, prior).astype(F))
    # canónico: re-estimar σ_max depois de mexer em W₀ (passo adaptativo)
    net.layers[0].refresh_device()
    ns = float(s_prev @ s_prev) + 1e-6
    mod_g = (h * (c - s_prev) * g * (1.0 - g)).astype(F)
    gx.xA.update((lr / ns) * np.outer((h * g * (1.0 - c * c)).astype(F), s_prev))
    gx.xG.update((lr / ns) * np.outer(mod_g, s_prev))
    gx.xb.update((F(lr) * mod_g)[:, None])


# ---------------------------------------------------------------------------
# uma célula da grelha
# ---------------------------------------------------------------------------
def run_cell(job):
    """job = (grupo, magnitude, seed, kwargs do CrossbarModel)."""
    group, mag, seed, kwargs = job
    t0 = time.time()
    model = CrossbarModel(**kwargs)
    net = make_xbar_net(seed, model)
    L = net.L
    for _ in range(EPOCHS):
        for fr in PE.MIXED:
            s_prev = net._z_prev[L].copy()
            net.step(fr, learn=False)          # assentamento (só inferência)
            two_time_update_xbar(net, np.asarray(fr, dtype=F), s_prev)
    if model.drift_nu > 0.0:                   # deriva entre treino e teste
        for xb in _xbars(net):
            xb.apply_drift()
        net.layers[0].refresh_device()         # σ_max acompanha a deriva
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
# grelha
# ---------------------------------------------------------------------------
# Magnitudes mild/realistic/pessimistic — fundamentação nos comentários de
# pcnet/crossbar.py (gamas da literatura citadas lá):
#   σ_write   1% / 5% / 20% da gama       (c2c típico 0.5–5%; 20% = mau)
#   α (p,d)   0.5,0.5 / 1,2 / 2,5         (quase linear / TaOx-HfOx / severo)
#   σ_read    1% / 3% / 10%               (PCM calibrado ~1%; RTN forte 10%)
#   ADC       12 / 8 / 6 bits             (transparente / ISAAC / agressivo)
#   ν drift   0.005 / 0.02 / 0.1          (cristalino / célula boa / amorfo)
#   IR γ      2% / 10% / 30%              (arrays pequenos / 64+ / fio pobre)
#   on/off    ∞ / 50 / 10, com σ_read=1%  (sozinho, o par diferencial
#             cancela G_min exatamente — só se vê contra ruído de leitura)
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

# tudo nos valores do meio — o número de cabeçalho
REALISTIC = dict(on_off=50.0, sigma_write=0.05, alpha_p=1.0, alpha_d=2.0,
                 sigma_read=0.03, adc_bits=8, drift_nu=0.02, ir_drop=0.10)

# comporta de endurance: θ_w em unidades de peso; mediana dos |Δw| sondados
# ~2e-4–4e-3, logo 1e-3 poupa ~metade dos impulsos e 5e-3 ~90%.
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
    print(f"{len(jobs)} treinos (80 épocas cada), 7 processos\n", flush=True)
    results: dict[tuple[str, str], list] = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=7) as ex:
        futs = [ex.submit(run_cell, j) for j in jobs]
        for fut in as_completed(futs):
            group, mag, seed, score, agg = fut.result()
            results.setdefault((group, mag), []).append((seed, score, agg))
            print(f"  [{time.time()-t0:5.0f}s] {group:15s} {mag:8s} "
                  f"seed {seed}: {score:.4f}  "
                  f"(imp {agg['pulses_issued']/1e6:.0f}M, "
                  f"poupados {agg['pulses_gated']/1e6:.0f}M, "
                  f"rails {agg['rail_frac_max']:.2f}, {agg['secs']:.0f}s)",
                  flush=True)

    # ---- tabela final ----------------------------------------------------
    base_scores = [sc for _, sc, _ in sorted(results[("ideal", "-")])]
    base = float(np.mean(base_scores))
    print("\n" + "=" * 76)
    print(f"baseline crossbar ideal: {base:.4f} ± {np.std(base_scores):.4f} "
          f"(marco float: {MARK} ± 0.004; igualdade bit-exata por seed)")
    print("=" * 76)
    header = (f"{'configuração':<28s} {'NRMSE':>15s} {'Δ vs ideal':>10s}  "
              f"{'impulsos':>9s}  veredicto")
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
