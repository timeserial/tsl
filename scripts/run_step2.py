#!/usr/bin/env python3
"""Passo 2 do plano: ternarização e ruído de dispositivo.

    python3 scripts/run_step2.py [--epochs 60] [--seeds 3] [--out runs/step2]

A pergunta é uma: quanto é que a rede perde quando os pesos deixam de ser
float32 e passam a ser dispositivos físicos imperfeitos? E, mais importante
para o argumento, *por que mecanismo* é que perde.

Nada aqui é reportado sem dispersão sobre seeds. Um dispositivo analógico é
uma amostra de uma distribuição — medir um e chamar-lhe resultado seria medir
a sorte.
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
    """Treina e guarda: cada configuração é treinada uma vez só."""

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
    """Avalia redes já treinadas num crossbar diferente daquele em que treinaram.

    Repõe o dispositivo original no fim: as redes ficam na cache e vão ser
    reutilizadas por outras secções.
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

    print(f"hierarquia   {' -> '.join(str(n) for n in cfg.sizes)}")
    print(f"tramas       {tr.n_frames} treino / {te.n_frames} teste")
    print(f"seeds        {args.seeds} (rede e dispositivo movem-se juntos)")

    TERNARY = DeviceModel(ternary=True)

    # ------------------------------------------------------------------
    rule("1. Quantização: float32 -> {-1, 0, 1}")
    float_nets = lab.nets(None)
    float_nrmse = stat(evaluate(n, te.frames).pred_nrmse for n in float_nets)
    ptq = stat(s.pred_nrmse for s in on_device(float_nets, TERNARY, te.frames))
    qat_nets = lab.nets(TERNARY)
    qat = stat(evaluate(n, te.frames).pred_nrmse for n in qat_nets)
    zeros = float(np.mean([
        np.mean(ternarize(W)[0] == 0) for n in float_nets for W in n.weights
    ]))

    table([
        {"pesos": "float32", "NRMSE": pm(*float_nrmse), "vs float": "—"},
        {"pesos": "ternário, quantizado no fim (PTQ)", "NRMSE": pm(*ptq),
         "vs float": f"{ptq[0] / float_nrmse[0]:.2f}×"},
        {"pesos": "ternário, treinado no crossbar (QAT)", "NRMSE": pm(*qat),
         "vs float": f"{qat[0] / float_nrmse[0]:.2f}×"},
    ], ["pesos", "NRMSE", "vs float"])
    print(f"  {100 * zeros:.0f}% dos pesos ternários ficam a zero: dispositivos que não")
    print("  é preciso programar nem ler.")
    print("  Treinar com o quantizador no laço é trivial aqui — a regra é local, não")
    print("  há gradiente para passar através dele; o peso float é o shadow weight.")
    results["quantization"] = {"float": float_nrmse, "ptq": ptq, "qat": qat,
                               "ternary_zero_frac": zeros}

    # ------------------------------------------------------------------
    rule("2. Variabilidade de programação: treinar no dispositivo ou não")
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
            if sigma else "—",
        })
    table(rows)
    print("  σ_rel é o desvio relativo de cada dispositivo face à condutância pedida,")
    print("  amostrado uma vez na programação e fixo a partir daí. Não se dilui com")
    print("  iterações — é isto que testa a tolerância, não o ruído de leitura.")
    print("  Variabilidade real de ReRAM/PCM anda nos 5-20%; acima disso a coluna da")
    print("  esquerda é ficção e a da direita é o que interessa.")
    results["programming_variability"] = rows

    # ------------------------------------------------------------------
    rule("3. Auto-correção: o que o ciclo fecha e o que não pode fechar")
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
    print("  O ciclo continua a explicar a trama mesmo com um crossbar muito")
    print("  defeituoso — a σ_rel=0.4 a energia em malha aberta dispara mas o")
    print("  assentamento fecha ~95% dela, com os estados limitados.")
    print("  O que se degrada é a *previsão* da trama seguinte, e tem de ser assim:")
    print("  a previsão desce pelos mesmos pesos partidos sem ter ainda erro que a")
    print("  corrija. Auto-correção não é omnisciência — é o que dá o argumento para")
    print("  treinar no dispositivo em vez de treinar limpo e programar depois.")
    results["self_correction"] = rows

    # ------------------------------------------------------------------
    rule("3b. O penhasco: onde cada crossbar deixa de funcionar")
    fine = (0.0, 0.08, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25, 0.30)
    rows = []
    for sigma in fine:
        stats = on_device(qat_nets, replace(TERNARY, sigma_rel=sigma), te.frames)
        row = {"σ_rel": sigma}
        for i, s in enumerate(stats):
            row[f"crossbar {i}"] = f"{s.pred_nrmse:.2f}"
        rows.append(row)
    table(rows)
    print("  Sem retreinar. A degradação é suave até ~15% e depois há crossbars que")
    print("  caem de um penhasco enquanto outros continuam a portar-se bem — a")
    print("  transição depende da amostra, não só de σ. Uma previsão que cai do")
    print("  penhasco fica *descorrelacionada* do alvo, não apenas mal escalada")
    print("  (reescalá-la não recupera nada), com os estados sempre limitados: é uma")
    print("  bifurcação da dinâmica em malha fechada, não saturação nem divergência.")
    print("  Treinar no dispositivo elimina o penhasco (secção 2).")
    results["cliff"] = rows

    # ------------------------------------------------------------------
    rule("4. O passo de assentamento contra o limite de estabilidade")
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
    print(f"  Os pesos ternários têm σ_max muito maior do que os float, e o passo fixo")
    print(f"  z_lr={cfg.z_lr} fica acima do limite 2/(1+σ_max²). Limitar o passo por esse")
    print("  bound (duas leituras do crossbar por atualização) resolve, à custa de")
    print("  mais iterações — que o early exit depois volta a cortar.")
    results["stability"] = rows

    # ------------------------------------------------------------------
    rule("5. Estático vs dinâmico: que defeitos doem mais")
    rows = []
    for label, model in (
        ("nenhum", TERNARY),
        ("variabilidade 20% (estática)", replace(TERNARY, sigma_rel=0.2)),
        ("ruído de leitura 0.05 (dinâmico)", replace(TERNARY, read_sigma=0.05)),
        ("5% dos dispositivos presos a zero", replace(TERNARY, stuck_frac=0.05)),
        ("offset absoluto 0.02", replace(TERNARY, sigma_abs=0.02)),
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
    rule("6. ADC: quantos bits precisa o erro que sobe")
    rows = []
    for bits in (0, 8, 6, 4, 3, 2):
        stats = on_device(
            qat_nets, replace(TERNARY, sigma_rel=0.1, adc_bits=bits), te.frames
        )
        rows.append({"bits": bits or "float",
                     "NRMSE": pm(*stat(s.pred_nrmse for s in stats))})
    table(rows, ["bits", "NRMSE"])
    print("  Com σ_rel=0.1. O ADC é o componente caro do crossbar: o limiar já reduziu")
    print("  o *número* de conversões, isto diz quanto custa cada uma.")
    results["adc"] = rows

    # ------------------------------------------------------------------
    rule("7. O limiar de esparsidade continua a compensar em ternário?")
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
    rule("Artefactos")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"  {args.out / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
