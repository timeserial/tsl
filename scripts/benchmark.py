#!/usr/bin/env python3
"""Comparar a rede preditiva com o que realmente compete neste nicho.

    .venv/bin/python scripts/benchmark.py --data data/ETTm1.csv --column OT
    .venv/bin/python scripts/benchmark.py --wav data/Funk.wav
    .venv/bin/python scripts/benchmark.py --toy

Não com um LLM: um LLM não resolve este problema e este modelo não resolve o
do LLM. Com as coisas que resolvem *este* problema — persistência, AR linear,
MLP, GRU e um transformer causal pequeno — todas com orçamento de parâmetros
comparável e todas no mesmo protocolo:

  **Streaming, um passo à frente.** No instante t o modelo vê tudo até t e
  tem de dizer a trama t+1. Ninguém vê o futuro, ninguém treina no teste, e a
  normalização usa só estatísticas do troço de treino.

Três colunas de custo, porque uma só mente:

  * **MACs/trama** — o custo num processador digital. Aqui a rede preditiva
    perde, e perde por uma margem grande: assentar iterativamente faz várias
    passagens onde os outros fazem uma. Convém dizê-lo antes de alguém o
    descobrir por nós.
  * **conversões ADC/trama** — o custo num crossbar analógico, onde as
    multiplicações são feitas pela física (lei de Ohm) e o que se paga é
    converter resultados para o domínio digital. Para os outros modelos é
    uma constante: toda a saída de cada camada tem de ser convertida. Para a
    rede preditiva é medido, e só conta o erro que passa o limiar.
  * **parâmetros** — quanta memória é preciso ter lá dentro.

É a segunda coluna que carrega a tese inteira. Se a rede preditiva não ganhar
aí, não ganha em lado nenhum, porque em MACs digitais já perdeu.

Aviso que vale a pena repetir: o tempo de relógio no Mac não é dado como
métrica de propósito. Aqui a esparsidade não acelera nada e mediria a máquina
errada (secção 6 do CONTEXTO.md). O que se compara é exatidão e operações.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pcnet import PCConfig, PCNetwork  # noqa: E402
from pcnet.datasets import (  # noqa: E402
    Dataset, load_csv_column, load_toy, load_uci_har_inertial, load_wav,
)
from pcnet.dtypes import F  # noqa: E402
from pcnet.report import pm, rule, table  # noqa: E402
from pcnet.train import train as pc_train  # noqa: E402

WARMUP = 32  # tramas de contexto dadas a toda a gente antes de pontuar


# ---------------------------------------------------------------------------
# métricas, calculadas exatamente da mesma maneira para todos
# ---------------------------------------------------------------------------
def nrmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Erro quadrático médio normalizado pela energia do alvo, agregado.

    Agregado e não média por trama: uma trama quase silenciosa tem RMS
    minúsculo e faria explodir uma média de rácios. Este número é robusto e é
    o que a literatura de forecasting usa.
    """
    err = float(np.mean(np.square(pred - target)))
    ref = float(np.mean(np.square(target)))
    return float(np.sqrt(err / ref)) if ref > 0 else float("inf")


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


# ---------------------------------------------------------------------------
# os modelos. Todos expõem fit(train) e stream(context, test) -> (preds, macs)
# ---------------------------------------------------------------------------
class Persistence:
    name = "persistência (repetir a anterior)"
    n_params = 0

    def fit(self, train):
        pass

    adc = 0.0

    def stream(self, context, test):
        prev = np.concatenate([context[-1:], test[:-1]])
        return prev, 0.0


class RidgeAR:
    """Regressão linear da próxima trama nas k anteriores. Forma fechada.

    A linha de base clássica que mais gente subestima: para sinais com
    estrutura linear é muito difícil de bater, e custa uma multiplicação
    matriz-vetor.
    """

    def __init__(self, order=4, ridge=1e-3):
        self.order, self.ridge = order, ridge
        self.name = f"AR linear (ordem {order})"

    def fit(self, train):
        k, D = self.order, train.shape[1]
        X = np.stack([train[i - k : i].ravel() for i in range(k, len(train))])
        Y = train[k:]
        X = np.hstack([X, np.ones((len(X), 1))])
        A = X.T @ X + self.ridge * len(X) * np.eye(X.shape[1])
        self.Wt = np.linalg.solve(A, X.T @ Y)
        self.n_params = int(self.Wt.size)

    def stream(self, context, test):
        k = self.order
        seq = np.concatenate([context, test])
        start = len(context)
        preds = []
        for i in range(start, len(seq)):
            feats = np.concatenate([seq[i - k : i].ravel(), [1.0]])
            preds.append(feats @ self.Wt)
        macs = float(self.Wt.shape[0] * self.Wt.shape[1])
        # uma conversão por saída: o crossbar produz correntes, o ADC lê-as
        self.adc = float(self.Wt.shape[1])
        return np.array(preds, dtype=F), macs


class PCNetModel:
    """A nossa. Online, aprende durante o treino e depois congela."""

    def __init__(self, label="rede preditiva", epochs=60, **cfg_kw):
        self.name = label
        self.epochs = epochs
        self.cfg_kw = cfg_kw
        self.net = None

    def fit(self, train):
        cfg = PCConfig(sizes=self._sizes(train.shape[1]), **self.cfg_kw)
        self.net = PCNetwork(cfg)
        pc_train(self.net, train, epochs=self.epochs)
        self.n_params = (
            sum(W.size for W in self.net.weights)
            + self.net.A.size
            + sum(t.n_params for t in self.net.transitions if t is not None)
            + (self.net.A0.size if self.net.A0 is not None else 0)
        )

    @staticmethod
    def _sizes(frame_len):
        sizes, n = [frame_len], frame_len
        while n > 8 and len(sizes) < 4:
            n //= 2
            sizes.append(n)
        return tuple(sizes)

    def stream(self, context, test):
        net = self.net
        net.reset()
        for frame in context:  # aquece o estado, sem pontuar
            net.step(frame, learn=False)
        preds, traces = [], []
        for frame in test:
            preds.append(net.predict_next())
            traces.append(net.step(frame, learn=False))
        macs = float(np.mean([t.macs_down + t.macs_up for t in traces]))
        # só o erro que passa o limiar é convertido — é este o número que a
        # arquitetura promete reduzir, e é medido, não estimado
        self.adc = float(np.mean([t.adc_conversions for t in traces]))
        self.last_traces = traces
        return np.array(preds, dtype=F), macs


# ---------------------------------------------------------------------------
# baselines em torch
# ---------------------------------------------------------------------------
def torch_models(frame_len: int, budget: int, device: str):
    import torch
    import torch.nn as nn

    class Wrapper:
        def __init__(self, name, module, kind, context=16, macs=0.0, adc=0.0):
            self.name, self.net, self.kind = name, module, kind
            self.context, self.macs, self.adc = context, macs, adc
            self.n_params = sum(p.numel() for p in module.parameters())

        def fit(self, train, epochs=200, lr=3e-3, seed=0):
            torch.manual_seed(seed)
            x = torch.tensor(np.asarray(train), dtype=torch.float32, device=device)
            opt = torch.optim.Adam(self.net.parameters(), lr=lr)
            n_val = max(8, len(x) // 10)
            xtr, xva = x[:-n_val], x[-n_val - self.context :]
            best, best_state, patience = float("inf"), None, 0
            for _ in range(epochs):
                self.net.train()
                loss = self._loss(xtr)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                self.net.eval()
                with torch.no_grad():
                    v = float(self._loss(xva))
                if v < best - 1e-5:
                    best, patience = v, 0
                    best_state = {k: t.clone() for k, t in self.net.state_dict().items()}
                else:
                    patience += 1
                    if patience > 30:
                        break
            if best_state is not None:
                self.net.load_state_dict(best_state)

        def _loss(self, seq):
            pred = self._forward(seq[:-1])
            return ((pred - seq[1:]) ** 2).mean()

        def _forward(self, seq):
            if self.kind == "gru":
                out, _ = self.net.rnn(seq.unsqueeze(0))
                return self.net.head(out.squeeze(0))
            if self.kind == "mlp":
                k = self.context
                pad = seq[:1].repeat(k - 1, 1)
                padded = torch.cat([pad, seq])
                feats = torch.stack([padded[i : i + k].reshape(-1)
                                     for i in range(len(seq))])
                return self.net(feats)
            # transformer causal
            return self.net(seq)

        def stream(self, context, test):
            seq = torch.tensor(np.concatenate([context, test]), dtype=torch.float32,
                               device=device)
            self.net.eval()
            with torch.no_grad():
                pred = self._forward(seq[:-1])
            return pred[len(context) - 1 :].cpu().numpy().astype(F), self.macs

    # -- tamanhos escolhidos para ficarem perto do orçamento de parâmetros --
    h = max(4, int((-frame_len + np.sqrt(frame_len**2 + 4 * budget / 4)) / 4))
    gru = nn.Module()
    gru.rnn = nn.GRU(frame_len, h, batch_first=True)
    gru.head = nn.Linear(h, frame_len)
    gru.to(device)
    gru_macs = 3 * h * (frame_len + h) + h * frame_len

    ctx_mlp = 4
    hm = max(4, budget // (ctx_mlp * frame_len + frame_len))
    mlp = nn.Sequential(
        nn.Linear(ctx_mlp * frame_len, hm), nn.Tanh(), nn.Linear(hm, frame_len)
    ).to(device)
    mlp_macs = ctx_mlp * frame_len * hm + hm * frame_len

    class TinyTransformer(nn.Module):
        """Um bloco causal: atenção de cabeça única + MLP, com posição aprendida."""

        def __init__(self, d_in, d_model, context):
            super().__init__()
            self.context = context
            self.inp = nn.Linear(d_in, d_model)
            self.pos = nn.Parameter(torch.zeros(context, d_model))
            self.q = nn.Linear(d_model, d_model, bias=False)
            self.k = nn.Linear(d_model, d_model, bias=False)
            self.v = nn.Linear(d_model, d_model, bias=False)
            self.ff = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(),
                                    nn.Linear(2 * d_model, d_model))
            self.n1 = nn.LayerNorm(d_model)
            self.n2 = nn.LayerNorm(d_model)
            self.out = nn.Linear(d_model, d_in)
            self.d = d_model

        def forward(self, seq):
            T = seq.shape[0]
            x = self.inp(seq)
            x = x + self.pos[: min(T, self.context)].repeat(
                (T + self.context - 1) // self.context, 1
            )[:T]
            hn = self.n1(x)
            att = self.q(hn) @ self.k(hn).T / np.sqrt(self.d)
            mask = torch.full((T, T), float("-inf"), device=seq.device).triu(1)
            # janela causal finita, como um contexto deslizante
            band = torch.ones(T, T, device=seq.device).tril(0).triu(-self.context + 1)
            att = att + mask
            att = att.masked_fill(band == 0, float("-inf"))
            x = x + torch.softmax(att, dim=-1) @ self.v(hn)
            x = x + self.ff(self.n2(x))
            return self.out(x)

    ctx_tf = 16
    dm = 8
    while True:
        cand = TinyTransformer(frame_len, dm + 4, ctx_tf)
        if sum(p.numel() for p in cand.parameters()) > budget * 1.25 or dm > 96:
            break
        dm += 4
    tf = TinyTransformer(frame_len, dm, ctx_tf).to(device)
    tf_macs = (2 * frame_len * dm) + 3 * dm * dm + ctx_tf * dm * 2 + 4 * dm * dm

    # conversões ADC: uma por cada saída analógica que tem de virar digital
    return [
        Wrapper("MLP", mlp, "mlp", ctx_mlp, mlp_macs, hm + frame_len),
        Wrapper("GRU", gru, "gru", 1, gru_macs, 4 * h + frame_len),
        Wrapper(f"transformer causal (d={dm}, ctx={ctx_tf})", tf, "tf", ctx_tf,
                tf_macs, 6 * dm + frame_len),
    ]


# ---------------------------------------------------------------------------
def evaluate_model(model, ds: Dataset, seeds=(0,)) -> dict:
    context = ds.train[-WARMUP:]
    scores, maes, macs, secs = [], [], [], []
    for seed in seeds:
        t0 = time.time()
        try:
            model.fit(ds.train, seed=seed) if _takes_seed(model) else model.fit(ds.train)
            pred, m = model.stream(context, ds.test)
        except Exception as exc:  # pragma: no cover - diagnóstico
            return {"name": model.name, "erro": f"{type(exc).__name__}: {exc}"}
        scores.append(nrmse(pred, ds.test))
        maes.append(mae(pred, ds.test))
        macs.append(m)
        secs.append(time.time() - t0)
    return {
        "name": model.name,
        "params": getattr(model, "n_params", 0),
        "nrmse": (float(np.mean(scores)), float(np.std(scores))),
        "mae": float(np.mean(maes)),
        "macs": float(np.mean(macs)),
        "adc": float(getattr(model, "adc", 0.0)),
        "train_s": float(np.mean(secs)),
    }


def _takes_seed(model) -> bool:
    import inspect

    try:
        return "seed" in inspect.signature(model.fit).parameters
    except (TypeError, ValueError):
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, help="CSV com cabeçalho")
    ap.add_argument("--column", default="OT")
    ap.add_argument("--wav", type=Path)
    ap.add_argument("--har", type=Path, help="raiz do 'UCI HAR Dataset'")
    ap.add_argument("--signal", default="total_acc_x", help="sinal inercial do HAR")
    ap.add_argument("--toy", action="store_true")
    ap.add_argument("--frame-len", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--budget", type=int, default=3000, help="orçamento de parâmetros")
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
    print(ds.describe())
    print(f"orçamento de parâmetros ~{args.budget}, {args.seeds} seeds\n")

    models = [
        Persistence(),
        RidgeAR(order=4),
        PCNetModel("rede preditiva (passo 1)", epochs=args.epochs),
        PCNetModel("rede preditiva (+ via rápida)", epochs=args.epochs,
                   fast_path=True),
        PCNetModel("rede preditiva (+ via rápida + precisão)", epochs=args.epochs,
                   fast_path=True, use_precision=True),
    ]
    try:
        import torch  # noqa: F401

        models += torch_models(ds.frame_len, args.budget, "cpu")
    except ImportError:
        print("  (torch não disponível — sem MLP/GRU/transformer)\n")

    rule(f"{ds.name} — previsão da trama seguinte, streaming")
    rows, results = [], []
    for model in models:
        r = evaluate_model(model, ds, seeds)
        results.append(r)
        if "erro" in r:
            rows.append({"modelo": r["name"], "NRMSE": r["erro"]})
            continue
        rows.append({
            "modelo": r["name"],
            "NRMSE": pm(*r["nrmse"]),
            "MAE": f"{r['mae']:.3f}",
            "parâmetros": r["params"],
            "MACs/trama": f"{r['macs']:,.0f}",
            "ADC/trama": f"{r['adc']:,.0f}",
        })
        print(f"  … {r['name']}")
    print()
    table(rows, ["modelo", "NRMSE", "MAE", "parâmetros", "MACs/trama", "ADC/trama"])
    print("\n  NRMSE agregado sobre o troço de teste; 1.0 = tão bom como prever zero.")
    print("  MACs/trama = custo num processador digital (a rede preditiva perde aqui:")
    print("  assenta em várias passagens onde os outros fazem uma).")
    print("  ADC/trama = custo num crossbar analógico, onde as multiplicações são")
    print("  feitas pela física e o que se paga é converter. Para a rede preditiva é")
    print("  medido e só conta o erro acima do limiar; para os outros, toda a saída")
    print("  de cada camada tem de ser convertida. É esta coluna que carrega a tese.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"dataset": ds.describe(), "results": results},
                                       indent=2, default=float))
        print(f"\n  {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
