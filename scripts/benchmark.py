#!/usr/bin/env python3
"""Compare the predictive network against what actually competes in this niche.

    .venv/bin/python scripts/benchmark.py --data data/ETTm1.csv --column OT
    .venv/bin/python scripts/benchmark.py --wav data/Funk.wav
    .venv/bin/python scripts/benchmark.py --toy

Not against an LLM: an LLM does not solve this problem and this model does not
solve the LLM's. Against the things that solve *this* problem - persistence,
linear AR, MLP, GRU and a small causal transformer - all with a comparable
parameter budget and all under the same protocol:

  **Streaming, one step ahead.** At time t the model sees everything up to t
  and has to produce frame t+1. Nobody sees the future, nobody trains on the
  test set, and normalization uses only statistics from the training segment.

Three cost columns, because a single one lies:

  * **MACs/trama** - the cost on a digital processor. Here the predictive
    network loses, and loses by a large margin: settling iteratively makes
    several passes where the others make one. Better to say it before someone
    discovers it for us.
  * **ADC conversions per frame** - the cost on an analog crossbar, where the
    multiplications are done by physics (Ohm's law) and what you pay for is
    converting results to the digital domain. For the other models it is a
    constant: every output of every layer has to be converted. For the
    predictive network it is measured, and only the error that crosses the
    threshold counts.
  * **parameters** - how much memory has to be kept inside.

It is the second column that carries the whole thesis. If the predictive
network does not win there, it wins nowhere, because in digital MACs it has
already lost.

A warning worth repeating: wall-clock time on the Mac is deliberately not given
as a metric. Here sparsity speeds nothing up and it would measure the wrong
machine (section 6 of CONTEXTO.md). What is compared is accuracy and operations.
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

WARMUP = 32  # context frames given to everyone before scoring


# ---------------------------------------------------------------------------
# metrics, computed exactly the same way for everyone
# ---------------------------------------------------------------------------
def nrmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error normalized by the target's energy, aggregated.

    Aggregated, not averaged per frame: a nearly silent frame has a tiny RMS
    and would blow up an average of ratios. This number is robust and is the
    one the forecasting literature uses.
    """
    err = float(np.mean(np.square(pred - target)))
    ref = float(np.mean(np.square(target)))
    return float(np.sqrt(err / ref)) if ref > 0 else float("inf")


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


# ---------------------------------------------------------------------------
# the models. All expose fit(train) and stream(context, test) -> (preds, macs)
# ---------------------------------------------------------------------------
class Persistence:
    name = "persistence (repeat previous)"
    n_params = 0

    def fit(self, train):
        pass

    adc = 0.0

    def stream(self, context, test):
        prev = np.concatenate([context[-1:], test[:-1]])
        return prev, 0.0


class RidgeAR:
    """Linear regression of the next frame on the k previous ones. Closed form.

    The classic baseline most people underestimate: for signals with linear
    structure it is very hard to beat, and it costs one matrix-vector
    multiplication.
    """

    def __init__(self, order=4, ridge=1e-3):
        self.order, self.ridge = order, ridge
        self.name = f"linear AR (order {order})"

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
        # one conversion per output: the crossbar produces currents, the ADC reads them
        self.adc = float(self.Wt.shape[1])
        return np.array(preds, dtype=F), macs


class PCNetModel:
    """Ours. Online, learns during training and then freezes."""

    def __init__(self, label="predictive network", epochs=60, **cfg_kw):
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
        for frame in context:  # warms up the state, without scoring
            net.step(frame, learn=False)
        preds, traces = [], []
        for frame in test:
            preds.append(net.predict_next())
            traces.append(net.step(frame, learn=False))
        macs = float(np.mean([t.macs_down + t.macs_up for t in traces]))
        # only the error that crosses the threshold is converted - this is the
        # number the architecture promises to reduce, and it is measured, not estimated
        self.adc = float(np.mean([t.adc_conversions for t in traces]))
        self.last_traces = traces
        return np.array(preds, dtype=F), macs


# ---------------------------------------------------------------------------
# torch baselines
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
            # causal transformer
            return self.net(seq)

        def stream(self, context, test):
            seq = torch.tensor(np.concatenate([context, test]), dtype=torch.float32,
                               device=device)
            self.net.eval()
            with torch.no_grad():
                pred = self._forward(seq[:-1])
            return pred[len(context) - 1 :].cpu().numpy().astype(F), self.macs

    # -- sizes chosen to land close to the parameter budget --
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
        """One causal block: single-head attention + MLP, with learned positions."""

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
            # finite causal window, like a sliding context
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

    # ADC conversions: one for each analog output that has to become digital
    return [
        Wrapper("MLP", mlp, "mlp", ctx_mlp, mlp_macs, hm + frame_len),
        Wrapper("GRU", gru, "gru", 1, gru_macs, 4 * h + frame_len),
        Wrapper(f"causal transformer (d={dm}, ctx={ctx_tf})", tf, "tf", ctx_tf,
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
        except Exception as exc:  # pragma: no cover - diagnostics
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
    ap.add_argument("--data", type=Path, help="CSV with a header")
    ap.add_argument("--column", default="OT")
    ap.add_argument("--wav", type=Path)
    ap.add_argument("--har", type=Path, help="root of the 'UCI HAR Dataset'")
    ap.add_argument("--signal", default="total_acc_x", help="HAR inertial signal")
    ap.add_argument("--toy", action="store_true")
    ap.add_argument("--frame-len", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--budget", type=int, default=3000, help="parameter budget")
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
    print(f"parameter budget ~{args.budget}, {args.seeds} seeds\n")

    models = [
        Persistence(),
        RidgeAR(order=4),
        PCNetModel("predictive network (step 1)", epochs=args.epochs),
        PCNetModel("predictive network (+ fast path)", epochs=args.epochs,
                   fast_path=True),
        PCNetModel("predictive network (+ fast path + precision)", epochs=args.epochs,
                   fast_path=True, use_precision=True),
    ]
    try:
        import torch  # noqa: F401

        models += torch_models(ds.frame_len, args.budget, "cpu")
    except ImportError:
        print("  (torch not available, no MLP/GRU/transformer)\n")

    rule(f"{ds.name} - next-frame prediction, streaming")
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
    print("\n  NRMSE aggregated over the test segment; 1.0 = as good as predicting zero.")
    print("  MACs/trama = cost on a digital processor (the predictive network loses")
    print("  here: it settles over several passes where the others make one).")
    print("  ADC/trama = cost on an analog crossbar, where the multiplications are")
    print("  done by physics and what you pay for is converting. For the predictive")
    print("  network it is measured and only the error above the threshold counts; for")
    print("  the others, every output of every layer has to be converted. This is the")
    print("  column that carries the thesis.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"dataset": ds.describe(), "results": results},
                                       indent=2, default=float))
        print(f"\n  {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
