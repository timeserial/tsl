#!/usr/bin/env python3
"""Ver o modelo a funcionar, no terminal.

    python3 scripts/demo.py                      # sinal de brinquedo
    python3 scripts/demo.py --events 8           # com transitórios (surpresas)
    python3 scripts/demo.py --wav ficheiro.wav   # o teu próprio áudio
    python3 scripts/demo.py --ternary            # pesos {-1,0,1}
    python3 scripts/demo.py --ternary --sigma 0.2  # ... num crossbar defeituoso

Mostra três linhas alinhadas no tempo: o que o sensor deu, o que a rede tinha
previsto *antes* de ver, e a surpresa. Onde as duas primeiras coincidem, a
terceira está silenciosa e a rede quase não gastou nada.
"""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pcnet import PCConfig, PCNetwork, make_signal, persistence_nrmse  # noqa: E402
from pcnet.device import DeviceModel  # noqa: E402
from pcnet.dtypes import F  # noqa: E402
from pcnet.signals import ToySignal, frame_signal  # noqa: E402
from pcnet.train import evaluate, train  # noqa: E402

BLOCKS = "▁▂▃▄▅▆▇█"


def spark(values: np.ndarray, lo: float | None = None, hi: float | None = None) -> str:
    """Uma linha de gráfico em caracteres."""
    v = np.asarray(values, dtype=float)
    lo = float(v.min()) if lo is None else lo
    hi = float(v.max()) if hi is None else hi
    if hi - lo < 1e-12:
        return BLOCKS[0] * len(v)
    idx = np.clip(((v - lo) / (hi - lo) * (len(BLOCKS) - 1)).round(), 0, len(BLOCKS) - 1)
    return "".join(BLOCKS[int(i)] for i in idx)


def load_wav(path: Path, frame_len: int) -> ToySignal:
    """Lê um WAV mono/estéreo de 16 bit e corta-o em tramas."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise SystemExit(f"{path}: só sei ler WAV de 16 bit (este tem "
                             f"{8 * w.getsampwidth()} bit)")
        raw = w.readframes(w.getnframes())
        x = np.frombuffer(raw, dtype="<i2").astype(np.float64)
        if w.getnchannels() > 1:
            x = x.reshape(-1, w.getnchannels()).mean(axis=1)
        rate = w.getframerate()

    peak = np.max(np.abs(x)) or 1.0
    x = (x / peak).astype(F)
    print(f"  {path.name}: {len(x)} amostras a {rate} Hz "
          f"({len(x) / rate:.1f} s, {len(x) // frame_len} tramas de {frame_len})")
    return ToySignal(samples=x, frames=frame_signal(x, frame_len),
                     frame_len=frame_len, event_frames=np.array([], dtype=int))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", type=Path, help="ficheiro WAV de 16 bit em vez do sinal de brinquedo")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--events", type=int, default=0, help="nº de transitórios a injetar")
    ap.add_argument("--theta", type=float, default=None, help="limiar de esparsidade")
    ap.add_argument("--ternary", action="store_true", help="pesos {-1, 0, 1}")
    ap.add_argument("--sigma", type=float, default=0.0, help="variabilidade do crossbar")
    ap.add_argument("--width", type=int, default=100, help="tramas a desenhar")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = PCConfig(seed=args.seed, **({"theta": args.theta} if args.theta else {}))
    frame_len = cfg.sizes[0]

    print("A preparar o sinal…")
    if args.wav:
        signal = load_wav(args.wav, frame_len)
        if signal.n_frames < 50:
            raise SystemExit("áudio curto demais — preciso de pelo menos 50 tramas")
    else:
        signal = make_signal(n_frames=args.frames, frame_len=frame_len,
                             seed=args.seed + 1, n_events=args.events)
    tr, te = signal.split(0.8)

    net = PCNetwork(cfg)
    substrato = "float32"
    if args.ternary or args.sigma:
        net.attach_device(DeviceModel(ternary=True, sigma_rel=args.sigma, seed=args.seed))
        substrato = f"ternário {{-1,0,1}}" + (f", crossbar σ={args.sigma}" if args.sigma else "")

    print(f"A treinar {args.epochs} passagens sobre {tr.n_frames} tramas "
          f"({substrato})…")
    train(net, tr.frames, epochs=args.epochs)

    # ------------------------------------------------------------------
    # o que a rede previu, trama a trama, antes de ver cada uma
    net.reset()
    preds, traces = [], []
    for frame in te.frames:
        preds.append(net.predict_next())
        traces.append(net.step(frame, learn=False))
    preds = np.array(preds)

    n = min(args.width, len(traces))
    real = te.frames[:n].mean(axis=1)
    pred = preds[:n].mean(axis=1)
    surprise = np.array([t.open_loop_surprise for t in traces[:n]])
    iters = np.array([t.iters for t in traces[:n]])
    lo, hi = float(min(real.min(), pred.min())), float(max(real.max(), pred.max()))

    print(f"\n{n} tramas de teste, uma coluna por trama "
          f"({frame_len} amostras cada):\n")
    print(f"  sensor     {spark(real, lo, hi)}")
    print(f"  previsão   {spark(pred, lo, hi)}   <- feita ANTES de ver a trama")
    print(f"  surpresa   {spark(surprise, 0.0, float(surprise.max()))}")
    print(f"  iterações  {''.join(str(min(i, 9)) for i in iters)}")
    if len(te.event_frames):
        marks = ["·"] * n
        for i in te.event_frames:
            if i < n:
                marks[i] = "V"
        print(f"  eventos    {''.join(marks)}   <- transitórios injetados")

    # ------------------------------------------------------------------
    stats = evaluate(net, te.frames)
    baseline = persistence_nrmse(te.frames)
    print(f"""
  erro de previsão      {stats.pred_nrmse:.3f}   (1.0 = tão bom como prever zero)
  repetir a anterior    {baseline:.3f}   <- a linha de base a bater
  erros silenciados     {100 * stats.silenced_frac:.0f}%   (|ε| < {net.cfg.theta})
  conversões ADC        {100 * stats.adc_frac:.1f}%   do que uma rede densa faria
  iterações por trama   {stats.mean_iters:.1f}   (teto {net.cfg.max_iters})
  tramas explicadas     {100 * stats.explained_frac:.0f}%   saíram sem transmitir nada
""")
    verdict = ("A rede aprendeu a estrutura do sinal."
               if stats.pred_nrmse < min(baseline, 0.9)
               else "A rede NÃO aprendeu — o sinal é demasiado difícil ou faltam passagens.")
    print(f"  {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
