#!/usr/bin/env python3
"""See the model working, in the terminal.

    python3 scripts/demo.py                      # toy signal
    python3 scripts/demo.py --events 8           # with transients (surprises)
    python3 scripts/demo.py --wav file.wav       # your own audio
    python3 scripts/demo.py --ternary            # weights {-1,0,1}
    python3 scripts/demo.py --ternary --sigma 0.2  # ... on a faulty crossbar

Shows three lines aligned in time: what the sensor gave, what the network had
predicted *before* seeing it, and the surprise. Where the first two coincide,
the third is silent and the network spent almost nothing.
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
    """A one-line chart in characters."""
    v = np.asarray(values, dtype=float)
    lo = float(v.min()) if lo is None else lo
    hi = float(v.max()) if hi is None else hi
    if hi - lo < 1e-12:
        return BLOCKS[0] * len(v)
    idx = np.clip(((v - lo) / (hi - lo) * (len(BLOCKS) - 1)).round(), 0, len(BLOCKS) - 1)
    return "".join(BLOCKS[int(i)] for i in idx)


def load_wav(path: Path, frame_len: int) -> ToySignal:
    """Reads a 16-bit mono/stereo WAV and cuts it into frames."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise SystemExit(f"{path}: can only read 16-bit WAV (this one has "
                             f"{8 * w.getsampwidth()} bit)")
        raw = w.readframes(w.getnframes())
        x = np.frombuffer(raw, dtype="<i2").astype(np.float64)
        if w.getnchannels() > 1:
            x = x.reshape(-1, w.getnchannels()).mean(axis=1)
        rate = w.getframerate()

    peak = np.max(np.abs(x)) or 1.0
    x = (x / peak).astype(F)
    print(f"  {path.name}: {len(x)} samples at {rate} Hz "
          f"({len(x) / rate:.1f} s, {len(x) // frame_len} frames of {frame_len})")
    return ToySignal(samples=x, frames=frame_signal(x, frame_len),
                     frame_len=frame_len, event_frames=np.array([], dtype=int))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", type=Path, help="16-bit WAV file instead of the toy signal")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--events", type=int, default=0, help="number of transients to inject")
    ap.add_argument("--theta", type=float, default=None, help="sparsity threshold")
    ap.add_argument("--ternary", action="store_true", help="weights {-1, 0, 1}")
    ap.add_argument("--sigma", type=float, default=0.0, help="crossbar variability")
    ap.add_argument("--width", type=int, default=100, help="frames to draw")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = PCConfig(seed=args.seed, **({"theta": args.theta} if args.theta else {}))
    frame_len = cfg.sizes[0]

    print("Preparing the signal…")
    if args.wav:
        signal = load_wav(args.wav, frame_len)
        if signal.n_frames < 50:
            raise SystemExit("audio too short, need at least 50 frames")
    else:
        signal = make_signal(n_frames=args.frames, frame_len=frame_len,
                             seed=args.seed + 1, n_events=args.events)
    tr, te = signal.split(0.8)

    net = PCNetwork(cfg)
    substrato = "float32"
    if args.ternary or args.sigma:
        net.attach_device(DeviceModel(ternary=True, sigma_rel=args.sigma, seed=args.seed))
        substrato = f"ternary {{-1,0,1}}" + (f", crossbar σ={args.sigma}" if args.sigma else "")

    print(f"Training {args.epochs} passes over {tr.n_frames} frames "
          f"({substrato})…")
    train(net, tr.frames, epochs=args.epochs)

    # ------------------------------------------------------------------
    # what the network predicted, frame by frame, before seeing each one
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

    print(f"\n{n} test frames, one column per frame "
          f"({frame_len} samples each):\n")
    print(f"  sensor     {spark(real, lo, hi)}")
    print(f"  predicted  {spark(pred, lo, hi)}   <- made BEFORE seeing the frame")
    print(f"  surprise   {spark(surprise, 0.0, float(surprise.max()))}")
    print(f"  iterations {''.join(str(min(i, 9)) for i in iters)}")
    if len(te.event_frames):
        marks = ["·"] * n
        for i in te.event_frames:
            if i < n:
                marks[i] = "V"
        print(f"  events     {''.join(marks)}   <- injected transients")

    # ------------------------------------------------------------------
    stats = evaluate(net, te.frames)
    baseline = persistence_nrmse(te.frames)
    print(f"""
  prediction error      {stats.pred_nrmse:.3f}   (1.0 = as good as predicting zero)
  repeat previous       {baseline:.3f}   <- the baseline to beat
  silenced errors       {100 * stats.silenced_frac:.0f}%   (|ε| < {net.cfg.theta})
  ADC conversions       {100 * stats.adc_frac:.1f}%   of what a dense network would do
  iterations per frame  {stats.mean_iters:.1f}   (ceiling {net.cfg.max_iters})
  explained frames      {100 * stats.explained_frac:.0f}%   left without transmitting anything
""")
    verdict = ("The network learned the signal's structure."
               if stats.pred_nrmse < min(baseline, 0.9)
               else "The network did NOT learn, the signal is too hard or more passes are needed.")
    print(f"  {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
