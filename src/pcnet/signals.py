"""Toy signals: the "sensor".

The problem is the simplest one that still has the right structure: predict
the next sample of a continuous signal. The signal is mostly mundane (a sum
of sinusoids with slowly drifting amplitude) and every now and then something
happens - a transient. That contrast is what makes the central claim
visible: correct prediction ≈ silence ≈ zero cost, and compute only rises
when surprise rises.

Frames are *non-overlapping* windows of `frame_len` samples. Overlapping
would make the task trivial (63 of the 64 samples would already have been
seen) and the result would be a vain measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dtypes import F

# Frequencies in cycles/sample, mutually incommensurable: no frame is a
# copy of the previous one, so the top really has to learn a transition.
# Calibrated for 64-sample frames - see `scaled_freqs`.
DEFAULT_FREQS = (0.031, 0.017, 0.0071)
REFERENCE_FRAME_LEN = 64


def scaled_freqs(frame_len: int) -> tuple[float, ...]:
    """Keeps the content of a frame constant when the frame changes size.

    What makes the problem hard is not the absolute frequency, it is how
    much phase advances within a frame and between frames. Without this
    scaling, shrinking the network to 32 sensory units changes the *task* at
    the same time as the model, and there is no telling which one to credit
    for the difference.
    """
    k = REFERENCE_FRAME_LEN / frame_len
    return tuple(f * k for f in DEFAULT_FREQS)


@dataclass
class ToySignal:
    """Signal + frames + where the surprising events are."""

    samples: np.ndarray  # (n_samples,)
    frames: np.ndarray  # (n_frames, frame_len)
    frame_len: int
    event_frames: np.ndarray  # indices of frames that contain a transient

    @property
    def n_frames(self) -> int:
        return self.frames.shape[0]

    def split(self, train_frac: float = 0.8) -> tuple["ToySignal", "ToySignal"]:
        """Temporal cut: train at the start, test at the end (no shuffling)."""
        cut = int(self.n_frames * train_frac)
        return self._slice(0, cut), self._slice(cut, self.n_frames)

    def _slice(self, i0: int, i1: int) -> "ToySignal":
        n = self.frame_len
        ev = self.event_frames
        return ToySignal(
            samples=self.samples[i0 * n : i1 * n],
            frames=self.frames[i0:i1],
            frame_len=n,
            event_frames=(ev[(ev >= i0) & (ev < i1)] - i0),
        )


def frame_signal(samples: np.ndarray, frame_len: int) -> np.ndarray:
    """Cuts the signal into non-overlapping frames, discarding the tail."""
    n = (len(samples) // frame_len) * frame_len
    return np.asarray(samples[:n], dtype=F).reshape(-1, frame_len)


def make_signal(
    n_frames: int = 400,
    frame_len: int = 64,
    freqs: tuple[float, ...] | None = None,
    am_rate: float = 3e-4,
    noise: float = 0.01,
    n_events: int = 0,
    event_gain: float = 3.0,
    event_len: int = 12,
    seed: int = 0,
) -> ToySignal:
    """Generates the toy signal.

    `n_events` transients are injected into frames at random (never the
    first ones, so the network has time to learn what "normal" is).
    `freqs=None` scales the frequencies to the frame size.
    """
    rng = np.random.default_rng(seed)
    if freqs is None:
        freqs = scaled_freqs(frame_len)
    n_samples = n_frames * frame_len
    t = np.arange(n_samples, dtype=np.float64)

    x = np.zeros(n_samples, dtype=np.float64)
    for i, f in enumerate(freqs):
        phase = rng.uniform(0, 2 * np.pi)
        # amplitude drifting slowly: the network has to keep tracking
        am = 1.0 + 0.3 * np.sin(2 * np.pi * am_rate * (i + 1) * t + phase)
        x += am * np.sin(2 * np.pi * f * t + phase)

    x /= np.max(np.abs(x)) + 1e-9  # amplitude in [-1, 1]

    event_frames: list[int] = []
    if n_events > 0:
        first = max(1, n_frames // 4)
        candidates = np.arange(first, n_frames)
        picks = rng.choice(candidates, size=min(n_events, len(candidates)), replace=False)
        for fi in np.sort(picks):
            start = int(fi) * frame_len + rng.integers(0, max(1, frame_len - event_len))
            burst = rng.standard_normal(event_len) * event_gain * 0.3
            burst *= np.hanning(event_len)
            x[start : start + event_len] += burst
            event_frames.append(int(fi))

    if noise > 0.0:
        x += rng.standard_normal(n_samples) * noise

    x = np.clip(x, -1.5, 1.5).astype(F)
    return ToySignal(
        samples=x,
        frames=frame_signal(x, frame_len),
        frame_len=frame_len,
        event_frames=np.array(event_frames, dtype=int),
    )


def persistence_nrmse(frames: np.ndarray) -> float:
    """Honest baseline: predict that the next frame equals the previous one.

    If the network does not beat this, it learned nothing temporal.
    """
    frames = np.asarray(frames, dtype=F)
    if len(frames) < 2:
        return 0.0
    err = frames[1:] - frames[:-1]
    rmse = np.sqrt(np.mean(np.square(err), axis=1))
    rms = np.sqrt(np.mean(np.square(frames[1:]), axis=1))
    return float(np.mean(rmse / np.maximum(rms, 1e-9)))
