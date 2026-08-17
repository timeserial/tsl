"""Real problems.

No data inside the repository: these functions take a path and return frames.
What matters is that any univariate signal sampled in time enters through the
same door as the toy signal, and comes out comparable.

Normalization: subtract the mean and divide by the standard deviation
**computed on the training stretch only**. It is the easiest precaution to
forget and the one that inflates results the most, normalizing with
whole-set statistics lets information leak from the future into the past.
"""

from __future__ import annotations

import csv
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dtypes import F
from .signals import frame_signal


@dataclass
class Dataset:
    """A real signal, already framed and already split in time."""

    name: str
    train: np.ndarray  # (n_train, frame_len)
    test: np.ndarray  # (n_test, frame_len)
    frame_len: int
    units: str = ""
    rate_hz: float | None = None
    rms_train: float = 0.0
    rms_test: float = 0.0

    @property
    def n_frames(self) -> int:
        return len(self.train) + len(self.test)

    def describe(self) -> str:
        rate = f", {self.rate_hz:g} Hz" if self.rate_hz else ""
        return (
            f"{self.name}: {self.n_frames} tramas de {self.frame_len} "
            f"({len(self.train)} treino / {len(self.test)} teste){rate}"
            f" | RMS treino {self.rms_train:.3g}, teste {self.rms_test:.3g}"
        )


def _split_and_normalize(
    x: np.ndarray, frame_len: int, name: str, train_frac: float, **kw
) -> Dataset:
    x = np.asarray(x, dtype=np.float64)
    cut = int(len(x) * train_frac)
    mu = float(x[:cut].mean())
    sd = float(x[:cut].std()) or 1.0
    x = ((x - mu) / sd).astype(F)

    train = frame_signal(x[:cut], frame_len)
    test = frame_signal(x[cut:], frame_len)
    if len(train) < 20 or len(test) < 10:
        raise ValueError(
            f"{name}: sinal curto demais ({len(train)} tramas de treino, "
            f"{len(test)} de teste)"
        )

    # Guard against the easiest mistake to make and the hardest to detect:
    # a test stretch with no energy. If the test is silence, "repeat the
    # previous one" scores perfectly, everything else takes an astronomical
    # normalized error, and the whole table ends up measuring the silence
    # instead of the model.
    rms_tr = float(np.sqrt(np.mean(np.square(train))))
    rms_te = float(np.sqrt(np.mean(np.square(test))))
    if rms_te < 0.05 * rms_tr:
        raise ValueError(
            f"{name}: o troço de teste não tem energia (RMS {rms_te:.4g} contra "
            f"{rms_tr:.4g} no treino). Qualquer métrica normalizada sobre isto "
            f"é ficção - corta o silêncio ou usa outro troço."
        )
    return Dataset(name=name, train=train, test=test, frame_len=frame_len,
                   rms_train=rms_tr, rms_test=rms_te, **kw)


# ---------------------------------------------------------------------------
def trim_silence(x: np.ndarray, rel: float = 0.02) -> np.ndarray:
    """Trims silence at the start and at the end.

    Short recordings almost always end in silence, and a silent test stretch
    destroys any normalized metric. The threshold is relative to the peak,
    so it does not depend on the recording's volume.
    """
    env = np.abs(np.asarray(x, dtype=np.float64))
    if env.max() <= 0:
        return x
    loud = np.flatnonzero(env >= rel * env.max())
    return x[loud[0] : loud[-1] + 1] if len(loud) else x


def load_wav(
    path: str | Path,
    frame_len: int = 64,
    train_frac: float = 0.8,
    trim: bool = True,
) -> Dataset:
    """Real audio from a 16-bit WAV (mono, or converted to mono)."""
    path = Path(path)
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: só leio WAV de 16 bit")
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64)
        if w.getnchannels() > 1:
            x = x.reshape(-1, w.getnchannels()).mean(axis=1)
        rate = float(w.getframerate())
    if trim:
        x = trim_silence(x)
    return _split_and_normalize(
        x, frame_len, path.stem, train_frac, units="amplitude", rate_hz=rate
    )


def load_csv_column(
    path: str | Path,
    column: str,
    frame_len: int = 64,
    train_frac: float = 0.8,
    name: str | None = None,
    rate_hz: float | None = None,
) -> Dataset:
    """One column of a time series in a CSV with a header."""
    path = Path(path)
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if column not in (reader.fieldnames or []):
            raise ValueError(
                f"{path}: não tem coluna {column!r} (tem {reader.fieldnames})"
            )
        x = np.array([float(row[column]) for row in reader], dtype=np.float64)
    return _split_and_normalize(
        x,
        frame_len,
        name or f"{path.stem}:{column}",
        train_frac,
        units=column,
        rate_hz=rate_hz,
    )


def load_uci_har_inertial(
    root: str | Path,
    signal: str = "total_acc_x",
    split: str = "train",
    frame_len: int = 64,
    train_frac: float = 0.8,
) -> Dataset:
    """Real 50 Hz accelerometer data from UCI HAR - the plan's target domain.

    The file comes in 128-sample windows with 50% overlap, and the windows
    are grouped by subject and by activity. Concatenating everything would
    create artificial jumps at the boundaries, which the model would learn to
    "predict" and which exist in no real sensor. Therefore:

      * the continuous signal is rebuilt by taking the first `128/2` samples
        of each window (which is exactly what the 50% overlap allows);
      * only the *longest contiguous run* of windows from the same subject
        is used.

    Grouping is by subject and not by activity: within a subject the windows
    come from the same continuous recording, and the activity changes are
    true transitions, the person really did switch from walking to climbing
    stairs. They are the kind of surprise this architecture exists to handle,
    not artifacts.
    """
    root = Path(root)
    base = root / split / "Inertial Signals" / f"{signal}_{split}.txt"
    if not base.exists():
        raise FileNotFoundError(f"não encontrei {base}")

    windows = np.loadtxt(base, dtype=np.float64)
    subject = np.loadtxt(root / split / f"subject_{split}.txt", dtype=int)
    activity = np.loadtxt(root / split / f"y_{split}.txt", dtype=int)

    key = subject
    best_start = best_len = cur_start = 0
    for i in range(1, len(key) + 1):
        if i == len(key) or key[i] != key[cur_start]:
            if i - cur_start > best_len:
                best_start, best_len = cur_start, i - cur_start
            cur_start = i
    run = windows[best_start : best_start + best_len]
    hop = run.shape[1] // 2
    x = np.concatenate([w[:hop] for w in run])

    n_act = len(np.unique(activity[best_start : best_start + best_len]))
    name = f"HAR {signal} (sujeito {subject[best_start]}, {n_act} atividades)"
    return _split_and_normalize(
        x, frame_len, name, train_frac, units="g", rate_hz=50.0
    )


def load_toy(n_frames: int = 900, frame_len: int = 64, seed: int = 1,
             train_frac: float = 0.8) -> Dataset:
    """The toy signal, through the same door - it serves as a control."""
    from .signals import make_signal

    sig = make_signal(n_frames=n_frames, frame_len=frame_len, seed=seed)
    return _split_and_normalize(sig.samples, frame_len, "brinquedo", train_frac)
