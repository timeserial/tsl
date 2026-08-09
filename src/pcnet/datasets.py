"""Problemas reais.

Nada de dados dentro do repositório: estas funções recebem um caminho e
devolvem tramas. O que interessa é que qualquer sinal univariado amostrado no
tempo entra pela mesma porta que o sinal de brinquedo, e sai comparável.

Normalização: subtrai-se a média e divide-se pelo desvio-padrão **calculados
só no troço de treino**. É o cuidado mais fácil de esquecer e o que mais
inflaciona resultados — normalizar com estatísticas do conjunto todo deixa
passar informação do futuro para o passado.
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
    """Um sinal real, já em tramas e já dividido no tempo."""

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

    # Guarda contra o erro mais fácil de cometer e o mais difícil de detetar:
    # um troço de teste sem energia. Se o teste for silêncio, "repetir a
    # anterior" acerta perfeitamente, tudo o resto leva um erro normalizado
    # astronómico, e a tabela toda passa a medir o silêncio em vez do modelo.
    rms_tr = float(np.sqrt(np.mean(np.square(train))))
    rms_te = float(np.sqrt(np.mean(np.square(test))))
    if rms_te < 0.05 * rms_tr:
        raise ValueError(
            f"{name}: o troço de teste não tem energia (RMS {rms_te:.4g} contra "
            f"{rms_tr:.4g} no treino). Qualquer métrica normalizada sobre isto "
            f"é ficção — corta o silêncio ou usa outro troço."
        )
    return Dataset(name=name, train=train, test=test, frame_len=frame_len,
                   rms_train=rms_tr, rms_test=rms_te, **kw)


# ---------------------------------------------------------------------------
def trim_silence(x: np.ndarray, rel: float = 0.02) -> np.ndarray:
    """Corta silêncio no início e no fim.

    Gravações curtas acabam quase sempre em silêncio, e um troço de teste
    silencioso destrói qualquer métrica normalizada. O limiar é relativo ao
    pico, para não depender do volume da gravação.
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
    """Áudio real de um WAV de 16 bit (mono, ou convertido para mono)."""
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
    """Uma coluna de uma série temporal em CSV com cabeçalho."""
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
    """Acelerómetro real a 50 Hz do UCI HAR — o domínio-alvo do plano.

    O ficheiro vem em janelas de 128 amostras com 50% de sobreposição, e as
    janelas estão agrupadas por sujeito e por atividade. Concatenar tudo
    criaria saltos artificiais nas fronteiras, que o modelo aprenderia a
    "prever" e que não existem em nenhum sensor real. Por isso:

      * reconstrói-se o sinal contínuo tomando as primeiras `128/2` amostras
        de cada janela (que é exatamente o que a sobreposição de 50% permite);
      * usa-se apenas a *maior corrida contígua* de janelas do mesmo sujeito.

    Agrupa-se por sujeito e não por atividade: dentro de um sujeito as janelas
    vêm da mesma gravação contínua, e as mudanças de atividade são transições
    verdadeiras — a pessoa mudou mesmo de andar para subir escadas. São o tipo
    de surpresa que esta arquitetura existe para tratar, não artefactos.
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
    """O sinal de brinquedo, pela mesma porta — serve de controlo."""
    from .signals import make_signal

    sig = make_signal(n_frames=n_frames, frame_len=frame_len, seed=seed)
    return _split_and_normalize(sig.samples, frame_len, "brinquedo", train_frac)
