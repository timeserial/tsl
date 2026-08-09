"""Carregar problemas reais sem enganar a métrica."""

import wave

import numpy as np
import pytest

from pcnet.datasets import load_csv_column, load_toy, load_wav, trim_silence
from pcnet.dtypes import F


def write_wav(path, x, rate=8000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return path


def tone(n, freq=0.02, amp=1.0):
    return amp * np.sin(2 * np.pi * freq * np.arange(n))


# --------------------------------------------------------------------------
def test_normalization_uses_only_the_training_split():
    """A fuga mais fácil de cometer: normalizar com estatísticas do futuro.

    Dá-se um sinal cuja segunda metade tem escala completamente diferente. Se
    a normalização visse o conjunto todo, o treino sairia com desvio-padrão
    diferente de 1.
    """
    from pcnet.datasets import _split_and_normalize

    x = np.concatenate([tone(4000), 50.0 + 30.0 * tone(1000, 0.03)])
    d = _split_and_normalize(x, 64, "t", 0.8)
    assert float(d.train.std()) == pytest.approx(1.0, abs=0.05)
    assert abs(float(d.train.mean())) < 0.05


def test_a_silent_test_split_is_refused(tmp_path):
    """Se o teste for silêncio, "repetir a anterior" ganha e a tabela mente."""
    x = np.concatenate([tone(8000), np.zeros(2000)])
    path = write_wav(tmp_path / "decai.wav", x)
    with pytest.raises(ValueError, match="não tem energia"):
        load_wav(path, frame_len=64, trim=False)


def test_trim_silence_rescues_a_decaying_recording(tmp_path):
    x = np.concatenate([np.zeros(500), tone(9000), np.zeros(2000)])
    path = write_wav(tmp_path / "decai.wav", x)
    ds = load_wav(path, frame_len=64, trim=True)
    assert ds.rms_test > 0.5 * ds.rms_train


def test_trim_silence_keeps_the_loud_part():
    x = np.concatenate([np.zeros(100), np.ones(50), np.zeros(100)])
    assert len(trim_silence(x)) == 50


def test_trim_silence_survives_an_all_zero_signal():
    x = np.zeros(10)
    assert np.array_equal(trim_silence(x), x)


def test_wav_roundtrip_shapes_and_dtype(tmp_path):
    path = write_wav(tmp_path / "t.wav", tone(12000), rate=16000)
    ds = load_wav(path, frame_len=32)
    assert ds.frame_len == 32
    assert ds.train.dtype == F and ds.train.shape[1] == 32
    assert ds.rate_hz == 16000
    assert ds.n_frames == len(ds.train) + len(ds.test)


def test_csv_column_is_read_and_missing_column_is_named(tmp_path):
    path = tmp_path / "s.csv"
    rows = ["t,a,b"] + [f"{i},{np.sin(i / 7):.5f},{np.cos(i / 11):.5f}"
                        for i in range(3000)]
    path.write_text("\n".join(rows))

    ds = load_csv_column(path, "a", frame_len=32)
    assert ds.units == "a" and ds.train.shape[1] == 32

    with pytest.raises(ValueError, match="não tem coluna 'z'"):
        load_csv_column(path, "z")


def test_too_short_is_refused(tmp_path):
    path = write_wav(tmp_path / "curto.wav", tone(200))
    with pytest.raises(ValueError, match="curto demais"):
        load_wav(path, frame_len=64)


def test_toy_comes_through_the_same_door():
    ds = load_toy(n_frames=300, frame_len=64)
    assert ds.train.shape[1] == 64
    assert ds.rms_test > 0.5 * ds.rms_train
    assert "brinquedo" in ds.describe()
