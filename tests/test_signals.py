import numpy as np

from pcnet import make_signal, persistence_nrmse
from pcnet.dtypes import F
from pcnet.signals import frame_signal


def test_frames_are_non_overlapping_and_cover_the_signal():
    sig = make_signal(n_frames=10, frame_len=8, seed=0)
    assert sig.frames.shape == (10, 8)
    assert sig.frames.dtype == F
    assert np.array_equal(sig.frames.ravel(), sig.samples[:80])


def test_framing_discards_the_tail():
    assert frame_signal(np.arange(10, dtype=F), 4).shape == (2, 4)


def test_signal_is_bounded_and_reproducible():
    a = make_signal(n_frames=20, seed=3)
    b = make_signal(n_frames=20, seed=3)
    assert np.array_equal(a.samples, b.samples)
    assert np.max(np.abs(a.samples)) <= 1.5


def test_split_is_temporal_not_shuffled():
    sig = make_signal(n_frames=20, frame_len=8, seed=0)
    tr, te = sig.split(0.75)
    assert tr.n_frames == 15 and te.n_frames == 5
    assert np.array_equal(tr.frames[0], sig.frames[0])
    assert np.array_equal(te.frames[0], sig.frames[15])


def test_events_are_injected_and_indexed():
    sig = make_signal(n_frames=60, seed=5, n_events=6, event_gain=5.0)
    assert len(sig.event_frames) == 6
    assert len(set(sig.event_frames.tolist())) == 6
    energy = np.abs(sig.frames).max(axis=1)
    mask = np.zeros(sig.n_frames, dtype=bool)
    mask[sig.event_frames] = True
    assert energy[mask].mean() > energy[~mask].mean()


def test_split_keeps_event_indices_relative():
    sig = make_signal(n_frames=40, seed=5, n_events=8)
    tr, te = sig.split(0.5)
    assert all(0 <= i < te.n_frames for i in te.event_frames)
    assert len(tr.event_frames) + len(te.event_frames) == len(sig.event_frames)


def test_persistence_baseline_is_a_real_baseline():
    sig = make_signal(n_frames=50, seed=1)
    nrmse = persistence_nrmse(sig.frames)
    assert np.isfinite(nrmse) and nrmse > 0
    # Non-overlapping frames of an oscillating signal: repeating the previous one is bad.
    assert nrmse > 0.5
