"""Episodic memory: record on the spot, consolidate later."""

import numpy as np
import pytest

from pcnet import PCConfig, PCNetwork, make_signal
from pcnet.dtypes import F
from pcnet.episodic import EpisodicConfig, EpisodicMemory


def mem(n_slots=8, **kw):
    return EpisodicMemory(4, 6, EpisodicConfig(n_slots=n_slots, **kw))


def episode(m, key, value, surprise=100.0):
    return m.write(np.asarray(key, dtype=F), np.asarray(value, dtype=F),
                   np.zeros(6, dtype=F), np.zeros(6, dtype=F), surprise)


# --------------------------------------------------------------------------
def test_only_surprising_things_get_written():
    """The threshold is relative to the usual: what is noteworthy depends on
    what tends to happen, not on a hand-tuned constant."""
    m = mem(write_threshold=2.0)
    for _ in range(200):
        episode(m, [1, 0, 0, 0], [1, 0, 0, 0, 0, 0], surprise=1.0)
    banal = m.n_written
    episode(m, [0, 1, 0, 0], [0, 1, 0, 0, 0, 0], surprise=50.0)
    assert m.n_written > banal


def test_recall_is_by_content_not_by_address():
    """A piece of the context is enough to bring back the episode - pattern
    completion, not indexing."""
    m = mem(read_threshold=0.5)
    episode(m, [1, 0, 0, 0], [9, 9, 9, 0, 0, 0])
    value, conf = m.read(np.array([0.9, 0.1, 0.0, 0.0], dtype=F))
    assert value is not None and conf > 0.9
    assert np.allclose(value[:3], 9.0, atol=1e-4)


def test_an_unfamiliar_context_gets_no_opinion():
    """A memory that always has an opinion is a memory that injects noise."""
    m = mem(read_threshold=0.6)
    episode(m, [1, 0, 0, 0], [5, 0, 0, 0, 0, 0])
    value, conf = m.read(np.array([0, 0, 1, 0], dtype=F))
    assert value is None and conf < 0.6


def test_a_repeated_context_reinforces_instead_of_duplicating():
    m = mem()
    episode(m, [1, 0, 0, 0], [2, 0, 0, 0, 0, 0])
    occupied = m.n_occupied
    strength = float(m.strength.max())
    episode(m, [1, 0, 0, 0], [2, 0, 0, 0, 0, 0])
    assert m.n_occupied == occupied
    assert float(m.strength.max()) > strength


def test_the_budget_is_fixed():
    """A memory that grows over time does not fit on a device."""
    m = mem(n_slots=8)
    for i in range(200):
        key = np.zeros(4, dtype=F)
        key[i % 4] = 1.0 + 0.01 * i
        episode(m, key, np.full(6, i, dtype=F))
    assert m.n_occupied <= 8


def test_reservoir_keeps_a_sample_of_the_whole_history():
    """The store has to keep a sample of *everything* it has seen, not the
    last arrivals.

    Without this, a new task sweeps the store in minutes and consolidation is
    left with nothing old to replay - which is the difference between a memory
    and a buffer.
    """
    # write_threshold=0 turns off the surprise filter: here the reservoir
    # is tested in isolation.
    m = mem(n_slots=16, reservoir=True, write_threshold=0.0)
    rng = np.random.default_rng(0)
    n_stream = 500
    for i in range(n_stream):
        key = rng.standard_normal(4).astype(F)
        episode(m, key, np.full(6, i, dtype=F))  # the value marks the age

    ages = m.values[m.strength > 0][:, 0]
    assert len(ages) == 16
    # spread across the whole history, not piled up at the end
    assert ages.min() < 0.25 * n_stream
    assert ages.max() > 0.75 * n_stream
    assert 0.25 * n_stream < float(ages.mean()) < 0.75 * n_stream


def test_replay_favours_what_repeated():
    m = mem(n_slots=8)
    episode(m, [1, 0, 0, 0], [1, 0, 0, 0, 0, 0])
    episode(m, [0, 1, 0, 0], [2, 0, 0, 0, 0, 0])
    m.strength[:] = 0.0
    strong = int(np.argmax(np.abs(m.keys).sum(axis=1) > 0))
    m.strength[strong] = 10.0
    picks = m.replay(1, np.random.default_rng(0))
    assert picks == [strong]


def test_replay_on_an_empty_memory_is_harmless():
    assert mem().replay(4, np.random.default_rng(0)) == []


def test_the_store_never_freezes():
    """After covering the key space, it has to keep accepting new things. With
    the merge threshold at 0.95 it stopped - and a memory that stops recording
    is a buffer with good manners."""
    m = mem(n_slots=16, write_threshold=0.0)
    for i in range(120):
        angle = 2 * np.pi * i / 120
        episode(m, [np.cos(angle), np.sin(angle), 0, 0], np.full(6, i, dtype=F))
    late = m.n_written
    for i in range(120, 240):
        angle = 2 * np.pi * i / 120
        episode(m, [np.cos(angle), np.sin(angle), 0, 0], np.full(6, i, dtype=F))
    assert m.n_written > late


# --------------------------------------------------------------------------
# integration with the network
# --------------------------------------------------------------------------
def small_net(**kw):
    return PCNetwork(PCConfig.recommended(sizes=(32, 16, 8), **kw))


def small_signal(n=200, seed=1):
    return make_signal(n_frames=n, frame_len=32, seed=seed)


def test_memory_fills_up_while_the_network_runs():
    net = small_net()
    m = net.attach_memory(EpisodicConfig(n_slots=32))
    net.run(small_signal(150).frames, learn=True)
    assert m.n_occupied > 0
    assert m.n_occupied <= 32


def test_detaching_the_memory_restores_the_plain_network():
    net = small_net()
    sig = small_signal(80)
    net.run(sig.frames[:40], learn=True)
    before = net.predict_next().copy()
    net.attach_memory()
    net.detach_memory()
    assert np.allclose(before, net.predict_next(), atol=1e-5)


def test_consolidation_replays_without_disturbing_the_running_state():
    """Sleep is offline: it cannot touch the state of whoever is watching."""
    net = small_net()
    net.attach_memory(EpisodicConfig(n_slots=32))
    net.run(small_signal(150).frames, learn=True)

    snapshot = net.snapshot_state()
    n = net.consolidate(n_episodes=8)
    zs, prevs = net.snapshot_state()

    assert n > 0
    assert all(np.array_equal(a, b) for a, b in zip(snapshot[0], zs))
    assert all(np.array_equal(a, b) for a, b in zip(snapshot[1], prevs))


def test_consolidation_changes_the_weights():
    net = small_net()
    net.attach_memory(EpisodicConfig(n_slots=32))
    net.run(small_signal(150).frames, learn=True)
    W = [w.copy() for w in net.weights]
    net.consolidate(n_episodes=16)
    assert any(not np.array_equal(a, b) for a, b in zip(W, net.weights))


def test_consolidation_without_memory_does_nothing():
    assert small_net().consolidate(8) == 0
