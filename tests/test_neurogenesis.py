"""Neurogenesis: frozen units stay truly frozen, and novelty recruits."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pcnet import PCConfig, make_signal  # noqa: E402
from pcnet.neurogenesis import NeurogenesisConfig, NeurogenesisNetwork  # noqa: E402


def _net(**ng_kw):
    cfg = PCConfig(seed=0, fast_path=False, use_precision=True)
    return NeurogenesisNetwork(cfg, NeurogenesisConfig(**ng_kw))


def test_frozen_units_stay_silent_and_unlearned():
    net = _net()
    sig = make_signal(n_frames=120, frame_len=64, seed=3)
    net.run(sig.frames, learn=True)
    for l in range(1, net.L + 1):
        frozen = net.factor[l] == 0.0
        assert frozen.any(), "half the reserve should start frozen"
        # state at zero, weights at zero - they neither participate nor learn
        assert np.all(net.z[l][frozen] == 0.0)
        assert np.all(net.layers[l - 1].W[:, frozen] == 0.0)
        if l < net.L:
            assert np.all(net.layers[l].W[frozen, :] == 0.0)
        else:
            assert np.all(net.A[frozen, :] == 0.0)
            assert np.all(net.A[:, frozen] == 0.0)


def test_novelty_recruits_and_protects():
    net = _net(warmup=30, sustain=10)
    a = make_signal(n_frames=200, frame_len=64, seed=3)
    # new world: other frequencies, different amplitude
    b = make_signal(n_frames=200, frame_len=64, freqs=(0.31, 0.11, 0.44), seed=9)
    net.run(a.frames, learn=True)
    assert net.n_recruited == 0, "without a world change it should not recruit"
    before = net.active_counts()
    net.run(3.0 * b.frames, learn=True)
    assert net.n_recruited > 0, "the world change should have recruited"
    after = net.active_counts()
    assert all(after[l] >= before[l] for l in after)
    # the veteran units stayed protected
    for l in range(1, net.L + 1):
        assert np.any(net.factor[l] == np.float32(net.ng.protect_factor))


def test_eval_does_not_touch_detector_or_weights():
    net = _net()
    sig = make_signal(n_frames=60, frame_len=64, seed=3)
    net.run(sig.frames, learn=True)
    t0, W0 = net._t, [lay.W.copy() for lay in net.layers]
    net.run(10.0 * sig.frames, learn=False)
    assert net._t == t0
    for lay, W in zip(net.layers, W0):
        assert np.array_equal(lay.W, W)
