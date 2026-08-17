"""The four claims of step 1, each as a test that can fail."""

import numpy as np
import pytest

from pcnet import PCConfig, PCNetwork, make_signal, persistence_nrmse
from pcnet.dtypes import F
from pcnet.metrics import EXIT_CEILING, EXIT_EXPLAINED, summarize
from pcnet.train import evaluate, evaluate_with_theta, train

SMALL = dict(sizes=(32, 16, 8))


def small_net(**kw):
    return PCNetwork(PCConfig(**{**SMALL, **kw}))


def small_signal(n_frames=200, seed=1, **kw):
    return make_signal(n_frames=n_frames, frame_len=32, seed=seed, **kw)


# --------------------------------------------------------------------------
# mechanics
# --------------------------------------------------------------------------
def test_frame_shape_is_checked():
    net = small_net()
    with pytest.raises(ValueError, match="forma"):
        net.step(np.zeros(7, dtype=F))


def test_states_have_the_declared_shapes():
    net = small_net()
    assert [z.shape[0] for z in net.z] == [32, 16, 8]
    assert [W.shape for W in net.weights] == [(32, 16), (16, 8)]
    assert net.eps_per_pass == 32 + 16 + 8


def test_same_seed_same_run():
    sig = small_signal(60)
    a = [t.pred_rmse for t in small_net(seed=7).run(sig.frames)]
    b = [t.pred_rmse for t in small_net(seed=7).run(sig.frames)]
    assert a == b


def test_reset_clears_the_temporal_state():
    net = small_net()
    net.run(small_signal(20).frames)
    assert np.any(net.z[net.L] != 0)
    net.reset()
    assert all(not np.any(z) for z in net.z)


def test_evaluating_does_not_disturb_the_training_state():
    """The network is online: measuring cannot alter what is being measured."""
    net = small_net()
    sig = small_signal(60)
    net.run(sig.frames[:30])
    before = net.snapshot_state()

    evaluate(net, sig.frames[30:])

    zs, prevs = net.snapshot_state()
    assert all(np.array_equal(a, b) for a, b in zip(before[0], zs))
    assert all(np.array_equal(a, b) for a, b in zip(before[1], prevs))


def test_training_curve_is_the_same_with_and_without_evaluation():
    sig = small_signal(150)
    tr, te = sig.split(0.8)
    quiet, watched = small_net(), small_net()
    train(quiet, tr.frames, epochs=10)
    train(watched, tr.frames, epochs=10, eval_frames=te.frames)
    for a, b in zip(quiet.weights, watched.weights):
        assert np.allclose(a, b, atol=1e-6)


def test_predict_next_is_the_open_loop_prediction():
    """ε_0 on the first evaluation is exactly x − predict_next() from the previous step."""
    net = small_net()
    sig = small_signal(30)
    net.run(sig.frames[:10])
    expected = net.predict_next()
    trace = net.step(sig.frames[10], learn=False)
    assert np.allclose(sig.frames[10] - expected, trace.pred_error, atol=1e-5)


def test_state_dict_roundtrip_reproduces_predictions():
    net = small_net()
    sig = small_signal(40)
    net.run(sig.frames)
    weights = net.state_dict()

    clone = small_net(seed=99)
    clone.load_state_dict(weights)
    clone._z_top_prev[:] = net._z_top_prev
    assert np.allclose(net.predict_next(), clone.predict_next(), atol=1e-6)


def test_load_state_dict_rejects_wrong_shapes():
    net = small_net()
    bad = net.state_dict()
    bad["W0"] = np.zeros((3, 3), dtype=F)
    with pytest.raises(ValueError, match="W0"):
        net.load_state_dict(bad)


# --------------------------------------------------------------------------
# 1. the settling dynamics converges
# --------------------------------------------------------------------------
def test_settling_decreases_the_energy():
    """Each iteration lowers the energy - except, at most, the last one.

    Settling is gradient descent with a fixed step: near the stability limit
    the last evaluation can rise a little. If it rises, that rise is what
    makes the minimum-gain criterion stop the loop, so it can only appear in
    the last position.
    """
    net = small_net(theta=0.0)
    sig = small_signal(120)
    train(net, sig.frames, epochs=5)
    for trace in net.run(sig.frames, learn=False, reset=True):
        s = trace.surprise
        assert all(s[k + 1] <= s[k] + 1e-6 for k in range(len(s) - 2)), s
        assert s[-1] <= s[0], s


def test_settling_never_exceeds_the_ceiling():
    net = small_net(max_iters=4, settle_min_gain=0.0, theta=0.0)
    for trace in net.run(small_signal(30).frames, learn=False):
        assert trace.iters <= 4
        assert len(trace.surprise) == trace.iters + 1


def test_more_settling_explains_more():
    """Letting it settle longer reduces the final energy. Without this, iterating is useless."""
    sig = small_signal(80)
    shallow = small_net(max_iters=1, settle_min_gain=0.0, theta=0.0)
    deep = small_net(max_iters=10, settle_min_gain=0.0, theta=0.0)
    s1 = summarize(shallow.run(sig.frames, learn=False), 1, shallow.eps_per_pass)
    s2 = summarize(deep.run(sig.frames, learn=False), 10, deep.eps_per_pass)
    assert s2.mean_final_surprise < s1.mean_final_surprise


# --------------------------------------------------------------------------
# 2. local learning works
# --------------------------------------------------------------------------
def test_local_learning_beats_the_persistence_baseline():
    sig = small_signal(300)
    tr, te = sig.split(0.8)
    net = small_net()
    before = evaluate(net, te.frames)
    train(net, tr.frames, epochs=30)
    after = evaluate(net, te.frames)

    assert after.pred_nrmse < before.pred_nrmse
    assert after.pred_nrmse < persistence_nrmse(te.frames)
    assert after.pred_nrmse < 0.3


def test_learning_is_off_when_asked():
    net = small_net()
    sig = small_signal(50)
    W0 = [W.copy() for W in net.weights]
    A0 = net.A.copy()
    net.run(sig.frames, learn=False)
    assert all(np.array_equal(a, b) for a, b in zip(W0, net.weights))
    assert np.array_equal(A0, net.A)


def test_transition_learns_the_temporal_structure():
    """Without a temporal transition the top has no way to predict the next frame."""
    sig = small_signal(300)
    tr, te = sig.split(0.8)
    with_t, without_t = small_net(), small_net(use_transition=False)
    train(with_t, tr.frames, epochs=40)
    train(without_t, tr.frames, epochs=40)
    assert evaluate(with_t, te.frames).pred_nrmse < evaluate(
        without_t, te.frames
    ).pred_nrmse


# --------------------------------------------------------------------------
# 3. the sparsity threshold
# --------------------------------------------------------------------------
def test_threshold_zero_silences_nothing():
    net = small_net(theta=0.0)
    stats = evaluate(net, small_signal(40).frames)
    assert stats.silenced_frac == 0.0


def test_higher_threshold_silences_more_and_converte_menos():
    sig = small_signal(300)
    tr, te = sig.split(0.8)
    net = small_net()
    train(net, tr.frames, epochs=30)

    sweep = [evaluate_with_theta(net, te.frames, th) for th in (0.0, 0.02, 0.1)]
    silenced = [s.silenced_frac for s in sweep]
    adc = [s.adc_frac for s in sweep]
    assert silenced[0] < silenced[1] < silenced[2]
    assert adc[0] > adc[1] > adc[2]
    # the threshold is an inference knob: it does not touch the weights
    assert net.cfg.theta == PCConfig(**SMALL).theta


def test_a_huge_threshold_silences_everything_and_the_frame_is_free():
    net = small_net(theta=10.0)
    for trace in net.run(small_signal(20).frames, learn=False):
        assert trace.iters == 0
        assert trace.exit_reason == EXIT_EXPLAINED
        assert trace.adc_conversions == 0
        assert trace.macs_up == 0


def test_sparsity_saves_upward_macs():
    sig = small_signal(200)
    tr, te = sig.split(0.8)
    net = small_net()
    train(net, tr.frames, epochs=20)
    dense = evaluate_with_theta(net, te.frames, 0.0)
    sparse = evaluate_with_theta(net, te.frames, 0.05)
    assert sparse.mac_up_frac < dense.mac_up_frac


# --------------------------------------------------------------------------
# 4. compute is proportional to surprise
# --------------------------------------------------------------------------
def test_surprising_frames_cost_more_than_banal_ones():
    sig = small_signal(300)
    tr, _ = sig.split(0.8)
    net = small_net()
    train(net, tr.frames, epochs=30)

    ev = small_signal(200, seed=42, n_events=12)
    traces = net.run(ev.frames, learn=False, reset=True)
    mask = np.zeros(len(traces), dtype=bool)
    mask[ev.event_frames] = True
    banal = [t for t, m in zip(traces, mask) if not m]
    events = [t for t, m in zip(traces, mask) if m]

    mean = lambda ts, f: float(np.mean([f(t) for t in ts]))  # noqa: E731
    assert mean(events, lambda t: t.open_loop_surprise) > 2 * mean(
        banal, lambda t: t.open_loop_surprise
    )
    assert mean(events, lambda t: t.adc_conversions) > mean(
        banal, lambda t: t.adc_conversions
    )
    assert mean(events, lambda t: t.iters) > mean(banal, lambda t: t.iters)


def test_a_perfectly_predicted_input_becomes_almost_free():
    """A repeated constant: once learned, it almost stops costing anything."""
    net = small_net(theta=0.02)
    x = np.full(32, 0.3, dtype=F)
    traces = net.run(np.tile(x, (400, 1)), learn=True)
    first, last = traces[0], traces[-1]
    assert last.iters < first.iters
    assert last.exit_reason == EXIT_EXPLAINED
    assert last.adc_conversions < 0.2 * first.adc_conversions
    assert last.open_loop_surprise < 0.02 * first.open_loop_surprise


def test_ceiling_is_reported_as_such():
    net = small_net(max_iters=1, settle_min_gain=0.0, settle_tol=0.0, theta=0.0)
    trace = net.step(np.full(32, 0.9, dtype=F), learn=False)
    assert trace.exit_reason == EXIT_CEILING
    assert not trace.early_exit


# --------------------------------------------------------------------------
# fast path (section 4 of CONTEXTO): a short route in parallel with the deep one
# --------------------------------------------------------------------------
def test_fast_path_adds_a_direct_route_and_its_parameters():
    plain, fast = small_net(), small_net(fast_path=True)
    assert plain.A0 is None
    assert fast.A0 is not None and fast.A0.shape == (32, 32)


def test_prediction_is_the_sum_of_the_two_routes():
    """ẑ_0 = fast path + hierarchy. The decomposition is additive on purpose:
    that way the error each route sees is the same, and it is the residual of
    the other."""
    net = small_net(fast_path=True)
    sig = small_signal(60)
    net.run(sig.frames[:20], learn=True)

    deep = net.layers[0].predict(
        net._combine_priors(1, net.layers[1].predict(net._temporal_prior()))
    ).copy()
    fast = net.A0 @ net._z_prev[0]
    assert np.allclose(net.predict_next(), deep + fast, atol=1e-5)


def test_each_route_learns_from_its_own_error():
    """The fast path learns from (x − A₀·x_prev), not from the hierarchy's residual.

    Training both on the same residual makes them fight over the same signal
    with nothing to assign credit - measured on ETTm1: 0.230 -> 0.724.
    """
    net = small_net(fast_path=True)
    A0 = net.A0.copy()
    net.run(small_signal(60).frames, learn=True)
    assert not np.array_equal(A0, net.A0)

    # The fast path, on its own (hierarchy frozen), has to close its error.
    # The step is normalized by ‖x‖², so fast_path_lr is the *fraction* of the
    # error corrected per frame and does not depend on the signal scale.
    net2 = small_net(fast_path=True, fast_path_lr=0.5, w_lr=0.0, a_lr=0.0)
    x_prev = np.full(32, 0.5, dtype=F)
    x_now = np.full(32, -0.5, dtype=F)
    err = lambda: float(np.abs(x_now - net2.A0 @ x_prev).sum())  # noqa: E731

    net2.step(x_prev, learn=False)
    before = err()
    for _ in range(30):
        net2.step(x_now, learn=True)
        net2._z_prev[0][:] = x_prev  # keep the (before, after) pair fixed
    assert err() < 0.1 * before


def test_fast_path_is_frozen_when_not_learning():
    net = small_net(fast_path=True)
    net.run(small_signal(40).frames, learn=True)
    A0 = net.A0.copy()
    net.run(small_signal(40).frames, learn=False)
    assert np.array_equal(A0, net.A0)


def test_hierarchy_still_contributes_with_the_fast_path_on():
    """The fast path cannot make the hierarchy decorative: if the deep
    weights change, the prediction has to change."""
    net = small_net(fast_path=True)
    net.run(small_signal(80).frames, learn=True)
    before = net.predict_next().copy()
    net.layers[0].W *= 0.5
    net.layers[0].refresh_device()
    assert not np.allclose(before, net.predict_next(), atol=1e-4)


def test_fast_path_step_is_scale_invariant():
    """The fast path's rate cannot depend on the signal scale.

    With an unnormalized step, `lr` has to stay below 1/‖x‖², which changes
    with the signal - and an `lr` that works on one dataset diverges on
    another. That is how the fast path went from 0.208 to 0.899 on ETTm1.
    With the normalized step, multiplying the signal by 100 does not change
    the *relative* error.
    """
    def relative_error(scale):
        net = small_net(fast_path=True, fast_path_lr=0.2, w_lr=0.0, a_lr=0.0)
        x_prev = np.full(32, 0.5 * scale, dtype=F)
        x_now = np.full(32, -0.5 * scale, dtype=F)
        net.step(x_prev, learn=False)
        for _ in range(40):
            net.step(x_now, learn=True)
            net._z_prev[0][:] = x_prev
        return float(np.abs(x_now - net.A0 @ x_prev).sum()) / float(np.abs(x_now).sum())

    small, large = relative_error(1.0), relative_error(100.0)
    assert small < 0.2
    assert large == pytest.approx(small, rel=0.05)


def test_the_critic_retracts_a_harmful_update():
    """Front 4: the local update is provisional until the next frame
    validates it. If the surprise spikes above the usual, the weights go back
    to the PRE-update state - hence the snapshot having to be taken before
    learning (the first version took it afterwards, and retracting undid
    nothing).
    """
    net = small_net(critic_retract=1.05, w_lr=0.5)
    sig = small_signal(80)
    net.run(sig.frames[:60], learn=True)  # establish the EMA

    W_before = [lay.W.copy() for lay in net.layers]
    net.step(sig.frames[60], learn=True)          # learns (provisional)
    changed = not all(np.array_equal(a, lay.W)
                      for a, lay in zip(W_before, net.layers))
    assert changed
    W_post_learn = [lay.W.copy() for lay in net.layers]

    net.step(np.full(32, 5.0, dtype=F), learn=False)  # huge surprise -> veto
    reverted = all(np.array_equal(a, lay.W)
                   for a, lay in zip(W_before, net.layers))
    kept = all(np.array_equal(a, lay.W)
               for a, lay in zip(W_post_learn, net.layers))
    assert reverted and not kept
