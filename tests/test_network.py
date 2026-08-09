"""As quatro afirmações do passo 1, cada uma como um teste que pode falhar."""

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
# mecânica
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
    """A rede é online: medir não pode alterar o que está a ser medido."""
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
    """ε_0 na primeira avaliação é exatamente x − predict_next() do passo anterior."""
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
# 1. a dinâmica de assentamento converge
# --------------------------------------------------------------------------
def test_settling_decreases_the_energy():
    """Cada iteração baixa a energia — exceto, quando muito, a última.

    O assentamento é descida de gradiente com passo fixo: perto do limite de
    estabilidade a última avaliação pode subir um pouco. Se subir, é essa
    subida que faz o critério de ganho mínimo parar o laço, por isso só pode
    aparecer na última posição.
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
    """Deixar assentar mais reduz a energia final. Sem isto, iterar é inútil."""
    sig = small_signal(80)
    shallow = small_net(max_iters=1, settle_min_gain=0.0, theta=0.0)
    deep = small_net(max_iters=10, settle_min_gain=0.0, theta=0.0)
    s1 = summarize(shallow.run(sig.frames, learn=False), 1, shallow.eps_per_pass)
    s2 = summarize(deep.run(sig.frames, learn=False), 10, deep.eps_per_pass)
    assert s2.mean_final_surprise < s1.mean_final_surprise


# --------------------------------------------------------------------------
# 2. a aprendizagem local funciona
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
    """Sem transição temporal o topo não tem como prever a trama seguinte."""
    sig = small_signal(300)
    tr, te = sig.split(0.8)
    with_t, without_t = small_net(), small_net(use_transition=False)
    train(with_t, tr.frames, epochs=40)
    train(without_t, tr.frames, epochs=40)
    assert evaluate(with_t, te.frames).pred_nrmse < evaluate(
        without_t, te.frames
    ).pred_nrmse


# --------------------------------------------------------------------------
# 3. o limiar de esparsidade
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
    # o limiar é um botão de inferência: não mexe nos pesos
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
# 4. o compute é proporcional à surpresa
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
    """Constante repetida: depois de aprendida, quase deixa de custar."""
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
# via rápida (secção 4 do CONTEXTO): uma via curta em paralelo com a profunda
# --------------------------------------------------------------------------
def test_fast_path_adds_a_direct_route_and_its_parameters():
    plain, fast = small_net(), small_net(fast_path=True)
    assert plain.A0 is None
    assert fast.A0 is not None and fast.A0.shape == (32, 32)


def test_prediction_is_the_sum_of_the_two_routes():
    """ẑ_0 = via rápida + hierarquia. A decomposição é aditiva de propósito:
    assim o erro que cada via vê é o mesmo, e é o resíduo da outra."""
    net = small_net(fast_path=True)
    sig = small_signal(60)
    net.run(sig.frames[:20], learn=True)

    deep = net.layers[0].predict(
        net._combine_priors(1, net.layers[1].predict(net._temporal_prior()))
    ).copy()
    fast = net.A0 @ net._z_prev[0]
    assert np.allclose(net.predict_next(), deep + fast, atol=1e-5)


def test_each_route_learns_from_its_own_error():
    """A via rápida aprende com (x − A₀·x_ant), não com o resíduo da hierarquia.

    Treinar as duas com o mesmo resíduo faz com que disputem o mesmo sinal sem
    nada que atribua crédito — medido em ETTm1: 0.230 -> 0.724.
    """
    net = small_net(fast_path=True)
    A0 = net.A0.copy()
    net.run(small_signal(60).frames, learn=True)
    assert not np.array_equal(A0, net.A0)

    # A via rápida, sozinha (hierarquia congelada), tem de fechar o seu erro.
    # O passo é normalizado por ‖x‖², logo fast_path_lr é a *fração* do erro
    # corrigida por trama e não depende da escala do sinal.
    net2 = small_net(fast_path=True, fast_path_lr=0.5, w_lr=0.0, a_lr=0.0)
    x_prev = np.full(32, 0.5, dtype=F)
    x_now = np.full(32, -0.5, dtype=F)
    err = lambda: float(np.abs(x_now - net2.A0 @ x_prev).sum())  # noqa: E731

    net2.step(x_prev, learn=False)
    before = err()
    for _ in range(30):
        net2.step(x_now, learn=True)
        net2._z_prev[0][:] = x_prev  # manter o par (antes, depois) fixo
    assert err() < 0.1 * before


def test_fast_path_is_frozen_when_not_learning():
    net = small_net(fast_path=True)
    net.run(small_signal(40).frames, learn=True)
    A0 = net.A0.copy()
    net.run(small_signal(40).frames, learn=False)
    assert np.array_equal(A0, net.A0)


def test_hierarchy_still_contributes_with_the_fast_path_on():
    """A via rápida não pode tornar a hierarquia decorativa: se os pesos
    profundos mudarem, a previsão tem de mudar."""
    net = small_net(fast_path=True)
    net.run(small_signal(80).frames, learn=True)
    before = net.predict_next().copy()
    net.layers[0].W *= 0.5
    net.layers[0].refresh_device()
    assert not np.allclose(before, net.predict_next(), atol=1e-4)


def test_fast_path_step_is_scale_invariant():
    """A taxa da via rápida não pode depender da escala do sinal.

    Com passo não normalizado, `lr` tem de ficar abaixo de 1/‖x‖², que muda
    com o sinal — e um `lr` que funciona num conjunto diverge noutro. Foi assim
    que a via rápida passou de 0.208 para 0.899 em ETTm1. Com o passo
    normalizado, multiplicar o sinal por 100 não muda o erro *relativo*.
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
