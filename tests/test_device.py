"""Passo 2: o substrato. Ternarização, variabilidade, ruído, ADC."""

import numpy as np
import pytest

from pcnet import PCConfig, PCNetwork, make_signal
from pcnet.device import AnalogArray, DeviceModel, quantize_adc, ternarize
from pcnet.dtypes import F
from pcnet.train import evaluate, train

SMALL = dict(sizes=(32, 16, 8))
TERNARY = DeviceModel(ternary=True)


def small_net(**kw):
    return PCNetwork(PCConfig(**{**SMALL, **kw}))


def small_signal(n=200, seed=1):
    return make_signal(n_frames=n, frame_len=32, seed=seed)


def trained(device=None, seed=0, epochs=25):
    net = small_net(seed=seed)
    if device is not None:
        net.attach_device(device)
    train(net, small_signal(300).split(0.8)[0].frames, epochs=epochs)
    return net


# --------------------------------------------------------------------------
# ternarização
# --------------------------------------------------------------------------
def test_ternary_weights_take_three_values():
    W = np.random.default_rng(0).standard_normal((16, 8)).astype(F)
    T, scale = ternarize(W)
    assert set(np.unique(T).tolist()) <= {-1.0, 0.0, 1.0}
    assert scale.shape == (16, 1)
    assert np.all(scale > 0)


def test_ternary_approximates_the_float_weights():
    W = np.random.default_rng(1).standard_normal((32, 16)).astype(F)
    T, scale = ternarize(W)
    err = np.linalg.norm(W - T * scale) / np.linalg.norm(W)
    assert err < 0.6  # 3 valores por peso não fazem milagres, mas seguem o sinal
    assert np.all(np.sign(T)[T != 0] == np.sign(W)[T != 0])


def test_higher_threshold_zeroes_more_devices():
    W = np.random.default_rng(2).standard_normal((32, 16)).astype(F)
    sparse = [np.mean(ternarize(W, threshold=t)[0] == 0) for t in (0.25, 0.75, 1.5)]
    assert sparse[0] < sparse[1] < sparse[2]


def test_per_row_scale_beats_a_single_global_gain():
    """Linhas com escalas muito diferentes: um ganho só não chega."""
    rng = np.random.default_rng(3)
    W = rng.standard_normal((8, 16)).astype(F)
    W[:4] *= 100.0
    err = {}
    for per_row in (True, False):
        T, scale = ternarize(W, per_row=per_row)
        err[per_row] = np.linalg.norm(W - T * scale) / np.linalg.norm(W)
    assert err[True] < err[False]


def test_all_zero_weights_do_not_produce_nan():
    T, scale = ternarize(np.zeros((4, 4), dtype=F))
    assert np.all(T == 0) and np.all(scale == 0)
    assert np.isfinite(T * scale).all()


# --------------------------------------------------------------------------
# o crossbar
# --------------------------------------------------------------------------
def test_programming_variability_is_static_not_noise():
    """O desvio de cada dispositivo é amostrado uma vez. Programar duas vezes
    os mesmos pesos tem de dar exatamente o mesmo crossbar."""
    W = np.random.default_rng(4).standard_normal((16, 8)).astype(F)
    arr = AnalogArray(W.shape, DeviceModel(sigma_rel=0.2, sigma_abs=0.01), seed=0)
    assert np.array_equal(arr.program(W), arr.program(W))


def test_read_noise_is_dynamic():
    """O ruído de leitura, esse, muda a cada MAC."""
    arr = AnalogArray((16, 8), DeviceModel(read_sigma=0.1), seed=0)
    out = np.ones(16, dtype=F)
    assert not np.array_equal(arr.read(out.copy()), arr.read(out.copy()))


def test_same_seed_same_device_different_seed_different_device():
    W = np.random.default_rng(5).standard_normal((16, 8)).astype(F)
    m = DeviceModel(sigma_rel=0.2)
    a = AnalogArray(W.shape, m, seed=1).program(W)
    b = AnalogArray(W.shape, m, seed=1).program(W)
    c = AnalogArray(W.shape, m, seed=2).program(W)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_stuck_devices_are_exactly_zero():
    W = np.ones((32, 32), dtype=F)
    out = AnalogArray(W.shape, DeviceModel(stuck_frac=0.5), seed=0).program(W)
    frac = float(np.mean(out == 0))
    assert 0.4 < frac < 0.6


def test_an_ideal_device_changes_nothing():
    W = np.random.default_rng(6).standard_normal((16, 8)).astype(F)
    assert DeviceModel().is_ideal
    assert np.array_equal(AnalogArray(W.shape, DeviceModel(), 0).program(W), W)


# --------------------------------------------------------------------------
# ADC
# --------------------------------------------------------------------------
def test_adc_zero_bits_is_a_passthrough():
    x = np.linspace(-1, 1, 33).astype(F)
    assert np.array_equal(quantize_adc(x, 0, 1.0), x)


def test_adc_saturates_and_discretises():
    x = np.linspace(-5, 5, 501).astype(F)
    q = quantize_adc(x, 4, 1.0)
    assert q.max() <= 1.0 + 1e-6 and q.min() >= -1.0 - 1e-6
    assert len(np.unique(q)) <= (1 << 4)


@pytest.mark.parametrize("bits", [8, 6, 4])
def test_more_adc_bits_means_less_error(bits):
    rng = np.random.default_rng(7)
    x = rng.uniform(-1, 1, 4096).astype(F)
    err = lambda b: float(np.abs(quantize_adc(x, b, 1.0) - x).mean())  # noqa: E731
    assert err(bits + 2) < err(bits)


# --------------------------------------------------------------------------
# integração com a rede
# --------------------------------------------------------------------------
def test_attach_and_detach_restores_the_float_network():
    """Os pesos voltam exatamente; o agregado volta a menos de um epsilon.

    Trama a trama não volta, e isso é uma propriedade do sistema, não um bug —
    ver `test_exit_decision_is_chaotic_but_aggregates_are_stable`.
    """
    net = trained()
    sig = small_signal(60)
    W_before = [W.copy() for W in net.weights]
    before = np.array([t.pred_rmse for t in net.run(sig.frames, learn=False, reset=True)])
    net.attach_device(DeviceModel(ternary=True, sigma_rel=0.2))
    net.detach_device()
    after = np.array([t.pred_rmse for t in net.run(sig.frames, learn=False, reset=True)])

    assert all(np.array_equal(a, b) for a, b in zip(W_before, net.weights))
    assert all(lay.W_eff is lay.W for lay in net.layers)
    assert before.mean() == pytest.approx(after.mean(), abs=0.02)


def test_an_ideal_device_changes_nothing_that_matters():
    net = trained()
    sig = small_signal(60)
    before = evaluate(net, sig.frames).pred_nrmse
    net.attach_device(DeviceModel())
    assert np.allclose(net.layers[0].W_eff, net.weights[0])
    assert evaluate(net, sig.frames).pred_nrmse == pytest.approx(before, abs=0.02)


def test_learning_updates_the_float_shadow_and_reprograms_the_crossbar():
    net = small_net()
    net.attach_device(DeviceModel(ternary=True))
    W_float = net.weights[0].copy()
    W_dev = net.layers[0].W_eff.copy()
    net.run(small_signal(40).frames, learn=True)

    assert not np.array_equal(W_float, net.weights[0])  # o shadow moveu-se
    assert not np.array_equal(W_dev, net.layers[0].W_eff)  # o crossbar seguiu
    assert set(np.unique(np.sign(net.layers[0].W_eff)).tolist()) <= {-1.0, 0.0, 1.0}


def test_training_on_the_crossbar_beats_quantising_at_the_end():
    """A afirmação principal do passo 2: QAT >> PTQ."""
    sig = small_signal(300)
    tr, te = sig.split(0.8)

    float_net = small_net()
    train(float_net, tr.frames, epochs=25)
    float_net.attach_device(TERNARY)
    ptq = evaluate(float_net, te.frames).pred_nrmse

    qat_net = small_net()
    qat_net.attach_device(TERNARY)
    train(qat_net, tr.frames, epochs=25)
    qat = evaluate(qat_net, te.frames).pred_nrmse

    assert qat < ptq


# --------------------------------------------------------------------------
# estabilidade
# --------------------------------------------------------------------------
def test_power_iteration_matches_the_true_largest_singular_value():
    net = trained(device=TERNARY)
    for lay in net.layers:
        true = float(np.linalg.svd(lay.W_eff, compute_uv=False)[0])
        assert abs(lay.sigma_max - true) / true < 0.05


def test_changing_device_re_estimates_sigma_max():
    """Uma só iteração de potência subestima σ_max ao trocar de crossbar — e um
    σ_max subestimado dá um passo grande demais. Foi assim que divergiu."""
    net = trained(device=TERNARY)
    net.attach_device(DeviceModel(ternary=True, sigma_rel=0.5, seed=3))
    for lay in net.layers:
        true = float(np.linalg.svd(lay.W_eff, compute_uv=False)[0])
        assert abs(lay.sigma_max - true) / true < 0.05


def test_adaptive_step_respects_the_stability_bound():
    net = trained(device=TERNARY)
    for level in range(1, net.L + 1):
        step = net._z_lr_for(level)
        assert step <= net.cfg.z_lr + 1e-9
        assert step <= net.cfg.z_lr_safety * net.layers[level - 1].max_stable_z_lr() + 1e-9


def test_fixed_step_can_be_forced_back_on():
    net = trained(device=TERNARY, seed=0)
    net.cfg = PCConfig(**SMALL, adaptive_z_lr=False)
    assert all(net._z_lr_for(l) == net.cfg.z_lr for l in range(1, net.L + 1))


# --------------------------------------------------------------------------
# um facto estrutural que custou a descobrir
# --------------------------------------------------------------------------
def test_settling_budget_below_the_depth_predicts_nothing():
    """O erro sensorial sobe um nível por iteração. Com menos iterações do que
    níveis, o topo nunca é corrigido, nunca sai de zero, e a rede prevê zero —
    o que a torna exatamente tão boa como não existir (NRMSE 1.0)."""
    sig = small_signal(300)
    tr, te = sig.split(0.8)
    depth = len(SMALL["sizes"]) - 1

    starved = small_net(max_iters=depth - 1, settle_min_gain=0.0)
    train(starved, tr.frames, epochs=15)
    assert evaluate(starved, te.frames).pred_nrmse == pytest.approx(1.0, abs=1e-6)

    ok = small_net(max_iters=depth, settle_min_gain=0.0)
    train(ok, tr.frames, epochs=15)
    assert evaluate(ok, te.frames).pred_nrmse < 0.9


def test_exit_decision_is_chaotic_but_aggregates_are_stable():
    """Uma perturbação ínfima no passo muda tramas individuais, não a média.

    O critério de saída é uma decisão discreta: uma diferença de 1e-4 no passo
    de assentamento chega para o laço parar uma iteração antes ou depois, e uma
    iteração muda o estado do topo, que atravessa para a trama seguinte. Daí
    que 0.01% de diferença no passo dê ~0.1 de diferença numa trama concreta.

    Consequência prática, e é por isso que este teste existe: validar o
    inferidor em C trama a trama, ou pelo nº de iterações, é validar ruído. O
    que se compara são agregados e tolerâncias.
    """
    sig = small_signal(300)
    tr, te = sig.split(0.8)
    net = small_net()
    train(net, tr.frames, epochs=25)

    a = np.array([t.pred_rmse for t in net.run(te.frames, learn=False, reset=True)])
    net.cfg = PCConfig(**SMALL, z_lr=net.cfg.z_lr * 1.0001)
    b = np.array([t.pred_rmse for t in net.run(te.frames, learn=False, reset=True)])

    assert np.abs(a - b).max() > 10 * np.abs(a.mean() - b.mean())
    assert a.mean() == pytest.approx(b.mean(), abs=0.05)
