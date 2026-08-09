import shutil
import subprocess

import numpy as np
import pytest

from pcnet import PCConfig, PCNetwork, make_signal
from pcnet.export import _c_float, load_npz, save_npz, write_c_header, write_golden
from pcnet.train import train

CFG = dict(sizes=(16, 8, 4))


def trained(seed=0):
    net = PCNetwork(PCConfig(seed=seed, **CFG))
    sig = make_signal(n_frames=120, frame_len=16, seed=2)
    train(net, sig.frames, epochs=10)
    return net, sig


def test_npz_roundtrip_preserves_the_model(tmp_path):
    net, sig = trained()
    path = save_npz(tmp_path / "model.npz", net)
    clone = load_npz(path)

    assert clone.cfg == net.cfg
    for a, b in zip(net.weights, clone.weights):
        assert np.array_equal(a, b)
    assert np.array_equal(net.A, clone.A)

    # e, o que interessa, produz as mesmas previsões a partir do mesmo estado.
    # A tolerância é porque σ_max vem de uma iteração de potência com critério
    # de paragem relativo: converge à mesma resposta, não aos mesmos bits.
    a = np.array([t.pred_rmse for t in net.run(sig.frames[:20], learn=False, reset=True)])
    b = np.array([t.pred_rmse for t in clone.run(sig.frames[:20], learn=False, reset=True)])
    assert np.allclose(a, b, rtol=1e-3, atol=1e-4)


def test_c_header_carries_config_and_weights(tmp_path):
    net, _ = trained()
    text = write_c_header(tmp_path / "model.h", net).read_text()

    assert "#define PC_N_LEVELS 3" in text
    assert "static const int pc_sizes[PC_N_LEVELS] = {16, 8, 4};" in text
    for name, W in (("pc_W0", net.weights[0]), ("pc_W1", net.weights[1])):
        assert f"static const float {name}[{W.shape[0]}][{W.shape[1]}]" in text
    assert "static const float pc_A[4][4]" in text
    # tantos literais float quantos os pesos
    assert text.count("f,") + text.count("f\n") >= sum(W.size for W in net.weights)


def test_golden_vectors_match_what_python_did(tmp_path):
    net, sig = trained()
    text = write_golden(tmp_path / "golden.h", net, sig.frames, n=4).read_text()
    assert "#define PC_GOLDEN_N 4" in text
    assert "pc_golden_input[4][16]" in text
    assert "pc_golden_pred[4][16]" in text
    assert "pc_golden_top[4][4]" in text
    assert "pc_golden_iters[PC_GOLDEN_N]" in text


@pytest.mark.parametrize(
    "value, expected",
    [(0.0, "0.0f"), (2.0, "2.0f"), (-1.0, "-1.0f"), (1e-6, "1e-06f"), (0.5, "0.5f")],
)
def test_float_literals_are_valid_c(value, expected):
    """"0f" é uma constante octal inválida — foi assim que o header parou de
    compilar da primeira vez."""
    assert _c_float(value) == expected


def test_non_finite_weights_are_refused():
    net, _ = trained()
    net.weights[0][0, 0] = np.inf
    with pytest.raises(ValueError, match="não finito"):
        write_c_header("/dev/null", net)


@pytest.mark.skipif(shutil.which("cc") is None, reason="sem compilador C")
def test_generated_headers_compile_clean(tmp_path):
    """O destino é C simples: os headers têm de compilar sem avisos."""
    net, sig = trained()
    write_c_header(tmp_path / "model.h", net)
    write_golden(tmp_path / "golden.h", net, sig.frames, n=4)
    (tmp_path / "main.c").write_text(
        '#include "model.h"\n'
        '#include "golden.h"\n'
        "int main(void) {\n"
        "    return (pc_sizes[0] == 16 && pc_W0[0][0] == pc_W0[0][0]\n"
        "            && pc_golden_iters[0] >= 0 && PC_GOLDEN_N == 4) ? 0 : 1;\n"
        "}\n"
    )
    proc = subprocess.run(
        ["cc", "-std=c89", "-Wall", "-Wextra", "-Werror", "-pedantic",
         str(tmp_path / "main.c"), "-o", str(tmp_path / "a.out")],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    assert subprocess.run([str(tmp_path / "a.out")]).returncode == 0


def test_golden_is_reproducible(tmp_path):
    """O C vai comparar contra isto; tem de ser estável entre execuções.

    `write_golden` faz reset ao estado antes de correr, logo escrever duas
    vezes seguidas — mesmo com o estado sujo pela primeira — tem de dar o
    mesmo ficheiro.
    """
    net, sig = trained()
    path = tmp_path / "golden.h"
    a = write_golden(path, net, sig.frames, n=4).read_text()
    b = write_golden(path, net, sig.frames, n=4).read_text()
    assert a == b
