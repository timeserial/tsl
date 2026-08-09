"""A fronteira treino-Python / inferência-C.

O Python treina; o C infere. Este módulo é o único sítio onde os dois se
tocam, e escreve três coisas:

  * `model.npz`  — pesos + config, para recarregar em Python.
  * `model.h`    — os mesmos pesos como arrays estáticos C (sem malloc).
  * `golden.h`   — tramas de entrada e as saídas que o Python produziu, para o
                   C validar com asserts (passo 3 do plano).

Enquanto os pesos forem float32 isto é só conveniência; quando forem
ternários {-1,0,1} passa a ser o artefacto que vai para o crossbar.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import PCConfig
from .dtypes import F
from .network import PCNetwork


# ---------------------------------------------------------------------------
# NumPy
# ---------------------------------------------------------------------------
def save_npz(path: str | Path, net: PCNetwork) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        config=np.array(json.dumps(net.cfg.to_dict())),
        **net.state_dict(),
    )
    return path


def load_npz(path: str | Path) -> PCNetwork:
    data = np.load(Path(path), allow_pickle=False)
    cfg = PCConfig.from_dict(json.loads(str(data["config"])))
    net = PCNetwork(cfg)
    net.load_state_dict({k: data[k] for k in data.files if k != "config"})
    return net


# ---------------------------------------------------------------------------
# C
# ---------------------------------------------------------------------------
def _c_float(v: float) -> str:
    """Literal float válido em C.

    `%.8g` de 0.0 dá "0", e "0f" é uma constante octal inválida — o header
    deixa de compilar. Garante-se sempre um ponto ou um expoente.
    """
    v = float(v)
    if not np.isfinite(v):
        raise ValueError(f"peso não finito ({v}): o modelo não é exportável")
    s = f"{v:.8g}"
    if "." not in s and "e" not in s and "E" not in s:
        s += ".0"
    return s + "f"


def _c_rows(values: np.ndarray, indent: str) -> list[str]:
    """Uma sequência de literais float, quebrada em linhas de ~76 colunas."""
    lines: list[str] = []
    line = indent
    for v in values:
        tok = _c_float(v) + ","
        if len(line) + len(tok) > 76 and line.strip():
            lines.append(line.rstrip())
            line = indent
        line += tok + " "
    if line.strip():
        lines.append(line.rstrip())
    return lines


def _c_array(name: str, arr: np.ndarray) -> str:
    """Array C estático. 2D leva chavetas aninhadas (senão o compilador avisa)."""
    arr = np.asarray(arr, dtype=F)
    if arr.ndim > 2:
        raise ValueError(f"{name}: só 1D e 2D (recebido {arr.ndim}D)")
    dims = "".join(f"[{d}]" for d in arr.shape)

    body: list[str] = []
    if arr.ndim == 1:
        body = _c_rows(arr, "    ")
    else:
        for row in arr:
            inner = _c_rows(row, "        ")
            inner[-1] = inner[-1].rstrip(",")
            body.append("    {")
            body.extend(inner)
            body.append("    },")
    body[-1] = body[-1].rstrip(",")
    return f"static const float {name}{dims} = {{\n" + "\n".join(body) + "\n};\n"


def write_c_header(path: str | Path, net: PCNetwork) -> Path:
    """Escreve os pesos e a config como header C auto-suficiente."""
    cfg = net.cfg
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    guard = path.name.upper().replace(".", "_").replace("-", "_")
    out = [
        "/* Gerado por pcnet.export — não editar à mão. */",
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        f"#define PC_N_LEVELS {cfg.n_levels}",
        f"#define PC_N_WEIGHTS {cfg.n_weights}",
        # Maior nível: dá o tamanho dos arrays estáticos, sem malloc.
        f"#define PC_MAX_SIZE {max(cfg.sizes)}",
        f"#define PC_MAX_ITERS {cfg.max_iters}",
        f"#define PC_Z_LR {_c_float(cfg.z_lr)}",
        f"#define PC_THETA {_c_float(cfg.theta)}",
        f"#define PC_Z_CLIP {_c_float(cfg.z_clip)}",
        f"#define PC_Z_MAX {_c_float(cfg.z_max)}",
        f"#define PC_SETTLE_TOL {_c_float(cfg.settle_tol)}",
        f"#define PC_SETTLE_MIN_GAIN {_c_float(cfg.settle_min_gain)}",
        f"#define PC_USE_TRANSITION {1 if cfg.use_transition else 0}",
        "",
        "static const int pc_sizes[PC_N_LEVELS] = {"
        + ", ".join(str(n) for n in cfg.sizes)
        + "};",
        "",
        "/* Passo de assentamento por nível (1..L), já limitado pela",
        "   estabilidade: min(z_lr, safety·2/(1+σ_max²)). Vai resolvido daqui",
        "   para que o C não tenha de estimar σ_max — os pesos são fixos no",
        "   destino, logo o limite também é. */",
        _c_array(
            "pc_z_lr_level",
            np.array([net._z_lr_for(l) for l in range(1, net.L + 1)], dtype=F),
        ),
    ]
    for l, W in enumerate(net.weights):
        out.append(f"/* W{l}: nível {l + 1} ({W.shape[1]}) -> nível {l} ({W.shape[0]}) */")
        out.append(_c_array(f"pc_W{l}", W))
    out.append("/* A: transição temporal do topo */")
    out.append(_c_array("pc_A", net.A))
    out.append(f"#endif /* {guard} */")

    path.write_text("\n".join(out) + "\n")
    return path


def write_golden(
    path: str | Path, net: PCNetwork, frames: np.ndarray, n: int = 8
) -> Path:
    """Vetores de referência: o C tem de reproduzir isto bit a bit (a menos de ε).

    Corre `n` tramas *sem aprender* a partir de um estado zerado e guarda, por
    trama, a previsão em malha aberta e o estado latente do topo depois do
    assentamento — as duas quantidades que revelam qualquer divergência.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.asarray(frames, dtype=F)[:n]

    net.reset()
    preds, tops, iters = [], [], []
    for frame in frames:
        preds.append(net.predict_next())
        trace = net.step(frame, learn=False)
        tops.append(net.z[net.L].copy())
        iters.append(trace.iters)

    guard = path.name.upper().replace(".", "_").replace("-", "_")
    out = [
        "/* Gerado por pcnet.export — vetores de referência do Python. */",
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        f"#define PC_GOLDEN_N {len(frames)}",
        "",
        _c_array("pc_golden_input", frames),
        _c_array("pc_golden_pred", np.array(preds, dtype=F)),
        _c_array("pc_golden_top", np.array(tops, dtype=F)),
        "static const int pc_golden_iters[PC_GOLDEN_N] = {"
        + ", ".join(str(i) for i in iters)
        + "};",
        "",
        f"#endif /* {guard} */",
    ]
    path.write_text("\n".join(out) + "\n")
    return path
