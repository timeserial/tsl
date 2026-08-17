"""The Python-training / C-inference boundary.

Python trains; C infers. This module is the only place where the two touch,
and it writes three things:

  * `model.npz`  - weights + config, to reload in Python.
  * `model.h`    - the same weights as static C arrays (no malloc).
  * `golden.h`   - input frames and the outputs Python produced, for the C
                   to validate with asserts (step 3 of the plan).

As long as the weights are float32 this is mere convenience; once they are
ternary {-1,0,1} it becomes the artifact that goes to the crossbar.
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
    """A float literal valid in C.

    `%.8g` of 0.0 gives "0", and "0f" is an invalid octal constant - the
    header stops compiling. A dot or an exponent is always guaranteed.
    """
    v = float(v)
    if not np.isfinite(v):
        raise ValueError(f"peso não finito ({v}): o modelo não é exportável")
    s = f"{v:.8g}"
    if "." not in s and "e" not in s and "E" not in s:
        s += ".0"
    return s + "f"


def _c_rows(values: np.ndarray, indent: str) -> list[str]:
    """A sequence of float literals, broken into lines of ~76 columns."""
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
    """Static C array. 2D gets nested braces (otherwise the compiler warns)."""
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
    """Writes the weights and the config as a self-contained C header."""
    cfg = net.cfg
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    guard = path.name.upper().replace(".", "_").replace("-", "_")
    out = [
        "/* Gerado por pcnet.export - não editar à mão. */",
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        f"#define PC_N_LEVELS {cfg.n_levels}",
        f"#define PC_N_WEIGHTS {cfg.n_weights}",
        # Largest level: gives the size of the static arrays, no malloc.
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
        "   para que o C não tenha de estimar σ_max - os pesos são fixos no",
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
    """Reference vectors: the C must reproduce this bit for bit (up to ε).

    Runs `n` frames *without learning* from a zeroed state and stores, per
    frame, the open-loop prediction and the top's latent state after
    settling - the two quantities that reveal any divergence.
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
        "/* Gerado por pcnet.export - vetores de referência do Python. */",
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
