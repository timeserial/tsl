"""Ativações e a sua derivada.

Mantidas deliberadamente pobres: `tanh` e identidade. Tudo o que aqui estiver
tem de ter uma tradução trivial para C com inteiros (tanh -> LUT de 256
entradas), caso contrário não cabe no crossbar.
"""

from __future__ import annotations

import numpy as np

from .dtypes import F

# --------------------------------------------------------------------------
# f(a) e f'(a). A derivada é escrita em função de *a* (a pré-ativação) para
# que a assinatura seja igual em todas as ativações; em C guardamos apenas
# `a` por nível, um array estático por camada.
# --------------------------------------------------------------------------


def identity(a: np.ndarray) -> np.ndarray:
    return a


def identity_prime(a: np.ndarray) -> np.ndarray:
    return np.ones_like(a)


def tanh(a: np.ndarray) -> np.ndarray:
    return np.tanh(a)


def tanh_prime(a: np.ndarray) -> np.ndarray:
    t = np.tanh(a)
    return (1.0 - t * t).astype(F, copy=False)


ACTIVATIONS = {
    "identity": (identity, identity_prime),
    "tanh": (tanh, tanh_prime),
}


def get(name: str):
    """Devolve o par (f, f') registado sob `name`."""
    try:
        return ACTIVATIONS[name]
    except KeyError:  # pragma: no cover - erro de configuração
        raise ValueError(
            f"ativação desconhecida: {name!r} (conhecidas: {sorted(ACTIVATIONS)})"
        ) from None
