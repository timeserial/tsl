"""Activations and their derivative.

Kept deliberately poor: `tanh` and identity. Anything placed here must have
a trivial translation to C with integers (tanh -> 256-entry LUT), otherwise
it does not fit in the crossbar.
"""

from __future__ import annotations

import numpy as np

from .dtypes import F

# --------------------------------------------------------------------------
# f(a) and f'(a). The derivative is written as a function of *a* (the
# pre-activation) so that the signature is the same across all activations;
# in C we store only `a` per level, one static array per layer.
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
    """Returns the (f, f') pair registered under `name`."""
    try:
        return ACTIVATIONS[name]
    except KeyError:  # pragma: no cover - configuration error
        raise ValueError(
            f"ativação desconhecida: {name!r} (conhecidas: {sorted(ACTIVATIONS)})"
        ) from None
