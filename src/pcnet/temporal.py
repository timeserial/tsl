"""Hierarquia de escalas de tempo.

A versão do passo 1 tinha um único modelo temporal, no topo. Todos os níveis
abaixo eram instantâneos: sabiam *o que* estava a acontecer mas não tinham
memória nenhuma do seu próprio passado. O córtex não é assim.

Hasson et al. mediram "janelas receptivas temporais" que crescem ao longo da
hierarquia: o córtex auditivo primário integra sobre dezenas de milissegundos,
áreas frontais sobre dezenas de segundos. Kiebel, Daunizeau & Friston (2008,
"A hierarchy of time-scales and the brain") mostram que isto cai naturalmente
de um modelo preditivo hierárquico onde cada nível tem a sua própria dinâmica,
progressivamente mais lenta.

Aqui, cada nível latente ganha:

    ẑ_l(t) = f( A_l · z_l(t-1) )        com   A_l = (1-λ_l)·I + λ_l·B_l

λ_l = 1/τ_l é a taxa do nível. τ cresce com l, portanto o topo quase não se
mexe entre tramas (memória longa) e os níveis baixos seguem o sinal (memória
curta). A parte identidade é o integrador com fuga; B_l é o que se aprende.

`kind="diagonal"` dá a cada unidade a sua própria constante de tempo e mais
nada — é a versão mais barata e a mais literalmente biológica (a constante de
tempo da membrana). `kind="dense"` deixa os níveis misturarem-se entre si.
"""

from __future__ import annotations

import numpy as np

from .dtypes import F


class Transition:
    """A previsão que um nível faz de si próprio no instante seguinte."""

    __slots__ = ("n", "kind", "lam", "B", "a", "_f", "_fprime")

    def __init__(
        self,
        n: int,
        kind: str,
        lam: float,
        rng: np.random.Generator,
        activation_pair,
        init_scale: float = 0.1,
    ) -> None:
        if kind not in ("none", "diagonal", "dense"):
            raise ValueError(f"transição desconhecida: {kind!r}")
        self.n = int(n)
        self.kind = kind
        self.lam = float(lam)
        self._f, self._fprime = activation_pair

        if kind == "dense":
            self.B = (
                np.eye(n, dtype=F)
                + (rng.standard_normal((n, n)) * (init_scale / np.sqrt(n))).astype(F)
            )
        elif kind == "diagonal":
            # Uma constante de tempo por unidade. Arranca em 1 (guardar o
            # estado) com uma pitada de variabilidade, como neurónios reais.
            self.B = (1.0 + init_scale * rng.standard_normal(n)).astype(F)
        else:
            self.B = np.zeros(0, dtype=F)

        self.a = np.zeros(n, dtype=F)

    # ------------------------------------------------------------------
    @property
    def n_params(self) -> int:
        return int(self.B.size)

    def predict(self, z_prev: np.ndarray) -> np.ndarray:
        """ẑ(t) = f((1-λ)·z(t-1) + λ·B·z(t-1))."""
        if self.kind == "none":
            self.a[:] = z_prev
            return self.a
        if self.kind == "diagonal":
            mixed = self.B * z_prev
        else:
            mixed = self.B.dot(z_prev)
        self.a[:] = (1.0 - self.lam) * z_prev + self.lam * mixed
        return self._f(self.a).astype(F, copy=False)

    def learn(
        self,
        eps: np.ndarray,
        z_prev: np.ndarray,
        lr: float,
        grad_clip: float = 0.0,
    ) -> None:
        """ΔB ∝ (ε ⊙ f') ⊗ z(t-1). Local: só o erro daqui e o estado de antes."""
        if self.kind == "none" or lr <= 0.0:
            return
        mod = (eps * self._fprime(self.a) * self.lam).astype(F, copy=False)
        if self.kind == "diagonal":
            dB = mod * z_prev
        else:
            dB = np.outer(mod, z_prev).astype(F, copy=False)
        if grad_clip > 0.0:
            np.clip(dB, -grad_clip, grad_clip, out=dB)
        self.B += F(lr) * dB

    def sigma_max(self) -> float:
        """Maior valor singular de A = (1-λ)I + λB, para o limite de estabilidade."""
        if self.kind == "none":
            return 1.0
        if self.kind == "diagonal":
            return float(np.max(np.abs((1.0 - self.lam) + self.lam * self.B)))
        A = (1.0 - self.lam) * np.eye(self.n, dtype=F) + self.lam * self.B
        return float(np.linalg.svd(A, compute_uv=False)[0])


def timescales(n_levels: int, base: float, ratio: float) -> tuple[float, ...]:
    """λ_l por nível: rápido em baixo, lento em cima.

    `base` é a taxa do nível latente mais baixo, `ratio` quanto ela abranda por
    cada degrau. ratio=2 significa que cada nível integra sobre o dobro do
    tempo do de baixo — a progressão que Hasson mede no córtex.
    """
    return tuple(min(1.0, base / (ratio**l)) for l in range(n_levels))
