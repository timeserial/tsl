"""Precisão: quanto vale a pena acreditar em cada erro.

Na formulação de energia livre (Friston), cada erro de previsão vem com uma
precisão π — o inverso da variância esperada desse erro. A energia não é
½‖ε‖², é ½π‖ε‖². Um nível cujo erro é habitualmente grande e imprevisível
merece π baixo: as suas queixas contam menos.

Friston identifica a modulação de precisão com a **atenção**, e o seu
substrato com a neuromodulação (acetilcolina, noradrenalina) a ajustar o ganho
pós-sináptico. Aqui é o mesmo objeto: um ganho por nível, aprendido
localmente.

Isto resolve também um problema concreto e nada teórico que a versão anterior
tinha: o nível sensorial (64 unidades, sinal cru) e o topo (8 unidades,
latente abstrato) têm erros de escalas completamente diferentes, e partilhavam
um único limiar θ afinado à mão. Com precisão, o limiar passa a ser aplicado
ao erro *normalizado* √π·ε — o mesmo θ quer dizer a mesma coisa em todo o
lado.

Regra de atualização, do gradiente da energia livre em ordem a log π:

    ∂F/∂logπ = ½(π·⟨ε²⟩ − 1)   ->   Δlogπ ∝ 1 − π·⟨ε²⟩

cujo ponto fixo é π = 1/⟨ε²⟩. Local, uma linha, e sem divisões no caminho
crítico se guardarmos log π.
"""

from __future__ import annotations

import numpy as np

from .dtypes import F


class Precision:
    """Ganho de um nível. Escalar por nível, ou um por unidade."""

    __slots__ = ("n", "per_unit", "log_pi", "lr", "lo", "hi")

    def __init__(
        self,
        n: int,
        per_unit: bool = False,
        lr: float = 0.01,
        init: float = 1.0,
        lo: float = 1e-2,
        hi: float = 1e3,
    ) -> None:
        self.n = int(n)
        self.per_unit = bool(per_unit)
        shape = (n,) if per_unit else (1,)
        self.log_pi = np.full(shape, np.log(init), dtype=F)
        self.lr = float(lr)
        self.lo, self.hi = float(lo), float(hi)

    @property
    def value(self) -> np.ndarray:
        return np.exp(self.log_pi).astype(F, copy=False)

    @property
    def scalar(self) -> float:
        """Uma precisão representativa do nível (para o limite de estabilidade)."""
        return float(np.exp(self.log_pi).max())

    def weight(self, eps: np.ndarray) -> np.ndarray:
        """π·ε — o erro tal como conta para a energia e para o estado."""
        return (self.value * eps).astype(F, copy=False)

    def normalize(self, eps: np.ndarray) -> np.ndarray:
        """√π·ε — o erro em unidades de desvio-padrão, para o limiar."""
        return (np.sqrt(self.value) * eps).astype(F, copy=False)

    def learn(self, eps: np.ndarray) -> None:
        """Δlogπ ∝ 1 − π·ε². Ponto fixo em π = 1/⟨ε²⟩."""
        if self.lr <= 0.0:
            return
        e2 = eps * eps
        if not self.per_unit:
            e2 = np.array([float(e2.mean())], dtype=F)
        grad = 1.0 - self.value * e2
        self.log_pi += F(self.lr) * grad.astype(F, copy=False)
        np.clip(self.log_pi, np.log(self.lo), np.log(self.hi), out=self.log_pi)

    def rescale(self, log_offset: float) -> None:
        """Desloca log π por uma constante comum a toda a rede.

        A escala *absoluta* das precisões não tem significado: multiplicar
        todas por uma constante só reescala a energia, e o mínimo é o mesmo.
        O que tem significado são os rácios entre níveis.

        Mas o assentamento é descida de gradiente com passo fixo, que não é
        invariante à escala: π a subir infla o Hessiano, o passo adaptativo
        encolhe para não divergir, e a inferência morre à fome. Foi
        exatamente isto que aconteceu — π chegou a 10³ e o NRMSE a 1.0.
        Ancorar a escala num nível de referência resolve, sem perder nada.
        """
        self.log_pi -= F(log_offset)
        np.clip(self.log_pi, np.log(self.lo), np.log(self.hi), out=self.log_pi)

    def __repr__(self) -> str:  # pragma: no cover
        v = self.value
        return f"Precision(n={self.n}, π={v.mean():.3g})"


class UnitPrecision:
    """Precisão fixa a 1. O caso sem precisão, sem ramos no código."""

    __slots__ = ()
    value = F(1.0)
    scalar = 1.0

    def weight(self, eps):
        return eps

    def normalize(self, eps):
        return eps

    def learn(self, eps):
        return None

    def rescale(self, log_offset):
        return None
