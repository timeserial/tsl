"""Transição temporal com portões: a dinâmica escolhida pelo erro.

De onde isto vem, porque é a decomposição mais importante que medimos: um GRU
proibido de atribuir crédito através do tempo (estado destacado a cada passo)
faz 0.576 nas três tarefas onde nós fazemos 0.778 e o BPTT completo faz 0.490.
Ou seja, a maior parte do muro não está no tempo — está no que a atualização
faz com o erro de *um* instante. E a única diferença estrutural relevante da
célula dele é o **portão multiplicativo**: decidir, em função do contexto,
quanto de cada unidade de estado atualizar, com o crédito a fluir através
dessa decisão.

As nossas tentativas anteriores de modularidade falharam exatamente por não
terem isto: a mistura de dinâmicas escolhia por semelhança (k-médias, sem
crédito), a esparsidade cortava por magnitude (cega ao erro). O portão do GRU
é escolhido *pelo erro*. E multiplicação entre sinais não é truque de
engenharia — inibição shunting, gating dendrítico, o tálamo a portar o córtex:
é dos mecanismos mais documentados da neurofisiologia.

A transição do topo passa de

    ẑ(t) = (1-λ)·z + λ·tanh(A·z)          λ fixo, igual para tudo

para a versão em que λ é aprendido e depende do contexto:

    c = tanh(A·z)                          o candidato (a dinâmica)
    g = σ(G·z + b)                         o portão (quanto atualizar)
    ẑ(t) = (1-g)⊙z + g⊙c

O crédito é local na mesma. Com ε = z_assentado − ẑ vindo do assentamento:

    ∂ẑ/∂c = g          ->  ΔA ∝ (ε ⊙ g ⊙ (1-c²)) ⊗ z
    ∂ẑ/∂g = c − z      ->  ΔG ∝ (ε ⊙ (c−z) ⊙ g(1-g)) ⊗ z

Cada fator existe no neurónio: o erro, o candidato, o valor do portão. É uma
regra de três fatores (pré × pós × modulador), que é o consenso atual sobre
como a plasticidade biológica realmente funciona. Passos normalizados por
‖z‖² (NLMS) — lição aprendida três vezes nesta sessão.
"""

from __future__ import annotations

import numpy as np

from .dtypes import F


def _sigmoid(a: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-a))).astype(F, copy=False)


class GatedTransition:
    """ẑ(t) = (1-g)⊙z + g⊙tanh(A·z), com g = σ(G·z + b) aprendido pelo erro."""

    __slots__ = ("n", "A", "G", "b", "_z", "_c", "_g")

    def __init__(self, n: int, rng: np.random.Generator, gate_bias: float = 0.0):
        self.n = int(n)
        self.A = (
            np.eye(n, dtype=F)
            + (rng.standard_normal((n, n)) * (0.2 / np.sqrt(n))).astype(F)
        )
        self.G = (rng.standard_normal((n, n)) * (0.3 / np.sqrt(n))).astype(F)
        # bias inicial do portão: 0 -> g≈0.5, meio aberto, sem preconceito.
        self.b = np.full(n, gate_bias, dtype=F)
        self._z = np.zeros(n, dtype=F)
        self._c = np.zeros(n, dtype=F)
        self._g = np.full(n, 0.5, dtype=F)

    # ------------------------------------------------------------------
    @property
    def n_params(self) -> int:
        return int(self.A.size + self.G.size + self.b.size)

    @property
    def gate(self) -> np.ndarray:
        """O portão da última previsão — instrumentação."""
        return self._g

    def predict(self, z_prev: np.ndarray) -> np.ndarray:
        self._z[:] = z_prev
        self._c[:] = np.tanh(self.A.dot(z_prev))
        self._g[:] = _sigmoid(self.G.dot(z_prev) + self.b)
        return ((1.0 - self._g) * z_prev + self._g * self._c).astype(F, copy=False)

    def learn(self, eps: np.ndarray, lr: float, grad_clip: float = 0.0) -> None:
        """Três fatores, tudo local, passo normalizado por ‖z‖²."""
        z, c, g = self._z, self._c, self._g
        norm = float(np.dot(z, z)) + 1e-6

        # crédito para a dinâmica: passa pelo portão
        mod_c = (eps * g * (1.0 - c * c)).astype(F, copy=False)
        dA = np.outer(mod_c / norm, z).astype(F, copy=False)
        # crédito para o portão: proporcional ao que a escolha teria mudado
        mod_g = (eps * (c - z) * g * (1.0 - g)).astype(F, copy=False)
        dG = np.outer(mod_g / norm, z).astype(F, copy=False)

        if grad_clip > 0.0:
            np.clip(dA, -grad_clip, grad_clip, out=dA)
            np.clip(dG, -grad_clip, grad_clip, out=dG)
        self.A += F(lr) * dA
        self.G += F(lr) * dG
        self.b += F(lr) * mod_g

    def sigma_max(self) -> float:
        """Limite para o passo de assentamento. O Jacobiano da mistura é
        (1-g)·I + g·diag(1-c²)·A mais termos do portão; majoramos pelo pior
        caso g=1, que é σ_max(A). Conservador e barato."""
        return float(np.linalg.svd(self.A, compute_uv=False)[0])
