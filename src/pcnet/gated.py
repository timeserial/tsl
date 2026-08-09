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

    __slots__ = ("n", "A", "G", "b", "_z", "_c", "_g", "eA", "eG", "eb",
                 "eligibility", "B", "H", "_u")

    def __init__(self, n: int, rng: np.random.Generator, gate_bias: float = 0.0,
                 eligibility: bool = False, input_dim: int = 0):
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

        # Traços de elegibilidade (e-prop / synaptic tagging): cada sinapse
        # guarda um rasto do que acabou de fazer, e o erro que chegar
        # enquanto o rasto dura credita-a — crédito através do tempo, sem
        # BPTT. O rasto decai à taxa de retenção do próprio portão (1-g):
        # uma unidade que retém muito propaga crédito para longe; uma que
        # atualiza sempre só é creditada pelo instante. A memória do crédito
        # e a memória do estado são a mesma coisa, como deve ser.
        self.eligibility = bool(eligibility)
        self.eA = np.zeros((n, n), dtype=F) if eligibility else None
        self.eG = np.zeros((n, n), dtype=F) if eligibility else None
        self.eb = np.zeros(n, dtype=F) if eligibility else None

        # O relé talâmico: o candidato e o portão veem também a entrada
        # anterior, não só o estado recorrente — c = tanh(A·z + B·u),
        # g = σ(G·z + H·u + b). É a última diferença estrutural para a célula
        # do GRU, e no córtex é literal: a entrada chega em paralelo com a
        # recorrência. Crédito local na mesma; u tem norma própria (NLMS).
        if input_dim > 0:
            self.B = (rng.standard_normal((n, input_dim))
                      * (0.2 / np.sqrt(input_dim))).astype(F)
            self.H = (rng.standard_normal((n, input_dim))
                      * (0.3 / np.sqrt(input_dim))).astype(F)
        else:
            self.B = None
            self.H = None
        self._u = np.zeros(max(input_dim, 1), dtype=F)

    # ------------------------------------------------------------------
    @property
    def n_params(self) -> int:
        return int(self.A.size + self.G.size + self.b.size)

    @property
    def gate(self) -> np.ndarray:
        """O portão da última previsão — instrumentação."""
        return self._g

    def predict(self, z_prev: np.ndarray,
                u: np.ndarray | None = None) -> np.ndarray:
        self._z[:] = z_prev
        a_c = self.A.dot(z_prev)
        a_g = self.G.dot(z_prev) + self.b
        if self.B is not None and u is not None:
            self._u[:] = u
            a_c = a_c + self.B.dot(u)
            a_g = a_g + self.H.dot(u)
        self._c[:] = np.tanh(a_c)
        self._g[:] = _sigmoid(a_g)
        return ((1.0 - self._g) * z_prev + self._g * self._c).astype(F, copy=False)

    def learn(self, eps: np.ndarray, lr: float, grad_clip: float = 0.0) -> None:
        """Três fatores, tudo local, passo normalizado por ‖z‖².

        Com elegibilidade, o produto instantâneo é acumulado num rasto por
        sinapse que decai a (1-g), e é o rasto que o erro credita.
        """
        z, c, g = self._z, self._c, self._g
        norm = float(np.dot(z, z)) + 1e-6

        inst_c = np.outer((g * (1.0 - c * c)) / norm, z).astype(F, copy=False)
        inst_g_vec = ((c - z) * g * (1.0 - g)).astype(F, copy=False)
        inst_g = np.outer(inst_g_vec / norm, z).astype(F, copy=False)

        if self.eligibility:
            retain = (1.0 - g)[:, None]
            self.eA *= retain
            self.eA += inst_c
            self.eG *= retain
            self.eG += inst_g
            self.eb *= (1.0 - g)
            self.eb += inst_g_vec
            dA = (eps[:, None] * self.eA).astype(F, copy=False)
            dG = (eps[:, None] * self.eG).astype(F, copy=False)
            db = (eps * self.eb).astype(F, copy=False)
        else:
            dA = (eps[:, None] * inst_c).astype(F, copy=False)
            dG = (eps[:, None] * inst_g).astype(F, copy=False)
            db = (eps * inst_g_vec).astype(F, copy=False)

        if grad_clip > 0.0:
            np.clip(dA, -grad_clip, grad_clip, out=dA)
            np.clip(dG, -grad_clip, grad_clip, out=dG)
        self.A += F(lr) * dA
        self.G += F(lr) * dG
        self.b += F(lr) * db

        if self.B is not None:
            u = self._u
            un = float(np.dot(u, u)) + 1e-6
            mod_c = (eps * g * (1.0 - c * c)).astype(F, copy=False)
            mod_g = ((eps * (c - z) * g * (1.0 - g))).astype(F, copy=False)
            dB = np.outer(mod_c / un, u).astype(F, copy=False)
            dH = np.outer(mod_g / un, u).astype(F, copy=False)
            if grad_clip > 0.0:
                np.clip(dB, -grad_clip, grad_clip, out=dB)
                np.clip(dH, -grad_clip, grad_clip, out=dH)
            self.B += F(lr) * dB
            self.H += F(lr) * dH

    def sigma_max(self) -> float:
        """Limite para o passo de assentamento. O Jacobiano da mistura é
        (1-g)·I + g·diag(1-c²)·A mais termos do portão; majoramos pelo pior
        caso g=1, que é σ_max(A). Conservador e barato."""
        return float(np.linalg.svd(self.A, compute_uv=False)[0])
