"""Várias dinâmicas, escolhidas pelo contexto.

Diagnóstico que levou aqui, e vale a pena tê-lo escrito: com uma só matriz de
transição no topo, a rede não consegue reter várias tarefas — nem quando as vê
todas ao mesmo tempo, nem com o dobro dos parâmetros (medido: 6848 -> 13824
parâmetros muda o resultado de 0.823 para 0.825, ou seja, nada).

A razão é estrutural e não de tamanho. Uma matriz linear tem *um* conjunto de
valores próprios, logo um conjunto de frequências de rotação. Duas tarefas com
avanços de fase diferentes puxam-na para uma média que não serve nenhuma. Em
princípio uma matriz 8×8 tem quatro pares de valores próprios e podia alojar
quatro dinâmicas em subespaços ortogonais — mas nada na regra local encoraja
essa separação, e por isso ela não acontece.

A solução é dar-lhe *várias* dinâmicas e um mecanismo para escolher:

    ẑ_L(t) = f( Σ_k g_k · A_k · z_L(t-1) )      g = softmax(sim(z, c_k) / τ)

Cada componente tem a sua matriz `A_k` e o seu protótipo de contexto `c_k`. O
protótipo diz "esta dinâmica aplica-se quando o mundo se parece com isto"; a
responsabilidade `g_k` mede quanto se parece. É um sistema dinâmico linear
comutado, e é dos modelos mais antigos e mais bem estudados para sinais com
vários regimes.

Continua tudo local. `A_k` aprende com o mesmo erro de sempre, pesado pela sua
responsabilidade — quem foi responsável pela previsão é quem paga pelo erro.
`c_k` segue a média dos contextos em que foi usado, que é k-médias online.

E encaixa na memória episódica em vez de competir com ela: os protótipos são
chaves, a responsabilidade é a recuperação. Saber *em que mundo se está* passa
a ser a mesma operação que reconhecer um contexto.
"""

from __future__ import annotations

import numpy as np

from .dtypes import F


class TopMixture:
    """Mistura de transições lineares no topo, com selector por contexto."""

    __slots__ = ("n", "k", "A", "protos", "tau", "_resp", "_z_prev", "a")

    def __init__(self, n: int, k: int, rng: np.random.Generator, tau: float = 0.25):
        self.n = int(n)
        self.k = max(1, int(k))
        self.tau = float(tau)
        # Cada componente arranca perto da identidade com uma perturbação
        # diferente: idênticas colapsariam para a mesma solução.
        self.A = np.stack([
            np.eye(n, dtype=F)
            + (rng.standard_normal((n, n)) * (0.15 / np.sqrt(n))).astype(F)
            for _ in range(self.k)
        ])
        # Protótipos de contexto, aleatórios e normalizados.
        p = rng.standard_normal((self.k, n)).astype(F)
        self.protos = p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-8)
        self._resp = np.full(self.k, 1.0 / self.k, dtype=F)
        self._z_prev = np.zeros(n, dtype=F)
        self.a = np.zeros(n, dtype=F)

    # ------------------------------------------------------------------
    @property
    def n_params(self) -> int:
        return int(self.A.size + self.protos.size)

    @property
    def responsibilities(self) -> np.ndarray:
        return self._resp

    def responsibility(self, z_prev: np.ndarray) -> np.ndarray:
        """Quanto é que cada dinâmica se aplica ao contexto atual."""
        if self.k == 1:
            self._resp[:] = 1.0
            return self._resp
        norm = float(np.linalg.norm(z_prev))
        if norm < 1e-8:
            self._resp[:] = 1.0 / self.k
            return self._resp
        sim = self.protos.dot(z_prev) / norm
        e = np.exp((sim - sim.max()) / max(self.tau, 1e-6))
        self._resp[:] = (e / e.sum()).astype(F)
        return self._resp

    def predict(self, z_prev: np.ndarray) -> np.ndarray:
        """Σ_k g_k · A_k · z(t-1). Devolve a pré-ativação."""
        g = self.responsibility(z_prev)
        self._z_prev[:] = z_prev
        np.dot(np.tensordot(g, self.A, axes=(0, 0)), z_prev, out=self.a)
        return self.a

    def learn(self, eps_mod: np.ndarray, lr: float, proto_lr: float = 0.02,
              grad_clip: float = 0.0) -> None:
        """Quem foi responsável pela previsão é quem paga pelo erro.

        ΔA_k ∝ g_k · ε ⊗ z(t-1), e o protótipo segue a média dos contextos em
        que a sua componente foi usada. Passo normalizado por ‖z‖², pela mesma
        razão de sempre: uma regra delta só é estável abaixo de 1/‖x‖².
        """
        z = self._z_prev
        norm = float(np.dot(z, z)) + 1e-6
        outer = np.outer(eps_mod, z).astype(F, copy=False) / norm
        if grad_clip > 0.0:
            np.clip(outer, -grad_clip, grad_clip, out=outer)
        for i in range(self.k):
            g = float(self._resp[i])
            if g > 1e-4:
                self.A[i] += F(lr * g) * outer
                # k-médias online: o protótipo aproxima-se do contexto em que
                # foi usado, na proporção em que foi usado.
                self.protos[i] += F(proto_lr * g) * (z - self.protos[i])
        n = np.linalg.norm(self.protos, axis=1, keepdims=True)
        self.protos /= np.maximum(n, 1e-8)

    def sigma_max(self) -> float:
        """Maior valor singular da mistura efetiva corrente."""
        mixed = np.tensordot(self._resp, self.A, axes=(0, 0))
        return float(np.linalg.svd(mixed, compute_uv=False)[0])

    def usage(self) -> np.ndarray:
        return self._resp.copy()
