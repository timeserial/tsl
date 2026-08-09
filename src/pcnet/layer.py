"""A unidade base: uma camada preditiva.

    ẑ_l = f(W_l · z_{l+1})        # a camada de cima PREVÊ a de baixo
    ε_l = z_l − ẑ_l               # só o erro sobe
    Δz_{l+1} ∝ W_lᵀ · (ε_l ⊙ f')  # o estado de cima ajusta-se
    ΔW_l ∝ (ε_l ⊙ f') · z_{l+1}ᵀ  # aprendizagem local Hebbiana

Nada aqui vê o resto da rede: `learn` usa apenas o erro que está na camada e o
estado que está imediatamente acima — as duas quantidades que existem
fisicamente na sinapse. É essa restrição que torna o C trivial (não há grafo,
não há ativações guardadas) e que permite treinar em hardware analógico.

Convenção de forma: W tem forma (n_abaixo, n_acima), C-contígua, float32 — a
mesma memória que o crossbar vai ter, linha = neurónio de baixo.
"""

from __future__ import annotations

import numpy as np

from . import activations
from .dtypes import F


class PCLayer:
    """Gerador de um nível: prevê `n_below` a partir de `n_above`."""

    __slots__ = (
        "n_below", "n_above", "W", "W_eff", "device",
        "act_name", "_f", "_fprime", "a", "zhat",
        "sigma_max", "_pi_v", "importance",
    )

    def __init__(
        self,
        n_below: int,
        n_above: int,
        activation: str,
        rng: np.random.Generator,
        init_scale: float = 1.0,
    ) -> None:
        self.n_below = int(n_below)
        self.n_above = int(n_above)
        self.act_name = activation
        self._f, self._fprime = activations.get(activation)
        self.device = None

        # ~1/sqrt(fan_in): mantém a pré-ativação em O(1) e o tanh fora da
        # saturação, que é onde a derivada morre e a aprendizagem local para.
        scale = init_scale / np.sqrt(self.n_above)
        self.W = (rng.standard_normal((self.n_below, self.n_above)) * scale).astype(F)
        # Sem dispositivo, W_eff é o próprio W (mesma memória, custo zero).
        self.W_eff = self.W

        # Buffers pré-alocados (o C não vai ter malloc; o Python também não).
        self.a = np.zeros(self.n_below, dtype=F)
        self.zhat = np.zeros(self.n_below, dtype=F)

        # Estimativa de σ_max(W_eff) por iteração de potência, para o passo de
        # assentamento saber quão longe pode ir sem divergir.
        self._pi_v = (rng.standard_normal(self.n_above)).astype(F)
        self._pi_v /= np.linalg.norm(self._pi_v) or 1.0
        self.sigma_max = 0.0
        self._estimate_sigma_max()

        # Importância de cada sinapse: média corrente do quadrado do seu
        # próprio gradiente local. É a diagonal da informação de Fisher, que
        # aqui está disponível na sinapse sem nenhuma passagem para trás.
        self.importance = np.zeros_like(self.W)

    # -- substrato ----------------------------------------------------------
    def attach_device(self, array) -> None:
        """Passa a inferir com os pesos que o dispositivo tem, não com os pedidos.

        `W` continua a ser o peso float que a regra local atualiza — o
        "shadow weight" do treino consciente da quantização. `W_eff` é o que
        está mesmo no crossbar. Sem backprop, esta separação é trivial: não há
        gradiente para passar através do quantizador.
        """
        self.device = array
        self.refresh_device()

    def detach_device(self) -> None:
        self.device = None
        self.W_eff = self.W
        self._estimate_sigma_max()

    def refresh_device(self) -> None:
        """Reprograma o crossbar a partir dos pesos float atuais."""
        if self.device is None:
            self.W_eff = self.W
        else:
            self.W_eff = self.device.program(self.W)
        self._estimate_sigma_max()

    # -- estabilidade -------------------------------------------------------
    def _estimate_sigma_max(self, max_iters: int = 32, tol: float = 1e-5) -> None:
        """Iteração de potência sobre W_eff, com arranque a quente.

        Itera até convergir, não um número fixo de vezes. Com arranque a
        quente são tipicamente 1-2 iterações (W muda devagar entre
        atualizações), mas trocar de crossbar precisa de mais — e um σ_max
        subestimado dá um passo de assentamento grande demais, que diverge.

        Iterar até convergir faz também com que σ_max seja função só de
        W_eff, e não do histórico de chamadas: dois modelos com os mesmos
        pesos comportam-se da mesma maneira, venham eles de treino ou de
        `load_state_dict`. São dois produtos matriz-vetor por iteração — no
        crossbar, duas leituras, uma em cada sentido.
        """
        previous = 0.0
        for _ in range(max_iters):
            u = self.W_eff.dot(self._pi_v)
            nu = float(np.linalg.norm(u))
            if nu <= 1e-12:
                self.sigma_max = 0.0
                return
            u /= nu
            v = self.W_eff.T.dot(u)
            nv = float(np.linalg.norm(v))
            if nv <= 1e-12:
                self.sigma_max = 0.0
                return
            self._pi_v = (v / nv).astype(F)
            self.sigma_max = nv
            if abs(nv - previous) <= tol * max(nv, 1e-12):
                return
            previous = nv

    def max_stable_z_lr(self) -> float:
        """z_lr < 2/(1 + σ_max²): o passo acima do qual o assentamento diverge.

        O Hessiano da energia em ordem ao estado de cima é I + WᵀW, logo o
        maior valor próprio é 1 + σ_max(W)². É este o número que a
        variabilidade do dispositivo estraga — não por adicionar ruído à
        resposta, mas por inflacionar σ_max até o passo fixo deixar de ser
        estável.
        """
        return 2.0 / (1.0 + self.sigma_max**2)

    # -- caminho descendente: a previsão -----------------------------------
    def predict(self, z_above: np.ndarray) -> np.ndarray:
        """ẑ = f(W · z_acima). Escreve em buffers internos e devolve ẑ."""
        np.dot(self.W_eff, z_above, out=self.a)
        if self.device is not None:
            self.a[:] = self.device.read(self.a)
        self.zhat = self._f(self.a).astype(F, copy=False)
        return self.zhat

    # -- caminho ascendente: o erro ----------------------------------------
    def modulated_error(self, eps_below: np.ndarray) -> np.ndarray:
        """ε ⊙ f'(a): o erro tal como o vê a sinapse, já pesado pelo ganho."""
        return (eps_below * self._fprime(self.a)).astype(F, copy=False)

    def backward(self, eps_mod: np.ndarray) -> np.ndarray:
        """Wᵀ · (ε ⊙ f'): a correção que sobe para o estado de cima.

        É o mesmo crossbar lido ao contrário, logo sofre os mesmos desvios de
        programação e mais uma dose de ruído de leitura.
        """
        out = self.W_eff.T.dot(eps_mod).astype(F, copy=False)
        return self.device.read(out) if self.device is not None else out

    # -- plasticidade -------------------------------------------------------
    def learn(
        self,
        eps_mod: np.ndarray,
        z_above: np.ndarray,
        lr: float,
        weight_decay: float = 0.0,
        grad_clip: float = 0.0,
        meta: float = 0.0,
        meta_decay: float = 0.999,
    ) -> None:
        """ΔW = lr · (ε ⊙ f') ⊗ z_acima. Puramente local, in-place.

        Atualiza sempre o peso float; se houver dispositivo, reprograma-o a
        seguir (treino consciente da quantização).
        """
        dW = np.outer(eps_mod, z_above).astype(F, copy=False)
        if grad_clip > 0.0:
            np.clip(dW, -grad_clip, grad_clip, out=dW)
        if weight_decay > 0.0:
            self.W *= F(1.0 - lr * weight_decay)

        if meta > 0.0:
            # A sinapse endurece na proporção do que já provou importar.
            self.importance *= F(meta_decay)
            self.importance += F(1.0 - meta_decay) * (dW * dW)
            scale = self.importance / max(float(self.importance.mean()), 1e-12)
            self.W += F(lr) * dW / (1.0 + F(meta) * scale)
        else:
            self.W += F(lr) * dW
        # Sempre, com ou sem dispositivo: σ_max cresce ao longo do treino e é
        # ele que fixa o passo de assentamento estável. Sem isto a rede float
        # ficava com a estimativa da inicialização e o passo adaptativo nunca
        # chegava a atuar.
        self.refresh_device()

    # -- contabilidade de energia ------------------------------------------
    def macs_down(self) -> int:
        """MACs da previsão: sempre densa (é o crossbar a fazê-la em analógico)."""
        return self.n_below * self.n_above

    def macs_up(self, nnz_eps: int) -> int:
        """MACs do erro a subir: proporcionais aos erros que passam o limiar."""
        return int(nnz_eps) * self.n_above

    def __repr__(self) -> str:  # pragma: no cover - conveniência
        return (
            f"PCLayer({self.n_above} -> {self.n_below}, act={self.act_name!r})"
        )
