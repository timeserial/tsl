"""Neurogénese por novidade: recrutar unidades virgens quando o mundo muda.

O giro dentado faz isto de verdade: neurónios novos, hiperplásticos, são
integrados quando o ambiente é novo, e os circuitos veteranos ficam
relativamente intocados. Aqui a versão mínima:

  1. A rede nasce com uma fração das unidades latentes ativas. As restantes
     existem mas estão CONGELADAS: pesos de entrada e de saída a zero e taxa
     de aprendizagem zero nas suas linhas/colunas. Com pesos nulos o estado
     delas fica em 0 por construção (tanh(0)=0, nenhum erro as puxa), logo
     não participam em nada — nem custo, nem interferência.

  2. Um detetor de novidade acompanha a surpresa em malha aberta
     (trace.open_loop_surprise) com duas EMAs: uma curta (o presente) e uma
     longa (o habitual). Quando a curta excede `novelty_ratio` vezes a longa
     durante `sustain` tramas seguidas, recruta-se um bloco de unidades
     virgens por nível: init pequeno nas linhas/colunas delas.

  3. No momento do recrutamento, TODAS as unidades até aí ativas passam a
     veteranas protegidas: a taxa de aprendizagem de qualquer sinapse na
     linha ou coluna de uma veterana é multiplicada por `protect_factor`.
     É metaplasticidade estrutural — por unidade, não por sinapse.

  4. Histerese: depois de recrutar, o detetor desarma e só rearma quando a
     surpresa curta volta a ~1× a longa (a tarefa nova foi absorvida). Sem
     isto, uma única mudança de mundo gastava todos os blocos de reserva.

A implementação não toca na regra local: `step` corre a rede normal e depois
reescreve W como W_old + S ⊙ (W_new − W_old), com S a matriz de escalas de
taxa por sinapse (0 congelada, `protect_factor` veterana, 1 virgem ativa).
Funciona por cima de qualquer mecanismo interno (metaplasticidade incluída).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import PCConfig
from .dtypes import F
from .network import PCNetwork


@dataclass
class NeurogenesisConfig:
    # Fração das unidades latentes ativa à nascença; o resto é reserva.
    initial_frac: float = 0.5
    # Em quantos blocos a reserva se divide (1 recrutamento = 1 bloco).
    n_blocks: int = 2
    # Multiplicador da taxa de aprendizagem nas linhas/colunas das veteranas.
    protect_factor: float = 0.05
    # Dispara quando EMA_curta > novelty_ratio × EMA_longa ...
    novelty_ratio: float = 1.5
    # ... durante `sustain` tramas seguidas.
    sustain: int = 20
    alpha_short: float = 0.2    # EMA curta (~5 tramas)
    alpha_long: float = 0.005   # EMA longa (~200 tramas)
    warmup: int = 500           # tramas antes de o detetor armar
    rearm_ratio: float = 1.1    # rearma quando curta < rearm_ratio × longa
    # Escala do init das unidades recrutadas (multiplica 1/sqrt(fan_in)).
    recruit_init: float = 0.2
    # Como se combina o fator dos dois extremos de uma sinapse:
    #   "min":   qualquer sinapse na linha/coluna de uma veterana leva
    #            protect_factor — proteção literal por unidade.
    #   "hyper": sinapse que toca uma unidade RECÉM-recrutada é totalmente
    #            plástica (neurónios novos são hiperplásticos); só as
    #            sinapses veterana-veterana ficam protegidas.
    protect_rule: str = "min"


class NeurogenesisNetwork(PCNetwork):
    """PCNetwork com reserva de unidades latentes recrutáveis por novidade."""

    def __init__(self, config: PCConfig | None = None,
                 ng: NeurogenesisConfig | None = None) -> None:
        super().__init__(config)
        self.ng = ng or NeurogenesisConfig()
        cfg = self.cfg
        if (cfg.level_transition != "none" or cfg.n_dynamics > 1
                or cfg.gated_transition or cfg.gated_layers):
            raise ValueError(
                "NeurogenesisNetwork só suporta a hierarquia simples "
                "(level_transition='none', sem mistura nem portões)"
            )
        self._ng_rng = np.random.default_rng(cfg.seed + 977)

        # factor[l][j]: 0 congelada, protect_factor veterana, 1 ativa virgem.
        # O nível 0 é sensorial: sempre 1.
        self.factor: list[np.ndarray] = [
            np.ones(n, dtype=F) for n in cfg.sizes
        ]
        # Reserva por nível: lista de blocos (arrays de índices), por ordem.
        self.blocks: list[list[np.ndarray]] = [[] for _ in cfg.sizes]
        for l in range(1, len(cfg.sizes)):
            n = cfg.sizes[l]
            k0 = max(1, min(n - 1, int(round(self.ng.initial_frac * n))))
            self.factor[l][k0:] = 0.0
            reserve = np.arange(k0, n)
            self.blocks[l] = [
                b for b in np.array_split(reserve, self.ng.n_blocks) if len(b)
            ]

        self._freeze_inactive()
        self._rebuild_scales()

        # Detetor de novidade.
        self._ema_short: float | None = None
        self._ema_long: float | None = None
        self._streak = 0
        self._armed = True
        self._t = 0
        self.n_recruited = 0
        self.recruit_log: list[dict] = []

    # ------------------------------------------------------------------
    # máscaras
    # ------------------------------------------------------------------
    def _freeze_inactive(self) -> None:
        """Zera os pesos de entrada e saída de todas as unidades congeladas."""
        for l in range(1, self.L + 1):
            frozen = self.factor[l] == 0.0
            if not frozen.any():
                continue
            # coluna no gerador de baixo: a saída da unidade
            self.layers[l - 1].W[:, frozen] = 0.0
            self.layers[l - 1].refresh_device()
            if l < self.L:
                # linha no gerador de cima: a entrada da unidade
                self.layers[l].W[frozen, :] = 0.0
                self.layers[l].refresh_device()
            else:
                self.A[frozen, :] = 0.0
                self.A[:, frozen] = 0.0
                self._refresh_transition()

    def _scale_matrix(self, l_row: int, l_col: int) -> np.ndarray:
        fr, fc = self.factor[l_row], self.factor[l_col]
        if self.ng.protect_rule == "min":
            return np.minimum.outer(fr, fc).astype(F)
        # "hyper": congelada domina; depois, tocar numa unidade virgem
        # (fator exatamente 1, e latente) dá plasticidade total; o resto
        # (veterana-veterana, ou veterana-sensorial) fica protegido.
        active = np.multiply.outer(fr > 0, fc > 0)
        fresh_r = (fr == 1.0) if l_row > 0 else np.zeros(fr.size, dtype=bool)
        fresh_c = (fc == 1.0) if l_col > 0 else np.zeros(fc.size, dtype=bool)
        fresh = np.logical_or.outer(fresh_r, fresh_c)
        # Antes do primeiro recrutamento todas as ativas têm fator 1 (são
        # "virgens"), logo tudo o que é ativo fica a 1 — como deve ser.
        S = np.where(fresh, F(1.0), F(self.ng.protect_factor))
        return np.where(active, S, F(0.0)).astype(F)

    def _rebuild_scales(self) -> None:
        """S por matriz: escala de lr de cada sinapse a partir dos fatores
        dos seus dois extremos. Congelada domina (0)."""
        self._S_layers = [
            self._scale_matrix(l, l + 1) for l in range(self.L)
        ]
        self._S_A = self._scale_matrix(self.L, self.L)

    # ------------------------------------------------------------------
    # um passo: rede normal + máscara de plasticidade + detetor
    # ------------------------------------------------------------------
    def step(self, x, learn: bool = True, use_memory: bool = True):
        if learn:
            pre_W = [lay.W.copy() for lay in self.layers]
            pre_A = self.A.copy()
        trace = super().step(x, learn=learn, use_memory=use_memory)
        if learn:
            for lay, W0, S in zip(self.layers, pre_W, self._S_layers):
                lay.W[:] = W0 + S * (lay.W - W0)
                lay.refresh_device()
                # Metaplasticidade: a importância das sinapses congeladas é
                # estruturalmente zero e diluía a média — a escala relativa
                # imp/média das vivas duplicava e o lr efetivo delas caía
                # até a rede não aprender nada (medido: NRMSE 1.000, ou
                # seja, prever zero). Manter as congeladas à média das vivas
                # torna a normalização cega à reserva.
                if self.cfg.metaplasticity > 0.0:
                    dead = S == 0.0
                    if dead.any() and not dead.all():
                        lay.importance[dead] = lay.importance[~dead].mean()
            self.A[:] = pre_A + self._S_A * (self.A - pre_A)
            self._refresh_transition()
            self._novelty(trace.open_loop_surprise)
        return trace

    # ------------------------------------------------------------------
    # detetor de novidade
    # ------------------------------------------------------------------
    def _novelty(self, s: float) -> None:
        self._t += 1
        if self._ema_short is None:
            self._ema_short = self._ema_long = s
            return
        self._ema_short += self.ng.alpha_short * (s - self._ema_short)
        self._ema_long += self.ng.alpha_long * (s - self._ema_long)
        if self._t < self.ng.warmup:
            return
        if not self._armed:
            if self._ema_short < self.ng.rearm_ratio * self._ema_long:
                self._armed = True
                self._streak = 0
            return
        if self._ema_short > self.ng.novelty_ratio * max(self._ema_long, 1e-12):
            self._streak += 1
        else:
            self._streak = 0
        if self._streak >= self.ng.sustain and any(
            self.blocks[l] for l in range(1, self.L + 1)
        ):
            self._recruit()
            self._armed = False
            self._streak = 0
            # O mundo novo passa a ser o novo "habitual": sem isto, a EMA
            # longa demorava ~200 tramas a subir e o mesmo salto de surpresa
            # disparava duas vezes, gastando dois blocos numa só mudança
            # (medido: recrutamentos [0, 12, 0] em vez de [0, 6, 6]).
            self._ema_long = self._ema_short

    # ------------------------------------------------------------------
    # recrutamento
    # ------------------------------------------------------------------
    def _recruit(self) -> None:
        """Desbloqueia um bloco por nível; protege todas as veteranas."""
        rng = self._ng_rng
        cfg = self.cfg
        recruited: dict[int, int] = {}
        for l in range(1, self.L + 1):
            # 1. quem estava ativa passa a veterana protegida
            active = self.factor[l] > 0.0
            self.factor[l][active] = F(self.ng.protect_factor)
            if not self.blocks[l]:
                recruited[l] = 0
                continue
            blk = self.blocks[l].pop(0)
            self.factor[l][blk] = 1.0
            recruited[l] = len(blk)
            self.n_recruited += len(blk)

            # 2. init pequeno nas linhas/colunas das recrutadas, só para
            # parceiros ativos — sinapse com extremo congelado fica a 0.
            below_ok = self.factor[l - 1] > 0.0
            scale_out = self.ng.recruit_init / np.sqrt(cfg.sizes[l])
            for j in blk:
                col = rng.standard_normal(cfg.sizes[l - 1]).astype(F) * F(scale_out)
                self.layers[l - 1].W[:, j] = np.where(below_ok, col, F(0.0))
            self.layers[l - 1].refresh_device()
            if l < self.L:
                above_ok = self.factor[l + 1] > 0.0
                scale_in = self.ng.recruit_init / np.sqrt(cfg.sizes[l + 1])
                for j in blk:
                    row = rng.standard_normal(cfg.sizes[l + 1]).astype(F) * F(scale_in)
                    self.layers[l].W[j, :] = np.where(above_ok, row, F(0.0))
                self.layers[l].refresh_device()
            else:
                top_ok = self.factor[l] > 0.0
                scale_a = self.ng.recruit_init / np.sqrt(cfg.sizes[l])
                for j in blk:
                    row = rng.standard_normal(cfg.sizes[l]).astype(F) * F(scale_a)
                    self.A[j, :] = np.where(top_ok, row, F(0.0))
                    self.A[j, j] = F(1.0)  # retenção própria, como no init
                self._refresh_transition()

        self._rebuild_scales()
        self.recruit_log.append({
            "step": self._t,
            "per_level": recruited,
            "total": sum(recruited.values()),
        })

    # ------------------------------------------------------------------
    def active_counts(self) -> dict[int, int]:
        """Unidades ativas (virgens + veteranas) por nível latente."""
        return {l: int(np.count_nonzero(self.factor[l]))
                for l in range(1, self.L + 1)}
