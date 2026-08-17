"""Novelty-driven neurogenesis: recruit virgin units when the world changes.

The dentate gyrus really does this: new, hyperplastic neurons are integrated
when the environment is new, and the veteran circuits remain relatively
untouched. Here, the minimal version:

  1. The network is born with a fraction of the latent units active. The
     rest exist but are FROZEN: input and output weights at zero and zero
     learning rate on their rows/columns. With null weights their state
     stays at 0 by construction (tanh(0)=0, no error pulls them), so they
     take part in nothing - neither cost nor interference.

  2. A novelty detector tracks the open-loop surprise
     (trace.open_loop_surprise) with two EMAs: a short one (the present) and
     a long one (the usual). When the short one exceeds `novelty_ratio`
     times the long one for `sustain` consecutive frames, a block of virgin
     units is recruited per level: small init on their rows/columns.

  3. At the moment of recruitment, ALL units active until then become
     protected veterans: the learning rate of any synapse on the row or
     column of a veteran is multiplied by `protect_factor`.
     It is structural metaplasticity - per unit, not per synapse.

  4. Hysteresis: after recruiting, the detector disarms and only rearms when
     the short surprise returns to ~1x the long one (the new task has been
     absorbed). Without this, a single world change would spend all the
     reserve blocks.

The implementation does not touch the local rule: `step` runs the normal
network and then rewrites W as W_old + S ⊙ (W_new − W_old), with S the
matrix of per-synapse rate scales (0 frozen, `protect_factor` veteran, 1
active virgin). It works on top of any internal mechanism (metaplasticity
included).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import PCConfig
from .dtypes import F
from .network import PCNetwork


@dataclass
class NeurogenesisConfig:
    # Fraction of the latent units active at birth; the rest is reserve.
    initial_frac: float = 0.5
    # How many blocks the reserve splits into (1 recruitment = 1 block).
    n_blocks: int = 2
    # Learning-rate multiplier on the rows/columns of the veterans.
    protect_factor: float = 0.05
    # Fires when EMA_short > novelty_ratio × EMA_long ...
    novelty_ratio: float = 1.5
    # ... for `sustain` consecutive frames.
    sustain: int = 20
    alpha_short: float = 0.2    # short EMA (~5 frames)
    alpha_long: float = 0.005   # long EMA (~200 frames)
    warmup: int = 500           # frames before the detector arms
    rearm_ratio: float = 1.1    # rearms when short < rearm_ratio × long
    # Init scale of the recruited units (multiplies 1/sqrt(fan_in)).
    recruit_init: float = 0.2
    # How the factor of a synapse's two endpoints is combined:
    #   "min":   any synapse on the row/column of a veteran gets
    #            protect_factor - literal per-unit protection.
    #   "hyper": a synapse touching a NEWLY recruited unit is fully
    #            plastic (new neurons are hyperplastic); only
    #            veteran-veteran synapses stay protected.
    protect_rule: str = "min"


class NeurogenesisNetwork(PCNetwork):
    """PCNetwork with a reserve of latent units recruitable by novelty."""

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

        # factor[l][j]: 0 frozen, protect_factor veteran, 1 active virgin.
        # Level 0 is sensory: always 1.
        self.factor: list[np.ndarray] = [
            np.ones(n, dtype=F) for n in cfg.sizes
        ]
        # Reserve per level: list of blocks (arrays of indices), in order.
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

        # Novelty detector.
        self._ema_short: float | None = None
        self._ema_long: float | None = None
        self._streak = 0
        self._armed = True
        self._t = 0
        self.n_recruited = 0
        self.recruit_log: list[dict] = []

    # ------------------------------------------------------------------
    # masks
    # ------------------------------------------------------------------
    def _freeze_inactive(self) -> None:
        """Zeroes the input and output weights of all frozen units."""
        for l in range(1, self.L + 1):
            frozen = self.factor[l] == 0.0
            if not frozen.any():
                continue
            # column in the generator below: the unit's output
            self.layers[l - 1].W[:, frozen] = 0.0
            self.layers[l - 1].refresh_device()
            if l < self.L:
                # row in the generator above: the unit's input
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
        # "hyper": frozen dominates; then, touching a virgin unit
        # (factor exactly 1, and latent) gives full plasticity; the rest
        # (veteran-veteran, or veteran-sensory) stays protected.
        active = np.multiply.outer(fr > 0, fc > 0)
        fresh_r = (fr == 1.0) if l_row > 0 else np.zeros(fr.size, dtype=bool)
        fresh_c = (fc == 1.0) if l_col > 0 else np.zeros(fc.size, dtype=bool)
        fresh = np.logical_or.outer(fresh_r, fresh_c)
        # Before the first recruitment all active units have factor 1 (they
        # are "virgins"), so everything active stays at 1 - as it should.
        S = np.where(fresh, F(1.0), F(self.ng.protect_factor))
        return np.where(active, S, F(0.0)).astype(F)

    def _rebuild_scales(self) -> None:
        """S per matrix: lr scale of each synapse from the factors of its
        two endpoints. Frozen dominates (0)."""
        self._S_layers = [
            self._scale_matrix(l, l + 1) for l in range(self.L)
        ]
        self._S_A = self._scale_matrix(self.L, self.L)

    # ------------------------------------------------------------------
    # one step: normal network + plasticity mask + detector
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
                # Metaplasticity: the importance of frozen synapses is
                # structurally zero and was diluting the mean - the relative
                # imp/mean scale of the live ones doubled and their effective
                # lr fell until the network learned nothing (measured: NRMSE
                # 1.000, that is, predicting zero). Keeping the frozen ones
                # at the mean of the live ones makes the normalization blind
                # to the reserve.
                if self.cfg.metaplasticity > 0.0:
                    dead = S == 0.0
                    if dead.any() and not dead.all():
                        lay.importance[dead] = lay.importance[~dead].mean()
            self.A[:] = pre_A + self._S_A * (self.A - pre_A)
            self._refresh_transition()
            self._novelty(trace.open_loop_surprise)
        return trace

    # ------------------------------------------------------------------
    # novelty detector
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
            # The new world becomes the new "usual": without this, the long
            # EMA took ~200 frames to rise and the same surprise jump fired
            # twice, spending two blocks on a single change
            # (measured: recruitments [0, 12, 0] instead of [0, 6, 6]).
            self._ema_long = self._ema_short

    # ------------------------------------------------------------------
    # recruitment
    # ------------------------------------------------------------------
    def _recruit(self) -> None:
        """Unlocks one block per level; protects all veterans."""
        rng = self._ng_rng
        cfg = self.cfg
        recruited: dict[int, int] = {}
        for l in range(1, self.L + 1):
            # 1. whoever was active becomes a protected veteran
            active = self.factor[l] > 0.0
            self.factor[l][active] = F(self.ng.protect_factor)
            if not self.blocks[l]:
                recruited[l] = 0
                continue
            blk = self.blocks[l].pop(0)
            self.factor[l][blk] = 1.0
            recruited[l] = len(blk)
            self.n_recruited += len(blk)

            # 2. small init on the rows/columns of the recruited units, only
            # for active partners - a synapse with a frozen endpoint stays at 0.
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
                    self.A[j, j] = F(1.0)  # self-retention, as at init
                self._refresh_transition()

        self._rebuild_scales()
        self.recruit_log.append({
            "step": self._t,
            "per_level": recruited,
            "total": sum(recruited.values()),
        })

    # ------------------------------------------------------------------
    def active_counts(self) -> dict[int, int]:
        """Active units (virgins + veterans) per latent level."""
        return {l: int(np.count_nonzero(self.factor[l]))
                for l in range(1, self.L + 1)}
