"""Episodic memory: record now, consolidate later.

Section 4 of `CONTEXTO.md` asks for two memories, and the distinction
between them is the whole point:

  * **Episodic** - a key-value store writable on the spot. A new fact enters
    here *without touching the weights*. That is why learning something new
    does not destroy what was already known.
  * **Offline consolidation** - later, at rest, the episodes are replayed
    and distilled into the generative weights by the same local rule as
    always. What repeats stays; what was accidental is forgotten.

It is the hippocampus and sleep. And it is the brain's answer to
catastrophic forgetting: a network that learns only in the weights has to
choose between plastic and stable - plastic forgets, stable does not learn.
Two memories with different speeds solve the dilemma instead of negotiating
it.

Three design decisions worth explaining:

**The key is the top state.** What addresses the memory is not the raw
input, it is the summary the network itself makes of the context.
Content-based retrieval, like pattern completion in the hippocampus - a
piece of the context is enough to bring back the whole episode.

**The value is the error, not the observation.** The memory stores what the
hierarchy got wrong, not what happened. That way the memory corrects the
residual instead of competing with the weights - it is the same division of
labor that made the fast path work, after it had failed by being trained on
the whole signal.

**Writing happens when there is surprise.** The same number that decides how
much to think decides what is worth recording. In the brain it is what
novelty does to hippocampal synapses; here it comes for free, because the
surprise was already computed.

The store has a fixed size, on purpose. Memory that grows with time does not
fit on a device - and that is exactly the limitation of Google's solution
("Memory Caching", arXiv:2602.24281), which solves the same problem by
spending memory proportional to the sequence length. Here the budget is
fixed and the interesting question becomes what to *throw away*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dtypes import F


@dataclass(frozen=True)
class EpisodicConfig:
    n_slots: int = 128  # fixed budget: number of episodes that fit
    write_threshold: float = 1.5  # surprise (× the running mean) to record
    read_threshold: float = 0.6  # minimum similarity for the memory to weigh in
    temperature: float = 0.1  # softness of the retrieval (0 = single winner)
    read_gain: float = 1.0  # how much of the retrieved correction is applied
    decay: float = 0.9999  # passive forgetting of unused episodes
    # Reservoir sampling: when the store is full, each new episode enters
    # with probability n_slots/n_seen and evicts one at random.
    # This keeps an approximately uniform sample of *everything* that was
    # seen, instead of filling up with the most recent. Without this, a new
    # task sweeps the previous ones out of the store within minutes - and
    # consolidation is left with nothing old to replay, which is exactly
    # what happened.
    reservoir: bool = True


class EpisodicMemory:
    """Fixed-size key-value store, content-addressable."""

    __slots__ = ("cfg", "keys", "values", "frames", "prev_frames", "strength",
                 "n_written", "_surprise_avg", "_reads", "_hits", "_seen", "_rng")

    def __init__(self, key_dim: int, value_dim: int, cfg: EpisodicConfig | None = None):
        self.cfg = cfg or EpisodicConfig()
        self.keys = np.zeros((self.cfg.n_slots, key_dim), dtype=F)
        # The value is the residual - what the hierarchy missed - because
        # that is what the memory has to correct without competing with the
        # weights.
        self.values = np.zeros((self.cfg.n_slots, value_dim), dtype=F)
        # The raw observation is stored separately, only for sleep replay:
        # consolidation needs the frame as it happened, not the residual
        # (which was measured against a prediction that has since changed).
        self.frames = np.zeros((self.cfg.n_slots, value_dim), dtype=F)
        # And the *previous* frame, because an episode is not an observation
        # - it is an observation in a context. Replaying the frame alone,
        # outside the sequence it came from, is recalling a word without the
        # sentence: the network receives a huge error and the weights end up
        # in the wrong places. Measured: with decontextualized replay the
        # network did not even learn the first task (NRMSE 0.97, the same as
        # predicting zero).
        self.prev_frames = np.zeros((self.cfg.n_slots, value_dim), dtype=F)
        # Strength of each trace: rises when recorded or useful, falls on its
        # own. Slots with zero strength are empty.
        self.strength = np.zeros(self.cfg.n_slots, dtype=F)
        self.n_written = 0
        self._surprise_avg = 0.0
        self._reads = 0
        self._hits = 0
        self._seen = 0  # candidates seen, for the reservoir
        self._rng = np.random.default_rng(0)

    # ------------------------------------------------------------------
    @property
    def n_occupied(self) -> int:
        return int(np.count_nonzero(self.strength))

    @property
    def hit_rate(self) -> float:
        return self._hits / self._reads if self._reads else 0.0

    def clear(self) -> None:
        self.keys.fill(0.0)
        self.values.fill(0.0)
        self.frames.fill(0.0)
        self.prev_frames.fill(0.0)
        self.strength.fill(0.0)
        self.n_written = 0
        self._surprise_avg = 0.0
        self._reads = self._hits = 0
        self._seen = 0

    # ------------------------------------------------------------------
    def _similarity(self, key: np.ndarray) -> np.ndarray:
        """Cosine between the key and each occupied trace; -1 for the empty ones."""
        occupied = self.strength > 0
        sim = np.full(self.cfg.n_slots, -1.0, dtype=F)
        if not np.any(occupied):
            return sim
        kn = float(np.linalg.norm(key))
        if kn < 1e-8:
            return sim
        norms = np.linalg.norm(self.keys[occupied], axis=1)
        norms = np.maximum(norms, 1e-8)
        sim[occupied] = self.keys[occupied].dot(key) / (norms * kn)
        return sim

    def read(self, key: np.ndarray) -> tuple[np.ndarray | None, float]:
        """Retrieves the correction associated with the current context.

        Returns (correction, confidence). Confidence is the similarity of
        the best trace; below the threshold it returns None, because a
        memory that always weighs in is a memory that injects noise
        everywhere.
        """
        self._reads += 1
        sim = self._similarity(key)
        best = float(sim.max())
        if best < self.cfg.read_threshold:
            return None, best

        # Mean of the nearby traces, weighted by similarity. With low
        # temperature it approaches the single winner.
        mask = sim >= self.cfg.read_threshold
        w = np.exp((sim[mask] - best) / max(self.cfg.temperature, 1e-6))
        w = (w / w.sum()).astype(F)
        value = (w[:, None] * self.values[mask]).sum(axis=0).astype(F)

        # Using a trace reinforces it: what serves survives the cleanup.
        self.strength[mask] += F(0.1) * w
        self._hits += 1
        return value, best

    def write(self, key: np.ndarray, value: np.ndarray, frame: np.ndarray,
              prev_frame: np.ndarray, surprise: float) -> bool:
        """Records an episode if it was surprising enough.

        Returns True if it recorded. The threshold is *relative* to the
        running mean surprise: what counts as noteworthy depends on what is
        usual, not on a hand-tuned constant.
        """
        self._surprise_avg = 0.99 * self._surprise_avg + 0.01 * surprise
        if self._surprise_avg <= 0:
            return False
        if surprise < self.cfg.write_threshold * self._surprise_avg:
            return False

        # If a practically identical trace already exists, reinforce it
        # instead of duplicating. The threshold is tight on purpose: with
        # 0.95 the store *froze* - as soon as the traces covered the key
        # space, everything arriving was considered a repetition and nothing
        # ever entered again. A memory that stops recording is a buffer with
        # good manners.
        self._seen += 1
        sim = self._similarity(key)
        best_idx = int(np.argmax(sim))
        if float(sim[best_idx]) > 0.995:
            self.values[best_idx] = 0.5 * (self.values[best_idx] + value)
            self.frames[best_idx] = 0.5 * (self.frames[best_idx] + frame)
            self.prev_frames[best_idx] = 0.5 * (self.prev_frames[best_idx] + prev_frame)
            self.strength[best_idx] += F(0.5)
            return True

        slot = self._victim()
        if slot < 0:
            return False
        self.keys[slot] = key
        self.values[slot] = value
        self.frames[slot] = frame
        self.prev_frames[slot] = prev_frame
        self.strength[slot] = F(1.0)
        self.n_written += 1
        return True

    def _victim(self) -> int:
        """Which slot to sacrifice. Returns -1 to refuse the write.

        With the reservoir, a full store accepts the new episode only with
        probability n_slots/n_seen, and in that case evicts one at random.
        It is the classic algorithm for keeping a uniform sample of a stream
        without knowing its length in advance - and it is what makes old
        memories survive the arrival of a new task.
        """
        empty = np.flatnonzero(self.strength <= 0)
        if len(empty):
            return int(empty[0])
        if not self.cfg.reservoir:
            return int(np.argmin(self.strength))
        if self._rng.random() < self.cfg.n_slots / self._seen:
            return int(self._rng.integers(self.cfg.n_slots))
        return -1

    def decay(self) -> None:
        """Passive forgetting. Called once per frame."""
        if self.cfg.decay < 1.0:
            self.strength *= F(self.cfg.decay)
            self.strength[self.strength < 1e-3] = 0.0

    # ------------------------------------------------------------------
    def replay(self, n: int, rng: np.random.Generator) -> list[int]:
        """Chooses episodes to consolidate, favoring the strongest.

        The strong ones are those that repeated or were useful again - which
        is exactly the criterion one wants to distill into the weights. What
        happened once and never again stays in the fast memory until it
        fades.
        """
        occupied = np.flatnonzero(self.strength > 0)
        if len(occupied) == 0:
            return []
        p = self.strength[occupied].astype(np.float64)
        p = p / p.sum()
        n = min(n, len(occupied))
        return [int(i) for i in rng.choice(occupied, size=n, replace=False, p=p)]

    def stats(self) -> dict:
        return {
            "ocupados": self.n_occupied,
            "escritos": self.n_written,
            "taxa_de_acerto": round(self.hit_rate, 3),
            "força_média": round(float(self.strength[self.strength > 0].mean())
                                 if self.n_occupied else 0.0, 3),
        }
