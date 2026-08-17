"""The sparse predictive hierarchy.

One time step (one sensor frame) runs like this:

  1. The prediction descends. The top propagates itself in time (ẑ_L = tanh(A·z_L(t-1)))
     and from there it descends level by level until it generates an expected
     frame ẑ_0. This happens *before* seeing the data: ε_0 on the first
     evaluation is the open-loop prediction error.
  2. The surprise rises. Compare, apply the threshold (|ε| < θ is not
     transmitted, silence is zero cost) and what remains corrects the states
     above. Repeat until the total surprise converges: *early exit* for
     trivial inputs, more iterations for the hard ones. Compute is
     proportional to surprise, not to the network's capacity.
  3. Learning is local. ΔW ∝ ε ⊗ z: only quantities present at the synapse.
     No backprop, no graph, no stored activations.

The top's latent state persists across frames - it is the working memory.
"""

from __future__ import annotations

import numpy as np

from . import activations
from .config import PCConfig
from .device import AnalogArray, DeviceModel, quantize_adc
from .dtypes import F
from .episodic import EpisodicConfig, EpisodicMemory
from .gated import GatedTransition
from .layer import PCLayer
from .mixture import TopMixture
from .metrics import EXIT_CEILING, EXIT_EXPLAINED, EXIT_STALLED, StepTrace
from .precision import Precision, UnitPrecision
from .temporal import Transition, timescales


class PCNetwork:
    def __init__(self, config: PCConfig | None = None) -> None:
        self.cfg = config or PCConfig()
        rng = np.random.default_rng(self.cfg.seed)

        sizes = self.cfg.sizes
        self.L = len(sizes) - 1  # index of the top level

        # layers[l] generates level l from level l+1.
        self.layers: list[PCLayer] = []
        for l in range(self.L):
            act = self.cfg.sensory_activation if l == 0 else self.cfg.latent_activation
            self.layers.append(
                PCLayer(sizes[l], sizes[l + 1], act, rng, self.cfg.init_scale,
                        gated=self.cfg.gated_layers)
            )

        # Temporal transition of the top: a linear recurrence + tanh.
        n_top = sizes[-1]
        self.A = (
            np.eye(n_top, dtype=F)
            + (rng.standard_normal((n_top, n_top)) * (0.1 / np.sqrt(n_top))).astype(F)
            if self.cfg.use_transition
            else np.eye(n_top, dtype=F)
        )
        self._f_top, self._fprime_top = activations.get(self.cfg.latent_activation)
        # Mixture of dynamics at the top. With n_dynamics=1 it is exactly the
        # single matrix from before and nothing changes.
        self.mixture = (
            TopMixture(n_top, self.cfg.n_dynamics, rng, self.cfg.dynamics_tau)
            if self.cfg.n_dynamics > 1
            else None
        )
        self.gated = (
            GatedTransition(n_top, rng, eligibility=self.cfg.eligibility,
                            input_dim=sizes[0] if self.cfg.thalamic else 0)
            if self.cfg.gated_transition else None
        )

        # States, preallocated (mirroring the C static arrays).
        self.z: list[np.ndarray] = [np.zeros(n, dtype=F) for n in sizes]
        self.prior_top = np.zeros(n_top, dtype=F)
        self.a_top = np.zeros(n_top, dtype=F)
        # The state of each level at the previous frame. Only the latents need
        # it: the sensory level is clamped by the data and does not predict itself.
        self._z_prev: list[np.ndarray] = [np.zeros(n, dtype=F) for n in sizes]

        # Timescales: one temporal model per latent level, ever slower as one
        # goes up (see temporal.py). The top keeps using `A`, which is already
        # dense and already knows how to live on a crossbar.
        # λ decreases with height: the top integrates over the longest window.
        lams = timescales(self.L, self.cfg.timescale_base, self.cfg.timescale_ratio)
        self._top_lam = (
            lams[self.L - 1] if self.cfg.level_transition != "none" else 1.0
        )
        self.transitions: list[Transition | None] = [None] * (self.L + 1)
        if self.cfg.level_transition != "none":
            for l in range(1, self.L):
                self.transitions[l] = Transition(
                    sizes[l],
                    self.cfg.level_transition,
                    lams[l - 1],
                    rng,
                    activations.get(self.cfg.latent_activation),
                )
        self.lams = lams

        # Precision: the gain with which each channel's error counts (see
        # precision.py). Two channels per latent level, the one coming from
        # above and the one from its own past, because they are errors of
        # different natures.
        def _prec(n: int):
            if not self.cfg.use_precision:
                return UnitPrecision()
            return Precision(n, self.cfg.precision_per_unit, self.cfg.precision_lr)

        self.prec_hier = [_prec(sizes[l]) for l in range(self.L)]
        self.prec_temp = [None] + [_prec(sizes[l]) for l in range(1, self.L + 1)]

        # Fast path: predicts the next frame directly from the previous one.
        # Linear and with no nonlinearity at all - it is the "coarse but
        # immediate" path.
        n0 = sizes[0]
        self.A0 = (
            (rng.standard_normal((n0, n0)) * (0.1 / np.sqrt(n0))).astype(F)
            if self.cfg.fast_path
            else None
        )
        self._fast = np.zeros(n0, dtype=F)

        # Episodic memory: off until `attach_memory`. The key is the state of
        # the top (the summary of the context), the value is the error the
        # hierarchy made - the memory corrects the residual, it does not
        # compete with the weights.
        self.memory: EpisodicMemory | None = None
        self._recall = np.zeros(n0, dtype=F)
        self._recall_conf = 0.0
        self._replay_rng = np.random.default_rng(self.cfg.seed + 31337)

        # Diagnostic: clamp dimensions of the top state to a given value, to
        # ask "what if the network *knew* which world it is in?". Without
        # this, an indicator placed at the input is 3 dimensions out of 67 and
        # barely pushes the latent - the oracle comes out weak and the
        # conclusion comes out wrong.
        self.top_clamp: np.ndarray | None = None

        # 1-bit critic: snapshot of the weights before the last learning
        # step, and moving average of the surprise for the veto.
        self._critic_snapshot = None
        self._critic_ema = 0.0

        # Accounting constants.
        self.macs_down_per_pass = sum(lay.macs_down() for lay in self.layers)
        self.eps_per_pass = sum(sizes)  # one ε per unit, top included

        # Substrate: None = ideal float32.
        self.device_model: DeviceModel | None = None
        self._A_device = None
        self.A_eff = self.A

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------
    @property
    def _z_top_prev(self) -> np.ndarray:
        """The state of the top at the previous frame."""
        return self._z_prev[self.L]

    def reset(self) -> None:
        """Zeroes the latent states. The weights stay."""
        for z in self.z:
            z.fill(0.0)
        for z in self._z_prev:
            z.fill(0.0)
        self.prior_top.fill(0.0)

    def snapshot_state(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Saves the dynamic state (not the weights).

        The network is online and the state crosses time, so *evaluating*
        touches the state. Without this, measuring the model mid-training
        alters the training - and the learning curve ends up measuring the
        ruler.
        """
        return ([z.copy() for z in self.z], [z.copy() for z in self._z_prev])

    def restore_state(self, snapshot: tuple[list[np.ndarray], list[np.ndarray]]) -> None:
        zs, prevs = snapshot
        for dst, src in zip(self.z, zs):
            dst[:] = src
        for dst, src in zip(self._z_prev, prevs):
            dst[:] = src

    # ------------------------------------------------------------------
    # substrate
    # ------------------------------------------------------------------
    def attach_device(self, model: DeviceModel) -> None:
        """Programs the weights onto a concrete crossbar (with its defects).

        The deviations are sampled once, here - reprogramming with the same
        parameters and the same seed gives the *same* device. A different
        crossbar is another seed, and that is why any tolerance measurement
        has to be done over several of them.
        """
        self.device_model = model
        for l, lay in enumerate(self.layers):
            lay.attach_device(AnalogArray(lay.W.shape, model, model.seed + 7919 * l))
        if model.include_transition:
            self._A_device = AnalogArray(self.A.shape, model, model.seed + 104729)
            self.A_eff = self._A_device.program(self.A)
        else:
            self._A_device = None
            self.A_eff = self.A

    def detach_device(self) -> None:
        self.device_model = None
        self._A_device = None
        self.A_eff = self.A
        for lay in self.layers:
            lay.detach_device()

    def _refresh_transition(self) -> None:
        if self._A_device is not None:
            self.A_eff = self._A_device.program(self.A)

    # ------------------------------------------------------------------
    # the descending path
    # ------------------------------------------------------------------
    def _temporal_prior(self) -> np.ndarray:
        """ẑ_L(t) = f(A · z_L(t-1)): what the top expects of itself."""
        if not self.cfg.use_transition:
            self.a_top[:] = self._z_top_prev
            self.prior_top[:] = self._z_top_prev
            return self.prior_top
        z_prev = self._z_top_prev
        if self.cfg.state_leak:
            z_prev = (F(1.0 - self.cfg.state_leak) * z_prev).astype(F, copy=False)
        if self.gated is not None:
            # The mixture (1-g)z + g·tanh(Az) already includes the
            # nonlinearity and the retention: the prior is returned directly,
            # with no extra tanh or λ.
            u = self._z_prev[0] if self.cfg.thalamic else None
            self.prior_top[:] = self.gated.predict(z_prev, u)
            self.a_top[:] = self.prior_top
            return self.prior_top
        if self.mixture is not None:
            self.a_top[:] = self.mixture.predict(z_prev)
        else:
            np.dot(self.A_eff, z_prev, out=self.a_top)
        if self._A_device is not None:
            self.a_top[:] = self._A_device.read(self.a_top)
        # The top's timescale. Without this the top is a dense A with no
        # retention at all - that is, the *fastest* level of the network,
        # exactly the opposite of what a hierarchy of temporal windows asks
        # for.
        if self._top_lam < 1.0:
            self.a_top *= F(self._top_lam)
            self.a_top += F(1.0 - self._top_lam) * z_prev
        self.prior_top[:] = self._f_top(self.a_top)
        return self.prior_top

    def _combine_priors(self, level: int, from_above: np.ndarray) -> np.ndarray:
        """Combines what the level above predicts with what one's own past predicts.

        Two independent sources of information about the same quantity, each
        with its own precision: the precision-weighted average is the Bayesian
        answer, and it is also the minimum of the energy if nothing else
        pulled the state. Without a transition at the level, only the
        prediction from above remains - which is exactly the behavior of
        step 1.
        """
        trans = self.transitions[level]
        if trans is None:
            return from_above
        from_past = trans.predict(self._z_prev[level])
        ph = np.asarray(self.prec_hier[level].value, dtype=F)
        pt = np.asarray(self.prec_temp[level].value, dtype=F)
        return ((ph * from_above + pt * from_past) / (ph + pt)).astype(F, copy=False)

    def predict_next(self) -> np.ndarray:
        """The frame the network expects to see next, without looking at the data.

        Open-loop prediction, useful for evaluating and for comparing against
        the C inferrer. Does not alter z or the state that crosses frames
        (it only touches scratch buffers, which the next step rewrites).
        """
        z = self._temporal_prior().copy()
        for l in range(self.L - 1, 0, -1):
            z = self._combine_priors(l, self.layers[l].predict(z)).copy()
        deep = self.layers[0].predict(z)
        out = (deep + self._fast_prediction()).astype(F, copy=False)
        if self.memory is not None:
            value, _ = self.memory.read(self._z_prev[self.L])
            if value is not None:
                out = (out + F(self.memory.cfg.read_gain) * value).astype(F)
        return out

    def _fast_prediction(self) -> np.ndarray:
        """The fast path: A_0 · (previous frame). Zero if it is off."""
        if self.A0 is None:
            return F(0.0)
        np.dot(self.A0, self._z_prev[0], out=self._fast)
        return self._fast

    # ------------------------------------------------------------------
    # episodic memory
    # ------------------------------------------------------------------
    def attach_memory(self, cfg: EpisodicConfig | None = None) -> EpisodicMemory:
        """Attaches a fixed-size episodic store (see episodic.py)."""
        self.memory = EpisodicMemory(
            key_dim=self.cfg.sizes[-1], value_dim=self.cfg.sizes[0], cfg=cfg
        )
        return self.memory

    def detach_memory(self) -> None:
        self.memory = None
        self._recall.fill(0.0)
        self._recall_conf = 0.0

    def _memory_recall(self) -> np.ndarray:
        """What the memory says will go wrong, from the current context.

        Addressed by the top state of the previous frame - that is the
        summary of the context the network has before seeing the new data.
        """
        self._recall.fill(0.0)
        self._recall_conf = 0.0
        if self.memory is None:
            return self._recall
        value, conf = self.memory.read(self._z_prev[self.L])
        self._recall_conf = conf
        if value is not None:
            self._recall[:] = F(self.memory.cfg.read_gain) * value
        return self._recall

    def dream(self, n_dreams: int = 16, noise: float = 0.15,
              anti: bool = False, anti_scale: float = 0.1,
              seed_states: np.ndarray | None = None) -> int:
        """REM: the cortex generates its own patterns and trains on them.

        Seeds the top with a stored context (plus noise) or with pure noise,
        descends through the generative model to *dream* a frame, and learns
        on it. It seems circular - training on what the model itself
        generates - and that is exactly the point: anchor the mapping that
        already exists, so that new learning deforms it less. It is Robins'
        (1995) pseudorehearsal, which REM sleep was already doing first:
        rehearsing old knowledge without storing the old data.
        """
        snapshot = self.snapshot_state()
        dreamed = 0
        # Nightmares (Crick & Mitchison 1983, "we dream in order to forget";
        # Hopfield 1983, unlearning): what the network generates spontaneously
        # and corresponds to nothing real - the spurious mixtures, the
        # compromise between worlds - receives NEGATIVE and small plasticity.
        # The asymmetry that protects true knowledge is automatic: the real is
        # reinforced awake every day; the spurious only exists in dreams, and
        # in dreams it gets hit. Our earlier dream failed for having the sign
        # backwards - training TOWARD what is generated anchors mediocrity.
        from dataclasses import replace as _replace
        original = self.cfg
        if anti:
            self.cfg = _replace(original,
                                w_lr=-original.w_lr * anti_scale,
                                a_lr=-original.a_lr * anti_scale,
                                fast_path_lr=0.0)
        try:
            for k in range(n_dreams):
                if seed_states is not None:
                    # Directed nightmares: the caller chooses the contexts to
                    # visit - typically mixtures of real states, which is
                    # where the spurious compromise lives.
                    z = seed_states[k % len(seed_states)].astype(F).copy()
                elif self.memory is not None and self.memory.n_occupied:
                    slot = self.memory.replay(1, self._replay_rng)
                    z = self.memory.keys[slot[0]].copy()
                else:
                    z = np.zeros(self.cfg.sizes[-1], dtype=F)
                z += F(noise) * self._replay_rng.standard_normal(z.size).astype(F)

                # dream: descend the generative model from that context
                self._z_prev[self.L][:] = z
                zz = self._f_top(self.A_eff.dot(z)).astype(F, copy=False)
                for l in range(self.L - 1, 0, -1):
                    zz = self.layers[l].predict(zz).copy()
                frame = self.layers[0].predict(zz).copy()
                self._z_prev[0][:] = frame  # the dream is its own context

                self.step(frame, learn=True, use_memory=False)
                dreamed += 1
        finally:
            self.cfg = original
            self.restore_state(snapshot)
        return dreamed

    def downscale(self, factor: float = 0.01) -> None:
        """Synaptic renormalization (Tononi & Cirelli): sleep shrinks all
        synapses multiplicatively, preserving the ratios. Wakefulness
        potentiates; sleep restores the budget."""
        f = F(1.0 - factor)
        for lay in self.layers:
            lay.W *= f
            lay.refresh_device()
        self.A *= f
        if self.A0 is not None:
            self.A0 *= f
        self._refresh_transition()

    def sleep(self, cycles: int = 3, sws: int = 16, rem: int = 8,
              downscale: float = 0.0, lr_scale: float = 0.1) -> dict:
        """One night: alternating SWS (episodic replay) / REM (dreaming).

        The order matters and it is the physiological one - first the
        episodic goes in (hippocampus -> cortex), then the dream re-anchors
        the whole.
        """
        # Sleep plasticity is not waking plasticity: the acetylcholine level
        # changes and replay acts with reduced gain. Dreaming at the full
        # rate drags the weights toward what the model generates badly
        # (measured: mean 1.01 against 0.85 without sleep).
        from dataclasses import replace as _replace
        original = self.cfg
        self.cfg = _replace(original, w_lr=original.w_lr * lr_scale,
                            a_lr=original.a_lr * lr_scale,
                            fast_path_lr=original.fast_path_lr * lr_scale)
        replayed = dreamed = 0
        try:
            for _ in range(cycles):
                replayed += self.consolidate(n_episodes=sws)
                dreamed += self.dream(n_dreams=rem)
        finally:
            self.cfg = original
        if downscale > 0.0:
            self.downscale(downscale)
        return {"replayed": replayed, "dreamed": dreamed}

    def consolidate(self, n_episodes: int = 16, passes: int = 1) -> int:
        """Sleep: replays episodes and distills them into the weights.

        Runs the same local rule as always over the stored frames, as if
        they were happening now. What repeated - the strong traces - is
        chosen first. Returns how many episodes were replayed.

        This is offline on purpose: it does not happen while observing, and
        so it does not compete with inference or spend energy at the sensor.
        """
        if self.memory is None:
            return 0
        snapshot = self.snapshot_state()
        replayed = 0
        try:
            for _ in range(passes):
                for slot in self.memory.replay(n_episodes, self._replay_rng):
                    # Restore the context before reliving the episode: the top
                    # state the memory stored, and the previous frame the fast
                    # path needs. It is pattern completion - the key brings
                    # back the state, not just the content.
                    self._z_prev[self.L][:] = self.memory.keys[slot]
                    self._z_prev[0][:] = self.memory.prev_frames[slot]
                    for l in range(1, self.L):
                        self._z_prev[l].fill(0.0)
                    self.step(self.memory.frames[slot], learn=True, use_memory=False)
                    replayed += 1
        finally:
            self.restore_state(snapshot)
        return replayed

    def _reconstruct(self, slot: int) -> np.ndarray:
        """The observation as it happened, stored in the episode."""
        assert self.memory is not None
        return self.memory.frames[slot]

    # ------------------------------------------------------------------
    # one time step
    # ------------------------------------------------------------------
    def step(self, x: np.ndarray, learn: bool = True,
             use_memory: bool = True) -> StepTrace:
        """Processes one frame. Returns the instrumentation trace."""
        cfg = self.cfg
        x = np.asarray(x, dtype=F)
        if x.shape != (cfg.sizes[0],):
            raise ValueError(
                f"trama com forma {x.shape}, esperado ({cfg.sizes[0]},)"
            )

        trace = StepTrace()
        trace.target_rms = float(np.sqrt(np.mean(np.square(x))))

        # --- 0. the critic judges the previous learning -----------------
        # The update from t-1 only proves itself in the prediction at t. If
        # the open-loop surprise of this frame exceeds the usual, it is undone.
        if cfg.critic_retract > 0.0 and self._critic_snapshot is not None:
            pred = self.predict_next()
            e_now = float(np.mean((x - pred) ** 2))
            if self._critic_ema > 0.0 and e_now > cfg.critic_retract * self._critic_ema:
                for lay, W in zip(self.layers, self._critic_snapshot["W"]):
                    lay.W[:] = W
                    lay.refresh_device()
                if self.gated is not None:
                    self.gated.A[:] = self._critic_snapshot["gA"]
                    self.gated.G[:] = self._critic_snapshot["gG"]
                    self.gated.b[:] = self._critic_snapshot["gb"]
            self._critic_ema = 0.98 * self._critic_ema + 0.02 * e_now
            self._critic_snapshot = None
        elif cfg.critic_retract > 0.0:
            pred = self.predict_next()
            e_now = float(np.mean((x - pred) ** 2))
            self._critic_ema = (0.98 * self._critic_ema + 0.02 * e_now
                                if self._critic_ema > 0.0 else e_now)

        # --- 1. the prediction descends ---------------------------------
        # Each latent level receives two predictions: the one from the level
        # above and the one from its own past, at its own timescale. They are
        # combined by the precision of each.
        fast = np.asarray(self._fast_prediction(), dtype=F).copy()
        # The memory weighs in before the data is seen, from the context:
        # "last time the world looked like this, the prediction missed by
        # this much".
        key = self._z_prev[self.L].copy()
        recall = (self._memory_recall().copy() if (use_memory and self.memory)
                  else np.zeros_like(fast))
        fast = (fast + recall).astype(F, copy=False)
        prior_top = self._temporal_prior()
        temporal_prior: list[np.ndarray | None] = [None] * (self.L + 1)
        temporal_prior[self.L] = prior_top
        self.z[self.L][:] = prior_top
        if self.top_clamp is not None:
            self.z[self.L][:len(self.top_clamp)] = self.top_clamp
        for l in range(self.L - 1, 0, -1):
            from_above = self.layers[l].predict(self.z[l + 1])
            trans = self.transitions[l]
            if trans is not None:
                temporal_prior[l] = trans.predict(self._z_prev[l]).copy()
            self.z[l][:] = self._combine_priors(l, from_above)
        self.z[0][:] = x  # the sensory level is clamped by the data

        # --- 2. the surprise rises (iterative settling) ----------------
        eps_thr: list[np.ndarray] = [None] * (self.L + 1)  # type: ignore[list-item]
        eps_mod: list[np.ndarray] = [None] * self.L  # type: ignore[list-item]
        eps_temp: list[np.ndarray] = [None] * (self.L + 1)  # type: ignore[list-item]
        prev_surprise = np.inf

        for k in range(cfg.max_iters + 1):
            # The "surprise" is the energy F = ½Σ‖ε_l‖², summed over the
            # levels. This is the quantity the settling dynamics descends, so
            # it is the only one usable as a convergence criterion. (The sum
            # of the |ε| does not work: the latent levels start with ε=0 by
            # construction - they were initialized by the prediction from
            # above - and grow as they take on their share of the
            # explanation, which makes the L1 sum oscillate even while the
            # energy descends.)
            surprise = 0.0
            transmitted = 0.0

            # hierarchical errors: ε^h_l = z_l − f(W_l z_{l+1})
            # At the sensory level the fast path is added: the hierarchy only
            # has to explain what it could not.
            for l in range(self.L):
                zhat = self.layers[l].predict(self.z[l + 1])
                eps = self.z[l] - zhat
                if l == 0 and (self.A0 is not None or self.memory is not None):
                    eps = (eps - fast).astype(F, copy=False)
                if k == 0 and l == 0:
                    # ε_0 before any correction: the honest prediction error.
                    trace.pred_error = eps.copy()
                pi = self.prec_hier[l]
                surprise += 0.5 * float(np.dot(eps, pi.weight(eps)))
                et = self._threshold(eps, trace, pi)
                transmitted += 0.5 * float(np.dot(et, pi.weight(et)))
                eps_thr[l] = et
                eps_mod[l] = self.layers[l].modulated_error(pi.weight(et))

            # temporal errors: ε^t_l = z_l − f(A_l z_l(t-1)), per latent level.
            # At the top it is the only error there is (there is no level above).
            for l in range(1, self.L + 1):
                prior = temporal_prior[l]
                if prior is None:
                    continue
                eps = self.z[l] - prior
                pi = self.prec_temp[l]
                surprise += 0.5 * float(np.dot(eps, pi.weight(eps)))
                et = self._threshold(eps, trace, pi)
                transmitted += 0.5 * float(np.dot(et, pi.weight(et)))
                eps_temp[l] = et
            eps_thr[self.L] = eps_temp[self.L]

            trace.macs_down += self.macs_down_per_pass
            trace.surprise.append(surprise)
            trace.transmitted.append(transmitted)

            # --- when to stop ------------------------------------------
            # Nothing passed the threshold: the prediction explained the
            # input. Silence.
            if transmitted <= cfg.settle_tol:
                trace.exit_reason = EXIT_EXPLAINED
                break
            # Iterating without profit: one more iteration does not pay for
            # what it costs.
            if k > 0:
                gain = prev_surprise - surprise
                if gain < cfg.settle_min_gain or (
                    cfg.settle_rel_tol > 0.0
                    and gain < cfg.settle_rel_tol * max(prev_surprise, 1e-12)
                ):
                    trace.exit_reason = EXIT_STALLED
                    break
            if k == cfg.max_iters:
                trace.exit_reason = EXIT_CEILING
                break
            prev_surprise = surprise

            # --- correction of the states above --------------------------
            # Δz_l = z_lr · ( W_{l-1}ᵀ(π ε̃^h_{l-1} ⊙ f') − π ε̃^h_l − π ε̃^t_l )
            #
            # Three forces on each latent state: the error rising from below
            # (what the level has not yet explained), the error against the
            # prediction of the level above, and the error against what it
            # itself predicted at the previous instant. It is the last one
            # that the step 1 version only had at the top.
            for l in range(1, self.L + 1):
                lay = self.layers[l - 1]
                up = lay.backward(eps_mod[l - 1], eps_thr[l - 1])
                nnz = int(np.count_nonzero(eps_thr[l - 1]))
                trace.macs_up += lay.macs_up(nnz)
                trace.macs_up_dense += lay.macs_down()

                pull = up
                if l < self.L and eps_thr[l] is not None:
                    pull = pull - self.prec_hier[l].weight(eps_thr[l])
                if eps_temp[l] is not None:
                    pull = pull - self.prec_temp[l].weight(eps_temp[l])
                dz = F(self._z_lr_for(l)) * pull
                if cfg.z_clip > 0.0:
                    np.clip(dz, -cfg.z_clip, cfg.z_clip, out=dz)
                self.z[l] += dz
                if cfg.z_max > 0.0:
                    np.clip(self.z[l], -cfg.z_max, cfg.z_max, out=self.z[l])
                if cfg.state_sparsity > 0.0:
                    self._compete(self.z[l], cfg.state_sparsity)
                if l == self.L and self.top_clamp is not None:
                    k = len(self.top_clamp)
                    self.z[l][:k] = self.top_clamp
            trace.iters += 1

        # --- 3. local learning -------------------------------------
        # The critic needs the PRE-update state to be able to retract:
        # the snapshot is taken before learning, judged on the next frame.
        if learn and cfg.critic_retract > 0.0:
            snap = {"W": [lay.W.copy() for lay in self.layers]}
            if self.gated is not None:
                snap["gA"] = self.gated.A.copy()
                snap["gG"] = self.gated.G.copy()
                snap["gb"] = self.gated.b.copy()
            self._critic_snapshot = snap
        if learn and cfg.plasticity_gate > 0.0:
            # Neuromodulation: plasticity only opens when there is novelty.
            self._surprise_avg = (0.99 * getattr(self, "_surprise_avg", 0.0)
                                  + 0.01 * trace.open_loop_surprise)
            if trace.open_loop_surprise < cfg.plasticity_gate * self._surprise_avg:
                learn = False
        if learn:
            for l in range(self.L):
                self.layers[l].learn(
                    eps_mod[l],
                    self.z[l + 1],
                    cfg.w_lr,
                    cfg.weight_decay,
                    cfg.grad_clip,
                    cfg.metaplasticity,
                    cfg.meta_decay,
                )
                self.prec_hier[l].learn(self.z[l] - self.layers[l].zhat)
                if eps_thr[l] is not None or l == 0:
                    self.layers[l].learn_gate(
                        eps_thr[l] if l > 0 else eps_thr[0],
                        self.z[l + 1], cfg.w_lr, cfg.grad_clip,
                    )
            if cfg.use_transition:
                self._learn_transition(eps_thr[self.L])
            # each transition learns from its own temporal error and its own
            # past - nothing more local than this
            for l in range(1, self.L):
                trans = self.transitions[l]
                if trans is not None and eps_temp[l] is not None:
                    trans.learn(eps_temp[l], self._z_prev[l], cfg.b_lr, cfg.grad_clip)
            for l in range(1, self.L + 1):
                if temporal_prior[l] is not None:
                    self.prec_temp[l].learn(self.z[l] - temporal_prior[l])
            self._anchor_precisions()
            if self.A0 is not None:
                # The fast path learns from its *own* error (x − A_0·x_prev),
                # not from the residual left over after the hierarchy.
                #
                # Training both paths with the same residual seems elegant
                # and is a disaster: both chase the same signal at the same
                # time, with nothing to assign credit, and they fight over
                # it. Measured: ETTm1 went from 0.230 to 0.724. With separate
                # targets each path does what it knows - the fast one
                # predicts what it can on its own, the deep one explains what
                # remained - which is the division of labor the two-path
                # hypothesis claims to exist.
                # Normalized step (NLMS). The delta rule ΔA ∝ ε·xᵀ is only
                # stable for lr < 1/‖x‖², and ‖x‖ depends on the frame and on
                # the signal's scale - a fixed lr is always wrong for some
                # signal. Dividing by ‖x‖² makes the step dimensionless:
                # `fast_path_lr` comes to mean "what fraction of the error to
                # correct per frame", between 0 and ~2, and stops being a
                # number to tune per dataset. It is the same trap as z_lr,
                # and the same fix.
                #
                # The normalization comes *before* the clip, on purpose:
                # clipping first reintroduces the scale dependence it exists
                # to eliminate (with the signal 100x larger, every entry
                # saturates at the clip and the direction of the step is
                # lost).
                # The fast path learns against its own target, not counting
                # what the memory retrieved - otherwise the two go back to
                # fighting over the same residual.
                own_error = (self.z[0] - fast + recall).astype(F, copy=False)
                norm = float(np.dot(self._z_prev[0], self._z_prev[0])) + 1e-6
                dA0 = np.outer(own_error / norm, self._z_prev[0]).astype(F, copy=False)
                if cfg.grad_clip > 0.0:
                    np.clip(dA0, -cfg.grad_clip, cfg.grad_clip, out=dA0)
                self.A0 += F(cfg.fast_path_lr) * dA0

        # --- 4. episodic memory --------------------------------------
        # What surprised is recorded, with the key that was in effect when
        # the prediction was made - so that the next time the context looks
        # like this one, the correction is there.
        if self.memory is not None:
            if learn and trace.pred_error is not None:
                self.memory.write(
                    key, trace.pred_error, self.z[0], self._z_prev[0],
                    trace.open_loop_surprise,
                )
            self.memory.decay()

        # the state crosses over to the next frame
        self._z_prev[0][:] = self.z[0]
        for l in range(1, self.L + 1):
            self._z_prev[l][:] = self.z[l]
        return trace

    # ------------------------------------------------------------------
    def _learn_transition(self, eps_top_thr: np.ndarray) -> None:
        """ΔA ∝ (ε_top ⊙ f') ⊗ z_top(t-1). Also local."""
        cfg = self.cfg
        if self.gated is not None:
            self.gated.learn(eps_top_thr, cfg.a_lr, cfg.grad_clip)
            return
        eps_mod = eps_top_thr * self._fprime_top(self.a_top)
        if self.mixture is not None:
            self.mixture.learn(eps_mod, cfg.a_lr, cfg.proto_lr, cfg.grad_clip)
            return
        dA = np.outer(eps_mod, self._z_top_prev).astype(F, copy=False)
        if cfg.grad_clip > 0.0:
            np.clip(dA, -cfg.grad_clip, cfg.grad_clip, out=dA)
        if cfg.weight_decay > 0.0:
            self.A *= F(1.0 - cfg.a_lr * cfg.weight_decay)
        self.A += F(cfg.a_lr) * dA
        self._refresh_transition()

    def _anchor_precisions(self) -> None:
        """Pins the scale of the precisions at the sensory channel.

        π of level 0 always stays at 1 and all the others come to mean "this
        error is worth k times a sensory error". Without this anchor, all
        precisions rise together as the network learns (π → 1/⟨ε²⟩), the
        Hessian inflates, the adaptive settling step shrinks to avoid
        diverging, and inference stops happening.
        """
        if not self.cfg.use_precision:
            return
        offset = float(np.mean(self.prec_hier[0].log_pi))
        for p in self.prec_hier:
            p.rescale(offset)
        for p in self.prec_temp[1:]:
            p.rescale(offset)

    @staticmethod
    def _compete(z: np.ndarray, frac: float) -> None:
        """Only the k strongest units survive; the others go to zero.

        Lateral inhibition, in-place. It is the mechanism by which different
        contexts come to use different populations - and two worlds that use
        different neurons do not erase each other.
        """
        k = max(1, int(round(frac * z.size)))
        if k >= z.size:
            return
        cut = np.partition(np.abs(z), z.size - k)[z.size - k]
        z[np.abs(z) < cut] = 0.0

    def _z_lr_for(self, level: int) -> float:
        """Settling step of level `level`, bounded by stability.

        The Hessian of the energy with respect to z_l is

            π^h_l·I  +  π^t_l·I  +  π^h_{l-1}·W_{l-1}ᵀW_{l-1}

        (the temporal term contributes identity because z_l(t-1) is fixed),
        so the largest eigenvalue is π^h_l + π^t_l + π^h_{l-1}·σ_max(W)² and
        the stable step is 2 over that. With precision on, the gains enter
        here - it is what prevents a high-precision level from destabilizing
        the settling.
        """
        if not self.cfg.adaptive_z_lr:
            return self.cfg.z_lr
        lam = self.prec_hier[level - 1].scalar * self.layers[level - 1].sigma_max**2
        if level < self.L:
            lam += self.prec_hier[level].scalar
        if self.transitions[level] is not None or level == self.L:
            lam += self.prec_temp[level].scalar
        return min(self.cfg.z_lr, self.cfg.z_lr_safety * 2.0 / max(lam, 1e-12))

    def _threshold(self, eps: np.ndarray, trace: StepTrace, prec=None) -> np.ndarray:
        """|ε| < θ is not transmitted. This is where real sparsity is born.

        The threshold applies to the *normalized* error √π·ε, not to the raw
        error: this way θ means "half a standard deviation of surprise" at
        any level, instead of meaning different things in the raw signal and
        in the abstract latent. Without precision, √π = 1 and it goes back to
        being the step 1 threshold.

        And this is where the ADC comes in: only what passes the threshold is
        converted, so quantization applies after silencing, not before.
        """
        trace.eps_total += eps.size
        if self.cfg.theta <= 0.0:
            out = eps.astype(F, copy=False)
        else:
            scale = eps if prec is None else prec.normalize(eps)
            mask = np.abs(scale) >= self.cfg.theta
            trace.eps_silenced += int(eps.size - np.count_nonzero(mask))
            out = np.where(mask, eps, F(0.0)).astype(F, copy=False)

        m = self.device_model
        if m is not None and m.adc_bits > 0:
            out = quantize_adc(out, m.adc_bits, m.adc_range)
        return out

    # ------------------------------------------------------------------
    # sequences
    # ------------------------------------------------------------------
    def run(
        self, frames: np.ndarray, learn: bool = True, reset: bool = False
    ) -> list[StepTrace]:
        """Runs a sequence of frames (n_frames, n_sensory)."""
        frames = np.asarray(frames, dtype=F)
        if frames.ndim != 2:
            raise ValueError("frames tem de ter forma (n_frames, n_sensory)")
        if reset:
            self.reset()
        return [self.step(frame, learn=learn) for frame in frames]

    # ------------------------------------------------------------------
    # weights (boundary with the C)
    # ------------------------------------------------------------------
    @property
    def weights(self) -> list[np.ndarray]:
        return [lay.W for lay in self.layers]

    def state_dict(self) -> dict:
        d = {f"W{l}": lay.W.copy() for l, lay in enumerate(self.layers)}
        d["A"] = self.A.copy()
        return d

    def load_state_dict(self, d: dict) -> None:
        for l, lay in enumerate(self.layers):
            W = np.asarray(d[f"W{l}"], dtype=F)
            if W.shape != lay.W.shape:
                raise ValueError(f"W{l}: forma {W.shape}, esperado {lay.W.shape}")
            lay.W[:] = W
            lay.refresh_device()
        A = np.asarray(d["A"], dtype=F)
        if A.shape != self.A.shape:
            raise ValueError(f"A: forma {A.shape}, esperado {self.A.shape}")
        self.A[:] = A
        self._refresh_transition()
