"""Configuration of the predictive network.

A single, serializable dataclass that is at the same time the contract with
the C inference engine: everything in here must be expressible as a `#define`
or as a static array.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PCConfig:
    # Hierarchy. sizes[0] is the sensory level (the observed frame), the last
    # is the top. Default: 64 -> 32 -> 16 -> 8, as in the plan.
    sizes: tuple[int, ...] = (64, 32, 16, 8)

    # Generator activation per level. Level 0 is the real signal, hence identity;
    # the latent levels use tanh (bounds the state, gives a cheap derivative).
    sensory_activation: str = "identity"
    latent_activation: str = "tanh"

    # --- settling dynamics -------------------------------------------------
    # Ceiling on settling steps. It is a ceiling, not a cost: the early exit
    # keeps the average at ~8. Note there is a structural minimum, the
    # sensory error climbs one level per iteration, so below `n_levels - 1`
    # the top never gets corrected and the network predicts zero.
    max_iters: int = 25
    # z_lr is a descent step on the gradient of the energy ½Σ‖ε‖²: it is only
    # stable for z_lr < 2/(1 + σ_max(W)²). Since σ_max grows with learning, the
    # safe value is low, at 0.5 the network diverges. See also `z_clip`.
    z_lr: float = 0.2
    # With this on, each level uses min(z_lr, z_lr_safety · 2/(1+σ_max²)) with
    # σ_max estimated by power iteration over the weights the device actually
    # has. It costs two crossbar reads per weight update and is what keeps
    # device variability from crossing the stability limit without anyone
    # noticing.
    adaptive_z_lr: bool = True
    z_lr_safety: float = 0.9
    # "explained" exit: nothing passes the threshold, there is nothing to transmit.
    settle_tol: float = 1e-6
    # "stalled" exit, absolute criterion: another iteration costs energy, so
    # it only pays off if it reduces the error energy by at least this much.
    # Trades joules for joules, it is the physical reading, and it is what
    # makes compute grow with surprise (a hard frame still has much to gain
    # per iteration).
    settle_min_gain: float = 0.003
    # Alternative relative criterion (0 = off). Careful: for a frame whose
    # residual is irreducible, the relative improvement collapses immediately
    # and the network gives up, it saves energy, but spends *less* where there
    # is *more* surprise, which is the opposite of what the architecture
    # promises.
    settle_rel_tol: float = 0.0
    # Saturation, the equivalent of int8 saturating arithmetic. Without this,
    # a very surprising frame makes the state blow up and the surprise *rises*.
    z_clip: float = 0.5  # ceiling on |Δz| per iteration (0 = off)
    z_max: float = 2.0  # ceiling on |z| in the latent levels (0 = off)

    # --- code sparsity (pattern separation) --------------------------------
    # Fraction of latent units that stay active on each frame (0 = all).
    # In cortex it is around 1-5%; in the dentate gyrus even less, and on
    # purpose: similar inputs are forced into sparse, nearly disjoint codes
    # so they do not erase one another. It is the brain's main defense
    # against catastrophic forgetting, and our network did not have it, all
    # units took part in every frame, so all weights were pulled by every
    # task.
    # Implemented by competition: the k largest stay, the others go to zero.
    # It is lateral inhibition, and comes for free in ADC conversions.
    state_sparsity: float = 0.0

    # --- error sparsity ----------------------------------------------------
    # |ε| < theta  ->  the error is not transmitted (silence = zero cost).
    theta: float = 0.02

    # --- local learning ----------------------------------------------------
    w_lr: float = 0.1  # ΔW ∝ ε · zᵀ
    a_lr: float = 0.1  # ΔA ∝ ε_topo · z_prevᵀ  (temporal transition)
    # --- metaplasticity: synapses that harden -----------------------------
    # Each weight stores the running mean square of its own local gradient,
    # its "importance", and its effective rate becomes lr/(1+λ·imp). A
    # synapse that has already proven to matter resists being rewritten, and
    # the next task is forced to use different synapses: union instead of
    # compromise.
    #
    # It is the analogue of dendritic spine enlargement and of Fusi's cascade
    # models. And note what is special about this: in ML, protecting
    # important weights (EWC) requires the Fisher information, which requires
    # backprop. Here the local gradient is already at the synapse, the
    # importance comes for free, and it is something a backprop network
    # cannot do without extra machinery.
    metaplasticity: float = 0.0
    meta_decay: float = 0.999  # forgetting of the accumulated importance

    # Plasticity opened by surprise (acetylcholine/noradrenaline): there is no
    # learning when the world is predictable. Protects the past and saves energy.
    plasticity_gate: float = 0.0  # 0 = always learning

    weight_decay: float = 0.0
    grad_clip: float = 1.0  # ceiling on |ΔW| per step, stability

    # --- temporal state ----------------------------------------------------
    # The top keeps memory across frames: ẑ_L(t) = tanh(A · z_L(t-1)).
    # It is the "fast path" in minimal form: a linear recurrence.
    use_transition: bool = True
    # Number of dynamics at the top (see mixture.py). 1 = a single matrix,
    # which is the previous behavior. >1 gives the network alternative regimes
    # and a context selector, without it, tasks with different dynamics pull
    # the single matrix toward an average that serves none, and growing the
    # network does not fix it (measured: doubling the parameters changes nothing).
    # Transition with multiplicative gates (see gated.py): the top's dynamics
    # becomes chosen by the error, unit by unit, as a function of the
    # context, the mechanism that the decomposition of the wall pointed to as
    # the largest part of the distance to backprop.
    gated_transition: bool = False
    # Eligibility traces in the gated transition (e-prop / synaptic
    # tagging): credit through time with per-synapse memory, without BPTT.
    eligibility: bool = False
    # 1-bit critic (front 4): the local update becomes provisional; if the
    # open-loop surprise of the NEXT frame exceeds this factor times the
    # running mean, it is retracted, the equivalent of dopamine saying "it
    # did not help".
    # 0 = off. It is the remedy for the defect measured in the thalamic relay:
    # the local rule does not know when its own updates get in their own way.
    critic_retract: float = 0.0
    # The thalamic relay: the transition's candidate and gate also see the
    # previous frame, in parallel with the recurrent state (see gated.py).
    thalamic: bool = False
    # Gates also in the generator layers: ẑ_l = g⊙f(W·z), g = σ(G·z + b).
    # Credit stays local, ΔW gains the factor g, ΔG uses ε⊙f(Wz)⊙g(1-g),
    # and the ascending path gains the gate term. It is the spatial analogue
    # of the temporal gate: deciding per unit *which part* of the prediction
    # applies in this context.
    gated_layers: bool = False
    n_dynamics: int = 1
    dynamics_tau: float = 0.25  # selector sharpness (low = more exclusive)
    proto_lr: float = 0.02
    state_leak: float = 0.0  # leak applied to z_L(t-1) before the transition

    # --- hierarchy of timescales (see temporal.py) -------------------------
    # "none" reproduces step 1: only the top has memory, the intermediate
    # levels are instantaneous. "diagonal" gives each unit its own time
    # constant (n parameters per level, the cheap and literally biological
    # version). "dense" lets each level mix with itself (n² per level).
    level_transition: str = "none"
    # --- fast path (section 4 of CONTEXTO) --------------------------------
    # A short, always-on path that predicts the next frame directly from the
    # previous one, without going through the hierarchy. The hierarchy no
    # longer has to explain the signal and only explains what the fast path
    # misses:
    #
    #     ẑ_0 = A_0·z_0(t-1)  +  f(W_0·z_1)
    #           └─ fast path ┘  └ hierarchy ┘
    #
    # The decomposition is additive on purpose: the gradients of everything
    # else stay exactly the same. But each path learns from *its own* error,
    # not from the same one, training both on the same residual makes them
    # fight over the same signal with nothing to assign credit (measured:
    # ETTm1 0.230 -> 0.724).
    # It is the "shallow brain hypothesis" in its poorest form, a short
    # cortico-subcortical path in parallel with the deep one.
    fast_path: bool = False
    # Fraction of the error the fast path corrects per frame (normalized
    # step, see network.py). Dimensionless: 1.0 would correct the whole error
    # at once.
    fast_path_lr: float = 0.1
    # λ of the lowest latent level, and how much it slows per step. ratio=2:
    # each level integrates over twice the time of the one below.
    timescale_base: float = 1.0
    timescale_ratio: float = 2.0
    b_lr: float = 0.05  # learning rate of the per-level transitions

    # --- precision (see precision.py) --------------------------------------
    # π per level: the gain with which each error counts. With this on, the
    # threshold θ is applied to the normalized error √π·ε, and the same
    # constant means the same thing at the sensory level and at the top.
    use_precision: bool = False
    precision_per_unit: bool = False
    precision_lr: float = 0.01

    seed: int = 0
    init_scale: float = 1.0  # multiplier on the init ~ 1/sqrt(fan_in)

    def __post_init__(self) -> None:
        if len(self.sizes) < 2:
            raise ValueError("é preciso pelo menos um nível sensorial e um latente")
        if any(n <= 0 for n in self.sizes):
            raise ValueError("todos os tamanhos têm de ser positivos")
        if not 0.0 < self.z_lr <= 1.0:
            raise ValueError("z_lr tem de estar em ]0, 1]")
        if self.z_clip < 0.0 or self.z_max < 0.0:
            raise ValueError("z_clip e z_max não podem ser negativos")
        if self.theta < 0.0:
            raise ValueError("theta não pode ser negativo")
        if self.max_iters < 0:
            raise ValueError("max_iters não pode ser negativo")
        if self.level_transition not in ("none", "diagonal", "dense"):
            raise ValueError(f"level_transition desconhecida: {self.level_transition!r}")
        if self.timescale_ratio < 1.0:
            raise ValueError("timescale_ratio < 1 poria o topo mais rápido que a base")

    # -- utilities ----------------------------------------------------------
    @classmethod
    def recommended(cls, **overrides) -> "PCConfig":
        """The best configuration measured so far.

        Fast path + precision: better accuracy and 5-10x fewer ADC
        conversions than the defaults, on three different datasets (see README).

        Why these are not the defaults: the defaults reproduce exactly steps
        1 and 2 of the plan, and there are published artifacts against them,
        `runs/phase0/model.h`, the C reference vectors, and every number in
        the README. Changing the defaults would invalidate that record
        without adding anything this method does not already give. It is a
        decision to make once a demonstrator has been chosen, not before.
        """
        base = dict(fast_path=True, fast_path_lr=0.05, use_precision=True)
        return cls(**{**base, **overrides})

    @property
    def n_levels(self) -> int:
        return len(self.sizes)

    @property
    def n_weights(self) -> int:
        """Number of generator matrices W_l (one per pair of levels)."""
        return len(self.sizes) - 1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PCConfig":
        d = dict(d)
        d["sizes"] = tuple(d["sizes"])
        return cls(**d)
