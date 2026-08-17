#!/usr/bin/env python3
"""DEPTH BY STACKING SHALLOW BRICKS (two-stroke rule).

The brick: a shallow 64->24 network with a gated transition at the top,
trained by the two-stroke rule - settling infers the state, credit is taken
from the backprojected open-loop error (h = W0^T e), with the exact chain
through the gate.
Milestone to reproduce: 0.579 +- 0.004 (3 worlds, interleaved 16, 80 ep, lr 0.1).

The stack: module 2 lives on top; its sensory input is module 1's temporal
RESIDUAL (r1 = s1_settled - prior1_raw) - what module 1's transition
could not predict from its own state - and its prediction descends as an
ADDITIVE correction to module 1's prior. Each module only uses its own local
quantities (its error, its state, its transpose). No gradient across
modules: module 1 treats the correction as an external constant; module 2
treats r1 as data.

Alternative variant (c): module 2's sensory input is the FULL STATE s1 and
its prediction still descends as an additive correction to the prior - the
two predictors end up contending for the same target (the fast-path lesson
says this ends badly; it gets measured anyway).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from continual import make_tasks, evaluate_task  # noqa: E402
from pcnet import PCConfig, PCNetwork  # noqa: E402
from pcnet.dtypes import F  # noqa: E402

TASKS = make_tasks(3, 64, 400, seed=1)
LR = 0.1
BASE = dict(fast_path=False, use_precision=True, a_lr=0.0, gated_transition=True)


def mix(block: int = 16) -> np.ndarray:
    n_blocks = min(len(t[0]) for t in TASKS) // block
    return np.concatenate(
        [tr[b * block:(b + 1) * block] for b in range(n_blocks) for tr, _ in TASKS]
    )


MIXED = mix(16)


def _sigm(x):
    return 1.0 / (1.0 + np.exp(-x))


def two_time_update(net: PCNetwork, target: np.ndarray, s_prev: np.ndarray,
                    prior_extra: np.ndarray | None = None,
                    lr: float = LR) -> np.ndarray:
    """The two-stroke rule on one brick. Returns the raw prior.

    s_prev: top state captured BEFORE net.step (the settling only
    infers). prior_extra: additive correction coming from above - constant
    for this module (no gradient passes through it).
    """
    gt = net.gated
    W0 = net.layers[0].W
    c = np.tanh(gt.A.dot(s_prev))
    g = _sigm(gt.G.dot(s_prev) + gt.b)
    prior_raw = ((1.0 - g) * s_prev + g * c).astype(F)
    prior = prior_raw if prior_extra is None else (prior_raw + prior_extra).astype(F)

    e = (target - W0.dot(prior)).astype(F)
    h = W0.T.dot(e)

    W0 += (lr / (float(prior @ prior) + 1e-6)) * np.outer(e, prior).astype(F)
    # canonical (validated against the 0.579+-0.004 milestone): re-estimate
    # sigma_max after touching W0, so the settling's adaptive step keeps up
    # with the growth of the weights. Without this: 0.569+-0.011.
    net.layers[0].refresh_device()
    ns = float(s_prev @ s_prev) + 1e-6
    mod_g = (h * (c - s_prev) * g * (1.0 - g)).astype(F)
    gt.A += (lr / ns) * np.outer((h * g * (1.0 - c * c)).astype(F), s_prev)
    gt.G += (lr / ns) * np.outer(mod_g, s_prev)
    gt.b += F(lr) * mod_g
    return prior_raw


# ---------------------------------------------------------------------------
# (a) single brick
# ---------------------------------------------------------------------------
def run_brick(seed: int, epochs: int) -> tuple[float, dict]:
    net = PCNetwork(PCConfig(seed=seed, sizes=(64, 24), **BASE))
    L = net.L
    for _ in range(epochs):
        for fr in MIXED:
            s_prev = net._z_prev[L].copy()
            net.step(fr, learn=False)
            two_time_update(net, np.asarray(fr, dtype=F), s_prev)
    score = float(np.mean([evaluate_task(net, te, tr) for tr, te in TASKS]))
    return score, {}


# ---------------------------------------------------------------------------
# (b)/(c) stack of 2 bricks
# ---------------------------------------------------------------------------
class Stack2:
    """Two shallow bricks, each trained by ITS OWN two-stroke rule.

    interface="residual": module 2's sensory input = s1 - prior1_raw.
    interface="estado":   module 2's sensory input = s1 (contested target).
    In both, module 2's prediction descends as an additive correction to
    module 1's prior (injected into the settling via _temporal_prior).
    Implements the PCNetwork interface that evaluate_task uses.
    """

    def __init__(self, seed: int, interface: str, n2: int = 24):
        self.net1 = PCNetwork(PCConfig(seed=seed, sizes=(64, 24), **BASE))
        self.net2 = PCNetwork(PCConfig(seed=seed + 50, sizes=(24, n2), **BASE))
        self.interface = interface
        self.corr = np.zeros(24, dtype=F)
        # counters: the norm of module 2's contribution vs that of the raw prior
        self.sum_corr = 0.0
        self.sum_prior = 0.0
        self.n_counted = 0
        corr = self.corr
        orig = PCNetwork._temporal_prior

        def patched(nself):
            p = orig(nself)
            p += corr  # writes into prior_top, in-place
            return p

        self.net1._temporal_prior = types.MethodType(patched, self.net1)

    # -- interface for evaluate_task -------------------------------------
    def snapshot_state(self):
        return (self.net1.snapshot_state(), self.net2.snapshot_state(),
                self.corr.copy())

    def restore_state(self, snap):
        self.net1.restore_state(snap[0])
        self.net2.restore_state(snap[1])
        self.corr[:] = snap[2]

    def reset(self):
        self.net1.reset()
        self.net2.reset()
        self.corr.fill(0.0)

    def _set_corr(self):
        # module 2's open-loop prediction = W1 @ prior2 (identity,
        # no fast path, no memory) - only module 2 quantities.
        self.corr[:] = self.net2.predict_next()

    def _prior_raw(self, s1_prev):
        gt = self.net1.gated
        c = np.tanh(gt.A.dot(s1_prev))
        g = _sigm(gt.G.dot(s1_prev) + gt.b)
        return ((1.0 - g) * s1_prev + g * c).astype(F)

    def predict_next(self):
        self._set_corr()
        return self.net1.predict_next()

    def step(self, fr, learn: bool = True, **_kw):
        fr = np.asarray(fr, dtype=F)
        s1_prev = self.net1._z_prev[self.net1.L].copy()
        s2_prev = self.net2._z_prev[self.net2.L].copy()
        self._set_corr()
        corr_now = self.corr.copy()

        trace = self.net1.step(fr, learn=False)  # module 1 settling
        s1 = self.net1.z[self.net1.L].copy()

        if learn:
            prior_raw = two_time_update(self.net1, fr, s1_prev,
                                        prior_extra=corr_now)
        else:
            prior_raw = self._prior_raw(s1_prev)

        target2 = (s1 - prior_raw) if self.interface == "residual" else s1
        target2 = target2.astype(F)
        self.net2.step(target2, learn=False)  # module 2 settling
        if learn:
            two_time_update(self.net2, target2, s2_prev)

        self.sum_corr += float(np.linalg.norm(corr_now))
        self.sum_prior += float(np.linalg.norm(prior_raw)) + 1e-12
        self.n_counted += 1
        return trace

    def counters(self) -> dict:
        n = max(self.n_counted, 1)
        return {"norm_corr": self.sum_corr / n,
                "norm_prior": self.sum_prior / n,
                "racio": self.sum_corr / max(self.sum_prior, 1e-12)}


def run_stack(seed: int, epochs: int, interface: str) -> tuple[float, dict]:
    st = Stack2(seed, interface)
    for _ in range(epochs):
        for fr in MIXED:
            st.step(fr, learn=True)
    train_counters = st.counters()
    st.sum_corr = st.sum_prior = 0.0
    st.n_counted = 0
    score = float(np.mean([evaluate_task(st, te, tr) for tr, te in TASKS]))
    eval_counters = st.counters()
    return score, {"treino": train_counters, "aval": eval_counters}


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["brick", "residual", "estado"],
                    required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    runners = {
        "brick": lambda s: run_brick(s, args.epochs),
        "residual": lambda s: run_stack(s, args.epochs, "residual"),
        "estado": lambda s: run_stack(s, args.epochs, "estado"),
    }
    fn = runners[args.arm]
    scores, extras = [], []
    for s in args.seeds:
        t0 = time.time()
        sc, extra = fn(s)
        scores.append(sc)
        extras.append(extra)
        print(f"[{args.arm}] seed {s}: {sc:.4f}  ({time.time()-t0:.0f}s)  "
              f"{json.dumps(extra, default=float) if extra else ''}", flush=True)
    arr = np.array(scores)
    print(f"[{args.arm}] media {arr.mean():.4f} +- {arr.std():.4f}  "
          f"(seeds {args.seeds})", flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"arm": args.arm, "seeds": args.seeds, "scores": scores,
             "mean": float(arr.mean()), "std": float(arr.std()),
             "extras": extras}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
