# Sparse hierarchical predictive network - Phase 0

Steps 1 and 2 of the plan in `CONTEXTO.md`: a predictive hierarchy of 3
generative layers (64 → 32 → 16 → 8) predicting the next frame of a signal,
with local learning (no backprop), threshold-based sparsity and early exit;
then the same weights ternarized to {-1,0,1} and programmed onto a crossbar
with defects.

The goal is **not** an ML metric - it is to answer questions with numbers
that can fail:

| # | question | where it is answered |
|---|----------|------------------|
| 1 | does the settling dynamics converge? | `test_settling_decreases_the_energy` |
| 2 | does local learning work? | `test_local_learning_beats_the_persistence_baseline` |
| 3 | does the sparsity threshold pay off? | `test_higher_threshold_silences_more_and_converte_menos` |
| 4 | is compute proportional to surprise? | `test_surprising_frames_cost_more_than_banal_ones` |
| 5 | can the weights survive being ternary? | `test_training_on_the_crossbar_beats_quantising_at_the_end` |
| 6 | does the loop tolerate a defective device? | section 3 of `run_step2.py` |

## Running

```sh
.venv/bin/python -m pytest            # 115 tests, ~60 s
.venv/bin/python scripts/run_phase0.py    # step 1: ~60 s
.venv/bin/python scripts/run_step2.py     # step 2: ~8 min
.venv/bin/python scripts/benchmark.py --data ETTm1.csv --column OT
.venv/bin/python scripts/ablate.py --har "UCI HAR Dataset"
.venv/bin/python scripts/demo.py --events 6   # watch it work in the terminal
```

The package depends only on NumPy. The `.venv` also brings torch, and that is
only so the baselines in `benchmark.py` are the real ones instead of versions
hand-written by someone with a stake in the result.

`scripts/run_phase0.py` writes `runs/phase0/` with `model.npz`, `model.h`,
`golden.h` and `results.json`; `run_step2.py` writes `runs/step2/results.json`.

## Step 1 - results (60 passes, defaults, seed 0)

```
NRMSE for next-frame prediction
  repeat the previous frame (baseline)     : 1.229
  network before training                  : 0.993
  network after training                   : 0.219

Settling: 120/120 frames with monotonically decreasing energy,
          final energy = 6% of the initial.

Sparsity threshold θ:
    θ  NRMSE  silenced_%  ADC_%  MACs_up_%  iters
  0.00  0.218         0.0   22.4       94.1   4.83
  0.02  0.219        73.9    5.9       35.4   4.84
  0.05  0.307        85.5    2.3       24.1   3.09
  0.10  0.417        89.1    1.3       21.0   2.19

Compute vs. surprise (12 transients in 200 frames):
  banal            surprise 0.12   5.4 iters   207 ADC conversions
  with transient   surprise 1.53   7.4 iters   686 ADC conversions
  -> a surprising frame costs 3.3× a banal one
```

The number that matters is the **ADC_%** column: conversions actually
performed over those a dense network would do running every time to the
iteration ceiling. At θ = 0.02 that is 5.9% - and the prediction error does
not get worse (0.219 vs. 0.218). Silencing what was already predicted costs
no accuracy; it only costs the accuracy that was being bought with noise.

Caveat: none of this is an energy measurement. It is a count of operations
and conversions on a machine that is not the right machine. The energy
argument is only demonstrated on the crossbar (step 4 of the plan); see
section 6 of `CONTEXTO.md`.

## Step 2 - the substrate (3 seeds, mean ± deviation)

Nothing here is reported without dispersion: an analog device is a sample
from a distribution, and measuring just one would be measuring luck.

```
Quantization                                    NRMSE      vs float
  float32                                0.281 ± 0.100         -
  ternary, quantized at the end (PTQ)    0.658 ± 0.062     2.34×
  ternary, trained on the crossbar (QAT) 0.314 ± 0.006     1.12×
  (43% of the ternary weights end up at zero: devices that
   need neither programming nor reading)

Programming variability             programmed    trained on
                                   only at end    the device
  σ_rel = 0.10                      0.336 ± 0.005  0.331 ± 0.015
  σ_rel = 0.20                      0.674 ± 0.419  0.321 ± 0.012
  σ_rel = 0.40                      1.835 ± 1.081  0.341 ± 0.012

ADC (σ_rel = 0.1)   8 bits 0.337   6 bits 0.522   4 bits 0.770
                                                  3 bits 1.000

Threshold θ in ternary (σ_rel = 0.1)
    θ = 0.00   NRMSE 0.408    0% silenced   ADC 47.6%
    θ = 0.02   NRMSE 0.336   58% silenced   ADC 14.0%
```

Training with the quantizer inside the loop costs almost nothing in code: the
rule is local, there is no gradient to push through it. The float weight is
the shadow weight, the ternary one is what sits on the crossbar. It is the
only part of the design where the absence of backprop gives an *engineering*
advantage, and not just an energy one.

### The loop is self-correcting - and there is an exact limit to that

| σ_rel | open-loop energy | after settling | closed by the loop | NRMSE |
|-------|------------------------|--------------------|--------------------|-------|
| 0.0   | 0.38 | 0.022 | 94% | 0.314 ± 0.006 |
| 0.1   | 0.43 | 0.029 | 93% | 0.336 ± 0.004 |
| 0.2   | 2.39 | 0.182 | 92% | 0.674 ± 0.420 |
| 0.4   | 16.47 | 0.918 | 94% | 1.835 ± 1.081 |

With a crossbar 40% out of spec the open-loop energy blows up 43× - and
settling still closes 94% of it, with the states always bounded. **The loop
corrects even the device's defects.** What it does not correct is the
*prediction* of the next frame, and it cannot: that prediction descends
through the same broken weights before any error exists to guide it.
Self-correction is not omniscience - and that is exactly why training has to
happen on the device, as section 6 of `CONTEXTO.md` anticipated. Training on
the crossbar recovers 98% of the degradation at σ_rel = 0.2 and 0.4.

### There is a cliff, and it is per device

Without retraining, sweeping σ_rel finely over three crossbars:

```
σ_rel   0.00  0.08  0.12  0.14  0.16  0.18  0.20  0.25  0.30
  #0    0.31  0.33  0.35  0.37  0.37  0.38  0.40  1.27  2.33
  #1    0.31  0.33  0.35  0.42  0.58  0.81  1.27  1.97  2.16
  #2    0.32  0.33  0.33  0.34  0.35  0.35  0.36  0.37  0.40
```

Smooth degradation up to ~15%, and then *some* crossbars fall off a cliff
while others remain fine. The prediction that falls becomes **decorrelated**
from the target, not merely badly scaled - rescaling it by the optimal gain
recovers nothing (NRMSE 0.97, as good as predicting zero) - and the states
remain bounded. It is neither saturation nor numerical divergence: it is a
bifurcation of the closed-loop dynamics. Training on the device eliminates
the cliff.

Practical consequence: **for this architecture you cannot train-and-program.**
Either you train on the device, or you calibrate on it.

## Comparison with real problems

`scripts/benchmark.py` puts the predictive network against what actually
competes in this niche - persistence, linear AR, MLP, GRU and a small causal
transformer - all with a comparable parameter budget and all under the same
streaming protocol: at instant *t* the model sees everything up to *t* and
has to say frame *t+1*. Nobody sees the future, and normalization uses only
statistics from the training segment.

**Not against an LLM.** An LLM does not solve this problem and this model
does not solve the LLM's; section 1 of `CONTEXTO.md` explicitly decided not
to compete with transformers on language. Comparing 2752 parameters
predicting a sensor with a language model is not a hard comparison, it is a
comparison with no common axis.

### ETTm1, transformer oil temperature (real data, 15 min)

The standard benchmark of transformer forecasting papers. 1088 frames of
64 samples, 871 train / 217 test.

| model | NRMSE | parameters | MACs/frame |
|--------|-------|-----------|-----------|
| **predictive network + fast path** | **0.212** | 6848 | - |
| linear AR (order 4) | 0.211 | 16448 | 16 448 |
| causal transformer (d=12, ctx=16) | 0.225 | 2896 | 2 928 |
| predictive network (step 1) | 0.235 | 2752 | 42 015 |
| MLP | 0.243 | 2953 | 2 880 |
| GRU | 0.246 | 1449 | 1 355 |
| persistence | 0.285 | 0 | 0 |

Honest reading: **in the step 1 version** the network lands mid-table - it
beats the MLP, the GRU and persistence, loses to the transformer and to a
linear regression of order 4 with six times more parameters. With the fast
path (next section) it moves to a tie with the best, using 2.4× fewer
parameters than the AR.

None of these differences is large. The reading that matters is not the
order of the table - it is that an architecture designed for other
constraints, trained without backprop, sits at the level of what exists. It
is what the strategy of section 1 of `CONTEXTO.md` needs to be true, and it
is the minimum it needed.

### The toy signal was too easy

The same test bench applied to the step 1 signal revealed something I should
have checked at the start: **a linear AR of order 4 gets 0.040 against our
0.118.** A sum of three sinusoids is, by construction, a linear system - and
validating a hierarchical non-linear architecture on a problem that a linear
regression solves better does not validate much. The step 1 numbers remain
correct; what changes is the weight they deserve. That is why the comparison
now runs on real data.

### HAR, gyroscope at 50 Hz: when nobody can

A control worth running before concluding anything at all about the model.
Frames of 64 samples at 50 Hz are 1.28 s of human movement, and at that
horizon the signal is practically unpredictable:

| model | NRMSE | parameters |
|--------|-------|-----------|
| **predictive network** | **0.986** | **2752** |
| causal transformer | 1.014 | 2896 |
| GRU | 1.044 | 1449 |
| MLP | 1.059 | 2953 |
| persistence | 1.179 | 0 |
| linear AR (order 4) | 1.983 | 16448 |

Everyone lands at ~1.0, which is the same as saying "I predict zero". It is
the task that has no signal at that horizon - and the linear AR, with 16 448
parameters for 327 training frames, ends up *worse* than predicting zero,
which is what overfitting does.

Being the only one below 1.0 on a problem where nobody can is a modest
victory, and that is how it should be read. What this control is really
worth is as an antidote: without it, the 0.986 would read as "our model
failed here" when what failed was the framing of the problem.

### What improved the model, and what did not

Three neuroscience ideas were implemented and measured in isolation
(`scripts/ablate.py`). Two did not pay off. One paid off a lot.

**Hierarchical timescales** (`temporal.py`) - Hasson measured temporal
receptive windows that grow along the cortex; Kiebel, Daunizeau & Friston
show this falls out of a predictive model where each level has its own,
progressively slower, dynamics.

The result **depends on the problem**, and that is what makes it
interesting. On the toy signal and on ETTm1 it does not help at all -
because those signals do not *have* a hierarchy of timescales to model. On
the HAR accelerometer it helps (0.594 → 0.559), because a sensor strapped to
a person really does have structure at several scales: the stride, the
gesture, the activity change.

The lesson is not about the idea, it is about the method: I spent a good
while concluding the idea was useless, when what was useless was the problem
I was measuring it on. It remains switchable (`level_transition`), off by
default until there is a demonstrator whose structure justifies it.

**Precision as attention** (`precision.py`) - π = inverse of the expected
variance of the error, which Friston identifies with attention and with
post-synaptic gain modulation. On its own, it is a tie in accuracy. **But it
pays off when there is a fast path**, and a lot: it cuts ADC conversions by
5-8× without losing accuracy. It makes sense - it is only worth estimating a
channel's reliability when there is more than one channel.

**Fast path** (section 4 of `CONTEXTO.md`, the missing piece) - a short,
always-on path that predicts the next frame directly from the previous one,
in parallel with the deep hierarchy, in the manner of the cortico-subcortical
pathways of the *shallow brain hypothesis*:

```
ẑ_0 = A_0·x(t-1)  +  f(W_0·z_1)
      └ fast path ┘   └ hierarchy ┘
```

The decomposition is additive on purpose - the gradients of everything else
stay the same - but **each path learns from its own error**, not from the
same one. The hierarchy no longer has to explain the signal and starts
explaining the residual.

| dataset | variant | NRMSE | ADC/frame | iterations |
|----------|----------|-------|-----------|-----------|
| toy¹ | step 1 | 0.186 ± 0.005 | 567 | - |
| toy¹ | fast path + precision | **0.074** | **99** | - |
| ETTm1¹ | step 1 | 0.230 ± 0.005 | 583 | - |
| ETTm1¹ | fast path (μ=0.05) | **0.212** | 913 | - |
| ETTm1¹ | fast path + precision | 0.218 | **112** | - |
| HAR acc_x² | step 1 | 0.594 ± 0.012 | 1 370 | 14.9 |
| HAR acc_x² | fast path | **0.532 ± 0.007** | 1 500 | 15.5 |
| HAR acc_x² | fast path + precision | 0.539 ± 0.000 | **141** | **1.2** |

¹ 2 seeds  ² 3 seeds. The full ETTm1 ablation with 3 seeds was left
unrun; it reproduces with
`scripts/ablate.py --data ETTm1.csv --column OT --seeds 3`.

Better accuracy *and* 5-10× fewer conversions, on three different datasets.
With the fast path the network matches the best of all the baselines on
ETTm1 (order-4 AR, 0.211) with 2.4× fewer parameters.

The most revealing number is the iteration count on HAR: **1.2**. The fast
path explains almost everything, precision correctly discounts what is left,
and the expensive hierarchy only wakes up when it is really needed. It is
exactly the mechanism section 4 of `CONTEXTO.md` describes - the always-on
coarse path that only calls the slow circuit when it is not enough - and it
appeared without being programmed as such: it falls out of the existing
stopping criterion as soon as there is a fast path lowering the initial
surprise.

**It failed twice before working**, and the two failures teach more than the
success. First, training the two paths with the same residual: they chase
the same signal with nothing assigning credit, and fight over it (ETTm1
0.230 → 0.724). Then, a learning rate above the stability limit
(0.208 → 0.899).

### Choosing the problem is half the work

Something you only see after running the test bench in several places:
**the problems within our reach are either too linear or impossible.**

- On the toy signal, an order-4 AR gets 0.040 against our 0.118.
- On ETTm1, an AR of **order 1** - a single 64×64 matrix fitted by least
  squares - gets 0.216, better than our 0.230. The task is dominated by
  linear structure from one frame to the next.
- On the HAR gyroscope, nobody gets below 1.0 (we sit at 0.986).

A predictive hierarchy has nothing to gain on a linear problem or on a
problem with no signal. This is not an excuse, it is a design constraint:
the step 4 demonstrator has to be chosen with this care, and the claim to
defend is not "we are more accurate than a transformer" - it is the one
section 1 of `CONTEXTO.md` had already chosen: **match the accuracy of what
exists, at a fraction of the conversions, in a niche where the transformer
does not even enter.** On that axis the numbers exist and are above. On the
pure-accuracy axis, the honest answer right now is "equivalent, not
superior".

### The three cost axes, and what each one hides

A single column lies. In **digital MACs** the predictive network loses by a
large margin - settling iteratively makes several passes where the others
make one, and 42 015 against the MLP's 2 880 is not a difference you can
argue away. In **ADC conversions** - what you pay on an analog crossbar,
where the multiplications are done by physics - the account flips, because
only the error above the threshold is converted.

It is that second column that carries the whole thesis. If the architecture
does not win there, it wins nowhere, because in digital operations it has
already lost. And measuring that seriously requires the crossbar, not the
Mac.

## Continual learning: episodic memory and consolidation

`src/pcnet/episodic.py` implements the missing piece of section 4 of
`CONTEXTO.md` - the hippocampus and sleep. A fixed-size key-value store,
addressed by the top state (the network's summary of the context), which
records when there is surprise and is replayed offline to distill into the
weights.

`scripts/continual.py` measures what that should buy: N tasks learned
**in sequence**, never going back, and at the end everything is measured on
all of them.

```
3 tasks, 25 passes each, sequential training

                              final NRMSE   forgetting
CEILING: joint training             0.823        0.000
predictive network (no memory)      0.950        1.069
+ episodic memory                   0.950        1.067
+ memory + consolidation            0.891        0.519
+ memory + interleaved replay       0.983        0.463
MLP (backprop)                      1.007        1.153
GRU (backprop)                      1.040        1.398
```

**Two readings, and the second cancels the first.**

The good one: consolidation **cuts forgetting in half** (1.069 → 0.52), and
the backprop-trained networks forget more than we do (1.15 and 1.40).
Episodic memory alone does nothing - it is the *replay* that counts, not the
store.

The one that matters: **the ceiling is 0.823.** Training on the three tasks
at the same time, under ideal, interleaved conditions, gives almost the same
as training in sequence and forgetting everything. In other words: **the
architecture cannot retain three tasks even when it sees them all
together.** What the table measures is not forgetting - it is lack of
capacity. You do not measure forgetting on tasks the model never retained.

It was the third time in this project that I measured a good idea on the
wrong problem (after the toy signal and the HAR gyroscope). This time the
control caught it before it became a conclusion. **The first version of the
control was wrong** - it concatenated the tasks instead of interleaving
them, so it was sequential training under another name, and it gave a
"ceiling" lower than the variants it was supposed to bound. A badly built
control is worse than none, because it looks like rigor.

### Why the ceiling does not rise

**Contamination warning, and the audit.** An entire first round of
hypotheses (capacity, mixture of dynamics, warm-up, oracle, sparsity,
expansion) was run with the fast path on - and the fast path, being a shared
linear predictor, pins the result at the compromise (~0.82) and masks any
effect of the hierarchy. It was discovered when two metaplasticity strengths
separated by a factor of 100 gave identical results. Everything that follows
was **redone without the fast path**.

Result of the clean rerun (joint training, 3 tasks):

| variant | mean NRMSE |
|----------|------------|
| base (64,32,16,8) | **0.778** |
| metaplasticity λ=10 | 0.833 |
| mixture of 3 dynamics | 0.842 |
| sparsity 10% | 1.001 (= predicting zero) |
| capacity (64,64,48,32) | 1.103 |
| expand+sparsify (dentate gyrus) | 1.048 |
| *linear AR(1), least squares* | *0.741* |
| *GRU with backprop* | ***0.397*** |

The rerun saved no mechanism - on the contrary: without the mask, they are
*actively harmful*, not neutral. And the simple base is still worse than a
single linear matrix.

### What not using backprop costs: 2×

The missing test was not another architecture knob - it was changing the
*learning rule*. The same three tasks, the same interleaved joint training,
a model of comparable capacity, trained with backprop:

| method | parameters | mean NRMSE (3 tasks) |
|--------|-----------|------------------------|
| GRU, backprop | 5024 | **0.397** |
| GRU, backprop | 2352 | 0.509 |
| **linear AR(1), least squares** | 4096 | **0.741** |
| predictive network, local rule | 6848 | 0.778 - 0.826 |

*(on a single task, the predictive network sits at 0.07 and the GRU at ~0.09)*

And this is where it gets serious: **our network is worse than a single
linear matrix.** Every variant I tested - capacity, mixture of dynamics,
sparsity, expansion, oracle, more iterations - lands between 0.78 and 0.83,
around the linear compromise. It is not that the hierarchy helps little: it
does not help at all. The whole result was being pinned by a linear part of
the model, and my architectural experiments were measuring noise on top of
that.

Two conclusions, and the first corrects an earlier claim of this README that
was too strong.

**It is not the architecture, it is the rule.** With backprop, comparable
capacity does 0.397. The representation exists; what fails is finding it
without global credit assignment.

**And the brain did not save us.** I tried, in this order, the biological
answers to catastrophic forgetting: complementary memory systems (fast
hippocampus + slow cortex), replay during sleep, interleaved replay, pattern
separation by sparse codes, expansion before sparsifying (dentate gyrus, the
fly's mushroom body), and context-switched dynamics. **None of them moved
the number.** Only replay moved anything - and it moved forgetting, not the
ceiling.

The honest reading is that these mechanisms presuppose that the underlying
learning can form distinct representations when given the room to. The local
rule does not: it converges to the compromise, and no amount of room,
sparsity or memory makes it change its mind.

This is the credit assignment wall, measured on our own system instead of
cited. It is not fatal for *one* task - there the network is good and
remains what the project needs. It is fatal for the ambition of accumulating
several worlds, and I will not disguise it.

## The attack on the open problem: local credit vs backprop

(The resulting method was christened **Two-Stroke Learning (TSL)**, learning
in two strokes; the cell/product is the **TS-1**.)

After ten system mechanisms failed, the investigation changed level: no
longer "which modules to put around the rule", but "what the rule is
missing". Two results structure everything that follows.

### The decomposition of the wall

The same GRU, the same 3-dynamics task, varying only how far the gradient
can travel in time:

| temporal credit horizon | NRMSE |
|---|---|
| 0 steps (credit only within the instant) | 0.576 |
| 4 steps | 0.501 |
| full BPTT | 0.490 |
| our local rule | 0.781 |

**Most of the wall is inside a single instant, not in time.** And the GRU's
only structural difference in that regime is its multiplicative gates
- with the credit flowing through them. Our modularity attempts failed for
exactly that reason: mixture chosen by similarity and sparsity by magnitude
are blind to the error; the GRU's gate is chosen *by* the error.

### What moved the ceiling (and what did not)

`gated.py`: the top transition becomes ẑ = (1-g)⊙z + g⊙tanh(A·z), with
g = σ(G·z+b) learned by a three-factor rule (pre × post × gate),
all local, NLMS step. Multiplicative gating is among the best documented
mechanisms in neurophysiology - shunting, dendritic gating, thalamus.

| step | NRMSE (3 tasks, joint) |
|---|---|
| base, local rule | 0.781 ± 0.003 |
| linear matrix (milestone) | 0.741 |
| + temporal gate at the top (16) | 0.702 ± 0.016 |
| **+ shallow brain: (64,24), 2 levels** | **0.659 ± 0.046** (6 seeds) |
| GRU without temporal credit (phase target) | 0.575 ± 0.019 (3 seeds) |
| full backprop (horizon) | 0.490 |

The two critical milestones were re-measured with proper seeds: the champion
was slightly inflated by seed luck (0.642 with 2 -> 0.659 with 6) and the
target was confirmed. The remaining gap (0.084, ~2σ) is real: the spatial
phase closed 59% and is not finished.

The gate was the first mechanism out of eleven to lower the ceiling.
Shortening the hierarchy - the *shallow brain hypothesis*, which has been in
section 2 of `CONTEXTO.md` since day one - lowered it again, with **fewer
parameters** (2712 against 3344) and the tightest dispersion of the whole
session. With no measured cost on the single task.

Eliminated with numbers, under the same protocol: gates on the generative
layers (0.804 alone, 0.904 combined - they interfere with settling), and
neighboring widths ((64,16) too narrow, (64,32) unstable - once again the
rate against the stability limit).

### Results of the next bricks

**Eligibility traces** (synaptic tagging; e-prop): implemented in the gated
cell - the trace decays at the retention rate of the gate itself, one memory
per synapse, all local. Result: **a tie** (0.638 ± 0.028 against
0.641 ± 0.055). Consistent with the decomposition, which gave temporal
credit only ~0.09 even for the GRU. It remains implemented (`eligibility`),
off by default.

**Nightmares (Crick & Mitchison): buried with counted firings.** The idea
- anti-learning on what the network generates spontaneously, to erase the
spurious compromise - was tested blind (0.675 ± 0.077, worse) and targeted
at the mixtures of real contexts with confirmed firings (1868 events:
0.660 ± 0.052, an exact null against 0.659 ± 0.046). Mechanism: Hopfield
unlearning needs discrete attractors; in a continuous predictor the
compromise is the function itself, and anti-learning on the mixtures
perturbs the very weights that serve the true worlds. Method note: the first
targeted version gave results bit-for-bit identical to the control - the
nightmares had never fired (contexts always sampled at the same point of the
cycle). Since then every episodic mechanism reports a firing count.

**The sparsity threshold is a plasticity gate.** Hypothesis tested: training
with the full error (θ=0) and censoring only at inference should recover
fine gradient. **Inverted by the data**: training without the threshold
gives 0.769 against 0.639 with it. The threshold during learning performs
anti-interference protection - a small error means "already well predicted,
do not touch", and in multi-task that stops each task from eroding the
others. Sparse transmission and selective plasticity are the same mechanism,
and it had been in the design since step 1 without us knowing its second
role.

## How it is organized

```
src/pcnet/
  config.py      PCConfig - also the contract with the C side
  layer.py       PCLayer  - predict / modulated_error / backward / learn
  network.py     PCNetwork - the settling loop and the temporal transition
  metrics.py     StepTrace, RunStats - the instrumentation
  signals.py     the toy signal and the persistence baseline
  device.py      ternarization, variability, read noise, ADC
  train.py       training loop, θ sweep
  export.py      npz, model.h, golden.h - the training/inference boundary
  report.py      text tables for the scripts
scripts/run_phase0.py   step 1
scripts/run_step2.py    step 2
tests/
c/               inference in C (step 3) - see c/README.md
```

## The algorithm

Per frame, three phases:

**1. The prediction descends.** The top propagates in time,
`ẑ_L = tanh(A·z_L(t-1))`, and from there it descends level by level,
`ẑ_l = f(W_l·z_{l+1})`, until it generates an expected frame. This happens
before seeing the data: the first `ε_0` is the open-loop prediction error.

**2. The surprise ascends.** `ε_l = z_l − ẑ_l`; whatever has `|ε| < θ` is
not transmitted; the rest corrects the states above,
`Δz_l ∝ W_{l-1}ᵀ(ε̃_{l-1} ⊙ f') − ε̃_l`. This repeats while it is worth it.

**3. Learning is local.** `ΔW_l ∝ (ε_l ⊙ f') ⊗ z_{l+1}` and
`ΔA ∝ ε_L ⊗ z_L(t-1)`. Only quantities present at the synapse: no graph, no
stored activations, no backprop.

### Decisions that are not obvious

**Surprise is measured by the energy `½Σ‖ε_l‖²`, not by the sum of the `|ε|`.**
The energy is the function the settling dynamics descends, so it is the only
one that serves as a convergence criterion. The L1 sum does not work: the
latent levels start with `ε = 0` by construction (they were initialized by
the prediction from above) and grow as they take on their share of the
explanation, which makes the L1 sum oscillate even when the energy decreases
monotonically. It was this detail that, in the first version, made the
*most* surprising frames look *cheaper*.

**Early exit has three reasons, and only one is good.**
`explicado` ("explained": nothing passed the threshold - silence, zero
cost), `estagnado` ("stalled": another iteration does not pay for what it
costs) and `teto` ("ceiling": the budget was spent). The distinction matters
because `estagnado` is the network giving up on something it does not know
how to explain: it saves energy, but it is an admission of ignorance, not a
victory.

**The stopping criterion is absolute, not relative.** `settle_min_gain`
compares the energy reduction with the cost of one more iteration - joules
against joules. With a *relative* criterion, a frame whose residual is
irreducible stalls immediately and the network spends *less* where there is
*more* surprise, which is exactly the opposite of what the architecture
promises. The relative criterion remains available (`settle_rel_tol`), off
by default.

**Every delta rule in this project has the same stability limit, and I
forgot it three times.** A rule `ΔW ∝ ε·xᵀ` is only stable for
`lr < 1/‖x‖²`. It happened with the settling `z_lr` (at 0.5 it diverges), it
happened again when the ternary weights inflated `σ_max`, and it happened a
third time with the fast path - where `lr = 0.02` was above the limit
`1/‖x‖² ≈ 0.016` and turned a 0.208 model into a 0.899 one. The definitive
fix is not tuning the number, it is **normalizing the step**
(`ΔA ∝ ε·xᵀ/‖x‖²`, which in adaptive filtering is called NLMS): the rate
becomes dimensionless - "what fraction of the error to correct per frame" -
and stops depending on the signal's scale, so it no longer needs to be
re-tuned per dataset.

**`z_lr` has to be small, and the limit is computable.** Settling is
gradient descent with a fixed step and is only stable for
`z_lr < 2/(1 + σ_max(W)²)`; since `σ_max` grows with learning, the safe
value decreases over the course of training. At 0.5 the network diverges. By
default the step is limited by that bound (`adaptive_z_lr`), with `σ_max`
estimated by power iteration - two crossbar reads per weight update. It is
not free: at σ_rel = 0 it costs a little (0.302 → 0.315) and at
σ_rel = 0.4 it saves a lot (0.417 → 0.341). `z_clip` and `z_max` are the
remaining safety net - and they are free on the target, where int8
arithmetic saturates by itself.

**The sensory error climbs one level per iteration.** With a settling
ceiling below `n_levels - 1`, the top state never gets corrected, never
leaves zero, and the network predicts zero - NRMSE exactly 1.0,
indistinguishable from not existing. The plan's "3-10 steps" are not an
arbitrary number: 3 is the depth of the hierarchy. This is pinned down in
`test_settling_budget_below_the_depth_predicts_nothing`.

**The exit decision is discrete, so per-frame behavior is chaotic.**
A 0.01% difference in the settling step is enough for the loop to stop one
iteration earlier or later; one iteration changes the top state, which
crosses into the next frame. Measured: a 1e-4 perturbation in the step gives
~0.1 of difference on a specific frame, with the mean intact. This **changes
the contract with the C side**: validating the inference engine frame by
frame, or by iteration count, is validating noise. See `c/README.md`.

**Evaluation must not perturb training.** The network is online and the top
state crosses frames, which is why `evaluate()` saves and restores the
dynamic state. Without that, measuring mid-training alters the training (it
cost ~0.02 NRMSE).

## Known limitations

- **Prediction quality is not monotonic in the number of passes**, and the
  seed dispersion of the float model is high (0.281 ± 0.100). The Hebbian
  rule has a fixed rate and no progressive cooling, so it keeps oscillating
  around the solution. I tested four annealing schemes: the aggressive ones
  freeze the network before it converges (0.65), the gentle one ties.
  `weight_decay` stabilizes but worsens the minimum. It remains unsolved.
  Meanwhile the measurement protocol works around it - everything that is a
  device comparison is done on the *same* weights, and the rest takes
  ≥3 seeds with dispersion reported.
  Curiously the ternary model is considerably more stable (0.314 ± 0.006):
  quantization acts as a regularizer.
- **The device noise is a generic model**, not the AIMC PoC's. The
  parameters (`sigma_rel`, `sigma_abs`, `stuck_frac`, `read_sigma`) are
  designed to receive values from ngspice, but the numbers are only valid
  when tied to the real circuit. The range that matters - ReRAM/PCM
  variability at 5-20% - is where the cliff starts to appear, so that
  calibration is not a detail.
- **The temporal transition A stays off the crossbar** by default
  (`include_transition=False`): it is 64 weights against 2688, and in the
  design it is the fast path, which makes sense to keep digital. The
  assumption is testable by turning on the flag; I have not tested it.
- **The threshold during training costs some accuracy** compared with
  training at θ = 0 and only applying it at inference. Not explored. On the
  other hand, with device noise the threshold *helps* (0.408 → 0.336):
  silencing small errors also suppresses the substrate's noise.
- **The "fast path" does not yet exist as such.** The top transition is its
  poorest form; the confidence gate that decides when to wake the slow path
  is missing (section 4 of `CONTEXTO.md`).
- **There is no episodic memory or offline consolidation** (section 4).

## Next steps

3. Inference in C, validated against `golden.h` - but with the contract
   revised in `c/README.md`: exact arithmetic on a forced iteration,
   aggregates with tolerance for the rest.
4. ESP32 + crossbar.

And, cutting across both: connect `DeviceModel` to the AIMC PoC's ngspice
model, so the cliff of section 3b is measured in silicon and not in
assumptions.
