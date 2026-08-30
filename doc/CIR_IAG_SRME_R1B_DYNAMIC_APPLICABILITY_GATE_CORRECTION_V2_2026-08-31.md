# IAG-SRME R1b v2: Dynamic Applicability Gate Correction

## Status

This document defines the corrected canonical R1b experiment. It supersedes the N+1 Entmax
Visual NULL design for future execution but does not erase its historical record. Implementation
and CPU mechanical validation are complete. The CUDA FP16 canary and full FashionIQ training are
pending, so the scientific result remains **INCONCLUSIVE**.

The causal experiment is:

```text
R1b v2 - R1a = one shared dynamic scalar applicability gate
```

Entmax WHERE, IAG-SRME recurrence, editor direction, readout, scorer, selector, STOP, objectives,
backbone, data, optimizer, and training schedule remain unchanged.

Trusted R1a comparison values are:

```text
FashionIQ-original Mean Recall  38.764146
mean executed edits              2.86586 / 3
selected gain t0/t1/t2          +0.07424 / +0.02488 / -0.00261
Delta-q norm t0/t1/t2           0.336634 / 0.272417 / 0.197111
support cosine                  approximately 0.999842
repeated-candidate trajectories approximately 95.7447%
```

## A. Why the original R1b was changed

### Original attempted design

R1b v1 appended one learnable NULL coordinate to all real spatial logits:

```math
P_k^{full}=Entmax_{1.5}([ell^{vis}_{k,1:N},\ell_k^{null}]),
```

```math
p_k^{null}=P^{full}_{k,N+1},\qquad c_k=1-p_k^{null}.
```

The real support mass was preserved in the editor as an execution confidence. That execution
wiring was mechanically correct, but the applicability variable itself lived inside a sparse
normalizer.

### Measured failure evidence

The real GPU investigation observed the following visual-logit distribution:

```text
mean     0.200777
std      0.191942
minimum -0.305997
maximum  0.849577

initial NULL logit 0.0
initial mean p_null approximately 8.738e-05
```

The FP16 canary then observed:

```text
step 3:
  mean p_null approximately 2.005e-04
  max p_null  approximately 1.497e-03
  visual-NULL gradient approximately 2.894e-05

step 4:
  visual-NULL gradient = 0
```

Entmax can assign an exact zero to a coordinate outside its active support. Once the NULL
coordinate is excluded, its local gradient can also be exactly zero. One zero-gradient minibatch
does not prove permanent global death, but it demonstrates an avoidable sparse-support
gradient-death mode in the R1b applicability parameterization. The correction removes
applicability from the sparse spatial random variable.

Blindly increasing the initial NULL logit was rejected: it would tune around one observed logit
distribution without fixing the identifiability error that WHERE and WHETHER shared one sparse
normalizer.

## B. Why Softmax was rejected as canonical

Replacing spatial Entmax with Softmax would avoid exact support exclusion, but it would also
change all of the following:

- spatial sparsity;
- support entropy;
- effective support size;
- exact support fraction;
- grounded-reader pooling behavior;
- editor locality;
- scorer support statistics.

That would confound applicability gating with a new WHERE representation. Softmax can be studied
only as a separate future ablation. Corrected R1b retains Entmax-1.5 over real visual tokens.

## C. Revised intuition: WHERE is not WHETHER

R1b v2 separates two questions:

```text
WHERE: which fixed reference-image tokens express this candidate?
WHETHER: is executing this candidate still applicable in the current recurrent state?
```

WHERE remains:

```math
\pi_k=Entmax_{1.5}(\ell^{vis}_{k,1:N}).
```

Current-state applicability is:

```math
c_{t,k}=\sigma(G_{app}(h_{t,k})).
```

Visual NULL is the Bernoulli complement:

```math
p^{null}_{t,k}=1-c_{t,k}.
```

Thus support sparsity remains a spatial property, while applicability is differentiable through a
finite sigmoid logit.

## D. Full mathematical formulation

### Stable text intent

For contextual text tokens `X [B,L,d]` and four learned query identities `Q [K,d]`, the existing
text-only intent encoder computes:

```math
I=LN(Q+MHA(Q,X,X)+FFN(\cdot)),\qquad I\in\mathbb{R}^{B\times K\times d}.
```

No image, target, or recurrent state enters intent construction.

### Fixed Entmax WHERE

For immutable anchor tokens `A [B,N,d]`:

```math
\ell^{vis}_{k,n}
=\frac{(W_Q I_k)^T(W_K A_n)}{\sqrt{d_g}},
```

```math
\pi_{k}=Entmax_{1.5}(\ell^{vis}_{k,1:N}),
```

```math
\sum_n\pi_{k,n}=1.
```

`pi` is computed once before recurrence and reused at every timestep.

### Grounded recurrent evidence

At state `Z_t`:

```math
O_k=\sum_n\pi_{k,n}A_n,
```

```math
C_{t,k}=\sum_n\pi_{k,n}Z_{t,n},
```

```math
D_{t,k}=\sum_n\pi_{k,n}(Z_{t,n}-A_n).
```

The existing context fuser remains:

```math
h_{t,k}=LN\left(I_k+MLP([I_k;O_k;C_{t,k};D_{t,k};I_k\odot C_{t,k};I_k-C_{t,k}])\right).
```

### Dynamic applicability

One shared head is evaluated for every candidate and timestep:

```math
r_{t,k}=W_{app}LN(h_{t,k})+b_{app},
```

```math
c_{t,k}=\sigma(r_{t,k}),
```

```math
p^{null}_{t,k}=1-c_{t,k}.
```

The head has no candidate-specific network and consumes no target, label, future state, or oracle
utility.

### Editor and candidate consequence

The legacy spatial gate is unchanged:

```math
S_{k,n}=\frac{\pi_{k,n}}{\max_j\pi_{k,j}+\epsilon}.
```

The existing shared editor constructs a bounded direction `u` from normalized current state,
anchor, context, and accumulated change. Corrected R1b executes:

```math
\Delta Z_{t,k,n}
=\lambda_z c_{t,k}S_{k,n}\tanh(u_{t,k,n}).
```

Confidence appears exactly once. Candidate previews retain the same-parent contract:

```math
\widehat Z_{t+1}^{(k)}=Z_t+\Delta Z_{t,k}.
```

The existing readout produces:

```math
\widehat q_{t+1}^{(k)}=Readout(\widehat Z_{t+1}^{(k)},A,q_{ref}),
```

and the target-free shared scorer evaluates each consequence. Four candidate scores are augmented
with fixed STOP score zero. The unchanged hard-forward selector commits one candidate or STOP.
STOP remains identity and absorbing. Applicability is not STOP and does not change STOP logits
directly.

## E. Architectural distinction

```text
R1a
  fixed Entmax WHERE
  no applicability gate

R1b v1 (superseded)
  fixed WHERE
  static NULL as N+1 Entmax coordinate

R1b v2 (canonical correction)
  fixed Entmax WHERE
  dynamic sigmoid WHETHER from current context

future R1c (not implemented)
  dynamic WHERE from Z_t
  dynamic WHETHER
```

R1b v2 does not call the grounder inside recurrence. Dynamic applicability is not dynamic
re-grounding.

## F. Initialization derivation

The configured initial confidence is:

```math
c_0=0.98,
```

```math
p^{null}_0=1-c_0=0.02.
```

The final applicability projection starts with:

```math
W_{app}=0,
```

```math
b_{app}=logit(0.98)=\log\frac{0.98}{0.02}\approx3.8918203.
```

This begins close to R1a: initial edits are scaled by only approximately two percent. The sigmoid
derivative is approximately 0.0196, twice the derivative at confidence 0.99, avoiding unnecessary
near-saturation while preserving baseline behavior. `initial_applicability` is one centralized,
validated config field and must lie strictly between zero and one.

## G. Gradient reasoning

For finite `r`, sigmoid has derivative:

```math
\frac{\partial c}{\partial r}=c(1-c)>0.
```

Unlike an Entmax coordinate outside the sparse active set, applicability is not assigned an exact
structural zero gradient merely because spatial competition excluded it. FP32 sigmoid arithmetic
is used under AMP before returning to the model dtype.

This does not guarantee useful learning. The objective may still learn confidence near one for
every action, collapse confidence near zero, or learn correlations unrelated to action utility.
Those are scientific outcomes to diagnose, not the eliminated sparse-coordinate numerical bug.

## H. Causal contract

The only R1b-v2 scientific intervention over R1a is:

```text
dynamic scalar applicability c_tk multiplying the existing editor effect
```

Preserved exactly:

- `query_cap=1000.0` R1a condition;
- K=4 and Tmax=3;
- stable text intents;
- immutable anchor;
- projected grounding logits;
- Entmax-1.5 over N real visual tokens;
- static spatial support;
- grounded reader and context equations;
- editor direction and spatial max gate;
- same-parent candidate construction;
- readout, scorer, selector, STOP, terminal loss, and marginal loss;
- FG-CLIP Base training regime, FashionIQ data, caption policy, optimizer, and schedule.

No target enters intent, grounding, context, applicability, editor, readout, scorer, selector, or
state transition.

## I. Code changes and tensor contracts

Core responsibilities:

```text
src/models/iag_srme/grounding.py
  exact R1a unit-mass spatial Entmax WHERE

src/models/iag_srme/applicability.py
  shared LayerNorm -> Linear -> sigmoid dynamic WHETHER

src/models/iag_srme/model.py
  grounder once before recurrence; applicability once per timestep

src/models/iag_srme/editor.py
  existing confidence interface, applied exactly once

src/models/iag_srme/outputs.py
  per-step logits/confidence/p_null trace fields
```

Canonical tensors:

```text
intents                     [B,4,256]
anchor                      [B,196,256]
spatial logits              [B,4,196]
spatial Entmax support      [B,4,196]
original evidence           [B,4,256]
current evidence            [B,4,256]
change evidence             [B,4,256]
contexts                    [B,4,256]
applicability logits        [B,4]
visual confidence           [B,4]
visual NULL probability     [B,4]
DeltaZ                      [B,4,196,256]
candidate states            [B,4,196,256]
candidate queries           [B,4,512]
top-level dynamic trace     [B,3,4]
```

## J. Tests and current results

Focused tests cover:

- exact spatial-grounding parity with legacy/R1a;
- unit support mass and exact Entmax zeros;
- exact initialization at confidence 0.98 and NULL 0.02;
- rejection of invalid initial probabilities;
- initial `DeltaZ_R1b = 0.98 * DeltaZ_R1a`;
- monotonic confidence/effect magnitude and exact zero confidence;
- full-confidence legacy editor parity;
- finite nonzero gate weight/bias gradient with small nonzero NULL;
- state-dependent logits after nonzero gate weights;
- one grounder call and Tmax applicability calls;
- same-parent candidates;
- Hydra cap/applicability configuration;
- deterministic self-describing checkpoint replay;
- explicit rejection of N+1 Entmax v1 checkpoints;
- timestep-specific diagnostics and target firewall.

Synthetic smoke measured:

```text
confidence                 0.98000008
p_null                     0.01999998
applicability weight grad  9.979e-03
applicability bias grad    1.917e-03
spatial mass max error     1.788e-07
finite loss               true
target firewall           true
```

The real pinned FG-CLIP Base CPU smoke measured:

```text
terminal loss              0.41788459
marginal loss              0.02000431
total loss                 0.42788675
applicability weight grad  1.13855e-03
applicability bias grad    7.47679e-05
applicability weight delta 2.48767e-07
confidence t0/t1/t2        0.98000002 / 0.98000002 / 0.98000002
p_null mean                0.01999998
spatial mass max error     5.96046e-07
finite                     true
```

Identical confidence across smoke timesteps is expected before learning because `W_app=0`.

## K. CUDA canary contract

The corrected canary reports, per successful step:

- terminal, marginal, and total losses;
- AMP scale, overflow count, successful/skipped steps;
- vision, text, intent, grounding, applicability, editor, readout, and scorer gradients;
- applicability parameter step and cumulative deltas;
- total successful steps with nonzero applicability gradient;
- p_null and confidence mean/min/max;
- p_null and confidence means at t0/t1/t2;
- grounding and applicability call counts;
- STOP, candidate distribution, support, and functional-collapse diagnostics.

It does not abort on one zero applicability-gradient minibatch. After 20 successful optimizer
steps, it requires at least one nonzero applicability-gradient step and nonzero applicability-head
parameter movement. It also requires at least 20 successful steps when the requested canary has at
least 20 attempts.

Command:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python src/canary_train_iag_srme.py \
  --dataset-root data/FashionIQ \
  --steps 100 \
  --precision fp16
```

CUDA was unavailable in the implementation environment. AMP behavior, gate-gradient frequency,
gate movement over 100 real FashionIQ steps, and t0/t1/t2 confidence evolution are pending and
must not be fabricated.

Full training must not start until that GPU canary passes. The matched command is:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python src/train.py \
  dataset.root=data/FashionIQ \
  model=iag_srme_r1b_visual_null \
  experiment=iag_srme_r1b_visual_null \
  protocol=fashioniq_original
```

## L. Scientific kill criteria

Corrected R1b fails or is inconclusive if one or more dominate:

- confidence remains approximately one everywhere after training;
- confidence collapses approximately zero everywhere;
- confidence is unrelated to candidate target-relative utility;
- harmful late t2 transitions remain unchanged;
- useful t0/t1 effects are suppressed;
- R1a Delta-q retention collapses;
- retrieval collapses;
- the gate changes numerically but does not control executable magnitude.

No additional loss may be added in this branch to rescue such an outcome.

## M. Scientific pass criteria

Aligned positive evidence requires multiple observations:

- applicability varies meaningfully with recurrent state/timestep;
- confidence for a previously executed action decreases when later execution is redundant;
- higher p_null associates with smaller DeltaZ and generally smaller Delta-q;
- higher p_null associates with weaker or negative offline target-relative utility;
- selected t2 target-relative gain improves toward zero or positive;
- useful early edits and R1a retention remain preserved;
- retrieval remains competitive with the trusted R1a baseline.

Mean Recall alone is insufficient to establish the applicability hypothesis.

## N. Next experiment

Only after R1b v2 is trained and diagnosed should a separate R1c experiment consider:

```math
\pi_{t,k}=Ground(I_k,Z_t).
```

That would change WHERE dynamically. It is not implemented here. Corrected R1b intentionally
tests fixed WHERE plus dynamic WHETHER only.

## O. Mixed-Precision Applicability Quantization Correction

### Pre-correction CUDA evidence

A real 100-step CUDA FP16 canary ran on an NVIDIA GeForce RTX 5070 Ti:

```text
attempted steps                                      100
successful optimizer steps                           96
AMP overflows                                          4
successful steps with nonzero applicability gradient 96 / 96
finite                                               true
collapse flags                                       all false
```

The applicability parameters moved:

```text
applicability weight max absolute delta  2.3145708837546408e-04
applicability bias max absolute delta    1.3303756713867188e-04
```

This confirms that the v2 sigmoid gate removed the earlier Entmax sparse-support gradient-death
failure. However, its forward actuator remained heavily quantized:

```text
p_null mean start  0.02001953125
p_null mean end    0.02001953125

step-100 confidence t0/t1/t2  0.97998046875 / 0.97998046875 / 0.97998046875
step-100 confidence minimum    0.97998046875
step-100 confidence maximum    0.97998046875
```

Earlier steps occasionally moved by exactly one FP16 quantum between `0.9794921875` and
`0.97998046875`. Around 0.98, FP16 spacing is approximately `4.8828125e-4`.

### Root cause

The earlier forward used:

```python
logits = projection(norm(contexts))       # autocast FP16
confidence = sigmoid(logits.float())      # temporary FP32
confidence = confidence.to(logits.dtype)  # quantized back to FP16
```

Parameters could therefore learn continuously while their forward confidence values collapsed to
a small set of FP16 representable values.

### Corrected FP32 pathway

The complete applicability head now runs with autocast disabled:

```python
with torch.autocast(device_type=contexts.device.type, enabled=False):
    contexts_fp32 = contexts.float()
    normalized = layer_norm(contexts_fp32)
    logits = linear(normalized)
    confidence = sigmoid(logits)
    p_null = 1.0 - confidence
```

Logits, confidence, and NULL probability remain FP32 under FP16 and BF16 autocast. They are not
cast back to the context dtype.

The editor first constructs the unchanged legacy spatial/directional effect, then performs the
applicability multiplication in an FP32 island:

```math
base\_delta=\lambda_zS\tanh(u),
```

```math
\Delta Z=base\_delta_{FP32}\,c_{FP32}.
```

Candidate states and the selected recurrent state remain FP32. State selection uses the same hard
or straight-through action and unchanged absorbing STOP semantics; only its einsum arithmetic is
protected from autocast down-conversion. Query/action policy semantics are unchanged.

### Updated canary diagnostics

Applicability statistics now include only samples live before each timestep. The canary reports:

- confidence and p_null mean/std/min/max globally and at t0/t1/t2;
- mean/max temporal confidence change over surviving trajectories;
- candidate-wise confidence and NULL standard deviation;
- ungated and gated DeltaZ norms;
- elementwise `DeltaZ = confidence * base_delta` error;
- applicability, DeltaZ, and candidate-state dtypes;
- nonzero applicability-gradient fraction;
- cumulative applicability parameter movement;
- maximum observed FP32 applicability variation.

After at least 20 successful optimizer steps, it fails only if applicability never receives a
nonzero gradient, its parameters never move, or all confidence variation remains below the
documented FP32 diagnostic tolerance `1e-7`. A single zero-gradient minibatch is not a failure.

This is a numerical precision correction, not a new scientific mechanism. Initialization remains
`c0=0.98`; Entmax still decides fixed WHERE and the same dynamic sigmoid gate decides WHETHER.

### Post-correction CUDA status

The post-fix GPU canary has not been run in the implementation environment. Its AMP behavior,
continuous confidence range, temporal variation, and actuator response remain pending. No
post-fix CUDA values are claimed here.
