# README — Diagnosis of `exp/e2e-iag-srme-openclip-b16-valsplit`

## 0. Purpose

This document records the controlled OpenCLIP ViT-B/16 ablation performed on the canonical IAG-SRME method and the resulting failure diagnosis.

The purpose of this branch was **not** to redesign IAG-SRME. It was to answer one narrow scientific question:

> Is the visual grounding collapse mainly caused by the local token geometry of FG-CLIP, or is the bottleneck inside the current WHAT→WHERE grounding/interface itself?

The controlled intervention was therefore:

```text
FG-CLIP backbone
        ↓ replace only backbone
OpenCLIP ViT-B/16
```

while preserving the IAG-SRME architecture, losses, recurrence, grounding formulation, selector, optimizer, training hyperparameters, and caption policy.

---

# 1. Branch and commit state

Repository:

```text
https://github.com/Le-Minh-Nhut/cir
```

Branch:

```text
exp/e2e-iag-srme-openclip-b16-valsplit
```

Important commits:

```text
9884228749d24237302db7106115c264e1b3b282
    OpenCLIP controlled ablation implementation

bbaa98874b2956f73db97ec718b1c5454e65f617
    fix: harden OpenCLIP ablation reproducibility contracts

e7e58ea0f397b2447e05cd0198fe88d0350ba2b2
    feat: evaluate both FashionIQ protocols each epoch
```

Final audited HEAD for this diagnosis:

```text
e7e58ea0f397b2447e05cd0198fe88d0350ba2b2
```

No scientific IAG-SRME mechanism was changed by the hardening or dual-evaluation commits.

---

# 2. Controlled OpenCLIP backbone

Pinned backbone:

```text
library              open-clip-torch
library version      3.3.0
architecture         ViT-B-16
pretrained tag       laion2b_s34b_b88k
weights repository   laion/CLIP-ViT-B-16-laion2B-s34B-b88K
weights revision     7288da5a0d6f0b51c4a2b27c624837a9236d0112
weights file         open_clip_model.safetensors
```

Input geometry:

```text
image size    224 × 224
patch size     16 × 16
grid           14 × 14
patch tokens   196
```

The reference visual path uses the normalized final-block OpenCLIP spatial tokens:

```text
[B, 196, 768]
      ↓ pretrained visual.proj
[B, 196, 512]
      ↓ trainable Linear + LayerNorm
[B, 196, 256]
```

The CLS/prefix token is excluded from the grounding anchor.

The reference global retrieval feature and gallery feature both live in the official OpenCLIP retrieval space:

```text
reference global: [B, 512]
gallery global:   [B, 512]
```

The real-checkpoint parity smoke established:

```text
reference_global vs encode_image global:
max absolute error = 0.0
cosine similarity  = 1.0
```

Therefore the reference and gallery retrieval geometry are exactly aligned for the pinned implementation.

---

# 3. Text token masking hardening

OpenCLIP uses an EOT-based pooling convention. Therefore token validity must **not** be inferred from:

```python
input_ids != 0
```

because token ID zero can be a legitimate content token.

The branch instead derives the valid SOT→EOT span from the EOT position:

```python
eot_positions = input_ids.argmax(dim=-1)
positions = torch.arange(input_ids.shape[1], device=input_ids.device)
attention_mask = positions.unsqueeze(0) <= eot_positions.unsqueeze(1)
```

The generic collator then removes:

```text
SOT
EOT
padding
```

and keeps only content tokens.

This was a correctness/reproducibility fix, not a scientific-model change.

---

# 4. Canonical IAG-SRME configuration preserved

The OpenCLIP experiment keeps the canonical method unchanged.

Core constants:

```text
K              = 4
Tmax           = 3
internal width = 256
lambda_z       = 0.10
query_cap      = 0.50
```

Core modules retained:

```text
TextIntentEncoder
AnchorGrounder
GroundedStateReader
GroundedEditContext
SharedTokenEditor
TokenStateReadout
ConsequenceScorer
HardStopSelector
```

Grounding remains:

```math
ell_{k,n} = ((W_Q e_k)^T (W_K A_n)) / sqrt(d_g)
```

followed by:

```math
P_k = Entmax_1.5(ell_k)
```

Candidate states remain same-parent counterfactuals:

```math
Z_{t,k}^{cand} = Z_t + Delta Z_{t,k}
```

Objective remains:

```math
L_total = L_terminal + 0.5 L_marginal
```

No new regularizer, diversity loss, orthogonality loss, binding loss, cross-attention module, or entity mechanism was introduced.

---

# 5. Training configuration

Training command:

```bash
python src/train.py   backbone=openclip_b16_full   experiment=iag_srme_openclip_b16_valsplit   protocol=fashioniq_val   objective=core   dataset.root=data/FashionIQ
```

Important hyperparameters:

```text
optimizer       AdamW
learning rate   1e-5
weight decay    0.01
batch size      32
precision       fp16
caption policy  ordered_and
num workers     8
```

The OpenCLIP vision tower and contextual text encoder are trainable.

The native OpenCLIP text retrieval projection and logit scale remain frozen.

---

# 6. Dual FashionIQ evaluation

After every epoch the exact same model state is evaluated on both:

```text
fashioniq_original
fashioniq_val
```

The two protocols are independent scientific measurements.

They do not affect:

```text
gradient
optimizer
loss
scheduler
training data
action selection
```

Training evaluation shares expensive computation:

```text
validation queries
    ↓ encoded once

full ORIGINAL gallery
    ↓ encoded once

fashioniq_val pair-union gallery
    ↓ selected by exact ordered ID indices
```

Checkpoint outputs:

```text
last.pt
best_original.pt
best_val.pt
```

`best_original.pt` and `best_val.pt` are selected independently.

No ambiguous `best.pt` is written.

---

# 7. Observed training curve

The run already showed a clear peak around epochs 3–4.

| Epoch | Train loss | Original Mean | VAL Mean |
|---:|---:|---:|---:|
| 1 | 1.4777 | 25.132 | 31.169 |
| 2 | 0.7583 | 27.498 | 33.948 |
| 3 | 0.4061 | 27.706 | **34.166** |
| 4 | 0.2270 | **27.759** | 34.120 |
| 5 | 0.1440 | 27.450 | 33.768 |
| 6 | 0.1110 | 26.666 | 32.931 |
| 7 | 0.0879 | 26.084 | 32.296 |

Best checkpoints at that point:

```text
best_val.pt       → epoch 3
best_original.pt  → epoch 4
```

The curve is important:

```text
training loss continues decreasing strongly
            ↓
validation peaks
            ↓
validation degrades
```

This is a strong overfitting pattern.

The OpenCLIP backbone therefore did **not** produce a meaningful retrieval breakthrough relative to the prior FG-CLIP CORE result.

Approximate controlled comparison:

```text
FG-CLIP CORE best original Mean Recall   ≈ 27.258
OpenCLIP B/16 best original Mean Recall  = 27.759
```

Difference:

```text
≈ +0.50 Mean Recall
```

This is too small to support the hypothesis that changing the visual backbone alone solves the canonical failure.

---

# 8. Diagnostic checkpoint

Diagnosed checkpoint:

```text
outputs/2026-08-30/12-43-46/best_original.pt
```

Checkpoint epoch:

```text
4
```

Protocol:

```text
fashioniq_original
```

Diagnostic command:

```bash
RUN=$(ls -td outputs/*/* | head -1)

CUBLAS_WORKSPACE_CONFIG=:4096:8 python src/diagnose_iag_srme_checkpoint.py   --checkpoint "$RUN/best_original.pt"   --dataset-root data/FashionIQ   --protocol fashioniq_original   --batch-size 32   --gallery-batch-size 128   --output reports/openclip_b16_core_best_original.json
```

Global retrieval:

```text
R@10   = 18.8317
R@50   = 36.6862
Mean   = 27.7589
```

---

# 9. CUDA determinism warning

The first diagnostic run emitted repeated warnings such as:

```text
Deterministic behavior was enabled ...
but this operation is not deterministic because it uses CuBLAS ...
set CUBLAS_WORKSPACE_CONFIG=:4096:8
```

These warnings were **not execution failures**.

The report completed successfully.

They mean that strict bitwise CUDA reproducibility was requested, but `CUBLAS_WORKSPACE_CONFIG` was not set before Python/CUDA initialized.

For strict reruns use:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python ...
```

This does not change the scientific interpretation of the current report.

---

# 10. Grounding diagnosis — decisive result

Global grounding statistics:

```text
visual token count                         196

pairwise support cosine mean off-diagonal  0.9991924167
pairwise support overlap mean off-diagonal 0.9863671660

support fraction                           0.2010056525
support effective size                     34.9318
support entropy                            3.43487

dominant tokenwise grounding mass share    0.256365
```

Per-candidate support fractions:

```text
candidate 0  0.19958
candidate 1  0.19966
candidate 2  0.20529
candidate 3  0.19949
```

Per-candidate effective support sizes:

```text
candidate 0  34.85
candidate 1  34.80
candidate 2  35.28
candidate 3  34.79
```

The diagnostic failure flag is:

```text
grounding_clone = true
```

This is the most important result of the entire ablation.

---

# 11. The supports are essentially clones

Global pairwise support matrix:

```text
[
  [1.00000, 0.99922, 0.99912, 0.99920],
  [0.99922, 1.00000, 0.99919, 0.99942],
  [0.99912, 0.99919, 1.00000, 0.99900],
  [0.99920, 0.99942, 0.99900, 1.00000]
]
```

Thus:

```text
P1 ≈ P2 ≈ P3 ≈ P4
```

despite four different candidate intents.

The collapse is also consistent across categories:

```text
Dress   support cosine ≈ 0.999322
Shirt   support cosine ≈ 0.998921
Toptee  support cosine ≈ 0.999340
Global  support cosine ≈ 0.999192
```

Therefore this is not a category-specific artifact.

---

# 12. Intent specialization exists, but WHERE destroys most of it

Global pairwise intent matrix:

```text
[
  [1.00000, 0.95525, 0.96087, 0.95597],
  [0.95525, 1.00000, 0.96202, 0.96439],
  [0.96087, 0.96202, 1.00000, 0.95767],
  [0.95597, 0.96439, 0.95767, 1.00000]
]
```

This is not excellent specialization, but the intents are clearly **less collapsed** than the visual supports.

Conceptually:

```text
candidate intents
    pairwise cosine ≈ 0.95–0.96
            ↓
        AnchorGrounder
            ↓
visual supports
    pairwise cosine ≈ 0.9992
```

This is the strongest evidence for a **WHAT→WHERE contraction bottleneck**.

The grounder maps moderately different candidate semantics to virtually identical spatial distributions.

---

# 13. Context remains highly similar

Global pairwise context cosine:

```text
[
  [1.00000, 0.95448, 0.95962, 0.95051],
  [0.95448, 1.00000, 0.96291, 0.96826],
  [0.95962, 0.96291, 1.00000, 0.95864],
  [0.95051, 0.96826, 0.95864, 1.00000]
]
```

The context representations remain highly correlated.

This is expected once the visual supports are nearly identical.

The system is therefore receiving:

```text
slightly different WHAT
+
almost identical WHERE
        ↓
highly similar grounded contexts
```

---

# 14. Token edit effects also remain highly aligned

Global pairwise token-edit cosine:

```text
delta_z cosine ≈ 0.986–0.990
```

Therefore the actual token-space edits generated by the four candidates are still nearly parallel.

This is downstream-consistent with cloned visual grounding.

---

# 15. Query-space candidate effects are somewhat healthier

Global pairwise final candidate effect cosine:

```text
mean delta_q off-diagonal cosine = 0.862113
```

Global mean functional effective rank:

```text
2.45417
```

So the entire network is **not globally collapsed to rank 1**.

This is important.

The failure is much more localized:

```text
grounding support
    ≈ cloned

token edit directions
    ≈ highly aligned

readout/query effects
    = somewhat more differentiated
```

Hence `candidate_clone_effects=false` at the global diagnostic threshold while `grounding_clone=true`.

---

# 16. Temporal behavior

Per-timestep candidate effect diagnostics:

## t = 0

```text
mean delta_q norm        0.35815
pairwise delta_q cosine  0.99019
functional rank          1.6518
```

The first-step candidate effects are almost identical.

## t = 1

```text
mean delta_q norm        0.09035
pairwise delta_q cosine  0.94736
functional rank          2.3101
```

The effects become more differentiated, but much weaker.

## t = 2

```text
mean delta_q norm        0.02216
pairwise delta_q cosine  0.63471
functional rank          3.4006
```

The effects are much more diverse by direction, but tiny in magnitude.

This reveals a second important pattern:

```text
early step:
large effect
but highly cloned

later steps:
more differentiated
but rapidly vanishing
```

So recurrence partially creates diversity only after the strongest useful update has already occurred.

---

# 17. Dynamic changes across recurrence

Global dynamic diagnostics:

```text
candidate_query mean change  ≈ 0.00193
context mean change          ≈ 0.00912
g_t mean change              ≈ 0.01548
d_t mean change              ≈ 0.02607
delta_z mean change          ≈ 0.000147
score mean change            ≈ 0.15072
```

The candidate scores change noticeably over time, but the candidate token update itself changes only slightly.

Again this suggests that later policy decisions are not driven by fundamentally new candidate interventions.

---

# 18. STOP behavior

Global selection statistics:

```text
STOP occupancy:
t0  0.000665
t1  0.062168
t2  0.438331
```

New STOP hazard:

```text
t0  0.000665
t1  0.061544
t2  0.401099
```

Mean executed edit count:

```text
2.49884
```

Therefore STOP does not collapse immediately and the model is genuinely executing multiple recurrent steps.

The STOP mechanism is **not** the dominant failure.

Diagnostic flags:

```text
all_stop_t0 = false
never_stop  = false
```

---

# 19. Candidate selection behavior

Conditional candidate distribution among executed edits:

```text
candidate 0  0.1561
candidate 1  0.2419
candidate 2  0.4086
candidate 3  0.1933
```

Maximum candidate share:

```text
0.4086
```

Therefore there is no hard single-candidate monopoly.

Diagnostic flag:

```text
single_candidate_monopoly = false
```

However:

```text
fraction of queries with repeated candidate selections = 0.859375
```

This is very high.

Given the strong support similarity, candidate identity has limited functional meaning.

---

# 20. Retrieval controls

Global controls:

```text
FULL                  27.75894
MEAN-CANDIDATE        27.71660
REFERENCE-ONLY        15.60197

REPEAT-0              27.91531
REPEAT-1              27.87459
REPEAT-2              27.81507
REPEAT-3              27.85909

best SINGLE           25.20707
```

Useful ratios:

```text
best_repeat / full     1.00563
best_single / full     0.90807
mean_candidate / full  0.99847
reference / full       0.56205
```

The most important control is:

```text
MEAN-CANDIDATE / FULL ≈ 0.9985
```

The selected multi-candidate policy therefore contributes almost nothing beyond simply averaging candidate effects.

Also:

```text
best REPEAT > FULL
```

although only slightly.

This again says that choosing among four distinct candidates is not currently providing useful specialization.

---

# 21. Counterfactual same-parent candidate controls

At timestep 0:

```text
candidate 0   25.1735
candidate 1   25.1572
candidate 2   25.2071
candidate 3   25.1973
mean          25.1814
oracle        25.4651
```

These values are almost indistinguishable.

The candidate oracle provides only a small gain over choosing any fixed candidate.

This means the four root candidate actions are not meaningfully different retrieval operators.

At timestep 1:

```text
candidate 0   27.4676
candidate 1   27.5519
candidate 2   27.4848
candidate 3   27.5433
mean          27.4844
oracle        27.8097
```

Again, candidate identity has only weak retrieval impact.

---

# 22. Category-level behavior

## Dress

```text
FULL                 25.6569
support cosine        0.999322
best repeat           25.8057
mean candidate        25.6073
repeated selections   0.9187
```

## Shirt

```text
FULL                 28.0177
support cosine        0.998921
best repeat           28.3366
mean candidate        28.0422
repeated selections   0.7875
```

## Toptee

```text
FULL                 29.6022
support cosine        0.999340
best repeat           29.7552
mean candidate        29.5003
repeated selections   0.8730
```

The same grounding-clone pattern is present across all categories.

Therefore the failure is systemic rather than dataset-category specific.

---

# 23. Direct FG-CLIP vs OpenCLIP hypothesis test

Prior FG-CLIP CORE:

```text
intent cosine   ≈ 0.9476
support cosine  ≈ 0.999795
```

Current OpenCLIP B/16 CORE:

```text
intent cosine   ≈ 0.95–0.96
support cosine  ≈ 0.999192
```

The critical observation is:

```text
0.999795 → 0.999192
```

This is only a tiny improvement.

Both backbones produce the same qualitative failure:

```text
four almost-identical WHERE supports
```

Therefore the controlled ablation strongly rejects the hypothesis that FG-CLIP local token geometry is the **main** cause of the grounding collapse.

---

# 24. Updated causal diagnosis

Initial competing hypotheses:

## H1 — backbone-local-geometry hypothesis

> FG-CLIP produces local visual representations that are poorly suited to CIR grounding, causing all actions to attend to the same region.

## H2 — WHAT→WHERE interface hypothesis

> The current AnchorGrounder mapping is too contractive and destroys candidate distinctions when converting semantic intent into spatial support.

## H3 — mixed hypothesis

> Both the backbone and the grounding interface materially contribute.

Observed experiment:

```text
FG-CLIP:
intent moderately differentiated
→ support cosine ≈ 0.9998

OpenCLIP:
intent moderately differentiated
→ support cosine ≈ 0.9992
```

Therefore:

```text
H1 as primary cause      strongly weakened
H2                       strongly supported
H3                       still technically possible,
                         but grounder/interface is now the dominant suspect
```

Current best diagnosis:

```text
THE DOMINANT BOTTLENECK IS THE WHAT→WHERE GROUNDING INTERFACE.
```

More specifically:

```text
e_k
 ↓ current bilinear Q/K grounding
P_k
```

is acting as a severe contraction:

```text
sim(e_i, e_j) ≈ 0.95
        ↓
sim(P_i, P_j) ≈ 0.9992
```

---

# 25. Failure chain

The current failure can be summarized as:

```text
4 text-conditioned candidate intents
          ↓
some semantic variation survives
          ↓
current AnchorGrounder
          ↓
almost identical visual WHERE supports
          ↓
almost identical grounded visual evidence
          ↓
highly similar contexts
          ↓
highly aligned token edits
          ↓
root candidate effects almost cloned
          ↓
candidate identity barely matters
          ↓
MEAN-CANDIDATE ≈ FULL
REPEAT ≥ FULL
          ↓
multi-candidate policy has little functional value
```

This is much more specific than the previous generic “slot collapse” diagnosis.

---

# 26. What is NOT currently the main failure

Based on this run, there is no evidence that the dominant failure is:

```text
STOP immediately collapsing              NO
never stopping                           NO
single-candidate policy monopoly         NO
reference-only retrieval dominance       NO
global functional rank ≈ 1               NO
FG-CLIP-specific local geometry          unlikely as primary cause
OpenCLIP reference/gallery mismatch      NO
incorrect CLS inclusion                  NO
incorrect OpenCLIP padding mask          NO
```

The diagnostic flags explicitly show:

```text
all_stop_t0                             false
never_stop                              false
single_candidate_monopoly               false
candidate_clone_effects                 false
functional_rank_collapse                false
reference_dominates                     false
grounding_clone                         TRUE
```

This greatly narrows the search space.

---

# 27. Interpretation of `candidate_clone_effects=false`

This flag must not be misread as “the candidate mechanism works.”

The global delta-q cosine is approximately:

```text
0.862
```

which is below the diagnostic clone threshold.

However, at the most important first step:

```text
t0 delta-q cosine ≈ 0.990
```

Thus the root candidates are still nearly clones.

Later timesteps become more diverse only while their effect norms collapse:

```text
t0 norm ≈ 0.358
t1 norm ≈ 0.090
t2 norm ≈ 0.022
```

So:

```text
strong updates  → cloned
diverse updates → weak
```

This remains functionally problematic even though the global aggregate flag does not trigger.

---

# 28. What this experiment accomplished

This run should **not** be treated as a wasted failed model.

It answered the intended ablation question.

Before this experiment there were at least two plausible explanations:

```text
A. FG-CLIP is the problem.
B. The grounder/interface is the problem.
```

After replacing FG-CLIP with a substantially different OpenCLIP ViT-B/16 representation while keeping the scientific method fixed:

```text
grounding collapse remains essentially unchanged.
```

Therefore the experiment removes a major branch of uncertainty.

The next architectural work should no longer begin by swapping visual backbones.

---

# 29. Recommended next research target

The next research problem should be stated narrowly:

```text
How should a text/action representation locate a distinct edit-relevant region
in the reference image?
```

The next design should specifically target:

```text
WHAT → WHERE
```

rather than redesigning the entire IAG-SRME system.

Any future mechanism should be evaluated first against the immediate diagnostic target:

```text
pairwise support cosine
```

before interpreting final retrieval.

A successful new grounding mechanism should ideally demonstrate:

```text
candidate intents differ
        ↓
candidate supports differ meaningfully
        ↓
candidate contexts differ
        ↓
candidate token edits differ
        ↓
same-parent candidate retrieval behavior differs
```

Only after that should policy quality be judged.

---

# 30. Suggested acceptance criteria for the next grounding experiment

Do not rely solely on final R@K.

At minimum track:

```text
1. pairwise intent cosine
2. pairwise support cosine
3. pairwise support overlap
4. support entropy/effective size
5. pairwise context cosine
6. pairwise delta_z cosine
7. t0 pairwise delta_q cosine
8. t0 functional effective rank
9. MEAN-CANDIDATE / FULL
10. best REPEAT / FULL
11. candidate oracle gain
12. repeated candidate selection fraction
```

A real improvement should not merely lower one cosine through noise.

It should create a consistent causal chain:

```text
different support
→ different grounded evidence
→ different edit
→ different retrieval consequence
```

---

# 31. Current canonical takeaway

The branch produces the following strongest conclusion:

```text
Changing FG-CLIP to OpenCLIP ViT-B/16 does not solve IAG-SRME grounding collapse.
```

The critical numbers are:

```text
FG-CLIP support cosine      ≈ 0.999795
OpenCLIP support cosine     = 0.999192

OpenCLIP support overlap    = 0.986367
OpenCLIP t0 delta-q cosine  = 0.990185

FULL                        = 27.75894
MEAN-CANDIDATE              = 27.71660
best REPEAT                 = 27.91531
```

Therefore:

```text
THE CURRENT WHAT→WHERE GROUNDING INTERFACE IS THE DOMINANT BOTTLENECK TO INVESTIGATE NEXT.
```

---

# 32. Status

```text
OpenCLIP controlled ablation            COMPLETE
OpenCLIP reproducibility hardening      COMPLETE
dual FashionIQ evaluation               COMPLETE
best_original diagnostic                COMPLETE

backbone-specific collapse hypothesis   strongly weakened
grounding/interface hypothesis          strongly supported

next scientific target                  WHAT → WHERE grounding
```

Do not add another backbone ablation before addressing the grounding interface unless a new experiment has a specific falsifiable backbone hypothesis.