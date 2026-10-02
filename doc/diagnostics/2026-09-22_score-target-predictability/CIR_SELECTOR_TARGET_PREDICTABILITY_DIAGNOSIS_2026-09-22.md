# CIR / IAG-SRME — Selector, Feature Sufficiency, Target Predictability & Teacher-Pool Diagnosis

**Diagnostic checkpoint date:** 2026-09-22  
**Repository:** `Le-Minh-Nhut/cir`  
**Branch:** `exp/e2e-iag-srme-v2-r0-functional-collapse-audit`  
**Latest remote HEAD verified during this diagnostic chain:** `c1058b1f2921318f51a34580c9aecd2f908bcca7` (`Match true validation diagnostic precision`)  
**Production behavior:** unchanged throughout this diagnostic sequence. All new work is diagnostic-only unless explicitly noted otherwise.  
**Primary purpose of this document:** preserve the full causal diagnosis and all key evidence so a new conversation/researcher can continue without reconstructing context from chat history.

---

## 0. Executive conclusion

The current CIR/IAG-SRME failure is **not well explained by a single “candidate collapse” problem anymore**. The evidence now supports a layered diagnosis:

1. **The original proposal functional-collapse problem has largely been removed by the `residual_ln` diagnostic baseline.** Candidate representations/actions/effects are no longer trivially identical.
2. **The production/source ScoreNet still performs poorly even at matched t=0 states.** This rules out the idea that all selector failure is merely downstream trajectory drift.
3. **Frozen refits change behavior, so online/non-stationary training contributes, but it is not the sole cause.**
4. **Basic ScoreNet capacity/optimization is not the fundamental blocker.** A small fixed-cohort overfit test drives the same target-free independent scorer to Pearson ≈ 0.991 and sign@0 ≈ 0.925 on 256 deterministic rows.
5. **Adding ordinary target-free retrieval-space features or set-relative scoring does not solve held-out prediction.** The gains are modest.
6. **Privileged true-target geometry changes the picture dramatically.** On the same TRUE VAL t=0 cohort, a matched diagnostic target-aware probe moves Pearson from ≈ 0.053 to ≈ 0.664, Spearman from ≈ 0.075 to ≈ 0.596, selected utility from ≈ 0.0033 to ≈ 0.0387, and regret from ≈ 0.0574 to ≈ 0.0220.
7. **The current teacher utility is itself strongly dependent on the sampled negative pool.** On TRUE VAL, changing only the negative bank while keeping the anchor query, candidate queries, and true positive target fixed gives canonical-vs-alternative Pearson ≈ 0.535, oracle-action agreement ≈ 47.5%, and pool-induced utility standard deviation ≈ 53.7% of the canonical utility standard deviation.
8. Therefore the strongest current diagnosis is **CASE 2**:  
   **the scorer is missing important target relation information, and the evaluator/teacher target also contains substantial negative-pool-dependent variation.**
9. The next highest-information experiment is **teacher utility decomposition**, not another production architecture change. The goal is to split the teacher into a stable positive-target improvement term and a negative-pool competition term, then test which term is target-free predictable.
10. **Do not yet tune DPP, force uniform slot usage, change STOP thresholds, add another ScoreNet architecture, or alter production behavior.** The evidence says those are downstream of a more fundamental supervision/predictability issue.

---

# 1. Canonical experimental provenance

## 1.1 Source checkpoint

Canonical source checkpoint:

```text
outputs/2026-09-21/20-02-45/best.pt
```

SHA256:

```text
75e47f40e4cd3c3367889b18d39da878668b0eac1d6a72a57ec2e7de06f93de0
```

Checkpoint source git SHA:

```text
7971dea2dd8d871fb3bfe6cccfa51a905ee2fc1f
```

Training state of the checkpoint:

```text
epoch = 1
optimizer step = 560
batch step = 563
proposal_mode = residual_ln
full FG-CLIP finetune
epsilon_stop = 0
```

Source ScoreNet loss:

```text
0.5 * L_pair + 0.5 * L_gain
pair epsilon = 0.01
pair temperature = 1
gain temperature = 1
Huber delta = 1
```

The source planner executes:

```text
K = 4 candidates
T = 3 outer steps
propose -> preview -> ScoreNet -> commit one candidate or STOP -> repeat
```

There is no new external observation after an internal edit; later states are model-generated latent states.

## 1.2 Frozen scorer refits used as diagnostics

Gain-only refit:

```text
outputs/2026-09-21/residual_ln_score_stream/score_stream.pt
```

Pair+gain refit:

```text
outputs/2026-09-21/residual_ln_pair_gain_stream/score_stream.pt
```

Pair+gain refit config:

```text
conf/scorer_refit/pair_gain_stream.yaml
```

Protocol:

```text
split = train
batch size = 32
scorer batch size = 8
AdamW lr = 1e-5
weight decay = 0.01
stream epochs = 3
lambda_pair = 0.5
lambda_gain = 0.5
```

These refits freeze the rest of the model and therefore isolate scorer behavior from producer changes.

---

# 2. Cohort and manifest correctness

This section is critical because an early naming mistake could otherwise invalidate interpretation.

## 2.1 Historical `shared_val160.json` is NOT validation

The historical file:

```text
outputs/diagnostics/2026-09-21/shared_val160.json
```

is actually a TRAIN-160 cohort despite the old filename.

Its metadata showed:

```text
split = train
caption_policy = ordered_and
seed = 42
requested_sample_count = 160
teacher_batch_size = 8
```

Cause: an older `diagnose_candidate_selector.py` defaulted `diagnostic_split` to `train`, and the old command did not override the split.

**Rule:** never call this cohort held-out, validation, or TRUE VAL. Do not rename or delete it because it is useful historical evidence, but label it TRAIN-160.

## 2.2 Canonical held-out TRUE VAL cohort

Correct held-out manifest:

```text
outputs/diagnostics/2026-09-21/shared_val160_true.json
```

Metadata:

```text
split = val
caption_policy = ordered_and
seed = 42
teacher_batch_size = 8
requested_sample_count = 160
```

The manifest loader strictly validates split, caption policy, teacher batch size, seed, and sample count.

## 2.3 TRUE VAL t=0 oracle invariant

The canonical t=0 TRUE VAL invariant is:

```text
oracle slot counts:
C0 = 37
C1 = 34
C2 = 27
C3 = 45
STOP = 17

oracle utility = 0.0607275553047657
oracle STOP rate = 0.10625000298023224
```

This invariant is intentionally used as a hard sanity gate in the target-predictability diagnostic. It caught a real precision-path mismatch later in the diagnostic chain.

---

# 3. Initial mechanism: proposal functional collapse and residual-ln baseline

The early failure mode was candidate specialization collapse / winner-take-most. Proposal candidates could become nearly equivalent, making later selector analysis meaningless because there was little functional choice to rank.

The diagnostic sequence established:

- attention-only proposal representations were functionally collapsed;
- a residual bypass restored meaningful functional diversity;
- the `residual_ln` mode became the diagnostic baseline;
- this does **not** mean the full CIR system is solved; it only removes the immediate “all candidate representations are effectively the same” explanation.

After `residual_ln`, proposal/effect diversity was visibly nontrivial on TRUE VAL. Example source TRUE VAL geometry included roughly:

```text
proposal cosine: ~0.607 to ~0.625
action C0-C3 cosine: ~0.0757
delta_q C0-C3 cosine: ~-0.6215
candidate-state L2: ~15.1 to ~26.8
```

Therefore later selector failure cannot simply be dismissed as “all four candidates are identical.”

**Current status:** proposal functional collapse is considered **largely resolved as the immediate bottleneck** for diagnosis. Do not mechanically revisit ProposalNet unless later evidence points back upstream.

---

# 4. Historical TRAIN-160 selector behavior

These numbers are context only. They are not held-out validation.

## 4.1 Source scorer on TRAIN-160

```text
Pearson                 = -0.0031
selected utility        = -0.00801
oracle utility          = +0.03834
regret                  = 0.04635
harmful / execute       = 69.4%
C3 selected among exec  = 98.1%
exact oracle accuracy   = ~22.2%
STOP                    = ~4.0%
```

This showed severe selector collapse even after candidate functional diversity had improved.

## 4.2 Frozen gain-only scorer on TRAIN-160

```text
Pearson                 = 0.1077
selected utility        = +0.00317
regret                  = 0.03477
harmful / execute       = 47.9%
STOP                    = 52.9%
oracle STOP             = 12.1%
missed STOP             = 46.3%
C0 among executes       = 78.5%
exact oracle            = ~23%
```

Gain-only altered behavior substantially and made the scorer overly pessimistic / over-STOP.

## 4.3 Frozen pair+gain scorer on TRAIN-160

```text
exact oracle accuracy = 0.3229
Pearson               = 0.106694
selected utility      = 0.00325
oracle utility        = 0.03613
regret                = 0.03288
harmful / execute     = 0.5333
execute rate          = 1.0
STOP                  = 0
C0 selection          = 64.167%
C3 selection          = 34.583%
oracle STOP           = 11.875%
sign agreement        = 0.5651
calibration bias      = -0.004053
MAE                   = 0.098887
RMSE                  = 0.126805
```

DPP / candidate usefulness context:

```text
mean useful candidates = 0.5333
rows with >=2 useful   = 0.1375
positive candidate frac= 0.3615
```

Caption shuffle diagnostic:

```text
proposal correct-vs-shuffled cosine = 0.99514
action cosine                       = 0.993173
delta_q cosine                      = 0.999687
candidate correct-shuffled utility  = 0.000098
terminal retrieval correct advantage= 0.009027
```

This was already a major warning that the producer path was only weakly sensitive to language semantics.

---

# 5. TRUE VAL aggregate selector behavior

Aggregate trajectories are policy-dependent after t=0. Therefore t=0 is the clean matched-state comparison and later timesteps must be interpreted as survivor-conditioned states.

## 5.1 Source scorer TRUE VAL aggregate

```text
C3 selected among executes = 97.9%
oracle most often prefers C0
harmful executions         = 59.0%
exact oracle               = 26.8%
DPP mean useful            = 0.96

exact                     = 0.2677
execute                   = 0.9668
selected utility          = 0.00005
oracle utility            = 0.05353
regret                    = 0.05348
Pearson                   = 0.0128
STOP                      = 3.319%
oracle STOP               = 8.186%
calibration bias          = -0.006548
MAE                       = 0.050879
RMSE                      = 0.075514
sign agreement            = 0.51604
```

Correct-vs-shuffled language remains almost unchanged:

```text
action cosine              = 0.993177
delta_q cosine             = 0.999735
raw candidate utility diff = ~0.000126
terminal retrieval advantage= 0.007951
```

## 5.2 Gain-only TRUE VAL aggregate

```text
Pearson           = 0.095643
exact             = 0.1847
selected utility  = 0.00459
oracle utility    = 0.05693
regret            = 0.05234
harmful / execute = 0.4123
STOP              = 54.217%
oracle STOP       = 10.843%
sign              = 0.618474
```

All score deciles were negative, including the strongest positive-teacher bin. This is a classic over-STOP / pessimistic calibration failure.

## 5.3 Pair+gain TRUE VAL aggregate

```text
Pearson           = 0.06345
exact             = 0.2812
selected utility  = 0.00303
oracle utility    = 0.06278
regret            = 0.05976
harmful           = 0.5062
STOP              = 0
oracle STOP       = 9.167%
sign              = 0.568229
C0 selected       = 63.75%
C3 selected       = 35.417%
```

The top teacher bin had positive utility around `+0.1553`, while predicted score was still around `-0.0234`, indicating severe magnitude/nonlinear calibration failure.

---

# 6. Per-timestep matched-state audit

Commit:

```text
d32d781244a4cc4ffa873bab0d98a1414306deb0
Add per-timestep selector diagnostics
```

The per-timestep patch records, per timestep:

- decision count;
- Pearson;
- bias / MAE / RMSE / sign agreement;
- exact oracle accuracy;
- STOP / execute rates;
- selected utility / oracle utility / regret;
- STOP metrics;
- harmful execution rate;
- occupancy / confusion / per-slot behavior.

## 6.1 Why t=0 matters most

At t=0, all three scorers see the same frozen parent states, same candidate set, and same teacher target. Therefore differences are attributable primarily to scorer behavior rather than different trajectories.

At t=1/t=2, policy choices already changed which states survive. Gain-only in particular over-STOPs so aggressively that its later cohorts are much smaller.

## 6.2 Source scorer TRUE VAL by timestep

### t=0

```text
decisions       = 160
Pearson         = 0.0026
exact oracle    = 0.3000
selected utility= 0.00480
oracle utility  = 0.06073
regret          = 0.05592
harmful/execute = 0.5342
STOP            = 0.0875
oracle STOP     = 0.1063
bias            = -0.013138
MAE             = 0.052851
RMSE            = 0.083313
sign            = 0.553125
stop/exec acc   = 0.8187
missed STOP     = 0.0813
STOP P/R/F1     = 0.0714 / 0.0588 / 0.0645
```

Selected occupancy:

```text
C0 = 3   (1.875%)
C1 = 0
C2 = 0
C3 = 143 (89.375%)
STOP = 14
```

Oracle occupancy:

```text
C0 = 37
C1 = 34
C2 = 27
C3 = 45
STOP = 17
```

### t=1

```text
decisions       = 146
Pearson         = 0.0202
exact           = 0.2534
selected        = -0.00080
oracle          = 0.05129
regret          = 0.05209
harmful/execute = 0.6096
STOP            = 0
oracle STOP     = 0.0890
sign            = 0.5000
C3              = 142/146 = 97.26%
```

### t=2

```text
decisions       = 146
Pearson         = 0.0279
exact           = 0.2466
selected        = -0.00432
oracle          = 0.04789
regret          = 0.05221
harmful/execute = 0.6276
STOP            = 0.0068
oracle STOP     = 0.0479
sign            = 0.491438
C3              = 143/146 = 97.945%
```

## 6.3 Gain-only TRUE VAL by timestep

### t=0

```text
decisions       = 160
Pearson         = 0.0682
exact           = 0.1875
selected        = 0.00620
oracle          = 0.06073
regret          = 0.05452
harmful/execute = 0.4035
STOP            = 0.6438
oracle STOP     = 0.1063
bias            = -0.013731
MAE             = 0.046577
RMSE            = 0.079161
sign            = 0.617188
stop/exec acc   = 0.3875
missed STOP     = 0.5750
STOP P/R/F1     = 0.1068 / 0.6471 / 0.1833
```

Selected occupancy:

```text
C0 = 54
C1 = 2
C2 = 0
C3 = 1
STOP = 103
```

### t=1

```text
decisions       = 57
Pearson         = 0.1352
exact           = 0.1754
selected        = -0.00974
oracle          = 0.04780
regret          = 0.05754
harmful/execute = 0.5312
STOP            = 0.4386
oracle STOP     = 0.1404
sign            = 0.631579
```

### t=2

```text
decisions       = 32
Pearson         = 0.3084
exact           = 0.1875
selected        = 0.02207
oracle          = 0.05420
regret          = 0.03213
harmful/execute = 0.2800
STOP            = 0.2188
oracle STOP     = 0.0625
sign            = 0.601562
```

The apparently higher t=2 Pearson is not directly comparable with full-cohort scorers because only 32 survivor states remain after aggressive STOP decisions.

## 6.4 Pair+gain TRUE VAL by timestep

### t=0

```text
decisions       = 160
Pearson         = 0.1023
exact           = 0.2812
selected        = 0.00720
oracle          = 0.06073
regret          = 0.05353
harmful/execute = 0.4938
STOP            = 0
oracle STOP     = 0.1063
bias            = -0.011644
MAE             = 0.099695
RMSE            = 0.131746
sign            = 0.5875
stop/exec acc   = 0.8938
```

Selected occupancy:

```text
C0 = 114 (71.25%)
C1 = 1
C2 = 1
C3 = 44
STOP = 0
```

### t=1

```text
decisions       = 160
Pearson         = 0.0076
exact           = 0.3000
selected        = -0.00404
oracle          = 0.06472
regret          = 0.06876
harmful         = 0.5000
STOP            = 0
oracle STOP     = 0.0875
sign            = 0.560938
C0 = 92
C3 = 67
```

### t=2

```text
decisions       = 160
Pearson         = 0.0851
exact           = 0.2625
selected        = 0.00591
oracle          = 0.06289
regret          = 0.05698
harmful         = 0.525
STOP            = 0
oracle STOP     = 0.0812
sign            = 0.55625
C0 = 100
C3 = 59
```

## 6.5 Causal conclusion from matched t=0

All three scorers have the same t=0 oracle invariant:

```text
decisions = 160
oracle utility = 0.06073
oracle STOP = 0.1063
oracle occupancy = [37, 34, 27, 45, 17]
```

Therefore:

- frozen refits improve some metrics relative to source, so online/non-stationary training **does contribute**;
- pair+gain still has only Pearson ≈ 0.102 at matched t=0 and a strong C0 monopoly;
- gain-only becomes severely over-STOP;
- source remains strongly C3-biased;
- neither ranking nor semantic zero calibration is solved by simply freezing/refitting the existing scorer.

This motivated a feature-sufficiency audit.

---

# 7. Feature-sufficiency audit

Key implementation commit:

```text
511574f5ce9db06bd6bd3bd1ad84e037fe59a39d
Add frozen score feature probes
```

Manifest correction commit:

```text
43b0f186ce91ec2f0691bcd60bc7406f2d9724f2
Use held-out feature probe manifest
```

## 7.1 Research question

At a fixed frozen t=0 state, are current target-free scorer inputs sufficient to predict teacher utility? Specifically:

- is independent candidate scoring the bottleneck?
- are retrieval-space features missing?
- or is the teacher target fundamentally hard to infer from target-free inputs?

## 7.2 Three diagnostic probes

### A. `LegacyIndependentProbe`

Uses target-free legacy-style inputs:

```text
current_global
text_global
actions
local_mean
candidate_global_delta
```

### B. `LegacySetRelativeProbe`

Uses the exact same raw information as A but adds one set-relative multi-head attention block over K candidates. It is permutation-equivariant.

Purpose: test whether relative comparison among candidates is missing from independent scoring.

### C. `RetrievalAugmentedIndependentProbe`

Uses legacy features plus target-free retrieval-space features:

```text
current_query
candidate_queries
delta_q
```

Purpose: test whether the scorer simply lacks direct access to the space in which teacher utility is measured.

## 7.3 No target leakage

Cached diagnostic inputs include only:

```text
current_global
text_global
actions
local_mean
candidate_global_delta
current_query
candidate_queries
delta_q
teacher_utility label
sample_ids
```

Target pixels, target embeddings, teacher losses, positive/negative masks, oracle choices, full states, and full deltas are explicitly forbidden from the target-free cache.

## 7.4 Training protocol

```text
FashionIQ train manifest
caption_policy = ordered_and
seed = 42
teacher batch size = 32
train loader shuffle = False
first 90% fixed groups = probe train
last 10% fixed groups = probe dev
objective = absolute_gain_loss, Huber delta 1
AdamW lr = 1e-4
weight decay = 0.01
5 epochs
FP32 probe training
TRUE VAL used only after training
STOP semantic boundary = score > 0; otherwise STOP
```

Important confound:

- TRAIN/probe-dev teacher groups use batch size 32;
- TRUE VAL teacher groups use batch size 8.

So TRAIN→DEV is the cleanest held-out generalization signal because both use the same teacher grouping regime. DEV→TRUE VAL combines dataset split shift **and** teacher-pool-size / label-distribution shift.

## 7.5 TRUE VAL results

### Legacy independent

```text
Pearson                = 0.0426
Spearman               = 0.0795
sign@0                 = 0.5703
exact oracle           = 0.2188
selected utility       = -0.00310
oracle utility         = 0.06073
regret                 = 0.06382
harmful/execute        = 0.5455
STOP                   = 0.4500
oracle STOP            = 0.1063
```

### Legacy set-relative

```text
Pearson                = 0.0660
Spearman               = 0.0779
sign@0                 = 0.5844
exact oracle           = 0.2875
selected utility       = 0.00464
oracle utility         = 0.06073
regret                 = 0.05609
harmful/execute        = 0.4494
STOP                   = 0.4437
oracle STOP            = 0.1063
```

### Retrieval-augmented independent

```text
Pearson                = 0.0537
Spearman               = 0.1153
sign@0                 = 0.5828
exact oracle           = 0.2188
selected utility       = 0.00514
oracle utility         = 0.06073
regret                 = 0.05559
harmful/execute        = 0.4706
STOP                   = 0.4688
oracle STOP            = 0.1063
```

## 7.6 Learning curves

### Legacy independent

```text
e1 train P .1398 sign .6079 loss .003377 | dev P .1214 sign .6058 loss .003132
e2 train P .1695 sign .6230 loss .003363 | dev P .1359 sign .6146 loss .003134
e3 train P .1917 sign .6227 loss .003336 | dev P .1398 sign .6146 loss .003138
e4 train P .2174 sign .6210 loss .003293 | dev P .1378 sign .6055 loss .003146
e5 train P .2432 sign .6185 loss .003248 | dev P .1299 sign .5989 loss .003167
```

### Legacy set-relative

```text
e1 train P .1322 sign .6157 loss .003405 | dev P .1212 sign .6108 loss .003146
e2 train P .1597 sign .6172 loss .003364 | dev P .1373 sign .6091 loss .003123
e3 train P .1793 sign .6154 loss .003340 | dev P .1431 sign .6062 loss .003124
e4 train P .1990 sign .6119 loss .003315 | dev P .1417 sign .5975 loss .003136
e5 train P .2208 sign .6137 loss .003281 | dev P .1361 sign .5969 loss .003148
```

### Retrieval augmented independent

```text
e1 train P .1245 sign .6175 loss .003404 | dev P .1165 sign .6149 loss .003138
e2 train P .1519 sign .6177 loss .003368 | dev P .1340 sign .6142 loss .003118
e3 train P .1735 sign .6183 loss .003343 | dev P .1413 sign .6126 loss .003115
e4 train P .1971 sign .6203 loss .003315 | dev P .1366 sign .6123 loss .003130
e5 train P .2201 sign .6248 loss .003285 | dev P .1277 sign .6127 loss .003149
```

## 7.7 Interpretation

Key observations:

1. No target-free architecture jumps dramatically.
2. Set-relative context helps some policy metrics but only modestly.
3. Direct retrieval-space features are not a silver bullet.
4. Train Pearson keeps rising while dev peaks around epoch 3 and then falls: simply training longer is unlikely to solve held-out behavior.
5. The small Huber loss (~0.0032) is not evidence of a good scorer because teacher utility scale is small; Huber operates effectively in its quadratic region.
6. Target-free retrieval augmentation does not beat legacy independent even on TRAIN by the final epoch (`~0.220` vs `~0.243`), weakening the hypothesis that missing `current_query/candidate_queries/delta_q` alone is the root cause.
7. All direct-regression probes over-STOP heavily (~44–47%), but this alone does not prove the zero boundary is intrinsically unlearnable. That hypothesis was tested next.

---

# 8. Small-cohort overfit sanity test

Commit:

```text
d04b0df9fefb1fc9dfe078091efde62bc23f1a26
Add small cohort probe overfit check
```

Output report:

```text
outputs/diagnostics/2026-09-22/score_feature_overfit_256/legacy_independent_overfit_report.md
```

## 8.1 Purpose

Before blaming raw-feature insufficiency or direct regression, test whether the unchanged `legacy_independent` architecture can memorize the deterministic mapping on a tiny fixed cohort.

## 8.2 Protocol

```text
first 8 cached TRAIN teacher groups
32 rows/group
256 examples total
teacher batch size = 32
fixed order
no reshuffling
no teacher recomputation
100 epochs
AdamW lr = 1e-4
weight decay = 0.01
objective = absolute gain Huber delta 1
same exact cohort for training and evaluation
```

## 8.3 Results

```text
epoch 1:
loss      .00452
Pearson   .06403
Spearman  .05977
sign      .58594
exact     .17188
selected  .00184
oracle    .05658
regret    .05474
STOP      .59375
oracleSTOP .125


epoch 2:
loss      .00420
Pearson   .12839
Spearman  .11908
sign      .62207
exact     .17188
selected  .00520
regret    .05138
STOP      .79688


epoch 5:
loss      .00315
Pearson   .29416
Spearman  .24772
sign      .50000
exact     .28906
selected  .00784
regret    .04875
STOP      .07812


epoch 10:
loss      .00248
Pearson   .48069
Spearman  .38715
sign      .61816
exact     .30078
selected  .01813
regret    .03845
STOP      .25781


epoch 20:
loss      .00171
Pearson   .69993
Spearman  .57229
sign      .66406
exact     .44922
selected  .03555
regret    .02103
STOP      .15234


epoch 50:
loss      .00065
Pearson   .89572
Spearman  .79683
sign      .80078
exact     .56641
selected  .04562
regret    .01096
STOP      .21094


epoch 100:
loss      .00007
Pearson   .99096
Spearman  .97435
sign      .92480
exact     .80859
selected  .05549
oracle    .05658
regret    .00109
STOP      .12891
oracleSTOP .12500
```

## 8.4 Consequences

This is a crucial negative result against several earlier suspicions:

- basic optimizer failure: **unlikely**;
- basic scorer capacity failure: **unlikely**;
- direct absolute-gain regression cannot learn the semantic zero boundary: **strongly weakened / largely rejected**;
- STOP boundary is intrinsically impossible: **rejected on the finite deterministic cohort**.

The same architecture can learn:

```text
Pearson ~0.991
sign@0 ~0.925
STOP 0.1289 vs oracle 0.125
regret ~0.0011
```

Therefore the full-data problem is much more consistent with **held-out predictability/generalization** and/or **unstable/partially unobservable labels** than with basic function-class or optimizer insufficiency.

This result directly motivated the privileged target-predictability audit.

---

# 9. Target-predictability and negative-pool stability audit

Initial diagnostic implementation commit:

```text
45f8d08d1a63fb995c3939147951eee237a9a5ef
Add target predictability diagnostic
```

Methodology fix for full TRAIN negative reservoir:

```text
8b8676b6f1463c9a40dc0ee391790bc5c63d4e61
Use full train reservoir for stability
```

CUDA target-bank assembly fix:

```text
406927ae32835ad8d7a7c873355af08ed9cbde1e
Fix target bank device assembly
```

TRUE VAL precision-consistency fix:

```text
c1058b1f2921318f51a34580c9aecd2f908bcca7
Match true validation diagnostic precision
```

Final canonical run command:

```bash
python src/diagnose_score_target_predictability.py \
  model.proposal_mode=residual_ln \
  runtime.device=cuda \
  runtime.precision=fp16 \
  hydra.run.dir=outputs/diagnostics/2026-09-22/score_target_predictability_fp16
```

Final run exited successfully:

```text
EXIT: 0
```

Canonical report:

```text
outputs/diagnostics/2026-09-22/score_target_predictability_fp16/score_target_predictability_report.md
outputs/diagnostics/2026-09-22/score_target_predictability_fp16/score_target_predictability_report.json
```

TRUE VAL oracle sanity:

```text
passed = True
```

## 9.1 Part A: target-aware privileged upper bound

The target-free matched baseline is `retrieval_augmented_independent`.

The target-aware diagnostic architecture is deliberately matched but gets three extra **diagnostic-only** scalar relations to the true target:

```text
parent_target_cos
candidate_target_cos
positive_similarity_gain = candidate_target_cos - parent_target_cos
```

No teacher utility, teacher loss, oracle choice, positive/negative mask, whole negative bank, or target pixels are exposed to the probe.

This is not deployable production information. It is a privileged-information upper-bound test.

Three matched seeds:

```text
42, 43, 44
```

Five epochs each, same train/dev protocol, TRUE VAL only after training.

## 9.2 Three-seed target-free results

TRUE VAL Pearson per seed:

```text
seed 42: 0.0270
seed 43: 0.0518
seed 44: 0.0793
```

Mean TRUE VAL metrics:

```text
Pearson             = 0.0527 ± 0.0214
Spearman            = 0.0747 ± 0.0093
sign@0              = 0.5766 ± 0.0056
exact oracle        = 0.2354
selected utility    = 0.00331
regret              = 0.05741
STOP                = 0.4583
```

Train epoch 5:

```text
Pearson             = 0.2271 ± 0.0151
Spearman            = 0.2129 ± 0.0051
sign@0              = 0.6157 ± 0.0062
exact oracle        = 0.2419
selected utility    = 0.00617
regret              = 0.05347
STOP                = 0.4724
```

Probe-dev epoch 5:

```text
Pearson             = 0.1253 ± 0.0040
Spearman            = 0.1337 ± 0.0079
sign@0              = 0.5930 ± 0.0086
exact oracle        = 0.2203
selected utility    = -0.00019
regret              = 0.05624
STOP                = 0.4917
```

This reproduces the prior result that target-free held-out prediction remains very weak.

## 9.3 Three-seed target-aware results

TRUE VAL Pearson per seed:

```text
seed 42: 0.6707
seed 43: 0.6641
seed 44: 0.6562
```

Mean TRUE VAL:

```text
Pearson             = 0.6637 ± 0.0059
Spearman            = 0.5960 ± 0.0058
sign@0              = 0.7365 ± 0.0121
exact oracle        = 0.4042
selected utility    = 0.03873
regret              = 0.02200
STOP                = 0.3896
```

Train epoch 5:

```text
Pearson             = 0.8403 ± 0.0062
Spearman            = 0.7912 ± 0.0072
sign@0              = 0.7565 ± 0.0183
exact oracle        = 0.4765
selected utility    = 0.04351
regret              = 0.01613
STOP                = 0.5489
```

Probe-dev epoch 5:

```text
Pearson             = 0.8317 ± 0.0057
Spearman            = 0.7787 ± 0.0069
sign@0              = 0.7513 ± 0.0164
exact oracle        = 0.4709
selected utility    = 0.04055
regret              = 0.01550
STOP                = 0.5542
```

## 9.4 TRUE VAL target-aware minus target-free

Exact reported deltas:

```text
Pearson                         +0.6109963767
Spearman                        +0.5213072225
sign agreement at zero          +0.1598957777
exact oracle accuracy           +0.1687500477
selected teacher utility        +0.0354112741
oracle regret                   -0.0354112741
harmful execution fraction      -0.2824735940
STOP calibration error          -0.0687499940
```

This is the strongest current evidence in the whole scorer diagnosis.

### Interpretation

A matched probe, with the same frozen candidate set and same t=0 state, becomes dramatically more predictive when it sees only simple true-target geometry.

Therefore:

- candidate outcomes contain useful variation;
- the target-free scorer is not merely failing because “all candidates are bad”;
- a large part of teacher utility is tied to information about the relation between candidate outcome and the hidden true target;
- that information is not sufficiently inferable from current target-free features on unseen samples.

This does **not** prove that the production scorer should be given target information; it cannot be at inference. The experiment diagnoses an information gap.

---

# 10. Negative-pool stability audit

The second half of the target-predictability diagnostic asks whether the teacher label itself is stable if only the negative bank changes.

## 10.1 Fixed quantities

For each anchor, keep fixed:

```text
source checkpoint
current_query
candidate_queries
true positive target
retrieval temperature
```

Only the negative bank changes.

The canonical teacher implementation remains the same production implementation via `marginal_teacher_utilities` / `teacher_retrieval_loss`.

## 10.2 TRUE VAL alternative banks

Protocol:

```text
anchors = all 160 TRUE VAL rows
reservoir = all 160 TRUE VAL target embeddings
bank size = 8
32 deterministic alternative banks per anchor
bank = exact anchor positive + 7 negatives
negative target_id must differ from anchor target_id
prefer distinct negative identities
```

All TRUE VAL canonical rows in this cohort are single-positive:

```text
canonical_single_positive_fraction = 1.0
canonical_multi_positive_fraction  = 0.0
```

## 10.3 TRUE VAL stability results

Canonical-vs-alternative:

```text
Pearson mean         = 0.5349533
Pearson std          = 0.0438152
Pearson min          = 0.4402787
Pearson max          = 0.6328449
Spearman mean        = 0.4974611
sign@0 agreement     = 0.7275391
```

Utility variance:

```text
mean per-entry std              = 0.0417594
median per-entry std            = 0.0331481
p90 per-entry std               = 0.0866241
canonical utility std           = 0.0777252
pool-induced / canonical std    = 0.5372700
```

Candidate ranking / oracle stability:

```text
pairwise order agreement        = 0.7196615
canonical oracle action agreement= 0.4748047
rows changing oracle at least once= 0.96875
```

STOP behavior:

```text
canonical oracle STOP rate      = 0.10625
alternative oracle STOP mean    = 0.1072266
alternative STOP std            = 0.0231814
per-row STOP/execute agreement  = 0.8525391
```

Sign instability:

```text
fraction of candidate entries that flip sign at least once = 0.81875

slot sign-flip-any rates:
C0 = 0.85625
C1 = 0.80000
C2 = 0.81875
C3 = 0.80000
```

Per-slot mean utility standard deviation:

```text
C0 = 0.0307076
C1 = 0.0561973
C2 = 0.0452328
C3 = 0.0348999
```

Pool-to-pool reliability:

```text
Pearson mean = 0.5772741
Pearson min  = 0.3831713
```

## 10.4 TRAIN stability protocol

Protocol:

```text
anchor cohort = first 16 canonical TRAIN teacher groups = first 512 rows
anchor targets = corresponding first 512 true targets
negative reservoir = FULL TRAIN target cache, not only the first 512 rows
bank size = 32
16 deterministic alternative banks per anchor
bank = exact anchor positive + 31 negatives
```

This protocol required a code correction because the initial implementation accidentally used the first 512 anchors as the reservoir. Commit `8b8676b6...` fixed it so the reservoir is the full TRAIN target cache.

All audited TRAIN canonical rows are also single-positive:

```text
canonical_single_positive_fraction = 1.0
canonical_multi_positive_fraction  = 0.0
```

## 10.5 TRAIN stability results

Canonical-vs-alternative:

```text
Pearson mean         = 0.7692894
Pearson std          = 0.0109953
Pearson min          = 0.7497663
Pearson max          = 0.7832169
Spearman mean        = 0.7243789
sign@0 agreement     = 0.7997742
```

Utility variance:

```text
mean per-entry std              = 0.0310849
median per-entry std            = 0.0270253
p90 per-entry std               = 0.0585900
canonical utility std           = 0.0769756
pool-induced / canonical std    = 0.4038288
```

Candidate ranking / oracle stability:

```text
pairwise order agreement         = 0.7980751
canonical oracle action agreement= 0.6212158
rows changing oracle at least once= 0.83984375
```

STOP behavior:

```text
canonical oracle STOP rate       = 0.1328125
alternative oracle STOP mean     = 0.1315918
alternative STOP std             = 0.0138085
per-row STOP/execute agreement   = 0.8576660
```

Sign instability:

```text
fraction flipping sign at least once = 0.6113281

C0 = 0.6113281
C1 = 0.6250000
C2 = 0.5800781
C3 = 0.6289063
```

Per-slot mean utility standard deviation:

```text
C0 = 0.0257663
C1 = 0.0391400
C2 = 0.0319440
C3 = 0.0274895
```

Pool-to-pool reliability:

```text
Pearson mean = 0.7583400
Pearson min  = 0.7086527
```

## 10.6 Interpretation of pool dependence

The teacher is not random, but it is far from invariant to the negative set.

TRUE VAL is especially unstable:

```text
canonical-vs-alt Pearson        ~0.535
oracle agreement                ~47.5%
pool noise / canonical std      ~53.7%
sign agreement                  ~72.8%
```

TRAIN with a larger bank is more stable but still materially dependent on pool composition:

```text
canonical-vs-alt Pearson        ~0.769
oracle agreement                ~62.1%
pool noise / canonical std      ~40.4%
```

The `rows_oracle_action_changes` metric is an “at least once across many pools” statistic and therefore naturally rises with the number of resamples. Do not overinterpret 96.9% as a single-pool disagreement probability. The more robust evidence is the low oracle agreement per alternative pool, moderate canonical-vs-alternative correlation, and large pool-induced standard deviation.

---

# 11. Final diagnostic classification: CASE 2

The diagnostic code's conservative interpretation matrix is:

```text
CASE 1:
target-aware improves strongly + pool stable
=> hidden target relation is the major missing variable

CASE 2:
target-aware improves strongly + pool unstable
=> both hidden target relation and evaluator-pool dependence contribute

CASE 3:
target-aware improvement limited + pool unstable
=> pool dependence / teacher noise dominates

CASE 4:
target-aware improvement limited + pool stable
=> semantic grounding / representation generalization is more likely
```

Observed result:

```text
CASE 2: privileged true-target geometry improves predictability,
but evaluator-pool dependence also contributes.
```

This classification is supported strongly, not marginally:

```text
target-aware TRUE VAL Pearson: 0.6637
vs target-free:                0.0527
Delta:                        +0.6110

TRUE VAL pool-induced std / canonical std: 0.5373
TRUE VAL oracle agreement:                  0.4748
```

---

# 12. What is now supported, weakened, or unresolved

## 12.1 Strongly supported

### A. Proposal functional collapse was a real earlier problem

`residual_ln` restores meaningful functional candidate diversity and is the right diagnosis baseline.

### B. Production/source scorer selection is poor even at t=0

Source t=0 Pearson is ≈ 0.0026 and selection is overwhelmingly C3 despite broad oracle occupancy.

### C. Online/non-stationary training contributes

Frozen refits materially change selection behavior and some metrics. Therefore moving features / co-adaptation are not irrelevant.

### D. Online/non-stationarity is NOT sufficient to explain failure

Pair+gain refit still only reaches t=0 Pearson ≈ 0.1023 and retains heavy slot bias.

### E. Basic target-free scorer capacity is sufficient to memorize a fixed mapping

Small-cohort Pearson ≈ 0.991 at epoch 100.

### F. True-target relation is a major missing predictor of current teacher utility

Target-aware Pearson ≈ 0.664 vs target-free ≈ 0.053 on TRUE VAL.

### G. Negative-pool composition materially changes the current teacher label

TRUE VAL pool-induced variability is large enough to change rankings, utility signs, and oracle choices often.

### H. Text / semantic grounding remains highly suspicious

Correct-vs-shuffled captions barely change proposal/action/delta_q. This suggests the producer path may not encode the modification text strongly enough to infer the hidden target direction.

## 12.2 Hypotheses significantly weakened

### “ScoreNet is simply too small / weak”

Weak explanation after the 256-row memorization test.

### “Absolute gain regression cannot learn STOP / zero boundary”

Weak explanation after sign@0 ≈ 0.925 and STOP ≈ oracle on the memorized cohort.

### “Just give ScoreNet current_query / candidate_query / delta_q”

Weak explanation because retrieval-augmented target-free probes only improve modestly and remain weak held-out.

### “Set-relative attention is the missing solution”

Set-relative context helps some metrics, but not remotely enough to close the gap.

### “Tune epsilon_stop”

Not justified. `epsilon_stop = 0` is the semantic KEEP boundary, and threshold tuning would hide calibration/predictability failure rather than solve it.

## 12.3 Still unresolved

### A. How much of target-free unpredictability remains after removing negative-pool competition from the label?

This is the next major question.

### B. Is weak text conditioning the main remaining reason the stable positive-target gain is hard to predict?

Caption shuffle suggests yes, but this has not yet been isolated against a pool-independent teacher.

### C. Does target-aware performance plateau because the three privileged scalars are only a partial description of the full teacher?

Likely possible. The current teacher also depends on the relation of query/candidates to the whole negative bank. The target-aware experiment intentionally did not expose the negative bank.

### D. Is the teacher itself the right selector target for the final CIR research objective?

Not yet established. It is a one-step retrieval-improvement teacher, not a full finite-horizon value function.

---

# 13. Important methodological caveats

## 13.1 Teacher utility is batch / pool dependent

The teacher is defined through contrastive retrieval against a target bank. Therefore the same candidate can receive different utility when the negative bank changes.

Consequences:

- TRAIN/probe-dev and TRUE VAL use different canonical bank sizes (32 vs 8);
- DEV→TRUE VAL degradation is not pure standard generalization shift;
- comparing raw utility scale across protocols requires caution;
- a scorer trained against these labels is partly learning properties of the evaluator batch, not only positive-target progress.

## 13.2 Later timesteps are survivor-conditioned

After t=0, different scorers create different state distributions. Do not compare t=1/t=2 metrics as if cohorts were matched.

## 13.3 Pairwise loss is translation-invariant

Pairwise supervision can improve relative ordering while leaving the absolute zero / STOP boundary unconstrained. This explains why pair+gain can rank somewhat better yet choose STOP=0.

## 13.4 Small-cohort memorization is not generalization evidence

The overfit test proves capacity/optimization on a fixed deterministic mapping. It does not prove the same mapping is learnable out of sample.

## 13.5 Privileged target-aware probe is not deployable

Its purpose is causal diagnosis of missing information, not a proposed production input.

## 13.6 Precision must match the canonical path

The canonical feature-sufficiency TRUE VAL path used FP16 autocast for the frozen model forward / target encoding. A later diagnostic accidentally recomputed TRUE VAL without autocast, shifting one near-tied oracle row.

This was correctly caught by the oracle sanity gate.

---

# 14. Diagnostic bugs caught during implementation and why they matter

This section is preserved because the failures are instructive for future reproducibility.

## 14.1 Wrong command omitted `model.proposal_mode=residual_ln`

Attempting to run target-predictability with only:

```text
runtime.device=cuda
runtime.precision=fp32
```

failed metadata validation:

```text
ValueError: checkpoint behavior mismatch (proposal_mode)
```

Correct fix: pass

```text
model.proposal_mode=residual_ln
```

Do not use `+allow_counterfactual_eval=true` because the intended run is not a counterfactual evaluation.

## 14.2 TRAIN stability initially used only the first 512 targets as negative reservoir

That would make the anchor cohort and negative reservoir the same small subset, biasing the stability estimate.

Fixed in commit:

```text
8b8676b6f1463c9a40dc0ee391790bc5c63d4e61
```

Correct protocol:

```text
anchors = first 512 TRAIN rows
negative reservoir = full TRAIN target cache
```

## 14.3 CUDA device mismatch before target-bank concatenation

The anchor target was on CUDA while sampled negatives remained on CPU before `torch.cat`, which would fail only on GPU.

Fixed in:

```text
406927ae32835ad8d7a7c873355af08ed9cbde1e
```

A regression test now verifies teacher inputs share the requested device.

## 14.4 TRUE VAL recomputation ignored resolved precision

The first target-predictability TRUE VAL path called:

```text
model(...)
model.encode_global_images(...)
```

outside autocast, so even `runtime.precision=fp16` recomputed in FP32.

Observed wrong invariant:

```text
(38, 34, 26, 45, 17)
oracle utility = 0.0607188940
STOP rate = 0.10625
```

Expected canonical FP16 invariant:

```text
(37, 34, 27, 45, 17)
oracle utility ~= 0.0607275553
STOP rate = 0.10625
```

Only one near-tied row changed C2→C0, but the hard gate correctly rejected the run.

Fixed in:

```text
c1058b1f2921318f51a34580c9aecd2f908bcca7
```

The frozen model and target encoder are now wrapped in the same resolved autocast structure as the canonical feature-sufficiency diagnostic.

## 14.5 Cache precision technical debt

The privileged target-cache metadata, at the time of this diagnostic chain, does not encode precision as a compatibility field. Therefore FP16 and FP32 target caches should not be silently reused across each other.

Operational rule:

```text
use separate output directories by precision
```

Canonical final run used:

```text
outputs/diagnostics/2026-09-22/score_target_predictability_fp16
```

The earlier failed FP32-oriented directory should not be treated as canonical evidence.

---

# 15. Current mathematical interpretation of the teacher

For a single-positive retrieval target bank, write the teacher retrieval loss for query `q` as:

```text
L(q) = log sum_j exp(s(q,t_j)/tau) - s(q,t+)/tau
```

For candidate query `q_k`, one-step utility is:

```text
u_k = L(q) - L(q_k)
```

This can be decomposed exactly:

```text
u_k
= [s(q_k,t+) - s(q,t+)] / tau
  + [log Z(q) - log Z(q_k)]
```

where:

```text
Z(q) = sum_j exp(s(q,t_j)/tau)
```

Define:

```text
U_positive(k) = [s(q_k,t+) - s(q,t+)] / tau
U_pool(k)     = log Z(q) - log Z(q_k)
U_total(k)    = U_positive(k) + U_pool(k)
```

Interpretation:

- `U_positive` asks: **did this candidate move closer to the true positive target?**
- `U_pool` asks: **how did this candidate change competition against the sampled target bank?**

The target-aware probe directly exposes geometry closely related to `U_positive`. The negative-pool resampling audit directly shows that `U_pool` can vary substantially with evaluator composition.

This decomposition now provides the clearest route to the next experiment.

---

# 16. Next experiment: teacher decomposition audit

## 16.1 Primary question

Is the current target-free scorer failing mainly because:

1. the teacher target contains avoidable pool-dependent noise/variation, or
2. even the stable positive-target progress term is not predictable because semantic grounding is weak?

## 16.2 Required diagnostic outputs

For the same frozen t=0 candidate rows, compute exactly:

```text
U_total
U_positive
U_pool
```

and verify numerically:

```text
U_total ~= U_positive + U_pool
```

Measure:

```text
corr(U_positive, U_total)
corr(U_pool, U_total)
variance(U_positive)
variance(U_pool)
sign agreement
candidate-order agreement
oracle action agreement including STOP
```

Under negative-pool resampling:

```text
stability(U_positive)
stability(U_pool)
stability(U_total)
```

`U_positive` should be invariant to negative-pool composition when query, candidate, target, and temperature are fixed. Any observed change there would indicate an implementation/protocol error.

## 16.3 Predictability experiment

Train the same target-free probe architecture and protocol against:

```text
A. U_total    (current teacher)
B. U_positive (pool-independent positive progress)
```

Use the same train/dev/TRUE VAL cohorts and multiple seeds.

### Decision rule

If target-free `U_positive` prediction improves substantially over `U_total`:

```text
=> current teacher formulation injects substantial avoidable evaluator-pool dependence
=> redesign selector supervision before redesigning scorer architecture
```

If target-free `U_positive` remains weak:

```text
=> removing pool noise is insufficient
=> upstream semantic/text grounding becomes the leading bottleneck
```

Given the caption-shuffle diagnostics, the second failure mode remains highly plausible.

---

# 17. What NOT to do yet

Do not introduce multiple production changes before the teacher decomposition audit.

Specifically avoid, for now:

```text
- arbitrary epsilon_stop tuning
- forcing uniform candidate usage
- DPP weight tuning as a primary fix
- adding MoE routing purely to flatten occupancy
- replacing ScoreNet with a larger network without evidence
- adding set attention and retrieval features together in production
- longer training of the same target-free direct regression probe
- interpreting t=1/t=2 survivor metrics as matched-state evidence
- calling old shared_val160.json “validation”
- treating privileged target-aware inputs as deployable
```

The current failure is more fundamental: **what target is being predicted, and whether that target is inferable from production-available information.**

---

# 18. Key code and commit map

```text
511574f5ce9db06bd6bd3bd1ad84e037fe59a39d
  Add frozen score feature probes
  + src/diagnose_score_feature_sufficiency.py
  + src/diagnostics/feature_sufficiency.py
  + tests/test_score_feature_sufficiency.py

43b0f186ce91ec2f0691bcd60bc7406f2d9724f2
  Use held-out feature probe manifest
  default TRUE VAL path fixed to shared_val160_true.json

d04b0df9fefb1fc9dfe078091efde62bc23f1a26
  Add small cohort probe overfit check
  + src/diagnose_score_feature_overfit.py

45f8d08d1a63fb995c3939147951eee237a9a5ef
  Add target predictability diagnostic
  + src/diagnose_score_target_predictability.py
  + src/diagnostics/target_predictability.py
  + tests/test_score_target_predictability.py

8b8676b6f1463c9a40dc0ee391790bc5c63d4e61
  Use full train reservoir for stability

406927ae32835ad8d7a7c873355af08ed9cbde1e
  Fix target bank device assembly

c1058b1f2921318f51a34580c9aecd2f908bcca7
  Match true validation diagnostic precision
```

Tests at the latest target-predictability commit:

```text
focused tests: 14 passed
full tests:    199 passed, 2 skipped
Ruff:          passed
compile:       passed
```

---

# 19. Important local output paths

Canonical target-predictability run:

```text
outputs/diagnostics/2026-09-22/score_target_predictability_fp16/
```

Important files:

```text
score_target_predictability_report.md
score_target_predictability_report.json
.hydra/config.yaml
.hydra/overrides.yaml
.hydra/hydra.yaml
privileged_target_cache/         # large / regenerable / do not commit
```

Feature-sufficiency output root:

```text
outputs/diagnostics/2026-09-21/score_feature_sufficiency_t0/
```

Important cache:

```text
compact_cache/                   # ~510 MiB estimated payload; do not commit
```

Small-cohort overfit output:

```text
outputs/diagnostics/2026-09-22/score_feature_overfit_256/
```

Canonical TRUE VAL manifest:

```text
outputs/diagnostics/2026-09-21/shared_val160_true.json
```

TRAIN feature-probe manifest:

```text
outputs/diagnostics/2026-09-21/score_feature_probe_train.json
```

Avoid pushing large caches/shards/checkpoints. Archive the reports, selected manifests, and run metadata under `doc/diagnostics/`.

---

# 20. Recommended GitHub archival layout

Recommended tracked destination:

```text
doc/diagnostics/2026-09-22_score-target-predictability/
├── CIR_SELECTOR_TARGET_PREDICTABILITY_DIAGNOSIS_2026-09-22.md
├── target_predictability/
│   ├── score_target_predictability_report.md
│   ├── score_target_predictability_report.json
│   └── run_metadata/
│       ├── config.yaml
│       ├── overrides.yaml
│       └── hydra.yaml
├── feature_sufficiency/
│   └── ...report files only...
├── feature_overfit/
│   └── ...report files only...
└── manifests/
    └── shared_val160_true.json
```

Do not commit:

```text
privileged_target_cache/
compact_cache/
*/shards/
*.pt checkpoint caches
large raw feature tensors
```

---

# 21. Minimal state for a new researcher/chat

If this file is the only context available, the next researcher should know:

1. Use `residual_ln` checkpoint `best.pt` with SHA256 `75e47...93de0`.
2. Never use historical `shared_val160.json` as validation; it is TRAIN-160.
3. TRUE VAL invariant is `[37,34,27,45,17]`, utility `~0.0607275553`, STOP `~0.10625`.
4. Source scorer fails badly at t=0; frozen refits change behavior but do not solve ranking/calibration.
5. Target-free feature probes generalize poorly; set-relative/retrieval features provide only modest gains.
6. The scorer can memorize a fixed 256-row mapping extremely well, so raw capacity/optimizer is not the core problem.
7. Privileged true-target geometry gives a huge TRUE VAL predictive jump (`~0.053 -> ~0.664 Pearson`).
8. Negative-pool composition materially changes current teacher utility (`~0.535 canonical-vs-alt Pearson on TRUE VAL`).
9. Therefore current state is **CASE 2: hidden-target relation + evaluator-pool dependence**.
10. Caption shuffle barely changes candidate outputs, so semantic grounding remains a leading upstream suspect.
11. Do not modify production yet.
12. Next task: decompose `U_total = U_positive + U_pool` and compare target-free predictability of `U_positive` versus current `U_total`.

---

# 22. Final scientific position at this checkpoint

The evidence no longer supports framing the project as merely “four candidates collapse, so add diversity.” The more precise current picture is:

```text
candidate diversity exists
        ↓
selector still cannot reliably identify useful candidate
        ↓
not explained by basic scorer capacity
        ↓
not solved by ordinary target-free retrieval features
        ↓
true-target relation dramatically increases predictability
        +
current teacher changes materially with sampled negatives
        ↓
production scorer is asked to approximate a partially hidden,
pool-dependent training target
```

The correct immediate research question is therefore not:

```text
“How do we make ScoreNet larger or more diverse?”
```

It is:

```text
“What portion of the teacher utility is stable, semantically meaningful,
and predictable from information available at inference?”
```

Answering that question should precede the next production intervention.
