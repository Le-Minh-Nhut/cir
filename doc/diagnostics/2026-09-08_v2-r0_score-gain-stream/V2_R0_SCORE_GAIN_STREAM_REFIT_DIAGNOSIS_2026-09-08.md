# V2 R0 ScoreNet Gain-Only Streaming Refit — Diagnostic Report

**Date:** 2026-09-08  
**Experiment:** `score_net_gain_only_stream_refit`  
**Source checkpoint:** `outputs/r0_ncls_text_strong_aux/best.pt`  
**Refit checkpoint:** `outputs/r0_ncls_text_strong_aux_score_gain_stream/score_gain_stream.pt`  
**Dataset:** FashionIQ  
**Backbone:** `qihoo360/fg-clip-base`, native CLS readout  
**Fine-tuning regime:** text-only source checkpoint; streaming rescue updates ScoreNet only  
**K:** 4 candidates  
**T:** 3 maximum edit steps  
**STOP:** enabled, threshold `epsilon_stop = 0.0`

---

## 1. Executive conclusion

The one-pass gain-only streaming refit **does not rescue the STRONG model**.

The source STRONG checkpoint carries a validation Mean Recall of:

- **28.2447**

The gain-only streaming refit obtains:

- **R@10:** 16.3293
- **R@50:** 34.5839
- **Mean Recall:** **25.4566**

This is a decrease of:

- **-2.7881 Mean Recall points**
- approximately **-9.87% relative**

The scorer intervention is therefore **not successful as a benchmark rescue**.

However, the experiment is diagnostically useful. It shows that absolute-gain regression can improve ordinary score/utility regression metrics while still failing at the decision boundary that matters for STOP. On the held-out VAL-160 cohort:

- Pearson improves.
- MAE improves strongly.
- RMSE improves.
- But sign agreement at utility zero becomes worse.
- STOP fires much more often without meaningful precision improvement.
- Premature STOP errors rise sharply.
- Overall selector utility does not improve.
- Official retrieval performance falls substantially.

The most important interpretation is:

> **ScoreNet calibration is a real secondary issue, but gain-only ScoreNet refitting is not sufficient to rescue the system. The dominant remaining bottleneck is still upstream candidate quality / candidate-state quality.**

This is consistent with the earlier OLD-vs-STRONG diagnosis: STRONG frequently presents the selector with harmful candidates, so improving the scorer alone cannot solve the full retrieval problem.

---

## 2. Experimental question

The controlled question was:

> Did the mixed pairwise-ranking + absolute-gain supervision used by STRONG damage ScoreNet's absolute utility calibration enough that removing pairwise supervision and refitting ScoreNet only with canonical absolute gain would substantially improve STOP, selection quality, and FashionIQ retrieval?

The canonical teacher utility is:

\[
u_k = L_{\mathrm{retrieval}}(q_t) - L_{\mathrm{retrieval}}(q_t^{(k)})
\]

Therefore:

- `u_k > 0`: candidate improves retrieval relative to KEEP.
- `u_k < 0`: candidate is harmful relative to KEEP.
- STOP anchor: `0`.
- Runtime decision: execute the highest-scoring candidate only if its predicted score is above zero.

The rescue objective was:

\[
L_{\mathrm{gain}} = \operatorname{Huber}(\hat u_k, u_k)
\]

with:

- `lambda_pair = 0`
- no candidate-generation changes
- no Proposal / Grounder / ActionFusion / Executor updates
- no vision changes
- no text-encoder changes during rescue
- no DPP changes
- no STOP-threshold changes

---

## 3. Intervention validity

The streaming implementation successfully isolates the intervention.

### 3.1 Source behavior policy remains frozen

The source behavior model fingerprint is identical before and after streaming:

```text
behavior before:
6ab08c16775da2f32c1d8e4b384d1229e805e07b1c9234c78ef623ebf0987873

behavior after:
6ab08c16775da2f32c1d8e4b384d1229e805e07b1c9234c78ef623ebf0987873
```

The source behavior ScoreNet fingerprint is also identical:

```text
behavior ScoreNet before:
8b8a14744f8a029c77c1fa0162f7297b9fb3296a7ee398e0dfb7a7a9260ad6b3

behavior ScoreNet after:
8b8a14744f8a029c77c1fa0162f7297b9fb3296a7ee398e0dfb7a7a9260ad6b3
```

Thus every TRAIN rollout used the original frozen STRONG policy.

### 3.2 The standalone ScoreNet actually changes

```text
refit ScoreNet before:
8b8a14744f8a029c77c1fa0162f7297b9fb3296a7ee398e0dfb7a7a9260ad6b3

refit ScoreNet after:
35c366135d6b4fb4030ac360e5ee2d3c7e8e7cb25073d4c47a6548f1d376dd88
```

Therefore the intervention is non-trivial.

### 3.3 Training exposure

Streaming statistics:

- TRAIN samples visited: **18,000**
- stream epochs: **1**
- valid decisions: **52,707**
- candidate teacher labels: **210,828**
- optimizer updates: **1,689**
- invalid teacher rows: **0**
- batch size: **32**
- learning rate: `1e-5`
- weight decay: `0.01`
- ScoreNet trainable parameters: **1,643,777**

Each optimizer update corresponds to valid rows from one live timestep:

```text
score_update_unit = one_live_timestep_valid_rows
```

### 3.4 No target leakage into ScoreNet inputs

Targets are used only to construct teacher utility. ScoreNet inputs remain:

- current global state
- text global representation
- candidate actions
- local residual delta
- execution mask
- candidate global representation

The target embedding is not a ScoreNet feature.

### 3.5 Checkpoint intervention is restricted to ScoreNet

The output checkpoint uses the frozen STRONG checkpoint as its base and replaces only `score_net.*` model weights.

The refit checkpoint is explicitly not an exact full-training resume checkpoint:

```text
optimizer = null
scaler = null
optimizer_scope = score_net_only
exact_full_training_resume = false
resume_semantics = warm_start_only_for_full_training
```

Therefore the comparison is interpretable as a scorer-only intervention.

---

## 4. Held-out VAL-160 comparison protocol

Both diagnostics use the same persistent FashionIQ VAL manifest:

```text
outputs/diagnostics/v2-r0/shared_val_160.json
```

Properties:

- sample count: **160**
- split: `val`
- caption policy: `ordered_and`
- same processed sample IDs
- same order
- same teacher batch grouping

This makes STRONG vs STREAM a paired live-policy diagnostic on the same starting cohort.

At the beginning of the trajectory, both have exactly the same average initial retrieval loss:

```text
initial retrieval loss = 1.2575838566
```

This is important: the two runs begin from the same VAL examples and same upstream model state.

Later recurrent states may differ because the refitted scorer takes different actions or STOP decisions. That later distribution shift is an intended consequence of changing the policy.

---

## 5. Main result: official retrieval degrades

### 5.1 Source STRONG

Source checkpoint validation metric:

```text
Mean Recall = 28.2446771860
```

### 5.2 Gain-only streaming refit

```text
R@10        = 16.3293465972
R@50        = 34.5838715633
Mean Recall = 25.4566090802
```

Per category:

```text
dress  R@10 = 13.2871
dress  R@50 = 32.0773

shirt  R@10 = 18.0569
shirt  R@50 = 35.6722

toptee R@10 = 17.6441
toptee R@50 = 36.0020
```

### 5.3 Delta

```text
Mean Recall:
28.2447 -> 25.4566

absolute delta:
-2.7881 points

relative delta:
-9.87%
```

This is too large and directionally wrong to call the scorer rescue successful.

One final sanity check should still be performed: evaluate the original STRONG checkpoint again under the current HEAD. If it reproduces approximately 28.2447, the benchmark conclusion is fully closed.

---

## 6. Overall live selector behavior

| Metric | STRONG | STREAM gain-only | Delta / interpretation |
|---|---:|---:|---|
| Selected utility | 0.23913 | 0.22290 | **worse** |
| Oracle utility | 0.39876 | 0.38140 | lower because later visited states differ |
| Regret | 0.15963 | 0.15850 | almost unchanged |
| Selected-oracle agreement | 59.27% | 55.92% | **worse** |
| Execute rate | 97.84% | 91.94% | scorer stops much more |
| Oracle execute rate | 89.44% | 88.39% | nearly unchanged |
| STOP rate | 2.16% | 8.06% | much higher |
| Oracle STOP rate | 10.56% | 11.61% | similar |
| Harmful / executions | 32.16% | 32.47% | essentially unchanged / slightly worse |
| Harmful / decisions | 31.47% | 29.86% | lower partly because more decisions STOP |

The tiny regret improvement:

```text
0.159626 -> 0.158503
```

is only about **0.7% relative** and is not accompanied by better selected utility or better benchmark retrieval.

This is not a meaningful selector rescue.

A particularly important denominator distinction is:

- harmful fraction **of decisions** falls because the new scorer executes less;
- harmful fraction **conditional on execution** does not improve.

Therefore the refit has not become substantially better at choosing safe edits when it actually edits.

---

## 7. Regression calibration improves, but decision calibration does not

Held-out VAL scorer calibration:

| Metric | STRONG | STREAM gain-only | Result |
|---|---:|---:|---|
| Pearson | 0.48603 | **0.51321** | better |
| Bias | 0.02724 | **0.02336** | slightly better |
| MAE | 0.39028 | **0.26295** | much better |
| RMSE | 0.66765 | **0.56637** | better |
| Sign agreement at zero | **63.79%** | 61.79% | worse |

This is the central diagnostic result.

The gain-only objective improves ordinary numeric regression:

```text
Pearson: +0.0272
MAE:     -0.1273  (~32.6% lower)
RMSE:    -0.1013  (~15.2% lower)
```

but the one quantity most tightly connected to STOP becomes worse:

```text
sign agreement at zero:
63.79% -> 61.79%
```

The runtime STOP rule is not based on MAE. It is based on whether the best predicted gain crosses exactly zero.

Therefore a scorer can have better MAE/RMSE while making worse policy decisions around the STOP boundary.

### Interpretation

The experiment falsifies the simple version of the hypothesis:

> "If absolute gain regression becomes numerically better, STOP and retrieval should automatically improve."

That implication does not hold here.

A likely mechanism is that the Huber objective is dominated by regression error across the entire utility range and does not directly optimize the zero-crossing decision that controls KEEP vs EXECUTE. This is especially relevant when many candidate utilities lie near zero.

This mechanism is strongly suggested by the data, although it should be described as an interpretation rather than a formally proven causal decomposition.

---

## 8. STOP behavior: recall rises by firing far more often

### STRONG

```text
TP = 2
FP = 8
TN = 407
FN = 47

precision = 20.00%
recall    = 4.08%
F1        = 6.78%

STOP count        = 10
oracle STOP count = 49
premature STOP    = 8
```

### STREAM gain-only

```text
TP = 7
FP = 27
TN = 346
FN = 42

precision = 20.59%
recall    = 14.29%
F1        = 16.87%

STOP count        = 34
oracle STOP count = 49
premature STOP    = 27
```

At first glance, STOP recall and F1 look better.

But the confusion matrix shows what actually happened:

```text
true STOPs gained:       +5
false STOPs added:       +19
premature STOPs:       8 -> 27
precision:          20.0% -> 20.6%
```

So STOP recall increases mostly because the model becomes much more willing to STOP, not because STOP precision becomes meaningfully better.

This is a poor tradeoff for sequential retrieval: an incorrect early STOP removes all future opportunities to improve the query.

---

## 9. Timestep-0 evidence isolates the failure before recurrent shift

The most useful evidence comes from timestep 0.

At `t=0`, both policies see the same parent states and the same candidate generator outputs. Therefore differences at this point are directly attributable to the scorer decision, not accumulated recurrent state drift.

### Candidate utility ranking at t=0

STRONG:

```text
selected utility = 0.23423
oracle utility   = 0.50597
regret           = 0.27174
```

STREAM:

```text
selected utility = 0.25317
oracle utility   = 0.50597
regret           = 0.25280
```

This is actually a small improvement:

```text
selected utility: +0.01894
regret:           -0.01894
```

Thus gain-only training appears capable of improving which candidate is chosen **conditional on continuing**.

### But STOP at t=0 becomes much worse

STRONG:

```text
TP = 2
FP = 4
FN = 26

STOP count     = 6
premature STOP = 4

precision = 33.33%
recall    = 7.14%
```

STREAM:

```text
TP = 7
FP = 20
FN = 21

STOP count     = 27
premature STOP = 20

precision = 25.93%
recall    = 25.00%
```

The STREAM scorer catches 5 additional true STOP cases but creates 16 additional false STOPs at the very first decision point.

This is critical because those premature STOP samples never reach later edit opportunities.

### Why this matters

This t=0 result rules out an overly simple explanation such as:

> "The scorer only looks worse because its new actions create off-policy recurrent states later."

There is indeed recurrent distribution shift after t=0, but the problematic STOP behavior is already visible at the first decision, before that shift can accumulate.

---

## 10. Recurrent consequences

Overall decision count changes:

```text
STRONG decisions = 464
STREAM decisions = 422
```

Overall executed actions:

```text
STRONG executions = 454
STREAM executions = 388
```

The STREAM policy terminates trajectories earlier.

The paired terminal-cohort retrieval statistics reflect this:

### STRONG

```text
initial retrieval loss  = 1.25758
terminal retrieval loss = 0.56409
improvement             = 0.69349

terminal positive - hardest-negative margin = 0.11578
```

### STREAM

```text
initial retrieval loss  = 1.25758
terminal retrieval loss = 0.66969
improvement             = 0.58790

terminal positive - hardest-negative margin = 0.10004
```

So the STREAM policy converts the same initial cohort into a worse terminal cohort:

```text
retrieval improvement:
0.69349 -> 0.58790
delta = -0.10559
```

This is consistent with the official retrieval degradation.

---

## 11. Candidate quality remains the dominant structural problem

The selector still operates over a candidate set with a large harmful fraction.

For the live VAL trajectories, mean candidate counts remain roughly:

```text
STRONG:
positive candidates ≈ 1.75 / 4
harmful candidates  ≈ 2.25 / 4

STREAM:
positive candidates ≈ 1.71 / 4
harmful candidates  ≈ 2.29 / 4
```

These overall values differ slightly because the two policies visit different recurrent states. They do not imply that the upstream generator weights changed.

The key controlled evidence is t=0: candidate/oracle quantities are generated from the same frozen upstream model before policy-induced state divergence.

A scorer cannot manufacture a good edit that is absent from its candidate set. Even an ideal selector can only choose among the candidate actions made available by Proposal + Grounder + ActionFusion + Executor.

The current results therefore support the earlier hierarchy:

```text
PRIMARY BOTTLENECK:
candidate quality / harmful candidate generation

SECONDARY BOTTLENECK:
ScoreNet ranking / absolute calibration / STOP
```

The ScoreNet issue is real, but it is not sufficient to explain or repair the benchmark failure.

---

## 12. What this experiment establishes

### Supported conclusions

1. The scorer-only streaming intervention was technically isolated.
2. Gain-only Huber training changes ScoreNet substantially without changing upstream weights.
3. Absolute-gain regression metrics can improve on held-out VAL.
4. Better MAE/RMSE does **not** imply better zero-boundary behavior.
5. STOP recall improves mainly by increasing STOP frequency.
6. Premature STOP errors increase sharply.
7. At t=0, candidate ranking can improve while STOP simultaneously degrades.
8. Overall selected utility does not improve.
9. Harmful execution conditional on executing does not improve.
10. Terminal retrieval improvement falls.
11. Official FashionIQ Mean Recall falls substantially.
12. Therefore gain-only scorer refitting is not a sufficient rescue.

### Conclusions that are NOT supported

Do **not** claim:

- pairwise loss was harmless;
- pairwise loss was the primary cause of STRONG failure;
- gain-only scorer refit is globally worse under every possible training schedule;
- STOP can never be fixed;
- ScoreNet is irrelevant;
- the candidate generator is the only source of error.

The experiment only establishes that this clean one-pass gain-only ScoreNet rescue does not solve the system and that the remaining candidate-quality problem is more fundamental.

---

## 13. What not to do next

Do not immediately chase the scorer with a large hyperparameter search.

Specifically, avoid making the next experiment a bundle of:

- learning-rate sweep
- many streaming epochs
- epsilon STOP sweep
- extra STOP BCE
- pairwise-loss reintroduction
- calibration temperature
- K/T changes

Doing this now would blur the causal picture and risks overfitting the VAL diagnostic.

Do not tune `epsilon_stop` merely to recover Recall. Zero has semantic meaning under the current marginal-gain definition, so threshold tuning could hide rather than solve gain-sign calibration.

---

## 14. Recommended immediate sanity check

Before formally closing the ScoreNet rescue experiment, reevaluate the original STRONG checkpoint using the **current HEAD evaluation code**:

```bash
python src/evaluate.py \
  backbone=fgclip_base_text_native_cls \
  dataset.root=data/FashionIQ \
  +checkpoint=outputs/r0_ncls_text_strong_aux/best.pt \
  hydra.run.dir=outputs/r0_ncls_text_strong_aux/current_head_eval
```

Expected result should be approximately the source checkpoint metric:

```text
Mean Recall ≈ 28.2447
```

If reproduced, the clean benchmark comparison is:

```text
STRONG current-head baseline ≈ 28.24
STREAM gain-only             = 25.46
```

and the gain-only streaming rescue should be considered closed as a negative result.

---

## 15. Recommended next research experiment

The next intervention should move upstream and directly supervise candidate safety.

The planned loss is:

\[
L_{\mathrm{safe}}
=
\operatorname{mean}
\max\left(
0,\;
L_{\mathrm{candidate}}
-
\operatorname{stopgrad}(L_{\mathrm{parent}})
\right)
\]

Conceptually:

```text
parent state
   |
   +--> candidate 0 --> retrieval loss
   +--> candidate 1 --> retrieval loss
   +--> candidate 2 --> retrieval loss
   +--> candidate 3 --> retrieval loss

candidate worse than parent
        |
        v
direct task penalty on the candidate-producing path
```

The purpose is not to teach ScoreNet to merely avoid harmful candidates.

The purpose is to make Proposal / grounding / action / executor pathways produce **fewer harmful candidates in the first place**.

This attacks the primary bottleneck identified by the OLD-vs-STRONG diagnostics and reinforced by the present negative scorer-rescue result.

`L_safe` should be implemented as a separate controlled experiment, without simultaneously changing DPP, K/T, vision fine-tuning, or selection mechanics.

---

## 16. Final decision

### ScoreNet gain-only streaming rescue

**Status: negative benchmark result / useful diagnostic**

```text
Regression calibration:   improved
Zero-sign agreement:      worsened
STOP recall:              improved
STOP precision:           essentially unchanged
Premature STOP:           strongly worsened
Overall selected utility: worsened
Overall regret:           essentially unchanged
Harmful/edit execution:   not improved
Terminal retrieval:       worsened
Official Mean Recall:     strongly worsened
```

### Research interpretation

> The STRONG ScoreNet has a genuine absolute-calibration problem, but the system is not primarily bottlenecked by that problem. Gain-only refitting improves regression quality without yielding a better decision policy. The next experiment should therefore stop optimizing the scorer in isolation and instead improve the candidate set itself.

---

## 17. Artifact checklist for repository archival

Recommended repository location:

```text
doc/diagnostics/2026-09-08_v2-r0_score-gain-stream/
```

Recommended contents:

```text
V2_R0_SCORE_GAIN_STREAM_REFIT_DIAGNOSIS_2026-09-08.md

shared_val_160.json

stream_refit/
  stream_refit_report.json
  metrics.jsonl
  resolved_config.yaml

strong_val160/
  <all diagnostic JSON / YAML artifacts>

stream_val160/
  <all diagnostic JSON / YAML artifacts>

official_stream_eval/
  <all official evaluation artifacts>

strong_current_head_eval/
  <current-HEAD STRONG evaluation artifacts, after sanity check>
```

This preserves the negative result and enough provenance to reproduce its interpretation.