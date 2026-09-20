# DAC K=8 Branch Report — FashionIQ

**Branch:** `exp/e2e-iag-srme-v2-dac-r0-k8`  
**Core DAC commit:** `37dca51c4cc1e32962bad598bdc9484c5cbb499f`  
**Diagnostics patch commit:** `ee2935303415d8fd6db6d88a412af12a4f6d83a5`  
**Date:** 2026-09-20

---

## 1. Purpose

This branch evaluates a paper-faithful Divide-and-Conquer (DAC) candidate-responsibility schedule inside the recurrent IAG-SRME CIR architecture.

The goal is to test whether hierarchical responsibility assignment can reduce early candidate starvation / winner-take-most collapse.

This experiment intentionally isolates DAC from Functional DPP:

```yaml
objective: core_dac_no_dpp
model: iag_srme_k8
```

The experiment therefore answers:

> Can DAC alone keep multiple candidate pathways trainable and useful before the final hard-WTA stage?

---

## 2. Model and Training Configuration

### Backbone

```text
FG-CLIP Base
checkpoint: qihoo360/fg-clip-base
finetune_policy: full
train_vision: true
train_text: true
train_text_projection: false
global_readout_mode: learned_qg
experiment_identity: R0-QG-FULL
```

The branch does **not** use FG-CLIP2-Large.

### IAG-SRME

```text
width          = 256
num_candidates = 8
max_steps      = 3
num_heads      = 8
exec_dim       = 256
STOP           = enabled
epsilon_stop   = 0.0
```

The K=8 model is architecturally the same IAG-SRME family as the previous K=4 branch, except the candidate dimension is doubled from 4 to 8.

This approximately doubles activation memory in candidate-dependent tensors such as:

```text
proposals
grounding
alpha_read
exec_mask
entities
actions
delta
candidate_states
candidate_global
candidate_queries
delta_q
score features
```

### Training Run

```text
batch size          = 16
learning rate       = 1e-5
weight decay        = 0.01
precision           = fp16
DAC split interval  = 2000 successful optimizer updates
DPP                  = disabled
```

The best checkpoint analyzed in this report:

```text
checkpoint: outputs/2026-09-20/20-30-14/best.pt
epoch:      6
global_step: 6744
mean_recall: 17.5630
```

---

## 3. DAC Schedule

For K=8, DAC uses the hierarchy:

```text
Stage 0:
[0,1,2,3,4,5,6,7]

Stage 1:
[0,1,2,3] [4,5,6,7]

Stage 2:
[0,1] [2,3] [4,5] [6,7]

Stage 3:
[0] [1] [2] [3] [4] [5] [6] [7]
```

Schedule:

```text
step    0–1999 -> stage 0
step 2000–3999 -> stage 1
step 4000–5999 -> stage 2
step >= 6000   -> stage 3
```

At stage 3 DAC becomes hard WTA with respect to candidate-credit assignment.

The analyzed checkpoint is therefore already in the final singleton stage.

---

## 4. Training Result

Validation mean recall during the run:

| Epoch | Mean Recall |
|---:|---:|
| 1 | 15.140 |
| 2 | 14.626 |
| 3 | 16.372 |
| 4 | 16.828 |
| 5 | 16.901 |
| 6 | **17.563** |

The curve improves only slowly and remains far below the stronger historical CIR runs.

Because the checkpoint has already crossed all DAC split stages, continuing the same 20-epoch run without diagnosis was not considered useful.

---

## 5. DAC Responsibility Behavior

At stage 3, candidate responsibility frequency measured by the selector diagnostic was:

```text
C0  11.46%
C1   4.58%
C2   8.75%
C3  13.75%
C4  16.88%
C5  24.58%
C6   9.17%
C7  10.83%
```

Responsibility concentration:

```text
0.15084
```

### Interpretation

This is **not** a catastrophic DAC assignment collapse.

No single candidate receives 70–90% of DAC credit.

Therefore the main failure is not:

```text
DAC responsibility -> single-slot domination
```

The responsibility mechanism itself remains reasonably distributed at the analyzed checkpoint.

---

## 6. Selector Failure

The strongest failure is in ScoreNet selection.

### Selection Histogram

Across 480 analyzed decisions:

```text
C0:   0
C1:   0
C2:  29
C3:   1
C4:   5
C5:  22
C6: 200
C7: 223
STOP: 0
```

Therefore:

```text
C6 + C7 = 423 / 480 = 88.1% of all selections
```

### Oracle Histogram

Teacher-oracle choices were:

```text
C0: 52
C1: 21
C2: 34
C3: 58
C4: 80
C5: 114
C6: 20
C7: 43
STOP: 58
```

Oracle C6+C7 frequency:

```text
63 / 480 = 13.1%
```

This creates an extreme mismatch:

```text
selector chooses C6+C7: 88.1%
oracle chooses C6+C7:   13.1%
```

### Selector Metrics

```text
exact oracle accuracy      = 9.17%
stop/execute accuracy      = 87.92%
harmful execution rate     = 36.88%
score-teacher Pearson      = 0.374
selected teacher utility   = 0.03695
oracle teacher utility     = 0.05482
oracle regret              = 0.01787
```

The selector is therefore not random, but its **top-1 decision quality and calibration are poor**.

---

## 7. Why Pairwise Accuracy Can Still Look Good

The broader diagnostic reports approximately:

```text
pairwise teacher accuracy ~= 82.8%
```

This does not contradict the 9.17% exact-oracle accuracy.

With K=8 there are:

```text
C(8,2) = 28
```

pairwise comparisons.

The selector can classify many easy pairs correctly while still misordering the few candidates near the top.

Inference only uses:

```python
best_idx = scores.argmax(dim=-1)
```

so the highest-ranked candidate matters much more than average pairwise ranking accuracy.

This experiment therefore exhibits:

```text
reasonable pairwise ordering
+
poor top-1 ranking
+
poor score calibration
```

---

## 8. STOP Calibration Failure

STOP uses score 0 as the keep-state baseline.

The model selected STOP:

```text
0 / 480
```

while the oracle selected STOP:

```text
58 / 480 = 12.1%
```

Therefore ScoreNet frequently predicts at least one positive candidate even when no edit should be executed.

This is consistent with the measured harmful execution rate of approximately 36.9%.

---

## 9. Per-Slot Candidate Quality

Mean teacher utility:

| Slot | Mean teacher utility | Positive utility rate | Selected rate | Oracle rate |
|---|---:|---:|---:|---:|
| C0 | -0.0434 | 35.42% | 0.00% | 10.83% |
| C1 | -0.1282 | 21.88% | 0.00% | 4.38% |
| C2 | 0.0231 | 57.92% | 6.04% | 7.08% |
| C3 | -0.0068 | 47.92% | 0.21% | 12.08% |
| C4 | -0.0294 | 28.96% | 1.04% | 16.67% |
| C5 | **0.0357** | 53.96% | 4.58% | **23.75%** |
| C6 | 0.0352 | **64.79%** | **41.67%** | 4.17% |
| C7 | 0.0338 | 63.33% | **46.46%** | 8.96% |

A particularly important observation is that C5 has slightly higher mean teacher utility than C6 and C7, yet ScoreNet strongly prefers C6/C7.

This is direct evidence that the selection imbalance cannot be explained purely by C6/C7 being objectively superior candidates.

---

## 10. Candidate Quality Is Still Insufficient

Selector diagnostic:

```text
mean useful candidate count = 0.89375
valid >=2 useful rate       = 0.15417
positive candidate fraction = 0.46771
```

This means that although many individual edits can have positive utility, most rows do not contain multiple clearly useful alternatives.

The branch therefore still suffers from a **candidate-quality problem**.

DAC may distribute responsibility, but distributed responsibility alone does not guarantee that each branch learns a useful edit mode.

---

## 11. Functional Diversity

Aggregate functional diagnostics:

```text
functional pairwise cosine ~= 0.370
effective rank             ~= 2.256
mean delta-q norm          ~= 0.341
```

This is substantially better than a complete representation collapse near cosine 1.0.

However, effective rank ~2.26 with K=8 indicates that the eight candidates still span a relatively low-dimensional set of effects.

### C6/C7 Near-Duplicate Pair

The two most selected candidates are also very similar:

```text
proposal cosine(C6,C7) ~= 0.988
action cosine(C6,C7)   ~= 0.988
delta-q cosine(C6,C7)  ~= 0.943
```

Thus the selector concentrates most decisions on two candidates that are themselves close to duplicates.

---

## 12. Gradient Routing

A critical diagnostic is the gradient entering upstream candidate-generation modules.

### Terminal vs DAC Candidate-Credit

Representative gradient L2:

| Module | Terminal | DAC credit | Terminal / DAC |
|---|---:|---:|---:|
| Proposal | 0.08495 | 0.04584 | 1.85x |
| Grounder | 0.002608 | 0.000148 | 17.6x |
| ActionFusion | 0.02555 | 0.009485 | 2.69x |
| Executor | 0.20223 | 0.02388 | 8.47x |
| Backbone q_g | 0.03432 | 0.01622 | 2.12x |
| Text backbone | 0.23844 | 0.02155 | 11.07x |
| Vision backbone | 0.23209 | 0.09824 | 2.36x |
| Visual projection | 0.84816 | 0.30342 | 2.80x |

The most concerning ratios are:

```text
Grounder:      terminal ~17.6x DAC
Text backbone: terminal ~11.1x DAC
Executor:      terminal ~8.5x DAC
```

### Interpretation

DAC controls the explicit candidate-credit assignment.

However, the terminal retrieval loss still backpropagates through the **hard-selected recurrent path**.

Therefore two competing routing mechanisms coexist:

```text
DAC candidate credit
    -> tries to distribute candidate training responsibility

hard selector + terminal retrieval
    -> reinforces only executed candidates
```

The terminal path can therefore reintroduce winner-style reinforcement outside the DAC loss.

---

## 13. Per-Slot Proposal Query Gradients

Total gradient L2 on the learned ProposalNet queries:

```text
C0  0.000145
C1  0.000110
C2  0.014644
C3  0.062307
C4  0.043498
C5  0.008966
C6  0.005224
C7  0.055142
```

Max/min ratio:

```text
~564x
```

All eight slots receive some total gradient, but the magnitude is extremely imbalanced.

### Terminal Gradient

Non-zero terminal gradient reaches only:

```text
5 / 8 slots
```

### DAC Candidate-Credit Gradient

Non-zero candidate-credit gradient reaches:

```text
7 / 8 slots
```

at the analyzed stage-3 batch.

This is expected to become sparse after DAC reaches singleton groups, but it also shows that the hard terminal path creates a different and strongly uneven routing pattern.

---

## 14. Loss Composition

Absolute contribution fractions:

```text
pair              ~64.9%
gain              ~19.0%
terminal           ~9.1%
candidate credit   ~6.3%
concept            ~0.6%
bind               negligible
DPP                0%
```

Approximately:

```text
pair + gain ~= 84%
```

of the absolute objective contribution.

Because ScoreNet inputs are detached from upstream candidate-generation features, pair/gain primarily optimize ScoreNet rather than Proposal/Grounder/Executor.

The current system can therefore heavily optimize selector fitting while the candidate generator improves much more slowly.

This is consistent with the observed mismatch:

```text
selector confidence / selection bias
>
actual candidate quality
```

---

## 15. Current Failure Hypothesis

The evidence does **not** support the simple explanation:

```text
DAC itself collapsed immediately to one slot.
```

A more accurate current hypothesis is:

```text
DAC responsibility remains distributed
        ↓
candidate quality remains mediocre
        ↓
ScoreNet develops strong C6/C7 selection bias
        ↓
hard argmax repeatedly executes C6/C7
        ↓
terminal retrieval backpropagates through selected paths
        ↓
hard-path reinforcement competes with DAC responsibility
        ↓
top-1 selector quality remains poor and candidate specialization becomes distorted
```

The dominant observed failure is therefore:

> **selector top-1 / calibration collapse combined with hard-terminal-path reinforcement, while candidate quality remains insufficient.**

---

## 16. Important Non-Issues

### DPP warnings

The diagnostic emits warnings such as:

```text
DPP_MOSTLY_GATED_OFF
DPP_LOW_ACTIVATION
DPP_NOT_REACHING_EXECUTOR_PROBE
```

These are not meaningful for this experiment because:

```text
dpp_enabled = false
lambda_dpp = 0
```

This run intentionally uses `core_dac_no_dpp`.

### Numerical stability

All monitored major tensors were finite.

No NaN/Inf failure was observed.

---

## 17. What This Experiment Establishes

This branch provides several useful conclusions even though retrieval performance is weak.

### Supported

1. Paper-style DAC hierarchical assignment is implemented and functioning.
2. Responsibility is not catastrophically concentrated at the analyzed checkpoint.
3. Functional diversity is better than the earlier near-identical candidate collapse.
4. Candidate usefulness remains low.
5. ScoreNet selection is strongly biased toward C6/C7.
6. ScoreNet top-1 accuracy is poor despite reasonable pairwise ranking accuracy.
7. STOP is underused.
8. Hard terminal gradients are substantially stronger than DAC-credit gradients in several upstream modules.
9. C6/C7 are near-duplicate functional modes despite dominating selection.

### Not yet established

This experiment does **not** yet prove:

1. exactly when C6/C7 selection collapse begins;
2. whether collapse first appears at DAC stage 0, 1, 2, or only after singleton stage;
3. whether reducing terminal hard-path gradients fixes the failure;
4. whether changing ScoreNet objective alone fixes it;
5. whether DAC + DPP improves or worsens candidate quality;
6. whether K=4 DAC behaves differently from K=8 DAC.

These require additional controlled experiments.

---

## 18. Next Diagnostic Priority

The highest-value remaining artifact is:

```text
train_steps.jsonl
```

The next analysis should reconstruct the trajectory across:

```text
stage 0 -> stage 1 -> stage 2 -> stage 3
```

and track:

```text
DAC responsibility frequency
group-winning frequency
per-slot gradient norms
functional cosine/rank
candidate-credit loss
terminal loss
selection behavior if available
```

The key question is:

> At what stage does C6/C7 dominance emerge?

That answer determines whether the next intervention should target:

```text
A. DAC split schedule
B. ScoreNet objective/calibration
C. hard terminal gradient routing
D. candidate-generation architecture
```

---

## 19. Recommended Controlled Experiments

Do not change multiple mechanisms simultaneously.

Recommended next sequence:

```text
1. Analyze train_steps.jsonl from this run.
2. Re-run DAC-K8 with terminal hard-path gradient reduced/frozen during early stages.
3. Compare DAC-K8 no-DPP vs DAC-K8 + DPP.
4. Run DAC-K4 vs aWTA-K4 for the cleanest responsibility-only comparison.
5. Add top-1 / listwise selector objective only if the diagnostics show selector collapse precedes upstream quality collapse.
```

The existing run should not be extended to 20 epochs before the stage-by-stage dynamics are understood.

---

## 20. Current Summary

Current evidence:

```text
DAC assignment:
    reasonably healthy

candidate diversity:
    partial / low-rank

candidate quality:
    insufficient

selector:
    strongly biased toward C6/C7

STOP calibration:
    poor

hard terminal path:
    strong competing gradient route

retrieval performance:
    weak
```

The branch is therefore useful as a negative-but-informative experiment:

> DAC successfully prevents immediate responsibility starvation, but responsibility balancing alone is insufficient when the hard recurrent selector and terminal retrieval path can independently create a winner-reinforcement loop.
