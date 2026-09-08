# CIR V2 R0 Text-Only Failure Diagnosis — OLD vs STRONG

**Date:** 2026-09-08  
**Repository:** `Le-Minh-Nhut/cir`  
**Branch:** `exp/e2e-iag-srme-v2-r0`  
**Experiment family:** `R0-NCLS-TEXT`  
**Backbone/readout:** `fgclip_base_text_native_cls` / native CLS readout  
**Fine-tune policy:** text-only; vision side remains frozen  
**Dataset/split for diagnostics:** FashionIQ TRAIN, fixed manifest of 160 samples  
**Purpose:** explain why the STRONG auxiliary checkpoint underperforms OLD and determine the next intervention without changing multiple things at once.

---

## 1. Executive diagnosis

The current evidence supports the following failure mechanism:

```text
STRONG auxiliary regime
(lambda_c, lambda_bind, lambda_rel, lambda_dpp all increased together)
        ↓
candidate slots become strongly asymmetric
        ↓
C0/C1/C2 lose substantial cross-input variation and become weak/harmful
C3 remains high-rank and becomes the main retrieval-useful specialist
        ↓
oracle itself strongly prefers C3
        ↓
ScoreNet also selects C3 most of the time
        ↓
the candidate set usually contains only ~1 useful candidate
        ↓
STOP calibration remains poor and the selector executes too often
        ↓
harmful executions + larger oracle regret
        ↓
lower official FashionIQ Recall
```

The failure is **not** well described as simple whole-model latent collapse.

The more accurate description is:

> **candidate-quality collapse / useful-slot specialization, accompanied by slot-specific loss of input dependence, transient recurrent rank concentration, and secondary ScoreNet/STOP calibration error.**

The strongest upstream evidence is not merely low pooled rank. It is the combination of:

1. per-slot geometry showing C0/C1/C2 becoming low-rank while C3 remains high-rank;
2. strong between-slot variance in STRONG but almost none in OLD;
3. three STRONG slots having negative mean teacher utility while C3 alone is positive;
4. the oracle itself choosing C3 about 75% of oracle-execute decisions;
5. correct captions providing almost no advantage over shuffled captions for mean candidate utility / best-candidate utility / STOP-aware oracle-policy utility;
6. only about one useful candidate per parent under STRONG.

ScoreNet and STOP are also faulty, but they are **not sufficient to explain the upstream candidate asymmetry**.

---

# 2. Experimental controls and comparability

## 2.1 Checkpoints

### OLD

```text
checkpoint: outputs/r0_ncls_text/best.pt
epoch:      15
metric:     30.824426313241325
```

### STRONG

```text
checkpoint: outputs/r0_ncls_text_strong_aux/best.pt
epoch:      18
metric:     28.244677186012268
```

The two checkpoints are **not matched by training epoch/update budget**, so they must not be interpreted as a clean single-factor training ablation.

---

## 2.2 Objective configuration

Shared important settings:

```text
lambda_pair = 0.5
lambda_gain = 0.5
terminal_weight = 1.0
retrieval_temperature = 0.07
num_candidates = 4
```

Changed auxiliary weights:

| Weight | OLD | STRONG |
|---|---:|---:|
| `lambda_c` | 0.01 | 0.6 |
| `lambda_bind` | 0.01 | 1.0 |
| `lambda_rel` | 0.001 | 0.6 |
| `lambda_dpp` | 0.01 | 0.6 |

Therefore:

> **The present evidence does not identify DPP alone as the causal source.**

All four auxiliary weights were increased together. The diagnostics establish that the **STRONG auxiliary regime** converged to a bad equilibrium; isolating which auxiliary is responsible requires controlled ablations later.

---

## 2.3 Fixed diagnostic cohort

Both selector diagnostics used:

```text
manifest_sample_count  = 160
processed_sample_count = 160
split                  = train
caption_policy         = ordered_and
diagnostic_batches     = 20
diagnostic_batch_size  = 8
teacher_invalid_rows   = 0
```

The OLD and STRONG reports share the same:

```text
processed_sample_ids_sha256
fc109e3d12fb101f1fd7cfc84c1a3be5bbc010b0bb3e825806083f2f0e6e6fc1
```

and:

```text
teacher_batch_grouping_sha256
a6676103c20c3dcb9b68da0babe861d00ff71f43841c72d17c76c8c8c1c34800
```

This is important because candidate utility is batch/negative-set dependent.

---

## 2.4 Matched OLD/STRONG cohorts by timestep

The cross-checkpoint matched geometry report gives:

| Timestep | OLD live | STRONG live | Intersection |
|---|---:|---:|---:|
| t0 | 160 | 160 | 160 |
| t1 | 158 | 155 | 154 |
| t2 | 149 | 154 | 145 |

Thus later-timestep comparisons are not based on completely different survivor populations.

There is still normal survival-policy divergence, so same-timestep OLD/STRONG comparisons should use the matched intersection, which the new report does.

---

# 3. Official benchmark regression is real

Official FashionIQ:

| Metric | OLD | STRONG | STRONG − OLD |
|---|---:|---:|---:|
| R@10 | 20.6236 | 18.3413 | **-2.2823** |
| R@50 | 41.0252 | 38.1481 | **-2.8772** |
| Mean Recall | **30.8244** | **28.2447** | **-2.5797** |

Category-level regression:

| Metric | OLD | STRONG | Delta |
|---|---:|---:|---:|
| Dress R@10 | 18.6911 | 16.3114 | -2.3798 |
| Dress R@50 | 39.7124 | 36.8865 | -2.8260 |
| Shirt R@10 | 21.0991 | 19.2836 | -1.8155 |
| Shirt R@50 | 40.5790 | 37.7821 | -2.7969 |
| Toptee R@10 | 22.0806 | 19.4289 | -2.6517 |
| Toptee R@50 | 42.7843 | 39.7756 | -3.0087 |

The degradation is broad across categories, not isolated to one FashionIQ class.

---

# 4. Geometry diagnosis

## 4.1 Why the old pooled-rank conclusion was insufficient

The old diagnostic flattened `[B,K,D] -> [BK,D]`.

That mixes:

- variation across input samples;
- fixed differences between candidate slots.

A low pooled participation ratio can therefore be caused by slot offsets even when every slot remains healthy across inputs.

The corrected diagnostic now measures:

- per-slot spectrum across samples;
- slot-centered pooled spectrum;
- between-slot / total variance fraction;
- raw and L2-normalized spectra;
- matched temporal geometry on the same survivor IDs.

This resolves the central ambiguity from the original audit.

---

# 5. t0 per-slot geometry: the key upstream finding

## 5.1 OLD is almost slot-symmetric

### Proposal PR

| Slot | PR |
|---|---:|
| C0 | 19.7489 |
| C1 | 19.7410 |
| C2 | 19.7890 |
| C3 | 19.7917 |

```text
slot-centered pooled PR       = 19.7782
between-slot variance fraction = 0.000095
```

### Action PR

| Slot | PR |
|---|---:|
| C0 | 19.2683 |
| C1 | 19.2565 |
| C2 | 19.3005 |
| C3 | 19.3082 |

```text
slot-centered pooled PR       = 19.2930
between-slot variance fraction = 0.000091
```

### delta_q PR

| Slot | PR |
|---|---:|
| C0 | 15.3494 |
| C1 | 15.3056 |
| C2 | 15.3821 |
| C3 | 15.3999 |

```text
slot-centered pooled PR       = 15.3604
between-slot variance fraction = 0.000016
```

Interpretation:

> OLD candidates are functionally highly redundant, but each slot preserves roughly the same level of cross-input variation.

This agrees with the previous functional-collapse observation for OLD: candidates are too similar to siblings, but there is no evidence that one slot alone owns the useful input-conditioned subspace.

---

## 5.2 STRONG is strongly slot-asymmetric

### Proposal PR

| Slot | PR |
|---|---:|
| C0 | **4.4620** |
| C1 | **7.5842** |
| C2 | **2.6416** |
| C3 | **25.0562** |

```text
slot-centered pooled PR       = 7.7881
between-slot variance fraction = 0.4533
```

### Action PR

| Slot | PR |
|---|---:|
| C0 | **3.9559** |
| C1 | **5.3232** |
| C2 | **2.0628** |
| C3 | **23.7392** |

```text
slot-centered pooled PR       = 7.4423
between-slot variance fraction = 0.4202
```

### delta_q PR

| Slot | PR |
|---|---:|
| C0 | **3.1600** |
| C1 | **4.4207** |
| C2 | **3.0448** |
| C3 | **19.0651** |

```text
slot-centered pooled PR       = 12.2044
between-slot variance fraction = 0.3478
```

This is qualitatively different from OLD.

The central result is:

> **C0/C1/C2 genuinely lose much of their cross-input variation, while C3 remains high-rank.**

This is not just an artifact produced by flattening all slots together.

---

# 6. Matched cross-checkpoint geometry confirms the same result

At t0 all 160 samples are shared.

Mean per-slot PR on the matched cohort:

| Feature | OLD mean | STRONG mean | OLD min/max | STRONG min/max |
|---|---:|---:|---:|---:|
| Proposal | 19.768 | 9.936 | 19.741 / 19.792 | **2.642 / 25.056** |
| Action | 19.283 | 8.770 | 19.256 / 19.308 | **2.063 / 23.739** |
| delta_q | 15.359 | 7.423 | 15.306 / 15.400 | **3.045 / 19.065** |

The enormous STRONG min/max spread demonstrates candidate-slot specialization directly.

---

# 7. The specialization evolves across recurrence

Matched OLD/STRONG intersections:

## t1 — 154 shared samples

| Feature | OLD mean per-slot PR | STRONG mean per-slot PR | STRONG min/max |
|---|---:|---:|---:|
| Proposal | 17.560 | 15.648 | 10.874 / 23.118 |
| Action | 17.122 | 9.136 | 3.912 / 21.992 |
| delta_q | 11.171 | 5.878 | **1.420 / 10.297** |

The proposer partially recovers cross-input richness at t1, but Action and delta_q remain strongly concentrated.

## t2 — 145 shared samples

| Feature | OLD mean per-slot PR | STRONG mean per-slot PR | STRONG min/max |
|---|---:|---:|---:|
| Proposal | 17.271 | 16.280 | 10.866 / 24.433 |
| Action | 17.269 | 14.424 | 9.186 / 22.639 |
| delta_q | 12.624 | **15.853** | 12.442 / 18.346 |

By t2, STRONG recovers substantial transition variation.

Therefore the failure is not a permanently rank-one transition system. It is more accurately:

> **very strong early slot specialization plus transient recurrent concentration, followed by partial recovery.**

---

# 8. Matched temporal geometry: transient recurrent concentration is real

The new temporal diagnostic compares t0 and later states on the **same survivor IDs within each checkpoint**.

## OLD

### t0 → t1

```text
matched samples: 158
baseline current-global PR: 30.1093
t1 current-global PR:       22.4639
rank drop:                  25.39%
pairwise cosine change:     -0.0207
```

### t0 → t2

```text
matched samples: 149
baseline current-global PR: 29.1753
t2 current-global PR:       22.1543
rank drop:                  24.07%
pairwise cosine change:     -0.1008
```

OLD does not cross the conservative automatic recurrent-collapse threshold.

---

## STRONG

### t0 → t1

```text
matched samples: 155
baseline current-global PR: 30.1628
t1 current-global PR:       10.7280
rank drop:                  64.43%
pairwise cosine change:     -0.0458
```

### t0 → t2

```text
matched samples: 154
baseline current-global PR: 30.0395
t2 current-global PR:       20.7975
rank drop:                  30.77%
pairwise cosine change:     +0.00637
```

The important shape is:

```text
t0
 ↓
very large concentration at t1
 ↓
substantial recovery by t2
```

Thus:

> **STRONG exhibits a genuine transient global-rank concentration after the first transition.**

The absence of a large cosine increase at t1 means this should not be simplified into “all embeddings collapse into the same cosine direction.”

---

# 9. Trajectory-level rank does NOT explain retrieval quality by itself

Retrieval-query participation ratio:

| Metric | OLD | STRONG |
|---|---:|---:|
| Initial query PR | 28.8831 | 28.8831 |
| Terminal query PR | 21.4621 | **22.9254** |

Patch-state pooled PR:

| Metric | OLD | STRONG |
|---|---:|---:|
| V0 pooled PR | 12.3001 | 12.3001 |
| Terminal state pooled PR | 10.5031 | **8.2418** |

STRONG has **higher terminal query PR** than OLD but **worse official Recall**.

Therefore:

> Increasing representation rank/diversity alone is not a valid optimization target.

The extra directions must be retrieval-useful.

The terminal pooled patch-state drop in STRONG is real enough to flag:

```text
PATCH_STATE_RANK_DROP
V0_pooled 12.30 -> terminal_state_pooled 8.24
drop ~33%
```

but mean-pooled patch state is only one statistic. It is not sufficient evidence that the full patch-token state has globally collapsed.

---

# 10. Candidate quality: OLD and STRONG are fundamentally different

## 10.1 Mean teacher utility per candidate slot

### OLD

| Slot | Mean utility |
|---|---:|
| C0 | +0.3003 |
| C1 | +0.3005 |
| C2 | +0.3001 |
| C3 | +0.3008 |

All four candidates are similarly useful on average.

### STRONG

| Slot | Mean utility |
|---|---:|
| C0 | **-0.0418** |
| C1 | **-0.0250** |
| C2 | **-0.0075** |
| C3 | **+0.0787** |

This is direct evidence of **candidate-quality specialization**.

Three slots are harmful on average. Only C3 has positive mean teacher utility.

This matches the geometry exactly:

```text
C0/C1/C2: low input variation + weak/harmful utility
C3:       high input variation + positive retrieval utility
```

---

# 11. Oracle occupancy proves C3 specialization is upstream

Oracle occupancy conditional on oracle execution:

### OLD

```text
C0 = 24.23%
C1 = 25.26%
C2 = 22.19%
C3 = 28.32%
```

Nearly balanced.

### STRONG

```text
C0 =  7.04%
C1 = 10.80%
C2 =  6.81%
C3 = 75.35%
```

Therefore C3 dominance is not merely a ScoreNet positional bias.

The target-privileged oracle itself says:

> **C3 is genuinely the best available candidate most of the time.**

---

# 12. Model selection occupancy follows the upstream asymmetry

Selected slot occupancy conditional on execution:

### OLD

```text
C0 = 27.71%
C1 = 20.72%
C2 = 18.55%
C3 = 33.01%
```

### STRONG

```text
C0 =  3.24%
C1 =  9.94%
C2 =  9.94%
C3 = 76.89%
```

The STRONG selector selecting C3 in ~77% of edits is therefore mostly a consequence of the candidate generator producing one dominant useful slot.

---

# 13. Candidate-set health collapses under STRONG

## OLD

```text
mean useful candidate count = 2.533 / 4
DPP-valid row rate          = 63.38%
positive candidate fraction = 83.35%
mean positive candidates    = 3.334 / 4
mean harmful candidates     = 0.666 / 4
```

## STRONG

```text
mean useful candidate count = 1.017 / 4
DPP-valid row rate          = 20.26%
positive candidate fraction = 44.40%
mean positive candidates    = 1.776 / 4
mean harmful candidates     = 2.224 / 4
```

Relative change:

```text
useful candidate count drops by ~59.9%
positive candidate fraction drops by ~38.95 percentage points
```

This is the clearest direct manifestation of:

> **candidate-quality collapse / useful-slot specialization.**

A diversity mechanism cannot choose among multiple good alternatives if the system usually produces only one useful alternative.

---

# 14. DPP interpretation

The STRONG checkpoint has:

```text
lambda_dpp = 0.6
```

versus:

```text
OLD lambda_dpp = 0.01
```

and candidate geometry is far more slot-separated.

However:

1. `lambda_c`, `lambda_bind`, and `lambda_rel` were also increased drastically;
2. the current experiment is not a single-variable DPP ablation;
3. DPP can reward diversity without guaranteeing retrieval usefulness;
4. DPP-valid rows become rare when only ~1 candidate is useful.

The correct conclusion is:

> **The STRONG auxiliary regime produced diversity/specialization without sufficient per-candidate quality.**

The present data do **not** prove:

> “DPP alone caused the failure.”

That requires controlled auxiliary ablations later.

---

# 15. Caption sensitivity: feature change exists, semantic utility mostly does not

This section requires careful wording.

## 15.1 The text pathway is not numerically dead

Action features change when captions are shuffled.

### OLD

```text
action cosine(correct, shuffled) mean = 0.4651
action norm difference mean           = 8.7140
```

### STRONG

```text
action cosine(correct, shuffled) mean = 0.5982
action norm difference mean           = 6.1793
```

So caption changes still alter the network state/action.

Therefore the claim:

> “STRONG completely ignores text”

would be too strong.

---

## 15.2 But correct text gives almost no candidate-level utility advantage in STRONG

### OLD

```text
best candidate utility:
correct - shuffled = +0.8804

mean candidate utility:
correct - shuffled = +0.8921

STOP-aware oracle-policy utility:
correct - shuffled = +0.4517
```

### STRONG

```text
best candidate utility:
correct - shuffled = +0.0543

mean candidate utility:
correct - shuffled = -0.0260

STOP-aware oracle-policy utility:
correct - shuffled = +0.0559
```

This is a major result.

Interpretation:

> In STRONG, text still changes representations, but **the correct semantic caption barely makes the candidate set better than a shuffled caption at the candidate/oracle-utility level**.

This is much stronger evidence of a semantic-routing problem than simply observing feature movement.

---

## 15.3 Do not overstate the caption result

STRONG still reports:

```text
selected utility correct - shuffled = +0.4700
terminal retrieval correct advantage = +1.5100
```

so text information is not globally absent.

The safest interpretation is:

> **The correct caption has lost most of its advantage in producing intrinsically good sibling candidates, even though later selection/trajectory behavior still retains some caption dependence.**

This points toward the text → Proposal → Action/Executor candidate-generation path rather than a total loss of all text information.

---

# 16. Candidate retrieval behavior confirms the upstream degradation

Aggregate diagnostic candidate behavior:

## OLD

```text
parent retrieval loss mean    = 0.4936
candidate retrieval loss mean = 0.1932

parent margin mean             = 0.1541
candidate margin mean          = 0.2211
```

OLD candidates substantially improve retrieval loss and margin on average.

## STRONG

```text
parent retrieval loss mean    = 0.6748
candidate retrieval loss mean = 0.6737

parent margin mean             = 0.1000
candidate margin mean          = 0.1038
```

STRONG candidate generation is nearly neutral on average.

This is consistent with the slot-wise utilities:

- many weak/harmful candidates;
- one useful specialist;
- low average candidate-set headroom.

---

# 17. First transition remains especially suspicious

Positive-similarity trajectory from geometry:

## OLD

```text
t0 current   0.5749 -> committed 0.5071
t1 current   0.5071 -> committed 0.5608
t2 current   0.5608 -> committed 0.5666
terminal                          0.5768
```

## STRONG

```text
t0 current   0.5749 -> committed 0.4576
t1 current   0.4576 -> committed 0.4914
t2 current   0.4914 -> committed 0.5651
terminal                          0.5648
```

STRONG suffers a much larger first-step positive-similarity drop.

However, positive similarity alone is not a retrieval metric because hard negatives can move too.

Therefore this is supporting evidence only; the stronger evidence is the teacher utility / candidate loss / margin diagnostics above.

---

# 18. Selector diagnosis

## 18.1 Exact action agreement can be misleading

### OLD

```text
exact oracle accuracy = 31.26%
score-teacher Pearson = 0.6678
```

### STRONG

```text
exact oracle accuracy = 72.49%
score-teacher Pearson = 0.6521
```

At first glance STRONG looks like the better selector.

But OLD candidates have almost identical utility, so selecting a different sibling from the oracle often costs almost nothing.

That is why oracle regret is the more informative metric.

---

# 19. Oracle headroom and regret

| Metric | OLD | STRONG |
|---|---:|---:|
| Selected utility | 0.2995 | 0.2787 |
| Oracle utility | 0.3116 | **0.3320** |
| Regret | **0.0122** | **0.0533** |

STRONG regret is about **4.38× OLD**.

This means:

- STRONG occasionally contains a much better action than the one actually executed;
- selection/STOP calibration matters more when sibling quality is highly asymmetric.

The selector is therefore a **secondary bottleneck**, even though it is not the source of the C3 specialization.

---

# 20. STOP is strongly under-used in STRONG

## OLD

```text
execute rate        = 88.87%
oracle execute rate = 83.94%

model STOP rate     = 11.13%
oracle STOP rate    = 16.06%

STOP precision      = 38.46%
STOP recall         = 26.67%
```

## STRONG

```text
execute rate        = 98.72%
oracle execute rate = 90.83%

model STOP rate     = 1.28%
oracle STOP rate    = 9.17%

STOP precision      = 16.67%
STOP recall         = 2.33%
```

STRONG essentially almost never stops.

STOP recall falls from:

```text
26.67% -> 2.33%
```

or to only about **8.7% of the OLD recall level**.

This is consistent with the earlier concern that pairwise ranking and gain regression may conflict with the fixed STOP anchor at score `0`.

---

# 21. Harmful execution is materially worse

| Metric | OLD | STRONG |
|---|---:|---:|
| Harmful executions / executions | 13.25% | **22.03%** |
| Harmful executions / decisions | 11.78% | **21.75%** |

STRONG adds approximately **+8.78 percentage points** harmful execution among executed actions.

This is not explained by candidate generation alone because the oracle can still achieve higher utility than the selected policy.

Therefore:

> ScoreNet/STOP calibration is a real contributor to the final performance loss.

---

# 22. Timestep policy behavior

## OLD

```text
t0 execute: 98.75%   oracle: 86.25%
t1 execute: 94.30%   oracle: 93.04%
t2 execute: 72.48%   oracle: 71.81%
```

OLD learns to stop substantially more by t2.

## STRONG

```text
t0 execute: 96.88%   oracle: 76.88%
t1 execute: 99.35%   oracle: 97.42%
t2 execute: 100.0%   oracle: 98.70%
```

STRONG keeps executing almost universally.

At t0, the policy also executes far more often than the oracle wants.

This makes the early bad transition especially costly.

---

# 23. Terminal retrieval quality

Terminal cohort:

| Metric | OLD | STRONG |
|---|---:|---:|
| Initial retrieval loss | 0.96895 | 0.96895 |
| Terminal retrieval loss | **0.09485** | **0.15201** |
| Initial→terminal improvement | **0.87409** | 0.81693 |
| Terminal positive similarity | **0.57678** | 0.56485 |
| Terminal hardest-negative similarity | **0.32113** | 0.34372 |
| Terminal margin | **0.25565** | 0.22113 |

STRONG ends with a worse retrieval margin and higher terminal retrieval loss.

This aligns with the official Recall regression.

---

# 24. Root-cause hierarchy

The current evidence supports the following ordering.

## Primary: candidate-generation / candidate-quality failure

Evidence:

- three STRONG slots have negative mean utility;
- only C3 has positive mean utility;
- oracle selects C3 ~75%;
- mean useful candidates falls to ~1.02/4;
- positive candidate fraction falls to ~44.4%;
- mean harmful candidates rises to ~2.22/4;
- correct caption barely improves best/oracle candidate utility;
- per-slot geometry shows C0/C1/C2 losing substantial cross-input variation.

This is the main upstream failure.

---

## Secondary: ScoreNet / STOP calibration failure

Evidence:

- STRONG oracle utility is 0.332 but selected utility is 0.279;
- regret is ~4.38× OLD;
- harmful execution is ~22%;
- STOP recall is only ~2.33%;
- model STOP rate 1.28% vs oracle STOP 9.17%.

ScoreNet does not create the C3 specialization, but it fails to exploit the available oracle headroom and fails to STOP reliably.

---

## Associated phenomenon: transient recurrent rank concentration

Evidence:

- matched t0→t1 current-global PR drops ~64.4% in STRONG;
- partial recovery occurs by t2;
- recurrent-rank and state-drift warnings trigger.

This is probably related to the bad early transition equilibrium, but it is not by itself a complete causal explanation.

---

# 25. What has now been ruled out

## 25.1 “The low pooled rank is only a flattening artifact”

**Ruled out as a complete explanation.**

Per-slot analysis shows real low cross-input variation in C0/C1/C2.

---

## 25.2 “All candidate slots collapse equally”

**Ruled out.**

C3 remains high-rank and retrieval-useful.

---

## 25.3 “ScoreNet randomly likes C3”

**Ruled out.**

The target-privileged oracle itself strongly prefers C3.

---

## 25.4 “Higher representation rank automatically means better retrieval”

**Ruled out.**

STRONG terminal query PR exceeds OLD while official Recall is lower.

---

## 25.5 “STRONG completely ignores captions”

**Not supported.**

Caption shuffling changes features and terminal behavior.

The supported claim is narrower:

> correct caption semantics provide very little intrinsic candidate/oracle utility advantage under STRONG.

---

# 26. What is NOT yet proven

The current diagnostics do **not** prove:

1. DPP alone caused the failure.
2. `lambda_bind` alone caused the failure.
3. `lambda_rel` alone caused the failure.
4. `lambda_c` alone caused the failure.
5. which exact module first loses semantic conditioning: Proposal vs Grounding vs Fusion vs Executor.
6. that full patch-token state has globally collapsed.
7. that an anti-collapse/rank regularizer would improve Recall.
8. that forcing 25% occupancy per candidate slot is desirable.
9. that increasing STOP frequency by itself will improve benchmark.
10. that the correct solution requires unfreezing the vision encoder.
11. that RL/ST/Gumbel is currently needed.
12. that extending K or T will help.

These remain intervention questions.

---

# 27. Scientific interpretation of OLD vs STRONG

The two checkpoints represent two different bad equilibria.

## OLD equilibrium

```text
four highly redundant siblings
+
all four have similarly good utility
+
low selector exact-oracle agreement
BUT
small regret because candidate choice barely matters
```

This is close to **functional sibling collapse**, but the candidates are generally useful.

---

## STRONG equilibrium

```text
siblings become functionally distinct
+
strong slot identities emerge
+
C3 carries most useful retrieval behavior
+
C0/C1/C2 are frequently harmful
+
selector now has a difficult high-stakes decision
+
STOP is badly calibrated
```

So STRONG appears to have solved the wrong problem:

> it increased diversity/specialization without preserving candidate-set quality.

---

# 28. Most likely training mechanism

A plausible mechanism consistent with all observations is:

```text
fixed learned candidate query identities
        +
strong auxiliary/diversity pressure
        +
hard selected rollout
        +
no direct live task-quality gradient to every unselected candidate
        ↓
slots can specialize
        ↓
one slot becomes retrieval specialist
other slots satisfy auxiliary/diversity structure without being useful
        ↓
C3 becomes dominant
```

This mechanism is consistent with the architecture and diagnostics, but the exact contribution of each auxiliary still requires controlled interventions.

---

# 29. Decision according to the original action plan

The original plan was:

```text
1. Fix config/checkpoint observability and diagnostics
2. Re-evaluate OLD/STRONG with per-slot geometry
3. ScoreNet gain-only rescue
4. Direct candidate-quality loss on unselected candidates
5. Full-K DPP vs useful-subset DPP
6. Native image+text + K1/T1 baseline
7. Split LR + TRAIN negative bank
8. Only then extend K/T/history
```

Steps 1–2 are now sufficiently informative to move forward.

---

# 30. Next experiment: ScoreNet gain-only rescue

## Purpose

Test the specific hypothesis:

> Pairwise ranking pressure conflicts with gain regression / STOP anchor and contributes to over-execution and regret.

This intervention should leave the candidate generator fixed if possible.

The clean test is a **scorer-only refit**:

```text
freeze candidate generator / executor / text-side generator behavior
train ScoreNet only
lambda_pair = 0
retain gain regression
```

Then re-run live rollout.

Required measurements:

- selected utility;
- oracle utility;
- regret;
- STOP precision;
- STOP recall;
- model/oracle STOP rates;
- harmful execution;
- official FashionIQ Recall;
- score–teacher correlation;
- score bias around utility zero;
- per-timestep execute/STOP rates.

Expected interpretation:

### If STOP/regret improve materially but Recall remains limited

Then ScoreNet was a secondary bottleneck, while candidate quality remains the main issue.

### If gain-only does not improve STOP/regret

Then pair loss is not the dominant ScoreNet problem; inspect scorer features/calibration and move on.

---

# 31. Next upstream experiment after ScoreNet: direct candidate-quality supervision

The evidence already strongly motivates the next upstream intervention.

The goal is:

> give live task-quality gradient to unselected sibling candidates instead of merely asking the selector to score them.

First candidate:

```text
L_safe = mean max(0, candidate_retrieval_loss - stopgrad(parent_retrieval_loss))
```

Desired behavior:

- harmful candidates receive gradient;
- already non-harmful candidates are not forced to become identical;
- candidate effects must not collapse to zero;
- best-of-K/oracle utility must increase;
- useful-candidate count must increase;
- final held-out retrieval with the real selector must improve.

Important:

> Do not enable L_safe, candidate CE bootstrap, new DPP semantics, and scorer changes simultaneously in one first run.

---

# 32. Diversity intervention only after quality signal

After candidate-quality learning is verified:

compare:

```text
full-K DPP
vs
useful-subset DPP
```

Question:

> Does diversity among useful candidates improve without rewarding harmful siblings merely for being far apart?

Required monitoring:

- useful candidate count;
- DPP valid-row rate;
- utility distribution per slot;
- slot occupancy;
- per-slot PR;
- slot-centered PR;
- best-of-K utility;
- selected utility;
- Recall.

---

# 33. Baselines remain mandatory

Before increasing architecture complexity:

1. native image + native text baseline;
2. K=1 / T=1 editor control.

Interpretation:

```text
K1/T1 bad
→ basic text-conditioned edit path is still not good enough
→ do not expand recurrence

K1/T1 good but T2/T3 bad
→ recurrence / state drift / STOP / horizon issue

simple native sum/combiner > editor
→ current IAG-SRME complexity has not justified itself
```

---

# 34. What NOT to do next

Do not:

- unfreeze vision merely because STRONG is bad;
- add anti-collapse loss because PR is low;
- force equal 25% slot occupancy;
- increase DPP further;
- increase K to 8;
- increase T beyond current rollout;
- add RL;
- add straight-through/Gumbel;
- change ScoreNet + candidate loss + DPP + LR simultaneously;
- treat terminal PR as the optimization target;
- conclude DPP alone is guilty from OLD vs STRONG.

---

# 35. Compact evidence table

| Observation | OLD | STRONG | Diagnosis |
|---|---:|---:|---|
| Mean Recall | 30.824 | 28.245 | STRONG regresses |
| Mean useful candidates | 2.533 | 1.017 | Candidate-quality collapse |
| Positive candidate fraction | 83.35% | 44.40% | Many STRONG siblings harmful |
| Mean harmful candidates / 4 | 0.666 | 2.224 | Upstream quality failure |
| C3 mean utility | +0.301 | +0.079 | Only positive STRONG specialist |
| C0/C1/C2 mean utility | all ~+0.300 | all negative | Useful-slot specialization |
| Oracle C3 occupancy | 28.3% | 75.4% | C3 dominance is upstream |
| Selected C3 occupancy | 33.0% | 76.9% | Selector follows upstream bias |
| Selected utility | 0.299 | 0.279 | STRONG policy worse |
| Oracle utility | 0.312 | 0.332 | STRONG still has headroom |
| Regret | 0.012 | 0.053 | ScoreNet/STOP secondary failure |
| Harmful execution | 13.25% | 22.03% | Over-execution hurts |
| STOP recall | 26.67% | 2.33% | STRONG STOP badly calibrated |
| Correct−shuffle candidate utility | +0.892 | -0.026 | Correct semantic text no longer helps average siblings |
| Correct−shuffle best utility | +0.880 | +0.054 | Candidate-set semantic gain nearly disappears |
| t0 Action min/max per-slot PR | 19.26 / 19.31 | 2.06 / 23.74 | Real slot-specific rank asymmetry |
| t0→t1 matched global PR drop | 25.4% | 64.4% | STRONG transient concentration |
| Terminal query PR | 21.46 | 22.93 | Rank ≠ retrieval quality |
| Terminal margin | 0.256 | 0.221 | STRONG terminal retrieval worse |

---

# 36. Final diagnosis

The current evidence is sufficient to state:

> The STRONG auxiliary checkpoint does not fail because every representation uniformly collapses. Instead, the auxiliary regime drives the fixed candidate slots into a strongly asymmetric equilibrium. C0–C2 lose substantial cross-input variation and are often harmful, while C3 retains a rich input-dependent representation and becomes the dominant retrieval-quality specialist. This produces a candidate set with roughly one useful candidate instead of several. Correct captions still change internal features, but they provide almost no candidate-level or oracle-level utility advantage over shuffled captions, showing that semantic conditioning is no longer being converted reliably into useful sibling edits. The selector largely follows the true upstream C3 asymmetry, so its slot bias is not the primary cause. Nevertheless, ScoreNet/STOP remains a secondary bottleneck: oracle headroom is larger, regret is ~4.4× OLD, harmful execution rises to ~22%, and STOP recall falls to ~2.3%. The next clean intervention is therefore a gain-only scorer rescue, followed by direct live candidate-quality supervision, and only then a useful-subset DPP ablation.

---

# 37. Artifact set that should be preserved for reproducibility

Recommended versioned directory:

```text
doc/diagnostics/2026-09-08_v2-r0_old-vs-strong/
```

Recommended contents:

```text
CIR_V2_R0_TEXT_ONLY_FAILURE_DIAGNOSIS_2026-09-08.md

old_geometry/
  latent_geometry_report.json
  latent_geometry_report.md
  candidate_features.pt

strong_geometry/
  latent_geometry_report.json
  latent_geometry_report.md
  candidate_features.pt

old_selector/
  candidate_selector_diagnostic.json
  candidate_selector_diagnostic.md

strong_selector/
  candidate_selector_diagnostic.json
  candidate_selector_diagnostic.md

old_vs_strong_matched/
  matched_geometry_comparison.json
  matched_geometry_comparison.md

shared_train_160.json
```

`candidate_features.pt` can be omitted from Git if it is large or if repository policy avoids binary diagnostic tensors. The JSON/MD reports and manifest are the most important reproducibility artifacts.

---

# 38. Status checkpoint

Completed:

- [x] OLD/STRONG checkpoint identity recovered
- [x] same diagnostic manifest
- [x] same teacher batch grouping
- [x] per-slot geometry
- [x] slot-centered geometry
- [x] between-slot variance decomposition
- [x] matched temporal survivor geometry
- [x] matched OLD/STRONG same-timestep comparison
- [x] candidate utility by slot
- [x] selected/oracle occupancy
- [x] selected utility / oracle utility / regret
- [x] harmful execution
- [x] STOP precision/recall
- [x] caption shuffle utility controls
- [x] official FashionIQ evaluation
- [x] root-cause hierarchy established

Next:

- [ ] scorer-only gain-only refit
- [ ] live rollout after scorer refit
- [ ] direct candidate-quality gradient test
- [ ] L_safe intervention
- [ ] useful-subset DPP matched ablation
- [ ] native image+text baseline
- [ ] K1/T1 editor control
- [ ] split LR / TRAIN negative bank only after a better one-step configuration exists
- [ ] extend recurrence only after one-step quality is demonstrated