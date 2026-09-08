# CIR V2 R0 — OLD vs STRONG Slot-Specialization Failure Diagnosis

**Date:** 2026-09-09  
**Branch:** `exp/e2e-iag-srme-v2-r0`  
**Diagnostic git SHA recorded in reports:** `597af115b4e4c208fa86c07c71bc96f6ce9e6613`  
**Dataset:** FashionIQ validation  
**Cohort:** 6,016 samples, identical persistent manifest for OLD and STRONG  
**Primary timestep:** `t=0`  
**Teacher:** canonical identity-aware, false-negative-safe, matched in-batch marginal teacher utility  
**Bootstrap:** 1,000 teacher-batch cluster-bootstrap replicates, 95% CI  
**Purpose:** distinguish candidate collapse, healthy specialization, selector collapse, gradient starvation, and auxiliary-driven symmetry breaking/interference.

---

## 1. Executive conclusion

The current evidence does **not** support the simple diagnosis that STRONG has collapsed to one effective candidate, nor does it support the claim that the selector itself simply collapses to C3.

The strongest evidence supports a more specific mechanism:

> **OLD contains four high-quality but highly redundant candidates. STRONG breaks this symmetry and produces real functional specialization, but does so at the cost of severe candidate-quality and representation degradation. C3 becomes a high-impact dominant branch, while C0/C1/C2 retain real but weaker complementary niches. Current task-gradient flow and Concept-MIL responsibility then reinforce this asymmetry.**

A compact causal hypothesis is:

```text
OLD
4 high-quality, near-redundant candidates
        |
        | stronger auxiliary regime / optimization pressure
        v
symmetry breaking + differentiation
        |
        +--> C3 becomes high-impact dominant candidate
        |
        +--> C0/C1/C2 become weaker niche candidates
        |
        v
MIL responsibility shifts toward C3
+
hard recurrent task gradient shifts toward C3
        |
        v
winner-take-most reinforcement loop
```

This is currently better described as:

> **auxiliary-induced specialization/symmetry breaking + winner-take-most gradient feedback**

rather than pure architectural bias, pure selector collapse, or pure gradient starvation from initialization.

The failure is also **not purely a routing failure**, because even the STRONG oracle candidate set is worse than the OLD oracle candidate set.

---

## 2. Checkpoints

### OLD

```text
checkpoint: outputs/r0_ncls_text/best.pt
epoch:      15
Mean Recall: 30.824426313241325
```

### STRONG

```text
checkpoint: outputs/r0_ncls_text_strong_aux/best.pt
epoch:      18
Mean Recall: 28.244677186012268
```

Observed benchmark delta:

```text
STRONG - OLD = -2.579749127229057 Mean Recall
```

This drop is consistent with the diagnostic evidence that candidate quality itself deteriorates under STRONG.

---

## 3. Experimental comparability

Both reports used:

```text
split:                 val
processed samples:     6016
valid t0 teacher rows: 6016
sample fingerprint:
97994df35e76d7166c5dff66d610ba782067b0ce94c37086012f54ce67718c40

diagnostic batch size: 8
teacher clusters:      752
bootstrap samples:     1000
bootstrap confidence:  0.95
```

This is important because the teacher utility depends on the in-batch negative pool. The persistent manifest fixes both sample order and teacher-batch grouping.

Therefore the main functional comparison is substantially stronger than the earlier 160-row probes.

---

# 4. Primary functional comparison

| Metric | OLD | STRONG | Interpretation |
|---|---:|---:|---|
| Checkpoint Mean Recall | 30.8244 | 28.2447 | STRONG is worse |
| All-K oracle value | 0.491764 | 0.453040 | candidate-set ceiling decreases |
| C3-only oracle value | 0.486045 | 0.347772 | C3 becomes much less universally safe/useful |
| All-K − C3 | 0.005719 | 0.105268 | complementarity increases dramatically |
| Relative complementarity | 1.163% | 23.236% | non-C3 slots become genuinely necessary |
| Effective functional K | ~4.000 | 2.701 | functional credit becomes concentrated |
| Oracle execute fraction | 76.08% | 78.96% | similar oracle need to execute |
| Live execute fraction | 97.82% | 97.26% | STOP is heavily underused in both |
| STOP gap | ~21.74 pp | ~18.30 pp | common selector/STOP problem |

STRONG all-K value also has a 95% cluster-bootstrap CI:

```text
0.433372 .. 0.471179
```

and STRONG all-K minus C3 has:

```text
estimate: 0.105268
95% CI:   0.100293 .. 0.110017
```

Relative complementarity:

```text
estimate: 23.236%
95% CI:   21.927% .. 24.476%
```

This is decisive evidence that C0/C1/C2 are **not dead candidates** in STRONG.

---

# 5. OLD is not healthy specialization — it is high-quality redundancy

OLD per-slot utility:

| Slot | Mean utility | Positive | Harmful | Oracle occupancy | Shapley |
|---|---:|---:|---:|---:|---:|
| C0 | 0.404659 | 75.399% | 24.601% | 22.482% | 0.122873 |
| C1 | 0.404772 | 75.382% | 24.618% | 24.962% | 0.122905 |
| C2 | 0.405597 | 75.515% | 24.485% | 30.151% | 0.123161 |
| C3 | 0.405082 | 75.399% | 24.584% | 22.406% | 0.122825 |

The four Shapley shares are essentially exactly equal:

```text
C0 24.986%
C1 24.993%
C2 25.045%
C3 24.976%
```

with extremely tight bootstrap intervals around 25%.

However, this must **not** be interpreted as four semantically distinct specialists.

The decisive redundancy statistic is:

```text
OLD all-K value:  0.491764
OLD C3-only value: 0.486045
difference:         0.005719
relative gain:      1.163%
```

Thus one fixed slot already recovers almost the entire all-K oracle value.

This means:

> `K_eff ≈ 4` in OLD mostly reflects equal Shapley credit among near-substitutable candidates, not four meaningfully complementary experts.

The geometry supports the same interpretation.

### OLD proposal participation ratio

```text
              all      utility>0    oracle winner
C0           24.896      23.672        23.780
C1           24.976      23.731        22.568
C2           24.915      23.690        23.330
C3           24.983      23.754        23.305
```

All four proposal streams have almost identical cross-sample rank/variation.

Concept-MIL is also nearly perfectly symmetric:

```text
mean responsibility:
C0 0.24986
C1 0.24951
C2 0.25033
C3 0.25030

mean entropy = 1.38622
maximum log(4) ≈ 1.38629
```

Therefore the OLD state is best characterized as:

> **high-quality, high-rank, highly redundant candidate generation.**

---

# 6. STRONG creates real specialization, but candidate quality collapses

STRONG per-slot utility:

| Slot | Mean utility | Positive | Harmful | Oracle occupancy | Shapley |
|---|---:|---:|---:|---:|---:|
| C0 | -0.077250 | 32.929% | 67.071% | 17.853% | 0.045129 |
| C1 | -0.102786 | 41.489% | 58.511% | 25.242% | 0.071594 |
| C2 | -0.009679 | 32.430% | 67.570% | 19.453% | 0.035699 |
| C3 | -0.384116 | 35.588% | 64.412% | 37.453% | 0.300617 |

STRONG normalized Shapley shares:

```text
C0  9.961%   95% CI ≈  9.36% .. 10.60%
C1 15.803%   95% CI ≈ 14.97% .. 16.62%
C2  7.880%   95% CI ≈  7.29% ..  8.52%
C3 66.356%   95% CI ≈ 64.97% .. 67.73%
```

This is genuine asymmetry.

But crucially, it is **not** pure one-slot collapse:

```text
STRONG all-K - C3 = 0.105268
relative complementarity = 23.236%
K_eff = 2.701
95% CI K_eff ≈ 2.636 .. 2.767
```

Thus C0/C1/C2 retain statistically robust functional contribution.

The problem is that specialization comes with a severe safety/quality cost:

```text
OLD positive candidate fraction:    ~75% for every slot
STRONG positive candidate fraction: ~32% .. 41%

OLD harmful candidate fraction:     ~24.5%
STRONG harmful candidate fraction:  ~58.5% .. 67.6%
```

The all-K oracle itself drops:

```text
0.491764 -> 0.453040
```

Therefore even a perfect selector cannot fully recover OLD performance.

This is direct evidence that:

> **candidate proposal/execution quality deteriorated, not merely ScoreNet selection quality.**

This matches the previous failed ScoreNet-only refit rescue.

---

# 7. Geometry: STRONG differentiates slots by destroying rank in C0/C1/C2

### STRONG proposal participation ratio

```text
              all      utility>0    oracle winner
C0            4.640       6.937         6.963
C1            8.278      10.148        11.436
C2            2.863       3.482         4.529
C3           27.938      27.728        27.851
```

Relative to OLD, the change is dramatic:

```text
OLD:    all four slots PR ≈ 25
STRONG: C0≈4.6, C1≈8.3, C2≈2.9, C3≈27.9
```

This means C0/C1/C2 lose most of their cross-input variation, while C3 retains a rich high-rank response.

However, conditional PR rises for C0/C1/C2 on their useful/oracle subsets.

Examples:

```text
C0 global 4.64 -> oracle 6.96
C1 global 8.28 -> oracle 11.44
C2 global 2.86 -> oracle 4.53
```

This supports:

> C0/C1/C2 are weak niche candidates rather than completely dead constant slots.

But the absolute conditional PR remains far below C3, so this is not a balanced four-specialist system.

---

# 8. Semantic specialization exists in STRONG

The semantic parser is explanatory rather than causal, but several conditional niches are visible.

Examples from STRONG:

### C2-like formal/button niche

`button up`:

```text
conditional Shapley:
C0 0.086
C1 0.071
C2 0.128
C3 0.054

C2 positive unique-advantage rate ≈ 34.8%
```

`more formal`:

```text
conditional Shapley:
C0 0.062
C1 0.032
C2 0.114
C3 0.020

C2 positive unique-advantage rate ≈ 31.3%
```

### C3 casual / T-shirt / print niche

`t shirt`:

```text
conditional Shapley:
C0 0.027
C1 0.079
C2 0.006
C3 1.048

C3 positive unique-advantage rate ≈ 60%
```

`tee shirt`:

```text
conditional Shapley:
C0 0.022
C1 0.079
C2 0.001
C3 1.504
```

`printed`:

```text
C3 conditional Shapley ≈ 0.735
C3 positive unique-advantage rate ≈ 62.7%
```

`more casual`:

```text
conditional Shapley:
C0 0.088
C1 0.110
C2 0.035
C3 0.791
```

Thus STRONG does produce some real conditional roles.

But the roles are extremely imbalanced: C3 becomes a high-impact branch, while other slots retain narrower and weaker niches.

---

# 9. Selector behavior: not C3 collapse on full VAL

Earlier small-cohort diagnostics suggested very high C3 selected occupancy. Full VAL changes this interpretation.

### STRONG

```text
              live selected    oracle
C0               16.89%         17.85%
C1               25.57%         25.24%
C2               29.38%         19.45%
C3               28.17%         37.45%
```

Therefore STRONG selector behavior on full VAL is:

```text
C2 over-selected by roughly +9.9 percentage points
C3 under-selected by roughly -9.3 percentage points
```

So:

> **there is no full-VAL t0 selector collapse into C3.**

The old ~70–75% C3 occupancy finding must be treated as cohort/timestep-specific, not as the global description of the checkpoint.

### OLD

```text
              live selected    oracle
C0               20.73%         22.48%
C1               25.01%         24.96%
C2               19.75%         30.15%
C3               34.51%         22.41%
```

OLD also has routing mismatch, especially over-selecting C3 and under-selecting C2.

Thus ScoreNet/routing calibration is a separate problem that already exists in OLD.

However, it cannot explain the STRONG degradation by itself because STRONG's oracle candidate-set ceiling is already lower than OLD.

---

# 10. STOP underuse is a separate failure shared by OLD and STRONG

### OLD

```text
oracle execute = 76.08%
live execute   = 97.82%

oracle STOP ≈ 23.92%
live STOP   ≈  2.18%
gap         ≈ 21.74 percentage points
```

### STRONG

```text
oracle execute = 78.96%
live execute   = 97.26%

oracle STOP ≈ 21.04%
live STOP   ≈  2.74%
gap         ≈ 18.30 percentage points
```

Therefore STOP underuse is **not created by STRONG**.

It is a shared selector-policy failure:

> The learned policy executes in many rows where the marginal teacher says every candidate is non-beneficial and STOP should win.

This should eventually be fixed, but it is orthogonal to the STRONG-specific collapse in candidate quality.

---

# 11. Current task-gradient exposure becomes strongly C3-dominated in STRONG

The gradient diagnostic uses the full configured recurrent loss component with respect to the live `t=0` proposal tensor.

It is **not** a local-only `t=0` gradient and does **not** prove historical causality.

### Proposal-tensor terminal gradient energy

OLD:

```text
C0 33.66%
C1 40.42%
C2  7.51%
C3 18.41%
```

STRONG:

```text
C0  5.08%
C1 14.23%
C2 16.71%
C3 63.98%
```

### Parameter-level proposal-query terminal gradient energy

OLD:

```text
C0 14.59%
C1 38.79%
C2 20.72%
C3 25.90%
```

STRONG:

```text
C0  2.49%
C1  1.28%
C2  2.72%
C3 93.52%
```

This is very strong evidence that, **at the final STRONG checkpoint**, task-specific learning signal is concentrated on C3.

The query-row result is especially extreme.

The correct interpretation is:

> STRONG currently exhibits winner-take-most task-gradient exposure that strongly favors C3.

The incorrect interpretation would be:

> Historical gradient starvation is already proven to have caused C3 dominance.

Static gradients do not establish the direction of causality.

---

# 12. Concept-MIL also shifts from symmetric to C3-biased

### OLD

```text
mean responsibility:
C0 24.99%
C1 24.95%
C2 25.03%
C3 25.03%

top-responsibility occupancy:
C0 25.77%
C1 19.85%
C2 29.84%
C3 24.53%

mean entropy: 1.38622
```

This is essentially maximum-entropy candidate responsibility.

### STRONG

```text
mean responsibility:
C0 22.54%
C1 17.61%
C2 20.56%
C3 39.29%

top-responsibility occupancy:
C0 16.87%
C1  0.86%
C2 18.07%
C3 64.19%

mean entropy: 1.18061
```

This is not total Concept-MIL collapse, but it is a strong winner-take-most shift.

It supports the following reinforcing mechanism:

```text
C3 develops an advantage
        ->
C3 obtains more MIL responsibility
        ->
Concept objective gives C3 stronger semantic responsibility
        ->
C3 advantage is reinforced
```

This is evidence of current reinforcement pressure, not proof that Concept-MIL alone initiated the asymmetry.

---

# 13. DPP / auxiliary gradients

STRONG proposal-tensor DPP gradient energy is highly concentrated on C2:

```text
C0 10.31%
C1  2.16%
C2 82.16%
C3  5.38%
```

OLD DPP gradient energy is instead:

```text
C0 65.27%
C1  9.19%
C2 12.88%
C3 12.67%
```

This confirms that auxiliary pressure has become slot-specific after symmetry breaking.

The automatic report also raises:

```text
NEGATIVE_TASK_AUX_ALIGNMENT
```

However, this flag should not yet be treated as strong evidence of destructive auxiliary interference because:

1. gradient attribution currently uses only 4 batches / 32 samples;
2. some cosine comparisons have very small valid counts;
3. some auxiliary-to-terminal norm ratios are tiny;
4. static gradient disagreement does not show which signal dominated earlier training.

Therefore:

> Auxiliary interference remains plausible, but the current strongest evidence is **symmetry breaking + winner reinforcement**, not yet a proven destructive-gradient mechanism.

---

# 14. Important audit caveat: OLD objective metadata inconsistency

The newly generated OLD report records:

```text
lambda_c    = 0.6
lambda_bind = 1.0
lambda_rel  = 0.6
lambda_dpp  = 0.6
```

which is the same high-weight regime reported for STRONG.

This conflicts with the historical OLD-vs-STRONG experiment description used in the earlier audit, where OLD was understood to use approximately:

```text
lambda_c    = 0.01
lambda_bind = 0.01
lambda_rel  = 0.001
lambda_dpp  = 0.01
```

This inconsistency must be resolved before making a strong historical causal statement such as:

> "Increasing the auxiliary coefficients from OLD to STRONG caused the failure."

### What remains valid despite this inconsistency

The following OLD-vs-STRONG comparisons remain valid because they depend primarily on the stored model behavior and canonical teacher:

- candidate utility;
- positive/harmful fractions;
- Shapley;
- coalition oracle;
- all-K vs C3;
- proposal/action/delta geometry;
- live selected occupancy;
- oracle occupancy;
- STOP behavior;
- benchmark checkpoint metric;
- terminal-gradient **distribution across slots** within each checkpoint.

### What needs caution

If the OLD objective metadata is stale/reconstructed incorrectly, then these quantities may not represent the historical OLD training objective exactly:

- coefficient-scaled auxiliary/task gradient norm ratios;
- claims about the historical magnitude of Concept/Bind/DPP optimization pressure;
- any causal statement based specifically on the recorded OLD coefficients.

Positive scalar loss coefficients do not change within-loss slot energy shares or cosine signs, but they do affect comparisons of magnitude across different loss components.

### Required audit

Before publication-quality causal claims, inspect:

```text
outputs/r0_ncls_text/best.pt metadata.objective_config
outputs/r0_ncls_text_strong_aux/best.pt metadata.objective_config
original Hydra config / .hydra/config.yaml for both runs
training logs or archived run configuration
```

If the OLD run truly used the smaller coefficients but the checkpoint metadata says otherwise, document why.

This does **not** block the functional diagnosis, but it blocks a definitive statement that coefficient magnitude alone caused the transition.

---

# 15. Hypotheses: current status

## H0 — Architectural candidate-index bias

**Status: low probability / previously audited against.**

Reason:

- candidate-specific parameters are mainly learned query rows;
- downstream Grounder, ActionFusion, Executor, ScoreNet are shared;
- no special C3 module/index embedding was found;
- deterministic max tie behavior would favor lower indices, not specifically C3.

No need to prioritize candidate-index architecture debugging now.

---

## H1 — Pure selector collapse into C3

**Status: rejected as global full-VAL diagnosis.**

STRONG live C3 selection is only 28.17%, while oracle C3 is 37.45%.

The selector actually over-selects C2 and under-selects C3 on full VAL t0.

---

## H2 — Effective candidate count collapsed to K≈1

**Status: rejected.**

Evidence:

```text
STRONG K_eff ≈ 2.70
all-K − C3 = 0.1053
relative complementarity ≈ 23.2%
CI lower bound strongly > 0
```

Non-C3 candidates provide real functional value.

---

## H3 — Healthy four-way specialization

**Status: rejected.**

STRONG specialization is too asymmetric and too costly:

- C3 receives ~66% of Shapley value;
- C0/C1/C2 proposal ranks collapse;
- 58–68% of candidates are harmful depending on slot;
- all-K oracle quality falls below OLD;
- benchmark Mean Recall falls.

This is not desirable specialization.

---

## H4 — Pure hard-selection gradient starvation is the root cause

**Status: plausible amplifier, not established root cause.**

Why it cannot be the entire explanation:

- OLD uses the same hard recurrent selection mechanism;
- OLD keeps high-rank, symmetric, high-quality candidates;
- STRONG alone develops extreme C3 task-gradient concentration.

Thus the hard route can reinforce an imbalance once it exists, but some additional training pressure is likely required to initiate the symmetry breaking.

---

## H5 — Auxiliary-induced symmetry breaking followed by winner-take-most feedback

**Status: leading hypothesis.**

Evidence:

1. OLD is near symmetric in utility, geometry, Shapley and Concept-MIL responsibility.
2. STRONG has strong functional and geometric asymmetry.
3. Concept-MIL top responsibility shifts to C3 ≈64%.
4. terminal task-gradient flow shifts to C3 ≈64% at proposal-tensor level and ≈94% at query-row level.
5. C0/C1/C2 retain niches, so this is differentiation rather than complete death.
6. overall candidate quality decreases substantially.

Caveat: the exact initiating auxiliary coefficient change must be verified because of the OLD metadata inconsistency.

---

# 16. Why ScoreNet-only repair was insufficient

A previous frozen-policy ScoreNet refit reduced retrieval performance:

```text
refit Mean Recall ≈ 25.46
STRONG Mean Recall ≈ 28.24
delta ≈ -2.79
```

The current functional diagnostic explains why.

Even the best possible STRONG candidate oracle has lower value than OLD:

```text
OLD all-K oracle    0.4918
STRONG all-K oracle 0.4530
```

Therefore routing/calibration alone cannot restore candidate representations that have already degraded.

ScoreNet remains a secondary issue, not the principal repair target.

---

# 17. Why L_safe is currently well-motivated

The current failure is not:

> "C3 is used too often, therefore force uniform 25/25/25/25 routing."

The actual problem is:

```text
STRONG:
functional differentiation/complementarity  ↑
candidate harmfulness                        ↑↑↑
candidate geometry for C0-C2                 ↓↓↓
overall oracle candidate quality             ↓
```

Therefore the desired intervention is:

> preserve useful specialization while preventing each candidate from becoming actively harmful.

The implemented safety objective:

```text
L_safe = E[max(0, retrieval_loss(candidate) - stop_gradient(retrieval_loss(parent)))]
```

gives every harmful candidate a direct corrective signal without forcing equal usage or eliminating specialization.

This makes `L_safe` a more targeted intervention than load balancing.

---

# 18. What should NOT be done next

Do not immediately:

- force uniform candidate occupancy;
- add strong load-balancing loss;
- randomize candidate selection;
- round-robin candidates;
- remove C3;
- retrain ScoreNet alone again;
- conclude that C3 dominance is architectural;
- conclude historical gradient starvation from the current static gradient alone;
- increase all diversity auxiliaries further.

These changes either attack the wrong failure or would confound the current mechanism.

---

# 19. Immediate next experiments

## Step A — Verify OLD objective provenance

Before causal claims about auxiliary coefficient changes:

```text
inspect OLD checkpoint metadata.objective_config
inspect OLD Hydra archived config
inspect training logs / run config
compare against STRONG
```

Goal:

```text
Was OLD actually trained with small auxiliary coefficients?
Or is the earlier OLD-vs-STRONG coefficient comparison incorrect?
```

This is a cheap but important audit.

---

## Step B — Increase gradient probe sample count

Current gradient statistics use only:

```text
gradient_batches = 4
gradient_batch_size = 8
N = 32
```

Repeat the read-only gradient probe on roughly 256–512 samples.

Suggested:

```text
gradient_batches = 32  -> N≈256
or
gradient_batches = 64  -> N≈512
gradient_batch_size = 8
```

Run this for both OLD and STRONG on the same manifest.

Questions:

1. Does STRONG C3 terminal-gradient concentration remain extreme?
2. Does query-row C3 energy remain close to winner-take-most?
3. Does DPP remain disproportionately focused on C2?
4. Are negative task/aux cosine fractions stable with larger valid counts?

If yes, the winner-feedback interpretation becomes much stronger.

---

## Step C — Controlled L_safe intervention

If the enlarged gradient probe supports the same pattern, compare:

```text
Control:
STRONG continuation / lambda_safe = 0

Treatment:
same STRONG checkpoint
same data order
same LR
same K/T/STOP
same auxiliaries
lambda_safe > 0
```

Primary mechanistic endpoints:

```text
candidate harmful fraction ↓
all-K oracle value ↑
non-C3 Shapley contribution preserved or ↑
C0-C2 conditional PR preserved or ↑
Mean Recall ↑
```

Desired result:

```text
less harmful candidates
without collapsing specialization
without forcing uniform routing
```

A future cleaner mechanistic test could inject the extra safety gradient only into `proposal.queries`, but this is not required for the first treatment.

---

# 20. Decision criteria

## Evidence for successful repair

A treatment is promising if:

1. STRONG harmful candidate rate decreases materially;
2. all-K oracle value moves toward or above OLD;
3. C3 does not need to be artificially suppressed;
4. C0/C1/C2 retain positive Shapley contribution;
5. `K_eff` does not collapse to 1;
6. benchmark Mean Recall improves;
7. STOP/routing behavior does not regress catastrophically.

## Evidence against the repair

Reject or reconsider if:

1. candidate safety improves but all-K oracle decreases;
2. specialization disappears back into four identical candidates without retrieval improvement;
3. routing becomes artificially uniform;
4. C3 contribution is destroyed;
5. only ScoreNet calibration changes while candidate oracle quality remains poor.

---

# 21. Current final diagnosis

The most defensible current statement is:

> **OLD learns four high-quality but nearly redundant candidate edits. STRONG transforms this into a genuinely complementary but strongly imbalanced candidate set: C3 becomes a high-impact dominant candidate, while C0/C1/C2 become lower-rank niche candidates. This differentiation increases cross-slot complementarity but substantially reduces candidate safety and the all-K oracle ceiling. At the final STRONG checkpoint, task-gradient exposure and Concept-MIL responsibility are strongly skewed toward C3, consistent with a winner-take-most reinforcement loop. Static gradients do not prove historical causality, and the exact role of auxiliary coefficient changes remains subject to an OLD checkpoint/config provenance audit.**

In short:

```text
OLD:
high quality
high redundancy
low specialization

STRONG:
real specialization
strong asymmetry
lower candidate quality
winner-take-most reinforcement
```

The research target should therefore be:

> **retain STRONG's useful complementarity while restoring per-candidate safety and representation quality.**

This is the current rationale for testing `L_safe`, rather than enforcing equal candidate usage.