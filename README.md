# CIR IAG-SRME — R0 Diagnostic Audit & Results README

**Document ID:** `CIR_IAG_SRME_R0_DIAGNOSTIC_RESULTS_README_2026-08-30`  
**Date:** 2026-08-30  
**Project:** Composed Image Retrieval (CIR) / IAG-SRME  
**Purpose:** canonical record of the R0 diagnostic phase before any R1 architecture repair  
**R0 branch:** `exp/e2e-iag-srme-r0-diagnostic-audit`  
**Base clean-rewrite SHA:** `f4bc1e8b91e5c43eec36e824fcd4c1d858f32308`  
**Initial R0 SHA:** `0ae06a07897664d71eb41771874cb90a57cb9030`  
**Patched/final R0 SHA:** `e3721e4dd41fa90cf6bd2a8e706822e0ec6e5f16`  
**Primary checkpoint analyzed here:** `outputs/2026-08-30/00-54-41/last.pt`  
**Checkpoint epoch:** 5  
**Backbone:** `qihoo360/fg-clip-base` @ `454d76372c2cf5eb48fa0d871fd0534481484d97`  
**Evaluation protocol:** `fashioniq_original`  
**R0 status:** **PASS — measurement layer is ready; no model/training behavior was changed.**

---

# 0. Executive Summary

R0 was created to answer a narrow question:

> **Where exactly does the current IAG-SRME candidate mechanism collapse, and are the late-step effects genuinely diverse or merely numerically weak?**

R0 intentionally does **not** repair the architecture. It only measures the existing computation graph more faithfully.

The strongest current R0 result on the FG-CLIP Base CORE-last checkpoint is:

```text
WHAT intents are already highly correlated
        ↓
WHERE support maps are almost identical
        ↓
shared editor produces almost parallel token edits
        ↓
token-space edit magnitude stays large across steps
        ↓
retrieval-space effect magnitude collapses rapidly
        ↓
late-step Δq vectors look less similar mostly while becoming very small
        ↓
FULL ≈ MEAN ≈ fixed REPEAT
        ↓
dynamic multi-candidate specialization is not functionally necessary yet
```

The two strongest measured bottlenecks are therefore:

\[
\boxed{B_1:\ \text{WHAT}\rightarrow\text{WHERE contraction}}
\]

and

\[
\boxed{B_2:\ \text{token-state edit}\rightarrow\text{retrieval-effect attenuation}}
\]

The current evidence **does not prove** that `query_cap=0.5` is the sole cause of B2. It only makes the readout/cumulative-query path the next causal target to isolate.

---

# 1. R0 Scope — What Was Allowed and What Was Forbidden

R0 is a **diagnostic-only phase**.

Allowed:

- add lineage-safe metrics;
- add same-parent counterfactual retrieval diagnostics;
- measure WHAT / WHERE / context / `ΔZ` / `Δq`;
- measure selected-path target-relative improvement offline;
- separate STOP occupancy from new STOP hazard;
- add explicit checkpoint configuration provenance;
- fix diagnostic handling of zero/dead effects;
- add timestep-specific failure flags;
- document metric definitions and interpretation limits.

Forbidden in R0:

- architecture repair;
- loss changes;
- grounding changes;
- editor changes;
- readout changes;
- selector changes;
- STOP changes;
- new teacher;
- TPVG;
- DPP;
- semantic residual;
- dynamic re-grounding;
- visual NULL;
- R1a/R1b/R1c mechanisms;
- retraining old checkpoints to make diagnostics work.

This boundary is important:

```text
R0 = measurement correctness
R1 = causal architecture intervention
```

---

# 2. R0 Code Audit and Numerical Freeze

Final R0 patch changed exactly:

```text
src/diagnostics/iag_srme.py
src/diagnose_iag_srme_checkpoint.py
tests/test_r0_diagnostic_semantics.py
doc/R0_DIAGNOSTIC_AUDIT_README.md
```

No files under the following paths were changed by the R0 patch:

```text
src/models/
src/losses/
src/training/
dataset/evaluation implementation
```

The diagnostic freeze test checks that R0 instrumentation does not alter:

```text
final_query
final_state
intents
supports
contexts
delta_z
candidate_states
candidate_queries
delta_q
scores
selected_index
next_state
next_query
```

Local test report supplied for the final patch:

```text
Targeted pytest: 24 passed
Full pytest:     74 passed, 1 skipped
ruff:            PASS
compileall:      PASS
git diff --check PASS
```

Independent GitHub audit confirmed the diff scope and code semantics. At audit time GitHub had no attached CI status for the commit, so the exact local test execution count is recorded as the local run report rather than independently reproduced CI evidence.

---

# 3. Diagnostic Contract

R0 measures the current computation chain:

```text
text
  ↓
WHAT / intent e_k
  ↓
WHERE / support P_k
  ↓
grounded current visual evidence
  ↓
context C_t,k
  ↓
token edit ΔZ_t,k
  ↓
candidate query q̂_{t+1,k}
  ↓
retrieval effect Δq_t,k = q̂_{t+1,k} - q_t
  ↓
score / selected action / STOP
  ↓
actual rollout
```

The main diagnostic families are:

1. **WHAT specialization**
   - pairwise intent cosine;
   - candidate intent norms.

2. **WHERE specialization**
   - pairwise support cosine;
   - support probability overlap;
   - support entropy;
   - support effective size;
   - support fraction;
   - dominant tokenwise grounding mass share.

3. **Context specialization**
   - pairwise context cosine per timestep.

4. **Token-space functional effects**
   - candidate-wise `||ΔZ||`;
   - pairwise `ΔZ` cosine;
   - `ΔZ` effective rank;
   - active/dead effect fractions.

5. **Retrieval-space functional effects**
   - candidate-wise `||Δq||`;
   - pairwise `Δq` cosine;
   - functional effective rank;
   - late-step effect retention;
   - active/dead effect fractions.

6. **Same-parent retrieval controls**
   - each candidate evaluated from the exact same current parent;
   - offline best-candidate oracle;
   - mean candidate query.

7. **Full-rollout controls**
   - FULL;
   - REFERENCE_ONLY;
   - SINGLE-k;
   - REPEAT-k;
   - MEAN-CANDIDATE.

8. **Policy diagnostics**
   - new STOP hazard;
   - absorbed STOP occupancy;
   - candidate selection distribution;
   - repeated-candidate trajectory fraction;
   - mean number of executed edits.

9. **Selected-path target-relative observation**
   - for the actually executed non-STOP transition:

\[
\Delta s_t
=
\cos(q_{t+1},y)-\cos(q_t,y).
\]

This is offline diagnostic evidence only. The target does not enter the model forward path.

---

# 4. Critical R0 Fix — Dead/Zero Functional Effects

The original cosine diagnostic had a dangerous edge case.

For a zero vector, `F.normalize(0)` returns zero. Therefore two dead candidates could numerically appear to have cosine zero and be misread as "orthogonal/diverse".

R0 fixes this by separating:

```text
EFFECT ACTIVITY
from
EFFECT DIVERSITY
```

Diagnostic activity rule:

\[
\boxed{
\text{active}(\Delta)=\mathbf 1[\|\Delta\|_2>10^{-8}]
}
\]

Functional pairwise cosine is defined only when **both** effects are active.

If no active-active pair exists:

```text
cosine cell / summary = null
```

not numerical zero and not NaN.

R0 also reports:

- active candidate count/fraction;
- dead candidate count/fraction;
- dead-parent count/fraction;
- valid cosine pair count;
- possible pair count;
- valid pair fraction.

Effective rank convention:

```text
all-zero effect matrix        → rank = 0
non-zero cloned effects       → rank ≈ 1
four orthogonal active effects→ rank ≈ 4
```

This correction is essential because late-step `Δq` is very small in the current model.

---

# 5. Checkpoint Replay / Provenance Contract

The analyzed CORE checkpoint is a legacy checkpoint and is **not fully self-describing**.

R0 therefore records:

```text
source = legacy_checkpoint_plus_canonical_assumption
fully_self_describing = false
```

State-dict-inferable fields:

```text
num_candidates = 4
width = 256
enable_claim_head = false
enable_factor_head = false
factor_dim = null
```

Canonical non-state-dict assumptions used for replay:

```yaml
max_steps: 3
num_heads: 8
lambda_z: 0.1
query_cap: 0.5
selector_temperature: 1.0
```

Resolved diagnostic retrieval dimension:

```text
retrieval_dim = 512
```

Deterministic diagnostic inference overrides:

```text
selector_gumbel_noise = false
```

This override is evaluation-only and explicitly reported. It is not treated as a claim about the training-time selector configuration.

Future self-describing checkpoints can expose serialized model configuration; R0 prefers that path and cross-checks inferable fields against the state dict/backbone.

---

# 6. Primary R0 Checkpoint

Checkpoint:

```text
outputs/2026-08-30/00-54-41/last.pt
```

Metadata:

```text
epoch = 5
checkpoint metric = 26.6968262692
backbone = qihoo360/fg-clip-base
revision = 454d76372c2cf5eb48fa0d871fd0534481484d97
precision = fp16
protocol = fashioniq_original
```

This is **CORE-last e5**, not the earlier CORE-best e3 checkpoint.

The earlier stored CORE trajectory was:

```text
epoch 1 MR = 25.315
epoch 2 MR = 26.545
epoch 3 MR = 27.258  ← earlier best
epoch 4 MR = 27.006
epoch 5 MR = 26.697  ← checkpoint analyzed by current R0
```

Therefore all new R0 numbers below must be labeled `CORE-last e5` unless the best e3 checkpoint is separately re-run through the patched R0 script.

---

# 7. Retrieval Results and Functional Controls

## 7.1 Main retrieval

| Control | Mean Recall | R@10 | R@50 |
|---|---:|---:|---:|
| REFERENCE_ONLY | 14.8435 | 8.9576 | 20.7294 |
| best SINGLE | 24.3948 | 16.0343 | 32.7553 |
| FULL | **26.6968** | 17.8470 | 35.5466 |
| MEAN-CANDIDATE | 26.6809 | 17.8984 | 35.4635 |
| best REPEAT (`repeat_1`) | **26.9151** | 18.0337 | 35.7966 |

Useful ratios:

```text
MEAN / FULL        = 0.999405
best REPEAT / FULL = 1.008178
best SINGLE / FULL = 0.913771
REF / FULL         = 0.556002
```

Interpretation:

```text
FULL ≈ MEAN
best REPEAT slightly > FULL
```

The gap is small and below the diagnostic `+2 Mean Recall` failure threshold, so the boolean `repeat_beats_full` flag remains false. Scientifically, however, the parity still matters: the dynamic policy has not demonstrated a clear retrieval advantage over repeatedly using one candidate identity.

---

# 8. WHAT — Candidate Intent Geometry

Top-level pairwise intent cosine:

\[
\boxed{
\text{mean cosine}(e_i,e_j)=0.947236
}
\]

Candidate intent norms are all approximately:

```text
15.9930
15.9931
15.9928
15.9926
```

Interpretation:

- the four candidate identities are not numerically identical;
- but they remain highly correlated in text-intent space;
- R0 does not interpret intent cosine alone as semantic correctness or failure.

The important question is whether any existing WHAT difference survives the next mapping into WHERE.

---

# 9. WHERE — Grounding Collapse

Top-level grounding diagnostics:

| Metric | CORE-last e5 |
|---|---:|
| Pairwise support cosine | **0.999821** |
| Pairwise support probability overlap | **0.995809** |
| Support fraction | 0.083240 = 8.32% |
| Support effective size | 15.782 / 196 tokens |
| Support entropy | 2.73393 |
| Dominant tokenwise grounding mass share | 0.25194 |
| Support recomputed each timestep | **false** |
| Support static by current architecture | **true** |

Therefore:

\[
\boxed{
P_1\approx P_2\approx P_3\approx P_4
}
\]

This is **not** simply an over-diffuse or over-sparse grounding problem.

The supports are relatively sparse/localized, but all four candidates choose almost the **same** sparse-ish region distribution.

The R0 failure flag:

```text
high_support_similarity = true
```

is therefore strongly supported by the measurements.

---

# 10. Context and Token-Edit Collapse

Aggregate context off-diagonal cosine is approximately:

```text
~0.959–0.961
```

Aggregate token-effect `ΔZ` off-diagonal cosine:

\[
\boxed{0.990194}
\]

Per timestep:

| Step | mean `||ΔZ||` | `ΔZ` cosine | `ΔZ` effective rank |
|---:|---:|---:|---:|
| t0 | 2.26185 | 0.99005 | 1.6670 |
| t1 | 2.26359 | 0.99007 | 1.6663 |
| t2 | 2.30105 | 0.99050 | 1.6553 |

All `ΔZ` candidates are active under the R0 `1e-8` activity convention.

The key observation is:

```text
ΔZ magnitude does NOT decay with timestep.
```

In fact it stays near 2.26 and slightly increases by t2.

So the editor is still making large token-state modifications; the recurrent failure is not simply "the editor stops editing".

---

# 11. Retrieval-Space Functional Effects — The Strongest R0 Result

Per-timestep same-parent `Δq` diagnostics:

| Step | mean `||Δq||` | pairwise `Δq` cosine | effective rank | active fraction |
|---:|---:|---:|---:|---:|
| t0 | **0.366457** | **0.992265** | 1.5953 | 1.0 |
| t1 | **0.082489** | 0.949220 | 2.2972 | 1.0 |
| t2 | **0.018950** | 0.605588 | 3.4397 | 1.0 |

Late-step retention:

\[
\frac{\mathbb E\|\Delta q_1\|}{\mathbb E\|\Delta q_0\|}
=0.22510
\]

\[
\frac{\mathbb E\|\Delta q_2\|}{\mathbb E\|\Delta q_0\|}
=0.05171
\]

\[
\frac{\mathbb E\|\Delta q_2\|}{\mathbb E\|\Delta q_1\|}
=0.22973
\]

So:

```text
0.3665 → 0.0825 → 0.0190
```

or approximately:

```text
t1 keeps 22.5% of t0 magnitude
t2 keeps  5.17% of t0 magnitude
```

This is the critical R0 distinction:

```text
Token-space effect:
||ΔZ|| ≈ 2.26 → 2.26 → 2.30

Retrieval-space effect:
||Δq|| ≈ 0.366 → 0.082 → 0.019
```

Therefore:

\[
\boxed{
\text{token edits remain large while retrieval consequences attenuate dramatically}
}
\]

---

# 12. Why Late Effective Rank Must Not Be Read Naively

At first glance:

```text
Δq cosine: 0.992 → 0.949 → 0.606
rank:      1.595 → 2.297 → 3.440
```

looks like candidate specialization improves over time.

But simultaneously:

```text
||Δq||: 0.366 → 0.082 → 0.019
```

Thus the model enters a regime where:

```text
strong effects are highly cloned
while
more diverse-looking effects are much weaker
```

R0 does not call t2 "dead" because `1e-8` is only a numerical activity floor, and all candidates remain above it. But the magnitude retention shows that t2 has very little retrieval leverage compared with t0.

Therefore the correct reading is:

> **The rise in late-step effective rank is not evidence that sequential specialization is healthy. It occurs while functional effect energy collapses.**

---

# 13. Selected-Path Target-Similarity Improvement

For the actually executed non-STOP action:

| Step | Mean target cosine improvement | Median | Selected non-STOP count |
|---:|---:|---:|---:|
| t0 | **+0.066652** | +0.069516 | 6011 |
| t1 | **+0.008443** | +0.009022 | 5420 |
| t2 | **+0.001166** | +0.001225 | 1775 |

This is an especially important R0 result.

The actual chosen action's average target-relative improvement falls roughly as:

```text
+0.0667
   ↓ ~8×
+0.00844
   ↓ ~7×
+0.00117
```

Yet `||ΔZ||` remains large.

This reinforces the same conclusion from `Δq`:

> **Later edits still move the token state, but the useful retrieval-space consequence becomes tiny.**

Target firewall remains intact: target gallery features are consumed only after the complete target-free rollout has been constructed.

---

# 14. Selector / STOP Behavior

Top-level selection diagnostics:

```text
queries                         = 6016
executed edits                  = 13206
mean executed edit count        = 2.19515
queries repeating a candidate   = 4798 / 6016
repeated-candidate fraction     = 79.754%
```

Candidate distribution conditional on executing an edit:

```text
candidate 0 = 23.19%
candidate 1 = 40.82%
candidate 2 = 18.07%
candidate 3 = 17.92%
```

This is **not** a single-candidate monopoly under the R0 95% threshold.

New STOP hazard:

```text
t0 = 0.083%
t1 = 9.832%
t2 = 67.251%
```

Absorbed STOP occupancy:

```text
t0 = 0.083%
t1 = 9.907%
t2 = 70.495%
```

The distinction matters:

- **new STOP hazard** = newly stopping among live parents;
- **absorbed STOP occupancy** = all trajectories currently in STOP, including those that stopped earlier.

STOP becomes aggressive late, but R0 does not identify STOP as the root cause. The stronger explanation is that later candidate consequences are already weak/redundant, making stopping increasingly competitive.

---

# 15. Same-Parent Candidate Oracle

Same-parent retrieval is computed by branching every candidate from the exact same current state/query.

### t0

```text
best real candidate ≈ 24.395 MR
best candidate oracle = 24.736 MR
mean candidate query   = 24.336 MR
```

### t1

```text
best real candidate ≈ 26.686 MR
best candidate oracle = 26.912 MR
mean candidate query   = 26.621 MR
```

### t2

```text
best real candidate ≈ 26.142 MR
best candidate oracle = 26.259 MR
mean candidate query   = 26.114 MR
```

The candidate oracle headroom is modest.

This is consistent with candidate effects being too similar/redundant for the selector to exploit a large latent advantage.

The oracle is evaluation-only and target-aware; it never enters the policy forward pass.

---

# 16. Failure Flags — Correct Interpretation

Top-level R0 flags:

```text
high_support_similarity          = true
high_delta_q_similarity_t0       = true
high_delta_q_similarity_t1       = false  # 0.94922 just below 0.95 threshold
high_delta_q_similarity_t2       = false
high_dead_delta_q_fraction_t0    = false
high_dead_delta_q_fraction_t1    = false
high_dead_delta_q_fraction_t2    = false
low_functional_effective_rank_t0 = false  # 1.595 > 1.5 threshold
low_functional_effective_rank_t1 = false
low_functional_effective_rank_t2 = false
single_candidate_monopoly        = false
repeat_beats_full                = false  # requires +2 MR margin
single_beats_full                = false
reference_dominates              = false
never_stop                       = false
high_stop_t0_occupancy           = false
```

These flags are **thresholded observations**, not the scientific conclusion themselves.

Two examples:

1. `repeat_beats_full=false` does **not** mean repeat is harmless. Best REPEAT is still slightly above FULL; it simply does not exceed FULL by the conservative +2 MR flag margin.

2. `low_functional_effective_rank_t2=false` does **not** mean t2 is healthy. Rank is high while `||Δq||` has fallen to only ~5.17% of t0.

For R0, the raw per-timestep measurements have priority over aggregate booleans.

---

# 17. Updated Causal Hypothesis After R0

The strongest current hypothesis is:

```text
4 learnable text queries
        ↓
highly correlated WHAT intents
        ↓
AnchorGrounder
        ↓
nearly identical WHERE maps
        ↓
nearly shared grounded evidence
        ↓
highly correlated edit contexts
        ↓
SharedTokenEditor
        ↓
nearly parallel ΔZ effects
        ↓
large cumulative token displacement still occurs
        ↓
TokenStateReadout / accumulated displacement geometry
        ↓
late Δq magnitude collapses
        ↓
late realized target gain approaches zero
        ↓
STOP becomes attractive
        ↓
FULL ≈ MEAN ≈ REPEAT
```

This is still a **hypothesis register**, not a proven causal graph.

R0 provides strong localization evidence for two interfaces:

### B1 — WHAT → WHERE

Evidence:

```text
intent cosine  = 0.9472
support cosine = 0.999821
support overlap= 0.995809
```

The grounding mapping is substantially more contractive than the intent space.

### B2 — ΔZ → Δq / recurrent readout

Evidence:

```text
||ΔZ|| : 2.262 → 2.264 → 2.301
||Δq|| : 0.366 → 0.082 → 0.019
```

Large token edits are not translated into proportional late retrieval-query changes.

---

# 18. What R0 Does NOT Prove

R0 does **not** prove any of the following:

```text
FG-CLIP is fundamentally unsuitable for CIR.
query_cap=0.5 is the only source of attenuation.
Grounding clone is caused solely by Entmax.
STOP is the root cause.
A lower support cosine automatically means better grounding.
A higher effective rank automatically means better actions.
Candidate semantic labels correspond to human-readable edits.
Target-relative diagnostics are available at inference.
```

R0 only establishes measured behavior under the current architecture.

---

# 19. Relationship to Earlier CORE-best e3 Diagnostic

Earlier stored CORE-best e3 measurements:

| Metric | CORE best e3 | CORE last e5 / patched R0 |
|---|---:|---:|
| Mean Recall | 27.258 | 26.697 |
| Support cosine | 0.999795 | 0.999821 |
| Support overlap | 0.995188 | 0.995809 |
| t0 `Δq` cosine | ~0.9897 | 0.9923 |
| t0 effect rank | 1.663 | 1.595 |
| t1 effect rank | 2.339 | 2.297 |
| t2 effect rank | 3.440 | 3.440 |
| mean edits | 2.682 | 2.195 |
| t2 STOP hazard | 25.74% | 67.25% |
| repeated-candidate fraction | 90.99% | 79.75% |

Both epochs expose the same structural pattern:

```text
WHERE clone
+
parallel root effects
+
late retrieval-effect attenuation
```

Longer training does not spontaneously create healthy specialization.

Because the patched R0 diagnostic improved dead-effect semantics and per-step reporting, future comparisons should prefer rerunning all important checkpoints through the same patched R0 script rather than mixing raw legacy and patched metrics indiscriminately.

---

# 20. Relationship to the Five-Run Semantic-Loss Ablations

Earlier same-stratum ablation results:

| Run | Best MR | Intent cosine | Support cosine | Main interpretation |
|---|---:|---:|---:|---|
| CORE | 27.258 | 0.9476 | 0.999795 | baseline collapse |
| BIND | **27.582** | 0.9423 | 0.999372 | mild WHAT improvement, WHERE still clone |
| COMP | 27.125 | 0.9636 | 0.999467 | negative |
| COMP+BIND | 27.492 | **0.9347** | **0.999901** | strongest WHAT→WHERE localization |
| FACTOR | 27.315 | 0.9569 | 0.999750 | no central repair |

The important scientific pattern is:

```text
semantic losses can move WHAT
but WHERE remains ~0.9994–0.9999 cosine
```

Thus adding more representation-only diversity pressure is not justified as the next default move.

Note: those legacy ablation numbers and the newly patched R0 CORE-last result were not all produced by the exact same patched diagnostic implementation. They should be treated as supporting historical evidence, while future causal decisions should use patched-R0 re-evaluations where possible.

---

# 21. Current R0 Verdict

## Measurement layer

\[
\boxed{\text{R0 implementation: PASS}}
\]

The diagnostic layer is now sufficiently reliable for the immediate repair ladder because it:

- preserves model behavior;
- keeps target out of the forward path;
- measures same-parent counterfactuals;
- separates numerical inactivity from diversity;
- reports per-timestep rather than only aggregate functional metrics;
- records checkpoint replay assumptions;
- distinguishes STOP occupancy from STOP hazard.

## Model diagnosis

\[
\boxed{
\text{Current IAG-SRME still has severe functional candidate redundancy.}
}
\]

The strongest signatures are:

```text
support cosine  ≈ 0.999821
ΔZ cosine       ≈ 0.990
root Δq cosine  ≈ 0.992
FULL            ≈ MEAN
REPEAT          ≳ FULL
```

plus the recurrent attenuation:

```text
||Δq|| = 0.366 → 0.082 → 0.019
```

while:

```text
||ΔZ|| ≈ 2.26 → 2.26 → 2.30
```

and realized selected-action target gain:

```text
+0.0667 → +0.00844 → +0.00117
```

---

# 22. Decision Gate for R1

R0 should now be considered complete enough to move to a **single-variable causal intervention**.

The first proposed R1 question is:

> Is the global cumulative retrieval-query cap/readout geometry materially responsible for late-step attenuation?

The current readout contains a bounded accumulated change with canonical:

```text
query_cap = 0.5
```

R0 has **not** established causality yet.

The correct next experiment is therefore not "redesign everything" but:

```text
R1a:
change only the cumulative query-cap behavior
keep architecture/loss/backbone/K/Tmax/data/protocol fixed
then rerun the exact same R0 diagnostics
```

Primary R1a success/falsification metrics:

```text
mean ||Δq|| t0/t1/t2
late-step retention t1/t0 and t2/t0
selected-path target gain t1/t2
FULL vs MEAN
FULL vs best REPEAT
same-parent oracle gap
stability / retrieval regression
```

If removing/relaxing the cumulative cap restores late-step `Δq` magnitude and useful marginal gain, B2 is causally supported.

If late-step attenuation remains almost unchanged, the cap hypothesis is falsified as the primary cause and the investigation should move to the next interface rather than protecting the hypothesis.

---

# 23. Reproduction Commands

## 23.1 Diagnose CORE-last e5

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python src/diagnose_iag_srme_checkpoint.py \
  --checkpoint outputs/2026-08-30/00-54-41/last.pt \
  --dataset-root data/FashionIQ \
  --protocol fashioniq_original \
  --batch-size 32 \
  --gallery-batch-size 128 \
  --output reports/r0_core_last_original.json
```

Expected main retrieval:

```text
R@10  ≈ 17.8470
R@50  ≈ 35.5466
MR    ≈ 26.6968
```

## 23.2 Re-run any compatible legacy checkpoint

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python src/diagnose_iag_srme_checkpoint.py \
  --checkpoint <CHECKPOINT.pt> \
  --dataset-root data/FashionIQ \
  --protocol fashioniq_original \
  --batch-size 32 \
  --gallery-batch-size 128 \
  --output reports/<NAME>.json
```

Always verify `checkpoint_model_config_provenance` before interpreting a legacy checkpoint with non-canonical historical overrides.

---

# 24. Minimal R0 Checklist for Every Future Checkpoint

Before making a causal claim, record at minimum:

```text
[ ] exact checkpoint path / epoch / metric
[ ] git SHA / branch
[ ] protocol
[ ] checkpoint config provenance
[ ] FULL / REF / SINGLE / REPEAT / MEAN
[ ] same-parent candidate oracle by timestep
[ ] intent cosine
[ ] support cosine + overlap + effective size
[ ] context cosine by timestep
[ ] ΔZ norm + cosine + rank by timestep
[ ] Δq norm + cosine + rank by timestep
[ ] active/dead effect fractions
[ ] late-step effect retention
[ ] selected-path target gain by timestep
[ ] STOP hazard by timestep
[ ] repeated-candidate fraction
[ ] candidate selection distribution
[ ] raw metrics, not only boolean failure flags
```

---

# 25. One-Screen Handoff

```text
R0 branch:
exp/e2e-iag-srme-r0-diagnostic-audit

final R0 SHA:
e3721e4dd41fa90cf6bd2a8e706822e0ec6e5f16

primary R0 checkpoint:
outputs/2026-08-30/00-54-41/last.pt
CORE-last epoch 5
FashionIQ original
FG-CLIP Base

retrieval:
FULL        26.6968
REF         14.8435
MEAN        26.6809
best REPEAT 26.9151
best SINGLE 24.3948

WHAT:
intent cosine 0.9472

WHERE:
support cosine  0.999821
support overlap 0.995809
support eff size 15.78 / 196
static support = true

TOKEN EFFECT:
||ΔZ||
2.262 → 2.264 → 2.301
ΔZ cosine ≈ 0.990 every step

RETRIEVAL EFFECT:
||Δq||
0.3665 → 0.0825 → 0.0190
retention:
t1/t0 = 22.5%
t2/t0 = 5.17%

Δq cosine:
0.9923 → 0.9492 → 0.6056
rank:
1.595 → 2.297 → 3.440

selected target gain:
+0.06665 → +0.00844 → +0.00117

policy:
mean edits = 2.195
repeat-query fraction = 79.75%
max candidate share = 40.82%
STOP hazard = 0.083% → 9.83% → 67.25%

main diagnosis:
B1 = WHAT→WHERE contraction
B2 = large token edits but severe late retrieval-effect attenuation

next causal gate:
R1a = isolate cumulative query-cap/readout contribution only
```

---

# 26. Final Scientific Statement

The R0 evidence supports the following cautious statement:

> **The current IAG-SRME implementation does not fail because the editor becomes numerically inactive. Instead, four candidate pathways remain highly redundant spatially and functionally: their grounding supports are nearly identical, their token-level edits are almost parallel, and although token-state edit magnitude remains large across recurrent steps, the resulting retrieval-query displacement decays by roughly an order of magnitude per two steps. The dynamic policy therefore has little measurable advantage over mean/repeated candidate controls. R0 localizes the next causal investigations to the WHAT→WHERE interface and the recurrent readout/accumulation path, without yet assigning sole causality to either mechanism.**

This is the boundary at which R0 ends and R1 begins.

# 27. R1a — Remove Global Query Cap: Causal Diagnostic Addendum

**Document role:** append this section directly after Section 26 of `CIR_IAG_SRME_R0_DIAGNOSTIC_RESULTS_README_2026-08-30.md`  
**Date:** 2026-08-30  
**Experiment:** R1a — isolate the contribution of the cumulative retrieval-query cap  
**Only intended intervention:** `model.query_cap: 0.5 -> 1000.0`  
**Backbone:** `qihoo360/fg-clip-base` @ `454d76372c2cf5eb48fa0d871fd0534481484d97`  
**K:** 4  
**Tmax:** 3  
**lambda_z:** 0.1  
**Protocol:** `fashioniq_original`  
**Best checkpoint epoch:** 3  
**Best checkpoint:** `outputs/2026-08-30/21-34-29/best.pt`  
**Self-describing diagnostic copy:** `outputs/2026-08-30/21-34-29/best_r1a_self_describing.pt`  
**Best Mean Recall:** **38.764146**  
**R1a verdict:** **PASS for the readout-cap hypothesis; candidate/grounding collapse remains.**

---

## 27.1 Hypothesis

R0 showed a sharp mismatch:

```text
token-space edit magnitude:
||ΔZ|| remains large through t0,t1,t2

retrieval-space edit magnitude:
||Δq|| collapses strongly through t0,t1,t2
```

The primary R0 CORE-last trajectory was approximately:

```text
||Δq||:
0.3665 -> 0.0825 -> 0.0190

late retention:
t1/t0 ≈ 22.5%
t2/t0 ≈ 5.17%
```

while:

```text
||ΔZ||:
2.262 -> 2.264 -> 2.301
```

The causal R1a hypothesis was therefore:

\[
H_{\rm R1a}:
\quad
\texttt{query\_cap}=0.5
\text{ is a major bottleneck that suppresses recurrent retrieval-space updates.}
\]

R1a deliberately did **not** modify:

- WHAT/intention generation;
- grounding;
- support construction;
- editor;
- candidate count;
- recurrent horizon;
- STOP formulation;
- selector;
- objective/loss set;
- teacher supervision;
- DPP/diversity regularization;
- semantic residual;
- dynamic re-grounding.

The intervention was only:

```yaml
model:
  query_cap: 1000.0
```

With the existing `cap_vector` formulation, `1000.0` makes the global cap effectively close to identity at the observed update scales.

Training command:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python src/train.py \
  model.query_cap=1000.0
```

---

# 28. Training Result — Large Performance Recovery

Training trajectory:

```text
epoch  1   MR 33.864
epoch  2   MR 37.375
epoch  3   MR 38.764   <- BEST
epoch  4   MR 38.637
epoch  5   MR 36.563
epoch  6   MR 36.641
epoch  7   MR 33.686
epoch  8   MR 33.388
epoch  9   MR 31.971
epoch 10   MR 33.134
epoch 11   MR 30.176
epoch 12   MR 30.054
epoch 13   MR 29.634
epoch 14   MR 31.207
epoch 15   MR 29.378
epoch 16   MR 28.984
epoch 17   MR 25.646
epoch 18   MR 28.078
epoch 19   MR 27.490
epoch 20   MR 29.626
```

The best checkpoint is therefore **epoch 3**, not `last.pt`.

Compared with the primary R0 CORE-last checkpoint:

```text
R0 CORE-last:
MR ≈ 26.697

R1a best:
MR = 38.764
```

Absolute gain relative to that R0 reference:

\[
\boxed{
38.764-26.697\approx +12.07\ {\rm Mean\ Recall}
}
\]

This is too large to treat the global cap as a minor implementation detail.

However, this performance delta should not be interpreted as a perfectly epoch-matched estimate because R0's primary recorded checkpoint is CORE-last epoch 5 while R1a uses the best checkpoint epoch 3. The strongest causal evidence comes from the internal effect diagnostics below, not from the headline MR difference alone.

---

# 29. Important Provenance Failure Discovered During R1a

The first R1a diagnostic replay was invalid.

The legacy checkpoint did not serialize `query_cap=1000.0`. The R0 diagnostic loader therefore fell back to its canonical legacy assumption:

```text
query_cap = 0.5
```

even though the checkpoint weights had been trained under:

```text
query_cap = 1000.0
```

This produced the contradiction:

```text
checkpoint_metric = 38.764
diagnostic FULL MR ≈ 27.639
```

and provenance explicitly showed:

```text
source = legacy_checkpoint_plus_canonical_assumption
resolved_diagnostic_config.query_cap = 0.5
```

That report is **not a valid R1a report**.

A self-describing diagnostic copy was then created with the actual training configuration.

The corrected replay reports:

```text
source = checkpoint
fully_self_describing = true
warning = null

serialized_training_config.query_cap = 1000.0
resolved_diagnostic_config.query_cap = 1000.0
```

and:

```text
checkpoint_metric = 38.764146
diagnostic FULL MR = 38.764146
```

Exact agreement establishes replay consistency.

### Reproducibility lesson

Future checkpoints must serialize replay-critical model hyperparameters, especially values not inferable from the state dict:

```text
query_cap
lambda_z
max_steps
num_heads
selector_temperature
selector_gumbel_noise
and any future non-state architecture scalar
```

Hydra CLI overrides must not disappear at checkpoint save time.

### Useful but non-canonical observation

The accidental mismatched replay is still mechanistically informative:

```text
same R1a-trained weights
cap=1000 replay -> MR 38.764
cap=0.5 replay  -> MR ~27.639
```

This should **not** be reported as a benchmark comparison because the weights were optimized for the cap=1000 regime, but it independently shows that reinstating the small cap at inference severely damages the learned recurrent query trajectory.

---

# 30. Correct R1a Retrieval Metrics

Corrected global FashionIQ-original metrics:

| Control | Mean Recall | R@10 | R@50 |
|---|---:|---:|---:|
| FULL | **38.7641** | 27.5447 | 49.9836 |
| MEAN candidate | **38.8160** | 27.5754 | 50.0565 |
| REPEAT-0 | 39.0013 | 27.8511 | 50.1515 |
| REPEAT-1 | **39.2232** | 28.0624 | 50.3840 |
| REPEAT-2 | 38.9622 | 27.8417 | 50.0826 |
| REPEAT-3 | 38.9954 | 27.6887 | 50.3022 |
| SINGLE-0 | 24.4947 | 15.7670 | 33.2224 |
| SINGLE-1 | **24.7096** | 15.9147 | 33.5045 |
| SINGLE-2 | 24.5529 | 15.8988 | 33.2070 |
| SINGLE-3 | 24.4862 | 15.8000 | 33.1724 |
| REFERENCE only | 14.5216 | 8.7899 | 20.2532 |

Useful ratios:

```text
best REPEAT / FULL = 1.01184
best SINGLE / FULL = 0.63743
MEAN / FULL        = 1.00134
REFERENCE / FULL   = 0.37461
```

Interpretation:

1. Multi-step execution is now extremely important:
   - SINGLE is only ~64% of FULL.
2. The reference image alone is far below the final composed retrieval result.
3. However candidate identity is still weak:
   - averaging candidates is essentially equal to FULL;
   - a fixed repeated candidate slightly beats the learned dynamic policy.

Therefore R1a restores **depth utility**, but does not restore **candidate specialization**.

---

# 31. Same-Parent Counterfactuals — Multi-Step Is Now Actually Useful

The mean same-parent candidate query improves dramatically by recurrent step:

```text
t0 mean candidate MR = 24.6032
t1 mean candidate MR = 34.3021
t2 mean candidate MR = 39.0305
```

Offline best-candidate oracle:

```text
t0 oracle = 25.0834
t1 oracle = 34.7513
t2 oracle = 39.5640
```

Therefore:

\[
\boxed{
24.60
\rightarrow
34.30
\rightarrow
39.03
}
\]

is a strong empirical result.

This directly rejects the strongest version of the hypothesis that:

> "the sequential/multi-step concept itself is fundamentally useless."

After removing the cap bottleneck, recurrence produces large retrieval improvements.

The model was previously trying to update token state, but the readout path prevented much of that accumulated change from surviving into retrieval space.

### Candidate headroom remains small

Within each timestep, the four candidates remain close:

```text
t0 candidates ≈ 24.49 .. 24.71
t1 candidates ≈ 34.21 .. 34.26
t2 candidates ≈ 38.98 .. 39.05
```

The offline oracle gains only a limited amount over each candidate.

Thus recurrence is useful, but choosing among four candidate identities still contributes relatively little.

---

# 32. R1a Successfully Repairs Late-Step Retrieval-Effect Attenuation

This is the central R1a result.

Corrected mean `||Δq||`:

```text
t0 = 0.336634
t1 = 0.272417
t2 = 0.197111
```

Retention:

```text
t1/t0 = 0.809238  = 80.92%
t2/t0 = 0.585534  = 58.55%
t2/t1 = 0.723562  = 72.36%
```

Compare with primary R0 CORE-last:

```text
R0:
0.3665 -> 0.0825 -> 0.0190

R1a:
0.3366 -> 0.2724 -> 0.1971
```

The important change is not t0. It is the preservation of later effects.

Approximate comparison:

| Metric | R0 CORE-last | R1a |
|---|---:|---:|
| `||Δq|| t0` | 0.3665 | 0.3366 |
| `||Δq|| t1` | 0.0825 | **0.2724** |
| `||Δq|| t2` | 0.0190 | **0.1971** |
| `t1/t0` | 22.5% | **80.9%** |
| `t2/t0` | 5.17% | **58.6%** |

Therefore:

\[
\boxed{
\texttt{query\_cap}=0.5
\text{ was a major causal bottleneck in recurrent retrieval-effect propagation.}
}
\]

This is stronger than the original R0 correlation.

R1a does **not** prove that every performance problem came from the cap.

It proves that the cap was responsible for a major part of the observed:

\[
\Delta Z\ {\rm alive}
\quad\text{but}\quad
\Delta q\ {\rm dying}
\]

failure mode.

---

# 33. Token-Space Effects Remain Highly Redundant

R1a does not repair the upstream candidate collapse.

Mean token-effect norms:

```text
||ΔZ||:
t0 = 1.38355
t1 = 1.39111
t2 = 1.40001
```

Thus the editor remains active at every timestep.

But pairwise `ΔZ` cosine remains:

```text
t0 = 0.98187
t1 = 0.98214
t2 = 0.98245
```

and `ΔZ` effective rank remains approximately:

```text
t0 = 1.856
t1 = 1.851
t2 = 1.846
```

Maximum possible candidate-axis rank is 4.

Therefore the token-level transition still behaves much closer to:

```text
candidate 0 -> nearly same edit direction
candidate 1 -> nearly same edit direction
candidate 2 -> nearly same edit direction
candidate 3 -> nearly same edit direction
```

than to four genuinely distinct functional edit factors.

R1a solves **effect survival**, not **effect diversity**.

---

# 34. Retrieval-Space Candidate Diversity Also Remains Weak

R1a `Δq` pairwise cosine:

```text
t0 = 0.98330
t1 = 0.98167
t2 = 0.97654
```

Functional effective rank:

```text
t0 = 1.818
t1 = 1.856
t2 = 1.954
```

All candidate effects are numerically active:

```text
active candidate fraction = 1.0
dead candidate fraction   = 0.0
dead parent fraction      = 0.0
```

So this is no longer a "dead effect" problem.

It is a **live-but-redundant** effect problem.

R0 had an important interpretability trap:

```text
late Δq cosine fell
late rank rose
but Δq energy was almost dead
```

R1a removes much of that ambiguity because late-step energy remains substantial.

Yet even with healthy magnitude:

```text
t2 ||Δq|| ≈ 0.197
```

the four effects still have:

```text
t2 cosine ≈ 0.977
rank ≈ 1.95 / 4
```

Therefore candidate redundancy is now a much cleaner finding.

---

# 35. WHAT and WHERE Were Not Repaired by R1a

Intent similarity:

```text
mean off-diagonal intent cosine = 0.949880
```

Grounding:

```text
support cosine  = 0.999842
support overlap = 0.995108

support effective size = 10.573 / 196 tokens
support fraction       = 0.05603
```

The grounder is also still architecturally static:

```text
support_recomputed_each_timestep = false
support_static_by_current_architecture = true
```

Thus R1a changes the chain from:

```text
WHAT correlated
  ↓
WHERE almost cloned
  ↓
ΔZ almost cloned
  ↓
Δq almost cloned AND rapidly attenuated
```

to:

```text
WHAT correlated
  ↓
WHERE almost cloned
  ↓
ΔZ almost cloned
  ↓
Δq almost cloned BUT now remains strong across recurrence
```

This is an important localization result.

The readout bottleneck and grounding/candidate redundancy are **separate failure modes**.

---

# 36. Selected-Path Utility Reveals a New Late Over-Edit Problem

Target-relative improvement on the actual selected non-STOP transition:

```text
t0 mean Δ target cosine = +0.074237
t1 mean Δ target cosine = +0.024877
t2 mean Δ target cosine = -0.002613
```

Counts:

```text
t0 selected non-STOP = 5917
t1 selected non-STOP = 5788
t2 selected non-STOP = 5536
```

The first two edits are strongly useful on average.

The third selected edit is slightly harmful on average:

\[
\boxed{
E[\Delta {\rm sim}_{target}\mid t=2,\ {\rm execute}]
\approx -0.00261
}
\]

This failure was difficult to see under the old cap because the third-step effect itself was almost annihilated.

After R1a, the third-step effect survives, so **over-editing becomes observable**.

This suggests a new distinction:

```text
R0 problem:
late actions cannot express enough retrieval movement.

R1a problem:
late actions can move strongly,
but the policy often should STOP instead of applying another redundant edit.
```

The negative t2 target-similarity observation is offline diagnostic evidence only. It does not mean target information entered the forward path.

---

# 37. STOP and Selection After R1a

Global selection:

```text
absorbed STOP occupancy:
t0 = 1.65%
t1 = 3.79%
t2 = 7.98%
```

New STOP counts:

```text
t0 =  99 / 6016
t1 = 129 / 5917
t2 = 252 / 5788
```

Approximate new STOP hazard:

```text
t0 ≈ 1.65%
t1 ≈ 2.18%
t2 ≈ 4.35%
```

Mean executed edits:

```text
2.86586 / maximum 3
```

Thus most examples are still pushed close to the full horizon.

Candidate distribution conditional on edit:

```text
candidate 0 =  8.86%
candidate 1 = 40.06%
candidate 2 = 26.05%
candidate 3 = 25.04%
```

There is no strict single-candidate monopoly under the R0 threshold.

However:

```text
queries with repeated candidate selections
= 5760 / 6016
= 95.74%
```

This is extremely high.

Therefore the current behavior is better summarized as:

\[
\boxed{
\text{repeat-heavy consensus/redundancy}
}
\]

rather than:

\[
\boxed{
\text{single-slot monopoly}
}
\]

The combination:

```text
95.7% repeated-candidate trajectories
+
best fixed REPEAT > FULL
+
t2 selected target gain < 0
```

shows that action identity and STOP policy are not yet exploiting a genuinely diverse candidate set.

---

# 38. R1a Red-Team Interpretation

## 38.1 What R1a proves strongly

R1a strongly supports:

\[
\boxed{
\text{The global query cap was a major causal cause of late retrieval-effect attenuation.}
}
\]

Evidence:

- only intended architectural/config intervention was query cap;
- corrected replay uses `query_cap=1000`;
- checkpoint MR and diagnostic MR match exactly;
- late `Δq` retention recovers from roughly 5% to roughly 59% by t2;
- same-parent recurrent retrieval rises dramatically through t0 -> t1 -> t2;
- headline performance increases very strongly.

## 38.2 What R1a does not prove

R1a does **not** prove:

- candidate decomposition is solved;
- grounding is correct;
- four actions correspond to four semantic edits;
- visual supports are meaningfully distinct;
- selector is optimal;
- STOP is calibrated;
- `query_cap=1000` is necessarily the final best production formulation;
- no norm control/trust region is ever needed;
- multi-step always needs exactly 3 edits.

## 38.3 Remaining collapse after R1a

The remaining collapse is approximately:

```text
WHAT similarity      ~0.950
WHERE cosine         ~0.99984
WHERE overlap        ~0.99511
ΔZ cosine            ~0.982
Δq cosine            ~0.977-0.983
functional rank      ~1.82-1.95 / 4
repeat trajectories  ~95.7%
```

This is no longer a low-energy artifact.

The candidate set is genuinely highly redundant at useful effect magnitude.

---

# 39. Updated Failure Decomposition

The current best decomposition is:

\[
\boxed{
B_1:
\text{candidate semantic/grounding redundancy}
}
\]

Observed as:

```text
WHAT highly correlated
WHERE nearly identical
static WHERE across recurrence
```

then:

\[
\boxed{
B_2:
\text{editor functional redundancy}
}
\]

Observed as:

```text
ΔZ cosine ≈ 0.982
ΔZ effective rank ≈ 1.85 / 4
```

R1a has largely repaired the previous:

\[
\boxed{
B_3:
\text{late retrieval-effect attenuation from global query capping}
}
\]

because:

```text
t2/t0 Δq retention:
~5.2% -> ~58.6%
```

and exposes another issue:

\[
\boxed{
B_4:
\text{late over-edit / insufficient STOP calibration}
}
\]

because:

```text
t2 selected target-relative gain ≈ -0.00261
mean edits ≈ 2.866 / 3
t2 new STOP hazard only ≈ 4.35%
```

---

# 40. Updated Causal Chain

Before R1a:

```text
correlated WHAT
     ↓
near-identical WHERE
     ↓
near-identical ΔZ
     ↓
global query cap compresses accumulated displacement
     ↓
Δq energy dies rapidly
     ↓
late recurrence contributes almost nothing
     ↓
FULL ≈ MEAN ≈ REPEAT
```

After R1a:

```text
correlated WHAT
     ↓
near-identical static WHERE
     ↓
near-identical ΔZ
     ↓
retrieval effect now survives recurrent accumulation
     ↓
multi-step retrieval improves strongly
     ↓
but candidate directions remain clones
     ↓
policy repeats candidates
     ↓
third edit is often unnecessary / mildly harmful
```

The scientific interpretation therefore changes from:

> "The whole multi-step mechanism may be invalid."

to:

> **"The multi-step mechanism can be highly useful once recurrent query attenuation is removed, but the current proposer-grounder-editor stack still produces a redundant candidate set and the policy does not yet know reliably when to stop."**

---

# 41. R1a Decision Gate

R1a success criteria:

```text
[PASS] corrected replay uses query_cap=1000
[PASS] diagnostic MR matches checkpoint MR
[PASS] late Δq magnitude recovers substantially
[PASS] t1/t0 retention improves strongly
[PASS] t2/t0 retention improves strongly
[PASS] recurrence produces large retrieval gains
```

R1a remaining failures:

```text
[FAIL] support maps remain near-identical
[FAIL] ΔZ candidates remain near-parallel
[FAIL] Δq candidates remain near-parallel
[FAIL] functional rank remains far below K=4
[FAIL] fixed REPEAT remains slightly better than FULL
[FAIL] MEAN remains essentially equal to FULL
[FAIL] repeated-candidate trajectory fraction remains extremely high
[FAIL] selected t2 edit has slightly negative target-relative mean gain
[FAIL] STOP hazard remains low relative to the observed t2 over-edit signal
```

Therefore:

\[
\boxed{
\textbf{R1a = PASS as a causal repair, but not a complete anti-collapse solution.}
}
\]

---

# 42. What Should Not Be Done Next

Do **not** interpret the large MR gain as justification to immediately stack every proposed mechanism.

Do not yet combine:

```text
dynamic grounding
+ semantic residual
+ DPP
+ teacher grounding
+ new STOP loss
+ new scorer
```

in one run.

That would destroy causal attribution.

R1a has cleanly removed one bottleneck. The next intervention should again isolate one hypothesis.

Also do not use only headline MR as the decision metric.

Every next experiment must preserve:

```text
FULL / MEAN / SINGLE / REPEAT
same-parent candidate retrieval
intent cosine
support cosine / overlap
ΔZ norm / cosine / rank
Δq norm / cosine / rank
late retention
selected target-relative gain
STOP hazard
repeated-candidate fraction
```

---

# 43. Recommended Next Investigation

The next bottleneck is no longer "can recurrent effects survive?"

They can.

The next question is:

> **Can each timestep generate a meaningfully different residual edit conditioned on the current state, instead of repeatedly executing almost the same static support/action program?**

The strongest architectural evidence motivating that question is:

```text
support cosine  ≈ 0.99984
support overlap ≈ 0.99511
support is static across t
ΔZ cosine       ≈ 0.982
repeat fraction ≈ 95.7%
```

Therefore the next clean candidate is a **current-state re-ground / re-propose causal test** rather than immediately adding a diversity loss.

Conceptually:

```text
current architecture:
text -> candidate intents -> support ONCE
                         ↓
Z0 -> Z1 -> Z2 -> Z3
      same supports reused

next causal question:
(Z_t, residual text/action state)
        ↓
recompute proposal/grounding at timestep t
        ↓
edit what is still missing NOW
```

This should be tested before DPP because DPP can force geometric separation without proving that the separated actions correspond to useful remaining edits.

Likewise, a target-aware grounding teacher should remain conditional until the teacher-free dynamic-grounding hypothesis is tested cleanly.

---

# 44. One-Screen R1a Handoff

```text
R1a intervention:
query_cap 0.5 -> 1000.0 only

best checkpoint:
outputs/2026-08-30/21-34-29/best.pt
epoch 3

correct diagnostic checkpoint:
best_r1a_self_describing.pt

provenance:
source = checkpoint
fully_self_describing = true
resolved query_cap = 1000.0

retrieval:
FULL        38.7641
MEAN        38.8160
best REPEAT 39.2232
best SINGLE 24.7096
REFERENCE   14.5216

same-parent MR:
t0 24.603
t1 34.302
t2 39.030

Δq norm:
0.3366 -> 0.2724 -> 0.1971

Δq retention:
t1/t0 = 80.9%
t2/t0 = 58.6%
t2/t1 = 72.4%

Δq cosine:
0.9833 -> 0.9817 -> 0.9765

functional rank:
1.818 -> 1.856 -> 1.954

ΔZ norm:
1.3836 -> 1.3911 -> 1.4000

ΔZ cosine:
0.9819 -> 0.9821 -> 0.9824

WHAT:
intent cosine = 0.9499

WHERE:
support cosine  = 0.999842
support overlap = 0.995108
support eff size = 10.57 / 196
support static = true

selected target-relative gain:
t0 +0.07424
t1 +0.02488
t2 -0.00261

policy:
mean edits = 2.866 / 3
repeat-query fraction = 95.74%
max candidate share = 40.06%
new STOP hazard ≈ 1.65% -> 2.18% -> 4.35%

R1a conclusion:
global query cap was a major causal bottleneck.
It suppressed late retrieval-space effects and hid the usefulness
of recurrence.

remaining failure:
candidate/grounding/editor specialization remains collapsed,
and the newly strengthened rollout now exposes late over-editing.

next clean question:
current-state re-ground / re-propose before adding diversity losses.
```

---

# 45. Final R1a Scientific Statement

The corrected R1a experiment supports the following statement:

> **Removing the small global cumulative query cap produces a large recovery in FashionIQ-original retrieval and, more importantly, restores recurrent retrieval-space effect magnitude: t2 retains roughly 58.6% of the t0 `Δq` norm instead of roughly 5% in the R0 CORE-last diagnostic. Same-parent retrieval improves strongly across recurrent steps, demonstrating that the sequential editing mechanism can provide substantial utility when its readout is not aggressively compressed. However, R1a does not solve candidate collapse: support maps remain almost identical, token-space and retrieval-space candidate effects remain highly parallel, fixed REPEAT and MEAN controls remain competitive with FULL, and approximately 95.7% of trajectories reuse a candidate identity. The stronger recurrence also reveals a new late-stage control problem: the selected third edit has slightly negative mean target-relative improvement while STOP remains rare. The next causal investigation should therefore target current-state-conditioned proposal/grounding and residual edit generation, while keeping diversity regularization and target-aware teacher mechanisms conditional on that result.**