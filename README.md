# CIR IAG-SRME R2 Semantic Residual / Claim Firewall — Negative Result Diagnostic README

**Date:** 2026-08-31  
**Branch:** `exp/e2e-iag-srme-r2-semantic-residual`  
**Architecture generation:** `r2_semantic_residual_claim_firewall_v2`  
**Status:** **MECHANICALLY VALID / SCIENTIFICALLY NEGATIVE — STOP AFTER 1-EPOCH PROBE**  
**Recommended action:** **Freeze this branch as a negative result. Do not continue the 20-epoch run and do not rescue it by stacking DPP/VISReg/RDMReg/teacher/extra STOP losses in the same branch.**

---

## 1. Executive conclusion

R2 was designed to test whether the dominant SRME collapse after R1a/R1b/R1c1/R1c2 was caused by the absence of an explicit **semantic responsibility / explaining-away state**.

The intended mechanism was:

\[
\rho_0 = 1
\]

on valid modification tokens, with candidate-specific semantic claim

\[
\alpha_{t,k,m}
\]

and independent candidate-specific consumption/satisfaction

\[
\gamma_{t,k,m}.
\]

The executed candidate was supposed to consume only the semantic evidence it actually claimed:

\[
\rho_{t+1,m}
=
\rho_{t,m}
\left(
1-\alpha_{t,a_t,m}\gamma_{t,a_t,m}
\right).
\]

The critical hypothesis was:

> after one selected action explains part of the modification, later actions should operate on a **different remaining semantic problem**, and candidate identity should matter causally.

The implementation is mechanically correct, checkpoint replay is exact, the semantic residual is active, gradients reach all required R2 modules, the target firewall is intact, the same-parent invariant is intact, and residual magnitude reaches executable semantics.

However, after a complete one-epoch FashionIQ probe, the semantic mechanism does **not** learn candidate-specific responsibility.

The strongest result is:

\[
\boxed{
\text{R2 learns semantic attenuation, not semantic explaining-away.}
}
\]

More specifically:

1. Candidate claims are effectively identical:
   \[
   \cos(\alpha_i,\alpha_j)\approx 0.999999995.
   \]

2. Candidate consumption predictions are effectively identical:
   \[
   \cos(\gamma_i,\gamma_j)\approx 0.9999995.
   \]

3. Effective consumption is effectively identical:
   \[
   \cos(\alpha_i\gamma_i,\alpha_j\gamma_j)\approx 0.9999995.
   \]

4. Claim overlap is approximately:
   \[
   0.99998.
   \]

5. Residual mass decreases, but its distributional entropy is essentially unchanged:

   ```text
   rho0 entropy ≈ 2.4541769
   rho1 entropy ≈ 2.4541621
   rho2 entropy ≈ 2.4541190
   rho3 entropy ≈ 2.4540486
   ```

   while mean token residual falls approximately:

   ```text
   rho0 = 1.0000
   rho1 = 0.9498
   rho2 = 0.9023
   rho3 = 0.8575
   ```

   Therefore the model is not selectively removing semantic factors. It is approximately applying a shared multiplicative decay to the whole instruction.

6. `claim_swap` produces **exactly the same retrieval** as FULL:

   ```text
   FULL       = 32.33521084
   claim_swap = 32.33521084
   ```

   Therefore candidate-to-claim assignment is functionally irrelevant.

7. `frozen_residual` is slightly better than FULL:

   ```text
   FULL            = 32.33521084
   frozen_residual = 32.35165204
   ```

   Thus learned semantic consumption has not produced positive causal retrieval value.

8. `no_claim_firewall` is also essentially unchanged:

   ```text
   FULL              = 32.33521084
   no_claim_firewall = 32.36038710
   ```

9. `MEAN` and fixed repeated candidates remain competitive or better:

   ```text
   FULL          = 32.3352
   MEAN          = 32.4946
   REPEAT-1      = 32.5673
   REPEAT-3      = 32.7686
   ```

   The best repeat exceeds FULL by approximately:

   \[
   +0.4334 \text{ MR}.
   \]

10. Dynamic visual grounding remains almost perfectly cloned:

    ```text
    support cosine:
    t0 = 0.999590
    t1 = 0.999533
    t2 = 0.999456
    ```

11. Functional edits remain highly parallel:

    ```text
    Δq cosine:
    t0 = 0.979994
    t1 = 0.974759
    t2 = 0.962804

    ΔZ cosine:
    t0 = 0.979152
    t1 = 0.977334
    t2 = 0.975137
    ```

12. Late recurrent utility again becomes harmful:

    ```text
    selected target-relative gain:
    t0 = +0.053460
    t1 = +0.002671
    t2 = -0.020842
    ```

The branch therefore satisfies the R2 kill criteria.

---

# 2. Experiment identity and checkpoint provenance

## Branch

```text
exp/e2e-iag-srme-r2-semantic-residual
```

Canonical corrected R2 HEAD used for this experiment:

```text
9e4aca05f8f6de68b0de8d8f350ca282adda41f4
```

Relevant R2-v2 commits:

```text
589ff93  fix(r2): decouple consumption and preserve semantic mass
1fc8bc2  test(r2): audit consumption and semantic magnitude
9e4aca0  docs(r2): record corrected v2 residual contract
```

The corrected v2 design was necessary because the first R2 implementation coupled read strength and consumption strength:

\[
\alpha = \text{claim} = \text{consumption},
\]

which would have initialized:

\[
\rho:
1
\rightarrow
0.01
\rightarrow
10^{-4}
\rightarrow
10^{-6}.
\]

R2-v2 repaired this before the scientific run.

---

## Checkpoint

```text
outputs/2026-08-31/23-04-46/best.pt
```

Checkpoint epoch:

```text
1
```

Saved metric:

```text
32.33521083990733
```

Diagnostic replay:

```text
32.335210839907326
```

Absolute replay error:

```text
7.105427357601002e-15
```

Therefore:

\[
\boxed{
\text{the diagnostic exactly replays the intended checkpoint.}
}
\]

Replay guard:

```text
trusted_r2_replay = true
architecture_generation = r2_semantic_residual_claim_firewall_v2
fully_self_describing_model_config = true
```

---

# 3. Canonical R2-v2 configuration

Relevant resolved model configuration:

```text
backbone                         = qihoo360/fg-clip-base
backbone revision                = 454d76372c2cf5eb48fa0d871fd0534481484d97
training precision               = fp16

K                                = 4
Tmax                             = 3
width                            = 256
retrieval_dim                    = 512

query_cap                        = 1000
grounding_normalization          = entmax15

enable_dynamic_regrounding       = true
enable_dynamic_reproposal        = false
enable_dynamic_applicability     = false

enable_semantic_residual         = true

initial_claim_probability        = 0.99
initial_consumption_probability  = 0.05

claim_activation                 = sigmoid
consumption_activation           = sigmoid

residual_update_rule             = selected_claim_times_consumption
residual_initialization          = valid_token_ones
semantic_residual_fp32           = true
```

No R2 run mechanism includes:

```text
DPP
functional DPP
VISReg
RDMReg
variance-floor
teacher grounding
target-aware proposal
new STOP objective
RL
new planner
```

This is therefore still a clean structural R2 experiment.

---

# 4. R2-v2 mathematical contract

For valid content-token mask \(M\):

\[
\rho_{0,m}=M_m.
\]

Candidate \(k\) predicts semantic read/claim allocation:

\[
\alpha_{t,k,m}
=
\sigma(g^\alpha_{t,k,m}),
\]

and independent consumption/satisfaction:

\[
\gamma_{t,k,m}
=
\sigma(g^\gamma_{t,k,m}).
\]

Executable semantic weighting:

\[
w_{t,k,m}
=
\alpha_{t,k,m}\rho_{t,m}.
\]

Semantic direction:

\[
d_{t,k}
=
\frac{
\sum_mw_{t,k,m}T_m
}{
\sum_mw_{t,k,m}+\epsilon
}.
\]

Remaining semantic mass:

\[
s_{t,k}
=
\frac{
\sum_mw_{t,k,m}
}{
N_{\mathrm{valid}}+\epsilon
}.
\]

Magnitude-aware semantic content:

\[
c_{t,k}=s_{t,k}d_{t,k}.
\]

Thus:

\[
\rho\rightarrow0
\Longrightarrow
c\rightarrow0.
\]

The selected non-STOP candidate commits:

\[
\rho_{t+1,m}
=
\rho_{t,m}
\left(
1-\alpha_{t,a_t,m}\gamma_{t,a_t,m}
\right).
\]

STOP leaves \(\rho_t\) unchanged.

All K residual previews share the same parent \(\rho_t\).

---

# 5. Mechanical validation before training

The 100-step CUDA canary mechanically passed.

Hardware:

```text
NVIDIA GeForce RTX 5070 Ti
```

Canary summary:

```text
attempted optimizer steps     = 100
successful optimizer steps    = 96
skipped AMP-overflow steps    = 4
mechanical_status             = PASS

initial GradScaler            = 65536
final GradScaler              = 4096
minimum GradScaler            = 4096

finite                        = true
same-parent residual          = true
```

Required gradient fractions:

```text
grounding                         = 1.0000
residual_conditioned_intent       = 1.0000
semantic_claim_output             = 1.0000
semantic_consumption_output       = 1.0000
semantic_claim_hidden             = 0.9896
semantic_claim_query              = 0.9896
semantic_claim_state              = 0.9896
semantic_claim_token              = 0.9896
```

Thus the one-epoch failure cannot reasonably be dismissed as:

```text
dead claim head
dead gamma head
no grounder gradient
FP16 semantic quantization
same-parent bug
residual implementation bug
checkpoint replay bug
```

The branch is trainable and mechanically valid.

---

# 6. One-epoch scientific probe

Training command:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python src/train.py \
  dataset.root=data/FashionIQ \
  model=iag_srme_r2_semantic_residual \
  experiment=iag_srme_r2_semantic_residual \
  experiment.epochs=1 \
  protocol=fashioniq_original
```

Observed training result:

```text
epoch 1/1
train total     = 1.4392
validation MR   = 32.335
```

The point of the one-epoch probe was **not** to claim final benchmark convergence.

It was to answer a more important scientific question:

\[
\boxed{
\text{Does the R2 semantic residual begin learning candidate-specific responsibility?}
}
\]

The diagnostic answer is:

\[
\boxed{
\text{No.}
}
\]

---

# 7. Retrieval controls

Global validation retrieval:

| Control | Mean Recall |
|---|---:|
| FULL | **32.3352** |
| frozen residual | **32.3517** |
| claim swap | **32.3352** |
| no claim firewall | **32.3604** |
| residual shuffle | **31.6041** |
| mean candidate | **32.4946** |
| repeat 0 | 32.2573 |
| repeat 1 | **32.5673** |
| repeat 2 | 32.2007 |
| repeat 3 | **32.7686** |
| reference only | 12.8631 |

Important ratios:

```text
claim_swap / FULL        = 1.000000
frozen_residual / FULL   = 1.000508
no_firewall / FULL       = 1.000779
mean_candidate / FULL    = 1.004929
best_repeat / FULL       = 1.013403
residual_shuffle / FULL  = 0.977389
```

---

# 8. Causal interpretation of the controls

## 8.1 `claim_swap == FULL` is decisive

R2 requires candidate identity and candidate semantic responsibility to be coupled.

A candidate claim swap should change the executable semantic evidence associated with each action if:

\[
\alpha_{t,k}
\]

actually means:

> this semantic evidence belongs to candidate \(k\).

Instead:

```text
claim_swap MR = FULL MR
```

to numerical equality.

This is consistent with the measured claim cosine:

\[
\cos(\alpha_i,\alpha_j)\approx1.
\]

The candidate-to-claim mapping is therefore functionally interchangeable.

---

## 8.2 `frozen_residual >= FULL`

If selected semantic consumption is useful, then:

```text
FULL
```

should outperform:

```text
frozen_residual
```

at the same checkpoint because FULL has access to the supposedly useful semantic explaining-away state.

Observed:

```text
FULL            = 32.33521084
frozen_residual = 32.35165204
```

Gap:

\[
\boxed{
-0.01644 \text{ MR}
}
\]

in favor of freezing the residual.

This is not a large retrieval difference, but its direction is enough to reject any claim that semantic consumption has already become useful.

---

## 8.3 `no_claim_firewall ≈ FULL`

Observed:

```text
FULL              = 32.33521084
no_claim_firewall = 32.36038710
```

The firewall does not provide meaningful retrieval benefit at this checkpoint.

This is consistent with the fact that the learned claim mask is almost the full text anyway.

---

## 8.4 `residual_shuffle` hurts, but this does not rescue R2

Observed:

```text
FULL             = 32.3352
residual_shuffle = 31.6041
```

Gap:

\[
-0.7311 \text{ MR}.
\]

This proves that the residual state is **not numerically ignored**.

However, the correct interpretation is not:

> semantic explaining-away works.

Instead, together with the other controls it says:

\[
\boxed{
\text{the residual carries sample-specific information but not candidate-specific responsibility.}
}
\]

The model cares about the per-sample residual magnitude/state.

It does not care which candidate owns which semantic evidence.

That distinction is central.

---

## 8.5 `MEAN` and `REPEAT` remain competitive

Observed:

```text
FULL      = 32.3352
MEAN      = 32.4946
REPEAT-1  = 32.5673
REPEAT-3  = 32.7686
```

The architecture therefore still permits a shortcut solution where:

```text
multiple candidates
≈
interchangeable versions of one common edit process
```

rather than four meaningful alternatives.

---

# 9. Semantic claim diagnosis

At \(t=0\):

```text
claim mean mass        ≈ 11.918 tokens
claim effective size  ≈ 12.043 tokens
```

For a typical modification with roughly 12 valid content tokens, this means each candidate is effectively claiming nearly the entire instruction.

Candidate claim similarity:

```text
t0 claim cosine = 0.9999999953
t1 claim cosine = 0.9999999948
t2 claim cosine = 0.9999999954
```

Claim overlap:

```text
t0 ≈ 0.9999803
t1 ≈ 0.9999792
t2 ≈ 0.9999781
```

Therefore:

\[
\boxed{
\alpha_{t,0}
\approx
\alpha_{t,1}
\approx
\alpha_{t,2}
\approx
\alpha_{t,3}.
}
\]

The claim module has not learned semantic ownership.

---

# 10. Consumption / gamma diagnosis

Consumption probability at \(t=0\):

```text
mean   ≈ 0.05105
std    ≈ 0.00583
min    ≈ 0.03781
max    ≈ 0.06039
```

So gamma is not numerically frozen at the initialization value.

It learns some sample/token-dependent variation.

But it is not candidate-specific.

Candidate gamma cosine:

```text
t0 = 0.9999995215
t1 = 0.9999994898
t2 = 0.9999994652
```

Effective consumption cosine:

```text
t0 = 0.9999995330
t1 = 0.9999995043
t2 = 0.9999994786
```

Thus:

\[
\boxed{
\gamma_{t,0}
\approx
\gamma_{t,1}
\approx
\gamma_{t,2}
\approx
\gamma_{t,3}.
}
\]

and therefore:

\[
\boxed{
\alpha_{t,0}\gamma_{t,0}
\approx
\cdots
\approx
\alpha_{t,3}\gamma_{t,3}.
}
\]

The consumption state is effectively candidate-agnostic.

---

# 11. The strongest structural failure: residual entropy does not change

Residual token weights:

```text
rho0 mean = 1.000000
rho1 mean = 0.949768
rho2 mean = 0.902284
rho3 mean = 0.857489
```

So residual mass clearly decreases.

But residual distribution entropy:

```text
rho0 entropy mean = 2.4541769
rho1 entropy mean = 2.4541621
rho2 entropy mean = 2.4541190
rho3 entropy mean = 2.4540486
```

Change from rho0 to rho3:

\[
\Delta H
\approx
-0.0001283.
\]

This is negligible.

For an actual explaining-away mechanism, one would expect a semantic pattern such as:

```text
before:
red       1.0
sleeves   1.0
longer    1.0

after color action:
red       ↓ strongly
sleeves   remains
longer    remains
```

Instead the learned state behaves much closer to:

```text
before:
red       1.00
sleeves   1.00
longer    1.00

after action:
red       0.95
sleeves   0.95
longer    0.95
```

then:

```text
0.90 / 0.90 / 0.90
```

then:

```text
0.86 / 0.86 / 0.86
```

This is not semantic factor removal.

It is approximately **global instruction decay**.

Hence:

\[
\boxed{
\text{R2 residual is active but semantically non-selective.}
}
\]

---

# 12. WHAT diagnosis

Initial intent pairwise cosine:

```text
mean ≈ 0.965914
```

Temporal candidate chain:

```text
intent cosine:
t0 = 0.966024
t1 = 0.962437
t2 = 0.958496
```

This is one of the few positive observations.

The residual-conditioned text pathway does cause later WHAT representations to diverge somewhat.

Therefore R2 does not simply reproduce the exact R1c2 common-mode WHAT failure.

However:

\[
\boxed{
\text{representational WHAT divergence does not become semantic responsibility.}
}
\]

The claim intervention remains ineffective, WHERE remains cloned, and executable effects remain highly parallel.

This is another instance of the project's recurring lesson:

\[
\boxed{
\text{latent diversity is not enough; executable causal diversity is the criterion.}
}
\]

---

# 13. WHERE diagnosis

Dynamic current-state WHERE remains enabled.

Nevertheless:

```text
support cosine:
t0 = 0.999590
t1 = 0.999533
t2 = 0.999456
```

Initial support probability overlap:

```text
mean ≈ 0.991322
```

Per-candidate support mass:

```text
1.0, 1.0, 1.0, 1.0
```

The grounder is active, trainable and recomputed each timestep.

But the candidates still look at nearly the same visual evidence.

Therefore:

\[
\boxed{
\text{R2 semantic residual does not repair the moving-WHERE clone problem.}
}
\]

---

# 14. Context diagnosis

Pairwise context cosine:

```text
t0 mean ≈ 0.949729
t1 mean ≈ 0.944887
t2 mean ≈ 0.939384
```

Context features are less collapsed than WHERE itself.

However, this does not translate into sufficiently distinct executable edits.

The chain is:

```text
claim/gamma almost identical
        ↓
residual approximately global decay
        ↓
WHAT modestly diverges
        ↓
WHERE almost identical
        ↓
context modestly diverges
        ↓
ΔZ / Δq remain strongly parallel
```

---

# 15. Token-space functional effects

All candidate effects are active.

There is no dead-effect explanation.

Mean \(\Delta Z\) norms:

```text
t0 = 3.2695
t1 = 3.3159
t2 = 3.3438
```

Thus token-space editor activity remains strong across recurrence.

But pairwise \(\Delta Z\) cosine:

```text
t0 = 0.979152
t1 = 0.977334
t2 = 0.975137
```

and effective rank:

```text
t0 ≈ 1.905
t1 ≈ 1.936
t2 ≈ 1.971
```

Maximum candidate-axis rank is 4.

Therefore:

\[
\boxed{
\text{the editor produces four active but highly redundant token edits.}
}
\]

---

# 16. Retrieval-space functional effects

Mean \(\Delta q\) norms:

```text
t0 = 0.347364
t1 = 0.274830
t2 = 0.194857
```

Late-step retention:

```text
t1 / t0 = 0.7912
t2 / t0 = 0.5610
t2 / t1 = 0.7090
```

This roughly preserves the healthy recurrent effect-survival pattern found after R1a.

So R2 does **not** reintroduce the original R0 readout attenuation catastrophe.

But candidate effects remain highly aligned:

```text
Δq cosine:
t0 = 0.979994
t1 = 0.974759
t2 = 0.962804
```

Effective rank:

```text
t0 = 1.8794
t1 = 1.9688
t2 = 2.1347
```

The late rank improves somewhat, but it remains far below 4 and must be interpreted together with the cosine and retrieval controls.

This is not a dead-effect problem.

It is still:

\[
\boxed{
\text{live-but-redundant functional editing.}
}
\]

---

# 17. Same-parent counterfactual retrieval

Same-parent mean candidate query:

```text
t0 = 21.2673
t1 = 28.8849
t2 = 32.4878
```

Offline best-candidate oracle:

```text
t0 = 21.8686
t1 = 29.5746
t2 = 33.2014
```

Thus recurrent depth still provides substantial retrieval improvement:

\[
21.27
\rightarrow
28.88
\rightarrow
32.49.
\]

This is consistent with the earlier R1a conclusion:

\[
\boxed{
\text{multi-step recurrence itself is not the fundamental failure.}
}
\]

The recurrent state continues to accumulate useful retrieval information.

The failure is that the **candidate decomposition and policy semantics are not meaningful**.

This distinction should survive any future architecture redesign.

---

# 18. Selected target-relative utility

Offline selected-path target similarity change:

```text
t0 = +0.0534599
t1 = +0.0026712
t2 = -0.0208417
```

The second action is almost neutral.

The third action is actively harmful on average.

Therefore:

\[
\boxed{
\text{R2 does not solve late over-edit.}
}
\]

The pattern remains:

```text
first edit:
strongly useful

second edit:
barely useful

third edit:
harmful
```

Yet STOP behavior does not halt aggressively enough.

---

# 19. Policy diagnosis

Validation selection:

```text
candidate conditional shares:
C0 = 20.93%
C1 = 25.30%
C2 = 34.00%
C3 = 19.77%
```

No single candidate monopoly.

This is important because the failure is not:

```text
one slot wins everything
```

Instead:

```text
all candidate IDs are used
while
their semantic claims/effects remain largely interchangeable
```

That is a stronger form of collapse than raw selection monopoly.

Repeated-selection trajectory fraction:

```text
0.9921
```

Mean executed edit count:

```text
2.9772 / 3
```

New STOP hazard:

```text
t0 ≈ 0.00694
t1 = 0
t2 ≈ 0.001997
```

The policy almost always executes the full horizon even though offline utility is negative at t2.

Thus the branch retains the late STOP/objective mismatch observed in earlier experiments.

---

# 20. Why this is a genuine negative result rather than an implementation failure

The following possible confounds were explicitly eliminated:

## Not checkpoint mismatch

Replay error:

```text
~7e-15
```

## Not wrong branch config

Replay verifies:

```text
query_cap=1000
dynamic WHERE ON
dynamic reproposal OFF
applicability OFF
semantic residual ON
gamma head present
FP32 residual
```

## Not dead gradients

All semantic residual parameter families receive gradients after the designed warm-start behavior.

## Not alpha/gamma initialization catastrophe

R2-v2 initializes:

\[
\alpha=0.99,\qquad\gamma=0.05,
\]

so the first action retains approximately:

\[
0.9505
\]

of semantic evidence rather than 0.01.

## Not residual magnitude normalization-away

Magnitude-aware semantic content satisfies:

\[
\rho\rightarrow0
\Rightarrow
c\rightarrow0.
\]

## Not dead recurrent effects

All \(\Delta Z\) and \(\Delta q\) candidates are active.

## Not candidate usage monopoly

All four candidate identities are selected.

Therefore the negative result is structurally meaningful.

---

# 21. Causal failure chain

The observed R2-v2 failure can be summarized as:

```text
full modification tokens
        ↓
shared candidate claim module
        ↓
alpha_k ≈ alpha_j for every candidate pair
        ↓
gamma_k ≈ gamma_j for every candidate pair
        ↓
alpha_k * gamma_k ≈ common consumption mask
        ↓
selected action reduces nearly all valid text tokens similarly
        ↓
rho behaves approximately like global instruction decay
        ↓
future candidates see lower-magnitude text
but not a different semantic subproblem
        ↓
WHAT diverges modestly due candidate queries / recurrent state
        ↓
WHERE remains ~0.9995 cosine
        ↓
contexts remain highly correlated
        ↓
ΔZ remains ~0.975–0.979 cosine
        ↓
Δq remains ~0.963–0.980 cosine
        ↓
candidate identity remains mostly interchangeable
        ↓
MEAN / REPEAT remain competitive
        ↓
claim swap has zero retrieval effect
        ↓
late steps over-edit
```

The key finding is:

\[
\boxed{
\text{representing a residual variable is not sufficient to create semantic responsibility.}
}
\]

---

# 22. Hypothesis verdict

Original R2 hypothesis:

> an explicit token-level remaining-evidence state plus selected candidate claim consumption is sufficient to make future proposals solve different unresolved semantic factors.

Verdict:

\[
\boxed{
\textbf{REJECTED under the current architecture and objective.}
}
\]

More precise statement:

\[
\boxed{
\text{The residual state is used, but it collapses into near-global attenuation instead of candidate-specific explaining-away.}
}
\]

---

# 23. R2 PASS criteria vs observed result

## Criterion A — residual changes

Required:

```text
rho changes nontrivially
```

Observed:

```text
PASS
```

But this alone is insufficient.

---

## Criterion B — residual changes selectively

Required:

```text
executing candidate k preferentially consumes semantic evidence associated with k
```

Observed:

```text
FAIL
```

Evidence:

```text
claim cosine ≈ 1
gamma cosine ≈ 1
effective-consumption cosine ≈ 1
residual entropy almost constant
```

---

## Criterion C — claim intervention changes execution

Required:

```text
claim_swap changes retrieval/effects
```

Observed:

```text
FAIL
```

```text
claim_swap == FULL
```

---

## Criterion D — FULL beats frozen residual

Required:

```text
FULL > frozen_residual
```

Observed:

```text
FAIL
```

Frozen residual is slightly better.

---

## Criterion E — clone controls weaken

Required:

```text
MEAN / REPEAT should become less competitive if candidate responsibilities matter
```

Observed:

```text
FAIL
```

MEAN and best REPEAT outperform FULL.

---

## Criterion F — WHERE becomes candidate-specific

Required:

```text
later candidate WHERE should respond differently to different remaining semantic problems
```

Observed:

```text
FAIL
```

Support cosine remains approximately 0.9995.

---

## Criterion G — executable effects become meaningfully distinct

Required:

```text
candidate ΔZ / Δq should separate functionally
```

Observed:

```text
FAIL
```

Effects remain active but highly parallel.

---

## Criterion H — recurrence utility remains healthy

Required:

```text
do not destroy R1a recurrent effect survival
```

Observed:

```text
PARTIAL PASS
```

Same-parent retrieval improves strongly with depth and \(\Delta q\) remains active.

However selected t2 utility is negative.

---

# 24. What R2 teaches us

Even though R2 is negative, it adds several useful scientific conclusions.

## Lesson 1 — residual magnitude alone is not semantic memory

A network can use:

\[
\rho_t
\]

without learning:

\[
\text{what specifically remains unresolved}.
\]

The current residual mostly represents:

```text
how much instruction remains
```

not:

```text
which semantic requirement remains.
```

---

## Lesson 2 — candidate-conditioned heads can still learn candidate-invariant functions

The claim module sees candidate query \(q_k\), text tokens and current state.

Yet training converges to:

\[
f(q_0,T,Z)
\approx
f(q_1,T,Z)
\approx
f(q_2,T,Z)
\approx
f(q_3,T,Z).
\]

Simply providing candidate identity/state access does not force semantic specialization.

This repeats the broader lesson from R1c1/R1c2:

\[
\boxed{
\text{state access does not imply specialization.}
}
\]

---

## Lesson 3 — structural bottlenecks need an objective reason to specialize

The current retrieval objective can be solved without assigning different semantic responsibilities to candidates.

Therefore the network has no strong incentive to make:

```text
candidate 0 = color
candidate 1 = length
candidate 2 = material
candidate 3 = shape
```

or any other stable decomposition.

The easiest solution remains:

```text
all candidates approximate the whole modification
```

with small candidate perturbations.

---

## Lesson 4 — recurrent depth is still worth preserving

Despite failure of candidate decomposition:

```text
same-parent MR:
21.27 → 28.88 → 32.49
```

So the future redesign should not casually discard stateful multi-step refinement.

The component worth retaining is:

\[
\boxed{
\text{recurrent state evolution}
}
\]

not necessarily:

\[
\boxed{
\text{the current four-candidate planner/decomposer machinery}.
}
\]

---

# 25. Relationship to previous causal ladder

Current cumulative picture:

## R0

Found:

```text
token recurrence alive
retrieval recurrence attenuated by cumulative query cap
candidate redundancy present
```

## R1a

Removed query cap bottleneck.

Result:

```text
PASS
multi-step retrieval effect survives
candidate redundancy remains
```

## R1b

Added dynamic applicability / visual NULL.

Result:

```text
NEGATIVE
scalar WHETHER does not solve candidate semantics
```

## R1c1

Recomputed WHERE from current state.

Result:

```text
NEGATIVE
dynamic WHERE creates moving-WHERE clones
```

## R1c2

Re-proposed WHAT from current state and full text.

Result:

```text
NEGATIVE
dynamic WHAT creates moving-WHAT clones
and is causally harmful relative to frozen WHAT
```

## R2

Added semantic residual + claim/consumption firewall.

Result:

```text
NEGATIVE
semantic residual becomes global attenuation
candidate responsibility remains cloned
```

The causal ladder now points away from local patching and toward architectural simplification/redesign.

---

# 26. What NOT to do on this branch

Do **not** continue by adding:

```text
DPP
functional DPP
RDMReg
VISReg
variance floor
orthogonality
Hungarian ownership
slot balancing
teacher grounding
target-supervised claim labels
new STOP loss
RL
```

into R2 and then reinterpret the experiment.

Doing that would destroy the clean conclusion.

Do not:

```text
train 19 more epochs hoping cosine suddenly separates
retune gamma until it visually looks diverse
force alpha entropy lower
force candidates to claim disjoint tokens
force 25% usage
```

without opening a new experiment with a new hypothesis.

R2 should be frozen as:

\[
\boxed{
\text{mechanically valid negative result}.
}
\]

---

# 27. Why not proceed directly to R3 Functional DPP

The original ladder made R3 conditional on R2 establishing real semantic/functional responsibility.

That gate has not been met.

Applying DPP now risks forcing diversity into:

```text
four candidates that still do not correspond to meaningful semantic responsibilities
```

which can produce:

```text
geometrically different actions
without
semantically useful actions.
```

This project has already observed analogous failures repeatedly.

Therefore:

\[
\boxed{
\text{R3 should NOT automatically follow this R2 result.}
}
\]

---

# 28. Recommended project decision after R2

The accumulated evidence now supports a larger redesign.

The current information path contains many successive bottlenecks:

```text
text
  ↓
candidate query
  ↓
WHAT
  ↓
semantic claim / residual
  ↓
WHERE
  ↓
grounded read
  ↓
context fusion
  ↓
shared editor
  ↓
token update
  ↓
retrieval readout
  ↓
scorer
  ↓
selector / STOP
```

Each stage can absorb candidate-specific information into a common solution.

The repeated experiments show:

```text
WHETHER did not fix it
dynamic WHERE did not fix it
dynamic WHAT did not fix it
semantic residual did not fix it
```

The next redesign should therefore begin from a stronger composition trunk with fewer semantic bottlenecks.

A reasonable architectural principle is:

```text
reference visual tokens Z_t
        +
high-bandwidth modification tokens T
        ↓
direct text-conditioned local updater
        ↓
Z_{t+1}
        ↓
direct retrieval readout
```

First establish:

```text
strong single-path composition
strong retrieval
healthy recurrent improvement
```

Then add branching/planning only if same-parent oracle evidence proves that branching provides additional value.

---

# 29. Components worth preserving in a redesign

Keep or reuse:

```text
FG-CLIP feature interface
token-level visual state
query_cap repair / unrestricted recurrent retrieval displacement
same-parent counterfactual diagnostics
target firewall
checkpoint replay guard
dynamic diagnostic suite
offline target-relative utility
recurrent depth as a tested useful primitive
```

Do not automatically preserve:

```text
current 4-candidate WHAT pipeline
current claim/residual module
current candidate selector
current shared decomposition stack
current multi-stage semantic compression path
```

These have not earned retention.

---

# 30. Final branch verdict

Mechanical status:

```text
PASS
```

Scientific status:

```text
NEGATIVE RESULT
```

Primary failure mode:

\[
\boxed{
\text{semantic residual collapse into global attenuation}
}
\]

Secondary persistent failure modes:

\[
\boxed{
\text{claim clones}
}
\]

\[
\boxed{
\text{gamma / consumption clones}
}
\]

\[
\boxed{
\text{moving-WHERE clones}
}
\]

\[
\boxed{
\text{live-but-redundant ΔZ / Δq effects}
}
\]

\[
\boxed{
\text{late over-edit}
}
\]

Most concise causal conclusion:

\[
\boxed{
\text{R2 changes HOW MUCH text remains, but not WHICH semantic requirement remains.}
}
\]

Therefore:

\[
\boxed{
\textbf{STOP R2. FREEZE THE BRANCH. DO NOT RUN THE FULL 20-EPOCH EXPERIMENT.}
}
\]

---

# 31. Reproduction commands

## One-epoch probe

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python src/train.py \
  dataset.root=data/FashionIQ \
  model=iag_srme_r2_semantic_residual \
  experiment=iag_srme_r2_semantic_residual \
  experiment.epochs=1 \
  protocol=fashioniq_original
```

## Diagnostic

```bash
RUN=outputs/2026-08-31/23-04-46

CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python src/diagnose_iag_srme_checkpoint.py \
  --checkpoint "$RUN/best.pt" \
  --dataset-root data/FashionIQ \
  --protocol fashioniq_original \
  --batch-size 32 \
  --gallery-batch-size 128 \
  --output reports/r2_semantic_residual_epoch1.json
```

Expected replay:

```text
checkpoint epoch = 1
architecture_generation = r2_semantic_residual_claim_firewall_v2
trusted_r2_replay = true
FULL MR = 32.335210839907326
```

---

# 32. Frozen scientific statement for future checkpoints

Use the following statement when handing off this project:

> **R2 semantic residual / claim firewall is a mechanically valid but scientifically negative result.** The corrected v2 mechanism successfully preserves semantic magnitude, has healthy gradients, uses the residual state, and passes same-parent/target-firewall/replay checks. However, after a full one-epoch probe, candidate claims, consumption probabilities, and effective consumption remain virtually identical across all four candidates; residual entropy remains almost unchanged while total mass decays, demonstrating near-global instruction attenuation rather than selective semantic explaining-away. `claim_swap` is exactly retrieval-equivalent to FULL, `frozen_residual` and `no_claim_firewall` are not worse, and MEAN / REPEAT controls remain competitive. Dynamic WHERE remains approximately 0.9995 cosine and executable ΔZ/Δq effects remain highly parallel. Recurrence itself still provides useful same-parent retrieval improvement, but late selected utility becomes negative. R2 therefore fails its causal promotion criteria and should be frozen rather than rescued with additional regularizers. The next step should be an architectural simplification/redesign rather than mechanism stacking.

---

# 33. Research ledger update

```text
R0    diagnostic audit                         DONE
R1a   remove global query cap                  PASS
R1b   dynamic applicability / visual NULL      NEGATIVE
R1c1  dynamic current-state WHERE              NEGATIVE
R1c2  dynamic current-state WHAT               NEGATIVE
R2    semantic residual / claim firewall       NEGATIVE
R3    functional DPP                           DO NOT AUTO-PROMOTE
R4    target-privileged grounding teacher      HOLD
R5    planning / STOP                          HOLD
```

Current architecture-level conclusion:

\[
\boxed{
\text{the dominant failure is no longer plausibly a single missing gate, grounding refresh, proposal refresh, or residual variable.}
}
\]

The evidence increasingly supports:

\[
\boxed{
\text{the current candidate decomposition pipeline itself is the bottleneck.}
}
\]
