# CIR / TAPER A3.4 — Functional Error Ownership
## Negative-Result Diagnostic README

**Date:** 2026-08-27  
**Branch:** `exp/e2e-a3.4-functional-error-ownership`  
**Frozen implementation commit:** `7756148772634d69acf513e0c3cbc0f85d501faa` (`v2.1`)  
**Status:** **NEGATIVE RESULT — functional specialization not achieved**  
**Next experimental branch:** recommended `exp/e2e-a3.4b-conditional-residual-credit`

---

# 1. Purpose of this checkpoint

This document freezes the scientific state of A3.4 before moving to the next branch.

A3.4 was created after the A3.3 2×2 ablation showed a sharp separation between:

- **representation / endpoint retrieval strength**, and
- **functional slot specialization**.

The A3.3 Run C baseline was chosen because it was strong at retrieval while still showing very clear functional collapse:

```yaml
slot_value_source: contextual
slot_effect_in_value: false
slot_value_assignment: soft_shared
```

Approximate Run C baseline:

```text
Mean Recall                    ≈ 58.75
Gradient error-mode rank       ≈ 2.84
Functional Phi rank            ≈ 1.00
Median SINGLE/FULL             ≈ 0.777
Median REPEAT/FULL             ≈ 1.020
Median MEANxK/FULL             ≈ 1.000
Mean K95 / K99                 ≈ 2.56 / 3.01
```

The A3.4 hypothesis was:

> If retrieval error is represented as multiple per-negative functional modes, and slot credit is assigned in functional error space rather than token/latent geometry, then Edit Slots may begin to specialize into different useful retrieval-error functions.

The implementation intentionally kept the Run C representation and executor stack fixed.

---

# 2. Scientific scope of A3.4

A3.4 was intended to probe the theory around:

```text
§6.4 Multi-error retrieval representation
§6.5 Functional mode assignment
§6.6 Block residual pursuit
```

The experiment did **not** attempt to solve the full TAPER-MERIT theory.

It kept unchanged:

```text
contextual VALUE
competitive token ownership
QASA
routing
Primitive Bank
Executor
teacher
retrieval objective
FashionIQ protocol
optimizer
training schedule
```

The new mechanisms were limited to the functional-credit side.

---

# 3. A3.4 v2.1 mechanism

The final A3.4 v2.1 implementation used:

```text
hard negatives
    ↓
per-negative pairwise retrieval errors
    ↓
task error-mode rank
    ↓
finite singleton functional effects Phi[s,j]
    ↓
rank-gated functional mode assignment B[s,j]
    ↓
upper mode capacity + NULL/unassigned modes
    ↓
block Gram-Schmidt residual acceptance
    ↓
pair synergy lookahead
    ↓
functional auxiliary loss
```

The per-negative error was of the form:

\[
\ell_j(q)
=
\operatorname{softplus}
\left(
\frac{
s(q,y_j^-)-s(q,y^+) + m
}{\tau}
\right).
\]

The finite singleton effect was:

\[
\Phi_{s,j}
=
\ell_j(q_{\emptyset})
-
\ell_j(q_s).
\]

Positive \(\Phi_{s,j}\) means slot \(s\) improves retrieval error mode \(j\).

---

# 4. Functional-credit implementation details

A3.4 v2.1 introduced two distinct ownership objects.

## 4.1 Training assignment

A first-round proposal was stored as:

```python
proposal_assignment
```

and then used as:

```python
training_assignment = proposal_assignment
training_credit = effects * training_assignment
```

This assignment was what actually drove the auxiliary functional loss.

## 4.2 Residual / uniqueness assignment

The implementation then performed iterative block residualization over the positive effect rows.

After selecting a block, the row span was projected out using a Gram-Schmidt style update.

This produced:

```python
assignment
credit
credited_mask
unique_mode_coverage
redundant_credit_fraction
unresolved_multimode
```

These quantities represented the residual-accepted / more unique functional structure.

**Critical fact discovered by this experiment:**

> The residual-accepted structure was mostly diagnostic. The actual training objective continued to use the first-round `training_assignment`.

This distinction becomes the central diagnosis of the negative result.

---

# 5. Auxiliary gradient isolation

A3.4 also added an auxiliary-only credit isolation path.

The main Run C forward was kept numerically exact while auxiliary gradients were routed through a recomputed isolated path:

```python
recomputed_slots = torch.cat(isolated_slots, dim=1)

result = output["edit_slots"].detach() + (
    recomputed_slots - recomputed_slots.detach()
)
```

Therefore:

```text
FORWARD:
functional auxiliary slots == Run C edit_slots

BACKWARD:
gradient flows through isolated recomputation
```

This fixed the earlier AMP mismatch runtime error and prevented direct functional-gradient leakage into competitor slot-query rows through the slot-axis softmax denominator.

This part of the implementation is considered **working plumbing**, not the source of the final scientific failure.

---

# 6. Training command

The final A3.4 v2.1 experiment was run with:

```bash
python src/train.py experiment=taper_e2e \
  experiment.model.slot_value_source=contextual \
  experiment.model.slot_effect_in_value=false \
  experiment.model.slot_value_assignment=soft_shared \
  experiment.functional_ownership.enabled=true
```

The run completed all 10 epochs.

---

# 7. Training trajectory

Selected epochs:

## Epoch 1

```text
Mean Recall       51.6844
active_slots      2.95
hard_active       2.76
qasa_k            2.76
value_k           2.95

func_loss        -0.305
func_rank         2.48
func_slots        3.31
func_coverage     0.700
func_redundant    0.820
func_k            2.92
func_owned       13.91
func_unresolved   0.858
```

## Epoch 4

```text
Mean Recall       58.3466
active_slots      2.36
hard_active       2.02
qasa_k            2.02
value_k           2.36

func_loss        -0.399
func_rank         2.19
func_slots        3.37
func_coverage     0.781
func_redundant    0.843
func_k            2.61
func_owned       15.43
func_unresolved   0.838
```

## Epoch 7 — best checkpoint

```text
Mean Recall       58.6448

active_slots      1.00
hard_active       1.01
dominant          0.806
qasa_k            1.01
value_k           1.00
value_dominant    1.000
value_empty       0.750

func_loss        -0.375
func_rank         2.15
func_slots        3.16
func_residual    12.64
func_coverage     0.811
func_redundant    0.793
func_pair         0.056
func_k            2.56
func_owned       15.56
func_unowned      0.08
func_giant        0.001
func_unresolved   0.836
```

## Epoch 9

```text
Mean Recall       58.0483
active_slots      1.00
hard_active       1.00
qasa_k            1.00
value_k           1.00
value_dominant    1.000

func_slots        3.06
func_k            2.56
func_owned       15.55
func_unresolved   0.836
```

---

# 8. Key training-time contradiction

The strongest diagnostic pattern is:

```text
deployed path:
qasa_k  ≈ 1
value_k ≈ 1

functional bookkeeping:
func_slots ≈ 3
func_k     ≈ 2.5
func_owned ≈ 15.5 / 16
```

At the same time:

```text
func_redundant  ≈ 0.77–0.84
func_unresolved ≈ 0.82–0.86
```

This means:

> The functional assignment system successfully distributed error-mode IDs across slots on paper, but the residual/finite functional evidence still judged the resulting slot effects to be heavily redundant and unresolved.

This is not a simple “loss disabled” bug.

The functional auxiliary term was active.

For example at epoch 7:

```text
total loss      = 0.4638
functional loss = -0.375
lambda_func     = 0.1
```

Thus the functional term contributed non-trivially to optimization.

The failure is therefore **credit semantics / training target mismatch**, not absence of gradient.

---

# 9. Frozen P0 audit

The full frozen-checkpoint P0 audit used all 6016 validation samples.

Result:

```text
==============================================================================
TAPER-MERIT P0 FUNCTIONAL INTERVENTION AUDIT
==============================================================================

Samples:                          6016
Hard partition mean K:           1.000
Dominant hard token share:       1.000

Gradient error-mode rank:        2.842
Functional Phi effective rank:   1.138
Functionally useful slots:       3.936

QASA functional precision:       0.974
QASA functional recall:          0.245

Median best SINGLE/FULL:         0.952
Median best REPEAT/FULL:         1.089
Median MEANxK/FULL:              1.001

Mean K95:                        1.745
Mean K99:                        2.082

qasa_full:
R@10                             46.79
R@50                             70.47
Mean Recall                      58.63

all_slots_full:
R@10                             46.50
R@50                             70.45
Mean Recall                      58.47

reference_only:
R@10                              4.24
R@50                             11.91
Mean Recall                       8.07
```

Report:

```text
reports/a3_4_v21_p0_full.json
```

---

# 10. Comparison against Run C

| Metric | Run C | A3.4 v2.1 | Interpretation |
|---|---:|---:|---|
| Mean Recall | ~58.75 | 58.63 | endpoint retrieval essentially unchanged |
| Gradient error rank | ~2.84 | 2.842 | task remains genuinely multi-directional |
| Phi effective rank | ~1.00 | 1.138 | only a weak increase; still near rank-1 |
| SINGLE/FULL | ~0.777 | 0.952 | much worse; one slot nearly explains FULL |
| REPEAT/FULL | ~1.020 | 1.089 | worse; repeated single slot beats FULL more strongly |
| MEANxK/FULL | 1.000 | 1.001 | clone/interchangeability signature remains |
| K95 | ~2.56 | 1.745 | fewer components needed |
| K99 | ~3.01 | 2.082 | fewer components needed |
| Hard partition K | ~2.08 | 1.000 | deployed ownership collapses completely |

---

# 11. Main scientific conclusion

The decisive geometry is:

\[
r_{\text{task}} \approx 2.842
\]

while

\[
r_{\Phi} \approx 1.138.
\]

Therefore:

\[
\boxed{
\text{task error geometry is multi-directional}
\;\not\Rightarrow\;
\text{learned slot interventions become multi-directional}
}
\]

The retrieval task exposes multiple independent error directions.

The learned Edit Slots still produce nearly rank-1 functional effects.

---

# 12. Clone / compute-ticket evidence

The strongest intervention results are:

```text
SINGLE/FULL ≈ 0.952
REPEAT/FULL ≈ 1.089
MEANxK/FULL ≈ 1.001
```

## SINGLE

A single best slot recovers roughly 95% of FULL gain.

This is much closer to a giant/global functional owner than a compositional decomposition.

## REPEAT

Repeating one slot recovers more than FULL:

```text
REPEAT/FULL > 1
```

This is strong evidence that the Executor can exploit repeated similar slot computations as useful compute tickets.

## MEAN

Replacing slot identities by a mean representation and repeating it preserves essentially all FULL gain:

```text
MEANxK/FULL ≈ 1
```

This strongly rejects genuine slot identity dependence.

---

# 13. “Functionally useful slots ≈ 3.936” is not specialization

P0 reports:

```text
Functionally useful slots ≈ 3.936
```

This must **not** be interpreted as four specialized modules.

Multiple slots can all have positive marginal utility while their functional effect rows remain highly collinear.

The correct distinction is:

\[
\boxed{
\text{positive marginal usefulness}
\neq
\text{unique functional responsibility}
}
\]

A3.4 reinforces this result.

---

# 14. QASA collapse is real but not the primary causal diagnosis

P0:

```text
Hard partition mean K       = 1.000
Dominant hard token share   = 1.000
```

Training also showed:

```text
qasa_k  → ~1
value_k → ~1
```

So the deployed token/slot selector collapsed.

However this should **not** be treated as the only explanation.

Even when all slots are forced into intervention analysis, P0 still gives:

```text
Phi rank      = 1.138
SINGLE/FULL   = 0.952
REPEAT/FULL   = 1.089
MEANxK/FULL   = 1.001
```

Therefore the failure is not merely:

> “QASA selected too few slots.”

The slot functions themselves remain highly redundant.

---

# 15. Code-level root cause

The central code-level issue is the split between:

```python
training_assignment
```

and:

```python
assignment / credit
```

The current training path uses:

```python
training_assignment = proposal_assignment
training_credit = effects * training_assignment
```

where `proposal_assignment` is frozen from the first ownership round.

Then:

```python
functional_loss = functional_credit_loss(
    singleton_objective,
    ownership["training_credit"],
    ...
)
```

Therefore the auxiliary loss optimizes the **initial mode assignment**.

The later block-residual process can discover that assigned slots are redundant, but that result does not fully replace the training credit.

Conceptually, A3.4 does:

```text
initial Phi from EMPTY
       ↓
assign job IDs
       ↓
train those assignments
       ↓
residual audit says:
"many of these are still redundant"
       ↓
mostly diagnostic
```

What the theory intended for block residual pursuit is closer to:

```text
A = EMPTY
    ↓
measure conditional Phi[s,j | A]
    ↓
assign residual jobs
    ↓
credit one accepted block
    ↓
A ← A ∪ block
    ↓
recompute conditional Phi[s,j | A]
    ↓
assign only genuinely remaining work
    ↓
credit next block
```

This difference is now empirically important.

---

# 16. Hypotheses tested by A3.4

## H-A3.4-1

> Per-negative retrieval error modes plus first-round functional assignment are sufficient to create functional slot specialization.

**Status:** REJECTED.

Evidence:

```text
Phi rank = 1.138
REPEAT/FULL = 1.089
MEANxK/FULL = 1.001
```

## H-A3.4-2

> Giving different slot IDs different hard-negative mode assignments is enough to produce different functional modules.

**Status:** STRONGLY REJECTED.

Training:

```text
func_owned ≈ 15.5 / 16
func_slots ≈ 3
```

yet frozen P0:

```text
Phi rank ≈ 1.14
```

Therefore:

\[
\boxed{
\text{different assigned error IDs}
\neq
\text{different learned functions}
}
\]

## H-A3.4-3

> Residual uniqueness analysis used primarily as an acceptance diagnostic is enough to steer learning.

**Status:** REJECTED / NOT SUPPORTED.

Residual diagnostics stayed strongly unresolved:

```text
func_redundant  ≈ 0.77–0.84
func_unresolved ≈ 0.82–0.86
```

while specialization did not emerge.

## H-A3.4-4

> Retrieval performance must fall substantially if functional specialization pressure is added.

**Status:** REJECTED.

Mean Recall stayed essentially unchanged:

```text
Run C     ≈ 58.75
A3.4 v2.1 = 58.63
```

This is useful because it shows the failure is not primarily catastrophic optimization instability.

---

# 17. What A3.4 did successfully establish

Although A3.4 failed its main specialization goal, it produced useful scientific information.

It established that:

1. the task has multiple error directions;
2. per-negative functional modes can be computed stably enough for training;
3. finite intervention Phi is usable as a functional diagnostic;
4. gradient-isolated auxiliary slot interventions can be implemented without changing Run C forward values;
5. functional assignment can be separated from token routing;
6. fake assignment diversity can coexist with rank-1 functional behavior;
7. residual redundancy metrics correctly warn about unresolved collapse;
8. endpoint Recall can remain strong while functional decomposition is absent.

These components should not all be discarded.

---

# 18. What must NOT be concluded

Do **not** conclude:

```text
multi-error functional modeling is useless
block residual pursuit is useless
contextual VALUE can never specialize
QASA alone caused the entire failure
more functional loss weight would solve the problem
four slots are unnecessary for every sample
the task itself is rank-1
```

The experiment does not support those claims.

The task error rank remains around 2.84.

The experiment specifically falsifies the current **credit realization**.

---

# 19. What must NOT be changed next

The next branch should not immediately add unrelated mechanisms.

Do not change:

```text
contextual VALUE
teacher
QASA
routing
Executor
Primitive Bank
retrieval loss
FashionIQ protocol
optimizer
training schedule
```

Do not add:

```text
local microencoder
teacher_raw VALUE
Entmax
Sinkhorn
OT
balance loss
cosine diversity
orthogonality
new semantic supervision
new data
LLM-generated labels
new solver architecture
```

The next experiment should preserve attribution.

---

# 20. Recommended next branch

Freeze A3.4 v2.1 as a negative-result branch.

Recommended next branch:

```bash
git checkout exp/e2e-a3.4-functional-error-ownership
git checkout -b exp/e2e-a3.4b-conditional-residual-credit
```

The next hypothesis should change **only the functional training-credit semantics**.

---

# 21. A3.4b hypothesis

A3.4b should test:

> Does conditional residual functional credit, recomputed after accepted coalition steps, produce real slot specialization where first-round mode assignment failed?

Target logic:

```text
A0 = EMPTY
    ↓
compute Phi[s,j | A0]
    ↓
select / assign useful residual block
    ↓
credit that block
    ↓
A1 = A0 ∪ block
    ↓
compute Phi[s,j | A1]
    ↓
select next residual block
    ↓
credit only remaining work
    ↓
...
```

The key change is:

\[
\boxed{
\text{residual analysis must determine training credit}
}
\]

rather than:

\[
\text{residual analysis mostly diagnoses a fixed first-round training assignment}.
\]

---

# 22. Acceptance criteria for the next branch

The next experiment should still use the same frozen P0 acceptance metrics.

Primary specialization indicators:

```text
Functional Phi rank      ↑ materially from ~1
REPEAT/FULL              ↓ clearly below current level
MEANxK/FULL              ↓ below ~1 when identity matters
SINGLE/FULL              ↓ on verified multi-mode samples
K95 / K99                ↑ when multiple functional components are necessary
```

Secondary endpoint constraint:

```text
Mean Recall should remain usable and not catastrophically collapse.
```

Do not optimize these diagnostics directly.

They remain acceptance metrics.

---

# 23. Frozen benchmark reference for future comparison

Use this A3.4 v2.1 checkpoint as a negative control:

```text
A3.4 v2.1

Mean Recall                  58.63
Gradient rank                 2.842
Phi rank                      1.138
Useful slots                  3.936

SINGLE/FULL                   0.952
REPEAT/FULL                   1.089
MEANxK/FULL                   1.001

K95                           1.745
K99                           2.082

Hard partition K              1.000
Dominant hard token share     1.000

QASA precision                0.974
QASA recall                   0.245
```

Future branches must not claim specialization unless they beat this profile on the functional metrics, not merely Recall.

---

# 24. Final verdict

A3.4 v2.1 should be recorded as:

\[
\boxed{
\textbf{NEGATIVE RESULT}
}
\]

Specifically:

\[
\boxed{
\text{first-round functional mode assignment}
+
\text{residual diagnostic}
\not\Rightarrow
\text{functional slot specialization}
}
\]

The task remains multi-error:

\[
r_{\text{task}}\approx2.842,
\]

but learned slot interventions remain near rank-1:

\[
r_{\Phi}\approx1.138.
\]

The next experiment should therefore not broaden the architecture.

It should make the smallest causal change consistent with the diagnosis:

\[
\boxed{
\text{conditional residual functional effects}
\rightarrow
\text{conditional residual training credit}
}
\]

A3.4 is now frozen as the empirical justification for that next step.