# A5.1 — Residual Evidence: Branch Status, Forensic Diagnosis, and Failure Report

**Date:** 2026-08-23  
**Branch:** `exp/e2e-a5.1-residual-evidence`  
**Base experiment:** A5.0 — Iterative Slot Refinement  
**Dataset:** FashionIQ  
**Training objective:** Retrieval only  
**Refinement rounds:** 3  
**Edit Slots:** 4  
**Primitives:** 8  
**Current status:** **CLOSED AS FACTORIZATION FAILURE**  
**Best observed FashionIQ Mean Recall:** **58.6299**

---

# 1. Purpose of A5.1

A5.1 was introduced after A5.0 produced a strong retrieval result but failed scientifically as a latent factorization model.

The A5.0 diagnosis was:

> Iterative refinement learns a strong global edit representation and then copies it across all Edit Slots.

The observed A5.0 failure chain was:

```text
distinct learned slot states
        ↓
nearly identical token ownership masks
        ↓
nearly identical pooled text evidence
        ↓
shared GRU update
        ↓
slot states rapidly converge
        ↓
final semantic / teacher-effect / Edit-Slot representations collapse
        ↓
four near-duplicate slots enter the Executor
        ↓
four slots effectively act as four tickets for Executor depth
```

A5.1 tested a narrower hypothesis:

> Is repeated reuse of already-explained evidence across refinement rounds the main cause of A5.0 consensus collapse?

The experiment therefore preserved the A5.0 architecture and added only a residual-evidence mechanism.

---

# 2. Scientific scope

A5.1 intentionally kept the following components unchanged:

- 4 Edit Slots;
- 3 iterative refinement rounds;
- dense soft Competitive-NULL ownership;
- learned slot initialization;
- the same `text_key_projection`;
- the same `text_value_projection`;
- the same shared `GRUCell` slot updater;
- final-round mass-aware semantic pooling;
- final-round frozen-teacher counterfactual effect;
- the same `slot_mlp`;
- the same slot gate;
- the same Router;
- the same Primitive Bank;
- the same Executor;
- the same retrieval loss;
- the same FashionIQ evaluation protocol.

A5.1 did **not** add:

- hard ownership;
- entmax/sparsemax;
- slot balancing loss;
- cosine diversity loss;
- orthogonality loss;
- functional slot selector;
- primitive balancing;
- dynamic factor count;
- fixed Executor depth;
- local/decontextualized value states;
- per-slot residual state.

Therefore the primary experimental variable was:

\[
\boxed{\text{A5.0 + global temporal residual evidence}}
\]

---

# 3. Implemented residual mechanism

For each valid text position \(n\), A5.1 maintains a scalar residual capacity:

\[
r_n^{(0)} = 1.
\]

Invalid or special positions receive zero residual capacity.

At refinement round \(t\), the residual modifies only Edit-Slot logits:

\[
\tilde z_{\ell n}^{(t)}
=
z_{\ell n}^{(t)}
+
\lambda_r
\log\left(r_n^{(t)}+\epsilon\right).
\]

NULL is not given this residual penalty.

Therefore:

```text
high residual
→ evidence remains available to Edit Slots

low residual
→ evidence becomes less attractive to all Edit Slots
→ NULL becomes relatively more competitive
```

The implemented residual consumption uses total Edit-Slot ownership:

\[
E_n^{(t)}
=
\sum_{\ell=1}^{L}P_{\ell n}^{(t)}
=
1-P_{\varnothing n}^{(t)}.
\]

The consumed residual is:

\[
c_n^{(t)}
=
\eta
r_n^{(t)}
E_n^{(t)}.
\]

The next residual is:

\[
r_n^{(t+1)}
=
r_n^{(t)}-c_n^{(t)}.
\]

Current experimental values:

```yaml
num_refine_iters: 3
residual_strength: 1.0
residual_depletion: 0.5
residual_eps: 1.0e-6
```

Residual is updated only between refinement rounds.

The frozen teacher is still evaluated only after final ownership is obtained.

---

# 4. A5.1 refinement pipeline

The implemented forward structure is:

```text
initial learned slot states
        │
        ▼
ROUND 0
        │
score text evidence
        │
apply residual r(0)=1
        │
Competitive NULL ownership
        │
mass-aware evidence aggregation
        │
shared GRU slot update
        │
consume global residual
        │
        ▼
ROUND 1
        │
score with updated slot states
        │
penalize already-consumed evidence
        │
Competitive NULL ownership
        │
mass-aware evidence aggregation
        │
shared GRU slot update
        │
consume residual again
        │
        ▼
ROUND 2
        │
final residual-conditioned ownership
        │
        ▼
final semantic pooling
        │
frozen teacher counterfactual
        │
slot_mlp
        │
slot gate
        │
Router + Primitive + Executor
        │
retrieval query
```

---

# 5. Training result

Five epochs were run.

Observed validation progression:

| Epoch | Loss | Mean Recall | Best | NULL | NULL argmax | Active slots | Dominant share |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.4515 | 53.5355 | 53.5355 | 0.374 | 0.820 | 4.00 | 0.250 |
| 2 | 0.7483 | 56.2362 | 56.2362 | 0.394 | 0.778 | 4.00 | 0.250 |
| 3 | 0.6442 | 57.6060 | 57.6060 | 0.424 | 0.802 | 4.00 | 0.250 |
| 4 | 0.5829 | **58.6299** | **58.6299** | 0.397 | 0.741 | 4.00 | 0.250 |
| 5 | 0.5377 | 57.8329 | 58.6299 | 0.419 | 0.758 | 4.00 | 0.250 |

This confirms that A5.1 is a strong retrieval model.

However:

\[
\boxed{\text{strong retrieval} \neq \text{successful factorization}}
\]

The forensic analysis below shows that the latent decomposition failure remains essentially unchanged.

---

# 6. Main conclusion

## A5.1 is another retrieval success but a factorization failure

The dominant observed failure remains:

> **Iterative Consensus Collapse + Global Edit Packing + Slot-as-Compute-Ticket shortcut**

Residual evidence does not produce complementary factors.

Instead, the four Edit Slots remain almost perfectly interchangeable.

The final model behaves approximately as:

\[
S_0
\approx
S_1
\approx
S_2
\approx
S_3
\approx
S_{\text{global-edit}}.
\]

The residual mechanism changes global evidence availability across refinement time, but it does not create meaningful explanatory responsibility between slots.

---

# 7. Smoking gun 1 — initial slot identities still disappear after one update

Pairwise cosine of recurrent slot states:

| Round | Slot-state pair cosine |
|---:|---:|
| Round 0 | **0.059602** |
| Round 1 | **0.999384** |
| Round 2 | **0.999900** |

At round 0:

\[
\operatorname{cos}
\left(
S_i^{(0)},S_j^{(0)}
\right)
\approx 0.06.
\]

The learned initial slots are strongly different.

After only one refinement update:

\[
\operatorname{cos}
\left(
S_i^{(1)},S_j^{(1)}
\right)
\approx 0.9994.
\]

After the second update:

\[
\operatorname{cos}
\left(
S_i^{(2)},S_j^{(2)}
\right)
\approx 0.9999.
\]

Therefore:

\[
\boxed{
\text{initial slot diversity exists,
but iterative evidence aggregation destroys it}
}
\]

A5.1 does not alter this fundamental trajectory.

---

# 8. Smoking gun 2 — ownership masks are already nearly identical

Pairwise cosine of Edit-Slot ownership masks:

| Round | Slot-mask pair cosine |
|---:|---:|
| Round 0 | **0.999977** |
| Round 1 | **0.999993** |
| Round 2 | **0.999999** |

Therefore, approximately:

\[
P_{0n}
\approx
P_{1n}
\approx
P_{2n}
\approx
P_{3n}.
\]

This is the same structural failure found in A5.0.

The four slots do not receive complementary evidence.

They receive almost the same ownership distribution.

Consequently:

\[
e_0^{(t)}
\approx
e_1^{(t)}
\approx
e_2^{(t)}
\approx
e_3^{(t)}.
\]

The shared GRU therefore continues to receive approximately the same evidence for all slots.

---

# 9. Round-by-round ownership dynamics

Observed ownership diagnostics:

| Metric | Round 0 | Round 1 | Round 2 |
|---|---:|---:|---:|
| Slot-state cosine | 0.059602 | 0.999384 | 0.999900 |
| Slot-mask cosine | 0.999977 | 0.999993 | 0.999999 |
| NULL mass | 0.200183 | 0.355034 | 0.449174 |
| Assignment entropy | 1.609385 | 1.438966 | 1.301608 |
| Winner confidence | 0.202422 | 0.381268 | 0.471633 |
| Top1–Top2 margin | 0.001809 | 0.219707 | 0.333796 |

Assignments become sharper over refinement rounds.

However, the sharpening is not evidence of slot specialization.

The Edit Slots become **more similar**, not less similar:

\[
0.999977
\rightarrow
0.999993
\rightarrow
0.999999.
\]

The most consistent interpretation is:

> Iteration plus residual learns a progressively stronger distinction between NULL and the collective EDIT group, while slot identity inside the EDIT group remains collapsed.

Conceptually:

```text
token
 ├─ NULL
 └─ EDIT
      ├─ S0 ≈ same
      ├─ S1 ≈ same
      ├─ S2 ≈ same
      └─ S3 ≈ same
```

---

# 10. Final factor representations remain fully collapsed

Final active-pair cosine:

| Representation | Pair cosine |
|---|---:|
| Slot semantics | **0.999999906** |
| Teacher effects | **0.999997629** |
| Final Edit Slots | **0.999999829** |

Collapse therefore survives every stage:

```text
ownership
   ↓
semantic evidence
   ↓
teacher counterfactual effect
   ↓
slot_mlp
   ↓
final Edit Slot
```

There is no evidence that downstream semantic construction recovers factors after ownership collapse.

---

# 11. Balanced utilization remains misleading

Final structural metrics:

```text
dominant slot share = 0.2502705
active slot count   = 4.0
valid exec steps    = 4.0
```

These values superficially look healthy.

All four slots are active.

Mass is almost perfectly balanced.

No slot monopolizes ownership.

But:

\[
\boxed{
\text{balanced activity does not imply semantic specialization}
}
\]

A5.1 is another direct counterexample.

Four equally used copies of the same representation are still a collapsed factorization.

---

# 12. Smoking gun 3 — every slot is functionally interchangeable

KEEP-one-slot retrieval:

| Variant | Mean Recall |
|---|---:|
| KEEP S0 | **27.46594** |
| KEEP S1 | **27.46594** |
| KEEP S2 | **27.45744** |
| KEEP S3 | **27.45767** |

The values are effectively identical.

Therefore no slot has a unique retrieval role detectable through this causal ablation.

Approximately:

\[
F(S_0)
\approx
F(S_1)
\approx
F(S_2)
\approx
F(S_3).
\]

---

# 13. Strongest causal result — mean slot ×4 exactly reconstructs FULL

Observed retrieval:

| Variant | Mean Recall |
|---|---:|
| REFERENCE ONLY | 1.77145 |
| MEAN SLOT ×1 | 27.45767 |
| MEAN SLOT ×2 | 51.47155 |
| MEAN SLOT ×3 | 56.97865 |
| MEAN SLOT ×4 | **58.62985** |
| FULL | **58.62985** |

Therefore:

\[
\boxed{
\text{MEAN SLOT}\times4
=
\text{FULL}
}
\]

within measured retrieval precision.

This is extremely strong evidence against a genuine four-factor interpretation.

The model can replace the full four-slot set with four copies of the average Edit Slot without losing retrieval performance.

---

# 14. Per-slot repeat test confirms compute-ticket behavior

Modification-gain recovery by repeated copies of individual slots:

| Repetition | Gain fraction |
|---|---:|
| ×1 | ≈ 45.18% |
| ×2 | ≈ 87.42% |
| ×3 | ≈ 97.05% |
| ×4 | ≈ 100% |

For example:

```text
REPEAT S0 ×1 → 45.19% of modification gain
REPEAT S0 ×2 → 87.42%
REPEAT S0 ×3 → 97.08%
REPEAT S0 ×4 → 99.99%
```

Equivalent behavior occurs for S1, S2 and S3.

Thus:

\[
\boxed{
\text{extra slot copies primarily provide extra execution opportunities}
}
\]

rather than complementary semantic information.

The current external architecture still permits:

```text
more slot instances
→ more Executor transitions
→ more retrieval improvement
```

even when the slot content is duplicated.

---

# 15. A5.0 versus A5.1

A5.0 best Mean Recall:

\[
58.5090
\]

A5.1 best Mean Recall:

\[
58.6299.
\]

Difference:

\[
\Delta MR \approx +0.12.
\]

This small retrieval difference is not accompanied by measurable factorization improvement.

A5.0 recurrent collapse:

```text
slot-state cosine:
R0 ≈ 0.0649
R1 ≈ 0.9993
R2 ≈ 0.9999
```

A5.1:

```text
slot-state cosine:
R0 = 0.0596
R1 = 0.9994
R2 = 0.9999
```

A5.0 mask collapse:

```text
≈ 0.99997
→ ≈ 0.99999
→ ≈ 1.00000
```

A5.1:

```text
0.999977
→ 0.999993
→ 0.999999
```

Therefore:

\[
\boxed{
\text{A5.1 does not materially alter the A5.0 failure geometry}
}
\]

---

# 16. Why the scalar residual is structurally limited

This section is a mathematical interpretation of the implemented mechanism.

For token \(n\), every Edit Slot receives the same residual correction:

\[
\tilde z_{\ell n}
=
z_{\ell n}
+
c_n
\]

where:

\[
c_n
=
\lambda_r\log(r_n+\epsilon).
\]

Consider two Edit Slots \(i\) and \(j\).

Their softmax probability ratio is:

\[
\frac{P_i}{P_j}
=
\frac{
e^{z_i+c_n}
}{
e^{z_j+c_n}
}
=
e^{z_i-z_j}.
\]

Therefore:

\[
\boxed{
\frac{P_i}{P_j}
\text{ is invariant to the global residual correction}
}
\]

for two Edit Slots at the same token.

This means the scalar residual can change:

```text
EDIT group vs NULL
```

but it cannot directly change:

```text
S0 vs S1
S0 vs S2
S1 vs S3
...
```

This limitation is highly relevant because the observed A5.0/A5.1 root failure is precisely:

\[
P_{0n}
\approx
P_{1n}
\approx
P_{2n}
\approx
P_{3n}.
\]

---

# 17. Why residual arrives too late for the current failure

At round 0:

\[
r_n^{(0)}=1.
\]

Therefore:

\[
\lambda\log(r_n^{(0)})=0.
\]

The first ownership round is therefore unaffected by residual evidence.

But the observed failure is already present at round 0:

\[
\text{slot-mask cosine}
=
0.999977.
\]

The sequence is approximately:

```text
ROUND 0
distinct slot states
        ↓
almost identical ownership
        ↓
almost identical pooled evidence
        ↓
large shared GRU update
        ↓
slot states collapse

ONLY THEN

residual modifies future evidence availability
```

Thus temporal residual is applied after the first causal collapse event has already occurred.

---

# 18. Exact failure mode of A5.1

A5.1 was designed to prevent:

> All slots repeatedly consuming the same already-explained evidence across refinement time.

But the dominant observed problem is stronger:

> All slots consume approximately the same evidence **inside the same refinement round**.

Therefore A5.1 attacks:

```text
cross-round reuse
```

while the primary collapse occurs through:

```text
within-round shared responsibility
```

A useful conceptual example is:

```text
ROUND 0
S0,S1,S2,S3 all consume evidence A
        ↓
A becomes residual-depleted

ROUND 1
S0,S1,S2,S3 all move together to evidence B
        ↓
B becomes depleted

ROUND 2
S0,S1,S2,S3 all move together again
```

Residual may alter the trajectory of the **group** without separating the members of the group.

---

# 19. What A5.1 successfully falsifies

The experiment provides evidence against the hypothesis:

> Repeated temporal evidence reuse alone is the dominant root cause of slot collapse.

More precisely:

\[
\boxed{
\text{global temporal evidence depletion is insufficient
to create inter-slot factor responsibility}
}
\]

This is an important narrowing result.

It means future work should not primarily focus on tuning:

```text
residual_strength
residual_depletion
residual_eps
```

without introducing a mechanism that changes competition **between Edit Slots**.

---

# 20. Why A5.1 should be closed instead of hyperparameter-swept

The current failure is structural:

```text
mask cosine ≈ 1
state cosine ≈ 1
semantic cosine ≈ 1
effect cosine ≈ 1
Edit-Slot cosine ≈ 1
MEAN SLOT ×4 = FULL
```

This is not merely a weak residual.

The implemented residual correction has a provable invariance:

\[
P(S_i|n)/P(S_j|n)
\]

is unchanged by the shared scalar residual correction.

Therefore a large sweep over:

```text
residual_strength
residual_depletion
```

would primarily vary the strength of EDIT-vs-NULL suppression.

It would not fundamentally introduce slot-vs-slot responsibility.

The A5.1 branch should therefore be treated as scientifically concluded.

---

# 21. Why A5.2 should not be the immediate next step

A5.2 proposes functional slot selection.

However, current candidates satisfy approximately:

\[
S_0
\approx
S_1
\approx
S_2
\approx
S_3
\approx
S_{\text{global}}.
\]

A selector can remove redundant copies.

It cannot transform duplicate global vectors into complementary factors.

For example:

```text
before selection:
GLOBAL
GLOBAL
GLOBAL
GLOBAL

after selecting two:
GLOBAL
GLOBAL
```

Therefore factor generation must improve before functional selection can solve the semantic decomposition problem.

---

# 22. Why A5.3 remains necessary but is not sufficient

A5.3 separates:

\[
K_{\text{factor}}
\]

from:

\[
T_{\text{exec}}.
\]

This is necessary because the forensic repeat-slot tests repeatedly show:

```text
duplicate slot
+
additional Executor transition
→ retrieval improvement
```

A5.3 is therefore still required eventually to eliminate the compute-ticket shortcut.

However:

\[
\boxed{
\text{compute decoupling does not itself create factors}
}
\]

It can prevent duplicated factors from being rewarded with additional compute, but it cannot solve the current within-round ownership collapse on its own.

---

# 23. Updated root-cause picture after A4, A5.0 and A5.1

The sequence of failed experiments now narrows the problem considerably.

## A4 — hard ownership

Observed failure:

```text
winner-take-all
→ one-slot monopoly
→ other slots die
```

Conclusion:

> Hard exclusivity alone is too aggressive and does not produce healthy decomposition.

---

## A5.0 — iterative refinement

Observed failure:

```text
all slots get almost identical evidence
→ shared GRU
→ consensus collapse
```

Conclusion:

> Iteration alone improves the global edit representation but does not create factors.

---

## A5.1 — global residual evidence

Observed failure:

```text
global evidence availability changes
but
relative Edit-Slot responsibilities remain almost identical
```

Conclusion:

> Global cross-round residual depletion does not create inter-slot explanatory responsibility.

---

# 24. Current best root-cause hypothesis

The strongest remaining diagnosis is:

\[
\boxed{
\textbf{
TAPER lacks a mechanism that creates soft inter-slot explanatory responsibility
before the first recurrent update.
}
}
\]

The critical event happens here:

```text
distinct slot states
        ↓
[ MISSING MECHANISM ]
        ↓
within-round differentiated responsibility
        ↓
different pooled evidence
        ↓
different recurrent updates
        ↓
persistent factor identity
```

Current architecture instead produces:

```text
distinct slot states
        ↓
nearly identical ownership
        ↓
nearly identical pooled evidence
        ↓
shared GRU
        ↓
consensus collapse
```

---

# 25. Proposed next experiment: A5.1b

The next experiment under consideration is:

> **Claim-Conditioned / Slot-Conditioned Competitive Availability**

Instead of a single:

\[
r_n
\]

shared by all Edit Slots, introduce availability that depends on the slot:

\[
r_{\ell n}.
\]

The purpose is not simply to remember whether evidence has been consumed globally.

The purpose is to alter relative explanatory responsibility between competing slots.

A candidate structure is:

1. compute raw soft ownership;
2. estimate how strongly other slots claim token \(n\);
3. build a slot-conditioned availability;
4. modify each slot's logit differently;
5. recompute final soft ownership;
6. only then pool evidence and update the GRU.

Conceptually:

```text
raw ownership
      ↓
cross-slot claim competition
      ↓
slot-conditioned availability
      ↓
slot-specific adjusted logits
      ↓
soft differentiated ownership
      ↓
different pooled evidence
      ↓
GRU update
```

This mechanism must remain soft.

A4 already demonstrated that aggressive winner-take-all competition can collapse into one-slot monopoly.

---

# 26. A5.1b success criteria

The next experiment should not be judged only by retrieval.

The primary structural success criteria should include:

### Round-0 ownership differentiation

A5.1 currently has:

\[
\text{mask cosine}_{R0}
=
0.999977.
\]

A5.1b should materially reduce this without causing one-slot monopoly.

---

### Recurrent identity survival

A5.1 currently has:

\[
0.0596
\rightarrow
0.9994
\rightarrow
0.9999.
\]

A successful mechanism should prevent the first refinement update from annihilating slot identity.

---

### Final semantic differentiation

Required improvement in:

```text
semantic pair cosine
teacher-effect pair cosine
final Edit-Slot pair cosine
```

---

### No A4-style monopoly

Must simultaneously maintain:

```text
multiple active slots
dominant slot share far below 1
no one-slot ownership collapse
```

---

### Retrieval remains viable

Factorization improvement that destroys retrieval completely is not sufficient.

---

### Repeat-slot shortcut weakens

Eventually:

```text
MEAN SLOT ×4
```

should no longer be exactly equivalent to:

```text
FULL
```

if the final slots actually carry complementary information.

---

# 27. A5.1b kill criteria

Immediately reject the mechanism if either of these occurs:

## Failure A — consensus remains

```text
slot mask cosine ≈ 1
slot state cosine ≈ 1
semantic/effect/edit cosine ≈ 1
```

Then the mechanism is too weak or structurally ineffective.

---

## Failure B — A4 monopoly returns

```text
dominant share → 1
active slots → 1
one slot receives most evidence
```

Then the mechanism is too winner-take-all.

The target is a middle regime:

```text
soft competition
+
meaningfully different responsibilities
+
multiple surviving factors
```

---

# 28. Branch verdict

## Retrieval

\[
\boxed{\text{SUCCESS}}
\]

Best Mean Recall:

\[
58.6299.
\]

---

## Slot utilization

\[
\boxed{\text{BALANCED BUT MISLEADING}}
\]

All four slots remain active and mass-balanced.

---

## Semantic factorization

\[
\boxed{\text{FAILURE}}
\]

Final semantic/effect/Edit-Slot cosine is effectively one.

---

## Functional specialization

\[
\boxed{\text{FAILURE}}
\]

All KEEP-one-slot variants are effectively identical.

---

## Compute shortcut

\[
\boxed{\text{STILL PRESENT}}
\]

Mean-slot ×4 exactly recovers FULL retrieval.

---

## Global residual hypothesis

\[
\boxed{\text{INSUFFICIENT}}
\]

Global temporal evidence depletion does not create inter-slot explanatory responsibility.

---

# 29. Final conclusion

A5.1 produces one of the strongest retrieval checkpoints so far, but it does not solve the scientific objective of TAPER factorization.

The decisive observations are:

\[
\text{slot-mask cosine}
\approx1,
\]

\[
\text{slot-state cosine after first update}
\approx1,
\]

\[
\text{semantic/effect/Edit-Slot cosine}
\approx1,
\]

and:

\[
\boxed{
\text{MEAN SLOT}\times4
=
\text{FULL}
}.
\]

The global scalar residual changes how much evidence remains available to the Edit-Slot group over time.

It does not determine which Edit Slot should own which explanatory factor.

The current evidence therefore supports the following updated design principle:

\[
\boxed{
\textbf{
The next mechanism must create differentiated soft responsibility
between Edit Slots inside each refinement round,
before shared recurrent updating destroys their identities.
}
}
\]

A5.1 should be considered complete and closed.

The next justified research direction is a carefully controlled **claim-conditioned / slot-conditioned soft competition experiment**, while preserving the A5.0/A5.1 teacher, downstream Executor, and retrieval objective so causal attribution remains possible.