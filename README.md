# A5.0 — Iterative Sparse Refinement: Forensic Analysis and Failure Diagnosis

**Branch:** `exp/e2e-a5.0-Iterative-sparse-refinement`  
**Checkpoint:** `outputs/2026-08-23/03-48-16/best.pt`  
**Best FashionIQ Mean Recall:** **58.5090**  
**Refinement rounds:** `3`  
**Edit Slots:** `4`  
**Primitives:** `8`  
**Gate mode:** `legacy_soft_train_hard_eval`  
**ST gate recovery:** `false`

---

## 1. Goal

A5.0 tested one isolated hypothesis:

> Can iterative, sample-conditioned slot refinement solve the A3 global-packing failure and the A4 monopoly failure without adding sparsity, residual regularization, slot selection, or new supervision?

Core loop:

```text
learned slot states
    ↓
score text evidence
    ↓
Competitive NULL + Edit ownership
    ↓
mass-aware evidence aggregation
    ↓
GRU update
    ↓
re-score with updated slot states
    ↓
repeat
```

A5.0 intentionally kept the rest of the retrieval system unchanged as much as possible:

- dense Competitive-NULL ownership,
- no entmax/sparsemax,
- no residual-evidence mechanism,
- no slot-state LayerNorm intervention,
- no post-GRU MLP,
- no noisy slot initialization,
- no functional selector,
- no explicit slot-diversity loss,
- no primitive balancing,
- final-round teacher counterfactual only,
- recurrent slot state is not directly injected into final `slot_mlp`.

So the experiment isolates the value of **iterative slot-state refinement itself**.

---

## 2. Retrieval result

| Metric | Value |
|---|---:|
| Recall@10 | 46.5069 |
| Recall@50 | 70.5112 |
| Mean Recall | **58.5090** |
| Reference-only Mean Recall | 1.7714 |
| Modification gain | 56.7376 |

Per-category:

| Category | R@10 | R@50 |
|---|---:|---:|
| Dress | 44.5711 | 70.4016 |
| Shirt | 44.2100 | 68.0569 |
| Toptee | 50.7394 | 73.0750 |

A5.0 is the strongest retrieval checkpoint so far. However, retrieval quality alone does not establish latent factorization.

---

# 3. Main conclusion

## A5.0 is a retrieval success but a factorization failure

The dominant failure mode is:

> **Iterative Consensus Collapse + Global Edit Packing + Slot-as-Compute-Ticket shortcut**

The model learns a strong global edit representation through iteration, but the four Edit Slots do not become complementary factors.

Observed causal pattern:

1. learned initial slot states are distinct,
2. all slots receive nearly identical ownership evidence,
3. the shared GRU overwrites the original identity differences,
4. recurrent states collapse to the same vector,
5. final semantic/effect/Edit-Slot representations collapse,
6. the Executor executes all four nearly-identical slots,
7. the extra slot count effectively supplies extra recurrent computation depth.

---

# 4. Smoking gun: distinct initial states collapse after one update

Pairwise cosine of recurrent slot states:

| Round | Slot-state pair cosine |
|---|---:|
| Round 0 | **0.06494** |
| Round 1 | **0.999293** |
| Round 2 | **0.999879** |

At initialization:

\[
\operatorname{cos}(S_i^{(0)},S_j^{(0)}) \approx 0.065
\]

The learned slots are already strongly differentiated. Therefore the failure is **not** caused by identical slot initialization or insufficient symmetry breaking.

After one refinement:

\[
\operatorname{cos}(S_i^{(1)},S_j^{(1)}) \approx 0.9993
\]

and after the second:

\[
\operatorname{cos}(S_i^{(2)},S_j^{(2)}) \approx 0.9999
\]

The iterative update does not preserve factor identity. It destroys it.

---

# 5. Root cause appears before the GRU: ownership masks are already identical

Pairwise slot-mask cosine:

| Round | Slot-mask pair cosine |
|---|---:|
| Round 0 | **0.999973** |
| Round 1 | **0.999994** |
| Round 2 | **0.999999** |

Per-slot mass is also nearly identical.

### Round 0

| Slot | Mean mass |
|---|---:|
| S0 | 2.4202 |
| S1 | 2.4197 |
| S2 | 2.4147 |
| S3 | 2.4318 |

### Round 1

| Slot | Mean mass |
|---|---:|
| S0 | 1.8747 |
| S1 | 1.8741 |
| S2 | 1.8726 |
| S3 | 1.8755 |

### Round 2

| Slot | Mean mass |
|---|---:|
| S0 | 1.6637 |
| S1 | 1.6629 |
| S2 | 1.6630 |
| S3 | 1.6627 |

Thus, approximately:

\[
P_{0n}\approx P_{1n}\approx P_{2n}\approx P_{3n}
\]

and therefore:

\[
e_0^{(t)}\approx e_1^{(t)}\approx e_2^{(t)}\approx e_3^{(t)}
\]

The shared GRU receives almost the same evidence for every slot and drives all hidden states toward the same attractor.

---

# 6. Causal chain

```text
distinct slot states
        │
        ▼
nearly identical ownership masks
        │
        ▼
nearly identical pooled evidence
        │
        ▼
shared GRU
        │
        ▼
slot identities overwritten
        │
        ▼
slot-state cosine → ~1
        │
        ▼
ownership becomes even more identical
        │
        ▼
semantic / effect / Edit Slot → ~1
```

A5.0 behaves like:

\[
S_0,S_1,S_2,S_3 \rightarrow S_{global}
\]

instead of:

\[
S_0,S_1,S_2,S_3 \rightarrow F_0,F_1,F_2,F_3
\]

---

# 7. GRU update scale supports the collapse mechanism

Slot-state norms:

| Round | Mean norm |
|---|---:|
| Round 0 | 0.4758 |
| Round 1 | 8.4161 |
| Round 2 | 11.9076 |

Update magnitudes:

| Transition | Mean update norm |
|---|---:|
| Round 0 → 1 | **8.3344** |
| Round 1 → 2 | **3.6867** |

The first update is much larger than the initial state magnitude:

\[
\frac{\|\Delta S^{0\rightarrow1}\|}{\|S^{(0)}\|}\approx17.5
\]

This is not by itself proof that update magnitude is the root cause, but together with shared evidence it explains why initial slot identity is erased almost immediately.

---

# 8. Iteration improves NULL-vs-EDIT, not SLOT-vs-SLOT

| Metric | R0 | R1 | R2 |
|---|---:|---:|---:|
| Assignment entropy | 1.6093 | 1.4246 | 1.2947 |
| Winner confidence | 0.2041 | 0.4007 | 0.4748 |
| Top1–Top2 margin | 0.0036 | 0.2462 | 0.3379 |
| NULL mass | 0.2032 | 0.3833 | 0.4528 |
| Ownership logit std | 0.0159 | 1.0323 | 1.4527 |

Assignments become sharper across rounds, but Edit-Slot mask cosine simultaneously approaches one:

\[
0.999973 \rightarrow 0.999994 \rightarrow 0.999999
\]

The most consistent interpretation is:

> A5.0 learns a stronger separation between **NULL** and the collective **EDIT group**, while S0/S1/S2/S3 remain nearly indistinguishable inside that group.

Conceptually:

```text
token
 ├─ NULL
 └─ EDIT
      ├─ S0 ≈ 25%
      ├─ S1 ≈ 25%
      ├─ S2 ≈ 25%
      └─ S3 ≈ 25%
```

---

# 9. Final factor representations are fully collapsed

| Representation | Active-pair cosine |
|---|---:|
| Slot semantics | **0.999999914** |
| Teacher effects | **0.999998255** |
| Final Edit Slots | **0.999999889** |

Collapse survives the whole pipeline:

```text
ownership
   ↓
slot semantics
   ↓
teacher counterfactual effect
   ↓
final Edit Slot
```

There is no evidence that the teacher-effect pathway or final `slot_mlp` restores specialization.

---

# 10. Balanced mass is not factorization

Final diagnostics:

- dominant slot share: `0.25021`,
- active slot count: `4.0`,
- every slot hard-active rate: `1.0`.

These numbers look balanced, but all masks and representations are nearly identical.

Therefore:

\[
\boxed{\text{balanced utilization} \neq \text{factorization}}
\]

A5.0 is a direct counterexample to using equal slot activity as evidence of semantic decomposition.

---

# 11. Every slot is functionally interchangeable

KEEP-one-slot retrieval:

| Variant | Mean Recall |
|---|---:|
| KEEP S0 | 27.4238 |
| KEEP S1 | 27.4238 |
| KEEP S2 | 27.4238 |
| KEEP S3 | 27.4238 |

The four values are identical.

No slot carries a unique retrieval function detectable by this ablation.

---

# 12. Repeating one slot reconstructs FULL retrieval

For any one slot:

| Copies | Mean Recall | Modification-gain fraction |
|---|---:|---:|
| ×1 | 27.4238 | 45.2% |
| ×2 | 51.5712 | 87.8% |
| ×3 | ~57.02 | ~97.4% |
| ×4 | ~58.51 | ~100% |

Example:

```text
REPEAT_S0_X1 = 27.4238
REPEAT_S0_X2 = 51.5712
REPEAT_S0_X3 = 57.0081
REPEAT_S0_X4 = 58.5090
```

The same behavior holds for S1, S2, and S3.

This is strong evidence that additional slot copies mainly provide additional Executor transitions, not additional semantic content.

---

# 13. Strongest forensic result: MEAN SLOT ×4 == FULL

| Variant | Mean Recall |
|---|---:|
| Mean Slot ×1 | 27.4238 |
| Mean Slot ×2 | 51.5712 |
| Mean Slot ×3 | 57.0166 |
| Mean Slot ×4 | **58.5090** |
| FULL | **58.5090** |

Therefore:

\[
\boxed{\text{MEAN SLOT}\times4 = \text{FULL}}
\]

within measured retrieval precision.

The full four-slot representation can be replaced by:

1. average all four Edit Slots,
2. clone the average four times,
3. execute four transitions.

Retrieval remains unchanged.

This makes slot identity functionally irrelevant in A5.0.

---

# 14. Slot count is acting as Executor compute depth

Executor diagnostics:

| Metric | Value |
|---|---:|
| Valid execution steps/sample | **4.0** |
| S0 execution rate | 1.0 |
| S1 execution rate | 1.0 |
| S2 execution rate | 1.0 |
| S3 execution rate | 1.0 |
| Slot gates | ~0.924 |
| Transition strength | **0.99955** |
| Actual state-change norm | 7.4529 |

Every slot executes, and transition strength is nearly saturated:

\[
\alpha \approx 1
\]

Together with `MEAN_SLOT ×4 == FULL`, this supports:

\[
\boxed{K_{factor} \text{ is currently coupled to } T_{exec}}
\]

The four slots behave much more like four recurrent compute tickets than four complementary factors.

---

# 15. Primitive bank does not trivially collapse

Observed primitive usage:

| Primitive | Fraction |
|---|---:|
| P0 | 0.0553 |
| P1 | 0.2331 |
| P2 | 0.1462 |
| P3 | 0.0724 |
| P4 | 0.0819 |
| P5 | 0.0846 |
| P6 | 0.1176 |
| P7 | 0.2089 |

The primitive router uses multiple primitives.

A plausible mechanism is:

```text
same / near-identical Edit Slot
        ↓
state changes after each Executor step
        ↓
router sees a different current state
        ↓
different primitives may be selected
        ↓
multiple recurrent transformations improve retrieval
```

Thus the main failure is not simply primitive collapse.

---

# 16. What A5.0 actually improved

A5.0 is still valuable.

It shows that iterative refinement improves the quality of a **global edit representation**.

Approximate behavior:

```text
text
 ↓
iterative refinement
 ↓
strong global edit representation
 ↓
four nearly-identical Edit Slots
 ↓
four recurrent Executor transitions
 ↓
58.509 Mean Recall
```

So iterative refinement is useful for retrieval, but insufficient for factorization.

---

# 17. Hypotheses ruled out or strongly weakened

## 17.1 Identical slot initialization — rejected

Round-0 slot-state cosine is only `0.06494`.

The initial learned slot identities are already strongly different.

Do not prioritize noisy initialization as the next intervention.

## 17.2 A4-style one-slot monopoly — rejected for A5.0

A5.0 has:

- 4 active slots,
- dominant share ≈ 0.25,
- 4 execution steps.

The failure is not winner-take-all monopoly. It is **consensus collapse**.

## 17.3 Sparse assignment as the immediate fix — not justified yet

At round 0:

- ownership logit std ≈ `0.0159`,
- winner confidence ≈ `0.204`,
- top1-top2 margin ≈ `0.0036`.

The destinations are almost tied.

Hardening this distribution immediately could amplify tiny accidental advantages and recreate an A4-style monopoly.

Therefore entmax/sparsemax should not be the first A5.1 intervention.

## 17.4 More GRU capacity — not justified

The GRU already performs a very large update.

A post-GRU MLP may only make the same global attractor more expressive.

## 17.5 Noise-based symmetry breaking — not justified

Initial states are already different. The collapse happens after shared evidence aggregation.

---

# 18. Formal failure statement

A5.0 approximately satisfies:

\[
S_0 \approx S_1 \approx S_2 \approx S_3 \approx S_{global}
\]

and:

\[
\operatorname{Retrieve}(S_{global},S_{global},S_{global},S_{global})
\approx
\operatorname{Retrieve}(S_0,S_1,S_2,S_3)
\]

The mean-slot experiment demonstrates this directly.

Therefore the learned latent decomposition is not functionally identifiable as multiple complementary factors.

---

# 19. Diagnosis table

| Failure hypothesis | Verdict |
|---|---|
| Initial slot symmetry | **Rejected** |
| A4-style monopoly | **Rejected for A5.0** |
| Dense shared-evidence leakage | **Confirmed** |
| Iterative consensus collapse | **Confirmed** |
| Semantic global packing | **Confirmed** |
| Teacher-effect global packing | **Confirmed** |
| Final Edit-Slot duplication | **Confirmed** |
| Slot-as-compute-ticket shortcut | **Confirmed strongly** |
| Primitive collapse | **Not primary / not confirmed** |
| Pure optimization plateau | **Not root cause** |

Short diagnosis:

> **A5.0 learns one strong global edit representation through iteration, then represents it four times and benefits from four state-dependent Executor transitions.**

---

# 20. Decision: move to A5.1

The next experiment is **A5.1**, with the target defined directly from the forensic evidence:

\[
\boxed{\text{Break shared-evidence consensus before the GRU}}
\]

A5.1 should not primarily try to make masks prettier or harder.

It must change the fact that all slots repeatedly receive almost the same evidence.

---

# 21. A5.1 target: residual / unexplained evidence

Current A5.0:

```text
full evidence
    ↓
S0/S1/S2/S3 receive almost the same content
    ↓
shared GRU
    ↓
consensus collapse
    ↓
full evidence again
```

Desired direction:

```text
available evidence
    ↓
factor-conditioned explanation
    ↓
remove/downweight explained evidence
    ↓
remaining unexplained evidence
    ↓
next refinement
```

Core principle:

> A factor should refine from evidence that remains useful after accounting for what has already been explained.

Important warning:

A single global residual shared identically by all slots can still produce:

\[
R_n \rightarrow S_0,S_1,S_2,S_3
\]

and recreate the same collapse.

A5.1 therefore needs a residual/explanation mechanism that changes **evidence availability across factor states**, rather than merely multiplying all slots by one shared residual mask.

---

# 22. A5.1 must remain a clean ablation

Define the first A5.1 run as:

\[
\boxed{\text{A5.1} = \text{A5.0-D3} + \text{Residual/Explained-Evidence mechanism}}
\]

Keep unchanged:

- same retrieval objective,
- same frozen teacher,
- same teacher representation contract,
- same 3 refinement rounds,
- same GRU,
- same learned initial slots,
- same dense Competitive-NULL ownership initially,
- same final `slot_mlp`,
- same Executor,
- same legacy gate control,
- same optimizer/data/protocol.

Do **not** add in the same first A5.1 run:

- entmax,
- sparsemax,
- slot-state LayerNorm,
- post-GRU MLP,
- noisy slot initialization,
- functional selection,
- extra slot-diversity regularizers,
- primitive balancing,
- A5.3 compute-depth decoupling.

Those remain deferred mechanisms with separate triggers.

---

# 23. A5.1 success criteria

A5.0 baseline:

```text
state cosine:
0.0649 → 0.9993 → 0.9999

mask cosine:
0.999973 → 0.999994 → 0.999999

final semantic cosine ≈ 1
final effect cosine   ≈ 1
final Edit-Slot cosine≈ 1

MEAN_SLOT ×4 == FULL
```

A5.1 should show several of the following:

### 23.1 Recurrent identities survive

Round 1 should **not** jump directly to ≈1 cosine.

### 23.2 Ownership support redundancy decreases

Pairwise mask cosine should move materially below the A5.0 near-identity regime.

Do not impose arbitrary orthogonality; measure the emergent change.

### 23.3 Functional slot representations separate

Active-pair:

- semantic cosine,
- teacher-effect cosine,
- final Edit-Slot cosine

should move materially away from one.

### 23.4 KEEP-one ablations become non-interchangeable

Desired forensic signature:

```text
KEEP_S0 != KEEP_S1 != KEEP_S2 != KEEP_S3
```

### 23.5 Repeating one slot no longer reconstructs FULL

Desired:

```text
REPEAT_Si_X4 < FULL
```

for at least most slots.

### 23.6 Mean-slot repetition loses information

Desired:

```text
MEAN_SLOT_X4 < FULL
```

If mean-slot ×4 still equals FULL, factor identity remains functionally irrelevant.

---

# 24. A5.3 remains necessary later

Even if A5.1 produces real factors, A5.0 strongly demonstrates that current factor count is coupled to Executor depth.

Eventually:

\[
K_{factor} \neq T_{exec}
\]

must become explicit.

Factor count asks:

> How many complementary latent edits exist?

Executor depth asks:

> How many state transitions are useful?

Those are different quantities.

Do not solve A5.3 before factor induction itself becomes credible.

---

# 25. Experimental roadmap

```text
A5.0
Iterative refinement
    ↓
RESULT:
better retrieval,
consensus collapse confirmed
    ↓
A5.1
Residual / unexplained evidence
    ↓
if slot identities survive
    ↓
Sparse ownership control
(entmax/sparsemax only when justified)
    ↓
A5.2
Functional factor selection
    ↓
A5.3
Decouple factor count from Executor depth
```

---

# 26. Research takeaways

### Positive findings

1. Iterative refinement improves retrieval.
2. NULL-vs-EDIT discrimination improves across rounds.
3. The recurrent mechanism learns useful task information.
4. The primitive router uses multiple primitives.
5. The forensic pipeline now exposes the internal failure mode directly.

### Negative findings

1. Iteration alone does not create factorization.
2. Distinct learned slot initialization is insufficient.
3. Balanced slot mass is not evidence of specialization.
4. Low monopoly is not evidence of specialization.
5. Multiple active slots are not evidence of multiple factors.
6. High retrieval does not imply factorized latent structure.
7. Duplicate slots can exploit Executor depth without carrying unique information.

---

# 27. Core lesson

\[
\boxed{\text{Different slot identities are useless if every slot observes the same evidence.}}
\]

The refinement update itself is insufficient.

For factorization to emerge, the model needs a mechanism that changes **what remains available to be explained** as factors are formed.

---

# 28. Status

**A5.0 status:** **CLOSED as a factorization solution.**

Retain it as:

- the strongest retrieval checkpoint so far,
- a useful global-edit iterative baseline,
- evidence that iterative refinement helps retrieval,
- evidence that shared-evidence consensus is the current factorization bottleneck.

**Next experiment:** **A5.1 — Residual / Unexplained Evidence Refinement**

Primary objective:

> **Prevent every Edit Slot from repeatedly consuming the same evidence and collapsing into the same global edit latent.**

---

## Final one-line summary

> **A5.0 reaches 58.509 Mean Recall by learning a stronger global edit representation, not by learning complementary Edit Slots; forensic ablations show that the four slots are nearly identical and mainly behave as four Executor compute steps, motivating A5.1 residual/unexplained-evidence factor induction.**