# TAPER A5.1b — Claim-Conditioned Cross-Slot Residual

## Complete Experiment Record, Mathematical Contract, Training Results, Forensic Diagnosis, Failure Analysis, Falsified Hypotheses, and Research Handoff

**Date:** 2026-08-23
**Project:** TAPER-CIR
**Repository:** `Le-Minh-Nhut/cir`
**Branch:** `exp/e2e-a5.1b-claim-conditioned-residual`
**Experiment:** A5.1b — Claim-Conditioned Cross-Slot Residual
**Parent experiment:** A5.1 — Global Temporal Residual Evidence
**Dataset:** FashionIQ
**Training objective:** Retrieval loss only
**Number of Edit Slots:** 4
**Refinement rounds:** 3
**Primitive Bank size:** 8
**Best observed Mean Recall:** **58.4945**
**Best observed epoch:** **4 / 5**
**Scientific status:** **CLOSED — RETRIEVAL SUCCESS / FACTORIZATION FAILURE**

---

# 1. Executive conclusion

A5.1b **fails as a factorization mechanism**.

The experiment was introduced to fix a precise mathematical weakness in A5.1.

A5.1 maintained one scalar residual per evidence position:

[
r_n.
]

That residual applied the same penalty to every Edit Slot:

[
z'_{\ell n}
===========

z_{\ell n}
+
\lambda\log(r_n+\epsilon).
]

Therefore it could change the relative competition between:

[
\text{NULL}
\quad\text{and}\quad
\text{EDIT group},
]

but could not change the relative odds between:

[
S_0,S_1,S_2,S_3.
]

A5.1b replaced this global scalar residual with a **slot-conditioned residual**:

[
r_{\ell n},
]

and introduced a cross-slot exclusion mechanism so that evidence claimed by other slots reduces its future availability to the current slot.

Unlike A5.1, this mechanism is mathematically capable of changing:

[
\frac{P(S_i\mid n)}
{P(S_j\mid n)}.
]

It therefore directly targets **inter-slot explanatory responsibility**.

However, after training and full forensic evaluation, the learned representation remains almost completely collapsed.

Final active-pair cosine similarities are:

```text
semantic_pair_cos = 0.99999953
effect_pair_cos   = 0.99998992
edit_slot_cos     = 0.99999941
```

All four slots remain active:

```text
active_slot_count = 4.0
```

with almost perfectly balanced mass:

```text
dominant_slot_share = 0.250421
```

but this balance is cosmetic rather than semantic.

The strongest causal result is:

```text
FULL         = 58.49454

MEAN_SLOT_X1 = 27.24135
MEAN_SLOT_X2 = 51.43116
MEAN_SLOT_X3 = 56.90281
MEAN_SLOT_X4 = 58.49454
```

Therefore:

[
\boxed{
\text{MEAN SLOT}\times4
=======================

\text{FULL}
}
]

to the precision of the evaluation.

Repeating any individual original slot four times also reproduces essentially all modification gain:

```text
REPEAT_S0_X4 gain_fraction = 1.000290
REPEAT_S1_X4 gain_fraction = 0.999854
REPEAT_S2_X4 gain_fraction = 0.999710
REPEAT_S3_X4 gain_fraction = 0.999710
```

The correct scientific interpretation is:

[
\boxed{
S_0
\approx
S_1
\approx
S_2
\approx
S_3
\approx
S_{\text{global-edit}}
}
]

and the four slots continue to behave primarily as **four nearly interchangeable tickets for Executor depth**.

A5.1b therefore does **not** solve:

> Global Edit Packing + Iterative Consensus Collapse + Slot-as-Compute-Ticket.

---

# 2. Why A5.1b was necessary

## 2.1 A5.0 failure

A5.0 introduced iterative slot refinement.

The intended process was:

```text
different initial slots
        ↓
different evidence ownership
        ↓
different pooled evidence
        ↓
different recurrent states
        ↓
stronger specialization next round
```

Instead, the observed trajectory was approximately:

```text
initial slot states are different
        ↓
ownership masks are almost identical
        ↓
pooled evidence is almost identical
        ↓
shared GRU update
        ↓
slot identities disappear
        ↓
later rounds become even more similar
```

Frozen A5.0 measurements:

```text
slot-state cosine:

round 0 = 0.064941
round 1 = 0.999293
round 2 = 0.999879
```

and:

```text
slot-mask cosine:

round 0 = 0.999973
round 1 = 0.999994
round 2 = 0.999999
```

The first recurrent transition was also extremely large relative to initial slot magnitude:

```text
initial slot-state norm ≈ 0.476
update 0 → 1 norm       ≈ 8.334
```

Therefore the recurrent update overwhelms a large fraction of the initial slot identity.

A5.0 reached strong retrieval:

```text
Mean Recall ≈ 58.509
```

but failed scientifically.

---

# 3. A5.1 failure that motivated A5.1b

A5.1 introduced global temporal residual evidence.

For each valid token:

[
r_n^{(0)}=1.
]

Residual modified all Edit-Slot logits:

[
\tilde z_{\ell n}
=================

z_{\ell n}
+
\lambda_r\log(r_n+\epsilon).
]

Total EDIT ownership consumed residual across refinement rounds.

This successfully introduced:

> memory of evidence already used by the EDIT group.

But mathematically:

[
\tilde z_{in}-\tilde z_{jn}
===========================

# (z_{in}+c_n)-(z_{jn}+c_n)

z_{in}-z_{jn}.
]

Therefore:

[
\boxed{
\frac{P(S_i\mid n)}
{P(S_j\mid n)}
\text{ is unchanged by the scalar residual}
}
]

for any two Edit Slots.

A5.1 could suppress:

```text
EDIT vs NULL
```

but not create:

```text
S0 vs S1 vs S2 vs S3 responsibility.
```

Experimentally, A5.1 failed exactly this way.

Observed A5.1:

```text
slot-state cosine
R0 = 0.059602
R1 = 0.999384
R2 = 0.999900
```

```text
slot-mask cosine
R0 = 0.999977
R1 = 0.999993
R2 = 0.999999
```

Final:

```text
semantic cosine = 0.999999906
effect cosine   = 0.999997629
Edit-Slot cosine= 0.999999829
```

Again:

```text
MEAN_SLOT_X4 ≈ FULL
```

Therefore A5.1 falsified the hypothesis:

> Temporal reuse of globally available evidence is, by itself, the main cause of slot collapse.

---

# 4. Primary A5.1b research question

A5.1b asks:

> If evidence availability is conditioned on which slot is trying to claim it, and if cross-slot competition is applied before the GRU receives evidence, can small initial slot differences be amplified into meaningful sample-conditioned factor specialization?

This is narrower than asking:

> Can any possible slot-conditioned mechanism ever work?

The experiment only tests the implemented mechanism under the tested configuration.

---

# 5. Experimental scope

A5.1b intentionally preserves almost the entire A5.1 pipeline.

The experiment changes only the **factor-induction ownership/residual mechanism**.

Kept fixed:

```text
FashionIQ dataset
caption policy
cached features
frozen CSMCIR teacher
teacher-native text representation
reference features
target/gallery features

4 Edit Slots
3 refinement rounds

text key projection
text value projection
shared GRU slot updater

final mass-aware semantic pooling
teacher counterfactual computation
slot MLP
slot gate

Primitive Bank
Router
Executor
query head

retrieval loss
optimizer family
evaluation protocol
```

Not introduced:

```text
hard winner-take-all ownership
entmax
sparsemax
top-k masking
slot balancing
orthogonality loss
cosine diversity loss
primitive balancing
functional slot selector
dynamic slot count
localized value representation
fixed / decoupled Executor budget
extra supervision
manual factor labels
LLM-generated factors
```

Thus the scientific variable is approximately:

[
\boxed{
\text{A5.1 global residual}
\rightarrow
\text{A5.1b slot-conditioned cross-claim residual}
}
]

---

# 6. Repository-level implementation scope

Relative to the A5.1 code baseline, the experimental code change is concentrated in:

```text
src/models/taper.py
```

The main training configuration remains:

```yaml
num_slots: 4
num_primitives: 8

num_refine_iters: 3

mask_temperature: 1.0
router_temperature: 1.0
retrieval_temperature: 0.07

residual_strength: 1.0
residual_depletion: 0.5
residual_eps: 1.0e-6

gate_mode: legacy_soft_train_hard_eval
st_gate_recovery: false

slot_gate_threshold: 0.4
```

Training objective:

```yaml
loss_weights:
  retrieval_loss: 1.0
```

There is no explicit factorization loss.

---

# 7. A5.1b residual state

A5.1 used:

[
r_n
\in
\mathbb R^{B\times N}.
]

A5.1b uses:

[
R
=

{r_{\ell n}}
\in
\mathbb R^{B\times L\times N}.
]

Where:

```text
B = batch size
L = number of Edit Slots = 4
N = number of text evidence positions
```

Initialization:

[
r_{\ell n}^{(0)}=1
]

for valid content positions.

Invalid / padding / excluded positions have zero residual availability.

Interpretation:

> (r_{\ell n}) measures how available evidence position (n) currently is to Edit Slot (\ell).

This differs conceptually from A5.1.

A5.1:

```text
r_n
= evidence capacity remaining for EDIT globally
```

A5.1b:

```text
r_l,n
= evidence availability remaining for a particular slot
```

---

# 8. Slot-conditioned residual logit bias

For each Edit Slot:

[
\log r_{\ell n}
]

is computed.

Instead of applying the raw residual directly, A5.1b centers it across slots:

[
b_{\ell n}
==========

\lambda_r
\left[
\log(r_{\ell n}+\epsilon)
-------------------------

\frac{1}{L}
\sum_j
\log(r_{jn}+\epsilon)
\right].
]

The effective Edit-Slot logit is:

[
\tilde z_{\ell n}
=================

z_{\ell n}
+
b_{\ell n}.
]

NULL is not directly given this residual bias.

---

# 9. Why centering was introduced

Without centering, if every slot residual decreases equally:

```text
r0 = r1 = r2 = r3 < 1
```

all Edit-Slot logits would decrease together.

This recreates the A5.1 behavior:

```text
EDIT group loses probability
→ NULL gains probability
```

without creating specialization.

With centered log residual:

If:

[
r_0=r_1=r_2=r_3,
]

then:

[
b_0=b_1=b_2=b_3=0.
]

Therefore a common residual decrease has no direct effect on the Edit-Slot logits.

Only **relative residual differences between slots** matter.

Important nuance:

> This does not mean NULL probability is mathematically invariant whenever residuals differ.

NULL's raw logit is unchanged, but changing individual Edit-Slot logits can still change the full softmax partition function and therefore affect NULL probability indirectly.

The precise statement is:

[
\boxed{
\text{NULL logit is not directly residual-biased}
}
]

not:

[
\boxed{
P(\text{NULL}) \text{ can never change}.
}
]

---

# 10. Cross-slot claim computation

A5.1b first computes a provisional / proposal ownership.

Let:

[
P_{\ell n}^{proposal}
]

be the proposal ownership for Edit Slot (\ell).

Total Edit-Slot claim:

[
E_n
===

\sum_jP_{jn}^{proposal}.
]

For slot (\ell), claim from all **other** slots is:

[
o_{\ell n}
==========

## E_n

P_{\ell n}^{proposal}.
]

Equivalently:

[
\boxed{
o_{\ell n}
==========

\sum_{j\ne\ell}
P_{jn}^{proposal}
}
]

This is the defining A5.1b quantity.

---

# 11. Residual update

Retention:

[
\rho_{\ell n}
=============

## 1

\eta o_{\ell n}.
]

With current:

[
\eta
====

# \text{residual_depletion}

0.5.
]

Then:

[
r_{\ell n}^{new}
================

r_{\ell n}^{old}
\rho_{\ell n}.
]

Implementation bounds retention to:

[
[0,1].
]

Therefore:

[
0
\le
r_{\ell n}^{new}
\le
r_{\ell n}^{old}
\le1.
]

Consumed availability:

[
c_{\ell n}
==========

## r_{\ell n}^{old}

r_{\ell n}^{new}.
]

---

# 12. Intended positive-feedback mechanism

Suppose for one evidence token:

```text
S0 = .22
S1 = .20
S2 = .19
S3 = .19
```

Then:

```text
other_claim(S0) = .58
other_claim(S1) = .60
other_claim(S2) = .61
other_claim(S3) = .61
```

Therefore:

```text
r0 > r1 > r2 ≈ r3
```

after depletion.

Centered residual bias then gives S0 a relative advantage in the final ownership calculation.

Conceptually:

```text
tiny S0 advantage
        ↓
other slots claim more relative to S0
        ↓
S0 loses less availability
        ↓
relative S0 bias increases
        ↓
final ownership difference increases
```

The goal was:

[
\text{small asymmetry}
\rightarrow
\text{larger asymmetry}
\rightarrow
\text{different evidence}
\rightarrow
\text{different GRU state}.
]

---

# 13. Exact-symmetry limitation

A5.1b does **not** mathematically create asymmetry from a perfectly symmetric state.

Suppose:

[
P_{0n}
======

# P_{1n}

# P_{2n}

P_{3n}.
]

Then:

[
o_{0n}
======

# o_{1n}

# o_{2n}

o_{3n}.
]

Therefore:

[
r_{0n}^{new}
============

# r_{1n}^{new}

# r_{2n}^{new}

r_{3n}^{new}.
]

Centered residual bias is then:

[
b_{\ell n}=0.
]

Thus exact symmetry remains a fixed point.

A5.1b is therefore:

> an **asymmetry amplifier**,

not:

> an unconditional deterministic symmetry generator.

This distinction matters for interpreting the negative result.

---

# 14. Same-round intervention

The most important difference from A5.1 is ordering.

A5.1 effectively used:

```text
ownership
    ↓
pool evidence
    ↓
GRU
    ↓
update global residual
    ↓
next round
```

This was too late because slot states had already collapsed after the first GRU update.

A5.1b uses:

```text
current slot states
        ↓
proposal ownership
        ↓
cross-slot claim
        ↓
slot-conditioned residual update
        ↓
re-score ownership IN THE SAME ROUND
        ↓
final ownership
        ↓
pool final evidence
        ↓
GRU
```

Therefore cross-slot responsibility acts **before the first recurrent update**.

This was the central causal intervention of A5.1b.

---

# 15. A5.1b refinement loop

For every refinement round (t):

### Step 1 — proposal ownership

[
P^{proposal,(t)}
================

\operatorname{CompetitiveOwnership}
(
Q^{(t)},
H,
R^{(t)}
).
]

### Step 2 — other-slot claim

[
o_{\ell n}^{(t)}
================

\sum_{j\ne\ell}
P_{jn}^{proposal,(t)}.
]

### Step 3 — update slot-specific residual

[
r_{\ell n}^{(t,+)}
==================

r_{\ell n}^{(t)}
\left(
1-\eta o_{\ell n}^{(t)}
\right).
]

### Step 4 — recompute final ownership

[
P^{final,(t)}
=============

\operatorname{CompetitiveOwnership}
(
Q^{(t)},
H,
R^{(t,+)}
).
]

### Step 5 — pool evidence using final masks

[
e_\ell^{(t)}
============

\operatorname{MassAwarePool}
(
P_{\ell}^{final,(t)},
V
).
]

### Step 6 — recurrent update

For non-final rounds:

[
Q^{(t+1)}
=========

GRU
(
e^{(t)},
Q^{(t)}
).
]

Final teacher counterfactual computation remains after the final ownership round.

---

# 16. Diagnostic tensors produced by the model

A5.1b exposes:

```text
refine_slot_states
refine_slot_masses
refine_update_norms

refine_slot_masks
refine_null_probs
refine_ownership_logits

refine_residuals
refine_consumed_residuals
refine_other_claims
refine_proposal_slot_masks
```

Important shapes:

```text
refine_slot_states
[B,T,L,D]

refine_slot_masks
[B,T,L,N]

refine_residuals
[B,T,L,N]

refine_consumed_residuals
[B,T,L,N]

refine_other_claims
[B,T,L,N]

refine_proposal_slot_masks
[B,T,L,N]
```

---

# 17. Current diagnostic limitation

`diagnose_a5.py` currently computes the standard A5 diagnostics from:

```text
refine_slot_states
refine_slot_masks
refine_null_probs
refine_ownership_logits
refine_slot_masses
refine_update_norms
```

but the frozen A5.1b forensic summary used here does **not** aggregate:

```text
refine_proposal_slot_masks
refine_other_claims
refine_residuals
```

into the JSON structural table.

Therefore this report can state exactly what the **final ownership of each round** did.

It cannot currently quantify from the frozen JSON:

[
\text{proposal mask cosine}
\rightarrow
\text{final mask cosine}
]

within the same A5.1b checkpoint.

This metric should be added before publication-quality final archival.

No proposal→final numerical difference is invented in this document.

---

# 18. Training budget

Although the default experiment YAML contains:

```yaml
num_epochs: 100
```

this A5.1b pilot was run for:

[
\boxed{5\text{ epochs}}
]

which was sufficient to expose the structural behavior.

The run should not be interpreted as a 100-epoch result.

---

# 19. Epoch-by-epoch training result

| Epoch |   Loss | Mean Recall |        Best |  NULL | NULL Argmax | All-NULL | Active Slots | Hard Active | Dominant | Monopoly |
| ----: | -----: | ----------: | ----------: | ----: | ----------: | -------: | -----------: | ----------: | -------: | -------: |
|     1 | 1.4511 |     53.3758 |     53.3758 | 0.379 |       0.789 |    0.161 |         4.00 |        4.00 |    0.251 |    0.000 |
|     2 | 0.7486 |     56.3206 |     56.3206 | 0.393 |       0.788 |    0.106 |         4.00 |        4.00 |    0.250 |    0.000 |
|     3 | 0.6440 |     57.6813 |     57.6813 | 0.410 |       0.782 |    0.111 |         4.00 |        4.00 |    0.250 |    0.000 |
|     4 | 0.5825 | **58.4945** | **58.4945** | 0.370 |       0.695 |    0.042 |         4.00 |        4.00 |    0.250 |    0.000 |
|     5 | 0.5373 |     57.9255 |     58.4945 | 0.395 |       0.723 |    0.041 |         4.00 |        4.00 |    0.251 |    0.000 |

Best checkpoint:

[
\boxed{
\text{Mean Recall}=58.4945
}
]

at epoch 4.

The exact checkpoint path is recorded inside:

```text
reports/a5_1b_forensic.json["checkpoint"]
```

and should be retained together with this README.

---

# 20. Training-level interpretation

The training log establishes several useful facts.

## 20.1 Retrieval learns normally

Mean Recall improves from:

```text
53.38 → 56.32 → 57.68 → 58.49
```

within four epochs.

Therefore A5.1b does not create an obvious optimization catastrophe.

## 20.2 NULL takeover does not occur

All-NULL decreases:

```text
.161
→ .106
→ .111
→ .042
→ .041
```

The failure cannot be summarized as:

> residual pressure forces everything into NULL.

## 20.3 A4-style one-slot monopoly does not occur

Throughout training:

```text
active_slots = 4.00
hard_active  = 4.00
dominant     ≈ .25
monopoly     = 0
```

Therefore A5.1b successfully avoids the particular A4 degeneracy:

```text
one slot owns everything
three slots die
```

But avoiding monopoly is not enough.

---

# 21. Forensic refinement dynamics

Observed A5.1b:

| Metric                 |      Round 0 |      Round 1 |      Round 2 |
| ---------------------- | -----------: | -----------: | -----------: |
| Slot-state pair cosine | **0.059928** | **0.999359** | **0.999882** |
| Slot-mask pair cosine  | **0.999951** | **0.999987** | **0.999994** |
| NULL mass              |     0.202889 |     0.361465 |     0.417966 |
| Assignment entropy     |     1.609336 |     1.445550 |     1.339949 |
| Winner confidence      |     0.203775 |     0.381989 |     0.444128 |
| Top1–Top2 margin       |     0.002772 |     0.221762 |     0.298297 |

---

# 22. Smoking gun 1 — initial slot identities still disappear

Round 0 slot-state cosine:

[
0.059928.
]

Thus initial slots are genuinely different.

After one update:

[
0.999359.
]

After the second update:

[
0.999882.
]

Therefore:

[
\boxed{
\text{A5.1b does not preserve slot identity through the first refinement transition}
}
]

The same consensus-collapse trajectory survives.

Conceptually:

```text
distinct initial slot queries
        ↓
first evidence aggregation
        ↓
shared recurrent update
        ↓
states become nearly collinear
        ↓
later iterations refine the same representation
```

---

# 23. Smoking gun 2 — ownership remains almost identical

Final per-round mask cosine:

```text
R0 = 0.999951
R1 = 0.999987
R2 = 0.999994
```

This means that despite the slot-conditioned residual:

[
P_{0n}
\approx
P_{1n}
\approx
P_{2n}
\approx
P_{3n}
]

for almost every sample/token pattern in aggregate geometry.

The mechanism produces, at most, extremely small deviations from the clone state.

---

# 24. A5.1 versus A5.1b

A5.1:

```text
R0 mask cos  = 0.9999767
R1 state cos = 0.9993840
R2 state cos = 0.9998999
```

A5.1b:

```text
R0 mask cos  = 0.9999509
R1 state cos = 0.9993591
R2 state cos = 0.9998818
```

Difference in R0 mask cosine:

[
0.9999767
---------

0.9999509
\approx
2.58\times10^{-5}.
]

Difference in R1 state cosine:

[
0.9993840
---------

0.9993591
\approx
2.49\times10^{-5}.
]

Thus A5.1b changes the geometry in the intended direction by a **microscopic amount**, but not enough to produce meaningful factorization.

This is not evidence of practical specialization.

---

# 25. A5.0 versus A5.1b reveals an even stronger point

A5.0:

```text
R0 mask cosine  = 0.9999731
R1 state cosine = 0.9992925
```

A5.1b:

```text
R0 mask cosine  = 0.9999509
R1 state cosine = 0.9993591
```

A5.1b has a slightly less identical Round-0 mask than A5.0.

Yet its Round-1 state cosine is **not lower than A5.0**.

It is slightly higher:

```text
A5.0 R1 = 0.9992925
A5.1b R1 = 0.9993591
```

This is important.

It suggests that merely introducing a tiny amount of assignment differentiation does not automatically generate state differentiation.

The distinction is largely erased by the subsequent:

```text
value pooling
+
shared GRU update
```

This shifts attention toward the **semantic content of pooled values and update dynamics**, rather than continuing to focus only on mask mechanics.

---

# 26. Ownership gets sharper, not more factorized

Assignment entropy decreases:

```text
1.6093
→ 1.4456
→ 1.3399
```

Winner confidence increases:

```text
0.2038
→ 0.3820
→ 0.4441
```

Top1–Top2 margin increases:

```text
0.0028
→ 0.2218
→ 0.2983
```

Yet slot-mask cosine remains approximately 1.

Therefore:

[
\boxed{
\text{sharper assignment}
\neq
\text{slot specialization}
}
]

The model becomes more certain about the overall competitive distribution without producing distinct Edit-Slot roles.

---

# 27. Final representation collapse

Final active-pair similarities:

| Representation         |     Pair cosine |
| ---------------------- | --------------: |
| Slot semantics         | **0.999999528** |
| Frozen-teacher effects | **0.999989916** |
| Final Edit Slots       | **0.999999415** |

Therefore collapse survives through:

```text
ownership
    ↓
semantic pooling
    ↓
teacher counterfactual effect
    ↓
slot MLP
    ↓
final Edit Slots
```

No downstream component restores factorization.

---

# 28. Balanced mass is a false positive

Observed:

```text
dominant_slot_share = 0.250421
active_slot_count   = 4.0
```

At first glance this appears ideal:

```text
4 active slots
+
equal usage
+
no monopoly
```

But pairwise semantic/effect/Edit-Slot cosine is effectively one.

Therefore:

[
\boxed{
\text{balanced slot utilization}
\neq
\text{factorization}
}
]

A5.1b is a particularly clean demonstration.

The model has learned:

> four equally used copies,

not:

> four complementary factors.

---

# 29. Executor still receives four steps

Observed:

```text
execution/valid_steps_per_sample = 4.0
```

Therefore every sample still grants four Executor transition opportunities.

This preserves the architectural incentive discovered in A3/A5.0:

```text
more active slots
→ more recurrent execution steps
→ more useful computation
```

Thus duplicate slots remain useful even if they carry no distinct semantic factor.

---

# 30. Retrieval forensic results

## FULL

[
58.4945376
]

## REFERENCE ONLY

[
1.7715144
]

Modification gain:

[
58.4945376
----------

# 1.7715144

56.7230232.
]

Therefore the model strongly uses the modification signal.

The problem is decomposition, not absence of text information.

---

# 31. KEEP-one-slot experiment

Observed:

```text
KEEP_S0 = 27.24976
KEEP_S1 = 27.24976
KEEP_S2 = 27.24135
KEEP_S3 = 27.22458
```

The four values are effectively indistinguishable.

Therefore:

[
F(S_0)
\approx
F(S_1)
\approx
F(S_2)
\approx
F(S_3).
]

No slot has a clearly unique retrieval function under this intervention.

---

# 32. One slot already carries a large global-edit component

For the mean slot:

```text
MEAN_SLOT_X1 = 27.24135
```

Relative modification-gain fraction:

```text
0.449021
```

Thus one representative slot recovers roughly:

[
44.9%
]

of the total modification gain.

This is already too large for a clean complementary decomposition if arbitrary slots are meant to represent different factors.

More importantly, additional copies of the same information rapidly recover the rest.

---

# 33. Mean-slot repeated control

Observed:

| Variant      |  Mean Recall | Modification gain fraction |
| ------------ | -----------: | -------------------------: |
| MEAN_SLOT_X1 |     27.24135 |                   0.449021 |
| MEAN_SLOT_X2 |     51.43116 |                   0.875476 |
| MEAN_SLOT_X3 |     56.90281 |                   0.971939 |
| MEAN_SLOT_X4 | **58.49454** |               **1.000000** |

Thus:

[
\boxed{
\text{one representative slot repeated four times}
==================================================

\text{full model}
}
]

This is the strongest functional diagnosis.

---

# 34. Every individual slot repeated four times works

Modification-gain fraction:

```text
REPEAT_S0_X4 = 1.000290
REPEAT_S1_X4 = 0.999854
REPEAT_S2_X4 = 0.999710
REPEAT_S3_X4 = 0.999710
```

Therefore no particular slot is uniquely responsible for FULL performance.

Any one of the four can approximately substitute for the complete factor set if given the same four execution opportunities.

This supports:

[
\boxed{
\text{slots behave as interchangeable compute tickets}
}
]

rather than complementary factors.

---

# 35. Repeat-depth trajectory

Representative behavior:

```text
x1 → ~44.9% modification gain
x2 → ~87.5%
x3 → ~97.2%
x4 → ~100%
```

This monotonic curve strongly indicates that much of the benefit of having four slots is related to repeated downstream transformation depth.

The model does not require four distinct semantic instructions to achieve FULL retrieval.

---

# 36. Final functional diagnosis

A5.1b behaves approximately as:

```text
modification text
        ↓
contextual evidence encoder
        ↓
four nearly identical ownership distributions
        ↓
four nearly identical semantic representations
        ↓
four nearly identical teacher effects
        ↓
four nearly identical Edit Slots
        ↓
4 Executor opportunities
        ↓
high retrieval
```

Equivalent abstraction:

[
S_0
\approx
S_1
\approx
S_2
\approx
S_3
\approx
g_{\text{edit}}.
]

Then:

[
Executor(
g_{\text{edit}},
g_{\text{edit}},
g_{\text{edit}},
g_{\text{edit}}
)
]

is sufficient to reproduce FULL behavior.

---

# 37. Comparison across A4 → A5.0 → A5.1 → A5.1b

| Experiment | Main intervention             | Active-slot outcome | Structural failure | Mean Recall |
| ---------- | ----------------------------- | ------------------- | ------------------ | ----------: |
| A4         | ST-hard ownership             | ~1 active slot      | Monopoly           |     55.3259 |
| A5.0       | Iterative refinement          | 4 active slots      | Consensus clones   |     58.5090 |
| A5.1       | Global residual (r_n)         | 4 active slots      | Consensus clones   |     58.6299 |
| A5.1b      | Slot-conditioned (r_{\ell n}) | 4 active slots      | Consensus clones   |     58.4945 |

Important:

> These retrieval numbers should not be treated as a statistically controlled leaderboard because training budgets/seeds are not necessarily matched for every historical experiment.

The structural comparisons are the primary scientific evidence.

---

# 38. Unified failure picture

## A4

Hard exclusivity:

```text
tiny asymmetry
    ↓
one hard winner
    ↓
winner receives useful evidence
    ↓
retrieval rewards winner
    ↓
positive feedback
    ↓
one-slot monopoly
```

## A5.0

Soft iterative refinement:

```text
small slot differences
    ↓
almost identical ownership
    ↓
same evidence
    ↓
shared GRU
    ↓
consensus collapse
```

## A5.1

Global residual:

```text
old evidence globally suppressed
    ↓
all slots affected similarly
    ↓
EDIT-vs-NULL changes
    ↓
inter-slot ratios remain collapsed
```

## A5.1b

Claim-conditioned residual:

```text
small relative slot differences can be amplified
    ↓
but actual differences remain microscopic
    ↓
pooled content / GRU reconverges
    ↓
final factors remain clones
```

---

# 39. What A5.1b falsifies

A5.1b falsifies the following **specific sufficiency hypothesis**:

> A soft, slot-conditioned cross-claim residual with centered relative-log residual bias, depletion (0.5), strength (1.0), three refinement rounds, and retrieval-only training is sufficient to transform small initial ownership differences into useful multi-slot factorization.

Observed answer:

[
\boxed{\text{No}}
]

---

# 40. What A5.1b does NOT falsify

Do **not** overclaim:

> "All slot-conditioned residual methods are impossible."

A5.1b does not prove that.

It also does not prove:

```text
all possible sparse assignment fails
all possible factorization from retrieval is impossible
localized value channels will work
GRU is definitely the only remaining problem
teacher is definitely innocent
functional selection is useless
```

The safe conclusion is narrower:

[
\boxed{
\text{this ownership/residual family is insufficient under the tested system}
}
]

---

# 41. Stronger conclusion after the A4–A5.1b sequence

The accumulated evidence makes the following explanation increasingly plausible:

> The main bottleneck is no longer merely how responsibility mass is distributed over tokens.

The model can fail under:

```text
soft sharing
hard exclusivity
iterative refinement
global temporal depletion
slot-conditioned cross-claim depletion
```

This suggests that the **information content carried by the evidence values and the learning objective itself** are more fundamental.

---

# 42. Remaining root-cause candidate 1 — contextual value leakage

Current refinement uses contextual text states as the substrate for pooled evidence.

A useful abstraction is:

[
h_n
===

g+\epsilon_n
]

where:

* (g) = global sentence / edit representation;
* (\epsilon_n) = token-local information.

If:

[
|g|\gg|\epsilon_n|,
]

then even different support masks can produce:

[
e_0
\approx
e_1
\approx
e_2
\approx
e_3
\approx
g.
]

Example:

```text
h("red")     = global_edit + color_local
h("longer")  = global_edit + length_local
h("sleeves") = global_edit + anchor_local
```

Even if:

```text
S0 prefers "red"
S1 prefers "longer"
```

both pooled values may still mostly contain:

```text
global_edit.
```

Therefore:

[
\boxed{
\text{different token support}
\not\Rightarrow
\text{different semantic content}
}
]

for contextual Transformer states.

---

# 43. Why A5.1b strengthens the contextual-leakage concern

A5.1b slightly changes Round-0 mask geometry.

But this tiny mask change does not create a corresponding improvement in post-GRU slot-state separation.

That observation is consistent with:

```text
mask differences exist
        ↓
pooled values remain globally similar
        ↓
GRU receives nearly same semantic input
        ↓
states reconverge
```

This is not yet proof.

It is a stronger motivation to test the value channel directly.

---

# 44. Remaining root-cause candidate 2 — recurrent update overwhelms identity

A5.0 established:

```text
initial slot norm ≈ 0.476
first update norm ≈ 8.334
```

Therefore the first recurrent update is much larger than the initial state scale.

If all slots receive approximately similar evidence, then:

```text
small initial differences
+
huge common update
```

naturally produce:

```text
almost identical states.
```

A5.1b does not isolate:

```text
global value leakage
```

from:

```text
oversized/shared recurrent update.
```

Both remain plausible contributors.

---

# 45. Cheap diagnostic recommended before another redesign

Before another expensive FashionIQ run, instrument the pre-GRU pooled evidence:

[
e_\ell^{(t)}.
]

Log per-round:

```text
pooled_evidence_pair_cos
pooled_evidence_norm
slot_state_pair_cos before GRU
slot_state_pair_cos after GRU
update pair cosine
update norm
```

Interpretation:

### Case A

```text
mask cosine somewhat lower
pooled evidence cosine ≈ 1
```

Then the value substrate itself is re-globalizing the factors.

### Case B

```text
pooled evidence cosine significantly < 1
but post-GRU state cosine → 1
```

Then recurrent update dynamics are the more immediate bottleneck.

### Case C

```text
mask cosine ≈ 1
pooled evidence cosine ≈ 1
```

Then ownership remains insufficiently differentiated before values can even be evaluated.

This diagnostic can distinguish the next causal target much more cheaply than another blind full experiment.

---

# 46. Recommended next architectural experiment

The strongest currently justified next candidate is:

> **Localized Value Channel**

Keep contextual states for relevance scoring:

[
k_n
===

W_kh_n^{ctx}.
]

But construct slot content from a less globally contextualized value representation:

[
v_n
===

W_vh_n^{local}.
]

Thus:

```text
contextual representation
→ tells slot WHERE / WHAT to attend to

localized representation
→ tells slot WHAT CONTENT it actually receives
```

This attempts to preserve useful contextual relevance while preventing every claimed token from carrying a full global edit representation.

---

# 47. Proposed next experiment naming

To preserve historical numbering, recommended:

```text
A5.1c-LV
Claim-Conditioned Ownership + Localized Value Channel
```

rather than reusing the original A5.2 label, which was reserved for Functional Selection.

Possible branch:

```text
exp/e2e-a5.1c-localized-values
```

---

# 48. What must stay frozen in A5.1c-LV

To isolate the value-channel hypothesis, keep:

```text
A5.1b claim-conditioned ownership
4 slots
3 refinement rounds
residual strength
residual depletion

slot initialization
key/scoring path
GRU architecture initially
teacher
slot_mlp
gate
Router
Primitive Bank
Executor
retrieval loss
FashionIQ protocol
```

Change only:

```text
value representation used for slot content
```

If GRU dynamics are changed simultaneously, causal attribution will be lost.

---

# 49. Why Functional Selection is not the immediate next fix

Current candidates are:

[
S_0
\approx
S_1
\approx
S_2
\approx
S_3.
]

A selector can decide:

```text
keep S0
drop S1/S2/S3
```

but that does not create:

```text
COLOR
LENGTH
STYLE
MATERIAL
```

from already collapsed candidates.

Functional selection is useful only after candidate factors become meaningfully distinct.

---

# 50. Why compute decoupling is still necessary later

The repeat-slot result demonstrates:

[
\boxed{
\text{duplicate factors still buy useful execution depth}
}
]

Therefore mature TAPER should eventually separate:

[
K_{\text{factor}}
]

from:

[
T_{\text{exec}}.
]

But compute decoupling alone does not solve factor induction.

It removes the incentive to manufacture duplicate slots.

It does not automatically create semantic factors.

Thus it remains important but orthogonal.

---

# 51. Why no residual-strength sweep is recommended

Possible temptation:

```text
strength 1 → 2 → 5
depletion .5 → .8 → 1.0
```

A large sweep is not recommended as the next main research action.

Reason:

The current structural failure is enormous:

```text
mask cosine ≈ .99995–.99999
state cosine after first update ≈ .99936
final semantic cosine ≈ .9999995
```

A5.1b already possesses the mathematical ability to alter inter-slot odds.

Yet factorization remains essentially absent.

Therefore endless residual tuning risks spending compute optimizing an increasingly secondary mechanism.

One small strength ablation could be used only as a diagnostic if needed.

It should not replace investigation of the value/update bottleneck.

---

# 52. Why retrieval differences should not be overinterpreted

Observed:

```text
A5.0  ≈ 58.509
A5.1  ≈ 58.630
A5.1b ≈ 58.495
```

Differences are small.

No multi-seed statistical analysis has been frozen here.

Therefore this report does not claim:

```text
A5.1 is significantly better than A5.1b
```

or:

```text
A5.1b damages retrieval.
```

The important result is structural:

> all three iterative variants learn strong retrieval while remaining factorization failures.

---

# 53. A5.1b scientific status

## Retrieval

**SUCCESSFUL**

Best:

[
58.4945.
]

## Multi-slot activity

**ACTIVE BUT MISLEADING**

```text
4 / 4 slots active
dominant share ≈ .25
```

## Ownership specialization

**FAILED**

```text
mask cosine ≈ 1
```

## Recurrent-state specialization

**FAILED**

```text
R1/R2 state cosine ≈ 1
```

## Semantic specialization

**FAILED**

```text
semantic cosine ≈ 1
```

## Teacher-effect specialization

**FAILED**

```text
effect cosine ≈ 1
```

## Final Edit-Slot specialization

**FAILED**

```text
Edit-Slot cosine ≈ 1
```

## Functional non-redundancy

**FAILED**

Any representative slot repeated four times approximately reproduces FULL.

---

# 54. Kill decision

[
\boxed{
\textbf{A5.1b CLOSED}
}
]

Do not continue the same run to 100 epochs merely hoping for spontaneous factorization.

The failure pattern is already structurally decisive after five epochs.

Do not promote A5.1b as the factorization solution.

Do not present balanced slot utilization as evidence of success.

---

# 55. Frozen observations from A5.1b

The following should be preserved as experiment facts.

1. A5.1b uses four Edit Slots.
2. A5.1b uses three refinement rounds.
3. Residual state is slot-conditioned: `[B,L,N]`.
4. Residual strength is `1.0`.
5. Residual depletion is `0.5`.
6. Residual epsilon is `1e-6`.
7. The mechanism uses a centered relative-log residual bias.
8. NULL receives no direct residual logit bias.
9. Proposal ownership is computed before cross-slot claim update.
10. Other-slot claim is:
    [
    \sum_{j\ne\ell}P_{jn}.
    ]
11. Residual is updated before final ownership of the same round.
12. Only final ownership masks enter the GRU.
13. Teacher counterfactuals remain final-round only.
14. A5.1b trained normally under retrieval supervision.
15. Best Mean Recall was `58.4945`.
16. Best checkpoint occurred at epoch 4 of the five-epoch pilot.
17. All four slots remained active.
18. Dominant slot share remained approximately `0.25`.
19. No A4-style monopoly occurred.
20. Round-0 slot-state cosine was `0.059928`.
21. Round-1 slot-state cosine was `0.999359`.
22. Round-2 slot-state cosine was `0.999882`.
23. Round-0 slot-mask cosine was `0.999951`.
24. Round-1 slot-mask cosine was `0.999987`.
25. Round-2 slot-mask cosine was `0.999994`.
26. Final semantic cosine was `0.999999528`.
27. Final teacher-effect cosine was `0.999989916`.
28. Final Edit-Slot cosine was `0.999999415`.
29. Mean valid Executor steps per sample was `4.0`.
30. FULL Mean Recall was `58.49454`.
31. Reference-only Mean Recall was `1.77151`.
32. KEEP S0/S1/S2/S3 were all approximately `27.2`.
33. One mean slot recovered approximately `44.9%` of modification gain.
34. Two repeated mean slots recovered approximately `87.5%`.
35. Three repeated mean slots recovered approximately `97.2%`.
36. Four repeated mean slots recovered `100%`.
37. Every individual slot repeated four times also recovered approximately `100%`.
38. A5.1b therefore remains functionally redundant.
39. The Slot-as-Compute-Ticket shortcut remains alive.
40. Claim-conditioned ownership does not solve factorization under the tested configuration.

---

# 56. Claims that are now unsafe

Do not claim:

> "A5.1b factorizes the modification."

Do not claim:

> "Four active slots mean four factors."

Do not claim:

> "Balanced slot mass demonstrates specialization."

Do not claim:

> "The residual successfully created explanatory responsibility."

Do not claim:

> "Lower Round-0 mask cosine proves factor discovery."

Do not claim:

> "Claim-conditioned residuals in general cannot work."

Do not claim:

> "The GRU is proven to be the sole cause."

Do not claim:

> "Contextual-value leakage is already proven."

---

# 57. Safe scientific wording

Safe:

> A5.1b introduces slot-conditioned evidence availability and same-round cross-slot claim competition, allowing relative Edit-Slot ownership odds to change.

Safe:

> Under the tested FashionIQ retrieval-only setup, this intervention produced only microscopic changes in slot ownership geometry and did not prevent recurrent consensus collapse.

Safe:

> Final semantic, teacher-effect, and Edit-Slot representations remained almost perfectly aligned.

Safe:

> Repeating a single representative slot four times reproduced FULL retrieval, demonstrating continued functional redundancy and the persistence of the Slot-as-Compute-Ticket shortcut.

Safe:

> These results motivate testing whether globally contextualized value representations and/or recurrent update dynamics erase the small ownership differences that A5.1b can create.

---

# 58. Compact causal record

```text
A3
soft Competitive-NULL
        ↓
multiple duplicate global slots
        ↓
Slot-as-Compute-Ticket

A4
hard exclusive ownership
        ↓
one-slot monopoly
        ↓
factorization fails

A5.0
iterative refinement
        ↓
almost identical ownership
        ↓
shared GRU
        ↓
consensus collapse
        ↓
duplicate global slots

A5.1
global residual r[n]
        ↓
temporal evidence reuse reduced
        ↓
EDIT-vs-NULL changes
        ↓
slot-vs-slot ratios unchanged
        ↓
duplicate global slots

A5.1b
slot-conditioned residual r[l,n]
        ↓
slot-vs-slot ratios can change
        ↓
small assignment differences
        ↓
differences remain microscopic
        ↓
GRU/state reconvergence survives
        ↓
duplicate global slots
```

---

# 59. Current strongest diagnosis

The accumulated evidence now supports the following working diagnosis:

[
\boxed{
\textbf{
The core failure is deeper than ownership overlap or residual reuse.
The model still has an easy globally packed edit representation,
and the current value/update/downstream objective does not force complementary factors.
}
}
]

More operationally:

```text
contextual evidence may already contain global edit information
                    +
retrieval loss does not require a decomposition
                    +
shared recurrent update rapidly erases weak slot differences
                    +
duplicate active slots receive extra Executor opportunities
```

Together, these conditions make the global-clone solution highly attractive.

---

# 60. Recommended immediate next diagnostic

Before another expensive redesign:

```text
measure pooled slot evidence BEFORE GRU
```

Required new per-round metrics:

```text
proposal_mask_pair_cos
final_mask_pair_cos

pooled_evidence_pair_cos
pooled_evidence_norm

pre-GRU state cosine
post-GRU state cosine

GRU update cosine
GRU update norm

residual pair dispersion
other-claim pair dispersion
```

This will determine whether the next intervention belongs primarily in:

```text
VALUE CONTENT
```

or:

```text
RECURRENT UPDATE DYNAMICS.
```

---

# 61. Recommended next experiment if pooled evidence is already collapsed

If:

```text
pooled_evidence_pair_cos ≈ 1
```

before the GRU:

Run:

```text
A5.1c-LV
Localized Value Channel
```

Core idea:

[
\text{contextual states}
\rightarrow
\text{keys / relevance}
]

while:

[
\text{localized states}
\rightarrow
\text{values / factor content}.
]

Goal:

> Do not allow every token value to independently carry the entire globally contextualized modification.

---

# 62. Recommended next experiment if evidence differs but GRU collapses it

If:

```text
pooled evidence cosine << 1
```

but:

```text
post-GRU state cosine → 1,
```

the more justified next experiment is an identity-preserving recurrent update.

Possible later family:

```text
bounded / gated residual update
```

with explicit control over:

[
|\Delta q|
]

relative to:

[
|q|.
]

This must be a separate experiment from Localized Value Channel.

Do not combine both immediately.

---

# 63. Reproduction commands

## Training

```bash
git switch exp/e2e-a5.1b-claim-conditioned-residual
git pull --ff-only

python src/train.py \
  experiment=taper_e2e \
  experiment.num_epochs=5
```

## Locate most recent best checkpoint

```bash
CKPT="$(find outputs -type f -name best.pt -printf '%T@ %p\n' \
  | sort -nr \
  | head -1 \
  | cut -d' ' -f2-)"

echo "$CKPT"
```

## Run forensic diagnosis

```bash
python src/diagnose_a5.py \
  experiment=taper_e2e \
  +checkpoint="$CKPT" \
  +report=reports/a5_1b_forensic.json
```

The JSON report must be archived beside this README.

---

# 64. Files that should be frozen with this experiment

Recommended archive set:

```text
src/models/taper.py
src/train.py
src/diagnose_a5.py
conf/experiment/taper_e2e.yaml

reports/a5_1b_forensic.json

best.pt metadata / path
git commit SHA
training console log

this README
```

Do not rely only on memory or screenshots for future comparison.

---

# 65. Minimum metadata to preserve

Record:

```text
branch
git HEAD
git dirty status
seed
checkpoint path

FashionIQ cache identity
teacher checkpoint identity

num slots
num primitives
num refine rounds

residual mode
residual strength
residual depletion
residual epsilon

gate mode
slot threshold

training epochs
best epoch
best Mean Recall
```

---

# 66. Final verdict

[
\boxed{
\textbf{
A5.1b is a negative factorization result.
}
}
]

It succeeds mechanically:

```text
slot-conditioned residual exists
cross-slot claims are computed
relative slot ownership can change
claim intervention occurs before GRU
retrieval remains strong
one-slot monopoly is avoided
```

But it fails scientifically:

```text
Round-0 masks remain nearly identical
slot states reconverge after one update
semantic representations collapse
teacher effects collapse
final Edit Slots collapse
all slots remain interchangeable
repeating one slot reproduces FULL retrieval
```

Thus:

[
\boxed{
\textbf{
Giving each slot its own claim-conditioned evidence availability is not sufficient to induce functional edit factors in the current TAPER architecture.
}
}
]

---

# 67. One-sentence research record

> **A5.1b replaced A5.1's global token residual with a same-round slot-conditioned cross-claim residual capable of changing inter-slot ownership odds, but after five FashionIQ epochs the model still converged to four almost perfectly interchangeable global-edit slots (`R0 mask cos≈0.999951`, `R1 state cos≈0.999359`, final semantic/effect/Edit-Slot cosines≈1), while repeating one representative slot four times exactly recovered FULL retrieval (`58.4945`), demonstrating that claim-conditioned ownership alone does not prevent Global Edit Packing or the Slot-as-Compute-Ticket shortcut and shifting the next causal investigation toward contextual value leakage and recurrent state-update dynamics.**

---

# 68. Final handoff

Current research sequence:

```text
A3
Soft Competitive-NULL
→ CLOSED: duplicate global slots

A4
ST-hard ownership
→ CLOSED: one-slot monopoly

A5.0
Iterative refinement
→ CLOSED: iterative consensus collapse

A5.1
Global residual evidence
→ CLOSED: temporal residual insufficient

A5.1b
Claim-conditioned cross-slot residual
→ CLOSED: inter-slot residual insufficient
```

Next action:

```text
diagnose pooled evidence before GRU
        ↓
if evidence already collapsed:
    Localized Value Channel

if evidence is distinct but state collapses:
    Identity-Preserving Recurrent Update
```

Do not return to blind ownership hardening unless new evidence directly identifies ownership as the bottleneck.

The current strongest operational principle is:

[
\boxed{
\textbf{
Before asking slots to compete harder,
verify that the information they receive can actually be different.
}
}
]
