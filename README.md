# R4b Diagnostic README — `exp/e2e-a7-r4-qisca`

## 1. Branch / experiment context

Current research branch:

```text
exp/e2e-a7-r4-qisca
```

Current R4 implementation family:

```text
R4a = QI-SCA row-budget only
R4b = QI-SCA row-budget + slot capacity + all-real-slot spillover
```

Main goal of this branch:

> Prevent Edit-Slot collapse / functional giant behavior while preserving useful multi-slot decomposition for CIR.

This README records the current negative results, what has actually been fixed, what remains broken, and the next diagnostic that should be run before any further architecture change.

---

## 2. Important controls

### R1 — Entmax routing

R1 was the strongest control so far.

Approximate final validation result:

```text
Mean Recall ≈ 32.27
```

R1 improved sparse token routing and avoided total probability-mass monopoly, but forensic analysis showed a **functional giant**:

```text
one slot did most of the useful downstream work
```

So:

```text
R1:
benchmark good
functional decomposition bad
```

---

## 3. R4a result

R4a used:

```yaml
routing_mode: qisca
r4_theta: 0.15
r4_lambda: 0.45
r4_capacity_enabled: false
r4_candidate_mode: qasa_selected
```

R4a adds only the per-token row budget:

\[
\sum_k A_{nk} \le 1
\]

### Result

Final:

```text
Mean Recall ≈ 26.42
```

Collapse occurred by epoch 3.

Observed pattern:

```text
qasa_k                  → 1.0
routing_active_slots    → 1.0
monopoly                → ~1.0
binding                 → 100%
routing support         → almost entirely one slot
```

Example late-training routing:

```text
routing_slot_support ≈ [0, 0, 11.86, 0]
```

### Diagnosis

R4a does **not** solve the giant-slot problem.

The row constraint:

\[
\sum_k A_{nk} \le 1
\]

prevents several slots from simultaneously over-consuming the same token, but still permits:

\[
A_{n,k^\*}=1
\]

for the same slot \(k^\*\) across every token.

Therefore:

```text
anti-overlap != anti-monopoly
```

R4a is considered a failed experiment.

Do not spend more time sweeping only `theta` / `lambda` for R4a.

---

## 4. R4b design

R4b was introduced to test whether slot-column capacity can prevent one slot from becoming a giant.

Current intended configuration:

```yaml
routing_mode: qisca
r4_theta: 0.15
r4_lambda: 0.45

r4_capacity_enabled: true
r4_candidate_mode: all_real_slots
r4_slot_capacity: 3.0
r4_solver_iters: 64
```

### Key architecture change

Old ordering:

```text
P^Q
→ hard QASA pruning
→ QI-SCA
→ capacity
```

Problem:

```text
if QASA keeps only one slot,
capacity cannot spill mass into another slot
```

Current R4b ordering:

```text
P^Q
↓
QASA computed normally for diagnostics
↓
QI-SCA over all real slots
↓
token row budget
+
slot column capacity
↓
routing_active_mask
↓
Executor
```

In `all_real_slots` mode, non-QASA slots may receive assignment mass and are allowed to execute if they have positive routing mass.

This part is now working as intended.

---

## 5. R4b training result

Full R4b run:

```text
Mean Recall = 29.5054
```

Comparison:

| Method | Mean Recall |
|---|---:|
| R1 | ~32.27 |
| R4a | ~26.42 |
| R4b | 29.51 |

So:

```text
R4b > R4a
R4b < R1
```

R4b recovers a significant part of R4a's loss, but still does not beat the R1 control.

---

## 6. R4b global forensic result

Full validation:

```text
samples: 6016
valid_tokens: 71719

qasa_selected_slot_count:             2.7131
routing_active_slot_count:            3.9995
execution_active_slot_count:          3.9995

routing_non_qasa_active_slot_count:   1.2864
execution_non_qasa_active_slot_count: 1.2864

routing_non_qasa_mass_mean:           1.3488
routing_non_qasa_mass_fraction:       0.2119

routing_token_mass_mean:              0.5337
routing_unassigned_mass_mean:         0.4663
routing_fully_unassigned_fraction:    0.0571

routing_support_overlap_mean:         0.4602

routing_capacity_utilization_mean:    0.5302
routing_capacity_binding_fraction:    0.2496
```

### Interpretation

R4b successfully creates real spillover:

```text
~21% routed mass goes to non-QASA slots
```

So the capacity mechanism is not fake.

However:

```text
~46.6% routing mass is unassigned / rejected
```

This is very large and is currently one of the main suspects behind the benchmark loss.

---

## 7. Per-slot forensic

Full retrieval:

```text
FULL           R@10=20.24 R@50=38.77 MR=29.51
REFERENCE ONLY R@10=5.41  R@50=13.30 MR=9.36
```

Per-slot functional result:

| Slot | Drop MR | Utility = FULL - Drop | Only MR | Only gain over ref | Routing freq | QASA freq | Soft dominant | Routing mass | Drop cosine | State change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | 26.689 | +2.817 | 15.179 | +5.822 | 1.000 | 0.691 | 0.000 | 1.075 | 0.03549 | 4.040 |
| S1 | 29.389 | +0.117 | 12.081 | +2.724 | 1.000 | 1.000 | 1.000 | 2.999 | 0.00130 | 0.446 |
| S2 | 26.356 | +3.149 | 15.495 | +6.137 | 1.000 | 0.085 | 0.000 | 1.097 | 0.03712 | 4.794 |
| S3 | 25.754 | +3.751 | 16.076 | +6.719 | 1.000 | 0.938 | 0.000 | 1.191 | 0.04057 | 5.195 |

---

## 8. Main finding: S1 became a routing sink / implicit dustbin

S1 has the following signature:

```text
soft dominant frequency = 1.000
QASA selected frequency = 1.000
routing frequency       = 1.000
execution frequency     = 1.000
routing mass            = 2.999 ≈ capacity 3.0
```

But:

```text
drop utility ≈ +0.117 MR
drop cosine change ≈ 0.0013
state change ≈ 0.446
```

This is extremely different from S0/S2/S3.

Interpretation:

> S1 absorbs ownership and routing mass, repeatedly hits the capacity ceiling, but the Executor learns to make that slot almost functionally irrelevant.

Operationally, S1 behaves like an **implicit dustbin / routing sink**, despite there being no explicit NULL slot in the architecture.

---

## 9. The useful slots are S0 / S2 / S3

The other three slots are not dead.

Their drop utilities:

```text
S0: +2.817 MR
S2: +3.149 MR
S3: +3.751 MR
```

Their only-slot gains above reference-only:

```text
S0: +5.822
S2: +6.137
S3: +6.719
```

Therefore:

```text
R4b did not completely fail decomposition
```

It appears to have produced:

```text
3 useful functional slots
+
1 high-mass low-utility sink slot
```

This is an important distinction from R4a.

---

## 10. Current suspected failure mechanism

QI-SCA uses thresholded utility:

\[
A^{pre}_{kn}
=
\frac{[P^Q_{kn}-\theta]_+}{\lambda}
\]

with:

```text
theta = 0.15
```

Suppose a token has:

```text
S1 = 0.70
S0 = 0.10
S2 = 0.11
S3 = 0.09
```

If S1 has already reached its capacity:

```text
capacity(S1) = 3.0
```

then ideally mass should spill to alternatives.

But:

```text
S0 < theta
S2 < theta
S3 < theta
```

so all alternatives have zero utility after thresholding.

Result:

```text
S1 capacity full
+
alternatives below theta
↓
token mass cannot be reassigned
↓
implicit rejection
```

This is consistent with:

```text
routing_unassigned_mass_mean ≈ 0.466
```

So the current main hypothesis is:

> Capacity successfully prevents the giant slot from taking unlimited mass, but thresholded alternatives may be too weak to accept the displaced evidence, causing excessive rejection.

This is not yet proven.

---

## 11. What is already established

### Established by experiments

```text
R1:
best benchmark so far
but functional giant remains
```

```text
R4a:
row-budget alone does not prevent giant-slot collapse
```

```text
R4b:
column capacity + all-real-slot candidates prevents the same giant collapse
```

```text
R4b:
real non-QASA spillover occurs
```

```text
R4b:
S0/S2/S3 have measurable functional utility
```

```text
R4b:
S1 is high-mass / high-ownership but extremely low functional utility
```

```text
R4b:
~46.6% routing mass is rejected
```

### Not established yet

Do NOT yet claim:

```text
S1 definitely causes the benchmark loss
```

Do NOT yet claim:

```text
theta is definitely the only problem
```

Do NOT yet claim:

```text
execution of all four slots is the main problem
```

The forensic result actually shows that three slots are useful.

---

## 12. Next required diagnostic: REROUTE-WITHOUT-SLOT

The current slot-drop diagnostic disables a slot only at Executor time:

```text
routing happens normally
↓
slot receives assignment mass
↓
Executor later disables slot
```

For S1 this means its ~3.0 routing mass has already been consumed before the slot is dropped.

That cannot answer:

> What would happen if S1 were removed from the assignment problem itself?

The next diagnostic should therefore implement:

```text
REROUTE_WITHOUT_S0
REROUTE_WITHOUT_S1
REROUTE_WITHOUT_S2
REROUTE_WITHOUT_S3
```

Meaning:

```text
remove one slot from QI-SCA candidate set
↓
solve assignment again
↓
allow mass to redistribute
↓
execute resulting routed slots
↓
measure retrieval
```

The most important case is:

```text
REROUTE_WITHOUT_S1
```

---

## 13. How to interpret reroute-without-S1

### Case A

```text
REROUTE_WITHOUT_S1 > FULL
```

Then:

> S1 is stealing useful routing mass / acting as a harmful sink.

This would strongly support a future anti-sink mechanism.

### Case B

```text
REROUTE_WITHOUT_S1 ≈ FULL
or
REROUTE_WITHOUT_S1 < FULL
```

Then simply removing S1 does not recover the lost evidence.

This would support:

> The bigger bottleneck is threshold-induced rejection / insufficient alternative utility.

Then the next research target should be the assignment/rejection formulation rather than merely suppressing S1.

---

## 14. Current decision

Do **not** launch another 10-epoch training run yet.

Do **not** sweep only capacity.

Do **not** add a generic execution gate yet.

Do **not** add balance/diversity losses yet.

First run:

```text
reroute-without-slot forensic
```

especially:

```text
reroute-without-S1
```

because this directly separates:

```text
harmful routing sink
```

from:

```text
threshold / rejection bottleneck
```

---

## 15. Current forensic command

Current R4b checkpoint should be diagnosed with the FG-CLIP2-compatible forensic script:

```bash
python src/diagnose_taper_r4b.py \
  --checkpoint <R4B_CHECKPOINT> \
  --dataset-root data/FashionIQ \
  --routing-mode qisca \
  --r4-theta 0.15 \
  --r4-lambda 0.45 \
  --r4-capacity-enabled \
  --r4-candidate-mode all_real_slots \
  --r4-slot-capacity 3.0 \
  --r4-solver-iters 64 \
  --json-output reports/taper_r4b_forensic.json
```

Expected key outputs:

```text
FULL / REFERENCE-ONLY
DROP slot
ONLY slot
routing mass
routing / execution frequency
QASA frequency
soft dominant frequency
drop cosine
state-change magnitude
capacity utilization
capacity binding
unassigned mass
non-QASA spillover
```

---

# Current concise state

```text
R1
→ best retrieval (~32.27)
→ functional giant remains

R4a
→ row-budget
→ giant still collapses
→ MR ~26.42

R4b
→ slot capacity + all-real-slot spillover
→ giant collapse prevented
→ S0/S2/S3 useful
→ S1 becomes capped routing sink / implicit dustbin
→ ~46.6% mass rejected
→ MR ~29.51

NEXT
→ reroute-without-slot forensic
→ especially reroute-without-S1
```