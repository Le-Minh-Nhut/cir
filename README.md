# A3.1 — QASA Slot Filter (No-NULL)  
## Forensic Diagnosis and Experiment Report

**Branch:** `exp/e2e-a3.1-qasa-slot-filter`  
**Experiment family:** TAPER Edit-Slot decomposition  
**Dataset / protocol:** FashionIQ validation  
**Primary supervision:** End-to-end retrieval loss  
**Slot count:** 4 Edit Slots  
**Primitive count:** 8  
**QASA configuration:** `tau=0.5`, `rho=0.8`, `mu=0.3`  
**NULL competitor:** removed for this experiment  
**QASA mask at evaluation:** enabled (`qasa_apply_at_eval=true`)

---

# 1. Experiment goal

The purpose of A3.1 is to test whether **QASA-style quality-guided slot selection** can replace the previous learned slot gate and improve Edit-Slot decomposition in TAPER.

The core hypothesis was:

> If Edit Slots compete over text tokens and QASA removes low-quality / redundant slots before execution, then the surviving slots may be pressured toward cleaner and more specialized responsibilities.

The desired behavior is not merely to reduce the number of active slots.

The actual target is:

```text
S0 -> one edit factor / role
S1 -> another edit factor / role
S2 -> another edit factor / role
S3 -> another edit factor / role
```

rather than:

```text
S0 ─┐
S1 ─┼──> approximately the same edit function
S2 ─┤
S3 ─┘
```

Therefore this experiment is judged primarily on **decomposition / specialization**, not retrieval score alone.

---

# 2. Why NULL was removed

Earlier TAPER variants used competition over:

```text
[NULL, S0, S1, S2, S3]
```

This introduced an ambiguity when adapting QASA:

- Should NULL participate in QASA quality?
- Should NULL-winning tokens be removed?
- Should Edit-Slot probabilities be re-normalized after removing NULL?
- What is the correct coverage universe?

Those choices could alter the mathematical meaning of QASA Quality.

To isolate the actual QASA mechanism, A3.1 removes NULL entirely.

For every valid content token `t`, Edit Slots compete directly:

\[
A_{t,i}
=
\operatorname{softmax}_i(z_{t,i})
\]

such that:

\[
\sum_{i=1}^{L} A_{t,i}=1.
\]

With `L=4`, the uniform ownership baseline is:

\[
A_{t,i}=0.25.
\]

This gives QASA the clean attention contract used by its selection algorithm without a special sink competitor.

---

# 3. Current architecture

The successful path is:

```text
text states
    |
    v
Edit-Slot query/key competition
    |
    v
A[t, i] = softmax over Edit Slots
    |
    +--------------------+
    |                    |
    v                    v
TAPER slot path       QASA FP32 path
    |                    |
slot semantics           winner
slot effects             quality
Edit Slots               novelty
                         coverage
                         selection
    |                    |
    +---------+----------+
              |
              v
       selected Edit Slots
              |
              v
        Slot x Primitive Router
              |
              v
      sequential Executor
              |
              v
        retrieval query
```

Important design constraint:

> QASA decides **slot existence / selection only**.

It does **not** continuously scale transition magnitude.

The transition strength remains learned separately inside the Executor.

---

# 4. QASA implementation

For normalized Edit-Slot attention \(A\), the token-wise winner is:

\[
w_t=\arg\max_i A_{t,i}.
\]

For slot \(i\):

\[
W_i^{win}
=
\sum_{t:w_t=i}A_{t,i}
\]

and:

\[
W_i
=
\sum_t A_{t,i}.
\]

Quality is:

\[
Q_i
=
\frac{W_i^{win}}
{W_i+\epsilon}.
\]

Slots are sorted by descending quality.

For a selected set \(S\), token coverage is:

\[
Covered_S(t)
=
\mathbf{1}
\left[
\sum_{i\in S}A_{t,i}\ge\tau
\right].
\]

Overall coverage is:

\[
Coverage(S)
=
\frac{1}{N}
\sum_t Covered_S(t).
\]

Novelty is:

\[
novelty(i|S)
=
1-
\frac{
\sum_{t\in C_S}A_{t,i}
}{
\sum_t A_{t,i}+\epsilon
}.
\]

Selection proceeds as:

```text
compute quality
    |
sort descending
    |
for each candidate:
    |
    +-> novelty < mu ? skip
    |
    +-> otherwise add
          |
          v
      recompute coverage
          |
          +-> coverage >= rho ? stop
```

A3.1 uses:

```text
tau = 0.5
rho = 0.8
mu  = 0.3
```

---

# 5. FP32 QASA evidence path

Because QASA makes discrete decisions using:

```text
argmax
argsort
novelty < mu
attention >= tau
coverage >= rho
```

its evidence path is deliberately recomputed in FP32.

The QASA path disables outer autocast and recomputes:

```text
text_states.float()
    |
FP32 query/key projection
    |
FP32 einsum logits
    |
FP32 softmax
    |
FP32 QASA selection
```

This avoids allowing AMP rounding alone to change selected slots.

The normal TAPER slot-construction path can still run under AMP.

---

# 6. Training result

Observed 10-epoch trajectory:

| Epoch | Loss | Mean Recall | Active Slots | Hard Active | Dominant | Monopoly | QASA K | QASA Q | QASA Coverage |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.5671 | 51.6269 | 3.08 | 2.67 | 0.254 | 0.000 | 2.67 | 0.258 | 0.976 |
| 2 | 0.7679 | 55.3462 | 2.12 | 2.16 | 0.272 | 0.000 | 2.16 | 0.254 | 0.984 |
| 3 | 0.6610 | 57.2067 | 1.88 | 2.04 | 0.346 | 0.000 | 2.04 | 0.260 | 0.986 |
| 4 | 0.6022 | 58.3406 | 1.43 | 1.91 | 0.405 | 0.000 | 1.91 | 0.256 | 0.987 |
| 5 | 0.5552 | 58.0314 | 1.62 | 1.98 | 0.411 | 0.000 | 1.98 | 0.261 | 0.997 |
| 6 | 0.5197 | 58.3558 | 1.59 | 1.95 | 0.425 | 0.000 | 1.95 | 0.260 | 0.995 |
| 7 | 0.4828 | **58.5844** | 1.50 | 1.99 | 0.428 | 0.000 | 1.99 | 0.258 | 0.998 |
| 8 | 0.4566 | 57.8388 | 1.47 | 1.99 | 0.439 | 0.000 | 1.99 | 0.255 | 0.998 |
| 9 | 0.4283 | 58.0665 | 1.68 | 1.97 | 0.449 | 0.000 | 1.97 | 0.258 | 0.996 |
| 10 | 0.3958 | 58.3716 | 1.63 | 1.96 | 0.453 | 0.000 | 1.96 | 0.257 | 0.995 |

Best validation result:

\[
\boxed{\text{Mean Recall}=58.5844}
\]

at epoch 7.

Retrieval improves substantially during training.

However, the decomposition indicators do **not** improve correspondingly.

The most important example is QASA quality:

```text
epoch 1 : 0.258
epoch 10: 0.257
```

For four near-uniform slots:

\[
\frac{1}{4}=0.25.
\]

Therefore QASA quality remains extremely close to the uniform/symmetric baseline throughout training.

This is a central failure signal:

> Retrieval optimization is succeeding without learning clean Edit-Slot decomposition.

---

# 7. Comprehensive checkpoint diagnosis

A dedicated diagnostic was run to test ownership, representation, causality, execution, routing, and retrieval counterfactuals.

Headline output:

```text
FAILURE MODE: monopoly collapse
Ownership: MONOPOLY / WINNER COLLAPSE evidence
Functional: STRONG shared-task / functional redundancy evidence
```

The automatic label `monopoly collapse` should be interpreted carefully.

The more precise scientific description is:

\[
\boxed{
\text{winner collapse}
+
\text{diffuse ownership}
+
\text{strong functional redundancy}
}
\]

because total soft mass is not concentrated at 90%+ in a single slot.

---

# 8. Ownership diagnosis

Measured:

```text
ownership/token_entropy_norm       = 0.8918
ownership/top1_probability         = 0.4329
ownership/top1_margin              = 0.1598
ownership/winner_active_slot_count = 1.1810
ownership/dominant_mass_share      = 0.4323
ownership/pairwise_cosine          = 0.9806
ownership/pairwise_js_token_map    = 0.0070
```

## 8.1 High entropy

Normalized token entropy:

\[
H_{norm}=0.8918
\]

is close to 1.

This means token ownership remains highly uncertain / diffuse.

It is far from a clean near-one-hot partition.

---

## 8.2 Ownership maps are almost identical

Pairwise ownership cosine:

\[
\boxed{0.9806}
\]

is extremely high.

Token-map Jensen-Shannon divergence:

\[
\boxed{0.0070}
\]

is extremely low.

Together these indicate that different Edit Slots are attending to almost the same token pattern.

This is strong evidence against meaningful token decomposition.

---

## 8.3 Winner collapse despite diffuse soft mass

Average winner-active slots:

\[
\boxed{1.181}
\]

Although soft attention mass is spread over multiple slots, hard `argmax` ownership is typically won by only approximately one slot.

Thus the model exhibits:

```text
soft ownership:
multiple slots retain probability mass

hard ownership:
approximately one slot wins most actual token assignments
```

This is better described as **winner collapse** than pure mass monopoly.

---

# 9. Why QASA coverage is misleading here

Measured:

```text
qasa_selected_k = 1.9974
qasa_quality    = 0.2519
qasa_coverage   = 0.9994
```

At first glance:

\[
Coverage\approx1
\]

could look excellent.

It is not.

For four uniform slots:

\[
A_i=0.25.
\]

With:

\[
\tau=0.5,
\]

two arbitrary slots already satisfy:

\[
0.25+0.25=0.5.
\]

Therefore under perfectly uniform attention, QASA can reach full coverage using only two slots.

This is almost exactly the observed behavior:

```text
QASA K      ~= 2
QASA Q      ~= 0.25
QASA coverage ~= 1
```

Therefore:

\[
\boxed{
\text{high QASA coverage does not imply good decomposition}
}
\]

in this experiment.

Instead, QASA is largely satisfied by diffuse probability mass.

---

# 10. Representation-level specialization

Measured:

```text
slot_effect_pairwise_cosine = 0.7795
slot_effect_effective_rank  = 1.2777
```

Four truly diverse slot effects should occupy meaningfully different directions.

Instead, effective rank is:

\[
\boxed{1.2777}
\]

despite having four Edit Slots.

This indicates that the slot-effect matrix is close to a low-dimensional / approximately rank-1 structure.

In other words:

```text
4 nominal slot effects
        |
        v
mostly one dominant shared direction
```

This is strong evidence of representational redundancy.

---

# 11. Strongest evidence: causal same-task tests

Representation similarity alone cannot prove functional equivalence.

Therefore the diagnosis also performs interventions.

---

## 11.1 Forced-only slot execution

Each slot is independently forced to execute while other slots are disabled.

The edit direction produced by each isolated slot is compared.

Measured:

\[
\boxed{
\text{forced-only effect cosine}=0.9936
}
\]

This is extremely close to 1.

Therefore:

```text
force S0 alone -> edit direction d0
force S1 alone -> edit direction d1
force S2 alone -> edit direction d2
force S3 alone -> edit direction d3
```

and empirically:

\[
d_0\approx d_1\approx d_2\approx d_3.
\]

This is one of the strongest pieces of evidence that:

\[
\boxed{
\text{the four Edit Slots are learning almost the same functional task}
}
\]

rather than distinct edit factors.

---

## 11.2 Drop-one causal intervention

Different selected slots are removed individually.

Measured:

\[
\boxed{
\text{drop-direction pairwise cosine}=0.9820
}
\]

Removing different slots perturbs the final query in almost the same direction.

Thus their causal contributions are highly redundant.

---

# 12. Executor-level specialization

Measured:

\[
\boxed{
\text{state-change pairwise cosine}=0.9890
}
\]

The actual sequential state changes produced by different executed slots are nearly collinear.

Therefore redundancy is not confined to attention maps.

It survives all the way into the Executor dynamics.

This is important:

```text
ownership redundancy
    |
representation redundancy
    |
causal redundancy
    |
Executor transition redundancy
```

All layers independently point to the same failure mode.

---

# 13. Router / primitive specialization

Measured Slot-to-Primitive normalized mutual information:

\[
\boxed{
NMI=0.0075
}
\]

This is approximately zero.

Therefore knowing which slot is being executed provides almost no information about which primitive the Router chooses.

If slots had learned stable, different functional roles, one plausible signal would be non-trivial dependence between:

```text
slot identity <-> primitive usage
```

This dependence is almost absent.

The Router therefore provides no evidence that the four Edit Slots have acquired distinct execution roles.

---

# 14. Retrieval counterfactuals

Measured:

| Variant | Mean Recall | R@10 | R@50 | Mean Rank |
|---|---:|---:|---:|---:|
| Current QASA policy | 58.14 | 45.44 | 70.83 | 124.85 |
| All slots | **58.37** | **45.51** | **71.22** | **116.02** |
| Winner-active only | 55.79 | 43.82 | 67.77 | 147.42 |
| Reference only | 8.72 | 4.56 | 12.89 | 1406.56 |

---

## 14.1 QASA pruning barely matters for retrieval

Difference:

\[
58.37-58.14=0.23.
\]

Therefore enabling all slots instead of QASA-selected slots changes Mean Recall only slightly.

This strongly supports the redundancy diagnosis.

If the selected slots had sharply different responsibilities, removing / restoring them should have a larger functional effect.

Instead:

> QASA versus all-slots is almost retrieval-equivalent because the slots themselves are highly redundant.

---

## 14.2 Winner-active-only inference is worse

The diagnostic counterfactual:

```text
execute only slots that win at least one token
```

obtains:

\[
MR=55.79.
\]

This is substantially below the current QASA policy.

Therefore the proposed follow-up idea of mapping QASA inference directly to `winner_active_mask` should **not** be treated as an obvious rescue mechanism.

The diagnostic has already provided an offline warning that this may degrade retrieval.

This can still be tested as a separate controlled experiment, but it is not supported as the primary fix for the current failure.

---

## 14.3 Reference-only baseline confirms execution is useful

Reference-only:

\[
MR=8.72.
\]

Therefore the Executor is absolutely contributing useful edit information.

The failure is not:

> “the Edit Slots do nothing.”

The actual failure is:

> “the Edit Slots collectively perform useful editing, but fail to decompose that editing into distinct slot-specific roles.”

This distinction is essential.

---

# 15. Final scientific diagnosis

The experiment rejects the strong hypothesis:

\[
\boxed{
\text{QASA selection alone is sufficient to induce TAPER Edit-Slot specialization}
}
\]

The evidence instead supports:

\[
\boxed{
\text{QASA behaves mainly as a slot selection / pruning operator}
}
\]

while the underlying Edit Slots remain strongly redundant.

The model learns retrieval while preserving a shortcut in which several nominal slots can encode approximately the same edit behavior.

---

# 16. Failure mode summary

A3.1 is best summarized as:

```text
          retrieval supervision
                   |
                   v
        useful global edit signal
                   |
        +----------+----------+
        |          |          |
       S0         S1         S2 ... S3
        \          |          /
         \         |         /
          +-- similar role --+
                   |
                   v
             QASA pruning
                   |
             ~2 slots kept
                   |
                   v
              Executor
                   |
                   v
          good retrieval result
```

rather than:

```text
text modification
      |
      +--> color slot
      +--> shape slot
      +--> garment slot
      +--> detail slot
```

---

# 17. What this experiment DOES prove

A3.1 provides several useful conclusions.

## 17.1 Selection is not specialization

A mechanism that decides:

```text
which slots survive
```

does not automatically create a reason for surviving slots to learn different functions.

---

## 17.2 Retrieval performance is not evidence of decomposition

Mean Recall reaches:

\[
58.5844
\]

while QASA quality remains approximately uniform and causal slot roles remain nearly identical.

Therefore retrieval can improve substantially through a non-decomposed shortcut.

---

## 17.3 Attention-only metrics are insufficient

The strongest conclusion comes from agreement among:

1. token ownership;
2. slot-effect geometry;
3. effective rank;
4. forced-only execution;
5. drop-one intervention;
6. actual state-change geometry;
7. slot/primitive routing statistics.

The failure is therefore not merely an attention visualization artifact.

---

## 17.4 QASA coverage can be trivially high

With four nearly uniform slots and `tau=0.5`, selecting two slots already reaches the coverage threshold.

Therefore QASA coverage must never be interpreted in isolation.

High coverage is only meaningful when accompanied by evidence of sharp / specialized attention.

---

# 18. What this experiment DOES NOT prove

This branch does **not** prove that QASA is generally ineffective.

It only shows:

> Under the current TAPER architecture, supervision, four-slot competition, and QASA adaptation, QASA selection is not sufficient to produce the desired slot specialization.

It does not rule out:

- a different ownership mechanism;
- iterative refinement;
- different QASA thresholds;
- using QASA after specialization has already emerged;
- QASA combined with explicit specialization pressure;
- alternative sparse mappings;
- object-centric architectures closer to canonical Slot Attention.

---

# 19. Recommended interpretation of the branch

This branch should be retained as a **negative but informative experiment**.

Recommended label:

> **A3.1 — QASA selection successfully prunes Edit Slots, but does not solve functional slot redundancy.**

Do not describe the branch as simply:

> “QASA failed because retrieval was bad.”

That is inaccurate.

Retrieval is reasonably strong.

The actual scientific failure is:

\[
\boxed{
\text{the mechanism fails the decomposition objective}
}
\]

despite successful retrieval learning.

---

# 20. Follow-up implications

The next research question should no longer be:

> How do we select fewer slots?

The stronger question exposed by A3.1 is:

\[
\boxed{
\text{What mechanism creates pressure for slots to acquire different causal roles?}
}
\]

A future solution should ideally attack one or more of:

- token ownership differentiation;
- residual / unclaimed evidence;
- functional anti-redundancy;
- mutually exclusive causal responsibility;
- role-conditioned routing;
- sequential residual specialization;
- sparse competition that changes relative Edit-vs-Edit assignment;
- explicit diversity in slot effects or transition directions;
- mechanisms that prevent multiple slots from explaining the same edit evidence.

However, any new mechanism must be audited carefully to ensure it does not merely make representations numerically different while preserving the same causal function.

The causal probes introduced for A3.1 should therefore be reused in all future branches.

---

# 21. Diagnostic script

Use:

```bash
python src/diagnose_taper_a31_qasa_specialization.py \
  --checkpoint "$(find outputs -type f -name best.pt -printf '%T@ %p\n' \
    | sort -nr \
    | head -1 \
    | cut -d' ' -f2-)" \
  --max-queries-per-category 512
```

Full validation:

```bash
python src/diagnose_taper_a31_qasa_specialization.py \
  --checkpoint "$(find outputs -type f -name best.pt -printf '%T@ %p\n' \
    | sort -nr \
    | head -1 \
    | cut -d' ' -f2-)" \
  --max-queries-per-category 0
```

Outputs:

```text
reports/taper_a31_qasa_specialization_diagnosis.json
reports/taper_a31_qasa_specialization_diagnosis.md
```

---

# 22. Core metrics to preserve for future experiments

Every future Edit-Slot experiment should retain at least:

```text
ownership/token_entropy_norm
ownership/top1_probability
ownership/top1_margin
ownership/winner_active_slot_count
ownership/dominant_mass_share
ownership/pairwise_cosine
ownership/pairwise_js_token_map

qasa/selected_k
qasa/quality_mean
qasa/final_coverage

representation/slot_effect_pairwise_cosine
representation/slot_effect_effective_rank

causal/forced_only_effect_pairwise_cosine
causal/drop_direction_pairwise_cosine

execution/state_change_pairwise_cosine

router slot<->primitive NMI
```

The most important rule is:

> Never claim Edit-Slot specialization from retrieval performance, selection count, or attention entropy alone.

A convincing decomposition requires agreement between **ownership, representation, causal intervention, and execution behavior**.

---

# 23. Final verdict

\[
\boxed{
\textbf{A3.1 FAILS THE EDIT-SLOT DECOMPOSITION OBJECTIVE}
}
\]

More precisely:

\[
\boxed{
\text{winner collapse}
+
\text{diffuse / overlapping ownership}
+
\text{low-rank slot effects}
+
\text{near-identical causal edit roles}
+
\text{near-identical state transitions}
+
\text{negligible slot/primitive specialization}
}
\]

while simultaneously:

\[
\boxed{
\text{retrieval learning remains successful}
}
\]

This makes A3.1 an important negative result:

> **QASA can prune TAPER Edit Slots, but pruning alone does not create specialized causal responsibility.**