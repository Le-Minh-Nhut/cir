# A3.2 — FG-CLIP2-Large / No-CSMCIR: Slot-Collapse Diagnosis

**Branch:** `exp/e2e-a3.2-fgclip2-no-csmcir`  
**Audited remote HEAD:** `48408986240dca7781478372b960822dacab87e1` (`v4`)  
**Experiment status:** **FAILED AS A MULTI-SLOT DECOMPOSITION EXPERIMENT**  
**Retrieval status:** still learns/improves despite decomposition collapse  
**Primary forensic conclusion:** replacing the previous representation/teacher path with frozen FG-CLIP2-Large token states **does not by itself prevent Edit-Slot collapse**.

---

## 1. Why this branch exists

A3.2 was designed to isolate a specific hypothesis from the previous TAPER experiments:

> The previous text representation may be too globally mixed / insufficiently local, making it difficult for multiple Edit Slots to specialize. If a stronger fine-grained text representation is supplied directly, slot decomposition may become easier and collapse may reduce.

To test this, the branch removes the active CSMCIR teacher path and uses frozen **FG-CLIP2-Large** features directly:

```text
FG-CLIP2-Large contextual text token states [B,N,1024]
                       ↓
             competitive ownership
                       ↓
              mass-aware pooling
                       ↓
              Edit Slots [B,4,1024]
                       ↓
                     QASA
                       ↓
                   Executor
        ↑                              
FG-CLIP2 reference image [B,1024]
                       ↓
                query [B,1024]
                       ↓
      FG-CLIP2 gallery retrieval
```

The intended scientific isolation is important:

- no active CSMCIR teacher composition;
- no `slot_effect` teacher counterfactual;
- no learned MLP/projection after slot pooling;
- Edit Slots use pooled FG-CLIP2 token semantics directly;
- QASA and Executor remain;
- optimization is end-to-end through retrieval loss.

Therefore this experiment asks whether **representation replacement alone**, while keeping the decomposition/execution structure, is sufficient to rescue multi-slot specialization.

---

## 2. Frozen FG-CLIP2 feature contract

The branch pins:

```text
model_id = qihoo360/fg-clip2-large
revision = 4d1d5dc35c716902f07c172dbfc23b82a7bc6bf3
```

Feature dimensions:

```text
text_dim      = 1024
slot_dim      = 1024
reference_dim = 1024
query_dim     = 1024
state_dim     = 512
num_slots     = 4
num_primitives = 8
```

Observed cache shapes in the actual run:

```text
Train FG-CLIP2 images: (45429, 1, 1024)
Val   FG-CLIP2 images: (15415, 1, 1024)
Train text:            (18000, 64, 1024)
Val   text:            (6016, 64, 1024)
```

FG-CLIP2 is frozen and precomputed offline. Training TAPER therefore optimizes the slot decomposition, QASA-dependent execution path, reference-state mapping, primitive/router/transition machinery, and query head—not the FG-CLIP2 backbone itself.

---

## 3. Exact Edit-Slot mechanism being diagnosed

### 3.1 Competitive ownership

For each valid text token `n` and Edit Slot `j`, current code computes approximately:

\[
z_{jn} = \frac{q_j^\top k_n}{\sqrt{D}\,T}
\]

followed by a softmax **across Edit Slots**:

\[
p(j\mid n)=\operatorname{softmax}_j(z_{jn}).
\]

Thus for each valid token:

\[
\sum_{j=1}^{4} p(j\mid n)=1.
\]

### 3.2 Critical implementation fact: no explicit NULL/dustbin in this branch

The active `_competitive_ownership` implementation currently contains only the four Edit Slots. There is no separate NULL query/logit participating in this softmax.

Therefore this run is specifically diagnosing:

> **4-way Edit-Slot competition without an explicit NULL/dustbin owner.**

There is a stale error string elsewhere in `compute_stage1_loss()` mentioning “competitive NULL ownership”, but that string does **not** match the actual active `_competitive_ownership` implementation in this branch.

This distinction should be preserved when comparing A3.2 with earlier NULL/dustbin hypotheses.

### 3.3 Mass-aware pooling

Current slot pooling is:

\[
m_j = \sum_n p(j\mid n)
\]

\[
s_j = \frac{\sum_n p(j\mid n)h_n}{\max(m_j,1)}
\]

with activity:

\[
a_j = \min(m_j,1)
\]

and final Edit Slot:

\[
e_j=s_j a_j.
\]

In code-equivalent form:

```python
slot_mass = slot_masks.sum(dim=2)
weighted_sum = einsum(slot_masks, text_states)
slot_activity = slot_mass.clamp(max=1.0)
slot_semantics = weighted_sum / slot_mass.clamp_min(1.0)
edit_slots = slot_semantics * slot_activity.unsqueeze(-1)
```

There is deliberately **no learned post-pooling transform**.

---

## 4. Objective actually optimized

The active training objective is only:

```yaml
loss_weights:
  retrieval_loss: 1.0
```

The slot diagnostics are computed under `torch.no_grad()` and appended only for logging. They do **not** produce optimization pressure.

Therefore the optimizer is never explicitly told that it should prefer:

- multiple non-empty slots;
- balanced ownership;
- semantic diversity;
- low slot overlap;
- low monopoly;
- one-edit-per-slot behavior;
- a minimum number of active slots;
- capacity constraints;
- coverage distributed across slots.

It is rewarded only for producing a query representation that retrieves the target image.

This fact is central to interpreting the collapse.

---

## 5. Meaning of the logged diagnostics

The relevant metrics in this branch are not heuristic names; they have specific code definitions.

### `active_slots`

Logged from `diagnostic/ownership_active_slot_count`.

For every valid token, the slot with maximal ownership probability is taken as the hard winner. `active_slots` is the average number of slots that win **at least one** valid token.

Interpretation:

```text
4.0 → all four slots win at least one token on average
1.0 → all valid tokens have the same hard-winning slot
```

### `hard_active`

Logged from `diagnostic/execution_hard_active_slot_count`.

This is the mean number of slots actually marked active for execution after QASA selection (minus explicitly disabled slots, if any).

### `dominant`

Logged from `diagnostic/dominant_slot_share`:

\[
\frac{\max_j m_j}{\sum_j m_j}
\]

averaged over samples.

Unlike `active_slots`, this uses **soft ownership mass**, not only argmax winners.

Therefore:

```text
dominant ≈ 1.0
```

means one slot owns almost all of the actual soft assignment mass, not merely that it wins a close argmax.

### `monopoly`

Logged from `diagnostic/near_monopoly_fraction`.

For each sample, monopoly is true when:

\[
\text{dominant slot share} \ge 0.90.
\]

The printed value is the fraction of samples satisfying this condition.

Thus:

```text
monopoly = 1.000
```

means essentially every training sample in the averaged epoch statistics has at least 90% of ownership mass concentrated in one slot.

### `qasa_k`

Mean number of slots selected by QASA.

### `qasa_q`

Mean QASA quality **across all slots**, not merely selected slots.

This matters greatly in the collapsed regime. If one slot has quality ≈1 and the other three have quality ≈0, then:

\[
\frac{1+0+0+0}{4}=0.25.
\]

Therefore the observed `qasa_q ≈ 0.250` is fully consistent with **one useful slot + three useless slots**. It is not evidence that selected-slot quality is only 0.25.

### `qasa_cov`

Mean final token coverage obtained by QASA's selected slots according to its thresholded coverage rule.

In a collapsed solution, `qasa_cov = 1.0` can be achieved by **one slot covering everything**. Therefore high QASA coverage does not imply semantic decomposition.

---

## 6. Observed collapse trajectory

Actual run log:

| Epoch | Loss | Mean Recall | Active Slots | Hard Active | Dominant Share | Monopoly Fraction | QASA K | QASA Q | QASA Coverage |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.6225 | 12.6952 | 2.03 | 2.06 | 0.265 | 0.000 | 2.06 | 0.253 | 0.968 |
| 2 | 2.1159 | 16.2049 | 1.37 | 1.57 | 0.659 | 0.384 | 1.57 | 0.257 | 0.988 |
| 3 | 1.8784 | 19.2868 | 1.00 | 1.00 | 1.000 | 1.000 | 1.00 | 0.250 | 1.000 |
| 4 | 1.7052 | 21.4593 | 1.00 | 1.00 | 1.000 | 1.000 | 1.00 | 0.250 | 1.000 |
| 5 | 1.5927 | 23.3709 | 1.00 | 1.00 | 1.000 | 1.000 | 1.00 | 0.250 | 1.000 |
| 6 | 1.4969 | 24.3342 | 1.00 | 1.00 | 1.000 | 1.000 | 1.00 | 0.250 | 1.000 |
| 7 | 1.4250 | 24.6440 | 1.00 | 1.00 | 1.000 | 1.000 | 1.00 | 0.250 | 1.000 |
| 8 | 1.3635 | 26.0462 | 1.00 | 1.00 | 1.000 | 1.000 | 1.00 | 0.250 | 1.000 |
| 9 | 1.3258 | 26.0715 | 1.07 | 1.03 | 0.972 | 0.933 | 1.03 | 0.259 | 0.995 |
| 10 | 1.2611 | 26.5121 | 1.00 | 1.00 | 0.999 | 1.000 | 1.00 | 0.250 | 1.000 |

### Immediate reading

The transition is extremely fast:

```text
Epoch 1
active_slots = 2.03
monopoly     = 0.000
        ↓
Epoch 2
active_slots = 1.37
monopoly     = 0.384
        ↓
Epoch 3
active_slots = 1.00
dominant     = 1.000
monopoly     = 1.000
```

By epoch 3, the decomposition is functionally collapsed.

Epoch 9 shows a small transient relaxation (`active_slots=1.07`, `dominant=0.972`), but epoch 10 immediately returns to near-perfect monopoly. This does not look like healthy specialization emerging late; it looks like a stable one-slot attractor with minor optimization noise.

---

## 7. The strongest empirical finding

The most important result is **not merely that collapse happened**.

It is that retrieval continued improving **after the decomposition had already collapsed**:

```text
Epoch 3 : mean_recall = 19.2868  | monopoly = 1.000
Epoch 4 : mean_recall = 21.4593  | monopoly = 1.000
Epoch 5 : mean_recall = 23.3709  | monopoly = 1.000
Epoch 6 : mean_recall = 24.3342  | monopoly = 1.000
Epoch 7 : mean_recall = 24.6440  | monopoly = 1.000
Epoch 8 : mean_recall = 26.0462  | monopoly = 1.000
Epoch 10: mean_recall = 26.5121  | monopoly = 1.000
```

Therefore, for this architecture and run:

\[
\boxed{\text{retrieval improvement is compatible with complete slot collapse}}
\]

The optimizer has discovered a valid shortcut:

```text
all modification evidence
        ↓
one dominant Edit Slot
        ↓
QASA selects that slot
        ↓
Executor performs the useful computation
        ↓
retrieval loss decreases
```

The retrieval objective has no reason to reject this solution.

This is a much stronger diagnosis than simply observing low slot diversity.

---

## 8. Where does the failure occur?

### 8.1 Collapse is already present before Executor

`active_slots` and `dominant` are computed from `slot_masks` / `slot_mass`, before the Executor performs recurrent transitions.

At epoch 3:

```text
active_slots = 1.00
dominant     = 1.000
```

Therefore the failure is already present in the **token-to-slot ownership stage**.

The Executor is receiving an already collapsed decomposition.

### 8.2 It is not merely a hard-argmax artifact

If only `active_slots=1.0` were observed, one could argue that probabilities might still be relatively soft and merely share the same argmax.

But simultaneously:

```text
dominant ≈ 1.0
monopoly ≈ 1.0
```

The dominant metric uses the actual soft slot mass. Therefore the distribution itself has concentrated, not merely the hard winner.

### 8.3 QASA is downstream of the collapse

QASA sees the attention induced by the same learned slot-query/key system. Once a single slot owns almost all evidence, QASA selecting one slot is rational under its own criteria.

The observed regime:

```text
qasa_k   = 1.00
qasa_q   = 0.250
qasa_cov = 1.000
```

is approximately what is expected from:

```text
Slot A: useful / wins everything / covers everything
Slot B: useless
Slot C: useless
Slot D: useless
```

QASA is therefore **not the origin of the first collapse signal**, and in the current integration it does not restore lost specialization.

---

## 9. Likely optimization mechanism

The competitive ownership operation creates a natural winner-take-all feedback loop.

Suppose one slot starts slightly better for a subset of tokens:

```text
S0  0.24
S1  0.24
S2  0.28   ← small initial advantage
S3  0.24
```

Because retrieval is the only optimized objective, if S2 happens to contribute more useful modification information, downstream gradients can strengthen the query/key configuration that routes even more evidence to S2:

```text
0.28 → 0.40 → 0.65 → 0.90 → ~1.00
```

As S2 captures more tokens, its pooled semantic vector becomes an increasingly complete representation of the whole modification. That makes it even more sufficient for retrieval, producing further useful gradient through the same path.

This is a self-reinforcing symmetry-breaking process:

\[
\text{small assignment advantage}
\rightarrow
\text{more information captured}
\rightarrow
\text{greater downstream usefulness}
\rightarrow
\text{more favorable gradient}
\rightarrow
\text{larger assignment advantage}.
\]

### Additional reinforcing mechanism: slot activity scaling

For a low-mass slot:

\[
a_j=\min(m_j,1).
\]

Once its mass falls below 1, its final Edit Slot is explicitly attenuated:

\[
e_j=s_j m_j \quad (m_j<1).
\]

A dominant slot with `m_j >= 1` remains at full activity `a_j=1`, while a dying slot gets progressively smaller in magnitude.

This is **not established as the initial cause**, because early in training multiple slots can all have mass above 1. However, once a loser crosses below unit mass, this mechanism can make recovery harder and reinforce an already-developing monopoly.

This should be treated as a mechanism-level risk to test, not yet as a proven sole root cause.

---

## 10. What this experiment falsifies

### Falsified / strongly rejected for the current setup

#### H1 — “Replacing the old text representation with FG-CLIP2-Large alone will prevent slot collapse.”

**Rejected.**

Collapse reaches approximately complete monopoly by epoch 3.

#### H2 — “Removing CSMCIR teacher dependence is sufficient to rescue multi-slot decomposition.”

**Rejected as a sufficient intervention.**

A3.2 has no active CSMCIR teacher composition, yet collapse remains severe.

This does **not** prove that the teacher had zero effect in earlier branches. It shows only that teacher removal alone is not enough.

#### H3 — “QASA selection alone is enough to preserve multiple meaningful Edit Slots.”

**Rejected for this integration.**

QASA converges to selecting the already dominant single slot.

#### H4 — “A strong retrieval score necessarily implies useful compositional slot decomposition.”

**Rejected.**

Retrieval improves substantially while the slot system remains collapsed.

---

## 11. What this experiment does NOT falsify

The result must not be overinterpreted.

It does **not** establish that:

- FG-CLIP2 token representations are perfectly local;
- encoder global mixing is irrelevant;
- correction dictionaries have no effect;
- a NULL/dustbin mechanism cannot help;
- balanced assignment cannot help;
- sparse assignment cannot help;
- capacity constraints cannot help;
- OT/Sinkhorn/partial OT cannot help;
- entropy/diversity/anti-monopoly pressure cannot help;
- iterative slot refinement cannot help under a different objective;
- the Executor is optimal;
- four slots is the correct number of slots.

The narrow conclusion is:

\[
\boxed{
\text{better/different frozen token representation alone is insufficient}
}
\]

for preventing collapse in the current competitive-ownership + retrieval-only setup.

---

## 12. Root-cause scope after A3.2

The evidence now shifts the priority away from treating the encoder as the sole bottleneck.

The strongest remaining problem is **identifiability / optimization pressure**:

> Why should four interchangeable slots learn four distinct semantic responsibilities when one slot can encode enough modification information to optimize retrieval?

With only retrieval supervision, many internal decompositions can produce the same useful final query. A one-slot solution is therefore not forbidden.

The current experiment provides direct evidence that the model is willing to exploit that equivalence.

A useful high-level statement is:

\[
\boxed{
\text{multi-slot structure exists architecturally, but is not identified by the objective}
}
\]

This is currently a stronger working diagnosis than “the encoder is too global.”

---

## 13. Why high QASA coverage is misleading here

The final epochs look superficially excellent if one reads only QASA coverage:

```text
qasa_cov = 1.000
```

But coverage asks whether selected slots cover sufficient token regions—not whether those regions are decomposed across multiple independent semantic slots.

A collapsed attention map can satisfy coverage trivially:

```text
one slot owns every valid token
        ↓
select that one slot
        ↓
all valid tokens are covered
        ↓
coverage = 1.0
```

Therefore:

\[
\boxed{\text{coverage} \neq \text{specialization}}
\]

and:

\[
\boxed{\text{QASA K}=1,\ \text{coverage}=1\text{ is compatible with total collapse}.}
\]

This metric distinction should be preserved in all later analyses.

---

## 14. Retrieval metric definition

The current FashionIQ evaluation computes per-category `Recall@10` and `Recall@50`, macro-averages them across categories, then reports:

\[
\text{mean\_recall}
=
\frac{\text{macro Recall@10}+\text{macro Recall@50}}{2}.
\]

The branch is configured for `fashioniq_original` gallery construction through the project configuration unless locally overridden.

Thus the reported `26.5121` is a retrieval metric, **not** a slot-specialization score.

---

## 15. Important reproducibility discrepancy to preserve

The observed terminal run prints:

```text
Epoch 1/10
...
Epoch 10/10
```

so the executed run used **10 epochs**.

However, the audited remote HEAD `4840898...` currently contains:

```yaml
num_epochs: 100
```

in `conf/experiment/taper_e2e.yaml`.

Therefore at least one of the following was true for the local run:

1. the local config had an uncommitted `num_epochs: 10` change; or
2. another local override/change affected the runtime.

Before treating this run as a fully reproducible archival result, save:

```bash
git status
git diff
```

and the exact Hydra resolved config/output directory.

This discrepancy does **not** affect the collapse diagnosis—the collapse already occurs by epoch 3—but it matters for exact experiment provenance.

---

## 16. Recommended branch verdict

Do **not** continue this run to 100 epochs merely to see whether collapse disappears.

The relevant question for A3.2 has already been answered cleanly:

```text
Can frozen FG-CLIP2-Large direct token semantics,
without active CSMCIR teacher composition,
prevent the current Edit-Slot system from collapsing?

Observed answer: NO.
```

The branch should be preserved as a negative-result baseline rather than repeatedly modified until it succeeds.

Suggested status label:

```text
A3.2 — FGCLIP2 direct semantics
RESULT: RETRIEVAL LEARNS, MULTI-SLOT DECOMPOSITION COLLAPSES
```

---

## 17. What should be attacked next

The next research branch should primarily attack the **assignment/objective geometry**, not merely swap another encoder.

The target property should be explicit:

> Make the one-slot solution either impossible, capacity-limited, or objectively worse than a genuine multi-part decomposition—without forcing arbitrary uniformity when the text truly contains only one edit.

Relevant families already under consideration include:

- explicit NULL/dustbin ownership;
- sparse but non-monopolistic assignment;
- balanced/capacity-constrained transport;
- partial/unbalanced optimal transport;
- Entmax/Sparsemax-style support selection;
- anti-monopoly or load-balancing pressure;
- information/capacity bottlenecks per slot;
- coverage/exclusivity constraints that distinguish “one slot covers all” from genuine decomposition.

These are **next research directions**, not conclusions of A3.2. They should be evaluated in separate branches so this negative result remains clean.

---

## 18. Minimal forensic checks worth preserving from this run

If the run output/checkpoint is still available, archive at minimum:

```text
best.pt
last.pt
resolved Hydra config
terminal/log output
branch SHA
git diff / git status
FG-CLIP2 image manifests
FG-CLIP2 text manifests
```

For a later deeper postmortem, the most useful additional measurements would be:

1. per-slot `slot_mass_mean` by epoch;
2. per-slot hard winner counts by epoch;
3. assignment entropy by epoch;
4. slot overlap by epoch;
5. which physical slot ID becomes the monopolist;
6. whether monopolist identity is seed-dependent;
7. gradients/norms of slot queries during epochs 1–3;
8. checkpoint-level slot-drop functional tests.

These tests would characterize **how** symmetry breaks, but they are not required to establish that collapse occurred.

---

# Final diagnosis

The experiment produced a very clear negative result.

A3.2 successfully removed the active teacher dependency and replaced the old representation path with frozen FG-CLIP2-Large contextual token states. Despite this, the four-way competitive slot system rapidly converged to a one-slot monopoly:

\[
\text{active slots}: 2.03 \rightarrow 1.37 \rightarrow 1.00
\]

\[
\text{dominant share}: 0.265 \rightarrow 0.659 \rightarrow 1.000
\]

\[
\text{monopoly fraction}: 0.000 \rightarrow 0.384 \rightarrow 1.000.
\]

At the same time:

\[
\text{mean recall}: 12.70 \rightarrow 26.51.
\]

The central scientific observation is therefore:

\[
\boxed{
\text{the retrieval objective can continue improving after the intended decomposition has died}
}
\]

and the central scope update is:

\[
\boxed{
\text{representation quality/locality alone is not sufficient to identify multiple Edit Slots}
}
\]

The next phase should treat **one-slot shortcut / decomposition identifiability / competitive-assignment dynamics** as first-class root-cause targets.

---

## Branch status summary

```text
FG-CLIP2 cache contract          PASS
Frozen backbone isolation        PASS
No active CSMCIR teacher         PASS
Direct 1024-D pooled Edit Slots  PASS
Retrieval optimization           WORKING
QASA execution path              WORKING MECHANICALLY
Multi-slot specialization        FAIL
Slot-collapse prevention         FAIL
One-slot shortcut eliminated     FAIL
Main hypothesis of A3.2          REJECTED AS SUFFICIENT
```

**Preserve this branch as evidence. Do not rewrite its failure away.**