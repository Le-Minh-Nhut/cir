# A3.4: multi-error functional ownership

## Hypothesis and fixed baseline

A3.4 tests whether distinct Edit Slots can be trained to reduce distinct useful
retrieval-error directions. It starts from A3.3 cell C and keeps its dataflow:

```text
slot_value_source=contextual
slot_effect_in_value=false
slot_value_assignment=soft_shared
```

The teacher, contextual KEY/VALUE, ownership, QASA, Executor, primitive bank,
retrieval objective, optimizer, schedule, caches, seed, and FashionIQ protocol
are unchanged. `functional_ownership.enabled=false` is the exact Run C control.

## Error modes and exact interventions

For each query, the current in-batch target bank is scored once and the highest
scoring `H` non-positive target IDs are retained. The implementation keeps the
per-negative loss instead of summing it:

```text
ell_j(q) = softplus((score(q, y_j-) - score(q, y+) + margin) / temperature)
```

It executes a fixed coalition set: EMPTY, every singleton, and every slot pair.
Every coalition still runs the normal fixed `num_slots` Executor loop, so
functional credit never changes the compute ticket. The singleton functional
effect is the exact finite intervention

```text
Phi[s,j] = ell_j(q_empty) - ell_j(q_singleton_s).
```

Attention, ownership mass, overlap, and slot cosine are not substituted for
`Phi`.

## Rank-gated functional mode ownership

A3.4 v1 selected the row with the largest total `Phi` before residualization.
That removed later clone credit but rewarded a global atom that already solved
every mode, and zero-credit clones received no new job. The fixed experiment
separates a training assignment from residual acceptance.

The task rank is the participation-ratio rank of the per-negative target-directed
gradient matrix `G`. With the default rank gate:

```text
rank <= 1 + rank_threshold  -> K_eff = 1
otherwise                   -> K_eff = ceil(rank), capped by slots and modes
```

For `K_eff=1`, one owner is allowed and no capacity is imposed. For verified
multi-mode samples, the default adaptive upper capacity is
`ceil(H / K_eff)` modes per owner. An explicit positive `mode_capacity` can
replace it, but never applies to rank-one samples.

The first ownership round constructs detached `B_train[s,j]` by descending
positive functional utility, subject to:

```text
sum_s B_train[s,j] <= 1
sum_j B_train[s,j] <= capacity       # multi-mode samples only
B_train[s,j] = 0 when Phi[s,j] <= eps
```

Unassigned/NULL modes are legal; there is no lower quota. `B_train` is a
mode-specific training job, not evidence that specialization already exists.
This distinction matters for identical Phi rows: the rank gate may assign
different jobs to create symmetry-breaking pressure, while the residual oracle
still reports the current effects as clones and marks the sample unresolved.

## Block residual acceptance

After assignment, the detached positive full rows of `Phi` are residualized by
block Gram–Schmidt. After each accepted block, ownership proposals are recomputed
on unsolved modes and remaining owner capacity. The full row—not a mode-ID-masked
row—is projected, so exact/span clones lose unique residual credit even if their
provisional jobs used different mode IDs.

This produces a separate `B_unique`: current finite effects accepted as unique
functional work. A multi-mode global giant with useful specialists cannot own
all jobs because of the upper capacity. If only the giant has positive utility,
its excess modes remain NULL and `giant_owner` plus `unresolved_multimode` are
reported. No useless slot is assigned merely to make the table balanced. A
true rank-one task still allows one owner without a penalty.

Pair lookahead uses the exact interaction

```text
H[a,b,j] = ell_j(q_a) + ell_j(q_b)
           - ell_j(q_pair_ab) - ell_j(q_empty).
```

The strongest positive candidate pair per sample participates only on modes
left unowned by `B_train`, and only where the pair improves over EMPTY. Positive
interaction that is still functionally worse than EMPTY is not credited. Thus a
pair cannot be rewarded for re-solving an owned singleton mode, while an XOR-like
pair whose singleton effects are weak survives. This is not a full arbitrary-
group solver; triple-only synergy remains an open limitation.

## Loss and gradient flow

The training objective is

```text
L = L_retrieval + lambda_func * L_functional.
```

For each `B_train` singleton or residual pair block, its detached positive
mode-specific utility is normalized over its assigned modes and weights
`ell_j(q_block) - stopgrad(ell_j(q_empty))`. `L_functional` is the mean over
assigned blocks, equivalently a safely baselined negative marginal improvement.
This minimizes error only through the exact credited singleton/group query;
non-credited slot tensors are masked out of that intervention. Shared
Executor/query-head parameters remain shared by design. The discrete mining,
residual ordering, rank gate, and ownership weights are detached, but the
assigned query/loss path is differentiable.

For the auxiliary view only, the credited slot's ownership-logit row remains
live while all competitor rows are value-identical detached constants inside
the slot-axis softmax. This preserves the exact Run-C forward slot but removes
direct functional-gradient leakage into non-owner `slot_queries` through the
softmax denominator. The main retrieval forward is never detached or changed.
Shared projections, Executor, and query head legitimately remain shared.

The real `q_empty` loss is used only to estimate detached finite improvement; it
is not optimized as a negative baseline, avoiding a trivial incentive to make
the empty coalition worse.

## Diagnostics and red-team interpretation

Training reports:

```text
functional/error_mode_rank
functional/residual_active_modes
functional/credited_slots
functional/unique_mode_coverage
functional/redundant_credit_fraction
functional/pair_synergy_fraction
functional/inferred_k_eff
functional/owned_mode_count
functional/unowned_positive_mode_count
functional/max_modes_per_owner
functional/giant_owner_fraction
functional/ownership_row_similarity
functional/unresolved_multimode_fraction
functional/loss
```

`residual_active_modes` counts modes still represented by another positive
residual immediately after the first accepted block. `owned_mode_count` and
`max_modes_per_owner` describe `B_train`; `unique_mode_coverage` and unresolved
status describe residual-accepted functional work. High assignment coverage is
therefore not itself success. `ownership_row_similarity` is the mean positive
Phi-row cosine diagnostic only; it is never a loss. `credited_slots` is not a
target and no minimum K is enforced.

`giant_owner` means an inferred multi-mode sample has only one residual-accepted
owner and that owner has positive utility on more than one mode.
`unresolved_multimode` means the residual-accepted owners are fewer than
`K_eff`, or some positive mode remains outside `B_unique`. These definitions
distinguish true rank one, available specialists, and giant/clone-only worlds.
The fixed `rank_threshold=0.25` is a predeclared numerical rank gate, not tuned
against the downstream P0 result.

The v0 negative bank is `in_batch`. The provenance records this explicitly and
`functional/heldout_validation_available=0` flags that ownership is not yet
cross-fitted on an independent bank B. Negative-ID memorization is therefore an
open loophole to test in the next iteration; the low-level credit-loss API
already separates detached bank-A credit from differentiable block losses so a
bank-B loss can be supplied without changing the residual oracle.

Success cannot be inferred from different masks, more active slots, or a lower
slot cosine. A global giant, a clone, and orthogonal junk are all rejected by
the synthetic tests without forcing a balanced ownership table. FashionIQ must
still be judged with the unchanged P0 metrics: especially Phi effective rank,
Phi row cosine, SINGLE/FULL, REPEAT/FULL, MEANxK/FULL, K95/K99, QASA functional
precision/recall, and retrieval recall. If Phi remains rank one and repeated or
mean slots still recover the full gain, the experiment failed functional
specialization even if recall is high.

## Commands

Run C reproduction:

```bash
python src/train.py experiment=taper_e2e \
  experiment.model.slot_value_source=contextual \
  experiment.model.slot_effect_in_value=false \
  experiment.model.slot_value_assignment=soft_shared \
  experiment.functional_ownership.enabled=false
```

A3.4:

```bash
python src/train.py experiment=taper_e2e \
  experiment.model.slot_value_source=contextual \
  experiment.model.slot_effect_in_value=false \
  experiment.model.slot_value_assignment=soft_shared \
  experiment.functional_ownership.enabled=true
```

Every run must start from fresh TAPER weights. Frozen CSMCIR teacher/cache reuse
is unchanged. Checkpoint provenance prevents loading across the enabled toggle
or any functional configuration field.
