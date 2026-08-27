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

## Block residual credit

The detached positive rows of `Phi` are treated as vectors over hard-negative
error modes. A greedy block Gram–Schmidt oracle repeatedly selects the remaining
row with the largest positive residual utility, records that residual as its
mode credit, and removes its span from all other rows. Consequently, an exact or
span clone of an already credited slot has approximately zero residual credit.
Independent useful effects survive. A global giant is not artificially split:
if every other effect is a clone of the giant, only the giant is credited. A
rank-one task is therefore allowed to produce one effective functional block.

Pair lookahead uses the exact interaction

```text
H[a,b,j] = ell_j(q_a) + ell_j(q_b)
           - ell_j(q_pair_ab) - ell_j(q_empty).
```

The strongest positive candidate pair per sample is retained as a group credit,
but only on modes where the pair also improves over EMPTY. Positive interaction
that is still functionally worse than EMPTY is not credited. This protects an
XOR-like pair whose singleton effects are weak. It is not a full arbitrary-group
solver; triple-only synergy remains an open limitation.

## Loss and gradient flow

The training objective is

```text
L = L_retrieval + lambda_func * L_functional.
```

For each credited singleton or pair block, its detached positive residual credit
is normalized over negative modes and weights
`ell_j(q_block) - stopgrad(ell_j(q_empty))`. `L_functional` is the mean over
credited blocks, equivalently a safely baselined negative marginal improvement.
This minimizes error only through the exact credited singleton/group query;
non-credited slot tensors are masked out of that intervention. Shared
Executor/query-head parameters remain shared by design. The discrete mining,
residual ordering, and credit weights are detached, but the credited query/loss
path is differentiable.

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
functional/loss
```

`residual_active_modes` counts modes still represented by another positive
residual immediately after the first credited block. `credited_slots` is not a
target and no minimum K is enforced.

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
