# Patch spec — `adapter_query_scope_resolution` counts "does not route" as "could not scope"

Status: **IMPLEMENTED 2026-11-13** (was: spec only). The runner now derives unresolved from the
RETURNED REASON (`runner._scope_resolution_record`): `(None, None)` = a resolved `global` scope,
`(None, "<reason>")` = genuinely unscopable. Every existing key is kept; `category_agnostic` and
`global_scoped` are added, and `adapter_routed` keeps its NARROW meaning (a query the adapter placed
on a NAMED matrix), so TG-CIR reports `adapter_routed: 0, global_scoped: 6003, still_unscoped: 0,
category_agnostic: true`, with `loader_scoped: 5884` as the separate DESCRIPTIVE count (the loader
assigned a garment to 5 884 queries; this adapter ignores it when picking a matrix, so all 6 003 land
on `global` — `loader_scoped` overlaps `global_scoped` rather than partitioning with it). The derived
publisher copies `category_agnostic` from the parent.
`7b1fa850…`'s artifact is NOT rewritten: it is immutable history and keeps its old block.
Acceptance tests: `workbench/tests/test_limn_scope_runner.py`,
`workbench/tests/test_derived_subset_status.py`. The defect description below is the record.

## Defect (reproduced)

Run `tgcir_tgcir_archive_unresolved_7b1fa85007bffdab931f7f19`,
`workbench/artifacts/composition/summary/tgcir_tgcir_archive_unresolved_7b1fa85007bffdab931f7f19/full.json`:

```json
"adapter_query_scope_resolution": {
  "resolver": null,
  "loader_scoped": 5884,
  "adapter_routed": 0,
  "still_unscoped": 6003,
  "unscoped_qids": [ ... 6003 ids ... ]
},
"category_scopes": [null],
"unresolved_category_query_count": 119,
"condition_status": {"multimodal": {"unscored_count": 0, "status": "COMPLETED"},
                     "text_only_black_image": {"unscored_count": 0, ...},
                     "image_only_empty_text": {"unscored_count": 0, ...}}
```

Read literally the block says *6003 queries could not be scoped*, on a run that scored all 6003 in
all three conditions. The two facts are not the same measurement:

* `still_unscoped` counts `_scope_for_query(...)[0] is None`
  (`runner.py:2534`, and see `_scope_for_query` at `runner.py:547-567`). For TG-CIR,
  `adapter.category_scopes == (None,)`, so the *third* branch of the resolver —
  `if None in scopes: return None, None` (`runner.py:561-562`) — fires and returns `None` **with no
  reason**. That is the adapter's declaration "this model is category-agnostic: it has exactly one
  embedding matrix", not a failure. `_scope_name(None) == "global"` (`runner.py:257-258`), so the
  gallery it scores against is a real, single, correct matrix.
* `condition_status.*.unscored_count` counts queries with **no rank**
  (`runner.py:2350-2351`). Those are 0 here, because every query got a rank.

So the block conflates "the adapter does not route by category" with "the query could not be
scoped". It is the only field in the summary that makes a clean run look broken, and it is the
field `workbench/scripts/derive_routing_cohort_gap.py:135-155` replays against to refuse a
derivation.

## Is the honest fix in `backend/composition.py`?

**No.** `backend/composition.py` never computes this block; it only serves the summary's own
recorded dict (`get_composition_run_detail` → `decorate_summary`). Rewriting or suppressing it on
read would:

* make the served payload disagree with the artifact on disk, and
* diverge from the value `derive_routing_cohort_gap.py` replays, which reads the artifact directly.

The number is wrong where it is measured, so the fix belongs in the runner. The backend must not
paper over it.

## Change (runner, `workbench/composition/runner.py:2528-2537`)

Record the three states the resolver actually distinguishes, keeping every existing key so nothing
that reads the block breaks:

```python
"adapter_query_scope_resolution": {
    "resolver": (...unchanged...),
    "loader_scoped": sum(query.category is not None for query in queries),
    # The adapter declares NO per-category routing (`category_scopes == (None,)`): every query is
    # scored against the single `global` matrix, which is a resolved scope, not an unresolved one.
    "category_agnostic": None in scopes,
    "adapter_routed": sum(scope is not None for scope in query_scopes),
    "still_unscoped": len(unresolved),
    "unscoped_qids": [query.qid for query, (_scope, reason) in zip(queries, assignments)
                      if reason],
},
```

where `unresolved` is derived from the **reasons**, not from the scope value:

```python
assignments = [_scope_for_query(adapter, query, data.candidate_names) for query in queries]
query_scopes = [scope for scope, _reason in assignments]
unresolved = [query for query, (_scope, reason) in zip(queries, assignments) if reason]
```

`_scope_for_query` already returns `(None, None)` for the category-agnostic case and
`(None, "<reason>")` for a genuinely unscopable query (`runner.py:561-567`), so the reason is the
authoritative signal and no new sentinel is needed. Consequence for TG-CIR:
`still_unscoped: 0`, `unscoped_qids: []`, `category_agnostic: true`, `adapter_routed: 5884`
(the loader-scoped ones) — and `still_unscoped == sum(condition_status[*].unscored_count)` becomes
true again, which is the invariant the publisher states.

`adapter_routed` semantics change for the agnostic case: it now counts queries the adapter placed
on a matrix, whether the loader or the adapter decided the name. That is what the word says; if the
narrower "decided by `query_scope()`" meaning is preferred, keep `adapter_routed` as
`sum(scope is not None and query.category is None ...)` and add `global_scoped` beside it. Either
way `still_unscoped` must stop counting resolved `global` scopes.

**Which was chosen (2026-11-13): the NARROW meaning plus `global_scoped`.** TG-CIR therefore reports
`adapter_routed: 0` (not 5 884) and `global_scoped: 6003`. The rejection reason for option (a) is not
cosmetic: `derive_routing_cohort_gap.py:145-147` replays `adapter_routed` against
`len(loader_qids) + len(resolved_by_caption)` and refuses the derivation on disagreement, so widening
the key would have broken a consumer that already exists. Measured on the real release (`gallery_count`
74 381, 6 003 queries) after the change: `adapter_routed 0 + global_scoped 6003 + still_unscoped 0 ==
6003`, with `loader_scoped` staying 5 884.

## The invariant that must not break

`workbench/tests/test_derived_subset_status.py:131-145` asserts, for every derived LIMN subset,
`routing["still_unscoped"] == own_unscored` and `len(routing["unscoped_qids"]) == own_unscored`.
LIMN **is** per-category (`category_scopes` is a real set of garment names) and its two unscoped
qids carry real reasons, so the reason-based count returns 2 there — unchanged. The publisher
(`publish_derived_runs.py:240-241,273-281`) writes `still_unscoped: len(subset_unscoped)` from the
membership arithmetic, not from the parent's block, so derived rows are unaffected as well. Add
`category_agnostic` to the derived block too (copied from the parent) so a reader of a derived
TG-CIR row cannot infer routing failure from `adapter_routed: 0`.

## Tests to add

1. `workbench/tests/test_limn_scope_runner.py` (owns `_scope_for_query`): add the category-agnostic
   case explicitly — `SimpleNamespace(category_scopes=(None,))` with a query whose `category` is
   `None` must return `(None, None)`, i.e. a resolved `global` scope with **no** failure reason.
   The existing `test_adapter_without_the_hook_keeps_the_previous_behaviour` (line 71-75) covers
   the *other* branch (declared scopes, no resolver) and must keep returning a reason.
2. A runner-level test (the gap-identity file already builds a fake adapter) that the recorded block
   distinguishes them:
   * category-agnostic adapter → `still_unscoped == 0`, `unscoped_qids == []`,
     `category_agnostic is True`;
   * category-scoped adapter with one query in an unknown category → `still_unscoped == 1`,
     `unscoped_qids == [that qid]`, `category_agnostic is False`.
3. A backend guard so the two numbers cannot drift again before the runner change lands: assert on
   the real artifact that `still_unscoped` is *not* presented as a per-condition unscored count —
   i.e. a test that would have failed on `7b1fa85007bffdab931f7f19` (6003 vs 0).
