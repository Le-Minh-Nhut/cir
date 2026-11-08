# Runner requests — LIMN query-to-checkpoint routing (adapter-owned resolver)

Owner of `workbench/composition/runner.py`: another agent. I own only
`workbench/composition/model_limn.py` and `workbench/tests/test_limn_*.py`, so everything that
touches the adapter is implemented and tested; this file lists the **two** runner edits needed to
let the runner use it.

## Background, in one line

`runner._scope_for_query` decides a query's scope from `query.category` alone. For 117 of the 119
CIRCUS/FashionIQ queries the release loader left unscoped, the garment category is determinable
from the author annotation, and `Adapter.query_scope()` returns it. Evidence and counts:
`workbench/docs/reviews/20261108T114545+0700-limn/LIMN_UNSCORED.vi.md`.

## 1. One call site — prefer the adapter's resolver when it has one

`runner.py:715`

```python
assignments = [_scope_for_query(adapter, query) for query in queries]
```

becomes

```python
assignments = [_scope_for_query(adapter, query) for query in queries]
```

with `_scope_for_query` (currently `runner.py:229`) consulting an optional adapter hook first:

```python
def _scope_for_query(adapter: Any, query: Any) -> tuple[str | None, str | None]:
    scopes = tuple(getattr(adapter, "category_scopes", (None,)))
    resolver = getattr(adapter, "query_scope", None)
    if callable(resolver):
        # The adapter may know a category the release loader could not derive; it returns the
        # loader's own category unchanged when there is one, so this is extend-only.
        resolved = resolver(query)
        if resolved is not None and resolved in scopes:
            return resolved, None
    if query.category in scopes:
        return query.category, None
    ...
```

`Adapter.query_scope(query)` returns `query.category` verbatim when the loader assigned one
(pinned by `test_loader_assigned_category_is_never_reinterpreted`), so the 5,884 already-scored
queries keep their scope byte-for-byte and no other adapter changes behaviour — they simply lack the
attribute and fall through to the current code. `tests/test_circus_resume_stable.py` uses a
`SimpleNamespace` adapter without the attribute, so it keeps passing.

## 2. Run identity must move, or a resume reuses the 119 stale nulls

This is the part that is easy to miss.

`_query_fingerprint` (`runner.py:211-215`) keys on `query.category` — the *loader's* value. After the
routing change the loader's value is still `None` for the 117, so the fingerprint is unchanged, the
existing `runs/<run_id>/ranks/*.json` still matches, and the resume path reuses it: `save` records
`query ... category unresolved has no adapter embedding scope`, which is in `_STABLE_FAILURE_MARKERS`,
so `transient` stays empty and the run is reported `REUSED` with the same 119 nulls and the same
`coverage 0.98018`. The fix would be invisible.

Include the *resolved* scope in the fingerprint, e.g.

```python
def _query_fingerprint(queries: Sequence[Any], scopes: Sequence[str | None] | None = None) -> str:
    return _digest([{"qid": query.qid, "index": query.index, "text": query.query_txt,
                     "reference_id": query.reference_id, "positive_ids": list(query.positive_ids),
                     "category": query.category,
                     "resolved_scope": None if scopes is None else scopes[position],
                     "subsets": list(query.subsets)} for position, query in enumerate(queries)])
```

with the call site using the already-computed `query_scopes` — but note `query_hash` is computed at
`runner.py:669`, i.e. *before* the per-model loop where `query_scopes` exists. Two options, both
fine:

- **preferred**: compute the resolution once per model and fold it into the *identity* dict instead
  (`"query_scope_resolution": _digest(sorted(query_scopes))`) next to `"query_order_identity"`. The
  `run_id` then changes, so a rescore lands in a new run dir and the old artifact stays as the
  historical record rather than being overwritten. This is the honest option: the old run genuinely
  measured the old behaviour.
- acceptable: keep the global `query_hash` and add the per-model `query_scope_resolution` digest to
  `identity`. Same effect, one narrower diff.

Do **not** mutate `data.queries` or `data.provenance` to carry the resolved category. `data.py`
documents (correctly, for its own split-based rule) that "positive/target IDs and caption text never
determine garment category"; the annotation-based rule is LIMN-specific protocol knowledge and
belongs to the adapter that needs it, which is why the hook is on the adapter and not in `data.py`.
The release provenance keeps reporting 5,884/119/0 as the release-level fact.

## 3. Optional, for the reports

`summary["category_resolution"]` comes from `data.provenance`, so it will keep saying
`ambiguous_count: 119` while `scored_count` becomes 6,001. That reads like a contradiction. One added
key disambiguates it:

```python
"adapter_query_scope_resolution": {
    "resolver": "Adapter.query_scope",
    "loader_scoped": sum(query.category is not None for query in queries),
    "adapter_routed": sum(scope is not None for scope in query_scopes),
    "still_unscoped": sum(scope is None for scope in query_scopes),
    "unscoped_qids": [q.qid for q, s in zip(queries, query_scopes) if s is None],
},
```

Expected on the LIMN FashionIQ full cohort: `loader_scoped: 5884`, `adapter_routed: 6001`,
`still_unscoped: 2`, `unscoped_qids: ["7:3038", "7:3790"]`.

## What does not change

`encode_queries` / `encode_targets` / `score` / `rank` are untouched. Same three checkpoints, same
transform, same `dataset_ids` mapping, same tie policy, same full-gallery matrices per scope. Only
the `scope -> [(query, ...)]` grouping changes, i.e. which of three pre-existing gallery matrices a
query row is scored against. That is the only thing the change claims.
