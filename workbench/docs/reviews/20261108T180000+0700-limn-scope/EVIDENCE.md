# LIMN scope hook wired into the runner — evidence

Date: 2026-11-08. Branch `research/cir-failure-workbench`. **No GPU run.** GPU was idle
(`nvidia-smi --query-compute-apps` empty, no `composition_eval` process) but a corrected LIMN number
needs a full encode+rank pass, which this change does not fabricate.

## 1. What changed in `workbench/composition/runner.py`

### 1.1 `_scope_for_query` now consults the adapter resolver

```python
def _scope_for_query(adapter, query, candidate_names=None):
    scopes = tuple(getattr(adapter, "category_scopes", (None,)))
    if query.category in scopes:          # loader assignment is final, resolver not asked
        return query.category, None
    resolver = getattr(adapter, "query_scope", None)
    if callable(resolver):
        resolved = resolver(query, candidate_names)
        if resolved in scopes:             # only a declared scope is accepted; anything else falls through
            return resolved, None
    if None in scopes:
        return None, None
    ... existing unresolved reason text, unchanged ...
```

Precedence preserved: `query.category in scopes` returns before the resolver is touched. A resolver
answer outside `scopes` (including `None`) is **not trusted** and falls to the existing unresolved
sentinel with the same reason text, so the query stays unscored with an unchanged, stable failure
message.

### 1.2 Threading `candidate_names` — only one production call site

`grep -n "_scope_for_query(" workbench/` → one production call:

* `runner.py:1130` `assignments = [_scope_for_query(adapter, query, data.candidate_names) for query in queries]`

No global was added; `data.candidate_names` is the release's own table, already in scope there.

### 1.3 Resolved scope must reach the adapter, or the hook is cosmetic

Calling the resolver but still handing `query.category` downstream would leave `None` in the encode
and rank calls, which LIMN's `encode_queries` / `score` reject (`missing_category`). Updated:

| Site | Before | After |
|---|---|---|
| `_encode_queries` batch | `categories.append(queries[index].category)` | `categories.append(scope_of(index))` |
| `_encode_queries` single-query retry | `adapter.encode_queries([path],[text],[queries[index].category])` | `...[scope_of(index)]` |
| `_encode_queries` caller | no scope arg | passes `query_scopes` |
| rank group cats (`runner.py:1377`) | `queries[index].category` | `query_scopes[index]` |
| `_backfill_export` cats (`runner.py:191`) | `queries[index].category` | `query_scopes[index]` |

`_encode_queries`'s new `scopes` parameter is optional and trailing, so the positional caller in
`test_circus_adapters_blip.py` stays valid and keeps the previous behaviour.

## 2. Identity / fingerprint: chosen option

**Chose: per-model identity digest**, not `_query_fingerprint` inputs.

Why: `query_hash = _query_fingerprint(queries)` is computed at `runner.py:1057`, *before* the
per-model loop, and the resolved scope map is a **per-adapter** fact (only LIMN declares a
resolver). Folding `query_scopes` into the global `query_hash` would require moving the resolution
out of the loop, and it would change `gallery_identity`-adjacent fields for models that have no
resolver at all. The identity digest keys exactly the model whose routing changed.

```python
"query_scope_resolution": _scope_resolution_identity(queries, query_scopes),
```

with

```python
def _scope_resolution_identity(queries, query_scopes) -> str:
    return _digest(sorted((query.qid, scope) for query, scope in zip(queries, query_scopes)))
```

Keyed by `(qid, scope)` so the same assignment in a different query order is the same experiment.

### Evidence: old vs new `run_id` (real published artifact)

`identity_proof.json`:

| | value |
|---|---|
| published run dir | `10173dc45dea8c90fd2a284b` |
| digest of the identity **stored in that run's ranks file** | `10173dc45dea8c90fd2a284b` (recomputes exactly) |
| stored `fingerprint` == digest of stored identity | `True` |
| new `run_id` with `query_scope_resolution` folded in | `2a4b19f335f5ec281375d09f` |
| `run_id` differs | `True` |
| fingerprint differs | `True` |
| **resume would match the old ranks artifact** | **`False`** |
| identical scope map → identical id | `True` |
| one flipped scope → new id | `True` |

So the old artifact `10173dc45dea8c90fd2a284b` can never be silently reused: `--resume` finds a
different fingerprint, does not take the reuse branch, and re-scores. The old run directory remains
as the historical record of the old behaviour.

## 3. Real scope resolution over all 6003 queries

`scope_resolution.json` — real release + real FashionIQ annotations + the real `model_limn.Adapter`
(no worker, no GPU; `query_scope` only reads annotation files):

| scope | queries |
|---|---|
| `dress` | 2010 |
| `shirt` | 2035 |
| `toptee` | 1956 |
| `None` (unscored) | **2** |
| total | 6003 |

Remaining `None` set: **`['7:3038', '7:3790']`** — exactly the two the prior investigation called
genuinely ambiguous (identical caption pair filed under both `shirt` and `toptee`).

Loader-assigned queries: **5884**, of which **0** changed scope. The resolver did not override the
loader anywhere.

Resolved map: 6001 scoped, 2 unscored. Scope digest `3a1c5a67…` vs loader-category-only digest
`8a5dd517…`.

## 4. Tests

`workbench/tests/test_limn_scope_runner.py` (new, 9 tests):

| test | pins |
|---|---|
| `test_resolver_is_consulted_when_the_loader_left_the_category_unset` | resolver called with `candidate_names` |
| `test_resolver_value_outside_declared_scopes_is_unresolved_not_trusted` | out-of-scope answer → sentinel + stable reason |
| `test_resolver_returning_none_leaves_the_query_unscored` | no silent default |
| `test_loader_assigned_category_wins_and_the_resolver_is_not_asked` | precedence; resolver not even called |
| `test_adapter_without_the_hook_keeps_the_previous_behaviour` | `SimpleNamespace` adapter unaffected |
| `test_resolved_scope_map_changes_the_run_identity` | map moves identity; order-independent; one flip moves it |
| `test_resolution_digest_reaches_the_identity_dict_through_the_helper` | old vs new identity digest differ |
| `test_encode_queries_hands_the_adapter_the_resolved_scope` | resolved scope reaches `encode_queries` |
| `test_encode_queries_without_a_scope_map_still_uses_the_loader_category` | prior behaviour preserved |

Results:

```
workbench/tests/test_limn_scope_runner.py test_limn_scope_routing.py
          test_circus_resume_stable.py            → 31 passed in 2.67s
+ test_circus_cli.py                             → 52 passed in 5.71s
```

## 5. Full suite

`17 failed, 2456 passed, 47 skipped in 2387s` (`-n 4`). Every failure reproduced unchanged on a
clean `git archive HEAD` tree; attribution in `FULL_SUITE.txt`. None touches `runner.py` — they are
the untracked DCNet symlink, a fixture adapter missing `batch_size`, unset `WORKBENCH_*_PYTHON`, and
the release verifier's `RELEASE_BLOCKED` verdict.

## 6. Did LIMN run? No.

The wiring is proven. **The published 30.06% MM R@10 remains the old 5884-denominator number**
(coverage `0.980176578377478`), and the table must keep saying so: its denominator is 5884 scored of
6003, not comparable at coverage 1.0 with the seven other models. A re-run is required to publish a
6001/6003 figure; expected coverage `0.9996668…` and ranks still `UNKNOWN`.

Re-run (single invocation, only when the GPU is confirmed free):

```bash
WORKBENCH_RUNTIME_ADAPTATION=sm_120 workbench/envs/ctl312h/bin/python \
  workbench/scripts/composition_eval.py --models limn \
  --subsets full,sf,validated --topk-depth 50 --keep-features
```

It will land in a **new** `run_id` (`2a4b19f335f5ec281375d09f` for the identical scope map and all
other identity inputs unchanged), leaving `10173dc45dea8c90fd2a284b` untouched.
