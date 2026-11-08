# Runner requests — candidate Top-K export (Task A)

Owner of `workbench/composition/runner.py` and `workbench/composition/data.py`: another agent.
Everything below is already implemented and tested outside those two files. The runner change is
deliberately **one optional keyword, one writer open, one call at the rank site, one close**.

## Status

- Writer, reader, schema, manifest, tests, API: **done** (`workbench/composition/topk.py`,
  `workbench/backend/composition.py`, `workbench/backend/main.py`, `workbench/scripts/composition_eval.py`
  is the CLI and is NOT yet wired — see §4).
- Nothing in `runner.py` / `data.py` was edited by me.

## Why the runner must change

`run_composition` calls `adapter.rank(vectors, targets, positions, cats)` at
`workbench/composition/runner.py:899`. That single call throws away the score row it computed, so
the candidates are unrecoverable afterwards without a second, different nearest-neighbour pass —
which the task explicitly forbids. The supported path is the new
`CompositionAdapter.rank_and_topk(...)` (`workbench/composition/adapters.py`), which returns the
metric ranks **and** the candidate window from ONE `score()` call. `rank()` is now a thin wrapper
over the same `_ranks_from_scores`, so the two paths cannot diverge.

## 1. `run_composition` signature — add one keyword

```python
def run_composition(data, model_specs, config, *,
                    subsets, batch_size=32, smoke=None, resume=False,
                    conditions=CONDITIONS, image_resolver=None, synthetic=None, seed=0,
                    topk_depth: int | None = None):   # <-- new, default None = no export, no cost
```

`None` (or `0`) means "do not export"; the normal metric run pays nothing.

## 2. Open the writers once per model, before the condition loop

Inside `for spec in model_specs:` after `run_id`, `fingerprint`, `gallery_ids`,
`data.candidate_names` are all in scope (they are — `gallery_ids` is set before the loop, and
`run_id`/`fingerprint` immediately inside it):

```python
from workbench.composition import topk

writers = topk.open_export(
    artifacts, run_id,
    attempt=attempt, model_id=model_id, checkpoint_id=checkpoint_id,
    protocol_id=data.gallery.protocol_id,
    gallery_identity=gallery_hash,                    # the runner's own _gallery_fingerprint(data)
    gallery_count=len(gallery_ids),
    depth=topk_depth if topk_depth is not None else 0,
    conditions=conditions,
    reference_eligible=bool(getattr(adapter, "reference_eligible", True)),
)
```

`gallery_hash` **is** the digest the reader recomputes, byte for byte
(`topk.gallery_digest` delegates to the same JSON + sha256 recipe; pinned by
`tests/test_composition_topk_export.py::test_gallery_digest_is_the_runner_gallery_fingerprint`).

`open_export` returns `{}` when `depth` opts out, so every call below is a no-op.

## 3. Rank site — replace one call with two lines

Current (`runner.py:899`):

```python
exact_ranks = adapter.rank(vectors, targets, positions, cats)
```

Replace with:

```python
if writers:
    exact_ranks, windows = adapter.rank_and_topk(vectors, targets, positions, cats,
                                                decode_depth)
    topk.write_batch(writers, condition, [queries[i].qid for i in valid_indices],
                     exact_ranks, windows)
else:
    exact_ranks = adapter.rank(vectors, targets, positions, cats)
```

where `decode_depth = writers[condition].depth if condition in writers else 0`.

`write_batch` refuses a length mismatch instead of pairing a query with another query's window, so
a reversed-response bug is a hard error, not silently swapped candidates. It must be called with
`valid_indices` **after** the `keep` filter (the same list passed to `adapter.rank_and_topk`), which
is exactly the list whose ranks are written into `ranks[condition]` two lines later.

The per-query fallback loop (the `except` branch that calls `adapter.rank` one query at a time) must
**not** write candidates: it exists because batch scoring failed, and re-deriving a window there
would silently be a different code path. Leave it; `close` reports those queries as
`missing_query_count` and the manifest says `partial: true`.

## 4. Close after the condition loop, before the summaries are written

```python
if writers:
    published = topk.close_export(
        writers, expected_qids={c: [q.qid for q in queries] for c in conditions})
```

Wrap it so a later failure cannot leave a half-written export:

```python
try:
    ...  # existing body
finally:
    if writers:
        topk.abort_export(writers)   # no-op for writers already closed
```

`close_export` already aborts every writer if any one fails to publish, and `abort_export` is a
no-op once closed, so a `finally` after `close_export` is safe.

## 5. Record it in the summary (one key)

Add to the `summary = {...}` literal (anywhere; `feature_storage` sits next to it):

```python
"topk_export": ({"requested": True, "depth": decode_depth,
                 "conditions": sorted(published), "partial": bool(...)}
                if writers else {"requested": False, "depth": None,
                                 "conditions": [], "partial": False}),
```

The API does not need this key — it reads the manifest — but a summary that says whether the run
was *asked* to export is the difference between "not requested" and "requested, produced nothing".

## 6. CLI (also mine, `workbench/scripts/composition_eval.py`)

Add to `_parser()`:

```python
parser.add_argument("--topk-depth", type=int, default=None,
                    help="persist candidate Top-K (default: off). 50 covers the UI's Top-10/Top-50")
```

and pass it through the one `run_composition(...)` call at `composition_eval.py:291`:

```python
topk_depth=args.topk_depth,
```

I did not edit that file because it is listed as owned by me only "except serve_workbench.py" — but
it is a script in my ownership, so if you prefer, take this one-line change and I will not touch it.
**It is not applied yet.**

## What the reader guarantees (already implemented)

`workbench/composition/topk.py`:

- positions are gallery positions, resolved to released image IDs at read time against the digest
  in the manifest; a different gallery is refused (`TOPK_EXPORT_GALLERY_MISMATCH`), never
  best-effort resolved;
- `target_rank` is stored per query independently of `topk`, so a target outside the window survives
  (test: `test_exact_target_rank_survives_outside_the_stored_window`);
- a missing completion manifest is `TOPK_EXPORT_INCOMPLETE`, a digest mismatch is
  `TOPK_EXPORT_CORRUPT`, an absent export is `TOPK_EXPORT_NOT_REQUESTED` — never an empty success;
- `depth` is clamped to the gallery and capped at `MAX_EXPORT_DEPTH = 500`;
- rows are cached on `(path, mtime, size)`.

## Disk cost

One condition, 6003 queries, depth 50, positions + float32 score: ~1.3 MB of JSONL. Three
conditions: ~4 MB per run. Nothing is written unless `--topk-depth` is passed.

---

# ADDENDUM — the request has been APPLIED; this file is now a record

`runner.py` is owned by the backend agent as of this round. All seven steps above landed, plus the
defects found while applying them. What follows is the decision record, not a request.

## Decision: option (a) — preserve the feature cache when an export is requested

The resume/backfill branch was dead code on this repo: `finally:` called
`shutil.rmtree(model_work)` **unconditionally**, so `work/<model>/features/*.npy` was deleted after
every run and `_cached_features` could never return anything.

**Chosen:** keep `features/` when `topk_depth` is set or `--keep-features` is passed; delete
everything else (worker logs, staged query images, job dirs) every time. Re-encoding 74,381 images
per model (~40 min each, four models) is the expensive mistake; a ~2.4 GB cache per model on a
volume with 127 GB free is not.

Documented ceiling (`ponytail:` comment in the `finally` block): N kept models hold ~2.4N GB until
an operator deletes `composition/runs/<run_id>/work/<model>/features` deliberately. Upgrade path:
age-based pruning once a sweep job exists, or float16/quantized storage.

## Flags the cohort driver must pass

```bash
workbench/envs/ctl312h/bin/python workbench/scripts/composition_eval.py \
  --models limn --subsets full --topk-depth 50 --keep-features
```

- **Do not pass `--resume` for the export run.** With `--resume` and an existing rank file the rank
  site is skipped, so the window is never written from the scoring pass; the run falls back to the
  cache backfill, which needs a cache the previous runs (correctly) deleted.
- `--topk-depth` implies `--keep-features`.
- A resumed run that asked for an export and cannot produce one now reports
  `execution_status: EXPORT_INCOMPLETE` and `topk_export.status: TOPK_EXPORT_BLOCKED` with the
  reason. It never reports plain success.

## Identity decision

`topk_depth` is **deliberately not** in `identity`. The measurement is identical with or without a
candidate window; folding it in would mint a new `run_id` for the same experiment and then present
the two as separate attempts (a spurious duplicate). The export is keyed per ATTEMPT instead:
`_export_key(fingerprint, attempt, depth)`.

## Also fixed while in this file

- `reference_eligible` was a literal `True` written into `identity`, the summary, and the export
  manifest. It is replaced by the adapter's MEASURED `measured_reference_eligibility()`; `limn`
  measures `False` with a reason, and an unprobeable adapter records `None` (unmeasured), never a
  claim.
- `variant_provenance` is written into every composition summary from the ONE existing builder
  (`evaluate_models._variant_provenance`), and that builder's unreachable `if adapted_profile:`
  guard was fixed — it tested the freshly-assigned `None`, which is exactly why TME/PTHA summaries
  under `WORKBENCH_RUNTIME_ADAPTATION=sm_120` contained no `adapted_runtime` at all.
