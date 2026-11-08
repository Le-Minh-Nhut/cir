# Reproducibility verdict — ENCODER and PTHA+MTST cannot be re-run from HEAD

Two composition adapters called throughput helpers they never imported. Both raised `NameError`
inside `_worker_main`, which each worker's own `except Exception` converts into
`reply(False, error_type="NameError", ...)` — so the run failed at model load, reported as a model
problem rather than a code defect. **ENCODER and PTHA+MTST could not produce a fresh run from this
tree at all.**

---

## 1. The defect

| adapter | call site | helper | imported? |
|---|---|---|---|
| `model_encoder.py` | `:330` | `limit_torch_threads()` | **no** |
| `model_encoder.py` | `:331` | `resolved_decode_workers()` | **no** |
| `model_encoder.py` | `:353` | `prefetch_rows(decode, batches, workers)` | **no** |
| `model_ptha_mtst.py` | `:320` | `limit_torch_threads()` | **no** |
| `model_ptha_mtst.py` | `:321` | `resolved_decode_workers()` | **no** |
| `model_ptha_mtst.py` | `:345` | `prefetch_rows(decode, batches, workers)` | **no** |

Every line number above was re-verified in the file before editing, not taken from a report.

Cause: the parallel-decode change added `prefetch_rows`/`limit_torch_threads`/
`resolved_decode_workers` to several adapters. Five of them (`habit`, `intent`, `hint`, `conesep`,
`tme`) received calls **and** imports; `limn` likewise (multi-line import at `:45-47`). `encoder` and
`ptha_mtst` received only the calls.

## 2. Static audit — all nine adapters, after the fix

Helper set derived from `workbench/composition/adapters.py`'s public names (15 names, including
`limit_torch_threads`, `prefetch_rows`, `resolved_decode_workers`, `resolved_batch_size`,
`worker_environment`, `require_batch_fits`), then checked against every `model_*.py`.

| adapter | helpers called | verdict |
|---|---|---|
| `model_conesep` | AdapterWorkRootError, AdapterWorkerError, limit_torch_threads, prefetch_rows, resolved_decode_workers, worker_environment | **OK** |
| `model_csmcir` | AdapterWorkRootError, AdapterWorkerError, worker_environment | **OK** (no decode pool; not runnable for other reasons already recorded) |
| `model_encoder` | AdapterStatus, AdapterWorkerError, limit_torch_threads, prefetch_rows, resolved_decode_workers, worker_environment | **OK** (was MISSING ×3) |
| `model_habit` | AdapterStatus, AdapterWorkRootError, AdapterWorkerError, limit_torch_threads, prefetch_rows, require_batch_fits, resolved_decode_workers, worker_environment | **OK** |
| `model_hint` | AdapterStatus, AdapterWorkRootError, AdapterWorkerError, limit_torch_threads, prefetch_rows, resolved_decode_workers, worker_environment | **OK** |
| `model_intent` | AdapterStatus, AdapterWorkRootError, AdapterWorkerError, limit_torch_threads, prefetch_rows, resolved_decode_workers, worker_environment | **OK** |
| `model_limn` | AdapterStatus, AdapterWorkerError, ExactRankUnavailable, limit_torch_threads, prefetch_rows, resolved_batch_size, resolved_decode_workers, worker_environment | **OK** |
| `model_ptha_mtst` | AdapterWorkRootError, AdapterWorkerError, limit_torch_threads, prefetch_rows, resolved_decode_workers, worker_environment | **OK** (was MISSING ×3) |
| `model_tme` | AdapterStatus, AdapterWorkRootError, AdapterWorkerError, limit_torch_threads, prefetch_rows, resolved_decode_workers, worker_environment | **OK** |

**Exactly two files were broken. All nine are now clean.** No other adapter had this defect.

Confirmed by running the helpers in the real model interpreters (not a stub):

```
workbench/envs/encoder310-adapted/bin/python
  limit_torch_threads() -> 10
  resolved_decode_workers() -> 10
  prefetch_rows -> True
workbench/envs/limn310-adapted/bin/python  -> same
workbench/envs/tme-adapted/bin/python      -> same
```

and by resolving the names in `_worker_main.__globals__` for both fixed adapters — all six
`(module, helper)` pairs report `in globals=True callable=True`.

## 3. Reproducibility verdict per model

The distinction that matters: a published number was produced by a *particular* `run_id`, and that id
was minted by a *particular* module source hash. If the module changed, the tree no longer
reproduces that id.

| model | published MM R@10 / R@50 | artifact run_id | runnable from HEAD before this fix? | status |
|---|---|---|---|---|
| **ENCODER** | 31.02 / 49.58 | `6a528e877c328772e2c7bb28`, `eda30a37c0bc17ac43844f09` | **NO** — `NameError` at `:330` | **pre-generated artifact, not runnable from HEAD** |
| **PTHA+MTST** | 19.47 / 34.45 | `315dd590eeb03d2c78d7937a` | **NO** — `NameError` at `:320` | **pre-generated artifact, not runnable from HEAD** |
| LIMN | 30.06 / 48.76 | `10173dc45dea8c90fd2a284b`, `3f86f58483b368377e2c151e` | yes — imports all three at `:45-47`, calls all three | runs; id moves for the F1/F6 reasons below |
| TME | 34.02 / 53.96 | `ad477312aa57c1b9114f9e51` | yes — imports all three at `:15` | runs |
| HABIT | — | `497c25c57fbeba885ecfb9d1`, `e0dd04a46ef7a58334b9e353` | yes | runs |
| INTENT | — | (live cohort, in flight) | yes | runs |
| HINT, CONSEP | — | — | yes | runs |
| CSMCIR | — | — | not runnable for an independent, already-recorded reason (`mode_supported` False for every condition) | not applicable |

**So ENCODER and PTHA+MTST rows must be labelled `pre-generated artifact, not runnable from HEAD`.**
Their numbers are not *wrong* — the code that produced them was correct — but the current tree could
not have produced them, and must not be presented as if a fresh run would regenerate them. That
label stays until the import fix is present **and** the model is re-run.

With the imports restored, both become runnable again. Their `run_id` will change (see below), so a
re-run is a *new* artifact under a new id, not a re-derivation of the stamped one.

## 4. run_id consequence

`runner._module_source_hash` hashes the adapter's own module file into
`identity.adapter_module_source_hash` (`runner.py:802-808`), so editing an adapter re-keys its runs.

```
model_encoder.py       sha256 now bf60d855be3d72dd…  HEAD 1a8b0dd25827ca0b…  differ
model_ptha_mtst.py     sha256 now 768bc3ae9a2bd0af…  HEAD 546d1a52d2b6ca58…  differ
```

For **ENCODER** this is the *third* change to the same file in this pass: F4 (mapping status), then
this import fix. They land as **one** new `run_id`, not two — the hash is computed from the file as it
stands at run time, so only the final content matters. Intermediate states never minted an id because
no run was made from them.

For **PTHA+MTST** this is a **separate, independent** `run_id` change, because it is a different
module and therefore a different `identity`.

Neither change alters a number: the imports bind exactly the functions that were intended to be
called, and no computation is modified.

## 5. Why no existing test caught this

`test_circus_adapters_blip.py` and `test_circus_adapters_rest.py` import the module and construct
`Adapter()` — which never enters `_worker_main`. A `NameError` in a function body is invisible until
that line executes, so a smoke import passes on a file that cannot encode a single image. The check
must be static; `workbench/tests/test_audit_fix_encoder_helpers_imported.py` is that check, and it is
proven by reverting each import:

```
# encoder imports removed        -> 3 FAILED (all naming model_encoder)
# ptha_mtst imports removed      -> 4 FAILED (both assertion layers + the explicit pin)
# both restored                  -> 23 passed
```
