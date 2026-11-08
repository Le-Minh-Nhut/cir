# Pre-F1 artifact inventory and cache-reuse verdict

Answers two questions: which existing run directories become pre-F1 artifacts, and whether their
feature caches can still be reused. Read `runner.py` and the live artifact tree on
`/home/heheboiz/data/cir/workbench/artifacts`, then confirmed empirically against a real 2.3 GB cache.

---

## 1. Why every run id changes

`run_id = _digest(identity)[:24]` and `fingerprint = _digest(identity)`. Before this pass, `identity`
contained `"seed": seed`. Removing that key changes the digest input for **every** run, including
`--seed 0`. Demonstrated with the two exact payloads:

```
identity pre-F1  = {"model_id": "habit", "batch_size": 32, "seed": 0}
identity post-F1 = {"model_id": "habit", "batch_size": 32}

pre-F1  run_id: 213a57e35a3359ac19c73660   post-F1 run_id: 7f952d4ebff4269e221cd207
pre-F1  fingerprint: 213a57e35a3359ac19c7366051d1000b...
post-F1 fingerprint: 7f952d4ebff4269e221cd207907ce804...
```

So there is no `--seed` value that preserves an old id. The default run is pre-F1 too.

**A second, independent reason for ENCODER specifically:** F4 edited `model_encoder.py`, and
`runner._module_source_hash` hashes the adapter's own module file
(`runner.py:802-808`) into `identity.adapter_module_source_hash`. Measured:

```
model_encoder.py sha256 now  : 3f4efc789448106478f8c73386b0669be4d85167bcde033e2c244da556d70687
model_encoder.py sha256 HEAD : 1a8b0dd25827ca0ba90f504d81f2fa190cede1ee3fb3bfe335e70ab359005823
```

So ENCODER re-keys from the F4 edit as well as from F1. Conversely, F6 (`limn_observer.py`) is **not**
in `identity` — `grep limn_observer runner.py adapters.py` is empty — so LIMN's id moves only because
of F1, not because of the comment fix.

---

## 2. Pre-F1 run directories on disk

All under `workbench/artifacts/composition/`. Sizes from `du -sh`.

| run_id | model / checkpoint | summary dir | results attempt dir | feature cache | size |
|---|---|---|---|---|---|
| `10173dc45dea8c90fd2a284b` | limn / base_iter0_all_categories | ✓ | ✓ | — | 440K |
| `315dd590eeb03d2c78d7937a` | ptha_mtst / cirr_transfer | ✓ | ✓ | — | 392K |
| `3f86f58483b368377e2c151e` | limn / base_iter0_all_categories | ✓ | ✓ | — | 20K |
| `497c25c57fbeba885ecfb9d1` | habit / fiq_n02 | ✓ | ✓ | **yes** | **2.4G** |
| `6a528e877c328772e2c7bb28` | encoder / fashioniq | ✓ | ✓ | — | 388K |
| `ad477312aa57c1b9114f9e51` | tme / fashioniq_0 | ✓ | ✓ | — | 384K |
| `e0dd04a46ef7a58334b9e353` | habit / fiq_n02 | ✓ | ✓ | — | 4.4M |
| `eda30a37c0bc17ac43844f09` | encoder / fashioniq | ✓ | ✓ | — | 20K |
| `37ee72aa298b276d1bfda0fa` | **intent** (no ranks, no summary) | ✗ | ✗ | **yes** | **2.4G** |

Not run-id keyed, so **not** orphaned by F1:

- `runs/SYNTHETIC_encoder_fashioniq_fixture001`, `runs/SYNTHETIC_habit_fiq_n02_fixture002` — test
  fixtures, named by construction, not by digest.
- `runs/released_audit_task7_mm_any` — the released-audit import path (`_component` of a literal id).
- `runs/37ee72aa…` is the **live cohort's in-flight `intent` run**: it has a gallery feature cache but
  no `ranks/` and no summary yet, which is exactly the shape of a run in progress. Do not delete it
  while pid 264541 is alive.

**How to identify pre-F1 artifacts without a list:** every pre-F1 summary JSON has
`"seed_applied": null` (the key did not exist); every post-F1 summary has `"seed_applied": false`.
Verified on `summary/encoder_fashioniq_6a528e877c328772e2c7bb28/full.json` →
`seed: 0, seed_applied: None`. That is an intrinsic, self-describing marker — labelling does not
require renaming anything.

---

## 3. Cache-reuse verdict: **cannot be reused. Re-encode.**

The cache is keyed **twice**, and both keys derive from `identity`:

1. **Path**: `work_root = composition/runs/<run_id>/work/<model_id>`, then
   `features/gallery_<scope>.npy` (`runner.py:1057-1059`, `:1155-1159`). A new `run_id` is a
   different directory, so the old files are never opened.
2. **Sidecar fingerprint**: even if the files were copied into the new directory, ownership is checked
   against `_digest({"run": fingerprint, "scope": ..., "kind": ...})` for the gallery
   (`runner.py:1160-1163`) and `_digest({"run": fingerprint, "condition": ..., "kind":
   "query-features"})` for queries (`:1199-1200`). `_load_memmap` refuses on
   `metadata["fingerprint"] != fingerprint`.

Measured against the real 2.3 GB habit gallery cache:

```
real cache sidecar fingerprint : c8aa7f0fa359b3574859293cab2b61ed...
recorded shape                 : (74381, 32, 256)

real cache loaded under its own (pre-F1) fingerprint : True
real cache loaded under a post-F1 fingerprint        : False
```

So both mechanisms independently refuse it. There is no copy, symlink, or `--resume` path that lets a
post-F1 run adopt a pre-F1 cache: `--resume` first looks for `runs/<new_id>/ranks/<model>_<ckpt>.json`,
which does not exist, and then re-encodes.

**Consequence:** labelling alone is *sufficient for provenance* but *insufficient to avoid compute*.
Any model whose numbers must be re-derived must have its gallery re-encoded under the new run id.
Concretely:

- The **F3 fix does not force a re-run** — it changes cache *trust*, not the fingerprint. Its effect is
  only that a mis-sized cache is now refused instead of silently used.
- The **F4 fix does not change any number** — the registry value is `VERIFIED_METADATA`, the loaded
  weights are byte-identical, and only the ENCODER `run_id` moves.
- The **F1 fix does force a re-key**, so the cohort must be re-run *if a single consistent run-id
  namespace is required across all published models*. If the existing per-model artifacts are simply
  labelled pre-F1 (via `seed_applied: null`), their numbers remain valid — the features and ranks are
  unchanged, because removing `seed` from the identity does not alter any computation.

That last point is the operationally important one: **F1 is a provenance/namespace change, not a
numerical change.** A re-run reproduces the same metrics under new ids; it is not needed for
correctness of the numbers, only for a clean, uniformly-keyed artifact tree.

> ## Headline verdict
>
> **No published number is invalidated by F1.** Every number in every existing summary was produced
> by a run whose features, scores and ranks are untouched by this change: `seed` was never read by any
> computation, so removing it from `identity` cannot alter a result. What F1 invalidates is the
> *addressability* of those artifacts — their run ids no longer recompute — not their correctness.
>
> Re-encoding is therefore an **optional** operation, required only to obtain a uniformly-keyed tree
> where `run --resume` and cache reuse work across all models. It is **not** required to stand behind
> any reported figure.
>
> The one exception to "no published number is invalidated" is orthogonal to F1 and is recorded in
> `MODEL_REPRODUCIBILITY_VERDICT.md`: ENCODER and PTHA+MTST had a `NameError` on their encode path, so
> those two rows are `pre-generated artifact, not runnable from HEAD`. That is a runnability problem,
> not a numeric one — their artifacts remain valid, the tree simply could not regenerate them.

### What to do with the 2.4 GB caches

- `runs/497c25c57fbeba885ecfb9d1/work/habit/features` (2.4G, habit) — pre-F1, unreachable by any
  post-F1 run. Delete when habit's numbers are final, or keep as the frozen artifact for the
  pre-F1 id it belongs to. It cannot be reused either way.
- `runs/37ee72aa298b276d1bfda0fa/work/intent/features` (2.4G, intent) — **in-flight**; leave alone
  until the cohort evaluator (pid 264541) exits, then it is subject to the same verdict.

---

## 4. Verification after applying both patches

Subset: `-k "composition or capability or topk or seed or cache or circus"`, `-n 4`.

| Run | Result |
|---|---|
| Baseline (before all edits) | 7 failed, 321 passed, 7 skipped |
| After F1 + F3 | 7 failed, 328 passed, 7 skipped |
| After F1 + F3 + F4 + F6 | see `final_subset_after_patches.txt` |

`test_audit_fix_f4_encoder_mapping_status.py` adds 5 checks. The three that matter assert the
**downgrade** path, because a happy-path-only test passes with the literal still in the source:

- `test_downgraded_registry_status_is_obeyed` — worker built from a job carrying `UNVERIFIED` must
  refuse with "does not match the model".
- `test_absent_registry_status_is_obeyed_and_fails_closed` — a missing key must not default to
  `VERIFIED_METADATA`.
- `test_adapter_reads_status_from_the_registry_record` — the literal is absent from the source and
  both read sites are present.

Proof these are not vacuous — restoring only the worker's literal makes them fail:

```
FAILED ...::test_downgraded_registry_status_is_obeyed
       AssertionError: assert 'does not match the model' in 'REACHED_PAST_MAPPING_GATE'
FAILED ...::test_absent_registry_status_is_obeyed_and_fails_closed
       AssertionError: assert 'does not match the model' in 'REACHED_PAST_MAPPING_GATE'
FAILED ...::test_adapter_reads_status_from_the_registry_record
       AssertionError: the worker must read the status, not assert it
```

`REACHED_PAST_MAPPING_GATE` is the sentinel the next call raises on, so with the literal restored the
worker **ignored `UNVERIFIED` and accepted the load** — precisely the defect. With the patch: 5 passed.
