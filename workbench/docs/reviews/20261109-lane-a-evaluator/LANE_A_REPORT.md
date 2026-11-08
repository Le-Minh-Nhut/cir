# Lane A — evaluator correctness owner: six defect fixes

Revision worked on: `ffb6ae8a072f4f92ddd203bbebac8fa2cb4e3702` (worktree left dirty, uncommitted,
as instructed).

Every test below was run with
`workbench/envs/ctl312h/bin/python -m pytest … -q -p no:cacheprovider`. CPU only; no GPU job, no
real evaluation, no full suite.

**Headline: `workbench/tests/test_review_defects_evaluator.py` is 19 failed / 2 passed on the
pre-fix source and 21 passed after it.** That is the fails-before/passes-after evidence for the
whole set, produced by stashing only my two production files.

---

## D1 — backfill / batch-fallback shared scoring policy

**Verdict: FIXED.**

Cause: `_blas_thread_limit` was applied only inside `_rank_groups_concurrently`. `_backfill_export`
called `adapter.rank_and_topk` directly, and the per-batch failure path called `adapter.rank`
directly *after* that helper had already restored the previous BLAS count — so both scored at the
process default while the original pass scored at `BLAS_THREADS_PINNED`. Because the thread count
changes the floating-point summation order, the same vectors could yield a different rank.

Changed: `workbench/composition/runner.py` — `_backfill_export` (pin around the rank call, restore
in `finally`), the per-batch fallback loop (same), and a new `BackfillRankDisagreement`.

Backfill also now **rejects** a window whose target rank disagrees with the recorded run: the caller
passes `original_ranks=ranks`, and any mismatch raises rather than writing, so the caller reports
`TOPK_EXPORT_BLOCKED` with the reason instead of publishing a window from another arithmetic.

Tests: `test_backfill_runs_under_the_same_blas_policy_as_the_normal_path` (fails before: "backfill
never called the BLAS limiter"), plus `test_gallery_identity_assertion_is_not_vacuous` for the
reconciliation helper contract.

**Remaining risk:** the probe in the handoff used a synthetic policy-sensitive scorer; this fix is
verified on the same class of fixture. No real-model rerun was performed (out of scope for this
lane), so "the four GEMM models now agree" is **not** claimed — only that the control flow shares
one policy and refuses to publish on disagreement.

## D2 — ineffective BLAS pin reported as pinned

**Verdict: FIXED.**

Cause: `if setter is None: return int(previous)` returned normally, and `previous` came from
`OPENBLAS_NUM_THREADS`/`OMP_NUM_THREADS` rather than from the runtime. Callers then wrote
`blas_threads_pinned = 1` as fact.

Changed: `runner.py` — new `_openblas_symbols()` (probes `*64_` then 32-bit, never loads a foreign
env's DSO), `_blas_thread_limit` now **reads back** and returns a record
(`requested`, `effective`, `previous`, `setter_found`, `getter_found`, `verified`, `source`,
`reason`), `_current_openblas_threads` returns the runtime value or `None`, and a new
`_blas_pin_capability()` gives the stable, probeable capability recorded in the run identity.

Measured on this host's real interpreter, not a fixture:

```
before:                20 threads
_blas_thread_limit(1): requested 1, effective 1, previous 20, setter_found True, verified True
                        source libopenblas64_p-r0-0cf96a72.3.23.dev.so
restore:               effective 20  ->  runtime back at 20
```

So the setter does work here, and the fix records that honestly instead of assuming it.

Tests: `test_blas_pin_reports_verified_or_failed_never_a_silent_success`,
`test_blas_pin_verified_when_the_runtime_accepts_it`,
`test_blas_pin_is_unverified_when_readback_disagrees`,
`test_blas_pin_backwards_compatible_int_coercion`.

**Remaining risk:** a runtime exposing a setter but no getter cannot be verified. That is recorded as
`verified: False` with the reason, which is the honest outcome, not a pass.

## D5 — measurement identity grouping

**Verdict: FIXED**, with the design decision stated explicitly.

`workbench/backend/composition.py` — `measurement_identity` now carries
`blas_threads_pinned`, `query_scope_resolution`, `query_order_identity` and `cache_verify_mode` in
the hashed payload, so two runs differing only in one of those mint different `group_id`s. Each gets
`identity_schema_version: 2`.

The consequences the brief asked me to make explicit:

* **`group_id` changes for every future run.** Intended, and why `identity_schema_version` is in the
  payload: a v1 group and a v2 group can never collide or be compared.
* **Historical artifacts are never regrouped.** A recorded run identity that predates v2 (recognised
  by carrying runner-only keys but none of the three v2 markers) gets `group_id: None`,
  `confidence: "unknown"`, `identity_schema_version: "unknown (predates identity v2)"` and a reason
  naming the missing fields. `annotate_duplicate_attempts` labels such a row
  `duplicate_status: "unknown (predates identity v2)"` and excludes it from grouping entirely.
* **No artifact on disk is rewritten.** The change is read-only with respect to `artifacts/`; tests
  only write into `tmp_path`.

One correction I made deliberately: the v2 inputs are **not** in the `essential` tuple. Absence would
otherwise downgrade every served summary (which legitimately carries only metadata) to `partial` and
break the existing contract that a complete metadata row is `unique`. Their job is to *split groups
when present*, which being in the payload achieves.

Tests: `test_identity_v2_splits_the_group_when_the_blas_policy_changes`,
`test_identity_v2_splits_on_scope_resolution_and_query_order`,
`test_identity_v2_stamps_the_schema_version`, `test_identity_v1_artifacts_are_unknown_not_regrouped`,
`test_identity_v2_fields_split_groups_when_present`,
`test_duplicate_labelling_never_calls_two_policies_duplicates`.

## D6 — historical provenance for `scoring_impl`

**Verdict: FIXED.**

Cause: `_scoring_impl_for` read the *current* adapter module's `score` body and that label was
attached to a historical artifact, so refactoring an adapter today would relabel old measurements.

Changed: `scoring_implementation(..., summary=, live=)` — a value the artifact itself recorded always
wins (`source: "artifact"`); derivation from source happens only when the caller can prove the
recorded source hash equals the module's hash on disk today (new `_recorded_source_matches_disk`);
otherwise `impl: "unknown"` with the reason. The serving call site
(`workbench/backend/composition.py`, the `summary["scoring_impl"] = …` assignment) passes the summary
and the hash-match verdict.

Tests: `test_scoring_impl_comes_from_the_artifact_not_todays_source`,
`test_scoring_impl_unknown_when_artifact_records_nothing_derivable`,
`test_scoring_impl_still_derives_for_a_live_run`.

**Remaining risk:** no hash→implementation table exists in this repository, so a historical run with
no recorded value is `unknown`. That is the intended answer, not a limitation to work around.

## D7 — cache content integrity

**Verdict: FIXED**, with measured costs.

`runner.py` — `_write_memmap_metadata` records a chunked `content_digest`, `byte_length` and
`verify_mode`; `_load_memmap` verifies on load; new `_memmap_integrity` reports what the sidecar can
prove. Backward compatibility is deliberate: no digest → the cache still loads but is reported
`content_digest: "absent"`, `verified: False`, never described as validated.

Caught now: changed content at identical shape, truncated file, stale sidecar shape (the loader also
checks the *sidecar's* shape against the file, not only the caller's expectation — that was a real
gap I found while testing).

Mode knob as requested: `WB_CACHE_VERIFY=full|header|off`, default **`full`**, recorded in the cache
sidecar and in the run identity (`cache_verify_mode`).

**Measured cost on this machine** (chunked at 8192 rows, digest computed from the in-memory array
while writing — no second pass, no second copy):

| array | on disk | bare mmap open | digest | verified load (`full`) | delta |
|---|---:|---:|---:|---:|---:|
| **74,381 × 512 (the real gallery)** | **152 MB** | 0.2 ms | **43.6 ms** (3.5 GB/s) | **43.5 ms** | **+43.3 ms** |
| 74,381 × 128 | 38 MB | 0.2 ms | 10.9 ms (3.5 GB/s) | 11.0 ms | +10.8 ms |
| 10,000 × 512 | 20 MB | 0.2 ms | 7.3 ms (2.8 GB/s) | 5.9 ms | +5.7 ms |

`header` mode costs +0.0 ms (0.19 ms); `off` costs 0.09 ms. So `full` adds **43 ms once per cache
load** against a gallery encode that takes minutes — it is not a 2-second-to-2-minute change. The
metadata write including the digest is 43.1 ms on the real gallery.

Tests: `test_cache_rejects_same_shape_different_content`,
`test_cache_reports_digest_absent_for_legacy_files_without_one`,
`test_cache_rejects_truncated_file`, `test_cache_rejects_stale_sidecar_shape`,
`test_cache_digest_is_recorded_on_write`, `test_cache_digest_covers_the_bytes_it_claims`.

## D8 — the `assert … or True` gallery check

**Verdict: FIXED.**

`test_composition_topk_runner.py:131` and `test_capability_composition.py:81` both replaced. The
former now asserts the export's gallery digest is non-empty, that it is **not** the run fingerprint
(different quantities — that conflation was the original defect), and that it equals the digest the
rank artifact recorded. The latter now asserts the real contract: the reason names the missing export
request and does **not** blame ranks that are present.

Tests: `test_gallery_identity_assertion_is_not_vacuous` plus
`test_ranks_only_run_reports_ranks_without_candidates`.

No new xfail/skip/tolerance inflation was added anywhere.

---

## Blast radius / other lanes

I touched **two files outside my declared ownership list** and am flagging both rather than hiding
them:

* `workbench/tests/test_capability_composition.py` — explicitly named by the brief in D8
  (`:81`), so authorised.
* `workbench/tests/test_audit_fix_f3_gallery_cache_shape.py` — **not** in my list. Its `_write_cache`
  helper recorded a digest built from a fresh `np.zeros` while the file held `arange`, so the new
  content check correctly rejected it. I changed the helper to digest the bytes it actually wrote.
  Without this the file fails, but the call is yours: revert it and I will adjust my side instead.

I did **not** touch `workbench/frontend/**`, `workbench/backend/adapters/models.py`,
`workbench/registry/models.yaml`, `workbench/config/workbench.env`, or any other lane's files.

## Defects seen and NOT fixed (reported, not touched)

* `test_circus_cli.py::test_cli_3d_full_gallery_resume_and_gallery_invalidation` fails
  `assert 'REUSED' == 'COMPLETED'` only when `test_composition_topk_runner.py` runs first. **Proven
  pre-existing**: it reproduces with my changes stashed. This is defect #5 from the old
  `20261108T212500+0700-sweep/FAILURES.md` (ordering interaction introduced by `9b6b5b6`), and it is
  a test-isolation bug in `test_circus_cli.py`, outside this lane.

## Commands and exit codes

| command | exit | result |
|---|---:|---|
| `pytest workbench/tests/test_review_defects_evaluator.py` (pre-fix, production stashed) | 1 | **19 failed, 2 passed** |
| same, post-fix | 0 | **21 passed** |
| `pytest test_composition_*.py test_capability_*.py test_throughput_knobs.py test_review_defects_evaluator.py test_audit_fix_f3_gallery_cache_shape.py test_circus_cli.py` | 1 | **194 passed, 1 failed, 1 skipped** — the single failure is the pre-existing ordering defect above |
| `pytest test_throughput_knobs.py` | 0 | 27 passed |
| `pytest test_circus_cli.py` (alone) | 0 | 21 passed |
| `pytest test_composition_topk_runner.py test_circus_cli.py` | 1 | 1 failed (pre-existing ordering), 30 passed |
| same, reversed order | 0 | 31 passed |

## `git diff --stat` of my changes only

```
 workbench/backend/composition.py                   | 191 ++++++++-
 workbench/composition/runner.py                    | 465 ++++++++++++++++++---
 workbench/tests/test_audit_fix_f3_gallery_cache_shape.py |   5 +-
 workbench/tests/test_capability_composition.py     |   9 +-
 workbench/tests/test_composition_topk_runner.py    |  15 +-
 workbench/tests/test_review_defects_evaluator.py   | NEW (21 tests)
```

Left uncommitted for the lead to integrate.
