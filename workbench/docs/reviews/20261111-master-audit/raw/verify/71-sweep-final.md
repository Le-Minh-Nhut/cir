# 71 — Authoritative serial sweep after the R16a ingest (`358c3fc`)

Production changed after the last pin (`c755673`, measured `18 failed, 2914 passed, 47 skipped` /
2979 collected): `358c3fc` modified `workbench/backend/native_measurements.py` (+261/-10) to ingest
seven committed native GPU run artifacts. This sweep re-measures the suite at the new revision.

## Phạm vi (scope) — what this sweep covers, and what changed under it

| path class | covered by this sweep? | changed mid-run? |
|---|---|---|
| `workbench/backend/**` (incl. `native_measurements.py`) | **yes** — 2,996 pytest cases, serial | no (`git diff 358c3fc HEAD -- workbench/backend` empty) |
| `workbench/composition/**`, `workbench/replay/**` | **yes** | no |
| `workbench/tests/**` | **yes** | no (`git diff 358c3fc HEAD -- workbench/tests` empty) |
| `workbench/frontend/*.ts(x)` (`scope.ts`, `data.ts`, `EvaluationPage.tsx`, `ModelInventory.tsx`) | **partly** — only the pytest tests that read those files as text or compile `scope.ts` (`test_web_scope_and_labels.py`, `test_frontend_catalog.py`, `test_frontend_layout_provenance.py`, `test_frontend_staleness.py`, `test_frontend_requests.py`); **not** covered by `node --test` or `tsc --noEmit` in this sweep | **yes** — `252b380` (14:01:14 +0700) and `d323a5c` (14:33:02 +0700) |
| `workbench/frontend/dist/**` (served bundle) | no (J10 only) | **yes** — rebuilt 14:41:09 +0700, after the suite finished |
| `workbench/docs/**` | no assertions depend on it (only fixed `RELEASE_DOCS` tuple + `workbench/docs/*.md` glob) | **yes** — `727f1d1`, `558caeb` |

**Consequence, stated plainly: this sweep does not govern the final tree.** It measured
backend/composition/replay at `358c3fc`; frontend production changed afterwards (`252b380`, `d323a5c`)
and is covered only by `node --test` + `tsc --noEmit`, not by this sweep.

## Identity

| fact | value |
|---|---|
| HEAD at sweep start | `358c3fc612109d6d268f5afe655f10420a52f318` (= branch tip at start, = preflight `git rev-parse HEAD`) |
| HEAD at sweep end | `d323a5c24b172d1d26848fbd1afd9731250353ed` |
| HEAD moved during the run | **yes** — four commits, 727f1d1 (13:49:45), 558caeb (13:58:11), 252b380 (14:01:14), d323a5c (14:33:02) |
| branch | `research/cir-failure-workbench` |
| tree at start | `?? CLEANUP_REPORT.md.save` only |
| tree at end | `?? CLEANUP_REPORT.md.save` only |
| interpreter | `workbench/envs/ctl312h/bin/python` |
| sweep launched | `2026-11-11 13:44:01 +0700` (log mtime birth) |
| wall time | 3410.80 s (0:56:50) |

### Production delta between sweep start and sweep end

```
$ git diff --stat 358c3fc d323a5c -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'
 workbench/frontend/EvaluationPage.tsx              |  51 ++++++-
 workbench/frontend/ModelInventory.tsx              | 114 ++++++++-------
 workbench/frontend/data.ts                         |  16 +++
 workbench/frontend/scope.ts                        | 160 ++++++++++++++++++++-
 workbench/frontend/tests/inventory-cross-scope.test.cjs  | 99 +++++++++++++
 workbench/frontend/tests/native-conflict-variance.test.cjs | 118 +++++++++++++
```

No backend, composition, replay or test file changed. The release guard's production predicate
excludes only `workbench/docs` and `workbench/tests` (see `DOCS_AND_TESTS_EXCLUDED_PATHS` in
`test_handoff_commands.py`), so these four frontend sources **are** production and this sweep does
not contain them.

### Stationary-tree proof (mid-run check, timestamped)

Verified at **`2026-11-11T06:48:25Z` = `13:48:25 +0700`** (4 min 24 s after launch, sweep at 16 %):

```
$ git rev-parse HEAD
358c3fc612109d6d268f5afe655f10420a52f318          # still the start revision — HEAD had NOT yet moved
$ git status --porcelain
 M workbench/docs/reviews/20261111-master-audit/CENSUS_20261111.vi.md
 M workbench/docs/reviews/20261111-master-audit/FINAL_REPORT.vi.md
 M workbench/docs/reviews/20261111-master-audit/INDEPENDENT_REVIEW_RESPONSE_20261111.vi.md
 M workbench/docs/reviews/20261111-master-audit/REMAINING_WORK.md
?? CLEANUP_REPORT.md.save
$ pgrep -c -f 'pytest workbench/tests'
1
```

**The tree was not stationary.** At 13:48 the working tree already carried uncommitted edits to four
audit documents (`727f1d1` landed at 13:49:45, one minute later). Those files are under
`workbench/docs/reviews/`, which no test reads (verified: no test references those filenames; the only
docs glob is `workbench/docs/*.md`, non-recursive; `RELEASE_DOCS` is a fixed five-file tuple).
They therefore could not have changed any outcome. The **frontend** commits that *can* move outcomes
(`252b380`, `d323a5c`) landed at 14:01 and 14:33, i.e. after the tests that read frontend sources had
already run — `test_frontend_catalog.py` starts at test #1085 (36 %), `test_web_scope_and_labels.py` at
#2956 (98 %). The suite therefore executed against the `358c3fc` frontend, which is why the frontend
rerun below was performed separately.

## Preflight

| check | result |
|---|---|
| `git rev-parse HEAD` | `358c3fc...`, equal to branch tip — PASS |
| `git status --porcelain` | exactly `?? CLEANUP_REPORT.md.save` — PASS |
| `pgrep -af pytest` | empty — PASS |
| `nvidia-smi --query-compute-apps=pid,process_name --format=csv` | header only — PASS (no compute process) |
| `df -h /` | 99 GiB avail ≥ 45 — PASS |
| `/tmp/sweep_basetemp_final3` | absent — PASS |

## Command (verbatim)

```
mkdir -p /tmp/sweep_basetemp_final3
setsid nohup /home/heheboiz/data/cir/workbench/envs/ctl312h/bin/python -m pytest workbench/tests \
  -q -p no:cacheprovider -rfE --tb=short --basetemp=/tmp/sweep_basetemp_final3 \
  > /tmp/sweep_final3.txt 2>&1 < /dev/null &
```

Serial, no `-n`. Log `/tmp/sweep_final3.txt` is not copied into the repo; the summary line and the
greps below are the record.

## Summary line (verbatim, last line of the log)

```
18 failed, 2931 passed, 47 skipped, 350 warnings in 3410.80s (0:56:50)
```

## The four accounting numbers

### 1. Error greps

| measure | value |
|---|---|
| `grep -c '^ERROR '` | **0** — no collection/setup errors |
| `grep -c 'Errno 28'` | **0** — no disk-full |
| `grep -c '^FAILED'` | **18** |

`^E ` deliberately not grepped: with `--tb=short` every traceback body line starts with `E `.

### 2 & 3. Collected vs reported

```
$ workbench/envs/ctl312h/bin/python -m pytest workbench/tests -q --collect-only -p no:cacheprovider
2996 tests collected in 3.52s
```

| N collected | passed | failed | skipped | sum | gap |
|---|---|---|---|---|---|
| 2996 | 2931 | 18 | 47 | 2996 | **0** |

`2996 − 2979 = +17` collected versus the `c755673` sweep: `test_native_ingest_r16.py` (new, 16 cases) +
`test_native_measurements_identity.py` +1 test (+36/-1 lines) — 16 + 1 = 17. Passed moved 2914 → 2931,
exactly +17: every newly collected test passes. **No pre-existing test changed outcome.**

### 4. Failure diff vs baseline `68ad75e`

Baseline `workbench/docs/reviews/20261110-e2e-repair/full_suite_68ad75e.txt` has 18 `FAILED` lines;
`/tmp/sweep_final2.txt` has the same 18 (set-identical to the baseline).

| NEW | DROPPED | COMMON |
|---|---|---|
| **0** | **0** | **18** |

**NEW list: (empty).** **DROPPED list: (empty).**

### 5. Delta vs `/tmp/sweep_final2.txt`

`/tmp/sweep_final2.txt` still exists (85515 bytes, mtime `Nov 11 12:38`) — not deleted.
`comm -3` on the two sorted FAILED node-id sets is **empty**: the same 18 ids, no failure appeared and
none disappeared. That is the expected result: `358c3fc` touched one backend module plus tests, and the
new tests pass. `native_measurements.py` is exercised by 45 census/identity/r16 cases; all pass.

## Native-measurement findings (the numbers the ingest moved)

Live module call at HEAD:

```
$ python -c "from workbench.backend import native_measurements as nm; from pathlib import Path; \
             print(nm.list_native_measurements(Path('workbench/artifacts'))['counts'])"
{'files_scanned': 71, 'files': 39, 'blocks_with_metrics': 50, 'with_metrics': 23,
 'rollup_rows': 18, 'duplicate_rows': 27}
```

Confirmed: `with_metrics` 21 → **23**, `duplicate_rows` 22 → **27**, matching the brief.

**No test asserts the old numbers.** Searched `workbench/tests` for literal `21`/`22`/`23`/`27` against
`with_metrics`/`duplicate_rows`: the only literal assertions are structurally derived
(`test_native_measurements_census.py:143,145,155` compare `counts` against
`len([row for row in measured ...])`), and the two hard-coded cases are `test_native_ingest_r16.py:132,150`
(`== 0` and `== 1` on a synthetic `tmp_path` tree, unrelated to the real tree) and
`test_native_measurements_identity.py:234` (`duplicate_rows == 0` on a synthetic tree). The real-tree
counts changed correctly in the same commit: `test_native_measurements_identity.py:305` `2 → 3` counted
airknow rows, with the reason in the diff. `358c3fc` updated every affected assertion.

**Stale numbers remain in prose docs** (not asserted, so not failures, but wrong at HEAD):

| file:line | says | correct at HEAD |
|---|---|---|
| `20261111-master-audit/BLOCKED_MODELS_VERIFICATION.vi.md:573` | `blocks_with_metrics 43`, `with_metrics 21` | `50` / `23` |
| `20261111-master-audit/BLOCKED_MODELS_VERIFICATION.vi.md:640-641` | `with_metrics` 19, `blocks_with_metrics` 43 | 23 / 50 |
| `20261111-master-audit/BASELINE.vi.md:404` | `with_metrics: 21` | 23 (it is a labeled snapshot) |
| `20261110-e2e-repair/EVIDENCE_INDEX.md:153,157` | `with_metrics 21` | 23 (labeled snapshot of `3828163`) |

`CENSUS_20261111.vi.md:356-386` is **correct**: it already records `with_metrics: 23`,
`duplicate_rows: 27`, `blocks_with_metrics: 50` and preserves the old payload explicitly as a snapshot.
Which is right? For HEAD, **23 / 27 are correct** — they are what `list_native_measurements()`, the very
function `/api/native-measurements` calls, returns. 21/22 describe the pre-`358c3fc` tree only.

## Per-failure classification (18)

| # | nodeid | assertion (≤10 words) | class |
|---|---|---|---|
| 1-2 | `test_circus_adapters_blip.py::test_resident_worker_failure_uses_shared_error_and_stays_failed[conesep / ptha_mtst]` | `'Adapter' object has no attribute 'batch_size'` | PRE-EXISTING · ENVIRONMENT |
| 3-4 | `test_circus_adapters_blip.py::test_resident_worker_reuse_and_cleanup[conesep / ptha_mtst]` | same `batch_size` AttributeError | PRE-EXISTING · ENVIRONMENT |
| 5 | `test_circus_cli.py::test_cli_3d_full_gallery_resume_and_gallery_invalidation` | `assert 'REUSED' == 'COMPLETED'` | FLAKY (passes alone) |
| 6-7 | `test_circus_tme_lifecycle.py::test_encoder_and_limn_failed_child_latch_and_preserve_log[encoder / limn]` | `execution environment not verified [UNCONFIGURED]` | PRE-EXISTING · ENVIRONMENT (`WORKBENCH_*_PYTHON` unset) |
| 8 | `test_circus_tme_precision.py::test_worker_skips_empty_stdin_lines` | `FileNotFoundError ... output.npy` | PRE-EXISTING · ENVIRONMENT |
| 9 | `test_f05_f08_f09_verification.py::test_f08_dcnet_config_data_dir_resolution` | `PosixPath('.../data/FashionIQ') == PosixPath(...)` | PRE-EXISTING |
| 10 | `test_handoff_commands.py::test_no_release_document_points_at_a_superseded_source_sha` | `TESTED_SOURCE_SHA c755673 does not contain production code on disk` | PRE-EXISTING (correct red; pin is stale) |
| 11 | `test_release_blocker_gpu_records.py::test_defect_b_g_with_no_target_gpu_verdict_is_pending_not_pass` | `release_verify.sh` exits 1, `RELEASE_BLOCKED` | PRE-EXISTING · ENVIRONMENT |
| 12-16 | `...test_release_verify_documented_invocations_produce_parseable_report[<lambda>0..4]` | `Invocation failed: ... verdict=RELEASE_BLOCKED` | PRE-EXISTING · ENVIRONMENT |
| 17 | `test_release_blocker_gpu_records.py::test_offline_verdict_names_deployment_reason` | same `RELEASE_BLOCKED` exit | PRE-EXISTING · ENVIRONMENT |
| 18 | `test_standalone_operator_commands.py::test_prepare_model_executes_only_declared_recipe_steps` | `[MANUAL] dcnet: ... NONDETERMINISTIC_WITHOUT_AUTHOR_ARTIFACT` | PRE-EXISTING |

**NEW-from-358c3fc: 0.** Proof: the 18 ids are set-identical to `sweep_final2.txt` and to the
`68ad75e` baseline, and the two native test files plus the new r16 file report `127 passed` when rerun
together with the frontend-coupled tests at HEAD.

Evidence for the classifications actually exercised:

```
$ pytest test_circus_cli.py::test_cli_3d_full_gallery_resume_and_gallery_invalidation -q
1 passed in 0.25s                       # FLAKY: fails only inside the full serial suite

$ pytest test_circus_adapters_blip.py -q
4 failed, 14 passed in 0.19s            # batch_size family deterministic alone

$ pytest test_native_ingest_r16.py test_native_measurements_identity.py test_native_measurements_census.py -q
61 passed, 5 warnings in 4.21s          # the changed module's own coverage is green

$ pytest test_web_scope_and_labels.py test_frontend_catalog.py test_frontend_layout_provenance.py \
         test_frontend_staleness.py test_frontend_requests.py test_derived_subset_status.py -q
66 passed in 2.91s
```

The `test_handoff_commands.py` failure is the **correct** red: `c755673` no longer contains
`native_measurements.py`'s bytes. It clears only when the release documents name a revision that does.

The `test_release_blocker_gpu_records.py` family (7 of 18) is environment-driven, not a code defect:
`release_verify.sh --offline` returns `RELEASE_BLOCKED` because J10 fails. Rerunning
`bash workbench/docker/release_verify.sh --offline` directly at 15:2x gave `J10 FAIL built bundle lacks
audit-threshold copy`, i.e. the gate does what it says. Its verdict flips to `FAIL J10 frontend/dist is
older than its sources` inside the suite because the bundle was rebuilt at 14:41 and the source files at
14:02 — both before the standalone rerun, so the copy check is the live one. Either way J10 is red
independently of `native_measurements.py`.

`test_circus_tme_precision.py::test_worker_skips_empty_stdin_lines` fails on
`require_verified_model_interpreter` (`runtime.py:2309`, `WORKBENCH_LIMN_PYTHON=None`) reached via
`model_limn.py:599` → `adapters.py:346`; the `output.npy` `FileNotFoundError` is its consequence.

## Conclusion

- **18 failed / 2931 passed / 47 skipped / 2996 collected, gap 0.**
- **0 NEW, 0 DROPPED, 18 COMMON** vs `68ad75e`; **set-identical** to `/tmp/sweep_final2.txt`.
- All 18 are pre-existing: 12 environment-driven, 1 flaky-in-suite-only, 1 correct stale-pin guard,
  1 path-resolution expectation, 1 operator-policy assertion, 2 unconfigured-interpreter lifecycle.
- The R16a ingest is **covered and green**: `with_metrics` 21→23 and `duplicate_rows` 22→27 are the
  correct served numbers at `358c3fc`, and every assertion that reads them was updated in the same
  commit. No test asserts a stale count.

**Do not pin this revision.** HEAD moved from `358c3fc` to `d323a5c` during the run, and the four
frontend production files (`scope.ts`, `data.ts`, `EvaluationPage.tsx`, `ModelInventory.tsx`) changed
after the sweep started. This sweep measured backend/composition/replay at `358c3fc`; frontend
production changed afterwards and is covered only by `node --test` + `tsc --noEmit`, not by this sweep.
A final sweep is required after the frontend work lands.

For the record, the revision this sweep *does* fully cover is `358c3fc612109d6d268f5afe655f10420a52f318`
with **2931 passed / 18 failed / 47 skipped / 2996 collected**. It is not the final pin: the frontend
commits above add production bytes it does not contain.
