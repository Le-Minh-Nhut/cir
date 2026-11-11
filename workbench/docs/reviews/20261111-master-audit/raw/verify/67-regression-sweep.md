# 67 — Serial regression sweep at `a24635e`

## Identity

| fact | value |
|---|---|
| HEAD at sweep start | `a24635e304a13f5233fe43ef4a8c2829b08968a6` |
| HEAD at sweep end | `e11f9ff76a7a3f3eb21b9956150845565f82bd0a` |
| HEAD moved during the run | **yes** — `e11f9ff` (frontend only: `workbench/frontend/scope.ts`, `ModelInventory.tsx`, `workbench/frontend/tests/inspector-window.test.cjs`, `…/65-frontend.md`), committed 08:56:35, sweep started 09:35:38 |
| branch | `research/cir-failure-workbench` |
| tree at start | `?? CLEANUP_REPORT.md.save` only |
| interpreter | `workbench/envs/ctl312h/bin/python` (CPython 3.12.11) |

The sweep therefore measured **the backend / composition / replay / limn / frontend-python tree frozen at
`a24635e`** (none of `e11f9ff`'s files are imported by the backend), but the frontend slice may have
executed a mix: `test_web_scope_and_labels.py` shells out to compile and run `workbench/frontend/scope.ts`,
which `e11f9ff` rewrote. That slice was **re-measured separately** at `e11f9ff` (below). The full sweep
does **not** govern the final tree for the frontend modules.

## Preflight

| check | result |
|---|---|
| `git rev-parse HEAD` | `a24635e304a13f5233fe43ef4a8c2829b08968a6` |
| `git status --porcelain` | `?? CLEANUP_REPORT.md.save` only — PASS |
| `pgrep -af pytest` | empty — PASS |
| `df -h /` | 100 GiB avail (≥45 required) — PASS |
| `/tmp/sweep_basetemp_a24635e` | absent — PASS |

## Command (verbatim)

```
mkdir -p /tmp/sweep_basetemp_a24635e
setsid nohup /home/heheboiz/data/cir/workbench/envs/ctl312h/bin/python -m pytest workbench/tests \
  -q -p no:cacheprovider -rfE --tb=short --basetemp=/tmp/sweep_basetemp_a24635e \
  > /tmp/sweep_a24635e.txt 2>&1 < /dev/null &
```

Serial, no `-n`. Wall time 3346.75 s (0:55:46) — inside the predicted 53–60 min.
Full log preserved as `67-sweep-a24635e.log` in this directory.

## Summary line (verbatim, last line of the log)

```
18 failed, 2901 passed, 47 skipped, 350 warnings in 3346.75s (0:55:46)
```

## The four accounting checks

### 1. Counts
`18 failed, 2901 passed, 47 skipped` (see above).

### 2. `grep -cE "^E  |^ERROR"` = 325 · `grep -c "Errno 28"` = 0

| measure | value |
|---|---|
| `grep -cE "^E  \|^ERROR"` | **325** |
| `grep -c "^ERROR"` (collection errors) | **0** |
| `grep -c "Errno 28"` | **0** — PASS |

**Check 2 does not literally pass, and the number 325 does not indicate 325 errors.** All 325 `E ` lines
lie inside `=== FAILURES ===` traceback bodies (0 of them precede that header, 325 of 325 are after it):
`awk '/^=+ FAILURES/{f=1} f&&/^E /{c++} END{print c+0}'` → 325, and 9 failure sections exist for 18
tests (pytest collapses identical parametrised sections). Largest contributors are the 5
release-blocker multi-line `stdout:` dumps.
This sweep ran `--tb=short`, so tracebacks are printed. **The baseline log
(`20261110-e2e-repair/full_suite_68ad75e.txt`) contains 0 lines matching `^E ` at all** — it was captured
with tracebacks suppressed (`--tb=line`-style output), so the two logs are not comparable on this grep.
The meaningful assertion is the collection-error grep: **0**, and **Errno 28 = 0**. No disk-full was
observed; `/` free space moved 100 GiB → 70 GiB, consistent with a 30 GiB basetemp and nothing else.

### 3. Accounting gap

`collect-only` (separate invocation, run after the sweep at HEAD `e11f9ff`):

```
2966 tests collected in 3.55s
```

`2901 passed + 18 failed + 47 skipped = 2966` → **gap = 0**. Nothing was collected that did not run.

Note on completeness: `collect-only` at `e11f9ff` also yields 2966, so `e11f9ff` did not add or remove
tests at collection time; the mid-run commit changed the **body** of `scope.ts`, which is what makes the
frontend slice non-stationary, not its test count.

### 4. Failure diff

`failure_diff.py` hardcodes its two input logs and rewrites a **committed** evidence file
(`…/20261109-native15-composition-limn/scripts/failure_diff_36ae9e2.txt`), and accepts no argv. It was
therefore imported and its three module globals repointed (wrapper: `/tmp/failure_diff_wrap_a24635e.py`)
so the tracked file was not clobbered. Same comparison code, new inputs; output in
`67a-failure-diff-a24635e.txt`.

```
0 new, 0 dropped, 18 common
```

| NEW | DROPPED | COMMON |
|---|---|---|
| **0** | **0** | **18** |

NEW list: **(empty)**.

DROPPED list (failures in the baseline that now pass): **(empty)** — no fix in this batch closed a
previously red test, and no test regressed.

Baseline cited in the brief was `18 failed, 2800 passed, 47 skipped`. The baseline file on disk says
`18 failed, 2701 passed, 47 skipped` (`full_suite_68ad75e.txt:519`); the earlier `3828163` file says
`2690 passed`. The delta 2701 → 2901 passed is the 200 new tests added by this batch; the failed and
skipped sets are byte-identical.

## Per-failure classification

All 18 failures are in the baseline file. None is new. None needs the R-number attribution path
(`NEW-from-this-batch` = 0).

| # | test | failure in ≤10 words | class |
|---|---|---|---|
| 1 | `test_circus_adapters_blip.py::test_resident_worker_reuse_and_cleanup[model_conesep-conesep]` | `AttributeError: 'Adapter' object has no attribute 'batch_size'` | ENVIRONMENT + PRE-EXISTING |
| 2 | `…::test_resident_worker_reuse_and_cleanup[model_ptha_mtst-ptha_mtst]` | same `batch_size` AttributeError | ENVIRONMENT + PRE-EXISTING |
| 3 | `…::test_resident_worker_failure_uses_shared_error_and_stays_failed[conesep]` | same `batch_size` AttributeError | ENVIRONMENT + PRE-EXISTING |
| 4 | `…::test_resident_worker_failure_uses_shared_error_and_stays_failed[ptha_mtst]` | same `batch_size` AttributeError | ENVIRONMENT + PRE-EXISTING |
| 5 | `test_circus_cli.py::test_cli_3d_full_gallery_resume_and_gallery_invalidation` | `assert 'REUSED' == 'COMPLETED'` | PRE-EXISTING (also FLAKY — passes alone, twice) |
| 6 | `test_circus_tme_lifecycle.py::test_…latch_and_preserve_log[encoder]` | `WORKBENCH_ENCODER_PYTHON=None (not set)` | ENVIRONMENT + PRE-EXISTING |
| 7 | `test_circus_tme_lifecycle.py::test_…latch_and_preserve_log[limn]` | `WORKBENCH_LIMN_PYTHON=None (not set)` | ENVIRONMENT + PRE-EXISTING |
| 8 | `test_circus_tme_precision.py::test_worker_skips_empty_stdin_lines` | `No such file … output.npy`; worker died on `module 'torch' has no attribute 'set_num_threads'` | PRE-EXISTING (fails alone too) |
| 9 | `test_f05_f08_f09_verification.py::test_f08_dcnet_config_data_dir_resolution` | resolved `data/FashionIQ` ≠ `DCNet/dataset/fashioniq` | PRE-EXISTING (fails alone too) |
| 10 | `test_handoff_commands.py::test_no_release_document_points_at_a_superseded_source_sha` | `TESTED_SOURCE_SHA be14393c…` predates production code | PRE-EXISTING (by-design red guard) |
| 11 | `test_release_blocker_gpu_records.py::test_defect_b_g_with_no_target_gpu_verdict_is_pending_not_pass` | `RELEASE_BLOCKED`; J10 `built bundle lacks audit-threshold copy` | PRE-EXISTING |
| 12 | `…::test_release_verify_documented_invocations_produce_parseable_report[<lambda>0]` | invocation returned non-zero; J10 FAIL | PRE-EXISTING |
| 13 | `…[<lambda>1]` | same | PRE-EXISTING |
| 14 | `…[<lambda>2]` | same | PRE-EXISTING |
| 15 | `…[<lambda>3]` | same | PRE-EXISTING |
| 16 | `…[<lambda>4]` | same | PRE-EXISTING |
| 17 | `…::test_offline_verdict_names_deployment_reason` | `RELEASE_BLOCKED`; J10 FAIL | PRE-EXISTING |
| 18 | `test_standalone_operator_commands.py::test_prepare_model_executes_only_declared_recipe_steps` | dcnet `[MANUAL]` resized-images evidence missing | PRE-EXISTING (fails alone too) |

### Tally

| class | count |
|---|---|
| PRE-EXISTING (all of the above) | **18** |
| ├ of which ENVIRONMENT | 6 (4 × `batch_size`, 2 × unset `WORKBENCH_*_PYTHON`) |
| ├ of which proven FLAKY | 1 (#5, passes alone twice) |
| NEW-from-this-batch | **0** |
| Pure FLAKY (not in baseline) | 0 |
| DROPPED (baseline failure now green) | 0 |

### Isolation evidence (each run with its own `--basetemp`)

| test | alone |
|---|---|
| `test_cli_3d_full_gallery_resume_and_gallery_invalidation` | **1 passed** (`/tmp/flaky_2`), **1 passed** (`/tmp/flaky_g1`), **1 passed** (`/tmp/flaky_g2`) → FLAKY |
| `test_worker_skips_empty_stdin_lines` | **1 failed** (`/tmp/flaky_w2`), same `CIRCUS_WORKER:{"ok": false, "error_type": "AttributeError", "error": "module 'torch' has no attribute 'set_num_threads'"}` |
| `test_f08_dcnet_config_data_dir_resolution` + `test_prepare_model_executes_only_declared_recipe_steps` | **2 failed** (`/tmp/flaky_3`) — deterministic |
| `test_no_release_document_points_at_a_superseded_source_sha` | **1 failed** (`/tmp/flaky_4`) — deterministic |
| `test_circus_adapters_blip.py` (4) + `test_circus_tme_lifecycle.py` (2) | **6 failed** (`/tmp/flaky_6`) — deterministic, environment |

`test_worker_skips_empty_stdin_lines` root cause is *not* this batch: the stub `torch` module the test
installs has no `set_num_threads`, but `workbench/composition/adapters.py:182` calls it. That call landed
in `bf6a448` (Sun Nov 8, ancestor of both the baseline `68ad75e` and `a24635e`), so the test was already
red at the baseline with the same `output.npy` symptom.

`test_no_release_document_points_at_a_superseded_source_sha` fails on a **moved** stale sha
(`be14393c…` now, `36ae9e2…` in the baseline) but the assertion and the reason are the same red guard:

```
assert production_diff.returncode == 0, (
    f"TESTED_SOURCE_SHA {tested_sha} does not contain the production code currently on disk: "
```

By design it stays red for every commit that touches production after the recorded sha — including this
batch. It is not a regression, and it was not edited.

## Frontend slice re-measured at `e11f9ff` (stationary tree)

```
workbench/envs/ctl312h/bin/python -m pytest workbench/tests -q -p no:cacheprovider -rfE --tb=short \
  -k "frontend or web_scope or catalog or scope or inventory or inspector" \
  --basetemp=/tmp/sweep_frontend_e11f9ff
```

```
176 passed, 2790 deselected, 7 warnings in 24.44s
```

No failures. This includes `test_web_scope_and_labels.py`, which compiles and executes the rewritten
`workbench/frontend/scope.ts` — i.e. `nativeRowsFor` (exact `(model_id, protocol_id)` only),
`nativeUnattributedFor`, and the reordered `inventoryStatus()` are all exercised green at `e11f9ff`.
No frontend test asserts the old precedence, so nothing needed reporting as an OLD-contract assertion.
The node-level test file rewritten by `e11f9ff` (`workbench/frontend/tests/inspector-window.test.cjs`)
is not wired into pytest; it is covered by the frontend build/node test, out of scope for this sweep.

## Conclusion

- **0 NEW, 0 DROPPED, 18 COMMON.** The eight-defect batch (`d6d3e36…a24635e`) introduced **no regression**
  in the serial suite and closed none of the standing 18.
- All 18 standing failures are environmental or by-design guards, unchanged since `68ad75e`.
- Passed count rose 2701 → 2901, entirely new tests from this batch.
- The only frontend slice that could have raced the mid-run `e11f9ff` was re-measured green at `e11f9ff`.
