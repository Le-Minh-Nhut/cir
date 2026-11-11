# 72 — Final authoritative sweep: the pin at `7babfd9`

Previous sweep `71-sweep-final.md` measured `2931 passed / 18 failed / 47 skipped / 2996 collected` at
`358c3fc`, but could not pin it: frontend production changed after it started (`252b380`, `d323a5c`),
and the release guard's production predicate (`DOCS_AND_TESTS_EXCLUDED_PATHS` in
`test_handoff_commands.py:351`) excludes only `workbench/docs` and `workbench/tests`. This sweep
re-measures the same suite at the new tip `7babfd9`, where all production is committed, and pins it.

## Identity

| fact | value |
|---|---|
| HEAD at sweep start | `7babfd96c1b5003d3fba7197bc6cbb1446a48a2b` (= branch tip, = `origin/research/cir-failure-workbench`) |
| HEAD at sweep end | `7babfd96c1b5003d3fba7197bc6cbb1446a48a2b` — **unchanged** |
| branch | `research/cir-failure-workbench` |
| tree at start | `?? CLEANUP_REPORT.md.save` only |
| tree at end | `?? CLEANUP_REPORT.md.save` only |
| interpreter | `workbench/envs/ctl312h/bin/python` |
| sweep launched | `2026-11-11T15:36:13+07:00` |
| sweep finished | `2026-11-11T16:33:49+07:00` |
| wall time | 3445.68 s (0:57:25) |

HEAD did not move during this run: `7babfd9` at launch, at both stationary proofs, and at the end.

### Production delta versus the un-pinnable `358c3fc`

```
$ git diff --stat 358c3fc 7babfd9 -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'
 workbench/frontend/EvaluationPage.tsx                    |  51 +++-
 workbench/frontend/ModelInventory.tsx                    | 114 +++++-----
 workbench/frontend/data.ts                               |  16 ++
 workbench/frontend/scope.ts                              | 160 ++++++++++++++-
 workbench/frontend/tests/inventory-cross-scope.test.cjs  |  99 ++++++++++
 workbench/frontend/tests/native-conflict-variance.test.cjs | 118 ++++++++++
 6 files changed, 499 insertions(+), 59 deletions(-)
```

Those six are exactly the production bytes `358c3fc` did not contain (respectively `252b380` and
`d323a5c`). No backend, composition or replay file differs. The served bundle
(`workbench/frontend/dist/assets/index-DyVaNikG.js`, `index-DecQnta9.css`, both mtime
`14:41:09 +0700`) is newer than every frontend source (latest `14:02:53`), i.e. built from these
sources in this tree — which is why J10 fails on the *copy* check, not the staleness check, below.

## Preflight (all mandatory, all PASS)

| check | result |
|---|---|
| `git rev-parse HEAD` | `7babfd96c1b5...`, equal to branch tip — PASS |
| `git status --porcelain` | exactly `?? CLEANUP_REPORT.md.save` (verified with `cat -A`) — PASS |
| `pgrep -af "pytest workbench/tests"` | empty — PASS |
| `nvidia-smi --query-compute-apps=pid,process_name --format=csv` | header only — PASS (no compute process) |
| `df -h /` | `630G 499G 99G 84% /` — 99 GiB avail ≥ 45 — PASS |
| `/tmp/sweep_basetemp_pin` | absent — PASS |
| `ls workbench/frontend/dist/assets/` | exactly `index-DecQnta9.css` + `index-DyVaNikG.js`, one `.js`, no stale bundle — PASS |

## Stationary-tree proof (twice, timestamped)

### Proof 1 — 4 min 47 s after launch

```
$ date -Is
2026-11-11T15:41:15+07:00
$ git rev-parse HEAD
7babfd96c1b5003d3fba7197bc6cbb1446a48a2b
$ git status --porcelain
?? CLEANUP_REPORT.md.save
```
(re-checked at `2026-11-11T15:41:20+07:00`: same HEAD, same single untracked entry.)

### Proof 2 — 30 min 10 s after launch

```
$ date -Is
2026-11-11T16:06:23+07:00
$ git rev-parse HEAD
7babfd96c1b5003d3fba7197bc6cbb1446a48a2b
$ git status --porcelain
?? CLEANUP_REPORT.md.save
$ pgrep -c -f 'pytest workbench/tests'
1
```
(re-checked at `2026-11-11T16:06:26+07:00`: same HEAD, same single untracked entry.)

The tree was stationary at both proofs: same commit, no modified tracked path, one untracked save file
that no test reads. Unlike sweep 71, nothing changed under the suite.

## Command (verbatim)

```
mkdir -p /tmp/sweep_basetemp_pin
setsid nohup /home/heheboiz/data/cir/workbench/envs/ctl312h/bin/python -m pytest workbench/tests \
  -q -p no:cacheprovider -rfE --tb=short --basetemp=/tmp/sweep_basetemp_pin \
  > /tmp/sweep_pin.txt 2>&1 < /dev/null &
```

Serial, no `-n`. Log `/tmp/sweep_pin.txt` is not copied into the repo; the summary line and greps
below are the record.

## Summary line (verbatim, last line of the log)

```
18 failed, 2931 passed, 47 skipped, 350 warnings in 3445.68s (0:57:25)
```

## The four accounting numbers

### 1. Error greps

| measure | value |
|---|---|
| `grep -c '^ERROR '` | **0** — no collection/setup errors |
| `grep -c 'Errno 28'` | **0** — no disk-full |
| `grep -c '^FAILED'` | **18** |

`^E ` deliberately not grepped: with `--tb=short` every traceback body line begins with `E `.

### 2. Collected vs reported

```
$ workbench/envs/ctl312h/bin/python -m pytest workbench/tests -q --collect-only -p no:cacheprovider
2996 tests collected in 3.59s
```

| N collected | passed | failed | skipped | sum | gap |
|---|---|---|---|---|---|
| 2996 | 2931 | 18 | 47 | 2996 | **0** |

### 3. Failure diff versus baseline `68ad75e`

Baseline `workbench/docs/reviews/20261110-e2e-repair/full_suite_68ad75e.txt`: 18 `FAILED` ids.
This run: the same 18 ids (`comm -3` on sorted node-id sets is empty).

| NEW | DROPPED | COMMON |
|---|---|---|
| **0** | **0** | **18** |

**NEW list: (empty).** **DROPPED list: (empty).**

### 4. Delta versus `/tmp/sweep_final3.txt` (sweep 71, `358c3fc`)

`comm -3` on the two sorted `FAILED` sets is **empty** — no failure appeared, none disappeared.
Same 18 ids, same 2931/18/47/2996 totals, same 0 gap. The difference between the two runs is identity,
not outcome: this run is at a HEAD that contains all production, the previous one was not.

Behavioural delta in the log, one line: J10's *reason* text changed from
`FAIL J10 frontend/dist is older than its sources (run vite build)` (sweep 71, bundle rebuilt after its
sources) to `FAIL J10 built bundle lacks audit-threshold copy` (now, bundle current, copy check live).
Both are J10 red for the same family of tests, so no node id moved.

## Per-failure classification (18)

| # | nodeid | assertion (≤10 words) | class |
|---|---|---|---|
| 1-2 | `test_circus_adapters_blip.py::test_resident_worker_failure_uses_shared_error_and_stays_failed[conesep / ptha_mtst]` | `'Adapter' object has no attribute 'batch_size'` | PRE-EXISTING · ENVIRONMENT |
| 3-4 | `test_circus_adapters_blip.py::test_resident_worker_reuse_and_cleanup[conesep / ptha_mtst]` | same `batch_size` AttributeError | PRE-EXISTING · ENVIRONMENT |
| 5 | `test_circus_cli.py::test_cli_3d_full_gallery_resume_and_gallery_invalidation` | `assert 'REUSED' == 'COMPLETED'` | FLAKY (passes alone, 0.20 s) |
| 6-7 | `test_circus_tme_lifecycle.py::test_encoder_and_limn_failed_child_latch_and_preserve_log[encoder / limn]` | `execution environment not verified [UNCONFIGURED]` | PRE-EXISTING · ENVIRONMENT (`WORKBENCH_ENCODER_PYTHON` / `WORKBENCH_LIMN_PYTHON` unset) |
| 8 | `test_circus_tme_precision.py::test_worker_skips_empty_stdin_lines` | `FileNotFoundError ... /output.npy` | PRE-EXISTING · ENVIRONMENT (unverified interpreter → no artifact) |
| 9 | `test_f05_f08_f09_verification.py::test_f08_dcnet_config_data_dir_resolution` | `PosixPath('.../data/FashionIQ') == PosixPath('.../third_party/DCNet/...')` | PRE-EXISTING |
| 10 | `test_handoff_commands.py::test_no_release_document_points_at_a_superseded_source_sha` | `TESTED_SOURCE_SHA c755673 does not contain production on disk` | PRE-EXISTING (correct red: `workbench/docs/PC_PILOT_HANDOFF.md:31,63` names `c755673`) |
| 11 | `test_release_blocker_gpu_records.py::test_defect_b_g_with_no_target_gpu_verdict_is_pending_not_pass` | `release_verify.sh` exits 1, `RELEASE_BLOCKED` | PRE-EXISTING · ENVIRONMENT (J10) |
| 12-16 | `...test_release_verify_documented_invocations_produce_parseable_report[<lambda>0..4]` | `Invocation failed ... verdict=RELEASE_BLOCKED` | PRE-EXISTING · ENVIRONMENT (J10) |
| 17 | `test_release_blocker_gpu_records.py::test_offline_verdict_names_deployment_reason` | same `RELEASE_BLOCKED` exit | PRE-EXISTING · ENVIRONMENT (J10) |
| 18 | `test_standalone_operator_commands.py::test_prepare_model_executes_only_declared_recipe_steps` | `[MANUAL] dcnet ... NONDETERMINISTIC_WITHOUT_AUTHOR_ARTIFACT` | PRE-EXISTING |

No failure is attributable to `7babfd9` or to the two frontend commits: the id set is identical to the
`68ad75e` baseline (a day earlier, before the frontend work) and to sweep 71 at `358c3fc` (before the
frontend commits landed). NEW = 0 is therefore doubly witnessed.

**Tally of the 18** — **13 PRE-EXISTING · ENVIRONMENT** (#1-4, #6-8, #11-17), **2 PRE-EXISTING
non-environment** (#9 path expectation, #18 operator policy), **1 PRE-EXISTING correct-red guard**
(#10 stale `c755673` pin), **1 FLAKY** (#5), **0 NEW**, **0 DROPPED**.

Evidence for the classes actually exercised:

```
$ pytest test_circus_cli.py::test_cli_3d_full_gallery_resume_and_gallery_invalidation -q   # own basetemp
1 passed in 0.20s                       # FLAKY: fails only inside the full serial suite

$ pytest test_circus_adapters_blip.py -q                                                  # own basetemp
4 failed, 14 passed in 0.20s            # batch_size family deterministic alone
```

The J10 family (#11-17, 7 of 18) is environment-driven, not a code defect: `release_verify.sh
--offline` reports `FAIL J10 built bundle lacks audit-threshold copy`, so the gate does what it says
while J10 is genuinely red. `#10` is the correct red: `PC_PILOT_HANDOFF.md:31` pins `c755673`, which no
longer matches the production tree; it clears only when a release document names a revision that does.

## The six frontend-related tests (evidence for `252b380`, `d323a5c`)

Each run individually at HEAD (`pytest workbench/tests/<file>.py -q`):

| test file | result |
|---|---|
| `test_web_scope_and_labels.py` (compiles/executes `scope.ts`) | **14 passed** |
| `test_frontend_catalog.py` (reads `dist/assets/*.js`) | **18 passed** |
| `test_frontend_requests.py` | **9 passed** |
| `test_frontend_staleness.py` | **7 passed** |
| `test_frontend_layout_provenance.py` | **13 passed** |
| `test_pc_launcher.py` | **101 passed** |

Combined: `162 passed in 120.81s`. All six green, inside the full sweep as well as alone. The
conflict/variance rendering (`scope.ts`, `EvaluationPage.tsx`, `data.ts`) and the cross-scope
`ModelInventory.tsx` fix broke nothing the suite can observe.

## Conclusion

- **18 failed / 2931 passed / 47 skipped / 2996 collected, gap 0.**
- HEAD stationary at `7babfd9` at launch, at 15:41, at 16:06, and at the end; tree carried only the
  untracked `CLEANUP_REPORT.md.save` throughout.
- **0 NEW, 0 DROPPED, 18 COMMON** vs `68ad75e`; **set-identical** to sweep 71.
- The six frontend-coupled test files pass 162/162, so the two frontend commits are covered.
- The 18 are pre-existing: 13 environment (unconfigured interpreters, J10 gate), 1 flaky-in-suite-only,
  1 correct stale-pin guard, 2 expectation/policy assertions.

**pin `7babfd96c1b5003d3fba7197bc6cbb1446a48a2b` with 2931 passed / 18 failed / 47 skipped / 2996 collected; production tree verified identical to HEAD at sweep start and end**
