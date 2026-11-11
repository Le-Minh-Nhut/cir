# 73 — Closing sweep: the pin moves to `0691f3e`

Record `72-sweep-pin.md` pinned `7babfd9` on the grounds that it was the first revision containing all
production then on disk. It no longer is: `0691f3e` (`frontend(scope): counted-only evidence in native
scope helpers (R08)`) changed `workbench/frontend/scope.ts` — the counted-only predicate inside
`nativeRowsFor`, `nativeUnattributedFor` and `nativeMeasuredIds` — plus the frontend test
`workbench/frontend/tests/inventory-cross-scope.test.cjs`. The release guard's production predicate
(`DOCS_AND_TESTS_EXCLUDED_PATHS` in `workbench/tests/test_handoff_commands.py:351`) excludes only
`workbench/docs` and `workbench/tests`, so `workbench/frontend/tests` **is** compared production: the
`7babfd9` pin is superseded by one production commit. This sweep measures the suite at `0691f3e` and
pins the tip.

`ModelInventory.tsx` is *not* part of this delta: its last commit is `d323a5c`, already inside
`7babfd9`, and `git diff 7babfd9 0691f3e -- workbench/frontend/ModelInventory.tsx` is empty.

## Identity

| fact | value |
|---|---|
| HEAD at sweep start | `0691f3e88a03770aec0395843509da8f91a5d7ee` (= branch tip, = `origin/research/cir-failure-workbench`) |
| HEAD at sweep end | `0691f3e88a03770aec0395843509da8f91a5d7ee` — **unchanged** |
| branch | `research/cir-failure-workbench` |
| tree at start | `?? CLEANUP_REPORT.md.save` only |
| tree at end | `?? CLEANUP_REPORT.md.save` only |
| interpreter | `workbench/envs/ctl312h/bin/python` |
| sweep launched | `2026-11-11T16:59:19+07:00` |
| sweep finished | `2026-11-11T17:53:00+07:00` |
| wall time | 3203.74 s (0:53:23) |

### Production delta versus `7babfd9`

```
$ git log --oneline 7babfd9..HEAD
0691f3e frontend(scope): counted-only evidence in native scope helpers (R08)
add9a25 docs: state which round each pin belongs to and pin the live bundle
70b6164 Pin the master-audit reports to TESTED_SOURCE_SHA 7babfd9 and the live bundle
3f6c6f8 docs(release): pin TESTED_SOURCE_SHA at 7babfd9 — 2931/18/47/2996
06150b1 docs(verify): 72-sweep-pin — 2931/18/47/2996 at 7babfd9, stationary, pin

$ git diff --stat 7babfd9 0691f3e -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'
 workbench/frontend/scope.ts                        | 17 ++++--
 .../frontend/tests/inventory-cross-scope.test.cjs  | 62 ++++++++++++++++++----
 2 files changed, 67 insertions(+), 12 deletions(-)
```

Four of the five commits are docs only (`workbench/docs/**`, excluded by the guard). Only `0691f3e`
moves the compared tree.

The one production change, verbatim from `git diff 7babfd9 0691f3e -- workbench/frontend/scope.ts`:

```ts
+function counts(row: NativeRowLike): boolean {
+  return statesNumber(row) && row.counted !== false;
+}
...
-  return rows.filter((row) => statesNumber(row) && row.model_id === model && row.protocol_id === protocol);
+  return rows.filter((row) => counts(row) && row.model_id === model && row.protocol_id === protocol);
...
-  return rows.filter((row) => statesNumber(row) && row.model_id === model && !row.protocol_id);
+  return rows.filter((row) => counts(row) && row.model_id === model && !row.protocol_id);
...
-    if (!row.model_id || !statesNumber(row) || row.protocol_id !== protocol) continue;
+    if (!row.model_id || !counts(row) || row.protocol_id !== protocol) continue;
```

The served bundle carries it: `grep -c 'counted!==!1' workbench/frontend/dist/assets/index-BLH2yome.js`
= `1`, and the minified helper reads `function Je(e){return qe(e)&&e.counted!==!1}`. Bundle mtime
`16:53:00 +0700` is 37 s after `scope.ts`'s `16:52:23`, i.e. built from these sources in this tree.

## Preflight (all mandatory, all PASS)

| check | result |
|---|---|
| `git rev-parse HEAD` | `0691f3e88a03...`, equal to `research/cir-failure-workbench` and to the checked-out HEAD — PASS |
| `git status --porcelain` | exactly `?? CLEANUP_REPORT.md.save` (`-uall` count = 1) — PASS |
| `pgrep -af "pytest workbench/tests"` | empty (`exit=1`) — PASS |
| `nvidia-smi --query-compute-apps=pid,process_name --format=csv` | header only, no compute process — PASS |
| `df -h /` | `630G 499G 99G 84% /` — 99 GiB avail ≥ 45 — PASS |
| `/tmp/sweep_basetemp_close` | absent — PASS |
| `ls workbench/frontend/dist/assets/` | exactly `index-BLH2yome.js` (376.8K) + `index-DecQnta9.css` (25.9K) — one `.js`, no stale bundle — PASS |

## Stationary-tree proof (twice, timestamped)

### Proof 1 — 4 min 12 s after launch

```
$ date -Is                      2026-11-11T17:03:31+07:00
$ git rev-parse HEAD            0691f3e88a03770aec0395843509da8f91a5d7ee
$ git status --porcelain        ?? CLEANUP_REPORT.md.save
progress                        282 bytes of log, 7% of the progress bar
```

### Proof 2 — 29 min 16 s after launch

```
$ date -Is                      2026-11-11T17:28:35+07:00
$ git rev-parse HEAD            0691f3e88a03770aec0395843509da8f91a5d7ee
$ git status --porcelain        ?? CLEANUP_REPORT.md.save
progress                        2732 bytes of log, 81% of the progress bar
```

HEAD equals the start value at both proofs; the tree never carried anything but the untracked
`CLEANUP_REPORT.md.save`. Neither proof failed, so a pin is admissible.

## Command (verbatim, serial — no `-n`)

```
mkdir -p /tmp/sweep_basetemp_close
setsid nohup /home/heheboiz/data/cir/workbench/envs/ctl312h/bin/python -m pytest workbench/tests \
  -q -p no:cacheprovider -rfE --tb=short --basetemp=/tmp/sweep_basetemp_close \
  > /tmp/sweep_close.txt 2>&1 < /dev/null &
```

## Summary line (verbatim, last line of `/tmp/sweep_close.txt`)

```
18 failed, 2931 passed, 47 skipped, 350 warnings in 3203.74s (0:53:23)
```

## The four accounting numbers

### 1. Error greps (never `^E `)

```
$ grep -c '^ERROR ' /tmp/sweep_close.txt     -> 0
$ grep -c 'Errno 28' /tmp/sweep_close.txt    -> 0
$ grep -c '^FAILED' /tmp/sweep_close.txt     -> 18
```

No disk-full event, no collection error, no setup error. Every one of the 18 is an assertion failure.

### 2. Collected versus reported

```
$ pytest workbench/tests -q --collect-only    ->  2996 tests collected in 3.14s  (exit 0)
reported at run end                           ->  2931 passed + 18 failed + 47 skipped = 2996
```

**2931 + 18 + 47 = 2996 = N. Gap = 0.** Nothing was silently deselected, errored out, or lost.

### 3. Failure diff versus baseline `68ad75e`

Baseline `workbench/docs/reviews/20261110-e2e-repair/full_suite_68ad75e.txt` (`18 failed, 2701 passed,
47 skipped`):

| set | count | members |
|---|---|---|
| COMMON | 18 | all 18 nodeids, set-identical |
| NEW | **0** | — |
| DROPPED | **0** | — |

NEW list: **empty**. The `passed` total differs from the baseline (2931 vs 2701) only because the
baseline predates ~230 tests added during the master-audit rounds; the failing id set did not move.

### 4. Delta versus `/tmp/sweep_pin.txt` (sweep 72, `7babfd9`)

| set | count | members |
|---|---|---|
| appeared | **0** | — |
| disappeared | **0** | — |

The failure set is **set-identical across three points**: `68ad75e` (a day earlier, before the frontend
work), `7babfd9` (immediately before `0691f3e`), and `0691f3e` (this run). NEW = 0 is therefore
witnessed three ways, and the counted-only change demonstrably breaks nothing the suite observes.

## Per-failure classification (18)

| # | nodeid | assertion (≤10 words) | class |
|---|---|---|---|
| 1-2 | `test_circus_adapters_blip.py::test_resident_worker_failure_uses_shared_error_and_stays_failed[conesep / ptha_mtst]` | `'Adapter' object has no attribute 'batch_size'` | PRE-EXISTING · ENVIRONMENT |
| 3-4 | `test_circus_adapters_blip.py::test_resident_worker_reuse_and_cleanup[conesep / ptha_mtst]` | same `batch_size` AttributeError | PRE-EXISTING · ENVIRONMENT |
| 5 | `test_circus_cli.py::test_cli_3d_full_gallery_resume_and_gallery_invalidation` | `assert 'REUSED' == 'COMPLETED'` | FLAKY (passes alone, 0.23 s) |
| 6-7 | `test_circus_tme_lifecycle.py::test_encoder_and_limn_failed_child_latch_and_preserve_log[encoder / limn]` | `execution environment not verified [UNCONFIGURED/UNCONFIGURED]` | PRE-EXISTING · ENVIRONMENT (`WORKBENCH_ENCODER_PYTHON` / `WORKBENCH_LIMN_PYTHON` unset) |
| 8 | `test_circus_tme_precision.py::test_worker_skips_empty_stdin_lines` | `FileNotFoundError ... /tmp/sweep_basetemp_close/...` | PRE-EXISTING · ENVIRONMENT (unverified interpreter → no artifact) |
| 9 | `test_f05_f08_f09_verification.py::test_f08_dcnet_config_data_dir_resolution` | `PosixPath('.../data/FashionIQ') == PosixPath('.../third_party/DCNet/...')` | PRE-EXISTING |
| 10 | `test_handoff_commands.py::test_no_release_document_points_at_a_superseded_source_sha` | `TESTED_SOURCE_SHA 7babfd9 does not contain production on disk` | PRE-EXISTING (correct red — see below) |
| 11 | `test_release_blocker_gpu_records.py::test_defect_b_g_with_no_target_gpu_verdict_is_pending_not_pass` | `release_verify.sh` exits 1, `RELEASE_BLOCKED` | PRE-EXISTING · ENVIRONMENT (J10) |
| 12-16 | `...test_release_verify_documented_invocations_produce_parseable_report[<lambda>0..4]` | `Invocation failed ... verdict=RELEASE_BLOCKED` | PRE-EXISTING · ENVIRONMENT (J10) |
| 17 | `test_release_blocker_gpu_records.py::test_offline_verdict_names_deployment_reason` | same `RELEASE_BLOCKED` exit | PRE-EXISTING · ENVIRONMENT (J10) |
| 18 | `test_standalone_operator_commands.py::test_prepare_model_executes_only_declared_recipe_steps` | `[MANUAL] dcnet ... NONDETERMINISTIC_WITHOUT_AUTHOR_ARTIFACT` | PRE-EXISTING |

**Tally** — 13 PRE-EXISTING · ENVIRONMENT (#1-4, #6-8, #11-17), 2 PRE-EXISTING non-environment (#9 path
expectation, #18 operator policy), 1 PRE-EXISTING correct-red guard (#10), 1 FLAKY (#5), **0 NEW**,
**0 DROPPED**.

Evidence actually exercised in this run:

```
$ pytest workbench/tests/test_circus_cli.py::test_cli_3d_full_gallery_resume_and_gallery_invalidation -q
1 passed in 0.23s                       # FLAKY: fails only inside the full serial suite

$ pytest workbench/tests/test_circus_adapters_blip.py -q
4 failed, 14 passed in 0.20s            # batch_size family deterministic alone
```

Note on #10: its assertion text has now advanced from `c755673` (`72`) to `7babfd9` — the same guard,
naming the pin the release documents currently carry. It is the correct red for this tree: the docs
still say `TESTED_SOURCE_SHA 7babfd9` while production on disk is `0691f3e`. It clears only when a
release document names `0691f3e`. The guard working *against the previous pin* is itself the reason
this sweep exists.

The J10 family (#11-17, 7 of 18) is environment-driven, not a code defect: the log carries
`FAIL J10 built bundle lacks audit-threshold copy` (lines 176, 238, 300, …), so the gate does what it
says. Confirmed independently: `grep -c 'audit-threshold' dist/assets/index-BLH2yome.js` = `0`, the
same bundle state as at `7babfd9`.

## The six frontend-coupled tests (the witnesses for `0691f3e`)

Each run individually at HEAD (`pytest workbench/tests/<file>.py -q`, own basetemp):

| test file | role | result |
|---|---|---|
| `test_web_scope_and_labels.py` | compiles and executes `workbench/frontend/scope.ts` (`support.compile_module`) | **14 passed** |
| `test_frontend_catalog.py` | compiles `catalog.ts`; reads the frontend unit surface | **18 passed** |
| `test_frontend_requests.py` | compiles `requests.ts` | **9 passed** |
| `test_frontend_staleness.py` | staleness rules over frontend source | **7 passed** |
| `test_frontend_layout_provenance.py` | compiles `catalog.ts` for layout provenance | **13 passed** |
| `test_pc_launcher.py` | reads `workbench/frontend/dist/assets` (bundle) | **101 passed** (115.66 s) |

Combined: **162 passed, 0 failed**. All six are green inside the full sweep as well as alone.

`test_web_scope_and_labels.py` is the true witness for the changed bytes: it recompiles `scope.ts`
from source, so the new `counts()` predicate is executed, not merely read. Its cases exercise
`chooseInitialScope`, `scopeCounts`, `chooseSubsetFor`, `otherScopeFor`, `resultLabels`,
`protocolHasSubsets` and `subsetLabelFor` — the same module the three native helpers live in, and
`otherScopeFor` (called at `scope.test` lines 168-170, 258, 275) is the inventory path the counted-only
rule changes. No assertion in that file inspects `nativeRowsFor` / `nativeUnattributedFor` /
`nativeMeasuredIds` by name (`grep` count 0 for each), so the *behaviour* is witnessed, the *new
predicate branch* (`row.counted === false` on an otherwise-counting row) is **not** directly
discriminated by this file — its dedicated coverage lives in the node suite
`workbench/frontend/tests/inventory-cross-scope.test.cjs`, which `pytest workbench/tests` does **not**
run (`grep -rn 'inventory-cross-scope' workbench/tests/*.py` = 0 matches; `frontend/package.json` has
no `test` script). Stated plainly: the python suite shows the change breaks nothing, it does not show
the new rule is correct. That gap is a coverage observation, not a failure.

## Conclusion

- **18 failed / 2931 passed / 47 skipped / 2996 collected, gap 0.**
- HEAD stationary at `0691f3e` at launch (16:59:19), at 17:03:31, at 17:28:35, and at the end
  (17:53:00). Tree carried only the untracked `CLEANUP_REPORT.md.save` throughout.
- **0 NEW, 0 DROPPED, 18 COMMON** vs `68ad75e`; **set-identical** to sweep 72 at `7babfd9`.
- **0 ERRORS**, **0 Errno 28**.
- The six frontend-coupled test files pass 162/162, so `0691f3e`'s production change is covered by the
  suite as far as the suite reaches; the counted-only branch itself is covered only by the node suite.
- The 18 are pre-existing: 13 environment (unconfigured interpreters, J10 gate), 1 flaky-in-suite-only,
  1 correct stale-pin guard (now naming `7babfd9`), 2 expectation/policy assertions.

Production tree at `0691f3e` is identical to the on-disk tree at sweep start and end (`git status`
clean apart from the untracked save file), so the pin below contains exactly the bytes measured —
including the counted-only `scope.ts` and the bundle built from it.

**pin `0691f3e88a03770aec0395843509da8f91a5d7ee` with 2931 passed / 18 failed / 47 skipped / 2996 collected; production tree verified identical to HEAD at sweep start and end**
