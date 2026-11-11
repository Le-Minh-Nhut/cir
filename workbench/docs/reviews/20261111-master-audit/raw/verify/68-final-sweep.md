# 68 — Final serial sweep at `cd980dc`

## Identity

| fact | value |
|---|---|
| HEAD at sweep start | `cd980dcecbb685b3cd6be4516d6550660a8e89b8` |
| HEAD at sweep end | `77ea629a08fdd1e5b6e0182c258e879e59fb7031` |
| HEAD moved during the run | **yes** — `77ea629` (`docs(audit): independent-review response, R01-R16 issue -> fix -> regression`), committed `2026-11-11 10:00:08 +0700`, sweep started `09:54:49 +0700` (5 min 19 s in) |
| branch | `research/cir-failure-workbench` |
| tree at start / end | `?? CLEANUP_REPORT.md.save` only (unchanged) |
| interpreter | `workbench/envs/ctl312h/bin/python` |
| wall time | 3486.46 s (0:58:06), sweep ended `2026-11-11T10:52:55+0700` |

### What mid-run `77ea629` did and did not change

One file, documentation only:

```
workbench/docs/reviews/20261111-master-audit/INDEPENDENT_REVIEW_RESPONSE_20261111.vi.md | 517 +++++
1 file changed, 517 insertions(+)
```

**The production tree is byte-identical between the start and end revisions:**

```
$ git diff --stat cd980dc 77ea629 -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'
(empty)
```

No test file changed either (the commit touches only `workbench/docs`). The added path is not one of
`test_handoff_commands.py`'s `RELEASE_DOCS` (fixed 5-file tuple) and no test recursively globs
`workbench/docs/reviews/**` (grepped: the only `workbench/docs` globs are the top-level
`workbench/docs/*.md` scan at `test_handoff_commands.py:206` and explicit per-file reads). The 18-failure
set is identical to the baseline and to the previous sweep, so nothing observable flipped across the
commit. **This sweep governs `cd980dc` and `77ea629` equally on every production and test path; the two
commits differ only in one untracked-by-tests markdown document.**

## Preflight

| check | result |
|---|---|
| `git rev-parse HEAD` | `cd980dcecbb685b3cd6be4516d6550660a8e89b8` — PASS (matches required revision) |
| `git status --porcelain` | `?? CLEANUP_REPORT.md.save` exactly — PASS |
| `pgrep -af pytest` | empty (exit 1) — PASS |
| `df -h /` | 100 GiB avail ≥ 45 — PASS |
| `/tmp/sweep_basetemp_cd980dc` | absent — PASS |

### Stationary-tree proof (5-minute check)

Verified at **`2026-11-11T09:58:57+07:00`** (4 min 8 s after launch):

```
$ git rev-parse HEAD
cd980dcecbb685b3cd6be4516d6550660a8e89b8
$ git status --porcelain
?? CLEANUP_REPORT.md.save
$ pgrep -c -f 'pytest workbench/tests'
1
```

Tree stationary and HEAD at the required revision through the 5-minute mark (sweep at 12 %).
The next check at `2026-11-11T10:01:30+07:00` found HEAD at `77ea629` (committed 10:00:08); see above —
docs-only, production tree identical.

## Command (verbatim)

```
mkdir -p /tmp/sweep_basetemp_cd980dc
setsid nohup /home/heheboiz/data/cir/workbench/envs/ctl312h/bin/python -m pytest workbench/tests \
  -q -p no:cacheprovider -rfE --tb=short --basetemp=/tmp/sweep_basetemp_cd980dc \
  > /tmp/sweep_cd980dc.txt 2>&1 < /dev/null &
```

Serial, no `-n`. Log: `/tmp/sweep_cd980dc.txt` (this brief permits writing only this one markdown file, so
the raw log is not copied into the repo; it is 30 GB of basetemp plus a ~50 KB log and is deleted at the
end of the run — the summary line and greps below are the record).

## Summary line (verbatim, last line of the log)

```
18 failed, 2901 passed, 47 skipped, 350 warnings in 3486.46s (0:58:06)
```

## The four accounting checks

### 1. Counts
`18 failed, 2901 passed, 47 skipped` (above).

### 2. Error greps — both 0

| measure | value |
|---|---|
| `grep -c '^ERROR ' /tmp/sweep_cd980dc.txt` | **0** — PASS (no collection/setup errors) |
| `grep -c 'Errno 28' /tmp/sweep_cd980dc.txt` | **0** — PASS (no disk-full) |
| `grep -c '^FAILED' /tmp/sweep_cd980dc.txt` | **18** |

**Method correction.** The previous sweep used `grep -cE '^E  |^ERROR'` and reported 325. That pattern is
wrong for a `--tb=short` log: every traceback body line begins with `E `, so the count measures traceback
volume, not errors. Measured both ways on this log to show the difference:

```
$ grep -cE '^E  |^ERROR' /tmp/sweep_cd980dc.txt            → 325
$ awk '/^=+ FAILURES/{f=1} f&&/^E  /{c++} END{print c}'    → 325   (all 325 are after the FAILURES header)
$ grep -c '^ERROR ' /tmp/sweep_cd980dc.txt                 → 0
```

All 325 `E ` lines lie inside `=== FAILURES ===` traceback bodies; 0 pathology lines precede that header.
The 18 real failures collapse into 9 printed sections (pytest merges identical parametrised tracebacks),
plus five large multi-line `release_verify.sh` stdout dumps. The `^E ` grep is therefore abandoned: the
meaningful assertions are collection errors **0**, `Errno 28` **0**, `^FAILED` **18**.

### 3. Accounting gap = 0

```
$ workbench/envs/ctl312h/bin/python -m pytest workbench/tests -q --collect-only -p no:cacheprovider 2>&1 | tail -3
2966 tests collected in 3.54s
```

| N collected | passed | failed | skipped | sum | gap |
|---|---|---|---|---|---|
| 2966 | 2901 | 18 | 47 | 2966 | **0** |

Every collected test reported a terminal outcome — nothing was collected-and-vanished, and no session
teardown truncated the run (`^ERROR` = 0 confirms no interrupted-module bookkeeping).

### 4. Failure diff vs baseline `68ad75e`

`failure_diff.py` takes no argv and rewrites a committed evidence file, so its comparison was reimplemented
identically on the raw logs (same rule as the script: every line starting `FAILED `, node id taken whole up
to the first ` - `, sorted, set-differenced):

```
grep '^FAILED' <log> | sed 's/^FAILED //; s/ - .*//' | sort
```

Baseline: `workbench/docs/reviews/20261110-e2e-repair/full_suite_68ad75e.txt` (18 `FAILED` lines).

| NEW | DROPPED | COMMON |
|---|---|---|
| **0** | **0** | **18** |

**NEW list: (empty).**
**DROPPED list: (empty).**

## Per-failure classification

| # | test | assertion in ≤10 words | class |
|---|---|---|---|
| 1 | `test_circus_adapters_blip.py::test_resident_worker_reuse_and_cleanup[model_conesep-conesep]` | `AttributeError: 'Adapter' object has no attribute 'batch_size'` | PRE-EXISTING · ENVIRONMENT |
| 2 | `…::test_resident_worker_reuse_and_cleanup[model_ptha_mtst-ptha_mtst]` | same `batch_size` AttributeError | PRE-EXISTING · ENVIRONMENT |
| 3 | `…::test_resident_worker_failure_uses_shared_error_and_stays_failed[conesep]` | same `batch_size` AttributeError | PRE-EXISTING · ENVIRONMENT |
| 4 | `…::test_resident_worker_failure_uses_shared_error_and_stays_failed[ptha_mtst]` | same `batch_size` AttributeError | PRE-EXISTING · ENVIRONMENT |
| 5 | `test_circus_cli.py::test_cli_3d_full_gallery_resume_and_gallery_invalidation` | `assert 'REUSED' == 'COMPLETED'` | PRE-EXISTING · FLAKY |
| 6 | `test_circus_tme_lifecycle.py::test_encoder_and_limn_failed_child_latch_and_preserve_log[encoder]` | `WORKBENCH_ENCODER_PYTHON=None (not set)` | PRE-EXISTING · ENVIRONMENT |
| 7 | `…::test_…latch_and_preserve_log[limn]` | `WORKBENCH_LIMN_PYTHON=None (not set)` | PRE-EXISTING · ENVIRONMENT |
| 8 | `test_circus_tme_precision.py::test_worker_skips_empty_stdin_lines` | `FileNotFoundError: …/output.npy` (worker died on stub `torch`) | PRE-EXISTING |
| 9 | `test_f05_f08_f09_verification.py::test_f08_dcnet_config_data_dir_resolution` | `data/FashionIQ` ≠ `DCNet/dataset/fashioniq` | PRE-EXISTING |
| 10 | `test_handoff_commands.py::test_no_release_document_points_at_a_superseded_source_sha` | `TESTED_SOURCE_SHA be14393c… does not contain production code` | PRE-EXISTING (by-design red guard) |
| 11 | `test_release_blocker_gpu_records.py::test_defect_b_g_with_no_target_gpu_verdict_is_pending_not_pass` | `RELEASE_BLOCKED`; J10 `built bundle lacks audit-threshold copy` | PRE-EXISTING |
| 12 | `…::test_release_verify_documented_invocations_produce_parseable_report[<lambda>0]` | invocation non-zero; J10 FAIL | PRE-EXISTING |
| 13 | `…[<lambda>1]` | same | PRE-EXISTING |
| 14 | `…[<lambda>2]` | same | PRE-EXISTING |
| 15 | `…[<lambda>3]` | same | PRE-EXISTING |
| 16 | `…[<lambda>4]` | same | PRE-EXISTING |
| 17 | `…::test_offline_verdict_names_deployment_reason` | `RELEASE_BLOCKED`; J10 FAIL | PRE-EXISTING |
| 18 | `test_standalone_operator_commands.py::test_prepare_model_executes_only_declared_recipe_steps` | dcnet `[MANUAL]` resized-images evidence missing | PRE-EXISTING |

### Tally

| class | count |
|---|---|
| PRE-EXISTING (all 18) | **18** |
| ├ ENVIRONMENT (`batch_size` ×4, unset `WORKBENCH_*_PYTHON` ×2) | 6 |
| ├ proven FLAKY | 1 (#5) |
| NEW-from-this-batch | **0** |
| Pure FLAKY (absent from baseline) | 0 |
| DROPPED | 0 |

### Isolation evidence for the single FLAKY candidate

| run | basetemp | result |
|---|---|---|
| `test_cli_3d_full_gallery_resume_and_gallery_invalidation` alone | `/tmp/flaky_final_1` | **1 passed in 0.20s** |
| `test_circus_cli.py` whole file (21 tests) | `/tmp/flaky_final_file` | **21 passed in 1.44s** |

It fails only inside the full serial suite (`'REUSED' == 'COMPLETED'`), passes alone and passes with its
own file — cross-module state, not a defect in this batch. Same single flaky test as the `a24635e` run.

`test_worker_skips_empty_stdin_lines` and the dcnet evidence pair remain deterministic red
(re-confirmed failing in isolation in `67`); `test_no_release_document_points_at_a_superseded_source_sha`
fails on the same `be14393c…` stale sha as `67` because `77ea629` adds a doc but no production change —
the guard compares the production tree, so a docs-only successor correctly stays red for the same reason
it was red at `a24635e`.

## Comparison against the `a24635e` run (`67-regression-sweep.md`)

| measure | `a24635e` run | this run (`cd980dc` → `77ea629`) | delta |
|---|---|---|---|
| failed | 18 | 18 | 0 |
| passed | 2901 | 2901 | 0 |
| skipped | 47 | 47 | 0 |
| collected | 2966 | 2966 | 0 |
| accounting gap | 0 | 0 | 0 |
| warnings | 350 | 350 | 0 |
| NEW / DROPPED vs `68ad75e` | 0 / 0 | 0 / 0 | 0 |
| COMMON | 18 | 18 | 0 |
| `^ERROR ` | 0 | 0 | 0 |
| `Errno 28` | 0 | 0 | 0 |
| wall time | 3346.75 s (0:55:46) | 3486.46 s (0:58:06) | +139.71 s |
| HEAD moved mid-run | yes (`e11f9ff`, **production** files) | yes (`77ea629`, **docs only**) | — |

**Every result number is identical; the only differences are wall time (+140 s, machine noise) and the
nature of the mid-run commit.** The `a24635e` sweep could not govern the final tree because `e11f9ff`
rewrote `workbench/frontend/scope.ts` and `ModelInventory.tsx` — files the release guard does not exclude
and which the frontend tests actually execute. This run's mid-run commit `77ea629` adds one markdown
document outside every test's read set and leaves the production/test tree byte-identical
(`git diff --stat … ':(exclude)workbench/docs' ':(exclude)workbench/tests'` = empty), so the `e11f9ff`
frontend changes have now been measured inside the full serial suite and the sweep is valid at both
`cd980dc` and `77ea629`.

## Conclusion

- **18 failed / 2901 passed / 47 skipped / 2966 collected, gap 0.**
- **0 NEW, 0 DROPPED, 18 COMMON** against the `68ad75e` baseline; identical to the `a24635e` run.
- All 18 are pre-existing: 6 environmental, 1 flaky-in-suite-only, 11 long-standing red guards/evidence
  gaps. Nothing regressed and nothing was newly introduced by the eight-defect batch or by `e11f9ff`.
- HEAD advanced from `cd980dc` to `77ea629` 5 min 19 s into the run; that commit is documentation-only and
  the production tree is unchanged, so this sweep is authoritative for both revisions. The stationary
  check at 5 minutes (`09:58:57`) confirmed the sweep started on a clean `cd980dc` tree.
