# 83 — Authoritative serial sweep result (`96d86010`) — COMPLETED

Status: **COMPLETED — usable for pinning.**

This is the post-run checklist for the sweep started in `82-sweep-stationary.md`. No source file
was edited, `dist/` was not rebuilt, and the web service was not restarted or killed.

## Identity

| Field | Value |
|---|---|
| Branch | `research/cir-failure-workbench` |
| Start revision (full) | `96d86010e85048d82f2dacf26e17d7b7206d31ab` |
| HEAD at start | `96d86010e85048d82f2dacf26e17d7b7206d31ab` |
| HEAD at finish | `d32dbb2efb5ebb2db6e499f999314383b137d60e` |
| Start timestamp | `2026-11-11T19:38:19+07:00` |
| Finish timestamp | `2026-11-11T20:37:29+07:00` |
| Wall time | `3550.44s` (0:59:10) |
| Sweep PID | `1046852` |
| Log path | `/tmp/sweep_96d86010.txt` (kept as evidence) |
| Basetemp | `/tmp/sweep_basetemp_96d86010` (31G, removed after run) |
| Bundle | `workbench/frontend/dist/assets/index-DYYSN7oG.js` |

Command actually run:

```
workbench/envs/ctl312h/bin/python -m pytest workbench/tests -q -p no:cacheprovider \
  -rfE --tb=short --basetemp=/tmp/sweep_basetemp_96d86010
```

## 1. Verbatim summary line

```
18 failed, 2939 passed, 47 skipped, 350 warnings in 3550.44s (0:59:10)
```

## 2. `grep -c '^ERROR ' /tmp/sweep_96d86010.txt`

```
0
```

PASS — required 0.

## 3. `grep -c "Errno 28" /tmp/sweep_96d86010.txt`

```
0
```

PASS — required 0. No disk exhaustion during the run.

## 4. Accounting

| Quantity | Value |
|---|---|
| passed | 2939 |
| failed | 18 |
| skipped | 47 |
| `passed+failed+skipped` | **3004** |
| `collected` (`--collect-only`) | **3004** |
| **Gap** | **0** |

Raw collector tail:

```
3004 tests collected in 3.71s
```

PASS — gap 0.

## 5. Failure-set diff vs baseline `full_suite_68ad75e.txt`

Baseline: `workbench/docs/reviews/20261110-e2e-repair/full_suite_68ad75e.txt` (18 `FAILED` lines).
Sweep: `/tmp/sweep_96d86010.txt` (18 `FAILED` lines).

| Set | Count |
|---|---|
| NEW (in sweep, not baseline) | **0** |
| DROPPED (in baseline, not sweep) | **0** |
| COMMON | **18** |

**NEW ids: none.**

Methodology note: the comparison was performed on the normalized pytest node id — each line
stripped of `^FAILED ` and of the ` - <reason>` tail. This is required because two of the
baseline failures embed per-run text in their reason, so a byte-wise `comm` falsely reports them
as both NEW and DROPPED:

- `test_circus_tme_precision.py::test_worker_skips_empty_stdin_lines` — reason embeds the basetemp
  path (`/tmp/sweep_basetemp_68ad75e` vs `/tmp/sweep_basetemp_96d86010`).
- `test_handoff_commands.py::test_no_release_document_points_at_a_superseded_source_sha` — reason
  embeds the recorded `TESTED_SOURCE_SHA` (`36ae9e21…` vs `0691f3e8…`), i.e. the docs commits
  advanced the recorded SHA between the two runs. The failure itself is the same defect.

Both are pre-existing baseline failures; neither is a new regression. On node ids the two sets
are **identical** (18 = 18, symmetric difference empty).

Baseline failure set (all 18 also failed in this sweep):

1. `test_circus_adapters_blip.py::test_resident_worker_reuse_and_cleanup[workbench.composition.model_conesep-conesep]`
2. `test_circus_adapters_blip.py::test_resident_worker_reuse_and_cleanup[workbench.composition.model_ptha_mtst-ptha_mtst]`
3. `test_circus_adapters_blip.py::test_resident_worker_failure_uses_shared_error_and_stays_failed[workbench.composition.model_conesep-conesep]`
4. `test_circus_adapters_blip.py::test_resident_worker_failure_uses_shared_error_and_stays_failed[workbench.composition.model_ptha_mtst-ptha_mtst]`
5. `test_circus_cli.py::test_cli_3d_full_gallery_resume_and_gallery_invalidation`
6. `test_circus_tme_lifecycle.py::test_encoder_and_limn_failed_child_latch_and_preserve_log[encoder]`
7. `test_circus_tme_lifecycle.py::test_encoder_and_limn_failed_child_latch_and_preserve_log[limn]`
8. `test_circus_tme_precision.py::test_worker_skips_empty_stdin_lines`
9. `test_f05_f08_f09_verification.py::test_f08_dcnet_config_data_dir_resolution`
10. `test_handoff_commands.py::test_no_release_document_points_at_a_superseded_source_sha`
11. `test_release_blocker_gpu_records.py::test_defect_b_g_with_no_target_gpu_verdict_is_pending_not_pass`
12. `test_release_blocker_gpu_records.py::test_release_verify_documented_invocations_produce_parseable_report[<lambda>0]`
13. `test_release_blocker_gpu_records.py::test_release_verify_documented_invocations_produce_parseable_report[<lambda>1]`
14. `test_release_blocker_gpu_records.py::test_release_verify_documented_invocations_produce_parseable_report[<lambda>2]`
15. `test_release_blocker_gpu_records.py::test_release_verify_documented_invocations_produce_parseable_report[<lambda>3]`
16. `test_release_blocker_gpu_records.py::test_release_verify_documented_invocations_produce_parseable_report[<lambda>4]`
17. `test_release_blocker_gpu_records.py::test_offline_verdict_names_deployment_reason`
18. `test_standalone_operator_commands.py::test_prepare_model_executes_only_declared_recipe_steps`

## 5b. The two `F` marks at 33% and 43% — accounted for

The progress lines carrying `F` marks, with the ordinal of the failure each mark corresponds to
in `-rfE` output order (18 `F` marks in progress lines total, matching 18 failures — no extra
`F` is hiding anywhere):

| Log line | Progress | Failures # | Baseline failure |
|---|---|---|---|
| 6 | 14% | #1–#5 | `test_circus_adapters_blip` ×4, `test_circus_cli` |
| 7 | 16% | #6–#8 | `test_circus_tme_lifecycle` ×2, `test_circus_tme_precision` |
| **14** | **33%** | **#9** | **`test_f05_f08_f09_verification.py::test_f08_dcnet_config_data_dir_resolution`** |
| **18** | **43%** | **#10** | **`test_handoff_commands.py::test_no_release_document_points_at_a_superseded_source_sha`** |
| 35 | 83% | #11–#17 | `test_release_blocker_gpu_records` ×7 |
| 40 | 95% | #18 | `test_standalone_operator_commands.py::test_prepare_model_executes_only_declared_recipe_steps` |

Verbatim progress lines:

```
...FFFF.............F................................................... [ 14%]
......................................FF...F............................ [ 16%]
............................s..............F............................ [ 33%]
........................F............................................... [ 43%]
```

**Both marks are accounted for by the baseline diff. Neither is NEW.** The 33% mark is failure
#9 (`test_f08_dcnet_config_data_dir_resolution`), present in the baseline. The 43% mark is failure
#10 (`test_no_release_document_points_at_a_superseded_source_sha`), present in the baseline. The
run is clean on this axis.

## 6. Stationary proof

```
$ git rev-parse HEAD
d32dbb2efb5ebb2db6e499f999314383b137d60e

$ git diff --stat 96d86010e85048d82f2dacf26e17d7b7206d31ab -- . \
    ':(exclude)workbench/docs' ':(exclude)workbench/tests'
(empty)
```

PASS — production diff empty.

HEAD did move off the start revision (`96d86010` → `d32dbb2e`), by 6 commits landed while the
sweep ran. Every one is **docs-only**, as proven independently of the pathspec:

```
$ git diff --name-only 96d86010..HEAD | grep -v '^workbench/docs/'
NONE_OUTSIDE_DOCS
```

The 6 commits:

```
d32dbb2 docs(audit): A5 verified 21/21 on served bundle; withdraw subset#4 misreading
a62b7fc docs(audit): close DERIVED marker and :8765, keep bundle chain coherent
3157d0c docs(e2e): recheck on bundle index-DYYSN7oG (74e4b4f8); 21/21 PASS
d955fb3 docs(audit): 82 - tolerate docs-only commits in stationary re-proof
a43ca5c docs(audit): c25d67f2 was killed at 31%, and record the stationary sweep that replaced it
5455f0e docs(audit): start authoritative serial sweep on stationary tree at 96d86010
```

All 21 changed paths are under `workbench/docs/` (reports, `raw/verify/`, `web_evidence/e2e_recheck/`
PNGs and `recheck_probe.py`). No `workbench/` source, no `dist/`, no `workbench/tests/` change.
The tree under test was therefore stationary for the whole 59m10s.

## 7. Bundle unchanged

```
74e4b4f86ce8b4f5061456138ce1ff79aa084d7662c54c24a123e5bb679d651f  workbench/frontend/dist/assets/index-DYYSN7oG.js
```

PASS — identical to the start value
`74e4b4f86ce8b4f5061456138ce1ff79aa084d7662c54c24a123e5bb679d651f`. Single JS asset, unchanged.

## 8. Cleanup

```
$ du -sh /tmp/sweep_basetemp_96d86010
31G     /tmp/sweep_basetemp_96d86010
$ rm -rf /tmp/sweep_basetemp_96d86010
$ df -h /
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme0n1p5  630G  499G   99G  84% /
```

Free space after cleanup: **99G available (84% used)**. The log `/tmp/sweep_96d86010.txt`
(85560 bytes) is retained as evidence.

## Verdict

**USABLE FOR PINNING.**

All five pinning conditions hold:

| Condition | Required | Actual | Result |
|---|---|---|---|
| `^ERROR ` count | 0 | 0 | PASS |
| `Errno 28` count | 0 | 0 | PASS |
| accounting gap | 0 | 0 (3004 = 3004) | PASS |
| production diff | empty | empty | PASS |
| bundle unchanged | unchanged | unchanged | PASS |

Failure set is identical to the baseline: 0 NEW, 0 DROPPED, 18 COMMON. The two `F` marks at 33%
and 43% are failures #9 and #10 and are both accounted for by the baseline diff — **not NEW**.
`18 failed, 2939 passed, 47 skipped in 3550.44s` is the reference result for revision
`96d86010e85048d82f2dacf26e17d7b7206d31ab`.
