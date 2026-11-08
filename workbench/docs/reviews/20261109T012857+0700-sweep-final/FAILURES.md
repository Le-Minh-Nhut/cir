# The 18 failures at `80842661` — attributed, compared against the 18 at `1129c58`

Serial sweep, no `-n`. Log: `full_suite_80842661.txt` in this directory (header with revision,
command, interpreter, duration, exit code).

```
18 failed, 2480 passed, 47 skipped, 290 warnings in 3484.30s (0:58:04)
pytest_exit=1
```

2480 + 18 + 47 = **2545 collected**. `--collect-only` reports 2545. HEAD was
`80842661e672736c3e2cf4bb4c49a9473db708f8` before and after the run and `git status --short` was
empty, so the log describes these bytes.

## Headline: no new failure, none fixed

| | `1129c58` | `80842661` |
|---|---:|---:|
| collected | 2532 | **2545** |
| passed | 2467 | **2480** |
| failed | 18 | **18** |
| skipped | 47 | 47 |
| duration | 3483.79 s | 3484.30 s |
| exit | 1 | 1 |

Compared **by id, not by count**: `comm` on the two sorted id lists reports **no id present only in the
new run and no id present only in the old run**. The failed set is byte-identical at the id level, and
17 of the 18 `FAILED` lines carry an error string byte-identical to the previous sweep. The one
exception is #10, whose message names a different SHA because the recorded SHA moved — same defect,
same assertion, later revision.

The **+13 collected / +13 passed** are `workbench/tests/test_frontend_layout_provenance.py`, added by
`80842661` (270 lines, 13 `def test_`). All 13 pass. That is the only test file the commit touches;
`80842661`'s production change is `workbench/frontend/**`, which the release guard counts as
production, which is why a new sweep was required at all.

**Therefore: 0 failures are caused by the frontend commit.** Nothing in `test_circus_*`,
`test_f05_f08_f09`, `test_release_blocker_gpu_records`, `test_standalone_operator_commands` or
`test_handoff_commands` touches frontend code; the only frontend-facing tests added by this commit all
pass.

## Categories (unchanged from the `1129c58` sweep)

- **(a)** caused by a production or test change **inside `b0f38bd..HEAD`**
- **(b)** environment/host state (a real machine condition, not a test's own construction)
- **(c)** pre-existing at `b0f38bd`
- **(d)** test pins obsolete behaviour / release-record lag
- **(h)** harness/test defect

Attribution is carried over from `20261108T212500+0700-sweep/FAILURES.md`, which was itself re-verified
row by row after an independent agent disproved three of its first-version rows. Nothing in
`1129c58..80842661` touches any of the code paths named below, so no category can have moved. The
evidence that the categories still hold on this machine is stated per row.

| # | test | category | new? | status vs previous 18 |
|---|---|---|---|---|
| 1 | `test_circus_adapters_blip.py::test_resident_worker_reuse_and_cleanup[workbench.composition.model_conesep-conesep]` | **(a)** in `aa0f530` | no | same id, same `AttributeError: 'Adapter' object has no attribute 'batch_size'` |
| 2 | same, `[workbench.composition.model_ptha_mtst-ptha_mtst]` | **(a)** in `aa0f530` | no | same id, same error |
| 3 | `test_circus_adapters_blip.py::test_resident_worker_failure_uses_shared_error_and_stays_failed[workbench.composition.model_conesep-conesep]` | **(a)** in `aa0f530` | no | same id, same error |
| 4 | same, `[workbench.composition.model_ptha_mtst-ptha_mtst]` | **(a)** in `aa0f530` | no | same id, same error |
| 5 | `test_circus_cli.py::test_cli_3d_full_gallery_resume_and_gallery_invalidation` | **(a)** — ordering interaction, introduced in `9b6b5b6` | no | same id, same `assert 'REUSED' == 'COMPLETED'`; still a test-ordering interaction with `test_composition_topk_runner.py` |
| 6 | `test_circus_tme_lifecycle.py::test_encoder_and_limn_failed_child_latch_and_preserve_log[encoder]` | **(a)** — unsatisfiable by construction, inside the range | no | same id, same `RuntimeError: execution environment not verified [UNCONFIGURED/UNCONFIGURED]: WORKBENCH_ENCODER_PYTHON=None` |
| 7 | same, `[limn]` | **(a)** | no | same id, same message with `WORKBENCH_LIMN_PYTHON=None` |
| 8 | `test_circus_tme_precision.py::test_worker_skips_empty_stdin_lines` | **(a)** in `aa0f530` | no | same id; `FileNotFoundError: .../output.npy`, which the previous sweep traced to the worker dying on `module 'torch' has no attribute 'set_num_threads'` |
| 9 | `test_f05_f08_f09_verification.py::test_f08_dcnet_config_data_dir_resolution` | **(b)** host state | no | same id. Cause still on disk: untracked symlink `workbench/third_party/DCNet/dataset/fashioniq -> /home/heheboiz/data/cir/data/FashionIQ`, created 2026-11-08 13:43 by another agent; `.resolve()` therefore returns the real tree |
| 10 | `test_handoff_commands.py::test_no_release_document_points_at_a_superseded_source_sha` | **(d)** release-record lag — **closed by this task** | no | same id, same assertion; only the SHA in the message changed (`c2d8cdeb` → `1129c58`). Fixed by advancing `TESTED_SOURCE_SHA` to `80842661`; re-run is `8 passed` |
| 11 | `test_release_blocker_gpu_records.py::test_defect_b_g_with_no_target_gpu_verdict_is_pending_not_pass` | **(b)** host state | no | same id, same `== J1 clean startup` / `RELEASE_BLOCKED` family |
| 12 | same file, `test_release_verify_documented_invocations_produce_parseable_report[<lambda>0]` | **(b)** | no | same id |
| 13 | same, `[<lambda>1]` | **(b)** | no | same id |
| 14 | same, `[<lambda>2]` | **(b)** | no | same id |
| 15 | same, `[<lambda>3]` | **(b)** | no | same id |
| 16 | same, `[<lambda>4]` | **(b)** | no | same id |
| 17 | same file, `test_offline_verdict_names_deployment_reason` | **(b)** | no | same id |
| 18 | `test_standalone_operator_commands.py::test_prepare_model_executes_only_declared_recipe_steps` | **(b)** | no | same id, same `[MANUAL] dcnet: policy=AUTHOR_ARTIFACT_PREFERRED deterministic=NONDETERMINISTIC_WITHOUT_AUTHOR_ARTIFACT`, driven by the same #9 symlink |

### Tally

| category | n | ids |
|---|---:|---|
| **(a)** caused inside `b0f38bd..HEAD` | **8** | #1–#4, #5, #6, #7, #8 |
| **(h)** harness/test defect (reclassified into (a)) | **0** | — |
| **(b)** host state | **9** | #9, #11–#17, #18 |
| **(c)** pre-existing at `b0f38bd` | **0** | — |
| **(d)** release-record lag, closed by this task | **1** | #10 |
| **total** | **18** | |

8 + 9 + 0 + 1 = 18, matching both the table's row count and the log's failure count.

### New versus known

- **New ids: 0.**
- **Ids that disappeared: 0.**
- **Ids whose category changed: 0.**
- **Ids closed since the previous sweep: 1** (#10), and it is closed by *this* task's SHA advance, not
  by a code change.

### Host state observed during this run (why (b) still holds)

- `workbench/artifacts/gpu_capabilities/` is still a polluted shared directory: **664** record files
  now (the previous sweep counted **632**; it grew during this run, since the tests themselves write
  probe records). It is gitignored (`workbench/artifacts/**`, `.gitignore:85`). This is the root cause
  of #11–#17.
- `workbench/config/workbench.env` is present and untracked (`git status --short --ignored` →
  `!! workbench/config/workbench.env`). It was **not modified** by this task.
- The DCNet symlink behind #9 and #18 is present, unchanged.

### What would settle the (b) rows definitively

Moving `workbench/artifacts/gpu_capabilities/` aside and removing the DCNet symlink, then re-running
#9 and #11–#18, would confirm both causes. **Not done here**: this pass is docs-and-sweep only, the
sweep is the deliverable, and the directory belongs to another agent's run. The previous sweep's
`FAILURES.md` recorded the same deferral.

## Against the two earlier serial sweeps

| | `b0f38bd` | `c2d8cdeb` | `1129c58` | `80842661` |
|---|---:|---:|---:|---:|
| collected | 2266 | 2528 | 2532 | **2545** |
| passed | 2186 | 2463 | 2467 | **2480** |
| failed | 33 | 18 | 18 | **18** |
| skipped | 47 | 47 | 47 | 47 |
| duration | 2211.38 s | 3464.03 s | 3483.79 s | 3484.30 s |
| exit | 1 | 1 | 1 | 1 |

The failure count has now been 18 for three consecutive serial sweeps, with the same 18 ids, while
collection grew 2528 → 2532 → 2545. Both increments were pure test additions (+4
`test_frontend_scoring_impl.py`, +13 `test_frontend_layout_provenance.py`), and every added id passes.
**A flat failure count across the frontend commit means the commit neither introduced nor fixed a
failure** — it did not touch the code paths the 18 failures live in.

## Sweep integrity note

The run was launched detached under `setsid` from `/tmp/run_sweep.sh`. Two defects in that wrapper are
recorded here so the log's provenance is not overstated:

1. **Header bug.** The wrapper's `echo "| revision | \`SHA\` |"` lines used unescaped backticks, so bash
   ran the SHA as a command. `/tmp/sweep_driver.log` shows
   `80842661e672736c3e2cf4bb4c49a9473db708f8: command not found` and
   `research/cir-failure-workbench: No such file or directory`, leaving `| revision |  |` and
   `| branch |  |` blank. The header in `full_suite_80842661.txt` in this directory was therefore
   **written by hand from `git rev-parse` and the raw log**, not by the wrapper.
2. **Duplicate second run.** After the first run finished, the same wrapper started a **second** pytest
   at 02:27 with its stdout on an already-unlinked inode (`/proc/<pid>/fd/1 -> /tmp/#39630065
   (deleted)`). It wrote no readable log. It was terminated so that it could not overwrite the evidence
   file or compete for `/tmp/pytest-of-heheboiz` with any later run.

The evidence file's raw body is byte-identical (md5 `378ee7f24d33acbe0d7321a4ec822d0d`, 45935 bytes) to the
completed first run's stdout, which was copied out before the duplicate could interfere. Only one pytest
temp root appears in the whole log (`pytest-of-heheboiz/pytest-507`), confirming it is a single
uninterrupted run; it contains 36 progress markers reaching `[100%]` and exactly one summary line.

Neither defect is in the repository. Neither is fixed here. The wrapper lives at `/tmp/run_sweep.sh`
and is scratch.
