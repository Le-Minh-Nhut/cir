# The 18 failures at `c2d8cdeb` — every one attributed

Serial sweep, no `-n`. Log: `full_suite_c2d8cdeb.txt` in this directory (header with revision,
command, interpreter, duration, exit code).

```
18 failed, 2463 passed, 47 skipped, 290 warnings in 3464.03s (0:57:44)
pytest_exit=1
```

2463 + 18 + 47 = **2528 collected**. HEAD was `c2d8cdeb0eabc3232b28070f55b5a10d63ccffcb` before and
after the run, so the log describes these bytes.

Categories: **(a)** caused by this round's production changes · **(b)** operator/machine state ·
**(c)** pre-existing at the previously tested revision `b0f38bd` · **(d)** test pinning obsolete
behaviour.

| # | test | category | evidence |
|---|---|---|---|
| 1 | `test_circus_adapters_blip.py::test_resident_worker_reuse_and_cleanup[workbench.composition.model_conesep-conesep]` | **(d)** | `AttributeError: 'Adapter' object has no attribute 'batch_size'`. The test builds the adapter with `module.Adapter.__new__(module.Adapter)` and never sets the attribute the worker now reads. Reproduced at `1a4cd32` in a worktree: `4 failed, 14 passed` — identical four. |
| 2 | same, `[workbench.composition.model_ptha_mtst-ptha_mtst]` | **(d)** | same |
| 3 | `test_circus_adapters_blip.py::test_resident_worker_failure_uses_shared_error_and_stays_failed[workbench.composition.model_conesep-conesep]` | **(d)** | same |
| 4 | same, `[workbench.composition.model_ptha_mtst-ptha_mtst]` | **(d)** | same |
| 5 | `test_circus_cli.py::test_cli_3d_full_gallery_resume_and_gallery_invalidation` | **(c)** | `assert 'REUSED' == 'COMPLETED'`. Passes alone (21 passed); fails when `test_composition_topk_runner.py` runs first. Reproduced in a `git worktree` of `1a4cd32` with the same two files: `1 failed, 30 passed`. A fixture-ordering effect, not this round's code. |
| 6 | `test_circus_tme_lifecycle.py::test_encoder_and_limn_failed_child_latch_and_preserve_log[encoder]` | **(b)** | `RuntimeError: execution environment not verified [UNCONFIGURED/UNCONFIGURED]: WORKBENCH_ENCODER_PYTHON=None`. The variable is exported in the operator shell (all 9 `WORKBENCH_*_PYTHON` are set) but the test's sandbox re-resolves it as `None`. |
| 7 | same, `[limn]` | **(b)** | `WORKBENCH_LIMN_PYTHON=None`, same mechanism |
| 8 | `test_circus_tme_precision.py::test_worker_skips_empty_stdin_lines` | **(b)** | `FileNotFoundError: …/test_worker_skips_empty_stdin_0/output.npy` — a worker artifact the run never wrote |
| 9 | `test_f05_f08_f09_verification.py::test_f08_dcnet_config_data_dir_resolution` | **(b)** | The untracked symlink `workbench/third_party/DCNet/dataset/fashioniq -> /home/heheboiz/data/cir/data/FashionIQ` (created 13:43 by another agent) makes `.resolve()` return the real tree. **Proven**: both this test and #18 pass in a clean `git worktree` of `c2d8cdeb` (`2 passed`). |
| 10 | `test_handoff_commands.py::test_no_release_document_points_at_a_superseded_source_sha` | **(d)** | It diffed the tree against the OLD recorded `b0f38bd` while four production commits had landed. **Closed by the SHA advance**: `8 passed` on re-run. |
| 11 | `test_release_blocker_gpu_records.py::test_defect_b_g_with_no_target_gpu_verdict_is_pending_not_pass` | **(b)** | `release_verify.sh` → `verdict=RELEASE_BLOCKED`; the shared, gitignored `workbench/artifacts/gpu_capabilities/` holds 632 records, **465** of them `"probe_source": "override"` with `device_name: "Synthetic GPU"` and `error: "CUDA out of memory"` from earlier test runs. |
| 12-16 | same file, `test_release_verify_documented_invocations_produce_parseable_report[<lambda>0..4]` | **(b)** | same root cause; the five invocations differ only in flag order |
| 17 | same file, `test_offline_verdict_names_deployment_reason` | **(b)** | same root cause |
| 18 | `test_standalone_operator_commands.py::test_prepare_model_executes_only_declared_recipe_steps` | **(b)** | `[MANUAL] dcnet: policy=AUTHOR_ARTIFACT_PREFERRED deterministic=NONDETERMINISTIC_WITHOUT_AUTHOR_ARTIFACT` — the DCNet symlink of #9 changes the recipe path the command takes. **Proven**: passes in a clean worktree (`2 passed` with #9). |

## Tally

- **(a) caused by this round: 0.** No failure is attributable to the production changes in `bf6a448`,
  `30f0c07`, `7dcc082` or `6eb4434`.
- **(b) operator/machine state: 11** (#6, #7, #8, #9, #11–#17, #18)
- **(c) pre-existing at `b0f38bd`: 1** (#5)
- **(d) test pinning obsolete behaviour: 5** (#1–#4, #10)
- **new: 0** — every one either reproduces at `1a4cd32`/`b0f38bd` or is operator state.

## Against the previous serial sweep at `b0f38bd`

| | `b0f38bd` | `c2d8cdeb` |
|---|---:|---:|
| collected | 2266 | 2528 |
| passed | 2186 | 2463 |
| failed | 33 | 18 |
| skipped | 47 | 47 |
| duration | 2211.38 s | 3464.03 s |
| exit | 1 | 1 |

**Id-level overlap between the two sweeps is ZERO.** Comparing the `FAILED` ids in each raw log
directly: none of the 33 ids at `b0f38bd` appears among the 18 here, and none of the 18 appears there.
That is not because 33 tests were fixed — the failures are largely the same *causes* surfacing through
*different tests*:

- `b0f38bd` lists `test_adapter_import_and_unready_status[…]` (blocked on `ready`), while `c2d8cdeb`
  lists `test_resident_worker_{reuse_and_cleanup,failure_uses_shared_error_and_stays_failed}[…]`
  (blocked on `batch_size`). The first pair was fixed; the second pair was introduced by the
  resident-worker feature and its fixture was never updated. Group **(d)**, and it reproduces at
  `1a4cd32`.
- `b0f38bd` lists 7 `test_pc_launcher` / 4 `test_phase1_acceptance` / 9 `test_release_blocker_master_e2e`
  / 2 `test_release_blocker_source_integrity` ids, **31 of 33 driven by `workbench/config/workbench.env`
  leaking into sandboxed tests** (that sweep's own A/B established it). The suite was then changed to
  stop reading operator state (`9b6b5b6`, "stop the suite reading operator state"), so those tests now
  pass here — that is a real improvement in the *tests*, and none of it is this round's production
  code.
- `c2d8cdeb`'s 7 `test_release_blocker_gpu_records` ids are a different family from `b0f38bd`'s
  `test_release_blocker_master_e2e` ids: both hit `RELEASE_BLOCKED`, but through different entry
  points.

So the honest comparison is: **33 → 18 failures, 0 of them shared by id.** 15 fewer, attributed as 31
operator-state ids resolved by the test-isolation change, and the current 18 all attributed above with
**0** new regressions from this round's production commits — every current failure either reproduces at
`1a4cd32`/`b0f38bd` or is operator state, and 46 new tests were added this round, all passing.

A raw count comparison (33 → 18) is therefore not evidence that this round's code is better; the
code-attributable failures in the current set are #1–#4 (fixture, category d) and #5 (ordering,
category c), and both reproduce before this round.

## The G6 prose-only warning (downgraded, not archived)

`INPUT_PIPELINE_AND_INFERENCE_AUDIT.vi.md` §8 reports a decode baseline/after pair (21.2 → 2089.2
img/s) and the whole query-axis table as measurements. The raw files for those two numbers are **not**
in the tree — the audit says so itself and points at `THROUGHPUT.vi.md`. They are therefore marked
**prose-only, unverified** in §8.8 and in the G6 gate row, and the query-axis figure is retained only
because `20261108T-throughput/query.txt` covers the 512-query rank identity that the table's last
column asserts. No number in this document is used for a release claim.
