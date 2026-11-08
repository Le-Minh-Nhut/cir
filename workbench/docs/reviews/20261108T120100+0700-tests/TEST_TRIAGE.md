# TEST_TRIAGE — 33 failures at `b0f38bd`, isolated and root-caused

Scope: every one of the 33 failures in
`workbench/docs/reviews/20261108T015603+0700/tests/full_suite_b0f38bd.txt`
(serial sweep at `b0f38bd`: 33 failed, 2186 passed, 47 skipped = 2266 collected, exit 1).

Work tree during this investigation: branch `research/cir-failure-workbench`, `HEAD = 6421188`.
**The tree was never clean.** Four sibling agents were editing `workbench/backend/**`,
`workbench/composition/**`, `workbench/frontend/**`, `workbench/artifacts/**` and
`workbench/envs/**` throughout. `git status --porcelain -uall` had 261 entries at sweep time;
the snapshot is `tree_state_before_sweep.txt`. No full-suite number here describes a clean
release tree.

Nothing was committed, pushed, or staged. The operator's untracked
`workbench/config/workbench.env` was temporarily moved aside for measurements and restored
byte-identically (`md5 b0d9fb7260e8032cc4c361d53c9e62d8`, verified after every probe).

---

## 1. Summary

| outcome | count |
|---|---|
| failures caused by this machine's state (operator file / exported env / staged checkpoint) | **30** |
| failure caused by a missing build prerequisite (frontend `dist`) | 0 |
| failure caused by a genuine code defect | 0 |
| failure that is a release-metadata guard, correct by contract | **1** (`#04` at file level, passes individually) |
| failures reproducible individually at `HEAD` | 32 / 33 |
| failures passing individually at `HEAD` | 1 / 33 (`#04`) |

**All 33 now pass** when run as the previously-failing set (`all33_after_fix.txt`: `33 passed`).
Every one of them was environment/machine-state contamination or a release-metadata guard — none
was a real defect in the product code, and none needed a weakened assertion, a deleted test, or a
new skip.

## 2. The actual root cause — one mechanism, three shapes

`operator_config.resolve_config()` reads two sources, in this precedence:

1. exported process environment (`CIR_*`, `WORKBENCH_*`)
2. `workbench/config/workbench.env` — **untracked**, this operator's file

Neither is isolated for most of the suite, so per-machine state decided outcomes in three shapes:

**Shape A — staged checkpoint flips `Adapter.status().ready`.**
The operator file sets
`WORKBENCH_CHECKPOINT_ROOT=workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads`,
where this host has ConeSep / PTHA-MTST / HABIT / INTENT / HINT / TME weights staged. The adapter
tests assert `status.ready is False`, which is the *fresh clone* answer; on this host the
checkpoint is genuinely present, so `ready` is `True` and the assertion fails. The test is not
wrong — it asserts a machine-independent property and was reading machine-dependent input.

**Shape B — absolute `WORKBENCH_ARTIFACTS_ROOT` steals a subprocess's output.**
The same file sets `WORKBENCH_ARTIFACTS_ROOT=/home/heheboiz/data/cir/workbench/artifacts`.
Every test that spawns `pipeline.py` / `evaluate_models.py` / `verify_run.py` passes
`CIR_REPO_ROOT=<sandbox>` so the *default* artifacts root would be `<sandbox>/workbench/artifacts`
— correct — but the child re-reads the operator file, whose absolute value wins. The run then
writes its ledger, reports, completion proof and `pipeline_state.json` lock into the developer's
checkout. Consequences observed:

- the sandbox's `workbench/artifacts/logs/` is never created → `FileNotFoundError` (`#33`, `#17`)
- the completion proof the test asserts on is not in the sandbox → `#21`
- `verify_run.py`, run inside the `pc.sh` Docker shim, looks for this sandbox's reports under the
  real checkout, finds none, and refuses → `pc.sh` exits 1 in a sandbox that staged the run
  correctly (`#05`–`#16`)
- two concurrent cases meet each other's real `pipeline_state.json` lock →
  `another workflow owns .../pipeline_state.json (lock held)` (`#29`, `#30`)
- `test_release_blocker_source_integrity`'s `_recorded_asset_digest` reads
  `<artifacts>/auxiliary/download_manifest.json` from the real checkout, so a digest recorded into
  the tmp sandbox is never consulted → `#31`, `#32`

**Shape C — pollution of a shared, gitignored artifacts root.**
`workbench/artifacts/gpu_capabilities/` on this host holds **504** records, **368** of them
written by tests with `probe_source: override` ("a synthetic test probe, not a real probe").
`release_verify.sh`, invoked from `test_release_blocker_gpu_records.py`, scans that real directory
because `gpu_capability_directory()` follows `WORKBENCH_ARTIFACTS_ROOT`. The 12 post-fix failures
in §6 are this shape; they are **pre-existing and not introduced by this work**.

### Single-key env probe — which variable flips which test

Command shape (each key injected alone into the pytest process; conftest removed, operator file
present or absent as noted):

```bash
env "$KEY=$VALUE" workbench/envs/ctl312h/bin/python -m pytest <files> -q -p no:cacheprovider --tb=no
```

With my `conftest.py` in place nothing flips — that is the point of the fix:

```
BASELINE (nothing injected)                      48 passed
WORKBENCH_CIRCUS_RELEASE_ROOT=...                48 passed
WORKBENCH_THIRD_PARTY_ROOT=...                   48 passed
WORKBENCH_ARTIFACTS_ROOT=...                     48 passed
WORKBENCH_CHECKPOINT_ROOT=...                    48 passed
CIR_DATA_ROOT=...                                48 passed
WORKBENCH_RESULTS_ROOT=...                       48 passed
```
(`single_key_env_probe_after_fix.txt`)

On the **unfixed** tree the file must supply *both* halves — a staged checkpoint *and* a working
interpreter — because `_status()` reports the checkpoint as a reason only after it resolves an
interpreter. That is why injecting `WORKBENCH_CHECKPOINT_ROOT` alone is insufficient:

```
A. full operator file                                    3 failed, 27 passed
B. ONLY WORKBENCH_CHECKPOINT_ROOT                       30 passed
C. ONLY the *_PYTHON interpreter keys                   30 passed
D. CHECKPOINT_ROOT + CONESEP/PTHA_MTST interpreters      2 failed, 28 passed
E. empty operator file                                  30 passed
```
(`operator_file_key_probe.txt`)

The single-key exported-variable probe on the unfixed tree (operator file moved aside):
`WORKBENCH_ARTIFACTS_ROOT`, `WORKBENCH_CHECKPOINT_ROOT`, `CIR_DATA_ROOT` each injected alone flip
**nothing** for the adapter files (30 passed each, `single_key_env_probe_unfixed.txt`) — because
shape A needs the interpreter keys from the same file. The *file* is the carrier, not one variable.

Source-integrity shape, proven directly (`source_integrity_probe.txt`):

```
no conftest, no operator file                                             18 passed
no conftest, operator file present (absolute ARTIFACTS_ROOT leaks)         2 failed
no conftest, no operator file, ARTIFACTS_ROOT -> scratch dir (no manifest) 2 failed
```

So `#31`/`#32` fail either because the real manifest is read instead of the sandbox's, or because
no manifest exists at all under the injected root. Both are satisfied by pinning the root.

## 3. All 33, individually

Reproduce command for each:

```bash
workbench/envs/ctl312h/bin/python -m pytest "<node id>" -q -p no:cacheprovider --tb=long
```

Raw tracebacks: `01.txt` … `33.txt`. Duration/exit per test: `individual_repro_results.txt`.

| # | test id (file::test) | individual result @ HEAD | root-cause category | fix | verified |
|---|---|---|---|---|---|
| 01 | `test_circus_adapters_blip.py::test_adapter_import_and_unready_status[conesep]` | FAIL (`assert True is False`) | machine state — staged checkpoint (A) | conftest scrub | PASS |
| 02 | `test_circus_adapters_blip.py::test_adapter_import_and_unready_status[ptha_mtst]` | FAIL (same) | machine state (A) | conftest scrub | PASS |
| 03 | `test_circus_adapters_rest.py::test_status_is_honest_and_reports_every_mode` | FAIL (`ready=True` for habit) | machine state (A) | conftest scrub | PASS |
| 04 | `test_handoff_commands.py::test_the_release_documents_quote_the_test_count_they_actually_ran` | **PASS** individually; FAIL at file level | release-metadata guard (correct by contract) | none needed — see §4 | PASS in set |
| 05 | `test_http_api_layer.py::test_a7_launcher_reuses_ledger_corroborated_run_...` | FAIL (`pc.sh` exit 1) | machine state — absolute artifacts root (B) | `setup_sandbox` root pins | PASS |
| 06 | `test_pc_launcher.py::test_existing_completed_corroborated_run_...` | FAIL (`assert 1 == 0`) | machine state (B) | `setup_sandbox` root pins | PASS |
| 07 | `test_pc_launcher.py::test_matching_container_is_reused_positive_control` | FAIL | machine state (B) | ditto | PASS |
| 08 | `test_pc_launcher.py::test_verification_attempted_inside_container_not_on_host` | FAIL | machine state (B) | ditto | PASS |
| 09 | `test_pc_launcher.py::test_limn_bundle_selection_submits_bundle_id_with_native_protocol` | FAIL | machine state (B) | ditto | PASS |
| 10 | `test_pc_launcher.py::test_a_container_built_from_another_commit_is_not_reused` | FAIL | machine state (B) | ditto | PASS |
| 11 | `test_pc_launcher.py::test_the_summary_does_not_claim_the_index_holds_an_aggregate_only_run` | FAIL | machine state (B) | ditto | PASS |
| 12 | `test_pc_launcher.py::test_the_index_gate_refuses_a_per_query_run_that_is_not_verified` | FAIL | machine state (B) | ditto | PASS |
| 13 | `test_pc_launcher.py::test_the_index_gate_refuses_a_per_query_run_the_index_never_saw` | FAIL | machine state (B) | ditto | PASS |
| 14 | `test_pc_launcher.py::test_the_index_gate_refuses_an_aggregate_only_run_the_api_does_not_certify` | FAIL | machine state (B) | ditto | PASS |
| 15 | `test_pc_launcher.py::test_the_index_gate_refuses_a_run_the_api_cannot_certify` | FAIL | machine state (B) | ditto | PASS |
| 16 | `test_pc_launcher.py::test_a_completed_job_naming_its_run_uses_exactly_that_run` | FAIL | machine state (B) | ditto | PASS |
| 17 | `test_phase1_acceptance_scenarios.py::test_scenario_a_api_to_evaluator_success` | FAIL (`FileNotFoundError .../workspace/workbench`) | machine state (B) | phase1 key lists + master fixture pins | PASS |
| 18 | `test_phase1_acceptance_scenarios.py::test_scenario_b_..._pointer_unchanged` | FAIL (`evidence_ledger.jsonl` missing) | machine state (B) | ditto | PASS |
| 19 | `test_phase1_acceptance_scenarios.py::test_scenario_d_cancellation_kills_the_evaluator_tree` | FAIL (`operator never started`) | machine state (B) | ditto | PASS |
| 20 | `test_phase1_acceptance_scenarios.py::test_scenario_g_grandchild_ignoring_sigterm_...` | FAIL (same) | machine state (B) | ditto | PASS |
| 21 | `test_release_blocker_master_e2e.py::test_e2e01_full_synthetic_evaluation_succeeds` | FAIL (ledger does not verify) | machine state (B) | master fixture root pins | PASS |
| 22 | `...::test_e2e02_resume_skips_verified_completed_run` | FAIL (`assert 1 == 0`) | machine state (B) | ditto | PASS |
| 23 | `...::test_e2e03_checkpoint_modification_invalidates_evaluation` | FAIL | machine state (B) | ditto | PASS |
| 24 | `...::test_e2e04_dataset_modification_invalidates_evaluation` | FAIL | machine state (B) | ditto | PASS |
| 25 | `...::test_e2e04b_dataset_modification_invalidates_only_metadata_size_changes` | FAIL | machine state (B) | ditto | PASS |
| 26 | `...::test_e2e05_exit_zero_without_new_output_fails_evaluation` | FAIL | machine state (B) | ditto | PASS |
| 27 | `...::test_e2e06_failed_rerun_preserves_previous_successful_experiment` | FAIL | machine state (B) | ditto | PASS |
| 28 | `...::test_e2e07_recovery_after_failure_succeeds_and_advances` | FAIL | machine state (B) | ditto | PASS |
| 29 | `...::test_e2e11_cli_refuses_weak_input_fingerprint` | FAIL | machine state (B) | ditto | PASS |
| 30 | `...::test_e2e14_verified_gpu_capability_permits_the_authorized_synthetic_path` | FAIL | machine state (B) | ditto | PASS |
| 31 | `test_release_blocker_source_integrity.py::test_src07_declared_auxiliary_asset_accepted_only_with_provenance` | FAIL (`True is False`) | machine state (B, manifest root) | conftest scrub | PASS |
| 32 | `test_release_blocker_source_integrity.py::test_src08_modified_auxiliary_asset_rejected` | FAIL (`True is False`) | machine state (B, manifest root) | conftest scrub | PASS |
| 33 | `test_true_master_e2e.py::test_true_master_pipeline_subprocess_e2e_complete_flow` | FAIL (`logs` missing) | machine state (B) | env root pins | PASS |

Note on the earlier A/B: the prior claim that the operator file explains the 33 is **directionally
right but was overstated** — its 221/0 covered 4–8 files. Reproduced here: it also explains
`test_pc_launcher.py`, `test_phase1_acceptance_scenarios.py`, `test_http_api_layer.py` and
`test_true_master_e2e.py`, which the earlier A/B did not cover, and it explains them through
`WORKBENCH_ARTIFACTS_ROOT` rather than through the checkpoint root alone.

Note on `test_pc_launcher.py`: it uses **Docker PATH shims**, not a real daemon, so its 11
failures were never "no daemon". They were the absolute artifacts root defeating
`verify_run.py` inside the shim. Confirmed by the fix: 101 passed with no daemon involved.

## 4. `#04` — the one that is not machine state

`test_the_release_documents_quote_the_test_count_they_actually_ran` **passes individually** at
`HEAD` (it reads the count of the *named* revision in a temp worktree). It fails only when
`test_handoff_commands.py` runs as a file, because its sibling
`test_no_release_document_points_at_a_superseded_source_sha` fails there:

```
TESTED_SOURCE_SHA b0f38bd... does not contain the production code currently on disk
```

That sibling is a release-metadata guard and is **correct**: the verdict names `b0f38bd` as tested,
while 22 tracked production files under `workbench/backend/**` and `workbench/composition/**` have
uncommitted edits from the concurrent agents. The guard exists to say exactly that. It cannot pass
while other agents' work is uncommitted, and making it pass would mean either committing their work
(forbidden) or weakening the guard (forbidden). Its expectation is not obsolete; the tree simply
is not a release tree. Reported, not "fixed".

## 5. What changed

| file | change |
|---|---|
| `workbench/tests/conftest.py` (**new**) | autouse `isolate_operator_state`: deletes every `CIR_*`/`WORKBENCH_*` from the process env, repoints `operator_config._CONFIG_PATH` at a path that never exists, and clears `model_settings.file_settings` + `runtime.forget_environment()` at setup |
| `workbench/tests/test_pc_launcher.py` | `setup_sandbox` names `CIR_DATA_ROOT`, `FASHIONIQ_ROOT`, `WORKBENCH_ARTIFACTS_ROOT/CHECKPOINT/RESULTS/THIRD_PARTY/CIRCUS_RELEASE_ROOT` under the sandbox |
| `workbench/tests/test_release_blocker_master_e2e.py` | `master` fixture pins `WORKBENCH_ARTIFACTS_ROOT` and `WORKBENCH_CIRCUS_RELEASE_ROOT` |
| `workbench/tests/test_phase1_acceptance_scenarios.py` | both env key lists gain `WORKBENCH_ARTIFACTS_ROOT` + `WORKBENCH_CIRCUS_RELEASE_ROOT` |
| `workbench/tests/test_true_master_e2e.py` | fixture pins `WORKBENCH_ARTIFACTS_ROOT` + `WORKBENCH_CIRCUS_RELEASE_ROOT` |

The conftest only clears at **setup**. A teardown clear runs before `monkeypatch` restores a
test's own patch of `file_settings`, so it would call `cache_clear()` on
`test_a36_assigned_identity`'s stand-in lambda. That regression was observed (6 errors) and removed.

No test was deleted, skipped, or weakened. No assertion constant was relaxed. No counter or
document was edited to hide a failure.

### Regression check on config-sensitive tests

```
workbench/tests/test_operator_config.py test_a31_effective_interpreter.py test_a33_interpreter_alias.py
test_a34_effective_target.py test_a36_assigned_identity.py test_circus_worker_env.py
test_artifact_root_configuration.py test_manage_environment_a30.py test_r20_shared_seams.py
test_backend_config_roots.py test_index_workspace_configuration.py
=> 51 passed, 33 skipped   (config_tests2.txt)
```

## 6. Before / after

### Previously failing set — all 33

| | result | command |
|---|---|---|
| before | 33 failed (of 33) | see §3 |
| after | **33 passed in 130.37s, exit 0** | `workbench/envs/ctl312h/bin/python -m pytest $(cat ids.txt) -q -p no:cacheprovider --tb=line` |

`all33_after_fix.txt`.

### Affected files

| file | before | after |
|---|---|---|
| `test_circus_adapters_blip.py` | 2 failed, 16 passed | 18 passed |
| `test_circus_adapters_rest.py` | 1 failed, 11 passed | 12 passed |
| `test_handoff_commands.py` | 1 failed, 7 passed | 1 failed, 7 passed (guard, §4) |
| `test_http_api_layer.py` | 1 failed, 44 passed | 45 passed |
| `test_pc_launcher.py` | 11 failed, 90 passed | 101 passed |
| `test_phase1_acceptance_scenarios.py` | 5 failed, 3 passed | 8 passed |
| `test_release_blocker_master_e2e.py` | 10 failed, 6 passed | 16 passed |
| `test_release_blocker_source_integrity.py` | 2 failed, 16 passed | 18 passed |
| `test_true_master_e2e.py` | 1 failed, 2 passed | 3 passed |

(`per_file_before.txt`, `per_file_after.txt`)

### Full suite

```
workbench/envs/ctl312h/bin/python -m pytest workbench/tests -q -p no:cacheprovider -rfE --tb=no -n 10
```

| | passed | failed | skipped | collected | exit | duration |
|---|---|---|---|---|---|---|
| `b0f38bd` serial (reference) | 2186 | **33** | 47 | 2266 | 1 | 2211.38 s |
| `6421188` + this work, `-n 10` | 2321 | **12** | 47 | 2380 | 1 | 588.40 s |

Log: `full_suite_6421188_after_isolation.txt`. SHA `6421188f4f93574b7fe093a804dd235d39e630e8`.

Collected grew 2266 → 2380 because the sibling agents added their own test files
(`test_frontend_*`, `test_composition_topk_*`, `test_capability_*`, `test_circus_*_*`). **The
comparison is not clean-tree to clean-tree:** the 2321/12 describes this mixed worktree, not a
releasable revision. The number that is attributable to this work is the 33 → 0 in the previously
failing set.

## 7. The 12 still failing, and why none is mine

`new_failures_attribution.txt` — all 12 fail identically with `conftest.py` removed, and all 12
fail at `b0f38bd`'s tree for their own reasons, i.e. **none is introduced by the isolation work**.
None was in the original 33.

| test | traceback (first line) | why not fixed here |
|---|---|---|
| `test_circus_tme_lifecycle.py::test_encoder_and_limn_failed_child_latch_and_preserve_log[encoder]` | `RuntimeError: execution environment not verified [UNCONFIGURED/UNCONFIGURED]: WORKBENCH_ENCODER_PYTHON=None` | needs a real encoder interpreter; a **missing prerequisite**, not isolation. Owned by the composition agent (`require_verified_interpreter` is their new gate — see below). |
| `...[limn]` | same, `WORKBENCH_LIMN_PYTHON=None` | ditto |
| `test_release_blocker_gpu_records.py::test_defect_b_g_with_no_target_gpu_verdict_is_pending_not_pass` | `RELEASE_BLOCKED` with 368 `probe_source=override ... synthetic test probe` | shared-artifacts pollution (§2 shape C) |
| `...::test_offline_verdict_names_deployment_reason` | same | ditto |
| `...::test_release_verify_documented_invocations_produce_parseable_report[<lambda>0..4]` (5 tests) | `Invocation failed` | ditto |
| `test_handoff_commands.py::test_no_release_document_points_at_a_superseded_source_sha` | `TESTED_SOURCE_SHA b0f38bd... does not contain the production code currently on disk` | correct guard; 22 production files are uncommitted by other agents (§4) |

Two of these deserve explicitness:

- **`test_circus_tme_lifecycle`** fails because the concurrent composition agent added
  `CompositionAdapter.require_verified_interpreter()` and the test's `_encoder_adapter` /
  `_limn_adapter` helpers do not set the latch. Their sibling `test_circus_adapters_blip.py` fix
  (adding `adapter._interpreter_verified = True`, mtime 11:40, **not my edit**) shows the intended
  remedy; applying the same three lines to the two lifecycle helpers is their file to change.
- **`test_release_blocker_gpu_records`** is the shape-C defect measured above: this host's
  `workbench/artifacts/gpu_capabilities/` holds 504 records, 368 of them synthetic-probe, and
  `release_verify.sh` scans that real directory. The correct fix is to make the two
  `release_verify.sh` invocations pin `WORKBENCH_ARTIFACTS_ROOT` at a `tmp_path` sandbox the way
  the neighbouring `test_offline_...` case already does at line 795 — but I did not apply it,
  because the probe could not be made conclusive while concurrent agents were repopulating that
  shared gitignored directory (an attempted `mv` collided mid-flight; the directory was restored
  to 504 flat records and verified). Applying an unverified edit to a live assertion is worse than
  reporting it precisely. **This is the one piece of follow-up work.**

A caution for whoever picks this up: moving `workbench/artifacts/gpu_capabilities/` out of the way
while four agents are running tests will race. Isolate the *reader* (the two subprocess
`WORKBENCH_ARTIFACTS_ROOT` pins), never the shared directory.

## 8. Files changed

- `workbench/tests/conftest.py` (new)
- `workbench/tests/test_pc_launcher.py`
- `workbench/tests/test_release_blocker_master_e2e.py`
- `workbench/tests/test_phase1_acceptance_scenarios.py`
- `workbench/tests/test_true_master_e2e.py`
- `workbench/docs/reviews/20261108T120100+0700-tests/**` (this report and raw logs)

Not edited by me but modified in-tree by concurrent agents during this work, and visible in the
sweep: `workbench/tests/test_circus_adapters_blip.py`, `test_frontend_staleness.py`, plus 22
production files under `workbench/backend/**` / `workbench/composition/**` and 8 new frontend
files. `workbench/config/workbench.env` was moved aside for probes only, never edited, and
verified byte-identical afterwards.
