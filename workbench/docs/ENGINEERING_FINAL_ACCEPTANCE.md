# Engineering final acceptance

Laptop-side engineering for the CIR failure-analysis workbench. This document covers the
engineering chain only; scientific source-faithfulness is covered by
`UPSTREAM_DATA_PREPARATION_AUDIT.md`, `SOURCE_TO_WORKBENCH_PARITY_MATRIX.md` and
`FASHIONIQ_PROTOCOL_PARITY.md`.

Verification: `python -m pytest tests/unit workbench/tests -q` — **565 passing**, plus
`python -m compileall -q src workbench` and `git diff --check` clean.

## The chain

    FastAPI  ->  SingleGpuQueue  ->  evaluate_models.py (trusted operator)
             ->  official evaluator subprocess (untrusted)
             ->  run directory + aggregate report
             ->  HMAC evidence ledger  ->  completion proof  ->  latest_successful

Each arrow now has one owner per artifact and one gate per launch.

## Defects closed in this pass

Every one was reproduced before being changed, and each has a regression test.

### API and queue

| Defect | Root cause | Fix | Test |
|---|---|---|---|
| A model the API accepted could never start | the handler required readiness, then queued a command without `--verify-gpu`, while the operator refused any CUDA-declaring model lacking that flag | `--verify-gpu` now means "establish or refresh the capability"; the launch gate is the single authority, and a valid record is sufficient | `API-GPU-*` in `test_phase1_integration_contracts.py` |
| A queued run could not authenticate its ledger entry | the queue stripped `WORKBENCH_LEDGER_KEY` from a child that *is* the trusted operator | role-aware environment: the queue passes the key and drops only secrets the operator does not need; the operator strips the key and every `WORKBENCH_SECRET_*` from model code | `test_the_queue_hands_the_operator_its_key_but_no_stray_secrets`, `test_the_official_evaluator_never_receives_the_operator_key` |
| The queue's `command.json` could stand in for the run's own record | both writers used `logs/<model>__<checkpoint>__<run_id>/` | the queue owns `queue_jobs/<job_id>/` with `queue_*.{json,log}` names; the operator owns the run directory | `test_queue_and_operator_own_separate_directories` |
| A cancelled job's evaluator kept running | only the operator's pid was signalled | jobs lead their own process group; cancellation signals the group with SIGTERM, bounded grace, SIGKILL escalation, and never reports cancelled while an owned process lives | `test_cancellation_stops_the_operator_and_the_evaluator_it_started`, `test_cancelling_one_job_does_not_touch_another` |

### Operator and evidence

| Defect | Root cause | Fix | Test |
|---|---|---|---|
| A collision could destroy another invocation's evidence | the run directory and a temp dir inside it were created before `Popen` and cleaned up on launch failure | the directory is claimed exclusively with `mkdir(exist_ok=False)` before anything starts; an existing directory is refused as-is, and only a self-created one is ever removed | `test_an_existing_run_directory_is_refused_before_the_evaluator_starts`, `RUN-*` |
| A duplicate run id started a second evaluator that had no ledger entry | uniqueness was enforced only at append time | checked before the child starts | `test_a_duplicate_run_id_is_refused_before_the_evaluator_starts` |
| The API queued the raw model evaluator, bypassing the ledger | the handler passed the adapter's command straight to the queue | the queue dispatches the audited orchestrator with an explicit run id | `test_the_api_queues_the_audited_orchestrator_not_the_raw_evaluator` |
| The evaluator wrote into the shared, world-readable host temp dir | no `TMPDIR` was set for the child | run-scoped `TMPDIR` under the run's log directory, mode `0700` | `test_the_evaluator_child_gets_no_operator_secrets_and_a_private_temp_dir` |

### GPU capability

| Defect | Root cause | Fix | Test |
|---|---|---|---|
| A hand-built `torch` could mint a signed capability | corroboration trusted the probe's own report | the orchestrator loads the reported core in a child process and requires `PyInit__C`; the module must live inside the verified environment and is hashed | `test_a_stub_module_cannot_supply_gpu_evidence`, `test_a_python_file_named_under_torch_cannot_claim_a_compiled_core`, `test_a_fake_shared_library_cannot_claim_a_compiled_core`, `test_an_unrelated_binary_named_C_so_cannot_corroborate`, `test_a_torch_module_outside_the_environment_cannot_corroborate` |
| A large distribution kept a stale capability | members past a fixed cap were identified by `RECORD` alone | every member is hashed in full | `test_cuda10_changed_dependency_fingerprint_invalidates_verification` |
| A rewrite preserving size *and* mtime returned the old digest | the digest memo was keyed on `(path, size, mtime)` | digests are computed every time; verification is not per-query | `test_a_same_mtime_rewrite_cannot_keep_an_old_member_digest` |
| Verifying model B overwrote model A's capability | one fixed `gpu_capability.json` | per model+interpreter+dependency key under `artifacts/gpu_capabilities/`, written atomically, with a legacy single-file read that only accepts a matching key | `test_two_models_keep_independent_capability_records`, `test_scenario_f_two_models_keep_independent_capabilities` |
| The documented first-run `--verify-gpu` workflow deadlocked | the strict preflight refused the pending GPU before the stage that probes it | the preflight immediately preceding an authorized probe takes `--allow-gpu-pending`, which tolerates only that state; a failed probe still stops the run | E2E-13 |
| A fallback-key operator could never persist a capability | the key file was read but never created (and the first fix shipped without `import tempfile`) | the file is created on first use, as the ledger does | `test_the_fallback_key_is_created_so_a_capability_can_be_signed` |

### Resume fingerprints

| Defect | Root cause | Fix | Test |
|---|---|---|---|
| A changed file inside a declared directory was invisible to resume | directory inputs were fingerprinted by their own metadata | directory inputs are inventoried with per-file content digests under a recorded budget | `test_fp13_same_size_member_rewrite_invalidates_a_directory_input` |
| A truncated inventory still counted as content-verified | truncation set a flag but not the category | truncation now yields `TREE_TRUNCATED`, which is weak evidence and refuses verified resume | FP tests |
| A retargeted symlink could go unnoticed | symlink targets were not hashed | link targets are part of the digest | FP tests |

### Scientific (laptop-testable)

| Defect | Root cause | Fix | Test |
|---|---|---|---|
| Five registered models named a protocol the evaluator could not build | `build_fashioniq_gallery` had two branches for three protocols | the third branch exists; the reference rule is enforced by the function signature | `tests/unit/test_fashioniq_protocols.py` |
| CSMCIR's extension mismatch was invisible until a GPU run | preflight checked directories and JSON files, never images | the registry declares `fashioniq_image_extensions` and both preflight views check every validation id | `test_an_official_evaluator_extension_mismatch_is_reported` |

## Explicitly unchanged

- The ledger key still lives outside the repository and is never passed to model code; a
  same-user attacker holding it is out of scope, as documented.
- The CUDA signal itself remains structurally unverifiable: it is produced by Python imported
  from the model's environment. A capability record is operator evidence, not proof against a
  hostile environment. `--verify-gpu-only` exists for the hardware preflight.
- Process-group cancellation does not contain a child that deliberately detaches into a new
  session; that needs an OS sandbox, which this workbench does not claim.

## Acceptance against the stated criteria

| Criterion | Status |
|---|---|
| API queue launches the trusted operator | yes |
| Trusted operator receives its signing credentials | yes |
| Native evaluator receives no signing key | yes |
| Queue and scientific logs have separate ownership | yes |
| Existing run directories cannot be overwritten | yes |
| Failed launches cannot delete previous evidence | yes |
| Cancellation terminates managed evaluator descendants | yes |
| Job status reflects actual operator completion | yes |
| GPU capability reusable when valid | yes |
| Multiple model capabilities persist independently | yes |
| GPU-only command does not run evaluation | yes |
| Resume fingerprints scientifically reliable | yes, with truncation marked weak |
| Real API-to-evaluator synthetic E2E passes | yes, six scenarios |
| Unresolved laptop-testable P0/P1 defects | none known |

## Verification commands

```bash
python -m compileall -q src workbench
python -m pytest tests/unit workbench/tests -q
git diff --check
```

Focused suites: `workbench/tests/test_phase1_integration_contracts.py`,
`workbench/tests/test_phase1_acceptance_scenarios.py`,
`workbench/tests/test_release_blocker_{cuda_readiness,evaluation_output,production_stage_fingerprint,mutations,master_e2e}.py`,
`tests/unit/test_fashioniq_{protocols,caption_policies}.py`.
