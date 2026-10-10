# Laptop Release A — Engineering Acceptance

Release A covers all laptop-verifiable engineering. Real GPU inference is Release B
(see `PC_REPRODUCTION_HANDOFF.md`).

## Scope

Delivered and verified **without** real checkpoints, real FashionIQ images, GPU
inference, package installation, or heavy downloads:

- Standalone operator scripts with documented CLI and side-effect policy
- Registry-derived per-model execution DAG with enforced dependencies
- Explicit authorization policy applied uniformly to every master mode
- Verified content-based resume (inputs, outputs, interrupt safety, atomic state)
- Environment isolation verification and safe-provisioning guards
- Model-specific preparation dispatch integrated into master setup
- Bundle-aware checkpoint identity (LIMN three-category, DCNet run directory)
- Source-backed aggregate metric extraction with independent macro recomputation
- Strict aggregate / schema-v2 separation and DuckDB indexing integrity
- Workflow reporting on SUCCESS / PARTIAL / FAILED / BLOCKED
- Offline synthetic Level-3 master E2E, adversarial, and mutation tests

## Test evidence

```bash
python -m compileall -q src workbench
PYTHONPATH=. python -m pytest workbench/tests -q
python -m pytest tests/unit -q
git diff --check
```

| Suite | File | Purpose |
|---|---|---|
| True master E2E | `workbench/tests/test_true_master_e2e.py` | Real `pipeline.py` subprocess: full graph, resume, invalidation, failure report |
| Authorization | `workbench/tests/test_authorization_policy.py` | A01–A06: no unauthorized subprocess, dry-run no mutation |
| Execution graph | `workbench/tests/test_execution_graph_and_dependencies.py` | Dependency enforcement, `SKIPPED_DEPENDENCY`, per-model jobs |
| Production resume | `workbench/tests/test_production_stage_resume_invalidation.py` | Actual stage input contracts invalidate on checkpoint/dataset/source change |
| Adversarial metrics | `workbench/tests/test_adversarial_metrics.py` | V01–V15: NaN, inf, duplicates, macro inconsistency |
| Adversarial environment | `workbench/tests/test_adversarial_environment.py` | E01–E10: isolation, system/base refusal |
| Adversarial resume | `workbench/tests/test_adversarial_resume.py` | R05–R11: output deletion/corruption, interruption, atomic state |
| Adversarial orchestration | `workbench/tests/test_adversarial_orchestration.py` | M/O/D criteria |
| Failure isolation | `workbench/tests/test_scenario_failure_isolation.py` | Blocked model does not block independent model |
| Mutation effectiveness | `workbench/tests/test_mutation_effectiveness.py` | 8 critical invariants killed if regressed |
| Workflow reporting | `workbench/tests/test_workflow_report_termination.py` | Report on SUCCESS/PARTIAL/FAILED/BLOCKED |

## Acceptance status

| Criterion | Status |
|---|---|
| Authorization bypass fixed for all modes | PASS |
| Registry-derived per-model execution graph | PASS |
| Dependencies enforced (`SKIPPED_DEPENDENCY`) | PASS |
| `--continue-on-error` isolates independent models | PASS |
| Production stages declare input/output identities | PASS |
| Resume rejects missing/corrupt outputs and changed inputs | PASS |
| Interrupted stage not skipped as COMPLETE | PASS |
| Environment verification does not overstate readiness | PASS |
| Unsafe system/base installation refused | PASS |
| Preparation dispatcher integrated into master setup | PASS |
| Checkpoint file/directory/bundle handling correct | PASS |
| LIMN uses three category checkpoints; aggregate recomputed | PASS |
| Aggregate vs per-query separation enforced | PASS |
| Workflow report on all terminations | PASS |
| True master synthetic E2E | PASS |
| Critical mutations detected | PASS |
| Real GPU inference required | NOT REQUIRED for Release A |

## Laptop / PC boundary

- **Finished on laptop:** all orchestration, verification, provenance, extraction,
  authorization, resume, and offline synthetic E2E.
- **Pending due to GPU:** actual official evaluation on real checkpoints, aggregate
  parity vs papers, per-query instrumentation.
- **Pending due to missing external assets:** ENCODER/PAIR/Combiner/CLIP4Cir/DCNet/
  CLVC-Net/TG-CIR/SPRC checkpoint mappings and auxiliary assets.
- **Not implemented:** per-query ranking export patches for external models
  (deferred until official aggregate parity on a GPU host).

## Correction pass (independent-review defects)

| Defect | Fix | Test |
|---|---|---|
| Environment readiness overstated | Tiered `verify_environment` (INTERPRETER_PRESENT → ISOLATION_VERIFIED → PYTHON_VERSION_VERIFIED → DEPENDENCIES_VERIFIED → MODEL_IMPORT_VERIFIED → CUDA_VERIFIED → RUNTIME_READY); no `RUNTIME_READY` without source-backed dependency proof | `test_adversarial_environment.py`, `test_mut6` |
| Linux venv misjudged via resolved path | Isolation now decided by runtime `sys.prefix`/`sys.base_prefix`/markers, not symlink target | `test_adversarial_environment.py` (real `venv`) |
| Stage COMPLETE without outputs | `validate_stage_outputs` is enforced after exit 0 for every stage; declared outputs must exist, be non-empty, parse, and follow run manifests | `test_pipeline.py`, `test_mut2`, `test_mut9` |
| Aggregate report overwrote experiments | Run-scoped immutable reports `<model>_<ckpt>_<run_id>_aggregate.json` + atomic `_latest.json` pointer | `test_mut10`, `test_true_master_e2e.py` |
| `--continue-on-error` failed at runtime-preflight | Removed name special-case; failed preflight records `FAILED`, dependents `SKIPPED_DEPENDENCY`, independent models continue | `test_pipeline.py::test_runtime_preflight_cannot_continue_into_evaluation` |
| Compound capabilities | `Stage.required_capabilities: frozenset[str]`; checkpoint needs `allow_large_downloads`+`allow_network`, sync needs `allow_network`+`allow_preparation`, eval needs `allow_gpu_eval`+`allow_preparation` | `test_authorization_policy.py`, `test_pipeline.py` |
| Dirty source reused by resume | Fingerprint includes commit + cleanliness + provenance digest; `_source_dirty` ignores only bytecode caches | `test_production_stage_resume_invalidation.py`, `test_true_master_e2e.py` |
| Mutation tests did not mutate | 10 mutations applied with `monkeypatch.context()` and required to be killed | `test_mutation_effectiveness.py` |
| Concurrency untested | Real multi-process flock ownership tests incl. crash recovery | `test_concurrency_ownership.py` |
| Synthetic adapter in production registry | Moved behind `register_test_adapter` / `WORKBENCH_ALLOW_SYNTHETIC_ADAPTERS`; metric routing no longer special-cases it | `test_only_synthetic.py` |

### Second review round (Gate5/Gate6)

| Defect | Fix | Test |
|---|---|---|
| Persisted FAILED dependency ignored in a later invocation | `_UNUSABLE_DEP_STATUSES` consulted against the state file | `test_persisted_failed_dependency_blocks_later_invocation` |
| Dependency string did not match qualified key (`checkpoint:limn` vs `checkpoint:limn:base_iter0_dress`) | prefix-aware lookup scoped to the stage's own model | `test_qualified_dependency_key_is_matched`, `test_bare_dependency_does_not_match_other_models` |
| `dataset-link`, provisioning stages could never reach COMPLETE | real output contract or durable `receipt=True` proof | `test_dataset_link_declares_output_contract`, `test_provisioning_stage_requires_receipt` |
| LIMN evaluation proof used the wrong checkpoint id | proof filename derived from the bundle id the evaluator writes | `test_limn_eval_output_uses_bundle_id` |
| Index/mock writers ungated | `allow_index_write` / `allow_mock_write`; `analyze` and `mock` self-authorize | `test_validation_index_requires_index_write_capability` |
| `--dry-run` wrote workflow state | every mutation path guarded by planning mode | `test_dry_run_never_mutates_state` |
| Model with no checkpoint produced a run-and-fail stage | stage is skipped when no registry output exists | `test_model_without_checkpoint_has_no_side_effecting_checkpoint_stage` |
| Committed bytecode counted as dirty; real edits inside `__pycache__` ignored | bytecode matched by concrete path/suffix only | `test_committed_bytecode_is_not_dirty`, `test_tracked_edit_inside_pycache_dir_detected` |
| `RUNTIME_READY` without an executed dependency probe | readiness requires a real probe | `test_runtime_ready_requires_dependency_probe` |
| Run report written with plain `write_text` | exclusive `open("x")` | `test_mut10` |
