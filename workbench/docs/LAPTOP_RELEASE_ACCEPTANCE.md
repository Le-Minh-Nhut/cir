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
| Synthetic adapter in production registry | Moved behind `register_test_adapter` / `WORKBENCH_ALLOW_SYNTHETIC_ADAPTERS`; metric routing no longer special-cases it | `workbench/tests/test_only_synthetic.py` |

### Release-blocker correction pass (interpreter, source, experiment identity)

| Defect | Fix | Regression tests |
|---|---|---|
| `python_executable()` consulted a non-probing verification, so a correctly installed model env could not build its command | split `inspect_environment()` (cheap, never ready) from `require_verified_model_interpreter()`; `python_executable()` delegates to the latter and raises otherwise | `test_release_blocker_interpreter.py` (INT-04, INT-05, INT-06, MUT-A) |
| A valid Linux venv was rejected because its `bin/python` resolved to the orchestrator's shared binary | isolation decided by `sys.prefix`/`sys.base_prefix`, reported `sys.executable`, and invocation-path markers; configured path preserved in `manage_environment.py` | INT-01..INT-03, ENV-01/02, MUT-B |
| Prefix version matching (`got.startswith(want)`) treated any prefix as agreement | `packaging.specifiers` matching; a bare version is an exact pin; `1.12` and `1.12.10` are now rejected for a required `1.12.1`; unusable constraints report `VERSION_CONSTRAINT_UNVERIFIED` | `test_release_blocker_interpreter.py` (VER-01..VER-06), `test_release_blocker_mutations.py` (MUT-F) |
| Verified environments could not be reused, yet any path-keyed cache would be unsound | contract + interpreter + marker + `lib/` mtime fingerprint invalidates reuse on dependency change | `test_cache_is_not_keyed_on_interpreter_path_alone` |
| CSMCIR preparation immediately dirtied the pinned checkout | classified source status with a narrowly scoped, provenance-verified runtime artifact allowlist shared by doctor and the gate | `test_release_blocker_source_integrity.py` (SRC-01..SRC-10, MUT-C) |
| A stale `_latest.json` could satisfy a new evaluation, so exit 0 with no new report reported success | invocation-bound completion proofs validated against run id, model, checkpoint, protocol, source commit, checkpoint digest, report digest, evidence digest | `test_release_blocker_evaluation_output.py` (OUT-01..OUT-14, MUT-D) |
| A failed run replaced the latest pointer associated with a successful experiment | separate `latest_attempt` / `latest_successful`; successes advance only after a proof-backed success | OUT-11..OUT-13, E2E-05..E2E-07, MUT-E |
| Checkpoint digest and report location were not part of the completion contract | proof must state and match the selected checkpoint digest; referenced report must live under the artifact root | OUT-08, OUT-10b |
| Exit 0 with no observable output still minted a proof | proof requires an evidence digest over the run's own logs and rejects an empty run | OUT-02, OUT-10c, E2E-05 |
| A spoofed shell "python" with hand-written `pyvenv.cfg` reached RUNTIME_READY | interpreter identity probe carries a per-call nonce the wrapper cannot know, requires a CPython implementation, requires markers under the interpreter's own reported prefix, and requires a real Python library tree | INT-01..INT-03, MUT-B |
| A legitimate venv under a directory named `miniconda3` was rejected as base conda | conda detection uses `conda-meta` markers and the `envs/` segment, never path substrings | INT-02 |
| A copied/renamed completion proof certified a run that never happened | the proof's run directory must equal this invocation's run id, and the validator derives the expected invocation identity from config | OUT-01, OUT-04 |
| Cache reuse survived an uninstall in symlinked venvs (`resolve()` landed on the shared base binary) | fingerprint follows the prefix the interpreter *reports* | `test_verified_environment_is_reused_and_content_change_invalidates_cache` |
| `python: '3.7.x'` passed as verified on Python 3.13 | an uninterpretable declared Python version reports `VERSION_CONSTRAINT_UNVERIFIED` | ENV-04, VER-04 |
| A GPU-deferred report skipped dependency evidence | dependency-probe guard precedes the CUDA branch | INT-09 |
| Models without an isolated environment got a bare `"python"` command | such models use the workbench interpreter explicitly | ENV-01 |
| A checkout elsewhere sharing a model's `source_dir` name borrowed its allowlist | the allowlist requires the configured `WORKBENCH_THIRD_PARTY_ROOT/<source_dir>` | SRC-01..SRC-10 |
| A compiled launcher carrying the strings `Py_Initialize`/`Py_Main` passed the byte scan | the unsound byte scan is **removed**; the interpreter is bound by an operator-pinned `environment.interpreter_sha256` (mismatch → `UNSAFE_INTERPRETER`, `UNKNOWN` → unpinned) | `test_a_declared_interpreter_digest_is_enforced`, `test_an_unpinned_interpreter_digest_is_not_required` |
| PEP 503 name mismatch (`annotated_doc` vs `annotated-doc`, `openai_clip` vs `openai-clip`) made a perfect environment read as forged | names normalised on both sides of the reconciliation | `test_distribution_names_are_normalized_on_both_sides`, `test_distribution_name_normalization_is_pep503` |
| A package visible only via editable/`.pth`/`system_site_packages` was reported as a forgery | three-way outcome: `DEPENDENCY_INSTALLATION_UNVERIFIED`, never ready | `test_editable_style_install_is_never_ready` |
| An environment could create a `dist-info` for itself while being probed | the on-disk baseline is read **before** any in-environment code runs and pinned for the session | `test_an_environment_cannot_fabricate_a_distribution_while_being_probed` |
| An undeclared extra file in a run directory flipped a directory-checkpoint verdict | digests are member-name-bound; undeclared files are ignored, declared members are not | `test_an_undeclared_file_does_not_change_a_directory_checkpoint_identity`, `test_directory_checkpoint_digest_distinguishes_member_names` |
| A `sitecustomize.py` inside the environment forged the dependency and import answers | probe answers are reconciled against the orchestrator's own on-disk inventory (`*.dist-info/METADATA`, `*.egg-info/PKG-INFO`); a contradiction is `DEPENDENCY_PROBE_FORGED` | `test_an_environment_cannot_forge_its_own_package_inventory`, `test_on_disk_inventory_ignores_sitecustomize` |
| Directory-checkpoint proofs were rejected by the validator (undeclared extra files changed the digest) | the validator hashes the registry-declared members, exactly as the planner does | `test_directory_checkpoint_digest_is_part_of_the_completion_contract` |
| Legacy `*.egg-info` installs were not fingerprinted | the fingerprint covers egg-info metadata and members | `test_legacy_egg_info_install_invalidates_cache` |
| The MUT-C detector accepted an unreadable checkout as proof of cleanliness | it asserts the classification is readable and the declared artifact is approved | `test_mut_c_treating_declared_runtime_artifacts_as_dirty_is_detected` |
| The backend wrote a second, incompatible run-directory shape | the backend uses the orchestrator's `{model}__{checkpoint}__{run_id}` convention | `test_run_directories_do_not_collide_across_models` |
| A compiled/shell launcher reflecting argv passed as CPython | interpreter identity is derived from the invocation path and an `-I` nonce probe, not from the target's self-report | `test_a_wrapper_script_is_never_a_model_interpreter`, `test_relocated_interpreter_with_forged_markers_is_rejected`, MUT-H |
| A `sitecustomize.py` inside the environment fabricated `sys.prefix` and forged a valid completion proof | prefix comes from the invocation path; dependency/import probes must report the same prefix | `test_a_foreign_environment_probe_cannot_answer_for_the_model_environment` |
| A legitimate `conda -p` environment was rejected as base conda | base conda detected by an `envs/` directory beside `conda-meta`, never by a path substring | `test_chained_venv_identity_is_never_taken_from_the_chain`, INT-02 |
| Run directories collided across models sharing a checkpoint id | run directory is qualified by model **and** checkpoint, and a directory collision kills the started child | `test_run_directories_do_not_collide_across_models`, `test_run_directories_are_distinct_per_checkpoint_variant` |
| Run/bundle directory checkpoints had no digest binding | directory checkpoints are hashed (declared members) and the digest is part of the completion contract | `test_directory_checkpoint_digest_is_part_of_the_completion_contract` |
| Cache survived an in-place package member rewrite | fingerprint hashes each distribution's `RECORD` members | `test_metadata_rewrite_in_place_invalidates_cache` |
| PE detection matched 4 bytes, accepting a 2-byte `MZ` and rejecting real stubs | header compared as exactly `MZ` | `test_a_wrapper_script_is_never_a_model_interpreter` |
| A crashed pointer update left a temp file behind | atomic write cleans up and preserves the previous pointer on failure | OUT-13b |

### Second review round (Gate5/Gate6)

| Defect | Fix | Test |
|---|---|---|
| Persisted FAILED dependency ignored in a later invocation | `_UNUSABLE_DEP_STATUSES` consulted against the state file | `workbench/tests/test_execution_graph_and_dependencies.py` |
| Dependency string did not match qualified key (`checkpoint:limn` vs `checkpoint:limn:base_iter0_dress`) | prefix-aware lookup scoped to the stage's own model | `test_qualified_dependency_key_is_matched`, `test_bare_dependency_does_not_match_other_models` |
| `dataset-link`, provisioning stages could never reach COMPLETE | real output contract or durable `receipt=True` proof | `test_dataset_link_declares_output_contract`, `test_provisioning_stage_requires_receipt` |
| LIMN evaluation proof used the wrong checkpoint id | proof filename derived from the bundle id the evaluator writes | `test_limn_eval_output_uses_bundle_id` |
| Index/mock writers ungated | `allow_index_write` / `allow_mock_write`; `analyze` and `mock` self-authorize | `test_validation_index_requires_index_write_capability` |
| `--dry-run` wrote workflow state | every mutation path guarded by planning mode | `test_dry_run_never_mutates_state` |
| Model with no checkpoint produced a run-and-fail stage | stage is skipped when no registry output exists | `test_model_without_checkpoint_has_no_side_effecting_checkpoint_stage` |
| Committed bytecode counted as dirty; real edits inside `__pycache__` ignored | bytecode matched by concrete path/suffix only | `test_committed_bytecode_is_not_dirty`, `test_tracked_edit_inside_pycache_dir_detected` |
| `RUNTIME_READY` without an executed dependency probe | readiness requires a real probe | `test_runtime_ready_requires_dependency_probe` |
| Run report written with plain `write_text` | exclusive `open("x")` | `test_mut10` |
