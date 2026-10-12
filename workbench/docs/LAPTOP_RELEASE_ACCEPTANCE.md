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

**The suite needs free space in the pytest temp root.** The master E2E and synthetic
scenarios build real venvs and workspaces per test — one full run retains roughly 1.3 GB under
`/tmp/pytest-of-$USER`. On a tmpfs `/tmp` that is shared with other work, a run can therefore
fail with `OSError: [Errno 122] Disk quota exceeded` in unrelated-looking tests; that is an
environment failure, not a defect, and it is order-independent. Clear
`/tmp/pytest-of-*` between runs, or point `--basetemp` at a filesystem with several GB free.
A run that stops producing correct results this way exits nonzero (48 failed, 29 errors were
observed with the filesystem exhausted), so it cannot be mistaken for a passing suite.

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
| A `dist-info` whose `METADATA` omitted `Version` made a forged probe verify | an unreadable version is recorded as unknown, never as `""`, and never corroborates a probe | `test_an_unreadable_installed_version_never_confirms_a_probe` |
| A changed `interpreter_sha256` pin was not enforced | the pin is part of the contract digest and is re-checked before any cached report is returned | `test_a_pinned_digest_is_enforced_even_after_a_cached_verification` |
| An environment could answer for itself (a `.pth`, `sitecustomize`, or rewritten `dist-info` claiming packages) | requirements are checked against the environment's on-disk inventory, read by the orchestrator via `importlib.metadata.distributions(path=...)`; no environment code executes during inspection | `test_probes_never_execute_code_belonging_to_the_environment`, `test_an_environment_cannot_fabricate_a_distribution_while_being_probed`, MUT-H |
| A `.pth` in the environment ran during inspection and fabricated a distribution, including via the CUDA rung | the directory is never added to `sys.path` (`site.addsitedir` executes `.pth` lines), and CUDA is not probed at all: a declared CUDA requirement is `GPU_VERIFICATION_DEFERRED` | `test_probes_never_execute_code_belonging_to_the_environment`, `test_int09_declared_cuda_requirement_is_deferred_not_verified`, MUT-H |
| A hollow install (module removed, `RECORD` listing only itself) verified | installed content excludes the distribution's own metadata directory | `test_mut_j_accepting_a_hollow_install_is_detected` |
| A legitimate `--system-site-packages` venv over a Debian `dist-packages` base was refused | `dist-packages` is read alongside `site-packages`, and the base prefix is consulted by the same walk | `test_system_site_packages_environment_resolves_from_its_base` |
| A `RECORD`-less `dist-info`, an empty `egg-info`, or a `RECORD` naming a file outside the tree still verified | one walk judges installed content: escaping/absolute/self members and descriptive-only metadata install nothing | `test_record_less_metadata_is_not_an_installation`, `test_empty_egg_info_is_not_an_installation`, `test_record_members_cannot_escape_the_environment` |
| Three parallel disk walks could disagree about one environment | a single `inspect_environment_distributions` produces the version inventory, the importable module set, and the set of underivable installs | `test_inventory_and_module_checks_agree` |
| A `deb`-installed distro tree looked empty: 37 of 108 real distributions were dropped by a variable-name clobber in the RECORD-less branch | the walk binds distinct locals and recognises the Debian layout (code beside the metadata); underivable module mappings are reported unverified | `test_distro_layout_without_record_is_recognised`, `test_a_manager_installed_distribution_with_an_opaque_module_is_unconfirmed`, `test_a_real_distro_environment_is_not_reported_empty` |
| A proof could re-certify a failed run by editing only itself | the proof is cross-checked against its run report (run id, model, checkpoint, protocol, return code, execution status, evidence digest) | `test_a_failed_run_cannot_be_re_certified_by_editing_the_proof` |
| Rewriting the report **and** the proof together re-certified a failure | `command.json`, written by the runner around the subprocess, is cross-checked too and its non-zero return code is decisive | `test_a_rewritten_failed_run_is_caught_by_the_invocation_record` |
| Rewriting all three run artifacts consistently still re-certified a failure | an append-only, hash-chained **evidence ledger** at the artifact root records each invocation out-of-band; it must contain a matching entry | `test_the_ledger_catches_artifacts_rewritten_consistently`, `test_the_ledger_records_every_invocation_and_chains` |
| A changed bundle member was accepted because the validator re-hashed members and compared them to fresh hashes | the proof's recorded bundle manifest digest is compared against the live members | `test_a_bundle_member_change_is_caught_by_the_recorded_manifest` |
| The pipeline passed a checkpoint path explicitly and so lost the declared member list, rejecting legitimate run directories | members are resolved from the registry in the validator as well | `test_the_pipeline_path_resolves_declared_directory_members` |
| A changed bundle member (LIMN shirt/toptee) left the proof valid | every recorded bundle member is re-hashed, not just the anchor | `test_a_changed_bundle_member_invalidates_the_proof` |
| `--all-runnable` evaluation always failed its own output contract | it is planned as skipped with an instruction to run per model | `test_all_runnable_evaluation_is_planned_skipped_not_failed` |
| A RECORD naming only an empty directory counted as installed content | a directory entry needs importable files; namespace packages still count | `test_a_record_naming_only_an_empty_directory_installs_nothing`, `test_a_namespace_package_directory_is_still_accepted` |
| A declared `required_imports` entry did not resolve an opaque manager install | the declared module is checked directly for such distributions | `test_a_declared_import_resolves_a_manager_installed_distribution` |
| A RECORD spelling its own metadata as `./x.dist-info/...` slipped past the self-metadata exclusion | RECORD members are normalised (`.`/repeated separators) before the exclusion and containment checks | `test_dot_prefixed_self_metadata_is_not_installed_content` |
| Metadata-only `dist-info` (no `RECORD`, no installed files) satisfied a requirement | corroboration requires real installed content | `test_metadata_only_distribution_is_not_corroboration` |
| An unversioned registry declaration (`openai_clip`, `Pillow`, `torchvision`) made 3 of 7 models permanently non-ready | an unversioned declaration means "any version": presence is the requirement; `VERSION_CONSTRAINT_UNVERIFIED` is only for unparseable constraints | `test_ver06_unversioned_declaration_requires_presence_only` |
| An orchestrator-supplied `--run-id` could escape the artifact root and leave debris | run ids are validated at the entry point (no separator, no `..`) | `test_unsafe_run_ids_are_rejected` |
| A CUDA-declaring model had no path from pending to executable (permanent `GPU_VERIFICATION_DEFERRED` dead end) | explicit GPU state machine: `GPU_VERIFICATION_PENDING` -> authorized `verify_gpu_runtime()` -> `GPU_RUNTIME_READY`, with separate plan-level (`READY_TIERS`) and launch-level (`LAUNCH_READY_TIERS`) gates | `test_release_blocker_cuda_readiness.py` (CUDA-01..CUDA-14), MUT-01, MUT-10, E2E-13, E2E-14 |
| `ledger_hmac_key` returned `b""` on failure — a predictable key | fails closed with `LedgerKeyError`; atomic key creation; an existing ledger never silently acquires a new key | KEY-02..KEY-11 |
| Ledger appends had no transaction boundary, so concurrent evaluators could fork the chain | exclusive `flock` around read-verify-append, shared lock for readers, duplicate run ids rejected, corruption never auto-reset | LEDGER-01..LEDGER-10 |
| No-output invocations never reached the ledger | every started invocation appends one entry with `invocation_status`; audit record is not a proof | AUDIT-01..AUDIT-10 |
| The standalone CLI could exit 0 without a valid completion proof | native return code preserved in metadata; operator status fails when the proof is missing or invalid | EXIT-01..EXIT-10 |
| A SHA-256 failure silently fell back to mtime and looked content-verified | explicit per-input evidence; only `CONTENT_VERIFIED` permits resume; weak inputs record a refusal reason | FP-01..FP-12, E2E-11 |
| A compiled launcher carrying the strings `Py_Initialize`/`Py_Main` passed the byte scan | the unsound byte scan is **removed**; the interpreter is bound by an operator-pinned `environment.interpreter_sha256` (mismatch → `UNSAFE_INTERPRETER`, `UNKNOWN` → unpinned) | `test_a_declared_interpreter_digest_is_enforced`, `test_an_unpinned_interpreter_digest_is_not_required` |
| PEP 503 name mismatch (`annotated_doc` vs `annotated-doc`, `openai_clip` vs `openai-clip`) made a perfect environment read as forged | names normalised on both sides of the reconciliation | `test_distribution_names_are_normalized_on_both_sides`, `test_distribution_name_normalization_is_pep503` |
| A package visible only via editable/`.pth`/`system_site_packages` was reported as a forgery | three-way outcome: `DEPENDENCY_INSTALLATION_UNVERIFIED`, never ready | `test_editable_style_install_is_never_ready` |
| An environment could create a `dist-info` for itself while being probed | the on-disk baseline is read **before** any in-environment code runs and pinned for the session | `test_an_environment_cannot_fabricate_a_distribution_while_being_probed` |
| An undeclared extra file in a run directory flipped a directory-checkpoint verdict | digests are member-name-bound; undeclared files are ignored, declared members are not | `test_an_undeclared_file_does_not_change_a_directory_checkpoint_identity`, `test_directory_checkpoint_digest_binds_member_names_and_layout` |
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

### Third adversarial round (GateA / GateB / FinalGate)

Each defect below was reproduced by running the code, and each contradicted a guarantee the
runbook already made. Every row has a regression test.

| Defect | Fix | Test |
|---|---|---|
| A hand-built `torch` package (plain module, or `_C` as `.py`, or a text file named `_C.so`, or a copy of `/bin/true`) could mint an operator-signed GPU capability | the orchestrator loads the reported core itself in a child process and requires CPython's `PyInit__C`; the module must live inside the verified environment and is hashed | `test_a_stub_module_cannot_supply_gpu_evidence`, `test_a_python_file_named_under_torch_cannot_claim_a_compiled_core`, `test_a_fake_shared_library_cannot_claim_a_compiled_core`, `test_an_unrelated_binary_named_C_so_cannot_corroborate`, `test_a_torch_module_outside_the_environment_cannot_corroborate` |
| A distribution with more members than the fingerprint cap kept a stale capability after an in-place member edit | every member is hashed in full | `test_cuda10_changed_dependency_fingerprint_invalidates_verification` |
| The member-digest memo was keyed on `(path, size, mtime)`, so a rewrite that restored mtime returned the old digest | digests are computed every time | `test_a_same_mtime_rewrite_cannot_keep_an_old_member_digest` |
| A cancellation landing between `Popen` and process registration was overwritten by `running`; the child ran to completion | creation, registration and the status transition share one critical section | `test_cancellation_cannot_slip_between_creation_and_registration` |
| A duplicate run id started a second evaluator that then had no ledger entry | uniqueness is checked before the child starts | `test_a_duplicate_run_id_is_refused_before_the_evaluator_starts` |
| Queued (API) evaluator children inherited `WORKBENCH_LEDGER_KEY` and `WORKBENCH_SECRET_*` | the queue filters the child environment | `test_queued_job_children_get_no_operator_secrets` |
| CLI evaluator children received the same secrets, and wrote temporary files into the shared host temp dir | the CLI filters secrets too and gives the child a run-scoped `TMPDIR` at mode `0700` | `test_the_evaluator_child_gets_no_operator_secrets_and_a_private_temp_dir` |
| `POST /api/jobs` queued the raw model evaluator, bypassing the invocation ledger entirely | the handler queues the audited orchestrator with an explicit run id | `test_the_api_queues_the_audited_orchestrator_not_the_raw_evaluator` |
| The documented first-run `--verify-gpu` workflow deadlocked: the strict preflight refused the pending GPU before the stage that probes it | the preflight immediately preceding an authorized probe takes `--allow-gpu-pending`, tolerating only that one state; a failed probe still stops the run | E2E-13, `test_e2e13_gpu_declaring_model_is_blocked_while_verification_is_pending` |
| A fallback-key-only operator could never persist a signed capability (`_gpu_cache_key` read a key file it never created, and the fix first shipped without `import tempfile`) | the key file is created on first use, as the ledger does | `test_the_fallback_key_is_created_so_a_capability_can_be_signed` |
| A directory stage input was fingerprinted by its own metadata, so a changed checkpoint member was invisible to resume | directory inputs are inventoried with per-file content digests under a recorded budget | `test_fp13_same_size_member_rewrite_invalidates_a_directory_input` |


### Release-closure pass (SCI-01, RES-01..03, JOB-01, SEC-01, GPU-01, PREP-01..05, TEST-01/02)

| Defect | Fix | Test |
|---|---|---|
| `fashioniq_val_split` was evaluated with the reference eligible, and only internal protocol aliases were accepted | one protocol table defines gallery and reference policy; unknown names are refused | `tests/unit/test_fashioniq_dispatch.py` drives the production entry point over a 53-image gallery |
| A directory could be reported content-verified when members were never read (size threshold, unreadable file, unread directory symlink) | entries and trees carry explicit evidence classes; only content evidence authorizes verified resume; every regular file is content-hashed by streaming | `test_release_blocker_production_stage_fingerprint.py`, `test_scientific_input_identity.py` |
| A retargeted directory symlink was invisible to the fingerprint | symlinks are inventoried with their targets, including symlinks to directories | `test_scientific_input_identity.py::test_data_id_08_...` |
| Image content, correction dictionaries, resized trees, prepared pickles and auxiliary captions were not evaluation inputs | the input graph is built from the model's layout, preparation contract, registered assets, evaluator implementation and recipe | `test_scientific_input_identity.py` (DATA-ID-01..12) |
| A non-cooperative evaluator survived cancellation while the job reported `cancelled` | liveness comes from the process group, escalation always runs, group id captured at dispatch, bounded wait outside the queue lock | `test_process_group_cancellation.py`, acceptance scenario F |
| The evaluator environment was the operator's whole environment minus two names | allowlist with a deny-prefix rule that a passthrough declaration cannot override | `test_credential_boundary.py` |
| A GPU capability was not bound to the host or driver | machine, driver and visible devices are part of the key; the record states its revalidation rule | `test_gpu_capability_host_identity.py` |
| The image-extension check looked in `images/` for every model | it follows the layout's declared image directory | `tests/unit/test_fashioniq_protocols.py` |
| The fixed dataset link was staged for CSMCIR only | one registry declaration drives staging, verification and source-artifact approval for CSMCIR, SPRC and both CLIP4Cir variants | `test_prepare_dataset.py` |
| CLVC-Net claimed a shared checkpoint policy | the per-category local/global structure is recorded, identities UNVERIFIED | registry contract |
| An `assert ... or True` made a test unfalsifiable | replaced by an assertion on the dispatched argv | `test_phase1_acceptance_scenarios.py` |
| The E2E never exercised the HTTP stack | ASGI-level tests over the real application | `test_http_api_layer.py` |
