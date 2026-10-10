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
