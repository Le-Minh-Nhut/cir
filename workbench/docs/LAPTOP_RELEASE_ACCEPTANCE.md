# Laptop Release A — Engineering Acceptance

Release A covers laptop-verifiable orchestration and evidence handling. Real GPU inference is Release B; use the [PC Reproduction Handoff](PC_REPRODUCTION_HANDOFF.md) and its primary [Autopilot Operator Guide](AUTOPILOT_GUIDE.md).

## Scope and acceptance status

The laptop work established guarded registry-driven execution stages, explicit authorization, content-based resume, environment isolation checks, model-specific preparation, checkpoint bundle identity, source-backed aggregate extraction, schema-v2/index integrity, and workflow status reporting. It does not establish that external checkpoints, FashionIQ assets, model environments, or a real GPU are ready.

| Criterion | Status |
|---|---|
| Authorization enforced for selected stages | PASS |
| Registry-derived execution dependencies enforced | PASS |
| Independent models can continue after another model blocks | PASS |
| Resume rejects missing/corrupt outputs and changed inputs | PASS |
| Environment verification does not overstate readiness | PASS |
| Checkpoint file/directory/bundle identity handled | PASS |
| Aggregate and per-query schemas remain distinct | PASS |
| Workflow reports success, partial, failure, or blocked outcomes | PASS |
| Real GPU inference required | NOT REQUIRED for Release A |

## Laptop / PC boundary

- **Finished on laptop:** orchestration, verification, provenance, authorization, resume, and offline synthetic acceptance.
- **Pending on GPU PC:** official evaluator runs on real checkpoints, real aggregate metric parity, and model-specific GPU capability verification.
- **Pending external evidence/assets:** unresolved checkpoint mappings and required model assets remain blockers as recorded in registry and [UPSTREAM_AUDIT.md](UPSTREAM_AUDIT.md).
- **Not available yet:** per-query ranking export for external models; aggregate output alone is not schema-v2 query evidence.

## Release B operator command

Use [Autopilot Operator Guide](AUTOPILOT_GUIDE.md) for GPU-PC prerequisites, configuration, authorization, checkpoint policy, GPU selection, budgets, ledger key, environment groups, resume, reports, and blockers. Its one primary execution command is:

```bash
./workbench/run.sh autopilot --config /path/to/pc_autopilot.yaml --apply --resume
```

Every selected stage still requires its config permission; blocked registry entries remain blocked. Do not instruct operators to run individual stage scripts as a reproduction procedure. The report's status and blocker rows, not process exit alone, determine model-level outcome.

## Existing engineering evidence

Detailed historical regression and test evidence remains in repository tests and prior engineering records. This acceptance summary makes no new test-run claim. No full test suite, linter, or formatter was run for this documentation change.
