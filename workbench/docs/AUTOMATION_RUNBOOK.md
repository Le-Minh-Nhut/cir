# CIR Failure Analysis Workbench — Automation Runbook

Operator guide for running, resuming, troubleshooting, and verifying CIR model reproduction workflows.

## Principles

1. **Research integrity first**: Never fake model reproduction. Never report a noisy checkpoint as clean.
2. **Explicit authorization**: Environment installations and preprocessing require dedicated `--allow-*` flags.
3. **Fail closed**: Missing provenance, mismatched checksums, and unverified mappings block execution.
4. **Resumable pipelines**: Stage states and input fingerprints are persisted in `workbench/artifacts/pipeline_state.json`.

---

## 1. Master Pipeline Commands

The master orchestrator is `workbench/scripts/pipeline.py` (or `workbench/run.sh`).

### A. One-Command Complete Workflow (`all`)
Executes full reproduction pipeline for selected model: preflight doctor, dataset check, source sync, dataset link, checkpoint acquisition, auxiliary asset download, runtime preflight, evaluation, schema validation, and derived index rebuild.

```bash
# Planning mode (default for all, setup, reproduce without --apply):
python workbench/scripts/pipeline.py all --model csmcir

# Explicit dry run (zero mutations, prints stage plans):
python workbench/scripts/pipeline.py all --model csmcir --dry-run

# Authorized complete execution with resume capability:
python workbench/scripts/pipeline.py all --model csmcir --apply \
  --allow-network --allow-large-downloads --allow-env-install \
  --allow-preparation --allow-gpu-eval --resume
```

### B. Master Setup Workflow (`setup`)
Executes preparation stages: doctor, dataset, sync, environment, preparation, checkpoint, auxiliary-assets.

```bash
# Planning mode:
python workbench/scripts/pipeline.py setup --model csmcir

# Authorized execution with resume capability:
python workbench/scripts/pipeline.py setup --model csmcir --apply \
  --allow-network --allow-large-downloads --allow-env-install \
  --allow-preparation --resume
```
### C. Master Reproduction Workflow (`reproduce`)
Runs runtime preflight and guarded evaluation.

```bash
# Planning mode:
python workbench/scripts/pipeline.py reproduce --model csmcir

# Authorized evaluation with resume:
python workbench/scripts/pipeline.py reproduce --model csmcir --apply --allow-gpu-eval --resume
```
### D. Master Analysis Workflow (`analyze`)
Validates result artifacts and rebuilds DuckDB index.

```bash
python workbench/scripts/pipeline.py analyze
```

### E. Master Report Workflow (`report`)
Generates machine-readable JSON status of all models and prerequisites.

```bash
python workbench/scripts/pipeline.py report --model csmcir
python workbench/scripts/pipeline.py report
```

---

## 2. Standalone Stage Operator Commands

Every workflow stage has a working standalone CLI entrypoint:

| Stage | Script | Purpose |
|---|---|---|
| Doctor / Preflight | `workbench/scripts/doctor.py` | Inspect local prerequisites without mutations |
| Environment Manager | `workbench/scripts/manage_environment.py` | Inspect or create isolated model environments |
| Model Preparation | `workbench/scripts/prepare_model.py` | Inspect or execute model-specific preparation |
| Dataset Layout | `workbench/scripts/prepare_dataset.py` | Validate standard FashionIQ & prepare CSMCIR link |
| Source Sync | `workbench/scripts/sync_upstreams.py` | Clone or pin upstream git repositories |
| Checkpoint Download | `workbench/scripts/download_checkpoints.py` | Acquire registry model checkpoints |
| Auxiliary Assets | `workbench/scripts/download_auxiliary_assets.py` | Acquire verified auxiliary files (e.g. Qwen captions) |
| Model Evaluation | `workbench/scripts/evaluate_models.py` | Guard and execute official evaluator |
| Result Validation | `workbench/scripts/validate_results.py` | Validate canonical schema-v2 JSON artifacts |
| Index Rebuilder | `workbench/scripts/rebuild_index.py` | Rebuild DuckDB index from canonical JSON |
| Workbench UI Server | `workbench/scripts/serve_workbench.py` | Launch FastAPI backend and Vite frontend |
| Artifact Cleanup | `workbench/scripts/clean_generated.py` | Remove derived mock/index/log artifacts safely |

### Standalone Examples:

```bash
# Doctor inspection
python workbench/scripts/doctor.py --model csmcir --json
python workbench/scripts/doctor.py --all

# Environment management
python workbench/scripts/manage_environment.py --list
python workbench/scripts/manage_environment.py --model csmcir
# Creation requires explicit authorization:
python workbench/scripts/manage_environment.py --model csmcir --create --allow-env-install --dry-run

# Model preparation
python workbench/scripts/prepare_model.py --list
python workbench/scripts/prepare_model.py --model csmcir
# Execution requires explicit authorization:
python workbench/scripts/prepare_model.py --model csmcir --execute --allow-preparation --dry-run

# Checkpoints
python workbench/scripts/download_checkpoints.py --list
python workbench/scripts/download_checkpoints.py --model csmcir --dry-run
python workbench/scripts/download_checkpoints.py --model csmcir --verify-only

# Auxiliary assets
python workbench/scripts/download_auxiliary_assets.py --list
python workbench/scripts/download_auxiliary_assets.py --model csmcir --dry-run
python workbench/scripts/download_auxiliary_assets.py --model csmcir --verify-only

# Upstream source sync
python workbench/scripts/sync_upstreams.py --list
python workbench/scripts/sync_upstreams.py --model csmcir --verify-only

# Evaluation dry-run
python workbench/scripts/evaluate_models.py --model csmcir --dry-run
```

---

## 3. Resumability and Failure Recovery

Stage states are tracked in `workbench/artifacts/pipeline_state.json`.

- **Resume**: `--resume` skips stages that completed successfully with identical input fingerprints.
- **Force specific stage**: `--force-stage <STAGE_NAME>` forces re-execution of a stage even when marked complete.
- **Fingerprint invalidation**: If stage arguments, commands, or inputs change, the fingerprint mismatch automatically triggers re-execution.
- **Atomic state writes**: Stage transitions (PENDING → RUNNING → COMPLETE / FAILED) are written via temporary file replacement (`.tmp` → `pipeline_state.json`), preventing corrupted states on process interruption.
- **Failure isolation**: A failed stage preserves records of previously completed stages.

---

## 4. Current Model Reproduction Readiness

| Model | Status | Primary Blocker |
|---|---|---|
| **CSMCIR** | Setup Ready / Evaluation Guarded | COT_ours2 captions require manual author placement |
| **Air-Know** | Blocked | Only noisy (50%, 80%) checkpoints published; evaluator unaudited |
| **ConeSep** | Blocked | Only noisy (20%, 50%, 80%) checkpoints published; evaluator unaudited |
| **HABIT** | Blocked | Only noisy (20%, 50%, 80%) checkpoints published; evaluator unaudited |
| **INTENT** | Blocked | Only noisy (20%, 50%, 80%) checkpoints published; evaluator unaudited |
| **HINT** | Blocked | Pinned test.py CLI not audited |
| **ENCODER** | Blocked | Google Drive checkpoint unresolved; missing datasets1.py in upstream |
| **PAIR** | Blocked | pair-B1.pt vs pair-B2.pt checkpoint mapping unverified |
| **PTHA + MTST**| Blocked | No verified FashionIQ checkpoint published |
| **CLVC-Net** | Blocked | Pinned test.py has no standalone CLI loader; ResNet50 weights unverified |
| **DCNet** | Command Audited | Google Drive run directory hash unverified; ResNet50 weights unverified |
| **Combiner (noft)**| Blocked | Drive checkpoint mapping unresolved; CLIP RN50x4 unverified |
| **CLIP4Cir (fullft)**| Blocked | Paired checkpoint bundle unresolved; CLIP RN50x4 unverified |
| **TG-CIR** | Blocked | Pinned test.py has no standalone CLI; checkpoint mapping unverified |
| **SPRC** | Blocked | sprc_fiq.pt backbone mapping unverified; LAVIS asset unverified |
| **LIMN** | Replay Code Ready | Three category models required; CUDA 12.4 / OpenCLIP env unverified |

---

## 5. Distinction Between Artifact Tiers

1. **Mock Artifacts (`data_kind: "mock"`)**:
   Deterministic fixture runs generated for UI testing. **Never research evidence.**
2. **Official Aggregate Evaluation Reports (`aggregate_report.json`)**:
   Recorded in `workbench/artifacts/logs/<timestamp>_<model>_<checkpoint>/`.
   Contains verbatim `stdout.log`, `stderr.log`, `command.json`, paper sanity reference metrics, and `per_query_export_available: false`.
   **Does not constitute reproduced per-query predictions.**
3. **Canonical Schema-v2 Results (`workbench/artifacts/results/`)**:
   Strict JSON containing per-query retrievals, target ranks, and verified parity metrics. Ingested into DuckDB only after validation.
---

## 6. Aggregate Metric Extraction and Bundle Provenance

### A. Model-Specific Parsers (`workbench/backend/metrics_extraction.py`)
- **LIMN (`limn_structured_json`)**:
  Extracts macro R@10, R@50, mean and per-category metrics from structured stdout JSON. Requires all three categories (`dress`, `shirt`, `toptee`) to be complete; arithmetic category macro mean is validated.
- **CSMCIR (`csmcir_stdout_json`)**:
  Extracts `average_recall_at10`, `average_recall_at50`, `average_recall` and per-category recalls from official `validate_blip_csmcir.py` stdout JSON.
- **Unaudited Models**:
  Marked `AGGREGATE_PARSER_UNAVAILABLE`. No fabricated Recall values; exit code 0 alone does not imply metric parity.

### B. LIMN Bundle Provenance
A complete LIMN evaluation run records the exact individual paths and SHA-256 hashes of all three category models (`0_dress_best_model.pt`, `0_shirt_best_model.pt`, `0_toptee_best_model.pt`), plus a deterministic `bundle_manifest_digest`. Dress is never treated as the sole identity of the three-model benchmark experiment.

---

## 7. Experiment run identity and immutability

Each evaluation writes an immutable, run-scoped directory and report:

```text
workbench/artifacts/logs/<utc>_<model>_<checkpoint>_<rand>/
    command.json          # argv, cwd, digests, checkpoint manifest, interpreter
    stdout.log            # verbatim evaluator stdout
    stderr.log            # verbatim evaluator stderr
    aggregate_report.json # this run's extraction result
workbench/artifacts/reports/
    <model>_<checkpoint>_<run_id>_aggregate.json   # immutable, one per run
    <model>_<checkpoint>_latest.json               # atomic pointer, not the artifact
```

- A second, failed, or interrupted run never overwrites a previous success.
- `_latest.json` is convenience only; it is updated atomically and never replaces
  a run-scoped report.
- Resume identifies a completed evaluation through the latest pointer and the
  stage's declared output fingerprint.

## 8. Workflow ownership and locking

`run_stages` takes an exclusive `flock` on `pipeline_state.json.lock` while
executing non-dry-run work. Two concurrent workflows cannot both write state;
the second is refused with a clear lock error. The lock is released when the
process exits, including on crash, so no stale lock survives.

## 9. Compound capability policy

Side-effecting stages declare a set of required capabilities. All must be granted:

| Stage | Required capabilities |
|---|---|
| source sync | `allow_network`, `allow_preparation` |
| checkpoint acquisition | `allow_large_downloads`, `allow_network` |
| auxiliary assets | `allow_network`, `allow_large_downloads` |
| environment provisioning | `allow_env_install`, `allow_network` |
| model preparation (mutating) | `allow_preparation`, `allow_network` |
| evaluation | `allow_gpu_eval`, `allow_preparation` |

Verification-only stages declare no capabilities and remain read-only. `all`,
`setup` and `reproduce` require `--model MODEL_ID` or `--all-models`.
