# CIR Failure Analysis Workbench

Local tooling for FashionIQ composed-image-retrieval failure analysis. Official upstream evaluators remain authoritative; the workbench records pinned-source/checkpoint facts, guards execution, validates canonical artifacts, builds a local DuckDB serving index, and presents evidence in a browser. It does not implement or substitute CIR models.

> **Status boundary:** no official model evaluation or reproduction has run in this checkout. Paper metrics are registry sanity references, never local metrics. Mock artifacts are development data, never research evidence.

## Moving to the GPU PC

| Document | What it answers |
| --- | --- |
| [docs/FINAL_PC_READINESS.md](docs/FINAL_PC_READINESS.md) | what this commit supports, the prerequisites, the one command, and what a run produces |
| [docs/PC_TRANSFER_CHECKLIST.md](docs/PC_TRANSFER_CHECKLIST.md) | what to copy, what to rebuild, and each permission's effect |
| [docs/AUXILIARY_ASSET_AUDIT.md](docs/AUXILIARY_ASSET_AUDIT.md) | every external asset each model needs, its source line, and its precise blocker |
| [docs/AUTOMATED_DATA_PREPARATION.md](docs/AUTOMATED_DATA_PREPARATION.md) | what the workbench prepares by itself, and the rule that decides |
| [docs/READINESS_MATRIX.md](docs/READINESS_MATRIX.md) | the current per-model state, generated from the registry |

## Operating rules

- Use one exact registered protocol per run, comparison, cohort, and export.
- Keep model environments isolated from workbench backend/frontend dependencies.
- Sync only registry-pinned sources. Download only registry-declared author-linked URLs.
- A locally computed hash records local evidence; it does not become an official hash.
- Run an official evaluator unmodified and establish aggregate metric parity before any observation-only per-query export.
- Treat `run_id`, not `model_id`, as artifact identity.
- A registered model is not necessarily runnable. Unresolved checkpoints, source integrations, model assets, or evaluator commands remain blockers.

## Protocols

| ID | Exact label | Literature split | Gallery and target rank |
| --- | --- | --- | --- |
| `fashioniq_original_split` | FashionIQ - Original Split (Full Gallery, Reference Included) | `original` | Full ordered validation gallery; reference remains eligible. CSMCIR. |
| `fashioniq_full_gallery_ref_excluded` | FashionIQ - Original Full Gallery, Reference Excluded | `original` | Full ordered gallery; exclude reference before Recall@K. Air-Know, ConeSep, HABIT, INTENT, DCNet. |
| `fashioniq_val_split` | FashionIQ - Val Split (Pair-Union Gallery, Reference Excluded) | `val` | First-seen ordered validation reference/target union; remove reference before Recall@K. HINT, ENCODER, PAIR. |

All use `dress`, `shirt`, and `toptee`, validation captions/splits, and clean evaluation (`evaluation_noise_pct: 0`). `fashioniq_full_gallery_ref_excluded` is hidden by default only in browser protocol selection. Full evidence: [docs/PROTOCOL_AUDIT.md](docs/PROTOCOL_AUDIT.md).

CSMCIR consumes standard FashionIQ `images`; iLearn model sources use `resized_image/{category}` and category correction dictionaries. Those dictionaries are local and unverified. No automatic DQU-CIR substitution occurs. Model-specific layout checks remain authoritative.

## Repository and data ownership

`backend/`, `frontend/`, `registry/`, `scripts/`, and `docs/` are tracked code and evidence. Upstream clones, model assets, canonical result JSON, caches, and generated artifacts are local/ignored. Keep generated material out of commits. Canonical result JSON is source of truth; DuckDB is rebuildable derived data. Configure artifact storage separately from source checkout when appropriate.

## Environment policy

The workbench keeps upstream model environments isolated from the workbench environment. Autopilot plans and PROVISIONS shared conda environments when authorized (`allow_env_install: true`), allocating one prefix per compatible group under the artifacts root, reading source requirements from the configured upstream root (`paths.third_party`), and recording four distinct states (`CONSTRAINTS_COMPATIBLE`, `SOLVER_COMPATIBLE`, `INSTALLED`, `RUNTIME_VERIFIED`). Only `RUNTIME_VERIFIED` permits evaluation, and existing verified environments are reused without reinstall. Authoritative assignments are recorded in `environment_assignment.json`. See [Autopilot Operator Guide](docs/AUTOPILOT_GUIDE.md).
## Local configuration

`workbench/config/workbench.env.example` documents local service/path defaults. Explicit CLI settings take precedence over process environment, local env file, and built-in default. Relative paths resolve against `CIR_REPO_ROOT`. Do not put secrets in this file.

For GPU-PC reproduction, use dedicated autopilot YAML config. It owns FashionIQ, artifact, checkpoint, result, registry, and upstream roots; do not confuse the CIR code checkout (`CIR_REPO_ROOT`) with the configured upstream-clone root. The complete setup and schema are in [Autopilot Operator Guide](docs/AUTOPILOT_GUIDE.md).

## GPU-PC reproduction

[Autopilot Operator Guide](docs/AUTOPILOT_GUIDE.md) is the primary operator document. Configure its paths, budgets (example 10 GiB to cover LIMN checkpoints and auxiliary assets), checkpoint policy (`strict_clean` default vs `include_flagged_conditions`, and optional `checkpoint_overrides`), and explicit capabilities first. Offline reuse validates already present and verified assets with no network permission. Its single primary execution command is:

```bash
./workbench/run.sh autopilot --config /path/to/pc_autopilot.yaml --apply --resume
```

This command is accepted by the autopilot parser. Keep config, selected models, artifact roots, GPU visibility, and HMAC ledger key stable when resuming. Autopilot orchestrates stages in order (doctor, dataset, sync, environment, preparation, checkpoint, auxiliary-assets, environment-plan, gpu-verification, runtime-preflight, evaluation, validation-index), executes GPU verification as an explicit authorized stage, and writes consolidated summary, CSV, and blocker reports with real aggregate metrics (Recall@10/@50 across dress/shirt/toptee, macro, protocol, checkpoint provenance, duration, and completion-proof references). Successful process exit does not imply every selected model completed. Never run individual stage scripts as an alternative reproduction workflow.

The registry marks `ptha_mtst` blocked for lack of a verified FashionIQ checkpoint/source integration; `tgcir` lacks a replay CLI and required assets; `pair`, `clvc_net`, `dcnet`, Combiner/CLIP4Cir, and `sprc` have unresolved checkpoint mappings/assets; `encoder` has unresolved checkpoint acquisition and missing pinned-source module; Air-Know, ConeSep, HABIT, and INTENT have noise-trained variants rather than clean-trained weights and no audited command constructor; HINT's command is unaudited; LIMN still needs checkpoints, resized images, dictionaries, and verified environment; CSMCIR still needs provenance-verified COT captions. These are blockers, not runnable model claims. Current status: [docs/UPSTREAM_AUDIT.md](docs/UPSTREAM_AUDIT.md).
### Optional diagnostics

These inspect CLI help or saved autopilot status; they do not replace the primary command.

```bash
# Optional diagnostic: inspect accepted autopilot options.
./workbench/run.sh autopilot --help

# Optional diagnostic: read saved autopilot state as JSON.
./workbench/run.sh autopilot --config /path/to/pc_autopilot.yaml --status --json
```

## Mock and manually imported artifacts

Mock results are development fixtures and must never be presented as experimental evidence. Manually imported real artifacts require verified observation-only instrumentation and canonical schema-v2 JSON. Every result needs a unique `run_id`; strict-real validation requires source commit, checkpoint digest, command digest, and environment digest. Schema v1 is rejected rather than guessed or migrated. See [docs/RESULT_SCHEMA.md](docs/RESULT_SCHEMA.md).

## Model-specific requirements and blockers

`registry/preparation_contracts.yaml` is declarative. DCNet requires author-generated GloVe validation PKLs and unresolved ResNet-50 weights; do not regenerate PKLs with unseeded OOV vectors. CLVC-Net, TG-CIR, and LIMN have independent resized-image contracts. Required assets with unknown local location fail closed. Legacy pretrained assets remain local and require provenance checks. See [docs/LEGACY_REPRODUCTION_RUNBOOK.md](docs/LEGACY_REPRODUCTION_RUNBOOK.md).

For CSMCIR, standard FashionIQ and author-provided Qwen captions are separate from required `COT_ours2/fashioniq` captions. The authorized Autopilot auxiliary stage can download registry-declared Qwen assets; the COT author acquisition source remains unverified, so the model stays blocked by default. Because the published Qwen caption for a category and split is source-compatible with what the COT path reads — same basename, exact gallery-id coverage, the `Final_Caption` every executing consumer uses — an explicitly opted-in run may stage it as a labelled variant (`--download-auxiliary-assets --stage-caption-aliases`). That is `SOURCE_COMPATIBLE_EQUIVALENCE_UNKNOWN`, never official parity; see `workbench/docs/COT_OURS2_AUDIT.md`. Never invent or substitute files silently. Aggregate-only evaluation is not a per-query export.

## Results and analysis

Schema-v2 contract: [docs/RESULT_SCHEMA.md](docs/RESULT_SCHEMA.md). DuckDB normalizes runs, queries, and top results for serving only. UI supports dashboard, model registry, evaluation runs, sample exploration, comparisons, common failures, disagreement, failure analytics, annotations, and same-protocol cohorts. Cohorts organize evidence; they are not conclusions. A reproducible failure-analysis hypothesis does not prove an architectural cause.

## Safety and maintenance

The evidence ledger and run-bound completion proofs are described in [docs/AUTOMATION_RUNBOOK.md](docs/AUTOMATION_RUNBOOK.md). Preserve registry source pins and audit evidence together. Never edit derived database to hide invalid source artifacts. Cleanup is separate from reproduction and must not remove experiment JSON, checkpoints, source clones, or dataset files.

For exact model evidence, see [docs/UPSTREAM_AUDIT.md](docs/UPSTREAM_AUDIT.md). For GPU-PC setup, blockers, resume, and report interpretation, use [docs/AUTOPILOT_GUIDE.md](docs/AUTOPILOT_GUIDE.md).
