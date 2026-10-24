# CIR Failure Analysis Workbench

Local tooling for FashionIQ composed-image-retrieval failure analysis. Official upstream evaluators remain authoritative; the workbench records pinned-source/checkpoint facts, guards execution, validates canonical artifacts, builds a local DuckDB serving index, and presents evidence in a browser. It does not implement or substitute CIR models.

> **Status boundary:** no real evaluation, checkpoint inference, or target-PC GPU validation has run on this host. Existing LIMN and ENCODER reports document earlier measured runs on an RTX 3050 Laptop (`sm_86`), not this current checkout's release status or an `sm_120` PC run. Paper metrics are registry sanity references, never local metrics. Mock artifacts are development data, never research evidence.

## Moving to the GPU PC

| Document | What it answers |
| --- | --- |
| [docs/FINAL_PC_READINESS.md](docs/FINAL_PC_READINESS.md) | prerequisites, the single setup/run/resume path, and what a run produces |
| [docs/PC_TRANSFER_CHECKLIST.md](docs/PC_TRANSFER_CHECKLIST.md) | what to copy, rebuild, and authorize |
| [docs/AUTOPILOT_GUIDE.md](docs/AUTOPILOT_GUIDE.md) | primary operator guide and supported commands |
| [docs/MODEL_STATUS_SPEC.md](docs/MODEL_STATUS_SPEC.md) | current 17-model evidence, capability, and blocker table |
| [docs/AUTOMATION_RUNBOOK.md](docs/AUTOMATION_RUNBOOK.md) | automation, resume, and evidence contract |

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

The workbench keeps upstream model environments isolated from the workbench environment. Autopilot provisions shared conda environments when authorized (`allow_env_install: true`) and records `CONSTRAINTS_COMPATIBLE`, `SOLVER_COMPATIBLE`, `INSTALLED`, and `RUNTIME_VERIFIED`; only `RUNTIME_VERIFIED` permits evaluation. Existing verified environments are reused. See [Autopilot Operator Guide](docs/AUTOPILOT_GUIDE.md).
## Local configuration

Copy `workbench/config/workbench.env.example` to `workbench/config/workbench.env`. Explicit CLI settings take precedence for paths and ports. For per-model interpreter and asset-location settings, process environment wins over `workbench.env`; absent an override, runtime uses a `RUNTIME_VERIFIED` provisioning assignment where available. Every registry-declared setting name is accepted, including `WORKBENCH_TME_PYTHON` and `WORKBENCH_PTHA_MTST_PYTHON`. Relative paths resolve against `CIR_REPO_ROOT`. Do not put secrets in this file.

For GPU-PC reproduction, use dedicated autopilot YAML config. It owns FashionIQ, artifact, checkpoint, result, registry, and upstream roots; do not confuse the CIR code checkout (`CIR_REPO_ROOT`) with the configured upstream-clone root. The complete setup and schema are in [Autopilot Operator Guide](docs/AUTOPILOT_GUIDE.md).

## GPU-PC reproduction

[Autopilot Operator Guide](docs/AUTOPILOT_GUIDE.md) is the primary operator document. Configure its paths, budgets (example 10 GiB to cover LIMN checkpoints and auxiliary assets), checkpoint policy (`strict_clean` default vs `include_flagged_conditions`, and optional `checkpoint_overrides`), and explicit capabilities first. Offline reuse validates already present and verified assets with no network permission. Its single primary execution command is:

```bash
./workbench/run.sh autopilot --config /path/to/pc_autopilot.yaml --apply --resume
```

This command is accepted by the autopilot parser. Keep config, selected models, artifact roots, GPU visibility, and HMAC ledger key stable when resuming. Autopilot orchestrates stages in order (doctor, dataset, sync, environment, preparation, checkpoint, auxiliary-assets, environment-plan, gpu-verification, runtime-preflight, evaluation, validation-index), executes GPU verification as an explicit authorized stage, and writes consolidated summary, CSV, and blocker reports with real aggregate metrics (Recall@10/@50 across dress/shirt/toptee, macro, protocol, checkpoint provenance, duration, and completion-proof references). Successful process exit does not imply every selected model completed. Never run individual stage scripts as an alternative reproduction workflow.

Use `doctor.py --checkpoint-policy {strict_clean,include_flagged_conditions}` to inspect readiness under a chosen checkpoint policy. Autopilot runtime-preflight checks the checkpoint selection the run actually resolved; it does not treat a different registered condition as a substitute. Read the current per-model capability and blocker table in [docs/MODEL_STATUS_SPEC.md](docs/MODEL_STATUS_SPEC.md), then use [docs/FINAL_PC_READINESS.md](docs/FINAL_PC_READINESS.md) for the single setup/run/resume workflow.

TME's EVA-ViT-g and BLIP-2 weights are acquired through `python -m workbench.scripts.download_auxiliary_assets --model tme`, which stages them under `workbench/third_party/TME/weight/`. Creating that directory or setting an undeclared asset environment variable does not install either weight. TME's optional validation-feature cache is generated only by authorized CUDA preparation (`--allow-gpu-preparation`, or autopilot `allow_gpu_preparation: true`) and needs about 10.47 GiB; this host did not generate it.
### Optional diagnostics

These inspect CLI help or saved autopilot status; they do not replace the primary command.

```bash
# Show accepted commands/options. Same parser is used by the dispatcher.
./workbench/run.sh autopilot --help

# Read-only plan with the checked-in example profile.
./workbench/run.sh autopilot --config workbench/configs/pc_autopilot.example.yaml --dry-run
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
