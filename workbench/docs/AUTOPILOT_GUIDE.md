# Autopilot operator guide

Primary GPU-PC runbook for guarded FashionIQ reproduction. Autopilot orchestrates pinned-source sync, checkpoint acquisition, preparation, evaluation, validation, and index rebuild. It does not make a blocked registry entry runnable, invent assets, or install environments unless explicitly authorized. A completed run is not proof of paper-metric parity.

## 1. GPU-PC prerequisites

Use the CIR checkout on the GPU PC with workbench runtime dependencies already available. Provide a CUDA-capable host, persistent storage, and operator-controlled secret storage. Keep each upstream's model environment separate from workbench environment; never install upstream dependencies into workbench backend environment.

FashionIQ root must already exist and contain standard `captions/`, `image_splits/`, and `images/` layout. Some models additionally require resized images, dictionaries, caches, captions, or other model-specific assets. Autopilot downloads registered auxiliary assets (for example CSMCIR Qwen captions) only when required permissions are granted. Other required local assets must be present with verified provenance; CSMCIR `COT_ours2/fashioniq` captions have no verified author download source. Do not substitute assets.

## 2. Set dataset, artifact, and upstream roots

Create PC-specific YAML from `workbench/configs/pc_autopilot.example.yaml`. `paths` requires exactly `registry`, `dataset_root`, `artifacts`, `checkpoints`, `results`, and `third_party`; unknown keys are rejected. Relative paths resolve from config file's directory. `registry` must name existing file and `dataset_root` existing directory; other paths need an existing directory parent.

Set `paths.dataset_root` to FashionIQ root itself, not `images/` or `captions/`. Set `paths.artifacts` to persistent storage; it owns run state, reports, evidence ledger, completion proofs, GPU capability records, environment plans, and derived DuckDB index. Keep `paths.checkpoints` and `paths.results` in persistent storage too. Set `paths.third_party` to location for pinned upstream checkouts. CIR code checkout is `CIR_REPO_ROOT`; it supplies workbench code and registry defaults, separate from configured upstream-clone root.

Example config (replace PC paths; preserve every key):

```yaml
schema_version: 1
selected_models: all_registered
paths:
  registry: /opt/cir/workbench/registry/models.yaml
  dataset_root: /data/FashionIQ
  artifacts: /data/cir-workbench
  checkpoints: /data/cir-workbench/checkpoints
  results: /data/cir-workbench/results
  third_party: /data/cir-upstreams
budgets:
  max_download_bytes: 5368709120
  max_models: 256
  top_k: 200
permissions:
  allow_network: true
  allow_large_downloads: true
  allow_env_install: false
  allow_preparation: true
  allow_gpu_eval: true
  allow_index_write: true
```

`selected_models` accepts `all_registered` or non-empty list of exact registry IDs. Prefer explicit list of models whose evidence and prerequisites you reviewed. `all_registered` includes blocked entries; it does not make them runnable. `max_models` must cover selection. Keep selected IDs unchanged after execution starts so resume can use existing state.

## 3. Configure conda or mamba

Put `conda`, `mamba`, or `micromamba` on operator `PATH` before planning. Environment planner checks source-backed Python/package constraints, groups compatible models, and attempts real solves when solver exists. It searches `conda-forge`, `pytorch`, and `defaults`; channel gap is reported separately from package conflict. Solver plan is not an installed or runtime-verified environment.

Provision isolated environments from pinned upstream requirements before evaluation. Planner groups recommend sharing; they do not install grouped environments. Set each model's registry-declared `WORKBENCH_<MODEL>_PYTHON` variable to exact executable path. Compatible models may share one environment. Explicit per-model variables are respected instead of replaced by group assignment. With `allow_env_install: false`, existing environments are inspected only. Enable it only to authorize source-backed per-model environment creation; also grant `allow_network`. This is not installation of a shared planner group. Missing/unverified interpreters stay blocked; runtime checks cover Python, isolation, dependencies, and required CUDA capability.

## 4. Authorize downloads and other side effects

Permissions live in config, not CLI flags. Full workflow stage capabilities:

- source sync: `allow_network` + `allow_preparation`;
- checkpoint acquisition: `allow_network` + `allow_large_downloads`;
- registered auxiliary-asset acquisition: `allow_network` + `allow_large_downloads`;
- environment creation, only if intentionally enabled: `allow_env_install` + `allow_network`;
- mutating model preparation: `allow_preparation` + `allow_network`;
- evaluation: `allow_preparation` + `allow_gpu_eval`;
- result validation/index rebuild: `allow_index_write`.

A selected stage missing any declared capability is blocked as `BLOCKED_AUTHORIZATION`; missing permission never means approval. Set unapproved permissions to `false` and expect those stages to be refused. Downloads use registry-declared URLs only; unresolved URLs, mappings, or required author assets stay blocked.

## 5. Choose checkpoint policy

Autopilot config has no checkpoint-ID override. Review selected model's `checkpoint_variants` and bundle metadata in `workbench/registry/models.yaml`. Keep checkpoint training-noise condition distinct from clean (0%) evaluation policy; noisy-trained weights are not clean-trained weights. Never guess unresolved mappings, URLs, hashes, or provenance. When registry requires multiple category files or paired bundle, every declared member is required.

Autopilot cannot choose between ambiguous checkpoint variants on operator's behalf. Exclude such models until registry mapping and intended policy are resolved. Download-byte budget requires registry sizes; unknown required size blocks acquisition rather than being estimated. Locally computed digest records local evidence, not author-published hash.

Do not describe registry-blocked models as runnable:

- `ptha_mtst`: no verified FashionIQ fine-tuned checkpoint and no eligible source integration.
- `tgcir`: no source replay CLI; ViT-B/16 and preparation/cache requirements unresolved.
- `pair`: official FashionIQ mapping of variants unverified.
- `clvc_net`: no standalone replay entrypoint; checkpoint mapping and ResNet-50 asset unresolved.
- `sprc`: checkpoint/backbone mapping unresolved.
- Air-Know, ConeSep, HABIT, INTENT: registered weights are noise-trained, not clean-trained; no audited official command constructor.
- `hint`: checkpoint evidence exists; official command constructor remains unaudited.
- `encoder`: direct checkpoint file URL/hash unresolved and pinned source imports missing `datasets1.py`; OpenCLIP binary also required.
- `dcnet`: checkpoint mapping/hash, author-generated PKLs, and ResNet-50 asset unresolved.
- Combiner and CLIP4Cir: checkpoint mapping/pairing and RN50x4 base asset unresolved.
- `limn`: replay implemented, but three checkpoints, resized images, correction dictionaries, and verified environment must pass preflight.
- `csmcir`: required Qwen and `COT_ours2/fashioniq` captions must be present before autopilot. COT author source remains unverified; preserve provenance and do not substitute.

These are current known blockers, not a readiness guarantee for other models. Actual preflight remains authoritative.

## 6. Select the GPU

Set PC scheduler/process environment's `CUDA_VISIBLE_DEVICES` to device(s) intended for evaluation before launch. GPU selection is not a config key. Keep visibility unchanged between capability verification and evaluation: records bind to host, driver, interpreter/dependencies, and visible devices. Stale or differently scoped capability blocks CUDA evaluation. Toolkit or driver version alone does not prove PyTorch can use a device.

## 7. Set resource and download budgets

`budgets` accepts exactly three integer fields: `max_download_bytes` (non-negative), `max_models` (positive), and `top_k` (positive). `max_models` must cover selected count. `top_k` sets retrieval depth. `max_download_bytes` caps registry-sized checkpoint acquisition across selected models; existing checkpoint files do not consume budget. Unknown required checkpoint size prevents budget verification and blocks acquisition.

Autopilot has no CPU, RAM, VRAM, runtime, or concurrency budget fields. Apply those through PC/scheduler policy and model selection; do not add unsupported keys.

## 8. Configure the HMAC ledger key

Before real run, provide stable high-entropy `WORKBENCH_LEDGER_KEY` via PC secret manager/environment. Keep it outside repository, config, artifacts, logs, and model environments. Reuse same key for later verification; changing or losing it makes existing HMAC evidence unverifiable. Evaluator child processes do not receive key.

Without environment key, documented fallback creates `.workbench_ledger_key` under `CIR_REPO_ROOT` with mode `0600`. Prefer external secret storage on shared/backed-up PC. Empty key or existing ledger without its key fails closed. See [AUTOMATION_RUNBOOK.md](AUTOMATION_RUNBOOK.md) for trust boundaries.

## 9. Review environment groups and assignments

An applied run writes `<paths.artifacts>/autopilot_execution_plan.json`, including `environment_plan`; its environment stage writes `<paths.artifacts>/environment_plan.json`. Inspect each group's `members`, `constraints`, `evidence`, `split_reasons`, `unresolved`, `conda_env_path`, `interpreter`, `solver_verification_status`, `installation_status`, and `runtime_verification_status`; check `blocked` for models without a usable source/checkpoint basis. Planner groups compatible constraints; it does not install or certify the interpreter. Do not infer readiness from solver result or proposed path.

Registry-derived planning groups 16 models into 10 environments. `ptha_mtst` is blocked because it has neither source nor checkpoint; `tgcir` has a replay/prerequisite blocker. Actual solves/runtime states depend on PC. Explicit `WORKBENCH_<MODEL>_PYTHON` assignments take precedence; point compatible group members to same executable to share it. `manage_environment.py` can inspect per-model interpreter readiness, but that does not clear registry blockers.

## 10. Launch autopilot

This is the one primary execution command. It uses config grants and resumes only verified completed stages. Keep config, selection, paths, GPU visibility, and key stable on retries.

```bash
./workbench/run.sh autopilot --config /path/to/pc_autopilot.yaml --apply --resume
```

`--apply` executes; `--resume` is safe on first invocation. Autopilot records attempts, evaluates after dependencies and guards pass, and continues independent models while reporting blocked/failed rows. Successful process exit does not imply all selected models completed; inspect reports.

## 11. Resume

After fixing reported local cause, rerun same primary command. Content-verified stages can be reused; changed inputs rerun dependent work. Preserve selected IDs and artifact roots. Do not delete ledger, evidence, reports, or checkpoints to force retry. If blocker is unresolved registry fact, resolve it with authoritative provenance rather than editing local data to satisfy gate.

## 12. Read reports

All paths below are under `paths.artifacts`:

- `autopilot_execution_plan.json`: selected IDs, plan, environment grouping, pre-execution blockers;
- `environment_plan.json`: group constraints, evidence, and interpreter assignments;
- `autopilot_pipeline_state.json`: stage state used for resume;
- `reports/autopilot_summary.md` and `reports/autopilot_summary.json`: outcome and per-model status/blockers;
- `reports/autopilot_results.csv`: per-model status, blockers, missing permissions, run ID, metrics;
- `reports/blockers.json`: machine-readable blocker details;
- `reports/run_manifest.json`: selected IDs, config digest, execution flag, outcome, pipeline exit code.

`COMPLETED` means guarded evaluation and output validation passed. `COMPLETED_WITH_LIMITATIONS` means workflow completed without solver verification. `BLOCKED_*` means prerequisite/authorization stopped model; `FAILED_*` means execution/output validation failed. Autopilot's per-model metrics field is currently null; aggregate evaluator reports and completion proofs are separate evidence. Paper values are references, not reproduced results.

## 13. Handle a blocked model

Read model row in `reports/autopilot_summary.md` or CSV, matching entry in `reports/blockers.json`, then stage state. Fix only verified local cause. Authorization blocker needs explicit approval and corresponding config permission. Dataset/preparation blocker needs exact layout/assets. Checkpoint blocker needs registry-linked file/mapping and integrity evidence. Environment/GPU blocker needs assigned interpreter to pass runtime checks and capability to match current host/device. Source blocker needs exact registry-pinned clean checkout. Evaluator blocker means no audited usable command exists.

If registry identifies unresolved author checkpoint, provenance, replay entrypoint, or model asset, keep model blocked and exclude it from selected-model list until authoritative evidence resolves it. Never substitute nearby checkpoint or claim model ran. Independent models may complete; report partial success as partial, not full reproduction.

## Optional diagnostics

These inspect parser options, saved state, or interpreter readiness; none replaces primary execution command.

```bash
# Optional diagnostic: show accepted autopilot options.
./workbench/run.sh autopilot --help

# Optional diagnostic: read saved autopilot status as JSON.
./workbench/run.sh autopilot --config /path/to/pc_autopilot.yaml --status --json

# Optional diagnostic: inspect registered model interpreter readiness.
python workbench/scripts/manage_environment.py --list --json

# Optional diagnostic: probe LIMN GPU capability only; does not clear its checkpoint/data/environment blockers.
python workbench/scripts/evaluate_models.py --model limn --verify-gpu-only
```

The LIMN probe needs verified configured interpreter and usable visible GPU. Keep `CUDA_VISIBLE_DEVICES`, key, and interpreter unchanged for evaluation. A successful probe is not model readiness.

For research status and protocol definitions, see [../README.md](../README.md), [UPSTREAM_AUDIT.md](UPSTREAM_AUDIT.md), and [RESULT_SCHEMA.md](RESULT_SCHEMA.md).
