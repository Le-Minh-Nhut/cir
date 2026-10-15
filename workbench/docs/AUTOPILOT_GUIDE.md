# Autopilot operator guide

Primary GPU-PC runbook for guarded FashionIQ reproduction. Autopilot orchestrates pinned-source sync, checkpoint acquisition, preparation, environment planning and provisioning, GPU capability verification, evaluation, validation, and index rebuild. It does not make a blocked registry entry runnable, invent assets, or claim paper parity.

## 1. GPU-PC prerequisites

Use the CIR checkout on the GPU PC with workbench runtime dependencies already available. Provide a Linux host with a CUDA-capable GPU, persistent disk storage, and operator-controlled secret storage. Keep each upstream's model environment separate from the workbench environment; never install upstream dependencies into the workbench backend environment.

FashionIQ root must already exist and contain the standard `captions/`, `image_splits/`, and `images/` directory layout. Some models additionally require resized images (`resized_image/{category}`), dictionaries, caches, captions, or other model-specific assets. Autopilot downloads registered auxiliary assets (for example CSMCIR Qwen captions) only when required permissions (`allow_network` + `allow_large_downloads`) are granted. Other required local assets must be present with verified provenance; CSMCIR `COT_ours2/fashioniq` captions have no verified author download source and must be supplied locally. Do not substitute assets.

Offline reuse: a source repository, checkpoint, or auxiliary asset that is already present and verified is validated and reused with NO network permission. Acquisition still requires explicit network permission. Checkpoint downloads verify size and SHA-256 on a temporary file (`.part`) and only then publish via atomic replacement (`os.replace`), so a rejected or interrupted download never leaves a corrupt file that blocks recovery.

## 2. Set dataset, artifact, checkpoint, results, and upstream roots

Create PC-specific YAML from `workbench/configs/pc_autopilot.example.yaml`. `paths` requires exactly `registry`, `dataset_root`, `artifacts`, `checkpoints`, `results`, and `third_party`; unknown keys are rejected. Relative paths resolve from the config file's directory. `registry` must name an existing file and `dataset_root` an existing directory; other paths need an existing directory parent.

Set `paths.dataset_root` to the FashionIQ root itself, not `images/` or `captions/`. Set `paths.artifacts` to persistent storage; it owns run state, reports, evidence ledger, completion proofs, GPU capability records, environment plans, `environment_assignment.json`, and the derived DuckDB index. Set `paths.checkpoints` and `paths.results` in persistent storage too. Set `paths.third_party` to the location for pinned upstream checkouts. Upstream requirement files are read from this configured upstream root (`paths.third_party`), never a hardcoded default. The CIR code checkout is `CIR_REPO_ROOT`; it supplies workbench code and registry defaults, separate from the configured upstream-clone root.

Example config (replace PC paths; preserve required keys):

```yaml
schema_version: 1
selected_models: all_registered
checkpoint_policy: strict_clean
checkpoint_overrides: {}
paths:
  registry: /opt/cir/workbench/registry/models.yaml
  dataset_root: /data/FashionIQ
  artifacts: /data/cir-workbench
  checkpoints: /data/cir-workbench/checkpoints
  results: /data/cir-workbench/results
  third_party: /data/cir-upstreams
budgets:
  max_download_bytes: 10737418240
  max_models: 256
  top_k: 200
permissions:
  allow_network: true
  allow_large_downloads: true
  allow_env_install: true
  allow_preparation: true
  allow_gpu_eval: true
  allow_index_write: true
```

`selected_models` accepts `all_registered` or a non-empty list of exact registry IDs. Prefer an explicit list of models whose evidence and prerequisites you reviewed. `all_registered` includes blocked entries; it does not make them runnable. `max_models` must cover the selection. Keep selected IDs unchanged after execution starts so resume can use existing state.

## 3. Install or point at conda or mamba

Put `conda`, `mamba`, or `micromamba` on the operator `PATH` before launching autopilot. The environment planner checks source-backed Python and package constraints read directly from the configured upstream root (`paths.third_party`). It groups compatible models, searches `conda-forge`, `pytorch`, and `defaults`, and provisions shared conda environments when authorized (`allow_env_install: true`).

With `allow_env_install: false`, the planner inspects existing environments without installing new packages; missing or unverified environments will block evaluation. Pre-existing environments can also be specified manually: setting a model's registry-declared `WORKBENCH_<MODEL>_PYTHON` variable points that model to an exact interpreter executable, taking precedence over automatic planner group assignments.

## 4. Authorize downloads and other side effects

Permissions live in config, not CLI flags. Full workflow stage capabilities:

- source sync: `allow_network` + `allow_preparation` (verified offline with no permission if already clean and pinned);
- checkpoint acquisition: `allow_network` + `allow_large_downloads` (verify-only with no permission if already valid);
- registered auxiliary-asset acquisition: `allow_network` + `allow_large_downloads` (verify-only with no permission if already valid);
- environment provisioning: `allow_env_install` + `allow_network` (when creating shared conda groups);
- mutating model preparation: `allow_preparation` + `allow_network`;
- GPU verification: `allow_gpu_eval` (runs only when probing; reuses still-valid capability with no permission);
- evaluation: `allow_preparation` + `allow_gpu_eval`;
- result validation/index rebuild: `allow_index_write`.

A selected stage missing any declared capability is blocked as `BLOCKED_AUTHORIZATION`; missing permission never means approval. Set unapproved permissions to `false` and expect those stages to be refused. Downloads use registry-declared URLs only; unresolved URLs, mappings, or required author assets stay blocked.

## 5. Choose checkpoint policy

The config supports two optional checkpoint keys: `checkpoint_policy` and `checkpoint_overrides`.

`checkpoint_policy` controls handling of recorded training conditions:
- `strict_clean` (default): evaluates only weights whose recorded training condition is exactly 0% noise. A model that publishes only noise-trained checkpoints (such as Air-Know, ConeSep, HABIT, and INTENT) is BLOCKED (`BLOCKED_CHECKPOINT`) rather than silently evaluated.
- `include_flagged_conditions`: runs those checkpoints as an exploratory secondary cohort, with each model badged by the condition the registry records.

`checkpoint_overrides` is an operator-resolved mapping (`model_id` -> list of checkpoint IDs). An override wins over the policy and is badged as an override rather than clean.

Checkpoint downloads verify size and SHA-256 on a temporary file (`.part`) and only publish via atomic replacement once verified. A rejected or interrupted download never leaves a partial file that blocks recovery.

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

Set the PC process environment's `CUDA_VISIBLE_DEVICES` to the device(s) intended for evaluation before launch. GPU selection is not a config key. Keep visibility unchanged between capability verification and evaluation: records bind to host, driver, interpreter/dependencies, and visible devices. Stale or differently scoped capability blocks CUDA evaluation.

GPU verification is an explicit authorized stage (`gpu-verification`) in the autopilot pipeline using the assigned model interpreter, preserving `CUDA_VISIBLE_DEVICES`. A still-valid capability is reused without re-probing; a stale capability is refused and re-probed. The operator no longer runs `evaluate_models.py --verify-gpu-only` by hand.

## 7. Set resource and download budgets

`budgets` accepts exactly three integer fields: `max_download_bytes` (non-negative), `max_models` (positive), and `top_k` (positive). `max_models` must cover the selected count. `top_k` sets retrieval depth. `max_download_bytes` caps registry-sized checkpoint acquisition across selected models. The example budget is 10 GiB (`10737418240`) because the three declared LIMN checkpoints total 5,376,030,471 bytes (~5.01 GiB), plus auxiliary assets and safety margin. Existing verified checkpoint files do not consume budget. Unknown required checkpoint size prevents budget verification and blocks acquisition.

Autopilot has no CPU, RAM, VRAM, runtime, or concurrency budget fields. Apply those through PC/scheduler policy and model selection; do not add unsupported keys.

## 8. Configure the HMAC ledger key

Before a real run, provide a stable high-entropy `WORKBENCH_LEDGER_KEY` via PC secret manager or process environment. Keep it outside repository, config, artifacts, logs, and model environments. Reuse the same key for later verification; changing or losing it makes existing HMAC evidence unverifiable. Evaluator child processes do not receive the key.

Without an environment key, a documented fallback creates `workbench_ledger_key` under `$XDG_CONFIG_HOME/cir/` (default `~/.config/cir/`) with mode `0600` in a `0700` directory -- outside the repository, because the repository root is visible to the official evaluator child. Prefer external secret storage on shared or backed-up PCs. An empty key or existing ledger without its key fails closed. See [AUTOMATION_RUNBOOK.md](AUTOMATION_RUNBOOK.md) for trust boundaries.

## 9. Environment grouping and provisioning

The environment planner groups compatible models and PROVISIONS the shared conda environments it plans when authorized (`allow_env_install: true`):

- **Prefix isolation:** one prefix per compatible group under `<paths.artifacts>/envs/<environment_id>`.
- **Authoritative assignment:** writes `<paths.artifacts>/environment_assignment.json`, mapping each model to its assigned environment group, prefix, interpreter, and provisioning state.
- **Configured source root:** source requirements are read from the configured upstream checkout root (`paths.third_party`), never a hardcoded default.
- **Two-phase install:** for each group, conda-resolvable constraints are solved and installed first (`conda create --prefix <prefix> ...`), followed by pip-only packages installed into that prefix (`python -m pip install ...`).
- **Four recorded states:**
  1. `CONSTRAINTS_COMPATIBLE`: constraints parsed and merged across group members without contradiction;
  2. `SOLVER_COMPATIBLE`: solver (conda/mamba) verified satisfiability for group constraints;
  3. `INSTALLED`: conda prefix and pip-only packages successfully installed;
  4. `RUNTIME_VERIFIED`: `pip check` passed, exact package versions verified, and per-member evaluator imports verified with the assigned interpreter.
- **Evaluation gate:** ONLY `RUNTIME_VERIFIED` permits evaluation. Any lower state blocks the model with `BLOCKED_ENVIRONMENT`.
- **Deterministic reuse:** a previously `RUNTIME_VERIFIED` group with an unchanged digest and interpreter is reused with no reinstall.
- **Splits:** when models have conflicting requirements (such as incompatible Python versions or contradictory package version ranges), the planner cannot combine them; it records the split reasons and separates them into distinct groups.

Explicit `WORKBENCH_<MODEL>_PYTHON` assignments take precedence over planner groups when set by the operator.

## 10. Launch autopilot

This is the one primary execution command. It uses config grants and resumes only verified completed stages. Keep config, selection, paths, GPU visibility, and key stable across invocations.

```bash
./workbench/run.sh autopilot --config /path/to/pc_autopilot.yaml --apply --resume
```

Pipeline stages executed in order by autopilot:
1. `doctor` (workbench environment preflight);
2. `dataset` (FashionIQ canonical layout validation);
3. `sync` (per model: pinned official upstream source checkout);
4. `environment` (per model: environment inspection/creation when `allow_env_install`);
5. `preparation` (per model: canonical dataset linking, one owner per shared link);
6. `checkpoint` (per model: download with temporary-file verification, or verify-only);
7. `auxiliary-assets` (per model: registered auxiliary evaluation assets);
8. `environment-plan` (shared conda environment planning and provisioning);
9. `gpu-verification` (per model: authorized probe using assigned interpreter, preserved `CUDA_VISIBLE_DEVICES`);
10. `runtime-preflight` (per model: preflight gate before evaluation);
11. `evaluation` (per model: guarded official evaluation);
12. `validation-index` (global: validate outputs and rebuild DuckDB serving index).

`--apply` executes side-effecting stages; `--resume` is safe on first invocation. Autopilot continues independent models when one blocks or fails. Successful process exit does not imply all selected models completed; inspect reports.

## 11. Resume

After fixing a reported local cause, rerun the same primary command with `--resume`:

```bash
./workbench/run.sh autopilot --config /path/to/pc_autopilot.yaml --apply --resume
```

Resume is bound to installed dependency content (not just the interpreter path and requirements files): the interpreter binary, environment marker files, and package directory modification times are fingerprinted. A rewritten dependency invalidates the affected evaluation while unrelated ones stay reusable. Content-verified stages are reused without redundant computation. Preserve selected IDs, artifact roots, and HMAC key across retries.

## 12. Read reports

All paths below are under `paths.artifacts`:

- `autopilot_execution_plan.json`: selected IDs, stage plan, environment grouping, pre-execution blockers;
- `environment_plan.json`: group constraints, evidence, solver status;
- `environment_assignment.json`: authoritative environment groups, prefix paths, assigned interpreters, and four-state provisioning statuses;
- `autopilot_pipeline_state.json`: stage state used for resume;
- `reports/autopilot_summary.md` and `reports/autopilot_summary.json`: outcome, execution status, and category recall table;
- `reports/autopilot_results.csv`: machine-readable CSV with status, blockers, and metric columns;
- `reports/blockers.json`: machine-readable blocker details;
- `reports/run_manifest.json`: selected IDs, config digest, outcome, pipeline exit code.

The consolidated report records real aggregate metrics:
- Run identity: `run_id`, `model_id`, `method_name`, `native_protocol`;
- Environment and weights: `environment_group`, `checkpoint_id`, `checkpoint_provenance`;
- Category Recall: Recall@10 and Recall@50 for `dress` (`dress_r10`, `dress_r50`), `shirt` (`shirt_r10`, `shirt_r50`), and `toptee` (`toptee_r10`, `toptee_r50`);
- Macro Recall: `macro_r10`, `macro_r50`;
- Reference comparison: `paper_r10`, `paper_r50`, `paper_mean`, and `measured_difference` (observed minus paper);
- Audit trail: `evaluation_duration_seconds`, `completion_proof` path, `aggregate_report` path;
- Evaluation kind: `SYNTHETIC` or `OFFICIAL_VERIFIED`.

`OFFICIAL_VERIFIED` records how the run was produced (a real execution, not a synthetic replay);
it is independent of the paper-gate verdict: whether the checkpoint-to-paper mapping was verified
lives in `benchmark_gate` (`checkpoint_mapping_verified` / `PASS_METRICS_ONLY_MAPPING_UNVERIFIED`),
not in this field. Known drift point: the `"VERIFIED_METADATA"` mapping literal is compared at a
dozen sites across `workbench/`; `checkpoint_selection._VERIFIED_MAPPING` exists and should be the
single constant if that vocabulary ever changes.

Metrics conventions:
- Absent values are `null` with a recorded reason (`missing_metrics_reason`), never zero.
- Evaluated models are marked `aggregate_only: true`; per-query export unavailability is explicitly stated (`per_query_export_available: false`).
- Model scientific status (`COMPLETED`, `COMPLETED_WITH_LIMITATIONS`, `BLOCKED_*`, `FAILED_*`) is independent of global index status (`global_index_status`: `COMPLETED`, `PARTIAL`, `NOT_APPLICABLE`).
- Global index rebuild applies only when per-query results are available.

## 13. Handle a blocked model

Read the model row in `reports/autopilot_summary.md` or CSV, matching entry in `reports/blockers.json`, then stage state. Fix only the verified local cause:
- Authorization blocker: grant missing permission in config `permissions`.
- Dataset or preparation blocker: check FashionIQ canonical directory layout and model-specific assets.
- Checkpoint blocker: check registry-linked URL, file hash, or training noise condition (`checkpoint_policy`).
- Environment blocker: check solver compatibility, conda/mamba availability, and whether `allow_env_install` was granted.
- GPU blocker: check `CUDA_VISIBLE_DEVICES`, driver, PyTorch CUDA capability for the assigned interpreter.
- Source blocker: check git clone status and commit pin in `paths.third_party`.

If the registry identifies an unresolved author checkpoint, provenance, replay entrypoint, or model asset, keep the model blocked and exclude it from the selected-model list until authoritative evidence resolves it. Never substitute an arbitrary checkpoint or claim a model ran. Independent models may complete; report partial success as partial, not full reproduction.

## Optional diagnostics

These commands inspect parser options, saved state, or interpreter readiness; none replaces the primary execution command.

```bash
# Optional diagnostic: show accepted autopilot options.
./workbench/run.sh autopilot --help

# Optional diagnostic: read saved autopilot status as JSON.
./workbench/run.sh autopilot --config /path/to/pc_autopilot.yaml --status --json

# Optional diagnostic: read latest autopilot report as JSON.
./workbench/run.sh autopilot --config /path/to/pc_autopilot.yaml --report --json

# Optional diagnostic: resolve plan without mutations or subprocesses.
./workbench/run.sh autopilot --config /path/to/pc_autopilot.yaml --dry-run

# Optional diagnostic: inspect registered model interpreter readiness.
python workbench/scripts/manage_environment.py --list --json
```

For research status and protocol definitions, see [../README.md](../README.md), [UPSTREAM_AUDIT.md](UPSTREAM_AUDIT.md), and [RESULT_SCHEMA.md](RESULT_SCHEMA.md).
