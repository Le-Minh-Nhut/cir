# PC reproduction handoff

Primary operator path: [AUTOPILOT_GUIDE.md](AUTOPILOT_GUIDE.md). Use the checked-in example profile only for read-only planning; copy it to a private PC-specific YAML and configure dataset, persistent paths, model selection, checkpoint policy, budgets, and permissions before any real run. [MODEL_STATUS_SPEC.md](MODEL_STATUS_SPEC.md) records per-model source, checkpoint, assets, capability, and blockers.

**Revision identity:** `<PRODUCTION_REVISION>` is the revision whose checks ran; `<FINAL_HEAD>` is the branch tip, which differs only in documentation. Production bytes are identical between them.

## Read-only host checks

```bash
./workbench/run.sh autopilot --help
.venv/bin/python workbench/scripts/doctor.py --scope workbench --json
.venv/bin/python workbench/scripts/prepare_model.py --list --json
.venv/bin/python workbench/scripts/pipeline.py prepare --model csmcir --dry-run
.venv/bin/python workbench/scripts/download_auxiliary_assets.py --model tme --dry-run
./workbench/run.sh autopilot --config workbench/configs/pc_autopilot.example.yaml --dry-run
```

A workbench-scope doctor may exit 2 when local dependencies are pending; its JSON is readiness evidence, not model evaluation. The checked-in example autopilot profile selects zero models, so its dry-run only confirms CLI/config parsing.

On this host the workbench-scope doctor returned 0, while `prepare_model.py --list --json` returned 1 because models have blockers; `manage_environment.py --list --json` likewise returned 1 because model interpreters are unconfigured. Those exit codes accompany valid read-only JSON diagnostics, not failed execution.

## One run/resume path

On the configured target GPU PC, use same profile for initial run and resume:

```bash
./workbench/run.sh autopilot --config /path/to/pc_autopilot.yaml --apply --resume
```

This live run was not executed here. Preconditions: configured CUDA-capable target and compatible driver, canonical FashionIQ inputs, verified selected checkpoint and model interpreter, sufficient storage, and explicit permissions for each requested side effect. Missing registered assets may require network permission. Live Docker deployment is also unexecuted; it requires Docker Engine/Compose, configured GPU passthrough where used, and the deployment stack.

TME's EVA-ViT-g and BLIP-2 weights use `download_auxiliary_assets --model tme` (or the authorized Autopilot auxiliary-assets stage), staging under `workbench/third_party/TME/weight/`. `mkdir` does not install them, and `WORKBENCH_TME_ASSET_EVA_VIT_G` is undeclared. Validation-feature generation is separate CUDA-only preparation: authorize `allow_preparation` and `allow_gpu_preparation` (optional Autopilot config key, default `false`); reserve about 10.47 GiB. This cache was not generated here.

`doctor.py` accepts `--checkpoint-policy {strict_clean,include_flagged_conditions}`. Runtime-preflight checks the checkpoint the run selected. TME 20% remains `CHECKPOINT_UNAVAILABLE`; do not substitute Air-Know 50% for the 20% condition. Per-model unresolved facts and next actions remain in [MODEL_STATUS_SPEC.md](MODEL_STATUS_SPEC.md).
