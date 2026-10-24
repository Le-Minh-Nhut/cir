# PC reproduction handoff

Primary operator path: [AUTOPILOT_GUIDE.md](AUTOPILOT_GUIDE.md). Use the checked-in example profile only for read-only planning; copy it to a private PC-specific YAML and configure dataset, persistent paths, model selection, checkpoint policy, budgets, and permissions before any real run. [MODEL_STATUS_SPEC.md](MODEL_STATUS_SPEC.md) records per-model source, checkpoint, assets, capability, and blockers.

**Revision identity:** `874ddcc` (`874ddccd7e3b869884b7f4ed25f7b9003d6b792d`) is the revision whose checks ran; `git rev-parse HEAD` on the branch tip (the docs-only successor) is the final head, which differs only in documentation. Production bytes are identical between them.

## Read-only host checks

```bash
./workbench/run.sh autopilot --help
.venv/bin/python workbench/scripts/doctor.py --scope workbench --json
.venv/bin/python workbench/scripts/prepare_model.py --list --json
.venv/bin/python workbench/scripts/pipeline.py prepare --model csmcir --dry-run
.venv/bin/python workbench/scripts/download_auxiliary_assets.py --model tme --dry-run
./workbench/run.sh autopilot --config workbench/configs/pc_autopilot.example.yaml --dry-run
```

## Which profile to use on the PC

| Profile | What it is | Use it when |
|---|---|---|
| `workbench/configs/pc_autopilot.pc_cohort.yaml` | **The cohort profile.** All 17 entries, `include_flagged_conditions`, the authorized 20% overrides resolved (`habit`/`intent`/`conesep` → `fiq_n02`, `ptha_mtst` → `cirr_transfer`, `encoder` → `fashioniq`), a 35 GiB budget derived from the registry, and every capability a cohort run needs including `allow_gpu_preparation` | **This is the one to run.** It is the only checked-in profile that combines a real selection with the authorization a cold run requires |
| `workbench/configs/pc_autopilot.all_available.yaml` | Every selectable verified variant, wider budget | You want the widest sweep rather than the authorized cohort |
| `workbench/configs/pc_autopilot.example.yaml` | A minimal, fully-locked example (`strict_clean`, no download, no env install, no preparation, no GPU) | You only want to confirm CLI/config parsing. **Do not** treat it as a runnable cohort: under `strict_clean` it leaves the noise-trained cohort blocked, and it authorizes nothing a cold run needs |
| `workbench/configs/pc_autopilot.pilot_{csmcir,limn}.yaml` | Single-model pilots | Debugging one model's path |

Copy whichever you choose outside the repository and set **all six** `paths` keys to absolute paths.

On this host the workbench-scope doctor returned 0, while `prepare_model.py --list --json` returned 1 because models have blockers; `manage_environment.py --list --json` likewise returned 1 because model interpreters are unconfigured. Those exit codes accompany valid read-only JSON diagnostics, not failed execution.

A workbench-scope doctor may exit 2 when local dependencies are pending; its JSON is readiness evidence, not model evaluation. The checked-in profiles select the whole registry: `selected_models: all_registered` resolves to all 17 entries, so a dry-run exercises real selection, checkpoint policy and override resolution. What a dry-run does NOT do is persist anything: it resolves the plan in memory, so `--status` (which reads the persisted state file) reports `NOT_RUN` with 0 models until an `--apply` run has written one. Use `--dry-run` to inspect the plan, and `--status` only after a run has started.

## One run/resume path

On the configured target GPU PC, use the same profile for the initial run and for resume:

```bash
# 1. Source only. This moves NO dataset, NO checkpoint, NO environment.
git fetch origin research/cir-failure-workbench
git checkout research/cir-failure-workbench     # the branch tip, not a pinned old revision

# 2. The cohort profile, with all six paths set absolute for the PC.
cp workbench/configs/pc_autopilot.pc_cohort.yaml ~/cir_pc.yaml
$EDITOR ~/cir_pc.yaml

# 3. Read-only plan first. This resolves selection actually (all 17 entries) but persists nothing.
./workbench/run.sh autopilot --config ~/cir_pc.yaml --dry-run

# 4. Cold run, then the same command to resume: `--resume` reuses every stage whose inputs are
#    unchanged and re-runs only what a changed input invalidated.
./workbench/run.sh autopilot --config ~/cir_pc.yaml --apply --resume
```

Per-model interpreters are PROVISIONED by the run, not copied: `allow_env_install: true` in the
cohort profile lets the planner create each model group's own prefix under `artifacts`. Never copy a
laptop virtualenv to the PC; the interpreter fingerprint is part of the model environment's identity.

A preflight that refuses a run is reporting a prerequisite, not failing: the refusal names the exact
item to fix (a missing permission by its flag, a missing asset by its path, an unverified
interpreter by its environment variable). Fix that one thing and re-run the same command with
`--resume`.

This live run was not executed here. Preconditions: configured CUDA-capable target and compatible
driver, canonical FashionIQ inputs, a verified selected checkpoint, provisioned model interpreters,
sufficient storage, and explicit permissions for each requested side effect. The capability gates
are per-invocation and named: preparation needs `allow_preparation`; TME's CUDA-only feature
generation additionally needs `allow_gpu_preparation`; evaluation needs `allow_gpu_eval`; downloads
need `allow_network` and `allow_large_downloads`. Withholding one gate refuses only the work that
needs it -- it never silently cancels work you authorized without it. Live Docker deployment is also
unexecuted; it requires Docker Engine/Compose, configured GPU passthrough where used, and the
deployment stack.

TME's EVA-ViT-g and BLIP-2 weights use `download_auxiliary_assets --model tme` (or the authorized
Autopilot auxiliary-assets stage), staging under `workbench/third_party/TME/weight/`. `mkdir` does
not install them, and `WORKBENCH_TME_ASSET_EVA_VIT_G` is undeclared. Validation-feature generation is
separate CUDA-only preparation: authorize `allow_preparation` and `allow_gpu_preparation`; reserve
about 10.47 GiB. This cache was not generated here.

TME generation now binds the CANONICAL FashionIQ images at the layout it declares, so it no longer
needs another model family to have produced a `resized_image/` tree first. A TME-only selection is
therefore valid.

`doctor.py` accepts `--checkpoint-policy {strict_clean,include_flagged_conditions}`. Runtime-preflight
checks the checkpoint the run selected. TME 20% remains `CHECKPOINT_UNAVAILABLE`; do not substitute
Air-Know 50% for the 20% condition. Per-model unresolved facts and next actions remain in
[MODEL_STATUS_SPEC.md](MODEL_STATUS_SPEC.md).
