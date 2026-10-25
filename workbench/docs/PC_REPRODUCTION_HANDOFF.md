# PC reproduction handoff

Primary operator path: [AUTOPILOT_GUIDE.md](AUTOPILOT_GUIDE.md). Use the checked-in example profile only for read-only planning; copy it to a private PC-specific YAML and configure dataset, persistent paths, model selection, checkpoint policy, budgets, and permissions before any real run. [MODEL_STATUS_SPEC.md](MODEL_STATUS_SPEC.md) records per-model source, checkpoint, assets, capability, and blockers.

**Revision identity:** delivered revision: `1b890b759bc6663e838c374484092eabb73133b2`. Suite accounting: Hai lần chạy xdist trên cùng cây đóng băng 4c73d962: (1) 1998/2013 thực thi — 1986 passed, 11 skipped, 1 failed (guard SHA, dự kiến trước commit), 0 errors, abort do đĩa ở 840s; (2) 1722/2013 thực thi — 1715 passed, 5 skipped, 2 failed (cùng guard SHA + một xdist lock race đã chứng minh pass khi chạy riêng), 0 errors, abort do đĩa ở 540s. KHÔNG có lần chạy đầy đủ một lượt: đĩa máy chủ (/ còn ~3.7 GB, /tmp ~1.9 GB, ba project dùng chung) không đủ cho ~4 GB basetemp. Phần chưa thực thi của hai lần chạy bù nhau, nhưng không có log để chứng minh hợp (union) nên không claim "full suite green". Last audited revision: `18b6a509f4cd0f50fef7f9ccef2bc7e80c2fa073`. This round ran without target-PC access: host RTX 3050 Laptop (`sm_86`), no model environment installed and no real weights downloaded. Every target-GPU execution remains `TARGET_GPU_PENDING`.

Run these checks first after preparing the controller `.venv`, before authorizing model downloads, environment installation, preparation, or evaluation. The status commands inspect local inputs; `--dry-run` and `--verify-only` make no changes and do not use network. `prepare_model --model ptha_mtst --json` names missing validation image IDs. Some status commands return nonzero when they report blockers; read their JSON rather than treating the report itself as a failed check.

```bash
./workbench/run.sh autopilot --help
.venv/bin/python workbench/scripts/doctor.py --scope workbench --json
.venv/bin/python -m pip check
.venv/bin/python workbench/scripts/prepare_model.py --list --json
.venv/bin/python workbench/scripts/prepare_model.py --model ptha_mtst --json
.venv/bin/python workbench/scripts/manage_environment.py --list --json
.venv/bin/python workbench/scripts/download_auxiliary_assets.py --all --verify-only
.venv/bin/python workbench/scripts/pipeline.py prepare --model csmcir --dry-run
.venv/bin/python workbench/scripts/download_auxiliary_assets.py --model tme --dry-run
./workbench/run.sh autopilot --config ~/cir_pc.yaml --dry-run
```

The controller install route below matches the manifests: `workbench/backend/requirements.txt` supplies `packaging` and `huggingface-hub`; installing the root project `.` supplies `torch`, `torchvision`, `numpy`, `pillow`, and `packaging`. The reported fresh-venv check installed both, started 14 CLI `--help` entrypoints, and passed `pip check`; no output artifact is retained. `test_controller_installation.py` checks declared dependencies, doctor coverage (ten modules), and CLI startup. Install only after reviewing the target's Python >=3.11/controller requirements; keep model interpreters separate.

## What changed this round / what remains PC-only

| Item | Established this round | Still requires target PC |
|---|---|---|
| Nine adapted profiles | Solver resolution only: each selected model resolves `torch==2.8.0+cu128` / `torchvision==0.23.0+cu128` from cu128 index; no environment was installed | Install each selected model's declared cu128 wheel set in its target interpreter, then import-check it and run a real CUDA capability check. Profile declaration or solver success is not installation, import success, or GPU execution; historical profiles remain default reproduction contracts. |
| PTHA and declared assets | R23 declares EVA-ViT-g, `bert-base-uncased`, validation split inputs, and exact missing PNG-ID diagnostics; no real weights downloaded | Stage pinned assets and convert/stage canonical upstream `.jpg` images as `fashionIQ_dataset/images/{image_name}.png`; re-run `prepare_model --model ptha_mtst --json` before evaluation. This CIRR→FashionIQ transfer is not a FashionIQ reproduction. |
| Evaluation | No R23 target execution; all such claims remain `TARGET_GPU_PENDING` | Run each authorized model/checkpoint under its declared protocol; keep aggregate-only and per-query evidence distinct. |
| Run verification | No R23 target run to verify | Run `verify_run.py` with `--run-id`, `--model`, `--checkpoint`, and `--protocol` for each completed run. |
| Unchanged resume | No target resume performed | Repeat same `autopilot --apply --resume` command with unchanged inputs and confirm completed stages reuse their state without downloads or reevaluation. |

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
#    A bare `fetch` + `checkout` does NOT advance an existing stale local branch, so the update is
#    an explicit fast-forward. `--ff-only` refuses rather than merging: divergent local work is never
#    discarded, it is reported.
git fetch origin research/cir-failure-workbench
git checkout research/cir-failure-workbench
git merge --ff-only origin/research/cir-failure-workbench
git rev-parse HEAD          # must print the delivered SHA recorded in MODEL_STATUS_SPEC.md

# 1b. No local branch yet (fresh clone or another machine):
#     git clone git@github.com:Le-Minh-Nhut/cir.git && cd cir
#     git checkout -b research/cir-failure-workbench origin/research/cir-failure-workbench

# 2. Controller interpreter. Install the controller requirements AND the root project into .venv:
#    `python3 -m venv .venv && .venv/bin/pip install -r workbench/backend/requirements.txt .`
#    (or uv: `uv venv .venv && uv pip install -r workbench/backend/requirements.txt . --python .venv/bin/python`).
#    The root project (`.`) provides core dependencies (torch, torchvision, numpy, pillow, packaging),
#    while requirements.txt supplies controller CLIs and Hub acquisition. Model interpreters stay
#    SEPARATE and are provisioned per model; never put a model environment on PATH. Run doctor:
.venv/bin/python workbench/scripts/doctor.py --scope workbench --json

# 3. The cohort profile, with all six paths set absolute for the PC, kept outside the repository.
cp workbench/configs/pc_autopilot.pc_cohort.yaml ~/cir_pc.yaml
$EDITOR ~/cir_pc.yaml

# 3b. Local per-model settings that are NOT paths: the interpreter for each model's own environment
#     and the operational batch for TME's CUDA feature generator. Copy the example and edit it; a
#     value here behaves exactly like the same variable exported into the process environment.
cp workbench/config/workbench.env.example workbench/config/workbench.env
$EDITOR workbench/config/workbench.env

# 4. Read-only plan first. This resolves selection actually (all 17 entries) but persists nothing.
./workbench/run.sh autopilot --config ~/cir_pc.yaml --dry-run

# 5. Cold run, then the same command to resume: `--resume` reuses every stage whose inputs are
#    unchanged and re-runs only what a changed input invalidated.
./workbench/run.sh autopilot --config ~/cir_pc.yaml --apply --resume

# 6. Verifier. All four flags are REQUIRED by its parser; a run cannot be verified without them.
#    Take RUN_ID/CHECKPOINT/PROTOCOL from the run's own row in
#    `<paths.artifacts>/reports/autopilot_results.csv` (or the aggregate report file name), not from
#    memory: the verifier refuses a run whose recorded identity differs from the one requested.
.venv/bin/python workbench/scripts/verify_run.py \
    --run-id <RUN_ID> --model <MODEL_ID> --checkpoint <CHECKPOINT_ID> --protocol <PROTOCOL>
```

**This route does not need Docker.** `run.sh autopilot` calls the pipeline directly, and the
autopilot pins `serve=False`: no container, no API, no UI, no port. Requesting the optional
index/UI surface is what engages the container stack (`pc.sh`, `docker compose`, `serve_workbench.py`),
and nothing in the research run above depends on it. Do not treat an unverified Docker deployment as
a blocker for a standalone research run, and do not claim Docker readiness without running it.

On a Blackwell target, also set the runtime adaptation in the same file (step 3b):

```bash
# One operator setting, in the file whose precedence the guide documents:
#     `WORKBENCH_RUNTIME_ADAPTATION=sm_120`
# In `workbench/config/workbench.env`, or exported into the process environment (which wins).
export WORKBENCH_RUNTIME_ADAPTATION=sm_120
```

Models that declare a matching `adapted_runtime` profile are provisioned with it and record it as
`variant_provenance.runtime_adaptation`; every other model keeps its historical author pins
unchanged. An adaptation value no model declares is refused, not ignored.

Per-model interpreters are PROVISIONED by the run, not copied: `allow_env_install: true` in the
cohort profile lets the planner create each model group's own prefix under `artifacts`. Never copy a
laptop virtualenv to the PC; the interpreter fingerprint is part of the model environment's identity.

A preflight that refuses a run is reporting a prerequisite, not failing: the refusal names the exact
item to fix (a missing permission by its flag, a missing asset by its path, an unverified
interpreter by its environment variable). Fix that one thing and re-run the same command with
`--resume`.

## Resume, and what invalidates a finished stage

`--resume` skips a stage whose INPUT fingerprint still matches the one recorded when it completed, so
a second identical invocation computes nothing and downloads nothing:

| Change one of these | What becomes owed again |
|---|---|
| a canonical image's bytes, the split JSON for a category, or a caption file | the affected stage's fingerprint, and everything downstream of it |
| the resolved `WORKBENCH_TME_BATCH_SIZE`, the pinned source commit, or a base weight | TME feature generation only (the resolved batch is bound into the precompute receipt and compared on reuse) |
| a checkpoint file, a model interpreter's content, or the registry declaration | that model's checkpoint/preparation/evaluation, not the other models' |
| `permissions` in the profile, or the selected model list | nothing by itself: the plan is re-derived, but completed stages keep their state |
| `paths.artifacts` | everything, because the recorded state lives under that root |

An interrupted acquisition leaves a `.part` file, never a published one, so re-running fetches only
what is still missing. A cache that arrived partially is refused by name rather than used (TME's hub
cache reports the exact missing declared files).

Two states are deliberately NOT resumable: an evaluation whose environment was rewritten (the
dependency fingerprint changed) and a receipt that no longer binds its inputs. Both report the reason
instead of silently reusing weaker evidence.

A preflight that refuses a run is reporting a prerequisite, not failing: the refusal names the exact
item to fix (a missing permission by its flag, a missing asset by its path, an unverified
interpreter by its environment variable). Fix that one thing and re-run the same command with
`--resume`.

This live run was not executed here. Preconditions: configured CUDA-capable target and compatible
driver, canonical FashionIQ inputs, a verified selected checkpoint, provisioned model interpreters,
sufficient storage, and explicit permissions for each requested side effect. The capability gates
are per-invocation and named: preparation needs `allow_preparation`; TME's CUDA-only feature
generation additionally needs `allow_gpu_preparation`; evaluation needs `allow_gpu_eval`; downloads
need `allow_network` and `allow_large_downloads`. Withholding a gate refuses the work that needs it,
and any work whose DECLARED prerequisite that stage provides: TME's feature generation genuinely
consumes its base weights and the Qformer cache, so a run that withholds downloads refuses TME
preparation by name (`SKIPPED_DEPENDENCY`), while every model whose prerequisites are untouched keeps
running. Nothing is cancelled silently, and an unrelated model's failure never cancels a valid one.
Live Docker deployment is also
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
