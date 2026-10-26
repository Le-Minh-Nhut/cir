# PC reproduction handoff

Primary operator route is this document's [One run/resume path](#one-runresume-path). Copy the checked-in `pc_autopilot.pc_cohort.yaml` to a private PC-specific YAML and configure all six absolute paths, finite budgets, capabilities, model settings, and operational batch before any real run. [MODEL_STATUS_SPEC.md](MODEL_STATUS_SPEC.md) records per-model source, checkpoint, assets, capability, and blockers.

**Revision identity:** delivered revision: `1b890b759bc6663e838c374484092eabb73133b2` (`TESTED_SOURCE_SHA`); its production tree was unchanged by documentation-only successors through the branch tip at accounting time, `5afec61`. Four pytest invocations ran SEQUENTIALLY against the same frozen production tree, NOT as one single pass. The deterministic partition is provable: `chunks.json` totals 2,013 IDs; the four `chunk-q<N>.txt` files match the manifest exactly with no duplicates; an independent `--collect-only` also collected 2,013.

| Chunk | Collected | Passed | Failed | Skipped | Errors | Wall |
|---|---:|---:|---:|---:|---:|---:|
| q1 | 504 | 503 | 0 | 1 | 0 | 743.42 s |
| q2 | 503 | 503 | 0 | 0 | 0 | 344.61 s |
| q3 | 503 | 493 | 0 | 10 | 0 | 572.85 s |
| q4 | 503 | 503 | 0 | 0 | 0 | 784.71 s |
| **Total** | **2,013** | **2,002** | **0** | **11** | **0** | **2,445.59 s (~41 min)** |

**2,013 collected = 2,002 passed + 11 skipped + 0 failed + 0 errors.** Each chunk ran `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 .venv/bin/python -m pytest @.scratch-tmp/chunk-q<N>.txt -n 2 --dist=loadfile --basetemp=.scratch-tmp/chunk-q<N> -p no:cacheprovider`; each `basetemp` was on `/` under `.scratch-tmp/chunk-q<N>` and deleted after its chunk. Chunking was REQUIRED: this 173 GB drive is shared with three other active projects; a single full-suite pass needs ~4 GB of transient `basetemp`. Both earlier full-suite attempts aborted at the disk floor: 1998/2013 with 1.30 GB free on `/`, then 1722/2013 with 0.87 GB free; each chunk needed ≤~2 GB peak instead, and the lowest `/` free space during chunking was 1.44 GB (q1). `test_no_release_document_points_at_a_superseded_source_sha` passed in these runs: the documentation now names the revision containing the production tree. The 11 skips are the host's known, unchanged skip set (q1: 1, q3: 10); the logs do not classify reasons and they were not investigated further. Evidence: `.scratch-tmp/chunks.json`, `chunk-q1.txt`…`chunk-q4.txt`, `chunk-q1.log`…`chunk-q4.log`, `chunk-q<N>-meta.json`, `suite-attempts.jsonl`; `.scratch-tmp/chunk-q1-partial.log` preserves an earlier killed partial run for provenance. Last audited revision: `18b6a509f4cd0f50fef7f9ccef2bc7e80c2fa073`. This round ran without target-PC access: host RTX 3050 Laptop (`sm_86`), no model environment installed and no real weights downloaded. Every target-GPU execution remains `TARGET_GPU_PENDING`.

After completing the source checkout, controller `.venv` install, and cohort-profile setup in [One run/resume path](#one-runresume-path) below, run these read-only checks before authorizing model downloads, environment installation, preparation, or evaluation. The status commands inspect local inputs; `--dry-run` and `--verify-only` make no changes and do not use network. `prepare_model --model ptha_mtst --json` names missing validation image IDs. Some status commands return nonzero when they report blockers; read their JSON rather than treating the report itself as a failed check.

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

Copy the cohort profile outside the repository. Set all six `paths` keys to PC-specific absolute paths: `registry`, `dataset_root`, `artifacts`, `checkpoints`, `results`, `third_party`.

On this host the workbench-scope doctor returned 0, while `prepare_model.py --list --json` returned 1 because models have blockers; `manage_environment.py --list --json` likewise returned 1 because model interpreters are unconfigured. Those exit codes accompany valid read-only JSON diagnostics, not failed execution.

A workbench-scope doctor may exit 2 when local dependencies are pending; its JSON is readiness evidence, not model evaluation. The checked-in profiles select the whole registry: `selected_models: all_registered` resolves to all 17 entries, so a dry-run exercises real selection, checkpoint policy and override resolution. What a dry-run does NOT do is persist anything: it resolves the plan in memory, so `--status` (which reads the persisted state file) reports `NOT_RUN` with 0 models until an `--apply` run has written one. Use `--dry-run` to inspect the plan, and `--status` only after a run has started.

## One run/resume path

Choose exactly one source checkout path; both continue with the shared setup below.

Existing checkout (safe fast-forward; divergence is refused):

```bash
git fetch origin research/cir-failure-workbench
git checkout research/cir-failure-workbench
git merge --ff-only origin/research/cir-failure-workbench
git rev-parse HEAD          # branch tip; may be a documentation-only successor
git rev-parse 1b890b7       # must print TESTED_SOURCE_SHA: 1b890b759bc6663e838c374484092eabb73133b2
git merge-base --is-ancestor 1b890b7 HEAD
git diff --exit-code 1b890b7 HEAD -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'
```

Fresh checkout (clone the research branch directly):

```bash
git clone --branch research/cir-failure-workbench git@github.com:Le-Minh-Nhut/cir.git cir
cd cir
git rev-parse HEAD          # branch tip; may be a documentation-only successor
git rev-parse 1b890b7       # must print TESTED_SOURCE_SHA: 1b890b759bc6663e838c374484092eabb73133b2
git merge-base --is-ancestor 1b890b7 HEAD
git diff --exit-code 1b890b7 HEAD -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'
```

Continue from the repository root:

```bash
# 2. Create controller .venv and install both real dependency sources.
python3 -m venv .venv
.venv/bin/pip install -r workbench/backend/requirements.txt .
# The root project (`.`) supplies core dependencies; requirements.txt supplies controller CLIs and Hub acquisition.
# Keep per-model environments SEPARATE; never put a model environment on PATH.
.venv/bin/python workbench/scripts/doctor.py --scope workbench --json

# 3. Copy actual cohort YAML. Set all six paths above; keep budgets finite and permissions explicit.
cp workbench/configs/pc_autopilot.pc_cohort.yaml ~/cir_pc.yaml
$EDITOR ~/cir_pc.yaml
# Existing finite budgets: max_download_bytes: 37580963840, max_models: 32, top_k: 200.
# Confirm authorized capabilities: allow_network, allow_large_downloads, allow_env_install,
# allow_preparation, allow_gpu_eval, allow_gpu_preparation, allow_index_write.

# 3b. Configure per-model interpreters and, for TME, a finite card-sustainable batch.
cp workbench/config/workbench.env.example workbench/config/workbench.env
$EDITOR workbench/config/workbench.env
# For TME, set WORKBENCH_TME_BATCH_SIZE to a batch this GPU sustains; do not assume default 512 fits.
# On a Blackwell target, select the declared adaptation before planning or applying:
export WORKBENCH_RUNTIME_ADAPTATION=sm_120

# 4. Inspect read-only plan first.
./workbench/run.sh autopilot --config ~/cir_pc.yaml --dry-run

# 5. Apply/resume the cohort.
./workbench/run.sh autopilot --config ~/cir_pc.yaml --apply --resume

# 6. For EACH completed run, read the model row in <paths.artifacts>/reports/autopilot_results.csv;
#    open its aggregate_report and use its exact run_id, model_id, checkpoint_id and protocol_id values.
.venv/bin/python workbench/scripts/verify_run.py \
    --run-id <RUN_ID> --model <MODEL_ID> --checkpoint <CHECKPOINT_ID> --protocol <PROTOCOL>

# 7. Repeat the unchanged apply/resume command and confirm completed stages are reused.
./workbench/run.sh autopilot --config ~/cir_pc.yaml --apply --resume
```

**This route does not need Docker.** `run.sh autopilot` calls the pipeline directly, and the
autopilot pins `serve=False`: no container, no API, no UI, no port. Requesting the optional
index/UI surface is what engages the container stack (`pc.sh`, `docker compose`, `serve_workbench.py`),
and nothing in the research run above depends on it. Do not treat an unverified Docker deployment as
a blocker for a standalone research run, and do not claim Docker readiness without running it.

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
