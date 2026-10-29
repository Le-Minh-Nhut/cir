# PC reproduction handoff

Primary operator route is this document's [One run/resume path](#one-runresume-path). Copy the checked-in `pc_autopilot.pc_cohort.yaml` to a private PC-specific YAML and configure all six absolute paths, finite budgets, capabilities, model settings, and operational batch before any real run. [MODEL_STATUS_SPEC.md](MODEL_STATUS_SPEC.md) records per-model source, checkpoint, assets, capability, and blockers.

**Current revision identity:** tested production revision: `496cdf637e6aee0da2bfc855935e42a3295f605c` (`TESTED_SOURCE_SHA`). One collection produced 2130 node IDs (`workbench/tests` + `tests/unit`); they were run as four sequential chunks, with no duplicate or omitted IDs. Scope `workbench/tests` (2071 collected):
| Chunk | Collected | Passed | Failed | Skipped | Errors | Wall |
|---|---:|---:|---:|---:|---:|---:|
| q1 | 532 | 532 | 0 | 0 | 0 | 1436 s |
| q2 | 533 | 531 | 1 | 1 | 0 | 706 s |
| q3 | 532 | 523 | 0 | 9 | 0 | 1382 s |
| q4 | 474 | 474 | 0 | 0 | 0 | 1599 s |
| **Total** | **2071** | **2060** | **1** | **10** | **0** | **5123 s** |

The fourth chunk also carried the 59 `tests/unit` node IDs (chunk 4 total 533 = 474 + 59): `tests/unit` 59 passed, 0 failed. Total wall for all four chunks: **5126 s (~85.4 min)**. The single `workbench/tests` failure is q2's `test_handoff_commands.py::test_no_release_document_points_at_a_superseded_source_sha`, which fails by construction on the revision whose documents still name the previous tested source; it passes once they name this revision. `tests/unit`: 59 passed in 2.4 s. Target-GPU execution remains `TARGET_GPU_PENDING`.


## Historical previous-release accounting

Each chunk result above comes from one sequential invocation over its exact partition; together they cover the complete node set, not a single full-suite pass.

**Historical R23 accounting (2026-10-25):** delivered revision `1b890b759bc6663e838c374484092eabb73133b2` (`TESTED_SOURCE_SHA` at that time); the chunk results and disk-floor notes below describe that earlier R23 run and remain historical.

| Chunk | Collected | Passed | Failed | Skipped | Errors | Wall |
|---|---:|---:|---:|---:|---:|---:|
| q1 | 504 | 503 | 0 | 1 | 0 | 743.42 s |
| q2 | 503 | 503 | 0 | 0 | 0 | 344.61 s |
| q3 | 503 | 493 | 0 | 10 | 0 | 572.85 s |
| q4 | 503 | 503 | 0 | 0 | 0 | 784.71 s |
| **Total** | **2,013** | **2,002** | **0** | **11** | **0** | **2,445.59 s (~41 min)** |

**2,013 collected = 2,002 passed + 11 skipped + 0 failed + 0 errors.** Each chunk ran `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 .venv/bin/python -m pytest @.scratch-tmp/chunk-q<N>.txt -n 2 --dist=loadfile --basetemp=.scratch-tmp/chunk-q<N> -p no:cacheprovider`; each `basetemp` was on `/` under `.scratch-tmp/chunk-q<N>` and deleted after its chunk. Chunking was REQUIRED: this 173 GB drive is shared with three other active projects; a single full-suite pass needs ~4 GB of transient `basetemp`. Both earlier full-suite attempts aborted at the disk floor: 1998/2013 with 1.30 GB free on `/`, then 1722/2013 with 0.87 GB free; each chunk needed ≤~2 GB peak instead, and the lowest `/` free space during chunking was 1.44 GB (q1). `test_no_release_document_points_at_a_superseded_source_sha` passed in these runs: the documentation now names the revision containing the production tree. The 11 skips are the host's known, unchanged skip set (q1: 1, q3: 10); the logs do not classify reasons and they were not investigated further. Evidence: `.scratch-tmp/chunks.json`, `chunk-q1.txt`…`chunk-q4.txt`, `chunk-q1.log`…`chunk-q4.log`, `chunk-q<N>-meta.json`, `suite-attempts.jsonl`; `.scratch-tmp/chunk-q1-partial.log` preserves an earlier killed partial run for provenance. Last audited revision: `18b6a509f4cd0f50fef7f9ccef2bc7e80c2fa073`. This round ran without target-PC access: host RTX 3050 Laptop (`sm_86`); no model environments were installed or provisioned in this round, and no real weights were downloaded. Every target-GPU execution remains `TARGET_GPU_PENDING`.

After completing source checkout, controller `.venv` install, and cohort-profile setup in [One run/resume path](#one-runresume-path), synchronize and prove the YAML roots match `workbench.env` before running any standalone preparation, inventory, doctor, or verification command. Those commands use `resolve_config()` (workbench.env plus process environment), not the autopilot YAML. Then run these read-only checks before authorizing model downloads, environment installation, preparation, or evaluation. The status commands inspect local inputs; `--dry-run` and `--verify-only` make no changes and do not use network. `prepare_model --model ptha_mtst --json` names missing validation image IDs. Some status commands return nonzero when they report blockers; read their JSON rather than treating the report itself as a failed check.

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
| Nine-model execution cohort and adapted profiles | The agreed execution set remains `limn`, `encoder`, `csmcir`, `ptha_mtst`, `tme`, `intent`, `habit`, `conesep`, and `hint`. Nine `sm_120` profiles have solver resolutions for `torch==2.8.0+cu128` / `torchvision==0.23.0+cu128`; this establishes solver resolution only, not installation, import success, or GPU execution. Historical profiles remain default reproduction contracts. | Provision each selected model's declared wheel set in its target interpreter, then import-check it and run a real CUDA capability check. |
| PTHA and declared assets | R23 declares EVA-ViT-g, `bert-base-uncased`, validation split inputs, and exact missing PNG-ID diagnostics; no real weights downloaded | Stage pinned assets and convert/stage canonical upstream `.jpg` images as `fashionIQ_dataset/images/{image_name}.png`; re-run `prepare_model --model ptha_mtst --json` before evaluation. This CIRR→FashionIQ transfer is not a FashionIQ reproduction. |
| Evaluation | No R23 target execution; all such claims remain `TARGET_GPU_PENDING` | Run each authorized model/checkpoint under its declared protocol; keep aggregate-only and per-query evidence distinct. |
| Run verification | No R23 target run to verify | Run `verify_run.py` with `--run-id`, `--model`, `--checkpoint`, and `--protocol` for each completed run. |
| Unchanged resume | No target resume performed | Repeat same `autopilot --apply --resume` command with unchanged inputs and confirm completed stages reuse their state without downloads or reevaluation. |

## Which profile to use on the PC

| Profile | What it is | Use it when |
|---|---|---|
| `workbench/configs/pc_autopilot.pc_cohort.yaml` | **The cohort profile.** `selected_models: all_registered`, `include_flagged_conditions`, authorized 20% overrides (`habit`/`intent`/`conesep` → `fiq_n02`, `ptha_mtst` → `cirr_transfer`, `encoder` → `fashioniq`), a 35 GiB budget, and capabilities needed for a cold run. The agreed nine-model execution set remains fixed; narrowing YAML `selected_models` to those nine IDs is an operator option, not a new cohort. | **This is the one to run.** It combines the registered selection with authorization for a cold run. |
| `workbench/configs/pc_autopilot.all_available.yaml` | Every selectable verified variant, wider budget | You want the widest sweep rather than the authorized cohort |
| `workbench/configs/pc_autopilot.example.yaml` | A minimal, fully-locked example (`strict_clean`, no download, no env install, no preparation, no GPU) | You only want to confirm CLI/config parsing. **Do not** treat it as a runnable cohort: it leaves some entries blocked and authorizes nothing a cold run needs |
| `workbench/configs/pc_autopilot.pilot_{csmcir,limn}.yaml` | Single-model pilots | Debugging one model's path |

Copy the cohort profile outside the repository. Set all six `paths` keys to PC-specific absolute paths: `registry`, `dataset_root`, `artifacts`, `checkpoints`, `results`, `third_party`. Standalone commands do not read this YAML, so synchronize the five roots they share into `workbench/config/workbench.env` as shown below. `paths.registry` is different: it maps to process variable `WORKBENCH_REGISTRY_PATH`, which is not a registry-declared setting and is rejected by `workbench.env`. For a custom registry, export `WORKBENCH_REGISTRY_PATH` to the same absolute file path before standalone commands. The checked-in registry remains the default when that variable is unset.

| YAML `paths.*` | `workbench.env` key or process-only setting |
|---|---|
| `paths.registry` | `WORKBENCH_REGISTRY_PATH` (process environment only; not accepted in `workbench.env`) |
| `paths.dataset_root` | `FASHIONIQ_ROOT` |
| `paths.artifacts` | `WORKBENCH_ARTIFACTS_ROOT` |
| `paths.checkpoints` | `WORKBENCH_CHECKPOINT_ROOT` |
| `paths.results` | `WORKBENCH_RESULTS_ROOT` |
| `paths.third_party` | `WORKBENCH_THIRD_PARTY_ROOT` |

Set `CIR_REPO_ROOT` to the actual checkout. Exported shell variables override matching `workbench.env` values, so remove or update stale exports before proving roots. To print and check that both loaders agree (and that registry paths agree), run from the checkout root after editing the environment file and, for a custom registry, exporting `WORKBENCH_REGISTRY_PATH`:

The helper prints the YAML paths, the five standalone roots, `CIR_REPO_ROOT`, and resolved registry; a disagreement exits nonzero and names the disagreeing key.

```bash
.venv/bin/python workbench/scripts/print_roots.py --config ~/cir_pc.yaml
```

The standalone read-only commands below have this same-roots precondition. A mismatch means do not run preparation, inventory, verification, or evaluation yet.

On this host the workbench-scope doctor returned 0, while `prepare_model.py --list --json` returned 1 because models have blockers; `manage_environment.py --list --json` likewise returned 1 because not all required model interpreters were ready. Those exit codes accompany valid read-only JSON diagnostics, not failed execution.

The doctor may exit 2 when local dependencies are pending; its JSON is readiness evidence, not model evaluation. `all_registered` selects the registry. The R25-01 code change in this same branch excludes blocked models from provisioning under `all_registered`; see `<paths.artifacts>/reports/autopilot_results.csv` and `<paths.artifacts>/reports/blockers.json` for actual per-model blockers. A dry-run resolves selection, checkpoint policy, and overrides without persisting state; `--status` reports `NOT_RUN` with 0 models until an `--apply` run writes state.

## One run/resume path

Choose exactly one source checkout path; both continue with the shared setup below.

Existing checkout (safe fast-forward; divergence is refused):

```bash
git fetch origin research/cir-failure-workbench
git checkout research/cir-failure-workbench
git merge --ff-only origin/research/cir-failure-workbench
git rev-parse HEAD          # branch tip; may be a documentation-only successor
git rev-parse 496cdf63       # must print TESTED_SOURCE_SHA: 496cdf637e6aee0da2bfc855935e42a3295f605c
git merge-base --is-ancestor 496cdf63 HEAD
git diff --exit-code 496cdf63 HEAD -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'
```

Fresh checkout (clone the research branch directly):

```bash
git clone --branch research/cir-failure-workbench git@github.com:Le-Minh-Nhut/cir.git cir
cd cir
git rev-parse HEAD          # branch tip; may be a documentation-only successor
git rev-parse 496cdf63       # must print TESTED_SOURCE_SHA: 496cdf637e6aee0da2bfc855935e42a3295f605c
git merge-base --is-ancestor 496cdf63 HEAD
git diff --exit-code 496cdf63 HEAD -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'
```

Continue from the repository root:

```bash
# 2. Create controller .venv and install both real dependency sources.
python3 -m venv .venv
.venv/bin/pip install -r workbench/backend/requirements.txt .
# The root project (`.`) supplies core dependencies; requirements.txt supplies controller CLIs and Hub acquisition.
# Keep per-model environments SEPARATE; never put a model environment on PATH.

# 3. Copy actual cohort YAML. Set all six paths above; keep budgets finite and permissions explicit.
cp workbench/configs/pc_autopilot.pc_cohort.yaml ~/cir_pc.yaml
$EDITOR ~/cir_pc.yaml
# Existing finite budgets: max_download_bytes: 37580963840, max_models: 32, top_k: 200.
# Confirm authorized capabilities: allow_network, allow_large_downloads, allow_env_install,
# allow_preparation, allow_gpu_eval, allow_gpu_preparation, allow_index_write.

# 3b. Copy and edit env file: set CIR_REPO_ROOT to this checkout and each mapped root to the
#     exact absolute path in ~/cir_pc.yaml (mapping and proof command above). Process exports win.
cp workbench/config/workbench.env.example workbench/config/workbench.env
$EDITOR workbench/config/workbench.env
# For a custom paths.registry, export WORKBENCH_REGISTRY_PATH=<same absolute registry file>.
# It is process-environment-only: the env-file parser rejects this undeclared setting.
# Run the root-proof one-liner above now; do not proceed unless all values agree.
# For TME, set WORKBENCH_TME_BATCH_SIZE to a batch this GPU sustains; default 512 is not guaranteed to fit.
# On a Blackwell target, select the declared adaptation before planning or applying:
export WORKBENCH_RUNTIME_ADAPTATION=sm_120
.venv/bin/python workbench/scripts/doctor.py --scope workbench --json

# 4. Inspect read-only plan first.
./workbench/run.sh autopilot --config ~/cir_pc.yaml --dry-run

# 5. Apply/resume the cohort.
./workbench/run.sh autopilot --config ~/cir_pc.yaml --apply --resume

# 6. For EACH completed run, read the model row in <paths.artifacts>/reports/autopilot_results.csv;
#    open its aggregate_report and use its exact run_id, model_id, checkpoint_id and protocol_id values.
# Both this standalone verifier and standalone preparation/inventory commands require the same-roots
# proof above; they do not read ~/cir_pc.yaml.
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
separate CUDA-only preparation: authorize `allow_preparation` and `allow_gpu_preparation`; budget about
10.47 GiB for feature-cache/storage. This is a storage estimate, not a VRAM guarantee; fit depends on
the GPU and chosen batch. This cache was not generated here.

TME generation now binds the CANONICAL FashionIQ images at the layout it declares, so it no longer
needs another model family to have produced a `resized_image/` tree first. A TME-only selection is
therefore valid.

`doctor.py` accepts `--checkpoint-policy {strict_clean,include_flagged_conditions}`. Runtime-preflight
checks the checkpoint the run selected. TME 20% remains `CHECKPOINT_UNAVAILABLE`; do not substitute
Air-Know 50% for the 20% condition. Per-model unresolved facts and next actions remain in
[MODEL_STATUS_SPEC.md](MODEL_STATUS_SPEC.md).

Checkpoint absence and mapping uncertainty are different. `NO_CHECKPOINT` / `CHECKPOINT_UNAVAILABLE`
means no public checkpoint is available for requested condition; `UNVERIFIED` /
`CHECKPOINT_MAPPING_UNVERIFIED` means candidate artifact mapping to benchmark row is unresolved.
Do not describe every deferred entry as missing a public checkpoint. Registry distinguishes the seven
other deferred entries as follows:

| Model | Registry distinction |
|---|---|
| `pair` | Hub lists `pair-B1.pt` and `pair-B2.pt`; neither is mapped to FashionIQ/training condition and full digests are unavailable. Native evaluation command is also blocked. |
| `clvc_net` | Google Drive `CLVCNET.zip` member/category mapping is unresolved; complete replay needs six category-specific local/global files, whose presence is not established. Source replay is blocked. |
| `dcnet` | Author Drive folder exists, but its run-directory association to the benchmark row is unresolved; expected artifact is `config.json` plus `trained_model.pth`. Native CLI is audited. |
| `combiner_rn50x4_noft` | Drive release names FashionIQ noft state, but artifact identity/checksum is unverified; noft/fullft distinction must not be guessed from an arbitrary file. |
| `clip4cir_rn50x4_fullft` | Requires paired Combiner and fine-tuned CLIP states; author Drive pairing is unresolved, so bundle mapping stays blocked. |
| `tgcir` | FashionIQ archive member and hash are unresolved; pinned source has no audited replay CLI. |
| `sprc` | Candidate author files exist, but model-class/key mapping remains unresolved; ViT-L file is a separate scientific variant, not verified paper-row checkpoint. |

Air-Know differs: public 50% and 80% weights are registered, but no 20% checkpoint is available; do not substitute conditions. Per-model status and next actions remain in [MODEL_STATUS_SPEC.md](MODEL_STATUS_SPEC.md).
