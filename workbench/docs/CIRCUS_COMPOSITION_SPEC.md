# CIRCUS composition evaluation spec

## Research question and interpretation

This opt-in evaluation asks whether specialized CIR models benefit from composing the reference image and modification text on the released CIRCUS subsets, compared across multimodal and two input-ablation conditions. Report any difference as **input-ablation dependence under this protocol**, not proof of intrinsic compositional reasoning. Specialized models were not trained on blank images or empty text, so a large gap may partly reflect distribution shift. Report multimodal (MM) retrieval quality next to every composition gap.

## Protocol and gallery

Primary protocol: `circus_fashioniq_release_v1`. Its gallery is the exact ordered list of 74,381 image identities in the pinned aggregated audit artifact. The paper's dataset paragraph states 18,853 images; the released artifact contains 74,381, matching M-BEIR's candidate pool. Stored ranks exceed 18,853: the maximum aggregated multimodal rank observed is 31,987 at sample index 4435. The discrepancy is therefore not an unused metadata field.

This workbench uses the released-artifact protocol. It does **not** claim exact reproduction of the paper's stated 18,853-image setting. No 18,853-image subset is invented, and the local native gallery is not silently substituted. The reference is eligible in ranking: the author retrieval code ranks without masking it. The native `-10e10` reference mask from ENCODER/LIMN val-split replays is deliberately not inherited.

## Conditions and upstream ablation naming

| Semantic condition | Input intervention |
|---|---|
| `multimodal` | Original reference image and edit text. |
| `text_only_black_image` | Replace reference with a solid black image of the same size **before** the model's own image transform; retain original text. |
| `image_only_empty_text` | Retain reference; remove edit text before tokenization. Ordinary tokenizer special tokens may remain. |

The upstream filenames are misleading. `ablation_configs/image_only_zero_image.json` contains `{"name": "img_black"}` but `retrieval/generate_retrieval_data.py` uses it for **text-only**. `ablation_configs/text_only_drop_text.json` contains `{"name": "text_drop_all"}` but is used for **image-only**. Names describe the removed input; this spec uses semantic mode names.

The black-image intervention follows `subset_evaluation/eval_subset.py::apply_ablation`: it replaces the RAW image via `_black_image_same_size`. A zero tensor after normalization or a zeroed feature vector is a different intervention and is not used here. Query text is passed as supplied; it is never re-split on `and`.

## Release data contract

The pinned author revision for these release inputs is `b7ed55104abd987e60f7099037e4cd8a69966428`.

| Set or artifact | Path / identity | Rows or count |
|---|---|---:|
| Full queries | M-BEIR `query/test/mbeir_fashioniq_task7_test.jsonl` | 6,003 |
| Shortcut-free (SF) | `shortcut_free_subsets/fashioniq/shortcut_free/mbeir_fashioniq_task7_test_multimodal.jsonl` | 4,069 |
| Validated solved | `final_dataset/query_jsonl/fashioniq/validated_solved/mbeir_fashioniq_task7_test_validated_solved.jsonl` | 368 |
| Validated unsolved | `final_dataset/query_jsonl/fashioniq/validated_unsolved/mbeir_fashioniq_task7_test_validated_unsolved.jsonl` | 218 |
| Validated union | Disjoint solved + unsolved query IDs | 586 |
| Gallery | Ordered `gallery_paths` from pinned aggregated audit artifact | 74,381 |
| Candidate pool | M-BEIR candidate-pool rows | 74,381 |

The validated parts have intersection 0 and `validated ⊆ sf ⊆ full` by query identity. `validated_unsolved` means unresolved by the ORIGINAL GME pool, not invalid; it remains in the cohort. The supplied `query_txt` is used unchanged.

These are locally computed SHA-256 values for the inspected bytes, not author-signed digests:

| File | SHA-256 |
|---|---|
| Aggregated audit | `27bdbef041200dc393b5478d3e57121bc16c70a9acc2e98a19fb48408a99533c` |
| SF | `95a93883334e351e4d3aaaf25e47e89b48d4daf02bdb786e8f00d58165ace37d` |
| Validated solved | `18f829da4d3b1ee8439f43998bae43dbbe8924c2eaafba84473ae849f564f5c7` |
| Validated unsolved | `ba019a30d3b45f4b29c34d2d48638dbeeed2e2a42409f625922c65de5b2a7efa` |
| M-BEIR query | `b601ef6a5411bf9385cff41e64bea863871ab1094cd09d78683de6a5141165ef` |
| M-BEIR candidate pool | `121ec284832c33289ee13bfac19d5d59dc3f8aeddf672d784fd5eb8bcdf79eef` |

## Local data availability

On this host, the released gallery names images `.jpg`, while the local FashionIQ root is flat and contains 74,381 `.png` files. All released gallery identities resolve by filename stem; zero need fetching. Query references (4,315 distinct basenames) and all 6,003 positive targets are present. Cached `data/fashionIQ_dataset/features/FashionIQ/val_images.pt` and `val_name_to_idx.json` are 1-byte / `{}` placeholders, not a reusable feature cache.

The garment-category resolver uses local annotations. It resolves 5,884 of 6,003 queries; 119 are `AMBIGUOUS` because a reference is shared across categories, so they remain unscoped rather than being assigned. There are 0 unresolved category identities. LIMN has separate whole-model checkpoints for `dress`, `shirt`, and `toptee`; it cannot score those 119 ambiguous-category queries. This is a coverage limitation, not a retrieval miss.

## Metrics and outcome labels

For each scored query, `r` is the 1-based exact target rank over the FULL eligible gallery:

```text
hit_at_10(q) = 1[r <= 10];  hit_at_50(q) = 1[r <= 50]
ndcg_full(q) = 1 / log2(r + 1)
reciprocal_rank(q) = 1 / r
MM = mean(ndcg_full) over the selected queries in multimodal mode; T and I likewise
composition_gap_ndcg = 1 - max(I, T) / MM      # ratio computed AFTER averaging the same cohort
```

The gap uses a common cohort with exact ranks in all three conditions. Compute the ratio after averaging, not by averaging per-query ratios. Retain negative gaps. Recall is **PERCENT**; nDCG and gap are **FRACTIONS**; never mix these units. An empty cohort, missing condition, or zero MM denominator produces an explicit unavailable value and reason, never 0, infinity, or NaN. An unscored query reduces coverage, is excluded from means, and is **not** a miss.

If reported, the reciprocal-rank difference is separately named `mrr_gap`. The paper's normalized nDCG gap is primary. A README-style Recall subtraction is not the paper's Composition Gap and must not be called `compgap`.

For K=10, report mutually exclusive outcomes: `text_only`, `image_only`, `both`, `composition_required` (MM succeeds and neither unimodal condition succeeds), and `unresolved` (all three fail). `unscored` is separate; a query missing any condition's exact rank is excluded from outcome bins, not counted as an unresolved retrieval miss. The current summary serializes this separate count under `neither_scored`.

## Checkpoint and training-condition labels

Every result carries the selected checkpoint and its actual training condition separately from evaluation annotation noise. Do not lump supervised FashionIQ checkpoints, noise-trained checkpoints, transfer checkpoints, and zero-shot embedding models together.

- HABIT, INTENT, and ConeSep retain selected `fiq_n02`: **training noise 20%**, evaluation annotation noise **0%**. ConeSep's registry notes only 20/50/80 noise variants exist.
- TME retains `fashioniq_0`, labelled **training noise 0%**. No 20% checkpoint exists.
- PTHA+MTST `cirr_transfer` is a **CIRR→FashionIQ transfer** checkpoint, not a FashionIQ reproduction.
- ENCODER's checkpoint-mapping uncertainty remains visible; successful composition execution resolves no unrelated mapping uncertainty.
- For models whose training noise is unstated in the existing status/registry, retain that label as unstated; do not infer 0% from evaluation noise.

## 17-model composition capability and status

“Supported” means the adapter declares the mode's input path; it does not imply full evaluation. Nine adapters are scaffolded; all remain `TARGET_GPU_PENDING` for the target-GPU (`sm_120`) full composition evaluation. On this host's RTX 3050 (`sm_86`, 4 GB), the new code has produced real-data GPU smoke evidence for ENCODER, HABIT, LIMN and TME (see the table). INTENT, HINT, ConeSep and PTHA+MTST remain plan-only, blocked on missing interpreter environments (HINT and ConeSep also on missing checkpoints; the PTHA checkpoint is staged). “Selected checkpoint / training” preserves the existing registry/status scientific label. For rows without adapters, “not implemented” says nothing about checkpoint availability.

| Model id | Composition adapter | Adapter module path | MM | Text-only black image | Image-only empty text | Category scopes | Selected checkpoint and training condition | Composition status / concrete blocker |
|---|---|---|---|---|---|---|---|---|
| `csmcir` | yes | `workbench.composition.model_csmcir` | blocked | blocked | blocked | shared, category-agnostic | `fashioniq`; 0% training noise, eval noise 0% | `TARGET_GPU_PENDING`; gallery target features require per-image `COT_ours2` captions absent from released CIRCUS inputs. Source basis: `workbench/third_party/CSMCIR/src/utils_csmcir.py:95-103` and `workbench/third_party/CSMCIR/src/lavis/models/blip2_models/blip2_qformer_cir_align_prompt_csmcir.py:498-535`. Local `qwen_captions` are a different generator and are not substituted. |
| `airknow` | not implemented | — | — | — | — | not applicable (no adapter) | `fiq_n05` / `fiq_n08`; training noise 50% / 80%, eval noise 0%; requested 20% unavailable | Existing `CODE_TESTED` / `BLOCKED_CHECKPOINT_CONDITION`; only 50/80 variants exist, never substitute for requested 20%. |
| `conesep` | yes | `workbench.composition.model_conesep` | yes | yes | yes | shared, category-agnostic | `fiq_n02`; training noise 20%, eval annotation noise 0% | Plan-only on this host: missing interpreter environment and missing checkpoint. Selected noisy condition still requires `include_flagged_conditions`; full composition evaluation remains `TARGET_GPU_PENDING`. |
| `ptha_mtst` | yes | `workbench.composition.model_ptha_mtst` | yes | yes | yes | shared, category-agnostic | `cirr_transfer`; CIRR→FashionIQ cross-dataset transfer, not a FashionIQ reproduction | Plan-only on this host: checkpoint staged, missing interpreter environment; requires isolated Python 3.9 / torch 2.0.1 runtime and declared inputs. Full composition evaluation remains `TARGET_GPU_PENDING`. |
| `habit` | yes | `workbench.composition.model_habit` | yes | yes | yes | shared, category-agnostic | `fiq_n02`; training noise 20%, eval annotation noise 0% | Real-data GPU smoke passed on RTX 3050 (`sm_86`): `fiq_n02` COMPLETED all three conditions (~70 min), `ndcg_mm` 0.2881, `composition_gap_ndcg` 0.2856; `--resume` reused the run in 26 s. Selected noisy condition still requires `include_flagged_conditions`; full composition evaluation remains `TARGET_GPU_PENDING`. |
| `intent` | yes | `workbench.composition.model_intent` | yes | yes | yes | shared, category-agnostic | `fiq_n02`; training noise 20%, eval annotation noise 0% | Plan-only on this host: missing interpreter environment (`WORKBENCH_INTENT_PYTHON`/provisioned env). Selected noisy condition still requires `include_flagged_conditions`; full composition evaluation remains `TARGET_GPU_PENDING`. |
| `hint` | yes | `workbench.composition.model_hint` | yes | yes | yes | shared, category-agnostic | `fashioniq`; training noise unstated, eval noise 0% | Plan-only on this host: missing interpreter environment and missing checkpoint; existing `PUBLIC_ASSET_SETUP` blocker remains. Full composition evaluation remains `TARGET_GPU_PENDING`. |
| `encoder` | yes | `workbench.composition.model_encoder` | yes | yes | yes | shared, category-agnostic | `fashioniq`; training condition unknown, eval noise 0%; checkpoint mapping remains `UNVERIFIED` | Real-data GPU smoke passed on RTX 3050 (`sm_86`): 8 queries × 3 conditions over the full 74,381-image gallery, COMPLETED, `composition_gap_ndcg` 0.5204 (new attempt `1410522b57d4fe79083c60a6`); the earlier smoke `17ef444e30b829ac22f2d3c8` reported 0.5227 under the old identity. `--resume` reused the run in 10.7 s with identical nDCG. The summary's `checkpoint_labels` now carries `reproduction_status: METRICS_MEASURED_MAPPING_UNVERIFIED` alongside `VERIFIED_METADATA`, so the mapping uncertainty travels with the artifact. Full composition evaluation remains `TARGET_GPU_PENDING`; checkpoint-mapping uncertainty remains. |
| `pair` | not implemented | — | — | — | — | not applicable (no adapter) | `pair_b1` / `pair_b2`; FashionIQ/training-condition mapping unverified | Existing `BLOCKED_SOURCE_ENTRYPOINT`; required pickle classes `model_vssp` / `MHTransformer_dis` are unpublished, and available checkpoint files do not establish FashionIQ mapping. |
| `clvc_net` | not implemented | — | — | — | — | not applicable (no adapter) | `author_drive_unresolved`; mapping unverified | Existing `BLOCKED_NO_SOURCE_REPLAY`; checkpoint mapping and source replay/entrypoint unresolved; ResNet-50 weights absent. |
| `dcnet` | not implemented | — | — | — | — | not applicable (no adapter) | `fashioniq_run_directory`; mapping unverified | Existing `AGGREGATE_PARSED`; run-directory mapping, validation-data preparation, GloVe validation PKLs, and ResNet-50 weights remain blocked. |
| `combiner_rn50x4_noft` | not implemented | — | — | — | — | not applicable (no adapter) | `rn50x4_noft_unresolved`; artifact mapping unverified | Existing `BLOCKED_CHECKPOINT_MAPPING`; author checkpoint mapping, RN50x4 base weights, and compatible data remain unresolved. |
| `clip4cir_rn50x4_fullft` | not implemented | — | — | — | — | not applicable (no adapter) | `rn50x4_fullft_unresolved`; paired Combiner/CLIP mapping unverified | Existing `BLOCKED_CHECKPOINT_BUNDLE`; author bundle pairing, RN50x4 base weights, and compatible data remain unresolved. |
| `tgcir` | not implemented | — | — | — | — | not applicable (no adapter) | `tgcir_archive_unresolved`; archive member/hash unverified | Existing `PREFLIGHT_IMPLEMENTED_BLOCKED`; archive mapping and evaluator entrypoint unresolved. |
| `sprc` | not implemented | — | — | — | — | not applicable (no adapter) | `sprc_fiq_unresolved` / separate `sprc_fiq_vitl`; both mappings unverified | Existing `BLOCKED_MODEL_MAPPING`; verify checkpoint/model mapping and assets; ViT-L is not the ViT-g paper-row variant. |
| `limn` | yes | `workbench.composition.model_limn` | yes | yes | yes | `dress`, `shirt`, `toptee` whole-model checkpoints | `base_iter0_all_categories`; training noise unstated, eval noise 0% | Bundle dispatch verified on this host, and the worker entrypoint `NameError` is fixed (top-level `import sys`). `checkpoint_by_id` now carries the bundle's own `mapping_status`, so a bundle-qualified LIMN no longer refuses with "mapping unresolved"; `WORKBENCH_LIMN_BATCH_SIZE` is honored (reuses `replay.limn.resolved_batch_size`) and recorded in the run identity as `adapter_batch_size`. With batch 8 the dress scope encoded the full 74,381 gallery; the full three-scope smoke needs a longer window and was budget-stopped, not failed. 119 ambiguous-category queries are out of scope and reduce coverage, not misses. Full composition evaluation remains `TARGET_GPU_PENDING` (`sm_120`). |
| `tme` | yes | `workbench.composition.model_tme` | yes | yes | yes | shared, category-agnostic | `fashioniq_0`; training noise 0%, eval noise 0%; no 20% checkpoint exists | Real-data GPU smoke on RTX 3050 (`sm_86`): with `WORKBENCH_TME_PYTHON` pointed at `envs/tme` (Python 3.10.13, torch 2.1.2+cu121) the adapter loads and encodes the FULL 74,381-image gallery, ~3.5 GiB peak VRAM, no OOM. The earlier "OOM on 4 GB" claim is **superseded**: the real gate was the wrong interpreter (`WORKBENCH_TME_PYTHON=envs/tme-adapted`, torch 2.8.0) against the registry pin `==2.1.2`. That historical-pin repair applies only to the older-GPU host with adaptation unset. On the target PC select `WORKBENCH_RUNTIME_ADAPTATION=sm_120`; TME's profile accepts the already-provisioned `envs/tme-adapted` (2.8.0 / 0.23.0 cu128), whose floor is the declared 2.7.1 / 0.22.1 cu128 pair — no downgrade needed. Do not point that PC profile at torch 2.1.2. Retain 0% checkpoint and do not substitute another model's noisy variant; full composition evaluation remains `TARGET_GPU_PENDING`. |

The `csmcir` blocker is specifically the target-feature dependency, not the query-mode input wiring. The adapter reports all three conditions unavailable while those gallery target features cannot be produced. The seven other deferred entries retain their existing checkpoint/source/data reasons above; they have no composition adapter.

## Summary provenance and output isolation

Every summary must carry: protocol ID; gallery count and identity; subset and query count; model/checkpoint identity; condition set; preprocessing and ablation policy; adapter identity; category scopes; actual interpreter/runtime/GPU; seed; release artifact hashes; a synthetic-fixture label where applicable; and the paper-vs-release gallery note. Synthetic runs must remain labelled as fixtures. Results are written under `<artifacts>/composition/`; composition paths are checked to remain within that namespace and cannot overwrite native results.

The summary records `runtime.model_interpreter` (path, version, source, verification status). On this smoke it recorded `DEPENDENCY_VERSION_MISMATCH` because the operator-selected `WORKBENCH_RUNTIME_ADAPTATION=sm_120` contract (at that time torch `>=2.8,<2.9`) was compared against the historical torch-1.13.1 env; this is an honest env-vs-contract record, not a probe failure, and it did not block the run. The active contract later narrowed: eight of the nine models now pin `==2.7.1` / `==0.22.1` exactly, TME accepts `>=2.7.1,<2.9` / `>=0.22.1,<0.24`.

## PC command surface

The CLI exists at `workbench/scripts/composition_eval.py`; its parser defines `--config`, `--models`, `--subsets`, `--resume`, `--smoke [QUERY_LIMIT]`, `--plan`, `--prepare-data`, and `--seed`. These commands use parser-confirmed flags. `WORKBENCH_LIMN_BATCH_SIZE` is the batch knob for LIMN (see the run-identity note below). ENCODER smoke run `17ef444e30b829ac22f2d3c8` covered 8 queries over the full gallery, with composition_gap_ndcg 0.5227 on the 8-query full subset; that attempt predates the `adapter_batch_size` identity field, and the newer attempt `1410522b57d4fe79083c60a6` reports 0.5204. The summary labels both a smoke, not a full evaluation. The run also exercised the fixed inventory path: `missing_assets.status: "extension_mismatch"`, `unresolvable_gallery_count: 0`, and `local_resolved_by_stem_count: 74381` (58,966 via `.png` stems, 15,415 exact). The earlier `--plan` count of 58,966 came from pre-fix code. Remaining commands are the full-cohort surface on the target PC:

```bash
# 1. Controller bootstrap. `workbench/run.sh` prefers `./.venv/bin/python` and never takes a model
#    interpreter from PATH, so the controller environment is built once, here. The controller needs
#    Python >= 3.11 (`pyproject.toml` `requires-python`); model interpreters are separate and
#    per-model. Composition needs no Docker and no `pc.sh`.
python3 -m venv .venv
.venv/bin/pip install -r workbench/backend/requirements.txt .

# 2. Roots. Every one is a placeholder the operator sets to an ABSOLUTE path on this PC before the
#    next line; nothing below repeats a hand-typed path. These are the names the code READS
#    (`workbench/backend/operator_config.py::_KEYS`).
export CIR_REPO_ROOT="$PWD"                              # SET: the CODE checkout (scripts + registry live here)
export CIR_PC_ROOT="$HOME/cir-pc"                        # SET: root on the PC's large volume (SHELL-LOCAL ONLY)
export FASHIONIQ_ROOT="$CIR_PC_ROOT/FashionIQ"           # SET: canonical FashionIQ: images/, image_splits/, captions/
export WORKBENCH_ARTIFACTS_ROOT="$CIR_PC_ROOT/artifacts"
export WORKBENCH_CHECKPOINT_ROOT="$CIR_PC_ROOT/artifacts/checkpoints"
export WORKBENCH_RESULTS_ROOT="$CIR_PC_ROOT/artifacts/results"
export WORKBENCH_THIRD_PARTY_ROOT="$CIR_PC_ROOT/third_party"      # SET: pinned upstream checkouts live here
export WORKBENCH_CIRCUS_RELEASE_ROOT="$CIR_PC_ROOT/circus_release"
export HF_HOME="$CIR_PC_ROOT/hf"                         # optional: Hugging Face cache root for declared assets
export CIR_ENVS_ROOT="$CIR_PC_ROOT/model-envs"           # SET: root holding the per-model interpreters (SHELL-LOCAL ONLY)

#    `CIR_DATASET_ROOT`/`CIR_ARTIFACTS_ROOT`/`CIR_CHECKPOINT_ROOT`/`CIR_RESULTS_ROOT`/
#    `CIR_THIRD_PARTY_ROOT`/`CIR_ENVS_ROOT`/`CIR_REGISTRY_PATH` used to be exported here: NOTHING
#    reads them, so they configured nothing while looking like they did. Only `CIR_REPO_ROOT` and
#    `CIR_DATA_ROOT` are real `CIR_*` settings. `CIR_PC_ROOT`/`CIR_ENVS_ROOT` remain above only as
#    shell-local shorthand. `WORKBENCH_REGISTRY_PATH` (the custom-registry override, below) is
#    PROCESS-ENVIRONMENT ONLY: `workbench.env` rejects it as an unknown setting.

# 2b. Mirror ONLY the five roots the standalone commands share into the operator env file, so the
#     planner/verifier/manager path (which uses `resolve_config()`, never the YAML) agrees with the
#     YAML. The agreement is proven in step 3, once the YAML config exists.
cp workbench/config/workbench.env.example workbench/config/workbench.env
$EDITOR workbench/config/workbench.env
#    Set CIR_REPO_ROOT and each mapped root to the exact absolute path above. Process exports win.
#    For a CUSTOM paths.registry, export WORKBENCH_REGISTRY_PATH to that absolute registry FILE
#    (process environment only — the env file rejects the name):
export WORKBENCH_REGISTRY_PATH="$CIR_REPO_ROOT/workbench/registry/models.yaml"

# 3. The composition config is the AUTOPILOT YAML, not `workbench.env`. Start from the checked-in
#    cohort profile — it already carries `checkpoint_policy: include_flagged_conditions` and the
#    authorized `checkpoint_overrides` — then set its six `paths` keys to the exported values above.
#    `paths.registry` must name an existing FILE and `paths.dataset_root` an existing DIRECTORY;
#    relative values resolve against the YAML's own directory, so use absolute paths here.
cp workbench/configs/pc_autopilot.pc_cohort.yaml ~/cir_pc.yaml
$EDITOR ~/cir_pc.yaml
#    paths.registry     = $WORKBENCH_REGISTRY_PATH     (an absolute FILE)
#    paths.dataset_root = $FASHIONIQ_ROOT
#    paths.artifacts    = $WORKBENCH_ARTIFACTS_ROOT
#    paths.checkpoints  = $WORKBENCH_CHECKPOINT_ROOT
#    paths.results      = $WORKBENCH_RESULTS_ROOT
#    paths.third_party  = $WORKBENCH_THIRD_PARTY_ROOT
#    Schema — `workbench/backend/autopilot.py:load_config` rejects an unknown OR missing key:
#      schema_version: 1 · selected_models · checkpoint_policy · checkpoint_overrides
#      paths: {registry, dataset_root, artifacts, checkpoints, results, third_party}
#      budgets: {max_download_bytes, max_models, top_k}
#      permissions: {allow_network, allow_large_downloads, allow_env_install, allow_preparation,
#                    allow_gpu_eval, allow_gpu_preparation, allow_index_write}
#    With the YAML in place, PROVE the agreement with the mirrored steps above: this must print
#    verdict=MATCH, and a mismatch means stop — standalone preparation, inventory, doctor and
#    verification would then read DIFFERENT roots than the run.
.venv/bin/python workbench/scripts/print_roots.py --config ~/cir_pc.yaml

# 4. Runtime profile. Select the declared adaptation BEFORE planning or provisioning.
export WORKBENCH_RUNTIME_ADAPTATION=sm_120

# 5. Model environments. PICK ONE MODE PER MODEL — the two modes damage each other on the same model.
#
# 5a. FRESH (the PC route): the planner creates each group's own prefix under `artifacts`, records
#     the interpreter in `<artifacts>/environment_assignment.json`, and every later stage reads it.
#     Preconditions: the pinned source is present under `WORKBENCH_THIRD_PARTY_ROOT` (stage it with
#     `sync_upstreams.py --model <id> --output-root "$WORKBENCH_THIRD_PARTY_ROOT"`, or let
#     `autopilot --apply` acquire it), and a `conda`/`mamba`/`micromamba` solver is on PATH.
#     NEVER preset `WORKBENCH_<MODEL>_PYTHON` to a path that does not exist yet: status reports
#     CONFIGURED_MISSING and `--create` refuses BEFORE the planner runs (exit 1). Because
#     `model_settings.setting` treats an EMPTY value as unset, `export WORKBENCH_ENCODER_PYTHON=`
#     does NOT clear a `workbench.env` entry for the same name — remove the line, or use `unset`.
unset WORKBENCH_ENCODER_PYTHON
.venv/bin/python workbench/scripts/manage_environment.py --model encoder --create --allow-env-install --dry-run
.venv/bin/python workbench/scripts/manage_environment.py --model encoder --create --allow-env-install --allow-network
#    The dry run is read-only and prints the group the planner WOULD form; the second command
#    provisions it. `--create` without `--allow-env-install` is a usage error (exit 2), and the
#    manager never installs into the controller environment.
#
# 5b. ALTERNATE (operator-staged interpreter): provision the environment yourself, then point the
#     registry's own `python_env_var` at it and verify. The prefix must satisfy the profile selected
#     in step 4 (`sm_120` selects the ADAPTED python/torch), or status reports
#     PYTHON_VERSION_MISMATCH. Do NOT do this and 5a for the same model.
export WORKBENCH_ENCODER_PYTHON="$CIR_ENVS_ROOT/encoder/bin/python"
.venv/bin/python workbench/scripts/manage_environment.py --model encoder --json

# 5c. Resolve the FRESH-mode interpreter and prove the CUDA runtime. Fresh mode (5a) publishes the
#     interpreter to `<artifacts>/environment_assignment.json`; a CHILD PROCESS CANNOT EXPORT A
#     SHELL VARIABLE BACK TO YOUR SHELL, so a pasted `"$WORKBENCH_ENCODER_PYTHON" ...` expands to an
#     EMPTY executable and fails with exit 127 (`command not found`). Resolve it from the assignment
#     instead (the same resolution every later stage uses), then check Python/Torch/vision/CUDA/GPU
#     and run a real CUDA allocation + matmul + synchronization before the GPU-heavy steps:
.venv/bin/python - <<'PY'
from pathlib import Path
import subprocess

from workbench.backend.operator_config import resolve_config
from workbench.backend.registry import model_by_id
from workbench.backend.runtime import resolve_model_interpreter

config = resolve_config()
interpreter = resolve_model_interpreter(model_by_id("encoder"), config)
if not interpreter or not Path(interpreter).is_file():
    raise SystemExit("ENCODER interpreter is unresolved; inspect roots/assignment/status first")
print("Resolved ENCODER interpreter:", interpreter, flush=True)

code = """import torch, torchvision
print('versions:', torch.__version__, torchvision.__version__, torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
a = torch.randn(512, 512, device='cuda')
b = a @ a
torch.cuda.synchronize()
assert b.isfinite().all().item()
print('CUDA allocation/matmul/synchronization OK')
"""
subprocess.run([interpreter, "-c", code], check=True)
PY

# 6. Status inspection — VALID flag combinations only:
#      --list [--json]        inventory every registry model
#      --model X [--json]     one model's status (the ONLY per-model form)
#    `--list` together with `--model` is a usage error (exit 2); `--json` combined with `--create` is
#    a VALID parse that prints status JSON and provisions NOTHING (exit 1, no environment created).
.venv/bin/python workbench/scripts/manage_environment.py --list --json
.venv/bin/python workbench/scripts/manage_environment.py --model encoder --json
#    Exit codes are evidence, not failure: `--list --json` returns 1 while any required environment
#    is unconfigured, and `--model X --json` returns 1 while that model is not `ready`.

# 7. Plan: prints the selected work and the expected output paths, writes no state.
./workbench/run.sh composition-eval --config ~/cir_pc.yaml --plan

# 8. Verify/inventory the released data. This checks the release files against their recorded
#    hashes and inventories missing gallery/reference/positive images; it does NOT download a
#    missing asset. A missing member is reported by name and has to be staged first.
./workbench/run.sh composition-eval --config ~/cir_pc.yaml --prepare-data

# 9. Smoke one GPU-heavy model at a time; the runner serializes model runs.
./workbench/run.sh composition-eval --config ~/cir_pc.yaml --models encoder,limn --smoke

# 10. Full cohort. Keep this on ONE line, or join the wrap with a trailing backslash: a copy that
#     wraps before `--resume` without the continuation runs the evaluation WITHOUT resume and then
#     tries `--resume` as the next command.
./workbench/run.sh composition-eval --config ~/cir_pc.yaml --models selected --subsets full,sf,validated --resume
```

Schedule one GPU-heavy model at a time; the runner serializes model runs. Fresh mode (5a) never exports the model interpreter into your shell — the manager publishes it to `<artifacts>/environment_assignment.json`, so resolve it there (step 5c) rather than assuming `$WORKBENCH_ENCODER_PYTHON` is set. Smoke mode limits queries (default 1) and preserves the full released gallery. Released data root is configured by `WORKBENCH_CIRCUS_RELEASE_ROOT`. `--prepare-data` verifies the released files against their recorded hashes and inventories missing gallery/reference/positive images — it never downloads a missing asset. `--models selected` reads the authorized cohort from `~/cir_pc.yaml` (`selected_models` and `checkpoint_overrides`), so it must be run with `--config`.

For the first pilot, name the models explicitly — use `--models encoder` (the per-model form shown in step 9) rather than `--models selected`. The checked-in cohort template ships `selected_models: all_registered`, which resolves every registry entry into the selection, including the eight PAUSED entries, and the run then ends with a named blocker per paused model instead of the pilot's results. Set a deliberate, verified list in your local `~/cir_pc.yaml` before using `--models selected`.

## Run identity, resume and worker lifecycle

The run identity now includes `adapter_batch_size`, the adapter's resolved batch. LIMN takes `WORKBENCH_LIMN_BATCH_SIZE` through `replay.limn.resolved_batch_size`, so that knob is honored by the composition adapter and recorded in the run identity. Because the field is part of the identity, the fingerprint changes for **every** run: ranks saved before this change are not reused across it, and a mismatch starts a fresh run by design.

Resume retries a recorded `PARTIAL` only when its per-query failure reasons are transient. Deterministic reasons never retry: the stable markers are `category scope failure`, `has no adapter embedding scope`, `no positive target resolved in exact released gallery`, `query features unavailable`, `invalid full-gallery exact rank`, and `query reference image is missing`.

Subprocess workers latch their first failure instead of restarting a doomed worker, tear the worker down deterministically, and leave the job directory in place so the runner copies the preserved worker stderr into `composition/logs` before `close()` removes the work tree (`rmtree` runs only on a clean close). Live evidence: a killed HABIT attempt still carried its worker stderr. This lifecycle covers TME, ENCODER and LIMN.

An explicit or `selected` request with N independent checkpoint variants dispatches N jobs; a LIMN bundle remains **one** job, and a partial bundle is still refused.

Composition workers inherit the process environment, and when `HF_HOME` is absent from it the file-resolved `HF_HOME` is injected instead. An exported `HF_HOME` wins over the file value; the sibling cache variables `HF_HUB_CACHE`/`HUGGINGFACE_HUB_CACHE` no longer suppress a file-only `HF_HOME` (they pass through untouched).

The frontend's mobile overflow is fixed (`workbench/frontend/style.css`: `section` `overflow-x: auto` plus a control `min-width`), so a 390×844 viewport now has `scrollWidth == 390`. A DOM-verified Playwright smoke for this lives at `workbench/frontend/tests/ui_smoke.py`; it is dev-only and is not collected by pytest.

## Target-PC runtime baseline (2026-10-29)

For the nine active composition models, `WORKBENCH_RUNTIME_ADAPTATION=sm_120` selects a CUDA 12.8
wheel profile. Eight pin exact `torch==2.7.1` / `torchvision==0.22.1` — PyTorch 2.7 is the first
stable Blackwell line and 2.7.1 is its bugfix release. TME instead declares a floor-and-range
`torch>=2.7.1,<2.9` / `torchvision>=0.22.1,<0.24`: its already-provisioned
`workbench/envs/tme-adapted` (2.8.0 / 0.23.0+cu128, arch list includes sm_120, carries the full
TME stack) stays valid and is NOT re-provisioned or downgraded. The upstream author pins remain
historical contracts for older GPUs. Sources: [PyTorch 2.7 release](https://pytorch.org/blog/pytorch-2-7/)
and [official wheel pairings](https://pytorch.org/get-started/previous-versions/).

The planner and verifier consume the same registry profile. Provision a matching environment for
each remaining model, then update the corresponding `WORKBENCH_<MODEL>_PYTHON` assignment; TME needs
no such re-provision — point `WORKBENCH_TME_PYTHON` at the existing `envs/tme-adapted`. Do not copy a
laptop virtualenv or downgrade the controller environment. Keep every model in its existing isolated
environment/group; sharing this Torch baseline does not merge incompatible model dependencies.

On the PC, first export `WORKBENCH_RUNTIME_ADAPTATION=sm_120` (or put it in `workbench.env`),
provision the selected profile, run its actual CUDA capability check and then the ENCODER pilot.
`HF_HOME` is passed to workers as an absolute path, for both file and exported values; exported
values keep precedence. An absolute path remains the clearest operator setting.
This baseline declaration and CPU tests do not establish target-GPU forwards or paper reproduction.
The eight deferred model entries remain deferred, with no new checkpoints or adapters added here.
