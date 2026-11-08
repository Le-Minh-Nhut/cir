# Native-scope inventory — the 8 non-CIRCUS models

Branch `research/cir-failure-workbench`. Read-only scoping pass: no GPU work was run, no repo
file outside this directory was modified, git state untouched.

Evidence collected 2026-11-08. Reproduce any figure below with the commands in
`.evidence/` and the paths cited inline.

---

## 0. Headline

| question | answer |
|---|---|
| models in registry | **17** (`workbench/registry/models.yaml`) |
| have `composition_adapter` (CIRCUS cohort, other agent) | **9**: `csmcir, conesep, ptha_mtst, habit, intent, hint, encoder, limn, tme` |
| **native cohort (this document)** | **8**: `airknow, pair, clvc_net, dcnet, combiner_rn50x4_noft, clip4cir_rn50x4_fullft, tgcir, sprc` |
| checkpoint-variant rows in scope | **11** (8 models; `airknow` and `sprc` have 2 variants each) |
| `RUNNABLE_NOW` | **0** |
| `RUNNABLE_AFTER_ENV` | **2** (both `airknow` rows — and both are simultaneously `RUNNABLE_WITH_DOCUMENTED_CONDITION`) |
| `BLOCKED_CHECKPOINT_MAPPING` | **7** |
| `BLOCKED_CHECKPOINT_MISSING` | **2** |
| native runs visible in `GET /api/runs` today | **0**, and 0 is structurally guaranteed (see §5) |

Three separate problems are tangled together and must not be conflated:

1. **A path/configuration defect** — `WORKBENCH_CHECKPOINT_ROOT` points at
   `.../cohort/downloads`, which holds only the 9 CIRCUS models. Every native weight is staged at
   `.../checkpoints/`, which nothing is configured to read. This makes *all 8* native rows report
   `checkpoint missing` and is a one-line config change, not 8 blocked models. It is **not** the
   real blocker for 7 of them.
2. **Genuine scientific blockers** — checkpoint→benchmark mapping is unverified for 7 of 8
   (`checkpoint_mapping_status: UNVERIFIED`), and for 2 the declared artifact does not exist at all.
   These survive any config fix.
3. **No execution environment for any native model** — all 8 declared `WORKBENCH_*_PYTHON` vars are
   unset, and every candidate interpreter on this host fails the declared Python/torch contract.

---

## 1. Full inventory

### 1.1 Cohort split (verified)

```
WITH composition_adapter: csmcir conesep ptha_mtst habit intent hint encoder limn tme   (9)
NATIVE (no composition_adapter): airknow pair clvc_net dcnet
                                 combiner_rn50x4_noft clip4cir_rn50x4_fullft
                                 tgcir sprc                                          (8)
```

Protocol registry (`models.yaml:14-26`), 3 protocols, all `evaluation_noise_pct: 0`:

| protocol_id | label | `literature_split_label` | gallery | reference eligible? |
|---|---|---|---|---|
| `fashioniq_original_split` | Original Split (Full Gallery, Reference Included) | `original` | full `split.<cat>.val.json` order | **yes** |
| `fashioniq_full_gallery_ref_excluded` | Original Full Gallery, Reference Excluded | `original` | same full split, same order | no (masked per query) |
| `fashioniq_val_split` | Val Split (Pair-Union Gallery, Reference Excluded) | `val` | first-seen union of each annotation's reference then target | no |

Semantics source: `src/evaluation/fashioniq.py:174-198` (`build_original_gallery`,
`build_fashioniq_gallery`, `build_pair_union_gallery`).

### 1.2 The 8 native models

Every row below was read from `workbench/registry/models.yaml`; `source_dir` presence and pin
verified with `workbench.backend.vendored_source.source_provenance` (all 8: `verified: true`,
`error: null`, `state: vendored_in_repo`, actual commit == `upstream_commit_sha`).

| model_id | method | year | native_protocol | supported | layout / ext | prep contract | source_dir | pin | src verified | reproduction_status | command_status | strict_clean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `airknow` | Air-Know | 2026 | `fashioniq_full_gallery_ref_excluded` | same | `fashioniq_ilearn_resized_training` / `.jpg` | `manual_ilearn_resized` | `AirKnow` | `6ab6fa95b746…` | ✅ | `NOT_RUN` | (none) | **false** |
| `pair` | PAIR | 2025 | `fashioniq_val_split` | same | `..._resized_training` / `.jpg` | `manual_ilearn_resized` (`pair_fashioniq`) | `PAIR` | `bfb0b98cdb4c…` | ✅ | `BLOCKED_CHECKPOINT_MAPPING_UNVERIFIED` | `COMMAND_BLOCKED_SOURCE_ENTRYPOINT` | false (null cond.) |
| `clvc_net` | CLVC-Net | 2021 | `fashioniq_val_split` | same | `fashioniq_clvc_resized` / `.jpg` | `manual_clvc_resized` (`clvc_net_fashioniq`) | `CLVCNet` | `bd9b68894898…` | ✅ | `BLOCKED_CHECKPOINT_MAPPING` | `COMMAND_BLOCKED_SOURCE_ENTRYPOINT` | null |
| `dcnet` | DCNet | 2021 | `fashioniq_full_gallery_ref_excluded` | same | `fashioniq_dcnet` / `.jpg` | `manual_dcnet` (`dcnet_fashioniq`) | `DCNet` | `68c79d38569f…` | ✅ | `BLOCKED_CHECKPOINT_MAPPING` | `COMMAND_AUDITED` | null |
| `combiner_rn50x4_noft` | Combiner RN50x4 noft | 2023 | `fashioniq_original_split` | same | `fashioniq_standard` / `.jpg` | `manual_standard` (`clip4cir_fashioniq`) | `CLIP4Cir` | `dfed9f748a8a…` | ✅ | `BLOCKED_CHECKPOINT_MAPPING` | `COMMAND_AUDITED` | null |
| `clip4cir_rn50x4_fullft` | CLIP4Cir RN50x4 fullft | 2023 | `fashioniq_original_split` | same | `fashioniq_standard` / `.jpg` | `manual_standard` (`clip4cir_fashioniq`) | `CLIP4Cir` | `dfed9f748a8a…` | ✅ | `BLOCKED_CHECKPOINT_BUNDLE` | `COMMAND_AUDITED` | null |
| `tgcir` | TG-CIR | 2023 | `fashioniq_val_split` | same | `..._resized_training` / `.jpg` | `manual_ilearn_resized` (`tgcir_fashioniq`) | `TGCIR` | `65fa78eaf8ca…` | ✅ | `BLOCKED_COMMAND` | `COMMAND_BLOCKED_SOURCE_ENTRYPOINT` | null |
| `sprc` | SPRC | 2024 | `fashioniq_original_split` | same | `fashioniq_standard` / **`.png`** | `manual_standard` (`sprc_fashioniq`) | `SPRC` | `2935a5397732…` | ✅ | `BLOCKED_CHECKPOINT_MAPPING` | `COMMAND_BLOCKED_MODEL_MAPPING` | null |

`canonical_dataset_link: fashionIQ_dataset` on `combiner_rn50x4_noft`, `clip4cir_rn50x4_fullft`,
`sprc` (in-checkout symlink). `approved_source_patches` only on `dcnet`
(`workbench/patches/dcnet_checkpoint_weights_only_optout.patch`, `base/base_trainer.py`,
expected derived sha `f265cefe…`). **Verified applied**: `approved_patch_state` returns
`applied: true, detail: "base/base_trainer.py matches the recorded post-patch digest"`. The other 7
declare no patch.

One-shot `cat > /dev/null` summary of the numeric/boolean fields:

| model | env python | env pytorch | env torchvision | env cuda | packages | `python_env_var` | `adapted_runtime` |
|---|---|---|---|---|---|---|---|
| airknow | 3.8.10 | 2.1.0 | 0.16.0 | 12.6 | 19 declared | `WORKBENCH_AIRKNOW_PYTHON` | **absent** |
| pair | 3.8.10 | 2.0.0 | UNKNOWN | 11.8 | none | `WORKBENCH_PAIR_PYTHON` | **absent** |
| clvc_net | 3.7.6 | 1.6.0 | UNKNOWN | 10.2 | `torchvision` | **`WORKBENCH_CLVC_PYTHON`** (not `_CLVC_NET_`) | **absent** |
| dcnet | 3.7.7 | 1.4.0 | 0.5.0 | UNKNOWN | `nltk==3.5`, `spacy==2.3.0` | `WORKBENCH_DCNET_PYTHON` | **absent** |
| combiner_rn50x4_noft | 3.8 | 1.11.0 | 0.12.0 | UNKNOWN | `pandas==1.4.2`, `comet-ml==3.21.0`, pinned `clip` | `WORKBENCH_CLIP4CIR_PYTHON` | **absent** |
| clip4cir_rn50x4_fullft | 3.8 | 1.11.0 | 0.12.0 | UNKNOWN | same as above | `WORKBENCH_CLIP4CIR_PYTHON` (shared) | **absent** |
| tgcir | UNKNOWN | 1.7.0 | 0.8.0 | 11.0 | `clip` pinned, `numpy==1.22.3`, `pandas==1.4.2`, `Pillow` | `WORKBENCH_TGCIR_PYTHON` | **absent** |
| sprc | 3.9 | 2.0.1 | 0.15.2 | UNKNOWN | 16 declared (`spacy==3.7.2`, `timm==0.9.12`, `transformers==4.36.2`, …) | `WORKBENCH_SPRC_PYTHON` | **absent** |

**`adapted_runtime` count across the whole registry is 9 — all nine are CIRCUS models.** No native
model declares an `sm_120` modernization path, so `WORKBENCH_RUNTIME_ADAPTATION` cannot legalize a
modern interpreter for any native model. Fixing that is a registry edit, which is out of this
agent's ownership and requires approval per `AGENTS.md`.

### 1.3 Checkpoint variants — declaration vs disk

Real digests below were computed in this session (`sha256sum`, disk I/O only) over the files under
`workbench/artifacts/checkpoint_selection/20261106T002617+0700/`.

| model | checkpoint_id | filename (declared) | mapping | noise% | expected sha / bytes | **on disk** | **measured sha256** | **measured bytes** |
|---|---|---|---|---|---|---|---|---|
| airknow | `fiq_n05` | `airknow_fiq_0.5.pt` | `VERIFIED_METADATA` | 50 | null / null | ✅ `checkpoints/airknow/airknow_fiq_0.5.pt` | `f4d25ed9e532f2ee3671f6edadb1048e73113246ec056b413bde75dfe3f79bc6` | 2 724 056 638 |
| airknow | `fiq_n08` | `airknow_fiq_0.8.pt` | `VERIFIED_METADATA` | 80 | null / null | ✅ `checkpoints/airknow/airknow_fiq_0.8.pt` | `756fc0bdc43d9e0f0cae13c8a02b0279f64e62092ce465c1e60bda45e179c54e` | 2 724 056 638 |
| pair | `pair_b1` | `pair-B1.pt` | `UNVERIFIED` | — | null / 694 958 565 | ✅ `checkpoints/pair/pair-B1.pt` | `c4d861ecd0baa3f1427fc7a853943f5614baf87def1cdd9c4bb7583ee925fbf2` | 694 958 565 (size matches) |
| pair | `pair_b2` | `pair-B2.pt` | `UNVERIFIED` | — | null / 694 872 371 | ✅ `checkpoints/pair/pair-B2.pt` | `74346fd7de6efdafe8dde70458316286a60ceb2d7b419efed55193418aeb3f03` | 694 872 371 (size matches) |
| clvc_net | `author_drive_unresolved` | `checkpoint-unresolved.pt` | `UNVERIFIED` | — | null / null | ❌ declared name absent. **Six different files exist**: `checkpoints/clvc/{dress,shirt,toptee}_{local,global}_model.pt` | local: `33d44186…`, `2ba8e320…`, `d1fa9170…`; global: `be41c2ba…`, `78b1d660…`, `c78a58c4…` | 217 168 235 / 218 603 243 / 218 308 843 / 176 437 419 / 177 872 427 / 177 578 027 |
| dcnet | `fashioniq_run_directory` | `fashioniq_dcnet` (dir; `config.json` + `trained_model.pth`) | `UNVERIFIED` | — | null / null | ❌ declared dir absent. Real run dir is `checkpoints/dcnet/` (also holds `val/{config.json,info.log,att_result.pt}`) | `trained_model.pth` = `130f031e0f7808ccf5f15630a691bd7af95ad9fadbf94f64acfce64e7d8b850f`; its `config.json` `data_dir` = `/home/heheboiz/data/cir/data/FashionIQ` | 2 877 310 172 |
| combiner_rn50x4_noft | `rn50x4_noft_unresolved` | `rn50x4-noft-unresolved.pt` | `UNVERIFIED` | 0 | null / null | ❌ declared name absent. Real weight: `checkpoints/clip4cir/FashionIQ/RN50x4_noft/fiq_comb_RN50x4_noft.pt` | `b31882fe6f7e3e194cb9cfda048926844ee0ccc9d4ebc5483fb3bacfa5972955` | 236 017 921 |
| clip4cir_rn50x4_fullft | `rn50x4_fullft_unresolved` | `rn50x4-fullft-unresolved.pt` (`bundle_placeholder`; needs `combiner_state.pt` **and** `clip_state.pt`) | `UNVERIFIED` | 0 | null / null | ❌ declared dir absent. Real pair: `.../RN50x4_fullft/fiq_comb_RN50x4_fullft.pt` + `fiq_clip_RN50x4_fullft.pt` | comb `7c7c0a10fb4c995cb1ded9ad70a58891d50c622263c54131523bfb232b493f8e`; clip `4111ac139cf96ac0cf31b30bf8f55b654247b06f0f5fbee4fe8dcc8c1c7cfd2f` | 236 017 921 / 713 874 695 |
| tgcir | `tgcir_archive_unresolved` | `tgcir-fashioniq-unresolved.pt` | `UNVERIFIED` | — | null / null | ❌ declared name absent. Candidate: `downloads/tgcir/x/checkpoints/Fashioniq.pt` (an extracted archive member, mapping never established) | `c37768f374c75bebff1299e870518c23bf6db3dc8a75c4d1f729ccd6dd02e966` | 607 928 637 |
| sprc | `sprc_fiq_unresolved` | `sprc_fiq.pt` | `UNVERIFIED` | — | null / null | ❌ **absent everywhere on disk** (OneDrive link never downloaded) | — | — |
| sprc | `sprc_fiq_vitl` | `sprc_fiq_vitl.pt` | `UNVERIFIED` | — | null / **1 459 756 541** | ✅ `checkpoints/sprc/sprc_fiq_vitl.pt` | `19386104271c01ea03ee2177575c0e55f98edd3ba61a116d78f2e61ab5dbea08` | **1 459 756 541 (exact match to declaration)** |

Notes that matter for the matrix:

- **`expected_sha256` is `null` on all 11 variants.** No upstream digest exists to enforce, so a
  byte-level provenance claim is unverifiable for every native checkpoint by construction. Only
  `airknow`'s two files have a recorded digest anywhere, and that digest is *self-recorded* by the
  workbench download step (`manifests/airknow_hashes.json`: `match: true`), not published upstream.
- `sprc_fiq_vitl` is declared `scientific_variant: true` — CLIP ViT-L/14 where the paper used
  EVA-CLIP ViT-g/14, per upstream issue #4. It is a *documented variant*, not a reproduction.
- `sprc_fiq_vitl` matching its declared byte count exactly (1 459 756 541) is the strongest
  provenance evidence of any native checkpoint.
- Prior sessions already produced real measured metrics for 8 of these
  (`measurements/all_measured_summary.json`, `measurements/selection_decisions.json`), but those runs
  went through `workbench/replay/*.py` wrappers launched ad hoc
  (`checkpoint_selection/.../scripts/run_clvc.py` etc.), **not** through `evaluate_models.py`, so
  they carry no completion proof and are not publishable evidence.

### 1.4 `WORKBENCH_CHECKPOINT_ROOT` — the path defect

```
workbench/config/workbench.env:
WORKBENCH_CHECKPOINT_ROOT=…/checkpoint_selection/20261106T002617+0700/cohort/downloads
```

`checkpoint_path()` (`workbench/backend/registry.py:1855-1858`) resolves every variant to
`<root>/<model_id>/<filename>`. Contents of the configured root (verified):

```
…/cohort/downloads/{conesep csmcir encoder habit hint intent limn ptha_mtst tme}   ← 9 CIRCUS only
```

Native weights live one level up in a **sibling** directory:
`…/checkpoint_selection/20261106T002617+0700/checkpoints/<model>/…`.

Consequence, measured with `checkpoint_missing_paths` under each root:

| variant | under configured root | under `…/checkpoints` |
|---|---|---|
| airknow fiq_n05, fiq_n08 | missing | **present** |
| pair pair_b1, pair_b2 | missing | **present** |
| sprc sprc_fiq_vitl | missing | **present** |
| clvc_net, dcnet, combiner, clip4cir, tgcir, sprc_fiq | missing | missing (declared names do not exist under either) |

So 5 of 11 rows are blocked *only* by the root mismatch; the other 6 are blocked by genuinely absent
or unmapped artifacts.

---

## 2. What "in scope" genuinely means per model

### 2.1 Data layouts — all present

`workbench.backend.fashioniq_layout.paths_for_layout(root=data/FashionIQ, layout)` returns **0
missing** for all 8 models. Measured image trees:

| tree | files | extension | covers val gallery? |
|---|---|---|---|
| `images/` | 74 381 | `.png` | full original split |
| `resized_image/dress\|shirt\|toptee` | 3 817 / 6 346 / 5 373 | `.jpg` | iLearn layouts (union 15 415) |
| `resized_images/` (flat) | 15 415 | `.jpg` | DCNet |
| `captions/cap.*.glove.val.pkl` | all 3 present | — | DCNet |
| `prepared/clip4cir/fashionIQ_dataset/images` | 15 415 `id.jpg` → symlink to `images/id.png` | `.jpg` | **0 missing ids, 0 dangling** |
| `prepared/sprc/fashionIQ_dataset/images` | 15 415 `id.png` | `.png` | **0 missing ids, 0 dangling** |

Val gallery sizes: dress 3 817, shirt 6 346, toptee 5 373, union 15 415 (all ids unique).

**But** `fashioniq_standard` declares extension `.jpg`
(`fashioniq_layout.py:93`) while `images/` contains only `.png`, which makes
`image_extension_problems` emit `3817 of 3817 dress validation images lack .jpg …` for
`combiner_rn50x4_noft`, `clip4cir_rn50x4_fullft`, and `sprc`. The `.jpg` views that do satisfy the
evaluators exist only under `prepared/`, and the checkout symlinks point there — which trips a
different blocker: `canonical dataset link resolves to …/prepared/clip4cir/fashionIQ_dataset,
expected /home/heheboiz/data/cir/data/FashionIQ`. Both are preflight *policy* conflicts, not missing
data. **The data is all there.**

### 2.2 External assets (measured)

| asset | declared | measured on disk | verdict |
|---|---|---|---|
| OpenAI CLIP **RN50x4** base | sha `7e526bd1…405fdd` | `~/.cache/clip/RN50x4.pt`, sha `7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd` | **VERIFIED_LOCAL_ASSET** — digest matches exactly |
| torchvision **ResNet-50** | prefix `19c8e357`, cache `hub/checkpoints/resnet50-19c8e357.pth` | `hub/checkpoints/resnet50-0676ba61.pth` (sha `0676ba61…`) | **CONFIGURED_MISSING** — wrong *filename*; see root cause below |
| OpenAI CLIP **ViT-B/16** (tgcir) | `BLOCKED_EXTERNAL_ASSET_PROVENANCE` | `~/.cache/clip/ViT-B-16.pt` sha `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f` present | **UNRESOLVED_PROVENANCE** — file exists, registry refuses to trust it |
| LAVIS **BLIP-2** (sprc) | `BLOCKED_EXTERNAL_ASSET_PROVENANCE`, `WORKBENCH_SPRC_BLIP2_WEIGHTS` unset | `hub/checkpoints/blip2_pretrained.pth` (712.4 MB) and `blip2_pretrained_vitL.pth` (698.9 MB) present | **UNRESOLVED_PROVENANCE** |

The ResNet-50 row is a **registry-metadata staleness bug, not a missing asset**: the declaration
(`preparation_contracts.yaml`, `clvc_net_fashioniq` / `dcnet_fashioniq`) carries both
`expected_local_location: null` **and** `cache_path_suffix: hub/checkpoints/resnet50-19c8e357.pth`.
`external_asset_statuses` (`registry.py:406-432`) treats a `cache_path_suffix` asset as
*"cache-shaped"* and resolves it **below the cache root by suffix**, ignoring
`location_env_var` entirely — which is why setting `WORKBENCH_{CLVC,DCNET}_RESNET50_WEIGHTS` had no
effect on the path checked. The file on disk is `resnet50-0676ba61.pth`, the current torchvision
ResNet-50 entry; the declaration names the legacy `19c8e357` file. Fixing it means amending the
contract's `cache_path_suffix` (registry change, needs owner approval) — the weights for CLVC-Net and
DCNet are effectively already present.

### 2.3 Interpreters — nothing wired, nothing compatible

All 8 `WORKBENCH_*_PYTHON` variables are **absent from `workbench.env`**. Candidate env dirs exist
for 5 of 8 (`airknow-adapted`, `pair-adapted`, `clip4cir-adapted`, `sprc-adapted`,
`dcnet-prepare`); **none exists for `clvc_net` or `tgcir`**.

Measured with `workbench.backend.runtime.verify_environment(model, probe=True)`:

| model | candidate interpreter | `python -V` | verdict |
|---|---|---|---|
| airknow | `envs/airknow-adapted/bin/python` | 3.11.15 | **BLOCKED / PYTHON_VERSION_MISMATCH** — `expected Python 3.8.10, found 3.11.15` |
| pair | `envs/pair-adapted/bin/python` | 3.11.15 | **BLOCKED / PYTHON_VERSION_MISMATCH** — `expected Python 3.8.10, found 3.11.15` |
| combiner + clip4cir | `envs/clip4cir-adapted/bin/python` | 3.11.15 | **BLOCKED / PYTHON_VERSION_MISMATCH** — `expected Python 3.8, found 3.11.15` |
| sprc | `envs/sprc-adapted/bin/python` | 3.11.15 | **BLOCKED / PYTHON_VERSION_MISMATCH** — `expected Python 3.9, found 3.11.15` |
| sprc (alt) | `envs/ilearn39-adapted/bin/python` | 3.9.25 | Python **OK**, then **BLOCKED / DEPENDENCY_VERSION_MISMATCH** — `torch: expected ==2.0.1, found 2.7.1+cu128` |
| dcnet | `envs/dcnet-prepare/bin/python` | 3.7.12 | **BLOCKED / PYTHON_VERSION_MISMATCH** — `expected Python 3.7.7, found 3.7.12`; and this env has **no torch at all** (`ModuleNotFoundError: No module named 'torch'`) |
| clvc_net | — | — | no candidate directory exists |
| tgcir | — | — | no candidate directory exists |

All five usable candidate envs carry **torch 2.7.1+cu128** (conda overlay on
`miniconda3/envs/teacher-sprc`) or nothing. Declared contracts are torch 1.4.0 / 1.6.0 / 1.7.0 /
1.11.0 / 2.0.0 / 2.0.1 / 2.1.0. Since no native model declares `adapted_runtime`, the runtime
verifier will refuse every one of these anyway — a modern stack is *not a legal contract* for these
models as currently declared.

Host GPU (metadata only, no load): RTX 5070 Ti, 16 303 MiB, driver 580.173.02, compute cap **12.0
(sm_120)**, CUDA 13.0 driver-level, torch wheels cu128, no `nvcc`.

### 2.4 The "no clean checkpoint" condition, stated exactly

`airknow` is the only native model whose checkpoint mapping is `VERIFIED_METADATA`, and its two
published weights are **noise-trained** (`checkpoint_training_noise_pct: 50` and `80`;
`strict_clean_checkpoint_available: false`). This is *not* "cannot run". It means the run is
legitimate **only** if labelled:

> `checkpoint_id` is `fiq_n05` / `fiq_n08`; `checkpoint_training_noise_pct` is **50** / **80**;
> `evaluation_noise_pct` is **0**. The label must read *trained with N% noise, evaluated on clean
> FashionIQ validation* — never "clean-trained". `select_checkpoints(airknow, STRICT_CLEAN)` returns
> `BLOCKED_CHECKPOINT`: *"no checkpoint is recorded as clean-trained under policy strict_clean:
> fiq_n05=50%, fiq_n08=80%."*

Under `INCLUDE_FLAGGED_CONDITIONS` airknow still returns
`BLOCKED_CHECKPOINT_SELECTION` — *"more than one source-backed variant is registered and none is
declared a bundle"*. So even at the policy level the operator must name the variant explicitly
(`--checkpoint fiq_n05`), which makes the run `policy_overridden: true`. The other 7 native models
record **no training condition at all** (`checkpoint_training_noise_pct: null`) — the honest label
there is *"training condition not published by the author"*, never 0%.

---

## 3. Runnability verdict matrix

One row per model × checkpoint × supported protocol. Each of the 11 variants supports exactly one
protocol (its `native_protocol`).

| # | model | checkpoint | protocol | verdict | root cause / condition label |
|---|---|---|---|---|---|
| 1 | `airknow` | `fiq_n05` | `fashioniq_full_gallery_ref_excluded` | **RUNNABLE_AFTER_ENV** (∧ `RUNNABLE_WITH_DOCUMENTED_CONDITION`) | needs an interpreter satisfying `python 3.8.10 / torch 2.1.0 / torchvision 0.16.0` **and** must be labelled `checkpoint_training_noise_pct=50`. Weights present + sha-recorded; mapping `VERIFIED_METADATA`; source pinned & clean. |
| 2 | `airknow` | `fiq_n08` | `fashioniq_full_gallery_ref_excluded` | **RUNNABLE_AFTER_ENV** (∧ documented condition) | same; label `checkpoint_training_noise_pct=80`. |
| 3 | `pair` | `pair_b1` | `fashioniq_val_split` | **BLOCKED_CHECKPOINT_MAPPING** | mapping `UNVERIFIED`: the author Hub lists `pair-B1.pt`/`pair-B2.pt` but establishes no FashionIQ–A/B identity. Also `command_status: COMMAND_BLOCKED_SOURCE_ENTRYPOINT` and no interpreter. Bytes present (sha recorded here). |
| 4 | `pair` | `pair_b2` | `fashioniq_val_split` | **BLOCKED_CHECKPOINT_MAPPING** | as above. |
| 5 | `clvc_net` | `author_drive_unresolved` | `fashioniq_val_split` | **BLOCKED_CHECKPOINT_MISSING** | declared `checkpoint-unresolved.pt` does not exist; the six real per-category files are staged under a different name nobody declared, and the Drive archive's member/category mapping is unresolved. Also `COMMAND_BLOCKED_SOURCE_ENTRYPOINT`, ResNet-50 asset missing, no interpreter, no env dir. |
| 6 | `dcnet` | `fashioniq_run_directory` | `fashioniq_full_gallery_ref_excluded` | **BLOCKED_CHECKPOINT_MAPPING** | run directory is staged as `checkpoints/dcnet/` but declared as `checkpoints/dcnet/fashioniq_dcnet`; mapping `UNVERIFIED`. Plus ResNet-50 filename mismatch, `dcnet-prepare` has no torch, and the `dataset/fashioniq` in-checkout symlink is absent. |
| 7 | `combiner_rn50x4_noft` | `rn50x4_noft_unresolved` | `fashioniq_original_split` | **BLOCKED_CHECKPOINT_MAPPING** | weights present (`fiq_comb_RN50x4_noft.pt`, sha measured) under a name the registry never declared; mapping `UNVERIFIED` — noft/fullft pairing with the author Drive folder is unresolved. Its declared assets (RN50x4 base) *are* verified. |
| 8 | `clip4cir_rn50x4_fullft` | `rn50x4_fullft_unresolved` | `fashioniq_original_split` | **BLOCKED_CHECKPOINT_MAPPING** | both required members present and hashed, but the declared `bundle_placeholder` path does not exist and the author's combiner/clip pairing is unverified. |
| 9 | `tgcir` | `tgcir_archive_unresolved` | `fashioniq_val_split` | **BLOCKED_CHECKPOINT_MAPPING** | archive member mapping + hash unresolved; `COMMAND_BLOCKED_SOURCE_ENTRYPOINT`; ViT-B/16 asset is `BLOCKED_EXTERNAL_ASSET_PROVENANCE`; no interpreter and no env dir at all. |
| 10 | `sprc` | `sprc_fiq_unresolved` | `fashioniq_original_split` | **BLOCKED_CHECKPOINT_MISSING** | `sprc_fiq.pt` appears **nowhere on disk** (author OneDrive link never acquired). Also `COMMAND_BLOCKED_MODEL_MAPPING` and BLIP-2 provenance unresolved. |
| 11 | `sprc` | `sprc_fiq_vitl` | `fashioniq_original_split` | **BLOCKED_CHECKPOINT_MAPPING** | file present, byte count **exactly** matches the declared 1 459 756 541 and sha recorded here — but mapping is `UNVERIFIED` and it is a declared `scientific_variant` (ViT-L instead of ViT-g). Also BLIP-2 provenance unresolved, `COMMAND_BLOCKED_MODEL_MAPPING`, no interpreter. |

### Counts

| verdict | rows | which |
|---|---|---|
| `RUNNABLE_NOW` | **0** | — |
| `RUNNABLE_AFTER_ENV` | **2** | airknow `fiq_n05`, `fiq_n08` (both also carry a mandatory condition label) |
| `RUNNABLE_WITH_DOCUMENTED_CONDITION` (as sole blocker) | **0** | none — every conditioned row also lacks an environment |
| `BLOCKED_CHECKPOINT_MAPPING` | **7** | pair_b1, pair_b2, dcnet, combiner, clip4cir_fullft, tgcir, sprc_vitl |
| `BLOCKED_CHECKPOINT_MISSING` | **2** | clvc_net, sprc_fiq |
| `NOT_APPLICABLE` | **0** | every variant supports its declared protocol |
| **total** | **11** | |

Notable: **8 of 8 models also fail the environment tier** and **7 of 8 fail
`checkpoint mapping unresolved`**. Even if the checkpoint root and every interpreter were fixed
today, 7 of 11 rows would still be refused by `runtime_blockers`
(`workbench/backend/runtime.py:3082-3084`) because `checkpoint_mapping_status !=
VERIFIED_METADATA`. That gate cannot be satisfied by any local action — it requires establishing the
upstream fact.

---

## 4. Exact commands

### 4.1 `RUNNABLE_NOW` rows

**There are none.** No native row can be launched today without changing something.

### 4.2 The two `RUNNABLE_AFTER_ENV` rows — command once the environment exists

Harness form (flags verified against `parser_for`, `evaluate_models.py:507-529` — `--model`,
`--checkpoint`, `--protocol`, `--dataset-root`, `--top-k`, `--run-id`, `--verify-gpu` all exist):

```bash
cd /home/heheboiz/data/cir
# GPU host only. Run one at a time (AGENTS.md: never two GPU jobs).
workbench/envs/ctl312h/bin/python workbench/scripts/evaluate_models.py \
  --model airknow \
  --checkpoint fiq_n05 \
  --protocol fashioniq_full_gallery_ref_excluded \
  --dataset-root /home/heheboiz/data/cir/data/FashionIQ \
  --top-k 200 \
  --verify-gpu
```

and the identical command with `--checkpoint fiq_n08`.

Prerequisites that must be true first (all currently false):

1. **`WORKBENCH_AIRKNOW_PYTHON` set** (in `workbench.env` — but that file is off-limits to this
   agent; use the process environment, which `resolve_config()` honours) to an
   interpreter whose `python -V` satisfies `3.8.10`, with `torch 2.1.0`, `torchvision 0.16.0`, CUDA
   12.6, and the 19 declared packages — or an `environment.adapted_runtime` block added to the
   registry so a modern stack is a declared, legal contract. **Not satisfiable by any interpreter
   present on this host.**
2. `WORKBENCH_CHECKPOINT_ROOT` must resolve `airknow/airknow_fiq_0.5.pt`. Two ways:
   set `WORKBENCH_CHECKPOINT_ROOT=…/checkpoint_selection/20261106T002617+0700/checkpoints`, or stage
   the two files under `<configured root>/airknow/`. **Do not edit `workbench.env`** (forbidden); use
   the process environment, which `resolve_config()` honours for allowlisted keys
   (`operator_config.py:132-135`).
3. A valid recorded GPU capability: `--verify-gpu` creates it (`evaluate_models.py:2760-2776`), and
   `require_launch_ready` is the single launch gate.
4. Label the result `checkpoint_training_noise_pct: 50` (or `80`), `evaluation_noise_pct: 0`. This is
   a `policy_overridden` selection; the aggregate report and any published artifact must say
   *"trained with 50% noise; evaluated on clean FashionIQ val"*.

The underlying argv the adapter will build (`models.py:167-212` → `_replay_command`,
`models.py:44-73`) is:

```bash
$WORKBENCH_AIRKNOW_PYTHON \
  /home/heheboiz/data/cir/workbench/replay/airknow.py \
  --checkpoint-root <checkpoints>/airknow \
  --dataset-root /home/heheboiz/data/cir/data/FashionIQ \
  --source-root /home/heheboiz/data/cir/workbench/third_party/AirKnow \
  --variant fiq_n05
```

### 4.3 `BLOCKED_*` rows — what the command *would* be

For completeness, the audited argv each blocked model would run (source: `ADAPTERS` table,
`workbench/backend/adapters/models.py:524`; every one of the 8 has a dedicated adapter class, none
falls through to `NotImplementedError`):

| model | launcher | argv |
|---|---|---|
| `pair` | replay | `$WORKBENCH_PAIR_PYTHON workbench/replay/pair.py --checkpoint-root <ckpt dir> --dataset-root <FQ> --source-root third_party/PAIR --checkpoint-id pair_b1` |
| `clvc_net` | replay | `$WORKBENCH_CLVC_PYTHON workbench/replay/clvc_net.py --checkpoint-root <ckpt dir> --dataset-root <FQ> --source-root third_party/CLVCNet` |
| `dcnet` | script | `$WORKBENCH_DCNET_PYTHON third_party/DCNet/test.py --resume <ckpt dir>` |
| `combiner_rn50x4_noft` | script | `$WORKBENCH_CLIP4CIR_PYTHON third_party/CLIP4Cir/src/validate.py --dataset fashionIQ --combining-function combiner --combiner-path <ckpt> --projection-dim 2560 --hidden-dim 5120 --clip-model-name RN50x4 --target-ratio 1.25 --transform targetpad` |
| `clip4cir_rn50x4_fullft` | script | same, plus `--clip-model-path <ckpt>/clip_state.pt`, with `--combiner-path <ckpt>/combiner_state.pt` |
| `tgcir` | replay | `$WORKBENCH_TGCIR_PYTHON workbench/replay/tgcir.py --checkpoint-root <ckpt dir> --dataset-root <FQ> --source-root third_party/TGCIR` |
| `sprc` | script | `$WORKBENCH_SPRC_PYTHON third_party/SPRC/src/blip_validate.py --dataset fashionIQ --blip-model-name blip2_cir_align_prompt --backbone <pretrain\|pretrain_vitL> --model-path <ckpt>` |

No `--transfer-*` flag applies to any native model; the only such flag in the codebase is
`ptha_mtst`'s `--transfer-cirr-to-fashioniq` (`models.py:515-521`), which is CIRCUS. SPRC's
condition is expressed as `--backbone`, derived from the **declared** variant and refused outright
when a variant declares none (`checkpoint_backbone_undeclared`).

`--top-k` and `--dataset-root` are effectively inert for `combiner`, `clip4cir`, `sprc`, `dcnet`
(their evaluators take no such option; they read `fashionIQ_dataset/images` relative to cwd).

### 4.4 Useful no-GPU invocations (safe to run now)

```bash
# list what the harness thinks is runnable — no plan built, no checkpoint loaded
workbench/envs/ctl312h/bin/python workbench/scripts/evaluate_models.py --list

# print the guarded command and stop (still refuses without a verified interpreter)
workbench/envs/ctl312h/bin/python workbench/scripts/evaluate_models.py \
  --model airknow --checkpoint fiq_n05 --dry-run
```

---

## 5. Web-ingestion path for native results

### 5.1 Why `GET /api/runs` is `[]` — and will stay `[]` for native work

`GET /api/runs` (`workbench/backend/main.py:218-223`) does **not** scan the filesystem. It is a
read of the DuckDB serving index at `<WORKBENCH_ARTIFACTS_ROOT>/workbench.duckdb`, opened read-only
(`storage.py:13-23`, `30-42`; `SELECT metadata, evidence_status FROM runs [WHERE protocol_id=?]
ORDER BY run_id`). That index is built by scanning `WORKBENCH_RESULTS_ROOT` recursively for `*.json`
(`index.py:55`), and is only built automatically **when the DB file is absent** (`main.py:79-80`);
otherwise it needs `POST /api/index/rebuild` (`main.py:1111-1121`) or
`workbench/scripts/rebuild_index.py`.

`workbench/artifacts/results/` contains only `.gitkeep`, so the index has zero `runs` rows.

**The native path writes nothing there.** `evaluate_models.py:433` constructs
`output_path = WORKBENCH_RESULTS_ROOT / <protocol> / <model_id> / <checkpoint_id>.json` and puts it
on the `EvalRequest` — but **no native adapter reads `request.output_path`**. The only writer of
canonical result artifacts is `workbench/replay/publication.py:138` (`publish`), and it is reached
only through a per-model observer (`*_export.py` → `*_publish.py`), which requires a
`per_query_observer` declaration. **No native model declares one**, and
`workbench/replay/{airknow,pair,clvc_net,dcnet,combiner_rn50x4_noft,clip4cir_rn50x4_fullft,tgcir,sprc}_export.py`
do not exist (verified: zero matches). So every native run is recorded as
`per_query_state: CAPABILITY_ABSENT` and produces **aggregate-only** evidence.

### 5.2 What a native run *does* write (aggregate path, works today)

Under `WORKBENCH_ARTIFACTS_ROOT` (`evaluate_models.py:1628-1658, 2277-2358`):

```
workbench/artifacts/reports/<model>_<checkpoint>_<run_id>_aggregate.json
workbench/artifacts/reports/<model>_<checkpoint>_<run_id>_completion.json
workbench/artifacts/reports/<model>_<checkpoint>_<run_id>_invocation_outcome.json
workbench/artifacts/reports/<model>_<checkpoint>_latest_{attempt,successful}.json   (pointer)
workbench/artifacts/logs/<model>__<checkpoint>__<run_id>/{command.json,stdout.log,stderr.log,
                                                          aggregate_report.json,
                                                          invocation_outcome.json}
workbench/artifacts/evidence_ledger.jsonl          (append-only HMAC chain, one entry per run)
```

`run_id` = `--run-id` or `f"{ts:%Y%m%dT%H%M%S%fZ}_{model}_{checkpoint}_{uuid4[:8]}"`
(`evaluate_models.py:551-553`).

These are surfaced by **`GET /api/benchmark-runs`** (`main.py:896`, reads
`reports/*_aggregate.json` requiring `artifact_type == "official_aggregate_evaluation_report"`) and
**`GET /api/runs/gate`** (`main.py:914`) — *not* by `/api/runs`.

### 5.3 What a native result MUST write to appear in `/api/runs`

To populate `/api/runs` a native run must produce a **schema-v2 canonical result artifact**
(`workbench/docs/RESULT_SCHEMA.md`; `workbench/backend/schemas/results.py:141-146`) at:

```
<WORKBENCH_RESULTS_ROOT>/<protocol_id>/<model_id>/<run_id>.json
```

(path used by `publication.py:163-164`; the loader tolerates any depth because it globs recursively,
but this is the canonical layout). Requirements from the parser:

1. `*.json`, basename not starting with `.`, and no `artifact_type` in
   `{official_aggregate_evaluation_report, limn_observation_payload, encoder_observation_payload}`
   (those are skipped, `index.py:37-41, 61-62, 71-72`).
2. Top-level `schema_version: 2`, `run: {…}`, `queries: [ … ]`.
3. `run` must satisfy `RunMetadata`: `run_id` (≥1 char, globally unique in the root),
   `dataset: "FashionIQ"`, `split: "val"`, a registry `protocol_id`,
   `literature_split_label` matching that protocol, `evaluation_noise_pct: 0`, `model_id`,
   `method_name`, `checkpoint_id`, `top_k_saved ≥ 1`, `gallery_size ≥ 1`, `timestamp`.
   `data_kind` defaults to `"experiment"`.
4. **If `data_kind == "experiment"`** (the honest value for real native runs):
   `run_id` ≥ 8 chars, not `dev-`-prefixed, **and both `command_digest` and `environment_digest`
   present** (`publication.py:215-229`) — otherwise the whole index rebuild is *refused*
   (`index.py:145-149`), not skipped.
5. Each `query` needs `query_id == "{category}:{annotation_index}:{reference_id}:{target_id}"`,
   `raw_captions` with **exactly 2** strings, `target_rank ≥ 1`, contiguous unique ranks from 1 in
   `top_results`, exactly `min(top_k_saved, gallery_bound)` rows, and
   `top_results[target_rank-1].image_id == target_id` whenever the target is within saved depth.
6. Then rebuild: `POST /api/index/rebuild` or
   `workbench/envs/ctl312h/bin/python workbench/scripts/rebuild_index.py --results-root <root> --database <duckdb> --validate-first`.

Minimum shape that *validates* (derived from parser code only — this is **not** an artifact I
produced, and `data_kind: "mock"` is used only because it skips the identity check):

```json
{"schema_version": 2,
 "run": {"run_id": "m", "dataset": "FashionIQ", "split": "val",
         "protocol_id": "fashioniq_original_split", "literature_split_label": "original",
         "evaluation_noise_pct": 0, "model_id": "m", "method_name": "m",
         "checkpoint_id": "c", "top_k_saved": 1, "gallery_size": 1,
         "timestamp": "2026-01-01T00:00:00Z", "data_kind": "mock"},
 "queries": []}
```

**Consequence for scheduling:** native models are aggregate-only by design. Getting them onto the
web *as per-query runs* requires either (a) adding `per_query_observer` declarations plus
`workbench/replay/<model>_export.py` + `_publish.py` implementations — new engineering in another
agent's ownership — or (b) accepting that native results appear only in `/api/benchmark-runs` and
`/api/runs/gate`, which the current frontend already consumes (`App.tsx:38,44`).

Also: no `.done`, `manifest.json` or `metrics.json` sidecar is consulted anywhere; a file appearing
under its final name is the completion signal, and publication is atomic no-clobber (`os.link`).

---

## 6. What I could not determine

| # | open question | evidence that would settle it |
|---|---|---|
| 1 | Whether `pair-B1.pt` / `pair-B2.pt` correspond to FashionIQ A/B splits at all | The author HF repo's model card or a load test that prints parameter names/last-layer shapes; or an upstream README line mapping B1/B2 to a dataset. Registry says the mapping is unresolved and I found no local evidence to the contrary. |
| 2 | Which of the six `clvc/{cat}_{local,global}_model.pt` files is the *published* evaluation artifact, and whether `checkpoint-unresolved.pt` was ever a real single file | The author Drive `CLVCNET.zip` member list, or a registry amendment naming the six files as a `bundle_directory` with `required_files`. The prior session's `run_clvc.py` proves the six files run, which proves nothing about provenance. |
| 3 | Whether the DCNet run directory (`config.json` + `trained_model.pth`, 2.88 GB) is the FashionIQ-trained checkpoint the paper's Table uses | The author Drive folder listing (`1wgygqF0…`) compared with the staged `val/info.log` epoch/step counters; or `expected_sha256` published upstream (none exists today). |
| 4 | Which Drive member is `rn50x4-noft` vs `rn50x4-fullft`, i.e. whether the staged `fiq_comb_*` / `fiq_clip_*` files are the intended pairing | The author Drive folder `1ny2hhzP…` file listing plus CLIP4Cir's README naming convention. Registry explicitly refuses to infer this. |
| 5 | Whether `downloads/tgcir/x/checkpoints/Fashioniq.pt` (607 928 637 B) is the TG-CIR FashionIQ checkpoint | The `TG-CIR.zip` member manifest; a load test printing the state-dict top-level keys against `third_party/TGCIR/model.py`. |
| 6 | Whether `sprc_fiq_vitl.pt` loads with the intended class keys (it is known to load **non-strictly**) | A `torch.load` + key-diff report against `src/blip_validate.py`'s expected registry class. Deliberately not run: loading a 1.46 GB checkpoint is disk+CPU work but the key mapping is the actual question and needs the model constructor, which I was told not to build. |
| 7 | Whether a *clean* (0%-noise) AirKnow weight exists anywhere | The author HF repo file listing for `iLearn-Lab/CVPR26-Air-Know`. The registry asserts only 50%/80% are published; I did not make a network request to confirm. |
| 8 | Whether BLIP-2 / ViT-B/16 cached bytes are the author-published files (`UNRESOLVED_PROVENANCE`) | Author-published SHA-256 for `blip2_pretrained.pth` and `ViT-B-16.pt`. Both files exist locally (hashed: `ViT-B-16.pt` = `5806e77c…`), but no upstream digest exists in the registry, so the asset checker cannot admit them. |
| 9 | Whether `resnet50-0676ba61.pth` is acceptable in place of the declared `resnet50-19c8e357.pth` | torchvision's own model-zoo index. `0676ba61` is the modern torchvision ResNet-50 entry; the registry still declares the legacy `19c8e357` hash prefix, and `WORKBENCH_{CLVC,DCNET}_RESNET50_WEIGHTS` did **not** change the checked path when I set it — the checker keeps reading the declared `cache_path_suffix`. This looks like a registry-metadata staleness bug, not a missing asset. |
| 10 | Whether any interpreter on this host can satisfy the *historical* native contracts (py3.7/3.8 + torch 1.x/2.0) | Building one would settle it. `dcnet-prepare` is py3.7.12 without torch; conda has only `VisRAG` (3.10.8) / `melt` (3.11.15) / five `teacher-*` (3.11.15). No py3.8 exists anywhere I could find. |
| 11 | Whether adding `adapted_runtime` blocks for the 8 native models is scientifically acceptable | A decision by the owner of `workbench/registry/**`. Nine CIRCUS models already carry an `sm_120` profile, which is precedent; without it, no modern interpreter is legal for native models and all 8 stay env-blocked. **This is the single highest-leverage unblocked decision.** |
| 12 | Whether the `fashioniq_standard` `.jpg` declaration vs `.png` `images/` tree is a registry bug or an intentional guard | Either the declaration is wrong (the pinned `data_utils.py` hard-codes `.jpg` for CLIP4Cir and `.png` for SPRC — both are satisfied by the `prepared/` views), or the `canonical_dataset_link` must point at `FASHIONIQ_ROOT` and the `.jpg` views must live there. The two preflight rules currently contradict each other on the same tree. |

---

## 7. Evidence index

Everything below was produced in this session; nothing was inferred from another agent's prose.

| artefact | content |
|---|---|
| `plan_blockers.txt` | `guarded_plan()` blocker list for all 11 native variants under the `checkpoints/` root |
| `blockers_configured_root.json` | `runtime_plan_blockers` + `preflight_blockers` per variant under the **configured** root |
| `checkpoint_sha256.tsv` | raw `size<TAB>sha256<TAB>path` for 17 native checkpoint files |
| `environment_probe.json` | `verify_environment` tier/status/reason per model × candidate interpreter |
| `source_provenance.json` | `source_provenance` per model (pin match, dirty state, undeclared files) |
| `checkpoint_selection_policy.json` | `select_checkpoints` under both policies |
| `layout_requirements.json` | `paths_for_layout` missing-path list per model |
| checkpoint digests | `sha256sum` over 17 native files, incl. all 11 declared variants' candidates (see §1.3) |
| environment probe | `verify_environment(model, probe=True)` per model × candidate interpreter |
| source provenance | `source_provenance()` → all 8 `verified: true`, pin matches `upstream_commit_sha` |
| patch state | `approved_patch_state()` → dcnet only, `applied: true` |
| layout probe | `paths_for_layout()` → 0 missing paths for all 8 |
| gallery coverage | val splits 3817/6346/5373, union 15415; prepared views 0 missing / 0 dangling |
| selection policy | `select_checkpoints()` under both `strict_clean` and `include_flagged_conditions` |
| asset status | `external_asset_statuses()` per model |
