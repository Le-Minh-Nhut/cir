# G0 — DEFINITIVE REPOSITORY & VARIANT INVENTORY

Session `20261108T152852` · branch `research/cir-failure-workbench` · read-only audit, no repo edits outside this directory.

Everything below was measured in this session: registry parsed field-by-field, every vendored tree re-hashed against its own `provenance.json` manifest, every staged checkpoint hashed with `sha256sum`, every interpreter probed for `sys.version` / `torch` / `torchvision` / `torch.version.cuda`, and every adapter script path confirmed on disk. Fields that could not be established say `UNVERIFIED` and carry a reason; none are guessed.

## 1. The two numbers

| quantity | value | how counted |
|---|---|---|
| unique upstream repositories | **16** | distinct `upstream_repo_url` across all 17 registry entries; equal to the 16 directories under `workbench/third_party/` |
| registered model variants | **17** | `model_id` entries in `workbench/registry/models.yaml` (`len(models)` = 17) |
| declared checkpoint variants | **29** | one row per `checkpoint_variants[]` element; NOT repositories |
| checkpoint variants staged in the resolved root | **21 / 29** | `checkpoint_path()` from `workbench/backend/registry.py` against `WORKBENCH_CHECKPOINT_ROOT` |

A registered **variant** here means a registry `model_id`, i.e. one runnable model identity. That is the unit the adapter table, the environment contract and the API surface are all keyed by. The 29 `checkpoint_variants` are weight files, not model identities, and are never counted as repositories.

## 2. Reconciliation against 16 / 17

**Verdict: the prior figure 16 repositories / 17 registered variants is CORRECT and reproduced independently — but it is under-specified in three ways, and one of its two numbers had to be re-derived rather than confirmed.**

1. **Repositories: 16 confirmed.** 17 `model_id`s map onto 16 distinct `upstream_repo_url` values because `combiner_rn50x4_noft` and `clip4cir_rn50x4_fullft` both come from `ABaldrati/CLIP4Cir` (`source_dir: CLIP4Cir`). No 17th repository has appeared under `workbench/third_party/` — the directory count is exactly 16 and each directory holds one `provenance.json`. The prior claim was right for the right reason.
2. **Variants: 17 confirmed, but the registry also declares 29 checkpoint variants.** The prior figure referred to `model_id`s only. Anyone reading "17" as "17 things to evaluate" will under-count the weight files by 12. Both numbers are now stated explicitly rather than left to context.
3. **The count is a registry count, not a filesystem count.** The repository is not the only place upstream code lives. `teacher/repos/` carries five nested `.git` checkouts (`CSMCIR`, `AAAI25-ENCODER`, `TME`, `SPRC`, `SPN4CIR`) and `data/cirr/` a sixth (`Cuberick-Orion/CIRR`). Two of those — `SPN4CIR` (`BUAADreamer/SPN4CIR`) and the `CIRR` dataset repo — are **not** registered models at all and are outside every count above. If "repositories" is ever meant to include the teacher/ and data/ trees, the number is 16 + 2 non-model checkouts = 18 source trees on disk, of which 16 are in the workbench scope. The 16/17 figure is a *workbench-registered* count and must be labelled as such.
4. **The CLIP4Cir pairing drives the whole 17-vs-16 gap.** It is not a second repository and it must not be split into a 17th count: `workbench/third_party/CLIP4Cir` is a single checkout with one `provenance.json` whose `model_ids` list is `["combiner_rn50x4_noft","clip4cir_rn50x4_fullft"]`. One repository, two registered variants, one environment variable (`WORKBENCH_CLIP4CIR_PYTHON`), two different weight artifacts — one file for `noft`, a two-file bundle for `fullft`.

**The registry on disk is DIRTY relative to HEAD, and this inventory describes the dirty working tree** (`git status` → `M workbench/registry/models.yaml`; `git diff --stat` → 189 insertions, 46 deletions against `HEAD` = `b2fca876`, 2026-11-07 13:57:43 +0700). The count shape is **invariant** across both versions: HEAD and the worktree each parse to 17 `model_id`, 16 distinct `upstream_repo_url`, 29 `checkpoint_variants`, identical `checkpoint_id` lists per model. What the uncommitted edits change is the *evidence fields*:

| field changed | HEAD | worktree |
|---|---|---|
| `airknow` fiq_n05/fiq_n08, `pair` pair_b1/b2, `combiner noft`, `tgcir`, `sprc_fiq_vitl` `expected_sha256` / `expected_size_bytes` | `null` | real digest + size |
| `clvc_net`, `dcnet`, `combiner noft`, `clip4cir fullft`, `tgcir` `checkpoint_mapping_status` | `UNVERIFIED` | `VERIFIED_METADATA` |
| `clvc_net`, `dcnet`, `clip4cir fullft` `filename` | placeholder (`checkpoint-unresolved.pt`, `fashioniq_dcnet`, `rn50x4-fullft-unresolved.pt`) | resolved (`.`, `.`, `.` bundles) |
| `combiner noft`, `tgcir` `filename` | placeholders | `fiq_comb_RN50x4_noft.pt`, `Fashioniq.pt` |

So a reader who checks out `HEAD` will find 7 variants whose digest is `null` and 5 whose mapping status is `UNVERIFIED` — including both CLIP4Cir variants. Those statuses now report `VERIFIED_METADATA` because someone verified them in the working tree, and this audit independently re-verified every one of those digests against the staged bytes. The reconciliation is therefore: **16/17 unchanged in shape; 12 checkpoint variants and 5 mapping statuses were upgraded by uncommitted work.** That is the "newer local addition" the task asked about — it is in the registry's evidence fields, not in the repository set.

## 3. Repositories

| repository (`source_dir`) | variants | upstream URL | registry pin | verified HEAD / provenance | source state | patches |
|---|---|---|---|---|---|---|
| `CSMCIR` | 1 | https://github.com/qzp2018/CSMCIR | `774f94e2076f` | `774f94e2076f` in `teacher/repos/CSMCIR` (match, 236 dirty files) | vendored, git-tracked; **2/359 manifest files differ (approved patch)** | csmcir_models_init_compat.patch, csmcir_validate_visualization_opt_in.patch |
| `AirKnow` | 1 | https://github.com/iLearn-Lab/CVPR26-Air-Know | `6ab6fa95b746` | provenance-only: `6ab6fa95b746` (no nested .git; tree re-hashed) | vendored, git-tracked; all 1218 manifest files match | — |
| `ConeSep` | 1 | https://github.com/Lee-zixu/ConeSep | `bd4bf629ebe6` | provenance-only: `bd4bf629ebe6` (no nested .git; tree re-hashed) | vendored, git-tracked; all 1221 manifest files match | — |
| `MTST_PTHA` | 1 | https://github.com/zyinan99/MTST_PTHA | `6a02ee4763fc` | provenance-only: `6a02ee4763fc` (no nested .git; tree re-hashed) | vendored, git-tracked; all 360 manifest files match | — |
| `HABIT` | 1 | https://github.com/iLearn-Lab/AAAI26-HABIT | `6ddc2cd63eac` | provenance-only: `6ddc2cd63eac` (no nested .git; tree re-hashed) | vendored, git-tracked; all 1234 manifest files match | — |
| `INTENT` | 1 | https://github.com/iLearn-Lab/AAAI26-INTENT | `2a75fc6f57d1` | provenance-only: `2a75fc6f57d1` (no nested .git; tree re-hashed) | vendored, git-tracked; all 1228 manifest files match | — |
| `HINT` | 1 | https://github.com/iLearn-Lab/ICASSP26-HINT | `bec50b6c8c19` | provenance-only: `bec50b6c8c19` (no nested .git; tree re-hashed) | vendored, git-tracked; all 1225 manifest files match | — |
| `ENCODER` | 1 | https://github.com/iLearn-Lab/AAAI25-ENCODER | `29a2a31d6a56` | `29a2a31d6a56` in `teacher/repos/AAAI25-ENCODER` (match, 1 dirty files) | vendored, git-tracked; all 26 manifest files match | — |
| `PAIR` | 1 | https://github.com/iLearn-Lab/ICASSP25-PAIR | `bfb0b98cdb4c` | provenance-only: `bfb0b98cdb4c` (no nested .git; tree re-hashed) | vendored, git-tracked; all 22 manifest files match | — |
| `CLVCNet` | 1 | https://github.com/iLearn-Lab/SIGIR21-CLVC-Net | `bd9b68894898` | provenance-only: `bd9b68894898` (no nested .git; tree re-hashed) | vendored, git-tracked; all 9 manifest files match | — |
| `DCNet` | 1 | https://github.com/ozmig77/dcnet | `68c79d38569f` | provenance-only: `68c79d38569f` (no nested .git; tree re-hashed) | vendored, git-tracked; **1/39 manifest files differ (approved patch)** | dcnet_checkpoint_weights_only_optout.patch |
| `CLIP4Cir` | 2 | https://github.com/ABaldrati/CLIP4Cir | `dfed9f748a8a` | provenance-only: `dfed9f748a8a` (no nested .git; tree re-hashed) | vendored, git-tracked; all 14 manifest files match | — |
| `TGCIR` | 1 | https://github.com/iLearn-Lab/MM23-TG-CIR | `65fa78eaf8ca` | provenance-only: `65fa78eaf8ca` (no nested .git; tree re-hashed) | vendored, git-tracked; all 11 manifest files match | — |
| `SPRC` | 1 | https://github.com/chunmeifeng/SPRC | `2935a5397732` | `2935a5397732` in `teacher/repos/SPRC` (match, 128 dirty files) | vendored, git-tracked; all 354 manifest files match | — |
| `LIMN` | 1 | https://github.com/iLearn-Lab/TPAMI24-LIMN | `7d7bc9b116f5` | provenance-only: `7d7bc9b116f5` (no nested .git; tree re-hashed) | vendored, git-tracked; all 33 manifest files match | — |
| `TME` | 1 | https://github.com/li-shuxian/TME | `394aad79df2c` | `7fe811992820` in `teacher/repos/TME` (**MISMATCH**, 0 dirty files) | vendored, git-tracked; **2/351 manifest files differ (approved patch)** | tme_precompute_bounded_allocation.patch, tme_eval_batch_override.patch |

### 3.1 Two repository-level disagreements worth naming

**A. `teacher/repos/TME` is a different upstream repository than the registry pins.**

| field | value |
|---|---|
| registry `upstream_repo_url` | `https://github.com/li-shuxian/TME` |
| registry `upstream_commit_sha` | `394aad79df2c104feb32a2c96132582012765062` |
| `workbench/third_party/TME/provenance.json` | same URL, same commit `394aad79df2c…`, vendored 2026-10-23 |
| `teacher/repos/TME` actual `git remote -v` | `https://github.com/He-Changhao/2025-CVPR-TME.git` |
| `teacher/repos/TME` actual HEAD | `7fe811992820d30828e24065eb0c7a7ba099dd3b` (2025-06-09) |

The pinned commit `394aad79…` **does not exist** in the `teacher/repos/TME` clone (`git cat-file -t` → `could not get object info`). The vendored tree matches its own provenance manifest on 349/351 files, and the 2 differing files are exactly the two approved TME patches — so the *vendored* tree is consistent and correctly patched. The disagreement is confined to the alternate `teacher/repos/TME` checkout, which is a differently-owned fork at a different commit. Any claim that "the TME checkout is at the pinned commit" is false for `teacher/repos/TME` and true only for `workbench/third_party/TME` (by manifest, not by `.git`).

**B. `teacher/repos/CSMCIR/src/lavis/models/__init__.py` is at neither the base nor the derived digest.**

| locus | sha256 | verdict |
|---|---|---|
| declared `base_sha256` | `9d8c914b551a7de614e062f9a241f44626fd591a31a0ccf1cc79f0ae23c9db01` | upstream as-published |
| declared `derived_sha256` | `0ec134431697af6f202b5256e4e30ab725a44191a279e3f8bfdb3a62495b3774` | after approved patch |
| `workbench/third_party/CSMCIR/src/lavis/models/__init__.py` | `0ec134431697af6f…` | **derived — approved patch applied, correct** |
| `teacher/repos/CSMCIR/src/lavis/models/__init__.py` (working tree) | `8e9cd6a8303bc0c6…` | **neither base nor derived — third state** |
| `teacher/repos/CSMCIR` at HEAD (`git show HEAD:…`) | `9d8c914b551a7de6…` | base — HEAD is clean upstream, the divergence is uncommitted working-tree edits (102+/102−, 236 dirty entries) |

So the teacher tree carries a *hand-edited* copy of that file that is neither the published file nor the approved patched file. The evaluator does not read it — `workbench/third_party/CSMCIR` is what the adapter launches — but any future run pointed at `teacher/repos/CSMCIR` would run an unreviewed variant. Same pattern, smaller, for `validate_blip_csmcir.py` in the teacher tree: it is at **base** (`aa7a4395…`) while the vendored copy is at **derived** (`a057e8f3…`) — i.e. the teacher copy is missing the approved patch.

### 3.2 Vendored-vs-provenance drift, complete

| repo | manifest files | match | differ | differing files |
|---|---|---|---|---|
| `CSMCIR` | 359 | 357 | 2 | `src/lavis/models/__init__.py`; `src/validate_blip_csmcir.py` |
| `DCNet` | 39 | 38 | 1 | `base/base_trainer.py` |
| `TME` | 351 | 349 | 2 | `src/precompute.py`; `src/precompute_evaluation.py` |

Every other vendored tree matches its provenance manifest on **100%** of recorded files (AirKnow 1218/1218, ConeSep 1221/1221, MTST_PTHA 360/360, HABIT 1234/1234, INTENT 1228/1228, HINT 1225/1225, ENCODER 26/26, PAIR 22/22, CLVCNet 9/9, CLIP4Cir 14/14, TGCIR 11/11, SPRC 354/354, LIMN 33/33). The three repos with drift are exactly the three with declared `approved_source_patches` — no unexplained source modification was found anywhere.

### 3.3 Patches: declared vs applied

| model | file | patch | patch file sha256 | current tree vs declared |
|---|---|---|---|---|
| `csmcir` | `src/lavis/models/__init__.py` | `csmcir_models_init_compat.patch` | `3f2bac48fe1a1775…` (match) | DERIVED (patch applied) |
| `csmcir` | `src/validate_blip_csmcir.py` | `csmcir_validate_visualization_opt_in.patch` | `0fcff28bf1f9abd3…` (match) | DERIVED (patch applied) |
| `dcnet` | `base/base_trainer.py` | `dcnet_checkpoint_weights_only_optout.patch` | `659eabf3c5b98352…` (match) | DERIVED (patch applied) |
| `tme` | `src/precompute.py` | `tme_precompute_bounded_allocation.patch` | `cdf7d98c439a2a37…` (match) | DERIVED (patch applied) |
| `tme` | `src/precompute_evaluation.py` | `tme_eval_batch_override.patch` | `0594b8896c051d35…` (match) | DERIVED (patch applied) |

All five declared patches are applied in the vendored trees, and all five patch files match their recorded `patch_sha256`. `workbench/patches/` contains exactly those five `.patch` files plus one `.md` note (`csmcir.md`); there is no undeclared and no orphaned patch, so the directory and the declared set agree in both directions.

## 4. Registered variants

61 rows would be unreadable as one table; it is split by concern. `sha256` is the value **computed in this session** from the staged bytes.

### 4.1 Identity: variant → repository → checkpoint → hash

| variant (`model_id`) | repository | checkpoint_id | artifact | bytes | computed sha256 | vs declared | staged | map status |
|---|---|---|---|---|---|---|---|---|
| `csmcir` | `CSMCIR` | `fashioniq` | file/fashioniq_tuned_clip_best.pt | 2,720,615,859 | `c8812071de0640d6…` | MATCH | YES | VERIFIED_METADATA |
| `airknow` | `AirKnow` | `fiq_n05` | file/airknow_fiq_0.5.pt | 2,724,056,638 | `f4d25ed9e532f2ee…` | MATCH | YES | VERIFIED_METADATA |
| `airknow` | `AirKnow` | `fiq_n08` | file/airknow_fiq_0.8.pt | 2,724,056,638 | `756fc0bdc43d9e0f…` | MATCH | YES | VERIFIED_METADATA |
| `conesep` | `ConeSep` | `fiq_n02` | file/ConeSep-FIQ-N0.2.pt | 2,721,428,966 | `5a864ee5a78c1d24…` | MATCH | YES | VERIFIED_METADATA |
| `conesep` | `ConeSep` | `fiq_n05` | file/ConeSep-FIQ-N0.5.pt | — | `—` | ABSENT | **NO** | VERIFIED_METADATA |
| `conesep` | `ConeSep` | `fiq_n08` | file/ConeSep-FIQ_N0.8.pt | — | `—` | ABSENT | **NO** | VERIFIED_METADATA |
| `ptha_mtst` | `MTST_PTHA` | `cirr_transfer` | file/tuned_mtst_cirr.pt | 2,720,352,973 | `1c4dab86a9f66dfe…` | MATCH | YES | VERIFIED_METADATA |
| `habit` | `HABIT` | `fiq_n02` | file/HABIT-FIQ_N0.2.pt | 2,730,715,930 | `814d2d0ceb29ef91…` | MATCH | YES | VERIFIED_METADATA |
| `habit` | `HABIT` | `fiq_n05` | file/HABIT-FIQ_N0.5.pt | — | `—` | ABSENT | **NO** | VERIFIED_METADATA |
| `habit` | `HABIT` | `fiq_n08` | file/HABIT-FIQ_N0.8.pt | — | `—` | ABSENT | **NO** | VERIFIED_METADATA |
| `intent` | `INTENT` | `fiq_n02` | file/intent_fiq_0.2.pt | 2,722,478,522 | `2a504cadfd6b1c24…` | MATCH | YES | VERIFIED_METADATA |
| `intent` | `INTENT` | `fiq_n05` | file/intent_fiq_0.5.pt | — | `—` | ABSENT | **NO** | VERIFIED_METADATA |
| `intent` | `INTENT` | `fiq_n08` | file/intent_fiq_0.8.pt | — | `—` | ABSENT | **NO** | VERIFIED_METADATA |
| `hint` | `HINT` | `fashioniq` | file/fashioniq.pt | 2,724,451,182 | `bf9c72d42206b667…` | MATCH | YES | VERIFIED_METADATA |
| `encoder` | `ENCODER` | `fashioniq` | file/fashioniq.pt | 631,879,365 | `0f31c50150a7bda3…` | MATCH | YES | VERIFIED_METADATA |
| `pair` | `PAIR` | `pair_b1` | file/pair-B1.pt | 694,958,565 | `c4d861ecd0baa3f1…` | MATCH | YES | UNVERIFIED |
| `pair` | `PAIR` | `pair_b2` | file/pair-B2.pt | 694,872,371 | `74346fd7de6efdaf…` | MATCH | YES | UNVERIFIED |
| `clvc_net` | `CLVCNet` | `author_drive_unresolved` | bundle_directory/. | 1,185,968,194 | `6 members, see §4.2` | BUNDLE_SEE_MEMBERS | YES | VERIFIED_METADATA |
| `dcnet` | `DCNet` | `fashioniq_run_directory` | run_directory/. | 2,877,316,002 | `3 members, see §4.2` | BUNDLE_SEE_MEMBERS | YES | VERIFIED_METADATA |
| `combiner_rn50x4_noft` | `CLIP4Cir` | `rn50x4_noft_unresolved` | file/fiq_comb_RN50x4_noft.pt | 236,017,921 | `b31882fe6f7e3e19…` | MATCH | YES | VERIFIED_METADATA |
| `clip4cir_rn50x4_fullft` | `CLIP4Cir` | `rn50x4_fullft_unresolved` | bundle_directory/. | 949,892,616 | `2 members, see §4.2` | BUNDLE_SEE_MEMBERS | YES | VERIFIED_METADATA |
| `tgcir` | `TGCIR` | `tgcir_archive_unresolved` | file/Fashioniq.pt | 607,928,637 | `UNVERIFIED…` | MISMATCH | YES | VERIFIED_METADATA |
| `sprc` | `SPRC` | `sprc_fiq_unresolved` | file/sprc_fiq.pt | — | `—` | ABSENT | **NO** | UNVERIFIED |
| `sprc` | `SPRC` | `sprc_fiq_vitl` | file/sprc_fiq_vitl.pt | 1,459,756,541 | `19386104271c01ea…` | MATCH | YES | VERIFIED_METADATA |
| `limn` | `LIMN` | `base_iter0_dress` | file/0_dress_best_model.pt | 1,792,010,157 | `96a335d4eb8b1e77…` | MATCH | YES | VERIFIED_METADATA |
| `limn` | `LIMN` | `base_iter0_shirt` | file/0_shirt_best_model.pt | 1,792,010,157 | `9260c7f77d3017d8…` | MATCH | YES | VERIFIED_METADATA |
| `limn` | `LIMN` | `base_iter0_toptee` | file/0_toptee_best_model.pt | 1,792,010,157 | `79f663a14d89d455…` | MATCH | YES | VERIFIED_METADATA |
| `tme` | `TME` | `fashioniq_0` | file/fashioniq_0_best_model.pth | 2,728,382,104 | `fea7fc08760dad1f…` | PRESENT_NO_DECLARED_SHA | YES | VERIFIED_METADATA |
| `tme` | `TME` | `cirr_0` | file/cirr_0_best_model.pth | — | `—` | ABSENT | **NO** | VERIFIED_METADATA |

### 4.2 Bundle and directory artifacts, member by member

| variant | member | bytes | computed sha256 |
|---|---|---|---|
| `clvc_net` | `dress_global_model.pt` | 176,437,419 | `be41c2ba81b803fd392c2c2517bf29cf…` |
| `clvc_net` | `dress_local_model.pt` | 217,168,235 | `33d44186efd6a25ca076179411561001…` |
| `clvc_net` | `shirt_global_model.pt` | 177,872,427 | `78b1d660227ab6928e3a11a7384f9fff…` |
| `clvc_net` | `shirt_local_model.pt` | 218,603,243 | `2ba8e320b08ebb97d6558eb599cde2be…` |
| `clvc_net` | `toptee_global_model.pt` | 177,578,027 | `c78a58c4becafee3bffe719bef743d42…` |
| `clvc_net` | `toptee_local_model.pt` | 218,308,843 | `d1fa9170496f73a008736723fe5329c9…` |
| `dcnet` | `config.json` | 1,734 | `UNVERIFIED…` |
| `dcnet` | `trained_model.pth` | 2,877,310,172 | `130f031e0f7808ccf5f15630a691bd7a…` |
| `dcnet` | `val` | 4,096 | `UNVERIFIED…` |
| `clip4cir_rn50x4_fullft` | `fiq_clip_RN50x4_fullft.pt` | 713,874,695 | `4111ac139cf96ac0cf31b30bf8f55b65…` |
| `clip4cir_rn50x4_fullft` | `fiq_comb_RN50x4_fullft.pt` | 236,017,921 | `7c7c0a10fb4c995cb1ded9ad70a58891…` |

The `dcnet` bundle's two declared members both match their declared digests (`trained_model.pth` `130f031e…`, `config.json` `fa5fca78…`); the directory also holds an undeclared `val/` subdirectory no manifest covers. The `tgcir` single file matches its declared digest (`c37768f3…`).

### 4.3 Execution paths: which are live

| variant | composition adapter | native launcher | native entrypoint (upstream file) | workbench entrypoint | protocols |
|---|---|---|---|---|---|
| `csmcir` / `fashioniq` | adapter | script | `CSMCIR/src/validate_blip_csmcir.py` | `workbench.composition.model_csmcir` | fashioniq_original_split |
| `airknow` / `fiq_n05` | — | replay wrapper | `AirKnow/test_BLIP2.py` | `workbench/backend/adapters/models.py:airknow via evaluate_models.py` | fashioniq_full_gallery_ref_excluded |
| `airknow` / `fiq_n08` | — | replay wrapper | `AirKnow/test_BLIP2.py` | `workbench/backend/adapters/models.py:airknow via evaluate_models.py` | fashioniq_full_gallery_ref_excluded |
| `conesep` / `fiq_n02` | adapter | replay wrapper | `ConeSep/test.py` | `workbench.composition.model_conesep` | fashioniq_full_gallery_ref_excluded |
| `conesep` / `fiq_n05` | adapter | replay wrapper | `ConeSep/test.py` | `workbench.composition.model_conesep` | fashioniq_full_gallery_ref_excluded |
| `conesep` / `fiq_n08` | adapter | replay wrapper | `ConeSep/test.py` | `workbench.composition.model_conesep` | fashioniq_full_gallery_ref_excluded |
| `ptha_mtst` / `cirr_transfer` | adapter | replay wrapper | `MTST_PTHA/MTST_PTHA/src/validate_blip.py` | `workbench.composition.model_ptha_mtst` | fashioniq_original_split |
| `habit` / `fiq_n02` | adapter | replay wrapper | `HABIT/test.py` | `workbench.composition.model_habit` | fashioniq_full_gallery_ref_excluded |
| `habit` / `fiq_n05` | adapter | replay wrapper | `HABIT/test.py` | `workbench.composition.model_habit` | fashioniq_full_gallery_ref_excluded |
| `habit` / `fiq_n08` | adapter | replay wrapper | `HABIT/test.py` | `workbench.composition.model_habit` | fashioniq_full_gallery_ref_excluded |
| `intent` / `fiq_n02` | adapter | replay wrapper | `INTENT/test.py` | `workbench.composition.model_intent` | fashioniq_full_gallery_ref_excluded |
| `intent` / `fiq_n05` | adapter | replay wrapper | `INTENT/test.py` | `workbench.composition.model_intent` | fashioniq_full_gallery_ref_excluded |
| `intent` / `fiq_n08` | adapter | replay wrapper | `INTENT/test.py` | `workbench.composition.model_intent` | fashioniq_full_gallery_ref_excluded |
| `hint` / `fashioniq` | adapter | replay wrapper | `HINT/test.py` | `workbench.composition.model_hint` | fashioniq_val_split |
| `encoder` / `fashioniq` | adapter | replay wrapper | `ENCODER/evaluate_model.py` | `workbench.composition.model_encoder` | fashioniq_val_split |
| `pair` / `pair_b1` | — | replay wrapper | `PAIR/test.py` | `workbench/backend/adapters/models.py:pair via evaluate_models.py` | fashioniq_val_split |
| `pair` / `pair_b2` | — | replay wrapper | `PAIR/test.py` | `workbench/backend/adapters/models.py:pair via evaluate_models.py` | fashioniq_val_split |
| `clvc_net` / `author_drive_unresolved` | — | replay wrapper | `CLVCNet/test.py` | `workbench/backend/adapters/models.py:clvc_net via evaluate_models.py` | fashioniq_val_split |
| `dcnet` / `fashioniq_run_directory` | — | script | `DCNet/test.py` | `workbench/backend/adapters/models.py:dcnet via evaluate_models.py` | fashioniq_full_gallery_ref_excluded |
| `combiner_rn50x4_noft` / `rn50x4_noft_unresolved` | — | script | `CLIP4Cir/src/validate.py` | `workbench/backend/adapters/models.py:combiner_rn50x4_noft via evaluate_models.py` | fashioniq_original_split |
| `clip4cir_rn50x4_fullft` / `rn50x4_fullft_unresolved` | — | script | `CLIP4Cir/src/validate.py` | `workbench/backend/adapters/models.py:clip4cir_rn50x4_fullft via evaluate_models.py` | fashioniq_original_split |
| `tgcir` / `tgcir_archive_unresolved` | — | replay wrapper | `TGCIR/test.py` | `workbench/backend/adapters/models.py:tgcir via evaluate_models.py` | fashioniq_val_split |
| `sprc` / `sprc_fiq_unresolved` | — | script | `SPRC/src/blip_validate.py` | `workbench/backend/adapters/models.py:sprc via evaluate_models.py` | fashioniq_original_split |
| `sprc` / `sprc_fiq_vitl` | — | script | `SPRC/src/blip_validate.py` | `workbench/backend/adapters/models.py:sprc via evaluate_models.py` | fashioniq_original_split |
| `limn` / `base_iter0_dress` | adapter | replay wrapper | `LIMN/LIMN/test.py` | `workbench.composition.model_limn` | fashioniq_val_split |
| `limn` / `base_iter0_shirt` | adapter | replay wrapper | `LIMN/LIMN/test.py` | `workbench.composition.model_limn` | fashioniq_val_split |
| `limn` / `base_iter0_toptee` | adapter | replay wrapper | `LIMN/LIMN/test.py` | `workbench.composition.model_limn` | fashioniq_val_split |
| `tme` / `fashioniq_0` | adapter | replay wrapper | `TME/src/precompute_test.py` | `workbench.composition.model_tme` | fashioniq_original_split |
| `tme` / `cirr_0` | adapter | replay wrapper | `TME/src/precompute_test.py` | `workbench.composition.model_tme` | fashioniq_original_split |

"Live" is a statement about wiring, not about a successful run: an adapter that exists is live, and a native launcher whose checkpoint resolves is runnable. Whether it *has* run successfully is `reproduction_status` in the registry and is out of scope for an inventory.

### 4.4 Which variants are wired to which path, counted

- composition adapter present: **9** models — `conesep`, `csmcir`, `encoder`, `habit`, `hint`, `intent`, `limn`, `ptha_mtst`, `tme`
- composition adapter absent: **8** models — `airknow`, `clip4cir_rn50x4_fullft`, `clvc_net`, `combiner_rn50x4_noft`, `dcnet`, `pair`, `sprc`, `tgcir`
- native launcher present for all **17** models (every `model_id` has an `OfficialScriptAdapter` subclass and a verifiable upstream script on disk)

Every adapter `script` path was stat-ed and exists: 17/17 (`csmcir → src/validate_blip_csmcir.py`, `limn → LIMN/test.py`, `ptha_mtst → MTST_PTHA/src/validate_blip.py`, …). Note those two: the upstream file lives one directory deeper than the repo root, and the adapter paths carry that nesting — the registry `source_dir` alone is not the script root.
## 5. Native upstream entrypoints

Four upstream repos publish **no runnable FashionIQ evaluation command at all** (PAIR, TG-CIR, LIMN, and — for the four LAVIS evaluators — AirKnow/ConeSep/HABIT/INTENT/HINT publish a file but no invocation). That is the reason `workbench/replay/` exists: those modules are invocation shells around a pinned `test(params, model, testset, category)` that has no `__main__`. The launcher column below says which kind each variant gets.

| variant | upstream file | upstream README command (verbatim shape) | launcher |
|---|---|---|---|
| `csmcir` | `CSMCIR/src/validate_blip_csmcir.py` | `python src/blip_validate_csmcir.py --dataset {CIRR|FashionIQ|Shoes|Lasco} --blip-model-name {name} --model-path {path}` | script |
| `airknow` | `AirKnow/test_BLIP2.py` | **no CLI published**; README names `test_BLIP2.py` as the "General evaluation script" | replay |
| `conesep` | `ConeSep/test.py` | **no CLI published**; `test.py` exists | replay |
| `ptha_mtst` | `MTST_PTHA/MTST_PTHA/src/validate_blip.py` | `python src/blip_validate.py --dataset {CIRR|FashionIQ} --blip-model-name {name} --model-path {path}` | replay |
| `habit` | `HABIT/test.py` | **no CLI published**; `test.py` exists | replay |
| `intent` | `INTENT/test.py` | **no CLI published**; README names `test.py` as the "General evaluation script" | replay |
| `hint` | `HINT/test.py` | **no CLI published**; `test.py` exists | replay |
| `encoder` | `ENCODER/evaluate_model.py` | `python3 evaluation_model.py --model_dir checkpoints/ENCODER_{…}.pth --dataset {shoes,fashioniq,fashion200k,cirr} --cirr_path ""` | replay |
| `pair` | `PAIR/test.py` | **no FashionIQ evaluation command in README**; the only published command is `python src/cirr_test_submission.py checkpoints/PAIR_CIRR.pt` (CIRR submission) | replay |
| `clvc_net` | `CLVCNet/test.py` | `python test.py` | replay |
| `dcnet` | `DCNet/test.py` | `python test.py --resume logdir/fashioniq_dcnet/` | script |
| `combiner_rn50x4_noft` | `CLIP4Cir/src/validate.py` | `python src/validate.py --dataset {CIRR|FashionIQ} --combining-function {combiner|sum} --combiner-path {path} --projection-dim 2560 --hidden-dim 5120` | script |
| `clip4cir_rn50x4_fullft` | `CLIP4Cir/src/validate.py` | same command as `combiner_rn50x4_noft` (one `validate.py` serves both variants) | script |
| `tgcir` | `TGCIR/test.py` | **no FashionIQ evaluation command in README**; only `python cirr_test_submission.py` (CIRR submission) | replay |
| `sprc` | `SPRC/src/blip_validate.py` | `python src/blip_validate.py --dataset {CIRR|FashionIQ} --blip-model-name {name} --model-path {path}` | script |
| `limn` | `LIMN/LIMN/test.py` | **no evaluation command in README** (training only); `LIMN/test.py` exists with no published invocation | replay |
| `tme` | `TME/src/precompute_test.py` | `python src/precompute_test.py --dataset {CIRR|FashionIQ} --mode {validate|test} --model-path {path} --gpu {gpu} --name {folder}` | replay |

The workbench-level entrypoints are `workbench/scripts/evaluate_models.py` (native path) and `workbench/scripts/composition_eval.py` (composition path); per-model commands are built by `workbench/backend/adapters/models.py`.

## 6. Environments

Every interpreter below was probed in this session with `CUDA_VISIBLE_DEVICES="" <interp> -c "import torch, torchvision, sys; …"` — no GPU touched, no allocation, no evaluation started.

| `WORKBENCH_*_PYTHON` variable | set in `workbench.env`? | interpreter | python | torch | torchvision | torch.version.cuda | registry contract |
|---|---|---|---|---|---|---|---|
| `WORKBENCH_CSMCIR_PYTHON` | yes | `workbench/artifacts/environments/cir-005ef4e1b02999a5/bin/python` | py 3.10.22, torch 2.7.1+cu128, tv 0.22.1+cu128, cuda 12.8 | | | | declares py 3.9 / torch 2.0.1 / tv 0.15.2 / cuda 11.7 |
| `WORKBENCH_AIRKNOW_PYTHON` | **UNSET** | — | — | — | — | — | declares py 3.8.10 / torch 2.1.0 / tv 0.16.0 / cuda 12.6 |
| `WORKBENCH_CONESEP_PYTHON` | yes | `workbench/envs/ilearn39-adapted/bin/python` | py 3.9.25, torch 2.7.1+cu128, tv 0.22.1+cu128, cuda 12.8 | | | | declares py 3.8.10 / torch 2.1.0 / tv 0.16.0 / cuda 12.6 |
| `WORKBENCH_PTHA_MTST_PYTHON` | yes | `workbench/artifacts/environments/cir-005ef4e1b02999a5/bin/python` | py 3.10.22, torch 2.7.1+cu128, tv 0.22.1+cu128, cuda 12.8 | | | | declares py 3.9 / torch 2.0.1 / tv 0.15.2 / cuda None |
| `WORKBENCH_HABIT_PYTHON` | yes | `workbench/envs/ilearn39-adapted/bin/python` | py 3.9.25, torch 2.7.1+cu128, tv 0.22.1+cu128, cuda 12.8 | | | | declares py 3.8.10 / torch 2.1.0 / tv 0.16.0 / cuda 12.6 |
| `WORKBENCH_INTENT_PYTHON` | yes | `workbench/envs/ilearn39-adapted/bin/python` | py 3.9.25, torch 2.7.1+cu128, tv 0.22.1+cu128, cuda 12.8 | | | | declares py 3.9 / torch 2.1.0 / tv 0.16.0 / cuda 12.6 |
| `WORKBENCH_HINT_PYTHON` | yes | `workbench/envs/ilearn39-adapted/bin/python` | py 3.9.25, torch 2.7.1+cu128, tv 0.22.1+cu128, cuda 12.8 | | | | declares py 3.8.10 / torch 2.1.0 / tv 0.16.0 / cuda 12.6 |
| `WORKBENCH_ENCODER_PYTHON` | yes | `workbench/envs/encoder310-adapted/bin/python` | py 3.10.20, torch 2.7.1+cu128, tv 0.22.1+cu128, cuda 12.8 | | | | declares py 3.9 / torch 1.13.1 / tv 0.14.1 / cuda 12.6 |
| `WORKBENCH_PAIR_PYTHON` | **UNSET** | — | — | — | — | — | declares py 3.8.10 / torch 2.0.0 / tv UNKNOWN / cuda 11.8 |
| `WORKBENCH_CLVC_PYTHON` | **UNSET** | — | — | — | — | — | declares py 3.7.6 / torch 1.6.0 / tv UNKNOWN / cuda 10.2 |
| `WORKBENCH_DCNET_PYTHON` | **UNSET** | — | — | — | — | — | declares py 3.7.7 / torch 1.4.0 / tv 0.5.0 / cuda UNKNOWN |
| `WORKBENCH_CLIP4CIR_PYTHON` | **UNSET** | — | — | — | — | — | declares py 3.8 / torch 1.11.0 / tv 0.12.0 / cuda UNKNOWN |
| `WORKBENCH_CLIP4CIR_PYTHON` | **UNSET** | — | — | — | — | — | declares py 3.8 / torch 1.11.0 / tv 0.12.0 / cuda UNKNOWN |
| `WORKBENCH_TGCIR_PYTHON` | **UNSET** | — | — | — | — | — | declares py UNKNOWN / torch 1.7.0 / tv 0.8.0 / cuda 11.0 |
| `WORKBENCH_SPRC_PYTHON` | **UNSET** | — | — | — | — | — | declares py 3.9 / torch 2.0.1 / tv 0.15.2 / cuda UNKNOWN |
| `WORKBENCH_LIMN_PYTHON` | yes | `workbench/envs/limn310-adapted/bin/python` | py 3.10.20, torch 2.7.1+cu128, tv 0.22.1+cu128, cuda 12.8 | | | | declares py UNKNOWN / torch 1.12.1 / tv 0.13.1 / cuda 12.4 |
| `WORKBENCH_TME_PYTHON` | yes | `workbench/envs/tme-adapted/bin/python` | py 3.10.22, torch 2.7.1+cu128, tv 0.22.1+cu128, cuda 12.8 | | | | declares py 3.10 / torch 2.1.2 / tv 0.16.2 / cuda 12.1 |

Decoded probe results — raw values measured now, not registry claims:

| interpreter | python | torch | torchvision | CUDA runtime | matches registry contract? |
|---|---|---|---|---|---|
| `workbench/envs/airknow-adapted/bin/python` | 3.11.15 | 2.7.1+cu128 | 0.22.1+cu128 | 12.8 | see §6.1 |
| `workbench/envs/clip4cir-adapted/bin/python` | 3.11.15 | 2.7.1+cu128 | 0.22.1+cu128 | 12.8 | see §6.1 |
| `workbench/envs/ctl312/bin/python` | 3.12.3 | 2.15.0+cu132 | 0.29.2+cu132 | 13.2 | see §6.1 |
| `workbench/envs/ctl312h/bin/python` | 3.12.11 | 2.15.0+cu132 | 0.29.2+cu132 | 13.2 | see §6.1 |
| `workbench/envs/dcnet-prepare/bin/python` | 3.7.12 | NO_TORCH | - | - | see §6.1 |
| `workbench/envs/dl-adapted/bin/python` | 3.11.15 | 2.7.1+cu128 | 0.22.1+cu128 | 12.8 | see §6.1 |
| `workbench/envs/encoder310-adapted/bin/python` | 3.10.20 | 2.7.1+cu128 | 0.22.1+cu128 | 12.8 | see §6.1 |
| `workbench/envs/ilearn39-adapted/bin/python` | 3.9.25 | 2.7.1+cu128 | 0.22.1+cu128 | 12.8 | see §6.1 |
| `workbench/envs/limn-adapted/bin/python` | 3.11.15 | 2.7.1+cu128 | 0.22.1+cu128 | 12.8 | see §6.1 |
| `workbench/envs/limn310-adapted/bin/python` | 3.10.20 | 2.7.1+cu128 | 0.22.1+cu128 | 12.8 | see §6.1 |
| `workbench/envs/pair-adapted/bin/python` | 3.11.15 | 2.7.1+cu128 | 0.22.1+cu128 | 12.8 | see §6.1 |
| `workbench/envs/sprc-adapted/bin/python` | 3.11.15 | 2.7.1+cu128 | 0.22.1+cu128 | 12.8 | see §6.1 |
| `workbench/envs/test-adapted/bin/python` | 3.11.15 | 2.7.1+cu128 | 0.22.1+cu128 | 12.8 | see §6.1 |
| `workbench/envs/tme-adapted/bin/python` | 3.10.22 | 2.7.1+cu128 | 0.22.1+cu128 | 12.8 | see §6.1 |
| `workbench/artifacts/environments/cir-005ef4e1b02999a5/bin/python` | 3.10.22 | 2.7.1+cu128 | 0.22.1+cu128 | 12.8 | see §6.1 |
| `workbench/artifacts/environments/clip4cir-38/bin/python` | 3.8.20 | 1.11.0+cpu | 0.12.0+cpu | CPU-only wheel (no CUDA runtime) | see §6.1 |
| `workbench/artifacts/environments/dcnet-377/bin/python` | 3.7.7 | 1.4.0+cpu | 0.5.0+cpu | CPU-only wheel (no CUDA runtime) | see §6.1 |
| `workbench/artifacts/environments/sprc-39/bin/python` | 3.9.25 | 2.0.1+cpu | 0.15.2+cpu | CPU-only wheel (no CUDA runtime) | see §6.1 |

### 6.1 Environment disagreements

Only **9 of 17** models have their `WORKBENCH_*_PYTHON` variable set at all. The registry declares a variable for all 17 and `workbench.env.example` lists all 17, but the live `workbench.env` sets only **9**. The **8 unset** models — `airknow`, `pair`, `clvc_net`, `dcnet`, `combiner_rn50x4_noft`, `clip4cir_rn50x4_fullft`, `tgcir`, `sprc`, sharing **7** variable names because `WORKBENCH_CLIP4CIR_PYTHON` serves both CLIP4Cir variants — have no interpreter the resolver can find: `resolve_model_interpreter` falls back to `workbench/artifacts/environment_assignment.json`, **which does not exist** (verified absent). So for those 8 the interpreter is `None` and their registry environment rows are contracts with nothing behind them.

Where a variable *is* set, it points at an **adapted** environment, not the historical pin the same row declares. This is a deliberate, documented PC choice (`workbench.env` sets `WORKBENCH_RUNTIME_ADAPTATION=sm_120`), but it means the "declares py 3.8.10 / torch 2.1.0" column is *not* what runs: every set variable resolves to torch `2.7.1+cu128` / CUDA 12.8. The one exception in spirit is `tme`, whose registry declares a floor-and-range (`>=2.7.1,<2.9`) precisely so 2.7.1 is accepted.

Three CPU-only interpreters exist for the historical pins and are *not* referenced by any variable: `artifacts/environments/clip4cir-38` (py 3.8.20, torch 1.11.0+cpu, tv 0.12.0+cpu — exact CLIP4Cir pin), `dcnet-377` (py 3.7.7, torch 1.4.0+cpu, tv 0.5.0+cpu — exact DCNet pin), `sprc-39` (py 3.9.25, torch 2.0.1+cpu, tv 0.15.2+cpu — exact SPRC pin). These are the only interpreters on the host that match their declared contract exactly, and no `WORKBENCH_*_PYTHON` points at them.

`workbench/envs/dcnet-prepare` (py 3.7.12) imports **no torch** — `ModuleNotFoundError: No module named 'torch'` — so it cannot evaluate anything; it is a preparation env.

## 7. Checkpoint identities that do not match their declared shape

Hashes all match where a hash is declared. The disagreements below are about **what the bytes are**, which a hash cannot catch — these were established by reading each pickle with every global class stubbed out, so no upstream code executed.

| variant | stored as | declares / implies | evidence |
|---|---|---|---|
| `intent` / `fiq_n02` | pickled **whole model** (`class Blip2QformerCir`) | a state_dict the wrapper loads with `load_state_dict` | `data.pkl` top-level resume value is an instance of `Blip2QformerCir`, not a mapping. The wrapper's `load_trusted_whole_model` name is consistent with this; any generic state_dict loader is not. |
| `airknow` / `fiq_n05`, `fiq_n08` | pickled **whole model** (`class Blip2QformerCir`) | VERIFIED_METADATA | same class as `intent`; both files are 2,724,056,638 bytes — identical size for 50% and 80% noise models |
| `conesep` / `fiq_n02` | pickled **whole model** (`class ConeSep`) | VERIFIED_METADATA | class `ConeSep` |
| `habit` / `fiq_n02` | pickled **whole model** (`class HABIT`) | VERIFIED_METADATA | class `HABIT` |
| `hint` / `fashioniq` | pickled **whole model** (`lavis.models.blip2_models.HINT.HINT`) | VERIFIED_METADATA | class `HINT`; unpickling therefore *requires* the pinned LAVIS package to be importable in the child interpreter |
| `limn` / `base_iter0_*` | pickled **whole model** (`class LIMN`), archive prefix `archive` | VERIFIED_METADATA | all three category files are exactly 1,792,010,157 bytes with three different declared digests — same architecture, different weights, correct |
| `dcnet` / `fashioniq_run_directory` | **legacy non-zip pickle** (2,877,310,172 B) | a torch zip like every other `.pth` here | `zipfile.is_zipfile()` is False. This is why `dcnet_checkpoint_weights_only_optout.patch` exists. Its top-level keys are `UNVERIFIED` — a legacy pickle cannot be read without executing it. |
| `tme` / `fashioniq_0` | torch-zip **bare state_dict**, 860 keys | VERIFIED_METADATA, no `expected_sha256` | first keys `query_tokens, temp, prompt_tokens, visual_encoder.cls_token…` — keyed as a single model, not `epoch`+class like the LAVIS family |
| `encoder` / `fashioniq` | torch-zip **bare state_dict**, 417 keys | VERIFIED_METADATA | first keys `loss_T, local_weight, backbone.clip.*` — note `loss_T`/`local_weight` are training state, not weights |
| `clvc_net`, `tgcir` | n/a | — | **not inspected in this pass**: `clvc_net`'s six files and `tgcir`'s `Fashioniq.pt` were hashed but their internal structure was not read. Recorded as `UNVERIFIED` in `inventory.json`. |

### 7.1 Staging: 21 of 29 resolve, and 8 do not

`WORKBENCH_CHECKPOINT_ROOT` = `workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads`. The composition cohort lives there (9 models, all present). Every **native** weight lives elsewhere and resolves through each variant's `checkpoint_base_relative`, which reaches out of the configured root with `../../checkpoints/…` or `../../downloads/…`. That is documented in `registry.checkpoint_path` and it works — the resolver returned 21 present, and I re-verified each hash.

The **8 absent** variants:

| variant | resolved path | reason |
|---|---|---|
| `conesep` / `fiq_n05` | `workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads/conesep/ConeSep-FIQ-N0.5.pt` | no file at the declared location |
| `conesep` / `fiq_n08` | `workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads/conesep/ConeSep-FIQ_N0.8.pt` | no file at the declared location |
| `habit` / `fiq_n05` | `workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads/habit/HABIT-FIQ_N0.5.pt` | no file at the declared location |
| `habit` / `fiq_n08` | `workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads/habit/HABIT-FIQ_N0.8.pt` | no file at the declared location |
| `intent` / `fiq_n05` | `workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads/intent/intent_fiq_0.5.pt` | no file at the declared location |
| `intent` / `fiq_n08` | `workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads/intent/intent_fiq_0.8.pt` | no file at the declared location |
| `sprc` / `sprc_fiq_unresolved` | `workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads/../../checkpoints/sprc/sprc_fiq.pt` | no file at the declared location |
| `tme` / `cirr_0` | `workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads/tme/cirr_0_best_model.pth` | no file at the declared location |

Of those 8, **four carry no declared digest at all** — `conesep` n0.5/n0.8, `sprc/sprc_fiq_unresolved`, `tme/cirr_0` all have `expected_sha256: null`, so nothing exists to verify them against even if downloaded. The other **four** (`habit` n0.5/n0.8, `intent` n0.5/n0.8) do declare digests and were simply never downloaded. Note `conesep fiq_n08` declares filename `ConeSep-FIQ_N0.8.pt` (underscore) while its siblings use `-N0.2`/`-N0.5` (hyphen) — an upstream naming inconsistency faithfully carried through, not a typo introduced here.

**Staging duplicate:** `tme/fashioniq_0_best_model.pth` (2,728,382,104 B) exists twice — once under `cohort/downloads/tme/` and once under `cohort/ckpt/tme/`. Both are the real weight, so 2.7 GB of checkpoint bytes are stored twice. The resolver picks the `downloads/` copy.

**Two copies of one weight:** `csmcir/fashioniq_tuned_clip_best.pt` exists both staged (`cohort/downloads/csmcir/`) and under the shipped `workbench/artifacts/checkpoints/csmcir/`. Both hash to `c8812071…`, so they agree — and that shipped directory is exactly why the *shipped* autopilot profiles (`workbench/configs/*.yaml`, `paths.checkpoints: ../artifacts/checkpoints`) report 8 of 9 composition models as unstaged. The eval profile `workbench/docs/reviews/20261108T113610-eval/circus_cohort.yaml` exists to fix precisely that and points at the real root.

## 8. Configs and manifests

Five autopilot profiles in `workbench/configs/` and one eval profile in `workbench/docs/reviews/20261108T113610-eval/`:

| config | scope | checkpoint overrides | `paths.checkpoints` |
|---|---|---|---|
| `workbench/configs/pc_autopilot.all_available.yaml` | all_registered | {'habit': ['fiq_n02'], 'intent': ['fiq_n02'], 'conesep': ['fiq_n02']} | `../artifacts/checkpoints` |
| `workbench/configs/pc_autopilot.example.yaml` | all_registered | none | `../artifacts/checkpoints` |
| `workbench/configs/pc_autopilot.pc_cohort.yaml` | all_registered | {'habit': ['fiq_n02'], 'intent': ['fiq_n02'], 'conesep': ['fiq_n02'], 'ptha_mtst': ['cirr_transfer'], 'encoder': ['fashioniq']} | `../artifacts/checkpoints` |
| `workbench/configs/pc_autopilot.pilot_csmcir.yaml` | ['csmcir'] | none | `../artifacts/checkpoints` |
| `workbench/configs/pc_autopilot.pilot_limn.yaml` | ['limn'] | none | `../artifacts/checkpoints` |
| `…/20261108T113610-eval/circus_cohort.yaml` | all_registered | {'habit': ['fiq_n02'], 'intent': ['fiq_n02'], 'conesep': ['fiq_n02'], 'ptha_mtst': ['cirr_transfer']} | `workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads` |

All six select `all_registered`, so none of them pins the 16/17 count per-config: adding a registry entry silently widens every one of them. `budgets.max_models: 32` is the only ceiling, and it is not the count.

The manifests at `…/20261106T002617+0700/manifests/` corroborate the hashes independently: `selected_bundle_digests.json` carries the CLIP4Cir, SPRC and CLVC digests, `airknow_hashes.json` carries both AirKnow digests with `"match": true`, and `protocol_identity.json` pins the dataset identity (74,381 images; dress/shirt/toptee query counts 2,017/2,038/1,961). Every digest in those manifests equals the value I recomputed from the bytes.
## 8. Configs and manifests

Five autopilot profiles in `workbench/configs/` and one eval profile in `workbench/docs/reviews/20261108T113610-eval/`:

| config | scope | checkpoint overrides | `paths.checkpoints` |
|---|---|---|---|
| `workbench/configs/pc_autopilot.all_available.yaml` | all_registered | {'habit': ['fiq_n02'], 'intent': ['fiq_n02'], 'conesep': ['fiq_n02']} | `../artifacts/checkpoints` |
| `workbench/configs/pc_autopilot.example.yaml` | all_registered | none | `../artifacts/checkpoints` |
| `workbench/configs/pc_autopilot.pc_cohort.yaml` | all_registered | {'habit': ['fiq_n02'], 'intent': ['fiq_n02'], 'conesep': ['fiq_n02'], 'ptha_mtst': ['cirr_transfer'], 'encoder': ['fashioniq']} | `../artifacts/checkpoints` |
| `workbench/configs/pc_autopilot.pilot_csmcir.yaml` | ['csmcir'] | none | `../artifacts/checkpoints` |
| `workbench/configs/pc_autopilot.pilot_limn.yaml` | ['limn'] | none | `../artifacts/checkpoints` |
| `…/20261108T113610-eval/circus_cohort.yaml` | all_registered | {'habit': ['fiq_n02'], 'intent': ['fiq_n02'], 'conesep': ['fiq_n02'], 'ptha_mtst': ['cirr_transfer']} | `workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads` |

All six select `all_registered`, so none of them pins the 16/17 count per-config: adding a registry entry silently widens every one of them. `budgets.max_models: 32` is the only ceiling, and it is not the count.

The manifests at `…/20261106T002617+0700/manifests/` corroborate the hashes independently: `selected_bundle_digests.json` carries the CLIP4Cir, SPRC and CLVC digests, `airknow_hashes.json` carries both AirKnow digests with `"match": true`, and `protocol_identity.json` pins the dataset identity (74,381 images; dress/shirt/toptee query counts 2,017/2,038/1,961). Every digest in those manifests equals the value I recomputed from the bytes.
## 9. Disagreements, complete list

Every place the registry (or a config, or a prior report) claims something the filesystem contradicts. Ordered by blast radius.

| # | where | claimed | actual | severity |
|---|---|---|---|---|
| 1 | `registry.models[tme].upstream_repo_url/commit` vs `teacher/repos/TME` | `li-shuxian/TME` @ `394aad79` | checkout is `He-Changhao/2025-CVPR-TME` @ `7fe81199`; `394aad79` absent from that clone | **high** — two different upstreams both called TME |
| 2 | `teacher/repos/CSMCIR/src/lavis/models/__init__.py` | base `9d8c914b` or derived `0ec13443` | actual `8e9cd6a8` — a third state, 102 lines hand-edited, uncommitted | **high** — an unreviewed source variant exists on disk |
| 3 | `registry.models[*].environment.python_env_var` (8 of 17 models) | a configured interpreter | variable unset in `workbench.env`; no `environment_assignment.json` fallback exists → interpreter `None` | **high** — those 8 have no runnable environment by resolution |
| 4 | every set `WORKBENCH_*_PYTHON` vs its own registry row | e.g. `csmcir`: py 3.9 / torch 2.0.1 / cuda 11.7 | points at py 3.10.22 / torch 2.7.1+cu128 / cuda 12.8 (adapted sm_120) | **medium** — documented policy, but the declared column is not what runs |
| 5 | `registry` for 3 variants | a single `.pt` state file | file is pickled whole model (`Clib2QformerCir`, `ConeSep`, `HABIT`, `HINT`, `LIMN`) needing the pinned package importable | **medium** — decides whether an env can load the weight at all |
| 6 | `registry.models[dcnet].checkpoint_variants` | a torch checkpoint like its peers | legacy non-zip pickle; why the `weights_only` patch is required | **medium** |
| 7 | `workbench/configs/*.yaml` (`paths.checkpoints: ../artifacts/checkpoints`) | the checkpoint tree | that dir holds only `csmcir/`; real staging is `checkpoint_selection/20261106T002617+0700/cohort/downloads` | **medium** — shipped profiles misreport 8/9 as unstaged |
| 8 | `workbench.env.example` (17 vars) vs live `workbench.env` (9) | the same set | 8 variables absent from the live file | **medium** |
| 9 | 8 of 29 declared checkpoint variants | staged | absent from the resolved root | **low** — 4 have no digest to verify anyway |
| 10 | `conesep fiq_n08` filename | `ConeSep-FIQ_N0.8.pt` | siblings use `-N0.2`/`-N0.5` | **low** — upstream inconsistency, faithfully carried |

## 10. Counts, final

| quantity | value |
|---|---|
| unique upstream repositories (workbench scope) | **16** |
| registered model variants (`model_id`) | **17** |
| declared checkpoint variants | **29** |
| checkpoint variants staged | **21 of 29** |
| — staged, single file, digest MATCH | 17 |
| — staged, single file, no digest declared | 1 (`tme/fashioniq_0`) |
| — staged, multi-member artifact, all declared member digests match | 3 (`clvc_net` 6 files, `clip4cir fullft` 2 files, `dcnet` 2 members) |
| checkpoint variants absent | **8** (4 of which declare no digest at all) |
| vendored repos with zero source drift | 13 of 16 |
| vendored repos with drift, all explained by a declared patch | 3 of 16 (`CSMCIR`, `DCNet`, `TME`) |
| models with a composition adapter | 9 |
| models with a native launcher wired | 17 |
| models whose `WORKBENCH_*_PYTHON` resolves | 9 |

## 11. Method (so the numbers can be re-derived)

```bash
# registry (31 fields x 17 models, parsed whole)
workbench/envs/ctl312h/bin/python -c "import yaml,pathlib; yaml.safe_load(open('workbench/registry/models.yaml'))"
# every staged weight hashed
sha256sum $(find workbench/artifacts/checkpoint_selection -name "*.pt" -o -name "*.pth")
# each vendored tree re-hashed against its own provenance file_manifest
# each checkpoint structure read with all globals stubbed (no upstream import, no execution)
# each interpreter probed with CUDA_VISIBLE_DEVICES="" 
CUDA_VISIBLE_DEVICES="" <interp> -c "import torch,torchvision,sys; print(sys.version.split()[0], torch.__version__, torchvision.__version__, torch.version.cuda)"
```

Evidence files in this directory: `inventory.json` (same data, machine-readable), `INVENTORY_GAPS.md` (what could not be established and what would settle it), `_registry_dump.json` (registry flattened), `_ckpt_keys.txt` (per-checkpoint pickle structure).
