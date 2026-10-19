# Specification and Registry Alignment Audit (SS_06_SPEC)

Review of `workbench/docs/MODEL_STATUS_SPEC.md` against authoritative executable configuration in `workbench/registry/models.yaml`, implementation code under `workbench/backend/`, `workbench/replay/`, `workbench/third_party/`, and handoff documents `workbench/docs/PC_PILOT_HANDOFF.md` and `workbench/docs/reviews/FINAL_PC_RELEASE_VERDICT.md`.

---

## 1. 16-Model Inventory: models.yaml vs MODEL_STATUS_SPEC.md

Every model in `workbench/registry/models.yaml` (16 total) and `workbench/docs/MODEL_STATUS_SPEC.md` (§2 table and §3 headings):

| # | `models.yaml` model_id | `models.yaml` Citation | `MODEL_STATUS_SPEC.md` Citation | Registry Status | Spec Actual Level | Missing from Either? |
|---|---|---|---|---|---|---|
| 1 | `limn` | `models.yaml:955` | `SPEC:58` (table), `:82` (§3.1) | `REPLAY_CODE_IMPLEMENTED` | `PER_QUERY_EXPORT_VERIFIED` (offline) | None (Present in both) |
| 2 | `encoder` | `models.yaml:431` | `SPEC:59` (table), `:94` (§3.2) | `reproduction_status: NOT_RUN` | `CHECKPOINT_MAPPING_VERIFIED` | None (Present in both) |
| 3 | `hint` | `models.yaml:384` | `SPEC:60` (table), `:102` (§3.3) | `reproduction_status: NOT_RUN` | `AGGREGATE_PARSED` | None (Present in both) |
| 4 | `pair` | `models.yaml:497` | `SPEC:61` (table), `:109` (§3.4) | `reproduction_status: NOT_RUN` | `CODE_TESTED` | None (Present in both) |
| 5 | `clvc_net` | `models.yaml:558` | `SPEC:62` (table), `:117` (§3.5) | `BLOCKED_NO_SOURCE_REPLAY` | `CODE_TESTED` | None (Present in both) |
| 6 | `tgcir` | `models.yaml:805` | `SPEC:63` (table), `:124` (§3.6) | `PREFLIGHT_ONLY` | `PREFLIGHT_PASS` | None (Present in both) |
| 7 | `habit` | `models.yaml:259` | `SPEC:64` (table), `:131` (§3.7) | `reproduction_status: NOT_RUN` | `CODE_TESTED` | None (Present in both) |
| 8 | `intent` | `models.yaml:321` | `SPEC:65` (table), `:131` (§3.8) | `reproduction_status: NOT_RUN` | `CODE_TESTED` | None (Present in both) |
| 9 | `conesep` | `models.yaml:156` | `SPEC:66` (table), `:131` (§3.9) | `reproduction_status: NOT_RUN` | `CODE_TESTED` | None (Present in both) |
| 10 | `airknow` | `models.yaml:103` | `SPEC:67` (table), `:140` (§3.10) | `reproduction_status: NOT_RUN` | `CODE_TESTED` | None (Present in both) |
| 11 | `dcnet` | `models.yaml:626` | `SPEC:68` (table), `:149` (§3.11) | `REPLAY_CODE_IMPLEMENTED` | `AGGREGATE_PARSED` | None (Present in both) |
| 12 | `csmcir` | `models.yaml:21` | `SPEC:69` (table), `:159` (§3.12) | `reproduction_status: NOT_RUN` | `CODE_TESTED` | None (Present in both) |
| 13 | `combiner_rn50x4_noft` | `models.yaml:680` | `SPEC:70` (table), `:174` (§3.13) | `REPLAY_CODE_IMPLEMENTED` | `SOURCE_READY` | None (Present in both) |
| 14 | `clip4cir_rn50x4_fullft` | `models.yaml:741` | `SPEC:71` (table), `:174` (§3.14) | `BLOCKED_CHECKPOINT_MAPPING` | `SOURCE_READY` | None (Present in both) |
| 15 | `sprc` | `models.yaml:867` | `SPEC:72` (table), `:185` (§3.15) | `BLOCKED_CHECKPOINT_MAPPING` | `CODE_TESTED` | None (Present in both) |
| 16 | `ptha_mtst` | `models.yaml:223` | `SPEC:73` (table), `:196` (§3.16) | `reproduction_status: NOT_RUN` | `PREFLIGHT_PASS` | None (Present in both) |

**Verification result:**
Both `workbench/registry/models.yaml` and `workbench/docs/MODEL_STATUS_SPEC.md` contain exactly 16 models. The set difference in both directions is empty (`set()`).

---

## 2. Per-Model Verification: Spec Claims vs Registry & Implementation Code

Detailed comparison of every model across patches, checkpoints, dataset layouts, protocols, and wrappers:

### 1. `limn`
- **Spec Quote (§2:58, §3:82-93):** `3 ckpt category, clean eval, training noise unstated` · `vendored, pin OK` · `fashioniq_val_split` · `limn_structured_json`. "Dataset: `fashioniq_ilearn_resized_training`, `.jpg`, correction dicts ở canonical `captions/correction_dict_{cat}.json`...".
- **Code & Registry Verification:**
  - Layout: `fashioniq_layout: fashioniq_ilearn_resized_training` (`models.yaml:981`). Matches.
  - Protocol: `native_protocol: fashioniq_val_split` (`models.yaml:984`). Matches.
  - Patches: `approved_source_patches` is empty (`models.yaml:955-1047`). Matches spec claim of unpatched upstream at commit `7d7bc9b116f5` (`models.yaml:977`).
  - Checkpoints: Registry declares 3 variants: `base_iter0_dress`, `base_iter0_shirt`, `base_iter0_toptee` (`models.yaml:1007, 1021, 1035`), filenames `0_{dress,shirt,toptee}_best_model.pt`, sizes 1792010157 bytes each, valid sha256 hashes, `checkpoint_mapping_status: VERIFIED_METADATA`. Matches.
  - Wrapper & Parser: `workbench/replay/limn.py` exists; `LIMNAdapter.command` calls `limn.py` (`workbench/backend/adapters/models.py:249-293`); parser `parse_limn_output` extracts `limn_structured_json` (`workbench/backend/metrics_extraction.py:83-225`). Matches.
- **Mismatches:** None.

### 2. `encoder`
- **Spec Quote (§2:59, §3:94-101):** `fashioniq.pt @rev dd9b0865` · `vendored + 2 local fix` · `CHECKPOINT_MAPPING_VERIFIED`. "Local fix: `datasets1.py` + `import datasets as datasets1`."
- **Code & Registry Verification:**
  - Layout: `fashioniq_layout: fashioniq_ilearn_resized_training` (`models.yaml:457`). Matches.
  - Protocol: `native_protocol: fashioniq_val_split` (`models.yaml:460`). Matches.
  - Checkpoint: `checkpoint_id: fashioniq`, filename `fashioniq.pt`, revision `dd9b086569fe69bd9111b6114546201c97934b01`, size 631879365, sha256 `0f31c501...`, `checkpoint_mapping_status: VERIFIED_METADATA` (`models.yaml:475-496`). Matches.
  - Patches: Registry declares **zero** `approved_source_patches` (`models.yaml:431-496`).
- **Mismatches:**
  - **Patches mismatch:** Spec claims `vendored + 2 local fix` (`SPEC:59, 95`), but `models.yaml:431-496` declares no `approved_source_patches`. The edits exist directly in `workbench/third_party/ENCODER/` (`datasets1.py` and `evaluate_model.py:7`) as vendored files rather than registered patch specifications.

### 3. `hint`
- **Spec Quote (§2:60, §3:102-108):** `fashioniq.pt 2 724 451 182 B` · `vendored, pin OK` · `AGGREGATE_PARSED`. "Wrapper `hint.py` chạy trong `isolated_cache_root` (F03/B3)... Parser `hint_native_recall` đọc cả hai shape thật... Hub: `iLearn-Lab/HINT`".
- **Code & Registry Verification:**
  - Layout: `fashioniq_layout: fashioniq_ilearn_resized_training` (`models.yaml:407`). Matches F05.
  - Protocol: `native_protocol: fashioniq_val_split` (`models.yaml:409`). Matches.
  - Checkpoint: 1 variant `fashioniq`, filename `fashioniq.pt`, size 2724451182, sha256 `bf9c72d4...`, `checkpoint_mapping_status: VERIFIED_METADATA` (`models.yaml:410-430`). Matches.
  - Wrapper & Parser: `workbench/replay/hint.py` exists; `HintAdapter.command` calls `hint.py` (`adapters/models.py:75-88`); `parse_hint_output` exists in `metrics_extraction.py:318-445` and dispatches `hint_native_recall`. Matches.
- **Mismatches:** None.

### 4. `pair`
- **Spec Quote (§2:61, §3:109-116):** `pair-B1.pt/pair-B2.pt` · `vendored + 1 local fix + data/datasets.py recovered` · mapping `UNRESOLVED` · `CODE_TESTED`. "Local fix: `models/MHTransformer_rev.py` = `EXPERIMENTAL_RECONSTRUCTION`... và `data/datasets.py`".
- **Code & Registry Verification:**
  - Layout: `fashioniq_layout: fashioniq_ilearn_resized_training` (`models.yaml:520`). Matches F05.
  - Protocol: `native_protocol: fashioniq_val_split` (`models.yaml:522`). Matches.
  - Checkpoint: 2 variants `pair_b1` and `pair_b2`, filenames `pair-B1.pt` and `pair-B2.pt`, sizes 694958565 and 694872371, `checkpoint_mapping_status: UNVERIFIED` (`models.yaml:523-557`). Matches.
  - Patches: Registry declares **zero** `approved_source_patches` (`models.yaml:497-557`).
- **Mismatches:**
  - **Patches mismatch:** Spec claims `vendored + 1 local fix + data/datasets.py recovered` (`SPEC:61, 109-114`), but `models.yaml:497-557` declares no `approved_source_patches`. The reconstructed `MHTransformer_rev.py` and restored `datasets.py` exist in `workbench/third_party/PAIR/` untracked by the registry patch mechanism.

### 5. `clvc_net`
- **Spec Quote (§2:62, §3:117-123):** `6 whole models (2/category)` · `vendored + torch.hub fix` · `archive CLVCNET.zip interior chưa xác minh` · `CODE_TESTED`. "Transform đã sửa (F03): native FashionIQ val = `Resize(256)→CenterCrop(224)→ToTensor→ImageNet Normalize`... wrapper trước dùng `Resize(224)`".
- **Code & Registry Verification:**
  - Layout: `fashioniq_layout: fashioniq_clvc_resized` (`models.yaml:581`). Matches.
  - Protocol: `native_protocol: fashioniq_val_split` (`models.yaml:584`). Matches.
  - Patches: Registry declares **zero** `approved_source_patches` (`models.yaml:558-625`).
  - Checkpoints: Registry declares only **1** variant: `checkpoint_id: author_drive_unresolved`, `filename: checkpoint-unresolved.pt`, `checkpoint_mapping_status: UNVERIFIED` (`models.yaml:614-625`).
- **Mismatches:**
  - **Checkpoint count mismatch:** Spec claims `6 whole models (2/category)` (`SPEC:62, 122`), but registry `models.yaml:614-625` declares only 1 dummy unresolved variant (`author_drive_unresolved`, `checkpoint-unresolved.pt`).
  - **Patches mismatch:** Spec claims `vendored + torch.hub fix` (`SPEC:62, 117`), but `models.yaml:558-625` declares no `approved_source_patches`.

### 6. `tgcir`
- **Spec Quote (§2:63, §3:124-130):** `TG-CIR.zip (Drive)` · `archive member/hash chưa xác minh; pin clip đã sửa` · `PREFLIGHT_PASS`. "F07 đã sửa: pin `clip==0.2.0` ... → OpenAI CLIP revision `a1d07173…`".
- **Code & Registry Verification:**
  - Layout: `fashioniq_layout: fashioniq_ilearn_resized_training` (`models.yaml:816`). Matches.
  - Protocol: `native_protocol: fashioniq_val_split` (`models.yaml:819`). Matches.
  - Environment: `models.yaml:852` declares `packages: ['clip @ git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33', numpy==1.22.3, pandas==1.4.2, Pillow]`. Matches F07.
  - Checkpoint: 1 variant `tgcir_archive_unresolved`, filename `tgcir-fashioniq-unresolved.pt`, `checkpoint_mapping_status: UNVERIFIED` (`models.yaml:855-866`).
  - Wrapper: `workbench/replay/tgcir.py` is preflight-only; `main()` unconditionally returns exit code 1 (`tgcir.py:92`).
- **Mismatches:**
  - **Label semantic mismatch:** Spec labels `tgcir` as `PREFLIGHT_PASS` (`SPEC:63`), but preflight execution unconditionally returns exit code 1 / `[BLOCKED]` (`workbench/replay/tgcir.py:92`).

### 7. `habit`
- **Spec Quote (§2:64, §3:131-139):** `N0.2/N0.5/N0.8, clean eval` · `vendored, pin OK` · `hash VERIFIED; chưa stage; không có bản clean` · `CODE_TESTED`. "Cả ba publish chỉ N0.2 / N0.5 / N0.8... Adapter truyền `--variant <checkpoint_id>`... Parser: shared replay (đọc `categories` — F02)".
- **Code & Registry Verification:**
  - Layout: `fashioniq_layout: fashioniq_ilearn_resized` (`models.yaml:273`). Matches.
  - Protocol: `native_protocol: fashioniq_full_gallery_ref_excluded` (`models.yaml:276`). Matches.
  - Checkpoint: 3 variants `fiq_n02`, `fiq_n05`, `fiq_n08` with filenames `HABIT-FIQ_N0.{2,5,8}.pt`, sizes 2730715930 bytes, sha256 recorded, `checkpoint_mapping_status: VERIFIED_METADATA` (`models.yaml:284-320`). Matches.
  - Adapter: `HabitAdapter.command` calls `_replay_command(..., variant=request.checkpoint_id)` (`adapters/models.py:137-139`). Matches.
- **Mismatches:** None.

### 8. `intent`
- **Spec Quote (§2:65, §3:131-139):** `N0.2/N0.5/N0.8` · `vendored, pin OK` · `hash VERIFIED; chưa stage` · `CODE_TESTED`.
- **Code & Registry Verification:**
  - Layout: `fashioniq_layout: fashioniq_ilearn_resized` (`models.yaml:335`). Matches.
  - Protocol: `native_protocol: fashioniq_full_gallery_ref_excluded` (`models.yaml:338`). Matches.
  - Checkpoint: 3 variants `fiq_n02`, `fiq_n05`, `fiq_n08` with filenames `intent_fiq_0.{2,5,8}.pt`, sizes 2722478522 bytes, sha256 recorded, `checkpoint_mapping_status: VERIFIED_METADATA` (`models.yaml:347-383`). Matches.
  - Adapter: `IntentAdapter.command` calls `_replay_command(..., variant=request.checkpoint_id)` (`adapters/models.py:147-149`). Matches.
- **Mismatches:** None.

### 9. `conesep`
- **Spec Quote (§2:66, §3:131-139):** `N0.2/N0.5/N0.8` · `vendored, pin OK` · `hash VERIFIED; chưa stage` · `CODE_TESTED`. "Lee-zixu/ConeSep (redirect)..."
- **Code & Registry Verification:**
  - Layout: `fashioniq_layout: fashioniq_ilearn_resized` (`models.yaml:173`). Matches.
  - Protocol: `native_protocol: fashioniq_full_gallery_ref_excluded` (`models.yaml:176`). Matches.
  - Checkpoints: Registry declares 3 variants `fiq_n02`, `fiq_n05`, `fiq_n08` (`models.yaml:186-222`).
    In `models.yaml:194-195`, `:206-207`, `:218-219`:
    `expected_size_bytes: null`
    `expected_sha256: null`
    `status: NO_CLEAN_CHECKPOINT`
- **Mismatches:**
  - **Checkpoint hash mismatch:** Spec line 66 claims `hash VERIFIED; chưa stage`. But `models.yaml:194-195, 206-207, 218-219` records `expected_sha256: null` and `expected_size_bytes: null` for all 3 ConeSep variants! The hashes are **not** verified in the authoritative registry.

### 10. `airknow`
- **Spec Quote (§2:67, §3:140-148):** `0.5/0.8, clean eval` · `vendored + 2 local fix` · `hash VERIFIED; chưa stage` · `CODE_TESTED`. "Local fix: guard 2 import vắng + `.png` fallback... F01 đã sửa: adapter truyền `--variant`... Native gọi `model.train()` ở `test_BLIP2.py:10`".
- **Code & Registry Verification:**
  - Layout: `fashioniq_layout: fashioniq_ilearn_resized` (`models.yaml:119`). Matches.
  - Protocol: `native_protocol: fashioniq_full_gallery_ref_excluded` (`models.yaml:122`). Matches.
  - Patches: Registry declares **zero** `approved_source_patches` (`models.yaml:103-155`).
  - Checkpoints: Registry declares 2 variants `fiq_n05` and `fiq_n08` (`models.yaml:129-155`).
    In `models.yaml:137-138` and `148-149`:
    `expected_size_bytes: null`
    `expected_sha256: null`
    `status: NO_CLEAN_CHECKPOINT`
- **Mismatches:**
  - **Checkpoint hash mismatch:** Spec line 67 claims `hash VERIFIED; chưa stage`. But `models.yaml:137-138, 148-149` records `expected_sha256: null` and `expected_size_bytes: null` for both AirKnow variants!
  - **Patches mismatch:** Spec claims `vendored + 2 local fix` (`SPEC:67, 140`), but `models.yaml:103-155` declares zero `approved_source_patches`.

### 11. `dcnet`
- **Spec Quote (§2:68, §3:149-158):** `trained_model.pth (Drive)` · `GloVe pkl chưa công bố; data_dir binding đã kiểm` · `AGGREGATE_PARSED`. "F08 đã thêm: evaluator đọc config.json:data_loader.args.data_dir... F09 đã thêm: parser dcnet_native_logger...".
- **Code & Registry Verification:**
  - Layout: `fashioniq_layout: fashioniq_dcnet` (`models.yaml:649`). Matches.
  - Protocol: `native_protocol: fashioniq_full_gallery_ref_excluded` (`models.yaml:652`). Matches.
  - Checkpoint: Registry declares 1 variant `fashioniq_run_directory`, `filename: fashioniq_dcnet` (`models.yaml:669-679`).
  - Parser: `parse_dcnet_output` implements `dcnet_native_logger` (`metrics_extraction.py:583-698`). Matches F09.
- **Mismatches:**
  - **Checkpoint filename mismatch:** Spec line 68 claims checkpoint variant is `trained_model.pth (Drive)`. In registry `models.yaml:671`, the declared filename is `fashioniq_dcnet` (with `checkpoint_serialization: run_directory`), not `trained_model.pth`.

### 12. `csmcir`
- **Spec Quote (§2:69, §3:159-173):** `fashioniq_tuned_clip_best.pt 2 720 615 859 B` · `vendored + 2 approved patches` · `ckpt hash VERIFIED; CoT captions đã có; chưa stage` · `csmcir_stdout_json` · `CODE_TESTED`. "2 approved patches: 1. src/lavis/models/__init__.py ... 2. src/validate_blip_csmcir.py (F04 mới)... Blocker captions unavailable đã rút lại...".
- **Code & Registry Verification:**
  - Layout: `fashioniq_layout: fashioniq_standard` (`models.yaml:42`). Matches.
  - Protocol: `native_protocol: fashioniq_original_split` (`models.yaml:51`). Matches.
  - Checkpoint: 1 variant `fashioniq`, filename `fashioniq_tuned_clip_best.pt`, size 2720615859 bytes, sha256 `c8812071...`, `checkpoint_mapping_status: VERIFIED_METADATA` (`models.yaml:83-102`). Matches.
  - Patches: Registry declares 2 `approved_source_patches` (`models.yaml:58-76`).
    Patch 1: `src/lavis/models/__init__.py`
    Patch 2: `src/validate_blip_csmcir.py`
  - Vendored Source Status: Neither patch is applied to `workbench/third_party/CSMCIR/`! Both files retain their `base_sha256` digest (`9d8c914b...` and `aa7a4395...`).
- **Mismatches:**
  - **Unapplied patch execution mismatch:** Spec treats F04 as `FIXED`, but the patch is not applied in the vendored tree and no runtime applier exists.

### 13. `combiner_rn50x4_noft`
- **Spec Quote (§2:70, §3:174-184):** `RN50x4 noft Combiner` · `vendored, pin OK` · `archive 5 parts, digest công bố, chưa tải` · `combiner_stdout_recall` · `SOURCE_READY`.
- **Code & Registry Verification:**
  - Layout: `fashioniq_layout: fashioniq_standard` (`models.yaml:689`). Matches.
  - Protocol: `native_protocol: fashioniq_original_split` (`models.yaml:693`). Matches.
  - Checkpoint: 1 variant `rn50x4_noft_unresolved`, `filename: rn50x4-noft-unresolved.pt`, `checkpoint_mapping_status: UNVERIFIED`, size and sha256 null (`models.yaml:728-740`).
- **Mismatches:** None.

### 14. `clip4cir_rn50x4_fullft`
- **Spec Quote (§2:71, §3:174-184):** `fullft CLIP + Combiner (cùng config)` · `vendored, pin OK` · `như trên; cần cặp khớp` · `combiner_stdout_recall` · `SOURCE_READY`.
- **Code & Registry Verification:**
  - Layout: `fashioniq_layout: fashioniq_standard` (`models.yaml:750`). Matches.
  - Protocol: `native_protocol: fashioniq_original_split` (`models.yaml:754`). Matches.
  - Checkpoint: 1 variant `rn50x4_fullft_unresolved`, `filename: rn50x4-fullft-unresolved.pt`, `checkpoint_mapping_status: UNVERIFIED`, size and sha256 null (`models.yaml:792-804`).
- **Mismatches:** None.

### 15. `sprc`
- **Spec Quote (§2:72, §3:185-195):** `ViT-g (paper) / ViT-L (variant)` · `vendored, pin OK` · `ViT-g OneDrive 403; ViT-L reachable` · `csmcir_stdout_json` · `CODE_TESTED`. "F14 đã sửa: backbone được khai báo theo variant... sprc_fiq_unresolved (backbone: pretrain = ViT-g) và sprc_fiq_vitl (backbone: pretrain_vitL)...".
- **Code & Registry Verification:**
  - Layout: `fashioniq_layout: fashioniq_standard` (`models.yaml:876`). Matches.
  - Protocol: `native_protocol: fashioniq_original_split` (`models.yaml:880`). Matches.
  - Checkpoints: Registry declares 2 variants: `sprc_fiq_unresolved` (`backbone: pretrain`, size/sha null) and `sprc_fiq_vitl` (`backbone: pretrain_vitL`, size 1459756541) (`models.yaml:916-954`). Matches F14.
  - Parser: `extract_aggregate_metrics` routes `sprc` to `parse_csmcir_output` (`metrics_extraction.py:795-798`). Matches.
- **Mismatches:** None.

### 16. `ptha_mtst`
- **Spec Quote (§2:73, §3:196-205):** `không có FashionIQ ckpt` · `vendored + 3 local fix` · `không có weights` · `PREFLIGHT_PASS`. "Upstream @ 6a02ee4763fc. Local fix: guard 1 import vắng; base_path đọc MTST_DATA_ROOT... Hub yinanzhou1/mtst_ptha publish chỉ CIRR... không có FashionIQ weight...".
- **Code & Registry Verification:**
  - Layout: `fashioniq_layout: fashioniq_standard` (`models.yaml:232`). Matches.
  - Protocol: `native_protocol: fashioniq_original_split` (`models.yaml:235`). Matches.
  - Checkpoints: `checkpoint_variants: []` (`models.yaml:258`). Matches.
  - Patches: Registry declares zero `approved_source_patches` (`models.yaml:223-258`).
  - Adapter: `ADAPTERS` in `workbench/backend/adapters/models.py:294` does **not** contain `ptha_mtst`.
- **Mismatches:**
  - **Patches mismatch:** Spec claims `vendored + 3 local fix` (`SPEC:73, 196`), but `models.yaml:223-258` declares no `approved_source_patches`.
  - **Label semantic mismatch:** Spec labels `ptha_mtst` as `PREFLIGHT_PASS` (`SPEC:73`), but the model has no checkpoint, no adapter in `ADAPTERS`, and preflight reports `[BLOCKED]`.

---

## 3. Auditing Claims of FIXED / DONE

Audit of every finding claimed as `FIXED`, `DONE`, or `ALREADY_FIXED` in `MODEL_STATUS_SPEC.md` (§4 table lines 208-233 and §3 prose):

| Finding ID | Spec Statement & Citation | Referenced Code Path | Wired into Production Launch Path? | Uncalled Helper / Broken Diagnostic Quote & Verification Details | Operational Status |
|---|---|---|---|---|---|
| F01 | `adapter airknow dùng --variant (trước --checkpoint-id)` (`SPEC:209`, `:143`) | `AirKnowAdapter.command` (`adapters/models.py:126-127`) passing `variant=(request.checkpoint_id or None)` | **YES** | Wired to `_replay_command` and accepted by `workbench/replay/airknow.py:201` (`--variant`). Callable from `evaluate_models.py:363`. | **WIRED** (Execution blocked on missing checkpoints & env) |
| F02 | `parser đọc cả categories và results; cả hai mà mâu thuẫn ⇒ từ chối` (`SPEC:210`, `:138`) | `_replay_records` (`metrics_extraction.py:465-472`) | **YES** | Called by `parse_workbench_replay_output` (`:475`), called by `extract_aggregate_metrics` (`:793`) for all 7 replay-parsed models. Tested in `test_component_boundaries.py`. | **WIRED** (Active in extraction) |
| F03 | `clvc_net.py dùng Resize(256) theo nhánh FashionIQ native` (`SPEC:211`, `:120`) | `workbench/replay/clvc_net.py:144` (`transforms.Resize(256)`) | **PARTIAL** | Code exists in wrapper, but `CLVCNetAdapter.command` calls `clvc_net.py` which unconditionally blocks at `clvc_net.py:55` (`CLVC-Net checkpoint mapping is unresolved...`). Evaluator cannot execute inference. | **PREFLIGHT-BLOCKED DIAGNOSTIC** |
| F04 | `--visualize off-by-default, qua approved patch thứ 2` (`SPEC:212`, `:163`) | `workbench/patches/csmcir_validate_visualization_opt_in.patch` declared in `models.yaml:69-76` | **NO** | **UNAPPLIED PATCH / STILL-BROKEN PATH**. The vendored file `workbench/third_party/CSMCIR/src/validate_blip_csmcir.py` is unmodified (`aa7a4395...` = `base_sha256`). There is no runtime tool applying approved patches. `CSMCIRAdapter.command` launches `src/validate_blip_csmcir.py` directly without `--visualize`, where `vis_fiq` runs unconditionally on line 78! | **UNAPPLIED PATCH** |
| F05 | `validator HINT/ENCODER/PAIR yêu cầu train; layout → ilearn_resized_training` (`SPEC:214`) | `models.yaml:407, 457, 520` and `fashioniq_layout.py:37-41, 70-71` | **YES** | Preflight (`preflight.py:173`) calls `model_requirements` which invokes `ilearn_resized_training_paths` and enforces `cap.{category}.train.json` existence. Tested in `test_verified_declarations.py`. | **WIRED** (Active in preflight) |
| F07 | `pin OpenAI CLIP revision thay clip==0.2.0` (`SPEC:217`, `:124`) | `models.yaml:852` under `tgcir.environment.packages` | **PARTIAL** | The package pin in `models.yaml:852` is updated to OpenAI CLIP revision `a1d07173...`. BUT `TGCIRAdapter` invokes `workbench/replay/tgcir.py`, whose `main()` at line 92 unconditionally returns exit code 1 (`[BLOCKED]`). It never provisions the env or executes inference. | **PREFLIGHT-ONLY DIAGNOSTIC** |
| F08 | `preflight báo data_dir mismatch/relative/missing` (`SPEC:218`, `:151`) | `dcnet.validate_dataset_binding` (`replay/dcnet.py:32-75`) | **NO** | **UNUSED HELPER**. `workbench/replay/dcnet.py:32`: ```python
def validate_dataset_binding(checkpoint_root: Path, dataset_root: Path) -> list[str]:
``` Called only at `dcnet.py:114` inside `dcnet.py:preflight`. `DCNetAdapter.command` (`adapters/models.py:161-163`) runs `source / "test.py"` directly; it does NOT invoke `dcnet.py`. Production preflight (`preflight.py:169-216`) does NOT import or call `validate_dataset_binding`. Nothing in the launch path calls it! | **UNUSED HELPER** |
| F09 | `parser DCNet, fraction→% một lần, nhánh comb:, tách comp/corr` (`SPEC:219`, `:154`) | `parse_dcnet_output` (`metrics_extraction.py:583-698`) | **YES** | `extract_aggregate_metrics` (`:790`) dispatches `model_id == "dcnet"` directly to `parse_dcnet_output`. Tested with 6 unit tests in `test_verified_declarations.py`. | **WIRED** (Active in extraction) |
| F10 | `ENCODER VERIFIED_METADATA + loader báo missing/unexpected` (`SPEC:220`, `:98`) | `models.yaml:496` and `workbench/replay/encoder.py:188-196` | **YES** | `EncoderAdapter.command` invokes `workbench/replay/encoder.py`. Lines 188-196 inspect `missing_keys` and `unexpected_keys` from `load_state_dict(strict=False)` and raise if mismatched. | **WIRED** (Active in wrapper) |
| F12 | `MHTransformer_rev.py đã label EXPERIMENTAL_RECONSTRUCTION; B1/B2 mapping giữ UNVERIFIED` (`SPEC:222`, `:110`) | `workbench/third_party/PAIR/models/MHTransformer_rev.py:1-15` | **YES** | Docstring and registry accurately record status as experimental reconstruction and unverified mapping. | **ALREADY_FIXED** (Documentation & source status) |
| F13 | `dims 2560/5120 đúng; noft/fullft tách; digest 5 part ghi lại` (`SPEC:223`, `:175`) | `adapters/models.py:174, 187` and `models.yaml:708-720, 770-785` | **YES** | `CombinerNoftAdapter` and `CLIP4CirFullftAdapter` pass `--projection-dim 2560 --hidden-dim 5120`. Checked against upstream issue #9. | **ALREADY_FIXED** (Active in adapter) |
| F14 | `backbone khai báo theo variant; không khai báo ⇒ refuse` (`SPEC:224`, `:187`) | `adapters/models.py:209-222` and `models.yaml:920, 934` | **YES** | `SPRCAdapter.command` inspects `variant.get("backbone")` from registry and raises `WorkbenchError("checkpoint_backbone_undeclared", ...)` if absent. Tested in `test_verified_declarations.py`. | **WIRED** (Active in adapter) |
| F16 | `176 test LIMN/export/replay/fresh-clone pass; không sửa code LIMN` (`SPEC:226`, `:90`) | Full test execution across LIMN suites | **YES** | Test pass confirmed. Code in `workbench/third_party/LIMN/` remains unmodified. | **ALREADY_FIXED** (Verification claim) |
| CoT Captions | `Blocker "captions unavailable" đã rút lại` (`SPEC:166`) | `auxiliary_assets.yaml:72-135` (`COT_ours2 captions`) | **YES** | `auxiliary_assets.yaml` stages published CoT captions from `peng12138/CSMCIR`. Provenance and sha256 recorded. | **WITHDRAWN BLOCKER** |

---

## 4. Spec Treatment of F06 (BLIP-2 Processor Reconstruction)

### Verbatim Quote of F06 from §4 Table (`SPEC:215` / `:226`)
```markdown
| F06 | `DEFERRED_WITH_REASON` | Thay `load_model_and_preprocess` bằng processor trực tiếp: **chưa áp**. Đã chứng minh được 5 model resolve **cùng** khai báo processor (`prompt=""`, `max_words=50`) từ config pinned (`test_blip2_text_processor_parity.py`), nhưng chưa chứng minh được **output parity** vì `lavis.processors` kéo `torchaudio` — không cài được trên host này; mọi assertion kiểu đó sẽ là test một stub. Điều kiện áp: chứng minh prompt/max_words/punctuation/spaces/truncation không đổi **trên runtime có torchaudio**. | `test_blip2_text_processor_parity.py` (6) | `UNFINISHED_ENGINEERING` |
```

### Treatment in Each Affected Model's Section
The 5 affected BLIP-2 models sharing this text processor configuration are: `hint`, `habit`, `intent`, `conesep`, and `airknow` (proven by `test_blip2_text_processor_parity.py`).

1. **`hint` (§3.3, `SPEC:102-108`):**
   Does **NOT** mention F06, `load_model_and_preprocess`, or processor reconstruction anywhere in lines 102–108.
2. **`habit` (§3.7–9, `SPEC:131-139`):**
   Grouped with `intent` and `conesep`. Does **NOT** mention F06, `load_model_and_preprocess`, or processor reconstruction anywhere in lines 131–139.
3. **`intent` (§3.7–9, `SPEC:131-139`):**
   Does **NOT** mention F06, `load_model_and_preprocess`, or processor reconstruction anywhere in lines 131–139.
4. **`conesep` (§3.7–9, `SPEC:131-139`):**
   Does **NOT** mention F06, `load_model_and_preprocess`, or processor reconstruction anywhere in lines 131–139.
5. **`airknow` (§3.10, `SPEC:140-148`):**
   Does **NOT** mention F06, `load_model_and_preprocess`, or processor reconstruction anywhere in lines 140–148.

### Missing Model Rows in Section 2 Table
In the Section 2 overview table (`SPEC:58-73`):
- Row 3 (`hint`), Row 7 (`habit`), Row 8 (`intent`), Row 9 (`conesep`), and Row 10 (`airknow`) contain **zero** mention of F06, text preprocessing, or processor reconstruction.
- In fact, **none** of the 16 rows in the Section 2 table mention F06. The treatment appears exclusively in the §4 findings table at line 226.

---

## 5. Stale Claims in Handoff Documents

Audit of stale claims in `workbench/docs/PC_PILOT_HANDOFF.md` and `workbench/docs/reviews/FINAL_PC_RELEASE_VERDICT.md`:

### (a) "N models still need wrappers"
- **Stale Claim 1:** `workbench/docs/PC_PILOT_HANDOFF.md:24`:
  ```markdown
  10 model còn lại cần wrapper. Ma trận đầy đủ ở workbench/docs/reviews/r8/R8_16_MODEL_STATUS.md.
  ```
- **Stale Claim 2:** `workbench/docs/reviews/FINAL_PC_RELEASE_VERDICT.md:106`:
  ```markdown
  | EXTERNAL_BLOCKER — no CLI entrypoint in the pinned upstream | habit, intent, hint, clvc_net, tgcir |
  ```
- **Contradicting Code Evidence:**
  Quote of `workbench/backend/adapters/models.py:294`:
  ```python
  ADAPTERS = {adapter.model_id: adapter for adapter in (CSMCIRAdapter, EncoderAdapter, HintAdapter, PairAdapter, AirKnowAdapter, ConeSepAdapter, HabitAdapter, IntentAdapter, CLVCNetAdapter, DCNetAdapter, CombinerNoftAdapter, CLIP4CirFullftAdapter, TGCIRAdapter, SPRCAdapter, LIMNAdapter)}
  ```
  `workbench/replay/` contains **12** replay wrapper modules:
  `airknow.py`, `clvc_net.py`, `conesep.py`, `dcnet.py`, `encoder.py`, `habit.py`, `hint.py`, `intent.py`, `limn.py`, `pair.py`, `ptha_mtst.py`, `tgcir.py`.
  In `workbench/backend/adapters/models.py`, 8 adapters route directly through `_replay_command` (`encoder`, `pair`, `airknow`, `conesep`, `habit`, `intent`, `clvc_net`, `tgcir`), `hint` and `limn` have dedicated wrapper calls, and 4 models use official CLI scripts (`csmcir`, `dcnet`, `combiner_rn50x4_noft`, `clip4cir_rn50x4_fullft`, `sprc`). The claim that "10 models still need wrappers" is completely stale.

### (b) Test Counts
- **Stale Claim 1:** `workbench/docs/PC_PILOT_HANDOFF.md:30, 56`:
  `SHA ĐÃ KIỂM THỬ (nguồn sản xuất): 119df5f — ... (1609 passed; acceptance, mutation và release_verify đều chạy lại trên cùng commit)`
- **Stale Claim 2:** `workbench/docs/reviews/FINAL_PC_RELEASE_VERDICT.md:53`:
  `| Backend suite | pytest workbench/tests -q | **1609 passed, 0 failed** (24:39); 1610 collected... |`
- **Contradicting Evidence:**
  Executed command: `.venv/bin/python -m pytest workbench/tests --collect-only -q`
  Observed output: `1610 tests collected in 1.87s`.
  The full collection count is 1610. The count "1609 passed" reflects a frozen run where the self-referential release document check was pending.
- **Stale Claim 3:** `workbench/docs/MODEL_STATUS_SPEC.md:90, 225`:
  `176 test LIMN/export/replay/fresh-clone pass`
  Observed: Historical snapshot count from a previous coordinator run; the active suite collects 1610 tests across all modules.

### (c) Stale Git SHAs
- **Stale Claim 1:** `workbench/docs/reviews/FINAL_PC_RELEASE_VERDICT.md:153-155`:
  ```bash
  git checkout 59949ff           # the TESTED_SOURCE_SHA above; never `git reset --hard`
  git rev-parse HEAD             # must print the full SHA whose short form is 59949ff
                                 # (59949ff8656cb1f3227425c07d8c94997b47670f)
  ```
  **Contradicting Evidence:**
  In the very same document (`FINAL_PC_RELEASE_VERDICT.md`):
  Line 7: `**TESTED_SOURCE_SHA** | **119df5f**`
  Line 19: `TESTED_SOURCE_SHA | 119df5f`
  Line 44: `The full verification ran against 119df5f.`
  Line 47: `The earlier 59949ff pass is superseded: the vendored-source work changed production files...`
  Line 153 directly contradicts lines 7, 19, 44, and 47 by embedding the superseded SHA `59949ff` in the operator shell instructions.
- **Stale Claim 2:** `workbench/docs/MODEL_STATUS_SPEC.md:12`:
  `| Source SHA đã kiểm | c3d747791417db29aa8c4cc5ab27fa3ac19e70ad |`
  **Contradicting Evidence:**
  `git log c3d7477..HEAD` shows `c3d7477` is 7 commits behind HEAD (`914703e`), and 4 commits behind tested source commit `119df5f`.
- **Stale Claim 3:** `workbench/docs/reviews/FINAL_PC_RELEASE_VERDICT.md:108`:
  `| EXTERNAL_BLOCKER — no pinned source at all | ptha_mtst |`
  **Contradicting Evidence:**
  `FINAL_PC_RELEASE_VERDICT.md:25`:
  "`ptha_mtst` is vendored and pinned at `6a02ee4763fc` with protocol `fashioniq_original_split`, and stays blocked only because no FashionIQ checkpoint is published for it"
  and `models.yaml:231`:
  `upstream_commit_sha: 6a02ee4763fc2260b6abfb18be78853facb91d86`.
  `ptha_mtst` is vendored in `workbench/third_party/MTST_PTHA` and pinned at `6a02ee4763fc`.

---

## 6. Preservation of the Three Protocols

### Definitions in Registry and Spec

1. **`workbench/registry/models.yaml:8-19`**:
   ```yaml
   protocols:
     fashioniq_original_split:
       label: FashionIQ - Original Split (Full Gallery, Reference Included)
       literature_split_label: original
       evaluation_noise_pct: 0
     fashioniq_full_gallery_ref_excluded:
       label: FashionIQ - Original Full Gallery, Reference Excluded
       literature_split_label: original
       evaluation_noise_pct: 0
     fashioniq_val_split:
       label: FashionIQ - Val Split (Pair-Union Gallery, Reference Excluded)
       literature_split_label: val
       evaluation_noise_pct: 0
   ```

2. **`workbench/docs/MODEL_STATUS_SPEC.md:43-48`**:
   ```markdown
   | protocol | gallery | reference |
   |---|---|---|
   | `fashioniq_original_split` | full split | **eligible** |
   | `fashioniq_full_gallery_ref_excluded` | full split | excluded |
   | `fashioniq_val_split` | pair-union (ordered) | excluded |
   ```

3. **`workbench/docs/reviews/FINAL_PC_RELEASE_VERDICT.md:118-124`**:
   ```markdown
   | Protocol | Candidate universe | Reference | Models |
   |---|---|---|---|
   | `fashioniq_original_split` | full validation gallery | **eligible** | csmcir, combiner_rn50x4_noft, clip4cir_rn50x4_fullft, sprc |
   | `fashioniq_full_gallery_ref_excluded` | full validation gallery | excluded | airknow, conesep, habit, intent, dcnet |
   | `fashioniq_val_split` | deduplicated ordered reference/target union | excluded | hint, encoder, pair, clvc_net, tgcir, limn |
   ```

### Conflation Assessment
- **Does the spec conflate the three protocols? NO.**
  The spec strictly preserves the tripartite separation:
  - Protocol A (`fashioniq_original_split`): 4 models (`csmcir`, `combiner_rn50x4_noft`, `clip4cir_rn50x4_fullft`, `sprc`, plus `ptha_mtst`).
  - Protocol B (`fashioniq_full_gallery_ref_excluded`): 5 models (`habit`, `intent`, `conesep`, `airknow`, `dcnet`).
  - Protocol C (`fashioniq_val_split`): 6 models (`limn`, `encoder`, `hint`, `pair`, `clvc_net`, `tgcir`).
  - §5 Rule 7 (`SPEC:248`): "Ba protocol giữ riêng; không trộn cohort trong một bảng so sánh."
- Note on historical defect: `FINAL_PC_RELEASE_VERDICT.md:27` marked `ptha_mtst` as `unresolved` (due to F-10 launcher defect), whereas `models.yaml:235` and `MODEL_STATUS_SPEC.md:73` assign it to `fashioniq_original_split` based on native `validate_blip.py:44-51` having no reference mask.

---

## 7. Evidence Levels and Status Label Accuracy

### Hierarchy of Evidence Levels in Spec (§1, lines 20-33)
The spec defines 11 distinct levels:
1. `SOURCE_READY` (static vendored source integrity)
2. `ENTRYPOINT_EXISTS` (static entrypoint file presence)
3. `PREFLIGHT_PASS` (preflight runs and reports blockers on missing input)
4. `CHECKPOINT_MAPPING_VERIFIED` (author metadata revision/size/hash verified)
5. `CHECKPOINT_LOADED` (real checkpoint loaded — explicitly marked `NOT_CHECKED toàn bộ`)
6. `CODE_TESTED` (adapter/argparse/parser tested by pytest)
7. `GPU_SMOKE_VERIFIED` (1 batch on real GPU — explicitly marked `NOT_CHECKED`)
8. `FULL_VAL_VERIFIED` (all 3 categories evaluated — explicitly marked `NOT_CHECKED`)
9. `AGGREGATE_PARSED` (parser unit-tested on real output format)
10. `PER_QUERY_EXPORT_VERIFIED` (exporter + parity with native, offline only for `limn`)
11. `PAPER_PARITY_CHECKED` (reproduced numbers compared with paper — explicitly marked `NOT_CHECKED`)

### Analysis of Inaccurate Status Labels

1. **Inaccurate `PREFLIGHT_PASS` for `tgcir` (`SPEC:63`) and `ptha_mtst` (`SPEC:73`):**
   - In `workbench/replay/tgcir.py:92`, `main()` unconditionally returns exit code 1 (`[BLOCKED]`). Preflight never passes.
   - For `ptha_mtst`, no FashionIQ checkpoint exists, `checkpoint_variants: []`, and no adapter exists in `ADAPTERS`.
   - Calling this state `PREFLIGHT_PASS` uses "PASS" to mean "the blocker-reporting diagnostic ran and reported blockers", which directly contradicts engineering intuition that preflight passed and execution is permitted. Spec rule §5.2 admits this danger: "`PREFLIGHT_PASS` không thành inference-ready bằng đổi tên status."
2. **Inconsistent `CHECKPOINT_MAPPING_VERIFIED` for `encoder` (`SPEC:59`):**
   - `encoder` has a tested replay wrapper (`test_encoder_replay.py:1-5`) and a tested replay parser (`test_replay_output_parser.py:test_every_wrapper_model_can_be_parsed[encoder]`).
   - Sibling models with unverified checkpoint mappings (`pair`, `clvc_net`) are labelled `CODE_TESTED`, but `encoder` is labelled with a metadata tag (`CHECKPOINT_MAPPING_VERIFIED`), obscuring that its code is equally tested.
3. **Inconsistent `CODE_TESTED` for `csmcir` (`SPEC:69`) and `sprc` (`SPEC:72`):**
   - `csmcir` has verified checkpoint metadata (`VERIFIED_METADATA`) and an active aggregate parser (`csmcir_stdout_json`). If `hint` is labelled `AGGREGATE_PARSED`, `csmcir` also satisfies `AGGREGATE_PARSED`.
   - Furthermore, `csmcir` execution relies on `csmcir_validate_visualization_opt_in.patch`, which is **unapplied** in the vendored tree, so the real code path is not ready.
4. **Inconsistent `SOURCE_READY` for CLIP4Cir models (`SPEC:70, 71`):**
   - `combiner_rn50x4_noft` and `clip4cir_rn50x4_fullft` have implemented CLI adapters and a tested parser (`parse_combiner_output` in `test_replay.py`). Labelling them only `SOURCE_READY` understates their code implementation relative to `pair` or `clvc_net`.
5. **Limn `PER_QUERY_EXPORT_VERIFIED (offline)` (`SPEC:58`):**
   - The spec honestly qualifies this with `(offline)`, which is accurate because no real GPU inference has taken place. Full verification remains pending until target hardware execution.

---

## Verified claims

| Claim | Command / Verification Route | Observed Output |
|---|---|---|
| Registry models total exactly 16 | `.venv/bin/python -c "import yaml; m=yaml.safe_load(open('workbench/registry/models.yaml'))['models']; print(len(m), [x['model_id'] for x in m])"` | `16 ['csmcir', 'airknow', 'conesep', 'ptha_mtst', 'habit', 'intent', 'hint', 'encoder', 'pair', 'clvc_net', 'dcnet', 'combiner_rn50x4_noft', 'clip4cir_rn50x4_fullft', 'tgcir', 'sprc', 'limn']` |
| Spec models total exactly 16 | Python regex over `workbench/docs/MODEL_STATUS_SPEC.md` §2 table and §3 headings | Exactly the same 16 model IDs; symmetric difference is `set()` |
| Pytest suite collects 1610 tests | `.venv/bin/python -m pytest workbench/tests --collect-only -q` | `1610 tests collected in 1.87s` |
| ADAPTERS dictionary contains 15 models | Python inspection of `workbench/backend/adapters/models.py:294` | `len(ADAPTERS) == 15`; all models present except `ptha_mtst` |
| Replay directory contains 12 wrappers | `ls workbench/replay/*.py` | `airknow.py`, `clvc_net.py`, `conesep.py`, `dcnet.py`, `encoder.py`, `habit.py`, `hint.py`, `intent.py`, `limn.py`, `pair.py`, `ptha_mtst.py`, `tgcir.py` (12 wrappers) |
| ConeSep checkpoint sha256 is null in registry | `.venv/bin/python -c "import yaml; m=[x for x in yaml.safe_load(open('workbench/registry/models.yaml'))['models'] if x['model_id']=='conesep'][0]; print([c['expected_sha256'] for c in m['checkpoint_variants']])"` | `[None, None, None]` |
| AirKnow checkpoint sha256 is null in registry | `.venv/bin/python -c "import yaml; m=[x for x in yaml.safe_load(open('workbench/registry/models.yaml'))['models'] if x['model_id']=='airknow'][0]; print([c['expected_sha256'] for c in m['checkpoint_variants']])"` | `[None, None]` |
| CSMCIR vendored validate_blip_csmcir.py is unpatched | `.venv/bin/python -c "import hashlib; print(hashlib.sha256(open('workbench/third_party/CSMCIR/src/validate_blip_csmcir.py','rb').read()).hexdigest())"` | `aa7a4395cdb700fded0567e0969da774a331927eaea327b5a23da82fdc7506a9` (= `base_sha256`) |
| CSMCIR vendored lavis __init__.py is unpatched | `.venv/bin/python -c "import hashlib; print(hashlib.sha256(open('workbench/third_party/CSMCIR/src/lavis/models/__init__.py','rb').read()).hexdigest())"` | `9d8c914b551a7de614e062f9a241f44626fd591a31a0ccf1cc79f0ae23c9db01` (= `base_sha256`) |
| `validate_dataset_binding` is dead code in production | `git grep "validate_dataset_binding"` | Found only in `workbench/replay/dcnet.py:32, 114` and `workbench/docs/MODEL_STATUS_SPEC.md:228` |
| TGCIR replayer main() unconditionally returns 1 | Code inspection of `workbench/replay/tgcir.py:85-92` | `blockers = preflight(...); for b in blockers: print(f'[BLOCKED] {b}', file=sys.stderr); return 1` |
| Stale checkout 59949ff in verdict document | Code inspection of `workbench/docs/reviews/FINAL_PC_RELEASE_VERDICT.md:153-155` | Contains `git checkout 59949ff` and claims it is `the TESTED_SOURCE_SHA above` |
| Stale "10 models need wrappers" claim | Code inspection of `workbench/docs/PC_PILOT_HANDOFF.md:24` | `10 model còn lại cần wrapper. Ma trận đầy đủ ở workbench/docs/reviews/r8/R8_16_MODEL_STATUS.md.` |
| Git HEAD is 914703e | `git rev-parse --short HEAD` | `914703e` |
