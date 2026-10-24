# MODEL_STATUS_SPEC — trạng thái 17 model CIR

Tài liệu trạng thái chuẩn. Registry (`workbench/registry/models.yaml`) vẫn là **nguồn tham số thực thi**;
file này mô tả **tình trạng, bằng chứng và điều kiện chuyển trạng thái**. Không có backend status
service, database hay dashboard mới.

## Current closure snapshot — 2026-10-24

This table supersedes the dated status snapshots below; it records current registry/source/readiness evidence, not a claim that GPU inference occurred. `aeff57a` (`aeff57ab6dae18e4b195cd7f6e584df952652db1`) is the revision whose checks ran; `git rev-parse HEAD` on the branch tip (the docs-only successor) is the final head, which differs only in documentation. Production bytes are identical between them. Source pins below match clean local pinned checkouts. `VERIFIED_METADATA` means source metadata binds checkpoint identity; it does not mean a local file is present or loaded. Local availability is from this host's configured checkpoint root. Training/evaluation noise values are registry conditions; unknown training values remain unknown. The encoder row records a REAL full three-category GPU evaluation (see below); no other model has GPU inference, target-GPU validation or a live Docker run recorded in this snapshot; last real validation command/revision/artifact is **not established here** (historical reports/artifacts below remain historical).



The per-model rows record registry/source pin checks, configured checkpoint-root presence, and `prepare_model.py --list --json` status on the working tree; that command exits 1 when expected model blockers are reported. These are inventory/preflight findings, not model inference or GPU validation. `aeff57a` (`aeff57ab6dae18e4b195cd7f6e584df952652db1`) identifies the production revision whose checks ran; no row claims an evaluation artifact unless explicitly stated.

Current operator surface: `doctor.py --checkpoint-policy {strict_clean,include_flagged_conditions}`; `pipeline.py --allow-gpu-preparation`; optional Autopilot `permissions.allow_gpu_preparation` defaults to `false` and separately gates TME's CUDA-only validation-feature generator (about 10.47 GiB). Both `allow_preparation` and GPU-preparation authorization are required to generate it. `workbench/config/workbench.env` accepts every registry-declared per-model interpreter and asset-location setting; process environment wins over file values, then verified provisioning assignment. This includes `WORKBENCH_TME_PYTHON` and `WORKBENCH_PTHA_MTST_PYTHON`. `WORKBENCH_TME_ASSET_EVA_VIT_G` is not declared; both TME base weights are downloaded with `python -m workbench.scripts.download_auxiliary_assets --model tme`.
| Model | Source URL @ pin; approved patch/integrity | Checkpoint(s): source, local, mapping/digest; train dataset/noise → eval noise | Base assets / data preparation | Runtime, hardware, capability; last real validation | Remaining blocker and next action | Revision whose checks ran |
|---|---|---|---|---|---|---|
| `csmcir` | `https://github.com/qzp2018/CSMCIR` @ `774f94e2076ff17ea91703a6239d2a08f0e1a44e`; two registry-approved patches unapplied; clean pin | `fashioniq` HF absent; `VERIFIED_METADATA`, SHA `c8812071…`; training dataset/noise unknown/0% registry → eval 0% | Patches, BLIP-2/EVA/tokenizer, COT captions and dataset link not ready | `WORKBENCH_CSMCIR_PYTHON` unset; source import blocked; no current hardware validation | `BLOCKED_SOURCE`, `BLOCKED_ASSET`, `BLOCKED_ENVIRONMENT`; apply approved patches, verify assets/captions, configure interpreter | `aeff57a` (`aeff57ab6dae18e4b195cd7f6e584df952652db1`) |
| `airknow` | `https://github.com/iLearn-Lab/CVPR26-Air-Know` @ `6ab6fa95b746f85b08eea4cb66eb51448bce99b7`; no patch; clean pin | `fiq_n05`/`fiq_n08` HF absent; metadata verified, no SHA; 50%/80% training → 0% eval; 20% unavailable | EVA/BLIP, tokenizer, correction dictionaries, training-resized images unverified | `WORKBENCH_AIRKNOW_PYTHON` unset; no hardware test | `BLOCKED_CHECKPOINT_CONDITION`, `BLOCKED_ASSET`, `BLOCKED_ENVIRONMENT`; never substitute 50% for 20%; resolve assets/runtime | `aeff57a` (`aeff57ab6dae18e4b195cd7f6e584df952652db1`) |
| `conesep` | `https://github.com/Lee-zixu/ConeSep` @ `bd4bf629ebe68045c1a54dcc61e4a3ea65f8759b`; no patch; clean pin | `fiq_n02`/`fiq_n05`/`fiq_n08` HF absent; mapping metadata verified; n02 SHA `5a864ee5…`; 20%/50%/80% training → 0% eval | EVA/BLIP, tokenizer, correction dictionaries, resized training images unverified | `WORKBENCH_CONESEP_PYTHON` unset; no hardware test | `BLOCKED_CHECKPOINT_CONDITION`, `BLOCKED_ASSET`, `BLOCKED_ENVIRONMENT`; select condition, verify assets/runtime | `aeff57a` (`aeff57ab6dae18e4b195cd7f6e584df952652db1`) |
| `ptha_mtst` | `https://github.com/zyinan99/MTST_PTHA` @ `6a02ee4763fc2260b6abfb18be78853facb91d86`; local source fix recorded; clean pin | `cirr_transfer` HF present; `VERIFIED_METADATA`, SHA `1c4dab86…`, 2,720,352,973 B; CIRR training → FashionIQ eval 0% (transfer only) | Standard dataset; no additional generated data listed | `WORKBENCH_PTHA_MTST_PYTHON` unset; no hardware test; no evaluation | `BLOCKED_ENVIRONMENT`; verify dedicated interpreter, retain transfer label, not FashionIQ reproduction | `aeff57a` (`aeff57ab6dae18e4b195cd7f6e584df952652db1`) |
| `habit` | `https://github.com/iLearn-Lab/AAAI26-HABIT` @ `6ddc2cd63eac1f65693f4e821c34fb3b42aee306`; no patch; clean pin | `fiq_n02` present; `fiq_n05`/`fiq_n08` absent; metadata verified with SHA; 20%/50%/80% training → 0% eval | EVA/BLIP, tokenizer, correction dictionaries, resized training images unverified | `WORKBENCH_HABIT_PYTHON` unset; no hardware test | `BLOCKED_CHECKPOINT_CONDITION`, `BLOCKED_ASSET`, `BLOCKED_ENVIRONMENT`; resolve condition/assets/runtime | `aeff57a` (`aeff57ab6dae18e4b195cd7f6e584df952652db1`) |
| `intent` | `https://github.com/iLearn-Lab/AAAI26-INTENT` @ `2a75fc6f57d1a036bf3167a7fe1d2cee072f51f3`; no patch; clean pin | `fiq_n02` present; `fiq_n05`/`fiq_n08` absent; metadata verified with SHA; 20%/50%/80% training → 0% eval | EVA/BLIP, tokenizer, correction dictionaries, resized training images unverified | `WORKBENCH_INTENT_PYTHON` unset; no hardware test | `BLOCKED_CHECKPOINT_CONDITION`, `BLOCKED_ASSET`, `BLOCKED_ENVIRONMENT`; resolve condition/assets/runtime | `aeff57a` (`aeff57ab6dae18e4b195cd7f6e584df952652db1`) |
| `hint` | `https://github.com/iLearn-Lab/ICASSP26-HINT` @ `bec50b6c8c19111893b617979502b948d1cea5b2`; no patch; clean pin | `fashioniq` HF absent; metadata SHA `bf9c72d4…`; train noise/dataset unknown → eval 0% | Correction dictionaries and resized training images unverified | `WORKBENCH_HINT_PYTHON` unset; no hardware test | `BLOCKED_ASSET`, `BLOCKED_ENVIRONMENT`; stage assets and verify interpreter | `aeff57a` (`aeff57ab6dae18e4b195cd7f6e584df952652db1`) |
| `encoder` | `https://github.com/iLearn-Lab/AAAI25-ENCODER` @ `29a2a31d6a56f677bf450c3be7cdaef423fb7018`; no patch; clean pin | `fashioniq` present; `VERIFIED_METADATA`, SHA `0f31c501…`, 631,879,365 B; training condition unknown → eval 0% | None missing; OpenCLIP binary asset present and hash-matched | `WORKBENCH_ENCODER_PYTHON=workbench/envs/encoder/bin/python`; GPU probe recorded: RTX 3050 Laptop, sm 8.6, driver 580.178.04, torch 1.13.1+cu117 | **FULL 3-CATEGORY GPU EVALUATION COMPLETED** 2026-10-24 (`--verify-gpu`, protocol `fashioniq_val_split`): dress R@10 51.56/R@50 76.25, shirt 54.76/75.37, toptee 61.86/81.13, macro 56.06/77.58 vs paper 56.13/77.59; `verify_run.py` → `verified=true, is_research_evidence=true`; gate `PASS_METRICS_ONLY_MAPPING_UNVERIFIED` (<5% deviation but checkpoint mapping unverified, so metric comparison ONLY, not a certified official reproduction) | None for this evaluation; an official-parity claim would need the author checkpoint mapping | `aeff57a` (`aeff57ab6dae18e4b195cd7f6e584df952652db1`) |
| `pair` | `https://github.com/iLearn-Lab/ICASSP25-PAIR` @ `bfb0b98cdb4c44f6b9190baaa84d783a77dd888f`; no patch; clean pin | `pair_b1`/`pair_b2` HF present; mapping `UNVERIFIED`, digest unknown; train dataset/noise unknown → eval 0% | Author correction dictionaries unpublished/unverified; no declared generated artifacts | `WORKBENCH_PAIR_PYTHON` unset; command `COMMAND_BLOCKED_SOURCE_ENTRYPOINT`; no hardware test | `BLOCKED_SOURCE_ENTRYPOINT`, `BLOCKED_CHECKPOINT_MAPPING`; obtain authoritative mapping/source integration and interpreter | `aeff57a` (`aeff57ab6dae18e4b195cd7f6e584df952652db1`) |
| `clvc_net` | `https://github.com/iLearn-Lab/SIGIR21-CLVC-Net` @ `bd9b6889489806cb60676597880f91d918b279cb`; no patch; clean pin | `author_drive_unresolved` Google Drive absent; mapping/digest unknown; train dataset/noise unknown → eval 0% | ResNet-50 weights absent; manual preparation; no mapped checkpoint | `WORKBENCH_CLVC_PYTHON` unset; no audited command constructor or hardware test | `BLOCKED_CHECKPOINT_MAPPING`, `BLOCKED_SOURCE_ENTRYPOINT`, `BLOCKED_ASSET`; obtain mapped checkpoint, audited command and asset | `aeff57a` (`aeff57ab6dae18e4b195cd7f6e584df952652db1`) |
| `dcnet` | `https://github.com/ozmig77/dcnet` @ `68c79d38569f01ca39c3b4ea641b3b7eb2ccb5fb`; no patch; clean pin | `fashioniq_run_directory` Google Drive absent; mapping/digest unknown; train dataset/noise unknown → eval 0% | Author GloVe caption PKLs, resized images and ResNet-50 weights absent; avoid nondeterministic regeneration | `WORKBENCH_DCNET_PYTHON` unset; command audited, no evaluation or hardware test | `BLOCKED_CHECKPOINT_MAPPING`, `BLOCKED_DATA_PREPARATION`, `BLOCKED_ASSET`, `BLOCKED_ENVIRONMENT`; obtain author mapping/artifacts and verify runtime | `aeff57a` (`aeff57ab6dae18e4b195cd7f6e584df952652db1`) |
| `combiner_rn50x4_noft` | `https://github.com/ABaldrati/CLIP4Cir` @ `dfed9f748a8a4d05abf613164e06d59b22dbdf13`; no patch; clean pin | `rn50x4_noft_unresolved` Google Drive absent; mapping/digest unknown; train condition unknown → eval 0% | RN50x4 weights absent; dataset image extension/layout checks fail | `WORKBENCH_CLIP4CIR_PYTHON` unset; no hardware test/evaluation | `BLOCKED_CHECKPOINT_MAPPING`, `BLOCKED_ASSET`, `BLOCKED_DATASET`, `BLOCKED_ENVIRONMENT`; map checkpoint/base weights and provide compatible data | `aeff57a` (`aeff57ab6dae18e4b195cd7f6e584df952652db1`) |
| `clip4cir_rn50x4_fullft` | Same CLIP4Cir pin `dfed9f748a8a4d05abf613164e06d59b22dbdf13`; no patch; clean pin | `rn50x4_fullft_unresolved` Google Drive absent; mapping/digest unknown; train condition unknown → eval 0% | Matching bundle and RN50x4 weights unresolved; dataset image extensions fail | Shared `WORKBENCH_CLIP4CIR_PYTHON` unset; no hardware test/evaluation | `BLOCKED_CHECKPOINT_BUNDLE`, `BLOCKED_CHECKPOINT_MAPPING`, `BLOCKED_ASSET`, `BLOCKED_DATASET`; establish bundle and inputs | `aeff57a` (`aeff57ab6dae18e4b195cd7f6e584df952652db1`) |
| `tgcir` | `https://github.com/iLearn-Lab/MM23-TG-CIR` @ `65fa78eaf8cabe8197fcc22ccab42207c591bdee`; no patch; clean pin | `tgcir_archive_unresolved` Google Drive absent; mapping/digest unknown; train dataset/noise unknown → eval 0% | ViT-B/16 provenance unresolved; source cache freshness unverified | `WORKBENCH_TGCIR_PYTHON` unset; `COMMAND_BLOCKED_SOURCE_ENTRYPOINT`; no hardware test | `BLOCKED_SOURCE_ENTRYPOINT`, `BLOCKED_CHECKPOINT_MAPPING`, `BLOCKED_ASSET`; establish entrypoint, archive mapping and asset provenance | `aeff57a` (`aeff57ab6dae18e4b195cd7f6e584df952652db1`) |
| `sprc` | `https://github.com/chunmeifeng/SPRC` @ `2935a5397732260d1db6fa577e5926f963e36f0f`; no patch; clean pin | `sprc_fiq_unresolved` present without verified digest/mapping; `sprc_fiq_vitl` absent; author sources; train dataset/noise unknown → eval 0% | BLIP-2/EVA assets missing/unverified; ViT-g/ViT-L mapping unresolved; manual preparation | `WORKBENCH_SPRC_PYTHON` unset; `COMMAND_BLOCKED_MODEL_MAPPING`; no hardware test | `BLOCKED_CHECKPOINT_MAPPING`, `BLOCKED_ASSET`, `BLOCKED_ENVIRONMENT`; verify author mapping/assets and interpreter | `aeff57a` (`aeff57ab6dae18e4b195cd7f6e584df952652db1`) |
| `limn` | `https://github.com/iLearn-Lab/TPAMI24-LIMN` @ `7d7bc9b116f594a65ac22457491edf28a88d3c3e`; no patch; clean pin | Dress/shirt/toptee iteration-0 present; `VERIFIED_METADATA`, declared SHA each; training noise unknown → eval 0%; all three required | No generated artifact missing; source caches not provenance-validated | `WORKBENCH_LIMN_PYTHON` unset; replay command implemented; old reports are historical; no current validation command/revision established | `BLOCKED_ENVIRONMENT`; verify dedicated interpreter, selected-checkpoint preflight and GPU capability | `aeff57a` (`aeff57ab6dae18e4b195cd7f6e584df952652db1`) |
| `tme` | `https://github.com/li-shuxian/TME` @ `394aad79df2c104feb32a2c96132582012765062`; no patch; clean pin | `fashioniq_0` Kaggle checkpoint present, metadata verified, publisher SHA unknown; `cirr_0` absent; training 0% → eval 0%; 20% `CHECKPOINT_UNAVAILABLE` | EVA-ViT-g and BLIP-2 both required and absent; acquire via `python -m workbench.scripts.download_auxiliary_assets --model tme`; `HF_HOME` tokenizer; CUDA feature cache absent (~10.47 GiB) | `WORKBENCH_TME_PYTHON` unset; no hardware/cache-prep test; replay command implemented; no real validation established | `BLOCKED_ASSET`, `BLOCKED_PREPARATION`, `BLOCKED_ENVIRONMENT`; acquire both assets, provision runtime, authorize CUDA preparation; never substitute 20% | `aeff57a` (`aeff57ab6dae18e4b195cd7f6e584df952652db1`)

`workbench/tests/test_handoff_commands.py` was run against docs, not as model validation. The per-model preparation summary comes from `prepare_model.py --list --json` on this host; its exit 1 means expected reported blockers. Source pins were also compared to live checkouts; checkpoint local presence was checked under configured roots. Missing publisher digest, training dataset, or training-noise metadata remains unknown rather than inferred.

| | |
|---|---|
| Cập nhật | 2026-10-19 |
| Branch | `research/cir-failure-workbench` |
| Source SHA đã kiểm | `e46ad65aa792b01461cc5319b2b8511b4714df41` + các sửa của lượt R10 (ghi ở `workbench/docs/reviews/r10/R10_REMEDIATION.md`) |
| Môi trường kiểm | laptop: Linux x64, `.venv` Python 3.13 + torch CPU, RTX 3050 4 GiB; **không có checkpoint nào staged** |
| Phạm vi đã chạy | static + fixture + parser/unit + native constructor + native LAVIS processor + **native DCNet ConfigParser** + venv thật. **Không có inference nào chạy.** |
| Lượt R9 | 4 lỗi tái hiện được ở `914703e` đã sửa (F08/F09/F04/F05), F06 áp dụng |
| Lượt R10 | 6 finding độc lập đã tái hiện và sửa: R10-01/02 (DCNet binding + staging), R10-03 (lifecycle processor→checkpoint), R10-04 (source_blocker lỗi thời), R10-05 (profile dependency + scanner đọc sai chương trình), R10-06 (venv cô lập mượn package của base) |

---

## 1. Quy tắc đọc trạng thái

Các tầng **độc lập**; một tầng đạt không suy ra tầng sau:

| Tầng | Nghĩa | Cách chứng minh |
|---|---|---|
| `SOURCE_READY` | source vendored, pin + digest manifest khớp, không nested `.git` | `test_fresh_clone_delivery.py`, `test_vendored_source_integrity.py` |
| `ENTRYPOINT_EXISTS` | có file entrypoint thật (CLI hoặc wrapper) | đọc source, `--help` chạy được |
| `PREFLIGHT_PASS` | preflight chạy và **báo blocker đúng** trên input thiếu | test per-model + `test_component_boundaries.py` |
| `PREFLIGHT_IMPLEMENTED_BLOCKED` | có đường preflight thật nhưng model **đang** bị chặn đúng, nên chưa lần nào chạy hết | test preflight + blocker list hiện tại |
| `CHECKPOINT_MAPPING_VERIFIED` | file/biến thể được bind bởi metadata tác giả (revision + size + hash) | HF LFS oid, digest archive |
| `CHECKPOINT_LOADED` | checkpoint thật đã `load_state_dict`/unpickle thành công | **NOT_CHECKED toàn bộ** — không có file local |
| `CODE_TESTED` | đường code (adapter→argparse→serializer→parser) có test hồi quy | pytest |
| `GPU_SMOKE_VERIFIED` | 1 batch forward trên GPU thật | **NOT_CHECKED** |
| `FULL_VAL_VERIFIED` | đủ dress/shirt/toptee + aggregate đúng protocol | **NOT_CHECKED** |
| `AGGREGATE_PARSED` | parser đọc được output native thật | test parser với output shape thật |
| `PER_QUERY_EXPORT_VERIFIED` | có exporter + parity so với native | chỉ `limn` |
| `PAPER_PARITY_CHECKED` | đã so với số paper | **NOT_CHECKED** |

**Quy tắc:**

- Chưa kiểm ⇒ ghi `NOT_CHECKED`/`UNKNOWN`, không suy ra pass/fail.
- **Không dùng một chữ `COMPLETE`.** Một model có thể sạch bug vẫn `BLOCKED` vì weights.
- Phân biệt: không có checkpoint local ≠ mapping chưa rõ ≠ hash chưa kiểm ≠ load thất bại ≠
  thiếu parser ≠ chưa chạy inference ≠ tác giả không công bố weights.
- Preflight-only **không** chuyển thành inference-ready bằng cách đổi tên status.
- Smoke 1 category **không** phải full benchmark; full val cần đủ 3 category.

**3 protocol không được trộn:**

| protocol | gallery | reference |
|---|---|---|
| `fashioniq_original_split` | full split | **eligible** |
| `fashioniq_full_gallery_ref_excluded` | full split | excluded |
| `fashioniq_val_split` | pair-union (ordered) | excluded |

Metric schema là **%**; native có thể trả fraction (DCNet) — đổi đơn vị đúng **một lần**. Macro giữ
quy ước native đã xác minh (mean không trọng số theo category), không query-weighted.

---

## 2. Bảng tổng quan 17 model

Mức kiểm chứng cao nhất thực tế đạt được (không phải mức mong đợi):

| # | model_id | upstream | protocol native | checkpoint variant / noise | source+patch integrity | ckpt/data/env readiness | replay capability | parser | per-query | mức thực tế | blocker |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `limn` | iLearn-Lab/TPAMI24-LIMN | `fashioniq_val_split` | 3 ckpt category, clean eval, training noise unstated | vendored, pin OK | **3 ckpt staged, size+sha256 VERIFIED**; env provisioned (torch 1.12.1+cu113) | native wrapper | `limn_structured_json` | **có** exporter+observer+publish, parity 0 mismatch | **`FULL_VAL_VERIFIED` + `PAPER_PARITY_CHECKED`** | — (blocker engineering đã đóng) |
| 2 | `encoder` | iLearn-Lab/AAAI25-ENCODER | `fashioniq_val_split` | `fashioniq.pt` @rev `dd9b0865` | vendored + 2 local fix | **ckpt staged, size 631 879 365 + sha256 `0f31c501…` VERIFIED**; LAION aux staged (605 219 813 B); env `workbench/envs/encoder` đã provision, CUDA chạy được; wrapper load được ckpt (2 key thừa đã biện luận) và **forward pass thật trên GPU OK** | wrapper | shared replay | **có** (`ENCODER_QUERY_OBSERVER_V1`) | `CHECKPOINT_MAPPING_VERIFIED` → **`CHECKPOINT_LOADED`** → **`GPU_SMOKE_VERIFIED`** → **`FULL_VAL_VERIFIED`** | `PAPER_PARITY_CHECKED` chưa chạy; **R@10 56.076 vs paper 56.13**, **R@50 77.900 vs paper 77.59** |
| 3 | `hint` | iLearn-Lab/ICASSP26-HINT | `fashioniq_val_split` | `fashioniq.pt` 2 724 451 182 B | vendored, pin OK | size/hash VERIFIED; chưa stage | wrapper + cache isolation | `hint_native_recall` | không | `AGGREGATE_PARSED` | `PUBLIC_ASSET_SETUP` |
| 4 | `pair` | iLearn-Lab/ICASSP25-PAIR | `fashioniq_val_split` | `pair-B1.pt`/`pair-B2.pt` | vendored + 1 local fix + `data/datasets.py` recovered | mapping **UNRESOLVED** | wrapper | shared replay | không | `CODE_TESTED` | `AUTHOR_ONLY_INFORMATION` + `SCIENTIFIC_VARIANT` |
| 5 | `clvc_net` | iLearn-Lab/SIGIR21-CLVC-Net | `fashioniq_val_split` | 6 whole models (2/category) | vendored + torch.hub fix | archive `CLVCNET.zip` interior chưa xác minh | wrapper | shared replay | không | `CODE_TESTED` | `AUTHOR_ONLY_INFORMATION` |
| 6 | `tgcir` | iLearn-Lab/MM23-TG-CIR | `fashioniq_val_split` | `TG-CIR.zip` (Drive) | vendored, pin OK | archive member/hash chưa xác minh; pin clip đã sửa | **preflight-only** | không | không | `PREFLIGHT_IMPLEMENTED_BLOCKED` | `AUTHOR_ONLY_INFORMATION` |
| 7 | `habit` | iLearn-Lab/AAAI26-HABIT | `fashioniq_full_gallery_ref_excluded` | **`fiq_n02` (train noise 20%, eval noise 0%)** được chọn | vendored, pin OK | hash VERIFIED; chưa stage; **không có bản clean** | wrapper | shared replay | không | `CODE_TESTED` | `PUBLIC_ASSET_SETUP` (còn thiếu file) |
| 8 | `intent` | iLearn-Lab/AAAI26-INTENT | `fashioniq_full_gallery_ref_excluded` | **`fiq_n02` (train noise 20%, eval noise 0%)** được chọn | vendored, pin OK | hash VERIFIED; chưa stage | wrapper | shared replay | không | `CODE_TESTED` | `PUBLIC_ASSET_SETUP` (còn thiếu file) |
| 9 | `conesep` | Lee-zixu/ConeSep (redirect) | `fashioniq_full_gallery_ref_excluded` | **`fiq_n02` (train noise 20%, eval noise 0%)** được chọn | vendored, pin OK | hash **chưa có** (`expected_sha256: null` trong registry); chưa stage | wrapper | shared replay | không | `CODE_TESTED` | `PUBLIC_ASSET_SETUP` |
| 10 | `airknow` | iLearn-Lab/CVPR26-Air-Know | `fashioniq_full_gallery_ref_excluded` | 0.5/0.8 — **hoãn**: không có checkpoint 20% đã verify, và **không** thay bằng 0.5 để lấp | vendored + 2 local fix | hash **chưa có** (`expected_sha256: null` trong registry); chưa stage | wrapper | shared replay | không | `CODE_TESTED` | `PUBLIC_ASSET_SETUP` + `SCIENTIFIC_VARIANT` (train mode) |
| 11 | `dcnet` | ozmig77/dcnet | `fashioniq_full_gallery_ref_excluded` | `trained_model.pth` (Drive) | vendored, pin OK | GloVe pkl chưa công bố; **data_dir binding kiểm trên cả launch path** | preflight + CLI | `dcnet_native_logger` (đọc thứ tự category) | không | `AGGREGATE_PARSED` | `AUTHOR_ONLY_INFORMATION` |
| 12 | `csmcir` | qzp2018/CSMCIR | `fashioniq_original_split` | `fashioniq_tuned_clip_best.pt` 2 720 615 859 B | vendored + **2 approved patches (cả hai bắt buộc)** | ckpt hash VERIFIED; CoT captions đã có; chưa stage | script CLI (`__main__`) | `csmcir_stdout_json` | không | `CODE_TESTED` | `PUBLIC_ASSET_SETUP` |
| 13 | `combiner_rn50x4_noft` | ABaldrati/CLIP4Cir | `fashioniq_original_split` | RN50x4 noft Combiner | vendored, pin OK | archive 5 parts, digest công bố, **chưa tải** | script CLI | `combiner_stdout_recall` | không | `SOURCE_READY` | `PUBLIC_ASSET_SETUP` |
| 14 | `clip4cir_rn50x4_fullft` | ABaldrati/CLIP4Cir | `fashioniq_original_split` | fullft CLIP + Combiner (cùng config) | vendored, pin OK | như trên; cần **cặp** khớp | script CLI | `combiner_stdout_recall` | không | `SOURCE_READY` | `PUBLIC_ASSET_SETUP` |
| 15 | `sprc` | chunmeifeng/SPRC | `fashioniq_original_split` | ViT-g (paper) / ViT-L (variant) | vendored, pin OK | ViT-g OneDrive 403; ViT-L reachable | script CLI | `csmcir_stdout_json` | không | `CODE_TESTED` | `PUBLIC_ASSET_SETUP` + `SCIENTIFIC_VARIANT` |
| 16 | `ptha_mtst` | zyinan99/MTST_PTHA | `fashioniq_original_split` | `cirr_transfer` (CIRR→FashionIQ, exploratory) | vendored + 3 local fix | ckpt **đã tải + hash khớp** `1c4dab86…`; env **đã khai** từ `requirements.txt` | wrapper + transfer opt-in | không | không | `PREFLIGHT_PASS` (transfer) | cần env py3.9/torch2.0.1 (~5 GB) |
| 17 | `tme` | li-shuxian/TME | `fashioniq_original_split` | `fashioniq_0` (noise 0%); **20% `CHECKPOINT_UNAVAILABLE`** | vendored @ `394aad79` (379 files) | ckpt **đã tải + trích member** `fashioniq_0_best_model.pth` 2 728 382 104 B | wrapper `replay/tme.py` | `tme_native_aggregate` | không | `ENVIRONMENT_VERIFIED` | cần ≥10.47 GiB feature cache + EVA-ViT-g 2 025 249 237 B |


> **Ảnh hưởng của R14** (21/10/2026): R14 chạy **thật** LIMN end-to-end và sửa **ba** lỗi chặn đường
> chạy, tất cả ở tầng hạ tầng — không đổi ranking, preprocessing, reference policy hay semantics
> khoa học của model nào. (1) `pc.sh` truyền `-T` cho `docker exec` trực tiếp; CLI thật từ chối
> (`unknown shorthand flag: 'T'`, exit 125), nên nhánh preparation (mọi model **trừ** `limn`/`csmcir`)
> và **verifier dùng chung** không bao giờ chạy được. (2) `runtime._exports_python_init` chạy con
> `dlopen` bằng `sys.executable` (Python của orchestrator); extension CPython gắn cứng ABI, nên
> `_C.cpython-310-*.so` báo `undefined symbol` và bị coi là **không load được** — khiến cổng GPU
> `capability` từ chối **mọi** model có interpreter riêng, và lỗi hiện ra như "CUDA unavailable".
> (3) `pinned_revision` chạy `git rev-parse HEAD` trong checkout đã bị gỡ `.git` lồng, nên Git đi
> ngược lên repo workbench và trả về **commit của workbench**; completion proof so nó với
> `upstream_commit_sha` của model rồi từ chối — không model vendored nào tạo được proof hợp lệ.
> Cả ba nay đã sửa và có regression bắt được khi revert.
>
> LIMN là model **đầu tiên** đạt `CHECKPOINT_LOADED`, `GPU_SMOKE_VERIFIED`, `FULL_VAL_VERIFIED` và
> `PAPER_PARITY_CHECKED`. `encoder` đã lên tới `CHECKPOINT_LOADED` và `GPU_SMOKE_VERIFIED`:
> checkpoint tác giả load qua đúng gate và **forward pass thật trên GPU** chạy được (R16). Nó
> **chưa** đạt `FULL_VAL_VERIFIED` vì constructor native cần ~12 GiB đĩa trong khi máy còn 3.3 GiB.
> `hint` vẫn ở mức cũ. Cả hai **không** khai `per_query_observer`, nên chưa có per-query để so
> cross-model — đó là hạn chế riêng, không phải cùng blocker với dung lượng.

> **Ảnh hưởng của R13** (21/10/2026): R13 sửa **năm** lỗi, tất cả ở tầng dependency/verifier, không đổi mức
> khoa học của model nào. (1) Verifier từ chối **nguồn đúng** vì khoảng trắng hợp lệ trước `git+` bị dính
> vào URL; (2) cùng lúc đó `runtime` **bỏ** `Requirement.url`, nên PyPI `clip==0.2.0` thoả một khai báo
> OpenAI CLIP và được `RUNTIME_READY` rồi được **reuse**; (3) hai entry CLIP4Cir vẫn khai `clip` trần;
> (4) `pip check` fail với stdout rỗng (interpreter không có pip) vẫn được coi là pass; (5) cùng một
> direct reference khai hai lần bị coi là hai nguồn khác nhau. Thêm hai giả thuyết đã kiểm và đóng:
> venv hợp lệ bị từ chối khi `bin/python` là symlink tới binary nền (nay so **environment identity**
> = `sys.prefix`, không so đường dẫn binary), và nguồn bị chặn không thể mượn readiness từ record GPU
> cũ (cache khoá theo contract digest, có `packages`).
> R13-02 chọn revision `a1d0717` cho CLIP4Cir: README upstream **không pin revision**, nên đây là
> **adaptation được ghi rõ**, không phải pin của tác giả.

> **Ảnh hưởng của R12** (20/10/2026): R12 sửa hai lỗi contract **không** đổi mức khoa học của model nào.
> (1) `scikit-learn` bị loại khỏi plan của `habit`, `intent`, `conesep`, `airknow` vì một census sai —
> census import `lavis.models`, import này **vỡ giữa đường** khi thiếu dependency, nên mọi module cần
> sau điểm vỡ đều không được đếm. Bốn class file native import `sklearn` ở module scope không guard,
> và INTENT import thêm `matplotlib.pyplot`; nay chúng được khai và plan cài chúng.
> (2) `clip @ git+https://github.com/openai/CLIP.git@a1d071...` của `tgcir` và `sprc` bị biến thành
> `clip` trần, nên installer cài `clip==0.2.0` — một package clipboard không có `load`/`tokenize`
> mà evaluator gọi. Nay source được giữ nguyên suốt parse → merge → grouping → lệnh cài → verify.
> Cả hai đều là lỗi **engineering**; không model nào được nâng lên `CHECKPOINT_LOADED`,
> `GPU_SMOKE_VERIFIED` hay `PAPER_PARITY_CHECKED`, vì mọi blocker còn lại vẫn là asset tác giả/GPU.

> **Ảnh hưởng của R11 lên bảng này** (20/10/2026): R11 sửa contract *môi trường* của `hint`, `habit`,
> `conesep`, `airknow` (Python 3.8 không cài được `pandas==2.2.0`/`scikit-image==0.22.0`) và của
> `sprc` (planner và verifier đọc hai nguồn version khác nhau). Các mức trong bảng **không** thay đổi
> vì chúng vẫn bị chặn bởi asset tác giả, không phải bởi environment: sửa R11 làm cho bước
> provisioning *có thể thành công*, chứ không tự sinh ra checkpoint hay GPU. Riêng cột
> `ckpt/data/env readiness` của 5 model LAVIS nay đúng hơn ở tầng dependency: profile đã cài được,
> và đường processor → checkpoint của cả 5 wrapper đã có bằng chứng chạy thật trong hai tiến trình
> riêng (`workbench/tests/test_r11_lavis_lifecycle.py`) — nhưng đó là **fixture**, không phải
> weight tác giả. Không model nào được nâng lên `CHECKPOINT_LOADED` hay `GPU_SMOKE_VERIFIED`.

**`limn` đạt `CHECKPOINT_LOADED`, `GPU_SMOKE_VERIFIED`, `FULL_VAL_VERIFIED` và
`PAPER_PARITY_CHECKED`** trên PC này (số đo ở §3.1). **15 model còn lại** chưa đạt mức nào trong bốn
mức đó: mọi số trong registry cho chúng vẫn là số paper, không phải số đo.

---

## 3. Chi tiết theo model

### 1. `limn`
- Upstream `iLearn-Lab/TPAMI24-LIMN` @ `7d7bc9b116f5`. Wrapper `workbench/replay/limn.py` (native
  `test.py:test()`), exporter `limn_export.py`, observer `limn_observer.py`, publish `limn_publish.py`.
- Checkpoint: 3 whole-model `0_{dress,shirt,toptee}_best_model.pt`, Hub LFS sha256 đã ghi trong registry.
  Training noise chưa nêu trong paper ⇒ `null` (không gán 0). Evaluation noise 0.
- Dataset: `fashioniq_ilearn_resized_training`, `.jpg`. Trên PC này dùng `data/fashionIQ_dataset`
  (`FashionIQ` là symlink cùng thư mục): `resized_image/{dress,shirt,toptee}` =
  3817 / 6346 / 5373 ảnh `.jpg`, khớp đúng layout native yêu cầu.
- Protocol `fashioniq_val_split`; ranking native `np.argsort(-sims)` + mask reference `-10e10`.
  Cache isolation: có. Per-query: **có**, parity so với native aggregate.
- **ĐÃ CHẠY THẬT (21/10/2026)** — checkpoint tác giả, dữ liệu tác giả, GPU thật (RTX 3050, sm_86):

| hạng mục | kết quả đo |
|---|---|
| checkpoint integrity | cả 3 file: size **OK**, sha256 **OK** |
| checkpoint load | `model.LIMN`, 440.26 M tham số, `cuda:0` |
| GPU probe | `cuda_available=true`, `computed=true`, cc **8.6** |
| **dress** | R1 **18.6911** · R10 **50.7685** · R50 **74.0208** |
| **shirt** | R1 **25.5152** · R10 **55.5447** · R50 **76.7910** |
| **toptee** | R1 **26.2111** · R10 **60.5813** · R50 **81.6930** |
| **macro** | R10 **55.6315** · R50 **77.5016** · mean **66.5665** |
| paper (làm tham chiếu) | R10 55.91 · R50 77.82 · mean 66.87 |

  Độ lệch so với paper: R10 **−0.28**, R50 **−0.32**, mean **−0.30** điểm. Sai số này nằm trong
  khoảng của một lần chạy lại hợp lệ (không cắt tỉa, không đổi ranking, không đổi reference policy).
- Per-query: exporter chạy thật trên cả 3 category, **6016 query**, `top_k=50`,
  `INSTRUMENTATION_PARITY_VERIFIED`, `instrumentation_mismatches: {}` — tức ranking do observer
  dựng lại **khớp tuyệt đối** với aggregate native. Artifact canonical đã publish:
  `workbench/artifacts/results/fashioniq_val_split/limn/20261021T040000000000Z_limn_base_iter0_all_categories_fashioniq_val_split.json`
- Hạ tầng: index rebuild → `/api/runs` trả run `data_kind: experiment` → `/api/queries` trả
  `total: 6016` kèm `top_results` → UI Sample Explorer hiển thị `1–20 of 6016` với ảnh thật.
- `WORKBENCH_LIMN_BATCH_SIZE` (mặc định 32 = thiết lập paper) là knob vận hành cho GPU nhỏ; nó chỉ
  đổi số ảnh mỗi lượt forward, không đổi thứ hạng (`test()` concat rồi chuẩn hoá sau). Lượt đo này
  dùng 4; provenance ghi rõ `batch_size_source: operator override`.
- **Còn thiếu để hoàn tất khoa học**: xác nhận trên GPU đích `sm_120` (RTX 5070 Ti). Lượt này chạy
  profile **historical pins** trên `sm_86`; profile `adapted_runtime: sm_120` vẫn chưa được đo.

### 2. `encoder`
- Upstream @ `29a2a31d6a56`. Local fix: `datasets1.py` + `import datasets as datasets1`.
- Checkpoint `fashioniq.pt`: **đã verify độc lập** — Hub revision `dd9b086569fe69bd9111b6114546201c97934b01`,
  size 631 879 365, LFS oid `0f31c50150a7bda3495e3378df13b4cec7924b52efd23b5e5dd9731459da9173`;
  Range read trả `PK..fashioniq/data.pkl` + `OrderedDict` bắt đầu `loss_T`.
- Loader: giữ `strict=False` (upstream) nhưng **không bỏ qua report** — missing parameter, lệch
  shape, và key lạ đều ⇒ raise. Hai key ngoại lệ của artifact tác giả được biện luận ở mục dưới;
  tình trạng "backbone + ckpt chưa stage" ghi ở đây **đã hết hiệu lực** (xem mục "Trạng thái trên
  máy này").
- Đã chạy: `test_encoder_replay.py` (5), `test_verified_declarations.py` (F10).
- **Trạng thái trên máy này (đo lại 2026-10-22).** Ba thứ đã được provision và kiểm chứng:
  checkpoint `fashioniq.pt` **đã tải và hash khớp** (631 879 365 B, sha256 `0f31c50150a7bda3495e3
  378df13b4cec7924b52efd23b5e5dd9731459da9173`); backbone LAION ViT-B/32 **đã tải** (605 219 813 B)
  vào `workbench/third_party/ENCODER/open_clip_pytorch_model.bin`; và một interpreter riêng
  `workbench/envs/encoder` (Python 3.9.25, `torch 1.13.1+cu117`, `torchvision 0.14.1+cu117`,
  `open-clip-torch 2.24.0`) chạy được CUDA trên RTX 3050 (`torch.cuda.is_available() == True`, phép
  nhân tensor trên `cuda:0` cho kết quả đúng). Preflight của wrapper (`--preflight-only`) trả về 0.
- **Blocker còn lại, đã đo chứ không suy đoán — và nó không phải GPU:**

  1. ~~**Không đủ dung lượng đĩa.**~~ **Đã chạy thật (2026-10-23).** Blocker cũ là **thật** nhưng
     nguyên nhân không phải dung lượng đĩa: `isolated_cache_root` dùng `tempfile`, nên cache
     ~11.9 GiB rơi vào `TMPDIR`. Với `TMPDIR=/tmp` (tmpfs RAM 7.2 GB) thì constructor chết đúng
     `datasets.py:83` với `OSError: [Errno 122]`. Trỏ `TMPDIR` sang ext4 đã gỡ được: cả ba category
     chạy trên GPU thật —
     **dress R@1 21.021 / R@10 51.760 / R@50 76.351**, **shirt 26.153 / 55.005 / 75.319**,
     **toptee 29.373 / 61.703 / 81.132**, **macro R@10 56.076 / R@50 77.900 / mean 66.988**.
     So với dòng paper của registry (`reported_r10 56.13`, `reported_r50 77.59`): **lệch trong 5%**.
     `parity_mismatches` rỗng ở cả ba category; `checkpoint_sha256` khớp registry.
     Cần **≥12 GiB trống** trên volume chứa `TMPDIR` (đỉnh đo được: 11.868 GB), và bộ nhớ đủ lớn
     vì constructor giữ cả ba category trong RAM trước khi ghi.
  2. ~~**Không đủ dung lượng đĩa (bản cũ).**~~ Constructor native (`datasets.py:74-88`) dựng và pickle **cả ba
     category ngay từ đầu**, trước mọi inference, ghi thẳng vào dataset root; wrapper chỉ có thể
     giảm nhẹ bằng `isolated_cache_root`.
     Trọng tải mỗi tensor là `3x224x224x4` = **602 112 B** (đo bằng chính transform đã pin:
     `open_clip.create_model_and_transforms('ViT-B-32')`, cả stage 0 và stage 1 đều ra `(3,224,224)`
     float32). `get_test_data` lưu **2 ảnh/query** (source + target) cộng **1 tensor cho mỗi chỉ số
     gallery phân biệt** mà hợp của các cặp nêu ra (không phải toàn bộ gallery):

     | category | query | gallery phân biệt | tensor payload |
     |---|---|---|---|
     | dress | 2 017 | 2 628 | 3.74 GiB |
     | shirt | 2 038 | 3 089 | 4.02 GiB |
     | toptee | 1 961 | 2 902 | 3.83 GiB |
     | **tổng** | | | **~11.6 GiB** |

     Đây là **payload tính từ annotation**, không phải kích thước pickle đo trên đĩa; overhead của
     pickle chưa được đo. Máy này còn **3.3 GiB**. Đã chạy thử hai lần và cả hai chết đúng chỗ đó
     (`OSError: [Errno 122] Disk quota exceeded` / `[Errno 28]`, tại `datasets.py:85` khi đang pickle
     `shirt`). Chặn **thật**, và **có thể gỡ** bằng dung lượng: cần **≥12 GiB trống** trên volume chứa
     `WORKBENCH_ARTIFACTS_ROOT` (khuyến nghị ≥16 GiB để có biên an toàn), không phải giới hạn của code.
     Số **15.46 GiB** ghi ở bản trước là **sai**: nó tính trên toàn bộ gallery thay vì hợp các cặp.
     **Kích thước pickle thật đã đo được** (đỉnh trong `TMPDIR` khi chạy ba category):
     `test_queries_dress.pkl` 2 317.8 MB, `test_queries_shirt.pkl` 2 341.9 MB,
     `test_targets_dress.pkl` 1 509.8 MB — tổng đỉnh **11.868 GB**. Con số tính từ annotation
     (~11.6 GiB) khớp với đo thực tế, nên ước lượng cũ là đúng; chỉ **nguyên nhân** thì sai.

  2. **Checkpoint lệch với source đã pin — ĐÃ GỠ (R16).** File tác giả chứa thêm hai tensor
     `backbone.binding_decoder.fc.0.{weight,bias}` mà `BindingDecoder` của revision đã pin
     (`29a2a31d6a56` = HEAD của upstream) **không** khai báo. Căn cứ để cho qua là về **artifact và
     source**, không phải về điểm số: `BindingDecoder.__init__` **không hề** tạo thuộc tính `fc`
     (`model_try2.py:89-116`), và quét **toàn bộ** checkout pin không thấy chỗ nào đọc
     `binding_decoder.fc`; `missing_keys` **rỗng** (mọi tham số model cần đều đã nạp), 415/417 tensor
     khớp shape với tham số, và **đổi giá trị** hai tensor đó khiến output của `binding()` **giống hệt**
     (chênh lệch tuyệt đối lớn nhất `0.0`). Ngoại lệ trong `_unused_checkpoint_keys`
     (`workbench/replay/encoder.py`) chỉ mở khi **đủ bốn điều kiện**: đúng digest đã đăng ký của
     artifact này, mapping của variant là `VERIFIED_METADATA`, source pin sạch và đúng commit, và
     module **không** khai hai tên đó. Thiếu tham số, lệch shape, hay bất kỳ key lạ nào khác **vẫn
     bị từ chối**; hai key bị bỏ qua được **in ra stderr**, không im lặng.
     Đã kiểm bằng checkpoint thật: load qua đúng gate → ACCEPTED kèm log; đổi digest → REFUSED; thêm
     key lạ → REFUSED. Và **forward pass thật trên GPU** với checkpoint tác giả chạy được: output
     `(2, 512)`, hữu hạn, chuẩn hoá, không suy biến. Bằng chứng đầy đủ:
     `workbench/docs/reviews/R16_ENCODER_KEY_COMPAT.md`.
  3. **Chưa chạy được native evaluation đầy đủ** vì chặn #1 (đĩa). Đây là **blocker của native evaluation đầy đủ**
     (việc còn lại của ENCODER còn có mục 4 bên dưới). Không có đường vòng: constructor native dựng cả ba category trước inference.
  4. **per-query integration đã có** (`workbench/replay/encoder_export.py` +
     `encoder_publish.py` + `observers`), và registry đã khai `per_query_observer:
     ENCODER_QUERY_OBSERVER_V1` cùng `per_query_checkpoint: fashioniq`.
     Trước khi sửa, observer **được gọi** nhưng **bị OOM-kill (`observer exited -9`)** vì tiến
     trình con đó chạy **không có `TMPDIR` riêng**, nên cache ~11.9 GiB của nó rơi vào `/tmp`
     (tmpfs). Bằng chứng để lại: `/tmp/cir-replay-cache-*` **5.3 GB** sau khi tiến trình bị kill.
     Sau khi cấp `TMPDIR` riêng theo run, đã chạy lại và **export hoàn tất**:
     `per_query_state: PER_QUERY_EXPORTED_UNVERIFIED`, `per_query_export_available: true`,
     `instrumentation_parity: INSTRUMENTATION_PARITY_VERIFIED`, artifact canonical
     **144 470 712 B** tại `artifacts/results/fashioniq_val_split/encoder/`.
     `verify_run.py` trả **`[OK] … is corroborated`**, và index rebuild đưa run vào DuckDB (10 run),
     đọc được qua `/api/runs`.
     Trạng thái tách bạch: **export đã chạy thật**; `PER_QUERY_VERIFIED` vẫn thuộc về verifier và
     gate paper vẫn dừng ở `PASS_METRICS_ONLY_MAPPING_UNVERIFIED` vì dòng reference của ENCODER
     trong `paper_benchmarks/papers.yaml` ghi `checkpoint_mapping_status: UNVERIFIED` (14/14 dòng
     paper đều vậy — mặc định fail-closed có chủ ý, không phải lỗi ở lượt này).
  5. **Artifact per-query của lượt đó KHÔNG tái lập được aggregate của chính nó** — 30 hit-count
     lệch trên sáu giá trị Recall (dress R@50: artifact 76.2023 vs report 76.5493), dù hai bên đều
     ghi `INSTRUMENTATION_PARITY_VERIFIED`.
     **Nguyên nhân (đã xác định, thay cho giải thích cũ):** constructor iLearn **nướng tensor query
     vào dataset lúc dựng** qua `get_img(..., stage=0)`, và replay truyền
     `[preprocess_train, preprocess_val]` ⇒ query đi qua `RandomResizedCrop` (stochastic) còn gallery
     dùng `stage=1`. Hai lần dựng dress cho **3994/4034** tensor query khác nhau (gallery giống hệt).
     Giải thích cũ — "cuDNN không ghim deterministic" — là **SAI và đã bị bỏ**: normalization, dot
     product và `argsort` đều chạy bằng NumPy sau khi feature về CPU, và check nội bộ của observer
     pass vì nó so **hai lượt trên cùng một cache cục bộ**. Xem D11.
     Đã sửa bằng `common.seeded_dataset_construction` (seed trước khi dựng, khôi phục RNG state) ở
     cả bốn nơi dựng dataset; đo lại **0/4034** khác.
     Đã thêm check ở **parent** (nơi duy nhất có thể): artifact phải tái lập `category_metrics` của
     lượt đó, so theo **số hit** chứ không theo dung sai, và artifact bị **xoá** khi từ chối (nếu để
     lại, một artifact trông hợp lệ sẽ nằm cạnh report nói export thất bại). Đối chứng: check này
     **từ chối** artifact ENCODER cũ (6 mâu thuẫn) và **chấp nhận** artifact LIMN (0 mâu thuẫn).
     Vì vậy **không** đọc lượng export của ENCODER như bằng chứng per-query đã kiểm chứng.
  6. **Chênh lệch per-query còn lại: ĐÃ XÁC ĐỊNH, và không phải vấn đề dung sai.** Sau khi sửa
     thứ tự tie cho khớp native, parity nội bộ của observer là
     `INSTRUMENTATION_PARITY_VERIFIED` với **0 mismatch**. Nhưng parent vẫn từ chối, và đúng:
     - Native metric là `np.argsort(-sims)`. Observer từng sắp tie theo cách khác ⇒ 8 dress / 2 shirt /
       4 toptee query có điểm target **bằng đúng** ứng viên top-1, và R@1 lệch đúng 1 hit mỗi category.
       Đã sửa: dùng đúng lời gọi native.
     - **Một forward pass thứ hai KHÔNG tái lập được pass thứ nhất** — đo được, không phải suy đoán:
       hai tiến trình, cùng seed, cùng input, cùng config, cuDNN đã ghim deterministic và
       `CUBLAS_WORKSPACE_CONFIG` đã set, vẫn cho **feature hash khác nhau** (`82dc55fc…` vs
       `af784fa2…`). Nên `test()` ở tiến trình aggregate và ở tiến trình observer tính ra metric khác
       nhau 1–2 hit.
     - `FeatureRecorder` (mới) cho phép observation dùng **chính tensor** mà native pass đã tính, thay
       vì tính lại. Đây là hướng đúng; chưa nối end-to-end.
     ⇒ **ĐÃ XONG (2026-10-25).** Ba nguyên nhân đã gỡ, mỗi cái đo được:
     (a) seed dựng dataset → 0/4034 tensor query khác;
     (b) aggregate ghi **chính feature** mà lượt đánh giá duy nhất của nó tính ra
     (`WORKBENCH_OBSERVATION_OUT`), export xếp hạng trên đó thay vì chạy lượt thứ hai;
     (c) xếp hạng đúng **dtype float32** và đúng `np.argsort(-sims)` như native — lệch dtype làm
     normalize round khác và R@1 lệch 1 hit ở hai category.
     Kết quả: `per_query_state: PER_QUERY_EXPORTED_UNVERIFIED`, `export_available: true`,
     `INSTRUMENTATION_PARITY_VERIFIED`, artifact **144 469 926 B**.
     **Kiểm độc lập từ artifact**: dress 1040/1534, shirt 1118/1535, toptee 1207/1590 — **6/6 hit
     count bằng đúng aggregate**. `verify_run.py` → corroborated; index có run; so LIMN–ENCODER
     trên cùng query trả về **cả hai** model. LIMN cũ vẫn tái lập aggregate của nó.
  7. **Blocker hiện tại của lượt full 3-category: dung lượng đĩa.** Một lần dựng ba category cần
     **11 868 MB** (đo được ở đỉnh), máy còn **11 456 MB** ⇒ thiếu **~412 MB**. Hai lượt chạy thật đã
     chết đúng chỗ này (`OSError: [Errno 28] No space left on device` tại `datasets.py`). Sau khi sửa
     D12, lượt hỏng **không còn để lại cache** (log dir 3 MB thay vì 10.4 GB), nên nó không còn tự
     làm nặng thêm. Cần giải phóng thêm ~1 GB trên volume chứa `TMPDIR` — xem §6.

### 3. `hint`
- Upstream @ `bec50b6c8c19`. Wrapper `hint.py` chạy trong `isolated_cache_root` (F03/B3):
  `fashion_iq_data.json` + `test_queries_*.pkl` là tensor **đã transform**, nên root tạm mới là bắt buộc.
- Parser `hint_native_recall` đọc **cả hai** shape thật: JSON của wrapper và print native
  `[('dress_r10', v), ...]` (`test.py:103-108`). Macro tính lại.
- Hub: `iLearn-Lab/HINT` (không phải `ICASSP26-HINT` — trả 401). `fashioniq.pt` 2 724 451 182 B,
  LFS `bf9c72d4…`.
- Đã chạy: `test_hint_replay.py` (4), `test_hint_metric_parser.py` (11),
  `test_r9_defect_regressions.py` (F05 train-inputs + F06 processor).
- **F05 đã sửa (lượt R9)**: `validate_dataset(root, categories=('dress',))` từng trả `[]` trên root
  chỉ có dress, nhưng constructor native (`datasets.py:35-47,59-73`) mở **cả ba** category bất kể
  `--category`, và `isolated_cache_root` luôn rỗng nên nhánh cold luôn chạy ⇒ checkpoint được load
  xong mới chết ở `cap.shirt.train.json`. Nay tập input của constructor là câu hỏi **riêng**, kiểm
  bằng `workbench/replay/common.py:constructor_input_blockers`, dùng chung cho `hint`, `habit`,
  `intent`, `conesep`, `airknow`, `pair`.
- **F06 đã sửa (lượt R9)**: `txt_processors` lấy từ config pinned qua
  `workbench/replay/lavis_processors.py`; không còn dựng model BLIP-2 thứ hai, không cần
  `/root/bert-base-uncased`. `--backbone pretrain_vitL` chọn đúng config ViT-L mà model class công bố.

### 4. `pair`
- Upstream @ `bfb0b98cdb4c`. Local fix: `models/MHTransformer_rev.py` =
  **`EXPERIMENTAL_RECONSTRUCTION`** (đã ghi rõ trong docstring: chỉ khác 2 dòng so với
  `MHTransformer.py` sau khi bỏ class ordinary-attention; `MultiHeadAttentionRev.forward` dùng
  `1 - attention.softmax(dim=2)`), và `data/datasets.py` (file nguồn **bắt buộc** từng bị `.gitignore`
  `data/` nuốt).
- Checkpoint: `pair-B1.pt` / `pair-B2.pt` công bố nhưng **không có nhãn dataset/variant** ⇒ mapping
  `UNVERIFIED`. Không chọn dataset/architecture thay tác giả.
- **R18 — hai checkpoint công bố KHÔNG load được, có bằng chứng.** Cả hai file tải thật và **khớp
  hoàn toàn metadata publisher**: `pair-B1.pt` 694 958 565 B sha256 `c4d861ec…5fbf2`,
  `pair-B2.pt` 694 872 371 B sha256 `74346fd7…eb3f03` (HuggingFace `iLearn-Lab/ICASSP25-PAIR`).
  Nên đây **không** phải lỗi tải hỏng.
  - `torch.load(..., weights_only=True)` dừng ở `GLOBAL model_vssp.Encoder` — pickle tham chiếu
    `model_vssp.Encoder` / `.Backbone` / `.FeatureWiseAffine`; loader an toàn báo
    `ModuleNotFoundError: No module named 'model_vssp'`. Lỗi xảy ra **trước** khi có state_dict, nên
    **không tồn tại** báo cáo key/shape — không được bịa một cái.
  - Kiểm lịch sử upstream: tại pin `bfb0b98c` cây đầy đủ chỉ có `models/MHTransformer.py` +
    `models/model.py`; `commits?path=model_vssp.py`, `...MHTransformer_dis.py`,
    `...MHTransformer_rev.py` đều rỗng; bản phát hành nguồn đầu tiên `5991d554` cũng vậy. ⇒ **không
    phải pin cũ**, mà là module **chưa từng được công bố**.
  - Khác biệt cấu trúc (chỉ nêu cái có bằng chứng): `transformer_c`/`transformer_d`,
    `backbone.weight_chanelImg`, `weight_chanelImg_t`, `cross_t2i` chỉ có trong pickle; `backbone.transformer`
    8 head/1 layer; `_a` pickle 4 head/2 layer vs pinned 8/1; `_b` pickle `MHTransformer_dis` vs pinned
    `MHTransformer_rev` (bản pinned là **reconstruction của workbench**, không phải code tác giả).
    `local_weight=(8,)`/`local_weight_dis=(7,)` **khớp** `local_token_num=6` nên **không** dùng làm
    bằng chứng lệch.
  - Operator path cũng từ chối: `COMMAND_BLOCKED_SOURCE_ENTRYPOINT` + `checkpoint mapping unresolved`.
  - **Kết luận: NOT RUN** cho cả B1 và B2 — không có metric, không GPU, không so dòng paper. Remedy tối
    thiểu: tác giả phải công bố định nghĩa `model_vssp` và `MHTransformer_dis`; **không** alias/ăli
    hay tự dựng lại topology.
- Đã chạy: `test_pair_replay.py` (5). Blocker: `AUTHOR_ONLY_INFORMATION` (R18 xác nhận cụ thể hơn:
  thiếu định nghĩa lớp, không phải thiếu file).
- **F05 đã sửa (lượt R9)**: `data/datasets.py:34-42` dựng cache và đọc cả ba category; validator
  từng chỉ kiểm category được chọn. Nay dùng chung `constructor_input_blockers`.

### 5. `clvc_net`
- Upstream @ `bd9b68894898`. Local fix `resnet.py` dùng `torch.hub.load_state_dict_from_url`
  (fallback torchvision cũ) — **cùng URL ResNet-50**.
- Transform **đã sửa (F03)**: native FashionIQ val = `Resize(256)→CenterCrop(224)→ToTensor→ImageNet
  Normalize` (`train.py:90-96`); wrapper trước dùng `Resize(224)` (nhánh fashion200k) — cùng shape
  nhưng tensor khác tới 4.24 đơn vị chuẩn hoá trên ảnh 300×500.
- Checkpoint: cần **6** whole model (local+global × 3 category). Tên trong wrapper là **giả định**,
  chưa phải member của archive tác giả ⇒ blocker.

### 6. `tgcir`
- Upstream @ `65fa78eaf8ca`. **F07 đã sửa**: pin `clip==0.2.0` (clipboard CLI) → OpenAI CLIP
  revision `a1d07173…` (verify: `clip.py` 200, có `load` + `tokenize`).
- Wrapper `tgcir.py` **preflight-only**, luôn blocked cho tới khi xác minh member/hash của
  `TG-CIR.zip`. Không đổi nhãn thành "implemented".
- Native `test.py:6` = `test(params, model, testset, category)`, gallery pair-union, mask reference
  `-10e10` ⇒ khớp `fashioniq_val_split`.

### 7–9. `habit`, `intent`, `conesep`
- Habit @ `6ddc2cd63eac`, Intent @ `2a75fc6f57d1`, ConeSep @ `bd4bf629ebe6` (redirect từ
  `iLearn-Lab/CVPR26-ConeSep` sang `Lee-zixu/ConeSep`, đã ghi provenance).
- Cả ba publish **chỉ** N0.2 / N0.5 / N0.8. Evaluate trên clean validation là hợp lệ **nhưng phải
  label**: `checkpoint_training_noise_pct` và `evaluation_noise_pct` giữ **tách biệt**; không mô tả
  checkpoint noisy như clean-trained.
- Adapter truyền `--variant <checkpoint_id>` ⇒ biến thể được chọn chính là biến thể load.
- Parser: shared replay (đọc `categories` — F02).
- **F05 + F06 đã sửa (lượt R9)**, cùng cách như `hint`: tập input của constructor native được kiểm
  độc lập với `--variant`/`--category`, và `txt_processors` lấy từ config pinned thay vì dựng model
  BLIP-2 thứ hai (3 model này từng chạm `/root/bert-base-uncased` qua `Blip2Base.init_tokenizer`).

### 10. `airknow`
- Upstream @ `6ab6fa95b746`. Local fix: guard 2 import vắng + `.png` fallback (không đổi hành vi khi
  `.jpg` có mặt).
- **F01 đã sửa**: adapter truyền `--variant` (trước là `--checkpoint-id`, không wrapper nào nhận ⇒
  argparse exit 2 trước khi chạm model). Test ép cả `fiq_n05` và `fiq_n08`.
- Native gọi `model.train()` ở `test_BLIP2.py:10` cho nhánh FashionIQ ⇒ giữ nguyên, ghi là
  `SCIENTIFIC_VARIANT`; `--eval-mode` là biến thể thí nghiệm được label riêng.
- **F05 + F06 đã sửa (lượt R9)**: như `hint`. `--variant` (F01) và `eval_mode` không đổi; chỉ đường
  lấy `txt_processors` và tập input preflight thay đổi.

### 11. `dcnet`
- Upstream @ `68c79d38569f`. CLI thật: `python test.py --resume <run_dir>`.
- **F08 đã sửa (lượt R9)**: evaluator đọc `config.json:data_loader.args.data_dir` và `CE_dataset`
  join literal đó vào **CWD của tiến trình** (không có `chdir` nào trong DCNet; `evaluate_models.py`
  launch với `cwd=source_cwd(...)` = checkout). Bản trước chỉ từ chối *relative* và chỉ nằm trong
  `dcnet.preflight` — một helper mà `DCNetAdapter` **không** gọi, nên official path không được bảo vệ.
  Nay: resolve theo CWD, so với `--dataset-root` cho **cả** absolute lẫn relative, **và** chạy trong
  `backend/preflight.preflight_blockers` — danh sách mà `doctor`, `pc_readiness`, `prepare_model` và
  `evaluate_models` đều đọc. Symlink canonical (dataset stage đúng cách) pass; thư mục thường trong
  checkout hoặc path absolute trỏ nơi khác ⇒ **refuse** kèm đường dẫn cụ thể.
  **ĐÃ BỎ (R10-01, xác nhận lại ở R11):** override `WORKBENCH_DCNET_DATA_DIR`. Evaluator native
  (`ConfigParser`) không đọc biến môi trường, nên override chỉ khiến validator trả lời một câu hỏi
  khác với consumer — tức là nó *gây ra* đúng loại lỗi mà validator tồn tại để chặn. Quyết định
  binding nay chỉ đọc `config.json` của run và CWD của evaluator. Đừng set biến này.
  **R11 bổ sung:** mọi cấp JSON của config nay được kiểm shape trước khi `.get()` — `[]`,
  `{"data_loader": ["x"]}`, `{"data_loader": {"args": null}}` đều trả blocker nêu tên field thay vì
  ném `AttributeError` ra khỏi readiness view.
  Run directory giờ chép `config.json` vào evidence dir để `verify_run` re-parse được stdout.
- **F09 đã sửa (lượt R9)**: parser `dcnet_native_logger`. `model/metric.py:5-21` trả **fraction 0..1**
  ⇒ đổi % đúng một lần. `test.py:74-84` in cả `recall` lẫn `comb` bằng **cùng** vòng
  `trainer.categories` (`base_trainer.py:32` = `config['data_loader']['args']['categories']`); sáu ô
  `comb:` không có tên, nên bản trước map theo tuple cứng `(dress, shirt, toptee)` và **transpose im
  lặng** khi config dùng thứ tự khác (`comb_avg` không phát hiện được vì tổng bất biến với hoán vị).
  Nay: thứ tự lấy từ chính stream (`[cat] comp corr r@10`), config là **nhân chứng thứ hai** — lệch
  nhau ⇒ từ chối; thiếu config ⇒ vẫn đúng, vì stream tự nó xác định thứ tự. composition/correction
  giữ **riêng**.
- Blocker: `cap.{cat}.glove.val.pkl` chưa công bố; OOV embedding chưa seed ⇒ reconstruction
  deterministic là **biến thể tuỳ chọn**, không phải mặc định.

### 12. `csmcir`
- Upstream @ `774f94e2076f`. **2 approved patches** (đều verify được, và edit sau patch bị từ chối):
  1. `src/lavis/models/__init__.py` — 98 import unguarded của module không tồn tại chặn registration.
  2. **`src/validate_blip_csmcir.py` (F04 mới)** — evaluator in metric JSON rồi **vô điều kiện** chạy
     `vis_fiq`, nơi `labels.append(data[1])` là target-id string nhưng bị xử như tensor ⇒
     AttributeError, exit ≠ 0 **sau khi metric đã tính**. Patch đặt dump sau cờ `--visualize` mặc định
     tắt; công cụ vẽ không bị đụng.
- **F04 đã sửa (lượt R9)**: `approved_patch_state` **return ngay trong vòng lặp**, nên chỉ cần patch
  #1 ⇒ `applied=True` trong khi patch #2 chưa áp; và không consumer nào phân biệt partial với complete.
  Nay hàm kiểm **mọi** patch đã khai báo, trả thêm `patches[]` (từng mục một), và `detail` nói rõ
  "*N* of *M* required patches are not applied". Một checkout nửa vời vì vậy bị báo là
  `source repair incomplete` ngay trong `model_requirements`, và `source_blocker` vẫn còn hiệu lực.
  Cả hai patch đều cần cho workflow benchmark mặc định; `--visualize` vẫn **off** khi chạy benchmark.
- CoT captions **có công bố** tại Hub `peng12138/CSMCIR` (`fashioniq_qwen_captions/qwen_captions/`),
  byte-identical với bản local (sha256 từng category đã verify). Blocker "captions unavailable" đã
  **rút lại**.
- Entrypoint `src/validate_blip_csmcir.py` có `__main__` thật. Data root `<checkout>/fashionIQ_dataset`,
  ảnh `.png`. Checkpoint key `Blip2QformerCirAlignPromptCSMCIR`.

### 13–14. `combiner_rn50x4_noft`, `clip4cir_rn50x4_fullft`
- Cùng checkout `CLIP4Cir` @ `dfed9f748a8a`; **hai entry riêng**, không trộn cặp.
- Dims adapter truyền `--projection-dim 2560 --hidden-dim 5120` = **đúng** cho RN50x4
  (`validate.py:269-270`, feature_dim 640 ×4/×8). Ví dụ CIRR trong README (4096/8192) là **typo đã
  được tác giả xác nhận** (issue #9) — không "sửa lại" theo README.
- Release `weights` có 5 part `pretrained_models.tar.gz` kèm sha256 (ghi ở registry/agent report);
  **chưa tải** (9.8 GB trống, archive ~8.55 GB). Member: `FashionIQ/RN50x4_noft/fiq_comb_RN50x4_noft.pt`,
  `FashionIQ/RN50x4_fullft/fiq_comb_RN50x4_fullft.pt`, `FashionIQ/RN50x4_fullft/fiq_clip_RN50x4_fullft.pt`.
- noft = CLIP RN50x4 gốc + Combiner; fullft = **cặp** CLIP fine-tuned + Combiner cùng config.
- Protocol: full gallery, reference **eligible** (`validate.py:43-56`, `127-132`) ⇒ `fashioniq_original_split`.

### 15. `sprc`
- Upstream @ `2935a5397732`. **F14 đã sửa**: `backbone` được **khai báo** theo variant trong registry,
  adapter đọc khai báo đó; selection không khai báo ⇒ **refuse** (không đoán từ filename, vì loader
  dùng `strict=False` sẽ im lặng đánh giá model khởi tạo một phần).
- Variant: `sprc_fiq_unresolved` (`backbone: pretrain` = ViT-g) và `sprc_fiq_vitl`
  (`backbone: pretrain_vitL`, `scientific_variant: true`, có `variant_provenance` + comparability).
  ViT-L reachable (200, 1 459 756 541 B); OneDrive ViT-g 403; Baidu cần mật khẩu.
- Adapter truyền `--blip-model-name blip2_cir_align_prompt` (default pinned
  `blip2_cir_rerank_learn` **không được register** ở đâu).

### 16. `ptha_mtst`
- Upstream @ `6a02ee4763fc`. Local fix: guard 1 import vắng; `base_path` đọc `MTST_DATA_ROOT` với
  literal upstream làm fallback.
- Hub `yinanzhou1/mtst_ptha` publish **chỉ** CIRR (`tuned_mtst_cirr.pt` 2 720 352 973 B,
  `cirr_modifier_generated_5p_completed_rev.json`, `cirr_train_modifier_rev_long.json`) — **không có**
  FashionIQ weight. Không thay bằng CIRR nếu chưa bật transfer tường minh; không random, không tự train.
- Vendored tree contains both `MTST_PTHA/src/blip_validate.py` and `src/validate_blip.py`; the wrapper now
  checks the latter module it imports for `compute_fiq_val_metrics`.
- `validate_blip.py` CIRR-only CLI is not transfer entrypoint. `blip_validate.py --dataset FashionIQ`
  does dispatch `clip_finetune_fiq` (`:186-200`), but that function describes training and loads with
  `strict=False`; wrapper is needed for explicit transfer labels and strict preflight.
- Chỉ chạy transfer khi truyền `--transfer-cirr-to-fashioniq`; nhãn bắt buộc:
  `PTHA+MTST | CIRR checkpoint -> FashionIQ | cross-dataset transfer | exploratory`.
  Ghi riêng `train_dataset=CIRR`, `evaluation_dataset=FashionIQ`, training-noise=`unknown`.
- Release format: save_model stores `epoch` plus `model.state_dict()` under the class name. Safe-load the
  actual checkpoint before selecting class/config; expected possibilities include `Blip2QformerCirCat`
  (`blip2_cir_cat`) and `Blip2Qformer4TGIRCESIM` (`blip2_tgir_ce_sim`). Compare full key/shape report;
  full SHA-256 and load report remain pending download/inspection.
- FashionIQ paper row `54.99 / 76.28 / 65.64` yêu cầu training `FashionIQ_MTST`; chỉ báo cáo
  chênh số học như ngữ cảnh có điều kiện huấn luyện khác, không phải tái lập hàng paper.
- CIRR sanity-check cần dữ liệu `data/cirr_dataset/cirr` mà hiện checkout không có; chưa chạy.
  Protocol native CIRR riêng biệt khi dữ liệu sẵn có.

---

## 4. Findings F01–F16

| # | Quyết định | Thay đổi / lý do | Bằng chứng | Blocker còn lại |
|---|---|---|---|---|
| F01 | `FIXED` | adapter `airknow` dùng `--variant` (trước `--checkpoint-id`) | `test_component_boundaries.py` (2 variant + sweep 15 adapter) | — |
| F02 | `FIXED` | parser đọc cả `categories` và `results`; cả hai mà mâu thuẫn ⇒ từ chối | `test_component_boundaries.py` (4+3 model + abuse cases) | — |
| F03 | `FIXED` | `clvc_net.py` dùng `Resize(256)` theo nhánh FashionIQ native | `test_component_boundaries.py` so tensor thật | — |
| F04 | `FIXED` | `--visualize` off-by-default, qua approved patch thứ 2 | `test_component_boundaries.py` (4 test patch + gate) | — |
| F05 | `FIXED` | validator HINT/ENCODER/PAIR yêu cầu train; layout → `ilearn_resized_training`. **Lượt R9**: tập input của constructor native được kiểm bằng `constructor_input_blockers` cho **cả bảy** wrapper (`hint`, `encoder`, `pair`, `habit`, `intent`, `conesep`, `airknow`) thay vì chỉ theo category được chọn; bốn khai báo `fashioniq_layout` của habit/intent/conesep/airknow được nâng lên `[..._training]` vì layout cũ bỏ `cap.*.train.json` mà constructor vẫn mở | `test_r9_defect_regressions.py` (D1–D3, 19 test) + `test_verified_declarations.py` | — |
| F06 | `UNFINISHED_ENGINEERING` | Thay `load_model_and_preprocess` bằng processor lấy trực tiếp từ config pinned: **đã áp** cho `hint`, `habit`, `intent`, `conesep`, `airknow`. `lavis.processors` trong repo là bản pinned; `BeatsAudioProcessor` (module import `torchaudio`) bị bỏ khỏi import graph vì torchaudio không có trong môi trường khai báo của **bất kỳ** model nào và không liên quan tới text preprocessing — đây là một dòng import chết, không phải một bản thay thế hành vi. Đường code `load_model_and_preprocess` không còn được gọi (assert bằng AST), và `BlipCaptionProcessor` thật trả `'  A dog, running!  '` → `'a dog, running'`, `'Dress.'` → `'dress'`. Vẫn **không** có output parity so với invocation native đầy đủ, vì bản native cần `model_cls.from_pretrained` → `/root/bert-base-uncased` + trọng số BLIP-2 gốc mà host này không có; biên đó ghi trong docstring `workbench/replay/lavis_processors.py` và trong provenance của mỗi run (`text_processor.loaded_via`). | `test_r9_defect_regressions.py` (E1–E4, 15 test) + `workbench/scripts/lavis_processor_parity.py` (chạy được) | `FIXED_WITH_RECORDED_BOUNDARY` |
| F07 | `FIXED` | pin OpenAI CLIP revision thay `clip==0.2.0` | `test_verified_declarations.py` | — |
| F08 | `FIXED` | preflight báo `data_dir` mismatch/relative/missing | `dcnet.validate_dataset_binding` + test | — |
| F09 | `FIXED` | parser DCNet, fraction→% một lần, nhánh `comb:`, tách comp/corr | `test_verified_declarations.py` (6 test) | — |
| F10 | `FIXED` | ENCODER `VERIFIED_METADATA` + loader báo missing/unexpected | `test_verified_declarations.py`; HF API + Range read | `PUBLIC_ASSET_SETUP` |
| F11 | `BLOCKED_EXTERNAL` | tên 6 member của `CLVCNET.zip` chưa xác minh; `torch.load` whole-model trên torch ≥2.6 cần `weights_only=False` sau khi verify hash — **forward-compat, không phải lỗi torch 1.6.0** | agent report | `AUTHOR_ONLY_INFORMATION` |
| F12 | `ALREADY_FIXED` | `MHTransformer_rev.py` đã label `EXPERIMENTAL_RECONSTRUCTION`; B1/B2 mapping giữ UNVERIFIED | agent diff + registry | `AUTHOR_ONLY_INFORMATION` |
| F13 | `ALREADY_FIXED` | dims 2560/5120 đúng; noft/fullft tách; digest 5 part ghi lại | agent report | `PUBLIC_ASSET_SETUP` |
| F14 | `FIXED` | `backbone` khai báo theo variant; không khai báo ⇒ refuse | `test_verified_declarations.py` (3 test) | `PUBLIC_ASSET_SETUP` + `SCIENTIFIC_VARIANT` |
| F15 | `BLOCKED_EXTERNAL` | tái xác minh Hub: vẫn **không có** FashionIQ weight | agent report + HF API | `AUTHOR_ONLY_INFORMATION` |
| F16 | `ALREADY_FIXED` | 176 test LIMN/export/replay/fresh-clone pass; không sửa code LIMN | agent report + coordinator run | `ENVIRONMENT_RESOURCE_LIMIT` |
| R10-01 | `FIXED` | Bỏ `WORKBENCH_DCNET_DATA_DIR` khỏi quyết định binding — evaluator native không đọc biến đó, nên nó chỉ làm validator tin sai. Quyết định nay chỉ đọc `config.json` + CWD evaluator | ConfigParser **native thật**: link→selected đọc selected; link→wrong đọc `['WRONG']`; override không cứu được | — |
| R10-02 | `FIXED` | Khai `canonical_dataset_link: dataset/fashioniq`; stager tạo thư mục cha khi ghi thật; `validate_source` nhận contract đầy đủ; preflight dùng CWD checkout | Bản sao checkout: verified `True` trước/sau staging, `unauthorized_untracked == []`; dry-run không tạo parent | — |
| R10-03 | `FIXED` | `prepare()` giữ context suốt đời process; `register_path` idempotent khi cùng giá trị (khác giá trị vẫn raise); `reset()` cho tool nhiều checkout | Fixture `lavis/__init__.py` + checkpoint `torch.save` thật: load thành công, digest sai vẫn bị refuse | — |
| R10-04 | `FIXED` | Xoá 3 `source_blocker` lỗi thời (`airknow`/`conesep`/`intent`) | `preflight_blockers` thật: 3→2, 1→0, 1→0; blocker asset/runtime còn nguyên | — |
| R10-05 | `FIXED` | Bỏ `salesforce-lavis` (không thể đồng thời thoả `timm==0.9.16` của vendored `eva_vit.py`); **thêm `Adapter.launcher`/`launcher_roots`** và cho `_evaluator_imports` đi theo chương trình chạy thật thay vì `script` | Wheel sha256 khớp PyPI; scanner `hint` từ `{numpy,torch,tqdm}` thành có cả `transformers`/`timm`/`omegaconf` | 6 môi trường cài thật chưa chạy (xem §2 của `R10_REMEDIATION.md`) |
| R10-06 | `FIXED` (đã đính chính ở R11) | `inspect_environment_distributions` chỉ đọc base inventory khi interpreter **thực sự kế thừa**: `pyvenv.cfg` có `include-system-site-packages = true`. Nhánh `conda-meta` **đã bị xoá** — đó chỉ là một tên thư mục, ai cũng tạo được, và Conda env vốn cô lập nên nó tái tạo đúng lỗi R10-06 ở dạng khác | Venv thật: cô lập ⇒ `packaging` biến khỏi inventory; `--system-site-packages` ⇒ còn; prefix chỉ có `conda-meta/` ⇒ inventory **rỗng**; version sai vẫn bị chặn | — |

---

## 4b. Defect đã sửa trong lượt này (2026-10-23)

| # | Defect | Bằng chứng trước khi sửa | Sửa |
|---|---|---|---|
| D1 | `doctor.source_state` gọi `git rev-parse HEAD` trong checkout vendored (không có `.git` lồng nhau) ⇒ Git đi lên repo CIR và trả commit của CIR | LIMN pin `7d7bc9b` bị báo `3fcf11f`/`dirty`, đổ lỗi `src/train_cirr.py` (file ở gốc CIR) | route qua `vendored_source.source_provenance`; regression dùng layout thật **không** có Git lồng nhau |
| D2 | Variant mang `status: NO_CLEAN_CHECKPOINT` bị coi là "không dùng được", dù đó chỉ nghĩa "không có bản clean" | `habit` 20% đã cài + hash khớp vẫn báo `registry checkpoint unavailable: NO_CLEAN_CHECKPOINT` | đánh giá theo fact của chính variant; `BLOCKED*` vẫn chặn; chỉ kiểm variant được chọn ⇒ 20% chạy được khi thiếu 50%/80% |
| D3 | Tiến trình con của observer **không có `TMPDIR` riêng**; `isolated_cache_root` dùng `tempfile` nên cache ~11.9 GiB rơi vào `/tmp` (tmpfs RAM) | `/tmp/cir-replay-cache-*` **5.3 GB** còn lại sau khi bị kill; log `observer exited -9` | cấp `TMPDIR` 0700 theo run cho observer, như evaluator |
| D4 | Gate observation so với **đúng chuỗi bundle của LIMN**, nên ENCODER (một checkpoint, không bundle) không bao giờ khớp ⇒ observer bị bỏ im lặng | chưa từng chạy observer cho ENCODER | gate theo khai báo registry `per_query_checkpoint` |
| D5 | 7 model khai `python_env_var` nhưng **thiếu** `environment_required` ⇒ `python_executable` trả interpreter của workbench | Đặt từng biến của `csmcir, airknow, conesep, habit, intent, hint, pair` tới `workbench/envs/encoder/bin/python`; cả bảy vẫn trả `.venv/bin/python` | runtime chọn và xác minh interpreter theo khai báo `python_env_var`; `environment_required` vẫn chỉ điều khiển provisioning, môi trường thiếu/sai bị báo rõ |
| D6 | `pipeline.all_stages` tham chiếu `model_obj` chỉ tồn tại trong `real_stages` ⇒ `NameError` | 3 test suite độc lập fail (`test_authorization_policy`, `test_adversarial_orchestration`, `test_adversarial_resume`) | tra registry record tại chỗ |
| D7 | Asset LAION backbone khai `expected_sha256: null` + `UNVERIFIED` ⇒ mọi lượt ENCODER chặn ở "operator-supplied unverified provenance" | preflight ENCODER BLOCKED | ghi `lfs.oid` **do publisher công bố** (kèm size); local hash = registry = HF oid |
| D8 | Vendored checkout thiếu hoặc hỏng `provenance.json` rơi xuống Git path; Git đi lên repo CIR và mượn commit không thuộc upstream | Checkout dưới repo Git cha, không có `.git` riêng và manifest thiếu/hỏng trả commit của repo cha | `runtime.source_provenance` chỉ dùng manifest cho checkout không có Git riêng; không có record hợp lệ thì `actual_commit: null`, `verified: false` |
| D11 | **Query image khác nhau giữa aggregate và exporter**: constructor iLearn nướng tensor query lúc dựng dataset qua `get_img(..., stage=0)`, mà replay truyền `[preprocess_train, preprocess_val]` ⇒ query đi qua `RandomResizedCrop`, còn gallery dùng `stage=1` deterministic | Hai lần dựng dress khác nhau **3994/4034** tensor query (gallery giống hệt). Đây **mới là nguyên nhân thật** của việc artifact ENCODER không tái lập aggregate; giải thích cũ "cuDNN nondeterministic" là **SAI** | `common.seeded_dataset_construction` seed ngay trước khi dựng và khôi phục RNG state; áp cho `encoder`, `encoder_export`, `limn`, `pair`. Đo lại: **0/4034** khác |
| D12 | **Cache native dựng HAI lần mỗi run**: aggregate và observer mỗi bên dựng cache riêng trong temp root mới | Observer chết `[Errno 28] No space left on device` trong `save_obj` (đọc như lỗi export). Cache để lại trong run dir chiếm **10.4 GB** | `common.run_cache_root` + `WORKBENCH_REPLAY_CACHE_ROOT`: cả hai con dùng **một** cache theo run, và cache được xoá sau khi con cuối đọc xong |
| D13 | **`pinned_revision` mượn revision repo bao ngoài**: không có model record thì gọi thẳng `git rev-parse`, mà trong checkout vendored Git đi lên repo chứa nó | Trả commit repo CIR cho checkout không có `provenance.json`; test cũ **pass sai** (tmp_path ngoài Git) và **fail** khi basetemp nằm trong repo | Chỉ trả lời khi checkout có `.git` riêng; regression dựng **repo bao ngoài thật** |

D6 do subagent đưa vào và **chính lượt chạy toàn bộ suite** phát hiện — bộ test hẹp của nó không phủ.

## 4c. TME — đã tích hợp (entry thứ 17)

**Đã làm:** entry `tme` trong registry (pin `394aad79df2c104feb32a2c96132582012765062`), source
vendored (379 file, `.git` lồng nhau đã bỏ, có `provenance.json`), wrapper `workbench/replay/tme.py`,
adapter `TMEAdapter`, và protocol **`fashioniq_original_split`**.

**Sửa lại lý do protocol (R19):** một revision trước đã tạo protocol riêng
`fashioniq_val_split_ref_included` với lý do `fashioniq_original_split` dùng gallery **train split**.
Lý do đó **sai**. Theo `workbench/docs/FASHIONIQ_PROTOCOL_PARITY.md` và source, họ
`fashioniq_original_split` luôn được định nghĩa bằng **toàn bộ `image_splits/split.<category>.val.json`
theo thứ tự file, reference ELIGIBLE**. TME dựng đúng cohort đó:
`data.py:175-188` nạp `split.<cat>.val.json` vào `image_type_names[<cat>]` rồi
`current_image_names` = danh sách đó; `precompute_evaluation.py:83` chỉ mask reference khi
`category == 'CIRR'` (comment "only for cirr"), nên trên FashionIQ reference vẫn là ứng viên.
Hai cohort **trùng nhau**, nên protocol mới là thừa và đã **bị xoá** khỏi registry thay vì được nối
qua mọi consumer. ID cũ vẫn được chấp nhận như **alias** trong `src/evaluation/fashioniq.py` để các
run record cũ đọc được thay vì lỗi.

**Dòng paper (Table 2, FashionIQ validation, σ=0):** Dress R@10/R@50 **49.73/71.69**,
Shirt **56.43/74.44**, Toptee **59.31/78.94**, Average **55.15/75.02**, AVG **65.09**.

> **Historical R19 TME evidence:** the detailed measurements and commands below are preserved as historical audit text. Current asset contract requires both BLIP-2 and EVA-ViT-g; acquire them using `python -m workbench.scripts.download_auxiliary_assets --model tme`, which stages under `workbench/third_party/TME/weight/`. The older claim that BLIP-2 is optional, and the `WORKBENCH_TME_ASSET_EVA_VIT_G` / `mkdir weight` setup commands, are superseded. TME cache generation requires the separate CUDA preparation permission and ~10.47 GiB.

## 4d. TME — số đo thật (R19, 2026-10-24)

Mọi số dưới đây **đo trên máy này**, không suy đoán:

| Mục | Giá trị | Cách kiểm |
|---|---|---|
| Container Kaggle (FashionIQ 0%) | `archive.tar.gz`, magic `1f8b0800` | range probe: 302 → `storage.googleapis.com/.../bundle/archive.tar.gz` |
| Member trong container | `best_model.pth`, 2 728 382 104 B | tar header đọc trực tiếp |
| Weight đã trích, trên đĩa | `workbench/artifacts/checkpoints/tme/fashioniq_0_best_model.pth`, **2 728 382 104 B** | downloader thật, có verify + atomic publish |
| sha256 local | `fea7fc08760dad1f5a66184ea7e65b45388f9781157cbd33a3b64db74103968a` | **hash LOCAL** — Kaggle không công bố checksum, nên không được gọi là checksum tác giả |
| Cấu trúc weight | torch-zip, 862 entry, 860 tensor; `visual_encoder.*` 511 key, `Qformer.*` 336 key | `zipfile` + `torch.load(weights_only=True)` |
| Môi trường | `workbench/envs/tme` Python 3.10.13, torch 2.1.2+cu121, torchvision 0.16.2+cu121 | `uv venv` + wheel index cu121 |
| GPU thật trong interpreter đó | `RTX 3050 Laptop`, `sm_86`, `torch_cuda_version 12.1`, `allocated_tensor: true`, `computed: true` | `verify_gpu_runtime()` |
| Feature cache cần | **10.47 GiB** (15 536 ảnh × 257 token × 1408 × **2 byte**) | dtype fp16 vì `precompute.py:37` dùng `autocast` và `create_eva_vit_g` mặc định `precision="fp16"` |
| EVA-ViT-g | 2 025 249 237 B, **bắt buộc** | `eva_vit.py:446` `torch.load('weight/eva_vit_g.pth')` chạy trực tiếp trong constructor; đã chứng minh constructor chết ở file này |
| BLIP-2 pretrain | 746 998 955 B, **KHÔNG bắt buộc** | `load_from_pretrained` không được gọi trên đường eval; đã chứng minh constructor không mở file này |
| 20% noise | **`CHECKPOINT_UNAVAILABLE`** | tác giả chỉ công bố `fashioniq-0.0-best` và `cirr-0.0-best` |

**Chưa chạy được end-to-end trên máy này**: 10.47 GiB feature + 2 025 249 237 B EVA vượt dung lượng
trống. Preflight báo đúng cả hai, kèm số byte cần và dung lượng trống tại đích.

### Lệnh trên PC

```sh
export WORKBENCH_TME_PYTHON=$PWD/workbench/envs/tme/bin/python
export WORKBENCH_TME_ASSET_EVA_VIT_G=/path/to/eva_vit_g.pth
mkdir -p workbench/third_party/TME/weight   # native đọc theo cwd của evaluator

.venv/bin/python workbench/scripts/download_checkpoints.py --model tme --checkpoint fashioniq_0
workbench/envs/tme/bin/python workbench/replay/tme.py \
  --checkpoint-root workbench/artifacts/checkpoints/tme \
  --dataset-root data/fashionIQ_dataset \
  --source-root workbench/third_party/TME \
  --checkpoint-id fashioniq_0 --preflight-only
```


**20% người dùng yêu cầu: `CHECKPOINT_UNAVAILABLE`.** Kaggle `charlesneilsucian/tme` chỉ có đúng hai
instance công khai `fashioniq-0.0-best` và `cirr-0.0-best`. **Không** thay bằng 0%, **không** tự làm
nhiễu val, **không** mượn checkpoint model khác.

**Blocker:** cần tải archive Kaggle 2 533 586 661 B (chứa `best_model.pth` 2 728 382 104 B) và sinh
feature cache. Chưa xác lập SHA-256 công bố ⇒ **không** coi hash prefix là hash checkpoint.

**Giấy phép:** GitHub metadata trả `license: null`, cây gốc không có `LICENSE`. Header BSD-3 bên
trong LAVIS là của LAVIS, không phải của project này ⇒ **không** khai giấy phép.

- **Pin:** `li-shuxian/TME` HEAD `394aad79df2c104feb32a2c96132582012765062` (nhánh `master`, cập nhật
  2026-10-19). Repo phụ `He-Changhao/2025-CVPR-TME` là fork `7fe8119928`, không có gì hơn.
  **Giấy phép: GitHub metadata trả `license: null` và cây gốc KHÔNG có `LICENSE`** — không suy ra
  giấy phép từ header BSD-3 của LAVIS bên trong. (Kaggle model page tự khai MIT — khác nguồn.)
- **Code cần cho FashionIQ validation:** `src/precompute_test.py`, `src/precompute_evaluation.py`,
  `src/data.py`, `src/utility.py`, config `src/lavis/configs/models/blip2/blip2_pretrain.yaml`, và
  dependency closure của `src/lavis`. `src/precompute_train.py` là **training-only**. Tokenizer/Q-Former
  nạp từ `bert-base-uncased` (cần file local hoặc cache HF).
- **Dữ liệu FashionIQ val mở đúng:** `captions/cap.{dress,shirt,toptee}.val.json`,
  `image_splits/split.{cat}.val.json`; chế độ relative **không mở file ảnh**. Nếu phải sinh cache:
  `precompute.py` áp `targetpad_transform(1.25, 224)`, chạy ViT đóng băng, ghi
  `./features/FashionIQ/val_images.pt` + `val_name_to_idx.json` (đúng đường dẫn mà val loader đọc).
- **Cache là derived asset**, không phải dataset mới; README có đường CPU tensor/dict khi thiếu đĩa.
- **Base assets:** `blip2_pretrained.pth` (746 998 955 B, GCS MD5 `/Hph5G5SyZWDTf7dWDHYdA==`, **không**
  có SHA-256) và `eva_vit_g.pth`. Native constructor **cần cả hai** kể cả khi sau đó nạp checkpoint TME.
- **Checkpoint công bố:** Kaggle `charlesneilsucian/tme` chỉ có **`fashioniq-0.0-best` và `cirr-0.0-best`**
  ⇒ **không có 20% nào được công bố**. README cũng chỉ nói cung cấp best weight "without synthetic noise".
  ⇒ điều kiện **20% người dùng yêu cầu: `CHECKPOINT_UNAVAILABLE`**, không thay bằng 0%, không tự làm
  nhiễu val, không mượn checkpoint model khác. Nếu sau này có checkpoint nhiễu thật: paper dùng
  **best-checkpoint khi σ=0** và **final-checkpoint khi σ>0** — phải kiểm đúng quy tắc chọn.
- **Dòng paper (Table 2, FashionIQ validation, trích từ PDF CVPR):** 0% — Dress R@10/R@50
  **49.73/71.69**, Shirt **56.43/74.44**, Toptee **59.31/78.94**, Avg **55.15/75.02**, AVG **65.09**.
  20% — **54.03/73.91**, AVG **63.97**. (Bảng CIRR/Table 1 riêng: 0% R@1 53.42.)
- **Blocker:** cần tải `fashioniq-0.0-best` (archive nén 2 533 586 661 B, chứa `best_model.pth`
  2 728 382 104 B) và sinh feature cache; cả hai đều cần dung lượng lớn hơn mức còn trống hiện tại.
  Chưa xác lập SHA-256 công bố cho archive ⇒ không được coi hash một prefix là hash checkpoint.

## 4d. Trạng thái 20% và transfer (2026-10-23 tối)

| Model | Variant | Đã tải? | Hash khớp publisher? | READY? |
|---|---|---|---|---|
| habit | `fiq_n02` | **có**, 2 730 715 930 B | **có** (`814d2d0c…`) | **READY** |
| intent | `fiq_n02` | **có**, 2 722 478 522 B | **có** (`2a504cad…`) | chưa provision env |
| conesep | `fiq_n02` | chưa | — | chưa provision env |
| ptha_mtst | `cirr_transfer` (CIRR→FashionIQ) | chưa (2.53 GiB) | metadata đã ghi | chờ tải |

- **HABIT là model nhiễu đầu tiên đạt `READY`**: checkpoint tải thật + hash khớp LFS oid, environment
  dựng đúng **Python 3.8.10** như registry khai, GPU capability đã ghi. Env provision thủ công vì
  checkout **không có** `environment.yml`/`requirements.txt` nên `manage_environment.py --create`
  từ chối đoán — công thức chính xác ở `workbench/docs/MODEL_ENV_RECIPES.md`.
- **Blocker của HABIT/INTENT/ConeSep giờ là dung lượng đĩa, không phải model**: constructor native
  nướng toàn bộ tensor query lúc dựng dataset = **11.58 GiB** (đo bằng chính transform đã pin), máy
  còn **~2 GB**. Đây là cùng ràng buộc đã chặn ENCODER cho tới khi giải phóng được ~12 GB.
- **PTHA: đã tải checkpoint + đã sửa 2 defect thật + đã khai environment.**
  - Checkpoint tải thật **2 720 352 973 B**, sha256 khớp publisher (`1c4dab86…`).
  - Inspect an toàn (`weights_only=True`): dict 2 key `['epoch','Blip2QformerCirCat']`, epoch 9,
    **857 tensor**; module top-level `visual_encoder` (511), `Qformer` (336), `ln_vision`,
    `vision_proj`, `text_proj`, `itm_head`, `query_tokens`, `temp`. Key **đúng class** wrapper chờ.
  - **Defect 1**: checkout vendored giữ thư mục gốc của repo bên trong ⇒ code ở
    `<checkout>/MTST_PTHA/src` còn `requirements.txt`/`provenance.json` ở tầng ngoài. Cả hai chỗ gọi
    đều trỏ `<checkout>/src` (không tồn tại) ⇒ preflight báo thiếu evaluator. Đã sửa bằng **một**
    resolver `source_tree()`, có regression.
  - **Defect 2**: model **không khai environment** ⇒ chạy bằng interpreter workbench (Python 3.13) và
    chết `ImportError: cannot import name 'apply_chunking_to_forward' from 'transformers.modeling_utils'`.
    Checkout **có** `requirements.txt` + README pin `python=3.9`, `torch==2.0.1`, `torchvision==0.15.2`
    ⇒ đã khai từ chính nguồn đó. `manage_environment.py --create` nay thấy spec
    (`env_spec_found: true`).
  - Preflight: **NO PROBLEMS** khi có `--transfer-cirr-to-fashioniq`, **từ chối** khi không có.
  - **Blocker còn lại**: cần dựng env Python 3.9 + torch 2.0.1 (~5 GB) nhưng máy còn **1.4 GB**.
- **PTHA transfer variant đã đăng ký** với điều kiện ghi thành dữ liệu
  (`checkpoint_training_dataset: CIRR`, `checkpoint_forwards_dataset: FashionIQ`,
  `transfer_condition: cross_dataset_transfer`), không phải ghi chú. Chưa tải nên **chưa** có load
  report và **chưa** có metric FashionIQ.
- **PAIR vẫn NOT RUN**, xác nhận độc lập lần nữa: `models/` của repo tác giả chỉ có
  `MHTransformer.py` + `model.py`, **không** có `model_vssp`.
- Hash của publisher cho cả ba file 20% đã **kiểm độc lập** qua Hub API và khớp registry; HABIT và
  INTENT nay được **pin theo revision** thay vì `resolve/main`.

## 4e. Năm model ngoài registry — ghi chú research (không dựng stub)

`§11` của prompt yêu cầu giữ ghi chú ngắn cho năm model **không** có entry trong registry, và
**không** thêm stub không chạy được. Đây là lý do từng cái vắng mặt:

| Model | Trạng thái | Căn cứ |
|---|---|---|
| RankVR | **không có checkpoint công khai đã xác lập** | chưa tìm được weight do tác giả công bố |
| Css-Net | **không có checkpoint công khai đã xác lập** | như trên |
| MAPNet | **không có checkpoint công khai đã xác lập** | như trên |
| MPCIR | **tài nguyên công khai tìm được là DATASET, không phải checkpoint** | lượt review này xác định vậy; một dataset không đủ để chạy model |
| ReCon | **không có checkpoint công khai đã xác lập** | chưa tìm được weight do tác giả công bố |

Không entry nào trong số này được thêm vào registry: registry mô tả những gì workbench **có thể
chạy**, và một dòng chỉ nói "supported" mà không có checkpoint thì không phải một workflow. Nếu một
trong số chúng công bố weight có thể kiểm chứng sau này, việc thêm entry đi theo cùng đường đã dùng
cho TME: pin revision, vendor source, khai protocol, rồi mới đăng ký.

## 5. Điều kiện cập nhật về sau

1. Sửa source / checkpoint / preprocessing / protocol / runtime ⇒ phải ghi **phạm vi bằng chứng cần
   kiểm lại**. Kết quả cũ **không** tự xác minh cho code/weights mới.
2. `PREFLIGHT_PASS` **không** thành inference-ready bằng đổi tên status.
3. Full FashionIQ val cần **đủ 3 category** + aggregate đúng protocol; smoke 1 category không phải
   full benchmark.
4. Metric % phải đổi đơn vị đúng **một lần**; macro giữ quy ước native đã xác minh.
5. Thiếu per-query exporter phải ghi **thiếu** — không dựng 15 observer mới cho đẹp checklist.
6. Không chạy được GPU ⇒ ghi `PENDING` + command PC cụ thể. Không đánh dấu "không còn bug".
7. Ba protocol giữ **riêng**; không trộn cohort trong một bảng so sánh.

---

## 6. Việc cần làm trên PC, theo thứ tự

Fresh clone requires a separate workbench environment and built frontend. Install only backend
requirements here; `workbench/backend/requirements.txt` includes DuckDB. Model dependencies belong in
isolated per-model environments, never the workbench environment.

`workbench/frontend/dist/` is NOT committed (0 tracked files), but the launcher suite reads the
built bundle to confirm the official-bundle `COMPLETE` label and the backend serves the UI from it. A
clone that skips the build fails with `AssertionError: Built frontend bundle missing under
workbench/frontend/dist/assets`, so the build is the first thing to do:

```bash
cd /path/to/cir
python -m venv .venv && . .venv/bin/activate
python -m pip install -r workbench/backend/requirements.txt pytest
cd workbench/frontend && npm ci && npm run build && cd ../..
python -m pytest workbench/tests/test_fresh_clone_delivery.py \
  workbench/tests/test_vendored_source_integrity.py -q
```


### 6b. LIMN — profiles and fresh-PC procedure

The reported RTX 3050 run used historical pins on `sm_86`; it does not validate adapted `sm_120`.
Use one profile only. Historical profile: `torch==1.12.1+cu113`, `torchvision==0.13.1+cu113`,
`open_clip_torch==2.20.0`. Adapted profile: `torch==2.7.1`, `torchvision==0.22.1`,
`open_clip_torch==2.26.1`, CUDA 12.8 wheel index. These adapted exact pins satisfy registry
constraints `>=2.7,<3`, `>=0.22,<1`, and `>=2.24,<2.27`; verifier, planner/installer and replay
runner resolve same registry profile when `WORKBENCH_RUNTIME_ADAPTATION=sm_120`. This is internal
consistency only; adapted profile has not been validated on an `sm_120` host.
Install exactly one profile in its isolated environment. Historical profile reproduces only the
reported RTX 3050 setup; adapted profile is the only `sm_120` selection:

```bash
# Historical RTX 3050 profile
uv venv --python 3.10 workbench/envs/limn-historical
uv pip install --python workbench/envs/limn-historical/bin/python \
  --index-url https://download.pytorch.org/whl/cu113 \
  "torch==1.12.1+cu113" "torchvision==0.13.1+cu113"
uv pip install --python workbench/envs/limn-historical/bin/python \
  "open_clip_torch==2.20.0" "numpy==1.26.4" ftfy regex tqdm

# Adapted sm_120 profile — do not mix with historical pins or patchelf workaround
uv venv --python 3.10 workbench/envs/limn-sm120
uv pip install --python workbench/envs/limn-sm120/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.7.1" "torchvision==0.22.1"
uv pip install --python workbench/envs/limn-sm120/bin/python \
  "open_clip_torch==2.26.1" "numpy==1.26.4" ftfy regex tqdm
```

If historical torch import fails specifically with `cannot enable executable stack`, install
`patchelf` (`sudo apt-get install patchelf`) and run `patchelf --clear-execstack` on that
environment's `torch/lib/libtorch_cpu.so`. Do not apply to adapted runtime.

Run commands below from repository root after completing §6 steps 0–2. Set `FIQ` to the actual
FashionIQ root and keep `CKPT` at configured checkpoint root; no symlink is assumed:

```bash
export FIQ="$PWD/data/fashionIQ_dataset"
export CKPT="$PWD/workbench/artifacts/checkpoints"
export WORKBENCH_CHECKPOINT_ROOT="$CKPT"
export WORKBENCH_RESULTS_ROOT="$PWD/workbench/artifacts/results"
export WORKBENCH_ARTIFACTS_ROOT="$PWD/workbench/artifacts"
SRC="$PWD/workbench/third_party/LIMN"
test -d "$FIQ/images" && test -d "$FIQ/image_splits" && test -d "$FIQ/captions"

# Acquire all bundle members into the same checkpoint root used by preflight and evaluation.
python workbench/scripts/download_checkpoints.py --model limn \
  --checkpoint base_iter0_all_categories --output-root "$CKPT"

# Select interpreter/profile matching chosen environment. Historical alternative:
# unset WORKBENCH_RUNTIME_ADAPTATION
# export WORKBENCH_LIMN_PYTHON="$PWD/workbench/envs/limn-historical/bin/python"
export WORKBENCH_RUNTIME_ADAPTATION=sm_120
export WORKBENCH_LIMN_PYTHON="$PWD/workbench/envs/limn-sm120/bin/python"

# Model-interpreter preflight; checkpoint bundle root equals downloader's $CKPT/limn.
"$WORKBENCH_LIMN_PYTHON" workbench/replay/limn.py --preflight-only \
  --checkpoint-root "$CKPT/limn" --dataset-root "$FIQ" --source-root "$SRC"

# Required first-host GPU record. GPU-only probe stops before checkpoint load or inference;
# evaluation --verify-gpu can instead force a fresh probe. Matching later records may be reused.
python -m workbench.scripts.evaluate_models --model limn --dataset-root "$FIQ" --verify-gpu-only

export WORKBENCH_LIMN_BATCH_SIZE=4
export WORKBENCH_EVALUATOR_ENV_PASSTHROUGH=WORKBENCH_LIMN_BATCH_SIZE
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_limn_base_iter0_all_categories_fashioniq_val_split"
python -m workbench.scripts.evaluate_models --model limn --dataset-root "$FIQ" \
  --checkpoint base_iter0_all_categories --run-id "$RUN_ID"
python workbench/scripts/verify_run.py --run-id "$RUN_ID" --model limn \
  --checkpoint base_iter0_all_categories --protocol fashioniq_val_split
python workbench/scripts/rebuild_index.py --results-root "$PWD/workbench/artifacts/results" \
  --database "$PWD/workbench/artifacts/workbench.duckdb" --validate-first
python workbench/scripts/serve_workbench.py --host 127.0.0.1
```

**Syntax vs execution:** every command above was checked against the real parsers and entrypoints
on this host, and the steps marked `Yes` in the table below were actually executed here with the
real dataset, the three author checkpoints, the historical interpreter and the real RTX 3050. The
`sm_120` profile was NOT executed: this host has no `sm_120` device, so no adapted-profile
validation is claimed. Checkpoint acquisition (`download_checkpoints.py`) was not re-run because all
three members are already staged and hash-verified in `$CKPT/limn`.

The evaluator needs free space on the volume holding `WORKBENCH_ARTIFACTS_ROOT`: it creates a
run-scoped private `tmp` directory under the run's log directory and points the child's `TMPDIR` at
it, and the pinned LIMN `datasets.FashionIQ` constructor materializes every query it scores as
transformed image data -- `get_test_img()` applies the supplied `open_clip.image_transform`, so this
is preprocessed tensors, not raw pixels -- and pickles it into that directory before inference
starts. Measured on this host: **756.7 MB for dress alone**, and the peak run-scoped usage observed
was **1692 MB**; the free space fell to 381 MB and the constructor died with
`OSError: [Errno 28] No space left on device` at `LIMN/datasets.py:60`, which reads like a dataset
problem and is a disk problem.

The directory survives a *successful* run -- it is removed only on the launch/error cleanup paths
(`evaluate_models.py:649,674`) -- so each run leaves its own cache behind under
`workbench/artifacts/logs/<run>/tmp` and a host accumulates them. Check
`df -h "$WORKBENCH_ARTIFACTS_ROOT"` before launching, point `WORKBENCH_ARTIFACTS_ROOT` at a volume
with room when the checkout's disk is nearly full, and clear stale `logs/*/tmp` between runs: on a
host with ~2 GB free, running the evaluator while anything else writes to the same filesystem fails
inside the native constructor.

`FASHIONIQ_ROOT` defaults to `data/FashionIQ`, absent in some clones; an explicit
`--dataset-root "$FIQ"` binds LIMN preflight and evaluation to the declared dataset, while
`--verify-gpu-only` returns before any plan is built (`evaluate_models.py:2391-2394`) and therefore
consumes no dataset root at all -- it only probes and records GPU/runtime capability.
Bundle downloader receives
`$CKPT`; registry `checkpoint_path()` puts bundle members in `$CKPT/limn/`, same as replay's root.
Results and DuckDB roots are exported explicitly. The backend uses `workbench/backend/requirements.txt`
(including DuckDB); keep model packages isolated.
**Per-command status on this host:**

| Command group | Syntax accepted by source | Could execute here | Reason not run |
|---|---|---|---|
| `python -m venv`, `pip install -r workbench/backend/requirements.txt pytest` | Yes | Not run | Would alter local environment; recipe only |
| `npm ci && npm run build` | Yes | Not run | Frontend build not needed for documentation change |
| Fresh-clone pytest commands | Yes | Not run | Not instructed to run suite |
| `uv venv` / `uv pip install` historical and adapted profiles | Yes | Not run | Environment provisioning not requested; adapted host validation explicitly not claimed |
| Historical `sudo apt-get install patchelf` and `patchelf --clear-execstack` | Yes, conditional | Not run | Only after specified loader error; not applicable to adapted profile |
| Check dataset layout | Yes | Yes | `images/`, `image_splits/`, `captions/` all present under `data/fashionIQ_dataset` |
| Download LIMN checkpoint bundle | Yes | Not run | All three members already staged and hash-verified in `$CKPT/limn`; re-downloading would move 5 GiB |
| LIMN `--preflight-only` | Yes | Yes | Ran with the historical interpreter over the real bundle and dataset: `[READY] LIMN preflight passed` |
| `evaluate_models --verify-gpu-only` | Yes | Yes | Ran on the real RTX 3050: `capability_verified: true`, compute capability 8.6, record written to `artifacts/gpu_capabilities/` (it takes no `--dataset-root`, so the `FIQ` prefix is inert on this row) |
| Full `evaluate_models` invocation | Yes | Yes | Executed on the real GPU with the three author checkpoints; see the run record for the artifact paths |
| `verify_run.py`, `rebuild_index.py` | Yes | Yes | Ran against the completed run; index rebuilt from the configured results root |
| `serve_workbench.py` | Yes | Not run | The already-running deployment was reused rather than restarting containers mid-session |


**Thứ tự chuẩn bị artifact:**

1. **LIMN** (3 ckpt, mapping VERIFIED, có exporter) — chạy được ngay, cho cả per-query parity.
2. **HINT** (`fashioniq.pt`) — mapping VERIFIED, parser sẵn.
3. **ENCODER** (backbone LAION ViT-B/32 + `fashioniq.pt`) — mapping VERIFIED.
4. **CSMCIR** — apply 2 patch, CoT captions đã có, ckpt 2.7 GB + BLIP-2/EVA-ViT-g caches.
5. **HABIT/INTENT/ConeSep/Air-Know** — checkpoint noisy, phải ghi nhãn noise khi báo cáo.
6. **CLIP4Cir ×2** — tải và xác minh 5 part trước khi giải nén.
7. **SPRC** — chỉ ViT-L public; ghi rõ đây là variant khác paper.
8. **PAIR / CLVC-Net / TG-CIR / DCNet / PTHA+MTST** — cần **tác giả** cung cấp mapping/weights.

**Artifact cần chuẩn bị (không commit vào Git):** FashionIQ canonical root (`captions/`,
`image_splits/`, `images/*.png`, `resized_image/{cat}/*.jpg`, `qwen_captions/`), checkpoint root theo
`workbench/artifacts/checkpoints/<model_id>/`, auxiliary caches (`~/.cache/lavis`,
`~/.cache/torch/hub/checkpoints`, `~/.cache/clip`).
