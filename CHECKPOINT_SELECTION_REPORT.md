# CIR — Báo cáo chọn checkpoint (overnight 2026-11-05/06)

**Session:** `20261106T002617+0700` · **Evidence:** `/home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700/` · giờ máy Asia/Ho_Chi_Minh, 2026-11-06 sáng.

Máy: RTX 5070 Ti (Blackwell `sm_120`, 16 303 MiB, driver 580.173.02), 20 CPU, 31 GiB RAM.  
Repo: `/home/heheboiz/data/cir`, branch `research/cir-failure-workbench`, HEAD `26ad2dd805f02aa724f8572e83e3a89d3d8f0a8f`, `git status --short` trống (không sửa source dùng chung).

## 0. Chỉ mục evidence

| thư mục | nội dung |
|---|---|
| `manifests/` | inventory weight cũ + hash, kết quả cleanup, digest candidate tải về, digest bundle chọn, identity protocol, ngân sách |
| `measurements/` | số đo nguyên bản từng model (per-category + macro + provenance), file khoảng cách D6/D2, blocker PAIR |
| `logs/` | log từng lần chạy, kể cả 8 lần thất bại có nguyên nhân (mục 8) |
| `checkpoints/` | bundle đã stage theo model (`airknow`, `clip4cir`, `clvc`, `dcnet`, `pair`, `sprc`) |
| `prepared/` | view dataset model-specific (`clip4cir` .jpg, `sprc` .png) — chỉ symlink, không sửa canonical |
| `scripts/` | runner session gọi native evaluator + shim `compat/sitecustomize.py` cho DCNet |
| `configs/selected_overrides.yaml` | mapping candidate-level kèm SHA-256 tuyệt đối, **chưa** áp vào global |
| `measurements/selection_decisions.json` | bảng quyết định máy đọc được (status + macro từng model) |

## 1. Bảng kết quả nhanh — 8 entry ưu tiên

| model | candidate | native protocol | train / eval | run | macro R@10 | macro R@50 | mean | bundle chọn | empirical vs author | blocker / kế tiếp |
|---|---|---|---|---|---|---|---|---|---|---|
| PAIR B1 | `pair-B1.pt` (HF `iLearn-Lab/ICASSP25-PAIR` rev `e7a4b16`; staged at `$EVID/checkpoints/pair/`) | `fashioniq_val_split` | train noise không công bố / eval 0 | **BLOCKED** | — | — | — | không | UNVERIFIED | pickle cần `model_vssp.Encoder/.Backbone/.FeatureWiseAffine` + `MHTransformer_dis.*`, không tồn tại trong source công khai nào (mục 3.9) |
| PAIR B2 | `pair-B2.pt` | `fashioniq_val_split` | như trên | **BLOCKED** | — | — | — | không | UNVERIFIED | như trên; key set trùng B1 100% ⇒ là biến thể, không phải hai mảnh của một hệ |
| Combiner RN50x4 noft | `FashionIQ/RN50x4_noft/fiq_comb_RN50x4_noft.pt` | `fashioniq_original_split` | frozen encoders / eval 0 | **COMPLETED** | 35.1407 | 58.4436 | 46.7922 | `RN50x4_noft` (1 file) | `EMPIRICALLY_SELECTED_MAPPING_UNVERIFIED` | paper TOMM2023 không có hàng RN50x4-noft ⇒ paper-distance `N/A` |
| CLIP4Cir RN50x4 fullft | `RN50x4_fullft/fiq_comb_…` + `fiq_clip_…` | `fashioniq_original_split` | full-FT bundle / eval 0 | **COMPLETED** | 43.8182 | 66.9501 | 55.3841 | `RN50x4_fullft` (2 file) | `EMPIRICALLY_SELECTED_MAPPING_UNVERIFIED` | D6 = 0.2734 pp vs Table 9 |
| DCNet | author run dir (`config.json` + `trained_model.pth`) | `fashioniq_full_gallery_ref_excluded` | eval 0 | **COMPLETED** | 27.9297 | 53.7272 | 40.8285 | run dir | `EMPIRICALLY_SELECTED_MAPPING_UNVERIFIED` | D6 = 0.8842 pp; GloVe PKL tự sinh (mục 3.5) |
| CLVC-Net | `FashionIQ/<Cat>/{local,global}_model.pt` | `fashioniq_val_split` | eval 0 | **COMPLETED** | 29.9707 | 58.6217 | 44.2962 | 6 file, cặp theo category | `EMPIRICALLY_SELECTED_MAPPING_UNVERIFIED` | D6 = 0.5052 pp vs Table 1 |
| TG-CIR | `checkpoints/Fashioniq.pt` | `fashioniq_val_split` | eval 0, ViT-B/16 | **COMPLETED** | 49.7208 | 71.1023 | 60.4116 | 1 whole-model | `EMPIRICALLY_SELECTED_MAPPING_UNVERIFIED` | D6 = 1.7959 pp vs chính `Fashioniq_metrics.json` trong archive |
| AirKnow 50% | `airknow_fiq_0.5.pt` | `fashioniq_full_gallery_ref_excluded` | train noise 50 / eval 0 | **COMPLETED** | 52.5466 | 72.9026 | 62.7246 | `fiq_n05` | `EMPIRICALLY_SELECTED` (metadata VERIFIED) | D6 = 1.0277 pp vs hàng 50% |
| AirKnow 80% | `airknow_fiq_0.8.pt` | như trên | train noise 80 / eval 0 | **COMPLETED** | 50.6417 | 71.4556 | 61.0486 | `fiq_n08` | `EMPIRICALLY_SELECTED` (metadata VERIFIED) | D6 = 1.3170 pp vs hàng 80% |
| SPRC (OneDrive) | — | `fashioniq_original_split` | eval 0 | **PENDING_USER_INPUT** | — | — | — | — | UNVERIFIED | file OneDrive không có trên máy; D-1 |
| SPRC ViT-L | `sprc_fiq_vitl.pt` | `fashioniq_original_split` | eval 0, biến thể ViT-L | **COMPLETED** | 51.0386 | 73.3263 | 62.1825 | `sprc_fiq_vitl` | `EMPIRICALLY_SELECTED_MAPPING_UNVERIFIED` | D6 = 0.0268 pp vs hàng ViT-L của tác giả |

`macro = (dress + shirt + toptee)/3`; `mean = (macro_R10 + macro_R50)/2`; mọi số theo **phần trăm**.

### Tồn kho — 9 entry còn lại (inventory-only, không chạy lại)

| model | dọn weight cũ | asset hiện tại | mapping chưa rõ | chạy thêm đêm nay |
|---|---|---|---|---|
| LIMN | không nằm trong repo | trống | không | không |
| ENCODER | đã xoá 4 checkpoint + `open_clip_pytorch_model.bin` | trống | checkpoint→evaluator vẫn UNVERIFIED | không |
| HINT | không có | trống | không | không |
| HABIT | không có | trống | không | không |
| INTENT | không có | trống | không | không |
| ConeSep | không có | trống | không | không |
| CSMCIR | đã xoá `fashioniq_tuned_clip_best.pt` | trống | CoT captions không có nguồn tác giả | không |
| PTHA/MTST | không có | trống | transfer CIRR→FashionIQ, không phải FashionIQ reproduction | không |
| TME | đã xoá `best_model.pth`, `eva_vit_g.pth`, `blip2_pretrained.pth` | trống | không | không |

Đây là **trạng thái tồn kho**, không phải kết quả đo đêm nay. Bằng chứng lịch sử: `workbench/docs/MODEL_STATUS_SPEC.md`.

## 2. Môi trường và dọn dẹp

### 2.1 Máy và root hiệu dụng

- GPU khả dụng 15.43 GiB, `sm_120`, 70 SM; Xorg chiếm ~0.37 GiB nền (không đụng vào).
- `CIR_REPO_ROOT=/home/heheboiz/data/cir` (không có `workbench/config/workbench.env` ⇒ resolver dùng mặc định tương đối; đã xác nhận bằng cách gọi thẳng `workbench/backend/operator_config.resolve_config`).
- Dataset `data/FashionIQ`: 74 381 ảnh `.png` (bytes thực chất là JPEG), captions + image_splits đầy đủ cho cả 3 category.
- Checkpoint root `workbench/artifacts/checkpoints`; artifacts `workbench/artifacts`; mọi output của session nằm trong `.../checkpoint_selection/20261106T002617+0700/`.

### 2.2 Dọn weight cũ (Phase 1) — đã thực thi

Manifest hash: `/home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700/manifests/old_weight_inventory.json` (11 payload, SHA-256 từng file, URL tải lại).  
Kết quả: `/home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700/manifests/cleanup_result.json` — **đã xoá 11 file, 13421279616 byte (12.50 GiB)**, 0 file bị từ chối.

| path đã xoá | byte | nguồn tải lại |
|---|---|---|
| `/home/heheboiz/data/cir/teacher/checkpoints/csmcir/fashioniq_tuned_clip_best.pt` | 2720615859 | https://huggingface.co/peng12138/CSMCIR/resolve/main/fashioniq_tuned_clip_best.pt |
| `/home/heheboiz/data/cir/teacher/checkpoints/tgcir/fiq_stage1.pt` | 607610098 | https://drive.google.com/file/d/1OdZTtJqy-RTpYXBaq5ThmH3IBnGvYCyi/view |
| `/home/heheboiz/data/cir/teacher/checkpoints/sprc/sprc_fiq_vitl.pt` | 1459756541 | https://drive.google.com/file/d/11p95msuuTAU6Pej7d_a-KjUHHpP3kH8Y/view |
| `/home/heheboiz/data/cir/teacher/checkpoints/encoder/ckpt/fashioniq/fashioniq.pt` | 631879365 | https://huggingface.co/iLearn-Lab/AAAI25-ENCODER/resolve/dd9b086569fe69bd9111b6114546201c97934b01/fashioniq.pt |
| `/home/heheboiz/data/cir/teacher/checkpoints/encoder/ckpt/cirr/cirr.pt` | 631839254 | registry: encoder cirr variant |
| `/home/heheboiz/data/cir/teacher/checkpoints/encoder/ckpt/shoes/shoes.pt` | 631852793 | registry: encoder shoes variant |
| `/home/heheboiz/data/cir/teacher/checkpoints/encoder/ckpt/fashion200k/fashion200k.pt` | 631875597 | registry: encoder fashion200k variant |
| `/home/heheboiz/data/cir/teacher/checkpoints/tme/best_model.pth` | 2728382104 | https://www.kaggle.com/api/v1/models/charlesneilsucian/tme/pytorch/fashioniq-0.0-best/1/download |
| `/home/heheboiz/data/cir/teacher/repos/TME/weight/eva_vit_g.pth` | 2025249237 | https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth |
| `/home/heheboiz/data/cir/teacher/repos/TME/weight/blip2_pretrained.pth` | 746998955 | https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained.pth |
| `/home/heheboiz/data/cir/teacher/repos/AAAI25-ENCODER/open_clip_pytorch_model.bin` | 605219813 | https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K/resolve/main/open_clip_pytorch_model.bin |

Cơ chế an toàn: allowlist tuyệt đối (không wildcard), kiểm containment trong repo, từ chối symlink, so lại `st_size` với manifest ngay trước `unlink`, không đụng `.git`/source/config/dataset/feature cache/kết quả đo.

**Bảo toàn:** `features/` (79 GiB cache feature), `outputs/` (115 GiB checkpoint người dùng tự train), `runs/`, `reports/`, `submissions/`, `data/`, `teacher/repos` (source upstream) — tất cả nguyên vẹn.

**Ngoài phạm vi (không xoá):** `~/.cache/clip`, `~/.cache/torch`, `~/.cache/huggingface`, `~/checkpoints`. SPRC OneDrive **không tồn tại** trên máy nên không có file thủ công nào bị ảnh hưởng.

### 2.3 Environment

| env | Python | Torch | torchvision | CUDA | dùng cho |
|---|---|---|---|---|---|
| `workbench/envs/airknow-adapted` | 3.11.15 | 2.7.1+cu128 | 0.22.1+cu128 | 12.8 | AirKnow, TG-CIR, CLVC-Net, DCNet |
| `workbench/envs/clip4cir-adapted` | 3.11.15 | 2.7.1+cu128 | 0.22.1+cu128 | 12.8 | CLIP4Cir noft + fullft |
| `workbench/envs/sprc-adapted` | 3.11.15 | 2.7.1+cu128 | 0.22.1+cu128 | 12.8 | SPRC ViT-L |
| `workbench/envs/pair-adapted` | 3.11.15 | 2.7.1+cu128 | 0.22.1+cu128 | 12.8 | PAIR (chỉ kiểm cấu trúc pickle) |
| `workbench/envs/dcnet-prepare` | 3.7.12 | — | — | — | chỉ sinh GloVe PKL (spaCy 2.3.7, NLTK 3.5) |

Mỗi env đều qua **forward CUDA thật** (2048×2048 matmul + `synchronize`), không chỉ `is_available()`.
Đây là runtime **adapted cho `sm_120`**, không phải môi trường gốc của tác giả.

### 2.4 Ngân sách

- Download: **22.27 GiB / 64 GiB** (checkpoint+archive 13.7 GiB, release CLIP4Cir 7.96 GiB, spaCy vectors 0.61 GiB).
- Đĩa còn ≥ 20 GiB trong suốt session; đã xoá bản sao trùng sau khi verify (part CLIP4Cir sau khi ghép + `gzip -t` OK; archive sau khi extract member cần).
- Không đổi cấu hình hệ thống/driver/systemd/Docker/mạng. GPU job chạy tuần tự, mỗi lúc tối đa 1 model evaluation.

## 3. Protocol, ứng viên và bằng chứng đo

### 3.1 Ba protocol (đóng băng trước khi xem điểm)

| protocol ID | gallery (thứ tự) | reference | entry dùng |
|---|---|---|---|
| `fashioniq_original_split` | toàn bộ `image_splits/split.<cat>.val.json`, đúng thứ tự file | **được tính** (không mask) | Combiner RN50x4 noft, CLIP4Cir RN50x4 fullft, SPRC |
| `fashioniq_full_gallery_ref_excluded` | cùng split đầy đủ | mask trước R@K | AirKnow, DCNet |
| `fashioniq_val_split` | union first-seen (reference→target) từ `cap.<cat>.val.json` | mask trước R@K | PAIR, CLVC-Net, TG-CIR |

Số đo identity (script: `manifests/protocol_identity.json`):

| category | queries | full gallery | pair-union gallery | ref∈gallery | target∈gallery | `cap` SHA-256 | `split` SHA-256 |
|---|---|---|---|---|---|---|---|
| dress | 2017 | 3817 | 2628 | True | True | `5e5117d45695df9c…` | `21ff91d53c23859d…` |
| shirt | 2038 | 6346 | 3089 | True | True | `7b7ca3797b85dddf…` | `b82effbf7352a6f8…` |
| toptee | 1961 | 5373 | 2902 | True | True | `b4e09f428e6c255c…` | `ee42b2505275dd7a…` |

Chú thích: `cap.<cat>.val.json` và `split.<cat>.val.json` là nguồn chính thức; correction dictionaries local khớp giữa các repo iLearn.

**Cách ly cache:** mỗi lần chạy native dùng `isolated_cache_root` (symlink chỉ đọc vào captions/image_splits, cache `fashion_iq_data.json` + `test_queries_*.pkl` ghi vào temp dir riêng), nên không có cache nào bị chia sẻ giữa các repo hay giữa hai variant.

### 3.2 Nguồn, commit, checkpoint

| model | source pin | checkpoint (URL/rev) | byte | SHA-256 (local) |
|---|---|---|---|---|
| pair-B1.pt | — | https://huggingface.co/iLearn-Lab/ICASSP25-PAIR/resolve/main/pair-B1.pt | 694958565 | `c4d861ecd0baa3f1427fc7a853943f5614baf87def1cdd9c4bb7583ee925fbf2` |
| pair-B2.pt | — | https://huggingface.co/iLearn-Lab/ICASSP25-PAIR/resolve/main/pair-B2.pt | 694872371 | `74346fd7de6efdafe8dde70458316286a60ceb2d7b419efed55193418aeb3f03` |
| airknow_fiq_0.5.pt | — | https://huggingface.co/iLearn-Lab/CVPR26-Air-Know/resolve/main/airknow_fiq_0.5.pt | 2724056638 | `f4d25ed9e532f2ee3671f6edadb1048e73113246ec056b413bde75dfe3f79bc6` |
| airknow_fiq_0.8.pt | — | https://huggingface.co/iLearn-Lab/CVPR26-Air-Know/resolve/main/airknow_fiq_0.8.pt | 2724056638 | `756fc0bdc43d9e0f0cae13c8a02b0279f64e62092ce465c1e60bda45e179c54e` |
| sprc_fiq_vitl.pt | — | https://drive.google.com/file/d/11p95msuuTAU6Pej7d_a-KjUHHpP3kH8Y/view | 1459756541 | `19386104271c01ea03ee2177575c0e55f98edd3ba61a116d78f2e61ab5dbea08` |
| CLVCNET.zip | — | https://drive.google.com/file/d/159rBhWyhkLN7sXAi8iyW_ljzFNLJinKa/view | 1838590115 | `3d46104878991866ce2a230bfccd23a40065678b324900db6cd8acd629054dbd` |
| TGCIR.zip | — | https://drive.google.com/file/d/1OdZTtJqy-RTpYXBaq5ThmH3IBnGvYCyi/view | 1691535454 | `1d18bc959de14373f57eba411d01fcf01dcb7bcece4b1102056856551ea08438` |
| trained_model.pth | — | https://drive.google.com/file/d/1Hzzc7RPMwQ_09Keb3aHDWE1UxWU5o-9w/view | 2877310172 | `130f031e0f7808ccf5f15630a691bd7af95ad9fadbf94f64acfce64e7d8b850f` |
| config.json | — | https://drive.google.com/file/d/1ntGlPYUecVJKAVg-eZhdmt0ueN-RK52e/view | 1982 | `80b33d43f6f49c2eeb8ddf948a205f33a78a4c19cdf25975af033ad3be1f320a` |
| CLIP4Cir release (5 parts) | `ABaldrati/CLIP4Cir` @ `dfed9f74` | `releases/tag/weights` | 8 545 777 978 | 5/5 asset digest GitHub khớp |

Các commit đã dùng: PAIR `bfb0b98…` (vendored; author GitHub `ZhihFu/PAIR@main` byte-identical), CLIP4Cir `dfed9f74…`, DCNet `68c79d38…`, CLVC-Net `bd9b6889…`, TG-CIR `65fa78ea…`, AirKnow `6ab6fa95…`, SPRC `2935a539…`.

### 3.3 AirKnow — 50% và 80%, cả hai evaluate ở noise 0

| variant | train noise | eval noise | dress R10/R50 | shirt R10/R50 | toptee R10/R50 | macro R10 | macro R50 | mean |
|---|---|---|---|---|---|---|---|---|
| fiq_n05 | 50% | 0% | 47.89 / 70.85 | 54.37 / 72.03 | 55.38 / 75.83 | 52.5466 | 72.9026 | 62.7246 |
| fiq_n08 | 80% | 0% | 45.76 / 68.52 | 52.11 / 71.30 | 54.05 / 74.55 | 50.6417 | 71.4556 | 61.0486 |

So với paper (Table 1, block noise tương ứng), D6 (trung bình 6 ô, đơn vị pp):

| variant | paper macro R10/R50 | đo macro R10/R50 | D6 | D2 |
|---|---|---|---|---|
| 50% | 53.19 / 74.04 | 52.5466 / 72.9026 | **1.0277** | 0.8887 |
| 80% | 49.23 / 70.23 | 50.6417 / 71.4556 | **1.3170** | 1.3170 |

Cả hai đều load với SHA-256 khớp đúng metadata LFS của tác giả (`f4d25ed9…`, `756fc0bd…`), 2 724 056 638 byte mỗi file.
**Chính sách chọn:** 50% là mức nhiễu thấp hơn trong hai mức có sẵn và cả hai đều chạy đầy đủ ⇒ **recommend `fiq_n05` làm default thực dụng**, giữ `fiq_n08` như biến thể riêng. Không so với hàng 0% (55.75/75.72) và không tuyên bố checkpoint 50/80 là clean-trained.

### 3.4 CLIP4Cir — hai entry tách biệt

| entry | bundle | dress | shirt | toptee | macro R10 | macro R50 | mean |
|---|---|---|---|---|---|---|---|
| noft (frozen encoders) | `fiq_comb_RN50x4_noft.pt` | 30.69 / 56.62 | 37.00 / 57.21 | 37.74 / 61.50 | 35.1407 | 58.4436 | 46.7922 |
| fullft (clip + comb) | 2 file | 39.12 / 64.15 | 44.50 / 65.46 | 47.83 / 71.24 | 43.8182 | 66.9501 | 55.3841 |

Paper Table 9 (RN-50x4, full-FT): dress 39.46/64.55, shirt 44.41/65.26, toptee 47.48/70.98, average 43.78/66.93.  
**D6 = 0.2734 pp**, D2 = 0.0291 pp. 6 ô lệch: [0.342, 0.395, 0.094, 0.196, 0.353, 0.259].

Không có hàng paper cho cấu hình noft RN50x4 (paper chỉ in RN-50 cho các bảng ablation; RN-50x4 chỉ xuất hiện ở bảng SotA full-FT) ⇒ paper-distance `N/A`, chọn theo cấu hình tác giả + inference đầy đủ thành công.

Nguyên văn tham số đã dùng: `--combining-function combiner --projection-dim 2560 --hidden-dim 5120 --clip-model-name RN50x4 --transform targetpad --target-ratio 1.25 --dataset FashionIQ`; fullft thêm `--clip-model-path fiq_clip_RN50x4_fullft.pt`. Base RN50x4 (OpenAI, 421 854 225 byte) tải đúng vào `~/.cache/clip`.

### 3.5 DCNet — fusion hai nhánh, gallery đầy đủ, reference masked

| category | fused R10 | fused R50 | comp R10 | comp R50 | corr R10 | corr R50 | paper R10 | paper R50 |
|---|---|---|---|---|---|---|---|---|
| dress | 29.1522 | 54.5364 | 27.962 | 53.495 | 28.855 | 54.983 | 28.95 | 56.07 |
| shirt | 22.8165 | 47.6447 | 21.982 | 46.271 | 22.718 | 47.154 | 23.95 | 47.3 |
| toptee | 31.8205 | 59.0005 | 30.087 | 58.032 | 31.464 | 58.185 | 30.44 | 58.29 |
| **macro** | **27.9297** | **53.7272** | — | — | — | — | 27.78 | 53.89 |

`comb_avg` do evaluator in ra = 40.828482 vs paper printed Avg **40.83** ⇒ **D6 = 0.8842 pp**, D2 = 0.1562 pp.
Điểm cuối là `log_softmax(score_comp) + log_softmax(score_corr)` sau khi cả hai nhánh đã mask reference bằng `-10e10` (`trainer_joint.py:216-243`); không báo cáo nhánh đơn lẻ như kết quả method.

Chuẩn bị đầu vào (tất cả model-specific, sinh trong session):
- `resized_images/<id>.jpg` phẳng 256×256 LANCZOS (theo `preprocess/resize_img.py` của tác giả), 15 415 ảnh id duy nhất.
- `captions/cap.<cat>.glove.{train,val,test}.pkl` + `glove_vecs.pkl` sinh bằng `dcnet-prepare` (Python 3.7.12, spaCy 2.3.7 + `en_vectors_web_lg 2.1.0` + NLTK 3.5 punkt), đúng `process_cap.py`. **Hai khác biệt đã ghi:** (a) seed NumPy trước khi sinh vector OOV (script gốc không seed) ⇒ file tái lập được nhưng OOV vectors không thể trùng byte với tác giả (tác giả không công bố PKL); (b) trỏ spaCy package bằng path thay vì tên.
- `config.json`: **chỉ đổi 3 giá trị đường dẫn/vận hành** — `data_loader.args.data_dir` → `/home/heheboiz/data/cir/data/FashionIQ`, `num_workers` 6 → 8, `trainer.save_dir` → thư mục session. Mọi siêu tham số kiến trúc/loss/optimizer giữ nguyên (diff đầy đủ trong log).
- Chạy đúng CLI tác giả: `python test.py --resume <run_dir>` từ checkout `workbench/third_party/DCNet` (cwd là một phần contract, `parse_config.py:28`).
- Torch 2.7 mặc định `weights_only=True` từ 2.6 nên `base_trainer.py:195` từ chối checkpoint; dùng shim `sitecustomize.py` chỉ inject vào subprocess DCNet, chỉ đặt lại default cũ cho `torch.load` (không sửa checkout).

### 3.6 CLVC-Net — mỗi category một cặp local/global

| category | local | global | R@1 | R@10 | R@50 |
|---|---|---|---|---|---|
| dress | `dress_local_model.pt` (33d44186efd6a25c…) | `dress_global_model.pt` | 6.792 | 29.3505 | 56.8171 |
| shirt | `shirt_local_model.pt` (2ba8e320b08ebb97…) | `shirt_global_model.pt` | 6.428 | 27.7723 | 55.1521 |
| toptee | `toptee_local_model.pt` (d1fa9170496f73a0…) | `toptee_global_model.pt` | 7.853 | 32.7894 | 63.8960 |
| **macro** | — | — | — | **29.9707** | **58.6217** |

Paper Table 1: dress 29.85/56.47, shirt 28.75/54.76, toptee 33.50/64.00, avg 30.70/58.41 ⇒ **D6 = 0.5052 pp**, D2 = 0.4705 pp. 6 ô lệch: [0.499, 0.347, 0.978, 0.392, 0.711, 0.104].
Archive tác giả `CLVCNET.zip` chứa đúng `checkpoint/FashionIQ/{Dress,Shirt,Toptee}/{local,global}_model.pt` (6 file, 175–219 MB mỗi file) — mapping giờ đã kiểm bằng tên + shape, không còn là giả định.
Transform native FashionIQ val (`train.py:90-96`): `Resize(256) → CenterCrop(224) → ToTensor → ImageNet Normalize`. Evaluator là `test.test(params, local, global, testset, category)` nguyên bản; gallery pair-union, reference mask `-10e10`.

### 3.7 TG-CIR — whole model, ViT-B/16

| category | R@1 | R@10 | R@50 |
|---|---|---|---|
| dress | 14.378 | 43.3813 | 68.3689 |
| shirt | 22.424 | 51.4720 | 69.8234 |
| toptee | 22.590 | 54.3090 | 75.1147 |
| **macro** | — | **49.7208** | **71.1023** |

Archive `TGCIR.zip` có `checkpoints/Fashioniq.pt` (607 928 637 byte) và chính tác giả kèm `Fashioniq_metrics.json` ghi macro R10 51.3204 / R50 73.0945 — trùng khít hàng paper `tgcir_fashioniq_val_table1` (51.32 / 73.09), nên hàng paper và archive **nhất quán**.
Đo được 49.7208 / 71.1023 ⇒ **D6 = 1.7959 pp** so với chính số tác giả. 6 ô lệch: [1.834, 1.289, 1.129, 2.699, 1.836, 1.989].
Lưu ý backbone: checkpoint dùng **ViT-B/16** (`clip.load('ViT-B/16')`, `fc 512×768`), khớp backbone paper ⇒ đây là biến thể hợp lệ, không phải variant lệch backbone.
Code pinned `workbench/third_party/TGCIR` và `codes/` trong archive **giống hệt từng byte** (`test.py`, `datasets.py`, `model.py`).
Tương thích pickle cũ (2 điều chỉnh state-completion, ghi rõ trong runner): `torch.nn.modules.linear._LinearWithBias → torch.nn.Linear`; gán `MultiheadAttention.batch_first = False` cho module load từ checkpoint torch 1.7 (giá trị duy nhất mà class cũ có). Không đổi forward/score.

### 3.8 SPRC — ViT-L chạy được, OneDrive thiếu

| category | gallery | queries | R@10 | R@50 | paper ViT-L R@10/R@50 |
|---|---|---|---|---|---|
| dress | 3817 | 2017 | 45.811 | 70.302 | 45.81 / 70.4 |
| toptee | 5373 | 1961 | 55.686 | 77.155 | 55.69 / 77.21 |
| shirt | 6346 | 2038 | 51.619 | 72.522 | 51.62 / 72.52 |
| **macro** | — | — | **51.0386** | **73.3263** | 51.04 / 73.38 (printed) |

**D6 = 0.0268 pp**; D2 so printed macro = 0.0275 pp, so macro tính lại từ ô = 0.0259 pp. 6 ô lệch: [0.001, 0.098, 0.001, 0.002, 0.004, 0.055].

Kiểm tra mapping: `torch.load(..., weights_only=False)` cho dict 2 khoá `{epoch: 7, Blip2QformerCirAlignPrompt: OrderedDict(628)}`; instantiate `blip2_cir_align_prompt` với `pretrain_vitL` (CLIP ViT-L, conv1 `1024×3×14×14`, positional 257×1024, Qformer hidden 768) rồi `load_state_dict(strict=False)` ⇒ **0 missing, 0 unexpected**. Vì vậy backbone được xác nhận bằng tensor shape, không phải filename.

Điều chỉnh vận hành đã ghi: runner gọi đúng `extract_index_blip_features` + `compute_fiq_val_metrics` của validator với `save_memory=True`, giữ feature index trên CPU giữa các batch. Trên card 16 GiB, `torch.vstack` của 3817×257×1408 fp16 (5.27 GiB) + activation làm OOM; `save_memory` chỉ đổi **nơi lưu** tensor, không đổi giá trị. Preprocessing TargetPad 1.25 + soạn caption giữ nguyên.

File OneDrive (`sprc_fiq.pt`, EVA-CLIP ViT-g) **không có** trên máy ⇒ `PENDING_USER_INPUT`, xem D-1. Không dùng ViT-L thay thế cho hàng ViT-g.

### 3.9 PAIR — blocker đã điều tra có giới hạn

Cả hai file tải đủ và hash ổn định:

| file | byte | SHA-256 | root class | state_dict entries | params |
|---|---|---|---|---|---|
| `pair_b1.pt` | 694958565 | `c4d861ecd0baa3f1427fc7a853943f5614baf87def1cdd9c4bb7583ee925fbf2` | `model_vssp.Encoder` | 455 | 172,814,185 |
| `pair_b2.pt` | 694872371 | `74346fd7de6efdafe8dde70458316286a60ceb2d7b419efed55193418aeb3f03` | `model_vssp.Encoder` | 455 | 172,814,185 |

Đã thử (trong giới hạn ~30 phút/model):
- Resolved HF repo iLearn-Lab/ICASSP25-PAIR @ e7a4b16ba757483637fe8a63745a833c8783edf6; files = README.md, .gitattributes, pair-B1.pt, pair-B2.pt only.
- Resolved author GitHub ZhihFu/PAIR main (tarball extracted, 15463584 bytes). models/model.py is byte-identical to the vendored pin bfb0b98 (md5 2a4427fdaa51da8e9c06dbfc03957074) and defines class PAIR + Backbone in module `model`, not `model_vssp`.
- Probed ZhihFu/PAIR branches main/master/dev/release/code/train: only `main` exists (200); no other revision of models/ ever shipped a model_vssp.py.
- Probed GitHub users ZhihFu (5 public repos) and ZhihengFu; no repo publishes model_vssp or MHTransformer_dis.
- Repo history: initial commit 4fa4de5 had no .py files; every later tree lists the same 7 python files.
- Structural inspection (safe stub classes, no forward) shows the released state dicts use transformed-module names absent from the public source: backbone.weight_chanelImg(.0/.1/.3), backbone.weight_chanelImg_t, backbone.cross_t2i, backbone.transformer_a..d, backbone.mlp_img, backbone.mlp_text. The public PAIR class has no such attributes, so a key mapping cannot be justified from source.
- The closest published sibling is AAAI25-ENCODER (model_try2.py) whose Backbone also has weight_chanelImg/cross_t2i, but it additionally requires weight_chanelText, weight_chanelText_i, cross_i2t, weight_cross, sparse_net, sparse_net_text, binding_decoder — none present in the PAIR blobs, and its Encoder.compose_feature returns a different tuple. Aliasing it would be an invented architecture.

Kết luận: **không thể** dựng model trung thực. Alias `model_vssp` → `model.PAIR` sẽ sai vì các key như `backbone.weight_chanelImg`, `backbone.cross_t2i`, `backbone.transformer_a..d`, `backbone.mlp_img/mlp_text` không tồn tại trong class công khai; mượn `AAAI25-ENCODER` cũng sai vì thiếu 7 module và `compose_feature` trả tuple khác. Không tạo dummy class, không thay model CIR khác. Hai biến thể giữ nguyên, chưa xếp hạng.


### 3.10 Empirical recommendation vs author-verified association

| entry | association checkpoint↔kiến trúc | nguồn xác nhận | kết quả đo |
|---|---|---|---|
| AirKnow 50% / 80% | **AUTHOR_VERIFIED** | SHA-256 khớp đúng LFS `oid` do tác giả công bố trên Hub (`f4d25ed9…`, `756fc0bd…`), đúng tên file, đúng Framework `iLearn-Lab/CVPR26-Air-Know` | đo đầy đủ, D6 1.03 / 1.32 pp |
| SPRC ViT-L | **AUTHOR_VERIFIED (architecture)** | state dict nạp vào đúng `Blip2QformerCirAlignPrompt` với `pretrain_vitL`: 0 missing / 0 unexpected; conv1 `1024×3×14×14`, positional `257×1024` ⇒ CLIP ViT-L/14 | đo đầy đủ, D6 0.027 pp |
| CLIP4Cir noft / fullft | **EMPIRICAL** | tên member + shape state dict khớp `Combiner(feature_dim=640, 2560, 5120)` và `CLIP` 669 key; không có checksum tác giả | đo đầy đủ |
| DCNet | **EMPIRICAL** | `config.json` + `trained_model.pth` cùng thư mục Drive tác giả; state dict 518 key khớp `CENet`; không có checksum | đo đầy đủ |
| CLVC-Net | **EMPIRICAL** | member `checkpoint/FashionIQ/<Cat>/{local,global}_model.pt` được archive tác giả đặt tên đúng; 415 key khớp `compose_local`/`compose_global` | đo đầy đủ |
| TG-CIR | **EMPIRICAL** | `checkpoints/Fashioniq.pt` + chính tác giả kèm `Fashioniq_metrics.json` cùng archive; 489 key khớp `VCG_CIR` | đo đầy đủ |
| SPRC OneDrive (ViT-g) | **UNVERIFIED** | file không có trên máy | chưa chạy |
| PAIR B1 / B2 | **UNVERIFIED (blocked)** | lớp nguồn không tồn tại công khai | chưa chạy |

Nói cách khác: kết quả đo là **empirical**; chỉ AirKnow (SHA tác giả) và SPRC ViT-L (khớp lớp tuyệt đối) có association do tác giả/kiến trúc xác nhận, và **không entry nào** được gọi là "author-verified reproduction" của một con số paper.

## 4. Hàng đợi quyết định buổi sáng

| ID | model/job | hành động đề xuất | vì sao | phạm vi/rủi ro | đã thử an toàn | bằng chứng | cần bạn quyết/ cung cấp |
|---|---|---|---|---|---|---|---|
| **D-1** | `sprc` (biến thể chính, EVA-CLIP ViT-g) | Bạn tải file OneDrive rồi đặt tại `/home/heheboiz/data/cir/workbench/artifacts/checkpoints/sprc/sprc_fiq.pt` | Bản chính (paper config) chỉ có trên OneDrive; tôi không thể login/tương tác | Chỉ thêm 1 file 1.4–2.6 GB vào checkpoint root; không đổi code | Đã chạy biến thể ViT-L công khai (D6 0.0268 pp) — **không** dùng thay hàng ViT-g | `measurements/sprc_vitl_run1.json`, `measurements/sprc_distance.json` | Tải https://1drv.ms/u/s!Aj0q22vyiZbabnUya4mnufIBtYI?e=n4ZVKj (Baidu dự phòng, pass `t1at`) và đặt đúng path trên |
| **D-2** | `pair` (B1 + B2) | Cho phép tôi (a) liên hệ tác giả để xin `model_vssp.py`/`MHTransformer_dis.py`, hoặc (b) chấp nhận bỏ PAIR khỏi cohort, hoặc (c) cung cấp source nếu bạn có bản khác | Không có class nguồn ⇒ không thể load trung thực; không được bịa kiến trúc | (a) cần bạn gửi email/issue; (b) chỉ là quyết định phạm vi | Đã probe HF, GitHub author/main + 5 branch khác, 2 namespace tác giả, git history — không có | `measurements/pair_blocker.json`, `measurements/pair_b1_structure.txt`, `pair_b2_structure.txt` | Chọn (a) hay (b), hoặc gửi file source nếu có |
| **D-3** | registry dùng chung | Duyệt diff registry tối thiểu để map 5 entry đã chạy được (xem mục 9.1) | Hiện registry vẫn ghi `CHECKPOINT_MAPPING_UNVERIFIED` cho các bundle nay đã đo được; không nên tự sửa default toàn cục qua đêm | Sửa `workbench/registry/models.yaml` (mapping/`reproduction_status`) + thêm `preparation_contracts` cho DCNet GloVe (hiện `MANUAL_REQUIRED`, không generator) | Không sửa gì; chỉ ghi `configs/selected_overrides.yaml` trong session | `configs/selected_overrides.yaml`, mục 9.1 | Duyệt/không duyệt diff registry |
| **D-4** | DCNet reproducibility | Cho phép ghi lại recipe sinh GloVe PKL thành generator trong `preparation_contracts.yaml` (seed cố định, sha256 của 10 file PKL) | Hiện contract ghi `NONDETERMINISTIC_WITHOUT_AUTHOR_ARTIFACT`, nên lần sau không tái lập được bằng workbench | Sửa 1 khối contract + thêm 10 entry hash; không đổi dataset canonical | Đã sinh trong session, giữ file + log, không sửa contract | `logs/dcnet_glove_prep.log`, `logs/dcnet_run2.log` | Duyệt việc chuẩn hoá recipe này |
| **D-5** | `torch.load` shim cho DCNet | Cho phép thêm 1 dòng `weights_only=False` (có hash-check trước) vào `base/base_trainer.py` như `approved_source_patches` thay vì shim qua `PYTHONPATH` | Hiện phải inject shim để unpickle; nếu duyệt thì DCNet chạy được bằng CLI chuẩn không cần shim | Sửa 1 dòng trong checkout pinned + khai báo patch như CSMCIR | Đã chạy bằng shim, kết quả đầy đủ | `scripts/compat/sitecustomize.py`, `logs/dcnet_run2.log` | Duyệt patch 1 dòng (kèm base/derived sha256) hay giữ shim |

Không mục nào chặn các job khác: PAIR bị block nhưng 8 candidate còn lại đã hoàn tất độc lập.

## 5. Lệnh chạy lại chính xác

*Tất cả path tuyệt đối; không có placeholder; không lệnh nào gọi training.*

```bash
# 0) biến chung
export EVID=/home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700
export CIR=/home/heheboiz/data/cir
export DS=$CIR/data/FashionIQ

# 1) AirKnow 50% (recommended) và 80% — protocol fashioniq_full_gallery_ref_excluded
cd $CIR
$CIR/workbench/envs/airknow-adapted/bin/python -m workbench.replay.airknow \
  --checkpoint-root "$EVID/checkpoints/airknow" --dataset-root "$DS" \
  --source-root $CIR/workbench/third_party/AirKnow --variant fiq_n05 \
  --batch-size 96 --output "$EVID/measurements/airknow_fiq_n05_rerun.json"
$CIR/workbench/envs/airknow-adapted/bin/python -m workbench.replay.airknow \
  --checkpoint-root "$EVID/checkpoints/airknow" --dataset-root "$DS" \
  --source-root $CIR/workbench/third_party/AirKnow --variant fiq_n08 \
  --batch-size 96 --output "$EVID/measurements/airknow_fiq_n08_rerun.json"

# 2) CLIP4Cir noft — protocol fashioniq_original_split
cd $CIR/workbench/third_party/CLIP4Cir
$CIR/workbench/envs/clip4cir-adapted/bin/python src/validate.py --dataset FashionIQ \
  --combining-function combiner \
  --combiner-path "$EVID/checkpoints/clip4cir/FashionIQ/RN50x4_noft/fiq_comb_RN50x4_noft.pt" \
  --projection-dim 2560 --hidden-dim 5120 --clip-model-name RN50x4 \
  --transform targetpad --target-ratio 1.25

# 3) CLIP4Cir fullft (bundle 2 state)
cd $CIR/workbench/third_party/CLIP4Cir
$CIR/workbench/envs/clip4cir-adapted/bin/python src/validate.py --dataset FashionIQ \
  --combining-function combiner \
  --combiner-path "$EVID/checkpoints/clip4cir/FashionIQ/RN50x4_fullft/fiq_comb_RN50x4_fullft.pt" \
  --clip-model-path "$EVID/checkpoints/clip4cir/FashionIQ/RN50x4_fullft/fiq_clip_RN50x4_fullft.pt" \
  --projection-dim 2560 --hidden-dim 5120 --clip-model-name RN50x4 \
  --transform targetpad --target-ratio 1.25

# 4) DCNet — CLI gốc của tác giả, run directory trong session
cd $CIR/workbench/third_party/DCNet
PYTHONPATH="$EVID/scripts/compat:$CIR/workbench/third_party/DCNet" \
  TORCH_HOME=$HOME/.cache/torch \
  $CIR/workbench/envs/airknow-adapted/bin/python test.py --resume "$EVID/checkpoints/dcnet"

# 5) CLVC-Net — runner gọi native test.test() với 6 checkpoint theo category
$CIR/workbench/envs/airknow-adapted/bin/python "$EVID/scripts/run_clvc.py" \
  "$EVID/checkpoints/clvc" "$DS" "$EVID/measurements/clvc_rerun.json" 48

# 6) TG-CIR — runner gọi native test.test() (whole model ViT-B/16)
$CIR/workbench/envs/airknow-adapted/bin/python "$EVID/scripts/run_tgcir.py" \
  "$EVID/downloads/tgcir/x/checkpoints/Fashioniq.pt" "$DS" \
  "$EVID/measurements/tgcir_rerun.json" 64

# 7) SPRC ViT-L — runner gọi native extract_index_blip_features + compute_fiq_val_metrics
cd $CIR/workbench/third_party/SPRC/src
HF_HOME=$HOME/.cache/huggingface $CIR/workbench/envs/sprc-adapted/bin/python \
  "$EVID/scripts/run_sprc.py" "$EVID/checkpoints/sprc/sprc_fiq_vitl.pt" "$EVID/measurements/sprc_rerun.json" pretrain_vitL

# 8) SPRC main (sau khi D-1 được cung cấp) — tương tự, đổi checkpoint + backbone
cd $CIR/workbench/third_party/SPRC/src
HF_HOME=$HOME/.cache/huggingface $CIR/workbench/envs/sprc-adapted/bin/python \
  "$EVID/scripts/run_sprc.py" "$CIR/workbench/artifacts/checkpoints/sprc/sprc_fiq.pt" "$EVID/measurements/sprc_main.json" pretrain

# 9) PAIR — cả hai candidate, sau khi D-2 được giải quyết
cd $CIR
$CIR/workbench/envs/pair-adapted/bin/python -m workbench.replay.pair \
  --checkpoint-root "$EVID/checkpoints/pair" --dataset-root "$DS" \
  --source-root $CIR/workbench/third_party/PAIR --checkpoint-id pair_b1 \
  --batch-size 64 --output "$EVID/measurements/pair_b1.json"
$CIR/workbench/envs/pair-adapted/bin/python -m workbench.replay.pair \
  --checkpoint-root "$EVID/checkpoints/pair" --dataset-root "$DS" \
  --source-root $CIR/workbench/third_party/PAIR --checkpoint-id pair_b2 \
  --batch-size 64 --output "$EVID/measurements/pair_b2.json"
```

## 6. Đối soát hoàn thành

| candidate | trạng thái | ghi chú |
|---|---|---|
| PAIR B1 | **BLOCKED** | lớp `model_vssp.Encoder/.Backbone/.FeatureWiseAffine` + `MHTransformer_dis.*` không tồn tại trong source công khai; đã probe HF/GitHub/branch/history |
| PAIR B2 | **BLOCKED** | cùng nguyên nhân; 455 key trùng B1 nên là biến thể |
| Combiner RN50x4 noft | **COMPLETED** | 35.1407 / 58.4436; không có hàng paper ⇒ distance N/A |
| CLIP4Cir RN50x4 fullft | **COMPLETED** | 43.8182 / 66.9501; D6 0.2734 |
| DCNet | **COMPLETED** | 27.9297 / 53.7272; D6 0.8842 |
| CLVC-Net | **COMPLETED** | 29.9707 / 58.6217; D6 0.5052 |
| TG-CIR | **COMPLETED** | 49.7208 / 71.1023; D6 1.7959 so với số chính tác giả |
| AirKnow 50% | **COMPLETED** | 52.5466 / 72.9026; D6 1.0277 |
| AirKnow 80% | **COMPLETED** | 50.6417 / 71.4556; D6 1.3170 |
| SPRC OneDrive | **PENDING_USER_INPUT** | file không có trên máy (D-1) |
| SPRC ViT-L | **COMPLETED** | 51.0386 / 73.3263; D6 0.0268 |

8/11 candidate chạy xong đầy đủ 3 category, có số đo thật (load checkpoint → forward → ranking → Recall), query/gallery coverage 100% (2017/3817, 2038/6346, 1961/5373).
PAIR: 2 candidate blocked ở tầng class nguồn (đã điều tra có giới hạn, xem mục 3.9). SPRC main: chỉ thiếu file người dùng (D-1).

## 7. Những gì KHÔNG được chứng minh

- Không số nào ở đây là `AUTHOR_VERIFIED_REPRODUCTION`. Hai mức được tách rõ ở mục 3.10: association (file ↔ lớp/kiến trúc) và kết quả đo.
- Validation **đã được dùng để chọn**; các số trên không phải so sánh held-out không thiên lệch.
- Không tuyên bố đã reproduce paper: khoảng cách D6 từ 0.027 đến 1.8 pp được nêu nguyên văn, không quy về 0.
- Không chạy composition adapter/CIRCUS cho các repo này; đó là hạng mục riêng.
- Không push, không merge, không deploy, không đổi dịch vụ toàn cục.

## 8. Chi tiết vận hành từng lần chạy

| model / bước | command (đã thực thi) | cwd | runtime | log | kết quả |
|---|---|---|---|---|---|
| PAIR B1/B2 cấu trúc | `inspect_pair_struct.py <ckpt> <stubs>` | /home/heheboiz/data/cir | pair-adapted 3.11 / torch 2.7.1+cu128 | `measurements/pair_b1_structure.txt`, `pair_b2_structure.txt` | 455 entries / 172 814 185 params; key set trùng nhau |
| AirKnow 50% | `python -m workbench.replay.airknow --variant fiq_n05 --batch-size 96` | /home/heheboiz/data/cir | airknow-adapted | `logs/airknow_n05_run1.log` | 52.5466 / 72.9026 / 62.7246 |
| AirKnow 80% | `python -m workbench.replay.airknow --variant fiq_n08 --batch-size 96` | /home/heheboiz/data/cir | airknow-adapted | `logs/airknow_n08_run1.log` | 50.6417 / 71.4556 / 61.0486 |
| CLIP4Cir noft (lần 1) | `src/validate.py --combining-function combiner --combiner-path …noft…` | third_party/CLIP4Cir | clip4cir-adapted | `logs/clip4cir_noft.log` | FAILED: thiếu link `fashionIQ_dataset` |
| CLIP4Cir noft (lần 2) | như trên sau khi stage dataset view 15 415 symlink `.jpg` | third_party/CLIP4Cir | clip4cir-adapted | `logs/clip4cir_noft_run2.log` | 35.1407 / 58.4436 / 46.7922 |
| CLIP4Cir fullft | `src/validate.py … --clip-model-path fiq_clip_RN50x4_fullft.pt` | third_party/CLIP4Cir | clip4cir-adapted | `logs/clip4cir_fullft_run1.log` | 43.8182 / 66.9501 / 55.3841 |
| CLVC-Net | `run_clvc.py <ckpt-dir> <dataset> <out> 48` | /home/heheboiz/data/cir | airknow-adapted | `logs/clvc_run1.log` | 29.9707 / 58.6217 / 44.2962 |
| TG-CIR (lần 1) | `run_tgcir.py …` | /home/heheboiz/data/cir | airknow-adapted | `logs/tgcir_run1.log` | FAILED: `_LinearWithBias` (pickle torch 1.7) |
| TG-CIR (lần 2) | `run_tgcir.py …` sau alias class | /home/heheboiz/data/cir | airknow-adapted | log đã bị lần 3 ghi đè; thông điệp lỗi giữ trong báo cáo | FAILED: `'MultiheadAttention' object has no attribute 'batch_first'` |
| TG-CIR (lần 3) | `run_tgcir.py …` sau khi bổ sung `batch_first=False` | /home/heheboiz/data/cir | airknow-adapted | `logs/tgcir_run2.log` | 49.7208 / 71.1023 / 60.4116 |
| DCNet (lần 1) | `test.py --resume <run_dir>` | third_party/DCNet | airknow-adapted | `logs/dcnet_run1.log` | FAILED: `weights_only=True` mặc định torch ≥2.6 |
| DCNet (lần 2) | `test.py --resume <run_dir>` với `PYTHONPATH` shim `torch.load` | third_party/DCNet | airknow-adapted | `logs/dcnet_run2.log`, `checkpoints/dcnet/val/info.log` | 27.9297 / 53.7272 / 40.8285 |
| SPRC ViT-L (lần 1) | `blip_validate.py --backbone pretrain_vitL --model-path …` | third_party/SPRC/src | sprc-adapted | `logs/sprc_vitl_run1.log` | FAILED: link dataset ở `SPRC/` không phải `SPRC/src/` |
| SPRC ViT-L (lần 2) | như trên sau khi sửa link | third_party/SPRC/src | sprc-adapted | `logs/sprc_vitl_run2.log` | FAILED: OOM khi `torch.vstack` index features |
| SPRC ViT-L (lần 3) | `blip_validate.py` + `expandable_segments:True` | third_party/SPRC/src | sprc-adapted | `logs/sprc_vitl_run3.log` | FAILED: vẫn OOM (5.27 GiB vstack) |
| SPRC ViT-L (lần 4) | `run_sprc.py … save_memory=True` | third_party/SPRC/src | sprc-adapted | `logs/sprc_vitl_run4.log` | FAILED: attribute `relative_captions` trong runner (không phải evaluator) |
| SPRC ViT-L (lần 5) | `run_sprc.py … save_memory=True` (đã sửa đếm query) | third_party/SPRC/src | sprc-adapted | `logs/sprc_vitl_run5.log` | 51.0386 / 73.3263 / 62.1825 |

Thời lượng quan sát (wall-clock, GPU độc quyền):

| model | thời gian | batch | ghi chú |
|---|---|---|---|
| AirKnow (mỗi variant) | ~4.5 phút | 96 | 3 category, 2017+2038+1961 query, gallery 3817/6346/5373 |
| CLIP4Cir (mỗi entry) | ~8 phút | 32 (native) | targetpad 1.25, RN50x4 |
| CLVC-Net | ~2 phút | 48 | 6 whole model, gallery pair-union |
| TG-CIR | ~3 phút | 64 | whole model ViT-B/16 |
| DCNet | ~8 phút | 32 (native) | full gallery + GloVe PKL đọc |
| SPRC ViT-L | ~3.5 phút | 64 index / 16 query (native) | 3 category |

Đo hiệu năng (mẫu `nvidia-smi` rời rạc trong lúc chạy): GPU util dao động **7–100%**, đạt 100% ở các pha forward đặc (AirKnow, CLIP4Cir, DCNet); các pha thấp là lúc DataLoader/PIL decode hoặc lúc build gallery, không phải GPU nhàn rỗi do cấu hình. VRAM đỉnh 9–15 GiB trên 16 GiB, không OOM, không swap. Batch 96 (AirKnow) và 64 (TG-CIR/SPRC) là mức đã kiểm bằng chạy thật; SPRC bị OOM ở batch native 64-index nên dùng `save_memory` thay vì giảm chất lượng.
DataLoader: giữ **nguyên** cấu hình của từng evaluator (worker/prefetch/pin_memory là một phần semantics tái lập của tác giả; đổi chúng sẽ làm số đo không còn so sánh được với paper). Riêng DCNet `num_workers` 6 → 8 trong config copy của session vì đó là tham số vận hành, đã ghi diff. Không tăng batch của AirKnow/CLIP4Cir vượt mức native để "cho nhanh": batch là tham số của chính evaluator và đổi nó sẽ đổi cách chia batch feature — thứ không được phép đổi trong một so sánh checkpoint.

## 9. Files bàn giao

| path | nội dung |
|---|---|
| `/home/heheboiz/data/cir/CHECKPOINT_SELECTION_REPORT.md` | báo cáo này |
| `/home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700/configs/selected_overrides.yaml` | mapping candidate-level + absolute path + ghi chú chưa áp vào global |
| `/home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700/manifests/old_weight_inventory.json` | inventory 11 weight cũ (hash, provenance) |
| `/home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700/manifests/cleanup_result.json` | 11 file đã xoá / 13 421 279 616 byte |
| `/home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700/manifests/acquired_candidates.json` | byte + SHA-256 của 9 payload tải trong session |
| `/home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700/manifests/selected_bundle_digests.json` | SHA-256 của 11 file bundle được chọn |
| `/home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700/manifests/protocol_identity.json` | hash annotation/split + query/gallery count theo protocol |
| `/home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700/manifests/download_budget.json` | 22.27 GiB / 64 GiB |
| `/home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700/measurements/*.json` | số đo nguyên bản từng model (category + macro + provenance) |
| `/home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700/scripts/*.py` | runner session + script kiểm tra cấu trúc; `scripts/compat/sitecustomize.py` shim DCNet |


### 9.1 Đề xuất diff registry tối thiểu (chưa áp dụng)

`workbench/registry/models.yaml` — thay mapping/trạng thái cho 5 entry có bằng chứng đo trong session này:

```yaml
# combiner_rn50x4_noft
  checkpoint_variants:
    - checkpoint_id: rn50x4_noft_release
      filename: fiq_comb_RN50x4_noft.pt
      download_url: https://github.com/ABaldrati/CLIP4Cir/releases/tag/weights  # part-000..004
      source_type: author_github_release
      release_member: FashionIQ/RN50x4_noft/fiq_comb_RN50x4_noft.pt
      expected_size_bytes: 236017921
      expected_sha256: b31882fe6f7e3e194cb9cfda048926844ee0ccc9d4ebc5483fb3bacfa5972955
      checkpoint_training_noise_pct: 0
      evaluation_noise_pct: 0
      checkpoint_mapping_status: EMPIRICAL_VERIFIED
  reproduction_status: EMPIRICALLY_EVALUATED

# clip4cir_rn50x4_fullft  (bundle 2 state, cùng release)
      required_files: [fiq_comb_RN50x4_fullft.pt, fiq_clip_RN50x4_fullft.pt]
      sha256:
        fiq_comb_RN50x4_fullft.pt: 7c7c0a10fb4c995cb1ded9ad70a58891d50c622263c54131523bfb232b493f8e
        fiq_clip_RN50x4_fullft.pt: 4111ac139cf96ac0cf31b30bf8f55b654247b06f0f5fbee4fe8dcc8c1c7cfd2f

# dcnet
      checkpoint_mapping_status: EMPIRICAL_VERIFIED   # author Drive run dir; không có hash tác giả
      reproduction_status: EMPIRICALLY_EVALUATED

# clvc_net  (6 file, đúng archive member)
      checkpoint_mapping_status: EMPIRICAL_VERIFIED
      category_checkpoint_structure:
        mapping_status: VERIFIED_BY_ARCHIVE_MEMBER

# tgcir
      filename: Fashioniq.pt
      artifact_member: checkpoints/Fashioniq.pt
      expected_sha256: c37768f374c75bebff1299e870518c23bf6db3dc8a75c4d1f729ccd6dd02e966
      checkpoint_mapping_status: EMPIRICAL_VERIFIED

# sprc   (chỉ biến thể ViT-L; bản chính vẫn PENDING_USER_INPUT)
      - checkpoint_id: sprc_fiq_vitl
        checkpoint_mapping_status: EMPIRICAL_VERIFIED
        expected_sha256: 19386104271c01ea03ee2177575c0e55f98edd3ba61a116d78f2e61ab5dbea08
```

`preparation_contracts.yaml` — thêm generator cho GloVe (D-4):

```yaml
# trong dcnet_fashioniq.generated_artifacts
  - name: GloVe validation annotation PKLs (reproducible regeneration)
    generator: dcnet_glove_regen
    parameters:
      recipe_script: session scripts/dcnet_glove_prep.py (mirrors preprocess/process_cap.py)
      python: "3.7"
      spacy: "2.3.7"
      vectors_model: en_vectors_web_lg-2.1.0
      nltk: "3.5"
      seed: 0
      deterministic_status: SEEDED_NUMPY_OOV
```

## 10. Tóm tắt một dòng

8/11 candidate chạy xong native FashionIQ validation với số đo thật; PAIR bị chặn vì class nguồn không tồn tại công khai; SPRC bản chính chờ file OneDrive. Không có association nào được nâng lên author-verified ngoài AirKnow (SHA LFS khớp) và SPRC ViT-L (state dict khớp tuyệt đối).

