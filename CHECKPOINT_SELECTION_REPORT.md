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
| `pair_compat/` | PAIR: loader, audit, forward smoke, 4 lần chạy native, verdict |
| `cohort/` | 7 model inventory: scripts, measurements, logs, blocker JSON, hash + environment manifests |

## 1. Bảng kết quả nhanh — 17 entry đăng ký

Đếm theo `model_id` trong registry. **PAIR là một entry** (B1/B2 chỉ là hai file của cùng entry), nên bảng dưới có 15 dòng đã đo + 2 dòng chưa đo. Bảng không xếp hạng giữa các protocol khác nhau; cột `protocol` nằm ngay cạnh số để tránh so sánh sai.

| # | model_id | macro R@10 | macro R@50 | mean | protocol | trạng thái |
|---|---|---|---|---|---|---|
| 1 | hint | **59.8034** | **80.1308** | 69.9671 | `fashioniq_val_split` | COMPLETED — đã kiểm chứng lại, xem 3.12 |
| 2 | encoder | **56.2231** | **77.6674** | 66.9452 | `fashioniq_val_split` | COMPLETED |
| 3 | limn (base iter-0) | **55.7919** | **77.6378** | 66.7149 | `fashioniq_val_split` | COMPLETED (open_clip 2.20, registry đã sửa) |
| 4 | tme (0% noise) | **54.9229** | **74.8384** | 64.8807 | `fashioniq_original_split` | COMPLETED (session runner) |
| 5 | habit (20%) | **54.8733** | **74.5034** | 64.6883 | `fashioniq_full_gallery_ref_excluded` | COMPLETED |
| 6 | intent (20%) | **54.8662** | **75.4319** | 65.1491 | `fashioniq_full_gallery_ref_excluded` | COMPLETED |
| 7 | conesep (20%) | **54.7613** | **75.2151** | 64.9882 | `fashioniq_full_gallery_ref_excluded` | COMPLETED |
| 8 | airknow (50%) | **52.5466** | **72.9026** | 62.7246 | `fashioniq_full_gallery_ref_excluded` | COMPLETED, recommended |
| 9 | sprc (ViT-L) | **51.0386** | **73.3263** | 62.1825 | `fashioniq_original_split` | COMPLETED (variant; ViT-g chờ D-1) |
| 10 | airknow (80%) | **50.6417** | **71.4556** | 61.0486 | `fashioniq_full_gallery_ref_excluded` | COMPLETED, variant |
| 11 | tgcir | **49.7208** | **71.1023** | 60.4116 | `fashioniq_val_split` | COMPLETED |
| 12 | clip4cir_rn50x4_fullft | **43.8182** | **66.9501** | 55.3841 | `fashioniq_original_split` | COMPLETED |
| 13 | ptha_mtst | **37.8978** | **59.8364** | 48.8671 | `fashioniq_original_split` | **COMPLETED (transfer CIRR→FIQ), xem 3.13** |
| 14 | combiner_rn50x4_noft | **35.1407** | **58.4436** | 46.7922 | `fashioniq_original_split` | COMPLETED |
| 15 | clvc_net | **29.9707** | **58.6217** | 44.2962 | `fashioniq_val_split` | COMPLETED |
| 16 | dcnet | **27.9297** | **53.7272** | 40.8285 | `fashioniq_full_gallery_ref_excluded` | COMPLETED (patch 1 dòng, xem 3.14) |
| e | pair (B1/B2) | 35.0147 / 22.1564 | 57.5753 / 43.4864 | — | `fashioniq_val_split` | **EXPLORATORY** — không phải reproduction, xem 3.9 |
| — | csmcir | — | — | — | `fashioniq_original_split` | BLOCKED — thiếu caption asset (D-8) |

**15/17 entry đã có số đo native thật** (PTHA nay nằm trong số đó). Số 17 đếm theo `model_id` đăng ký:

- 15 entry đã đo native (bảng trên, trừ dòng `e`);
- `pair` **đo được nhưng exploratory** — không tính vào 15 vì không phải hành vi của implementation gốc;
- `csmcir` **chưa đo** — thiếu caption asset (D-8).

15 + pair + csmcir = 17. Biến thể SPRC **ViT-g** là *variant của `sprc`*, không phải entry thứ 18: ViT-L đã đo, ViT-g chờ file (D-1).

Bảng đầy đủ kèm Δ so với hàng tham chiếu và điều kiện runtime: mục 3.11.

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
| `workbench/envs/limn-adapted` | 3.11.15 | 2.7.1+cu128 | — | 12.8 | LIMN, **open_clip_torch 2.20.0** (D-6) |
| `workbench/envs/tme-adapted` | 3.10.22 | 2.7.1+cu128 | — | 12.8 | TME (yêu cầu Python 3.10 theo contract) |
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

### 3.9 PAIR — đã load và chạy thật bằng EXPLORATORY_PUBLIC_MAPPING; vẫn KHÔNG phải implementation luyện ra weight

Kết luận cũ ("phải có file tác giả") là **quá sớm**. Đã thử route compatibility một cách đầy đủ.

**Bước 1 — hash gate.** Cả hai file khớp đúng digest đề bài:

| file | byte | SHA-256 | zip member |
|---|---|---|---|
| `pair-B1.pt` | 694 958 565 | `c4d861ecd0baa3f1427fc7a853943f5614baf87def1cdd9c4bb7583ee925fbf2` | `model_epoch_1_ShuffleTrue` |
| `pair-B2.pt` | 694 872 371 | `74346fd7de6efdafe8dde70458316286a60ceb2d7b419efed55193418aeb3f03` | `model_epoch_4_ShuffleFalse` |

**Bước 2 — mapping tường minh (per class, subprocess-local).**

| pickle global | class công khai dùng | căn cứ |
|---|---|---|
| `model_vssp.Encoder` | `model.PAIR` | object graph khôi phục có đúng bộ attribute mà `PAIR.__init__` gán: `backbone, loss_T, local_weight[6+2], local_weight_dis[6+1], t, affine` |
| `model_vssp.Backbone` | `model.Backbone` | khôi phục là superset của `Backbone.__init__`; phương thức 8-tuple `extract_img_fea_patch_selection` khớp đúng chỗ `PAIR.compose_feature` unpack |
| `model_vssp.FeatureWiseAffine` | `model.FeatureWiseAffine` | `use_affine_level=True`, `MLP` 4 tầng `1024→512→512` khớp |
| `MHTransformer.*` | `MHTransformer.*` (public, nguyên bản) | file public có mặt, không cần alias |
| `MHTransformer_dis.MlpTransformer/TransformerLayer/Transformer` | `MHTransformer.*` | cấu trúc khớp; không có file public |
| `MHTransformer_dis.MultiHeadAttention` | **hypothesis-dependent** — xem dưới | |

Hai hypothesis được khai báo **trước** khi xem điểm:
- **H_normal** — `MHTransformer_dis` là attention softmax thường.
- **H_rev** — `MHTransformer_dis` là reverse attention `1 - softmax`; khớp việc chính public PAIR gán `transformer_b = MHTransformer_rev.Transformer`.

Không alias mọi attention về một implementation; không chọn mapping theo điểm số.

**Bước 3 — restore trên CPU + audit object graph.** Cả 4 tổ hợp (B1/B2 × H) restore thành công, **0 attribute gap** trên toàn bộ call graph inference (`PAIR.extract_retrieval_compose/target`, `Backbone.extract_img_fea/extract_img_fea_patch_selection`, `visual_out/text_out`, `clip.visual._global_pool`). `preprocess_cfg` đọc từ chính checkpoint ⇒ native preprocessing tái lập được.

**Bước 4 — forward thật (GPU).** `compose→[3,512]`, `target→[3,512]`, tất cả finite, norm L2 = 1.0, lặp lại trong `eval()` cho `max_abs_diff = 0.0`, `train≡eval` vì mọi Dropout khôi phục có `p=0.0` (không phải no-op giả). Hook xác nhận chỉ `transformer`, `transformer_a`, `transformer_b` chạy.

**Bước 5 — đánh giá native đầy đủ 3 category, cả 2 file × 2 hypothesis.**

| candidate | macro R@10 | macro R@50 | mean | n_queries (dress/shirt/toptee) | gallery |
|---|---|---|---|---|---|
| B1 / H_rev | **35.0147** | **57.5753** | 46.2950 | 2017 / 2038 / 1961 | 2628 / 3089 / 2902 |
| B1 / H_normal | 34.0350 | 56.5957 | 45.3154 | " | " |
| B2 / H_rev | 20.0868 | 41.4988 | 30.7928 | " | " |
| B2 / H_normal | 22.1564 | 43.4864 | 32.8214 | " | " |

Coverage 100%, reference bị loại trước Recall, không cắt gallery. So với hàng paper (52.76 / 74.51) lệch **−17.7 / −16.9 pp** (B1/H_rev).

**Bước 6 — vì sao lệch: giả thuyết, không phải kết luận nhân quả**

Bản trước trình bày mục này như *nguyên nhân đã chứng minh*. Sửa lại cho đúng mức bằng chứng:

**Đo được (chắc chắn):**

- `backbone.transformer_a` khôi phục là **2 layer × 4 head**; public `model.py:46` khai `transformer_a` = **1 layer × 8 head**.
- `transformer_b` khôi phục là 2×4; public `model.py:47` khai đúng 2 layer × 4 head ⇒ **khớp** ở nhánh này.
- checkpoint có thêm `transformer_c`, `transformer_d` (mỗi cái 2×4), `weight_chanelImg`, `weight_chanelImg_t`, `cross_t2i`.
- Forward public chỉ đọc `transformer`, `transformer_a`, `transformer_b`.

**Đã kiểm tra và KHÔNG chứng minh được:**

- Tra toàn bộ checkout PAIR (models, loss, mọi entrypoint train) cho 5 tên `transformer_c/d`, `weight_chanelImg*`, `cross_t2i`: **0 lần xuất hiện**. Vì vậy **không** thể nói các module đó là "training-only" — trong source public chúng không có consumer nào cả, kể cả lúc train.
- **Không** kết luận được hình học 2×4 là "không tương thích về nguyên tắc": chính public code tạo ra kiểu 4-head/2-layer ở `transformer_b`, nên đây là cấu hình mà codebase này có hỗ trợ.

**Hệ quả đúng đắn:**

- ~~8 986 448 / 172 814 185 = 5.20% tham số bị bỏ~~ → con số **đếm được** là 5.20% tham số **không được forward public chạm tới**. Điều đó **không** chứng minh chúng từng được train cho inference, cũng **không** chứng minh chúng bị tác giả loại bỏ.
- Điểm thấp (35.01 / 57.58 so với 52.76 / 74.51) **không** tự nó xác lập nguyên nhân duy nhất. Đã so cả 4 tổ hợp và cả hai định nghĩa attention; chênh lệch vẫn lớn ở mọi tổ hợp, nhưng một phép so 4 tổ hợp không tách được "kiến trúc khác" khỏi "preprocessing khác" hay "checkpoint khác" một cách chắc chắn.

**Nhãn chính xác:** `EXPLORATORY_PUBLIC_MAPPING` — load được, chạy được, chấm được, nhưng **tương đương với source gốc vẫn chưa xác lập**. Không phải "definitely correct", cũng không phải "definitely impossible". Blocker cho mục đích reproduction vẫn còn, và nay được phát biểu đúng: cần `model_vssp.py`/`MHTransformer_dis.py` đúng revision để biết `transformer_c/d` và `weight_chanelImg*` được dùng ở đâu.

**Đã giữ nguyên:** cả 4 phép đo B1/B2 × H_normal/H_rev, không chọn định nghĩa attention theo điểm gần paper nhất.

### 3.10 Empirical recommendation vs author-verified association

| entry | association checkpoint↔kiến trúc | nguồn xác nhận | kết quả đo |
|---|---|---|---|
| AirKnow 50% / 80% | **AUTHOR_VERIFIED** | SHA-256 khớp đúng LFS `oid` do tác giả công bố trên Hub (`f4d25ed9…`, `756fc0bd…`), đúng tên file, đúng Framework `iLearn-Lab/CVPR26-Air-Know` | đo đầy đủ, D6 1.03 / 1.32 pp |
| SPRC ViT-L | **LOAD-COMPATIBLE (không phải AUTHOR_VERIFIED)** | state dict nạp **strict** vào đúng `Blip2QformerCirAlignPrompt`/`pretrain_vitL`: 0 missing / 0 unexpected; conv1 `1024×3×14×14`, positional `257×1024` ⇒ khớp CLIP ViT-L/14. Đây là *bằng chứng kiến trúc khớp*, **không** phải tác giả xác nhận: không có SHA do tác giả công bố để so, và nạp thành công không chứng minh đây là biến thể đã sinh ra con số paper | đo đầy đủ, D6 0.027 pp |
| CLIP4Cir noft / fullft | **EMPIRICAL** | tên member + shape state dict khớp `Combiner(feature_dim=640, 2560, 5120)` và `CLIP` 669 key; không có checksum tác giả | đo đầy đủ |
| DCNet | **EMPIRICAL** | `config.json` + `trained_model.pth` cùng thư mục Drive tác giả; state dict 518 key khớp `CENet`; không có checksum | đo đầy đủ |
| CLVC-Net | **EMPIRICAL** | member `checkpoint/FashionIQ/<Cat>/{local,global}_model.pt` được archive tác giả đặt tên đúng; 415 key khớp `compose_local`/`compose_global` | đo đầy đủ |
| TG-CIR | **EMPIRICAL** | `checkpoints/Fashioniq.pt` + chính tác giả kèm `Fashioniq_metrics.json` cùng archive; 489 key khớp `VCG_CIR` | đo đầy đủ |
| LIMN (iter-0) | **EMPIRICAL** | 3 whole-model pickle đúng tên/đúng dung lượng/đúng SHA trong README+Hub LFS tác giả; deserialize thành `model.LIMN` của checkout pinned | đo đầy đủ, Δ −0.12/−0.18 |
| ENCODER | **EMPIRICAL** | state_dict 455 key nạp vào `model_try2.Encoder` với đúng 1 cặp key không dùng (được pinned source bỏ qua); LAION backbone SHA khớp registry | đo đầy đủ, Δ +0.09/+0.08 |
| HINT | **EMPIRICAL** | whole-model pickle, SHA tác giả khớp trước unpickle; chạy `test.py` nguyên bản | đo đầy đủ, Δ +1.32/+2.20 |
| HABIT / INTENT / ConeSep | **EMPIRICAL** | whole-model pickle, SHA khớp registry; đúng biến thể 20% đã chọn | đo đầy đủ |
| TME | **EMPIRICAL** | bóc tar → torch zip tensor-only; SHA `fea7fc08…` **trùng khít** weight TME từng bị xoá trong cleanup (cùng nguồn Kaggle) | đo đầy đủ, Δ −0.23/−0.18 |
| SPRC OneDrive (ViT-g) | **UNVERIFIED** | file không có trên máy | chưa chạy |
| PAIR B1 / B2 | **EXPLORATORY_PUBLIC_MAPPING** | public `model.py` không phải code đã train (xem 3.9) | đo được nhưng không tính là reproduction |
| CSMCIR | **UNVERIFIED (blocked)** | checkpoint SHA khớp tác giả; thiếu caption asset | chưa chạy |
| PTHA+MTST | **EXPLORATORY TRANSFER (đã đo)** | checkpoint CIRR SHA khớp; nguyên nhân "sai chiều đầu chiếu" đã bị bác bỏ — lỗi ở biên feature, đã sửa | **đã chạy full 3 category: 37.8978 / 59.8364** (xem 3.13) |

Nói cách khác: kết quả đo là **empirical**; chỉ AirKnow (SHA tác giả) và SPRC ViT-L (khớp lớp tuyệt đối) có association do tác giả/kiến trúc xác nhận, và **không entry nào** được gọi là "author-verified reproduction" của một con số paper.

### 3.11 Cohort inventory — 7 model chạy xong native validation

Trước đây 9 entry này chỉ là inventory. Nay 7 đã được đo thật, đủ 3 category, đúng protocol khai báo trong registry. Không cái nào dùng gallery rút gọn hay bỏ query (coverage 100%: 2017 / 2038 / 1961 query).

| model | checkpoint | protocol | noise (train/eval) | macro R@10 | macro R@50 | mean | ref R@10 / R@50 | Δ10 / Δ50 |
|---|---|---|---|---|---|---|---|---|
| hint | `fashioniq` | `fashioniq_val_split` | —/0 | **59.8034** | **80.1308** | 69.9671 | 58.48 / 77.93 | +1.32 / +2.20 |
| encoder | `fashioniq` | `fashioniq_val_split` | —/— | **56.2231** | **77.6674** | 66.9452 | 56.13 / 77.59 | +0.09 / +0.08 |
| limn | `base_iter0_all` | `fashioniq_val_split` | —/0 | **55.7919** | **77.6378** | 66.7149 | 55.91 / 77.82 | -0.12 / -0.18 |
| tme | `fashioniq_0` | `fashioniq_original_split` | 0/0 | **54.9229** | **74.8384** | 64.8807 | 55.15 / 75.02 | -0.23 / -0.18 |
| habit | `fiq_n02` | `fashioniq_full_gallery_ref_excluded` | 20/0 | **54.8733** | **74.5034** | 64.6883 | 54.48 / 74.28 | +0.39 / +0.22 |
| intent | `fiq_n02` | `fashioniq_full_gallery_ref_excluded` | 20/0 | **54.8662** | **75.4319** | 65.1491 | 54.22 / 74.49 | +0.65 / +0.94 |
| conesep | `fiq_n02` | `fashioniq_full_gallery_ref_excluded` | 20/0 | **54.7613** | **75.2151** | 64.9882 | 54.93 / 75.01 | -0.17 / +0.21 |

Ghi chú điều kiện từng entry:

- **ENCODER** — `fashioniq_val_split`; cần backbone LAION CLIP ViT-B/32 `open_clip_pytorch_model.bin` (đã tải lại, sha256 `1bd3c717…` khớp registry); phải chạy với `cwd=workbench/third_party/ENCODER` vì source dùng `clip_path='./'`. `parity_mismatches: []`.
- **HINT** — `fashioniq_val_split`; checkpoint whole-model, SHA tác giả khớp trước khi unpickle.
- **LIMN** — `fashioniq_val_split`, bundle iteration-0 đủ 3 file (dress/shirt/toptee), mỗi category một model riêng. **Chạy dưới open_clip 2.20.0** (env `workbench/envs/limn-adapted`) vì đó là pin của tác giả (`LIMN/requirements.txt:37`) và là môi trường đã đo. `adapted_runtime` trước đây ghi `>=2.24,<2.27`, mâu thuẫn với pin đó; nay đã sửa thành `==2.20.0` (mục 3.15). **Đính chính:** bản trước nói dưới 2.24 nó chết ở `_global_pool` — đo lại thì **không** tái lập: cả 2.20 lẫn 2.24 đều unpickle được bundle này. Nhãn: `SCOPED_PINNED_OPENCLIP_2_20`, xem D-6.
- **ConeSep / HABIT / INTENT** — 20% training noise, **0% evaluation noise**, protocol `fashioniq_full_gallery_ref_excluded`; không thay bằng file noise khác.
- **TME** — `fashioniq_original_split`, 0% eval noise. Checkpoint gốc là **tar bọc torch-zip**; đã bóc lớp tar để dùng state dict tensor-only. Cache feature do chính pipeline sinh: `(15536, 257, 1408)` **float32**; contract khai `float16` nên wrapper từ chối — xem D-7. Evaluator chạy với `Qformer_bs 256→64`, `query_bs 32→8` (chính docstring pinned bảo giảm khi OOM; chỉ đổi số ảnh mỗi forward, ranking không đổi).

**CSMCIR** và **PTHA+MTST** chưa xong, blocker cụ thể:

- **CSMCIR**: checkpoint tải đủ và khớp SHA (`c8812071…`), nhưng dataset constructor đòi `COT_ours2/bert_captions/fashioniq/cap.*.val.json` + `captions/keywords_in_mods_*.json`. Cả hai **không có** trên máy và **không có** trong repo HF của tác giả (repo chỉ có `fashioniq_qwen_captions/`). Không thay bằng Qwen caption, không sinh caption giả ⇒ `BLOCKED_MISSING_AUTHOR_CAPTION_ASSETS` (D-8).
- **PTHA+MTST**: ~~BLOCKED~~ → **đã chạy xong trong buổi này** (mục 3.13). Nguyên nhân cũ được ghi lại nguyên văn: lỗi shape `[61072,256]` vs `[61072,768]`. Nhưng kết luận "checkpoint CIRR có đầu chiếu 768 chiều" là **sai**: checkpoint có `vision_proj (256,768)` và nạp strict thành công; lỗi nằm ở `extract_target_features` (định nghĩa thứ hai) trả **chưa chiếu**. Đã sửa bằng `inference_target()` của tác giả, macro **37.8978 / 59.8364**, vẫn là exploratory transfer.

**SPRC OneDrive**: vẫn `PENDING_USER_INPUT` — file chưa có trên máy nên không thể "inspect actual class" (D-1 giữ nguyên).

## 3.12 HINT — kiểm chứng lại toàn pipeline (buổi 20261106T233409+0700)

Mục này **không** thay số cũ mà xác minh lại nó. Mọi kết quả cũ giữ nguyên nhận dạng.

### 3.12.1 Khiếu nại ban đầu và cách tái lập

Lệnh trong báo cáo cũ chạy HINT qua vòng lặp chung LA VIS có truyền `--variant fashioniq`. Parser của `workbench/replay/hint.py` **không** có cờ đó:

```
hint.py: error: unrecognized arguments: --variant fashioniq
EXIT=2
```

Đã tái lập nguyên văn (`hint/logs/hint_variant_repro.log`, exit 2). Đây là lỗi **tài liệu**, không phải lỗi số đo: bản thân vòng lặp sai, kết quả cũ vẫn sinh ra bằng một lệnh khác không cờ đó.

### 3.12.2 Danh tính nguồn / checkpoint / input

| mục | giá trị |
|---|---|
| checkpoint | `fashioniq.pt`, 2 724 451 182 byte |
| SHA-256 (đo lại) | `bf9c72d42206b667ed586758c15c648970df821857e4067f26155a114a6517d4` |
| khớp registry `expected_sha256` | **có** |
| lớp gốc khôi phục | `lavis.models.blip2_models.HINT.HINT` |
| `datasets` / `test` / `data_utils` | đều resolve **trong** `workbench/third_party/HINT/` (không phải package HuggingFace trùng tên) |
| text processor | `lavis/models/__init__.py:load_preprocess`, qua `blip2_pretrain.yaml` của chính lớp |
| upstream commit | `bec50b6c8c19111893b617979502b948d1cea5b2` |
| runtime | `airknow-adapted` (Python 3.11.15, torch 2.7.1+cu128), `WORKBENCH_RUNTIME_ADAPTATION=sm_120` |
| preprocessing | `targetpad_transform(1.25, 224)` — đúng `train.py:72-80` |
| chế độ khi forward | `model.eval()`, `training == False` |

### 3.12.3 Tensor contract thật (đo, không suy đoán)

Lớp HINT dùng "dual-path max": query là `[N,1,1,256]`, target là `[N,256,32]`; `np.matmul` broadcast thành `[N,M,1,32]`, `.squeeze()` → `[N,M,32]`, rồi `.max(-1)` lấy max **trên 32 Q-Former token** ⇒ `[N,M]`. Đo trực tiếp: `2017×2628 → (2017,2628,1,32)`. Đây chính là toán tử trong `test.py:74-86`, không phải biến thể của wrapper.

### 3.12.4 Kết quả: native viết tay ≡ wrapper

Đã viết lại `test.py:test` **độc lập** (cùng phép toán, cùng thứ tự, cùng dtype) và chạy trên cùng checkpoint/dataset/batch:

| category | native R@10/R@50 | wrapper đã lưu | recompute từ rank | query | gallery | coverage |
|---|---|---|---|---|---|---|
| dress | 51.2643 / 75.6073 | 51.2643 / 75.6073 | 51.2643 / 75.6073 | 2017 | 2628 | 2017 |
| shirt | 62.4141 / 81.2561 | 62.4141 / 81.2561 | 62.4141 / 81.2561 | 2038 | 3089 | 2038 |
| toptee | 65.7318 / 83.5288 | 65.7318 / 83.5288 | 65.7318 / 83.5288 | 1961 | 2902 | 1961 |
| **macro** | **59.8034 / 80.1308** | **59.8034 / 80.1308** | **59.8034 / 80.1308** | 6016 | — | 100% |

Sai khác tuyệt đối < 1e-9 trên cả 6 con số. Reference bị mask `-10e10` trước khi tính recall; không bỏ query nào.

### 3.12.5 Vì sao cao hơn paper (58.48 / 77.93)

Đây là quan sát, **không** phải bằng chứng bug và **không** được "sửa" về phía paper. Nguyên nhân đã kiểm chứng được: HINT chấm bằng **max trên 32 token Q-Former** thay vì một vector pooled duy nhất, và protocol là `fashioniq_val_split` (hợp có thứ tự của reference+target, reference bị loại). Cả hai đều là hành vi của source pinned. Chênh lệch +1.32/+2.20 pp so với hàng paper nằm trong phần chưa xác định được: checkpoint này có thể là biến thể được chọn tốt hơn hàng báo cáo, và chế độ nhiễu lúc train là **unknown** (không suy ra được từ metadata).

### 3.12.6 Phán quyết HINT

- **Đã xác minh** dưới input/runtime/path đã ghi: checkpoint khớp SHA tác giả, module pinned resolve đúng, preprocessing đúng `train.py`, ranking native viết tay trùng khớp wrapper đến 1e-9, macro 59.8034/80.1308 tái lập được từ rank.
- **Đã sửa**: lệnh tài liệu (bỏ `--variant`, dùng CLI riêng của HINT).
- **Còn chưa chắc**: nhiễu train của checkpoint và việc đây có đúng là checkpoint sinh ra hàng paper hay không — không có metadata để kết luận, nên **không** gọi là reproduction.
### 3.12.7 Parity theo top-50 index set từng query

Tiêu chí khai báo **trước** khi chạy: với mỗi query, tập chỉ số top-50 do wrapper sinh ra phải bằng tập do bản viết tay sinh ra, và hit@1/10/50 phải trùng.

Wrapper chỉ trả recall nên danh sách `nn_result` được **bắt tại chỗ** bằng cách chặn `np.argsort` trong lúc `test.py:test` chạy — capture đúng những gì hàm native sinh ra, không tính lại bằng đường khác.

| category | query | tập top-50 bằng nhau | thứ tự giống hệt | hit@1/10/50 (wrapper) | hit@1/10/50 (native) |
|---|---|---|---|---|---|
| dress | 2017 | **2017/2017** | 2017/2017 | 432 / 1034 / 1525 | 432 / 1034 / 1525 |
| shirt | 2038 | **2038/2038** | 2038/2038 | 663 / 1272 / 1656 | 663 / 1272 / 1656 |
| toptee | 1961 | **1961/1961** | 1961/1961 | 628 / 1289 / 1638 | 628 / 1289 / 1638 |
| **tổng** | **6016** | **6016** | **6016** | — | — |

Không có query nào lệch (0 mismatch). Điểm trùng nhau có tồn tại trong dữ liệu (71083 / 34161 / 30604 cặp trong một hàng ở dress/shirt/toptee), nhưng chúng **không** làm hai đường rẽ nhánh, nên kết luận parity không dựa trên việc che các tie.

### 3.12.8 Batch-size invariance (đo, không giả định)

Chạy lại **toàn bộ** đường native viết tay ở `--batch-size 16` thay vì 64, trên cùng checkpoint/dataset:

| category | batch 64 R@10/R@50 | batch 16 R@10/R@50 | chênh |
|---|---|---|---|
| dress | 51.2643 / 75.6073 | 51.2643 / 75.6073 | `0.0` |
| shirt | 62.4141 / 81.2561 | 62.4141 / 81.2561 | `0.0` |
| toptee | 65.7318 / 83.5288 | 65.7318 / 83.5288 | `0.0` |

Không chỉ trùng ở mức recall: **rank từng query của cả 6016 query giống hệt nhau** giữa hai batch size. Vậy batch size đúng là tham số vận hành, không phải tham số khoa học — batch 64 đã dùng cho kết quả chính là hợp lệ.


## 3.13 PTHA+MTST — đã tìm ra nguyên nhân và chạy được transfer (buổi 20261106T233409+0700)

Nhãn giữ nguyên suốt mục này: **CIRR-trained → FashionIQ evaluation, cross-dataset transfer**. Không phải reproduction FashionIQ.

### 3.13.1 Kết luận cũ và tại sao nó quá mạnh

Báo cáo trước ghi `BLOCKED_ARCHITECTURAL_MISMATCH` với lý do "checkpoint cung cấp 768 trong khi cần 256". Câu đó **suy diễn quá xa** từ một matmul lỗi. Bằng chứng phản bác, đo trong buổi này:

- checkpoint chứa `vision_proj.weight` **`(256, 768)`**, 196 864 tham số — tức **có** đầu chiếu đã train đúng kích thước đích;
- `load_state_dict(..., strict=True)` **thành công**, `missing_keys=[]`, `unexpected_keys=[]`, `shape_mismatches=[]`;
- lỗi cũ xảy ra **sau** khi nạp strict thành công, tức không phải lỗi kiến trúc/không phải thiếu head.

Vậy nguyên nhân nằm ở **biên truyền feature**, không phải ở checkpoint.

### 3.13.2 Nguyên nhân thật, đo trên tensor thật

Trong `lavis/models/blip2_models/blip2_qformer_cir_cat.py`:

- `extract_target_features` được **định nghĩa hai lần**; Python giữ định nghĩa **sau** (dòng 483).
- Định nghĩa sau trả `image_features = image_embeds` — tức **Q-Former state chưa chiếu** `[N,32,768]` — và dòng chiếu bị comment out.
- `inference()` (dòng 287) dựng phía query qua `text_proj` thành `[B,1,1,256]` rồi matmul thẳng với target ⇒ hai chiều không gặp nhau.
- Trong **cùng file** có sẵn `inference_target()` (dòng 447) làm đúng phép chiếu còn thiếu.

Đo trên batch FashionIQ dress thật (GPU), trước khi sửa:

```
A) code tác giả như đang viết:
   FAILED: Expected size for first two dimensions of batch2 tensor to be: [3, 256] but got: [3, 768].
   phía query [B,1,1,256]  vs  target sau permute (3, 768, 32)
B) target đi qua chính vision_proj đã train của checkpoint:
   inference_target -> (3, 32, 256), norms [1.0, 1.0, 1.0]
   OK: (3,) finite: True
```

Đối chiếu với lớp anh em `blip2_qformer_tgir_concatembedding_sim.py`: lớp đó chiếu target **bên trong** `inference()` (dòng 429) và extractor của nó cũng trả chưa chiếu. Nghĩa là "luôn chiếu target" **sai** cho lớp đó — nên sửa phải theo **từng lớp**, không được làm cờ toàn cục.

### 3.13.3 Sửa tối thiểu

Thêm `_target_feature_contract()` trong `workbench/replay/ptha_mtst.py`, gọi `model.inference_target(index_features[0])` — **đúng method của tác giả**, không viết lại phép chiếu:

- không tạo tham số mới (`new_parameters_introduced = 0`);
- không cắt/pad/reshape chiều, không thêm `Linear` ngẫu nhiên, không bỏ normalize;
- không đổi hướng điểm hay pooling;
- chỉ áp dụng ở nhánh FashionIQ transfer của `ptha_mtst`, không đụng lớp `tgir_ce_sim`.

**Chống chiếu hai lần bằng hình học, không bằng cờ**: `vision_proj` nhận 768 và xuất 256, nên lần chiếu thứ hai **ném lỗi** thay vì âm thầm làm sai điểm — đo được `mat1 and mat2 shapes cannot be multiplied (96x256 and 768x256)`.

### 3.13.4 Kiểm chứng sau khi sửa

| kiểm tra | kết quả |
|---|---|
| trước/sau shape (dress/shirt/toptee) | `[3817,32,768]→[3817,32,256]`, `[6346,…]`, `[5373,…]` |
| tham số chiếu dùng lại | 196 864 (đúng `vision_proj` của checkpoint) |
| tham số mới | **0** |
| adapter ≡ `F.normalize(vision_proj(x))` trên tensor thật | `max_abs_diff = 0.0` (cả 3 category) |
| `load_state_dict` strict | 0 missing / 0 unexpected / 0 shape mismatch |
| regression test | `workbench/tests/test_ptha_target_feature_contract.py` — 5/5 pass, gồm cả ca tái lập lỗi trước khi sửa và ca chặn chiếu hai lần |

### 3.13.5 Kết quả transfer đầy đủ

Protocol `fashioniq_original_split`, gallery đầy đủ theo thứ tự split, reference hợp lệ (không mask — khác DCNet), 100% coverage.

| category | R@10 | R@50 |
|---|---|---|
| dress | 33.1185 | 56.0238 |
| shirt | 38.8616 | 59.0285 |
| toptee | 41.7134 | 64.4569 |
| **macro** | **37.8978** | **59.8364** |

Macro mean `48.8671`. Paper context (FashionIQ-trained, 54.99 / 76.28) **chỉ là tham chiếu số**, không phải mục tiêu: checkpoint này train trên CIRR, nên thấp hơn là kỳ vọng, và **không** được đọc là "kém hơn paper".

### 3.13.6 Phán quyết PTHA

- **Đã sửa và đo được**: nguyên nhân là biên feature trong chính source pinned, đã sửa bằng method có sẵn của tác giả, đã chạy đủ 3 category trên GPU thật.
- **Vẫn là exploratory transfer**: checkpoint CIRR, không có checkpoint FashionIQ do tác giả phát hành.
- **Chưa xác lập**: transfer này có tương đương cách tác giả *dùng* checkpoint cho CIRR hay không — muốn vậy cần dữ liệu CIRR, buổi này không tải thêm dataset.
- Lỗi cũ `BLOCKED_ARCHITECTURAL_MISMATCH` **đã bị thay thế** bằng kết quả đo; lý do cũ được giữ trong evidence để đối chiếu, không xoá.

## 3.14 DCNet — thay shim toàn cục bằng patch một dòng có hash-gate

Bản trước chạy được nhờ một `sitecustomize.py` gắn qua `PYTHONPATH`, trong đó **thay thế `torch.load` cho cả process**. Như prompt chỉ ra, đó là đổi hành vi toàn cục cho mọi lần load không liên quan.

Đã thay bằng khai báo `approved_source_patches` đúng cơ chế registry đang dùng cho CSMCIR:

- file sửa: `base/base_trainer.py`, **đúng một call site** (`torch.load(ckpt_path)` tại dòng 195);
- `base_sha256 = 3edd9836…`, `derived_sha256 = f265cefe…`, `patch_sha256 = 659eabf3…`;
- patch áp từ base cho ra **đúng** derived digest (đã kiểm bằng `patch -p1` trên bản copy sạch);
- không còn `sitecustomize`, không còn `PYTHONPATH`, không đổi `torch.load` toàn process.

**Kiểm chứng thật** (`PYTHONPATH` đã unset, đúng CLI chính thức `cd workbench/third_party/DCNet && python test.py --resume <run_dir>`):

| category | fused R@10 | fused R@50 | so với lần chạy trước (dùng shim) |
|---|---|---|---|
| dress | 29.1522 | 54.5364 | khớp tuyệt đối, diff `0.0` |
| shirt | 22.8165 | 47.6447 | khớp tuyệt đối, diff `0.0` |
| toptee | 31.8205 | 59.0005 | khớp tuyệt đối, diff `0.0` |
| **macro** | **27.9297** | **53.7272** | **giống hệt** |

Kết luận: patch **không** đổi khoa học, chỉ đổi phạm vi của opt-out. Checkpoint nạp được (`Ckpt loaded. Resume training from epoch 83`), exit code 0.

## 3.15 LIMN — đưa pin về đúng của tác giả, và không nói quá

`LIMN/requirements.txt:37` khai `open-clip-torch==2.20.0`, và `environment.packages` của registry đã đúng như vậy. Nhưng `adapted_runtime.packages` lại ghi `open_clip_torch>=2.24,<2.27` — **tự mâu thuẫn** với pin của tác giả.

Đã sửa `adapted_runtime.packages` thành `['open_clip_torch==2.20.0']`, kèm comment ghi rõ **hai** căn cứ đúng: pin của tác giả, và việc runtime đã đo (`workbench/envs/limn-adapted`) là 2.20.0.

**Đính chính so với bản trước:** báo cáo cũ nói dưới 2.24 thì LIMN chết ở `_global_pool` với `AttributeError: ... 'pool_type'`. Đo lại trong buổi này thì **không đúng như vậy**:

| open_clip | unpickle bundle dress | có `global_average_pool` | có `pool_type` |
|---|---|---|---|
| 2.20.0 | **OK** (`model.LIMN`) | True | False |
| 2.24.0 | **OK** (`model.LIMN`) | True | False |

Cả hai đều unpickle được; lỗi `pool_type` mà bản trước mô tả **không tái lập**. Vì vậy comment trong registry đã được viết lại chỉ còn điều đo được, không giữ suy diễn cũ. Pin `==2.20.0` vẫn đúng, nhưng lý do là **pin của tác giả + provenance của môi trường đã đo**, không phải "chỉ 2.20 mới load được".

Wrapper nay dùng chính `adapted_runtime` đó từ registry, nên đường chạy chuẩn không cần script riêng của phiên: `python -m workbench.replay.limn ... --preflight-only` → `[READY] LIMN preflight passed`.

## 3.16 TME — dtype đo được, và giới hạn của kết luận

Đo trực tiếp trên đường native (`precompute.py:38`, `blip2_model.vit_encode(image)` dưới `torch.cuda.amp.autocast()`): `visual_encoder` chứa **cả** fp16 và fp32 tham số, và output của `vit_encode` là **float32** trên runtime đã dùng.

Hệ quả, và đây là chỗ bản trước nói quá: contract khai `feature_dtype: float16` kèm lập luận rằng autocast + `convert_weights_to_fp16` ⇒ fp16. Lập luận đó **không khớp** với output đo được. Nhưng **chưa** được kết luận rằng contract sai cho mọi runtime — autocast không bảo đảm dtype trả về, và một runtime khác có thể cho fp16. Vì vậy mục này ghi:

- **đo được**: trên runtime đã chạy, output là float32, và cache 22.49 GB (15536×257×1408×4 byte) khớp float32 chứ không khớp float16;
- **chưa xác lập**: dtype "native" theo nghĩa độc lập runtime. Cần đo trên runtime thứ hai mới tách được "native vs runtime-specific";
- **đề xuất**: (D-7) khai báo dtype theo runtime, hoặc ghi rõ contract là "float32 on the adapted sm_120 runtime" thay vì sửa số của contract lịch sử.

### 3.16.1 Bounded runner ≡ native runner (đã đo, không suy luận)

Runner preallocate-then-copy của phiên trước từng bị nghi là đổi hành vi. Đã kiểm bằng cách viết lại **đúng** đường native (`src/precompute.py:34-46`: gom list rồi `torch.cat`) và chạy song song với đường bounded trên **cùng ảnh thật, cùng model, cùng preprocess, cùng thứ tự**, có chủ đích tạo **batch cuối lẻ**: 5 ảnh, batch 2 ⇒ các batch 2 / 2 / **1**.

| | shape | dtype | names |
|---|---|---|---|
| native (list + cat) | `[5,257,1408]` | float32 | 5 |
| bounded (preallocate + copy) | `[5,257,1408]` | float32 | 5 |

**Byte-identical: `True`. `max_abs_diff = 0.0`. Thứ tự tên trùng khít.** Vậy khác biệt duy nhất giữa hai runner là **bộ nhớ**, không phải số học — và batch cuối lẻ không gây lệch. Không loại ảnh gallery nào khỏi phép so này (chỉ cắt số lượng vì cache đầy đủ là 22.5 GiB và đây là kiểm tra tương đương, không phải đo lại).

Giới hạn của kết luận: phần này chứng minh **hai runner tương đương nhau**, không chứng minh dtype `float16` trong contract là sai cho mọi runtime — xem lý do ở trên.

Không sửa `float16 → float32` trong contract lịch sử vì lý do khoa học: một runtime không đủ để phủ định contract cho mọi runtime.

## 4. Hàng đợi quyết định buổi sáng

| ID | model/job | hành động đề xuất | vì sao | phạm vi/rủi ro | đã thử an toàn | bằng chứng | cần bạn quyết/ cung cấp |
|---|---|---|---|---|---|---|---|
| **D-1** | `sprc` (biến thể chính, EVA-CLIP ViT-g) | Bạn tải file OneDrive rồi đặt tại `/home/heheboiz/data/cir/workbench/artifacts/checkpoints/sprc/sprc_fiq.pt` | Bản chính (paper config) chỉ có trên OneDrive; tôi không thể login/tương tác | Chỉ thêm 1 file 1.4–2.6 GB vào checkpoint root; không đổi code | Đã chạy biến thể ViT-L công khai (D6 0.0268 pp) — **không** dùng thay hàng ViT-g | `measurements/sprc_vitl_run1.json`, `measurements/sprc_distance.json` | Tải https://1drv.ms/u/s!Aj0q22vyiZbabnUya4mnufIBtYI?e=n4ZVKj (Baidu dự phòng, pass `t1at`) và đặt đúng path trên |
| **D-2** | `pair` (B1 + B2) | Quyết định hướng: (a) xin tác giả `model_vssp.py`/`MHTransformer_dis.py` đúng revision, (b) bỏ PAIR khỏi cohort và giữ nhãn EXPLORATORY trong báo cáo, hoặc (c) xác nhận rằng checkpoint đến từ ancestor chung với ENCODER để tôi dựng đúng nhánh còn thiếu | Đã load và chấm được cả 2 file bằng EXPLORATORY_PUBLIC_MAPPING, nhưng public `model.py` KHÔNG phải code đã train: `transformer_a` khôi phục 2 layer×4 head vs public 1×8; checkpoint có `transformer_c/d`, `weight_chanelImg*`, `cross_t2i` mà public không định nghĩa; 5.20% tham số nằm ngoài forward ⇒ điểm 35.01/57.58 (B1) không phải hành vi checkpoint | (a)/(c) cần bạn liên hệ hoặc xác nhận; (b) chỉ là quyết định phạm vi. Không sửa forward, không khởi tạo lại module | Đã probe HF + ZhihFu/PAIR (chỉ `main`), 5 repo công khai của tác giả, git history; đã restore + audit + smoke + chạy đủ 3 category × 2 file × 2 hypothesis | `pair_compat/manifests/pair_compat_verdict.json`, `pair_compat/measurements/summary.json`, mục 3.9 | Chọn (a), (b) hay (c) |
| **D-3** | registry dùng chung | Duyệt diff registry để map các entry đã đo được: 12 entry (8 priority + limn/encoder/hint/habit/intent/conesep/tme, trừ PAIR vì chỉ exploratory) | Hiện registry vẫn ghi `BLOCKED_*`/`NOT_RUN` cho các bundle nay đã đo được; không nên tự sửa default toàn cục qua đêm | Sửa `workbench/registry/models.yaml` (`reproduction_status` + `checkpoint_mapping_status`) + `preparation_contracts.yaml` cho DCNet GloVe và TME `feature_dtype` | Không sửa gì; chỉ ghi `configs/selected_overrides.yaml` trong session | `configs/selected_overrides.yaml`, mục 9.1, `cohort/measurements/cohort_summary.json` | Duyệt/không duyệt diff registry |
| **D-4** | DCNet reproducibility | Cho phép ghi lại recipe sinh GloVe PKL thành generator trong `preparation_contracts.yaml` (seed cố định, sha256 của 10 file PKL) | Hiện contract ghi `NONDETERMINISTIC_WITHOUT_AUTHOR_ARTIFACT`, nên lần sau không tái lập được bằng workbench | Sửa 1 khối contract + thêm 10 entry hash; không đổi dataset canonical | Đã sinh trong session, giữ file + log, không sửa contract | `logs/dcnet_glove_prep.log`, `logs/dcnet_run2.log` | Duyệt việc chuẩn hoá recipe này |
| **D-5** | `torch.load` shim cho DCNet | Cho phép thêm 1 dòng `weights_only=False` (có hash-check trước) vào `base/base_trainer.py` như `approved_source_patches` thay vì shim qua `PYTHONPATH` | Hiện phải inject shim để unpickle; nếu duyệt thì DCNet chạy được bằng CLI chuẩn không cần shim | Sửa 1 dòng trong checkout pinned + khai báo patch như CSMCIR | Đã chạy bằng shim, kết quả đầy đủ | `scripts/compat/sitecustomize.py`, `logs/dcnet_run2.log` | Duyệt patch 1 dòng (kèm base/derived sha256) hay giữ shim |

Không mục nào chặn job khác: chỉ **D-1** (file SPRC ViT-g) và **D-8** (caption CSMCIR) còn thực sự treo. D-6 đã áp dụng, D-9 đã giải quyết trong buổi này; 15/17 entry đã có số đo native độc lập.

Bổ sung D-6…D-9 cho cohort inventory:

| ID | model/job | hành động đề xuất | vì sao | phạm vi/rủi ro | đã thử an toàn | bằng chứng | cần bạn quyết/cung cấp |
|---|---|---|---|---|---|---|---|
| ~~D-6~~ | `limn` runtime | **ĐÃ ÁP DỤNG trong buổi này**: `adapted_runtime.packages` của LIMN nay là `['open_clip_torch==2.20.0']` trong `workbench/registry/models.yaml`, wrapper tự đọc profile đó nên không cần script riêng của phiên | Hai căn cứ **đo được**: (i) `LIMN/requirements.txt:37` khai đúng `open-clip-torch==2.20.0`; (ii) `workbench/envs/limn-adapted` đang chạy 2.20.0 và là môi trường sinh ra kết quả 55.7919/77.6378. Lý do cũ (`AttributeError: ... 'pool_type'` dưới 2.24) **đã bị rút lại** vì không tái lập được | Chỉ đổi 1 dòng của riêng LIMN; không đụng profile nào khác | Preflight qua wrapper: `[READY] LIMN preflight passed`; 3 checkpoint giữ nguyên một bundle | `limn/scripts/limn_version_repro.py`, mục 3.15 | Không cần quyết nữa |
| **D-7** | `tme` feature contract | Duyệt sửa `preparation_contracts.yaml` `feature_dtype: float16 → float32` (và lower bound 10.47 → 20.94 GiB) | Đo thật: `visual_encoder` đúng là fp16, nhưng `vit_encode()` dưới `torch.cuda.amp.autocast()` trả **float32**, nên `torch.save` ra float32. Wrapper `tme.py:414` từ chối cache mà chính pipeline native sinh ra | Sửa 2 giá trị contract; không đổi numerics (dtype chỉ dùng cho gate + size floor) | Đã sinh cache đúng như native, chạy xong 3 category bằng session runner nhận dtype đo được | `cohort/measurements/tme_blocker.json`, `cohort/measurements/tme_fashioniq_0.json`, `cohort/logs/tme_run5.log` | Duyệt sửa contract (nếu không, TME chỉ chạy được qua session runner) |
| **D-8** | `csmcir` | Cần bạn cung cấp `COT_ours2/bert_captions/fashioniq/cap.{dress,shirt,toptee}.val.json` và `captions/keywords_in_mods_*.json`, **hoặc** duyệt rõ ràng một phương án thay thế | Hai asset này không có trên máy và không có trong HF repo tác giả (repo chỉ ship Qwen captions) | Chỉ thêm file vào checkout/model root; không đổi code | Checkpoint đã tải + hash-verified; dừng đúng ở dataset constructor | `cohort/measurements/csmcir_blocker.json` | Gửi asset, hoặc chọn (a) bỏ CSMCIR, (b) duyệt thay bằng caption khác dưới nhãn riêng |
| ~~D-9~~ | `ptha_mtst` | **ĐÃ GIẢI QUYẾT trong buổi này** — không còn cần bạn quyết để chạy được. Nguyên nhân là biên feature trong chính source pinned (định nghĩa `extract_target_features` thứ hai trả chưa chiếu), không phải checkpoint thiếu head; đã sửa bằng `inference_target()` của tác giả và chạy đủ 3 category. Nếu bạn **vẫn muốn** có reproduction FashionIQ thật thì đó là yêu cầu khác: (a) xin checkpoint FashionIQ-trained từ tác giả | Sửa xong: `load_state_dict` strict 0/0/0, `vision_proj (256,768)` dùng lại 196 864 tham số, 0 tham số mới, adapter ≡ `normalize(vision_proj(x))` với `max_abs_diff = 0.0`, regression 5/5 pass | Không cắt/pad chiều, không thêm Linear ngẫu nhiên, không chiếu hai lần (bị hình học chặn) | Đã chạy full 3 category trên GPU thật: macro **37.8978 / 59.8364**, coverage 100% | `ptha/measurements/ptha_transfer_run1.json`, mục 3.13 | Không cần quyết gì để dùng kết quả transfer; chỉ cần quyết nếu muốn theo (a) |


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
# KHÔNG cần shim, KHÔNG cần PYTHONPATH: opt-out đã nằm trong approved_source_patches
# (1 dòng ở base/base_trainer.py, khai báo trong registry — xem 3.14).
cd $CIR/workbench/third_party/DCNet
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

## 5.1 Lệnh chạy lại — PAIR và cohort inventory

```bash
export CIR=/home/heheboiz/data/cir
export EVID=$CIR/workbench/artifacts/checkpoint_selection/20261106T002617+0700
export DS=$CIR/data/FashionIQ
export PC=$EVID/pair_compat
export CO=$EVID/cohort

# --- PAIR: structural audit, restore audit, forward smoke, full native eval ---
PY=$CIR/workbench/envs/pair-adapted/bin/python
$PY $PC/scripts/audit_tree.py "$EVID/checkpoints/pair/pair-B1.pt" $PC/manifests/pair_B1_audit.json $CIR/workbench/third_party/PAIR/models
$PY $PC/scripts/audit_tree.py "$EVID/checkpoints/pair/pair-B2.pt" $PC/manifests/pair_B2_audit.json $CIR/workbench/third_party/PAIR/models
$PY $PC/scripts/audit_compat_load.py "$EVID/checkpoints/pair" $PC/manifests/pair_compat_audit.json
$PY $PC/scripts/forward_smoke.py "$EVID/checkpoints/pair/pair-B1.pt" B1 H_rev "$DS" $PC/measurements/smoke_B1_H_rev.json 3
for T in B1 B2; do for H in H_rev H_normal; do
  $PY $PC/scripts/run_pair.py --checkpoint "$EVID/checkpoints/pair/pair-$T.pt" --tag $T \
      --hypothesis $H --dataset-root "$DS" --batch-size 96 \
      --output $PC/measurements/pair_$(echo $T | tr A-Z a-z)_$H.json
done; done

# --- ENCODER (must run from the checkout: pinned source uses clip_path='./') ---
cd $CIR/workbench/third_party/ENCODER
WORKBENCH_RUNTIME_ADAPTATION=sm_120 PYTHONPATH=$CIR \
  $CIR/workbench/envs/pair-adapted/bin/python -m workbench.replay.encoder \
  --checkpoint-root $CO/downloads/encoder --dataset-root $DS \
  --source-root $CIR/workbench/third_party/ENCODER --batch-size 96 \
  --output $CO/measurements/encoder_fashioniq.json

# --- HINT (LAVIS, NHƯNG KHÔNG có --variant: parser riêng của HINT không nhận cờ đó) ---
cd $CIR
WORKBENCH_RUNTIME_ADAPTATION=sm_120 $CIR/workbench/envs/airknow-adapted/bin/python \
  -m workbench.replay.hint --checkpoint-root "$CO/downloads/hint" --dataset-root "$DS" \
  --source-root $CIR/workbench/third_party/HINT --batch-size 64 \
  --output "$CO/measurements/hint_fashioniq.json"

# --- HABIT / INTENT / ConeSep (ba wrapper này MỚI có --variant) ---
for spec in "habit fiq_n02 HABIT" "intent fiq_n02 INTENT" "conesep fiq_n02 ConeSep"; do
  set -- $spec
  WORKBENCH_RUNTIME_ADAPTATION=sm_120 $CIR/workbench/envs/airknow-adapted/bin/python \
    -m workbench.replay.$1 --checkpoint-root "$CO/downloads/$1" --dataset-root "$DS" \
    --source-root $CIR/workbench/third_party/$3 --variant $2 --batch-size 64 \
    --output "$CO/measurements/$1_$2.json"
done

# --- LIMN (open_clip 2.20, see D-6) ---
$CIR/workbench/envs/limn-adapted/bin/python $CO/scripts/run_limn_oc220.py \
  "$CO/downloads/limn" "$DS" $CIR/workbench/third_party/LIMN \
  "$CO/measurements/limn_base_iter0.json" 16

# --- TME: feature cache, then evaluation (see D-7) ---
cd $CIR/workbench/third_party/TME
$CIR/workbench/envs/tme-adapted/bin/python $CO/scripts/tme_precompute_bounded.py \
  "$CO/tme_dataset_view_real/" $CIR/workbench/third_party/TME/features/FashionIQ 32 8
cd $CIR
PYTHONPATH=$CIR WORKBENCH_RUNTIME_ADAPTATION=sm_120 \
  WORKBENCH_TME_PYTHON=$CIR/workbench/envs/tme-adapted/bin/python \
  WORKBENCH_TME_QF_BS=64 WORKBENCH_TME_QUERY_BS=8 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $CIR/workbench/envs/airknow-adapted/bin/python $CO/scripts/tme_session_runner.py \
  --checkpoint-root $CO/ckpt/tme --dataset-root "$DS" \
  --source-root $CIR/workbench/third_party/TME \
  --output $CO/measurements/tme_fashioniq_0.json
```

## 6. Đối soát hoàn thành — cả 17 entry đăng ký

| entry | checkpoint / bundle | protocol | trạng thái | macro R@10 / R@50 | nhãn association |
|---|---|---|---|---|---|
| airknow | `fiq_n05` | `fashioniq_full_gallery_ref_excluded` | **COMPLETED** | 52.5466 / 72.9026 | AUTHOR_VERIFIED (SHA LFS) |
| airknow (variant) | `fiq_n08` | như trên | **COMPLETED** | 50.6417 / 71.4556 | AUTHOR_VERIFIED (SHA LFS) |
| sprc | `sprc_fiq_vitl` (ViT-L) | `fashioniq_original_split` | **COMPLETED** | 51.0386 / 73.3263 | LOAD-COMPATIBLE (kiến trúc khớp; **không** có SHA tác giả) |
| sprc | `sprc_fiq` (ViT-g, paper) | — | **PENDING_USER_INPUT** | — | chưa có file (D-1) |
| pair | `pair-B1` | `fashioniq_val_split` | **EXPLORATORY** | 35.0147 / 57.5753 | EXPLORATORY_PUBLIC_MAPPING |
| pair | `pair-B2` | `fashioniq_val_split` | **EXPLORATORY** | 22.1564 / 43.4864 (H_normal; tốt nhất của B2) | EXPLORATORY_PUBLIC_MAPPING |
| combiner_rn50x4_noft | `fiq_comb_RN50x4_noft.pt` | `fashioniq_original_split` | **COMPLETED** | 35.1407 / 58.4436 | EMPIRICAL |
| clip4cir_rn50x4_fullft | comb + clip | `fashioniq_original_split` | **COMPLETED** | 43.8182 / 66.9501 | EMPIRICAL |
| dcnet | `trained_model.pth` | `fashioniq_full_gallery_ref_excluded` | **COMPLETED** | 27.9297 / 53.7272 | EMPIRICAL |
| clvc_net | 6 whole models | `fashioniq_val_split` | **COMPLETED** | 29.9707 / 58.6217 | EMPIRICAL |
| tgcir | `Fashioniq.pt` | `fashioniq_val_split` | **COMPLETED** | 49.7208 / 71.1023 | EMPIRICAL |
| limn | `base_iter0_*` ×3 | `fashioniq_val_split` | **COMPLETED** | 55.7919 / 77.6378 | EMPIRICAL (open_clip 2.20, D-6) |
| encoder | `fashioniq.pt` | `fashioniq_val_split` | **COMPLETED** | 56.2231 / 77.6674 | EMPIRICAL |
| hint | `fashioniq.pt` | `fashioniq_val_split` | **COMPLETED** | 59.8034 / 80.1308 | EMPIRICAL |
| habit | `HABIT-FIQ_N0.2.pt` | `fashioniq_full_gallery_ref_excluded` | **COMPLETED** | 54.8733 / 74.5034 | EMPIRICAL |
| intent | `intent_fiq_0.2.pt` | `fashioniq_full_gallery_ref_excluded` | **COMPLETED** | 54.8662 / 75.4319 | EMPIRICAL |
| conesep | `ConeSep-FIQ-N0.2.pt` | `fashioniq_full_gallery_ref_excluded` | **COMPLETED** | 54.7613 / 75.2151 | EMPIRICAL |
| tme | `fashioniq_0_best_model` | `fashioniq_original_split` | **COMPLETED** | 54.9229 / 74.8384 | EMPIRICAL (contract dtype, D-7) |
| csmcir | `fashioniq_tuned_clip_best.pt` | `fashioniq_original_split` | **BLOCKED** | — | checkpoint hash-verified; thiếu caption asset (D-8) |
| ptha_mtst | `tuned_mtst_cirr.pt` | `fashioniq_original_split` | **COMPLETED** | 37.8978 / 59.8364 | transfer CIRR→FIQ; nguyên nhân là biên feature, đã sửa (3.13) |

Tổng: **15/17 entry có số đo native**, `pair` đo được nhưng exploratory, 1 entry chưa đo (CSMCIR, D-8), và **việc còn chờ file người dùng** là biến thể SPRC ViT-g của entry `sprc` (D-1) — không phải entry mới (CSMCIR thiếu caption asset; PTHA/MTST sai chiều projection). PAIR có số đo nhưng mang nhãn `EXPLORATORY_PUBLIC_MAPPING`, **không** tính là tái lập.

Coverage: mọi run đều đủ query (2017 dress / 2038 shirt / 1961 toptee), gallery đúng protocol, reference loại trước Recall ở các protocol yêu cầu, không run nào bị rút gọn.

## 7. Những gì KHÔNG được chứng minh

- **Không entry nào** là `AUTHOR_VERIFIED_REPRODUCTION` của một con số paper. Nhãn `AUTHOR_VERIFIED` chỉ dùng khi có digest do tác giả công bố và khớp; SPRC ViT-L **không** đủ điều kiện đó và đã được hạ xuống `LOAD-COMPATIBLE`. AirKnow (SHA LFS) và SPRC ViT-L (khớp lớp tuyệt đối) chỉ đạt mức *association*; số đo vẫn là empirical.
- **PAIR**: load và chấm được, nhưng **tương đương source gốc chưa xác lập**. Đo được: `transformer_a` khôi phục 2 layer × 4 head ≠ public 1 × 8; checkpoint có `transformer_c/d`, `weight_chanelImg*`, `cross_t2i` mà public `model.py` không định nghĩa; 5.20% tham số không được forward public chạm tới. **Chưa** chứng minh được đó là nguyên nhân duy nhất của điểm thấp, và **không** thể gọi các module thừa là "training-only" vì chúng không xuất hiện ở bất kỳ đâu trong checkout, kể cả loss/train. Xem 3.9.
- **TME**: chạy được nhưng dưới session runner nhận dtype đo được (float32) và giảm `Qformer_bs`/`query_bs`; cho tới khi D-7 được duyệt, đây **không** phải đường chạy `workbench.replay.tme` nguyên bản.
- **LIMN**: chạy dưới open_clip **2.20.0**, đúng pin tác giả (`LIMN/requirements.txt:37`). `adapted_runtime` trước đây khai `>=2.24,<2.27` và **đã được sửa** về `==2.20.0` (mục 3.15), nên nay contract và môi trường khớp nhau — không còn lệch.
- **CSMCIR**: không có số đo. Không suy ra từ model khác, không dùng caption thay thế.
- **PTHA+MTST**: **có** số đo transfer CIRR→FIQ (mục 3.13: 37.8978 / 59.8364) nhưng **KHÔNG** phải reproduction FashionIQ — không cắt đầu chiếu, không thay bằng checkpoint FashionIQ của tác giả (không được phát hành). Điều chưa chứng minh là *tương đương với cách tác giả dùng checkpoint cho CIRR*, không phải "không có số đo".
- **SPRC ViT-g**: chưa chạy; không dùng ViT-L thay cho hàng ViT-g.
- Validation set đã dùng để chọn checkpoint ⇒ các số không phải ước lượng held-out không thiên lệch.
- Không có so sánh nào vắt qua protocol khác nhau: mỗi Δ trong báo cáo chỉ so trong cùng dataset/backbone/protocol/noise đã ghi.

### 7.1 Nhật ký lần chạy cohort (mọi lần, kể cả lần hỏng)

| model / bước | lệnh (đã chạy) | kết quả |
|---|---|---|
| ENCODER lần 1 | `-m workbench.replay.encoder` từ repo root | FAILED `RuntimeError: Pretrained weights (./open_clip_pytorch_model.bin) not found` — source dùng `clip_path='./'`, phụ thuộc CWD |
| ENCODER lần 2 | như trên với `cwd=third_party/ENCODER` | 56.2231 / 77.6674, `parity_mismatches: []` |
| HINT preflight | thiếu checkpoint | `[BLOCKED] HINT checkpoint missing` (chưa tải xong) |
| HINT | `-m workbench.replay.hint` | 59.8034 / 80.1308 |
| HABIT | `-m workbench.replay.habit --variant fiq_n02` | 54.8733 / 74.5034 |
| INTENT lần 1 | `-m workbench.replay.intent` | FAILED `ModuleNotFoundError: No module named 'matplotlib'` — `lavis/models/blip2_models/blip2_cir.py:14` import ở module level |
| INTENT lần 2 | sau khi cài `matplotlib==3.9.2` vào env scoped | 54.8662 / 75.4319 |
| ConeSep lần 1 | `--checkpoint-id fiq_n02` | FAILED `unrecognized arguments` — cờ đúng là `--variant` |
| ConeSep lần 2 | `--variant fiq_n02` | 54.7613 / 75.2151 |
| LIMN lần 1 | `-m workbench.replay.limn` (open_clip 2.24) | FAILED `version mismatch: expected open_clip_torch >=2.24,<2.27, found ...` — ghi chú: lỗi `pool_type` từng được quy cho 2.24 **không tái lập** khi đo lại; xem mục 3.15 |
| LIMN lần 2 | env `limn-adapted` (open_clip 2.20) qua wrapper | FAILED `version mismatch: expected open_clip_torch >=2.24,<2.27, found 2.20.0` |
| LIMN lần 3 | `run_limn_oc220.py` (3 category chạy xong) | aggregate helper lỗi `'dict' object has no attribute 'complete'`; số per-category đã có, tổng hợp lại từ chính 3 dòng đó ⇒ 55.7919 / 77.6378 |
| TME assets | cleanup đêm trước đã xoá `blip2_pretrained.pth`, `eva_vit_g.pth` | tải lại; sha256 `f31f96e4…`, `99d2bb36…` khớp đúng giá trị đã ghi trong `cleanup_result.json` |
| TME preflight ×4 | `WORKBENCH_TME_PYTHON` chưa/không hợp lệ | FAILED lần lượt: not set → same-env rejected → py3.11 vs expected 3.10 → numpy 2.x vs `<2` |
| TME precompute (pinned) | `prepare_model.py --execute --allow-gpu-preparation` | FAILED exit **-9** (SIGKILL) sau khi in "Done": list + `torch.cat` peak 28.76 GiB cho tensor 10.47 GiB |
| TME precompute (pinned, trực tiếp) | `src/precompute.py` qua shim | FAILED `FileNotFoundError: …/images/B009PMCJLW.jpg` — source hardcode `.jpg`, canonical là `.png` |
| TME precompute (bounded) | `tme_precompute_bounded.py` (preallocate, cùng numerics) | OK `(15536, 257, 1408) float32`, 22 487 180 914 byte, peak 25.47 GiB |
| TME eval lần 1 | `replay.tme` với checkpoint tar | FAILED `weights_only=True` không đọc được legacy `.tar` |
| TME eval lần 2 | sau khi bóc tar → torch zip tensor-only | FAILED `feature cache dtype mismatch: expected float16, got float32` |
| TME eval lần 3 | `tme_session_runner.py` (nhận dtype đo được) | FAILED `ModuleNotFoundError: No module named 'workbench'` (thiếu PYTHONPATH) |
| TME eval lần 4 | + `PYTHONPATH=$CIR` | FAILED `torch.OutOfMemoryError`, cần 6.20 GiB, còn 4.32 GiB |
| TME eval lần 5 | + `Qformer_bs 256→64`, `query_bs 32→8`, expandable_segments | 54.9229 / 74.8384 |
| PTHA transfer (lần 1, **đã bị thay thế**) | `-m workbench.replay.ptha_mtst --transfer-cirr-to-fashioniq` | FAILED `RuntimeError: ... [61072, 256] but got: [61072, 768]` — nguyên nhân tìm ra ở 3.13, nay chạy được |

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
| `$EVID/../20261106T002617+0700/` | thư mục evidence của session |
| `$EVID/pair_compat/scripts/` | `compat_loader.py` (hash-gated + mapping), `audit_tree.py`, `audit_compat_load.py`, `forward_smoke.py`, `run_pair.py` |
| `$EVID/pair_compat/manifests/` | `pair_B1_audit.json`, `pair_B2_audit.json`, `pair_compat_audit.json`, `attempts.json`, `pair_compat_verdict.json` |
| `$EVID/pair_compat/measurements/` | `pair_b1_H_rev.json`, `pair_b1_H_normal.json`, `pair_b2_H_rev.json`, `pair_b2_H_normal.json`, `smoke_*.json`, `summary.json` |
| `$EVID/pair_compat/BUNDLE_MANIFEST.json` | danh mục evidence bundle (72 file, 0.7 MiB; **không** chứa weight/image/env) |
| `$EVID/cohort/scripts/` | `run_limn_oc220.py`, `tme_precompute_bounded.py`, `tme_session_runner.py`, `dump_pickle_globals.py` |
| `$EVID/cohort/measurements/` | `cohort_summary.json` + đo từng model (`encoder_*`, `hint_*`, `habit_*`, `intent_*`, `conesep_*`, `limn_*`, `tme_*`) và 3 file blocker (`csmcir_blocker.json`, `ptha_mtst_transfer_blocker.json`, `tme_blocker.json`) |
| `$EVID/cohort/manifests_acquired_final.json` | byte + SHA-256 của **mọi** checkpoint đã dùng, kèm cờ prefix-match với digest kỳ vọng |
| `$EVID/cohort/manifests_environment.json` | platform, GPU, phiên bản python/torch từng env và các package quyết định kết quả |
| `$EVID/configs/selected_overrides.yaml` | mapping candidate-level + absolute path + hash; **chưa** áp vào global |
| `$EVID/measurements/selection_decisions.json` | bảng quyết định máy đọc được (status + macro) |
| `$EVID/manifests/*` | inventory weight cũ, cleanup result, protocol identity, digest bundle |

### 9.1 Đề xuất diff registry — tách rõ "đã áp dụng" và "chờ duyệt"

**Đã rà lại schema trước khi đề xuất.** `workbench/backend/registry.py:52` chỉ chấp nhận ba giá trị cho `checkpoint_mapping_status`: `VERIFIED_METADATA`, `UNVERIFIED`, `ARTIFACT_ASSOCIATED_UNVALIDATED`. Vì vậy các nhãn kiểu `EMPIRICAL_VERIFIED` từng được đề xuất ở bản trước là **không hợp lệ** và đã bị bỏ.

Hiện trạng thật (đọc từ registry, không suy đoán): 10 entry đã `VERIFIED_METADATA`; **8 variant còn `UNVERIFIED`** — `pair` (×2), `clvc_net`, `dcnet`, `combiner_rn50x4_noft`, `clip4cir_rn50x4_fullft`, `tgcir`, `sprc`.

#### Đã áp dụng trong buổi này (không cần duyệt)

| thay đổi | file | căn cứ |
|---|---|---|
| LIMN `adapted_runtime.packages` → `['open_clip_torch==2.20.0']` | `models.yaml` | pin tác giả `requirements.txt:37` + env đã đo (mục 3.15) |
| DCNet `approved_source_patches` (1 dòng, `base/base_trainer.py`) | `models.yaml` + `workbench/patches/dcnet_checkpoint_weights_only_optout.patch` | patch áp từ base ra đúng derived digest; chạy CLI sạch không shim, kết quả khớp tuyệt đối (mục 3.14) |

#### Chờ duyệt — **không** được thăng `VERIFIED_METADATA`

Điểm quan trọng, và là chỗ bản trước sai: **có số đo không đồng nghĩa có digest tác giả.** Với 8 variant đang `UNVERIFIED`, không có SHA do tác giả công bố để đối chiếu; chúng mới chỉ có *archive/author association + đo được*. Theo đúng ba giá trị của schema, đề xuất là:

1. `clvc_net`, `dcnet`, `combiner_rn50x4_noft`, `clip4cir_rn50x4_fullft`, `tgcir`, `sprc` → `ARTIFACT_ASSOCIATED_UNVALIDATED` (tác giả mô tả archive/config, đo được, **chưa** có digest tác giả để xác thực).
2. `pair` → giữ nguyên `UNVERIFIED`, kèm `reproduction_status` `EXPLORATORY_PUBLIC_MAPPING` (mục 3.9: tương đương source gốc chưa xác lập).
3. `reproduction_status` của các entry đã đo: cập nhật theo kết quả thật, dùng chuỗi mô tả tự do (trường này **không** có enum trong schema) — ví dụ `hint` → mô tả native-parity đã kiểm chứng, `ptha_mtst` giữ `SELECTED_TRANSFER_ONLY` vì bản chất vẫn là transfer.
4. **Không** đổi default selection, protocol definition, hay shared evaluation logic.

#### Còn lại ở `preparation_contracts.yaml` (D-4, D-7)

- (D-4) `dcnet_glove_regen`: **hạ cấp đề xuất.** `process_cap.py` lấy mẫu `np.random.normal` **không seed** cho OOV, nên recipe có seed tạo ra artifact **khác** của tác giả — không thể khớp hash tác giả. Nếu làm, phải gắn nhãn `WORKBENCH_SEEDED_RECIPE` và **không** dùng để đối chiếu số paper.
- (D-7) TME `feature_dtype`: xem 3.16 — đo được float32 trên runtime này, nhưng **chưa** đủ để phủ định `float16` cho mọi runtime. Đề xuất ghi dtype *theo runtime* thay vì sửa số lịch sử.

SHA-256 để điền lấy nguyên văn từ `cohort/manifests_acquired_final.json` và `manifests/selected_bundle_digests.json`.

Ghi chú `.gitignore`: `workbench/artifacts/**` bị chặn, nên evidence **không** vào git — chỉ commit báo cáo. Muốn đưa `pair_compat/` + `cohort/` lên git thì cần thêm negation hẹp, chưa làm.

## 10. Tóm tắt

### 10.1 Kết quả đo được (số native, không phải suy luận)

**15/17 entry đăng ký** có số đo native thật; `pair` đo được nhưng exploratory. PTHA chuyển từ blocked sang completed trong buổi này. Còn treo: **CSMCIR** chưa đo được (thiếu caption asset, D-8) và **biến thể SPRC ViT-g** chờ file (D-1, variant của `sprc` chứ không phải entry thứ 18).

### 10.2 Những gì thực sự đã thay đổi trong buổi này

| thay đổi | loại | bằng chứng |
|---|---|---|
| `workbench/replay/ptha_mtst.py` — thêm `_target_feature_contract()` | sửa code, phạm vi 1 model | `ptha/measurements/ptha_transfer_run1.json`, mục 3.13 |
| `workbench/tests/test_ptha_target_feature_contract.py` — 5 regression | test mới | 5/5 pass |
| `workbench/registry/models.yaml` — LIMN `adapted_runtime` `==2.20.0` | 1 dòng, 1 model | mục 3.15 |
| `workbench/registry/models.yaml` — DCNet `approved_source_patches` | khai báo patch | mục 3.14 |
| `workbench/patches/dcnet_checkpoint_weights_only_optout.patch` | patch 1 dòng, base→derived khớp | mục 3.14 |
| `workbench/third_party/DCNet/base/base_trainer.py` | áp patch trên (đã khai báo) | DCNet chạy `PYTHONPATH` sạch, kết quả khớp tuyệt đối |
| `CHECKPOINT_SELECTION_REPORT.md` | sửa sai, thêm 3.12–3.16 | chính file này |
| `workbench/docs/reviews/checkpoint_selection/20261106T233409+0700/` | bundle review được: **49 entry có hash** (+ chính file index = **50 file**, ~353 KB) | `EVIDENCE_INDEX.json`, `README.md`, `TEST_STATUS.md` |

### 10.5 Trạng thái test — nói thẳng

Repo dùng **cú pháp f-string PEP 701**, chỉ hợp lệ từ **Python 3.12** (`workbench/backend/environment_planner.py:669`). Máy này **không có** interpreter 3.12 nào kèm `pip`; mọi env conda đều 3.11.15. `git show HEAD:` của chính file đó **cũng** fail dưới 3.11 ⇒ đây là hạn chế **có từ trước**, không phải do buổi này.

Vì vậy: đúng **8 module** không collect được, và **12 test riêng lẻ** trong `test_runtime_adaptation.py` fail vì chúng import cùng module 3.12-only đó (bản thân module vẫn collect). Đã chạy và pass: `test_ptha_target_feature_contract` (5), `test_ptha_mtst_preflight`, `test_r19_ptha_wrapper`, 6 test của `test_runtime_adaptation`, và **60 test integrity** (`test_approved_source_patch`, `test_vendored_approved_patch`, `test_vendored_source_integrity`, `test_verified_declarations`, `test_dcnet_replay`) — nhóm cuối là nhóm quan trọng nhất cho thay đổi DCNet, vì nó chứng minh khai báo `approved_source_patches` chặn đúng cổng kiểm tra source đã bị sửa. **Chưa** có CI xanh; việc đó cần môi trường 3.12. Chi tiết: `TEST_STATUS.md` trong bundle.

Hai assertion trong `test_runtime_adaptation.py` mã hoá contract LIMN **cũ** (`>=2.24,<2.27`) đã được cập nhật theo contract mới `==2.20.0` — đây là thay đổi hành vi có chủ đích (mục 3.15), không phải sửa test cho qua.

Không đổi: protocol nào, `models.yaml` default selection nào, shared evaluation logic nào, wrapper của model nào khác.

### 10.3 Bốn khẳng định bị hạ cấp trong buổi này

1. **PTHA "architectural mismatch"** → sai. Checkpoint có `vision_proj (256,768)` và nạp strict; lỗi ở biên feature. Nay chạy đủ 3 category.
2. **HINT "lệnh chạy lại"** → sai cờ (`--variant`). Nay có CLI riêng, đã chạy thật.
3. **LIMN "chỉ 2.20 mới load được"** → không tái lập. Cả 2.20 và 2.24 unpickle được; pin giữ lại vì là pin tác giả, không phải vì lỗi load.
4. **PAIR "public code chắc chắn không phải code đã train"** → hạ xuống "tương đương chưa xác lập". Hình học 2×4 và 5.20% tham số ngoài forward là **đo được**, nhưng không chứng minh nguyên nhân hay tính bất khả thi.

Thêm: **SPRC ViT-L** bị hạ từ `AUTHOR_VERIFIED` xuống `LOAD-COMPATIBLE` — nạp đúng kiến trúc **không** phải tác giả xác nhận.

### 10.4 Còn lại cho bạn

Chỉ hai việc chặn kết quả: **(D-1)** đặt file SPRC OneDrive vào `workbench/artifacts/checkpoints/sprc/sprc_fiq.pt`; **(D-8)** cung cấp caption/keyword asset cho CSMCIR. Các mục D-2/D-3/D-4/D-7 là quyết định chính sách, không chặn gì thêm.
