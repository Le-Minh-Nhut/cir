# CIR — Báo cáo chọn checkpoint (overnight 2026-11-05/06)

**Session:** `20261106T002617+0700` · **Evidence:** `/home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700/` · giờ máy Asia/Ho_Chi_Minh, 2026-11-06 sáng.

Máy: RTX 5070 Ti (Blackwell `sm_120`, 16 303 MiB, driver 580.173.02), 20 CPU, 31 GiB RAM.  
Repo: `/home/heheboiz/data/cir`, branch `research/cir-failure-workbench`, HEAD `26ad2dd805f02aa724f8572e83e3a89d3d8f0a8f`, `git status --short` trống (không sửa source dùng chung).

## 0. Chỉ mục evidence

Bundle của buổi 20261107 (CSMCIR + GloVe recipe):
`workbench/docs/reviews/checkpoint_selection/20261107T021500+0700/`. Mỗi file có `bytes` và `sha256`
Riêng trong `EVIDENCE_INDEX.json` (index tự nó không tính); tổng là `sum(bytes)` trong file đó, không
chép lại ở đây vì con số chép tay sẽ lệch ngay sau lần sửa bundle kế tiếp. Đọc `README.md` trong đó trước.

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

Đếm theo `model_id` trong registry. **PAIR là một entry** (B1/B2 chỉ là hai file của cùng entry), nên bảng dưới có 16 dòng đã đo (AirKnow hai variant) + 1 dòng exploratory (`pair`). Bảng không xếp hạng giữa các protocol khác nhau; cột `protocol` nằm ngay cạnh số để tránh so sánh sai.

| # | model_id | macro R@10 | macro R@50 | mean | protocol | trạng thái |
|---|---|---|---|---|---|---|
| 1 | hint | **59.8034** | **80.1308** | 69.9671 | `fashioniq_val_split` | COMPLETED — đã kiểm chứng lại, xem 3.12 |
| 2 | encoder | **56.2231** | **77.6674** | 66.9452 | `fashioniq_val_split` | COMPLETED |
| 3 | limn (base iter-0) | **55.7919** | **77.6378** | 66.7149 | `fashioniq_val_split` | COMPLETED (open_clip 2.20, registry đã sửa) |
| 4 | tme (0% noise) | **54.9229** | **74.8384** | 64.8807 | `fashioniq_original_split` | COMPLETED (normal path `workbench.replay.tme`, xem 3.16.6) |
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
| d | csmcir | **57.0507** | **77.2986** | 67.1747 | `fashioniq_original_split` | COMPLETED — public-caption path-mapped run, run `f03887d2` (đối chứng `f564570e`), xem 3.18 |

**16/17 entry đã có số đo native thật** (PTHA và CSMCIR nay nằm trong số đó). Số 17 đếm theo `model_id` đăng ký:

- 16 entry đã đo native (bảng trên, trừ dòng `e`);
- `pair` **đo được nhưng exploratory** — không tính vào 15 vì không phải hành vi của implementation gốc;
- `csmcir` **đã đo** trong buổi 20261107 (3.18): macro 57.0507 / 77.2986 trên caption công khai, run `f03887d2` (đối chứng `f564570e`, cùng số).

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
| CSMCIR | **COMPLETED** (research variant) | checkpoint SHA khớp tác giả; caption công khai khớp sha256; macro 57.0507 / 77.2986 (3.18, run `f03887d2`; `f564570e` cùng số) | đã chạy, exit 0 cả hai run |
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
- **TME** — `fashioniq_original_split`, 0% eval noise. Checkpoint gốc là **tar bọc torch-zip**; đã bóc lớp tar để dùng state dict tensor-only. Cache feature do chính pipeline sinh: `(15536, 257, 1408)` **float32**; contract lịch sử khai `float16` nên wrapper **từng** từ chối; **D-7 đã đóng** bằng `runtime_overrides.sm_120.feature_dtype: float32` và đường chuẩn đã chạy xong (3.16.6). Evaluator chạy với `Qformer_bs 256→64`, `query_bs 32→8` (chính docstring pinned bảo giảm khi OOM; chỉ đổi số ảnh mỗi forward, ranking không đổi).

**CSMCIR** và **PTHA+MTST**: cả hai blocker cũ đã đóng — chi tiết:

- **CSMCIR**: ~~checkpoint tải đủ nhưng thiếu caption~~ → **đã đo trong buổi 20261107** (3.18): checkpoint khớp SHA (`c8812071…`), caption công khai khớp sha256, exit 0, macro **57.0507 / 77.2986** (hai run `f564570e` + `f03887d2`). D-8 đóng vì `bert_captions`/`keywords_in_mods` chỉ được đọc bởi `FashionIQDataset_imageedit` — class chết, không nơi nào khởi tạo.
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

### 3.13.3 Sửa: dispatch theo ĐÚNG contract của lớp đã load

Bản đầu tiên áp `_target_feature_contract()` **vô điều kiện cho cả hai lớp**. Đó là một lỗi thật:
hai lớp hỗ trợ cùng trả về target thô từ `extract_target_features`, nhưng **chiếu ở hai chỗ khác nhau**.

| lớp | `extract_target_features` | `inference()` có tự chiếu target? | biên phải làm gì |
|---|---|---|---|
| `Blip2QformerCirCat` | trả Q-Former thô `[N,32,768]` (hàm định nghĩa **hai lần**, bản thứ hai ở dòng 483 thắng, dòng chiếu bị comment) | **Không** — `inference()` dòng 287 chỉ matmul query 256 chiều với target được đưa vào | **phải chiếu**, nếu không shape không gặp nhau |
| `Blip2Qformer4TGIRCESIM` | trả Q-Former thô `[N,32,768]` (dòng 501) | **Có** — `inference()` dòng 429 chạy `F.normalize(self.vision_proj(target_embeds), dim=-1)` | **phải để nguyên**, chiếu ở biên sẽ thành chiếu hai lần |

Sửa: bảng `TARGET_FEATURE_CONTRACTS` ánh xạ **tên lớp đã construct và load** → contract, và
`evaluate_transfer` resolve contract từ `type(model).__name__` **sau khi** `load_state_dict(strict=True)`,
không từ key của checkpoint và không từ độ rộng tensor.

- `project_at_boundary` → gọi `model.inference_target(raw)`; **kiểm tra độ rộng TRƯỚC khi chiếu**, để
  extraction đổi thì ra chẩn đoán rõ ràng chứ không phải lỗi `nn.Linear` mờ nghĩa;
- `project_inside_inference` → trả `(raw_targets, frozen_vision)` nguyên vẹn, tham chiếu chính object gốc;
- lớp không khai báo → `ValueError` nêu tên lớp và danh sách lớp đã khai báo, **không đoán**;
- `index_features` không phải cặp `(targets, frozen)` → `ValueError`.

**Sửa lại một lập luận sai của bản trước.** Bản trước viết "chiếu hai lần là bất khả thi vì
`vision_proj` nhận 768 xuất 256 nên lần thứ hai sẽ ném lỗi". Đo lại thì đúng là **có ném**
(`mat1 and mat2 shapes cannot be multiplied (96x256 and 768x256)`) — nhưng đó **không phải** lý do
biện minh cho dispatch đúng. Một exception xảy ra *sau khi đã làm sai phép toán* không phải là
dispatch đúng; nó chỉ chứng minh nhánh đó **không dùng được**, không chứng minh nhánh kia đúng.
Lý do đúng là: mỗi lớp phải nhận đúng representation mà `inference()` của nó tiêu thụ.

### 3.13.4 Kiểm chứng sau khi sửa

| kiểm tra | kết quả |
|---|---|
| trước/sau shape (dress/shirt/toptee) | `[3817,32,768]→[3817,32,256]`, `[6346,…]`, `[5373,…]` |
| contract resolve từ lớp đã load | `loaded_class = Blip2QformerCirCat`, `contract = project_at_boundary`, `action = projected_at_boundary_via_inference_target` |
| tham số chiếu dùng lại | 196 864 (đúng `vision_proj` của checkpoint) |
| tham số mới | **0** |
| biên ≡ `F.normalize(vision_proj(x))` trên tensor thật | `max_abs_diff = 0.0` (cả 3 category) |
| `load_state_dict` strict | 0 missing / 0 unexpected / 0 shape mismatch |
| regression PTHA | 11/11 pass (viết lại, xem dưới) |
| chạy GPU lại toàn bộ 3 category | exit 0, **bit-identical** với lần trước (bảng dưới) |

**Regression mới thay cho bản cũ.** Bản cũ đếm số lần xuất hiện chuỗi `_target_feature_contract`
trong source — không chạy contract của lớp nào cả. Bản mới gọi **đúng dispatcher production** với
fixture có hình dạng như lớp thật, và đếm số lần `vision_proj` **thực sự được áp dụng end-to-end**:

| ca | kiểm chứng |
|---|---|
| CirCat | biên chiếu **đúng một lần**; `inference()` sau đó **không** chiếu thêm (counter reset rồi assert `== 0`); điểm hữu hạn `[B,N]`; kết quả bằng `F.normalize(vision_proj(x))` cùng weight, `torch.equal == True` |
| TGIR | biên **không** chiếu (`calls == 0`, trả về **chính object** gốc); `inference()` chiếu đúng một lần; điểm qua biên **giống hệt** điểm không qua biên (`torch.equal`) |
| TGIR nếu dispatch sai | chiếu trước rồi `inference()` chiếu lần nữa → **ném** `mat1 and mat2 shapes cannot be multiplied`; ghi lại đúng failure mode đo được, không bịa "âm thầm sai điểm" |
| lớp lạ / contract lạ | `ValueError` có tên lớp; không đoán |
| sai độ rộng extraction | `ValueError` "expects unprojected targets of width 768" — kiểm **trước** khi chiếu |
| cặp `index_features` sai | `ValueError`, nêu rõ cần cặp `(target_features, raw_vision_embeds)` |

**Giới hạn của regression này, nói rõ:** CirCat được chạy **checkpoint thật trên GPU** (dưới đây).
TGIR **chỉ** được kiểm bằng fixture hành vi — **không** có checkpoint TGIR nào trong phiên này, nên
nhánh đó **chưa** được xác nhận bằng checkpoint/GPU. Fixture chứng minh luồng điều khiển, không
chứng minh một checkpoint tác giả chạy được.

### 3.13.5 Kết quả transfer đầy đủ (chạy lại sau khi sửa dispatch)

Protocol `fashioniq_original_split`, gallery đầy đủ theo thứ tự split, reference hợp lệ (không mask — khác DCNet), 100% coverage.

| category | R@10 | R@50 | so với run1 (dispatch cũ) |
|---|---|---|---|
| dress | 33.1185 | 56.0238 | `+0.000000` |
| shirt | 38.8616 | 59.0285 | `+0.000000` |
| toptee | 41.7134 | 64.4569 | `+0.000000` |
| **macro** | **37.8978** | **59.8364** | **`0.000000`** |

Macro mean `48.8671`. Việc refactor dispatch **không đổi một chữ số nào** — đúng như kỳ vọng, vì
nhánh CirCat vẫn chiếu đúng một lần như trước; thay đổi chỉ làm nhánh TGIR thôi bị chiếu hai lần.

Paper context (FashionIQ-trained, 54.99 / 76.28) **chỉ là tham chiếu số**, không phải mục tiêu: checkpoint này train trên CIRR, nên thấp hơn là kỳ vọng, và **không** được đọc là "kém hơn paper".

### 3.13.6 Phán quyết PTHA

- **Đã sửa và đo được**: nguyên nhân là biên feature trong chính source pinned; sửa bằng method có sẵn của tác giả, **dispatch theo lớp đã load**, đã chạy đủ 3 category trên GPU thật, kết quả không đổi so với bản trước đó.
- **Cả hai nhánh giờ nhận đúng representation**: CirCat chiếu ở biên, TGIR để nguyên cho `inference()` tự chiếu.
- **Vẫn là exploratory transfer**: checkpoint CIRR, không có checkpoint FashionIQ do tác giả phát hành.
- **Chưa xác lập**: transfer này có tương đương cách tác giả *dùng* checkpoint cho CIRR hay không — cần dữ liệu CIRR, buổi này không tải thêm.
- **TGIR chưa được xác nhận bằng checkpoint**: chỉ fixture hành vi; không có checkpoint TGIR trong phiên.
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

## 3.16 TME — đã đưa dtype theo runtime vào ĐƯỜNG CHẠY CHUẨN (buổi 20261107)

Đo trực tiếp trên đường native (`precompute.py:38`, `blip2_model.vit_encode(image)` dưới
`torch.cuda.amp.autocast()`): output của `vit_encode` là **float32** trên runtime đã dùng, dù
`create_eva_vit_g` có `convert_weights_to_fp16` tháp ViT. Nguồn: `convert_weights_to_fp16` chỉ hạ
`Conv/Linear` (`eva_vit.py:410-425`), còn LayerNorm và residual/norm path vẫn float32, và autocast
không ép độ rộng activation cuối. Cache 22.49 GB (15536×257×1408×4) khớp float32, không khớp fp16.

### 3.16.1 Đã sửa gì để đường chuẩn chạy được

Trước buổi này: contract khai `feature_dtype: float16`, nên wrapper **từ chối chính cache mà
pipeline native sinh ra**; TME chỉ chạy được bằng session runner riêng. Đó là lỗ hổng thật —
"chạy được" mà phải qua script riêng thì đường chuẩn chưa hoàn thành.

Cách sửa: **không** đổi số của contract lịch sử, mà thêm khối `runtime_overrides` vào
`workbench/registry/preparation_contracts.yaml`, và một resolver duy nhất
`workbench/backend/registry.py:generated_artifact_parameters()`:

```yaml
runtime_overrides:
  sm_120:
    feature_dtype: float32
```

- Override **chỉ** áp khi operator chọn đúng `WORKBENCH_RUNTIME_ADAPTATION=sm_120` cho model này.
  Không chọn → giữ `float16`, và cache float32 **vẫn bị từ chối**. Contract lịch sử không bị
  ghi đè; nó là mặc định.
- Override bị **giới hạn ở `parameters`**. `paths`, `scope`, `generator` là identity của artifact,
  runtime không được sửa. Key không tồn tại trong parameters → `ValueError` (gõ sai không im lặng).
- Resolver trả **bản copy**; contract dùng chung không bị mutate.

Mọi nơi tiêu thụ đều đi qua resolver đó, nên preparation và replay **không thể** bất đồng về dtype:

| nơi dùng | trước | sau |
|---|---|---|
| `replay/tme.py` preflight (size lower bound) | `artifact["parameters"]` | `generated_artifact_parameters(...)` |
| `replay/tme.py` `_prior_feature_provenance` | `artifact["parameters"]` | như trên |
| `replay/tme.py` replay dtype gate | `artifact["parameters"]` | như trên |
| `scripts/prepare_model.py` `_tme_artifact`/`_tme_inputs`/`_tme_validate`/`_tme_run_precompute` | `artifact["parameters"]` | như trên, có truyền `model` |

### 3.16.2 Phân giải contract: mặc định vs adapted (đo bằng chính lệnh chuẩn)

| profile | dtype | gallery | lower bound | preflight nói |
|---|---|---|---|---|
| mặc định (không đặt biến) | `float16` | 15536 | 11 243 589 632 B = **10.47 GiB** | `... 1408 features x 2 bytes` |
| `WORKBENCH_RUNTIME_ADAPTATION=sm_120` | `float32` | 15536 | 22 487 179 264 B = **20.94 GiB** | `... 1408 features x 4 bytes` |

Lệnh (cùng một lệnh, chỉ khác biến môi trường; đường dẫn checkpoint thật dưới đây đã có trên đĩa):

```bash
cd /home/heheboiz/data/cir
env -u WORKBENCH_RUNTIME_ADAPTATION workbench/envs/tme-adapted/bin/python -m workbench.replay.tme \
  --checkpoint-root workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/ckpt/tme --dataset-root data/FashionIQ \
  --source-root workbench/third_party/TME --preflight-only
```

### 3.16.3 Kiểm chứng bằng model THẬT, không chỉ bằng số học

Script `tme_real_cache_smoke.py` load checkpoint TME thật (`fashioniq_0_best_model.pth`,
`load_state_dict` 0 missing / 0 unexpected), chạy `vit_encode` trên 24 ảnh gallery thật với
**batch có chủ đích lẻ cuối** (`10 / 10 / 4`), rồi đưa đúng bytes đó vào `prepare_model._tme_validate`
— hàm production.

| kiểm tra | kết quả |
|---|---|
| shape | `[24, 257, 1408]` |
| dtype đo được | **`float32`** |
| `batch_sizes_seen` | `[10, 10, 4]`, batch cuối lẻ 4 dòng |
| `validator_accepts_adapted_contract` | **True** (float32, contract adapted) |
| `validator_rejects_default_contract` | **True** (cùng bytes đó, contract float16) |
| `decision_is_dtype_only` | **True** — identity guard thoả, nên khác biệt duy nhất là dtype |
| `validator_rejects_float64_under_adapted_contract` | **True** — không phải "mở cổng" |

**Một thiết kế sai đã bị bỏ trước khi chạy.** Bản đầu tiên của script ghi thẳng vào
`<checkout>/features/FashionIQ` — đúng chỗ preflight và prepare tìm. Nếu để nguyên, một cache 24 ảnh
sẽ nằm chỗ mà lần chạy sau coi là cache thật, và file `val_name_to_idx.json` (312 726 B) — bằng chứng
byte cuối cùng về identity của cache 22.49 GB đã xoá — sẽ bị ghi đè. Bản đang dùng chỉ ghi vào
`tempfile.mkdtemp()`, và ghi lại `real_cache_location_written: false`.

Trạng thái thật của cache gốc sau các lần chạy: `features/FashionIQ/val_name_to_idx.json`,
312 726 B, sha256 `5cfc07a1…`, 15415 entry, mtime 23:05 — **không bị đụng**.

### 3.16.4 Kiểm chứng regression (dtype contract)

`workbench/tests/test_tme_feature_dtype_contract.py` — **14/14 pass**, chạy trên production resolver
và production validator:

- mặc định vẫn `float16`; model chỉ *khai* profile mà operator **không** chọn → vẫn `float16`;
- chọn đúng → `float32`; override không đụng parameter khác; resolver trả copy;
- model khai adaptation **khác** mà operator chọn `sm_120` → **refuse** (`UNKNOWN_RUNTIME_ADAPTATION`),
  không im lặng fallback;
- lower bound: 10.47 ↔ 20.94 GiB đúng tỉ lệ 2×;
- validator: float32 **được nhận** chỉ khi contract mong float32; **bị từ chối** dưới float16;
  float64 **bị từ chối**; shape sai **bị từ chối**; thứ tự gallery đổi **bị từ chối**; receipt digest
  cũ **bị từ chối**.

### 3.16.5 Batch-size: dùng qua đường chuẩn, không qua sửa tay

`WORKBENCH_TME_BATCH_SIZE` là đường chuẩn có sẵn (`batch_size_override_env_var` trong contract,
đọc qua `prepare_model.tme_generation_parameters`, ghi vào receipt và so lại khi reuse). Không có
"chỉnh tay session-only" nào được thêm; giá trị hiện hành của phiên là operational, và batch đã đo
(`[10,10,4]`) chỉ dùng cho smoke.

### 3.16.6 Giới hạn còn lại — cập nhật sau khi dọn đĩa (2026-11-07)

Ba giới hạn ghi ở bản trước đã được **gỡ bỏ bằng phép đo thật**, không phải bằng lập luận:

- **Đã chạy lại full 3 category của TME qua đúng `workbench.replay.tme`.**
  `tme/measurements/tme_normal_path_full.json`, exit 0, macro **54.9229 / 74.8384** (54.922901590665184 /
  74.83842174212138) — khớp **từng chữ số** với số đo trước đó khi TME còn phải chạy qua session runner.
  Cache sinh lại có sha256 `9fead93ac746…` **trùng khít** cache đã bị xoá, và
  `feature_name_to_idx_sha256` `5cfc07a1…` trùng khít file mapping gốc còn sống — nên đây là cùng một
  đầu vào, không chỉ cùng kích thước.

  **`feature_cache_provenance: VERIFIED_PRIOR_RUN`**, `feature_cache_unverified_inputs: []`. Đây là
  điểm từng hỏng và đã sửa: lượt chạy đầu ghi `UNVERIFIED` vì lúc sinh cache tôi đặt
  `WORKBENCH_TME_BATCH_SIZE=32` nhưng lúc eval **không** đặt, nên resolver tính ra 512 và receipt ghi 32
  ⇒ hai bên lệch `recipe.batch_size`. Chạy lại với đủ biến môi trường thì khớp hoàn toàn. Bài học ghi
  lại: batch là tham số vận hành **có mặt trong receipt**, nên phải đặt giống nhau ở cả hai bước.
- **dtype vẫn là kết luận theo runtime**, không phải phán quyết tuyệt đối: `sm_120` đo được float32,
  contract lịch sử giữ float16 làm default. Không có runtime thứ hai để đo trong phiên này.
- **Giới hạn tài nguyên đã hết**: đĩa đủ chỗ sau khi dọn checkpoint cũ, nên lượt ranking 3-category
  chạy được thật.

Hai điều kiện vận hành **bắt buộc** phải ghi lại cho lượt đo này, vì không có chúng thì không chạy được.
Cả hai đều là tham số vận hành, không phải ngữ nghĩa:

| điều kiện | giá trị dùng | default của tác giả | vì sao |
|---|---|---|---|
| `WORKBENCH_TME_QFORMER_BS` | 64 | 256 | gallery float32 là 20.94 GiB trên thẻ 15.44 GiB; `inference()` ném CUDA OOM. Đúng như docstring pinned khuyên ("reduce Qformer_bs and query_bs") |
| `WORKBENCH_TME_QUERY_BS` | 8 | 32 | cùng lý do |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | — | giảm phân mảnh allocator |
| `WORKBENCH_TME_BATCH_SIZE` | 32 | 512 | sinh cache, ghi trong receipt |

Lệnh tái lập đầy đủ (cả hai bước phải dùng **cùng** `WORKBENCH_TME_BATCH_SIZE`, vì giá trị đó nằm trong receipt):

```bash
export CIR=/home/heheboiz/data/cir
cd $CIR
export E1=$CIR/workbench/artifacts/checkpoint_selection/20261106T002617+0700
export WORKBENCH_TME_PYTHON=$CIR/workbench/envs/tme-adapted/bin/python
export WORKBENCH_RUNTIME_ADAPTATION=sm_120
export WORKBENCH_TME_BATCH_SIZE=32

# sinh cache (đã chạy; cache hiện tại có sha 9fead93ac746…)
workbench/envs/test-adapted/bin/python workbench/scripts/prepare_model.py \
  --model tme --execute --allow-preparation --allow-gpu-preparation

# eval full 3 category
WORKBENCH_TME_QFORMER_BS=64 WORKBENCH_TME_QUERY_BS=8 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
workbench/envs/tme-adapted/bin/python -m workbench.replay.tme \
  --checkpoint-root $E1/cohort/ckpt/tme --dataset-root data/FashionIQ \
  --source-root workbench/third_party/TME --output tme_full.json
```

Đường sinh cache cũng phải sửa: `src/precompute.py` gom mọi batch vào list rồi `torch.cat`, cần **hai**
bản sao tensor cùng lúc. Đo được 28.76 GiB RSS cho kết quả fp16 10.47 GiB (SIGKILL), và ở float32 tensor
đã là 20.94 GiB nên không thể chạy trên máy 31 GiB RAM. Patch được khai báo qua `approved_source_patches`
(preallocate một lần rồi copy từng batch), đã đo **bit-identical** với đường native (`max_abs_diff 0.0`).
Sửa ở đây là **chiến lược cấp phát**, không phải số học — model, preprocess, thứ tự loader, autocast và
dtype/file đầu ra đều giữ nguyên.

Một bug thật đã lộ ra trong lúc chạy: bộ ghi receipt dùng `len(expected)` (số tên **duy nhất**, 15415)
làm số hàng, trong khi FashionIQ val liệt kê **15536 hàng** và `_tme_validate` luôn dùng `len(names)`.
Hai bên lệch 121 hàng nên **không receipt nào ghi được**, và một cache đúng bị báo "shape invalid".
Đã sửa bộ ghi về `len(names)` cho khớp validator.

### 3.16.7 Bounded runner ≡ native runner (đã đo, không suy luận)

Runner preallocate-then-copy của phiên trước từng bị nghi là đổi hành vi. Đã kiểm bằng cách viết lại **đúng** đường native (`src/precompute.py:34-46`: gom list rồi `torch.cat`) và chạy song song với đường bounded trên **cùng ảnh thật, cùng model, cùng preprocess, cùng thứ tự**, có chủ đích tạo **batch cuối lẻ**: 5 ảnh, batch 2 ⇒ các batch 2 / 2 / **1**.

| | shape | dtype | names |
|---|---|---|---|
| native (list + cat) | `[5,257,1408]` | float32 | 5 |
| bounded (preallocate + copy) | `[5,257,1408]` | float32 | 5 |

**Byte-identical: `True`. `max_abs_diff = 0.0`. Thứ tự tên trùng khít.** Vậy khác biệt duy nhất giữa hai runner là **bộ nhớ**, không phải số học — và batch cuối lẻ không gây lệch. Không loại ảnh gallery nào khỏi phép so này (chỉ cắt số lượng vì cache đầy đủ là 22.5 GiB và đây là kiểm tra tương đương, không phải đo lại).

Giới hạn của kết luận: phần này chứng minh **hai runner tương đương nhau**, không chứng minh dtype `float16` trong contract là sai cho mọi runtime — xem lý do ở trên.

Không sửa `float16 → float32` trong contract lịch sử vì lý do khoa học: một runtime không đủ để phủ định contract cho mọi runtime. Giá trị lịch sử vẫn là default; runtime đã đo được khai riêng trong `runtime_overrides.sm_120`.

## 3.17 Dọn checkpoint và cache cũ (2026-11-07)

Bạn cho phép bỏ workflow cũ và làm lại. Việc dọn dựa trên **nội dung file** và **registry hiện tại**,
không dựa vào tên thư mục hay việc file đó có phải do tự train hay không.

### 3.17.1 Cách phân loại

Mỗi file weight được mở thật bằng `torch.load` rồi phân loại theo nội dung:

| nội dung | kết luận | xử lý |
|---|---|---|
| dict có `model` / `optimizer` / `state_dict` / key `.weight`·`.bias` | **weight model đã train** | xoá |
| tensor thô, hoặc dict có `sample_ids`/`features`/`timesteps` | **feature dump / kết quả đo** | **giữ** |
| nằm trong đường dẫn registry hiện tại khai báo | checkpoint của cohort 17 model | **giữ** |

Nhờ vậy không xoá mù theo đuôi `.pt/.pth/.pkl/.bin`: 12 file `.pt` hoá ra là feature dump
(`teacher/outputs/**`, `outputs/diagnostics/**`) và được giữ nguyên.

### 3.17.2 Đã xoá

| nhóm | số file | dung lượng | lý do |
|---|---|---|---|
| checkpoint của workflow TAPER/iag_srme cũ (`outputs/`) | 142 | ~93.9 GiB | kiến trúc đã bị gỡ khỏi repo (`LEGACY_CLEANUP_AUDIT.md`); loss/objective không còn consumer nào |
| checkpoint TAPER trong `runs/` | 8 | 1.72 GiB | cùng workflow TAPER; metadata (`metrics_val.jsonl`, `config_resolved.yaml`) **vẫn giữ** |
| trọng số trùng / sai dataset của CLVC và TG-CIR | 10 | 2.50 GiB | 6 file CLVC trùng byte-for-byte với bản canonical trong `checkpoints/clvc/` (đối chiếu `selected_bundle_digests.json`); Fashion200K/Shoes/CIRR không thuộc FashionIQ |
| **tổng** | **160** | **~115.68 GiB** | |

Cache feature cũ đã xoá trước đó trong cùng đợt dọn: `features/cirr/csmcir` (37 GiB),
`features/fashioniq/csmcir` (29 GiB), `features/fashioniq/fgclip2-base` (14 GiB) — tổng **78.80 GiB**.
Cả ba do các script `precompute_*` **đã bị gỡ khỏi repo** sinh ra, và `grep` trên toàn bộ code hiện
hành không còn tham chiếu nào (chỉ `LEGACY_CLEANUP_AUDIT.md` nhắc tên).

### 3.17.3 Giữ lại và vì sao

| đường dẫn | dung lượng | lý do |
|---|---|---|
| `workbench/artifacts/checkpoint_selection/20261106T002617+0700/checkpoints/`, `cohort/`, `downloads/` | 22 checkpoint | cohort 17 model đang dùng; **22/22 còn đủ**, verify lại đúng sha256 |
| `workbench/third_party/TME/weight/{eva_vit_g,blip2_pretrained}.pth` | 2.65 GiB | base weight bắt buộc của TME |
| `workbench/third_party/TME/features/FashionIQ/` | 20.94 GiB | cache vừa sinh lại, sha `9fead93a…` |
| `workbench/third_party/ENCODER/open_clip_pytorch_model.bin` | 577 MiB | base weight của ENCODER |
| `data/**` (CIRR, FashionIQ, GloVe PKL) | 25 GiB | dataset + caption — **không** phải weight model |
| `teacher/outputs/**`, `outputs/diagnostics/**` | 0.7 GiB | **feature dump và audit artifact**, không phải trọng số (đã mở kiểm) |
| `workbench/envs/**` | 8.8 GiB | môi trường chạy |
| `outputs/**/*.{yaml,json,jsonl,md,log}`, `runs/**/*.{json,yaml,jsonl}` | — | cấu hình, metric, log, báo cáo — đều giữ |

### 3.17.4 Dung lượng

| mốc | trống | ghi chú |
|---|---|---|
| đầu phiên | 24.70 GiB (26 525 630 464 B) | trước mọi thao tác |
| sau xoá cache feature | 107.10 GiB | −78.80 GiB cache |
| sau xoá checkpoint cũ | 201.92 GiB (216 810 348 544 B) | −115.68 GiB weight |
| **net thu hồi** | **+177.22 GiB** (190 284 718 080 B) | |

Net **nhỏ hơn** tổng đã xoá (194.48 GiB) vì trong lúc dọn đã **sinh lại** cache TME 20.94 GiB
(`val_images.pt`, sha `9fead93a…`) để chạy verification. Nếu tính riêng phần xoá thì đúng 194.48 GiB.

Phân vùng 630 GiB, dùng 67%. Xoá thực (`os.remove`), không thùng rác, không thư mục backup trên cùng ổ.
Không đi theo symlink ra ngoài repo (107 975 symlink đều nằm trong `envs/`, trỏ tới conda — để nguyên).

### 3.17.5 Đã kiểm lại sau khi xoá

- 22/22 checkpoint cohort còn đủ; 11 entry của `selected_bundle_digests.json` verify đúng **bytes + sha256**.
- `workbench/envs/test-adapted` chạy **67 test** liên quan (TME preparation, dtype contract, integrity,
  approved patch, verified declarations) — pass.
- **TME full verification qua đường chuẩn**: exit 0, macro 54.9229 / 74.8384, khớp chính xác số đo cũ —
  tức là việc dọn đĩa không làm mất hay đổi kết quả nào (mục 3.16.6).
- Quét lại toàn repo: không còn file weight nào ngoài keep-list, trừ 2 `candidate_features.pt` (feature dump, cố ý giữ).

### 3.18. CSMCIR — chạy thật bằng caption công khai, và vì sao D-8 là blocker ma

**Kết quả: exit 0, macro R@10 57.0507 / R@50 77.2986 (mean 67.1747), paper 57.07 / 77.27.**
Hai run exit 0 (`f564570e`, `f03887d2`) cho cùng số; chi tiết ở 3.18.5 và 3.23.5.
Đây là số đo độc lập đầu tiên của entry này; trước đó entry bị đánh `BLOCKED` vì thiếu caption.

#### 3.18.1 Hai yêu cầu sai trong báo cáo cũ

Báo cáo trước nói dataset constructor "đòi `COT_ours2/bert_captions/fashioniq/cap.*.val.json` +
`captions/keywords_in_mods_*.json`". Sai. Hai đường đó chỉ xuất hiện trong
`FashionIQDataset_imageedit` (`data_utils_csmcir.py:175`), và `grep -rn FashionIQDataset_imageedit`
trên toàn checkout chỉ trả về **một** dòng — chính định nghĩa. Không nơi nào khởi tạo nó.

Grep không phải call path, nên bằng chứng là lần chạy thật: patch `builtins.open`, rồi dựng đúng các
constructor mà `blip_validate_fashioniq` dựng (`validate_blip_csmcir.py:596,598`) và gọi
`extract_index_blip_caption_features` như nhánh gallery. Kết quả
(`csmcir/call_path_probe.txt`):

| đo | giá trị |
|---|---|
| file được mở | **12** |
| `imageedit` được khởi tạo | `false` |
| đọc `bert_captions` | **0** |
| đọc `keywords_in_mods` | **0** |
| đọc `COT_ours2` | **3** (gallery caption) |

Vậy D-8 yêu cầu hai file mà **đường eval không bao giờ mở**. Báo cáo cũ mâu thuẫn với audit là do
báo cáo tin vào docstring thay vì đo. Đã sửa.

#### 3.18.2 Caption cần thật — đã có, khớp bytes chính thức

Ba file công khai, tải bằng `download_auxiliary_assets.py` (không copy tay):

| category | sha256 | bytes | records | so với gallery |
|---|---|---|---|---|
| dress | `b7129baf…` | 2 817 386 | 3817 | 3817/3817 |
| shirt | `bbe6f689…` | 5 015 979 | 6346 | 6346/6346 |
| toptee | `acdacc3d…` | 3 929 303 | 5373 | 5373/5373 |

Cả `COT_ours2/fashioniq/` và `fashionIQ_dataset/qwen_captions/` đều khớp three-way. `Final_Caption`
là string không rỗng ở **100%** record; reference và target của mọi triplet đều phủ.

**121 ID trùng shirt/toptee: khác nhau hoàn toàn (0/121 giống).** Merge order
`utils_csmcir.py:97-103` là dress → shirt → toptee qua `dict.update`, nên **toptee thắng** ở cả 121 ID.
Ghi nhận, không "sửa" — sửa sẽ đổi input so với tác giả.

#### 3.18.3 Checkpoint buộc đúng một class mà repo không có source

`fashioniq_tuned_clip_best.pt` khớp `c8812071…` và 2 720 615 859 byte. Nhưng nạp vào class mặc định
thì `KeyError: 'Blip2QformerCirAlignPromptCSMCIR'`: checkpoint lưu dưới key
**`Blip2QformerCirAlignPromptTargetImageCaptionshiyan92`**.

Đối chiếu hai class:

| | tensor | thiếu | thừa | shape lệch |
|---|---|---|---|---|
| CSMCIR (source) | 860 | 0 | 1 (`token_importance`) | 0 |
| `shiyan92` (từ `.pyc` tác giả) | 860 | **0** | 1 (`token_importance`) | **0** |

`token_importance` (32 giá trị đã train, 0.993–1.007) không nằm trong `state_dict` của **cả hai**
class. Nguyên nhân đã kiểm trực tiếp, không suy đoán:

| đo trên instance thật | CSMCIR | `shiyan92` |
|---|---|---|
| `len(state_dict())` | 860 | 860 |
| `"token_importance" in state_dict()` | **False** | **False** |
| `isinstance(m.token_importance, nn.Parameter)` | **False** (Tensor) | **False** (Tensor) |
| đăng ký trong `named_parameters()` | False | False |
| đăng ký trong `named_buffers()` | False | False |
| giá trị lúc chạy | **toàn 1** | **toàn 1** |

Tác giả viết `self.token_importance = nn.Parameter(torch.ones(n)).unsqueeze(0).unsqueeze(-1)`.
`unsqueeze` trả **Tensor thường**, nên parameter bị mất đăng ký ngay tại dòng gán.

Hệ quả cần nói rõ: **32 giá trị đã train trong checkpoint không bao giờ được dùng**. Cả hai class đều
chạy với vector ones lúc init, bất kể checkpoint chứa gì. Nên việc `load_state_dict(strict=False)` bỏ
qua khoá đó **không** làm sai lệch số đo — nó chỉ là hệ quả của một khiếm khuyết upstream, và
`strict=True` cũng sẽ từ chối vì lý do y hệt. Đây không phải chỗ tôi được phép "sửa": gán lại thành
`nn.Parameter` sẽ đổi hành vi forward so với tác giả.

Module `shiyan92` **không có `.py`** ở revision pin; tác giả chỉ publish `.pyc` CPython 3.10 (khớp đúng
env). Import trực tiếp `.pyc` đó chạy được và tự đăng ký. Vì vậy run dùng đúng class đã lưu checkpoint.

#### 3.18.3b Hai class có cùng bề mặt evaluator gọi

Vì class chạy là `shiyan92` chứ không phải `blip2_cir_align_prompt_csmcir` của entrypoint, phải kiểm
ràng các method mà evaluator gọi đều tồn tại và không bị thừa kế từ đâu khác:

| method | CSMCIR tự định nghĩa | `shiyan92` tự định nghĩa |
|---|---|---|
| `inference` | có | có |
| `forward` | có | có |
| `extract_target_caption_features` | có | có |
| `extract_reference_features` | không (không bên nào có) | không |

MRO của cả hai: `…→ Blip2Base → BaseModel → Module`, cùng gốc. Nên evaluator gọi đúng cùng một bộ
method. Điều này **không** chứng minh hai forward giống nhau từng byte — chỉ chứng minh bề mặt gọi
khớp và state_dict khớp từng tên+shape. Macro trùng paper là quan sát số học củng cố, không phải
chứng minh.

#### 3.18.4 Bốn delta vận hành, tất cả đều ghi lại

Đây **không** phải thay đổi ngữ nghĩa: cùng model, cùng gallery, cùng caption, cùng thứ tự.

| # | thay đổi | vì sao | loại |
|---|---|---|---|
| 1 | checkout derived riêng thay vì sửa `third_party/CSMCIR` | preflight yêu cầu derived checkout; patch in-place làm 19 test đỏ | vị trí patch |
| 2 | khôi phục 1 dòng import `shiyan92` **sau** khối `try` | khối 98 import bị patch bọc `try/except ImportError`; import đầu tiên thiếu sẽ nuốt cả khối | cần để nạp checkpoint |
| 3 | `save_memory=True` ở cả extractor và predictor | mặc định giữ mọi batch gallery trên GPU rồi `vstack`: cần thêm 5.15 GiB, card chỉ còn 1.41 GiB. Đường `save_memory` là của chính tác giả | vận hành |
| 4 | query batch 64 → 8 | `inference` cần matmul 10.49 GiB khi chỉ còn 8.82 GiB | vận hành |

Delta 1–2 dùng `.pyc` tác giả publish, **không** tạo source mới. Delta 3–4 chỉ đổi chỗ cư trú của
tensor và số query mỗi lượt.

#### 3.18.5 Số đo

Nguồn: **hai** run exit 0, mỗi run có thư mục log riêng — `f564570e`
(`artifacts/logs/csmcir__fashioniq__20261107T054343058424Z_csmcir_fashioniq_f564570e/`) và `f03887d2`
(`artifacts/logs/csmcir__fashioniq__20261107T054933277898Z_csmcir_fashioniq_f03887d2/`). R@10/R@50 dưới
đây đọc từ `aggregate_report.json` của cả hai run và trùng nhau đến chữ số cuối; cột `queries`/`gallery`
lấy từ `csmcir/measurement.json` (`aggregate_report.json` không xuất số query).

| category | R@10 | R@50 | queries | gallery |
|---|---|---|---|---|
| dress | 52.20624804496765 | 74.86366033554077 | 2017 | 3817 |
| shirt | 57.75269865989685 | 76.1040210723877 | 2038 | 6346 |
| toptee | 61.19326949119568 | 80.92809915542603 | 1961 | 5373 |
| **macro** | **57.05073873202006** | **77.29859352111816** | | |

Paper: 57.07 / 77.27. Chênh 0.02 / 0.03 điểm. `Missing keys []`, protocol
`fashioniq_original_split`, eval noise 0%, full gallery, reference eligible.

Chỉ `f03887d2` ghi lại knob thật đã tới evaluator; `f564570e` có
`environment.evaluator_environment_knobs: null`. Chi tiết hai run: `csmcir/runs/README.md` và 3.23.5.

#### 3.18.6 Nhãn bằng chứng — đọc kỹ trước khi trích dẫn

Đây là **public-caption path-mapped research run**, **không phải reproduction chính thức**:

- **`downloaded-file provenance`: VERIFIED.** Bytes khớp sha256 công bố tại revision pin.
- **`historical evaluation-input equivalence`: NOT PROVEN.** Không có tuyên bố nào của tác giả nói
  ba file này là caption họ dùng để ra số paper. Hash trùng giữa hai bản copy **không** chứng minh
  được điều đó.
- **Không** dùng việc score gần paper để nâng lên official parity.

#### 3.18.6b Cách tái lập hiện hành — xem 3.23

**Lệnh hiện hành duy nhất là recipe ở mục 3.23.5b.**
Mục này chỉ còn giữ bốn delta vận hành và nhãn bằng chứng; cách chạy đã đổi.

##### HISTORICAL — lệnh của lần đo đầu (2026-11-07 sáng), KHÔNG dùng nữa

Đây là chuỗi đã tạo ra số ở 3.18.5 trên checkout `workbench/artifacts/csmcir_derived`, ghi lại để
truy vết. **Không chạy lại**: patch hiện hành đã biến hai tham số thành env-overridable, nên ba helper
vận hành **dừng có chủ đích** trên **mọi** cây mang derived digest hiện hành (đo lại 2026-11-07, xem
`csmcir/recipe_check.txt`):

| helper | kết quả trên cây đã áp patch hiện hành |
|---|---|
| `apply_operational_save_memory.py` | `expected exactly 1 metric call site, found 0; refusing to guess` → exit 1 |
| `thread_save_memory.py` | `metric call block not found; refusing to guess` → exit 1 |
| `set_query_batch.py` | `expected exactly 1 query loader, found 0` → exit 1 |

Lý do: patch `csmcir_validate_visualization_opt_in.patch` đã biến hai tham số thành env-overridable
(`WORKBENCH_CSMCIR_QUERY_BS`, `WORKBENCH_CSMCIR_SAVE_MEMORY`), nên anchor literal mà các helper tìm
không còn tồn tại. Chúng fail-closed chứ không sửa mù — đúng thiết kế. Riêng builder vẫn tái tạo đúng
`0ec13443…` / `a057e8f3…` khi chạy lại, nên phần dựng cây trong chuỗi dưới đây vẫn đúng; chỉ ba bước
sửa tay là hết hiệu lực.

**Phân định bản sao (authoritative vs working copy):** Thư mục `workbench/docs/reviews/checkpoint_selection/20261107T021500+0700` là **hồ sơ lưu trữ chính thức** (authoritative archived record, lập chỉ mục qua `EVIDENCE_INDEX.json`) chứa phiên bản script chuẩn đã được làm cứng (chặn digest mismatch, tự link dataset). Bản sao tại `workbench/artifacts/checkpoint_selection/20261107T021500+0700` chỉ là **working copy** của phiên chạy. Lệnh tham chiếu script chuẩn từ bundle lưu trữ `E2`.

```bash
# HISTORICAL ONLY — không phải bước hiện hành; thiếu đầu vào <author_pyc> phải cung cấp. Xem recipe hiện hành ở 3.23.5b.
export CIR=/home/heheboiz/data/cir
cd $CIR
export WORKBENCH_RUNTIME_ADAPTATION=sm_120
export WORKBENCH_CSMCIR_PYTHON=$CIR/workbench/artifacts/environments/cir-005ef4e1b02999a5/bin/python
# Hồ sơ lưu trữ chính thức (authoritative record); bản artifacts tương ứng chỉ là working copy:
E2=workbench/docs/reviews/checkpoint_selection/20261107T021500+0700
workbench/envs/ctl312h/bin/python $E2/csmcir/scripts/build_csmcir_derived.py \
  workbench/artifacts/csmcir_derived --with-shiyan92-pyc <author_pyc>
workbench/envs/ctl312h/bin/python $E2/csmcir/scripts/apply_operational_save_memory.py
workbench/envs/ctl312h/bin/python $E2/csmcir/scripts/thread_save_memory.py
workbench/envs/ctl312h/bin/python $E2/csmcir/scripts/set_query_batch.py \
  workbench/artifacts/csmcir_derived/src/validate_blip_csmcir.py 8
cd workbench/artifacts/csmcir_derived/src
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $WORKBENCH_CSMCIR_PYTHON \
  validate_blip_csmcir.py --dataset fashionIQ \
  --blip-model-name blip2_cir_align_prompt_target_image_caption_shiyan92 \
  --blip-model-path $CIR/workbench/artifacts/checkpoints/csmcir/fashioniq_tuned_clip_best.pt
```

#### 3.18.7 Run id: dòng `[RUN] log directory` không phải run id

`evaluate_models.py` in `[RUN] log directory: <path>` **trước** khi chạy, dùng id từ `make_run_id()`
(`evaluate_models.py:551`, có `uuid.uuid4()`), rồi `execute()` sinh một id **khác**
(`evaluate_models.py:2090`). Đường in ra là **dự đoán**, không phải run. Hệ quả thực tế: một bản
bằng chứng cũ đã trích một run id **không tồn tại trên đĩa và chưa từng tồn tại**. Đã sửa: mọi run id
trong bundle này lấy từ `workbench/artifacts/logs/<run_id>/command.json`, và bundle liệt kê hai run
exit 0 ở `csmcir/runs/`.

### 3.19. Ba việc còn lại: SPRC ViT-g, PAIR, và recipe GloVe

#### 3.19.1 SPRC ViT-g — vẫn không tải được, và lý do cụ thể

Link trong registry (`https://1drv.ms/u/s!Aj0q22vyiZbabnUya4mnufIBtYI?e=n4ZVKj`) trả về:

| thử | kết quả |
|---|---|
| `HEAD` theo dõi redirect | `301` → `onedrive.live.com/…?redeem=…`, rồi **`403`** |
| `GET` với User-Agent trình duyệt | `200 text/html`, 327 203 byte — **trang share shell**, không phải file |
| tìm `downloadUrl` / `.pt` / `authkey` trong HTML | không có |
| `api.onedrive.com/v1.0/shares/u!…/driveItem` | `401 Unauthorized` |
| `api.onedrive.com/v2.0/…` | `400 Bad Request` |
| `…&download=1` | `403`, body là HTML lỗi 871 byte |

Kết luận: share yêu cầu đăng nhập/phiên SharePoint, và **không** có đường ẩn danh. Không lưu HTML thành
`.pt` (đó là bẫy đã nêu trong prompt). **D-1 vẫn mở**: cần bạn tải thủ công bằng trình duyệt đã đăng
nhập, rồi đặt vào `workbench/artifacts/checkpoints/sprc/sprc_fiq.pt`.

#### 3.19.2 PAIR — upstream không đổi, kết luận giữ nguyên

`git ls-remote https://github.com/iLearn-Lab/ICASSP25-PAIR` → **một** ref,
`bfb0b98cdb4c44f6b9190baaa84d783a77dd888f`, trùng đúng commit đã vendor. Không có nhánh mới, không có
bản sửa entrypoint. `https://huggingface.co/api/models/ZhihengFu/PAIR` trả **401**, nên model card
không đọc được ẩn danh và không có tuyên bố nào ghép `pair-B1.pt`/`pair-B2.pt` với FashionIQ.

Vì vậy PAIR giữ `EXPLORATORY_PUBLIC_MAPPING` + `COMMAND_BLOCKED_SOURCE_ENTRYPOINT`. Không có giả thuyết
mới có căn cứ, nên không mở lại bốn mapping B1/B2.

#### 3.19.3 Recipe GloVe — tái lập được, và khớp byte với artifact đã đo

Script tách riêng (`glove_recipe/scripts/dcnet_glove_recipe.py`), chạy **hai lần** với `--seed 0`:

| đo | giá trị |
|---|---|
| vocab tokens | 3886 |
| vocab words | 3491 |
| OOV điền ngẫu nhiên | 13 |
| run1 == run2 | **true** (9/9 file + `glove_vecs.pkl`) |
| khớp artifact đã đo trước | **10/10** |

Điểm quan trọng: **không** tuyên bố tái tạo byte-identical artifact của tác giả. `process_cap.py` lấy
mẫu OOV bằng `np.random.normal` **không seed**, nên bản của tác giả không thể tái lập và chưa từng
được công bố. Cái đã chứng minh là: recipe seed cố định cho cùng hash giữa hai lần chạy, và artifact
DCNet đang được đo **trùng khít** 10/10 file của recipe này.

Script đọc canonical `captions/` và ghi ra `--out` riêng — không ghi đè input (khác script gốc của
tác giả, vốn ghi vào chính `captions/`). Đã sửa `glob('cap*.json')` thành chỉ `.json` để recipe
idempotent (glob gốc khớp cả `.pkl` nó vừa ghi).

Diff đề xuất cho `preparation_contracts.yaml` nằm ở mục 6 dưới nhãn `WORKBENCH_SEEDED_RECIPE`; **chưa
áp dụng**.

### 3.20. Test suite trên Python 3.12 — trạng thái TRƯỚC khi sửa (số cuối ở 3.21.5)

Interpreter 3.12 không có sẵn trong repo; `python3.12` của distro thiếu `pip`. Đã dựng
`workbench/envs/ctl312h` bằng `uv` (CPython 3.12.11, **có** `Python.h`) — user-local, không `sudo`,
không đụng Python hệ thống.

| môi trường | kết quả |
|---|---|
| 3.11 `test-adapted` (trước buổi này) | 2044 collect, 8 module không collect được do PEP 701 |
| 3.12.3 `ctl312`, **không** headers | 2105 passed, **61 failed**, 8 errors |
| 3.12.11 `ctl312h`, có headers (trước khi sửa) | 2155 passed, 47 skipped, 20 failed |
| 3.12.11 `ctl312h` — **cuối lượt 20261107 (3)** | **2203 passed, 0 failed, 47 skipped** = 2250 collected (3.24) |

Chênh lệch 41 failure giữa hai env 3.12 là `Python.h` — fixture test tự biên dịch extension C. Đó là
lỗi công cụ, không phải code.

20 failure ở trạng thái đó (số cuối là 0 — xem 3.24). Bảng dưới là chẩn đoán lúc đó, giữ làm lịch sử:

| nhóm | số | nguyên nhân thật |
|---|---|---|
| `release_verify.sh` | 7 | script dùng `/usr/bin/python3` của distro (không `pip`, không `packaging`). Đã tạo `.venv` trỏ env có headers nên hết lỗi `packaging`; phần còn lại là **8 record GPU ký bằng operator key của phiên trước** ⇒ verdict ra `RELEASE_BLOCKED` thay vì `ENGINEERING_INCOMPLETE`. Đây là host-state di sản, không phải defect code |
| frontend | 4 | `frontend/dist` chưa build; cần `npx vite build` (cần `npm install`) |
| vendor digest | 2 | `DCNet/base/base_trainer.py` đã áp patch từ phiên trước nên lệch manifest — đúng, đã khai báo |
| khác | 7 | `TESTED_SOURCE_SHA` trong handoff cũ, planner pandas pin, mutation m4, `test_pc_launcher` |

**Đã dọn** 18 record GPU do test để lại (`probe_source=override` ×16, `forged` ×2) trong
`workbench/artifacts/gpu_capabilities/` — chúng là rác test, không phải bằng chứng.

Một test đã sửa vì nó pin **trạng thái host** chứ không pin invariant:
`test_release_gate_scenarios.py::test_scenario_d_…` khẳng định literal
`{allow_preparation, allow_gpu_eval}`, nhưng `allow_preparation` chỉ được cấp **khi** canonical link
chưa tồn tại. Test pass trên clone sạch và fail ngay sau khi preparation chạy lần đầu. Đổi thành
assertion kép: `allow_gpu_eval` luôn có, và `allow_preparation` ⟺ link chưa sẵn sàng.

### 3.21. Kiểm chứng lại bốn phát hiện review (2026-11-07, lượt 2)

#### 3.21.1 Recipe GloVe: manifest thiếu `glove_vecs.pkl` — CONFIRMED, đã sửa

Bản trước của `dcnet_glove_recipe.py` ghi `glove_vecs.pkl` **trước** khi khởi tạo danh sách `written`,
nên manifest chỉ có 9 file trong khi recipe sinh ra 10. Kết luận "10/10" trong báo cáo trước dựa vào
một phép đo riêng lẻ ngoài manifest, không kiểm được lại từ bundle.

Đã sửa và chạy lại hai lần với seed 0:

| đo | giá trị |
|---|---|
| file trong manifest | **10** (gồm `glove_vecs.pkl`) |
| run1 == run2 | **true** |
| khớp 10/10 file artifact DCNet đã dùng để đo | **true** |

Nhãn giữ là `WORKBENCH_SEEDED_RECIPE`. Không gọi là byte-identical với artifact tác giả:
`process_cap.py` lấy mẫu OOV bằng `np.random.normal` **không seed**, các byte đó chưa từng công bố.

#### 3.21.2 Quy trình tái lập: ba lỗi — CONFIRMED, đã sửa

| # | lỗi | bằng chứng | sửa |
|---|---|---|---|
| a | đường dẫn checkpoint trong hướng dẫn sai một cấp | từ `artifacts/csmcir_derived/src`, `../../../checkpoints/csmcir` **không tồn tại**; `../../checkpoints/csmcir` đúng | hướng dẫn dùng đường tuyệt đối `$CIR/workbench/artifacts/checkpoints/csmcir/...` |
| b | `build_csmcir_derived.py` chỉ copy `src`, không stage data | `grep fashionIQ_dataset` trong builder = 0 hit; cây mới chết ở lần đọc dataset đầu tiên | thêm `--dataset-root`, tự tạo cả hai symlink, kiểm `qwen_captions/` tồn tại trước khi link |
| c | `csmcir_call_path_probe.py` tính CHECKOUT sai từ bundle | `parents[1]` = `.../csmcir`, nên tìm `.../csmcir/third_party/CSMCIR` — không tồn tại | tìm repo root bằng cách đi lên tới khi thấy `workbench/third_party/CSMCIR/src`; thêm `--checkout` |

Builder cũng được làm cứng: **dừng** nếu digest của approved patch không khớp `derived_sha256`, và kiểm
sha256 của `.pyc` tác giả trước khi dùng (`--expect-pyc-sha256`). Trước đó nó chỉ in `MISMATCH` rồi
tiếp tục, để lại cây nửa sửa mà sau này đọc như một approved variant.

**Đã chạy lại toàn bộ từ cây mới** (`/tmp/derive_fresh`, không dùng symlink hay file stage của phiên
trước), đúng các lệnh trong hướng dẫn:

```
build_csmcir_derived.py  -> aa68fe5e OK, 84a1d476 OK, 2 symlink, .pyc sha 46d00b0b…
apply_operational_save_memory.py -> 4dd11e11…
thread_save_memory.py            -> f3c35335…
set_query_batch.py 8             -> 9af8d120…
validate_blip_csmcir.py          -> exit 0
```

Kết quả: **dress 52.2062/74.8637, shirt 57.7527/76.1040, toptee 61.1933/80.9281, macro
57.05073873202006 / 77.29859352111816** — khớp **từng chữ số** với mốc. Chuỗi digest cũng khớp cái đã
ghi, nên đường dẫn sửa không làm đổi kết quả.

**Đây là số của lần đo HISTORICAL** (đường B: builder + ba helper cũ, cây `/tmp/derive_fresh` đã xoá).
Không trích lại như bằng chứng hiện hành. Hai run pipeline có thư mục log còn trên đĩa cho **cùng**
macro — `f564570e` và `f03887d2`, xem 3.23.5 — và đó là nguồn phải trích.

#### 3.21.3 Adapter/web và đường chính thức — CONFIRMED (adapter), BLOCKED (tích hợp)

Ba lỗi thật trong `workbench/composition/model_csmcir.py` và adapter lệnh chính thức:

1. `_status()` trả `ready=False` **vô điều kiện**, lý do vẫn nói caption `COT_ours2` thiếu — đã bị
   bác bỏ bằng đo (3.18). Đã sửa lý do cho đúng sự thật.
2. Worker chọn `blip2_cir_align_prompt_csmcir`, rồi `state[model.__class__.__name__]` → `KeyError`
   với checkpoint công bố, vốn lưu dưới `Blip2QformerCirAlignPromptTargetImageCaptionshiyan92`.
3. `CSMCIRAdapter.command` (lệnh chính thức) **không** truyền `--blip-model-name`, nên nó dùng default
   → cùng `KeyError`. Đã thêm flag, khớp cách SPRC khai `--backbone`.

Đo trực tiếp để chứng minh (2): worker với class `shiyan92` + derived root cho `missing 0,
unexpected ['token_importance']`, `rc 0`.

**Nhưng tích hợp vẫn BLOCKED**, và đây là lý do cụ thể:

- Adapter và `evaluate_models.py` đều lấy source là
  `WORKBENCH_THIRD_PARTY_ROOT/<source_dir>` (`composition/model_csmcir.py:146`,
  `evaluate_models.py:blockers`). **Không có** cơ chế trỏ sang derived checkout: preflight bảo
  "register that checkout as the model source" nhưng không có API nào làm việc đó.
- Vì vậy mọi đường đều chạy trên checkout **pristine**, nơi `import lavis.models` chết ngay:
  `ModuleNotFoundError: …blip2_qformer_cir_align_prompt_caption_query`. Đó là blocker thật, đã tái
  hiện, không phải suy đoán.
- Áp patch **in-place** sẽ làm đường chính thức chạy được (DCNet và TME đều đang ở trạng thái DERIVED,
  nên đó là tiền lệ trong repo), nhưng nó mâu thuẫn với `test_csmcir_source_blocker.py`, vốn khẳng
  định workbench *không* patch checkout. Chọn hướng nào là quyết định nghiên cứu → **D-10** — **đã giải quyết ở 3.23** bằng cách gộp vào patch hiện có và tái ký digest, không nới guard.
- Adapter còn một khác biệt ngữ nghĩa không thể vá bằng cấu hình: `score()` của adapter tính
  `reshape(N,32,256)[:,0,:]` rồi `einsum(...).max(-1)` trên vector 8192 chiều, còn evaluator gốc dùng
  `inference()` (attention-weighted token importance rồi max trên 32 query token). Hai đại lượng khác
  nhau, nên **không** dùng chung được. Ghi rõ, không đổi thuật toán.

#### 3.21.4 Báo cáo còn mâu thuẫn — CONFIRMED, đã sửa

| chỗ | cũ | mới |
|---|---|---|
| bảng trạng thái | `CSMCIR UNVERIFIED (blocked) … chưa chạy` | `COMPLETED (research variant) … đã chạy, exit 0` |
| lệnh TME | `(see D-7)` | `(D-7 closed; see 3.16.6)` |

#### 3.21.5 Full suite — 19 → số cuối, có đối chứng baseline

Lần trước tôi nói "20 failure, không liên quan CSMCIR" **mà không có baseline**. Đã dựng worktree riêng
tại `ea5ff05` và chạy cùng interpreter:

| lần chạy | kết quả |
|---|---|
| baseline `ea5ff05` (worktree riêng) | **23 failed**, 2152 passed, 47 skipped |
| sau khi sửa trong lượt 20261107 (2) | 3 failed, 2172 passed, 47 skipped |
| **cuối lượt 20261107 (3) — sạch** | **0 failed**, **2179 passed**, 47 skipped |

Ba failure còn lại đều **có trong baseline**, đã kiểm từng cái (dòng `test_handoff_commands` đã xanh sau khi nâng SHA; giữ lại để đối chiếu):

| test | nguyên nhân thật | vì sao không sửa |
|---|---|---|
| `test_final_hardening_regressions::test_planner_verifier_uses_the_resolved_contract` | LIMN `open_clip_torch==2.20.0` trong registry từ commit `200b7c7` của phiên trước; test còn kỳ vọng hợp đồng `<2.27,>=2.24` | sửa registry hay test đều là quyết định về pin LIMN, không thuộc phạm vi lượt này |
| `test_fresh_clone_delivery::test_every_manifest_path_matches_its_recorded_digest` | đúng ba file có approved patch áp **in-place**: `DCNet/base/base_trainer.py`, `TME/src/precompute.py`, `TME/src/precompute_evaluation.py`; `provenance.json` ghi digest **base** | manifest cần ghi derived digest — thay đổi có chủ đích của phiên trước, không tự sửa expected hash |
| ~~`test_handoff_commands::test_no_release_document_points_at_a_superseded_source_sha`~~ | đã xử lý theo đúng thủ tục có sẵn (tiền lệ A40): nâng `TESTED_SOURCE_SHA` → `5ca7043` tại `FINAL_PC_RELEASE_VERDICT.md` và `PC_PILOT_HANDOFF.md`, kèm số suite đo thật | nay **xanh** (8/8 trong file đó) |
| `test_release_blocker_interpreter::test_a_real_distro_environment_is_not_reported_empty` | **đã sửa** (xem 3.25.1): hai bug thật trong `runtime.inspect_environment_distributions` — `.egg-info` dạng file và glob khớp thư mục giả dạng extension. Nay 85/85 tên chuẩn hóa được giải trình | xanh |

Không nhóm nào bị "làm xanh" bằng cách bỏ assertion hay thêm skip. Việc nâng `TESTED_SOURCE_SHA` không phải cập nhật hash tuỳ tiện: test tự chỉ dẫn và có tiền lệ A40 (`da209b5e`), và SHA mới là commit production thật `5ca7043` chứa đúng các thay đổi của lượt này.

Khác biệt được phân loại từng file, không gộp. Nguyên nhân đã sửa trong lượt này:

| nhóm | cách xử lý |
|---|---|
| `test_frontend_staleness`, `test_pc_launcher` | `npm install` + `npx vite build` — nay xanh (108 passed) |
| `test_circus_adapters_blip` | pin chuỗi `"COT_ours2"` trong lý do đã bị bác bỏ; đổi sang kiểm lý do cụ thể, không rỗng |
| `test_evaluate_models`, `test_backend_config_roots` | pin **lệnh cũ** (thiếu `--blip-model-name`), tức pin một lệnh `KeyError` với checkpoint công bố; cập nhật theo lệnh đúng |

### 3.21.6 PENDING_USER_DECISION phát sinh từ lượt này

| ID | vấn đề | lựa chọn | tác động |
|---|---|---|---|
| ~~D-10~~ | `csmcir` đường chính thức | **ĐÃ GIẢI QUYẾT trong lượt này** (3.23): gộp `shiyan92` vào patch `__init__.py` và tái ký `derived_sha256`/`patch_sha256`; thêm `_apply_approved_source_patches`; khai `.pyc` là asset; sửa `AF_UNIX path too long` và chặn `PYTORCH_CUDA_ALLOC_CONF` | Không nới guard nào; digest vẫn phải khớp chính xác | Không còn gì để quyết cho luồng A 
| ~~D-11~~ | `TESTED_SOURCE_SHA` trong `FINAL_PC_RELEASE_VERDICT.md` chưa tiến sau khi production code đổi trong lượt này | **ĐÃ XỬ LÝ**: `TESTED_SOURCE_SHA` nay là `c13ac29dfaa88eab8f175e7f6d3adc9115b10fd8` trong cả hai tài liệu phát hành | là nghi thức phát hành, không phải lỗi code |

### 3.22. Overlay `WORKBENCH_THIRD_PARTY_ROOT` — thử và bị chặn bởi mâu thuẫn digest

Review đề xuất một đường không đụng vendor: dựng `third_party` overlay, mọi model là symlink, riêng
CSMCIR trỏ derived checkout, rồi chạy với `WORKBENCH_THIRD_PARTY_ROOT=<overlay>`. Đã thử và ghi lại
đầy đủ, vì nó chạm đúng một mâu thuẫn cần quyết định.

Đã dựng `/tmp/tp_derived`: symlink mọi model sang vendor thật, `CSMCIR -> /tmp/derive_fresh`. Để
derived trông như một checkout thật, phải chép `provenance.json`, `README.md`, `requirements.txt`, và
`COT_ours2` phải nằm **trong** cây (guard `_guard_auxiliary_destination` từ chối đường dẫn thoát khỏi
root — guard đúng, không phải lỗi).

Kết quả là hai blocker, cái thứ hai là mâu thuẫn thật:

```
[BLOCKED] source cannot run: … 2 of 2 required patches are not applied
[BLOCKED] src/lavis/models/__init__.py: modified source in an approved-patch checkout
          (matches neither the published base nor the recorded post-patch digest)
```

Đo trực tiếp, cùng một file:

| trạng thái `__init__.py` | sha256 | khớp registry? |
|---|---|---|
| chỉ áp `csmcir_models_init_compat.patch` | `aa68fe5e20bcda5e…` | **đúng `derived_sha256`** |
| thêm dòng import `shiyan92` | `0ec134431697af6f…` | không khớp gì |

Đây là mâu thuẫn **không thể vá bằng cấu hình**:

- `approved_patch_state` chỉ chấp nhận cây có digest **đúng** `aa68fe5e…`.
- Ở cây đó, `shiyan92` **không** được đăng ký: khối 98 import nằm trong `try/except ImportError`, và
  module thiếu đầu tiên (dòng 49) làm hỏng cả khối trước khi tới dòng import `shiyan92`.
- Không đăng ký ⇒ `load_model_and_preprocess` trả `None` ⇒ `AttributeError: 'NoneType' object has no
  attribute 'from_pretrained'`.

Nên để đường chính thức chạy được, registry **phải** khai thêm một delta cho dòng import đó. Đó là
D-10 phương án (a). **Lượt này đã làm đúng phương án đó** (3.23): nó đổi `derived_sha256`, tức đổi thứ guard dùng để nói "cây này là
bản đã được duyệt", và phải tái ký `patch_sha256` — quyết định có ảnh hưởng tới tính toàn vẹn, không
phải sửa cục bộ.

Ghi chú thêm: advisory đề xuất thêm một biến `WORKBENCH_CSMCIR_SOURCE_ROOT`. Cách đó **không** giải
được, vì `preflight`/`runtime_blockers` vẫn tính source từ `WORKBENCH_THIRD_PARTY_ROOT`, nên nó vẫn
thấy pristine và vẫn chặn — trừ khi sửa cả hai chỗ, tức lớn hơn phương án (a) chứ không nhỏ hơn.

### 3.23. D-10 ĐÓNG — native evaluation chạy qua pipeline chuẩn

**Mục này là hướng dẫn hiện hành duy nhất cho CSMCIR.** Recipe chạy được ở 3.23.5b; digest đã đo lại ở
3.23.5c; hai run exit 0 ở 3.23.5. Các lệnh trong 3.18.6b được giữ với nhãn **HISTORICAL** chỉ để truy
vết, không phải bước cần làm.

Trước mục này, CSMCIR chỉ chạy được qua script trong evidence bundle, không qua `evaluate_models.py`.
Vấn đề không phải cấu hình: `approved_patch_state` đòi digest **đúng** `derived_sha256`, mà cây cho
`shiyan92` đăng ký lại có digest khác. Đã giải bằng cách gộp, không bằng cách nới guard.

#### 3.23.1 Gộp vào patch hiện có, tái ký digest

Hai patch CSMCIR giờ là **base → trạng thái cuối cần dùng**, không phải base → trung gian:

| file | base | derived | patch_sha256 |
|---|---|---|---|
| `src/lavis/models/__init__.py` | `9d8c914b…` | `0ec13443…` | `3f2bac48…` |
| `src/validate_blip_csmcir.py` | `aa7a4395…` | `a057e8f3…` | `0fcff28b…` |

`__init__.py` gộp cả hai việc trong một patch: bọc khối 98 import trong `try/except`, **và** đăng ký
`shiyan92` sau khối đó (không phải bên trong, vì module thiếu đầu tiên làm hỏng cả khối).

`validate_blip_csmcir.py` gộp ba việc: làm `vis_fiq` thành opt-in, và hai tham số vận hành thành
env-overridable với **default tác giả giữ nguyên** (`WORKBENCH_CSMCIR_QUERY_BS` mặc định 64,
`WORKBENCH_CSMCIR_SAVE_MEMORY` mặc định 0).

Không có whitelist nào được nới; digest vẫn phải khớp chính xác.

#### 3.23.2 Cơ chế áp patch — trước đó không tồn tại

`prepare_model.py` chỉ in trạng thái, không áp gì. Đã thêm `_apply_approved_source_patches`, giữ guard chặt:

- patch file phải khớp `patch_sha256` của nó;
- target phải đang là `base_sha256` **trước** khi áp (nếu lệch → từ chối, vì "unexpected modification
  is not covered by the declaration");
- **sau** khi áp, target phải đúng `derived_sha256` — nếu không, dừng;
- target đã ở derived → no-op.

#### 3.23.3 `.pyc` là asset có khai báo, không phải file chép tay

Khai trong `auxiliary_assets.yaml`: `direct_url` trỏ đúng blob tại revision pin, `expected_sha256`
`46d00b0b…`, stage vào tên module (CPython chỉ nhận `.pyc` cạnh package, không nhận trong
`__pycache__`). Downloader tải và verify; không có file nào được tạo tại chỗ.

Ghi chú kỹ thuật: asset này **không** được khai `content_contract`, vì `registry.py:1551` coi mọi asset
có contract là JSON và sẽ báo `INVALID JSON` cho một `.pyc`.

#### 3.23.4 Hai lỗi vận hành chặn đường pipeline — đã sửa

| lỗi | bằng chứng | sửa |
|---|---|---|
| `OSError: AF_UNIX path too long` | `TMPDIR` do pipeline đặt là `artifacts/logs/<run-id>/tmp`, ~142 ký tự; `multiprocessing.resource_sharer` bind `<TMPDIR>/listener-<10 ký tự>` và kernel chặn ở 107 byte. Chết trước batch đầu tiên | temp dir của run giờ tạo bằng `mkdtemp(prefix="wb-", dir=/tmp)` — vẫn riêng cho từng run, vẫn 0700, nhưng đường dẫn ngắn |
| `PYTORCH_CUDA_ALLOC_CONF` bị chặn | allowlist của evaluator không có biến này, nên biến operator đặt **không tới** evaluator, trong khi chính message OOM khuyên dùng nó. Chạy tay thì được, chạy pipeline thì không | thêm vào allowlist (là chiến lược cấp phát, không đổi phép tính) |

Thêm nữa, các biến vận hành riêng của model nay khai qua `environment.runtime_env` trong registry và
được `_declared_model_env_names` đưa vào allowlist — không hardcode tên model vào danh sách chung.

Evidence cũng ghi lại knob thật đã tới evaluator (`command.json` →
`environment.evaluator_environment_knobs`), vì trước đó `command.json` chỉ có digest và một người đọc
sau không biết batch/save_memory có được đặt hay không.

#### 3.23.5 Kết quả — hai lần, khớp bit, mỗi lần truy vết được

Hai run exit 0, mỗi run có log dir + `command.json` + `invocation_outcome.json` + `aggregate_report.json`
riêng, chép nguyên vào `csmcir/runs/<suffix>/`. Bảng đầy đủ: `csmcir/runs/README.md`.

| | run A | run B |
|---|---|---|
| run_id | `20261107T054343058424Z_csmcir_fashioniq_f564570e` | `20261107T054933277898Z_csmcir_fashioniq_f03887d2` |
| log dir | `artifacts/logs/csmcir__fashioniq__20261107T054343058424Z_csmcir_fashioniq_f564570e/` | `artifacts/logs/csmcir__fashioniq__20261107T054933277898Z_csmcir_fashioniq_f03887d2/` |
| `return_code` | 0 | 0 |
| `verdict` | `PROOF_VALIDATED` | `PROOF_VALIDATED` |
| R@10 / R@50 | 57.05073873202006 / 77.29859352111816 | 57.05073873202006 / 77.29859352111816 |
| `evaluator_environment_knobs` | **`null`** | đủ bốn knob |
| provenance đầy đủ | **KHÔNG** | **CÓ** |

Số từng category, giống nhau tuyệt đối ở cả hai run:

| category | R@10 | R@50 |
|---|---|---|
| dress | 52.20624804496765 | 74.86366033554077 |
| shirt | 57.75269865989685 | 76.1040210723877 |
| toptee | 61.19326949119568 | 80.92809915542603 |
| **macro** | **57.05073873202006** | **77.29859352111816** |

Run A chạy trước khi `command.json` ghi knob thật đã tới evaluator. Vì vậy **chỉ run B là run đủ
provenance**, và đó là run được dùng cho `csmcir/full_eval_pipeline.txt` + `csmcir/pipeline_command.json`.
Run A là đối chứng độc lập: cùng `command_digest` (`7aa7a6db…`), cùng pin `774f94e2…`, cùng
`checkpoint_local_sha256` `c8812071…`, nên nó chứng minh số đo lặp lại được, không chứng minh được
knob đã đặt.

#### 3.23.5b Recipe hiện hành — kiểm lại được, phần không cần GPU đã chạy

Đây là **một lệnh duy nhất**. Không có helper nào phải chạy tay: `prepare_model` áp patch đã khai,
`download_auxiliary_assets` stage `.pyc`.

```bash
export CIR=/home/heheboiz/data/cir
export EVID=$CIR/workbench/artifacts/checkpoint_selection/20261106T002617+0700
export E2=$CIR/workbench/docs/reviews/checkpoint_selection/20261107T021500+0700
cd $CIR
export WORKBENCH_RUNTIME_ADAPTATION=sm_120
export WORKBENCH_CSMCIR_PYTHON=$CIR/workbench/artifacts/environments/cir-005ef4e1b02999a5/bin/python
export WORKBENCH_CSMCIR_QUERY_BS=8        # tác giả: 64
export WORKBENCH_CSMCIR_SAVE_MEMORY=1     # tác giả: 0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 1) source: áp approved patch + stage link fashionIQ_dataset (idempotent, no-op nếu đã derived)
workbench/envs/ctl312h/bin/python workbench/scripts/prepare_model.py \
  --model csmcir --execute --allow-preparation

# 2) asset: caption công khai + .pyc tác giả (verify sha256 tại chỗ, không tải lại nếu đã có)
workbench/envs/ctl312h/bin/python workbench/scripts/download_auxiliary_assets.py --model csmcir

# 3) chạy thật
workbench/envs/ctl312h/bin/python workbench/scripts/evaluate_models.py --model csmcir
```

Điều kiện tiên quyết đã kiểm trên máy này (`csmcir/recipe_check.txt`):

| mục | trạng thái |
|---|---|
| source `workbench/third_party/CSMCIR/src` | đã ở `derived_sha256` của cả hai patch; `prepare_model --execute` → cả hai `already carries derived digest`, rc=0 |
| link `fashionIQ_dataset` | `-> data/FashionIQ`, `prepare_model` báo `already correct` |
| caption 3 category (2 bản mỗi cái) | `OFFICIAL SHA256 MATCH` ×6 |
| `.pyc` tác giả | `12281 bytes`, `OFFICIAL SHA256 MATCH` |
| interpreter evaluator | `Python 3.10.22`, `cache_tag = cpython-310` → khớp `.pyc` khai CPython 3.10 |
| orchestrator | `workbench/envs/ctl312h/bin/python` = 3.12.11, chỉ chạy script workbench, không nạp `.pyc` |
| runtime sm_120 | capability đã verify: RTX 5070 Ti, `compute_capability 12.0`, `torch 2.7.1+cu128`, arch list có `sm_120` |
| booster LAVIS/EVA-ViT-g | có trên đĩa nhưng **`UNVERIFIED` provenance** (không có checksum upstream để đối chiếu) — ghi nhận, không phải claim đã verify |
| `evaluate_models.py --dry-run` với đủ 5 biến | guard pass: protocol `fashioniq_original_split`, source derived, GPU capability hợp lệ |
| builder (đường B) chạy lại | `0ec134431697af6f OK`, `a057e8f3ad7097ad OK`, `.pyc sha256=46d00b0b…`, rc=0 — vẫn tái tạo đúng derived |
| ba helper vận hành cũ | `exit 1` trên mọi cây đã áp patch hiện hành (`found 0`) — HISTORICAL, xem 3.18.6b |
| adapter CIRCUS | `ready=False`, reason nêu đúng ràng buộc; `data/circus_release/` vắng → `FileNotFoundError: .../circus_release/mbeir_fashioniq_task7_test.jsonl` |

#### 3.23.5c Digest — giá trị đo lại (không suy từ registry)

Đo trên máy này; lệnh và output đầy đủ ở `csmcir/digest_verification.txt`.

| hạng mục | giá trị khai | đo lại | khớp |
|---|---|---|---|
| base `src/lavis/models/__init__.py` | `9d8c914b551a7de614e062f9a241f44626fd591a31a0ccf1cc79f0ae23c9db01` | `9d8c914b551a7de614e062f9a241f44626fd591a31a0ccf1cc79f0ae23c9db01` (reverse-apply patch, `patch -R -p1` rc=0) | **CÓ** |
| base `src/validate_blip_csmcir.py` | `aa7a4395cdb700fded0567e0969da774a331927eaea327b5a23da82fdc7506a9` | `aa7a4395cdb700fded0567e0969da774a331927eaea327b5a23da82fdc7506a9` (như trên) | **CÓ** |
| derived `src/lavis/models/__init__.py` | `0ec134431697af6f202b5256e4e30ab725a44191a279e3f8bfdb3a62495b3774` | `0ec134431697af6f202b5256e4e30ab725a44191a279e3f8bfdb3a62495b3774` | **CÓ** |
| derived `src/validate_blip_csmcir.py` | `a057e8f3ad7097adcfabb2b8fd8b4747588ff3461cfb43f00cd3483dfd6d512d` | `a057e8f3ad7097adcfabb2b8fd8b4747588ff3461cfb43f00cd3483dfd6d512d` | **CÓ** |
| patch `csmcir_models_init_compat.patch` | `3f2bac48…` | `3f2bac48fe1a1775d2e5e6a09f81affa3809a31ed6b7c4213d5da8c2a0ada3bd` | **CÓ** |
| patch `csmcir_validate_visualization_opt_in.patch` | `0fcff28b…` | `0fcff28bf1f9abd31a14a06b0f200eecfa46af8c2d7624a3420a68204ef50fa8` | **CÓ** |
| `.pyc` `shiyan92` | `46d00b0b8521291fef5b032afbaf20846fcfaf800ed9b4522c08b2ab991a15a8`, 12281 B | `46d00b0b8521291fef5b032afbaf20846fcfaf800ed9b4522c08b2ab991a15a8`, `size 12281` | **CÓ** |
| checkpoint `fashioniq_tuned_clip_best.pt` | `c8812071de0640d647c5ba2a70628b80a3289ee2c47676f7a1b6abaa313fa49e`, 2720615859 B | `c8812071de0640d647c5ba2a70628b80a3289ee2c47676f7a1b6abaa313fa49e`, `size 2720615859` | **CÓ** |

Base không đọc từ đâu khác: nó được dựng lại bằng cách reverse-apply **chính patch đã khai** lên
derived đang dùng, rồi so với `base_sha256` trong registry. Cả hai chiều đều khớp, nên declaration
không tự chứng minh vòng tròn.

Một điểm không khớp, và đây là điểm **đã biết**: `workbench/artifacts/csmcir_derived/src/validate_blip_csmcir.py`
có sha256 `9af8d120…`, **khác** derived `a057e8f3…`. Đó là checkout đường B (builder + ba helper cũ),
không phải cây đường chuẩn; `measurement.json` đã ghi rõ chuỗi digest riêng của nó. Đường chuẩn dùng
`workbench/third_party/CSMCIR/src`, và cây đó đúng `a057e8f3…`.

#### 3.23.5d Bốn thứ không đổi

Recipe trên không chạm vào bốn điểm đã tuyên bố, và chúng vẫn giữ nguyên:

| thuộc tính | giá trị | kiểm ở đâu |
|---|---|---|
| scoring | native `inference()` của evaluator tác giả, không qua adapter | `validate_blip_csmcir.py`, `runs/f03887d2/command.json` |
| protocol | `fashioniq_original_split` (full gallery, reference **eligible**) | `command.json` → `protocol_id`; dry-run in `[OK] protocol: fashioniq_original_split` |
| nhãn bằng chứng | `public-caption path-mapped research run` | 3.18.6, `measurement.json` → `evidence_class` |
| CIRCUS | luồng riêng, `ready=False`, `data/circus_release/` vắng — lý do giữ nguyên, không "sửa" bằng cách thay input | 3.23.6, `model_csmcir.py` `_TARGET_REASON` |

#### 3.23.6 Luồng B (composition adapter) vẫn tách — và đúng là phải tách

Adapter CIRCUS **không** được nối vào số đo này, vì hợp đồng scoring khác:

| | native (đường đo) | adapter |
|---|---|---|
| cách tính | `inference()`: attention-weighted token importance, matmul `(1,1,256) × (queries,32,gallery)`, max trên 32 query token | `score()`: `reshape(N,32,256)[:,0,:]`, chuẩn hoá, `einsum().max(-1)` trên vector 8192 chiều |
| đại lượng | điểm tương đồng per-query | dot product trên embedding mở rộng |

Ép hai cái bằng nhau là đổi thuật toán. `_status()` giữ `ready=False` nhưng **lý do đã sửa cho đúng**:
trước nói thiếu caption `COT_ours2` (đã bị bác bỏ ở 3.18), nay nói adapter không được nối vào đường đã đo
và nêu đúng hai ràng buộc thật.

Blocker riêng của CIRCUS, không liên quan CSMCIR: `data/circus_release/` không tồn tại. Kiểm bằng
`composition_eval.py --models limn --plan` → cùng `FileNotFoundError`. Nên đây là hạ tầng chung, và bổ
sung dataset cũng **không** giải được khác biệt scoring ở bảng trên.

### 3.24. Suite sạch và ba test đỏ cuối — trạng thái cuối

| lần chạy | failed | passed | skipped |
|---|---|---|---|
| baseline `ea5ff05` | **23** | 2152 | 47 |
| sau lượt trước | 3 | 2172 | 47 |
| **cuối lượt này** | **0** | **2179** | **47** |

`workbench/envs/ctl312h/bin/python -m pytest workbench/tests`, exit 0, 2226 collected.

#### 3.24.1 Ba test đỏ — nguyên nhân thật của từng cái

**(a) `test_planner_verifier_uses_the_resolved_contract`** — CONFIRMED, tiền đề test sai. Registry pin
`open_clip_torch==2.20.0` (tác giả, `LIMN/requirements.txt:37`), test lại hardcode range
`>=2.24,<2.27`. Chú ý: **không phải `torch==2.20.0`** — bản torch đó không tồn tại. Invariant nay là
**agreement**: verifier, installer và declaration phải là cùng một specifier set; pin bị bỏ vẫn fail
(kiểm bằng negative control: đổi registry thành `open_clip_torch` trần → test đỏ).

**(b) `test_every_manifest_path_matches_its_recorded_digest`** — CONFIRMED, tiền đề test sai. Ba file
DCNet/TME đã áp patch **in-place** nên bytes là derived, còn `provenance.json` ghi base. Đo bằng hash:

| file | committed | registry base | registry derived | manifest |
|---|---|---|---|---|
| `DCNet/base/base_trainer.py` | `f265cefe…` | `3edd9836…` | `f265cefe…` | `3edd9836…` |
| `TME/src/precompute.py` | `97bdbc14…` | `c3da9c9d…` | `97bdbc14…` | `c3da9c9d…` |
| `TME/src/precompute_evaluation.py` | `1f13dbd5…` | `0f6a0573…` | `1f13dbd5…` | `0f6a0573…` |

Reverse-apply từng patch tái tạo đúng `base_sha256`. Kết luận: **derived là thứ ship**, vì DCNet cần
`weights_only` opt-out và TME cần phân bổ có giới hạn để không OOM. Test nay chấp nhận base **hoặc**
derived cho đường đã khai báo, và **drift vẫn fail** (có test riêng).

**(c) `test_a_real_distro_environment_is_not_reported_empty`** — CONFIRMED, **bug implementation thật**
trong `runtime.py`, không phải test hỏng. Hai lỗi:

1. `.egg-info` legacy là **file** chứ không phải thư mục, nên `dist/"RECORD"` không tồn tại →
   rơi vào nhánh `versions.pop(key)` → **một package đã cài biến mất khỏi inventory**.
2. `top_level.txt` trỏ tới extension có tag ABI (`Brlapi` → `brlapi.cpython-312-*.so`); phép kiểm cũ
   `(sp/f"{head}.py").is_file() or (sp/head).is_dir()` không khớp `.so`.

Đã sửa (1) trong `runtime.py`; test chuyển sang kiểm **exact set** trên mọi layout hợp đồng hỗ trợ,
cộng một integration check thật trên host (56/88 decidable, 0 bị bỏ). Không hạ threshold, không skip.

#### 3.24.2 Bài học về fixture "pristine"

Áp patch in-place làm **33 test** đỏ, tất cả cùng một lý do: fixture `copytree(vendor)` rồi gọi đó là
"pristine". Sửa đúng không phải nới assertion, mà là **dựng lại base bằng cách đảo patch**:

```python
# trong fixture
if sha256(target) == declaration["derived_sha256"]:
    patch --reverse  # base được phục hồi chính xác, vì patch CHÍNH LÀ hợp đồng
assert sha256(target) == declaration["base_sha256"]
```

Nhờ vậy "pristine" là phát biểu về **revision đã pin**, không phải về việc host đã chạy preparation hay
chưa. Áp dụng cho `test_vendored_approved_patch`, `test_approved_source_patch`,
`test_component_boundaries`, `test_r9_defect_regressions`, `test_pc_readiness`, `test_prepare_dataset`.

#### 3.24.3 Lỗi find-replace toàn cục — đã sửa

Khi nâng `TESTED_SOURCE_SHA`, một lần thay thế toàn cục đã đè cả các dòng **historical** (A34/A35
`1415e56`, A33 `e3f0dfa`, sm_120 `4e0c074`) và cả dòng `origin/main` (thực tế `8ebcc74d`, không đổi).
Đã revert file và chỉ nâng đúng các dòng **current**. Kiểm lại: `8ebcc74d` còn 2 hit, bốn SHA lịch sử
nguyên vẹn.

### 3.25. Hai bug inventory thật, rác temp dir, và thư mục run bị bịa

#### 3.25.1 Inventory distro: 85/85 package, không phải 85+3

Hai lỗi trong `runtime.inspect_environment_distributions`, cả hai đều làm một distribution **hiện hai lần
hoặc được tính là đã cài khi không có gì**:

| lỗi | cơ chế | đo được |
|---|---|---|
| `.egg-info` legacy là **file** | `dist / "PKG-INFO"` ghép vào một file → path không tồn tại → `_metadata_name_version` trả `(None, None)` → tên lấy bằng `dist_name.split("-")[0]`, nên `my-pkg-1.0` thành `my`; distribution sau đó được ghi **hai lần**, một lần tên thật một lần tên cắt | `versions` chứa cả `my` lẫn `my-pkg` |
| `glob` khớp cả **thư mục** | `site_packages.glob(f"{head}.*.so")` nhận mọi path khớp mẫu, kể cả thư mục rỗng đặt tên `brlapi.cpython-312-x86_64-linux-gnu.so` | thư mục giả khiến `brlapi` thành module đã cài |

Sau sửa, đo trên host thật (`inspect_environment_distributions("/usr")`):

| | trước | sau |
|---|---|---|
| `len(versions)` | 85 | 85 |
| `len(unconfirmed)` | 3 | 3 |
| **package duy nhất** `len(set(versions) \| set(unconfirmed))` | **85** | **85** |
| modules | 201 | 201 |

`unconfirmed` là **tập con** của `versions` (cả 3 — `command-not-found`, `pycairo`, `pygobject` — đều đã có
version), nên số package duy nhất là 85, **không phải** 85+3=88. Cách đếm `len(versions)+len(unconfirmed)`
là sai và từng được ghi nhầm ở đây.

`/usr/lib/python3/dist-packages` có 88 metadata entry nhưng chỉ **87 tên chuẩn hóa**:
`cryptography-41.0.7.dist-info` và `cryptography.egg-info` là cùng một distribution. 85 tên được giải
trình; 2 tên còn lại (`screen-resolution-extra`, `unattended-upgrades`) có `top_level.txt` **chỉ chứa một
dòng trắng** — chúng khai không ship module nào, nên việc inventory bỏ qua là đúng hợp đồng hiện có.

#### 3.25.2 Temp dir rò rỉ

`wb-*` tạo mỗi lần chạy nhưng cleanup chỉ gắn vào `try` **thứ hai**, tức đoạn sau khi `Popen` thành công.
Mọi lỗi trước đó — `replay_cache.mkdir`, `chmod`, hay `Popen` ném `OSError` vì interpreter không tồn tại —
thoát **trước** `finally` và để lại một thư mục. Đo được 314 thư mục tích trong `/tmp`.

Nay toàn bộ phần sau `mkdtemp` nằm trong một `try` ngoài, cleanup trong `finally` tương ứng. `keep_temp=True`
giữ thư mục trên **mọi** đường, kể cả đường launch thất bại. Child được kill và reap **trước** khi xoá bất
cứ gì. Lỗi gốc giữ nguyên kiểu: `OSError` từ `Popen` vẫn thành `RuntimeError(...) from error`. Mode 0700
được **kiểm** chứ không giả định.

#### 3.25.3 Thư mục run được in ra là dự đoán, không phải run thật

`main()` gọi `log_dir(config, plan, ...)` **không** truyền `run_id`, nên `log_dir` tự sinh một id bằng
`make_run_id()` (kết thúc bằng `uuid.uuid4().hex[:8]`), in ra, rồi `execute()` sinh id **thứ hai**. Thư
mục được quảng cáo chưa từng tồn tại.

Đây chính là gốc của evidence ghép sai: bundle ghi run `..._20261107T054914376731Z_csmcir_fashioniq_46f7bc56`,
không có trên đĩa; run thật là `f03887d2`. Nay id được sinh **một lần** trước khi in và dùng lại cho
`execute()`. Định dạng id không đổi.

#### 3.25.4 Builder evidence đồng bộ lại

`build_csmcir_derived.py` còn pin digest **trước khi gộp patch** (`aa68fe5e`/`84a1d476`), nên chạy lại sẽ
fail. Đã nâng về `0ec13443`/`a057e8f3`, và thêm bước dựng base bằng cách đảo patch (vendor giờ đã derived).
Docstring nói rõ builder **không còn** là đường chuẩn: `prepare_model` + downloader phủ việc đó.

## 4. Hàng đợi quyết định buổi sáng

| ID | model/job | hành động đề xuất | vì sao | phạm vi/rủi ro | đã thử an toàn | bằng chứng | cần bạn quyết/ cung cấp |
|---|---|---|---|---|---|---|---|
| **D-1** | `sprc` (biến thể chính, EVA-CLIP ViT-g) | Bạn tải file OneDrive rồi đặt tại `/home/heheboiz/data/cir/workbench/artifacts/checkpoints/sprc/sprc_fiq.pt` | Bản chính (paper config) chỉ có trên OneDrive; tôi không thể login/tương tác | Chỉ thêm 1 file 1.4–2.6 GB vào checkpoint root; không đổi code | Đã chạy biến thể ViT-L công khai (D6 0.0268 pp) — **không** dùng thay hàng ViT-g | `measurements/sprc_vitl_run1.json`, `measurements/sprc_distance.json` | Tải https://1drv.ms/u/s!Aj0q22vyiZbabnUya4mnufIBtYI?e=n4ZVKj (Baidu dự phòng, pass `t1at`) và đặt đúng path trên |
| **D-2** | `pair` (B1 + B2) | Quyết định hướng: (a) xin tác giả `model_vssp.py`/`MHTransformer_dis.py` đúng revision, (b) bỏ PAIR khỏi cohort và giữ nhãn EXPLORATORY trong báo cáo, hoặc (c) xác nhận rằng checkpoint đến từ ancestor chung với ENCODER để tôi dựng đúng nhánh còn thiếu | Đã load và chấm được cả 2 file bằng EXPLORATORY_PUBLIC_MAPPING, nhưng public `model.py` KHÔNG phải code đã train: `transformer_a` khôi phục 2 layer×4 head vs public 1×8; checkpoint có `transformer_c/d`, `weight_chanelImg*`, `cross_t2i` mà public không định nghĩa; 5.20% tham số nằm ngoài forward ⇒ điểm 35.01/57.58 (B1) không phải hành vi checkpoint | (a)/(c) cần bạn liên hệ hoặc xác nhận; (b) chỉ là quyết định phạm vi. Không sửa forward, không khởi tạo lại module | Đã probe HF + ZhihFu/PAIR (chỉ `main`), 5 repo công khai của tác giả, git history; đã restore + audit + smoke + chạy đủ 3 category × 2 file × 2 hypothesis | `pair_compat/manifests/pair_compat_verdict.json`, `pair_compat/measurements/summary.json`, mục 3.9 | Chọn (a), (b) hay (c) |
| **D-3** | registry dùng chung | Duyệt diff registry để map các entry đã đo được: 16 entry (trừ `pair` vì chỉ exploratory; diff `csmcir` riêng ở 6.1); nguồn cũ ghi 8 priority + limn/encoder/hint/habit/intent/conesep/tme, trừ PAIR vì chỉ exploratory) | Hiện registry vẫn ghi `BLOCKED_*`/`NOT_RUN` cho các bundle nay đã đo được; không nên tự sửa default toàn cục qua đêm | Sửa `workbench/registry/models.yaml` (`reproduction_status` + `checkpoint_mapping_status`) + `preparation_contracts.yaml` cho DCNet GloVe và TME `feature_dtype` | Không sửa gì; chỉ ghi `configs/selected_overrides.yaml` trong session | `configs/selected_overrides.yaml`, mục 9.1, `cohort/measurements/cohort_summary.json` | Duyệt/không duyệt diff registry |
| **D-4** | DCNet reproducibility | Cho phép ghi lại recipe sinh GloVe PKL thành generator trong `preparation_contracts.yaml` (seed cố định, sha256 của 10 file PKL) | Hiện contract ghi `NONDETERMINISTIC_WITHOUT_AUTHOR_ARTIFACT`, nên lần sau không tái lập được bằng workbench | Sửa 1 khối contract + thêm 10 entry hash; không đổi dataset canonical | Đã sinh trong session, giữ file + log, không sửa contract | `logs/dcnet_glove_prep.log`, `logs/dcnet_run2.log` | Duyệt việc chuẩn hoá recipe này |
| ~~D-5~~ | `torch.load` shim cho DCNet | **ĐÃ ÁP DỤNG** trong buổi trước: `approved_source_patches` 1 dòng `weights_only=False` trong `base/base_trainer.py`, base/derived sha256 trong registry (mục 3.14) | Hiện phải inject shim để unpickle; nếu duyệt thì DCNet chạy được bằng CLI chuẩn không cần shim | Sửa 1 dòng trong checkout pinned + khai báo patch như CSMCIR | Đã chạy bằng shim, kết quả đầy đủ | `scripts/compat/sitecustomize.py`, `logs/dcnet_run2.log` | Không cần quyết nữa |

Không mục nào chặn job khác: chỉ **D-1** (file SPRC ViT-g) còn thực sự treo. **D-8 đã đóng** trong buổi 20261107 — xem 3.18: yêu cầu `bert_captions`/`keywords_in_mods` là của một class chết, và caption cần thiết đã tải được từ Hub tác giả. D-5, D-6, D-7, D-9 đã áp dụng/giải quyết; 16/17 entry đã có số đo native độc lập. Còn chờ quyết định chính sách: D-2 (PAIR), D-3 (diff registry), D-4 (GloVe recipe).

Bổ sung D-6…D-9 cho cohort inventory:

| ID | model/job | hành động đề xuất | vì sao | phạm vi/rủi ro | đã thử an toàn | bằng chứng | cần bạn quyết/cung cấp |
|---|---|---|---|---|---|---|---|
| ~~D-6~~ | `limn` runtime | **ĐÃ ÁP DỤNG trong buổi này**: `adapted_runtime.packages` của LIMN nay là `['open_clip_torch==2.20.0']` trong `workbench/registry/models.yaml`, wrapper tự đọc profile đó nên không cần script riêng của phiên | Hai căn cứ **đo được**: (i) `LIMN/requirements.txt:37` khai đúng `open-clip-torch==2.20.0`; (ii) `workbench/envs/limn-adapted` đang chạy 2.20.0 và là môi trường sinh ra kết quả 55.7919/77.6378. Lý do cũ (`AttributeError: ... 'pool_type'` dưới 2.24) **đã bị rút lại** vì không tái lập được | Chỉ đổi 1 dòng của riêng LIMN; không đụng profile nào khác | Preflight qua wrapper: `[READY] LIMN preflight passed`; 3 checkpoint giữ nguyên một bundle | `limn/scripts/limn_version_repro.py`, mục 3.15 | Không cần quyết nữa |
| ~~D-7~~ | `tme` feature contract | **ĐÃ ÁP DỤNG trong buổi này, theo hướng hẹp hơn đề xuất cũ**: không sửa số của contract lịch sử, mà thêm `runtime_overrides.sm_120.feature_dtype: float32` + resolver `generated_artifact_parameters()`; lower bound tự thành 20.94 GiB khi chọn profile, vẫn 10.47 GiB khi không chọn | Đo thật: `visual_encoder` đúng là fp16, nhưng `vit_encode()` dưới `torch.cuda.amp.autocast()` trả **float32**, nên `torch.save` ra float32. Wrapper `tme.py:414` từ chối cache mà chính pipeline native sinh ra | Sửa 2 giá trị contract; không đổi numerics (dtype chỉ dùng cho gate + size floor) | Đã sinh cache đúng như native, chạy xong 3 category bằng session runner nhận dtype đo được | `cohort/measurements/tme_blocker.json`, `cohort/measurements/tme_fashioniq_0.json`, `cohort/logs/tme_run5.log` | Không cần quyết nữa — đường chuẩn đã chạy được. Nếu sau này có runtime khác cho dtype khác, khai thêm một khối `runtime_overrides` tương ứng |
| ~~D-8~~ | `csmcir` | **ĐÃ GIẢI QUYẾT trong buổi 20261107** — không cần bạn cung cấp gì | Yêu cầu `bert_captions`/`keywords_in_mods` là của `FashionIQDataset_imageedit`, một **class chết**; đo bằng `builtins.open` cho thấy đường eval thật chỉ mở 12 file và **không** mở hai đường đó. Caption thật sự cần đã tải được từ Hub tác giả, sha256 khớp | Không đổi scoring; ba file caption công khai, nhãn biến thể riêng | Macro 57.0507 / 77.2986, exit 0 — hai run `f564570e` / `f03887d2` (3.18, 3.23.5) | `csmcir/measurement.json`, `csmcir/call_path_probe.txt` | — |
| ~~D-9~~ | `ptha_mtst` | **ĐÃ GIẢI QUYẾT trong buổi này** — không còn cần bạn quyết để chạy được. Nguyên nhân là biên feature trong chính source pinned (định nghĩa `extract_target_features` thứ hai trả chưa chiếu), không phải checkpoint thiếu head; đã sửa bằng `inference_target()` của tác giả và chạy đủ 3 category. Nếu bạn **vẫn muốn** có reproduction FashionIQ thật thì đó là yêu cầu khác: (a) xin checkpoint FashionIQ-trained từ tác giả | Sửa xong: `load_state_dict` strict 0/0/0, `vision_proj (256,768)` dùng lại 196 864 tham số, 0 tham số mới, adapter ≡ `normalize(vision_proj(x))` với `max_abs_diff = 0.0`, regression **11/11 pass** (viết lại: gọi dispatcher production, đếm số lần `vision_proj` được áp dụng thật cho cả hai lớp) | Không cắt/pad chiều, không thêm Linear ngẫu nhiên, dispatch theo lớp đã load: CirCat chiếu một lần ở biên, TGIR **để nguyên** cho `inference()` tự chiếu. Chạy lại GPU: bit-identical với lần trước (37.897844 / 59.836390, delta 0.0 cả 3 category) | Đã chạy full 3 category trên GPU thật: macro **37.8978 / 59.8364**, coverage 100% | `ptha/measurements/ptha_transfer_run1.json`, mục 3.13 | Không cần quyết gì để dùng kết quả transfer; chỉ cần quyết nếu muốn theo (a) |


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

# --- TME: feature cache, then evaluation (D-7 closed; see 3.16.6) ---
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

## 5.2 Lệnh đã thực sự chạy trong buổi 20261107 (cwd, interpreter, exit)

Mọi lệnh dưới đây đã chạy thật trong buổi này. `CIR=/home/heheboiz/data/cir`.

| # | lệnh (rút gọn còn phần quyết định) | cwd | interpreter | exit | kết quả |
|---|---|---|---|---|---|
| 1 | `python -m pytest workbench/tests/test_ptha_target_feature_contract.py -q` | `$CIR` | `workbench/envs/test-adapted/bin/python` (3.11.15) | 0 | 11 passed |
| 2 | `python -m pytest <11 module PTHA+TME+bundle> -q` | `$CIR` | như trên | 0 | **110 passed** |
| 3 | `python -m workbench.replay.ptha_mtst --source-root workbench/third_party/MTST_PTHA --dataset-root data/FashionIQ --checkpoint-root .../ptha_mtst --checkpoint-id cirr_transfer --transfer-cirr-to-fashioniq` | `$CIR` | `workbench/envs/airknow-adapted/bin/python`, `WORKBENCH_RUNTIME_ADAPTATION=sm_120` | **0** | macro `37.897844 / 59.836390`, **bit-identical** run1 |
| 4 | `python -m workbench.replay.tme ... --preflight-only` (`env -u WORKBENCH_RUNTIME_ADAPTATION`) | `$CIR` | `workbench/envs/tme-adapted/bin/python` (3.10.22) | 1 (blocker thật) | lower bound **11 243 589 632 B**, `2 bytes` |
| 5 | như #4 nhưng `WORKBENCH_RUNTIME_ADAPTATION=sm_120` | `$CIR` | như trên | 1 (blocker thật) | lower bound **22 487 179 264 B**, `4 bytes` |
| 6 | `python tme_real_cache_smoke.py <ckpt> <out> 24 10` | `workbench/third_party/TME` | `tme-adapted`, `sm_120` | **0** | float32 `[24,257,1408]`, batch `[10,10,4]`, `decision_is_dtype_only=true` |
| 7 | `python -c "generated_artifact_parameters(...)"` mặc định / `sm_120` | `$CIR` | `test-adapted` | 0 | `float16` / `float32` |

Exit 1 ở #4/#5 **là đúng**: hai blocker có thật (path checkpoint chưa có trong `downloads/`, và cache 22.49 GB đã xoá). Blocker không phải lỗi code, và không bị che bằng cách nới preflight.

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
| tme | `fashioniq_0_best_model` | `fashioniq_original_split` | **COMPLETED** | 54.9229 / 74.8384 | EMPIRICAL (contract dtype, D-7 closed) |
| csmcir | `fashioniq_tuned_clip_best.pt` | `fashioniq_original_split` | **COMPLETED** | 57.0507 / 77.2986 | public-caption path-mapped run (3.18), run `f03887d2` + `f564570e`; checkpoint hash-verified |
| ptha_mtst | `tuned_mtst_cirr.pt` | `fashioniq_original_split` | **COMPLETED** | 37.8978 / 59.8364 | transfer CIRR→FIQ; nguyên nhân là biên feature, đã sửa (3.13) |

Tổng: **16/17 entry có số đo native**, `pair` đo được nhưng exploratory, và **việc còn chờ file người dùng** là biến thể SPRC ViT-g của entry `sprc` (D-1) — không phải entry mới. PTHA/MTST đã đóng blocker sai chiều projection; CSMCIR đã đóng ở mức bằng chứng riêng (3.18). PAIR có số đo nhưng mang nhãn `EXPLORATORY_PUBLIC_MAPPING`, **không** tính là tái lập.

Coverage: mọi run đều đủ query (2017 dress / 2038 shirt / 1961 toptee), gallery đúng protocol, reference loại trước Recall ở các protocol yêu cầu, không run nào bị rút gọn.

### 6.1 Diff registry đề xuất — CHƯA ÁP DỤNG

Bản đầy đủ kèm lý do từng field:
`workbench/docs/reviews/checkpoint_selection/20261107T021500+0700/registry_diff_csmcir.md`.

Tóm tắt cho entry `csmcir`: `reproduction_status` → `RESEARCH_RUN_PUBLIC_CAPTION_PATH_MAPPED` (không
dùng `reproduced`, vì equivalence với input lịch sử chưa chứng minh); thêm `measured_r10/r50/mean`,
`measurement_protocol`, `measurement_evidence`, `checkpoint_class`, `checkpoint_class_provenance`.

**Không** đề xuất: đổi `reported_*` (số paper giữ tách biệt); nâng `checkpoint_mapping_status`;
thêm patch thứ ba cho thay đổi vận hành (`save_memory`, batch size) — chúng là
`operational_parameters`, không phải sửa source.

`reproduction_status` **không** phải enum đóng (chỉ so sánh với literal `"VERIFIED"` tại
`registry.py:2011`), nên nhãn mới không làm vỡ parsing.

## 7. Những gì KHÔNG được chứng minh

- **Không entry nào** là `AUTHOR_VERIFIED_REPRODUCTION` của một con số paper. Nhãn `AUTHOR_VERIFIED` chỉ dùng khi có digest do tác giả công bố và khớp; SPRC ViT-L **không** đủ điều kiện đó và đã được hạ xuống `LOAD-COMPATIBLE`. AirKnow (SHA LFS) và SPRC ViT-L (khớp lớp tuyệt đối) chỉ đạt mức *association*; số đo vẫn là empirical.
- **PAIR**: load và chấm được, nhưng **tương đương source gốc chưa xác lập**. Đo được: `transformer_a` khôi phục 2 layer × 4 head ≠ public 1 × 8; checkpoint có `transformer_c/d`, `weight_chanelImg*`, `cross_t2i` mà public `model.py` không định nghĩa; 5.20% tham số không được forward public chạm tới. **Chưa** chứng minh được đó là nguyên nhân duy nhất của điểm thấp, và **không** thể gọi các module thừa là "training-only" vì chúng không xuất hiện ở bất kỳ đâu trong checkout, kể cả loss/train. Xem 3.9.
- **TME**: **đã chạy qua đúng `workbench.replay.tme`** (3.16.6, exit 0, 54.9229 / 74.8384); trước đó từng chạy dưới session runner nhận dtype đo được (float32) và giảm `Qformer_bs`/`query_bs`; cho tới khi D-7 được duyệt, đây **không** phải đường chạy `workbench.replay.tme` nguyên bản.
- **LIMN**: chạy dưới open_clip **2.20.0**, đúng pin tác giả (`LIMN/requirements.txt:37`). `adapted_runtime` trước đây khai `>=2.24,<2.27` và **đã được sửa** về `==2.20.0` (mục 3.15), nên nay contract và môi trường khớp nhau — không còn lệch.
- **CSMCIR**: **có** số đo **57.0507 / 77.2986** (3.18, hai run `f564570e` + `f03887d2`) — public-caption path-mapped research run, **không** phải reproduction chính thức: equivalence giữa caption công khai và input ra số paper là `NOT PROVEN`. Không suy ra từ model khác.
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
- (D-7) TME `feature_dtype`: **đã áp dụng** trong buổi 20261107 theo đúng hướng "ghi dtype *theo runtime*" — `runtime_overrides.sm_120.feature_dtype: float32` trong `preparation_contracts.yaml`, cộng resolver `generated_artifact_parameters()`. Không sửa số của contract lịch sử; xem 3.16.
- Ghi chú `.gitignore`: ngoài `workbench/artifacts/**`, dòng `*.log` (`.gitignore:116`) cũng chặn log — đây chính là lý do 9 log trong bundle "khai present nhưng vắng trên remote". Chúng nay được commit dưới tên `.txt`, giữ nguyên byte.

SHA-256 để điền lấy nguyên văn từ `cohort/manifests_acquired_final.json` và `manifests/selected_bundle_digests.json`.

Ghi chú `.gitignore`: `workbench/artifacts/**` bị chặn, nên evidence **không** vào git — chỉ commit báo cáo. Muốn đưa `pair_compat/` + `cohort/` lên git thì cần thêm negation hẹp, chưa làm.

## 10. Tóm tắt

### 10.1 Kết quả đo được (số native, không phải suy luận)

**16/17 entry đăng ký** có số đo native thật; `pair` đo được nhưng exploratory. PTHA chuyển từ blocked sang completed trong buổi trước; **CSMCIR chuyển sang completed** trong buổi 20261107 (3.18). Còn treo duy nhất: **biến thể SPRC ViT-g** chờ file (D-1, variant của `sprc` chứ không phải entry thứ 18).

### 10.2 Những gì thực sự đã thay đổi trong buổi này

| thay đổi | loại | bằng chứng |
|---|---|---|
| `workbench/replay/ptha_mtst.py` — `TARGET_FEATURE_CONTRACTS` + dispatch theo lớp đã load (buổi 20261107) | sửa code, phạm vi 1 model | `ptha/measurements/ptha_transfer_run2.json` (bit-identical run1), mục 3.13 |
| `workbench/tests/test_ptha_target_feature_contract.py` — viết lại 11 regression hành vi | test mới | 11/11 pass |
| `workbench/backend/registry.py` — `generated_artifact_parameters()` + validate `runtime_overrides` | sửa shared code, hẹp | mục 3.16 |
| `workbench/registry/preparation_contracts.yaml` — TME `runtime_overrides.sm_120` | 1 khối contract | mục 3.16.2 |
| `workbench/replay/tme.py`, `workbench/scripts/prepare_model.py` — mọi consumer đi qua resolver | wiring, 2 file | mục 3.16.1 |
| `workbench/tests/test_tme_feature_dtype_contract.py` — 14 regression | test mới | 14/14 pass |
| bundle — 9 log `.log` → `.txt`, + `ptha_transfer_run2`, + `tme_real_cache_smoke` | evidence | 54 entry / 55 file |
| `workbench/registry/models.yaml` — LIMN `adapted_runtime` `==2.20.0` | 1 dòng, 1 model | mục 3.15 |
| `workbench/registry/models.yaml` — DCNet `approved_source_patches` | khai báo patch | mục 3.14 |
| `workbench/patches/dcnet_checkpoint_weights_only_optout.patch` | patch 1 dòng, base→derived khớp | mục 3.14 |
| `workbench/third_party/DCNet/base/base_trainer.py` | áp patch trên (đã khai báo) | DCNet chạy `PYTHONPATH` sạch, kết quả khớp tuyệt đối |
| `CHECKPOINT_SELECTION_REPORT.md` | sửa sai, thêm 3.12–3.16 | chính file này |
| `workbench/docs/reviews/checkpoint_selection/20261106T233409+0700/` | bundle review được: **55 entry có hash** (+ chính file index = **56 file**; 456940 B = 446 KiB, đo lại toàn bộ sau khi thêm). Chín log trước đây **khai là present nhưng vắng trên remote** vì `.gitignore:116` bỏ qua `*.log`; nay đã commit dưới tên `.txt`, giữ nguyên byte, kèm provenance `original_path`. Số byte khai trước đó nhỏ hơn file thật (ước lượng), nay là số đo. | `EVIDENCE_INDEX.json`, `README.md`, `TEST_STATUS.md` |

### 10.5 Trạng thái test — nói thẳng

Repo dùng **cú pháp f-string PEP 701**, chỉ hợp lệ từ **Python 3.12** (`workbench/backend/environment_planner.py:669`). Máy này **không có** interpreter 3.12 nào kèm `pip`; mọi env conda đều 3.11.15. `git show HEAD:` của chính file đó **cũng** fail dưới 3.11 ⇒ đây là hạn chế **có từ trước**, không phải do buổi này.

Vì vậy: đúng ****Python 3.12 interpreter đã dựng trong buổi 20261107** — `workbench/envs/ctl312h` (CPython 3.12.11
do `uv` provision, có headers). Toàn bộ suite chạy được trên nó: **2155 passed, 47 skipped,
20 failed** (trước đó trên `ctl312` không headers: 2105 passed, 61 failed — 41 failure là do thiếu
`Python.h`, không phải code).

Phân loại 20 failure còn lại (không cái nào do CSMCIR):

| nhóm | số | nguyên nhân |
|---|---|---|
| verifier host-state | 7 | `release_verify.sh` chạy dưới `/usr/bin/python3` của distro (không có `pip`, không có `packaging`); 8 record GPU còn lại được ký bằng operator key của phiên trước. Verdict ra `RELEASE_BLOCKED` thay vì `ENGINEERING_INCOMPLETE` |
| frontend chưa build | 4 | `frontend/dist` chưa tồn tại; cần `npx vite build` |
| DCNet/LIMN vendor digest | 2 | manifest lệch do patch DCNet đã áp từ phiên trước |
| ln còn lại | 7 | `TESTED_SOURCE_SHA` trong handoff cũ, planner pandas pin, mutation m4, `test_pc_launcher` |

Trong buổi này **107 test** của các file bị đụng đều pass (`test_approved_source_patch`,
`test_vendored_approved_patch`, `test_component_boundaries`, `test_csmcir_source_blocker`,
`test_vendored_source_integrity`, `test_verified_declarations`, `test_release_gate_scenarios`,
`test_pc_readiness`).

Trước đó, PEP 701 trên 3.11:  và **12 test riêng lẻ** trong `test_runtime_adaptation.py` fail vì chúng import cùng module 3.12-only đó (bản thân module vẫn collect). Đã chạy và pass trong buổi 20261107 — **110 test** trong một lượt: `test_ptha_target_feature_contract` (11), `test_ptha_mtst_preflight`, `test_r19_ptha_wrapper`, `test_tme_feature_dtype_contract` (14), `test_tme_preparation`, `test_r21_tme_lifecycle`, và **60 test integrity** (`test_approved_source_patch`, `test_vendored_approved_patch`, `test_vendored_source_integrity`, `test_verified_declarations`, `test_dcnet_replay`). Trước đó 6 test của `test_runtime_adaptation` cũng pass — nhóm cuối là nhóm quan trọng nhất cho thay đổi DCNet, vì nó chứng minh khai báo `approved_source_patches` chặn đúng cổng kiểm tra source đã bị sửa. **Chưa** có CI xanh; việc đó cần môi trường 3.12. Chi tiết: `TEST_STATUS.md` trong bundle.

Hai assertion trong `test_runtime_adaptation.py` mã hoá contract LIMN **cũ** (`>=2.24,<2.27`) đã được cập nhật theo contract mới `==2.20.0` — đây là thay đổi hành vi có chủ đích (mục 3.15), không phải sửa test cho qua.

Không đổi: protocol nào, `models.yaml` default selection nào, shared evaluation logic nào, wrapper của model nào khác.

### 10.3 Bốn khẳng định bị hạ cấp trong buổi này

1. **PTHA "architectural mismatch"** → sai. Checkpoint có `vision_proj (256,768)` và nạp strict; lỗi ở biên feature. Nay chạy đủ 3 category.
2. **HINT "lệnh chạy lại"** → sai cờ (`--variant`). Nay có CLI riêng, đã chạy thật.
3. **LIMN "chỉ 2.20 mới load được"** → không tái lập. Cả 2.20 và 2.24 unpickle được; pin giữ lại vì là pin tác giả, không phải vì lỗi load.
4. **PAIR "public code chắc chắn không phải code đã train"** → hạ xuống "tương đương chưa xác lập". Hình học 2×4 và 5.20% tham số ngoài forward là **đo được**, nhưng không chứng minh nguyên nhân hay tính bất khả thi.

Thêm: **SPRC ViT-L** bị hạ từ `AUTHOR_VERIFIED` xuống `LOAD-COMPATIBLE` — nạp đúng kiến trúc **không** phải tác giả xác nhận.

### 10.4 Còn lại cho bạn

Chỉ còn **một** việc chặn kết quả: **(D-1)** đặt file SPRC ViT-g vào `workbench/artifacts/checkpoints/sprc/sprc_fiq.pt`. **D-8 đã đóng** — caption CSMCIR tải được từ Hub tác giả và entry đã chạy xong (3.18). **D-7 đã đóng** trong buổi 20261107 (áp dụng hẹp hơn đề xuất cũ: `runtime_overrides` thay vì sửa số lịch sử). Còn lại **D-2** (PAIR), **D-3** (diff registry), **D-4** (GloVe recipe), ~~D-10~~ đã đóng ở 3.23; chỉ còn D-2/D-3/D-4 là quyết định chính sách. **D-5/D-6/D-9** đã áp dụng.