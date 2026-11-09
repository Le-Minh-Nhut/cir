# Các lần chạy native FashionIQ — bằng chứng thực thi

Phạm vi: thực thi đánh giá native FashionIQ cho các model đã có interpreter + checkpoint hoạt
động. Mọi con số dưới đây đều đến từ một lần chạy thực tế trên máy này; log nằm trong
`raw/native/`. Không có gì ở đây là tái lập paper trừ khi được ghi rõ.

Máy: `heheboiz-nhut-pc`, RTX 5070 Ti (15.44 GiB VRAM khả dụng), **32 GB RAM (31 GiB dùng được)**,
dùng chung với worker khác (`limn-parity` sở hữu LIMN). GPU tuần tự hoá; **RAM là ràng buộc chính
cho hầu hết model, VRAM là ràng buộc của TME**.

## Kết quả: 7/7 model hoàn tất native thật

| model | protocol | R@10 dress/shirt/toptee | macro R@10 | macro R@50 | wall | peak RSS |
|---|---|---|---|---|---|---|
| **encoder** | fashioniq_val_split | 51.96 / 54.96 / 61.96 | **56.29** | **77.63** | 131 s | 18.63 GiB |
| **hint** | fashioniq_val_split | 51.26 / 62.41 / 65.73 | **59.80** | **80.13** | 271 s | 20.51 GiB |
| **habit** | fashioniq_full_gallery_ref_excluded | 49.18 / 56.18 / 59.26 | **54.87** | **74.50** | 421 s | 27.54 GiB |
| **intent** | fashioniq_full_gallery_ref_excluded | 49.93 / 56.18 / 58.49 | **54.87** | **75.43** | 440 s | 26.11 GiB |
| **conesep** | fashioniq_full_gallery_ref_excluded | 49.63 / 56.62 / 58.03 | **54.76** | **75.22** | 441 s | 27.05 GiB |
| **tme** | fashioniq_original_split | 49.33 / 56.18 / 59.26 | **54.92** | **74.84** | 120 s | 25.54 GiB |
| **ptha_mtst** (TRANSFER) | fashioniq_original_split | 33.12 / 38.86 / 41.71 | **37.90** | **59.84** | 240 s | 25.79 GiB |

Tất cả 7 đều `complete = true`, đủ 3 category dress/shirt/toptee.
Không lần nào chạy khi GPU đang bận: mỗi lần đều kiểm tra
`nvidia-smi --query-compute-apps` = NONE ngay trước khi khởi động.

**ptha_mtst là TRANSFER CIRR→FashionIQ** — không phải tái lập paper FashionIQ. JSON của nó tự
ghi `label: "PTHA+MTST | CIRR checkpoint -> FashionIQ | cross-dataset transfer | exploratory"`
và `train_dataset: CIRR`. Macro R@10 = 37.90 thấp hơn ngữ cảnh paper (54.99) đúng như kỳ vọng của
một checkpoint huấn luyện trên dataset khác; chênh lệch −17.09 được ghi trong JSON.

## TME — hoàn tất sau khi hạ query batch (knob có sẵn của upstream)

TME qua preflight. Lần 1 chết vì **VRAM GPU** ở batch mặc định 32:
`torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 6.20 GiB. GPU 0 has a total
capacity of 15.44 GiB of which 4.37 GiB is free.` (host RSS chỉ 22.98 GiB, RAM không phải nút thắt).

Lần 2 chạy với `WORKBENCH_TME_QUERY_BS=8`. Đây **không phải thay đổi protocol**: chính
`TME/src/precompute_evaluation.py:65` đọc biến này (`int(os.environ.get("WORKBENCH_TME_QUERY_BS", 32))`),
và dòng 79 trong cùng file đã ghi chú cách xử lý OOM. Giảm query batch làm nhỏ phép `torch.matmul`
tại `blip2_qformer_cir_image_diff_features.py:250` — đúng chỗ đã OOM.

Kết quả: `EXITCODE=0`, wall 120 s, **VRAM đỉnh 11.0 GiB** (vừa card 15.44 GiB), host RSS đỉnh
25.54 GiB. macro R@10 = **54.92**, R@50 = **74.84**, `aggregate.complete = true`, đủ 3 category.

Lệnh:

```
export WORKBENCH_TME_QUERY_BS=8
workbench/envs/tme-adapted/bin/python workbench/replay/tme.py \
  --checkpoint-root .../cohort/downloads/tme --dataset-root data/FashionIQ \
  --source-root workbench/third_party/TME --output .../raw/native/tme.json
```

## Ai giữ 13 GB (điều tra theo yêu cầu)

`AnonPages` 13.1–13.8 GB lúc điều tra **không phải** rò rỉ của tôi. Đó là `limn-parity` chạy
**hai** tiến trình `t4_parity.py` song song:
- pid 2292915 `--category dress` — RSS 7.27 GB
- pid 2293499 `--category toptee` — RSS 7.42 GB

Cả hai là job sống của worker khác; tôi **không** kill. Sau khi LIMN xong, `AnonPages` rơi về
0.6–0.7 GB. Không tìm thấy tiến trình rò rỉ nào của chính tôi.

## RAM guard (dùng thật, trước mỗi lần chạy)

`raw/native/ram_guard.py` đọc `/proc/meminfo` và từ chối nếu `MemAvailable < predicted + headroom`.
Số đo `MemAvailable` ngay trước mỗi lần khởi động:

| model | predicted (GiB) | headroom | need | MemAvailable đo | verdict | kết quả |
|---|---|---|---|---|---|---|
| encoder | 11.6 | 6.0 | 17.6 | 28.56 | GO | OK, đỉnh 18.63 |
| hint | 11.6 | 6.0 | 17.6 | 28.63 | GO | OK, đỉnh 20.51 |
| conesep #1 | 15.5 | 6.0 | 21.5 | 28.64 | GO | OOM, đỉnh 25.82 (dự đoán sai) |
| conesep #2 | 25.9 | 3.0 | 28.9 | 28.50 | REFUSE | không chạy |
| conesep #3 | 25.9 | 2.0 | 27.9 | 28.52 | GO | OK, đỉnh 27.05 |
| intent | 25.9 | 2.0 | 27.9 | 28.10 | GO | OK, đỉnh 26.11 |
| habit | 20.7 | 4.0 | 24.7 | 28.51 | GO | OK, đỉnh 27.54 |
| ptha_mtst | 17.0 | 4.0 | 21.0 | 28.46 | GO | OK, đỉnh 25.79 |
| tme #1 | 24.0 | 2.0 | 26.0 | 28.35 | GO | VRAM OOM (host OK 22.98) |
| tme #2 (BS=8) | 24.0 | 2.0 | 26.0 | 28.50 | GO | OK, host 25.54, VRAM 11.0 |

Bài học: dự đoán từ công thức tensor thuần **thấp hơn thực tế ~10 GiB** do overhead dựng dataset
(PIL + float trung gian). Sau khi đo, tôi dùng số đo thật làm `predicted`. Guard đã chặn đúng một
lần (conesep #2) và chỉ cho qua khi dư ≥2 GiB.

## Vì sao `--category` KHÔNG giúp giảm RAM (đã kiểm code)

- `HABIT/datasets.py:45-47` — `__init__` gọi `get_test_data('dress'/'shirt'/'toptee')` **vô điều kiện**.
- `ENCODER/datasets.py:80-82`, `HINT/datasets.py:39-41`, `INTENT`, `ConeSep` — cùng khuôn iLearn.

`--category` chỉ thu hẹp vòng `test()` (tính recall), **không** thu hẹp bước dựng dataset. Vì vậy
chạy từng category không giảm đỉnh RAM. Khác biệt thật giữa model là **split gallery**:
- `encoder`, `hint`: chỉ nạp target được query tham chiếu (val-split) → ~11.58 GiB tensor.
- `habit`, `intent`, `conesep`, `ptha`: nạp **toàn bộ** gallery (original-split) → ~15.46 GiB tensor,
  đỉnh thực 26–28 GiB.

## Ghi chú encoder: lỗi CWD, không phải lỗi code

Lần đầu encoder chạy từ `/home/heheboiz/data/cir` thất bại tức thì (exit 1):
`RuntimeError: Pretrained weights (./open_clip_pytorch_model.bin) not found`. Nguyên nhân:
`model_try2.py:134` đặt `clip_path = './'` cứng. Sửa bằng cách chạy với CWD =
`workbench/third_party/ENCODER` (nơi asset nằm, upstream cũng chạy từ repo root) — **không sửa
file nguồn nào**. Đây là invocation source-faithful.

## Checkpoint digests (sha256, tính trên đĩa)

| model | checkpoint | sha256 |
|---|---|---|
| habit | HABIT-FIQ_N0.2.pt | 814d2d0ceb29ef910f37562e8bea8a76cf38aa546fbc2102e495df687c276b03 |
| intent | intent_fiq_0.2.pt | 2a504cadfd6b1c2486a2c508538b0160c5653f62c7cd11f4f3d8a9534dc26afc |
| hint | fashioniq.pt | bf9c72d42206b667ed586758c15c648970df821857e4067f26155a114a6517d4 |
| conesep | ConeSep-FIQ-N0.2.pt | 5a864ee5a78c1d249320c4f997cc7d14b529b329b2efba5577c82dbe019ed9d2 |
| encoder | fashioniq.pt | 0f31c50150a7bda3495e3378df13b4cec7924b52efd23b5e5dd9731459da9173 |
| tme | fashioniq_0_best_model.pth | fea7fc08760dad1f5a66184ea7e65b45388f9781157cbd33a3b64db74103968a |
| ptha_mtst | tuned_mtst_cirr.pt | 1c4dab86a9f66dfe10aff75b39790432eb6cc746cf27b6cfab74b66d93b7588f |

ptha_mtst JSON tự xác nhận `checkpoint_sha256 = 1c4dab86…` và `strict_load: true` (0 missing,
0 unexpected key).

## Phiên bản interpreter (đo thực tế)

| interpreter | python | torch | cuda |
|---|---|---|---|
| `workbench/envs/ilearn39-adapted/bin/python` | 3.9.25 | 2.7.1+cu128 | True |
| `workbench/envs/encoder310-adapted/bin/python` | 3.10.20 | 2.7.1+cu128 | True |
| `workbench/envs/tme-adapted/bin/python` | 3.10.22 | 2.7.1+cu128 | True |
| `workbench/artifacts/environments/cir-005ef4e1b02999a5/bin/python` | 3.10.22 | 2.7.1+cu128 | True |

## Preflight (cả 7)

Tất cả READY (tme kèm 2 cảnh báo `EXISTS_UNVERIFIED_CHECKSUM`; ptha_mtst không có
`--preflight-only` — gọi vào là chạy luôn).

Số query val (tự đếm từ `captions/cap.*.val.json`): dress 2017, shirt 2038, toptee 1961, tổng
6016 — khớp với `test_queries` mà evaluator dùng (mỗi caption val = 1 query).

## Artifact đã ghi

`raw/native/`:
- `encoder.json`, `encoder.meta.txt`, `encoder.ramguard.txt`, `encoder.stdout.log`, `encoder.stderr.log`
- `hint.json`, `hint.meta.txt`, `hint.ramguard.txt`, `hint.stdout.log`, `hint.stderr.log`
- `conesep.json`, `conesep.meta.txt`, `conesep.ramguard.txt`, `conesep.stdout.log`, `conesep.stderr.log`
- `habit.json`, `habit.meta.txt`, `habit.ramguard.txt`, `habit.stdout.log`, `habit.stderr.log`
- `intent.json`, `intent.meta.txt`, `intent.ramguard.txt`, `intent.stdout.log`, `intent.stderr.log`
- `ptha_mtst.json` (stdout thô), `ptha_mtst.result.json` (JSON sạch), `ptha_mtst.meta.txt`,
  `ptha_mtst.ramguard.txt`, `ptha_mtst.stdout.log`, `ptha_mtst.stderr.log`
- `tme.json`, `tme.meta.txt` (có `PEAK_VRAM_MIB=11014`), `tme.ramguard.txt`, `tme.stdout.log`,
  `tme.stderr.log`
- `ram_guard.py`, `oom_evidence.log`
