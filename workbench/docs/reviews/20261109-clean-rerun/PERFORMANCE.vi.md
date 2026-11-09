# Hiệu năng đo được — decode worker sweep

**Sinh tự động từ `perf_raw.jsonl`** bởi `gen_performance_doc.py`. Không có ô nào được gõ tay; mọi con số dưới đây đọc từ file raw.

## Phạm vi đo — đọc trước khi trích bất kỳ số nào

Đây là phép đo **producer stage** (decode → transform) trên một mẫu gallery cố định, không phải thời gian end-to-end của một lượt chạy đầy đủ. Model forward pass không nằm trong số nào ở đây. Trục query đã được chứng minh bit-identical ở sub-batch 2..128 trước đó; phép đo này kiểm trục **decode**.

- sample ids: `/home/heheboiz/data/cir/workbench/docs/reviews/20261109-clean-rerun/perf_sample_ids.txt` (9 id, đã kiểm rank)
- reference ranks: `/home/heheboiz/data/cir/workbench/artifacts/composition/runs/5dee7425dd46c5f154808aac/ranks/encoder_fashioniq.json`
- loadavg lúc bắt đầu đo: `1.00 1.49 1.54 2/2456 1880127` (box rảnh: sweep đã thoát, xác nhận 2 lần)
- warmup: 1 lượt cold + >=3 lượt steady; cold **bị loại** khỏi median
- timed scope: decode + transform cho từng batch, producer bounded bởi `adapters.prefetch_rows` như evaluator dùng

## Bảng kết quả

| workers | steady trials | median img/s | range (min–max) | cold img/s | peak RSS MiB | VRAM MiB | parity |
|---:|---:|---:|---|---:|---:|---:|---|
| 0 | 3 | 103.8 | 103.7–104.1 | 104.0 | 169.3 | 852 | IDENTICAL |
| 2 | 3 | 235.9 | 235.1–236.4 | 236.0 | 170.9 | 852 | IDENTICAL |
| 4 | 3 | 428.3 | 427.6–429.5 | 426.4 | 185.4 | 852 | IDENTICAL |
| 6 | 3 | 432.0 | 431.3–432.5 | 434.2 | 210.2 | 852 | IDENTICAL |
| 8 | 3 | 429.2 | 428.4–429.2 | 431.8 | 229.0 | 852 | IDENTICAL |

**Đề xuất:** `workers=4` — median 428.3 img/s, peak RSS 185.4 MiB.

Setting trong khoảng dung sai 2% của nhanh nhất: 4 (428.3), 6 (432.0), 8 (429.2). Chênh lệch giữa chúng **nhỏ hơn biên độ nhiễu** (xem cột range), nên chọn cái **rẻ nhất** về RAM và thread thay vì cái nhô lên cao nhất trong một lần đo. Toàn dải 0→8 worker trải rộng 76% so với nhanh nhất, tức knobs này **có** tác dụng thật.

Hiệu lực **chỉ** cho producer stage; chưa chứng minh tăng tốc end-to-end. Model forward pass không nằm trong phép đo này.

## Row schema theo handoff §C

| trường | giá trị trong lần đo này |
|---|---|
| baseline | chưa có baseline sạch trước đó; đây là lần đo nền đầu tiên |
| SHA/env | `085774488449e774df22e20fe410f30ad008f9d1` (**dirty tree** — the measurement describes the working tree, not the commit) · `/home/heheboiz/data/cir/workbench/envs/ctl312h/bin/python` · Python 3.12 |
| sample IDs | `/home/heheboiz/data/cir/workbench/docs/reviews/20261109-clean-rerun/perf_sample_ids.txt` |
| CPU budget | 10 thread (AGENTS.md); decode workers × 1, BLAS 1 khi chấm |
| decode workers | 0, 2, 4, 6, 8 |
| worker threads | 1 mỗi decode worker (PIL/GIL) |
| BLAS effective threads | 1, đọc lại từ runtime (`BLAS_THREADS_PINNED`) |
| query/gallery batch | 32 |
| dtype/flags | không đổi: float32, không AMP/TF32 |
| warmup policy | 1 cold + 3 steady; cold loại khỏi median |
| timed scope | decode + transform mỗi batch |
| elapsed | xem cột median/range |
| samples/s | xem `median img/s` |
| peak RSS/VRAM | xem 2 cột tương ứng |
| wait time | không tách riêng trong harness này — **unknown** |
| parity verdict | xem cột `parity` |
| raw log | `perf_raw.jsonl` |


Bảng do harness sinh: `PERF_TABLE.md`.

