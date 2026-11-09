# Hiệu năng đo được — decode worker sweep

**Sinh tự động từ `perf_raw.jsonl`** bởi `gen_performance_doc.py`. Không có ô nào được gõ tay; mọi con số dưới đây đọc từ file raw.

## Phạm vi đo — đọc trước khi trích bất kỳ số nào

Phép đo này đo **producer stage: decode + transform**, đúng hình dạng mà adapter chạy trong production — `map_fn` của thread pool làm `Image.open` → `convert("RGB")` → `transform(...)` → stack, y như `model_habit.py:358-364`; không có vòng PNG nào. Nó **không** đo model forward pass, và **không** phải thời gian end-to-end của một lượt chạy đầy đủ.

**Tám lượt composition run vẫn KHÔNG có timing.** Chúng chạy khi bốn lane khác đang hoạt động, nên wall-time của chúng ghi `not_measured (concurrent load)`. Phép đo producer stage dưới đây không biến chúng thành baseline, và không được trích thay cho một baseline end-to-end.

- sample ids: `workbench/docs/reviews/20261109-clean-rerun/perf_sample_ids.txt` (9 id, đã kiểm rank)
- reference ranks: `workbench/artifacts/composition/runs/5dee7425dd46c5f154808aac/ranks/encoder_fashioniq.json`
- loadavg lúc bắt đầu đo: `0.83 0.85 1.20 1/2457 1886148` (box rảnh: sweep đã thoát, xác nhận 2 lần)
- warmup: 1 lượt cold + >=3 lượt steady; cold **bị loại** khỏi median
- timed scope: decode + transform cho từng batch, producer bounded bởi `adapters.prefetch_rows` như evaluator dùng

## Bảng kết quả

| workers | steady trials | median img/s | range (min–max) | cold img/s | peak RSS MiB | VRAM MiB | parity |
|---:|---:|---:|---|---:|---:|---:|---|
| 0 | 3 | 554.5 | 548.1–554.6 | 548.0 | 231.0 | 852 | IDENTICAL |
| 2 | 3 | 1032.8 | 998.1–1041.7 | 988.4 | 309.9 | 852 | IDENTICAL |
| 4 | 3 | 1756.2 | 1724.5–1826.1 | 1752.7 | 398.2 | 852 | IDENTICAL |
| 6 | 3 | 2282.1 | 2240.8–2288.9 | 2342.5 | 504.3 | 852 | IDENTICAL |
| 8 | 3 | 2688.1 | 2580.2–2688.9 | 2784.3 | 602.9 | 852 | IDENTICAL |

**Đề xuất:** `workers=8` — median 2688.1 img/s, peak RSS 602.9 MiB.


Hiệu lực **chỉ** cho producer stage; chưa chứng minh tăng tốc end-to-end. Model forward pass không nằm trong phép đo này.

**Đường cong CHƯA phẳng ở setting cao nhất đã đo.** 8 vẫn nhanh hơn 6 18%, nên đây là điểm cao nhất **đã thử**, không phải điểm tối ưu — một setting cao hơn có thể còn nhanh hơn nhưng nằm ngoài ngân sách CPU nên không thử.

**Ràng buộc ngân sách:** `workers=8` trên tổng 10 thread của CIR chỉ còn 2 thread cho tiến trình chính, torch intra-op và BLAS. Đây là **trần**, không phải mức thoải mái: khi một lượt chạy thật vừa decode vừa chấm, tiến trình chính cần chỗ. Peak RSS ở setting này là 603 MiB, tăng mạnh so với serial — chọn theo end-to-end của một lượt thật, không chỉ theo img/s ở đây.

## Row schema theo handoff §C

| trường | giá trị trong lần đo này |
|---|---|
| baseline | chưa có baseline sạch trước đó; đây là lần đo nền đầu tiên |
| SHA/env | `861cfafabc297f56d17a4bc62cdc2952d9c0fe3c` (**dirty tree** — the measurement describes the working tree, not the commit) · `/home/heheboiz/data/cir/workbench/envs/ctl312h/bin/python` · Python 3.12.11 |
| sample IDs | `workbench/docs/reviews/20261109-clean-rerun/perf_sample_ids.txt` |
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

