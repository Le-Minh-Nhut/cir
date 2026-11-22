# Báo Cáo Tạm Dừng Phase Publish Gate V3 (`CURRENT.json`) & Bàn Giao Trạng Thái G1/G2

- **Ngày chốt:** 2026-11-22
- **Nhánh / Commit nền:** `research/cir-failure-workbench` (`7be7091`)
- **Báo cáo kỹ thuật đi kèm:**
  - Tổng hợp 16 cấu hình & verdict `parity_gate`: [`REPORT.md`](file:///home/heheboiz/data/cir/workbench/docs/reviews/20261122-g1-g2-parity-verification/REPORT.md)
  - Đối chứng 4 sorter (14/14 query Top-10) & DCNet `comp_scores`/`corr_scores`: [`TOP10_BOUNDARY_AND_DCNET_DIFFS.md`](file:///home/heheboiz/data/cir/workbench/docs/reviews/20261122-g1-g2-parity-verification/TOP10_BOUNDARY_AND_DCNET_DIFFS.md)
  - Bằng chứng G1 worker thật vs native (9 model nhóm đầu): [`g1_real_worker_9models.json`](file:///home/heheboiz/data/cir/workbench/docs/reviews/20261122-g1-g2-parity-verification/g1_real_worker_9models.json)

---

## 1. Quyết Định Tạm Dừng & Tài Sản Đang Sử Dụng Cho Research

1. **Quyết định:** Tạm dừng việc hoàn thiện hạ tầng publish **Gate V3 (`workbench/artifacts/parity/v3/CURRENT.json`)** để tập trung trực tiếp vào mục tiêu chính là **nghiên cứu failure cases của 15 paper CIR trên FashionIQ**.
2. **Nguồn dữ liệu research giữ nguyên (100% nguyên vẹn):**
   - Toàn bộ **16/16 cấu hình (15 paper, `PAIR` hoãn)** chạy trực tiếp bằng **code evaluation public của tác giả** nằm tại `workbench/artifacts/results/` (`6.016` query/cấu hình, `top_k_saved = 200`).
   - Database DuckDB đã index đủ 16 run: [`workbench/artifacts/workbench.duckdb`](file:///home/heheboiz/data/cir/workbench/artifacts/workbench.duckdb) (`627 MB`), đã verify hiển thị trên Workbench Web (`16/16 WEB_VERIFIED`).
3. **Trạng thái adapter đã sửa (sẵn sàng dùng lại khi cần chạy trên tập dữ liệu mới):**
   - Đã sửa triệt để toàn bộ lỗi thật tại adapter/harness cho `TME`, `CSMCIR`, `ENCODER`, `CLIP4Cir` (`fullft`/`noft`), `DCNet`, `Air-Know`, `CLVC-Net` và harness kiểm chứng `verify_g1_g2_adapter.py`.
   - Kiểm chứng chẩn đoán G2 trên toàn bộ 3 category (`6.016` query/cấu hình): **16/16 cấu hình đạt `0` non-tie rank change, `+0/-0` Top-10, `+0/-0` Top-50, `ΔR@10 = 0.0000`, `ΔR@50 = 0.0000`, `0/60.160` vi phạm ngưỡng V3 (`atol=1e-4, rtol=1e-4, tie_epsilon=1e-6`)**; riêng **DCNet** và **Air-Know** đạt **bit-exact `0.00e+00`**.

---

## 2. Lý Do Đến Nay Chưa Có Repo Nào Đạt `PARITY_PASS` Chính Thức Ở Gate V3

Gate chính thức (`workbench/backend/parity_gate.py`, chuẩn `parity-absolute-relative/v3`) yêu cầu mỗi model phải có **plan V3 bất biến**, đủ **10 role** (`native_adapter_*`, `*_batch_1/n`, `*_repeat_1/2`), file **`score-ranks.jsonl` toàn gallery**, **G1 tensor/text capture**, **khớp tuyệt đối hit count ở cả `r1`, `r10`, `r50`**, và được đóng gói/publish qua `publish_v3_generation` vào `CURRENT.json`.

Chưa có model nào được gắn `PARITY_PASS` do **3 nhóm blocker ở lớp gate/harness**, không phải do sai lệch lớn ở model:

### Nhóm A — Đã chạy thử đủ 10 role V3 thật nhưng chạm blocker cụ thể (2 cấu hình)
| Model | Verdict thật `parity_gate` V3 | Nguyên nhân cụ thể |
|---|---|---|
| **LIMN** (`base_iter0_all`) | `PARITY_MANDATORY_SUBCHECK_FAILED` (đã gỡ khỏi `CURRENT.json`) | Vượt toàn bộ G1 + 10 role + `r10`/`r50` tuyệt đối, nhưng fail **1 check duy nhất** `shirt.metrics.hit_count_equality` ở **`r1` (`516` vs `517`)** do 1 query tie-only đổi hạng `#1↔#2` (chênh FP32 `~1e-7 < tie_epsilon`). |
| **CLVC-Net** | `PARITY_EVIDENCE_MALFORMED` (chưa publish) | Đã chạy đủ 30 role (10 role × 3 cat), đạt `max_abs ≤ 1.13e-6`, `0` rank đổi ở `shirt`/`toptee`, 1 tie `#2↔#3` ở `dress`, khớp tuyệt đối `r1`/`r10`/`r50`. Bị chặn bởi schema của [`prepare_clvc_tme_v3_plans.py`](file:///home/heheboiz/data/cir/workbench/replay/prepare_clvc_tme_v3_plans.py): `v3 identity.checkpoint must be a SHA256 digest of verified checkpoint bytes` (và `publish()` trong [`clvc_tme_v3_parity.py`](file:///home/heheboiz/data/cir/workbench/scripts/clvc_tme_v3_parity.py) chưa gom đủ 8 role control vào `parity_v3.runs`). |

### Nhóm B — Gate vẫn đang đọc artifact V3 cũ từ trước đợt sửa (4 cấu hình)
- **ENCODER, TG-CIR, HINT, HABIT:** `parity_gate.evaluate_model(..., standard="parity-absolute-relative/v3")` đang trả `PARITY_MANDATORY_SUBCHECK_FAILED` vì đọc lại bộ artifact cũ trước khi sửa (còn lỗi `batch_1_vs_n.max_rel` cũ và thiếu `relation.*.digest`). Chưa chạy lại 10 role V3 trên bản code đã sửa.

### Nhóm C — Chưa có V3 publisher nối với `publish_v3_generation` (`CURRENT.json`) (10 cấu hình)
- **DCNet, Air-Know, ConeSep, INTENT, CSMCIR, SPRC, PTHA+MTST (`cirr_transfer`), TME, CLIP4Cir (`fullft`), CLIP4Cir (`noft`):**
  - Chỉ mới có `limn_v3_parity.py` và `clvc_tme_v3_parity.py` nối với `publish_v3_generation`; 9 cấu hình còn lại (ngoài `TME` dùng chung `clvc_tme_v3_parity.py`) chưa có pipeline đóng gói 10-role V3 để publish vào `CURRENT.json`.
  - Riêng 7 cấu hình nhóm sau (`CSMCIR`, `SPRC`, `PTHA+MTST`, `TME`, `CLIP4Cir FullFT/NoFT`, `DCNet`) mới xác minh G1 cấu trúc/embedding, chưa chạy script thu G1 pixel+token 2-tiến-trình độc lập đầy đủ như 9 model nhóm đầu.

---

## 3. Lộ Trình Nếu Sau Này Cần Tiếp Tục Publish V3 Gate

Khi cần mở lại việc chứng nhận `PARITY_PASS` vào `CURRENT.json`, thứ tự hiệu quả nhất là:
1. **Gỡ blocker schema của `clvc_net` / `tme`:** sửa định dạng `identity.checkpoint` (chuỗi hex SHA256 đơn) trong [`prepare_clvc_tme_v3_plans.py`](file:///home/heheboiz/data/cir/workbench/replay/prepare_clvc_tme_v3_plans.py) và gom đủ 8 role control vào `parity_v3.runs` trong [`clvc_tme_v3_parity.py`](file:///home/heheboiz/data/cir/workbench/scripts/clvc_tme_v3_parity.py) → chạy lại publish **`clvc_net`** (đã chứng minh khớp `r1`/`r10`/`r50`) để lấy `PARITY_PASS` đầu tiên, sau đó chạy **`tme`**.
2. **Nối publisher V3 cho nhóm `0` rank đổi (chắc chắn không bị tie-flip ở `r1`):** `dcnet`, `airknow`, `conesep`, `encoder` (và `ptha_mtst` giữ nhãn `cirr_transfer`).
3. **Bổ sung thu G1 tensor 2-tiến-trình** cho 7 cấu hình nhóm sau và xử lý chính sách tie ở `r1` (nếu cần) cho nhóm còn tie-swap như `limn`.
4. **Lưu ý vận hành:** Mỗi model full 10-role V3 tốn ~15–30 phút GPU, ~2.9 GB scratch tạm + ~1.1 GB generation publish; cần chạy tuần tự 1 GPU job và xóa scratch ngay sau mỗi model.

---

## 4. Bộ Dataset Mới Bổ Sung Cho Research (`EDIR` & `PinPoint`)

Đã tải bổ sung 2 benchmark CIR mới vào thư mục `data/` (2026-11-22):

1. **EDIR (`EDIR-BENCH/EDIR`)** — `data/EDIR` (`32 GB`):
   - Đã tải đủ từ HuggingFace (`EDIR-BENCH/EDIR`), giải nén `imgs_mc/` (**33,429 ảnh**), giữ đầy đủ `corpus.jsonl`, `instances.jsonl`, `queries.jsonl`, `README.md` và đã xóa các file `imgs_mc.tar.part*` + `.cache`.

2. **PinPoint** — `data/PinPoint` (`6.8 GB`, metadata từ `pinpoint-dataset/`):
   - Đã tải **109,598 / 109,600 ảnh (99.998%)** về `data/PinPoint/images/{2-ký-tự-đầu}/{signature}.jpg` kèm `pinpoint_licensed.parquet` (7,635 query), `index_signatures.txt`, `image_attribution.json`, `README.md`.
   - **Ghi nhận 2 ảnh lỗi (`HTTP 403` từ Pinterest CDN, lưu tại `data/PinPoint/failed_signatures.txt`):**
     - `1269714db704dc89eb1fcfe18fbc8c4e`
     - `a5b5625b26942bd4d5cd5513b235bc83`
   - **Kiểm tra tác động trên toàn bộ 7,635 query trong `pinpoint_licensed.parquet`:**
     - `query_image_signature` / `query_image_signature2`: **0 query** bị ảnh hưởng (đủ 100% ảnh query).
     - `positive_candidates`: **0 query** bị ảnh hưởng (không mất bất kỳ đáp án đúng nào; ground-truth cho `Precision@k`, `Recall@k`, `mAP@k` nguyên vẹn).
     - `negative_candidates`: chỉ xuất hiện là **1/50 hard negative** ở đúng **23 / 7,635 query (~0.3%, không chồng lấn)**:
       - `1269714db704dc89eb1fcfe18fbc8c4e` (**11 query**): `query_01569`–`query_01574`, `query_05297`–`query_05301`.
       - `a5b5625b26942bd4d5cd5513b235bc83` (**12 query**): `query_01694`–`query_01699`, `query_06731`–`query_06736`.
     - **Lưu ý khi build FAISS index từ thư mục local:** bỏ qua 2 signature không có file ảnh trong `index_signatures.txt`; phần đánh giá `evaluate.py` và các metric chính không bị ảnh hưởng đáng kể.
