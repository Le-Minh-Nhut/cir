# Per-query Top-K export status — LIMN and the Workbench analysis path

Kết luận: **LIMN KHÔNG xuất xếp hạng Top-K theo từng query.** Đường đánh giá thật chỉ sinh ra
Recall tổng hợp theo category (R@1/R@10/R@50) và macro aggregate. Toàn bộ tầng *tiêu thụ* dữ liệu
per-query đã được xây xong và đang chờ dữ liệu; tầng *sản xuất* thì thiếu.

## Bằng chứng (đọc từ nguồn, không suy đoán)

| Thành phần | Trạng thái | Vị trí |
|---|---|---|
| `ReplayResult` của LIMN | chỉ có `r1`/`r10`/`r50` theo category + `complete`; **không** có `queries`/`top_results` | `workbench/replay/limn.py:179-187` |
| `macro_aggregate` | trung bình cộng theo category, không trọng số theo query | `workbench/replay/common.py:98-108` |
| Đầu ra của `limn.py --output` | `categories[]` + `aggregate` + `runtime`; không có ranking | `workbench/replay/limn.py:226-243` |
| `top_k` trong `EvalRequest` | chỉ được khai báo, **không** nơi nào tiêu thụ | `workbench/backend/adapters/base.py:18`; không có trong `adapters/models.py`, `replay/limn.py`, `metrics_extraction.py` |
| Schema `ResultRun` (v2) | **đã hỗ trợ đầy đủ** per-query: `QueryResult.top_results[]`, `target_rank`, `top_k_saved` | `workbench/backend/schemas/results.py:26-113` |
| Index DuckDB | **đã có** bảng `queries` + `top_results`, COPY theo từng query | `workbench/backend/index.py:89-98` |
| API phân tích | **đã có** `/api/queries`, `/api/compare/{query_id}`, `/api/analysis`, `/api/analysis/failure-jaccard` | `workbench/backend/main.py:255-287` |
| Nhà sản xuất `ResultRun` schema v2 | **chỉ có mock loader**; không evaluator thật nào ghi file này | `workbench/scripts/load_mock_results.py` (không có `ResultRun(` trong `evaluate_models.py`, `replay/*`, `backend/*`) |

## Hệ quả trung thực

- Phân tích thất bại ở mức query (failure analysis theo từng truy vấn, so sánh cặp model trên cùng
  query, Jaccard của tập thất bại) **hiện KHÔNG vận hành được** với kết quả LIMN thật. Nó chỉ chạy
  trên dữ liệu mock.
- Vì vậy không được tuyên bố "query-level failure analysis đã hoạt động". Không có kết quả truy hồi
  nào được bịa ra ở bất kỳ đâu trong wave này.
- Đánh giá LIMN vẫn hợp lệ và có giá trị nghiên cứu ở mức category/macro: đó là thứ paper công bố
  và là thứ gate so khớp.

## Triển khai tối thiểu cần có (chưa làm — ngoài phạm vi wave này)

1. **Thu ma trận điểm theo category trong replay.** `src/evaluation/fashioniq.py` đã tính `scores`
   (shape `query_count × gallery_count`) và lấy `top_k` từ đó để tính Recall. Cần trả ma trận này
   ra ngoài đường `replay_category` thay vì chỉ trả Recall — không thay đổi phép tính Recall.
2. **Sinh `ResultRun` schema v2.** Với mỗi category: gắn `query_id`, `category`, `annotation_index`,
   `reference_id`, `target_id`, `raw_captions` (2 câu), `model_input_text`, `target_rank` (vị trí
   thật của target trong xếp hạng), và `top_results` = `min(top_k_saved, gallery_size)` mục đầu.
   Trường `top_k_saved` lấy từ `EvalRequest.top_k` (mặc định 200) — đây chính là chỗ `top_k` phải
   được tiêu thụ.
3. **Ghi file kết quả** vào `WORKBENCH_RESULTS_ROOT/<protocol>/<model_id>/<checkpoint_id>.json`
   (đường `EvalRequest.output_path` đã được tính sẵn ở `evaluate_models.py:348` nhưng hiện bị bỏ
   qua). Sau đó `rebuild_index.py` sẽ tự nạp vì `load_runs` quét `*.json` dưới results root.
4. **Bất biến phải giữ:** các validator trong `schemas/results.py` đã kiểm tra `target_rank` khớp
   `top_results` và độ sâu lưu khớp `top_k_saved`; chúng phải được thoả mãn bằng số liệu thật.

Rủi ro nếu làm ẩu: một producer nửa vời sẽ khiến `/api/queries` trả về xếp hạng không khớp Recall,
tức là đưa ra bằng chứng sai cho phân tích thất bại — tệ hơn là không có dữ liệu. Vì vậy bước 2 và
bước 4 phải đi cùng nhau.
