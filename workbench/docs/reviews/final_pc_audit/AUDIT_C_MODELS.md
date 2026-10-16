# AUDIT C — Models & Evaluation (main-agent audit; subagent harness unavailable)

Phạm vi: `workbench/backend/adapters/`, `workbench/replay/`, `scripts/evaluate_models.py`,
`scripts/prepare_model.py`, `scripts/download_checkpoints.py`, `scripts/validate_results.py`,
`registry/models.yaml`. Phương pháp: đọc nguồn + probe rẻ (không chạy 16 đánh giá GPU).

## VERIFIED — bundle LIMN là một lựa chọn hạng nhất

- `registry.checkpoint_by_id` resolve `base_iter0_all_categories` → `bundle_id` +
  `bundle_members = [base_iter0_dress, base_iter0_shirt, base_iter0_toptee]`; `validate_request`
  của adapter chấp nhận; lệnh dựng ra **không** kèm `--category` (tức cả 3 category).
- Run dress-only (`--category dress`) tạo aggregate `complete: false`, `observed_metrics: None`
  → không thể được phục vụ như một bản tái lập đầy đủ. Đã kiểm ở đợt trước và còn hiệu lực.

## VERIFIED — không có dữ liệu per-query nào bị bịa

- `replay/limn.py` chỉ trả `r1/r10/r50` theo category + macro; không có `queries`/`top_results`.
- `top_k` chỉ tồn tại ở `EvalRequest` (khai báo), không nơi nào sinh dữ liệu per-query.
- Chỉ `load_mock_results.py` ghi `ResultRun` schema v2. Không evaluator thật nào ghi.
- Đã ghi thành KNOWN LIMITATION (`PER_QUERY_EXPORT_STATUS.md`, §15 của guide). Không có chỗ nào
  trong đường sản xuất tự tạo `top_results`.

## VERIFIED — extraction status có thể phân biệt “đã trích được” và “không”

`extract_aggregate_metrics` phân loại: có metrics → `EXTRACTED_VERIFIED`; exit 0 nhưng không parse
được → `EVALUATION_COMPLETED_METRICS_UNPARSED` với `observed_metrics = None`. `verify_run.py`
(kể cả sau khi sửa) yêu cầu `metric_extraction_available is True` **và** status nằm trong tập đã
trích, nên một run exit 0 mà không có số không thể chứng nhận (probe của audit trước đã xác nhận
`OLD verified: True []` cho case này).

## VERIFIED — từ chối model thiếu tài nguyên là tường minh

- `airknow`, `ptha_mtst`, `combiner_rn50x4_noft`, `tgcir`, `sprc`… có `source_blocker`/reference
  chưa xác minh; `preflight_blockers` trả lý do có tên, và gate/verifier từ chối thay vì bịa số.
- Sau đợt sửa D-2/D-4, kiểu từ chối được phân biệt: “không có so sánh được công bố”
  (`NO_PUBLISHED_COMPARISON`) khác với “xác minh thất bại” — để operator biết đó là điều kiện cố
  hữu của model, không phải lỗi dữ liệu.

## CONFIRMED — C-1 (P2) `checkpoint_by_id` trả anchor cho bundle, không trả tập đầy đủ

`checkpoint_by_id` resolve bundle về member đầu tiên để giữ xử lý path/hash đồng nhất, nhưng một
caller chỉ đọc `checkpoint["filename"]` mà bỏ qua `bundle_members` sẽ vô tình đánh giá **một**
category. Đường sản xuất hiện tại (`evaluate_models.py`) mang theo `checkpoint_bundle`/
`directory_members` nên đúng, nhưng ràng buộc này dựa vào từng caller nhớ đọc `bundle_members`.
**Khắc phục tối thiểu (chưa làm):** khi `bundle_members` có mặt, cảnh báo hoặc từ chối ở những
caller chỉ dùng `filename`. Ghi nhận là rủi ro cấu trúc, không phải lỗi đang hoạt động.

## Chưa kiểm (nói rõ)
- Chạy thật bất kỳ model nào (không có GPU/dữ liệu author đầy đủ ở đây).
- Tải checkpoint (không tải để tiết kiệm băng thông/đĩa).
- Tính đúng của từng parser native của model ngoài LIMN/CSMCIR (chỉ đọc hợp đồng).
