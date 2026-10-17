# PER-QUERY IMPLEMENTATION REPORT — LIMN

Mục tiêu §9: xuất ranking theo từng truy vấn cho LIMN, nối vào schema canonical, DuckDB, API và
frontend — **mà không** sao chép hay thay thế evaluator gốc.

## 1. Vấn đề thật sự

`LIMN/test.py::test` tính **toàn bộ ranking** rồi **vứt đi**:

```python
nn_result = [np.argsort(-sims[i, :])[:50] for i in range(sims.shape[0])]
# rồi chỉ giữ lại [(f'{name}_r{k}', r) for k in (1, 10, 50)]
```

Mọi màn hình phân tích theo truy vấn (`/api/queries`, so sánh cặp, Jaccard thất bại) cần đúng thứ bị
vứt đó. Trước lượt này, `ResultRun`/`QueryResult` **đã tồn tại** trong schema, `load_runs`, DuckDB
(`queries`, `top_results`), `analysis.py` và endpoint `/api/analysis` **đã tiêu thụ** nó — nhưng
**không có gì sản xuất ra nó**. Nhánh này chưa từng chạy thật.

## 2. Kiến trúc

Chọn **chiến lược B của §11**: tái lập thủ tục native trong một observer riêng, có thể review độc lập,
và **đối chiếu với `test()` nguyên bản**.

```
LIMNAdapter → workbench/replay/limn.py
                ├── native_test.test(...)            ← tổng hợp CHÍNH THỨC (không đổi)
                └── limn_observer.observe_category() ← ranking theo truy vấn (mới)
                          ↓
                    cross_check_with_native()         ← PARITY 1 (hit count, khớp tuyệt đối)
                          ↓
                    publication.publish()             ← validate → atomic → đọc lại → xuất
```

Cái gì **không** bị đổi: `models/` source ghim (không patch), hàm `test()` (không sửa), trọng số
(không sửa), tham số batching/dtype/device/normalize/mask/`np.argsort`.

Cái gì observer **tái sử dụng** chứ không viết lại: cùng `model.compose_query`/`compose_target` với
cùng tham số, cùng object dataset native (`test_queries`/`test_targets`), nên batching, thứ tự query,
thứ tự gallery và cả mask đều là của native.

## 3. Những chỗ dễ sai và cách chốt

| Vấn đề | Chốt |
|---|---|
| **Rank phải tính trên toàn bộ ranking**, không phải trên Top-K đã lưu | `rank_queries` duyệt `np.argsort` đầy đủ; test tham số hoá `target_rank ∈ {1,10,50,51,200,201,1000}` và test riêng "target ngoài Top-K vẫn có rank thật" |
| **Mask, không xoá**: reference vẫn chiếm một slot | Test chốt gallery vẫn đủ `size` phần tử và reference xuống **cuối** (`[0,1,2,4,5,3]`), không bị bỏ |
| **Sentinel đúng** | Test chốt `NATIVE_REFERENCE_MASK == -10e10` (không phải `-1e11` hay `-inf`) |
| **Danh tính ảnh**: dataset native dùng **vị trí nguyên**, không phải ID | Observer map `images[pos]` một lần duy nhất. Test chốt `query_id = category:index:reference:target` và **không** dùng vị trí |
| **Caption thô ≠ text đưa vào model** | `raw_captions` đọc từ `cap.{cat}.val.json` gốc; `model_input_text = entry['mod']['str']` (đã lowercase, bỏ punctuation, áp correction dict, nối `" and "`). Test chốt hai trường khác nhau và không bị ghi đè lẫn nhau |
| **Gallery size khác nhau theo category** | `category_gallery_sizes` + `category_eligible_counts`; validator dùng bound **theo category**, và test chứng minh rank kiểu toptee không lọt qua bound của dress |
| **Không được tự chứng thực** | `publish()` validate schema → ghi atomic → **đọc lại từ đĩa** → validate lần nữa; `verify_run_identity()` tính lại `run_id` từ provenance |

## 4. PARITY 1 — điều kiện để được xuất bản

`cross_check_with_native()` so **Recall tính lại từ ranking quan sát được** với **Recall native** cho
`R@1`, `R@10`, `R@50` **từng category**, ở dạng **số nguyên hit**, không phải so với một tolerance:

> Lệch ⇒ `QUERY_EXPORT_PARITY_MISMATCH` ⇒ **native aggregate vẫn được xuất, artifact per-query bị
> giữ lại**. Không sửa metric native để tạo đồng thuận.

Đây là khác biệt cốt lõi với "observer viết lại evaluator": observer **không được phép** tự quyết là
đúng. Ba test chốt hành vi: mismatch thì giữ artifact lại, parity sạch thì xuất (positive control), và
một category đơn lẻ thì bị từ chối trước khi chạy suy luận.

## 5. Đối chiếu PARITY 2 (khác hẳn)

Parity 1 (instrumentation) ≠ parity 2 (paper). Observer có thể trung thành tuyệt đối với evaluator
native mà điểm native **vẫn khác bài báo**. Đó không phải lỗi instrumentation, và được kiểm bằng
`benchmark_gate` sẵn có, không trộn vào cổng parity ở trên. Vì vậy trường trong output tên là
`instrumentation_parity`, **không** phải `parity_status` (tên sau đã mang nghĩa paper-parity trong
`metrics_extraction` và UI).

## 6. Nối vào chuỗi bằng chứng

`verify_run.py` đã đọc `data_kind` của report và **từ chối mọi thứ không phải `experiment`** — nên
mock/synthetic **không thể** trở thành bằng chứng nghiên cứu qua đường này. Artifact per-query đi kèm
`run_id` dẫn xuất, `checkpoint_sha256`, `upstream_commit`, `runtime_adaptation`, và
`category_gallery_sizes`; `validate_results.py` kiểm cả danh tính lẫn provenance.

## 7. Trạng thái

| Mục | Trạng thái |
|---|---|
| `QUERY_EXPORT_IMPLEMENTED` | **ĐÚNG** — có CLI (`--publish-query-results`), có test end-to-end |
| `QUERY_EXPORT_NATIVE_PARITY_VERIFIED` | **PENDING** — cần suy luận GPU thật trên PC. Máy này không chạy được LIMN (`WORKBENCH_LIMN_PYTHON` chưa đặt, không có GPU) |
| `GPU_SCIENTIFIC_VALIDATION_READY` | **CHƯA** — xem §14 của báo cáo cuối |

Không có chứng thư GPU nào được tạo ra trên máy này.

## 8. Cách chạy trên PC

```bash
# Chạy đầy đủ ba category + xuất artifact per-query canonical
python -m workbench.replay.limn \
  --checkpoint-root "$WORKBENCH_CHECKPOINT_ROOT" \
  --dataset-root    "$FASHIONIQ_ROOT" \
  --source-root     "$WORKBENCH_THIRD_PARTY_ROOT" \
  --publish-query-results "$WORKBENCH_RESULTS_ROOT"
```

Từ chối theo tên nếu: thiếu bundle ba category, chạy một category đơn lẻ cùng cờ xuất bản, hoặc
instrumentation parity lệch. Sau đó rebuild chỉ mục và mở UI:

```bash
python -m workbench.backend.main --rebuild-index   # hoặc POST /api/index/rebuild
```
