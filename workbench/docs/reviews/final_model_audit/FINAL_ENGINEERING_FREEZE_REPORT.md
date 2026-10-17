# FINAL ENGINEERING FREEZE REPORT — CIR FAILURE ANALYSIS WORKBENCH

Ngày: 2026-10-17 · Nhánh: `research/cir-failure-workbench` · Nhánh `main`: **không đổi**

## 1. Phán quyết

# `ENGINEERING_FREEZE_READY_WITH_DOCUMENTED_EXTERNAL_BLOCKERS`

Mọi hạng mục định nghĩa `ENGINEERING_FREEZE_READY` (§37) đều đạt. Các blocker còn lại là
**artifact bên ngoài** (checkpoint/dataset/asset tác giả chưa công bố, và GPU đích không có trên máy
này), không phải khuyết điểm phần mềm. **Không** có xác thực khoa học trên GPU, và điều đó được ghi
rõ chứ không che.

## 2. Git

| | |
|---|---|
| HEAD ban đầu | `5726d5f504b60cc41f7932165781eeb4de947a16` |
| Nhánh | `research/cir-failure-workbench` |
| `main` | không đổi, không merge, không push |
| Force-push / rebase / reset --hard | **không** dùng |

## 3. 16/16 model đã audit

Mỗi model có một báo cáo riêng trong `workbench/docs/reviews/final_model_audit/<model_id>.md` với cấu
trúc A–J và trục phán quyết riêng. Tóm tắt ở `CONSOLIDATED_MODEL_MATRIX.md`. Không có model nào nhận
một "PASS" chung chung, và không model nào bị gán protocol chỉ để đủ số 16.

## 4. Phát hiện nghiêm trọng nhất: `ptha_mtst`

Hai dòng benchmark gán `gallery_protocol: fashioniq_original_split` cho một bài báo **không nêu** cách
dựng gallery, dựa trên một file của repo **chưa được sync hay pin**, và mâu thuẫn với chính registry
của model đó (`native_protocol: null`). Tệ hơn, `matches()` coi protocol vắng mặt là **wildcard**, nên
dòng đó khớp với **mọi** protocol. Sửa ở cả ba tầng (dữ liệu extracted, `papers.yaml`, và logic so
khớp), kèm test hồi quy cho cả hướng "không wildcard" lẫn hướng "vẫn phải báo cáo dòng đó cho người
vận hành".

Đây đúng là loại P0 mà mục 8 cấm: một so sánh giữa hai candidate universe khác nhau có thể được
chứng thực.

## 5. Per-query export cho LIMN

Đã hoàn thành, có entry point riêng, có test đầu-cuối. Chi tiết ở `PER_QUERY_IMPLEMENTATION_REPORT.md`.

| Trạng thái | Giá trị |
|---|---|
| `QUERY_EXPORT_IMPLEMENTED` | **YES** |
| `QUERY_EXPORT_NATIVE_PARITY_VERIFIED` | **PENDING** (cần GPU đích) |
| `GPU_SCIENTIFIC_VALIDATION_READY` | **NO** |

Hai parity được tách bạch và **không** bị trộn: parity thiết bị đo (observer vs native, khớp tuyệt đối
theo hit count, chặn xuất bản) và parity bài báo (native vs paper, do `benchmark_gate` sẵn có quyết).

## 6. Một lỗi hồi quy do chính lượt này gây ra, và đã sửa

Trung thực mà nói: việc đặt observer + schema canonical ngay trong `workbench/replay/limn.py` đã làm
**hỏng cold start**. Bộ lập kế hoạch môi trường truy vết import từ file đó (kể cả import bên trong
hàm, vì nó dùng `ast.walk`), nên LIMN bỗng đòi interpreter cho `numpy`/`pydantic` — những gói đường
chạy native không hề dùng. Hai bài kiểm thử cold-start thất bại.

Đã sửa bằng cách **tách** năng lực tuỳ chọn ra `workbench/replay/limn_export.py`, rút một loader
nguyên bản thuần trong `limn.py` (không thêm import nào), và giữ `limn.py` đúng như đồ thị import ban
đầu. Test hồi quy chốt rằng không tên nào của observer/publication rò được vào module bị truy vết.

Bài học được ghi thành test, không chỉ thành lời: `test_the_export_entry_point_is_separate_from_the_traced_native_replay`.

## 7. Bốn defect khác đã sửa (xem `DEFECT_LEDGER.md`)

| ID | Mức | Tóm tắt |
|---|---|---|
| F-01 | P0 | protocol gán không có bằng chứng cho `ptha_mtst` |
| F-02 | P0 | protocol vắng mặt bị coi là wildcard trong so khớp reference |
| F-03 | P1 | hai bộ phân giải khoá ledger với ưu tiên khác nhau → khoá thứ hai, capability ký bằng khoá không kiểm chứng được |
| F-04 | P1 | `pc_readiness` phân loại theo vị trí needle → checkpoint thiếu bị báo là dataset thiếu |
| F-05 | P1 | `run_id` không được tính lại → artifact bị đổi tên vẫn hợp lệ schema |

## 8. Trạng thái xác thực GPU

**Chưa có.** Không có GPU đích trên máy này, `WORKBENCH_LIMN_PYTHON` chưa đặt, và không có suy luận
thật nào được chạy. Không chứng thư GPU nào được tạo ra.

Các mục PENDING được nêu tên thay vì được coi là PASS:
`GPU_RUNTIME`, `MODEL_INFERENCE`, `MODEL_EVALUATION`, `SCIENTIFIC_OUTPUT`, `DEPLOYMENT`,
`CONFIGURATION`, `OPERATIONS`.

## 9. Điều lượt này **không** làm, và vì sao

* **Không viết 11 parser metric còn thiếu.** Cần output thật của evaluator; 11 model chưa chạy được.
  Viết parser từ phỏng đoán định dạng sẽ tạo ra chính loại xác thực giả mà dự án đang chống.
* **Không bịa entrypoint CLI** cho 10 model mà source gốc không có.
* **Không gán protocol** cho `ptha_mtst`.
* **Không tải** FashionIQ đầy đủ, checkpoint nhiều GB, hay cài môi trường GPU.
* **Không dựng bảng điều khiển mới**, không thêm dataset mới, không đụng `main`.
