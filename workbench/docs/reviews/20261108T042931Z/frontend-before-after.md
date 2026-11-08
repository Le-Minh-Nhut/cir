# Frontend before → after

## Kiến trúc cũ

- `App.tsx` 112 dòng, gần như toàn bộ logic trong **một** chỗ: 12 tab ngang, mọi state React local,
  không URL, 17 `useState` + 5 `useEffect` gọi API trên cùng component. Sự kiện "đổi context" và
  "nhận response" nằm chung một scope nên chỉ có thể phân biệt bằng đếm `useRef`.
- `CompositionView.tsx` (464 dòng) và `BenchmarkView.tsx` (210 dòng) là hai trang tự quản lý fetch,
  tự quyết định loading/empty, tự `.then(setState)` **không** kiểm tra context.
- Nguồn dữ liệu và màn hình trộn lẫn: "run" có thể là CIRCUS summary, row của DuckDB index, hoặc
  report aggregate, tuỳ trang — nên Dashboard đếm 0 experiment trong khi Composition hiện 15 dòng.
- Một cờ duy nhất (`aggregate-only`) đứng thay cho bốn câu hỏi khác nhau, và nó là cờ **toàn deployment**
  nên hiện cả trên trang có hạng từng query.

## Kiến trúc mới

```
types.ts        kiểu dữ liệu API (mở rộng theo payload thật)
routes.ts       khu vực, view, URL ↔ state, ánh xạ 12 tab cũ, validate + báo lỗi
requests.ts     RequestRunner: token + context key; AbortController là tối ưu chi phí
useRegion.ts    hook: mỗi vùng dữ liệu có Region{status,data,error,fresh}
catalog.ts      4 cờ capability riêng + lý do; gap/công thức; provenance theo schema; nhóm measurement
format.ts       nĐơn vị: nDCG (phân số) / recall (%) / gap / pp; thiếu = "—"
analysis.ts     join theo run, cohort = giao, unscored tách riêng, preset/lọc/phân trang thuần
data.ts         useWorkbenchData: mọi nguồn là một region có khoá riêng
ui.tsx          Badge/Card/EmptyState(5 loại)/ErrorState/Detail/Drawer/SortHeader/SaveFeedback
App.tsx         shell: URL → context → area, sidebar, topbar, contextbar
Overview|Experiments|Evaluation|Samples|Failures|Notebook Page.tsx
QueryInspector.tsx  inspector + so sánh cùng scope (2 region có khoá)
inspector.ts    đọc điều kiện + giải thích hạng
style.css       design token + primitive, thay cho 6 khối override tích luỹ
```

Đã **xoá** `CompositionView.tsx`, `BenchmarkView.tsx`, `components.tsx` (không còn nơi nào import).
Sau khi adopt contract backend, xoá tiếp hai module **0 importer**:
`staleness.ts` (luật đã chuyển hẳn vào `requests.ts`) và `protocols.ts` (danh sách protocol chuyển
vào `routes.ts` với tên `KNOWN_PROTOCOL_IDS`, dùng chung cho parser và cross-check registry).

## Contract backend đã adopt

`capabilities` (4 cờ) + `capability_detail.status`/`.reason`; `measurement.sample_count` /
`denominator` / `scored_count` / `unscored_query_count` / `unscored_by_kind` (unscored tách khỏi
lỗi truy hồi); `query_state` (`unscored` / `retrieval_failure` / `hit`) làm bộ lọc server-side;
`transfer` (PTHA = `TRANSFER · CIRR→FashionIQ`); `duplicate_status` / `duplicate_attempt_count`
thay heuristic "thiếu knobs ⇒ trùng"; `provenance_source` cho artifact import (opt-in
`include_imported=true`, có nhãn nguồn); `top_results[].gallery_position` và `score: null` → `—`;
`target_rank` ngoài cửa sổ ứng viên được **nêu rõ**, không clamp. Dashboard đọc
`/api/composition/capabilities` và `/api/evidence/summary` thay cho mọi literal.

## Thay đổi hành vi quan trọng

| Trước | Sau |
|---|---|
| 12 tab ngang + 2 fieldset toàn cục chiếm ~470px đầu trang | 6 khu vực + sidebar 216px; nội dung ở ~190px |
| State chỉ trong RAM, refresh mất context | URL là nguồn sự thật; Back/Forward; deep link; giá trị sai được báo |
| Response cũ có thể ghi đè context mới | token + context key; response sai bị loại; lỗi thuộc đúng region |
| Dashboard "0 experiment runs" dù có kết quả CIRCUS | đếm từ 3 nguồn, nêu nguồn dưới mỗi số |
| Banner "aggregate-only" cho mọi case | thông báo theo **lựa chọn**, ẩn được |
| Cờ `aggregate-only` duy nhất | `aggregate_metrics` / `per_query_target_ranks` / `candidate_topk` / `image_availability`, mỗi cờ có lý do |
| Top-K toàn cục ngầm điều khiển outcome | `@10 cố định theo protocol`; Top-K chỉ = số ứng viên hiển thị; @K đồng thuận là điều khiển thật và ghi rõ |
| "1 unique measurement"; thiếu knob ⇒ coi là trùng | nhóm theo measurement identity; thiếu ⇒ **"undetermined"** |
| "Worst" đẩy rank null lên đầu | unscored có bucket, bộ lọc và bộ đếm riêng; luôn xếp cuối |
| Gap có thể in 0 khi thiếu mẫu số | `unavailable` + lý do; giữ gap âm; từ chối in khi lệch công thức |
| "Save filtered set" = trang đang xem | lưu **toàn bộ** tập đã lọc; nút ghi số lượng; có saving/saved/error |
| Annotation là heading rỗng khi chưa chọn query | hướng dẫn + 2 hành động thật; hiển thị (query, protocol) và khoá lưu |
| Hàng bảng cao hàng trăm px vì hash/path | hàng một dòng; chi tiết trong `Detail`/drawer; số căn phải, tabular-nums, đơn vị |
| Mobile: bảng pan cả trang | cột lặp định danh ẩn dưới 860px; cuộn ngang chỉ trong `.table-wrap`; sidebar thành drawer |

## Ràng buộc kỹ thuật đã giữ

- Không thêm framework test JS: module thuần được biên dịch bằng `tsc` có sẵn và chạy bằng `node`
  từ pytest (`test_frontend_support.py`), cùng cách `test_frontend_staleness.py` đã làm.
- Không sửa `workbench/config/workbench.env`, không chạm backend/scripts/registry.
- Mọi test đều chạy **mã đang ship** (`workbench/frontend/*.ts`), không phải bản sao.
