# QA baseline — browser harness (CIR Failure Analysis Workbench)

Thư mục bằng chứng: `workbench/docs/reviews/20261108T115856-qa/`
Harness: `workbench/scripts/browser_qa/run_flows.py` (+ `harness.py`, `dump_report.py`)
UI: `http://127.0.0.1:5173` · API: `http://127.0.0.1:8000`
Viewport: 1440×900, 1920×1080, 390×844

## 1. Harness làm gì

`run_flows.py` mở Chromium thật (Playwright 1.64.1, managed chromium trong
`workbench/envs/ctl312h/.playwright-browsers`) trỏ vào app đang serve, chạy 8 flow độc lập, và ghi
`qa_report.json` **sau từng flow** — nên nếu frontend/backend bị agent khác build lại giữa lượt chạy,
bằng chứng đã thu không mất.

Mỗi flow ghi: `status` ∈ `pass | fail | blocked | selector_miss`, từng assertion kèm giá trị kỳ
vọng vs thực tế, console error, page error, request thất bại, response HTTP ≥ 400, số pixel overflow
ngang, và đường dẫn screenshot.

**Nguyên tắc chống "pass giả":**

- Mọi assertion so với **ground truth không phải UI**: HTTP API và file JSON trên đĩa. Không có
  assertion nào chỉ so UI với chính UI.
- "Không có console error" **chỉ được ghi làm bằng chứng**, không bao giờ là tiêu chí pass. Một flow
  có thể `fail` với 0 console error (và đúng là vậy — xem flow b).
- Request do chính harness chèn (abort / giữ lại để đảo thứ tự) được khai báo trước và liệt kê riêng
  ở `injected_failures`, để lỗi giả không lẫn vào lỗi thật.
- Flow b **fail nếu không chặn được response nào**: một mạng nhanh không chứng minh được gì.
- Ảnh chụp là app thật đang serve; không có dữ liệu nào được bịa.

### Chọn selector bền với redesign

`harness.py:SELECTORS` là registry: mỗi tên logic (`comp.pager`, `annot.note`, …) có danh sách
chiến lược theo thứ tự **`data-testid` → role/accessible-name → CSS/text cũ**. Tab 12 cũ và 6 khu
vực mới đều có alias (`AREAS`). Không match được chiến lược nào → sinh **category riêng
`selector_miss`** trong report và flow vẫn chạy tiếp, không crash.

### Cách chạy

```bash
# đầy đủ, có baseline before-*.png
workbench/envs/ctl312h/bin/python workbench/scripts/browser_qa/run_flows.py \
    --baseline --out workbench/docs/reviews/<timestamp>-qa

# một flow, một viewport (lặp nhanh trong lúc UI đang đổi)
workbench/envs/ctl312h/bin/python workbench/scripts/browser_qa/run_flows.py \
    --flows b --viewports 1440x900 --out /tmp/qa-b

# đọc lại report
workbench/envs/ctl312h/bin/python workbench/scripts/browser_qa/dump_report.py \
    <out>/qa_report.json [flow] [viewport]
```

Tuỳ chọn khác: `--ui-url`, `--api-url`, `--headed`, `--read-only` (bỏ flow f/g vì chúng ghi
cohort/annotation), `--timeout`.

## 2. Kết quả hiện tại

Lượt đầy đủ cuối cùng (`qa_report.json`, 24 kết quả): **pass 12 · fail 6 · blocked 6 ·
selector_miss 0**, console error **0**, page error **0**, HTTP ≥ 400 **0**. Sáu `fail` là **hai lỗi
thật × ba viewport** (mục 3); sáu `blocked` là ba viewport của flow a và flow g (giới hạn dữ liệu,
mục 2).

| Flow | Nội dung | 1440×900 | 1920×1080 | 390×844 |
|---|---|---|---|---|
| a | end-to-end (2 run → disagreement → ảnh/rank/Top-K → note → cohort → reload) | blocked | blocked | blocked |
| b | race (đảo thứ tự response) | **fail — lỗi thật** | **fail — lỗi thật** | **fail — lỗi thật** |
| c | degraded states (8 trường hợp) | pass | pass | pass |
| d | navigation (refresh, deep link 6 khu, Back/Forward, URL sai) | pass | pass | pass |
| e | artifact cross-check (JSON → API → DOM) | **pass** | **pass** | **pass** |
| f | cohort: tập lưu = toàn bộ scope công bố | **pass** | **pass** | **pass** |
| g | annotation persistence | blocked (backend từ chối, UI nói rõ) | blocked | blocked |
| h | layout (overflow, target <24px, ảnh méo, focus, sticky) | fail — 1 lỗi thật | fail — 1 lỗi thật | fail — 1 lỗi thật |

Flow h chỉ còn **một** loại lỗi (click target < 24 px), lặp trên 3 bề mặt; overflow, ảnh méo, focus
và sticky đều pass ở cả 3 viewport.

### Những gì **không thể** pass vì dữ liệu/UI chưa có

Đây là baseline trung thực, không phải lỗi của harness:

1. **Flow a — nhánh index per-query.** `/api/runs` trả **0 run**. Đường "chọn 2 mô hình tương thích →
   query bất đồng → Top-K → cohort" **không thể chạy** vì index DuckDB
   (`workbench/artifacts/workbench.duckdb`) có bảng `runs/queries/top_results` đều rỗng: không có
   artifact per-query nào được export trên máy này. Flow ghi `a.two_per_query_runs` = FAIL và
   `blocked` cho 8 bước phụ thuộc — đúng như sự thật dữ liệu, không giả vờ.
2. **Top-K ứng viên.** Run CIRCUS release (`315dd590eeb0` = ptha_mtst) trả `top_results: []` cho mọi
   query. UI hiện `data-testid="candidates-unavailable"` và nói rõ đây là giới hạn dữ liệu. Flow
   assert `a.topk_honest` = PASS: **API không có ứng viên thì DOM không được render lưới ứng viên
   nào** — và đúng là 0 lưới.
3. **Flow g — annotation.** Backend từ chối lưu note cho query id release CIRCUS (`7:603`) vì
   `/api/annotations` chỉ nhận `category:index:reference:target`. UI nói rõ lý do bằng tiếng Việt.
   Harness assert "hoặc lưu được, hoặc từ chối và **nói rõ**"; trường hợp thứ hai → `blocked`, không
   phải `pass`. `g.annotation_bound_after_reload` bị block vì không có gì để đọc lại.

### Nhánh thay thế đã chạy được (app thật, dữ liệu thật)

Vì index rỗng, flow a chạy nhánh **CIRCUS composition inspector** — bề mặt query duy nhất máy này
serve được — và tất cả đều PASS: query `7:603` mở đúng, tiêu đề đúng, 2 ảnh (reference + target)
đúng src, 3 rank theo điều kiện render đúng, cohort scope = 4069 = đúng scope `sf` công bố.

## 3. Lỗi thật mà harness tìm được

### (1) Race: response đến muộn của scope bị bỏ làm mất luôn run đang chọn — **flow b**

Kịch bản: đang xem `subset=sf`, chuyển sang `validated` rồi bấm về `sf`; response `validated` bị
giữ lại và **thả sau** response `sf`. Kết quả đo được:

```
label hiển thị:            "sf"        (đúng)
URL sau khi response muộn: ?area=samples&view=explorer&protocol=…&subset=sf
                           ^ mất hẳn runs=c:<run_id>
pager:                     (không có)
danh sách query:           0 dòng
```

Nghĩa là câu trả lời của scope **đã bị bỏ** vẫn ghi đè được lựa chọn run, và UI hiển thị `sf` với
một bảng rỗng. Đây đúng là lỗi "nhãn nói SF nhưng dữ liệu không phải SF" mà harness phải bắt — và
nó bắt được, với 0 console error. Assertion: `b.selection_survives_late_response`,
`b.data_matches_label`, `b.first_row_matches_label`.

Phía điều kiện (condition) **không** bị: `b.condition_data_matches_label` = PASS, response
`multimodal` muộn không ghi đè `image_only_empty_text`.

### (2) Target bấm < 24×24 px — **flow h**

Cả ba viewport, ba bề mặt:

- **Khu Thí nghiệm**: 17/35–36 phần tử — `<input type="checkbox" aria-label="Chọn <model>…">` chọn
  run, hit area đo được **15×15 px**.
- **Khu Đánh giá**: 5/26–27 phần tử — cùng loại checkbox.
- **Inspector query**: 20/45–46 — các nút id query (`data-testid="query-id"`) cao **17 px**, có nút
  rộng 25 px.

Ngưỡng 24×24 px là mức AA của WCAG 2.2 (2.5.8). Bộ đo lấy hit area của `<label>` bao checkbox khi
label đủ lớn — chỉ khi **cả** label cũng nhỏ mới báo, nên 15×15 ở đây là kích thước thật của vùng
bấm được.

### (3) Mọi thứ khác đều xanh (và đây là bằng chứng, không phải lời hứa)

- `h.*.no_page_overflow` = PASS ở cả 3 viewport. Bộ đo **phân biệt** overflow thật của trang với nội
  dung nằm trong container `overflow-x:auto` (bảng rộng cuộn được không bị tính là lỗi).
- `h.*.images_not_distorted` = PASS: 0 ảnh bị méo do `object-fit:fill` lệch tỉ lệ > 2%; app dùng
  `cover` nên cắt ảnh chứ không kéo méo. `ratioDrift` = `null` cho mọi ảnh.
- `h.focus_visible` = PASS: 30 tab stop, tất cả có outline hoặc box-shadow nhìn thấy được.
- `h.*.sticky_no_occlusion` = PASS: không text nào bị phần tử sticky/fixed che (sidebar mobile nằm
  ngoài viewport, không che nội dung).
- Flow e: JSON artifact → `/api/.../queries` → DOM khớp **tuyệt đối** trên 4 query cố định, gồm id,
  reference/target image, cả 3 rank theo điều kiện, thứ tự phân trang, mẫu số pager (4,069) và ô
  MM nDCG so với `summary/sf.json`.
- Flow d: cả 6 khu vực deep-link được qua `?area=`, cả 6 tab cũ qua `?page=`, Back/Forward đúng,
  URL sai (`area`, `view`, `subset`, `page`) được **nêu tên và giải thích** trong thẻ "Giá trị URL
  không hợp lệ" chứ không im lặng đổi.

## 4. Ảnh chụp

Baseline (app thật, trước/sau redesign đều dùng cùng tên để so):

```
before-dashboard-<viewport>.png
before-evaluation-runs-<viewport>.png
before-composition-table-<viewport>.png
before-query-inspector-<viewport>.png
```

Bằng chứng từng flow nằm trong `screens/`: `<flow>-<viewport>-<tên>.png` — ví dụ
`b-1440x900-race-subset.png`, `c-1440x900-error-state.png`, `e-1440x900-artifact-7-603.png`,
`h-390x844-focus.png`.

## 5. `data-testid` cần bổ sung

Chi tiết ở [`testid-requests.md`](testid-requests.md). Phần lớn hook đã có; các yêu cầu còn lại:

1. `data-testid="error-banner"` / `"error-dismiss"` — hiện guard error là `.error` ở shell cũ và
   `notice-dismiss` ở shell mới, nên "có lỗi đang hiện" phụ thuộc shell nào render.
2. `data-testid="query-image"` + `data-image-id` trên ảnh trong inspector — hiện phải đoán qua `src`.
3. `data-testid="area-notebook"` trên phần tử ngoài của khu Sổ nghiên cứu (5 khu kia đã có).
4. `data-status="idle|saving|saved|error"` trên feedback của save (cohort/annotation) — hiện harness
   đọc `role="alert"` rồi phải phân tích câu tiếng Việt để biết lưu được hay bị từ chối.
5. `data-scope-count="<n>"` trên `cohort-save` — để assert "lưu đúng toàn bộ scope" trực tiếp thay vì
   suy từ nhãn nút.

## 6. Phải chạy lại sau khi redesign hoàn tất

Danh sách chính xác — mọi mục dưới đây đổi theo UI, không đổi theo dữ liệu:

1. **Toàn bộ 24 kết quả** (8 flow × 3 viewport) bằng một lượt đầy đủ:
   `… run_flows.py --baseline --out workbench/docs/reviews/<timestamp>-qa`.
2. **`before-*.png`** — 12 ảnh phải chụp lại trên UI cuối cùng; bộ hiện tại là UI giữa chừng
   (redesign đã land một phần).
3. **Flow b sau khi sửa race** — assertion `b.selection_survives_late_response` phải chuyển sang PASS;
   nếu vẫn fail thì race chưa được sửa, không phải harness sai.
4. **Flow h `click_targets`** — sau khi checkbox có hit area ≥ 24 px.
5. **Flow a nhánh index** — chỉ chạy được khi `/api/runs` có ≥ 2 run per-query; lúc đó kiểm lại:
   chọn 2 run, mở query bất đồng, ảnh/rank/Top-K, ghi note, lưu cohort, reload và mở lại thấy đúng
   dữ liệu cũ.
6. **Flow g** — sau khi `/api/annotations` nhận query id release CIRCUS (hoặc sau khi index có query
   dạng `category:index:reference:target`).
7. **Kiểm `selector_miss`** — lượt cuối phải có `selector_miss = 0`; mỗi mục còn lại chỉ ra một hook
   cần thêm (mục 5).
8. **`testid-requests.md`** — đánh dấu những hook đã được thêm.

## 7. Giới hạn của harness (nói rõ để không đọc quá sức report)

- Harness **không** kiểm khoa học của số liệu; nó kiểm JSON → API → DOM có nhất quán không.
- Flow h đo **layout tại đúng 3 viewport** đã cấu hình, không phải mọi kích thước.
- Flow b chứng minh hành vi dưới **một** kiểu đảo thứ tự (response cũ đến sau). Nó không chứng minh
  vắng mặt của mọi race, và không thay thế test đơn vị cho state machine.
- Mọi trạng thái `blocked` đều kèm lý do dữ liệu cụ thể; đọc `summary.blocked_items` trong
  `qa_report.json` để biết chính xác cái gì đang thiếu.
