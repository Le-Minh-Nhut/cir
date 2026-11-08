# UI_ACCEPTANCE — CIR Failure Analysis Workbench (frontend pass)

Thời điểm: 2026-11-08 · nhánh `research/cir-failure-workbench` · không commit (giám đốc commit sau).
Bằng chứng ảnh: `workbench/docs/reviews/20261108T042931Z/shots/`.
Interpreter dùng để chạy test/Playwright: `workbench/envs/ctl312h/bin/python`.

## 0. Tóm tắt trung thực

| Hạng mục | Kết quả |
|---|---|
| `npm run build --prefix workbench/frontend` | **exit 0** (`tsc -b && vite build`, 35 module, bundle 315.57 kB) |
| Test frontend thuần (do tôi viết) | **51 passed** trên 6 file, gồm `test_frontend_staleness.py` (đã được giao lại và sửa) |
| `browser_qa` flow **b** (race) | **pass** @ 1440×900, 1920×1080, 390×844 — 0 selector miss, 0 console error |
| `browser_qa` flow **h** (layout) | **pass** @ 1440×900, 1920×1080, 390×844 — 0 click target < 24×24 |
| `browser_qa` flow **g** (annotations) | 1 assertion đỏ **do probe của harness**, không phải UI — xem `QA_flow_g_textarea_probe.md` |
| `test_frontend_staleness.py` | **7 passed** — đã repoint sang `requests.ts`/`useRegion.ts`, xem §7 |
| Lỗi console/page error | **0** trên 4 màn hình × 2 viewport sau khi adopt contract |
| Stack chạy lại | `cir-workbench` (pid ghi trong service), API `200`, FE `200` |
| Ảnh before/after | 33 ảnh, 3 viewport, 6 màn hình sau + 4 màn hình trước |
| Overflow (đo `scrollWidth`) | **18/18 tổ hợp viewport×màn hình sạch** (`capture-log.txt`) |
| Lỗi console/page error | **0** |

## 1. Ảnh chụp — trước và sau

`before-*` chụp trên bundle CŨ trước khi sửa (12 tab, không URL state).
`after-*` chụp trên bundle MỚI (6 khu vực, URL là nguồn sự thật).

| Màn hình | 1440×900 | 1920×1080 | 390×844 |
|---|---|---|---|
| Tổng quan | `before-dashboard-1440x900.png` → `after-overview-1440x900.png` | `before-dashboard-1920x1080.png` → `after-overview-1920x1080.png` | `before-dashboard-390x844.png` → `after-overview-390x844.png` |
| Thí nghiệm (Evaluation Runs cũ) | `before-evaluation_runs-1440x900.png` → `after-experiments-1440x900.png` | `before-evaluation_runs-1920x1080.png` → `after-experiments-1920x1080.png` | `before-evaluation_runs-390x844.png` → `after-experiments-390x844.png` |
| Đánh giá (Composition cũ) | `before-composition-1440x900.png` → `after-evaluation-1440x900.png` | `before-composition-1920x1080.png` → `after-evaluation-1920x1080.png` | `before-composition-390x844.png` → `after-evaluation-390x844.png` |
| Inspector query | `before-inspector-1440x900.png` → `after-inspector-1440x900.png` | `before-inspector-1920x1080.png` → `after-inspector-1920x1080.png` | `before-inspector-390x844.png` → `after-inspector-390x844.png` |
| Phân tích lỗi (mới) | `after-failures-1440x900.png` | `after-failures-1920x1080.png` | `after-failures-390x844.png` |
| Sổ nghiên cứu (mới) | `after-notebook-1440x900.png` | `after-notebook-1920x1080.png` | `after-notebook-390x844.png` |

## 2. Đánh giá thị giác thật (không phải "CSS đã đổi")

**Trước.** Ở 1440×900 nội dung hữu ích đầu tiên bắt đầu ở **~y=470px**: tiêu đề, 12 tab, rồi hai
`fieldset` (ngưỡng lệch paper + protocol visibility) chiếm gần nửa màn hình trước bất kỳ con số nào.
Banner nâu "AGGREGATE-ONLY … per-query Top-K analysis unavailable" phủ lên mọi trang kể cả trang
Composition đang hiển thị hạng từng query. Bảng Composition ở 390px pan ngang toàn trang (6.29× theo
tỉ lệ ảnh chụp), cột model bị sticky chiếm 156px, phần còn lại đọc như ô trống. Inspector và
"Evaluation Runs" là hai trang khác nhau cho cùng một lượt chạy, không có đường nối.

**Sau.** Nội dung hữu ích đầu tiên ở **~y=190px** (topbar 52 + contextbar 48 + card header). Sidebar
216px tĩnh, header 2 dòng ngắn ("Đánh giá / Chỉ số đo được theo protocol và subset, kèm mẫu số"),
contextbar một dòng (Protocol · Subset · Lượt chọn · tab view). Bảng số căn phải, `tabular-nums`,
đơn vị ghi trong header (`[0,1]`, `%`, `pp`), cột "Đã chấm" luôn kèm mẫu số `4,000 / 4,069 (98.3%)`.
Hàng `habit` FAILED hiện `—` chứ không phải 0, và có dải riêng giải thích lý do runtime mismatch.
Lượt `ptha_mtst` mang nhãn `TRANSFER · CIRR→FashionIQ transfer` nền hổ phách, không đứng như dòng paper.

**Còn chưa tốt — trạng thái sau khi vá trong cùng lượt này:**

1. *Bảng Đánh giá ở 390px tràn 519px* — **đã vá và đã kiểm lại**: nguyên nhân thật không phải bảng
   mà là một token không ngắt được trong thông báo lỗi runtime (`habit: RuntimeError: … [BLOCKED/
   PYTHON_VERSION_MISMATCH]`), làm rộng cả trang. Đã đặt `overflow-wrap: anywhere` cho `.empty-detail`
   và `break-word` cho ô bảng; đo lại `scrollWidth = 390` với 0 phần tử tràn. Ngoài ra: 4 cột lặp định danh (`Nguồn/transfer`,
   `Text nDCG`, `Image nDCG`, `Chi tiết`) ẩn dưới 860px (`.t-col-narrow`), cuộn ngang chỉ còn
   **trong** `.table-wrap`. Ảnh `after-evaluation-390x844.png` đã chụp lại sau bản vá; số
   `scrollWidth` từng viewport ghi trong `capture-log.txt` cùng thư mục.
2. *Inspector bị lồng hai lớp split làm card query chỉ ~200px và hai ảnh xếp dọc* — **đã vá**:
   `.inspector` chuyển thành một luồng dọc (card query full width), hai card phụ nằm dưới trong
   `.inspector-secondary`; danh sách query + inspector dùng `.samples-layout` (list 340px,
   inspector phần còn lại). Ảnh tham chiếu/đích nay rộng ~350px, đọc được chi tiết.
   Lưới ảnh đổi từ 3 cột `1/1` thành
   `repeat(auto-fit, minmax(180px, 1fr))` với `aspect-ratio: 4/5`, nên ảnh tham chiếu/đích rộng
   ~180px+ và giữ đúng tỉ lệ (`object-fit: contain`). Tile ứng viên vẫn vuông (`1/1`).
3. *Cột phải inspector hẹp cố định 320px* — đã vá: `minmax(280px, 26%)`, nên ở 1920px cột phải
   rộng theo tỉ lệ thay vì cố định, cột trái không còn quá rộng.
4. *"Sao chép liên kết" trông như text* — đã vá: thành nút `secondary` có icon và viền; khi
   clipboard không dùng được (quyền/insecure origin) nó **hiện URL trong ô chọn được** thay vì chỉ
   báo lỗi, nên liên kết vẫn lấy được.
5. *Số hạng bị ngắt dòng giữa chừng* (`#70,01` / `6`) do `overflow-wrap` áp lên cả ô số — **đã vá**:
   `td.num` đặt `white-space: nowrap`, chỉ ô chữ mới được ngắt.
6. *Sidebar counts "nhảy" số sau khi tải* — **chấp nhận**: chỉ hiện khi đã có dữ liệu thật; thà
   nhảy còn hơn hiện 0 sai. Nếu muốn hết nhảy: render skeleton cố định chiều rộng.
7. *Bảng dài chưa virtualize* — **chấp nhận**: đã phân trang 20/25 dòng nên không render 4,069 dòng
   cùng lúc; virtualize chỉ cần nếu bỏ phân trang.

## 3. Luồng đã kiểm bằng browser (và cách kiểm)

Ghi chú: mọi kiểm tra dưới đây chạy trên **dữ liệu thật của deployment này**, không phải fixture.

| Luồng | Trạng thái | Bằng chứng |
|---|---|---|
| (a) Overview → chọn 2 mô hình → query bất đồng → ảnh/rank/Top-K → ghi chú → lưu cohort → reload | **Một phần** | `after-*`. Chọn lượt hoạt động; "So sánh trên cùng query" hiện 5 lượt cùng scope với hạng MM thật (70,016 vs 38,423 …). **Chặn thật:** lưu cohort cho CIRCUS release bị API từ chối (`run_not_found`) — xem §5. Ghi chú bị API từ chối vì query id release `7:603` không khớp schema — UI báo rõ lý do thay vì no-op. |
| (b) Chuyển model/subset/filter nhanh, response đảo thứ tự | **Đạt ở tầng logic, có test** | `test_frontend_requests.py::test_the_older_response_is_dropped_when_it_resolves_after_the_newer_one` và `..._newer_rejecting_request...` — response cũ resolve SAU response mới bị loại; request mới reject thì region `error` + `data=null` (không còn dữ liệu context cũ). Ở browser: đổi subset/computed `compositionPage` key theo (run, subset, condition, outcome, category, order, offset, limit, query) nên nhãn và dữ liệu không thể lệch. **Chưa** chạy kịch bản Playwright giữ route để trễ response — xem §7. |
| (c) Lỗi request + retry; filter không khớp; không chọn query; aggregate-only; rank-only; coverage một phần; run lỗi; model bị chặn | **Đạt** | `empty-*` phân biệt 5 loại (`none-selected`, `no-results`, `no-match`, `capability`, `error`) + `ErrorState` có nút `data-testid="error-retry"`. `after-evaluation-*` cho thấy `habit` FAILED + lý do; `before-*` cho thấy run lỗi từng hiện "0.0000". Inspector khi `candidate_topk=false` hiện "Ứng viên chưa được export cho lượt này" + lý do capability. |
| (d) Refresh / deep link / Back / Forward / URL sai | **Đạt** | `test_frontend_routes.py`: 12 tab cũ ánh xạ đúng, round-trip URL, giá trị sai bị **báo** không bị ép. `?page=Composition&comp_run=…` (link cũ) → area=Đánh giá + protocol `circus_fashioniq_release_v1` (nhãn đi theo dữ liệu). `popstate` + `pushState` trong App.tsx. |
| (e) Lưu cohort trên tập > 1 trang | **Chặn** | Nút ghi rõ "Lưu tập đã lọc (N query)" với N là **toàn bộ** tập đã lọc (`filtered.length`, không phải `pageWindow.items.length`), và `data-total` trên pager. Nhưng API cohort từ chối run CIRCUS (§5) nên chưa lưu được thật. |
| (f) Ghi chú lưu + reload không mất/không nhảy query | **Một phần** | Key là `(query_id, protocol_id)`; UI hiển thị cả hai và khoá lưu. Reload: effect đọc lại theo cặp đó. Nhưng API chỉ nhận query id canonical `category:index:ref:target`, nên với id release UI báo lỗi rõ thay vì lưu sai (§5). |
| (g) Overflow, ảnh, click target, sticky, focus | **Đạt một phần** | `scrollWidth == innerWidth` ở 1440 và 1920 cho cả 4 màn hình. Focus ring `:focus-visible` toàn cục. Nút/ô nhập `min-height: 32px` (38px ở ≤860px) ≥ 24px. Ảnh `object-fit: contain` (không méo). 390px Đánh giá còn tràn — đã vá, cần chụp lại. |
| (h) Dữ liệu dài: cuộn, đổi filter, phân trang | **Đạt** | 4,069 query/run; danh sách phân trang 20, `data-total`/`data-shown`; phân tích lỗi lấy **toàn bộ** universe (`limit=20000`) rồi lọc/phân trang client-side với `total` là toàn tập. Không virtualize — 4,069 dòng không render cùng lúc vì đã phân trang. |

## 4. Những gì đã sửa (ánh xạ yêu cầu → mã)

1. **Stale response** — `requests.ts` (`RequestRunner`: token + context key, AbortController như tối ưu
   chi phí) + `useRegion.ts` (region `idle/loading/ready/error`, `data` bị xoá khi đổi context).
   Mọi fetch dữ liệu đi qua đây; test cấm `.then(...setState)` trần trong 8 file trang.
2. **Một context thống nhất** — `catalog.ts`: 4 cờ riêng (`aggregate_metrics`,
   `per_query_target_ranks`, `candidate_topk`, `image_availability`), mỗi cờ có `reason` để nút bị
   vô hiệu hoá nói được lý do. Dashboard đếm từ cả 3 nguồn (`/api/composition/runs`,
   `/api/benchmark-runs`, `/api/runs`) nên không còn "0 experiments" khi có kết quả CIRCUS.
   Banner toàn cục kiểu "aggregate-only" thay bằng thông báo **theo lựa chọn**.
3. **URL/route** — `routes.ts` + App.tsx: `area/view/protocol/subset/run/runs/query/condition/
   outcome/category/order/offset/q/k/preset`, 12 tab cũ ánh xạ, giá trị sai được liệt kê kèm giá trị
   hợp lệ. Deep link cũ `?page=…&comp_*` vẫn mở đúng chỗ.
4. **Nhãn & thống kê** — bỏ "1 unique measurement"; `groupMeasurements` gom theo (model,
   checkpoint digest, protocol, dataset, subset, noise) và nói **"undetermined"** khi thiếu thành
   phần (thiếu `evaluator_environment_knobs` KHÔNG còn là bằng chứng trùng lặp).
   `compositionGap` giữ gap âm, trả `unavailable` + lý do khi thiếu mẫu số, và **từ chối** công bố
   khi số ghi sẵn lệch công thức. `scoredCounts` tách `unscored` khỏi thất bại truy hồi;
   `analysis.ts` đẩy unscored xuống cuối cho mọi kiểu xếp hạng. `transferLabel` dán nhãn TRANSFER.
   `format.ts` tách ba đơn vị (nDCG / % / pp) và in `—` cho giá trị thiếu.
5. **IA 6 khu vực** — Tổng quan, Thí nghiệm (+Mô hình, Hàng đợi), Đánh giá (+Benchmark), Khám phá mẫu
   (+So sánh), Phân tích lỗi, Sổ nghiên cứu (+Cohort). Slider ngưỡng paper rời khỏi thanh toàn cục.
6. **Top-K** — nhãn `@10 (cố định theo protocol)`; Top-K trong inspector ghi rõ "số ứng viên HIỂN
   THỊ, không phải ngưỡng phân loại"; cutoff đồng thuận @K là điều khiển thật của `/api/analysis`
   và được ghi cạnh cột.

## 5. Giới hạn chưa gỡ (đúng nguyên nhân)

- **Cohort cho CIRCUS release:** `PUT /api/cohorts` gọi `validate_analysis_selection`, chỉ biết run
  trong chỉ mục per-query → `run_not_found` cho mọi run `circus_fashioniq_release_v1`. UI chặn trước
  và nói lý do; yêu cầu backend ở `frontend-backend-requests.md` (#1).
- **Annotation cho query id release:** `ANNOTATION_QUERY_ID` bắt buộc `category:index:ref:target`;
  release dùng `7:603`. UI báo lỗi rõ, không gửi request chắc chắn 400. Yêu cầu #2.
- **Candidate Top-K:** `top_results` rỗng cho mọi query composition → UI hiện trạng thái
  "chưa export", không bịa ảnh. Khi backend trả `top_results`, `QueryInspector` hiển thị ngay
  (đọc `query.top_results`, có nhánh `k` và đánh dấu ảnh đích).
- **Category:** release không gán `category` (mọi query `null`) nên lọc theo danh mục trả 0; UI nói
  rõ đây là dữ liệu chứ không phải lỗi lọc.

## 6. Test đã viết

`workbench/tests/` (đều chạy module TS đang ship qua `node`, không framework mới):

| File | Nội dung |
|---|---|
| `test_frontend_support.py` | helper biên dịch `tsc` + chạy ES module (không phải test) |
| `test_frontend_requests.py` | 7 test: đảo thứ tự response, request mới reject, out-of-order cùng key, abort không thành lỗi, counter theo family, key không va chạm, wiring |
| `test_frontend_routes.py` | 7 test: round-trip, 12 tab cũ, link `comp_*`, giá trị sai bị báo, mặc định, App đọc/ghi URL |
| `test_frontend_catalog.py` | 9 test: 4 cờ capability, run lỗi, aggregate-only, gap (âm/thiếu mẫu số/lệch công thức), scored/unscored, provenance theo schema không theo tên, nhóm measurement "undetermined", cutoff cố định, TRANSFER |
| `test_frontend_format.py` | 5 test: ba đơn vị, gap âm, thiếu = `—`, rank 1-based, mẫu số + coverage |
| `test_frontend_analysis.py` | 5 test: unscored không phải worst, cohort/partial, rank theo run+điều kiện, cutoff đồng thuận, phân trang giữ `total` |

## 7. Việc còn lại (nêu chính xác, không giấu)

1. ~~`test_frontend_staleness.py` đỏ~~ — **đã sửa** (file được giao lại). Chọn hướng **(b)**:
   `staleness.ts` có **0 importer** sau khi `App.tsx` cũ bị thay bằng các region có khoá, nên nó là
   dead code bị một test pin dead code — tệ hơn là không có cả hai. Đã xoá `staleness.ts` và
   `protocols.ts` (cũng 0 importer), rồi **repoint toàn bộ bảng transition** sang primitive đang
   chạy thật (`requests.ts`: `contextKey`/`requestKey`/`openRequest`/`isCurrent`), giữ nguyên 10 dòng
   transition và thêm 3 assert chứng minh `requestKey` uỷ quyền cho `contextKey` (một quy tắc
   separator duy nhất). Assertion "guard phải được dùng thật" nay trỏ vào `useRegion.ts` +
   `useApp`, và cấm mọi `.then(...setState)` trần trong 8 file trang. Các dòng cơ chế (thứ tự token,
   abort không thành lỗi, counter theo family, response trùng key) nằm ở
   `test_frontend_requests.py` nên **không mất luật nào**. Danh sách protocol registry chuyển vào
   `routes.ts` (`KNOWN_PROTOCOL_IDS`) để cross-check registry vẫn còn đúng một chỗ. Kiểm tra
   "bundle mới hơn source" nay dùng literal do UI hiện tại render (tên định danh bị minify nên
   không dùng được).
2. **Đã adopt contract backend** (`frontend-backend-requests.md`): 4 cờ + `capability_detail.status`,
   `measurement.unscored_query_count`, `query_state`, `transfer`, `duplicate_status`,
   `provenance_source` (opt-in `include_imported=true`), `gallery_position`, `score: null` → `—`,
   và dashboard đọc `/api/composition/capabilities` + `/api/evidence/summary` thay cho literal.
3. **Hai lỗi backend phát hiện qua payload thật** (đã ghi vào `frontend-backend-requests.md` §2–§3):
   `measurement.scored_count` là tổng toàn lượt trong khi `sample_count`/`denominator` theo subset →
   từng render `5,884 / 4,069 (144.6%)`. UI nay cross-check và fallback về bucket theo điều kiện,
   có test hồi quy. Và `include_imported=true` **không** trả row imported nào trên máy này (artifact
   audit 19 MB có trên đĩa nhưng chưa được materialise thành summary), nên **Top-K vẫn chưa mở
   khoá** — inspector nói đúng "chưa export" thay vì bịa ảnh.
2. ~~Chụp lại 390×844~~ — **đã làm**, và bản vá thật là wrap token lỗi (mục 2.1), không phải ẩn cột.
   Sau khi adopt contract: 8/8 tổ hợp (4 màn hình × 1440/390) `scrollWidth == innerWidth`, 0 lỗi console.
3. Kịch bản Playwright route-delay cho luồng (b): logic đã chứng minh bằng test node (đảo thứ tự
   response, request mới reject). Bản browser cần dựng lại cho IA mới; `workbench/scripts/browser_qa/`
   do agent khác sở hữu và selector registry của họ đã chờ đúng các `data-testid` tôi đã thêm
   (`area-*`, `run-option-*`, `composition-table`, `pager-total`, `query-id`, `rank-*`, `empty-state`,
   `error-retry`, `notice`, `annotation-note`, `cohort-save`, `analysis-row`, `query-title`, `topk`).
4. ~~Ảnh inspector còn nhỏ~~ — **đã làm** (luồng dọc + ảnh ~350px).
5. Service `cir-workbench` từng chết (exit 130) khi chạy Playwright đồng thời với job nặng; đã khởi
   động lại, API/FE trả 200. Capture nên chạy tuần tự.
6. **Chưa kiểm**: luồng (e) lưu cohort thật trên >1 trang, vì API chặn run CIRCUS (§5). Khi backend
   request #1 xong, chỉ cần chạy lại `FailuresPage` — phần đếm toàn tập đã có
   (`filtered.length`, không phải `pageWindow.items.length`).

## 8. Vòng sửa theo phản hồi harness thật (browser_qa)

Hai lỗi harness bắt được, đều là lỗi của tôi, đã sửa và **kiểm lại bằng cách chạy harness**:

### 8.1 `race-drops-selection` — late response xoá mất selection và `runs=` khỏi URL

Nguyên nhân thật không nằm ở guard dữ liệu (guard đúng: payload Validated bị bỏ). Nó nằm ở
`changeSubset` trong `App.tsx`: đổi subset thì xoá `runs: []`. Nhưng một lượt composition có **cùng
`run_id`** ở full/sf/validated (mỗi subset một dòng summary), nên selection vẫn hợp lệ sau khi đổi —
xoá nó làm nhãn ghi "SF" trên pager rỗng và URL mất `runs=`.

Sửa: đổi subset **giữ** selection (chỉ xoá query đang xem). Việc loại selection không còn hợp lệ do
**một** chỗ làm, đọc từ catalog, và **chỉ khi catalog `ready`** — nên một response đang bay hoặc lỗi
không thể trông giống "không có lượt nào". Vì `useRegion` đã bỏ response cũ trước khi nó chạm
catalog, thứ tự đảo ngược không còn ảnh hưởng tới selection.

Thêm test `test_frontend_requests.py::test_a_late_response_cannot_clobber_the_selection_or_the_url`
+ `::test_the_shell_only_clears_a_selection_from_a_ready_catalog` — đúng luật đã thiếu: suite cũ chỉ
assert payload, không assert selection/URL.

Kết quả harness: `b 1440x900=pass 1920x1080=pass 390x844=pass`;
`b.selection_survives_late_response`, `b.data_matches_label`, `b.first_row_matches_label` đều PASS.

### 8.2 `small-click-targets` — checkbox 15×15, nút query-id cao 17px

Sửa bằng cách **tăng hộp bấm**, không thu nhỏ glyph: `input[type=checkbox]` → 24×24 (glyph vẫn 15px,
phần dư là padding), `.rowbtn` (query id) → `min-height: 24px` + padding, `.badge` dùng làm nhãn →
`min-height: 24px`.

Đo lại bằng probe riêng (không phải giả định): **0 phần tử < 24×24** trên experiments / evaluation /
inspector ở cả 1440×900 và 390×844. Harness: `h` pass ở cả ba viewport.

### 8.3 Annotation: bỏ validator trùng lặp trong client

`NotebookPage.tsx` giữ bản sao regex `^(dress|shirt|toptee):\d+:[^:]+:[^:]+$` và **return trước khi
gọi fetch**, nên flow vẫn "blocked" dù backend đã nhận `7:603`. Đã xoá: request luôn được gửi, lỗi
có tên của server (`invalid_query_id`, `query_id_protocol_mismatch`, `unknown_query_id`,
`unknown_dataset`, `unknown_protocol`) hiện nguyên văn dạng `error: message`.

Kiểm live: `PUT /api/annotations` với `7:603` + `circus_fashioniq_release_v1` → **200**, đọc lại sau
reload → textarea có đúng note.

### 8.4 `g.annotation_note_restored` — probe của harness đọc sai thuộc tính

Assertion này dùng `H.texts(...)` → `all_inner_texts()` → `innerText`. Với `<textarea>`, **Chromium
luôn trả `innerText` rỗng** dù nội dung nằm trong `value`. Đã kiểm bằng thực nghiệm (textarea có text
child, có `value` attr, và có `defaultValue` do JS gán — cả ba đều `innerText: ''`, `value` đúng),
nên **không thay đổi UI nào có thể làm assertion này xanh**. Bằng chứng + bản vá một dòng cho harness
nằm ở `QA_flow_g_textarea_probe.md`. Phần còn lại của flow g **pass** (đã lưu, đã bind đúng query,
đúng protocol, 0 HTTP ≥ 400).

### 8.5 `data-testid` đã thêm (đã kiểm trong DOM, không phải chỉ trong source)

| hook | chỗ nào | đã kiểm |
|---|---|---|
| `error-banner` (+ `data-notice-kind`) | banner bằng chứng/lựa chọn ở shell | 1 phần tử |
| `error-dismiss` | nút ẩn banner | 1 phần tử |
| `query-image` + `data-image-id` | mọi ảnh trong inspector (kể cả ảnh thiếu) | 2 ảnh, `B00BJ1NFNI.jpg`, `B00936I6J4.jpg` |
| `area-notebook` | cả 3 nhánh của Sổ nghiên cứu (có query / chưa có query / tab cohort) | 1 phần tử |
| `save-feedback` + `data-status` + `data-count` | feedback lưu ghi chú, luôn render kể cả `idle` | `data-status=idle` |
| `cohort-save` + `data-scope-count` + `data-scope-cohort` | nút lưu cohort | `data-scope-count=4069` |

`area-notebook` lúc đầu chỉ nằm ở nhánh cohort và bị **chính probe xác minh** phát hiện thiếu; đã bọc
cả ba nhánh.
