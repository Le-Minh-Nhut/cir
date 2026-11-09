# Xác minh web đang chạy — LIMN dual-scope diagnostic + toàn bộ inventory mô hình

**Ngày:** 2026-11-09 · **Nhánh:** `research/cir-failure-workbench`
**Dịch vụ:** frontend `http://127.0.0.1:5173`, API `http://127.0.0.1:8000` (pid 2305187, khởi động 20:45:11, không restart trong quá trình đo)
**Bằng chứng thô:** `raw/ui_shots_verify.log` (JSON), `raw/ui/verify-*.png`

Kết luận ngắn: **feature chạy đúng và khớp dữ liệu ở cả 3 viewport**; 12/12 hạng khớp file evidence; inventory 17/17. Có **1 lỗi thật (mức trung bình) trong guard fail-closed** và **1 lỗi cũ mức thấp** (sticky topbar), cả hai ghi ở §6 — không sửa code production.

---

## 1. Bundle đang phục vụ có phải bản mới nhất

| Mục | Giá trị |
|---|---|
| Asset trong `index.html` được phục vụ | `assets/index-KjvBRIjz.js`, `assets/index-DAZAlgSy.css` |
| File tồn tại trong `workbench/frontend/dist/assets/` | `index-KjvBRIjz.js` (351 890 B), `index-DAZAlgSy.css` (26 003 B) |
| MD5 `index.html` phục vụ vs `dist/index.html` | `5212135a8aa553f4c558dbbf602583d2` = `5212135a8aa553f4c558dbbf602583d2` |
| MD5 JS phục vụ vs `dist` | `e076fe9f59c28e4fca141b6ab7338c85` = `e076fe9f59c28e4fca141b6ab7338c85` |
| mtime `dist` vs source mới nhất | dist 20:54:39.54 > `style.css` 20:54:39.04 (mọi `.tsx/.ts/.css` đều cũ hơn) |

Không cần build lại: bản đang phục vụ **chính là** bản trong `dist`, và `dist` mới hơn mọi file source. Bundle có chứa cả chuỗi endpoint `limn-dual-scope` và tiêu đề panel `Ambiguous category — diagnostic comparison`, và CSS đã build có rule `[data-testid="limn-diagnostic-table"]{min-width:560px}` — tức là bản sửa tràn ngang trên mobile cũng đã nằm trong bundle.

## 2. Endpoint chẩn đoán

| Endpoint | Status |
|---|---|
| `GET /api/composition/diagnostics/limn-dual-scope` | **200** |
| `GET /api/composition/diagnostics/limn-dual-scope/candidates?qid=…&scope=…&condition=…` | **200** cho **12/12** tổ hợp (2 query × 2 scope × 3 điều kiện) |
| `GET /api/models` | 200, 17 bản ghi |
| `GET /api/benchmark-runs` (dùng làm readiness) | 200 |

Payload chính (`status: DIAGNOSTIC_COMPLETE`, `available: true`):

- `canonical_status`: `status=PARTIAL`, `scored_count=6001`, `unscored_count=2`,
  `unscored_qids = ["7:3038", "7:3790"]` — **đúng chính xác hai qid chưa chấm**, không thừa không thiếu.
- `scopes = ["shirt", "toptee"]` — đủ **2 scope**.
- `conditions = ["multimodal", "text_only_black_image", "image_only_empty_text"]` — đủ **3 điều kiện**.
- `evidence_files.per_scope` liệt kê đủ 6 file `limn_dual_scope.<scope>.<condition>.jsonl`, và 6 file này có thật trên đĩa.
- 12/12 `target_rank` từ endpoint `/candidates` **bằng đúng** `target_rank` trong payload tổng (không lệch giữa hai đường đọc).

### Guard fail-closed trong `workbench/backend/limn_diagnostic.py`

Các chốt **có thật** trong code:

| Tình huống | Hành vi | Test phủ |
|---|---|---|
| Thiếu file tổng `limn_dual_scope.json` | `available=False`, `status="DIAGNOSTIC_NOT_PUBLISHED"`, `queries={}` — **không** trả so sánh rỗng | `test_absent_evidence_is_a_named_absence_not_an_empty_comparison` |
| Payload chỉ có 1 scope | `DiagnosticEvidenceError: missing scope toptee` | `test_one_scope_alone_is_refused` |
| Thiếu 1 điều kiện trong 1 scope | `DiagnosticEvidenceError: … missing condition(s) [...]` | `test_missing_condition_is_refused` |
| `canonical_status.unscored_qids` ≠ đúng 2 qid | `DiagnosticEvidenceError: … does not declare the two canonical unscored queries` | `test_canonical_unscored_set_must_match_the_two_queries` |
| `artifact_type` sai | `DiagnosticEvidenceError: unexpected artifact type` | `test_wrong_artifact_type_is_refused` |
| `/candidates` với qid/scope/condition lạ | `DiagnosticEvidenceError` | `test_unknown_scope_and_condition_are_refused`, `test_unknown_query_is_refused` |
| `/candidates` khi thiếu file `.jsonl` của scope | `DiagnosticEvidenceError: no per-scope export at …` | *(không có test — xem D1)* |

API map các lỗi này thành HTTP có tên: `503 diagnostic_evidence_inconsistent` cho endpoint tổng, `404 diagnostic_candidates_unavailable` cho `/candidates` — không có nhánh nào trả 200 với dữ liệu nửa vời.

Chạy bộ test của module: `10 passed in 0.13s` (chỉ file `test_limn_dual_scope_diagnostic.py`, không chạy suite lớn).

Tôi **không** sửa file evidence thật. Để kiểm tra guard, tôi dựng một repo root tạm (`tempfile.mkdtemp`) chép `limn_dual_scope.json` vào đúng đường dẫn tương đối, rồi gọi reader với `CIR_REPO_ROOT` trỏ vào đó:

- Xoá key scope `toptee` → **REFUSED**: `missing scope toptee for 7:3038`.
- Xoá 1 điều kiện của `7:3038/toptee` → **REFUSED**: `7:3038/toptee is missing condition(s) ['image_only_empty_text']`.
- Chỉ giữ file tổng, **xoá 0/6 file `.jsonl` per-scope** → **KHÔNG bị từ chối** (xem **D1**).

## 3. Screenshot và số đo 3 viewport

Trang đo: `/?area=samples&view=explorer&protocol=circus_fashioniq_release_v1&subset=sf`.
Harness: `playwright` (Chromium 1248) trong `workbench/envs/ctl312h` với `PLAYWRIGHT_BROWSERS_PATH` trỏ vào `workbench/envs/ctl312h/.playwright-browsers`. Không có harness sẵn nào nhắm panel này (`browser_qa/harness.py` không có selector LIMN), nên tôi viết script đo riêng dùng `data-testid` của panel.

| Viewport | `body.scrollWidth` | `window.innerWidth` | Bằng nhau | Số hàng chẩn đoán render |
|---|---|---|---|---|
| 1920×1080 | **1920** | **1920** | ✅ | **6** (2 query × 3 điều kiện) |
| 2560×1440 | **2560** | **2560** | ✅ | **6** |
| 390×844 | **390** | **390** | ✅ | **6** |

`document.documentElement.scrollWidth` cũng bằng `innerWidth` ở cả ba. Không có lỗi console, không có page error, không có response ≥ 400 ở cả ba viewport.

Ghi chú về bảng trên mobile: ở 390 px, bảng rộng **560 px** nằm trong `.table-wrap` rộng **326 px** → `scrollWidth 560 > clientWidth 326`, tức bảng **cuộn bên trong hộp của nó**, và trang **không** tràn ngang (`body.scrollWidth` vẫn 390, kể cả sau khi cuộn hộp sang phải hết `scrollLeft=234`). Đây là hành vi đúng theo rule `min-width:560px` + `overflow-x:auto`.

Ảnh (đều nằm trong `raw/ui/`):
- Toàn trang: `verify-1920x1080.png` (1920×3400), `verify-2560x1440.png` (2560×3313), `verify-390x844.png` (390×4730)
- Card: `verify-card-1920x1080.png`, `verify-card-2560x1440.png`, `verify-card-390x844.png`
- Bảng cắt ở **độ phân giải gốc**: `verify-table-1920x1080.png` (1548×646), `verify-table-2560x1440.png` (2172×646), `verify-table-390x844.png`
- Mobile khi cuộn hộp: `verify-mobile-table-scrolled-left.png`, `verify-mobile-table-scrolled-right.png`
- Inventory: `verify-models-1920x1080.png` (1920×2863) + các crop `verify-models-*.png`

## 4. Đối chiếu nội dung bảng với file dữ liệu

Đọc `limn_dual_scope.json` và so từng ô với DOM (đọc bằng `innerText` từng ô), rồi **đọc lại ảnh crop bảng ở độ phân giải gốc** (`verify-table-*.png`, cộng thêm zoom 3× ô `#3` và zoom header) trước khi kết luận. Không có lỗi chữ nào: `#3`, `score 0.5704`, `trên 74,381 ảnh`, `cách #1 0.2444` đều đúng như file; header là `SCOPE SHIRT / BASE_ITER0_SHIRT` và `SCOPE TOPTEE / BASE_ITER0_TOPTEE` — **không phải** lỗi gõ.

12 hạng đích (2 query × 2 scope × 3 điều kiện), tất cả **12/12 khớp** ở **cả 3 viewport** (36/36 phép so):

| Query | Điều kiện | shirt | toptee |
|---|---|---|---|
| 7:3038 | multimodal | **#3** (0.5704) | **#43** (0.5678) |
| 7:3038 | text-only | **#88** (0.4221) | **#389** (0.3436) |
| 7:3038 | image-only | **#152** (0.3678) | **#67** (0.4541) |
| 7:3790 | multimodal | **#65** (0.5141) | **#43** (0.6190) |
| 7:3790 | text-only | **#6,891** (0.2173) | **#4,730** (0.2725) |
| 7:3790 | image-only | **#2,200** (0.3020) | **#4,047** (0.3000) |

Không chỉ hạng: `target_score` (4 chữ số), `target_margin_vs_rank1`, và `gallery_size_scored=74 381` cũng khớp từng ô. Panel còn hiển thị đúng badge `Canonical: 6,001/6,003 scored` và cảnh báo "Đây là chẩn đoán, không phải metric canonical" với hai qid `7:3038, 7:3790` — tức là trạng thái canonical đi kèm so sánh, không thể đọc rời.

**12 ranks verified: YES.**

Vị trí render: panel chỉ xuất hiện ở `samples/explorer` (1 panel). Ở `samples/compare`, `evaluation/composition`, `experiments/models`, `overview` đều **0** panel — đúng như mô tả.

## 5. Inventory mô hình

Đo tại `/?area=experiments&view=models&protocol=circus_fashioniq_release_v1&subset=sf`, viewport 1920×1080: **17 hàng**, `body.scrollWidth = innerWidth = 1920`.

Danh sách id trên UI **bằng đúng** `/api/models` **và** `workbench/registry/models.yaml` (17 = 17 = 17, so khớp tập hợp).

Thanh đếm hiển thị: `17 Mô hình đăng ký · 8 Có số đo trong phạm vi này · 8 Lượt CIRCUS có dữ liệu ở phạm vi này · 7 Cần chạy hoặc xử lý blocker · 2 Tạm gác theo yêu cầu`.

Hai mô hình tạm gác **phân biệt được** với 15 mô hình còn lại, không chỉ bằng màu:

- `CSMCIR / csmcir` — badge **"Tạm gác"** + "Tạm gác theo yêu cầu: thiếu dữ liệu phụ trợ", cột kết quả ghi "Ngoài đợt chạy này".
- `SPRC / sprc` — badge **"Tạm gác"** + "Tạm gác theo yêu cầu: chưa tải checkpoint", cột kết quả "Ngoài đợt chạy này".

Đếm badge trên 17 hàng: **"Có số đo" 7** + **"Chấm thiếu" 1** (`LIMN`, "69 / 4 069 query chưa chấm", coverage 4 000/4 069 = 98.3 %) = **8 mô hình có số đo**, khớp đúng ô đếm "8 Có số đo trong phạm vi này"; **"Chưa có số đo" 7** khớp "7 Cần chạy hoặc xử lý blocker"; **"Tạm gác" 2**. Tổng 8 + 7 + 2 = 17, không hàng nào bị ẩn. Thứ tự sắp xếp đẩy 2 hàng tạm gác xuống cuối, nhưng chúng vẫn hiện.

## 6. Lỗi phát hiện

### D1 — `load_diagnostic` vẫn phục vụ 12 hạng khi **không còn file per-scope nào** (mức trung bình)

**Đo được** (repo root tạm, file tổng còn nguyên, **0/6** file `.jsonl` per-scope):

```
status: DIAGNOSTIC_COMPLETE   available: True
12/12 target_rank vẫn được trả (ví dụ 7:3038/shirt/multimodal = 3)
evidence_files.per_scope: []          <- rỗng
mỗi ô vẫn khai báo export: limn_dual_scope.shirt.multimodal.jsonl  <- file không tồn tại
GET /candidates -> 404 "no per-scope export at limn_dual_scope.shirt.multimodal.jsonl"
```

**Vì sao là lỗi:** docstring của module tuyên bố "each scope's evidence is refused unless **its own file is present**", và panel in dòng "Bằng chứng: `limn_dual_scope.json` + **6 file per-scope**". Trong trạng thái trên, UI vẫn hiện đủ 12 con số kèm lời khai có 6 file — nhưng không có file nào để kiểm chứng, và chính panel ở chế độ drill-down sẽ 404. Đây là kiểu "phục vụ nửa vời" mà guard sinh ra để chặn. Hiện **không** ảnh hưởng bản đang chạy (6/6 file có thật, đã kiểm), nhưng guard không giữ được lời hứa của nó.

**Nguyên nhân:** `load_diagnostic` chỉ đọc `limn_dual_scope.json`; nó lấy tên file từ trường `top100_export` và **không `stat`** các file đó. Chỉ `load_candidates` mới kiểm tra tồn tại. Không có unit test nào phủ trường hợp này.

**Đề xuất sửa (chờ director quyết, tôi không sửa):** trong vòng lặp scope của `load_diagnostic`, sau khi dựng `scopes[scope]`, kiểm tra `(root / f"limn_dual_scope.{scope}.{condition}.jsonl").is_file()` cho từng điều kiện và `raise DiagnosticEvidenceError` nếu thiếu; thêm test `test_missing_per_scope_export_is_refused` vào `test_limn_dual_scope_diagnostic.py`. Nếu chủ ý cho phép "chỉ summary" thì phải nói rõ trong docstring và bỏ dòng "6 file per-scope" khỏi panel — nhưng cách đó mâu thuẫn với chính sách fail-closed đã tuyên bố.

### D2 — Sticky `.topbar` che `thead`/tiêu đề card khi cuộn tới đỉnh viewport (mức thấp, **có từ trước**, không do panel này)

**Đo được** (viewport 390×844, cuộn bảng vào `block:'start'`):

```
topbar_bottom = 126 px, thead_top = 0 px  ->  che 125 px
elementFromPoint tại đỉnh thead = HEADER.topbar
```

Tại 1920×1080, tiêu đề card ở `top=24` trong khi `topbar_bottom=89` → che 65 px (`elementFromPoint` trả `H1`, tức topbar nằm trên).

**Không phải do feature mới:** tái lập y hệt trên 2 bảng khác — `composition-table` che 103 px, `inventory-table` che 125 px. `.topbar { position: sticky }` có từ `HEAD` (dòng 109 `style.css`), và không có `scroll-margin-top` nào trong repo. Chỉ ảnh hưởng khi có thao tác cuộn/neo tới đúng đỉnh phần tử; không ảnh hưởng ảnh chụp toàn trang hay đọc bình thường.

**Đề xuất sửa:** thêm `scroll-margin-top: calc(var(--topbar-h) + 8px)` cho `table`, `.card`, hoặc `:target`/`[data-testid]` neo; hoặc cho `.topbar` `position: sticky` kèm `z-index` thấp hơn. Việc này thuộc phạm vi thiết kế chung, không phải bản vá riêng cho panel LIMN.

### D3 — Badge/notice của panel cố định ở subset Full (6 001/6 003) nhưng panel hiện trên mọi subset mà không ghi rõ subset (mức thấp)

**Đo được** (cùng một panel, ba trang subset khác nhau, badge không đổi):

| Trang subset | Hàng `LIMN` trong inventory | Badge của panel |
|---|---|---|
| `subset=sf` | `4 000 / 4 069 (98.3 %)`, 69 query chưa chấm | `Canonical: 6,001/6,003 scored` |
| `subset=validated` | `581 / 586 (99.1 %)`, 5 query chưa chấm | `Canonical: 6,001/6,003 scored` |
| `subset=full` | "2 lượt trong phạm vi này; chọn một lượt" | `Canonical: 6,001/6,003 scored` |

Trên trang SF, hai mức coverage canonical khác nhau xuất hiện cùng lúc (4 000/4 069 ở inventory, 6 001/6 003 ở panel). Câu "coverage giữ nguyên 6,001/6,003" không có bộ định lượng subset. Payload chẩn đoán **không** có trường `subset` (chỉ `gallery.protocol_id` + `release_provenance`), nên UI không có cách nào tự nói đúng.

Ở `subset=sf` endpoint kiểm tra `7:3038` trả `total=0` (không có trong cohort SF); ở `subset=full` trả `unscored=1` với `unscored_by_kind.category_unresolved` — nên phép so sánh này thuộc cohort Full.

**Đề xuất sửa:** thêm một câu vào notice kiểu "số liệu dưới đây đo trên cohort **Full** (6 003 query), không phải subset đang chọn", hoặc để backend trả `subset`/`cohort` trong payload và render nó. Không phải lỗi số liệu — 12 hạng vẫn đúng — mà là nguy cơ đọc nhầm phạm vi.

## 7. Phạm vi đã ghi

Chỉ ghi trong `workbench/docs/reviews/20261109-native15-composition-limn/`:
`WEB_VERIFICATION.vi.md` (file này), `raw/ui_shots_verify.log`, `raw/ui/verify-*.png`.
Không sửa `FINAL_REPORT.vi.md`, các CSV, `EVIDENCE_INDEX.md`, `INPUT_FORWARD_PARITY.vi.md`, `LIMN_SCOPE_AUDIT.vi.md`, `BLOCKER_RESOLUTION.vi.md`.
Không sửa code production, không đụng `workbench/config/workbench.env`, không chạy GPU, không restart server, không sửa file evidence thật.
