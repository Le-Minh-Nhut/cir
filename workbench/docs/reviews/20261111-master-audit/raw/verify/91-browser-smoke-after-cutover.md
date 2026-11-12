# 91 — Browser smoke on the served bundle, after cutover

Read-only verification. No file in `workbench/frontend/`, `dist/`, or any service unit was modified;
no `systemctl`, no rebuild, no GPU touch. One Playwright session (`workbench/envs/ctl312h/bin/python`
+ its bundled `chromium-1248` under `.playwright-browsers`), viewport **2560×1440**, `deviceScaleFactor=1`,
locale `vi-VN`, headless. Driver script lived at `/tmp/vfy/smoke.py` (outside the repo).

## 0. What is actually served

```
$ curl -s http://127.0.0.1:5173/ | grep -o 'assets/index-[^"]*\.js'
assets/index-DYYSN7oG.js

$ sha256sum workbench/frontend/dist/assets/index-DYYSN7oG.js
74e4b4f86ce8b4f5061456138ce1ff79aa084d7662c54c24a123e5bb679d651f  workbench/frontend/dist/assets/index-DYYSN7oG.js
```

Served HTML (`/tmp/vfy/index.html`) references exactly one module:

```html
<script type="module" crossorigin src="/assets/index-DYYSN7oG.js"
```

Response headers: `HTTP/1.1 200 OK`, `Content-Type: text/html`, `Cache-Control: no-cache`,
`Etag: W/"10f-HfE6Vh2BxhuQuuBuvSfaAHs7pyk"`. On-disk `dist/index.html` is 271 B, mtime
2026-11-11 19:35:53 +0700; `dist/assets/` holds only `index-DYYSN7oG.js` (389 252 B) and
`index-DecQnta9.css` (26 552 B).

**Matches the expected `index-DYYSN7oG.js`.** Nothing was rebuilt; the hash above is the identity of
the bundle every assertion below was taken against.

Artifacts: `workbench/docs/reviews/20261111-master-audit/web_evidence/e2e_post_cutover/`
(`smoke_result.json` holds every raw DOM read quoted below; six PNGs, all 2560 px wide).

---

## 1. Deep link loads at 2560×1440

URL: `http://127.0.0.1:5173/?area=samples&view=explorer&protocol=circus_fashioniq_release_v1&subset=full`

Landed with `#run-select` present and `#run-select option` count > 1. Screenshot
`01_explorer_deep_link_2560x1440.png` (2560×3694). Breadcrumb reads
`Protocol: CIRCUS FashionIQ Release v1 · Subset: Full · Lượt chọn 0 · Query: (trống)`, i.e. the deep
link's protocol/subset were honoured before any run was picked.

**PASS.**

## 2. Run picker — `#run-select`

`document.querySelectorAll('#run-select option')` → **21 nodes = 1 placeholder + 20 real options**.
Placeholder is `— chọn lượt —` (value `""`). Every real option matched
`^.+ / .+ · [0-9a-f]{8} · (Top-K \d+|không có Top-K)( \(TRANSFER\))?( — không có rank)?$`
— **20/20 conform, 0 exceptions** (checked by regex over the labels read out of the DOM).

Three labels, verbatim:

1. `limn / base_iter0_all_categories · 10173dc4 · không có Top-K`
2. `conesep / fiq_n02 · 30f6de42 · Top-K 50`
3. `ptha_mtst / cirr_transfer · 315dd590 · không có Top-K (TRANSFER)`

(Full 20-label list in `smoke_result.json` → `steps.a_run_picker.all_labels`.) The banner above the
page independently states `per-query candidate Top-K available on 15 of 20 composed runs in view`,
and 15 of the 20 labels do carry a `Top-K <n>`.

**PASS** (≥10 options; label shape holds).

## 3. Query `7:1242` on run `f461f391c6d4696011be8067` (HINT, Top-K 100)

URL: `…&view=explorer&runs=c%3Af461f391c6d4696011be8067&query=7%3A1242`
Screenshot `02_query_7-1242_hint.png` (2560×5925).

| read | value |
|---|---|
| `[data-testid=query-title]` text | `Query 7:1242` |
| reference `<img data-testid=query-image>` src | `/api/images/circus/B00CQAJI7S.jpg` |
| `data-asset-state` | `present` |
| `naturalWidth` (immediately, then after 1.2 s + scroll into view) | **322**, then **322** (stable) |
| `complete` | `true` |
| caption `[data-testid=query-caption]` | `Is shorter skirt with brighter colors and is colorful with sleeves.` |
| target `<img>` in `[data-testid=query-targets]` | `B00B79JENC.jpg`, `naturalWidth` 289, no `data-asset-state="missing"` |
| window verdict testid | `target-outside-window` |

Caption is non-empty (57 chars) and matches the API record
(`query_txt` = `Is shorter skirt with brighter colors and is colorful with sleeves.`).

Target-window block, verbatim:

> Ảnh đích ở hạng #98 **theo điều kiện Ảnh + văn bản (MM)** — **ngoài cửa sổ Top-10 đang hiện**
> (10 ứng viên được serve, export lưu 100 dòng). Hạng #98 của Ảnh + văn bản (MM) đã vượt K, nên với
> K=10 hay K=20 ảnh đích không nằm trong danh sách. Đây là hạng toàn gallery, không phải vị trí trong
> danh sách ứng viên.

API agrees the target is outside the shown window: `ranks.multimodal = 98` at `top_k=10`,
`candidate_depth = 100`, `positive_ids = ["mbeir_images/fashioniq_images/B00B79JENC.jpg"]` — not among
the 10 served rows. The block is rendered because the target is genuinely outside, and it names the
condition (MM) rather than borrowing another condition's rank.

**PASS** (reference image loads, caption non-empty, outside-window state shown and correct).

## 4. Top-K 10 vs 50, condition `multimodal`

API truth first —
`curl -s "http://127.0.0.1:8000/api/composition/runs/f461f391c6d4696011be8067/queries?subset=full&condition=multimodal&query_id=7%3A1242&top_k=50&limit=1"`:

```json
{"total": 1, "limit": 1, "offset": 0, "condition": "multimodal"}
query_id = "7:1242", candidate_depth = 100, len(top_results) = 50
{"rank": 1, "image_id": "mbeir_images/fashioniq_images/B009CCA04I.jpg", "gallery_position": 45873, "score": 0.5162218809127808}
{"rank": 2, "image_id": "mbeir_images/fashioniq_images/B00BU8ZRX6.jpg", "gallery_position": 60312, "score": 0.5126523375511169}
{"rank": 3, "image_id": "mbeir_images/fashioniq_images/B00870UZ0K.jpg", "gallery_position": 37421, "score": 0.4979403018951416}
```

DOM, read from `[data-testid=topk-grid] article.candidate` after each `select_option` on
`[data-testid=topk]` (option list is `1, 5, 10, 20, 50, 100`; the request identity includes K, so each
change is a new fetch):

| K | rows in DOM | first three (rank · image_id) | select value |
|---|---|---|---|
| 10 | **10** | `#1 B009CCA04I.jpg` · `#2 B00BU8ZRX6.jpg` · `#3 B00870UZ0K.jpg` | `10` |
| 50 | **50** | `#1 B009CCA04I.jpg` · `#2 B00BU8ZRX6.jpg` · `#3 B00870UZ0K.jpg` | `50` |

Rows match `min(K, available)` = `min(50, 50)` = 50 at K=50 and 10 at K=10 (50 candidates served,
100 rows in the export). First id and rank are identical between K=10 and K=50 **and identical to the
API**. Counter line at K=50, verbatim:

> Đang hiển thị **50** ứng viên · export lưu 100 dòng · Ảnh + văn bản (MM) · ảnh đích KHÔNG nằm trong danh sách này

Screenshots: `03a_topk_10_multimodal.png` (2560×5921), `03b_topk_50_multimodal.png` (2560×7206).

**PASS.**

## 5. Compare view, two runs of the same model

URL: `…&view=compare&runs=c%3A10173dc45dea8c90fd2a284b,c%3A9e4918a6b5c243412cbc58c9&query=7%3A1242`
Screenshot `04_compare_two_limn_runs.png` (2560×2383).

Both columns are the same `model_id`/`checkpoint_id` (`limn / base_iter0_all_categories`) yet resolve
to different labels, so they are distinguishable:

- `limn / base_iter0_all_categories · 10173dc4 · không có Top-K (đang xem)` — rank `#193`
- `limn / base_iter0_all_categories · 9e4918a6 · Top-K 100` — rank `#193`

Table caption: `Hạng multimodal của cùng query trên các lượt cùng scope`. Cohort line:
`Cohort: 2 lượt, cùng protocol circus_fashioniq_release_v1 và subset full; hạng thiếu được ghi riêng là chưa chấm.`
Neither header is a bare `limn`: each carries the 8-char run stub and its own Top-K state
(`không có Top-K` vs `Top-K 100`), which is exactly the ambiguity (two LIMN runs both printing
"limn #193") this view has to survive.

Peer picker was populated (`— chọn lượt —` + same-scope runs), so the peer list is reachable, not
a two-row dead end.

**PASS.**

## 6. No-results path — query `7:99999` on the same run

URL: `…&view=explorer&runs=c%3Af461f391c6d4696011be8067&query=7%3A99999`
Screenshot `05_no_results_7-99999.png` (2560×4406).

The backend answers this with `HTTP 200` and `{"items":[],"total":0,...}` (not 404/500), and the UI
renders the inspector slot's `EmptyState kind="no-results"`:

```html
<div class="empty empty-no-results" data-testid="empty-state" data-empty-kind="no-results" role="status">
```

Text, verbatim:

> **Chưa có kết quả** — Không tìm thấy đúng query 7:99999 ở lượt này hoặc subset đang chọn. Không tự
> thay bằng query khác. — [Chọn lượt khác]

The state names the qid (`7:99999`), is `role="status"` (not `role="alert"`), and there is no
`data-empty-kind="error"` node anywhere on the page — so this is a no-results state, not an error
dialog. It is also not a blank page: the query list, the rank grid for the list's selected row, the
ambiguity diagnostic and the header all still render around it. The inspector is deliberately absent
(`inspector_present: false`) because the requested query does not exist on this run — the empty state
is the inspector's own answer, replacing it rather than falling back to `items[0]`.

**PASS.**

## 7. Deliberately skipped

- **Cohort save** — not exercised. Cohort/annotation writes on CIRCUS runs are a known dead end in
  this deployment (documented in the master-audit E2E reports); the run was read-only by design, so
  the write path was left untouched rather than provoked.
- **`/api/models`** — not requested. That endpoint has a ~154 s cold path; hitting it would have
  stalled this smoke behind an unrelated cache warm-up and proved nothing about the bundle.

Neither skip weakens any assertion above: every check was read from a request that had already
answered.

## 8. Browser health

`console_errors: []`, `page_errors: []` for the whole session (six page loads). No crash, no
traceback, exit 0.

---

## Kết luận

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | Served bundle is `index-DYYSN7oG.js` | **PASS** | `assets/index-DYYSN7oG.js`; sha256 `74e4b4f86ce8b4f5061456138ce1ff79aa084d7662c54c24a123e5bb679d651f` |
| 2 | Deep link loads at 2560×1440 | **PASS** | `01_explorer_deep_link_2560x1440.png` (2560×3694) |
| 3a | `#run-select` ≥10 options, label shape | **PASS** | 20 real options (+1 placeholder), 20/20 regex-conform; `limn / base_iter0_all_categories · 10173dc4 · không có Top-K` |
| 3b | `7:1242` reference loads, caption non-empty, target outside window | **PASS** | `naturalWidth` 322 (`asset-state=present`), caption `Is shorter skirt with brighter colors and is colorful with sleeves.`, `target-outside-window` #98/MM; `02_query_7-1242_hint.png` |
| 3c | Top-K 10 vs 50 row count = `min(50, available)`, ids+ranks = API | **PASS** | DOM 10 rows / 50 rows; API & DOM both `#1 B009CCA04I.jpg`, `#2 B00BU8ZRX6.jpg`, `#3 B00870UZ0K.jpg`; `03a_…png`, `03b_…png` |
| 3d | Compare: two same-model columns distinguishable | **PASS** | `limn / base_iter0_all_categories · 10173dc4 · không có Top-K (đang xem)` vs `limn / base_iter0_all_categories · 9e4918a6 · Top-K 100`; `04_compare_two_limn_runs.png` |
| 3e | No-results for `7:99999` names the qid, not an error/blank | **PASS** | `data-empty-kind="no-results"`, role=status, text `Không tìm thấy đúng query 7:99999 …`; no `data-empty-kind="error"`; `05_no_results_7-99999.png` |
| 4 | Cohort save / `/api/models` skipped on purpose | **SKIPPED** | §7 (known dead end; 154 s cold path) |
| 5 | No console/page errors | **PASS** | `console_errors: []`, `page_errors: []` |

**7 PASS, 0 FAIL, 1 deliberately skipped (two items), 0 selector misses.**
