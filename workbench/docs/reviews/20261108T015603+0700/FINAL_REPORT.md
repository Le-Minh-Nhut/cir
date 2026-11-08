# BÁO CÁO CUỐI — round 20261108T015603+0700

> **CHƯA COMMIT, CHƯA PUSH.** Branch `research/cir-failure-workbench`, HEAD **`1716e81`**.
> Toàn bộ công việc nằm trong working tree.

`git status --short` (chạy lại lúc viết file):

```
 M workbench/backend/main.py
 M workbench/composition/model_csmcir.py
 M workbench/frontend/App.tsx
 M workbench/frontend/gate.ts
 M workbench/frontend/index.html
 M workbench/frontend/style.css
 M workbench/frontend/tests/ui_smoke.py
 M workbench/frontend/types.ts
?? workbench/backend/composition.py
?? workbench/docs/reviews/20261107T230700+0700/
?? workbench/docs/reviews/20261107T234628+0700/
?? workbench/docs/reviews/20261108T015603+0700/
?? workbench/frontend/BenchmarkView.tsx
?? workbench/frontend/CompositionView.tsx
?? workbench/frontend/tests/browser_qa.py
?? workbench/tests/test_composition_api.py
?? workbench/tests/test_csmcir_aggregate_api_parity.py
```

**Xong:** dữ liệu CIRCUS pinned, 4/9 model có số thật đủ 3 subset, web composition + native chạy
trên dữ liệu thật, browser QA 3 viewport, review độc lập 21/21 metric khớp.
**Chưa xong:** 4 model bị gate chặn (F1), CSMCIR bị chặn (contract scorer), chưa commit.

---

## 0. Web ĐANG CHẠY — URL để mở ngay

**http://127.0.0.1:5173** (frontend) · API **http://127.0.0.1:8000**

Stack để **CHẠY TIẾP**, không tắt. Kiểm lúc viết file: frontend **200**, API **200**
(`/api/benchmark-runs`).

- Chạy dưới service tên `cir-workbench`, PID `3213589` (`serve_workbench.py`), con: uvicorn
  `3213598` (port 8000), vite preview `3213611` (port 5173).
- **Dừng:** `write proc://cir-workbench/kill`
- **Khởi động lại:**

```bash
cd /home/heheboiz/data/cir
WORKBENCH_CIRCUS_RELEASE_ROOT=/home/heheboiz/data/cir/workbench/artifacts/circus_release \
workbench/envs/ctl312h/bin/python workbench/scripts/serve_workbench.py \
  --production-frontend --host 127.0.0.1 --backend-port 8000 --frontend-port 5173
```

Cách khác, một lệnh: `workbench/run.sh serve` (mode `serve` của `pipeline.py`, tự chọn interpreter
`.venv/bin/python`; host/port lấy default của config = `127.0.0.1:8000`/`5173`, nhưng **không** bật
`--production-frontend`, nên phục vụ dev server chứ không phải `dist/`).

**Build đang phục vụ** — đọc từ `workbench/frontend/dist/index.html` và `curl` cùng URL, khớp:

| | giá trị |
|---|---|
| JS | `assets/index-CsxrZyq6.js` |
| CSS | `assets/index-uzz8fv-H.css` |

```bash
cat workbench/frontend/dist/index.html | grep -o 'assets/index-[^"]*'
curl -s http://127.0.0.1:5173/ | grep -o 'assets/index-[^"]*'
# -> index-CsxrZyq6.js, index-uzz8fv-H.css ở cả hai
```

Sửa frontend xong phải `npm run build --prefix workbench/frontend` rồi restart, nếu không
`vite preview` vẫn phục vụ bundle cũ.

---

## 1. Bảng kết quả CIRCUS — 4/9 model có số thật

Đọc bằng `json.load` từ `workbench/artifacts/composition/summary/<dir>/<subset>.json`.

### 1.1 full (6003 query, gallery 74,381)

| model | checkpoint | run_id | MM R@10 | MM R@50 | MM nDCG | gap | coverage | exec | repro_status |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| **tme** | fashioniq_0 | `ad477312aa57c1b9114f9e51` | **34.0163** | **53.9564** | **0.32039** | **0.35860** | 1.00000 | COMPLETED | NOT_RUN |
| **encoder** | fashioniq | `6a528e877c328772e2c7bb28` | **31.0178** | **49.5752** | **0.29586** | **0.47230** | 1.00000 | COMPLETED | METRICS_MEASURED_MAPPING_UNVERIFIED |
| **limn** | base_iter0_all_categories | `10173dc45dea8c90fd2a284b` | **30.0646** | **48.7593** | **0.29084** | **0.54174** | **0.98018** | PARTIAL | BLOCKED_RUNTIME_PREREQUISITES |
| **ptha_mtst** | cirr_transfer | `315dd590eeb03d2c78d7937a` | **19.4736** | **34.4494** | **0.22493** | **0.24362** | 1.00000 | COMPLETED | SELECTED_TRANSFER_ONLY |

### 1.2 sf (4069 query)

| model | scored | cov | R@10 | R@50 | nDCG | gap | text nDCG | image nDCG |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tme | 4069 | 1.00000 | 19.0219 | 39.3463 | 0.22508 | 0.46284 | 0.12090 | 0.10169 |
| encoder | 4069 | 1.00000 | 18.6778 | 36.3234 | 0.21886 | 0.50768 | 0.10775 | 0.09765 |
| limn | 4000 | 0.98304 | 18.6000 | 36.2250 | 0.21767 | 0.54462 | 0.09912 | 0.08928 |
| ptha_mtst | 4069 | 1.00000 | 7.0779 | 18.8990 | 0.14979 | 0.28278 | 0.10743 | 0.08855 |

### 1.3 validated (586 query)

| model | scored | cov | R@10 | R@50 | nDCG | gap | text nDCG | image nDCG |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tme | 586 | 1.00000 | 33.9590 | 60.4096 | 0.31096 | 0.56010 | 0.13679 | 0.11021 |
| encoder | 586 | 1.00000 | 34.6416 | 55.2901 | 0.29646 | 0.60835 | 0.11611 | 0.10776 |
| limn | 581 | 0.99147 | 30.2926 | 53.7005 | 0.28619 | 0.64297 | 0.10218 | 0.09735 |
| ptha_mtst | 586 | 1.00000 | 12.9693 | 30.3754 | 0.18767 | 0.37858 | 0.11662 | 0.09521 |

### 1.4 Outcome bins tại K=10, subset full

Đây là thứ cho Composition Gap ý nghĩa: gap = 1 − max(text, image)/multimodal trên cohort có đủ 3 rank.

| model | both | composition_required | image_only | text_only | unresolved | neither_scored | tổng |
|---|---:|---:|---:|---:|---:|---:|---:|
| tme | 23 | 1105 | 139 | 942 | 3794 | 0 | 6003 |
| encoder | 5 | 1282 | 155 | 510 | 4051 | 0 | 6003 |
| limn | 4 | 1386 | 143 | 367 | 3984 | **119** | 6003 |
| ptha_mtst | 5 | 571 | 75 | 644 | 4708 | 0 | 6003 |

Đọc bảng này: `text_only` (chỉ ablation text hit) cao ở tme (942) và ptha (644) nghĩa là chúng
thoát được phụ thuộc ảnh tham chiếu nhiều hơn encoder (510). `unresolved` 3794–4708 / 6003 phản ánh
sàn ngẫu nhiên trên gallery 74,381 — không phải model hỏng.

### 1.5 Giới hạn phải đọc kèm

- **ptha_mtst là CIRR→FashionIQ transfer.** `reproduction_status: SELECTED_TRANSFER_ONLY`.
  **KHÔNG phải dòng so sánh paper.** Đừng xếp nó cạnh tme/encoder/limn như cùng hạng.
- **limn coverage 0.98018** — 119/6003 không scored vì category mơ hồ (qid kiểu `7:2179` thuộc cả
  `shirt` lẫn `toptee`). LIMN có 3 checkpoint theo category nên **không auto-assign** — hành vi đúng,
  không phải lỗi. `category_scope_failures` = 119 ở cả 3 condition.
- **Mọi con số ở đây là trên `circus_fashioniq_release_v1`, gallery 74,381.** Paper ENCODER công bố
  R@10 56.13 trên `fashioniq_val_split` — **khác protocol, không so được**. Paper CIRCUS nói 18,853
  ảnh; release có 74,381. Summary tự khai trong `paper_vs_release_gallery`.

---

## 2. Trạng thái 9 model

| model | trạng thái | lý do |
|---|---|---|
| **tme** | CÓ SỐ (3 subset) | chạy dưới `WORKBENCH_RUNTIME_ADAPTATION=sm_120`, interpreter verified=true, GPU_RUNTIME_READY, py 3.10.22 / torch 2.7.1+cu128 |
| **encoder** | CÓ SỐ (3 subset) | không gate; advisory `PYTHON_VERSION_MISMATCH`, verified=false, vẫn score |
| **limn** | CÓ SỐ (3 subset, PARTIAL) | không gate; advisory `DEPENDENCY_VERSION_MISMATCH`, verified=false |
| **ptha_mtst** | CÓ SỐ (3 subset) | adapter không có gate; sm_120, py 3.10.22, verified=true |
| **habit** | **FAILED** | gate chặn: expected Python 3.8.10, `airknow-adapted` là 3.11.15. sm_120 pin là 3.9 → vẫn chặn |
| **intent** | **KHÔNG CHẠY (sẽ bị chặn)** | expected 3.9 (cả 2 contract), `airknow-adapted` 3.11.15 |
| **hint** | **KHÔNG CHẠY (sẽ bị chặn)** | expected 3.8.10 (hist) / 3.9 (sm_120), 3.11.15 |
| **conesep** | **KHÔNG CHẠY (sẽ bị chặn)** | expected 3.8.10 / 3.9, 3.11.15 |
| **csmcir** | **KHÔNG THỬ** | ngoài 6 model được yêu cầu; checkpoint staged nhưng `WORKBENCH_CSMCIR_PYTHON` chưa set. Xem §5 |

Nguồn: `workbench/artifacts/composition/cohort_status.txt`.

**4/9 có số. 4 chặn bởi F1 gate. 1 chặn bởi contract CSMCIR.**

---

## 3. F1 — gate môi trường không nhất quán (4 model bị chặn)

Ba phần, mỗi phần có file:line:

**(a) Gate chỉ được 5/9 adapter áp dụng.** Năm adapter gọi
`require_verified_model_interpreter()` trong `_run_job`:
`model_habit.py:152`, `model_intent.py:148`, `model_hint.py:150`, `model_conesep.py:164`,
`model_tme.py:164`. Bốn adapter **không gọi**: `model_ptha_mtst.py`, `model_encoder.py`,
`model_limn.py`, `model_csmcir.py` — chúng vẫn score với chỉ advisory record `verified=false`.
Nghĩa là **cùng một mismatch**: chí mạng ở adapter này, chỉ là cảnh báo ở adapter khác.

**(b) `workbench/backend/runtime.py:1644-1649`** set `TIER_BLOCKED` và **return ngay**, nên khối
dependency verification bên dưới **không bao giờ chạy** trên nhánh đó.

**(c) `--plan` chỉ probe CUDA/capability** (`model_habit.py:111-127`), in `ready:true` /
`status:READY` / `reason:null`, rồi run chết ở gate. Plan và gate kiểm hai contract khác nhau.

**Không có bypass.** Grep `SKIP_ENV|ALLOW_UNVERIFIED|BYPASS|FORCE_UNVERIFIED` trong `runtime.py`
= **0 match**. Switch duy nhất `WORKBENCH_RUNTIME_ADAPTATION=sm_120` **đổi contract** (block
`adapted_runtime` trong registry), **không** bypass gate.

**Không có interpreter 3.9 hay 3.8.10 trên host.** Quét `workbench/envs/*`:

```
airknow-adapted  3.11.15    limn-adapted  3.11.15
clip4cir-adapted 3.11.15    pair-adapted  3.11.15
ctl312           3.12.3     sprc-adapted  3.11.15
ctl312h          3.12.11    test-adapted  3.11.15
dcnet-prepare    3.7.12     tme-adapted   3.10.22
dl-adapted       3.11.15
```

Không có 3.9, không có 3.8.10.

**Hành động tiếp cho bạn:** provision prefix Python 3.9 dùng chung (wheel cp39 + cu128 đã publish,
`uv` có sẵn tại `/home/heheboiz/.local/bin/uv`) — khoảng **1–2 h** — hoặc chấp nhận cohort 4 model.

---

## 4. Gap provenance mới — phải quyết định

TME và PTHA chạy dưới `WORKBENCH_RUNTIME_ADAPTATION=sm_120`, nhưng **cả hai summary không chứa
`variant_provenance`**, và `grep -c sm_120` trong `full.json` = **0** (tôi đếm bằng
`json.dumps(x).count("sm_120")`).

Nghĩa là artifact **không tự khai** nó chạy trên contract adapted. Badge hiện chỉ được ghi trong
`cohort_status.txt` — một file phụ, không phải artifact.

Hệ quả: ai đọc JSON một mình sẽ tưởng đây là reproduction theo contract pinned. **Không phải.**

Quyết định cần chọn: patch runner để ghi `variant_provenance.runtime_adaptation` vào summary, hoặc
chấp nhận chỉ ghi trong docs. Tôi nghiêng về patch runner — badge trong docs sẽ mất khi docs bị
viết lại.

Lưu ý ngược lại: encoder và limn **cũng** không có `variant_provenance` (đúng, vì chúng không dùng
sm_120 — chúng chạy advisory-mismatch). Nhưng encoder cũng không tự khai mismatch trong field riêng,
chỉ có `runtime.model_interpreter.verification_status`.

---

## 5. CSMCIR — chốt lần cuối

Hai blocker **độc lập**, cả hai đã đo:

**5.1 Contract scoring khác nhau.** Adapter `score()` = text embedding 256-d nở ×32 thành vector
8192-d rồi dot product. Native `inference()` = attention-weighted token importance, sau đó max trên
32 query token của matmul 1×1×256. Hai đại lượng khác nhau — thay cái này bằng cái kia là đổi thuật
toán. Nguồn: `workbench/composition/model_csmcir.py:7-9` và `:53-59`.

**5.2 Caption chỉ phủ 20.72% gallery.** Đếm lại bằng `json.load`:

| file | entries |
|---|---:|
| `dress_cot_val.json` | 3817 |
| `shirt_cot_val.json` | 6346 |
| `toptee_cot_val.json` | 5373 |
| **tổng** | **15536** |
| **distinct** | **15415** |

`15415 / 74381 = 20.72%` → **58,966 ảnh không có caption.**

### ĐÃ RÚT LẠI (RETRACTED)

Hai khẳng định từng nằm trong code và trong reply trước của vòng này:

1. ~~"caption `COT_ours2` không có"~~ — **SAI**. Chúng được publish ở revision Hub đã pin với sha256
   khớp. `CHECKPOINT_SELECTION_REPORT.md` 3.18; **D-8 đã đóng**.
2. ~~"checkout pinned không import được / chỉ ship bytecode"~~ — **SAI**. `.pyc` của derived checkout
   import được và native run hoàn tất **rc 0**.

Chuỗi lý do trong `model_csmcir.py` đã được sửa trong vòng này để chỉ nêu đúng 2 lý do thật.
**41 test csmcir pass** (đã chạy lại).

### NATIVE result đứng riêng

Run `f03887d2`, `is_synthetic: false`, `parity_status: PARITY_VERIFIED`, `return_code 0` /
`INVOCATION_COMPLETED` / `PROOF_VALIDATED`:

| chỉ số | giá trị |
|---|---:|
| macro R@10 | **57.05073873202006** |
| macro R@50 | **77.29859352111816** |
| macro mean | 67.1747 |

Nhãn đúng: **"public-caption path-mapped research run"**, không phải official reproduction.

> Đã kiểm: `f564570e` có `observed_metrics` **giống hệt** `f03887d2` (57.0507 / 77.2986 / 67.1747).
> Đây là lý do F3 tồn tại — hai report là một phép đo, không phải hai.

---

## 6. Web — đã làm gì

- **Composition read-only API**: `GET /api/composition/runs[?subset=&include_fixtures=&include_smoke=]`,
  `/runs/{run_id}`, `/runs/{run_id}/queries` (phân trang + filter). Reader
  `workbench/backend/composition.py`.
- **Chuẩn hoá nested → flat** để run thật render được. Trước đó mọi cột ra "—" vì UI đọc key phẳng
  còn runner ghi nested (`backend/composition.py:135-146`).
- **Fixture loại mặc định, sort dưới measurement.** Trước đó thứ tự byte đặt `SYNTHETIC_*` trên
  `encoder_*` nên dòng giả mở đầu bảng. Nay tier 0 measurement / 1 smoke / 2 fixture, rồi run_id.
  Opt-in vẫn chạy.
- **Cột R@50 đã thêm**: `CompositionView.tsx:174` (`<th>MM R@50`) + `:211`
  (`formatRecall(m.recall_at_50_multimodal)`).
- **Provenance hiện cạnh số**: `env: PYTHON_VERSION_MISMATCH · METRICS_MEASURED_MAPPING_UNVERIFIED`
  ở dòng bảng (`CompositionView.tsx:198`) và drill-down (`:251`).
- **Badge CERTIFIED giờ data-driven ở CẢ HAI chỗ**: `App.tsx:69` (Dashboard) và `:71`
  (Evaluation Runs) gọi `isCertifiedEvidence(...)`. `App.tsx:70` (Models) chưa từng có badge.
  `BenchmarkView.tsx:32,56-57` vốn đã đúng. **Chứng minh bằng thực nghiệm**: chặn
  `/api/benchmark-runs` và ghi lại `verification.state` → badge đổi thành `UNCORROBORATED` /
  `UNCERTIFIED`. Ảnh trong `native/`.
- **Duplicate CSMCIR labelled**: `duplicate attempt — provenance knobs missing`; headline đổi thành
  **"2 certified research results (1 unique measurement)"** (`App.tsx:69`).
- **Cột Model sticky trên mobile**: pin, nền đục, `z-index` 10/5, `scroll-padding-left: 157px`
  (`style.css:77-105`). Đo `partially_occluded: []` tại 25/50/75/100% max scroll trên 390x844 và
  768x1024; overflow trang 0.

### Test — 3 test cũ đã sửa xong

| lệnh | kết quả |
|---|---|
| `pytest test_composition_api.py test_csmcir_aggregate_api_parity.py` | **15 passed** |
| `pytest test_circus_resume_stable.py` | **9 passed** |
| `pytest -k csmcir` | 41 passed |
| `npm run build --prefix workbench/frontend` | clean |

Ba test mới thêm trong vòng này:

| test | khẳng định |
|---|---|
| `test_failed_run_exposes_is_failed_flag` | summary `execution_status: FAILED` mang `is_failed: true` ra API |
| `test_failure_message_never_shows_python_none_sentinel` | text trả cho người dùng không chứa `category None` |
| `test_unresolved_category_reason_is_readable` | lý do category mơ hồ đọc được, marker phân loại resume giữ nguyên |

Ba test fail cũ là **assertion lỗi thời, không phải bug app**: chúng khẳng định
`runs[0]["run_id"] == real["run_id"]` với `real` là summary đầu tiên theo thứ tự alphabet — tức test
giả định chỉ có **một** run thật, trong khi đã có 5. Đã viết lại thành invariant thật (measurement →
failed → smoke → fixture, determinism, không synthetic/smoke trong list mặc định); revert-check xác
nhận test mới fail khi bẻ tier logic.

---

## 7. Browser QA

Chromium thật (Playwright **1.64.1** trong `workbench/envs/ctl312h`, managed browser **681 MB**).
**Không có OMP browser facade trong session này** — đã kiểm; Playwright là đường duy nhất.

Bundle: `browser_qa/` — 165 PNG, `flows.json`, `flows.py`, `zooms.py`, JSON + screenshot per page.

**Kết quả: 13/15 flow PASS** ở 1440x900 / 768x1024 / 390x844. Console error **0**, page error **0**,
failed request **0**, HTTP ≥400 **0** — trong cả 15 flow.

**QUAN TRỌNG — phạm vi hiệu lực:** `flows.json` ghi lúc **02:19**, còn bản fix sticky ghi lúc
**03:15**. Nghĩa là 13/15 đó **là kết quả TRƯỚC khi fix sticky**. Hai flow fail chính là
`mobile_layout` tại 768 và 390 với lý do `row identity survives horizontal scroll` — đúng cái đã sửa.

### Đã re-run trên stack live (sau fix)

`browser_qa.py` chạy lại trên stack đang sống, `live/browser_qa.json`, `ok: true`:

| viewport | pass | nodes | overflow px | console | page | req | HTTP ≥400 |
|---|---|---:|---:|---:|---:|---:|---:|
| 1440x900 | **True** | 111 | **0** | 0 | 0 | 0 | 0 |
| 390x844 | **True** | 111 | **0** | 0 | 0 | 0 | 0 |

Fix sticky verify riêng: `partially_occluded: []` tại **4 vị trí scroll × 2 viewport** (25/50/75/100%
max scroll trên 390x844 và 768x1024). Ảnh: `browser_qa/screens/fixed_sticky_maxscroll_390x844.png`,
`..._768x1024.png`.

**DOM check có mục tiêu** (`live/composition_dom_check.json`, **16/16 check pass**): dòng FAILED
(HABIT) hiện đủ 7 ô số đều `—`; bốn model còn lại giữ nguyên số
(limn `0.2908/30.1%/48.8%/…/98.0%`, ptha `0.2249/19.5%/…`, encoder `0.2959/31.0%/…`, tme giữ nguyên);
`category None` xuất hiện **0 lần** trong DOM.

**Vẫn còn nợ:** harness đầy đủ **15 flow `flows.py`** chưa chạy lại từ sau fix sticky — chỉ
`browser_qa.py` và các DOM check mục tiêu đã chạy. Không được nói 15/15 PASS.

Giá trị thật khớp UI: cả 3 subset mọi ô số khớp byte-for-byte với JSON. Drill-down: **40 ảnh**
`naturalWidth > 0`, **0 broken**; rank thiếu render `unscored` tường minh, **không bao giờ 0**.

---

## 8. Kiểm chứng độc lập

`REVIEW_ENCODER.md` (363 dòng). Reviewer read-only, tính tay toàn bộ bằng `math.log2` + vòng lặp
dict, **không** import helper từ `workbench.composition.metrics`.

**Kết quả: 21/21 ô metric và 18/18 shortcut bin khớp chính xác, abs diff 0.0** (so `float` bằng
`==`, không làm tròn). Gallery 74,381, 6 release pin, checkpoint sha256
`0f31c501…9173` (631,879,365 B), adapter + 5 method hash, và **run-identity binding** — tất cả tái
tạo độc lập và khớp. Cross-check artifact thứ hai: 18 file per-query jsonl khớp rank table **mọi
qid, 0 mismatch**.

Chi tiết đáng nhớ: `runtime_identity` (chứa cả `verification_status`) nằm **trong** digest run-id;
reviewer tính lại `sha256(json(identity, sort_keys))` = `fingerprint` = `resume_key`, `[:24]` =
run_id. Đổi environment → run id khác → **so sánh với run tương lai vẫn an toàn**.

Findings F1–F15. Còn mở có chủ đích:

| id | nội dung |
|---|---|
| **F1** | Gate environment không nhất quán (§3) |
| **F7** | `reference_eligible` **assert, không đo**: `runner.py:707-708` `getattr(..., True)` default True rồi ghi literal `True` tại `:748` và `:1004`. Đúng thực tế cho run encoder (6003/6003 reference trong gallery, 0 mask) nhưng không suy ra từ ranking |
| **F5** | Fixture tự mâu thuẫn — gap habit khai 0.2856 vs suy ra 0.4724; bins cộng 6001 (habit) và 5881 (encoder) ≠ 6003 |
| **F9** | Spec dẫn 2 smoke run không tồn tại, bỏ sót run thật `6a528e87…` |

F2/F3/F4/F8/F10 **đã fix** (§6). F6/F11/F12/F13 **ACCEPTABLE**. F14 **UNVERIFIABLE** (không chạy
lại GPU).

---

## 9. Evidence — cách kiểm lại

`workbench/docs/reviews/20261108T015603+0700/` — **212 file, 44,114,992 B ≈ 44.1 MB** (toàn bundle,
trừ chính file index). `composition/` chỉ là **18 file, 2,439,933 B**: phần `summary/` + `ranks/` +
PNG QA, tức lớp bằng chứng composition.

`EVIDENCE_INDEX.json` **ở gốc bundle** (một index duy nhất, `dest` relative với gốc bundle) có
**212 entry: 186 PNG, 19 JSON, 5 MD, 2 PY**, tôi verify lại **212/212 hash khớp**.
Index cũ `composition/EVIDENCE_INDEX.json` (18 entry, `dest` relative với `composition/`) đã bị xoá
để không còn hai nguồn sự thật lệch nhau.

Nội dung dữ liệu `composition/`:
- `summary/encoder_fashioniq_6a528e87…/{full,sf,validated}.json`
- `summary/limn_base_iter0_all_categories_10173dc4…/{full,sf,validated}.json`
- `summary/habit_fiq_n02_e0dd04a46ef7a58334b9e353/full.json` (run **FAILED**, xem cuối §9)
- `ranks/6a528e877c328772e2c7bb28/encoder_fashioniq.json`
- `ranks/10173dc45dea8c90fd2a284b/limn_base_iter0_all_categories.json`
- `ranks/ad477312aa57c1b9114f9e51/tme_fashioniq_0.json` (TME, 378,976 B)
- `ranks/315dd590eeb03d2c78d7937a/ptha_mtst_cirr_transfer.json` (PTHA, 385,302 B)

**Kiểm lại bằng tay:**

```bash
cd workbench/docs/reviews/20261108T015603+0700
sha256sum composition/summary/encoder_fashioniq_6a528e877c328772e2c7bb28/full.json
# so với entry dest="composition/summary/encoder_fashioniq_6a528e877c328772e2c7bb28/full.json"
# trong EVIDENCE_INDEX.json ở gốc bundle

/home/heheboiz/data/cir/workbench/envs/ctl312h/bin/python -c \
  "import json;print(json.load(open('composition/summary/encoder_fashioniq_6a528e877c328772e2c7bb28/full.json'))['metrics'])"
```

**Ba field chứng minh đây là phép đo thật:**

| field | giá trị |
|---|---|
| `gallery_count` | `74381` |
| `gallery_identity_sha256` | `1db68c59d96f014cdbecf936e108653de98abef0bd9a98a8abc2fe08b5dec4b4` |
| `is_synthetic` | `false` |

**Bổ sung lần này:** rank file của **TME**
(`composition/ranks/ad477312aa57c1b9114f9e51/tme_fashioniq_0.json`, 378,976 B, sha256 `2aa9f699…426d`) và
**PTHA** (`composition/ranks/315dd590eeb03d2c78d7937a/ptha_mtst_cirr_transfer.json`, 385,302 B, sha256
`eee6b94f…c443`) đã có trong bundle. Lớp `composition/` giờ **18 file** (11 dữ liệu + 7 PNG), tổng
**2,439,933 B ≈ 2.4 MB**; toàn bundle (thêm `browser_qa/` 176 file, `live/` 8, `native/` 6, 4 file MD
gốc) là **212 file, 44,114,992 B ≈ 44.1 MB**, **212/212 hash khớp**, và hai file mới
khớp byte-for-byte với nguồn trong `workbench/artifacts/composition/runs/`. Người đọc không cần đọc
thẳng từ `workbench/artifacts/` nữa.

### Evidence của run FAILED (HABIT)

`composition/summary/habit_fiq_n02_e0dd04a46ef7a58334b9e353/full.json` được commit **đúng để** claim
ở §2/§3 kiểm được mà không cần artifact root (`workbench/artifacts/**` bị gitignore, nên nếu thiếu
file này thì claim FAILED chết trên mọi checkout khác).

Người đọc tự chứng minh FAILED bằng một lệnh, chạy trong `workbench/docs/reviews/20261108T015603+0700/`:

```bash
/home/heheboiz/data/cir/workbench/envs/ctl312h/bin/python -c "import json;d=json.load(open('composition/summary/habit_fiq_n02_e0dd04a46ef7a58334b9e353/full.json'));print(d['execution_status'], d['condition_status']['multimodal']['status'], d['condition_status']['multimodal']['reason'])"
```

Output mong đợi (tôi đã chạy đúng lệnh này và dán nguyên văn output):

```
FAILED MODEL_UNAVAILABLE RuntimeError: execution environment not verified [BLOCKED/PYTHON_VERSION_MISMATCH]: WORKBENCH_HABIT_PYTHON=/home/heheboiz/data/cir/workbench/envs/airknow-adapted/bin/python (expected Python 3.8.10, found 3.11.15)
```

Run này **cố ý** vẫn hiện trong list mặc định `/api/composition/runs` ở **tier 1** (dưới measurement,
trên smoke), **không bao giờ bị drop**: một run fail là thông tin, không phải rác — giấu nó đi mới là
làm sai lệch cohort (§10, defect tier đã fix).

### Evidence của lần chạy trên stack live

`workbench/docs/reviews/20261108T015603+0700/live/` — 8 file:

| file | nội dung |
|---|---|
| `1440x900.png` | sweep `browser_qa.py` @1440x900 |
| `390x844.png` | sweep `browser_qa.py` @390x844 |
| `composition-1440x900.png` | trang Composition đầy đủ @1440x900 |
| `composition-390x844.png` | trang Composition đầy đủ @390x844 |
| `composition-table-1440x900.png` | riêng bảng Model Runs (5 dòng, gồm dòng FAILED) |
| `composition-drilldown-1440x900.png` | drill-down một query card |
| `browser_qa.json` | output thô: 2 viewport, đều `pass: true`, overflow 0, 0 lỗi |
| `composition_dom_check.json` | 16/16 check DOM pass |

`browser_qa.json` ở cả hai viewport: `pass: true`, `nodeCount: 111`, `overflowPixels: 0`,
console/page/request/HTTP≥400 đều **0**.

```bash
# tự đọc lại hai file JSON này
cd workbench/docs/reviews/20261108T015603+0700/live
python3 -c "import json;d=json.load(open('browser_qa.json'));print(d['ok'],[ (v['viewport'],v['pass'],v['consoleErrorCount'],v['overflowPixels']) for v in d['viewports']])"
python3 -c "import json;d=json.load(open('composition_dom_check.json'));print(len(d['checks']), all(c['ok'] for c in d['checks']))"
```

---

## 10. Gate tự đánh giá

| gate | kết quả | bằng chứng tôi đã đọc |
|---|---|---|
| **CIRCUS data** | **PASS** | 6/6 sha256 khớp `EXPECTED_SHA256`; counts 6003/4069/586, validated 368+218=586 giao = 0; gallery 74381 |
| **CIRCUS runs** | **PARTIAL** | 4/9 có số (§1); 4 bị F1 (§3); 1 CSMCIR (§5) |
| **Web composition** | **PASS** | giá trị thật render đúng, fixture bị loại mặc định, R@50 + provenance hiện |
| **Web native** | **PASS** | 2 CSMCIR certified, badge data-driven (chứng minh bằng intercept) |
| **Browser QA** | **PASS** | re-run trên stack live: `live/browser_qa.json` `ok: true`, 1440x900 + 390x844 đều `pass: true`, nodes 111, **overflow 0**, console/page/req/HTTP≥400 đều **0**; DOM check 16/16; sticky `partially_occluded: []` tại 4 vị trí scroll × 2 viewport. **Còn nợ:** harness đầy đủ 15 flow `flows.py` chưa chạy lại từ sau fix sticky |
| **Independent review** | **PASS** | 21/21 metric + 18/18 bin, abs diff 0.0 |
| **Test suite** | **PASS** | `test_composition_api.py` + `test_csmcir_aggregate_api_parity.py` = **15 passed**; `test_circus_resume_stable.py` = **9 passed**; `-k csmcir` = 41 passed. Ba test mới: `test_failed_run_exposes_is_failed_flag`, `test_failure_message_never_shows_python_none_sentinel`, `test_unresolved_category_reason_is_readable` |
| **Commit/push** | **NOT DONE** | HEAD vẫn `1716e81`, 8 modified + 9 untracked |

### Defect tìm thấy khi sửa test — đã fix: FAILED run được xếp như measurement

`workbench/backend/composition.py` chia tier chỉ theo `is_synthetic` và `smoke.limited`, nên HABIT
`e0dd04a46ef7a58334b9e353` — `execution_status: FAILED`, mọi metric null, **6003 unscored** do gate
Python — nằm cùng tier 0 với bốn kết quả thật, hiện lên bảng Composition với "—" ở mọi cột cạnh số
thật.

Đã sửa: thêm `is_failed_summary()` (`composition.py:190`) và tier **0 measurement / 1 failed-hoặc-
không-có-số / 2 smoke / 3 fixture**. Giá trị `execution_status` runner thực sự phát ra:
`COMPLETED`, `PARTIAL`, `REUSED`, `FAILED` (không có `ABORTED`).

Test mới `test_failed_runs_sort_after_scored_runs_regardless_of_run_id` dùng `tmp_path` với `run_id`
sắp xếp **ngược** để chứng minh tier chứ không phải byte order; revert-check xác nhận test fail khi
bỏ tier.

### Hai defect tìm thấy khi soát cell trên UI thật — đã fix

**(a) Dòng FAILED hiện `0.0%` coverage thay vì `—`.**

`CompositionView.tsx:220` render coverage vô điều kiện (`(m.coverage * 100).toFixed(1)`), trong khi
backend fallback cấp `0.0` cho run không scored (`metrics.py:162`: 0/6003 scored). Kết quả: HABIT
`execution_status: FAILED` trông như "đo được coverage 0" thay vì "không đo được gì" — trong khi sáu
cột metric khác đều đúng là `—`. Một con số 0 trông như dữ liệu, một dấu `—` trông như thiếu dữ liệu.

Đã sửa: backend expose `is_failed` (`backend/composition.py:138`, tái dùng `is_failed_summary` tại
`:191`) và UI blank coverage khi `is_failed`:

```tsx
{m.coverage != null && !r.is_failed ? `${(m.coverage * 100).toFixed(1)}%` : "—"}
```

Giá trị metric **không đổi**; dòng vẫn hiển thị (không bị ẩn khỏi cohort).

**(b) Sentinel Python `None` lọt vào text cho người dùng.**

`runner.py:235` in `category {query.category!r}`. Với 119 query category mơ hồ, `query.category` là
`None`, nên failure record ghi:

```
query 7:2179: category None has no adapter embedding scope (...)
```

Người đọc không phân biệt được "Python `None`" với "category tên là None". Đã sửa tại
`runner.py:235-236`:

```python
category = "unresolved" if query.category is None else repr(query.category)
```

Kèm `readable_failure_message()` (`backend/composition.py:199-208`) chuẩn hoá **357 record cũ** đã
persist trong `ranks/limn_base_iter0_all_categories_10173dc4…json`, vì chúng được ghi trước khi sửa
producer. Đo được: **357 record `category None` trong artifact; 0 lần `category None` trên DOM live**
(`live/composition_dom_check.json`, `unresolved` xuất hiện 9 lần). Số lần `None` **hiện trên DOM
trước** khi sửa: **CHƯA XÁC MINH** — không có ảnh/DOM snapshot nào chụp trước bản fix. Marker
`has no adapter embedding scope` **giữ nguyên** nên khoá phân loại resume không đổi.

---

## 11. Việc tiếp theo, xếp theo giá trị

**Trạng thái hiện tại: chưa commit gì trong round này — cố ý, không phải bỏ sót.** Bạn yêu cầu báo
cáo, không yêu cầu commit; vì vậy toàn bộ change set vẫn nằm nguyên trong working tree để review
được (`git status` vẫn là 8 modified + 9 untracked). Việc commit là hành động đề xuất ở mục 1, chưa
phải việc đã làm — nên §10 ghi `Commit/push: NOT DONE` là đúng.

1. **Commit + push ngay.** Đây là rủi ro lớn nhất hiện tại: 8 file sửa + toàn bộ bundle review
   chưa được bảo vệ. Một `git checkout` nhầm là mất hết.
2. **Provision Python 3.9 prefix** → mở khoá habit/intent/hint/conesep (4 model, ~1–2 h setup +
   ~40 phút/model GPU).
3. **Quyết gap `variant_provenance`** — patch runner ghi `sm_120`, hay chỉ ghi trong docs.
4. **Re-run harness đầy đủ 15 flow `flows.py`** — phần targeted trên stack live đã xong
   (`browser_qa.py` + DOM check + sticky đo riêng), nhưng bộ 15 flow chưa chạy lại từ sau fix sticky.
5. **Viết lại 357 chuỗi `query_failures` legacy đã persist** trong
   `runs/limn_base_iter0_all_categories_10173dc4…/ranks/*.json` nếu cần artifact sạch byte-for-byte.
   Hiện chúng chỉ được chuẩn hoá **lúc đọc** (`readable_failure_message`), file trên đĩa vẫn còn
   `category None`.
6. **Sau đó** mới tới F1/F7 code fixes như defect thật.

---

## Phụ lục — mọi con số trong báo cáo này đọc từ

| mục | nguồn |
|---|---|
| §1 mọi metric | `workbench/artifacts/composition/summary/<dir>/{full,sf,validated}.json` qua `json.load` |
| §2 trạng thái model | `workbench/artifacts/composition/cohort_status.txt` |
| §3 F1 file:line | `workbench/composition/model_*.py`, `workbench/backend/runtime.py:1644-1649` |
| §3 không bypass | grep `runtime.py` = 0 match |
| §3 không py3.9 | `workbench/envs/*/bin/python -V` |
| §4 sm_120 gap | `json.dumps(summary).count("sm_120")` = 0 |
| §5 caption | `data/FashionIQ/qwen_captions/*_cot_val.json` |
| §5 CSMCIR native | `workbench/artifacts/reports/csmcir_fashioniq_*_f03887d2_aggregate.json` |
| §6 web | `CompositionView.tsx`, `App.tsx`, `BenchmarkView.tsx`, `style.css`, `backend/composition.py` |
| §6 test | `pytest` chạy lại trong session này |
| §7 QA | `browser_qa/flows.json` + `screens/` |
| §8 review | `REVIEW_ENCODER.md` |
| §9 evidence | `EVIDENCE_INDEX.json` (gốc bundle), verify 212/212 |
| §0 URL/build | `curl -s http://127.0.0.1:5173/` vs `workbench/frontend/dist/index.html` |
| §7 QA live | `live/browser_qa.json`, `live/composition_dom_check.json`, `live/*.png` |
| §10 defect (a) | `CompositionView.tsx:220`, `backend/composition.py:138`, `metrics.py:162` |
| §10 defect (b) | `runner.py:235-236`, `backend/composition.py:199-208`, 357 record trong `runs/limn…/ranks/*.json` |
