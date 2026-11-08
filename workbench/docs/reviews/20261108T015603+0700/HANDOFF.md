# HANDOFF — round 20261108T015603+0700

> **Vòng này CHƯA XONG. KHÔNG có gì được commit hay push.** Toàn bộ thay đổi còn nằm ở working tree
> trên branch `research/cir-failure-workbench`, HEAD `1716e81`.

Tài liệu này ghi lại trạng thái đo được tại thời điểm bàn giao. Số nào không kiểm được từ file thì ghi
`CHƯA XÁC MINH`, không suy đoán.

---

## 1. Trạng thái hiện tại

### Git

```
branch   research/cir-failure-workbench
HEAD     1716e81
commit   KHÔNG
```

`git status --short`:

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

8 file sửa + 9 mục chưa track (trong đó 2 thư mục review cũ và 1 bundle mới của vòng này).

### Web

Tại lúc viết file này, port 8000/5173 **đang được một worker KHÁC chạy** (start 02:23:32, tiến trình
`serve_workbench.py` pid 3058639). Tôi đã free port của mình xong trước đó; cái đang chạy không phải
của tôi nên **không kill**. Cả hai endpoint trả 200.

Lệnh khởi động lại khi cần:

```bash
cd /home/heheboiz/data/cir
WORKBENCH_CIRCUS_RELEASE_ROOT=/home/heheboiz/data/cir/workbench/artifacts/circus_release \
workbench/envs/ctl312h/bin/python workbench/scripts/serve_workbench.py --production-frontend \
  --host 127.0.0.1 --backend-port 8000 --frontend-port 5173
```

- URL: `http://127.0.0.1:5173`
- Rebuild frontend sau khi sửa source: `npm run build --prefix workbench/frontend`
- `vite preview` chạy một mình sẽ làm `/api` trả 404 vì `WORKBENCH_BACKEND_URL` chỉ được
  `serve_workbench.py` set. Đó là proxy, không phải bug app.

### Model config

`~/.omp/agent/config.yml` — quota gemini đã hồi:

```yaml
perCallModel: "on"
perCallModelAllow:
  - 9router/ag/gemini-3.8-flash-high
  - 9router/wdb/deepseek-ai/DeepSeek-V4.1-Flash-1017
agentModelOverrides:
  task: 9router/ag/gemini-3.8-flash-high
  sonic: 9router/ag/gemini-3.8-flash-high
```

DeepSeek vẫn nằm trong `perCallModelAllow` làm fallback.

---

## 2. Dữ liệu CIRCUS — XONG

6 file pinned tại `workbench/artifacts/circus_release/`, **tất cả SHA-256 khớp** `EXPECTED_SHA256`
trong `workbench/composition/data.py:19`. Kiểm bằng `cd.verify_release_hashes()`: 6/6 `expected == actual`.

| file | bytes | sha256 |
|---|---:|---|
| `mbeir_fashioniq_task7_test.jsonl` | 1926189 | `b601ef6a5411bf9385cff41e64bea863871ab1094cd09d78683de6a5141165ef` |
| `mbeir_fashioniq_task7_cand_pool.jsonl` | 11889854 | `121ec284832c33289ee13bfac19d5d59dc3f8aeddf672d784fd5eb8bcdf79eef` |
| `shortcut_free_subsets/fashioniq/shortcut_free/mbeir_fashioniq_task7_test_multimodal.jsonl` | 1298794 | `95a93883334e351e4d3aaaf25e47e89b48d4daf02bdb786e8f00d58165ace37d` |
| `final_dataset/query_jsonl/fashioniq/validated_solved/mbeir_fashioniq_task7_test_validated_solved.jsonl` | 118644 | `18f829da4d3b1ee8439f43998bae43dbbe8924c2eaafba84473ae849f564f5c7` |
| `final_dataset/query_jsonl/fashioniq/validated_unsolved/mbeir_fashioniq_task7_test_validated_unsolved.jsonl` | 70356 | `ba019a30d3b45f4b29c34d2d48638dbeeed2e2a42409f625922c65de5b2a7efa` |
| `shortcut_audit/retrieval_data_fashioniq_task7_aggregated_best_rank_mm_any.json` | 19404614 | `27bdbef041200dc393b5478d3e57121bc16c70a9acc2e98a19fb48408a99533c` |

Revision CIRCUS pinned: `b7ed55104abd987e60f7099037e4cd8a69966428`
(`workbench/docs/CIRCUS_COMPOSITION_SPEC.md:27`). Repo là `EdinburghNLP/CIRCUS` — link tác giả trong
paper redirect sang đó.

Đếm dòng, kiểm bằng `data.load_release()`:

| đại lượng | giá trị |
|---|---:|
| full | 6003 |
| sf | 4069 |
| validated | 586 |
| validated_solved | 368 |
| validated_unsolved | 218 |
| gallery (`candidate_names`) | 74381 |

`validated ⊆ sf ⊆ full`: **True** (so sánh theo qid). `368 + 218 = 586` khớp.
`validated_solved ∩ validated_unsolved = 0`, union = 586: **đã kiểm** (đọc qid từ 2 file jsonl).
Gallery identity sha256 `1db68c59d96f014cdbecf936e108653de98abef0bd9a98a8abc2fe08b5dec4b4`
(đọc từ `summary/encoder_.../full.json`). Category mơ hồ: **119** query.
`--prepare-data` exit 0: **CHƯA XÁC MINH** trong round này (không chạy lại).

---

## 3. Composition — 1 model xong

Run ENCODER `6a528e877c328772e2c7bb28`, `is_synthetic: false`, gallery 74,381, cả 3 condition
`COMPLETED`. Nguồn: `workbench/artifacts/composition/summary/encoder_fashioniq_6a528e877c328772e2c7bb28/{full,sf,validated}.json`.

| subset | queries | MM R@10 | MM R@50 | MM nDCG | text nDCG | image nDCG | gap | coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 6003 | 31.0178 | 49.5752 | 0.29586 | 0.15613 | 0.10728 | 0.47230 | 1.0 |
| sf | 4069 | 18.6778 | 36.3234 | 0.21886 | 0.10775 | 0.09765 | 0.50768 | 1.0 |
| validated | 586 | 34.6416 | 55.2901 | 0.29646 | 0.11611 | 0.10776 | 0.60835 | 1.0 |

### CẢNH BÁO bắt buộc đọc kèm bảng trên

Run này **protocol đã chạy; environment pin KHÔNG thoả**. Đọc từ `full.json`:

```
runtime.model_interpreter.verification_status : PYTHON_VERSION_MISMATCH
runtime.model_interpreter.verified            : false
runtime.model_interpreter.verification_reason : expected Python 3.9, found 3.11.15
runtime.model_interpreter.version             : Python 3.11.15; torch 2.7.1+cu128
runtime.model_interpreter.path                : workbench/envs/pair-adapted/bin/python
reproduction_status                           : METRICS_MEASURED_MAPPING_UNVERIFIED
checkpoint_mapping_status                     : VERIFIED_METADATA
checkpoint_training_noise_pct                 : unknown
hash_verification                             : matched
```

Registry pin (`workbench/registry/models.yaml:697`) là Python `3.9` / torch `1.13.1` /
torchvision `0.14.1`. **Không được gọi đây là "protocol run sạch".**

### Run SMOKE

`eda30a37c0bc17ac43844f09` là slice SMOKE 8 query (`smoke.limited`), bị ẩn mặc định
(`include_smoke=true` mới hiện). **Không bao giờ trích nó như kết quả đầy đủ.**

---

## 4. Trạng thái 9 model

| model | trạng thái | chi tiết |
|---|---|---|
| encoder | **FULL** | mục 3, đủ 3 subset |
| limn | **PARTIAL** | run `limn_base_iter0_all_categories_10173dc45dea8c90fd2a284b`, `smoke.limited: false`, `is_synthetic: false`, gallery 74,381 (identity sha256 khớp pin `1db68c59…`). **6003 query, chỉ 5884 scored → coverage 0.98018**; 119 query không scored vì category mơ hồ (qid kiểu `7:2179` thuộc cả `shirt` lẫn `toptee`) — LIMN có 3 checkpoint theo category nên **không được tự gán**, đó là hành vi đúng, không phải lỗi. `category_scope_failures`: 119 ở cả 3 condition. Chỉ số full: MM nDCG **0.29084**, R@10 **30.0646**, R@50 **48.7593**; text nDCG **0.13328**; image nDCG **0.09970**, R@10 2.4983, R@50 6.9850; gap **0.54174**, cohort 5884. `execution_status: PARTIAL`, `reproduction_status: BLOCKED_RUNTIME_PREREQUISITES`, `category_scopes: [dress, shirt, toptee]`. **Environment caveat: `DEPENDENCY_VERSION_MISMATCH`, `verified: false`, Python 3.11.15 / torch 2.7.1+cu128.** Subset khác: sf 4069/4000 scored (0.98304), validated 586/581 (0.99147), cũng `PARTIAL`. |
| habit | NOT STARTED | checkpoint staged, 1 file |
| tme | NOT STARTED | checkpoint staged, 1 file |
| intent | NOT STARTED | checkpoint staged, 1 file |
| hint | NOT STARTED | checkpoint staged, 1 file |
| conesep | NOT STARTED | checkpoint staged, 1 file |
| ptha_mtst | NOT STARTED | checkpoint staged, 1 file |
| csmcir | **BLOCKED** | 2 lý do độc lập, mục 5 |

6 model NOT STARTED đã preflight (checkpoint tại
`checkpoint_selection/20261106T002617+0700/cohort/downloads/<model>/`, SHA-256 verified, interpreter
import OK). Chưa chạy subset nào.

**Chưa có bảng đủ 9 model.**

---

## 5. CSMCIR — blocker chính xác

Hai lý do **độc lập**, cả hai đã đo:

### 5.1 Contract scoring khác nhau

Adapter `score()` = text embedding 256-d nở ×32 thành vector 8192-d rồi dot product.
Native `inference()` = attention-weighted token importance, sau đó max trên 32 query token của matmul
1×1×256. **Hai đại lượng khác nhau** — thay cái này bằng cái kia là đổi thuật toán.

Nguồn: `workbench/composition/model_csmcir.py:7-9` (docstring) và `:53-59` (`_TARGET_REASON`);
`CHECKPOINT_SELECTION_REPORT.md` mục 3.23.6 (dòng 1491).

### 5.2 Caption chỉ phủ 20.72%

`data/FashionIQ/qwen_captions/{dress,shirt,toptee}_cot_val.json`:

| file | entries | bytes |
|---|---:|---:|
| `dress_cot_val.json` | 3817 | 2817386 |
| `shirt_cot_val.json` | 6346 | 5015979 |
| `toptee_cot_val.json` | 5373 | 3929303 |
| **tổng** | **15536** | |
| **distinct** | **15415** | |

`15415 / 74381 = 20.72%` gallery. **58,966 ảnh không có caption.** Chỉ phủ split validation của
FashionIQ.

### ĐÃ RÚT LẠI (RETRACTED) — từng nằm trong code và trong reply của vòng này

1. ~~"caption `COT_ours2` không có"~~ — **SAI**. Chúng được publish ở revision Hub đã pin với sha256
   khớp. `CHECKPOINT_SELECTION_REPORT.md` 3.18 (dòng 865); D-8 đã đóng.
2. ~~"checkout pinned không import được / chỉ ship bytecode"~~ — **SAI**. `.pyc` của derived checkout
   import được và native run hoàn tất `rc 0`.

Chuỗi lý do trong `model_csmcir.py` đã được sửa trong vòng này để chỉ nêu đúng 2 lý do thật.
**41 test csmcir pass.**

### Bằng chứng NATIVE của CSMCIR là thật và tách riêng

Run `f03887d2` (`20261107T054933277898Z_csmcir_fashioniq_f03887d2`), `is_synthetic: false`,
`parity_status: PARITY_VERIFIED`, `reproduction_status: EXTRACTED_VERIFIED`,
`invocation_outcome: return_code 0 / INVOCATION_COMPLETED / PROOF_VALIDATED`.

| chỉ số | giá trị |
|---|---:|
| macro R@10 | 57.05073873202006 |
| macro R@50 | 77.29859352111816 |
| macro mean | 67.1747 |
| dress R@10 / R@50 | 52.20624804496765 / 74.86366033554077 |
| shirt R@10 / R@50 | 57.75269865989685 / 76.1040210723877 |
| toptee R@10 / R@50 | 61.19326949119568 / 80.92809915542603 |

Nhãn đúng: **"public-caption path-mapped research run"**, không phải official reproduction
(`docs/reviews/checkpoint_selection/20261107T021500+0700/README.md:92`).

> Lưu ý kiểm chứng: file `artifacts/reports/...f03887d2_aggregate.json` **không** có key
> `verification`; trạng thái certified do backend tổng hợp. API `/api/benchmark-runs` trả
> `verification.state = certified`, `certified: true` cho cả 2 run CSMCIR — đó là nguồn đã kiểm.

---

## 6. Native side của web

- Chỉ CSMCIR có bằng chứng native thật: **2 aggregate run certified** (f564570e, f03887d2).
- `workbench/artifacts/results/` **RỖNG** (chỉ `.gitkeep`) → `GET /api/runs` trả `[]`.
  **Đúng, không phải bug.** Không tồn tại artifact per-query schema-v2 nào.
- `GET /api/benchmark-runs` trả 2 run CSMCIR.
- Web tách measured vs paper sanity, đánh dấu aggregate-only, disable per-query action kèm lý do.
- Parity test `workbench/tests/test_csmcir_aggregate_api_parity.py` khẳng định API == bytes artifact
  thô.

---

## 7. Việc web/UI trong vòng này

- **Composition API read-only**: `GET /api/composition/runs[?subset=&include_fixtures=&include_smoke=]`,
  `/runs/{run_id}`, `/runs/{run_id}/queries` (phân trang + filter). Reader:
  `workbench/backend/composition.py`.
- **Chuẩn hoá nested → flat** để run thật render được (trước đó mọi cột ra "—" vì UI đọc key phẳng
  còn runner ghi nested).
- **Fixture mặc định bị loại, sort dưới measurement.** Trước đó thứ tự byte đặt `SYNTHETIC_*` lên trên
  `encoder_*` nên dòng giả mở đầu bảng. Nay tier 0 measurement / 1 smoke / 2 fixture, rồi run_id.
  Fixture vẫn lấy được bằng `include_fixtures=true`.
  Kiểm bằng data layer: default mỗi subset = **1 run** (`6a528e...`); opt-in = **4 run**
  (`6a528e...`, `eda30a...`, 2× `SYNTHETIC_*`), thứ tự đo trước giả sau.
- **Đếm model trên dashboard** đã sửa (trước render em-dash).

Test:

| lệnh | kết quả |
|---|---|
| `pytest workbench/tests/test_composition_api.py` | **8 passed** |
| `pytest test_composition_api.py test_csmcir_aggregate_api_parity.py` | **10 passed** |
| `pytest workbench/tests -k csmcir` | **41 passed**, 2219 deselected |

Frontend build pass (`tsc -b && vite build`, 24 modules).

---

## 8. Browser QA — Chromium thật, dữ liệu thật

Playwright **1.64.1** trong `workbench/envs/ctl312h`; managed Chromium tại
`workbench/envs/ctl312h/.playwright-browsers` (**681 MB**). Không có OMP browser facade trong session
này — đã kiểm.

Bundle: `workbench/docs/reviews/20261108T015603+0700/browser_qa/` — **36 MB**, **165 PNG**
(146 trong `screens/`, 19 trong các thư mục harness), `flows.json`, `flows.py`, `zooms.py`, JSON +
screenshot từng trang, `README.md`.

### Kết quả: 13/15 flow PASS

| viewport | benchmark_native | composition_switch | composition_drilldown | rapid_filter | mobile_layout |
|---|---|---|---|---|---|
| 1440x900 | PASS 41/41 | PASS 48/48 | PASS 12/12 | PASS 8/8 | PASS 8/8 |
| 768x1024 | PASS 41/41 | PASS 48/48 | PASS 12/12 | PASS 8/8 | **FAIL 7/8** |
| 390x844 | PASS 41/41 | PASS 48/48 | PASS 12/12 | PASS 8/8 | **FAIL 7/8** |

Console error **0**, page error **0**, failed request **0**, HTTP ≥400 **0** — trong cả 15 flow, cả 3
viewport. Overflow ngang toàn trang: **0 px** ở cả 3.

Harness sweep phụ (`harness_*.json`, đều `ok: true`):

| trang | nodes | overflow px | console/req/http | ảnh | broken |
|---|---:|---:|---|---:|---:|
| Benchmark | 405 | 0 | 0/0/0 | 0 | 0 |
| Composition | 756 | 0 | 0/0/0 | 40 | 0 |
| Drill-down (validated) | 756 | 0 | 0/0/0 | 40 | 0 |
| Dashboard | 106 | 0 | 0/0/0 | 0 | 0 |
| Evaluation Runs | 278 | 0 | 0/0/0 | 0 | 0 |

### Giá trị thật khớp trong UI

Cả 3 subset: mọi ô số render trong bảng **khớp byte-for-byte** với JSON API của subset đang chọn
(so sánh bằng so khớp chuỗi đã format, không bằng hằng số trong test).
Drill-down: **40 ảnh** giải mã `naturalWidth > 0`, **0 broken**; rank thật `#50,729 / #19,161 / #37,597`.
Rank thiếu: query chưa có rank render `unscored` tường minh, **không bao giờ ra 0 hay #0**.
Rapid switching: 4 đợt 3–5 click; sau mỗi đợt bảng render == JSON của subset được chọn cuối.

### Defect còn lại

**1. CONFIRMED — cột Model trôi khỏi màn hình trên máy hẹp (768 và 390)**

Bảng composition rộng 1193 px trong wrapper 366 px. Cột đầu là `position: static`, nên khi scroll
phải ô Model chạy tới **x = −827** (390, scroll 853 px) và **x = −461** (768, scroll 499 px);
`visibleAfter: false`. Người đọc thấy `12.5% | 0.1176` mà không biết thuộc run nào.

Ảnh: `screens/zoom_table_scrolled_right_390x844.png`,
`screens/zoom_table_scrolled_right_768x1024.png`, `screens/mobile_composition_390x844.png`.

**Chỉ báo cáo, không sửa.** Fix đề xuất `position: sticky; left: 0` cho cột đầu — đã thử cục bộ
(ô Model đứng ở x=13 sau khi scroll hết) rồi **revert**, vì `style.css` thuộc worker khác.

**2. COSMETIC — first paint dashboard hiện `— registered models`**

Khoảng 300 ms trước khi `/api/models` xong, sau đó `17 registered models`. Trạng thái cuối đúng
(đã so với độ dài `/api/models` = 17). Ghi nhận, không gate.

### Ghi chú về harness

**18 assertion cũ đã lỗi thời**: chúng đòi dòng fixture phải hiện **mặc định**, mâu thuẫn chính sách
mới (`include_fixtures=false`). Đã sửa **harness**, không sửa app. Nhánh assertion hiện tại:

- **Đường mặc định**: bảng == danh sách API mặc định, không badge `FIXTURE`, không chữ `SYNTHETIC_`.
- **Đường opt-in** (`include_fixtures=true`): 2 dòng fixture hiện, cả hai có badge `FIXTURE`, tên
  `SYNTHETIC_*`, số khác mọi dòng đo, run encoder thật đứng **đầu**, mọi dòng đo đứng trước mọi dòng
  giả.

---

## 9. Chưa xong / bước tiếp theo

- **LIMN full `10173dc45dea8c90fd2a284b` ĐÃ XONG — trạng thái PARTIAL** (`smoke.limited: false`).
  Summary `summary/limn_base_iter0_all_categories_10173dc45dea8c90fd2a284b/{full,sf,validated}.json`:
  6003 query nhưng **5884 scored → coverage 0.98018**; **119 query không scored** vì category mơ hồ
  (qid kiểu `7:2179` thuộc cả `shirt` và `toptee`) — LIMN có 3 checkpoint theo category nên **không tự
  gán category**, đây là hành vi đúng, không phải lỗi. MM nDCG **0.29084**, R@10 **30.0646**,
  R@50 **48.7593**; text nDCG 0.13328; image nDCG 0.09970; gap **0.54174**, cohort 5884.
  `reproduction_status: BLOCKED_RUNTIME_PREREQUISITES`; interpreter `DEPENDENCY_VERSION_MISMATCH`
  (Python 3.11.15 / torch 2.7.1+cu128). Chi tiết đầy đủ ở bảng §4.
- **LIMN smoke `3f86f58483b368377e2c151e` vẫn là smoke 8 query, không đổi** — `smoke.limited: true`,
  MM nDCG 0.2714, R@10 25.0%, R@50 37.5%, gap 0.5214. Không bao giờ trích như kết quả đầy đủ;
  opt-in bằng `include_smoke=true`.
- 6 model chưa start, **chưa có bảng đủ 9 model**.
- **Cohort execution đã bắt đầu** (worker song song đang chạy 6 model còn lại). Tiến độ live:
  `workbench/artifacts/composition/cohort_status.txt`.
  > Lưu ý lúc ghi file này: `cohort_status.txt` **chưa tồn tại** và chưa thấy tiến trình cohort nào
  > đang chạy (`ps` không có `composition_eval` / `model_*` worker). Nếu file vẫn vắng khi bạn đọc,
  > nghĩa là worker đó chưa tới bước ghi trạng thái — kiểm lại bằng `ps` trước khi kết luận.
- **Fix sticky column mobile — ĐÃ XONG** (pin cột Model/Run, nền đục, `scroll-padding-left: 157px`;
  đo `partially_occluded: []` tại 25/50/75/100% max scroll trên 390x844 và 768x1024; overflow trang 0).
  Bằng chứng ở §11.F; ảnh `browser_qa/screens/fixed_sticky_maxscroll_390x844.png`,
  `..._768x1024.png`.
- **Chưa commit, chưa push** gì trong vòng này.
- **Quyết định cần bạn chọn**: để LIMN xong rồi chạy 6 model còn lại tuần tự, mỗi model **một lệnh**
  `--subsets full,sf,validated` kèm `--resume`; hoặc cắt cohort.
  Encode gallery 74,381 chiếm phần lớn thời gian (~40 phút/model; LIMN ×3 scope).
- **Không nên kỳ vọng câu chuyện environment-pin sạch.** Cohort chạy trên venv adapted có Python/torch
  khác registry pin. Hoặc chấp nhận và gắn nhãn từng dòng "protocol executed; environment pin not
  satisfied", hoặc provision env pinned trước.
- **CSMCIR ở lại BLOCKED trên CIRCUS.** Kết quả native đã công bố. Không có gì để sửa nếu không có
  contract scorer mới hoặc nguồn caption phủ toàn gallery.

---

## 10. Đường dẫn artifact

```
workbench/docs/reviews/20261108T015603+0700/
  HANDOFF.md                     file này
  browser_qa/
    README.md                    mô tả bundle + lệnh chạy
    flows.json                   output flow, 13/15 pass, chi tiết từng check
    flows.py                     driver flow (tái dùng browser_qa.py)
    zooms.py                     chụp element 2x DSF
    harness_benchmark.json       \  output browser_qa.py,
    harness_composition.json      |  3 viewport mỗi file
    harness_dashboard.json        |
    harness_evaluation_runs.json  |
    harness_drilldown.json       /
    harness_*/                   screenshot từ các sweep trên
    benchmark_initial/           sweep khám phá ban đầu
    comp_explore/                render composition khám phá
    screens/                     146 PNG: <flow>_<viewport>.png, _full.png, zoom_*.png
```

---

## 11. Kiểm chứng độc lập — ENCODER (nguồn: `REVIEW_ENCODER.md`, 363 dòng)

Reviewer độc lập, read-only. Tính tay toàn bộ (`math.log2`, vòng lặp dict thuần), **không** import
helper nào từ `workbench.composition.metrics`. Tôi đọc báo cáo và đối chiếu các vị trí code được dẫn.

### 11.A Kết quả kiểm chứng số — ĐẠT

Reviewer tính lại từ rank artifact thô `workbench/artifacts/composition/runs/6a528e877c328772e2c7bb28/ranks/encoder_fashioniq.json`:

| hạng mục | kết quả |
|---|---|
| ô metric | **21/21 khớp**, abs diff **0.0** (so `float` bằng `==`, không làm tròn) |
| shortcut bin K=10 | **18/18 khớp** (6 bin × 3 subset) |
| gallery, release pin, checkpoint hash, source hash, run-identity binding | tái tạo độc lập, **khớp exact** |
| cross-check artifact thứ hai | 18 file per-query jsonl (3 subset × 3 condition × ...) khớp rank table **mọi qid, 0 mismatch** |

Chi tiết đáng ghi: `runtime_identity` (chứa cả `verification_status`) nằm **trong** digest run-id;
reviewer tính lại `sha256(json(identity, sort_keys))` ra `6a528e877c328772e2c7bb278...` = `fingerprint`
= `resume_key`, `[:24]` = run_id. Nghĩa là đổi environment sẽ tạo run id khác → **so sánh với run
tương lai vẫn an toàn**.

Lưu ý định nghĩa: summary dùng `ndcg_full` (toàn 74,381), không phải nDCG@K. Reviewer đã kiểm độ nhạy:
nếu dùng nDCG@10 thì gap là 0.7378/0.9557/0.9624; dùng R@10 thì 0.7234/0.9447/0.9458. So 0.4723 với
một con số "nDCG@10 gap" là **so sai định nghĩa**, không phải summary sai.

**Kết luận A: số ENCODER đúng như đã công bố.**

### 11.B Bảy defect thật

| id | mức độ | mô tả | chứng cứ file:line |
|---|---|---|---|
| F1 | REAL | Runner composition chạy một environment mà chính verifier của nó đánh BLOCKED; bước kiểm dependency **không bao giờ chạy** (`report.packages == {}`, `content_verified == False`, `validated_environment_fingerprint is None`) | `backend/runtime.py:1645-1652` return sớm khi `python_ok is False`; cửa hậu ở `composition/model_encoder.py:65-105` + `scripts/composition_eval.py:64-78,278-290` không kiểm Python version |
| F2 | REAL (dormant) | Dashboard render `CERTIFIED` **vô điều kiện**, bỏ qua receipt `verification` mà backend có phục vụ. `BenchmarkView` làm đúng | `frontend/App.tsx:69` vs `backend/main.py:790,847,856` (`_gate_fields`) vs `frontend/BenchmarkView.tsx:32,56-57` |
| F3 | REAL | `PASS_WITHIN_5_PERCENT_VARIANT` + lý do "variant" không hiện nơi có số (chỉ trong `<details>` đóng). Hai run CSMCIR là **cùng một phép đo** (metrics giống hệt, khác `run_id`/`environment_digest`/`evaluator_environment_knobs`) nhưng hiện thành **2 card "2 certified research results"** | `frontend/BenchmarkView.tsx:163-166`; `backend/paper_gate.py:403`; `scripts/evaluate_models.py:161-170` |
| F4 | REAL | `GET /api/composition/runs/{run_id}/queries` trả **HTTP 200 + `total: 6003`** cho run id **không tồn tại**; đúng ra phải 404. Endpoint chi tiết thì đúng | `backend/main.py:948-975` (queries route không kiểm tồn tại) vs `:940-946` (detail có 404); `backend/composition.py:264-375` |
| F5 | REAL | Hai summary `SYNTHETIC_*` tự mâu thuẫn: gap khai 0.2856 vs suy ra 0.4724 (lệch 0.1868); bins cộng **6001** và **5881** ≠ 6003; coverage 0.98 vs `common_cohort_count` 6003; `smoke.query_limit` 8 vs `query_count` 6003. Không tới được UI (fixture opt-in) | `artifacts/composition/summary/SYNTHETIC_*/full.json`; literal hardcode `backend/composition.py:400-460`; UI không truyền opt-in ở `frontend/CompositionView.tsx:73` |
| F7 | REAL (gốc rễ) | `reference_eligible` được **assert**, không **đo**: `getattr(..., True)` rồi ghi literal `True`. Đúng trên thực tế cho run này (6003/6003 reference trong gallery, 0 mask) nhưng không suy ra từ ranking | `composition/runner.py:707-708` (getattr default True), `:748`, `:1004` (literal `True`) |
| F8 | REAL (UI) | **R@50 không render ở đâu** trong Composition view dù API đã chuẩn hoá | `frontend/CompositionView.tsx` không có `recall_at_50`; `types.ts:17` khai nhưng không dùng |

### 11.C Lỗ hổng tài liệu

| id | mức độ | mô tả |
|---|---|---|
| F9 | DOC GAP | `docs/CIRCUS_COMPOSITION_SPEC.md:99` chỉ dẫn **2 smoke run không tồn tại trên đĩa** (`1410522b57d4fe79083c60a6`, `17ef444e30b829ac22f2d3c8` — không có `summary/` lẫn `runs/`), và **không nhắc** run full thật `6a528e877c328772e2c7bb28` (0 hit trong `docs/`). Spec vẫn ghi ENCODER "TARGET_GPU_PENDING" trong khi đã có full 6003 query, coverage 100% |
| F10 | DOC GAP | `reproduction_status: METRICS_MEASURED_MAPPING_UNVERIFIED` và `runtime.model_interpreter.verification_status` **không hiện cạnh con số**: grep trong `CompositionView.tsx` → 0 match. Trái với spec:97 ("checkpoint-mapping uncertainty remains visible") |

### 11.D Được đánh giá ACCEPTABLE / UNVERIFIABLE

| id | phán xét |
|---|---|
| F6 | **ACCEPTABLE** — run LIMN đang chạy không thể bị phục vụ như đã xong: không có `summary/`, không có `ranks/`, 404 ở endpoint chi tiết, không nằm trong danh sách |
| F11 | **ACCEPTABLE (caveat diễn giải)** — gap bị pha loãng bởi sàn ngẫu nhiên `E[1/log2(r+1)] = 0.06874`: image-only chỉ 1.42–1.57× sàn, nên gap 0.4723 (full) chứa nhiều "nhiễu trên nền sàn"; gap hiệu chỉnh sàn là 0.6153 / 0.7402 / 0.7920. Bootstrap 2000 mẫu: full `[0.4608,0.4848]`, sf `[0.4942,0.5213]`, validated `[0.5820,0.6320]` — chênh lệch giữa subset nằm ngoài nhiễu |
| F12 | **ACCEPTABLE** — protocol integrity đạt: 6 release hash, `gallery_identity_sha256 1db68c59…`, checkpoint `0f31c501…` (631,879,365 B), adapter + 5 method hash, rank exact trên toàn 74,381 (max rank 74,147 < 74,381, không bão hoà cutoff) |
| F13 | **ACCEPTABLE** — paper nói 18,853 ảnh, release 74,381; summary **tự khai** trong `paper_vs_release_gallery`. Không thể so trực tiếp với R@10 56.13 của paper (protocol khác: `fashioniq_val_split`) |
| F14 | **UNVERIFIABLE** — không chạy lại GPU: rank được verify **số học**, không tái tạo từ model. Byte-exact site-packages và provenance commit GitHub của upstream không kiểm được (thư mục `third_party/ENCODER` không phải repo git riêng) |

### 11.E Phán xét `PYTHON_VERSION_MISMATCH` (trích đúng)

**KHÔNG invalidate:** số học rank và mọi metric (mục 11.A); gallery 74,381 đúng thứ tự pinned;
checkpoint byte; source hash; và **tính định danh run** — `runtime_identity` nằm trong digest run-id
và đã tính lại khớp.

**CÓ invalidate:** mọi claim rằng run tuân thủ **contract runtime đã khai** (không đạt contract lịch sử
`python 3.9`/`torch 1.13.1`, cũng không đạt contract adapted đã khai `python 3.10`/`torch ==2.7.1`);
mọi claim rằng **dependency đã được verify**; và biến so sánh với paper thành **so sánh variant đã
khai**, không phải reproduction chính thức.

Một câu của reviewer: **"run là một phép đo thật; nó không phải một reproduction tuân thủ contract."**
Chạy được cho thấy gate đường composition (F1) không thi hành kết quả của verifier.

### 11.F Trạng thái fix — 5 mục ĐÃ LANDED và ĐÃ VERIFY; 4 mục để mở

#### Đã fix và đã kiểm (đọc trực tiếp source + 11 test pass)

| id | đã fix | chứng cứ |
|---|---|---|
| **F2** — chỉ site Dashboard | Dashboard **không còn** hardcode badge. `App.tsx:69` gọi `isCertifiedEvidence(b)` và render state thật từ receipt API. **Đây là fix theo TỪNG SITE, không phải fix toàn bộ defect** — xem **F15**: Evaluation Runs (`App.tsx:71`) vẫn hardcode và chỉ được sửa sau đó | `frontend/App.tsx:69` (`isCertifiedEvidence(b)`); ảnh `composition/dashboard_1440x900.png` — badge CERTIFIED giờ do API quyết định. Bằng chứng chỉ phủ dòng `:69`; **không** suy ra được trạng thái của `:71` từ ảnh dashboard |
| **F3** (một phần) | Run CSMCIR trùng (knobs `null`) mang note amber inline **"duplicate attempt — provenance knobs missing"**; headline đổi thành **"2 certified research results (1 unique measurement)"** | `frontend/App.tsx:69` (`isDuplicateWeak=!b.evaluator_environment_knobs`, hai chuỗi trên); cùng ảnh dashboard |
| **F4** | `GET /api/composition/runs/{run_id}/queries` kiểm tồn tại → **404 `composition_run_not_found`** cho id giả, 200 cho run thật | `backend/main.py:965` (raise 404 trong route queries; `:944` là route detail); test mới trong `workbench/tests/test_composition_api.py` |
| **F8** | Bảng Composition render cột **"MM R@50"** (49.6% cho encoder), tabular numerals | `frontend/CompositionView.tsx:174` (`<th>MM R@50`) + `:211` (`formatRecall(m.recall_at_50_multimodal)`) |
| **F10** | API record phẳng expose `reproduction_status` + `interpreter_verification_status`; dòng bảng và drill-down hiện **"env: PYTHON_VERSION_MISMATCH · METRICS_MEASURED_MAPPING_UNVERIFIED"** | `backend/composition.py:139-146` (normalise); `frontend/CompositionView.tsx:198` (dòng bảng) + `:251` (drill-down); ảnh `composition/390x844_table.png` |

**Bonus — mobile sticky column (defect #1 của browser QA) đã fix:** cột Model/Run được pin với nền
đục (`th` rgb(22,27,34), `td` rgb(28,33,40)), `z-index` 10/5, `scroll-padding-left: 157px` + scroll
snapping. Đo bất biến tại 25/50/75/100% của max scroll trên **cả** 390x844 và 768x1024:
`partially_occluded: []` — không header hay body text nào bị cắt bởi mép sticky.
`style.css:77-105`. Overflow ngang toàn trang = 0 ở cả 3 width; desktop 1440 không scroll ngang.
Ảnh: `browser_qa/screens/fixed_sticky_maxscroll_390x844.png`, `..._768x1024.png`.

**Trạng thái test/build hiện tại:**

| lệnh | kết quả |
|---|---|
| `pytest test_composition_api.py test_csmcir_aggregate_api_parity.py` | **11 passed** (tôi chạy lại, xác nhận) |
| `npm run build --prefix workbench/frontend` | pass |
| browser QA cả 2 viewport | 0 console error, 0 page error, 0 failed request, 0 HTTP ≥400 |

#### Vẫn để mở có chủ đích (cần quyết định thiết kế)

| id | lý do chưa sửa |
|---|---|
| **F1** | Gate environment bị override trên đường composition — cần quyết định có chặn cứng hay không |
| **F5** | Fixture JSON tự mâu thuẫn — cần quyết định sửa số hay bỏ hẳn fixture |
| **F7** | `reference_eligible` assert chứ chưa measure — cần sửa runner để suy ra từ ranking |
| **F9** | Docs dẫn 2 smoke run không tồn tại và bỏ sót run thật — cần viết lại spec |

#### Known issues — addendum 2026-11-08 (F15: defect badge CERTIFIED ở site thứ hai)

| id | trạng thái | nội dung |
|---|---|---|
| **F15** — `App.tsx:71` (Evaluation Runs, nhánh aggregate) | **ĐÃ ĐÓNG** (addendum này) | Badge `<span className="badge badge-pass">CERTIFIED</span>` hardcode, không đọc receipt API. Nay dùng `isCertifiedEvidence(b)` + `b.verification?.state?.toUpperCase()`, và có nhãn **"duplicate attempt — provenance knobs missing"** khi `evaluator_environment_knobs` null. Chứng cứ: chặn `**/api/benchmark-runs*` rewrite `verification.state` → `CERTIFIED` biến thành `UNCORROBORATED` (và `UNCERTIFIED` khi `certified:false`), chuỗi `CERTIFIED` không còn trong body. Ảnh: `native/evaluation_runs_badge_1440x900.png`, `..._390x844.png`, `..._stub_uncorroborated_1440x900.png`. Chi tiết: `REVIEW_ENCODER.md` → ADDENDUM F15 |
| **F15.a** — `App.tsx:70` (Models) | **KHÔNG PHẢI DEFECT** | Trang Models **chưa bao giờ** có badge CERTIFIED; chỉ render `aggregate-only (CAPABILITY_ABSENT)`. Ghi rõ để không ai đi tìm defect không tồn tại |
| **F15.b** — `BenchmarkView.tsx` | **ĐÃ ĐÚNG TỪ TRƯỚC** | `isCertifiedEvidence(run)` (`:32`) + `run.verification?.state` (`:57`). F2 chưa bao giờ áp dụng cho trang Benchmark |

**Đính chính F2:** F2 chỉ nêu Dashboard (`:69`), nên "F2 đã fix" chỉ đúng **theo từng site**. Hai site
hardcode là `:69` và `:71`; sửa `:69` một mình là fix chưa trọn. Mục F2 trong bảng phía trên đã được
sửa lại thành per-site để không đọc thành "F2 fully verified".

> Ghi chú lịch sử: lần kiểm của tôi lúc **02:29:38** chưa thấy thay đổi nào landed (App.tsx còn
> hardcode, CompositionView còn 0 match `recall_at_50`, route queries chưa có 404). Năm mục trên
> landed **sau** thời điểm đó. Mục 11.F này giờ phản ánh trạng thái đã verify.

### 11.G Giới hạn của chính review này

1. Không chạy lại phép tính model (cấm GPU job) → verify rank **từ rank artifact**, không từ model.
2. Không verify dependency set bằng verifier (vì F1); chỉ `pip list` thủ công tại thời điểm review.
3. Không verify byte-exact site-packages (`validated_environment_fingerprint is None`).
4. Không chứng minh 2 report CSMCIR là 2 lần chạy độc lập — chỉ kết luận "trình bày trùng một kết quả".
5. Không đối chiếu paper ENCODER (khác protocol).

### 11.H Commit state

**KHÔNG có gì được commit hay push.** HEAD vẫn `1716e81`. Toàn bộ công việc nằm trong working tree:

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

8 file sửa + 9 mục chưa track.

---

## 12. Cách kiểm lại số từ bytes đã commit

Toàn bộ số composition trong handoff này được đo từ artifact nằm dưới `workbench/artifacts/**`.
Thư mục đó **bị `.gitignore` loại bỏ toàn bộ** (`.gitignore:85: workbench/artifacts/**`), nên nó
**không thể** được commit: ai clone repo ở máy khác sẽ không thấy `summary/` lẫn `ranks/`, và handoff
sẽ trỏ vào đường dẫn rỗng. Vì vậy lớp bằng chứng bền (durable mirror) được đóng gói lại ở
`workbench/docs/reviews/20261108T015603+0700/composition/` — thư mục này **được** commit, kèm
`EVIDENCE_INDEX.json` ở **gốc bundle** ghi `bytes` + `sha256` cho **mọi** file trong bundle (212
entry, `dest` relative với gốc bundle: `composition/…`, `browser_qa/…`, `live/…`, `native/…`, và 4
file MD gốc).

### 12.A Cấu trúc bundle

```
workbench/docs/reviews/20261108T015603+0700/          (bundle root; index ở đây)
  EVIDENCE_INDEX.json                                  index: dest/bytes/sha256 cho MỌI file (212)
  FINAL_REPORT.md, FINAL_REPORT.vi.md, HANDOFF.md, REVIEW_ENCODER.md
  browser_qa/                                          176 file (167 PNG + harness JSON + flows.py/zooms.py)
  live/                                                8 file (6 PNG + browser_qa.json + composition_dom_check.json)
  native/                                              6 PNG
  composition/                                        18 file (11 JSON dữ liệu + 7 PNG QA) — mục 7/8
    summary/<run_dir>/full.json        \  summary composition, ~10–11 KB mỗi file
    summary/<run_dir>/sf.json           |  (3 subset điều kiện: full / sf / validated)
    summary/<run_dir>/validated.json   /
    ranks/6a528e877c328772e2c7bb28/encoder_fashioniq.json                  381,477 B
    ranks/10173dc45dea8c90fd2a284b/limn_base_iter0_all_categories.json    438,051 B
    1440x900.png, 390x844.png, 390x844_table.png, composition_1440x900.png,
    composition_390x844.png, dashboard_1440x900.png, dashboard_390x844.png
```

Rank artifact là **bảng hạng thô theo từng query** (`ranks.<condition>` cho
`image_only_empty_text` / `multimodal` / `text_only_black_image`, 6003 query mỗi điều kiện) — đây là
thứ khiến metric **tái tính được**, không chỉ tin được.

### 12.B Lệnh kiểm: hash trước, số sau

Chạy từ gốc bundle:

```bash
cd workbench/docs/reviews/20261108T015603+0700
sha256sum composition/summary/encoder_fashioniq_6a528e877c328772e2c7bb28/full.json
```

So chuỗi in ra với `sha256` của entry `dest` tương ứng trong `EVIDENCE_INDEX.json` (~36.2 KB).
Kiểm **toàn bộ** bundle trong một lệnh:

```bash
python3 - <<'PY'
import hashlib, json, os
B = "workbench/docs/reviews/20261108T015603+0700"
idx = json.load(open(os.path.join(B, "EVIDENCE_INDEX.json")))
bad = 0
for e in idx["entries"]:
    b = open(os.path.join(B, e["dest"]), "rb").read()
    if len(b) != e["bytes"] or hashlib.sha256(b).hexdigest() != e["sha256"]:
        bad += 1
        print("MISMATCH", e["dest"])
print(f"{len(idx['entries'])} entries, {bad} mismatch")
PY
```

Kỳ vọng: `212 entries, 0 mismatch`.

Xem metric (dùng `json.load`, **không** dùng query selector `?q=` — nó đã trả `null` cho key có giá
trị thật, một false-negative thật gặp hai lần):

```bash
python3 -c "import json;d=json.load(open('composition/summary/encoder_fashioniq_6a528e877c328772e2c7bb28/full.json'));print(d['metrics'])"
```

hoặc xem nhanh đúng ba số đã công bố:

```bash
python3 -c "import json;d=json.load(open('composition/summary/encoder_fashioniq_6a528e877c328772e2c7bb28/full.json'));m=d['metrics'];print(m['composition_gap_ndcg'], m['multimodal']['ndcg_full'], m['multimodal']['recall_at_50'])"
```

Nếu không có Python, `grep -A3 '"metrics"' composition/summary/.../full.json` cũng thấy được khối `metrics` (file
pretty-print 2 space), nhưng số đã trích trong handoff **luôn** được đọc từ `json.load`.

### 12.C Xác nhận đây là phép đo thật, không phải fixture

Mỗi file summary trong bundle đều mang:

| field | giá trị |
|---|---|
| `gallery_count` | `74381` |
| `gallery_identity_sha256` | `1db68c59d96f014cdbecf936e108653de98abef0bd9a98a8abc2fe08b5dec4b4` |
| `is_synthetic` | `false` |

Đó là cách phân biệt một phép đo thật với fixture: hai bundle `SYNTHETIC_*` khai
`is_synthetic: true` và tự mâu thuẫn (F5 ở mục 11.B). Kiểm bằng:

```bash
python3 -c "import json;d=json.load(open('summary/limn_base_iter0_all_categories_10173dc45dea8c90fd2a284b/full.json'));print(d['gallery_count'], d['gallery_identity_sha256'], d['is_synthetic'])"
```

### 12.D Phạm vi bundle

Run id **có** trong bundle (2 run):

| run_id | model / scope | trạng thái |
|---|---|---|
| `6a528e877c328772e2c7bb28` | ENCODER, FashionIQ, 6003 query | cả 3 điều kiện `COMPLETED`, coverage 1.0 |
| `10173dc45dea8c90fd2a284b` | LIMN base iter0, all categories | cả 3 điều kiện `PARTIAL`: 119 `category_scope_failure`, coverage 0.980176578377478 (5884/6003 query được score) |

Lưu ý cho người đọc số LIMN: metric của nó tính trên cohort 5884 query đã score, **không** phải toàn
6003 — đừng so trực tiếp gap 0.5417 với gap 0.4723 của ENCODER mà bỏ qua chênh cohort.

Mục **cố ý loại trừ**, không có trong bundle:

1. Run smoke `3f86f58483b368377e2c151e` — chỉ 8 query (`smoke.query_limit: 8`), không phải phép đo.
2. Hai bundle fixture `SYNTHETIC_encoder_fashioniq_fixture001`,
   `SYNTHETIC_habit_fiq_n02_fixture002` — `is_synthetic: true`, số tự mâu thuẫn (F5).
3. Toàn bộ shard feature `.npy` — quá lớn và bị xoá sau bước ranking theo thiết kế, nên không tái tạo
   được từ bất kỳ checkout nào.
