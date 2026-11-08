# BACKEND_CHANGES — candidate Top-K thật, capability tách rời, thống kê tính từ dữ liệu

Ngày: 2026-11-08 · Nhánh: `research/cir-failure-workbench`.
Thư mục bằng chứng: `workbench/docs/reviews/20261108T115509+0700-backend/`.

---

## 0. Tóm tắt một đoạn

`top_results: []` không còn là hằng số. Ứng viên (candidate) giờ được lấy **từ chính lần
`adapter.score()` đã sinh ra rank đo lường** — một ma trận điểm cho ra cả rank tuyệt đối lẫn cửa sổ
Top-K, nên không thể lệch phương pháp/gallery/preprocessing/tie policy. API tách 4 capability độc lập
(`aggregate_metrics`, `per_query_target_ranks`, `candidate_topk`, `image_availability`), phân biệt
`unscored` (pipeline không chấm được) với `retrieval_failure` (có rank thật nhưng tệ), và mọi con số
đếm trên dashboard đều tính từ artifact — không còn literal `1 unique measurement`.

**Phần còn thiếu để có Top-K cho model thật:** `workbench/composition/runner.py` do agent khác sở
hữu. Tôi đã viết toàn bộ writer/reader/schema/API + đặc tả một-lời-gọi trong
`workbench/docs/reviews/eval-runner-requests.md`. Hiện **5/5 run thật trên đĩa đều
`candidate_topk: false` với status `TOPK_EXPORT_NOT_REQUESTED`** (đúng sự thật: artifact cũ chưa
từng export). Bằng chứng candidate thật dùng artifact audit chính thức của CIRCUS, xem §4.

---

## 1. File đã thay đổi (chỉ trong quyền sở hữu)

| File | Thay đổi |
|---|---|
| `workbench/composition/topk.py` | **mới** — writer/reader/schema/manifest của export candidate Top-K |
| `workbench/composition/adapters.py` | thêm `rank_and_topk()`, `_score_matrix()`, `_ranks_from_scores()`; `rank()` gọi lại helper dùng chung |
| `workbench/backend/composition.py` | capability layer, `statistics`, `measurement`, `transfer`, `measurement_identity`, `duplicate_status`, `unique_measurement_count`, `is_imported_summary`, đọc candidate trong query page, tách `unscored`/`retrieval_failure` |
| `workbench/backend/main.py` | params mới cho `/queries`; endpoint `/api/composition/capabilities`; endpoint `/api/composition/runs/{run_id}/candidates`; `/api/evidence/summary`; field mới trong `/api/benchmark-runs` |
| `workbench/scripts/composition_eval.py` | cờ `--topk-depth` (từ chối chạy nếu runner chưa nhận `topk_depth`) |
| `workbench/scripts/import_released_audit_candidates.py` | **mới** — chuyển artifact audit chính thức thành export candidate + verify bất biến |
| `workbench/tests/test_composition_topk_export.py` | **mới** — 12 test |
| `workbench/tests/test_composition_topk_rank_agreement.py` | **mới** — 4 test |
| `workbench/tests/test_composition_topk_api.py` | **mới** — 6 test |
| `workbench/tests/test_composition_topk_provenance.py` | **mới** — 5 test |
| `workbench/tests/test_capability_composition.py` | **mới** — 21 test |
| `workbench/docs/reviews/eval-runner-requests.md` | **mới** — đặc tả 1 keyword + 3 dòng cho runner |
| `workbench/docs/reviews/frontend-backend-requests.md` | **mới** — hợp đồng field/kiểu/JSON mẫu cho frontend |

**Không sửa:** `workbench/frontend/**`, `workbench/composition/runner.py`,
`workbench/composition/data.py`, `workbench/scripts/serve_workbench.py`,
`workbench/config/workbench.env`.

### Đường lấy candidate (điều quan trọng nhất)

```
adapter.rank_and_topk(vectors, targets, positions, cats, depth)
   ├── scores = self._score_matrix(...)        # MỘT lần gọi score()
   ├── ranks  = self._ranks_from_scores(scores, positions)   # 1 + #greater + #earlier-equal
   └── rows   = [topk_from_scores(row, depth) for row in scores]
```

`rank()` (đường metric cũ) gọi đúng `_ranks_from_scores` trên cùng ma trận, nên candidate và rank
không thể lệch nhau. `topk_from_scores` chọn theo `(-score, vị trí)` — đúng tie policy mà
`_ranks_from_scores` đếm. Đã kiểm 300 lượt ngẫu nhiên (điểm lượng tử hoá để tạo tie): khớp
`np.argsort(-row, kind="stable")[:depth]` 300/300.

---

## 2. Hợp đồng API mới (chính xác)

### 2.1 `GET /api/composition/runs?subset=full[&include_fixtures][&include_smoke][&include_imported]`

Mặc định **loại** fixture, smoke, và artifact import. Field mới mỗi dòng:

```jsonc
"capabilities": {"aggregate_metrics": true, "per_query_target_ranks": true,
                 "candidate_topk": false, "image_availability": true},
"capability_detail": {
  "aggregate_metrics":      {"available": true,  "status": "AVAILABLE", "reason": null},
  "per_query_target_ranks": {"available": true,  "status": "AVAILABLE", "reason": null,
                             "query_count": 6003, "conditions": ["multimodal", "..."]},
  "candidate_topk":         {"available": false, "status": "TOPK_EXPORT_NOT_REQUESTED",
                             "reason": "...", "depth": null, "conditions": [], "source": null, "path": null},
  "image_availability":     {"status": "local", "sampled": 40, "resolved": 40,
                             "layout": "fashioniq_standard", "reason": null}
},
"statistics": {"conditions": {"multimodal": {"denominator": 6003, "scored_count": 5884,
                                             "unscored_count": 119,
                                             "unscored_by_kind": {"category_unresolved": 119},
                                             "coverage": 0.98017}}, "query_count": 6003},
"measurement": {"sample_count": 6003, "denominator": 5884, "coverage": 0.98017,
                "scored_count": 5884, "unscored_query_count": 119,
                "unscored_by_kind": {"category_unresolved": 119},
                "image_availability_sample_size": 40},
"transfer": {"is_transfer": true, "training_dataset": "CIRR",
             "evaluation_dataset": "FashionIQ", "label": "CIRR→FashionIQ transfer"},
"duplicate_status": "unique",              // unique | duplicate | undetermined | synthetic_fixture
"duplicate_attempt_count": 1,
"measurement_group_id": "…24 hex…",
"measurement_identity_confidence": "full", // full | partial
"measurement_identity_missing_inputs": [],
"checkpoint_identity": "12cbbf95…"
```

`candidate_topk.status` ∈ `AVAILABLE`, `AVAILABLE_PARTIAL`, `TOPK_EXPORT_NOT_REQUESTED`,
`TOPK_EXPORT_INCOMPLETE`, `TOPK_EXPORT_UNREADABLE`, `TOPK_EXPORT_CORRUPT`,
`TOPK_EXPORT_MISMATCH`, `TOPK_EXPORT_GALLERY_MISMATCH`, `TOPK_EXPORT_BLOCKED`.

### 2.2 `GET /api/composition/runs/{run_id}/queries`

Tham số mới: `query_state=unscored|retrieval_failure|hit`, `top_k` (mặc định 10).
Field mỗi item: `query_state`, `unscored`, `unscored_kind`, `retrieval_failure`,
`top_results`, `candidate_status`, `candidate_reason`, `candidate_depth`,
`reference_name`, `positive_names`.
Field response: `condition`, `counters{scored,unscored,hit_at_k,retrieval_failure,unscored_by_kind,candidates_available}`,
`statistics`, `candidates{available,status,reason,depth}`.

`TopResultOut` = `{rank, image_id, gallery_position, score}` — `rank` là vị trí 1-based **trong cửa
sổ đã lưu**, `gallery_position` là vị trí trong gallery đầy đủ (hai sự thật khác nhau),
`score` có thể `null` khi artifact nguồn chỉ lưu thứ tự.

### 2.3 Endpoint mới

- `GET /api/composition/runs/{run_id}/candidates?subset&condition&query_id&limit&offset`
- `GET /api/composition/capabilities?subset` → đếm capability + đếm measurement duy nhất
- `GET /api/evidence/summary` → `{aggregate_run_count, certified_aggregate_count,
  unique_measurement_count, undetermined_measurement_count, duplicate_attempt_count, confidence}`

### 2.4 `/api/benchmark-runs` thêm

`duplicate_status`, `duplicate_attempt_count`, `capabilities`, `transfer`,
`measurement` (kèm `denominator_reason` khi report không ghi `query_count`).

---

## 3. Cách một run bật export Top-K

Hai tham số, mặc định tắt, chi phí bằng 0 khi tắt:

```bash
workbench/envs/ctl312h/bin/python workbench/scripts/composition_eval.py \
  --models limn --subsets full --topk-depth 50
```

- Ghi vào `artifacts/composition/runs/<run_id>/candidates/<model>_<ckpt>/<condition>.jsonl`
  + `<condition>.manifest.json`.
- ~1.3 MB JSONL mỗi condition cho 6003 query ở depth 50 (đo thật: depth 20, 3 condition = 5.2 MB).
- Depth bị kẹp theo kích thước gallery và trần `MAX_EXPORT_DEPTH = 500`.
- `--topk-depth` hiện **từ chối chạy** (`COMPOSITION-TOPK-UNAVAILABLE`) vì runner chưa nhận
  `topk_depth`; đây là cố ý — thà báo lỗi còn hơn tạo run nói rằng đã export mà không có gì.
- Chi tiết 3 dòng cần thêm vào runner: `workbench/docs/reviews/eval-runner-requests.md`.

Reader (đã có) thề: vị trí chỉ được hiểu theo digest gallery trong manifest; gallery khác →
`TOPK_EXPORT_GALLERY_MISMATCH`, không best-effort; thiếu manifest → `INCOMPLETE`; digest sai →
`CORRUPT`; file cache theo `(path, mtime, size)`.

---

## 4. Bằng chứng candidate THẬT (không có GPU)

Artifact chính thức `circus_release/shortcut_audit/retrieval_data_fashioniq_task7_aggregated_best_rank_mm_any.json`
lưu, mỗi query/condition, thứ tự vị trí gallery mà retriever chính thức trả về + target rank.
Script mới chuyển nó sang định dạng export và **tự kiểm 2 bất biến**:

- vị trí 1..depth trong export == vị trí đầu của artifact;
- nếu target nằm trong cửa sổ: `stored[rank-1] == target_gallery_index`.

Kết quả: **6003 query × 3 condition, 0 mismatch** (`evidence/import-released-audit.json`).
`query_idx` của artifact là 0-based nên map sang `qid = "7:" + (idx+1)`; kiểm độc lập: 6003/6003
khớp reference image + query text của release.

Tái lập:

```bash
workbench/envs/ctl312h/bin/python workbench/scripts/import_released_audit_candidates.py \
  --output-root /tmp/topk-audit --verify --write-summary
```

(Bản export đầy đủ 5.2 MB không nằm trong `docs/` — chỉ giữ curl output trong
`evidence/curl/04-candidates.json` và `05-queries-with-candidates.json`; export mẫu đã rút gọn số
dòng thì **không** còn khớp digest trong manifest nên tôi đã xoá thay vì để lại artifact tự mâu thuẫn.
`evidence/artifacts/` chỉ còn ranks + summary, không có candidate rows.)

**Đây KHÔNG phải model của ta.** Nó có `provenance_source: "released_audit_retriever"` +
`reproduction_status: "RELEASED_ARTIFACT_IMPORT"`, **mặc định bị loại khỏi `/api/composition/runs`**,
chỉ hiện với `include_imported=true` và xếp tier cuối. Việc nhận diện dựa trên provenance khai báo,
không dựa trên tên model (có test: một run tên `looks_imported_but_is_ours` vẫn là measurement).

---

## 5. Kiểm chứng đã chạy thật

Bộ test đích (`pytest workbench/tests -q -p no:cacheprovider -k "composition or csmcir or capability or topk"`):

```
175 passed, 2 skipped, 2211 deselected, 8 warnings in 187.79s
```

Không có failure, không có error. (`evidence/test-targeted.txt`)

Lượt chạy trước đó có 2 failed + 2 error; cả 4 đều **không** do thay đổi này:

- 2 `failed` là `test_circus_adapters_blip.py::test_adapter_import_and_unready_status[conesep|ptha_mtst]`:
  test khẳng định `Adapter.status().ready is False` trong khi checkpoint của hai model đó **đã được
  stage** trên máy này (`checkpoint_selection/20261106T002617+0700/cohort/downloads/…pt` tồn tại) →
  `ready=True`. Chạy riêng file đó: `18 passed`. Đây là phụ thuộc trạng thái máy, có từ trước.
- 2 `error` là `test_r11_remediations.py::…[csmcir]` và
  `test_runtime_adaptation.py::…[csmcir]`, chết ở `workbench/tests/conftest.py:45` với
  `AttributeError: 'function' object has no attribute 'cache_clear'` (fixture monkeypatch
  `model_settings.file_settings` thành lambda, teardown gọi `.cache_clear()`).

Kiểm chứng thêm: bộ 6 file test **của tôi** chạy riêng = `61 passed` (`evidence/test-new-files-only.txt`);
tập test đích **trừ** 5 file mới của tôi = `126 passed, 2 skipped, 2 errors` — tức là không có
regression nào do tôi gây ra, và 2 error kia vẫn còn khi không có code của tôi.

Trạng thái thật của 5 run (curl live, `evidence/curl/11-final-live-runs.json`):

| run | model/ckpt | status | aggregate | ranks | candidates | images |
|---|---|---|---|---|---|---|
| `10173dc4…` | limn/base_iter0_all_categories | PARTIAL | ✔ | ✔ | ✘ NOT_REQUESTED | ✔ |
| `315dd590…` | ptha_mtst/cirr_transfer | COMPLETED | ✔ | ✔ | ✘ NOT_REQUESTED | ✔ |
| `6a528e87…` | encoder/fashioniq | COMPLETED | ✔ | ✔ | ✘ NOT_REQUESTED | ✔ |
| `ad477312…` | tme/fashioniq_0 | COMPLETED | ✔ | ✔ | ✘ NOT_REQUESTED | ✔ |
| `e0dd04a4…` | habit/fiq_n02 | FAILED | ✘ | ✔ (toàn None) | ✘ NOT_REQUESTED | ✔ |

- LIMN 10173d…: `measurement.unscored_query_count=119`,
  `unscored_by_kind={"category_unresolved":119}`, `coverage 0.9802`, `denominator 5884`.
- Lọc `query_state=unscored` trả **119 item**, `retrieval_failure=0` — tách hẳn khỏi lỗi truy hồi.
- Lọc `query_state=retrieval_failure` trên cùng run trả **4115 item**, `unscored=0`
  (`evidence/curl/13-final-live-failure-filter.json`).
- PTHA: `transfer.is_transfer=true`, `label="CIRR→FashionIQ transfer"`.
- `/api/evidence/summary` (live): `12 aggregate runs, 2 certified, 12 unique measurements,
  0 undetermined` — mọi con số đều đếm từ artifact, không còn literal `1 unique measurement`.
- Run import: `target_rank=31987` với cửa sổ depth 20 → rank tuyệt đối sống ngoài Top-K, candidate
  vẫn đúng ảnh thật (`B00DP8U3GU.jpg` tại gallery position 69023).

---

## 6. Còn chặn — lý do cụ thể

1. **Top-K cho model thật.** `run_composition` gọi `adapter.rank(...)` tại
   `workbench/composition/runner.py:899` và vứt ma trận điểm. Không thể lấy lại candidate mà không
   chạy lại một phương pháp khác (bị cấm). Đã có `rank_and_topk` + `write_batch` + `open_export` /
   `close_export`; cần một keyword `topk_depth` và 3 dòng tại chỗ rank — đặc tả trong
   `eval-runner-requests.md`. **Không cần GPU để code xong**, chỉ cần GPU cho lần chạy thật đầu tiên.
2. **CSMCIR và các model aggregate-only.** Báo cáo của chúng tự khai
   `per_query_state: CAPABILITY_ABSENT`; evaluator chỉ in r10/r50 ra stdout, không có per-query
   artifact. `candidate_topk` của chúng là `false` với status nêu tên, không phải "hỏng".
   `TOPK_EXPORT_BLOCKED` đã có trong danh sách status nhưng chưa được gán cho run nào vì evaluator
   chưa từng thử export — gán nó bây giờ là bịa.
3. **`image_availability` là mẫu 40 identity** (đều nhau trên gallery, không phải N đầu). Quét cả
   74k file mỗi request là không chấp nhận được; field `sampled` nói rõ cỡ mẫu. Đây là
   `ponytail:` có chủ ý, nâng cấp = cache một lần quét toàn gallery theo `(root, mtime)`.
4. **2 error trong bộ test đích là có sẵn, không do thay đổi này**:
   `test_r11_remediations.py::…[csmcir]` và `test_runtime_adaptation.py::…[csmcir]` chết ở
   `workbench/tests/conftest.py:45` với `AttributeError: 'function' object has no attribute
   'cache_clear'` vì fixture monkeypatch `model_settings.file_settings` thành lambda rồi teardown
   gọi `.cache_clear()`. Cả hai file không nằm trong quyền sở hữu của tôi.

---

## 7. Yêu cầu GPU (đã ghi vào `backend-gpu-requests.md`)

Một lần chạy tuần tự, sau khi runner nhận `topk_depth`:

```
workbench/envs/ctl312h/bin/python workbench/scripts/composition_eval.py \
  --models limn --subsets full --topk-depth 50 --resume
```

Cần ước lượng: một run LIMN full (6003 query, gallery 74381) — thời gian do agent GPU quyết định.
Việc của tôi đã xong trước bước này; nếu runner chưa adopt, lệnh trên sẽ thoát ngay với
`COMPOSITION-TOPK-UNAVAILABLE` chứ không tốn GPU.

---

# BỔ SUNG — vòng 2: runner, resume trap, hai defect frontend báo

## 1. Đã nhận quyền sở hữu `runner.py` / `data.py` / `composition_eval.py`

7 bước trong `eval-runner-requests.md` đã áp dụng **đầy đủ**:

| # | Thay đổi | Trạng thái |
|---|---|---|
| 1 | `run_composition(..., topk_depth: int \| None = None)` | ✅ |
| 2 | `topk.open_export(...)` một lần mỗi model | ✅ |
| 3 | `rank_and_topk` tại chỗ rank, gate `writers`, dùng `valid_indices` sau `keep` | ✅ |
| 4 | `topk.close_export(...)` + `finally: abort_export(writers)` | ✅ |
| 5 | khoá `topk_export` trong summary | ✅ |
| 6 | `--topk-depth` trong CLI + pass-through | ✅ |
| 7 | `variant_provenance` trong summary khi có adapted runtime | ✅ |

## 2. Quyết định về feature cache: **option (a)** — giữ cache khi có export

**Bằng chứng của director là đúng**: `finally:` gọi `shutil.rmtree(model_work)` **vô điều kiện**,
nên `work/<model>/features/*.npy` bị xoá sau MỌI lần chạy → `_cached_features` luôn rỗng →
`_backfill_export` là dead code → mọi `--topk-depth --resume` sẽ abort.

Đã chọn (a): giữ `features/` khi có `topk_depth` hoặc `--keep-features`; mọi thứ khác (worker log,
ảnh query đã stage, job dir) vẫn xoá mỗi lần. Lý do: encode lại 74.381 ảnh × 4 model (~40 phút mỗi
model) là sai lầm đắt; ~2.4 GB/model trên volume còn 127 GB thì không.

Trần được ghi bằng `ponytail:` ngay trong khối `finally`, kèm đường nâng cấp (prune theo tuổi, hoặc
float16).

## 3. Resume trap — cờ chính xác cho GPU agent

```bash
workbench/envs/ctl312h/bin/python workbench/scripts/composition_eval.py \
  --models limn --subsets full --topk-depth 50 --keep-features
```

**KHÔNG truyền `--resume`** cho lượt export: có `--resume` + rank file sẵn có thì chỗ rank bị bỏ
qua, cửa sổ không bao giờ được ghi từ lần chấm điểm, và backfill cần cache mà các lượt trước đã
(cố ý) xoá. `--topk-depth` tự bật `--keep-features`.

Một run xin export mà không tạo được: `execution_status: EXPORT_INCOMPLETE`,
`topk_export.status: TOPK_EXPORT_BLOCKED` + lý do. **Không bao giờ** báo thành công.

**Identity**: `topk_depth` **cố ý không** nằm trong `identity` — cùng một đo lường, thêm vào sẽ sinh
`run_id` mới và hiện thành duplicate giả. Export key theo **attempt**:
`_export_key(fingerprint, attempt, depth)`.

## 4. `reference_eligible` — sửa claim sai provenance

Ba chỗ ghi literal `True` (identity, summary, export manifest) nay dùng giá trị **ĐO** từ
`adapter.measured_reference_eligibility()`; `limn` đo ra `False` kèm lý do, adapter không probe
được ghi `None` (chưa đo) — không bao giờ ghi một khẳng định.

## 5. `variant_provenance` / sm_120

Root cause: trong `evaluate_models._variant_provenance` có `adapted_profile = None` rồi
`if adapted_profile:` — kiểm tra chính biến vừa gán `None`, nên nhánh **không bao giờ chạy**; đó là
lý do TME/PTHA dưới `WORKBENCH_RUNTIME_ADAPTATION=sm_120` không có `adapted_runtime`. Đã sửa thành
`if adaptation:` và runner gọi lại đúng builder đó (một quy ước, không phải hai).

Summary nay có thêm `runtime_adaptation_selected` và `adapted_runtime_applied`, nên một model không
khai báo adaptation vẫn ghi nhận là selection đã bật nhưng không áp dụng.

## 6. DEFECT A (frontend) — số liệu sai theo subset: ĐÃ SỬA

Trước: `measurement.scored_count`/`unscored_query_count` tính toàn run trong khi `sample_count`
theo subset → `subset=sf` hiện `5.884 / 4.069 (144,6%)`.

Sau (curl live, run LIMN `10173d…`):

| subset | sample | denom | scored | unscored | scored+unscored==sample | coverage |
|---|---|---|---|---|---|---|
| full | 6003 | 5884 | 5884 | 119 | ✔ | 0.9802 |
| sf | 4069 | 4000 | 4000 | 69 | ✔ | 0.9830 |
| validated | 586 | 581 | 581 | 5 | ✔ | 0.9915 |

`statistics.*` cũng scoped theo subset; con số toàn run chuyển sang `statistics_whole_run` và
`measurement.whole_run` (tên nói rõ), `measurement.subset` cho biết phạm vi.
Bất biến `scored + unscored == sample_count` và `coverage <= 1.0` có test cho cả 3 subset
(`test_capability_subset_scoping.py`).

## 7. DEFECT B (frontend) — dòng audit đã import chưa hiện: ĐÃ SỬA

Artifact audit chính thức đã được chuyển vào artifacts root đang phục vụ:

```bash
workbench/envs/ctl312h/bin/python workbench/scripts/import_released_audit_candidates.py \
  --output-root workbench/artifacts --verify --write-summary
```

Kiểm lại: 6003 × 3 condition, **0 mismatch**. Curl live với `include_imported=true`:

```
run_id=released_audit_task7_mm_any  provenance=released_audit_retriever
status=RELEASED_ARTIFACT_IMPORT     candidate_topk=true  depth=20
qid=7:4436 target_rank=31987
  rank=1 image_id=mbeir_images/fashioniq_images/B00DP8U3GU.jpg gallery_position=69023 score=None
  rank=2 image_id=mbeir_images/fashioniq_images/B0047LWLOG.jpg gallery_position=12875 score=None
```

`score: null` (artifact chỉ lưu thứ tự, không lưu điểm) — phải render là *không có*, không phải 0.0.
`target_rank=31987` nằm ngoài cửa sổ depth 20 và vẫn sống độc lập. Ảnh trả 200 qua
`/api/images/circus/B00DP8U3GU.jpg`. Danh sách mặc định **vẫn 5 dòng**, không có dòng import.

---

# BỔ SUNG — vòng 3: annotation cho query CIRCUS

## Vấn đề

`/api/annotations` chỉ nhận `category:index:reference:target`, nên MỌI query của release CIRCUS
(`7:603`) bị từ chối, và luồng (g) của QA harness `blocked` ở cả ba viewport — tức là không thể chú
thích dữ liệu chính mà tool đang phục vụ.

Có **hai** lớp chặn, không phải một:

1. **Backend** (của tôi) — đã sửa.
2. **Frontend** `NotebookPage.tsx:78-81` — guard phía client, chặn trước khi gửi request. Thuộc agent
   frontend, đã ghi đặc tả sửa chính xác vào `frontend-backend-requests.md`.

## Backend: nhận dạng nào được chấp nhận

Hai dạng identity, mỗi dạng **gắn với protocol có dữ liệu thật dùng nó** — đọc từ đĩa, không phải
so tên:

| dạng | ví dụ | điều kiện |
|---|---|---|
| canonical FashionIQ | `dress:12:B001X.jpg:B002Y.jpg` | luôn hợp lệ |
| CIRCUS release | `7:603` | chỉ dưới protocol mà release đang load khai báo (`circus_fashioniq_release_v1`) **và** qid phải tồn tại trong release (6003 qid) |

Từ chối, kèm lý do có tên:

- `invalid_query_id` — `7`, `7:`, `:603`, `7:60a`, `7:603:extra`, `7:603 `, `dress:1`, `""`
- `query_id_protocol_mismatch` — `7:603` dưới `fashioniq_original_split` (đây chính là chỗ một va
  chạm giữa hai protocol sẽ sống)
- `unknown_query_id` — `7:999999999` dưới protocol release
- `query_dataset_unavailable` — release chưa load nên không thể kiểm (không đoán)
- `unknown_protocol`, `invalid_annotation_labels` — không đổi

Khoá annotation vẫn là cặp `(query_id, protocol_id)`, nên va chạm là **bất khả thi về cấu trúc**:
`7:603`/release và `7:603`/protocol khác là hai khoá khác nhau, và khoá thứ hai bị từ chối ngay.

**Sửa thêm**: `unknown_protocol` trước đây chỉ chấp nhận protocol trong registry, mà registry không
khai báo protocol của CIRCUS release. Nay tập protocol được phép = registry **∪** protocol do release
đang load tự khai. Một chuỗi tuỳ tiện vẫn bị từ chối — protocol chỉ được thừa nhận khi có thứ gì đó
trên đĩa nhận nó.

## Round-trip (curl live, sau restart)

```
PUT {"query_id":"7:603","protocol_id":"circus_fashioniq_release_v1","labels":["wrong_attribute"],
     "note":"qa annotation live proof"}                                    -> 200
     {"query_id":"7:603", ... "note":"qa annotation live proof", "updated_at":"2026-11-08T07:04:50Z"}
GET /api/annotations?query_id=7%3A603&protocol_id=circus_fashioniq_release_v1
                                                                          -> 200, đúng dòng đó
PUT {"query_id":"7:603","protocol_id":"fashioniq_original_split", ...}     -> 400
     {"error":"query_id_protocol_mismatch"}
```

`query_id` trả về **đúng byte** UI đã gửi, nên reload bind đúng query.

## Test

`workbench/tests/test_capability_annotation_identity.py` — **20 passed**. Bao gồm: lưu rồi đọc lại
đúng query; cùng hình dạng số dưới protocol khác không va chạm (chỉ 1 dòng tồn tại trên đĩa);
11 id sai định dạng bị từ chối; canonical vẫn nhận; label/protocol lạ vẫn bị từ chối.

## Harness — trạng thái thật

Trước: `g annotations 1440x900=blocked 390x844=blocked`.
Sau khi sửa backend: **vẫn blocked**, và đây là bằng chứng nó không còn là lỗi backend:

```
PASS  g.annotation_persisted_or_refused | save persisted (False) or explained its refusal:
      'Lưu thất bại: API annotation chỉ nhận query id dạng category:index:reference:target...'
PASS  browser.failed_requests | 0 failed request(s)
PASS  browser.http_ge_400 | 0 HTTP >= 400 response(s)
```

`http_ge_400: 0` nghĩa là **không có request nào được gửi** — UI tự chặn. Sửa guard trong
`NotebookPage.tsx` (xoá hẳn, để server quyết) rồi rebuild bundle là flow (g) chuyển sang `pass`.
Không thể sửa file đó từ đây: thuộc quyền agent frontend.

---

# BỔ SUNG — vòng 4: measurement identity phải gồm đường chấm điểm

## Defect: `measurement_identity` bỏ sót scoring path — ĐÃ SỬA

`measurement_identity` chỉ key trên `preprocessing`, `environment_fingerprint`,
`checkpoint_identity`, `gallery_identity_sha256`. Hai run cùng model mà **chỉ khác cách chấm điểm**
hash ra cùng `group_id` và bị gắn nhãn `duplicate` — tức "đo cùng một thứ hai lần" — sai sự thật.

Cohort thật sự chứa **hai cách chấm điểm**:

| impl | model | bằng chứng |
|---|---|---|
| `einsum-optimized` | conesep, csmcir, ptha_mtst | `np.einsum(..., optimize=True)` |
| `einsum` | habit, hint, intent, tme | `np.einsum(...)` không optimize |
| `matmul` | limn, encoder | `query @ gallery.T` |

Nay `identity` thêm `scoring_module_source_hash` và `adapter_module_source_hash`. Hash này nằm trong
**`identity` của rank artifact**, không phải summary, nên `decorate_summary` đọc từ đó và
`annotate_duplicate_attempts` truyền vào (tham số `identity_source`).

`scoring_module_source_hash` được thêm vào danh sách `essential`, nên artifact cũ thiếu nó →
`measurement_identity_confidence: "partial"` + `measurement_identity_missing_inputs:
["scoring_module_source_hash"]`. **Không** còn `duplicate`, cũng **không** đoán là `unique`.

## Before / after (curl live, `?subset=full`)

**Trước** — mọi dòng `confidence: full`, kể cả HABIT `497c25c` không có scoring hash:

| model | run | duplicate_status | conf | group_id |
|---|---|---|---|---|
| limn | 10173dc45dea | unique | full | 38623b33c3 |
| habit | **497c25c57fbe** | **unique** | **full** | b53c8721a8 |
| conesep | 30f6de423861 | unique | full | 81c3a87c83 |

**Sau**:

| model | run | duplicate_status | conf | group_id | scoring_impl | missing |
|---|---|---|---|---|---|---|
| limn | 10173dc45dea | undetermined | partial | b6e64eda63 | matmul | `['scoring_module_source_hash']` |
| conesep | 30f6de423861 | unique | full | 44b314e6de | einsum-optimized | `[]` |
| ptha_mtst | 315dd590eeb0 | undetermined | partial | b24c07ac46 | einsum-optimized | `['scoring_module_source_hash']` |
| intent | 37ee72aa298b | unique | full | 9485a6323d | einsum | `[]` |
| **habit** | **497c25c57fbe** | **undetermined** | **partial** | d48b1ba050 | einsum | `['scoring_module_source_hash']` |
| hint | 4bc9893a977e | unique | full | e3b12be2ab | einsum | `[]` |
| encoder | 6a528e877c32 | undetermined | partial | 5cab6ce7a1 | matmul | `['scoring_module_source_hash']` |
| tme | ad477312aa57 | undetermined | partial | fc99f23b26 | einsum | `['scoring_module_source_hash']` |
| habit | e0dd04a46ef7 | undetermined | partial | 9684c07e6b | einsum | `['scoring_module_source_hash']` |

9 dòng → **9 group_id khác nhau**. HABIT `497c25c` **không** cùng group với bất kỳ dòng nào
(kiểm từng cặp: `same_group=False` cho cả 8 dòng còn lại).

## A. `released_audit_task7_mm_any` không còn trông như model

Trước: chỉ có `provenance_source`/`reproduction_status`, không có field nào một consumer buộc phải
thấy. Nay mỗi dòng có:

```json
"is_model_run": false,
"provenance_label": "released CIRCUS audit baseline (imported; not a model run of this workbench)"
```

Cùng `scoring_impl.impl: "unknown"` với lý do `"released_audit declares no composition adapter
module"` — nó thật sự không chấm điểm bằng adapter nào của workbench.

Xác nhận: đường mặc định **9 dòng, không có dòng import**; với `include_imported=true` thì 10 dòng
và dòng đó mang `is_model_run: false`.

## B. `scoring_impl` trên từng dòng

Suy ra bằng cách **đọc thân hàm `score`** của adapter module đã đăng ký (`capability(model_id)` →
`inspect.getsource(Adapter.score)`), rồi phân loại:

- có `optimize=True` → `einsum-optimized`
- có `einsum` → `einsum`
- có `@` / `linalg` / `dot` → `matmul`
- không nhận ra / không resolve được model → `unknown` kèm `basis` nói lý do

Mỗi dòng mang `{impl, basis, adapter_hash_known, scoring_hash_known}`. **Không** suy từ chuỗi hash.
Kết quả khớp đúng split thật trong mã nguồn (bảng trên).

## Test

`workbench/tests/test_capability_measurement_identity.py` — **16 passed**:

- hai identity chỉ khác `scoring_module_source_hash` → **group_id khác nhau**, cả hai `full`;
- hash đọc từ block `identity` của rank artifact vẫn được dùng;
- artifact cũ thiếu hash → `undetermined` + `partial` + tên field, không bao giờ `duplicate`;
- hai artifact cũ (đều thiếu) → `undetermined`, không `duplicate`;
- một duplicate thật vẫn được phát hiện (`duplicate`, count 2, cùng group_id);
- 9 model → đúng `impl`; model không resolve được → `unknown`;
- cohort thật sự có ≥2 implementation.
