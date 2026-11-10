# Web surface — W1 / R13 / R7 / §6

Ngày 2026-11-10 · nhánh `research/cir-failure-workbench` · HEAD `2e2e2d2` (không commit)
Dịch vụ: frontend `http://127.0.0.1:5173/`, API `http://127.0.0.1:8000`, log `/tmp/workbench_serve.log`
Bằng chứng thô: `workbench/docs/reviews/20261110-e2e-repair/web_evidence/`

## 0. Trạng thái từng finding

| ID | Verdict | Nội dung |
|---|---|---|
| **W1** | **CONFIRMED**, sửa một phần, còn hai mục con **BLOCKED** | Trang mở bằng URL trần in `FashionIQ Original / SF / 0 lượt`, không có control subset, và một mô hình có kết quả ở scope khác bị ghi "Chưa có số đo". Đã sửa: scope chọn từ dữ liệu thật, nhãn subset chỉ in khi protocol khai báo subset, liên kết scope khác cho từng mô hình (hàm thuần đã có test). Chưa làm: bảng chọn scope khác trong `ModelInventory` (xem §4), trạng thái lỗi/loading/empty của từng nguồn đã có sẵn và đã kiểm. |
| **R13** | **CONFIRMED** — nội dung finding sai một nửa, đã sửa bằng "unverified catalog có provenance" | Native run **có** raw artifact nhưng **không phải** run bị thiếu trên web như mô tả: **34 file** measurement native nằm ngoài mọi đường ingest. Đã thêm `/api/native-measurements` + module `native_measurements.py`. **Không** tạo receipt, **không** đưa số nào vào leaderboard verified. |
| **R7** | **NOT_REPRODUCED** cho "subset trộn thế hệ", **CONFIRMED** cho "subset không khai báo nguồn" | Đo trực tiếp 17 run × 3 subset: không subset nào chứa query ngoài `full`, không subset nào có unscored nhiều hơn `full`, và R@10 phục dựng từ per-query artifact khớp số đã phục vụ tới 1e-9. Cái thiếu là **khai báo**: `source_run_id`, membership, mẫu số. Đã thêm `subset_provenance` vào mọi row. |
| **§6** | **CONFIRMED**, sửa phần lớn | Đã tách nhãn 5 phần (`resultLabels`, có test); mọi nhãn subset đã scoped theo protocol; trạng thái loading/error/empty/zero tách rời ở scope strip; protocol 0 row composition giờ nói nó **có** bao nhiêu report + link mở. Chưa xong: chip trong explorer, gallery ứng viên (§4). |

## 1. W1 — web báo 0

### 1.1 Tái hiện (trước khi sửa)

```
workbench/envs/ctl312h/bin/python /tmp/wsa/probe_bare.py "http://127.0.0.1:5173/"
```

Kết quả trước (lưu ở `web_evidence/before-bare-url-probe.json`):

```
requested    : http://127.0.0.1:5173/
final_search : ?area=overview&protocol=fashioniq_original_split&subset=sf
subset_select: 0        (không tồn tại control subset trên protocol này)
scope_strip  : FashionIQ — Original Split (Full Gallery) / Shortcut-Free (SF)
               0 lượt có dữ liệu ở phạm vi này, chưa gồm lượt lỗi
inventory    : 17 mô hình đăng ký · 0 có số đo · 0 lượt CIRCUS · 15 cần chạy · 2 tạm gác
```

Nguyên nhân, đọc từ code:

* `App.tsx` (effect "First load") chọn `data.protocolIds[0]` — **thứ tự chèn**, không phải nhiều dữ
  liệu nhất — rồi **giữ nguyên subset mặc định `sf`** của protocol trước. Với
  `fashioniq_original_split` (13 report tổng hợp, 0 row composition) thì `sf` không tồn tại →
  0/0/0.
* Nhãn subset in vô điều kiện ở sidebar và ở `scope-strip`, nên câu "Subset Shortcut-Free (SF)"
  đứng cạnh một protocol chỉ có một cohort.
* `ModelInventory` chỉ tra `data.catalog` đã lọc theo đúng `protocol + subset`; mô hình có row ở
  scope khác rơi vào nhánh "Chưa có số đo / Cần chạy" mà không nói rằng nó **có** kết quả ở nơi khác.

### 1.2 Đối chiếu ID giữa các endpoint (trước và sau khi sửa, dữ liệu không đổi)

| Endpoint | Status | Nội dung | Ghi chú |
|---|---|---|---|
| `GET /api/models` | 200 | 17 entry | 17/17 khớp registry; 2 mô hình tạm gác (`csmcir`, `sprc`) |
| `GET /api/runs` | 200 | `[]` | DuckDB index rỗng: 0 run / 0 query / 0 top_results (đã kiểm bằng duckdb read-only) |
| `GET /api/benchmark-runs` | 200 | 13 report, **tất cả** `protocol_id=fashioniq_original_split`, 2 `certified`, 11 `refused`/`uncertified` | không có report nào thuộc `circus_fashioniq_release_v1` |
| `GET /api/evidence/summary` | 200 | `aggregate_run_count=13`, `certified_aggregate_count=2`, `unique_measurement_count=13` | khớp 13 report ở trên |
| `GET /api/composition/runs` | 200 | 35 row = 17 run × subset; **tất cả** `circus_fashioniq_release_v1` | full 17, sf 9, validated 9 |
| `GET /api/composition/capabilities?subset=sf` | 200 | `run_count=9`, `per_query_target_ranks=9`, `candidate_topk=4` | khớp 9 row sf |
| catalog (`buildCatalog`) | — | 22 entry (17 composition full + 5 aggregate certified/research) | con số sidebar "22 lượt khả dụng" là **catalog của subset đang chọn**, không phải deployment |
| bảng render | — | 9 row sf | khớp `/api/composition/runs?subset=sf` |

**Kết luận đối chiếu:** không có ID nào lệch. Hai nguồn đo **hai tập rời nhau** — composition
(CIRCUS release) và aggregate (FashionIQ Original) — và defect là **landing scope**, không phải số.

### 1.3 Sửa

| File | Thay đổi |
|---|---|
| `workbench/frontend/scope.ts` (mới) | `chooseInitialScope` (protocol/subset nhiều row research nhất, bỏ row failed/synthetic/imported), `chooseSubsetFor` (một protocol), `protocolHasSubsets`, `scopeCounts` (6 đếm tách rời), `otherScopeFor`, `resultLabels` (5 phần), `QUERY_STATE_CHIPS`. Hàm thuần, không import. |
| `workbench/frontend/routes.ts` | `parseLocation` trả thêm `scope: {protocolNamed, subsetNamed}` — shell cần biết URL **có nêu** hay không, không chỉ giá trị đã resolve. |
| `workbench/frontend/App.tsx` | Thay effect "First load" bằng effect điền **nửa scope chưa được nêu** từ `chooseInitialScope`/`chooseSubsetFor`; `replaceState`; không ghi đè nửa đã nêu. Control subset và nhãn sidebar chỉ render khi `protocolHasSubsets()`. Thêm `data-testid="sidebar-scope"`, `sidebar-catalog-count`. |
| `workbench/tests/test_web_scope_and_labels.py` (mới) | 12 test chạy `scope.ts` qua node. |

### 1.4 Sau khi sửa (đo trên service thật)

Xem §5 choện log; tóm tắt:

```
http://127.0.0.1:5173/
final_search : ?area=overview&protocol=circus_fashioniq_release_v1&subset=full
subset_select: 1  value=full
```

Landing là CIRCUS **Full** — đúng yêu cầu "CIRCUS pages default to CIRCUS Full when the URL names no
scope", và chọn từ dữ liệu thật (17 row full > 9 row sf).

## 2. R13 — native artifact không lên web

### 2.1 Truy vết

| Bước | Kết quả |
|---|---|
| job → log | `workbench/artifacts/logs/*` có 21 thư mục. 11 là `synthetic_model` (fixture), 2 là `csmcir` fail ở `AF_UNIX path too long`, 2 `combiner` (015 vs 020), 2 `csmcir` completed, 1 `dcnet`. |
| report/log | `logs/*/aggregate_report.json` có 19 file; 19/19 cũng có mặt trong `artifacts/reports/*_aggregate.json`. **Không** có report nào chỉ tồn tại ở log. |
| identity/digest/proof/ledger | `artifacts/evidence_ledger.jsonl` có **19 entry**. 7 report bị `_aggregate_reports()` bỏ vì ledger **không khớp** (log: `skipping aggregate report …: disagrees with its evidence-ledger entry`). Đây là guard đúng: ví dụ `csmcir 20261106T192313…` có `return_code=1`, `invocation_status=INVOCATION_FAILED`. |
| API | `/api/benchmark-runs` phục vụ 13 report (12 synthetic/refused + 2 certified csmcir… thực tế là 13 row: 11 synthetic + 1 combiner + 2 csmcir, đếm theo response). |
| UI | Benchmark view render 13 row với badge trạng thái + `TEST WORKSPACE` cho 11 row fixture. |

**Artifact "có raw nhưng thiếu trên web" thật sự** là nhóm khác hẳn: **các file measurement native**
dưới `checkpoint_selection/*/measurements/` và `*/cohort/measurements/`. Chúng là số đo macro thật
(`model_id`, `protocol`, `aggregate.macro_r10/r50/mean`) nhưng **không đi qua** queue/verifier nên
không có completion proof, không có `invocation_outcome.json`, không có entry trong ledger, không có
verifier receipt. Trước lượt này chúng **không có endpoint nào** — web đọc 0 số cho
`fashioniq_val_split` và `fashioniq_full_gallery_ref_excluded` dù 15 file có số.

Ví dụ (`/tmp/wsa/nat2.txt`, đã lưu `web_evidence/r13-native-catalog.txt`):

```
airknow_fiq_n05_run1.json  airknow  fashioniq_full_gallery_ref_excluded  ['r10','r50','mean']
conesep_fiq_n02.json       conesep  fashioniq_full_gallery_ref_excluded  ['r10','r50','mean']
clvc_run1.json             clvc_net fashioniq_val_split                  ['r10','r50','mean']
encoder_fashioniq.json     encoder  fashioniq_val_split                  ['r10','r50','mean']
hint_fashioniq.json        hint     fashioniq_val_split                  ['r10','r50','mean']
limn_base_iter0.json       limn     fashioniq_val_split                  ['r10','r50','mean']
sprc_vitl_run1.json        sprc     fashioniq_original_split             ['r10','r50','mean']
tme_normal_path_full.json  tme      fashioniq_original_split             ['r10','r50','mean']
… 15 file có số, 4 file blocker, 1 file không đọc được (ptha_transfer_run2.json rỗng)
```

### 2.2 Sửa — "unverified catalog có provenance", đúng như yêu cầu

| File | Thay đổi |
|---|---|
| `workbench/backend/native_measurements.py` (mới) | Đọc 2 glob dưới `checkpoint_selection/`; mỗi row mang `source.path` + `source.sha256` + `bytes`, `verified: false`, `verification.state = "UNVERIFIED_NATIVE_MEASUREMENT"`, lý do nguyên văn. File không phải measurement (`resized_images.json`, `cohort_summary.json` array) **không** bị tính là measurement; file JSON hỏng (rỗng) được báo `unreadable` chứ không thành số 0. |
| `workbench/backend/main.py` | `GET /api/native-measurements`. **Không** đụng `/api/benchmark-runs`, **không** đụng `/api/evidence/summary`. |
| `workbench/frontend/data.ts` | `nativeMeasurements` region (`RegionHandle<NativeMeasurementCatalog>`), key `"native"`, có `retry`. |

**Không** làm, có lý do: không tạo `completion.json`/`invocation_outcome.json` cho các file này
(cấm tạo receipt giả), không chèn row vào `reports/`, không cộng vào `evidence_summary` (đếm verified
vẫn là 0 và `counts.verified` trả 0 tường minh).

### 2.3 Đo trên service thật

```
GET /api/native-measurements -> 200  17 700 B
{ "files": 21, "with_metrics": 15, "unreadable": 1, "signposted_blockers": 4, "verified": 0 }
```

Còn **BLOCKED**: UI chưa có màn hình render catalog này (§4).

## 3. R7 — subset trộn thế hệ

### 3.1 Phát biểu finding vs đo được

Đo 17 run × 3 subset qua API (`web_evidence/r7-api-counters.json`):

```
run             full(total/scored/unscored)   sf                    validated
10173dc45dea    6003/5884/119                 4069/4000/69          586/581/5
30f6de423861    6003/6003/0                   4069/4069/0           586/586/0
315dd590eeb0    6003/6003/0                   4069/4069/0           586/586/0
e0dd04a46ef7    6003/0/6003                   4069/0/4069           586/0/586
… (17 run)
```

Không run nào có `sf_unscored > full_unscored` hay `sf_total > full_total`.

Đo trực tiếp trên artifact per-query (`multimodal.jsonl`) cho 3 run:

```
limn      full 6003 (119 unscored) · sf 4069 (69) · validated 586 (5)
          sf ⊂ full = True · validated ⊂ full = True · validated ⊂ sf = True
conesep   full 6003 (0) · sf 4069 (0) · validated 586 (0) · tập con đúng
ptha_mtst full 6003 (0) · sf 4069 (0) · validated 586 (0) · tập con đúng
```

Và phục dựng R@10 từ chính per-query file (mẫu số = **scored của subset đó**) khớp số phục vụ
tới 1e-9 ở **mọi** row đo được:

```
limn/sf     18.6000 vs served 18.6          limn/full     30.0646 vs 30.064582
conesep/sf  20.1278 vs 20.127796            conesep/val   34.4710 vs 34.470990
habit/sf    20.5210 vs 20.521013            hint/val      29.3515 vs 29.351536
encoder/sf  18.6778 vs 18.677808            tme/sf        19.0219 vs 19.021872
```

→ **NOT_REPRODUCED** cho "một subset có nhiều query chưa chấm hơn parent", và **NOT_REPRODUCED** cho
"metric không thể phục dựng". Ba thế hệ `run_id` khác nhau (`10173dc45dea` vs `9e4918a6b5c2`,
`30f6de423861` vs `a4f786d8af4d`) là **run khác nhau**, và mỗi row tự nhất quán với artifact của nó;
chúng không bị trộn trong một subset.

**Cái thật sự thiếu** (và là CONFIRMED): không row nào khai báo `source_run_id` / membership hash /
mẫu số, nên người đọc **không thể** tự kiểm ba điều trên — và một subset lỗi trong tương lai sẽ
không bị phát hiện.

### 3.2 Sửa

| File | Thay đổi |
|---|---|
| `workbench/backend/subset_provenance.py` (mới) | `subset_provenance()` đọc `composition/results/<protocol>/<model>/<checkpoint>/<run>/<subset>/multimodal.jsonl`; trả `own` (path, `membership_sha256`, query_count, scored, unscored), `full` (parent), `containment` (`subset_of_full`, `extra_query_count`, `unscored_exceeds_full`, `query_count_exceeds_full`), `metric_check` (served vs recomputed R@10, `agrees` tri-state), `consistent`, `reasons[]`. Cache theo `(path, mtime_ns, size)`. |
| `workbench/backend/main.py` | `/api/composition/runs` trả `decorate_run_rows(...)`; **số phục vụ không đổi**, chỉ thêm khoá. |

Không thay metric nào bằng số phục dựng: `metric_check` là **đối chứng**, không phải nguồn.
`ndcg_full` cố ý không phục dựng (nó là nDCG toàn độ sâu, không phải @10; phục dựng @10 sẽ "lệch"
theo thiết kế).

### 3.3 Đo trên service thật

```
GET /api/composition/runs?subset=sf -> 200 (963 340 B)
limn      cons=True sub_of_full=True extra=0 unsc>full=False
conesep   cons=True sub_of_full=True extra=0 unsc>full=False
ptha_mtst cons=True …   intent cons=True …   habit cons=True …   hint cons=True …   tme cons=True …
GET /api/composition/runs?subset=full -> 200; conesep subset_provenance.consistent=True
```

Thời gian: 0.29 s cho 35 row (35 file, ~66 MiB) lần đầu; 0.20 s khi warm.

## 3bis. Ba defect phụ phát hiện trong lúc đo (CONFIRMED, đã sửa)

### a. Scope strip in "0 lượt" trong khi request còn bay

Probe `probe_land.py` (không dùng `networkidle`, vì `/api/models` cold start) bắt được:

```
"scope_strip": "CIRCUS … | Shortcut-Free (SF) | 0 lượt phục vụ · 0 có dữ liệu ở phạm vi này"
"final_search": ""
```

Đây chính là W1 ở tầng nhỏ hơn: số 0 được in cho một scope **chưa được đếm xong**. Sửa: `scope-strip`
có `data-scope-state` và ba nhánh — `error` → "Không tải được danh sách lượt — đây là lỗi yêu cầu,
không phải 0 lượt"; chưa `ready` → "Đang tải danh sách lượt…"; `ready` → số thật. Đo lại:

```
"scope_strip": "… | Đang tải danh sách lượt…"          (khi chưa xong)
"scope_strip": "… | 17 lượt phục vụ · 17 có dữ liệu…"  (khi xong, Circus Full)
```

### b. Card "Lượt CIRCUS release" in số của CIRCUS dưới tiêu đề FashionIQ Original

`/api/composition/capabilities?subset=…` **không có tham số protocol** và luôn đếm CIRCUS. Với
`protocol=fashioniq_original_split`, trang in:

```
Lượt CIRCUS release
FashionIQ — Original Split (Full Gallery)
artifact 9 · measurement 9 · có rank từng query 9
Nguồn: /api/composition/capabilities?subset=sf
```

Chín row CIRCUS dưới tiêu đề FashionIQ Original. Sửa: khi protocol không khai báo subset, card đổi
thành **"Số đo native (chưa xác minh)"** và đọc `/api/native-measurements` lọc theo protocol đó.

### c. "0 lượt" trên protocol chỉ có báo cáo tổng hợp

Cùng trang, scope strip in `0 lượt phục vụ · 0 có dữ liệu`. Đó là đúng về composition nhưng đọc như
"workbench không có gì". Sửa: khi `served === 0` và protocol có report, in
`0 lượt composition (protocol này không có) · 13 báo cáo tổng hợp có số đo` + nút
`Mở 13 báo cáo tổng hợp`. Đo được (sau00/after01):

```
FashionIQ — Original Split (Full Gallery)
0 lượt composition (protocol này không có) · 13 báo cáo tổng hợp có số đo
Mở 13 báo cáo tổng hợp
Số đo native (chưa xác minh)
FashionIQ — Original Split (Full Gallery) · 3 file có số · KHÔNG vào leaderboard đã xác minh
sprc  R@10 51.04% · R@50 73.33%   sprc_vitl_run1.json
tme   R@10 54.92% · R@50 74.84%   tme_fashioniq_0.json
tme   R@10 54.92% · R@50 74.84%   tme_normal_path_full.json
Nguồn: /api/native-measurements — 15 file có số, 4 file chỉ ghi blocker, 0 đã xác minh.
```

## 4. §6 display contract — đã làm / còn lại

Đã làm:

* **Ba lối vào**: area `overview` (tổng quan + inventory), `evaluation/composition` (CIRCUS), `evaluation/benchmark` (FashionIQ native aggregate) — đã có sẵn và đã kiểm trên service.
* **Sáu đếm tách rời** (`scopeCounts`: served / completed / artifactPresent / membershipDerived / verified / failed) + test khẳng định `verified === 0` cho composition row — không được claim system-wide.
* **Nhãn 5 phần** `resultLabels`: measurement · verification · coverage · blocker · samples; có test cho đúng các câu operator yêu cầu ("Đã đo" + "Chưa xác minh", "Không có hạng từng query", "Có hạng · chưa xuất ứng viên", "Đã chấm 6,001/6,003").
* **Nhãn subset có scope**: chỉ in khi protocol khai báo subset.
* **Chip trạng thái** `QUERY_STATE_CHIPS` (All / Correct@10 / Miss@10 / Unscored) định nghĩa trên contract backend `query_state`, có test.

**Còn BLOCKED / chưa làm** (lý do chính xác):

1. **`ModelInventory` chưa dùng `otherScopeFor`** — hàm + test đã có, chưa nối vào JSX. Lý do: cần sửa bảng inventory (224 dòng) và chưa kịp build lại + đo lại trong ngân sách lượt này.
2. ~~Màn hình catalog native~~ — **đã xong** ở mức card tổng quan (§3bis.b): hiện 8 row đầu, số file, số blocker, `0 đã xác minh`, và `note` nguyên văn của API. Chưa có bảng đầy đủ 15 row + sha256/path (dữ liệu đã có trong payload, chỉ thiếu bảng).
3. **Dải chip trong explorer** + **"Xem mẫu" cho mọi run có dữ liệu**: `SamplesPage` hiện dùng select trạng thái trong "bộ lọc nâng cao"; chip chưa render. `OverviewPage` đã có nút "Inspector" nhưng chưa áp nhãn 5 phần.
4. **Ảnh đích ngoài Top-K**: `QueryInspector` render mọi `positive_ids` trong `query-brief` (mỗi ảnh một tile) nhưng **không** in hạng cho từng ảnh đích; `ranks` chỉ có một hạng cho điều kiện, không có hạng per-positive. Không thể bịa hạng per-positive — dữ liệu không có.
5. **Clamp K và MM/Text/Image cạnh gallery**: `candidateDisplayOptions(depth)` đã clamp theo `depth` đã xuất (có test ở `light-workspace.test.cjs`), và select điều kiện ở trên; chưa dời vào cùng hàng gallery.

## 5. Lệnh build / restart đã dùng

```
npm run build --prefix workbench/frontend
PID=$(pgrep -f "serve_workbench.py --production-frontend" | head -1); kill $PID
setsid nohup workbench/envs/ctl312h/bin/python workbench/scripts/serve_workbench.py \
  --production-frontend --host 127.0.0.1 --backend-port 8000 --frontend-port 5173 \
  > /tmp/workbench_serve.log 2>&1 < /dev/null &
```

Lần restart đầu dùng `kill 2759507` (launcher của lượt trước); ba lần sau dùng `pgrep` vì launcher
giữ PID mới mỗi lần. Bundle cuối: `dist/assets/index-CG4hbuvP.js` + `index-DAZAlgSy.css`, và
`GET /` trỏ đúng hai file đó.

Xác nhận bundle được phục vụ tồn tại trong `dist/assets/` (mỗi lần build):

```
GET http://127.0.0.1:5173/  -> 200
index.html trỏ assets/index-CG4hbuvP.js   (có trong dist/assets/)
                assets/index-DAZAlgSy.css  (có trong dist/assets/)
```

Kiểm API sau restart: `benchmark-runs` 200 · `native-measurements` 200 · `composition/runs` 200 ·
`evidence/summary` 200.

## 6. Test đã chạy

```
workbench/envs/ctl312h/bin/python -m pytest workbench/tests/test_web_scope_and_labels.py -q
-> 12 passed

workbench/envs/ctl312h/bin/python -m pytest \
  test_web_scope_and_labels.py test_frontend_routes.py test_frontend_catalog.py \
  test_frontend_requests.py test_frontend_format.py -q
-> 51 passed
```

Test này chạy module shipping qua node (`test_frontend_support.compile_module`), không phải bản copy.

## 6bis. Trạng thái "0" còn lại cần nói rõ

`/api/runs` trả `[]` vì DuckDB index rỗng (0 run / 0 query / 0 top_results, đọc bằng
`duckdb.connect(..., read_only=True)`). Đây là **dữ liệu**, không phải lỗi: index được build từ
`WORKBENCH_RESULTS_ROOT = workbench/artifacts/results` và thư mục đó rỗng (`ls` = 0 entry, chỉ có
`replay-cache/` và `tmp/`). Vì vậy mọi con số per-query index trên web là 0 và **không** được sửa
cho khác đi; ba nguồn còn lại (composition 35 row, benchmark 13 report, native 15 file) là chỗ dữ
liệu thật nằm.

## 6ter. File tôi đã sửa (và file KHÔNG phải của tôi)

Sửa trong lượt này:

```
workbench/frontend/scope.ts              (mới)
workbench/frontend/data.ts
workbench/frontend/routes.ts
workbench/frontend/App.tsx
workbench/frontend/OverviewPage.tsx
workbench/frontend/EvaluationPage.tsx
workbench/frontend/ExperimentsPage.tsx
workbench/frontend/SamplesPage.tsx
workbench/backend/native_measurements.py (mới)
workbench/backend/subset_provenance.py   (mới)
workbench/backend/main.py
workbench/tests/test_web_scope_and_labels.py (mới)
workbench/docs/reviews/20261110-e2e-repair/  (mới)
workbench/frontend/dist/**               (chỉ qua `npm run build`)
```

Thay đổi **song song của phiên khác**, tôi không chạm và không đụng: `workbench/backend/limn_diagnostic.py`,
`workbench/tests/test_limn_dual_scope_diagnostic.py`, `workbench/backend/parity_gate.py`,
`workbench/scripts/migrate_protocol_label.py`, `workbench/tests/test_gate_parity_subchecks.py`,
`workbench/tests/test_limn_diagnostic_record.py`, `workbench/tests/test_protocol_label_migration.py`,
`CLEANUP_REPORT.md.save`. Không commit gì (một phiên khác sở hữu git).

## 7. Việc cần người khác xử lý (không sửa `workbench/registry/**`)

Không có đề xuất sửa registry. `DEFERRED_MODELS` (`csmcir`, `sprc`) vẫn là hằng trong
`ModelInventory.tsx` và vẫn đúng theo registry hiện tại; dạng đúng là một field `deferred_reason`
phục vụ từ `/api/models` — đây là thay đổi **backend/registry**, chưa làm vì nằm ngoài phạm vi được
giao ở lượt này và cần người sở hữu registry quyết định.

---

# Phụ lục A — D1 / D2 / D3 (lượt sửa thứ hai)

Ba defect consolidator chứng minh. Tất cả **CONFIRMED**, đã sửa, đã test, đã verify trên service thật.

## A.1 D1 — `native_measurements.py` lặp lại lỗi "census v1"

### Verdict: CONFIRMED

Bản đầu chỉ đọc `aggregate.{macro_r10,macro_r50,macro_mean}` qua hai glob, nên bỏ sót 4/5 hình dạng
có thật trong cây và gọi kết quả là "census". Đếm trước/sau (cùng artifacts root):

| | trước | sau |
|---|---|---|
| file có row | 21 | **32** (quét 71 file JSON) |
| block có metric | 15 | **43** |
| measurement thật (không roll-up) | 15 | **21** |
| row roll-up | 0 | **22** |
| model có metric | 15 | **28** |
| model chỉ có số trong roll-up | 0 | **13** |
| file không đọc được | 1 | 2 (`ptha_transfer_run2.json`, `csmcir/call_path_probe.json`) |
| blocker có dấu hiệu | 4 | 4 |
| verified | 0 | **0** |

### Hình dạng đã hỗ trợ (mỗi cái có test riêng)

| Hình dạng | File | Row tiêu biểu |
|---|---|---|
| `aggregate.{macro_r10,macro_r50,macro_mean}` | 12 file measurement + cohort | `conesep` 54.7613, `encoder` 56.2231, `limn`, `tme`, `airknow`×2, `clvc_net`, `dcnet`, `sprc`, `tgcir`, `habit`, `hint`, `intent` |
| `macro.{macro_r1,macro_r10,macro_r50,macro_mean}` | `pair_b1_H_normal/rev`, `pair_b2_H_normal/rev` | `pair` 34.0350 / 35.0147 / … , protocol `fashioniq_val_split` |
| `macro_metrics.{r10,r50,mean}` | `ptha_transfer_run1.json` | `ptha_mtst` **37.8978**, `fashioniq_original_split` |
| `macro_r10` ở gốc document | `csmcir/measurement.json` | `csmcir` **57.0507**, `fashioniq_original_split`, `metric_path = "(top level)"` |
| `<variant>.macro_r10` | `clip4cir_parsed.json` | `clip4cir_noft` 35.1407 + `clip4cir_fullft` 43.8182, **hai row**, `rollup: true` |
| roll-up tổng | `all_measured_summary.json`, `selection_decisions.json`, `pair_compat/manifests/pair_compat_verdict.json`, `pair_compat/measurements/summary.json` | `airknow_n05`, `airknow_n08`, `clvc`, `tgcir`, `combiner_rn50x4_noft` 35.1407, `clip4cir_rn50x4_fullft` 43.8182, `sprc_vitl`, `B1/H_*`, `B2/H_*` |

`/api/native-measurements` (live, 200) chứa **tất cả** model được yêu cầu, mỗi row có
`verified: false`, `verification.state = UNVERIFIED_NATIVE_MEASUREMENT`, `source.path` và
`source.sha256` (0 row thiếu sha):

```
ptha_mtst True · pair True · clip4cir_rn50x4_fullft True · combiner_rn50x4_noft True · csmcir True
models = 28
```

### Hai luật chống trùng

1. **Roll-up được phục vụ, bị loại khỏi đếm per-model.** `counts.with_metrics` chỉ đếm row
   `rollup:false`; `counts.rollup_rows` đếm phần còn lại; `counts.models_rollup_only` đếm model chỉ
   xuất hiện trong roll-up (13). `rollup_sources` liệt kê 5 file. Vì vậy `combiner_rn50x4_noft`
   **có** row (nó chỉ tồn tại trong `selection_decisions.json`) mà không bị tính là một measurement
   thứ hai của chính nó.
2. **Block tham chiếu giấy không phải measurement.** `dcnet_run1.paper` và
   `pair_*.paper_reference` bị loại (`EXCLUDED_TOKENS`), nếu không số của bài báo sẽ được phục vụ
   như kết quả của deployment này.

Thêm luật thứ ba phát hiện trong lúc sửa: `r10`/`r50` trần **chỉ** được đọc khi block nằm dưới khoá
macro-shaped (`aggregate`/`macro`/`macro_metrics`/`metrics` hoặc gốc). Nếu không,
`all_measured_summary.json` có 3 block `dress`/`shirt`/`toptee` mỗi block đọc thành một macro
measurement.

### Không hỗ trợ được (báo cáo trung thực)

* `hint_native_b16.json`, `hint_native_verify.json`, `hint_parity_top50.json`, `tme_bounded_equiv.json`,
  `tme_real_cache_smoke.json`, `clip4cir_fullft_distance.json`, `clvc_distance.json`,
  `tgcir_distance.json`, `sprc_distance.json`, `dcnet_patched_cli.json`: **không** phục vụ.
  Chúng là so sánh/parity/distance, hoặc có macro chỉ ở dạng `macro_fused_r10` (một biến thể fused
  của CÙNG run `dcnet` đã có `aggregate` trong `dcnet_run1.json` — phục vụ thêm sẽ trình bày một
  model hai lần). Không có hình dạng nào trong số này là một macro của một model chưa có row.
* `clip4cir_fullft_distance.json` có `cells` per-category chứ không có macro cho một model mới;
  `clip4cir_rn50x4_fullft` đã có row từ roll-up.
* `cohort_summary.json` là **list** các object per-model: `_blocks` đi được vào list, nhưng file này
  không có macro block nên không tạo row metric (đúng: nó chỉ lặp lại cohort membership).

### File / test

```
workbench/backend/native_measurements.py        (viết lại: walker theo độ sâu + roll-up + excluded)
workbench/tests/test_native_measurements_census.py  (16 test, mới)
```

## A.2 D2 — banner bằng chứng nói dối khi đang tải

### Verdict: CONFIRMED

`evidenceState` tính `noRealEvidence = aggregateRuns === 0 && experimentRuns === 0` trên
`region.data ?? []`, nên trước khi region nào settle, `evidenceBanner` in
`MOCK DEVELOPMENT DATA — no real evaluation results are loaded`. Đó là khẳng định về dữ liệu chưa
tới. Đúng lớp defect đã sửa cho `scope-strip`.

### Sửa

`workbench/frontend/evidence.ts`:

* thêm `EvidenceSourceStatus` / `EvidenceSources` và hai field `loading`, `failedSources`,
  `pendingSources` trong `EvidenceState`;
* `evidenceState(..., sources?)` — nguồn không được nêu được coi là đã settle, nên mọi caller cũ
  (test, CLI) giữ nguyên nghĩa;
* `evidenceBanner` có nhánh **đầu tiên** cho loading/failure: `Đang tải bằng chứng ·
  Không tải được <nguồn> — đây là lỗi yêu cầu, không phải 'không có kết quả' · N đã chứng nhận trong
  M report đã về`. Nhánh này không bao giờ in "no real evaluation results".

`workbench/frontend/App.tsx` truyền `data.indexRuns/aggregates/compositionRuns.region.status` và đặt
`data-evidence-loading` lên banner để harness đọc được.

### Test

```
workbench/tests/test_web_evidence_banner_loading.py — 7 test
  test_a_loading_source_is_never_reported_as_no_evidence      <- failing-first
  test_an_idle_source_also_counts_as_unsettled
  test_a_failed_source_is_reported_as_a_failure_not_as_zero
  test_a_settled_empty_deployment_still_says_so
  test_data_that_arrived_while_another_source_loads_is_counted_not_claimed_absent
  test_a_caller_that_states_no_statuses_keeps_the_old_meaning
  test_the_shell_passes_the_region_statuses
```

Failing-first chứng minh: chạy bộ test này trên `evidence.ts` của HEAD (`git show HEAD:…`) →
**6 failed** với `KeyError: 'loading'` / `'failedSources'`. Sau khi sửa → **7 passed**
(`web_evidence/before-tests-d2-failing-first.log`).

### Đo trên service thật (Chromium, poll 400 ms suốt cold start)

```
t=0  loading=true   "Đang tải bằng chứng"
t=1  loading=true   "Đang tải bằng chứng · 2 đã chứng nhận trong 13 report đã về"
t=3  loading=None   "AGGREGATE-ONLY REAL EVALUATION — 2 certified research results … 12 of 17 composed runs in view"
false-claim-seen: False
```

## A.3 D3 — kỷ luật restart

### Verdict: CONFIRMED

Backend `:8000` cũ (pid 2788121, khởi động 08:16:44) cũ hơn `main.py` (08:29:09) → `/api/runs/parity-gate`
trả 404, và bundle frontend mới hơn backend.

Đã chạy đúng chuỗi:

```
npm run build --prefix workbench/frontend          # -> assets/index-iNen-aFi.js
PID=$(pgrep -f "serve_workbench.py --production-frontend" | head -1); [ -n "$PID" ] && kill $PID; sleep 3
setsid nohup env WORKBENCH_CIRCUS_RELEASE_ROOT=/home/heheboiz/data/cir/workbench/artifacts/circus_release \
  workbench/envs/ctl312h/bin/python workbench/scripts/serve_workbench.py \
  --production-frontend --host 127.0.0.1 --backend-port 8000 --frontend-port 5173 \
  > /tmp/workbench_serve.log 2>&1 < /dev/null &
```

### Xác nhận

| Kiểm | Kết quả |
|---|---|
| backend pid / giờ khởi động | **2820650**, `Tue Nov 10 08:47:09 2026` — **sau** `main.py` 08:29:09 |
| `GET /api/runs/parity-gate` | **200** |
| `passed_models` | **`[]`** (0 model); `models` = `["limn","encoder"]` cả hai NOT_RUN |
| `GET /api/native-measurements` | 200, 32 file / 28 model / 0 verified |
| `GET /api/benchmark-runs` | 200 |
| `GET /api/runs` | 200 |
| bundle phục vụ | `assets/index-iNen-aFi.js` = file trong `dist/assets/` |
| landing | `?area=overview&protocol=circus_fashioniq_release_v1&subset=full`, `17 lượt phục vụ` |
| page errors | không có |

Bằng chứng: `web_evidence/after-parity-gate.json`, `after-native-measurements.json`,
`after-banner-loading.txt`, `after-tests-d1.log`, `after-tests-d2.log`,
`before-tests-d2-failing-first.log`, `r13-native-census-before-d1.txt`.

## A.4 Test đã chạy sau khi sửa

```
workbench/envs/ctl312h/bin/python -m pytest workbench/tests/test_native_measurements_census.py -q   -> 16 passed
workbench/envs/ctl312h/bin/python -m pytest workbench/tests/test_web_evidence_banner_loading.py -q  -> 7 passed
```

Không commit. Không sửa `workbench/config/workbench.env`, `parity_gate.py`, `limn_diagnostic.py`,
`registry/`, `composition/`, hay bất cứ gì dưới `20261109-native15-composition-limn/`.

## A.5 Xác nhận số phục vụ = số trong file nguồn

Không chỉ đếm row: mỗi hình dạng mới được đối chiếu với chính file nguồn (script
`verify_nums.txt`, lưu `web_evidence/after-native-numbers-verified.txt`).

| model | khoá nguồn | giá trị nguồn | giá trị phục vụ | khớp |
|---|---|---|---|---|
| `csmcir` | `macro_r10` (gốc) | `57.05073873202006` | `57.05073873202006`, `metric_path="(top level)"`, `measurement.json` | ✅ 1e-12 |
| `ptha_mtst` | `macro_metrics.r10` | `37.897844115893044` | `37.897844115893044`, `metric_path="macro_metrics"`, `ptha_transfer_run1.json` | ✅ 1e-12 |

Card native trên trang `?protocol=fashioniq_original_split` (đo bằng Chromium, ảnh chụp state ở
`web_evidence/after-native-card-live.txt`) hiện đúng 5 row: `csmcir 57.05%`, `ptha_mtst 37.90%`,
`sprc 51.04%`, `tme 54.92%` ×2 — và câu `21 file có số, 4 file chỉ ghi blocker, 0 đã xác minh`.
`tme` xuất hiện hai lần là đúng: `tme_fashioniq_0.json` và `tme_normal_path_full.json` là hai lần
đo riêng, cùng số vì cùng checkpoint — không phải một row bị nhân đôi.

## Nghiệm thu

Ngày 2026-11-10 · HEAD `2e2e2d2` + thay đổi của lượt này (không commit) · dịch vụ `http://127.0.0.1:5173/`
Bundle đang phục vụ: `dist/assets/index-DjAje2d2.js` (367.32 kB, gzip 108.27 kB) + `index-DecQnta9.css`,
`GET /` trỏ đúng hai file đó. Service chạy **có** biến môi trường (đã đọc `/proc/<pid>/environ`):
`WORKBENCH_CIRCUS_RELEASE_ROOT=/home/heheboiz/data/cir/workbench/artifacts/circus_release`.

Công cụ: `workbench/docs/reviews/20261110-e2e-repair/scripts/web_acceptance.py`
(Playwright/Chromium 158, interpreter `workbench/envs/ctl312h/bin/python`), chạy lại được toàn bộ:
`workbench/envs/ctl312h/bin/python workbench/docs/reviews/20261110-e2e-repair/scripts/web_acceptance.py`
→ `ACCEPTANCE OK`, `raw/web_acceptance.json` là kết quả máy kiểm được.

### 1. Năm mục còn lại — đã đóng

| # | Nội dung | Bằng chứng đo được |
|---|---|---|
| 1 | `otherScopeFor` nối vào `ModelInventory` | `subset=validated`: `combiner_rn50x4_noft` (không có lượt ở scope này) render nút `other-scope-combiner_rn50x4_noft-fashioniq_original_split-all` → "FashionIQ — Original Split (Full Gallery) · 13 số đo (toàn phạm vi)". Nguồn dòng gồm **ba** nơi đo: composition rows, `/api/native-measurements`, `/api/benchmark-runs`; link mở đúng view của nguồn đó. Hàng không có kết quả ở đâu vẫn là "Cần chạy" + bước tiếp theo (6 hàng). |
| 2 | Dải chip trong explorer | 4 chip render từ `QUERY_STATE_CHIPS` với số của chính nó: `Tất cả 6,003 · Correct@10 1,802 · Miss@10 4,199 · Unscored 2`, khớp `state_counts` của API (backend đếm **trước** bộ lọc trạng thái, nên chip không active vẫn có số). Chip ghi `query_state` vào URL; `offset=20` giữ `query_state=hit`; **reload** và **Back** khôi phục đúng URL đã lưu. Category/outcome/sort/imported nằm trong `<details>` đóng sẵn (`open:false`). |
| 3 | Gallery ứng viên | Run `9e4918a6b5c2…` (limn/full, depth 100), query `7:3045`. K=10 → 50 → 100 cho **đúng** 10/50/100 tile; ID, `#rank` và điểm khớp export từng phần tử (so với `/api/composition/runs/<run>/queries?top_k=K`). Thứ tự DOM: ảnh tham chiếu → caption → ảnh đích → ghi chú ngoài cửa sổ. Request key có run/protocol/subset/condition/query/K. Ba trạng thái tách rời: `miss` (7:3045, hạng 73,369), `missing-rank` (7:3038, cả ba `Chưa chấm`), `missing-export` (`candidates-unavailable`), ảnh thiếu `data-asset-state=missing` (một id không tồn tại → `/api/images/...` 404). Stale: chuyển 7:3045 (làm chậm 3 s) → 7:1466, sau 4 s chờ title vẫn là `Query 7:1466`. |
| 4 | `LimnScopeDiagnostic` (patch §3.3) | `Shot` export từ `QueryInspector.tsx` và dùng lại — **một** gallery. Bỏ `&depth=20`; `depth_served/depth_stored/depth_requested` in từ response (100/100/100). `d.target` render **luôn**, tách khỏi cửa sổ, kèm "ngoài cửa sổ 100 ứng viên đang hiện" khi `in_window=false` (7:3790/shirt/text-only: hạng 6,891). `complete=false` + `defects` hiện badge và danh sách `code`. Đoạn văn "Ảnh không được render ở đây" đã xoá (assert trên `body`). 12/12 tổ hợp mở được với 100 tile + ảnh đích decode thật. |
| 5 | `GET /api/runs/parity-gate` | `evaluate_all()` phục vụ nguyên payload. `passed_models: []`; `limn` và `encoder` đều `PARITY_MANDATORY_SUBCHECK_FAILED` (3 và 13 subcheck FAIL). Trang Đánh giá render card "G1/G2 — cổng chẵn lẻ (từ artifact)" với badge `KHÔNG mô hình nào PASS (0/2)` và câu nói rõ đó **là kết luận**, không phải lỗi tải. |

### 2. Ảnh chụp (23 ảnh của lượt này; đọc ở độ phân giải gốc)

`web_evidence/1920-01-landing-bare-url.png` (CIRCUS **Full**, 17 lượt) ·
`1920-02-fashioniq-original.png` (không có control Subset, "0 lượt composition … 13 báo cáo tổng hợp") ·
`1920-03-subset-{full,sf,validated}.png` (17/9/9 hàng, khớp API) ·
`1920-04-scope-no-data.png` · `1920-05-error-state.png` (API 500 → `empty-kind=error`) ·
`1920-05b-url-problem.png` (`subset=bogus` → "chỉ nhận full, sf, validated") ·
`1920-06-loading-state.png` (`data-scope-state=loading`) ·
`1920-07-inspector-query-images.png` · `1920-07b-inspector-missing-rank.png` ·
`1920-08-topk-{10,50,100}.png` · `1920-09-chip-correct-at-10.png` · `1920-10-advanced-collapsed.png` ·
`1920-11-stale-response-guard.png` · `1920-12-limn-outside-window.png` · `1920-13-limn-defects.png` ·
`1920-14-parity-gate.png` · `1920-15-model-inventory-validated.png` ·
`2560-16-inspector.png` · `390-16-inspector.png`.

Viewport: 1920×1080, 2560×1440, 390×844 — `docWidth == viewport` ở cả ba (2560: 2560/2560, 390: 390/390),
nav hiện. Console sạch trên các trang được kiểm; lỗi console duy nhất là **cố ý** (context tự phục vụ 500
để chụp trạng thái lỗi), ghi riêng ở `intentional_console_errors`.

### 3. Định thời

Đo bằng `performance`-free timing trong cùng một phiên Playwright: mốc bắt đầu = `goto`, mốc kết thúc =
`data-shell-state=ready` **và** số đã render (không dùng `networkidle`, không chờ `/api/models`).

| Phép đo | Lần nguội | Lần ấm |
|---|---|---|
| Tải trang → số hiện trên màn hình | **510 ms** | **597 ms** |
| Chọn query → ảnh đầu tiên decode xong | **455 ms** | — |

(Lượt đo trước trong cùng phiên: 2,119 ms / 613 ms — dao động do cache HTTP của Chromium, không do server.)

Nút cổ chai còn lại, đo được chứ không phỏng đoán: **`GET /api/models`**. Nó băm mọi checkpoint trong
registry nên lần đầu trả **153.7 s** (đã đo bằng `curl`), lần sau **0.061 s**; các endpoint khác 0.004–2.8 s
(`/api/runs/parity-gate` 0.004 s, `/api/benchmark-runs` 2.8 s nguội). Vì vậy: (a) readiness không bao giờ
chờ endpoint này, (b) màn "Tất cả mô hình" là chỗ duy nhất cần nó và hiện `LoadingState` "Đang đọc danh sách
mô hình…" cho tới khi tới — đúng, nhưng nghĩa là **mở trang inventory ngay sau khi restart là chậm**, và đó là
giới hạn đã biết, không phải lỗi mới. Sửa đúng chỗ là cache/incremental hoá việc băm checkpoint ở backend.

### 4. Test đã chạy

```
workbench/envs/ctl312h/bin/python -m pytest workbench/tests/test_web_scope_and_labels.py \
  workbench/tests/test_composition_topk_api.py workbench/tests/test_frontend_routes.py \
  workbench/tests/test_frontend_requests.py -q   -> 34 passed
```

### 5. Còn mở (nói rõ, không giấu)

* `LIMN` `defects` là **6 defect thật** trên `7:3038` (`target_did_not_own_query`): artifact ghi target
  `7:42304` không phải positive của query (positive riêng là `7:3416`). Panel render đúng defect đó; **không**
  sửa artifact, không đổi số (ngoài quyền của lượt này).
* `/api/models` 153.7 s nguội — xem §3.
* `CLEANUP_REPORT.md.save` vẫn untracked, không chạm. Không commit gì.

## A.6 Hồi quy: thay đổi `evidence.ts` không phá caller cũ

`evidenceState` đổi signature (thêm tham số thứ 5 tuỳ chọn), nên các test đã có phải vẫn xanh. Đã
chạy từng module (một lượt chạy gộp 5 module bị timeout 600 s vì `test_e2e_release_chain` dựng chain
thật, không phải vì fail):

| Module | Kết quả |
|---|---|
| `test_frontend_staleness.py` | **7 passed** |
| `test_a39_gate_certification.py` | **1 passed** |
| `test_release_blocker_gpu_records.py -k "evidence or banner"` | **9 passed** (26 deselected) |
| `test_e2e_release_chain.py` — contract banner | chạy lại bằng chính module shipping (support.compile_module), không dựng chain: banner = `AGGREGATE-ONLY REAL EVALUATION — 1 certified research result; …` → **không** claim "no real evaluation results", **có** `AGGREGATE-ONLY` (`web_evidence/after-release-chain-banner-contract.txt`) |
| gộp: `test_native_measurements_census` + `test_web_evidence_banner_loading` + `test_web_scope_and_labels` + `test_composition_api` | **49 passed** |

Vì `test_e2e_release_chain.py` gọi `evidenceBanner(evidenceState([], [row]))` với **hai** tham số,
contract đó nay được pin thêm bằng một unit test riêng
(`test_a_certified_aggregate_without_region_state_still_reads_aggregate_only`), để một lần đổi nhánh
loading nữa sẽ hỏng ở đây thay vì trong một chuỗi E2E mất 10 phút.
