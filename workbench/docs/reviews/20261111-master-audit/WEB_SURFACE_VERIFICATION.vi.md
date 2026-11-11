# W-01…W-05 — Xác minh bề mặt web trên service thật

Repo `/home/heheboiz/data/cir`, nhánh `research/cir-failure-workbench`, HEAD `92ed66de4f359ef479da18bf8939723b6f9313e5`
+ thay đổi chưa commit của nhiều worker. Không commit gì trong lượt này.

Service: frontend `http://127.0.0.1:5173`, backend `http://127.0.0.1:8000`.
Uvicorn trên `:8765` là tiến trình cũ thiếu route mới — **không** dùng để đo.

## Trạng thái bàn giao (đọc mục này trước)

| issue_id | status | evidence file | artifact | API response | UI screenshot | fix | regression |
|---|---|---|---|---|---|---|---|
| W-01 | **XONG** | `raw/w01_w03_targeted_tests.txt` | `workbench/backend/native_measurements.py`, `workbench/frontend/{format,data,EvaluationPage,OverviewPage}.ts*` | `w01_native_measurements_BEFORE_backing.json` → `w01_native_measurements_after.json` | `w_shots/w01_native_table.png`, `w_shots/w01_native_overview.png` | prune path `paper*`; unit theo declaration rồi mới suy; `mapping_status`; `undeclared_unit_rows`; **roll-up không cấp số đo thiếu artifact**, alias gộp, `counted`/`duplicate_of` | `test_native_measurements_census.py` (28), `test_frontend_format.py` |
| W-02 | **XONG** | `raw/w02_*_notes.txt` | `workbench/frontend/{scope,ModelInventory,EvaluationPage,data,routes}.ts*` | `w_shots/w02_models_inventory.json`, `w_shots/w02_native_scope_populated.json` | `w_shots/w02_overview_circus.png`, `w_shots/w02_models_inventory.png`, `w_shots/w02_native_scope_populated.png` | view `native`; link theo `family`; 9 đếm tách rời | `test_web_scope_and_labels.py`, `test_frontend_routes.py` |
| W-03 | **XONG** | `raw/w03_before_summary.txt`, `raw/w03_after_summary.txt` | `workbench/backend/limn_diagnostic.py`, `workbench/frontend/{QueryInspector,LimnScopeDiagnostic}.tsx` | `w03_limn_3038_toptee_mm_depth{10,20,43,100}.json`, `pre_limn_*` | `w_shots/w03_inspector_7-3038_limn_full.png`, `w03_inspector_midrank_encoder.png`, `w03_limn_diagnostic_toptee.png`, `w03_limn_probe_toptee_mm_K10.png` | presence trên window được serve; hai depth; probe có control K | `test_limn_dual_scope_diagnostic.py` (17) |
| W-04 | **XONG** | `web_evidence/w04/w04_results.json` | — | 503 cưỡng bức trong `w04_induced_api_error.json` | `w04_late_answer_a_after_b.png`, `w04_deeplink_out_of_range.png`, `w04_reload_and_history.png`, `w04_induced_api_error.png` | subtitle không in 0 khi region lỗi | `test_frontend_requests.py` + 4 kiểm browser |
| W-05 | **XONG** | xem mục W-05 | — | `w01_native_measurements_after.json`, `w05_benchmark_runs_after.json`, `w05_parity_gate_after.json` | 17 ảnh trong `w_shots/` (13 trong `manifest.json`) | — | — |

## Trạng thái nửa chừng nhận lại (trước khi sửa)

`git status --porcelain` khi nhận: 21 file tracked đã sửa, `.scratch/` chưa được gitignore,
`CLEANUP_REPORT.md.save` chưa track.

Frontend đã ở trạng thái **không build được**:

```
OverviewPage.tsx(256,70): error TS2304: Cannot find name 'fmtPercentValue'.
```

Nguyên nhân: worker trước đổi `import` sang `fmtRecall` nhưng để lại một chỗ gọi `fmtPercentValue`
ở bảng composition. Đã sửa thành `fmtRecall(run.metrics?.recall_at_10_multimodal, "percent", 1)`
(nhánh này là recall đã ở thang percent, nên `"percent"` là khai báo đúng, không phải suy đoán).

Bài học giữ lại: `dist/` được build từ cây hỏng thì service phục vụ bundle hỏng mà không có lỗi
nào ở phía server. Luôn `npm run build` trước khi restart.

## Service được phục vụ (W-05: pin)

| mục | giá trị |
|---|---|
| bundle **đang phục vụ** | **`assets/index-DWpK0yeS.js`** sha256 **`7e34724ebf55092e36524b98980cca55e71f3ddbaa62489c65577f92e18a6a16`** |
| css | `index-DecQnta9.css` |
| lý do phải pin | `workbench/frontend/dist/` bị gitignore nên **artifact build không được track**; build thì tái lập được (`npm run build --prefix workbench/frontend` — nguồn, `tsconfig.json`, `vite.config.mts`, `package.json`/`package-lock.json` đều tracked và `tsc`/`vite` đã cài), nhưng **byte đang phục vụ phải được pin bằng digest tại mỗi lần đo** |
| superseded — bundle phục vụ trước `relation_state`/`relation_parity` | `index-CEgaMiLf.js` sha256 `0b6b15a80782d04459e78e3bd71fc0d2532a8c16bb09d01987814d16bab6eb87` — **không còn trên đĩa**, đã bị build mới ghi đè |
| superseded — bundle của lần W-05 đầu | `index-Bpang-Qk.js` sha256 `81371928af647c4a8ee994f060a2fd268ed76a04c2dc4855928692b32626b414` — **không còn trên đĩa** (tệp không tồn tại); đây là build trước khi có các bản sửa ingest/`relation_state`/`relation_parity` |
| superseded — bundle trước lần nhận W-05 | `index-DjAje2d2.js` sha256 `74a9319ed6f05ce0508cf93424d5a0dcd8909e9a0116dcec4e87a1f24d50f867` — **không còn trên đĩa** |
| frontend | `http://127.0.0.1:5173` (vite `preview`, proxy `/api` → `http://127.0.0.1:8000`) |
| backend | `http://127.0.0.1:8000` |
| restart | đúng **một lần** trong lượt W-05 này (stack đã restart nhiều lần trong cả đợt — xem hàng PID bên dưới), `WORKBENCH_CIRCUS_RELEASE_ROOT=/home/heheboiz/data/cir/workbench/artifacts/circus_release` |
| PID **đang phục vụ** | `serve_workbench.py` **4171850** → uvicorn **4171851** (superseded: `4082631/4082632`, `4062950/4062957`, `4040573/4040574`, `3390138/3390139`, `3316577/3316578`, `3228456`, `3212146`) |
| cảnh báo về pin PID | PID **không bền** trong đợt này: stack đã chết **ít nhất bốn lần** do SIGTERM từ bên ngoài, mỗi lần một PID khác, và **số chính xác không chốt được** — mỗi lần restart lại `> /tmp/…log` nên log cũ bị truncate và chữ ký của lần trước đó biến mất. Bốn lần còn log header: `3691345` (`/tmp/workbench_serve.log`, mtime `2026-11-11 02:25:51`; mốc thời gian lấy từ mtime của tệp, **không** có trong nội dung log), `4040574` (`/tmp/workbench_serve_final.log`), `4062957` (`/tmp/serve_probe.log`), `4082632` (`/tmp/serve_handover.log`, mtime `2026-11-11 07:29:33`). Tất cả có đúng chữ ký `[WARN] npm exited with status -15` rồi backend `Shutting down` êm. Bản sao đã commit: `raw/service-death-log.txt` (lần 1) và `raw/service-death-log-4.txt` (lần 4). Đừng đọc con số này như một pin: nó chỉ tăng và mỗi log mới truncate log cũ. **Digest bundle mới là pin bền**; PID chỉ đúng tại thời điểm đo |
| readiness | `/api/benchmark-runs` → 200 sau 1 s; **không** dùng `/api/models` (cold start ~154 s vì hash mọi checkpoint) |
| digest khớp | `sha256(served /assets/index-DWpK0yeS.js)` = `sha256(dist/assets/index-DWpK0yeS.js)` = `7e34724e…8a6a16`; `cmp` → IDENTICAL |

Lần restart này cũng là lần duy nhất nạp `workbench/backend/parity_gate.py` và
`workbench/composition/runner.py` do worker khác sửa. Verdict của gate **không đổi**:

```
standard=parity-relative/v1  passed_models=[]
encoder: PARITY_MANDATORY_SUBCHECK_FAILED  failures=13  checks=96
limn:    PARITY_MANDATORY_SUBCHECK_FAILED  failures=3   checks=51
```

`limn` có 51 check (trước là 48) và có thêm `missing`/`applied_standard`/`requested_standard` — đều
là field cộng thêm, nên kết luận vẫn là `PARITY_MANDATORY_SUBCHECK_FAILED` cho cả hai và không mô
hình nào PASS. Bằng chứng: `web_evidence/w05_parity_gate_after.json`.

## W-03 · Top-K và ảnh

### Lỗi đo được trên service thật, TRƯỚC khi sửa

`GET /api/composition/diagnostics/limn-dual-scope/candidates?qid=7:3038&scope=toptee&condition=multimodal&depth=…`

| depth | `in_window` | hạng đích | `depth_served` | hạng 43 có trong `candidates`? | đọc ra |
|---|---|---|---|---|---|
| 10 | **true** | 43 | 10 | **không** | SAI |
| 20 | **true** | 43 | 20 | **không** | SAI |
| 43 | true | 43 | 43 | có | đúng |
| 100 | true | 43 | 100 | có | đúng |

Nguyên nhân: `combination["target"]["in_window"]` được quyết định trên **toàn bộ 100 dòng đã lưu**,
không phải trên cửa sổ response đang trả. Vì vậy depth=10 trả `in_window: true` trên mười dòng
không chứa ảnh đích.

Bằng chứng: `web_evidence/pre_limn_3038_toptee_mm_depth{10,20,43,100}.json`,
tổng hợp `raw/w03_before_summary.txt`.

### Sau khi sửa (cùng URL, cùng service, sau restart)

| depth | `in_window` | `in_stored_window` | `depth_served` | `depth_stored` | hạng 43 trong window | nhất quán |
|---|---|---|---|---|---|---|
| 10 | **false** | true | 10 | 100 | không | có |
| 20 | **false** | true | 20 | 100 | không | có |
| 43 | true | true | 43 | 100 | có | có |
| 100 | true | true | 100 | 100 | có | có |

Bất biến được kiểm: `target.in_window == (target.rank ∈ {rank của candidates})` cho cả bốn depth.
Đây là điều duy nhất một view thật sự nhìn thấy.

Bằng chứng: `web_evidence/w03_limn_3038_toptee_mm_depth{10,20,43,100}.json`,
`raw/w03_after_summary.txt`.

### Sửa ở phía backend

`workbench/backend/limn_diagnostic.py::load_candidates` tính presence trên chính `window` được serve:

```python
served_ranks = {row.get("rank") for row in window}
in_window = isinstance(target_rank, int) and not isinstance(target_rank, bool) \
    and target_rank in served_ranks
```

và trả thêm `in_stored_window`, `window_served`, `window_stored` để hai depth là hai con số riêng,
không phải một suy luận.

### Sửa ở phía UI

**Inspector thường** (`QueryInspector.tsx`) — đây là chỗ W-03 yêu cầu kiểm riêng, không chỉ view
diagnostic:

* presence tính bằng `shown.some(c => positive_ids.includes(c.image_id))`, tức trên đúng mảng đang
  render, không phải `best > K`;
* note `data-testid="target-outside-window"` in cả `Top-K đang hiện`, `số ứng viên được serve` và
  `export lưu N dòng`;
* dòng đầu của card ứng viên in `export lưu N dòng, API serve M` và nói rõ ảnh đích **có/không** nằm
  trong danh sách;
* ảnh đích vẫn render từ query record (không từ candidate window) nên nó tồn tại kể cả khi ngoài K.

**View diagnostic** (`LimnScopeDiagnostic.tsx`): note đầu khối in `đích nằm trong/ngoài cửa sổ này`
và, khi khác nhau, `(trong toàn bộ N dòng đã lưu thì có/không)`; note ngoài cửa sổ nói thẳng "Với
K=10 hay K=20 thì ảnh đích **không** nằm trong cửa sổ".

Regression: `workbench/tests/test_limn_dual_scope_diagnostic.py`
`test_presence_is_decided_on_the_window_that_is_served[10/20/43/100]` và
`test_the_stored_window_is_reported_even_when_the_served_one_excludes_the_target`.

## W-01 · Điểm paper lẫn số đo, đơn vị, mapping status

### (a) `paper_reference.aggregate.macro_r10` không vào census

Gate cũ kiểm **leaf key**: block `aggregate` nằm trong `paper_reference` vẫn thoả `MACRO_BLOCK_KEYS`
nên macro của bài báo được đếm như số đo của deployment. Nay **cả đường dẫn** quyết định:

```python
excluded = any(token in part.lower() for part in trail for token in EXCLUDED_TOKENS)
```

Fixture tái hiện đúng ca này (`paper_reference.aggregate.macro_r10 = 52.76` bên cạnh
`aggregate.macro_r10 = 42.0`) nằm ở `raw/w01_w03_prescratch/fx_paper_nested.json`.
Regression `test_a_paper_reference_block_anywhere_in_the_path_is_not_a_measurement` khẳng định
`metric_path == ["aggregate"]`, giá trị 42.0, và 52.76 **không** xuất hiện ở bất kỳ dòng nào.

Trên catalog thật (`web_evidence/w01_native_measurements_after.json`):
`paper rows present: []` — mọi `metric_path` chứa "paper" đều đã bị prune.

### (b) `0.2793` và `27.93` không được là hai điểm khác nhau

Đơn vị là thuộc tính của **dòng**, không phải của người gọi:

* `native_measurements.py` đọc unit khai báo ở **root document** (`metric_unit`/`units`/`unit`/
  `recall_unit`). Nếu file không khai, unit chỉ được **suy** khi toàn bộ giá trị trong block nằm
  trong `0..1` (fraction) hoặc toàn bộ `>= 0` (percent), và luôn ghi lại là
  `inferred_from_magnitude` — không bao giờ ghi là `declared`.
* `format.ts::fmtRecall(value, unit, digits)` là hàm duy nhất in recall trong bảng native:
  `fraction` → `*100`, `percent` → giữ nguyên, `null` → in giá trị thô kèm `(đơn vị không ghi)`.
* Regression `test_recall_names_the_scale_the_artifact_states` khẳng định
  `fmtRecall(0.2793,"fraction")` **bằng** `fmtRecall(27.93,"percent")` = `27.93%`, và rằng một dòng
  không có unit thì **không** in `%`.

Trên UI: `?area=evaluation&view=native` là bảng duy nhất in recall native, và cột "Đơn vị" in cả
`unit` lẫn `unit_source` để một dòng `inferred_from_magnitude` đọc ra là suy luận, không phải khai báo.

### (c) PAIR exploratory giữ mapping status

`pair_b{1,2}_H_{normal,rev}.json` khai `"mapping_status": "EXPLORATORY_PUBLIC_MAPPING"`. Dòng trước
đây bỏ field này nên một số đo với class mapping chưa xác minh đọc như kết quả tái lập. Nay field
được đọc (`_first_text(payload, "mapping_status", "provenance_label")`) và render thành một badge
`warn` trong bảng native, cùng `data-testid="native-mapping-<id>"` ở Overview.

Live: `mapping rows` = 4 dòng `pair_*` với `EXPLORATORY_PUBLIC_MAPPING`; 45 dòng còn lại `None`
(một giá trị `None` được in là "không ghi", không phải "đã xác minh").

### (d) Danh sách mọi dòng thiếu đơn vị khai báo

Đếm mà không liệt kê thì không kiểm toán được, nên response có thêm mảng `undeclared_unit_rows`.
Live (sau restart):

```
unit_declared      = 4     # khai ở root document
unit_sibling_key   = 2     # khai bằng key có hậu tố trong chính block (dcnet)
unit_inferred      = 39    # còn lại: suy theo độ lớn — KHÔNG có khai báo nào tồn tại
unit_undeclared    = 0     # không suy được độ lớn
unit_not_declared  = 39    # KHÔNG có khai báo: 39/43 dòng đo
unbacked_counted   = 0
```

Đo lại trên payload sống (`/api/native-measurements`, HTTP 200). **Bốn** dòng/blocks nay mang khai báo
thật (`unit_declared = 4`, trong đó 2 là `unit_sibling_key`); **39** dòng vẫn dựa trên suy luận độ
lớn, và từng dòng đã được audit: **0** khai báo bị bỏ sót
(các key `*_pct` có mặt như `dead_parameters_pct`, `relative_error_pct`,
`checkpoint_training_noise_pct` mô tả **đại lượng khác**, không phải thang macro).
Bằng chứng: `raw/verify/31-live-unit-audit.txt` (payload sống, liệt kê từng dòng theo `unit_source`).
**Không** dùng `web_evidence/w01_native_measurements_after.json` cho các con số này — bản đó chụp
**trước** khi có đường khai báo, nên nó ghi `unit_declared: 0`, `unit_inferred: 43`.

Nói cách khác: trong **43** dòng có macro, **39** dòng vẫn đang được đọc trên thang suy ra theo độ
lớn và **4** dòng mang khai báo thật — 2 dòng khai qua `categories[*].provenance.units` của
`clip4cir_fullft.json`, 2 dòng khai bằng key có hậu tố `macro_r10_percent`/`macro_r50_percent` của
`dcnet.json` — **0** khai báo bị bỏ sót. Lưu ý `unit_declared = 4` **đã bao gồm** hai dòng
`unit_sibling_key`, nên cộng `4 + 2` là đếm trùng (45 > 43); tổng số dòng có khai báo là **4/43**,
không phải 6. 39/43 vẫn là con số lớn, và các key `*_pct` có mặt
(`dead_parameters_pct`, `relative_error_pct`, `checkpoint_training_noise_pct`) mô tả **đại lượng
khác** nên không được tính là khai báo thang macro. UI in con số này ở đầu bảng native kèm danh
sách mở được (`data-testid="native-undeclared-list"`), thay vì để người đọc tự suy.

Regression: `test_every_row_without_a_declared_unit_is_listed_not_only_counted`,
`test_a_fraction_and_a_percent_are_one_measurement_not_two_scores`,
`test_the_pair_exploratory_run_keeps_its_mapping_status`.

## W-02 · Scope, trạng thái và link

### (a) Native-only row mở đúng bảng native

Trước: link "có kết quả ở scope khác" của một dòng native trỏ tới
`?area=evaluation&view=benchmark&protocol=…`. Bảng benchmark chỉ liệt kê **aggregate report**, nên
một macro native mở ra một trang rỗng dưới đúng tên protocol — người đọc thấy "không có gì" ở một
scope thật sự có số.

Sau: `view=native` là một view thật của khu vực Đánh giá (`routes.ts AREAS[].views`), render
`NativeMeasurementTable`, và `ScopeRow.family` quyết định link mở bảng nào:

| family | view mở | bảng liệt kê |
|---|---|---|
| `composition` | `evaluation/composition` + `subset` | dòng CIRCUS |
| `native` | `evaluation/native` | `NativeMeasurementTable` |
| `aggregate` | `evaluation/benchmark` | `BenchmarkTable` |

`family` cũng là một phần **identity** của scope choice, không chỉ của link: một model có thể có
macro native **và** report aggregate dưới cùng một `protocol_id`, và gộp chúng thành một link là
cách cũ. Regression
`test_a_native_scope_and_an_aggregate_scope_are_two_links_not_one` khẳng định hai entry, hai family.
Regression `test_the_native_scope_has_a_view_that_lists_it` khẳng định URL `view=native` được parser
chấp nhận và round-trip.

### (b) Dòng native thật sự render được

`NativeMeasurementTable` không chỉ là đích của link: nó in `metric_path`, file, digest, bytes, R@10,
R@50, `unit` + `unit_source`, `mapping_status`, và mở detail đọc từ record. Cùng một `fmtRecall` với
Overview, nên một số đo không thể in khác nhau ở hai chỗ.

### (c) CIRCUS link đổi cả protocol và subset

`selectRun` đã mang `protocol: entry.protocolId` (có sẵn từ trước, đã kiểm lại); link scope-else
nay mang cả `protocol` và `subset` của scope đích, nên một subset không tồn tại ở protocol kia không
bị giữ lại. `SUBSET_LABEL` chỉ in khi `protocolHasSubsets(protocol)`.

### (d) Đếm model / run / attempt / measurement / blocker là số riêng

Header inventory trước in 5 ô, trong đó "17" có thể là model hoặc run tuỳ người đọc đoán. Nay 9 ô,
mỗi ô có `data-testid` riêng:

| `data-testid` | nghĩa | câu hỏi nó trả lời |
|---|---|---|
| `count-models` | `models.length` | registry có bao nhiêu model |
| `count-composition-rows` | mọi dòng composition, kể cả FAILED | có bao nhiêu dòng ở scope này |
| `count-runs` | dòng composition, trừ FAILED | bao nhiêu lượt dùng được |
| `count-attempts` | entry + native rows | có bao nhiêu lần thử được công bố |
| `count-measurements` | native row có macro, không roll-up | đo được bao nhiêu |
| `count-blockers` | `counts.signposted_blockers` | bao nhiêu file chỉ ghi blocker |
| `count-active-measured` | model có số ở scope này | bao nhiêu model có số |
| `count-pending` | model chưa có số | cần chạy bao nhiêu |
| `count-deferred` | `DEFERRED_MODELS` | bao nhiêu tạm gác |

Một `count-attempts` ≠ `count-runs` là **đúng**: một model chạy lại hai lần là hai attempt, một run
id khác, cùng một kết luận.

## W-01 (bổ sung bắt buộc) · Roll-up không được cấp số đo mà không có artifact chống lưng

### Lỗi, đo trên service thật TRƯỚC khi sửa

`/api/native-measurements` phục vụ **bốn** dòng cho hai cấu hình, trong đó hai dòng là **cùng một con
số dưới hai tên**:

| model_id phục vụ | R@10 | nguồn |
|---|---|---|
| `clip4cir_fullft` | 43.81821850935618 | `clip4cir_parsed.json` (roll-up) |
| `clip4cir_rn50x4_fullft` | 43.8182 | `selection_decisions.json` (roll-up) |
| `clip4cir_noft` | 35.14068226019541 | `clip4cir_parsed.json` (roll-up) |
| `combiner_rn50x4_noft` | 35.1407 | `selection_decisions.json` (roll-up) |

Số của roll-up **không** khớp số mà artifact đo được:

| nguồn | FullFT R@10 | FullFT R@50 | dress r10 | toptee r50 |
|---|---|---|---|---|
| replay `raw/native/clip4cir_fullft.json` | **43.8016931215922** | **66.98358456293742** | 39.06792402267456 | 71.29015922546387 |
| roll-up `clip4cir_parsed.json` | 43.81821850935618 | 66.95005893707275 | 39.11750018596649 | 71.23916149139404 |

`shirt` khớp tới từng chữ số (44.50441598892212 / 65.4563307762146); `dress` và `toptee_r50` thì
không. Đó là **hai lượt chạy khác nhau**, và không file nào dưới `checkpoint_selection/` ứng với ô
dress/toptee của roll-up. Combiner tương tự: roll-up 35.1407, report của chính lượt đó 35.1078.

Bằng chứng: `web_evidence/w01_native_measurements_BEFORE_backing.json`.

### Quy tắc đã cưỡng chế trong payload

Một giá trị được phục vụ **phải** là giá trị một artifact nêu. Cài đặt:

1. **Backing thay giá trị.** `_backing_reports` chỉ đọc hai nguồn cấu trúc — `reports/*_aggregate.json`
   và `docs/reviews/*/raw/native/*.json` có `replay_type` + `aggregate`. Một roll-up dưới
   `checkpoint_selection/` **không thể** tự chống lưng cho chính nó. Khi có backing, giá trị được
   thay bằng giá trị của artifact, số cũ giữ nguyên ở `raw_metrics`, và `backed_by` /
   `backed_by_sha256` / `backed_by_run_id` / `backed_by_kind` nêu đúng file.
2. **Alias gộp thành một cấu hình.** `ALIASES` ánh xạ `clip4cir_fullft → clip4cir_rn50x4_fullft`,
   `clip4cir_noft → clip4cir_rn50x4_noft`, `clvc → clvc_net`, `sprc_vitl → sprc`. Tên alias không
   còn xuất hiện như một model riêng.
3. **Không chống lưng được thì nói ra.** Dòng đó vẫn được **liệt kê** (người đọc phải thấy cây
   artifact đang khai gì) nhưng có `artifact_backed: false`, `source_kind: "rollup:<file>"`, và
   **không** vào `counts.models_with_metrics`.
4. **Trùng lặp bị đánh dấu, không bị xoá.** `counted: false` + `duplicate_of: <file>`. Xoá một dòng
   là không kiểm toán được nữa; `duplicate_of` giữ nguyên khả năng đối chiếu.
5. **Combiner dùng số của lượt thật.** 35.1078 từ `reports/…_020a70b0_aggregate.json`, không phải
   35.1407 của roll-up.

### Sau khi sửa — kiểm qua API live (sau restart)

| model_id | R@10 phục vụ | artifact chống lưng | counted |
|---|---|---|---|
| `clip4cir_rn50x4_fullft` | **43.8017** (r50 66.9836) | `…/raw/native/clip4cir_fullft.json` | true |
| `clip4cir_rn50x4_fullft` (bản ghi roll-up) | 43.8017, `raw_metrics.r10` = 43.8182 | như trên | **false** (trùng) |
| `clip4cir_rn50x4_noft` | 35.1407 | không có → `rollup:clip4cir_parsed.json` | liệt kê, **không** vào model count |
| `combiner_rn50x4_noft` | **35.1078** (r50 58.4756) | `reports/…_020a70b0_aggregate.json` | true |

Đếm sau khi sửa: `with_metrics` 21 dòng artifact-backed **phân biệt**, `duplicate_rows` 10,
`unbacked_rows` 18, `models_with_metrics` 17 (trước: 15, và 13 roll-up-only).

Bằng chứng: `web_evidence/w01_native_measurements_after.json`.

### `categories[].protocol` còn sai ở artifact CLIP4Cir đã migrate

**Nhận định gốc là ĐÚNG, và tôi đã từng ghi sai nó thành "bị bác bỏ"** — vì tôi quét sai cây. Ghi lại
đúng, kèm phạm vi quét để lỗi này không tái diễn:

**Phạm vi quét — phải đọc kỹ.** `workbench/artifacts/**/*.json` **không** chứa artifact đã migrate, nên
quét cây đó thấy **0** artifact lệch cấp category. Artifact bị ảnh hưởng nằm ở
`workbench/docs/reviews/**/raw/**`, **ngoài** phạm vi quét đó. Kết luận "không tái lập được" của tôi là
**vắng bằng chứng do tìm sai chỗ**, không phải bằng chứng vắng mặt.

Quét đúng phạm vi (`workbench/docs/reviews/**/raw/**/*.json`, 291 file; 14 file khai protocol ở **cả hai**
cấp): **2 file lệch**, cả hai là cùng một model:

| file | document | `categories[].protocol` |
|---|---|---|
| `20261110-e2e-repair/raw/clip4cir_fullft.migrated.json` | `fashioniq_original_split` | `fashioniq_full_gallery_ref_excluded` (cả 3 category) |
| `20261110-e2e-repair/raw/guards/clip4cir_fullft.migrated.r1.json` | `fashioniq_original_split` | `fashioniq_full_gallery_ref_excluded` (cả 3 category) |

Vậy: migration sửa **cấp document**, để nguyên **cấp category** — đúng như nhận định ban đầu.

**Không sửa hai file đó.** `20261110-e2e-repair` là bản ghi bất biến của chiến dịch trước; viết lại nó là
sửa bằng chứng đã công bố. Nhãn đã sửa chỉ tồn tại ở trường cấp document; phần lệch cấp category được
ghi ở đây như **defect đã biết trong artifact lịch sử**.

**Có phải chỉ là chuyện hình thức không?** Không có consumer nào đọc `categories[].protocol`: bộ đọc
(`backend/native_measurements.py:282,289`) lấy `protocol_id`/`protocol` ở **cấp document**, và
`grep` trên `backend/`, `composition/`, `scripts/`, `replay/` không tìm thấy chỗ nào đọc protocol theo
từng category. Nên số **phục vụ** không bị ảnh hưởng. Nhưng một người đọc artifact muốn biết "category
này chạy protocol nào" sẽ gặp giá trị sai, nên nó vẫn là defect dữ liệu, chỉ không phải defect hiển thị.

### Đơn vị: declaration thắng magnitude inference

`unit_declared` trước là 0/43 vì bộ đọc chỉ biết `unit` cấp document. `dcnet.json` đã khai đơn vị
**ngay trong block** bằng key có hậu tố: `aggregate.macro_r10_percent = 27.929730914548916`.
Thêm `UNIT_SUFFIX_DECLARATIONS` (`_percent`/`_pct`/`_fraction`) và đọc nó trong `_blocks`, nên thứ
tự quyết định bây giờ là:

1. `unit` khai ở root document → `declared`;
2. key có hậu tố trong chính block → `declared`;
3. suy theo độ lớn (mọi giá trị trong `0..1` → fraction) → `inferred_from_magnitude`, **chỉ** là
   fallback.

Điều này chặn đúng ca nguy hiểm: một block percent mà cả ba category đều ≤ 1 % sẽ được đọc là
fraction và render nhỏ đi 100 lần. Regression `test_a_suffixed_sibling_key_declares_the_unit`.

## W-04 · Regression giao diện

Bốn kiểm tra hành vi trên browser thật (Chrome), ở `1920x1080`, chạy bằng
`scripts/w04_web_regression.py`, kết quả `web_evidence/w04/w04_results.json`.

| kiểm | cách ép | kết quả | bằng chứng |
|---|---|---|---|
| A về sau B | `page.route` giữ response `/queries` của run A 6 s, rồi điều hướng sang run B | **PASS** — DOM sau 8 s nói `encoder`, không còn `limn` | `w04_late_answer_a_after_b.png` |
| deep link qid ngoài dữ liệu | `query=7:99999` trên run B subset `sf` | **PASS** — in đúng tên qid, nói "Không tự thay bằng query khác", **không** dựng `query-inspector` | `w04_deeplink_out_of_range.png` |
| reload + Back/Forward | reload trang trên, rồi `go_back()` / `go_forward()` | **PASS** — reload giữ `7:99999` + subset `Shortcut-Free`; Forward trả lại đúng nội dung | `w04_reload_and_history.png` |
| lỗi API | `route.fulfill` trả 503 cho `/api/native-measurements` | **PASS** — vùng dữ liệu in `Yêu cầu thất bại` + body lỗi + nút `Thử lại`, **không** phải khoảng trắng | `w04_induced_api_error.png` |

### Một lỗi thật do chính phép thử lỗi API phát hiện

Trước khi sửa, subtitle của card native in `0 dòng có số cho …` **trong khi** vùng dữ liệu báo lỗi.
Đó là con số 0 cho một phạm vi chưa hề đếm được — cùng lớp lỗi với "0 lượt" trên một scope đang tải.
Đã sửa: subtitle nay phân biệt ba trạng thái

```
ready   → "N dòng có số cho <protocol> · không vào leaderboard đã xác minh"
error   → "không đọc được catalog native cho <protocol> — đây là lỗi yêu cầu, không phải 0 dòng"
khác    → "đang đọc catalog native cho <protocol>…"
```

Chi tiết: `w04_induced_api_error.json` (`data_region_excerpt`).

### Về bộ lọc worst

`sortRows(..., "worst")` đặt `(target_rank is None, -(target_rank or 0))` nên query **unscored luôn
xuống cuối**, không bị trộn vào danh sách hạng xấu. Đã có test từ trước
(`test_frontend_analysis.py`, "the unscored query was not last in worst-first order") và nó vẫn xanh
trong lượt này. Không sửa gì ở đây.

## W-05 · Publish

### Pin

| mục | giá trị |
|---|---|
| HEAD khi đo | `92ed66de4f359ef479da18bf8939723b6f9313e5` + thay đổi chưa commit — **superseded** |
| HEAD hiện tại | **`46d1e1a4fcf2a494e32905731f0fe4d5de3e89a0`**; revision đã được sweep là **`be14393c97b100390a9a45c6b0a46857b24a78b9`** (`TESTED_SOURCE_SHA`). Bảng bằng chứng dưới đây được thu trên cây tương ứng `be14393` |
| bundle phục vụ | **`index-DWpK0yeS.js`**, sha256 **`7e34724ebf55092e36524b98980cca55e71f3ddbaa62489c65577f92e18a6a16`** |
| dist | cùng hash (`cmp` → IDENTICAL giữa byte phục vụ và `dist/assets/index-DWpK0yeS.js`) |
| superseded | `index-CEgaMiLf.js` sha256 `0b6b15a8…6eb87` (build trước `relation_state`/`relation_parity`; không còn trên đĩa) |
| css | `index-DecQnta9.css` |
| root | `WORKBENCH_CIRCUS_RELEASE_ROOT=/home/heheboiz/data/cir/workbench/artifacts/circus_release` |
| proxy | vite `preview` proxy `/api` → `http://127.0.0.1:8000` (`vite.config.mts`, `WORKBENCH_BACKEND_URL`) |
| PID | **4171850** (`serve_workbench.py`) → **4171851** (uvicorn); superseded `4082631/4082632`, `4062950/4062957`, `4040573/4040574`, `3390138/3390139`, `3316577/3316578`, `3228456`, `3212146` |
| quy tắc pin | **digest bundle là pin bền; PID chỉ đúng tại thời điểm đo.** PID đã cũ đi **ít nhất bốn lần** trong một đợt (mỗi lần một lần chết do SIGTERM; số chính xác không chốt được vì log bị truncate ở mỗi restart — xem mục "Service được phục vụ"). Không bao giờ dùng hàng PID làm pin build |
| readiness | `/api/benchmark-runs` 200 sau 1 s |

### Bằng chứng theo nhóm scope có dữ liệu

Mỗi ảnh nằm cạnh chính response API mà trang đã nhận, trong cùng file `.json`:

| scope | ảnh | API kèm theo |
|---|---|---|
| CIRCUS release, subset `sf` (Tổng quan) | `w_shots/w02_overview_circus.png` | `/api/composition/runs` |
| CIRCUS release, `full`, inspector 7:3038 | `w_shots/w03_inspector_7-3038_limn_full.png` | `/api/composition/runs/<limn>/queries?…` |
| CIRCUS release, `validated` (composition + G1/G2) | `w_shots/w05_pe_histograms.png` | `/api/composition/capabilities?subset=validated`, `/api/runs/parity-gate` |
| FashionIQ Original (báo cáo tổng hợp) | `w_shots/w05_benchmark_reports.png` | `/api/benchmark-runs`, `/api/evidence/summary` |
| FashionIQ Full-Gallery-Ref-Excluded (native) | `w_shots/w01_native_table.png`, `w_shots/w02_native_scope_populated.png` | `/api/native-measurements`, `/api/benchmark-runs` |
| model inventory | `w_shots/w02_models_inventory.png` | `/api/composition/runs?subset=sf` |
| diagnostic LIMN, cả hai scope | `w_shots/w03_limn_diagnostic_toptee.png` | `/api/composition/diagnostics/limn-dual-scope` (+ `…/candidates?depth=20`) |
| Probe K=10 / K=100 (chụp vùng) | `w_shots/w03_limn_probe_toptee_mm_K10.png`, `w_shots/w03_limn_probe_target_tile.png` | — |

`w_shots/manifest.json` liệt kê **13** ảnh với `url`, `png_sha256` và `dom_text_sha256` (không có
trường `viewport`), nên một ảnh trong manifest không thể bị đọc là bằng chứng cho một payload khác.
Bốn ảnh trong `w_shots/` **không** nằm trong manifest — `w03_inspector_ordinary_7-3038_K50.png`,
`w03_limn_probe_target_tile.png`, `w03_limn_probe_toptee_mm.png`, `w03_limn_probe_toptee_mm_K10.png` —
nên với bốn ảnh đó không có `png_sha256`/`dom_text_sha256` để đối chiếu, và chúng chỉ dùng làm hình
minh hoạ, không dùng làm bằng chứng duy nhất cho một khẳng định.

### `networkidle` không bao giờ xảy ra trên stack này

`/api/models` hash mọi checkpoint (~154 s), nên mạng **không bao giờ** im. Chờ `networkidle` là
timeout 90 s rồi bỏ cuộc. Bộ evidence chờ `domcontentloaded` + một selector của region. Đây là lý do
kỹ thuật để không dùng `/api/models` làm readiness probe.

### Historical trước khi verified

Bảng benchmark hiện `refused` / `uncertified` / `N lần thử` từ chính `verification.state` và
`duplicate_status` của row; chỉ `certified && is_research_evidence` mới được badge `ok`. Không có
đường nào để một dòng chưa xác minh đọc thành verified.

## Kiểm thử đã chạy

```
workbench/envs/ctl312h/bin/python -m pytest \
  test_native_measurements_census test_frontend_format test_web_scope_and_labels \
  test_limn_dual_scope_diagnostic test_limn_scope_routing test_limn_diagnostic_record \
  test_frontend_routes test_frontend_requests test_frontend_catalog -q
→ 145 passed
```

Log: `raw/w01_w03_targeted_tests.txt`. Không chạy toàn bộ suite (theo yêu cầu).

## Việc còn lại (nói rõ, không giấu)

* `clip4cir_rn50x4_noft` vẫn chỉ có roll-up: được liệt kê, `artifact_backed: false`, không đếm.
  Cần một lượt replay cho NoFT thì mới có số có artifact chống lưng.
* `categories[].protocol` trong artifact CLIP4Cir đã migrate **vẫn sai** — nhận định gốc đúng.
  Hai file: `20261110-e2e-repair/raw/clip4cir_fullft.migrated.json` và
  `…/raw/guards/clip4cir_fullft.migrated.r1.json`, document `fashioniq_original_split` nhưng cả 3
  category vẫn `fashioniq_full_gallery_ref_excluded`. Không sửa (bản ghi cũ là bất biến). Không consumer
  nào đọc trường này, nên số phục vụ không đổi — xem mục W-01 bổ sung để biết **phạm vi quét** và vì sao
  một lần quét `workbench/artifacts/**` không thể thấy nó.
* `unit_declared` **không còn là 0/43** (con số đó đã cũ): payload sống báo `unit_declared: 4`,
  `unit_sibling_key: 2`, `unit_inferred: 39`, `unit_not_declared: 39`. **Bốn** dòng nay mang khai báo
  thật — `unit_declared: 4` **đã gồm** hai dòng `unit_sibling_key` (`clip4cir_fullft.json` khai
  `categories[*].provenance.units`; `dcnet.json` khai bằng hậu tố `macro_r10_percent`) — nên tổng là
  4/43 chứ không phải 6. **39** dòng vẫn dựa trên suy luận độ lớn và đã được audit từng dòng: **0** khai
  báo bị bỏ sót.
* `.scratch/` đã xoá; nội dung có giá trị đã chuyển vào `raw/` và `scripts/`.
