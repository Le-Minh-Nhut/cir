# XÁC MINH VÀ SWEEP PHÁT HÀNH — 20261111-master-audit

Vai trò: worker xác minh / cổng phát hành. Không tin báo cáo của sáu worker trước; mọi giá trị dưới
đây đọc trực tiếp từ cây làm việc và từ API đang phục vụ.

- Repo: `/home/heheboiz/data/cir`, nhánh `research/cir-failure-workbench`
- HEAD lúc đo: `92ed66de4f359ef479da18bf8939723b6f9313e5` + work chưa commit — **superseded**; mốc bền nay là `TESTED_SOURCE_SHA` `0691f3e` (§ sweep chốt, `raw/verify/73-sweep-close.md`; `7babfd9` của lượt trước cũng đã bị thay thế)
- Python: `workbench/envs/ctl312h/bin/python` (CPython 3.12.11)
- Raw: `workbench/docs/reviews/20261111-master-audit/raw/verify/`

## Điều đã sửa trong đợt này (ba lỗi thật, không phải ghi chú)

Phase 1 lần đầu phát hiện hai lỗi, lượt xác minh thứ hai phát hiện thêm một lỗi cùng họ. Cả ba đều
nằm trong `workbench/backend/native_measurements.py` và đã sửa:

### 1. Dòng không có artifact vẫn được đếm — SỬA

`_backed_by_artifact` đặt `artifact_backed = False` nhưng **không** đặt `counted = False`, nên 18
dòng roll-up không có artifact vẫn mang `counted: true`. Sửa: nhánh không có backing nay đặt
`counted = False` kèm chú thích; `counts.unbacked_counted` mới được thêm để con số này tự kiểm.

- Trước: `unbacked AND counted = 18`
- Sau: **`unbacked AND counted = 0`**, `counts.unbacked_counted = 0`

`rollup_rows` đổi nghĩa cho đúng: nay là **mọi** dòng không có artifact (18), không phải "dòng
roll-up đang được đếm".

### 2. DCNet phục vụ sai thang 100 lần — SỬA (blocker W-01)

`dcnet.json` ghi `aggregate.macro_r10 = 0.2793` **và** `aggregate.macro_r10_percent = 27.9297`.
`_report_metrics` đọc khoá trần; `_backed_by_artifact` thay `row["metrics"]` bằng số của report nhưng
**không** tính lại `unit`/`unit_source`, nên `unit` còn là suy luận từ file nguồn khác. Kết quả phục
vụ: `0.2792973` với `unit: "percent"` → `fmtRecall` in **0.28 %** cạnh ~50 % của mọi model khác.

Đã sửa ba điểm:

1. `_report_metrics` đọc theo thứ tự ưu tiên hậu tố `_percent` → `_pct` → trần → `_fraction`, và trả về
   bộ ba `(values, unit, unit_source)`; hậu tố thắng ⇒ `unit_source = "sibling_key"`.
2. `_backed_by_artifact` tính lại `unit`/`unit_source` **từ chính report đã thay số**, giữ bản cũ ở
   `raw_unit`/`raw_unit_source` cạnh `raw_metrics`.
3. Khai báo ở cấp tài liệu cũng được thu hoạch: `_provenance_unit` đọc
   `categories[*].provenance.units` khi **toàn bộ** category đồng ý (`clip4cir_fullft.json` ghi
   `"percent, as printed by upstream (recall_at10 * 100)"`) ⇒ `unit_source = "declared"`.

`DECLARED_UNIT_SOURCES = ("declared", "sibling_key")` — cả hai là khai báo; `unit_declared` và
`unit_not_declared` đều tính theo tập này.

Dòng DCNet được phục vụ, nguyên văn (raw `09-api-native-measurements-postfix.txt`):

```json
{
 "model_id": "dcnet",
 "protocol_id": "fashioniq_full_gallery_ref_excluded",
 "metrics": {"r10": 27.929730914548916, "r50": 53.76075764301528},
 "unit": "percent",
 "unit_source": "sibling_key",
 "counted": true, "artifact_backed": true, "duplicate_of": null,
 "backed_by": "…/20261109-native15-composition-limn/raw/native/dcnet.json",
 "raw_metrics": {"r10": 27.929730914548923, "r50": 53.72723331879208, "mean": 40.8284821166705},
 "raw_unit": "percent", "raw_unit_source": "inferred_from_magnitude"
}
```

`metrics.r10 = 27.929730914548916` khớp `macro_r10_percent` của artifact (sai khác 7e-15), `unit` là
`percent`, `unit_source` là `sibling_key`. Không còn dòng nào phục vụ `0.279…` với unit `percent`.

### 3. Trùng chỉ khác độ chính xác float không gộp — SỬA

`_mark_duplicates` dùng `round(value, 6)` làm danh tính. `airknow` ghi 52.5466 trong roll-up và
52.54661498980954 trong file run — cùng một thí nghiệm, hai độ chính xác — nên **không** gộp; còn
`round(...,6)` thì không gộp được cặp nào thực sự khác nhau. Đổi sang `round(value, 4)`: sai khác
dưới 1e-4 là một danh tính, khác hơn thì là hai run khác nhau.

Kết quả sau sửa (raw `08-duplicates-and-units.txt`):

| model | rows | counted | giá trị counted phân biệt | ghi chú |
|---|---|---|---|---|
| `airknow` | 3 | **2** | 2 | 52.5466 roll-up gộp vào 52.54661… (n05); 50.6417 (n08) giữ riêng |
| `clvc_net` | 3 | **1** | 1 | hai roll-up + một run file |
| `tgcir` | 3 | **1** | 1 | hai roll-up + một run file |
| `sprc` | 2 | **1** | 1 | roll-up + run file |
| `pair` | 4 | **4** | 4 | bốn cấu hình THẬT SỰ khác nhau, giữ nguyên, không gộp |

Dòng khác giá trị thật vẫn hiển thị đầy đủ và tách biệt (`pair` B1/B2 × H_normal/H_rev), không bị gộp.

### Test đi kèm

`workbench/tests/test_native_measurements_census.py` được cập nhật theo hợp đồng mới và thêm hai test
hồi quy: `test_dcnet_is_served_on_the_scale_its_sibling_key_declares` (assert đúng dòng DCNet:
`metrics.r10 ≈ 27.9297`, `unit == "percent"`, `unit_source == "sibling_key"`, và không bao giờ để
`< 1.0` đi cùng unit `percent`) và
`test_no_counted_row_carries_an_inference_where_a_declaration_exists` (assert `unbacked_counted == 0`).
`test_a_suffixed_sibling_key_declares_the_unit` nay chờ `sibling_key` thay vì `declared` — cùng một sự
thật, nhãn chính xác hơn.

Frontend: `data.ts` khai báo thêm `sibling_key`; `EvaluationPage.tsx` cho cả hai dạng khai báo badge
`ok`, và phân biệt lý do một dòng không được đếm ("trùng với …" vs "không có artifact chống lưng").

## Phase 1 — bốn khẳng định (chạy lại trên stack đã khởi động lại)

### 1. Frontend build — **PASS**

`npm run build --prefix workbench/frontend` → exit 0, `✓ 39 modules transformed`, `✓ built in 190ms`.
Không có import chết `fmtPercentValue`; không phải sửa tệp nào vì lý do build. Raw:
`03-frontend-build.txt` (bản trước khi sửa frontend) và lần build sau sửa (bundle `index-DWpK0yeS.js` — bundle
của **lượt đó**, nay **superseded**; bundle đang phục vụ là `index-CBR6lHgq.js` sha256 `acfb5fbb…ac21a`, xem
`REMAINING_WORK.md`).

### 2. `workbench/backend/native_measurements.py` — **PASS**

Đã đọc toàn bộ tệp. `ast.parse` OK, import OK.

- (a) parse: OK.
- (b) lớp đơn vị: `UNIT_KEYS`, `UNIT_SUFFIX_DECLARATIONS`, `_suffix_declared_unit`, cộng thêm
  `UNIT_SUFFIX_ORDER`, `DECLARED_UNIT_SOURCES`, `_provenance_unit`.
- (c) hậu thuẫn artifact + đánh dấu không phá huỷ: `_backing_reports` chỉ nhận `reports/*_aggregate.json`
  và `docs/reviews/*/raw/native/*.json` có `replay_type` + `aggregate`; `_backed_by_artifact` thay số và
  tính lại đơn vị; `_mark_duplicates` đặt `counted=false`/`duplicate_of`, **không xoá dòng**.
  `ALIASES` gộp `clip4cir_fullft → clip4cir_rn50x4_fullft`.
- (d) cắt `paper*`: `EXCLUDED_TOKENS = ("paper", "reported", "reference", "context_only")` áp trên cả
  đường dẫn khoá. Payload phục vụ có **0** lần xuất hiện chuỗi `paper`.

### 3. API chỉ phục vụ số có artifact hậu thuẫn — **PASS**

`GET /api/native-measurements` → HTTP 200, 73381 byte, 49 dòng.
`counts`: `files_scanned 71, files 32, blocks_with_metrics 43, with_metrics 21, rollup_rows 18,
duplicate_rows 22, unbacked_rows 18, unbacked_counted 0, unit_declared 4, unit_sibling_key 2,
unit_inferred 39, unit_undeclared 0, unit_not_declared 39, unreadable 2, signposted_blockers 4,
models_with_metrics 17, models_rollup_only 8, verified 0`.

| Khẳng định | Quan sát | Kết luận |
|---|---|---|
| `clip4cir_rn50x4_fullft` macro R@10 | **43.8016931215922** (r50 66.98358456293742), `artifact_backed: true`, `unit: percent`, `unit_source: declared`, `backed_by: …/native/clip4cir_fullft.json` | PASS |
| 43.81821850935618 không phục vụ như số của model | không dòng nào có `metrics.r10 == 43.81821850935618`; chỉ còn ở `raw_metrics` | PASS |
| `combiner_rn50x4_noft` | **35.1078 / 58.4756 / 46.7917**, `artifact_backed: true`, `backed_by: reports/combiner_…_020a70b0_aggregate.json` | PASS |
| Cặp trùng `clip4cir_fullft`/`clip4cir_rn50x4_fullft` | dòng `model_id == clip4cir_fullft`: **0**; dòng thứ hai cùng số mang `counted: false`, `duplicate_of: …/clip4cir_parsed.json` | PASS |
| Dòng `artifact_backed: false` không đóng góp số model | `unbacked AND counted = **0**`; `counted ∧ backed ∧ có metrics` = 21 dòng / **17 model** = `models_with_metrics` | PASS |

Còn **8 model** chỉ có số từ roll-up (`B1/H_normal`, `B1/H_rev`, `B2/H_normal`, `B2/H_rev`,
`airknow_n05`, `airknow_n08`, `airknow_variant_80`, `clip4cir_rn50x4_noft`): vẫn được **hiển thị** với
`artifact_backed: false`, `counted: false`, và **không** vào `models_with_metrics`. Đây là hành vi đã
thiết kế, không phải số bịa.

Còn **39 dòng** dựa trên suy luận độ lớn (`unit_inferred`), trong đó **19 dòng đang được đếm**. Đã quét
từng artifact hậu thuẫn của chúng: **không file nào khai báo đơn vị ở bất kỳ dạng nào** đã biết (không
`unit`/`metric_unit`/`units`, không khoá hậu tố `*_percent`/`*_pct`/`*_fraction`, không
`provenance.units` đồng thuận). Danh sách: `airknow`, `clvc_net`, `combiner_rn50x4_noft`, `conesep`,
`csmcir`, `encoder`, `habit`, `hint`, `intent`, `limn`, `pair`, `ptha_mtst`, `sprc`, `tgcir`, `tme`.
Vì không có khai báo nào tồn tại, không có suy luận nào bị một khai báo vượt mặt — nhưng chúng vẫn là
suy luận, và `counts.unit_inferred` cùng `undeclared_unit_rows` liệt kê đủ để kiểm.

Số `csmcir`/`tme` từng có khoá `*_pct` trong nhánh khác của tài liệu (ví dụ
`benchmark_gate.comparisons.*.relative_error_pct`, `provenance.checkpoint_training_noise_pct`); đó là
đơn vị của một đại lượng khác (sai số tương đối / nhiễu), không phải của `macro_r10`, nên không được
dùng làm khai báo cho macro — dùng nó sẽ là gán sai thang.

### 4. W-03 trên inspector THƯỜNG — **PASS**

`GET /api/composition/diagnostics/limn-dual-scope/candidates?qid=7:3038&scope=toptee&condition=multimodal&depth=N`.
Raw: `10-limn-w03-postfix.txt`. Target DID `7:3416`, rank lưu **43**.

| depth | depth_served | depth_stored | target_rank | `target.in_window` | `target.in_stored_window` | rank 43 trong `candidates` |
|---|---|---|---|---|---|---|
| 10 | 10 | 100 | 43 | **false** | true | không |
| 20 | 20 | 100 | 43 | **false** | true | không |
| 43 | 43 | 100 | 43 | **true** | true | có |
| 100 | 100 | 100 | 43 | **true** | true | có |

Cả `depth_served` và `depth_stored` có mặt ở mọi phản hồi; `depth_served = min(depth, depth_stored)`
đúng cả bốn. `in_window` tính trên `served_ranks` của chính cửa sổ phục vụ
(`limn_diagnostic.py:~612`), không dùng `combination["target"]["in_window"]` đã quyết trên 100 dòng lưu;
`in_stored_window` giữ nguyên nghĩa cũ.

## Ba tồn đọng của lượt xác minh thứ hai — đã đóng

### 1. `35.14068226019541` xuất hiện ở đâu — ĐÓNG, không phải lỗi

Truy vết toàn payload (raw `15-final-four-assertions.txt`, mục A3): giá trị này chỉ nằm ở **một** dòng,
`model_id = clip4cir_rn50x4_noft`, `counted: false`, `artifact_backed: false`,
`source = clip4cir_parsed.json`. Nó **không** nằm trên `combiner_rn50x4_noft` (dòng đó phục vụ 35.1078,
và giữ 35.1407 ở `raw_metrics` — đúng nghĩa "số roll-up đã bị thay", có nhãn).

Hai tên này là **hai run khác nhau của cùng một checkpoint**, không phải alias. Bằng chứng: ô
per-category khác nhau — dress R@10 30.689141 (roll-up) vs 30.639565 (report), shirt 36.997056 vs
36.947989. Cùng một lần chạy thì các ô này phải trùng tới từng chữ số (như trường hợp `clip4cir_fullft`,
nơi shirt trùng và dress/toptee khác). Vì vậy chúng **không** bị gộp, và đó là hành vi đúng: gộp hai run
khác nhau mới là bịa một danh tính.

Trạng thái chấp nhận được: dòng `clip4cir_rn50x4_noft` là roll-up **không được đếm**, không có artifact
chống lưng, và tên model khác với `combiner_rn50x4_noft`. Không thể hiển thị như một con số danh giá.

### 2. `combiner_rn50x4_noft` `unit_source: inferred_from_magnitude` — ĐÓNG, suy luận là trung thực

Đã quét toàn bộ report chống lưng (`reports/combiner_…_020a70b0_aggregate.json`) tìm mọi khoá chứa
`unit`/`percent`/`pct`/`fraction`/`scale`/`ratio`: **0 khoá**. `observed_metrics` chỉ có `{r10, r50, mean}`;
`category_metrics` chỉ có `{r10, r50}`. Không có `metric_unit`, không sibling `*_percent`, không
`provenance.units`. Vậy `inferred_from_magnitude` là câu trả lời trung thực — report **không** khai báo
đơn vị, không phải module bỏ sót khai báo.

### 3. `unit_inferred: 39` / 43 block — ĐÓNG, 0 khai báo bị bỏ sót

Audit từng dòng suy luận, đọc chính artifact chống lưng (raw `14-unit-inference-audit.txt`), kiểm ba dạng
khai báo: khoá đơn vị ở gốc tài liệu; `categories[*].provenance.units` đồng thuận; sibling
`*_percent`/`*_pct`/`*_fraction` của chính khoá macro được phục vụ.

- **Nhóm A — có khai báo mà vẫn bị gắn `inferred`: 0.** (Đây là con số bắt buộc phải bằng 0.)
- **Nhóm B — không có khai báo nào: 39.** Trong đó **19 dòng đang được đếm**, 20 dòng không đếm.
- **Nhóm C — kiểm mâu thuẫn:** hai dòng đã gắn `declared` (`clip4cir_rn50x4_fullft`) và `sibling_key`
  (`dcnet`) đều đọc đúng khai báo của chúng.

Các khoá hậu tố `*_pct` **có** xuất hiện trong một số artifact nhưng không phải khai báo cho macro:
`dead_parameters_pct` / `dead_parameters_under_public_forward_pct` (`pair_compat_verdict.json`),
`checkpoint_training_noise_pct` / `evaluation_noise_pct` (`airknow`, `conesep`, `habit`, `intent`,
`limn`, `hint`), `benchmark_gate.comparisons.*.relative_error_pct` (`csmcir`). Đó là đơn vị của **đại
lượng khác** (tham số chết, nhiễu, sai số tương đối) — dùng chúng làm khai báo cho `macro_r10` sẽ là gán
sai thang, đúng lớp lỗi W-01 chống. Vì thế 0 là câu trả lời đúng.

Test khoá lại hành vi này: `test_no_artifact_declaring_a_macro_scale_is_labelled_inferred` — với mỗi dòng
`inferred_from_magnitude`, nó mở artifact chống lưng và assert rằng **không** tồn tại cả ba dạng khai báo.

## `.gitignore`: log bằng chứng bị nuốt — ĐÃ SỬA

`.gitignore:116` có `*.log` không có ngoại lệ, nên **121 file log bằng chứng** mà các báo cáo
`BLOCKED_MODELS_VERIFICATION.vi.md` trích dẫn bị bỏ qua âm thầm: chúng thậm chí **không xuất hiện** trong
`git status`, nên không ai thấy. Đã thêm ngoại lệ có phạm vi ngay sau luật `*.log` (luật khớp sau cùng
thắng):

```
*.log

# Review evidence logs are claims' raw backing, not runtime noise: the repo-wide *.log rule above
# would drop them silently and they do not even appear in `git status`.
!workbench/docs/reviews/**/*.log
```

Kiểm chứng (raw `13-gitignore-evidence-logs.txt`):

- `git check-ignore -q …/airknow_forward_gpu_native.log` → exit **1** (KHÔNG còn bị ignore); trước khi
  sửa, `git check-ignore -v` trả `.gitignore:116:*.log`.
- Số log bằng chứng nay hiện ra trong git, theo từng cây:

| cây | hiện trong git | có trên đĩa | còn bị ignore |
|---|---|---|---|
| `20261111-master-audit/raw` | **25** | 25 | 0 |
| `20261110-e2e-repair` | **24** | 24 | 0 |
| `20261109-native15-composition-limn` | **72** | 72 | 0 |

**Tổng: 121 log nay hiện ra** (trước khi sửa: 0). `git ls-files | grep -c '\.log$'` = **0** — nghĩa là
**chưa từng có** log nào được commit trong toàn repo, nên các chiến dịch trước thật sự đã ship claim
bằng văn xuôi mà không kèm log thô. Đây là phát hiện phụ quan trọng: `20261110-e2e-repair` và
`20261109-native15-composition-limn` cũng bị cùng cái bẫy, và nay đã sửa cho cả ba cây.

Hệ quả lên `git status`: số mục tăng từ 239 lên **397** (chỉ vì log hiện ra, không có tệp nguồn mới).

## Phase 2 — danh tính dịch vụ

Có sửa backend + frontend ⇒ build lại và khởi động lại **đúng một lần**.

- Backend `:8000`: PID **3390139** (`uvicorn workbench.backend.main:app`), con của PID **3390138**
  (`serve_workbench.py --production-frontend`), khởi động `2026-11-10T22:34:23+07:00`.
- Env: `WORKBENCH_CIRCUS_RELEASE_ROOT=/home/heheboiz/data/cir/workbench/artifacts/circus_release`.
- Sẵn sàng (poll `/api/benchmark-runs`, KHÔNG dùng `/api/models`): HTTP 200, 13 mục.
- Frontend `:5173`: bundle `assets/index-DWpK0yeS.js` (377.82 kB), sha256
  **`7e34724ebf55092e36524b98980cca55e71f3ddbaa62489c65577f92e18a6a16`**;
  `dist/assets/index-DWpK0yeS.js` cùng hash, `cmp` → IDENTICAL. **Đây là bản ghi của lượt đó và đã
  SUPERSEDED** — bundle đang phục vụ nay là `assets/index-CBR6lHgq.js`, sha256 `acfb5fbbced77e88b86ee542e5f261c6df71c0e0e8039d753965e1fc873ac21a` (`index-BLH2yome.js` / `702ed2e9…d681` và `index-DyVaNikG.js` / `81d61f65…0981` của các lượt kế tiếp cũng đã superseded) (đọc lại bằng
  `curl -s http://127.0.0.1:5173/ | grep -o 'assets/index-[^"]*\.js'` rồi `sha256sum` trên `dist/`).
- Uvicorn cũ PID 1227919 trên `:8765` vẫn sống — **không** dùng để đo bất cứ gì.
- Raw: `17-final-service-identity.txt` (bản cuối), `11-service-identity-postfix.txt` (bản trước khi sửa
  `.gitignore`).

`git status --porcelain` không có `raw/`, `scripts/`, `.scratch/` ở gốc repo. Mọi scratch nằm dưới
`workbench/docs/reviews/20261111-master-audit/`.

## Sáu hồi quy MỚI do sweep đầu tiên phát hiện — đã sửa

Sweep trên (trước khi sửa) cho `24 failed, 2793 passed, 47 skipped in 3318.19s`; diff với baseline
`20261110-e2e-repair/full_suite_68ad75e.txt`: **6 NEW, 0 DROPPED, 18 COMMON**. Cả 6 NEW đều thuộc chiến
dịch này và đã sửa; đây là phần "chiến dịch chịu trách nhiệm":

### 1-3. `test_throughput_knobs.py` — 3 test (stub không trung thực)

`AttributeError: 'int' object has no attribute 'get'` tại `runner.py:1000`. Nguyên nhân: campaign thêm
`blas_pin_scope`, đọc `pin.get("effective")` trên **record** mà `_blas_thread_limit` trả về — đúng như
docstring của nó ("a record, not a bare int"). Ba test double cũ trả **int trần**, nên không thể chạy
qua scope mới. Sửa: ba double trả record đúng hình dạng (`requested/effective/previous/setter_found/
getter_found/verified/source/reason`), và xử lý cả nhánh nhận dict (restore) — vẫn giữ nguyên ý định
từng test (pin một lần cho cả hai nhánh, restore đúng giá trị `previous`, không pin theo worker).
`27 passed`.

### 4. `test_controller_installation.py::test_doctor_workbench_checks_surface_all_required_imports`

Doctor bị đòi kiểm `open_clip`, `datasets`, `torchvision`, `clip`, `data_utils`. Cả năm chỉ đến từ
`workbench/scripts/warm_replay_cache.py` (script mới của chiến dịch) và **đều là import trong thân hàm**
(`line 38/41/48/51/60`, `col_offset > 0`), chạy trên `--source-root` do người gọi cấp (checkout CLIP4CIR /
AirKnow), không phải trên venv controller mà doctor kiểm. Nếu bắt doctor báo chúng thì doctor
`BLOCKED` vĩnh viễn và che mất các check thật.

Sửa: `_scripts_third_party_imports` chỉ thu thập import **cấp module** (duyệt `tree.body`, đi vào
`if`/`try` vì chúng vẫn chạy lúc import, nhưng không đi vào thân hàm), kèm docstring giải thích
deferred-vs-controller. `6 passed`.

### 5-6. `test_web_scope_and_labels.py` — 2 test (artifact cũ dùng lại)

`KeyError: 'family'` và danh sách thiếu `family` — chính xác là hành vi **trước W-02**. Hai test này
**pass khi chạy một mình** (14 passed) và pass khi chạy cùng các module frontend khác (85 passed); chỉ
fail trong sweep. Dấu vết: `/tmp/cir_frontend_scope/scope.js` mang mtime **sau** khi sweep kết thúc, tức
trong sweep nó đang giữ byte cũ.

Nguyên nhân: `compile_module` ghi vào **đường dẫn cố định** `/tmp/cir_frontend_scope`, nên một artifact
từ lần chạy trước có thể tồn tại và được phục vụ. Sửa: mỗi lần gọi emit vào thư mục con **mới**
(`build-<pid>-<hash>`), để artifact được phục vụ luôn là output của chính lần chạy đó. `118 passed`
trên toàn bộ nhóm module frontend.

### Sweep lại

Sau khi sửa cả 6, một lần sweep phụ được khởi động rồi **dừng giữa chừng** theo yêu cầu operator. Sweep
CHỐT sau đó chạy **một lần** trên cây cuối cùng — kết quả ở phần "SWEEP CHỐT" bên dưới.

## Phase 3 — sweep phát hành

Preflight của lần chốt (raw `25-sweep-final-preflight.txt`, `2026-11-11T00:20:43+07:00`):

- `pgrep -af pytest` → **không có gì** (exit 1); kiểm thêm `gw0|gw1|execnet` (xdist worker sót lại) →
  **không có gì** (exit 1).
- `df -h /` → `/dev/nvme0n1p5 630G, **còn 121 GiB**` (≥45 GiB).
- `/tmp/sweep_basetemp_final` **chưa tồn tại** trước khi chạy.
- `git status --porcelain -uall` → **408 mục** (đã ghi; work chưa commit của chiến dịch là dự kiến,
  **không commit**). Không có `raw/`, `scripts/`, `.scratch/` ở gốc repo.

### SWEEP TRUNG GIAN — cây TRƯỚC `relation_state` (18/2799/47)

Chạy một lần, tuần tự, tách rời, trên đúng cây sẽ phát hành:

```
mkdir -p /tmp/sweep_basetemp_final
setsid nohup workbench/envs/ctl312h/bin/python -m pytest workbench/tests \
  -q -p no:cacheprovider -rfE --tb=no --basetemp=/tmp/sweep_basetemp_final \
  > /tmp/sweep_final.txt 2>&1 < /dev/null &
```

**1. Dòng tổng kết nguyên văn:**

```
18 failed, 2799 passed, 47 skipped, 350 warnings in 3194.17s (0:53:14)
```

**2. Accounting — khớp tuyệt đối:**

| | |
|---|---|
| failed | 18 |
| passed | 2799 |
| skipped | 47 |
| **tổng** | **2864** |
| `--collect-only -q` | **2864 tests collected** |
| **GAP** | **0** |

**3. Sức khoẻ log:** `E` row = **0**; `ERROR` row = **0**; `OSError: [Errno 28]` = **0**;
log **không bị cắt** (540 dòng, byte cuối `\n`, dòng tổng kết trọn vẹn).

**4. Failure diff** vs baseline `20261110-e2e-repair/full_suite_68ad75e.txt` (18 failed):

| | |
|---|---|
| new failed | 18 |
| baseline failed | 18 |
| **NEW** | **0** |
| **DROPPED** | **0** |
| **COMMON** | **18** |

**NEW = 0** — không còn hồi quy nào thuộc chiến dịch này. 18 lỗi còn lại là tập COMMON có sẵn trong
baseline, không thay đổi bởi đợt làm việc này (BLIP resident-worker, circus CLI resume, TME lifecycle/
precision, F08 DCNet config, handoff SHA, release-blocker GPU records, prepare_model recipe).
Ba tệp từng fail trong sweep đầu (`test_throughput_knobs.py`, `test_controller_installation.py`,
`test_web_scope_and_labels.py`) nay **pass bên trong lần chạy đầy đủ**.

**5. Dọn dẹp:** `/tmp/sweep_basetemp_final` đã xoá; không còn tiến trình `pytest` nào (kể cả xdist
`gw0/gw1/execnet`) sau khi kết thúc.

Raw: `25-sweep-final-preflight.txt`, `26-sweep-final-result.txt`, `27-sweep-final-failure-diff.txt`,
`28-sweep-final-accounting.txt`, log đầy đủ `raw/full_suite_final_sweep.txt`.

### Sweep sớm nhất — cây trước khi sửa 6 hồi quy (24/2793/47), giữ để đối chiếu

- **`24 failed, 2793 passed, 47 skipped, 350 warnings in 3318.19s (0:55:18)`**
- Accounting: 24+2793+47 = **2864** = collected ⇒ **GAP = 0**.
- `E`/`ERROR` = 0; `Errno 28` = 0; log không cắt (562 dòng).
- Diff: **NEW = 6, DROPPED = 0, COMMON = 18** — chính 6 hồi quy đã sửa ở mục dưới.
- Raw: `19-sweep-result.txt`, `20-failure-diff.txt`, `raw/full_suite_verification_sweep.txt`.

### Khác biệt cây giữa hai lần sweep — bằng chứng

Xem `24-production-diff-verdict.txt`. Không đường dẫn production nào đổi nội dung: `workbench/backend`,
`workbench/frontend`, `workbench/scripts`, `workbench/replay`, `workbench/registry` = 0 tệp đổi;
`workbench/composition/inputs.py` có mtime mới hơn nhưng **nội dung y hệt HEAD** (`git diff --quiet`
pass — chỉ bị test ghi chạm mtime). Ba tệp test + `.gitignore` là toàn bộ khác biệt.
Hạn chế được nói rõ: **không có snapshot byte của cây lúc run 1**; bằng chứng là thứ tự mtime cộng so
sánh nội dung của git với HEAD, không phải bản sao lưu.

### `compile_module`: token tất định (sửa sau sweep)

`hash()` trên `str` bị **salt theo từng process** (`PYTHONHASHSEED`), nên cùng một module emit vào thư
mục khác nhau mỗi lần chạy: tên không tái lập được và các thư mục `build-*` tích tụ vô hạn. Đã đổi sang
digest nội dung `hashlib.sha1(f"{out_dir}\x00{source}")[:12]`.

Bằng chứng (raw `29-compile-token-determinism.txt`): hai lần gọi trong **hai process riêng biệt** cho
token **giống nhau** — `build-3684818-6fc5ccda0bdd` và `build-3684860-6fc5ccda0bdd`; trong khi `hash()`
cũ cho ba giá trị khác nhau ở ba lần chạy. `test_web_scope_and_labels.py` → **14 passed**; toàn bộ nhóm
module frontend → **85 passed**.

Số đã đo trước khi sửa: **34** thư mục `build-*` trên toàn bộ `/tmp/cir_frontend_*`, tổng **676K**, do
**5** process pytest khác nhau sinh ra (`3496789 3497155 3498553 3581523 3582641`) — tức mỗi process
sinh ~7-9 thư mục và không có gì dọn chúng.

## Bốn việc chốt sau sweep

### 1. `compile_module` — token tất định (ĐÃ SỬA)

`hash((str(out_dir), source))` bị salt theo process. Đổi sang digest nội dung:

```python
token = hashlib.sha1(f"{out_dir}\x00{source}".encode()).hexdigest()[:12]
out_dir = out_dir / f"build-{os.getpid()}-{token}"
```

Bằng chứng (raw `30-compile-token-and-cleanup.txt`): hai lần gọi trong **hai process riêng** →
`build-3686400-53d4f6eec7d6` và `build-3686445-53d4f6eec7d6` — token **giống nhau**
(`53d4f6eec7d6`). `hash()` cũ cho ba giá trị khác nhau ở ba lần chạy.

Đã xoá thư mục tích tụ: **50** thư mục `build-*` trên toàn bộ `/tmp/cir_frontend_*`, tổng **1016K**
(tăng từ 34/676K ở lần đo đầu, do chính các lần verify của tôi). Sau khi xoá còn 2 thư mục / 32K là
output của phép chứng minh. `test_web_scope_and_labels.py` → 14 passed; nhóm frontend → 85 passed.

### 2. `runner.py` — comment về seeding đã sai (ĐÃ SỬA)

Câu "no seeding call exists anywhere in the composition path" nay sai: `model_encoder.py:392` pin
`torch.manual_seed(RELATION_SEED)` ngay trước `Encoder(...)`. Comment mới nói đúng thực tế: `--seed`
KHÔNG được hash (không forward pass nào rút ngẫu nhiên), nhưng seeding **có tồn tại và theo từng model**;
model nào rút từ RNG toàn cục lúc khởi tạo thì phải pin và **ghi lại** — bản ghi đó là `relation_state`,
và nó nằm trong `identity`.

### 3. `relation_state` — nối vào identity, summary, parity gate (ĐÃ SỬA)

- **`identity`**: thêm `"relation_state": _relation_identity(adapter.relation_state)` gồm `status`,
  `seed`, `sha256`, `expected_sha256`, `matches_pinned_digest`. Chỉ ba trường mô tả **cái đã rút** đi vào
  hash (không có `path`, `transform_policy`) để run id phụ thuộc phép đo chứ không phụ thuộc nơi lưu tệp.
- **helper**: `RELATION_NOT_RECORDED`, `_relation_identity`, `relation_state_from_summary` — cùng hợp
  đồng với `policy_observed_from_payload`: artifact cũ không có trường này đọc là `not_recorded`,
  **không bao giờ** là pass.
- **summary**: `"relation_state": identity["relation_state"]` dưới khoá riêng để gate đọc được mà không
  phải parse hash.
- **`parity_gate.py`**: dimension mới `relation_parity` + `_relation_checks`. Ba kết cục tách biệt:
  cùng digest → ok; khác digest → **không ok, mandatory** (nên không thể `PARITY_PASS`); một bên không
  ghi → **không ok nhưng không mandatory**, dimension đọc `False` kèm `note` "not_recorded".

Vì sao tách mandatory: `REQUIRED_DIMENSIONS` không đổi, nên 18 artifact cũ (không có `relation_state`)
**giữ nguyên verdict đã đạt** — đúng yêu cầu "keep old artifacts behaving", theo đúng khuôn
`policy_observed`. Nhưng chúng không được đọc là pass ở dimension mới. Đã kiểm:
`test_gate_parity_subchecks.py` (30 passed cùng các test relation/encoder).

### 4. `native_measurements` unit trên dữ liệu thật — advisory đã cũ (ĐÃ CHỨNG MINH)

Đo trên **payload sống** (`/api/native-measurements`), không phải fixture — raw `31-live-unit-audit.txt`:

| nhóm (chỉ dòng `artifact_backed: true`) | số dòng |
|---|---|
| backed + `declared` | **2** (`clip4cir_rn50x4_fullft` ×2) |
| backed + `sibling_key` | **2** (`dcnet` ×2) |
| backed + `inferred_from_magnitude` (không khai báo nào tồn tại) | **21** |
| **tổng backed** | **25** (2+2+21 = 25 ✔) |

Advisory nói `clip4cir` còn `inferred_from_magnitude`: **sai** — nó là `percent/declared`.
Advisory nói chỉ đúng trong fixture: **sai** — payload sống cho cùng kết quả.
`combiner_rn50x4_noft` **đúng là** `inferred_from_magnitude`, và điều đó trung thực: report chống lưng
của nó có **0** khoá đơn vị (đã quét `unit`/`percent`/`pct`/`fraction`/`scale`/`ratio`).

## Dịch vụ sau khi sửa (khởi động lại đúng một lần)

- Backend `:8000`: PID **3691345** (con của **3691344**), env
  `WORKBENCH_CIRCUS_RELEASE_ROOT=…/artifacts/circus_release`.
- `/api/benchmark-runs` → **HTTP 200** (ready).
- Bundle: **`assets/index-DWpK0yeS.js`**, sha256
  **`7e34724ebf55092e36524b98980cca55e71f3ddbaa62489c65577f92e18a6a16`**; bytes phục vụ **trùng khớp**
  `dist/` (`cmp` → IDENTICAL). **Đã SUPERSEDED** — bundle sống là `assets/index-CBR6lHgq.js`, sha256
  `acfb5fbbced77e88b86ee542e5f261c6df71c0e0e8039d753965e1fc873ac21a` (`cmp` → IDENTICAL với `dist/`; `index-BLH2yome.js` / `702ed2e9…d681` và `index-DyVaNikG.js` / `81d61f65…0981` cũng đã superseded); đọc
  lại mỗi lần build bằng `curl -s http://127.0.0.1:5173/ | grep -o 'assets/index-[^"]*\.js'` rồi
  `sha256sum workbench/frontend/dist/assets/<file vừa in ra>`.
- Raw: `32-service-identity-final.txt` (bản ghi của lượt này).

## Tóm tắt phát hành

- **Sweep trung gian** (trước `relation_state`, bị các lượt sau vượt qua): `18 failed, 2799 passed, 47 skipped`
  — **NEW = 0 / DROPPED = 0 / COMMON = 18**.
- **GAP = 0** (18+2799+47 = 2864 = collected).
- Bốn khẳng định Phase 1: **PASS** cả bốn.
- Không còn hồi quy nào thuộc chiến dịch này.

## SWEEP TẠI `be14393` — **BỊ THAY THẾ (superseded)**

> **Đã superseded.** Sweep có thẩm quyền là sweep chốt tại `0691f3e` (= `TESTED_SOURCE_SHA`):
> `18 failed, 2931 passed, 47 skipped`, `2996 collected`, gap 0, NEW=0 / DROPPED=0 / COMMON=18 —
> bản ghi `raw/verify/73-sweep-close.md`. Lượt `7babfd9` (cùng 2931/18/47 = 2996,
> `raw/verify/72-sweep-pin.md`), `cd980dc` (`18 failed, 2901 passed, 47 skipped`, 2966,
> `raw/verify/68-final-sweep.md`) và `c755673` (2914/18/47 = 2979, `raw/verify/70-sweep-final.md`)
> cũng đã **bị thay thế**. Mục dưới đây giữ nguyên số của lượt `be14393` như **bản ghi
> lịch sử của lượt đó**, không còn là "số chốt" của cây phát hành.

Đây là lần đo trên cây sau `relation_state` (`runner.py`) và `relation_parity`
(`parity_gate.py`), sau khi `limn-encoder` chốt option (b) cho derived-status, và sau khi test doubles
BLAS / import scanner / `compile_module` được sửa. Production của cây đó đã bị `cd980dc` rồi `0691f3e` vượt qua
(`git diff be14393 0691f3e` ngoài `workbench/docs` và `workbench/tests` **không** rỗng).

**1. Dòng tổng kết nguyên văn** (của lượt `be14393`, đã **superseded**; dòng có thẩm quyền là
`18 failed, 2931 passed, 47 skipped, 350 warnings in 3203.74s (0:53:23)` tại `0691f3e`):

```
18 failed, 2800 passed, 47 skipped, 350 warnings in 3540.59s (0:59:00)
```

**2. Accounting:**

| | |
|---|---|
| failed | 18 |
| passed | 2800 (của lượt `be14393`, đã superseded) |
| skipped | 47 |
| **tổng** | **2865** |
| `--collect-only -q` (trước run) | 2865 |
| `--collect-only -q` (sau run) | 2865 |
| **GAP** | **0** |

Tổng collected **tăng +1** so với lần sweep trước (2864 → 2865). Nguồn: `test_derived_subset_status.py`
được `limn-encoder` viết lại lúc 01:22:41 để chuyển sang option (b); file này hiện có **5 node**. Delta
chính xác **không chứng minh được** vì file còn untracked (không có snapshot git của số node trước đó) —
nêu rõ để không ai đọc con số +1 như một phép đo. Sau đó số collected còn tăng tiếp 2865 → 2966 → 2996 khi lô
`d6d3e36…cd980dc` thêm test và hai commit frontend `252b380`/`d323a5c` vào tới `7babfd9`, rồi `0691f3e` thêm test
node; lượt **có thẩm quyền** là lượt cuối tại `0691f3e` (`2996 collected`, `2931 passed`), xem
`raw/verify/73-sweep-close.md`.

**3. Sức khoẻ log:** `ERROR` row = **0** (nghĩa là không có collection error; 325 dòng bắt đầu bằng `E `
là **thân traceback** của `--tb=short`, không phải lỗi collect). `OSError: [Errno 28]` = **0**. Log
**đầy đủ**: 975 dòng, byte cuối `\n`, dòng tổng kết trọn vẹn.

**4. Failure diff** vs `full_suite_68ad75e.txt`:

| | |
|---|---|
| **NEW** | **0** |
| **DROPPED** | **0** |
| **COMMON** | **18** |

**NEW = 0.** Không có hồi quy nào. QUAN TRỌNG: **`relation_parity` KHÔNG làm fail thêm test nào** trong
lần chạy đầy đủ — dimension mới báo `not_recorded` cho artifact cũ nhưng vì nó **không mandatory** nên
18 artifact cũ giữ nguyên verdict, đúng thiết kế. Không có test gauge nào mới hỏng.

**5. `F` ở ~5% của lần chạy bị huỷ — KHÔNG tái diễn.** Test tại vị trí 204
(`test_autopilot_cold_start.py::test_mutation_invalidates_only_dependent_evaluations`) không xuất hiện
trong danh sách FAILED của lần chạy này. Kết luận: nó là **order-dependent**, không phải hồi quy của
`relation_state`/`relation_parity`. (Nó cũng không nằm trong baseline 18.)

**6. Dọn dẹp:** `/tmp/sweep_basetemp_ship` (**30 GB**) đã xoá; không còn tiến trình `pytest` hay xdist
nào sau khi kết thúc.

**7. Cây không đổi trong lúc chạy:** `git status --porcelain -uall` được chụp trước khi launch và một lần
**trong lúc sweep đang chạy** — hai bản **byte-identical** (sha256
`d271b76baedf98ec154761061051cd847294ed6f9cb34ba5ce22c43ad6d16115`). Không có drift, nên lần đo này hợp
lệ. Các tệp `raw/verify/*` được ghi **sau** khi sweep kết thúc, không phải trong lúc chạy.

Raw: `33-ship-sweep-preflight.txt`, `37-ship-sweep-result.txt`, `38-ship-failure-diff.txt`,
`39-ship-accounting-and-drift.txt`, log đầy đủ `raw/full_suite_ship_sweep.txt`.

### Bảng số qua bảy lần sweep (giữ nguyên lịch sử)

| cây | dòng tổng kết | NEW | DROPPED | COMMON | collected | GAP |
|---|---|---|---|---|---|---|
| trước khi sửa 6 hồi quy | 24 failed, 2793 passed, 47 skipped | 6 | 0 | 18 | 2864 | 0 |
| trước `relation_state` | 18 failed, 2799 passed, 47 skipped | 0 | 0 | 18 | 2864 | 0 |
| `be14393` — **superseded**, bản ghi lịch sử | 18 failed, 2800 passed, 47 skipped | 0 | 0 | 18 | 2865 | 0 |
| **`cd980dc` — superseded, bản ghi lịch sử** | 18 failed, 2901 passed, 47 skipped | 0 | 0 | 18 | 2966 | 0 |
| `c755673` — **superseded**, bản ghi lịch sử | 18 failed, 2914 passed, 47 skipped | 0 | 0 | 18 | 2979 | 0 |
| `7babfd9` — **superseded**, bản ghi lịch sử | 18 failed, 2931 passed, 47 skipped | 0 | 0 | 18 | 2996 | 0 |
| **`0691f3e` — cây phát hành, CÓ THẨM QUYỀN** | **18 failed, 2931 passed, 47 skipped** | **0** | **0** | **18** | **2996** | **0** |

Đọc bảng này theo thứ tự thời gian: mỗi dòng là một cây KHÁC NHAU, không phải cùng một cây chạy lại.
Dòng cuối là cây phát hành và là số **có thẩm quyền** (`raw/verify/73-sweep-close.md`, 3203.74 s, `^ERROR ` = 0,
`Errno 28` = 0, hai phép chứng minh cây đứng yên); các dòng `be14393`, `cd980dc`, `c755673` và `7babfd9` được giữ
làm bản ghi của lượt đó và đã bị thay thế.
