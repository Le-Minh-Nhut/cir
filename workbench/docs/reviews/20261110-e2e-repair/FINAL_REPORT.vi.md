# FINAL REPORT — chiến dịch sửa lỗi `20261110-e2e-repair`

Nhánh `research/cir-failure-workbench`, HEAD `d850f9811f7670fa7c0485cd6618ec59dd662ae3` (`d850f98`),
bằng `origin/research/cir-failure-workbench`. Hai mốc trước: `3828163` (commit chiến dịch) và
`2e2e2d2` (mốc đầu chiến dịch); kết luận nào chỉ đúng ở mốc cũ đều được gọi tên đích danh.
Báo cáo này thay thế phần kết luận của
`workbench/docs/reviews/20261109-native15-composition-limn/FINAL_REPORT.vi.md` (gọi tắt **BC-cũ**).
Mọi số dưới đây ghi kèm file. Bản đồ artifact → phát biểu ở `EVIDENCE_INDEX.md` cùng thư mục.
Sổ phát hiện đầy đủ 14 dòng: `FINDINGS_AND_FIXES.md`.

> **Quy ước trạng thái (đã cập nhật lần cuối).** `HEAD` = `d850f98`. Toàn bộ 14 phát hiện được sửa ở
> `3828163` hoặc `68ad75e`; phần còn mở **không** phải "chưa sửa code" mà là "sửa rồi nhưng chưa nghiệm
> thu" hoặc "không thể sửa vì upstream". Mọi tuyên bố "còn mở" dưới đây đều nêu đúng lý do đó.
> Không file nào trong báo cáo này commit gì; git do phiên khác sở hữu.

> **Tầng bằng chứng.** `EXEC` = log/command+meta của lần chạy thật · `MEAS` = file measurement mang
> số · `READ` = chỉ đọc source · `NOT_RUN` = không có gì.

## 1. Tổng quan — cái gì xong, cái gì không

| hạng mục | đạt / tổng | trạng thái | nguồn |
|---|---|---|---|
| Native: target active có evaluation đã thực thi | **14 / 15** | `pair` thiếu | `raw/r13_extra_measurements_audit.md` §9, `raw/r13_ptha_proof.md` |
| — trong đó có `EXITCODE=0` (hoặc `[OK] result written`) | **13 / 15** | `limn` không có exit code; `pair` không có gì | log ở mục 3 |
| — khai `native_evaluator` ngay trong measurement | **12 / 15** | `clvc_net`, `tgcir`, `ptha_mtst` chỉ có log | `raw/r13_extra_measurements_audit.md` §2 |
| G1/G2 có artifact | **2 / 15** | chỉ `limn`, `encoder` | `PARITY.vi.md` |
| G1/G2 **PASS** | **0 / 15** | cả hai model có artifact đều `PARITY_MANDATORY_SUBCHECK_FAILED`, `passed_models = []` | `raw/guards/r9_after.txt` |
| Composition CIRCUS phục vụ được | **35 row** (17 full / 9 sf / 9 validated) | 14 `COMPLETED`, 2 `PARTIAL`, 1 `FAILED` ở subset full | `/api/composition/runs`, `RUN_MATRIX.csv` |
| LIMN canonical | **PARTIAL 6001/6003** | 2 query mơ hồ vẫn chưa chấm | `limn_dual_scope.json` |
| W1 (web báo 0) | **đã sửa** | landing = CIRCUS Full; banner có nhánh loading, `false-claim-seen: False` | `web_evidence/after-banner-loading.txt`, §7.2 |
| R13 (native artifact không lên web) | **đã sửa** | 32 file / 43 block / 28 model có số, `verified: 0`; 5 model từng bị bỏ nay đều có mặt | `web_evidence/after-native-measurements.json`, §3.5 |
| R9 (gate) | **đã code hoá + đã commit** | `parity_gate.py` + route ở `3828163`; `GET /api/runs/parity-gate` → **200**, `passed_models: []` | `web_evidence/after-parity-gate.json`, `web_evidence/1920-14-parity-gate.png`, §4.3 |
| R12 (5 model) | **4 BLOCKED + 1 NOT_A_BLOCKER** | nguyên nhân đo được từng cái | `BLOCKERS_R12.vi.md` §6 |
| R5 (LIMN ghi sai target) | **đã sửa + đã vá artifact** | producer lấy DID theo qid lúc serialize + assert 12 ô; artifact lệch đúng **6 ô**, rank/score 0 đổi | `limn_dual_scope.json` sha256 `d1b4aadb…73e3f`, §10.2 |
| R1 (CLIP4Cir label) | **đã sửa, chưa re-ingest** | migration sửa được câu trích sai, giữ bản cũ ở `superseded_protocol_source`; artifact đã sửa mới là **bản sao** | `raw/guards/clip4cir_fullft.migrated.r1.json`, §10.2 |
| Sweep đầy đủ ở `68ad75e` | **18 failed / 2701 passed / 47 skipped** | gap accounting **0**; diff so baseline **0 new / 0 dropped**; guard **8 passed** | `full_suite_68ad75e.txt`, `full_suite_68ad75e.failure_diff.txt`, §10.1 |

**Không tuyên bố chiến dịch đã pass.** Gate bắt buộc còn `FAIL` (G1/G2), `NOT_RUN` (13 model G1/G2,
toàn bộ 41 dòng validation composition, 10/11 dòng native validation), và `BLOCKED` (4 model).

Số model G1/G2 phải đọc là **0/15 pass**, không phải "2/15". BC-cũ ghi "G1+G2 2/15" và ở `:128-129`
gọi limn `PASS` / encoder `PASS-with-caveat`; gate gọi **cả hai FAIL**. Đây là mâu thuẫn
report-vs-code trực tiếp, và cái đúng là của gate (xem §4.3).

## 2. Ba đính chính bắt buộc (đảo lại BC-cũ)

### 2.1 Census native đã sai — "11/15, bốn model thiếu" là đếm thiếu

**BC-cũ `:1`, `:15`, `:308`** nói `airknow`, `clvc_net`, `tgcir`, `pair` chưa có native run.

**Đo lại.** `raw/r13_census_v2.md` quét **toàn bộ** JSON dưới `checkpoint_selection/` (trừ
`downloads/`, `site-packages/`) tìm `macro_r10` ở **mọi độ sâu**: **17/17** model trong registry có
≥1 file mang macro. `raw/r13_extra_measurements_audit.md` chứng minh bằng log thực thi rằng ba trong
bốn cái "thiếu" **đã chạy** (`$D1` = `…/checkpoint_selection/20261106T002617+0700`):

| model | log | nội dung quyết định |
|---|---|---|
| `airknow` | `$D1/logs/airknow_n05_run1.log` | dòng cuối `[OK] result written: …/airknow_fiq_n05_run1.json`; measurement tự khai `native_evaluator = test_BLIP2.py:test` |
| `clvc_net` | `$D1/logs/clvc_run1.log` | 3 dòng per-category + `aggregate {"macro_r10": 29.970746517160162 …}` + **`EXIT=0`**, trùng từng chữ số với `measurements/clvc_run1.json` |
| `tgcir` | `$D1/logs/tgcir_run2.log` | 3 dòng per-category + `aggregate {"macro_r10": 49.720772235486656 …}` + **`EXIT=0`**; `tgcir_run1.log` là traceback `EXIT=1` |

**Vì sao BC-cũ sai:** nó chỉ đọc `raw/native/` **của chính nó**; `artifacts/logs/` không có thư mục cho
ba model này (chạy **trước** quy ước đó; log ở `$D1/logs/`, 18 file, mtime 11-06 00:44–01:17).
**Phát biểu bảo vệ được:** 14/15 target active có native evaluation đã thực thi, mỗi cái có log stdout
**hoặc** command+meta với `EXITCODE=0` khớp measurement mang macro; `pair` là cái duy nhất còn thiếu.

**Carve-out tôi tự đo, chưa worker nào ghi** — vế `EXITCODE=0` **không đúng cho `limn`**:
`$D1/cohort/logs/limn_run2.log` có đủ 3 dòng `{"r10": 51.1651 …}` nhưng **kết thúc bằng traceback**
`AttributeError: 'dict' object has no attribute 'complete'` tại `run_limn_oc220.py:73`, **không** có
`EXIT=0`. Measurement mtime `22:36:11` **sau** log `22:35:49` 22 giây, tức do một lần chạy khác (dòng
73 crash **trước** `out.write_text`). Đọc chặt vế `EXITCODE=0` thì con số là **13/15**; 14/15 chỉ đứng
nếu chấp nhận "số có trong log, không có exit code". `limn` là **EXEC_WEAK**, không phải EXEC.

Hai khiếm khuyết ghi chép (không phải số): `tgcir_run1.json` sinh bởi `run2.log`;
`clvc_net`/`tgcir` thiếu `provenance.native_evaluator`.

**Cạm bẫy census v1** (`raw/native_measurement_census.txt`, `raw/r13_census_v2.md`): chỉ quét
`**/measurements/*.json` với đúng một hình dạng `aggregate.macro_r10`, bỏ sót **bốn** hình dạng — mỗi
cái làm một model "biến mất": `macro.macro_r10` (`pair`), `macro_metrics.{r10,r50}` (`ptha_mtst`),
top-level `macro_r10` (`csmcir`), `clip4cir_{noft,fullft}.macro_r10` (hai entry CLIP4Cir). Đây **cùng
một lớp lỗi** với ca `?q=` FALSE NULL: selector sai bị đọc thành dữ liệu vắng.

### 2.2 Nhãn G1/G2 đã sai — không phải "2", encoder là **5**, limn là **3**

`PARITY.vi.md` §"R2 × R9" đo sàn native-vs-native trực tiếp từ `raw/encoder_parity/native.{cat}.npz`
(file có **cả** population train-transform và val-transform), rồi phân loại 4 block
`within_tolerance: false` mỗi category của encoder:

| block | số đo (dress / shirt / toptee) | phân loại |
|---|---|---|
| `layer4_forward_sample.query_native_tensors` | 0.021249 / 0.041421 / 0.023708 — **trùng 6 chữ số** với sàn sample-level | `NOT_COMPARABLE` |
| `population.features_vs_native_train_query` | 0.120392 / 0.122951 / 0.125173 — **trùng 6 chữ số** với sàn population | `NOT_COMPARABLE` |
| `population.features_vs_native_val_control` | max_abs 1.10e-6 / 1.28e-6 / 0.99e-6 (**dưới** `forward_max_abs=1e-5`); chỉ `max_rel` fail, mẫu số 3.3e-7 | `DENOMINATOR_ARTEFACT` |
| `batch_size_1_vs_n` | max_abs 3.20e-7 / 2.81e-7 / 3.28e-7; `max_rel` 7.80e-3 / 3.13e-3 / 1.16e-2 vs khai 1e-3 | `GENUINE_FAIL` |

Hai block đầu **không phải phép kiểm parity**: chúng so native-train-transform với adapter, mà train
transform là `RandomResizedCrop` **ngẫu nhiên** (`raw/r2_transform_tensors.json`: hai lần rút cùng ảnh
lệch `max_abs 3.8270`, control val-vs-val `0.0`). Chúng trùng sàn tới 6 chữ số vì chúng **chính là**
sàn đó.

**Đếm bảo vệ được:** `encoder` **5** (3 block `batch_size_1_vs_n` + 2 thay đổi hạng **không-tie**),
`limn` **3** (1 block `batch_size_1_vs_n` mỗi category), và **0** đổi R@10/R@50 ở cả hai. Hai thay đổi
hạng của encoder, đọc từ `g1_g2.encoder.{dress,shirt}.adapter.json` →
`scoring_ranking.rank_differences_vs_native_val_control`: dress index 1569 (adapter 1216 vs native
1215, `min_gap_abs 6.5717e-4`), shirt index 1352 (1372 vs 1373, `1.6041e-4`), toptee **0**.

Ba con số đã lưu hành, hai trong ba sai: **"2"** (BC-cũ `:207`, `GATES.csv:4`) bỏ sót 3 block batch;
**"13"** (gate) đếm **theo check** không theo sai lệch — 5 check/category + 3 cho toptee, gồm cả
`DENOMINATOR_ARTEFACT` và `LABEL-DISAGREEMENT`, nên không dùng làm "số sai lệch" được; **"1"** cho
limn (BC-cũ) đếm một block chung cho ba category.

### 2.3 CLIP4Cir FullFT bị dán nhãn sai, và "đính chính" của BC-cũ cũng sai

BC-cũ §2.2 đặt `clip4cir_rn50x4_fullft` dưới `fashioniq_full_gallery_ref_excluded` và **gọi đó là một
đính chính**. Sai hai lần. Đọc source upstream đã pin (`third_party/CLIP4Cir`, `dfed9f74…`):

* `src/validate.py:21-57` `compute_fiq_val_metrics` — **không** ghi mask nào: `F.normalize(index)`
  (`:41`), `distances = 1 - pred @ index.T` (`:44`), `argsort` ascending (`:45`), label theo target
  (`:49-50`).
* `reference_mask` xuất hiện **đúng hai dòng**, `:163-165`, **bên trong** `compute_cirr_val_metrics`
  (bắt đầu `:135`) — nhánh CIRR, không chạy cho FashionIQ.
* Gallery của chính run: **3817 / 6346 / 5373** (`raw/native/clip4cir_fullft.json`) = full split.
* `registry/models.yaml:1184` đã khai `native_protocol: fashioniq_original_split` — registry **đúng
  từ trước**.

⇒ Nhãn đúng là `fashioniq_original_split`. Câu `protocol_source` của artifact ("the reference is
excluded by masking its own index entry") **sai**, và nó là câu **tự khai của artifact**.

**Đã sửa bằng migration có kiểm chứng, không rerun:** `scripts/migrate_protocol_label.py`,
`tests/test_protocol_label_migration.py`, `raw/clip4cir_fullft.migrated.json` + `.migration.json`,
`raw/r1_migration_evidence.json`. Công cụ **từ chối** khi thiếu citation / `gallery_count` mâu thuẫn /
đổi sang chính nhãn cũ; đóng băng `aggregate`, `categories`, `checkpoint`, `source`, `runtime`,
`side_effects`, `complete_benchmark`, `model_id`, `protocol_source` và **assert bằng nhau**. Đo được:
sha256 `fb97cfabdc69…` → `dd9176cf3778…`; `macro_r10` giữ `43.8016931215922`, `macro_r50`
`66.98358456293742`, mọi `categories[]` không đổi; 5 test pass. Artifact gốc **không** bị chạm.

**Khiếm khuyết của chính remedy, tôi đo được:** `protocol_source` nằm trong `FROZEN_FIELDS`
(`:33-34`) với lý do "source does not change" — nhưng ở ca này **câu đó sai về source**. Test
`:57` khẳng định `after["protocol_source"] == before[...]`, tức **khoá câu sai vào bản đã migrate**.
Kết quả: file migrate mang nhãn đúng **và** citation sai; phải sửa tay trước khi dùng làm bằng chứng.

**Reference có xếp trên target của nó không:** reference **là** ứng viên hợp lệ, không phải target,
không mask nào loại nó — đúng như source; cùng lý do đó nhãn `…_ref_excluded` không mô tả phép chấm nào.

## 3. Native — kết quả theo protocol, đơn vị, mức xác minh

Ba protocol là ba định nghĩa gallery/mẫu số khác nhau. **Không** xếp chung leaderboard, **không**
lấy trung bình chéo protocol. `macro R@k` = trung bình không trọng số trên 3 category của **chính
model đó**.

### 3.1 `fashioniq_full_gallery_ref_excluded` — gallery full split, reference bị mask `-10e10`

| model | R@10 | R@50 | đơn vị | tầng | file |
|---|---:|---:|---|---|---|
| `conesep` | 54.7613 | 75.2151 | percent | EXEC (`[OK] result written` + `conesep.meta.txt EXITCODE=0`) | `$D1/cohort/measurements/conesep_fiq_n02.json` |
| `habit` | 54.8733 | 74.5034 | percent | EXEC (idem) | `$D1/cohort/measurements/habit_fiq_n02.json` |
| `intent` | 54.8662 | 75.4319 | percent | EXEC (idem) | `$D1/cohort/measurements/intent_fiq_n02.json` |
| `airknow` `fiq_n05` | 52.5466 | 72.9026 | percent | EXEC (`[OK] result written`) + measurement tự khai `native_evaluator` | `$D1/measurements/airknow_fiq_n05_run1.json` |
| `airknow` `fiq_n08` | 50.6417 | 71.4556 | percent | EXEC (idem) | `$D1/measurements/airknow_fiq_n08_run1.json` |
| `dcnet` | **27.9297** | **53.7272** | percent ở file measurement, **fraction** ở file raw (`0.27929730914548917` / `0.5376075764301528`) | EXEC (`dcnet.time.log`: `Exit status: 0`, `Elapsed 2:07:35`) | `$D1/measurements/dcnet_run1.json`, `raw/native/dcnet.json` |

`dcnet` là **cái bẫy đơn vị**: hai file cùng lần chạy, một ghi `0.2793`, một ghi `27.9297`. `r1 = null`
ở cả ba category là **đúng** (evaluator DCNet chỉ tính R@10/R@50). `comb_avg 0.40845` là trung bình
sáu ô, **không** phải macro.

### 3.2 `fashioniq_val_split` — gallery pair-union

| model | R@10 | R@50 | ghi chú | tầng | file |
|---|---:|---:|---|---|---|
| `hint` | 59.8034 | 80.1308 | — | EXEC (idem) | `$D1/cohort/measurements/hint_fashioniq.json` |
| `encoder` | **56.2231** | 77.6674 | **hai giá trị** cho cùng checkpoint `0f31c501…9173`, xem §3.4 | EXEC (log có `EXITCODE=0` nhưng **không** in số) | web phục vụ file này |
| `limn` | 55.7919 | 77.6378 | + R@1 **23.3394** | **EXEC_WEAK** (số có trong log, log kết thúc bằng traceback, **không** exit code) | `$D1/cohort/measurements/limn_base_iter0.json` |
| `tgcir` | 49.7208 | 71.1023 | measurement tên `run1`, sinh bởi `run2.log` | EXEC | `$D1/measurements/tgcir_run1.json` |
| `clvc_net` | 29.9707 | 58.6217 | `native_evaluator` **trống** | EXEC | `$D1/measurements/clvc_run1.json` |
| `pair` | 34.0350 | 56.5957 | **không đếm** — `mapping_status = EXPLORATORY_PUBLIC_MAPPING`; chính file ghi `paper_reference.macro_r10 = 52.76` | MEAS (không phải native) | `$D1/pair_compat/measurements/pair_b1_H_normal.json` |

`pair` là target duy nhất **không có** native run. Con số 34.04 so với 52.76 do chính file đó ghi là
dấu hiệu mapping sai, không phải model yếu; nó **không** được vào bảng nào.

### 3.3 `fashioniq_original_split` — gallery full split, reference eligible

| model | R@10 | R@50 | ghi chú | tầng |
|---|---:|---:|---|---|
| `tme` | 54.9229 | 74.8384 | — | EXEC (`tme.meta.txt EXITCODE=0`; stdout in `num_targets/num_queries` ba category) |
| `ptha_mtst` | 37.8978 | 59.8364 | **TRANSFER CIRR→FashionIQ, KHÔNG phải tái lập**; log tự ghi `mismatch: "training dataset/checkpoint differs; numerical context only, not reproduction"`, Δ = −17.09 r10 / −16.44 r50 | EXEC (`ptha_transfer_run1.log` `exit=0` + command + meta) |
| `combiner_rn50x4_noft` | **35.1407** / **35.1078** | **58.4436** / **58.4756** | **hai lần chạy khác nhau**, xem dưới | EXEC (queue proof) |
| `clip4cir_rn50x4_fullft` | **43.8017** / **43.8182** | **66.9836** / **66.9501** | **hai lần chạy khác nhau**, xem dưới | EXEC (`clip4cir_fullft.time.log` `Exit status: 0` + stdout đủ 8 số) |
| `csmcir` | 57.0507 | 77.2986 | deferred, ngoài 15 active | MEAS (`evidence_class = PUBLIC_CAPTION PATH-MAPPED RESEARCH RUN`) |
| `sprc` | 51.0386 | 73.3263 | deferred, ngoài 15 active | MEAS (`sprc_vitl_run5.log` có dòng aggregate khớp, **không** có `EXIT=`) |

**Hai cặp số cho cùng một checkpoint — chưa worker nào ghi.** Cùng `combiner`/`clip4cir` có **hai**
lần đo khác nhau:

| model | nguồn A | nguồn B | Δ R@10 |
|---|---|---|---|
| `clip4cir_rn50x4_fullft` | `43.8016931215922` (`raw/native/clip4cir_fullft.json`, có provenance, đã migrate nhãn) | `43.81821850935618` (`$D1/measurements/clip4cir_parsed.json`, **không** có `aggregate`, **không** provenance) | **0.0165** |
| `combiner_rn50x4_noft` | `35.14068226019541` (`clip4cir_parsed.json`) | `35.1078` (queue proof `…020a70b0`, `EXTRACTED_VERIFIED`) | **0.0329** |

**Sửa lại sau `3828163`:** câu "nguồn B không được phục vụ" chỉ đúng ở `2e2e2d2`. Catalog native hiện
nhận cả `clip4cir_parsed.json` và các roll-up (đánh dấu `rollup: true`), nên `clip4cir_parsed.json`
**có** row với `clip4cir_fullft 43.8182` / `clip4cir_noft 35.1407`
(`web_evidence/after-native-measurements.json`). Vẫn đúng và quan trọng: hai cặp là **hai lần chạy
khác nhau**, không cộng/gộp, và phải nói rõ đang trích lần nào — roll-up chỉ lặp lại số một file
khác đã nêu, không phải lần đo thứ ba.

### 3.4 `encoder` — hai macro cho một checkpoint, và đó là đúng như dự đoán R4

| nguồn | macro R@10 | macro R@50 | per-category (dress/shirt/toptee) R@10 |
|---|---:|---:|---|
| `$D1/cohort/measurements/encoder_fashioniq.json` (**web phục vụ**) | 56.2231 | 77.6674 | 51.8592 / 55.0049 / 61.8052 |
| `raw/native/encoder.json` (BC-cũ §3.6 trích) | 56.2908 | 77.6334 | 51.9584 / 54.9558 / 61.9582 |
| Δ | **0.0676** | 0.0340 | +0.0992 / −0.0491 / +0.1530 |

Hai file cùng `checkpoint_sha256 0f31c501…9173`, cùng `source_commit 29a2a31d…`, cùng protocol; khác
`recorded_at` (2026-11-06 vs 2026-11-09). Đây **chính là** biên độ do `learnable_relation` bị rút lại
mỗi process (R4, §5.2): BC-cũ §6c đo Δ nằm trong −0.05 … +0.10 mỗi category, và ba delta ở đây nằm
đúng trong khoảng đó. Hệ quả phải nói: **không có "con số native của encoder"**; có hai lần rút. Web
phục vụ lần rút 2026-11-06 mà không nói điều đó.

### 3.5 Catalog native — năm hình dạng macro, và bốn model từng bị bỏ
**`2e2e2d2`:** `native_measurements.py` chỉ đọc `aggregate.{macro_r10,macro_r50,macro_mean}` trên hai
glob ⇒ `ptha_mtst` (`macro_metrics`), `combiner_rn50x4_noft` + `clip4cir_rn50x4_fullft` (nested trong
`clip4cir_parsed.json`, bị `_is_measurement()` loại) và `pair` (`pair_compat/measurements/`, ngoài glob)
không có số trên web: 12 model.

**`3828163`:** reader được viết lại để nhận cả năm hình dạng — `aggregate.*`, `macro.*`, `macro_metrics`,
top-level `macro_r10`, `<variant>.macro_r10` — và gắn `rollup: true` cho các file tổng hợp, loại chúng
khỏi `counts.with_metrics`. Đo trên service (`web_evidence/after-native-measurements.json`):

| đại lượng | `2e2e2d2` | `3828163` |
|---|---:|---:|
| file có row | 21 | **32** |
| block mang metric | 15 | **43** |
| measurement/row không phải roll-up | 15 | **21** |
| model có metric (tính cả roll-up) | 12 | **28** |

`ptha_mtst`, `pair`, `clip4cir_rn50x4_fullft`, `combiner_rn50x4_noft`, `csmcir` nay đều có mặt với
`verified: false`; `csmcir 57.05073873202006` và `ptha_mtst 37.897844115893044` đã được đối chiếu lại
với chính file nguồn tới 1e-12. `rollup: true` là 22/49 row và **không** vào `with_metrics`.
Test: `workbench/tests/test_native_measurements_census.py` (16 test).

**Bài học giữ nguyên, đây là lý do phần này tồn tại:** census v1 (`raw/native_measurement_census.txt`)
chỉ quét `**/measurements/*.json` với đúng một hình dạng, nên báo "5 config không có file" — sai.
Lỗi lặp lại lần hai trong `native_measurements.py`, và lần đó cũng đã được sửa bằng danh sách hình
dạng thay vì thêm một `if` cho từng model.

## 4. G1/G2 theo model, tách theo dimension

### 4.1 Số đo từng model

| model | input | forward | batch | scoring | metrics | verdict gate | số sai lệch thật |
|---|---|---|---|---|---|---|---|
| `limn` | **PASS** | PASS | **FAIL** | PASS | — | `PARITY_MANDATORY_SUBCHECK_FAILED`, 3 failure | **3** |
| `encoder` | **PASS** | **FAIL** | **FAIL** | **FAIL** | PASS | `PARITY_MANDATORY_SUBCHECK_FAILED`, 13 failure | **5** |
| 13 model còn lại | — | — | — | — | — | `PARITY_NOT_RUN` | 0 (không đo) |

`raw/guards/r9_after.txt`: `ALL standard=parity-relative/v1 passed_models=[]`. limn: mỗi category
**đúng một** block fail, cùng block `layer2_4_forward/batch_size_1_vs_n`:

| category | max_abs | max_rel | khai `batch_max_rel` | min_cosine |
|---|---:|---:|---:|---:|
| dress | 1.751e-07 | **1.140e-02** | 1e-3 | 0.99999999999919 |
| shirt | 1.974e-07 | **2.698e-03** | 1e-3 | 0.99999999999894 |
| toptee | 1.714e-07 | **7.509e-03** | 1e-3 | 0.99999999999934 |

Không có thay đổi hạng nào ở limn (`unexplained_rank_differences = 0`).

### 4.2 Chuẩn đo tạo ra các cờ này — và nó không được đổi

`max_rel = max(|a−b| / max(|b|, 1e-12))`, thuần tương đối, không `atol`
(`t4_parity.py:222-224`, `encoder_parity.py` cùng dạng). Cặp `(max_abs, max_rel)` cho biết độ lớn
phần tử quyết định cờ: `|b| ≈ max_abs/max_rel` (`raw/r9_denominator.txt`):

| file | max_abs | max_rel | suy ra `|b|` |
|---|---|---|---|
| limn dress | 1.751e-07 | 1.1399e-02 | 1.54e-05 |
| limn shirt | 1.974e-07 | 2.6983e-03 | 7.32e-05 |
| limn toptee | 1.714e-07 | 7.5093e-03 | 2.28e-05 |
| encoder shirt val_control | 1.278e-06 | 0.75197 | 1.70e-06 |

Tức cờ `False` do những phần tử cỡ 1e-5…1e-6, trong khi `max_abs` nhỏ hơn `batch_max_abs=1e-4`
hai–ba bậc. Đây là lý do phải **giữ nguyên nhãn `False` cũ** và ghi nó là "fail theo chuẩn v1",
đồng thời **version hoá** chuẩn `abs(a-b) <= atol + rtol*|b|` trước khi đo mới — không được viết lại
lịch sử thành "pass". `parity_gate.py` đã có `STANDARD_ABS_REL_V2 = "parity-absolute-relative/v2"`
nhưng `evaluate_all()` mặc định vẫn `parity-relative/v1`.

### 4.3 Lỗi thật của R9 là **không có code nào nối verdict với artifact**

Ở `2e2e2d2`, trạng thái gate chỉ là văn xuôi (`GATES.csv` dòng 3–4, các `*.vi.md`); không aggregator,
không test nào đọc `g1_g2.*.json` để quyết định PASS/FAIL. Hệ quả đo được:

* `GATES.csv:4` ghi "TWO honestly-declared failures" trong khi artifact mang **4** cờ
  `False`/category (`raw/r9_encoder_false_flags.txt`).
* `GATES.csv:6` "LIMN dual-scope PASS" kiểm 12 record mà **không** kiểm `target_did` — nên bỏ sót
  việc 6/12 ô ghi DID của query kia (R5, `raw/guards/r5_reconcile.json`: `problems: 6`).
* Một cờ `False` không thể xuất hiện ở bất kỳ surface nào vì không surface nào đọc nó.

`workbench/backend/parity_gate.py` (ở `3828163`; untracked tại `2e2e2d2`) sửa đúng chỗ đó: verdict suy từ artifact, 6
dimension tách rời (`input/forward/batch/scoring/metrics/observer`), `passed` chỉ True khi không
mandatory nào fail, `None` không được coi là pass, và nó in `LABEL-DISAGREEMENT` khi boolean của
harness nói `True` còn số của chính nó vượt tolerance khai — 3 ca `population.gallery_vs_native`
của encoder. 21 test pass (`tests/test_gate_parity_subchecks.py`, chạy lại lượt này).

**Trạng thái phục vụ, đo lại sau khi commit:** `3828163` chứa **cả** module **cả** route
(`main.py:1044`); backend `:8000` khởi động lại **08:47:09** (pid 2820650) và
`GET /api/runs/parity-gate` trả **200** với `standard: parity-relative/v1`, `passed_models: []`,
`limn` 3 failure / `encoder` 13 failure — đúng bằng số của `evaluate_all()` import trực tiếp.
Bằng chứng: `web_evidence/after-parity-gate.json`, `web_evidence/1920-14-parity-gate.png`.
Ở `2e2e2d2` route **không tồn tại**; trong khoảng 08:16:44–08:29:09 nó từng trả 404 vì process cũ —
đó là trạng thái lịch sử, không phải trạng thái hiện tại.

Một điểm phải nói kèm: `evaluate_all()` chỉ chạy cho `EVIDENCED_MODELS = ("limn", "encoder")`; 13
model còn lại **không** xuất hiện trong `models`, nên `passed_models: []` là phát biểu về 2 model đã
đo, không phải về 15. `evaluate_model("hint")` trả `PARITY_NOT_RUN` đúng, nhưng `evaluate_all()`
không liệt kê nó.

## 5. Phase C — bốn phát hiện còn lại

### 5.1 R2 — bất đối xứng transform là thật, và nó ở phía **native**

`raw/r2_transform_tensors.json`, cùng ảnh 256×256 `dress/B0007WIZYE.jpg`:

| nhánh | max_abs | nghĩa |
|---|---:|---|
| `control_val_vs_val` | **0.0** (equal) | control hợp lệ |
| `preprocess_val` lặp lại | **0.0** (equal) | val transform tất định |
| `preprocess_train` lặp lại cùng ảnh | **3.8270** | train transform **ngẫu nhiên** |
| `train_vs_val_same_image` | **3.8270** | khác biệt thật, không phải nhiễu đo |

Category nào dùng transform nào — **đọc source**: `datasets.py:139,141,177,178` `stage=0` (=train) cho **ảnh query**; `:195,202` `stage=1` (=val) cho **gallery**; target của query trong nhánh val-split cũng đi `:178` stage=0. Adapter dùng `preprocess_val` cho **mọi** ảnh (`model_encoder.py:329,348`). Ba thứ tách đúng: native fidelity (train transform ngẫu nhiên, đo được), same-input control (val-vs-val = 0 tuyệt đối), CIRCUS adaptation (adapter chọn val — khác native ở phía query, và đó là nguồn khác biệt thật). Hệ quả: so sánh sample-level adapter-vs-native-train-transform **không thể** là parity test — đúng 2 trong 4 block `False` của encoder (§2.2).

### 5.2 R4 — `learnable_relation` không nằm trong checkpoint; và có 2 key lạ

`raw/r4_determinism_p1.json` / `_p2.json` / `r4_determinism.raw.log`, hai process sạch: checkpoint `fashioniq.pt` `0f31c501…9173` có **417** key, **0** key chứa `relation`; `state_dict()` **415** entry; `learnable_relation` **không** ở trong đó; kiểu attribute = `Tensor` (không phải `Parameter`) ⇒ không được đăng ký; `named_parameters()` **[]**; `requires_grad True` nhưng không vào optimizer; device `cuda:0`. **Unexpected keys** khi `load_state_dict`: `backbone.binding_decoder.fc.0.{bias,weight}` — `miss=0, unexpected=2`, và `strict=False` che mất. Hai construction **không seed**: max_abs **4.580 / 4.678 / 4.923** (ba lần đo); **cùng seed trước construction**: **0.0**; hai seed khác nhau: **4.862**.

Nguồn: `third_party/ENCODER/model_try2.py:93` — `.cuda()` trên `nn.Parameter` trả **Tensor thường**. Số không-seed **không phải hằng số** (trị "~4.2" của operator không tái lập). Chính sách khuyến nghị (**chưa áp dụng**): seed **trước** `Encoder(...)` (đủ, rẻ, `seeded_same_seed_max_abs = 0.0`), ghi seed vào run identity/cache; hoặc lưu tensor + sha256. **Không** dùng tensor `arange` làm "released weight". `seeded_dataset_construction` sẵn có **không** che được vì nó seed **sau** khi `Encoder(...)` đã rút xong.

### 5.3 R3 — tie probe dùng sai mask; và "hai ca thật là tie" bị **bác**

`raw/r3_tie_harness.json`, ca đồ chơi (reference 1.0, ảnh khác 0.8, target 0.5,
candidate 0.5000001): rank **không** mask **4**, **có** mask **3** (target #4 → #3); gap trong ma
trận không mask **1.192e-07** (≈ float32), trong ma trận có mask **99999997952.0** (= do `-10e10`);
vị trí thực sự vượt **chỉ `0`** (chính reference); bản 3 phần tử #3 → #2 cùng delta **1.192e-07**.
⇒ "gap đổi ~1e-7" và "rank đổi" là **hai phép đo khác mask**; mask cho rank phải là mask cho probe.
Cơ chế harness **CONFIRMED**.

Nhưng "hai ca thật là tie" bị **DISPROVED** cho ca đo được (`raw/r3_real_case.txt`, dress, query 3,
target position 2627): `preprocess_train` rank **28**, `preprocess_val` rank **31**, gap nhỏ nhất
**1.559e-3** / **3.639e-4** — lớn hơn epsilon 1559× / 364×. Sửa harness **không** cứu verdict "tie".
Phạm vi: chỉ **1/4** target của sample khớp chính xác một hàng gallery ⇒ **một** ca, không population.

### 5.4 R11 — BLAS: `encoder` **bị loại trừ**, `conesep` có near-tie thật, `limn` **0 ca**

`COMPOSITION_MATRIX.csv` cột `blas_pin`, mỗi dòng ghi phạm vi:

| model | threads | đường tính | A/B | suy ra |
|---|---|---|---|---|
| `encoder` | requested=1 / effective=1 (verified) | GEMM `queries @ gallery.T` | `changed=1`, `max_drank=1`, **r10_delta 0.0**, r50_delta 0.0, ndcg_delta 2.18e-05, `cross_k10/50/100 = 0` | BLAS **không** phải nguyên nhân chênh cũ→mới |
| `conesep` | 1/1 | `np.einsum(optimize=True)` → gemm | 17/18009 rank đổi **1** bậc, ndcg ~1e-7, R@10/R@50 **y hệt**, cùng interpreter | near-tie GEMM **thật**, không đổi metric |
| `limn` | 1/1 | `query @ gallery.T` theo block 8192 dòng | `changed=0`, `max_drank=0`, delta **0.0** | không có ca nào để kết luận |

A/B đầy đủ: `20261108T-throughput/d3b_encoder_aggregate_scan.txt` (cả ba condition đều `changed 1`,
`r10_delta 0.0`). `ptha_mtst` có 12/20/14 rank đổi 1–2 bậc — cùng lớp 1e-7 — nhưng **encode
UNPINNABLE** (`mkl_set_num_threads` segfault exit 139; chỉ `MKL_NUM_THREADS` có tác dụng, không set
được giữa process).

Ba cách suy luận **không** được dùng (theo đúng bài học): margin tới #1, cùng source hash, hay
R@10/R@50 ngược chiều — mỗi thứ **không đủ** để chỉ định nguyên nhân. Ngoài ba model này **không** có
phạm vi nào được tuyên bố.

### 5.5 R10 — LIMN: canonical giữ PARTIAL, nhưng con số canonical phụ thuộc một routing không đọc được

`limn_dual_scope.json`: `canonical_status.status = PARTIAL`, `scored_count = 6001`,
`unscored_count = 2`, `unscored_qids = ["7:3038","7:3790"]`,
`canonical_cohort.full.query_count = 6003`; `gallery.count = 74381`, note nguyên văn "the FULL
released gallery for every scope: no category filtering" ⇒ scope quyết định **embedding space**,
**không** quyết định gallery; `queries["7:3038"].release.loader_category = null`,
`matched_subsets = ["full"]` ⇒ unscored vì loader không gán được category, **không** vì rank kém;
`rank_convention_crosscheck`: `checked = 36`, `all_agree = true`. 12 ranking chẩn đoán:
`7:3038` shirt 3/88/152, toptee 43/389/67; `7:3790` shirt 65/6891/2200, toptee 43/4730/4047
(multimodal / text-only / image-only). Không scope nào được chọn canonical.

**Phát hiện tôi tự đo, chưa worker nào ghi.** Cùng model + checkpoint `base_iter0_all_categories`,
hai run trong `summary/` có **chính sách routing khác nhau**, cùng `query_count 6003`, cùng
`query_order_identity 8214a154…`:

| run | scored | unscored | `category_scope_failure_count` | R@10 | `query_scope_resolution` |
|---|---:|---:|---:|---:|---|
| `9e4918a6b5c2…` (canonical, 11-09) | 6001 | **2** | 2 | 30.0283 | có: `3a1c5a67…` |
| `10173dc45dea…` (11-08) | 5884 | **119** | 119 | 30.0646 | **vắng mặt** |

Tập unscored của `10173dc4` **đúng bằng** 119 `ambiguous_qids` của `circus_cohorts.json`; của
`9e4918a6` chỉ còn 2 ⇒ **117 query mà cohort khai là mơ hồ đã được chấm**, mỗi query buộc phải chọn
**một** trong ba embedding space, và bản ghi duy nhất của lựa chọn đó là một **digest**, không phải
mapping đọc được. R10 cấm chọn scope theo rank/score; báo cáo này **không cáo buộc** điều đó, nhưng
**không kiểm được** — nên 30.0283 phải đọc kèm câu đó. Run `10173dc4` có routing hiện rõ và trung
thực (119 unscored, mẫu số 5884). Cả hai row đều được phục vụ cạnh nhau.

### 5.6 R7 — subset **không** trộn thế hệ; nhưng "thế hệ" có thật ở tầng run

`web_evidence/r7-api-counters.json` (17 run × 3 subset) + `r7-subset-provenance.txt` (35 row, đọc từ chính `multimodal.jsonl`): **không** run nào có `sf_total > full_total` hay `sf_unscored > full_unscored`; `sf ⊂ full`, `validated ⊂ full`, `validated ⊂ sf` đều True; R@10 phục dựng khớp số phục vụ tới **1e-9** trên mọi row có metric (35/35; `e0dd04a46ef7` FAILED có `scored=0`). ⇒ **NOT_REPRODUCED** cho "trộn thế hệ" và cho "metric không phục dựng được".

Cái **CONFIRMED** là thiếu **khai báo**: trước lượt này không row nào nêu `source_run_id` / membership hash / mẫu số. `backend/subset_provenance.py` (ở `3828163`) nay trả `own` (path, `membership_sha256`, counts), `containment`, `metric_check` (served vs recomputed, tri-state), `consistent`, `reasons[]`; 0.29 s/35 row cold, 0.20 s warm. **Số phục vụ không đổi** — `metric_check` là đối chứng, không phải nguồn; `ndcg_full` cố ý **không** phục dựng.

## 6. Thành phần CIRCUS theo subset

Nguồn: `/api/composition/runs` (35 row) + `RUN_MATRIX.csv`. **R@10/R@50 multimodal trên cohort CIRCUS 74381**,
**không** phải metric native; **không** so với §3.

| model | checkpoint | full (n/scored → R@10 / R@50) | sf (4069) | validated (586) |
|---|---|---|---|---|
| `habit` | `fiq_n02` | 6003/6003 → **35.616 / 54.373** | 20.521 / 39.543 | 34.130 / 59.898 |
| `conesep` | `fiq_n02` | 6003/6003 → **35.482 / 54.140** | 20.128 / 39.690 | 34.471 / 59.044 |
| `intent` | `fiq_n02` | 6003/6003 → **34.399 / 54.006** | 18.801 / 38.658 | 33.788 / 56.485 |
| `tme` | `fashioniq_0` | 6003/6003 → **34.016 / 53.956** | 19.022 / 39.346 | 33.959 / 60.410 |
| `hint` | `fashioniq` | 6003/6003 → **32.817 / 51.774** | 18.358 / 37.356 | 29.352 / 53.925 |
| `encoder` | `fashioniq` | 6003/6003 → **31.034 / 49.475** (`5dee7425`)** / 31.018 / 49.575 (`6a528e87`) | 18.678 / 36.323 (chỉ `6a528e87`) | 34.642 / 55.290 (chỉ `6a528e87`) |
| `limn` | `base_iter0_all_categories` | 6003/**6001** → **30.028 / 48.675** (`9e4918a6`) · 6003/5884 → 30.065 / 48.759 (`10173dc4`) | 18.600 / 36.225 (chỉ `10173dc4`) | 30.293 / 53.701 (chỉ `10173dc4`) |
| `ptha_mtst` | `cirr_transfer` | 6003/6003 → **19.474 / 34.449** — **TRANSFER CIRR→FashionIQ**, không phải tái lập | 7.078 / 18.899 | 12.969 / 30.375 |

**Hình dạng:** 35 row = **17 full + 9 sf + 9 validated**; `full` chỉ có **8 cặp** model+checkpoint ⇒
mỗi cặp 1–3 lần chạy. `duplicate_status` phục vụ cho cả 17 row full (`unique` 5, `undetermined` 12).

**Bốn nhóm trùng, xác định bằng số** (`PHASE_B.vi.md`), cùng model + checkpoint + subset +
`query_count` + `gap` trùng tới chữ số cuối: `habit 497c25c5 ≡ c6eb9d57` (gap `0.388704727214254`),
`hint 4bc9893a ≡ f461f391`, `intent 37ee72aa ≡ 4901da13`, `tme ad477312 ≡ cd1a696d`. **Không** trùng:
ba run `encoder` (gap 0.46986… / 0.47230… / 0.53293… q=8 smoke) — khác `resume_key`, cùng
`checkpoint_identity 0f31c50150a7…`.

**Hai "thế hệ" không phải cùng một phép tính:** `encoder` `6a528e87` → `5dee7425` khác
`adapter_source_hashes.rank` (`5f828b13…` → `f0914b12…`) trong khi `score` **giống hệt**
(`0ab4fe1d…`), cộng interpreter đổi (`pair-adapted` 3.11.15 → `encoder310-adapted` 3.10.20). Không
metric nào bị sửa; chỉ attribution.

**Ngữ nghĩa subset (đã kiểm):** `sf` = shortcut-free, `validated` = validated_solved ∪
validated_unsolved (368 + 218 = 586, `raw/circus_cohorts.json`); cả hai là **tập con của `full`**
trên cùng gallery 74381; gallery **không** lọc category (32/74381 ảnh là mode `L`).

## 7. Bề mặt web

### 7.1 URL và trạng thái sau sửa

`http://127.0.0.1:5173/` → `?area=overview&protocol=circus_fashioniq_release_v1&subset=full`
(`web_evidence/after-shots.log` dòng 0). Trước sửa:
`?area=overview&protocol=fashioniq_original_split&subset=sf` với `subset_select: 0` và "0 lượt có dữ
liệu ở phạm vi này" (`before-bare-url-probe.json`).

Đã landing (`3828163`): `scope.ts`, `routes.ts` (`scope: {protocolNamed, subsetNamed}`), `App.tsx`
(điền **nửa scope chưa nêu**, `replaceState`, control subset chỉ render khi `protocolHasSubsets()`),
`ModelInventory.tsx` (`otherScopeFor`), 12 test `test_web_scope_and_labels.py`; 51 test frontend pass
(`after-tests.log`).

**Đính chính `WEB_SURFACE_AUDIT.vi.md` §4.1 mục 1**, vốn ghi "`ModelInventory` chưa dùng
`otherScopeFor`": ở `3828163` điều đó **không còn đúng** — `ModelInventory.tsx:8` import nó, `:137`
dùng nó và sinh nhãn "Có kết quả ở scope khác"; bundle phục vụ chứa chuỗi đó. Mục này đóng.

Bốn thứ **còn** hở ở §6 display contract: chip trong explorer; `SamplesPage` dùng select thay vì chip;
"Xem mẫu" cho mọi run có dữ liệu; hạng per-positive cho ảnh đích (dữ liệu không có — không bịa);
clamp K + MM/Text/Image cùng hàng gallery.

### 7.2 W1 — nguyên nhân, và banner đã đóng

Ba nguyên nhân, mỗi cái có dòng code:

1. `App.tsx` effect "First load" chọn `data.protocolIds[0]` (**thứ tự chèn**) và **giữ** subset mặc
   định `sf` của protocol trước ⇒ protocol một-cohort nhận `sf` không tồn tại ⇒ 0/0/0.
2. Nhãn subset in **vô điều kiện** ở sidebar và `scope-strip` ⇒ "Subset Shortcut-Free (SF)" đứng cạnh
   một protocol chỉ có một cohort.
3. `ModelInventory` chỉ tra `catalog` đã lọc theo `protocol + subset` ⇒ model có row ở scope khác rơi
   vào nhánh "Chưa có số đo" mà không nói nó **có** kết quả ở nơi khác.

**Đối chiếu ID: không ID nào lệch.** `/api/models` 17 (khớp registry, 2 deferred); `/api/runs` `[]`
(DuckDB index rỗng — `WORKBENCH_RESULTS_ROOT` rỗng); `/api/benchmark-runs` 13 report **tất cả**
`fashioniq_original_split` (2 certified, 11 refused); `/api/composition/runs` 35 row **tất cả**
`circus_fashioniq_release_v1` (full 17 / sf 9 / validated 9); `capabilities?subset=sf` khớp 9. Hai
nguồn đo **hai tập rời nhau**; defect là **landing scope**, không phải số.

**Điều audit không nêu:** `after00.txt:37` in `MOCK DEVELOPMENT DATA — no real evaluation results are
loaded` **ngay trên** một trang có 17 dòng CIRCUS thật. Nguyên nhân: `evidenceState()` tính
`aggregateRuns`/`experimentRuns` từ `region.data ?? []` ⇒ `[]` khi request còn bay ⇒
`noRealEvidence = true` ⇒ `evidenceBanner()` trả nhánh MOCK; nó **không có nhánh loading**.

**Đã sửa ở `3828163`:** `evidenceBanner` có nhánh loading riêng, chứng minh theo kiểu failing-first —
6 failure với `evidence.ts` cũ, **8 pass** sau khi sửa
(`test_web_evidence_banner_loading.py::test_a_loading_source_is_never_reported_as_no_evidence`).
Trace trên service (`web_evidence/after-banner-loading.txt`):

```
t=0 loading=true   Đang tải bằng chứng | Ẩn
t=1 loading=true   Đang tải bằng chứng · 2 đã chứng nhận trong 13 report đã về | Ẩn
t=3 loading=None   AGGREGATE-ONLY REAL EVALUATION — 2 certified research results … | Ẩn
false-claim-seen: False
```

Tức một trang đang tải **không còn** bị đọc thành "không có bằng chứng thật" — cùng lớp lỗi mà
§3bis.a của audit đã sửa cho `scope-strip`, nay đóng cho cả banner. Xem thêm
`web_evidence/after-release-chain-banner-contract.txt` (`aggregate-only: True`, banner đúng hợp đồng).

### 7.3 R13 — nội dung finding đúng một nửa; và catalog tự nó đếm thiếu

Finding nói "native run có raw artifact nhưng thiếu trên web". Đo lại: nhóm artifact "có raw nhưng
không lên web" **không phải** run bị thiếu — **34 file** measurement native nằm ngoài mọi đường
ingest (`web_evidence/r13-native-catalog.txt`). Trước lượt này chúng không có endpoint nào, nên web
đọc 0 số cho `fashioniq_val_split` và `fashioniq_full_gallery_ref_excluded` dù 15 file có số.

Đã thêm `GET /api/native-measurements` (`workbench/backend/native_measurements.py`, mới) đúng như
remedy: mỗi row mang `source.path` + `source.sha256` + `bytes`, `verified: false`,
`verification.state = "UNVERIFIED_NATIVE_MEASUREMENT"` + lý do nguyên văn. **Không** tạo
`completion.json`/`invocation_outcome.json` (cấm tạo receipt giả), **không** chèn row vào `reports/`,
**không** cộng vào `/api/evidence/summary` — `counts.verified` trả **0** tường minh.

**Đo ở `2e2e2d2`** (bản reader đầu tiên): `{ "files": 21, "with_metrics": 15, "unreadable": 1,
"signposted_blockers": 4, "verified": 0 }`, 12 `model_id` có số. Bốn model active có macro mà catalog
**không** phục vụ: `ptha_mtst` (shape `macro_metrics`, hiện diện nhưng `metrics: null`),
`combiner_rn50x4_noft` + `clip4cir_rn50x4_fullft` (nested trong `clip4cir_parsed.json`, bị
`_is_measurement()` loại), `pair` (`pair_compat/measurements/` ngoài glob). `csmcir` ngoài glob.

⇒ Ở `2e2e2d2`, R13 chỉ **một phần**: đường ingest đúng và trung thực, nhưng nó là **bản sao lỗi
census v1** — chỉ đọc `aggregate.{macro_r10,macro_r50}` và chỉ hai glob.

**Đã sửa ở `3828163`** (`web_evidence/after-native-measurements.json`): reader nhận cả năm hình dạng
(`aggregate.*`, `macro.*`, `macro_metrics`, top-level `macro_r10`, `<variant>.macro_r10`); row tổng hợp
mang `rollup: true` và bị loại khỏi `counts.with_metrics`. Bảng đối chiếu `2e2e2d2` → `3828163` ở §3.5.

Năm model từng bị bỏ nay đều có mặt với `verified: false`: `ptha_mtst 37.897844115893044`,
`clip4cir_rn50x4_fullft 43.8182`, `combiner_rn50x4_noft 35.1407`, `pair 34.03503453772374`,
`csmcir 57.05073873202006` — hai giá trị `ptha_mtst`/`csmcir` đã đối chiếu lại với file nguồn tới
1e-12. Test: `workbench/tests/test_native_measurements_census.py` (16 test). Các bất biến của remedy
**không đổi**: không receipt giả, không chèn `reports/`, `counts.verified` = **0**.

### 7.4 R6 / R8 — backend đã sửa, kiểm lại trên instance sống

Process `:8000`: `depth=100` → 200 với `depth_requested/served/stored = 100/100/100`; candidate có
`image_url` (resolve HTTP 200 `image/png`); `target` tách riêng (`rank 6891`, `in_window false`,
`delta_to_rank_1 0.3837`); `depth=0` → **404** với lý do nguyên văn ("a zero or negative depth cannot
return a window and silently returning one row would misstate the request"). Ở `2e2e2d2`,
`depth=0` trả 200 với **1 dòng**, và export JSONL hỏng gây **500**. Bundle phục vụ không còn
`depth=20`. Quy tắc từ chối ảnh query staged **giữ nguyên** (`STAGED_PREFIX`,
`limn_diagnostic.py:34-36`) — hiện ảnh đích hợp lệ, hiện ảnh query đen thì không. `complete: false`
trên artifact thật là 6 defect R5 còn lại, không phải lỗi mới. Chi tiết: `FINDINGS_AND_FIXES.md`
§3.3b–3.3c.

### 7.5 R5 — chỉ một field sai, và nó đã được vá

`raw/guards/r5_reconcile.json`: 12 ô (2 query × 2 scope × 3 condition), `problems: 6` — **cả 6** đều
`did_matches_own_query` ở `7:3038`. Chín kiểm tra còn lại đạt **12/12**:
`asset_matches_own_positive`, `target_flag_consistent_with_window`, `rank_summary_eq_export`,
`score_summary_eq_export`, `position_summary_eq_export`, `checkpoint_summary_eq_export`,
`no_duplicate_candidate_ids`, `export_depth_is_candidate_count`, `at_most_one_target_row`.

| | `7:3038` | `7:3790` |
|---|---|---|
| positive id **của chính query** | `7:3416` (`B004D8QCN4`) | `7:42304` (`B00E3WVAD2`) |
| `target_did` **đã ghi** ở cả 6 ô | `7:42304` ❌ | `7:42304` ✅ |

Nguyên nhân ở producer: `t2_dual_scope.py:273` `"target_did": query.positive_ids[0]`, với `query` còn
nguyên giá trị lần lặp cuối vòng ngoài (`7:3790`). 6/12 ô có target **ngoài** Top-100 (152, 389,
2200, 4047, 4730, 6891) nên **không** candidate nào mang `is_target` — đúng thiết kế, không phải
thiếu sót.

Sửa hai lớp, cả hai **không** rescore: (a) tầng phục vụ lấy `target_did` từ record của chính qid, báo
defect `target_did_not_own_query` nếu lệch; (b) sinh **bản sao** artifact (6 record `7:42304`→`7:3416`,
không đổi rank/score/position/candidate), file gốc sha256 `04c4419e…`, `"source_untouched": true`.
Verify: đặt bản sửa vào repo root tạm → `DIAGNOSTIC_COMPLETE`, `complete: true`, **0 defect**.

**Đã đóng ở `68ad75e`** (§10.2): `t2_dual_scope.py` lấy DID theo qid **tại thời điểm serialize**
(`cell_target_did`) và có `assert_cells_target_own_query()` chạy trên **cả 12 ô trước khi ghi file**,
nên regenerate bằng script cũ không còn tái sinh lỗi — nếu một ô sai, script **từ chối ghi**.
Artifact trên đĩa đã vá: đúng **6** ô `7:42304 → 7:3416`, leaf diff đệ quy **62** = 6 ô + 56 field
của `provenance_correction_r5`, **0 khác**; bản gốc giữ ở `limn_dual_scope.json.pre-r5`
(`04c4419e…abf0`), bản vá `d1b4aadb…73e3f`. Rank / score / gallery position / asset / export / model
text **không đổi một byte**. `test_limn_dual_scope_producer.py` = 8 test; serve layer trả
`DIAGNOSTIC_COMPLETE`, `complete: true`, `defects: 0`.

Điều **không** được tuyên bố: artifact được **vá**, không được **sinh lại** — không có lần suy luận
mới nào. Đó là lựa chọn đúng vì ranking đã được chứng minh khớp export + release (6/6 ô trong cửa sổ,
36/36 crosscheck) trước khi vá, nên chạy lại chỉ để đổi một nhãn là lãng phí và tạo rủi ro mới.

## 8. Blocker còn lại, kèm lý do đo được

### 8.1 Bốn model không chạy được native

| model | blocker đo được (nguồn: `BLOCKERS_R12.vi.md`, `raw/r12/*`) | verdict |
|---|---|---|
| `airknow` | interpreter **mất** (symlink gãy → `teacher-sprc` đã xoá; "3.11.15 / torch 2.7.1" của BC-cũ **không tái lập**); **11** biểu thức `.cuda()` cứng trên scoring path. Lỗi import đầu tiên là `torchaudio` (`audio_processors.py:9`), **không** phải `moviepy` (dòng 11, ngoài inference path); `sklearn` (`blip2_cir.py:19`) chặn class. `clavis` là **false positive grep nhị phân** (`0x63`=`c`=GLOBAL). Checkpoint load **được** CPU: 863 tensor / 1 196 753 657 param | **BLOCKED** |
| `clvc_net` | không có env `clvc*`; **14 dòng / 15 biểu thức** `.cuda()` toàn tree, **6 dòng / 7** trên scoring path; không `cuda.is_available()` ⇒ chết ở `test.py:28`; `test.py` 0 `__main__`. Device-shim **trung thực về số** (eval không `autocast`/`.half()`, fp32 matmul) nhưng **không trung thực về provenance** nếu không dán nhãn "adapted runtime" | **BLOCKED** |
| `tgcir` | không env; `ilearn39` thiếu `clip`. Tác giả **không** phát hành script eval: `test.py` 162 dòng, **0** `__main__`/argparse/`torch.load`; eval nằm trong training loop (`train.py:200`, `:242`); `--restore_file` khai `:74` **không bao giờ dùng** (`grep` = 1 khớp, chính dòng khai) | **BLOCKED / upstream-only** |
| `pair` | interpreter **mất**; pickle cần 38 GLOBAL op gồm `model_vssp.Encoder/Backbone/FeatureWiseAffine` + `MHTransformer_dis.*`, **không** có trong repo pin; chỉ load qua alias loader của workbench ⇒ `EXPLORATORY_PUBLIC_MAPPING`. 455 state-dict (456 = +1 buffer); dưới mapping giả thuyết: 381 key, **74 unexpected**, **70 shape mismatch**, **5.2001 %** param không qua forward. Không reconstruct được: repo có **hai** biến thể `MHTransformer{,_rev}.py` mà pickle không nói biến thể nào | **BLOCKED / upstream-only** |

Hai claim BC-cũ bị sửa: "repo không có entrypoint" **sai một phần** cho `tgcir`; "`model.py:41
clip.load(device='cuda')`" **không tái lập** (quan sát được `train.py:88`).

### 8.2 `combiner_rn50x4_noft` — không blocker chạy, nhưng vĩnh viễn không-phải-paper-evidence

Run hoàn tất thật: `EXECUTION_COMPLETED`, `INVOCATION_COMPLETED`, `return_code 0`,
`PROOF_VALIDATED`, `EXTRACTED_VERIFIED`, metric 35.1078 / 58.4756 / mean 46.7917. Web hiện **REFUSED**
vì gate `METRIC_REFERENCE_UNVERIFIED` (`paper_gate.py:333-335`), `blockers: ["no committed reference
describes this experiment"]`, và `unbenchmarked('combiner_rn50x4_noft') = True`
(`benchmark_references.py:207-225`) — **không dòng paper nào** mô tả cấu hình noft RN50x4. Đó là
thuộc tính thường trực của những gì tác giả công bố, không phải lỗi run. Đóng đúng cách **không
phải** nới gate: hoặc chấp nhận vĩnh viễn nhãn "no publishable comparison" và dùng số này nội bộ,
hoặc commit một row `VERIFIED` từ nguồn tác giả khác — quyết định của operator.

### 8.3 Các ô `NOT_RUN` phải đọc là trống, không phải "pass"

`RUN_MATRIX.csv` (61 dòng = 50 composition + 11 native), bảy cột surface tách rời: execution (37 `COMPLETED`, 9 `EXECUTION_COMPLETED`, 4 `PARTIAL`, 3 `FAILED`, 6 fixture, 1 `UNREADABLE_JSON`, 1 `NO_AGGREGATE`); validation (**41** `NOT_RUN` — không receipt nào trong `artifacts/receipts/`, 9 `NOT_RUN` native `native_verified=N`, 6 `N/A fixture`, 3 `NOT_APPLICABLE`, **1** `PARITY_VERIFIED` = `limn_native`); export (12 `EXPORTED`, 8 `COVERED BY RUN EXPORT`, 21 `NOT_EXPORTED`, 19 `N/A`); serving (41 `RANKS_PRESENT`, 6 `NO_RANKS`, 3 `IMPORTED_ARTIFACT`); rerun (12 `KEEP`, 17 `EXPORT ONLY`, 9 `KEEP unless…`, 8 `DO_NOT_RERUN`, 6 `NEVER`, 4 `RESUME`, 3 `RERUN`); cache (`resume_key` + `gallery_identity_sha256 1db68c59…` trên mọi run thật). Một dòng **hỏng**: `native:…/raw/native/ptha_mtst.json` = `UNREADABLE_JSON` (bản đọc được là `ptha_mtst.result.json`) — không dùng làm bằng chứng.

### 8.4 Trạng thái dịch vụ — đo lại sau khi commit

| thứ | giá trị đo được (`3828163`) | hệ quả |
|---|---|---|
| backend `:8000` | pid 2820650, khởi động **08:47:09** (sau `main.py` 08:29) | `/api/runs/parity-gate` → **200**, `passed_models: []`; `/api/native-measurements` → 200 |
| backend `:8765` | pid 1227919, khởi động **09-11 01:02** | process **cũ**, không có các route mới — không rõ chủ, **không** dùng để đo |
| frontend `:5173` | phục vụ `dist/assets/index-DjAje2d2.js` (08:54, 367 327 byte) | mới hơn backend ⇒ UI gọi endpoint backend thiếu |
| bundle trong ảnh chụp | `index-CG4hbuvP.js` (08:16) và `index-C35K848A.js` (08:33) là các bản **trước** | ảnh `after00/after01` chụp tổ hợp build **cũ hơn** bundle đang phục vụ |

Hai điều phải nói, và chúng khác nhau:

* **Lịch sử:** 08:16:44–08:29:09 backend cũ phục vụ `main.py` cũ nên route gate **404** — lý do §7 ghi
  mốc cho từng ảnh; không phải trạng thái hiện tại.
* **Hiện tại:** gate và catalog native **đều 200** trên `:8000`; số trong
  `after-parity-gate.json` / `after-native-measurements.json` đến từ process này.

Ảnh chụp `after00/after01` (08:17–08:18) vẫn là ảnh thật của một tổ hợp bundle **cũ hơn** bản đang
phục vụ. Chúng chứng minh landing đã đổi sang CIRCUS Full, **không** chứng minh banner/native catalog
đã sửa — hai thứ đó có bằng chứng riêng (`after-banner-loading.txt`,
`after-native-measurements.json`) ghi đúng mốc sau commit.

## 9. Cái gì **không** được tuyên bố

1. **Không** 15/15 native: **14/15** theo nghĩa "có số trong log/command+meta khớp measurement";
   **13/15** nếu đòi đúng `EXITCODE=0` (limn không có exit code, log kết thúc bằng traceback). `pair`
   không có gì.
2. **Không** model nào đạt G1/G2 PASS. `passed_models = []`. Ba cách đếm đã lưu hành ("2", "13",
   "1") đều sai — dùng **5** cho encoder, **3** cho limn, và luôn nói **0** đổi R@10/R@50.
3. **Không** metric native nào là tái lập paper: chạy lại trên dữ liệu thật với runtime adapted
   (`sm_120`, torch 2.7.1), trừ `combiner`/`clip4cir`/`csmcir`/`sprc` dùng stack cũ hơn.
   `ptha_mtst` là **transfer**, `pair` là **exploratory**.
4. **Không** xếp 15 model native thành leaderboard: bốn protocol khác nhau; `dcnet` báo **phân số**
   trong `raw/native/dcnet.json` và **phần trăm** trong `$D1/measurements/dcnet_run1.json`.
5. **Không** dùng metric CIRCUS để nói về native, và ngược lại.
6. **Không** sửa registry. Đề nghị cho người sở hữu: `tests/fixtures/pc_launcher/models.json:1722-1725`
   còn khai `fashioniq_val_split_ref_included` cho `tme` trong khi `models.yaml:1669` nói
   `fashioniq_original_split`; alias **resolve** trong evaluator (`src/evaluation/fashioniq.py:186`)
   nhưng **bị từ chối** cho artifact đã lưu (`schemas/results.py:12-16,136`). Diff: `HANDBACK.vi.md` §1.
7. **Không** chạy full test suite ở `d850f98`. Sweep đầy đủ chạy ở `68ad75e` — **18 failed / 2701
   passed / 47 skipped**, gap accounting **0**, diff so baseline **0 new / 0 dropped** (§10.1).
   Commit `d850f98` là **docs-only** (4 file, chỉ `TESTED_SOURCE_SHA` + chuỗi đếm test), nên theo quy
   ước release guard nó **không** làm luật sweep cũ. Ngoài sweep: **105** test batch campaign, **41**
   frontend, **28** focused sau khi sửa R5/R1, guard **8 passed**. Không chạy lại cohort, không job GPU.
8. **Không** nói `9e4918a6` (LIMN canonical) là số duy nhất đúng: nó phụ thuộc một routing policy chỉ
   tồn tại dưới dạng digest (§5.5); run `10173dc4` có routing hiện rõ nhưng mẫu số 5884 và R@10 30.0646.
9. **Không** nói sub-check FAIL đã đóng. Không tolerance nào bị dời; nhãn `False` của
   `parity-relative/v1` được **giữ nguyên** và phải giữ, kể cả sau khi có chuẩn v2.
10. **Không** nói provenance đã sạch: 5 file lệch `file_manifest` là `approved_source_patches` (khớp
    `derived_sha256`) — kiểm chỉ bằng `file_manifest` sẽ báo sai 5 "local modification".
11. **Không** nói R1 đã được đưa vào endpoint sống: migration **đã** sửa được câu trích sai (xem
    §10.2), nhưng bản đã sửa mới nằm ở `raw/guards/clip4cir_fullft.migrated.r1.json`; **chưa**
    re-ingest vào service đang chạy, và artifact gốc `raw/native/clip4cir_fullft.json` giữ nguyên
    bytes (sha256 `fb97cfab…8e`) để bảo toàn lịch sử.
12. **Không** nói R5 đã được sinh lại: `t2_dual_scope.py` **đã** sửa và `limn_dual_scope.json` **đã**
    hết `7:42304` ở ô của `7:3038` (xem §10.2), nhưng artifact được **vá 6 ô**, không phải chạy lại
    producer. Không có lần suy luận mới nào; rank/score/asset của cả 12 ô là bytes cũ, và đó là điều
    đúng vì chúng đã được chứng minh khớp export + release trước khi vá.
13. **Không** nói release đã sẵn sàng: `TESTED_SOURCE_SHA` **đã** nâng lên `68ad75e` cùng một lượt
    sweep đầy đủ (xem §10.1), guard **8 passed**; nhưng đó là sweep ở `68ad75e`, và mọi prose trong
    báo cáo này viết ở `d850f98` — chênh lệch đó là commit **docs-only**, đúng loại được miễn sweep.

## 10. Commit, sweep và release guard — trạng thái đóng

Ba commit, **không** phải bởi phiên viết báo cáo này:

| # | commit | nội dung | push |
|---|---|---|---|
| 1 | `3828163` | `Close the E2E repair campaign: gate, protocol labels, LIMN guards, web scope` — 182 file, `21378 insertions(+), 177 deletions(-)` | `2e2e2d2..3828163` |
| 2 | `68ad75e` | `Fix the R5 producer's target DID and allow cited protocol_source correction` — 20 file | `3828163..68ad75e` |
| 3 | `d850f98` | `Advance TESTED_SOURCE_SHA to 68ad75e with the serial sweep's own numbers` — docs-only, 4 file | `68ad75e..d850f98` |

**HEAD cuối = `d850f9811f7670fa7c0485cd6618ec59dd662ae3` = `origin/research/cir-failure-workbench`.**
Không còn đường nào khác biệt; chỉ `CLEANUP_REPORT.md.save` (file của operator) là untracked.

Bốn thứ báo cáo này từng ghi là "còn mở" đã đóng ở `3828163`: route gate phục vụ được (§4.3), banner
có nhánh loading (§7.2), catalog native nhận 5 hình dạng macro (§3.5, §7.3), `ModelInventory` dùng
`otherScopeFor` (§7.1). Hai thứ nữa đóng ở `68ad75e`, ghi ở §10.2.

### 10.1 Sweep ở `68ad75e` và release guard **xanh**

Điều kiện trước khi đo, cả ba đều đạt: `git status --porcelain` chỉ còn `?? CLEANUP_REPORT.md.save`
(không path production/docs/test nào chưa commit) · `pgrep -af pytest` = 0 khớp · `df` = **153 GiB**
trống (ngưỡng 45 GiB).

```
18 failed, 2701 passed, 47 skipped, 290 warnings in 2846.09s (0:47:26)
```

| kiểm | giá trị |
|---|---|
| accounting | `2701 + 18 + 47 = 2766` == `2766` thu thập riêng bằng `--collect-only -q` ⇒ **gap 0** |
| dòng `E` | **0** |
| ENOSPC | **0** (153 GiB lúc bắt đầu, mẫu giữa lượt 153 GiB, 123 GiB sau) |
| cụt log | **không** |
| cây đổi trong lúc chạy | **không** |
| diff so `full_suite_36ae9e2.txt` | **0 new / 0 dropped / 18 common** |

Log: `full_suite_68ad75e.txt`; diff: `full_suite_68ad75e.failure_diff.txt`. Guard sau khi nâng:
`test_handoff_commands.py` → **8 passed**. Chuỗi `**2701 passed, 18 failed**` xuất hiện **đúng một
lần** trong mỗi file và **giống nhau** giữa `FINAL_PC_RELEASE_VERDICT.md` và `PC_PILOT_HANDOFF.md`.

**Vì sao phải bỏ sweep trước đó ở `3828163`:** lượt đó chạy 09:01–09:51 trong khi session khác còn
sửa `migrate_protocol_label.py` (mtime 09:07:51), `test_limn_dual_scope_producer.py` (09:07:43) và
`test_protocol_label_migration.py` (09:08:10). Bằng chứng không thể chối: nó đếm `2690+18+47 = 2755`
trong khi cây lúc đó thu thập **2766** — hụt đúng **11** = 8 test file mới + 3 test thêm. Một lượt đo
trên cây đang đổi thì không đo revision nào cả. Nó **không** được ghi là pass.

### 10.2 Hai lỗi còn lại đã sửa ở `68ad75e`

**R5 — producer dùng biến vòng lặp.** `t2_dual_scope.py` có `cell_target_did(qid, by_qid)` lấy DID
theo qid **tại thời điểm serialize**, cộng `assert_cells_target_own_query()` chạy trên **cả 12 ô**
trước khi ghi file, nên một lần merge/copy tái đưa giá trị dùng chung vào sẽ **từ chối ghi**.
`test_limn_dual_scope_producer.py` = **8 test**. Artifact vá bằng script có ghi digest
(`r5_patch_artifact.py`, `--dry-run` trước):

| | |
|---|---|
| ô đổi | **6**, tất cả của `7:3038`, `7:42304 → 7:3416` |
| leaf diff đệ quy | **62** leaf = 6 ô + 56 field của `provenance_correction_r5`, **0 khác** |
| rank / score / gallery position / asset / export / model text | **không đổi một byte** |
| bản gốc | `limn_dual_scope.json.pre-r5`, sha256 `04c4419e…abf0` |
| bản vá | sha256 `d1b4aadb6735b8b3c57e9fbfe234b1e12c15a13b977ace23500916af94473e3f` |
| serve layer | `DIAGNOSTIC_COMPLETE`, `complete: true`, `defects: 0` |

Một lỗi do chính bản vá đầu gây ra và đã sửa: nó thêm `target_did` top-level vào block `7:3038` mà
không thêm vào `7:3790`, phá tính đối xứng và vượt tuyên bố "chỉ 6 ô". Đã bỏ; hai block nay cùng tập
khoá `qid, release, resolved_query_image_path, scopes, target_gallery_id, target_position_in_gallery`.

**R1 — migration đóng băng câu trích sai.** `protocol_source` ra khỏi `FROZEN_FIELDS`; thêm
`CORRECTABLE_FIELDS` + đường hiệu chỉnh **bắt buộc có citation**, ghi bản cũ nguyên văn vào
`superseded_protocol_source` và một block `protocol_source_correction`. Chạy lại trên **bản sao**
(`raw/guards/clip4cir_fullft.migrated.r1.json`), artifact gốc không chạm: **11/11 check pass**, nhãn
thành `fashioniq_original_split`. Câu trích mới nói full-split gallery không mask và chỉ ra
`validate.py:162-166` là nhánh CIRR. Khi **không** đưa correction thì câu trích vẫn **không** bị sửa
— im lặng không đổi gì.

### 10.3 Defect có sẵn, cố ý **không** sửa

`workbench/tests/test_handoff_commands.py` cụt ở **703 dòng** trong cả `2e2e2d2`, `36ae9e2` và
`3828163`. Hệ quả: `test_the_verdict_names_the_same_test_count_the_handoff_does` dừng sau
`assert counts` và **không** có phép so chéo hai tài liệu, nên verdict và handoff có thể ghi hai con
số khác nhau mà không ai bắt được.

Không sửa trong chiến dịch này vì sửa nó biến commit docs-only `d850f98` thành thay đổi production,
làm vô hiệu lượt sweep vừa chạy. Nó cần một lượt riêng + một sweep mới.

### Phụ lục — lệnh chạy lại

```bash
cd /home/heheboiz/data/cir
PYTHONPATH=$PWD workbench/envs/ctl312h/bin/python -m pytest \
  workbench/tests/test_protocol_label_migration.py workbench/tests/test_gate_parity_subchecks.py -q
G=workbench/docs/reviews/20261110-e2e-repair/raw/guards
workbench/envs/ctl312h/bin/python $G/r5_reconcile.py
workbench/envs/ctl312h/bin/python $G/r5_correct_summary.py
workbench/envs/ctl312h/bin/python $G/r6_probe.py          # cần service
S=workbench/docs/reviews/20261110-e2e-repair/scripts
PYTHONPATH=$PWD workbench/envs/encoder310-adapted/bin/python $S/pc_r2_transform_tensors.py
PYTHONPATH=$PWD workbench/envs/ctl312h/bin/python             $S/pc_r3_tie_harness.py
PYTHONPATH=$PWD workbench/envs/encoder310-adapted/bin/python $S/pc_r4_relation.py --seed 0 --repeat 2
```
