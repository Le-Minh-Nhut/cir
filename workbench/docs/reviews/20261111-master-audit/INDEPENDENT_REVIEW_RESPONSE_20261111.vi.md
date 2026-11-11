# Phản hồi kiểm tra độc lập `CIR_INDEPENDENT_REVIEW_06e30e8c` — 2026-11-11

> Đối tượng: `~/Downloads/CIR_INDEPENDENT_REVIEW_06e30e8c.vi.md` (154 dòng, mốc source `06e30e8c`).
> Trả lời tại HEAD `cd980dcecbb685b3cd6be4516d6550660a8e89b8`, nhánh `research/cir-failure-workbench`.
> Cấu trúc mỗi mục: **Rxx — phát hiện → đã sửa gì → tái hiện trước → sau → commit**.
> Ba kết luận dùng được: **Đã sửa**, **Bác bỏ**, **Còn chặn**. Không có mục nào bị làm mềm thành "đã cải thiện".
>
> **Tổng: 16/16 mục có kết luận. Đã sửa 15. Bác bỏ 0. Còn chặn 1 (R16).**
> **`research_ready` = 0/17** (`CENSUS_20261111.vi.md:92,169`). Mọi sửa đổi dưới đây làm gate **trung thực**,
> không làm model nào **đạt**.

---

## 1. Bảng kết luận

| ID | Kết luận | Sửa ở đâu | Commit | Test chặn tái phát |
|---|---|---|---|---|
| R01 | Đã sửa | `backend/native_measurements.py:616,636,749` | `9a951c5`, `ae6fa2b` | `test_native_measurements_identity.py` (16) |
| R02 | Đã sửa | `backend/parity_gate.py:114-139,267,445,623,659` | `d6d3e36`, `39ac8c4`, `f5ce7a5` | `test_gate_contract_negatives.py` |
| R03 | Đã sửa | `composition/adapters.py:306`, `model_encoder.py:58,240`, `runner.py:1635,2206,2258`, `replay/encoder.py:36,240,270` | `f856338` | `test_encoder_relation_handshake.py` (8), `…_native_parity.py` (7) |
| R04 | Đã sửa | `replay/common.py:119,128,130,238-533`, `scripts/warm_replay_cache.py:49,157` | `ff8d062`, `7c689fb`, `1346ebb` | `test_r04_cache_manifest.py` (16) |
| R05 | Đã sửa | `composition/model_limn.py:189-222`, `composition/data.py:477` | `a24635e` | `test_limn_annotation_provenance_r05.py` |
| R06 | Đã sửa | `frontend/scope.ts:221,230,237,288`, `ModelInventory.tsx:69,192-204` | `ab64911`, `e11f9ff` | `test_web_scope_and_labels.py` |
| R07 | Đã sửa | `backend/native_measurements.py:124` (parser duy nhất), `EvaluationPage.tsx:643-653` | `9a951c5`, `ab64911` | `test_native_measurements_identity.py` |
| R08 | Đã sửa | `backend/native_measurements.py:253,749` | `9a951c5`, `ae6fa2b` | `test_native_measurements_census.py` |
| R09 | Đã sửa | `scripts/m03/publish_derived_runs.py:38,140,153,199`, `backend/composition.py` | `5b93476` | `test_derived_subset_status.py` |
| R10 | Đã sửa | `frontend/scope.ts:329`, `QueryInspector.tsx` | `ab64911` | `frontend/tests/inspector-window.test.cjs` |
| R11 | Đã sửa | `CENSUS_20261111.vi.md` (mới) | `7c56510` | — (tài liệu) |
| R12 | Đã sửa | `CENSUS_20261111.vi.md` §R12.1–R12.6 | `7c56510` | — (tài liệu) |
| R13 | Đã sửa | `replay/pair_absence.py:101,220,249` | `9f6246f` | `test_blocked_models_verification.py` |
| R14 | Đã sửa | `scripts/setup_blocked_model_envs.sh:56-70` | `9f6246f` | recipe chạy `exit 0` |
| R15 | Đã sửa | `replay/airknow.py:83,144,223,512` | `8996b4a` | `test_r15_airknow_variance_group.py` (11) |
| R16 | **Còn chặn** | — | — | — (xem §17) |

---

## 2. R01 — Collector ghép số đo sai checkpoint/protocol

**Trước.** `_backed_by_artifact` có fallback `by_model_any_protocol`: khi không tìm được report cùng
protocol, nó lấy report bất kỳ của model rồi thay `metrics` trong khi giữ `checkpoint_id`/`protocol_id`
của row. Probe (`raw/verify/60-collector-identity.md:151-152`, chạy trên snapshot `06e30e8c`):
artifact A `checkpoint_A`/`protocol_A` R@10=12 + report B `checkpoint_B`/`protocol_B` R@10=88
⇒ response `r10 = 88.0`, giữ nhãn `checkpoint_A`, `artifact_backed=true`.

**Đã sửa.** `IDENTITY_JOIN_FIELDS = ("model_id", "protocol_id", "checkpoint_id")`
(`native_measurements.py:616`); `_identity_matches` (`:636`) so từng trường **cả hai bên cùng khai**;
fallback `by_model_any_protocol` **xoá**. Chính sách **từ chối, không dán nhãn lại**: row giữ số của mình,
`backed_by` trỏ về chính nó. Thay thế hợp lệ ghi `raw_model_id`/`raw_checkpoint_id`/`raw_protocol_id`/
`raw_source_path` cạnh `raw_metrics`. 2+ ứng viên ⇒ `identity_ambiguous: [paths]`, không thay.
Một relaxation duy nhất (`row.protocol_id is None` + đúng 1 ứng viên) — và **lỗ hổng bên hông** mà chính
review chỉ ra ở dòng "migration phải chỉ định run" đã đóng: relaxation chỉ áp dụng khi join **đã** khớp,
cộng guard cứng

```python
if report is not None and identity_join != "exact" and not _identity_matches(row, report):
    raise AssertionError(...)
```

**Sau.** Probe: `r10 = 12.0`, `raw_metrics=None`, `backed_by` = nguồn của chính row,
`identity_ambiguous` không rỗng (`60-collector-identity.md:160`, test
`test_a_single_candidate_cannot_back_a_row_that_names_another_checkpoint`).
Live: `csmcir` giữ `r10 = 57.05073873202006` dưới nhãn riêng `checkpoint_id=fashioniq` /
`fashioniq_original_split`, `identity_ambiguous` liệt kê cả hai report — không gì bị thay.

**Giới hạn phải nói:** `dataset`/`benchmark` **không** nằm trong join vì **không bên nào** ghi chúng
(bảng trong `60-collector-identity.md`); đây là trường vắng mặt ở cả hai phía, không phải trường bị bỏ sót.

Commit `9a951c5`, `ae6fa2b`.

## 3. R02 — Gate còn nhiều đường PASS giả

**Trước.** Probe `review_06e30e8c/gate_probe.py`: 5/6 fixture âm trả `passed=true, complete=true,
failure_count=0` (`raw/verify/61-gate-contract.md:3-4`).

**Đã sửa, từng đường một.**

| đường PASS giả | sửa | fixture sau |
|---|---|---|
| dimension bắt buộc gom toàn cục, một category đo tốt che category rỗng | phạm vi **theo category**: `missing_by_category`/`mandatory_by_category`, `missing` = hợp của các category | `other_categories_empty` → `MANDATORY_FAILED`, 1 failure |
| identity đọc từ boolean trong payload | `_identity_cross_checks` (`:445`) đọc **cả hai** document: `model_id`, `checkpoint_id`, `checkpoint_identity`, `protocol_id`, `query_order_sha256`, `gallery_order_sha256` | `contradictory_identity` → `MANDATORY_FAILED`, 5 failure |
| relation thiếu ⇒ `mandatory=False` | `RELATION_DRAWING_MODELS = ("encoder",)` (`:267`), **và** bắt buộc khi **một trong hai** bên có khoá `relation_state` | `complete_fixture_no_relation` → `MANDATORY_FAILED`, 1 |
| chỉ có score delta, không có đối chứng hạng | `_scoring_evidence` (`:623`) đòi **cả** score **và** rank | `score_only_no_rank_evidence` → `MANDATORY_FAILED`, 2 |
| body tolerance đè top-level | **một** nguồn tolerance; tolerance khác/lỏng hơn ném `_Malformed` (`:213`) | `local_tolerance_overrides_global` → `EVIDENCE_MALFORMED` |
| "vắng bằng chứng" và "claim sai" bị trộn | `ParityCheck.absence` (`:132`): vắng = `missing` (INCOMPLETE), vắng **bên cạnh** claim thật = `MANDATORY_FAILED` — quyết định **theo category** vì luật toàn cục làm mất đúng ca shirt fail của `other_categories_empty` | `empty_adapter_control` → `INCOMPLETE`, 0 failure, `missing=[forward,batch,scoring]` |
| PASS không thể kiểm | `_check_invariants` (`:659`): `passed` ⇒ `PASS` ∧ không failure ∧ không missing ∧ **mỗi** dimension bắt buộc có check mandatory trong **từng** category | mọi `to_dict()`/đường return sớm |

Số liệu probe sau (`raw/verify/61-gate-probe-after.json`): `complete_fixture_no_relation` 1,
`contradictory_identity` 5, `score_only_no_rank_evidence` 2, `other_categories_empty` 1,
`local_tolerance_overrides_global` = `PARITY_EVIDENCE_MALFORMED`, `empty_adapter_control` = `INCOMPLETE`.

**Verdict lịch sử.** ENCODER **13 → 16 failure** (ba digest relation trở thành bắt buộc); LIMN **3 failure
+ thiếu `metrics_parity`**, không đổi. Live `/api/runs/parity-gate`: `encoder`
`PARITY_MANDATORY_SUBCHECK_FAILED` 16 failure, `limn` 3 failure + `missing: ["metrics_parity"]`,
`passed_models: []`.

**Dư (chưa đóng, không giấu).** Review nêu "metric dimension hiện dùng
`native_metrics_match_feature_rerank` từ riêng phía native". Đúng và **vẫn còn**: `_metrics_checks`
(`parity_gate.py:677-681`) đọc cờ phía native. Nó là **observer nội bộ**, không phải đối chứng
adapter↔native; phần đối chứng thật nằm ở `scoring_parity` (score+rank hai phía). Không đổi nhãn
dimension này thành "đã đối chứng adapter↔native" — nó không phải.

Commit `d6d3e36`, `39ac8c4`, `f5ce7a5`.

## 4. R03 — ENCODER chốt identity trước khi có relation digest

**Trước.** `run_composition` thật với adapter tổng hợp: `COMPLETED`, adapter sau encode có digest hợp lệ,
summary vẫn `not_recorded`. Native replay tạo model **trước** khi seed.

**Đã sửa.**
- Hook `CompositionAdapter.prepare_identity()` (`adapters.py:306`) — base **no-op** trả `None`, nên run id
  của mọi model khác **không đổi một byte**. ENCODER override (`model_encoder.py:240`) khởi động worker
  ở phase `handshake` và digest `learnable_relation` **không encode** ảnh nào.
- Gọi tại `runner.py:1635`, **trước** khi `identity` được hash (`:1766`) và trước `run_dir`/`work_root`/
  `rank_path`/lock.
- `relation_bearing` + không digest ⇒ `ValueError relation_state_unrecorded` (`runner.py:1641`) — không
  bao giờ publish một run `not_recorded`.
- Summary ghi block **EFFECTIVE** đọc sau encode (`runner.py:2258`); lệch với digest trong identity ⇒
  `relation_state_drift` (`:2209`).
- Resume chỉ tái dùng payload khi `_recorded_relation_matches` (`:1241`, gọi ở `:1771`) — payload cũ
  (ghi `None`) **không** được tái dùng.
- Native: `torch.manual_seed(RELATION_SEED)` ở `replay/encoder.py:270`, **ngay trước** constructor `:272`;
  hai hằng số **import** từ `model_encoder` (`:36`) nên hai đường không thể trôi; persist
  `relation-seed<seed>.npy` + `.json` (`:287`); trả về cả `aggregate['relation_state']` và
  `ReplayResult.provenance` (`:380,434,435`).

**Sau.** Hai process sạch, 60 ảnh / 4 query, đồng ý digest
`20db0c27c979e9a0938c71af5142458dfa2070468d0ad3fcb6f24134c2f9ce92` = `RELATION_EXPECTED_DIGEST`
(`raw/m01/cross_process_bit_identity.txt:9`, `raw/m01/forward_delta_runs.txt:6`).
Test trước/sau ghi trong `raw/verify/62-encoder-relation.md` §4 (5 ca FAILED-before → PASS).

**Hệ quả phải nói rõ:** bốn run id **superseded** — `5dee7425dd46c5f154808aac`, `6a528e877c328772e2c7bb28`,
`ce89a9a27c4cceadeaaf30b0`, `eda30a37c0bc17ac43844f09` — vì identity của chúng **không** phủ tensor
relation. `ce89a9a2`, trước là run canonical của ENCODER, **nay không còn canonical**.

Commit `f856338`.

## 5. R04 — Warm cache native nhận dữ liệu pipeline khác

**Trước.** Probe: cache trỏ `OLD_ANNOTATIONS`, tensor của một CLIP transform lạ, gallery 2 phần tử,
transform `TargetPad` được yêu cầu chạy **0 lần**, `error=null`. Helper thật nhận cache đó.

**Đã sửa** (`replay/common.py`).
- `CACHE_MANIFEST_NAME = "workbench_cache_manifest.json"` (`:119`), ghi **atomic**
  (`NamedTemporaryFile` + `fsync` + `os.replace`, `:238-252`), do **cold constructor** ghi khi ra
  (`:520-533`).
- Trường manifest: `schema_version`, `model_id`, checkpoint, transform descriptor,
  `annotations{requested_path, resolved_path, digest}`, `gallery{count, digest, first, last}`, `state`,
  `tensors{sha256}`, `written_by`, `written_at`.
- `CacheManifestError` (`:130`) liệt kê **mọi** trường lệch. Symlink so bằng `Path.resolve()`.
  Root **không** có manifest ⇒ từ chối. Opt-out tường minh `WORKBENCH_CACHE_MANIFEST_OPT_OUT=1`
  (`:128`) ghi `not_verified_opt_out` — không bao giờ im lặng.
- Consumer `airknow`/`tgcir`/`clvc_net` ghi `provenance.cache_manifest`
  (`airknow.py:435`, `tgcir.py:273`, `clvc_net.py:254`).
- `scripts/warm_replay_cache.py`: `TENSOR_WRITING_MODELS = ("airknow", "tgcir", "pair")` (`:49`);
  `--model clvc_net` báo `cache_reusable: false`, `cache_files: []`, kèm `reason` (`:157-165`).

**Sau.** Probe ném `CacheManifestError` nêu đích danh ba đường dẫn annotation
(`annotations.captions.resolved_path`, `annotations.image_splits.resolved_path`,
`annotations.resized_image.resolved_path`); **không** load gì (`63-cache-manifest.md:65-70`).

**Sửa lại lời giải thích của chiến dịch:** câu "CLVC reuse được giải thích bằng tensor" là **SAI**.
`CLVCNet/datasets.py:152` giữ tensor trong bộ nhớ, không ghi file ⇒ **chưa từng có cache** để reuse.
Đây là lỗi của báo cáo cũ, không phải của review.

Commit `ff8d062`, `7c689fb`, `1346ebb`.

## 6. R05 — LIMN chọn scope sai với JSON hợp lệ nhưng thiếu dữ liệu

**Trước.** Probe (`raw/verify/64-limn-provenance.md` §"Probe, before / after"): `toptee_empty` → `'shirt'`,
`toptee_invalid_row` → `'shirt'`, loader toptee bị làm rỗng → `'shirt'`. Tức một file **hợp lệ nhưng
không dùng được** vẫn để resolver chọn category còn lại.

**Đã sửa.**
- `model_limn.py`: validate **từng row**; tập `unreadable` / `malformed` / `empty` / `mismatch`
  (`:76-100`); count+digest phải khớp dataset pin; **bất kỳ** tập nào không rỗng ⇒ **từ chối** route cho
  mọi query không có loader-scope.
- `composition/data.py::assign_garment_categories` (`:477`) validate **cả sáu** file identity trước, và
  đi **UNAVAILABLE** (không gán một phần) khi có file không dùng được **hoặc** split trùng stem.
- `resolve_query_scope` (`model_limn.py:189`) trả rationale từng query: `loader` / `annotation:<cat>` /
  `unresolved:<why>` + `provenance_incomplete`, xuất ra `<run>/scope/query_scope_limn.json`.

**Sau.** `toptee_empty` → `None`, `toptee_invalid_row` → `None`, `loader_toptee_emptied` → `None`;
control "đủ sáu file, 1 holder" **vẫn** resolve `'shirt'` (không chặn quá tay).

**Dữ liệu thật** (`64-limn-provenance.md` §5): 9 file pin khớp count+sha256 — captions val
2017/2038/1961, splits val 3817/6346/5373. Index `complete`, 6003 khoá, loader `resolved 5884 /
ambiguous 119 / unresolved 0`; trong 119, **đúng 2** vẫn unscored: `7:3038`, `7:3790` —
`reason = unresolved:annotation_names_two_categories`, `holding_categories = ("shirt","toptee")`,
`provenance_incomplete = false`, kèm block `candidates` nêu từng file, số row, usable, có chứa cặp hay
không. **Không** category nào được chọn.

**Chưa chứng minh được (và review đúng khi nêu):** 117 query lịch sử có thực sự bị định tuyến bằng
annotation hỏng hay không — không có pixel/annotation trace của các lần chạy cũ.

Commit `a24635e`.

## 7. R06 — Inventory báo "Chưa có số đo" cho scope đang có số

**Trước.** Catalog không chứa native rows; native rows chỉ vào danh sách "scope khác" rồi bị lọc khi
trùng protocol hiện tại. Probe: ConeSep có native row trong protocol đang chọn → `catalogEntries=0`,
`measured=false`, `elsewhere=[]`, trạng thái "Chưa có số đo".

**Đã sửa** (`frontend/scope.ts`).
- `nativeRowsFor` (`:221`), `nativeMeasuredIds` (`:237`), `nativeUnattributedFor` (`:230`) — khớp
  **đúng `(model_id, protocol_id)`**.
- `inventoryStatus()` (`:288`) là **một** hàm thuần giữ toàn bộ chuỗi ưu tiên (trước là tháp ternary).
  Bốn mâu thuẫn bị bỏ: (1) `nativeOnly` từng phủ nhận run composition đang có; (2) mọi `entry` không
  phải composition bị dán nhãn "Có dữ liệu native" (trỏ sai bảng); (3) `aggregate_metrics: false` vẫn
  đọc "Có số đo" trong khi counter loại nó; (4) nhiều run đã có số đo vẫn đọc "cần chọn".
- `measurementCount` (`ModelInventory.tsx:69`) dùng **cùng** vị từ với backend:
  `counted ∧ artifact_backed` (trước chỉ `!rollup`).
- Row native **không có** protocol **không** được thổi vào mọi protocol (đó chính là R01 ở tầng UI) — nó
  có trạng thái riêng "Số đo native không ghi protocol" + link tới bảng native. Model deferred mà có
  native row hiện **cả hai** sự thật.

Commit `ab64911`, `e11f9ff` (kèm `raw/verify/65-frontend.md` §D1/D2 + audit chuỗi ưu tiên).

## 8. R07 — Unit và raw disclosure sai 100 lần

**Trước.** Fixture `macro_r10=0.2793` + `macro_r10_percent=27.93`, không có backing report: reader chọn
`0.2793` rồi gán `percent` từ suffix của sibling ⇒ phục vụ **0.2793%**. `dead_parameters_pct` cùng block
được nhận làm khai báo thang Recall. UI raw disclosure dùng `row.unit`.

**Đã sửa.**
- `metric_values(block)` (`native_measurements.py:124`) là **parser duy nhất**; `_report_metrics` và
  `_blocks`/`_rows_for_file` đều gọi nó. Mỗi metric thử `UNIT_SUFFIX_ORDER` (`_percent`, `_pct`, bare,
  `_fraction`) và lấy key **đầu tiên là số thật** ⇒ giá trị và unit **luôn** mô tả **cùng một key**.
  Suffix chỉ tính khi gắn vào alias **của chính metric đó** ⇒ `dead_parameters_pct` không khai báo gì cho
  recall. `_suffix_declared_unit()` **đã xoá** (0 tham chiếu). Trả thêm `per_metric_units` để block trộn
  thang là **thấy được**, không bị gộp thành một phỏng đoán.
- `EvaluationPage.tsx:643-653`: `row.raw_unit` (fallback `row.unit`), câu nêu thang đang hiện, và khi
  `raw_unit_source === "inferred_from_magnitude"` ghi "suy theo độ lớn, không phải khai báo".

**Sau** (`60-collector-identity.md:163-164`): `r10 = 27.93`, `unit=percent`, `unit_source=sibling_key`,
`per_metric_units={r10:_percent, r50:_percent}`; ca chỉ có `dead_parameters_pct` → `r10 = 0.2793`,
`unit=fraction/inferred_from_magnitude`, `per_metric_units={r10:"", r50:""}`.

**Live** (`/api/native-measurements`, HEAD này): DCNet `r10 = 27.929730914548916`, `unit=percent`,
`unit_source=sibling_key`, `raw_unit_source=inferred_from_magnitude`. Counts: `unit_declared 4`,
`unit_sibling_key 2`, `unit_inferred 39`, `unit_undeclared 0`.

Commit `9a951c5`, `ab64911`.

## 9. R08 — Dedup theo điểm làm tròn gộp hai checkpoint

**Trước.** `_mark_duplicates` khoá theo `round(value, 4)` ⇒ hai checkpoint khác nhau cùng model/protocol
mà recall bằng nhau bị gộp.

**Đã sửa** (`native_measurements.py:749`). `duplicate_of` chỉ ghi trong **hai** quan hệ ancestry:
(1) roll-up và một row khác dùng **cùng** artifact cho cùng model+protocol (một run ghi hai lần);
(2) roll-up nhắc lại số của artifact có trong cây (quyết định bằng **source KIND**, bằng nhau về giá trị
chỉ dùng để chọn artifact nào). Mọi trường hợp khác giữ `counted: true` + `same_metrics_as: <path>`.
Kèm `_checkpoint_fingerprint` (`:253`) đọc digest mà tài liệu **tự khai** (top-level, hoặc nhất trí qua
`categories`/`results`) — `rule 2` chỉ hạ cấp khi cùng model+protocol+digest+số.

**Điều tra riêng của chiến dịch (phải ghi vì nó đi ngược):** `with_metrics` có lúc 21 → 23 rồi **quay
lại 21**. Lý do đảo: cặp `hint` và `tme` mỗi cặp khai **cùng** `provenance.checkpoint_sha256`
(hai bản export của **một** checkpoint) ⇒ là **một** phép đo. Bảng chứng minh:
`60-collector-identity.md` §"with_metrics 21 → 23" (digest `bf9c72d4…` cho hint, `fea7fc08…` cho tme).
Rule mới **không** gộp được hai checkpoint thật: `airknow_n05` (`f4d25ed9…`) và `airknow_n08`
(`756fc0bd…`) khác digest, vẫn **hai** row counted.

**Sau.** Live `/api/native-measurements`: `with_metrics 21`, `duplicate_rows 22`, `unbacked_counted 0`,
`blocks_with_metrics 43`, `models_with_metrics 17`.

Commit `9a951c5`, `ae6fa2b`.

## 10. R09 — Derived LIMN mất lý do unscored, chưa dùng candidate của parent

**Trước.** Publisher ghi `query_failures` rỗng; reader không truy candidate parent.

**Đã sửa.**
- `publish_derived_runs.py`: chép lý do failure của parent **theo condition**, **chỉ trên membership của
  subset dẫn xuất**; parent không có lý do ⇒ `parent_recorded_no_reason` (`:38`); `SystemExit` nếu một qid
  mang lý do nằm ngoài membership **hoặc** một qid unscored thiếu lý do. `derived_from` thêm
  `parent_topk_export` (requested/status/depth/conditions/partial) (`:140,153`).
- `backend/composition.py`: xác minh `parent_rank_artifact_sha256` mà run khai, đọc export của parent,
  báo `NOT_REQUESTED`/`NOT_EXPORTED`/`PARTIAL`/`COMPLETE`, từ chối bằng `parent_topk_unavailable` kèm lý do;
  `candidate_status` mang nguyên trạng thái của parent (không đổi nhãn thành `AVAILABLE`);
  `candidate_source='parent_run'` + `candidate_source_run_id`. Block `candidates` của page thêm `source`,
  `source_run_id`, `parent_run_id`, `lineage_verified`; thêm block `lineage`.
- Hai run dẫn xuất trên đĩa được vá, bản cũ giữ `<name>.pre-r9`; script vá **assert** ranks và metrics
  byte-identical trước/sau (sha256 niêm yết trong `raw/verify/66-derived-lineage.md`).

**Sau** (live `/api/runs`): `6a96c5c199526b61c1470b0e` — `subset=sf`, `membership_count=4069`,
`derived_from.own_status_label=DERIVED_PARTIAL`, `coverage=0.9997542393708528`;
`aee2e8d39214d877a84d7c08` — `subset=validated`, `membership_count=586`, `DERIVED_COMPLETE`,
`coverage=1`. `execution_status` của cả hai là `PARTIAL` (kế thừa từ parent, đúng option (b)) — **không**
gán `DERIVED_*` vào enum đó.

Commit `5b93476`.

## 11. R10 — Inspector giải thích gallery bằng hạng của condition khác

**Trước.** `Math.min` trên hạng của cả ba condition. Ví dụ MM rank 50 + text rank 5, K=10 ⇒ câu
"ảnh đích ở hạng tốt nhất 5 — ngoài cửa sổ Top-10" trong khi đang mô tả **MM**.

**Đã sửa.** `candidateWindowVerdict()` (`scope.ts:329`) khoá theo condition **đang xem** (cũng là
condition mà request candidate nêu tên), câu nêu tên condition, và **từ chối kết luận** khi vùng của cửa
sổ đang `idle`/`loading` ("Đang tải cửa sổ … Chưa kết luận") hoặc `error`/không có exact rank. Phân biệt
"rank ≤ K nhưng export không chứa ảnh đó" với "rank > K".

**Sau.** Query `7:3542` trên `limn`: ranks `multimodal 6010`, `text_only_black_image 5`,
`image_only_empty_text 49136` (đọc từ `workbench/artifacts/composition/runs/*/ranks/*.json`).
Câu render: "#6,010 theo điều kiện Ảnh + văn bản (MM)" — trước đó trích #5 của text.

Commit `ab64911`.

## 12. R11 — "17 model dùng được" và "15 Y / 2 N"

**Trước.** `FINAL_REPORT.vi.md:198` định nghĩa "dùng được để so sánh nghiên cứu" = `counted ∧
artifact_backed` = 17 model / 21 dòng, và bảng dẫn `NATIVE_EVAL_MATRIX.csv` nói 15 Y / 2 N.

**Đã sửa bằng cách ĐẾM, không khẳng định** — tài liệu mới `CENSUS_20261111.vi.md` (`7c56510`).

| phép đếm | kết quả | nguồn |
|---|---|---|
| `research_ready` | **0 / 17** | `CENSUS:92,169`; định nghĩa `G1∧G2∧exit_code_0_proof∧artifact_backed` |
| G1 PASS | 2 (encoder, limn) | gọi thẳng `evaluate_model` (`CENSUS:60`) |
| G2 PASS | **0** | như trên |
| G1 PASS / G2 FAIL | 2 (encoder 16 failure, limn 3) | `/api/runs/parity-gate` |
| NOT_RUN | 15 | không tồn tại artifact `g1_g2.*` |
| `native_executed` đúng như CSV | **12 Y / 5 N** | đếm lại: `N` = airknow, pair, clvc_net, tgcir, sprc |
| CSV có phản ánh 3 run GPU mới? | **không** | `airknow`/`tgcir`/`clvc_net` chạy 2026-11-10, sau chiến dịch 20261109 |
| `exit_code_0` chứng minh được | **7** | 7 file `raw/native/*.meta.txt` chứa `EXITCODE=0` |
| mức bằng chứng | `native_fidelity 7`, `circus_adaptation 6`, `transfer 1`, `historical 2`, `EXPLORATORY 1` | `CENSUS` §1 |

Review đúng ở cả hai chỗ: CSV là 12/5 (không phải 15/2), và 21 row counted+backed **không** đồng nghĩa
"dùng được". Live xác nhận `models_with_metrics 17`, `verified 0` — tức **không** row nào là
`is_research_evidence`.

**Dư (phải nói):** câu `FINAL_REPORT.vi.md:198` **vẫn còn** trong file master report; census mới là nơi
đúng. Đây là drift chéo tài liệu, đã ghi ở §17.

## 13. R12 — Số liệu và các câu "bác bỏ audit cũ"

Đối soát từng mục trong `CENSUS_20261111.vi.md` §R12.1–R12.6.

| # | phát hiện của review | đối chiếu | kết quả |
|---|---|---|---|
| 1 | ENCODER Full gap phải là `0.474394 / 6003`, không `0.539144 / 6001` | `summary/encoder_fashioniq_ce89a9a2…/full.json`: `composition_gap_ndcg 0.47439372489324116`, `gap_cohort_count 6003`, `query_count 6003`; `0.5391438513723383 / 6001` là của **LIMN** `9e4918a6…` (`raw/m03/derived_subset_runs.json:56,67`), dán nhầm sang mục ENCODER | **review đúng** |
| 2 | tie-rule là **7/17** không phải 10/17 | 6 config `torch.argsort` tăng dần trên `1-cos` (CLIP4Cir ×2 config chung 1 file, MTST_PTHA, TME, CSMCIR, SPRC) + DCNet `torch.topk` ×3 nhánh = **7**; số 10 đến từ đếm **dòng** (CSMCIR 5 lời gọi, MTST 3, SPRC 3, CLIP4Cir 2, DCNet 3 = 16 dòng) | **review đúng**; 10 là số config **không** bị ảnh hưởng |
| 3 | `unit_declared(4)` **đã bao gồm** `unit_sibling_key(2)` ⇒ 4+2 đếm đôi | `DECLARED_UNIT_SOURCES = ("declared","sibling_key")` (`native_measurements.py:83`), `:908-913` | **review đúng**; `4 + 39 = 43` mới là phép cộng đúng |
| 4 | CLVC transform **chưa từng** được đo; equal Recall không chứng minh transform vô can | không có counterfactual `Resize(224)`, không có pixel trace lịch sử | **review đúng**; claim cũ bị rút |
| 5 | TG-CIR không "digit-for-digit" | dress R@1 `14.377788795240455` (lịch sử) vs `14.328210213187903` (`raw/tgcir_run3cat_gpu.json:11`) — lệch 0.0496 pp; R@10 `43.38125929598414` và R@50 `68.368864650471` khớp | **review đúng**; claim chỉ đúng cho R@10/R@50 |
| 6 | trong 10 câu "bác bỏ": #1 (URL registry) và #5 (BLAS) **không** đứng vì khác đối tượng; #2 đúng hướng sai số | #1: audit cũ nói về `PROTOCOL_MATRIX.csv` (`20261110-e2e-repair/`), không nói `registry/models.yaml`. #5: audit cũ đo **môi trường của chính nó** | **review đúng**: **7 đứng** (3,4,6,7,8,9,10), 1 đúng hướng sai số (2), 2 **không đứng** (1,5) |

Verdict trên 10 câu: **7 đứng, 1 đứng-về-hướng-nhưng-sai-số, 2 không đứng**.

## 14. R13 — PAIR absence checker kiểm sai thư mục

**Trước.** Checker kiểm direct child của `third_party/`, nên không nhạy với đường import thật; và
`undeclared_state_names` regex `backbone.<name>` nên trả **0** cho những tên mà checkpoint **có**.

**Đã sửa** (`replay/pair_absence.py`).
- `replay_import_roots` (`:101`) đọc **root import thật** của replay bằng AST trên `pair.py`
  (`""`, `data`, `models`, `utils`); presence xét **từng class**, không xét module tồn tại.
- Bốn verdict dẫn xuất (`:220`): `BLOCKED_MISSING_SOURCE_CLASSES`, `BLOCKED_INCOMPLETE_SOURCE`,
  `SOURCE_CLASSES_PRESENT`, + trường hợp không xác định.
- `undeclared_state_names` (`:249`) đọc **opcode pickle** thay vì regex ⇒ **1** mỗi cái trên **cả hai**
  checkpoint PAIR.

**Sau.** Trên cây thật verdict vẫn `BLOCKED_MISSING_SOURCE_CLASSES`
(`raw/pair_absence_evidence.json:verdict`) — **blocker đứng nguyên**, nhưng nay được chứng minh bằng
**giải được import**, không phải bằng phỏng đoán đường dẫn. Probe của review (đặt đủ 7 class vào đúng
`third_party/PAIR/models/…`) là bằng chứng checker cũ **không đủ** làm bằng chứng vắng mặt — điều đó
đúng và đã được sửa.

Commit `9f6246f`.

## 15. R14 — Recipe PAIR chọn Python 3.8 cho Torch 2.7.1

**Trước.** Recipe tạo env Python 3.8 rồi cài `torch==2.7.1` — không thể resolve.

**Đã sửa** (`scripts/setup_blocked_model_envs.sh:56-70`). Env `workbench/envs/pair39-la`, và sàn Python
**đo từ metadata đã cài**, không từ README:

| gói | phiên bản | `Requires-Python` |
|---|---|---|
| `torch` | `2.7.1+cu128` | `>=3.9.0` |
| `torchvision` | `0.22.1+cu128` | `>=3.9` |
| `open-clip-torch` | `2.24.0` | `>=3.7` |

Interpreter `workbench/envs/pair39-la/bin/python` → **Python 3.9.25**. Recipe chạy **exit 0**.
Comment trong script ghi rõ: thiếu `model_vssp`/`MHTransformer_dis` **không** phải vấn đề phiên bản
Python — chúng vắng ở **mọi** root mà replay thêm vào `sys.path`.

Commit `9f6246f`.

## 16. R15 — Air-Know ghi biến thiên lịch sử thành thuộc tính lượt mới

**Trước.** `MEASURED_RUN_TO_RUN_DELTA_MACRO_R10` và nhánh `0.0 if eval_mode else 0.4744` gán delta lịch
sử vào output của lượt chạy **tương lai**; `repeat_count` chỉ gắn nhãn một draw.

**Đã sửa** (`replay/airknow.py`). Hai hằng/nhánh đó **đã xoá**. `assemble_variance_group()` (`:144`) dựng
thống kê **tại thời điểm emit** từ các sibling run **cùng config** (checkpoint, `eval_mode`, protocol,
batch, transform). `n == 1 ⇒ sd = null` + `insufficient_draws: true` (`:223`). `eval_mode` trộn ⇒ raise.
Sibling **không** có identity đã ghi ⇒ **loại**, không gộp. Delta lịch sử chỉ còn ở dạng
`historical_campaign_delta_macro_r10 {applies_to_this_run: false}` (`:83`, `:512`) — hai nửa không tách
rời được.

**Sau.** Bốn raw Air-Know cho mean R@10 **52.54676006631348**, sample SD **0.26728186433775697**, spread
**0.6252866852209422** (`raw/airknow_run3cat_gpu_variance_correction.md:21`) — khớp con số review nêu.
Artifact đã commit **không** bị viết lại; correction ghi trong raw nêu trên.

**Giới hạn (review nêu, chấp nhận):** CPU/GPU log chưa khoá mode/RNG/input digest nên **chưa** đủ để quy
toàn bộ chênh lệch cho precision.

Commit `8996b4a`.

## 17. R16 — Ingest mới, adapter từ chối, service tự hồi phục

Ba vế, ba kết luận khác nhau. **Tổng: Còn chặn.**

**(a) Native mới chưa ingest — CÒN CHẶN.** `/api/native-measurements` phục vụ `airknow`/`tgcir`/`clvc_net`
từ `checkpoint_selection/20261106T002617+0700/measurements/{airknow_fiq_n05_run1,airknow_fiq_n08_run1,
tgcir_run1,clvc_run1}.json`. Ba run GPU mới (`raw/airknow_run3cat_gpu_run{1..4}.json`,
`raw/tgcir_run3cat_gpu.json`, `raw/clvc_run3cat_gpu.json`) **không** có trong cây ingest; `tgcir_run1`
phục vụ `r10 = 49.720772235486656` trong khi `tgcir_run3cat_gpu.json` ghi `43.38125929598414`. Nguyên
nhân: cơ chế ingest **không có khái niệm `superseded_by`**, nên chép file mới vào là thêm row chứ không
thay row cũ (`BLOCKED_MODELS_VERIFICATION.vi.md` §13.4–13.6). **Điều kiện thiếu:** một trong hai —
(1) `native_measurements.py` hiểu `superseded_by` + row cũ khai trường đó, hoặc (2) quyết định của chủ dự
án rằng run cũ bị thay thế và row mới được thêm.

**(b) CLVC/TG-CIR composition adapter luôn từ chối — CÒN CHẶN, có chủ đích.** `model_clvc_net.py:199,205`
và `model_tgcir.py:171,177` ném `Refused("no interpreter environment resolved for WORKBENCH_CLVC_PYTHON")`
/ `Refused(_refusal(adapter))`. Từ chối là kết quả **đúng** khi chưa có interpreter; native replay đã mở
là đường khác, không thay thế được composition. **Điều kiện thiếu:** interpreter cho stack pin của từng
model (CLVC: Python 3.7.6 / torch 1.6.0 / cu102; TG-CIR: torch 1.7.0 / cu110).

**(c) Service tự chết rồi không tự hồi phục — CÒN CHẶN một phần.** Đường chết **đã xác định**:
`serve_workbench.py:152-166` — `forward_stop` nhận SIGTERM/SIGHUP, gửi `os.kill(child.pid, SIGTERM)` cho
từng child rồi `raise_signal(SIGINT)`; child **không** tự chết. Bốn chữ ký đã commit:
`raw/service-death-log.txt` (PID 3691345), `-2` (4040574), `-3` (4062957), `-4` (4082632).
**Điều kiện thiếu:** chưa xác định **ai** gửi SIGTERM. Đã kiểm và loại trừ `jobs.py` (spawn với
`start_new_session=True` nên `killpg` chỉ chạm group riêng của job). Không suy đoán thêm.

**(d) `/api/models` cold 154 s — đo lại tại HEAD này: 200 trong 0.008 s** (warm; không xoá cache nên
không phủ nhận số cold). Không còn là vật cản poll.

---

## 18. Chặn còn lại — trạng thái trung thực

| # | chặn | bằng chứng | điều kiện thiếu |
|---|---|---|---|
| 1 | PAIR | `raw/pair_absence_evidence.json` `verdict: BLOCKED_MISSING_SOURCE_CLASSES`; không revision nào trong **21 commit** của `iLearn-Lab/ICASSP25-PAIR` chứa `model_vssp*`/`MHTransformer_dis*`, `git grep` trên mọi rev = 0 | upstream phải publish hai module đó |
| 2 | 13 model **chưa có** artifact G1/G2 (NOT_RUN, không phải FAIL) | `CENSUS:169` | mỗi model một differential native↔adapter trên **một mẫu thật cố định**, tolerance khai **trước** khi đo; hàng giờ GPU mỗi model, một job GPU tại một thời điểm |
| 3 | `7:3038` / `7:3790` không thể resolve canonical | preprocessor upstream **chưa bao giờ** ghi khoá garment-category; hai qid nằm ở **cả** shirt và toptee | dữ liệu upstream, không phải code |
| 4 | 39/43 dòng vẫn dựa trên unit **suy theo độ lớn** | live counts `unit_inferred 39` | sửa producer của **từng** artifact để tự khai thang |
| 5 | Chỉ 2/16 model có bất kỳ bằng chứng G1/G2, **cả hai FAIL** | `/api/runs/parity-gate` `passed_models: []` | chạy differential thật (mục 2) |
| 6 | Service chết bởi SIGTERM ngoài, không rõ nguồn | §17(c) | xác định tiến trình gửi tín hiệu |
| 7 | Nhất quán chéo tài liệu | sáu drift đã sửa tay (`FINAL_REPORT.vi.md` §6 "Hạn chế đã biết"); sweep R12 tìm thêm số sai, nay ghi trong CENSUS | chín tài liệu còn lại **không** được bảo đảm nhất quán nội bộ |
| 8 | `FINAL_REPORT.vi.md:31` còn ghim bundle **cũ** `index-DWpK0yeS.js` / `7e34724e…` | bundle sống là `index-DeTvxLlD.js` / `1503e891…` (§19) | sửa dòng đó (thuộc tài liệu master, ngoài phạm vi được giao lần này) |
| 9 | `MODEL_LIMN_ENCODER_VERIFICATION.vi.md:434` còn `0.539144 / 6001` và câu "không đổi so với run canonical cũ" | §13 mục 1 | chủ dự án duyệt sửa; census đã ghi số đúng |
| 10 | `FINAL_REPORT.vi.md:198` còn "17 model / 21 dòng dùng được" | §12 | như trên |

**Kết luận không đổi:** mọi sửa trong đợt này làm gate **trung thực**, không làm model nào **đạt**.
`passed_models: []` là kết quả **đúng**.

## 19. Trạng thái sống tại HEAD

| sự thật | giá trị | nguồn |
|---|---|---|
| revision được nghiệm thu | `cd980dcecbb685b3cd6be4516d6550660a8e89b8` (= `TESTED_SOURCE_SHA`) | dòng `TESTED_SOURCE_SHA` trong `reviews/FINAL_PC_RELEASE_VERDICT.md` |
| sản xuất đang có = sản xuất đã sweep | `git diff cd980dc HEAD -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'` **rỗng** (exit 0) | chạy đúng lệnh đó |
| HEAD sống | **không ghim** — `HEAD` trôi theo từng commit; đọc bằng `git rev-parse HEAD` | `git rev-parse HEAD` |
| bundle | `assets/index-DeTvxLlD.js` | `workbench/frontend/dist/assets/` |
| sha256 bundle (dist) | `1503e8911122587ba8c81e8ecf93d151312fda41e12ff5ba9580ef1797e18155` | `sha256sum` |
| sha256 bundle (phục vụ) | `1503e8911122587ba8c81e8ecf93d151312fda41e12ff5ba9580ef1797e18155` | `curl :5173/assets/index-DeTvxLlD.js \| sha256sum` |
| `cmp` phục vụ ↔ `dist/` | **identical** | `cmp /tmp/served_now.js workbench/frontend/dist/assets/index-DeTvxLlD.js` → exit 0 |
| `/api/native-measurements` counts | `with_metrics 21`, `blocks_with_metrics 43`, `rollup_rows 18`, `duplicate_rows 22`, `unbacked_counted 0`, `models_with_metrics 17`, `models_rollup_only 8`, `verified 0`, `unit_declared 4`, `unit_sibling_key 2`, `unit_inferred 39` | `curl :8000/api/native-measurements` |
| DCNet row | `r10 = 27.929730914548916`, `unit=percent`, `unit_source=sibling_key` | cùng payload |
| `/api/runs/parity-gate` | `passed_models: []`; encoder `MANDATORY_FAILED` 16 failure; limn `MANDATORY_FAILED` 3 failure + `missing ["metrics_parity"]`; `standard parity-relative/v1` | cùng API |
| derived runs | `6a96c5c199526b61c1470b0e` `DERIVED_PARTIAL`, `coverage 0.9997542393708528`; `aee2e8d39214d877a84d7c08` `DERIVED_COMPLETE`, `coverage 1` | `/api/runs` → `derived_from` |
| `/api/models` | 200 trong **0.008 s** (warm) | `curl -w '%{time_total}'` |
| `research_ready` | **0 / 17** | `CENSUS_20261111.vi.md:169` |
| số ca `EXITCODE=0` chứng minh | 7 | 7 file `raw/native/*.meta.txt` |

Lệnh gọi API nêu trên chạy trên `:8000`; `:8765` là uvicorn cũ và trả 404 cho các route mới — nhận diện
bằng **cổng** + hành vi route, không bằng PID.

## 20. Sweep

Sweep serial đầy đủ đã chạy xong tại `cd980dc` (`/home/heheboiz/data/cir/workbench/envs/ctl312h/bin/python -m
pytest workbench/tests -q -p no:cacheprovider -rfE --tb=short`); số dưới đây lấy từ bản ghi đã commit, không từ trích dẫn miệng.

**Kết quả sweep (đã xong).** Sweep serial tại revision `cd980dcecbb685b3cd6be4516d6550660a8e89b8`, dòng tổng kết nguyên văn:

```
18 failed, 2901 passed, 47 skipped
```

2966 collected, **accounting gap 0**; `^ERROR ` = **0**; `Errno 28` = **0**. So bộ failure với baseline `20261110-e2e-repair/full_suite_68ad75e.txt`: **NEW=0, DROPPED=0, COMMON=18** — không có hồi quy. HEAD hiện tại chỉ đi trước revision đó bằng commit tài liệu (`git diff cd980dc HEAD` ngoài `workbench/docs` và `workbench/tests` là rỗng), nên sweep này nghiệm thu đúng cây sản xuất đang có. Phân loại 18 failure: **18 PRE-EXISTING** — trong đó **6 ENVIRONMENT** và **1 FLAKY** (flaky chỉ trong suite đầy đủ; pass khi chạy riêng). Bản ghi đầy đủ: `raw/verify/68-final-sweep.md`.

Bối cảnh đã có, dùng được ngay: sweep serial gần nhất (`raw/verify/67-regression-sweep.md`) tại
`a24635e` → `18 failed, 2901 passed, 47 skipped`, 2966 collected, **accounting gap 0**, **0 NEW / 0 DROPPED
/ 18 COMMON** so với baseline `68ad75e` (`raw/verify/67a-failure-diff-a24635e.txt`). Passed tăng
2701 → 2901, toàn bộ là test mới của lô `d6d3e36…a24635e`. Sweep tại `cd980dc` bao trùm **toàn bộ** lô sửa
— từ `d6d3e36` đến `cd980dc`, gồm cả `f856338` (bản sửa ENCODER, commit **trước** `cd980dc`) — và là bản có
thẩm quyền. Nó **dư thừa với** lượt `a24635e`, không phải bản kế tiếp của lượt đó: lượt `a24635e` chỉ được giữ
vì là lượt serial đầy đủ **đầu tiên** của lô, và số của nó khớp chính xác với lượt `cd980dc`.

Guard đã thật sự có hiệu lực: `test_handoff_commands.py::test_the_verdict_names_the_same_test_count_the_handoff_does`
bị cắt cụt từ `cb505c0` (nó chỉ đọc file verdict, không hề mở bản handoff nên không so sánh hai tài liệu) và
được phục hồi ở `bd9845d`; nay nó **fail** khi hai tài liệu ghi số khác nhau, đã chứng minh trên một cặp tài
liệu cố tình lệch (`2901 passed` vs `2800 passed`).

## 21. Chưa kiểm được

| điều | vì sao |
|---|---|
| số sweep tại `cd980dc` | **đã có** (§20); sweep chạy serial, không song song |
| `/api/models` **cold** 154 s còn đúng không | đo được 0.008 s **warm**; không xoá cache / không restart service (bị cấm trong đợt này) |
| 117 query LIMN lịch sử có thật sự bị định tuyến bằng annotation hỏng | không tồn tại annotation/pixel trace của các lần chạy cũ |
| CLVC transform có vô can không | **chưa từng** đo counterfactual `Resize(224)`; equal Recall không chứng minh |
| ai gửi SIGTERM cho launcher | chưa xác định; đã loại trừ `jobs.py` (§17c) |
| G1/G2 cho 13 model NOT_RUN | chưa có artifact; cần GPU |
| eval trên browser live | chỉ đọc dist + `curl` bundle + node test; không mở browser |
| run GPU mới của airknow/tgcir/clvc_net có khớp run cũ | cần ingest trước (§17a) |

## 22. Tổng kết

- **16 mục review, 16 kết luận: Đã sửa 15, Bác bỏ 0, Còn chặn 1 (R16).**
- **0 phát hiện bị bác bỏ** — mọi phát hiện P1/P2 đều tái hiện được trên source/code, và mọi phép so
  số của review (CSV 12/5, gap `0.474394/6003`, tie-rule 7/17, `unit_declared` bao gồm `sibling_key`,
  TG-CIR lệch R@1, CLVC không có cache) đều đúng.
- **`research_ready` = 0/17.** `passed_models: []` không đổi: các sửa đổi làm gate **không thể PASS giả**,
  không làm model nào đạt.
- Còn chặn thật: PAIR (upstream chưa publish hai module), 13 model chưa có G1/G2, `7:3038`/`7:3790`,
  39/43 row dùng unit suy luận, ingest run GPU mới, nguồn SIGTERM, nhất quán chéo tài liệu.
- Sweep tại `cd980dc` **đã xong**: `18 failed, 2901 passed, 47 skipped`, gap 0, NEW=0 / DROPPED=0 / COMMON=18 (§20).
