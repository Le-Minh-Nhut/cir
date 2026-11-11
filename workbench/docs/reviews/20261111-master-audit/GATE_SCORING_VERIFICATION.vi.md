# Gate, scoring, cache — xác minh trên HEAD `92ed66de`

Phạm vi: G-01, G-02, S-01…S-06 của `CIR_MASTER_FIX_AND_UPSTREAM_VERIFICATION.vi.md` §3.
Đối tượng kiểm: `workbench/backend/parity_gate.py`, `workbench/composition/{runner,metrics}.py`,
các test `test_gate_*`, `test_composition_*`, `test_parity_*`, và (được cấp quyền ở lượt follow-up)
`workbench/docs/reviews/20261110-e2e-repair/scripts/pc_r3_real_case.py`.

Mọi kết luận dưới đây đọc từ code trên HEAD, rồi dựng fixture nhỏ nhất chứng minh. Bằng chứng thô ở
`workbench/docs/reviews/20261111-master-audit/raw/`, script tái lập ở `…/scripts/`. Không commit,
không chạy full suite, không chạy GPU, không restart service.

**Cập nhật lượt follow-up:** G-02 đã được cấp quyền file và đã sửa + rerun (§4); gate change được
kiểm live-safe (§9); câu hỏi về count trên web đã trả lời (§9.1).

## 0. Cơ chế đo và giới hạn của nó

**BLAS (S-02).** `threadpoolctl` KHÔNG có trong `ctl312h`, nên không dùng nó. Cơ chế đã dùng, và đã
xác minh nó nói về đúng thư viện đang tính toán:

- `runner._openblas_symbols()` lấy `openblas_set_num_threads64_`/`openblas_get_num_threads64_` bằng
  `ctypes` từ `numpy.libs/*openblas*`;
- đối chiếu với `/proc/self/maps` của chính process: đường dẫn helper glob **trùng** đường dẫn
  openblas đang được map (`source_is_the_mapped_library: True`,
  `raw/gate_s02_pool_probe.txt`), nên con số đọc được là của phép nhân đang chạy, không phải của một
  thư viện khác trên đĩa;
- `torch.get_num_threads()` = 20 và **không** bị lệnh pin BLAS chạm tới — hai pool khác nhau, không
  được lẫn.

Giới hạn: giá trị OpenBLAS là **process-global**. Đọc được nó không nói gì về process khác, và không
có cách nào pin pool của process khác từ đây. Đó chính là lý do phần "drift" bên dưới phải **phát
hiện** chứ không thể **ngăn**.

**Artifact JSON.** Không dùng `?q=` lồng; đọc bằng `json.load`.

## 1. Bảng kết luận

| issue_id | status | file:line | lệnh failing-first + output | fix | lệnh regression + output | affected_run_ids |
|---|---|---|---|---|---|---|
| **G-01** | `CONFIRMED` → `FIXED_VERIFIED` | `backend/parity_gate.py:596-830`, `:181-341`, `:459-511`, `:833-881` | `raw/gate_negatives_before.txt` — 23/25 fail, gồm `PARITY_PASS` cho adapter `{}`, `PARITY_PASS` cho delta `0.1` với cờ `within_tolerance: true`, `AttributeError` khi native là list, `PARITY_PASS` với tolerance `inf` | validate schema/kiểu/hữu hạn; verdict tính lại từ số; `INCOMPLETE` tách khỏi `PASS`; chuẩn áp dụng đọc từ artifact; tolerance hai bên lệch → từ chối; model thứ ba được gate | `raw/gate_negatives_after.txt` — `25 passed` | không có run nào bị đổi (gate chỉ đọc); verdict mới ghi ở `raw/gate_head_verdicts_after.txt` |
| **G-02** | `CONFIRMED` → `FIXED_VERIFIED` (viết lại + rerun) | `docs/reviews/20261110-e2e-repair/scripts/pc_r3_real_case.py` (viết lại toàn bộ) | `raw/gate_g02_r3_fixtures.txt` — fixture chứng minh bản cũ probe gap trên **row khác** row đã rank và đọc crossing từ slice `order[lo-1:hi]`; `raw/gate_g02_r3_rerun.txt` — bản cũ nhận diện **1/11** target, bản mới **11/11** | join theo gallery **position** từ `inputs.<cat>.json`; cùng mask cho hai ma trận; crossing pairwise kèm distance tới target **và** cutoff; `TIE_EPSILON=1e-6` khai trước khi đo | `raw/gate_g02_r3_fixtures.txt` — `5 passed`; `raw/gate_g02_r3_rerun.txt` — `28→31` sống, phạm vi 1 → **5/11** ca đổi rank | harness, không sinh run production; đọc cũ trong `20261110-e2e-repair` đánh dấu superseded |
| **S-01** | `FIXED_ALREADY` + `FIXED_VERIFIED` (phần thiếu) | `composition/runner.py:158-246`, `:1774` | backfill đã pin BLAS và đã có `BackfillRankDisagreement` trên HEAD; phần **chưa** có test là refusal khi rank lệch | thêm test khoá refusal, kể cả `None`/`{}` | `raw/gate_s01_backfill_refusal.txt` — `5 passed`; `raw/gate_s01_s03_s04_regression.txt` — `73 passed` | không có |
| **S-02** | `REFUTED` (phần "pin không có tác dụng") + `CONFIRMED` (phần "job đồng thời") → `FIXED_VERIFIED` | `composition/runner.py:959-1008`, `:1012-1065`, `:219-227`, `:1972-1990` | `raw/gate_s02_drift_before_after.txt`: trên HEAD ghim 1 rồi để setter khác đặt 10 ⇒ `policy_observed = verified`, `HEAD has blas_pin_scope: False`. Test mới fail trước khi sửa (`raw/gate_s02_isolation_before.txt`, 4 failed) | thêm `blas_pin_scope()`: lock nội process + đọc lại pool trước khi restore, `drift_detected` ⇒ `verified=False`; 3 call-site dùng scope, record lấy **sau** khi scope đóng | `raw/gate_s02_isolation_after.txt` — `4 passed`; `raw/gate_s02_regression.txt` — `40 passed` | run LIMN `10173dc45dea8c90fd2a284b` (legacy, `policy_observed` = `not_recorded`) |
| **S-03** | `FIXED_ALREADY` | `backend/composition.py:631-738`, `:760-777`; `runner.py:1580-1600` | — | không sửa code; thêm test khoá: 3 input v2 + `cache_verify_mode` + `blas_pin_capability` phải nằm trong identity đã hash; pre-v2 ⇒ `unknown`, `group_id=None` | `raw/gate_s03_s04_lock.txt` — `10 passed` | không có |
| **S-04** | `FIXED_ALREADY` | `composition/runner.py:386-500` | — | không sửa code; thêm test khoá: cùng dtype/shape khác bytes ⇒ `None`, dtype khác ⇒ `None`, sai fingerprint ⇒ `None`, file cụt ⇒ `None` kể cả `WB_CACHE_VERIFY=off`, sidecar stale ⇒ `None` | `raw/gate_s03_s04_lock.txt` — `10 passed` | không có |
| **S-05** | `CONFIRMED` (harness), `REFUTED` (lan sang production) | `docs/reviews/20261108T-throughput/d3b_limn_aggregate_scan.py:50` | `raw/gate_s05_limn_conditions.txt`: production **không** trùng — `multimodal` vs `text_only` 179/6003 rank trùng (2,98%) | không sửa production (không có lỗi); ghi rõ scan cũ bị lỗi và **không** được trích làm kết luận | `raw/gate_s05_limn_conditions.txt` — `all_three_conditions_identical: False` | `10173dc45dea8c90fd2a284b` (3 condition khác nhau ⇒ không lan) |
| **S-06** | `FIXED_ALREADY` (assert) + `REFUTED` (diễn giải) | `tests/test_review_defects_evaluator.py:418+`; `tests/test_capability_composition.py:81` | `grep -rn "assert .*or True"` trên `workbench/tests` chỉ còn trong **docstring** mô tả lỗi cũ | không sửa; ghi nhận hai chỗ `or True` đã được thay bằng assert falsifiable | `raw/gate_final_regression.txt` | không có |

## 2. G-01 — chi tiết từng negative bắt buộc

7 negative yêu cầu, tất cả đã thành test trong `workbench/tests/test_gate_contract_negatives.py`:

1. **adapter `{}` + G1 identity** → trước sửa: `PARITY_PASS`, `complete=True`. Sau: `PARITY_INCOMPLETE`,
   `missing = {forward_parity, batch_parity, scoring_parity}`.
   (`metrics_parity` vẫn đo được vì phía native có `native_metrics_match_feature_rerank`.)
2. **delta `0.1` với tolerance `1e-6`, JSON khai `within_tolerance: true`** → trước: `PARITY_PASS`.
   Sau: `PARITY_MANDATORY_SUBCHECK_FAILED`, `declared=1e-06`, `measured=0.1`,
   `harness_recorded_within_tolerance=True` (cờ được **giữ làm bản ghi**, không dùng làm verdict).
3. **model thứ ba** → trước: `EVIDENCED_MODELS = ("limn","encoder")` và tên file chỉ namespace cho
   `encoder`, nên `habit`/`pair` luôn `PARITY_NOT_RUN`. Sau: `_discover_models()` đọc tên file,
   `_artifact_names()` namespace theo model; `habit` có artifact thì được chấm.
4. **malformed / NaN / Inf / ngưỡng mâu thuẫn** → trước: list ⇒ `AttributeError` ra khỏi gate;
   tolerance `inf` ⇒ `PARITY_PASS` với measured `999.0`. Sau: `_Malformed` ⇒
   `PARITY_EVIDENCE_MALFORMED`, không crash. `False`/chuỗi bị loại khỏi so sánh số.
5. **bốn verdict phân biệt** → trước chỉ có `PASS/MANDATORY_FAILED/NOT_RUN/EVIDENCE_MISSING/
   MALFORMED`, thiếu `INCOMPLETE`; `non_tie_count: null` được đọc là `0` (tức "không đổi rank").
   Sau: `PASS`, `INCOMPLETE`, `MANDATORY_FAILED`, `NOT_RUN` là 4 giá trị riêng; `None` ⇒ `MALFORMED`.
6. **chuẩn áp dụng tách khỏi advisory** → trước: `standard` chỉ là tham số, verdict vẫn v1, nhưng
   không có trường nào nói *đã áp dụng* chuẩn nào. Sau: `applied_standard` đọc từ artifact
   (`standard` field), `requested_standard` là yêu cầu, `advisory` ghi rõ `applies: False`.
   Artifact khai v2 nhưng chỉ ghi `max_abs`/`max_rel` ⇒ `MALFORMED` (v2 định nghĩa **theo từng
   phần tử**; không thể áp dụng cho số liệu tổng hợp).
   **Quy tắc giữ nguyên**: `parity-relative/v1` vẫn là chuẩn áp dụng; không retro-apply v2.
7. **tolerance native ≠ adapter** → trước: `merged` giữ native, **bỏ im lặng** adapter, nên bên lỏng
   hơn thắng nếu nó viết trước. Sau: từ chối, `MALFORMED`, `detail` nêu **cả hai** giá trị
   (`forward_max_abs: native 1e-05, adapter 1.0`).

Verdict trên artifact thật **không đổi** sau khi sửa (`raw/gate_head_verdicts_after.txt`):
`limn` vẫn `PARITY_MANDATORY_SUBCHECK_FAILED` (3 failure), `encoder` **16** failure, `passed_models: []`.
Encoder đi từ 13 → 16 vì ba subcheck `relation.{dress,shirt,toptee}.digest` (dimension `relation_parity`)
trở thành **mandatory** ở R02/R03 — artifact cũ không ghi `relation_state`, nên đó là thông tin thiếu được
nêu ra, không phải hồi quy mới (`raw/verify/61-gate-contract.md`, bảng before → after).
Khác biệt duy nhất: `limn` nay báo thêm `missing: ["metrics_parity"]` — artifact LIMN **không** ghi
`native_metrics_match_feature_rerank` (đã kiểm bằng `json.load`), nên chiều đó chưa từng được đo.
Đây là thông tin đúng bị thiếu trước đây, không phải hạ verdict.

## 3. S-02 — pin, drift, và điều đã bị bác bỏ

**Bác bỏ**: audit cũ đo OpenBLAS đứng ở **9** khi code xin **1**. Trên interpreter thực thi của
composition (`ctl312h`, numpy 1.26.4, `libopenblas64_p-r0-0cf96a72.3.23.dev.so`) pin **có** tác dụng:

```
pool_before: 20
record: {'requested': 1, 'previous': 20, 'effective': 1, 'verified': True, ...}
pool_inside: 1
pool_after: 20
```

(`raw/gate_s02_pool_probe.txt`). Con số 9 của audit cũ không tái hiện được ở đây; khác biệt
interpreter/build OpenBLAS là cách giải thích hợp lý nhất nhưng **chưa** chứng minh được, nên ghi là
suy luận chứ không phải sự kiện: `[INFERENCE]` host/build khác.

**Xác nhận**: phần "job đồng thời có thể đổi pool process-global mà không ai biết" là **đúng**.
`_blas_thread_limit` chỉ đọc lại **ngay sau** setter. Một vùng scoring kéo dài hàng phút; nếu thread
khác đặt pool về 10 giữa lúc đó, artifact vẫn khai `verified` — tức **in ra một pin không còn hiệu
lực**, đúng điều cấm.

Trước sửa (`raw/gate_s02_drift_before_after.txt`, chạy module HEAD qua `git show`):
`policy_observed on HEAD: verified`.

Sau sửa: `drift_detected: True`, `verified: False`, lý do nêu rõ "pinned to 1 but read back 10".

Cơ chế đã thêm (`runner.blas_pin_scope`):
- `threading.RLock` bao vùng pin ⇒ hai scope trong cùng process **loại trừ nhau** thay vì tính trong
  pool của nhau (test: hai scope đồng thời, cả hai thấy pool = 1);
- `finally` đọc lại pool **trước khi** restore ⇒ phát hiện mọi thay đổi mà lock không loại trừ được
  (thread khác, thư viện khác, setter trực tiếp), và ghi `drift_detected`/`observed_before_restore`;
- restore **luôn** chạy kể cả khi body raise (test có);
- record được lấy **sau** khi scope đóng, nên `verified` trong artifact là kết quả của cả vùng, không
  phải của mili-giây đầu tiên. Đã áp cho cả 3 call-site: rank pass, backfill, per-query fallback.

Áp dụng cho artifact đã publish: cả 22 run composition hiện có **không** mang `policy_observed`
(legacy), và `policy_observed_from_payload` trả `not_recorded` — **không** backfill, đúng yêu cầu
"thà ghi unverified còn hơn in một pin không xảy ra". Run LIMN `10173dc45dea8c90fd2a284b`: không có
block này ⇒ `not_recorded`.

## 4. G-02 — ĐÃ sửa và rerun (file được cấp quyền ở lượt follow-up)

`workbench/docs/reviews/20261110-e2e-repair/scripts/pc_r3_real_case.py` đã được viết lại. Bản cũ
(`20261110-e2e-repair/`) **không** bị xoá; byte của artifact lịch sử trong `20261110-e2e-repair/`
không bị chạm, chỉ script được sửa tại chỗ vì nó là nguồn sinh số, không phải số. Cách đọc cũ được
đánh dấu **superseded** ngay trong docstring của script mới và ở §4.3 dưới đây.

### 4.1 Ba lỗi của bản cũ, và cách bản mới sửa

| bản cũ | bản mới |
|---|---|
| join target bằng so vector tuyệt đối ⇒ **1/11** target nhận diện được | join bằng **gallery position** ghi trong `inputs.<cat>.json` ⇒ **11/11** |
| `rank_and_order(..., mask_position=None)` mặc định ⇒ rank không mask | cùng một `reference_position` áp cho **cả hai** ma trận |
| crossing đọc từ `order[lo-1:hi]` của một ma trận | crossing theo định nghĩa **pairwise**, kèm mỗi candidate |
| gap = max trên slice, đo trên row **khác** row đã rank | mỗi crosser có `distance_to_target_train/val` **và** `distance_to_cutoff_train/val` trên đúng row đã rank |
| tolerance ngầm, chọn sau khi thấy số | `TIE_EPSILON = 1e-06`, `RANK_AGREEMENT = "exact"` khai ở đầu file, **trước** mọi phép đo |

`TIE_EPSILON = 1e-6` không phải tôi chọn: nó là `tolerances.tie_epsilon` mà
`g1_g2.encoder.*.native.json` đã khai và `encoder_parity.py:50` đã dùng khi đo. `DEEP_CUTOFF = 100`
chỉ là ngưỡng **báo cáo** (R@1/10/50 là các cut đã công bố), không phải tiêu chí pass/fail.

### 4.2 Kết quả rerun

`raw/gate_g02_r3_rerun.txt` (raw, đầy đủ crossing evidence từng candidate):

| category | row | qidx | target pos | ref pos | rank train | rank val | rank adapter | khớp | #crossing |
|---|---|---|---|---|---|---|---|---|---|
| dress | 0 | 0 | 1 | 0 | 3 | 3 | 3 | ✓ | 0 |
| dress | 1 | 3 | 7 | 6 | 1 | 1 | 1 | ✓ | 0 |
| dress | 2 | 14 | 29 | 28 | 101 | 103 | 103 | ✗ | 6 |
| **dress** | **3** | **2016** | **2627** | **746** | **28** | **31** | **31** | **✗** | **3** |
| shirt | 0 | 0 | 1 | 0 | 25 | 21 | 21 | ✗ | 6 |
| shirt | 1 | 2 | 5 | 4 | 21 | 20 | 20 | ✗ | 3 |
| shirt | 2 | 3 | 7 | 6 | 1 | 1 | 1 | ✓ | 0 |
| shirt | 3 | 2037 | 3088 | 3087 | 7 | 11 | 11 | ✗ | 4 |
| toptee | 0 | 0 | 1 | 0 | 1 | 1 | 1 | ✓ | 0 |
| toptee | 1 | 3 | 7 | 6 | 1 | 1 | 1 | ✓ | 0 |
| toptee | 2 | 1960 | 468 | 2763 | 1 | 1 | 1 | ✓ | 0 |

### 4.3 Phán quyết cho `28→31`

**SỐNG.** Con số **giữ nguyên** `28 → 31`, nhưng giờ nó là số **được suy ra đúng** chứ không phải
số may mắn:

- bản cũ báo 1/11 target nhận diện được, nên `28→31` là **dòng duy nhất** có rank — và nó đến từ
  một rank **không mask** cộng một gap đo trên **row khác**;
- bản mới nhận diện **11/11** target, và dòng dress/qidx 2016 vẫn ra đúng `28 → 31` với mask
  2627/746 áp cho cả hai ma trận;
- nghĩa là kết luận "preprocess_train và preprocess_val cho rank khác nhau" **không** phải artefact
  của join hỏng. Nhưng **phạm vi** thì đổi hẳn: không còn là "1 ca duy nhất", mà là **5/11 ca đổi
  rank**, và bản cũ đã bỏ sót 4 ca (dress qidx 14: 101→103; shirt qidx 0: 25→21; shirt qidx 2:
  21→20; shirt qidx 2037: 7→11).

Adapter khớp native_val ở **cả 5 ca đổi rank** (`adapter_ranks_with_native_val: true` toàn bộ 11
dòng) — tức khác biệt nằm giữa hai transform của **native**, không phải giữa native và adapter.
Đây là điều bản cũ **không thể** nói, vì nó chỉ có 1 dòng.

Cách đọc cũ trong `COMPOSITION_MATRIX.csv` / báo cáo `20261110-e2e-repair` được đánh dấu
**superseded by `raw/gate_g02_r3_rerun.txt`**: con số không sai, nhưng mẫu số sai (1 thay vì 11),
nên câu "chỉ 1 ca" phải bỏ.

### 4.4 Test khoá

`workbench/tests/test_gate_r3_mask_join.py` (5 test) khoá bốn quy tắc trên bằng fixture: mask làm
đổi rank, probe phải cùng row, crossing phải pairwise, join phải theo ID/position. Test cuối đọc
npz thật và khẳng định **1/11** — con số chứng minh vì sao join-vector là sai.
## 5. S-05 — lỗi harness, và vì sao không lan

Harness: `d3b_limn_aggregate_scan.py:50` gọi
`adapter.encode_queries([str(p) for p in paths], [q.query_txt for q in selected], ...)` **trong**
`for cond in CONDS` mà không hề gọi `_stage_query_path`/`adapt_query_inputs` (grep = 0 lần trong
file). Ba nhãn nhận **cùng** ảnh và **cùng** caption. Hệ quả: cả ba condition ra cùng rank, nên bảng
"changed 0 / ΔR@10 = 0 / ΔnDCG = 0" là hiển nhiên, không phải phát hiện.

Đối chứng: `blas_three_adapters.py` (bản đã sửa) **có** `_stage_query_path`, và bản `_FLAWED.py`
không có — tức lỗi đã biết và đã sửa ở harness ENCODER, chỉ file LIMN là còn sót.

Production: **không** lan. Đo trên rank thật của run `10173dc45dea8c90fd2a284b`:
`multimodal` vs `text_only_black_image` chỉ trùng 179/6003 (2,98%); `text_only` vs `image_only`
120/6003. Nếu lỗi lan, cả ba sẽ trùng 6003/6003. Production đi qua `_stage_query_path`
(`runner.py:668`) — đúng đường bị thiếu ở harness.

Vì vậy kết luận "LIMN không nhạy BLAS" trong `BAO_CAO_CUOI.vi.md:607`, dẫn từ
`d3b_limn_aggregate_scan`, **không đứng vững như một phép đo ablation** — nhưng cơ chế nó viện dẫn
(LIMN chấm từng row nên không có chiều M cho BLAS) vẫn là một lập luận độc lập, và **không** bị bác
bỏ ở đây. Ghi nhận: cần rerun ablation qua `_stage_query_path` nếu muốn giữ câu đó như số đo.

## 6. S-06 — những gì code bác bỏ

- `assert ... or True`: **đã sửa trước** campaign này. `grep -rn "assert .*or True"` trên
  `workbench/tests` chỉ còn trong docstring mô tả lỗi cũ
  (`test_review_defects_evaluator.py:13,418`, `test_capability_composition.py:81`). `FIXED_ALREADY`.
- "test handoff chỉ assert counts tồn tại": không tái hiện được trong phạm vi file tôi sở hữu;
  các test composition hiện kiểm identity/digest thật (`test_composition_provenance_identity.py`).
  Không kết luận thay phần ngoài phạm vi.
- Diễn giải: `unresolved@10` **không** phải random baseline — `metrics.shortcut_outcomes` đếm
  `unresolved` là "không condition nào hit trong K", một tập con của cohort, không phải mẫu ngẫu
  nhiên. Không có chỗ nào trong `metrics.py` gọi nó là baseline. `REFUTED` cho cách đọc đó.
- Chênh paper ≠ "chủ yếu do protocol": không có code nào khẳng định điều này; `metrics.py` chỉ tính
  số. Không sửa.

## 7. Test đã thêm (5 file, 49 test)

| file | test | nội dung |
|---|---|---|
| `tests/test_gate_contract_negatives.py` | 25 | 7 negative bắt buộc của G-01 |
| `tests/test_gate_r3_mask_join.py` | 5 | G-02: mask, probe-row, crossing, join-by-ID, 1/11 |
| `tests/test_composition_blas_pool_isolation.py` | 4 | S-02: drift, restore khi raise, scope sạch, hai scope đồng thời |
| `tests/test_composition_backfill_refusal.py` | 5 | S-01: refuse khi rank lệch, `None`, `{}`, thiếu condition |
| `tests/test_composition_cache_and_identity_lock.py` | 10 | S-04 (7) + S-03 (3) |

Kết quả: `243 passed` trên 20 file liên quan (`raw/gate_final_regression.txt`), `70 passed` trên
API/browser-facing (`raw/gate_api_regression.txt`). Không chạy full suite.

## 8. File đã sửa

- `workbench/backend/parity_gate.py` — validate, verdict từ số, `INCOMPLETE`, chuẩn áp dụng, từ chối
  tolerance lệch, discovery model thứ ba.
- `workbench/composition/runner.py` — `blas_pin_scope()`, lock nội process, phát hiện drift, 3
  call-site dùng scope, record lấy sau scope.

Không sửa `native_measurements.py`, `limn_diagnostic.py`, `model_limn.py`, `frontend/**`.

### 8.2 File sửa ở lượt follow-up

- `workbench/docs/reviews/20261110-e2e-repair/scripts/pc_r3_real_case.py` — viết lại theo bốn quy tắc
  ở §4.1 (được cấp quyền tường minh ở lượt follow-up).

Không sửa/xoá byte nào của artifact lịch sử trong `20261110-e2e-repair/`; chỉ script đổi, và cách
đọc cũ được đánh dấu superseded trong docstring + §4.3.

## 9. Gate change có live-safe không

`parity_gate.py` đã đổi, service trên `:8000` vẫn giữ module **cũ** trong bộ nhớ. Tôi **không**
restart (service thuộc `web-api`).

Bằng chứng: `raw/gate_live_safety.txt` — so JSON đang được phục vụ (`GET /api/runs/parity-gate`,
**HTTP 200**) với verdict tính trực tiếp trong process bằng module mới.

**Điều KHÔNG đổi** (mọi thứ UI branch tới):

| | served | local |
|---|---|---|
| `GET` status | 200 | — |
| `standard` | `parity-relative/v1` | `parity-relative/v1` |
| `passed_models` | `[]` | `[]` |
| `encoder.status` | `PARITY_MANDATORY_SUBCHECK_FAILED` | giống |
| `encoder.failure_count` | 13 | 13 |
| `encoder.passed` | `false` | `false` |
| `limn.status` | `PARITY_MANDATORY_SUBCHECK_FAILED` | giống |
| `limn.failure_count` | 3 | 3 |
| `limn.passed` | `false` | `false` |

**Điều THÊM** (không phải đổi): `missing` (mới), `applied_standard`/`requested_standard` (mới),
`limn.checks` 48 → 51 (3 check `scores.max_abs_vs_dot` mà module cũ bỏ qua vì đọc sai khoá LIMN).

### 9.1 Câu hỏi: `missing: ["metrics_parity"]` của LIMN có đổi count nào trên web?

**Không.** Đã đọc `EvaluationPage.tsx:50-125` — card G1/G2 chỉ dùng:
`gate.standard`, `gate.passed_models.length`, `gate.models[*].{model_id,passed,status,failure_count,failures,dimensions}`.
Không chỗ nào đọc `missing`, `complete`, `applied_standard`, `requested_standard`, `coverage`,
`advisory`, hay `checks.length`. Vì vậy:

- badge "KHÔNG mô hình nào PASS (0/2)" — không đổi;
- cột "Subcheck FAIL" — không đổi (3 và 13);
- "Chiều đạt" — không đổi.

`missing` là **thông tin mới, đúng, và bị phục vụ nhưng chưa render**. Nếu sau này UI thêm cột đó,
`encoder` hiện `[]` còn `limn` hiện `["metrics_parity"]` — và đó là sự thật: artifact LIMN không ghi
`native_metrics_match_feature_rerank`, nên chiều đó chưa từng được đo. Không có measurement nào đổi
để tạo ra nó.

### 9.2 Lệnh restart cần (để anh sequence với `web-api`)

```bash
# PID hiện tại của service :8000 (đọc, không kill ở đây)
ss -ltnp 'sport = :8000'
# hoặc: pgrep -af "uvicorn workbench.backend.main:app"

# lệnh restart (web-api chạy, không phải tôi):
kill <PID_8000>
cd /home/heheboiz/data/cir && \
  workbench/envs/ctl312h/bin/python -m uvicorn workbench.backend.main:app \
    --host 127.0.0.1 --port 8000
```

`cwd` phải là `/home/heheboiz/data/cir` (đã đối chiếu `/proc/<pid>/cwd` của service hiện tại trong
`BASELINE.vi.md:121`). Không có biến môi trường đặc biệt nào cần thêm; `workbench.env` giữ nguyên.

Sau restart, kiểm lại **đúng** hai lệnh này và so với `raw/gate_live_safety.txt`:

```bash
curl -s -o /tmp/live_gate_after.json -w '%{http_code}\n' http://127.0.0.1:8000/api/runs/parity-gate
workbench/envs/ctl312h/bin/python \
  workbench/docs/reviews/20261111-master-audit/scripts/gate_live_safety.py /tmp/live_gate_after.json
```

Kỳ vọng: `HTTP 200`, `passed_models: []`, `failure_count` 13 và 3, và `ALL_COMPARED_FIELDS_IDENTICAL=True`
(ba trường mới sẽ khớp vì cả hai phía đều là module mới).

## 10. Còn lại

- **S-02**: pin đã đo được trên host này; artifact legacy vẫn `not_recorded`. Một run mới sẽ mang
  `policy_observed` đầy đủ. Không rerun GPU ở đây.
- **S-05**: nếu muốn giữ câu "LIMN không nhạy BLAS" như số đo, phải rerun ablation qua
  `_stage_query_path`; hiện nó chỉ là lập luận cơ chế.
- **G-02**: đã xong. `28→31` sống, phạm vi mở rộng 1 → 5/11 ca; xem §4.3.
