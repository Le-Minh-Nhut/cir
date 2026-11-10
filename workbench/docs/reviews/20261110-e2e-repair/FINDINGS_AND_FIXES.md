# FINDINGS_AND_FIXES — ledger 14 dòng (W1, R13, R1–R12)

Lượt: 2026-11-10, HEAD `2e2e2d299e0cd3f27d6bcaff09fdf64d7c778dc6`, nhánh `research/cir-failure-workbench`.
Baseline: `BASELINE.md` cùng thư mục. Bằng chứng thô: `raw/`.

## 0. Quy ước cột — đọc trước khi dùng bảng

Có **hai** cột status, và chúng không được gộp:

- `status` = kết luận trên **HEAD `2e2e2d2`**, tức bytes đã commit.
- `status_working_tree` = kết luận trên **working tree lúc 08:30**, nơi 16 file tracked đã bị sửa
  và 11 file untracked xuất hiện (`git status --short`). Mốc 07:57 trong `BASELINE.md` §9 chỉ ghi
  đúng một file trong số đó (`limn_diagnostic.py`, 218 → 615 dòng lúc đó); tới 08:30 nó còn
  `main.py`, `ModelInventory.tsx`, `LimnScopeDiagnostic.tsx`, `composition.py`, `evidence.ts`-adjacent
  frontend và bốn file test/module mới. **Không** dùng mốc 07:57 để nói về trạng thái hiện tại.

Lý do phải tách: campaign này **đã từng** kết luận sai đúng theo kiểu này — một phát hiện đúng trên
bản cũ bị tuyên là "còn lỗi" sau khi bản mới đã vá, và ngược lại. `raw/r6_guard.txt` đo HEAD. Với R5/R6, cột working tree dựa trên **đọc source** cho tới hết lượt;
với R6, lượt này **đã có** probe hành vi trên service sống (`:8000`, process mang code mới) — kết quả
ở §3.3b, nên cột working tree của R6 nay có **cả** source **và** số đo. R5 vẫn chỉ có source +
artifact đã sửa; `raw/r6_working_tree.txt` không tồn tại và không cần nữa.

`NOT_MEASURED` trong cột bằng chứng = **không có** số đo, và tôi nói rõ cái gì sẽ kết luận được nó.
Không có số nào trong file này là suy ra hay làm tròn.

**Sở hữu `raw/` và các thư mục bằng chứng.** Ledger này trích bằng chứng từ **năm** nguồn trong cùng
thư mục, và mỗi trích dẫn ghi rõ nguồn:

| nguồn | ai tạo | ledger dùng cho |
|---|---|---|
| `raw/*.txt`, `raw/*.json`, `raw/audit_*.py` | worker lượt này | baseline, census, R2/R3/R4, R9 |
| `raw/guards/**` | worker R5/R6/R8/R9 | R5, R6, R8, R9 (`r9_after.*`) |
| `raw/r12/**` | worker R12 | R12 |
| `raw/r1_*.json`, `raw/clip4cir_fullft.migrated.json*`, `raw/r2_*`, `raw/r3_*`, `raw/r4_*` | worker Phase A/C | R1, R2, R3, R4 |
| `web_evidence/**` | worker W1/R13/R7 | W1, R13, R7, §6 display contract |

Bản đầu của §0 ghi "các file còn lại không phải bằng chứng của tôi và không được trích" — câu đó **sai
với chính ledger này**, vì §1 trích `raw/guards/`, `raw/r12/`, `web_evidence/`. Đã sửa thành bảng
trên. Điều **vẫn** đúng: `raw/scout/`, `raw/*.png` trong `web_evidence/` chỉ dùng làm ảnh chụp, và mọi
số đều truy được về một file cụ thể trong bảng đó hoặc trong `EVIDENCE_INDEX.md`.

## 1. Bảng tổng hợp

| ID | Claim đang kiểm | status (HEAD) | status_working_tree | commit | Lệnh / test tái lập | Input / run ID | Kỳ vọng vs quan sát | Before-evidence | After-evidence | Phán quyết remedy | Files changed | Run/cache impact | Regression |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **W1** | "Web reports 0. Open the URL with FashionIQ Original + SF; reconcile every ID across API, catalog and table, then try CIRCUS Release v1 + Full. Remedy: filter by the correct protocol/subset, show scope next to counts, links change both protocol and subset, default CIRCUS pages to Full when the URL names no scope, and show a link instead of 'needs run' for a model that has results in another scope." | **CONFIRMED** | **FIXED một phần** (source + bundle; xem cảnh báo build §1.2) | `2e2e2d2` | `web_evidence/before-bare-url-probe.json` (trước) · `after-landing-probe.json`, `after00/01.txt|png` (sau) · `pytest workbench/tests/test_web_scope_and_labels.py` → 12 pass | URL trần `/`; `?protocol=fashioniq_original_split`; `?protocol=circus_fashioniq_release_v1&subset=validated` | URL trần kỳ vọng → CIRCUS **Full**; HEAD quan sát `fashioniq_original_split&subset=sf`, `subset_select: 0`, "0 lượt có dữ liệu ở phạm vi này". **Không ID nào lệch**: `/api/models` 17, `/api/benchmark-runs` 13 (tất cả `fashioniq_original_split`), `/api/composition/runs` 35 (tất cả `circus_fashioniq_release_v1`: full 17 / sf 9 / validated 9), bảng sf 9 row | `web_evidence/before-*.png` (6 ảnh) + `before-bare-url-probe.json` | `after-shots.log` dòng 0 = `?area=overview&protocol=circus_fashioniq_release_v1&subset=full`; `after01.txt` = "0 lượt composition (protocol này không có) · 13 báo cáo tổng hợp" + card "Số đo native (chưa xác minh)" 3 file | **phù hợp**, trừ một lỗi remedy không nhắc: banner `MOCK DEVELOPMENT DATA — no real evaluation results are loaded` vẫn in khi region còn tải (`after00.txt:37`; `evidence.ts:141,163,178-181` — `noRealEvidence = aggregateRuns===0 && experimentRuns===0` với `region.data ?? []`; `evidenceBanner` **thiếu nhánh loading**). Đúng lớp lỗi §3bis.a đã sửa cho `scope-strip` | `frontend/scope.ts` (mới), `routes.ts`, `App.tsx`, `ModelInventory.tsx`, `tests/test_web_scope_and_labels.py` (mới) | 0 (chỉ scope landing) | không chạy full suite; 51 test frontend pass (`web_evidence/after-tests.log`) |
| **R13** | "A native run has a raw artifact but is missing on the web. Remedy: fix the config, ingest through the official path, or record the measurement in an explicitly unverified catalog with provenance; never fabricate a receipt for raw JSON lacking evidence and never put such a number in the verified leaderboard." | **CONFIRMED** | **FIXED một phần** — catalog đúng thái độ, thiếu 4 hình dạng macro + 4 model | `2e2e2d2` | `GET /api/native-measurements`; `backend/native_measurements.py`; `web_evidence/r13-native-catalog.txt` | 34 file measurement native ngoài mọi đường ingest; 15 file mang số | Kỳ vọng số native lên web **có provenance, không thành receipt**. HEAD: **0 endpoint**, web đọc 0 số cho `fashioniq_val_split` và `…_ref_excluded` dù 15 file có số. Sau: `{files 21, with_metrics 15, unreadable 1, signposted_blockers 4, verified 0}`, 12 `model_id` | `web_evidence/r13-native-catalog.txt` | mỗi row có `source.path`/`sha256`/`bytes`, `verified: false`, `verification.state=UNVERIFIED_NATIVE_MEASUREMENT`; **không** tạo `completion.json`/`invocation_outcome.json`; **không** vào `/api/evidence/summary` (`counts.verified = 0`) | **phù hợp về thái độ, thiếu về độ phủ.** `METRIC_KEYS` chỉ đọc `aggregate.{macro_r10,macro_r50,macro_mean}` + 2 glob ⇒ (a) `ptha_transfer_run1.json` trong glob nhưng số ở `macro_metrics` ⇒ `metrics: null`; (b) `clip4cir_parsed.json` bị `_is_measurement()` loại ⇒ mất **cả** `combiner_rn50x4_noft` và `clip4cir_rn50x4_fullft`; (c) `pair_compat/measurements/*.json` và `csmcir/measurement.json` ngoài glob. **Cùng lỗi census v1** (`raw/r13_census_v2.md`) | `backend/native_measurements.py` (mới), `backend/main.py`, `frontend/data.ts` | 0 (không sửa số nào) | không |
| **R1** | "CLIP4Cir FullFT: read the FashionIQ branch of the pinned upstream, the command, gallery, masks and actual log. Check whether a reference case ranks above its own target. Remedy: if it ran correctly and only the label is wrong, fix provenance with a verified migration keeping history and original metrics; if the ranking path is wrong, rerun under the pinned contract. Reconcile membership before calling anything Full Gallery." | **CONFIRMED** — lỗi provenance, **không** lỗi ranking | **FIXED** (nhãn) + 1 khiếm khuyết còn lại ở `protocol_source` | `2e2e2d2` | `scripts/migrate_protocol_label.py`; `pytest workbench/tests/test_protocol_label_migration.py` → 5 pass | `raw/native/clip4cir_fullft.json`; checkpoint `rn50x4_fullft_unresolved` | Kỳ vọng `fashioniq_original_split`; quan sát `fashioniq_full_gallery_ref_excluded`. `validate.py:21-57` `compute_fiq_val_metrics` **không có** lệnh mask nào; `reference_mask` chỉ ở `:163-165` trong `compute_cirr_val_metrics` (`:135`). Gallery 3817/6346/5373 = full split. `models.yaml:1184` đã đúng | `raw/native/clip4cir_fullft.json` (nhãn cũ) | `raw/clip4cir_fullft.migrated.json` (sha256 `fb97cfab…` → `dd9176cf…`; `aggregate.macro_r10` giữ `43.8016931215922`; `categories` không đổi) + `.migration.json` + `raw/r1_migration_evidence.json` | **phù hợp, với một lỗi của chính remedy:** `protocol_source` nằm trong `FROZEN_FIELDS` (`migrate_protocol_label.py:33-34`) và bị test khoá (`test_protocol_label_migration.py:57`), nên câu **sai** "reference is excluded by masking its own index entry" vẫn còn trong bản đã migrate. Reference **là** ứng viên hợp lệ, không mask nào loại nó ⇒ nhãn `…_ref_excluded` không mô tả phép chấm nào đã chạy | `scripts/migrate_protocol_label.py` (mới), `tests/test_protocol_label_migration.py` (mới) | 0 — không rerun, không sửa metric | không chạy full suite |
| **R2** | "ENCODER G1: compare the real native path with the adapter on identical queries. Save tensors from native `preprocess_train`, the adapter, and a `preprocess_val` control. Check which transform each category actually used. Remedy: separate native fidelity, same-input control and CIRCUS adaptation; a control with identical pixels only validates that check." | **CONFIRMED** — bất đối xứng thật, ở phía **native** | **CONFIRMED** (không sửa; protocol divergence, không phải bug) | `2e2e2d2` | `scripts/pc_r2_transform_tensors.py` (interpreter `encoder310-adapted`) | `data/FashionIQ/resized_image/dress/B0007WIZYE.jpg` (256×256) | Kỳ vọng control = 0 và tách được ba thứ. Quan sát: `control_val_vs_val` **0.0**; `preprocess_val` lặp lại **0.0**; `preprocess_train` lặp lại **3.8270**; train-vs-val **3.8270** | `raw/r2_transform_tensors.json` | `raw/r2_transform_tensors.json` (repr đầy đủ hai transform) | **phù hợp.** Transform theo category, đọc source: `datasets.py:139,141,177,178` `stage=0` (=train) cho **query**; `:195,202` `stage=1` (=val) cho **gallery**; target của query trong nhánh val-split cũng đi `:178` stage=0. Adapter dùng `preprocess_val` cho **mọi** ảnh (`model_encoder.py:329,348`). Hệ quả: so sánh sample-level adapter-vs-native-train-transform **không thể** là parity test — đúng 2 trong 4 block `False` của encoder | không | 0 | không |
| **R3** | "ENCODER harness: check sort/mask in the parity harness. Reproduce with reference 1.0, another image 0.8, target 0.5→0.5000002, candidate 0.5000001. Remedy: use the same eligibility mask for ranking and the tie probe; compare the candidate IDs that actually cross between the two score arrays and at the cutoff; do not declare two real cases tie/non-tie when the data is insufficient." | **CONFIRMED** cơ chế mask; **DISPROVED** "hai ca thật là tie" | **CONFIRMED** (harness chưa sửa) | `2e2e2d2` | `scripts/pc_r3_tie_harness.py`; `scripts/pc_r3_real_case.py` | ca đồ chơi 1.0 / 0.8 / 0.5 / 0.5000001; ca thật `native.{cat}.npz` + `adapter.{cat}.npz` | Kỳ vọng rank và probe dùng **cùng** mask. Quan sát: rank không mask **4**, có mask **3**; gap không mask **1.192e-07**, có mask **1e11**; chỉ vị trí `0` thực sự vượt; bản 3 phần tử #3→#2 cùng delta. Ca thật dress: rank **28** vs **31**, gap nhỏ nhất **1.559e-3** / **3.639e-4** — lớn hơn tie_epsilon 1559×/364× | `raw/r3_tie_harness.json`, `raw/r3_real_case.txt` | cùng hai file | **phù hợp về cơ chế, không đủ để kết luận tie.** Sửa harness làm phép đo trung thực hơn nhưng **không** cứu verdict "tie". Phạm vi: chỉ **1/4** target của sample định danh được một hàng gallery ⇒ **một** ca, không phải population | không | 0 | không |
| **R4** | "ENCODER random relation: check `state_dict`, checkpoint keys, seed timing, and the tensor after device transfer. Compare two clean processes on identical input. Remedy: if the checkpoint stores no relation, define a reproducible init policy compatible with the objective; seed before construction or persist the drawn tensor with a digest; keep the new policy separate from history and put it in run identity/cache. A control `arange` tensor is not a released weight." | **CONFIRMED**, nặng hơn mô tả (thêm 2 key lạ) | **CONFIRMED** (chính sách đề xuất, chưa áp dụng) | `2e2e2d2` | `scripts/pc_r4_relation.py --seed 0 --repeat 2`; `scripts/pc_r4_determinism.py` (interpreter `encoder310-adapted`) | checkpoint `fashioniq.pt` `0f31c501…9173`; hai process sạch | Kỳ vọng relation không vào `state_dict`. Quan sát: **417** key checkpoint, **0** key chứa `relation`; **415** state-dict entry; `is_parameter: Tensor`; `named_parameters()` chứa relation **[]**; `requires_grad True` nhưng không vào optimizer; device `cuda:0` | `raw/r4_determinism_p1.json`, `_p2.json`, `r4_determinism.raw.log`, `r4_relation_p1/p2.json` | idem — `seeded_same_seed_max_abs: 0.0` (nhân quả); không seed: **4.580 / 4.678 / 4.923**; khác seed: **4.862** | **phù hợp.** Nguồn `third_party/ENCODER/model_try2.py:93` — `.cuda()` trên `nn.Parameter` trả `Tensor` thường. Số không-seed **không phải hằng số** (4.58/4.68/4.92; trị "~4.2" của operator không tái lập). Thêm: `unexpected_keys` = `backbone.binding_decoder.fc.0.{bias,weight}` ⇒ `miss=0, unexpected=2`, `strict=False` che mất. `seeded_dataset_construction` **không** che được vì seed **sau** `Encoder(...)` | không (Phase C không được tự đổi khoa học) | 0 — nhưng run identity **nên** mang seed/digest; `identity.seed` chỉ có ở run `10173dc4`, không có ở `9e4918a6` | không |
| **R5** | Summary ghi `target_did` của query khác | **CONFIRMED** | **ALREADY_FIXED** | `2e2e2d2` | `raw/audit_r5_queries.py`, `raw/audit_r5_reconcile.py` | run `9e4918a6b5c243412cbc58c9`; qid `7:3038`, `7:3790` | kỳ vọng `7:3416`; quan sát `7:42304` ở **cả 12** ô | `raw/r5_queries.txt`, `raw/r5_reconcile.txt` | source working tree `limn_diagnostic.py:191-200` | **phù hợp** | `t2_dual_scope.py:273` (chưa sửa) | 0 — chỉ 1 field summary | không chạy |
| **R6** | Endpoint nhận input dị dạng | **CONFIRMED** | **ALREADY_FIXED** (source) / probe NOT_MEASURED | `2e2e2d2` | `raw/audit_r6_guard.py` trên bản copy `/tmp` | 12 biến thể, xem §3 | nhận 8/12 loại sai; 1 loại **crash** | `raw/r6_guard.txt` | `limn_diagnostic.py:316,328,349,363,402` | **phù hợp** | working tree `limn_diagnostic.py` | cache theo content digest, mới | không chạy |
| **R7** | "CIRCUS subsets mix run generations. Reconcile `source_run_id`, query membership, scoring policy, denominator and candidate exports across Full/SF/Validated. For one parent run, a subset cannot have more unscored queries than the parent. Remedy: derive the subset from verified per-query artifacts when the protocol permits; record the parent run and membership hash; if semantics differ or artifacts are missing, relabel honestly as historical. Do not replace coverage with an inference from a screenshot." | **NOT_REPRODUCED** cho "subset trộn thế hệ"; **CONFIRMED** cho "subset không khai báo nguồn" | **FIXED** (khai báo); số phục vụ không đổi | `2e2e2d2` | `GET /api/composition/runs?subset={full,sf,validated}`; `backend/subset_provenance.py` | 17 run × 3 subset; 3 run kiểm trực tiếp trên `multimodal.jsonl` | Kỳ vọng không subset nào vượt parent. Quan sát 17 run: **không** run nào có `sf_total > full_total` hay `sf_unscored > full_unscored`; `sf ⊂ full`, `validated ⊂ full`, `validated ⊂ sf` đều True; R@10 phục dựng khớp số phục vụ tới **1e-9** trên mọi row có metric (35/35; `e0dd04a46ef7` FAILED có `scored=0`, `agree=None`) | `web_evidence/r7-api-counters.json` (trước: không có khoá provenance) | `web_evidence/r7-subset-provenance.txt`: mỗi row có `parent_run_id`, `membership_sha256`, `query_count/scored/unscored`, `containment`, `metric_check` (served vs recomputed), `consistent`, `reasons[]`; 0.29 s/35 row cold, 0.20 s warm | **phù hợp.** Phải nói kèm: ba thế hệ `run_id` là **run khác nhau** và khác nhau về routing, không phải bản sao — `limn` `9e4918a6` (unscored 2) vs `10173dc4` (unscored 119 = đúng 119 `ambiguous_qids`); xem FINAL_REPORT §5.5. `ndcg_full` cố ý **không** phục dựng | `backend/subset_provenance.py` (mới), `backend/main.py` | 0 — chỉ thêm khoá | không |
| **R8** | LIMN chỉ hiện text và Top-20 | **CONFIRMED** | CONFIRMED (chưa ai sửa UI) | `2e2e2d2` | `raw/r8_probe.txt`; `grep -n depth=20 LimnScopeDiagnostic.tsx` | endpoint `/api/composition/diagnostics/limn-dual-scope/candidates` | kỳ vọng ảnh + K chọn được; quan sát `depth=20` cứng, 0 thẻ `<img>` | `raw/r8_probe.txt` | – (không sửa) | **cần sửa phương án** | `LimnScopeDiagnostic.tsx:288` | 0 | không chạy |
| **R9** | Gate PASS khi subcheck fail | **CONFIRMED** | **ALREADY_FIXED (đã đo: cả limn và encoder FAIL)** | `2e2e2d2` | `raw/r9_subchecks.txt`, `raw/r9_encoder_numbers.txt`, `raw/r9_encoder_false_flags.txt`, `raw/r9_denominator.txt` | `g1_g2.{shirt,toptee,dress}.adapter.json`, `g1_g2.encoder.*.adapter.json` | kỳ vọng PASS ⇔ mọi subcheck mandatory pass; quan sát **4** cờ `False`/category (encoder) và **1** (limn) | `raw/r9_subchecks.txt`, `raw/r9_encoder_false_flags.txt` | `workbench/backend/parity_gate.py` (untracked, worker khác) | **cần sửa phương án** | `parity_gate.py` (mới), `tests/test_gate_parity_subchecks.py` | 0 | không chạy |
| **R10** | "LIMN category is ambiguous. Rebuild the CIRCUS source with the pinned upstream generator, annotations and duplicate lineage. Separate the representative row's category from the unique category of a merged query. Check `7:3038` and `7:3790`. Remedy: assign a canonical scope only when the contract permits that provenance and the evidence is clear; otherwise keep partial with the canonical denominator and show shirt/toptee diagnostics separately. Never choose scope by rank/score or infer category from the target." | **CONFIRMED** — giữ PARTIAL 6001/6003 | **CONFIRMED** (không chọn scope nào; chỉ thêm diagnostics) | `2e2e2d2` | đọc `limn_dual_scope.json`; `raw/guards/r5_reconcile.py` cho 12 ô chẩn đoán | run canonical `9e4918a6b5c243412cbc58c9`; qid `7:3038`, `7:3790` | Kỳ vọng: không gán category khi provenance không cho. Quan sát: `canonical_status.status = PARTIAL`, `scored 6001`, `unscored 2`, `unscored_qids = [7:3038, 7:3790]`, `canonical_cohort.full.query_count = 6003`; `gallery 74381` với note "the FULL released gallery for every scope: no category filtering"; `loader_category = null` cho cả hai qid; `rank_convention_crosscheck` 36/36 `all_agree` | `limn_dual_scope.json` | 12 ranking chẩn đoán tách riêng (`7:3038` shirt 3/88/152, toptee 43/389/67; `7:3790` shirt 65/6891/2200, toptee 43/4730/4047) | **phù hợp** ở vế "không chọn scope". **Nhưng** con số canonical phụ thuộc một routing policy: run `10173dc4` (2026-11-08, unscored 119 = đúng tập `ambiguous_qids`) làm nó hiện rõ, còn `9e4918a6` chỉ để lại **digest** `query_scope_resolution` — tức **117 query mơ hồ đã được chấm**, mỗi query phải chọn một trong ba embedding space, và lựa chọn đó không đọc được. Cả hai row đều được phục vụ cạnh nhau | không | 0 — canonical metrics không đổi | không |
| **R11** | "Ties and the BLAS attribution. Compute margin to neighbouring candidates and to the cutoff on the same eligible set. Isolate each variable in A/B: embeddings, rank code, interpreter, precision, thread count. Remedy: re-measure on the same pinned data and policy; margin to #1, identical source hash, or R@10/R@50 moving in opposite directions are each insufficient to conclude a cause; publish the scope of each measurement." | **CONFIRMED** phạm vi; **DISPROVED** là nguyên nhân cho `encoder` | **CONFIRMED** (attribution; không metric nào đổi) | `2e2e2d2` | đọc `COMPOSITION_MATRIX.csv` cột `blas_pin`; `20261108T-throughput/d3b_encoder_aggregate_scan.txt` | `encoder` `6a528e87`→`5dee7425`; `conesep`; `limn`; `ptha_mtst` | Kỳ vọng A/B chỉ đổi thread. Quan sát `encoder`: `changed=1`, `max_drank=1`, **r10_delta 0.0**, r50_delta 0.0, ndcg_delta 2.18e-05, `cross_k10/50/100 = 0`, ở **cả ba** condition. `conesep`: 17/18009 rank đổi 1 bậc, R@10/R@50 y hệt, cùng interpreter. `limn`: `changed=0`, delta 0.0 | `d3b_encoder_aggregate_scan.txt`, `COMPOSITION_MATRIX.csv` | idem | **phù hợp. `encoder` loại trừ BLAS**; nguyên nhân thật là `adapter_source_hashes.rank` đổi (`5f828b13…`→`f0914b12…`) trong khi `score` giống hệt (`0ab4fe1d…`), cộng interpreter đổi. `ptha_mtst` có 12/20/14 rank đổi 1–2 bậc (cùng lớp 1e-7) nhưng **đường encode UNPINNABLE** (`mkl_set_num_threads` segfault exit 139). Phạm vi: **chỉ** 3 model + ptha; phần còn lại **BLOCKED** | không | 0 | không |
| **R12** | "Native models still missing and slow runs. Re-confirm the four models plus Combiner. Build the correct env version, check actual checkpoint/class/shape and entrypoint. Remedy: try a separate env and a wrapper calling the original evaluator for Air-Know/CLVC/TG-CIR; for PAIR trace architecture/checkpoint mapping rather than dropping parameters to get a number; probe parity before moving DCNet/CLIP4Cir to GPU; keep the commands tried and the real blocker." | **CONFIRMED** — 4 BLOCKED + 1 NOT_A_BLOCKER | **CONFIRMED** (không sửa code; ghi blocker thật) | `2e2e2d2` | `BLOCKERS_R12.vi.md`; `raw/r12/*.log|.txt|.json` | `airknow` `fiq_n05`; `clvc_net`; `tgcir`; `pair` `b1/b2`; `combiner_rn50x4_noft` `…020a70b0` | Kỳ vọng mỗi model có lý do đo được. Quan sát: `airknow` interpreter **mất** (symlink gãy tới `teacher-sprc`) + **11** biểu thức `.cuda()` cứng trên scoring path; lỗi import đầu tiên là `torchaudio` (**không** phải `moviepy`, dòng 11); `clavis` là **false positive grep nhị phân** (`0x63`=`c`=GLOBAL); checkpoint load **được** 863 tensor / 1 196 753 657 param. `clvc_net`: 14 dòng/15 biểu thức `.cuda()`, **6 dòng/7** trên scoring path, không entrypoint. `tgcir`: không có đường nạp checkpoint phát hành (`--restore_file` khai `:74`, **không dùng**), eval nằm trong training loop `:200`. `pair`: pickle cần `model_vssp.*` + `MHTransformer_dis.*` **không có** trong repo pin. `combiner`: `EXECUTION_COMPLETED` nhưng `METRIC_REFERENCE_UNVERIFIED` | `raw/r12/airknow_import_lavis_ctl312h.log`, `airknow_pkl_globals.txt`, `pair_pkl_globals.log`, `combiner_gate_full.txt` | `airknow_fullload.log` (LOAD_OK CPU), `combiner_completion.txt` (PROOF_VALIDATED), `combiner_verify_run.json` (`is_research_evidence: false`) | **phù hợp.** Thêm điều chưa nêu: `combiner` **vĩnh viễn** không thành paper evidence vì `unbenchmarked() == True` — đóng đúng cách **không phải** nới gate. Ba claim của campaign trước bị sửa: thứ tự lỗi `airknow`; `tgcir` "không có entrypoint"; `combiner` "NOT_A_BLOCKER" thiếu vế paper-evidence | không | 0 — không có job GPU nào; baseline `nvidia-smi` 430 MiB, 0 process compute | không |

**Đếm (sau khi điền đủ claim text):** 14 CONFIRMED — 9 CONFIRMED-đã-sửa, 3 CONFIRMED-cần-sửa-phương-án, 1 CONFIRMED-BLOCKED-một-phần (W1), 1 CONFIRMED-một-phần (R13), 1 CONFIRMED-phạm-vi-hẹp (R11). 0 BLOCKED vì briefing. 0 ALREADY_FIXED ở cột HEAD; ở cột working tree:
R5, R6, R9 = ALREADY_FIXED — **cả ba đều là đọc source, chưa có probe hành vi**.

### 1.1 Ba thay đổi đồng thời của worker khác (không phải của tôi)

`git status --short` lúc đóng lượt:

```
 M workbench/backend/limn_diagnostic.py                  (218 -> 615 dòng)
 M workbench/tests/test_limn_dual_scope_diagnostic.py
?? workbench/backend/parity_gate.py                      (mới, cho R9)
?? workbench/tests/test_limn_diagnostic_record.py
?? workbench/tests/test_gate_parity_subchecks.py
```

`parity_gate.py` tự khai ở đầu file: "Before it, the G1/G2 verdict lived only in hand-written
surfaces (`GATES.csv`, `ENCODER_PARITY.vi.md`, `FINAL_REPORT.vi.md`)" và nó liệt kê đúng các cờ
`False` mà §5.3 của tôi đo được. Nhưng nó **chưa được chạy** trong lượt này, nên R9 ở cột working
tree là `ALREADY_FIXED (source, chưa đo)` — không phải một kết luận đã kiểm.

---

## 2. R5 — LIMN ghi sai target DID

### 2.1 Cơ chế: một biến `query` bị dùng lại

`workbench/docs/reviews/20261109-native15-composition-limn/scripts/t2_dual_scope.py`:

```
240            for qid in QIDS:                 # vòng 1: dựng texts/images
241                query = by_qid[qid]          # <- query cuối cùng = by_qid["7:3790"]
...
257            for row, qid in enumerate(QIDS): # vòng 2: serialize, KHÔNG gán lại query
...
273                    "target_did": query.positive_ids[0],   # <- luôn là positive của 7:3790
```

Vòng 2 **không** gán lại `query`, nên `query` còn nguyên giá trị của lần lặp cuối vòng 1 —
`by_qid["7:3790"]`, mà `positive_ids == ["7:42304"]`. Đó chính xác là giá trị xuất hiện 12 lần.

Bằng chứng dữ liệu (`raw/r5_queries.txt`):

```
qid=7:3038 ... positive_ids': ['7:3416']  target_gallery_id=.../B004D8QCN4.jpg
   shirt multimodal target_did='7:42304' MATCH=False
   ... (6 ô của 7:3038 đều MATCH=False)
qid=7:3790 ... positive_ids': ['7:42304'] target_gallery_id=.../B00E3WVAD2.jpg
   ... (6 ô của 7:3790 đều MATCH=True)
```

`raw/r5_reconcile.txt` mục A in đủ 12 dòng: **6 False / 6 True**, và 6 False đều thuộc `7:3038`.

### 2.2 qid → DID → asset → rank, cả 12 ô

Đối chiếu DID với release pool (`raw/r5_did_asset.txt`, đọc `mbeir_fashioniq_task7_cand_pool.jsonl`):

| DID | `img_path` |
|---|---|
| `7:3416` | `mbeir_images/fashioniq_images/B004D8QCN4.jpg` |
| `7:42304` | `mbeir_images/fashioniq_images/B00E3WVAD2.jpg` |
| `7:53224` | `.../B004TMGDME.jpg` (reference của 7:3038) |
| `7:33523` | `.../B0079K0YBO.jpg` (reference của 7:3790) |

Nên chuỗi đúng là **`7:3038` → `7:3416` → `B004D8QCN4` → rank {3, 88, 152, 43, 389, 67}**, và
**`7:3790` → `7:42304` → `B00E3WVAD2` → rank {65, 6891, 2200, 43, 4730, 4047}**.

`target_gallery_id` trong summary **khớp** chuỗi này (`B004D8QCN4` cho 7:3038, `B00E3WVAD2` cho
7:3790) — chỉ `target_did` sai. Vị trí gallery cũng khớp: `13614` và `70410`, đúng như
`LIMN_SCOPE_AUDIT.vi.md` §1.1 ghi.

### 2.3 Rank / score / candidate ID có đúng không? — **ĐÚNG**

`raw/r5_reconcile.txt` mục B kiểm từng ô: lấy `candidates[target_rank-1]` trong cửa sổ 100 dòng
của chính file export đó, rồi so `gallery_id` với `target_gallery_id` và `score` với `target_score`.

- 6 ô có `target_rank ≤ 100` → `window_check=PASS` **cả 6**: đúng `gallery_id`, `is_target=True`,
  score trùng tới 1e-6.
- 6 ô còn lại có rank 152…6891, ngoài cửa sổ lưu → `n/a` (không kiểm được bằng cửa sổ, nhưng
  `target_score` vẫn là score của đúng target).
- Mọi cửa sổ `window_sorted_desc=True`, `ranks_1..n = 1..100`.
- `rank_convention_crosscheck`: `checked=36`, `all_agree=true` (`raw/r5_positions.txt`) — 36 rank
  của query đã chấm được tính lại từ chính ma trận cache và khớp tuyệt đối.

**Kết luận R5:** rank, score, candidate ID, gallery position đều đúng. **Chỉ một field
`target_did` trong summary sai**, và field đó **không được serve**: `grep -rn target_did
workbench/backend/ workbench/frontend/` chỉ ra `limn_diagnostic.py` (bản working tree) chứ không
có consumer nào đọc nó; endpoint summary (`load_diagnostic`) không đưa `target_did` ra payload.
Vì vậy **không cần rescore, không cần rerun** — đúng như remedy đề xuất.

### 2.4 Phán quyết remedy R5 — **phù hợp**

"Lấy query theo qid tại thời điểm serialize; sinh lại summary từ bằng chứng đúng và verify digest;
chỉ rescore/rerun nếu không chứng minh được ranking."

- "Lấy query theo qid tại serialize time": đúng chỗ. Sửa tối thiểu là `query = by_qid[qid]` **bên
  trong** vòng `for row, qid in enumerate(QIDS)` ở `t2_dual_scope.py:257`.
- "verify digest": làm được ngay, vì `limn_dual_scope.json` đã có `gallery_matrices[*].recorded_content_digest`
  và `release_provenance.sha256` (`raw/r5_positions.txt`) — digest không đổi thì chứng minh được
  là chỉ summary sai.
- "chỉ rescore/rerun nếu không chứng minh được": theo §2.3 thì **không cần**, và tôi chứng minh được
  bằng 6/6 cửa sổ + 36/36 crosscheck. Nên remedy đúng, chỉ cần nói thêm rằng **bước rerun là no-op
  ở đây** để người sau không chạy lại 60 s GPU vô ích.
- **Cần bổ sung một điều**: remedy chỉ nói "regenerate summary". Nhưng nếu regenerate bằng chính
  script cũ thì lỗi tái sinh (cùng biến `query`). Phải **sửa script trước, rồi regenerate**, và
  thêm một assert `target_did in release.positive_ids` cho từng ô — nếu không, một lần regenerate
  nữa lại ghi đúng 12 lần `7:42304` và trông như "đã sửa".

### 2.5 Working tree

`workbench/backend/limn_diagnostic.py` (bản đang sửa) **đã** có guard:
`:191` `declared_did = record.get("target_did")`, `:192` `target_did = own_positives[0] if own_positives else declared_did`,
`:195-196` defect `target_did_not_own_query`. Tức nó **không tin** `target_did` nữa mà suy từ
`positive_ids` của chính qid. ⇒ `status_working_tree = ALREADY_FIXED` **ở tầng serve**.
Nhưng file sinh artifact (`t2_dual_scope.py:273`) **vẫn còn lỗi** — artifact trên đĩa vẫn ghi
`7:42304` cho `7:3038`. Nên nếu ai đọc thẳng JSONL/JSON (như `LIMN_SCOPE_AUDIT.vi.md` làm) thì vẫn
gặp số sai.

---

## 3. R6 — Guard của diagnostic endpoint

Chạy `raw/audit_r6_guard.py`: copy nguyên thư mục evidence sang `/tmp`, trỏ `CIR_REPO_ROOT` vào bản
copy, gọi thẳng `load_diagnostic` / `load_candidates` — đúng hai hàm mà hai endpoint gọi
(`main.py:1111-1139`). Không ghi gì vào repo. Output: `raw/r6_guard.txt`.

### 3.1 Trên HEAD — cái gì được nhận

| # | Input dị dạng | summary | candidates |
|---|---|---|---|
| 0 | control (bản copy nguyên) | ACCEPTED | ACCEPTED |
| 1 | `queries = {}` (payload rỗng) | **REFUSED** (`missing query 7:3038`) | ACCEPTED |
| 2 | checkpoint summary bị tráo (shirt ← toptee id+sha) | **ACCEPTED** | **ACCEPTED** |
| 3 | `export.checkpoint_sha256 = "0"*64` (sai checkpoint) | **ACCEPTED** | **ACCEPTED** |
| 4 | qid sai / scope sai / condition sai | – | REFUSED (3/3, thông báo cụ thể) |
| 5 | `target_rank` 3 → **77**, cửa sổ giữ nguyên | **ACCEPTED** | **ACCEPTED** |
| 5b | `target_rank = 1` nhưng `candidates[0].is_target = false` | **ACCEPTED** | **ACCEPTED** |
| 6 | trùng qid (2 dòng, dòng 2 nói rank 999) | – | **ACCEPTED**, trả rank 3 — **im lặng lấy dòng đầu** |
| 7 | `depth=100` nhưng chỉ lưu 3 candidate | **ACCEPTED** | **ACCEPTED** |
| 7b | `depth = 0` / `-5` / `500` | – | **ACCEPTED** cả ba; `0` và `-5` trả **1** candidate, `500` trả 100 |
| 8 | xoá 1 file export | **REFUSED** (đúng, có tên file) | – |
| 9 | export bị cắt còn 0 byte | **ACCEPTED** | REFUSED (`no row for 7:3038`) |
| 10 | export là JSON hỏng | **ACCEPTED** | **CRASH** |

### 3.2 R6-C — defect riêng: crash thay vì refuse

Đây là dòng cuối `raw/r6_guard.txt`:

```
File ".../limn_diagnostic.py", line 201, in load_candidates
    rows.append(json.loads(line))
json.decoder.JSONDecodeError: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)
```

`limn_diagnostic.py:194-201` đọc JSONL bằng `json.loads` trần, không bọc `try`, nên **chỉ**
`OSError`/`ValueError` của `_read_json` (dùng cho file summary) mới được đổi thành
`DiagnosticEvidenceError`. Một file JSONL hỏng ⇒ `JSONDecodeError` bay qua `except
DiagnosticEvidenceError` ở `main.py:1134-1139` ⇒ FastAPI trả **500**, không phải 404 có tên
`diagnostic_candidates_unavailable`. Tức: một file hỏng làm endpoint **không nói được lý do**, đúng
cái mà docstring của module tuyên bố là không bao giờ xảy ra ("report it rather than serve partial
data").

Trên HEAD, cùng một lỗi còn tệ hơn ở summary: **summary ACCEPTED** trong khi candidates crash
(dòng #10) — panel sẽ vẽ 12 ô rank rồi mỗi link chết bằng 500.

### 3.3 Working tree

Probe hành vi: **NOT_MEASURED** — lượt chạy `r6_working_tree.txt` bị huỷ trước khi ghi. Cái đã đo
được là **source**: bản working tree (615 dòng) có thêm
`export_duplicate_candidate` (`:316`), `summary_rank_invalid` (`:328`), `checkpoint_summary_vs_export`
(`:349`), `export_target_not_flagged` (`:363`), `target_did_not_own_query` (`:402`), cùng một
manifest + `_RECORD_CACHE` theo content digest (`:93-132`, `:257`), và trả `manifest_digest`
(`:593`). Đúng là remedy R6 được mô tả. Nhưng **tôi không có số đo** để nói input nào bị từ chối —
kết luận `ALREADY_FIXED` ở đây là **đọc code**, và cần một lần chạy `audit_r6_guard.py` trên working
tree để đóng lại.

### 3.3b Đo lại trên working tree — service sống, process có code mới

Mục này **thay** cho `raw/r6_working_tree.txt` (lượt chạy cũ bị huỷ). Process backend `:8000`
(pid 2788121, khởi động 08:16:44) đã nạp `limn_diagnostic.py` bản mới, nên probe trực tiếp là probe
hành vi thật, không phải đọc source.

```
GET /api/composition/diagnostics/limn-dual-scope/candidates
    ?qid=7:3790&scope=shirt&condition=text_only_black_image&depth=100
→ 200
  depth_requested 100 / depth_served 100 / depth_stored 100 / depth_truncated false
  target {rank 6891, score 0.21734005212783813, gallery_id mbeir_images/…/B00E3WVAD2.jpg,
          gallery_position 70410, in_window false, delta_to_rank_1 0.38371944427490234,
          image_url /api/images/shirt/B00E3WVAD2.jpg}
  candidate keys: gallery_id, gallery_position, image_category, image_path, image_stem,
                  image_url, is_target, rank, score
  complete: false     ← artifact thật còn 6 defect target_did_not_own_query
  top-level: identity, label, manifest, defects, target_asset, target_did, …

GET …&depth=0
→ 404 {"error":"diagnostic_candidates_unavailable",
       "message":"depth must be at least 1, got 0; a zero or negative depth cannot return a window
                  and silently returning one row would misstate the request"}
```

So với HEAD (§3.1):

| input | HEAD | working tree (đo trên service) |
|---|---|---|
| `depth=0` / `depth=-5` | 200, cửa sổ **1 dòng**, không báo | **404** + lý do nguyên văn |
| `depth=1000` | 200, **100 dòng** im lặng | `> 200` → từ chối (test `…[201/10000]`) |
| candidate có ảnh? | không (`image_url` vắng) | **có**, `image_url` resolve được (HTTP 200 `image/png`) |
| target riêng? | không | **có**, `target` luôn tách khỏi cửa sổ, kèm `in_window`/`delta_to_rank_1` |
| summary vs candidates | 12/12 mismatch (không có trường `export`) | cùng `manifest_digest`, cùng rank/score/checkpoint/target_did cho 12/12 |
| `complete` khi có defect | vẫn `true` | **`false`**, kèm `defects[]` có `code`+`qid`+`detail` |

⇒ R6 cột working tree: **ALREADY_FIXED và đã đo**, không còn "source, chưa đo".

### 3.3c Một điều probe này **không** chứng minh

`complete: false` khi đọc thẳng artifact thật **không** phải lỗi mới: đó là 6 defect
`target_did_not_own_query` mà R5 để lại trên `limn_dual_scope.json` (§2.5). Module mới **không**
che nó, cũng **không** sửa nó — nó chỉ không tin `target_did` nữa khi phục vụ. Ai muốn `complete:
true` phải dùng bản summary đã sửa (`raw/guards/limn_dual_scope.corrected.json`), và bản đó **chỉ**
đúng khi được đặt vào một repo root tạm — chưa có đường production.

### 3.4 Phán quyết remedy R6 — **phù hợp**

"Chia sẻ một record đã validate; bind summary/export vào manifest + digest; trả lý do cụ thể cho
missing/wrong; chỉ complete khi nội dung hợp lệ; cache theo content identity."

Năm vế đều trúng đúng 5 lỗ ở §3.1: (a) summary và candidates hiện **không** chia sẻ record — #2/#3/#9
chứng minh (summary accepted, export sai); (b) manifest+digest là thứ duy nhất bắt được #2/#3 vì
`_read_json` hiện chỉ parse, không so; (c) "lý do cụ thể" sửa #6 và #10 (im lặng / crash);
(d) "complete only on valid content" sửa #5/#5b/#7; (e) cache theo content identity sửa #9.
**Một điểm cần sửa**: remedy không nói `depth` phải bị chặn. #7b cho thấy `depth=0` và `depth=-5`
được **nhận** và trả 1 candidate (`limn_diagnostic.py:206` `max(1, min(int(depth), 200))`) — một
request vô nghĩa trả về một dòng trông hợp lệ. Cần từ chối `depth < 1` thay vì kẹp, nếu không thì
"cache by content identity" sẽ cache luôn kết quả của một tham số rác.

---

## 4. R8 — LIMN chỉ hiện text và Top-20

Bằng chứng: `raw/r8_probe.txt`; đọc `workbench/frontend/LimnScopeDiagnostic.tsx` và
`workbench/backend/limn_diagnostic.py`.

| vế của claim | Kết luận | Bằng chứng |
|---|---|---|
| Request depth bị chặn ở 20 | **CONFIRMED** | `LimnScopeDiagnostic.tsx:288` — `+ "&depth=20"` là **hằng chuỗi**, không có state K, không có selector. Không có chỗ nào cho người đọc xin 50/100. |
| Response chỉ có text | **CONFIRMED** | `raw/r8_probe.txt`: candidate keys = `gallery_id, gallery_position, image_path, is_target, rank, score` — **không có URL**. `grep -rn "<img" workbench/frontend/*.tsx` chỉ ra `QueryInspector.tsx:300,313`; `LimnScopeDiagnostic.tsx` render `<table>` (rank / gallery_id / position / score) — **0 thẻ ảnh**. |
| Image decoding | **KHÔNG ÁP DỤNG** | Không có ảnh nào được request, nên không có decode nào để hỏng. Đây là lý do *gốc* của "chỉ text", không phải một lỗi decode thứ hai. |
| Tiles khi K đổi | **CONFIRMED (không đổi được)** | Vì K là hằng, không có đường nào đổi K; và vì không có tile, "tiles khi K đổi" chưa tồn tại. |
| Diagnostic gallery **dùng chung** với Inspector | **CONFIRMED là KHÔNG dùng chung** | Hai renderer riêng: Inspector = `data-testid="topk-grid"` lưới `<article class="candidate">` + component `Shot` (`QueryInspector.tsx:212-247`); diagnostic = `<table>` trong `data-testid="limn-diagnostic-probe"` (`LimnScopeDiagnostic.tsx:331-343`). |
| Scope nằm trong identity | **DISPROVED** | `LimnScopeDiagnostic.tsx:281` — `const key = \`${qid}\u001f${scope}\u001f${condition}\``; và response có field `scope` (`raw/r8_probe.txt`). Scope **đã** là một phần identity của request/response. |
| Ảnh đích luôn hiện riêng | **CONFIRMED là KHÔNG có** | Không có `<img>` nào; và module **cố ý** từ chối ảnh query staged (`STAGED_PREFIX`, `limn_diagnostic.py:34-36`), nên ảnh đích không có đường hiện. |

### 4.1 Phán quyết remedy R8 — **cần sửa phương án**

"Gallery dùng chung cho diagnostic và Inspector; scope trong identity; giữ nhãn diagnostic theo
scope; chọn K theo export depth; luôn hiện ảnh đích riêng."

- "Gallery dùng chung": **đúng** — hiện có hai bản render lệch nhau, và bản diagnostic không có ảnh.
- "Scope trong identity": **đã có rồi** (`:281`), nên vế này **thừa**. Nếu đưa vào kế hoạch như một
  việc phải làm thì người thực hiện sẽ "sửa" một chỗ không hỏng và bỏ qua chỗ thật.
- "Chọn K theo export depth": **đúng nhưng chưa đủ** — phải chọn theo `depth_stored` **và** theo
  `candidateDisplayOptions()` mà Inspector đang dùng (`candidateRequest.ts:31-36`), nếu không hai
  panel sẽ lại lệch nhau.
- "Luôn hiện ảnh đích riêng": **đúng**, và cần nói rõ ảnh đích lấy qua
  `/api/images/{category}/{image_id}` (`main.py:1518`) — đường mà Inspector đã dùng
  (`QueryInspector.tsx:288`). Nhưng phải **giữ** quy tắc từ chối ảnh query staged: hiện ảnh đích
  (một ảnh dataset thật) là hợp lệ, hiện ảnh query đen là không.

---

## 5. R9 — Gate PASS trong khi subcheck fail

### 5.1 Không có chương trình nào tính G1/G2

`grep -rln "g1_g2\." workbench/` chỉ ra **hai** nơi ghi file: `t4_parity.py:400` và
`encoder_parity.py:976`. Không có aggregator, không có test nào đọc `g1_g2.*.json` để quyết định
PASS/FAIL. Trạng thái gate là **văn xuôi** trong `GATES.csv` dòng 3–4 và trong các `*.vi.md`.
`workbench/backend/paper_gate.py` là gate **paper-metric** (so với số công bố), **không** phải G1/G2.
⇒ Câu trả lời cho "mandatory subcheck nào có thể fail mà gate vẫn PASS" là: **tất cả**, vì không
có cơ chế nào ràng buộc. Đây là lỗi kiến trúc, không phải một con số sai.

### 5.2 Số đo — limn (`raw/r9_subchecks.txt` dòng 1–4, 29–38)

```
g1_g2.dress.adapter.json   sample_forward.query  = True
                           sample_forward.target = True
                           batch_size_1_vs_n     = False   max_abs=1.751e-07 max_rel=1.139977e-02
                           cache_off_restart     = True
g1_g2.shirt.adapter.json   batch_size_1_vs_n     = False   max_abs=1.974e-07 max_rel=2.698280e-03
g1_g2.toptee.adapter.json  batch_size_1_vs_n     = False   max_abs=1.714e-07 max_rel=7.509344e-03
```

Tolerance khai trong chính file: `batch_max_abs=1e-4`, `batch_max_rel=1e-3`. Nên **3/3 category**
fail `max_rel`. `GATES.csv:4` ghi "measured 1.14e-2" — đó là **giá trị dress**, không phải max của
cả ba (max là dress, 1.14e-2, nên câu đó đúng nhưng chỉ nêu một category).

### 5.3 Số đo — encoder (`raw/r9_encoder_numbers.txt`)

```
### encoder shirt
  sample.query_val_pixels:        max_abs=0.0      max_rel=0.0      within=True  bitwise=True
  sample.query_native_tensors:    max_abs=0.041421 max_rel=606.2216 within=False bitwise=False
  sample.target:                  max_abs=0.0      max_rel=0.0      within=True  bitwise=True
  batch_size_1_vs_n:              max_abs=2.813e-07 max_rel=0.0031309 within=False
  cache_off_restart:              max_abs=0.0      max_rel=0.0      within=None
  population.features_vs_native_val_control: max_abs=1.2778e-06 max_rel=0.75197  within=False
  population.features_vs_native_train_query: max_abs=0.122951 max_rel=84856.687  within=False
  population.gallery_vs_native:   max_abs=1.6484e-06 max_rel=7.74999  within=True
### encoder toptee
  sample.query_native_tensors:    max_abs=0.023708 max_rel=207.2901 within=False
  batch_size_1_vs_n:              max_abs=3.278e-07 max_rel=0.0116467 within=False
  population.features_vs_native_val_control: max_abs=9.872e-07 max_rel=7.85000  within=False
  population.features_vs_native_train_query: max_abs=0.125173 max_rel=233071.609 within=False
  population.gallery_vs_native:   max_abs=9.984e-07 max_rel=3.89759  within=True
### encoder dress
  sample.query_native_tensors:    max_abs=0.021249 max_rel=46.67525 within=False
  batch_size_1_vs_n:              max_abs=3.204e-07 max_rel=0.0077950 within=False
  population.features_vs_native_val_control: max_abs=1.0999e-06 max_rel=3.35484  within=False
  population.features_vs_native_train_query: max_abs=0.120392 max_rel=223487.729 within=False
  population.gallery_vs_native:   max_abs=1.1325e-06 max_rel=1.36364  within=True
```

**Bốn** cờ `False` mỗi category (kiểm lại bằng `raw/r9_encoder_false_flags.txt`, đọc thẳng
`within_tolerance` trên cả 3 file encoder):

| # | sub-check | max_rel (shirt / toptee / dress) |
|---|---|---|
| 1 | `layer4_forward_sample.query_native_tensors` | 606.2216 / 207.2901 / 46.67525 |
| 2 | `batch_size_1_vs_n` | 3.1309e-03 / 1.1647e-02 / 7.7950e-03 |
| 3 | `population.features_vs_native_val_control` | 0.75197 / 7.85000 / 3.35484 |
| 4 | `population.features_vs_native_train_query` | 84856.687 / 233071.609 / 223487.729 |

`sample.query_val_pixels`, `sample.target`, `population.gallery_vs_native` đều `true`.
`cache_off_restart` **không mang field `within_tolerance`** ở cả 3 file (`encoder_parity.py:757` chỉ
gán cho `batch_size_1_vs_n`) — nó **không** phải một cờ `False`, và tôi đã từng đếm nhầm nó thành
cờ thứ năm. Số đúng là **4/category, 12 trên ba category**.

**Sửa lại một kết luận trước đó của tôi:** tôi đã viết "ba cờ bị bỏ không nhắc". **Sai.**
`FINAL_REPORT.vi.md:231` nêu rõ population `max_rel` FAIL tới 7.85, và `:236-240` mô tả divergence
`preprocess_train` vs `preprocess_val` kèm ΔR@10 0.2975 / ΔR@50 0.1530. Vậy ba trong bốn cờ **có**
được công bố. Không có gì bị giấu.

Cái còn sai thì hẹp hơn, và đúng ba điểm:

1. **Đếm sai ở tiêu đề.** `FINAL_REPORT.vi.md:207` ghi "**hai** failure đã khai báo";
   `ENCODER_PARITY.vi.md:399` ghi "one declared sub-check FAILED"; `GATES.csv:4` ghi "TWO
   honestly-declared failures". Artifact mang **4** cờ `False`/category. Hai cờ được dán nhãn
   "failure", hai cờ được xếp vào "protocol divergence" — đó là **lựa chọn dán nhãn**, không phải
   một sự công bố, và **con số ở tiêu đề thì sai**.
2. **`query_native_tensors` (max_rel 606.2) là một sub-check của G2 *forward*.** Xếp một cờ
   forward-parity vào mục "protocol limitation" mà không nói kèm "và sub-check này đã fail" là chỗ
   thiếu chính xác cần gọi tên.
3. **Divergence chỉ được lượng hoá bằng metric delta**, chưa bao giờ được phát biểu là "hai sub-check
   này fail theo thiết kế". Người đọc có thể kết luận G2 forward đã pass mọi nơi trừ batch.

### 5.4 Tái lập tiêu chuẩn "mẫu số gần 0" (`raw/r9_denominator.txt`)

Code: `t4_parity.py:222-224` và `encoder_parity.py` cùng dạng —

```python
scale = np.maximum(np.abs(reference.astype(np.float64)), 1e-12)
return {..., "max_rel": float((delta / scale).max())}
```

tức `max_rel = max(|a-b| / max(|b|, 1e-12))` — **thuần tương đối, không có `atol`**. Hệ quả đo được:
cặp `(max_abs, max_rel)` cho biết phần tử gây ra max_rel có độ lớn bao nhiêu:

| file | max_abs | max_rel | suy ra `|b|` tại max_rel |
|---|---|---|---|
| limn dress | 1.751e-07 | 1.139977e-02 | **1.54e-05** |
| limn shirt | 1.974e-07 | 2.698280e-03 | **7.32e-05** |
| limn toptee | 1.714e-07 | 7.509344e-03 | **2.28e-05** |
| encoder shirt val_control | 1.278e-06 | 0.75197 | **1.70e-06** |

Nghĩa là cờ `False` của `max_rel` được quyết định bởi những phần tử có |giá trị| cỡ 1e-5…1e-6,
trong khi `max_abs` chỉ 1.7e-07…1.3e-06 — **nhỏ hơn `batch_max_abs=1e-4` khoảng 2–3 bậc**. Đây
đúng là bệnh lý mà remedy mô tả, và nó tái lập được **chỉ từ số đã công bố**, không cần chạy lại GPU.

### 5.5 Bốn loại subcheck, tách rõ

| loại | limn | encoder |
|---|---|---|
| observer khớp macro | `rank_convention_crosscheck` 36/36 agree, 0 rank difference; recall adapter ≠ native (55.05 vs 55.99 R@10) — **khác do mask, đã khai** | `native_metrics_match_feature_rerank: true`; nhưng stored aggregate lệch tới 0.198 R@1 (6/18 hit đổi) do `learnable_relation` — **D1, đã khai** |
| input parity (G1) | PASS, khai trong `INPUT_FORWARD_PARITY.vi.md:257` | PASS, `ENCODER_PARITY.vi.md:398` |
| forward/scoring parity | sample bitwise 0.0; `score_max_abs_delta_vs_dot` 5.36e-07 (< 1e-6) | sample bitwise 0.0 **chỉ trên `query_val_pixels`**; `query_native_tensors` fail 0.0414; score 1.19e-07 |
| batch parity | `max_rel` fail 3/3 category | `max_rel` fail 3/3 category |

### 5.5b Gate đã được code hoá — và nó kết luận **limn cũng FAIL**

`workbench/backend/parity_gate.py` (untracked, worker khác) nay có `evaluate_all()`. Chạy thật:

```
PYTHONPATH=/home/heheboiz/data/cir workbench/envs/ctl312h/bin/python \
  -c "from workbench.backend.parity_gate import evaluate_all; import json; print(json.dumps(evaluate_all(), indent=1))"
```

| model | status | passed | failure_count |
|---|---|---|---|
| limn | `PARITY_MANDATORY_SUBCHECK_FAILED` | **False** | 3 |
| encoder | `PARITY_MANDATORY_SUBCHECK_FAILED` | **False** | 13 |
| hint | `PARITY_NOT_RUN` | False | 0 |
| `passed_models` | **`[]`** | | |

limn `dims = {input_parity: True, forward_parity: True, batch_parity: False, scoring_parity: True}`;
3 failure đều là `batch_1_vs_n.max_rel` (0.0114 / 0.00270 / 0.00751 vs declared 1e-3) — khớp đúng
`raw/r9_subchecks.txt`.

⇒ **`FINAL_REPORT.vi.md` và code mâu thuẫn trực tiếp.** Report nói limn = "G1 PASS, G2
PASS-with-caveat" (`:128`), encoder = "PASS_WITH_DECLARED_FAILURES" (`:129`). Gate nói **cả hai
FAIL**, `passed_models = []`.

Claim nào bảo vệ được? **Của gate**, vì: (a) nó là hàm thực thi, đọc artifact, không phải văn xuôi;
(b) nó tách 5 dimension và không chuyển status giữa chúng; (c) nó không nới tolerance nào — dùng
đúng `batch_max_rel = 1e-3` mà `t4_parity.py:38` đã khai trước khi đo. Report chỉ đúng ở chỗ **các
con số** nó nêu (1.14e-2 v.v.), sai ở chỗ **gọi chúng là PASS**. Muốn report đúng thì phải đổi nhãn
thành `PARITY_MANDATORY_SUBCHECK_FAILED`, không phải đổi gate.

Phát hiện thêm, không có trong §5.3: gate bắt **13** failure cho encoder, nhiều hơn 12 tôi đếm, vì
nó kiểm thêm `population.gallery_vs_native.max_rel` — cờ này artifact ghi `within_tolerance: true`
nhưng `max_rel` = 1.36 / 7.75 / 3.90 **vượt** `forward_max_rel = 1e-4`. Gate in
`LABEL-DISAGREEMENT ... harness_recorded=True gate_verdict=False` cho cả 3 category. Tức harness tự
dán nhãn `true` cho một check mà chính số của nó vi phạm tolerance đã khai — thêm một lý do để
`FINAL_REPORT` không thể tự nhận PASS.

### 5.6 Phán quyết remedy R9 — **cần sửa phương án**

"Mandatory PASS chỉ khi mọi mandatory check pass; nếu cần chuẩn `abs(a-b) <= atol + rtol*abs(b)` thì
giải thích và version hoá trước khi đo mới, giữ kết quả cũ; kiểm rank và cutoff riêng."

- "Chỉ khi mọi mandatory check pass": **đúng hướng nhưng thiếu định nghĩa `mandatory`**. Hiện
  `cache_off_restart.within_tolerance` là **`None`** ở cả 3 file encoder (không được set —
  `encoder_parity.py:757` chỉ gán cho `batch_size_1_vs_n`, không gán cho `cache_off_restart`). Nếu
  "mọi check" bị đọc là "mọi key có mặt", thì `None` sẽ chặn PASS vĩnh viễn. Phải **khai báo danh
  sách mandatory có tên** và **cấm `None`** trong danh sách đó.
- "Chuẩn `atol + rtol*|b|`, version hoá trước khi đo mới, giữ kết quả cũ": **phù hợp và cần** —
  §5.4 chứng minh chuẩn hiện tại (thuần tương đối, sàn 1e-12) tạo ra đúng các `False` vô nghĩa này.
  Nhưng phải nói thêm: **giữ nguyên nhãn `False` cũ** và ghi nó là "fail theo chuẩn v1", không được
  viết lại lịch sử thành "pass".
- "Kiểm rank và cutoff riêng": **phù hợp** — encoder có 2 rank change không phải tie ở rank ≫ 50,
  hit-count không đổi; gộp nó vào một cờ "G2" là mất thông tin.

---

## 6. W1, R13, R1–R4, R7, R10–R12 — đã điền claim text

Mười dòng này từng là `BLOCKED` **không phải vì thiếu bằng chứng, mà vì briefing thiếu claim text**:
worker đầu chỉ nhận ID. Claim nguyên văn của operator nay nằm ở cột 2 của bảng §1 cho cả mười dòng,
và mỗi dòng có status + bằng chứng riêng.

Tổng hợp verdict sau khi điền (đọc kèm §1):

| ID | verdict | một dòng | nơi đo chi tiết |
|---|---|---|---|
| **W1** | CONFIRMED · **FIXED một phần** | landing đã đổi sang CIRCUS Full; còn banner `MOCK DEVELOPMENT DATA` in khi đang tải | `FINAL_REPORT.vi.md` §7.1–7.2 |
| **R13** | CONFIRMED · **FIXED một phần** | catalog "unverified có provenance" đúng thái độ, nhưng vẫn chỉ đọc 1 hình dạng macro + 2 glob ⇒ thiếu 4 model active | `FINAL_REPORT.vi.md` §7.3 |
| **R1** | CONFIRMED · **FIXED** | nhãn `fashioniq_full_gallery_ref_excluded` → `fashioniq_original_split` bằng migration có kiểm chứng; `protocol_source` sai bị đóng băng | `FINAL_REPORT.vi.md` §2.3 |
| **R2** | CONFIRMED (không sửa) | bất đối xứng transform là thật và ở phía native; control val-vs-val = 0 tuyệt đối | `PARITY.vi.md` §R2 |
| **R3** | CONFIRMED cơ chế · **DISPROVED** "ca thật là tie" | mask lệch giữa rank và probe tạo kết luận tie giả; ca thật gap 1.6e-3, không phải 1e-7 | `PARITY.vi.md` §R3 |
| **R4** | CONFIRMED | checkpoint 0 key `relation`, 415 state-dict, `.cuda()` làm mất registration, +2 key lạ | `PARITY.vi.md` §R4 |
| **R7** | **NOT_REPRODUCED** (trộn thế hệ) · CONFIRMED (thiếu khai báo) · **FIXED** | không subset nào vượt parent; R@10 phục dựng khớp 1e-9; nay có `source_run_id`/membership/mẫu số | `FINAL_REPORT.vi.md` §5.6 |
| **R10** | CONFIRMED · **giữ PARTIAL 6001/6003** | canonical không chọn scope; diagnostics shirt/toptee tách riêng | `FINAL_REPORT.vi.md` §5.5 |
| **R11** | CONFIRMED phạm vi · **DISPROVED** BLAS là nguyên nhân `encoder` | A/B threads 1 vs 10 ⇒ `changed=1`, r10/r50 delta 0.0 | `FINAL_REPORT.vi.md` §5.4 |
| **R12** | CONFIRMED · **4 BLOCKED + 1 NOT_A_BLOCKER** | mỗi model có lý do đo được, không cái nào là "chưa thử" | `BLOCKERS_R12.vi.md`, `FINAL_REPORT.vi.md` §8.1 |

Bốn phát hiện trong bảng trên **vượt ra ngoài** phần các worker đã viết, và được đo trong lượt này
(chi tiết + file ở `FINAL_REPORT.vi.md`):

1. **`limn` không có `EXITCODE=0`** — `cohort/logs/limn_run2.log` có đủ 3 dòng số nhưng kết thúc bằng
   `AttributeError: 'dict' object has no attribute 'complete'` tại `run_limn_oc220.py:73`; measurement
   ghi **sau** log 22 giây. Vậy vế "mỗi cái có `EXITCODE=0`" của headline 14/15 **không đúng cho
   limn**; đọc chặt thì 13/15.
2. **`protocol_source` bị migration đóng băng** — `FROZEN_FIELDS` giữ trường này và test
   `test_protocol_label_migration.py:57` khoá nó, nên câu **sai** ("the reference is excluded by
   masking its own index entry") vẫn còn trong `clip4cir_fullft.migrated.json`.
3. **Hai cặp macro cho một checkpoint** — `encoder` 56.2231 (web phục vụ) vs 56.2908 (BC-cũ trích);
   `clip4cir_fullft` 43.8017 vs 43.8182; `combiner` 35.1078 vs 35.1407. Không cặp nào được gộp.
4. **Hai "thế hệ" `limn` khác routing** — `9e4918a6` (unscored 2) vs `10173dc4` (unscored 119 = đúng
   119 `ambiguous_qids`); run canonical chỉ để lại digest `query_scope_resolution`.

**Cái gì vẫn sẽ kết luận được phần còn mở:** R12 cần `model_vssp.py` + `MHTransformer_dis.py` đúng
revision tác giả (hoặc quyết định của operator về trần "wrapper trung thực về số, không trung thực về
provenance"); R13 cần một reader nhận 5 hình dạng macro; R11 cần pin được đường encode của `ptha_mtst`
(`mkl_set_num_threads` segfault) trước khi nói gì thêm về model đó.

---

## 6b. Đo lại "native còn thiếu" — hai file bị báo sai là thiếu

Lệnh: `workbench/envs/ctl312h/bin/python workbench/docs/reviews/20261110-e2e-repair/raw/audit_native_measurements.py`
(script + output: `raw/audit_native_measurements.py`, `raw/native_measurement_census.txt`).

**Cạm bẫy đã tránh:** metric nằm trong `aggregate.macro_r10`, **không** ở top level. Quét
`d["macro_r10"]` trả **0 file** và sẽ "chứng minh" rằng không có native run nào trong cây. Đây đúng
lớp lỗi của `?q=` FALSE NULL: selector sai, không phải dữ liệu sai.

43 file khớp glob; **15** file mang macro số. **12** model_id phân biệt có `macro_r10`:
airknow, clvc_net, conesep, dcnet, encoder, habit, hint, intent, limn, sprc, tgcir, tme.
**5** config active **không** có: csmcir, ptha_mtst, pair, combiner_rn50x4_noft, clip4cir_rn50x4_fullft.

### 6b.1 `airknow_fiq_n05_run1.json` — native, **phải tính**

`model_id=airknow`, `protocol=fashioniq_full_gallery_ref_excluded`, `macro_r10=52.54661498980954`,
`macro_r50=72.9026186457433`, `complete=true`, `replay_type=SOURCE_FAITHFUL_WORKBENCH_REPLAY`.
Trường quyết định, theo từng category: `provenance.native_evaluator = "test_BLIP2.py:test"`, kèm
`native_output` mang đúng các số per-category. `replay_type` **không** có nghĩa synthetic —
`workbench/replay/airknow.py:1-12` nói module "runs the native FashionIQ evaluator" và chỉ bù phần
invocation shell vì `test_BLIP2.py` không có `__main__` hay checkpoint loader.
⇒ **native evaluation của checkpoint đã release.**

### 6b.2 `clvc_run1.json` — native, **phải tính** (kèm một thiếu sót ghi chép)

`model_id=clvc_net`, `protocol=fashioniq_val_split`, `macro_r10=29.970746517160162`,
`macro_r50=58.62171212901515`, `complete=true`, `replay_type` **vắng mặt**.
Trường quyết định: `provenance.source_commit = "bd9b6889489806cb60676597880f91d918b279cb"` —
trùng khít `upstream_commit` trong `workbench/third_party/CLVCNet/provenance.json` — cộng
`local`/`global` trỏ tới checkpoint thật và `parity_mismatches: []`. Per-category 29.35 / 27.77 /
32.79 đúng như đã nêu.
`native_evaluator` là **`None`**: không có trường nào chứng nhận entrypoint upstream. Nhưng **không**
trường nào đánh dấu nó là replay/synthetic. ⇒ **native evaluation**, với caveat ghi chép rằng
`native_evaluator` chưa được điền — đó là lỗ hổng tài liệu, không phải bằng chứng chống lại nó.

### 6b.3 Hệ quả

`FINAL_REPORT.vi.md` nói airknow và clvc_net "still missing native". **Sai theo artifact.** Câu
đúng: cả hai **có** native evaluation; airknow có `native_evaluator` chứng nhận entrypoint, clvc_net
thì không. Không được gộp hai mức bằng chứng đó thành một câu "còn thiếu".

**Đã kiểm thêm bằng log thực thi** — `raw/r13_extra_measurements_audit.md`. Cả ba file
(`tgcir_run1.json`, `clvc_run1.json`, `airknow_fiq_n05_run1.json`) là **EXECUTED_PROVEN**: mỗi file
có một log stdout trong `$D/logs/` chứa đúng ba dòng per-category và `aggregate` trùng tới từng chữ
số. `tgcir` chứng minh bằng `tgcir_run2.log` (2017/2038/1961 query + 2628/3089/2902 gallery, rồi
`aggregate {"macro_r10": 49.720772235486656, ...}`), **không** phải `tgcir_run1.log` — log đó là
traceback `AttributeError: 'MultiheadAttention' object has no attribute 'batch_first'`, `EXIT=1`.
`airknow` chứng minh bằng dòng cuối log: `[OK] result written: .../airknow_fiq_n05_run1.json`.

Hai khiếm khuyết ghi chép, không phải khiếm khuyết số liệu: (i) measurement của tgcir tên `run1`
nhưng sinh bởi `run2.log`; (ii) `clvc_net` và `tgcir` không có field `provenance.native_evaluator`
(bằng chứng nằm ở log, không ở measurement).

⇒ Headline "11/15 native, bốn missing" phải đổi. **Đính chính census v1 của tôi** (`raw/r13_census_v2.md`):
tôi đã quét thiếu bốn hình dạng (`pair_compat/measurements/`, `macro.macro_r10`, `macro_metrics`,
top-level `macro_r10`), nên câu "5 config không có file" là sai — **17/17 model đều có file mang macro**.

Nhưng "có macro" ≠ "native run hợp lệ". Sau khi tìm đủ bằng chứng (`raw/r13_ptha_proof.md`),
con số bảo vệ được là **14/15**:

> 14/15 target active có native evaluation đã thực thi, mỗi cái có log stdout hoặc command+meta với
> `EXITCODE=0` khớp với measurement mang macro. `pair` là cái duy nhất còn thiếu: chỉ có replay
> `mapping_status = "EXPLORATORY_PUBLIC_MAPPING"`, 34.04 so với `paper_reference.macro_r10 = 52.76`
> do chính file đó ghi. `csmcir`, `sprc` deferred, ngoài 15 active.

`ptha_mtst` **có** bằng chứng: `20261106T233409+0700/logs/ptha_transfer_run1.log` (719 dòng,
`exit=0`), `ptha_mtst.command.sh`, `ptha_mtst.meta.txt` (`EXITCODE=0 WALL_SECONDS=240`). Nhưng log
tự ghi `"mismatch": "training dataset/checkpoint differs; numerical context only, not reproduction"`
và Δ = −17.09 r10 / −16.44 r50: đây là **CIRR→FashionIQ transfer**, đếm vào "đã thực thi", **không**
đếm vào "tái lập paper".

Trong 14, **12** khai `native_evaluator` ngay trong measurement; `clvc_net` và `tgcir` chỉ chứng minh
qua log + script gọi hàm native, `ptha_mtst` qua log + command + meta.

## 7. Báo cáo hiện tại của campaign sai ở đâu

Bốn chỗ, xếp theo mức thiệt hại. Chỗ (a) đã được một người review sửa lại: bản đầu của tôi
nói "năm cờ" và "ba cờ bị bỏ không nhắc" — cả hai đều **sai** và đã được đính chính tại §5.3.

**(a) `ENCODER_PARITY.vi.md:399` + `GATES.csv:4` + `FINAL_REPORT.vi.md:207` đếm sai số failure của
encoder.** Tiêu đề ghi "hai" / "one" / "TWO"; artifact mang **4** cờ `False` mỗi category (12 trên ba
category — `raw/r9_encoder_false_flags.txt`). Ba trong bốn cờ **có** được công bố trong
`FINAL_REPORT.vi.md:231` và `:236-240`, nên đây **không** phải che giấu. Lỗi là: (i) con số ở tiêu đề
sai; (ii) `query_native_tensors` là sub-check **forward** nhưng bị xếp vào "protocol divergence" mà
không nói nó đã fail; (iii) divergence chỉ được lượng hoá bằng metric delta, không bằng câu "hai
sub-check này fail theo thiết kế".

**(a2) `FINAL_REPORT.vi.md:128-129` gọi limn và encoder là PASS, trong khi gate đã code hoá nói cả
hai FAIL.** `parity_gate.evaluate_all()` → limn `PARITY_MANDATORY_SUBCHECK_FAILED` (3 failure),
encoder (13 failure), `passed_models=[]`. Xem §5.5b. Đây là mâu thuẫn report-vs-code trực tiếp, không
phải một sắc thái diễn đạt.

**(b) `GATES.csv:6` "LIMN dual-scope PASS" kiểm 12 record mà không kiểm `target_did`.** Dòng đó liệt
kê đúng "100 ordered candidates (rank 1..100), depth=100, target_rank present" và 4 rank mẫu — tất
cả đều **đúng** (§2.3). Nhưng nó không so `target_did` với `positive_ids`, nên bỏ sót việc 6/12 ô
ghi DID của query kia. Gate PASS vì nó chỉ kiểm những gì nó nghĩ tới.

**(b2) `FINAL_REPORT.vi.md` nói airknow và clvc_net "still missing native" — artifact nói ngược lại.**
`airknow_fiq_n05_run1.json` có `macro_r10=52.5466` với `provenance.native_evaluator="test_BLIP2.py:test"`;
`clvc_run1.json` có `macro_r10=29.9707` với `provenance.source_commit` trùng `upstream_commit` của
CLVCNet. Xem §6b. Đúng ra chỉ được nói: airknow có entrypoint được chứng nhận, clvc_net thì chưa điền
`native_evaluator`.

**(c) Provenance: kiểm bằng `file_manifest` sẽ báo sai 5 "local modification".** 5 file ở
`BASELINE.md` §2.1 lệch manifest nhưng **trùng khít `derived_sha256`** của `approved_source_patches`
trong `workbench/registry/models.yaml` — chúng là patch đã duyệt, không phải drift. Ai viết lại
báo cáo provenance mà chỉ dùng `provenance.json:file_manifest` sẽ tạo ra 5 phát hiện giả.

Ngoài ra một điều **không** phải lỗi của báo cáo nhưng phải ghi: `limn_diagnostic.py` đã đổi
218 → 615 dòng giữa lượt này. Mọi câu "R5/R6 còn lỗi" chỉ đúng cho HEAD `2e2e2d2`; artifact
`limn_dual_scope.json` trên đĩa **vẫn** chứa 12 lần `7:42304`.
