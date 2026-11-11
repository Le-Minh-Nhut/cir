# FINAL REPORT — master audit 2026-11-11 (workbench CIR)

Tài liệu bàn giao. **Không** nhắc lại nội dung tám báo cáo chi tiết; mỗi mục chỉ nêu kết luận và trỏ tới
báo cáo/raw tương ứng. Đọc kèm:

| báo cáo | nội dung |
|---|---|
| `BASELINE.vi.md` | mốc git, HEAD, remote, 6 commit cuối |
| `GATE_SCORING_VERIFICATION.vi.md` | G-01, G-02, S-01…S-06 |
| `MODEL_LIMN_ENCODER_VERIFICATION.vi.md` | M-01, M-02, M-03 |
| `BLOCKED_MODELS_VERIFICATION.vi.md` | M-04…M-07 (CLVC, TG-CIR, PAIR, Air-Know) |
| `MODELS_REST_VERIFICATION.vi.md` | M-08, M-09, M-10 |
| `UPSTREAM_INPUT_AND_PROTOCOL_MATRIX.vi.md` | upstream, đầu vào, protocol |
| `WEB_SURFACE_VERIFICATION.vi.md` | W-01…W-05 |
| `VERIFICATION_AND_SWEEP.vi.md` | bốn khẳng định Phase 1 + ba lần sweep |
| `REMAINING_WORK.md` | việc còn lại: 4 mục chưa sửa + 8 mục cần người khác; đọc trước khi tiếp tục |

---

## 1. Mốc và môi trường

| | |
|---|---|
| base HEAD | `92ed66de4f359ef479da18bf8939723b6f9313e5` |
| branch | `research/cir-failure-workbench` |
| cây phát hành | `7babfd96c1b5003d3fba7197bc6cbb1446a48a2b` — revision mà sweep pin đo (`18 failed, 2931 passed, 47 skipped` = 2996 collected, bản ghi `raw/verify/72-sweep-pin.md`), = `TESTED_SOURCE_SHA`; **đây là mốc bền**, không đổi nữa. `cd980dc` (`18 failed, 2901 passed, 47 skipped` = 2966, `raw/verify/68-final-sweep.md`) và `be14393` (18/2800/47) là các lượt sweep **bị thay thế**; production của chúng đã đi trước `7babfd9` (`git diff cd980dc 7babfd9` và `git diff be14393 7babfd9` ngoài `workbench/docs` và `workbench/tests` **không** rỗng), nên số của chúng chỉ còn là lịch sử. HEAD có thêm các commit **docs-only** sau `7babfd9`; lấy giá trị sống bằng `git rev-parse HEAD`. Tính chất phải giữ (là cái khiến sweep còn hiệu lực cho cả cây): `git diff 7babfd9 HEAD -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'` **rỗng**, và `git status --porcelain` đúng bằng `?? CLEANUP_REPORT.md.save`. *Một tài liệu bàn giao ghim SHA/count của HEAD sẽ cũ ngay commit sau — và bàn giao trong chiến dịch về drift báo cáo–thực tế thì không được là thứ đầu tiên nói sai thực tế.* |
| Python | `workbench/envs/ctl312h/bin/python` — CPython **3.12.11** |
| torch | **2.15.0+cu132** |
| GPU | **NVIDIA GeForce RTX 5070 Ti** |
| backend | `:8000`, env `WORKBENCH_CIRCUS_RELEASE_ROOT=…/artifacts/circus_release`. Lấy PID sống bằng `pgrep -af "serve_workbench.py --production-frontend"` / `pgrep -af "uvicorn workbench.backend.main"`. **PID chỉ đúng tại thời điểm đo** (mỗi restart đổi PID; stack đã chết/restart nhiều lần trong đợt) — **pin bền là digest bundle ở hàng frontend**, không phải PID |
| frontend | `:5173`, bundle **`assets/index-DyVaNikG.js`**, sha256 **`81d61f65ec827b83cabc958c157d65f6ad4ff99919ccae90cc1fdc29f03b0981`** (bytes phục vụ `cmp`-identical với `dist/`). Tên bundle là giá trị **di động** — đọc lại bằng `curl -s http://127.0.0.1:5173/ \| grep -o 'assets/index-[^"]*\.js'` rồi `sha256sum workbench/frontend/dist/assets/<tên vừa đọc>`: pin là digest, không phải tên file |

Sẵn sàng đo bằng `/api/benchmark-runs` (HTTP 200). **Không** dùng `/api/models` để poll: 154 s cold vì nó
hash mọi checkpoint. Có một uvicorn cũ trên `:8765` thiếu route mới — nhận diện bằng
`pgrep -af "port 8765"` (PID đổi sau restart, đừng ghim số) — **không** đo vào nó.

---

## 2. Bảng issue → fix → regression

Cột "fail-first" là lệnh/raw chứng minh lỗi **trước** khi sửa; cột "regression" là lệnh giữ nó đã sửa.

### Gate / scoring / cache

| ID | Trạng thái | file:line | fail-first | regression |
|---|---|---|---|---|
| G-01 | `FIXED_VERIFIED` | `backend/parity_gate.py:596-830`, `:181-341`, `:459-511`, `:833-881` | `raw/gate_negatives_before.txt` — 23/25 fail (`PARITY_PASS` cho adapter `{}`, cho delta `0.1` với `within_tolerance: true`, tolerance `inf`) | `raw/gate_negatives_after.txt` 25 passed; `tests/test_gate_contract_negatives.py` (25) |
| G-02 | `FIXED_VERIFIED` | `docs/reviews/20261110-e2e-repair/scripts/pc_r3_real_case.py` (viết lại) | `raw/gate_g02_r3_fixtures.txt`; bản cũ nhận diện **1/11** target | bản mới **11/11**; `tests/test_gate_r3_mask_join.py` (5) |
| S-01 | `FIXED_ALREADY` + `FIXED_VERIFIED` (phần thiếu) | `composition/runner.py:158-246`, `:1774` | thiếu test khoá refusal khi rank lệch | `raw/gate_s01_backfill_refusal.txt` 5 passed; `tests/test_composition_backfill_refusal.py` |
| S-02 | `REFUTED` (pin vô hiệu) + `CONFIRMED` (drift đồng thời) → `FIXED_VERIFIED` | `composition/runner.py:959-1008`, `:1012-1065`, `:219-227`, `:1972-1990` | `raw/gate_s02_drift_before_after.txt`; `raw/gate_s02_isolation_before.txt` 4 failed | `raw/gate_s02_isolation_after.txt` 4 passed; `raw/gate_s02_regression.txt` 40 passed |
| S-03 | `FIXED_VERIFIED` | xem `GATE_SCORING_VERIFICATION.vi.md:42` | `raw/gate_s01_s03_s04_regression.txt` | `73 passed` |
| S-04 | `FIXED_VERIFIED` | như trên | như trên | `73 passed` |
| S-05 | `CONFIRMED` (harness), **`REFUTED`** (lan sang production) | `docs/reviews/20261108T-throughput/d3b_limn_aggregate_scan.py:50` | `raw/gate_s05_limn_conditions.txt`: production chỉ **2,98%** rank trùng (179/6003) | `all_three_conditions_identical: False` — không sửa production vì không có lỗi |
| S-06 | `FIXED_VERIFIED` | xem `GATE_SCORING_VERIFICATION.vi.md` | raw cùng thư mục | cùng bộ test |

### Model

| ID | Trạng thái | file:line | fail-first | regression |
|---|---|---|---|---|
| M-01-a | `FIXED_VERIFIED` — code đúng, **và** đã có run mint sau fix mang cả identity lẫn tensor | `composition/model_encoder.py:42-63`, `:148-157`, `:262` (`_adopt_scratch_relation`), `:340-380` | `raw/m01/tests_failing_first.txt` 4 failed | `raw/m01/tests_encoder.txt` 64 passed; `31.0178 / 49.5586` **đo lần đầu trên** `ce89a9a2…`, **re-confirmed trên** `acc9b20e…` (delta `0.0`, 0/18009 rank khác). Bằng chứng verify: run `acc9b20e1aea48487f65f944` có `relation_state.sha256` `20db0c27…f9ce92` = `expected_sha256`, `matches_pinned_digest: true`, **và** `runs/acc9b20e…/construction_state/relation-seed20261111.npy` |
| M-01-b | `CONFIRMED`, **không** đổi production (đo là độ nhạy adaptation) | `model_encoder.py:390-400` | `raw/m01/forward_delta.json` | — |
| M-01-c | `CONFIRMED`; rerun mới dùng cold + `cache_verify_mode=full` | `runner.py:1612-1618` | `raw/m01/` | — |
| M-02 | xem `MODEL_LIMN_ENCODER_VERIFICATION.vi.md` | — | raw/m02 | — |
| M-03 | xem cùng báo cáo | — | raw/m03 | — |
| M-04 (CLVC) | `FIXED_VERIFIED` | `replay/clvc_net.py` | `tests/test_clvc_net_replay.py` | 2 run trùng nhau: **30.0208 / 58.6719** |
| M-05 (TG-CIR) | `FIXED_VERIFIED` | `replay/tgcir.py` | `tests/test_tgcir_replay.py` | **49.7208 / 71.1023**, tái lập digit-for-digit |
| M-06 (PAIR) | **`BLOCKED`** | `replay/pair_absence.py` | `raw/` trong `BLOCKED_MODELS_VERIFICATION.vi.md` | cần upstream `model_vssp` + `MHTransformer_dis` |
| M-07 (Air-Know) | `FIXED_VERIFIED` (kèm công bố nondeterminism) | `replay/airknow.py` | `tests/test_blocked_models_verification.py` | **52.5468 ± 0.2673** qua 4 run |
| M-08 / M-09 / M-10 | xem `MODELS_REST_VERIFICATION.vi.md` | — | raw tương ứng | — |

### Web surface

| ID | Trạng thái | file:line | fail-first | regression |
|---|---|---|---|---|
| W-01 | `FIXED_VERIFIED` | `backend/native_measurements.py`; `frontend/{format,data,EvaluationPage,OverviewPage}.ts*` | `raw/w01_native_measurements_BEFORE_backing.json` | `raw/w01_native_measurements_after.json`; `tests/test_native_measurements_census.py` (28) |
| W-02 | `FIXED_VERIFIED` | `frontend/scope.ts` (`family` trong identity) | `tests/test_web_scope_and_labels.py` | 14 passed |
| W-03 | `FIXED_VERIFIED` | `backend/limn_diagnostic.py:580-660`; `frontend/{QueryInspector,LimnScopeDiagnostic}.tsx` | `raw/w03_before_summary.txt` | `raw/w03_after_summary.txt`; `tests/test_limn_dual_scope_diagnostic.py` (17) |
| W-04 | `FIXED_VERIFIED` | xem `WEB_SURFACE_VERIFICATION.vi.md` | raw | raw |
| W-05 | `FIXED_VERIFIED` | xem cùng báo cáo | raw | raw |

### Phát hiện NGOÀI danh sách master file (đợt này tìm ra)

| phát hiện | trạng thái | bằng chứng / fix |
|---|---|---|
| **DCNet phục vụ sai thang 100×** — `metrics.r10=0.2793` với `unit:"percent"` ⇒ render **0.28%** cạnh ~50% của mọi model khác | `FIXED_VERIFIED` | `native_measurements.py` `_report_metrics` đọc sibling `_percent`; `_backed_by_artifact` tính lại unit **sau** khi thay số; `_provenance_unit`. Nay `27.929730914548916`, `unit_source: "sibling_key"`. Raw `raw/verify/31-live-unit-audit.txt`; test `test_dcnet_is_served_on_the_scale_its_sibling_key_declares` |
| **CLIP4Cir FullFT phục vụ 43.8182 không có artifact chống lưng** | `FIXED_VERIFIED` | thay bằng số của artifact **43.8016931215922**; 43.8182 chỉ còn ở `raw_metrics`. Test trong `test_native_measurements_census.py` |
| **`.gitignore` nuốt 121 log bằng chứng** — `*.log` không ngoại lệ nên log thậm chí không hiện trong `git status`; **0** file `.log` từng được commit trong toàn repo | `FIXED_VERIFIED` | thêm `!workbench/docs/reviews/**/*.log`. 25 (master-audit) + 24 (e2e-repair) + 72 (native15) = **121** log nay hiện. Raw `raw/verify/13-gitignore-evidence-logs.txt` |
| **LIMN `unscoped` vs `unscored` lẫn nhau** | `FIXED_VERIFIED` | `category_resolution` (119 ambiguous, release-level) tách khỏi `adapter_query_scope_resolution` (thực tế route). Summary mang cả hai |
| **Hợp đồng status `derived`** | `FIXED_VERIFIED` (option b) | `execution_status` **không** bị mở rộng — giữ `PARTIAL` của parent; subset tự khai trong `derived_from.own_status_label` / `own_coverage` / `own_scored_count` / `own_unscored_count`. `tests/test_derived_subset_status.py` (5) |
| **`relation_state` không tới được identity/summary/gate** | `FIXED_VERIFIED` | `runner.py` `_relation_identity`/`relation_state_from_summary`, vào `identity` + `summary`; `parity_gate.py` dimension `relation_parity`. Artifact cũ đọc `not_recorded`, không bao giờ pass |
| **Comment `runner.py` về seeding sai** | `FIXED_VERIFIED` | sửa: seeding **có** tồn tại nhưng theo từng model (`model_encoder.py:392`) |

---

## 3. Nhận định audit cũ bị BÁC BỎ

Phần quan trọng nhất với người đọc: mỗi câu dưới đây từng được coi là "sự thật" và **không đúng**.

| # | Nhận định cũ | Kết luận | Bằng chứng |
|---|---|---|---|
| 1 | "URL sai ở 7 repo" | **BÁC BỎ** — không tái lập được: 16/16 SHA khớp, 0 redirect, 0 fork | `UPSTREAM_INPUT_AND_PROTOCOL_MATRIX.vi.md` |
| 2 | "`np.argsort(-sims)` áp cho mọi model" | **SAI với 10/17 config**: DCNet dùng `torch.topk` ×3 nhánh; CLIP4Cir/PTHA/TME/CSMCIR/SPRC dùng `torch.argsort` **tăng dần** trên `1-cos` | cùng báo cáo, ma trận |
| 3 | "ENCODER dùng train transform cho mọi ảnh" | **SAI** — query `stage=0`, gallery `stage=1` | `MODEL_LIMN_ENCODER_VERIFICATION.vi.md` M-01-b |
| 4 | "PTHA có entrypoint FashionIQ" | **SAI** — `validate_blip.py:497-511` chỉ dispatch CIRR | `UPSTREAM_INPUT_AND_PROTOCOL_MATRIX.vi.md` |
| 5 | "BLAS pin không có tác dụng (9→9)" | **SAI trên host này** — pin chạy: **20→1→20**. Nửa "drift đồng thời" thì **đúng** và nay đã được phát hiện (`drift_detected`) | `raw/gate_s02_drift_before_after.txt`, `raw/gate_s02_isolation_after.txt` |
| 6 | "S-05 ablation bug lan sang production" | **BÁC BỎ** — production `multimodal` vs `text_only` chỉ **2,98%** rank trùng (179/6003), không phải 100% | `raw/gate_s05_limn_conditions.txt` |
| 7 | "Air-Know có 11 `.cuda()` trên scoring path" | **SAI** — chỉ **2** trên path thật; 12/9 là con số của **toàn file** | `BLOCKED_MODELS_VERIFICATION.vi.md` |
| 8 | "PAIR 455 state-dict entry + 1 buffer = 456" | **SAI** — gộp ba phép đếm: **547** tên, **456** object phân biệt, **90** alias | cùng báo cáo |
| 9 | "CLIP4Cir FullFT thuộc `ref_excluded`" | **SAI** — nó là `fashioniq_original_split`; `compute_fiq_val_metrics` **không** mask, chỉ nhánh CIRR mask | `UPSTREAM_INPUT_AND_PROTOCOL_MATRIX.vi.md` |
| 10 | "LIMN artifact trên đĩa vẫn còn `7:42304`" | **SAI** — đã sửa; chỉ **6** ô DID đổi, rank/score **byte-identical** | `MODEL_LIMN_ENCODER_VERIFICATION.vi.md` |
| 11 | "release guard đỏ / `TESTED_SOURCE_SHA` chưa nâng" | **SAI** — guard **8 passed**, SHA đã nâng kèm sweep riêng | `tests/test_handoff_commands.py` |

---

## 4. Kết quả đủ bằng chứng (có thể trích)

Chỉ những số dưới đây đủ căn cứ để trích — **trừ hai hàng ghi rõ là CHƯA CHỐT** (CLVC-Net, Air-Know):
hai model đó có **hai** số counted mâu thuẫn trên cùng protocol, nên chỉ được trích **kèm cả hai** và kèm
câu "chưa chốt". Mỗi số kèm nơi nó sống.

| model | R@10 / R@50 | căn cứ | nơi lưu |
|---|---|---|---|
| **TG-CIR** | **49.7208 / 71.1023** | khớp số lịch sử **ở R@10/R@50** (dress `43.38125929598414` / `68.368864650471` giống hệt); **dress R@1 lệch** `14.3778` vs `14.3282` (0.0496 pp), nên **không** phải digit-for-digit toàn bộ (`CENSUS_20261111.vi.md` §R12.5). Qua eval path do WORKBENCH thêm; `--restore_file` vẫn **dead** | `BLOCKED_MODELS_VERIFICATION.vi.md`; `raw/tgcir_run3cat_gpu.json:11` |
| **CLVC-Net** | **HAI SỐ counted, CHƯA CHỐT: 29.9707 / 58.6217 và 30.0208 / 58.6719** | hai artifact cùng `protocol_id: fashioniq_val_split` cho **hai** số khác nhau (lệch 0.0501 pp R@10); `list_native_measurements()` tại `358c3fc` phục vụ **cả hai** với `counted: true` và `conflict_with` nêu nhau — **không** resolved by recency. Không được trích một số như đã chốt | `CENSUS_20261111.vi.md` §1.2; `checkpoint_selection/20261106T002617+0700/measurements/clvc_run1.json` + `workbench/docs/reviews/20261111-master-audit/raw/clvc_run3cat_gpu.json` |
| **Air-Know** | **HAI SỐ counted + spread đo được: 52.5466 (n05 hist) và 52.9241 (n05 replay), mean nhóm 4 draw 52.5468 ± 0.2673** | `fiq_n05` có hai row counted mâu thuẫn (`conflict_with` nêu nhau); nhóm 4 draw có `variance_group` n 4, mean 52.54676006631348, sd 0.26728186433775697, spread 0.6252866852209422. Công bố nondeterminism của upstream, không giấu | `list_native_measurements()` tại `358c3fc`; `raw/airknow_run3cat_gpu_run{1..4}.json`; `raw/airknow_run3cat_gpu_variance_correction.md:21` |
| **ENCODER** | **31.0178 / 49.5586** | **đo lần đầu trên** `ce89a9a2…`, **re-confirmed trên** canonical `acc9b20e…` — hai run delta `0.0` trên cả 36 giá trị per-condition và mọi gap, 0/18009 rank khác. Canonical `acc9b20e1aea48487f65f944` partition tensor relation: `relation_state.sha256` `20db0c27…f9ce92` `matches_pinned_digest: true`, và tensor persist ở `runs/acc9b20e…/construction_state/` | `MODEL_LIMN_ENCODER_VERIFICATION.vi.md`; `summary/encoder_fashioniq_acc9b20e1aea48487f65f944/full.json`; `summary/encoder_fashioniq_ce89a9a27c4cceadeaaf30b0/full.json` |
| **LIMN** canonical `9e4918a6` | **PARTIAL 6001/6003** | + hai subset derived: sf 4068/4069 — `execution_status` **PARTIAL** (của parent), `derived_from.own_status_label` = **`DERIVED_PARTIAL`**; validated 586/586 — `execution_status` **PARTIAL** (của parent), `derived_from.own_status_label` = **`DERIVED_COMPLETE`**. `execution_status` **không** mang nhãn derived (option b) | cùng báo cáo; `tests/test_derived_subset_status.py` |
| **Gate** | **0 / 2 mô hình có artifact parity; 0 / 17 cấu hình PASS** | `passed_models: []`; `failure_count` **16** (encoder) và **3** (limn) — đều là **discrepancy thật**, không phải thiếu dữ liệu. Encoder tăng 13 → 16 vì ba subcheck `relation.<category>.digest` (dress / shirt / toptee, dimension `relation_parity`) trở thành **mandatory** ở R02/R03, không phải hồi quy mới. Mẫu số là cohort **17** cấu hình như §6: 13 model G1/G2 `NOT_RUN` chưa có artifact parity nên không thể pass | `GATE_SCORING_VERIFICATION.vi.md`; `/api/runs/parity-gate` |

Bốn run ENCODER đầu — `ce89a9a27c4cceadeaaf30b0`, `5dee7425dd46c5f154808aac`,
`6a528e877c328772e2c7bb28`, `eda30a37c0bc17ac43844f09` — đều mint trước `f856338` nên id của chúng
hash `relation_state: not_recorded`; không id nào trong bốn cái đó partition tensor relation. Run thứ
năm `9b806c22221e944ff10e514e` mint sau `f856338`: identity đã recorded nhưng `construction_state/`
vắng (defect, đóng ở `34aaa10`), nên vẫn superseded. Run thứ sáu `acc9b20e…` có **cả hai** — canonical.

<!-- ENCODER_CANONICAL_RUN: acc9b20e1aea48487f65f944 -->
**Canonical run ENCODER: `acc9b20e1aea48487f65f944`** — run **đầu tiên** có **đồng thời** (a) identity
partition tensor relation: `relation_state.status "recorded"`, `sha256 == expected_sha256 ==`
`20db0c27c979e9a0938c71af5142458dfa2070468d0ad3fcb6f24134c2f9ce92`, `matches_pinned_digest: true`,
`seed 20261111` (`summary/encoder_fashioniq_acc9b20e1aea48487f65f944/full.json` →
`relation_state`; block y hệt trong `sf.json`, `validated.json` và trong
`runs/acc9b20e…/ranks/encoder_fashioniq.json::identity`), và (b) tensor đã persist:
`runs/acc9b20e1aea48487f65f944/construction_state/relation-seed20261111.npy` (4224 B; 128 B header + 4096 B payload) + `.npy.json`
sidecar. Digest tensor (payload `float32 (2,512).tobytes()`) = `20db0c27…f9ce92` **== pin**; digest
**file** `.npy` là `6ce23bbc6a367c34f9536a612c9e32e3b9cec213687c7bd73378a16001ccf244` vì gồm cả header
numpy — pin ở repo này là trên byte tensor, không phải trên envelope, nên hai con số **khác nhau là
đúng**. File byte-identical với `.npy` của `ce89a9a2`.

Mọi số **không đổi**: 36 giá trị per-condition (3 subset × 3 condition × R@10/R@50/nDCG/MRR) và 5
aggregate gap trên cả 3 subset đều **delta đúng `0.0`**, và **0 / 18009** ô rank khác
(`ranks.encoder_fashioniq.json`, so từng query). Run `COMPLETED` cả 3 subset, unscored `0`, query
6003 / 4069 / 586, gallery 74381, `topk_export TOPK_EXPORT_COMPLETE depth 100`,
`cache_verify_mode full`.

Năm predecessor bị **superseded, lý do khác nhau** — không gộp: `ce89a9a2…` (identity hash khi
`relation_state` chưa có ⇒ `not_recorded`); `9b806c22…` (identity **đã** recorded nhưng **không** có
`construction_state/` — defect thật, đóng ở `34aaa10`); `5dee7425…` / `6a528e87…` / `eda30a37…`
(relation draw từ generator không pin; `5dee7425` thêm nữa `--subsets full` only).
Raw: `raw/verify/69-encoder-rerun.md` §R2.

Nói rõ: **0 PASS không phải lỗi của gate** — đó là 19 discrepancy thật (16 + 3) giữa hai phía, đọc từ
artifact. Gate cũ báo `PARITY_PASS` cho cả hai; gate mới báo đúng. Limn còn `missing ["metrics_parity"]`, tức
thiếu dimension, không phải trùng khớp.

---

## 5. Blocker còn lại

**Chặn cứng:**

- **PAIR** — cần upstream `model_vssp` + `MHTransformer_dis` (hai file pickle nguyên module). Không có
  cách nào trong repo này tái tạo. Xem `replay/pair_absence.py` (bằng chứng **vắng mặt**, không phải
  bằng chứng sai).

**Nói thẳng, không tô hồng:**

- **18 test fail có sẵn** (baseline `full_suite_68ad75e.txt`), **NEW = 0**. Chúng thuộc BLIP resident
  worker, circus CLI resume, TME lifecycle/precision, F08 DCNet config, handoff SHA, release-blocker GPU
  records, prepare_model recipe. Không phải của đợt này, nhưng vẫn đỏ.
- **13 model G1/G2 `NOT_RUN`** — chỉ encoder và limn có artifact parity.
- **8 model chỉ có roll-up** (`B1/H_normal`, `B1/H_rev`, `B2/H_normal`, `B2/H_rev`, `airknow_n05`,
  `airknow_n08`, `airknow_variant_80`, `clip4cir_rn50x4_noft`) — được **hiển thị** với
  `artifact_backed: false`, `counted: false`, **không** vào `models_with_metrics` (17).
- **46 dòng dựa trên đơn vị suy luận** (`inferred_from_magnitude`, trên **50** dòng có metrics; đọc
  `list_native_measurements()` tại `358c3fc`, `counts.unit_inferred` — trước ingest là 39/43). Đã audit
  từng dòng: **0** khai báo bị bỏ sót. Các khoá `*_pct` có mặt (`dead_parameters_pct`,
  `relative_error_pct`, `checkpoint_training_noise_pct`) mô tả **đại lượng khác**, không được dùng làm
  khai báo thang macro.
- **Derived row chưa có marker `DERIVED` trên UI** — dữ liệu đúng (`derived_from`), bề mặt web chưa hiện.
- **`:8765` stale uvicorn** vẫn sống, thiếu route mới (`/api/runs/parity-gate` → 404, so với 200 ở
  `:8000`). Tìm nó bằng `pgrep -af "port 8765"`; PID đổi sau restart nên không ghim số.
- **Stack không tự hồi phục — cơ chế đã xác định một phần, thủ phạm thì chưa.** Service `:8000`/`:5173`
  đã chết **ít nhất bốn lần** trong đợt này, mỗi lần một PID; **số chính xác không chốt được** vì mỗi
  lần restart lại redirect bằng một dấu `>` vào log `/tmp/…` nên chữ ký của lần trước biến mất. Bốn lần
  còn log header — PID 3691345, 4040574, 4062957, 4082632 — tất cả cùng một chữ ký log:
  `[WARN] npm exited with status -15` (bị
  SIGTERM) rồi backend `Shutting down` êm. Cả bốn chữ ký nay **đã commit trong repo**, nên bằng chứng
  không còn phụ thuộc `/tmp`:
  `raw/service-death-log.txt` (PID 3691345), `raw/service-death-log-2.txt` (4040574),
  `raw/service-death-log-3.txt` (4062957), `raw/service-death-log-4.txt` (4082632). Đường dẫn `/tmp`
  (`/tmp/workbench_serve.log`, `/tmp/workbench_serve_final.log`, `/tmp/serve_probe.log`,
  `/tmp/serve_handover.log`) chỉ là **provenance** của bản sao, không phải nơi trích dẫn; mtime
  `02:25:51` / `03:35:06` / `03:53:06` / `07:29:33` ngày 2026-11-11 lấy từ mtime tệp, **không** có
  trong nội dung log. Các restart khác trong danh sách PID superseded không còn log nên nguyên nhân của
  chúng chưa rõ. **Đã xác định:** đó chính xác là đường
  `serve_workbench.py:152-166` — `forward_stop` nhận SIGTERM/SIGHUP, gửi `os.kill(child.pid, SIGTERM)`
  cho từng child rồi `raise_signal(SIGINT)`. Nghĩa là launcher **của chính nó** nhận SIGTERM; child
  không tự chết. **Chưa xác định:** ai gửi SIGTERM đó. Đã kiểm và loại trừ: `jobs.py` spawn job bằng
  `start_new_session=True` nên `killpg` của nó chỉ chạm group riêng của job; không có test/script nào
  gửi tín hiệu tới npm/uvicorn theo tên; instance hiện tại có `SID == PGID == PID` (session riêng) nên
  miễn nhiễm với việc group của một shell khác bị reap. **Không suy đoán thêm.**
  Người đọc có thể gặp service **down**: `curl /api/benchmark-runs` trả **000** nghĩa là restart bằng
  lệnh ở §8. **PID không phải pin bền** (mỗi restart đổi PID); **digest bundle mới là pin** — hiện tại
  `assets/index-DyVaNikG.js` sha256 `81d61f65ec827b83cabc958c157d65f6ad4ff99919ccae90cc1fdc29f03b0981`
  (đọc lại bằng `curl -s http://127.0.0.1:5173/ | grep -o 'assets/index-[^"]*\.js'` rồi `sha256sum` trên `dist/`; tên file trôi theo mỗi build, digest mới là pin)
  (đo lại sau restart: `/api/benchmark-runs` 200 với 13 mục, `/api/native-measurements` 200 với
  `unbacked_counted: 0`, DCNet `27.929730914548916` / `percent` / `sibling_key`).
- **`/api/models` cold 154 s** — đừng dùng để poll; dùng `/api/benchmark-runs`.

---

## 6. Tổng kết điều kiện kết thúc

Mỗi hàng dưới đây được **đọc từ file hoặc API sống**, không chép từ báo cáo khác. Cột "nguồn" là chỗ
tôi thực sự lấy số.

| hạng mục | số | nguồn (đã tự đo) |
|---|---|---|
| cấu hình trong registry | **17 cấu hình / 16 upstream** | `registry/models.yaml` qua `load_registry()`: 17 model, 16 `upstream_repo_url` phân biệt; CLIP4Cir chứa 2 (`combiner_rn50x4_noft`, `clip4cir_rn50x4_fullft`) |
| cohort native FashionIQ | **17** (không phải 15) | `20261109-native15-composition-limn/NATIVE_EVAL_MATRIX.csv` có **17** dòng, và 17/17 model registry đều có mặt trong catalog native. Con số "15" trong tên chiến dịch cũ là **cũ** |
| đã thực thi thật (`native_executed = Y` trong CSV) | **12 / 17** | chỉ đọc cột `native_executed` của `20261109-native15-composition-limn/NATIVE_EVAL_MATRIX.csv`: 12 `Y`, 5 `N` (`airknow`, `pair`, `clvc_net`, `tgcir`, `sprc`). Đếm lại bằng `csv.DictReader`, không suy ra từ `models_with_metrics` |
| trong đó có `EXITCODE=0` kiểm chứng được | **9 / 17** | quét `docs/reviews/**/*.log` tìm `EXITCODE=0` + `*.json` có `runtime/side_effects.exit_code == 0`: chỉ 9 model có bằng chứng. **Không** phải 13/15 |
| qua **G1** | **0** | `GET /api/runs/parity-gate` → `passed_models: []` |
| qua **G2** | **0** | cùng response; `encoder` 16 failure, `limn` 3 failure (discrepancy thật, không phải thiếu dữ liệu) |
| dùng được để **so sánh nghiên cứu** | **17 model / 23 dòng** | `list_native_measurements()` tại `358c3fc` (đúng hàm route `/api/native-measurements` gọi, `main.py:1037`): `counted ∧ artifact_backed` = **23** dòng thuộc **17** model, `models_with_metrics: 17`. Trước ingest (`ae6fa2b`) là 21 dòng; advisory ghi 15 — **sai** |
| có **ảnh Top-K** | **full 13/18, sf 5/11, validated 5/11** | `GET /api/composition/capabilities?subset=…` → `capability_counts.candidate_topk` so với `run_count` |
| **deferred** | **2** | `frontend/ModelInventory.tsx:16-19`: `csmcir` ("thiếu dữ liệu phụ trợ"), `sprc` ("chưa tải checkpoint") |
| **blocked** | **1** | `pair` — cần upstream `model_vssp` + `MHTransformer_dis`; `replay/pair_absence.py` là bằng chứng vắng mặt |
| **cần rerun** | **6** | xem dưới |

### Sáu ca cần rerun (liệt kê thật, có căn cứ)

| ca | lý do | căn cứ |
|---|---|---|
| `clip4cir_rn50x4_noft` | chỉ có roll-up, không artifact | `/api/native-measurements`: `artifact_backed: false`, `counted: false` |
| `airknow` (3 dòng counted) | n05 và n08 là **checkpoint khác** nên vẫn hai số; riêng `fiq_n05` nay có **hai** row counted mâu thuẫn (`52.54661498980954` vs `52.924086629607395`, `conflict_with` nêu nhau). Upstream nondeterministic: `variance_group` n=4, sd `0.26728186433775697` | `list_native_measurements()` tại `358c3fc`: 3 dòng `counted ∧ artifact_backed`, `checkpoint_selection/…/airknow_fiq_n05_run1.json` + `airknow_fiq_n08_run1.json` + `raw/airknow_run3cat_gpu_run1.json` |
| `clvc_net` (**2 dòng counted mâu thuẫn**) | cùng `protocol_id: fashioniq_val_split`, hai số counted: `29.970746517160162` và `30.02079703540087`; API **không** chọn theo độ mới, chỉ ghi `conflict_with` cho cả hai. Cần chốt canonical hoặc rerun | `list_native_measurements()` tại `358c3fc`: `measurements[]` hai row `clvc_net` + `fashioniq_val_split` `counted: true`; code `native_measurements.py` `_mark_duplicates` nhánh `# 3d`. Chi tiết: `CENSUS_20261111.vi.md` §1.2 |
| `pair` (4 dòng) | 4 cấu hình B1/B2 × H_normal/H_rev **khác nhau thật** — cần chọn canonical hoặc công bố cả 4 | payload: 4 dòng `counted: true`, values khác nhau |
| ENCODER — **đã xong**: 5 id cũ `ce89a9a2…`, `9b806c22…`, `5dee7425…`, `6a528e87…`, `eda30a37…` | **superseded, lý do riêng từng cái**: `ce89a9a2` id hash `relation_state: not_recorded` (mint trước `f856338`); `9b806c22` recorded nhưng **không** có `construction_state/` (defect, đóng ở `34aaa10`); `5dee7425`/`6a528e87`/`eda30a37` draw từ generator không pin (`5dee7425` thêm nữa `full` only). **Canonical nay là `acc9b20e…`**: recorded + tensor persist | `MODEL_LIMN_ENCODER_VERIFICATION.vi.md` §M-01-a, §A; `runs/acc9b20e1aea48487f65f944/{construction_state,ranks}`; `raw/verify/69-encoder-rerun.md` §R2 |
| `habit` (run `e0dd04a4…`, subset `validated`) | artifact ghi `FAILED` vì interpreter trỏ `airknow-adapted` (Python 3.11.15 ≠ 3.8.10); env var **nay đã đúng** nên đây là artifact cũ | `summary/habit_fiq_n02_e0dd04a4…/validated.json` → `condition_status.*.reason`, `runtime.model_interpreter.path` |

### Điều **không** được tính là parity (theo master file)

**Không** tính `EXIT=0`, smoke pass, điểm gần paper, fixture, hay rank-only là parity. Vì thế G1/G2 là
**0** dù nhiều run `COMPLETED`; `native_verified = Y` chỉ có `limn`, và điều đó **không** nâng G1/G2 lên.

## 6.1. Hai defect phát hiện khi kiểm chứng bảng trên

### DEFECT A — bằng chứng `w_shots/` cũ, tự mâu thuẫn

Quét 14 file `web_evidence/w_shots/*.json` (raw `raw/verify/41-defectA-stale-evidence.txt`):

- **4 file** còn render `DERIVED_*` **trong cột `execution_status`** — đó là cách hiển thị của option (a)
  mà ta đã bỏ: `w02_models_inventory`, `w02_overview_circus`, `w04_reload_full`, `w05_pe_histograms`.
  Đối chiếu API sống: `DERIVED_*` nay **chỉ** nằm trong `derived_from.own_status_label`, còn
  `execution_status` là `PARTIAL`. ⇒ ảnh chụp cũ, **không** đại diện hành vi đã phát hành.
- **2 file** có record API khác 200:
  - `w04_deeplink_out_of_range` → 404 là **chủ đích** (probe `query_id 7:99999`), bằng chứng **hợp lệ**;
  - `w05_pe_histograms` → `/api/parity-gate` 404 vì **sai URL**: route thật là
    `@app.get("/api/runs/parity-gate")` (`backend/main.py:1044`), trả 200 với `passed_models: []`.

Cả 5 file vẫn được giữ (bằng chứng lịch sử không xoá), nhưng **không** được trích làm bằng chứng W-05 cho
hành vi hiện tại cho tới khi chụp lại trên bundle `index-DyVaNikG.js`.

### DEFECT B — `habit` FAILED vì interpreter (artifact cũ, không phải env hiện tại)

`summary/habit_fiq_n02_e0dd04a46ef7a58334b9e353/validated.json` ghi
`RuntimeError: execution environment not verified [BLOCKED/PYTHON_VERSION_MISMATCH]:
WORKBENCH_HABIT_PYTHON=…/envs/airknow-adapted/bin/python (expected Python 3.8.10, found 3.11.15)`.

Kiểm lại trạng thái hiện tại:

- `workbench/config/workbench.env:12` **nay đã đúng**:
  `WORKBENCH_HABIT_PYTHON=/home/heheboiz/data/cir/workbench/envs/ilearn39-adapted/bin/python`
  (cùng dòng 13/14/15 cho intent/hint/conesep).
- `workbench/envs/ilearn39-adapted/bin/python` tồn tại, **Python 3.9.25**; `airknow-adapted/bin/python`
  **không còn tồn tại**.

⇒ Đây **không** phải operator action đang mở, mà là **artifact cũ** mang đường dẫn đã chết. Cần
**rerun `habit`** để artifact mới ghi interpreter đúng. Registry vẫn khai `python: 3.8.10` cho habit,
trong khi env thực tế là 3.9 — registry dùng `adapted_runtime.python: 3.9`, nên dùng env 3.9 là đúng
theo `adapted_runtime`; **không** sửa `workbench.env` (bị cấm).

### Hạn chế đã biết — nhất quán chéo giữa chín báo cáo

Chín báo cáo trong thư mục này do **sáu worker viết song song**, và **một số con số đã dịch chuyển sau đó**
(mẫu số cohort, trạng thái run derived, đơn vị của DCNet, giá trị được serve của CLIP4Cir, và việc
`relation_state` đi tới gate). Báo cáo này đã được sửa tay ở những chỗ đó; các mục dưới đây là những gì
**thực sự** đã phát hiện và sửa, mỗi mục kèm bất biến hoặc lệnh chứng minh nó **nay** đúng:

- **§1 — mốc git.** SHA HEAD di động + số commit đã được thay bằng mốc bền `7babfd96c1b5003d3fba7197bc6cbb1446a48a2b`
  (= `TESTED_SOURCE_SHA`), kèm lệnh `git rev-parse HEAD` để lấy giá trị sống. Bất biến:
  `git diff 7babfd9 HEAD -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'` **rỗng**. Mốc cũ
  `cd980dc` và `be14393` được nêu rõ là lượt sweep **bị thay thế**, không còn là mốc.
- **§1 — hàng backend.** PID đã được thay bằng hướng dẫn `pgrep`, kèm quy tắc "PID chỉ đúng tại thời điểm đo;
  pin bền là digest bundle `81d61f65…0981`" (sha256 đầy đủ ở hàng frontend).
- **§1 note và §5 — uvicorn `:8765` cũ.** Nhận diện bằng **cổng** và bằng hành vi route:
  `/api/runs/parity-gate` trả **404** trên `:8765` so với **200** trên `:8000` — không bằng PID.
- **§4 — hàng gate.** `0/15 PASS` đã đối soát thành `0 / 2 mô hình có artifact parity; 0 / 17 cấu hình PASS`,
  khớp cohort **17** cấu hình của §6.
- **§4 — hàng LIMN.** `DERIVED_PARTIAL` / `DERIVED_COMPLETE` nay được gán đúng cho
  `derived_from.own_status_label`, còn `execution_status` được nêu là `PARTIAL` (option b) — **không** gán cho enum.
- **`WEB_SURFACE_VERIFICATION.vi.md`.** Tám build pin cũ đã được sửa; các bundle bị thay thế được **dán nhãn**
  superseded chứ không xoá.

**Chưa làm — và đây là điều người đọc phải biết:** một lượt **quét chéo có hệ thống** trên cả chín báo cáo
**đã không hoàn thành**. Worker được giao việc đó bị dừng trước khi kịp tạo output, và file bằng chứng dự kiến
— mục **42** của loạt `raw/verify/` (tên file nói về drift chéo giữa các tài liệu) — **không tồn tại**; không
file nào trong thư mục này trích dẫn nó (grep tên đó trong `workbench/docs/reviews/20261111-master-audit/`
cho **0** kết quả). Vì vậy **không** được giả định các báo cáo kia nhất quán nội bộ:
những gì đã kiểm là những gì nêu trong báo cáo này, cộng với những gì **từng báo cáo tự trích trong raw evidence
của chính nó**.

Thứ chống lưng cho trạng thái phát hành **không** phải văn xuôi của từng báo cáo, mà là: `18 failed / 2931 passed / 47 skipped`
với **accounting gap 0**, 2996 collected tại `7babfd9`, guard `workbench/tests/test_handoff_commands.py` **8 passed**, và
**production diff rỗng**.

## 7. Bảng sweep

Năm dòng là **năm cây KHÁC NHAU**, không phải một cây chạy năm lần.

| cây | dòng tổng kết | NEW | DROPPED | COMMON | collected | GAP |
|---|---|---|---|---|---|---|
| trước khi sửa 6 hồi quy | `24 failed, 2793 passed, 47 skipped` | 6 | 0 | 18 | 2864 | **0** |
| trước `relation_state` | `18 failed, 2799 passed, 47 skipped` | 0 | 0 | 18 | 2864 | **0** |
| `be14393` — **bị thay thế**, chỉ là lịch sử | `18 failed, 2800 passed, 47 skipped` | 0 | 0 | 18 | 2865 | **0** |
| `cd980dc` — **bị thay thế**, chỉ là lịch sử | `18 failed, 2901 passed, 47 skipped` | 0 | 0 | 18 | 2966 | **0** |
| **cây phát hành (SHIP) — `7babfd9`** | **`18 failed, 2931 passed, 47 skipped`** | **0** | **0** | **18** | **2996** | **0** |

Dòng `7babfd9` là bản có thẩm quyền: `350 warnings in 3445.68s (0:57:25)`, `^ERROR ` = 0, `Errno 28` = **0**,
`^FAILED` = 18, hai phép chứng minh cây đứng yên (đầu và cuối lượt cùng một SHA), và nó nghiệm thu đúng cây sản
xuất đang có (`git diff 7babfd9 HEAD` ngoài `workbench/docs` và `workbench/tests` rỗng). Bản ghi:
`raw/verify/72-sweep-pin.md`.

Dòng `cd980dc` giữ lại làm **lịch sử**: `350 warnings in 3486.46s (0:58:06)`, cùng bộ 18 id, nhưng production của
nó đã bị `7babfd9` vượt qua (`git diff cd980dc 7babfd9` ngoài `workbench/docs` và `workbench/tests` **không** rỗng
— hai commit frontend `252b380`, `d323a5c`), nên số của nó không còn mô tả cây đang ship. Bản ghi:
`raw/verify/68-final-sweep.md`.

Dòng `be14393` giữ lại làm **lịch sử**: ship sweep cũ `350 warnings in 3540.59s (0:59:00)`, 975 dòng, `ERROR`
row = 0, `Errno 28` = **0**, log đầy đủ; `git status` chụp **trong lúc chạy** byte-identical với trước khi chạy
(sha256 `d271b76b…6115`) ⇒ lượt đó không drift, nhưng production của nó đã bị `cd980dc` rồi `7babfd9` vượt qua
(`git diff be14393 7babfd9` ngoài `workbench/docs` và `workbench/tests` **không** rỗng), nên số của nó không còn mô tả
cây đang ship. Raw `raw/full_suite_ship_sweep.txt`, `raw/verify/{37,38,39}-*`.

---

## 8. Lệnh chạy lại

```bash
# toàn bộ suite, tuần tự, trên cây phát hành
mkdir -p /tmp/sweep && workbench/envs/ctl312h/bin/python -m pytest workbench/tests \
  -q -p no:cacheprovider -rfE --tb=short --basetemp=/tmp/sweep

# accounting (phải bằng passed+failed+skipped)
workbench/envs/ctl312h/bin/python -m pytest workbench/tests -q --collect-only -p no:cacheprovider | tail -3

# diff hồi quy so với baseline
workbench/envs/ctl312h/bin/python \
  workbench/docs/reviews/20261109-native15-composition-limn/scripts/failure_diff.py

# gate (0 PASS là ĐÚNG: 16+3 discrepancy thật)
workbench/envs/ctl312h/bin/python -c "import workbench.backend.parity_gate as g; \
  r=g.evaluate_all(); print(r['passed_models'], {m:v['status'] for m,v in r['models'].items()})"

# native measurements: số phải có artifact chống lưng
curl -s http://127.0.0.1:8000/api/native-measurements | \
  workbench/envs/ctl312h/bin/python -c "import json,sys; d=json.load(sys.stdin); \
  print(d['counts'])"

# W-03: presence trên window được serve
for D in 10 20 43 100; do curl -s "http://127.0.0.1:8000/api/composition/diagnostics/\
limn-dual-scope/candidates?qid=7:3038&scope=toptee&condition=multimodal&depth=$D" | \
  workbench/envs/ctl312h/bin/python -c "import json,sys; p=json.load(sys.stdin); \
  print(p['depth_served'], p['depth_stored'], p['target']['in_window'])"; done

# nội dung từng issue
workbench/envs/ctl312h/bin/python -m pytest workbench/tests/test_native_measurements_census.py \
  workbench/tests/test_gate_parity_subchecks.py workbench/tests/test_derived_subset_status.py \
  workbench/tests/test_limn_dual_scope_diagnostic.py workbench/tests/test_web_scope_and_labels.py -q

# khởi động lại service (bắt buộc có env var, nếu không composition view rỗng)
setsid nohup env WORKBENCH_CIRCUS_RELEASE_ROOT=/home/heheboiz/data/cir/workbench/artifacts/circus_release \
  workbench/envs/ctl312h/bin/python workbench/scripts/serve_workbench.py \
  --production-frontend --host 127.0.0.1 --backend-port 8000 --frontend-port 5173 \
  >> /tmp/workbench_serve.log 2>&1 < /dev/null &
```

Phải dùng `>>`, **không** dùng `>`: một dấu `>` truncate log bằng chứng ở mỗi lần restart, và đó chính
là cách chữ ký chết của các lần trước bị xoá — nên `>>` là bắt buộc.

---

## 9. Tóm tắt một câu

Cây phát hành đạt **`18 failed, 2931 passed, 47 skipped`**, 2996 collected tại `7babfd9`, gap 0, NEW = 0 so với baseline — không còn
hồi quy nào thuộc đợt này; phần còn đỏ là 18 lỗi có sẵn, PAIR bị chặn bởi upstream, và 19 discrepancy
parity **thật** (16 encoder + 3 limn) mà gate cũ đã che bằng `PARITY_PASS`.
