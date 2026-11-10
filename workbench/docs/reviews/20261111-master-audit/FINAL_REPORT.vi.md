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

---

## 1. Mốc và môi trường

| | |
|---|---|
| base HEAD | `92ed66de4f359ef479da18bf8939723b6f9313e5` |
| branch | `research/cir-failure-workbench` |
| cây phát hành | HEAD trên + work chưa commit của đợt này (xem §8 bảng sweep) |
| Python | `workbench/envs/ctl312h/bin/python` — CPython **3.12.11** |
| torch | **2.15.0+cu132** |
| GPU | **NVIDIA GeForce RTX 5070 Ti** |
| backend | `:8000`, PID **3691345** (con của `serve_workbench.py` PID 3691344), env `WORKBENCH_CIRCUS_RELEASE_ROOT=…/artifacts/circus_release` |
| frontend | `:5173`, bundle **`assets/index-DWpK0yeS.js`**, sha256 **`7e34724ebf55092e36524b98980cca55e71f3ddbaa62489c65577f92e18a6a16`** (bytes phục vụ `cmp`-identical với `dist/`) |

Sẵn sàng đo bằng `/api/benchmark-runs` (HTTP 200). **Không** dùng `/api/models` để poll: 154 s cold vì nó
hash mọi checkpoint. Uvicorn cũ trên `:8765` (PID 1227919) thiếu route mới — **không** đo vào nó.

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
| M-01-a | `FIXED_VERIFIED` | `composition/model_encoder.py:42-63`, `:148-157`, `:340-380` | `raw/m01/tests_failing_first.txt` 4 failed | `raw/m01/tests_encoder.txt` 64 passed; run mới `ce89a9a2…` |
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

Chỉ những số dưới đây đủ căn cứ để trích. Mỗi số kèm nơi nó sống.

| model | R@10 / R@50 | căn cứ | nơi lưu |
|---|---|---|---|
| **TG-CIR** | **49.7208 / 71.1023** | tái lập digit-for-digit với số lịch sử, qua eval path do WORKBENCH thêm; `--restore_file` vẫn **dead** | `BLOCKED_MODELS_VERIFICATION.vi.md` |
| **CLVC-Net** | **30.0208 / 58.6719** | hai run độc lập **trùng nhau** | cùng báo cáo |
| **Air-Know** | **52.5468 ± 0.2673** (4 run) | **kèm công bố** nondeterminism của upstream, không giấu | cùng báo cáo |
| **ENCODER** | **31.0178 / 49.5586** | run pinned `ce89a9a2…` | `MODEL_LIMN_ENCODER_VERIFICATION.vi.md` |
| **LIMN** canonical `9e4918a6` | **PARTIAL 6001/6003** | + hai subset derived (sf: 4068/4069 `DERIVED_PARTIAL`; validated: 586/586 `DERIVED_COMPLETE`) | cùng báo cáo; `tests/test_derived_subset_status.py` |
| **Gate** | **0/15 PASS**, 2 model được gate | `passed_models: []`; `failure_count` **13** (encoder) và **3** (limn) — đều là **discrepancy thật**, không phải thiếu dữ liệu | `GATE_SCORING_VERIFICATION.vi.md`; `/api/runs/parity-gate` |

Nói rõ: **0 PASS không phải lỗi của gate** — đó là 16 discrepancy thật (13 + 3) giữa hai phía, đọc từ
artifact. Gate cũ báo `PARITY_PASS` cho cả hai; gate mới báo đúng.

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
- **39 dòng dựa trên đơn vị suy luận** (`inferred_from_magnitude`). Đã audit từng dòng: **0** khai báo bị
  bỏ sót. Các khoá `*_pct` có mặt (`dead_parameters_pct`, `relative_error_pct`,
  `checkpoint_training_noise_pct`) mô tả **đại lượng khác**, không được dùng làm khai báo thang macro.
- **Derived row chưa có marker `DERIVED` trên UI** — dữ liệu đúng (`derived_from`), bề mặt web chưa hiện.
- **`:8765` stale uvicorn** (PID 1227919) vẫn sống, thiếu route mới.
- **Stack không tự hồi phục.** Trong đợt này service `:8000`/`:5173` đã chết **hai lần** bởi SIGTERM từ
  bên ngoài (`[WARN] npm exited with status -15` rồi shutdown êm), **nguyên nhân chưa được xác định** —
  không có commit nào xen vào giữa hai lần. Người đọc có thể gặp service đang **down**:
  `curl /api/benchmark-runs` trả **000** nghĩa là phải restart bằng lệnh ở §7. **PID không phải pin bền**
  (mỗi lần restart đổi PID); **digest bundle mới là pin** — hiện tại
  `assets/index-DWpK0yeS.js` sha256 `7e34724ebf55092e36524b98980cca55e71f3ddbaa62489c65577f92e18a6a16`.
- **`/api/models` cold 154 s** — đừng dùng để poll; dùng `/api/benchmark-runs`.

---

## 6. Bảng sweep

Ba dòng là **ba cây KHÁC NHAU**, không phải một cây chạy ba lần.

| cây | dòng tổng kết | NEW | DROPPED | COMMON | collected | GAP |
|---|---|---|---|---|---|---|
| trước khi sửa 6 hồi quy | `24 failed, 2793 passed, 47 skipped` | 6 | 0 | 18 | 2864 | **0** |
| trước `relation_state` | `18 failed, 2799 passed, 47 skipped` | 0 | 0 | 18 | 2864 | **0** |
| **cây phát hành (SHIP)** | **`18 failed, 2800 passed, 47 skipped`** | **0** | **0** | **18** | **2865** | **0** |

Ship sweep: `350 warnings in 3540.59s (0:59:00)`, 975 dòng, `ERROR` row = 0, `Errno 28` = **0**, log đầy
đủ. `git status` chụp **trong lúc chạy** byte-identical với trước khi chạy
(sha256 `d271b76b…6115`) ⇒ không có drift. Raw `raw/full_suite_ship_sweep.txt`,
`raw/verify/{37,38,39}-*`.

---

## 7. Lệnh chạy lại

```bash
# toàn bộ suite, tuần tự, trên cây phát hành
mkdir -p /tmp/sweep && workbench/envs/ctl312h/bin/python -m pytest workbench/tests \
  -q -p no:cacheprovider -rfE --tb=short --basetemp=/tmp/sweep

# accounting (phải bằng passed+failed+skipped)
workbench/envs/ctl312h/bin/python -m pytest workbench/tests -q --collect-only -p no:cacheprovider | tail -3

# diff hồi quy so với baseline
workbench/envs/ctl312h/bin/python \
  workbench/docs/reviews/20261109-native15-composition-limn/scripts/failure_diff.py

# gate (0 PASS là ĐÚNG: 13+3 discrepancy thật)
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
  > /tmp/workbench_serve.log 2>&1 < /dev/null &
```

---

## 8. Tóm tắt một câu

Cây phát hành đạt **`18 failed, 2800 passed, 47 skipped`, gap 0, NEW = 0** so với baseline — không còn
hồi quy nào thuộc đợt này; phần còn đỏ là 18 lỗi có sẵn, PAIR bị chặn bởi upstream, và 16 discrepancy
parity **thật** mà gate cũ đã che bằng `PARITY_PASS`.
