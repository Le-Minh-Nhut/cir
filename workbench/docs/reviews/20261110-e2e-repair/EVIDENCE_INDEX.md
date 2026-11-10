# EVIDENCE_INDEX — mỗi artifact chứng minh điều gì

Lượt 2026-11-10. HEAD `382816315d0ab7c2e13d7a92d2fd68bc61a11a97` (`3828163`); mốc trước là `2e2e2d2`.
Bằng chứng nào đo ở `2e2e2d2` thì ghi đích danh `2e2e2d2` ở cột hoặc trong ô.
Đường dẫn dưới đây tương đối với `workbench/docs/reviews/20261110-e2e-repair/` trừ khi ghi `repo:`.

Quy ước: **một dòng = một artifact = một phát biểu được phép rút ra**. Artifact không nằm trong
bảng này thì không được trích số. Cột "chứng minh" ghi phát biểu **hẹp nhất** mà file đỡ được —
không phải phát biểu rộng nhất người đọc muốn.

> **Tầng bằng chứng** (dùng xuyên suốt báo cáo):
> `EXEC` = có log/command+meta của lần chạy thật · `MEAS` = có file measurement mang số
> · `READ` = đọc source/artifact, không chạy · `NOT_RUN` = không có gì.

---

## 1. Baseline và môi trường

| artifact | tầng | chứng minh |
|---|---|---|
| `BASELINE.md` | MEAS | HEAD, nhánh, upstream, 0 commit chưa push; service config 8000/5173; 17 model × interpreter; bundle 271 byte; disk 156 G trống |
| `raw/baseline_git.txt` | MEAS | `git rev-parse HEAD` = `2e2e2d2…` **(mốc trước)**; `git status --short` lúc chụp |
| `raw/baseline_provenance.txt` | MEAS | 16 repo vendor: `upstream_commit` + băm lại `file_manifest` |
| `raw/baseline_provenance_drift.txt` | MEAS | 5 file lệch manifest **khớp** `derived_sha256` của `approved_source_patches` ⇒ không phải drift |
| `raw/baseline_services.txt`, `raw/baseline_env.txt`, `raw/baseline_log.txt` | MEAS | pid 2759507/2759510, bind 8000/5173, chỉ 1 biến `WORKBENCH_CIRCUS_RELEASE_ROOT` được export, log `/tmp/workbench_serve.log` |
| `raw/baseline_interpreters.py` / `.txt` | MEAS | bảng 17 model: upstream pin vs adapted vs resolved vs reported |
| `raw/baseline_roots.txt`, `raw/baseline_disk.txt`, `raw/baseline_bundle.txt`, `raw/baseline_index.txt` | MEAS | root artifacts 84 G / data 25 G; `index.html` 271 byte trỏ 2 asset |
| `raw/baseline_jobs.txt` | MEAS | không có job GPU nào chạy (nvidia-smi 430 MiB, 0 process compute) |

## 2. Ma trận (dữ liệu tổng hợp, không phải số đo mới)

| artifact | tầng | chứng minh |
|---|---|---|
| `PROTOCOL_MATRIX.csv` (26 dòng × 58 cột) | MEAS (sinh từ `raw/*.json`) | 17 dòng native + 4 dòng protocol + 3 dòng execution-path; mỗi dòng có `gallery_count`, `reference_policy`, `raw_units`, `evidence_path` |
| `raw/protocol_cohorts.json` | MEAS | 3 protocol FashionIQ: gallery theo thứ tự + sha256 + đếm; `fashioniq_original_split` và `…_ref_excluded` **cùng** gallery sha256 |
| `raw/circus_cohorts.json` | MEAS | gallery CIRCUS 74381, sha256 `232d3b36…`, 0 dup; full 6003 / sf 4069 / validated 586; `category_resolution`: resolved 5884, **ambiguous 119** |
| `raw/membership.json` | MEAS | membership + tồn tại file ảnh (`.png` vs `.jpg`), union 15415 |
| `raw/checkpoint_hashes.tsv` | MEAS | sha256 của 15 checkpoint resolve qua `checkpoint_path()` |
| `raw/registry_dump.txt`, `raw/registry_active.txt` | READ | protocol + env + checkpoint variant đọc từ `models.yaml` |
| `RUN_MATRIX.csv` (61 dòng) | MEAS | 50 dòng composition (22 run × subset) + 11 dòng native; 7 cột surface tách rời (execution/validation/export/serving/cache/rerun) |
| `raw/run_inventory.json` | MEAS | nguyên liệu thô của RUN_MATRIX |
| `PHASE_A.vi.md`, `PHASE_B.vi.md` | MEAS | diễn giải hai ma trận; ghi rõ phần chưa làm |

Lưu ý định dạng: `RUN_MATRIX.csv` có **một** dòng hỏng do `ptha_mtst.json` không parse được
(`UNREADABLE_JSON`, `raw/native/ptha_mtst.json`); bản đọc được là `ptha_mtst.result.json`.

## 3. Native — file nào chứng minh lần chạy nào

`$D1` = `repo:workbench/artifacts/checkpoint_selection/20261106T002617+0700`
`$D2` = `repo:workbench/artifacts/checkpoint_selection/20261106T233409+0700`

| artifact | tầng | chứng minh |
|---|---|---|
| `raw/r13_census_v2.md` | MEAS | **17/17** model có ≥1 file mang `macro_r10` dưới `checkpoint_selection/`; liệt kê cả 4 hình dạng khác `aggregate.macro_r10` |
| `raw/native_measurement_census.txt` | MEAS | census **v1**: 43 file khớp glob, 15 file có `aggregate.macro_r10`, 12 `model_id`. Chỉ đúng trong phạm vi nó quét; **sai** nếu đọc như "model đã chạy" |
| `raw/r13_extra_measurements_audit.md` | MEAS | `tgcir_run1.json`, `clvc_run1.json`, `airknow_fiq_n05_run1.json` = EXECUTED_PROVEN; log ở `$D1/logs/`, **không** ở `artifacts/logs/` |
| `raw/r13_ptha_proof.md` | EXEC | `ptha_transfer_run1.log` 719 dòng `exit=0` + `ptha_mtst.command.sh` + `ptha_mtst.meta.txt` (`EXITCODE=0 WALL_SECONDS=240`) khớp `macro_metrics` |
| `raw/r13_pair_check.md` | MEAS | `pair_b1_H_normal.json`: `macro.macro_r10 = 34.03503453772374`, `paper_reference.macro_r10 = 52.76` (chính file tự ghi); `mapping_status = EXPLORATORY_PUBLIC_MAPPING` |
| `raw/r13_ptha_log_search.md`, `raw/r13_ptha_log_search.md` (log search) | MEAS | `artifacts/logs/` **không** có thư mục cho tgcir/clvc/airknow; log của chúng ở `$D1/logs/` (18 file, mtime 11-06 00:44–01:17) |
| `$D1/logs/tgcir_run2.log` | EXEC | 3 dòng per-category + `aggregate {"macro_r10": 49.720772235486656, …}` + `EXIT=0`; **trùng tới từng chữ số** với `measurements/tgcir_run1.json`. `tgcir_run1.log` là traceback `EXIT=1` |
| `$D1/logs/clvc_run1.log` | EXEC | 3 dòng per-category + `aggregate {... 29.970746517160162 …}` + `EXIT=0`, khớp `clvc_run1.json` |
| `$D1/logs/airknow_n05_run1.log` | EXEC | dòng cuối `[OK] result written: …/airknow_fiq_n05_run1.json` (không có chuỗi `EXIT=`) |
| `$D1/cohort/logs/limn_run2.log` | EXEC (yếu) | 3 dòng `{"r10": 51.1651 …}` khớp `limn_base_iter0.json`, **nhưng** kết thúc bằng `AttributeError: 'dict' object has no attribute 'complete'` tại `run_limn_oc220.py:73`; **không** có `EXIT=0` |
| `$D1/cohort/logs/{conesep,habit,hint,intent}_run*.log` | EXEC | dòng `[OK] result written: …/cohort/measurements/<model>.json` |
| `$D1/cohort/logs/encoder_run1.log` / `encoder_run2.log` | EXEC (một phần) | run1 = traceback (interpreter `pair-adapted` py3.11); run2 chạy hết 2902 ảnh nhưng **không** in metric và **không** có `EXIT=0` |
| `raw/native/*.stdout.log` + `*.meta.txt` | EXEC | `EXITCODE=0` cho conesep/encoder/habit/hint/intent/ptha_mtst/tme; tme in đủ `num_targets/num_queries` ba category |
| `raw/native/dcnet.time.log` | EXEC | `Exit status: 0`, `Elapsed 2:07:35` |
| `raw/native/clip4cir_fullft.time.log` + `.stdout.log` | EXEC | `Exit status: 0`, `Elapsed 37:26.07`; stdout in đủ 6 recall + 2 average |
| `raw/native/dcnet.json` | MEAS | macro **phân số** `0.27929730914548917 / 0.5376075764301528` + `macro_r10_percent` 27.9297 |
| `$D1/measurements/dcnet_run1.json` | MEAS | macro **phần trăm** `27.929730914548923 / 53.72723331879208` — cùng lần chạy, khác đơn vị ở hai file |
| `$D1/measurements/clip4cir_parsed.json` | MEAS | `clip4cir_noft.macro_r10 = 35.1407`, `clip4cir_fullft.macro_r10 = 43.8182`; **không** có `aggregate`, **không** có provenance ⇒ không được `/api/native-measurements` đọc |
| `$D2/measurements/ptha_transfer_run1.json` | MEAS | `macro_metrics {r10 37.8978, r50 59.8364}` — hình dạng `macro_metrics`, không phải `aggregate` |
| `$D1/cohort/measurements/{conesep,encoder,habit,hint,intent,limn,tme}*.json` | MEAS | số đang được web phục vụ; encoder ở đây = 56.2231, khác `raw/native/encoder.json` = 56.2908 |
| `$D1/measurements/sprc_vitl_run1.json` | MEAS | 51.0386 / 73.3263; `sprc_vitl_run5.log` có dòng aggregate khớp nhưng **không** có `EXIT=` |
| `$D1/../20261107T021500+0700/csmcir/measurement.json` | MEAS | macro **top-level** `macro_r10 = 57.05073873202006`, `evidence_class = PUBLIC_CAPTION PATH-MAPPED RESEARCH RUN` |

## 4. G1/G2 và gate

| artifact | tầng | chứng minh |
|---|---|---|
| `repo:workbench/docs/reviews/20261109-native15-composition-limn/g1_g2.{dress,shirt,toptee}.adapter.json` | MEAS | limn: mỗi category **đúng 1** block `within_tolerance: false`, đều là `batch_size_1_vs_n` (max_rel 1.140e-2 / 2.698e-3 / 7.509e-3) |
| `…/g1_g2.encoder.{dress,shirt,toptee}.adapter.json` | MEAS | encoder: **4** block `false`/category (12 trên ba); `rank_differences_vs_native_val_control` cho `non_tie_count` 1/1/0 và `min_gap_abs` 6.57e-4 / 1.60e-4 / 0.0 |
| `raw/r9_subchecks.txt` | MEAS | đọc lại đúng 4+4+3 khoá `within_tolerance` thô; xác nhận `cache_off_restart` **không** mang field đó ⇒ không phải cờ thứ năm |
| `raw/r9_encoder_false_flags.txt`, `raw/r9_encoder_numbers.txt` | MEAS | 4 cờ `False`/category với `max_abs`/`max_rel` từng block |
| `raw/r9_denominator.txt` | MEAS | `|b|` tại max_rel suy ra = `max_abs/max_rel` (1.5e-5 … 7.3e-5) ⇒ cờ do mẫu số gần 0, không do sai số tuyệt đối |
| `PARITY.vi.md` §"R2 × R9" | MEAS | sàn native-vs-native (0.120392/0.122951/0.125173) trùng 6 chữ số với 2 block ⇒ NOT_COMPARABLE; 1 block DENOMINATOR_ARTEFACT; 1 block GENUINE_FAIL |
| `raw/guards/r9_after.py` / `r9_after.txt` | MEAS | chạy `parity_gate.evaluate_all()`: limn 3 failure, encoder 13, `passed_models=[]`, `standard=parity-relative/v1` |
| `repo:workbench/backend/parity_gate.py` | READ | verdict suy từ artifact, 6 dimension tách rời, `passed` chỉ True khi không mandatory nào fail |
| `repo:workbench/tests/test_gate_parity_subchecks.py` | MEAS | 21 test pass (đã chạy lại lượt này) |
| `repo:workbench/docs/reviews/20261109-native15-composition-limn/GATES.csv` | READ | dòng G1/G2 vẫn `NOT_RUN` toàn cục; dòng 4 ghi "TWO honestly-declared failures" — đếm sai so với 4/4/3 cờ |

## 5. Web / W1 / R13 / R7

| artifact | tầng | chứng minh |
|---|---|---|
| `web_evidence/before-bare-url-probe.json` | MEAS | URL trần → `?area=overview&protocol=fashioniq_original_split&subset=sf`; `subset_select: 0`; "0 lượt có dữ liệu ở phạm vi này" |
| `web_evidence/before-*.png` (6 file) | MEAS | ảnh trước sửa: landing, benchmark, inspector, CIRCUS sf |
| `web_evidence/after-landing-probe.json` | MEAS | sau sửa (giữa lúc tải): protocol `circus_fashioniq_release_v1`, `subset_control: sf`, scope strip "Đang tải danh sách lượt…" |
| `web_evidence/after00.png` / `.txt` | MEAS | landing = CIRCUS **Full**, "17 lượt phục vụ · 17 có dữ liệu", bảng 17 row; **đồng thời** banner "MOCK DEVELOPMENT DATA — no real evaluation results are loaded" ở dòng 37 |
| `web_evidence/after01.png` / `.txt` | MEAS | `?protocol=fashioniq_original_split`: không control subset, "0 lượt composition (protocol này không có) · 13 báo cáo tổng hợp", card "Số đo native (chưa xác minh)" 3 file |
| `web_evidence/r7-api-counters.json` | MEAS | 17 run × 3 subset: không subset nào có `total`/`unscored` > `full` |
| `web_evidence/r7-subset-provenance.txt` | MEAS | 35 row: `cons=True`, `agree=True` cho mọi row có metric; `served == recomp` tới 1e-9; `e0dd04a46ef7` (FAILED) `scored=0 agree=None` |
| `web_evidence/r13-native-catalog.txt` | MEAS | 15 file measurement native có số + 4 file blocker + 1 file rỗng |
| `web_evidence/after-build.log` | MEAS | build 39 module, `dist/assets/index-CG4hbuvP.js` 357.68 kB |
| `web_evidence/after-tests.log` | MEAS | 51 test frontend pass |
| `raw/guards/r6_probe.json` | MEAS | HEAD: `depth=0/-5` → HTTP 200 cửa sổ 1 dòng; `depth=1000` → 100 dòng im lặng; candidates **không** có `export`; 12/12 summary-vs-candidates mismatch |
| `raw/guards/after_candidates_7_3790.json` | MEAS | sau sửa (instance 8011): `target.rank 6891`, `in_window false`, `image_url /api/images/shirt/B00E3WVAD2.jpg`, `depth_served 100`, `depth_stored 100` |
| `raw/guards/after_summary.json` | MEAS | sau sửa: `DIAGNOSTIC_EVIDENCE_DEFECT`, `complete: false`, 6 defect `target_did_not_own_query`, digest manifest trên 7 file |
| `raw/guards/r5_reconcile.json` / `.out` | MEAS | 12 ô: 6 problem, tất cả `did_matches_own_query` ở `7:3038`; 9 kiểm tra còn lại đạt 12/12 |
| `raw/guards/r5_correction.json`, `limn_dual_scope.corrected.json` | MEAS | bản sửa: đổi 6 record `7:42304`→`7:3416`; file gốc sha256 `04c4419e…` `source_untouched: true` |

## 6. Phase C (R1–R4, R10, R11)

| artifact | tầng | chứng minh |
|---|---|---|
| `raw/r1_migration_evidence.json` | READ+MEAS | verdict `MISLABELED` + 7 citation `path:line` + `gallery_counts` 6346/3817/5373 |
| `raw/clip4cir_fullft.migrated.json` + `.migration.json` | MEAS | nhãn đổi sang `fashioniq_original_split`; sha256 trước `fb97cfab…` / sau `dd9176cf…`; `aggregate.macro_r10` giữ `43.8016931215922` |
| `repo:workbench/scripts/migrate_protocol_label.py` | READ | từ chối khi thiếu citation / gallery mâu thuẫn / đổi sang chính nhãn cũ; `FROZEN_FIELDS` gồm `protocol_source` |
| `repo:workbench/tests/test_protocol_label_migration.py` | MEAS | 5 test pass (chạy lại lượt này); test khẳng định `protocol_source` **không** đổi |
| `raw/r2_transform_tensors.json` | MEAS | cùng ảnh 256×256: `control_val_vs_val` 0.0; `preprocess_train` lặp lại 3.8270; train-vs-val 3.8270 |
| `raw/r3_tie_harness.json` | MEAS | rank 4→3 (mask) và 3→2 (3 phần tử); gap không mask 1.192e-07, gap có mask 1e11; chỉ vị trí 0 vượt |
| `raw/r3_real_case.txt` | MEAS | dress: target rank 28 vs 31, gap nhỏ nhất 1.559e-3 / 3.639e-4 ⇒ **không** phải tie; chỉ 1/4 target định danh được |
| `raw/r4_determinism_p1.json`, `_p2.json`, `r4_determinism.raw.log` | MEAS | 417 key checkpoint, 0 key `relation`; 415 state-dict; `is_parameter: Tensor`; `seeded_same_seed_max_abs: 0.0`; không seed: 4.58…4.92 (ba lần đo khác nhau) |
| `raw/r4_relation_p1.json` / `_p2.json` | MEAS | cùng kết luận ở hai process sạch |
| `repo:workbench/docs/reviews/20261109-native15-composition-limn/limn_dual_scope.json` | MEAS | `canonical_status PARTIAL` 6001/6003, `unscored_qids [7:3038, 7:3790]`, gallery 74381, `rank_convention_crosscheck` 36/36 |
| `repo:workbench/docs/reviews/20261109-native15-composition-limn/COMPOSITION_MATRIX.csv` | MEAS | 8 run: R@10/R@50 + `blas_pin` từng dòng (encoder `changed=1, r10_delta 0.0`; conesep `17/18009` 1 bậc; limn `changed=0`) |
| `repo:workbench/docs/reviews/20261108T-throughput/d3b_encoder_aggregate_scan.txt` | MEAS | A/B threads 1 vs 10: `changed 1`, `max_drank 1`, `r10_delta 0.0`, `r50_delta 0.0`, `ndcg_delta 2.18e-05`, `cross_k10/50/100 = 0` |

## 7. R12 — blocker

| artifact | tầng | chứng minh |
|---|---|---|
| `BLOCKERS_R12.vi.md` | READ+MEAS | 5 model: interpreter / arch code / checkpoint load / entrypoint / verdict, mỗi dòng ghi ĐÃ QUAN SÁT hoặc NOT_OBSERVED |
| `raw/r12/airknow_pkl_globals.txt` | MEAS | 46 GLOBAL op, op đầu `lavis.models.blip2_models.blip2_cir Blip2QformerCir` ⇒ `clavis` là false positive của grep nhị phân (`0x63` = `c`) |
| `raw/r12/airknow_fullload.log` | MEAS | `LOAD_OK`, 863 tensor, 1 196 753 657 param, load CPU được |
| `raw/r12/airknow_import_lavis_ctl312h.log` | MEAS | lỗi import đầu tiên là `torchaudio` (`audio_processors.py:9`), `moviepy` ở dòng 11 |
| `raw/r12/pair_pkl_globals.log` | MEAS | 38 GLOBAL op, 36 unique; `model_vssp.Encoder`, `MHTransformer_dis.*` **không** có trong repo pin |
| `raw/r12/pair_tensor_count.log` | MEAS | 455 state-dict entry (456 = param + 1 buffer) |
| `raw/r12/combiner_gate_full.txt`, `combiner_verify_run.json` | MEAS | `METRIC_REFERENCE_UNVERIFIED`, `passed: false`, `is_research_evidence: false`; run vẫn `EXECUTION_COMPLETED`, `return_code 0` |
| `raw/r12/combiner_completion.txt` | MEAS | `EXECUTION_COMPLETED` / `PROOF_VALIDATED` / `EXTRACTED_VERIFIED`, metric 35.1078 / 58.4756 / 46.7917 |

## 7bis. Bằng chứng phát sinh sau commit `3828163`

Ba defect mà báo cáo từng ghi là mở đã được sửa và **commit**; mỗi cái có bằng chứng riêng, đo trên
process backend `:8000` khởi động **08:47:09** (pid 2820650, sau `main.py` 08:29:09).

| artifact | tầng | chứng minh |
|---|---|---|
| `web_evidence/after-parity-gate.json` | MEAS | `GET /api/runs/parity-gate` → **200**; `standard: parity-relative/v1`; `passed_models: []`; `limn` 3 failure, `encoder` 13 failure, cả hai `PARITY_MANDATORY_SUBCHECK_FAILED`. Ở `2e2e2d2` route **không tồn tại** |
| `web_evidence/1920-14-parity-gate.png` | MEAS | ảnh chụp route đó ở 1920px — bằng chứng định dạng hiển thị, không phải nguồn số |
| `web_evidence/after-banner-loading.txt` | MEAS | trace 3 mốc: `Đang tải bằng chứng` → `Đang tải bằng chứng · 2 đã chứng nhận trong 13 report đã về` → banner settled; `false-claim-seen: False`. Chứng minh banner **không** còn nói "không có bằng chứng" khi đang tải |
| `repo:workbench/tests/test_web_evidence_banner_loading.py` | MEAS | `test_a_loading_source_is_never_reported_as_no_evidence`; failing-first 6 failure trên `evidence.ts` cũ, **8 pass** sau sửa |
| `web_evidence/after-native-measurements.json` | MEAS | `{files_scanned 71, files 32, blocks_with_metrics 43, with_metrics 21, rollup_rows 22, unreadable 2, signposted_blockers 4, models_with_metrics 15, models_rollup_only 13, verified 0}`; `rollup: true` = 22/49 row; `ptha_mtst`/`pair`/`clip4cir_rn50x4_fullft`/`combiner_rn50x4_noft`/`csmcir` đều có mặt với `verified: false` |
| `repo:workbench/tests/test_native_measurements_census.py` | MEAS | 16 test, gồm năm hình dạng macro và việc loại roll-up khỏi `with_metrics` |
| `web_evidence/after-release-chain-banner-contract.txt` | MEAS | `aggregate-only: True`, `claims-absent: False` — banner đúng hợp đồng trên một deployment chỉ có aggregate |

Lưu ý đếm: `models_with_metrics 15` là **model không tính roll-up** (khớp `with_metrics 21` trừ phần
trùng); 28 là số `model_id` phân biệt nếu tính **cả** roll-up. Hai con số đo hai thứ khác nhau và
không được tráo cho nhau.

---

## 8. Artifact không được dùng làm bằng chứng

| artifact | lý do |
|---|---|
| `repo:workbench/docs/reviews/r13/R13_REMEDIATION.md`, `r8/*`, `r9/*`, `r10/*`, `r11/*`, `r12/*` (thư mục review cũ) | tài liệu lịch sử trên baseline khác; không phải số đo của `2e2e2d2` hay `3828163` |
| `CLEANUP_REPORT.md.save` | không rõ nguồn, không phải artifact của lượt |
| `raw/native/ptha_mtst.json` | JSON hỏng cú pháp (4715 byte); bản đọc được là `ptha_mtst.result.json` |
| `repo:workbench/tests/test_handoff_commands.py` (kết quả chạy) | **1 failed, 7 passed** — guard ĐỎ có chủ ý: `TESTED_SOURCE_SHA 36ae9e2…` không chứa production code hiện tại; SHA **không** được nâng |
| `raw/native/encoder.json` | measurement **khác** với file đang phục vụ (`cohort/measurements/encoder_fashioniq.json`); chỉ dùng khi nói rõ là bản thứ hai |
| `repo:workbench/artifacts/composition/summary/limn_base_iter0_all_categories_3f86f58483b368377e2c151e/` | run smoke q=8 (`COMPLETED` nhưng 8 query, R@10 25.0); không dùng làm metric |
