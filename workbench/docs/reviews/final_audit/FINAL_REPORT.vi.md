# Báo cáo cuối: vòng audit mô hình & rà soát tổng thể (CIR Failure Analysis Workbench)

Nhánh: `research/cir-failure-workbench`. Cơ sở: `eb36aa5` (đã push trước vòng này).
Từ vựng verdict: `CONFIRMED_BUG`, `CONFIRMED_FIXED`, `UNVERIFIED_RISK`,
`EXTERNAL_BLOCKER`, `GPU_VALIDATION_PENDING`, `RESOLVED`.

## 1. Verdict 16 mô hình (audit độc lập từng mô hình)

Báo cáo chi tiết: `models/<model_id>.md`. Tổng hợp: `CONSOLIDATION.md`,
tracker: `AUDIT_TRACKER.md`.

| Model | Verdict | Tóm tắt |
|---|---|---|
| limn | BLOCKED | P0 noise null↔0 đã sửa; còn external blockers |
| csmcir | VARIANT | caption-alias staged (`CAPTION THAY THẾ`); patch approved, không claim official parity |
| airknow | BLOCKED | external |
| conesep | BLOCKED | external |
| habit | BLOCKED | external |
| intent | BLOCKED | external |
| hint | BLOCKED | VARIANT-flagged; external blockers |
| encoder | BLOCKED | mapping UNVERIFIED (honest); checkpoint không có URL trực tiếp |
| pair | BLOCKED | 2 checkpoint, mapping UNVERIFIED; contract mới `pair_fashioniq` |
| clvc_net | BLOCKED | headline 30.7/58.41/44.56; external |
| dcnet | BLOCKED | mean 40.83 (paper-stated); external |
| clip4cir_rn50x4_fullft | BLOCKED | headline 43.78/66.93/55.36; external |
| combiner_rn50x4_noft | BLOCKED | stdout parser mới; external |
| tgcir | BLOCKED | replay preflight-only; parser REJECTED (không có command) |
| sprc | BLOCKED | parser csmcir-shape wired; SPRC2 attribution conflict ghi trong provenance |
| ptha_mtst | BLOCKED | 2 rows cùng condition — `select_reference` refuse (fail-closed đúng) |

Không model nào đạt `RESOLVED`: mọi reproduction status hiện nay do blocker phía
upstream, không phải lỗi workbench.

## 2. `CONFIRMED_FIXED` (đã sửa + có evidence)

Nhóm registry/papers (19 mục ở `CONSOLIDATION.md`) + các sửa bổ sung từ holistic
review:

1. Noise 0→null cho các row paper KHÔNG ghi noise (limn, hint, encoder, tgcir,
   ptha). Row có bảng noise (airknow, habit, intent, conesep) giữ 0.
2. `build_paper_registry.build_record` override per-experiment `paper_*`
   (encoder/pair trước đây bị kẹt ở paper-block identity).
3. Headline chỉ điền từ cell paper: clvc_net, clip4cir fullft, dcnet mean;
   giá trị derived ghi provenance rõ.
4. `parse_combiner_output`: reject duplicate category/average line; malformed
   average (`average recall50 = 1e`) trả UNPARSED thay vì raise. 2 test mới.
5. `_derived_is_bound_to_source`: receipt thiếu/khác `resampling` → khôngreuse,
   regenerate (không thể "giặt" ảnh kernel khác thành receipt mới).
6. Gate: status mới `PASS_WITH_UNCHECKED_CATEGORIES` — pass nhưng KHÔNG pass
   trơn khi paper in category metrics mà run không observe. (Thay thế disposition
   cũ "hint C5 REJECTED".)
7. ENCODER `checkpoint_mapping_status` → `UNVERIFIED`: không có artifact
   author-linked nào gắn file Drive với variant FashionIQ của paper; local install
   đầy đủ cũng không chứng minh danh tính khoa học. Registry coherence rule sửa
   thành một chiều: URL_UNRESOLVED không ép mapping VERIFIED. (Thay thế dispute
   "encoder C4".)
8. API project `is_synthetic`; UI badge `SYNTHETIC SOURCE` cho aggregate-only run.
   KHÔNG thêm warning label thứ 3 (vocabulary đóng).
9. CLIP4CirFullftAdapter đọc bundle members dưới đúng bundle path.
10. Evidence honesty: CSMCIR ghi cả 2 chênh lệch category↔average (R@10 0.027pp,
    R@50 0.087pp); AirKnow AVG 65.73 — câu "arithmetic proof" sửa (half-up ra
    65.74; cell in được transcribe, không bịa justify); `checkpoint_training_noise_pct: 0`
    csmcir/clip4cir annotate `WORKBENCH_INFERENCE`.

## 3. `REJECTED` / `BY-DESIGN` (rationale)

- limn P2 (macro từ forged categories): fail-closed đúng — recomputing macro từ
  output bịa sẽ cho fabricated pass.
- limn P3 (dataset-root dual identity): family name, symlink caught bởi preflight.
- ptha C1 (2 verified rows cùng condition): refuse là đúng; cần discriminator
  paper-backed trước khi chọn row.
- tgcir parser: replay preflight-only, không command → không có output để parse.
- Slider threshold ≠ backend gate: BY-DESIGN — slider re-flag recorded
  comparisons; verdict chính thức ở 5% cố định (`_gate_fields` docstring).
- Mock runs hiển thị: có badge rõ, không thay thế data thật.

## 4. `DEFERRED` / `EXTERNAL_BLOCKER`

- combiner C-1: đăng ký row VERIFIED từ CVPR22 cần three-route verification
  (số liệu audit là secondhand).
- `canonical_dataset_link` nested name (dcnet): validator yêu cầu single
  in-checkout name — EXTERNAL_BLOCKER.
- CSMCIR parity: không claim official parity; provenance ghi trong report.

## 5. Verification matrix (evidence chạy thật)

| # | Kiểm chứng | Kết quả |
|---|---|---|
| 1 | Full suite (`pytest workbench/tests/ -p no:cacheprovider -q`) | run trước: 897 passed; run cuối: **899 passed, 0 failed** (5 encoder expectation đã sửa trước run cuối) |
| 2 | `test_adversarial_metrics.py` (sau thêm 2 test parser) | 25 passed |
| 3 | `test_paper_gate.py` + `test_http_api_layer.py` (sau status mới) | 68 passed |
| 4 | `test_checkpoint_selection.py` + `test_protocols.py` + `test_operator_contracts.py` + `test_strict_acceptance_matrix.py` (sau encoder mapping) | 47 passed |
| 5 | `test_evaluate_models.py` + `test_workbench.py` (sau encoder blocker) | 41 passed |
| 6 | Parser smoke probe: good / duplicate line / `average recall50 = 1e` | EXTRACTED_VERIFIED / UNPARSED / UNPARSED |
| 7 | Downloader dry-run probe: encoder | `SKIPPED: unresolved checkpoint mapping (encoder / fashioniq)` |
| 8 | Registry load: models / reference rows / version | 16 / 33 / `extract-79c32139b606` |
| 9 | Frontend `npx tsc --noEmit` | rc=0 |
| 10 | Frontend production build (`npx vite build`) | pass |
| 11 | `build_paper_registry.py` rebuild | 33 rows, identity đúng |

Lưu ý trung thực: một run suite giữa chừng (203 failed) là INVALID — pytest import
`registry.py` trước khi file được sửa giữa lúc chạy; kết quả không dùng làm evidence.
Các run dùng làm evidence: 897 passed (trước 2 test parser mới) và 899 passed (final,
mục 5a).

## 5a. Kết quả run cuối

Full suite run cuối (toàn bộ thay đổi của vòng này, gồm 5 test encoder đã sửa và
2 test parser mới): **899 passed, 4 warnings, 0 failed** (17m55s). Suite 899 =
897 (run trước) + 2 test parser mới; 4 warnings đều là DeprecationWarning từ
`runtime.py` metadata (đã tồn tại từ trước vòng này).

## 5b. Vòng 2 (sửa 10 defect + phản hồi 3 reviewer độc lập): kết quả suite

Full suite sau vòng sửa 10 defect: **907 passed, 4 warnings, 0 failed** (19m06s).
Suite 907 = 899 (vòng 1) + 8 test hồi quy mới (isolation báo cáo malformed,
mapping-unverified metrics-only, recipe-digest reuse, per-checkpoint blockers,
`--config` pc_readiness, GPU-pending per-model).

Full suite sau khi đóng hết phản hồi của ba reviewer độc lập (wiring test cho
`checkpoint_mapping_verified`, chuẩn hoá `category_metrics` lồng nhau, truthiness
`is True`, manifest regen theo splitlines): **910 passed, 4 warnings, 0 failed**
(19m51s). Suite 910 = 907 + 3 test mới (mutation-guard wiring mapping 2 hướng,
`{"dress": null}` category normalization).

## 6. Phạm vi thay đổi

Vòng 1: 30 modified + 3 untracked (16 audit reports, CONSOLIDATION, AUDIT_TRACKER,
3 holistic reports, manifest regenerated, gate.ts, gate_types.ts).
Vòng 2: 18 modified (paper_gate.py, main.py, evaluate_models.py, prepare_model.py,
pc_readiness.py, App.tsx, gate.ts, style.css, AUTOPILOT_GUIDE.md,
PC_TRANSFER_CHECKLIST.md, PC_RELEASE_READINESS.md, FINAL_REPORT.vi.md,
SOURCE_COVERAGE_MANIFEST.md regenerated, 5 test files) — manifest regen theo
splitlines (126 rows, 0 mismatch machine check).
