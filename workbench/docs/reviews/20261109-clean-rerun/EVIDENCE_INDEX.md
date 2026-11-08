# EVIDENCE_INDEX — phiên clean-rerun 2026-11-09

Mỗi dòng là một file **có thật trên đĩa** (đã kiểm bằng `test -f`), kèm cái nó chứng minh và gate
mà nó phục vụ. Không có mục nào là dự định.

Revision khi lập chỉ mục: `20a406a` (worktree = `ffb6ae8` + năm commit của các lane).
`TESTED_SOURCE_SHA` vẫn là `80842661` — xem G11.

## 1. Sản phẩm kiểm kê và quyết định

| file | bytes | chứng minh | gate | nguồn sinh ra |
|---|---:|---|---|---|
| `MODEL_MATRIX.csv` | 14080 | 29 hàng / 17 `model_id`, mỗi hàng có upstream URL+commit, checkpoint path+trạng thái, interpreter, run id, coverage, export depth, gate_status, decision, blocker | G0, G5, G6 | đọc `registry/models.yaml` + `artifacts/composition/` + `checkpoint_selection` |
| `RERUN_DECISIONS.csv` | 6749 | 29 quyết định REUSE/REEXPORT/RERUN_FULL/BLOCKED/DEFERRED kèm lý do và `required_depth` | G0, G6, §6 của FINAL_REPORT | dẫn xuất từ MODEL_MATRIX + luật trong `01_MASTER_PROMPT.en.md` §4 |
| `RUN_LOG.csv` | 1154 | **1 hàng**: encoder, head `ffb6ae8`, `6003/6003`, depth 100, `blas_verified=true`, `identity_schema_version=2`, `invariants=PASS`, `verdict=PASS`, và ghi chú wall-time **"not a baseline"** | G5, G7, G8 | `composition_eval --models encoder --subsets full --topk-depth 100 --keep-features` |
| `GATE_RESULTS.json` | 13242 | trạng thái 13 gate G0–G12, mỗi gate có state/scope/evidence/reject_reason/checked_at | tất cả | tổng hợp từ các file trong chỉ mục này |

## 2. Sửa lỗi evaluator (lane A)

| file | bytes | chứng minh | gate | nguồn |
|---|---:|---|---|---|
| `../20261109-lane-a-evaluator/LANE_A_REPORT.md` | 12132 | sáu lỗi: nguyên nhân, hàm đã đổi, test fails-before/passes-after, rủi ro còn lại; bảng đo chi phí cache | G2, G3, G4, G7 | phân tích + chạy test |
| `../20261109-lane-a-evaluator/RERUN_IMPACT.md` | 17133 | mọi run cũ trước `bf6a448`; **bảng đo BLAS-sensitivity**; per-interpreter BLAS; vì sao không backfill được | G3, G4, G5, G7 | probe + đọc artifact |
| `../../tests/test_review_defects_evaluator.py` | 24735 | 25 test; **19 failed / 2 passed khi stash hai file production, 25 passed khi có fix** | G2, G3, G4, G6 | `pytest … test_review_defects_evaluator.py` |

Test fails-before/passes-after: `19 failed, 2 passed` → `25 passed`.

## 3. Checkpoint resolution (CLIP4Cir)

| file | bytes | chứng minh | gate | nguồn |
|---|---:|---|---|---|
| `../../tests/test_clip4cir_checkpoint_resolution.py` | 9460 | adapter đọc member theo `role` từ registry, không dùng tên hard-code; role thiếu → `checkpoint_members_unresolved` | G1 (một phần), G2 | 5 passed |

## 4. Bằng chứng BLAS

| file | bytes | chứng minh | gate | nguồn |
|---|---:|---|---|---|
| `blas_gate.py` | 1611 | gate bước 0: chứng minh pin OpenBLAS hoạt động trong **đúng interpreter** (setter/getter/previous/effective/verified) | G3, G7 | chạy trong từng env |
| `RERUN_IMPACT.md` §1 | — | 14 interpreter, **không env nào fail pin** (20 → 1, verified) | G3 | cùng probe |
| `RERUN_IMPACT.md` §0 | — | bảng đo: encoder 29/400 rank đổi (max\|Δrank\| 5, 6/8 tie cluster), bảy model còn lại 0/400 với 0.000e+00 | G7 | gọi `score()` trên ma trận tổng hợp, seed 20261109 |

Cơ chế đã cô lập: GEMM theo batch (M>1) thay đổi theo thread count; matvec một query thì không;
`np.linalg.norm(axis=1)` bất biến.

## 5. Frontend

| file | bytes | chứng minh | gate | nguồn |
|---|---:|---|---|---|
| `../../frontend/tests/ui_smoke.py` | 46136 | 5 session (cold/fixture/composition/samples/mobile), chạy lại live: **cold 3, composition 8, mobile 4, samples 2, tất cả exit 0** | G9, G10 | `.venv/bin/python workbench/frontend/tests/ui_smoke.py --url http://127.0.0.1:5173 --session <s>` |
| `../../frontend/tests/light-workspace.test.cjs` | 8113 | 13 test hợp đồng: URL round-trip, identity của request K, menu theo depth, race response cũ, query vắng mặt | G9 | `node --test …` → 13 passed |
| `ui/composition-1440x900.png` | 318988 | bảng 9 hàng nằm gọn trong box, banner "4 of 9 composed runs in view" | G9 | `ui_smoke.py --session composition --shots` |
| `ui/overview-1920x1080.png` | 776572 | 17 dòng model ở FHD | G9, G10 | `--session cold` |
| `ui/overview-2560x1440.png` | 787837 | 17 dòng ở QHD, không tràn ngang | G10 | `--session cold` |
| `ui/samples-top50-1920x1080.png` | 1505579 | Top 50 render đủ 50 ứng viên | G9 | `--session samples` |
| `ui/mobile-390x844.png` | 486479 | 390 px, `body.scrollWidth == innerWidth` | G10 | `--session mobile` |

Số đo giao diện: `style.css:51-52` → 17px mặc định, 19px khi `data-density=large`; nền
`#f5f7fa` = `rgb(245,247,250)`; 17 entry từ `/api/models` và 17 dòng render.

## 6. Sweep và guard

| file | bytes | chứng minh | gate | nguồn |
|---|---:|---|---|---|
| `../20261109T012857+0700-sweep-final/full_suite_80842661.txt` | 47679 | serial sweep tại `80842661`: **18 failed, 2480 passed, 47 skipped = 2545**, exit 1, 3484.30 s | G5, G11 (lịch sử) | `pytest workbench/tests -q -p no:cacheprovider -rfE --tb=no` |
| `../20261109T012857+0700-sweep-final/FAILURES.md` | 9964 | 18 id đó, attribution từng cái; xác nhận **0 mới** so với `1129c58` | G5 | so sánh id |
| `../20261109T012857+0700-sweep-final/sweep_ffb6ae8_ABORTED.txt` | 2244 | sweep tại `ffb6ae8` **bị huỷ ở 22%**, không có count, exit code không ghi | G11 | log một phần |
| `../20261109T012857+0700-sweep-final/UI_HARNESS_REVERIFY_80842661.md` | 6681 | harness viết lại trong `80842661` và **đã pass**; coverage: overview/experiments/evaluation, không gồm samples/failures/notebook | G9 | chạy harness |
| `../20261108T212500+0700-sweep/FAILURES.md` | 16536 | 18 lỗi lịch sử + phân loại; nguồn của "defect #5" (test ordering) mà lane A **không** sửa | G5 | — |

## 7. Log của lượt chạy GPU

| file | bytes | chứng minh | gate | nguồn |
|---|---:|---|---|---|
| `logs/encoder.log` | 2058 | stdout của lượt encoder | G5, G7 | `composition_eval` |
| `logs/encoder.vram.txt` | 95 | VRAM đỉnh | G8 (dữ liệu thô, không phải baseline) | `nvidia-smi` trong lúc chạy |

`logs/conesep.log` (54 B) và `logs/conesep.vram.txt` (5 B) tồn tại nhưng **lượt conesep còn đang
chạy** tại thời điểm lập chỉ mục — chưa có kết quả, nên không được trích làm bằng chứng.

## 8. Những gì KHÔNG có trong chỉ mục này

- Không có differential native↔adapter theo từng model (G1 chưa chạy).
- Không có forward pass nào được đối chứng (G2 chưa chạy).
- Không có timing sạch (G8 chưa chạy — hai lane khác đang chạy).
- Không có sweep hoàn chỉnh tại revision hiện tại → guard đỏ (G11).
