# EVIDENCE_INDEX — phiên clean-rerun 2026-11-09

Mỗi dòng là một file **có thật trên đĩa** (đã kiểm bằng `test -f`), kèm cái nó chứng minh và gate
mà nó phục vụ. Không có mục nào là dự định.

Revision khi lập chỉ mục: `TESTED_SOURCE_SHA` là `2a832bc0efe482d1536e466232a3e2803929d1d8`, đã đo bằng
một lượt sweep serial tại chính revision đó (`full_suite_2a832bc.txt`: 2510 passed / 21 failed / 47 skipped
= 2578 collected, exit 1) và guard `test_handoff_commands.py` sau đó **8 passed**. Bốn lượt chạy sớm hơn
(trước khi HEAD chuyển `ffb6ae8` → `20b83e5` lúc 03:53 do lane khác) đã được đối chiếu lại:
`scoring_module_source_hash` và `condition_transform_source_hash` ghi trong artifact bằng đúng file nguồn
hiện tại.

Tám lượt chạy đã xong; mục 1 và 6 ghi run id + đường dẫn artifact cho từng lượt.

## 1. Sản phẩm kiểm kê và quyết định

| file | bytes | chứng minh | gate | nguồn sinh ra |
|---|---:|---|---|---|
| `MODEL_MATRIX.csv` | 14949 | 29 hàng / 17 `model_id`; mỗi hàng có upstream URL+commit, checkpoint path+trạng thái, interpreter, coverage, export depth, gate_status, decision, blocker, **và cột `previous_run_id` mới ngay sau `variant`**. Mười hàng đã rerun nay ghi run id MỚI, `candidate_topk=true`, `export_depth=100`, `decision=RERUN_FULL_DONE`, và `blocker` nêu **sự thật còn lại** thay vì việc phải làm: encoder 12727/18009 rank đổi (run cũ là measurement KHÁC); limn 6001/6003 với 2 query không chấm được (`7:3038`, `7:3790`) và cohort khác `10173dc4`; ptha_mtst encode không pin được (env chỉ có MKL) | G0, G5, G6 | `RUN_LOG.csv` + artifact; đã kiểm `grep -c` cho cả tám run id |
| `RERUN_DECISIONS.csv` | 7063 | 29 hàng; mười hàng nay là `RERUN_FULL_DONE` với `requires_rerun=NO` và `required_depth=100` (đã đạt), `evidence_path` trỏ tới `RUN_LOG.csv` + `rank_diffs/` + thư mục run. Còn **2** hàng `RERUN_FULL` (airknow `fiq_n05`/`fiq_n08`: adapter có, chưa có forward GPU), **14** BLOCKED, **3** DEFERRED. Cột `previous_run_id` đã có sẵn từ trước và được giữ nguyên | G0, G6, §6 của FINAL_REPORT | dẫn xuất từ `RUN_LOG.csv` |
| `RUN_LOG.csv` | 10295 | **8 hàng**, mỗi hàng: model, variant, run id, command, interpreter, env, HEAD lúc chạy, start/end, wall-time (mọi hàng ghi **"not a baseline"**), throughput `not_measured (concurrent load)`, VRAM, coverage, depth, pin, `identity_schema_version`, invariants, verdict, notes | G5, G6, G7, G8 | 8 lượt `composition_eval --subsets full --topk-depth 100 --keep-features` |
| `GPU_LANE_REPORT.md` | 5860 | báo cáo lane GPU: bảng gate BLAS, bảng 8 lượt chạy, kiểm tra invariant, những gì KHÔNG làm | G5, G6, G7 | tổng hợp từ RUN_LOG + rank_diffs |
| `GATE_RESULTS.json` | 18519 | trạng thái 13 gate G0–G12, mỗi gate có state/scope/evidence/reject_reason/checked_at | tất cả | tổng hợp từ các file trong chỉ mục này |


## 2. Tám lượt chạy đã hoàn tất — run id và artifact

Mọi đường dẫn dưới đây nằm trong `workbench/artifacts/composition/` (dữ liệu gitignored, **không**
commit) và đã được kiểm `os.path.exists` trước khi ghi bảng. `6001/6003` của limn là trung thực:
hai query `7:3038`, `7:3790` không chấm được và có lý do ghi trong artifact.

| model | variant | run_id | coverage | summary · ranks · candidate manifest |
|---|---|---|---|---|
| `encoder` | `fashioniq` | `5dee7425dd46c5f154808aac` | `6003/6003` | `../../../artifacts/composition/summary/encoder_fashioniq_5dee7425dd46c5f154808aac/full.json`<br>`…/runs/5dee7425dd46c5f154808aac/ranks/encoder_fashioniq.json`<br>`…/runs/5dee7425dd46c5f154808aac/candidates/encoder_fashioniq/multimodal.manifest.json` |
| `conesep` | `fiq_n02` | `a4f786d8af4d46ac2a85ab9c` | `6003/6003` | `../../../artifacts/composition/summary/conesep_fiq_n02_a4f786d8af4d46ac2a85ab9c/full.json`<br>`…/runs/a4f786d8af4d46ac2a85ab9c/ranks/conesep_fiq_n02.json`<br>`…/runs/a4f786d8af4d46ac2a85ab9c/candidates/conesep_fiq_n02/multimodal.manifest.json` |
| `habit` | `fiq_n02` | `c6eb9d5775ae636a3b8addbc` | `6003/6003` | `../../../artifacts/composition/summary/habit_fiq_n02_c6eb9d5775ae636a3b8addbc/full.json`<br>`…/runs/c6eb9d5775ae636a3b8addbc/ranks/habit_fiq_n02.json`<br>`…/runs/c6eb9d5775ae636a3b8addbc/candidates/habit_fiq_n02/multimodal.manifest.json` |
| `intent` | `fiq_n02` | `4901da139ac40b874d2363c5` | `6003/6003` | `../../../artifacts/composition/summary/intent_fiq_n02_4901da139ac40b874d2363c5/full.json`<br>`…/runs/4901da139ac40b874d2363c5/ranks/intent_fiq_n02.json`<br>`…/runs/4901da139ac40b874d2363c5/candidates/intent_fiq_n02/multimodal.manifest.json` |
| `hint` | `fashioniq` | `f461f391c6d4696011be8067` | `6003/6003` | `../../../artifacts/composition/summary/hint_fashioniq_f461f391c6d4696011be8067/full.json`<br>`…/runs/f461f391c6d4696011be8067/ranks/hint_fashioniq.json`<br>`…/runs/f461f391c6d4696011be8067/candidates/hint_fashioniq/multimodal.manifest.json` |
| `limn` | `base_iter0_all_categories` | `9e4918a6b5c243412cbc58c9` | `6001/6003` | `../../../artifacts/composition/summary/limn_base_iter0_all_categories_9e4918a6b5c243412cbc58c9/full.json`<br>`…/runs/9e4918a6b5c243412cbc58c9/ranks/limn_base_iter0_all_categories.json`<br>`…/runs/9e4918a6b5c243412cbc58c9/candidates/limn_base_iter0_all_categories/multimodal.manifest.json` |
| `ptha_mtst` | `cirr_transfer` | `f54a77450ebd2126590d4385` | `6003/6003` | `../../../artifacts/composition/summary/ptha_mtst_cirr_transfer_f54a77450ebd2126590d4385/full.json`<br>`…/runs/f54a77450ebd2126590d4385/ranks/ptha_mtst_cirr_transfer.json`<br>`…/runs/f54a77450ebd2126590d4385/candidates/ptha_mtst_cirr_transfer/multimodal.manifest.json` |
| `tme` | `fashioniq_0` | `cd1a696d8bca0fc4bce6925b` | `6003/6003` | `../../../artifacts/composition/summary/tme_fashioniq_0_cd1a696d8bca0fc4bce6925b/full.json`<br>`…/runs/cd1a696d8bca0fc4bce6925b/ranks/tme_fashioniq_0.json`<br>`…/runs/cd1a696d8bca0fc4bce6925b/candidates/tme_fashioniq_0/multimodal.manifest.json` |

## 3. So sánh rank từng query (pinned vs pre-pin)

Sinh bằng `rank_diff.py`, lưu nguyên stdout — mỗi con số trong bảng của `BLAS_IDENTITY_FINDINGS.md`
dẫn về đúng một file ở đây. `queries_compared` = 18009 cho mọi model (6003 × 3 điều kiện).

| model | file | queries compared | changed ranks | max abs Δrank (mm / image / text) |
|---|---|---:|---:|---|
| `encoder` | `rank_diffs/encoder_6a528e87_vs_5dee7425.json` | 18009 | 12727 | 805 / 852 / 1706 |
| `conesep` | `rank_diffs/conesep_30f6de42_vs_a4f786d8.json` | 18009 | 17 | 1 / 1 / 1 |
| `habit` | `rank_diffs/habit_497c25c5_vs_c6eb9d57.json` | 18009 | 0 | 0 |
| `intent` | `rank_diffs/intent_37ee72aa_vs_4901da13.json` | 18009 | 0 | 0 |
| `hint` | `rank_diffs/hint_4bc9893a_vs_f461f391.json` | 18009 | 0 | 0 |
| `ptha_mtst` | `rank_diffs/ptha_mtst_315dd590_vs_f54a7745.json` | 18009 | 46 | 2 / 1 / 1 |
| `tme` | `rank_diffs/tme_ad477312_vs_cd1a696d.json` | 18009 | 0 | 0 |
| `limn` | `rank_diffs/limn_cohort_note.json` | — | không so được | cohort 5884 vs 6001 khác nhau, ghi rõ trong file |

Tệp `.err` cạnh mỗi `.json` đều **0 byte**, tức không lượt nào in cảnh báo ra stderr.

## 4. Sửa lỗi evaluator (lane A)

| file | bytes | chứng minh | gate | nguồn |
|---|---:|---|---|---|
| `../20261109-lane-a-evaluator/LANE_A_REPORT.md` | 12132 | sáu lỗi: nguyên nhân, hàm đã đổi, test fails-before/passes-after, rủi ro còn lại; bảng đo chi phí cache | G2, G3, G4, G7 | phân tích + chạy test |
| `../20261109-lane-a-evaluator/RERUN_IMPACT.md` | 17133 | mọi run cũ trước `bf6a448`; **bảng đo BLAS-sensitivity**; per-interpreter BLAS; vì sao không backfill được | G3, G4, G5, G7 | probe + đọc artifact |
| `../../../tests/test_review_defects_evaluator.py` | 24735 | 25 test; **19 failed / 2 passed khi stash hai file production, 25 passed khi có fix** | G2, G3, G4, G6 | `pytest … test_review_defects_evaluator.py` |

Test fails-before/passes-after: `19 failed, 2 passed` → `25 passed`.

## 5. Checkpoint resolution (CLIP4Cir)

| file | bytes | chứng minh | gate | nguồn |
|---|---:|---|---|---|
| `../../../tests/test_clip4cir_checkpoint_resolution.py` | 9460 | adapter đọc member theo `role` từ registry, không dùng tên hard-code; role thiếu → `checkpoint_members_unresolved` | G1 (một phần), G2 | 5 passed |

## 6. Bằng chứng BLAS

| file | bytes | chứng minh | gate | nguồn |
|---|---:|---|---|---|
| `blas_gate.py` | 1611 | gate bước 0: chứng minh pin OpenBLAS hoạt động trong **đúng interpreter** (setter/getter/previous/effective/verified) | G3, G7 | chạy trong từng env |
| `RERUN_IMPACT.md` §1 | — | 14 interpreter, **không env nào fail pin** (20 → 1, verified) | G3 | cùng probe |
| `RERUN_IMPACT.md` §0 | — | bảng đo: encoder 29/400 rank đổi (max\|Δrank\| 5, 6/8 tie cluster), bảy model còn lại 0/400 với 0.000e+00 | G7 | gọi `score()` trên ma trận tổng hợp, seed 20261109 |

Cơ chế đã cô lập: GEMM theo batch (M>1) thay đổi theo thread count; matvec một query thì không;
`np.linalg.norm(axis=1)` bất biến.

## 7. Frontend

| file | bytes | chứng minh | gate | nguồn |
|---|---:|---|---|---|
| `../../../frontend/tests/ui_smoke.py` | 46136 | 5 session (cold/fixture/composition/samples/mobile), chạy lại live: **cold 3, composition 8, mobile 4, samples 2, tất cả exit 0** | G9, G10 | `.venv/bin/python workbench/frontend/tests/ui_smoke.py --url http://127.0.0.1:5173 --session <s>` |
| `../../../frontend/tests/light-workspace.test.cjs` | 8113 | 13 test hợp đồng: URL round-trip, identity của request K, menu theo depth, race response cũ, query vắng mặt | G9 | `node --test …` → 13 passed |
| `ui/composition-1440x900.png` | 318988 | bảng 9 hàng nằm gọn trong box, banner "4 of 9 composed runs in view" | G9 | `ui_smoke.py --session composition --shots` |
| `ui/overview-1920x1080.png` | 776572 | 17 dòng model ở FHD | G9, G10 | `--session cold` |
| `ui/overview-2560x1440.png` | 787837 | 17 dòng ở QHD, không tràn ngang | G10 | `--session cold` |
| `ui/samples-top50-1920x1080.png` | 1505579 | Top 50 render đủ 50 ứng viên | G9 | `--session samples` |
| `ui/mobile-390x844.png` | 486479 | 390 px, `body.scrollWidth == innerWidth` | G10 | `--session mobile` |

Số đo giao diện: `style.css:51-52` → 17px mặc định, 19px khi `data-density=large`; nền
`#f5f7fa` = `rgb(245,247,250)`; 17 entry từ `/api/models` và 17 dòng render.

## 8. Log của lượt chạy GPU

| file | bytes | chứng minh | gate | nguồn |
|---|---:|---|---|---|
| `logs/encoder.log` | 2058 | stdout đầy đủ của lượt encoder (JSON cuối) | G5, G7 | `composition_eval --models encoder …` |
| `logs/encoder.vram.txt` | 95 | VRAM đỉnh 1882 MiB, sampler 10 s | G8 (dữ liệu thô, **không** phải baseline) | `nvidia-smi` |
| `logs/conesep.log` | 1947 | stdout lượt conesep, run `a4f786d8af4d46ac2a85ab9c` | G5, G6 | `composition_eval --models conesep …` |
| `logs/conesep.vram.txt` | 5 | VRAM đỉnh 4494 MiB | G8 (thô) | `nvidia-smi` |
| `logs/habit.log` | 1942 | stdout lượt habit, run `c6eb9d5775ae636a3b8addbc` | G5, G6 | `composition_eval --models habit …` |
| `logs/habit.vram.txt` | 861 | chuỗi sampler 10 s, đỉnh 4502 MiB | G8 (thô) | `nvidia-smi` |
| `logs/intent.log` | 1944 | stdout lượt intent, run `4901da139ac40b874d2363c5` | G5, G6 | `composition_eval --models intent …` |
| `logs/intent.vram.txt` | 837 | chuỗi sampler 10 s, đỉnh 4502 MiB | G8 (thô) | `nvidia-smi` |
| `logs/hint.log` | 1944 | stdout lượt hint, run `f461f391c6d4696011be8067` | G5, G6 | `composition_eval --models hint …` |
| `logs/limn.log` | 2119 | stdout lượt limn, run `9e4918a6b5c243412cbc58c9` | G5, G6 | `composition_eval --models limn …` |
| `logs/limn.vram.observed.txt` | 1066 | ba lần đọc `nvidia-smi` thủ công (5270 → 7036 → 7154 MiB), **không phải sampler**; ghi rõ để con số 7154 trong `RUN_LOG.csv` có file dẫn | G8 (thô) | đọc tay trong lúc chạy |
| `logs/ptha_mtst.log` | 2587 | stdout lượt ptha_mtst, run `f54a77450ebd2126590d4385` | G5, G6 | `composition_eval --models ptha_mtst …` |
| `logs/tme.log` | 2548 | stdout lượt tme, run `cd1a696d8bca0fc4bce6925b` | G5, G6 | `composition_eval --models tme …` |

Ba lượt (**hint**, **ptha_mtst**, **tme**) không có file VRAM: sampler không được nối lại sau khi
HEAD đổi giữa phiên. `RUN_LOG.csv` ghi `not_sampled (no sampler wired for this run)` ở đúng ba ô đó —
không suy ra số. Limn có số nhưng là **đọc tay**, và cột của nó nói đúng như vậy.

## 9. Sweep và guard

| file | bytes | chứng minh | gate | nguồn |
|---|---:|---|---|---|
| `../20261109T012857+0700-sweep-final/full_suite_80842661.txt` | 47679 | serial sweep tại `80842661`: **18 failed, 2480 passed, 47 skipped = 2545**, exit 1, 3484.30 s | G5, G11 (lịch sử) | `pytest workbench/tests -q -p no:cacheprovider -rfE --tb=no` |
| `../20261109T012857+0700-sweep-final/FAILURES.md` | 9964 | 18 id đó, attribution từng cái; xác nhận **0 mới** so với `1129c58` | G5 | so sánh id |
| `../20261109T012857+0700-sweep-final/sweep_ffb6ae8_ABORTED.txt` | 2244 | sweep tại `ffb6ae8` **bị huỷ ở 22%**, không có count, exit code không ghi | G11 | log một phần |
| `../20261109T012857+0700-sweep-final/UI_HARNESS_REVERIFY_80842661.md` | 6681 | harness viết lại trong `80842661` và **đã pass**; coverage: overview/experiments/evaluation, không gồm samples/failures/notebook | G9 | chạy harness |
| `../20261108T212500+0700-sweep/FAILURES.md` | 16536 | 18 lỗi lịch sử + phân loại; nguồn của "defect #5" (test ordering) mà lane A **không** sửa | G5 | — |

## 10. Những gì KHÔNG có trong chỉ mục này

- Không có differential native↔adapter theo từng model (G1 chưa chạy).
- Không có forward pass nào được đối chứng (G2 chưa chạy).
- Không có timing sạch (G8 chưa chạy — hai lane khác đang chạy).
- **ĐÃ CÓ** sweep hoàn chỉnh tại revision hiện tại: `full_suite_2a832bc.txt` (G11 nay là PASS). Sweep tại
  `ffb6ae8` bị huỷ ở 22% vẫn được giữ làm lịch sử, không phải lượt đo của cây hiện tại.

## 11. Phép đo hiệu năng (sau khi sweep thoát)

| file | bytes | chứng minh | gate | nguồn |
|---|---:|---|---|---|
| `perf_bench.py` | 16483 | harness: sweep decode worker 0/2/4/6/8 qua `adapters.prefetch_rows` thật, cold tách khỏi warm, **từ chối chạy khi box bận** (exit 2, đã kiểm), shape **khớp `model_habit.py:358-364`** (open→convert→transform→stack trong pool, không PNG), parity gate theo 2 digest (thứ tự hàng + mảng sau transform) | G8 | `python perf_bench.py …` |
| `perf_watcher.sh` | 3425 | watcher detached: đợi sweep thoát, ngủ 30 s, ghi loadavg, **kiểm lại** idleness, rồi đo và sinh tài liệu; fail-closed ghi `perf_run.FAILED` | G8 | `setsid nohup bash perf_watcher.sh` |
| `gen_performance_doc.py` | 11654 | sinh `PERFORMANCE.vi.md` **từ** `perf_raw.jsonl`; từ chối nếu <3 trial steady / thiếu cold / thiếu parity; đánh REJECTED và **không đề xuất** setting đổi digest | G8 | `python gen_performance_doc.py --raw … --out …` |
| `perf_sample_ids.txt` | — | 9 query id cố định (3 category, 3 điều kiện, rank 9 gần K=10, rank 49 gần K=50, batch cuối); kiểm drift: sửa 1 rank → generator/harness từ chối | G8 | dẫn xuất từ `runs/5dee7425…/ranks/encoder_fashioniq.json` |
| `perf_raw.jsonl` | 11243 | **dữ liệu thô hiện hành**: 5 setting × (1 cold + 3 steady) + parity rows + provenance (sha `0857744`, loadavg `0.47`). **Lần đo đầu đã bị thay thế** — nó đo một vòng PNG không tồn tại trong production và tách transform ra khỏi pool, nên đường cong bỏ mất stage song song hoá. Số cũ 103,8/235,9/428,3/432,0/429,2 và đề xuất `workers=4` **không còn hiệu lực**. | G8 | `perf_bench.py` |
| `PERF_TABLE.md` | 897 | bảng do harness sinh: median **554.5→1032.8→1756.2→2282.1→2688.1 img/s**, cả 5 setting IDENTICAL | G8 | `perf_bench.py` |
| `PERFORMANCE.vi.md` | 4276 | tài liệu **sinh tự động**, không gõ tay; nêu rõ phạm vi (decode+transform producer stage, không phải end-to-end) và đề xuất **`workers=8`** kèm cảnh báo đường cong chưa phẳng + ngân sách thread | G8 | `gen_performance_doc.py` |
| `logs/perf_run.log` | — | nhật ký watcher có timestamp: các lần "still waiting", `sweep exited after ~2540s`, `post-settle loadavg`, `perf_bench exit=0`, `gen_performance_doc exit=0` | G8 | watcher |
| `logs/perf_run.first_attempt.log` | — | lần chạy watcher đầu, bị dừng vì tôi sửa script khi nó đang chạy (bash đọc file tăng dần) — giữ lại thay vì xoá | G8 | — |

`INPUT_PIPELINE_AUDIT.vi.md` (14206 B) và `AGENT_LEDGER.json` (23400 B, 9 task, đủ 26 trường bắt buộc)
là hai tài liệu bàn giao còn lại; mọi mục trong đó ghi **mức bằng chứng** (static review / synthetic
differential / real differential / NOT VERIFIED), và G1/G2 vẫn NOT_RUN.
