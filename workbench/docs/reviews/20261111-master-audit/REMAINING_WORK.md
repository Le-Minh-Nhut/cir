> File này là bàn giao việc còn lại. Trạng thái chốt ở mục "Trạng thái chốt" bên dưới; các mục "CHƯA LÀM" là việc phải làm tiếp.

# VIỆC CÒN LẠI — master audit 2026-11-11

## Đã xong trong đợt này (không còn là việc phải làm)

R01–R15 **đã sửa**, và R16 vế (a) **đã xong** ở `358c3fc` (ingest bảy run review) — kết luận từng mục,
file sửa và test chứng minh ở `INDEPENDENT_REVIEW_RESPONSE_20261111.vi.md` §17. Bốn mục drift tài liệu
của bản `REMAINING_WORK` trước cũng đã sửa và đã kiểm lại: `FINAL_REPORT.vi.md` không còn mệnh đề "`models_with_metrics = 17` trừ đi 2
model chỉ có roll-up"; `MODELS_REST_VERIFICATION.vi.md` nay ghi `models_with_metrics = 17` /
`models_rollup_only = 8`; lệnh restart ở `FINAL_REPORT.vi.md` dùng `>>` (không truncate); và cả bốn chữ
ký chết stack được trích bằng đường dẫn **đã commit** (`raw/service-death-log.txt` PID 3691345, `-2`
4040574, `-3` 4062957, `-4` 4082632), `/tmp` chỉ còn là provenance. Không mục nào ở trên cần làm lại.

## CHƯA LÀM — phải làm trước khi coi là xong

1. **PAIR** — cần upstream công bố `model_vssp` + `MHTransformer_dis`. Đã quét 21 commit, không tồn
   tại. Xem `replay/pair_absence.py`.
2. **13 model G1/G2 `NOT_RUN`** — cần artifact parity; mỗi model cần GPU.
3. **6 ca cần rerun** (bảng §6 `FINAL_REPORT.vi.md`): `clip4cir_rn50x4_noft`, `airknow` (3 dòng counted
   — n05 và n08 khác checkpoint, cộng một cặp mâu thuẫn trên `fiq_n05`), `clvc_net` (hai số counted mâu
   thuẫn, xem mục 9), `pair` (4 dòng), ENCODER `5dee7425`/`6a528e87` (superseded), `habit` `e0dd04a4`
   (artifact trỏ interpreter chết).
4. **LIMN 2 query mơ hồ** `7:3038`, `7:3790` — provenance upstream không cho phép chốt category. Cần
   metadata category từ upstream.
5. **Derived row chưa có marker `DERIVED` trên UI** — dữ liệu đúng ở `derived_from`, bề mặt web chưa
   hiện. `grep -rn 'derived_from' workbench/frontend/*.ts*` → **0** kết quả, nên đây vẫn là việc mở.
6. **46/50 dòng dùng đơn vị suy luận** — `list_native_measurements()` tại `358c3fc` cho
   `unit_inferred 46` / `unit_declared 4` / `unit_sibling_key 2` trên 50 dòng có metrics (trước ingest là
   39/43). Đã audit từng dòng, 0 khai báo bị bỏ sót; muốn hết suy luận phải sửa producer của từng
   artifact.
7. **Quét chéo 9 báo cáo CHƯA hoàn thành** — worker bị kill, file `raw/verify/42-*` không tồn tại. Đã
   sửa tay 6 chỗ (liệt kê trong `FINAL_REPORT.vi.md` §6.1), phần còn lại chưa kiểm; `FINAL_REPORT.vi.md`
   §6.1 đã ghi rõ giới hạn này để người đọc không giả định nhất quán.
8. **R16 — đã đóng vế (a), còn (b) và (c):** (a) **ĐÃ LÀM** ở `358c3fc` — bảy run GPU
   (`airknow_run3cat_gpu_run{1..4}.json`, `tgcir_run3cat_gpu.json`, `clvc_run3cat_gpu.json`,
   `clvc_run3cat_gpu_run2.json`) nay được `native_measurements.py` phục vụ qua
   `_review_run_payload`/`_review_run_rows` (`source_kind: review_run`); (b) CLVC/TG-CIR composition
   adapter luôn từ chối vì thiếu interpreter stack pin; (c) **chưa xác định ai gửi SIGTERM** giết stack —
   đã xác định `serve_workbench.py:152-166` là đường nhận, đã loại trừ `jobs.py`, test/script trong repo
   và session reaping.
9. **CLVC có HAI số counted mâu thuẫn — phải chốt hoặc rerun.** Cùng `model_id: clvc_net`, cùng
   `protocol_id: fashioniq_val_split`, cả hai `counted: true`: `29.970746517160162`
   (`checkpoint_selection/20261106T002617+0700/measurements/clvc_run1.json`) và `30.02079703540087`
   (`workbench/docs/reviews/20261111-master-audit/raw/clvc_run3cat_gpu.json`). API ghi `conflict_with`
   cho **cả hai** và **không** chọn theo độ mới (`native_measurements.py`, `_mark_duplicates` nhánh
   `# 3d`). Cần: (i) chủ dự án chốt run canonical, hoặc (ii) rerun CLVC và so **từng qid** với cả hai.
   Cho tới đó **không** được trích "CLVC = 29.97" hay "CLVC = 30.02" như số đã chốt — xem
   `CENSUS_20261111.vi.md` §1.2. (Air-Know có cùng dạng hai row counted, nhưng đã có `variance_group`
   n=4 / sd `0.26728186433775697` nên mức bất định **đã đo**; CLVC thì chưa.)
10. **uvicorn `:8765` cũ vẫn sống** — thiếu route mới (`/api/runs/parity-gate` → 404 so với 200 ở
   `:8000`). Nhận diện bằng cổng và hành vi route, không bằng PID. Cần kill tiến trình đó; PID đổi
   sau restart nên không ghim số.
11. **UI KHÔNG hiện `conflict_with` / `variance_group` — lỗ hổng thật, cần chủ frontend** (xác nhận
   2026-11-11 sau khi restart `:8000` tại `727f1d1`). `native_measurements.py` phục vụ hai field đó trên
   row (`_review_run_payload`, `_mark_duplicates` nhánh `# 3d` / `# 3b`), nhưng bề mặt web bỏ qua:
   `grep -rn 'conflict_with\|variance_group\|duplicate_reason' workbench/frontend/*.ts*` → **0** kết quả.
   Hệ quả đo được trên `:5173` (`?area=evaluation&view=native&protocol=fashioniq_val_split`):
   `clvc_net` hiện **ba dòng** — `29.97%` (`clvc_run1.json`), `30.02%` (`clvc_run3cat_gpu.json`), và
   `30.02%` thứ ba gắn nhãn "trùng với clvc_run3cat_gpu.json, không đếm"
   (`workbench/frontend/EvaluationPage.tsx:662-663`); **không** dòng nào nói hai số 29.97/30.02 đang
   **mâu thuẫn** với nhau, nên người đọc thấy hai số như cùng đã chốt. `airknow` tương tự với
   `52.5466` / `52.9241`. **Hành vi đúng:** dòng mang `conflict_with` phải hiện một dấu hiệu mâu thuẫn
   (badge + liệt kê path đối thủ) và một câu "chưa chốt"; dòng mang `variance_group` phải hiện
   `n`/`mean`/`sd`/`spread` cạnh số của draw giữ lại. Hai field này cũng phải có trong `NativeRowLike`
   (`workbench/frontend/scope.ts:194-198`) nếu inventory cần dùng — hiện interface chỉ khai
   `model_id`/`protocol_id`/`metrics`. **Không** sửa ở đợt của tôi: ngoài phạm vi ghi.
12. **Nhãn "Số đo native không ghi protocol" — ĐÃ KIỂM, KHÔNG phải defect (giữ nguyên).** Kiểm bằng
   payload sống `:8000/api/native-measurements` tại `727f1d1`: `clvc_net` có 5 row — 3 row
   `protocol_id: fashioniq_val_split` (`clvc_run1.json` 29.970746517160162 counted, `clvc_run3cat_gpu.json`
   30.02079703540087 counted, `clvc_run3cat_gpu_run2.json` duplicate) và **2 row `protocol_id: null`**
   (`rollup:all_measured_summary.json`, `rollup:selection_decisions.json`, cả hai r10 29.9707,
   `counted: false`); `tgcir` có 4 row theo đúng khuôn đó (2 row null: cùng hai file roll-up, r10
   49.7208). `nativeUnattributedFor()` (`workbench/frontend/scope.ts:230-233`) lọc
   `statesNumber ∧ model_id ∧ !protocol_id` ⇒ chọn **đúng 2 row null** mỗi model, **không** nuốt row nào
   có `protocol_id`. Nhãn ở `ModelInventory.tsx:255-259` và `:316-317` vì thế **đúng theo dữ liệu**.
   Chỗ dễ đọc sai (không phải lỗi): hai row null đó là **roll-up** phát lại số của chính row có protocol,
   nên "2 số đo native" trùng giá trị với số attributed — nếu muốn rõ hơn thì câu nên nói "2 dòng roll-up
   chưa ghi protocol", nhưng đó là cải thiện câu chữ, không phải sửa logic, và **không** được đổi bộ lọc.

## Trạng thái chốt
- Revision được sweep: `c755673487744876d12fdd5a676a220329bd3468` (= `TESTED_SOURCE_SHA`; **chính là**
  revision của sweep, không phải một revision khác được trích dẫn). Lượt `cd980dc` **bị thay thế**.
- Sweep: `2914 passed / 18 failed / 47 skipped`, **2979 collected**, accounting gap **0**, `^ERROR ` 0,
  `Errno 28` 0, NEW=0 / DROPPED=0 / COMMON=18 so với baseline `20261110-e2e-repair/full_suite_68ad75e.txt`.
  Bản ghi: `raw/verify/70-sweep-final.md`.
- Guard: `workbench/tests/test_handoff_commands.py` → kỳ vọng **8 passed** sau commit tài liệu này (7
  passed + 1 guard SHA đỏ theo thiết kế trước khi pin).
- Production diff `c755673..HEAD` ngoài `workbench/docs` và `workbench/tests`: **KHÔNG còn rỗng** —
  `358c3fc` sửa `workbench/backend/native_measurements.py` (+261/−10) sau revision đã sweep. Nghĩa là
  lượt sweep ghim ở `c755673` **không** bao trùm thay đổi ingest; bằng chứng cho nó là
  `workbench/tests/test_native_ingest_r16.py` (16 passed khi chạy riêng). Ai pin lại `TESTED_SOURCE_SHA`
  phải sweep lại từ `358c3fc` trở đi, hoặc ghim đúng một revision **sau** nó.
- Bất biến: `git status --porcelain` = `?? CLEANUP_REPORT.md.save`
- Bundle phục vụ: `index-DeTvxLlD.js` sha256 `1503e8911122587ba8c81e8ecf93d151312fda41e12ff5ba9580ef1797e18155` — đọc lại
  hôm nay bằng `curl -s http://127.0.0.1:5173/ | grep -o 'assets/index-[^"]*\.js'` rồi `sha256sum` trên `dist/`;
  đây là **fact đo được**, không phải hằng số: mỗi lần build lại phải đọc lại đúng hai lệnh đó.
- Web: http://127.0.0.1:5173/ · API: http://127.0.0.1:8000 (poll `/api/benchmark-runs`, **không** `/api/models`)

## Dọn dẹp đã làm
`/tmp`: 50G → 5.1G (**~45G** giải phóng). Đã xoá: 3 git worktree qua `git worktree remove` (`/tmp/cir1716`, `/tmp/cirb0`, `/tmp/wtc`), `cirwarm-{airknow,tgcir,clvc}` (~27G cache tái tạo được), `torchprobe-py`, `cudaenv`, `r6-scratch-*`, `cir-upstream-audit-*`, `gr_a`, `r12probe`, `asrenv`, và toàn bộ basetemp/probe scratch.
