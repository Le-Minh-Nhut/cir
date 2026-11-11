> File này là bàn giao việc còn lại. Trạng thái chốt ở mục "Trạng thái chốt" bên dưới; các mục "CHƯA SỬA" là việc phải làm tiếp.

# VIỆC CÒN LẠI — master audit 2026-11-11

## CHƯA SỬA — phải làm trước khi coi là xong
1. `FINAL_REPORT.vi.md:189` — bỏ mệnh đề sai "Khớp với `models_with_metrics = 17` trừ đi 2 model chỉ có roll-up". Sai hai lần: §5 liệt kê **8** rollup-only, và `airknow`/`pair` nằm **trong** 17.
2. `MODELS_REST_VERIFICATION.vi.md:473` — `models_with_metrics = 15` / `models_rollup_only = 13` đã cũ; payload sống là **17 / 8**.
3. `FINAL_REPORT.vi.md:339` — lệnh restart dùng `> /tmp/workbench_serve.log`, chính là cơ chế xoá log bằng chứng của các lần chết trước. Đổi sang `>>`.
4. §5 `FINAL_REPORT` và `WEB_SURFACE_VERIFICATION` khai "bốn lần chết còn log header" với đường dẫn `/tmp/*`; `/tmp/serve_handover.log` đã bị truncate (0 hit `exited with status -15`, 0 hit `4082632`). Chỉ **hai** bản copy đã commit: `raw/service-death-log.txt` (PID 3691345) và `raw/service-death-log-4.txt` (PID 4082632). Sửa claim để chỉ trích đường dẫn đã commit.
1a. `FINAL_REPORT.vi.md:189` — vẫn còn "Khớp với `models_with_metrics = 17` trừ đi 2 model chỉ có roll-up". Xoá mệnh đề, chỉ trích cột `native_executed` của `NATIVE_EVAL_MATRIX.csv`.
1b. `MODELS_REST_VERIFICATION.vi.md:473` — vẫn `models_with_metrics = 15` / `models_rollup_only = 13`; payload sống là **17 / 8**.
1c. `FINAL_REPORT.vi.md:339` — lệnh restart vẫn `> /tmp/workbench_serve.log` (truncate). Đổi `>>`.
1d. Hai log chết **chưa được commit**: `/tmp/workbench_serve_final.log` (PID 4040574) và `/tmp/serve_probe.log` (PID 4062957). Chép chữ ký sang `raw/service-death-log-2.txt` và `raw/service-death-log-3.txt` rồi commit; nếu không, restart tiếp theo sẽ xoá chúng.

## Chưa chốt được — cần người khác
5. **PAIR** — cần upstream công bố `model_vssp` + `MHTransformer_dis`. Đã quét 21 commit, không tồn tại. Xem `replay/pair_absence.py`.
6. **13 model G1/G2 `NOT_RUN`** — cần artifact parity; mỗi model cần GPU.
7. **5 ca cần rerun** (bảng §6 `FINAL_REPORT`): `clip4cir_rn50x4_noft`, `airknow` (2 dòng), `pair` (4 dòng), ENCODER `5dee7425`/`6a528e87` (superseded), `habit` `e0dd04a4` (artifact trỏ interpreter chết).
8. **LIMN 2 query mơ hồ** `7:3038`, `7:3790` — provenance upstream không cho phép chốt category. Cần metadata category từ upstream.
9. **Derived row chưa có marker `DERIVED` trên UI** — dữ liệu đúng ở `derived_from`, bề mặt web chưa hiện.
10. **39/43 dòng dùng đơn vị suy luận** — đã audit từng dòng, 0 khai báo bị bỏ sót; muốn hết suy luận phải sửa producer của từng artifact.
11. **Quét chéo 9 báo cáo CHƯA hoàn thành** — worker bị kill, file `raw/verify/42-*` không tồn tại. Đã sửa tay 6 chỗ (liệt kê trong `FINAL_REPORT` §6.1), phần còn lại chưa kiểm.
12. **Stack chết do SIGTERM bên ngoài, chưa xác định ai gửi.** Đã xác định `serve_workbench.py:152-166` là đường nhận. Đã loại trừ `jobs.py`, test/script trong repo, session reaping.

## Trạng thái chốt
- Revision được sweep: `be14393c97b100390a9a45c6b0a46857b24a78b9` (= `TESTED_SOURCE_SHA`)
- Sweep: `18 failed / 2800 passed / 47 skipped`, accounting gap **0**, NEW=0
- Guard: `workbench/tests/test_handoff_commands.py` → 8 passed
- Production diff `be14393..HEAD` ngoài `workbench/docs` và `workbench/tests`: **rỗng**
- Bất biến: `git status --porcelain` = `?? CLEANUP_REPORT.md.save`
- Bundle phục vụ: `index-DWpK0yeS.js` sha256 `7e34724ebf55092e36524b98980cca55e71f3ddbaa62489c65577f92e18a6a16`
- Web: http://127.0.0.1:5173/ · API: http://127.0.0.1:8000 (poll `/api/benchmark-runs`, **không** `/api/models`)

## Dọn dẹp đã làm
`/tmp`: 50G → 5.1G (**~45G** giải phóng). Đã xoá: 3 git worktree qua `git worktree remove` (`/tmp/cir1716`, `/tmp/cirb0`, `/tmp/wtc`), `cirwarm-{airknow,tgcir,clvc}` (~27G cache tái tạo được), `torchprobe-py`, `cudaenv`, `r6-scratch-*`, `cir-upstream-audit-*`, `gr_a`, `r12probe`, `asrenv`, và toàn bộ basetemp/probe scratch.
