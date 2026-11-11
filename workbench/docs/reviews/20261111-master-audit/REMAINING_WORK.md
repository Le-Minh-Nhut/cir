> File này là bàn giao việc còn lại. Trạng thái chốt ở mục "Trạng thái chốt" bên dưới; các mục "CHƯA LÀM" là việc phải làm tiếp.

# VIỆC CÒN LẠI — master audit 2026-11-11

## Đã xong trong đợt này (không còn là việc phải làm)

R01–R15 **đã sửa** — kết luận từng mục, file sửa và test chứng minh ở
`INDEPENDENT_REVIEW_RESPONSE_20261111.vi.md` §1. Bốn mục drift tài liệu của bản `REMAINING_WORK` trước
cũng đã sửa và đã kiểm lại: `FINAL_REPORT.vi.md` không còn mệnh đề "`models_with_metrics = 17` trừ đi 2
model chỉ có roll-up"; `MODELS_REST_VERIFICATION.vi.md` nay ghi `models_with_metrics = 17` /
`models_rollup_only = 8`; lệnh restart ở `FINAL_REPORT.vi.md` dùng `>>` (không truncate); và cả bốn chữ
ký chết stack được trích bằng đường dẫn **đã commit** (`raw/service-death-log.txt` PID 3691345, `-2`
4040574, `-3` 4062957, `-4` 4082632), `/tmp` chỉ còn là provenance. Không mục nào ở trên cần làm lại.

## CHƯA LÀM — phải làm trước khi coi là xong

1. **PAIR** — cần upstream công bố `model_vssp` + `MHTransformer_dis`. Đã quét 21 commit, không tồn
   tại. Xem `replay/pair_absence.py`.
2. **13 model G1/G2 `NOT_RUN`** — cần artifact parity; mỗi model cần GPU.
3. **5 ca cần rerun** (bảng §6 `FINAL_REPORT.vi.md`): `clip4cir_rn50x4_noft`, `airknow` (2 dòng),
   `pair` (4 dòng), ENCODER `5dee7425`/`6a528e87` (superseded), `habit` `e0dd04a4` (artifact trỏ
   interpreter chết).
4. **LIMN 2 query mơ hồ** `7:3038`, `7:3790` — provenance upstream không cho phép chốt category. Cần
   metadata category từ upstream.
5. **Derived row chưa có marker `DERIVED` trên UI** — dữ liệu đúng ở `derived_from`, bề mặt web chưa
   hiện. `grep -rn 'derived_from' workbench/frontend/*.ts*` → **0** kết quả, nên đây vẫn là việc mở.
6. **39/43 dòng dùng đơn vị suy luận** — đã audit từng dòng, 0 khai báo bị bỏ sót; muốn hết suy luận
   phải sửa producer của từng artifact.
7. **Quét chéo 9 báo cáo CHƯA hoàn thành** — worker bị kill, file `raw/verify/42-*` không tồn tại. Đã
   sửa tay 6 chỗ (liệt kê trong `FINAL_REPORT.vi.md` §6.1), phần còn lại chưa kiểm; `FINAL_REPORT.vi.md`
   §6.1 đã ghi rõ giới hạn này để người đọc không giả định nhất quán.
8. **R16 — còn chặn một phần:** (a) native mới (`airknow_run3cat_gpu_run{1..4}.json`,
   `tgcir_run3cat_gpu.json`, `clvc_run3cat_gpu.json`) chưa ingest vì cơ chế ingest không có khái niệm
   `superseded_by`; (b) CLVC/TG-CIR composition adapter luôn từ chối vì thiếu interpreter stack pin;
   (c) **chưa xác định ai gửi SIGTERM** giết stack — đã xác định `serve_workbench.py:152-166` là đường
   nhận, đã loại trừ `jobs.py`, test/script trong repo và session reaping.
9. **uvicorn `:8765` cũ vẫn sống** — thiếu route mới (`/api/runs/parity-gate` → 404 so với 200 ở
   `:8000`). Nhận diện bằng cổng và hành vi route, không bằng PID. Cần kill tiến trình đó; PID đổi
   sau restart nên không ghim số.

## Trạng thái chốt
- Revision được sweep: `c755673487744876d12fdd5a676a220329bd3468` (= `TESTED_SOURCE_SHA`; **chính là**
  revision của sweep, không phải một revision khác được trích dẫn). Lượt `cd980dc` **bị thay thế**.
- Sweep: `2914 passed / 18 failed / 47 skipped`, **2979 collected**, accounting gap **0**, `^ERROR ` 0,
  `Errno 28` 0, NEW=0 / DROPPED=0 / COMMON=18 so với baseline `20261110-e2e-repair/full_suite_68ad75e.txt`.
  Bản ghi: `raw/verify/70-sweep-final.md`.
- Guard: `workbench/tests/test_handoff_commands.py` → kỳ vọng **8 passed** sau commit tài liệu này (7
  passed + 1 guard SHA đỏ theo thiết kế trước khi pin).
- Production diff `c755673..HEAD` ngoài `workbench/docs` và `workbench/tests`: **rỗng** (đã kiểm lại
  trước khi pin); bất biến này (swept revision = `TESTED_SOURCE_SHA`, và nó chứa mọi byte sản xuất đang
  có) là thứ phải giữ, không phải một SHA cũ hơn.
- Bất biến: `git status --porcelain` = `?? CLEANUP_REPORT.md.save`
- Bundle phục vụ: `index-DeTvxLlD.js` sha256 `1503e8911122587ba8c81e8ecf93d151312fda41e12ff5ba9580ef1797e18155` — đọc lại
  hôm nay bằng `curl -s http://127.0.0.1:5173/ | grep -o 'assets/index-[^"]*\.js'` rồi `sha256sum` trên `dist/`;
  đây là **fact đo được**, không phải hằng số: mỗi lần build lại phải đọc lại đúng hai lệnh đó.
- Web: http://127.0.0.1:5173/ · API: http://127.0.0.1:8000 (poll `/api/benchmark-runs`, **không** `/api/models`)

## Dọn dẹp đã làm
`/tmp`: 50G → 5.1G (**~45G** giải phóng). Đã xoá: 3 git worktree qua `git worktree remove` (`/tmp/cir1716`, `/tmp/cirb0`, `/tmp/wtc`), `cirwarm-{airknow,tgcir,clvc}` (~27G cache tái tạo được), `torchprobe-py`, `cudaenv`, `r6-scratch-*`, `cir-upstream-audit-*`, `gr_a`, `r12probe`, `asrenv`, và toàn bộ basetemp/probe scratch.
