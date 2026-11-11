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

**Đóng ở lượt pin `7babfd9`** (lượt đó nay superseded bởi `0691f3e`; mục này vẫn đóng — `scope.ts` chỉ đổi predicate `counts()`, không gỡ bề mặt `conflict_with`/`variance_group`): mục 11 cũ (UI không hiện `conflict_with` / `variance_group`) **đã xong** ở
`252b380` — `scope.ts` xuất `nativeConflictChips` / `nativeVarianceSummary` / `nativeRepeatLabel`,
`EvaluationPage.tsx:614-616` dùng chúng, `NativeRowLike` khai cả hai field, bundle phục vụ có chuỗi
`"Giá trị CHƯA chốt"`, và `test_web_scope_and_labels.py` (14) + `test_frontend_catalog.py` (18) pass.
Mục 12 cũ (nhãn native không protocol) **không phải defect** — lý do giữ ở đoạn "Không phải việc" dưới.

## CHƯA LÀM — phải làm trước khi coi là xong
1. **PAIR** — cần upstream công bố `model_vssp` + `MHTransformer_dis`. Đã quét 21 commit, không tồn
   tại. Xem `replay/pair_absence.py`.
2. **15 model G1/G2 `NOT_RUN`** (13 đang hoạt động + 2 deferred) — cần artifact parity; mỗi model cần GPU.
   Nguồn duy nhất của bảng: `CENSUS_20261111.vi.md` §1 (17 trừ encoder và limn); trích lại từ đó, không tự đếm.
3. **5 ca cần rerun** (bảng §6 `FINAL_REPORT.vi.md`): `clip4cir_rn50x4_noft`, `airknow` (3 dòng counted
   — n05 và n08 khác checkpoint, cộng một cặp mâu thuẫn trên `fiq_n05`), `clvc_net` (hai số counted mâu
   thuẫn, xem mục 10), `pair` (4 dòng), `habit` `e0dd04a4` (artifact trỏ interpreter chết). ENCODER **đã
   xong** nên **không** tính vào 5: `acc9b20e` là canonical; `5dee7425`/`6a528e87` superseded, lý do từng
   id ghi ở §6 `FINAL_REPORT.vi.md`. Số ca phải rerun còn **5**.
4. **LIMN 2 query mơ hồ** `7:3038`, `7:3790` — provenance upstream không cho phép chốt category. Cần
   metadata category từ upstream.
5. **Derived row chưa có marker `DERIVED` trên UI** — dữ liệu đúng ở `derived_from`, bề mặt web chưa
   hiện. Đo lại tại `0691f3e`: `grep -rn 'derived_from' workbench/frontend/*.ts*` → **0** kết quả, và
   bundle đang phục vụ (`index-BLH2yome.js`) cũng có **0** lần chuỗi `derived_from`. Vẫn là việc mở.
6. **46/50 dòng dùng đơn vị suy luận** — payload sống `:8000/api/native-measurements` tại `0691f3e`
   cho `unit_inferred 46` / `unit_declared 4` / `unit_sibling_key 2` / `unit_not_declared 46` trên 56
   row (50 row có metrics). Đã audit từng dòng, 0 khai báo bị bỏ sót; muốn hết suy luận phải sửa
   producer của từng artifact.
7. **Quét chéo 9 báo cáo CHƯA hoàn thành** — worker bị kill, file `raw/verify/42-*` không tồn tại. Đã
   sửa tay 6 chỗ (liệt kê trong `FINAL_REPORT.vi.md` §6.1), phần còn lại chưa kiểm; `FINAL_REPORT.vi.md`
   §6.1 đã ghi rõ giới hạn này để người đọc không giả định nhất quán.
8. **R16(b) — composition adapter của CLVC/TG-CIR luôn từ chối** vì thiếu interpreter stack pin. Cần
   pin stack rồi chạy lại composition cho hai model đó. (R16 vế (a) đã đóng ở `358c3fc`: bảy run GPU
   `airknow_run3cat_gpu_run{1..4}.json`, `tgcir_run3cat_gpu.json`, `clvc_run3cat_gpu.json`,
   `clvc_run3cat_gpu_run2.json` nay được `native_measurements.py` phục vụ qua
   `_review_run_payload`/`_review_run_rows`, `source_kind: review_run`.)
9. **R16(c) — chưa xác định ai gửi SIGTERM** giết stack. Đã xác định `serve_workbench.py:152-166` là
   đường nhận, đã loại trừ `jobs.py`, test/script trong repo và session reaping; người gửi vẫn chưa biết.
10. **CLVC có HAI số counted mâu thuẫn — phải chốt hoặc rerun.** Cùng `model_id: clvc_net`, cùng
   `protocol_id: fashioniq_val_split`, cả hai `counted: true`: `29.970746517160162`
   (`checkpoint_selection/20261106T002617+0700/measurements/clvc_run1.json`) và `30.02079703540087`
   (`workbench/docs/reviews/20261111-master-audit/raw/clvc_run3cat_gpu.json`). API ghi `conflict_with`
   cho **cả hai** và **không** chọn theo độ mới (`native_measurements.py`, `_mark_duplicates` nhánh
   `# 3d`). Cần: (i) chủ dự án chốt run canonical, hoặc (ii) rerun CLVC và so **từng qid** với cả hai.
   Cho tới đó **không** được trích "CLVC = 29.97" hay "CLVC = 30.02" như số đã chốt — xem
   `CENSUS_20261111.vi.md` §1.2. (Air-Know có cùng dạng hai row counted, nhưng đã có `variance_group`
   n=4 / sd `0.26728186433775697` nên mức bất định **đã đo**; CLVC thì chưa.)
11. **uvicorn `:8765` cũ vẫn sống** — thiếu route mới (`/api/runs/parity-gate` → 404 so với 200 ở
   `:8000`; đo lại tại `0691f3e`: đúng 404 vs 200). Nhận diện bằng cổng và hành vi route, không bằng
   PID. Cần kill tiến trình đó; PID đổi sau restart nên không ghim số.

**Đã sửa, không còn là việc:** nhãn "Số đo native không ghi protocol" ở `ModelInventory.tsx` **là**
defect cùng họ R06/R08. Payload sống `:8000/api/native-measurements` cho `clvc_net` / `tgcir`: mọi row
`protocol_id: null` là roll-up `counted: false` (`all_measured_summary.json`,
`selection_decisions.json`), nên `nativeUnattributedFor()` cũ (lọc `statesNumber ∧ !protocol_id`) đếm 2
row và báo "2 số đo native nhưng file không ghi protocol" — trong khi model có row counted ở
`fashioniq_val_split`. Cùng lỗi ở `airknow` @ `fashioniq_full_gallery_ref_excluded`: 6 row, chỉ 3
`counted: true`, 3 row còn lại là repeat `duplicate_of` (`run3cat_gpu_run{1..4}`), nên "6 số đo native"
là sai. Nay `scope.ts` có predicate `counts` (`statesNumber ∧ counted !== false`) dùng trong cả
`nativeRowsFor` / `nativeUnattributedFor` / `nativeMeasuredIds`; row `counted: false` **vẫn hiện** trong
bảng native với nhãn `duplicate_of`, chỉ không được tính là bằng chứng cho câu hỏi phạm vi.

## Trạng thái chốt
- Revision được sweep: `0691f3e88a03770aec0395843509da8f91a5d7ee` (= `TESTED_SOURCE_SHA`; **chính là**
  revision của sweep, không phải một revision khác được trích dẫn, và là revision đầu tiên chứa đủ toàn
  bộ production đang có trên đĩa: `0691f3e` đổi `workbench/frontend/scope.ts` — predicate `counts()` —
  cộng `workbench/frontend/tests/inventory-cross-scope.test.cjs`, và `workbench/frontend/tests` nằm
  trong cây production mà guard release so sánh). Các lượt `7babfd9` (2931/18/47 = 2996),
  `c755673` (2914/18/47 = 2979) và `cd980dc` (2901/18/47 = 2966) **bị thay thế**.
- Sweep: `2931 passed / 18 failed / 47 skipped`, **2996 collected**, accounting gap **0**, `^ERROR ` 0,
  `Errno 28` 0, NEW=0 / DROPPED=0 / COMMON=18 so với baseline `20261110-e2e-repair/full_suite_68ad75e.txt`.
  Cây đứng yên tại hai mốc 17:03:31 và 17:28:35 (+0700) và tại lúc kết thúc. Bản ghi: `raw/verify/73-sweep-close.md`.
- Guard: `workbench/tests/test_handoff_commands.py` → **8 passed** tại revision này, gồm
  `test_no_release_document_points_at_a_superseded_source_sha` (pin này là thứ làm nó xanh; nó đỏ đúng
  theo thiết kế ở mọi revision trước khi tài liệu được pin).
- Phạm vi frontend của sweep, nói đúng: **sáu** file test python dính frontend
  (`test_web_scope_and_labels`, `test_frontend_catalog`, `test_frontend_requests`,
  `test_frontend_staleness`, `test_frontend_layout_provenance`, `test_pc_launcher`) → **162 passed**;
  nhưng nhánh predicate counted-only (`counts()`, `row.counted === false`) chỉ được phủ bởi node suite
  `workbench/frontend/tests/inventory-cross-scope.test.cjs`, mà `pytest workbench/tests` **KHÔNG** chạy.
  Vậy độ phủ frontend của sweep = sáu test python **cộng** node suite đó; không đọc là "sweep đã chạy
  test frontend".
- Production diff `0691f3e..HEAD` ngoài `workbench/docs` và `workbench/tests`: **rỗng** (đo trước khi pin).
  Commit sau mốc chỉ được sửa tài liệu; guard so từng byte cây sản xuất sẽ đỏ nếu không còn đúng.
- Bất biến: `git status --porcelain` = `?? CLEANUP_REPORT.md.save`
- Bundle phục vụ: `index-BLH2yome.js` sha256
  `702ed2e93a1bf86ba2094fe62791d4272dcdbf36e8caf7f757f08cf96805d681` — **fact đo được**, đọc lại bằng
  `curl -s http://127.0.0.1:5173/ | grep -o 'assets/index-[^"]*\.js'` rồi
  `sha256sum workbench/frontend/dist/assets/index-BLH2yome.js`; `dist/assets/` lúc đo chỉ có
  `index-DecQnta9.css` + `index-BLH2yome.js`, và bytes phục vụ `cmp`-identical với `dist/`. Mỗi lần build
  lại phải đọc lại đúng hai lệnh đó, không chép số này. `index-DyVaNikG.js` / `81d61f65…0981` là bundle
  **superseded**; `index-DeTvxLlD.js` / `1503e891…8155` và các tên cũ hơn cũng vậy.
- Web: http://127.0.0.1:5173/ · API: http://127.0.0.1:8000 (poll `/api/benchmark-runs`, **không** `/api/models`)

## Dọn dẹp đã làm
`/tmp`: 50G → 5.1G (**~45G** giải phóng). Đã xoá: 3 git worktree qua `git worktree remove` (`/tmp/cir1716`, `/tmp/cirb0`, `/tmp/wtc`), `cirwarm-{airknow,tgcir,clvc}` (~27G cache tái tạo được), `torchprobe-py`, `cudaenv`, `r6-scratch-*`, `cir-upstream-audit-*`, `gr_a`, `r12probe`, `asrenv`, và toàn bộ basetemp/probe scratch.
