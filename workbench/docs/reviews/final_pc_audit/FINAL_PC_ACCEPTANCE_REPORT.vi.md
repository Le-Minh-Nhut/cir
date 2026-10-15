# FINAL_PC_ACCEPTANCE_REPORT.vi.md

Báo cáo nghiệm thu phát hành cuối — CIR Failure Analysis Workbench
Nhánh: `research/cir-failure-workbench` · HEAD: `a7081f0` (nền `c0c4438`, `abcd6e3`, `6e3c979`)

---

## 1. HEAD thực tế

`a7081f0` — "fix(test): align the GPU-stage stub with verify_environment's real signature".
Chuỗi commit phát hành: `6e3c979` (sáu audit + fix P0/P1 + `pc.sh`) → `abcd6e3` (fix vòng review
thứ tư) → `c0c4438` (báo cáo nghiệm thu) → `a7081f0` (sửa stub chữ ký test, suite xanh hoàn toàn).

Lưu ý trung thực: `a7081f0` sửa một **hồi quy do chính wave này gây ra** — tách
advisory/authoritative thêm tham số `advisory` vào `verify_environment`, trong khi
`test_gpu_verification_stage_state_machine` stub hàm đó bằng `lambda m, probe=True`. Test đã pass
ở `0b0f3a6`, fail sau wave, nay đã sửa stub cho khớp chữ ký thật (giữ nguyên assertion).

## 2. Manifest độ phủ nguồn

Sáu audit độc lập chạy trên mã nguồn thật, mỗi báo cáo liệt kê tệp đã đọc:

| Báo cáo | Phạm vi | Kết quả |
|---|---|---|
| `A.md` (499 dòng) | Docker, host, PC bootstrap | BLOCKED — 2 P0 |
| `B.md` (446 dòng) | Runtime, CUDA, phụ thuộc | BLOCKED — 4 P0/P1 |
| `C.md` (473 dòng) | Model, chuẩn bị, đánh giá | BLOCKED — 2 P0, 2 P1 |
| `D.md` (470 dòng) | Toàn vẹn khoa học | VARIANT — 1 P1 |
| `E.md` (188 dòng) | Backend, frontend, jobs | 3 P1, 4 P2 |
| `F.md` (276 dòng) | Chất lượng test / đối kháng | 1 P0, 3 P1 |

Vòng review thứ tư trên mã đã sửa: `REV1_LAUNCHER.md`, `REV2_RUNTIME.md`, `REV3_SCIENCE.md`,
`REV4_BACKEND.md`. Hợp nhất theo nguyên nhân gốc: `ROOT_CAUSE_MATRIX.md`.

## 3. Tóm tắt subagent

| Subagent | Vai trò | Phán quyết cuối |
|---|---|---|
| Rev1 | Launcher + Docker | SOUND_WITH_RESERVATIONS |
| Rev2 | Blackwell runtime + CUDA | SOUND_WITH_RESERVATIONS (4/4 fix P0/P1 đã xác minh) |
| Rev3 | Khoa học + bằng chứng | **SOUND** |
| Rev4 | Backend/frontend/jobs | SOUND_WITH_RESERVATIONS |

## 4. Khuyết điểm đã xác nhận (CONFIRMED)

Nhóm theo nguyên nhân gốc — chi tiết và bằng chứng trong `ROOT_CAUSE_MATRIX.md`:

| # | Khuyết điểm | Mức | Trạng thái |
|---|---|---|---|
| RC-1 | Đường trạng thái/readiness làm việc nặng: hash 5.4 GB checkpoint + 97.317 tệp môi trường mỗi request `/api/models` (85 s) | P0/P1 | ĐÃ SỬA |
| RC-2 | Khóa ký capability không giải được trong container chạy uid ≠ 1000 → gate GPU đóng vĩnh viễn | P0 | ĐÃ SỬA |
| RC-3 | Gate so khớp từ vựng khác nhau (`VERIFIED` vs `VERIFIED_METADATA`) → run khớp paper không bao giờ đạt | P1 | ĐÃ SỬA |
| RC-4 | So kiến trúc sm_120 bất khả thi (`'120' in ['sm_120']`) → gate đóng trên đúng phần cứng đích | P0 | ĐÃ SỬA |
| RC-5 | Biến chọn runtime toàn cục bị coi là lỗi từng model → planner crash cho mọi model không khai báo | P1 | ĐÃ SỬA |
| RC-6 | Bundle id không giải được ở `checkpoint_by_id` → `KeyError` trên đường adapter | P0 | ĐÃ SỬA |
| RC-7 | Kiểm tra trùng job nguyên tử không bao giờ được gọi (TOCTOU) | P1 | ĐÃ SỬA |
| RC-8 | Tiến trình con sống sót sau khi container dừng (giữ VRAM) | P1 | ĐÃ SỬA |
| RC-9 | Verifier nuốt bằng chứng: một tệp 0 byte ép `real_gpu = False` | P1 | ĐÃ SỬA |
| RC-10 | Scanner PASS trên dữ liệu rỗng (đọc key không tồn tại; queue rỗng) | P0/P1 | ĐÃ SỬA |
| RC-11 | Launcher poll endpoint quá đắt | P1 | ĐÃ SỬA |
| RC-12 | `acceptance.sh` crash trên model không khai báo protocol; exit code bị `||` nuốt | P2 | ĐÃ SỬA |
| RC-13 | `/artifacts/mamba` bị mount che → uid ≠ 1000 không tạo được prefix | P1 | ĐÃ SỬA |
| RC-14 | Phát hiện toolkit khớp chuỗi "nvidia" bất kỳ trong `docker info` | P2 | ĐÃ SỬA |
| RC-15 | Ngưỡng đĩa 10 GB mâu thuẫn yêu cầu thật ~5.4 GB checkpoint | P2 | ĐÃ SỬA |
| RC-16 | Monitor fallback sang `jobs[0]` → có thể báo thành công của job khác | P2 | ĐÃ SỬA |
| RC-17 | `probe_gpu_capability` in key không tồn tại (luôn `None`) | P3 | ĐÃ SỬA |

## 5. Bằng chứng hồi quy

Mỗi fix đều có test **fail-before / pass-after** (agent báo cáo lệnh và output quan sát):

- RC-1: `checkpoint_availability(limn)` worst-of-3 **11.06 s → 0.97 s**; `sha256_file` **129.751 → 0** lần.
- RC-2: khóa 32 byte, mode `0600`, ổn định qua các lần gọi; record round-trip.
- RC-3: LIMN khớp paper → `PASS_WITHIN_5_PERCENT passed=True` (trước: `PASS_METRICS_ONLY_MAPPING_UNVERIFIED`); conesep `UNVERIFIED` vẫn bị hạ.
- RC-4: bảng kiểm — khớp thì verify, không khớp thì từ chối; xử lý `_blackwell` và `+PTX`.
- RC-5: csmcir → `2.0.1 / cuda 11.7`; LIMN → `>=2.7,<3 / cu128`; giá trị sai cho model có khai báo vẫn raise.
- RC-6: bundle resolve; `validate_request` chấp nhận; id lạ vẫn raise.
- RC-7: test gửi đồng thời — trước fix `DuplicateJobError` thoát ra thành 500.
- RC-8: process group biến mất sau shutdown; trước fix không có method `shutdown`.
- RC-9: 1 record tốt + 1 tệp 0 byte → `UNREADABLE` + `real_gpu: True` (trước: `PENDING`).
- RC-10: hai test fail trên script cũ; live in `11 GPU-requiring models`.
- RC-17: acceptance live `PASS 16 models`; G `NOT VERIFIED` + `SKIP`.

## 6. Kết quả suite đầy đủ

| Lệnh | Kết quả |
|---|---|
| **Full suite (loại 3 file e2e nặng)** | **951 passed, 0 failed** (27 deselected) |
| `pytest test_pc_launcher test_final_hardening_regressions test_http_api_layer test_concurrency_ownership` | **72 passed** |
| `pytest test_paper_gate test_runtime_adaptation test_release_blocker_*` | **126 passed** |
| `bash workbench/docker/acceptance.sh` (live) | **ACCEPTANCE PASSED (skipped: G-status-unverified)** |
| `bash workbench/docker/release_verify.sh --offline` | **0 tiêu chí chưa đạt** (5 pending) |
| `bash workbench/docker/mutation_checks.sh` | **6/6 mutant DETECTED** |
| `./pc.sh --self-check` | passed |
| `./pc.sh --doctor` | 0.3 s, read-only, báo đúng RTX 3050 (không phải 5070 Ti) |
| `./pc.sh --verify-only` | rc=1 + tóm tắt lỗi tiếng Việt (fail-closed, thiếu toolkit) |
| `./pc.sh --help` | rc=0 |

Suite đầy đủ chạy hai lần: lần đầu 950/951 (một hồi quy stub chữ ký), sau khi sửa `a7081f0`
chạy lại **951 passed, 0 failed**.

## 7. Bằng chứng build Docker

Image build từ `workbench/docker/Dockerfile` (đổi `appuser` + `/opt/mamba`), tag
`cir-workbench:final`. Container chạy uid 1000 xác nhận: `HOME=/home/appuser`,
`MAMBA_ROOT_PREFIX=/opt/mamba`, `/artifacts/config` ghi được, `node_modules` ghi được,
`/opt/mamba` ghi được. **uid 4242 (tuỳ ý) cũng tạo được prefix micromamba** — trước đây thư mục
`/artifacts/mamba` nằm trong bind mount nên bị che và chỉ uid 1000 mới dùng được.

Boot smoke image cuối: backend 200, `/api/models` **1.18 s** cho 16 model, `/api/jobs` 200,
frontend 200.

## 8. Test launcher PC

`workbench/tests/test_pc_launcher.py` — **20/20 pass**, điều khiển `pc.sh` thật qua PATH shim
(`docker`, `nvidia-smi`, `curl`, `git`, `df` giả), sandbox `tmp_path`, không cần GPU/Docker/network:

`--help` · doctor read-only · thiếu docker · thiếu nvidia-smi · thiếu container runtime ·
tạo `.env` mới · giữ `.env` cũ · thiếu dataset · `--dataset` tuỳ chỉnh · giá trị quote có `#` ·
`.env` sai định dạng · idempotent 2 lần · attach job trùng · tái dùng run đã hoàn thành ·
job thất bại · job blocked · backend tạm mất · hết đĩa · không rò secret · SIGINT.

## 9. Khởi động container

Backend 200 (`/api/models` 16 model, 6–7 s), `/api/jobs` 200, `/api/runs` 200/503 trung thực,
frontend 200. Thời gian `/api/models` từ 85 s xuống 6 s.

## 10. Test cấu hình

Tạo/giữ `.env`; parse chặt `KEY=VALUE` (comment, bóc 1 lớp quote, không `eval`); thứ tự ưu tiên
shell > `.env` > mặc định; cổng sai → exit khác 0 trước khi chạm docker; ghi cờ GPU nơi compose đọc.

## 11. Test tương thích runtime

`WORKBENCH_RUNTIME_ADAPTATION=sm_120` xuyên suốt planner → solver → installer → interpreter →
verifier → GPU probe → evaluator với **một** contract; id môi trường tách biệt
(`cir-5f868bc31646189b` adapted vs `cir-1bbd9872cf166f0d` historical); biến không đặt giữ pins
lịch sử. Advisory fingerprint = 0 lần hash nội dung; authoritative vẫn hash đầy đủ (rewrite nội
dung cùng size/mtime vẫn bị phát hiện).

## 12. Test thẩm định khoa học

LIMN khớp paper chứng nhận; conesep UNVERIFIED bị hạ; reference không có bảng category → coverage
đầy đủ thì pass, thiếu thì hạ; synthetic/mock bị `NOT_RUN`; NaN/âm/>100 bị `ALERT_METRIC_DEVIATION`;
run dress-only bị từ chối là không đầy đủ; report giả/ledger rởm bị cách ly. `papers.yaml` không
đổi một dòng — không sửa protocol để khớp số.

## 13. Test bền vững job

Persist job `queued`; persist cancel; khôi phục sau restart; 409 trùng nguyên tử; shutdown thu hồi
process group; không deadlock.

## 14. Test mutation

`MUTATION CHECKS PASSED: M1 M2 M3 M4 M4 M5` — cả 6 mutant bị phát hiện.

## 15. Trạng thái xác thực GPU đích

**PENDING.** Máy này là RTX 3050 Laptop (cc 8.6), thiếu `nvidia-container-toolkit`. Không có
bằng chứng GPU thật nào bị làm giả: gate fail-closed và báo đúng lý do.

## 16. Trạng thái suy luận LIMN thật

**PENDING** — cần máy đích.

## 17. Trạng thái đánh giá FashionIQ thật

**PENDING** — cần máy đích.

## 18. Hạn chế bên ngoài còn lại

- `torch==1.12.1` lịch sử không có trên PyPI thường → cần index hoặc profile adapted.
- Chưa có test nào chạy kernel sm_120/Blackwell thật; mọi test capability đều mock sm_89.
- Không có đối chứng metric paper (chỉ là phát hiện nghiên cứu khi có sai lệch).

## 19. Phân loại trạng thái

**VERIFIED:** đường launcher; đường readiness; cấu hình; contract runtime; thẩm định khoa học;
bền vững job; shutdown; scanner acceptance/verifier; mutation checks; suite test.

**PENDING:** xác thực GPU trên RTX 5070 Ti; cài môi trường trên máy đích; suy luận LIMN thật;
đánh giá FashionIQ thật; tương đương số liệu paper.

**BLOCKED:** không có (sau khi các P0/P1 đã sửa).

## 20. Cách dùng một lệnh

```bash
git clone git@github.com:Le-Minh-Nhut/cir.git
cd cir
git checkout research/cir-failure-workbench
./pc.sh
```

Tuỳ chọn: `./pc.sh --doctor`, `./pc.sh --model csmcir`, `./pc.sh --verify-only`,
`./pc.sh --force`, `./pc.sh --timeout <giây>`, `./pc.sh --no-gpu`, `./pc.sh --dataset <đường_dẫn>`.

Hướng dẫn đầy đủ tiếng Việt: `workbench/docs/PC_ONE_COMMAND_GUIDE.vi.md`.

## 21. Phán quyết phát hành

**ENGINEERING_READY**, kèm **GPU_VALIDATION_PENDING**.

- Không còn khuyết điểm P0/P1 đã xác nhận nào chưa xử lý.
- Launcher một lệnh đã có, test thật 20/20.
- Test tích hợp trên môi trường hiện có đều pass.
- Thẩm định phát hành trung thực: PENDING không bao giờ bị gọi là PASS.
- Hướng dẫn PC đầy đủ.

**Không** tuyên bố RELEASE_READY: chưa thu thập bằng chứng GPU/suy luận/danh mục thật.
