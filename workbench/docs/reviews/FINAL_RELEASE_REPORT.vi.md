# FINAL_RELEASE_REPORT.vi.md

Báo cáo phát hành cuối cùng — CIR Failure Analysis Workbench
Nhánh: `research/cir-failure-workbench` · Commit phát hành: `e4868bb`
(bao gồm base `e9cafaf` — vòng fix BUG 1-8 — và toàn bộ vòng ổn định hóa trước đó)

---

## 1. Tóm tắt phán quyết

**Verdict: GPU_VALIDATION_PENDING (release-ready phần mềm)**

- Hai vòng review đối kháng độc lập (vòng 1: A/B/C/D; vòng 2: A2/B2/C2/D2) đã chạy trên mã
  nguồn cuối; mọi phát hiện P0/P1 CONFIRMED đều đã sửa, có test hồi quy, và được verify lại.
- Suite backend đầy đủ (loại 3 file e2e nặng): **916 passed** chạy serial, 0 failed.
- `release_verify.sh --offline`: **không có tiêu chí chưa đạt** (5 PENDING do không có live
  stack/GPU); `mutation_checks.sh`: **cả 6 mutant bị phát hiện** (M1–M5).
- Docker image build lại và chạy thật với user không phải root: backend 200 (16 model),
  frontend 200, thư mục ghi (config/node_modules/mamba) đều ghi được.
- **Chưa xác thực được GPU** trên máy phát hành (không phải Blackwell, thiếu
  `nvidia-container-toolkit`): mọi kết luận GPU vẫn là PENDING, fail-closed, không bị làm giả.

## 2. Vòng review 1 — P0/P1 đã sửa (chi tiết trong `ROOT_CAUSE_MATRIX.md`)

| # | Phát hiện | Bản chất | Xử lý |
|---|-----------|----------|-------|
| P0-A | Planner dựng lại package list bằng tay, bỏ qua runtime đã adapt | P0 runtime | `_runtime_model` dùng contract resolve chuẩn (`resolved_runtime_environment`) |
| P0-B | Env cho evaluator child thiếu `WORKBENCH_RUNTIME_ADAPTATION` | P0 runtime | Cho phép tên này trong allowlist (kèm deny-prefix cho secret) |
| P1-C | GPU doctor probe nhầm compose config | P1 deploy | Probe container ĐANG chạy thật (`docker exec`) |
| P1-D | `run.sh` nạp `.env` không nhất quán với compose | P1 deploy | Nạp `.env` theo đúng thứ tự compose, guard thiếu biến |
| P1-E | Verifier chuyển PENDING thành PASS | P1 a | Fail-closed: PENDING không bao giờ thành PASS |
| P1-F | Verifier restart container đang chạy | P1 a | Không restart; probe container đang sống |

## 3. Vòng review 2 — phát hiện + xử lý (chi tiết trong `ROOT_CAUSE_MATRIX.md` § wave 2)

### A2 — Deploy (BLOCKED → đã xử lý hết)
- **P1 `user: 1000:1000` làm chết deployment**: container không root không ghi được
  `node_modules/.vite-temp`, micromamba root prefix, cache dưới `HOME=/`. Sửa trong
  `Dockerfile`: tạo `appuser` (uid 1000) có `HOME=/home/appuser`,
  `MAMBA_ROOT_PREFIX=/artifacts/mamba`, chown `node_modules`; compose đặt
  `XDG_CONFIG_HOME=/artifacts/config`. Verify live: boot backend + frontend, 16 model, mọi
  thư mục ghi được.
- **P2**: vòng khôi phục mutation restore sai đường dẫn (decode `_`→`/` + `mkdir -p`);
  `--filter name=workbench` đụng container lạ → `^cir-workbench$`; J1 hardcode `16` → so
  với `models.yaml`; J4(b) grep biểu thức chọn bundle thật (thắng mutant M2);
  `acceptance.sh` đổi 400 → 503; job khôi phục lúc startup giờ hiện trong `/api/jobs`.

### B2 — Runtime (VARIANT → đã xử lý)
- Xác nhận `cuda: '11.7'` có căn cứ (wheel torch 2.0.1+cu117, đọc trực tiếp `version.py`),
  gate `_requires_gpu` chặn csmcir trên host CPU đúng, YAML quote giữ 1 declaration.
- **B2-3 (P1)**: `_runtime_model` trùng entry không version, mất version floor → dùng
  contract resolve; `_required_packages` key theo tên distribution chuẩn qua
  `packaging.Requirement` (constraint hỏng → `VERSION_CONSTRAINT_UNVERIFIED`, không
  `DEPENDENCY_MISSING`).
- **B2-2/4 (P2)**: group dict + readiness row mang `runtime_adaptation`
  (`environment_adapted` không còn luôn False).

### C2 — Khoa học (BLOCKED → đã xử lý hết)
- **F-01 (P1)** evaluator child đọc được ledger key qua `HOME` → child không bao giờ nhận
  `HOME`/`XDG_CONFIG_HOME` (cache đi qua `TORCH_HOME`/`HF_HOME`/…). Repro: child bị
  `LedgerKeyError`.
- **F-02 (P1)** corroboration key theo run_id → key theo digest file + identity khớp; file
  copy không thừa hưởng corroboration, run_id trùng chỉ serve 1 lần.
- **F-03 (P1)** kiểm tra `return_code == 0` và `INVOCATION_COMPLETED` — run fail không
  chứng minh report nào cả.
- **F-04 (P1)** container `HOME=/` không ghi được → key nằm ở `/artifacts/config` (bind
  mount, sống qua rebuild).
- **F-05 (P2)** reference chỉ in "Avg." làm run đủ coverage bị hạ → hạ chỉ khi coverage của
  run thiếu; đủ coverage → pass thật (test cả 2 chiều).
- **F-06 (P2)** `checkpoint_mapping_status` của reference vào `as_evidence` và trần hóa
  verdict: khác `VERIFIED_METADATA`/`NO_CHECKPOINT` → `PASS_METRICS_ONLY_MAPPING_UNVERIFIED`
  (conesep khớp số tuyệt đối không còn được chứng nhận full pass).
- **F-07 (P2)** guard reports-dir so tên thư mục đã `resolve()` (chặn symlink alias).
- **F-08 (P3)** lý do `NOT_RUN` cho synthetic đổi thành "pipeline self-test".

### D2 — API/Frontend/Jobs (BLOCKED → đã xử lý hết)
- **P0**: corroboration đòi status không tồn tại (`EXECUTION_OK`) → dùng giá trị thật
  `INVOCATION_COMPLETED`; report thật được serve lại.
- **P1**: job `queued` không được persist (thư mục log chỉ tạo lúc dispatch) và cancel
  queued không persist → `_persist` tự tạo thư mục; `cancel` persist; worker
  `mkdir exist_ok=True`. Test hồi quy cho cả ba.
- **P2**: TOCTOU duplicate-job → check nguyên tử trong `JOBS.enqueue` (409); `dist/` build
  lại; test digest tautology thay bằng test contract thật của P0-A.
- **P3**: recovery root dời vào FastAPI startup hook (không còn key theo import-time root).

## 4. Acceptance đã chạy thật

| Kiểm tra | Kết quả |
|---|---|
| Suite backend đầy đủ (serial, không e2e nặng) | 916 passed / 0 failed |
| `release_verify.sh --offline` | Không tiêu chí chưa đạt, verdict `GPU_VALIDATION_PENDING` |
| `mutation_checks.sh` | 6/6 mutant DETECTED (M1, M2, M3, M4, M4, M5) |
| Frontend build (`tsc -b && vite build`) | OK, `dist` mới hơn `App.tsx` |
| Docker image build + boot (uid 1000) | Backend 200 (16 model), frontend 200, first-boot 503 trung thực |
| Docker secret scan | Không `.env`/key vào context build (control âm+dương) |
| Smoke ledger corroboration | Forge bị chặn (`LedgerKeyError`), run fail không corroborate, run hoàn thành được serve |

## 5. Khoảng trống còn biết (đã ghi nhận, không chặn phát hành)

- **GPU validation**: cần máy có RTX 5070 Ti (sm_120) + `nvidia-container-toolkit` để chuyển
  `GPU_VALIDATION_PENDING` thành verified. Không có bằng chứng GPU nào bị làm giả.
- M2 mutant vẫn queue job dress-only vào stub server (không tới server thật) — dò phát hiện
  bằng output pilot, đủ cho mục đích mutation check.
- C-16/C-18 (không inspect được dữ liệu upstream) remain UNKNOWN.
- D-7 (UI polish) được chấp nhận hoãn.

## 6. Artifacts review

- Vòng 1: `workbench/docs/reviews/final_hardening/{A,B,C,D}.md` + `ROOT_CAUSE_MATRIX.md`
- Vòng 2: `workbench/docs/reviews/final_hardening/{A2,B2,C2,D2}.md`
- Phản ứng xử lý cả hai vòng: `ROOT_CAUSE_MATRIX.md` (mục wave 2 ở cuối)
- Test hồi quy mới: `workbench/tests/test_final_hardening_regressions.py` (19 test)
