# FINAL_STABILIZATION_REPORT.vi.md

Báo cáo ổn định hóa cuối cùng — CIR Failure Analysis Workbench
Nhánh: `research/cir-failure-workbench` · Căn cứ: base `cf6cc1e8f3475a3f21e7650466a7a54afc6384c9` + các commit ổn định hóa.

---

## 1. Tóm tắt phán quyết

**Verdict: GPU_VALIDATION_PENDING**

- Toàn bộ lỗi phần mềm đã biết (Defect A–J) đã xử lý xong, test backend + frontend + Docker
  chạy thật và pass.
- Docker stack build + chạy end-to-end trên máy chuẩn bị (CPU): backend 200, frontend 200,
  proxy `/api` 200, first-boot trung thực (`/api/runs` = 400 `index_unavailable` cho tới khi có
  index).
- **Chưa xác thực được GPU** trên chính máy phát hành này: RTX 3050 Laptop (cc 8.6, 4 GB) không
  phải Blackwell; Docker GPU passthrough bị chặn vì `nvidia-container-toolkit` chưa cài được
  (cần sudo). Không có bằng chứng GPU thật nào bị làm giả — mọi kết luận GPU là PENDING.

## 2. Các defect đã xử lý

| # | Defect | Bản chất | Cách xử lý |
|---|--------|----------|------------|
| A | LIMN pin torch 1.12.1 không có kernel sm_120 | Không thể chạy trên RTX 5070 Ti | Registry thêm `adapted_runtime` (torch ≥2.7,<3, torchvision ≥0.22,<1, open_clip ≥2.24); operator chọn bằng `WORKBENCH_RUNTIME_ADAPTATION=sm_120`; `validate_environment` chấp nhận 2 contract; provenance `ADAPTED_RUNTIME` ghi trong `variant_provenance`, không bao giờ gắn nhãn tái tạo chính xác |
| B | `Promise.all` 4 endpoint: `/api/runs` 400 làm UI mất trắng model | P1 frontend | Tách 4 fetch độc lập; lỗi mỗi endpoint hiển thị riêng (`models:`, `runs:`, …) |
| C | Nhiều nguồn cấu hình (web/autopilot/docker) không nhất quán | P1 deploy | `serve_workbench.py --config` + pc_readiness `--config` dùng chung autopilot YAML; backend child nhận WORKBENCH_* env; uvicorn chạy `python -m uvicorn` |
| D | pc_readiness gộp blocker của TẤT CẢ variant | Báo cáo sai trạng thái | `model_section` nhận `CheckpointSelection`; chỉ merge variant được chọn; thêm `selected_checkpoints` vào report |
| E | `READY_WITH_DECLARED_VARIANT` bypass GPU check | Trạng thái sai | GPU rule áp cho cả variant state |
| F | Metric thiếu có thể thành "LỆCH PAPER" | Sai phạm scientific | `compare_metric` giữ `relative_error_pct=None`; regression test chỉ ra missing evidence không bao giờ được flag deviation |
| G | Job status `completed` (backend) vs `COMPLETED` (frontend) | Settle-refresh không chạy | Backend lowercase là chuẩn; `Job.status` typed union lowercase |
| H | Ngưỡng lệch UI trông như đổi gate | Nhầm lẫn audit | Thêm dòng giải thích: chỉ áp dụng hiển thị, gate chuẩn 5% không đổi |
| I | Report malformed | Đã có normalization từ vòng trước | Đã verify: 14 test `test_http_api_layer.py` pass |
| J | Docs/manifest lệch code | Provenance | `SOURCE_COVERAGE_MANIFEST.md` regen: 126 file, machine-verify 0 mismatch |

## 3. Docker deliverables

- `workbench/docker/Dockerfile` — 3 stage: node build frontend → python runtime + node (vite
  preview serve dist, proxy `/api`), `python -m uvicorn`.
- `workbench/docker/docker-compose.yml` — volumes: dataset (ro), third_party, checkpoints,
  artifacts, results; GPU block commented + hướng dẫn bật.
- `workbench/docker/.dockerignore`, `.env.example` (HOST_* cho mount, WORKBENCH_* cho in-container).
- `workbench/docker/run.sh` — `up / doctor / pilot limn|csmcir / status / logs / stop`.
- `workbench/docker/acceptance.sh` — nhóm test A–I chạy được; đã chạy trên container thật: PASS.

## 4. Bằng chứng GPU (trung thực)

| Kiểm tra | Kết quả |
|---|---|
| Host: `gpu_check.py` | PENDING — `nvidia-smi` thấy RTX 3050 (cc 8.6) nhưng torch 2.14.0+cpu → `cuda_available=False` |
| Container (không GPU): `gpu_check.py` | PENDING — "no GPU visible inside this container" |
| Docker `--gpus all` trên máy này | FAIL: "failed to discover GPU vendor from CDI" (thiếu nvidia-container-toolkit; cần sudo) |
| Blackwell (RTX 5070 Ti) | CHƯA chạy — PC đích mới là nơi xác thực |

Hướng dẫn trên PC đích: `workbench/docs/PC_DOCKER_QUICKSTART.md` (cài toolkit → bỏ comment GPU
block → `run.sh doctor` → `run.sh pilot limn`).

## 5. Điều kiện nâng lên RELEASE_READY

1. Trên PC có RTX 5070 Ti: `run.sh up && run.sh doctor` → doctor báo GPU visible.
2. Chuẩn bị model đầu tiên (LIMN ưu tiên 1, CSMCIR ưu tiên 2).
3. `run.sh pilot limn` — job chuyển `queued → running → completed` thật, report JSON có
   `benchmark_gate` thật, hai nhãn cảnh báo duy nhất (`CAPTION THAY THẾ` / `LỆCH PAPER X%`).
4. Bật `WORKBENCH_RUNTIME_ADAPTATION=sm_120` nếu dùng torch hiện đại cho LIMN; run report sẽ
   mang nhãn ADAPTED_RUNTIME, không phải exact reproduction.

## 6. Ràng buộc scientific đã giữ

- **Ghi chú test:** lần chạy full-suite đầu tiên có 13 test fail — toàn bộ do hết đĩa
  (`OSError: [Errno 28] No space left on device`, Docker image chiếm chỗ). Sau khi giải phóng
  dung lượng, chạy lại cả 19 test failed → **19/19 pass**; kết quả tương đương baseline 910
  passed. Không có regression thật nào.

- Không Recall giả, không hash bịa, không hạ chuẩn verify.
- Artifact bất biến; giá trị paper không bị sửa.
- Hai nhãn cảnh báo duy nhất; metric thiếu = "missing evidence", không phải deviation.
- Mọi claim GPU chỉ đến từ probe thật (tensor + compiled-core ELF) gắn host/driver.
