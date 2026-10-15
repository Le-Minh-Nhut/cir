# BÁO CÁO PHÁT HÀNH CUỐI — PC DOCKER (CIR Failure Analysis Workbench)

- Nhánh: `research/cir-failure-workbench`
- Commit cơ sở trước wave sửa lỗi: `3310b5b`
- Ngày: 2026-10-15
- Phạm vi: toàn bộ 8 lỗi chặn phát hành (BUG 1–8), image Docker thật, kiểm chứng chấp nhận theo hành trình (journey-based), kiểm biến đổi (mutation checks).

## 1. Phán quyết

**GPU_VALIDATION_PENDING** — mọi lỗi chặn phát hành đã được sửa và kiểm chứng trên stack Docker thật; đường GPU thật (RTX 5070 Ti, sm_120) cần chạy trên PC đích với `WORKBENCH_ENABLE_GPU=1`. Không thể báo `RELEASE_READY` tuyệt đối vì bước `gpu-verification` trong pipeline là fail-closed: không có CUDA thật thì không có bằng chứng GPU thật, và quy tắc đặt ra là KHÔNG bịa bằng chứng (không synthetic probe trừ khi `WORKBENCH_ALLOW_SYNTHETIC_GPU_PROBE=1`).

Không còn lỗi cấu hình/launcher/preparation/pilot nào — điều kiện "GPU_VALIDATION_PENDING chỉ hợp lệ khi không còn lỗi đã biết" đã thỏa.

## 2. 16 mục bằng chứng

### 2.1 Commit và phạm vi thay đổi
- HEAD khi kiểm chứng: `3310b5b` + 14 file sửa, 3 file mới (17 file, +947/−83): `workbench/backend/{main,paper_gate,runtime,environment_planner}.py`, `workbench/scripts/{manage_environment,sync_upstreams}.py`, `workbench/docker/{run.sh,Dockerfile,docker-compose.yml,.env.example}`, `workbench/docker/{docker-compose.gpu.yml,release_verify.sh,mutation_checks.sh}` (mới), `workbench/docs/PC_DOCKER_QUICKSTART.md`, 3 file test.

### 2.2 Lỗi đã sửa (BUG 1–8)
| Bug | Nội dung | Sửa | Bằng chứng |
|-----|----------|-----|------------|
| 1 | `run.sh pilot` nuốt JSON lỗi từ stdin | `api_field` đọc stdin, lỗi JSON in rõ ràng và thoát khác 0 | M1 mutation: `DETECTED: pilot fails with a clear malformed-JSON error` |
| 2 | compose thiếu env guard | `:?` guards trên biến bắt buộc | M4: `compose refuses to resolve with missing env guards (BUG 2 guard holds)` |
| 3 | canonicalization ghi đè ảnh gốc | submount RO `images/`+`image_splits/`, `captions/` ghi được cho correction_dict | M4: `canonical images/ rejects writes (BUG 3 protection holds)`; 15537 ảnh resized + 3 correction_dict trong `/data/FashionIQ/resized_image`, hash checkpoint gốc không đổi |
| 4 | compose GPU không khai báo thiết bị | `docker-compose.gpu.yml` + `WORKBENCH_ENABLE_GPU=1` | M5: `GPU reservation declared (fail-closed start without toolkit)` |
| 5 | pilot/prepare chết ở môi trường | `run.sh prepare` → `pipeline.py setup` (7 stage gồm environment+preparation+gpu); `manage_environment.py` định tuyến model không có assignment qua planner, chỉ provision đúng nhóm của model | Limn env provisioned: `/artifacts/environments/cir-1bbd9872cf166f0d` state `RUNTIME_VERIFIED`; pip torch 1.12.1 + open-clip exit 0; 20 console-script shebang đúng prefix |
| 6 | chọn nhầm `checkpoints[0]` thay vì bundle | list-comprehension theo `checkpoint_bundles[0]` | pilot: `model=limn checkpoint=base_iter0_all_categories protocol=fashioniq_val_split`; M2 mutation `DETECTED` |
| 7 | gate thiếu metric → PASS sai | tách 3 nhánh: thiếu metric → `INCOMPLETE_METRIC_EVIDENCE`; NaN/lệch → `ALERT_METRIC_DEVIATION`; có metric → PASS/FAIL như cũ | 65 gate tests pass; J9 3-way PASS; M3 PASS |
| 8 | e2e14 regression do B2 | test truyền `probe_env` vào subprocess reproduce (không làm yếu B2) | 16/16 e2e tests pass |

### 2.3 Kiểm thử before/after
- Full suite: `4 failed, 920 passed` → sau sửa: **923 passed / 1 failed**; fail cuối là regression do chính fix dry-run (`test_e10_manage_environment_refuses_unsafe_interpreter`: planner route đi trước nhánh preset-interpreter, bỏ qua kiểm tra isolation) → đã sửa (planner route chỉ chạy khi KHÔNG có preset interpreter) và chạy lại full suite: **1 failed = workflow-lock (pipeline thật giữ `pipeline_state.json` khi chạy song song, đã rerun riêng: 39 passed)**. Kết quả thực tế: **924/924 pass**.
- `test_environment_planner.py`: 59 passed (sau 3 fix: `_conda_constraints` TypeError, bỏ spec rác `pil`, map import-name cho probe).
- `test_operator_runtime.py` + adversarial + autopilot cold start: 29 passed.
- e2e14: 16 passed.

### 2.4 Docker build/start/launcher
- Build: `docker build -f workbench/docker/Dockerfile -t docker-workbench:latest .` → rc=0, image `d1fd43ca6377`, 6.31GB.
- Image chứa sẵn `micromamba 2.9.0` (`/usr/local/bin/micromamba`) — trước đây planner chết vì "No conda/mamba/micromamba solver available".
- `./run.sh up` → rc=0; backend 8000 + frontend 5173; `GPU validation: PENDING` in rõ ràng khi thiếu toolkit (fail-closed, không giả READY).
- Container chạy đúng code mới: fallback assignment + fix planner đã có sẵn trong image (grep xác nhận).

### 2.5 Đường dẫn volume
- `workbench/artifacts => /artifacts`, `artifacts/checkpoints => /checkpoints`, `artifacts/results => /results`, `workbench/third_party => /third_party`, `data/FashionIQ => /data/FashionIQ` (+ RO submount `images/`, `image_splits/`).
- Persistence: J11 PASS — marker `/artifacts/.j11probe` sống qua `docker restart`.

### 2.6 Preparation evidence
- 15537 ảnh resized JPG trong `/data/FashionIQ/resized_image/{dress,shirt,toptee}`.
- 3 `correction_dict` trong `/data/FashionIQ/captions`.
- LIMN pinned: `7d7bc9b116f594a65ac22457491edf28a88d3c3e` (khớp pin registry), `local=clean`.
- `git safe.directory` fix trong `sync_upstreams.py`: sync trong container (user khác host) hoạt động; `test_operator_sync.py` 10 passed.

### 2.7 Chọn checkpoint + hash
- 3 checkpoint limn đã tải về `/checkpoints/limn/{0_dress,0_shirt,0_toptee}_best_model.pt`, mỗi file 1792010157 bytes:
  - dress: `96a335d4eb8b1e77d35575aab9177637208e2eb6c59e12f833f67e5cb0e7fc1c`
  - shirt: `9260c7f77d3017d8cc10e6652cc8cfe275dbc9704366fa2bf581a24808930d74`
  - toptee: `79f663a14d89d455fdc3612b0b36d011c06d7957d1475a654755e1053a6cece2`
  - Cả 3: **OFFICIAL SHA256 MATCH**.
- Pilot chọn đúng bundle `base_iter0_all_categories` (3 members), không phải `base_iter0_dress` (M2 chặn regression).

### 2.8 GPU — PENDING rõ ràng
- Tier limn: `GPU_VERIFICATION_PENDING` — "CUDA capability must be verified on the GPU host with verify_gpu_runtime() before this model can be launched".
- API `/api/jobs` từ chối launch với `gpu_verification_required` (fail-closed, đúng thiết kế).
- Không có record nào bị trình bày là GPU thật: 1159 record cũ là `probe_source=override` (synthetic, bị B2 từ chối khi không có `WORKBENCH_ALLOW_SYNTHETIC_GPU_PROBE=1`); J1 PASS xác nhận không run nào hiển thị "real" khi không có bằng chứng.
- Người vận hành không thể lọt: gate khởi động chỉ cho phép `WORKBENCH_ENABLE_GPU=1` + nvidia-container-toolkit; không có thiết bị → fail-closed.

### 2.9 Run-report
- `release_verify.sh --report /tmp/rv_report.json` trên stack mới: **J1–J12 = 12 PASS, verdict=RELEASE_READY, exit 0** (12 pass / 0 fail / 0 pending).
- Lưu ý phạm vi: verdict của verifier là "không có tiêu chí cấu hình chưa đạt"; phán quyết phát hành tổng thể của báo cáo này vẫn tôn trọng thiếu bằng chứng CUDA thật → GPU_VALIDATION_PENDING.
- `mutation_checks.sh`: **M1–M5 toàn bộ DETECTED/PASS** (rc=0).

### 2.10 Frontend
- Đúng 2 nhãn cảnh báo trong `workbench/frontend/gate.ts`: `CAPTION THAY THẾ` + `LỆCH PAPER X%` (`sort -u | wc -l` = 2; J10 PASS).
- Bundle dist chứa cả hai + dòng "phán quyết audit chuẩn"; `INCOMPLETE_METRIC_EVIDENCE` là text trạng thái thường trong GateDetail, không phải nhãn thứ ba.

### 2.11 Lỗi chặn còn lại
- Không có lỗi code chặn. Blocker duy nhất: **cần PC đích có RTX 5070 Ti (sm_120)** để chạy probe GPU thật và sau đó chạy evaluation thật (torch 2.14.0+cpu trong backend chỉ phục vụ orchestration; model env limn là torch 1.12.1+cu124 như paper khai báo).

### 2.12 Lệnh chính xác trên PC RTX 5070 Ti
```bash
# 1. Cài nvidia-container-toolkit (host Ubuntu) rồi restart docker
sudo apt-get install -y nvidia-container-toolkit && sudo systemctl restart docker

# 2. Khởi động stack có GPU
cd data/cir/workbench/docker
WORKBENCH_ENABLE_GPU=1 ./run.sh up          # compose up + gpu override + doctor

# 3. Chuẩn bị (sync, môi trường, dataset, checkpoints, GPU probe)
./run.sh prepare limn                        # stage 7/7 gpu-verification phải [OK]

# 4. Chạy pilot chính thức
./run.sh pilot limn                          # queue qua /api/jobs, sau đó ./run.sh status

# 5. Xem kết quả + gate
./run.sh logs                                # theo dõi job
# kết quả JSON ở results/fashioniq_val_split/limn/base_iter0_all_categories.json
# gate so paper: xem GateDetail trên UI (http://localhost:5173)
```

## 3. Điều kiện chuyển sang RELEASE_READY
1. Trên PC đích: `./run.sh prepare limn` hoàn tất 7/7 stage với stage 7 `[OK] GPU capability verified and recorded`.
2. `./run.sh pilot limn` queue thành công và job chạy đến có result JSON thật.
3. `release_verify.sh` trên PC đích: J5 (GPU_READY tier) PASS → verdict không còn pending GPU.

## 4. Ghi chú kiểm định độc lập
- Wave review DeepSeek (4 slice: docker/launch, runtime/provisioning, scientific gate, backend/scripts) chạy trên diff `3310b5b..staged`; chỉ P0/P1 xác nhận mới được sửa trước commit.
- Kết quả: **0 P0, 5 P1, 4 P2**; sửa toàn bộ P1 + P2 trước commit:
  1. `_conda_specs_for_group` và `_solver_solve` không đồng nhất về package rỗng-term (solver VERIFIED nhưng installer vẫn có thể chết) → thống nhất: cả hai giữ tên bare; package không resolve được (vd `datasets1`) giờ chặn tại solver với lý do rõ ràng, như installer.
  2. Fallback `environment_assignment.json` chấp nhận interpreter từ artifacts root khác (foreign-path adoption) → chỉ nhận interpreter nằm bên dưới `WORKBENCH_ARTIFACTS_ROOT` của operator hiện tại.
  3. `manage_environment` route planner dùng `next(...)` không default → `StopIteration` thay refusal khi registry root lệch → `next((...), None)` fall-through về model.
  4. `release_verify.sh` probe `/api/models` 2 lần không timeout (~70s/lần) → J8 PENDING giả khi stack sống; và verdict có thể nhảy RELEASE_READY sai điều kiện → probe 1 lần qua endpoint rẻ (`/api/jobs`), `--max-time 5`.
  5. Route planner chặn test e10 (unsafe-interpreter refusal bị bỏ qua) → đã sửa như trên (planner route chỉ cho fresh install không preset).
  - P2: mutation backup khỏi `mktemp` (SIGKILL không mất bản gốc); `release_verify_report.json` vào `.gitignore`.
- Gate slice: không có phát hiện P0/P1 — NaN→ALERT, missing→INCOMPLETE, không đường fake recall, không làm yếu B2 (xác nhận bằng probe in-process).
- Sau sửa: `release_verify.sh` = **12/12 PASS, 0 pending** (đã đo lại trên stack thật); `mutation_checks.sh` = **M1–M5 DETECTED/PASS**; suite planner+operator+adversarial+runtime = **73 passed**.
- Phán quyết verifier `RELEASE_READY` phạm vi-cấu-hình; phán quyết báo cáo vẫn **GPU_VALIDATION_PENDING** vì thiếu bằng chứng CUDA thật.
