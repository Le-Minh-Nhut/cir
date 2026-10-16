# AUDIT B — Runtime & GPU (main-agent audit; subagent harness unavailable)

Phạm vi: `workbench/backend/runtime.py`, `environment_planner.py`, `scripts/manage_environment.py`,
`scripts/pc_readiness.py`, `scripts/pipeline.py`, `backend/registry.py`, `registry/models.yaml`.
Phương pháp: chạy thật các probe bằng `.venv/bin/python` + đọc nguồn. Không có GPU thật trên máy này.

## VERIFIED — contract runtime thống nhất

`WORKBENCH_RUNTIME_ADAPTATION=sm_120` đi qua mọi hop với **một** contract:

| Model | adaptation | `resolved_runtime_environment` | planner `_runtime_model` | digest |
|---|---|---|---|---|
| limn | `sm_120` | `pytorch>=2.7,<3` | `pytorch>=2.7,<3` | `042baec6…` |
| csmcir | none | `2.0.1` (lịch sử) | `2.0.1` | `f7f829a3…` |

→ Biến toàn cục không ép model không khai báo vào profile của LIMN; hai contract có digest khác nhau
nên cache/environment id không lẫn. Đây là hành vi đúng, giữ nguyên.

## VERIFIED — ràng buộc danh tính capability (đã sửa ở đợt trước, kiểm lại)

- `_record_environment_fingerprint` tính fingerprint **authoritative** cùng nguồn với
  `_gpu_capability_key` (cùng interpreter, cùng prefix `interpreter.parent.parent` không resolve,
  cùng contract digest) → người ghi và mọi người đọc tính ra cùng giá trị.
- Bằng chứng: hai interpreter khác nhau → fingerprint khác nhau; record của interpreter A **bị từ
  chối** khi interpreter đang cấu hình là B (`_cached_gpu_capability` trả `None`).
- Record hợp lệ vẫn được tái dùng mà không probe lại (`test_first_run_gpu_capability_state_machine`):
  `probe_invocations == 1` sau `forget_gpu_capability()`.

## VERIFIED — phân tách advisory vs authoritative đúng chỗ

- Đường advisory (đọc trạng thái/readiness, `environment_status`) thực hiện **0** lần hash nội dung
  thành viên; memo theo stat/mtime.
- Đường authoritative (probe GPU, launch gate, receipt) hash đầy đủ; rewrite nội dung cùng
  size+mtime vẫn bị phát hiện.
- `_cached_gpu_capability` so fingerprint authoritative trên **cả hai** đường, nên không có
  đường nào bỏ qua ràng buộc danh tính.
- Tác động hiệu năng đã đo trước đây: `checkpoint_availability(limn)` 11.06 s → 0.97 s;
  `sha256_file` 129,751 → 0 lần trên đường availability.

## CONFIRMED — B-1 (P1) GPU capability không thể ký trong deployment (cùng gốc A-1)

`runtime._gpu_cache_key` và `evaluate_models.ledger_hmac_key` đều ưu tiên legacy
`<CIR_REPO_ROOT>/.workbench_ledger_key`. Trong container `/repo` là checkout root-owned, nên khóa
được tạo/đọc ở đó nhưng (a) uid 1000 không ghi được, (b) **không nằm trong bind mount** nên mất khi
container được tạo lại → mọi record capability và entry ledger cũ mất khả năng xác minh →
`require_launch_ready` đóng gate GPU.

Trước sửa (đo được): container có `/repo/.workbench_ledger_key` (root:root), **không** có
`/artifacts/config/cir/workbench_ledger_key`; `read_evidence_ledger` → `False`.

**ĐÃ SỬA** (`ledger_hmac_key`): khóa mới tạo ở config-home bền vững (bind mount, uid operator ghi
được); legacy chỉ được ưu tiên khi nó **đã tồn tại và** chưa có khóa config-home (vì nó đã ký ledger
hiện có — đổi vị trí sẽ làm mọi entry cũ không xác minh được); hai khóa khác nhau → từ chối như cũ.
Ba case đã chứng minh: fresh → config-home; legacy có sẵn → vẫn dùng; hai khóa khác bytes → refuse.
Cùng logic đã được áp cho `_gpu_cache_key` từ đợt trước.

**Còn lại cho deployment:** cấp `WORKBENCH_LEDGER_KEY` qua secret store là cách chắc chắn nhất;
hoặc xác nhận `/artifacts/config/cir/workbench_ledger_key` tồn tại sau lần chạy đầu (nó sẽ được tạo
tự động).

## CONFIRMED — B-2 (P2) `docker exec` không thấy `XDG_CONFIG_HOME` của compose

Xem A-2: `docker exec` không kế thừa `environment:` không có trong `ENV` của image khi môi trường
được đặt bởi `docker exec` session hiện tại. Hệ quả trực tiếp: `pc.sh` chạy verifier trong container
qua `docker exec` — nếu `XDG_CONFIG_HOME` không hiện diện, khóa sẽ resolve về `$HOME/.config` của
uid trong container, khác với khóa ledger đang dùng. **Khắc phục tối thiểu:** truyền tường minh
`docker exec -e XDG_CONFIG_HOME=/artifacts/config` (và các biến `WORKBENCH_*` cần thiết) trong
`run_verifier_in_container`, hoặc đọc giá trị thật từ `docker inspect .Config.Env`.

## Chưa kiểm (nói rõ)
- Thực thi kernel CUDA thật trên sm_120 (không có GPU Blackwell ở đây).
- `pip install` của environment adapted trên máy đích (cần mạng + driver đúng).
