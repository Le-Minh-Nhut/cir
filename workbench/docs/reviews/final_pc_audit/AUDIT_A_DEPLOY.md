# AUDIT A — Docker & PC deployment (main-agent audit; subagent harness unavailable)

Phạm vi: `workbench/docker/` (Dockerfile, compose, compose.gpu, .env.example, run.sh, acceptance.sh,
release_verify.sh, mutation_checks.sh), `workbench/config/`, `.dockerignore`,
`workbench/scripts/serve_workbench.py`, `pc.sh`, `workbench/backend/operator_config.py`.
Phương pháp: đọc nguồn + chạy thật trên host này (stack đang chạy) + `docker compose config` +
`docker inspect`. Không rebuild image.

## Đã VERIFY đúng

| Kiểm tra | Bằng chứng |
|---|---|
| Resolve `HOST_*` tương đối từ `workbench/docker` | `docker compose config` → `/home/heheboiz/data/cir/data/FashionIQ -> /data/FashionIQ` (đúng quy ước) |
| Dataset submount read-only kernel-enforced | `.Mounts` cho thấy `/data/FashionIQ/images` và `/image_splits` ở mode `ro` |
| Mount thật khớp cấu hình mong đợi | 7 mount, source đều nằm dưới repo |
| `pc.sh` chạy verifier trong container | `run_verifier_in_container` → `docker exec -T cir-workbench python3 workbench/scripts/verify_run.py` |
| `pc.sh --doctor` read-only, phát hiện đúng phần cứng | báo RTX 3050 (không phải 5070 Ti), driver 580.178.04 |
| Compose hợp lệ | `docker compose config` exit 0 |
| `pc.sh` cần mỗi chính nó | `bash` + `python3` (stdlib) + `docker`/`curl`; không cần venv host |
| Container chạy uid 1000, HOME/`/opt/mamba`/`node_modules` ghi được | smoke boot trước đó |

## CONFIRMED — A-1 (P1) khóa ledger không có nhà bền vững cho uid ≠ 1000

**Vị trí:** `workbench/backend/runtime.py::_gpu_cache_key` và
`workbench/scripts/evaluate_models.py::ledger_hmac_key` (thứ tự: `WORKBENCH_LEDGER_KEY` → legacy
`<CIR_REPO_ROOT>/.workbench_ledger_key` → `$XDG_CONFIG_HOME/cir/workbench_ledger_key`).

**Nguyên nhân gốc:** ưu tiên legacy đi trước config-home. Trong deployment, `/repo` là checkout
(root-owned, baked vào image), nên `/repo/.workbench_ledger_key` **luôn tồn tại** với chủ sở hữu root
và được chọn trước — bất kể `XDG_CONFIG_HOME=/artifacts/config` (bind mount ghi được) đã được cấu
hình. Đo được:

```
$ docker exec cir-workbench python -c "...ledger_hmac_key..."
container key len: 32
$ docker exec cir-workbench ls -la /repo/.workbench_ledger_key
-rw------- 1 root root 32 Oct 15 13:21 /repo/.workbench_ledger_key
$ docker exec cir-workbench ls -la /artifacts/config/cir/workbench_ledger_key
No such file or directory
$ $XDG_CONFIG_HOME trong container: None (docker exec không kế thừa env của compose)
```

**Tác động:** với container chạy uid 1000 (mặc định compose), `/repo` không ghi được → key không tạo/
không cập nhật được nữa; và vì key nằm ngoài bind mount nên **mất khi container được tạo lại** →
mọi entry ledger cũ trở thành không xác minh được. Đây là bất biến khoa học (ledger là bằng chứng),
không phải chuyện tiện dụng.

**Trạng thái hiện tại:** máy này `read_evidence_ledger` trả `False` (chuỗi HMAC không xác minh với
ledger on-disk). Vì `verify_run.py` từ chối mọi run không có witness xác minh, điều này sẽ chặn mọi
lần xác thực trên PC cho tới khi khóa được đặt đúng chỗ.

**Khắc phục tối thiểu:** đảo thứ tự ưu tiên — config-home (bền vững, nằm trong bind mount) TRƯỚC
legacy; hoặc ghim `WORKBENCH_LEDGER_KEY` trong `.env`/secret store của deployment. Không được đổi
ngầm: ledger đã ký bằng khóa cũ phải được di trú có báo cáo (hàm đã có nhánh “hai khóa khác nhau →
từ chối” — giữ nhánh đó).

## CONFIRMED — A-2 (P2) `docker exec` không kế thừa môi trường compose

**Vị trí:** `pc.sh` dùng `docker exec -T cir-workbench ...` cho pipeline, GPU probe và verifier;
`docker exec` chỉ lấy `ENV` của image + `environment:` của service, **không** lấy các biến chỉ có
trong `.env` khi chúng không được khai báo trong `environment:`.

**Bằng chứng:** `docker exec cir-workbench sh -c 'echo XDG=$XDG_CONFIG_HOME'` → rỗng, trong khi
compose khai `XDG_CONFIG_HOME: /artifacts/config` (biến này có trong `.Config.Env`, nhưng tiến trình
do `docker exec` chạy hiện tại không thấy).

**Tác động:** mọi quyết định trong container dựa trên `XDG_CONFIG_HOME`/`WORKBENCH_*` có thể khác với
quyết định của tiến trình chính của container. Đây là biên HOST↔CONTAINER cần kiểm.

**Khắc phục tối thiểu:** khi cần một biến cho lệnh `docker exec`, truyền tường minh
`docker exec -e KEY=VALUE`, hoặc đọc lại từ `.env` ở host và truyền vào; và/hoặc kiểm tra giá trị
thật bằng `docker inspect .Config.Env` thay vì giả định.

## CONFIRMED — A-3 (P2) `pc.sh` gọi `docker exec` cho nhiều lệnh không kiểm exit code

**Vị trí:** `pc.sh` quanh dòng 657-692: `docker exec cir-workbench python -c "import torch; ..."`
và `docker exec cir-workbench test -d /data/FashionIQ/images` được dùng làm điều kiện, nhưng lệnh
`sanitize_exec`/`|| true` ở vài chỗ nuốt lỗi, nên “không chạy được” và “chạy và thất bại” có thể bị
trộn. Cần rà từng chỗ để chắc chắn fail-closed (partial: một số chỗ đã đúng nhờ `|| true` + kiểm mã).

## Chưa kiểm (nói rõ)
- GPU passthrough thật (không có nvidia-container-toolkit trên máy này).
- `docker-compose.gpu.yml` tương tác với `deploy.resources` trên Docker 29 (không có GPU để thử).
- Build lại image từ đầu (không rebuild để tiết kiệm thời gian/token; image hiện tại đã smoke).
