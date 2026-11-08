# ENVIRONMENTS — nhật ký môi trường

Phiên này **không tự tạo conda/uv env**. Việc provisioning do agent riêng `env-py39` sở hữu; báo cáo
của họ ở `workbench/docs/reviews/20261108T113110-envs/` (`env_map.json`, `ENVS.vi.md`, cùng 20 file
log). Dưới đây là những gì **tôi tự kiểm lại** và những gì **tôi dùng để chạy**.

## 1. Interpreter dùng để chạy model

| env | path | python | torch | torchvision | CUDA build | dùng cho |
|---|---|---|---|---|---|---|
| `ilearn39-adapted` | `workbench/envs/ilearn39-adapted/bin/python` | **3.9.25** | **2.7.1+cu128** | **0.22.1+cu128** | 12.8 | habit, intent, hint, conesep |
| `tme-adapted` | `workbench/envs/tme-adapted/bin/python` | 3.10.22 | 2.7.1+cu128 | (2.8.0/0.23.0 theo env map) | 12.8 | tme |
| `cir-005ef4e1b02999a5` | `workbench/artifacts/environments/cir-005ef4e1b02999a5/bin/python` | 3.10.22 | 2.7.1+cu128 | — | 12.8 | ptha_mtst |
| `limn-adapted` | `workbench/envs/limn-adapted/bin/python` | 3.11.15 | — | — | — | limn (**chưa** xác nhận khớp contract) |
| `pair-adapted` | `workbench/envs/pair-adapted/bin/python` | 3.11.15 | — | — | — | encoder (**chưa** khớp contract py 3.10) |

Control interpreter (không phải model): `workbench/envs/ctl312h/bin/python`, CPython **3.12.11**.

## 2. Kiểm chứng độc lập của tôi trên `ilearn39-adapted`

Chạy lại, không tin báo cáo của agent kia:

```sh
workbench/envs/ilearn39-adapted/bin/python -c "
import sys, torch, torchvision
print('python', '%d.%d.%d'%sys.version_info[:3])
print('torch', torch.__version__); print('torchvision', torchvision.__version__)
print('cuda avail', torch.cuda.is_available())
print('capability', torch.cuda.get_device_capability())
print('arch_list', torch.cuda.get_arch_list()[-4:])
x=torch.randn(64,64,device='cuda'); print('matmul finite', bool((x@x).isfinite().all()))"
```

Kết quả quan sát được (2026-11-08):

```
python 3.9.25
torch 2.7.1+cu128
torchvision 0.22.1+cu128
cuda avail True
capability (12, 0)
arch_list ['sm_90', 'sm_100', 'sm_120', 'compute_120']
matmul finite True
```

→ đúng: Python 3.9, torch/torchvision cu128, CUDA khả dụng, **capability (12, 0)**, có kernel `sm_120`,
tensor op thật chạy và hữu hạn.

## 3. GPU probe chính thức (`--verify-gpu-only`) — do tôi chạy

Đây là phần **agent env không làm** (họ không được chạm GPU). Log ở `logs/gpu_probe_*.log`.

| model | checkpoint | exit | `compute_capability` | `allocated_tensor` | `computed` | `capability_verified` | `dependency_fingerprint` |
|---|---|---|---|---|---|---|---|
| habit | fiq_n02 | 0 | 12.0 | true | true | **true** | `20035a87d309` |
| intent | fiq_n02 | 0 | 12.0 | true | true | **true** | `09ed1ee20950` |
| hint | fashioniq | 0 | 12.0 | true | true | **true** | `496b8d4da683` |
| conesep | fiq_n02 | 0 | 12.0 | true | true | **true** | `02337376d0be` |

Ghi nhận vào `workbench/artifacts/gpu_capabilities/gpu_capability.json` và file `<sha256>.json` tương ứng.

**Lần chạy đầu `intent` exit 137** — nguyên nhân là **tôi** `kill` cả cây process khi dừng một vòng
probe quá rộng. Chạy lại riêng: exit 0. Không phải lỗi môi trường. Đã ghi ở `RUNS.vi.md` §1.

## 4. Biến môi trường bắt buộc khi chạy

`WORKBENCH_RUNTIME_ADAPTATION=sm_120` là **bắt buộc** cho habit/intent/hint/conesep: không có nó,
contract lịch sử (py 3.8.10 / torch 2.1.0) được áp và interpreter 3.9 bị từ chối với
`PYTHON_VERSION_MISMATCH`. Bằng chứng phía agent env: `verify_no_adaptation.txt`.

Hệ quả: mọi run của bốn model này là **ADAPTED RUNTIME**, **không** phải reproduction nguyên bản.
Nhãn đó phải đi cùng mọi con số.

## 5. Vì sao không có Python 3.8.10

Registry khai `python: '3.8.10'` cho habit/hint/conesep (airknow cũng vậy), nhưng
`https://download.pytorch.org/whl/cu128/torch/` **không có wheel cp38 nào**; torch ≥2.7 là bản đầu
tiên có kernel sm_120. Registry **tự nó** đã khai `adapted_runtime.python: '3.9'` cho các model này,
nên 3.9 là contract adapted chính thức, không phải né tránh.

## 6. Điều chưa làm được

- `encoder`: env `pair-adapted` là py 3.11.15, contract adapted là **3.10**. Chưa có env 3.10 +
  `open_clip_torch==2.24.0`. Không chạy `encoder` trong phiên này (BLOCKERS §B2).
- `limn`: env `limn-adapted` là py 3.11.15, contract adapted là **3.10** + `open_clip_torch==2.20.0`.
  Chưa xác nhận. (`limn-adapted` là env đã sinh ra kết quả cũ theo registry, nên cần đối chiếu.)
- `csmcir`: contract adapted là py 3.10; chưa provision. Nhưng model này **bị chặn ở adapter**, không
  phải môi trường (BLOCKERS §B1) — dựng env cũng không chạy được.
