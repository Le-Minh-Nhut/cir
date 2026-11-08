# Môi trường Python cho cohort FashionIQ bị chặn bởi cổng Python

Ngày: 2026-11-08. Nhánh `research/cir-failure-workbench`. Không commit, không đụng git, không chạy
evaluation/training/encode gallery. GPU chỉ dùng cho smoke test cỡ nhỏ, tuần tự.

## Kết luận ngắn

Bốn model `habit`, `intent`, `hint`, `conesep` **dùng chung một môi trường duy nhất**:

```
/home/heheboiz/data/cir/workbench/envs/ilearn39-adapted/bin/python
```

Đây là env mới (`uv venv --python 3.9`, CPython 3.9.25) với `torch==2.7.1+cu128`,
`torchvision==0.22.1+cu128`, CUDA build 12.8, chạy được trên RTX 5070 Ti `sm_120`.

Ba môi trường "adapted" đang có (`airknow-adapted`, `pair-adapted`, `limn-adapted`) đều là Python
3.11; `tme-adapted` là 3.10. Không env nào khớp `python: '3.9'` trong `adapted_runtime`, nên contract
3.9 buộc phải dựng env mới. `workbench.env` hiện trỏ cả bốn model vào `airknow-adapted`, và
`verify_environment` từ chối nó:

```
status=PYTHON_VERSION_MISMATCH reason="expected Python 3.9, found 3.11.15"
```
(log: `verify_airknow_baseline.txt`)

## Vì sao không dùng Python 3.8.10 (pin lịch sử)

`https://download.pytorch.org/whl/cu128/torch/` **không có wheel cp38 nào cho x86_64** (0 kết quả
khớp `cp38-cp38-manylinux*x86_64`); các bản 2.7.1/0.22.1 cu128 chỉ phát hành cp39+. Đồng thời
`sm_120` cần torch ≥ 2.7. Đây là lý do khách quan khiến pin lịch sử 3.8.10 + torch 2.1.0 không thể
chạy trên máy này, và registry đã ghi rõ `adapted_runtime.python: '3.9'` cho đúng lý do đó. Chọn
Python **3.9** là khớp `adapted_runtime`, không phải tự hạ cấp.

## Môi trường đã tạo

| trường | giá trị |
|---|---|
| Đường dẫn tuyệt đối | `/home/heheboiz/data/cir/workbench/envs/ilearn39-adapted` |
| Interpreter | `/home/heheboiz/data/cir/workbench/envs/ilearn39-adapted/bin/python` |
| Python | 3.9.25 (uv-managed CPython, `~/.local/share/uv/python/cpython-3.9.25-linux-x86_64-gnu`) |
| torch | `2.7.1+cu128` |
| torchvision | `0.22.1+cu128` |
| CUDA build | 12.8 (`torch.version.cuda == "12.8"`) |
| `torch.cuda.get_arch_list()` | `sm_75, sm_80, sm_86, sm_90, sm_100, sm_120, compute_120` |
| GPU quan sát được | `NVIDIA GeForce RTX 5070 Ti`, capability `[12, 0]` = `sm_120` |
| `Python.h` | có: `<uv python>/include/python3.9/Python.h`; build C extension đã chứng minh OK |
| Dung lượng | 7.0 GiB (`du -sh`) |
| Model phục vụ | `habit`, `intent`, `hint`, `conesep` |

Bốn model dùng chung được vì `adapted_runtime` của cả bốn **giống hệt nhau**: python 3.9, torch
==2.7.1, torchvision ==0.22.1, cuda 12.8, cùng một `pip_index_url`, và cùng ba pin lõi
`transformers==4.25.0`, `timm==0.9.16`, `fairscale==0.4.13`, `numpy<2`.

## Lệnh cài đặt, theo thứ tự, kèm exit code

Chạy tại `/home/heheboiz/data/cir`. Log nằm cùng thư mục này.

```sh
# 1) tạo env. exit=0
~/.local/bin/uv venv --python 3.9 workbench/envs/ilearn39-adapted
#    (cảnh báo vô hại: "resolved to Python 3.9.25, which is incompatible with the project's
#     Python requirement >=3.11" -- pyproject.toml là của orchestrator, không phải của model)

# 2) torch cu128. exit=0  -> install_torch.log
~/.local/bin/uv pip install --python workbench/envs/ilearn39-adapted/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1 torchvision==0.22.1

# 3) danh sách packages của registry (hợp của cả bốn model). exit=0 -> install_packages.log
~/.local/bin/uv pip install --python workbench/envs/ilearn39-adapted/bin/python \
  "transformers==4.25.0" "timm==0.9.16" "omegaconf==2.3.0" "opencv-python-headless==4.5.5.64" \
  "pandas==2.0.3" "scikit-image==0.21.0" "decord==0.6.0" "einops==0.7.0" "fairscale==0.4.13" \
  ftfy "iopath==0.1.10" "peft==0.6.2" "diffusers==0.19.3" "huggingface-hub==0.19.4" \
  "accelerate==0.25.0" "pycocoevalcap==1.2" pycocotools "webdataset==0.2.86" "scikit-learn==1.3.2" \
  matplotlib "numpy<2" pyyaml regex requests tqdm packaging Pillow
```

Không có lần cài nào thất bại, nên **không có pin nào bị bỏ hay bị hạ**. Danh sách đầy đủ 84
distribution: `freeze.txt`.

`uv pip install` in `transformers==4.25.0` là bản bị yank ("Version was not properly set") — đây là
pin registry và là bản đã được kiểm chứng trước đó, cài được bình thường.

## Kiểm chứng (đã chạy thật, tuần tự)

| phép kiểm | kết quả | log |
|---|---|---|
| `python -V` | 3.9.25 | `verify_packages.txt` |
| `torch.__version__` / `torch.version.cuda` | `2.7.1+cu128` / `12.8` | `gpu_smoke.txt` |
| `torch.cuda.is_available()` | `True` | `gpu_smoke.txt` |
| device name / capability | `NVIDIA GeForce RTX 5070 Ti` / `[12, 0]` | `gpu_smoke.txt` |
| matmul 64×64 fp32 trên GPU | tổng `-361.07`, hữu hạn | `gpu_smoke.txt` |
| matmul fp32 + bf16, conv2d, layer_norm, SDPA attention | tất cả hữu hạn, `reserved_MiB=22.0` (đã free) | `gpu_ops_smoke.txt` |
| probe cô lập `-I -S` (đúng cách runtime chạy) | import torch OK, capability `[12, 0]` | `iso_probe.txt` |
| import module adapter (`workbench.composition.model_<m>`) | OK cả bốn | `worker_module.txt`, `import_chain_39.txt` |
| `--worker-job` entrypoint thật của worker | chạy tới kiểm tra checkpoint và báo lỗi nghiệp vụ, không lỗi import | `worker_job_habit.txt` |
| `_load_replay_modules` (import `datasets`, `data_utils`, `test` **của checkout**) | OK cả bốn; origin đúng thư mục `third_party/<Model>` | `module_origins.txt` |
| class model trong `lavis` (`HABIT`, `Blip2QformerCir`, `HINT`, `ConeSep`) | OK cả bốn | `unpickle.txt` |
| unpickle checkpoint thật trên CPU (SHA-256 khớp trước khi load) | OK: 1 172 624 574 / 1 172 821 439 / 1 173 374 720 / 1 172 624 573 tham số | `unpickle.txt`, `checkpoint_check.txt` |
| `runtime.verify_environment` với `WORKBENCH_RUNTIME_ADAPTATION=sm_120` | tier `GPU_VERIFICATION_PENDING` cho cả bốn (**đúng**: đã qua Python + dependency, chỉ còn chờ probe GPU chính thức) | `verify_environment.txt`, `verify_deps.txt` |
| `ADAPTER.status()` của cả bốn adapter | `ready: true` | `adapter_status.txt` |

`GPU_VERIFICATION_PENDING` là trạng thái mong đợi chứ không phải lỗi: nó chỉ hết khi chạy
`workbench.scripts.evaluate_models --verify-gpu-only`, việc này bị cấm trong lượt này (agent khác
giữ GPU). Bước đó chỉ ghi lại capability; mọi thứ nó cần để thành công (interpreter đúng, CUDA
dùng được, `sm_120` nằm trong arch list) đã được chứng minh ở trên.

## Khác biệt so với contract, có chủ ý

1. **`salesforce-lavis` không cài.** Registry đã giải thích: `lavis` là bản trong checkout, được
   `_load_replay_modules` đưa lên `sys.path` nên nó che distribution cùng tên; wheel PyPI pin
   `timm==0.4.12` trong khi `eva_vit.py` cần `timm==0.9.16`. Đã xác nhận bằng `module_origins.txt`
   rằng `datasets`/`data_utils`/`test` resolve vào `third_party/<Model>`, và các class model import
   từ `lavis.models.blip2_models.<X>` của checkout cũng OK.
2. **`torchaudio` không cài.** README có nhắc `torchaudio==2.1.0`, nhưng grep toàn bộ đường đi
   evaluator (`test.py`, `datasets.py`, `data_utils.py`, các class `blip2_models`, `base_model`) không
   có chỗ nào import nó; `lavis_processors.py` cố tình chèn skeleton để tránh `lavis.processors`
   kéo `torchaudio`. Kiểm tra runtime xác nhận `torchaudio` **không** nằm trong `sys.modules` sau
   `_load_replay_modules` (`reachable_modules.txt`).
3. **`open_clip-torch==2.24.0`, `thop`, `setproctitle`, `sentencepiece` không cài.** Chỉ `train.py` /
   `train_INTENT.py` / `cirr_sub*.py` dùng; không đường nào tới evaluator. Cùng file
   `reachable_modules.txt` xác nhận cả bốn model không import chúng.
4. **`pandas==2.0.3`, `scikit-image==0.21.0`, `opencv-python-headless==4.5.5.64`** cài đúng pin (khác
   với `airknow-adapted` đang có pandas 2.2.0 / skimage 0.22.0 / cv2 4.5.5).
5. **`numpy==1.26.4`**: thoả `numpy<2` của adapted block.

## `pip check` báo một mục — đã kiểm tra, không ảnh hưởng

```
The package `decord` was built for a different platform
```
(`pip_check.log`). `import decord` chạy được, `decord.__version__ == 0.6.0`, `VideoReader` có mặt.
Metadata wheel của `decord` ghi platform cũ; và cả bốn evaluator không dùng `decord` cho FashionIQ
(ảnh tĩnh). Giữ pin vì registry khai nó.

## Điều còn lại trên lý thuyết nhưng cần GPU chính thức

- `evaluate_models --verify-gpu-only` cho bốn model: chưa chạy (bị cấm trong lượt này).
- Load model lên GPU và forward thật: chưa chạy (bị cấm). Đã chạy phần CPU tương đương — unpickle
  toàn bộ checkpoint có kiểm SHA.
- `workbench/backend/environment_planner.py` **không import được trên Python 3.9**:
  `SyntaxError: EOL while scanning string literal` ở dòng 669 (f-string lồng dấu nháy, syntax chỉ hợp
  lệ từ 3.12). Xem `import_chain_39.txt`, `compileall_39.log`. Đây là code của orchestrator, **không**
  phải code của model: nó chạy bằng `workbench/envs/ctl312h` (3.12) ở tiến trình cha, còn worker model
  chạy bằng env này và không import nó. Nếu có ai định chạy một script *trong* env model mà import
  `environment_planner`, nó sẽ hỏng — sửa ở file đó là việc của chủ sở hữu backend, không phải ở đây.

## Giá trị biến môi trường

```
WORKBENCH_HABIT_PYTHON=/home/heheboiz/data/cir/workbench/envs/ilearn39-adapted/bin/python
WORKBENCH_INTENT_PYTHON=/home/heheboiz/data/cir/workbench/envs/ilearn39-adapted/bin/python
WORKBENCH_HINT_PYTHON=/home/heheboiz/data/cir/workbench/envs/ilearn39-adapted/bin/python
WORKBENCH_CONESEP_PYTHON=/home/heheboiz/data/cir/workbench/envs/ilearn39-adapted/bin/python
WORKBENCH_RUNTIME_ADAPTATION=sm_120
```

`WORKBENCH_RUNTIME_ADAPTATION=sm_120` là **bắt buộc**: thiếu nó, contract quay về pin lịch sử và
interpreter bị từ chối (`verify_no_adaptation.txt`):
`expected Python 3.8.10, found 3.9.25` với `habit`/`hint`/`conesep`, và
`torch: expected ==2.1.0, found 2.7.1+cu128` với `intent`.
