# Môi trường Python 3.10 cho `encoder` và `limn` (adapted_runtime sm_120)

Tiếp nối `ENVS.vi.md` (cùng thư mục). Ngày 2026-11-08. Không commit, không đụng git, không restart
web service, GPU dùng cho một phép toán nhỏ rồi giải phóng.

## Kết luận

Hai `adapted_runtime` **không tương thích**, nên phải dựng **hai** env:

| model | env | python | open_clip_torch |
|---|---|---|---|
| `limn` | `workbench/envs/limn310-adapted` | 3.10.20 | **2.20.0** |
| `encoder` | `workbench/envs/encoder310-adapted` | 3.10.20 | **2.24.0** |

### Vì sao không gộp được một env

Hai block giống nhau ở mọi trường (`python: '3.10'`, `pytorch: '==2.7.1'`, `torchvision: '==0.22.1'`,
`cuda: '12.8'`, cùng `pip_index_url` cu128) **trừ đúng một pin, và pin đó là exact-match ở cả hai phía**:

- `encoder.adapted_runtime.packages: ['open_clip_torch==2.24.0']`
  (registry dẫn nguồn: `workbench/third_party/ENCODER/requirements.txt:155`).
- `limn.adapted_runtime.packages: ['open_clip_torch==2.20.0']`
  (registry dẫn nguồn: `workbench/third_party/LIMN/requirements.txt:37`).

Không chỉ là metadata: `workbench/replay/limn.py:94` có `validate_environment()` đọc chính
`adapted_runtime` của LIMN rồi so `importlib.metadata.version(...)` với `SpecifierSet` — cài 2.24.0
vào env LIMN sẽ bị chính replay wrapper từ chối (`LIMN environment version mismatch: expected
open_clip_torch ==2.20.0, found 2.24.0`), và ngược lại. `runtime.verify_environment` kiểm cùng pin
theo hướng thứ hai. Một env không thể thoả cả hai mà không sửa declaration — mà sửa declaration là
việc phải tranh luận, không phải việc của lượt này.

Ghi chú có đo: registry đã ghi rõ cả 2.20.0 và 2.24.0 đều unpickle được bundle LIMN, nên khác biệt
này không lộ ra lúc load — nó chỉ hiện ở gate phiên bản. Vì vậy chọn đúng pin theo nguồn, không chọn
"bản nào cũng chạy".

## Lệnh cài, exit code, log

Chạy tại `/home/heheboiz/data/cir`. Tất cả exit=0; **không có pin nào bị bỏ hay hạ cấp**.

### LIMN — `workbench/envs/limn310-adapted`

```sh
~/.local/bin/uv venv --python 3.10 workbench/envs/limn310-adapted          # exit 0
~/.local/bin/uv pip install --python workbench/envs/limn310-adapted/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1 torchvision==0.22.1
                                                                            # exit 0  f_up_install_torch_limn310.log
~/.local/bin/uv pip install --python workbench/envs/limn310-adapted/bin/python \
  "open_clip_torch==2.20.0"                                                 # exit 0  f_up_install_openclip_limn310.log
~/.local/bin/uv pip install --python workbench/envs/limn310-adapted/bin/python \
  "numpy<2" Pillow tqdm                                                     # exit 0  f_up_install_limn310_base.log
```

### ENCODER — `workbench/envs/encoder310-adapted`

```sh
~/.local/bin/uv venv --python 3.10 workbench/envs/encoder310-adapted        # exit 0
~/.local/bin/uv pip install --python workbench/envs/encoder310-adapted/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1 torchvision==0.22.1
                                                                            # exit 0  f_up_install_torch_encoder310.log
~/.local/bin/uv pip install --python workbench/envs/encoder310-adapted/bin/python \
  "open_clip_torch==2.24.0"                                                 # exit 0  f_up_install_openclip_encoder310.log
~/.local/bin/uv pip install --python workbench/envs/encoder310-adapted/bin/python \
  "transformers==4.30.2" "einops==0.4.1" "opencv-python-headless==4.5.5.64" \
  "numpy<2" Pillow tqdm                                                     # exit 0  f_up_install_encoder310_deps.log
```

`transformers==4.30.2` và `einops==0.4.1` là pin của `ENCODER/requirements.txt` (dòng 284 và 56);
chúng cần thiết thật: `model_try2.py:8` import `CLIPTextModel, CLIPVisionModelWithProjection,
CLIPImageProcessor, AutoTokenizer, CLIPTextModelWithProjection` từ `transformers`, và dòng 13
`from einops import rearrange`. `cv2` cần cho `datasets.py:13`.

LIMN **không** cần `transformers`/`einops`/`cv2`: import của `LIMN/LIMN/` chỉ là
`numpy, PIL, torch, tqdm, open_clip` cộng module nội bộ `Transformer` của chính nó.

`uv venv --python 3.10` in cảnh báo "incompatible with the project's Python requirement >=3.11" —
đó là `pyproject.toml` của orchestrator, không phải của model.

## Kiểm chứng

### Định danh

| | limn310-adapted | encoder310-adapted |
|---|---|---|
| python | 3.10.20 | 3.10.20 |
| torch | 2.7.1+cu128 | 2.7.1+cu128 |
| torchvision | 0.22.1+cu128 | 0.22.1+cu128 |
| `torch.version.cuda` | 12.8 | 12.8 |
| arch list | `sm_75, sm_80, sm_86, sm_90, sm_100, sm_120, compute_120` | giống |
| device | RTX 5070 Ti | RTX 5070 Ti |
| capability | `[12, 0]` = `sm_120` | `[12, 0]` = `sm_120` |
| matmul 64×64 trên GPU | `-1332.275146484375`, hữu hạn | `336.82635498046875`, hữu hạn |
| VRAM đỉnh | 22.0 MiB, đã `empty_cache()` | 22.0 MiB, đã `empty_cache()` |

Log: `f_up_packages.txt`, `f_up_cuda_smoke.txt`.

Hai phép matmul chạy **tuần tự**, mỗi phép ~1.5 s. Lúc chạy, GPU đang ở 11 % (agent kia vừa xong
một lượt); không giữ VRAM sau khi xong.

### `open_clip` tạo đúng biến thể mà evaluator hỏi

- **encoder** — đúng call site `workbench/replay/encoder.py:291` và
  `workbench/composition/model_encoder.py:320`:
  `open_clip.create_model_and_transforms("ViT-B-32", pretrained=<checkout>/open_clip_pytorch_model.bin)`
  → OK, class `CLIP`, `visual_image_size [224,224]`, `visual.output_dim 512`,
  `encode_image(zeros(1,3,224,224))` trên CPU → `[1, 512]`. (`open_clip_pytorch_model.bin`
  605 MB có sẵn trong checkout.)
- **limn** — đúng call site `workbench/replay/limn.py:178-190`: unpickle bundle
  `0_dress_best_model.pt` (SHA-256 khớp trước khi load) → `model.LIMN` (440 258 572 tham số),
  rồi `open_clip.image_transform(visual.image_size, is_train=..., mean=visual.image_mean,
  std=visual.image_std)` → OK `Compose`.

Log: `f_up_openclip_variant.txt`.

### Adapter / entrypoint

| phép kiểm | limn310 | encoder310 |
|---|---|---|
| `import workbench.composition.model_encoder` | OK | OK |
| `import workbench.composition.model_limn` | OK | OK |
| `workbench.replay.encoder._load_replay_modules` | (không áp dụng — thiếu `cv2`, xem dưới) | **OK** (5 module pinned) |
| module pinned của LIMN (`model`, `datasets`, `test`, `Transformer`) | **OK**, origin đúng checkout | không áp dụng |
| `limn.validate_environment()` | pin torch/torchvision/open_clip **đạt**; chỉ còn dòng "requires CUDA" vì lượt đó chạy với `CUDA_VISIBLE_DEVICES=""` | — |
| `ADAPTER.status()` sau khi đổi config | `ready: true` | `ready: true` |

Log: `f_up_adapters.txt`, `f_up_replay.txt`, `f_up_limn_native.txt`, `f_up_adapter_status.txt`.

`workbench.replay.limn` không có hàm `_load_replay_modules` (khác họ LAVIS): nó nạp module pinned
trực tiếp trong `load_native()`, nên phép kiểm tương đương là import bốn module pinned + kiểm
`__file__` nằm trong checkout, đã làm và đạt.

### `runtime.verify_environment` (qua config, không export biến)

`f_up_final_verify.txt`, chạy với `WORKBENCH_ENCODER_PYTHON`/`WORKBENCH_LIMN_PYTHON` **bị unset** để
đọc đúng từ `workbench/config/workbench.env`:

```
encoder: interpreter=.../envs/encoder310-adapted/bin/python  py=3.10.20  contract_python=3.10
         torch=2.7.1+cu128  torchvision=0.22.1+cu128  open_clip_torch=2.24.0
         tier=GPU_VERIFICATION_PENDING
limn:    interpreter=.../envs/limn310-adapted/bin/python     py=3.10.20  contract_python=3.10
         torch=2.7.1+cu128  torchvision=0.22.1+cu128  open_clip_torch=2.20.0
         tier=GPU_VERIFICATION_PENDING
```

Trước khi sửa config, cả hai là `PYTHON_VERSION_MISMATCH`:

```
encoder: expected Python 3.10, found 3.11.15   (pair-adapted)     f_up_verify_environment.txt
limn:    expected Python 3.10, found 3.11.15   (limn-adapted)     (cùng lý do)
```

`GPU_VERIFICATION_PENDING` là trạng thái **mong đợi**: nó chỉ hết sau
`evaluate_models --verify-gpu-only`, việc này chưa chạy (xem mục dưới).

## Thay đổi `workbench/config/workbench.env`

Backup: `workbench.env.bak` (cùng thư mục, `cp -p`, cùng mtime/kích thước 1342 byte).

```
$ diff -u workbench.env.bak workbench/config/workbench.env        -> f_up_config.diff
  @@ -7,8 +7,8 @@
  -WORKBENCH_ENCODER_PYTHON=/home/heheboiz/data/cir/workbench/envs/pair-adapted/bin/python
  -WORKBENCH_LIMN_PYTHON=/home/heheboiz/data/cir/workbench/envs/limn-adapted/bin/python
  +WORKBENCH_ENCODER_PYTHON=/home/heheboiz/data/cir/workbench/envs/encoder310-adapted/bin/python
  +WORKBENCH_LIMN_PYTHON=/home/heheboiz/data/cir/workbench/envs/limn310-adapted/bin/python
   WORKBENCH_HABIT_PYTHON=...ilearn39-adapted...
```

Đúng **hai dòng**. Không đụng `WORKBENCH_CHECKPOINT_ROOT`, các artifact root,
`WORKBENCH_RUNTIME_ADAPTATION=sm_120`, bốn biến ilearn, `WORKBENCH_PTHA_MTST_PYTHON`,
`WORKBENCH_TME_PYTHON`.

## Cần restart web service

`workbench.backend.operator_config.resolve_config()` đọc file một lần lúc import và giữ trong
`_CONFIG`; tiến trình web đang chạy vẫn giữ giá trị cũ. **Phải restart service để có hiệu lực** —
lượt này cố ý không restart vì agent khác đang dùng. Cho tới lúc restart, web vẫn báo hai model này
`PYTHON_VERSION_MISMATCH` dù file đã đúng.

Trong lúc chờ, ai muốn kiểm ngay thì export biến môi trường (biến môi trường **thắng** file):

```sh
export WORKBENCH_ENCODER_PYTHON=/home/heheboiz/data/cir/workbench/envs/encoder310-adapted/bin/python
export WORKBENCH_LIMN_PYTHON=/home/heheboiz/data/cir/workbench/envs/limn310-adapted/bin/python
export WORKBENCH_RUNTIME_ADAPTATION=sm_120
```

## Đĩa

| | |
|---|---|
| `encoder310-adapted` | 6.6 GiB (apparent) |
| `limn310-adapted` | 6.4 GiB (apparent) |
| **tổng thực tế cả hai** | **6.63 GiB** — uv hardlink chung wheel giữa hai env, `du -sc` xác nhận |
| `/home` | 503 G dùng / 95 G trống (trước lượt: 468 G / 131 G) |

## Còn lại

1. **Chưa chạy `--verify-gpu-only`** cho `encoder`/`limn`: GPU do agent khác sở hữu. Mọi thứ probe đó
   cần đã được chứng minh ở trên (interpreter đúng, CUDA dùng được, `sm_120` có trong arch list,
   capability `(12,0)`).
2. **Chưa load model lên GPU / chưa forward thật**: bị cấm trong lượt này. Phần tương đương trên CPU
   đã chạy: unpickle LIMN + `image_transform`, và encoder tạo `ViT-B-32` từ bin của checkout rồi
   `encode_image` trên CPU.
3. **Không có pin nào thất bại**, nên **không cần đề xuất sửa `adapted_runtime.python` thành 3.11**.
   Điều kiện "nếu 3.10 không thoả" trong yêu cầu đã không xảy ra: 3.10.20 cài đủ mọi pin, gồm cả
   `torch==2.7.1+cu128` / `torchvision==0.22.1+cu128` (cp310 có trên index).
