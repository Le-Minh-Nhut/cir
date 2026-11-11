# Chặn composition CIRCUS — CLVC-Net & TG-CIR (chẩn đoán, CPU-only)

Đo tại HEAD `a58b51db3529696519b2ce878ab302c5b0e3e837`, nhánh `research/cir-failure-workbench`, 2026-11-11;
lượt xác nhận lại chạy trên `a21e1318` (hai commit chen vào chỉ chạm `serve_workbench.py`, test
supervision và hai báo cáo; `git diff a58b51d..a21e131 -- workbench/composition workbench/registry
workbench/backend/runtime.py workbench/scripts/composition_eval.py workbench/replay` → **rỗng**).
Không chạy GPU, không forward pass, không sửa file nguồn nào. Mọi log nằm ở
`raw/compose_probe/`. `torch.cuda.is_available()` chỉ được **gọi**, không dùng để chạy gì:
`ctl312h` → `True`, 1 device, capability `(12, 0)` (`raw/compose_probe/cuda_availability.log`).

Ba interpreter dùng trong báo cáo (đều đọc được, đều có torch):

| env | python | torch | `cuda.is_available()` |
|---|---|---|---|
| `workbench/envs/ctl312h/bin/python` | 3.12.11 | 2.15.0+cu132 | True |
| `workbench/envs/clvc39-la/bin/python` | 3.9.25 | 2.7.1+cu128 / tv 0.22.1+cu128 | True |
| `workbench/envs/tgcir39-la/bin/python` | 3.9.25 | 2.7.1+cu128 / tv 0.22.1+cu128 | True |

(`raw/compose_probe/interpreter_cuda_check.log`)

---

## 1. M-04 · CLVC-Net — `workbench/composition/model_clvc_net.py`

| giai đoạn | kết quả | bằng chứng (file:dòng hoặc lệnh+exit) |
|---|---|---|
| Import adapter | **QUA**. Không có import nào fail, ở cả ba interpreter | `ctl312h/clvc39-la/tgcir39-la -c "import workbench.composition.model_clvc_net"` → `OK …/model_clvc_net.py`, `EXIT=0` |
| Import mã upstream đã pin | **QUA** | `clvc39-la -c "import datasets, test, text_model, model"` trong `third_party/CLVCNet` → `EXIT=0` (`raw/compose_probe/upstream_import_clvc.log`) |
| Checkpoint resolve | **QUA**, 6/6 thành viên, 6/6 sha256 khớp `expected_sha256` | `checkpoint_path("clvc_net", …)` → `…/cohort/downloads/../../checkpoints/clvc` (dir); `checkpoint_is_present` → `True`. Tổng 1 185 968 194 B (`raw/compose_probe/ckpt_resolve.log`) |
| Preprocess 1 mẫu thật | **QUA**: `(3, 224, 224)` `torch.float32`, min `-2.1179`, max `2.6400` | `Resize(256)→CenterCrop(224)→ToTensor→Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`, `workbench/replay/clvc_net.py:183-188`; mẫu `dress/B005X4PL1G.jpg` (15 286 B, sha256 `0804c5bb…7abe`); `EXIT=0` (`raw/compose_probe/preprocess_clvc.log`) |
| `Adapter.status()` khi **chưa** export `WORKBENCH_CLVC_PYTHON` | `ready=False`; lý do đầu tiên = "no interpreter environment resolved" | `workbench/config/workbench.env` **không** khai biến này (chỉ `.example:35`); `raw/compose_probe/status_default_env.log`, `EXIT=0` |
| `Adapter.status()` khi **đã** export | `ready=False`; hết lý do interpreter, chỉ còn cáo buộc CUDA | `raw/compose_probe/status_with_envvars.log`, `EXIT=0` |
| Cổng môi trường (`require_verified_interpreter`) | **DỪNG**: `RuntimeError` `BLOCKED/PYTHON_VERSION_MISMATCH`, expected 3.7.6 / found 3.9.25 | `workbench/composition/model_clvc_net.py:203` → `adapters.py:346` → `runtime.py:2309`; nguyên văn trong `raw/compose_probe/adapter_run_refusal.log` |
| Encoder | **KHÔNG TỒN TẠI**. Sau khi cổng được thoả, `_run` vẫn `raise Refused` vô điều kiện | `workbench/composition/model_clvc_net.py:205` `raise Refused(_refusal(adapter))`; không có `subprocess` nào trong module (chỉ 2 lần xuất hiện chuỗi, đều ở docstring `:192`, `:224`) |
| Đường vào thật của runner | `SELECTED_BUT_BLOCKED`, `EXIT=1`, **không** chạm `encode_queries` | `workbench/scripts/composition_eval.py:368-372` chặn theo `adapter_status.ready`; `raw/compose_probe/composition_eval_no_plan.log` |
| Preflight của đường replay (đối chứng) | **QUA**, `EXIT=0` | `clvc39-la -m workbench.replay.clvc_net … --preflight-only` → `raw/compose_probe/preflight_clvc.log` |

## 2. M-05 · TG-CIR — `workbench/composition/model_tgcir.py`

| giai đoạn | kết quả | bằng chứng (file:dòng hoặc lệnh+exit) |
|---|---|---|
| Import adapter | **QUA** ở cả ba interpreter | `ctl312h/clvc39-la/tgcir39-la -c "import workbench.composition.model_tgcir"` → `OK …/model_tgcir.py`, `EXIT=0` |
| Import mã upstream đã pin | **QUA** (`clip`, `datasets`, `test`, `model`) | `tgcir39-la -c "import clip, datasets, test, model"` → `EXIT=0` (`raw/compose_probe/upstream_import_tgcir.log`) |
| Checkpoint resolve | **QUA**: `Fashioniq.pt`, 607 928 637 B, sha256 `c37768f374c75bebff1299e870518c23bf6db3dc8a75c4d1f729ccd6dd02e966` — khớp `expected_sha256` **và** `expected_size_bytes` | `checkpoint_path("tgcir", …)` → `…/downloads/tgcir/x/checkpoints/Fashioniq.pt`; `checkpoint_is_present` → `True` (`raw/compose_probe/ckpt_resolve.log`) |
| Preprocess 1 mẫu thật | **QUA**: `(3, 224, 224)` `torch.float32`, min `-1.7923`, max `2.1459` | `clip.load("ViT-B/16", device="cpu", jit=False).preprocess` (đúng lời gọi ở `workbench/replay/tgcir.py:223`); `Normalize(mean=(0.48145466,0.4578275,0.40821073), std=(0.26862954,0.26130258,0.27577711))`; `input_resolution 224`; `EXIT=0` (`raw/compose_probe/preprocess_tgcir.log`) |
| `Adapter.status()` chưa export biến | `ready=False`; lý do đầu tiên = "no interpreter environment resolved for WORKBENCH_TGCIR_PYTHON" | `raw/compose_probe/status_default_env.log` |
| `Adapter.status()` đã export biến | `ready=False`; chỉ còn "test.py không có CLI" + "CUDA-hard" | `raw/compose_probe/status_with_envvars.log` |
| Cổng môi trường | **DỪNG**: `RuntimeError` `BLOCKED/DEPENDENCY_VERSION_MISMATCH`, `torch: expected ==1.7.0, found 2.7.1+cu128` | `model_tgcir.py:175` → `adapters.py:346` → `runtime.py:2309`; `raw/compose_probe/adapter_run_refusal.log` |
| Encoder | **KHÔNG TỒN TẠI**, `raise Refused` vô điều kiện | `model_tgcir.py:177`; không có `subprocess` nào trong module (chỉ `:164`, `:196` docstring) |
| Đường vào thật của runner | `SELECTED_BUT_BLOCKED`, `EXIT=1` | `composition_eval.py:368-372`; `raw/compose_probe/composition_eval_no_plan.log` |
| Preflight đường replay (đối chứng) | **QUA**: `[READY] … no inference executed`, `EXIT=0` | `tgcir39-la -m workbench.replay.tgcir … --preflight-only` (`raw/compose_probe/preflight_tgcir.log`) |

---

## 3. Chặn thật vs. Chặn do môi trường

### 3.1 Chặn thật (không sửa môi trường nào cứu được)

1. **Hai adapter composition không có encoder.** `_run()` trong cả hai module kết thúc bằng
   `raise Refused(...)` **vô điều kiện** (`model_clvc_net.py:205`, `model_tgcir.py:177`), sau khi đã
   qua mọi cổng. Không module nào có `subprocess`, không gọi `workbench.replay.*`, không nạp
   checkpoint, không dựng transform. Đây là lý do **thật** khiến "adapter luôn từ chối".
2. **Bằng chứng phản chứng (stub cổng, không sửa file nào).** Tiêm `adapted_runtime` (khối `sm_120`) **trong bộ nhớ** vào
   record registry + `WORKBENCH_RUNTIME_ADAPTATION=sm_120` ⇒ `verify_environment` trả
   `GPU_VERIFICATION_PENDING`, tier này **nằm trong** `READY_TIERS`
   (`workbench/backend/runtime.py:1867`) — tức cổng `require_verified_model_interpreter` mà hai
   adapter gọi sẽ **cho qua** (`raw/compose_probe/adapted_runtime_simulation.log`). Khi cổng đó được
   stub cho thoả, `encode_queries` vẫn dừng đúng tại `raise Refused`
   (`raw/compose_probe/adapter_terminal_refusal.log`, script `compose_probe_gate_stubbed.py`:
   `model_clvc_net.py:205`, `model_tgcir.py:177`).
   Hai log đó cộng lại là chứng minh: **pin stack không mở được composition**.
3. **TG-CIR:** registry ghi `command_status: COMMAND_BLOCKED_SOURCE_ENTRYPOINT`
   (`workbench/registry/models.yaml:1282`) — đúng với `third_party/TGCIR/test.py` (không `__main__`,
   không `argparse`). Nhưng `workbench/replay/tgcir.py` **đã** dựng đường eval thay thế và chạy được
   (preflight `EXIT=0`), nên đây là chặn **thiết kế adapter**, không phải chặn khả thi.
4. **CLVC:** cáo buộc "unguarded CUDA" là đúng (`cuda_census` đo 14 dòng/15 biểu thức toàn cây,
   6 dòng/7 biểu thức trên đường tính điểm) nhưng **không phải** điều kiện tiên quyết: đường replay
   đã có `_cpu_device_patch` (`workbench/replay/clvc_net.py:137`, áp ở `:169`) và preflight `EXIT=0`.
   Nên CUDA cũng không phải chặn ở tầng composition.

### 3.2 Chặn do môi trường (sửa được, và sửa **không đủ**)

| # | chặn | bằng chứng | sửa bằng gì |
|---|---|---|---|
| E1 | `WORKBENCH_CLVC_PYTHON` / `WORKBENCH_TGCIR_PYTHON` **không có** trong `workbench/config/workbench.env`; không có `workbench/artifacts/environment_assignment.json` | `.example:35,38` có, file thật không; `ls workbench/artifacts/environment_assignment.json` → `No such file` | export biến, hoặc ghi `environment_assignment.json` (RUNTIME_VERIFIED) |
| E2 | Registry pin `python 3.7.6 / torch 1.6.0 / cuda 10.2` (CLVC, `models.yaml:928-935`) và `torch==1.7.0 / tv 0.8.0 / cuda 11.0` (TG-CIR, `:1287-1288,1293`); máy có 3.9.25 / 2.7.1+cu128 | `raw/compose_probe/env_verify.log`: `BLOCKED/PYTHON_VERSION_MISMATCH` và `BLOCKED/DEPENDENCY_VERSION_MISMATCH` | thêm khối `adapted_runtime` (sm_120) vào registry cho hai model — 9 model khác đã có (vd `conesep`, `models.yaml:233`) |
| E3 | GPU capability phải được verify trước khi **launch** bất kỳ tiến trình model nào | `adapted_runtime_simulation.log` → tier `GPU_VERIFICATION_PENDING` (nằm trong `READY_TIERS` nhưng **không** trong `LAUNCH_READY_TIERS`, `runtime.py:1867,1870`) | `verify_gpu_runtime()` — cần GPU. Hiện **chưa bind** vì hai adapter này không launch gì cả; chỉ bind khi có encoder thật |

**Kết luận về giả thuyết "unpinned interpreter stack":**
**XÁC NHẬN một phần, BÁC BỎ ở tư cách nguyên nhân gốc.** Nó là exception **đầu tiên** được ném ra
(`PYTHON_VERSION_MISMATCH` / `DEPENDENCY_VERSION_MISMATCH`), nhưng khi cổng đó được thoả (mô phỏng
in-memory, tier `GPU_VERIFICATION_PENDING ∈ READY_TIERS`), hai adapter **vẫn** dừng ở
`raise Refused` vì **không có encoder**. Ghi chú "R16(b) — thiếu interpreter stack pin" trong
`REMAINING_WORK.md:43-44` mô tả đúng *triệu chứng đầu tiên*, không đúng *nguyên nhân*.

Ghi chú phạm vi: `composition_eval --models clvc_net,tgcir --plan` **không** cấu hình
(`EXIT=0`) loại cả hai ở tầng checkpoint-policy (`strict_clean`, không variant nào khai điều kiện
huấn luyện) — với `--config …/circus_cohort.yaml` (`include_flagged_conditions`) thì qua tầng đó và
dừng ở `SELECTED_BUT_BLOCKED` của adapter. Hai tầng này độc lập; xem
`raw/compose_probe/composition_plan.log` vs `composition_plan_with_policy.log`.

**CSMCIR: composition vẫn DEFERRED** — ngoài phạm vi báo cáo này (không đo, không kết luận).

---

## 4. Lệnh tiếp theo

Việc phải làm là **viết encoder** cho hai adapter (uỷ quyền cho `workbench/replay/*`, đường đã chứng
minh chạy được), không phải pin stack. Lệnh dưới đây là bước đo **nhỏ nhất** tạo ra dữ liệu mà
encoder đó cần, và chứng minh đường uỷ quyền chạy được trên một category — CPU, batch 1. Thời gian CPU
của một category **chưa được đo** trong báo cáo này; `raw/clvc_forward_gpu.log` và
`raw/tgcir_forward_gpu.log` chỉ ghi peak VRAM (419 MiB / 950 MiB) của lượt GPU, không ghi wall time.

CLVC-Net (CPU, 1 category):

```bash
/home/heheboiz/data/cir/workbench/envs/clvc39-la/bin/python -m workbench.replay.clvc_net \
  --checkpoint-root /home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700/checkpoints/clvc \
  --dataset-root /home/heheboiz/data/cir/data/FashionIQ \
  --source-root /home/heheboiz/data/cir/workbench/third_party/CLVCNet \
  --category dress --device cpu --batch-size 1 --output /tmp/clvc_dress_cpu.json
```

TG-CIR (CPU, 1 category):

```bash
/home/heheboiz/data/cir/workbench/envs/tgcir39-la/bin/python -m workbench.replay.tgcir \
  --checkpoint-root /home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700/downloads/tgcir/x/checkpoints \
  --dataset-root /home/heheboiz/data/cir/data/FashionIQ \
  --source-root /home/heheboiz/data/cir/workbench/third_party/TGCIR \
  --category dress --device cpu --batch-size 1 --output /tmp/tgcir_dress_cpu.json
```

Sau đó, để tái lập đúng chỗ dừng hiện tại (2 s, không cần GPU):

```bash
WORKBENCH_CLVC_PYTHON=/home/heheboiz/data/cir/workbench/envs/clvc39-la/bin/python \
WORKBENCH_TGCIR_PYTHON=/home/heheboiz/data/cir/workbench/envs/tgcir39-la/bin/python \
/home/heheboiz/data/cir/workbench/envs/ctl312h/bin/python -m workbench.scripts.composition_eval \
  --config workbench/docs/reviews/20261108T113610-eval/circus_cohort.yaml \
  --models clvc_net,tgcir --smoke 1
# kỳ vọng: EXIT=1, cả hai SELECTED_BUT_BLOCKED
```

Điều kiện để lệnh thứ ba có thể trả `COMPLETED`: `encode_queries`/`encode_targets` trong
`model_clvc_net.py` và `model_tgcir.py` phải gọi `workbench.replay.*` (hoặc tương đương) thay cho
`raise Refused`, **và** cả hai model phải có khối `adapted_runtime: sm_120` trong registry **và**
GPU capability được verify.

---

## 5. Những gì KHÔNG đo (NOT_VERIFIED)

- **NOT_VERIFIED** — một query thật đi hết đường replay trên CPU (`--device cpu`). Chưa chạy; hai lệnh
  ở §4 là bước kế tiếp, không phải kết quả.
- **NOT_VERIFIED** — wall time CPU của một category cho cả hai model.
- **NOT_VERIFIED** — `verify_gpu_runtime()` trả tier nào trên host này (chỉ gọi
  `torch.cuda.is_available()` = `True`, capability `(12, 0)`; không chạy kernel nào).
- **NOT_VERIFIED** — hành vi composition khi registry **thật** có khối `adapted_runtime`: khối đó chỉ
  được tiêm **trong bộ nhớ** (`compose_probe_adapted_simulation.py`); không file registry nào bị sửa.
- **NOT_VERIFIED** — lý do khoa học của việc hai adapter cố ý không có encoder; báo cáo này chỉ ghi
  quan sát rằng chúng không có.
- **CSMCIR: composition DEFERRED.** Không đo, không kết luận.

