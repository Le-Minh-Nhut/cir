# Comprehensive RTX 5070 Ti / Blackwell sm_120 Compatibility Review

**Document Status:** STATIC SOURCE AND REGISTRY AUDIT  
**Target Hardware Evaluated:** NVIDIA GeForce RTX 5070 Ti (Blackwell Architecture, Compute Capability `sm_120`)  
**Repository HEAD:** `b70d092`  
**Disclaimer:** `[INFERENCE]` — Hardware validation was **NOT** performed. There is no RTX 5070 Ti or Blackwell GPU present on this audit workstation (`linux 7.0.0-38-generic`, CPU environment). Every compatibility assessment, kernel availability assertion, and failure mode described in this report is an explicit analytical inference derived from declared dependency constraints, committed upstream source code, NVIDIA CUDA architecture compatibility rules, and PyTorch release specifications.

---

## 1. Executive Summary & Hardware Architectural Context

### 1.1 NVIDIA Blackwell `sm_120` Architectural Requirements
- **Architecture:** NVIDIA Blackwell (GeForce RTX 50-series desktop GPUs: RTX 5070, 5070 Ti, 5080, 5090; compute capability `sm_120`).
- **Minimum CUDA Toolkit:** CUDA Toolkit **12.8+** is strictly required to compile or run native machine code for `sm_120`.
- **Minimum PyTorch Binary:** Official PyTorch prebuilt binary wheels shipping native `sm_120` CUDA kernels start at **PyTorch 2.7+** (specifically wheels built against CUDA 12.8, e.g., index `https://download.pytorch.org/whl/cu128`).
- **Fundamental Incompatibility with Historical Stacks:** All historical PyTorch releases between **PyTorch 1.4.0 and PyTorch 2.1.0** (released 2020–2023) were compiled with CUDA 10.2, 11.x, or 12.1. These prebuilt wheels contain binary kernels capping at `sm_80` (Ampere), `sm_86`/`sm_89` (Ada Lovelace), or `sm_90` (Hopper). They contain **ZERO binary kernels for `sm_120`**, and their embedded PTX bytecode cannot forward-JIT on Blackwell drivers without CUDA 12.8 support.
- **Hardware Consequence:** Attempting to execute any model using its historical PyTorch binary (<= 2.1.0) on an RTX 5070 Ti will unconditionally terminate at the very first CUDA kernel launch with:
  `RuntimeError: CUDA error: no kernel image is available for execution on the device` (CUDA error 800).
- **Core Conclusion:** **Zero models can run on RTX 5070 Ti without an environment adaptation.** Native hardware execution on `sm_120` strictly requires adapting the Python/PyTorch runtime to PyTorch >= 2.7 and CUDA >= 12.8.

---

## 2. Orchestrator vs Model Container Stack

The workbench container architecture separates the orchestrator service from individual model execution environments.

### 2.1 Orchestrator Stack
Defined in `workbench/docker/Dockerfile:46-50`:
```dockerfile
# The workbench backend needs: fastapi, uvicorn, duckdb, pyyaml, packaging, numpy, pillow,
# torch/torchvision (CPU wheels suffice for orchestration; model environments are separate).
# Pinned to what the repository is developed against; update with care.
RUN python -m pip install --no-cache-dir \
    "fastapi==0.142.2" "uvicorn==0.54.0" "duckdb==1.5.6" "pyyaml==6.0.3" \
    "pydantic==2.13.5" "packaging==26.3" numpy pillow \
    "torch==2.14.0" "torchvision==0.29.0"
```
Base image: `python:3.13-slim-bookworm` (`workbench/docker/Dockerfile:7, 18`).

**Role of Orchestrator:**
The container orchestrator runs under Python 3.13 with CPU-only wheels (`torch==2.14.0`, `torchvision==0.29.0`). It owns:
1. HTTP API and Web UI service (`fastapi`, `uvicorn`, `vite preview`).
2. Run database and SQLite/DuckDB ledgers (`duckdb==1.5.6`).
3. Model environment planning, verification, and subprocess dispatch (`micromamba` solver at `/usr/local/bin/micromamba`).
4. Result evaluation parsing, aggregate metrics computation, and ledger signing.

The orchestrator **NEVER executes model GPU inference directly inside its own Python process**.

### 2.2 Model Environments
Model evaluation commands run inside dedicated, isolated Python environments resolved by `workbench/backend/environment_planner.py` and executed via `WORKBENCH_<MODEL>_PYTHON` interpreters or provisioned conda/micromamba prefixes under `/opt/mamba`.

---

## 3. Workbench Adaptation Mechanism

The workbench environment planner (`workbench/backend/environment_planner.py`) implements deterministic planning and runtime adaptation:

### 3.1 How Adaptation is Handled in Code
1. **Binary Pairing Table:** `workbench/backend/environment_planner.py:186-190` extends known PyTorch-to-TorchVision pairings to include modern CUDA 12.8 releases:
   ```python
   ("2.7.0", "0.22.0"), ("2.7.1", "0.22.1"),
   ("2.8.0", "0.23.0"), ("2.9.0", "0.24.0"), ("2.9.1", "0.24.1"),
   ```
2. **Runtime Adaptation Switch:** `workbench/backend/runtime.py:776-802` reads `WORKBENCH_RUNTIME_ADAPTATION` (e.g. `WORKBENCH_RUNTIME_ADAPTATION=sm_120`).
3. **Contract Replacement:** In `workbench/backend/environment_planner.py:371-396` and `471-483`:
   When `WORKBENCH_RUNTIME_ADAPTATION=sm_120` is set, `model_constraints()` checks if the model declares an `adapted_runtime` block in `workbench/registry/models.yaml`.
   - If `adapted_runtime` is declared (matching `sm_120`), the historical pins are replaced with the adapted constraints (`pytorch`, `torchvision`, `cuda`, `pip_index_url`), and the model plans as an **ADAPTED_ENVIRONMENT**.
   - If the model does NOT declare an `adapted_runtime` block, it preserves its historical constraints and plans as a **NATIVE_ENVIRONMENT** (historical contract).
   - If a model has no source and no environment declared (`ptha_mtst`), it cannot plan a GPU execution environment and is treated as **CPU_DIAGNOSTIC_ENVIRONMENT** / unplannable.

### 3.2 Registry Breakdown
In `workbench/registry/models.yaml`:
- **ADAPTED_ENVIRONMENT (1 model):**
  Only `limn` declares an explicit `adapted_runtime` block (`workbench/registry/models.yaml:948-956`):
  ```yaml
      adapted_runtime:
        target_compute_capability: sm_120
        pytorch: '>=2.7,<3'
        torchvision: '>=0.22,<1'
        cuda: '12.8'
        packages: ['open_clip_torch>=2.24,<2.27']
        pip_index_url: https://download.pytorch.org/whl/cu128
  ```
- **NATIVE_ENVIRONMENT (14 models):**
  `csmcir`, `airknow`, `conesep`, `habit`, `intent`, `hint`, `encoder`, `pair`, `clvc_net`, `dcnet`, `combiner_rn50x4_noft`, `clip4cir_rn50x4_fullft`, `tgcir`, `sprc` have `environment` blocks specifying historical pins and do **NOT** declare `adapted_runtime`. When evaluated, the planner plans their historical native environments.
- **CPU_DIAGNOSTIC_ENVIRONMENT (1 model):**
  `ptha_mtst` (`workbench/registry/models.yaml:217-242`) has `source_available: false`, `source_dir: null`, `native_protocol: null`, and no `environment` block. It has no runnable GPU environment.

---

## 4. Per-Tree Declared Stack & Source Code Audit

Across all 14 pinned source trees in `workbench/third_party/`:
- **Custom CUDA C++/CUDA Extensions:** **NONE** (0 `setup.py` files with `CUDAExtension`, 0 `.cu`/`.cuh`/`.cpp` files).
- **Hardcoded Compute Architectures:** **NONE** (no `sm_XX` or `arch=` compiler flags in any tracked source file).
- **Vendored LAVIS annotator utils:** `AirKnow`, `ConeSep`, `HABIT`, `HINT`, `INTENT` vendor OpenMMLab's `parrots_wrapper.py` and `env.py`, which define utility wrappers around `torch.utils.cpp_extension.CUDAExtension`, but these are never compiled or invoked.

Below is the exhaustive audit for each pinned repository.

---

### 1. AirKnow (`workbench/third_party/AirKnow`)
- **Model:** `airknow`
- **Declared Stack:**
  - Files: No `requirements*.txt`, no `environment*.yml`, no `setup.py`.
  - README: `workbench/third_party/AirKnow/README.md:14-20`:
    ```bash
    conda create -n conesep python=3.8
    conda activate conesep
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
    pip install scikit-learn==1.3.2 transformers==4.25.0 salesforce-lavis==1.0.2 timm==0.9.16
    ```
  - Registry: `workbench/registry/models.yaml:57-67`:
    `python: '3.8.10'`, `pytorch: '2.1.0'`, `torchvision: '0.16.0'`, `cuda: '12.6'`, `packages: ['scikit-learn==1.3.2', 'transformers==4.25.0', 'salesforce-lavis==1.0.2', 'timm==0.9.16']`.
- **Custom CUDA Ext?** No.
- **Hardcoded Arch?** No.
- **Code Breaking on Modern Stack:**
  - `torch.load` without `weights_only`: `AirKnow/utils.py:139` (`checkpoint = torch.load(checkpoint)`). Fails on PyTorch 2.6+ default `weights_only=True` when deserializing complex objects. (Note: `AirKnow/cirr_sub_BLIP2.py:125` explicitly passed `weights_only=False`).
  - Deprecated Variable & hardcoded `.cuda()`: `AirKnow/test_BLIP2.py:31, 49` (`imgs = torch.autograd.Variable(imgs).cuda()`), `AirKnow/test_BLIP2.py:111, 128` (`imgs = torch.stack(imgs).float().cuda()`).
- **Workbench-Planned Environment:** `NATIVE_ENVIRONMENT` (`models.yaml:57-67`).
- **Wall vs Adaptation:** **Adaptation**.
- **What Specifically Must Change:**
  1. Python version must be upgraded from 3.8 to >= 3.10 (PyTorch 2.7 drops Python 3.8 support).
  2. Install PyTorch >= 2.7 (cu128) and TorchVision >= 0.22.
  3. `salesforce-lavis==1.0.2` and `transformers==4.25.0`: verify compatibility with newer torch or relax pin.
  4. Pass `weights_only=False` to `torch.load` in `AirKnow/utils.py:139`.
- **Confidence:** **HIGH**. Pure Python/PyTorch codebase; standard BLIP-2 Q-Former model.

---

### 2. CLIP4Cir (`workbench/third_party/CLIP4Cir`)
- **Models:** `combiner_rn50x4_noft`, `clip4cir_rn50x4_fullft`
- **Declared Stack:**
  - Files: No `requirements*.txt`, no `environment*.yml`, no `setup.py`.
  - README: `workbench/third_party/CLIP4Cir/README.md:14-19`:
    ```bash
    conda create -n clip4cir -y python=3.8
    conda activate clip4cir
    conda install -y -c pytorch pytorch=1.11.0 torchvision=0.12.0
    conda install -y -c anaconda pandas=1.4.2
    pip install comet-ml==3.21.0
    pip install git+https://github.com/openai/CLIP.git
    ```
  - Registry: `workbench/registry/models.yaml:703-713` and `776-786`:
    `python: '3.8'`, `pytorch: '1.11.0'`, `torchvision: '0.12.0'`, `cuda: UNKNOWN`, `packages: ['pandas==1.4.2', 'comet-ml==3.21.0', 'clip']`.
- **Custom CUDA Ext?** No.
- **Hardcoded Arch?** No.
- **Code Breaking on Modern Stack:**
  - `torch.load` without `weights_only`: `CLIP4Cir/src/validate.py:285, 306` (`torch.load(..., map_location=device)`), `CLIP4Cir/src/combiner_train.py:79, 282`, `CLIP4Cir/src/cirr_test_submission.py:191`.
  - Clean device handling: uses `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
- **Workbench-Planned Environment:** `NATIVE_ENVIRONMENT`.
- **Wall vs Adaptation:** **Adaptation**.
- **What Specifically Must Change:**
  1. Upgrade Python 3.8 -> 3.10+.
  2. Install PyTorch >= 2.7 (cu128) and TorchVision >= 0.22.
  3. Pass `weights_only=False` to `torch.load` if loading pickled state dicts containing non-primitive types.
- **Confidence:** **HIGH**. Standard PyTorch MLP combiner and OpenAI CLIP backbone.

---

### 3. CLVCNet (`workbench/third_party/CLVCNet`)
- **Model:** `clvc_net`
- **Declared Stack:**
  - Files: No `requirements*.txt`, no `setup.py`.
  - README: No install instructions (execution commands only).
  - Registry: `workbench/registry/models.yaml:574-583`:
    `python: 3.7.6`, `pytorch: 1.6.0`, `torchvision: UNKNOWN`, `cuda: 10.2`, `packages: [torchvision]`.
- **Custom CUDA Ext?** No.
- **Hardcoded Arch?** No.
- **Code Breaking on Modern Stack:**
  - **FATAL REMOVED API:** `CLVCNet/resnet.py:3`:
    ```python
    from torchvision.models.utils import load_state_dict_from_url
    ```
    `torchvision.models.utils` was deprecated in torchvision 0.12 and **removed** in torchvision 0.15+. Under any modern torchvision, this raises `ModuleNotFoundError: No module named 'torchvision.models.utils'`.
  - Deprecated Variable & hardcoded `.cuda()`:
    - `CLVCNet/text_model.py:96`: `itexts = torch.autograd.Variable(itexts).cuda()`
    - `CLVCNet/text_model.py:108`: `first_hidden = (first_hidden[0].cuda(), first_hidden[1].cuda())`
    - `CLVCNet/train.py:156, 157`: `.cuda()`
  - `torch.load` without `weights_only`: `CLVCNet/utils.py:136`: `checkpoint = torch.load(checkpoint)`.
- **Workbench-Planned Environment:** `NATIVE_ENVIRONMENT`.
- **Wall vs Adaptation:** **Code Adaptation**.
- **What Specifically Must Change:**
  1. Source edit: change `CLVCNet/resnet.py:3` to `from torch.hub import load_state_dict_from_url`.
  2. Upgrade Python 3.7.6 -> 3.10+.
  3. Upgrade PyTorch 1.6.0 -> >= 2.7 (cu128), TorchVision -> >= 0.22.
  4. Pass `weights_only=False` to `torch.load`.
- **Confidence:** **HIGH**. Identified exact import removal and replacement.

---

### 4. CSMCIR (`workbench/third_party/CSMCIR`)
- **Model:** `csmcir`
- **Declared Stack:**
  - Requirements: `workbench/third_party/CSMCIR/requirements.txt`:
    `pandas==2.2.0`, `tqdm==4.66.1`, `omegaconf==2.3.0`, `iopath==0.1.10`, `timm==0.9.12`, `decord==0.6.0`, `webdataset==0.2.86`, `opencv-python`, `transformers==4.36.2`, `fairscale==0.4.13`, `einops==0.7.0`, `scikit-image==0.22.0`, `spacy==3.7.2`, `pycocoevalcap==1.2`.
  - README: `workbench/third_party/CSMCIR/README.md:12-14`:
    ```bash
    conda create -n csmcir -y python=3.9
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
    pip install -r requirements.txt
    ```
  - Registry: `workbench/registry/models.yaml:23-33`:
    `python: 3.9`, `pytorch: 2.0.1`, `torchvision: 0.15.2`, `cuda: 11.7`, `packages: []`.
- **Custom CUDA Ext?** No.
- **Hardcoded Arch?** No.
- **Code Breaking on Modern Stack:**
  - `torch.load` without `weights_only`: `CSMCIR/src/validate_blip_csmcir.py:524, 570`, `CSMCIR/src/blip_validate.py:48, 107`, `CSMCIR/src/cirr_test_submission_csmcir.py:215`.
  - Device hardcoding: `CSMCIR/src/lavis/common/dist_utils.py:72` (`torch.cuda.set_device(args.gpu)`), `CSMCIR/src/lavis/datasets/data_utils.py:71` (`return tensor.cuda()`).
  - Note: Upstream source bug in checkout: `CSMCIR/src/lavis/models/blip2_models` contains missing/broken module imports (C1 audit finding).
- **Workbench-Planned Environment:** `NATIVE_ENVIRONMENT`.
- **Wall vs Adaptation:** **Adaptation** (for sm_120; upstream checkout has orthogonal import bugs).
- **What Specifically Must Change:**
  1. Upgrade Python 3.9 -> 3.10+.
  2. Upgrade PyTorch 2.0.1 -> >= 2.7 (cu128), TorchVision 0.15.2 -> >= 0.22.
  3. Set `weights_only=False` in checkpoint loads.
- **Confidence:** **HIGH** on sm_120 inference; **MEDIUM** on upstream package integrity.

---

### 5. ConeSep (`workbench/third_party/ConeSep`)
- **Model:** `conesep`
- **Declared Stack:**
  - Files: No `requirements*.txt`, no `setup.py`.
  - README: `workbench/third_party/ConeSep/README.md:14-20`:
    ```bash
    conda create -n conesep python=3.8
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
    pip install scikit-learn==1.3.2 transformers==4.25.0 salesforce-lavis==1.0.2 timm==0.9.16
    ```
  - Registry: `workbench/registry/models.yaml:127-137`:
    `python: 3.8.10`, `pytorch: 2.1.0`, `torchvision: 0.16.0`, `cuda: 12.6`, `packages: [scikit-learn==1.3.2, transformers==4.25.0, salesforce-lavis==1.0.2, timm==0.9.16]`.
- **Custom CUDA Ext?** No.
- **Hardcoded Arch?** No.
- **Code Breaking on Modern Stack:**
  - `torch.load` without `weights_only`: `ConeSep/cirr_sub.py:124`, `ConeSep/utils.py:139`.
  - Hardcoded `.cuda()`: `ConeSep/cirr_sub.py:37, 55`, `ConeSep/test.py:34` (`imgs = torch.stack(imgs).float().cuda()`).
- **Workbench-Planned Environment:** `NATIVE_ENVIRONMENT`.
- **Wall vs Adaptation:** **Adaptation**.
- **What Specifically Must Change:**
  1. Upgrade Python 3.8 -> 3.10+.
  2. Upgrade PyTorch 2.1.0 -> >= 2.7 (cu128), TorchVision 0.16.0 -> >= 0.22.
  3. Set `weights_only=False` on `torch.load`.
- **Confidence:** **HIGH**. Standard BLIP-2 architecture.

---

### 6. DCNet (`workbench/third_party/DCNet`)
- **Model:** `dcnet`
- **Declared Stack:**
  - Requirements: `workbench/third_party/DCNet/requirement.txt`:
    `torch==1.4.0`, `torchvision==0.5.0`, `nltk==3.5`, `spacy==2.3.0`, `einops`, `tqdm`, `pudb`, `ipdb`, `pillow`, `h5py`, `pandas`.
  - README: `workbench/third_party/DCNet/README.md:14`: `pip install -r requirement.txt`.
  - Registry: `workbench/registry/models.yaml:639-648`:
    `python: 3.7.7`, `pytorch: 1.4.0`, `torchvision: 0.5.0`, `cuda: UNKNOWN`, `packages: [nltk==3.5, spacy==2.3.0]`.
- **Custom CUDA Ext?** No.
- **Hardcoded Arch?** No.
- **Code Breaking on Modern Stack:**
  - **FATAL REMOVED API:** `DCNet/model/fusion.py:63, 64, 68`:
    ```python
    fft1 = torch.rfft(x1.matmul(self.sketch1), signal_ndim = 1)
    fft2 = torch.rfft(x2.matmul(self.sketch2), signal_ndim = 1)
    ...
    cbp = torch.irfft(fft_product, signal_ndim = 1, signal_sizes = (self.output_dim, )) * self.output_dim
    ```
    `torch.rfft` and `torch.irfft` were **completely removed** in PyTorch 1.8.0. Under any modern PyTorch (>= 1.8, including 2.7+ for sm_120), calling `torch.rfft` immediately crashes with `AttributeError: module 'torch' has no attribute 'rfft'`.
  - Deprecated torchvision models argument: `DCNet/model/model.py:283, 287, 291` calls `models.resnet50(pretrained=True)` (deprecated in torchvision >= 0.13, error in newer versions).
  - Deprecated Variable & hardcoded `.cuda()`: `DCNet/model/loss.py:38, 64, 87` (`cost_s = Variable(torch.zeros(1)).cuda()`).
  - `torch.load` without `weights_only`: `DCNet/base/base_trainer.py:195`, `DCNet/model/model.py:299`.
  - Obsolete dependencies: `spacy==2.3.0` cannot compile or install on Python >= 3.10.
- **Workbench-Planned Environment:** `NATIVE_ENVIRONMENT`.
- **Wall vs Adaptation:** **Code Adaptation**.
- **What Specifically Must Change:**
  1. Source edit: Rewrite `CompactBilinearPooling.forward` in `DCNet/model/fusion.py:62-70` to use `torch.fft.rfft` and `torch.fft.irfft` with complex tensor multiplication instead of 2-channel real/imag tensors.
  2. Upgrade Python 3.7.7 -> 3.10+.
  3. Upgrade PyTorch 1.4.0 -> >= 2.7 (cu128), TorchVision 0.5.0 -> >= 0.22.
  4. Upgrade `spacy==2.3.0` -> `spacy>=3.7.0` (with corresponding modern model package).
  5. Update `models.resnet50(pretrained=True)` to modern weights API.
- **Confidence:** **HIGH**. Line-verified removed APIs.

---

### 7. ENCODER (`workbench/third_party/ENCODER`)
- **Model:** `encoder`
- **Declared Stack:**
  - Requirements: `workbench/third_party/ENCODER/requirements.txt`: 211 packages pinned, including `torch==1.13.1`, `torchvision==0.14.1`, `torchaudio==0.13.1`, `open-clip-torch==2.24.0`, `transformers==4.30.2`, `faiss-gpu==1.7.2`, `clip @ git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33`.
  - README: `workbench/third_party/ENCODER/README.md:14-19`:
    ```bash
    conda create -n encoder_env python=3.9
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install -r requirements.txt
    ```
  - Registry: `workbench/registry/models.yaml:444-453`:
    `python: 3.9`, `pytorch: 1.13.1`, `torchvision: 0.14.1`, `cuda: 12.6`, `packages: []`.
- **Custom CUDA Ext?** No.
- **Hardcoded Arch?** No.
- **Code Breaking on Modern Stack:**
  - **Whole-model `torch.load`:** `ENCODER/evaluate_model.py:60`:
    `model = torch.load(args.ckpt_path)` loads a pickled `model_try2.Backbone` instance. Under PyTorch 2.6+ default `weights_only=True`, unpickling crashes with `_pickle.UnpicklingError`. Must pass `weights_only=False`.
  - Constructor `.cuda()` on `nn.Parameter`: `ENCODER/model_try2.py:93`:
    ```python
    self.learnable_relation = nn.Parameter(torch.randn(N_p, dim, dtype=torch.float32)).cuda()
    ```
    Calling `.cuda()` on an `nn.Parameter` inside `__init__` hardcodes device `cuda:0` at module creation.
  - Upstream import bug: `ENCODER/evaluate_model.py:10` imports `datasets1` which is not present in the pinned repository.
- **Workbench-Planned Environment:** `NATIVE_ENVIRONMENT`.
- **Wall vs Adaptation:** **Code Adaptation**.
- **What Specifically Must Change:**
  1. Pass `weights_only=False` to `torch.load` in `ENCODER/evaluate_model.py:60`.
  2. Fix `ENCODER/model_try2.py:93` to avoid `.cuda()` inside parameter initialization.
  3. Fix `ENCODER/evaluate_model.py:10` import (`datasets1` -> `datasets`).
  4. Upgrade Python 3.9 -> 3.10+, PyTorch 1.13.1 -> >= 2.7 (cu128), TorchVision 0.14.1 -> >= 0.22.
  5. Drop `faiss-gpu==1.7.2` from requirements (unused by evaluator; no sm_120 binary for 1.7.2).
- **Confidence:** **HIGH**.

---

### 8. HABIT (`workbench/third_party/HABIT`)
- **Model:** `habit`
- **Declared Stack:**
  - Files: No `requirements*.txt`, no `setup.py`.
  - README: `workbench/third_party/HABIT/README.md:14-20`:
    ```bash
    conda create -n habit python=3.8
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
    pip install open-clip-torch==2.24.0 scikit-learn==1.3.2 transformers==4.25.0 salesforce-lavis==1.0.2 timm==0.9.16
    ```
  - Registry: `workbench/registry/models.yaml:244-251`:
    `python: 3.8.10`, `pytorch: 2.1.0`, `torchvision: 0.16.0`, `cuda: 12.6`, `packages: [open-clip-torch==2.24.0, scikit-learn==1.3.2, transformers==4.25.0, salesforce-lavis==1.0.2, timm==0.9.16]`.
- **Custom CUDA Ext?** No.
- **Hardcoded Arch?** No.
- **Code Breaking on Modern Stack:**
  - `torch.load` without `weights_only`: `HABIT/cirr_test_submission.py:125`, `HABIT/utils.py:139`.
  - Hardcoded `.cuda()` and Variable: `HABIT/train.py:181` (`model.cuda()`), `HABIT/train.py:202, 205` (`img1 = torch.autograd.Variable(img1).cuda()`).
- **Workbench-Planned Environment:** `NATIVE_ENVIRONMENT`.
- **Wall vs Adaptation:** **Adaptation**.
- **What Specifically Must Change:**
  1. Upgrade Python 3.8 -> 3.10+.
  2. Upgrade PyTorch 2.1.0 -> >= 2.7 (cu128), TorchVision 0.16.0 -> >= 0.22.
  3. Ensure `open_clip_torch>=2.24` and `weights_only=False` on model deserialization.
- **Confidence:** **HIGH**.

---

### 9. HINT (`workbench/third_party/HINT`)
- **Model:** `hint`
- **Declared Stack:**
  - Files: No `requirements*.txt`, no `setup.py`.
  - README: `workbench/third_party/HINT/README.md:14-20`:
    ```bash
    conda create -n habit python=3.8
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
    pip install open-clip-torch==2.24.0 scikit-learn==1.3.2 transformers==4.25.0 salesforce-lavis==1.0.2 timm==0.9.16
    ```
  - Registry: `workbench/registry/models.yaml:373-383`:
    `python: 3.8.10`, `pytorch: 2.1.0`, `torchvision: 0.16.0`, `cuda: 12.6`, `packages: [open-clip-torch==2.24.0, scikit-learn==1.3.2, transformers==4.25.0, salesforce-lavis==1.0.2, timm==0.9.16]`.
- **Custom CUDA Ext?** No.
- **Hardcoded Arch?** No.
- **Code Breaking on Modern Stack:**
  - `torch.load` without `weights_only`: `HINT/cirr_test_submission.py:115`, `HINT/utils.py:139`.
  - Hardcoded `.cuda()` and Variable: `HINT/train.py:122` (`model.cuda()`), `HINT/train.py:142, 145` (`img1 = torch.autograd.Variable(img1).cuda()`).
- **Workbench-Planned Environment:** `NATIVE_ENVIRONMENT`.
- **Wall vs Adaptation:** **Adaptation**.
- **What Specifically Must Change:**
  1. Upgrade Python 3.8 -> 3.10+.
  2. Upgrade PyTorch 2.1.0 -> >= 2.7 (cu128), TorchVision 0.16.0 -> >= 0.22.
  3. Set `weights_only=False` on `torch.load`.
- **Confidence:** **HIGH**.

---

### 10. INTENT (`workbench/third_party/INTENT`)
- **Model:** `intent`
- **Declared Stack:**
  - Files: No `requirements*.txt` on disk, no `setup.py`.
  - README: `workbench/third_party/INTENT/README.md:14-21`:
    ```bash
    conda create -n intent_env python=3.9
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
    pip install open-clip-torch==2.24.0 scikit-learn==1.3.2 transformers==4.25.0 salesforce-lavis==1.0.2 timm==0.9.16
    pip install -r requirements.txt
    ```
  - Registry: `workbench/registry/models.yaml:304-314`:
    `python: 3.9`, `pytorch: 2.1.0`, `torchvision: 0.16.0`, `cuda: 12.6`, `packages: [open-clip-torch==2.24.0, scikit-learn==1.3.2, transformers==4.25.0, salesforce-lavis==1.0.2, timm==0.9.16]`.
- **Custom CUDA Ext?** No.
- **Hardcoded Arch?** No.
- **Code Breaking on Modern Stack:**
  - `torch.load` without `weights_only`: `INTENT/cirr_sub_BLIP2.py:128`, `INTENT/utils.py:139`.
  - Hardcoded `.cuda()`: `INTENT/train_INTENT.py:167` (`model.cuda()`), `INTENT/train_INTENT.py:189, 192`.
- **Workbench-Planned Environment:** `NATIVE_ENVIRONMENT`.
- **Wall vs Adaptation:** **Adaptation**.
- **What Specifically Must Change:**
  1. Upgrade Python 3.9 -> 3.10+.
  2. Upgrade PyTorch 2.1.0 -> >= 2.7 (cu128), TorchVision 0.16.0 -> >= 0.22.
  3. Pass `weights_only=False` on `torch.load`.
- **Confidence:** **HIGH**.

---

### 11. LIMN (`workbench/third_party/LIMN`)
- **Model:** `limn`
- **Declared Stack:**
  - Requirements: `workbench/third_party/LIMN/requirements.txt`: 65 packages pinned, including `torch==1.12.1`, `torchvision==0.13.1`, `torchaudio==0.12.1`, `open-clip-torch==2.20.0`, `timm==0.9.2`, `clip==1.0`.
  - README: Execution commands only.
  - Registry: `workbench/registry/models.yaml:935-956`:
    ```yaml
        python_env_var: WORKBENCH_LIMN_PYTHON
        confidence: VERIFIED_README
        python: UNKNOWN
        pytorch: '1.12.1'
        torchvision: '0.13.1'
        cuda: '12.4'
        packages: [open_clip_torch==2.20.0]
        adapted_runtime:
          target_compute_capability: sm_120
          pytorch: '>=2.7,<3'
          torchvision: '>=0.22,<1'
          cuda: '12.8'
          packages: ['open_clip_torch>=2.24,<2.27']
          pip_index_url: https://download.pytorch.org/whl/cu128
    ```
- **Custom CUDA Ext?** No.
- **Hardcoded Arch?** No.
- **Code Breaking on Modern Stack:**
  - Whole-model `torch.load`: `LIMN/LIMN+/train.py:166` (`pre_model = torch.load(...).cuda()`), `LIMN/LIMN/cirr_test_submission.py:118`.
    In PyTorch 2.7+, loading `0_dress_best_model.pt` (a serialized PyTorch 1.12 whole `nn.Module` pickle) crashes under `weights_only=True`. The workbench replay runner (`workbench/replay/common.py:36-48`) already wraps this in `load_trusted_whole_model(..., weights_only=False)` verified by SHA-256.
  - Hardcoded `.cuda()`: `LIMN/LIMN+/train.py:154, 166, 175, 176`.
- **Workbench-Planned Environment:** **`ADAPTED_ENVIRONMENT`** (when `WORKBENCH_RUNTIME_ADAPTATION=sm_120` is selected); `NATIVE_ENVIRONMENT` if unset.
- **Wall vs Adaptation:** **Adaptation** (formally implemented in the workbench).
- **What Specifically Must Change:**
  1. Already codified in `models.yaml`: PyTorch `>=2.7,<3`, TorchVision `>=0.22,<1`, `open_clip_torch>=2.24,<2.27`, CUDA 12.8 wheel index.
  2. Whole-model deserialization requires `weights_only=False` (implemented in workbench replay).
- **Confidence:** **HIGH**.

---

### 12. PAIR (`workbench/third_party/PAIR`)
- **Model:** `pair`
- **Declared Stack:**
  - Files: No `requirements*.txt` on disk, no `setup.py`.
  - README: `workbench/third_party/PAIR/README.md:14-19`:
    ```bash
    conda create -n pair python=3.8.10
    pip install torch==2.0.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install -r requirements.txt
    ```
  - Registry: `workbench/registry/models.yaml:509-518`:
    `python: 3.8.10`, `pytorch: 2.0.0`, `torchvision: UNKNOWN`, `cuda: 11.8`, `packages: []`.
- **Custom CUDA Ext?** No.
- **Hardcoded Arch?** No.
- **Code Breaking on Modern Stack:**
  - `torch.load` without `weights_only`: `PAIR/cirr_test_submission.py:103`: `model = torch.load(os.path.join(model_dir, i))`, `PAIR/utils/utils.py:136`.
  - Hardcoded `.cuda()`: `PAIR/train.py:92` (`PAIR_model.cuda()`), lines 110, 111 (`data['source_img_data'].cuda()`), `PAIR/test.py:32` (`imgs = torch.stack(imgs).float().cuda()`).
- **Workbench-Planned Environment:** `NATIVE_ENVIRONMENT`.
- **Wall vs Adaptation:** **Adaptation**.
- **What Specifically Must Change:**
  1. Upgrade Python 3.8 -> 3.10+.
  2. Upgrade PyTorch 2.0.0 -> >= 2.7 (cu128), TorchVision -> >= 0.22.
  3. Pass `weights_only=False` to `torch.load`.
- **Confidence:** **HIGH**.

---

### 13. SPRC (`workbench/third_party/SPRC`)
- **Model:** `sprc`
- **Declared Stack:**
  - Requirements: `workbench/third_party/SPRC/requirements.txt`:
    `pandas==2.2.0`, `tqdm==4.66.1`, `omegaconf==2.3.0`, `iopath==0.1.10`, `timm==0.9.12`, `decord==0.6.0`, `webdataset==0.2.86`, `opencv-python`, `transformers==4.36.2`, `fairscale==0.4.13`, `einops==0.7.0`, `scikit-image==0.22.0`, `spacy==3.7.2`, `pycocoevalcap==1.2`.
  - README: `workbench/third_party/SPRC/README.md:12-14`:
    ```bash
    conda create -n cir_sprc -y python=3.9
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
    pip install -r requirements.txt
    ```
  - Registry: `workbench/registry/models.yaml:885-895`:
    `python: 3.9`, `pytorch: 2.0.1`, `torchvision: 0.15.2`, `cuda: UNKNOWN`, `packages: [transformers==4.36.2, timm==0.9.12, spacy==3.7.2, salesforce-lavis==1.0.2]`.
- **Custom CUDA Ext?** No.
- **Hardcoded Arch?** No.
- **Code Breaking on Modern Stack:**
  - `torch.load` without `weights_only`: `SPRC/src/validate_blip.py:519`, `SPRC/src/blip_validate.py:48, 107`, `SPRC/src/cirr_test_submission.py:215`.
  - `torch.cuda.amp.autocast`: `SPRC/src/blip_fine_tune_2.py:128, 292` uses `torch.cuda.amp.autocast()` (deprecated in PyTorch 2.4+ in favor of `torch.amp.autocast('cuda')`, but still functional).
- **Workbench-Planned Environment:** `NATIVE_ENVIRONMENT`.
- **Wall vs Adaptation:** **Adaptation**.
- **What Specifically Must Change:**
  1. Upgrade Python 3.9 -> 3.10+.
  2. Upgrade PyTorch 2.0.1 -> >= 2.7 (cu128), TorchVision 0.15.2 -> >= 0.22.
  3. Pass `weights_only=False` to `torch.load`.
- **Confidence:** **HIGH**.

---

### 14. TGCIR (`workbench/third_party/TGCIR`)
- **Model:** `tgcir`
- **Declared Stack:**
  - Files: No `requirements*.txt` file on disk, no `setup.py`.
  - README: `workbench/third_party/TGCIR/README.md:12-21`:
    `clip==0.2.0`, `matplotlib==3.5.1`, `numpy==1.22.3`, `pandas==1.4.2`, `Pillow==9.0.1 / 10.0.0`, `seaborn==0.12.2`, `torch==1.7.0`, `torchvision==0.8.0`, `tqdm==4.65.0`.
  - Registry: `workbench/registry/models.yaml:839-848`:
    `python: UNKNOWN`, `pytorch: 1.7.0`, `torchvision: 0.8.0`, `cuda: 11.0`, `packages: [clip==0.2.0, numpy==1.22.3, pandas==1.4.2, Pillow]`.
- **Custom CUDA Ext?** No.
- **Hardcoded Arch?** No.
- **Code Breaking on Modern Stack:**
  - `torch.load` without `weights_only`: `TGCIR/cirr_test_submission.py:99`, `TGCIR/utils.py:136`.
  - Uninitialized Tensor allocation & hardcoded `.cuda()`: `TGCIR/model.py:32`:
    ```python
    Z = torch.Tensor(B, self.S, C).cuda()
    ```
    Also hardcoded `.cuda()` in `TGCIR/model.py:107, 120, 136, 244, 251` and `TGCIR/train.py:125`.
- **Workbench-Planned Environment:** `NATIVE_ENVIRONMENT`.
- **Wall vs Adaptation:** **Adaptation**.
- **What Specifically Must Change:**
  1. Upgrade PyTorch 1.7.0 -> >= 2.7 (cu128), TorchVision 0.8.0 -> >= 0.22, Python -> 3.10+.
  2. Pass `weights_only=False` to `torch.load`.
  3. Replace uninitialized `torch.Tensor(...).cuda()` with `torch.empty(..., device=device)`.
- **Confidence:** **HIGH**.

---

### 15. PTHA + MTST (No Pinned Source)
- **Model:** `ptha_mtst`
- **Declared Stack:**
  - Repository: `https://github.com/zyinan99/MTST_PTHA` (not checked out; `source_dir: null`, `source_available: false`).
  - Registry: `workbench/registry/models.yaml:217-242`: no `environment` block, `native_protocol: null`.
- **Custom CUDA Ext?** UNKNOWN (no source checked out).
- **Hardcoded Arch?** UNKNOWN.
- **Workbench-Planned Environment:** **`CPU_DIAGNOSTIC_ENVIRONMENT`** / None.
- **Wall vs Adaptation:** **HARD WALL (Upstream Availability Wall)**.
  No source checkout exists in the workbench, no environment is declared, no native protocol is supported, and no fine-tuned FashionIQ checkpoint was ever released by the authors. Cannot be built or run on sm_120 (or any GPU) without an upstream release and integration.
- **Confidence:** **HIGH** regarding unrunnable state in workbench; **LOW** regarding upstream C++ internals.

---

## 5. Master Model Compatibility Matrix

| Model ID | Python Pin | PyTorch Pin | TorchVision Pin | Key Declared Packages | Custom CUDA Ext? | Hardcoded Arch? | Workbench Planned Env | Wall vs Adaptation | What Specifically Must Change | Confidence & Basis |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :--- | :--- | :--- | :--- |
| **`airknow`** | 3.8.10 | 2.1.0 | 0.16.0 | `transformers==4.25.0`, `salesforce-lavis==1.0.2`, `timm==0.9.16` | No | No | `NATIVE_ENVIRONMENT` | Adaptation | Python -> 3.10+; torch -> >=2.7 (cu128); torchvision -> >=0.22; `weights_only=False` in `utils.py:139` | **HIGH** (`README.md:14-20`, verified pure PyTorch ops) |
| **`clip4cir_rn50x4_fullft`** | 3.8 | 1.11.0 | 0.12.0 | `pandas==1.4.2`, `comet-ml==3.21.0`, `clip` (OpenAI) | No | No | `NATIVE_ENVIRONMENT` | Adaptation | Python -> 3.10+; torch -> >=2.7 (cu128); torchvision -> >=0.22; `weights_only=False` in `validate.py:285` | **HIGH** (`README.md:14-19`, clean PyTorch MLP + CLIP) |
| **`clvc_net`** | 3.7.6 | 1.6.0 | UNKNOWN | `torchvision` | No | No | `NATIVE_ENVIRONMENT` | **Code Adaptation** | **Code fix:** `resnet.py:3` change `from torchvision.models.utils` to `from torch.hub`; Python -> 3.10+; torch -> >=2.7 (cu128); torchvision -> >=0.22 | **HIGH** (`resnet.py:3`, exact API removal identified) |
| **`combiner_rn50x4_noft`** | 3.8 | 1.11.0 | 0.12.0 | `pandas==1.4.2`, `comet-ml==3.21.0`, `clip` (OpenAI) | No | No | `NATIVE_ENVIRONMENT` | Adaptation | Python -> 3.10+; torch -> >=2.7 (cu128); torchvision -> >=0.22; `weights_only=False` in `validate.py:285` | **HIGH** (`README.md:14-19`, verified identical to CLIP4Cir) |
| **`conesep`** | 3.8.10 | 2.1.0 | 0.16.0 | `transformers==4.25.0`, `salesforce-lavis==1.0.2`, `timm==0.9.16` | No | No | `NATIVE_ENVIRONMENT` | Adaptation | Python -> 3.10+; torch -> >=2.7 (cu128); torchvision -> >=0.22; `weights_only=False` in `cirr_sub.py:124` | **HIGH** (`README.md:14-20`, verified pure PyTorch ops) |
| **`csmcir`** | 3.9 | 2.0.1 | 0.15.2 | `transformers==4.36.2`, `timm==0.9.12`, `fairscale==0.4.13`, `spacy==3.7.2` | No | No | `NATIVE_ENVIRONMENT` | Adaptation | Python -> 3.10+; torch -> >=2.7 (cu128); torchvision -> >=0.22; `weights_only=False` in `validate_blip_csmcir.py:524` | **MEDIUM** (`requirements.txt`, orthogonal upstream LAVIS bug) |
| **`dcnet`** | 3.7.7 | 1.4.0 | 0.5.0 | `nltk==3.5`, `spacy==2.3.0`, `einops`, `h5py`, `pillow` | No | No | `NATIVE_ENVIRONMENT` | **Code Adaptation** | **Code rewrite:** `model/fusion.py:63-68` rewrite `torch.rfft`/`irfft` to `torch.fft.rfft`; Python -> 3.10+; torch -> >=2.7 (cu128); spacy -> >=3.7 | **HIGH** (`fusion.py:63`, verified removed PyTorch 1.8 API) |
| **`encoder`** | 3.9 | 1.13.1 | 0.14.1 | `open-clip-torch==2.24.0`, `transformers==4.30.2`, `faiss-gpu==1.7.2` | No | No | `NATIVE_ENVIRONMENT` | **Code Adaptation** | **Code fix:** `evaluate_model.py:60` add `weights_only=False`; fix `model_try2.py:93` `.cuda()` in init; fix `datasets1` import; torch -> >=2.7 (cu128) | **HIGH** (`evaluate_model.py:60`, `model_try2.py:93`) |
| **`habit`** | 3.8.10 | 2.1.0 | 0.16.0 | `open-clip-torch==2.24.0`, `transformers==4.25.0`, `salesforce-lavis==1.0.2` | No | No | `NATIVE_ENVIRONMENT` | Adaptation | Python -> 3.10+; torch -> >=2.7 (cu128); torchvision -> >=0.22; `weights_only=False` in `cirr_test_submission.py:125` | **HIGH** (`README.md:14-20`, verified OpenCLIP/LAVIS stack) |
| **`hint`** | 3.8.10 | 2.1.0 | 0.16.0 | `open-clip-torch==2.24.0`, `transformers==4.25.0`, `salesforce-lavis==1.0.2` | No | No | `NATIVE_ENVIRONMENT` | Adaptation | Python -> 3.10+; torch -> >=2.7 (cu128); torchvision -> >=0.22; `weights_only=False` in `cirr_test_submission.py:115` | **HIGH** (`README.md:14-20`, verified OpenCLIP/LAVIS stack) |
| **`intent`** | 3.9 | 2.1.0 | 0.16.0 | `open-clip-torch==2.24.0`, `transformers==4.25.0`, `salesforce-lavis==1.0.2` | No | No | `NATIVE_ENVIRONMENT` | Adaptation | Python -> 3.10+; torch -> >=2.7 (cu128); torchvision -> >=0.22; `weights_only=False` in `cirr_sub_BLIP2.py:128` | **HIGH** (`README.md:14-21`, verified OpenCLIP/LAVIS stack) |
| **`limn`** | UNKNOWN (Hist: 3.10) | 1.12.1 (Adapted: `>=2.7,<3`) | 0.13.1 (Adapted: `>=0.22,<1`) | `open_clip_torch==2.20.0` (Adapted: `>=2.24,<2.27`) | No | No | **`ADAPTED_ENVIRONMENT`** | Adaptation | **Already adapted in workbench registry**: provision with `WORKBENCH_RUNTIME_ADAPTATION=sm_120`, PyTorch cu128 index, and trusted whole-model deserializer | **HIGH** (`models.yaml:948-956`, `workbench/replay/limn.py`) |
| **`pair`** | 3.8.10 | 2.0.0 | UNKNOWN | `torchvision`, `torchaudio` | No | No | `NATIVE_ENVIRONMENT` | Adaptation | Python -> 3.10+; torch -> >=2.7 (cu128); torchvision -> >=0.22; `weights_only=False` in `cirr_test_submission.py:103` | **HIGH** (`README.md:14-19`, verified standard Attention/CLIP) |
| **`ptha_mtst`** | UNKNOWN | UNKNOWN | UNKNOWN | None declared (No source checkout) | UNKNOWN | UNKNOWN | `CPU_DIAGNOSTIC_ENVIRONMENT` | **HARD WALL** | Cannot run on GPU: author repository not checked out, no native protocol, no FashionIQ fine-tuned checkpoint published | **HIGH** (`models.yaml:217-242`, unrunnable state verified) |
| **`sprc`** | 3.9 | 2.0.1 | 0.15.2 | `transformers==4.36.2`, `timm==0.9.12`, `salesforce-lavis==1.0.2` | No | No | `NATIVE_ENVIRONMENT` | Adaptation | Python -> 3.10+; torch -> >=2.7 (cu128); torchvision -> >=0.22; `weights_only=False` in `validate_blip.py:519` | **HIGH** (`requirements.txt`, verified BLIP-2 Q-Former ops) |
| **`tgcir`** | UNKNOWN (Hist: 3.8) | 1.7.0 | 0.8.0 | `clip==0.2.0`, `numpy==1.22.3`, `pandas==1.4.2`, `Pillow` | No | No | `NATIVE_ENVIRONMENT` | Adaptation | Python -> 3.10+; torch -> >=2.7 (cu128); torchvision -> >=0.22; `weights_only=False` in `utils.py:136` | **HIGH** (`README.md:12-21`, verified CLIP + Transformer ops) |

---

## 6. Synthesis: Classification by Remediation Tier

### 6.1 Models That Need NO Adaptation
- **Hardware/CUDA Layer:** **NONE (0 models).**  
  Every single historical stack declares PyTorch <= 2.1.0. None of these binaries contain `sm_120` machine code or compatible PTX. Running any model in its historical environment on an RTX 5070 Ti results in an immediate CUDA initialization error or kernel failure.
- **Source Code Layer (Environment Adaptation Only — No Source Code Diffs Required):**  
  The following **11 models** have clean, forward-compatible PyTorch source code that requires **no internal algorithmic or API rewrites**; they only require environment upgrading (Python >= 3.10, PyTorch >= 2.7 with CUDA 12.8 index, and passing `weights_only=False` via launcher/wrapper):
  1. `combiner_rn50x4_noft`
  2. `clip4cir_rn50x4_fullft`
  3. `limn` *(already formally modeled in workbench as `adapted_runtime: sm_120`)*
  4. `pair`
  5. `tgcir`
  6. `sprc`
  7. `airknow`
  8. `conesep`
  9. `habit`
  10. `hint`
  11. `intent`

### 6.2 Models Needing a Code Adaptation
The following **3 models** contain source code in their pinned trees that is syntactically or semantically broken on a modern PyTorch/TorchVision stack, and will crash even if provisioned with PyTorch >= 2.7:
1. **`clvc_net`:**
   - **Breaking Location:** `workbench/third_party/CLVCNet/resnet.py:3`.
   - **Defect:** `from torchvision.models.utils import load_state_dict_from_url`. Submodule removed in `torchvision>=0.15`.
   - **Fix:** Change import to `from torch.hub import load_state_dict_from_url`.
2. **`dcnet`:**
   - **Breaking Location:** `workbench/third_party/DCNet/model/fusion.py:63, 64, 68`.
   - **Defect:** Calls `torch.rfft(..., signal_ndim=1)` and `torch.irfft(...)`. Both APIs were completely removed in PyTorch 1.8.0.
   - **Fix:** Rewrite `CompactBilinearPooling.forward` using `torch.fft.rfft` and `torch.fft.irfft` with complex tensor multiplication. Also replace `models.resnet50(pretrained=True)` in `DCNet/model/model.py:283`.
3. **`encoder`:**
   - **Breaking Location:** `workbench/third_party/ENCODER/model_try2.py:93` and `evaluate_model.py:10, 60`.
   - **Defect:** Calling `.cuda()` on `nn.Parameter` inside constructor; importing non-existent `datasets1`; unpickling whole model without `weights_only=False`.
   - **Fix:** Remove `.cuda()` from `nn.Parameter` initialization; fix import; pass `weights_only=False`.

### 6.3 Models Whose Declared Stack Cannot Be Rebuilt on `sm_120` Without an Upstream Fix
1. **`ptha_mtst` (Hard Wall):**
   - **Blocker:** No source checkout exists in `workbench/third_party/` (`source_dir: null`), no native protocol exists, and no author FashionIQ checkpoint was ever released. The model is an external paper reference with no executable artifact in the workbench.
2. **`csmcir` (Upstream Source Integrity Blocker):**
   - **Blocker:** Independent of `sm_120`, the pinned source checkout `workbench/third_party/CSMCIR` has broken relative imports and missing vendored files in `src/lavis/models/blip2_models/`, preventing `import lavis` from completing under any environment without upstream source repair.
3. **`dcnet` & `clvc_net` (Strict Historical Environment Rebuild Blocker):**
   - **Blocker:** If constrained strictly to their declared stack contracts (`python: 3.7.6` and `python: 3.7.7`), neither model can be rebuilt on `sm_120`. Python 3.7 reached end-of-life in 2023 and has no PyTorch >= 2.7 wheels. Rebuilding on `sm_120` strictly requires breaking the declared Python 3.7 contract and upgrading to Python 3.10+.
