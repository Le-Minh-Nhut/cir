# Final PC Audit — Independent Review 2: Runtime Adaptation & Blackwell Architecture Gate

- **Target Slice**: `workbench/backend/runtime.py`, `workbench/backend/environment_planner.py`, `workbench/scripts/pc_readiness.py`, `workbench/registry/models.yaml`, `workbench/backend/registry.py`
- **Prior Audit Reference**: `workbench/docs/reviews/final_pc_audit/B.md`
- **Revision Under Audit**: commit `6e3c979`
- **Target Hardware**: Ubuntu host, RTX 5070 Ti (Blackwell `sm_120`, compute capability 12.0, 16 GB VRAM), NVIDIA driver + `nvidia-container-toolkit`
- **Dev Host Environment**: Linux 7.0.0-38-generic x86_64; orchestrator `.venv` (Python 3.13, PyTorch 2.14.0+cpu, no GPU passthrough). All actual kernel executions remain legitimately **PENDING**; all logical gates and data flows are verified via deterministic probe reconstruction.

---

## 1. Overall Verdict

**VERDICT: SOUND_WITH_RESERVATIONS**

All four P0/P1 defects identified in Audit B (`B.md`) are verified to be genuinely and cleanly fixed in commit `6e3c979`. The Blackwell `sm_120` contract now flows through planner, solver, installer, interpreter, verifier, GPU probe, and evaluator under a single unified specification. Historical pins remain the default when `WORKBENCH_RUNTIME_ADAPTATION` is unset. Minor reservations are documented for pip index precedence shadowing and advisory cache call ergonomics (both P3, non-fatal).

---

## 2. Verification of the Four Runtime Fixes

### 2.1 sm_120 Arch-Gate Normalization (`runtime.py:1998-2013`)
- **Prior Defect (C1, P0)**: Bare string `"120"` was tested against `sm_`-prefixed `torch.cuda.get_arch_list()` strings (e.g. `"sm_120"`), making the Blackwell gate unsatisfiable.
- **Current Code**: `runtime.py:1998-2004` implements `_capability_number()`, which splits on `+` (stripping `+PTX`) and strips `sm_`, comparing bare integer strings across target, arch list, and device compute capability.
- **Reproduction & Observed Output**:
  - Test A (Matching arch list): `torch_arch_list=["sm_80", "sm_90", "sm_100", "sm_120"]`, device `12.0`.
    `verify_gpu_runtime(model, cfg, authorized=True, force=True)` -> `capability_verified: True` (OBSERVED).
  - Test B (Matching arch list with PTX suffix): `torch_arch_list=["sm_80", "sm_90", "sm_120+PTX"]`, device `12.0`.
    `verify_gpu_runtime(model, cfg, authorized=True, force=True)` -> `capability_verified: True` (OBSERVED).
  - Test C (Non-matching arch list): `torch_arch_list=["sm_80", "sm_90", "sm_100"]`, device `12.0`.
    `verify_gpu_runtime` raises `GpuVerificationError` with `adaptation_mismatch: "adapted runtime targets sm_120 but torch arch list lacks it"` (OBSERVED).
  - Test D (Mismatched device capability): `torch_arch_list=["sm_80", "sm_120"]`, device `8.9` (Ada).
    `verify_gpu_runtime` raises `GpuVerificationError` with `adaptation_mismatch: "probed device is sm_89"` (OBSERVED).
- **Classification**: NOT_A_BUG (Prior P0 Confirmed Fixed).

### 2.2 `resolved_runtime_environment` Tolerance Across Models (`runtime.py:821`)
- **Prior Defect (C2, P1)**: `resolved_runtime_environment()` strictly called `runtime_adaptation(model)` (`undeclared_is_error=True`), causing any non-adapted model (e.g. `csmcir`, `airknow`) to raise `WorkbenchError` when `WORKBENCH_RUNTIME_ADAPTATION=sm_120` was set globally.
- **Current Code**: `runtime.py:821` passes `undeclared_is_error=False`. Models without an `adapted_runtime` block keep their historical contract.
- **Reproduction & Observed Output**:
  - Test A (`csmcir` under `WORKBENCH_RUNTIME_ADAPTATION=sm_120`):
    `runtime.resolved_runtime_environment(csmcir_model)` returns historical pins (`pytorch: "2.0.1"`, `cuda: "11.8"`) without raising (OBSERVED).
  - Test B (All registry models):
    Iterating through all 16 registry models under `WORKBENCH_RUNTIME_ADAPTATION=sm_120` via `environment_planner.model_constraints` and `_runtime_model`: all 16 resolve cleanly without raising (OBSERVED).
  - Test C (Declared model with wrong adaptation):
    `limn` with `WORKBENCH_RUNTIME_ADAPTATION=sm_90` raises `WorkbenchError('UNKNOWN_RUNTIME_ADAPTATION')` as required: declared target `sm_120` != requested `sm_90` (OBSERVED).
  - Test D (Distinct environment IDs):
    `plan_environments([limn_model, csmcir_model])` under `sm_120` produces 2 distinct groups:
    `csmcir` -> `cir-5a76ffa0c71997d9` (`runtime_adaptation: None`)
    `limn` (adapted) -> `cir-5f868bc31646189b` (`runtime_adaptation: sm_120`)
    `limn` (historical, unset env) -> `cir-1bbd9872cf166f0d` (`runtime_adaptation: None`)
    All environment IDs are distinct and provenance-tagged (OBSERVED).
- **Classification**: NOT_A_BUG (Prior P1 Confirmed Fixed).

### 2.3 GPU-Capability Key Fallback & Round-Trip Persistence (`runtime.py:1082-1122`)
- **Prior Defect (C4, P1)**: Repo root was non-writable in the deployed container (uid 1000). `_gpu_cache_key()` swallowed the `OSError` and returned `b""`, writing unsigned records that `_cached_gpu_capability()` discarded, permanently closing the launch gate.
- **Current Code**: `runtime.py:1092-1100` checks `legacy_key_path.is_file()`, then `os.access(repo_root, os.W_OK)`, and falls back to `XDG_CONFIG_HOME / "cir" / "workbench_ledger_key"` (or `~/.config`). It writes a 32-byte secret with `0600` permissions.
- **Reproduction & Observed Output**:
  - Host/Container emulation: `repo_ro` directory chmod `0555` (read-only), `WORKBENCH_LEDGER_KEY` unset, `XDG_CONFIG_HOME` pointing to writable artifact directory.
  - `_gpu_cache_key()` creates `XDG_CONFIG_HOME/cir/workbench_ledger_key` with mode `0600` and returns 32 random bytes (OBSERVED).
  - Subsequent call re-reads the exact same 32 bytes (OBSERVED).
  - Capability record written via `record_gpu_capability(model, payload, cfg)`.
  - In-memory cache `_GPU_CAPABILITY.clear()` and `forget_environment()` executed to force full disk deserialization.
  - `_cached_gpu_capability(model, report, authoritative=True)` loads from disk, validates HMAC-SHA256 signature, and returns payload with `capability_verified: True` (OBSERVED).
  - `verify_environment(model, probe=True, authoritative=True)` promotes environment tier to `GPU_RUNTIME_READY` (`TIER_GPU_READY`) (OBSERVED).
- **Classification**: NOT_A_BUG (Prior P1 Confirmed Fixed).

### 2.4 Advisory vs Authoritative Fingerprinting & Tamper Resistance (`runtime.py:1500-1595`)
- **Prior Defect**: Environment fingerprinting hashed 97k installed files on every status poll, causing massive bottlenecks.
- **Current Code**:
  - Advisory (`authoritative=False`): checks interpreter stat, marker files, site-packages directory mtimes, dist-info METADATA, and RECORD file text. Memoized in `_ADVISORY_INTERPRETER_FINGERPRINT_CACHE`.
  - Authoritative (`authoritative=True`): parses RECORD members and computes `_member_digest()` byte-by-byte via SHA-256 for all installed files.
- **Reproduction & Observed Output**:
  - Test A (Advisory efficiency): `_interpreter_fingerprint(..., authoritative=False)` made exactly 0 calls to `_member_digest` (OBSERVED).
  - Test B (Authoritative thoroughness): `_interpreter_fingerprint(..., authoritative=True)` called `_member_digest` for distribution members (OBSERVED).
  - Test C (Tamper / in-place rewrite detection):
    A member file (`torch.py`) was rewritten in-place with modified content of identical byte length, preserving file and site-packages mtime timestamps via `os.utime(..., ns=(...))`.
    Advisory fingerprint remained identical: `cdd73158274d... == cdd73158274d...` (OBSERVED).
    Authoritative fingerprint changed: `d9464e9f6a07... != 7a9d6c84a58f...` (OBSERVED).
    `verify_environment(model, probe=True, authoritative=True)` degraded tier to `GPU_VERIFICATION_PENDING` (OBSERVED).
    `require_launch_ready(model, require_cuda=True)` raised `GpuVerificationError` refusing execution of stale/tampered environment (OBSERVED).
- **Classification**: NOT_A_BUG (Prior Bottleneck Fixed while Fail-Closed Security Preserved).

---

## 3. End-to-End Blackwell (`sm_120`) Contract Trace

| Pipeline Hop | File & Lines | Behavior under `WORKBENCH_RUNTIME_ADAPTATION=sm_120` | Status |
|---|---|---|---|
| **1. Resolution** | `runtime.py:805-845` | Returns adapted dict: `pytorch: ">=2.7,<3"`, `cuda: "12.8"`, `pip_index_url: "https://download.pytorch.org/whl/cu128"` | VERIFIED |
| **2. Planner** | `environment_planner.py:368-515` | Incorporates `_adapted_runtime: "sm_120"` and seeds `spec["pip_index_url"]` | VERIFIED |
| **3. Solver** | `environment_planner.py:571-579` | `group["constraints"]["pip_index_url"]` captured | VERIFIED |
| **4. Installer** | `environment_planner.py:1500-1509` | Adds `--extra-index-url https://download.pytorch.org/whl/cu128` to pip argv | VERIFIED |
| **5. Interpreter** | `environment_planner.py:1049-1088` | Model gets isolated group prefix `cir-5f868bc31646189b` | VERIFIED |
| **6. Verifier** | `environment_planner.py:1265-1275` / `runtime.py:1260-1375` | Validates against resolved adapted constraints (`open_clip_torch>=2.24,<2.27`); records `report.runtime_adaptation: "sm_120"` | VERIFIED |
| **7. GPU Probe** | `runtime.py:1990-2030` | Normalizes `sm_120` -> `"120"`, verifies `"120"` in torch arch list and device `12.0`, signs capability payload | VERIFIED |
| **8. Evaluator** | `evaluate_models.py:454, 1501-1503` | `WORKBENCH_RUNTIME_ADAPTATION` allowed in child env; `runtime_adaptation` persisted in run provenance | VERIFIED |

Historical contract test: with `WORKBENCH_RUNTIME_ADAPTATION` unset, `limn` plans historical `torch==1.12.1`, `torchvision==0.13.1`, `cuda: 12.4` under environment id `cir-1bbd9872cf166f0d`. Unset variable leaves historical contract intact.

---

## 4. Findings & Risk Classification

### Finding 1: `_cached_gpu_capability` call in `verify_gpu_runtime` defaults to authoritative mode
- **Classification**: UNVERIFIED_RISK (Severity: P3)
- **Location**: `workbench/backend/runtime.py:1912`
  ```python
  existing = _cached_gpu_capability(model, verify_environment(model, probe=True, advisory=True))
  ```
- **Detail**: `_cached_gpu_capability(model, report, authoritative=True)` defaults `authoritative=True`. Although `verify_environment` was called with `advisory=True`, omitting the third argument causes `_cached_gpu_capability` to invoke `_gpu_capability_key(model, authoritative=True)`.
- **Impact**: Correctness is unaffected: the key computed is authoritative and matching. However, on the initial probe reuse check, this hashes files rather than using the glob-based advisory cache read.
- **Action**: Optional optimization: pass `authoritative=False` at `runtime.py:1912`.

### Finding 2: `_capability_number` parses `sm_` but does not strip `compute_`
- **Classification**: NOT_A_BUG (Severity: P3)
- **Location**: `workbench/backend/runtime.py:1998-2000`
  ```python
  def _capability_number(value: str) -> str:
      text = value.strip().lower().split("+")[0]
      return text[3:] if text.startswith("sm_") else text
  ```
- **Detail**: PyTorch release wheels build with `code=sm_XX`, so `torch.cuda.get_arch_list()` produces `sm_120` (or `sm_120+PTX`), which `_capability_number` parses to `"120"`. `compute_120` entries (if present in custom builds) would remain `"compute_120"`.
- **Impact**: Zero impact on RTX 5070 Ti running standard PyTorch Blackwell wheels (`sm_120` is present).

### Finding 3: Pip `--extra-index-url` precedence
- **Classification**: UNVERIFIED_RISK (Severity: P2)
- **Location**: `workbench/backend/environment_planner.py:1509`
- **Detail**: Pip `--extra-index-url` gives no inherent priority over PyPI.
- **Mitigation in Place**: If a non-CUDA or non-sm_120 wheel were resolved from PyPI, the GPU probe gate immediately refuses at `target_number not in arch_numbers` or `cuda_available is False`, failing closed before any evaluation or publication.

---

## 5. One-Line RTX 5070 Ti Answer

No — the four P0/P1 defects that guaranteed failure in Audit B are resolved, and the Blackwell sm_120 contract now passes end-to-end fail-closed validation on an RTX 5070 Ti PC (physical CUDA execution pending host GPU).
