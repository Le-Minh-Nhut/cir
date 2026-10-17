# FIRST-PARTY RUNTIME, LAUNCHER, CONTAINER & CONTRACT AUDIT

**Target:** First-party runtime, launcher, container, API/frontend contracts end-to-end.  
**Auditor:** `AuditFirstPartyRuntime`  
**Date:** 2026-10-17  
**Commit:** `5726d5f`  
**Standard:** Rigorous source citation (`file:line`), classification (`CONFIRMED_BUG`, `UNVERIFIED_RISK`, `BY_DESIGN`, `REJECTED_FINDING`), verification mode (`STATIC_AUDIT` or `RUNTIME_VERIFIED`).

---

## Executive Summary Matrix

| ID | Category | Status | Severity | File & Line | Summary |
|---|---|---|---|---|---|
| **DEF-01** | PC Readiness | `CONFIRMED_BUG` | P1 | `workbench/scripts/pc_readiness.py:55` | Needle-ordering bug misclassifies missing checkpoint as `BLOCKED_DATASET` for models with "fashioniq" in checkpoint filename (`encoder`). |
| **DEF-02** | Security / Secret | `CONFIRMED_BUG` | P2 | `workbench/backend/runtime.py:1162` vs `workbench/scripts/evaluate_models.py:1216` | Key divergence: `_gpu_cache_key()` prioritizes writable repo-root before private config-home, creating a conflicting second key and triggering `LedgerKeyError`. |
| **DEF-03** | Release Gates | `CONFIRMED_BUG` | P2 | `workbench/docker/release_verify.sh:1016` | `release_verify.sh` exits 0 on `ENGINEERING_INCOMPLETE` and `GPU_VALIDATION_PENDING`, allowing CI/CD to pass without exercising mandatory target execution. |
| **DEF-04** | Release Gates | `BY_DESIGN` / `UNVERIFIED_RISK` | P2 | `workbench/docker/acceptance.sh:308-311` | `acceptance.sh` exits 0 when all live API checks (Groups A, G, C, I) are skipped due to offline server. |
| **DEF-05** | Dataset Protection | `UNVERIFIED_RISK` | P2 | `workbench/docker/docker-compose.yml:48-50` | Dataset root `/data/FashionIQ` and `captions/` mounted `rw`; only `images/` and `image_splits/` are `:ro`. Third-party datasets write unauthenticated `.pkl` caches into root. |
| **DEF-06** | Container / User | `UNVERIFIED_RISK` | P3 | `workbench/docker/Dockerfile:61`, `workbench/docker/docker-compose.yml:22` | Host UID != 1000 has no `/etc/passwd` entry and cannot write to container `$HOME` (`/home/appuser`); relies entirely on `XDG_CONFIG_HOME` redirection. |
| **DEF-07** | Frontend Contract | `UNVERIFIED_RISK` | P3 | `workbench/frontend/evidence.ts:70-72` | `isCertifiedAggregate` does not check `!isSyntheticAggregate`, relying strictly on backend `is_research_evidence` truthfulness. |

---

## 1. Launch Path

**Path Traced:** `pc.sh` -> `docker compose` -> `run.sh` -> `workbench/scripts/pipeline.py` -> `workbench/scripts/evaluate_models.py`.

### 1.1 Environment Variable & Mount Resolution

| Variable / Mount | Boundary | Source (`file:line`) | Unset Check & Refusal Mode | Classification |
|---|---|---|---|---|
| `WORKBENCH_BACKEND_PORT` | Host -> Container | `pc.sh:101-108,139-141`, `run.sh:76-83` | Strictly validated as integer in range `1..65535`. Aborts execution via `fail_stage` with explicit diagnostic. | `REJECTED_FINDING` (Correct) |
| `WORKBENCH_FRONTEND_PORT` | Host -> Container | `pc.sh:101-108,139-141`, `run.sh:76-83` | Validated as integer in range `1..65535`. Aborts via `fail_stage`. | `REJECTED_FINDING` (Correct) |
| `WORKBENCH_ENABLE_GPU` | Host -> Compose | `pc.sh:572-576`, `run.sh:96-97` | Strictly validated against `0\|1`. Refuses start if invalid. Defaults fail-closed to `0` unless `--no-gpu` flag or GPU toolkit present. | `REJECTED_FINDING` (Correct) |
| `WORKBENCH_RUNTIME_ADAPTATION` | Host -> Compose | `pc.sh:549-585` | Extracted from registry `models.yaml`. Passed to container environment. Validated inside container against declared model adaptations. | `REJECTED_FINDING` (Correct) |
| `HOST_*` Paths (5 mounts) | Host -> Docker Compose | `workbench/docker/docker-compose.yml:48-57` | Parameter expansion `${HOST_*:?set HOST_* in .env}` enforced by Compose. Compose halts with non-zero error if any are null/unset. | `REJECTED_FINDING` (Correct) |
| Bind Mount Provenance | Container Runtime | `pc.sh:668-757` | `inspect_container_config` compares `docker inspect {{json .Mounts}}` against expected realpaths for `/data/FashionIQ`, `/checkpoints`, `/artifacts`. Fails closed if missing or mismatched. | `REJECTED_FINDING` (Correct) |

### 1.2 Credential Protection & Boundary Leakage

- **Process Sanitization (`pc.sh:156-166`):** `sanitize_exec()` scrubs all environment variables matching `(TOKEN|KEY|SECRET|LEDGER)` prior to invoking child scripts or container executions (`RUNTIME_VERIFIED`).
- **Queue Child Execution (`workbench/backend/jobs.py:40-70`):** The background queue allowlists only `WORKBENCH_LEDGER_KEY` for the trusted operator process (`_OPERATOR_ENV_ALLOWLIST`), while stripping all cloud provider tokens (`AWS_*`, `GCP_*`, `HF_TOKEN`, `WANDB_API_KEY`, `GITHUB_TOKEN`) and all `WORKBENCH_SECRET_*` variables (`STATIC_AUDIT`).
- **Official Evaluator Isolation (`workbench/backend/jobs.py:67-68`, `workbench/backend/runtime.py:66-70`):** The trusted operator strips `WORKBENCH_LEDGER_KEY` and all `WORKBENCH_SECRET_*` variables from the child environment passed to untrusted model code. Child process runs with private `TMPDIR` (`evaluate_models.py:481-483`).

### 1.3 Key Resolution Divergence (DEF-02)

- **Classification:** `CONFIRMED_BUG` (Severity: P2)
- **Mode:** `STATIC_AUDIT` & `RUNTIME_VERIFIED`
- **Location:** `workbench/backend/runtime.py:1160-1165` vs `workbench/scripts/evaluate_models.py:1209-1224`.
- **Precondition:** Running on a host or development environment where the repository root directory is writable by the process user (`os.access(repo_root, os.W_OK)` is `True`), with no `WORKBENCH_LEDGER_KEY` explicitly exported.
- **Actual Behavior:**
  1. `evaluate_models.py::ledger_hmac_key` initializes first. Because `_config_home_is_usable(config_home)` is `True`, it generates a key at `$XDG_CONFIG_HOME/cir/workbench_ledger_key` (`evaluate_models.py:1216`).
  2. `runtime.py::_gpu_cache_key` runs later. In lines 1160-1165:
     ```python
     if legacy_key_path.is_file():
         key_path = legacy_key_path
     elif os.access(repo_root, os.W_OK):
         key_path = legacy_key_path
     else:
         key_path = private_key_path
     ```
     Because `legacy_key_path` does not exist yet and `repo_root` is writable, it selects `legacy_key_path` (`<repo>/.workbench_ledger_key`), ignores the existing `private_key_path`, and creates a second distinct 32-byte key!
  3. On subsequent invocations, `evaluate_models.py:1205-1208` detects two different key files and raises `LedgerKeyError("two different ledger keys exist...")`, blocking all evaluations and reports.
- **Expected Behavior:** `runtime.py::_gpu_cache_key` should mirror `evaluate_models.py:1216` and prefer `private_key_path.is_file()` before falling back to `legacy_key_path`.
- **Mitigation in Shipped Docker:** In the container, `/repo` is owned by `root` and non-writable by UID 1000, so `os.access(repo_root, os.W_OK)` is `False`, forcing both paths to `$XDG_CONFIG_HOME/cir/workbench_ledger_key`. The defect manifests on host runs and development environments.

---

## 2. Dataset Protection

**Files Traced:** `workbench/docker/docker-compose.yml`, `workbench/backend/adapters/models.py`, `workbench/third_party/**/datasets.py`.

### 2.1 Mount Protection & Separation

- **Observed Configuration (`docker-compose.yml:48-50`):**
  ```yaml
  - ${HOST_FASHIONIQ_ROOT}:/data/FashionIQ
  - ${HOST_FASHIONIQ_ROOT}/images:/data/FashionIQ/images:ro
  - ${HOST_FASHIONIQ_ROOT}/image_splits:/data/FashionIQ/image_splits:ro
  ```
- **Findings:**
  - `/data/FashionIQ/images` is read-only (`:ro`): Validated by mutation check M4 (`mutation_checks.sh:201-206`) where `touch /data/FashionIQ/images/.m4` is rejected (`RUNTIME_VERIFIED`).
  - `/data/FashionIQ/image_splits` is read-only (`:ro`).
  - **DEF-05 (`UNVERIFIED_RISK`):** `/data/FashionIQ` and `/data/FashionIQ/captions` are mounted `rw`. The stated intent (`docker-compose.yml:45-47`) is to permit derived artifacts (resized images, evaluator caches) without touching canonical files (`BY_DESIGN`). However, canonical files under `captions/` (`cap.dress.val.json`, etc.) lack submount protection and are vulnerable to write mutations by un-isolated code.

### 2.2 Native Model `.pkl` Cache Writes

- **Upstream Pattern:** Native third-party code in `AirKnow`, `ConeSep`, `ENCODER`, `HABIT`, `INTENT`, `TGCIR` contains calls to `save_obj(..., os.path.join(self.path, 'test_queries_dress.pkl'))` (`datasets.py`).
- **Workbench Defense:**
  1. `AirKnow`, `ConeSep`, `Habit`, `Intent`, `TGCIR`, `SPRC`, `CLVCNet` adapters do NOT implement `command()` (`workbench/backend/adapters/models.py:24-25`). They raise `NotImplementedError("upstream evaluation command requires audit before execution")`.
  2. `workbench/backend/preflight.py:186-206` refuses to build or launch commands for these models, reporting `command not runnable: COMMAND_UNAUDITED` or `COMMAND_BLOCKED_*`.
  3. `limn` is replayed through `workbench/replay/limn.py`, which is pure read-only and does not write `.pkl` caches.
  4. `encoder` adapter passes `--fashioniq_path /data/FashionIQ/` (`adapters/models.py:41`). If `fashion_iq_data.json` is missing, `ENCODER/datasets.py:83-88` will write `.pkl` files into `/data/FashionIQ`.

---

## 3. Ownership & Permissions

**Files Traced:** `workbench/docker/Dockerfile`, `workbench/docker/docker-compose.yml`, `pc.sh`.

### 3.1 Container vs Host Identity

- **Execution Identity (`docker-compose.yml:22`):** `user: "${WORKBENCH_UID:-1000}:${WORKBENCH_GID:-1000}"`.
- **Host User Alignment (`pc.sh:606-612`):** Detects host `id -u` / `id -g` and sets `WORKBENCH_UID` / `WORKBENCH_GID`.
- **DEF-06 (`UNVERIFIED_RISK`):**
  - In `Dockerfile:61`, only user `appuser` with UID `1000` is provisioned in `/etc/passwd`.
  - When the host operator runs with UID != 1000, container processes run as an unmapped UID.
  - `/home/appuser` is owned by `1000:1000`. An unmapped UID cannot write to `/home/appuser`.
  - The runtime protects itself by redirecting `MAMBA_ROOT_PREFIX=/opt/mamba` (chmod `0777` in `Dockerfile:63`) and `XDG_CONFIG_HOME=/artifacts/config` (`Dockerfile:86`, `docker-compose.yml:40`).

### 3.2 Failure Modes on Read-Only or Root-Owned Artifact Root

If the host directory mounted at `/artifacts` is read-only or owned by `root`:
1. **Index Service (`storage.py:16`, `main.py:187-193`):** Cannot write `/artifacts/index/duckdb.db`. Backend raises `WorkbenchError("index_unavailable")` and serves HTTP 503 on `/api/runs`.
2. **Ledger Key (`evaluate_models.py:1269-1270`, `runtime.py:1188-1189`):** Cannot write `/artifacts/config/cir/workbench_ledger_key`. Serves HTTP 400 `ledger_key_unusable`.
3. **GPU Capability Records (`runtime.py:988-992`):** Cannot write `/artifacts/gpu_capabilities/<key>.json`. Launch gate refuses to promote tier to `TIER_GPU_READY`.
4. **Queue Persistence (`jobs.py`):** Cannot create `/artifacts/queue_jobs/<job_id>/`. Returns HTTP 500 / 409.

---

## 4. GPU & Runtime Integrity

**Files Traced:** `workbench/backend/runtime.py`, `workbench/backend/operator_config.py`, `workbench/scripts/pipeline.py`.

### 4.1 Launch Readiness & Gates

- `require_launch_ready(model)` (`runtime.py:1838-1853`) enforces that models requiring CUDA must reach `TIER_GPU_READY` via an authoritative environment probe. If unverified, raises `GpuVerificationError`.
- API endpoint `POST /api/jobs` (`main.py:1001-1050`) executes `guarded_plan` which invokes `require_launch_ready()`. An unverified GPU returns HTTP 409 `gpu_verification_required` (`STATIC_AUDIT`).

### 4.2 Interpreter Resolution & Environment Variables

- `resolve_model_interpreter(model)` (`runtime.py:1035-1076`):
  1. Reads `model.environment.python_env_var` (e.g. `WORKBENCH_LIMN_PYTHON`).
  2. If unset, reads `/artifacts/environment_assignment.json`.
  3. Verifies that the interpreter path is absolute and contained within `WORKBENCH_ARTIFACTS_ROOT` to prevent path traversal.
  4. Returns `None` if unconfigured, causing `require_verified_model_interpreter` to raise `RuntimeError("execution environment not verified")`.

### 4.3 Stale Capability Re-use Protection

- **Question:** Can a capability record be reused if the underlying machine or driver changes?
- **Finding:** `REJECTED_FINDING` (Airtight Protection).
- **Mechanism:**
  - `_gpu_capability_key()` (`runtime.py:1081-1100`) binds the SHA-256 digest of:
    1. `model_id`
    2. `interpreter` path
    3. `interpreter_fingerprint` (digest of torch binary and python packages)
    4. `contract` (digest including `adapted_runtime`)
    5. `host`: `_host_identity()` (`runtime.py:1000-1033`), which captures `os.uname().nodename`, `os.uname().machine`, `/proc/driver/nvidia/version`, and `CUDA_VISIBLE_DEVICES` / `NVIDIA_VISIBLE_DEVICES`.
  - Authoritative check (`runtime.py:1303`) requires `payload.get("capability_key") == key` and valid HMAC signature.
  - Advisory check (`runtime.py:1255,1274`) requires `payload.get("host_identity") == _host_identity()`.
  - Any hardware, driver, or container device change invalidates the key and forces a fresh probe (`RUNTIME_VERIFIED`).

---

## 5. API Contracts

**Files Traced:** `workbench/backend/main.py`, `workbench/backend/storage.py`, `workbench/backend/analysis.py`.

| Endpoint | Query / Path Params | Failure Modes (400 / 404 / 503) | Empty Response vs Refusal |
|---|---|---|---|
| `/api/runs` | `protocol_id: str \| None` | 503 `index_unavailable` if DuckDB index not built. | Returns `[]` if no runs match `protocol_id`. |
| `/api/benchmark-runs` | None | 400 `ledger_key_unusable` if ledger key missing/conflicting. 400 `ledger_corrupt` if HMAC chain invalid. | Refuses with 400 if ledger damaged. Returns `[]` only on pristine unpopulated workspace. |
| `/api/queries` | `run_id`, `limit`, `offset`, `order`, etc. | 400 `invalid_pagination` if limit not 1..200 or offset < 0. 400 `invalid_query_order` if order not worst/best. 400 `run_not_found` if run_id missing. 503 `index_unavailable`. | Returns `{"items": [], "total": 0}` if filter criteria match 0 rows. Refuses missing run. |
| `/api/compare/{query_id}` | `query_id`, `run_ids`, `k` | 400 `run_not_found` (duplicate or missing runs). 400 `cross_protocol`. 400 `mixed_data_kind`. 400 `insufficient_top_k_depth`. 400 `query_alignment_mismatch`. 400 `query_not_found` if query absent in run. | Refuses invalid selections with specific 400 error codes. |
| `/api/analysis` | `run_ids`, `k` | Same validation as `/api/compare`. 400 for cross-protocol, mixed data kinds, or query alignment mismatch. | Refuses invalid requests. |
| `/api/analysis/failure-jaccard` | `run_ids`, `k` | Validates selection with `require_top_k=False`. Rejects cross-protocol or mismatched queries. | Returns pairwise Jaccard scores. |
| `/api/annotations` | `query_id`, `protocol_id` | PUT `/api/annotations`: 400 `invalid_annotation_labels` if labels not in `LABELS` enum. | GET returns `[]` if no annotations match filters. |
| `/api/cohorts` | None | PUT `/api/cohorts`: 400 `cross_protocol`, 400 `query_alignment_mismatch`. GET `export/{format}`: 404 `cohort_not_found`, 400 `invalid_export_format`. | Refuses invalid cohort operations. |

---

## 6. Frontend Contracts

**Files Traced:** `workbench/frontend/types.ts`, `workbench/frontend/evidence.ts`, `workbench/frontend/gate.ts`, `workbench/frontend/client.ts`, `workbench/frontend/App.tsx`.

### 6.1 Type Projection Completeness

- `Run` (`types.ts:3`) matches `/api/runs` metadata projection (`storage.py:39`).
- `BenchmarkRun` (`gate.ts:47-52`) matches `/api/benchmark-runs` projection (`main.py:850-852`).
- `Query` (`types.ts:2`) matches `/api/queries` response schema (`storage.py:120`).
- `Model` (`types.ts:6`) matches `/api/models` projection including `checkpoints` and `checkpoint_bundles` (`main.py:164-175`).

### 6.2 Evidence Integrity & Truthfulness

- **Warning Badges (`gate.ts:75-108`):** If a run is not certified research evidence (`!isCertifiedEvidence(gate)`), `warningLabels()` attaches explicit non-certified labels: `"CHƯA CÓ PROOF (PROOF UNAVAILABLE)"`, `"TỪ CHỐI CHỨNG NHẬN (REFUSED)"`, `"CHƯA ĐỐI SOÁT LEDGER (UNCORROBORATED)"`, or `"CHƯA CHỨNG NHẬN (UNCERTIFIED)"` (`RUNTIME_VERIFIED`).
- **Gate Evidence Disclosure (`App.tsx:58-61`):** Renders `"(not certified research evidence)"` if `gate?.verification && !isCertified`.
- **Synthetic Run Isolation in Evidence Banner (`evidence.ts:114-118`):**
  If only synthetic aggregate runs exist, banner renders: `"MOCK DEVELOPMENT DATA — N synthetic aggregate runs loaded; no real evaluation results"`.
- **DEF-07 (`UNVERIFIED_RISK`):** `isCertifiedAggregate` (`evidence.ts:70-72`) checks:
  ```typescript
  return run?.verification?.certified === true && run?.verification?.is_research_evidence === true;
  ```
  It does not include `&& !isSyntheticAggregate(run)`. While backend `verify_run.py:620` guarantees `is_research_evidence: False` for synthetic/mock runs, the frontend helper lacks an explicit defensive guard.

---

## 7. PC Readiness & Classification Bug (DEF-01)

**Files Traced:** `workbench/scripts/pc_readiness.py`, `workbench/backend/preflight.py`, `workbench/backend/runtime.py`.

### 7.1 Defect Analysis: Needle-Ordering Bug (DEF-01)

- **Classification:** `CONFIRMED_BUG` (Severity: P1)
- **Mode:** `RUNTIME_VERIFIED`
- **Location:** `workbench/scripts/pc_readiness.py:47-78`
- **Reproduction:**
  Run `python3 workbench/scripts/pc_readiness.py --model encoder --json`.
- **Observed Output:**
  ```json
  {
    "state": "BLOCKED_DATASET",
    "reason": "checkpoint missing: /home/heheboiz/data/cir/workbench/artifacts/checkpoints/encoder/fashioniq.pt"
  }
  ```
- **Root Cause:**
  In `BLOCKER_KINDS` (`pc_readiness.py:47-67`):
  ```python
  BLOCKER_KINDS = (
      ("source cannot run", BLOCKED_SOURCE),
      ...
      ("FashionIQ", BLOCKED_DATASET),                  # Line 55 (index 7)
      ("raw benchmark input missing", BLOCKED_DATASET),
      ...
      ("checkpoint missing", BLOCKED_EXTERNAL_ASSET),  # Line 64 (index 16)
  )
  ```
  In `classify()`:
  ```python
  for needle, state in BLOCKER_KINDS:
      for blocker in blockers:
          if needle.lower() in blocker.lower():
              return state, blocker
  ```
  The checkpoint file for model `encoder` is `fashioniq.pt`. The generated blocker is:
  `checkpoint missing: .../encoder/fashioniq.pt`.
  Because the outer loop scans needles in sequence, it matches `"FashionIQ"` (index 7) before reaching `"checkpoint missing"` (index 16).
  Consequently, `encoder`'s missing checkpoint is reported as `BLOCKED_DATASET` instead of `BLOCKED_EXTERNAL_ASSET`.
- **Expected Behavior:** Specific blocker prefixes (such as `"checkpoint missing"`) must be matched before generic dataset layout tokens like `"FashionIQ"`, or `classify()` should match based on prefix semantics rather than substring containment across the whole path.

---

## 8. Release Gates & Verification Suites

**Files Traced:** `workbench/docker/release_verify.sh`, `workbench/docker/acceptance.sh`, `workbench/docker/mutation_checks.sh`.

### 8.1 Assertion Coverage & Blind Spots

| Script | Journey / Group | What It Actually Asserts | Blind Spots & Can It Pass Without Exercising? | Classification |
|---|---|---|---|---|
| `release_verify.sh` | J1–J12, Dimensions | J1: Clean boot / models served.<br>J2: Mount paths match.<br>J3: Prep writable, images RO.<br>J4: Stdin parse / bundle select.<br>J5: GPU fail closed.<br>J6: sm_120 contract.<br>J7: Complete LIMN macro.<br>J8: Job statuses enum.<br>J9: Paper gate tolerance.<br>J10: Frontend labels.<br>J11: Persistence across restart.<br>J12: Checkpoint failure plan. | **DEF-03 (`CONFIRMED_BUG`):** Lines 998-1016 calculate verdict. If offline or lacking GPU, verdict is `ENGINEERING_INCOMPLETE` or `GPU_VALIDATION_PENDING`. Line 1016 executes `sys.exit(3 if out["verdict"] == "RELEASE_BLOCKED" else 0)`. The script exits with **exit code 0** on `ENGINEERING_INCOMPLETE`! | `CONFIRMED_BUG` (P2) |
| `release_verify.sh` | `MODEL_INFERENCE`, `MODEL_EVALUATION` | Inspects `/artifacts/reports/*_aggregate.json`. Promotes to `PASS` only if corroborated research evidence exists. | Dimensions have empty `checks: []`. They rely entirely on post-hoc artifact scanning rather than driving execution. | `BY_DESIGN` |
| `acceptance.sh` | Groups A–I | A, G, C, I test live API.<br>B, F, D, E, H test source and offline rules. | **DEF-04 (`UNVERIFIED_RISK`):** If live server is down, Groups A, G, C, I are skipped. Lines 308-311 print `ACCEPTANCE PASSED (skipped: ...)` and exit with **code 0**. | `UNVERIFIED_RISK` (P2) |
| `mutation_checks.sh` | M1–M5 | M1: `run.sh` JSON parse mutation.<br>M2: `run.sh` bundle degradation.<br>M3: `paper_gate.py` missing metric mutation.<br>M4: Compose empty path & images RO.<br>M5: Compose GPU reservation. | All 5 mutations are verified killed. If docker daemon is unavailable, M4 skips live mount check and exits 0. | `REJECTED_FINDING` (Robust) |

---

## 9. Recommendations & Remediation Order

1. **Fix DEF-01 in `pc_readiness.py`:** Move `"checkpoint missing"`, `"checkpoint bundle incomplete"`, `"official checkpoint hash mismatch"`, and `"captions"` BEFORE `"FashionIQ"` in `BLOCKER_KINDS`.
2. **Fix DEF-02 in `runtime.py`:** Update `_gpu_cache_key()` lines 1160-1165 to check `private_key_path.is_file()` before `os.access(repo_root, os.W_OK)`.
3. **Fix DEF-03 in `release_verify.sh`:** Exit non-zero on `ENGINEERING_INCOMPLETE` when `--offline` was not explicitly requested.
4. **Harden DEF-05 in `docker-compose.yml`:** Add explicit read-only mount for captions:
   `- ${HOST_FASHIONIQ_ROOT}/captions:/data/FashionIQ/captions:ro`.
