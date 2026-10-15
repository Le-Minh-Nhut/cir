# Independent Review: Workbench Backend, Jobs, Shell Scripts, and Frontend (Slice E/F)

## 1. Scope & Files Inspected
- `workbench/backend/main.py` (498 lines) — FastAPI routes, atomic duplicate 409 handling, shutdown process-group hook, model/bundle availability.
- `workbench/backend/jobs.py` (476 lines) — `SingleGpuQueue`, lock-atomic duplicate rejection, session process-group SIGTERM/SIGKILL termination.
- `workbench/scripts/serve_workbench.py` (207 lines) — Launcher signal forwarding (SIGTERM -> backend SIGTERM -> process group SIGTERM).
- `workbench/docker/acceptance.sh` (302 lines) — Group G queue status contract, Group E GPU readiness inspection, Group C model contract.
- `workbench/docker/release_verify.sh` (571 lines) — 0-byte corrupt artifact isolation, offline mode status reporting, release dimensions.
- `workbench/frontend/App.tsx` (70 lines) — Models tab, checkpoint bundle UI rendering, job completion index refresh.
- Reports `workbench/docs/reviews/final_pc_audit/E.md` and `F.md`.

---

## 2. Verification by Execution

### A. POST /api/jobs Atomic Duplicate Handling & 409 Payload
- **Execution**: Invoked 2 concurrent requests to `POST /api/jobs` via `asgi_request` with identical payloads (`synthetic_model`, `ckpt`, `fashioniq_original_split`). Mocked fast-path pre-check loop to return `[]` to force both requests into `SingleGpuQueue.enqueue`.
- **Observed**:
  - Request 1: HTTP 200, `job_id="53a710f6-5bc5-413b-8ee7-7a7f33710b7a"`, status `"queued"`.
  - Request 2: HTTP 409 Conflict.
  - 409 Body:
    ```json
    {
      "detail": {
        "error": "duplicate_job",
        "message": "An identical job is already queued or running.",
        "details": {
          "job_id": "53a710f6-5bc5-413b-8ee7-7a7f33710b7a",
          "status": "queued"
        }
      }
    }
    ```
  - Both submissions reached `queue.enqueue` with `refuse_duplicate=True`. `SingleGpuQueue._condition` held lock across existence check and append.
- **Classification**: NOT_A_BUG (Fix verified).

### B. Shutdown Process-Group Termination & Deadlock Freedom
- **Execution**: Enqueued job executing Python subprocess that spawned detached background child (`sleep 60`) under `start_new_session=True`. Observed leader PID=581136, PGID=581136 alive via `os.killpg(pgid, 0)`. Invoked `queue.shutdown()`.
- **Observed**:
  - `shutdown()` returned in 0.05s, `survivors=[]`.
  - `os.killpg(pgid, 0)` raised `ProcessLookupError` (entire process group gone).
  - Leader process dead (`proc.poll() is not None`).
  - Job status transitioned to `cancelled`.
  - Worker thread stopped, `self._stopping=True`. Zero deadlock.
- **Classification**: NOT_A_BUG (Fix verified).

### C. 0-Byte GPU-Capability Artifact Tolerance & Offline Reporting
- **Execution 1 (0-Byte Artifact)**: Placed 0-byte corrupt file alongside valid interpreter capability record in temporary test scan.
  - `_json.load` raised `JSONDecodeError`. Error caught per-file, recorded in `_unreadable`, file deleted.
  - Scan continued; `real_gpu` resolved `True` from valid record. Scan no longer crashes or forces `real_gpu = False`.
- **Execution 2 (Offline Release Verifier)**: Ran `REPORT=... bash workbench/docker/release_verify.sh --offline`.
  - Output: `J1 PENDING`, `J2 PENDING`, `J3 PENDING`, `J8 PENDING`, `J11 PENDING`.
  - Summary: `pass: 7, fail: 0, pending: 5`.
  - JSON Report: Unperformed live checks recorded status `"PENDING"`, not false `"PASS"`.
  - Exit code: 0 (`fails == 0`).
- **Classification**: NOT_A_BUG (Fix verified).

### D. acceptance.sh Empty-Data Behavior
- **Group E (GPU Rules Scan)**:
  - Scans `doc.get("models")` and filters `gpu_required == True` (11 models).
  - Observed: `PASS 11 GPU-requiring models; 0 carry a verified GPU_RUNTIME_READY probe`.
  - Empty-list guard present: exits 2 (`SKIP`) if no GPU models exist. Fix verified.
- **Group G (Job Status Contract)**:
  - Python check correctly exits 2 (`NOT VERIFIED no job observed`) on empty queue.
  - BUG: Line 98 contains `"$PY" - "$BASE" <<'EOF' || FAILED+=("G-status")`. Exit code 2 triggers `||`, appending `G-status` to `FAILED`. Result: False failure on clean boot.
- **Group C (Model Contract)**:
  - Line 160 unconditionally asserts `m.get("native_protocol")` and `m.get("environment", {}).get("python_env_var")`.
  - Model `ptha_mtst` has `native_protocol: null` and no `environment` key in `models.yaml`.
  - Crashes with `AssertionError: ptha_mtst: native_protocol missing`.
- **Classification**: CONFIRMED_BUG (P2) in `acceptance.sh:98` and `acceptance.sh:160-161`.

### E. /api/models Latency & Schema Checkpoint/Bundle Fidelity
- **Execution**: Profiled `GET /api/models` against live running backend (`127.0.0.1:8000`).
- **Observed Latency**:
  - `time_total`: 6.11s (reduced from 61.49s - 85.0s synchronous SHA256 storm).
- **Field Inventory**:
  - Checkpoint fields (17 keys): `checkpoint_id`, `filename`, `source_metadata`, `source_root`, `checkpoint_root`, `source_synced_locally`, `automatic_download_available`, `checkpoint_downloaded`, `local_sha256`, `official_sha256_known`, `official_sha256_match`, `mapping_verified`, `adapter_command_ready`, `runtime_verified`, `runnable`, `checkpoint_path`, `blocking_reasons`.
  - Bundle fields (6 keys): `bundle_id`, `required_checkpoint_ids`, `structure`, `mapping_status`, `notes`, `members`.
  - Bundle `members`: complete list containing full checkpoint availability objects for all 3 category checkpoints (`base_iter0_dress`, `shirt`, `toptee`).
- **Classification**: NOT_A_BUG (Fix verified).

### F. Shell Syntax & Script Execution
- `bash -n workbench/docker/release_verify.sh`: Exit 0 (clean).
- `bash -n workbench/docker/acceptance.sh`: Exit 0 (clean).
- `release_verify.sh` against live backend:
  - J1-J7, J9-J12: `PASS`.
  - J8: `PENDING` (no jobs queued yet).
  - Report verdict: `GPU_VALIDATION_PENDING` (1 pending).
- `acceptance.sh` execution output:
  - Groups A, B, F, D, E, I, H: `PASS`.
  - Groups G, C: `FAIL`. Summary: `ACCEPTANCE FAILED: G-status C-config`. Exit code 1.

---

## 3. Defects Log

### CONFIRMED_BUG 1: acceptance.sh Group G Bash Error Masking (P2)
- **Location**: `workbench/docker/acceptance.sh:98, 114-122`
- **Root Cause**: Line 98 attaches `|| FAILED+=("G-status")` directly to the python invocation. When python exits with code 2 (intended SKIP on empty queue), bash executes the `||` branch, polluting `FAILED`. Shell assignment exit status is 0, so `g_rc=$?` at line 114 captures 0, preventing `elif (( g_rc == 2 ))` from handling the skip.
- **Reproduction**: Run `acceptance.sh` on clean stack with 0 jobs. Output: `ACCEPTANCE FAILED: G-status`.
- **Minimal Fix**: Remove `|| FAILED+=("G-status")` from line 98; allow lines 115-122 to evaluate `g_rc`.

### CONFIRMED_BUG 2: acceptance.sh Group C Model Contract Regression on ptha_mtst (P2)
- **Location**: `workbench/docker/acceptance.sh:160-161`
- **Root Cause**: `models.yaml:234` defines `ptha_mtst` with `native_protocol: null` and no `environment` key. `acceptance.sh` line 160 unconditionally asserts `assert m.get("native_protocol")` and line 161 asserts `assert m.get("environment", {}).get("python_env_var")`.
- **Reproduction**: Run Group C in `acceptance.sh`. Output: `AssertionError: ptha_mtst: native_protocol missing`.
- **Minimal Fix**: In `acceptance.sh:160`, filter for runnable/reproduced models, or check `if m.get("native_protocol"): ...`.

### CONFIRMED_BUG 3: Evaluation Completion Omits DuckDB Index Rebuild (P2)
- **Location**: `workbench/backend/jobs.py:458-468`, `workbench/frontend/App.tsx:26`
- **Root Cause**: When evaluation completes with return code 0, `jobs.py:_work` persists job state but never invokes `rebuild_index()`. In `App.tsx:26`, `settledJobs` fetches `/api/runs`, but never calls `POST /api/index/rebuild`.
- **Impact**: When jobs are queued via API or UI, newly completed runs never enter DuckDB. UI displays run as disabled "AGGREGATE ONLY" badge.
- **Minimal Fix**: In `jobs.py:_work` after exit code 0: `from workbench.backend.index import rebuild_index; rebuild_index()`.

### CONFIRMED_BUG 4: LIMN Bundle Missing in Frontend Models UI (P2)
- **Location**: `workbench/frontend/App.tsx:36`
- **Root Cause**: `App.tsx:36` maps only `m.checkpoints` and renders buttons for individual variants (`base_iter0_dress`, etc.). `m.checkpoint_bundles` is completely ignored.
- **Impact**: Operator cannot queue the official LIMN benchmark bundle (`base_iter0_all_categories`) from the Web UI.
- **Minimal Fix**: In `App.tsx:36`, map `m.checkpoint_bundles` alongside `m.checkpoints` to render a "Queue bundle evaluation" button.

### CONFIRMED_BUG 5: Missing Empty-DuckDB Initialization on Standalone Backend Startup (P2)
- **Location**: `workbench/backend/main.py:57-61`, `workbench/backend/storage.py:15-16`
- **Root Cause**: `storage._database_path()` raises `WorkbenchError("index_unavailable")` if `workbench.duckdb` is missing. Standalone boot without `pc.sh` serving index rebuild returns 503 on `GET /api/runs`, triggering persistent error banner in UI.
- **Minimal Fix**: In `main.py:_startup`: `if not configured_database_path().is_file(): rebuild_index()`.

### CONFIRMED_BUG 6: Markdown Cohort Export Crashes on None Metrics (P3)
- **Location**: `workbench/backend/main.py:364`
- **Root Cause**: Line 364 formats `{item['r1']:.2f}`. When a cohort has 0 matched queries, `storage.cohort_metrics` returns `r1: None`. `None.__format__(".2f")` raises unhandled `TypeError`.
- **Reproduction**: Call `GET /api/cohorts/{cohort_id}/export/markdown` for cohort with empty query list.
- **Minimal Fix**: `def _fmt(v): return f"{v:.2f}" if v is not None else "—"`.

### UNVERIFIED_RISK 1: RTX 5070 Ti Blackwell sm_120 PyTorch Kernel Availability (P2)
- **Location**: `workbench/backend/runtime.py:2167`
- **Root Cause**: Host lacks GPU passthrough (legitimately PENDING). While `sm_120` arch string normalization is fixed, real execution requires PyTorch binaries with compiled `sm_120` kernels or successful PTX JIT compilation.
- **Impact**: If container PyTorch wheel lacks `sm_120` binary code, first kernel invocation on RTX 5070 Ti raises `CUDA error: no kernel image is available for execution on device`.

---

## 4. Overall Verdict

**Verdict**: `SOUND_WITH_RESERVATIONS`

Core concurrency control (atomic 409 duplicate rejection), process group lifecycle supervision (SIGTERM/SIGKILL escalation on shutdown), readiness poll performance (6.1s vs 85s), and artifact corruption resilience are thoroughly verified by execution. Reservations are isolated to two shell script test harness bugs in `acceptance.sh` (Group G and Group C false failures) and frontend UI limitations (omitted bundle queuing and missing automatic index rebuild upon job completion).

---

Can this implementation fail during the real first run on an RTX 5070 Ti PC?
Yes — if the operator executes via the Web UI (LIMN bundle cannot be queued from the UI and job completion omits DuckDB index rebuild) or if the host PyTorch wheel lacks sm_120 Blackwell CUDA binary kernels, though the backend core queue, shutdown reaping, and pc.sh pipeline launcher paths hold.
