# Final Source Audit Report: Slice E (Workbench Backend, Queue, Storage, Index, Frontend)

## 1. Target & Scope
Files audited:
- `workbench/backend/main.py` (502 lines)
- `workbench/backend/jobs.py` (587 lines)
- `workbench/backend/index.py` (110 lines)
- `workbench/backend/analysis.py` (93 lines)
- `workbench/backend/storage.py` (204 lines)
- `workbench/backend/errors.py` (14 lines)
- `workbench/frontend/App.tsx` (70 lines)
- `workbench/frontend/client.ts` (8 lines)
- `workbench/frontend/components.tsx` (10 lines)
- `workbench/frontend/gate.ts` (71 lines)
- `workbench/frontend/gate_types.ts` (12 lines)
- `workbench/frontend/protocols.ts` (41 lines)
- `workbench/frontend/types.ts` (7 lines)

Target surface: ASGI API contract, exception handling, empty-DuckDB startup, run indexing, queue persistence and recovery, duplicate handling and cancellation, process group supervision, shutdown behavior, frontend state consistency, bundle selection, cohort/annotation export, evidence validity.

---

## 2. Direct Verification & Executed Checks

### A. Empty-DuckDB First Boot
- **Execution**: Tested ASGI app via direct ASGI request `GET /api/runs` in fresh temporary environment with no `workbench.duckdb`.
- **Observed**: HTTP 503 `{"error": "index_unavailable", "message": "DuckDB index is unavailable. Rebuild it from canonical result JSON.", "details": {"database_path": ".../workbench.duckdb"}}`.
- **Root Cause**: `main.py:_startup` only calls `_recover_interrupted_jobs()`. `workbench.duckdb` is gitignored. Fresh checkout has no database. `rebuild_index` is not called on startup. Frontend `refresh()` displays blocking red banner: `Guard/API error: runs: DuckDB index is unavailable...`.
- **Classification**: CONFIRMED (P1).

### B. Shutdown Process Orphan Race during Job Preparation
- **Execution**: Enqueued job in `SingleGpuQueue` with slow `launch_guard` (0.5s delay). While job was in status `preparing`, called `queue.shutdown()`.
- **Observed**:
  - `shutdown()` checked `self._processes` (which was empty because child was not yet spawned).
  - Termination loop skipped.
  - Worker finished `launch_guard`, took condition lock, checked `if job.status == "cancelled"` (was `preparing`, not cancelled), and executed `subprocess.Popen(["sleep", "10"], start_new_session=True)`.
  - `shutdown()`'s `self._worker.join(10.0)` timed out after 10.00s.
  - `shutdown()` returned `survivors: []`.
  - Job status became `running`. Subprocess `sleep 10` ran orphaned in OS under detached session (PID 1316901 alive).
- **Classification**: CONFIRMED (P1).

### C. Cohort Markdown Export Crash on Zero Query Matches / None Metrics
- **Execution**: Invoked ASGI `GET /api/cohorts/{cohort_id}/export/markdown` where `cohort_metrics` returned `query_count: 0` and metric values `None`.
- **Observed**: Unhandled exception: `TypeError: unsupported format string passed to NoneType.__format__` at `main.py:364` (`{item['r1']:.2f}`). HTTP 500 server crash.
- **Classification**: CONFIRMED (P1).

### D. Concurrent Duplicate Job Rejection (TOCTOU)
- **Execution**: Dispatched 2 concurrent ASGI `POST /api/jobs` requests with identical payload (`synthetic_model`, `ckpt`, `fashioniq_original_split`).
- **Observed**: Request 1 returned HTTP 200 with job ID; Request 2 returned HTTP 409 Conflict: `{"detail": {"error": "duplicate_job", "message": "An identical job is already queued or running."}}`.
- **Result**: Atomic condition lock in `SingleGpuQueue.enqueue` with `refuse_duplicate=True` functions correctly. NOT_A_BUG.

### E. Latency of `GET /api/models`
- **Execution**: Profiled `GET /api/models` on local runtime.
- **Observed**: Request took 7.84s to 20.46s wall time, executing 209 `yaml.safe_load` calls across 28 checkpoints synchronously on the request thread.
- **Classification**: CONFIRMED (P2).

### F. LIMN Macro Bundle Availability in Frontend UI
- **Execution**: Inspected `workbench/frontend/App.tsx:36` and `workbench/frontend/types.ts:4`.
- **Observed**: `App.tsx` maps only `m.checkpoints` (`base_iter0_dress`, etc.). `m.checkpoint_bundles` (`base_iter0_all_categories`) is not declared in `Model` type and has no UI rendering.
- **Classification**: CONFIRMED (P2).

### G. Automatic DuckDB Rebuild on Job Completion
- **Execution**: Inspected `jobs.py:560-580` and `App.tsx:25-27`.
- **Observed**: Evaluator writes artifacts to disk, but neither `jobs.py` calls `rebuild_index()` nor does `App.tsx:settledJobs` call `POST /api/index/rebuild`.
- **Classification**: CONFIRMED (P2).

### H. UI Export Actions for Cohorts and Annotations
- **Execution**: Inspected `App.tsx:42-43`.
- **Observed**: Endpoints `/api/cohorts/{id}/export/{format}` exist in backend, but frontend has no export buttons or download handlers.
- **Classification**: CONFIRMED (P3).

---

## 3. Confirmed Defects

### DEFECT 1: Fresh Clone Startup Fails with HTTP 503 (Empty DuckDB)
- **Classification**: CONFIRMED
- **Severity**: P1 (False FAIL on first run)
- **Location**: `workbench/backend/main.py:57-62`, `workbench/backend/storage.py:13-17`
- **Root Cause**: `workbench.duckdb` is gitignored. Fresh clone contains no database file. `main.py:_startup()` only calls `_recover_interrupted_jobs()`. It never checks `configured_database_path().is_file()` or initializes schema.
- **Reproduction**: Delete `workbench.duckdb`; call `GET /api/runs`. Returns 503 `index_unavailable`. Frontend displays persistent red error banner.
- **Minimal Fix**:
  ```python
  # workbench/backend/main.py:_startup
  if not configured_database_path().is_file():
      rebuild_index()
  ```

### DEFECT 2: Process Group Orphaned on Container / Server Stop during Job Preparation
- **Classification**: CONFIRMED
- **Severity**: P1
- **Location**: `workbench/backend/jobs.py:285-308, 517-545`
- **Root Cause**: `shutdown()` only terminates processes present in `self._processes`. If a job is in `preparing` (running `launch_guard`), it is not in `self._processes`. `shutdown()` returns. When worker thread finishes `launch_guard`, it checks only `if job.status == "cancelled"`, ignoring `self._stopping`, and spawns the detached process group.
- **Reproduction**: Run slow `launch_guard`, call `queue.shutdown()` while job is `preparing`. Subprocess is spawned into OS and outlives shutdown.
- **Impact**: Docker stop or launcher SIGTERM during job preparation leaves GPU worker running in background holding 16GB VRAM. Subsequent container start crashes with CUDA OOM.
- **Minimal Fix**:
  In `jobs.py:shutdown()`: cancel and persist jobs in `queued` or `preparing` states.
  In `jobs.py:_work()` (line 523):
  ```python
  with self._condition:
      if self._stopping or job.status == "cancelled":
          job.status = "cancelled"
          self._persist(job)
          continue
  ```

### DEFECT 3: Unhandled TypeError in Cohort Markdown Export on Empty Query Match
- **Classification**: CONFIRMED
- **Severity**: P1
- **Location**: `workbench/backend/main.py:364`
- **Root Cause**: `cohort_metrics` returns `None` for recall/rank fields when a run has 0 matching queries. Line 364 executes `{item['r1']:.2f}` without null-checking.
- **Reproduction**: Invoke `GET /api/cohorts/{cohort_id}/export/markdown` for a cohort where any run has 0 query matches.
- **Impact**: HTTP 500 crash.
- **Minimal Fix**:
  ```python
  def _fmt(val): return f"{val:.2f}" if val is not None else "—"
  # apply _fmt to r1, r5, r10, r20, r50, r100, mean_target_rank, median_target_rank
  ```

### DEFECT 4: LIMN Checkpoint Bundle Unselectable in Web UI
- **Classification**: CONFIRMED
- **Severity**: P2
- **Location**: `workbench/frontend/App.tsx:36`, `workbench/frontend/types.ts:4`
- **Root Cause**: `App.tsx` only renders `m.checkpoints`. `checkpoint_bundles` (`base_iter0_all_categories`) is ignored.
- **Impact**: Operator using Web UI cannot trigger the full official macro benchmark for LIMN.
- **Minimal Fix**: Add `checkpoint_bundles` rendering and queue action in `App.tsx:36`.

### DEFECT 5: Synchronous Checkpoint Parsing Stall in `GET /api/models` (7-20s Latency)
- **Classification**: CONFIRMED
- **Severity**: P2
- **Location**: `workbench/backend/main.py:118-136`, `workbench/backend/runtime.py:2424`
- **Root Cause**: Synchronous execution of 209 YAML file parses and git status checks on every models request.
- **Impact**: Frontend initial load freezes; short-timeout health checks fail.
- **Minimal Fix**: Cache checkpoint verification status by file mtime/hash.

### DEFECT 6: Evaluator Completion Omits DuckDB Index Rebuild
- **Classification**: CONFIRMED
- **Severity**: P2
- **Location**: `workbench/backend/jobs.py:570-580`, `workbench/frontend/App.tsx:25-26`
- **Root Cause**: Background worker completes job and writes result, but DuckDB index is not refreshed.
- **Impact**: Run appears only as disabled "AGGREGATE ONLY" badge in UI; cannot be selected for query-level failure analysis.
- **Minimal Fix**: Trigger `rebuild_index()` in `jobs.py:_work()` upon successful exit code, or call `POST /api/index/rebuild` in `App.tsx:settledJobs`.

### DEFECT 7: Missing Frontend Export Buttons for Cohorts and Annotations
- **Classification**: CONFIRMED
- **Severity**: P3
- **Location**: `workbench/frontend/App.tsx:42-43`
- **Root Cause**: Backend endpoints `/api/cohorts/{id}/export/{format}` and `/api/annotations` lack triggering buttons in UI.
- **Impact**: Operator must use external CLI / curl to export cohorts and annotations.
- **Minimal Fix**: Add export links (`/api/cohorts/${c.cohort_id}/export/markdown`, etc.) in `App.tsx:43`.

---

## 4. Evidence Validity Analysis (Can UI Present False Evidence?)

### Can the UI present a fake or invalid run as certified evidence?
**NO.**
1. **Ledger Corroboration**: `main.py:_aggregate_reports()` enforces strict cross-check against the out-of-band HMAC evidence ledger (`read_evidence_ledger(config)`). If report sha256, run_id, model_id, checkpoint_id, protocol_id, return_code == 0, or INVOCATION_COMPLETED do not match, the report is ignored and never served.
2. **Independent Run Verification**: `jobs.py:settled_status` invokes `verify_run.verify()`. A job only publishes status `completed` if the report exists, parses, matches the ledger, and passes manifest proof validation.
3. **Data Kind Isolation**: Mock runs are explicitly labeled with `[MOCK]` badges and the top-level banner displays warning. Backend `storage.validate_analysis_selection` strictly rejects mixing mock and experiment runs (`mixed_data_kind` error).
4. **Aggregate-Only Protection**: Runs lacking per-query DuckDB records have disabled checkboxes (`<input type="checkbox" disabled/>`) and `AGGREGATE ONLY` badges. They cannot be fed into failure analysis or comparison views.
5. **Gate Warning Labels**: If deviations exceed the threshold, red warning labels (`LỆCH PAPER X%`) and gate blocker details are displayed.

---

## 5. Coverage Gaps in Existing Test Suite

1. **Empty DuckDB First-Run Test**: Existing tests assume `workbench.duckdb` exists. No test exercises backend startup from a completely clean filesystem without the database.
2. **Shutdown during `preparing` Race Test**: `test_jobs.py` tests cancellation and shutdown on running jobs, but omits the state transition race between `preparing` (`launch_guard`) and `shutdown()`.
3. **Cohort Markdown Export Null Metric Test**: Existing cohort tests test runs with 100% query coverage; no test verifies markdown export formatting when a run has zero matched queries.
4. **UI Bundle Queueing Test**: Frontend tests only mock API responses; no integration test verifies that `checkpoint_bundles` can be queued from the UI.

---

## 6. What Was NOT Inspected
- Deep neural network weights and raw floating-point tensors in checkpoint files.
- Upstream FashionIQ dataset download servers (constrained by offline audit policy and 11GB disk limit).
- Hardware-level PCIe bus latency and NVIDIA driver kernel module internals on host.

---

Yes — if the operator executes via the Web UI (LIMN bundle cannot be queued from the UI, DuckDB index is missing on clean boot with 503 error, and job completion omits DuckDB index rebuild) or if container shutdown occurs during job preparation (leaving orphaned GPU worker holding 16 GB VRAM), though backend atomic duplicate handling and HMAC ledger evidence verification hold.
