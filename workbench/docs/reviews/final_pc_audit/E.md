# Final Source Audit Report: Slice E (Workbench Backend, Queue, Index, Frontend, Pipeline)

## A. Files Inspected
- `workbench/backend/main.py` (467 lines) — FastAPI ASGI app, endpoints, exception handling, job routing, aggregate report filtering, cohort exports.
- `workbench/backend/jobs.py` (431 lines) — `SingleGpuQueue`, process group supervision, persistence, startup recovery, environment sanitization.
- `workbench/backend/index.py` (110 lines) — DuckDB index creation, CSV bulk ingest, atomic replacement, `rebuild_index`.
- `workbench/frontend/App.tsx` (69 lines / 2549 chars compacted) — React UI state, protocol visibility, gate thresholds, sample explorer, analysis tables.
- `workbench/scripts/pipeline.py` (2345 lines) — Stage execution DAG, fingerprints, CLI parser, capability gating.
- `workbench/backend/storage.py` (204 lines) — DuckDB queries, cohort metrics, query universe alignment.
- `workbench/backend/registry.py` (745 lines) — Model registry loading, checkpoint availability, sha256 hashing.
- `workbench/backend/runtime.py` (2322 lines) — Runtime blockers, GPU capability verification, Blackwell `sm_120` profile.

---

## B. Paths Traced
1. **Empty-DB First Boot**: Clean host clone (no `workbench.duckdb`) -> `serve_workbench.py` / `uvicorn main:app` -> `_startup()` -> `_recover_interrupted_jobs()` -> client calls `GET /api/runs` -> `storage.list_runs()` -> `_database_path()` -> raises `WorkbenchError("index_unavailable")` -> `main.py` exception handler returns HTTP 503 -> Frontend `refresh()` catches error -> displays persistent error banner `Guard/API error: runs: DuckDB index is unavailable` -> dashboard/run explorer blanked.
2. **Readiness Probe & Checkpoint Hashing**: Client / `./pc.sh` / Docker probes `GET /api/models` -> `models()` calls `checkpoint_availability()` for all 28 checkpoints -> triggers `sha256_file()` directly and inside `runtime_blockers()` -> 56 sha256 file reads on multi-GB weights -> 61.49s synchronous latency -> readiness probe times out.
3. **Queue Ingestion & Duplicate Check (TOCTOU)**: Concurrent `POST /api/jobs` requests -> `main.py:398` reads `JOBS.list()` outside lock -> checks pass -> slow `guarded_plan()` and `require_launch_ready()` execute -> `JOBS.enqueue(..., refuse_duplicate=False)` called -> both identical jobs enqueued into RAM queue -> GPU double-booked.
4. **Job Execution to Result Refresh**: `POST /api/jobs` enqueues job -> worker runs `evaluate_models.py` -> child outputs results to `results/` and reports to `reports/` -> job marked `completed` -> Frontend `settledJobs` detects completion -> calls `loadReports()` and `api<Run[]>("/api/runs")` -> `rebuild_index()` never triggered -> DuckDB index remains stale -> new run absent from `/api/runs` -> run unselectable in UI analysis tabs.
5. **Server Shutdown & Process Group Leak**: `docker stop` or SIGTERM delivered to container -> Python main thread exits -> `SingleGpuQueue._worker` daemon thread abruptly terminated without executing cleanup -> child process spawned with `start_new_session=True` resides in separate session -> SIGTERM not forwarded to child group -> evaluation process continues running as orphan holding 16 GB VRAM on RTX 5070 Ti.
6. **LIMN Bundle UI Execution**: User opens Web UI -> navigates to Models -> LIMN entry shows only individual checkpoints (`base_iter0_dress`, etc.) -> `checkpoint_bundles` (`base_iter0_all_categories`) not rendered -> user cannot queue macro bundle evaluation from UI.
7. **Markdown Cohort Export**: User creates cohort -> exports via `GET /api/cohorts/{cohort_id}/export/markdown` -> `cohort_metrics()` returns `None` for runs with zero matching queries -> line 349 executes `{item['r1']:.2f}` on `None` -> raises `TypeError` -> unhandled HTTP 500 error.

---

## C. Confirmed Defects

### DEFECT 1: Missing Empty-DuckDB Initialization on Clean Startup
- **Classification**: CONFIRMED
- **Severity**: P1
- **Location**: `workbench/backend/main.py:57-62`, `workbench/backend/storage.py:13-17`
- **Root Cause**: `_startup()` only runs job recovery. `workbench.duckdb` is gitignored. Fresh clone contains no database file. `storage._database_path()` raises `WorkbenchError("index_unavailable")`.
- **Reproduction**: Run backend without `workbench.duckdb`. Invoke `GET /api/runs` via ASGI: returns HTTP 503 `{"error": "index_unavailable", "message": "DuckDB index is unavailable..."}`.
- **Expected vs Actual**: Expected: clean startup creates empty schema or indexes existing results, returning 200 `[]`. Actual: returns 503; frontend displays blocking error banner and disables all run explorer views.
- **Impact**: Immediate user-facing failure on every clean machine first run.
- **Minimal Fix**: In `main.py:_startup`, check `if not configured_database_path().is_file(): rebuild_index()`.

### DEFECT 2: TOCTOU Duplicate Job Race Condition
- **Classification**: CONFIRMED
- **Severity**: P1
- **Location**: `workbench/backend/main.py:398-404, 439-440`, `workbench/backend/jobs.py:108-118`
- **Root Cause**: `main.py` checks `JOBS.list()` outside the condition lock, spends several seconds in `guarded_plan` and `require_launch_ready`, then calls `JOBS.enqueue` with default `refuse_duplicate=False`. Atomic duplicate rejection in `jobs.py` is bypassed.
- **Reproduction**: Invoke concurrent `POST /api/jobs` for identical payload (`limn`, `base_iter0_all_categories`, `fashioniq_val_split`). Both return HTTP 200 with unique job IDs; queue contains 2 identical in-flight jobs.
- **Expected vs Actual**: Expected: second request rejected with HTTP 409 `duplicate_job`. Actual: both jobs enqueued; duplicate GPU executions run concurrently or back-to-back.
- **Impact**: Double-clicks in UI or concurrent scripts duplicate expensive GPU evaluations and overwrite outputs.
- **Minimal Fix**: Pass `refuse_duplicate=True` in `JOBS.enqueue(...)`, wrap in `try/except DuplicateJobError as e: raise HTTPException(409, {"error": "duplicate_job", ...})`.

### DEFECT 3: Evaluation Process Group Orphaned on Container / Server Shutdown
- **Classification**: CONFIRMED
- **Severity**: P1
- **Location**: `workbench/backend/jobs.py:105, 385-388`, `workbench/backend/main.py:34-36`
- **Root Cause**: Queue worker is a daemon thread (`daemon=True`). Evaluator child is spawned with `start_new_session=True`. FastAPI lacks `@app.on_event("shutdown")`. On SIGTERM to Uvicorn/Docker, main process dies, daemon thread is killed without cleanup, and detached child process group continues executing.
- **Reproduction**: Enqueue long-running command (`sleep 30`), send SIGTERM to backend process. Child PID remains active in OS (`pgrep -f "sleep 30"` returns active PIDs).
- **Expected vs Actual**: Expected: backend shutdown terminates child process groups. Actual: GPU worker process survives parent shutdown as an orphan holding 16 GB VRAM.
- **Impact**: Restarting container or `./pc.sh` after abort leaves GPU memory occupied; next run crashes with CUDA OOM.
- **Minimal Fix**: Add `@app.on_event("shutdown")` in `main.py` to iterate active `JOBS._processes` and invoke `SingleGpuQueue._terminate_group(proc, group=grp)`.

### DEFECT 4: Queue Job Completion Omits Index Rebuild
- **Classification**: CONFIRMED
- **Severity**: P2
- **Location**: `workbench/backend/jobs.py:413-423`, `workbench/frontend/App.tsx:26`
- **Root Cause**: Evaluator completes and writes results to disk, but `jobs.py` does not invoke `rebuild_index()`. In `App.tsx`, `settledJobs` fetches `/api/runs/gate` and `/api/runs`, but never calls `POST /api/index/rebuild`. DuckDB remains unindexed.
- **Reproduction**: Complete an evaluation job via API; query `GET /api/benchmark-runs` (returns 1 run from aggregate reports) vs `GET /api/runs` (returns 0 runs from DuckDB).
- **Expected vs Actual**: Expected: completed run indexed into DuckDB and selectable in UI. Actual: run visible only as "AGGREGATE ONLY" badge with disabled checkbox.
- **Impact**: Newly evaluated models cannot be analyzed, compared, or inspected in UI without manual external CLI command.
- **Minimal Fix**: Call `rebuild_index()` in `jobs.py:_work` when `code == 0`, or trigger `POST /api/index/rebuild` in `App.tsx:26` before `setRuns`.

### DEFECT 5: Synchronous Checkpoint Hashing in `GET /api/models`
- **Classification**: CONFIRMED
- **Severity**: P2
- **Location**: `workbench/backend/main.py:103-121`, `workbench/backend/registry.py:703-742`, `workbench/backend/runtime.py:2300-2321`
- **Root Cause**: `models()` synchronously executes `checkpoint_availability` on 28 checkpoints, executing 56 `sha256_file` passes on multi-GB weights on every request.
- **Reproduction**: Direct ASGI profiling of `GET /api/models`: 61.49 seconds wall time; 41.0s spent in `_hashlib.HASH.update`.
- **Expected vs Actual**: Expected: response under 1s. Actual: 61s request latency blocks worker threads; standard readiness polling fails.
- **Impact**: Frontend initial load freezes; `./pc.sh` / Docker readiness checks time out if configured with standard timeouts.
- **Minimal Fix**: Cache checkpoint verification status by file `(mtime, size)` instead of recomputing SHA256 on every API query.

### DEFECT 6: LIMN Checkpoint Bundle Missing in Frontend UI
- **Classification**: CONFIRMED
- **Severity**: P2
- **Location**: `workbench/frontend/App.tsx:36`
- **Root Cause**: `App.tsx` only iterates `m.checkpoints.map(...)` and renders queue buttons for individual variants (`base_iter0_dress`, etc.). It ignores `m.checkpoint_bundles`, which contains `base_iter0_all_categories`.
- **Reproduction**: Inspect `App.tsx:36`: no element renders `m.checkpoint_bundles`.
- **Expected vs Actual**: Expected: user can select and queue `base_iter0_all_categories` bundle. Actual: UI only allows queuing individual category checkpoints.
- **Impact**: Core mission benchmark (LIMN macro evaluation) cannot be dispatched from GUI.
- **Minimal Fix**: In `App.tsx:36`, map `m.checkpoint_bundles` alongside `m.checkpoints` to render a "Queue bundle evaluation" button.

### DEFECT 7: Unhandled `TypeError` in Markdown Cohort Export on None Metrics
- **Classification**: CONFIRMED
- **Severity**: P3
- **Location**: `workbench/backend/main.py:348-350`
- **Root Cause**: Line 349 formats `item['r1']:.2f` directly. When cohort queries have no matches for a run, `cohort_metrics()` explicitly returns `None` for metrics. `NoneType.__format__` raises `TypeError`.
- **Reproduction**: Execute `f"{None:.2f}"` in Python; raises `TypeError: unsupported format string passed to NoneType.__format__`.
- **Expected vs Actual**: Expected: renders "—" for None metrics in Markdown table. Actual: unhandled HTTP 500 error.
- **Impact**: Cohort export crashes whenever a run has zero matched queries.
- **Minimal Fix**: In `main.py:349`, format values with `f"{v:.2f}" if v is not None else "—"`.

---

## D. Probable Defects

### DEFECT 8: Lack of React Error Boundary Causes Full UI Whiteout
- **Classification**: PROBABLE
- **Severity**: P3
- **Location**: `workbench/frontend/App.tsx:12-44`
- **Root Cause**: Entire application lacks an `ErrorBoundary`. If any child component encounters an unexpected null property (e.g. malformed query or missing distractor array in `AnalysisTable` line 68), React unmounts root.
- **Impact**: Single malformed record crashes entire UI into a blank white page.
- **Minimal Fix**: Wrap tab view in a React Error Boundary with fallback reset.

### DEFECT 9: Deselecting All Runs Leaves Stale Query Explorer State
- **Classification**: PROBABLE
- **Severity**: P3
- **Location**: `workbench/frontend/App.tsx:28`
- **Root Cause**: `const runId=selected[0]; if(!runId) return;` aborts without clearing `queries`, `total`, or `query`.
- **Impact**: Sample Explorer continues displaying queries from previous run even when zero runs are selected.
- **Minimal Fix**: If `!runId`, call `setQueries([])`, `setTotal(0)`, `setQuery(null)`.

---

## E. Coverage Gaps
1. No automated test covers server startup on a filesystem without `workbench.duckdb` (`test_http_api_layer.py` uses pre-seeded test fixtures).
2. No test exercises ASGI shutdown behavior or verifies that child process groups spawned by `SingleGpuQueue` receive SIGTERM on server stop.
3. No end-to-end integration test covers the UI loop: queuing a job from frontend -> completion -> DuckDB index rebuild -> run display in `/api/runs`.
4. No test validates Markdown cohort export when `query_count == 0` or metrics contain `None`.

---

## F. Minimal Fixes

```python
# Fix 1 & Fix 3: main.py startup & shutdown hooks
@app.on_event("startup")
def _startup() -> None:
    _recover_interrupted_jobs()
    if not configured_database_path().is_file():
        rebuild_index()

@app.on_event("shutdown")
def _shutdown() -> None:
    for job_id in list(JOBS._processes):
        try:
            JOBS.cancel(job_id)
        except Exception:
            pass

# Fix 2: main.py atomic duplicate enqueue
@app.post("/api/jobs")
def queue_job(payload: JobInput):
    ...
    try:
        return JOBS.enqueue(plan.model_id, plan.checkpoint_id, plan.protocol, argv, ROOT,
                            job_directory, provenance, refuse_duplicate=True)
    except DuplicateJobError as error:
        raise HTTPException(409, {"error": "duplicate_job", "message": str(error),
                                  "details": {"job_id": error.job_id, "status": error.status}})

# Fix 4: jobs.py index rebuild on completion
# In SingleGpuQueue._work when code == 0:
if code == 0:
    try:
        from workbench.backend.index import rebuild_index
        rebuild_index()
    except Exception:
        pass

# Fix 7: main.py markdown cohort export formatting
def _fmt(val): return f"{val:.2f}" if isinstance(val, (int, float)) else "—"
body = "\n".join(f"| {item['run_id']} | {item['query_count']} | {_fmt(item['r1'])} | {_fmt(item['r5'])} | {_fmt(item['r10'])} | {_fmt(item['r20'])} | {_fmt(item['r50'])} | {_fmt(item['r100'])} | {_fmt(item['mean_target_rank'])} | {_fmt(item['median_target_rank'])} |" for item in metrics)
```

```tsx
// Fix 6: App.tsx bundle rendering
{m.checkpoint_bundles?.map(b=><div className="checkpoint bundle" key={b.bundle_id}>{b.bundle_id} (bundle)<br/><button onClick={()=>queue(m,b.bundle_id)}>Queue official bundle</button></div>)}
```

---

## G. Required Regression Tests
1. `test_empty_duckdb_startup`: Delete `workbench.duckdb`, start ASGI app, send `GET /api/runs`, assert status 200 and body `[]`.
2. `test_concurrent_duplicate_job_enqueue`: Send 5 concurrent ASGI requests to `POST /api/jobs` with identical parameters; assert exactly 1 returns 200 and 4 return 409.
3. `test_shutdown_cleans_process_groups`: Enqueue long-running command, trigger FastAPI shutdown event, assert process group is terminated (ESRCH probe).
4. `test_job_completion_updates_duckdb_index`: Dispatch job, wait for completion, assert `GET /api/runs` immediately includes the new run.
5. `test_markdown_export_with_none_metrics`: Call `GET /api/cohorts/{cohort_id}/export/markdown` on cohort with 0 matched queries; assert status 200 and no 500 TypeError.

---

Can this implementation fail during the real first run on an RTX 5070 Ti PC?
Yes.
