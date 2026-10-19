# R10 E2E Surface and Test Coverage Audit

## 1. Promise Inventory
1. Source clone: Operator clones git repository and fetches branch `research/cir-failure-workbench` (`workbench/docs/PC_PILOT_HANDOFF.md:65-66`, `workbench/docs/MODEL_STATUS_SPEC.md:294-295`).
2. Source revision pin: Operator checks out byte-identical tested commit `23dca02` without hard reset (`workbench/docs/PC_PILOT_HANDOFF.md:70-71`, `workbench/docs/MODEL_STATUS_SPEC.md:296`).
3. Fresh clone self-check: Operator verifies clone integrity via `test_fresh_clone_delivery.py` and `test_vendored_source_integrity.py` without checkpoints (`workbench/docs/MODEL_STATUS_SPEC.md:298-301`).
4. Asset dry-run inspection: Operator identifies missing checkpoints and auxiliary files using `--dry-run` scripts before downloading (`workbench/docs/MODEL_STATUS_SPEC.md:304-306`).
5. Asset staging: Operator downloads checkpoints and auxiliary assets for target models (`workbench/docs/MODEL_STATUS_SPEC.md:308-311`).
6. Preflight verification: Operator runs `--preflight-only` to confirm blockers reported cleanly without runtime crashes (`workbench/docs/MODEL_STATUS_SPEC.md:313-318`).
7. Single-command entrypoint: Operator executes `./pc.sh` as single orchestration command (`workbench/docs/PC_PILOT_HANDOFF.md:78-83`, `pc.sh:4-5`).
8. Host and runtime preflight: Launcher checks OS, memory (>=8GB), disk (>=25GB), docker daemon, compose v2, and nvidia runtime (`workbench/docs/PC_PILOT_HANDOFF.md:80`, `pc.sh:7`, `pc.sh:310-535`).
9. Docker stack orchestration: Launcher verifies `.env`, mount paths, commit label matching, spins up container `cir-workbench` via `run.sh up` (`workbench/docs/PC_PILOT_HANDOFF.md:80`, `pc.sh:564-850,862-1086`).
10. Stack readiness polling: Launcher polls backend `/api/jobs` and frontend until HTTP 200/304 ready (`workbench/docs/PC_PILOT_HANDOFF.md:80`, `pc.sh:1056-1080`).
11. Dataset validation: Launcher asserts FashionIQ images and splits accessible at `/data/FashionIQ` (`workbench/docs/PC_PILOT_HANDOFF.md:81,110`, `pc.sh:1082-1085`).
12. Checkpoint bundle validation: Launcher asserts full 3-checkpoint bundle for LIMN (`0_{dress,shirt,toptee}_best_model.pt` + SHA256) present, failing closed if any member missing (`workbench/docs/PC_PILOT_HANDOFF.md:81,111-112,118`, `pc.sh:1270-1282`).
13. Environment planning: Launcher assigns model interpreter (`WORKBENCH_LIMN_PYTHON`) and verifies exactly one HMAC ledger key exists (`workbench/docs/PC_PILOT_HANDOFF.md:81,109,113`).
14. GPU runtime verification: Launcher probes container GPU runtime capability before evaluation launch (`workbench/docs/PC_PILOT_HANDOFF.md:81,202-211`, `pc.sh:1114-1122`).
15. Official evaluation dispatch: Launcher enqueues evaluation job via `POST /api/jobs` under audited orchestrator (`workbench/docs/PC_PILOT_HANDOFF.md:81-82`, `pc.sh:1508-1529`).
16. Automatic per-query observation: Observer derives per-query rankings under model interpreter without flags (`workbench/docs/PC_PILOT_HANDOFF.md:82,88-97`).
17. Per-query canonical publication: Publisher validates payload under workbench interpreter and publishes atomic canonical JSON (`workbench/docs/PC_PILOT_HANDOFF.md:82,98-100`).
18. Aggregate metric parsing: Parser extracts native evaluator stdout/stderr into category and macro Recall@K (`workbench/docs/PC_PILOT_HANDOFF.md:82,102-104`).
19. Instrumentation parity verification: Verifier compares per-query rankings to native aggregate hit counts, rejecting artifact on mismatch (`workbench/docs/PC_PILOT_HANDOFF.md:82,120`).
20. Completion proof generation: Runner writes immutable completion proof JSON binding run identity, digests, and return code (`workbench/docs/PC_PILOT_HANDOFF.md:82-83,126-135`).
21. HMAC evidence ledger append: Runner appends chained HMAC-SHA256 authenticated invocation record to `evidence_ledger.jsonl` (`workbench/docs/PC_PILOT_HANDOFF.md:83,130-134`).
22. Strict independent verification: Runner invokes `verify_run.py` to authenticate report, ledger HMAC chain, and proof (`workbench/docs/PC_PILOT_HANDOFF.md:83,212-216`, `pc.sh:1144`).
23. DuckDB index rebuild: System rebuilds `workbench.duckdb` from canonical per-query results (`workbench/docs/PC_PILOT_HANDOFF.md:83,220-221`, `pc.sh:1630-1638`).
24. API run serving: Backend serves canonical per-query runs on `/api/runs` and aggregate-only runs on `/api/benchmark-runs` (`workbench/docs/PC_PILOT_HANDOFF.md:83,101-104,223-228`).
25. UI visualization: Web UI displays aggregate reports in Models tab and per-query runs in Evaluation Runs tab (`workbench/docs/PC_PILOT_HANDOFF.md:83,229-232`).
26. Terminal summary with exit codes: Launcher reports Vietnamese terminal summary with distinct exit codes (0 for pass, 3 per-query export fail, 4 unverified proof, 5 unreadable report, 7 authenticated aggregate lacking proof, 130 signal interrupt) (`workbench/docs/PC_PILOT_HANDOFF.md:105-136`, `pc.sh:49-62,2075-2380`).

## 2. Launch Path
Trace `pc.sh`:
`pc.sh` queues model evaluation via HTTP POST to backend API (`pc.sh:1508-1512`):
```bash
        say "Xếp hàng đánh giá qua API POST /api/jobs..."
        local post_resp
        post_resp="$(curl -s -w '\n%{http_code}' --max-time 60 -X POST "${base}/api/jobs" \
            -H "Content-Type: application/json" \
            -d "{\"model_id\":\"${MODEL}\",\"checkpoint_id\":\"${checkpoint}\",\"protocol_id\":\"${protocol}\",\"top_k\":200}")"
```
Backend API receives POST at `/api/jobs` (`workbench/backend/main.py:1044-1110`):
```python
@app.post("/api/jobs")
def queue_job(payload: JobInput):
...
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}_{uuid.uuid4().hex[:8]}"
    argv = [sys.executable, str(SCRIPTS_DIR / "evaluate_models.py"),
            "--model", plan.model_id, "--checkpoint", plan.checkpoint_id,
            "--protocol", plan.protocol, "--dataset-root", str(plan.dataset_root),
            "--top-k", str(payload.top_k), "--run-id", run_id]
...
    return JOBS.enqueue(plan.model_id, plan.checkpoint_id, plan.protocol, argv, ROOT,
                        job_directory, provenance, refuse_duplicate=True)
```
Queue worker dispatches subprocess (`workbench/backend/jobs.py:539-543`):
```python
    process = subprocess.Popen(job.command, cwd=job.cwd, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               env=_child_environment(job),
                               start_new_session=True)
```
Script invoked: `workbench/scripts/evaluate_models.py` (via jobs queue `SingleGpuQueue`). Not `pipeline.py`.
`pipeline.py` invoked only for preparation (`pc.sh:1096-1102`: `pipeline.py setup`).
API-triggered run uses SAME path: YES. `pc.sh` invokes `POST /api/jobs`. Direct API call hits identical handler in `main.py`, constructs identical `evaluate_models.py` argv, enqueues in `JOBS`.

## 3. Job Lifecycle
In `workbench/backend/jobs.py`:
- Enqueue: `SingleGpuQueue.enqueue()`. Acquires condition lock. Rejects submission with `QueueShutdownError` if stopping. Rejects duplicate active job with `DuplicateJobError` if `refuse_duplicate=True`. Creates `Job` dataclass with status `"queued"`. Persists `queue_state.json` via `_persist()`. Wakes worker via `condition.notify()`. Returns dictionary.
- Execution: Daemon thread worker `_work()` loops. Calls `_next_job()`. Transitions status from `"queued"` to `"preparing"`. Runs `launch_guard` (transitions to `"blocked"` on error). Writes `queue_command.json`. Starts child process with `start_new_session=True`. Transitions status to `"running"`. Captures process group ID. Streams stdout/stderr to log files and in-memory tails. Waits for process exit. Reads `invocation_verdict()`. Settles terminal status via `settled_status()` to `"completed"` or `"failed"` by running `_verify_recorded_run()`. Persists `queue_state.json`.
- Status transitions:
  - `"queued"`
  - `"preparing"`
  - `"running"`
  - `"blocked"`
  - `"completed"`
  - `"failed"`
  - `"cancelled"`
- Log locations:
  - Queue dispatch directory (`<artifacts>/queue_jobs/<uuid>/`): `queue_state.json`, `queue_command.json`, `queue_stdout.log`, `queue_stderr.log`.
  - Run directory (`<artifacts>/logs/<model>__<checkpoint>__<run_id>/`): `stdout.log`, `stderr.log`, `command.json`, `aggregate_report.json`, `invocation_outcome.json`.
- Cancellation: `SingleGpuQueue.cancel(job_id)`. If `"queued"` or `"preparing"`, sets status to `"cancelled"`, persists, returns. If `"running"`, marks status `"cancelled"`, terminates session process group via `_terminate_group()` (SIGTERM with grace period, then SIGKILL). If group survives, writes error to `stderr_tail`. If already finished, raises `ValueError`.
- Shutdown: `SingleGpuQueue.shutdown()`. Sets `_stopping=True`. Cancels queued/preparing jobs with error `"workbench shut down before dispatch"`. Terminates running process groups. Joins worker thread with 10s deadline. Returns surviving job IDs.
- Recovery: `SingleGpuQueue.recover_interrupted(roots)`. Reads all `queue_state.json` on startup. Transitions `"queued"`, `"preparing"`, `"running"` to `"failed"` with error `"workbench restarted while the job was in flight"`. Rewrites file. `adopt_recovered()` loads into in-memory queue.
- `list()` return: Returns list of job dictionaries via `asdict(job)`: `job_id`, `model_id`, `checkpoint_id`, `protocol_id`, `command`, `cwd`, `log_directory`, `provenance`, `status`, `created_at`, `started_at`, `completed_at`, `return_code`, `error`, `stdout_tail`, `stderr_tail`.

Tests exercising each transition:
- `queued -> preparing`: `workbench/tests/test_concurrency_ownership.py:258-285` (`test_concurrency_ownership_worker_launch_guard_cancellation_race`).
- `preparing -> running`: `workbench/tests/test_process_group_cancellation.py:18,69` (`test_sigterm_escalates_to_sigkill_when_group_ignores_termination`) and `workbench/tests/test_phase1_integration_contracts.py:21,41` (`_wait`).
- `preparing -> blocked`: `workbench/tests/test_release_blocker_cuda_readiness.py:421-440` (`refusing_guard` asserts `job["status"] == "blocked"`).
- `running -> completed`: `workbench/tests/test_phase1_integration_contracts.py:40-50` (`test_single_gpu_queue_executes_job_and_settles_completed`) and `workbench/tests/test_release_blocker_invocation_proof.py:244-255` (`test_a_completed_verified_run_is_published_as_completed`).
- `running -> failed`: `workbench/tests/test_phase1_integration_contracts.py:62-72` (`test_single_gpu_queue_records_failure_for_nonzero_exit`), `workbench/tests/test_release_blocker_invocation_proof.py:263-275` (`test_an_unverified_run_is_published_as_failed`), and `workbench/tests/test_release_blocker_adversarial_final.py:303-315` (silent evaluation).
- `queued -> cancelled`: `workbench/tests/test_phase1_integration_contracts.py:243-255` (`test_cancelling_queued_job_settles_immediately_without_running_command`) and `workbench/tests/test_concurrency_ownership.py:329-348`.
- `preparing -> cancelled`: `workbench/tests/test_concurrency_ownership.py:257-285` (`test_concurrency_ownership_worker_launch_guard_cancellation_race`).
- `running -> cancelled`: `workbench/tests/test_process_group_cancellation.py:87-105` (`test_sigterm_escalates_to_sigkill_when_group_ignores_termination`) and `workbench/tests/test_phase1_integration_contracts.py:206-235`.
- `queued/preparing/running -> failed` via `recover_interrupted`: `workbench/tests/test_final_hardening_regressions.py:160-178` (`test_recover_interrupted_flips_phantom_running_to_failed`).
- Transitions that do not exist:
  - `blocked -> running` / `blocked -> completed`: No transition in code or tests (blocked is worker terminal state).
  - `completed -> running` / `failed -> running`: Cannot restart settled job; cancel raises ValueError (`workbench/tests/test_concurrency_ownership.py:380-394`).
  - `launch_guard` failure on job already cancelled: Guarded by `if job.status != "cancelled"` at `jobs.py:520`; no dedicated test.

## 4. Result Persistence
Artifacts written at end of run:
1. `command.json`:
   - Location: `<artifacts>/logs/<model>__<checkpoint>__<run_id>/command.json`
   - Writer: `workbench/scripts/evaluate_models.py:1865`
     ```python
     metadata["invocation_status"] = invocation_status
     (directory / "command.json").write_text(json.dumps(metadata, indent=2) + "
", encoding="utf-8")
     ```
   - Validator: `workbench/scripts/evaluate_models.py:1048-1070` (`validate_run_manifest`). Checks path exists, parses JSON, verifies `invocation_status == "INVOCATION_COMPLETED"`, matches run/model/checkpoint/protocol.
2. Aggregate Report:
   - Location: `<artifacts>/reports/<model>_<checkpoint>_<run_id>_aggregate.json` and copy at `<artifacts>/logs/.../aggregate_report.json`
   - Writer: `workbench/scripts/evaluate_models.py:2002-2010`
     ```python
     _atomic_exclusive_write(run_report, report_json)
     _atomic_replace(directory / "aggregate_report.json", report_json)
     ```
   - Validator: `workbench/scripts/verify_run.py:90-131,271-295,348-420,441-477` and `workbench/scripts/evaluate_models.py:932-975`. Validates identity matching, macro Recall arithmetic, stdout/stderr metric re-extraction, committed benchmark gate.
3. Completion Proof:
   - Location: `<artifacts>/reports/<model>_<checkpoint>_<run_id>_completion.json`
   - Writer: `workbench/scripts/evaluate_models.py:2108-2111`
     ```python
     with proof_path.open("x", encoding="utf-8") as handle:
         handle.write(json.dumps(proof, indent=2) + "
")
     ```
   - Validator: `workbench/scripts/evaluate_models.py:679-1087` (`validate_run_manifest`), called by `verify_run.py:175-181`. Validates file presence, non-zero size, containment, return code 0, checkpoint digest, bundle member digests, report digest, and ledger entry match.
4. Evidence Ledger Entry:
   - Location: `<artifacts>/evidence_ledger.jsonl`
   - Writer: `workbench/scripts/evaluate_models.py:1413-1433, 2015-2027` (`append_evidence_entry`). Appends HMAC-SHA256 authenticated JSON line under exclusive `fcntl.LOCK_EX` lock.
   - Validator: `workbench/scripts/evaluate_models.py:1351-1398` (`_ledger_entries_unlocked`) and `workbench/scripts/verify_run.py:126-155`. Validates UTF-8, trailing newline, entry sequence, continuous HMAC chain, matching report digest, and return code 0.
5. Invocation Outcome:
   - Location: `<artifacts>/logs/<model>__<checkpoint>__<run_id>/invocation_outcome.json`
   - Writer: `workbench/scripts/evaluate_models.py:2030,2046,2054` (`record_invocation_outcome`)
   - Validator: `workbench/backend/jobs.py:84-122` (`invocation_verdict`).

Tests using REAL files rather than mocks:
- `workbench/tests/test_release_blocker_evaluation_output.py:35-70` (`test_e2e_successful_evaluation_manifest_validates`): Creates real temporary disk files, computes real SHA-256 digests, tests `validate_run_manifest`.
- `workbench/tests/test_release_blocker_invocation_proof.py:140-220` (`test_an_invalidated_recorded_run_refuses_verification_and_blocks_publication`): Writes real disk artifacts, appends real HMAC ledger entries, exercises `verify_run.verify()`.
- `workbench/tests/test_e2e_release_chain.py:309-375` (`test_the_real_chain_produces_a_certified_result_then_refuses_a_tampered_one`): Writes real weights and reports to disk, appends real HMAC ledger, performs file tampering and recovery tests.
- `workbench/tests/test_true_master_e2e.py:182-250` (`test_true_master_pipeline_subprocess_e2e_complete_flow`): Full subprocess execution of `pipeline.py all`, writes real report, proof, and ledger files to disk.
- `workbench/tests/test_final_hardening_regressions.py:140-158`: Rejects forged completion proofs on disk without mocks.

## 5. Index and UI Visibility
How a completed run becomes visible:
- DuckDB index rebuild: System invokes `rebuild_index()` in `workbench/backend/index.py:161-199` (via `POST /api/index/rebuild` or script).
- Index builder scans `WORKBENCH_RESULTS_ROOT` via `result_files()` and `load_runs_with_digests()`.
- Reads canonical `ResultRun` per-query JSON artifacts, extracts `run`, `queries`, `top_results`.
- Obtains verifier receipts from `verifier_verdicts()` via `_aggregate_reports()`.
- Evaluates admission policy and verifier promotion via `evidence_status()`.
- Populates DuckDB tables: `runs`, `queries`, `top_results`. Replaces database file atomically (`os.replace`).
- Endpoint `/api/runs` executes `SELECT metadata, evidence_status FROM runs ORDER BY run_id` (`workbench/backend/storage.py:30-38`), powering UI Evaluation Runs tab.
- Endpoint `/api/benchmark-runs` reads all `<artifacts>/reports/*_aggregate.json` files via `_aggregate_reports()` (`workbench/backend/main.py:862-872`), powering UI Models tab.

Quote index builder (`workbench/backend/index.py:161-182`):
```python
def rebuild_index(results_root: Path | None = None, database_path: Path | None = None) -> int:
    import duckdb

    database_path = database_path or configured_database_path()

    loaded = load_runs_with_digests(results_root)
    runs = [result for _, result in loaded]
    digests = {result.run.run_id: digest for digest, result in loaded}
    verdicts = verifier_verdicts()
...
        for result in runs:
            run = result.run
            rows["runs"].append([run.run_id, run.protocol_id, run.model_id, run.checkpoint_id, run.checkpoint_training_noise_pct, run.data_kind, run.top_k_saved, run.gallery_size, json.dumps(run.category_gallery_sizes) if run.category_gallery_sizes else None, json.dumps(run.category_eligible_counts) if run.category_eligible_counts else None, evidence_status(result, verdicts.get(run.run_id), digests.get(run.run_id)), run.reproduced_metrics.r10, run.reproduced_metrics.r50, run.reproduced_metrics.mean, run.metadata])
```

Quote visibility filter (`workbench/backend/index.py:102-118`, `workbench/backend/schemas/results.py:240-247, 321-360`):
`workbench/backend/index.py:102-118`:
```python
def evidence_status(result: ResultRun, verifier_verdict: dict | None = None,
                    artifact_digest: str | None = None) -> str:
    from workbench.backend.schemas.results import admission, promote

    state, _ = promote(result, *admission(result)[:1], verifier_verdict,
                       artifact_digest=artifact_digest)
    return state
```
`workbench/backend/schemas/results.py:240-247`:
```python
def admission(result: "ResultRun") -> tuple[str, list[str]]:
...
    run = result.run
    if run.data_kind == "mock":
        return SYNTHETIC_OR_MOCK, [f"data_kind is {run.data_kind!r}: development data, never evidence"]
    if run.data_kind != "experiment":
        return PER_QUERY_UNVERIFIED, [
            f"data_kind is {run.data_kind!r}: a declared self-test, not evidence about reproduction"]
```
`workbench/backend/schemas/results.py:321-340`:
```python
def promote(result: "ResultRun", state: str, verifier_verdict: dict | None,
            artifact_digest: str | None = None) -> tuple[str, list[str]]:
...
    if not all(verifier_verdict.get(key) is True for key in ("certified", "is_research_evidence")):
        return state, reasons
    recorded = verifier_verdict.get("per_query_artifact_digest")
    claimed = artifact_digest
    if not recorded or not claimed:
        return state, ["the verifier's receipt does not bind a per-query artifact digest, so this "
                       "artifact cannot be attributed to the verified run"]
    if recorded != claimed:
        return state, [f"the artifact digest {claimed[:16]}... is not the digest the verified run "
                       f"recorded ({recorded[:16]}...); this artifact is not the one that was checked"]
    return PER_QUERY_VERIFIED, reasons
```

Which models can appear in `/api/runs` and why:
ONLY `limn`.
Reason: LIMN is sole registered model with implemented per-query observer and publisher (`limn_export.py`, `limn_observer.py`, `limn_publish.py`). Remaining 15 models (`encoder`, `hint`, `pair`, `clvc_net`, `tgcir`, `habit`, `intent`, `conesep`, `airknow`, `dcnet`, `csmcir`, `combiner_rn50x4_noft`, `clip4cir_rn50x4_fullft`, `sprc`, `ptha_mtst`) have `per_query_capability: null` (`CAPABILITY_ABSENT`), producing zero per-query JSON files in `WORKBENCH_RESULTS_ROOT`. DuckDB index only ingests `WORKBENCH_RESULTS_ROOT` JSON files. Aggregate-only model runs appear exclusively in `/api/benchmark-runs` and UI Models tab.

## 6. Recovery
1. Interrupted run:
   - Behavior: Launcher traps `SIGINT`/`SIGTERM` in `pc.sh:49-62` (`cleanup_on_signal`), exits 130, preserves artifacts, logs, and downloads without deletion. Backend startup executes `SingleGpuQueue.recover_interrupted(roots)` (`workbench/backend/jobs.py:488-513`), scans `queue_state.json`, rewrites active `"queued"`, `"preparing"`, `"running"` jobs to `"failed"` with error `"workbench restarted while the job was in flight"`.
   - Test: `workbench/tests/test_final_hardening_regressions.py:160-178` (`test_recover_interrupted_flips_phantom_running_to_failed`) and `workbench/tests/test_pc_launcher.py:657-675` (`test_signal_handling_preserves_artifacts_and_exits_130`).
2. Resume:
   - Behavior: In `pc.sh:1285-1475`, launcher queries `/api/runs` and `/api/benchmark-runs` for matching `(model_id, checkpoint_id, protocol_id)`. Verifies verdict with `run_verifier_in_container`. If `is_research_evidence` holds, skips evaluation:
     `say "Phát hiện kết quả đánh giá đã hoàn thành và được sổ cái chứng thực: run_id=${target_run_id}"`
     `say "Tái sử dụng kết quả có sẵn (sử dụng cờ --force nếu muốn buộc đánh giá lại)."`
     `FINAL_RUN_ID="$target_run_id"`; returns 0.
     In `pipeline.py`: `--resume` skips verified stages; re-runs stage if input checkpoint or dataset hash changed.
   - Test: `workbench/tests/test_pc_launcher.py:557-580` (`test_job_dedup_skips_post_when_corroborated_run_exists`), `workbench/tests/test_true_master_e2e.py:209-250` (resume skip and mutation invalidation), and `workbench/tests/test_autopilot_cold_start.py:579-620`.
3. Stale cache:
   - Behavior:
     - In `workbench/backend/main.py:258-277, 434-438, 477-480`: `_CERTIFICATION_CACHE` keyed on `(report digest, certification mode, ledger stamp, proof stamp, checkpoint stamp, run-artifacts stamp)`. Member digests memoized on `(path, ctime_ns, mtime_ns, size, ino)`. Editing any member or log file invalidates cache key, refusing stale certificates.
     - In `workbench/backend/runtime.py:1014-1016, 1265-1271`: `_cached_gpu_capability` validates `dependency_fingerprint`, `host_identity`, `interpreter_fingerprint`. Environment changes invalidate cached capability.
     - In `workbench/replay/limn.py:104` and `hint.py:112`: `isolated_cache_root` generates isolated temporary directories, refusing stale pickle caches.
   - Test: `workbench/tests/test_e2e_release_chain.py:346-375` (checkpoint member byte modification invalidates certificate; restoring byte restores certificate) and `workbench/tests/test_release_blocker_adversarial_final.py:47-49` (`test_a_same_mtime_rewrite_cannot_keep_an_old_member_digest`).
4. Duplicate run id:
   - Behavior:
     - In `workbench/scripts/evaluate_models.py:1827-1830`: `if run_report.exists() or proof_path.exists(): raise RuntimeError(f"refusing to overwrite existing run artifacts for run id {run_id}")`.
     - In `workbench/scripts/evaluate_models.py:1843-1849` and `1420-1422`: Checks `read_evidence_ledger(config)`. Raises `LedgerDuplicateRunError("evidence ledger already records this run id")`.
     - In `workbench/backend/main.py:1070-1077`: Rejects submission with HTTP 409 `duplicate_job` if identical job active.
   - Test: `workbench/tests/test_phase1_acceptance_scenarios.py:239-269` (`test_scenario_e_duplicate_run_id_never_overwrites_a_run`) and `workbench/tests/test_final_hardening_regressions.py:323-337` (`test_enqueue_with_refuse_duplicate_rejects_colliding_active_job`).
5. Cancelled job:
   - Behavior:
     - In `workbench/backend/jobs.py:237-277` (`cancel(job_id)`): If queued or preparing, sets status to `"cancelled"`, persists. If running, sets status to `"cancelled"`, terminates session process group via `_terminate_group()` (SIGTERM with grace period, then SIGKILL). Records warning if processes linger.
     - In `SingleGpuQueue._work()`: Worker checks cancellation before and after execution; never transitions cancelled job to `"completed"`, never writes completion proof.
   - Test: `workbench/tests/test_process_group_cancellation.py:87-105` (`test_sigterm_escalates_to_sigkill_when_group_ignores_termination`), `workbench/tests/test_process_group_cancellation.py:173-186` (`test_worker_does_not_publish_completed_for_cancelled_job`), and `workbench/tests/test_phase1_integration_contracts.py:206-235`.

## 7. Coverage Gaps

| # | Workflow | Boundary Crossed | Named Test (`file:line`) | Details / Gap |
|---|---|---|---|---|
| 1 | Fresh setup | Real subprocess (`git clone`, `run.sh autopilot`) | `workbench/tests/test_fresh_clone_delivery.py:48` and `workbench/tests/test_autopilot_cold_start.py:435` | Clones repo in isolated subprocess, verifies HEAD pin and git tracking; runs cold start with controlled fake conda. Real Docker host spin-up unexercised in fresh setup tests. |
| 2 | Readiness | None (no real subprocess/socket) | no test | Polling `/api/jobs` and frontend port tested via in-process ASGI (`test_http_api_layer.py`) and mock curl shims (`test_pc_launcher.py`). Zero tests run real HTTP server + client across network socket boundary. |
| 3 | DCNet | None (in-process preflight/parser only) | no test | Preflight, dataset layout, and output parsing tested in-process (`test_dcnet_replay.py:19`, `test_r9_defect_regressions.py:20`). Zero tests launch DCNet CLI subprocess or execute DCNet via API. |
| 4 | LAVIS family | None (in-process preflight/processors only) | no test | CSMCIR, SPRC, HINT, Air-Know, ConeSep, HABIT, INTENT tested in-process (`test_hint_replay.py:112`, `test_csmcir_source_blocker.py:1`, `test_blip2_text_processor_parity.py:1`). Zero tests launch LAVIS model subprocesses or execute via API. |
| 5 | Launch + status | Real API + Subprocess queue | `workbench/tests/test_phase1_acceptance_scenarios.py:87` | Calls `queue_job`, dispatches real `evaluate_models.py` subprocess, observes status transition `queued -> preparing -> running -> completed`. Real pipeline CLI tested in `test_true_master_e2e.py:182`. |
| 6 | Result handling | Subprocess to disk artifacts | `workbench/tests/test_phase1_acceptance_scenarios.py:87` and `workbench/tests/test_true_master_e2e.py:182` | Subprocess generates `command.json`, aggregate report, completion proof, and HMAC ledger entry; verifier validates real disk files. |
| 7 | Failure handling | Subprocess exit code to ledger | `workbench/tests/test_phase1_acceptance_scenarios.py:138` and `workbench/tests/test_phase1_acceptance_scenarios.py:164` | Subprocess exit code 3 recorded as failure; exit 0 with empty output rejected without proof. |
| 8 | Recovery | Subprocess group kill & resume | `workbench/tests/test_process_group_cancellation.py:87` and `workbench/tests/test_true_master_e2e.py:182` | Real process group SIGTERM/SIGKILL termination; real `pipeline.py --resume` mutation invalidation across subprocess runs. |
| 9 | Real model execution | None (synthetic/controlled models only) | no test | Zero real models (LIMN, CSMCIR, HINT, etc.) executed with real weights or GPU in test suite. Every execution test uses `synthetic_model` or controlled test fixtures. |
| 10 | Remote handoff | None (static docs/git checks only) | no test | `test_handoff_commands.py:161` audits document command strings and git commit equality. Zero tests cross remote machine, SSH, or remote container boundary. |

## Verified claims

The following claims were verified by direct command execution on this repository:

1. Launcher Help Interface:
   - Command: `./pc.sh --help`
   - Observed Output:
     ```
     Usage: ./pc.sh [OPTIONS]

     One-command PC entrypoint for the CIR Failure Analysis Workbench.
     Orchestrates preflight, configuration, Docker stack, model preparation,
     GPU runtime verification, evaluation, validation, and Vietnamese summary.

     Options:
       --model <id>       Model to evaluate (default: limn; e.g. csmcir)
       --doctor           Run host and runtime preflight checks only (read-only)
       --verify-only      Verify the model's GPU RUNTIME inside the container and stop. It does NOT
                          verify a recorded run, does NOT rebuild the DuckDB index, and does NOT
                          download checkpoints. To verify an existing run and refresh the index, run
                          ./pc.sh without this flag: it reuses a corroborated run instead of
                          re-evaluating.
       --force            Force re-evaluation even if completed corroborated run exists
       --timeout <sec>    Maximum wait time for evaluation job in seconds (default: 3600)
       --no-gpu           Disable GPU override (WORKBENCH_ENABLE_GPU=0)
       --dataset <path>   Override FashionIQ dataset path
       --checkpoint <id>  Pin one exact checkpoint or bundle id. Use to run an INTENTIONAL
                          single-category evaluation; omit it to run the model's declared bundle,
                          which refuses by name when any member is missing rather than silently
                          narrowing itself to one category.
       --self-check       Run internal self-tests for helper functions and exit
       -h, --help         Show this help message and exit
     ```
   - Citation: `pc.sh:196-224`.

2. Fresh Clone Delivery Test Suite:
   - Command: `.venv/bin/python -m pytest workbench/tests/test_fresh_clone_delivery.py -q -p no:cacheprovider`
   - Observed Output:
     ```
     ..........                                                               [100%]
     10 passed in 5.77s
     ```
   - Citation: `workbench/tests/test_fresh_clone_delivery.py:48-65`.

3. Queue Concurrency and Ownership Test Suite:
   - Command: `.venv/bin/python -m pytest workbench/tests/test_concurrency_ownership.py -q -p no:cacheprovider`
   - Observed Output:
     ```
     ...............                                                          [100%]
     15 passed in 4.57s
     ```
   - Citation: `workbench/tests/test_concurrency_ownership.py:140-455`.

4. Process Group Cancellation Escalation Test Suite:
   - Command: `.venv/bin/python -m pytest workbench/tests/test_process_group_cancellation.py -q -p no:cacheprovider`
   - Observed Output:
     ```
     ......                                                                   [100%]
     6 passed in 60.20s (0:01:00)
     ```
   - Citation: `workbench/tests/test_process_group_cancellation.py:87-195`.

5. Phase I API and Subprocess Acceptance Scenarios:
   - Command: `.venv/bin/python -m pytest workbench/tests/test_phase1_acceptance_scenarios.py -q -p no:cacheprovider`
   - Observed Output:
     ```
     ........                                                                 [100%]
     8 passed, 20 warnings in 24.73s
     ```
   - Citation: `workbench/tests/test_phase1_acceptance_scenarios.py:87-400`.

6. Release Blocker Evaluation Output Validation:
   - Command: `.venv/bin/python -m pytest workbench/tests/test_release_blocker_evaluation_output.py -q -p no:cacheprovider`
   - Observed Output:
     ```
     ........................................................................ [100%]
     72 passed in 23.09s
     ```
   - Citation: `workbench/tests/test_release_blocker_evaluation_output.py:35-1000`.

7. Startup Interrupted Job Recovery:
   - Command: `.venv/bin/python -m pytest workbench/tests/test_final_hardening_regressions.py -k "test_recover_interrupted_flips_phantom_running_to_failed" -q -p no:cacheprovider`
   - Observed Output:
     ```
     .                                                                        [100%]
     1 passed, 44 deselected in 0.16s
     ```
   - Citation: `workbench/tests/test_final_hardening_regressions.py:160-178`.

8. End-to-End Release Chain and Checkpoint Tampering:
   - Command: `.venv/bin/python -m pytest workbench/tests/test_e2e_release_chain.py -q -p no:cacheprovider`
   - Observed Output:
     ```
     .                                                                        [100%]
     1 passed, 4 warnings in 3.47s
     ```
   - Citation: `workbench/tests/test_e2e_release_chain.py:309-375`.

9. True Master Subprocess Pipeline Flow and Resume:
   - Command: `.venv/bin/python -m pytest workbench/tests/test_true_master_e2e.py -q -p no:cacheprovider`
   - Observed Output:
     ```
     ...                                                                      [100%]
     3 passed in 27.21s
     ```
   - Citation: `workbench/tests/test_true_master_e2e.py:182-250`.

10. LIMN Query Export Publication API:
    - Command: `.venv/bin/python -m pytest workbench/tests/test_limn_query_export_api.py -q -p no:cacheprovider`
    - Observed Output:
      ```
      ...............                                                          [100%]
      15 passed, 4 warnings in 1.38s
      ```
    - Citation: `workbench/tests/test_limn_query_export_api.py:27-150`.

11. DCNet Replay Preflight:
    - Command: `.venv/bin/python -m pytest workbench/tests/test_dcnet_replay.py -q -p no:cacheprovider`
    - Observed Output:
      ```
      ....                                                                     [100%]
      4 passed in 0.10s
      ```
    - Citation: `workbench/tests/test_dcnet_replay.py:19-50`.

12. CSMCIR Source Blocker Validation:
    - Command: `.venv/bin/python -m pytest workbench/tests/test_csmcir_source_blocker.py -q -p no:cacheprovider`
    - Observed Output:
      ```
      ....                                                                     [100%]
      4 passed in 0.02s
      ```
    - Citation: `workbench/tests/test_csmcir_source_blocker.py:1-40`.

13. Summary Exit Codes Agreement:
    - Command: `.venv/bin/python -m pytest workbench/tests/test_pc_launcher.py -k "test_the_summary_exit_codes_match_the_handoff_document" -q -p no:cacheprovider`
    - Observed Output:
      ```
      .                                                                        [100%]
      1 passed, 99 deselected in 0.04s
      ```
    - Citation: `workbench/tests/test_pc_launcher.py:1851-1861`.

14. Autopilot Cold Start and Controlled Fake Conda Labeling:
    - Command: `.venv/bin/python -m pytest workbench/tests/test_autopilot_cold_start.py -k "test_controlled_fake_package_manager_is_labeled" -s -q -p no:cacheprovider`
    - Observed Output:
      ```
      [PACKAGE_MANAGER_EXERCISED: CONTROLLED_FAKE_CLI; REAL_CONDA: NOT_EXERCISED]
      .
      1 passed, 9 deselected in 0.48s
      ```
    - Citation: `workbench/tests/test_autopilot_cold_start.py:1012-1018`.

15. Autopilot Cold Start Top-Level Command Flow:
    - Command: `.venv/bin/python -m pytest workbench/tests/test_autopilot_cold_start.py -k "test_cold_start_runs_top_level_command_without_prior_environments" -s -q -p no:cacheprovider`
    - Observed Output:
      ```
      [PACKAGE_MANAGER_EXERCISED: CONTROLLED_FAKE_CLI]
      [GPU_EVIDENCE: NOT_EXERCISED; no real CUDA claim]
      .
      1 passed, 9 deselected in 20.51s
      ```
    - Citation: `workbench/tests/test_autopilot_cold_start.py:435-505`.

16. Unexecuted Runtime Claims:
    - `[INFERENCE]` Full GPU inference on RTX 5070 Ti: unexecuted. No GPU certificate exists in repo (`workbench/docs/PC_PILOT_HANDOFF.md:15-18,290-297`).
    - `[INFERENCE]` Target PC Docker stack initialization: unexecuted. Host lacks NVIDIA container runtime for RTX 5070 Ti in dev environment.
    - `[INFERENCE]` 15 aggregate-only model execution: unexecuted. Checkpoints unstaged, author assets missing, or preflight-only.
