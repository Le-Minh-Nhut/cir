# A38 W8 — cross-module integration, concurrency, recovery

## job state machine

Command: `PYTHONPATH=. .venv/bin/python /dev/shm/a38-w8/queue_probe.py` (direct `SingleGpuQueue` calls; Python sleep/exit-only children; no model evaluation). Raw results:

```json
{
  "success": {"sequence": [[1793452898.116016,"queued"],[1793452898.126115,"running"],[1793452898.1564126,"completed"]], "final":"completed", "return_code":0},
  "cancel": {"returned":"cancelled", "after_natural_end":"cancelled", "return_code":-15},
  "failure": {"sequence": [[1793452899.61744,"queued"],[1793452899.6279774,"running"],[1793452899.6482305,"failed"]], "status":"failed", "return_code":7,"error":null},
  "concurrent": {"ids":["f033ac39-968e-4279-b1fb-b5e171abad08","f0a9a69a-111c-4cdb-8474-529b029f67c9","400484b7-0be9-4b14-98f0-37528bbbde9e","d664e3cd-a23b-44e9-a675-8ce8bdf52788","63921896-e11b-497f-bdfd-b3bcad36df49"],"errors":[],"final_states":["completed","completed","completed","completed","completed"]},
  "stale": {"sequence":[[1793452899.8228178,"queued"],[1793452899.8329608,"running"],[1793452899.8532918,"failed"]],"status":"failed","code":9,"outcome":{"run_id":"old","verdict":"PROOF_VALIDATED"}}
}
```

Status: **PASS** for the executed cases. Exit code 7 is surfaced as `failed` with return_code 7; no error text is expected for ordinary nonzero exit. Terminal determination is based on `process.wait()`'s actual code plus the run-bound invocation verdict and independent verifier, not stale artifacts. `settled_status` in `workbench/backend/jobs.py:143-157` makes a named run require matching verdict, independent verification; no-run bookkeeping uses `code != 0` → failed. `invocation_verdict` at `:115-123` rejects missing or different `run_id`. The stale attempt's old valid-looking verdict remained untouched and did not adopt as success.

Cancellation: returned and later observed `cancelled`, return code -15. In `jobs.py:263-276` cancel marks state under condition lock and removes process handle before teardown; worker finalization checks `job.status != "cancelled" and job.job_id not in self._cancelling` under that same lock at `:607-621`. This is the observed overwrite prevention; probe waited beyond the child's natural sleep, though cancel's SIGTERM terminated it early.

Concurrent submission: five thread-synchronized enqueues, distinct IDs above, no exceptions/500-equivalent; every `queue_state.json` parsed as JSON and had terminal `completed`. The queue's in-process `Condition` lock protects enqueue and `_persist` (see `jobs.py:216-233`, `:463-486`).

Stale output adoptability: precreated `invocation_outcome.json` declared `PROOF_VALIDATED` for `run_id=old`; failed job was `failed` with code 9. `jobs.py:120-123` requires exact `run_id` match; `settled_status` at `:143-152` only accepts verified completion proof.

Cancellation: initial probe returned `cancelled` and later observed `cancelled`, return code -15; that initial ordinary cancel terminated the child early. A second controlled in-process probe made the queue's termination helper wait instead of signal, then let the fake child naturally exit 0 after printing. Exact command: `PYTHONPATH=. .venv/bin/python /dev/shm/a38-w8/cancel_natural.py`. Raw: `cancel_return cancelled terminate_wait_elapsed 0.5648927688598633`; `after_child_natural_exit cancelled return_code 0 stdout 'natural-end\n'`. **PASS** for the required late-natural-exit no-overwrite condition. In `jobs.py:263-276` cancel marks state under condition lock and removes process handle before teardown; worker finalization checks `job.status != "cancelled" and job.job_id not in self._cancelling` under that same lock at `:607-621`.

Same direct queue experiment: fake child held in running; killed with `Popen.kill()` and waited (`poll=-9`); called `SingleGpuQueue.recover_interrupted([root])` on its persisted queue record.

Before:
```json
{"job_id":"d16ceaea-e722-4a42-b1d0-610913a1df36","model_id":"recover","checkpoint_id":"ck","protocol_id":"p","status":"running","started_at":"2026-10-31T13:21:39.853731+00:00","completed_at":null,"return_code":null,"error":null}
```
After:
```json
{"job_id":"d16ceaea-e722-4a42-b1d0-610913a1df36","model_id":"recover","checkpoint_id":"ck","protocol_id":"p","status":"failed","started_at":"2026-10-31T13:21:39.853731+00:00","completed_at":"2026-10-31T13:21:39.865215+00:00","return_code":null,"error":"workbench restarted while the job was in flight"}
```
Raw recovery return: one record with status `failed` and same restart reason. **PASS**. Routine verified as `recover_interrupted` (`jobs.py:488-514`); it rewrites queued/preparing/running as failed. `main.py:41-58` invokes it on startup and adopts recovered failures for API visibility.

## concurrency on index / pipeline state / capability record

Command: `PYTHONPATH=. .venv/bin/python /dev/shm/a38-w8/concurrency_probe.py`.

- **Index: EXPECTED_BEHAVIOR (atomic publication); executed** two barrier-synchronized processes rebuilt same empty result root into same DuckDB path. Raw: `INDEX {"exitcodes":[0,0],"workers":["OK:0","OK:0"],"integrity_run_count":0}`. Each rebuild uses a unique `.{database}.{uuid}.tmp`, creates/builds independent DuckDB there, closes, then `os.replace(temporary, database_path)` (`workbench/backend/index.py:184-212`). DuckDB reopened read-only and queried `runs`; count 0. This demonstrates valid concurrent publication for empty index, not non-empty data merge. Expected last successful replacement wins. No fixed shared temp DB.
- **Pipeline state: EXPECTED_BEHAVIOR (exclusive workflow lock); executed** two actual `run_stages(... resume=True)` invocations synchronized; one ran to completion, other got `WorkflowLockError`. Raw: `SYNC {"exitcodes":[0,0],"results":["EXIT:0","LOCK:another workflow owns /dev/shm/a38-w8/pipeline/sync.json (lock held); retry later"],"state":{"stages":{"slow":{"status":"COMPLETE",...}}}}`. `acquire_workflow_lock` uses exclusive nonblocking `fcntl.flock` (`pipeline.py:715-743`); `run_stages` acquires before load/write (`:2152-2156`) and releases in finally (`:2420-2423`). State writes use PID-specific temp and replace (`:771-777`), but lock is the cross-process guard. No interleaved writes; only one invocation accepted.
- **GPU capability: EXPECTED_BEHAVIOR (atomic same-key replacement); executed** two barrier-synchronized processes called real `record_gpu_capability` on forced identical key. Raw: `CAPABILITY {"exitcodes":[0,0],"record":{"value":"B","model_id":"x","capability_key":"same-key",...}}`. Record parsed as valid JSON, value B (last writer measured). Writer makes per-process temp and atomically `os.replace`s at `runtime.py:1131-1139`; no partial record. This permits last-writer-wins for concurrent probes of same key; contents represent complete one probe. Test used reduced payload, disabled key/signature functions solely to force same key without hardware probing.

## failure propagation

Command: `PYTHONPATH=. .venv/bin/python /dev/shm/a38-w8/pipeline_probe.py`.

Raw: stage `bad` child exited 17; downstream stage logged `SKIP ... SKIPPED_DEPENDENCY`; JSON `{"exit":1,"records":{"bad":"FAILED","downstream":"SKIPPED_DEPENDENCY"},"downstream_artifact_exists":false}`. **PASS**. `run_stages` marks nonzero command `FAILED` (`pipeline.py:2342-2355`), sets failed dependency key, blocks dependent stages (`:2188-2243`) and exits 1 (`:2425`).

Model adapter malformed/empty feature data → valid rank file: **STATICALLY_CHECKED**. No actual adapter/model evaluated (explicitly avoided heavy evaluation). Parent export path rejects nonzero observer/publisher, missing staged/published path, unreadable/noncanonical JSON, mismatched run identity, then verifies plan identity and bundle digest (`evaluate_models.py:1896-1980`). It compares per-query rank-derived hit counts with aggregate counts and refuses disagreement (`:1981-2043`). Cannot claim runtime result for empty/malformed features without invoking an actual model adapter. No demonstrated concealment.

Broad exception review (`workbench/backend/main.py` and `jobs.py`):
- `main.py:55` recovery catches broad error and returns normally from startup. This can conceal recovery inability, but startup comment says disk records remain available; no poisoned-dependency handler/API call was executed to demonstrate a 2xx/default response. **STATICALLY_CHECKED; not counted as confirmed concealment.**
- `main.py:84` startup index build records/logs exception (`_index_build_error`); not a silent success. `:345`, `:382`, `:406` convert stamp/path resolution faults to `unresolved`/None; `:537` logs verifier exception and returns None. These paths fail closed for certification; no 2xx empty-success concealment demonstrated.
- `jobs.py:176`, `:180` verifier import/run failures return failed verification reason; fail closed. `:535` launch guard exception marks `blocked`. `:622` worker exception marks `failed` unless cancellation already won. `:628` only suppresses secondary command-metadata write failure while preserving primary failed job. No demonstrated concealment.

## findings

None confirmed in this workstream. No `W8-xx` defect blocks reported; observed concurrent same-key capability behavior is atomic last-writer-wins by design and index publication is atomic.

## blockers
- Non-empty index rebuild race not exercised: repository experiment used empty results root to avoid consuming production artifacts; index integrity measured as valid DuckDB with zero runs. Atomic unique-temp + replace statically establishes no shared-temp truncation, but concurrent non-empty table content equivalence remains **STATICALLY_CHECKED**.
- GPU capability concurrency qualification: probe executed with synthetic reduced payloads and disabled HMAC signing to force a key collision and observe last-writer replacement; full-payload signed concurrency was not exercised.
- Model adapter malformed/empty feature behavior not executed: requires running actual model adapter/evaluation; intentionally excluded to comply with no heavy evaluation. Parent validation is **STATICALLY_CHECKED** only.
- Broad-exception handler poisoning not demonstrated through API request context; source review shows listed broad catches either log/retain error state or produce fail-closed values. No demonstrated 2xx concealment.

## files written + scratch size

Written: `workbench/docs/reviews/a38/A38_FINDINGS_PARTIAL_W8.md`.
Scratch scripts/results were under `/dev/shm/a38-w8/` (peak reported size 428K). Cleaned after recording results; final size 0 bytes.

Tree note: at start, other workstreams had modified `workbench/composition/model_csmcir.py`, `workbench/composition/runner.py`, `workbench/tests/test_composition_image_identity.py`, `workbench/tests/test_pipeline.py`, `workbench/tests/test_release_invariants.py`; untracked operator files `pc_composition_smoke.yaml`, `src/train_cirr.py`, `uv.lock`; untracked `workbench/tests/test_csmcir_checkpoint.py`; and existing `workbench/docs/reviews/a38/`. None were touched by this task. Only `workbench/docs/reviews/a38/A38_FINDINGS_PARTIAL_W8.md` was written in repo.
