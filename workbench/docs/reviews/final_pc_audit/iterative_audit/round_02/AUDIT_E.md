# AUDIT_E — Round 02 (slice E backend, queue, autopilot, frontend)

Target: `workbench/backend/{main,jobs,index,storage,analysis,autopilot}.py`, `workbench/frontend/` (TS sources).
Branch: `research/cir-failure-workbench` at `92ecbd9` + 3 uncommitted fix waves.
Method: Direct code audit, ASGI application execution via asyncio runner, SingleGpuQueue live harness, runtime capability key derivation analysis, targeted pytest execution (`test_http_api_layer.py` 40 passed, `test_concurrency_ownership.py` 15 passed, `test_process_group_cancellation.py` 6 passed).
Classification: CONFIRMED / PROBABLE / FALSE_POSITIVE / EXPECTED_BEHAVIOR / EXTERNAL_DEPENDENCY_PENDING / KNOWN_LIMITATION; severity P0–P3.

---

## 1. Resolution of Round-01 Open Leads

### A7 / F4 — `proof_unavailable` and development certification without proof
- **Location**: `workbench/backend/main.py:225-245, 332-359, 406-409`; `pc.sh:1232-1244, 1286-1290, 1488-1515`.
- **Classification**: CONFIRMED, severity P1 (policy discrepancy / UI false PASS).
- **Mechanism**:
  1. `main.py:240-244`: when proof absent in production, `_certification_mode` returns `"proof_unavailable"`. `main.py:406-409` sets `certified=False, is_research_evidence=False, state="proof_unavailable"`. API behaves correctly: refuses to certify.
  2. `pc.sh:1238, 1286-1290`: launcher recognizes `proof_unavailable` + ledger corroboration and reuses run (`FINAL_RUN_PROOF_UNAVAILABLE=1`). In `pc.sh:1488-1515`, missing proof is bypassed, printing `"KẾT QUẢ ĐÁNH GIÁ CHÍNH THỨC"` and exiting 0.
  3. `workbench/docker/release_verify.sh:758`: release verifier requires `_v.get("is_research_evidence") is True`. Because API set it to False, `release_verify.sh` marks `MODEL_EVALUATION` and `SCIENTIFIC_OUTPUT` as `PENDING`.
  4. `main.py:332-359` (F4): with `WORKBENCH_DEVELOPMENT_MODE=1` and `WORKBENCH_ALLOW_DEVELOPMENT_CERTIFICATES=1`, `_development_certification` returns `certified=True, is_research_evidence=True` with no completion proof file on disk. Restrained to development processes only (`_development_process()` requires pytest or dev flag).
- **Observed behavior**: Launcher considers reused run successful, API marks it uncertified, release verifier marks it PENDING, and frontend UI renders `PASS_WITHIN_5_PERCENT` without refusal notice.

### A9 — `OPERATIONS PENDING` on empty queue yields `ENGINEERING_INCOMPLETE`
- **Location**: `workbench/docker/release_verify.sh:364-378, 531, 811-820`; `workbench/backend/jobs.py:505`; `workbench/backend/main.py:96-100`.
- **Classification**: CONFIRMED, severity P2 (false release blocker).
- **Mechanism**:
  1. `release_verify.sh:372-374`: J8 queries `GET /api/jobs`. If `not jobs`, raises `SystemExit(3)` (`NO_JOBS_OBSERVED`) -> J8 status `PENDING`.
  2. `release_verify.sh:531`: `OPERATIONS` dimension requires check `["J8"]` and `mandatory=True`. With J8 `PENDING`, `OPERATIONS` becomes `PENDING`.
  3. `jobs.py:505` and `main.py:96-100`: On container/process startup, `recover_interrupted` only reads jobs in `("queued", "preparing", "running")` and adopts them as `failed`. Completed jobs (`"completed"`) are NEVER adopted into memory.
  4. On any clean restart of backend, `SingleGpuQueue._jobs` is empty (`[]`). Running `release_verify.sh` against healthy finished host sees empty queue -> `OPERATIONS: PENDING` -> mandatory pending -> `verdict: ENGINEERING_INCOMPLETE` (`release_verify.sh:818`).
- **Verdict**: False failure on healthy finished PC unless operator manually triggers dummy job.

### B-1 / A5 — `_gpu_capability_key` derives from `os.environ` only, ignoring `environment_assignment.json`
- **Location**: `workbench/backend/runtime.py:1030-1047, 1267-1289`.
- **Classification**: CONFIRMED, severity P1 (false FAIL on target PC).
- **Mechanism**:
  1. `runtime.py:1031`: `_gpu_capability_key` reads `configured = os.environ.get(env.get("python_env_var") or "", "")`. If env var unset in current process, `configured = ""` and `fingerprint = None`.
  2. `runtime.py:1267-1289`: `verify_environment` falls back to `environment_assignment.json` when env var is unset, adopting `configured = interp`.
  3. Reader in process without exported env var derives key with `configured=""`, while writer with env var exported derived key with full interpreter path.
- **Observed output**:
  ```python
  Key with env var set:    f29c35e356fd0e90905cdce83de8f3f374c7c0913f6bea86311b05724ead11bd
  Key without env var set: 057c0d328b02dfd7963a1e774b46a38e3ad47ce41eb9c01f4391d3bfbe524e28
  Keys match: False
  ```
- **Effect**: Persisted capability record is invisible to reader resolving via `environment_assignment.json`.

### B-2 — `_cached_gpu_capability(authoritative=False)` omits interpreter fingerprint comparison
- **Location**: `workbench/backend/runtime.py:1193-1200, 1211-1216`.
- **Classification**: CONFIRMED, severity P2 (stale cache acceptance).
- **Mechanism**: When `authoritative=False`, cache lookup checks only `model_id`, `interpreter == configured`, `dependency_fingerprint`, `host_identity`, and HMAC signature. It never compares `interpreter_fingerprint`. Drifted interpreter binary or modified site-packages returns cached `GPU_RUNTIME_READY`.

### B-4 — In-memory capability map not `probe_source` filtered
- **Location**: `workbench/backend/runtime.py:1193-1200`.
- **Classification**: CONFIRMED, severity P3.
- **Mechanism**: On disk check (`runtime.py:1208-1210`), `probe_source` is filtered (`probe_source == "interpreter" or (synthetic_ok and probe_source == "override")`). In RAM check (`runtime.py:1193-1200`), `probe_source` check is omitted.

### D-F1 — Unregistered model with `observed_metrics=None` certifies
- **Location**: `workbench/scripts/verify_run.py:427-446`.
- **Classification**: CONFIRMED, severity P1 (false PASS hole).
- **Mechanism**:
  For model with `outcome == SELECTION_UNREGISTERED`, `verify_run.py:436-439`:
  ```python
  observed = report.get("observed_metrics") or {}
  honest.extend(f"the report's {key} is not a finite Recall in [0, 100]: {value!r}"
                for key, value in observed.items()
                if key in paper_gate.METRIC_KEYS and not _is_recall(value))
  ```
  If `observed_metrics` is `None` or `{}`, `observed.items()` is empty. Loop produces zero reasons.
- **Observed output**:
  `metric_evidence_reasons("unregistered_xyz", "c", "p", {"metric_extraction_available": True, "extraction_status": "EXTRACTED_VERIFIED", "observed_metrics": None})` returns `([], "UNREGISTERED")`.
- **Effect**: Report with no metric numbers passes verification and certifies as research evidence.

### F8-release — `release_verify.sh` substring proxy for proof presence
- **Location**: `workbench/docker/release_verify.sh:752, 764`.
- **Classification**: PROBABLE, severity P2.
- **Mechanism**: `_has_proof = not any("completion proof" in r for r in _reasons)`. If run fails verification for checksum mismatch or identity failure, `_reasons` has no `"completion proof"` substring. If `data_kind == "experiment"`, line 764 adds `_run_id` to `_inference_evidence`, yielding `MODEL_INFERENCE: PASS` from refuted run.

---

## 2. Backend, Queue, and Autopilot Audit

### Autopilot ledger corroboration bypass when ledger is None
- **Location**: `workbench/backend/autopilot.py:728-740`.
- **Classification**: CONFIRMED, severity P1 (false PASS hole).
- **Mechanism**:
  ```python
  try:
      ledger = read_evidence_ledger(wb_cfg)
  except LedgerKeyError as error:
      return base_fields, f"the evidence ledger key is unusable: {error}"
  if ledger is not None:
      matching = [item for item in ledger if item.get("run_id") == run_id]
      if not matching:
          return base_fields, f"corrupted evidence rejected: run {run_id} missing from evidence ledger"
  ```
  When `evidence_ledger.jsonl` is missing or HMAC verification fails, `read_evidence_ledger` returns `None`. `if ledger is not None:` evaluates to `False`. All ledger corroboration checks are bypassed. Autopilot proceeds to line 770: `base_fields["evaluation_kind"] = "OFFICIAL_VERIFIED"`.
- **Effect**: Uncorroborated or HMAC-corrupted run accepted as `OFFICIAL_VERIFIED`.

### Job Lifecycle: Enqueue-after-shutdown
- **Location**: `workbench/backend/jobs.py:218-219`; `workbench/backend/main.py:847-852`.
- **Classification**: VERIFIED (no defect).
- **Evidence**: `q.shutdown()` sets `_stopping=True`. Subsequent `q.enqueue(...)` raises `QueueShutdownError("queue has shut down")`. `main.py:847` catches `QueueShutdownError` and returns HTTP 503 (`queue_unavailable`).

### Job Lifecycle: Shutdown termination and cause persistence
- **Location**: `workbench/backend/jobs.py:303-309, 317-323`.
- **Classification**: VERIFIED (no defect).
- **Evidence**: `shutdown()` transitions queued/preparing jobs to `cancelled` with `job.error = "workbench shut down before dispatch"`. Running jobs receive SIGTERM then SIGKILL to their process group (`killpg`), with persisted status `cancelled` and error `"workbench is shutting down"`.

### Job Lifecycle: Dispatch-record ordering
- **Location**: `workbench/backend/jobs.py:452-460, 550, 605-606`.
- **Classification**: VERIFIED (no defect).
- **Evidence**: `_write_command_metadata` writes `queue_command.json` atomically via temporary file and replace before publishing terminal state in `_persist(job)`.

### Cancellation precedence
- **Location**: `workbench/backend/jobs.py:252-265, 564, 612`.
- **Classification**: VERIFIED (no defect).
- **Evidence**: Calling `cancel()` on running job sets status `cancelled` and populates `_cancelling`. Worker checks `job.status != "cancelled" and job.job_id not in self._cancelling` before publishing completion. In-flight cancellation always wins over exit code.

### Restart recovery drops completed jobs
- **Location**: `workbench/backend/jobs.py:505-509`; `workbench/backend/main.py:96-100`.
- **Classification**: CONFIRMED, severity P2.
- **Mechanism**: `recover_interrupted` globs `*/queue_state.json` but processes only `status in ("queued", "preparing", "running")`. Finished jobs (`"completed"`, `"cancelled"`, `"failed"`) are ignored. `main.py` only adopts what `recover_interrupted` returns. After restart, `/api/jobs` is empty (`[]`). Direct cause of A9 false failure.

### Duplicate jobs: bundle vs member identity gap
- **Location**: `workbench/backend/jobs.py:223-228`; `workbench/backend/main.py:778-786`.
- **Classification**: CONFIRMED, severity P3.
- **Mechanism**: `main.py` maps bundle selection to `checkpoint_id = bundle_id`, whereas individual member keeps `checkpoint_id = member_id`. Queue checks `(model_id, checkpoint_id, protocol_id)`. Simultaneous submission of `base_iter0_all_categories` and `0_dress_best_model.pt` bypasses duplicate check.

### Result index & Serving index readiness
- **Location**: `workbench/backend/main.py:62-90, 180-193`; `workbench/backend/index.py:75-110`.
- **Classification**: VERIFIED (no defect).
- **Evidence**: `_ensure_index()` checks `db_path.is_file()`. If absent, builds in daemon thread. `/api/runs` during build returns HTTP 503 (`index_unavailable`, `status: "building"`). Existing index left intact on boot.

### Dead weight in backend
- **Location**: `workbench/backend/index.py:30-31`; `workbench/backend/analysis.py:91-93`.
- **Classification**: KNOWN_LIMITATION, severity P3.
- **Evidence**: `DATABASE_PATH` and `RESULTS_ROOT` in `index.py` frozen at import time; `research_interest` in `analysis.py` has no callers.

---

## 3. Frontend Audit (TypeScript)

### Verification receipt completely invisible in UI
- **Location**: `workbench/frontend/App.tsx:49-67`; `workbench/frontend/gate.ts:15-23`.
- **Classification**: CONFIRMED, severity P1 (false PASS in presentation).
- **Mechanism**:
  1. `main.py:580-587`: API sends `verification` object in `/api/benchmark-runs` and `/api/runs/gate` (`artifact_exists`, `execution_authenticated`, `is_research_evidence`, `certified`, `state`, `evidence_reasons`).
  2. `gate.ts:15-23`: `RunGate` type declaration completely omits `verification`.
  3. `App.tsx:49-67`: `GateDetail` component only inspects `gate.benchmark_gate` and `gate.parity_status`. Line 58 only appends `"(not a certified pass)"` if `verdict.passed === false`.
  4. If run achieves paper metric tolerance (`verdict.passed === true`), but proof is missing (`state: "proof_unavailable"`), ledger missing (`state: "uncorroborated"`), or verifier rejected (`state: "refused"`):
     UI displays **`PASS_WITHIN_5_PERCENT`** with no refusal badge, no warning label, and no reason text.

### Frontend silently swallows unusable ledger key error
- **Location**: `workbench/frontend/App.tsx:21`.
- **Classification**: CONFIRMED, severity P2.
- **Mechanism**:
  ```typescript
  const loadReports = () => {
    api<Record<string,RunGate>>("/api/runs/gate").then(g=>setGates(g||{})).catch(()=>setGates({}));
    api<BenchmarkRun[]>("/api/benchmark-runs").then(r=>setBenchmarkRuns(r||[])).catch(()=>setBenchmarkRuns([]));
  };
  ```
  When backend encounters conflicting or missing ledger keys, `/api/benchmark-runs` raises HTTP 400 `ledger_key_unusable` with remediation instructions. `loadReports` catches error and silently sets `benchmarkRuns` to `[]`. UI error banner (`error` state) is not set. Operator sees empty benchmark runs without explanation.

---

## 4. Findings Matrix

| ID | Sev | Classification | Location | Root cause / Summary |
|---|---|---|---|---|
| F10-R2 | P1 | CONFIRMED | `App.tsx:49-67`, `gate.ts:15` | Frontend ignores `verification`; uncertified/proof-unavailable/uncorroborated runs render as PASS |
| B-1/A5 | P1 | CONFIRMED | `runtime.py:1031, 1278` | `_gpu_capability_key` ignores `environment_assignment.json`; cached records unreadable without env var |
| D-F1 | P1 | CONFIRMED | `verify_run.py:436` | Unregistered model with `observed_metrics=None` produces 0 refusal reasons and certifies |
| AP-1 | P1 | CONFIRMED | `autopilot.py:728` | When `ledger is None`, ledger corroboration is skipped; uncorroborated run accepted as `OFFICIAL_VERIFIED` |
| A9 | P2 | CONFIRMED | `release_verify.sh:372`, `jobs.py:505` | Empty `/api/jobs` marks `OPERATIONS` PENDING, causing false `ENGINEERING_INCOMPLETE` release verdict |
| B-2 | P2 | CONFIRMED | `runtime.py:1193, 1211` | Advisory `_cached_gpu_capability` omits interpreter fingerprint comparison |
| F8-rel | P2 | PROBABLE | `release_verify.sh:752` | Substring proxy `completion proof` allows refuted experiment to grant `MODEL_INFERENCE: PASS` |
| UI-1 | P2 | CONFIRMED | `App.tsx:21` | `loadReports` `.catch()` silently swallows HTTP 400 `ledger_key_unusable` from `/api/benchmark-runs` |
| F4 | P2 | KNOWN_LIMITATION | `main.py:332` | Development opt-in certifies without proof; constrained to pytest/dev processes |
| F13 | P3 | CONFIRMED | `jobs.py:223`, `main.py:778` | Bundle ID and anchor member ID have distinct duplicate keys; GPU work can be duplicated |
| B-4 | P3 | CONFIRMED | `runtime.py:1193` | In-memory `_GPU_CAPABILITY` lookup lacks `probe_source` filtering |
| F12 | P3 | KNOWN_LIMITATION | `index.py:30`, `analysis.py:91` | Dead weight: import-frozen globals in `index.py`, unused `research_interest` in `analysis.py` |

---

## 5. Evidence Not Obtained

- Live GPU execution on RTX 5070 Ti (host lacks sm_120 hardware and nvidia-container-toolkit).
- Full frontend end-to-end browser execution (Node/Vite build was not executed; audited via static TS source and AST structure).
- External network downloads or Docker image rebuilds.
- Files outside slice E (`paper_gate.py`, `preflight.py`, `fashioniq_layout.py`, `operator_config.py`, `benchmark_references.py`) audited only at callsite boundaries.

---

## 6. Can this fail on the real first run on an RTX 5070 Ti PC?

**Yes.** Even after hardware drivers and dataset are present, the stack faces immediate blockers and false failures:
1. **B-1/A5 (P1)**: The container or shell executing without individual model env vars exported (e.g. `CSMCIR_PYTHON`, `LIMN_PYTHON`) will fail to find existing GPU capability records because `_gpu_capability_key` hashes an empty string while `verify_environment` resolves via `environment_assignment.json`.
2. **A9 (P2)**: On any clean boot or restart of the backend container, `SingleGpuQueue` has no memory of completed jobs. Running `release_verify.sh` against the live stack will observe `jobs == []`, failing J8 and forcing release verdict to `ENGINEERING_INCOMPLETE` on an otherwise healthy host.
3. **UI-1 / F10-R2 (P1/P2)**: If two conflicting ledger keys exist in the environment, `/api/benchmark-runs` returns HTTP 400, which the UI silently swallows, displaying empty benchmark tables without explanation. If reports are served, any uncertified or proof-unavailable run within 5% paper tolerance presents in the dashboard as a certified pass.
