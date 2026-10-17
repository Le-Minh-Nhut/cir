# AUDIT_B — Round 02 Verification Audit (Slice B)

Audited slice: `runtime.py`, `environment_planner.py`, `manage_environment.py`, `pc_readiness.py`, `pipeline.py`, `gpu_check.py`, `registry.py`, `models.yaml`.
Context: Round-01 open leads re-evaluated, bidirectional capability proofs executed, recent fix waves audited for regressions.

---

## 1. Round-01 Open Leads: Execution & Static Re-evaluation

### B-1 / A5: Divergent `_gpu_capability_key` & Cross-Process Failure
- **Verdict**: `CONFIRMED` | **Severity**: `P1`
- **Location**: `workbench/backend/runtime.py:1027-1047`, `1185-1222`, `1267-1288`; `workbench/docker/release_verify.sh:581-583`, `637-648`.
- **Mechanism**: `_gpu_capability_key` reads interpreter solely from `os.environ[python_env_var]`. When unset, key derives with `interpreter=""` and `fingerprint=null`. `verify_environment` adopts interpreter from `environment_assignment.json`. Key mismatch prevents reading valid records across processes.
- **Observed Execution**:
  ```python
  # Set env var:
  Key with env var:    fc9a0a27889c868b7afe6d87a17c7d40535b25bd2a4900c47e768e1563c4dfda
  # Unset env var:
  Key without env var: 6fc960908a71635d299549ba1c00f06253cfffc6a0de11d9e9dfb7e05841042b
  # Result:
  Keys match? False
  ```
  Process 1 writes record with env var set (`key=9c23...`). Process 2 has env var unset and adopts interpreter via `environment_assignment.json`:
  ```
  Report interpreter adopted from assignment: /tmp/.../env/bin/python
  Authoritative cached record found: False
  Advisory cached record found: False
  ```
  `release_verify.sh:637` checks `record.capability_key != _expected_key(model)`. `_expected_key` computes key with env var unset (in-container), rejecting genuine host record with `"the record's capability key is not this environment's key"`.
- **Target PC Impact**: False FAIL. Release verifier pins `GPU_RUNTIME` to `PENDING` on genuine hardware.

### B-2: Advisory Read Omits Interpreter Fingerprint Comparison
- **Verdict**: `CONFIRMED` | **Severity**: `P2`
- **Location**: `workbench/backend/runtime.py:1192-1220` vs `1252-1255`.
- **Mechanism**: Advisory path (`authoritative=False`) compares only interpreter string path, contract digest, and host identity. Never evaluates `_record_environment_fingerprint` or `cached["interpreter_fingerprint"]`.
- **Observed Execution**:
  ```
  [EXEC] Mutated interpreter bytes on disk:
  Advisory read accepted drifted interpreter: True
  Authoritative read accepted drifted interpreter: False
  ```
  Advisory lookup returns stale/mutated environment as ready.

### B-4: In-Memory `_GPU_CAPABILITY` Cache Ignores `probe_source`
- **Verdict**: `CONFIRMED` | **Severity**: `P3`
- **Location**: `workbench/backend/runtime.py:1193-1200` vs `1208-1210`.
- **Mechanism**: In-memory scan in `_cached_gpu_capability` checks `model_id`, `interpreter`, `contract`, `host_identity`, `signature`. Omits `probe_source` filter present in disk pass.
- **Observed Execution**:
  ```
  [EXEC] Injected synthetic record into _GPU_CAPABILITY with override switch=1, then unset switch:
  Synthetic record returned despite switch unset: True
  ```

### B-6: `pc.sh` Hardcoded Registry Path & LIMN Fallback
- **Verdict**: `CONFIRMED` | **Severity**: `P3`
- **Location**: `pc.sh:539-552`.
- **Mechanism**: Directly queries `${REPO_ROOT}/workbench/registry/models.yaml` instead of resolving `WORKBENCH_REGISTRY_PATH`. Uses hardcoded fallback `if model == 'limn': print('sm_120')` on YAML parse failure.

### B-9: `pc.sh` Adaptation Omitted on Existing `.env`
- **Verdict**: `CONFIRMED` | **Severity**: `P2`
- **Location**: `pc.sh:568-582`, `644-648`.
- **Mechanism**: `WORKBENCH_RUNTIME_ADAPTATION` written to `.env` only if `created_new_env == 1`. Existing checkout with pre-existing `.env` retains old/blank adaptation. Restarted container lacks `sm_120`, triggering `inspect_adaptation_mismatch`.

### B-11: Falsy Bypass of Interpreter Fingerprint Check
- **Verdict**: `CONFIRMED` | **Severity**: `P3`
- **Location**: `workbench/backend/runtime.py:1253-1255`.
- **Mechanism**: Guard `if cached.get("interpreter_fingerprint") and expected_fingerprint:` skips check if record omits fingerprint (`None` or `""`), returning `cached`.
- **Observed Execution**:
  ```
  [EXEC] Record with "interpreter_fingerprint": None
  Record with None fingerprint accepted by authoritative reader: True
  ```

### F4: `_proof_stamp == "absent"` Certification Relaxations
- **Verdict**: `CONFIRMED` (in development/test), `EXPECTED_BEHAVIOR` (in deployment) | **Severity**: `P2`
- **Location**: `workbench/backend/main.py:240-244`, `402-409`.
- **Mechanism**: Recent fix wave separated deployment vs development. Deployment returns `state="proof_unavailable"`, `certified=False`. But in development mode (`PYTEST_CURRENT_TEST` or `WORKBENCH_DEVELOPMENT_MODE`) with `WORKBENCH_ALLOW_DEVELOPMENT_CERTIFICATES=1`, `_development_certification` issues `certified=True`, `is_research_evidence=True` without evaluating run manifest.

### F8-release: Negative Substring Proxy for Proof Presence in `release_verify.sh`
- **Verdict**: `CONFIRMED` | **Severity**: `P1`
- **Location**: `workbench/docker/release_verify.sh:752`, `764-765`, `782-784`.
- **Mechanism**: `_has_proof = not any("completion proof" in r for r in _reasons)`. If run is rejected for reasons not containing literal substring `"completion proof"` (e.g. exit code non-zero, ledger entry missing/invalid), `_has_proof` evaluates `True`. Run added to `_inference_evidence`, upgrading `MODEL_INFERENCE` to `PASS`.

### D-F1: Unregistered Model Empty Observed Metrics Accepted
- **Verdict**: `CONFIRMED` | **Severity**: `P1`
- **Location**: `workbench/scripts/verify_run.py:435-445`.
- **Mechanism**: In `metric_evidence_reasons` for `SELECTION_UNREGISTERED`, loop checks `observed.items()`. If `observed_metrics` is `None` or `{}`, 0 iterations execute, `honest` remains empty, and report with no numbers certifies.

### A9: `OPERATIONS` PENDING on Pristine Host Empty Queue
- **Verdict**: `CONFIRMED` (design flaw) / `KNOWN_LIMITATION` | **Severity**: `P2`
- **Location**: `workbench/docker/release_verify.sh:370-377`, `531`, `816-818`.
- **Mechanism**: J8 requires non-empty `/api/jobs` queue (`if not jobs: exit(3)`). On clean deployment or host where runs occurred via CLI, empty queue marks J8 `PENDING`. `OPERATIONS` becomes `PENDING`, capping release verdict at `ENGINEERING_INCOMPLETE`.

---

## 2. Bidirectional Capability Verification Proofs

### Direction 1: Foreign Host & Foreign Interpreter Refusal
- **Foreign Host**: Record signed with foreign host identity (`node="foreign_host"`).
  ```
  Foreign host advisory accepted: False
  Foreign host authoritative accepted: False
  ```
  Both paths fail closed. Authoritative key derivation includes `_host_identity()`. Advisory explicitly checks `cached_payload.get("host_identity") == _host_identity()`.
- **Foreign Interpreter Path**: Record created for venv A, evaluated against venv B.
  ```
  Foreign interpreter (different path) advisory accepted: False
  Foreign interpreter (different path) authoritative accepted: False
  ```
  Both paths fail closed.
- **Foreign Interpreter Binary (Drift at same path)**:
  - Authoritative: `REFUSED` (calculates `_record_environment_fingerprint`).
  - Advisory: `ACCEPTED` (Defect B-2).

### Direction 2: Legitimate Record Shared Across Resolvers
- **Scenario**: Process 1 verifies GPU with `WORKBENCH_<MODEL>_PYTHON` set in environment. Process 2 runs without env var, resolving interpreter via `environment_assignment.json`.
  ```
  Process 1: writes record under key 9c230a756a2b2fac...
  Process 2: adopts interpreter /tmp/.../env/bin/python
  Authoritative read: False (looks for key 6fc960908a71635d...; file does not exist)
  Advisory read: False (compares payload interpreter against configured="", mismatches)
  ```
  **Direction 2 FAILS CLOSED BROKEN**. Legitimate record cannot be consumed across legitimate resolution paths.

---

## 3. Audit of Recent Fix Waves & Regressions

### 1. `proof_unavailable` State Lets Uncertified Report Be Reused by `pc.sh`
- **Verdict**: `CONFIRMED` | **Severity**: `P1`
- **Location**: `pc.sh:1238-1243`, `1286-1289`, `1563-1564`.
- **Mechanism**: Recent backend fix marked runs without proof as `state="proof_unavailable"`, `certified=False`. However, `pc.sh` added lines 1238-1243 treating `proof_unavailable` as exit code 2:
  ```bash
  elif (( check_rc == 2 )); then
      target_run_id="$cand_id"
      FINAL_RUN_PROOF_UNAVAILABLE=1
      break
  ```
  `pc.sh` skips re-running evaluation, prints "KẾT QUẢ ĐÁNH GIÁ CHÍNH THỨC" and exits 0. Uncertified run without proof is reused as official final output.

### 2. Cause-Specific Remedy in `pc.sh` Fail-Closed Behavior
- **Verdict**: `VERIFIED` | **Severity**: `N/A`
- **Location**: `pc.sh:752-789`.
- **Mechanism**: `require_corrected_configuration` invokes `fail_stage` when `inspect_config_satisfies == 0`. Calls `exit 1` with specific remediation strings. Fails closed properly.

### 3. `release_verify.sh` Masking Real Verdicts & Deleting Files
- **Verdict**: `CONFIRMED` | **Severity**: `P2`
- **Location**: `workbench/docker/release_verify.sh:683-690`, `702-706`.
- **Defects**:
  1. **Destructive Mutation in Verifier**: Lines 683-690 execute `os.remove(_path)` on any artifact failing `json.load`. Verification script silently deletes host artifacts.
  2. **Scan Status Masking**: Line 702 sets `status = "UNREADABLE"` if any unreadable file is encountered, or `"WRONG_KEY"` if any old record exists, even when a valid record exists and `real_gpu == True`.

### 4. `_cache_key` Interpreter Omission
- **Verdict**: `CONFIRMED` | **Severity**: `P3`
- **Location**: `workbench/backend/runtime.py:1646-1650`.
- **Mechanism**: In-memory environment cache key derives from `os.environ.get(python_env_var)`. Does not use resolved interpreter, causing redundant verifications when env var is unset.

---

## 4. Comprehensive Findings Table

| # | Verdict | Sev | Location | Summary |
|---|---|---|---|---|
| B-1 / A5 | CONFIRMED | P1 | `runtime.py:1031`, `release_verify.sh:581` | `_gpu_capability_key` reads only env var; cross-process readers resolving via assignment get different key → GPU_RUNTIME false FAIL |
| B-2 | CONFIRMED | P2 | `runtime.py:1192-1220` | Advisory reader omits interpreter fingerprint check; accepts drifted/mutated interpreter |
| B-3 (F8) | CONFIRMED | P1 | `release_verify.sh:752`, `783` | Negative substring `"completion proof"` proxy treats non-proof refusals as valid inference evidence → MODEL_INFERENCE false PASS |
| B-4 | CONFIRMED | P3 | `runtime.py:1193-1200` | In-memory `_GPU_CAPABILITY` map not filtered by `probe_source`; synthetic record persists after switch unset |
| B-6 | CONFIRMED | P3 | `pc.sh:539-552` | Adaptation parsed from static YAML file with hardcoded limn fallback, ignoring `WORKBENCH_REGISTRY_PATH` |
| B-7 (F4) | CONFIRMED | P2 | `main.py:240-244`, `402-409` | Development mode certification bypasses proof validation under opt-in switch |
| B-9 | CONFIRMED | P2 | `pc.sh:568-582` | `WORKBENCH_RUNTIME_ADAPTATION` written to `.env` only on initial file creation; existing `.env` triggers restart abort |
| B-11 | CONFIRMED | P3 | `runtime.py:1253-1255` | `interpreter_fingerprint: null` bypasses authoritative fingerprint mismatch check |
| B-12 | KNOWN_LIMITATION | P3 | `pipeline.py:1401` | `probe_gpu_capability` calls `verify_gpu_runtime(force=False)`, reusing cached record instead of forced probe |
| B-13 | CONFIRMED | P3 | `runtime.py:1646-1650` | `_cache_key` keys on raw env var rather than resolved interpreter |
| B-14 | CONFIRMED | P1 | `verify_run.py:435-445` | Unregistered model with empty/missing `observed_metrics` certifies with zero numbers (D-F1) |
| B-15 | CONFIRMED | P2 | `release_verify.sh:372-377` | Empty queue in J8 sets `OPERATIONS` PENDING, causing spurious `ENGINEERING_INCOMPLETE` release verdict (A9) |
| B-16 | CONFIRMED | P1 | `pc.sh:1238-1243`, `1286` | `pc.sh` reuses `proof_unavailable` runs as official final result despite `certified=False` |
| B-17 | CONFIRMED | P2 | `release_verify.sh:683-690` | Verifier script deletes files (`os.remove`) in `artifacts/gpu_capabilities/` on parse error |
| B-18 | CONFIRMED | P2 | `release_verify.sh:702-706` | Residual legacy records force scan status to `WRONG_KEY`/`UNREADABLE`, masking valid `real_gpu` |

---

## 5. Scope & Inspection Boundaries

- **Inspected in Full / Range**:
  - `workbench/backend/runtime.py` (full capability, key derivation, verification, and caching paths)
  - `workbench/backend/environment_planner.py` (assignment loading, group state, provisioning)
  - `workbench/scripts/manage_environment.py` (interpreter inspection, assignment resolution)
  - `workbench/scripts/pc_readiness.py` (readiness classification, blocker aggregation)
  - `workbench/scripts/pipeline.py` (GPU probe stage, execution hooks)
  - `workbench/scripts/gpu_check.py` (torch probe, hardware query)
  - `workbench/backend/registry.py` (model/checkpoint resolution, contract digests)
  - `workbench/registry/models.yaml` (model definitions, adaptation contracts)
  - `workbench/backend/main.py` (certification modes, proof checks)
  - `workbench/scripts/verify_run.py` (metric extraction, evidence checks)
  - `workbench/docker/release_verify.sh` (verification dimensions, scan logic)
  - `pc.sh` (restart verification, reuse logic, configuration checks)
- **NOT Inspected**:
  - `src/datasets/**` and `src/evaluation/**` (Slice D/E)
  - `workbench/backend/fashioniq_layout.py`, `workbench/backend/checkpoint_selection.py`
  - `workbench/frontend/**` (Slice A)
  - Training scripts, upstream weights binaries, raw dataset images

---

## 6. Can this fail on the real first run on an RTX 5070 Ti PC?

**YES — False FAIL is guaranteed on the first run of the release verifier.**

1. **B-1 / A5 Failure Flow**:
   - On host execution, `pc.sh` or autopilot provisions an environment, storing the path in `/artifacts/environment_assignment.json`.
   - Autopilot exports `WORKBENCH_LIMN_PYTHON` to the worker, producing a valid record keyed to the interpreter fingerprint (`key_A`).
   - When `./pc.sh` runs `release_verify.sh` inside the container (where `WORKBENCH_LIMN_PYTHON` is not exported in `docker-compose.yml`), `release_verify.sh:582` calls `_expected_key(model)` with env var unset, generating `key_B` (`6fc96090...`).
   - `release_verify.sh:637` rejects the host capability record because `record.capability_key != _expected_key(model)`.
   - `real_gpu` remains `False`, `GPU_RUNTIME` is set to `PENDING`, and release verification halts at `GPU_VALIDATION_PENDING`.

2. **B-9 Failure Flow**:
   - On an existing repository checkout with a pre-existing `.env`, `./pc.sh --model limn` fails to write `WORKBENCH_RUNTIME_ADAPTATION=sm_120` to `.env`.
   - Container restarts without the adaptation, failing `require_corrected_configuration` on adaptation mismatch.

3. **A9 Failure Flow**:
   - On a clean host where evaluations ran directly, an empty `/api/jobs` queue triggers exit code 3 in J8, resulting in `ENGINEERING_INCOMPLETE`.

4. **False PASS Flow (B-16 / F8)**:
   - If an uncertified run without a completion proof exists in the ledger, `pc.sh:1238-1243` reuses it rather than executing the model on the RTX 5070 Ti, while `release_verify.sh:752` marks `MODEL_INFERENCE` as `PASS`.
