# Final PC Audit Report: Test Suite & Release Verification Integrity

## A) Files Inspected

- `workbench/docker/release_verify.sh`
- `workbench/docker/acceptance.sh`
- `workbench/docker/mutation_checks.sh`
- `workbench/docker/run.sh`
- `workbench/backend/runtime.py`
- `workbench/backend/jobs.py`
- `workbench/scripts/pc_readiness.py`
- `workbench/scripts/evaluate_models.py`
- `workbench/scripts/pipeline.py`
- `workbench/tests/test_http_api_layer.py`
- `workbench/tests/test_release_blocker_cuda_readiness.py`
- `workbench/tests/test_release_blocker_master_e2e.py`
- `workbench/tests/test_release_blocker_interpreter.py`
- `workbench/tests/test_release_blocker_mutations.py`
- `workbench/tests/test_strict_acceptance_matrix.py`
- `workbench/tests/test_runtime_adaptation.py`
- `workbench/tests/test_pc_readiness.py`
- `workbench/tests/test_autopilot_cold_start.py`
- `workbench/tests/test_autopilot_e2e.py`
- `workbench/tests/test_concurrency_ownership.py`
- `workbench/tests/test_execution_graph_and_dependencies.py`
- `workbench/tests/test_pipeline.py`
- `workbench/tests/test_paper_gate.py`
- `workbench/tests/test_doctor.py`
- `workbench/tests/test_download_auxiliary_assets.py`

## B) Paths Traced

1. **Acceptance Group E GPU Bypass Path**:
   `acceptance.sh:213-232` -> executes `pc_readiness.py --json` -> reads output `doc` -> extracts `doc.get("rules", [])` -> iterates over empty list -> evaluates `assert not any(...)` -> asserts `True` -> prints false PASS with 0 rules checked.
2. **Acceptance Group G Status Contract Path**:
   `acceptance.sh:96-110` -> queries `GET /api/jobs` -> extracts statuses into set -> evaluates `statuses - allowed` -> on clean first boot with 0 jobs, evaluates `set() - allowed == set()` -> prints false PASS on empty list.
3. **Release Verifier Offline & Exit Code Path**:
   `release_verify.sh:52-57, 61-93, 430-438, 507-527` -> runs with `--offline` -> marks J1, J2, J3, J8, J11 as `PENDING` -> counts `fails=0` -> line 522 checks `if (( fails > 0 ))` -> exits with code `0` (PASS) despite zero deployment checks executed -> line 513 reports verdict `GPU_VALIDATION_PENDING` masking missing deployment checks.
4. **Release Verifier Capability Scan & Artifact Corruption Path**:
   `release_verify.sh:492-506` -> scans `glob.glob(str(_gcd(_rc()) / "*.json"))` -> parses each file with `_json.loads` in generator -> hits 0-byte corrupt file `1a3bb380...json` left by test run in `workbench/artifacts/gpu_capabilities/` -> raises `JSONDecodeError` -> caught by outer `except Exception:` -> sets `real_gpu = False` unconditionally.
5. **Mutation Check M4 Docker Daemon Failure Path**:
   `mutation_checks.sh:180-186` -> runs `docker compose config` without environment -> when Docker daemon down or CLI unavailable, command fails with nonzero exit code -> script enters `else` branch -> marks M4 `PASSED` instead of failing.
6. **API Layer Queue Mocking Path**:
   `test_http_api_layer.py:70-145` -> fixture `api` replaces `module.JOBS` with dummy object returning hardcoded lambda `list: lambda self: [{"job_id": "job-1", "status": "queued"}]` -> tests never touch `SingleGpuQueue`, background dispatcher thread, or job process group.
7. **Runtime Adaptation Host Dependency Path**:
   `test_runtime_adaptation.py:70-81` -> calls `limn.validate_environment()` -> inspects host Python `.venv` instead of isolated test environment -> asserts exact package diff matching host developer venv.
8. **E2E GPU Capability Seam Path**:
   `test_release_blocker_master_e2e.py:513-579` -> installs test override via `runtime.set_gpu_probe_for_tests` -> hardcodes `compute_capability: '8.9'` (sm_89 Ada Lovelace) -> no test path exercises `sm_120` (Blackwell RTX 5070 Ti).

## C) Confirmed Defects

### Defect 1: Group E In `acceptance.sh` Produces An Empty-List False PASS On GPU Readiness
- **Severity**: P0
- **Location**: `workbench/docker/acceptance.sh:218-222`
- **Root Cause**: Script extracts `doc.get("rules", [])` from `pc_readiness.py --json`. `pc_readiness.py` output schema has no top-level `rules` key (it emits `gpu`, `intended_gpu`, `models`, `config`, etc.). `gpu_rules` is always `[]`. `assert not any(...)` on an empty list evaluates to `True`.
- **Reproduction**:
  ```bash
  /home/heheboiz/data/cir/.venv/bin/python -c '
  import json, subprocess, sys
  out = subprocess.run([sys.executable, "workbench/scripts/pc_readiness.py", "--json", "--model", "limn"], capture_output=True, text=True)
  doc = json.loads(out.stdout)
  gpu_rules = [rule for rule in doc.get("rules", []) if "GPU" in str(rule.get("rule_id", ""))]
  assert not any(r.get("status") in ("READY", "GPU_RUNTIME_READY") for r in gpu_rules)
  print(f"PASS ({len(gpu_rules)} rules)")
  '
  # Output: PASS (0 rules)
  ```
- **Expected vs Actual**:
  - Expected: Inspect `doc["models"]["limn"]["gpu_verified"]` or `doc["gpu"]` to confirm unprobed GPU is pending.
  - Actual: Asserts on non-existent `rules` key, evaluates 0 rules, prints false PASS even if GPU readiness was bypassed.
- **Impact**: Code bypassing the GPU verification gate passes acceptance checks undetected.
- **Minimal Fix**:
  ```bash
  # In workbench/docker/acceptance.sh:219-222
  limn_model = doc.get("models", {}).get("limn", {})
  assert limn_model.get("gpu_verified") is not True, "readiness must not claim GPU verified without probe"
  assert limn_model.get("state") != "READY_FOR_FULL_EVALUATION"
  ```

### Defect 2: Corrupted Test Artifacts In `gpu_capabilities/` Unconditionally Break `release_verify.sh` GPU Detection
- **Severity**: P1
- **Location**: `workbench/docker/release_verify.sh:496-506`
- **Root Cause**: Pytest executions wrote over 1,690 test JSON files directly into production artifacts tree `workbench/artifacts/gpu_capabilities/`. File `1a3bb380cba4518172218c7f22978d745145db7b37c37ac2d2f2a6d1ead42d75.json` is 0 bytes. `release_verify.sh` uses a generator parsing all files with `_json.loads` without per-file exception handling. The 0-byte file raises `JSONDecodeError`, outer `try...except Exception:` catches it, and forces `real_gpu = False`.
- **Reproduction**:
  ```bash
  /home/heheboiz/data/cir/.venv/bin/python -c '
  import glob, json
  from workbench.backend.operator_config import resolve_config as _rc
  from workbench.backend.runtime import gpu_capability_directory as _gcd
  records = [json.loads(open(p).read()) for p in glob.glob(str(_gcd(_rc()) / "*.json"))]
  '
  # Raises: json.decoder.JSONDecodeError on 1a3bb380...json
  ```
- **Expected vs Actual**:
  - Expected: Per-file exception handling ignoring unparseable records; tests isolated from production artifacts.
  - Actual: One corrupt file crashes the scan; verifier never recognizes a real GPU probe.
- **Impact**: On a real RTX 5070 Ti machine, a genuine interpreter probe is ignored by `release_verify.sh`.
- **Minimal Fix**:
  ```python
  # In workbench/docker/release_verify.sh:496-504
  def _safe_load(p):
      try:
          return _json.loads(open(p, encoding="utf-8").read())
      except Exception:
          return None
  real_gpu = any(
      isinstance(record := _safe_load(p), dict)
      and record.get("probe_source") == "interpreter"
      and record.get("capability_verified") is True
      for p in _glob.glob(str(_gcd(_rc()) / "*.json"))
  )
  ```

### Defect 3: Group G In `acceptance.sh` Reports False PASS On Empty Job Queue
- **Severity**: P1
- **Location**: `workbench/docker/acceptance.sh:98-107`
- **Root Cause**: Script tests `/api/jobs` statuses by computing `statuses - allowed`. When `/api/jobs` returns `[]` (standard clean boot), `statuses` is empty, diff is empty, and check reports PASS without observing a single job lifecycle.
- **Reproduction**:
  Execute `acceptance.sh` against newly booted server with 0 jobs. Outputs `PASS job statuses lowercase enum`.
- **Expected vs Actual**:
  - Expected: Fail or report SKIP/PENDING if 0 jobs are observed (as in `release_verify.sh:312`).
  - Actual: Prints PASS on empty list.
- **Impact**: Backend job serialization regressions pass acceptance tests on every fresh boot.
- **Minimal Fix**:
  ```bash
  # In workbench/docker/acceptance.sh:104-106
  if not jobs:
      print("  SKIP G (no jobs observed to verify lifecycle)")
      sys.exit(2)
  ```

### Defect 4: `release_verify.sh` Exits 0 On Incomplete Run And Misattributes Verdict
- **Severity**: P1
- **Location**: `workbench/docker/release_verify.sh:507-527`
- **Root Cause**: Lines 522-527 count only `fails`. Unrun or skipped live checks (J1, J2, J3, J8, J11) marked `PENDING` do not increment `fails`. Script prints `RELEASE VERIFIER: no unmet criteria (N pending)` and exits 0. Additionally, line 513 sets `out["verdict"] = "GPU_VALIDATION_PENDING"` whenever `GPU_RUNTIME` is pending, even when deployment checks J1-J3 failed or skipped.
- **Reproduction**:
  ```bash
  bash workbench/docker/release_verify.sh --offline
  # Exits: 0
  # Output: RELEASE VERIFIER: no unmet criteria (6 pending)
  # Report verdict: GPU_VALIDATION_PENDING
  ```
- **Expected vs Actual**:
  - Expected: Exit nonzero when mandatory deployment checks did not run; report `ENGINEERING_INCOMPLETE` if non-GPU checks are pending.
  - Actual: Exits 0 and reports `GPU_VALIDATION_PENDING`.
  - Minimal Fix:
  ```bash
  # In release_verify.sh:512-513
  non_gpu_pending = [d for d in mandatory_pending if d != "GPU_RUNTIME"]
  out["verdict"] = "ENGINEERING_INCOMPLETE" if non_gpu_pending else ("GPU_VALIDATION_PENDING" if mandatory_pending else "RELEASE_READY")
  # At line 522:
  if (( fails > 0 )) || (( pendings > 0 && OFFLINE == 0 )); then exit 1; fi
  ```

### Defect 5: Mutation M4 In `mutation_checks.sh` Reports False PASS On Docker Daemon Failure
- **Severity**: P2
- **Location**: `workbench/docker/mutation_checks.sh:180-186`
- **Root Cause**: `if docker compose -f ... config; then FAILED+=(M4); else PASSED+=(M4); fi`. If Docker is stopped or missing, `docker compose` fails with an engine connection error. Script treats any exit error as successful detection of missing environment variables and marks M4 PASSED.
- **Reproduction**:
  Stop Docker daemon or point `DOCKER_HOST` to invalid socket. Run `mutation_checks.sh`. M4 outputs `DETECTED` and passes.
- **Expected vs Actual**:
  - Expected: Verify stderr explicitly contains variable interpolation errors (`variable is not set`).
  - Actual: Any docker failure passes the mutation check.
- **Impact**: Mutation suite gives false confidence when testing environments with broken Docker configurations.
- **Minimal Fix**:
  ```bash
  out="$(docker compose -f "${HERE}/docker-compose.yml" --env-file /dev/null config 2>&1 || true)"
  if echo "$out" | grep -qE "variable is not set|required variable"; then
      PASSED+=(M4)
  else
      FAILED+=(M4)
  fi
  ```

### Defect 6: `test_runtime_adaptation.py` Asserts Exact State Of Developer Workstation Virtualenv
- **Severity**: P2
- **Location**: `workbench/tests/test_runtime_adaptation.py:77-81`
- **Root Cause**: `test_limn_validate_environment_accepts_both_contracts` runs `limn.validate_environment()` against the current active interpreter (`sys.executable`) without isolation. It hardcodes `assert adapted == ["LIMN environment package missing: open_clip_torch >=2.24,<2.27"]`. This passes only because host `.venv` currently has `torch==2.14.0+cpu` (which semver satisfies `>=2.7,<3`) and lacks `open_clip_torch`.
- **Reproduction**:
  Run test in venv containing `open_clip_torch` or PyTorch 2.6: test immediately fails with assertion mismatch.
- **Expected vs Actual**:
  - Expected: Execute test against an isolated venv fixture (like `test_release_blocker_interpreter.py`).
  - Actual: Test depends on uncontrolled developer machine site-packages.
- **Impact**: Test suite breaks or falsely passes depending on developer host package installs.
- **Minimal Fix**: Mock `runtime.installed_distributions` in test or point to a fixture directory.

### Defect 7: `test_http_api_layer.py` Mocks Real Queue And All Core Services
- **Severity**: P2
- **Location**: `workbench/tests/test_http_api_layer.py:112-144`
- **Root Cause**: Fixture `api` monkeypatches `module.JOBS` with a static dummy class whose `list` method returns `[{"job_id": "job-1", "status": "queued"}]`. It also mocks `guarded_plan`, `require_launch_ready`, and `read_evidence_ledger`.
- **Reproduction**:
  Inspect `workbench/tests/test_http_api_layer.py:139-144`.
- **Expected vs Actual**:
  - Expected: Exercise HTTP routes with real `SingleGpuQueue` instances.
  - Actual: Tests run against static lambda returns; concurrency, state transitions, process group killing, and log streaming never exercised through API.
- **Impact**: Critical bugs in API queue dispatching go undetected by the HTTP test suite.
- **Minimal Fix**: Initialize real `SingleGpuQueue` backed by a temporary directory in fixture.

### Defect 8: Release Verifier J4 Tests An Inline Subshell Rather Than `run.sh` Pilot
- **Severity**: P2
- **Location**: `workbench/docker/release_verify.sh:152-170, 440-443`
- **Root Cause**:
  1. J4(a) executes an inline bash string `printf ... | bash -c '...'`, testing its own embedded snippet instead of `workbench/docker/run.sh`.
  2. J4(a) calculates `j4a_mut=1` (line 165) but never asserts on it (line 187 only checks `j4a == 0 && j4b == 0`).
  3. J4(b) performs a static `grep -q` on `run.sh` source instead of invoking the launcher.
  4. Lines 440-443 contain dead code: an unreachable Python invocation executing `report_path, = sys.argv[1:2]` right before the actual report script at line 451.
- **Reproduction**:
  Inspect `workbench/docker/release_verify.sh:152-170, 440-443`.
- **Expected vs Actual**:
  - Expected: Execute `run.sh pilot limn` against a local stub API server (similar to `mutation_checks.sh`); delete dead lines 440-443.
  - Actual: Static greps and inline subshell string tests; dead code in report generation.
- **Impact**: Regressions in `run.sh` pilot parameter handling or bash execution flow are not caught by J4.
- **Minimal Fix**: Replace inline snippet with execution of `run.sh`, assert `j4a_mut == 0`, and remove lines 440-443.

## D) Probable Defects

1. **Unverified sm_120 Blackwell Kernel Execution (Probable P1)**:
   - Location: `workbench/tests/test_release_blocker_master_e2e.py:535-541`, `workbench/scripts/pc_readiness.py:301-306`
   - Description: All tests claiming GPU verification mock the capability with `'compute_capability': '8.9'` (Ada Lovelace). Blackwell sm_120 requires PyTorch compiled with sm_120 CUDA arch support (cu128). If PyTorch lacks sm_120 kernels, kernel launch fails at runtime with `CUDA error: no kernel image is available for execution on the device`. Zero tests in the repo exercise sm_120 compatibility.
2. **Missing Connect Timeout In `run.sh pilot` (Probable P2)**:
   - Location: `workbench/docker/run.sh:226`
   - Description: `curl -fsS --max-time 120 "${BASE}/api/models"` lacks `--connect-timeout`. When the backend is down or starting up, `run.sh pilot` hangs for 120 seconds before failing.
3. **M5 In `mutation_checks.sh` Is A Static Grep, Not A Mutation (Probable P2)**:
   - Location: `workbench/docker/mutation_checks.sh:201-208`
   - Description: M5 does not mutate code; it runs `grep -q "driver: nvidia"`. It provides false assurance of mutation testing.
4. **Pytest Run Contamination Of Production Artifact Tree (Probable P2)**:
   - Location: `workbench/backend/runtime.py:923-927, 1073-1077`
   - Description: When tests invoke `resolve_config()` without isolation, capability records and `.workbench_ledger_key` are written directly to `workbench/artifacts/` and the repo root. 1,690 capability files accumulated in the working tree.

## E) Coverage Gaps

1. **Blackwell sm_120 Real Execution Gap**: No test asserts kernel launch, memory allocation, or tensor operations on sm_120 architecture.
2. **Job Queue End-to-End HTTP Lifecycle Gap**: No integration test runs `POST /api/jobs` -> `SingleGpuQueue` execution -> process group monitoring -> completion report without mocking.
3. **Acceptance Test Negative Assertions Gap**: `acceptance.sh` Group A accepts 200 or 503; Group I accepts 200 or 404. Neither asserts error body structures or negative fault injection.
4. **Setup Stage Pipeline Execution Gap**: In `test_strict_acceptance_matrix.py:191-199`, `test_14_and_15` only checks that stage names exist in `stages_for()`; it never runs `setup` or validates environment provisioning.
5. **Frontend Runtime Verification Gap**: Groups B and H in `acceptance.sh` and J10 in `release_verify.sh` rely strictly on `grep` across source files and Vite dist bundles. No DOM rendering or API contract integration test exists for the frontend.

## F) Minimal Fixes

1. **Fix `acceptance.sh` Group E (Defect 1)**:
   Point check to `doc["models"]["limn"]`:
   ```bash
   limn_model = doc.get("models", {}).get("limn", {})
   assert limn_model.get("gpu_verified") is False, "readiness must not claim GPU verified without probe"
   ```
2. **Fix `release_verify.sh` JSON Loader (Defect 2)**:
   Wrap JSON parsing in a helper that swallows decode errors per file:
   ```python
   def _safe_load(p):
       try: return _json.loads(open(p, encoding="utf-8").read())
       except Exception: return None
   ```
3. **Fix `acceptance.sh` Group G (Defect 3)**:
   Require `len(jobs) > 0` before checking status enum; emit SKIP if no jobs exist.
4. **Fix `release_verify.sh` Exit Code & Verdict (Defect 4)**:
   Distinguish GPU pending from deployment incomplete; exit nonzero if mandatory deployment checks did not run.
5. **Fix `mutation_checks.sh` M4 (Defect 5)**:
   Assert on specific stderr text `variable is not set` rather than generic nonzero exit code.
6. **Fix `run.sh` Curl Timeouts**:
   Add `--connect-timeout 5` to all `curl` commands in `workbench/docker/run.sh`.
7. **Clean Corrupted Artifacts**:
   Delete empty/corrupted JSON files from `workbench/artifacts/gpu_capabilities/` and isolate pytest fixture artifact roots.

## G) Required Regression Tests

1. **`test_acceptance_group_e_fails_on_gpu_bypass()`**:
   Inject synthetic document where `models.limn.gpu_verified = True` without probe; assert `acceptance.sh` Group E fails.
2. **`test_acceptance_group_g_refuses_empty_jobs()`**:
   Run Group G against empty `/api/jobs`; assert it returns non-zero or SKIP, never PASS.
3. **`test_release_verify_handles_corrupt_capability_json()`**:
   Place 0-byte `.json` in `gpu_capabilities/`; run capability resolver; assert it skips corrupt record and reads remaining records without throwing `JSONDecodeError`.
4. **`test_mutation_m4_fails_on_docker_daemon_error()`**:
   Mock `docker compose` to fail with `Cannot connect to Docker daemon`; assert M4 reports FAIL, not DETECTED.
5. **`test_http_api_jobs_real_queue_integration()`**:
   Instantiate FastAPI app with real `SingleGpuQueue` and test full lifecycle of a synthetic job through ASGI `POST /api/jobs` to `GET /api/jobs`.
Can this implementation fail during the real first run on an RTX 5070 Ti PC? YES — The test suite uses synthetic seams pinned to sm_89 (Ada Lovelace) and never exercises sm_120 Blackwell kernel compatibility; release_verify.sh unconditionally crashes on GPU capability resolution due to a 0-byte corrupt test artifact in workbench/artifacts/gpu_capabilities/; acceptance.sh Group E passes on an empty list without verifying GPU status; and acceptance.sh Group G falsely passes on an empty job queue.
