# AUDIT F: Adversarial & Test Integrity Audit Report

## 1. Executive Summary & Scope

Audit of entire workbench test suite (`workbench/tests/`), shell harnesses (`workbench/docker/release_verify.sh`, `workbench/docker/mutation_checks.sh`, `workbench/docker/acceptance.sh`), and the four Phase-1 defect fixes in `pc.sh`, `workbench/scripts/verify_run.py`, and `workbench/tests/test_pc_launcher.py`.

Findings: Existing test suite provides strong regression pinning for file formats, hashes, and isolated unit functions, but exhibits significant coverage gaps: extensive mock-only testing of the execution pipeline, weak source-grep assertions in place of behavioral verification, empty-list false passes in mutation testing, and a critical defect in `verify_run.py` where synthetic data and contradictory macro metrics pass as verified evidence.

---

## 2. Assessment of Existing Test Suite & Harnesses as Evidence

### 2.1 False PASS & Empty-List PASS
- **`workbench/docker/mutation_checks.sh:180-186` (M4 Compose Guard)**: Evaluates `if docker compose -f ... --env-file /dev/null config >/dev/null 2>&1; then FAILED; else PASSED; fi`. When Docker daemon is stopped, CLI missing, or permissions fail, `docker compose config` exits non-zero. Script takes `else` branch and reports `DETECTED` (false PASS on broken infrastructure).
- **`workbench/docker/mutation_checks.sh:202-208` (M5 GPU Override)**: Tests `grep -q "driver: nvidia" "${HERE}/docker-compose.gpu.yml"`. Passes unconditionally on static string presence; never verifies whether daemon accepts compose definition or if NVIDIA container runtime is present.
- **`workbench/docker/release_verify.sh:710-715` (Offline False PASS)**: When executed with `--offline`, live checks J1, J2, J3, J11 are marked `PENDING`. Bash logic evaluates `if (( fails > 0 ))`. Because `fails == 0`, script prints `RELEASE VERIFIER: no unmet criteria (5 pending)` and exits 0, masking complete absence of deployment verification.
- **`workbench/docker/acceptance.sh:43-56, 149-182`**: If backend server is down, groups A, C, G, I report `SKIP`. Script exits 0 with `ACCEPTANCE PASSED (skipped: ...)`, allowing unverified releases to pass CI.

### 2.2 Mock-Only Success & Shim Illusions
- **`workbench/tests/test_pc_launcher.py` (All 32 tests)**: Uses sandbox shim `bin/docker`, `bin/curl`, `bin/python3`. Tests assert strictly against shim-generated JSON logs. No test exercises real Docker daemon, live bind mount behavior, real GPU passthrough, or real container lifecycle.
- **`workbench/tests/test_http_api_layer.py:138-142`**: Monkeypatches `module.JOBS` with dummy class returning static lambda `{"job_id": "job-1", "status": "queued"}`. Real `SingleGpuQueue`, background worker thread, DuckDB state tracking, process group isolation, and cancellation never execute in API tests.
- **`workbench/tests/test_release_blocker_master_e2e.py:534-539`**: GPU capability verification simulated via `runtime.set_gpu_probe_for_tests` returning hardcoded Ada Lovelace `compute_capability: '8.9'`. Real CUDA kernel launch, PyTorch CUDA initialization, and Blackwell (`sm_120`) execution never run in E2E suite.

### 2.3 Weak Source-Grep Assertions
- **`workbench/docker/release_verify.sh:216` (J4b Bundle Preference)**: Uses literal line grep `grep -q "bundles\[0\].get('bundle_id') or m\['checkpoints'\]\[0\]\['checkpoint_id'\]" "${HERE}/run.sh"`. Any valid code refactor or formatting change triggers false FAIL; putting exact string in a comment triggers false PASS.
- **`workbench/docker/release_verify.sh:254` (J5 GPU Contract)**: Asserts `"probe_source" in src and "GPU_PROBE_OVERRIDE_ENV" in src` via `inspect.getsource(runtime)`. Validates string presence in source text, not runtime enforcement.
- **`workbench/docker/acceptance.sh:289-299` (Group H Threshold Copy)**: Uses `grep -rq "phán quyết audit chuẩn"` on `App.tsx` and `dist/`. Never verifies UI component render, event binding, or gate calculation logic.
- **`workbench/tests/test_operator_runtime.py:62-65`**: Asserts `"WORKBENCH_BACKEND_URL" in config` on `vite.config.mts` text instead of verifying proxy resolution.

### 2.4 State Pollution & Mutation of Production Data
- **`workbench/docker/release_verify.sh:660-667`**: In `gpu_capability_scan`, calls `os.remove(_path)` for any unreadable or corrupted JSON file in `workbench/artifacts/gpu_capabilities/`. Verification script actively mutates/deletes artifact data on disk.
- **`workbench/docker/mutation_checks.sh:49-58`**: Mutates production code in-place (`run.sh`, `paper_gate.py`). If process receives SIGKILL before exit trap restores backups from `workbench/artifacts/.mutation_backups`, production files remain permanently corrupted.

---

## 3. Adversarial Break Attempts on the Four Phase-1 Defect Fixes

### 3.1 Break Attempt 1: Verification Running on Host Instead of Container
- **Scenario**: Legitimate run produced inside container with ledger HMAC key stored at container path `XDG_CONFIG_HOME=/artifacts/config` (`/artifacts/config/cir/workbench_ledger_key`). Host verification runs `python3 workbench/scripts/verify_run.py`.
- **Command**:
  ```bash
  python3 -c "
  import tempfile, os, subprocess
  from pathlib import Path
  with tempfile.TemporaryDirectory() as tmp:
      t = Path(tmp); repo = t / 'repo'; repo.mkdir()
      artifacts = repo / 'workbench' / 'artifacts'; artifacts.mkdir(parents=True)
      cont_config = artifacts / 'config' / 'cir'; cont_config.mkdir(parents=True)
      (cont_config / 'workbench_ledger_key').write_bytes(b'0' * 32)
      env = dict(os.environ)
      env['CIR_REPO_ROOT'] = str(repo)
      env['WORKBENCH_ARTIFACTS_ROOT'] = str(artifacts)
      env['XDG_CONFIG_HOME'] = str(t / 'host_config')
      cmd = ['python3', 'workbench/scripts/verify_run.py', '--run-id', 'r1', '--model', 'limn', '--checkpoint', 'base_iter0_all_categories', '--protocol', 'fashioniq_val_split', '--json']
      p = subprocess.run(cmd, env=env, capture_output=True, text=True)
      print('RC:', p.returncode, 'STDOUT:', p.stdout)
  "
  ```
- **Observed Result**:
  `RC: 1`
  `STDOUT: {"verified": false, "reasons": ["aggregate report not found: ...", "the evidence ledger is missing or its HMAC chain does not verify", ...], ...}`
- **Analysis**: Host execution resolves `XDG_CONFIG_HOME` to `~/.config`, cannot locate container key at `/artifacts/config`, and rejects legitimate container runs (False FAIL). Phase-1 fix routing `pc.sh` verification through `run_verifier_in_container` (`docker exec -T cir-workbench`) is mandatory.

### 3.2 Break Attempt 2: Container Down When Verification Needed
- **Scenario**: Verification requested while container `cir-workbench` is stopped or crashed.
- **Command**:
  `.venv/bin/pytest workbench/tests/test_pc_launcher.py -k "test_verification_fails_with_named_reason_when_container_not_running"`
- **Observed Result**:
  `1 passed in 1.40s`. Sandbox output:
  `Giai đoạn thất bại (Failed stage) : VERIFICATION`
  `Chi tiết lỗi: Container cir-workbench không chạy. Xác thực bằng chứng thực thi khoa học phải chạy bên trong container execution environment.`
  `Exit code: 1`
- **Analysis**: Phase-1 fix in `pc.sh:883-893` holds: `run_verifier_in_container` checks `docker ps -q --filter name=^cir-workbench$` and fails closed via `fail_stage` without host fallback.

### 3.3 Break Attempt 3: LIMN Report with Unexecuted Extraction, Contradictory Macro, or Synthetic Marker
- **Scenario**: Test whether `verify_run.py` detects: (a) `is_synthetic: True` / `data_kind: "synthetic"`, and (b) macro `observed_metrics` blatant contradiction against category averages (e.g. macro `r10: 99.9` vs category mean `27.93`).
- **Command**:
  ```bash
  .venv/bin/python -c "
  import tempfile, os, json, hashlib
  from pathlib import Path
  from workbench.tests.test_pc_launcher import _create_corroborated_run
  import workbench.scripts.verify_run as vr
  from workbench.backend.operator_config import WorkbenchConfig
  from workbench.scripts.evaluate_models import append_evidence_entry

  with tempfile.TemporaryDirectory() as tmp:
      sandbox = Path(tmp); run_id = 'run_macro_synthetic_test'
      _create_corroborated_run(sandbox, run_id=run_id)
      rep_file = sandbox / 'workbench' / 'artifacts' / 'reports' / f'limn_base_iter0_all_categories_{run_id}_aggregate.json'
      data = json.loads(rep_file.read_text())
      data['is_synthetic'] = True
      data['data_kind'] = 'synthetic'
      data['observed_metrics'] = {'r10': 99.9, 'r50': 99.9, 'mean': 99.9}
      rep_bytes = json.dumps(data, indent=2).encode('utf-8')
      rep_file.write_bytes(rep_bytes)
      actual_digest = hashlib.sha256(rep_bytes).hexdigest()
      (sandbox / 'workbench' / 'artifacts' / 'evidence_ledger.jsonl').unlink()
      config = WorkbenchConfig(CIR_REPO_ROOT=sandbox, CIR_DATA_ROOT=sandbox/'data', FASHIONIQ_ROOT=sandbox/'data'/'FashionIQ', WORKBENCH_HOST='127.0.0.1', WORKBENCH_BACKEND_PORT=8000, WORKBENCH_FRONTEND_PORT=5173, WORKBENCH_CHECKPOINT_ROOT=sandbox/'workbench'/'artifacts'/'checkpoints', WORKBENCH_RESULTS_ROOT=sandbox/'workbench'/'artifacts'/'results', WORKBENCH_THIRD_PARTY_ROOT=sandbox/'workbench'/'third_party')
      os.environ['WORKBENCH_LEDGER_KEY'] = 'mock_ledger_key_for_test_harness_32b!'
      rec = append_evidence_entry(config, {'run_id': run_id, 'model_id': 'limn', 'checkpoint_id': 'base_iter0_all_categories', 'protocol_id': 'fashioniq_val_split', 'return_code': 0, 'invocation_status': 'INVOCATION_COMPLETED', 'report_digest': actual_digest})
      proof_file = sandbox / 'workbench' / 'artifacts' / 'reports' / f'limn_base_iter0_all_categories_{run_id}_completion.json'
      proof = json.loads(proof_file.read_text())
      proof['report_digest'] = actual_digest
      proof['ledger_entry_digest'] = rec['entry_digest']
      proof_file.write_text(json.dumps(proof, indent=2))
      os.environ['CIR_REPO_ROOT'] = str(sandbox)
      os.environ['WORKBENCH_ARTIFACTS_ROOT'] = str(sandbox / 'workbench' / 'artifacts')
      os.environ['WORKBENCH_CHECKPOINT_ROOT'] = str(sandbox / 'workbench' / 'artifacts' / 'checkpoints')
      os.environ['WORKBENCH_RESULTS_ROOT'] = str(sandbox / 'workbench' / 'artifacts' / 'results')
      verdict = vr.verify(run_id, 'limn', 'base_iter0_all_categories', 'fashioniq_val_split')
      print('RESULT:', 'verified:', verdict['verified'], 'reasons:', verdict['reasons'], 'is_synthetic:', data['is_synthetic'], 'macro:', data['observed_metrics'], 'categories:', data['category_metrics'])
  "
  ```
- **Observed Result**:
  `RESULT: verified: True reasons: [] is_synthetic: True macro: {'r10': 99.9, 'r50': 99.9, 'mean': 99.9} categories: {'dress': {'r10': 25.5, 'r50': 55.2}, 'shirt': {'r10': 28.1, 'r50': 58.4}, 'toptee': {'r10': 30.2, 'r50': 60.1}}`
- **Analysis (CONFIRMED DEFECT)**: `verify_run.py:112-127` verifies only that `category_metrics` has dress, shirt, toptee with finite float values. It **completely ignores** `report.get("is_synthetic")`, `report.get("data_kind")`, and never checks whether the macro `observed_metrics` equals the average of the category recalls! As a result, a declared synthetic report and a report with fabricated macro metrics passes verification with `verified: True`, and `pc.sh:995` / `pc.sh:1191` prints `Xác thực Sổ cái (Ledger) : HỢP LỆ (CORROBORATED)`.

### 3.4 Break Attempt 4: Container Dataset Mount Resolving Elsewhere
- **Scenario**: Container is running with `/data/FashionIQ` mapped to an incorrect host directory.
- **Subcase 4a (Active Job Running)**:
  `.venv/bin/pytest workbench/tests/test_pc_launcher.py -k "test_container_dataset_mount_source_mismatch_refused"`
  Observed: `1 passed in 0.86s`. Container reuse refused with `Giai đoạn thất bại (Failed stage) : CONTAINER_REUSE`.
- **Subcase 4b (No Active Job)**:
  `pc.sh:796-798` executes `"${RUN_SH}" up` and sets `container_running=1`.
- **Analysis (CONFIRMED DEFECT)**: After `"${RUN_SH}" up` executes, `pc.sh` **never re-verifies** that the restart resolved the mount mismatch. If the mismatch arose from an override variable or host `.env` that was not corrected, the restarted container still has the wrong mounts, but `pc.sh` proceeds directly to readiness polling and model evaluation.

### 3.5 Break Attempt 5: Configuration Correction Starting Docker Twice
- **Scenario**: Configuration mismatch triggers restart via `run.sh up`.
- **Command**:
  `.venv/bin/pytest workbench/tests/test_pc_launcher.py -k "test_incompatible_configuration_correction_invokes_docker_startup_exactly_once"`
- **Observed Result**:
  `1 passed in 1.31s`. Log recorded exactly 1 `docker compose up` call.
- **Analysis**: Phase-1 fix changing `container_running=0` to `container_running=1` in `pc.sh:798` successfully prevents duplicate container startup when mismatch is handled.
- **Residual Defect**: In `pc.sh:612-616`, if `c_id` exists but `/api/jobs` returns non-200 (backend still initializing), `container_running` remains 0, triggering an uncoordinated `run.sh up` at line 805 without inspecting active jobs.

---

## 4. Defect Findings Table

| ID | Classification | Severity | Defect Summary | Root Cause | File:Line |
|---|---|---|---|---|---|
| D1 | CONFIRMED | P1 | `verify_run.py` accepts synthetic runs and contradictory macro metrics as corroborated evidence | `verify_run.py` lacks validation for `is_synthetic`/`data_kind` and never cross-checks `observed_metrics` against category averages | `workbench/scripts/verify_run.py:112-127` |
| D2 | CONFIRMED | P2 | `mutation_checks.sh` M4 reports false PASS when Docker daemon is down | Inverted logic: tests `if docker compose ... config; then FAIL; else PASS; fi`; infrastructure failure triggers PASS | `workbench/docker/mutation_checks.sh:180-186` |
| D3 | CONFIRMED | P2 | `release_verify.sh` exits 0 under `--offline` with zero live container checks executed | Line 710 checks only `fails > 0`; pending checks are ignored in exit code | `workbench/docker/release_verify.sh:710-715` |
| D4 | CONFIRMED | P2 | `pc.sh` fails to re-verify container mount/adaptation after restart | After `run.sh up`, `pc.sh` sets `container_running=1` and proceeds without verifying that new container satisfies requirements | `pc.sh:797-807` |
| D5 | CONFIRMED | P3 | `acceptance.sh` Group H uses weak string grep instead of gate calculation test | `grep -rq "phán quyết audit chuẩn"` matches comments/strings in source and built bundle | `workbench/docker/acceptance.sh:289-302` |
| D6 | PROBABLE | P2 | Race condition between `/api/jobs` idle check and `run.sh up` container restart in `pc.sh` | No lock held between inspecting job queue and restarting container | `pc.sh:626-798` |

### Detailed Defect Analyses & Minimal Corrective Actions

#### Defect 1: `verify_run.py` Blind to Synthetic Marker and Macro Contradictions
- **Root Cause**: `verify_run.py` only validates `report_path`, HMAC ledger entry, proof manifest, and category float bounds. It does not check `data_kind`/`is_synthetic` and does not compute `mean(r10)` / `mean(r50)` across categories.
- **Reproduction**: See §3.3 command; returns `verified: True` for synthetic report with `r10: 99.9` vs category mean `27.93`.
- **Minimal Corrective Action**: In `workbench/scripts/verify_run.py`:
  ```python
  if report.get("is_synthetic") or report.get("data_kind") in ("mock", "synthetic"):
      reasons.append("the report is declared synthetic/mock data, not empirical evidence")
  if model_id == "limn" and categories:
      for key in ("r10", "r50"):
          cat_vals = [categories[c].get(key) for c in LIMN_CATEGORIES if _is_recall(categories[c].get(key))]
          if len(cat_vals) == len(LIMN_CATEGORIES):
              expected_macro = sum(cat_vals) / len(cat_vals)
              obs_val = (report.get("observed_metrics") or {}).get(key)
              if obs_val is None or abs(obs_val - expected_macro) > 0.05:
                  reasons.append(f"macro {key} ({obs_val}) disagrees with category average ({expected_macro:.2f})")
  ```

#### Defect 2: `mutation_checks.sh` M4 False PASS on Docker Daemon Failure
- **Root Cause**: Command failure due to stopped daemon is treated as compose rejecting empty environment variables.
- **Minimal Corrective Action**: Verify docker daemon connectivity before running compose mutation, and assert specific stderr error message:
  ```bash
  out="$(docker compose -f "${HERE}/docker-compose.yml" --env-file /dev/null config 2>&1 || true)"
  if echo "$out" | grep -q "HOST_FASHIONIQ_ROOT"; then PASSED+=(M4); else FAILED+=(M4); fi
  ```

#### Defect 3: `release_verify.sh` Offline Exit Code Honesty
- **Root Cause**: `fails > 0` condition allows zero failures with 5 pending journeys to exit 0.
- **Minimal Corrective Action**: In `workbench/docker/release_verify.sh:710`:
  ```bash
  if (( fails > 0 )); then exit 1; fi
  if (( pendings > 0 )) && (( OFFLINE == 0 )); then exit 2; fi
  ```

---

## 5. Practical Coverage-Gap Report & High-Confidence Recommendations

Only 3 targeted tests are needed to materially improve release confidence without adding bloat:

1. **Negative Verifier Test (`test_verify_run_rejects_synthetic_and_macro_disagreement`)**:
   Add test in `workbench/tests/test_release_blocker_adversarial_final.py` feeding `verify_run.py` a signed synthetic report and a signed report where category average does not match `observed_metrics`. Assert `verified is False`.
2. **Post-Restart Configuration Re-verification Test in `pc.sh`**:
   Add test verifying that if `run.sh up` fails to fix a mount mismatch, `pc.sh` aborts with `CONTAINER_REUSE` error rather than continuing into evaluation.
3. **Blackwell Kernel & GPU Capability Integration Test (`sm_120`)**:
   Add non-mock integration test that probes real PyTorch CUDA device capability and verifies compatibility against compute capability `12.0` (Blackwell sm_120), rejecting legacy sm_89 mocks.

---

## 6. Uninspected Areas

- Full Docker image build (`docker build`) and in-container `vite preview` proxy execution (per contract disk constraints: ~11 GB free).
- Multi-gigabyte weights downloading and model inference execution for non-LIMN models (CSMCIR, DCNet, CLVC).
- Upstream third-party Git checkouts in `workbench/third_party/`.

---

Can this fail on the real first run on an RTX 5070 Ti PC?
YES — `verify_run.py` certifiably validates synthetic runs and contradictory macro metrics as genuine evidence; `pc.sh` proceeds into evaluation without re-checking container mounts if `run.sh up` restarts with uncorrected environment variables; and all test-suite GPU verification paths remain mocked to sm_89 without ever executing an sm_120 Blackwell CUDA kernel.
