# AUDIT_A — slice A: `pc.sh`, `workbench/docker/`, `workbench/config/`, `.dockerignore`, `serve_workbench.py`, `operator_config.py`

Iterative audit round 02 / slice A. Branch `research/cir-failure-workbench`, base commit `92ecbd9` + 3 uncommitted fix waves.
Audit method: READ-ONLY. No code files modified. Targeted checks run on live stack; temporary probes isolated.
Target files: 13 first-party files in slice A (`pc.sh`, `workbench/docker/*`, `workbench/config/*`, `.dockerignore`, `serve_workbench.py`, `operator_config.py`).

---

## 1. Round-01 Open Items Follow-Up

### B-1 / A5 — `_gpu_capability_key` vs `verify_environment` Interpreter Mismatch
- Classification: **CONFIRMED**
- Severity: **P1**
- Location: `workbench/backend/runtime.py:1030-1047` vs `workbench/backend/runtime.py:1274-1288`.
- Evidence: `_gpu_capability_key` derives key strictly from `os.environ[python_env_var]`. When env var unset, `configured=""`, `fingerprint=None`. In contrast, `verify_environment` adopts interpreter from `environment_assignment.json` (`persisted.get("interpreter")`). When `verify_environment` calls `_reuse(authoritative=True)` at line 1222, `_gpu_capability_key(model)` computes key with `configured=""` and looks up `<empty_hash>.json` instead of the recorded file `<assigned_interpreter_hash>.json`.
- Live reproduction:
  ```bash
  python3 -c '
  from workbench.backend import runtime; from workbench.backend.registry import load_registry
  m = next(m for m in load_registry()["models"] if m["model_id"] == "limn")
  k1 = runtime._gpu_capability_key(m)
  import os, sys; os.environ["WORKBENCH_LIMN_PYTHON"] = sys.executable
  k2 = runtime._gpu_capability_key(m)
  print(k1 == k2, k1, k2)
  '
  # Output: False 28c012ba... 4fd57a0c...
  ```
- Impact: Subprocesses reading capability without explicit env var set report `UNCONFIGURED` and fail to reuse valid persisted GPU capabilities. False FAIL on target PC when running doctor/pipeline commands outside autopilot shell.

### B-2 — `_cached_gpu_capability(authoritative=False)` Omits Interpreter Fingerprint
- Classification: **CONFIRMED**
- Severity: **P2**
- Location: `workbench/backend/runtime.py:1193-1220`.
- Evidence: In `_cached_gpu_capability(authoritative=False)`, iteration over `_GPU_CAPABILITY.values()` and `directory.glob("*.json")` checks `model_id`, `interpreter` path string, `dependency_fingerprint`, `host_identity`, `probe_source`, and HMAC signature. It never compares `payload.get("interpreter_fingerprint")`.
- Impact: If packages in the environment change while interpreter path stays identical, non-authoritative caller (advisory probe / status check) returns stale `GPU_RUNTIME_READY` on drifted environment.

### B-4 / B-6 / B-9 / B-11 — In-Memory Capability Map Filter & Adaptation File Hop
- Classification: **KNOWN_LIMITATION** (B-4, B-6) / **CONFIRMED** (B-9, B-11)
- Severity: **P2** (B-9), **P3** (B-4, B-6, B-11)
- Location: `workbench/backend/runtime.py:1193, 1223`, `workbench/docker/release_verify.sh:628-632`.
- Evidence:
  1. `_GPU_CAPABILITY` cache in `runtime.py:1193-1200` re-checks `probe_source` when reading cache, but `record_gpu_capability` line 976 caches override probes without tag distinction.
  2. `release_verify.sh:628` calls `runtime_adaptation(model, undeclared_is_error=False)` which re-reads `models.yaml` from disk on every capability check rather than using registry memory structure.
  3. `runtime.py:1095-1106`: Conflict detection prints warning to stderr for every single capability lookup if two keys exist (over 600 spam lines during `release_verify.sh`).

### F4 — `_proof_stamp == "absent"` With Dev Opt-in Certifies Unproven Report
- Classification: **EXPECTED_BEHAVIOR** (Restricted to Development Mode)
- Severity: **P3**
- Location: `workbench/backend/main.py:240-244, 332-358`.
- Evidence: `_certification_mode()` only returns `"development"` if `_development_process()` (`pytest` or `WORKBENCH_DEVELOPMENT_MODE=1`) AND `WORKBENCH_ALLOW_DEVELOPMENT_CERTIFICATES=1`. In production deployment, it returns `"proof_unavailable"` (`state="proof_unavailable"`, `certified=False`). In development, receipts are explicitly marked `development_mode=True` and log warning to stderr. Cannot trigger in production.

### F8-release — `release_verify.sh` Substring Check Promotes Refuted Run to PASS
- Classification: **CONFIRMED**
- Severity: **P1**
- Location: `workbench/docker/release_verify.sh:752, 764-766, 782-785`.
- Evidence:
  ```python
  _has_proof = not any("completion proof" in r for r in _reasons)
  ...
  if not _is_synthetic and _rep.get("data_kind") == "experiment" and _has_proof:
      _inference_evidence.append(_run_id)
  ```
  If a run fails ledger corroboration (e.g. `witness.return_code != 0`, ledger report digest mismatch, or missing ledger entry), `_reasons` contains ledger failure strings, NOT `"completion proof"`. `_has_proof` evaluates to `True`. Because `data_kind == "experiment"`, line 765 appends `_run_id` to `_inference_evidence`. Line 783 then marks `out["dimensions"]["MODEL_INFERENCE"]["status"] = "PASS"`.
- Impact: A refuted run with failed exit code or tampered ledger grants `MODEL_INFERENCE: PASS`.

### D-F1 — Unregistered Model With No Metrics Certifies As Valid Research Evidence
- Classification: **CONFIRMED**
- Severity: **P1**
- Location: `workbench/scripts/verify_run.py:427-446`.
- Evidence: In `metric_evidence_reasons()` for `outcome == benchmarks.SELECTION_UNREGISTERED`:
  ```python
  honest = list(_extraction_reasons(report))
  observed = report.get("observed_metrics") or {}
  honest.extend(f"the report's {key} is not a finite Recall in [0, 100]: {value!r}"
                for key, value in observed.items()
                if key in paper_gate.METRIC_KEYS and not _is_recall(value))
  ```
  If `report` sets `metric_extraction_available: True`, `extraction_status: "EXTRACTION_COMPLETED"`, but `observed_metrics: None` (or `{}`) and `category_metrics: None`: loop over `observed` executes 0 times. `categories` check skipped. Returns `honest=[]`. Report with NO numbers certifies as `is_research_evidence=True`.

### A9 — Empty In-Memory Queue Demotes Healthy Finished Stack to `ENGINEERING_INCOMPLETE`
- Classification: **KNOWN_LIMITATION / FALSE_FAILURE**
- Severity: **P2**
- Location: `workbench/docker/release_verify.sh:370-377, 531, 811-820`.
- Evidence: J8 checks `/api/jobs`. If empty, line 374 raises `SystemExit(3)` (`NO_JOBS_OBSERVED`) -> J8 is `PENDING`. `jobs.py:489-514` (`recover_interrupted`) only loads incomplete jobs into RAM, never completed ones. After restart on healthy host with completed runs, `/api/jobs` is empty. Line 817:
  ```python
  _gpu_or_research_pending = any(d in mandatory_pending for d in ("GPU_RUNTIME", "MODEL_INFERENCE", "MODEL_EVALUATION", "SCIENTIFIC_OUTPUT"))
  out["verdict"] = "GPU_VALIDATION_PENDING" if _gpu_or_research_pending else "ENGINEERING_INCOMPLETE"
  ```
  When GPU/models PASS, `_gpu_or_research_pending` is False. The sole pending dimension is `OPERATIONS` (J8). Verdict becomes `ENGINEERING_INCOMPLETE`. Clean restart turns passing host into degraded verdict.

---

## 2. Audit of Recent Fixes (Last Wave)

### Deleted `sys.argv[1:2]` Block in `release_verify.sh`
- Classification: **EXPECTED_BEHAVIOR / VERIFIED_FIXED**
- Severity: N/A
- Location: `workbench/docker/release_verify.sh:507, 825-828`.
- Evidence: Previous `report_path, = sys.argv[1:2]` crashed when options were passed. Replaced with `"${PY}" - "$REPORT" <<'EOF'` and `sys.argv[1]`.
- Verified live:
  ```bash
  bash workbench/docker/release_verify.sh --report /tmp/test_report.json --base http://127.0.0.1:8000 --offline
  # Output: report written: /tmp/test_report.json verdict=GPU_VALIDATION_PENDING (RC=0)
  ```
  Every documented flag combination (`--report`, `--base`, `--offline`) parses cleanly.

### Report Verdict Masking in `release_verify.sh`
- Classification: **CONFIRMED**
- Severity: **P1**
- Location: `workbench/docker/release_verify.sh:811-820, 840-845`.
- Evidence:
  1. Under `--offline`, `DEPLOYMENT` and `CONFIGURATION` are PENDING. Line 817 sets `_gpu_or_research_pending = True` (because `GPU_RUNTIME` is pending). Line 818 forces `out["verdict"] = "GPU_VALIDATION_PENDING"`. It completely hides that deployment and container configuration were unverified.
  2. Line 840: `if (( fails > 0 )); then exit 1; fi`. Bash exits with code 0 because pending checks do not increment `fails`. Verifier claims `RELEASE VERIFIER: no unmet criteria (5 pending)` with zero deployment checks executed.

### Cause-Specific Remedies in `pc.sh`
- Classification: **EXPECTED_BEHAVIOR / VERIFIED_FIXED**
- Severity: N/A
- Location: `pc.sh:604-789`.
- Evidence: `inspect_container_config` flags specific causes: `inspect_gpu_mismatch`, `inspect_adaptation_mismatch`, `inspect_mount_mismatch`, `inspect_env_mismatch`. `require_corrected_configuration` names specific remedy matching the cause. Calls `fail_stage` which prints structured error and terminates execution (exit 1). Verified single restart flow: restarts only once when active jobs == 0 and configuration mismatches.

### `proof_unavailable` Reuse Path in `pc.sh`
- Classification: **EXPECTED_BEHAVIOR (Documented Policy) with Narrow Scope**
- Severity: **P2** (Documentation nuance)
- Location: `pc.sh:1227-1280, 1488-1515`.
- Evidence:
  1. `pc.sh` reuses run only if authenticated by evidence ledger (`v.get("execution_authenticated")`, `witness.return_code == 0`, `INVOCATION_COMPLETED`) AND `verify_run.py` rejection reasons are exclusively missing proof file/declared output (`proof_only == True`).
  2. If any other defect exists (tampered digest, wrong metrics, failed status), `pc.sh` rejects reuse and exits 1.
  3. `validate_and_summarize` prints explicit notice: `Xác thực Sổ cái (Ledger) : HỢP LỆ SỔ CÁI (TỆP PROOF KHÔNG CÒN TỒN TẠI)`.
  4. Non-leakage: `release_verify.sh:752` rejects unproven runs (`_has_proof=False`), so unproven runs never promote release verifier dimensions to PASS.

---

## 3. Sweeping Slice A for Round-01 Misses

### Docker/Host Ledger Key Disconnect & Inaccurate Remedy Command
- Classification: **CONFIRMED**
- Severity: **P0**
- Location: `workbench/docker/Dockerfile:86`, `workbench/docker/docker-compose.yml:40`, `workbench/scripts/evaluate_models.py:1164-1198`, `workbench/backend/runtime.py:1078-1106`.
- Evidence:
  1. `Dockerfile:86` and `docker-compose.yml:40` set `XDG_CONFIG_HOME=/artifacts/config`. Inside container, operator key is written to `/artifacts/config/cir/workbench_ledger_key` (host: `<artifacts>/config/cir/workbench_ledger_key`).
  2. Host process (`release_verify.sh`, local scripts) runs with host `XDG_CONFIG_HOME` (default `~/.config/cir/workbench_ledger_key`) or legacy `<repo>/.workbench_ledger_key`. It NEVER reads `<artifacts>/config/cir/workbench_ledger_key`.
  3. If host has `<repo>/.workbench_ledger_key` AND `~/.config/cir/workbench_ledger_key` with different contents, `evaluate_models.ledger_hmac_key()` raises `LedgerKeyError: two different ledger keys exist`. Observed live in `release_verify.sh`:
     ```
     AssertionError: the evidence ledger cannot be read, so no boot state is observable: the evidence ledger key is unusable, so no report can be corroborated: two different ledger keys exist (/home/heheboiz/data/cir/.workbench_ledger_key and /home/heheboiz/.config/cir/workbench_ledger_key); keep one file and remove the other, or set WORKBENCH_LEDGER_KEY explicitly to choose one
     ```
  4. Defective remedy text: The error message tells the operator to "set WORKBENCH_LEDGER_KEY explicitly to choose one". But `WORKBENCH_LEDGER_KEY` treats its value as a raw UTF-8 secret string (`configured.strip().encode("utf-8")`), NOT a file path. Setting it to a path changes the key bytes, causing all existing HMAC signatures to fail validation.

### `.dockerignore` Missing Ledger Key and Lock Files
- Classification: **CONFIRMED**
- Severity: **P2**
- Location: `.dockerignore:20-27`.
- Evidence: `.dockerignore` excludes `workbench/config/workbench.env` and `**/.env`. It does NOT exclude `.workbench_ledger_key` or `.workbench_ledger_key.lock`. If docker build context is repo root, any host-generated repo-root key risks being baked into the image, shadowing container key resolution.

### Non-Root UID `node_modules` Permissions
- Classification: **CONFIRMED**
- Severity: **P2**
- Location: `workbench/docker/Dockerfile:64`.
- Evidence: `Dockerfile:64` executes `chown -R 1000:1000 /repo/workbench/frontend/node_modules`. If host operator UID is 1001 (configured via `WORKBENCH_UID` in `docker-compose.yml:22`), container user has no write access to `node_modules`. `vite preview` fails when attempting to create `.vite-temp`.

### Disk Full Threshold in `pc.sh` Preflight
- Classification: **EXPECTED_BEHAVIOR**
- Severity: **P3**
- Location: `pc.sh:295-303`.
- Evidence: Requires >= 25 GB free disk (`free_disk_gb < 25` fails preflight). Observed live on test host:
  `[FAIL] Dung lượng đĩa trống : 5 GB (thấp hơn mức tối thiểu 25 GB)`. Preflight correctly guards against disk exhaustion during multi-GB checkpoint downloads.

### Verified Solid Invariants in Slice A
1. **Dataset Read-Only Protection**: Canonical directories `images/` and `image_splits/` mounted `:ro` (`docker-compose.yml:49-50`). Verified live via `acceptance.sh` and `release_verify.sh` J3.
2. **Active-Job Invariant**: `pc.sh:814-928` checks `/api/jobs` and inspects persisted `queue_state.json`. Refuses container restart or reconfiguration when jobs are queued, preparing, or running.
3. **Single-Startup Invariant**: `pc.sh` checks running container configuration before recreation, starts stack at most once, and re-inspects immediately after startup via `require_corrected_configuration`.
4. **`docker exec` Environment Propagation**: `pc.sh:1056-1067` extracts declared container environment via `docker inspect` and passes `-e` flags explicitly to `verify_run.py`.

---

## 4. Live Script Invocations & Observed Evidence

| Command | Exit Code | Observed Output / Key Findings |
|---|---|---|
| `./pc.sh --self-check` | 0 | `pc.sh self-check passed.` |
| `./pc.sh --doctor` | 0 | Reported RTX 3050 (Compute 8.6), disk 5 GB [FAIL < 25GB], NVIDIA runtime [PENDING]. Fully read-only. |
| `bash workbench/docker/release_verify.sh --offline` | 0 | 7 PASS, 5 PENDING. J1/J2/J3/J8/J11 PENDING. `verdict=GPU_VALIDATION_PENDING` (masks offline deployment status). |
| `bash workbench/docker/release_verify.sh` (live) | 1 | J1 FAIL (`LedgerKeyError: two different ledger keys exist`). Verdict `RELEASE_BLOCKED`. |
| `bash workbench/docker/acceptance.sh` | 0 | Groups A, B, C, D, E, F, H, I PASS. Group G SKIP (`G-status-unverified`, empty queue). |
| `bash workbench/docker/mutation_checks.sh` | 0 | M1-M5 all DETECTED / PASS (6/6 mutants caught). |
| `.venv/bin/pytest workbench/tests/test_pc_launcher.py` | 0 | 45 passed in 87.55s. |
| `.venv/bin/pytest workbench/tests/test_final_hardening_regressions.py` | 0 | 41 passed in 2.64s. |

---

## 5. Scope & What Was NOT Inspected

- **Out of Slice**: `workbench/backend/` internals (outside `operator_config.py`, `runtime.py` capability key paths, and `main.py` report certification), `workbench/frontend/` components (outside built dist checks and Dockerfile node build), model training/inference scripts.
- **Unexercised Operations**: Full docker image rebuild (`docker compose build` not run per contract), destructive git modifications, physical GPU inference on sm_120 (host has RTX 3050, sm_86).

---

## 6. Target Hardware Verdict: Can this fail on the real first run on an RTX 5070 Ti PC?

**YES — It will fail or report false blocker unless two key configuration issues are addressed:**

1. **Ledger Key Conflict / Resolution Failure (P0 Blocker)**:
   If `<CIR_REPO_ROOT>/.workbench_ledger_key` was created during local dev or git checkout while `~/.config/cir/workbench_ledger_key` exists on host, `release_verify.sh` crashes immediately with `AssertionError: two different ledger keys exist`. Furthermore, the container writes its key to `/artifacts/config/cir/workbench_ledger_key`, which host `release_verify.sh` does not read unless host `XDG_CONFIG_HOME` is pointed there.
2. **Capability Resolution Failure via Assignment (P1 Blocker)**:
   On a clean boot where `WORKBENCH_LIMN_PYTHON` is not exported in host shell, `_gpu_capability_key` computes hash from empty string while `verify_environment` resolves mamba interpreter from `environment_assignment.json`. Persistent GPU capability recorded during setup cannot be matched by authoritative readers.
3. **Empty Job Queue Downgrade (P2 False Blocker)**:
   After evaluation finishes and container restarts, `/api/jobs` is empty. J8 is marked `PENDING`, causing `release_verify.sh` to output verdict `ENGINEERING_INCOMPLETE` instead of `RELEASE_READY`.
