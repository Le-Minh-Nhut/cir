# Review Report REV1_LAUNCHER: Docker Deployment and `./pc.sh` One-Command Launcher

Branch: `research/cir-failure-workbench` | HEAD commit: `6e3c979`
Auditor: Independent adversarial audit on launcher subsystem.
Target hardware: Ubuntu x86_64, NVIDIA GeForce RTX 5070 Ti (Blackwell sm_120, 16 GB VRAM), NVIDIA Driver 580+, `nvidia-container-toolkit`.

---

## 1. Scope & Files Inspected

- `pc.sh` (1118 lines, root entrypoint)
- `workbench/docker/run.sh` (273 lines, Docker launcher)
- `workbench/docker/docker-compose.yml` (58 lines)
- `workbench/docker/docker-compose.gpu.yml` (14 lines)
- `workbench/docker/.env.example` (50 lines)
- `workbench/tests/test_pc_launcher.py` (477 lines, 20 executable test cases)
- `workbench/tests/fixtures/pc_launcher/bin/*` (`docker`, `curl`, `nvidia-smi`, `df`, `git` shims)
- Prior audit: `workbench/docs/reviews/final_pc_audit/A.md`

All findings below verified by real command execution and sandbox test harness.

---

## 2. Specific Claim Confirmations & Refutations

### Claim (a): `pc.sh passes WORKBENCH_ENABLE_GPU where compose actually reads it`
**Verdict:** CONFIRMED FOR COMPOSE CLI OVERRIDE; NOT_A_BUG FOR IN-CONTAINER ENVIRONMENT.
- **Trace:**
  - `pc.sh:546,548`: Exports `WORKBENCH_ENABLE_GPU="1"` (or `"0"` on `--no-gpu`).
  - `pc.sh:559-564`: If `.env` newly created from `.env.example`, writes `WORKBENCH_ENABLE_GPU=1` into `.env`. If `.env` already existed, leaves it untouched.
  - `run.sh:76`: Resolves `ENABLE_GPU="$(env_value WORKBENCH_ENABLE_GPU 0)"`. `env_value` prioritizes shell env (`${!key:-}`). Shell export wins.
  - `run.sh:81-83`: When `ENABLE_GPU == 1`, appends `-f ${HERE}/docker-compose.gpu.yml` to compose arguments.
  - `docker compose` receives `-f docker-compose.yml -f docker-compose.gpu.yml`. Device reservation (`driver: nvidia, capabilities: [gpu]`) active.
- **In-container observation:**
  - `docker-compose.yml:28-40` (`environment:`) does NOT declare `WORKBENCH_ENABLE_GPU`.
  - When `.env` has `# WORKBENCH_ENABLE_GPU=1` commented out, `docker compose config` container environment does not contain `WORKBENCH_ENABLE_GPU` (`WORKBENCH_ENABLE_GPU in config: False`).
  - Grep in `workbench/backend/` and `workbench/scripts/` shows 0 callers of `WORKBENCH_ENABLE_GPU`. No backend, runtime, or pipeline code reads this variable.
  - Container GPU verification (`runtime.verify_gpu_runtime`) probes hardware environment directly (`torch.cuda.is_available()`, compiled C extension, kernel execution).
- **Conclusion:** `pc.sh` successfully passes `WORKBENCH_ENABLE_GPU` where it matters: to `run.sh`, which triggers `docker-compose.gpu.yml`.

### Claim (b): `Readiness no longer depends on a slow endpoint`
**Verdict:** CONFIRMED.
- **Previous defect (A-3):** `pc.sh` polled `/api/models` with 5s timeout. With LIMN checkpoints (5.37 GB), `/api/models` took 45–85s due to unmemoized file hashing and 129k SHA256 calls in environment fingerprinting. Resulted in HTTP 000 timeout.
- **Current implementation:**
  - `pc.sh:636-637`: Polling loop now queries `${base}/api/jobs` and `http://127.0.0.1:${frontend_port}`.
  - `main.py:375-376`: `/api/jobs` executes `return JOBS.list()`, an in-memory sub-millisecond call.
  - `pc.sh:712`: `/api/models` moved to `execute_evaluation`, called only after readiness passes, with `--max-time 300` and 2 attempts.
  - Commit `6e3c979` optimized `/api/models` via manifest-backed digest checks and split advisory vs authoritative environment verification (85s -> 6s).
- **Observed:** Sandbox readiness loop succeeds immediately; test harness runs in <0.2s per poll.

---

## 3. Real Program Behavioral Verification (Observed Execution)

Harness execution: `pytest workbench/tests/test_pc_launcher.py` -> `20 passed in 17.58s`.

### 3.1 Help, Doctor, and Self-Check
- `./pc.sh --help`: Exit code 0. Prints complete option listing (`--model`, `--doctor`, `--verify-only`, `--force`, `--timeout`, `--no-gpu`, `--dataset`, `--self-check`).
- `./pc.sh --self-check`: Exit code 0. Validates `env_value` parsing, quote stripping, comment handling, and whitespace trimming on synthetic inputs.
- `./pc.sh --doctor`: Exit code 0. Preflight report outputs all system checks (OS, git, RAM, disk, UID/ownership, Docker daemon, compose, NVIDIA driver, FashionIQ dataset).

### 3.2 Read-Only Doctor
- **Observed:** SHA256 and mtime manifest of sandbox files before and after `./pc.sh --doctor` match 100%. `is_doctor == 1` skips directory creation (`mkdir -p`) and skips container startup.

### 3.3 Strict `.env` Lifecycle
- **Fresh run:** Sandbox without `.env` copies `.env.example`, writes `WORKBENCH_ENABLE_GPU=1` and `WORKBENCH_RUNTIME_ADAPTATION=sm_120`.
- **Existing run:** Sandbox `.env` with custom ports (9876/5432) and dataset paths preserves values verbatim.
- **Strict parsing:** Quoted strings with `#` preserved (`WORKBENCH_TAG="run #1 with special characters"`).
- **Malformed protection:** `WORKBENCH_BACKEND_PORT=abc` exits non-zero at `validate_env` before invoking Docker.

### 3.4 Override Selection & Port Handling
- Default: `WORKBENCH_ENABLE_GPU=1` -> `run.sh` injects `docker-compose.gpu.yml`.
- `--no-gpu`: `WORKBENCH_ENABLE_GPU=0` -> `run.sh` loads only base `docker-compose.yml`.

### 3.5 Readiness Polling Safety
- `pc.sh:619-624`: `ready_timeout` clamped between 2s and 180s.
- `curl` uses `--max-time 5`. Loop advances by 2s sleep.
- Inducing total backend failure (`MOCK_CURL_ALWAYS_FAIL=1`, `--timeout 2`) exits in 2.63s with `fail_stage "DOCKER_READINESS"`. Hanging is impossible.

### 3.6 Dedup & Resume Behavior
- **Corroborated run exists:** `pc.sh` checks `/api/runs` and `/api/benchmark-runs`. Finds matching completed run -> prints reuse notice, exits 0, makes 0 POST calls to `/api/jobs`.
- **Force re-run:** `--force` bypasses check, posts new evaluation job to `/api/jobs`.
- **Active job running:** Attaches to existing job (`status in {'queued', 'preparing', 'running'}`) without duplicate POST.
- **Concurrent race 409:** API 409 `duplicate_job` response caught at `pc.sh:830-833`, extracts `details.job_id`, attaches cleanly.

### 3.7 Vietnamese Failure Summary
- `pc.sh:46-62` (`fail_stage`): Generates 4-line boxed summary:
  - `Giai đoạn thất bại (Failed stage)`
  - `Chi tiết lỗi (Error details)`
  - `Vị trí nhật ký (Log location)`
  - `Hành động khắc phục (Remedy)`
- Tested on `MOCK_JOB_STATUS=failed`: outputs stage `EVALUATION_EXECUTION`, details `CUDA out of memory`, log path, exits 1.

### 3.8 Signal Handling & Clean Termination
- `trap 'cleanup_on_signal SIGINT' SIGINT`, `trap 'cleanup_on_signal SIGTERM' SIGTERM`.
- Sending `SIGINT` to live `./pc.sh` prints:
  `ĐÃ NHẬN TÍN HIỆU DỪNG (SIGINT)`
  `Bảo toàn nguyên vẹn artifacts, nhật ký và trạng thái hệ thống.`
- Artifact files preserved on disk. Exits with code 130.

### 3.9 Truthful Exit Codes
- Success / Help / Doctor / Self-check: 0.
- Preflight missing tool / Malformed env: 1 or 2.
- Stage failure (readiness, dataset, prepare, probe, eval): 1.
- Signal interrupt: 130.

---

## 4. Defect Findings & Classification

### 4.1 Prior Audit P0 Status
- **A-1 (GPU capability signing key failure in non-root container):**
  - **Verdict:** RESOLVED (commit `6e3c979`).
  - **Evidence:** `runtime.py:1093-1105` now falls back to `XDG_CONFIG_HOME / "cir" / "workbench_ledger_key"`. In container, `XDG_CONFIG_HOME=/artifacts/config`, writable by UID 1000. Verified in Python probe: returns 32-byte key, HMAC signature valid, round-trip cache verification passes.
- **A-3 (Readiness timeout on `/api/models`):**
  - **Verdict:** RESOLVED (commit `6e3c979`).
  - **Evidence:** Readiness queries `/api/jobs` (<1ms). `/api/models` optimized to ~6s and queried later with 300s timeout.

### 4.2 Confirmed Open Defects

#### DEF-01 — P2 — CONFIRMED_BUG: Preflight false-positive on Docker NVIDIA runtime
- **Location:** `pc.sh:402`
- **Code:**
  `if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"' || docker info 2>/dev/null | grep -iq nvidia; then`
- **Root cause:** `grep -iq nvidia` matches any text in unfiltered `docker info` (hostname, node label, storage driver, plugin).
- **Reproduction:** Shimmed Docker returning `Runtimes: runc` and hostname `nvidia-pc` causes `pc.sh` to output `[OK] NVIDIA Container Runtime: Đã cấu hình...` (`has_nvidia_rt=1`).
- **Impact:** Host without `nvidia-container-toolkit` bypasses preflight warning/install prompt. Compose subsequently fails with `could not select device driver "nvidia"`.
- **Fix:** Remove `|| docker info 2>/dev/null | grep -iq nvidia`.

#### DEF-02 — P2 — CONFIRMED_BUG: Preflight disk gate threshold contradicts documentation
- **Location:** `pc.sh:283-298` vs `workbench/docs/PC_ONE_COMMAND_GUIDE.vi.md:30`
- **Root cause:** `pc.sh` allows `>= 10 GB` to pass preflight (`free_disk_gb >= 10`). Documentation specifies minimum 50 GB. LIMN bundle checkpoints are 5.37 GB, conda environment is ~4 GB, plus Docker layers, datasets, and cache.
- **Impact:** A host with 11–18 GB free passes preflight, then fails with `ENOSPC` mid-pipeline during checkpoint download or unpacking.
- **Fix:** Align `pc.sh` warning/error threshold with 50 GB / 25 GB minimum.

#### DEF-03 — P1 — CONFIRMED_BUG: Baked image directory ownership fails for operator UID != 1000
- **Location:** `workbench/docker/Dockerfile:57-59`, `docker-compose.yml:22`, `pc.sh:580`
- **Root cause:** Prebuilt image `cir-workbench:fix2` bakes `/artifacts/mamba` owned by `1000:1000` (mode 0755). When operator on target PC runs with UID != 1000, `pc.sh` exports `WORKBENCH_UID="$(id -u)"`.
- **Impact:** In-container `micromamba create` fails with permission denied during model preparation.
- **Fix:** In Dockerfile, set `chmod 0777 /artifacts /artifacts/mamba`, or document requirement that host operator must run as UID 1000 unless image rebuilt.

#### DEF-04 — P3 — CONFIRMED_BUG: `probe_gpu_capability` prints `None` as recorded capability key
- **Location:** `workbench/scripts/pipeline.py:1403`, `workbench/backend/runtime.py:2040-2042`
- **Root cause:** `verify_gpu_runtime` returns local `payload` before `record_gpu_capability` attached `"capability_key"`.
- **Observed:** Output prints `[OK] GPU capability verified and recorded: None`.
- **Impact:** Cosmetic only (exit code is 0, record is written to disk with key).
- **Fix:** Change `runtime.py:2040` to `return record_gpu_capability(model, payload, config)`.

### 4.3 Unverified Risks

#### RISK-01 — P2 — UNVERIFIED_RISK: Job monitor fallback to `jobs[0]`
- **Location:** `pc.sh:858-859`
- **Code:**
  ```python
  target = next((j for j in jobs if j.get('job_id') == job_id), None)
  if target:
      print(json.dumps(target))
  elif jobs:
      print(json.dumps(jobs[0]))
  ```
- **Risk:** If `jobs` contains multiple items and `target` is temporarily missing during state transitions, launcher grabs `jobs[0]`. If `jobs[0]` is a failed job from an earlier model run, launcher terminates immediately with false failure.
- **Fix:** Delete `elif jobs: print(json.dumps(jobs[0]))`; fall through to query `/api/jobs/${job_id}` directly.

#### RISK-02 — P1 — UNVERIFIED_RISK: Blackwell sm_120 real CUDA execution is PENDING
- **Status:** PENDING (Hardware limitation of development workstation).
- **Evidence:** Development machine has RTX 3050 Laptop GPU (compute 8.6) without Docker GPU passthrough. The sm_120 normalization fix in `runtime.py:1998-2005` correctly equates `sm_120` to `120`. However, physical kernel compilation and memory execution on real Blackwell hardware cannot be certified on this machine. Never claim PASS without hardware execution.

---

## 5. Summary Table

| Finding ID | Classification | Severity | Component | Summary |
|------------|----------------|----------|-----------|---------|
| A-1 | RESOLVED | P0 | `runtime.py` | `_gpu_cache_key` fallback to `XDG_CONFIG_HOME` enables signed capability in container |
| A-3 | RESOLVED | P0 | `pc.sh` | Readiness polls `/api/jobs` (<1ms) instead of slow `/api/models` |
| DEF-01 | CONFIRMED_BUG | P2 | `pc.sh:402` | `grep -iq nvidia` matches arbitrary host strings in `docker info` |
| DEF-02 | CONFIRMED_BUG | P2 | `pc.sh:283` | Preflight disk threshold (10 GB) contradicts 50 GB requirement; risks ENOSPC |
| DEF-03 | CONFIRMED_BUG | P1 | `Dockerfile:59` / `pc.sh:580` | Prebuilt image `/artifacts/mamba` owned by 1000:1000; fails for UID != 1000 |
| DEF-04 | CONFIRMED_BUG | P3 | `pipeline.py:1403` | `probe_gpu_capability` prints capability key as `None` (cosmetic) |
| RISK-01 | UNVERIFIED_RISK | P2 | `pc.sh:858` | Monitor fallback to `jobs[0]` risks latching onto stale failed jobs |
| RISK-02 | UNVERIFIED_RISK | P1 | Runtime / CUDA | Blackwell sm_120 real kernel execution remains PENDING hardware test |

---

## 6. Overall Verdict

**Verdict:** **SOUND_WITH_RESERVATIONS**

**Rationale:**
The critical show-stoppers that previously prevented the one-command launcher from ever completing (P0 capability key signing failure A-1 and P0 readiness polling timeout A-3) are confirmed fixed. The launcher behaves robustly across preflight, configuration lifecycle, idempotent runs, dedup/resume, readiness timeout capping, signal traps, and Vietnamese failure summaries.
Reservations remain regarding:
1. Operator account UID != 1000 against prebuilt image permissions (DEF-03).
2. Disk exhaustion risk when free disk is 10–25 GB (DEF-02).
3. Docker runtime false-positive detection on hosts without toolkit (DEF-01).
4. Physical CUDA kernel execution on Blackwell sm_120 hardware is PENDING.

---

Can this implementation fail during the real first run on an RTX 5070 Ti PC? YES — if the operator account is not UID 1000 (prebuilt `/artifacts/mamba` permission denied, DEF-03), if free disk space is between 10 GB and 25 GB (preflight passes but downloads exhaust space, DEF-02), or if the unverified sm_120 PyTorch wheel encounters hardware kernel initialization errors (RISK-02).
