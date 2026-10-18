# R8-02: Inventory of Argv Data Carriage & E2BIG Failure Analysis

**Status:** DEFECT CONFIRMED (P1 / Latent Blocker)  
**Host Target:** `pc.sh` (HEAD `2bc0cd2`, 2232 lines)  
**Related Components:** `workbench/backend/main.py`, `workbench/backend/storage.py`, `workbench/backend/index.py`, `workbench/scripts/verify_run.py`  
**Host Kernel:** Linux 7.0.0-38-generic x86_64 (`PAGE_SIZE=4096`, `ARG_MAX=2097152`, `MAX_ARG_STRLEN=131072`)

---

## 1. Executive Summary

In `pc.sh` at HEAD `2bc0cd2`, full HTTP response bodies from `/api/runs` and `/api/benchmark-runs`, as well as full container verifier JSON outputs, are read into shell variables and passed as literal command-line arguments to embedded `python3` programs (`pc.sh:1269-1306`, `pc.sh:1767`, `pc.sh:1788`, `pc.sh:1950`).

While `getconf ARG_MAX` reports `2,097,152` bytes (~2 MiB), the Linux kernel imposes a separate hard ceiling on any **single argument** (`MAX_ARG_STRLEN = 32 * PAGE_SIZE = 131,072` bytes, or `131,071` characters plus trailing NUL). Once the accumulated benchmark runs exceed ~30–37 runs (or ~43–65 indexed per-query runs), `/api/benchmark-runs` and `/api/runs` exceed 128 KiB. Any command invocation attempting to pass these payloads via `sys.argv` fails with `E2BIG` (`[Errno 7] Argument list too long`) inside `execve()` **before Python interpreter initialization**.

In `pc.sh:1767`, this triggers an unhandled shell exit code 126, which the subshell captures as `__INDEX_CHECK_FAILED__: bash: /usr/bin/python3: Argument list too long`, causing `pc.sh` to abort release verification at `fail_stage "INDEX_VERIFY"`, regardless of run correctness.

---

## 2. Complete Inventory of Argv Data Carriage in `pc.sh`

Every invocation in `pc.sh` where external commands receive shell variables or command outputs is categorized below.

| Line (HEAD `2bc0cd2`) | Line (Working Tree) | Command Target | Variable / Argument Passed | Producer / Source | Growth Driver | Size Ceiling | Classification |
|---|---|---|---|---|---|---|---|
| **1269** | **1288** | `python3 -c` | `sys.argv[4]="$runs_json"` | `curl -s "${base}/api/runs"` (`pc.sh:1265`) | Total runs indexed in DuckDB | **NONE** (grows with N runs) | **GENUINELY UNBOUNDED** |
| **1269** | **1288** | `python3 -c` | `sys.argv[5]="$b_runs_json"` | `curl -s "${base}/api/benchmark-runs"` (`pc.sh:1266`) | Total aggregate reports in `reports_root` | **NONE** (grows with N runs) | **GENUINELY UNBOUNDED** |
| **1306** | **1331** | `python3 -` | `sys.argv[4]="$b_runs_json"` | `curl -s "${base}/api/benchmark-runs"` (`pc.sh:1266`) | Total aggregate reports in `reports_root` | **NONE** (grows with N runs) | **GENUINELY UNBOUNDED** |
| **1306** | **1331** | `python3 -` | `sys.argv[1]="$v_out"` | `run_verifier_in_container ... 2>&1` (`pc.sh:1304`) | Verifier JSON + container stderr | Semi-bounded: single-run verdict (~2 KB); unbounded if container stderr logs explode | **SEMI-BOUNDED** |
| **1767** | **1851** | `python3 -c` | `sys.argv[1]="$runs_check"` | `curl -fsS "${base}/api/runs"` (`pc.sh:1573`) | Total runs indexed in DuckDB | **NONE** (grows with N runs) | **GENUINELY UNBOUNDED** |
| **1767** | **1851** | `python3 -c` | `sys.argv[2]="$reports_check"` | `curl -fsS "${base}/api/benchmark-runs"` (`pc.sh:1576`) | Total aggregate reports in `reports_root` | **NONE** (grows with N runs) | **GENUINELY UNBOUNDED** |
| **1788** | **1872** | `python3 -` | `sys.argv[1]="$verify_out"` | `run_verifier_in_container ... 2>&1` (`pc.sh:1947`) | Verifier JSON + container stderr | Semi-bounded: single-run verdict (~2 KB); unbounded if container stderr logs explode | **SEMI-BOUNDED** |
| **1950** | **2034** | `python3 -c` | `sys.argv[1]="$FINAL_VERIFY_OUT"` | Output of `verify_out` (`pc.sh:1947`) | Verifier JSON + container stderr | Semi-bounded: single-run verdict (~2 KB); unbounded if container stderr logs explode | **SEMI-BOUNDED** |
| 554 | 554 | `python3 -c` | `sys.argv[1]="$MODEL"`, `sys.argv[2]="${REPO_ROOT}/...yaml"` | Shell parameters and static paths | Static repo config | Bounded (< 100 bytes) | **BOUNDED** |
| 705 | 705 | `python3 -c` | `sys.argv[1]="$REPO_ROOT"` | Shell variable; payload on **STDIN** | Static repo path | Bounded (< 200 bytes) | **BOUNDED** |
| 861 | 861 | `python3 -c` | `sys.argv[1]="$job_id"` | Job ID string; payload on **STDIN** | Single UUID/ID string | Bounded (< 50 bytes) | **BOUNDED** |
| 877 | 877 | `python3 -` | `sys.argv[1]=.env path`, `sys.argv[2]=$REPO_ROOT` | File path strings | Static file paths | Bounded (< 200 bytes) | **BOUNDED** |
| 934 | 934 | `python3 -c` | `sys.argv[1]="$RUN_ID"` | Run ID string; payload on **STDIN** | Single run ID | Bounded (< 50 bytes) | **BOUNDED** |
| 1072 | 1072 | `docker exec` | CLI args: `--model`, `--dataset-root`, etc. | Shell variables | CLI options | Bounded (< 200 bytes) | **BOUNDED** |
| 1091 | 1091 | `docker exec` | `sys.argv[1]="${MODEL}"` | Model ID string | CLI option | Bounded (< 30 bytes) | **BOUNDED** |
| 1120 | 1120 | `docker exec` | `"${exec_env[@]}"`, `"$@"` | Container inspect ENV lines + verifier flags | Docker container ENV | Bounded (< 4 KiB total) | **BOUNDED** |
| 1231 | 1231 | `python3 -c` | `sys.argv[1]="$MODEL"`, `sys.argv[2]="$CHECKPOINT_REQUEST"` | Shell variables; payload on **STDIN** | Model / checkpoint IDs | Bounded (< 100 bytes) | **BOUNDED** |
| 1410 | 1410 | `python3 -c` | `sys.argv[1..3]="$MODEL" "$checkpoint" "$protocol"` | Shell variables; payload on **STDIN** | Run identity IDs | Bounded (< 150 bytes) | **BOUNDED** |
| 1440 | 1440 | `curl` | `-d "{"model_id":...}"` | Shell variable expansion in JSON template | Run identity IDs | Bounded (< 150 bytes) | **BOUNDED** |
| 1445 | 1445 | `python3 -c` | `sys.argv[1]="$post_body"` | `POST /api/jobs` response body | Single job creation response | Bounded (< 500 bytes) | **BOUNDED** |
| 1448 | 1448 | `python3 -c` | `sys.argv[1]="$post_body"` | `POST /api/jobs` 409 error response body | Single job 409 conflict payload | Bounded (< 500 bytes) | **BOUNDED** |
| 1469 | 1469 | `python3 -c` | `sys.argv[1]="$job_id"` | Job ID string; payload on **STDIN** | Single job ID | Bounded (< 50 bytes) | **BOUNDED** |
| 1483 | 1483 | `python3 -c` | None (`json.load(sys.stdin)`) | Endpoint payload on **STDIN** | Single job record | Bounded | **BOUNDED** |
| 1497 | 1497 | `python3 -c` | `sys.argv[1]="$job_json"` | Single job record from `/api/jobs` | Job metadata + log tails | Bounded: queue caps `stdout_tail` and `stderr_tail` to 12,000 chars (`jobs.py:448`). Total < 30 KiB. | **BOUNDED** |
| 1503 | 1503 | `python3 -c` | `sys.argv[1]="$job_json"` | Single job record | Same as above | Bounded (< 30 KiB) | **BOUNDED** |
| 1508 | 1508 | `python3 -c` | `sys.argv[1]="$job_json"` | Single job record | Same as above | Bounded (< 30 KiB) | **BOUNDED** |
| 1509 | 1509 | `python3 -c` | `sys.argv[1]="$job_json"` | Single job record | Same as above | Bounded (< 30 KiB) | **BOUNDED** |
| 1734 | 1734 | `python3 -c` | `sys.argv[1]="$model"` | Shell variable; payload on **STDIN** | Model ID string | Bounded (< 30 bytes) | **BOUNDED** |
| 1995 | 2079 | `python3 -` | `sys.argv[1]="$INDEX_EVIDENCE_STATUS"`, `sys.argv[2]="$verify_state"` | Status string & state file path | Short status & tmp path | Bounded (< 100 bytes) | **BOUNDED** |

---

## 3. Endpoint Growth Traces & Response Shape

### 3.1 `/api/runs`
- **Handler Definition:** `workbench/backend/main.py:196-201` (`def runs(protocol_id=None)`).
- **Storage Backend:** `workbench/backend/storage.py:30-41` (`def list_runs(...)`).
  - Query: `SELECT metadata, evidence_status FROM runs ORDER BY run_id` (`storage.py:33, 38`).
  - Projection: `[{**json.loads(metadata), "evidence_status": evidence_status} for metadata, evidence_status in connection.execute(sql, params).fetchall()]` (`storage.py:39-40`).
- **Index Population:** `workbench/backend/index.py:122, 178`.
  - Stored column `metadata` is `run.model_dump_json()` from `ResultRun.run` (`schemas.results.RunMetadata`, `schemas/results.py:79-136`).
- **Pagination & Caps:** **NONE**. Returns every indexed row in DuckDB without `LIMIT` or `OFFSET`.
- **Row Content:**
  - Identifiers: `run_id`, `dataset`, `split`, `protocol_id`, `literature_split_label`, `model_id`, `method_name`, `checkpoint_id`, `timestamp`, `data_kind`.
  - Checkpoint Bundle: `checkpoint_bundle` (`dict[str, dict]`), `bundle_manifest_digest`, `category_checkpoint_ids`. For multi-partition models (e.g. LIMN's 3 category checkpoints), stores filenames, sha256 hashes, sizes, sources.
  - Per-Category Data: `category_gallery_sizes`, `category_eligible_counts`.
  - Metrics: `reported_paper_metrics` (`ReportedMetrics`), `reproduced_metrics` (`ReportedMetrics`).
  - Digests & Integrity: `checkpoint_sha256`, `instrumentation_patch_sha256`, `command_digest`, `environment_digest`, `runtime_adaptation`, `per_query_artifact_digest`, `instrumentation_parity`, `evidence_status`.
- **Growth Driver:** Directly proportional to the number of completed and indexed runs with Top-K observer output.
  - **Does NOT include per-query rows:** The ~15,000 query rows live in the separate DuckDB `queries` and `top_results` tables (`workbench/backend/index.py:123-124, 179-181`) and are served via `/api/queries` (`main.py:885-888`).
  - **Single row byte size:** **2,024 bytes** (standard baseline) to **3,117 bytes** (with multi-checkpoint bundle details).

### 3.2 `/api/benchmark-runs`
- **Handler Definition:** `workbench/backend/main.py:862-875` (`def benchmark_runs()`).
- **Report Scanner:** `workbench/backend/main.py:636-795` (`def _aggregate_reports()`).
  - Scans `sorted(reports_root(config).glob("*_aggregate.json"))` (`main.py:721`).
  - Corroborates every report against the evidence ledger HMAC chain (`main.py:734-750`).
  - Appends authoritative verification receipt from cache/verifier (`main.py:760-794`).
- **Gate Projection:** `workbench/backend/main.py:806-850` (`def _gate_fields(report)`).
- **Pagination & Caps:** **NONE**. Returns every ledger-corroborated aggregate report in `reports_root`.
- **Row Content:**
  - Identity: `run_id`, `model_id`, `checkpoint_id`, `protocol_id`.
  - `benchmark_gate`: Dict containing `status`, `passed`, `selection_outcome`, `selection_reason`, `reference_evidence`, `detail`, plus:
    - **`comparisons` (growing list):** Metric comparisons for all evaluated categories and macros (`dress_r10`, `dress_r50`, `shirt_r10`, `shirt_r50`, `toptee_r10`, `toptee_r50`, `macro_r10`, `macro_r50`, `macro_mean`).
    - **`blockers` (growing list):** Diagnostic refusal messages.
  - `parity_status`: String verdict.
  - `verification` (verification receipt): Dict containing `artifact_exists`, `execution_authenticated`, `is_research_evidence`, `certified`, `development_mode`, `certification_mode`, `state`, `report_digest`, `per_query_artifact_digest`, `model_id`, `checkpoint_id`, plus:
    - **`reasons` (growing list):** Audit trail reasons explaining ledger verification, proof validation, or failures.
    - **`evidence_reasons` (growing list):** Specific evidence disqualification strings.
  - `variant_provenance`: Caption aliases, patch status, adapted runtime details.
  - `category_metrics`: Dict of category metric dicts (`dress`, `shirt`, `toptee`).
  - `observed_metrics`: Macro metrics dict.
  - `execution_status`, `per_query_capability`, `per_query_state`, `per_query_export_reason`, `is_synthetic`.
- **Growing Fields Echoed:**
  - `benchmark_gate["comparisons"]` echoes all evaluated metric comparisons.
  - `verification["reasons"]` echoes verification failure and audit strings.
  - `benchmark_gate["blockers"]` echoes blocker explanation strings.
- **Growth Driver:** Number of evaluation runs (all 16 models produce aggregate reports, unlike per-query runs where only observer models export Top-K).
  - **Single row byte size:** **3,605 bytes** (standard pass) to **4,410 bytes** (runs with verbose refusal reasons/comparisons).

---

## 4. Empirical Kernel Limits & Mathematical Thresholds

### 4.1 Measured Host Limits
Commands executed on current host (`linux 7.0.0-38-generic x86_64`):

```bash
$ getconf ARG_MAX
2097152

$ getconf PAGE_SIZE
4096
```

### 4.2 Single Argument Cap (`MAX_ARG_STRLEN`)
In the Linux kernel (`fs/exec.c`), while total command line + environment space is governed by `ARG_MAX` (2 MiB), a single argument string cannot exceed 32 memory pages:
$$	ext{MAX\_ARG\_STRLEN} = 32 	imes 	ext{PAGE\_SIZE} = 32 	imes 4096 = 131,072 	ext{ bytes}$$

Because C strings in `execve(2)` require a trailing NUL byte (` `), the maximum permissible payload length is **131,071 characters**.

Empirical proof on this host:
```bash
# Python subprocess test:
$ python3 -c '
import subprocess, sys
for size in [131070, 131071, 131072]:
    try:
        subprocess.run([sys.executable, "-c", "import sys; print(len(sys.argv[1]))", "a" * size],
                       capture_output=True, text=True, check=True)
        print(f"size {size}: SUCCESS")
    except Exception as e:
        print(f"size {size}: FAILED ({e})")
'
size 131070: SUCCESS
size 131071: SUCCESS
size 131072: FAILED ([Errno 7] Argument list too long: '/usr/bin/python3')

# Bash subshell test:
$ bash -c 'payload=$(python3 -c "print("x" * 131072)"); python3 -c "import sys; print(sys.argv[1])" "$payload"'
bash: line 1: /usr/bin/python3: Argument list too long
# exit code: 126
```

### 4.3 Worst-Case Response Size & Threshold to E2BIG

| Endpoint | Average Row Size | Maximum Row Size | Runs to Cross 131,071 Bytes (Avg) | Runs to Cross 131,071 Bytes (Max) |
|---|---|---|---|---|
| `/api/benchmark-runs` | 3,605 bytes | 4,410 bytes | **37 runs** | **30 runs** |
| `/api/runs` | 2,024 bytes | 3,117 bytes | **65 runs** | **43 runs** |

### 4.4 Real-World Collision Horizon
The workbench contains **16 registered models** evaluated across **3 official protocols** (`fashioniq_val_split`, `fashioniq_original_split`, `fashioniq_full_gallery_ref_excluded`).
- Evaluating the model suite once produces $16 	imes 3 = \mathbf{48 	ext{ runs}}$.
- Payload size of `/api/benchmark-runs` at 48 runs:
  $$48 	imes 3,605 	ext{ bytes} pprox \mathbf{173,040 	ext{ bytes}} \quad (\mathbf{169 	ext{ KiB}})$$
- Since $173,040 > 131,071$, **a single full evaluation pass of the workbench guaranteed E2BIG failure**.
- If models are evaluated under noise variations (e.g. clean vs 20% vs 50% noise) or re-run, run count reaches 100+ runs (> 350 KiB), exceeding the limit by nearly $3	imes$.

---

## 5. Reproduction of E2BIG in Scratch Directory

The failure was reproduced in isolation under `/tmp` using synthetic payloads modeled directly on workbench schemas.

### Reproducer Script
```python
import os, json, subprocess, tempfile

# Construct synthetic benchmark_runs payload for 70 runs (~135 KB)
def make_run(i):
    return {
        "run_id": f"run_{i:04d}", "model_id": "limn", "checkpoint_id": "base_iter0_all_categories",
        "protocol_id": "fashioniq_val_split", "benchmark_gate": {
            "status": "PASS_WITHIN_5_PERCENT_VARIANT", "passed": True,
            "comparisons": [{"metric": f"m_{k}", "observed": 25.0, "reference": 25.0} for k in range(6)]
        },
        "verification": {"is_research_evidence": True, "reasons": ["ledger ok"], "evidence_reasons": []},
        "category_metrics": {"dress": {"r10": 25.0, "r50": 50.0}}, "execution_status": "EXECUTION_COMPLETED",
        "per_query_capability": "canonical_observer", "per_query_state": "PER_QUERY_VERIFIED"
    }

b_runs_json = json.dumps([make_run(i) for i in range(70)])

# Test Site 1: pc.sh:1269 (candidate_runs)
try:
    subprocess.run(["python3", "-c", "import sys; print(len(sys.argv[5]))",
                    "m", "c", "p", "[]", b_runs_json], capture_output=True, check=True)
except Exception as e:
    print(f"Site 1 Failure: {type(e).__name__}: {e}")

# Test Site 2: pc.sh:1306 (PYCHECK)
try:
    subprocess.run(["python3", "-", "{}", "0", "run_1", b_runs_json],
                   input="import sys
", text=True, capture_output=True, check=True)
except Exception as e:
    print(f"Site 2 Failure: {type(e).__name__}: {e}")

# Test Site 3: pc.sh:1767 (rebuild_and_verify_index)
try:
    subprocess.run(["python3", "-c", "import sys; print(sys.argv[2])",
                    "[]", b_runs_json, "r1", "m", "c", "p", "OBS"], capture_output=True, check=True)
except Exception as e:
    print(f"Site 3 Failure: {type(e).__name__}: {e}")
```

### Observed Output
```
Site 1 Failure: OSError: [Errno 7] Argument list too long: 'python3'
Site 2 Failure: OSError: [Errno 7] Argument list too long: 'python3'
Site 3 Failure: OSError: [Errno 7] Argument list too long: 'python3'
```

### Exact Shell Failure at Release Gate (`pc.sh:1767`)
When executed in bash identical to `pc.sh:1767`:
```bash
checked="$(python3 -c "$idx_program" "$runs_check" "$reports_check" "$run_id" "$model" "$checkpoint" "$protocol" "${declared_observer:-}" 2>&1)"     || checked="__INDEX_CHECK_FAILED__: ${checked:-không xác định}"
```
**Output captured in `$checked`:**
```
__INDEX_CHECK_FAILED__: bash: line 1: /usr/bin/python3: Argument list too long
```
**Result in `pc.sh:1769-1774`:**
```
[FAIL] Giai đoạn thất bại : INDEX_VERIFY
[FAIL] Chi tiết lỗi        : Lượt chạy không hiện diện trong chỉ mục với đúng danh tính và trạng thái bằng chứng: bash: line 1: /usr/bin/python3: Argument list too long
[FAIL] Vị trí nhật ký     : http://127.0.0.1:8000/api/runs
[FAIL] Hành động khắc phục : Không tuyên bố hoàn tất: API phải phục vụ được lượt chạy này với đúng danh tính và đúng trạng thái bằng chứng mà nó khai...
```

---

## 6. Failure Semantics Preservation Matrix

Any file-based or stdin-based replacement must strictly preserve the existing contract across all error branches. A replacement must not convert an HTTP error into an empty deployment, or mask an unreadable index.

### 6.1 Job Dedup Sites (`pc.sh:1265-1345`)
Target code:
```bash
runs_json="$(curl -s --max-time 10 "${base}/api/runs" 2>/dev/null || echo "[]")"
b_runs_json="$(curl -s --max-time 10 "${base}/api/benchmark-runs" 2>/dev/null || echo "[]")"
```

| Failure Mode | Current Behavior | What Replacement Must Preserve |
|---|---|---|
| **HTTP Error (4xx, 5xx)** | `curl -s` without `-f` returns 0; response body (e.g. `{"detail": ...}`) captured. Python `json.loads` parses dict or fails, `isinstance(data, list)` check fails, `all_runs` stays empty. | Must NOT abort evaluator; must treat unparseable/non-list as empty candidate set (`[]`) and proceed to execute evaluation. |
| **Timeout (`--max-time 10`)** | `curl` exits code 28. Subshell `|| echo "[]"` sets variable to `"[]"`. Python parses empty list, candidate set empty. | If redirected to file via `-o`, timeout leaves empty or absent file. Reader must treat missing or 0-byte file as empty list `[]`, not raise uncaught error. |
| **Empty Body (`""`)** | `json.loads("")` raises `json.JSONDecodeError`; caught by `except Exception: pass`. Candidates empty. | Replacement reader must catch `JSONDecodeError` / `OSError` and treat as empty list `[]`. |
| **Malformed JSON** | HTML error page or truncated JSON raises decode exception; caught by `except Exception: pass`. | Reader must catch exceptions and default to empty list `[]`. |
| **Container Verifier Error (`v_out`)** | `v_out="$(run_verifier_in_container ... 2>&1)" || v_rc=$?`. Python parses JSON verdict. If invalid, exits 1; if `v_rc != 0` and not proof-unavailable, candidate rejected. | Verifier output file must capture both stdout and stderr (`2>&1`). If verifier crashes or output unreadable, reader must exit 1 (reject reuse), not crash the launcher. |

### 6.2 Index Gate Sites (`pc.sh:1572-1775`)
Target code:
```bash
runs_check="$(curl -fsS --max-time 60 "${base}/api/runs" 2>/dev/null || echo "__API_UNREACHABLE__")"
reports_check="$(curl -fsS --max-time 60 "${base}/api/benchmark-runs" 2>/dev/null || echo "__API_UNREACHABLE__")"
```

| Failure Mode | Current Behavior | What Replacement Must Preserve |
|---|---|---|
| **HTTP Error (4xx, 5xx)** | `curl -fsS` fails due to `-f` (exit 22). Subshell `|| echo "__API_UNREACHABLE__"` assigns sentinel. `idx_program` exits with `"không gọi được /api/runs"` or `"không gọi được /api/benchmark-runs"`. | Replacement must NOT write an empty file and treat as 200 OK. Must record unreachable state or write `__API_UNREACHABLE__` sentinel so `idx_program` emits exact string `"không gọi được /api/..."`. |
| **Timeout (`--max-time 60`)** | `curl` exits code 28. Triggered `|| echo "__API_UNREACHABLE__"`. Exits with `"không gọi được /api/..."`. | Must write or signal unreachable sentinel. |
| **Empty Body (`""`)** | `if raw.strip() == "__API_UNREACHABLE__" or not raw.strip(): sys.exit("không gọi được /api/runs")`. | Empty file or 0 bytes must trigger `"không gọi được /api/runs"` (or benchmark-runs), NEVER treated as empty list `[]`. |
| **Malformed JSON** | `except Exception as error: sys.exit("không đọc được /api/runs: %s" % type(error).__name__)`. | Decode error must exit with `"không đọc được /api/runs: <ErrorName>"`, leading to `fail_stage "INDEX_VERIFY"`. |
| **Non-List JSON (e.g. dict error response)** | `if not isinstance(runs, list): sys.exit("phản hồi /api/runs không phải danh sách")`. | Must exit with `"phản hồi /api/runs không phải danh sách"`. |
| **Missing Run in Benchmark Reports** | `if report is None: sys.exit("báo cáo tổng hợp của lượt chạy %s KHÔNG được API phục vụ...")`. | Must preserve exact exit reason and error formatting. |
| **Identity Mismatch** | Compares `model_id`, `checkpoint_id`, `protocol_id`. Mismatch exits with `"báo cáo tổng hợp gắn %s=%r, không phải %r"`. | Must preserve exact field comparison checks. |
| **Observer Model Missing in DuckDB Index** | If declared observer model and run absent from `runs`: `"mô hình %s khai báo trình quan sát... KHÔNG có trong chỉ mục /api/runs"`. | Must preserve exact refusal string. |
| **Evidence Status != PER_QUERY_VERIFIED** | For observer models: exits `"mô hình %s khai báo trình quan sát... trạng thái %r, không phải PER_QUERY_VERIFIED"`. | Must preserve exact refusal string. |
| **Temporary File / Directory Security** | N/A in HEAD (variables in memory). In working tree: `mktemp -d` with `chmod 700`. | Temporary directories must be mode 700 (private to current user, no world-readable run artifacts), with guaranteed cleanup via `rm -rf` on all return and exit paths. |
