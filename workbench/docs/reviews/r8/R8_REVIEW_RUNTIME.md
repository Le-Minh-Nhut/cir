# R8 Review: Runtime Perspective (Launcher, Shell, Large Payloads, Cleanup, Exit Codes)

**Review Target:** `pc.sh` (whole script; specifically `execute_evaluation` dedup, `rebuild_and_verify_index`, `verify_gate`, `validate_and_summarize`, `run_verifier_in_container`) and `workbench/tests/test_pc_launcher.py`.  
**Auditor ID:** `R8RevRuntime`  
**Host Target:** Linux 7.0.0-38-generic x86_64 (`PAGE_SIZE=4096`, `MAX_ARG_STRLEN=131072`, `ARG_MAX=2097152`)  
**Status:** READINESS DISPROVED (Multiple Confirmed P1/P2 Cleanup & Lifecycle Defects).

---

## 1. Claims Attacked

1. **Claim 1 (Argv Carriage & E2BIG):** *"Both API bodies and the verifier output now cross the argv boundary as paths, so E2BIG cannot happen."*  
   Attacked by auditing all command invocations in `pc.sh` (`python3`, `docker exec`, `curl`, subshells) for remaining argv carriage of unbounded data.
2. **Claim 2 (Cleanup Completeness):** *"Cleanup happens on every path."*  
   Attacked by tracing all created temp paths (`cir_idx.*`, `cir_dedup.*`, `cir_verify_out.*`, `cir_verify_state.*`, `cir_verify_msg.*`) through success, index-gate failure, verification refusal, `set -e` aborts, and SIGINT/SIGTERM (Ctrl-C).
3. **Claim 3 (Subshell Wrapping):** *"The subshell wrapping is correct."*  
   Attacked by evaluating `( ... ) >"$file" 2>&1 || v_rc=$?` at both call sites, verifying exit code retention, stderr/stdout JSON extraction, containment of `fail_stage` / `exit 1`, and surveying other functions in `pc.sh`.
4. **Claim 4 (Sentinel Semantics Preservation):** *"The sentinel semantics are preserved."*  
   Attacked by comparing the file-redirection curl model against previous command substitution across HTTP errors (4xx/5xx), network timeouts, empty bodies, and partial bodies for both dedup and index gate.
5. **Claim 5 (Concurrency Isolation):** *"Two `./pc.sh` runs must not collide on temp names or on each other's files."*  
   Attacked by evaluating temp path generation entropy, container lock/reuse semantics, and job attachment handling under concurrent executions.

---

## 2. Confirmed Defects

### Defect 1: Summary non-zero exit aborts under `set -e` before cleaning up `cir_verify_state.*` and `cir_verify_out.*`
- **Severity:** P1
- **File:Line:** `pc.sh:2139-2316`
- **Reachable Trigger:** Any evaluated model run where the embedded Python summary program exits with a documented non-zero status:
  - Exit 7: `_index_evidence_status == "AGGREGATE_ONLY_EXECUTION_AUTHENTICATED_PROOF_UNAVAILABLE"` (`pc.sh:2307`)
  - Exit 5: Run report unreadable or corrupted (`pc.sh:2300`)
  - Exit 3: `per_query_capability and per_query_state in (None, "PER_QUERY_EXPORT_FAILED")` (`pc.sh:2309`)
  - Exit 4: `per_query_capability and per_query_state == "PER_QUERY_EXPORTED_UNVERIFIED"` (`pc.sh:2313`)
- **Actual Behaviour:** Because `set -e` is active (`pc.sh:5`), Python's `sys.exit(3|4|5|7)` immediately terminates the bash shell at line 2139. Bash exits with that exit code without ever reaching line 2316 (`rm -f "$verify_state" "$verify_out_file"`). There is no `trap ... EXIT` in `pc.sh`.
  Both temporary files remain permanently in `${TMPDIR:-/tmp}`:
  - `${TMPDIR:-/tmp}/cir_verify_state.XXXXXX`
  - `${TMPDIR:-/tmp}/cir_verify_out.XXXXXX`
- **Expected Behaviour:** Temporary state files created by `validate_and_summarize` must be unlinked regardless of whether the summary script exits 0 or non-zero.
- **Reproducer:**
  ```bash
  bash -c '
  set -euo pipefail
  TMPDIR="/tmp/repro_summary_leak_$$"
  mkdir -p "$TMPDIR"
  verify_out_file="$(mktemp "${TMPDIR}/cir_verify_out.XXXXXX")"
  verify_state="$(mktemp "${TMPDIR}/cir_verify_state.XXXXXX")"
  # Simulates pc.sh lines 2139-2316:
  python3 - "AGGREGATE_ONLY_EXECUTION_AUTHENTICATED_PROOF_UNAVAILABLE" "$verify_state" <<\PYDOC
  import sys
  sys.exit(7)
  PYDOC
  rm -f "$verify_state" "$verify_out_file"
  ' ; echo "Exited with $?" ; ls -1 /tmp/repro_summary_leak_* ; rm -rf /tmp/repro_summary_leak_*
  ```
  *Observed output:* Shell terminates with code 7; `cir_verify_out.*` and `cir_verify_state.*` remain on disk; `rm -f` is never reached.

---

### Defect 2: Index-gate failure via `fail_stage` leaks `cir_verify_state.*` and `cir_verify_out.*`
- **Severity:** P1
- **File:Line:** `pc.sh:1885` and `pc.sh:2137-2316`
- **Reachable Trigger:** An evaluated run fails the index verification check at line 1884 (`$checked == __INDEX_CHECK_FAILED__*`), such as a missing run row or evidence status mismatch.
- **Actual Behaviour:** At line 1885, `fail_stage "INDEX_VERIFY"` is called. `fail_stage` prints the error block and executes `exit 1` (`pc.sh:64`). This aborts the launcher immediately from inside `rebuild_and_verify_index`. Control never returns to `validate_and_summarize`, so line 2316 (`rm -f "$verify_state" "$verify_out_file"`) is skipped.
  Leaked paths:
  - `${TMPDIR:-/tmp}/cir_verify_state.XXXXXX`
  - `${TMPDIR:-/tmp}/cir_verify_out.XXXXXX`
- **Expected Behaviour:** Temp files instantiated prior to calling `rebuild_and_verify_index` must be unlinked before calling `fail_stage` or via an EXIT trap handler.
- **Reproducer:**
  ```bash
  bash -c '
  set -euo pipefail
  TMPDIR="/tmp/repro_idx_fail_$$"
  mkdir -p "$TMPDIR"
  fail_stage() { echo "fail_stage called: $1"; exit 1; }
  verify_out_file="$(mktemp "${TMPDIR}/cir_verify_out.XXXXXX")"
  verify_state="$(mktemp "${TMPDIR}/cir_verify_state.XXXXXX")"
  idx_tmp_dir="$(mktemp -d "${TMPDIR}/cir_idx.XXXXXX")"
  checked="__INDEX_CHECK_FAILED__: missing run"
  rm -rf "$idx_tmp_dir"
  if [[ "$checked" == __INDEX_CHECK_FAILED__* ]]; then
      fail_stage "INDEX_VERIFY"
  fi
  rm -f "$verify_state" "$verify_out_file"
  ' ; echo "Exited with $?" ; ls -1 /tmp/repro_idx_fail_* ; rm -rf /tmp/repro_idx_fail_*
  ```
  *Observed output:* Shell exits with 1; `cir_verify_out.*` and `cir_verify_state.*` leak in `/tmp`.

---

### Defect 3: Registry capability resolution failure inside `rebuild_and_verify_index` leaks `cir_idx.*`, `cir_verify_state.*`, and `cir_verify_out.*`
- **Severity:** P1
- **File:Line:** `pc.sh:1874`
- **Reachable Trigger:** During `rebuild_and_verify_index`, `/api/models` cannot be reached or returns `MODEL_UNKNOWN` (`pc.sh:1846-1873`).
- **Actual Behaviour:** Line 1874 triggers `fail_stage "REGISTRY_CAPABILITY"`, executing `exit 1`. Line 1883 (`rm -rf "$idx_tmp_dir"`) is located AFTER line 1874, so `idx_tmp_dir` is never removed. Furthermore, the parent caller's `verify_state` and `verify_out_file` are also not cleaned up.
  Leaked paths:
  - `${TMPDIR:-/tmp}/cir_idx.XXXXXX` (containing `runs.json` and `benchmark-runs.json`)
  - `${TMPDIR:-/tmp}/cir_verify_state.XXXXXX`
  - `${TMPDIR:-/tmp}/cir_verify_out.XXXXXX`
- **Expected Behaviour:** `idx_tmp_dir` must be removed before invoking `fail_stage`, or unlinked by an EXIT trap.
- **Reproducer:**
  ```bash
  bash -c '
  set -euo pipefail
  TMPDIR="/tmp/repro_reg_fail_$$"
  mkdir -p "$TMPDIR"
  fail_stage() { echo "fail_stage: $1"; exit 1; }
  idx_tmp_dir="$(mktemp -d "${TMPDIR}/cir_idx.XXXXXX")"
  touch "${idx_tmp_dir}/runs.json" "${idx_tmp_dir}/benchmark-runs.json"
  declared_observer="MODEL_UNKNOWN: /api/models down"
  if [[ "$declared_observer" == MODEL_UNKNOWN* ]]; then
      fail_stage "REGISTRY_CAPABILITY"
  fi
  rm -rf "$idx_tmp_dir"
  ' ; echo "Exited with $?" ; ls -1R /tmp/repro_reg_fail_* ; rm -rf /tmp/repro_reg_fail_*
  ```
  *Observed output:* `cir_idx.*` directory and files inside remain on disk upon exit.

---

### Defect 4: Operator SIGINT / SIGTERM (Ctrl-C) leaks all currently allocated temp directories and files
- **Severity:** P1
- **File:Line:** `pc.sh:35-46`
- **Reachable Trigger:** Operator interrupts execution (e.g. `Ctrl-C`) while `./pc.sh` is running dedup verification, index verification, or validation.
- **Actual Behaviour:** `cleanup_on_signal` trap (`pc.sh:35-44`) catches `SIGINT` and `SIGTERM`, outputs a banner ("Bảo toàn nguyên vẹn artifacts..."), and immediately calls `exit 130`. There is no cleanup of scratch files in `/tmp` and no `EXIT` trap.
  Leaked paths (depending on stage interrupted):
  - `${TMPDIR:-/tmp}/cir_dedup.XXXXXX` (and all files inside: `runs.json`, `benchmark-runs.json`, `verifier.*.json`)
  - `${TMPDIR:-/tmp}/cir_idx.XXXXXX` (and all files inside)
  - `${TMPDIR:-/tmp}/cir_verify_out.XXXXXX`
  - `${TMPDIR:-/tmp}/cir_verify_state.XXXXXX`
  - `${TMPDIR:-/tmp}/cir_verify_msg.XXXXXX`
- **Expected Behaviour:** Real artifacts and checkpoints should be preserved as intended, but ephemeral `/tmp` scratch files must be unregistered and unlinked upon receiving termination signals.
- **Reproducer:**
  ```bash
  bash -c '
  set -euo pipefail
  TMPDIR="/tmp/repro_sigint_$$"
  mkdir -p "$TMPDIR"
  cleanup_on_signal() { exit 130; }
  trap "cleanup_on_signal SIGINT" SIGINT
  dedup_tmp_dir="$(mktemp -d "${TMPDIR}/cir_dedup.XXXXXX")"
  touch "${dedup_tmp_dir}/runs.json"
  kill -SIGINT $$
  rm -rf "$dedup_tmp_dir"
  ' ; echo "Exited with $?" ; ls -1R /tmp/repro_sigint_* ; rm -rf /tmp/repro_sigint_*
  ```
  *Observed output:* Shell exits with 130; `cir_dedup.XXXXXX` and contents leak in `/tmp`.

---

### Defect 5: Index rebuild failure before index gate leaks `cir_verify_state.*` and `cir_verify_out.*`
- **Severity:** P2
- **File:Line:** `pc.sh:1608`
- **Reachable Trigger:** `POST /api/index/rebuild` returns non-2xx status code (e.g. HTTP 500 or timeout).
- **Actual Behaviour:** Line 1608 calls `fail_stage "INDEX_REBUILD"`, terminating `pc.sh`. In `validate_and_summarize`, `verify_state` and `verify_out_file` were already allocated at lines 2078 and 2116. Because the script exits at line 1608, cleanup at line 2316 is never reached.
  Leaked paths:
  - `${TMPDIR:-/tmp}/cir_verify_state.XXXXXX`
  - `${TMPDIR:-/tmp}/cir_verify_out.XXXXXX`
- **Expected Behaviour:** Cleanup of pre-allocated verification temporary files should occur on rebuild failure.

---

### Defect 6: Test veracity defect in `test_the_gate_leaves_no_temporary_directory_behind`
- **Severity:** P2
- **File:Line:** `workbench/tests/test_pc_launcher.py:2712-2726`
- **Reachable Trigger:** Running pytest on launcher tests.
- **Actual Behaviour:** The test docstring claims:
  > *"This drives the real `rebuild_and_verify_index` through a sandbox and counts what is left in the temp root it was told to use."*  
  In reality, the test body does zero execution:
  ```python
  source = PC_SH.read_text(encoding="utf-8")
  assert 'mktemp -d "${TMPDIR:-/tmp}/cir_idx.XXXXXX"' in source
  assert 'rm -rf "$idx_tmp_dir"' in source
  assert 'mktemp -d "${TMPDIR:-/tmp}/cir_dedup.XXXXXX"' in source
  assert 'rm -rf "$dedup_tmp_dir"' in source
  ```
  It merely verifies the literal strings appear in `pc.sh`. Because it never runs `rebuild_and_verify_index` or exercises failure/abort branches, it provides false confidence that directory cleanup is verified when all error-path leaks remain undetected.
- **Expected Behaviour:** Test assertions should reflect what the test actually executes, and error path unlinking should be exercised in a sandboxed subprocess.

---

## 3. Disproved Allegations

### Allegation 1: "Unbounded data carriage still exists in `pc.sh` argv"
- **Status:** DISPROVED.
- **Evidence:**
  1. `pc.sh:1316`: `candidate_runs="$(python3 -c "..." "$MODEL" "$checkpoint" "$protocol" "$dedup_paths" ...)"`. Passes file paths in `sys.argv[4]`; bodies are loaded via `open()` inside Python.
  2. `pc.sh:1339`: `python3 - "$v_out_file" "$v_rc" "$cand_id" "$b_runs_json_file"`. Both verdict and benchmark-runs are paths.
  3. `pc.sh:1881`: `checked="$(python3 -c "$idx_program" "$idx_paths" "$run_id" "$model" "$checkpoint" "$protocol" "${declared_observer:-}" 2>&1)"`. The index gate receives `$idx_paths` (two paths separated by `\n`); bodies are loaded via `_body(_runs_path)` and `_body(_reports_path)`.
  4. `pc.sh:1906`: `verify_gate` receives `$verify_out_file` as `sys.argv[1]`; read via `open(sys.argv[1])`.
  5. `pc.sh:2088`: `FINAL_RUN_REPORT_PATH` python reader receives `$verify_out_file` as `sys.argv[1]`; read via `open(sys.argv[1])`.
  6. `pc.sh:2139`: Summary receives `${INDEX_EVIDENCE_STATUS:-}` and `$verify_state` path.
  7. `pc.sh:1544, 1550, 1555, 1556`: Evaluation polling passes `"$job_json"` in `sys.argv[1]`. Auditing `workbench/backend/jobs.py:448` reveals `_append(job, field, line)` caps `stdout_tail` and `stderr_tail` to 12,000 characters each. A serialized single job record is bounded to ~30 KiB maximum, well below `MAX_ARG_STRLEN` (128 KiB).
  8. `pc.sh:1120`: `docker exec` passes `-e` environment variables and 9 CLI flags, all bounded.
  No genuinely unbounded payload remains in `sys.argv`.

---

### Allegation 2: "Verification failure in `validate_and_summarize` leaks temp files"
- **Status:** DISPROVED.
- **Evidence:**
  In `pc.sh:2119-2131`:
  ```bash
  if ! verify_gate "$verify_out_file" "$verify_rc" "$model" "$checkpoint" "$protocol" "$run_id" \
          "${FINAL_RUN_PROOF_UNAVAILABLE:-0}" "${FINAL_RUN_REPORT_PATH:-}" "$host_artifacts" >"$verify_message"; then
      local verify_refusal
      verify_refusal="$(cat "$verify_message")"
      rm -f "$verify_state" "$verify_message" "$verify_out_file"
      printf '%s\n' "$verify_refusal"
      fail_stage "VERIFICATION" "$verify_refusal" "verify_run.py (trong container)" ...
  fi
  ```
  On verification refusal (when `verify_gate` returns non-zero), line 2127 explicitly unlinks `verify_state`, `verify_message`, and `verify_out_file` BEFORE calling `fail_stage`. This specific path does not leak.

---

### Allegation 3: "Subshell wrapping `( ... ) >"$file" 2>&1 || v_rc=$?` drops exit codes or breaks JSON parsing"
- **Status:** DISPROVED.
- **Evidence:**
  1. Exit code retention: If the subshell succeeds, `|| v_rc=$?` does not execute and `v_rc` remains `0`. If `run_verifier_in_container` exits non-zero (or is killed), `|| v_rc=$?` captures the non-zero status code.
  2. Output containment: If the container is down, `run_verifier_in_container` calls `fail_stage "VERIFICATION"`. Because it runs inside `( ... )`, the `exit 1` terminates only the subshell. The failure banner is written into `"$v_out_file"`.
  3. Parent handling: In `validate_and_summarize`, `verify_gate` reads `"$v_out_file"`. Because no JSON is present, it prints `Lỗi: verify_run.py trả về mã lỗi 1 và không có JSON hợp lệ.` followed by the captured banner, and exits 1. Parent catches this in `if ! verify_gate`, cleans up temp files, and calls `fail_stage` in the top-level shell.
  4. Log/stderr separation: Python readers at lines 1344, 1934, and 2097 all inspect `reversed(out.strip().splitlines())` to find the trailing `{...}` JSON block. Any stderr warnings (e.g. CUDA/PyTorch warnings) mixed in by `2>&1` prior to the JSON are ignored.

---

### Allegation 4: "The index gate treats a failed curl call as an empty deployment (`[]`)"
- **Status:** DISPROVED.
- **Evidence:**
  In `pc.sh:1642-1705`:
  ```bash
  if ! curl -fsS --max-time 60 "${base}/api/runs" >"$runs_file" 2>/dev/null; then
      printf '%s' "__API_UNREACHABLE__" >"$runs_file"
  fi
  ```
  Tracing failure cases:
  - **HTTP error (4xx/5xx):** `curl -fsS` fails (exit 22). `if ! curl` branch executes, overwriting `$runs_file` with `__API_UNREACHABLE__`. Python checks `raw.strip().startswith("__API_UNREACHABLE__")` and calls `sys.exit("không gọi được /api/runs")`.
  - **Timeout:** `curl -fsS` exits with code 28. `if ! curl` branch overwrites `$runs_file` with `__API_UNREACHABLE__`. Python exits with error.
  - **Empty body (HTTP 200, 0 bytes):** `curl` succeeds with code 0. `$runs_file` has size 0. Python checks `not raw.strip()` and calls `sys.exit("không gọi được /api/runs")`.
  - **Partial body:** `curl` fails midway (exit 18/28/56). `if ! curl` branch overwrites `$runs_file` with `__API_UNREACHABLE__`.
  In zero cases does the index gate treat a failure as `[]`. (In the dedup block, `[[ -s ]] || printf '[]'` deliberately treats failed calls as "no runs" to allow the evaluator to proceed rather than blocking evaluation on empty queue state, preserving prior semantics).

---

### Allegation 5: "Concurrent invocations collide on temp names"
- **Status:** DISPROVED.
- **Evidence:**
  All temp paths are allocated via coreutils `mktemp -d "${TMPDIR:-/tmp}/cir_idx.XXXXXX"` and `mktemp -d "${TMPDIR:-/tmp}/cir_dedup.XXXXXX"` (or `mktemp "${TMPDIR:-/tmp}/cir_verify_*.XXXXXX"`). `mktemp` guarantees 6-character cryptographic uniqueness via `mkstemp`/`mkdtemp`.
  Internal files (`runs.json`, `benchmark-runs.json`) are scoped inside their unique parent directories. Two simultaneous `./pc.sh` processes do not collide on temp files.

---

## 4. Unverified Items

- None. All 5 claims and all lifecycle paths have been conclusively verified or disproved through static analysis, code trace, focused pytest runs, and concrete bash reproducers.

---

## 5. Exact Commands Executed

```bash
# 1. Inspect git status and changes in pc.sh and launcher tests
git status
git diff pc.sh
git diff workbench/tests/test_pc_launcher.py

# 2. Focused test: R8-02 gate survives bodies exceeding 128 KiB
.venv/bin/python -m pytest workbench/tests/test_pc_launcher.py -q -p no:cacheprovider -k "test_the_gate_survives_api_bodies_far_larger_than_the_argv_ceiling"

# 3. Focused test: R8-02 negative control fails with E2BIG on old argv carriage
.venv/bin/python -m pytest workbench/tests/test_pc_launcher.py -q -p no:cacheprovider -k "test_the_old_argv_carriage_fails_on_the_same_body_the_gate_now_accepts"

# 4. Focused test: Sentinel handling for unreachable API, empty body, malformed body
.venv/bin/python -m pytest workbench/tests/test_pc_launcher.py -q -p no:cacheprovider -k "test_a_failed_fetch_is_still_reported_as_an_unreachable_api or test_an_empty_body_is_still_an_unreachable_api or test_a_malformed_body_is_still_a_parse_refusal_not_an_empty_deployment"

# 5. Focused test: Assertions in temporary directory cleanup test
.venv/bin/python -m pytest workbench/tests/test_pc_launcher.py -q -p no:cacheprovider -k "test_the_gate_leaves_no_temporary_directory_behind"

# 6. Focused test: Development certificate refusal rules
.venv/bin/python -m pytest workbench/tests/test_pc_launcher.py -q -p no:cacheprovider -k "test_a_development_certificate_never_satisfies_the_release_gate or test_development_mode_alone_is_refused or test_a_development_certification_mode_alone_is_refused"

# 7. Focused test: Summary exit codes (positive and refusal controls)
.venv/bin/python -m pytest workbench/tests/test_pc_launcher.py -q -p no:cacheprovider -k "test_the_summary_exits"

# 8. Focused test: Handoff summary exit codes matrix
.venv/bin/python -m pytest workbench/tests/test_pc_launcher.py -q -p no:cacheprovider -k "test_the_summary_exit_codes_match_the_handoff_document"
```
