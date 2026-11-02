# Pipeline Interruption and Resume Contract Report (Gate G4 — Round A40)

**Repository**: `/home/heheboiz/data/cir`  
**Branch**: `research/cir-failure-workbench`  
**Frozen Source Tested**: `624fcf9607d7397c8eef51a3453792a8a9f93de6` (phase history: b1b3511 initial defect fixes, 39ca45a reviewer rework, 624fcf9 verified tested source)
**Audit Round**: A40  
**Evidence Sources**: `/dev/shm/a40-pipe/contract.md`, `gaps.md`, `workbench/tests/test_a40_pipeline_interrupt.py`

---

## 1. Stage State Machine

### Execution Stages
Standard workflows (`pipeline.py:1259-2075`) execute through defined topological stages:
- `doctor`: Workspace and platform diagnostics.
- `dataset`: FashionIQ split and canonical asset validation.
- `sync`: Upstream Git repository synchronization.
- `dataset-link`: CSMCIR symlink/data link provisioning.
- `environment`: Per-model virtual environment installation.
- `checkpoint`: Pretrained weight acquisition and SHA256 integrity checks.
- `auxiliary-assets`: Tokenizers, vocabularies, and feature caches.
- `gpu-verification`: CUDA/GPU hardware capability probe.
- `preparation`: Recipe and feature layout transformation.
- `runtime-preflight`: Pre-evaluation environment and execution preflight.
- `evaluation`: Model benchmark evaluation run.
- `validation-index`: Validation and DuckDB index rebuild.
- `serve`: Interactive operator dashboard server.
- `report`: Diagnostic report generation.

### Stage States & Transitions
- `RUNNING`: Stage commands are executing or about to execute.
- `COMPLETE`: All stage commands exited 0 and output verification succeeded.
- `FAILED`: Command exited non-zero or declared outputs failed validation.
- `SKIPPED_DEPENDENCY`: Upstream dependency failed, is missing, or is unusable.
- `BLOCKED_AUTHORIZATION_REQUIRED`: Side-effecting stage lacks required operator capability.
- *Interruption Semantics*: On unhandled process termination, the in-flight stage remains persisted as `RUNNING` in `pipeline_state.json`.

---

## 2. Resume Semantics & Validation Rules

When the pipeline is invoked with `--resume`:
1. **Re-invocation Reuse Decision**:
   - The stage must be recorded with `status: "COMPLETE"`.
   - The computed input fingerprint must match the recorded fingerprint.
   - Declared file outputs must exist, be non-empty, and match recorded hashes.
   - If outputs are missing or corrupt, `can_skip` evaluates to `False`, forcing the stage and its downstream dependents into `rerun_stages`.
2. **Handling of Stale `RUNNING` State (Fixed in `A40-PIPE-02`)**:
   - Prior to A40, `_UNUSABLE_DEP_STATUSES` omitted `RUNNING`. If a stage was interrupted mid-flight, a subsequent selective run targeting a downstream dependent would execute against incomplete upstream outputs.
   - **Fix**: Added `RUNNING` to `_UNUSABLE_DEP_STATUSES` (`pipeline.py:703`). Now, downstream dependents correctly fail fast with `SKIPPED_DEPENDENCY` (`return_code: 125`).

---

## 3. Required Scenario Map (12 Scenarios)

The Gate G4 contract mapped 12 critical interruption and concurrency scenarios:

| # | Scenario Description | What SHOULD Happen | What DOES Happen | Status & Citation |
|---|---|---|---|---|
| **1** | Interrupt before stage execution begins | Stage unstarted; lock released; state clean; on resume runs normally. | `KeyboardInterrupt` unwinds; lock released in `finally`; stage unrecorded; `write_report` records status cleanly. | `NOT_COVERED` |
| **2** | Interrupt during subprocess execution | Subprocess & process group terminated; state marked `INTERRUPTED`/`FAILED`; lock released; on resume re-runs. | Process group killed via `killpg`; stage left `RUNNING` in state file; lock released in `finally`. Treated as unusable on subsequent run. | `NOT_COVERED` (marked in state at death) |
| **3** | Interrupt after subprocess success but before receipt publication | No receipt published; stage unfinalized; on resume output validation fails and re-runs. | No receipt written; stage remains `RUNNING`; on resume `validate_stage_outputs` detects missing receipt, forcing re-run. | **COVERED** (`workbench/tests/test_a40_pipeline_interrupt.py::test_interruption_between_subprocess_exit_and_receipt_publication`) |
| **4** | Interrupt during receipt writing | Atomic write guarantees either old receipt or complete new receipt; no corrupt file. | Temporary file `path.tmp.<pid>` written and atomic `Path.replace` ensures no torn files. | `NOT_COVERED` (atomic rename contract) |
| **5** | Resume after interruption | Interrupted stage (`RUNNING`) is detected and not skipped; re-executes. | `prev.get("status") == "COMPLETE"` evaluates to `False`; stage re-runs. | **COVERED** (`test_adversarial_resume.py::test_r07_interrupted_not_skipped`) |
| **6** | Resume with unchanged inputs | Matching fingerprint and valid outputs allow stage skip; zero commands run. | Fingerprint matches; `validate_stage_outputs` passes; commands skipped. | **COVERED** (`test_adversarial_resume.py::test_r10_independent_stages_cached`) |
| **7** | Resume with changed inputs | Changed input alters fingerprint; stage and dependents re-run. | Fingerprint mismatch; added to `rerun_stages`; downstream invalidated. | **COVERED** (`test_execution_identity.py::test_execution_identity_detects_evaluator_mutation`) |
| **8** | Resume with missing outputs | Missing declared output detected; skip refused; stage re-runs. | `validate_stage_outputs` detects `not p.exists()`; skip refused. | **COVERED** (`test_adversarial_resume.py::test_r05_output_deletion_invalidates`) |
| **9** | Resume with corrupt outputs | Corrupt output (JSON error / hash mismatch) detected; stage re-runs. | `validate_stage_outputs` detects hash mismatch; skip refused. | **COVERED** (`test_adversarial_resume.py::test_r06_output_corruption_invalidates`) |
| **10** | Resume after a failed subprocess | Prior failed stage (`FAILED`) is re-executed on resume re-invocation. | `prev.get("status") == "COMPLETE"` evaluates to `False`; stage re-executes. | **COVERED** (`workbench/tests/test_a40_pipeline_interrupt.py::test_resume_after_failed_subprocess_reexecutes`) |
| **11** | Concurrent invocation while stage owns lock | Contender refused with `WorkflowLockError` / exit 3; no state corruption. | Contender non-blocking `flock` fails; raises `WorkflowLockError`; exits 3. | **COVERED** (`test_concurrency_ownership.py::test_two_processes_contend_for_same_state`) |
| **12** | Recovery when previous owner crashed | Stale lock auto-cleared on process death; next process acquires lock cleanly. | Kernel releases `flock` on process crash; next invocation acquires lock cleanly. | **COVERED** (`test_concurrency_ownership.py::test_lock_released_after_crash`) |

### Scenario Accounting
- **COVERED**: **9** scenarios (7 baseline + 2 added in commit `b1b3511`: Scenario 3 and Scenario 10).
- **NOT_COVERED**: **3** scenarios (Scenario 1, Scenario 2, Scenario 4).

---

## 4. Key Fixes Delivered in A40

1. **`A40-PIPE-01` (Workflow Report Truthfulness)**:
   - In `run_stages`, `failed = False` was not updated when `KeyboardInterrupt` or `SystemExit` occurred, causing `write_report` to generate `workflow_report.json` with `"success": true, "status": "SUCCESS"`.
   - **Fix**: Added explicit `interrupted` tracking and signal handling in `run_stages` to record `"success": false` and `"status": "INTERRUPTED"`.
2. **`A40-PIPE-02` (Stale RUNNING Unusable Dependency)**:
   - Added `"RUNNING"` to `_UNUSABLE_DEP_STATUSES` (`pipeline.py:703`), preventing downstream dependent stages from executing against an incomplete interrupted dependency.
3. **`A40-PIPE-03` & `REVIEW-03` (Process Group Isolation and Workflow-Lifetime Signal Handling)**:
   - Subprocesses are spawned via `Popen(argv, start_new_session=True)`.
   - `run_command` installs signal handlers executing `os.killpg(proc.pid, signal.SIGTERM)` with bounded wait and escalation to `SIGKILL`, ensuring no child or grandchild process outlives the pipeline.
   - Furthermore, `REVIEW-03` (commit `39ca45a`) wrapped the entire `run_stages` execution in workflow-lifetime `SIGINT`/`SIGTERM` handlers, ensuring signals arriving in workflow gaps between subprocesses invoke `write_report()` and record `INTERRUPTED` rather than bypassing cleanup.

---

## 5. Residual Limitations

1. **Kernel `SIGKILL` vs Unhandled State Rewriting**:
   - Workflow-lifetime handlers cleanly catch `SIGINT` and `SIGTERM`. However, uncatchable `SIGKILL` or hardware power loss terminates the Python process abruptly without executing `finally: write_report()`. In that event, the in-flight stage remains persisted as `RUNNING` in `pipeline_state.json`. (Mitigated: `A40-PIPE-02` ensures `RUNNING` stages are treated as unusable on resume).
2. **`INTERRUPTED` as a Report Status vs Stage Status**:
   - `INTERRUPTED` is recorded in `workflow_report.json`, but individual stage records in `pipeline_state.json` transition to `FAILED` or remain `RUNNING`.
