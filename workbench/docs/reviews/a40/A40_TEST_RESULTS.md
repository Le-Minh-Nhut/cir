# Automated Test Execution and Regression Report (Round A40)

**Repository**: `/home/heheboiz/data/cir`  
**Branch**: `research/cir-failure-workbench`  
**Tested Source SHA**: `b1b35117f3ab1b4fc0a5a21056d3799a71c06a66`  
**Audit Round**: A40  
**Interpreter**: Python 3.13.12 (`.venv/bin/python`)  
**Node.js**: v20.19.5  
**Execution Environment**: Linux 7.0.0-38-generic x86_64 (16 CPUs, ~10 GiB available RAM)

---

## 1. Targeted Regression Test Results (RED → GREEN Verifications)

Targeted regression suites were executed against the frozen code to verify fixes for confirmed defects D1–D4, harness defect `A40-UI-01`, and pipeline interruption gaps.

**Evidence Source**: `/dev/shm/a40-fix/regressions.txt`

| Test Target / Suite | Tests Executed | Passed | Failed | Duration | Verification Focus |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `workbench/tests/test_a40_defects.py` | 4 | 4 | 0 | 0.45s | Confirmed fixes for `A40-PIPE-01`, `A40-PIPE-02`, `A40-PIPE-03`, `A40-SEC-01` |
| `workbench/tests/test_a40_pipeline_interrupt.py` | 3 | 3 | 0 | 0.85s | Scenario 3 (receipt publication interrupt) and Scenario 10 (resume after failure) |
| `workbench/tests/test_pipeline.py` | 32 | 32 | 0 | 52.39s | Full core pipeline state machine, locks, and stage transitions |
| `workbench/tests/test_adversarial_resume.py` | 8 | 8 | 0 | 0.79s | Cache validation, output corruption detection, and resume skip guards |
| `workbench/tests/test_result_operations.py` | 7 | 7 | 0 | 0.59s | Result metrics, ledger parsing, and Recall@K calculation invariance |
| `workbench/tests/test_circus_adapters_blip.py` & `test_csmcir_checkpoint.py` | 23 | 23 | 0 | 0.32s | Model adapters and CSMCIR work-root containment explicit ValueError |
| `workbench/tests/test_operator_contracts.py` | 10 | 10 | 0 | 109.53s | Operator release contracts, path boundaries, and protected roots preservation |
| **Targeted Total** | **87** | **87** | **0** | **~165s** | **100% PASS on targeted regressions** |

### Breakdown of the Two New Test Files
1. **`workbench/tests/test_a40_defects.py`** (Created in commit `bd342a4`):
   - `test_pipeline_interrupt_report_is_truthful`: Validates that `KeyboardInterrupt` in `run_stages` marks `workflow_report.json` with `success: false` and `status: "INTERRUPTED"`.
   - `test_pipeline_stale_running_dependency_blocks_downstream`: Validates that `_UNUSABLE_DEP_STATUSES` contains `RUNNING` and prevents executing dependents on interrupted upstream stages.
   - `test_pipeline_interrupt_terminates_child_process_group`: Validates process group kill on signal delivery with zero orphaned descendants.
   - `test_load_mock_results_refuses_out_of_root_target`: Validates `is_safe_output_root` and `is_safe_target` boundary enforcement.
2. **`workbench/tests/test_a40_pipeline_interrupt.py`** (Created in commit `b1b3511`):
   - `test_resume_after_failed_subprocess_reexecutes`: Covers Scenario 10 (re-execution of failed stage on resume).
   - `test_interruption_between_subprocess_exit_and_receipt_publication`: Covers Scenario 3 (missing receipt after interrupt forces re-run).
   - `test_resume_with_changed_inputs_invalidates_stage_and_downstream`: Covers Scenario 7 (changed inputs invalidate stage and downstream).

---

## 2. Full Frozen Test Suite Status

- **Status**: **IN_PROGRESS — accounting to be recorded by the coordinator**
- **Tested Source Commit**: `b1b35117f3ab1b4fc0a5a21056d3799a71c06a66`
- **Execution Command**:
  ```bash
  export TMPDIR=/dev/shm/a40-suite && \
  .venv/bin/python -m pytest workbench/tests/ -v --tb=short
  ```
- **Reason for In-Progress State**: The full workbench regression suite encompasses hundreds of integration and operator contract tests requiring ~60–70 minutes of sequential execution time. At the time of report authoring, the background runner was actively progressing (`suite-output.txt` actively accumulating pass markers; final summary file `/dev/shm/a40-suite/summary.txt` not yet published).
- **Protocol Rule Enforced**: No test counts or pass/fail ratios are fabricated. The coordinator will record the final accounting upon suite completion.

---

## 3. The `TESTED_SOURCE_SHA` Concept

In multi-lane audit rounds, documentation files and audit reports are written concurrently with or following the test run.
- **`TESTED_SOURCE_SHA`**: `b1b35117f3ab1b4fc0a5a21056d3799a71c06a66`.
  This is the frozen commit containing all production and test code changes. The full automated test suite executes strictly against this commit.
- **Documentation HEAD**: The final git commit will encompass the authoring of review documents under `workbench/docs/reviews/a40/`. Because review docs do not modify production or test logic, the testing evidence from `TESTED_SOURCE_SHA` remains mathematically valid for the final documentation commit.
