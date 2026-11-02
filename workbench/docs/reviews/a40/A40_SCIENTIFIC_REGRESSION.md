# Scientific Regression and Methodological Invariance Report (Round A40)

**Repository**: `/home/heheboiz/data/cir`  
**Branch**: `research/cir-failure-workbench`  
**Frozen Source Tested**: `624fcf9607d7397c8eef51a3453792a8a9f93de6` (phase history: b1b3511 initial defect fixes, 39ca45a reviewer rework, 624fcf9 verified tested source)  
**Candidate Delivery HEAD**: `4e57c544d2299e7fee2bfc298b9309ecebcf1a1f` (production tree byte-identical to tested source: `git diff --quiet 624fcf9 4e57c54 -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'` rc 0)  
**Audit Round**: A40
**Status**: ZERO SCIENTIFIC DEFINITION MUTATIONS

---

## 1. Executive Summary

Audit Round A40 strictly focused on operational safety, harness robustness, pipeline resume durability, and test defect resolution. **No scientific definitions, metric calculations, ranking formulas, dataset splits, or model evaluation recipes were modified.**

All changes between the baseline and tested source `624fcf9607d7397c8eef51a3453792a8a9f93de6` (byte-identical in production to candidate delivery HEAD `4e57c544d2299e7fee2bfc298b9309ecebcf1a1f`) were confined to operational tooling, process lifecycle management, path containment, responsive mobile viewport meta, and automated test coverage.

---

## 2. Inventory of Production Files Changed

Between the baseline audit start and tested source `624fcf9` (and candidate HEAD `4e57c54`), exactly four production source files were touched:

1. **`workbench/scripts/pipeline.py`** (Commits `bd342a4`, `1609635`, `39ca45a`):
   - Added process group session creation (`start_new_session=True`) and signal handlers in `run_command` (`A40-PIPE-03`).
   - Added `RUNNING` to `_UNUSABLE_DEP_STATUSES` (`A40-PIPE-02`).
   - Added `interrupted` flag handling in `run_stages` to report `INTERRUPTED` on signal/abort (`A40-PIPE-01`).
   - Wrapped `run_stages` in workflow-lifetime `SIGINT`/`SIGTERM` handlers (`REVIEW-03`).
   - *Impact on Science*: Zero. Stage execution ordering, input hashing, and command argument templates are completely unchanged.
2. **`workbench/scripts/load_mock_results.py`** (Commits `bd342a4`, `39ca45a`):
   - Added `is_safe_output_root` and `is_safe_target` path containment checks before recursive deletion of mock results (`A40-SEC-01`, `REVIEW-01`).
   - *Impact on Science*: Zero. Only affects synthetic mock fixture loading; real evaluation results are not processed by this script.
3. **`workbench/composition/model_csmcir.py`** (Commits `bd342a4`, `1609635`):
   - Added explicit `if not ... raise ValueError` ensuring worker `_job_root` is strictly within `work_root` (`A40-SEC-02`).
   - *Impact on Science*: Zero. Feature extraction, query composition math, visual projection, and similarity ranking logic are untouched.
4. **`workbench/frontend/index.html`** (Commit `39ca45a`):
   - Added viewport meta tag `<meta name="viewport" content="width=device-width, initial-scale=1" />` (`REVIEW-02`).
   - *Impact on Science*: Zero. Affects client-side browser viewport rendering only; no model or calculation impact.
---

## 3. Scientific Invariants and Verification Mechanisms

The research workbench enforces strict mathematical and dataset invariants to protect empirical validity. The following invariants were reviewed and confirmed unchanged:

| Invariant Category | Core Mathematical / Methodological Requirement | Protecting Automated Test Suite | Verification Status |
| :--- | :--- | :--- | :---: |
| **Recall@K Metric Denominators** | Recall@10 and Recall@50 denominators must strictly equal total evaluation split query counts; no candidate filtering or partial denominator scaling. | `workbench/tests/test_result_operations.py`, `workbench/tests/test_release_blocker_evaluation_output.py` | **CONFIRMED UNCHANGED** |
| **Evaluation Splits** | FashionIQ `val` splits (`dress`, `shirt`, `toptee`) and test pairs remain canonical. Zero data leakage across split boundaries. | `workbench/tests/test_prepare_dataset.py`, `workbench/tests/test_circus_adapters_blip.py` | **CONFIRMED UNCHANGED** |
| **Top-K Ranking Exactness** | Cosine similarity ranking and tie-breaking ordering produce bit-for-bit reproducible index rankings. | `workbench/tests/test_duckdb_schema.py`, `workbench/tests/test_sample_explorer_filtering.py` | **CONFIRMED UNCHANGED** |
| **Execution Identity Fingerprinting** | Input hashing incorporates model checkpoint digest, auxiliary weights, and evaluation configuration. Any parameter drift invalidates execution cache. | `workbench/tests/test_execution_identity.py`, `workbench/tests/test_adversarial_resume.py` | **CONFIRMED UNCHANGED** |
| **Model Weight Mapping** | Pinned checkpoint SHA256 digests in `models.yaml` map directly to official upstream releases. | `workbench/tests/test_artifact_hashing.py`, `workbench/tests/test_csmcir_checkpoint.py` | **CONFIRMED UNCHANGED** |

---

## 4. Release Count-Guard Interaction

The workbench contains automated sanity guards that verify model registration counts and evaluation result cardinality:
- **Model Registry Count Guard**: The platform strictly verifies that exactly **17** models are declared and registered in `models.yaml`. The test suite verifies this count in `test_operator_contracts.py` and the frontend validates "17 registered models". None of the A40 commits altered model definitions or registry counts.
- **Evaluation Cardinality Guard**: Automated validators check that benchmark results contain expected split totals. Because no evaluation pipelines were run against real weights during this round, no result ledgers were overwritten or altered.

---

## 5. Ruling

**RULING: ZERO SCIENTIFIC REGRESSION.**
All scientific code paths, benchmark metrics, feature extractors, and experimental artifacts remain strictly invariant under the A40 changes.
