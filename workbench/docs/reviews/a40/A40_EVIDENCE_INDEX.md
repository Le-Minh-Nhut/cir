# A40 Evidence Index

| Claim / Report Area | Primary Evidence Path | Specific Evidence / Reproducible Command |
|---|---|---|
| Complete First-Party Source Coverage Inventory | `workbench/docs/reviews/a40/A40_SOURCE_COVERAGE.csv` | 610 tracked first-party files classified by inspection status, method, and test references |
| Complete 77-Branch Coverage Census | `workbench/docs/reviews/a40/A40_BRANCH_MATRIX.csv` | 77 execution branches with file, line, expected behavior, mapped tests, and execution status |
| Individual 35-Feature Verification Matrix | `workbench/docs/reviews/a40/A40_FEATURE_MATRIX.csv` | 35 individual functional features with evidence class (FIXTURE_EXECUTED vs REAL_ASSET_EXECUTED) and prerequisites |
| A39 to A40 Ledger Reconciliation & Contradictions | `workbench/docs/reviews/a40/A40_COVERAGE_RECONCILIATION.md` | Programmatic recomputation showing exact arithmetic contradictions in A39 summaries |
| Pipeline Interruption & Report Truthfulness Defect Fixes | `workbench/tests/test_a40_defects.py` | `pytest workbench/tests/test_a40_defects.py`: verifies interrupt report truthfulness, stale RUNNING dep blocking, process-group SIGINT/SIGTERM termination, mock results path check |
| Pipeline Interruption & Resume Durability Tests | `workbench/tests/test_a40_pipeline_interrupt.py` | `pytest workbench/tests/test_a40_pipeline_interrupt.py`: verifies resume after failed subprocess, interruption before receipt publication, changed-input invalidation |
| Pipeline Execution & Interruption Contract Specification | `/dev/shm/a40-pipe/contract.md` (ephemeral); documented in `test_a40_pipeline_interrupt.py` | Formal stage state machine transitions and receipt invariants |
| Pipeline Interruption Test Gaps Specification | `/dev/shm/a40-pipe/gaps.md` (ephemeral); addressed in `test_a40_defects.py` | Priority ranking of signal handling, process group orphan prevention, and report truthfulness |
| A39 Audit Base Ledger & Defect Classification | `workbench/docs/reviews/a39/A39_FINDINGS.csv` | 45 cataloged findings (16 FIXED, 7 BLOCKED, 5 EXPECTED, 5 NOT_REPRODUCED, 4 DOC, 4 INFO, 3 UNFIXED, 1 GAP_FIXED) |
| Scientific Metric & Layout Verification | `workbench/docs/reviews/a39/A39_SCIENTIFIC_VALIDATION.md` | FashionIQ 74,381 candidate paths verification; recall metric recomputation |
| Security Exposure & Token Scans | `workbench/docs/reviews/a39/A39_SECURITY_REVIEW.md` | Scans for API secrets, tokens, and path-traversal safeguards |

---

### Ephemeral Storage and Reproducibility Notice
Note: Paths located under `/dev/shm/a40-*` are hosted on ephemeral `tmpfs` filesystems and will not persist across system reboots or scratch cleanups. Tracked documentation and audit reports must not rely on files residing strictly in `/dev/shm`. All verification claims in A40 reports are backed by reproducible CLI commands, checked-in regression test suites (`workbench/tests/test_a40_*.py`), and committed CSV matrices in `workbench/docs/reviews/a40/`.
