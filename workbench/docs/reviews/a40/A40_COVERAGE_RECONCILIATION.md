# A40 Coverage Reconciliation & Contradiction Report (Gate G5)

## 1. Programmatic Recomputation of A39 Ledger & Matrices
Recomputed directly from the committed A39 review assets (`workbench/docs/reviews/a39/`) using `.scratch-tmp/a40-cover/recompute.py`:

- **Finding Ledger (`A39_FINDINGS.csv`)**:
  - Exact data row count: **45**
  - Status distribution across 8 classes:
    - `CONFIRMED_FIXED`: 16
    - `BLOCKED`: 7
    - `EXPECTED_BEHAVIOR`: 5
    - `NOT_REPRODUCED`: 5
    - `DOCUMENTATION_CORRECTED`: 4
    - `INFORMATIONAL`: 4
    - `CONFIRMED_UNFIXED`: 3
    - `TEST_GAP_FIXED`: 1

## 2. A39 Contradiction Table: Claims vs Recomputed CSV Data

The following table contrasts claims made in A39 markdown summaries (`A39_AUDIT_REPORT.md`) against actual data rows present in A39 CSV files:

| Claimed Metric / Inventory | Stated in A39 Report Narrative | Actual Rows in A39 CSV Export | Arithmetic & Semantic Status |
|---|---|---|---|
| **Source Coverage Census** | "360-row source inventory" (8 FULLY_REVIEWED, 14 EXECUTION_VERIFIED, 338 NOT_REVIEWED) | **14 rows** (`A39_SOURCE_COVERAGE.csv` contains 3 methodology/summary rows + 11 delta rows) | **CONTRADICT**: Full 360-row file-by-file inventory was omitted from CSV export. |
| **Functional Feature Inventory** | "35 functional features across 20 categories" | **21 rows** (`A39_FEATURE_MATRIX.csv` contains 20 category rollup rows F00–F19 + 1 summary total row) | **CONTRADICT**: The 35 distinct features were not individually itemized in machine-readable form. |
| **Branch Coverage Census** | "77 total branches cataloged" (63 NOT_TESTED, 14 NOT_APPLICABLE, 0 PASS) | **65 rows** (`A39_BRANCH_MATRIX.csv` contains 1 header summary row + 63 branch rows + 1 fix summary row) | **CONTRADICT**: 14 `NOT_APPLICABLE` branches were omitted as data rows from the CSV export. |
| **Branch Test Mapping** | "57 mapped to existing tests, 6 do not" | CSV contains 58 `TEST_EXISTS`, 3 `NO_TEST`, 2 `NOT_APPLICABLE` (+ 2 summary rows); narrative identifies 6 unmapped (`F06` interrupted/retry/unexpected_exception, `F03` retry/missing_dependency/interrupted) | **CONTRADICT**: Prose claim of 6 unmapped paths contradicts the 3 `NO_TEST` rows in the exported CSV. |
| **Finding Count & Distribution** | 45 findings, 16 CONFIRMED_FIXED | **45 data rows**, 16 CONFIRMED_FIXED in `A39_FINDINGS.csv` | **MATCH**: Findings ledger is fully consistent between narrative and CSV. |

## 3. How A40 Matrices Supersede A39 Summaries

Round A40 replaces incomplete rollups and partial summaries with exhaustive, machine-readable inventories:

- **`A40_SOURCE_COVERAGE.csv` (610 rows)**:
  - Enumerates tracked first-party files from `git ls-files` outside vendored subtrees, plus one aggregate range entry for `workbench/third_party/**` (7,728 files).
  - Recomputed inspection distribution: 229 `FULLY_REVIEWED_EXECUTED`, 127 `STATICALLY_REVIEWED`, 253 `ENUMERATED_NOT_REVIEWED`, and 1 `NOT_APPLICABLE`.
  - **Coverage caveat:** the 174 `category == "test"` rows are classified by tracked test-suite membership (`PYTEST_SUITE_MEMBERSHIP`), not per-test execution tracing. No claim is made that each test file ran in this round.
  - For Python modules, `FULLY_REVIEWED_EXECUTED` requires AST-confirmed test import and a later reference to the imported module binding. Filename/path substrings do not qualify. Source-manifest checksum tables do not qualify as reviews.
  - Compared with the prior generated matrix, 46 rows moved from `FULLY_REVIEWED_EXECUTED` to `ENUMERATED_NOT_REVIEWED`; 34 moved from `STATICALLY_REVIEWED` to `ENUMERATED_NOT_REVIEWED` because their only cited evidence was a checksum manifest. Counts are recomputed from status transitions: prior 275/161/173/1; corrected 229/127/253/1.
  - 174 test-category rows remain suite-membership inventory entries, not evidence that every test ran. They remain classified separately from production modules so this limitation is explicit.

2. **`A40_FEATURE_MATRIX.csv` (35 rows)**:
   - Expands the 20 category rollup rows from A39 into all 35 distinct functional features (F01–F14, F20–F25).
   - Recomputed evidence-class distribution: 34 `FIXTURE_EXECUTED`, 1 `STATIC_ONLY`, 0 `REAL_ASSET_EXECUTED`, 0 `NOT_VERIFIED`.
   - F02/F02.1 tests use synthetic `tmp_path` fixtures; the 74,381-path figure is from a prior A39 static audit. F11/F11.1 use this round's headless Chrome E2E against synthetic mock fixtures. F14 remains static-only.
   - Explicitly notes prerequisite blockers (e.g. absent model checkpoints blocking real model evaluation inference).

3. **`A40_BRANCH_MATRIX.csv` (77 rows)**:
   - Full census of all 77 execution branches (60 EXECUTED_PASS, 16 NOT_APPLICABLE, 1 NO_TEST).
   - Incorporates A40 pipeline defect fixes and interrupt regression tests (`workbench/tests/test_a40_defects.py` and `test_a40_pipeline_interrupt.py`), converting previously unexercised pipeline interruption and recovery branches into verified passing branches (`EXECUTED_PASS`).
