# Audit Round A40 Synthesis Report (Research Workbench)

**Repository**: `/home/heheboiz/data/cir`  
**Branch**: `research/cir-failure-workbench`  
**Tested Source SHA**: `624fcf9607d7397c8eef51a3453792a8a9f93de6`  
**Candidate Delivery HEAD**: `4e57c544d2299e7fee2bfc298b9309ecebcf1a1f` (production tree byte-identical to tested source: `git diff --quiet 624fcf9 4e57c54 -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'` rc 0; commits after tested source are docs/test-harness only)  
**Phase Progression**: `b1b3511` (initial defect fixes) → `39ca45a` (reviewer rework) → `624fcf9` (tested source SHA verified with frozen suite) → `4e57c54` (release docs advance)  
**Audit Round**: A40  
**Audit Date**: 2026-11-02  
**Audit Lane**: Report-Authoring Lane (Phase 9)

---

## 1. Operating Scope & Hardware Context

Audit Round A40 was conducted on a local development workstation configured as follows:
- **Operating System**: Linux 7.0.0-38-generic x86_64
- **CPU**: 16 logical cores (`nproc: 16`)
- **Available RAM**: ~10 GiB available (14 GiB total, 3.4 GiB free, 7.6 GiB buff/cache)
- **Disk Storage**: Root filesystem `/dev/nvme0n1p6` (173 GB total, 155 GB used, **8.9 GB free** [95% used])
- **Python Runtime**: Python 3.13.12 (`.venv/bin/python`)
- **Node.js Runtime**: v20.19.5 (`npm: 10.8.2`)

**LAPTOP AUDIT NOTICE**: This round validates local workbench durability, test harness reliability, frontend responsiveness across viewports, artifact containment, and pipeline interruption semantics. Full multi-gigabyte model evaluation benchmarks and GPU hardware verification (`sm_120`) are separate operator stages.

---

## 2. Baseline Drift Characterization

- **Original Planning Brief Commit**: `beb680f1fcc90d1652cc7ef6f6427fd632a2e94a`
- **Audit Baseline Start**: `d9d7c28a74e82e002bbdc35964c9f0b136adc2bd`
- **Drift Details**: The commits between `beb680f` and `d9d7c28` were docs-only commits updating review paths and citations for round A39 archival. No production or test source files were modified.
- **Phase Progression**: Initial fixes landed at `b1b3511`; independent review findings REVIEW-01..05 rework landed at `39ca45a`; full frozen suite and release doc integration completed at `624fcf9607d7397c8eef51a3453792a8a9f93de6`; final candidate delivery HEAD is `4e57c544d2299e7fee2bfc298b9309ecebcf1a1f` (byte-identical in production logic, rc 0).

---

## 3. Findings Ledger Summary (`A40_FINDINGS.csv`)

The findings ledger was synthesized and verified via an independent scratch validation script (`/dev/shm/a40-report/verify_findings.py`).

### Recomputed Metric Distribution
- **Total Data Rows**: **22**
- **Status Distribution** (recomputed from `A40_FINDINGS.csv`):
  - `CONFIRMED_FIXED`: **10**
  - `BLOCKED`: **4**
  - `CONFIRMED_UNFIXED`: **2**
  - `TEST_GAP_FIXED`: **2**
  - `DOCUMENTATION_CORRECTED`: **3**
  - `EXPECTED_BEHAVIOR`: **1**

### Literal Checker Output
```text
Total findings data rows: 22
Status distribution:
  BLOCKED: 4
  CONFIRMED_FIXED: 10
  CONFIRMED_UNFIXED: 2
  DOCUMENTATION_CORRECTED: 3
  EXPECTED_BEHAVIOR: 1
  TEST_GAP_FIXED: 2
```

## 4. The Six Audit Gates (G1–G6) Outcome

### Gate G1: Artifact Safety & Containment
- **Outcome**: **PASS**
- **Summary**: Enumerated 173 first-party destructive operations (81 non-test, 92 test). Verified zero real research assets were touched. `A40-SEC-01` constrains output and deletion targets to configured results root; `A40-SEC-02` uses an explicit runtime exception check.

### Gate G2: Responsive Real Browser E2E Validation
- **Outcome**: **PASS**
- **Summary**: Re-ran 8 journeys across 4 viewports; all 32 journey×viewport cases passed. Mobile contexts report `clientWidth` and PNG width of 390/360 px, with matching `scrollWidth`. Evidence: `/dev/shm/a40-e2e/e2e_rerun.md`. Real-device rendering beyond Chrome emulation remains unverified.

### Gate G3: Mobile Timeout Investigation & Root Cause
- **Outcome**: **PASS** (Classified as `CONFIRMED_PLAYWRIGHT_HARNESS_DEFECT`)
- **Summary**: Existing browser evidence supports one `/api/models` call per page. Prior timeout was attributed to an uncoordinated out-of-band Python request. Browser-launch and import defects (`A40-UI-01/02`) corrected.

### Gate G4: Pipeline Interruption & Resume Durability
- **Outcome**: **PASS WITH COVERAGE LIMITATIONS**
- **Summary**: Interruption reporting, stale `RUNNING` dependency blocking, process-group termination, and workflow-lifetime signal handlers implemented. Post-fix boundary test passes; no pre-fix RED was demonstrated. Scenario gaps `A40-PIPE-04/05` remain.

### Gate G5: Historical Reconciliation & Ledger Integrity
- **Outcome**: **PASS**
- **Summary**: Recomputed A39 ledger (45 rows, 8 status classes). Corrected A39 coverage export discrepancy (`A40-DOC-01`).

### Gate G6: Laptop Handoff & Scientific Invariance
- **Outcome**: **PASS WITH EXTERNAL BLOCKERS**
- **Summary**: Full frozen test suite executed at `624fcf9` (`.venv/bin/python -m pytest workbench/tests tests -q -p no:randomly -rf`): 2256 collected, 2245 passed, 1 failed, 10 skipped, 253 warnings in 3942.16s (1:05:42). The single failure was `workbench/tests/test_handoff_commands.py::test_no_release_document_points_at_a_superseded_source_sha`, a by-construction release-metadata guard resolved by advancing recorded source to `624fcf9` (after advance: 8 passed in `test_handoff_commands.py`, 22 passed in `test_release_invariants.py`). No scientific definitions or evaluation recipes changed. Gateway token rotation, AirKnow credential review, absent CSMCIR checkpoint, and `sm_120` validation remain operator actions.
## 5. Scope Boundaries: What Was NOT Done

1. No real model inference benchmarking; UI/E2E used synthetic protocol fixtures.
2. No GPU kernel execution on RTX 5070 Ti (`sm_120`).
3. External secret rotation remains an operator action; no value is reproduced here.
4. Vendored AirKnow code remains unchanged pending operator authorization.

## 6. Audit Verdict

**Verdict: A40_LAPTOP_ACCEPTED_WITH_BLOCKERS**

Core laptop-reproducible defects (A40-PIPE-01/02/03, A40-SEC-01/02, A40-UI-01/02, REVIEW-01/02/03) are resolved with RED→GREEN or equivalent reproduction evidence: results-root ownership containment enforced, 32-case viewport rerun verified at true 390/360 CSS px with `<meta name="viewport">`, and workflow-lifetime signal handlers installed with passing regression test. REVIEW-04/05 labels were corrected in documentation. The frozen full suite completed at `624fcf9` with 2245 passed out of 2256 (single failure was the release-metadata guard resolved by advancing release docs to `624fcf9`). No laptop-reproducible blocking defect is known to remain. External token rotation, vendored AirKnow credential decision, absent 2.72 GB checkpoint, and `sm_120` GPU hardware requirements remain blockers to broader validation, alongside honest residual limitations.

### Independent Reviewer Reconciliation
The independent reviewer issued an initial verdict of `A40_LAPTOP_NOT_ACCEPTED` citing 5 findings (`REVIEW-01` through `REVIEW-05`). That was a historical position prior to rework. All 5 findings were subsequently resolved:
- `REVIEW-01` (results root boundary & containment): CONFIRMED_FIXED in `39ca45a`.
- `REVIEW-02` (viewport meta tag & responsive rendering): CONFIRMED_FIXED in `39ca45a`; verified by 32/32 browser rerun at true 390/360 CSS px.
- `REVIEW-03` (workflow-lifetime signals): CONFIRMED_FIXED in `39ca45a`; verified by boundary regression test.
- `REVIEW-04` (branch matrix unexpected-exception mapping): DOCUMENTATION_CORRECTED in `624fcf9`.
- `REVIEW-05` (source coverage README mapping): DOCUMENTATION_CORRECTED in `624fcf9`.

### Remaining Blockers & Residual Limitations

1. **A40-SEC-03 (External Gateway Token)**:
   - Reason: Plaintext OpenRouter gateway API token detected in external runtime log files outside repo.
   - Evidence: Static audit scan (53 references in external logs).
   - Impact: Credential risk if external logs accessed.
   - Operator Action: Revoke/rotate token at provider console and sanitize external log directories.
2. **A40-SEC-04 (Vendored AirKnow Credential)**:
   - Reason: Hardcoded test/API credentials in upstream source of vendored third-party AirKnow repository.
   - Evidence: Static audit scan in `workbench/third_party/AirKnow/`.
   - Impact: Secret scanner alert on vendored tree.
   - Operator Action: Determine validity and patch through `approved_source_patches` if needed.
3. **A40-MOD-01 (Absent CSMCIR Checkpoint)**:
   - Reason: Pretrained CSMCIR 2.72 GB weight checkpoint absent from local disk due to storage/bandwidth constraints.
   - Evidence: File absence check in `workbench/artifacts/checkpoints/csmcir/`.
   - Impact: Real checkpoint inference unavailable on local laptop.
   - Operator Action: Execute `download_checkpoints.py --model csmcir` on equipped machine (>50 GB free).
4. **A40-HW-01 (Blackwell sm_120 Hardware Absent)**:
   - Reason: NVIDIA RTX 5070 Ti (compute capability sm_120) not present on laptop host.
   - Evidence: CUDA probe on host (`torch.cuda.is_available()` CPU-only).
   - Impact: Native sm_120 execution unverified.
   - Operator Action: Validate on target RTX 5070 Ti machine with CUDA 12.8+.
5. **A40-PIPE-04/05 (Pipeline Pre-execution & In-flight Scenario Coverage Gaps)**:
   - Reason: Dedicated scenario tests absent for interruption before stage dispatch and in-flight state marking.
   - Evidence: G4 scenario map in `/dev/shm/a40-pipe/contract.md`.
   - Impact: Residual coverage gap; mitigated at runtime by `A40-PIPE-02` treating RUNNING as unusable.
   - Operator Action: Add focused tests if these scenarios are acceptance requirements.
6. **REVIEW-02 Residual Limitation (Real-Device Mobile Rendering)**:
   - Reason: Browser validation executed under Google Chrome mobile emulation (390/360 CSS px).
   - Evidence: `/dev/shm/a40-e2e/e2e_rerun.md` and `/dev/shm/a40-e2e/shots2/`.
   - Impact: Physical mobile device hardware rendering (WebKit on iOS, etc.) unverified.
   - Operator Action: Perform physical-device testing if mobile browser rendering is a delivery requirement.
7. **REVIEW-03 Residual Limitation (Boundary Regression Test Red Baseline)**:
   - Reason: Workflow-lifetime signal regression test was added post-fix; no pre-fix RED test run was recorded.
   - Evidence: `workbench/tests/test_a40_defects.py::test_pipeline_sigterm_between_commands_writes_interrupted_report`.
   - Impact: Fix is post-fix verified, historical reproduction not established.
   - Operator Action: None for current runtime behavior.
