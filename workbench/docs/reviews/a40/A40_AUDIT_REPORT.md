# Audit Round A40 Synthesis Report (Research Workbench)

**Repository**: `/home/heheboiz/data/cir`  
**Branch**: `research/cir-failure-workbench`  
**Frozen Production SHA**: `b1b35117f3ab1b4fc0a5a21056d3799a71c06a66`  
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
- **Frozen Test HEAD**: `b1b35117f3ab1b4fc0a5a21056d3799a71c06a66` (incorporating fixes for D1–D4, frontend launch, and interrupt test gaps).

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
- **Summary**: No scientific definitions or evaluation recipes changed. Gateway token rotation, AirKnow credential review, absent CSMCIR checkpoint, and `sm_120` validation remain operator actions.

## 5. Scope Boundaries: What Was NOT Done

1. No real model inference benchmarking; UI/E2E used synthetic protocol fixtures.
2. No GPU kernel execution on RTX 5070 Ti (`sm_120`).
3. External secret rotation remains an operator action; no value is reproduced here.
4. Vendored AirKnow code remains unchanged pending operator authorization.

## 6. Audit Verdict

**Verdict: A40_LAPTOP_ACCEPTED_WITH_BLOCKERS**

Core laptop-reproducible defects REVIEW-01/02/03 are fixed; targeted ownership/SIGTERM regressions and 32-case viewport rerun provide evidence. REVIEW-04/05 labels were corrected, and their residual coverage limitations are recorded rather than overstated. No laptop-reproducible defect is known from the reviewed findings, but the frozen full-suite result is still PENDING; therefore full acceptance is not claimed. External token/AirKnow, checkpoint, and hardware requirements remain blockers to broader validation, alongside pipeline scenario coverage gaps.

### Remaining blockers
- **Frozen full suite** — pending: no final suite result recorded yet. Impact: suite-wide readiness unconfirmed. Operator action: complete frozen-suite run and record result.
- **A40-SEC-03** — external gateway token exposure remains unrotated. Impact: credential risk. Operator action: revoke/rotate token and sanitize external logs.
- **A40-SEC-04** — vendored AirKnow credential awaits review/approved patch. Impact: credential status unresolved. Operator action: determine validity and patch through approved mechanism if needed.
- **A40-MOD-01** — 2.72 GB CSMCIR checkpoint absent. Impact: real checkpoint inference unavailable. Operator action: download and verify checkpoint.
- **A40-HW-01** — `sm_120` hardware not available. Impact: target GPU execution unverified. Operator action: validate on RTX 5070 Ti with supported CUDA.
- **A40-PIPE-04/05** — dedicated scenario tests remain absent; existing state behavior mitigates downstream risk but does not close test coverage. Impact: coverage incomplete. Operator action: add focused tests if these scenarios are acceptance requirements.
- **REVIEW-02** — real-device rendering beyond Chrome emulation unverified. Impact: device-specific behavior not established. Operator action: perform physical-device validation if required.
- **REVIEW-03** — boundary regression has no pre-fix RED evidence. Impact: fix is post-fix verified, historical reproduction not established. Operator action: none for current behavior.
- **REVIEW-04/05** — unexpected-exception semantics and exhaustive per-file assertion coverage remain unverified; labels were corrected. Impact: evidence scope remains limited. Operator action: broaden tests only if those claims are required.

**PENDING: the frozen full-suite run.**
