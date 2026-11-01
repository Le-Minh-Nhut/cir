# A39 Audit Report

## A. Git and scope

Branch: `research/cir-failure-workbench`. Baseline: `1208029d90d28b4c5e780e32363acbe1e850981d`; frozen test source: `3015df489a3e3a1d7db47d606d1fe465c12ddfe8`. Current production source: `2ff698e5b201cbc5bd578e77d1dd7bbe952f1069`; subsequent commits in this reconciliation are documentation-only. Frozen results remain historical, not a current full-suite result. Evidence: `.scratch-tmp/a39-core/baseline.md`, `/dev/shm/a39-freeze/suite.md`, `/dev/shm/a39-fix-inputs/regression.md`.

## B. Findings

A38 carried items are reclassified in `A39_FINDINGS.csv`, including 8 reviewer objections. A39 confirmed fixes include model/source provenance and condition-transform binding in composition identity; requested `query_count` separated from `scored_count`; result reader skips only files that do not claim result status and refuses corrupt result-shaped JSON; truncated autopilot state returns explicit failure; availability digest includes `required_files` and member stats; frontend certification excludes development evidence. Both index regression tests pass unchanged. Final production source is `2ff698e`; full sweep and corrected release accounting are recorded below. Ledger includes A39-INC-01, documenting the checkpoint probe process violation and forensic no-data-loss conclusion.
Current ledger has 45 data rows: `CONFIRMED_FIXED` 16, `CONFIRMED_UNFIXED` 3, `BLOCKED` 7, `EXPECTED_BEHAVIOR` 5, `NOT_REPRODUCED` 5, `DOCUMENTATION_CORRECTED` 4, `INFORMATIONAL` 4, `TEST_GAP_FIXED` 1. Remaining confirmed-unfixed rows concern external credential remediation and incomplete viewport validation, not fixed A39 code defects.

## C. Scientific correctness

Independent metric recomputation on raw real ranks agrees with repository summary to `<1e-15`. Real FashionIQ asset tree and all 74,381 candidate paths were verified; ranking edge fixtures and tensor condition construction were checked. Adapter evidence tally: 1 real-asset, 1 fixture, 7 static. CSMCIR real checkpoint absent; no sm_120 validation. Details/classes in `A39_SCIENTIFIC_VALIDATION.md`.

## D. Test results

Full serial sweep on named production SHA `2ff698e5b201cbc5bd578e77d1dd7bbe952f1069`: 2247 collected; 2236 passed, 1 failed, 10 skipped, 0 errors, 3800.01s. The sole failure was `test_the_release_documents_quote_the_test_count_they_actually_ran`, due to stale 3015df4 accounting in release docs; after correcting to this sweep's scoped counts, standalone handoff guards pass. Both previously failing index tests pass unchanged in this sweep. Earlier `3015df4` sweep remains historical (2246/2233/3/10). Current required test outputs are in `/dev/shm/a39-ship/verification.md`.

## E. Frontend

Mobile Playwright timeout investigation ranks unsynchronized concurrent `/api/models` requests as primary diagnosis; cold/warm in-process latency measured 43.383s / 8.891s. This round ran **no browser session** and captured no screenshot. Harness supports desktop 1440×900 and mobile 390×844 only; brief asks four viewports, with two unsupported profiles unspecified. No WebKit or physical iPhone. See `A39_UI_VALIDATION.md`.

## Audit-process honesty

An audit probe wrote two dummy files into the real DCNet checkpoint path and then ran `rm -rf` on that path instead of using a synthetic tree. Forensic evidence confirms only those 13-byte dummies and a pre-existing empty placeholder were removed; no operator weights existed or were lost. Probes must use synthetic trees only. Full incident record: `A39_SECURITY_REVIEW.md`, `A39_FINDINGS.csv`.
## F. Coverage

A39 recomputed source matrix: 360 rows (8 FULLY_REVIEWED, 14 EXECUTION_VERIFIED, 338 NOT_REVIEWED); categories: 165 test, 121 source, 47 config, 16 vendored, 8 other, 2 asset, 1 doc. Feature inventory: 35 across 20 categories. Branch inventory: 77 (63 NOT_TESTED, 14 NOT_APPLICABLE, 0 PASS); 57 of the 63 listed branches map to existing tests, 6 do not. A39 additions are separately listed in coverage matrix; not represented as a new full source census. This differs from A38 report's 78 / 64 branch claim; A39 source recount is controlling. Method and updated rows: `A39_SOURCE_COVERAGE.csv`, `A39_FEATURE_MATRIX.csv`, `A39_BRANCH_MATRIX.csv`.

## G. Security

Gateway token confirmed in 53 external files; absent from tracked files. Provider-side rotation remains required and is not documented locally. AirKnow credential remains in two tracked vendored files / five assignment lines; unreachable from supported workflows, not default-on, and provenance-pinned. No secret value appears in reports. Operator actions and approved-patch requirement: `A39_SECURITY_REVIEW.md`, `A39_LAPTOP_HANDOFF.md`.

## H. Blockers

- Real 2.72 GB CSMCIR checkpoint absent: `REAL_CHECKPOINT_BLOCKED`.
- RTX 5070 Ti `sm_120` not exercised.
- Playwright mobile phase was not executed; no WebKit or physical iPhone.
- No browser session/WebKit/iPhone; no `sm_120`; no CSMCIR checkpoint; gateway token rotation and AirKnow remediation remain external/operator blockers.
- Gateway token rotation remains pending; AirKnow remediation requires operator approval.

## I. Verdict

**A39_LAPTOP_ACCEPTED_WITH_BLOCKERS** — full serial suite ran on the named source; both historical index regressions pass unchanged, and the sole initial failure was stale release-count text, corrected with its guard now passing. No confirmed laptop-fixable correctness defect remains. External blockers remain: absent CSMCIR checkpoint, no `sm_120` GPU, no browser validation, and pending credential rotation/AirKnow approval. Incident record discloses checkpoint-path process violation and confirms no operator weights were lost.