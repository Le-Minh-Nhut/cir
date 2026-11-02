# A39 Evidence Index

| Claim / report area | Primary evidence path | Specific evidence |
|---|---|---|
| Baseline branch, SHA, initial worktree | `.scratch-tmp/a39-core/baseline.md` | Lines 1–23, 72–95 |
| Recomputed source/feature/branch counts and mappings | `.scratch-tmp/a39-core/coverage-recount.md` | Lines 3–53, 55–137 |
| A38 carried findings and eight objections | `.scratch-tmp/a39-core/a38-reconciliation.md` | Lines 14–53 |
| Frozen suite command, counts, failures | `.scratch-tmp/a39-archive/a39-freeze/suite.md` | Lines 1–137; literal output `.scratch-tmp/a39-archive/a39-freeze/suite-output.txt` |
| Frozen source identity | `.scratch-tmp/a39-archive/a39-freeze/frozen-sha.txt`; `.scratch-tmp/a39-archive/a39-freeze/freeze.md` | Frozen SHA and audit freeze state |
| Vite build result | `.scratch-tmp/a39-archive/a39-freeze/build.md` | Command and success result |
| Composition provenance defect reproduction | `.scratch-tmp/a39-id/composition-identity.md` | Lines 35–93 |
| Composition provenance regression | `.scratch-tmp/a39-fix-id/regression.md` | Lines 6–16, 154–198 |
| Export, gate, index, autopilot, availability fixes | `.scratch-tmp/a39-fix-rest/fixes.md` | Lines 1–31 |
| A39 targeted regression counts | `.scratch-tmp/a39-fix-rest/regression.md` | Lines 1–24 |
| Security exposure and redacted file/line/count evidence | `.scratch-tmp/a39-sec/security-review.md` | Lines 11–86, 90–176 |
| Security operator actions | `.scratch-tmp/a39-sec/operator-actions.md` | Lines 9–84 |
| Metric recomputation, tensors, real dataset, ranking | `.scratch-tmp/a39-sci/scientific-validation.md` | Lines 5–168 |
| Nine-adapter evidence class tally | `.scratch-tmp/a39-sci/adapter-matrix.md` | Lines 1–19 |
| CSMCIR checkpoint blocker | `.scratch-tmp/a39-sci/csmcir-investigation.md`; `.scratch-tmp/a39-core/open-questions.md` | Checkpoint absent and exact open question |
| Pipeline resume and targeted results | `.scratch-tmp/a39-pipe/pipeline-resume.md` | Lines 1–126 |
| Mobile timeout diagnosis, latency, viewports | `.scratch-tmp/a39-pipe/playwright-timeout.md` | Lines 1–96 |
| API route inventory | `.scratch-tmp/a39-api/routes.md` | Route reference; count not repeated unless sourced there |
| API targeted tests | `.scratch-tmp/a39-api/api-tests.md` | Recorded API test outcomes |
| Consolidated discovery findings | `.scratch-tmp/a39-discover/findings.md` | DISC-01 through DISC-08 |
| Static/adversarial/cross-module discovery | `.scratch-tmp/a39-discover/pass-a-static.md`; `.scratch-tmp/a39-discover/pass-b-adversarial.md`; `.scratch-tmp/a39-discover/pass-c-crossmodule.md` | Each finding's methods and reproductions |
| Coverage inventory and A39 row additions | `A39_SOURCE_COVERAGE.csv`; `A39_FEATURE_MATRIX.csv`; `A39_BRANCH_MATRIX.csv` | Derived from recount and lane evidence listed above |
| Current reconciliation status at report authoring | Agent report from `a39-reconcile2` | Work in progress; no commits yet; no updated verification evidence available |

All eleven reports are tied to available evidence above. No secrets are transcribed; credential evidence uses only redacted path/line/count descriptions.

The `/dev/shm/a39-*` evidence paths cited in these reports were tmpfs scratch. They were archived to `.scratch-tmp/a39-archive/a39-*/` (same filenames) during a disk cleanup; the `/dev/shm` originals no longer exist. Paths under `.scratch-tmp/a39-*` (non-archive) remain in place.