# ROOT_CAUSE_MATRIX.md — audit round 02

Source: post-wave-3 tree. Slices: A deploy (`AUDIT_A.md`), B runtime (`AUDIT_B.md`),
C+D science (`AUDIT_D.md`), E backend/frontend (`AUDIT_E.md`). Four independent agents.

## CONFIRMED and fixed in wave 4

| ID | Sev | Root cause | Status |
|---|---|---|---|
| B-1/A5 | P1 | `_gpu_capability_key` resolved the interpreter ONLY from `os.environ[python_env_var]`, while `verify_environment` ALSO adopts the interpreter persisted in `environment_assignment.json` → the same environment produced two different capability keys (measured `54b83730…` vs `6b6c0b63…`), so a record written one way was unreadable the other way. On the target PC that pins `GPU_RUNTIME` to PENDING for a genuinely verified host | FIXED — one shared `resolve_model_interpreter` used by both; bidirectional read proven |
| B-2 | P2 | the advisory `_cached_gpu_capability` lookup omitted the interpreter-fingerprint comparison, so a drifted environment returned `GPU_RUNTIME_READY` | FIXED — zero-length/e falsy fingerprint now REFUSES instead of skipping after the primary R2 deletion. |
| B-4 | P3 | the in-memory capability map was not `probe_source`-filtered, so a synthetic record could serve after the seam was unset | FIXED |
| B-11 | P3 | a falsy guard skipped the fingerprint check when the record carried none | FIXED |
| D-F1 | P1 | for an UNREGISTERED model, `metric_extraction_available=True` with `observed_metrics=None` produced no reasons — a report carrying no numbers certified as research evidence | FIXED — extraction claim now requires at least one usable value |
| AP-1 | P1 | `autopilot.py` treated a ledger that was MISSING or whose HMAC chain did NOT verify (`None`) as "nothing to check" and certified the run `OFFICIAL_VERIFIED` | FIXED — `None` means "cannot corroborate"; refusal is named |
| F8-release | P1 | `release_verify.sh` used a negative substring proxy for proof presence, awarding `MODEL_INFERENCE: PASS` to refuted runs | FIXED — authoritative validator + non-metric corroboration reasons |
| D-R2-01 | P2 | `_category_reasons(required=False)` tolerated only an EMPTY dict and otherwise demanded LIMN's three categories, so a non-LIMN model with a legitimate partial table was refused (false FAIL) | FIXED — validate only the categories carried; LIMN still requires all three |
| D-R2-02 | P3 | a `LedgerKeyError` was appended to `reasons` twice | FIXED |
| A9 | P2 | J8 read `/api/jobs`; a clean restart leaves the in-memory queue empty, forcing mandatory `OPERATIONS` to PENDING and degrading a healthy finished host to `ENGINEERING_INCOMPLETE` (false FAIL) | FIXED — empty queue is the normal idle state; a running job is still strictly validated |
| Verdict masking | P1 | `--offline` reported `GPU_VALIDATION_PENDING` while `DEPLOYMENT`/`OPERATIONS` were PENDING for a deployment reason | FIXED — the verdict now names the true cause (`ENGINEERING_INCOMPLETE`) |
| F10-R2 / UI-1 | P1/P2 | the frontend ignored the `verification` object, so an uncertified / `proof_unavailable` / `uncorroborated` run could render as a clean pass, and an API error from `/api/benchmark-runs` was swallowed into an empty table with no reason | FIXED — distinct certification labels; `proof_unavailable` distinguishable from `uncorroborated`; API errors surface |

## Verdict on the round-01 "wave 3" changes (audited, no regression found)

- The `proof_unavailable` reuse path in `pc.sh` is restricted to runs the HMAC ledger corroborates
  (`return_code == 0`, `INVOCATION_COMPLETED`) with a missing proof file as the SOLE refusal reason.
- The cause-specific remedies in `pc.sh` name matching causes (GPU/toolkit, adaptation, mounts,
  environment) and still fail closed; the single-startup and active-job invariants hold.
- The `release_verify.sh` report status is now checked; a crash inside report generation exits
  non-zero instead of exiting 0 with no report.

## FALSE_POSITIVE / EXPECTED_BEHAVIOR / KNOWN_LIMITATION

| ID | Classification | Why |
|---|---|---|
| "Active `os.remove` on unreadable artifacts" | EXPECTED_BEHAVIOR | Deliberate and documented: an unreadable 0-byte leftover carries no evidence, is named in the report, and removing it stops the same condition recurring on every later scan. Readable records are never touched |
| "`real_gpu=True` masked by UNREADABLE/WRONG_KEY" | FALSE_POSITIVE | The report carries `real_gpu` as its own field alongside the scan status; the scan condition and the GPU verdict are separate, which is the documented shape |
| Remedy text for `WORKBENCH_LEDGER_KEY` | PARTIALLY CONFIRMED | The env var takes a raw secret, not a path; the remedy line says "set `WORKBENCH_LEDGER_KEY` explicitly to choose one" without claiming it is a path. Not a defect, but the guide states the raw-secret form |
| LIMN per-query Top-K | KNOWN_LIMITATION | No per-query export; documented in `PER_QUERY_EXPORT_STATUS.md` and guide §15. Nothing fabricates per-query data |
| Target GPU / sm_120 / toolkit / dataset | EXTERNAL_DEPENDENCY_PENDING | No RTX 5070 Ti, no `nvidia-container-toolkit` on this host |
| Dual host ledger key on THIS machine | ENVIRONMENT, not product | Two differing key files exist in the dev environment; the product now surfaces it by name in every consumer (that was the F2 fix). The operator remedy is to keep one file or set `WORKBENCH_LEDGER_KEY` |
