# ROOT_CAUSE_MATRIX.md — audit round 01

Commit at audit start: `92ecbd9` + fix wave 1 (uncommitted). Coverage target: 100 first-party
production files (`SOURCE_MANIFEST.md`). Slices: A deploy (`AUDIT_A.md`), B runtime (`AUDIT_B.md`),
C models + D science (main-agent + `AUDIT_D_SCIENCE.md`), E backend/jobs (`AUDIT_E.md`).

## CONFIRMED and fixed

| ID | Sev | Root cause | Status |
|---|---|---|---|
| F1 | P0 | `main.py::_development_receipt_refusal` refused in EVERY production process (neither pytest nor dev-mode), so the authoritative proof-validating branch was unreachable and every served report was `certified=false` — the inverse of the documented intent | FIXED — mode chosen by the report's own proof; deployment validates through the verifier; opt-in is dev-only and cannot skip proof validation |
| F2 | P0 | Three possible ledger-key locations; two differing files made `ledger_hmac_key` raise while consumers degraded to `[]`/None with no reason, so the API silently served nothing and the launcher's reuse path blocked | FIXED — ambiguity now surfaces in every consumer naming BOTH paths + the one-line remedy; fail-closed kept; no preference-based resolution |
| F3 | P1 | Bundle selection folds to the anchor keeping `checkpoint_id == bundle_id`, and the strip test compared `checkpoint_id != bundle_id` (always false for a bundle), so the launcher's OWN preferred LIMN bundle was refused 409 | FIXED — identity is `bundle_id`; a complete bundle is queueable, an incomplete one still names the missing member |
| F5 | P1 | Certification cache key lacked the certification MODE, so a development receipt could be re-served for a report that later gained a proof | FIXED — mode is part of the key |
| F6 | P1 | Shutdown rewrote queued/preparing jobs to `cancelled` without persisting the cause, indistinguishable from an operator cancellation | FIXED — cause persisted |
| F7 | P1 | `enqueue` ignored `_stopping`, so `shutdown()` then `enqueue` left a job `queued` forever with a dead worker | FIXED — explicit `QueueShutdownError`; the API maps it to 503 |
| F8 | P1 | `queue_command.json` was written AFTER the terminal status became observable, so a consumer reading the record on seeing `completed` could read an empty file (flaked 1-in-N in the full suite) | FIXED — record written atomically before the terminal state is published |
| A1 | P0 | `pc.sh` re-declared `local mismatch_reasons=()`, wiping a reason appended earlier, so a fail-closed message could lose its cause | FIXED |
| A3 | P1 | `release_verify.sh` unpacked `sys.argv[1:2]` into one name, raising on ANY accepted flag | FIXED |
| A8 | P1 | `set -uo pipefail` without `set -e`, and the report block's status unchecked → a crash inside it exited 0 with NO report | FIXED |
| A4 | P1 | `pc.sh` could never satisfy its in-container GPU/adaptation requirement on a toolkit-less host, and the remedy named dataset paths for a GPU/toolkit problem, after two startups | FIXED |
| A7 | P1 (policy) | A report whose proof was pruned was `certified=false`, so the launcher's dedup re-ran an expensive completed evaluation | FIXED — distinct named state; launcher reuses only ledger-corroborated runs |

## FALSE_POSITIVE / EXPECTED_BEHAVIOR / KNOWN_LIMITATION

| ID | Classification | Why |
|---|---|---|
| A2 | FALSE_POSITIVE on this host | Claimed `release_verify.sh` J1 fails because `.venv` lacks `yaml`/`fastapi`; measured live: **J1 PASS** (16 models, boot state named), exit 0 |
| A9 | FALSE_POSITIVE | Claimed a destination dimension is permanently PENDING; `DEPLOYMENT`/`CONFIGURATION` reach PASS live |
| A6 | FIXED_AND_VERIFIED | The API false-FAIL was real at 23:01 and closed by the F1 fix; both revisions were tested |
| A5 | CONFIRMED_OUTSTANDING (see B-1) | Host and container derive different capability keys because `_gpu_capability_key` reads only the env var while `verify_environment` also adopts `environment_assignment.json` |
| B-1 | CONFIRMED_OUTSTANDING | Same as A5: a record written with the env var set is unreadable by a reader that resolves the interpreter from the persisted assignment → false FAIL on the target PC |
| B-2 | CONFIRMED_OUTSTANDING | `_cached_gpu_capability(authoritative=False)` never compares the interpreter fingerprint, so an advisory read can return `GPU_RUNTIME_READY` on a drifted environment |
| B-4/B-6/B-9/B-11 | CONFIRMED_OUTSTANDING (P3/P2) | In-memory capability map not `probe_source`-filtered; the in-memory map is not filtered; a hop reads a file for the adaptation; a few consumers do not name the key condition |
| F4 | CONFIRMED_OUTSTANDING (P1) | `_proof_stamp == "absent"` plus the development opt-in certifies a report with NO proof |
| F8-release | PROBABLE (P1) | `release_verify.sh` uses a substring proxy for proof presence, so `MODEL_INFERENCE` can be PASS from a refuted run |
| D-F1 | CONFIRMED_OUTSTANDING | For an UNREGISTERED model, `metric_extraction_available=True` with `observed_metrics=None` is accepted — a report carrying no numbers still certifies |
| LIMN per-query Top-K | KNOWN_LIMITATION | No per-query export exists; documented in `PER_QUERY_EXPORT_STATUS.md` and the guide. Nothing fabricates per-query data |
| Target GPU / sm_120 / dataset | EXTERNAL_DEPENDENCY_PENDING | No RTX 5070 Ti and no `nvidia-container-toolkit` on this host |
