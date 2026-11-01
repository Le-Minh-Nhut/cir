# A39 Test Results

## Frozen full suite

Frozen source: `3015df489a3e3a1d7db47d606d1fe465c12ddfe8` (`/dev/shm/a39-freeze/frozen-sha.txt`). Literal command:

```bash
export TMPDIR=/dev/shm/a39-freeze; /home/heheboiz/data/cir/.venv/bin/python -m pytest workbench/tests tests -q -p no:randomly 2>&1 | tee /dev/shm/a39-freeze/suite-output.txt | tail -40
```

Literal summary: `3 failed, 2233 passed, 10 skipped, 253 warnings in 3932.25s (1:05:32)`; 2246 collected; 0 errors.

### Failures (literal assertion/error text)

1. `workbench/tests/test_adversarial_orchestration.py:140`, `test_d01_d03_d04_index_integrity`: `E       Failed: DID NOT RAISE Exception` on frozen `3015df4`. Historical failure was a real regression: result-shaped or malformed-object JSON must be refused, not skipped. Fixed by the index contract split at `4ec5985`; unchanged test passes at current HEAD (covered in the targeted 30-test index suite evidence in `/dev/shm/a39-reconcile2/evidence.md`).
2. `workbench/tests/test_handoff_commands.py:485`, `test_no_release_document_points_at_a_superseded_source_sha`: frozen `3015df4` failure named `3786e1d` while production had advanced. The release source is now advanced to `2ff698e`; the production-tree equality predicate itself passes. Its sibling count guard rejects an honest record because this SHA's collection (2188 `workbench/tests`) has no post-fix full-suite outcome to supply a pass/fail/skip triple. Do not fabricate one; see `/dev/shm/a39-ship/verification.md`.
3. `workbench/tests/test_strict_acceptance_matrix.py:251`, `test_27_to_29_index_separation_and_validation`: `E       Failed: DID NOT RAISE Exception` on frozen `3015df4`. Historical failure was a real regression: an object declaring result fields but failing validation is corrupt evidence and must be refused. Fixed by the index contract split at `4ec5985`; unchanged test passes at current HEAD (covered in the targeted 30-test index suite evidence in `/dev/shm/a39-reconcile2/evidence.md`).

The `3015df4` sweep is historical and remains `2246 collected, 2233 passed, 3 failed, 10 skipped, 0 errors`. The two index regressions and release-SHA production-tree mismatch were subsequently fixed; the FULL suite was not rerun after those fixes. Current targeted verification results are not a replacement full-suite claim.

## Index contract clarification

The reader skips/logs files that do not claim to be result artifacts (no `schema_version`, `run`, or `queries` field); it refuses malformed object-shaped JSON and validation failures for objects claiming result fields. Thus not all malformed JSON is skipped. The stray-JSON regression remains `1 passed`.

Current full serial sweep ran at named source `2ff698e5b201cbc5bd578e77d1dd7bbe952f1069`: combined `2247 collected, 2236 passed, 1 failed, 10 skipped, 0 errors, 3800.01s`. Its sole failure was the release count guard against stale docs; docs were corrected to the sweep's actual counts, after which the required handoff command passed (`8 passed in 9.24s`). `workbench/tests` collected 2188 (2177 passed, 1 failed, 10 skipped); `tests/unit` added 59 passes. Production-tree equality against named source is rc=0. No product test failed; the one count-guard failure was documentation-only. The two formerly failing index tests pass unchanged in the full sweep. Frozen `3015df4` result remains historical. Full output: `/dev/shm/a39-ship/final-sweep.txt`.


## Targeted regressions (literal recorded counts)

- Composition provenance: `pytest workbench/tests/test_composition_provenance_identity.py` → `5 passed in 1.14s` after fix (before: `3 failed, 2 passed in 1.14s`).
- Composition image identity: `6 passed in 0.32s`.
- Execution identity: `7 passed in 178.61s`.
- D1 export query count: `1 passed in 0.41s`; existing cohort API: `1 passed, 44 deselected, 4 warnings in 0.42s`.
- D2 certification predicate: `1 passed in 0.61s`.
- D3 stray index JSON: `1 passed, 4 warnings in 40.37s`.
- D4 autopilot truncated state: `1 passed in 0.14s`.
- D5 availability digest: `1 passed in 0.05s`.
- Existing directly affected tests (`test_frontend_staleness.py`, `test_http_api_layer.py`, `test_result_operations.py`, `test_autopilot_config.py`, `test_artifact_hashing.py`): `82 passed, 176 warnings in 270.71s`.
- API targeted suite (`TMPDIR=/dev/shm/a39-api PYTHONPATH=. /home/heheboiz/data/cir/.venv/bin/pytest workbench/tests/test_http_api_layer.py workbench/tests/test_result_operations.py workbench/tests/test_index_workspace_configuration.py workbench/tests/test_backend_config_roots.py workbench/tests/test_authorization_policy.py`): `68 passed, 176 warnings in 11.62s` (`.scratch-tmp/a39-api/api-tests.md`).
- Frontend build: `export TMPDIR=/dev/shm/a39-freeze; npx vite build` → success (`vite v8.3.2`, built in 181ms).

## Not run / limits

No tests rerun for report writing; full suite was already run. No GPU work. No Playwright browser session, no WebKit, no physical iPhone. Real CSMCIR checkpoint evaluation blocked because 2.72 GB checkpoint is absent. A39 freeze suite reflects SHA `3015df489a3e3a1d7db47d606d1fe465c12ddfe8`; any later reconciliation commit must not be described as suite-tested unless its evidence establishes that.