# REGRESSION_RESULTS.md — iterative audit

Rounds 01 and 02; each wave was followed by targeted suites and, at the end of the wave, the
complete suite. Every entry below was observed by execution, not inferred.

| Wave | Verification | Result |
|---|---|---|
| wave 1 (defects A–F) | full suite incl. heavy E2E | 1078 passed, 1 failed (queue_command.json race, flaked 1-in-N) |
| wave 2 | full suite incl. heavy E2E | 1103 passed, 0 failed |
| wave 3 | targeted: pc_launcher, http_api, gpu_records | 107 passed |
| wave 4 | targeted: invocation_proof, autopilot_cold_start, gpu_records, pc_launcher, http_api | 162 passed |
| wave 5 (test isolation) | the two previously-failing files | 74 passed; poison-injection control now passes |

## Test-isolation defect found and fixed in round 02

The full suite ended with 2 failures that both passed in isolation:

- `test_autopilot_config.py::test_offline_blocked_source_acquisition_survives_into_final_rows`
  → `expected BLOCKED_SOURCE but got BLOCKED_ENVIRONMENT`
- `test_environment_planner.py::test_group_evidence_records_channels_used`
  → `IndexError: list index out of range`

Root cause: `environment_planner._SOLVE_CACHE` is a module-global map keyed by the canonical
constraint set, never cleared between tests. A test that records a FAILED conda solve poisons
every later test whose constraints canonicalize to the same key — including tests that monkeypatch
the solver, because `_solve` consults the cache BEFORE calling the patched function.

Reproduced deterministically by injecting the poisoned key from a pytest plugin; both victims then
failed with exactly the recorded errors. Fixed by clearing the cache around every test in
`test_environment_planner.py` (autouse fixture) and at the start of the autopilot test that depends
on a fresh solve. With the same poison still injected, both tests now PASS.

No test was deselected, skipped, xfailed or reordered to obtain a green suite.
