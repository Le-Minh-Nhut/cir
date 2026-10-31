# A38 Evidence Index

This document maps every finding from `workbench/docs/reviews/a38/A38_FINDINGS.csv` (21 data rows) to its supporting evidentiary artifact, verification type, and reproduction procedure.

## Findings Evidence Map

| Finding ID | Severity | Status | Feature / Description | Evidence Artifact | Type | Reproduction Note |
|---|---|---|---|---|---|---|
| `A38-01` | LOW | CONFIRMED_FIXED | Pipeline command-fingerprint collision | `workbench/docs/reviews/a38/A38_INDEPENDENT_REVIEW.md` (R1 section) | executed | Replay stage `sMODEL:abc` vs `s` with model `abc`; verify collision under old hasher and distinct digests under new framing helper. |
| `A38-02` | MEDIUM | CONFIRMED_FIXED | Multi-command stage output-validation test gap | `workbench/tests/test_pipeline.py:307-380` | executed | Run `pytest workbench/tests/test_pipeline.py::test_multi_command_output_validation_resume` covering 8 cases (success, failures, corrupt JSON, stale outputs). |
| `A38-03` | MEDIUM | CONFIRMED_FIXED | PC resume instructions omitted opt-in --resume | `workbench/docs/reviews/a38/A38_PC_HANDOFF.md` (step 9); `A38_DOC_ERRATUM.md` | static | Inspect `composition_eval.py --help` confirming `--resume` is opt-in; check step 9 in `A38_PC_HANDOFF.md`. |
| `A38-04` | LOW | EXPECTED_BEHAVIOR | GPU capability workspace isolation | `workbench/docs/reviews/a38/A38_FINDINGS.csv` row 4; director re-derivation stdout | executed | Probe workspace B (empty) vs A (record); verify CLI autopilot propagates configured root via `WORKBENCH_ARTIFACTS_ROOT`. |
| `A38-05` | LOW | CONFIRMED_FIXED | Release metadata and PC handoff documentation drift | `workbench/docs/reviews/a38/A38_DOC_DRIFT.csv`; `workbench/docs/PC_PILOT_HANDOFF.md` | static | Compare `PC_PILOT_HANDOFF.md` observer list and export reasons against source code in `runtime.py`. |
| `C-01` | HIGH | CONFIRMED_FIXED | Rank cache accepts malformed payloads | `workbench/docs/reviews/a38/A38_FINDINGS_PARTIAL_W4.md` (#6); `runner.py:67-73` | executed | Run `test_rank_validation_catches_old_permissive_guard` or supply malformed payload (string/missing qid/rank 9999) to `run_composition(..., resume=True)`. |
| `C-03` | HIGH | CONFIRMED_FIXED | Concurrent feature memmap writers interleave | `workbench/docs/reviews/a38/A38_FINDINGS_PARTIAL_W4.md` (#16); `A38_INDEPENDENT_REVIEW.md` | executed | Concurrently write 8 MB memmap without lock to observe mixed `[1.0, 2.0]` values; run with `fcntl.flock` to verify exclusion. |
| `R-A38-01` | HIGH | CONFIRMED_FIXED | MappingProxyType regression in registry caching | `workbench/backend/registry.py:110,165,222`; `prepare_model.py:507` | executed | Pass cached `MappingProxyType` to `prepare_model.py:507` `resize_is_complete` -> raises `TypeError` on `json.dumps`. Verify `copy.deepcopy` fixes it. |
| `S-01` | HIGH | CONFIRMED_FIXED | CSMCIR checkpoint incompatibility silently discarded | `workbench/docs/reviews/a38/A38_SCIENTIFIC_VALIDATION.md` (T5); `model_csmcir.py:185-201` | executed | Run `pytest workbench/tests/test_csmcir_checkpoint.py` with synthetic checkpoint missing or adding unexpected keys. |
| `T-01` | INFO | NOT_REPRODUCED | Runtime adaptation config setting rejected | `workbench/docs/reviews/a38/A38_FINDINGS.csv` row 10; frozen-tree check stdout | executed | Run `test_autopilot_e2e.py::test_a_dry_run_changes_nothing` with `WORKBENCH_RUNTIME_ADAPTATION=sm_120`; verify permitted keys list. |
| `T-A38-01` | HIGH | CONFIRMED_FIXED | Release invariants satisfied by comments | `workbench/docs/reviews/a38/A38_INDEPENDENT_REVIEW.md` (R4); `test_release_invariants.py` | executed | Mutate target files inserting invariant tokens only into `#` comments; verify AST and `_shell_code` checkers reject spoofed tokens. |
| `UI-01` | MEDIUM | CONFIRMED_FIXED | `/api/models` latency | `workbench/docs/reviews/a38/A38_UI_VALIDATION.md`; `main.py:159-182` | executed | Execute `curl -w '%{time_total}' http://127.0.0.1:8000/api/models` cold vs warm; profile `models()` handler call count for `yaml.safe_load`. |
| `B-01` | LOW | INFORMATIONAL | API error-envelope inconsistency | `workbench/docs/reviews/a38/A38_UI_VALIDATION.md`; `A38_FINDINGS.csv` row 13 | executed | Curl `GET /api/queries?run_id=missing` (HTTP 400 `{error}`), `GET /api/cohorts/nope/export/json` (HTTP 404 `{detail:{error}}`), `GET /api/nope` (HTTP 404 `{detail:'Not Found'}`). |
| `C-02` | INFO | INFORMATIONAL | Optional composition checkpoint identity for library callers | `workbench/docs/reviews/a38/A38_FINDINGS_PARTIAL_W4.md`; `runner.py:25-34` | static | Inspect `ModelSpec` default `checkpoint_identity` (empty string) vs CLI caller `composition_eval.py:163-176` passing SHA256. |
| `C-04` | INFO | INFORMATIONAL | Composition identity omits model-source revision | `workbench/docs/reviews/a38/A38_SCIENTIFIC_VALIDATION.md` (T6) | static | Inspect `runner.py:634-643`: run identity fingerprints adapter code hashes but omits vendored model source git commit hash. |
| `E-01` | INFO | INFORMATIONAL | Environment planning and provisioning | `workbench/docs/reviews/a38/A38_FINDINGS_PARTIAL_W1.csv` row 1 | static | Inspect `environment_planner.py` constraint matching and `manage_environment.py --json` output structure. |
| `T-02` | INFO | BLOCKED | Autopilot apply-mode summary absent | `workbench/docs/reviews/a38/A38_TEST_RESULTS.md`; `A38_FINDINGS.csv` row 17 | blocked | Run `pytest workbench/tests/test_autopilot_e2e.py::test_no_model_reports_a_fabricated_metric`; observe missing summary; capture subprocess logs. |
| `B-02` | INFO | INFORMATIONAL | Backend job lifecycle | `workbench/docs/reviews/a38/A38_FINDINGS_PARTIAL_W8.md` | executed | Run queue probes exercising submit, cancel, natural exit, failure code 7, 5 concurrent enqueues, and restart recovery. |
| `W8-01` | INFO | EXPECTED_BEHAVIOR | Cross-module concurrency | `workbench/docs/reviews/a38/A38_FINDINGS_PARTIAL_W8.md` | executed | Run synchronized probes on DuckDB index rebuild, `run_stages` flock lock contention (`WorkflowLockError`), and GPU capability atomic file writes. |
| `G-01` | N/A | BLOCKED | No sm_120 execution, capability record, or inference | `workbench/artifacts/gpu_capabilities/`; `runtime.py:2470-2493` | blocked | Census of `workbench/artifacts/gpu_capabilities/` shows zero records for `sm_120`; runtime gate enforces check. |
| `X-01` | MEDIUM | CONFIRMED_UNFIXED | security | `workbench/third_party/AirKnow/lavis/models/Llm_online/gpt.py`; `workbench/third_party/AirKnow/lavis/models/Llm_online/MLLM.py` | static | Run `git grep -I -l -E '(sk-[A-Za-z0-9]{16,})' -- workbench/third_party/AirKnow` (file list only, no values). Pre-existing upstream credential introduced in `5e6e284`. |

## Durable Regression Tests Added This Round

The following durable regression tests were added or hardened during the A38 audit round:
- **`workbench/tests/test_pipeline.py`**:
  - `test_fingerprint_fields_are_unambiguously_framed`: Proves distinct digests for boundary pairs and stage name / prefix aliasing (`A38-01`).
  - `test_multi_command_output_validation_resume`: Exercises 8 distinct execution cases including success resume, command failures, corrupt JSON, stale outputs, and interrupted recovery (`A38-02`).
- **`workbench/tests/test_composition_image_identity.py`**:
  - `test_rank_validation_catches_old_permissive_guard`: Verifies rejection of malformed payloads (strings, missing query IDs, out-of-range ranks) (`C-01`).
- **`workbench/tests/test_csmcir_checkpoint.py`**:
  - `test_csmcir_checkpoint_missing_keys_fails_and_old_discard_path_does_not`: Proves missing keys raise `ValueError` (`S-01`).
  - `test_csmcir_checkpoint_report_rejects_missing_keys`: Verifies error report generation on missing keys (`S-01`).
  - `test_csmcir_checkpoint_report_accepts_empty_report`: Verifies clean checkpoint pass-through (`S-01`).
- **`workbench/tests/test_release_invariants.py`**:
  - `test_i11_an_invalid_mount_cannot_reach_evaluation` (AST/quote-aware scanning)
  - `test_i13_no_unverified_gpu_record_produces_gpu_runtime_pass`
  - `test_i14_no_synthetic_inference_produces_model_inference_pass`
  - `test_i17_an_api_error_never_becomes_an_empty_success`
  - `test_i19_a_harness_does_not_pass_without_its_infrastructure`
  - `test_i20_a_verdict_never_claims_more_than_its_evidence`
  Guards release invariants against comment-only token spoofing (`T-A38-01`).

## Volatile Evidence Locations

The following evidentiary artifacts were generated in tmpfs (`/dev/shm` or `/tmp`) to respect host disk constraints (100% capacity, ~954 MiB remaining). These files will vanish on reboot or cleanup:
- `/dev/shm/a38-w3/`: Ranking, condition input, and metric probes (`rank_probe.py`, `metric_probe.py`).
- `/dev/shm/a38-w4/`: Rank cache poisoning, atomic JSON race, and concurrent memmap interleaving test scripts (`focused.py`).
- `/dev/shm/a38-w5/`, `/dev/shm/a38-w5b/`: UI matrix screenshots (40 viewport PNGs) and Playwright trace logs (`measurements.json`).
- `/dev/shm/a38-w6/logs/`: Test collection output (`collect-final.txt`), chunk partitions (`chunk1.txt`..`chunk4.txt`), and chunk 1 logs. Authoritative accounting in `A38_TEST_RESULTS.md`.
- `/dev/shm/a38-w8/`: Job queue lifecycle probes (`queue_probe.py`, `cancel_natural.py`) and concurrency probes (`concurrency_probe.py`, `pipeline_probe.py`).
- `/dev/shm/a38-rep/`: Temporary audit report drafting directory (cleaned at conclusion).
- `/tmp/pw-browsers`: Playwright Chromium binaries used for UI validation.
