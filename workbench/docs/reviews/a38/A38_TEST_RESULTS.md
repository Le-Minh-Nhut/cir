## HEAD at start / at final accounting
Start and final HEAD: `a8af3d2df45cb6d0aff8775cd3402f8c97b742cf`. Tree DIRTY. Frozen status: 10 modified tracked files (`workbench/backend/main.py`, `registry.py`, `runtime.py`, `workbench/composition/model_csmcir.py`, `runner.py`, `workbench/docs/PC_PILOT_HANDOFF.md`, `workbench/scripts/pipeline.py`, tests `test_composition_image_identity.py`, `test_pipeline.py`, `test_release_invariants.py`) and 5 untracked paths (operator files `pc_composition_smoke.yaml`, `src/train_cirr.py`, `uv.lock`, review directory, and `workbench/tests/test_csmcir_checkpoint.py`). `git diff --stat`: 10 files changed, 469 insertions(+), 85 deletions(-). Dirty-tree accounting.

Environment: Python 3.13.12; `nproc` 16; torch 2.14.1+cu130.

## collect-only total + collection errors
`2175 tests collected in 7.95s`; no collection errors. Command: `PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TMPDIR=/dev/shm .venv/bin/python -m pytest -q --collect-only workbench/tests --basetemp=/dev/shm/a38-w6t -p no:cacheprovider`. Raw: `/dev/shm/a38-w6/logs/collect-final.txt`.

## partition proof (union ok?, intersection empty?)
149 test files; chunks contain 38, 37, 37, 37 files. Round-robin proof: union ok `True`; intersection empty `True`. Lists/proof at `/dev/shm/a38-w6/all_tests.txt`, `/dev/shm/a38-w6/chunk{1..4}-final.txt`, `/dev/shm/a38-w6/partition-final.txt`.

## per-chunk table (files | collected | passed | failed | skipped | errors | duration)
| Chunk | Files | Collected | Passed | Failed | Skipped | Errors | Duration |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 38 | 631 | 629 | 0 | 2 | 0 | 1126.58s |
| 2 | 37 | 490 | 489 | 0 | 1 | 0 | 838.03s |
| 3 | 37 | 580 | 579 | 0 | 1 | 0 | 1394.71s |
| 4 | 37 | 474 | 467 | 1 | 6 | 0 | 1263.39s |

Exact commands used each chunk (paths are the exact newline-delimited contents of each chunk list):
`PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TMPDIR=/dev/shm .venv/bin/python -m pytest -q <paths in /dev/shm/a38-w6/chunkN-final.txt> --basetemp=/dev/shm/a38-w6t -p no:cacheprovider`
Raw logs: `/dev/shm/a38-w6/logs/chunk{1..4}-final.txt`.

## aggregate (sum + explicit caveat)
PER-CHUNK SUM, not a single-process run: 2164 passed, 1 failed, 10 skipped, 0 errors of 2175 collected. Chunks ran sequentially against same frozen dirty tree. Chunk 1 includes the four autopilot apply-with-missing-summary cases and the autopilot dry-run test; all passed. No transient registry NameError reproduced.

## failures diagnosed (each: test, cause, real-defect? flaky?)
- Only failure: chunk 4 `workbench/tests/test_handoff_commands.py::test_no_release_document_points_at_a_superseded_source_sha`. Expected dirty-tree guard failure. It explicitly compares tested source SHA `27554b2ac731c40385cbf72daf06d5ed5a0a24d7` to current production files and refuses because uncommitted changes exist. Not a product defect. Observed once; no flakiness rerun.
- All 10 registry-related failures from the earlier mid-edit chunk1 were transient; rerun of those 10 after registry edit settled passed `10 passed in 337.09s` (`registry-rerun.txt`). Earlier chunk1 was void and excluded from final totals.
- `test_autopilot_e2e.py::test_a_dry_run_changes_nothing`: passed in final chunk1; prior mid-edit failure was not reproduced.
- Four apply-with-missing-summary tests passed in final chunk1; no persistent failure.

## test-quality: mutation-survives list
Scratch mutation demos under `/dev/shm/a38-w6/mut/` previously showed comment-satisfiable release invariant checks I11/I14/I17/I19/I20. Upstream changed `test_release_invariants.py` to comment-insensitive AST/source checks. Final chunk1 included this file and passed. Earlier focused re-run also observed 22 passed. Pipeline boundary test injects old fingerprint algorithm and asserts collision is rejected; composition rank test injects permissive guard and asserts rejection. API negative request cases and GPU empty/cross-workspace refusal cases exist. No tests modified by W6.

## tests strengthened (diff summary)
None by W6; upstream release-invariant test hardening is present in final tree.

## files written
`workbench/docs/reviews/a38/A38_TEST_RESULTS.md`; `workbench/docs/reviews/a38/A38_FINDINGS_PARTIAL_W6.csv`. No commits.