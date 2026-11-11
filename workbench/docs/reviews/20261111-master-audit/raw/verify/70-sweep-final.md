# 70 — Authoritative final serial sweep at `c755673`

This is the authoritative sweep for the release guard, because production changed after the previous
authoritative sweep at `cd980dc`.

## Identity

| fact | value |
|---|---|
| HEAD at sweep start | `8230358d1a2f71f769834da06e5646846363a98f` (branch tip at start; verified equal to `git rev-parse research/cir-failure-workbench`) |
| HEAD at sweep end | `c755673487744876d12fdd5a676a220329bd3468` |
| HEAD moved during the run | **yes** — `c755673` (`docs(audit): promote acc9b20e to canonical ENCODER run; M-01-a verified`), committed `2026-11-11 11:41:30 +0700`, sweep started `2026-11-11T11:38:50+07:00` (2 min 40 s in) |
| branch | `research/cir-failure-workbench` |
| tree at start / end | `?? CLEANUP_REPORT.md.save` only (unchanged) |
| interpreter | `workbench/envs/ctl312h/bin/python` |
| wall time | 3562.27 s (0:59:22) |

### What the mid-run commit `c755673` changed

Documentation only, four files under `workbench/docs/reviews/20261111-master-audit/`:

```
 .../20261111-master-audit/CENSUS_20261111.vi.md    | 80 +++++++++++++++-------
 .../20261111-master-audit/FINAL_REPORT.vi.md       | 39 +++++++++--
 .../20261111-master-audit/INDEPENDENT_REVIEW_RESPONSE_20261111.vi.md | 42 ++++++++++----
 .../20261111-master-audit/MODEL_LIMN_ENCODER_VERIFICATION.vi.md     | 34 ++++++++---
 4 files changed, 141 insertions(+), 54 deletions(-)
```

**The production tree is byte-identical between start and end revisions:**

```
$ git diff --stat 8230358 c755673 -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'
(empty)
```

No test file changed either. The four edited documents are not in `test_handoff_commands.py`'s
`RELEASE_DOCS` (a fixed 5-file tuple: `PC_PILOT_HANDOFF.md`, `PC_ONE_COMMAND_GUIDE.vi.md`,
`reviews/FINAL_PC_RELEASE_VERDICT.md`, `reviews/FINAL_DEFECT_LEDGER.md`, `AUTOMATION_RUNBOOK.md`), and
the only documentation glob in the suite is `workbench/docs/*.md` (top level, `test_handoff_commands.py:206`)
which does not recurse into `workbench/docs/reviews/`. Grep for the four filenames across `workbench/tests`
and `workbench/*.py` returns nothing.

Both `8230358` and `c755673` have the production tree that is on disk:

```
$ git diff --quiet c755673 -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'; echo $?   → 0
$ git diff --quiet 8230358 -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'; echo $?   → 0
```

This sweep therefore governs both revisions equally on every production and test path.

## Production delta since the previous sweep at `cd980dc`

Exactly the two files named in the brief:

```
$ git diff --stat cd980dc c755673 -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'
 workbench/backend/parity_gate.py       | 30 +++++++++---
 workbench/composition/model_encoder.py | 87 +++++++++++++++++++++++-----------
 2 files changed, 82 insertions(+), 35 deletions(-)
```

Test-side delta over the same span (+13 collected tests):

```
$ git diff --stat cd980dc c755673 -- workbench/tests
 workbench/tests/test_encoder_relation_handshake.py | 122 +++++++++++++++++++++
 workbench/tests/test_gate_contract_negatives.py    |  81 ++++++++++++++
 workbench/tests/test_handoff_commands.py           |  34 +++++-
 3 files changed, 232 insertions(+), 5 deletions(-)
```

## Preflight

| check | result |
|---|---|
| `git rev-parse HEAD` | `8230358d1a2f71f769834da06e5646846363a98f`, equal to the branch tip — PASS (no specific sha required) |
| `git status --porcelain` | exactly `?? CLEANUP_REPORT.md.save` — PASS |
| `pgrep -af pytest` | empty — PASS (the earlier GPU rerun had finished) |
| `nvidia-smi --query-compute-apps=pid,process_name --format=csv` | header only (`pid, process_name`), no compute process — PASS |
| `df -h /` | 99 GiB avail ≥ 45 — PASS |
| `/tmp/sweep_basetemp_final2` | absent — PASS |

## Command (verbatim)

```
mkdir -p /tmp/sweep_basetemp_final2
setsid nohup /home/heheboiz/data/cir/workbench/envs/ctl312h/bin/python -m pytest workbench/tests \
  -q -p no:cacheprovider -rfE --tb=short --basetemp=/tmp/sweep_basetemp_final2 \
  > /tmp/sweep_final2.txt 2>&1 < /dev/null &
```

Serial, no `-n`. Log: `/tmp/sweep_final2.txt` (not copied into the repo; the basetemp is deleted at the
end of the run — the summary line and greps below are the record).

### Stationary-tree proof (5-minute check)

Verified at **`2026-11-11T11:44:17+07:00`** (5 min 27 s after launch, sweep at ~29 %):

```
$ git rev-parse HEAD
c755673487744876d12fdd5a676a220329bd3468
$ git status --porcelain
?? CLEANUP_REPORT.md.save
$ pgrep -c -f 'pytest workbench/tests'
1
```

The tree was stationary apart from the one documentation-only commit `c755673` landing 2 min 40 s in;
see above — production and test trees identical, so the sweep is valid on both revisions.

## Summary line (verbatim, last line of the log)

```
18 failed, 2914 passed, 47 skipped, 350 warnings in 3562.27s (0:59:22)
```

## The four accounting checks

### 1. Counts
`18 failed, 2914 passed, 47 skipped` (above).

### 2. Error greps — both 0

| measure | value |
|---|---|
| `grep -c '^ERROR ' /tmp/sweep_final2.txt` | **0** — PASS (no collection/setup errors) |
| `grep -c 'Errno 28' /tmp/sweep_final2.txt` | **0** — PASS (no disk-full) |
| `grep -c '^FAILED' /tmp/sweep_final2.txt` | **18** |

`^E ` is deliberately not grepped: with `--tb=short` every traceback body line starts with `E `, so it
counts traceback volume, not errors.

### 3. Accounting gap = 0

```
$ workbench/envs/ctl312h/bin/python -m pytest workbench/tests -q --collect-only -p no:cacheprovider 2>&1 | tail -1
2979 tests collected in 3.48s
```

| N collected | passed | failed | skipped | sum | gap |
|---|---|---|---|---|---|
| 2979 | 2914 | 18 | 47 | 2979 | **0** |

`2979 − 2966 = +13` collected versus the `cd980dc` sweep, matching the three added/expanded test files
(`test_encoder_relation_handshake.py` 6 → 9 test functions, `test_gate_contract_negatives.py` +4 test
functions, `test_handoff_commands.py` one rewritten test). Every collected test reported a terminal
outcome; `^ERROR ` = 0 confirms no interrupted-module bookkeeping.

### 4. Failure diff vs baseline `68ad75e`

Rule (same as `failure_diff.py`): every `^FAILED` line, node id up to the first ` - `, sorted,
set-differenced. Baseline
`workbench/docs/reviews/20261110-e2e-repair/full_suite_68ad75e.txt` has 18 `FAILED` lines.

| NEW | DROPPED | COMMON |
|---|---|---|
| **0** | **0** | **18** |

**NEW list: (empty).**
**DROPPED list: (empty).**

### 5. Delta vs the previous authoritative sweep `cd980dc`

`/tmp/sweep_cd980dc.txt` still exists (85537 bytes, mtime `Nov 11 10:52`) — it was **not** deleted.

| measure | `cd980dc` sweep | this sweep (`8230358` → `c755673`) | delta |
|---|---|---|---|
| failed node ids | 18 | 18 | **identical set** |
| NEW failures | — | **0** | 0 |
| DROPPED failures | — | **0** | 0 |
| COMMON | — | **18** | 18 |
| collected | 2966 | 2979 | +13 (added tests) |
| passed | 2901 | 2914 | +13 |
| skipped | 47 | 47 | 0 |

**No failure appeared and none disappeared.** This is the correct outcome for a two-production-file
change: both changed modules are covered by newly added tests that pass, and no existing red test
depends on the code that changed. Specifically, no `test_encoder_relation_*` test fails (0 of the
relation-handshake/parity/seed files appear in `^FAILED`), and no gate test fails: the 47 collected
`test_gate_contract_negatives.py` cases and the 11 `test_encoder_relation_handshake.py` cases all pass —
verified directly:

```
$ pytest workbench/tests/test_encoder_relation_handshake.py test_encoder_relation_native_parity.py \
         test_encoder_relation_seed.py test_gate_contract_negatives.py test_handoff_commands.py -q
1 failed, 77 passed in 10.10s          # the single failure is the TESTED_SOURCE_SHA guard below
```

The single `test_handoff_commands.py` failure is exactly the expected one and no more: the file now shows
1 failed / 7 passed (8 collected), the same single guard, so the new cross-document count comparison added
in `bd9845d` passes.

## Per-failure classification

| # | nodeid | assertion (≤10 words) | class |
|---|---|---|---|
| 1 | `test_circus_adapters_blip.py::test_resident_worker_failure_uses_shared_error_and_stays_failed[model_conesep-conesep]` | `'Adapter' object has no attribute 'batch_size'` | PRE-EXISTING · ENVIRONMENT |
| 2 | `…[model_ptha_mtst-ptha_mtst]` | same `batch_size` AttributeError | PRE-EXISTING · ENVIRONMENT |
| 3 | `test_circus_adapters_blip.py::test_resident_worker_reuse_and_cleanup[model_conesep-conesep]` | same `batch_size` AttributeError | PRE-EXISTING · ENVIRONMENT |
| 4 | `…[model_ptha_mtst-ptha_mtst]` | same `batch_size` AttributeError | PRE-EXISTING · ENVIRONMENT |
| 5 | `test_circus_cli.py::test_cli_3d_full_gallery_resume_and_gallery_invalidation` | `assert 'REUSED' == 'COMPLETED'` | PRE-EXISTING · FLAKY |
| 6 | `test_circus_tme_lifecycle.py::test_encoder_and_limn_failed_child_latch_and_preserve_log[encoder]` | `WORKBENCH_ENCODER_PYTHON not set` | PRE-EXISTING · ENVIRONMENT |
| 7 | `…[limn]` | `WORKBENCH_LIMN_PYTHON not set` | PRE-EXISTING · ENVIRONMENT |
| 8 | `test_circus_tme_precision.py::test_worker_skips_empty_stdin_lines` | `FileNotFoundError: …/output.npy` (worker died on stub `torch`) | PRE-EXISTING |
| 9 | `test_f05_f08_f09_verification.py::test_f08_dcnet_config_data_dir_resolution` | `data/FashionIQ` ≠ `DCNet/dataset/fashioniq` | PRE-EXISTING |
| 10 | `test_handoff_commands.py::test_no_release_document_points_at_a_superseded_source_sha` | `TESTED_SOURCE_SHA cd980dc does not contain production code` | PRE-EXISTING (by-design red guard; repaired by the pin below) |
| 11 | `test_release_blocker_gpu_records.py::test_defect_b_g_with_no_target_gpu_verdict_is_pending_not_pass` | `RELEASE_BLOCKED`; J10 `built bundle lacks audit-threshold copy` | PRE-EXISTING |
| 12–16 | `…::test_release_verify_documented_invocations_produce_parseable_report[<lambda>0..4]` | invocation non-zero; J10 FAIL | PRE-EXISTING |
| 17 | `…::test_offline_verdict_names_deployment_reason` | `RELEASE_BLOCKED`; J10 FAIL | PRE-EXISTING |
| 18 | `test_standalone_operator_commands.py::test_prepare_model_executes_only_declared_recipe_steps` | dcnet `[MANUAL]` resized-images evidence missing | PRE-EXISTING |

### Tally

| class | count |
|---|---|
| PRE-EXISTING (all 18) | **18** |
| ├ ENVIRONMENT (`batch_size` ×4, unset `WORKBENCH_*_PYTHON` ×2) | 6 |
| ├ proven FLAKY | 1 (#5) |
| ├ by-design red guard, cleared by the pin | 1 (#10) |
| ├ deterministic red guards/evidence gaps | 10 |
| NEW-from-`9cefa36`-or-`34aaa10` | **0** |
| DROPPED | **0** |

Nothing is attributable to `9cefa36` (`parity_gate.py` count agreement) or `34aaa10`
(`model_encoder.py` construction-time relation persist): neither changed area contributes a failure, and
the 18 ids are set-identical to the baseline and to the `cd980dc` sweep.

### Evidence for the single FLAKY candidate

```
$ pytest workbench/tests/test_circus_cli.py::test_cli_3d_full_gallery_resume_and_gallery_invalidation \
         -q --basetemp=/tmp/flaky_final2_1
1 passed in 0.25s
```

It fails only inside the full serial suite (`'REUSED' == 'COMPLETED'`), passes alone: cross-module state,
not a defect introduced by this batch. The `batch_size` family re-confirmed deterministic in isolation
(`4 failed, 14 passed` for the whole `test_circus_adapters_blip.py` file alone).

## Conclusion

- **18 failed / 2914 passed / 47 skipped / 2979 collected, gap 0.**
- **0 NEW, 0 DROPPED, 18 COMMON** against the `68ad75e` baseline, and **set-identical to the `cd980dc`
  sweep** — no failure appeared or disappeared across the two production changes.
- All 18 are pre-existing: 6 environmental, 1 flaky-in-suite-only, 1 by-design red guard that the pin
  below clears, 10 long-standing red guards/evidence gaps.
- **This revision `c755673487744876d12fdd5a676a220329bd3468` is the one to pin as `TESTED_SOURCE_SHA`,
  with counts 2914 passed / 18 failed / 47 skipped / 2979 collected.** (`8230358`, the sweep-start
  revision, is production-tree identical and equally valid; `c755673` is named because it is the branch
  tip at sweep end and the handoff should name the furthest revision the sweep covers.)

After the pin, `test_no_release_document_points_at_a_superseded_source_sha` becomes green — but the two
documents must then also record `2914 passed, 18 failed, 47 skipped`, because
`test_the_verdict_names_the_same_test_count_the_handoff_does` requires both files to quote the same
triple, and neither may contradict itself.
