# A37 EVIDENCE INDEX

One line per piece of evidence. Everything lives under `/tmp/a37-*` or `/dev/shm/a37-*` (scratch) or in
the repository read-only. Nothing here was written into the repository.

## Reproduction helpers (re-runnable)

| Pointer | What it proves |
|---|---|
| `/tmp/a37-h1/repro.py` | H1: a config-less caller reuses another workspace's cached GPU capability record. Run: `PYTHONPATH=/home/heheboiz/data/cir .venv/bin/python /tmp/a37-h1/repro.py` → `probe_sequence_reused_workspace_a=True` |
| `/tmp/a37-h2h3/image_identity_repro.py` | H3: same path, same size, restored mtime, different bytes → identity changes. Run with the same `PYTHONPATH` → `same_identity: False` |
| `/tmp/a37-h2h3/hash_exact.py` | Cost of hashing the resolved gallery (count, total bytes, wall seconds) |
| `/tmp/a37-h3rev/runner_old_identity.py` | Exact pre-fix `_image_identity` body used as the RED baseline |
| `/tmp/a37-h3rev/test_e2e_resume_hole.py` | End-to-end: a same-size byte flip plus restored mtime produces a NEW run_id and no REUSE |
| `/tmp/a37-h3rev/test_reuse_still_works.py` | Unchanged content still resumes |
| `/tmp/a37-h3rev/test_memo_stale_reuse.py` | H3b: the memo variant served EARLIER identities against later-mutated bytes |
| `/tmp/a37-h3rev/test_memo_unsound.py`, `test_memo_layouts.py`, `test_memo_positive.py` | Memo soundness matrix → memo removed |
| `/tmp/a37-h3/count_calls.py` | Call-count restored: two models sharing one layout call `_image_identity` 4 times, not 2 |
| `/tmp/a37-h3/grep_old.txt`, `grep_new.txt`, `grep_consumers.txt` | No consumer reads the old inner keys (`file_stats` / `stat_identity_method`) |
| `/tmp/a37-cross/pipeline_command_collision.py` | P-01: two different command lists hash identically → `bec5599f…843c` |
| `/tmp/a37-cross/pipeline_resume_collision.py` | P-01 through `--resume` (fingerprint match at `pipeline.py:2283`) |
| `/tmp/a37-cross/fingerprint_mechanism.py` | Records the production update-chunks via a `hashlib.sha256` wrapper |
| `/tmp/a37-t1/probe_plugin.py` | T-01: `[PROBE-RESIDUE] total=14` before the fix |
| `/tmp/a37-t1/test_zz_no_post_purge_r19.py` | Shows the post-yield pop is load-bearing (removing it restores the residue) |
| `/tmp/a37-t1/test_zz_production_probe.py` | Production import path hijacked after the leak (`import data` resolves into the deleted tmpdir) |
| `/tmp/a37-t1/test_aa_sentinel.py` | Sentinel identity: the incoming objects are restored, not replaced |
| `/tmp/a37-at/mutate.py`, `/tmp/a37-at/matrix.py`, `/tmp/a37-at/run.sh` | T-02: the I09/I10 mutation matrix against shim roots, including the `pass / else: continue` and partial-body spellings that now fail |
| `workbench/tests/test_release_invariants.py:329-361`, `:446-461` | The tightened predicates: `_terminates`/`_live_statements` (terminator tests) and `_abandons_job` (only the TRUE branch, must abandon on every path) |
| `/tmp/a37-tq/pipe_nocheck_multi.py`, `pipe_nocheck_single.py` | T-03: real output validation disabled at both call sites |
| `/tmp/a37-tq/multicmd_probe.py` | T-07: a 2-command stage with no declared output is accepted when the branch is disabled |
| `/tmp/a37-tq/probe_plugin.py`, `probe_plugin_sentinel_aware.py` | T-01 probes used by the verifier |
| `/tmp/a37-tq/jobs_stopping_dead.py`, `jobs_stopping_never_set.py`, `mut_guard_first_only.py`, `mut_guard_second_only.py` | T-02 guard mutations |
| `/tmp/a37-tq/planner_no_guard.py`, `planner_other_bug.py` | T-05: the `KeyError` mutation that stayed green under the old assertion |
| `/tmp/a37-tq/tsdemo/` (`broken.ts`), `tsreal/` (`staleness.ts`) | T-06: `tsc` writes diagnostics to stdout, stderr empty |
| `/tmp/a37-tq/REPORT.md` | The full test-suite-quality audit behind T-01…T-08 |
| `/tmp/a37-va/probe*.py`, `vacuity.py`, `probe_i09_i10.py` | Strength checks for the strengthened tests |
| `/tmp/a37-va/leaky/`, `withfixture/`, `fxless/` | T-01 before/with-fixture/after trees and their consumer probes |
| `/tmp/a37-va/uismoke_mobile.txt` | UI-03: the mobile smoke's post-fix DOM numbers, incl. `left=195 right=378 width=183` |
| `/tmp/a37-va/bench.json` | 18 benchmark reports fetched from the REAL artifacts root |

## Suite evidence

| Pointer | What it holds |
|---|---|
| `/dev/shm/a37sf-out1.txt` | PRE-fix chunk 1: `541 passed, 1 skipped, 28 warnings in 703.29s` |
| `/dev/shm/a37sf-out2.txt` | PRE-fix chunk 2: `533 passed, 9 skipped, 36 warnings in 966.81s` |
| `/dev/shm/a37sf-out3.txt` | PRE-fix chunk 3: `1 failed, 540 passed, 19 warnings in 1319.00s`, with the SHA-guard traceback at `test_handoff_commands.py:485` |
| `/dev/shm/a37sf-out4.txt` | PRE-fix chunk 4: **truncated mid-run** (491 bytes, no summary) — the pre-fix chunk-4 count is derived by subtraction |
| `/dev/shm/a37sf-chunk1..4.txt` | Per-chunk test-file assignment (the partition actually used) |
| `/dev/shm/a37sf-files.txt`, `/dev/shm/a37sf-counts.txt` | Full file list and per-directory counts |
| `/dev/shm/a37sf-4/` | Basetemp of pre-fix chunk 4 |
| POST-fix counts (`2166 = 2154 passed + 1 failed + 11 skipped`) | [unverified — **no log on this host**; the four post-fix chunk logs are not in `/dev/shm`. The figures are the round's report, per-chunk `541+0+1, 533+0+9, 540+1+0, 540+0+1`] |

## Post-commit verification (no logs on this host either)

| Pointer | What it holds |
|---|---|
| `test_handoff_commands.py` before / after the docs commit | 2 failed / 6 passed → **8 passed** |
| `-k release_documents_quote_the_test_count` | **1 passed**; the only `**N passed, N failed**` string in `workbench/docs/` is `**2154 passed, 1 failed**` (re-derived by grep), first `N skipped` = 11, 2154+1+11 = 2166 |
| `test_handoff_commands.py` + `test_composition_image_identity.py` | **10 passed** |
| Focused production run (`test_composition_image_identity`, `test_circus_resume_stable`, `test_execution_identity`, `test_scientific_input_identity`) | **31 passed** |
| Affected-tests run (release_invariants + mutation_effectiveness + r12_remediations + r19_tme_replay + frontend_staleness + handoff_commands) | 72 passed, 1 failed — the by-construction SHA guard, before the docs commit |

## Git evidence (re-derived read-only)

| Pointer | What it holds |
|---|---|
| `git rev-parse HEAD` | `a134091bdd3ebd37ddaa6617ec1fa0ad6277f5e0` |
| `git ls-remote origin refs/heads/research/cir-failure-workbench` | the same SHA; `refs/heads/main` = `8ebcc74d5fd1934df4f3772a8ca5a96dbd00f333`, untouched |
| `git log --oneline -4` | `40dbc3b`, `8c510be`, `27554b2`, `a134091` on top of `e06b4e6` |
| `git diff --name-only 27554b2 HEAD -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'` | EMPTY (exit 0) — the release guard predicate holds |
| `git diff --name-only 8c510be HEAD -- …same pathspec` | `workbench/frontend/tests/ui_smoke.py` — why the production commit alone is insufficient |
| `git status --porcelain` | three untracked operator files only |
| `sha256sum pc_composition_smoke.yaml src/train_cirr.py uv.lock` | `aa4ccd6f…369e`, `bb394437…fb4`, `fba52f42…325` |

## Frontend evidence

| Pointer | What it holds |
|---|---|
| `/dev/shm/a37-ui/` (35 `.png`, 13 `.json`) | Screenshot campaign: `initial-*`, `loading-models-*`, `inventory-*`, `blocked-inventory-*`, `empty-explorer-*`, `query-explorer-*`, `pagination-*`, `topk-50-*`, `compare-*`, `cohort-*`, `annotation-*`, `api-error-*`, `recovery-*`, `stale-*` across 1440x900 / 1280x800 / 390x844 / 360x800 |
| `/dev/shm/a37-ui/shot-review.md` | The vision reviewer's per-screenshot notes, including which shots are NOT pixel-reviewed |
| `/dev/shm/a37-ui/annotation-proof.json` | Annotation persisted across reload |
| `/dev/shm/a37-ui/stale-evidence.json`, `topk-stale.json` | Out-of-order stale response and Top-K above saved depth |
| `/dev/shm/a37-uifix/finalverify.json` | Post-fix geometry with and without the nested table open (`left=195 right=378`, `anyClipped: 0`) |
| `/dev/shm/a37-uifix/realdataprobe.json` | Real-artifacts-root probe: `directRows=17` vs `allRows_descendant=197`, cell geometry at 390 and 360 |
| `/dev/shm/a37-uifix/gatetable.json` | V-2: nested gate-detail header widths, descendant vs direct-child |
| `/dev/shm/a37-uifix/bench-real.json`, `has.json` | The 18 benchmark reports served by the real root |
| `/dev/shm/a37-uifix/final-check.json`, `verify.json`, `verify-final.json`, `measure.json`, `measure-after.json` | Before/after measurements and the consolidated-diff verification |
| `/dev/shm/a37-uifix/cand*.json`, `cand-*-390.png` | The four candidate CSS variants and their crops |
| `/dev/shm/a37-uifix/tile-{inflight,settled}-{1440x900,1280x800,390x844,360x800}.png` | UI-02: the first tile while in flight and once settled |
| `/dev/shm/a37-uifix/shots-all/`, `shots-final/`, `shots-guard/`, `shots-guard2/` | Post-fix capture sets, incl. the guard run where the assertion is proven to bite |
| `/dev/shm/a37-uifix/ab.mjs`, `ab2.mjs`, `dash.mjs`, `tile.mjs`, `verify.mjs`, `widthtest.mjs`, `scopecheck.mjs`, `settle.mjs`, `sim.mjs` | The Playwright drivers used for each measurement |

## Repository artifacts referenced by the ledger

| Pointer | What it holds |
|---|---|
| `workbench/tests/test_composition_image_identity.py` (untracked, 42 lines) | H3 regression test: byte change with restored mtime must change the identity; identical bytes must reproduce it; plus the roll-up assertion |
| `workbench/artifacts/composition/runs/1410522b57d4fe79083c60a6/ranks/encoder_fashioniq.json` | Stored per-query exact ranks per condition — the input of the independent metric recomputation |
| `workbench/artifacts/composition/summary/encoder_fashioniq_1410522b57d4fe79083c60a6/full.json:338` | Production values: `gap_multimodal_ndcg 0.2506690694032824`, `gap_text_only_ndcg 0.12021218172563472`, `gap_image_only_ndcg 0.08499005522926514`, `composition_gap_ndcg 0.5204347229125645` |
| `workbench/artifacts/gpu_capabilities/` (5,215 records) | Census: interpreter 1,018 / override 3,857 / forged 340; capability 8.6 x9, 8.9 x3,181, absent x2,025; ZERO at 12.0 |
| `workbench/third_party/*/provenance.json` (16 files) | `upstream_commit` agrees with `models.yaml:upstream_commit_sha` for all declared model ids (17 match, 0 mismatch); no nested `.git` |
| `workbench/docs/reviews/FINAL_PC_RELEASE_VERDICT.md` | Where the verdict actually lives (the pilot handoff points one directory too high) |
