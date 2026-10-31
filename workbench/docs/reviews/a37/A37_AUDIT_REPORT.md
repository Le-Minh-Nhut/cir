# A37 AUDIT REPORT — CIR Failure Analysis Workbench

Repo: `/home/heheboiz/data/cir`. Branch: `research/cir-failure-workbench`.
The round is **committed and pushed**; final HEAD `a134091bdd3ebd37ddaa6617ec1fa0ad6277f5e0`.
Nothing was written into the repository by this drafting pass.

Every claim below cites a `file:line`, a command, or a measured number. Claims that could not be
re-derived in this drafting pass carry `[unverified]`.

---

## 1. Baseline

| Item | Value |
|---|---|
| Branch | `research/cir-failure-workbench` |
| Starting HEAD | `e06b4e61d894cd895034838b126ad8046a0602a5` (`git rev-parse HEAD`) |
| Final HEAD | `a134091bdd3ebd37ddaa6617ec1fa0ad6277f5e0` (`git rev-parse HEAD`) |
| Remote tip at end | `a134091bdd3ebd37ddaa6617ec1fa0ad6277f5e0` — same SHA as final HEAD (`git ls-remote origin refs/heads/research/cir-failure-workbench`) |
| `origin/main` | untouched at `8ebcc74d5fd1934df4f3772a8ca5a96dbd00f333` (`git ls-remote origin refs/heads/main`) |
| Push | fast-forward `e06b4e6..a134091`; no force |
| `TESTED_SOURCE_SHA` at start | `a6449e060e77f0deed0c79545dd0d5c1019e333c` (the A36 round) |
| `TESTED_SOURCE_SHA` at end | `27554b2ac731c40385cbf72daf06d5ed5a0a24d7` — the A37 **test** commit; see the justification below |
| Recorded production bytes (A36) | `b8adc465dbde00e7ca7160b4ab075a37506c4cbb` [unverified — doc value, not re-derived] |
| A37 production commit | `8c510be2ab9a9eefdfbacac0ca3b3698033c6664` |
| Working tree at end | clean except exactly three untracked operator files, sha256 unchanged: `pc_composition_smoke.yaml` `aa4ccd6f…369e`, `src/train_cirr.py` `bb394437…fb4`, `uv.lock` `fba52f42…325` |
| Host | RTX 3050 Laptop GPU, 4096 MiB, sm_86 (`nvidia-smi`); target RTX 5070 Ti sm_120 NOT present |
| Controller env | `.venv` Python 3.13.12, torch 2.14.1+cu130; node v20.19.5; Playwright browsers at `/tmp/pw-browsers` |

The round's change set, now committed in four commits:

| # | SHA | Subject | Files |
|---|---|---|---|
| 1 | `40dbc3b05f0504e94be4f949f6a72b495e85feff` | `chore(repo): ignore the operator-local .omp/ directory` | `.gitignore`, +3 |
| 2 | `8c510be2ab9a9eefdfbacac0ca3b3698033c6664` | `fix(A37): bind composition dataset identity to image bytes, fix mobile model-card clipping and the dashboard count placeholder` | `composition/runner.py`, `frontend/App.tsx`, `frontend/style.css` — 3 files, +23/−14 |
| 3 | `27554b2ac731c40385cbf72daf06d5ed5a0a24d7` | `test(A37): repair the regression tests that could not fail, and pin the composition content identity` | 7 files, +517/−34 |
| 4 | `a134091bdd3ebd37ddaa6617ec1fa0ad6277f5e0` | `docs(A37): advance the tested source revision to 27554b2 and record the A37 evidence` | 6 docs, +52/−26 |

### Why `TESTED_SOURCE_SHA` is the test commit, not the production commit

Recorded as `27554b2ac731c40385cbf72daf06d5ed5a0a24d7`
(`workbench/docs/PC_REPRODUCTION_HANDOFF.md:5,127,138`;
`workbench/docs/reviews/FINAL_PC_RELEASE_VERDICT.md:22,38`). This is a documented,
mechanically-justified choice, **not** a deviation:

- the release guard's exclusion list is only `workbench/docs` and `workbench/tests`, so
  `workbench/frontend/tests/ui_smoke.py` counts as **production** for that predicate;
- `8c510be` alone therefore returns **nonzero** on the guard — re-derived here:
  `git diff --name-only 8c510be HEAD -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'`
  prints `workbench/frontend/tests/ui_smoke.py`;
- `27554b2` returns **zero** on the same predicate (re-derived: empty output, exit 0) and is an
  ancestor of HEAD containing every production change.

The guard predicate now holds at HEAD, so
`workbench/tests/test_handoff_commands.py::test_no_release_document_points_at_a_superseded_source_sha`
passes (8 passed for that file after the docs commit, see section 6).

---

## 2. Coverage

Inventory measured on this tree: 241 tracked first-party `.py` files / 94,954 lines under
`workbench/` excluding `third_party/` (`git ls-files 'workbench/**/*.py' | grep -v third_party`),
of which 148 are `workbench/tests/*.py`, 28 `workbench/scripts/*.py`, 23 `workbench/replay/*.py`,
21 `workbench/backend/*.py` + 3 `backend/adapters/*.py` + 2 `backend/schemas/*.py`,
15 `workbench/composition/*.py`; frontend 12 tracked TS/TSX/CSS files / 573 lines.
Vendored `workbench/third_party/` is 3.6 GB, 3,495 tracked `.py` files across 16 directories
(3,513 on disk including untracked); registry `workbench/registry/models.yaml` declares 17 models
(9 active: limn, encoder, csmcir, ptha_mtst, tme, intent, habit, conesep, hint; 8 deferred:
airknow, pair, clvc_net, dcnet, combiner_rn50x4_noft, clip4cir_rn50x4_fullft, tgcir, sprc). All 17
declare `source_available: true`; no vendored dir contains a nested `.git` (`find workbench/third_party -maxdepth 3 -name .git` → 0); all 16 per-directory `provenance.json` files agree with
`models.yaml:upstream_commit_sha` for every declared `model_ids` entry (re-derived: 17 match, 0 mismatch).

| Surface | Verdict | Reason |
|---|---|---|
| `workbench/composition/` (runner, metrics, inputs, data, 9 model adapters) | AUDITED | Read end to end for this round; identity, resume, metric, condition-key paths re-derived by execution |
| `workbench/backend/runtime.py` (env, GPU probe, capability gate) | AUDITED | Gate, cache keys, override hatch read at line; H1 repro re-run |
| `workbench/backend/adapters/` + `workbench/replay/` | PARTIALLY AUDITED | Adapter class identity/`launcher`/`launcher_roots`/`reference_eligible`/`category_scopes` re-derived for the 9 active composition adapters; the 23 replay wrappers read only at their integration seams (`tme.py:354-359`, `ptha_mtst.py:189`) |
| `workbench/scripts/pipeline.py` | PARTIALLY AUDITED | Fingerprint, resume/skip, stage-output validation and stage construction read; the other ~2,400 lines only spot-read |
| `workbench/frontend/` (12 files / 573 lines) | AUDITED | Every file read or rendered; 35 screenshots captured; CSS/TS fixes measured |
| `workbench/tests/` (148 files) | PARTIALLY AUDITED | 6 test files audited line by line and strengthened; the rest only via the full-suite run and targeted mutation probes |
| `workbench/third_party/` (3,495 `.py`) | NOT AUDITED | Only integration boundaries and per-model modifications were examined (adapters, replay wrappers, `provenance.json`, hashes). The vendored upstream sources were NOT hand-audited |
| `pc.sh`, `workbench/run.sh` | PARTIALLY AUDITED | Exit codes, flag list, `--self-check` body, subcommand dispatch read; the Docker/preflight bodies only spot-read |
| `workbench/docs/` (217 `.md`) | PARTIALLY AUDITED | The release/handoff/reproduction docs were checked against code for the drift list in the ledger; the review archives were not re-audited |
| PC/sm_120 execution path | NOT AUDITED | No sm_120 host, no sm_120 probe, no sm_120 inference exists in this repository (see PC readiness) |

---

## 3. Findings ledger

| ID | Title | Class | Severity | file:line | Evidence |
|---|---|---|---|---|---|
| H1 | GPU capability config scope: a caller's no-`config=` lookup can reuse another workspace's cached record | EXPECTED_BEHAVIOR | LOW | `workbench/scripts/pipeline.py:1440-1441`, `:1764-1765` | `/tmp/a37-h1/repro.py` → `probe_sequence_reused_workspace_a=True`; NOT reachable in CLI/autopilot: `pipeline.py:2471` resolves config, `backend/autopilot.py:532` exports `WORKBENCH_ARTIFACTS_ROOT` for the child |
| H2 | PC cohort ships `selected_models: all_registered`, so 8 deferred models are in the selection | NOT_A_DEFECT_EXPECTED_BEHAVIOR | INFO | `workbench/configs/pc_autopilot.pc_cohort.yaml:24`; `backend/autopilot.py:105` | No PAUSED filter in `autopilot.py:101-106`; `composition_eval.py:108` plans `dry_run=True`, `:262-286` reports `SELECTED_BUT_BLOCKED` per row, `:310-312` returns 1 only when the whole cohort is empty; disclosed at `docs/CIRCUS_COMPOSITION_SPEC.md:274`, `docs/PC_PILOT_HANDOFF.md:23-24`, `docs/PC_REPRODUCTION_HANDOFF.md:114` |
| H3 | Composition dataset identity was built from `(st_size, st_mtime_ns)`, never from bytes | CONFIRMED_FIXED | HIGH | `workbench/composition/runner.py:530-540` (was `:530-540`, `file_stats`/`stat_identity_method`) | `/tmp/a37-h2h3/image_identity_repro.py` → `same_identity: False` after the fix; pre-fix body preserved at `/tmp/a37-h3rev/runner_old_identity.py`; new test `workbench/tests/test_composition_image_identity.py` (42 lines) fails on the pre-fix body, passes after. Accepted cost: no caching, so the resolved gallery is re-hashed once per model — ~2.6 s per 1.13 GB gallery, ~+45 s across 17 models [unverified — worker-measured] |
| H3b | A per-run identity memo was tried and REMOVED (stale-reuse regression) | CONFIRMED_FIXED (negative result) | HIGH | `workbench/composition/runner.py:534` | `/tmp/a37-h3rev/test_memo_stale_reuse.py`, `test_memo_unsound.py`; call count restored to 4 per two sharing models (`/tmp/a37-h3/count_calls.py`) |
| P-01 | Stage fingerprints concatenate `shlex.join(cmd)` with no boundary, so two different command lists hash identically | CONFIRMED_UNFIXED (robustness nit) | LOW | `workbench/scripts/pipeline.py:309-310` | Re-derived: both lists hash `bec5599f596cb9a033930bbc1ad3405874595a7d22fcbd2cb02948c23045843c` (`/tmp/a37-cross/pipeline_command_collision.py`); reachable only via `--resume` (`pipeline.py:2283`, `:2309`) and no realistic collision pair exists in the stage-construction sites |
| T-01 | `test_r19_tme_replay.py` leaked fixture modules into `sys.modules` process-wide | CONFIRMED_FIXED | HIGH | `workbench/tests/test_r19_tme_replay.py:22-46` (was `:94-100`) | `/tmp/a37-t1/probe_plugin.py`, `test_zz_no_post_purge_r19.py`; residue `total=0` after the autouse fixture; production unreachable: `replay/tme.py:354-359`, `ptha_mtst.py:189` run in subprocess CLIs |
| T-02 | I09/I10 asserted on substrings satisfied by comments and dead code | CONFIRMED_FIXED | HIGH | `workbench/tests/test_release_invariants.py:329-361` (`_terminates`, `_live_statements`), `:446-461` (`_abandons_job`) | AST-grounded rows; after the second tightening round `_abandons_job` reads ONLY the branch taken on a TRUE condition and requires the body to abandon on EVERY path. `if self._stopping: pass / else: continue` now FAILS; the partial-body spelling `if self._stopping: <nested if>: continue` also FAILS. Mutation matrix: `/tmp/a37-at/mutate.py` + shim roots — every named mutation FAILS, real file passes 22/22 |
| T-03 | mut2/mut9 (and mut3) were tautological — the mutator rebound the symbol the detector called | CONFIRMED_FIXED | HIGH | `workbench/tests/test_mutation_effectiveness.py:51-63`, `:168-176` | Detectors now drive `pipeline.run_stages`; `/tmp/a37-tq/pipe_nocheck_multi.py` mutation: 10 passed before → 3 failed after |
| T-05 | r12 conda-solve test asserted only "did not raise TypeError" | CONFIRMED_FIXED | MEDIUM | `workbench/tests/test_r12_remediations.py:190-236` | Now patches `planner.subprocess.run` and asserts the real argv (one launch, `create --dry-run --json`, `-c` channel pairs, pinned requirement present, source requirement absent) |
| T-06 | Any nonzero `tsc` exit became `pytest.skip(...)`, but `tsc` writes diagnostics to stdout | CONFIRMED_FIXED | MEDIUM | `workbench/tests/test_frontend_staleness.py:91-102`, `:244-253` | Now skips only when `npx` is unresolvable; otherwise asserts exit 0 and surfaces stdout |
| T-07 | Multi-command post-loop output validation has no dedicated repository test | CONFIRMED_UNFIXED (known gap) | MEDIUM | `workbench/scripts/pipeline.py:2385-2386` | Disabling only that branch leaves the repo tests green; closest coverage is the strengthened `test_mutation_effectiveness.py` detector (`/tmp/a37-tq/multicmd_probe.py` fails under it) |
| UI-01 | Models page: entire detail column off-screen at mobile widths | CONFIRMED_FIXED | HIGH | `workbench/frontend/style.css:21-22` | Pre-fix at 390 px: table min-content 1041.17 px in a 366 px section, detail `<td>` at x=387.27 with zero non-background pixels in the body region; post-fix left=195 right=378, 50,694 non-background px at >10; desktop capture byte-identical. `/dev/shm/a37-uifix/{finalverify.json,realdataprobe.json}` |
| UI-02 | Dashboard first tile showed a literal `…` for ~12-14 s | CONFIRMED_FIXED | MEDIUM | `workbench/frontend/App.tsx:60` | Was `models===null?"…"`; now `"—"`. Timings 13.72 s / 14.08 s for `/api/models` are [unverified — worker-measured]. `/dev/shm/a37-uifix/tile-{inflight,settled}-*.png` |
| UI-03 | Smoke's mobile assertion could never see the clipping | CONFIRMED_FIXED | MEDIUM | `workbench/frontend/tests/ui_smoke.py:275-299`, `:596-632` | `body.scrollWidth <= innerWidth+2` passes while the section clips internally; geometry + API-derived row-count assertion added; with the CSS neutralised the new assertion FAILS naming `{'left': 387, 'right': 1053, …}` while the old check PASSED (`/tmp/a37-va/uismoke_mobile.txt`) |
| UI-04 | Smoke selector `section table tbody tr` was a descendant path matching the nested gate table | CONFIRMED_FIXED | MEDIUM | `workbench/frontend/tests/ui_smoke.py:288`, `:605` | Real root: 197 descendant rows vs 17 direct rows (`/dev/shm/a37-uifix/realdataprobe.json`); now `section > table > tbody > tr` with the count from `/api/models` |
| UI-05 | CSS selector was a descendant path and reshaped the nested gate-detail table | CONFIRMED_FIXED | MEDIUM | `workbench/frontend/style.css:21-22` | Nested gate table header cells collapsed to a flat 20 px each under the descendant form; direct-child path restores auto widths 33/27/28/28/30/31/32 (`/dev/shm/a37-uifix/gatetable.json`) |
| D-01 | `PC_PILOT_HANDOFF.md` drift (8 verified items, unfixed — the docs round was out of scope after the production commits) | CONFIRMED_UNFIXED (out of scope) | LOW | see below | See D-01 detail |
| G-01 | No sm_120 execution, capability record or inference has ever existed here | BLOCKED | — | census below | ZERO records with `compute_capability 12.0`; target GPU absent from the host |
| S-01 | `test_no_release_document_points_at_a_superseded_source_sha` failed by construction while the tree was uncommitted | RESOLVED | — | `workbench/tests/test_handoff_commands.py:485` | Pre-commit: the tree carried the fixes, so `git diff --quiet <sha> -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'` returned 1. The docs commit advanced the recorded SHA to `27554b2`; the guard predicate is now EMPTY and the file reports 8 passed |
| M-01 | Independently recomputed composition metrics match production exactly | INFORMATIONAL | — | `workbench/artifacts/composition/…` | Recomputed from the stored per-query rank record; see section 4 |

### D-01 detail (all re-verified in this pass)

| # | Claim in `workbench/docs/PC_PILOT_HANDOFF.md` | Reality |
|---|---|---|
| 1 | ENCODER declares no per-query observer (`:355`) | `workbench/registry/models.yaml:721` declares `per_query_observer: ENCODER_QUERY_OBSERVER_V1` |
| 2 | "only LIMN has an observer" (`:260`) | two observers exist: `models.yaml:721` (ENCODER) and `:1284` (LIMN) |
| 3 | exit codes 0/3/4/5/7 (`:115-125`) | `pc.sh:2245` and `:2250` also emit 6 |
| 4 | flag list of 9 flags (`:86-87`) | omits `--self-check` (`pc.sh:217`, `:245`) |
| 5 | `--self-check` implies more than it does | it self-tests the `.env` parser and `resolve_path` only (`pc.sh:254-274`) |
| 6 | `--models encoder` as a bare command (`:23`) | `workbench/run.sh:20-27` requires the `composition-eval` subcommand for that script |
| 7 | points at `workbench/docs/FINAL_PC_RELEASE_VERDICT.md` (`:67`) | file is at `workbench/docs/reviews/FINAL_PC_RELEASE_VERDICT.md` |
| 8 | `CAPABILITY_ABSENT_FOR_DIAGNOSTIC` listed as a `per_query_state` (`:119`) | it is only a `per_query_export_reason` (`scripts/evaluate_models.py:2185` → `:2225`) |

---

## 4. Scientific correctness

| Axis | Status in this round |
|---|---|
| Protocol identity | READ. `docs/CIRCUS_COMPOSITION_SPEC.md` protocol handling and `composition/runner.py:629` (`subset_memberships`, `conditions`) read; not executed here |
| Retrieval | PARTIALLY EXECUTED. Reconciled from a stored per-query rank artifact (below) |
| Composition — the three conditions distinct | VERIFIED BY READING. `workbench/composition/inputs.py:6` defines `("multimodal", "text_only_black_image", "image_only_empty_text")`; `:19-27` maps each to a distinct (pixels, text) pair; they stay distinct in the run identity (`runner.py:629`) and in the query-feature memmap key (`runner.py:736`, which includes `"condition": condition`) |
| Adapters | PARTIALLY EXECUTED. 9 active composition adapters resolved and their `reference_eligible`/`category_scopes` read; native condition support is per-model (`model_csmcir.py:73` refuses all conditions; `model_conesep.py:55`, `model_ptha_mtst.py:62`, `model_habit.py:131`, `model_intent.py:127`, `model_hint.py:129`, `model_tme.py:144` return `(True, None)` for known conditions; `model_encoder.py`/`model_limn.py` use `composition/adapters.py:106`). No native GPU forward was executed in this round |
| Checkpoint identity | READ. `checkpoint_identity` is part of the identity dict (`runner.py:626-628`); the H3 fix changed only the image half |
| Metric validity | EXECUTED (independent recomputation) |
| Resume validity | EXECUTED (H3 pre/post, `--resume` reuse) and READ for the pipeline fingerprint (`pipeline.py:2283`, `:2309`) |
| Result provenance | READ. `adapter_module_source_hash`, `adapter_source_hashes`, `runtime_identity`, `gpu_identity`, `preprocessing` are all in the identity dict (`runner.py:626-644`) |
| NOT verified this round | Native per-model GPU forward on the target capability; end-to-end protocol comparison against paper references; the 3,495 vendored sources themselves |

### Metric recomputation (independently re-derived in this pass)

Source artifact: `workbench/artifacts/composition/runs/1410522b57d4fe79083c60a6/ranks/encoder_fashioniq.json`
(per-query exact ranks per condition) and the published summary
`workbench/artifacts/composition/summary/encoder_fashioniq_1410522b57d4fe79083c60a6/full.json:338`.

Recomputing `1/log2(rank+1)` per query and averaging over the 8-query common cohort, and recomputing
`1 - max(text, image) / multimodal`, reproduced every published value exactly:

| Quantity | Production | Recomputed |
|---|---|---|
| mean NDCG_MM | 0.2506690694032824 | 0.2506690694032824 |
| NDCG_T | 0.12021218172563472 | 0.12021218172563472 |
| NDCG_I | 0.08499005522926514 | 0.08499005522926514 |
| gap = 1 − max(T,I)/MM | 0.5204347229125645 | 0.5204347229125645 |
| MM Recall@1 / @10 / @50 | 12.5 / 12.5 / 25.0 % | 12.5 / 12.5 / 25.0 % |

---

## 5. Adapter matrix (active cohort)

`reference_eligible` and `category_scopes` re-derived by loading each registry `composition_adapter`
and reading the class attributes (`composition/adapters.py:68-69` is the default). The per-property
verdict `EXPECTED_BEHAVIOR` is the round's finding; the file:line citations below are the code the
verdict rests on, re-checked in this pass. [unverified — the round's per-adapter verdict artifact
(`/tmp/a37-adapters/`) is empty, so only the cited lines were re-derived.]

| Model | Registry declaration | Adapter class | `reference_eligible` | `category_scopes` | `mode_supported` | Verdict |
|---|---|---|---|---|---|---|
| csmcir | `models.yaml:29` | `backend/adapters/models.py:73` (`script` launcher) | True | `(None,)` | refuses all conditions (`composition/model_csmcir.py:73`) | EXPECTED_BEHAVIOR |
| encoder | `models.yaml:683` | `backend/adapters/models.py:82` (`replay`, no extra roots) | True | `(None,)` | contract default (`composition/adapters.py:106`) | EXPECTED_BEHAVIOR |
| hint | `models.yaml:608` | `backend/adapters/models.py:96` (`replay`, `lavis`) | True | `(None,)` | `composition/model_hint.py:129` | EXPECTED_BEHAVIOR |
| conesep | `models.yaml:195` | `backend/adapters/models.py:205` (`replay`, `lavis`) | True | `(None,)` | `composition/model_conesep.py:55` | EXPECTED_BEHAVIOR |
| habit | `models.yaml:417` | `backend/adapters/models.py:246` (`replay`, `lavis`) | True | `(None,)` | `composition/model_habit.py:131` | EXPECTED_BEHAVIOR |
| intent | `models.yaml:514` | `backend/adapters/models.py:286` (`replay`, `lavis`) | True | `(None,)` | `composition/model_intent.py:127` | EXPECTED_BEHAVIOR |
| ptha_mtst | `models.yaml:302` | `backend/adapters/models.py:489` (`replay`, `MTST_PTHA/src`) | True | `(None,)` | `composition/model_ptha_mtst.py:62` | EXPECTED_BEHAVIOR |
| limn | `models.yaml:1280` | `backend/adapters/models.py:434` (`replay`, `LIMN`) | True | `('dress','shirt','toptee')` | contract default | EXPECTED_BEHAVIOR |
| tme | `models.yaml:1376` | `backend/adapters/models.py:466` (`replay`, `src`) | True | `(None,)` | `composition/model_tme.py:144` | EXPECTED_BEHAVIOR |

The three composition conditions are distinct (`composition/inputs.py:6`) and remain distinct in the
query cache key (`composition/runner.py:736` keys on `condition`, `scope`, `kind` and the run
fingerprint).

---

## 6. Verification evidence

### Real-GPU

| What | Command / source | Result |
|---|---|---|
| Host GPU | `nvidia-smi --query-gpu=name,memory.total --format=csv` | `NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB` |
| sm_120 inference | none | **NOT EXECUTED — no sm_120 device in this repository's history** |

### CPU / full suite

| What | Command | Result |
|---|---|---|
| Pre-fix suite, 4 sequential thread-capped chunks, `/dev/shm` basetemps | logs `/dev/shm/a37sf-out{1..4}.txt`, file lists `/dev/shm/a37sf-chunk{1..4}.txt` | 2164 collected = **2153 passed + 1 failed + 10 skipped**. Measured chunk summaries: out1 `541 passed, 1 skipped`, out2 `533 passed, 9 skipped`, out3 `1 failed, 540 passed`. **out4's log is truncated mid-run (491 bytes, no summary line)**; its 539 passed are derived by subtraction (2153 − 541 − 533 − 540), not measured |
| **Post-fix suite** (the fixed tree), 4 sequential thread-capped chunks, `/dev/shm` basetemps | reported by the round | **2166 collected = 2154 passed + 1 failed + 11 skipped**. Per-chunk (passed+failed+skipped): 541+0+1, 533+0+9, 540+1+0, 540+0+1. These are a **per-chunk sum, not a single-process run** [unverified — chunk logs not retrievable on this host; `/dev/shm` holds only the pre-fix set] |
| Post-fix delta vs pre-fix | derived | +2 collected (the new identity test plus the strengthened rows), +1 passed, +1 skipped. The added skip is a host boundary, not a pass |
| The single failure (both runs) | `workbench/tests/test_handoff_commands.py:485` | `AssertionError: TESTED_SOURCE_SHA … does not contain the production code currently on disk` — S-01, fails by construction on the revision it describes; resolved by the docs commit advancing the recorded SHA |
| The 10 skips | `test_r11_lavis_lifecycle.py:149` (×6), `test_r11_remediations.py:63` (×2), `test_r10_remediations.py:99` (×1), `test_execution_graph_and_dependencies.py:317` (×1) | all four sites re-read; each skips only for the stated host condition |
| `compileall` over backend/composition/replay/scripts | — | exit 0 [unverified — reported, not re-run in this pass] |
| `npm run build` (`tsc -b && vite build`) | — | exit 0 [unverified — reported, not re-run in this pass] |

### Post-commit verification runs

| Run | Result |
|---|---|
| `test_handoff_commands.py` before the docs commit | 2 failed / 6 passed — both failures by construction |
| `test_handoff_commands.py` after the docs commit | **8 passed** |
| `-k release_documents_quote_the_test_count` | **1 passed** — all bolded `**N passed, N failed**` pairs collapse to the single set `**2154 passed, 1 failed**` (re-derived: that string is the only `**N passed, N failed**` form in `workbench/docs/`), first `N skipped` = 11, and 2154+1+11 = 2166, which is what the named revision's own worktree collects |
| `test_handoff_commands.py` + `test_composition_image_identity.py` | **10 passed** |
| Focused production run (`test_composition_image_identity` + `test_circus_resume_stable` + `test_execution_identity` + `test_scientific_input_identity`) | **31 passed** |
| Affected-tests run (release_invariants + mutation_effectiveness + r12_remediations + r19_tme_replay + frontend_staleness + handoff_commands) | 72 passed, 1 failed — the failure was the same by-construction SHA guard *before* the docs commit advanced the SHA |

### Synthetic / mocked

| What | Command | Result |
|---|---|---|
| H1 config-scope repro | `PYTHONPATH=/home/heheboiz/data/cir .venv/bin/python /tmp/a37-h1/repro.py` | `probe_sequence_reused_workspace_a=True` (re-run in this pass) |
| H3 identity repro | `…python /tmp/a37-h2h3/image_identity_repro.py` | `same_identity: False`, identity carries `resolved_paths_sha256` + rolled-up `content_digest` (re-run) |
| Stage-fingerprint collision | `…python /tmp/a37-cross/pipeline_command_collision.py` | both digests `bec5599f…843c`, `True True` (re-run) |
| New regression test (H3) | `workbench/tests/test_composition_image_identity.py` | RED pre-fix (KeyError/AssertionError), GREEN post-fix; 42 lines |
| r19 module-leak probes | `/tmp/a37-t1/probe_plugin.py`, `test_zz_no_post_purge_r19.py` | residue `total=0` after the fix; removing the post-yield pop restores it |
| I09/I10 mutation matrix | `/tmp/a37-at/mutate.py` + shim roots | every named mutation FAILS; real file passes 22/22 |
| Mutation-tautology probe | `/tmp/a37-tq/pipe_nocheck_multi.py` | real output validation disabled: mut suite reported 10 passed before the fix, 3 failed after |
| Multi-command gap probe | `/tmp/a37-tq/multicmd_probe.py` | unmutated 1 passed; multi-branch disabled → assertion names an accepted 2-command stage with no output |

### Browser (Playwright, vision-reviewed where noted)

| What | Command / source | Result |
|---|---|---|
| Screenshot campaign | `/dev/shm/a37-ui/` — 35 `.png`, 13 `.json` evidence files | 1440x900 / 1280x800 / 390x844 / 360x800 across loading, inventory, blocked model, empty, explorer, pagination, Top-K, comparison, cohort, annotation, API error, recovery, slow network, stale out-of-order |
| Fix campaign | `/dev/shm/a37-uifix/` — `shots*/`, `measure*.json`, `verify*.json`, `realdataprobe.json`, `gatetable.json`, `has.json` | pre/post geometry, real-artifacts-root probe (18 benchmark reports in `bench-real.json`), nested-table widths, desktop byte-compare |
| Mobile geometry after the fix | `/dev/shm/a37-uifix/realdataprobe.json` | 390: `directRows=17`, `allRows_descendant=197`, first detail cell `left=195 right=378`; 360: `left=180`; all cells inside, non-empty, not collapsed |

---

## 7. Git state

| Item | Value |
|---|---|
| Starting SHA | `e06b4e61d894cd895034838b126ad8046a0602a5` |
| Final HEAD | `a134091bdd3ebd37ddaa6617ec1fa0ad6277f5e0` |
| Remote tip | `a134091bdd3ebd37ddaa6617ec1fa0ad6277f5e0` — the same SHA (verified with `git ls-remote`) |
| Push | fast-forward `e06b4e6..a134091`; no force |
| `origin/main` | untouched at `8ebcc74d5fd1934df4f3772a8ca5a96dbd00f333` |
| Commits | 4, in the order listed in section 1 |
| Uncommitted at end | none; working tree clean except the three untracked operator files below |
| Untracked, preserved, sha256 unchanged | `pc_composition_smoke.yaml` `aa4ccd6f405791662f8be2442e3a93e735bd2b81753c79815c30506638cf369e`; `src/train_cirr.py` `bb394437c13be97f3ff041274a8952c6054afc425308ef5f6edbc092abfcefb4`; `uv.lock` `fba52f424b9fed6776dd04c656c9024a7b6abe329aa2eb96a58adbd9454b4325` (all three re-derived with `sha256sum` in this pass) |
| Release guard predicate | `git diff --name-only 27554b2 HEAD -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'` → EMPTY (re-derived, exit 0), so `test_no_release_document_points_at_a_superseded_source_sha` passes |
| Read-only discipline | this drafting pass made no edits, staging, commits, push, or checkout/stash/reset/clean; scratch only under `/tmp/a37-*` |

Commit order rationale: `.gitignore` first (unrelated housekeeping); then production (the H3 identity
fix plus its regression test); then the test-quality repairs; then the docs that advance
`TESTED_SOURCE_SHA`. The last commit is what clears S-01 — the guard keeps failing until the recorded
SHA names a revision whose production tree equals the disk.

---

## 8. PC handoff

Six operator paths to set in a private `~/cir_pc.yaml` (`workbench/docs/PC_REPRODUCTION_HANDOFF.md:84`):

| YAML `paths.*` | standalone setting |
|---|---|
| `registry` | `WORKBENCH_REGISTRY_PATH` (process env only; rejected by `workbench.env`) |
| `dataset_root` | `FASHIONIQ_ROOT` |
| `artifacts` | `WORKBENCH_ARTIFACTS_ROOT` |
| `checkpoints` | `WORKBENCH_CHECKPOINT_ROOT` |
| `results` | `WORKBENCH_RESULTS_ROOT` |
| `third_party` | `WORKBENCH_THIRD_PARTY_ROOT` |

Numbered sequence (copy-pasteable version in `A37_PC_HANDOFF.md`):

1. Environment setup — create the controller `.venv`, install `workbench/backend/requirements.txt`
   and the root project; keep per-model environments separate.
2. Configuration — copy `workbench/configs/pc_autopilot.pc_cohort.yaml` to `~/cir_pc.yaml`, set all six
   absolute paths, finite budgets and explicit permissions; set `CIR_REPO_ROOT` and the five mapped
   roots in `workbench/config/workbench.env`; `export WORKBENCH_RUNTIME_ADAPTATION=sm_120` on a
   Blackwell target; run `workbench/scripts/print_roots.py --config ~/cir_pc.yaml` and refuse a
   disagreement.
3. Dataset/checkpoint preparation — `run.sh composition-eval --prepare-data` verifies released files
   against recorded hashes and inventories missing assets; it never downloads one. Six of the nine
   active models are operator-provisioned (csmcir, conesep, habit, intent, hint, tme); only limn,
   encoder and ptha_mtst carry `environment_required` and get `--create --allow-env-install`.
4. ENCODER CUDA probe — `pc.sh --verify-only`, and the step 5c block that resolves the interpreter from
   `<artifacts>/environment_assignment.json` and runs a real CUDA allocation + matmul + synchronisation.
5. ENCODER smoke — `run.sh composition-eval --config ~/cir_pc.yaml --models encoder --smoke`; name
   models explicitly rather than `--models selected`, which resolves all 17 registry entries including
   the 8 deferred ones.
6. Resume verification — repeat the unchanged command and confirm completed stages are reused
   (`[SKIP] resume: already completed with matching fingerprint…`); note that the H3 fix intentionally
   invalidates rank records written under the old stat-based identity.
7. Full run — `run.sh composition-eval --config ~/cir_pc.yaml --models selected --subsets full,sf,validated --resume`
   on one line, after replacing `selected_models: all_registered` with a deliberate list.

**The target GPU remains `TARGET_GPU_PENDING`.** No sm_120 execution, capability record or inference has
ever occurred here. Census of `workbench/artifacts/gpu_capabilities/` (5,215 records on disk at
drafting time; the round's snapshot was 5,183): `probe_source` = interpreter 1,018 / override 3,857 /
forged 340; `compute_capability` = 8.6 for 9 records, 8.9 for 3,181, absent for 2,025; **zero records
report 12.0**, and the 9 verified interpreter-probed records are all sm_86. The gate at
`backend/runtime.py:2470-2493` compares the declared target against `torch.cuda.get_arch_list()` and,
when a device is probed, against the device's own capability — it is capability-agnostic and needs no
sm_120 branch. `WORKBENCH_ALLOW_SYNTHETIC_GPU_PROBE` (`runtime.py:1326`, comment `:1324-1325`) is a
test-only hatch; records written under it carry `probe_source: 'override'` and every reader refuses them
unless the variable is set again. A synthetic record MUST NEVER be presented as sm_120 proof.

Selective-Torch policy is registry data: `registry/models.yaml:6-11` (baseline torch 2.7.1 /
torchvision 0.22.1, cu128) and `:1429-1436` (TME expressed as `>=2.7.1,<2.9` / `>=0.22.1,<0.24`).

---

## 9. Remaining blockers

1. **G-01 — no sm_120 evidence.** The target GPU has never been present. This blocks every claim about
   Blackwell execution; it is the reason the handoff stays `TARGET_GPU_PENDING`.
2. **T-07 — the multi-command output-validation branch has no dedicated repository test.** Reachable,
   unfixed, evidence-gated fix deferred. Its only coverage is the strengthened
   `test_mutation_effectiveness.py` detector.
3. **A37 cannot certify per-model native GPU forward or paper-parity on the target hardware.** Nothing
   in this round executed a model forward on sm_120.
4. **The AST invariant rows are structural, with residual ceilings** (see below) — they are not a
   substitute for the behavioural tests.

Explicitly NOT listed as blockers: H2 is documented behaviour; H1 is unreachable in the real path;
P-01 has no realistic trigger; D-01 is documentation drift in a document the round did not own;
S-01 is resolved by the commit series.

### Residual ceilings of the AST rows (T-02)

- `_terminates` / `_live_statements` (`workbench/tests/test_release_invariants.py:329-361`) are
  **terminator tests, not control-flow graphs**. A body whose exit sits behind `while True:`, or is
  reachable only through a `try`, is still judged terminating. The `ponytail:` comment at `:341` states
  this in the source.
- The rows remain **structural**. The behaviour is exercised for real in
  `test_concurrency_ownership.py` and `test_final_hardening_regressions.py:163`.
- One deliberate narrowing, stated honestly: a guard whose only flag read **and** only exit both live in
  an `elif` (e.g. `if <other>: pass / elif self._stopping: continue`) does **not** satisfy the row. That
  is not a spelling the real code uses, and it failed before the tightening too.

### P-01 proposed fix (not applied)

Insert a NUL byte between joined commands in the stage fingerprint. Single-command stages keep
byte-identical digests; multi-command stages get one forced rerun. It remains `CONFIRMED_UNFIXED`
because no realistic pair of pipeline invocations was found to collide — a robustness nit, not a
demonstrated scientific-correctness defect.

---

## 10. Closing statement

The system is **NOT** claimed to be bug-free. This round fixed what it could reproduce with evidence,
recorded what it could not (P-01, T-07), recorded a negative result that killed a tempting optimization
(H3b), and declined to certify hardware it has never run on. Findings H2, H1 and the adapter matrix are
recorded as expected behaviour on the strength of the cited code, not on execution of the target
capability.
