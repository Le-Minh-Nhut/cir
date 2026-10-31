# A38 Acceptance Review

**Reviewer:** Independent Acceptance Reviewer  
**Date:** 2026-10-31  
**Repository:** `/home/heheboiz/data/cir`  
**Branch:** `research/cir-failure-workbench`  
**HEAD:** `a8af3d2df45cb6d0aff8775cd3402f8c97b742cf`  
**Tree State:** Dirty (10 modified tracked files, 4 untracked operator files / tests)

---

## Strongest Single Objection

**S-01 claimed `CONFIRMED_FIXED` despite zero empirical checkpoint key verification and direct upstream code proving tolerated missing keys.**  
In `workbench/composition/model_csmcir.py:185-197`, `_check_checkpoint_load` raises a fatal `ValueError` on ANY missing or unexpected state key. However, upstream author code in vendored `workbench/third_party/CSMCIR/src/validate_blip_csmcir.py:525-526` and `blip_validate.py:49-50` deliberately executes `blip_model.load_state_dict(checkpoint[...], strict=False)` and logs `print("Missing keys {}".format(msg.missing_keys))`. In complex multi-modal architectures (such as BLIP-2 with prompt alignment), frozen vision towers, language backbones, or prompt tokens are routinely omitted from task checkpoint state dicts. The repository has never downloaded the actual 2.72 GB public checkpoint (`workbench/artifacts/checkpoints/csmcir` is empty/absent), has no tensor key manifest, and has never loaded the model with real weights on this host. The regression test in `test_csmcir_checkpoint.py` only validates against a 3-line synthetic `FakeModel`. If the legitimate public checkpoint relies on the upstream-tolerated omissions, the round's fix converts a functioning model into an immediate crash. This fix substitutes an unverified risk of silent partial loading with an unverified regression risk that breaks legitimate weights, while overclaiming `CONFIRMED_FIXED`.

---

## Detailed Question Responses

### Q1: Fixed-Claims Verification

Each of the 10 findings marked `CONFIRMED_FIXED` in `workbench/docs/reviews/a38/A38_FINDINGS.csv` was verified against `git diff -- workbench/`:

1. **A38-01 (Pipeline command-fingerprint collision)**:
   - *Claim:* Command boundary fixed, whole serializer made injective.
   - *Code diff:* `workbench/scripts/pipeline.py:302-315` introduced `update(*values)` which formats each value with length-prefix framing (`str(len(raw)).encode("ascii") + b":" + raw`). Applied across `STAGE`, `MODEL`, `CKPT`, `PROTO`, `COMMAND`, and all subsidiary sections.
   - *Verification:* The length prefix separates token names and fields unambiguously. The prior counterexample (`sMODEL:abc` vs `s` with `model_id="abc"`) now yields distinct digests. Tested in `test_pipeline.py::test_fingerprint_fields_are_unambiguously_framed`.
   - *Status:* `SOUND`.

2. **A38-02 (Multi-command stage output-validation test gap)**:
   - *Claim:* Tests only, production unchanged.
   - *Code diff:* `workbench/tests/test_pipeline.py:312-390` added `test_multi_command_output_validation_resume`.
   - *Verification:* The test drives 8 cases with multiple commands through `run_stages` with real validation. It explicitly verifies that when output artifacts are missing, empty, or corrupt, `run_stages` returns rc=1 and records `FAILED`, even when both commands exit 0. It also verifies with a noop validator that bypassing validation yields rc=0 `COMPLETE`, proving the test is sensitive to post-loop validation at `pipeline.py:2389`.
   - *Status:* `SOUND`.

3. **A38-03 (PC resume instructions omitted opt-in `--resume`)**:
   - *Claim:* Updated `A38_PC_HANDOFF.md`; historical A37 archive untouched.
   - *Code diff:* `workbench/docs/reviews/a38/A38_PC_HANDOFF.md` step 9 and erratum; `workbench/docs/reviews/a38/A38_DOC_ERRATUM.md`.
   - *Verification:* Commands include `--resume`. A37 ledger and review documents were preserved without retrospective edits.
   - *Status:* `SOUND`.

4. **A38-05 (Release metadata and PC handoff documentation drift)**:
   - *Claim:* Corrected confirmed drift; retained ambiguous items as unverified; archived A37 untouched.
   - *Code diff:* `workbench/docs/PC_PILOT_HANDOFF.md:105-110, 118, 257-263, 350-355` updated.
   - *Verification:* Accurately reflects that ENCODER and LIMN declare observers (15 models aggregate-only) and clarifies `CAPABILITY_ABSENT_FOR_DIAGNOSTIC` as an observation note / export reason.
   - *Status:* `SOUND`.

5. **C-01 (Rank cache accepts malformed payloads)**:
   - *Claim:* Added `_rank_payload_is_valid` validation.
   - *Code diff:* `workbench/composition/runner.py:67-73` implements `_rank_payload_is_valid(ranks, conditions, qids, gallery_count)`. Called at `:701` before reusing rank cache.
   - *Verification:* Enforces dictionary structure, matching condition sets, exact query ID sets, and integer ranks within `[1, gallery_count]` (or `None`). Verified by `test_composition_image_identity.py:46-107`.
   - *Status:* `SOUND`.

6. **C-03 (Concurrent feature memmap writers interleave)**:
   - *Claim:* Added per-model `fcntl.flock`; fixed fd leak on flock exception.
   - *Code diff:* `workbench/composition/runner.py:76-88` implements `_acquire_model_lock` and `_release_model_lock`.
   - *Verification:* Acquired at `:689` and released at `:876` and `:1013` (in `finally:`). `try...except BaseException: os.close(descriptor); raise` prevents fd leaks if flock raises.
   - *Status:* `SOUND_WITH_CAVEAT` (Caveat: relies on filesystem flock support; ineffective/no-op network locks are undetected).

7. **R-A38-01 (MappingProxyType regression in registry caching)**:
   - *Claim:* Changed to `copy.deepcopy` on every path; parse win retained.
   - *Code diff:* `workbench/backend/registry.py:133, 160, 193, 225, 253` returns `copy.deepcopy(...)` instead of `MappingProxyType`.
   - *Verification:* Resolves `TypeError` during `json.dumps` serialization in `prepare_model.py`.
   - *Status:* `SOUND`.

8. **S-01 (CSMCIR checkpoint incompatibility silently discarded)**:
   - *Claim:* Changed loader to fail on any missing/unexpected key.
   - *Code diff:* `workbench/composition/model_csmcir.py:185-197, 215-225`.
   - *Verification:* Fails closed on any discrepancy reported by `load_state_dict(..., strict=False)`.
   - *Status:* `NOT_SOUND` (high regression risk on unverified real weights; upstream author code explicitly used `strict=False` and printed missing keys).

9. **T-A38-01 (Release invariants satisfied by comments)**:
   - *Claim:* Hardened Python targets to AST checks and shell targets to quote-aware `_shell_code` scan.
   - *Code diff:* `workbench/tests/test_release_invariants.py:61-89` adds `_shell_code(text)` and AST-based node checks for I11, I13, I14, I17, I19, I20.
   - *Verification:* Comment-only mutations fail.
   - *Status:* `SOUND_WITH_CAVEAT` (Caveat: `_shell_code` scan mis-strips comments on line continuations inside quotes; exact AST string assertions are brittle to formatting).

10. **UI-01 (`/api/models` latency)**:
    - *Claim:* Warm 26.3→8.5 s; cold 89.7→69.3 s; `_member_digest` left uncached.
    - *Code diff:* `workbench/backend/registry.py:20-40` added `_YAML_PARSE_CACHE` and `_VALIDATED_CONFIG_CACHE`; `workbench/backend/main.py:162-178` hoisted model preflight blockers; `workbench/backend/runtime.py:2920` accepts precomputed blockers.
    - *Verification:* Eliminates redundant YAML parsing (307 parses over 3 files) and hoists per-model requirements out of variant loops.
    - *Status:* `SOUND_WITH_CAVEAT` (Caveat: cache key `(path, size, mtime_ns)` can serve stale config if an in-place edit preserves size and mtime).

---

### Q2: Overclaim Probes

1. **A38-04 (GPU capability workspace isolation — EXPECTED_BEHAVIOR)**:
   - *Investigation:* `probe_gpu_capability` in `workbench/scripts/pipeline.py:1428-1455` accepts `config: WorkbenchConfig | None = None`. At line 1438, `runtime._cached_gpu_capability(model, report)` is called without passing `config`, so it resolves `resolve_config()` (the default). In pipeline stages, `pipeline.py:1774` launches `probe_gpu_capability` as a child subprocess via `python -c "import sys, workbench.scripts.pipeline as p; sys.exit(p.probe_gpu_capability(sys.argv[1]))" mid`. The CLI parser in `pipeline.py:2430-2465` has NO `--config` argument. When `run_command` executes child subprocesses, it inherits `os.environ`. In `workbench/backend/operator_config.py:135`, `WORKBENCH_ARTIFACTS_ROOT` set in the environment overrides the default file path and is inherited by all children.
   - *Verdict:* `EXPECTED_BEHAVIOR` holds for all supported CLI entry points. The cross-workspace reuse is reachable only through a direct, in-process Python helper call passing an explicit config object that differs from `os.environ`.

2. **A38-02 (Multi-command output validation — CONFIRMED_FIXED)**:
   - *Investigation:* `test_pipeline.py::test_multi_command_output_validation_resume` constructs a stage with 2 commands (`["first"], ["second"]`). Inside `run_stages` (`pipeline.py:2362`), single-command validation is skipped because `len(stage.commands) > 1`. Post-loop validation occurs at line 2389:
     ```python
     if not stage_planning and not validate_stage_outputs(stage, config=cfg, record=stage_records.get(skey)):
     ```
     The test runs cases where both commands succeed (exit 0) but output files are missing, empty, or malformed JSON. The test asserts `rc == 1` and `record["status"] == "FAILED"`. If line 2389 were removed, both commands would exit 0, `stage_failed` would remain `False`, the stage would record `COMPLETE`, and the test would fail.
   - *Verdict:* `CONFIRMED_FIXED` holds.

3. **T-01 (Runtime adaptation config setting rejected — NOT_REPRODUCED)**:
   - *Investigation:* In `workbench/backend/model_settings.py:34-42`, `_declared_names()` imports `ADAPTATION_ENV` (`WORKBENCH_RUNTIME_ADAPTATION`) from `workbench.backend.runtime` and adds it to `names`. In `workbench/backend/operator_config.py:47-53`, `_permitted_keys()` unions `_KEYS` with `declared_setting_names()`. Direct Python evaluation confirms `"WORKBENCH_RUNTIME_ADAPTATION" in _permitted_keys()` is `True`. Running `pytest workbench/tests/test_autopilot_e2e.py -k "test_a_dry_run_changes_nothing"` passed in 2.75s.
   - *Root cause of prior failure:* `_declared_names()` wraps imports in `try...except Exception: pass`. During W6 chunk 1, `workbench/backend/registry.py` had a transient syntax/NameError during concurrent edits. This caused `_declared_names()` to degrade silently and return a partial set, omitting registry-derived settings and causing `operator_config` to reject `WORKBENCH_RUNTIME_ADAPTATION`.
   - *Verdict:* `NOT_REPRODUCED` on the frozen tree holds. However, the silent `except Exception: pass` in `model_settings.py` is a latent vulnerability that masks import errors.

4. **B-02 / W8-01 (Backend job lifecycle & concurrency — INFORMATIONAL / EXPECTED_BEHAVIOR)**:
   - *Investigation:* In `A38_FINDINGS_PARTIAL_W8.md`, the job lifecycle checks (`SingleGpuQueue`) executed successfully with mock children (`sleep` / `exit 0` / `exit 7`), not real model evaluations. The concurrency check for pipeline state (`run_stages` lock) successfully demonstrated `WorkflowLockError` on concurrent invocation. The GPU capability concurrency check used dummy payloads and disabled cryptographic signing to force key collisions. Crucially, the DuckDB index rebuild concurrency check was executed ONLY on an empty results root (0 rows); W8 explicitly noted in its blockers that non-empty merge equivalence remains `STATICALLY_CHECKED`.
   - *Verdict:* The status classifications hold, but the claims of "executed" operational concurrency are qualified: index merge and heavy evaluation paths were synthetic or empty.

---

### Q3: Unrecorded Consequences of the Round's Own Fixes

1. **CSMCIR Loader Key Strictness (`workbench/composition/model_csmcir.py`)**:
   `_check_checkpoint_load` raises on any missing or unexpected key. Upstream `validate_blip_csmcir.py` and `blip_validate.py` explicitly load with `strict=False` and log missing keys. Because real model weights were never inspected, this fix risks breaking working evaluations if legitimate checkpoints omit non-critical weights.
2. **Pipeline Fingerprint Serialization Breakage (`workbench/scripts/pipeline.py`)**:
   The length-prefixed format alters the calculated digest for all stages. On resume, existing stage receipts will fail fingerprint equality (`prev.get("fingerprint") == fingerprint` is `False`). Tracing `pipeline.py:2285-2315` confirms that `can_skip` evaluates to `False`, forcing a clean rerun of all pipeline stages. This does not crash or silently skip, but invalidates existing valid stage caches.
3. **In-Place File Edits with Unchanged Size and Mtime (`workbench/backend/registry.py`)**:
   The caching mechanism uses `(str(path), stat.st_size, stat.st_mtime_ns)`. An in-place modification that preserves length and restores mtime (or occurs within the same timestamp resolution) will serve stale parsed configuration until process termination.
4. **Silent Exception Masking in Settings Declaration (`workbench/backend/model_settings.py`)**:
   `_declared_names()` catches all exceptions with `pass`. If `registry.py` or `runtime.py` introduces an import error, settings like `WORKBENCH_RUNTIME_ADAPTATION` silently vanish from `_permitted_keys()`, generating misleading config-parsing failures instead of reporting the underlying syntax or import defect.
5. **Lock Scope Contention in Composition Runner (`workbench/composition/runner.py`)**:
   `_acquire_model_lock` holds an exclusive flock across the entire model feature encoding and evaluation loop. While self-deadlock is prevented, concurrent evaluation of the same model/checkpoint across processes is fully serialized.

---

### Q4: Coverage Mislabels

1. **`A38_SOURCE_COVERAGE.csv` Mislabeling:**
   All 360 rows declare `inspection_status = NOT_REVIEWED` using `inspection_method = ls-files`. This misrepresents the audit by declaring that core files (`workbench/scripts/pipeline.py`, `workbench/backend/main.py`, `workbench/backend/registry.py`, `workbench/composition/runner.py`) were never reviewed, even though they were extensively audited, modified, and verified.
2. **`A38_BRANCH_MATRIX.csv` Duplicate Branch:**
   Feature `F07` (`evaluate_models.py`) contains duplicate entries for `branch_id = retry`:
   - Line 28: `F07, ..., retry, retry failed model evaluator automatically, ..., NOT_APPLICABLE`
   - Line 71: `F07, ..., retry, continue-on-error enabled for multi-model selection, ..., NOT_TESTED`
3. **`A38_SCIENTIFIC_VALIDATION.md` Dressing Static Inspection as `PASS`:**
   In Table T4 (Dataset integrity and leakage), four items are labeled `PASS`:
   - `Release path/split identity` (Method: "Read loader and selection")
   - `Category boundary` (Method: "Reference stem against FashionIQ validation splits/test annotations")
   - `Gallery membership/order` (Method: "Loader validates exact basename equality...")
   - `Full gallery on subset run` (Method: "Read query selection + run")
   Static code reading and data schema inspection are labeled `PASS` rather than `STATICALLY_CHECKED`, conflating static review with executed test verification.

---

### Q5: Verdict Supported by Evidence

**Verdict Supported:** `A38_ACCEPTED_WITH_BLOCKERS`  
**Strict Audit Alternative:** `A38_NOT_ACCEPTED`

**Evidence Contradicting `A38_ACCEPTED`:**
1. **Target GPU (Blackwell sm_120) Never Run (`G-01`, BLOCKED):** Zero executions, capability records, or kernel inferences exist for the target RTX 5070 Ti sm_120. All runs occurred on an RTX 3050 Laptop GPU (sm_86).
2. **Test Suite Incomplete and Abandoned (`T-02`, BLOCKED):** The full test suite was stopped after Chunk 1 due to 16 failures. Chunks 2, 3, and 4 (comprising 1,754 of the 2,175 collected tests, or 80.6%) were never executed. Four apply-mode autopilot tests fail due to missing summary artifacts.
3. **Uncommitted Dirty Tree:** Production modifications remain uncommitted in the working tree, causing release identity guards (`test_handoff_commands.py::test_no_release_document_points_at_a_superseded_source_sha`) to fail.
4. **Unsettled Checkpoint Key Integrity (`S-01`):** The CSMCIR strict checkpoint loader has never been validated against the actual 2.72 GB model weights, introducing unmitigated risk of crashing on legitimate upstream weights.

Any claim of `A38_ACCEPTED` is demonstrably false and contradicted by repository evidence.

---

## Objections Table

| ID | Finding / Area | Claim Objected To | Evidence | Classification |
|---|---|---|---|---|
| OBJ-01 | S-01 / CSMCIR loader | `CONFIRMED_FIXED` via strict `load_state_dict` key check | `validate_blip_csmcir.py:525-526` explicitly tolerated missing keys with `strict=False`; real 2.72 GB checkpoint never downloaded or tested; mock test only uses 3-line dummy. | `CONFIRMED_OBJECTION` |
| OBJ-02 | Test Execution | Complete test suite verification | Full suite stopped after Chunk 1; 1,754 of 2,175 tests never run; 4 apply-mode autopilot tests fail (`T-02`, BLOCKED). | `CONFIRMED_OBJECTION` |
| OBJ-03 | Scientific Validation | T4 checks marked `PASS` | Method column documents "Read loader and selection", "Reference stem against splits", etc. Inspection dressed as execution. | `CONFIRMED_OBJECTION` |
| OBJ-04 | W8 Concurrency | Concurrency and recovery fully established by execution | DuckDB rebuild executed on 0-row empty root only (merge equivalence acknowledged as static); capability test disabled HMAC signing. | `UNCONFIRMED_OBJECTION` |
| OBJ-05 | T-01 / Settings | T-01 is transient; no code improvement required | `model_settings.py:41-43` uses `except Exception: pass`, silently hiding core module import/syntax errors behind invalid config setting errors. | `UNCONFIRMED_OBJECTION` |
| OBJ-06 | UI-01 / Cache Key | Safe config caching | `(path, size, mtime_ns)` key serves stale parsed YAML if an in-place edit preserves size and mtime. | `NITPICK` |
| OBJ-07 | Branch Matrix | Clean branch classification | Feature F07 has duplicate `branch_id = retry` entries (lines 28 and 71) with conflicting statuses (`NOT_APPLICABLE` vs `NOT_TESTED`). | `NITPICK` |
| OBJ-08 | Source Coverage | Complete repository coverage accounting | 360 of 360 files (100%) marked `NOT_REVIEWED` with `ls-files`, including all files audited and modified during this round. | `NITPICK` |

**Summary Counts:**
- `CONFIRMED_OBJECTION`: 3
- `UNCONFIRMED_OBJECTION`: 2
- `NITPICK`: 3
- **Total Objections:** 8

---

## Per-Fix Table

| Finding ID | Title | Assessment | Notes |
|---|---|---|---|
| A38-01 | Pipeline command-fingerprint collision | `SOUND` | Length-prefixed framing in `update()` makes serialization injective; invalidates old fingerprints safely. |
| A38-02 | Multi-command stage output-validation test gap | `SOUND` | Regression test drives real multi-command execution and fails if post-loop validation is removed. |
| A38-03 | PC resume instructions omitted `--resume` | `SOUND` | Flag added to documented handoff commands; historical archives untouched. |
| A38-05 | Release metadata and PC handoff documentation drift | `SOUND` | Observer declarations (ENCODER/LIMN) and export reason statuses corrected. |
| C-01 | Rank cache accepts malformed payloads | `SOUND` | `_rank_payload_is_valid()` validates conditions, qid sets, and rank range `[1, gallery_count]`. |
| C-03 | Concurrent feature memmap writers interleave | `SOUND_WITH_CAVEAT` | Per-model `fcntl.flock` with exception handling prevents races; relies on filesystem supporting `flock`. |
| R-A38-01 | MappingProxyType regression in registry caching | `SOUND` | Replaced with `copy.deepcopy()`, restoring JSON serialization while retaining parsing win. |
| S-01 | CSMCIR checkpoint incompatibility silently discarded | `NOT_SOUND` | Fails on any key discrepancy; unverified against real 2.72 GB weights; conflicts with author's `strict=False`. |
| T-A38-01 | Release invariants satisfied by comments | `SOUND_WITH_CAVEAT` | Hardened with AST checks and `_shell_code`; scanner brittle to quoted continuations; string checks brittle. |
| UI-01 | `/api/models` latency | `SOUND_WITH_CAVEAT` | Hoisted model preflight and memoized YAML; cache key assumes edits change size or mtime. |
