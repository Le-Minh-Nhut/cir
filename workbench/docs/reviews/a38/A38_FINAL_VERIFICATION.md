# A38 Final Verification Report

Independent verification of the A38 audit round on `/home/heheboiz/data/cir`, branch `research/cir-failure-workbench`.

## Discrepancies
no discrepancies found

*Notes on conventions:*
- `workbench/docs/PC_PILOT_HANDOFF.md:61` pins `git checkout 3786e1d` (7-character short hex SHA rather than 40-character full hex SHA). It resolves unambiguously to `3786e1d23db369c4869dd52c0e04e1530f2aba59` and satisfies production-tree equivalence predicate 2.
- `A38_AUDIT_REPORT.md:25` records baseline HEAD `a8af3d2df45cb6d0aff8775cd3402f8c97b742cf` representing pre-commit audit state. Authoritative live tip is specified via `git rev-parse HEAD` in `FINAL_PC_RELEASE_VERDICT.md` and `A38_PC_HANDOFF.md`.

---

## A. Git State
- **Local HEAD:** `8a0e8778cc707d7ece28617c9b05015038f40771`
- **Current branch:** `research/cir-failure-workbench`
- **Recent commits (`git log --oneline -8`):**
  - `8a0e877` docs(A38): advance the tested source revision to the A38 production commit
  - `32df34f` docs(A38): record the audit report, findings ledger, coverage matrices and the corrected PC handoff
  - `99e4059` test(A38): pin fingerprint framing, rank-payload validation, feature-write locking, CSMCIR key policy and comment-insensitive release invariants
  - `3786e1d` fix(A38): make the pipeline fingerprint injective, validate rank-cache payloads, lock concurrent feature writes, harden the CSMCIR checkpoint load and cut the /api/models cost
  - `a8af3d2` docs(A37): make the ledger git rows drift-immune
  - `b53a2ac` docs(A37): make the commit-count row count-free
  - `4b3b66c` docs(A37): remove the last pinned-tip sentence
  - `838e367` docs(A37): drop the last pinned-tip references from the report
- **Remote tracking status:**
  - `origin/refs/heads/research/cir-failure-workbench`: `8a0e8778cc707d7ece28617c9b05015038f40771` (identical to local HEAD)
  - `origin/refs/heads/main`: `8ebcc74d5fd1934df4f3772a8ca5a96dbd00f333` (unchanged)
- **Commits ahead of anchor `e06b4e61d894cd895034838b126ad8046a0602a5`:** `19`
- **Working tree status (`git status --porcelain=v1`):** Exactly 3 untracked operator files, 0 modified tracked files:
  - `?? pc_composition_smoke.yaml`
  - `?? src/train_cirr.py`
  - `?? uv.lock`
- **Operator SHA-256 validation:**
  - `pc_composition_smoke.yaml`: `aa4ccd6f405791662f8be2442e3a93e735bd2b81753c79815c30506638cf369e` (MATCH)
  - `src/train_cirr.py`: `bb394437c13be97f3ff041274a8952c6054afc425308ef5f6edbc092abfcefb4` (MATCH)
  - `uv.lock`: `fba52f424b9fed6776dd04c656c9024a7b6abe329aa2eb96a58adbd9454b4325` (MATCH)
- **Four A38 commits partitioning:**
  1. `3786e1d` (Production): Touched exactly 6 production files: `workbench/backend/main.py`, `workbench/backend/registry.py`, `workbench/backend/runtime.py`, `workbench/composition/model_csmcir.py`, `workbench/composition/runner.py`, `workbench/scripts/pipeline.py`.
  2. `99e4059` (Tests): Touched exactly 4 test files: `workbench/tests/test_composition_image_identity.py`, `workbench/tests/test_csmcir_checkpoint.py`, `workbench/tests/test_pipeline.py`, `workbench/tests/test_release_invariants.py`.
  3. `32df34f` (Docs): Touched exactly 21 documentation files: `workbench/docs/PC_PILOT_HANDOFF.md` and 20 files under `workbench/docs/reviews/a38/**`.
  4. `8a0e877` (Release metadata advance): Touched exactly 2 metadata files: `workbench/docs/PC_PILOT_HANDOFF.md`, `workbench/docs/reviews/FINAL_PC_RELEASE_VERDICT.md`.
  - No operator files touched. No file placed in incorrect commit group.
- **A37 review archive integrity:**
  - `git diff --stat 3786e1d^ HEAD -- workbench/docs/reviews/a37/`: empty.
  - A37 review subtree was not modified in A38.

---

## B. The Release Guard
- **Named `TESTED_SOURCE_SHA` in `FINAL_PC_RELEASE_VERDICT.md`:** `3786e1d23db369c4869dd52c0e04e1530f2aba59` (line 22 and line 38).
- **Predicate 2 executions:**
  - `git diff --name-only 3786e1d23db369c4869dd52c0e04e1530f2aba59 -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'`: `rc=0`, output empty.
  - `git diff --name-only 3786e1d23db369c4869dd52c0e04e1530f2aba59 HEAD -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'`: `rc=0`, output empty.
- **Ancestry check:**
  - `git merge-base --is-ancestor 3786e1d23db369c4869dd52c0e04e1530f2aba59 HEAD`: `rc=0`, `ancestor-ok`.
- **Handoff commands test gate:**
  - `pytest -q workbench/tests/test_handoff_commands.py`: `8 passed in 8.74s` (`rc=0`).
- **Checkout pins in `workbench/docs/PC_PILOT_HANDOFF.md`:**
  - Line 61: `git checkout 3786e1d`
  - Resolves to commit `3786e1d23db369c4869dd52c0e04e1530f2aba59`.
  - Predicate 2 against working tree: `rc=0`, empty diff.
  - Predicate 2 against HEAD: `rc=0`, empty diff.

---

## C. Deliverable Consistency
- **Files present in `workbench/docs/reviews/a38/`:**
  - `A38_ACCEPTANCE_REVIEW.md`
  - `A38_AUDIT_REPORT.md`
  - `A38_BASELINE.md`
  - `A38_BRANCH_MATRIX.csv`
  - `A38_DOC_DRIFT.csv`
  - `A38_DOC_ERRATUM.md`
  - `A38_EVIDENCE_INDEX.md`
  - `A38_FEATURE_MATRIX.csv`
  - `A38_FINDINGS.csv`
  - `A38_INDEPENDENT_REVIEW.md`
  - `A38_PC_HANDOFF.md`
  - `A38_SCIENTIFIC_VALIDATION.md`
  - `A38_SOURCE_COVERAGE.csv`
  - `A38_TEST_RESULTS.md`
  - `A38_UI_VALIDATION.md`
  All 15 required deliverables present.
- **CSV shapes and field-count invariance:**
  - `A38_FINDINGS.csv`: 21 lines (20 data rows), field count set `{13}`.
  - `A38_SOURCE_COVERAGE.csv`: 361 lines (360 data rows), field count set `{10}`.
  - `A38_FEATURE_MATRIX.csv`: 36 lines (35 data rows), field count set `{10}`.
  - `A38_BRANCH_MATRIX.csv`: 78 lines (77 data rows), field count set `{10}`.
  - `A38_DOC_DRIFT.csv`: 13 lines (12 data rows), field count set `{9}`.
  Each CSV contains exactly one distinct field count.
- **Findings ledger counts and cross-check:**
  - Total findings in `A38_FINDINGS.csv`: 20 data rows.
  - Status counts: `CONFIRMED_FIXED`: 10, `INFORMATIONAL`: 5, `EXPECTED_BEHAVIOR`: 2, `BLOCKED`: 2, `NOT_REPRODUCED`: 1.
  - Severity counts: `HIGH`: 5, `LOW`: 4, `MEDIUM`: 3, `INFO`: 7, `N/A`: 1.
  - HIGH findings: `C-01`, `C-03`, `R-A38-01`, `S-01`, `T-A38-01`.
  - Cross-check against `A38_AUDIT_REPORT.md` summary table: 20 findings in report match 20 findings in CSV with 0 mismatches in ID, severity, or status.
- **`A38_TEST_RESULTS.md` accounting & report decoupling:**
  - Per-chunk execution:
    - Chunk 1: 38 files, 631 collected, 629 passed, 0 failed, 2 skipped, 0 errors, 1126.58s
    - Chunk 2: 37 files, 490 collected, 489 passed, 0 failed, 1 skipped, 0 errors, 838.03s
    - Chunk 3: 37 files, 580 collected, 579 passed, 0 failed, 1 skipped, 0 errors, 1394.71s
    - Chunk 4: 37 files, 474 collected, 467 passed, 1 failed, 6 skipped, 0 errors, 1263.39s
  - Aggregate explicitly labelled: `PER-CHUNK SUM, not a single-process run: 2164 passed, 1 failed, 10 skipped, 0 errors of 2175 collected`.
  - Single failure documented as by-construction guard: `workbench/tests/test_handoff_commands.py::test_no_release_document_points_at_a_superseded_source_sha` failing as expected against dirty tree comparing to historical A37 SHA.
  - `A38_AUDIT_REPORT.md` states no full-suite pass/fail total; explicitly points to `A38_TEST_RESULTS.md` as sole authority.
- **Tip pinning and staleness check:**
  - `A38_AUDIT_REPORT.md:25` pins `a8af3d2df45cb6d0aff8775cd3402f8c97b742cf` as pre-commit audit baseline.
  - `A38_PC_HANDOFF.md:22` and `FINAL_PC_RELEASE_VERDICT.md:43` explicitly instruct authoritative-live resolution via `git rev-parse HEAD`.

---

## D. Fix Mechanisms in Committed Tree
1. `workbench/scripts/pipeline.py` (lines 301-316):
   - Injective fingerprint serializer via length-prefixed encoding: `hasher.update(str(len(raw)).encode("ascii") + b":" + raw)`.
   - `stage.name` is length-prefixed via `update("STAGE", stage.name)` instead of raw byte feeding.
2. `workbench/composition/runner.py`:
   - `_rank_payload_is_valid` (lines 67-73): Validates conditions match, `set(ranks[condition]) == qids`, and all ranks are either `None` or `type(rank) is int and 1 <= rank <= gallery_count`.
   - Per-model lock (lines 76-90, 689-691, 876-878, 1013-1015): `_acquire_model_lock` opens `.model_id_checkpoint_id.lock` with `fcntl.flock(descriptor, fcntl.LOCK_EX)`. Released via `_release_model_lock` on completion (line 877) and in outer `finally:` block (line 1014).
3. `workbench/composition/model_csmcir.py` (lines 196-218):
   - `_check_checkpoint_load`: Extracts `missing_keys` and `unexpected_keys` from `report = model.load_state_dict(state, strict=False)`.
   - Unexpected keys logged to stderr as tolerated.
   - Missing keys raise `ValueError("CSMCIR checkpoint does not match the model: ...")`.
4. `workbench/backend/registry.py`:
   - `_YAML_PARSE_CACHE` (lines 28-34) and `_VALIDATED_CONFIG_CACHE` (line 35, 161, 193, 199, 253).
   - Functions return isolated copies via `copy.deepcopy(...)` (lines 136, 162, 194, 200, 254).
   - `_freeze_config` and `MappingProxyType` completely removed.
5. `workbench/backend/runtime.py` (lines 1751-1763):
   - `_member_digest` remains uncached, directly returning `sha256_file(_Path(path))`.
   - Docstring documents why: in-place rewrites preserve `(path, size, mtime)` via `os.utime`, making cached memos vulnerable to stale digest reuse.

---

## E. Verdict
Delivered state is fully consistent with claims made in `A38_AUDIT_REPORT.md` and release metadata. All gates pass.
