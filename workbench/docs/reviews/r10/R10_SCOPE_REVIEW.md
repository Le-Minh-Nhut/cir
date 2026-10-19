# R10 SCOPE, COHERENCE, AND HONESTY REVIEW

Audit of R10 changes (`git diff e46ad65..cf84cfa` and successor documentation commits up to `HEAD`).

## Executive Summary

- Native evaluation semantics: **VERIFIED**. Zero alterations to gallery, reference masking, ranking, metric units, preprocessing, prompts, macro arithmetic, or protocol identity.
- Scope and Abstractions: **VARIANT**. Redundant two-variable launcher representation in `Adapter`, dead helper `reset()`, and architecturally incorrect Conda inheritance branch.
- Test and Regression Surface: **VARIANT**. 11 of 12 test suites passed. `test_handoff_commands.py` failed (`rc=1`) due to production tree drift after `cf84cfa` (`3a91870`).
- Report Honesty: **VARIANT**. Status spec and remediation document honestly note real environments were uninstalled and distinguish fixtures from scientific evidence. However, claim that `git diff cf84cfa HEAD -- . ":(exclude)workbench/docs"` is empty is factually false.
- Project Promises: **VERIFIED**. Untested boundary-crossing surface remains unchanged.
- Fresh Operator Runbook: **VARIANT**. Step 0 omits `open-clip-torch` and `tqdm` required by Step 6 commands, while declaring unused API dependencies (`fastapi`, `uvicorn`, `pydantic`, `httpx`).

---

## 1. Unnecessary Abstraction

### 1.1 `Adapter.launcher` and `launcher_roots`
- **Verdict**: VARIANT
- **Severity**: P3
- **Evidence**:
  - `workbench/backend/adapters/models.py:25-27`:
    ```python
    launcher: str = "script"
    launcher_roots: tuple[str, ...] = ()
    ```
  - `OfficialScriptAdapter.build_command` and `_replay_command` (`workbench/backend/adapters/models.py:31-63`) do not read `adapter.launcher` or `adapter.launcher_roots`.
  - Sole reader across repo: `workbench/backend/environment_planner.py:256, 265`:
    ```python
    launcher = getattr(adapter, "launcher", "script")
    if launcher == "replay":
        entry = WORKBENCH_ROOT / "replay" / f"{model['model_id']}.py"
        roots = [WORKBENCH_ROOT.parent, source]
        roots += [source / extra for extra in getattr(adapter, "launcher_roots", ())]
    ```
  - `launcher` is binary (`"script"` vs `"replay"`). `launcher_roots` is only read when `launcher == "replay"`.
- **Finding**: Two separate class attributes represent a single state. Redundant representation for a single call site in an AST scanner.
- **Recommended Action**: Replace with single optional field `replay_roots: tuple[str, ...] | None`.

### 1.2 `lavis_processors.reset()`
- **Verdict**: VARIANT
- **Severity**: P3
- **Evidence**:
  - `workbench/replay/lavis_processors.py:140-146`:
    ```python
    def reset() -> None:
        \"\"\"Tear the package context down so another checkout can be prepared in this process.
        A production wrapper runs one model in one subprocess and never calls this
        (`evaluate_models.py:594`). It exists for tools that deliberately compare several checkouts --
        `workbench/scripts/lavis_processor_parity.py` is the one caller...
        \"\"\"
    ```
  - Production callers: **0**.
  - Callers: `workbench/scripts/lavis_processor_parity.py:73` (diagnostic tool) and unit test fixtures.
- **Finding**: Dead code in production path; test/tooling scaffold placed in production replay module.
- **Recommended Action**: Move to diagnostic script or label strictly internal to test harnesses.

### 1.3 `_inherits_base_site_packages` Conda Branch
- **Verdict**: VARIANT
- **Severity**: P3
- **Evidence**:
  - `workbench/backend/runtime.py:397-399`:
    ```python
    if (prefix / "conda-meta").is_dir():
        return True
    ```
  - Caller in `workbench/backend/runtime.py:615`:
    ```python
    if base_prefix and str(base_prefix) != str(prefix):
        if _inherits_base_site_packages(Path(prefix), Path(base_prefix)):
    ```
  - Conda runtime semantics: Standard Conda environments do not set `pyvenv.cfg`; CPython sets `sys.base_prefix == sys.prefix`. Thus `str(base_prefix) != str(prefix)` is `False` in real Conda environments, making this branch unreachable in production.
  - Furthermore, Conda environments are isolated from `base`; they do not inherit site-packages. Returning `True` is semantically invalid.
  - Sole test coverage: mock directory creation in `workbench/tests/test_r10_remediations.py:121`.
- **Finding**: Unreachable in real execution and semantically incorrect for Conda packaging.
- **Recommended Action**: Remove Conda inheritance branch; require explicit demonstration of inheritance.

### 1.4 `_patch_registry_idempotence` vs `_registry_for`
- **Verdict**: VARIANT
- **Severity**: P3
- **Evidence**:
  - `workbench/replay/lavis_processors.py:73-90` (`_registry_for`) and `:95-115` (`_patch_registry_idempotence`).
  - `_registry_for` checks if `library_root` exists and matches `lavis_root`.
  - `_patch_registry_idempotence` monkey-patches `Registry.register_path` to ignore duplicate same-path registrations.
- **Finding**: `_registry_for` alone cannot prevent `lavis/__init__.py:24` from throwing `KeyError` during checkpoint unpickling because native upstream code directly invokes `Registry.register_path`. However, both functions duplicate duplicate-check logic.
- **Recommended Action**: Apply patch first; simplify `_registry_for` to invoke `register_path` directly.

---

## 2. Accidental Scientific Change

- **Verdict**: VERIFIED
- **Severity**: NONE
- **Hunk Analysis (`git diff e46ad65..cf84cfa`)**:
  1. `workbench/backend/adapters/models.py`: Added static declarations (`launcher`, `launcher_roots`). No impact on evaluation command, flags, or data inputs.
  2. `workbench/backend/environment_planner.py`: AST dependency inspection redirected to replay wrapper. No impact on execution.
  3. `workbench/backend/registry.py`: Permitted nested relative paths (depth <= 2) in `canonical_dataset_link`. Validation check only.
  4. `workbench/backend/runtime.py`: Added `_inherits_base_site_packages` gate. Affects preflight distribution inspection, not inference.
  5. `workbench/registry/models.yaml`:
     - Removed stale `source_blocker` on `airknow`, `conesep`, `intent`. Affects registry reporting; checkpoints and weights remain blocked by external prerequisites.
     - Removed `salesforce-lavis==1.0.2` from profiles. Vendored `lavis` is placed first on `sys.path`; shadows PyPI wheel.
     - Added `canonical_dataset_link: dataset/fashioniq` for `dcnet`. Restores native path expectation without modifying model logic.
  6. `workbench/replay/dcnet.py`: Removed non-native `WORKBENCH_DCNET_DATA_DIR` environment override. Native DCNet `ConfigParser` does not read environment variables; removing override forces validator to evaluate actual disk configuration.
  7. `workbench/replay/lavis_processors.py`: Retained module context in `sys.modules` across unpickling; made `register_path` idempotent. Native text preprocessing configuration unchanged (`load_preprocess(cfg.preprocess)`).
  8. `workbench/scripts/prepare_dataset.py`: Creates destination parent directory when staging symlinks. No metric impact.
- **Scientific Invariance**:
  - Gallery construction: Unchanged.
  - Reference masking: Unchanged.
  - Ranking order: Unchanged.
  - Metric units and scaling: Unchanged.
  - Preprocessing transforms: Unchanged.
  - Prompt handling: Unchanged.
  - Macro arithmetic: Unchanged.
  - Protocol identity: Unchanged.
- **Recommended Action**: None.

---

## 3. Regression Surface

Test execution with `.venv/bin/python -m pytest <test_file> --basetemp=/home/heheboiz/.cache/cir-pytest6 -q`:

| Test Suite | Exit Code (rc) | Result | Runtime |
|---|---|---|---|
| `test_environment_planner.py` | 0 | 59 passed | 52.41s |
| `test_evaluate_models.py` | 0 | 32 passed | 9.26s |
| `test_replay.py` | 0 | 41 passed | 7.80s |
| `test_verified_declarations.py` | 0 | 19 passed | 0.69s |
| `test_handoff_commands.py` | **1** | **6 passed, 1 failed** | 6.61s |
| `test_prepare_dataset.py` | 0 | 29 passed | 6.88s |
| `test_r9_defect_regressions.py` | 0 | 57 passed, 4 warnings | 5.07s |
| `test_r10_remediations.py` | 0 | 18 passed, 1 skipped | 10.99s |
| `test_pc_readiness.py` | 0 | 15 passed | 34.68s |
| `test_vendored_source_integrity.py` | 0 | 13 passed | 0.70s |
| `test_fresh_clone_delivery.py` | 0 | 10 passed | 6.14s |
| `test_component_boundaries.py` | 0 | 23 passed | 15.71s |

### Failure Details (`test_handoff_commands.py`)
- **Command Output**:
  ```
  FAILED workbench/tests/test_handoff_commands.py::test_no_release_document_points_at_a_superseded_source_sha
  AssertionError: TESTED_SOURCE_SHA cf84cfa does not contain the production code currently on disk:
  production files changed after the revision the release documents name as tested. Advance the
  recorded source to a revision that includes them, or the handoff installs code that was never gated.
  assert 1 == 0
   + where 1 = CompletedProcess(args=['git', 'diff', '--quiet', 'cf84cfa', '--', '.', ':(exclude)workbench/docs', ':(exclude)workbench/tests'], returncode=1, stdout='', stderr='').returncode
  ```
- **Root Cause**: Commit `3a91870` modified `workbench/docker/mutation_checks.sh` after commit `cf84cfa`. Because `workbench/docker/mutation_checks.sh` is outside `workbench/docs` and `workbench/tests`, the working tree has production-level modifications relative to `cf84cfa`. The release documents (`FINAL_PC_RELEASE_VERDICT.md` and `PC_PILOT_HANDOFF.md`) still claim `TESTED_SOURCE_SHA = cf84cfa`.
- **Severity**: P1
- **Recommended Action**: Advance `TESTED_SOURCE_SHA` in release docs or update path exclusions in the test.

---

## 4. Honesty of the Report

- **Verdict**: VARIANT
- **Severity**: P1
- **Evidence Assessment**:
  1. **Did report claim the six profiles install correctly?**
     - **No**. `R10_REMEDIATION.md` §2 table under "Không chạy được ở đây" explicitly lists: `sáu môi trường cài thật`.
     - `MODEL_STATUS_SPEC.md:279` (R10-05 row) explicitly states: `6 môi trường cài thật chưa chạy (xem §2 của R10_REMEDIATION.md)`.
  2. **Did report distinguish fixture evidence from scientific evidence?**
     - **Yes**. `R10_REMEDIATION.md` §2 explicitly states:
       `Fixture không phải bằng chứng khoa học. Checkpoint fixture chứng minh đường class resolution sống qua bước processor; nó không chứng minh HINT suy luận đúng.`
  3. **Did status spec R10-05 row admit no real environment was installed?**
     - **Yes**: `6 môi trường cài thật chưa chạy (xem §2 của R10_REMEDIATION.md)`.
  4. **Overstated Claim in Release Verdict**:
     - `workbench/docs/reviews/FINAL_PC_RELEASE_VERDICT.md:8` asserts:
       `| Production tree | identical in cf84cfa and the branch tip (verified: git diff cf84cfa HEAD -- . ":(exclude)workbench/docs" is empty) |`
     - Command execution: `git diff cf84cfa HEAD -- . ":(exclude)workbench/docs"`
     - Raw Output:
       ```
       workbench/docker/mutation_checks.sh | 21 +++++++++++++++++++--
       workbench/tests/test_phase1_acceptance_scenarios.py | 7 ++++++-
       2 files changed, 25 insertions(+), 3 deletions(-)
       ```
     - Finding: Claim that production diff is empty is factually false.
- **Recommended Action**: Update release verdict documentation.

---

## 5. The Project's Own Promise

- **Verdict**: VERIFIED
- **Severity**: NONE
- **Analysis**:
  - `workbench/docs/reviews/r10/R10_E2E_SURFACE.md:270-282` §7 enumerates 10 workflow areas and marks untestable promises:
    1. Real Docker host spin-up in fresh setup tests.
    2. Real network socket / HTTP server-client communication across network boundary.
    3. Subprocess execution of DCNet (in-process preflight/parser only).
    4. Subprocess execution of LAVIS family models (in-process AST/parity only).
    5. Real model execution with weights on GPU (only synthetic models tested).
    6. Remote handoff over SSH / remote host.
  - Did R10 change this list?
    - **No**. R10 added no new boundary-crossing tests for these areas.
  - Remaining untested promises:
    - Real container daemon orchestration (`run.sh up`).
    - Socket-level API verification.
    - Subprocess execution of DCNet and LAVIS family wrappers.
    - True GPU inference and metric reproduction.
- **Recommended Action**: Retain explicit boundary-gap disclosures.

---

## 6. Fresh Operator Audit

- **Verdict**: VARIANT
- **Severity**: P2
- **Command Analysis (`MODEL_STATUS_SPEC.md` §6)**:
  - **Step 0**:
    `pip install pytest pyyaml omegaconf "torch" torchvision fastapi uvicorn pydantic httpx packaging`
  - **Steps 1–5**:
    - Step 1: Git clone / checkout.
    - Step 2: Fresh clone delivery tests (`pytest test_fresh_clone_delivery.py test_vendored_source_integrity.py`). Passes.
    - Step 3: Dry-run download scripts (`download_checkpoints.py`, `download_auxiliary_assets.py`). Runnable.
    - Step 4: Asset downloads. Runnable.
    - Step 4b: `prepare_dataset.py` and `dcnet.binding_mismatch` inline check. Runnable; outputs `BOUND OK`.
    - Step 5: `hint.py --preflight-only` and `pair.py --preflight-only`. Runnable; exits 0/1 without crash.
  - **Step 6**:
    - Command:
      ```bash
      python workbench/replay/limn.py --checkpoint-root <dir> --dataset-root <FIQ> --source-root workbench/third_party/LIMN
      python workbench/replay/encoder.py --checkpoint-root <dir> --dataset-root <FIQ> --source-root workbench/third_party/ENCODER
      ```
    - In `workbench/replay/limn.py:168`: `import open_clip`
    - In `workbench/replay/encoder.py:140`: `import open_clip`
    - Upstream `workbench/third_party/LIMN/LIMN/test.py:10`: `import tqdm`
    - Neither `open-clip-torch` nor `tqdm` is installed by Step 0.
    - Result: Fresh operator running Step 6 in the Step 0 virtual environment experiences immediate crash:
      `ModuleNotFoundError: No module named 'open_clip'`.
  - **Unused Dependencies Declared in Step 0**:
    - `fastapi`, `uvicorn`, `pydantic`, `httpx` are declared in Step 0 but never invoked by any command across Steps 1–6. (Download scripts in Steps 3–4 use stdlib `urllib.request`).
- **Recommended Action**:
  - Add `open-clip-torch` and `tqdm` to Step 0 dependencies in `MODEL_STATUS_SPEC.md:288`.
  - Remove or annotate server-only dependencies (`fastapi`, `uvicorn`, `pydantic`, `httpx`).

---

## Defects found

### Defect 1: P1 — Release document SHA mismatch causing regression failure
- **File:Line**: `workbench/docs/reviews/FINAL_PC_RELEASE_VERDICT.md:8`, `workbench/docs/PC_PILOT_HANDOFF.md:25`, `workbench/tests/test_handoff_commands.py:395`
- **Reproduction**:
  ```bash
  .venv/bin/python -m pytest workbench/tests/test_handoff_commands.py -k "test_no_release_document_points_at_a_superseded_source_sha" -q
  ```
- **Observed Output**:
  ```
  FAILED workbench/tests/test_handoff_commands.py::test_no_release_document_points_at_a_superseded_source_sha - AssertionError: TESTED_SOURCE_SHA cf84cfa does not contain the production code currently on disk
  ```
- **Root Cause**: Commit `3a91870` modified `workbench/docker/mutation_checks.sh`, which is outside `workbench/docs` and `workbench/tests`. `FINAL_PC_RELEASE_VERDICT.md` still points at `cf84cfa`.

### Defect 2: P2 — Operator runbook step 0 omits runtime dependencies for step 6
- **File:Line**: `workbench/docs/MODEL_STATUS_SPEC.md:288`
- **Reproduction**:
  ```bash
  .venv/bin/python -c "import open_clip" # In an environment created strictly following Step 0
  ```
- **Observed Output**:
  ```
  ModuleNotFoundError: No module named 'open_clip'
  ```
- **Root Cause**: Step 0 declares minimal dependencies for steps 1-6 but omits `open-clip-torch` and `tqdm`, which are directly imported by `workbench/replay/limn.py:168` and `workbench/replay/encoder.py:140`.

### Defect 3: P3 — Semantically invalid and unreachable Conda inheritance branch
- **File:Line**: `workbench/backend/runtime.py:397-399`
- **Reproduction**:
  Static audit: standard Conda environment has `sys.base_prefix == sys.prefix`. `inspect_environment_distributions` guards call with `if base_prefix and str(base_prefix) != str(prefix):`. The branch `if (prefix / "conda-meta").is_dir(): return True` is never called in real Conda runtime. Furthermore, Conda environments do not inherit packages from `base`.

### Defect 4: P3 — Redundant two-variable launcher state abstraction
- **File:Line**: `workbench/backend/adapters/models.py:25-27`
- **Reproduction**:
  Static audit: `launcher` and `launcher_roots` are only read in `workbench/backend/environment_planner.py:256, 265`. No command builder reads `adapter.launcher`.

### Defect 5: P3 — Dead helper in production replay module
- **File:Line**: `workbench/replay/lavis_processors.py:140-155`
- **Reproduction**:
  Grep across codebase reveals zero production callers of `lavis_processors.reset()`. Used exclusively in parity tool and unit tests.

---

## Execution Trace and Final Verdict

- **Static Audit Scope**: `git diff e46ad65..cf84cfa`, commits `42178c6`, `937f64d`, `3a91870`, `MODEL_STATUS_SPEC.md`, `R10_REMEDIATION.md`, `R10_E2E_SURFACE.md`, `R10_LAVIS_DEPS.md`.
- **Runtime Verification Scope**: 12 pytest suites with `--basetemp=/home/heheboiz/.cache/cir-pytest6`, CLI dry-runs of download and preflight scripts.
- **Final Verdict**: **VARIANT**
  Remediation successfully resolves reported findings without corrupting native scientific evaluation semantics. However, subsequent commit `3a91870` created a release documentation hash drift failing `test_handoff_commands.py` (P1), and operator Step 0 documentation omits necessary CLI dependencies for Step 6 (P2).
