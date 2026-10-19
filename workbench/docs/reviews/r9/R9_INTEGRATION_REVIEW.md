# R9 Integration Review: Combined Diff and Working Tree Audit

- **Date**: 2026-10-19
- **Reviewer**: Independent Scientific Reproducibility Reviewer
- **Target**: Working tree diff on top of `914703e`, plus untracked files (`workbench/replay/lavis_processors.py`, `workbench/tests/test_r9_defect_regressions.py`, `workbench/scripts/lavis_processor_parity.py`, `workbench/docs/reviews/r9/*`)
- **Mode**: STATIC_AUDIT + RUNTIME_VERIFIED (read-only unit/fixture/script inspection)

---

## Question 1: Accidental Scientific Change

**Verdict**: VARIANT (shipped DCNet metric parser behavior changed to fix transposition defect; no evaluator semantics changed; no native gallery, masking, ranking, preprocessing, prompt, macro arithmetic, or protocol identity changed).

### Analysis
All modified hunks across the working tree diff were inspected against the pinned sources in `workbench/third_party/`:
1. **Gallery Construction**: Unchanged. No changes to gallery indexing in any model wrapper or adapter.
2. **Reference Masking**: Unchanged. Masking logic (e.g. `-10e10` masking in `TGCIR`, gallery pair filtering in `LIMN`, `HINT`, `HABIT`, `INTENT`, `ConeSep`, `AirKnow`) remains identical to pinned sources.
3. **Ranking**: Unchanged. Distance metrics (inner product, cosine similarity) and sorting routines are untouched.
4. **Metric Units**: In `workbench/backend/metrics_extraction.py:627, 667`, DCNet floats are multiplied by `100.0` to convert native fractional output `[0.0, 1.0]` from `model/metric.py` to percentages `[0.0, 100.0]`. This matches FashionIQ literature and registry units.
5. **Category Order in Evaluator vs Parser**:
   - Upstream evaluator (`workbench/third_party/DCNet/trainer/trainer_joint.py:133-138` and `test.py:74-84`) iterates categories using `config['data_loader']['args']['categories']`. Evaluator iteration order was not touched.
   - Parser (`workbench/backend/metrics_extraction.py:612-680`) previously assumed static category tuple `("dress", "shirt", "toptee")` when mapping the six anonymous floats in `comb:`. The parser now extracts the order dynamically from stdout lines `\[(dress|shirt|toptee)\] comp corr r@10...` and cross-validates against `executed_config`. Category order in an EVALUATOR was NOT modified; category order handling in the PARSER was repaired.
6. **Preprocessing Transforms & Prompt Handling**:
   - Wrappers for `hint`, `habit`, `intent`, `conesep`, `airknow` replace `load_model_and_preprocess(...)` with `workbench.replay.lavis_processors.text_processors`.
   - The resolved text processor config remains `configs/models/blip2/blip2_pretrain.yaml` (`name: "blip_caption"`, `prompt: ""`, `max_words: 50`).
   - The instantiated class remains `lavis.processors.blip_processors.BlipCaptionProcessor`. Captions undergo identical lowercasing, whitespace trimming, punctuation stripping, and truncation.
7. **Macro Arithmetic**:
   - Macro arithmetic in `parse_dcnet_output` remains the unweighted average of the three category recall values (`macro_r10`, `macro_r50`, `macro_mean = (macro_r10 + macro_r50)/2`).
8. **Protocol Identity**: Unchanged across all models.

### Hunks Touching Shipped Metrics

1. `workbench/backend/metrics_extraction.py:581-602`:
   - Adds `_dcnet_category_order(config)`: extracts category ordering list from `config['data_loader']['args']['categories']`.
2. `workbench/backend/metrics_extraction.py:612-680`:
   - In `parse_dcnet_output`: preserves stream category order `order = [m.group(1) for m in re.finditer(...)]`, matches cells of `comb:` against `order`, verifies with `_dcnet_category_order`, converts fractions to percentages (`* 100.0`), and verifies against `[comb_avg]`.
   - Impact: Prevents silent metric transposition between dress, shirt, and toptee for DCNet.
3. `workbench/backend/metrics_extraction.py:814-840`:
   - In `extract_aggregate_metrics`: adds optional `executed_config: dict[str, Any] | None = None` and passes it to `parse_dcnet_output`.
4. `workbench/scripts/evaluate_models.py:472-493`:
   - Adds `read_executed_config(plan)`: extracts `<checkpoint>/config.json` for DCNet run directory.
5. `workbench/scripts/evaluate_models.py:606-611`:
   - In `invoke_evaluator`: copies `config.json` into run artifact directory `directory / "config.json"`.
6. `workbench/scripts/evaluate_models.py:1901`:
   - Passes `executed_config=read_executed_config(plan)` to `extract_aggregate_metrics`.
7. `workbench/scripts/verify_run.py:330-345`:
   - Adds `_recorded_executed_config(model_id, directory)`: reads `config.json` from the run artifact directory.
8. `workbench/scripts/verify_run.py:375-378`:
   - In `_reextraction_reasons`: forwards `executed_config=_recorded_executed_config(model_id, directory)` to `extract_aggregate_metrics`.

- **Severity**: NONE
- **Recommended Action**: Keep metric extraction changes; ensure category ordering remains tied to evaluator output.

---

## Question 2: Unnecessary Abstraction / Dead Code

**Verdict**: VARIANT (dual path in `_load_preprocess_function` contains dead branch on standard environments; script docstring contains invalid path; helper pair is justified by lifecycle).

### Evidence & Checks

1. **`workbench/replay/lavis_processors.py:_load_preprocess_function`**:
   - Contains dual path:
     - Branch 1: `importlib.import_module("lavis.models").load_preprocess`
     - Branch 2: AST parse and `compile()` of `load_preprocess` function definition from `lavis/models/__init__.py`.
   - Command:
     ```bash
     .venv/bin/python -c "
     import sys, types, importlib
     from pathlib import Path
     package = types.ModuleType('lavis')
     package.__path__ = [str(Path('workbench/third_party/HINT/lavis'))]
     sys.modules['lavis'] = package
     try:
         importlib.import_module('lavis.models')
     except Exception as e:
         print('Branch 1 failure:', type(e), e)
     "
     ```
   - Output:
     `Branch 1 failure: <class 'ImportError'> cannot import name 'apply_chunking_to_forward' from 'transformers.modeling_utils' (/home/heheboiz/data/cir/.venv/lib/python3.13/site-packages/transformers/modeling_utils.py)`
   - Analysis: Branch 1 is dead on normal hosts because `lavis.models` imports 30 model families requiring exact pinned legacy dependencies (including older `transformers`, `decord`, etc.). Branch 2 (AST compile) is what always executes.

2. **`workbench/replay/common.py:constructor_input_blockers`**:
   - Not single-call-site or dead code.
   - Imported and called by 6 active replay wrappers: `hint.py:127`, `habit.py:98`, `intent.py:98`, `conesep.py:110`, `airknow.py:113`, and `pair.py:113`.
   - Consolidates constructor dataset checks.

3. **`workbench/scripts/lavis_processor_parity.py`**:
   - Standalone verification script; not invoked by CI or production pipeline.
   - Path discrepancy: Docstrings in `lavis_processors.py:33` and `lavis_processor_parity.py:12` cite non-existent path `benchmarks/lavis_processor_parity.py`.

4. **`read_executed_config` (`evaluate_models.py`) vs `_recorded_executed_config` (`verify_run.py`)**:
   - Both are needed due to distinct lifecycle stages:
     - `read_executed_config`: executed by orchestrator during evaluation run; reads from `plan.checkpoint` (`config.json` in staged checkpoint directory).
     - `_recorded_executed_config`: executed by post-hoc verifier `verify_run.py`; reads from `directory` (`config.json` preserved in output log directory).
   - Logic is minimally redundant but serves separate entry points.

- **Severity**: P3
- **Recommended Action**: Fix docstring reference from `benchmarks/lavis_processor_parity.py` to `workbench/scripts/lavis_processor_parity.py`. Consider removing dead `import lavis.models` branch if AST compile is the intended universal path.

---

## Question 3: Incomplete Wiring

**Verdict**: BLOCKED (wiring of DCNet binding check is broken in `preflight.py` and omitted from `pc_readiness.py`, `doctor.py`, and `prepare_model.py`).

### Evidence & Checks

1. **DCNet Binding Refusal in Launch vs `pc_readiness`**:
   - Launch stopping: In `evaluate_models.py:350-380`, `guarded_plan` calls `blockers(...)`, which calls `preflight_blockers(model, config, dataset_root=dataset_root)`. If mismatched, `guarded_plan` returns `None, blocked` and halts launch.
   - **`pc_readiness` JSON exclusion**:
     In `workbench/scripts/pc_readiness.py:223`:
     ```python
     blockers = preflight_blockers(model, config)
     ```
     `dataset_root` is NOT passed (defaults to `None`).
     In `workbench/backend/preflight.py:239`:
     ```python
     if model_id == "dcnet" and dataset_root is not None and source is not None:
     ```
     Because `dataset_root` is `None`, the check is completely skipped!
     Verification Command:
     ```bash
     .venv/bin/python workbench/scripts/pc_readiness.py --json | jq '.models.dcnet.blockers[] | select(contains("binding"))'
     ```
     Observed Output: (empty — no binding mismatch reported).
   - **Missing `evaluator_cwd` in `preflight.py`**:
     In `workbench/backend/preflight.py:249-250`:
     ```python
     mismatch = binding_mismatch(checkpoint_path(model["model_id"], model["checkpoint_variants"][0],
                                                 config.WORKBENCH_CHECKPOINT_ROOT), dataset_root)
     ```
     `evaluator_cwd` is omitted! In `workbench/replay/dcnet.py:80`, `configured_data_dir` defaults `evaluator_cwd` to `Path(os.getcwd())` (the repo root `/home/heheboiz/data/cir`).
     When an operator correctly stages the symlink inside `source / "dataset" / "fashioniq"`, `preflight_blockers` resolves against repo root and falsely raises a blocker!
     Verification Command:
     ```bash
     .venv/bin/python -c "
     from workbench.replay.dcnet import binding_mismatch
     from pathlib import Path
     import tempfile, json
     with tempfile.TemporaryDirectory() as tmp:
         tmp_path = Path(tmp)
         source = tmp_path / 'DCNet'
         real = tmp_path / 'real_dataset'; real.mkdir()
         (source / 'dataset').mkdir(parents=True)
         (source / 'dataset' / 'fashioniq').symlink_to(real)
         run = tmp_path / 'run'; run.mkdir()
         (run / 'config.json').write_text(json.dumps({'data_loader': {'args': {'data_dir': 'dataset/fashioniq'}}}))
         print('With evaluator_cwd=source:', binding_mismatch(run, real, evaluator_cwd=source))
         print('Without evaluator_cwd:', binding_mismatch(run, real))
     "
     ```
     Observed Output:
     ```text
     With evaluator_cwd=source: None
     Without evaluator_cwd: DCNet dataset binding mismatch: the evaluator resolves 'dataset/fashioniq' to /home/heheboiz/data/cir/dataset/fashioniq, not the verified dataset root... Stage the canonical dataset at /home/heheboiz/data/cir/dataset/fashioniq as a symlink...
     ```

2. **Copied `config.json` Survival to `verify_run`**:
   - `evaluate_models.invoke_evaluator` creates `directory = log_dir(...)` and copies `source_config` to `directory / "config.json"` (`:606-611`).
   - `verify_run._recorded_executed_config(model_id, directory)` reads `directory / "config.json"`.
   - File survives and is successfully re-read by the post-run verifier.

3. **LIMN Path Safety**:
   - `read_executed_config` guards with `if plan.model_id != "dcnet": return None`.
   - LIMN returns `None` safely.

4. **Frontend / API Consumption of `approved_patch`**:
   - `approved_patch_state` returns keys `("applied", "patch_file", "patch_sha256", "detail", "patches")`.
   - Backward-compatible with existing consumers expecting `applied` or `detail`.
   - `_variant_provenance` stores `"approved_source_patch": approved_patch` including `"patches"`.
   - No schema breakages observed.

- **Severity**: P1
- **Recommended Action**:
  1. Pass `evaluator_cwd=source` in `workbench/backend/preflight.py:249`.
  2. Pass `dataset_root=config.FASHIONIQ_ROOT` in `pc_readiness.py:223`, `doctor.py:151`, and `prepare_model.py:40`.

---

## Question 4: Fresh-Clone Reproducibility

**Verdict**: BLOCKED (untracked files prevent fresh-clone execution; tracked checkouts exist but uncommitted code is missing).

### Evidence & Checks

1. **Third-Party Checkouts Tracking**:
   - Command: `git check-ignore workbench/third_party` (exit code 1: NOT ignored).
   - Command: `git ls-files workbench/third_party | wc -l` (observed: 7,376 tracked files).
   - Pinned checkouts under `workbench/third_party/` are fully tracked in Git.

2. **Clean Checkout Verification**:
   - Command: `git clone --no-hardlinks . /tmp/clean_clone`
   - Command 1:
     `/home/heheboiz/data/cir/.venv/bin/python -m pytest /tmp/clean_clone/workbench/tests/test_r9_defect_regressions.py -q -p no:cacheprovider`
     - Observed rc: `4`
     - Observed output:
       `ERROR: file or directory not found: /tmp/clean_clone/workbench/tests/test_r9_defect_regressions.py`
   - Command 2:
     `/home/heheboiz/data/cir/.venv/bin/python /tmp/clean_clone/workbench/scripts/lavis_processor_parity.py`
     - Observed rc: `2`
     - Observed output:
       `/home/heheboiz/data/cir/.venv/bin/python: can't open file '/tmp/clean_clone/workbench/scripts/lavis_processor_parity.py': [Errno 2] No such file or directory`
   - Command 3 (Modified tests at HEAD `914703e` in clean clone):
     `/home/heheboiz/data/cir/.venv/bin/python -m pytest /tmp/clean_clone/workbench/tests/test_approved_source_patch.py /tmp/clean_clone/workbench/tests/test_blip2_text_processor_parity.py /tmp/clean_clone/workbench/tests/test_hint_replay.py /tmp/clean_clone/workbench/tests/test_pc_readiness.py -q -p no:cacheprovider`
     - Observed rc: `1` (1 failed, 33 passed)
     - Failure: `test_the_cli_accepts_the_autopilot_config` fails assertion because `/tmp/clean_clone/data/FashionIQ` directory does not exist.

- **Severity**: P0
- **Recommended Action**: Track and commit `workbench/replay/lavis_processors.py`, `workbench/tests/test_r9_defect_regressions.py`, `workbench/scripts/lavis_processor_parity.py`, and `workbench/docs/reviews/r9/*` into git.

---

## Question 5: Status-Spec Truth

**Verdict**: BLOCKED (multiple factual claims in `MODEL_STATUS_SPEC.md` are falsified by the code).

### Claims Evaluated Against Code

1. **Claim (Header :11)**: R9 changes documented in `workbench/docs/reviews/R9_FINAL_REPORT.md`.
   - **Finding**: FALSIFIED. `workbench/docs/reviews/R9_FINAL_REPORT.md` does not exist on disk.
2. **Claim (§3.11 :154)**: DCNet preflight binding check runs in `backend/preflight.preflight_blockers` — "danh sách mà `doctor`, `pc_readiness`, `prepare_model` và `evaluate_models` đều đọc."
   - **Finding**: FALSIFIED. `pc_readiness.py:223`, `doctor.py:151`, and `prepare_model.py:40` all invoke `preflight_blockers` without `dataset_root`. Since `dataset_root is None`, `preflight.py:239` skips the binding check entirely. Only `evaluate_models.py` runs it.
3. **Claim (§3.11 :154)**: Canonical symlink passes and relative path inside checkout refuses.
   - **Finding**: FALSIFIED. Because `preflight.py:249` does not pass `evaluator_cwd=source` to `binding_mismatch`, `binding_mismatch` defaults to `Path.cwd()` (repo root). A canonical symlink staged at `workbench/third_party/DCNet/dataset/fashioniq` fails preflight when run from repo root.
4. **Claim (§4 :261, F05)**: "validator HINT/ENCODER/PAIR yêu cầu train; layout → ilearn_resized_training".
   - **Finding**: FALSIFIED for ENCODER. `workbench/replay/encoder.py:90-112` was NOT updated to use `constructor_input_blockers`. Its `validate_dataset(root, categories=('dress',))` loops `for category in categories:` and only inspects `dress`. Calling it with a dress-only dataset passes validation (`[]`), but native `workbench/third_party/ENCODER/datasets1.py:59-73` crashes on missing `cap.shirt.train.json`.
5. **Claim (§4 :260, F04)**: Evidence cited is `test_component_boundaries.py (4 test patch + gate)`.
   - **Finding**: FALSIFIED. `test_component_boundaries.py:280-320` tests `source_provenance` (vendored allowlist), not `approved_patch_state`. The regression tests for `approved_patch_state` live in `test_r9_defect_regressions.py` and `test_approved_source_patch.py`.
6. **Claim (§2 :75-76)**: Checkpoint hashes for `conesep` and `airknow` are `hash VERIFIED; chưa stage`.
   - **Finding**: FALSIFIED. In `workbench/registry/models.yaml:137-138, 148-149, 194-195, 206-207, 218-219`, both models declare `expected_sha256: null` and `expected_size_bytes: null` with status `NO_CLEAN_CHECKPOINT`.
7. **Claim (§2 :77)**: DCNet checkpoint filename is `trained_model.pth (Drive)`.
   - **Finding**: FALSIFIED. In `workbench/registry/models.yaml:671`, the registered filename is `fashioniq_dcnet` with `checkpoint_serialization: run_directory`.
8. **Claim (§2 :78, §3.12 :170)**: CSMCIR "2 approved patches (cả hai bắt buộc)" and "F04 đã sửa".
   - **Finding**: FALSIFIED at runtime. Neither patch is applied to `workbench/third_party/CSMCIR` (both files retain base SHA256), and no automated patch execution step exists in `run.sh` or `pipeline.py`.

- **Severity**: P1
- **Recommended Action**: Align `MODEL_STATUS_SPEC.md` with actual code status; fix encoder validator; wire `dataset_root` in entry points.

---

## Question 6: Cross-Review of Prior Investigation (DC_02_PARSER.md Claim 3)

**Verdict**: VERIFIED (transposition defect in `parse_dcnet_output` is resolved on current code).

### Reproduction Command & Literal Output
Command:
```bash
.venv/bin/python -c "
import json
from workbench.backend.metrics_extraction import parse_dcnet_output

stdout = '''2026-10-19 10:00:00,000 - INFO - [toptee] comp corr r@10, r@50: 0.65\t0.85\t0.60\t0.80
2026-10-19 10:00:00,001 - INFO - [dress] comp corr r@10, r@50: 0.15\t0.35\t0.10\t0.30
2026-10-19 10:00:00,002 - INFO - [shirt] comp corr r@10, r@50: 0.25\t0.45\t0.20\t0.40
2026-10-19 10:00:00,003 - INFO - comb: 0.7 0.9 0.2 0.4 0.3 0.5
2026-10-19 10:00:00,004 - INFO - [Avg Recall]     : 0.4417
2026-10-19 10:00:00,005 - INFO - [Avg Recall corr]: 0.4000
2026-10-19 10:00:00,006 - INFO - [comb_avg]       : 0.5
'''

res = parse_dcnet_output(stdout)
print('extraction_status =', repr(res.extraction_status))
print('observed_metrics =', json.dumps(res.observed_metrics, indent=2))
print('category_metrics =', json.dumps(res.category_metrics, indent=2))
"
```

Literal Output:
```text
extraction_status = 'EXTRACTED_VERIFIED'
observed_metrics = {
  "r10": 40.0,
  "r50": 60.0,
  "mean": 50.0
}
category_metrics = {
  "dress": {
    "r10": 20.0,
    "r50": 40.0
  },
  "shirt": {
    "r10": 30.0,
    "r50": 50.0
  },
  "toptee": {
    "r10": 70.0,
    "r50": 90.0
  },
  "dress__composition": {
    "r10": 15.0,
    "r50": 35.0
  },
  "shirt__composition": {
    "r10": 25.0,
    "r50": 45.0
  },
  "toptee__composition": {
    "r10": 65.0,
    "r50": 85.0
  },
  "dress__correction": {
    "r10": 10.0,
    "r50": 30.0
  },
  "shirt__correction": {
    "r10": 20.0,
    "r50": 40.0
  },
  "toptee__correction": {
    "r10": 60.0,
    "r50": 80.0
  }
}
```

### Observation
In `DC_02_PARSER.md`, `dress` received `(70.0, 90.0)`, `shirt` received `(20.0, 40.0)`, and `toptee` received `(30.0, 50.0)`.
Under current `parse_dcnet_output`, pair 1 (`0.7, 0.9` -> `70.0, 90.0`) is correctly mapped to `toptee`, pair 2 (`0.2, 0.4` -> `20.0, 40.0`) is mapped to `dress`, and pair 3 (`0.3, 0.5` -> `30.0, 50.0`) is mapped to `shirt`.
The transposition is fully resolved.

- **Severity**: NONE
- **Recommended Action**: None. Fix confirmed.

---

## Question 7: Test Suite Regressions

**Verdict**: BLOCKED (`workbench/tests/test_handoff_commands.py` fails with rc=1; 14 other test files pass).

### Test Execution Matrix

| Test File | Command | Observed Result | Return Code |
|---|---|---|---|
| `workbench/tests/test_r9_defect_regressions.py` | `.venv/bin/python -m pytest workbench/tests/test_r9_defect_regressions.py -q -p no:cacheprovider` | `56 passed, 4 warnings in 4.86s` | 0 |
| `workbench/tests/test_approved_source_patch.py` | `.venv/bin/python -m pytest workbench/tests/test_approved_source_patch.py -q -p no:cacheprovider` | `9 passed in 3.33s` | 0 |
| `workbench/tests/test_pc_readiness.py` | `.venv/bin/python -m pytest workbench/tests/test_pc_readiness.py -q -p no:cacheprovider` | `15 passed in 40.36s` | 0 |
| `workbench/tests/test_vendored_approved_patch.py` | `.venv/bin/python -m pytest workbench/tests/test_vendored_approved_patch.py -q -p no:cacheprovider` | `11 passed in 2.06s` | 0 |
| `workbench/tests/test_handoff_commands.py` | `.venv/bin/python -m pytest workbench/tests/test_handoff_commands.py -q -p no:cacheprovider` | `1 failed, 6 passed in 6.69s` | **1 (FAIL)** |
| `workbench/tests/test_component_boundaries.py` | `.venv/bin/python -m pytest workbench/tests/test_component_boundaries.py -q -p no:cacheprovider` | `23 passed in 14.62s` | 0 |
| `workbench/tests/test_hint_replay.py` | `.venv/bin/python -m pytest workbench/tests/test_hint_replay.py -q -p no:cacheprovider` | `4 passed in 0.09s` | 0 |
| `workbench/tests/test_habit_replay.py` | `.venv/bin/python -m pytest workbench/tests/test_habit_replay.py -q -p no:cacheprovider` | `5 passed in 0.30s` | 0 |
| `workbench/tests/test_intent_replay.py` | `.venv/bin/python -m pytest workbench/tests/test_intent_replay.py -q -p no:cacheprovider` | `5 passed in 0.25s` | 0 |
| `workbench/tests/test_conesep_replay.py` | `.venv/bin/python -m pytest workbench/tests/test_conesep_replay.py -q -p no:cacheprovider` | `5 passed in 0.23s` | 0 |
| `workbench/tests/test_airknow_replay.py` | `.venv/bin/python -m pytest workbench/tests/test_airknow_replay.py -q -p no:cacheprovider` | `5 passed in 0.24s` | 0 |
| `workbench/tests/test_pair_replay.py` | `.venv/bin/python -m pytest workbench/tests/test_pair_replay.py -q -p no:cacheprovider` | `5 passed in 0.09s` | 0 |
| `workbench/tests/test_dcnet_replay.py` | `.venv/bin/python -m pytest workbench/tests/test_dcnet_replay.py -q -p no:cacheprovider` | `4 passed in 0.09s` | 0 |
| `workbench/tests/test_hint_metric_parser.py` | `.venv/bin/python -m pytest workbench/tests/test_hint_metric_parser.py -q -p no:cacheprovider` | `11 passed in 0.02s` | 0 |
| `workbench/tests/test_verified_declarations.py` | `.venv/bin/python -m pytest workbench/tests/test_verified_declarations.py -q -p no:cacheprovider` | `19 passed in 0.67s` | 0 |

### Failures in `test_handoff_commands.py`
`test_no_release_document_points_at_a_superseded_source_sha`:
- Assertion: `TESTED_SOURCE_SHA 119df5f does not contain the production code currently on disk`
- Command `git diff --quiet 119df5f -- . :(exclude)workbench/docs :(exclude)workbench/tests` exited with return code 1 because production code has changed on disk without updating `FINAL_PC_RELEASE_VERDICT.md`.

- **Severity**: P2
- **Recommended Action**: Update `FINAL_PC_RELEASE_VERDICT.md` commit hash after committing production edits.

---

## Defects Found

### Defect 1: DCNet Binding Mismatch Falsely Triggers Due to Missing `evaluator_cwd` in `preflight.py`
- **Location**: `workbench/backend/preflight.py:249-250`
- **Severity**: P1
- **Description**:
  When `preflight_blockers` executes for `dcnet`, it calls:
  `mismatch = binding_mismatch(checkpoint_path(...), dataset_root)`
  It does NOT pass `evaluator_cwd=source`.
  In `workbench/replay/dcnet.py:80`, `configured_data_dir` defaults `evaluator_cwd` to `Path.cwd()` (the orchestrator working directory, e.g. `/home/heheboiz/data/cir`).
  Even when an operator stages the required canonical symlink at `workbench/third_party/DCNet/dataset/fashioniq -> <dataset_root>` (the directory where DCNet actually runs), `preflight_blockers` resolves the relative path against repo root (`/home/heheboiz/data/cir/dataset/fashioniq`) and reports a binding mismatch blocker.
- **Minimal Reproduction**:
  ```python
  import tempfile, json
  from pathlib import Path
  from workbench.replay.dcnet import binding_mismatch

  with tempfile.TemporaryDirectory() as tmp:
      tmp_path = Path(tmp)
      source = tmp_path / "DCNet"
      dataset = tmp_path / "fashioniq_data"; dataset.mkdir()
      (source / "dataset").mkdir(parents=True)
      (source / "dataset" / "fashioniq").symlink_to(dataset)
      run = tmp_path / "run"; run.mkdir()
      (run / "config.json").write_text(json.dumps({"data_loader": {"args": {"data_dir": "dataset/fashioniq"}}}))

      # Evaluator runs from source directory, but preflight.py doesn't pass evaluator_cwd:
      assert binding_mismatch(run, dataset, evaluator_cwd=source) is None
      assert binding_mismatch(run, dataset) is not None  # FAILS FALSELY
  ```

---

### Defect 2: DCNet Dataset Binding Check Skipped in `pc_readiness.py`, `doctor.py`, and `prepare_model.py`
- **Location**: `workbench/scripts/pc_readiness.py:223`, `workbench/scripts/doctor.py:151`, `workbench/scripts/prepare_model.py:40`
- **Severity**: P1
- **Description**:
  `workbench/backend/preflight.py:239` gates the DCNet binding check behind `if model_id == "dcnet" and dataset_root is not None and source is not None:`.
  `pc_readiness.py:223` calls `preflight_blockers(model, config)` with no `dataset_root`.
  `doctor.py:151` and `prepare_model.py:40` call `preflight_blockers(model, config, canonical_root=config.FASHIONIQ_ROOT)` without `dataset_root`.
  Because `dataset_root` is `None`, the DCNet dataset binding check never executes on any of these three entry points, contradicting claims in `MODEL_STATUS_SPEC.md:154`.
- **Minimal Reproduction**:
  ```bash
  .venv/bin/python workbench/scripts/pc_readiness.py --json | jq '.models.dcnet.blockers[] | select(contains("binding"))'
  # Returns nothing, even if checkpoint and source are present and misbound.
  ```

---

### Defect 3: `encoder.py:validate_dataset` Does Not Use `constructor_input_blockers`, Passing on Dress-Only Subset and Crashing at Runtime
- **Location**: `workbench/replay/encoder.py:90-112`
- **Severity**: P1
- **Description**:
  Unlike `hint.py`, `habit.py`, `intent.py`, `conesep.py`, `airknow.py`, and `pair.py`, `encoder.py` was not updated to invoke `constructor_input_blockers`.
  Its `validate_dataset` function iterates `for category in categories:`.
  When invoked as `validate_dataset(root, categories=('dress',))`, it validates only `dress` files and returns `[]`.
  However, native constructor `workbench/third_party/ENCODER/datasets1.py:59-73` unconditionally loops `for name in ['dress', 'shirt', 'toptee']:` during initialization, immediately raising `FileNotFoundError` when `cap.shirt.train.json` is absent.
- **Minimal Reproduction**:
  ```python
  import tempfile
  from pathlib import Path
  from workbench.replay import encoder

  with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      for d in ("captions", "image_splits"): (root / d).mkdir(parents=True)
      (root / "resized_image" / "dress").mkdir(parents=True)
      for f in ("cap.dress.train.json", "cap.dress.val.json", "correction_dict_dress.json"):
          (root / "captions" / f).write_text("[]" if "cap" in f else "{}")
      (root / "image_splits" / "split.dress.val.json").write_text("[]")

      blockers = encoder.validate_dataset(root, categories=("dress",))
      assert blockers == []  # FAILS TO BLOCK: returns clean even though native constructor crashes
  ```

---

### Defect 4: Fresh-Clone Execution Broken by Untracked Working Tree Modules
- **Location**: Untracked files `workbench/replay/lavis_processors.py`, `workbench/tests/test_r9_defect_regressions.py`, `workbench/scripts/lavis_processor_parity.py`
- **Severity**: P0
- **Description**:
  The new modules and regression tests exist only as untracked files in the local working directory. A clean `git clone` or `git worktree add` without untracked files lacks `workbench/tests/test_r9_defect_regressions.py` (exit code 4) and `workbench/scripts/lavis_processor_parity.py` (exit code 2).
- **Minimal Reproduction**:
  ```bash
  git clone --no-hardlinks . /tmp/clean_test
  python3 -m pytest /tmp/clean_test/workbench/tests/test_r9_defect_regressions.py
  # Output: ERROR: file or directory not found
  ```

---

### Defect 5: Documentation References Non-Existent File and Non-Existent Directory
- **Location**:
  - `workbench/docs/MODEL_STATUS_SPEC.md:11`
  - `workbench/replay/lavis_processors.py:33`
  - `workbench/scripts/lavis_processor_parity.py:12`
- **Severity**: P3
- **Description**:
  1. `MODEL_STATUS_SPEC.md:11` references `workbench/docs/reviews/R9_FINAL_REPORT.md` which does not exist.
  2. `lavis_processors.py:33` and `lavis_processor_parity.py:12` instruct the user to run `.venv/bin/python benchmarks/lavis_processor_parity.py workbench/third_party/HINT`, but no `benchmarks/` directory exists in the repository.
- **Minimal Reproduction**:
  ```bash
  ls workbench/docs/reviews/R9_FINAL_REPORT.md  # No such file
  ls -d benchmarks/                            # No such directory
  ```

---

## Execution Trace & Verification Summary

1. `workbench/backend/metrics_extraction.py`: Verified `parse_dcnet_output` resolves category transposition dynamically using stream order; macro consistency check and metric scaling verified.
2. `workbench/backend/preflight.py`: Verified `approved_patch_state` validates all items in `approved_source_patches`. Verified missing `evaluator_cwd` in DCNet check.
3. `workbench/replay/lavis_processors.py`: Verified `text_processors` builds `BlipCaptionProcessor` from pinned config without loading secondary model or touching `/root/bert-base-uncased`.
4. `workbench/tests/test_r9_defect_regressions.py`: 56/56 passing tests on current tree.
5. `workbench/scripts/lavis_processor_parity.py`: Verified 5 BLIP-2 models load pinned caption processor with status `LOADED`.
6. Fresh clone audit: Tested in `/tmp/clean_clone`; identified missing untracked files.
7. Suite audit: 14 test files pass; `test_handoff_commands.py` fails on documented commit hash drift.

## Final Review Verdict

**BLOCKED**

The core algorithm and parser repairs (DCNet transposition fix, multi-patch verification loop, and LAVIS processor extraction without model duplication) are functionally sound and verified. However, delivery is BLOCKED by:
1. P0 untracked files missing from Git (`lavis_processors.py`, `test_r9_defect_regressions.py`, `lavis_processor_parity.py`).
2. P1 preflight binding check defect (omitted `evaluator_cwd=source` in `preflight.py`).
3. P1 wiring omission (`dataset_root` omitted from `pc_readiness.py`, `doctor.py`, and `prepare_model.py`).
4. P1 F05 omission in `workbench/replay/encoder.py`.
