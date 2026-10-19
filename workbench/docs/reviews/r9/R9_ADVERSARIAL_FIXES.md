# Adversarial Audit: R9 Candidate Defect Fixes and F06 Processor Reconstruction

**Document**: `workbench/docs/reviews/r9/R9_ADVERSARIAL_FIXES.md`  
**Date**: 2026-10-19  
**Target Git Tree**: `914703e` with working-tree changes  
**Evaluator Environment**: Linux x86_64 (`.venv/bin/python`)  

---

## 1. Verdict Table

| Claim | Verdict | Evidence Command | Raw Output Excerpt |
|---|---|---|---|
| **Claim 1: DCNet dataset binding runs on real launch path** | **PARTIAL** | `.venv/bin/python -c "from workbench.backend.preflight import preflight_blockers; from workbench.backend.operator_config import resolve_config; from workbench.backend.registry import model_by_id, checkpoint_path; cfg=resolve_config(); m=model_by_id('dcnet'); p=checkpoint_path('dcnet', m['checkpoint_variants'][0], cfg.WORKBENCH_CHECKPOINT_ROOT); p.mkdir(parents=True, exist_ok=True); (p/'config.json').write_text('{\"data_loader\":{\"args\":{\"data_dir\":\"dataset/fashioniq\"}}}'); print('doctor:', any('binding mismatch' in b for b in preflight_blockers(m, cfg, canonical_root=cfg.FASHIONIQ_ROOT))); print('evaluate:', any('binding mismatch' in b for b in preflight_blockers(m, cfg, dataset_root=cfg.FASHIONIQ_ROOT)))"` | `doctor: False`<br>`evaluate: True` |
| **Claim 2: DCNet parser no longer transposes categories** | **HOLDS** | `.venv/bin/python -c "from workbench.backend.metrics_extraction import parse_dcnet_output; stdout = '[toptee] comp corr r@10, r@50: 0.65 0.85 0.60 0.80\n[dress] comp corr r@10, r@50: 0.15 0.35 0.10 0.30\n[shirt] comp corr r@10, r@50: 0.25 0.45 0.20 0.40\ncomb: 0.7 0.9 0.2 0.4 0.3 0.5\n[comb_avg] : 0.5'; res = parse_dcnet_output(stdout, None, {'data_loader': {'args': {'categories': ['toptee', 'dress', 'shirt']}}}); print('status:', res.extraction_status, 'toptee:', res.category_metrics['toptee'], 'dress:', res.category_metrics['dress'])"` | `status: EXTRACTED_VERIFIED toptee: {'r10': 70.0, 'r50': 90.0} dress: {'r10': 20.0, 'r50': 40.0}` |
| **Claim 3: CSMCIR requires whole declared patch set** | **PARTIAL** | `.venv/bin/python -c "import tempfile, hashlib; from pathlib import Path; from workbench.backend.preflight import approved_patch_state; t=Path(tempfile.mkdtemp()); s=t/'src'; s.mkdir(); (t/'r.py').write_text('x'); (s/'f.py').symlink_to(t/'r.py'); m={'model_id':'csmcir','approved_source_patches':[{'path':'f.py','patch_file':'p.patch','patch_sha256':'s','base_sha256':'b','derived_sha256':hashlib.sha256(b'x').hexdigest()}]}; approved_patch_state(m, s)"` | `ValueError: artifact is not a regular file or non-symlink directory: .../src/f.py` |
| **Claim 4: LAVIS wrappers validate constructor input set** | **PARTIAL** | `.venv/bin/python -c "from workbench.backend.fashioniq_layout import paths_for_layout, missing_paths; from workbench.replay.habit import validate_dataset; from workbench.backend.registry import model_by_id; import tempfile, pathlib; t=pathlib.Path(tempfile.mkdtemp()); [p.mkdir(parents=True, exist_ok=True) if p.suffix=='' else (p.parent.mkdir(parents=True, exist_ok=True), p.write_text('{}')) for p in paths_for_layout(t, 'fashioniq_ilearn_resized')]; print('layout_missing:', missing_paths(paths_for_layout(t, 'fashioniq_ilearn_resized'))); print('wrapper_blockers:', len(validate_dataset(t)))"` | `layout_missing: []`<br>`wrapper_blockers: 3` |
| **Claim 5: F06 processor path** | **HOLDS** | `.venv/bin/python workbench/scripts/lavis_processor_parity.py` | Exit code `0`<br>`all 5 models LOADED (BlipCaptionProcessor, native punctuation stripping observed)` |

---

## 2. Detailed Claim Falsification Analysis

### Claim 1: DCNet Dataset Binding on Real Launch Path
- **Files Cited**: `workbench/replay/dcnet.py:34-93`, `workbench/backend/preflight.py:242-255`.
- **Subclaims**:
  - (a) Relative `data_dir` resolving to another tree is refused: **HOLDS**. `configured_data_dir` computes `cwd / declared` and checks `resolved != expected`.
  - (b) Relative `data_dir` staged as symlink to requested root is accepted: **FALSIFIED in production preflight**. In `workbench/backend/preflight.py:251-252`:
    ```python
    mismatch = binding_mismatch(checkpoint_path(model["model_id"], model["checkpoint_variants"][0],
                                                config.WORKBENCH_CHECKPOINT_ROOT), dataset_root)
    ```
    `evaluator_cwd` is omitted. `configured_data_dir` defaults `evaluator_cwd` to `os.getcwd()` (repo root `/home/heheboiz/data/cir`). Staging a symlink at the evaluator child process CWD (`workbench/third_party/DCNet/dataset/fashioniq -> dataset_root`) is rejected by `preflight_blockers` with: `Stage the canonical dataset at /home/heheboiz/data/cir/dataset/fashioniq as a symlink...`.
  - (c) Absolute `data_dir` naming another tree is refused: **HOLDS**. `resolved != expected` checks both absolute and relative paths.
  - (d) Refusal reaches `preflight_blockers`, which is consulted by `doctor.py`/`pc_readiness.py`/`prepare_model.py`/`evaluate_models.py`: **FALSIFIED**. In `workbench/backend/preflight.py:242`:
    ```python
    if model_id == "dcnet" and dataset_root is not None and source is not None:
    ```
    `doctor.py:151`, `pc_readiness.py:223`, and `prepare_model.py:40` all call `preflight_blockers` with `dataset_root=None`. The DCNet binding check is skipped for all three tools. Only `evaluate_models.py:357` passes `dataset_root`. Furthermore, if `config.json` is a directory or missing, `configured_data_dir` returns `(None, [])` (`dcnet.py:58-59`), and `binding_mismatch` returns `None`, silently bypassing the check. If `checkpoint_variants` is empty, `preflight.py:251` crashes with `IndexError`.
  - (e) Author's `config.json` is never rewritten: **HOLDS**. `workbench/third_party/DCNet/parse_config.py:39-42, 57-60` writes derived config to `<resume>/val/config.json`. `evaluate_models.py:598` copies config to `<evidence_dir>/config.json`. The original `<resume>/config.json` is never mutated.

### Claim 2: DCNet Parser Category Transposition
- **Files Cited**: `workbench/backend/metrics_extraction.py:584-601, 644-687, 707-723`.
- **Falsification Findings**:
  - Named category order in stdout: `named_order` is extracted sequentially from `\[(dress|shirt|toptee)\] comp corr r@10` regex matches (`metrics_extraction.py:646-652`).
  - Duplicated category lines: `if category not in named_order: named_order.append(category)` deduplicates by first appearance.
  - Incomplete category set: `if len(named_order) != 3: return unparsed` (`:678`) refuses truncated outputs.
  - Config contradiction: `if executed_order is not None and named_order != executed_order: return unparsed` (`:683`). A list declaring a different order returns `unparsed`.
  - Config as tuple: `_dcnet_category_order` checks `isinstance(categories, list)` (`:598`). Passing a tuple returns `None`, so `executed_order` is `None` and the config consistency check is bypassed (though `named_order` from stdout still protects against transposition).
  - Split separation: `__composition` and `__correction` metrics are keyed separately under distinct suffixes (`:708-713`).
  - `[comb_avg]` cross-check: `if declared and abs(float(declared.group(1)) * 100.0 - macro_mean) > 0.5: return unparsed` (`:720`) works as expected.

### Claim 3: CSMCIR Required Patch Set
- **Files Cited**: `workbench/backend/preflight.py:49-64, 130-191`.
- **Falsification Findings**:
  - Full set enforcement: `missing = [state for state in states if not state["applied"]]`; `summary["applied"] = not missing` (`preflight.py:176-178`).
  - Partial repair: When patch 1 is applied and patch 2 is missing, `applied: False`, and `detail` states `"1 of 2 required patches are not applied"`.
  - Notes formatting: `model_requirements` emits `"source repair incomplete: 1 of 2 required patches are not applied (src/validate_blip_csmcir.py)..."` (`:62-63`).
  - Edge case crash on symlink: When a declared patch target is a symlink to a regular file, `target.is_file()` (`:157`) evaluates to `True`. `sha256_file(target)` is called and raises `ValueError("artifact is not a regular file or non-symlink directory: ...")` (`registry.py:927`), crashing `approved_patch_state` with an unhandled exception.
  - Edge case false positive: When a patch declares `derived_sha256 == base_sha256`, `digest == patch.get("derived_sha256")` (`:161`) fires first, falsely reporting an unpatched file as `applied: True`.
  - Edge case empty source: When `source is None or not source.is_dir()`, `approved_patch_state` returns `patches: []` (`:147`). `model_requirements` formats this as `"source repair incomplete: 0 of 0 required patches are not applied ()"` (`:62`).
  - Scalar vs list consumers: `pc_readiness.py:278` and `evaluate_models.py:301` read `patch["patch_file"]` and `patch["patch_sha256"]`. When both patches are applied, these fields are comma-joined strings (`"workbench/patches/csmcir_models_init_compat.patch, workbench/patches/csmcir_validate_visualization_opt_in.patch"`). `runtime.py:2270` does not consume `approved_patch_state`; it reads `models.yaml` directly via `source_runtime_artifacts`.

### Claim 4: Constructor Input Set for LAVIS Wrappers
- **Files Cited**: `workbench/replay/common.py:102-135`, `workbench/replay/{hint,habit,intent,conesep,airknow,pair}.py`, `workbench/backend/fashioniq_layout.py:25-41`, `workbench/registry/models.yaml:124, 181, 279, 342`.
- **Falsification Findings**:
  - Subset request with dress-only root: In all 6 wrappers (`hint`, `habit`, `intent`, `conesep`, `airknow`, `pair`), `validate_dataset(dress_only, ("dress",))` returns 10 blockers (5 files/dirs each for missing `shirt` and `toptee`).
  - Complete root: `validate_dataset(complete, ("dress",))` returns 0 blockers across all 6 wrappers.
  - Native file opening: Native constructors in `HINT/datasets.py:33-47`, `HABIT/datasets.py:41-59`, `INTENT/datasets.py:39-57`, `ConeSep/datasets.py:37-55`, `AirKnow/datasets.py:51-69`, `PAIR/data/datasets.py:34-56` all execute `train_init_process()` and `get_test_data()` for all three categories before scoring.
  - Native image directory: Native code reads `resized_image/<category>/<id>.jpg`. `constructor_input_blockers` checks `dataset_root / "resized_image" / category`. Note: individual `.jpg` files inside `resized_image/<category>` are not verified by `validate_dataset` (only the directory).
  - Registry & layout contradiction: `models.yaml` declares `fashioniq_layout: fashioniq_ilearn_resized` for `habit` (`line 279`), `intent` (`line 342`), `conesep` (`line 181`), and `airknow` (`line 124`). `paths_for_layout(root, "fashioniq_ilearn_resized")` does **not** include `cap.{cat}.train.json`. As a result, `model_requirements` reports these models as clean without train captions, while their wrappers' `validate_dataset` blocks with 3 missing train caption errors.

### Claim 5: F06 LAVIS Processor Path
- **Files Cited**: `workbench/replay/lavis_processors.py:46-138`, `workbench/scripts/lavis_processor_parity.py:56-81`, `workbench/third_party/HINT/lavis/processors/blip_processors.py:28-68`.
- **Falsification Findings**:
  - Class built: Resolves `BlipCaptionProcessor` from pinned config `configs/models/blip2/blip2_pretrain.yaml`. Default parameters: `prompt=""`, `max_words=50`.
  - Output transformation: Verified native behaviour — strips punctuation (`re.sub(r"([.!\"()*#:;~])", " ", ...)`), lowercases, collapses whitespace, trims.
  - Isolation and cleanup: `lavis_processors.py:131-137` restores `sys.modules["lavis.processors"]` and `sys.modules["lavis"]` in `finally` block.
  - Class identity across repeated calls: Calling `text_processors` repeatedly returns instances of the exact same class object (`type(p1['eval']) is type(p2['eval'])`).
  - Zero base weights in wrappers: `hint.py:178`, `habit.py:147`, `intent.py:147`, `conesep.py:160`, `airknow.py:164` unpickle checkpoints using `load_trusted_whole_model(model_path, ...)`; none of them calls `load_model_and_preprocess` or accesses `/root/bert-base-uncased`.
  - Parity script: `workbench/scripts/lavis_processor_parity.py` runs cleanly with returncode `0`, all 5 models reporting `status: LOADED`.

---

## 3. Falsification Attempts That Failed to Falsify (Mechanisms That Held Up)

1. **DCNet category order transposition during `comb:` line attribution**:
   Attempted to feed stdout where named category lines were ordered `toptee, dress, shirt` while `comb:` line cells were permuted. The parser strictly bound the cells to `named_order` rather than the legacy static tuple `("dress", "shirt", "toptee")`. It extracted `toptee: 70.0/90.0`, `dress: 20.0/40.0`, `shirt: 30.0/50.0`.
2. **DCNet `[comb_avg]` checksum evasion**:
   Attempted to feed stdout where `macro_mean` (45.0) diverged from `[comb_avg]` (44.17). Parser rejected the stream with `EVALUATION_COMPLETED_METRICS_UNPARSED` due to tolerance threshold `0.5`.
3. **Partial CSMCIR patch state bypass**:
   Attempted to stage a checkout containing only patch 1 (`csmcir_models_init_compat.patch`). The old defect accepted this as `applied: True`. The new implementation returned `applied: False` with `"1 of 2 required patches are not applied"`. Staging only patch 2 similarly returned `applied: False`.
4. **Post-patch modification on CSMCIR**:
   Staged both patches and appended `# unapproved` to `src/validate_blip_csmcir.py`. `approved_patch_state` detected digest divergence from `derived_sha256` and returned `applied: False`.
5. **Single-category evaluation filtering in LAVIS wrappers**:
   Called `validate_dataset(dress_only_root, categories=('dress',))` across all six wrappers (`hint`, `habit`, `intent`, `conesep`, `airknow`, `pair`). All six returned 10 blockers demanding the files required by the cold cache initialization in native `train_init_process()` and `get_test_data()`.
6. **LAVIS processor isolation without torchaudio**:
   Imported and constructed `text_processors` across all 5 models without installing `torchaudio` or constructing a second BLIP-2 model. No import errors occurred, and `sys.modules` was restored after execution.

---

## 4. Reversion Test Falsification Analysis (`test_r9_defect_regressions.py`)

Each defect fix was reverted in an isolated copy of the repository (`/tmp/cir_copy`) and the sharpest regression test per defect was executed:

| Defect Area | Test Name | Reversion Change in Code | Test Result on Revert | Falsified? |
|---|---|---|---|---|
| **Defect 1: DCNet binding** | `test_the_binding_refusal_reaches_the_shared_preflight_list` | Removed `if model_id == "dcnet"` block from `workbench/backend/preflight.py:242-255` | **FAILED** (`AssertionError: assert False where False = any(...)`) | **YES** |
| **Defect 2: DCNet parser** | `test_a_non_default_category_order_is_attributed_per_category` | Reverted `named_order` enumeration in `metrics_extraction.py:685` back to fixed tuple `("dress", "shirt", "toptee")` | **FAILED** (`AssertionError: assert {'r10': 30.0, 'r50': 50.0} == {'r10': 70.0, 'r50': 90.0}`) | **YES** |
| **Defect 3: CSMCIR patches** | `test_only_the_first_patch_is_not_the_full_repair` | Reverted `preflight.py:approved_patch_state` to loop-iteration-0 early return | **FAILED** (`AssertionError: assert True is False`) | **YES** |
| **Defect 4: LAVIS constructor** | `test_a_subset_request_still_requires_every_constructor_input[hint]` | Reverted `hint.py:validate_dataset` to return `blockers` without calling `constructor_input_blockers` | **FAILED** (`AssertionError: hint: a dress-only root must not preflight clean for a subset run; assert []`) | **YES** |
| **Defect 5: F06 processor** | `test_no_wrapper_builds_a_second_model_for_its_processor[hint]` | Added aliased call `from lavis.models import load_model_and_preprocess as loader; loader()` | **PASSED** (Test stayed green despite defect presence!) | **NO (Could not falsify)** |

---

## 5. Tests in `test_r9_defect_regressions.py` That Still Pass With Defect Present

1. `test_no_wrapper_builds_a_second_model_for_its_processor` (`lines 413-425`):
   - **Flaw**: Performs AST inspection checking if any AST call node has `id == "load_model_and_preprocess"` or `attr == "load_model_and_preprocess"`.
   - **Passes when defect present**: If the wrapper imports the function with an alias (`from lavis.models import load_model_and_preprocess as get_proc; get_proc(...)`) or invokes it via `getattr(sys.modules["lavis.models"], "load_model_and_preprocess")(...)`, the AST check remains green while constructing a second model.
2. `test_the_shared_helper_is_the_one_the_wrappers_call` (`lines 380-388`):
   - **Flaw**: Asserts `"constructor_input_blockers" in source` as a plain string containment test on `workbench/replay/{wrapper}.py`.
   - **Passes when defect present**: A commented-out mention (`# constructor_input_blockers`) satisfies the assertion even if the wrapper never calls the helper.
3. `test_the_declared_config_is_the_one_the_model_class_publishes` (`lines 427-444`):
   - **Flaw**: Asserts that `f'"{config}"' in wrapper_source`.
   - **Passes when defect present**: String containment test only; does not execute `_load_replay_modules` or verify that the config is parsed at runtime.
4. `test_the_staged_run_records_the_config_it_read` (`lines 154-168`):
   - **Flaw**: Only calls `read_executed_config(plan)`.
   - **Passes when defect present**: Never invokes `evaluate_models.execute` or `invoke_evaluator`; does not verify that `evaluate_models.py:598` copies `config.json` into the run's evidence directory at runtime.
5. `test_the_binding_refusal_reaches_the_shared_preflight_list` (`lines 41-78`):
   - **Flaw**: Passes `dataset_root=requested` explicitly and mocks `model_requirements`.
   - **Passes when defect present**: Masks the fact that `preflight_blockers` defaults `evaluator_cwd` to `os.getcwd()` rather than `source_cwd`, and masks the fact that production callers (`doctor.py`, `pc_readiness.py`, `prepare_model.py`) pass `dataset_root=None` and skip the check entirely.

---

## 6. Concrete Defects Found with Minimal Reproductions

### Defect A: `preflight_blockers` omits `evaluator_cwd` for DCNet binding check
- **Location**: `workbench/backend/preflight.py:251-252`
- **Root Cause**: `binding_mismatch(checkpoint_path(...), dataset_root)` is called without passing `evaluator_cwd=source`. `configured_data_dir` falls back to `os.getcwd()` (the repo root `/home/heheboiz/data/cir`). Staging a symlink inside the checkout (`workbench/third_party/DCNet/dataset/fashioniq -> dataset_root`) where the evaluator actually runs is rejected.
- **Minimal Reproduction**:
  ```bash
  .venv/bin/python -c "
  import tempfile, os
  from pathlib import Path
  from workbench.replay.dcnet import binding_mismatch

  with tempfile.TemporaryDirectory() as tmp:
      t = Path(tmp)
      ckpt = t / 'ckpt'; ckpt.mkdir()
      (ckpt / 'config.json').write_text('{\"data_loader\": {\"args\": {\"data_dir\": \"dataset/fashioniq\"}}}')
      ds = t / 'dataset'; ds.mkdir()
      source = t / 'DCNet'; source.mkdir()
      (source / 'dataset').mkdir()
      (source / 'dataset' / 'fashioniq').symlink_to(ds)

      # Production preflight_blockers calls binding_mismatch without evaluator_cwd:
      err = binding_mismatch(ckpt, ds)
      assert err is not None, 'Expected pass but failed'
      print('OBSERVED REFUSAL DESPITE VALID STAGED SYMLINK:\n', err)
  "
  ```
- **Observed Output**:
  ```text
  OBSERVED REFUSAL DESPITE VALID STAGED SYMLINK:
  DCNet dataset binding mismatch: the evaluator resolves 'dataset/fashioniq' to /home/heheboiz/data/cir/dataset/fashioniq, not the verified dataset root /tmp/.../dataset. Stage the canonical dataset at /home/heheboiz/data/cir/dataset/fashioniq as a symlink...
  ```

---

### Defect B: DCNet binding check bypassed when `dataset_root=None`
- **Location**: `workbench/backend/preflight.py:242`
- **Root Cause**: Guarded by `if model_id == "dcnet" and dataset_root is not None and source is not None:`. Callers that do not pass `dataset_root` (`doctor.py:151`, `pc_readiness.py:223`, `prepare_model.py:40`) completely bypass the binding check.
- **Minimal Reproduction**:
  ```bash
  .venv/bin/python -c "
  from workbench.backend.operator_config import resolve_config
  from workbench.backend.registry import model_by_id, checkpoint_path
  from workbench.backend.preflight import preflight_blockers

  cfg = resolve_config()
  m = model_by_id('dcnet')
  ckpt = checkpoint_path(m['model_id'], m['checkpoint_variants'][0], cfg.WORKBENCH_CHECKPOINT_ROOT)
  ckpt.mkdir(parents=True, exist_ok=True)
  (ckpt / 'config.json').write_text('{\"data_loader\": {\"args\": {\"data_dir\": \"dataset/fashioniq\"}}}')

  # As doctor.py calls it:
  b_doc = preflight_blockers(m, cfg, canonical_root=cfg.FASHIONIQ_ROOT)
  assert not any('binding mismatch' in b for b in b_doc)
  print('doctor.py skips check: binding mismatch blocker present?', any('binding mismatch' in b for b in b_doc))

  # Clean up
  (ckpt / 'config.json').unlink()
  ckpt.rmdir()
  "
  ```
- **Observed Output**:
  ```text
  doctor.py skips check: binding mismatch blocker present? False
  ```

---

### Defect C: `approved_patch_state` crashes with unhandled `ValueError` on symlinked patch target
- **Location**: `workbench/backend/preflight.py:157-160`
- **Root Cause**: `target.is_file()` returns `True` for symlinks pointing to regular files. Line 160 invokes `sha256_file(target)`, which raises `ValueError: artifact is not a regular file or non-symlink directory: ...`. The exception is not caught and crashes preflight.
- **Minimal Reproduction**:
  ```bash
  .venv/bin/python -c "
  import tempfile, hashlib
  from pathlib import Path
  from workbench.backend.preflight import approved_patch_state

  with tempfile.TemporaryDirectory() as tmp:
      t = Path(tmp)
      src = t / 'src'; src.mkdir()
      real = t / 'real.py'; real.write_text('content')
      (src / 'target.py').symlink_to(real)

      model = {'model_id': 'csmcir', 'approved_source_patches': [{
          'path': 'target.py', 'patch_file': 'p.patch', 'patch_sha256': 's',
          'base_sha256': 'b', 'derived_sha256': hashlib.sha256(b'content').hexdigest()
      }]}
      approved_patch_state(model, src)
  "
  ```
- **Observed Output**:
  ```text
  ValueError: artifact is not a regular file or non-symlink directory: /tmp/.../src/target.py
  ```

---

### Defect D: Layout declaration contradiction for HABIT, INTENT, CONESEP, and AIRKNOW
- **Location**: `workbench/registry/models.yaml:124, 181, 279, 342` vs `workbench/replay/{habit,intent,conesep,airknow}.py`
- **Root Cause**: `models.yaml` sets `fashioniq_layout: fashioniq_ilearn_resized`. `paths_for_layout` for that layout does not include `cap.{cat}.train.json`. However, all four wrappers call `constructor_input_blockers(..., train_captions=True)`, which requires `cap.{cat}.train.json`. Preflight via `model_requirements` reports the dataset ready, but the wrapper preflight blocks execution.
- **Minimal Reproduction**:
  ```bash
  .venv/bin/python -c "
  import tempfile, pathlib
  from workbench.backend.fashioniq_layout import paths_for_layout, missing_paths
  from workbench.replay.habit import validate_dataset

  t = pathlib.Path(tempfile.mkdtemp())
  for p in paths_for_layout(t, 'fashioniq_ilearn_resized'):
      if p.suffix == '': p.mkdir(parents=True, exist_ok=True)
      else: p.parent.mkdir(parents=True, exist_ok=True); p.write_text('{}')

  print('paths_for_layout missing:', missing_paths(paths_for_layout(t, 'fashioniq_ilearn_resized')))
  print('habit validate_dataset blockers:', validate_dataset(t))
  "
  ```
- **Observed Output**:
  ```text
  paths_for_layout missing: []
  habit validate_dataset blockers: ['HABIT dataset requirement missing: /tmp/.../captions/cap.dress.train.json', 'HABIT dataset requirement missing: /tmp/.../captions/cap.shirt.train.json', 'HABIT dataset requirement missing: /tmp/.../captions/cap.toptee.train.json']
  ```
