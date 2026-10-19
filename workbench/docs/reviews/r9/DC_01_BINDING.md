# DCNet Dataset Binding Audit

## 1. Reachability of `validate_dataset_binding` and Call Chain

### Dead Code Assessment
`workbench/replay/dcnet.py:validate_dataset_binding` is **dead code**. It is unreachable from any production entry point (`pc.sh`, `evaluate_models.py`, `preflight_blockers`, `doctor.py`, `pc_readiness.py`, or the job queue API).

### Implementation Location
Defined at `workbench/replay/dcnet.py:32`:
```python
def validate_dataset_binding(checkpoint_root: Path, dataset_root: Path) -> list[str]:
```
Called only at `workbench/replay/dcnet.py:114` inside `preflight(...)`:
```python
def preflight(
    checkpoint_root: Path,
    dataset_root: Path,
    source_root: Path,
    categories: tuple[str, ...] = CATEGORIES,
) -> list[str]:
    return (
        validate_source(source_root)
        + validate_run_directory(checkpoint_root)
        + validate_dataset_binding(checkpoint_root, dataset_root)
        + validate_dataset(dataset_root, categories)
    )
```
`preflight(...)` is called only by `workbench/replay/dcnet.py:main` at line 128 under CLI check `if __name__ == '__main__': sys.exit(main())` (`workbench/replay/dcnet.py:141`).

### Production Path Analysis

1. **`pc.sh`**:
   - `pc.sh:2390` executes `run_preflight "$DOCTOR_MODE"`.
   - `pc.sh:278-360`: `run_preflight` performs host shell checks (OS `/etc/os-release`, git repo, RAM `/proc/meminfo`, disk `df`, Docker socket, NVIDIA container toolkit). It never invokes Python preflight or any model-specific validation.
   - Reproduction workflows in `pc.sh` delegate to `run.sh` or `pipeline.py`.
   - Call chain to `dcnet.preflight`: **NONE**.

2. **`preflight_blockers` (`workbench/backend/preflight.py:162-216`)**:
   - `preflight_blockers(model, config, ...)` calls `model_requirements(...)` (`preflight.py:168`), checks `source_clean_and_pinned` (`:174`), checks `adapter.command` (`:180`), appends `environment_blockers(model)` (`:210`), and special-cases `model_id == "encoder"` (`:211-215`).
   - Does not import or call `workbench.replay.dcnet`, `dcnet.preflight`, or `validate_dataset_binding`.
   - Call chain to `dcnet.preflight`: **NONE**.

3. **`doctor.py` (`workbench/scripts/doctor.py`)**:
   - `doctor.py:151` calls `preflight_blockers(model, config, canonical_root=config.FASHIONIQ_ROOT)`.
   - Does not import or call `workbench.replay.dcnet`.
   - Call chain to `dcnet.preflight`: **NONE**.

4. **`pc_readiness.py` (`workbench/scripts/pc_readiness.py`)**:
   - `pc_readiness.py:223` calls `preflight_blockers(model, config)`.
   - Does not import or call `workbench.replay.dcnet`.
   - Call chain to `dcnet.preflight`: **NONE**.

5. **`evaluate_models.py` (`workbench/scripts/evaluate_models.py`)**:
   - `evaluate_models.py:357` calls `preflight_blockers(model, config, dataset_root=dataset_root)`.
   - `evaluate_models.py:430` gets command from `ADAPTERS[model_id].build_command(request)`.
   - For DCNet, `ADAPTERS["dcnet"]` is `DCNetAdapter` (`workbench/backend/adapters/models.py:172-178`):
     ```python
     class DCNetAdapter(OfficialScriptAdapter):
         model_id = "dcnet"
         script = "test.py"

         def command(self, source, checkpoint, request: EvalRequest) -> list[str]:
             return [_runtime.python_executable(self.model), str(source / self.script), "--resume", str(checkpoint)]
     ```
   - `DCNetAdapter` invokes upstream `workbench/third_party/DCNet/test.py` directly. Unlike `hint`, `tgcir`, `pair`, `encoder`, or `limn`, `DCNetAdapter` does **not** route through `_replay_command` (`workbench/backend/adapters/models.py:27-46`), so it never references `workbench/replay/dcnet.py`.
   - `evaluate_models.py:594` executes child process `subprocess.Popen(plan.command, cwd=plan.cwd, ...)`.
   - Call chain to `dcnet.preflight`: **NONE**.

6. **Import Graph Verification**:
   - Command: `git grep "workbench.replay.dcnet"`
   - Output: `workbench/tests/test_dcnet_replay.py:8:from workbench.replay.dcnet import (`
   - Zero occurrences in `workbench/backend/`, `workbench/scripts/`, `workbench/docker/`, or `pc.sh`.

---

## 2. Actual Working Directory of the DCNet Evaluator Child Process

### In `workbench/backend/adapters/models.py` (`DCNetAdapter`)
Lines 172-178:
```python
class DCNetAdapter(OfficialScriptAdapter):
    model_id = "dcnet"
    script = "test.py"

    def command(self, source, checkpoint, request: EvalRequest) -> list[str]:
        return [_runtime.python_executable(self.model), str(source / self.script), "--resume", str(checkpoint)]
```
`DCNetAdapter.command()` takes `(self, source, checkpoint, request)`. It has **no `cwd=` argument** and does not set or return working directory metadata.
Base class `OfficialScriptAdapter.build_command` (`workbench/backend/adapters/models.py:16-24`) and `ModelAdapter` (`workbench/backend/adapters/base.py:21-39`) also have no `cwd` parameter or attribute.

### In `run.sh` (`workbench/docker/run.sh` and `workbench/run.sh`)
- `workbench/docker/run.sh`:
  Lines 205-207 and 221-223:
  ```bash
  cmd_prepare() {
      local model="${1:-}"
      case "${model}" in
          limn|csmcir) ;;
          *) say "usage: run.sh prepare <limn|csmcir>"; return 2;;
      esac
  ```
  ```bash
  cmd_pilot() {
      local model="${1:-}"
      case "${model}" in
          limn|csmcir) ;;
          *) say "usage: run.sh pilot <limn|csmcir>"; return 2;;
      esac
  ```
  `workbench/docker/run.sh` does not support `dcnet` and does not set any working directory for DCNet. Dockerfile sets container default `WORKDIR /repo` (`workbench/docker/Dockerfile:19`).
- `workbench/run.sh`:
  Lines 1-7:
  ```bash
  #!/usr/bin/env sh
  if [ "${1-}" = "autopilot" ]; then
      shift
      exec python3 "$(dirname "$0")/scripts/autopilot.py" "$@"
  fi
  exec python3 "$(dirname "$0")/scripts/pipeline.py" "$@"
  ```
  Does not set working directory for child processes.

### In `workbench/scripts/evaluate_models.py`
Lines 214-215:
```python
def source_cwd(model: dict, source: Path) -> Path:
    return source / "src" if model["model_id"] == "csmcir" else source
```
Line 434 (inside `guarded_plan`):
```python
    return EvaluationPlan(
        model["model_id"],
        checkpoint["checkpoint_id"],
        protocol,
        dataset_root,
        source,
        checkpoint_file,
        source_cwd(model, source),
        command,
...
```
Line 594 (inside `_run_evaluation`):
```python
        process = subprocess.Popen(plan.command, cwd=plan.cwd, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=child_env)
```

### Actual Evaluator Child Working Directory
For `dcnet`, `source_cwd` returns `source` (`config.WORKBENCH_THIRD_PARTY_ROOT / model["source_dir"]`), resolving to:
`/home/heheboiz/data/cir/workbench/third_party/DCNet` (host) or `/repo/workbench/third_party/DCNet` (container).

---

## 3. Shipped `config.json` (`data_loader.args.data_dir`) in Pinned Repo

Three JSON files exist under `workbench/third_party/DCNet/configs/`:

1. `workbench/third_party/DCNet/configs/ce/fashioniq_dcnet_deep.json:20`:
   ```json
               "data_dir": "dataset/fashioniq",
   ```
   Literal string: `"dataset/fashioniq"`

2. `workbench/third_party/DCNet/configs/ce/shoe_dcnet.json:21`:
   ```json
               "data_dir": "dataset/shoe",
   ```
   Literal string: `"dataset/shoe"`

3. `workbench/third_party/DCNet/configs/ce/fashioniq_dcnet.json:20`:
   ```json
               "data_dir": "dataset/fashioniq",
   ```
   Literal string: `"dataset/fashioniq"`

### Absolute Path Evaluation
**None** of the shipped configs set an absolute path. All three values (`"dataset/fashioniq"`, `"dataset/shoe"`) are relative path strings.

---

## 4. Runtime Resolution of `data_dir` in `parse_config.py` and `data_loader`

### `parse_config.py` Behavior
`workbench/third_party/DCNet/parse_config.py` does not resolve, absolutize, or join `data_dir` to any path.
- Line 28:
  ```python
  self._config = read_json(self.cfg_fname)
  ```
- Lines 65-70:
  ```python
  def init(self, name, module, *args, **kwargs):
      module_name = self[name]['type']
      module_args = dict(self[name]['args'])
      msg = "Overwriting kwargs given in config file is not allowed"
      assert all([k not in module_args for k in kwargs]), msg
      module_args.update(kwargs)
      return getattr(module, module_name)(*args, **module_args)
  ```
  `init` forwards `config['data_loader']['args']` unmodified to `ExpertDataLoader`.

### `data_loader` Path Joining
In `workbench/third_party/DCNet/data_loader/data_loaders.py:16, 36, 58, 173, 182`, `data_dir` is passed into `CE_dataset.py:CE`.

In `workbench/third_party/DCNet/data_loader/CE_dataset.py`:
- Line 22:
  ```python
  self.image_dir = os.path.join(data_dir, "resized_images")
  ```
- Line 27:
  ```python
  self.data = read_json(Path(data_dir + f"/image_splits/split.{category}.{split.split('_')[0]}.json"))
  ```
- Lines 34-35:
  ```python
  elif text_feat == "glove":
      text_path = f"/captions/cap.{category}.glove.{split}.pkl"
      self.data = memcache(data_dir + text_path)
  ```
- Lines 72-73:
  ```python
  candidate_fname = candidate_id + '.jpg'
  candidate_image_orig = Image.open(os.path.join(self.image_dir, candidate_fname)).convert('RGB')
  ```
- Lines 77-78:
  ```python
  target_fname = target_id + '.jpg'
  target_image_orig = Image.open(os.path.join(self.image_dir, target_fname)).convert('RGB')
  ```

### Resolution Target
`data_dir` is joined as a string or `Path` without base directory rebinding. Under Python/POSIX file APIs (`open`, `Path.read_text`, PIL `Image.open`), relative paths resolve strictly against the **process Current Working Directory (CWD)**.
Because `evaluate_models.py:594` executes the evaluator with `cwd = source` (`workbench/third_party/DCNet`), `data_dir` resolves at runtime against:
`workbench/third_party/DCNet/` -> `workbench/third_party/DCNet/dataset/fashioniq/`.
It does **not** resolve against the caller's `--dataset-root` (e.g. `/data/FashionIQ`).

---

## 5. Dataset-Root Paths Opened by DCNet Evaluator at Runtime

During validation evaluation (`test.py --resume <run_dir> --mode val`), the evaluator opens the following paths under `data_dir`:

1. **Validation Split JSON**:
   - Path: `<data_dir>/image_splits/split.{category}.val.json`
   - File: `workbench/third_party/DCNet/data_loader/CE_dataset.py:27`
   - Opened for categories `dress`, `shirt`, `toptee` when `split == 'val_trg'`.

2. **Preprocessed GloVe Caption Embeddings (PKL)**:
   - Path: `<data_dir>/captions/cap.{category}.glove.val.pkl` (or `cap.{category}.w2v.val.pkl` if `text_feat == "w2v"`)
   - File: `workbench/third_party/DCNet/data_loader/CE_dataset.py:34-35`
   - Opened for each category via `memcache(...)` (unpickled into memory).

3. **Flat Resized Images (JPEG)**:
   - Path: `<data_dir>/resized_images/{candidate_id}.jpg` and `<data_dir>/resized_images/{target_id}.jpg`
   - File: `workbench/third_party/DCNet/data_loader/CE_dataset.py:22, 73, 78`
   - Opened via `PIL.Image.open(...)`.

4. **Attribute Vocabulary (Conditional)**:
   - Path: `<data_dir>/attributes/dict.attr.json`
   - File: `workbench/third_party/DCNet/test.py:30`:
     ```python
     if "attr" in config['experts']['modalities']:
         attr_vocab = Vocabulary()
         attr_vocab.load(os.path.join(config['data_loader']['args']['data_dir'],'attributes/dict.attr.json'))
     ```
     (Inactive for default FashionIQ configs where modalities exclude `"attr"`).

### Trainer Inspection
`TrainerJoint` (`workbench/third_party/DCNet/trainer/trainer_joint.py`) and `BaseTrainer` (`workbench/third_party/DCNet/base/base_trainer.py:193-195`) do not open dataset files; they load checkpoint weights `<resume>/trained_model.pth`.

### Required Dataset-Root Comparison Coverage
A correct dataset-root comparison must verify that the path resolved from `data_loader.args.data_dir` against evaluator CWD (`workbench/third_party/DCNet/`) matches the operator's verified dataset root providing:
- `resized_images/` containing `{image_id}.jpg`
- `image_splits/split.{category}.val.json`
- `captions/cap.{category}.glove.val.pkl`

---

## 6. Adapter and Upstream `config.json` Handling in Run Directory

### Adapter Behavior
`DCNetAdapter` (`workbench/backend/adapters/models.py:172-178`) does **not** copy, derive, or modify `config.json`. It passes `--resume <checkpoint>` directly on the CLI.

### Upstream Evaluator Behavior (`parse_config.py`)
`workbench/third_party/DCNet/parse_config.py` loads and mutates configuration at runtime:
1. Lines 18-28:
   ```python
   if args.resume:
       self.resume = Path(args.resume)
       self.cfg_fname = self.resume / 'config.json'
   ...
   self._config = read_json(self.cfg_fname)
   ```
2. Lines 39-42:
   ```python
   if args.mode == 'val':
       save_dir = Path(self.config['trainer']['save_dir'])
       self._save_dir = self.resume / "val"
       self._log_dir = self.resume / "val"
       self._config['data_loader']['args']['mode'] = 'val'
   ```
3. Lines 57-60:
   ```python
   self.save_dir.mkdir(parents=True, exist_ok=True)
   self.log_dir.mkdir(parents=True, exist_ok=True)
   write_json(self.config, self._save_dir / 'config.json')
   ```

### Output Location and Original Preservation
- **Where written**: `<checkpoint_dir>/val/config.json` (or `<checkpoint_dir>/test/config.json` in test mode).
- **Author's original preserved**: **Yes**. The author's original file `<checkpoint_dir>/config.json` is read at line 28 and never overwritten. The derived copy (with modified `mode: 'val'`) is written into the child directory `<checkpoint_dir>/val/`.

---

## 7. Existing Regression Tests and Function Calling

### Tests in `workbench/tests/test_f05_f08_f09_verification.py`
In `workbench/tests/test_f05_f08_f09_verification.py:117-168`:
1. `test_f08_dcnet_config_data_dir_resolution`:
   Asserts `raw_data_dir == "dataset/fashioniq"` in `workbench/third_party/DCNet/configs/ce/fashioniq_dcnet.json` and that resolving against `dcnet_source` yields `dcnet_source / "dataset" / "fashioniq"`, differing from mock dataset path.
2. `test_f08_preflight_refusal_on_data_dir_mismatch`:
   Defines an inline helper inside the test body (`lines 133-146`):
   ```python
   def validate_dcnet_data_dir_alignment(checkpoint_root: Path, source_root: Path, dataset_root: Path) -> list[str]:
       cfg_file = checkpoint_root / "config.json"
       if not cfg_file.is_file():
           return [f"DCNet config missing: {cfg_file}"]
       cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
       data_dir_str = cfg.get("data_loader", {}).get("args", {}).get("data_dir")
       if not data_dir_str:
           return ["DCNet config missing data_loader.args.data_dir"]
       configured_path = (source_root / data_dir_str).resolve() if not Path(data_dir_str).is_absolute() else Path(data_dir_str).resolve()
       if configured_path != dataset_root.resolve():
           return [
               f"DCNet dataset root mismatch: checkpoint config.json data_loader.args.data_dir '{data_dir_str}' "
               f"resolves to '{configured_path}', but evaluator received dataset_root '{dataset_root.resolve()}'. "
               f"To resolve without modifying author config.json, stage a symlink at {source_root / data_dir_str} -> {dataset_root.resolve()}."
           ]
       return []
   ```
   Asserts that `validate_dcnet_data_dir_alignment` returns a blocker when unlinked, and returns `[]` when `(src_dir / "dataset" / "fashioniq").symlink_to(ds_dir)`.

### Production vs Test Copy
- Does it call the production function?
  **NO**. It calls a locally defined copy (`validate_dcnet_data_dir_alignment`). It never calls `workbench/replay/dcnet.py:validate_dataset_binding`.
- Semantic contradiction:
  - Test helper `validate_dcnet_data_dir_alignment` accepts relative paths when a symlink is staged at `source_root / data_dir_str`.
  - Production function `workbench/replay/dcnet.py:validate_dataset_binding` unconditionally blocks any relative path:
    ```python
    if not declared_path.is_absolute():
        blockers.append(
            f"DCNet config data_dir is a relative path ({declared!r}), so the evaluator reads it "
            f"relative to its own working directory, not the --dataset-root {dataset_root} the "
            "operator passed. Bind it absolutely (or run the evaluator from the directory the "
            "literal was written for) before trusting which tree was evaluated.")
    ```

### Tests in `workbench/tests/test_dcnet_replay.py`
Lines 1-48 assert:
- `MODEL["model_id"] == "dcnet"`, `PROTOCOL`, and `REQUIRED_RUN_FILES` (lines 19-23).
- `validate_source(dcnet_source) == []` (lines 26-29).
- `validate_run_directory` detects missing `config.json` and `trained_model.pth` (lines 32-37).
- `validate_dataset` detects missing `resized_images` and `glove` PKLs (lines 40-48).
`test_dcnet_replay.py` does **not** call or test `validate_dataset_binding`.

---

## Verified claims

| Claim | Command / Verification Route | Observed Output |
|---|---|---|
| `validate_dataset_binding` not imported in backend or scripts | `git grep "validate_dataset_binding"` | Matches only `workbench/replay/dcnet.py:32,114` and `workbench/docs/MODEL_STATUS_SPEC.md:228` |
| `workbench.replay.dcnet` imported only in test file | `git grep "workbench.replay.dcnet"` | `workbench/tests/test_dcnet_replay.py:8:from workbench.replay.dcnet import (` |
| `DCNetAdapter` does not accept or pass `cwd=` | Code inspection `workbench/backend/adapters/models.py:172-178` | Returns `[_runtime.python_executable(self.model), str(source / self.script), "--resume", str(checkpoint)]` with no `cwd` |
| `evaluate_models.py` sets evaluator child `cwd` to `source` | Code inspection `workbench/scripts/evaluate_models.py:214-215, 434, 594` | `source_cwd` returns `source`; `Popen(plan.command, cwd=plan.cwd)` executes with `plan.cwd = source` |
| `run.sh` does not support dcnet | Code inspection `workbench/docker/run.sh:205-207, 221-223` | `case "${model}" in limn|csmcir) ;; *) say "usage: run.sh ..."; return 2;; esac` |
| Shipped DCNet configs set relative paths | `.venv/bin/python -c "import json; [print(p, json.load(open(p))['data_loader']['args']['data_dir']) for p in ['workbench/third_party/DCNet/configs/ce/fashioniq_dcnet_deep.json', 'workbench/third_party/DCNet/configs/ce/shoe_dcnet.json', 'workbench/third_party/DCNet/configs/ce/fashioniq_dcnet.json']]"` | `fashioniq_dcnet_deep.json dataset/fashioniq`, `shoe_dcnet.json dataset/shoe`, `fashioniq_dcnet.json dataset/fashioniq` |
| No shipped DCNet config sets absolute path | Code inspection of all 3 JSON configs under `workbench/third_party/DCNet/configs/ce/` | All values are relative strings (`dataset/fashioniq`, `dataset/shoe`) |
| `parse_config.py` does not resolve `data_dir` | Code inspection `workbench/third_party/DCNet/parse_config.py:28, 65-70` | Reads `cfg_fname`, forwards `data_loader.args` untouched to module |
| `data_loader` joins `resized_images`, `image_splits`, `captions` to `data_dir` | Code inspection `workbench/third_party/DCNet/data_loader/CE_dataset.py:22, 27, 34-35, 73, 78` | `os.path.join(data_dir, "resized_images")`, `Path(data_dir + "/image_splits/...")`, `data_dir + "/captions/..."` |
| Evaluator writes derived config to `<resume>/val/config.json` | Code inspection `workbench/third_party/DCNet/parse_config.py:40-42, 60` | `self._save_dir = self.resume / "val"`; `write_json(self.config, self._save_dir / 'config.json')` |
| Author original `<resume>/config.json` is preserved | Code inspection `workbench/third_party/DCNet/parse_config.py:20, 28, 60` | `self.cfg_fname = self.resume / 'config.json'` read-only; write target is `self._save_dir / 'config.json'` |
| Regression test calls local copy, not production function | Code inspection `workbench/tests/test_f05_f08_f09_verification.py:133-167` | Defines and calls inner `def validate_dcnet_data_dir_alignment(...)`; no import of `validate_dataset_binding` |
| Test suite passes on existing tests | `.venv/bin/python -m pytest workbench/tests/test_dcnet_replay.py workbench/tests/test_f05_f08_f09_verification.py` | `15 passed in 0.12s` |
