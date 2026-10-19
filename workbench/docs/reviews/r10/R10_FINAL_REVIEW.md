# R10 Final Adversarial Review

Adversarial audit of R10 remediation in `git diff e46ad65..cf84cfa`.
Repository: `/home/heheboiz/data/cir` at commit `cf84cfa` (docs at `937f64d`).
Python environment: `.venv/bin/python` (Python 3.13.12).

---

## 1. Verdict Table

| Claim | Target | Verdict | Command | Summary Raw Output |
|---|---|---|---|---|
| **Claim 1 (R10-06)**: Verifier no longer certifies packages an isolated interpreter cannot import | `workbench/backend/runtime.py:_inherits_base_site_packages`, `inspect_environment_distributions` | **FALSIFIED** | `.venv/bin/python -c "<conda-meta attack script>"` | `inherits_base_site_packages: True`<br>`probe find_spec: pip: None`<br>`inspect_env versions: ['shadowpkg', 'foo', 'pip']`<br>`prefix=None raised TypeError` |
| **Claim 2 (R10-01)**: DCNet validator agrees with native evaluator | `workbench/replay/dcnet.py:configured_data_dir`, `binding_mismatch` | **PARTIAL** | `.venv/bin/python -c "<DCNet 8-shape attack script>"` | `config.json as dir: ConfigParser -> IsADirectoryError; binding_mismatch -> None (silent pass)`<br>`data_dir=[...] -> TypeError: argument should be str, not 'list'`<br>`data_dir=12345 -> TypeError: argument should be str, not 'int'` |
| **Claim 3 (R10-02)**: Staging is authorized and nothing else is | `workbench/registry/models.yaml`, `workbench/backend/registry.py`, `workbench/scripts/prepare_dataset.py`, `workbench/backend/vendored_source.py` | **HOLDS** | `.venv/bin/python -c "<DCNet staging and integrity attack script>"` | `staged: verified=True, unauthorized_untracked=[]`<br>`extra file: verified=False, unauthorized_untracked=['dataset/extra.py']`<br>`second link: verified=False, unauthorized_untracked=['second_link']`<br>`modified test.py: verified=False, tracked_modifications=['test.py']`<br>`check_only/dry_run parent created? False`<br>`stager on real dir: ok=False, preserves dir`<br>`stager on diff link: ok=False, preserves link`<br>`escaping links (../x, /abs, a/../../b, sub/sub2/link): REJECTED` |
| **Claim 4 (R10-03)**: Checkpoint import path survives processor setup | `workbench/replay/lavis_processors.py:prepare`, `_patch_registry_idempotence`, `reset` | **PARTIAL** | `.venv/bin/python -c "<lavis_processors edge cases & cleanup attack script>"` | `register_path('num_path', 12345) when existing '12345': ACCEPTED WITHOUT ERROR (bypasses assert isinstance(path, str))`<br>`after reset(): sys.path retains checkout_a, import lavis.unique_mod SUCCEEDS`<br>`patch leaks to new Registry instance: True` |
| **Claim 5 (R10-05)**: Scanner follows launcher, profiles satisfiable | `workbench/backend/adapters/models.py:launcher, launcher_roots`, `workbench/backend/environment_planner.py:_evaluator_imports` | **PARTIAL** | `.venv/bin/python -c "<launcher check & traced vs declared comparison script>"` | `(a) 15/15 launcher declarations match command()[1]`<br>`(b) 6/6 launcher_roots dirs exist and sys.path.insert confirmed`<br>`(c) Traced closure includes 20-26 undeclared packages per model (apex, tensorflow, horovod, knn-cuda) marked UNKNOWN; declared packages (opencv-python-headless, pillow, pyyaml, pycocoevalcap, scikit-image) missing from traced AST` |
| **Claim 6 (R10-04)**: No stale blocker remains and no real blocker was lost | `workbench/backend/preflight.py:preflight_blockers`, `workbench/registry/models.yaml` | **HOLDS** | `.venv/bin/python -c "<preflight_blockers before vs after comparison script>"` | `airknow: 3 -> 2 blockers (removed source_blocker, retained BLIP-2 & EVA-ViT-g weights)`<br>`conesep: 1 -> 0 blockers`<br>`intent: 1 -> 0 blockers`<br>`no protective blocker lost` |

---

## 2. Attacks That Failed to Falsify

### Claim 1 (R10-06)
1. **Real isolated venv without system site packages**:
   Command:
   ```bash
   .venv/bin/python -c '
   import sys, subprocess, json
   from pathlib import Path
   from workbench.backend import runtime
   work = Path("/home/heheboiz/.cache/cir-pytest/claim1_v")
   venv_iso = work / "iso"
   subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv_iso)], check=True)
   py = venv_iso / "bin" / "python"
   ident = runtime._interpreter_identity(py)
   dist, _, _ = runtime.inspect_environment_distributions(str(ident["prefix"]), str(ident["base_prefix"]))
   probe = subprocess.run([str(py), "-I", "-c", "import importlib.util, json; print(json.dumps({p: importlib.util.find_spec(p) is not None for p in [\"pip\",\"fooowner\"]}))"], capture_output=True, text=True)
   print("probe:", probe.stdout.strip())
   print("dist:", dist)
   '
   ```
   Output:
   ```
   probe: {"pip": false, "fooowner": false}
   dist: {}
   ```
   Ground truth find_spec and verifier inventory match. Base packages not certified.

2. **Real venv with `--system-site-packages`**:
   Command:
   ```bash
   .venv/bin/python -c '
   import sys, subprocess, json
   from pathlib import Path
   from workbench.backend import runtime
   work = Path("/home/heheboiz/.cache/cir-pytest/claim1_v")
   venv_sys = work / "sys"
   subprocess.run([sys.executable, "-m", "venv", "--without-pip", "--system-site-packages", str(venv_sys)], check=True)
   py = venv_sys / "bin" / "python"
   ident = runtime._interpreter_identity(py)
   dist, _, _ = runtime.inspect_environment_distributions(str(ident["prefix"]), str(ident["base_prefix"]))
   probe = subprocess.run([str(py), "-I", "-c", "import importlib.util, json; print(json.dumps({p: importlib.util.find_spec(p) is not None for p in [\"pip\",\"fooowner\"]}))"], capture_output=True, text=True)
   print("probe:", probe.stdout.strip())
   print("dist:", {k: dist.get(k) for k in ["pip", "foo"] if k in dist})
   '
   ```
   Output:
   ```
   probe: {"pip": true, "fooowner": true}
   dist: {'pip': '26.0.1', 'foo': '1.0'}
   ```
   Base packages inherited legitimately.

3. **`pyvenv.cfg` with `include-system-site-packages = True` (capitalized)**:
   Case-insensitive parsing `value.strip().lower() == "true"` handles uppercase correctly. Both interpreter and verifier agree.

4. **`prefix == base_prefix`**:
   `str(base_prefix) != str(prefix)` evaluates to `False`. `base_roots` not appended; `primary_roots` reads the prefix site-packages. No contradiction.

---

### Claim 2 (R10-01)
1. **Relative literal with symlink (`data_dir: "dataset/fashioniq"`)**:
   Resolves through symlink to verified `dataset_root`. `binding_mismatch` returns `None`. Native evaluator and validator agree.
2. **Relative literal with real directory (`data_dir: "rel_real"`)**:
   Resolves inside checkout. `binding_mismatch` returns refusal string naming exact mismatch path.
3. **Absolute literal (`data_dir: "/path/to/expected"`)**:
   Matches expected root. `binding_mismatch` returns `None`.
4. **Literal with `..` (`data_dir: "sub/../target"`)**:
   Normalized by `Path.resolve()`. Evaluator and validator resolve identical path.
5. **Literal with trailing slash (`data_dir: "target/"`)**:
   Path resolution normalizes trailing slash; evaluator and validator agree.
6. **Environment override ignored**:
   Setting `WORKBENCH_DCNET_DATA_DIR` does not change `binding_mismatch` or `configured_data_dir`. The removed override is completely inert.

---

### Claim 3 (R10-02)
1. **Staging canonical dataset link on clean DCNet checkout copy**:
   Command:
   ```bash
   .venv/bin/python -c '
   from pathlib import Path
   from workbench.backend.registry import model_by_id
   from workbench.backend.vendored_source import source_provenance
   from workbench.scripts.prepare_dataset import prepare_canonical_link
   import shutil
   work = Path("/home/heheboiz/.cache/cir-pytest/claim3_audit")
   shutil.rmtree(work, ignore_errors=True); work.mkdir(parents=True)
   src = work / "DCNet"; shutil.copytree("workbench/third_party/DCNet", src)
   ds = work / "ds"; ds.mkdir()
   m = model_by_id("dcnet")
   ok, msgs = prepare_canonical_link(ds, src, m["canonical_dataset_link"], check_only=False, dry_run=False)
   prov = source_provenance(m, src)
   print("prepare:", ok, msgs)
   print("prov:", prov["verified"], prov["unauthorized_untracked"], prov["error"])
   '
   ```
   Output:
   ```
   prepare: True ['[OK] dataset link created: .../DCNet/dataset/fashioniq -> .../ds']
   prov: True [] None
   ```
   Integrity verification succeeds with authorized link.

2. **Integrity gate rejects unauthorized additions and modifications**:
   - Extra file in `dataset/` (`dataset/extra.py`): `verified: False`, `unauthorized_untracked: ['dataset/extra.py']`, error: `undeclared files present in checkout: 1 (first: dataset/extra.py)`.
   - Second undeclared link (`second_link -> ds`): `verified: False`, `unauthorized_untracked: ['second_link']`.
   - Modifying tracked source (`test.py`): `verified: False`, `tracked_modifications: ['test.py']`, error: `vendored source changed since vendoring: 1 modified, 0 removed (first: test.py)`.

3. **`check_only` and `dry_run` do not write parent directory**:
   Neither creates `dataset/` on checkout. Parent directory creation deferred until actual write.

4. **Stager protects existing occupants**:
   - Destination is existing directory: `prepare_canonical_link` returns `False`, message: `[BLOCKED] dataset destination already exists and is not the expected link: ...`. Existing directory preserved.
   - Destination is differently-targeted link: returns `False`, message: `[BLOCKED] dataset link resolves to ..., expected ...`. Target unchanged.

5. **Registry validation rejects escaping paths**:
   `../x`, `/abs/path`, `a/../../b` raise `ValueError("canonical_dataset_link must be a relative in-checkout path")`.
   `sub/sub2/link` raises `ValueError("canonical_dataset_link must name a path at most one directory deep")`.

---

### Claim 4 (R10-03)
1. **Repeated `prepare()` calls**:
   Calling `prepare()` twice with same checkout root is a no-op returning `{"reused": True}`.
2. **Refusal of second checkout in same process**:
   Calling `prepare(checkout_b)` after `prepare(checkout_a)` without `reset()` raises `ImportError: this process already prepared the lavis context for ...`.
3. **Conflict detection on different path**:
   Calling `registry.register_path("library_root", "/different/path")` raises `KeyError: "Name 'library_root' already registered."`.
4. **Instance invocation of classmethod**:
   `registry.register_path(...)` called directly on `Registry()` instance succeeds idempotently.

---

### Claim 5 (R10-05)
1. **Adapter `launcher` matches `command()[1]`**:
   All 15 registered adapters match their declared `launcher`:
   - `launcher == "replay"`: `encoder`, `hint`, `pair`, `airknow`, `conesep`, `habit`, `intent`, `clvc_net`, `tgcir`, `limn`. `command()[1]` is `workbench/replay/<model_id>.py`.
   - `launcher == "script"`: `csmcir`, `dcnet`, `combiner_rn50x4_noft`, `clip4cir_rn50x4_fullft`, `sprc`. `command()[1]` is `<source>/<script>`.
2. **`launcher_roots` validity**:
   - `HINT/lavis`: exists. `hint.py:149` executes `sys.path.insert(0, entry)`.
   - `AirKnow/lavis`: exists. `airknow.py:135` executes `sys.path.insert(0, entry)`.
   - `ConeSep/lavis`: exists. `conesep.py:132` executes `sys.path.insert(0, entry)`.
   - `HABIT/lavis`: exists. `habit.py:120` executes `sys.path.insert(0, entry)`.
   - `INTENT/lavis`: exists. `intent.py:120` executes `sys.path.insert(0, entry)`.
   - `LIMN/LIMN`: exists. `limn.py:157` executes `sys.path.insert(0, str(source_path))`.

---

### Claim 6 (R10-04)
1. **Blocker diff before vs after**:
   Command:
   ```bash
   .venv/bin/python -c '
   import yaml, subprocess
   from workbench.backend.preflight import preflight_blockers
   from workbench.backend.operator_config import resolve_config
   cfg = resolve_config()
   mb = {m["model_id"]: m for m in yaml.safe_load(subprocess.check_output(["git", "show", "e46ad65:workbench/registry/models.yaml"]))["models"]}
   ma = {m["model_id"]: m for m in yaml.safe_load(open("workbench/registry/models.yaml"))["models"]}
   for mid in ["airknow", "conesep", "intent"]:
       bb, ba = preflight_blockers(mb[mid], cfg), preflight_blockers(ma[mid], cfg)
       print(f"{mid}: before={len(bb)} after={len(ba)} diff_removed={set(bb)-set(ba)} diff_added={set(ba)-set(bb)}")
   '
   ```
   Output:
   ```
   airknow: before=3 after=2 diff_removed={"source cannot run: no runnable evaluation path is registered for this model: the pinned evaluator is a callable with no CLI or checkpoint loader, the adapter deliberately declines to build a command, and the required correction dictionaries and resized images have no published source; the paper row is the 0%-noise row and no 0% checkpoint is published (see workbench/docs/reviews/models for the model's audit). An earlier note in this registry claimed a tensor-shape failure in the similarity computation; that claim was WRONG and is withdrawn - np.matmul broadcasts and the ranking matrix has the shape the indexing expects"} diff_added=set()
   conesep: before=1 after=0 diff_removed={"source cannot run: no runnable evaluation path is registered for this model: the pinned evaluator is a callable with no CLI or checkpoint loader, the adapter deliberately declines to build a command, and the required correction dictionaries and resized images have no published source; the paper row is the 0%-noise row and no 0% checkpoint is published (see workbench/docs/reviews/models for the model's audit). An earlier note in this registry claimed a tensor-shape failure in the similarity computation; that claim was WRONG and is withdrawn - np.matmul broadcasts and the ranking matrix has the shape the indexing expects"} diff_added=set()
   intent: before=1 after=0 diff_removed={"source cannot run: no runnable evaluation path is registered for this model: the pinned evaluator is a callable with no CLI or checkpoint loader, the adapter deliberately declines to build a command, and the required correction dictionaries and resized images have no published source; the paper row is the 0%-noise row and no 0% checkpoint is published (see workbench/docs/reviews/models for the model's audit). An earlier note in this registry claimed a tensor-shape failure in the similarity computation; that claim was WRONG and is withdrawn - np.matmul broadcasts and the ranking matrix has the shape the indexing expects"} diff_added=set()
   ```
   Only the stale claim was removed. Real asset blockers on `airknow` (BLIP-2 pretrain weights, EVA-ViT-g backbone) remained.

---

## 3. Concrete Defects With Minimal Reproduction

### Defect 1: False package certification on Conda-looking or Conda environment
- **Location**: `workbench/backend/runtime.py:402-403`, `workbench/backend/runtime.py:635`
- **Root Cause**:
  `_inherits_base_site_packages` assumes:
  ```python
  if (prefix / "conda-meta").is_dir():
      return True
  ```
  Conda environments are completely isolated and do not inherit packages from `base_prefix`. Furthermore, any prefix containing a `conda-meta/` directory causes `inspect_environment_distributions` to append `base_prefix/site-packages` to inventory, certifying packages the interpreter cannot import.
- **Minimal Reproduction**:
  ```bash
  .venv/bin/python -c '
  import sys, shutil, subprocess
  from pathlib import Path
  from workbench.backend import runtime

  work = Path("/home/heheboiz/.cache/cir-pytest/defect1")
  shutil.rmtree(work, ignore_errors=True); work.mkdir(parents=True)
  venv = work / "env"
  subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)
  (venv / "conda-meta").mkdir()
  (venv / "bin" / "python").unlink()
  shutil.copy2(Path(sys.base_prefix) / "bin" / "python3", venv / "bin" / "python")
  (venv / "pyvenv.cfg").unlink()

  stdlib_src = Path(sys.base_prefix) / "lib" / "python3.13"
  stdlib_dst = venv / "lib" / "python3.13"
  for item in stdlib_src.iterdir():
      if item.name != "site-packages" and not (stdlib_dst / item.name).exists():
          (stdlib_dst / item.name).symlink_to(item)

  py = venv / "bin" / "python"
  probe = subprocess.run([str(py), "-I", "-c", "import importlib.util; print(\"pip:\", importlib.util.find_spec(\"pip\"))"], capture_output=True, text=True)
  print("Interpreter probe:", probe.stdout.strip())

  inherits = runtime._inherits_base_site_packages(venv, Path(sys.base_prefix))
  print("_inherits_base_site_packages:", inherits)

  dist, _, _ = runtime.inspect_environment_distributions(str(venv), sys.base_prefix)
  print("Certified packages:", list(dist.keys()))
  '
  ```
- **Raw Output**:
  ```
  Interpreter probe: pip: None
  _inherits_base_site_packages: True
  Certified packages: ['shadowpkg', 'foo', 'pip']
  ```
  Interpreter cannot import `pip`, but verifier certifies `pip`, `foo`, `shadowpkg`.

---

### Defect 2: `inspect_environment_distributions(prefix=None, base_prefix=...)` crashes with TypeError
- **Location**: `workbench/backend/runtime.py:634-635`
- **Root Cause**:
  `inspect_environment_distributions` accepts `prefix: str | None`. On line 634:
  `if base_prefix and str(base_prefix) != str(prefix):`
  When `prefix=None`, `str(None) != str(base_prefix)` is `True`. Line 635 executes `_inherits_base_site_packages(Path(prefix), Path(base_prefix))`. `Path(None)` raises `TypeError`.
- **Minimal Reproduction**:
  ```bash
  .venv/bin/python -c '
  from workbench.backend.runtime import inspect_environment_distributions
  try:
      inspect_environment_distributions(None, "/some/base/prefix")
  except Exception as e:
      print(f"Crashed with {type(e).__name__}: {e}")
  '
  ```
- **Raw Output**:
  ```
  Crashed with TypeError: argument should be a str or an os.PathLike object where __fspath__ returns a str, not 'NoneType'
  ```

---

### Defect 3: DCNet `binding_mismatch` returns silent pass (`None`) when `config.json` is a directory
- **Location**: `workbench/replay/dcnet.py:62-63, 107-110`
- **Root Cause**:
  In `configured_data_dir`:
  ```python
  if not config_file.is_file() or config_file.stat().st_size == 0:
      return None, blockers  # returns None, []
  ```
  In `binding_mismatch`:
  ```python
  resolved, blockers = configured_data_dir(...)
  if resolved is None:
      return blockers[0] if blockers else None  # returns None!
  ```
  When `config.json` is a directory, `configured_data_dir` returns `None, []`. `binding_mismatch` returns `None` (indicating no mismatch / success). Native `ConfigParser` crashes with `IsADirectoryError`.
- **Minimal Reproduction**:
  ```bash
  .venv/bin/python -c '
  import shutil, argparse
  from pathlib import Path
  from workbench.replay import dcnet
  work = Path("/home/heheboiz/.cache/cir-pytest/defect3")
  shutil.rmtree(work, ignore_errors=True); work.mkdir(parents=True)
  (work / "config.json").mkdir()
  ds = work / "expected_dataset"; ds.mkdir()

  mismatch = dcnet.binding_mismatch(work, ds, evaluator_cwd=work)
  print("binding_mismatch result:", mismatch)

  import sys
  sys.path.insert(0, "workbench/third_party/DCNet")
  from parse_config import ConfigParser
  parser = argparse.ArgumentParser()
  parser.add_argument("-r", "--resume", default=str(work))
  parser.add_argument("-d", "--device", default=None)
  parser.add_argument("-m", "--mode", default="test")
  try:
      ConfigParser(parser, mode="test")
  except Exception as e:
      print(f"ConfigParser error: {type(e).__name__}: {e}")
  '
  ```
- **Raw Output**:
  ```
  binding_mismatch result: None
  ConfigParser error: IsADirectoryError: [Errno 21] Is a directory: '/home/heheboiz/.cache/cir-pytest/defect3/config.json'
  ```

---

### Defect 4: DCNet `configured_data_dir` crashes with TypeError when `data_dir` is not a string
- **Location**: `workbench/replay/dcnet.py:77-78`
- **Root Cause**:
  `declared = (config.get("data_loader") or {}).get("args", {}).get("data_dir")`
  If `declared` is a list or number (e.g. `12345` or `["dataset/fashioniq"]`), `not declared` evaluates to `False`. Line 77 executes `Path(declared)`. `Path(12345)` raises `TypeError` unhandled.
- **Minimal Reproduction**:
  ```bash
  .venv/bin/python -c '
  import json, shutil
  from pathlib import Path
  from workbench.replay import dcnet

  work = Path("/home/heheboiz/.cache/cir-pytest/defect4")
  shutil.rmtree(work, ignore_errors=True); work.mkdir(parents=True)
  cfg = work / "config.json"
  cfg.write_text(json.dumps({"data_loader": {"args": {"data_dir": 12345}}}))
  try:
      dcnet.binding_mismatch(work, work / "ds", evaluator_cwd=work)
  except Exception as e:
      print(f"Crashed with {type(e).__name__}: {e}")
  '
  ```
- **Raw Output**:
  ```
  Crashed with TypeError: argument should be a str or an os.PathLike object where __fspath__ returns a str, not 'int'
  ```

---

### Defect 5: `lavis_processors._patch_registry_idempotence` bypasses `isinstance(path, str)` type check
- **Location**: `workbench/replay/lavis_processors.py:109-114`
- **Root Cause**:
  ```python
  def register_path(cls, name, path):
      existing = cls.mapping["paths"].get(name)
      if existing is not None and str(existing) == str(path):
          return
      return original(name, path)
  ```
  The original pinned method has:
  ```python
  assert isinstance(path, str)
  if name in cls.mapping["paths"]:
      raise KeyError("Name '{}' already registered.".format(name))
  ```
  If `name` was previously registered with string `"12345"`, and subsequently called with `register_path(name, 12345)` (integer), `str(existing) == str(path)` is `True`. It returns without executing `assert isinstance(path, str)`.
- **Minimal Reproduction**:
  ```bash
  .venv/bin/python -c '
  from pathlib import Path
  from workbench.tests.test_r10_remediations import _checkpoint_fixture
  from workbench.replay import lavis_processors
  import sys

  work = Path("/home/heheboiz/.cache/cir-pytest/defect5")
  fixture = _checkpoint_fixture(work)
  lavis_processors.reset()
  lavis_processors.prepare(fixture)
  reg = sys.modules["lavis.common.registry"].registry

  reg.register_path("pin_test", "9999")
  # Pinned method would raise AssertionError for non-string
  reg.register_path("pin_test", 9999)
  print("Non-string integer accepted without AssertionError")
  '
  ```
- **Raw Output**:
  ```
  Non-string integer accepted without AssertionError
  ```

---

### Defect 6: `lavis_processors.reset()` does not clean `sys.path`, leaking previous checkout imports
- **Location**: `workbench/replay/lavis_processors.py:150-153`
- **Root Cause**:
  `prepare(source_root)` inserts `str(source_root)` and `str(lavis_root)` into `sys.path`.
  `reset()` only deletes `sys.modules` keys matching `"lavis"` or `"lavis.*"`. It leaves `sys.path` polluted. Any module from the previous checkout is immediately re-importable, and falls back into the namespace if a subsequent checkout lacks a submodule.
- **Minimal Reproduction**:
  ```bash
  .venv/bin/python -c '
  import sys, shutil
  from pathlib import Path
  from workbench.tests.test_r10_remediations import _checkpoint_fixture
  from workbench.replay import lavis_processors

  work = Path("/home/heheboiz/.cache/cir-pytest/defect6")
  shutil.rmtree(work, ignore_errors=True); work.mkdir(parents=True)
  ck_a = _checkpoint_fixture(work / "a")
  (ck_a / "lavis" / "secret.py").write_text("TOKEN = 42\n")

  lavis_processors.reset()
  lavis_processors.prepare(ck_a)
  import lavis.secret
  print("Imported before reset:", lavis.secret.TOKEN)

  lavis_processors.reset()
  print("sys.modules has lavis?", "lavis" in sys.modules)
  print("sys.path contains ck_a?", str(ck_a) in sys.path)

  import lavis.secret
  print("Imported AFTER reset from sys.path:", lavis.secret.TOKEN)
  '
  ```
- **Raw Output**:
  ```
  Imported before reset: 42
  sys.modules has lavis? False
  sys.path contains ck_a? True
  Imported AFTER reset from sys.path: 42
  ```

---

### Defect 7: `_evaluator_imports` traces 20–26 undeclared packages per model; tests contain dummy assertion
- **Location**: `workbench/backend/environment_planner.py:239-348`, `workbench/tests/test_r10_remediations.py:405-406`
- **Root Cause**:
  `_evaluator_imports` runs un-scoped AST traversal over all files imported by the wrapper, reaching unused/optional third-party branches in LAVIS (`apex`, `tensorflow`, `horovod`, `knn-cuda`, `diffusers`, `easydict`). `model_constraints` marks every unpinned traced import as an `"UNKNOWN"` requirement.
  In `test_r10_remediations.py:405-406`:
  ```python
  for module in sorted(traced):
      assert module or True  # the loop documents the comparison; the assertion is below
  ```
  The test contains a no-op assertion `assert module or True` and fails to check that traced imports are declared or satisfiable.
- **Minimal Reproduction**:
  ```bash
  .venv/bin/python -c '
  from workbench.backend.registry import load_registry
  from workbench.backend.environment_planner import model_constraints

  reg = load_registry()
  hint = next(m for m in reg["models"] if m["model_id"] == "hint")
  constraints = model_constraints({**hint, "source_available": True})
  pkgs = constraints["packages"]
  unknowns = [k for k, v in pkgs.items() if v == "UNKNOWN"]
  print(f"Total packages demanded: {len(pkgs)}")
  print(f"UNKNOWN packages ({len(unknowns)}): {sorted(unknowns)}")
  '
  ```
- **Raw Output**:
  ```
  Total packages demanded: 37
  UNKNOWN packages (25): ['apex', 'diffusers', 'easydict', 'ftfy', 'h5py', 'horovod', 'ipython', 'knn-cuda', 'matplotlib', 'moviepy', 'mpl-toolkits', 'nltk', 'numpy', 'packaging', 'peft', 'pillow', 'pyyaml', 'regex', 'requests', 'spacy', 'tensorflow', 'termcolor', 'torchaudio', 'tqdm', 'typing-extensions']
  ```

---

## 4. Unsupported Claims in `R10_REMEDIATION.md`

1. **R10-06 (Line 20)**:
   - *Claim*: `runtime.py:_inherits_base_site_packages` checks `conda-meta`, and Conda environments inherit packages from base install.
   - *Reality*: Conda environments do not inherit packages from base install. A directory containing `conda-meta/` causes `_inherits_base_site_packages` to return `True`, certifying packages an isolated interpreter cannot import.

2. **R10-03 (Line 25)**:
   - *Claim*: `reset()` tears down package context completely so another checkout can be prepared without inheriting previous checkout's classes.
   - *Reality*: `reset()` does not remove `source_root` or `lavis_root` from `sys.path`. Modules from the previous checkout remain importable.

3. **R10-05 (Line 29, 37)**:
   - *Claim*: Scanner follows launcher so profiles and reachable dependencies are coherent without contradiction.
   - *Reality*: AST scan pulls 25+ unneeded dependencies (`apex`, `tensorflow`, `horovod`, `knn-cuda`) marked `UNKNOWN` in `model_constraints`. Declared pins (`opencv-python-headless`, `pillow`, `pyyaml`, `pycocoevalcap`, `scikit-image`) are absent from the traced AST.

4. **R10-01 (Line 23)**:
   - *Claim*: DCNet validator agrees with native ConfigParser.
   - *Reality*: Disagrees on non-string `data_dir` (crashes with unhandled `TypeError`) and on directory `config.json` (returns `None` / silent pass while `ConfigParser` raises `IsADirectoryError`).

---

## 5. Tests Passing With Fixes Reverted (Proved by Git Revert in Clean Copy)

Reverting changes from `cf84cfa` back to `e46ad65` in a separate copy of the repository revealed that **8 of the 19 tests in `test_r10_remediations.py` still pass**:

1. `test_an_isolated_venv_cannot_borrow_the_base_interpreter_packages`:
   **PASSED on e46ad65**.
   *Proof*: The test checks `names=['packaging','torch','pytest','numpy']`. Host base Python (`/home/heheboiz/.local/share/uv/python/cpython-3.13.12-linux-x86_64-gnu`) only contains `pip`, `foo`, `shadowpkg`. None of the four tested packages exist in base Python, so the test passes vacuously even when base packages are read into inventory.
2. `test_a_wrong_declared_version_still_blocks_in_an_isolated_venv`:
   **PASSED on e46ad65**.
   *Proof*: Asserts `pytest` not in versions. `pytest` is absent from base Python, so it passed prior to the fix.
3. `test_the_wrapper_remains_the_registered_entrypoint`:
   **PASSED on e46ad65**.
   *Proof*: Replay wrappers and adapter `command()` implementations for `airknow`, `conesep`, `intent` were already registered and functioning in `e46ad65`.
4. `test_the_validator_and_the_native_evaluator_agree_on_the_root`:
   **PASSED on e46ad65**.
   *Proof*: Test does not set `WORKBENCH_DCNET_DATA_DIR`. In `e46ad65`, without the environment override, `configured_data_dir` already resolved `config.json` `data_dir` against checkout.
5. `test_the_stager_refuses_an_unexpected_occupant`:
   **PASSED on e46ad65**.
   *Proof*: `prepare_dataset.prepare_canonical_link` already checked `os.path.lexists(destination)` and refused existing real directories prior to R10.
6. `test_a_link_pointing_elsewhere_is_refused`:
   **PASSED on e46ad65**.
   *Proof*: `destination.is_symlink()` and `linked_root != root` refusal existed in `e46ad65`.
7. `test_the_declared_profile_is_coherent_with_the_traced_imports`:
   **PASSED on e46ad65**.
   *Proof*: Line 406 contains `assert module or True` (no-op assertion). The test only asserts that `transformers==` and `timm==` are declared in `packages`, which was already true in `e46ad65`.
8. `test_the_vendored_imports_are_served_by_the_declared_pins`:
   **PASSED on e46ad65**.
   *Proof*: `timm==0.9.16` was already declared in `models.yaml` in `e46ad65`.
