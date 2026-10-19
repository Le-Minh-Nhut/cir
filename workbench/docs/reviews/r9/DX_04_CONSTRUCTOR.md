# Single-Category Validation vs Constructor Inputs for LAVIS-Family Wrappers

## 1. Native Constructor Category Loop and File Opening Sequence

### Native Constructor Category Loop
In `workbench/third_party/HINT/datasets.py:59-73` (class `FashionIQ`), `train_init_process` loops over all three categories:

```python
    def train_init_process(self):
        for name in ['dress', 'shirt', 'toptee']:
            with open(os.path.join(self.caption_dir, "cap.{}.{}.json".format(name, 'train')), 'r') as f:
                ref_captions = json.load(f)
            with open(os.path.join(self.caption_dir, 'correction_dict_{}.json'.format(name)), 'r') as f:
                correction_dict = json.load(f)
            for triplets in ref_captions:
                ref_id = triplets['candidate']
                tag_id = triplets['target']
                cap = self.concat_text(triplets['captions'], correction_dict)
                self.fashioniq_data.append({
                    'target': name + '_' + tag_id,
                    'candidate': name + '_' + ref_id,
                    'captions': cap
                })
```

Identical loops exist in all vendored siblings:
- `workbench/third_party/HABIT/datasets.py:96-110` (`for name in ['dress', 'shirt', 'toptee']:`)
- `workbench/third_party/INTENT/datasets.py:94-108` (`for name in ['dress', 'shirt', 'toptee']:`)
- `workbench/third_party/ConeSep/datasets.py:92-106` (`for name in ['dress', 'shirt', 'toptee']:`)
- `workbench/third_party/AirKnow/datasets.py:106-120` (`for name in ['dress', 'shirt', 'toptee']:`)
- `workbench/third_party/PAIR/data/datasets.py:61-75` (`for name in ['dress', 'shirt', 'toptee']:`)

### File Opening Sequence for a Run Evaluating ONLY `dress`
Because each wrapper evaluates within `isolated_cache_root(dataset_root)` (`workbench/replay/common.py:90-98`), the constructor executes against a fresh temporary directory where `fashion_iq_data.json` does not exist (`HINT/datasets.py:33`).

For a run that evaluates only `dress`, the constructor unconditionally opens the following files in this exact chronological order:

1. **`captions/cap.dress.train.json`** [read] — `HINT/datasets.py:61` (inside `train_init_process()`, iteration `name='dress'`)
2. **`captions/correction_dict_dress.json`** [read] — `HINT/datasets.py:63`
3. **`captions/cap.shirt.train.json`** [read] — `HINT/datasets.py:61` (inside `train_init_process()`, iteration `name='shirt'`)
4. **`captions/correction_dict_shirt.json`** [read] — `HINT/datasets.py:63`
5. **`captions/cap.toptee.train.json`** [read] — `HINT/datasets.py:61` (inside `train_init_process()`, iteration `name='toptee'`)
6. **`captions/correction_dict_toptee.json`** [read] — `HINT/datasets.py:63`
7. **`fashion_iq_data.json`** [write] — `HINT/datasets.py:36`
8. **`image_splits/split.dress.val.json`** [read] — `HINT/datasets.py:133` (inside `get_test_data('dress')`, called from line 39)
9. **`captions/cap.dress.val.json`** [read] — `HINT/datasets.py:135`
10. **`captions/correction_dict_dress.json`** [read] — `HINT/datasets.py:137`
11. **`resized_image/dress/<id>.jpg`** [read] — `HINT/datasets.py:118` via `get_test_img` (lines 143, 145, 157) for all query and target images
12. **`image_splits/split.shirt.val.json`** [read] — `HINT/datasets.py:133` (inside `get_test_data('shirt')`, called from line 40)
13. **`captions/cap.shirt.val.json`** [read] — `HINT/datasets.py:135`
14. **`captions/correction_dict_shirt.json`** [read] — `HINT/datasets.py:137`
15. **`resized_image/shirt/<id>.jpg`** [read] — `HINT/datasets.py:118` via `get_test_img` (lines 143, 145, 157) for all query and target images
16. **`image_splits/split.toptee.val.json`** [read] — `HINT/datasets.py:133` (inside `get_test_data('toptee')`, called from line 41)
17. **`captions/cap.toptee.val.json`** [read] — `HINT/datasets.py:135`
18. **`captions/correction_dict_toptee.json`** [read] — `HINT/datasets.py:137`
19. **`resized_image/toptee/<id>.jpg`** [read] — `HINT/datasets.py:118` via `get_test_img` (lines 143, 145, 157) for all query and target images
20. **`test_queries_dress.pkl`** [write] — `HINT/datasets.py:42`
21. **`test_targets_dress.pkl`** [write] — `HINT/datasets.py:43`
22. **`test_queries_shirt.pkl`** [write] — `HINT/datasets.py:44`
23. **`test_targets_shirt.pkl`** [write] — `HINT/datasets.py:45`
24. **`test_queries_toptee.pkl`** [write] — `HINT/datasets.py:46`
25. **`test_targets_toptee.pkl`** [write] — `HINT/datasets.py:47`

If `fashion_iq_data.json` had existed beforehand in a non-isolated environment (`HINT/datasets.py:48-56`):
1. `fashion_iq_data.json` [read] — `HINT/datasets.py:49`
2. `test_queries_dress.pkl` [read] — `HINT/datasets.py:51`
3. `test_targets_dress.pkl` [read] — `HINT/datasets.py:52`
4. `test_queries_shirt.pkl` [read] — `HINT/datasets.py:53`
5. `test_targets_shirt.pkl` [read] — `HINT/datasets.py:54`
6. `test_queries_toptee.pkl` [read] — `HINT/datasets.py:55`
7. `test_targets_toptee.pkl` [read] — `HINT/datasets.py:56`

In both cold and warm cache paths, shirt and toptee files are unconditionally opened even if evaluation is requested only for `dress`.

---

## 2. Wrapper `validate_dataset` Implementations & Single-Category Behavior

### Code Quotes of `validate_dataset`

#### HINT (`workbench/replay/hint.py:83-112`)
```python
def validate_dataset(dataset_root: Path, categories: tuple[str, ...] = CATEGORIES) -> list[str]:
    """The inputs `datasets.FashionIQ` opens before `test()` can run.

    `datasets.py:29-31` concatenates the root with `resized_image`, `image_splits` and `captions`, so
    the root must carry a trailing separator; `:137` additionally requires the correction dictionary
    the caption pre-processing reads.
    """
    blockers: list[str] = []
    for category in categories:
        for relative in (
            f"captions/cap.{category}.val.json",
            f"captions/correction_dict_{category}.json",
            f"image_splits/split.{category}.val.json",
        ):
            if not (dataset_root / relative).is_file():
                blockers.append(f"HINT dataset input missing: {dataset_root / relative}")
        if not (dataset_root / "resized_image" / category).is_dir():
            blockers.append(f"HINT dataset input missing: {dataset_root / 'resized_image' / category}")
        # `datasets.py` builds a FRESH cache with `train_init_process`, which opens
        # `cap.<category>.train.json` and the correction dictionary for all three categories BEFORE
        # reading any validation file (`train_init_process` at :59-73, called from `__init__`).
        # `isolated_cache_root` gives that constructor an empty temp root every run, so a cache in the
        # canonical root does NOT satisfy it: without the train captions the constructor raises
        # FileNotFoundError at dataset construction, long after an expensive model load.
        for relative in (
            f"captions/cap.{category}.train.json",
            f"captions/correction_dict_{category}.json",
        ):
            if not (dataset_root / relative).is_file():
                blockers.append(f"{'HINT'} dataset input missing: {dataset_root / relative}")
    return blockers
```

#### HABIT (`workbench/replay/habit.py:69-81`)
```python
def validate_dataset(dataset_root: Path, categories: tuple[str, ...] = CATEGORIES) -> list[str]:
    blockers: list[str] = []
    for category in categories:
        for relative in (
            f"captions/cap.{category}.train.json",
            f"captions/cap.{category}.val.json",
            f"captions/correction_dict_{category}.json",
            f"image_splits/split.{category}.val.json",
        ):
            if not (dataset_root / relative).is_file():
                blockers.append(f"HABIT dataset requirement missing: {dataset_root / relative}")
        if not (dataset_root / "resized_image" / category).is_dir():
            blockers.append(f"HABIT dataset requirement missing: {dataset_root / 'resized_image' / category}")
    return blockers
```

#### INTENT (`workbench/replay/intent.py:69-81`)
```python
def validate_dataset(dataset_root: Path, categories: tuple[str, ...] = CATEGORIES) -> list[str]:
    blockers: list[str] = []
    for category in categories:
        for relative in (
            f"captions/cap.{category}.train.json",
            f"captions/cap.{category}.val.json",
            f"captions/correction_dict_{category}.json",
            f"image_splits/split.{category}.val.json",
        ):
            if not (dataset_root / relative).is_file():
                blockers.append(f"INTENT dataset requirement missing: {dataset_root / relative}")
        if not (dataset_root / "resized_image" / category).is_dir():
            blockers.append(f"INTENT dataset requirement missing: {dataset_root / 'resized_image' / category}")
    return blockers
```

#### ConeSep (`workbench/replay/conesep.py:81-93`)
```python
def validate_dataset(dataset_root: Path, categories: tuple[str, ...] = CATEGORIES) -> list[str]:
    blockers: list[str] = []
    for category in categories:
        for relative in (
            f"captions/cap.{category}.train.json",
            f"captions/cap.{category}.val.json",
            f"captions/correction_dict_{category}.json",
            f"image_splits/split.{category}.val.json",
        ):
            if not (dataset_root / relative).is_file():
                blockers.append(f"ConeSep dataset requirement missing: {dataset_root / relative}")
        if not (dataset_root / "resized_image" / category).is_dir():
            blockers.append(f"ConeSep dataset requirement missing: {dataset_root / 'resized_image' / category}")
    return blockers
```

#### Air-Know (`workbench/replay/airknow.py:84-96`)
```python
def validate_dataset(dataset_root: Path, categories: tuple[str, ...] = CATEGORIES) -> list[str]:
    blockers: list[str] = []
    for category in categories:
        for relative in (
            f"captions/cap.{category}.train.json",
            f"captions/cap.{category}.val.json",
            f"captions/correction_dict_{category}.json",
            f"image_splits/split.{category}.val.json",
        ):
            if not (dataset_root / relative).is_file():
                blockers.append(f"Air-Know dataset requirement missing: {dataset_root / relative}")
        if not (dataset_root / "resized_image" / category).is_dir():
            blockers.append(f"Air-Know dataset requirement missing: {dataset_root / 'resized_image' / category}")
    return blockers
```

#### PAIR (`workbench/replay/pair.py:83-106`)
```python
def validate_dataset(dataset_root: Path, categories: tuple[str, ...] = CATEGORIES) -> list[str]:
    blockers: list[str] = []
    for category in categories:
        for relative in (
            f"captions/cap.{category}.val.json",
            f"captions/correction_dict_{category}.json",
            f"image_splits/split.{category}.val.json",
        ):
            if not (dataset_root / relative).is_file():
                blockers.append(f"PAIR dataset input missing: {dataset_root / relative}")
        cat_resized = dataset_root / "resized_image" / category
        if not cat_resized.is_dir():
            blockers.append(f"PAIR dataset input missing: {cat_resized}")
        # `data/datasets.py:34-36` builds a FRESH cache with `train_init_process`, which opens the
        # TRAIN captions for all three categories before any validation file. The standalone preflight
        # therefore has to require them, or the constructor raises FileNotFoundError after the
        # checkpoint has been loaded. (The orchestrator's preparation contract already declares the
        # training layout, so this aligns the standalone path with it rather than adding a new rule.)
        for relative in (
            f"captions/cap.{category}.train.json",
            f"captions/correction_dict_{category}.json",
        ):
            if not (dataset_root / relative).is_file():
                blockers.append(f"PAIR dataset input missing: {dataset_root / relative}")
    return blockers
```

### Validation Behavior When Called as `validate_dataset(root, categories=('dress',))`
Every wrapper loops `for category in categories:`. When called with `categories=('dress',)`:
- Each wrapper validates **only** the `dress` category files:
  - `captions/cap.dress.train.json`
  - `captions/cap.dress.val.json`
  - `captions/correction_dict_dress.json`
  - `image_splits/split.dress.val.json`
  - `resized_image/dress/` directory
- None of the six wrappers checks for `shirt` or `toptee` files when `categories=('dress',)` is passed.
- If only `dress` files are on disk, `validate_dataset(root, categories=('dress',))` returns `[]` (passes).
- However, the native dataset constructor `datasets.FashionIQ(...)` does **not** accept or heed the subset, and immediately attempts to open `shirt` and `toptee` files, throwing `FileNotFoundError`.

### Per-Wrapper Validation & Constructor Gating Table

| Wrapper | `validate_dataset` file:line | Gates on `categories` argument only? | Category set validated when `categories=('dress',)` | Native constructor heeds `categories`? | Does native constructor still open shirt/toptee files? |
|---|---|---|---|---|---|
| `hint` | `workbench/replay/hint.py:83-112` | **Yes** (`for category in categories:`) | `{'dress'}` only | **No** (signature has no `category` parameter; `path, transform`) | **Yes** (`train_init_process` + `get_test_data` for all 3) |
| `habit` | `workbench/replay/habit.py:69-81` | **Yes** (`for category in categories:`) | `{'dress'}` only | **No** (`category` parameter ignored during `__init__` cache init) | **Yes** (`train_init_process` + `get_test_data` for all 3) |
| `intent` | `workbench/replay/intent.py:69-81` | **Yes** (`for category in categories:`) | `{'dress'}` only | **No** (`category` parameter ignored during `__init__` cache init) | **Yes** (`train_init_process` + `get_test_data` for all 3) |
| `conesep` | `workbench/replay/conesep.py:81-93` | **Yes** (`for category in categories:`) | `{'dress'}` only | **No** (signature has no `category` parameter; `path, transform, noise_ratio`) | **Yes** (`train_init_process` + `get_test_data` for all 3) |
| `airknow` | `workbench/replay/airknow.py:84-96` | **Yes** (`for category in categories:`) | `{'dress'}` only | **No** (signature has no `category` parameter; `path, transform, noise_ratio`) | **Yes** (`train_init_process` + `get_test_data` for all 3) |
| `pair` | `workbench/replay/pair.py:83-106` | **Yes** (`for category in categories:`) | `{'dress'}` only | **No** (signature has no `category` parameter; `path, transform, split`) | **Yes** (`train_init_process` + `get_test_data` for all 3) |

---

## 3. Production Callers of `validate_dataset` and Parameter Passing

### Exhaustive Caller Analysis
Across the repository, every caller of `validate_dataset` in production code is enumerated below:

1. **`workbench/replay/hint.py:119`**:
   ```python
   def preflight(checkpoint_root: Path, dataset_root: Path, source_root: Path,
                 categories: tuple[str, ...] = CATEGORIES) -> list[str]:
       return (validate_source(source_root) + validate_checkpoint(checkpoint_root)
               + validate_dataset(dataset_root, categories))
   ```
   - Value passed: `categories` from `preflight` parameter.
   - Caller: `hint.py:main:202-203`:
     ```python
     categories = (args.category,) if args.category else CATEGORIES
     problems = preflight(args.checkpoint_root, args.dataset_root, args.source_root, categories)
     ```
     When invoked from CLI with `--category dress`, passes `('dress',)` (a subset). When invoked without `--category`, passes default `CATEGORIES`.

2. **`workbench/replay/habit.py:95`**:
   ```python
   def preflight(
       checkpoint_root: Path,
       dataset_root: Path,
       source_root: Path,
       categories: tuple[str, ...] = CATEGORIES,
       variant_id: str = DEFAULT_VARIANT,
   ) -> list[str]:
       return (
           validate_source(source_root)
           + validate_checkpoint(checkpoint_root, variant_id)
           + validate_dataset(dataset_root, categories)
       )
   ```
   - Value passed: `categories` from `preflight` parameter.
   - Caller: `habit.py:main:185-186`:
     ```python
     categories = (args.category,) if args.category else CATEGORIES
     problems = preflight(args.checkpoint_root, args.dataset_root, args.source_root, categories, args.variant)
     ```
     When invoked from CLI with `--category dress`, passes `('dress',)`.

3. **`workbench/replay/intent.py:95`**:
   ```python
   def preflight(
       checkpoint_root: Path,
       dataset_root: Path,
       source_root: Path,
       categories: tuple[str, ...] = CATEGORIES,
       variant_id: str = DEFAULT_VARIANT,
   ) -> list[str]:
       return (
           validate_source(source_root)
           + validate_checkpoint(checkpoint_root, variant_id)
           + validate_dataset(dataset_root, categories)
       )
   ```
   - Value passed: `categories` from `preflight` parameter.
   - Caller: `intent.py:main:185-186`:
     ```python
     categories = (args.category,) if args.category else CATEGORIES
     problems = preflight(args.checkpoint_root, args.dataset_root, args.source_root, categories, args.variant)
     ```
     When invoked from CLI with `--category dress`, passes `('dress',)`.

4. **`workbench/replay/conesep.py:107`**:
   ```python
   def preflight(
       checkpoint_root: Path,
       dataset_root: Path,
       source_root: Path,
       categories: tuple[str, ...] = CATEGORIES,
       variant_id: str = DEFAULT_VARIANT,
   ) -> list[str]:
       return (
           validate_source(source_root)
           + validate_checkpoint(checkpoint_root, variant_id)
           + validate_dataset(dataset_root, categories)
       )
   ```
   - Value passed: `categories` from `preflight` parameter.
   - Caller: `conesep.py:main:199-200`:
     ```python
     categories = (args.category,) if args.category else CATEGORIES
     problems = preflight(args.checkpoint_root, args.dataset_root, args.source_root, categories, args.variant)
     ```
     When invoked from CLI with `--category dress`, passes `('dress',)`.

5. **`workbench/replay/airknow.py:110`**:
   ```python
   def preflight(
       checkpoint_root: Path,
       dataset_root: Path,
       source_root: Path,
       categories: tuple[str, ...] = CATEGORIES,
       variant_id: str = DEFAULT_VARIANT,
   ) -> list[str]:
       return (
           validate_source(source_root)
           + validate_checkpoint(checkpoint_root, variant_id)
           + validate_dataset(dataset_root, categories)
       )
   ```
   - Value passed: `categories` from `preflight` parameter.
   - Caller: `airknow.py:main:214-215`:
     ```python
     categories = (args.category,) if args.category else CATEGORIES
     problems = preflight(args.checkpoint_root, args.dataset_root, args.source_root, categories, args.variant)
     ```
     When invoked from CLI with `--category dress`, passes `('dress',)`.

6. **`workbench/replay/pair.py:120`**:
   ```python
   def preflight(
       checkpoint_root: Path,
       dataset_root: Path,
       source_root: Path,
       categories: tuple[str, ...] = CATEGORIES,
       checkpoint_id: str | None = None,
   ) -> list[str]:
       return (
           validate_source(source_root)
           + validate_checkpoint(checkpoint_root, checkpoint_id)
           + validate_dataset(dataset_root, categories)
       )
   ```
   - Value passed: `categories` from `preflight` parameter.
   - Caller: `pair.py:main:202-203`:
     ```python
     selected_categories = tuple(args.categories) if args.categories else CATEGORIES
     blockers = preflight(
         args.checkpoint_root,
         args.dataset_root,
         args.source_root,
         selected_categories,
         checkpoint_id=args.checkpoint_id,
     )
     ```
     When invoked from CLI with `--categories dress`, passes `('dress',)`.

7. **`workbench/replay/limn.py:133` and `150`**:
   - `limn.py:133`: inside `preflight(..., categories=CATEGORIES)`:
     ```python
     + validate_dataset(dataset_root, categories) + validate_environment()
     ```
   - `limn.py:150`: inside `load_native(category: str, ...)`:
     ```python
     blockers = (validate_checkpoint_bundle(checkpoint_root, (category,))
                 + validate_dataset(dataset_root, (category,)) + validate_environment()
                 + validate_source(source_root))
     ```
     Passes explicit 1-element tuple `(category,)`. (LIMN is an independent-per-category model family, not a shared-checkpoint LAVIS-family wrapper).

### Are Subsets Passed by Orchestrated Production Pipelines?
**No.**
In `workbench/backend/adapters/models.py`:
- `HintAdapter.command` (`lines 84-100`) constructs:
  ```python
  [
      _runtime.python_executable(self.model),
      str(Path(__file__).resolve().parents[3] / "workbench" / "replay" / "hint.py"),
      "--checkpoint-root", str(checkpoint_member_root(checkpoint)),
      "--dataset-root", str(request.dataset_root),
      "--source-root", str(source),
  ]
  ```
- `PairAdapter`, `AirKnowAdapter`, `ConeSepAdapter`, `HabitAdapter`, `IntentAdapter` all invoke `_replay_command(self, checkpoint, request, source, ...)` (`lines 30-52`).
- Neither `_replay_command` nor `HintAdapter` ever adds `--category`.
- The only adapter in `workbench/backend/adapters/models.py` that ever emits `--category` is `LIMNAdapter` (`line 292`), because LIMN checkpoints are per-category.

Therefore:
- In production pipelines (autopilot, orchestrator, evaluation runner), `validate_dataset` is **never** passed a subset for the LAVIS family wrappers; it always receives the full `CATEGORIES` default.
- In manual CLI diagnostic invocations (`python -m workbench.replay.<model> --category <cat>`), the CLI parser accepts a subset and forwards it to `validate_dataset(..., (args.category,))`.

---

## 4. Shared Preflight (`model_requirements`) and Reachability Analysis

### Branch in `workbench/backend/preflight.py:model_requirements`
In `workbench/backend/preflight.py:20-27`:
```python
    contract = preparation_contract_for_model(model)
    raw = preparation_paths(contract, root, "raw_inputs") if contract else fashioniq_required_paths(model, root)
    generated = preparation_paths(contract, root, "generated_artifacts") if contract else ()
    paths = list(dict.fromkeys((*raw, *generated)))
    raw_missing, generated_missing = missing_paths(raw), missing_paths(generated)
```
And lines 32-40:
```python
    if contract:
        blocks.extend(f"raw benchmark input missing: {path}" for path in raw_missing)
        blocks.extend(f"required generated artifact missing: {path}" for path in generated_missing)
    else:
        blocks.extend(f"FashionIQ {layout} requirement missing: {path}" for path in raw_missing)
    image_problems = [problem for extension in model.get("fashioniq_image_extensions") or ()
                      for problem in image_extension_problems(root, extension, model.get("fashioniq_layout"))]
    blocks.extend(f"FashionIQ image extension requirement: {problem}" for problem in image_problems)
```

### Contract Resolution vs Layout Resolution for These Models
In `workbench/backend/registry.py:191-196`:
```python
def preparation_contract_for_model(model: dict[str, Any]) -> dict[str, Any] | None:
    preparation_id = model.get("preparation_id")
    if preparation_id is None:
        return None
    try:
        return load_preparation_contracts()[preparation_id]
```

Looking at each model in `workbench/registry/models.yaml`:
1. **`pair`**: Declares `preparation_id: pair_fashioniq` (`models.yaml:510`).
   - `preparation_contract_for_model(model)` returns `pair_fashioniq` from `workbench/registry/preparation_contracts.yaml:302-339`.
   - `pair_fashioniq` explicitly requires raw inputs for all three categories:
     `captions/cap.{dress,shirt,toptee}.train.json`,
     `captions/cap.{dress,shirt,toptee}.val.json`,
     `image_splits/split.{dress,shirt,toptee}.val.json`, and
     `resized_image/{dress,shirt,toptee}`.
2. **`hint`**: Has no `preparation_id` (`preparation_contract_for_model` returns `None`).
   - Evaluates `fashioniq_required_paths(model, root)` (`registry.py:949`), which reads `model["fashioniq_layout"] = "fashioniq_ilearn_resized_training"` (`models.yaml:402`).
   - Calls `ilearn_resized_training_paths(root)` in `fashioniq_layout.py:36-40`, requiring validation splits, val captions, correction dictionaries, train captions, and resized image folders for all three categories `CATEGORIES = ("dress", "shirt", "toptee")`.
3. **`habit`**, **`intent`**, **`conesep`**, **`airknow`**: Have no `preparation_id` (`preparation_contract_for_model` returns `None`).
   - Their `fashioniq_layout` in `models.yaml` is `fashioniq_ilearn_resized`.
   - Calls `ilearn_resized_paths(root)` in `fashioniq_layout.py:23-33`, requiring val captions, correction dicts, val splits, and resized image directories for all three categories `("dress", "shirt", "toptee")`.
   - In addition, line 38 checks `image_extension_problems`, which iterates `for category in CATEGORIES:` (`dress`, `shirt`, `toptee`).

Neither `model_requirements` nor `preflight_blockers` (`preflight.py:162`) takes a `categories` parameter. Both check all three categories unconditionally.

### Reachability Verdict
1. **Production / Orchestrated Pipeline**: **UNREACHABLE**.
   The shared backend preflight (`model_requirements` / `preflight_blockers`) runs as a prerequisite for automated pipeline execution. Because it validates all three categories without exception, and the adapters never pass `--category`, the wrappers are never executed in production on a partial-category root.
2. **Direct CLI Diagnostic Invocation**: **REACHABLE & BROKEN**.
   If an operator runs `python -m workbench.replay.<model> --category dress --preflight-only`, it invokes the wrapper's standalone `preflight(..., categories=('dress',))` which bypasses `model_requirements`. The standalone `validate_dataset` declares the dress-only root `[READY]`. When the operator then runs evaluation without `--preflight-only`, the wrapper loads the expensive neural network weights, invokes `isolated_cache_root`, calls `datasets.FashionIQ`, and crashes with `FileNotFoundError` on `cap.shirt.train.json`.

---

## 5. Dataset Construction: Wrapper vs Native Dataset Class

The wrappers do **not** construct custom torch datasets. Each wrapper instantiates the native vendored `datasets.FashionIQ` class directly:

- **`hint`** (`workbench/replay/hint.py:169`):
  ```python
  dataset = datasets.FashionIQ(path=root, transform=transform)
  ```
- **`habit`** (`workbench/replay/habit.py:134`):
  ```python
  dataset = datasets.FashionIQ(path=root_str, transform=transform, noise_ratio=0, split="original-split")
  ```
- **`intent`** (`workbench/replay/intent.py:134`):
  ```python
  dataset = datasets.FashionIQ(path=root_str, transform=transform, noise_ratio=0, split="original-split")
  ```
- **`conesep`** (`workbench/replay/conesep.py:147`):
  ```python
  dataset = datasets.FashionIQ(path=root_str, transform=transform, noise_ratio=0)
  ```
- **`airknow`** (`workbench/replay/airknow.py:146`):
  ```python
  dataset = datasets.FashionIQ(path=root_str, transform=transform, noise_ratio=0)
  ```
- **`pair`** (`workbench/replay/pair.py:150`):
  ```python
  dataset = datasets.FashionIQ(path=root_str, transform=transform, split="val-split")
  ```

### Timing of Dataset Construction Relative to Model Loading
In all six wrappers, dataset construction occurs **after** model deserialization and device transfer:
- `hint.py:152` loads model -> `154` loads processor -> `156` moves to GPU -> `169` constructs dataset.
- `habit.py:118` loads model -> `120` moves to GPU -> `122` loads processor -> `134` constructs dataset.
- `intent.py:118` loads model -> `120` moves to GPU -> `122` loads processor -> `134` constructs dataset.
- `conesep.py:131` loads model -> `133` moves to GPU -> `135` loads processor -> `147` constructs dataset.
- `airknow.py:125` loads model -> `127` moves to GPU -> `134` loads processor -> `146` constructs dataset.
- `pair.py:138` loads model -> `142` moves to GPU -> `144` builds open_clip transforms -> `150` constructs dataset.

Thus, when missing category files cause a crash at dataset initialization, the failure occurs *after* the computational and memory expense of model checkpoint loading.

---

## 6. Cache Isolation and Cold-Cache Constructor Behavior

### Usage of `isolated_cache_root`
All six wrappers wrap dataset instantiation inside `with isolated_cache_root(dataset_root) as cache_root:`:
- `hint.py:167`
- `habit.py:132`
- `intent.py:132`
- `conesep.py:144`
- `airknow.py:143`
- `pair.py:147`

### Implementation of `isolated_cache_root`
In `workbench/replay/common.py:90-98`:
```python
@contextmanager
def isolated_cache_root(dataset_root: Path, directory_names: tuple[str, ...] = ("captions", "image_splits", "resized_image")) -> Iterator[Path]:
    """Use native caches in a fresh temp root, symlinking immutable inputs only."""
    with tempfile.TemporaryDirectory(prefix="cir-replay-cache-") as temporary:
        cache_root = Path(temporary)
        for name in directory_names:
            source = dataset_root / name
            if not source.exists():
                raise FileNotFoundError(f"native replay input missing: {source}")
            (cache_root / name).symlink_to(source.resolve(), target_is_directory=source.is_dir())
        yield cache_root
```

### Impact of Fresh Cache Root on Constructor File Access
1. `isolated_cache_root` creates a new `tempfile.TemporaryDirectory` and symlinks only `captions`, `image_splits`, and `resized_image`.
2. Any pre-existing cache file in `dataset_root` (such as `dataset_root / "fashion_iq_data.json"` or `dataset_root / "test_queries_dress.pkl"`) is **not** symlinked and is invisible inside `cache_root`.
3. In `datasets.FashionIQ.__init__` (`HINT/datasets.py:33`):
   ```python
   if not os.path.exists(os.path.join(self.path, 'fashion_iq_data.json')):
       self.fashioniq_data = []
       self.train_init_process()
   ```
4. Because `cache_root / "fashion_iq_data.json"` never exists when `cache_root` is freshly yielded, the native constructor **ALWAYS** enters the `not os.path.exists` branch and **ALWAYS** executes `self.train_init_process()`.
5. Therefore, under cache isolation, running `train_init_process()` across `['dress', 'shirt', 'toptee']` is deterministic and unavoidable.

---

## 7. Existing Tests Analysis: Production vs Redefined Copies

### Per-Test Analysis

1. **`workbench/tests/test_hint_replay.py`**:
   - Does **not** call `validate_dataset`.
   - Imports `evaluate_categories` and `isolated_cache_root` (`lines 13-14`).
   - Mocks `datasets.FashionIQ` with a stub `fake_fashioniq` (`line 44`) to test cache isolation invariants without touching disk.

2. **`workbench/tests/test_habit_replay.py`**:
   - Imports production `validate_dataset` and `preflight` from `workbench.replay.habit` (`lines 13-18`):
     ```python
     from workbench.replay.habit import (
         CHECKPOINTS,
         DEFAULT_VARIANT,
         MODEL,
         PROTOCOL,
         preflight,
         validate_checkpoint,
         validate_dataset,
         validate_source,
     )
     ```
   - Calls production `preflight(ckpt_dir, data_dir, source_dir)` at `line 42`, which invokes production `validate_dataset(data_dir, categories)`.
   - Does not redefine `validate_dataset`.

3. **`workbench/tests/test_conesep_replay.py`**:
   - Imports production `validate_dataset` and `preflight` from `workbench.replay.conesep` (`lines 13-19`):
     ```python
     from workbench.replay.conesep import (
         CHECKPOINTS,
         DEFAULT_VARIANT,
         KNOWN_AUTHOR_SHA256,
         MODEL,
         PROTOCOL,
         preflight,
         validate_checkpoint,
         validate_dataset,
         validate_source,
     )
     ```
   - Calls production `preflight(ckpt_dir, data_dir, source_dir)` at `line 44`, which invokes production `validate_dataset(data_dir, categories)`.
   - Does not redefine `validate_dataset`.

4. **`workbench/tests/test_airknow_replay.py`**:
   - Imports production `validate_dataset` and `preflight` from `workbench.replay.airknow` (`lines 13-19`):
     ```python
     from workbench.replay.airknow import (
         CHECKPOINTS,
         DEFAULT_VARIANT,
         KNOWN_AUTHOR_SHA256,
         MODEL,
         PROTOCOL,
         preflight,
         validate_checkpoint,
         validate_dataset,
         validate_source,
     )
     ```
   - Calls production `preflight(ckpt_dir, data_dir, source_dir)` at `line 44`, which invokes production `validate_dataset(data_dir, categories)`.
   - Does not redefine `validate_dataset`.

5. **`workbench/tests/test_f05_f08_f09_verification.py`**:
   - Does **not** import or call production `validate_dataset`.
   - Redefines a local test helper `validate_dataset_with_train` inside `test_f05_extended_validator_detects_missing_train_captions` (`lines 79-95`):
     ```python
     def test_f05_extended_validator_detects_missing_train_captions():
         def validate_dataset_with_train(dataset_root: Path, categories: tuple[str, ...] = CATEGORIES) -> list[str]:
             blockers: list[str] = []
             for cat in CATEGORIES:
                 for rel in (
                     f"captions/cap.{cat}.train.json",
                     f"captions/cap.{cat}.val.json",
                     f"captions/correction_dict_{cat}.json",
                     f"image_splits/split.{cat}.val.json",
                 ):
                     if not (dataset_root / rel).is_file():
                         blockers.append(f"dataset input missing: {dataset_root / rel}")
                 cat_resized = dataset_root / "resized_image" / cat
                 if not cat_resized.is_dir():
                     blockers.append(f"dataset input missing: {cat_resized}")
             return blockers
     ```
     Note: In this test-redefined copy, line 81 hardcodes `for cat in CATEGORIES:`, ignoring the `categories` parameter completely.

6. **`workbench/tests/test_verified_declarations.py`**:
   - Calls production `module.validate_dataset(root)` for `hint`, `encoder`, and `pair` at `lines 57` and `63`:
     ```python
     module = importlib.import_module(f"workbench.replay.{module_name}")
     root = _val_only_root(tmp_path)
     blockers = module.validate_dataset(root)
     ```
   - Uses default parameter `categories=CATEGORIES`.

---

## 8. Comprehensive Per-Wrapper Evidence Table (Questions 2, 3, 7)

| Wrapper | `validate_dataset` definition file:line | Gates on `categories` argument only? | Category set checked on `categories=('dress',)` | Native constructor heeds `categories`? | Production caller of `validate_dataset` (file:line & value passed) | Existing tests invoking it (production vs copy & file:line) |
|---|---|---|---|---|---|---|
| `hint` | `workbench/replay/hint.py:83-112` | **Yes** (`for category in categories:`) | `{'dress'}` only | **No** (signature: `path, transform`) | `hint.py:119` in `preflight` (passes `categories` from `main:202`: `(args.category,)` if specified, else `CATEGORIES`) | `test_hint_replay.py`: does not call.<br>`test_verified_declarations.py:57`: calls production with default. |
| `habit` | `workbench/replay/habit.py:69-81` | **Yes** (`for category in categories:`) | `{'dress'}` only | **No** (signature has `category=None`, but `__init__` unconditionally loops `['dress', 'shirt', 'toptee']`) | `habit.py:95` in `preflight` (passes `categories` from `main:185`: `(args.category,)` if specified, else `CATEGORIES`) | `test_habit_replay.py:42`: calls production via `preflight` with default `CATEGORIES`. |
| `intent` | `workbench/replay/intent.py:69-81` | **Yes** (`for category in categories:`) | `{'dress'}` only | **No** (signature has `category=None`, but `__init__` unconditionally loops `['dress', 'shirt', 'toptee']`) | `intent.py:95` in `preflight` (passes `categories` from `main:185`: `(args.category,)` if specified, else `CATEGORIES`) | `test_intent_replay.py:45`: calls production via `preflight` with default `CATEGORIES`. |
| `conesep` | `workbench/replay/conesep.py:81-93` | **Yes** (`for category in categories:`) | `{'dress'}` only | **No** (signature: `path, transform, noise_ratio`) | `conesep.py:107` in `preflight` (passes `categories` from `main:199`: `(args.category,)` if specified, else `CATEGORIES`) | `test_conesep_replay.py:44`: calls production via `preflight` with default `CATEGORIES`. |
| `airknow` | `workbench/replay/airknow.py:84-96` | **Yes** (`for category in categories:`) | `{'dress'}` only | **No** (signature: `path, transform, noise_ratio`) | `airknow.py:110` in `preflight` (passes `categories` from `main:214`: `(args.category,)` if specified, else `CATEGORIES`) | `test_airknow_replay.py:44`: calls production via `preflight` with default `CATEGORIES`. |
| `pair` | `workbench/replay/pair.py:83-106` | **Yes** (`for category in categories:`) | `{'dress'}` only | **No** (signature: `path, transform, split`) | `pair.py:120` in `preflight` (passes `categories` from `main:202`: `tuple(args.categories)` if specified, else `CATEGORIES`) | `test_pair_replay.py:40`: calls production via `preflight` with default.<br>`test_verified_declarations.py:57`: calls production with default. |

---

## Verified claims

| Claim | Verification Route / Command | Observed Output |
|---|---|---|
| Native `FashionIQ` loops over `['dress', 'shirt', 'toptee']` in `train_init_process` | Source inspection `workbench/third_party/HINT/datasets.py:59-73` | `for name in ['dress', 'shirt', 'toptee']:` opens `cap.{name}.train.json` and `correction_dict_{name}.json` |
| Native `FashionIQ` loads all 3 categories in `get_test_data` | Source inspection `workbench/third_party/HINT/datasets.py:39-41` | Calls `get_test_data('dress')`, `get_test_data('shirt')`, `get_test_data('toptee')` |
| All 6 wrappers return 0 blockers on dress-only root when `categories=('dress',)` | `.venv/bin/python -c "from workbench.replay.hint import validate_dataset as v; ..."` | `hint: []`, `habit: []`, `intent: []`, `conesep: []`, `airknow: []`, `pair: []` |
| Native constructor throws `FileNotFoundError` on dress-only root in isolated cache | `.venv/bin/python -c "from workbench.replay.common import isolated_cache_root; ..."` | `CRASH: FileNotFoundError opening .../captions/cap.shirt.train.json` |
| `isolated_cache_root` drops existing cache files | `.venv/bin/python -c "from workbench.replay.common import isolated_cache_root; ..."` | `fashion_iq_data.json in cache_root: False`, `test_queries_dress.pkl in cache_root: False` |
| Backend shared preflight (`model_requirements`) blocks on dress-only root for all 6 models | `.venv/bin/python -c "from workbench.backend.preflight import model_requirements; ..."` | Blockers for missing shirt/toptee on all models: `hint: 12`, `habit: 10`, `intent: 10`, `conesep: 10`, `airknow: 10`, `pair: 10` |
| Shared adapters do not pass `--category` to replay scripts | Source inspection `workbench/backend/adapters/models.py:30-160` | `_replay_command` and `HintAdapter.command` never append `--category` (only `LIMNAdapter` appends it) |
| Wrappers construct native dataset after loading model | Source inspection `workbench/replay/hint.py:152-169`, `habit.py:118-134`, `pair.py:138-150` | `load_trusted_whole_model` precedes `datasets.FashionIQ(...)` |
| `test_f05_f08_f09_verification.py` defines local copy of validator | Source inspection `workbench/tests/test_f05_f08_f09_verification.py:79-95` | `def validate_dataset_with_train(dataset_root, categories=CATEGORIES)` defined inside test function |
| `test_habit_replay.py`, `test_conesep_replay.py`, `test_airknow_replay.py` call production `preflight` | Source inspection `workbench/tests/test_habit_replay.py:42`, `test_conesep_replay.py:44`, `test_airknow_replay.py:44` | Directly import and invoke production `preflight`, which calls production `validate_dataset` |
