# CSMCIR Required Patch Set Review (CS_03_PATCHSET)

## 1. approved_patch_state analysis and partial patch state evaluation

`approved_patch_state` is defined in `workbench/backend/preflight.py:121-160`:

```python
def approved_patch_state(model: dict, source: Path | None) -> dict[str, Any] | None:
    """Whether a model's declared approved patch is applied to the checkout at *source*.

    Returns `None` when the model declares no approved patch. Otherwise `applied` says whether the
    checkout's file carries the recorded post-patch digest, and `detail` explains a checkout that is
    neither the published original nor the approved variant.
    """
    patches = model.get("approved_source_patches") or []
    if not patches:
        return None
    if source is None or not source.is_dir():
        return {"applied": False, "patch_file": None, "patch_sha256": None,
                "detail": "the source checkout is not present, so no patch state can be observed"}

    from workbench.backend.registry import sha256_file

    for patch in patches:
        target = source / patch["path"]
        if not target.is_file():
            return {"applied": False, "patch_file": patch.get("patch_file"),
                    "patch_sha256": patch.get("patch_sha256"),
                    "detail": f"the target file is missing: {target}"}
        digest = sha256_file(target)
        if digest == patch.get("derived_sha256"):
            return {"applied": True, "patch_file": patch.get("patch_file"),
                    "patch_sha256": patch.get("patch_sha256"),
                    "detail": f"{patch['path']} matches the recorded post-patch digest"}
        if digest == patch.get("base_sha256"):
            return {"applied": False, "patch_file": patch.get("patch_file"),
                    "patch_sha256": patch.get("patch_sha256"),
                    "detail": (f"{patch['path']} is the unmodified published file; apply "
                               f"{patch.get('patch_file')} to a derived checkout of the pinned "
                               f"revision and register that checkout as the model source")}
        return {"applied": False, "patch_file": patch.get("patch_file"),
                "patch_sha256": patch.get("patch_sha256"),
                "detail": (f"{patch['path']} matches neither the published file nor the approved "
                           f"variant (sha256={digest[:16]}...); an unexpected modification is not "
                           f"covered by the approved patch")}
    return None
```

### Proof of loop early exit
The loop over `patches` begins at line 137 (`for patch in patches:`). Every conditional path inside the loop terminates with an unconditional `return`:
1. Line 139-143: `if not target.is_file(): return {"applied": False, ...}` (line 140)
2. Line 144-147: `if digest == patch.get("derived_sha256"): return {"applied": True, ...}` (line 145)
3. Line 148-153: `if digest == patch.get("base_sha256"): return {"applied": False, ...}` (line 149)
4. Line 154-158: final fallback inside the loop: `return {"applied": False, ...}` (line 154)

There is no `continue`, no accumulator, and no branch that falls through to subsequent loop iterations. As a consequence:
- The loop unconditionally exits on iteration 0 (`patches[0]`).
- Any declared patches beyond `patches[0]` are never inspected.
- `applied=True` requires only that `patches[0]` matches its `derived_sha256`.

### Verdict on simulated partial state
In a simulated checkout where patch 1 (`src/lavis/models/__init__.py`) is applied (`digest == derived_sha256`) and patch 2 (`src/validate_blip_csmcir.py`) is unapplied (`digest == base_sha256` or even absent), `approved_patch_state` returns:
```python
{
    'applied': True,
    'patch_file': 'workbench/patches/csmcir_models_init_compat.patch',
    'patch_sha256': '0e703584a7ecb5df3781574c2516fd406745c414491b6aa487fbb3e76532b9bd',
    'detail': 'src/lavis/models/__init__.py matches the recorded post-patch digest'
}
```
Evaluation verdict: The function reports `applied: True` despite patch 2 being unapplied. Even if the file for patch 2 is deleted from disk, the function returns `applied: True`.

---

## 2. Models declaring multiple approved_source_patches

Registry sweep of all 16 models in `workbench/registry/models.yaml`:

| Model ID | `approved_source_patches` Count | Target Paths |
|---|---|---|
| `csmcir` | 2 | `src/lavis/models/__init__.py`, `src/validate_blip_csmcir.py` |
| `airknow` | 0 (`None`) | None |
| `conesep` | 0 (`None`) | None |
| `ptha_mtst` | 0 (`None`) | None |
| `habit` | 0 (`None`) | None |
| `intent` | 0 (`None`) | None |
| `hint` | 0 (`None`) | None |
| `encoder` | 0 (`None`) | None |
| `pair` | 0 (`None`) | None |
| `clvc_net` | 0 (`None`) | None |
| `dcnet` | 0 (`None`) | None |
| `combiner_rn50x4_noft` | 0 (`None`) | None |
| `clip4cir_rn50x4_fullft` | 0 (`None`) | None |
| `tgcir` | 0 (`None`) | None |
| `sprc` | 0 (`None`) | None |
| `limn` | 0 (`None`) | None |

Only **`csmcir`** declares more than one `approved_source_patches` entry (count: 2).
Currently affected models: `csmcir` is the sole model affected by the first-patch-only behaviour.

---

## 3. The two CSMCIR patches (targets, purposes, digests, reasons)

Registry entries quoted verbatim from `workbench/registry/models.yaml:73-86`:

```yaml
    approved_source_patches:
      - path: src/lavis/models/__init__.py
        patch_file: workbench/patches/csmcir_models_init_compat.patch
        patch_sha256: 0e703584a7ecb5df3781574c2516fd406745c414491b6aa487fbb3e76532b9bd
        base_sha256: 9d8c914b551a7de614e062f9a241f44626fd591a31a0ccf1cc79f0ae23c9db01
        derived_sha256: aa68fe5e20bcda5e17d978799deb8ec0ff4636e5b3b2a6ef637ce5e385b0c818
        reason: "98 unguarded imports of modules absent from the pinned public revision prevent the package from importing, so the shipped CSMCIR class is never registered"
      - path: src/validate_blip_csmcir.py
        patch_file: workbench/patches/csmcir_validate_visualization_opt_in.patch
        patch_sha256: eef0594109aa7e65a024a49bf3784f287ec21cbb466c7fc84712e5ba98582f68
        base_sha256: aa7a4395cdb700fded0567e0969da774a331927eaea327b5a23da82fdc7506a9
        derived_sha256: 84a1d476e99d9687b16b1a9c5238f89295f23d2a6f9f4dfb449acba2c118262c
        reason: "the evaluator prints its metric JSON and then unconditionally runs the `vis_fiq` diagnostic dump, which raises AttributeError on the val-relative dataset's target-id field and exits non-zero after the metrics were already computed; the patch makes that dump opt-in and leaves the metric stage untouched"
```

### Patch 1: `csmcir_models_init_compat.patch`
- Target file: `src/lavis/models/__init__.py`
- Base SHA256: `9d8c914b551a7de614e062f9a241f44626fd591a31a0ccf1cc79f0ae23c9db01`
- Derived SHA256: `aa68fe5e20bcda5e17d978799deb8ec0ff4636e5b3b2a6ef637ce5e385b0c818`
- Patch SHA256: `0e703584a7ecb5df3781574c2516fd406745c414491b6aa487fbb3e76532b9bd`
- Reason: `"98 unguarded imports of modules absent from the pinned public revision prevent the package from importing, so the shipped CSMCIR class is never registered"`
- Purpose: Upstream lines 44–151 contain 98 unguarded import statements for modules not committed to upstream git. Because line 152 imports and registers `Blip2QformerCirAlignPromptCSMCIR` (`@registry.register_model("blip2_cir_align_prompt_csmcir")`), the unhandled `ModuleNotFoundError` halts package initialization before registration. The patch wraps the missing imports into four `try/except ImportError: pass` blocks, allowing LAVIS to register the CSMCIR model.

### Patch 2: `csmcir_validate_visualization_opt_in.patch`
- Target file: `src/validate_blip_csmcir.py`
- Base SHA256: `aa7a4395cdb700fded0567e0969da774a331927eaea327b5a23da82fdc7506a9`
- Derived SHA256: `84a1d476e99d9687b16b1a9c5238f89295f23d2a6f9f4dfb449acba2c118262c`
- Patch SHA256: `eef0594109aa7e65a024a49bf3784f287ec21cbb466c7fc84712e5ba98582f68`
- Reason: `"the evaluator prints its metric JSON and then unconditionally runs the `vis_fiq` diagnostic dump, which raises AttributeError on the val-relative dataset's target-id field and exits non-zero after the metrics were already computed; the patch makes that dump opt-in and leaves the metric stage untouched"`
- Purpose: Upstream `validate_blip_csmcir.py` unconditionally runs `vis_fiq(...)` diagnostic image dumping after printing evaluation metrics to stdout. In validation mode, `vis_fiq` executes `label[0].sum() > 0`, where `label` is the target ID string (`data[1]`), raising `AttributeError: 'str' object has no attribute 'sum'` and terminating with exit code 1. The patch adds `--visualize` CLI parameter (defaulting to False) and inserts `if not visualize: return` immediately after metric printing, making the diagnostic dump opt-in and allowing clean process exit.

---

## 4. Call sites of approved_patch_state and consumption analysis

`approved_patch_state` has 4 production call sites and 1 test module call site:

### 1. `workbench/backend/preflight.py:49` in `model_requirements()`
```python
    if model.get("source_blocker"):
        repair = approved_patch_state(model, source)
        if repair is not None and repair["applied"]:
            note = (f"source repaired by the approved patch {repair['patch_file']} "
                    f"(sha256={repair['patch_sha256']}); the published-source blocker no longer applies "
                    f"and the run is an approved compatibility variant, not official parity")
            notes.append(note)
```
- Action: Checks `repair["applied"]`. If True, adds a warning note declaring source repaired and explicitly formats `repair['patch_file']` (which is patch 1).
- Distinguishes partial from complete? **No.** Treats any `applied: True` as complete repair.

### 2. `workbench/backend/preflight.py:200` in `preflight_blockers()`
```python
    declared_source_blocker = model.get("source_blocker")
    if declared_source_blocker:
        repair = approved_patch_state(model, source)
        if repair is None:
            blockers.insert(0, f"source cannot run: {declared_source_blocker}")
        elif not repair["applied"]:
            blockers.insert(0, f"source cannot run: {declared_source_blocker}; {repair['detail']}")
```
- Action: If `repair["applied"]` is True, no blocker is added. If False, prepends blocker string with `repair['detail']`.
- Distinguishes partial from complete? **No.** If patch 1 is applied, `repair["applied"]` is True, and the blocker is suppressed entirely, even if patch 2 is unapplied.

### 3. `workbench/scripts/evaluate_models.py:297` in `_variant_provenance()`
```python
    source = config.WORKBENCH_THIRD_PARTY_ROOT / model["source_dir"] if model.get("source_dir") else None
    patch = approved_patch_state(model, source)
    approved_patch = None
    if patch is not None and patch.get("applied"):
        approved_patch = {key: patch[key] for key in
                          ("patch_file", "patch_sha256", "applied") if key in patch}
```
- Action: Packages `patch_file`, `patch_sha256`, and `applied` into the provenance record under `"approved_source_patch"`.
- Distinguishes partial from complete? **No.** Records only single patch metadata from patch 1.

### 4. `workbench/scripts/pc_readiness.py:226` in `model_section()`
```python
    source = config.WORKBENCH_THIRD_PARTY_ROOT / model["source_dir"] if model.get("source_dir") else None
    patch = approved_patch_state(model, source)
    ...
    return {
        ...
        "approved_patch": patch,
        ...
    }
```
- Action: Places `patch` dict directly into readiness section output (`line 300`). Combined with `preflight_blockers` (`line 223`), clears source blockers.
- Distinguishes partial from complete? **No.**

### Indirect consumers via `preflight_blockers()`:
- `workbench/scripts/prepare_model.py:40`: `blockers = preflight_blockers(model, config, canonical_root=config.FASHIONIQ_ROOT)`
- `workbench/scripts/doctor.py:151`: `blockers = preflight_blockers(model, config, canonical_root=config.FASHIONIQ_ROOT)`

Neither indirect consumer distinguishes partial from complete.

---

## 5. Registry schema expressiveness for required vs optional patches

In `workbench/registry/models.yaml`:
The schema does **not** distinguish patches required for the default benchmark workflow from optional or diagnostic patches.

### Schema fields on `csmcir` (`workbench/registry/models.yaml:3-138`):
- Model-level fields:
  `model_id`, `environment`, `canonical_dataset_link`, `method_name`, `research_generation`, `publication_year`, `preparation_contract`, `paper_title`, `paper_url`, `upstream_repo_url`, `upstream_default_branch`, `upstream_commit_sha`, `source_available`, `source_dir`, `fashioniq_layout`, `fashioniq_image_extensions`, `literature_split_label`, `native_protocol`, `supported_protocols`, `reported_r10`, `reported_r50`, `reported_mean`, `reproduction_status`, `strict_clean_checkpoint_available`, `approved_source_patches`, `source_blocker`, `notes`, `checkpoint_variants`.
- Fields present inside each `approved_source_patches` item:
  `path`, `patch_file`, `patch_sha256`, `base_sha256`, `derived_sha256`, `reason`.

### Schema fields on comparison model `combiner_rn50x4_noft` (`workbench/registry/models.yaml:806-860`):
`model_id`, `canonical_dataset_link`, `fashioniq_image_extensions`, `replay_status`, `category_model_policy`, `checkpoint_serialization`, `replay_type`, `environment_required`, `method_name`, `research_generation`, `publication_year`, `paper_title`, `paper_url`, `upstream_repo_url`, `upstream_default_branch`, `upstream_commit_sha`, `source_available`, `source_dir`, `fashioniq_layout`, `preparation_contract`, `preparation_id`, `literature_split_label`, `native_protocol`, `supported_protocols`, `reported_r10`, `reported_r50`, `reported_mean`, `reproduction_status`, `command_status`, `environment`, `strict_clean_checkpoint_available`, `notes`, `checkpoint_variants`.

### Evaluation:
There is **no** existing field expressing "required for workflow X", "optional", "scope", or "stage" anywhere in `models.yaml`. Both patches in `approved_source_patches` carry identical schema structures.

---

## 6. Native CSMCIR benchmark shell and visualization crash

### Native shell execution
In `workbench/third_party/CSMCIR/src/validate_blip_csmcir.py`:
- `main()` (lines 500-517): parses CLI arguments (`--dataset`, `--blip-model-name`, `--blip-model-path`, `--target-ratio`, `--transform`). When `--dataset fashionIQ`, calls:
  `blip_validate_fashioniq(args.blip_model_name, args.blip_model_path, args.transform, args.target_ratio)` (line 515).
- `blip_validate_fashioniq()` (lines 565-657):
  - Evaluates validation sets across categories (`dress`, `toptee`, `shirt`), computing Recall@10, Recall@50, and mean recall.
  - Prints metric JSON to stdout at line 631:
    ```python
    print(json.dumps(results_dict, indent=4))
    ```
  - Unconditionally starts visualization dump at lines 634-655:
    ```python
    # 可视化部分
    for relative_val_dataset, classic_val_dataset, idx in zip(relative_val_datasets, classic_val_datasets,
                                                              idx_to_dress_mapping):
        index_features, index_names = extract_index_blip_caption_features(classic_val_dataset, blip_model)
        captions_all = []
        reference_names = []
        labels = []
        for data in relative_val_dataset:
            try:
                captions_all.append(data[2])
                reference_names.append(data[0])
                labels.append(data[1])
            except:
                continue
        sorted_index_names = []
        # 这里需要根据具体的排序逻辑对 index_names 进行排序
        # 假设已经有排序逻辑，这里简单示例
        sorted_index_names = index_names
        dress_type = idx_to_dress_mapping[idx]
        vis_fiq(sorted_index_names, reference_names, captions_all, labels, dress_type)
    ```

### The line that raises `'str' object has no attribute 'sum'`
Location: `workbench/third_party/CSMCIR/src/validate_blip_csmcir.py:67` inside `vis_fiq`:
```python
def vis_fiq(sorted_index_names, reference_names, captions_all, labels, dress_type):
    base_path = os.path.join(os.getcwd(), f'fiq_main/vis{dress_type}')
    img_path = os.path.join(os.getcwd(), 'fashionIQ_dataset/images')
    for i in range(len(captions_all)):
        index_name = sorted_index_names[i]
        label = labels[i]
        caption = captions_all[i]
        reference = reference_names[i]
        if (label[0].sum() > 0).item() is True: 
```

**Root cause**:
In `workbench/third_party/CSMCIR/src/data_utils_csmcir.py:454-457`:
```python
                elif self.split == 'val':
                    target_name = self.triplets[index]['target']
                    return reference_name, target_name, image_captions,image_captions_cot,target_image_caption
```
`data[1]` is `target_name` (a string target identifier such as `"B00006XX"`).
At line 648 of `validate_blip_csmcir.py`, `labels.append(data[1])` stores this string into `labels`.
At line 64 of `validate_blip_csmcir.py`, `label = labels[i]` is a Python string.
At line 67, `label[0]` is a 1-character string, and `label[0].sum()` raises:
`AttributeError: 'str' object has no attribute 'sum'`.

### Lines changed by `csmcir_validate_visualization_opt_in.patch`
Quoted from `workbench/patches/csmcir_validate_visualization_opt_in.patch`:
```diff
--- a/src/validate_blip_csmcir.py
+++ b/src/validate_blip_csmcir.py
@@ -506,13 +506,21 @@
     parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
     parser.add_argument("--transform", default="targetpad", type=str,
                         help="Preprocess pipeline, should be in ['clip', 'squarepad', 'targetpad'] ")
+    # Diagnostic image dumps (`vis_fiq`) are a development aid, not part of measurement. They were
+    # previously unconditional and ran AFTER the metrics were printed, so a benchmark run died in the
+    # drawing code -- `vis_fiq` treats `data[1]` of the val-relative dataset as a score tensor while
+    # it is a target id string -- and exited non-zero with the metrics already computed. Making it
+    # opt-in keeps the drawing tools untouched and leaves the metric stage unaffected.
+    parser.add_argument("--visualize", action="store_true",
+                        help="write diagnostic image dumps after evaluation (off by default)")
 
     args = parser.parse_args()
 
     if args.dataset == 'CIRR':
         blip_validate_cirr(args.blip_model_name, args.blip_model_path, args.transform, args.target_ratio)
     elif args.dataset == 'fashionIQ':
-        blip_validate_fashioniq(args.blip_model_name, args.blip_model_path, args.transform, args.target_ratio)
+        blip_validate_fashioniq(args.blip_model_name, args.blip_model_path, args.transform,
+                                args.target_ratio, visualize=args.visualize)
 
 
 
@@ -562,7 +570,8 @@
 
 
 
-def blip_validate_fashioniq(blip_model_name, blip_model_path, transform, target_ratio):
+def blip_validate_fashioniq(blip_model_name, blip_model_path, transform, target_ratio,
+                            visualize: bool = False):
 
     # clip_model, clip_preprocess = clip.load(clip_model_name, device=device, jit=False)
     blip_model, _, txt_processors = load_model_and_preprocess(name=blip_model_name, model_type="pretrain", is_eval=False, device=device)
@@ -630,7 +639,10 @@
     print(json.dumps(results_dict, indent=4))
 
 
-    # 可视化部分
+    # 可视化部分 -- opt-in only. `--visualize` is off for evaluation so the metric stage above is the
+    # last thing a benchmark run depends on.
+    if not visualize:
+        return
     for relative_val_dataset, classic_val_dataset, idx in zip(relative_val_datasets, classic_val_datasets,
                                                               idx_to_dress_mapping):
         index_features, index_names = extract_index_blip_caption_features(classic_val_dataset, blip_model)
```

---

## 7. Existing test coverage for partial patch states

**Verdict**: No existing test covers a partial patch state for `approved_patch_state`.

### Test suite survey:
1. `workbench/tests/test_approved_source_patch.py:33-34, 57-73`:
   Explicitly isolates only patch 0:
   ```python
   def _declared() -> dict:
       return _csmcir()["approved_source_patches"][0]
   ```
   All test fixtures copy only `_declared()["path"]` (`src/lavis/models/__init__.py`). Patch 1 (`src/validate_blip_csmcir.py`) is never staged or asserted.
2. `workbench/tests/test_pc_readiness.py:38-42, 94-115`:
   Explicitly stages only patch 0:
   ```python
   relative = _model("csmcir")["approved_source_patches"][0]["path"]
   ...
   patch = REPO / model["approved_source_patches"][0]["patch_file"]
   subprocess.run(["patch", "-p1", "--directory", str(root / "CSMCIR"), "-i", str(patch)], check=True)
   ...
   assert after["approved_patch"]["applied"] is True
   ```
   `src/validate_blip_csmcir.py` is not even present in the checkout, yet `approved_patch["applied"]` asserts `True` due to the loop defect.
3. `workbench/tests/test_vendored_approved_patch.py:49-53`:
   Hardcodes patch 0:
   ```python
   def _contract():
       patches = _model().get("approved_source_patches") or []
       assert patches, "the registry must declare the CSMCIR approved patch"
       return patches[0]
   ```
4. `workbench/tests/test_component_boundaries.py:280-320`:
   - `test_f04_the_patched_checkout_is_approved_and_an_edit_after_it_is_not` (lines 280-302) applies patch 1 only and tests `source_provenance` (the vendored allowlist in `vendored_source.py`), **not** `approved_patch_state`.
   - `test_f04_both_csmcir_patches_apply_together` (lines 305-320) applies both patches sequentially and tests `source_provenance`.
   Neither test exercises `approved_patch_state`.
5. `workbench/tests/test_csmcir_source_blocker.py`:
   Tests only the upstream unmodified unimportable state of `src/lavis/models/__init__.py`.

---

## Verified claims

1. **`approved_patch_state` returns on loop iteration 0**:
   - Verification command:
     ```python
     python -c '
     import tempfile, shutil, subprocess
     from pathlib import Path
     from workbench.backend.preflight import approved_patch_state
     from workbench.backend.registry import model_by_id
     csmcir = model_by_id("csmcir")
     p0 = csmcir["approved_source_patches"][0]
     with tempfile.TemporaryDirectory() as tmp:
         src = Path(tmp) / "CSMCIR"
         (src / Path(p0["path"]).parent).mkdir(parents=True)
         shutil.copyfile(Path("workbench/third_party/CSMCIR") / p0["path"], src / p0["path"])
         subprocess.run(["patch", "-p1", "-d", str(src), "-i", str(Path("workbench/patches/csmcir_models_init_compat.patch").resolve())], check=True)
         print(approved_patch_state(csmcir, src))
     '
     ```
   - Observed output:
     `{'applied': True, 'patch_file': 'workbench/patches/csmcir_models_init_compat.patch', 'patch_sha256': '0e703584a7ecb5df3781574c2516fd406745c414491b6aa487fbb3e76532b9bd', 'detail': 'src/lavis/models/__init__.py matches the recorded post-patch digest'}`

2. **Only `csmcir` declares more than one patch in `models.yaml`**:
   - Verification command:
     ```python
     python -c '
     import yaml
     models = yaml.safe_load(open("workbench/registry/models.yaml"))["models"]
     multi = [(m["model_id"], len(m.get("approved_source_patches") or [])) for m in models if len(m.get("approved_source_patches") or []) > 1]
     print(multi)
     '
     ```
   - Observed output:
     `[('csmcir', 2)]`

3. **`vis_fiq` string `.sum()` failure reproduced**:
   - Verification command:
     ```python
     python -c '
     label = "B00006XX"
     try:
         label[0].sum()
     except Exception as e:
         print(f"{type(e).__name__}: {e}")
     '
     ```
   - Observed output:
     `AttributeError: 'str' object has no attribute 'sum'`

4. **Both patch files match registry SHA256 declarations**:
   - Verification command:
     ```python
     python -c '
     import hashlib
     from workbench.backend.registry import model_by_id
     for p in model_by_id("csmcir")["approved_source_patches"]:
         h = hashlib.sha256(open(p["patch_file"], "rb").read()).hexdigest()
         assert h == p["patch_sha256"]
         print(p["patch_file"], "MATCH")
     '
     ```
   - Observed output:
     `workbench/patches/csmcir_models_init_compat.patch MATCH`
     `workbench/patches/csmcir_validate_visualization_opt_in.patch MATCH`
