# Final Model Audit: combiner_rn50x4_noft

Audit of registered model `combiner_rn50x4_noft` against pinned source `workbench/third_party/CLIP4Cir/` at commit `dfed9f748a8a4d05abf613164e06d59b22dbdf13`.
Protocol: Protocol A (`fashioniq_original_split` — full validation split gallery, reference remains eligible).

---

## A. MODEL IDENTITY

- **Model ID**: `combiner_rn50x4_noft` (`workbench/registry/models.yaml:660`)
- **Method Name**: `Combiner RN50x4 noft` (`workbench/registry/models.yaml:672`)
- **Research Generation**: `legacy` (`workbench/registry/models.yaml:673`)
- **Publication Year**: `2023` (`workbench/registry/models.yaml:674`)
- **Paper Title**: *Composed Image Retrieval using Contrastive Learning and Task-oriented CLIP-based Features* (`workbench/registry/models.yaml:675`)
- **Paper URL**: `https://arxiv.org/abs/2308.11485` (arXiv:2308.11485v1, published 2023-08-22; ACM TOMM 2023)
- **Upstream Repository**: `https://github.com/ABaldrati/CLIP4Cir` (`workbench/registry/models.yaml:677`)
- **Pinned Commit SHA**: `dfed9f748a8a4d05abf613164e06d59b22dbdf13` (`workbench/registry/models.yaml:679`)
  - Verified: `git -C workbench/third_party/CLIP4Cir rev-parse HEAD` outputs `dfed9f748a8a4d05abf613164e06d59b22dbdf13`.
  - Clean working tree: `git -C workbench/third_party/CLIP4Cir status --short` is empty.
- **Checkpoint Variant**: `rn50x4_noft_unresolved` (`workbench/registry/models.yaml:704-720`)
  - Filename: `rn50x4-noft-unresolved.pt`
  - Artifact Type: `file`
  - Checkpoint Serialization: `state_dict` (`models.yaml:669`)
  - Download URL: `null`
  - Source URL: `https://drive.google.com/drive/folders/1ny2hhzP8HZBnXhjvDEni8P8G4_inCNTv?usp=sharing` (author Google Drive folder)
  - Expected Size: `null`
  - Expected SHA-256: `null`
  - Checkpoint Mapping Status: `UNVERIFIED` (`models.yaml:720`)
- **Backbone & Evaluation Transform**:
  - Backbone: OpenAI CLIP `RN50x4` (`input_resolution = 288`, `feature_dim = 640`). Both image and text encoders are frozen (no fine-tuning).
  - Evaluation Transform: Source transform `targetpad_transform(target_ratio=1.25, dim=288)` defined in `workbench/third_party/CLIP4Cir/src/data_utils.py:82-96`. It is NOT an inherited transform from `clip.load` (`clip_preprocess` is bypassed when `--transform targetpad` is specified).
- **Paper Benchmark Row**:
  - Registered reference: `clip4cir_rn50x4_noft_fiq_absent_in_paper` (`workbench/registry/paper_benchmarks/papers.yaml:293-355`).
  - Critical Finding: The cited paper TOMM 2023 (arXiv:2308.11485) reports **only fine-tuned CLIP4Cir** (Table 9) for RN50x4 on FashionIQ. It contains **no evaluation row** for RN50x4 noft.
  - Published metrics: `r10: null`, `r50: null`, `mean: null` (`papers.yaml:325-327`, `models.yaml:688-690`).
  - Reference status: `reference_status: UNVERIFIED`, `review_status: PENDING`.
  - External publication: Baldrati et al., CVPR 2022 Demo paper (*Effective Conditioned and Composed Image Retrieval Combining CLIP-Based Features*), Table 1 reports FashionIQ validation metrics for frozen RN50x4 + Combiner:
    - Dress: R@10 = 36.36, R@50 = 58.00
    - Shirt: R@10 = 31.63, R@50 = 56.67
    - Toptee: R@10 = 38.19, R@50 = 62.42
    - Average: R@10 = 35.39, R@50 = 59.03, Mean = 47.21 (unweighted arithmetic macro mean)
    - Note: This CVPR 2022 paper is **not registered** in `papers.yaml`. Per requirement, no published checkpoint-paper comparison can be claimed against the cited paper.

---

## B. SOURCE COVERAGE

### Tree Inventory
Pinned tree: `workbench/third_party/CLIP4Cir` (11 tracked files in git):
- `.gitignore`
- `LICENSE`
- `README.md`
- `src/validate.py` (360 lines; core evaluator for FashionIQ and CIRR)
- `src/combiner.py` (65 lines; Combiner fusion module)
- `src/data_utils.py` (277 lines; datasets, TargetPad, SquarePad)
- `src/utils.py` (131 lines; extract_index_features, collate_fn, device)
- `src/clip_fine_tune.py` (340 lines; CLIP fine-tuning training script)
- `src/combiner_train.py` (477 lines; Combiner training script)
- `src/cirr_test_submission.py` (221 lines; CIRR test server submission formatting)
- `images/` (4 architectural and training diagram images)

### Reachable Evaluation Call Graph
1. Entrypoint: `src/validate.py:main`
2. Argument parsing: `validate.py:263-277` (`--dataset fashionIQ --combining-function combiner --combiner-path <path> --projection-dim 2560 --hidden-dim 5120 --clip-model-name RN50x4 --target-ratio 1.25 --transform targetpad`)
3. Backbone initialization: `clip.load(args.clip_model_name, device=device, jit=False)` (`validate.py:279`)
4. Combiner construction: `Combiner(feature_dim=640, projection_dim=2560, hidden_dim=5120)` (`validate.py:305`, `combiner.py:11-31`)
5. Combiner weight loading: `torch.load(args.combiner_path)` -> `combiner.load_state_dict(state_dict["Combiner"])` (`validate.py:306-307`)
6. Preprocess pipeline: `targetpad_transform(args.target_ratio=1.25, dim=288)` (`validate.py:291`, `data_utils.py:82-96`)
7. Evaluation coordinator: `fashioniq_val_retrieval(dress_type, combining_function, clip_model, preprocess)` (`validate.py:113-132, 329-342`)
8. Gallery feature extraction: `extract_index_features(classic_val_dataset, clip_model)` (`validate.py:128`, `utils.py:21-44`)
9. Query feature generation: `generate_fiq_val_predictions(clip_model, relative_val_dataset, combining_function, index_names, index_features)` (`validate.py:60-110`)
10. Metric computation: `compute_fiq_val_metrics(...)` (`validate.py:21-57`)
11. Macro calculation: `mean(average_recall10_list)`, `mean(average_recall50_list)` (`validate.py:353-354`)

### Inspected vs Excluded
- **Inspected**:
  - `src/validate.py`: core evaluation loop, scoring, ranking, non-masking protocol, macro metric reporting.
  - `src/combiner.py`: Combiner architecture, projection layers, dynamic gating scalar, `combine_features()`.
  - `src/data_utils.py`: dataset initialization, image paths, TargetPad aspect ratio padding, Compose transform.
  - `src/utils.py`: feature extraction loop, DataLoader `collate_fn`, device fallback.
  - `README.md`: author instructions, environment setup, CLI commands.
- **Excluded**:
  - `src/clip_fine_tune.py`: stage 1 contrastive training of CLIP backbone; not executed during evaluation.
  - `src/combiner_train.py`: stage 2 training of Combiner network; not executed during evaluation.
  - `src/cirr_test_submission.py`: test server submission generation for CIRR dataset; irrelevant to FashionIQ evaluation.

### README vs Executable Code
1. **Unpinned CLIP dependency**: `README.md:118` directs `pip install git+https://github.com/openai/CLIP.git` (unpinned git default branch).
2. **Image extension mismatch**: `README.md:152-155` displays dataset images with `.jpg` extension (`B00006M009.jpg`). Executable code in `src/data_utils.py:158, 161, 170, 176` hardcodes `f"{name}.jpg"`. Local canonical FashionIQ dataset stores images strictly as `.png`.
3. **Validation CLI arguments**: `README.md:258-269` documents `--dataset fashionIQ --combining-function combiner --combiner-path ... --clip-model-name RN50x4`. This matches `src/validate.py:263-277` and `CombinerNoftAdapter`.

---

## C. NATIVE EVALUATION

### 1. Gallery Universe and Ordering
- In `src/validate.py:126-130`:
  ```python
  classic_val_dataset = FashionIQDataset('val', [dress_type], 'classic', preprocess)
  index_features, index_names = extract_index_features(classic_val_dataset, clip_model)
  relative_val_dataset = FashionIQDataset('val', [dress_type], 'relative', preprocess)
  ```
- In `src/data_utils.py:144-147`:
  ```python
  self.image_names: list = []
  for dress_type in dress_types:
      with open(base_path / 'fashionIQ_dataset' / 'image_splits' / f'split.{dress_type}.{split}.json') as f:
          self.image_names.extend(json.load(f))
  ```
- In `src/data_utils.py:174-178`:
  ```python
  elif self.mode == 'classic':
      image_name = self.image_names[index]
      image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{image_name}.jpg"
      image = self.preprocess(PIL.Image.open(image_path))
      return image_name, image
  ```
- Gallery items comprise the entire validation split file (`split.{dress_type}.val.json`), preserving sequential order. Gallery counts:
  - `dress`: 3,817 images
  - `shirt`: 6,346 images
  - `toptee`: 5,373 images
  - Total: 15,536 images

### 2. Reference Eligibility (Proof of Non-Masking)
In `src/validate.py:40-56`:
```python
    # Normalize the index features
    index_features = F.normalize(index_features, dim=-1).float()

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100

    return recall_at10, recall_at50
```
- **Proof of Non-Masking**:
  1. `distances` is computed across all `len(index_names)` gallery candidates.
  2. In `generate_fiq_val_predictions` (`validate.py:60-110`), only `target_names` is returned alongside `predicted_features`; `reference_names` is not even passed to `compute_fiq_val_metrics`.
  3. No mask (`-inf`, `1e10`, or boolean indexing) is applied to `distances` or `index_names`.
  4. Contrast with CIRR in the same file (`validate.py:164`):
     ```python
     # Mask the reference image
     distances[sorted_indices == reference_indices] = 1e10
     ```
     CIRR explicitly masks the reference image; FashionIQ does not.
  5. The reference image remains eligible in the candidate pool, can rank within the top 10/50, and can outrank the target image. This is unambiguously Protocol A (`fashioniq_original_split`).

### 3. Evaluation Transforms
- Defined in `src/data_utils.py:82-96`:
  ```python
  def targetpad_transform(target_ratio: float, dim: int):
      return Compose([
          TargetPad(target_ratio, dim),
          Resize(dim, interpolation=PIL.Image.BICUBIC),
          CenterCrop(dim),
          _convert_image_to_rgb,
          ToTensor(),
          Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
      ])
  ```
- Invoked at `src/validate.py:289-291` with `--target-ratio 1.25` and `input_dim = clip_model.visual.input_resolution` (= 288 for RN50x4).
- Applied symmetrically to gallery images (`classic_val_dataset`) and reference images (`relative_val_dataset`).

### 4. Similarity Scoring & Fusion
- Model: `Combiner` (`src/combiner.py:11-31, 50-65`):
  ```python
  text_projected_features = self.dropout1(F.relu(self.text_projection_layer(text_features)))
  image_projected_features = self.dropout2(F.relu(self.image_projection_layer(image_features)))
  raw_combined_features = torch.cat((text_projected_features, image_projected_features), -1)
  combined_features = self.dropout3(F.relu(self.combiner_layer(raw_combined_features)))
  dynamic_scalar = self.dynamic_scalar(raw_combined_features)
  output = self.output_layer(combined_features) + dynamic_scalar * text_features + (1 - dynamic_scalar) * image_features
  return F.normalize(output, dim=-1)
  ```
- Query representations: L2 normalized fused features `batch_predicted_features`.
- Distance metric: Cosine distance `distances = 1 - predicted_features @ index_features.T`.

### 5. Sorting, Top-K, and Recall Definition
- Sorting: `sorted_indices = torch.argsort(distances, dim=-1).cpu()` (ascending distance = descending similarity).
- Ground truth match: `labels = (sorted_index_names == target_names)`.
- Invariant check: `assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())` enforces exactly one target per query across the candidate pool.
- Recall@K: `torch.sum(labels[:, :K]) / len(labels) * 100` for $K \in \{10, 50\}$.

### 6. Category Handling & Macro Aggregation
In `src/validate.py:325-354`:
- Evaluates `shirt`, `dress`, `toptee` sequentially.
- Category recalls appended to `average_recall10_list` and `average_recall50_list`.
- Macro aggregation is the unweighted arithmetic mean:
  `average recall10 = mean(average_recall10_list)`
  `average recall50 = mean(average_recall50_list)`
- Overall mean: `(average recall10 + average recall50) / 2` (derived by workbench).

---

## D. PREPARATION & VARIANT SEPARATION

### 1. Dataset Layout & Image Format
- Root expectation: `src/data_utils.py:11` sets `base_path = Path(__file__).absolute().parents[1].absolute()`, requiring `<checkout>/fashionIQ_dataset`.
- Expected structure:
  - `fashionIQ_dataset/captions/cap.{dress,shirt,toptee}.{train,val,test}.json`
  - `fashionIQ_dataset/image_splits/split.{dress,shirt,toptee}.{train,val,test}.json`
  - `fashionIQ_dataset/images/{image_name}.jpg`
- Canonical repository layout: `data/fashionIQ_dataset/images/` contains 74,381 `.png` files and 0 `.jpg` files.
- Silent failure defect: `FashionIQDataset.__getitem__` (`data_utils.py:182-183`) swallows `FileNotFoundError` with `except Exception as e: print(...)` returning `None`. `collate_fn` (`utils.py:87`) filters out `None`. In relative mode, missing images silently drop queries and shrink `len(labels)` (the denominator of Recall@K).

### 2. Assets & Weights
- Base weights required: OpenAI CLIP RN50x4 base weights (`~/.cache/clip/RN50x4.pt`, SHA-256 `7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd`).
- Registered in `workbench/registry/auxiliary_assets.yaml:162-181` and `workbench/registry/preparation_contracts.yaml:155-166`.
- Deterministic at evaluation: Caption composition is deterministic (`validate.py:89-92` concatenates `cap1 and cap2`). Random caption selection (`utils.py:57-78`) is restricted to training scripts.

### 3. Variant Separation (noft vs fullft)
Both variants share the checkout `workbench/third_party/CLIP4Cir`. What separates them:
1. **Source Code Execution Branch**:
   - `combiner_rn50x4_noft`: Invokes `validate.py` **without** `--clip-model-path`. Line 283 (`if args.clip_model_path:`) evaluates `False`, so the frozen OpenAI CLIP weights loaded at line 279 remain unmodified.
   - `clip4cir_rn50x4_fullft`: Invokes `validate.py` **with** `--clip-model-path <path/to/clip_state.pt>`. Lines 283-287 overwrite the CLIP weights with author fine-tuned weights (`clip_model.load_state_dict(saved_state_dict["CLIP"])`).
2. **Checkpoint Artifact Set**:
   - `noft`: Single checkpoint file containing `state_dict["Combiner"]` (`fiq_comb_RN50x4_noft.pt`).
   - `fullft`: Checkpoint bundle containing two state dicts: `combiner_state.pt` (`state_dict["Combiner"]`) and `clip_state.pt` (`saved_state_dict["CLIP"]`).
3. **Encoders & Feature Spaces**:
   - `noft`: Image and text representations originate from out-of-the-box pre-trained OpenAI CLIP RN50x4.
   - `fullft`: Representations originate from task-oriented fine-tuned CLIP encoders.
4. **Shared Components**:
   - Preprocessing transform: Identical `targetpad_transform(1.25, 288)`.
   - Combiner dimensions: Identical projection dimension 2560 and hidden dimension 5120.
   - Normalization: Identical CLIP mean and standard deviation.
5. **Workbench Architecture Separation**:
   - Registered under distinct `model_id`s: `combiner_rn50x4_noft` vs `clip4cir_rn50x4_fullft`.
   - Distinct adapter classes: `CombinerNoftAdapter` vs `CLIP4CirFullftAdapter`.
   - Distinct reproduction statuses: `BLOCKED_CHECKPOINT_MAPPING` vs `BLOCKED_CHECKPOINT_BUNDLE`.
   - The workbench does **not** collapse the two variants.

---

## E. CHECKPOINT

- **Author Checkpoint Identity**:
  - In author Google Drive (`1ny2hhzP8HZBnXhjvDEni8P8G4_inCNTv`), subfolder `FashionIQ/RN50x4_noft/` contains exactly one file: `fiq_comb_RN50x4_noft.pt`.
  - Author Drive README (id `1G7kZo7paiaRd3mif9_saERyhPevRCM88`) documents item 2 as *"RN50x4-based Combiner trained on out-of-the-box CLIP features - CVPR22 Demo paper"*, confirming the state dict key `state_dict["Combiner"]`, projection dim 2560, and hidden dim 5120.
- **Loader Mechanism**:
  ```python
  combiner = Combiner(feature_dim, args.projection_dim, args.hidden_dim).to(device, non_blocking=True)
  state_dict = torch.load(args.combiner_path, map_location=device)
  combiner.load_state_dict(state_dict["Combiner"])
  combiner.eval()
  ```
- **Provenance & Mapping Status**:
  - `checkpoint_id`: `rn50x4_noft_unresolved`
  - Status: `CHECKPOINT_MAPPING_UNVERIFIED`
  - Local weight file is missing (`workbench/artifacts/checkpoints/combiner_rn50x4_noft/rn50x4-noft-unresolved.pt` does not exist).
  - No byte-level digest, size, or direct download URL is published in the registry.
- **Paper Comparison Status**:
  - Pinned paper (`clip4cir_tomm2023`) publishes no evaluation numbers for RN50x4 noft on FashionIQ.
  - Committed benchmark reference `clip4cir_rn50x4_noft_fiq_absent_in_paper` records `reported_metrics: {r10: null, r50: null, mean: null}`.
  - Honest reporting: No published checkpoint-paper comparison exists in `papers.yaml`.

---

## F. WORKBENCH INTEGRATION

1. **Adapter**:
   - `CombinerNoftAdapter` in `workbench/backend/adapters/models.py:86-94`:
     ```python
     class CombinerNoftAdapter(OfficialScriptAdapter):
         model_id = "combiner_rn50x4_noft"
         script = "src/validate.py"

         def command(self, source, checkpoint, request: EvalRequest) -> list[str]:
             return [_runtime.python_executable(self.model), str(source / self.script), "--dataset", "fashionIQ",
                     "--combining-function", "combiner", "--combiner-path", str(checkpoint),
                     "--projection-dim", "2560", "--hidden-dim", "5120",
                     "--clip-model-name", "RN50x4", "--target-ratio", "1.25", "--transform", "targetpad"]
     ```
   - Matches `src/validate.py` CLI arguments exactly, passing `str(checkpoint)` as `--combiner-path` and correctly omitting `--clip-model-path`.
2. **Preparation Contract**:
   - Declared as `preparation_id: clip4cir_fashioniq` (`workbench/registry/preparation_contracts.yaml:140-169`).
   - Declares raw dataset inputs, base RN50x4 weights requirement, deterministic evaluation, and manual authorization policy.
3. **Runtime Preflight & Doctor**:
   - `workbench/backend/preflight.py:preflight_blockers` checks:
     - Canonical dataset symlink (`workbench/third_party/CLIP4Cir/fashionIQ_dataset -> data/fashionIQ_dataset`).
     - Image extension contract (`.jpg`).
     - Auxiliary asset existence (`~/.cache/clip/RN50x4.pt`).
     - Execution environment (`WORKBENCH_CLIP4CIR_PYTHON`).
     - Checkpoint file existence.
4. **Metrics Parser**:
   - Implemented as `parse_combiner_output` in `workbench/backend/metrics_extraction.py:309-382`.
   - Dispatched at line 406 for `combiner_rn50x4_noft`.
   - Parses regex `^\s*([a-z_]+_recallat\d+)\s*=\s*([-+0-9.eE]+)\s*$` and `^\s*average recall(\d+)\s*=\s*([-+0-9.eE]+)\s*$`.
   - Validates internal consistency between category recalls and declared macro averages within 0.05 tolerance. Computes macro R@10, R@50, and derived mean.
5. **Benchmark Gate**:
   - `workbench/backend/paper_gate.py` attempts comparison against `papers.yaml`.
   - Because all committed references for `combiner_rn50x4_noft` are `UNVERIFIED` (`select_reference` returns `None`), `evaluate_gate` returns `METRIC_REFERENCE_UNVERIFIED`. The model is ungradeable by design until a verified benchmark row is committed.

---

## G. OBSERVATION CAPABILITY

### 1. Score & Ranking Production Point
- In `src/validate.py:44-46`:
  ```python
  distances = 1 - predicted_features @ index_features.T
  sorted_indices = torch.argsort(distances, dim=-1).cpu()
  sorted_index_names = np.array(index_names)[sorted_indices]
  ```
- `sorted_index_names` has shape `(num_queries, num_gallery)`.
- Query alignment: Exactly sequential with `relative_val_dataset.triplets`:
  - Reference name: `reference_names[i]`
  - Target name: `target_names[i]`
  - Query captions: `triplets[i]['captions']`
  - Combined text: `f"{cap1.strip('.?, ').capitalize()} and {cap2.strip('.?, ')}"`

### 2. Per-Query Export Feasibility
- Verdict: `QUERY_EXPORT_IMPLEMENTABLE`.
- Native script prints aggregate recall numbers only; no per-query rankings or distances are serialized to disk or stdout.
- Export mechanism: An observer hook or wrapper at line 46 of `compute_fiq_val_metrics` can serialize `sorted_index_names[:, :50]`, target rank indices, cosine similarity scores, and query IDs directly to JSON or Parquet without perturbing evaluation metrics.

---

## H. DEFECTS

### Confirmed Bugs (`CONFIRMED_BUG`)
1. **Silent Query Drop on Missing Image (P0)**:
   - *Path*: `workbench/third_party/CLIP4Cir/src/data_utils.py:182-183` and `src/utils.py:87`
   - *Precondition*: Any missing or unreadable image file in `FashionIQDataset`.
   - *Actual*: `FashionIQDataset.__getitem__` catches the exception and returns `None`. `collate_fn` filters out `None` items. In relative mode, this silently drops queries from the batch, shrinking `len(labels)` (the denominator of Recall@K at `validate.py:54-55`). In classic mode, it silently shrinks the gallery.
   - *Expected*: Should raise `FileNotFoundError` or fail fast loudly.
2. **Hardcoded `.jpg` Image Extension (P0)**:
   - *Path*: `workbench/third_party/CLIP4Cir/src/data_utils.py:158, 161, 170, 176`
   - *Precondition*: Running against canonical FashionIQ dataset where images are stored as `.png`.
   - *Actual*: Attempts to open `f"{name}.jpg"`. All 74,381 images fail to open, triggering Bug 1 above.
   - *Expected*: Extension should match dataset layout or be configurable.
3. **Unpinned Upstream CLIP Dependency (P2)**:
   - *Path*: `workbench/third_party/CLIP4Cir/README.md:118`
   - *Precondition*: Installing environment per README instructions.
   - *Actual*: `pip install git+https://github.com/openai/CLIP.git` installs unpinned git main, exposing tokenizer and model definitions to floating changes.
   - *Expected*: Pinned commit hash or release tag.

### External Blockers (`EXTERNAL_BLOCKER`)
1. **Missing Checkpoint File & Unresolved Digest**:
   - Checkpoint file `workbench/artifacts/checkpoints/combiner_rn50x4_noft/rn50x4-noft-unresolved.pt` is missing locally. Author Google Drive folder has no published checksum or byte size.
2. **Missing OpenAI CLIP RN50x4 Base Weights**:
   - File `/home/heheboiz/.cache/clip/RN50x4.pt` is missing locally.
3. **Missing Canonical Dataset Symlink**:
   - Path `workbench/third_party/CLIP4Cir/fashionIQ_dataset -> data/fashionIQ_dataset` is not linked.
4. **Unconfigured Python Runtime**:
   - Environment variable `WORKBENCH_CLIP4CIR_PYTHON` is not set; no conda environment provisioned.

### Workbench Defects (`WORKBENCH_DEFECT`)
1. **Unverifiable Paper Benchmark Row (P1)**:
   - *Path*: `workbench/registry/paper_benchmarks/papers.yaml:293-355`
   - *Issue*: Cited TOMM 2023 paper publishes no row for RN50x4 noft. Committed row has all-null metrics and `reference_status: UNVERIFIED`. The model is unconditionally ungradeable by `paper_gate.py` (`METRIC_REFERENCE_UNVERIFIED`).
2. **Misleading Registry Dataset Comment (P2)**:
   - *Path*: `workbench/registry/models.yaml:661-664`
   - *Issue*: Asserts that canonical root is staged at `<checkout>/fashionIQ_dataset` without addressing the `.png` vs `.jpg` format conflict.
3. **Order-Dependent Blocker Classification (P3)**:
   - *Path*: `workbench/scripts/pc_readiness.py:44-71`
   - *Issue*: Matches `"environment"` before `"FashionIQ"`, reporting `BLOCKED_ENVIRONMENT` even when dataset blockers are printed first.

### Rejected Findings (`REJECTED_FINDING`)
1. **"Combiner noft uses inherited CLIP preprocessing"**: REJECTED. Evaluator explicitly instantiates and executes `targetpad_transform(1.25, 288)` defined in `src/data_utils.py:82-96`.
2. **"Reference image is masked in FashionIQ evaluation"**: REJECTED. `distances` is computed and sorted across the full candidate gallery without masking; `reference_names` is not even passed to metric calculation.
3. **"Workbench collapses noft and fullft variants"**: REJECTED. Separate model IDs, separate adapter classes, separate CLI commands, and separate checkpoint representations are strictly enforced.
4. **"Metric parser missing for combiner output"**: REJECTED. `parse_combiner_output` was implemented in `workbench/backend/metrics_extraction.py:309-382` and verified working.

---

## I. EVIDENCE

1. **Commit SHA Verification**:
   ```bash
   git -C workbench/third_party/CLIP4Cir rev-parse HEAD
   # Output: dfed9f748a8a4d05abf613164e06d59b22dbdf13
   ```
2. **PC Readiness Classification**:
   ```bash
   python3 workbench/scripts/pc_readiness.py --model combiner_rn50x4_noft --json
   # Output state: "BLOCKED_ENVIRONMENT"
   # Reason: "execution environment not verified: WORKBENCH_CLIP4CIR_PYTHON=None [UNCONFIGURED/UNCONFIGURED]"
   # Blockers: 9 blockers (dataset .jpg extension, base weights, symlink, env var, checkpoint)
   ```
3. **Non-Masking Protocol Verification**:
   In `src/validate.py:44-55`:
   `distances = 1 - predicted_features @ index_features.T`
   `sorted_indices = torch.argsort(distances, dim=-1).cpu()`
   `sorted_index_names = np.array(index_names)[sorted_indices]`
   No masking step exists between distance computation and rank sorting.
4. **Parser Verification**:
   Executed `extract_aggregate_metrics("combiner_rn50x4_noft", test_stdout, "", 0)` in `workbench/backend/metrics_extraction.py`:
   Returns `MetricExtractionResult(extraction_status='EXTRACTED_VERIFIED', parser_id='clip4cir_stdout_recalls', observed_metrics={'r10': 35.3933, 'r50': 59.03, 'mean': 47.2117})`.

---

## J. VERDICT

| Axis | Verdict | Justification |
| :--- | :--- | :--- |
| **Source Status** | **SOURCE_AUDITED** | Pinned checkout clean and matches registry SHA `dfed9f748a8a4d05abf613164e06d59b22dbdf13`. Evaluation graph fully traced. |
| **Protocol Status** | **PROTOCOL_VERIFIED** | Full validation split gallery (3817/6346/5373); reference remains eligible without masking (Protocol A `fashioniq_original_split`). |
| **Command Status** | **COMMAND_AUDITED** | `CombinerNoftAdapter` matches native `src/validate.py` CLI arguments, correctly omitting `--clip-model-path`. |
| **Checkpoint Status** | **CHECKPOINT_BLOCKED** | Checkpoint weights missing locally; Drive folder has no published hash/size; cited paper reports no row for this variant. |
| **Query Export Status** | **QUERY_EXPORT_IMPLEMENTABLE** | Native script outputs aggregate recalls only; ranking matrix interceptable at line 46 of `validate.py`. |
| **Validation Readiness**| **BLOCKED_EXTERNAL_DEPENDENCY** | Blocked on checkpoint, OpenAI CLIP RN50x4 base weights, Python environment, dataset symlink, and `.jpg` image extension. |
