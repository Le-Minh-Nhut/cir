# CLIP4Cir RN50x4 fullft Final Model Audit

Model registered as `clip4cir_rn50x4_fullft` in CIR Failure Analysis Workbench (`workbench/registry/models.yaml:721-784`).
Evaluated against pinned upstream checkout `workbench/third_party/CLIP4Cir` at commit `dfed9f748a8a4d05abf613164e06d59b22dbdf13`.
Evaluation Protocol: Protocol A (`fashioniq_original_split` — full validation split gallery, reference remains eligible).

---

## A. MODEL IDENTITY

- **Model ID**: `clip4cir_rn50x4_fullft` (`workbench/registry/models.yaml:721`)
- **Method Name**: `CLIP4Cir RN50x4 fullft` (`models.yaml:733`)
- **Paper Title**: *Composed Image Retrieval using Contrastive Learning and Task-oriented CLIP-based Features*
- **Publication & Year**: ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM 2023), Vol. 20, Issue 2, Article 53, pp. 1–21 (published online August 2023 / in issue 2024; arXiv:2308.11485v1 published 2023-08-22).
- **Paper Authors**: Alberto Baldrati, Marco Bertini, Tiberio Uricchio, Alberto Del Bimbo.
- **Paper URL**: `https://arxiv.org/abs/2308.11485` (`models.yaml:738`)
- **Upstream Repository**: `https://github.com/ABaldrati/CLIP4Cir` (`models.yaml:739`)
- **Pinned Commit SHA**: `dfed9f748a8a4d05abf613164e06d59b22dbdf13` (`models.yaml:741`). Verified via `git -C workbench/third_party/CLIP4Cir rev-parse HEAD`. Working tree clean.
- **Checkpoint Variant**: `rn50x4_fullft_unresolved` (`models.yaml:769-784`):
  - Artifact Type: `bundle_placeholder` (`required_files: [combiner_state.pt, clip_state.pt]`).
  - Source URL: `https://drive.google.com/drive/folders/1ny2hhzP8HZBnXhjvDEni8P8G4_inCNTv?usp=sharing` (author Google Drive folder `pretrained_models`).
  - Mapping Status: `UNVERIFIED` / `CHECKPOINT_MAPPING_UNVERIFIED`.
  - Checkpoint Training Noise: 0% (`checkpoint_training_noise_pct: 0`).
  - Evaluation Noise: 0% (`evaluation_noise_pct: 0`).
  - Expected SHA-256 / Size: `null` (not published by authors).
- **Paper Benchmark Row**:
  - Registered reference: `clip4cir_rn50x4_fullft_fiq_tab9` (`workbench/registry/paper_benchmarks/papers.yaml:211-292`).
  - Table 9 (Fashion-IQ validation set, row `Ours | RN-50x4 | Transformer`):
    - Shirt: R@10 = 44.41, R@50 = 65.26
    - Dress: R@10 = 39.46, R@50 = 64.55
    - Toptee: R@10 = 47.48, R@50 = 70.98
    - Average: R@10 = 43.78, R@50 = 66.93
    - Derived Mean: (43.78 + 66.93) / 2 = 55.355 -> 55.36 (`WORKBENCH_DERIVED`, round-half-up).
- **Explicit Distinction from `combiner_rn50x4_noft`**:
  1. *Methodology*: `fullft` is the two-stage proposed method (§4.1): Stage 1 task-oriented fine-tuning updating BOTH CLIP visual and text encoders (IFT=yes, TFT=yes); Stage 2 Combiner network trained on top of fine-tuned encoders. `noft` skips Stage 1 (IFT=no, TFT=no), training Combiner directly on out-of-the-box frozen OpenAI CLIP features.
  2. *Checkpoints*: `fullft` requires a PAIRED checkpoint bundle (`combiner_state.pt` + `clip_state.pt`) plus base OpenAI `RN50x4.pt`. `noft` requires only a single checkpoint (`fiq_comb_RN50x4_noft.pt`) plus base OpenAI `RN50x4.pt`.
  3. *CLI Invocation*: `fullft` passes `--clip-model-path <clip_state.pt>` AND `--combiner-path <combiner_state.pt>`. `noft` passes `--combiner-path <combiner.pt>` and OMITS `--clip-model-path` (leaving base CLIP encoders untouched).
  4. *Paper Reporting*: `fullft` is the paper's headline Fashion-IQ result printed in Table 9 (Ours RN-50x4). `noft` has NO Fashion-IQ row in TOMM 2023 (released for CVPR 2022 demo; registered in `papers.yaml:293` as `clip4cir_rn50x4_noft_fiq_absent_in_paper`).

---

## B. SOURCE COVERAGE

### Tree Inventory
Pinned tree `workbench/third_party/CLIP4Cir` contains 11 git-tracked files:
- Root: `.gitignore`, `LICENSE`, `README.md`
- `src/`:
  1. `validate.py` (360 lines): Official evaluation entrypoint for FashionIQ and CIRR.
  2. `combiner.py` (65 lines): Combiner network architecture definition (`Combiner`).
  3. `data_utils.py` (277 lines): Dataset loaders (`FashionIQDataset`, `CIRRDataset`), padding and transforms (`TargetPad`, `SquarePad`, `targetpad_transform`, `squarepad_transform`).
  4. `utils.py` (131 lines): Feature extraction (`extract_index_features`), element-wise sum baseline, device setup, caption randomization, checkpoint saving.
  5. `combiner_train.py` (289 lines): Stage-2 Combiner training script.
  6. `clip_fine_tune.py` (234 lines): Stage-1 CLIP task-oriented fine-tuning script.
  7. `cirr_test_submission.py` (152 lines): CIRR evaluation server submission generator.
- `images/`: 4 architecture / workflow diagrams (`cir-overview.png`, `clip-fine-tuning.png`, `combiner-training.png`, `Combiner-architecture.png`).

### Reachable Evaluation Graph
Execution path for FashionIQ validation evaluation (`src/validate.py`):
1. Entrypoint: `main()` (`src/validate.py:263-360`).
2. Base Model Load: `clip.load(args.clip_model_name, device=device, jit=False)` (`src/validate.py:279`) loads base OpenAI CLIP RN50x4.
3. Fine-tuned CLIP Overlay: `src/validate.py:283-287` loads `saved_state_dict = torch.load(args.clip_model_path)` and executes `clip_model.load_state_dict(saved_state_dict["CLIP"])`.
4. Preprocessing Setup: `targetpad_transform(args.target_ratio, input_dim)` (`src/validate.py:291`, `src/data_utils.py:82-96`).
5. Combiner Load: `src/validate.py:304-309` instantiates `Combiner(feature_dim, args.projection_dim, args.hidden_dim)`, loads `state_dict = torch.load(args.combiner_path)`, executes `combiner.load_state_dict(state_dict["Combiner"])`, and calls `combiner.eval()`.
6. Dataset & Feature Extraction: `fashioniq_val_retrieval()` (`src/validate.py:113-132`):
   - Instantiates `classic_val_dataset = FashionIQDataset('val', [dress_type], 'classic', preprocess)`.
   - Calls `extract_index_features(classic_val_dataset, clip_model)` (`src/utils.py:21-44`).
   - Instantiates `relative_val_dataset = FashionIQDataset('val', [dress_type], 'relative', preprocess)`.
7. Query Prediction: `generate_fiq_val_predictions()` (`src/validate.py:60-110`):
   - Feeds concatenated relative captions to `clip.tokenize()` and `clip_model.encode_text()`.
   - Extracts reference image features from cached index features via `itemgetter(*reference_names)(name_to_feat)`.
   - Fuses multimodal features via `combining_function(reference_image_features, text_features)`.
8. Ranking & Metrics: `compute_fiq_val_metrics()` (`src/validate.py:21-57`):
   - Computes cosine distance `distances = 1 - predicted_features @ index_features.T`.
   - Sorts `sorted_indices = torch.argsort(distances, dim=-1)`.
   - Computes indicator `labels` and calculates Recall@10 and Recall@50.
9. Category Loop & Aggregation: `src/validate.py:325-354` loops over `shirt`, `dress`, `toptee`, and prints category recalls plus `average recall10` and `average recall50`.

### Inspected vs Excluded
- **Inspected**:
  - `src/validate.py`: Core evaluation loop, CLI args, dual checkpoint loading, metric computation, aggregation.
  - `src/combiner.py`: Architecture, layers, forward pass, dynamic scalar $\lambda$, projection layers.
  - `src/data_utils.py`: `TargetPad`, dataset layout paths, image opening, triplet parsing.
  - `src/utils.py`: Index feature caching, device selection, model save format (`model.__class__.__name__`).
  - `README.md`: Author installation instructions, validation commands, checkpoint links.
- **Excluded**:
  - `src/clip_fine_tune.py`: Stage 1 training script only.
  - `src/combiner_train.py`: Stage 2 training script only.
  - `src/cirr_test_submission.py`: CIRR test server submission formatting only.
  - `images/`: Documentation figures.

### README vs Executable Code
1. *CLI Argument Names*: `README.md:258-269` accurately documents `--dataset`, `--combining-function combiner`, `--combiner-path`, `--projection-dim 2560`, `--hidden-dim 5120`, `--clip-model-name RN50x4`, `--clip-model-path`, `--target-ratio 1.25`, `--transform targetpad`. CLI flags match `src/validate.py:264-277` verbatim.
2. *Hardcoded Dataset Path*: `README.md:144-163` illustrates dataset layout under `project_base_path`. Executable code in `src/data_utils.py:11` fixes `base_path = Path(__file__).absolute().parents[1].absolute()` (`workbench/third_party/CLIP4Cir`), requiring `<checkout>/fashionIQ_dataset`. No CLI flag allows overriding this root.
3. *Image File Extension*: `README.md:153-156` shows images as `.jpg` (`B00006M009.jpg`). Executable code in `src/data_utils.py:158, 161, 170, 176` hardcodes `f"{name}.jpg"`. Local canonical FashionIQ dataset stores images as `.png`.
4. *Unpinned Dependencies*: `README.md:118` specifies `pip install git+https://github.com/openai/CLIP.git` (unpinned master branch).

---

## C. NATIVE EVALUATION

### 1. Gallery Universe and Ordering (Proof of Full Val Split)
In `src/validate.py:126-129`:
```python
    # Define the validation datasets and extract the index features
    classic_val_dataset = FashionIQDataset('val', [dress_type], 'classic', preprocess)
    index_features, index_names = extract_index_features(classic_val_dataset, clip_model)
    relative_val_dataset = FashionIQDataset('val', [dress_type], 'relative', preprocess)
```
In `src/data_utils.py:144-147, 174-178`:
```python
        # get the image names
        self.image_names: list = []
        for dress_type in dress_types:
            with open(base_path / 'fashionIQ_dataset' / 'image_splits' / f'split.{dress_type}.{split}.json') as f:
                self.image_names.extend(json.load(f))
...
            elif self.mode == 'classic':
                image_name = self.image_names[index]
                image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{image_name}.jpg"
                image = self.preprocess(PIL.Image.open(image_path))
                return image_name, image
```
- In `classic` mode, `classic_val_dataset` loads image names sequentially from `split.{dress_type}.val.json`.
- `index_names` matches `self.image_names` in file order without deduplication or sorting (`src/utils.py:38-44`).
- Local validation split sizes: `dress`: 3,817; `shirt`: 6,346; `toptee`: 5,373 (total 15,536).
- The gallery is strictly the full validation split (Protocol A), NOT the pair-union of references and targets (Protocol C).

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
```
- Contrast with CIRR path in the same file (`src/validate.py:162-166`):
```python
    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(target_names), -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)
```
- In FashionIQ, reference images are **NOT** masked with $-\infty$ or $+10^{10}$, and are **NOT** removed from `sorted_index_names`.
- All 6,016 reference images in FashionIQ validation are present in their category gallery (`index_names`), and 0 reference images equal their target image (`target_names`).
- A reference image ranking within the top K occupies a rank slot and can push the target image out of the top K.
- Proof of Protocol A (`fashioniq_original_split`): reference REMAINS ELIGIBLE.

### 3. Evaluation Transforms vs noft Variant
- Both `clip4cir_rn50x4_fullft` and `combiner_rn50x4_noft` use identical preprocessing parameters: `--transform targetpad --target-ratio 1.25`.
- Implemented in `src/data_utils.py:40-63, 82-96`:
  - `TargetPad`: pads image aspect ratio with constant 0 (black padding) if `actual_ratio >= target_ratio (1.25)`.
  - `Resize(288, interpolation=PIL.Image.BICUBIC)`.
  - `CenterCrop(288)`.
  - `ToTensor()`.
  - `Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))`.
- Resolution: 288x288 (RN50x4 visual input resolution).

### 4. Similarity Scoring
- In `generate_fiq_val_predictions()` (`src/validate.py:105-107`):
  - `batch_predicted_features = combining_function(reference_image_features, text_features)`
  - `predicted_features = torch.vstack((predicted_features, F.normalize(batch_predicted_features, dim=-1)))`
- In `Combiner.combine_features()` (`src/combiner.py:50-65`):
  ```python
  text_projected_features = self.dropout1(F.relu(self.text_projection_layer(text_features)))
  image_projected_features = self.dropout2(F.relu(self.image_projection_layer(image_features)))
  raw_combined_features = torch.cat((text_projected_features, image_projected_features), -1)
  combined_features = self.dropout3(F.relu(self.combiner_layer(raw_combined_features)))
  dynamic_scalar = self.dynamic_scalar(raw_combined_features)
  output = self.output_layer(combined_features) + dynamic_scalar * text_features + (1 - dynamic_scalar) * image_features
  return F.normalize(output, dim=-1)
  ```
- Multimodal query embedding is L2-normalized.
- Gallery embedding is L2-normalized: `index_features = F.normalize(index_features, dim=-1).float()` (`src/validate.py:41`).
- Similarity: matrix dot product `predicted_features @ index_features.T`.
- Distance: `distances = 1 - predicted_features @ index_features.T`.

### 5. Sorting, Top-K, and Recall Definition
- Sorting: `sorted_indices = torch.argsort(distances, dim=-1).cpu()` sorts by ascending cosine distance (= descending similarity).
- Alignment: `sorted_index_names = np.array(index_names)[sorted_indices]`.
- Ground truth binary matrix: `labels = torch.tensor(sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))`.
- Assertion: `assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())` proves every query has exactly one positive target image in the gallery.
- Recall@K formula:
  - `recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100`
  - `recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100`
  - Denominator is strictly `len(labels) = num_queries` (queries without target in top K count as 0).

### 6. Category Handling & Macro Aggregation
In `src/validate.py:325-354`:
```python
    shirt_recallat10, shirt_recallat50 = fashioniq_val_retrieval('shirt', combining_function, clip_model, preprocess)
    dress_recallat10, dress_recallat50 = fashioniq_val_retrieval('dress', combining_function, clip_model, preprocess)
    toptee_recallat10, toptee_recallat50 = fashioniq_val_retrieval('toptee', combining_function, clip_model, preprocess)
...
    print(f"average recall10 = {mean(average_recall10_list)}")
    print(f"average recall50 = {mean(average_recall50_list)}")
```
- Evaluates `shirt`, `dress`, `toptee` sequentially.
- Category recall counts:
  - `dress`: 2,017 queries against 3,817 gallery images.
  - `shirt`: 2,038 queries against 6,346 gallery images.
  - `toptee`: 1,961 queries against 5,373 gallery images.
  - Total: 6,016 queries against 15,536 gallery images.
- Aggregation is the unweighted arithmetic mean over the three categories:
  - `average recall10 = (shirt_r10 + dress_r10 + toptee_r10) / 3`
  - `average recall50 = (shirt_r50 + dress_r50 + toptee_r50) / 3`
- Overall mean: `(average recall10 + average recall50) / 2 = 55.36` (`WORKBENCH_DERIVED`).

---

## D. PREPARATION

### 1. Dataset Layout & Paths
- Registry declaration: `fashioniq_layout: fashioniq_standard` (`models.yaml:744`), `preparation_id: clip4cir_fashioniq` (`models.yaml:746`).
- Contract registered in `workbench/registry/preparation_contracts.yaml:140-169`:
  - Raw inputs: `captions/cap.{dress,shirt,toptee}.val.json`, `image_splits/split.{dress,shirt,toptee}.val.json`, `images/`.
  - Generated artifacts: None.
  - External assets: OpenAI CLIP RN50x4 base weights (`~/.cache/clip/RN50x4.pt`, SHA-256 `7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd`).
  - Automation policy: `MANUAL_REQUIRED`.
  - Deterministic status: `DETERMINISTIC_AT_EVALUATION`.
- Evaluator hardcoded path: `src/data_utils.py:11` fixes `base_path` to `<checkout>`. Requires symlink:
  `workbench/third_party/CLIP4Cir/fashionIQ_dataset -> data/fashionIQ_dataset`.
  Currently ABSENT.

### 2. Image Extension Mismatch (`.jpg` vs `.png`)
- Pinned evaluator code in `src/data_utils.py:158, 161, 170, 176` hardcodes `f"{name}.jpg"`.
- Canonical FashionIQ dataset at `data/fashionIQ_dataset/images` contains 74,381 `.png` files and 0 `.jpg` files.
- `prepare_model.py` and `preflight.py` report 3817/3817 dress, 6346/6346 shirt, and 5373/5373 toptee validation images lack `.jpg`.
- Evaluator fails on first image load unless `.jpg` symlinks or format conversions are staged.

### 3. Base Backbone Asset (OpenAI CLIP RN50x4)
- Native dependency: `clip.load("RN50x4", device=device, jit=False)` (`src/validate.py:279`).
- Upstream `openai/CLIP` downloads to `download_root or os.path.expanduser("~/.cache/clip")`.
- Expected file: `~/.cache/clip/RN50x4.pt` (421,854,225 bytes, SHA-256 `7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd`).
- Currently MISSING on local machine.

### 4. Working Directory & Stochasticity
- Working directory: `source_cwd` returns `workbench/third_party/CLIP4Cir`.
- Stochasticity: 0% (`evaluation_noise_pct: 0`).
  - `DataLoader(..., shuffle=False)`.
  - `torch.no_grad()`.
  - `combiner.eval()` disables all dropout layers (`dropout1`, `dropout2`, `dropout3`, and dynamic scalar dropout).
  - Caption concatenation deterministic: `strip('.?, ').capitalize() + ' and ' + strip('.?, ')` (`src/validate.py:88-92`).
  - No random seeds required for evaluation.

---

## E. THE PAIR & CHECKPOINT

### 1. The Critical Two-State Requirement
`clip4cir_rn50x4_fullft` requires a PAIRED model state loaded across two distinct checkpoint files:
1. **Combiner State Checkpoint**:
   - Class: `Combiner` (`src/combiner.py:6`).
   - CLI Argument: `--combiner-path <path>`.
   - Loader:
     ```python
     combiner = Combiner(feature_dim, args.projection_dim, args.hidden_dim).to(device, non_blocking=True)
     state_dict = torch.load(args.combiner_path, map_location=device)
     combiner.load_state_dict(state_dict["Combiner"])
     combiner.eval()
     ```
   - Saved under key: `"Combiner"` (`src/utils.py:127-131` saves `model_to_save.__class__.__name__`).
2. **Fine-tuned CLIP State Checkpoint**:
   - Class: `clip.model.CLIP`.
   - CLI Argument: `--clip-model-path <path>`.
   - Loader:
     ```python
     if args.clip_model_path:
         saved_state_dict = torch.load(args.clip_model_path, map_location=device)
         clip_model.load_state_dict(saved_state_dict["CLIP"])
     ```
   - Saved under key: `"CLIP"` (`src/clip_fine_tune.py:226-228`).
3. **Base Backbone (Third Asset)**:
   - Base OpenAI RN50x4 model must be loaded into memory first via `clip.load()`; fine-tuned CLIP weights are overlaid on top of this structure.

### 2. Failure Behaviors on Missing Member
- If `args.clip_model_path` is omitted: Evaluator does not crash, but silently evaluates Combiner against the **frozen base CLIP** (becoming `combiner_rn50x4_noft`), producing incorrect results.
- If `args.clip_model_path` points to nonexistent file: `torch.load` raises `FileNotFoundError`.
- If `args.combiner_path` is omitted / nonexistent: `torch.load` raises `FileNotFoundError` or `TypeError`.
- If either file lacks the expected root key (`"CLIP"` or `"Combiner"`): `KeyError` is raised.

### 3. Registry Bundle Definition & Modeling Deficiencies
- Registry declaration in `models.yaml:768-784`:
  ```yaml
  checkpoint_variants:
    - checkpoint_id: rn50x4_fullft_unresolved
      filename: rn50x4-fullft-unresolved.pt
      artifact_type: bundle_placeholder
      required_files: [combiner_state.pt, clip_state.pt]
      download_url: null
      source_url: https://drive.google.com/drive/folders/1ny2hhzP8HZBnXhjvDEni8P8G4_inCNTv?usp=sharing
      source_type: author_google_drive
      checkpoint_training_noise_pct: 0
      evaluation_noise_pct: 0
      expected_size_bytes: null
      expected_sha256: null
      status: CHECKPOINT_MAPPING_UNVERIFIED
      checkpoint_mapping_status: UNVERIFIED
  ```
- **Top-Level `checkpoint_bundles` Missing**: `models.yaml` does not define `checkpoint_bundles` for `clip4cir_rn50x4_fullft` (only `limn` defines it). Instead, it encodes the pair as a single variant with `artifact_type: bundle_placeholder`.
- **Integrity Check (`checkpoint_missing_paths`)**:
  - `workbench/backend/registry.py:371-375` recognizes `bundle_placeholder` and checks `_regular_directory(path)`.
  - It iterates over `required_files: [combiner_state.pt, clip_state.pt]`.
  - Verified: An empty bundle directory returns both missing files; a directory containing only `combiner_state.pt` returns `[PosixPath('.../clip_state.pt')]`.
- **Selection Hole in `checkpoint_selection.py`**:
  - `select_checkpoints` calls `_obtainable()` -> `_present(variant, checkpoint_root, model_id)`.
  - In `checkpoint_selection.py:90-96`, `_present` checks `checkpoint_path(model_id, variant, checkpoint_root).exists()`.
  - For a `bundle_placeholder`, `checkpoint_path` returns the directory path `.../rn50x4-fullft-unresolved.pt`.
  - If the directory exists on disk, `_present()` returns `True` even if `clip_state.pt` or `combiner_state.pt` is missing from the directory.
- **Manifest Tracking Gap in `evaluate_models.py`**:
  - In `workbench/scripts/evaluate_models.py:394`, `elif artifact_type in ("run_directory", "bundle_directory"):` omits `bundle_placeholder`.
  - As a result, `directory_members` and `bundle_manifest_digest` are NOT constructed for `clip4cir_rn50x4_fullft`.
  - In `evaluate_models.py:725-728`, `multi_member` evaluates to `False`, bypassing member change-detection.
- **Evidence Chain Recording**:
  - Because `bundle_manifest_digest` is `None`, the run plan does not pin individual member digests.
  - Checkpoint mapping status is `UNVERIFIED` with no author download URLs or hashes.

---

## F. WORKBENCH INTEGRATION

### 1. Adapter Implementation
In `workbench/backend/adapters/models.py:97-112`:
```python
class CLIP4CirFullftAdapter(OfficialScriptAdapter):
    model_id = "clip4cir_rn50x4_fullft"
    script = "src/validate.py"

    def command(self, source, checkpoint, request: EvalRequest) -> list[str]:
        # A bundle_placeholder path IS the bundle directory (registry.checkpoint_missing_paths
        # resolves members under it); the previous `checkpoint.parent` disagreed with that and
        # pointed the evaluator at sibling files that the integrity check never validates.
        combiner_path = checkpoint / "combiner_state.pt"
        clip_path = checkpoint / "clip_state.pt"
        return [_runtime.python_executable(self.model), str(source / self.script), "--dataset", "fashionIQ",
                "--combining-function", "combiner", "--combiner-path", str(combiner_path),
                "--clip-model-path", str(clip_path),
                "--projection-dim", "2560", "--hidden-dim", "5120",
                "--clip-model-name", "RN50x4", "--target-ratio", "1.25", "--transform", "targetpad"]
```
- Passes both `--combiner-path` and `--clip-model-path`.
- Resolves members under `checkpoint` directory (`rn50x4-fullft-unresolved.pt/`). Coherent with `checkpoint_missing_paths`.
- Flags and parameters (`2560`, `5120`, `RN50x4`, `1.25`, `targetpad`) match source defaults and README.

### 2. Metrics Parser
- Function: `parse_combiner_output(stdout_text, paper_metrics)` in `workbench/backend/metrics_extraction.py:309-382`.
- Dispatched for `clip4cir_rn50x4_fullft` at `metrics_extraction.py:406-407`.
- Extracts `shirt_recallat10/50`, `dress_recallat10/50`, `toptee_recallat10/50`, `average recall10/50`.
- Validates internal consistency: asserts `abs(avg10 - sum(cats)/3) <= 0.05` and `abs(avg50 - sum(cats)/3) <= 0.05`.
- Emits `MetricExtractionResult` with `parser_id="clip4cir_stdout_recalls"`, status `EXTRACTED_VERIFIED`.

### 3. Preflight & Readiness
- `preflight_blockers` in `workbench/backend/preflight.py` enforces:
  - Canonical dataset link `<checkout>/fashionIQ_dataset`.
  - Image extensions `.jpg` (checks presence of `.jpg` for all validation images).
  - External assets: base weights at `~/.cache/clip/RN50x4.pt`.
  - Checkpoint presence: checks `checkpoint_missing_paths`.
  - Environment: checks `WORKBENCH_CLIP4CIR_PYTHON`.
- `workbench/scripts/pc_readiness.py`:
  - Reports 9 blockers: 3 image extension blockers, base weights missing, dataset link missing, runtime asset missing, environment unverified, checkpoint missing, checkpoint mapping unresolved.
  - Needle-ordering flaw: Classifies as `BLOCKED_ENVIRONMENT` because `("environment", BLOCKED_ENVIRONMENT)` precedes `("FashionIQ", BLOCKED_DATASET)` in `BLOCKER_KINDS`.

---

## G. OBSERVATION CAPABILITY

### 1. Score and Ranking Production Point
In `workbench/third_party/CLIP4Cir/src/validate.py:43-47` (`compute_fiq_val_metrics`):
```python
    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]
```
- `distances`: Shape `(num_queries, num_gallery)`.
- `sorted_indices`: Shape `(num_queries, num_gallery)` sorted in ascending distance order.
- `sorted_index_names`: Shape `(num_queries, num_gallery)` containing exact gallery image IDs in descending similarity rank.
- Query order aligns with `relative_val_dataset.triplets`:
  - Reference image: `triplets[i]['candidate']`
  - Target image: `triplets[i]['target']`
  - Relative captions: `triplets[i]['captions']`

### 2. Observation Hook Feasibility
- **Verdict**: `QUERY_EXPORT_IMPLEMENTABLE`.
- Feasibility: An observer hook can intercept `compute_fiq_val_metrics` right after line 46 to export Top-K ranked image names (`sorted_index_names[:, :K]`) and distances/similarity scores alongside query triplet metadata.
- Native evaluator currently returns only scalar recalls `recall_at10, recall_at50` and prints text summaries. It has no command-line flag to export rankings.

---

## H. DEFECTS

### Confirmed Bugs (`CONFIRMED_BUG`)
1. **C1 (P0) — Hardcoded `.jpg` Image Extension**:
   - *File*: `workbench/third_party/CLIP4Cir/src/data_utils.py:158, 161, 170, 176`
   - *Issue*: Code opens `f"{name}.jpg"`. Local canonical FashionIQ dataset contains 74,381 `.png` files and 0 `.jpg` files.
   - *Impact*: Evaluator crashes immediately with `FileNotFoundError` on first image access.
2. **C2 (P0) — Hardcoded Dataset Root Path in Evaluator**:
   - *File*: `workbench/third_party/CLIP4Cir/src/data_utils.py:11, 140, 146, 176`
   - *Issue*: Evaluator fixes `base_path` to `<checkout>` root and unconditionally opens `base_path / 'fashionIQ_dataset'`. No CLI flag exists to specify a dataset root.
   - *Impact*: Evaluator crashes unless canonical dataset is symlinked into the git checkout.
3. **C3 (P2) — `bundle_placeholder` Invisible to Plan Builder Manifest in `evaluate_models.py`**:
   - *File*: `workbench/scripts/evaluate_models.py:394, 725`
   - *Issue*: Line 394 checks `artifact_type in ("run_directory", "bundle_directory")` and line 725 checks `artifact_type in ("run_directory", "bundle_directory", "category_specific_whole_models")`. Both omit `bundle_placeholder`.
   - *Impact*: `directory_members` and `bundle_manifest_digest` are not generated for `clip4cir_rn50x4_fullft`. Member change-detection fails open (`multi_member = False`).
4. **C4 (P2) — Checkpoint Selection `_present()` Ignores Bundle Member Presence**:
   - *File*: `workbench/backend/checkpoint_selection.py:90-96`
   - *Issue*: `_present()` checks only `checkpoint_path().exists()`. For `bundle_placeholder`, `checkpoint_path` is the directory.
   - *Impact*: If the directory exists, `_present()` returns `True` even if `clip_state.pt` or `combiner_state.pt` is missing.
5. **C5 (P3) — Inconsistent Status Fields in `models.yaml`**:
   - *File*: `workbench/registry/models.yaml:728, 757`
   - *Issue*: `replay_status: BLOCKED_CHECKPOINT_MAPPING` contradicts `reproduction_status: BLOCKED_CHECKPOINT_BUNDLE`.

### External Blockers (`EXTERNAL_BLOCKER`)
1. **E1 (P0) — Checkpoint Pair Unmapped and Unstaged**:
   - Pinned author Google Drive folder contains unmapped `.pt` files without hashes or pairing documentation.
   - `models.yaml:773, 781-782` records `download_url: null`, `expected_sha256: null`, `checkpoint_mapping_status: UNVERIFIED`.
   - No local files staged at `workbench/artifacts/checkpoints/clip4cir_rn50x4_fullft/rn50x4-fullft-unresolved.pt/`.
2. **E2 (P0) — Missing OpenAI CLIP RN50x4 Base Weights**:
   - File `~/.cache/clip/RN50x4.pt` (SHA-256 `7e526bd1...`) missing.
3. **E3 (P1) — Missing Execution Environment**:
   - `WORKBENCH_CLIP4CIR_PYTHON` is not set; Python 3.8 environment unconfigured.
4. **E4 (P1) — Canonical Dataset Symlink Missing**:
   - Link `workbench/third_party/CLIP4Cir/fashionIQ_dataset -> data/fashionIQ_dataset` not staged.

### Workbench Defects (`WORKBENCH_DEFECT`)
1. **W1 (P2) — Needle-Ordering Classification Distortion in `pc_readiness.py`**:
   - *File*: `workbench/scripts/pc_readiness.py:47-67`
   - *Issue*: `BLOCKER_KINDS` checks `("environment", BLOCKED_ENVIRONMENT)` before `("FashionIQ", BLOCKED_DATASET)`.
   - *Impact*: Masks missing dataset images and links behind unconfigured environment.

### Rejected Findings (`REJECTED_FINDING`)
1. **R1 — "Reference image is excluded from FashionIQ candidate gallery"**: REFUTED. `src/validate.py:40-55` performs no reference removal or score masking. Reference remains fully eligible in the candidate gallery (Protocol A `fashioniq_original_split`).
2. **R2 — "A single checkpoint contains both CLIP and Combiner weights"**: REFUTED. Evaluator loads `args.clip_model_path` under key `"CLIP"` (`src/validate.py:285-286`) and `args.combiner_path` under key `"Combiner"` (`src/validate.py:306-307`). Both checkpoint files are strictly required.
3. **R3 — "Metrics parser is missing for clip4cir"**: REFUTED. `parse_combiner_output` in `workbench/backend/metrics_extraction.py:309-382` parses `src/validate.py` stdout and is dispatched for `clip4cir_rn50x4_fullft` at line 406.

---

## I. EVIDENCE

1. **Commit Pin Verification**:
   ```bash
   git -C workbench/third_party/CLIP4Cir rev-parse HEAD
   # Output: dfed9f748a8a4d05abf613164e06d59b22dbdf13
   ```
   Matches `workbench/registry/models.yaml:741`. Working tree clean.

2. **Dual Checkpoint Loader Verification**:
   - CLIP overlay loader (`src/validate.py:283-287`):
     ```python
     if args.clip_model_path:
         print('Trying to load the CLIP model')
         saved_state_dict = torch.load(args.clip_model_path, map_location=device)
         clip_model.load_state_dict(saved_state_dict["CLIP"])
         print('CLIP model loaded successfully')
     ```
   - Combiner loader (`src/validate.py:304-309`):
     ```python
     elif args.combining_function.lower() == 'combiner':
         combiner = Combiner(feature_dim, args.projection_dim, args.hidden_dim).to(device, non_blocking=True)
         state_dict = torch.load(args.combiner_path, map_location=device)
         combiner.load_state_dict(state_dict["Combiner"])
         combiner.eval()
         combining_function = combiner.combine_features
     ```

3. **Protocol A Verification (Non-Masking Proof)**:
   - FashionIQ path (`src/validate.py:43-51`):
     ```python
     distances = 1 - predicted_features @ index_features.T
     sorted_indices = torch.argsort(distances, dim=-1).cpu()
     sorted_index_names = np.array(index_names)[sorted_indices]
     labels = torch.tensor(
         sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))
     assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
     ```
   - No reference masking applied; all 6,016 references eligible in gallery.

4. **Bundle Member Validation Behavior (`checkpoint_missing_paths`)**:
   - Execution against simulated bundle:
     - Absent dir: returns `[PosixPath('.../bundle')]`
     - Empty dir: returns `[PosixPath('.../combiner_state.pt'), PosixPath('.../clip_state.pt')]`
     - Partial dir (`combiner_state.pt` only): returns `[PosixPath('.../clip_state.pt')]`
     - Full dir (both present): returns `[]`

5. **`pc_readiness.py` Execution**:
   - Command: `python3 workbench/scripts/pc_readiness.py --model clip4cir_rn50x4_fullft --json`
   - State: `BLOCKED_ENVIRONMENT` (matches `environment` needle at line 54 before dataset needles).
   - Blockers reported: 9 total (image extensions, base weights, dataset link, environment, checkpoint).

6. **`prepare_model.py` Execution**:
   - Command: `python3 workbench/scripts/prepare_model.py --model clip4cir_rn50x4_fullft --json`
   - Result: `fully_prepared: false`, exit code 1. Reports missing `.jpg` image extensions, missing base weights, missing dataset link.

---

## J. VERDICT

| Axis | Classification | Justification |
| :--- | :--- | :--- |
| **Source Status** | **SOURCE_AUDITED** | Pinned checkout matches commit `dfed9f7...`; clean working tree. Audited end-to-end statically. Requires runtime dataset symlink and `.jpg` image extension handling. |
| **Protocol Status** | **PROTOCOL_VERIFIED** | Full validation split gallery; reference image remains eligible in candidate pool (Protocol A `fashioniq_original_split`). |
| **Command Status** | **COMMAND_AUDITED** | `CLIP4CirFullftAdapter.command` matches native `src/validate.py` CLI interface and passes both `--combiner-path` and `--clip-model-path`. |
| **Checkpoint Status** | **CHECKPOINT_BLOCKED** | Checkpoint mapping unverified in author Google Drive; no hashes or sizes; local bundle weights unstaged. |
| **Query Export Status** | **QUERY_EXPORT_IMPLEMENTABLE** | Native evaluation emits aggregate recalls only; ranking tensor `sorted_index_names` at line 46 of `compute_fiq_val_metrics` is directly interceptable. |
| **Validation Readiness** | **BLOCKED_EXTERNAL_DEPENDENCY** | Blocked on OpenAI CLIP RN50x4 base weights, checkpoint bundle pairing/download, dataset `.jpg` extension compatibility, dataset symlink, and Python 3.8 environment. |
