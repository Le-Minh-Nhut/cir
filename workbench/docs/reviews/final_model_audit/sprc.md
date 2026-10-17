# SPRC Final Model Audit

Model registered as `sprc` in CIR Failure Analysis Workbench.
Evaluated against pinned upstream checkout `workbench/third_party/SPRC` at commit `2935a5397732260d1db6fa577e5926f963e36f0f`.
Evaluation Protocol: Protocol A (`fashioniq_original_split` — full validation split gallery, reference remains eligible).

---

## A. MODEL IDENTITY

- **Model ID**: `sprc` (`workbench/registry/models.yaml:843`)
- **Method Name**: `SPRC` (`workbench/registry/models.yaml:853`)
- **Paper Title**: *Sentence-level Prompts Benefit Composed Image Retrieval*
- **Publication & Year**: ICLR 2024 Spotlight (camera-ready PDF sha256 `2c250ddb9d0251e21fdad6a2a9400f1d0f688e8427be3f536e9a8404f5409a3b`; arXiv:2310.05473).
- **Paper URL**: `https://arxiv.org/abs/2310.05473` (registered in `workbench/registry/models.yaml:858`).
- **Upstream Repository**: `https://github.com/chunmeifeng/SPRC` (registered in `workbench/registry/models.yaml:859`).
- **Pinned Commit SHA**: `2935a5397732260d1db6fa577e5926f963e36f0f`
  - Verified: `git -C workbench/third_party/SPRC rev-parse HEAD` returns `2935a5397732260d1db6fa577e5926f963e36f0f` (matches `models.yaml:861`).
- **Checkpoint Variant**: `sprc_fiq_unresolved` (`models.yaml:889`)
  - Filename: `sprc_fiq.pt` (`models.yaml:890`)
  - Author Source URL: `https://1drv.ms/u/s!Aj0q22vyiZbabnUya4mnufIBtYI?e=n4ZVKj` (OneDrive, returns HTTP 403)
  - Secondary Sources: BaiduCloud (`pan.baidu.com/s/18196NRV0Cdbn5uPc3LIgwg`, pwd `t1at`), Google Drive for ViT-L variant (`sprc_fiq_vitl.pt`)
  - Local Path: `workbench/artifacts/checkpoints/sprc/sprc_fiq.pt` (MISSING locally)
  - Mapping Status: `CHECKPOINT_MAPPING_UNVERIFIED` / `UNVERIFIED`
- **Paper Benchmark Row**:
  - Registered reference: `sprc_fiq_val_ours_vitg` (`workbench/registry/paper_benchmarks/papers.yaml:1775-1837`).
  - Table 1 (`Ours (SPRC)` on Fashion-IQ validation set, p. 6):
    - Dress: R@10 = 49.18, R@50 = 72.43
    - Shirt: R@10 = 55.64, R@50 = 73.89
    - Toptee: R@10 = 59.35, R@50 = 78.58
    - Printed Average Block: R@10 = 54.92, R@50 = 74.97, Overall Avg. = 64.85
  - Provenance: All three macro numbers verified as `PAPER_REPORTED` from ICLR 2024 camera-ready PDF line 407 and rendered page 6 image.

---

## B. SOURCE COVERAGE

### 1. Pinned Source Inventory (`workbench/third_party/SPRC/`)
The pinned checkout contains 515 tracked files in git, including a vendored LAVIS framework under `src/lavis/`.
Reachable evaluation dependency graph:
1. Entrypoint: `src/blip_validate.py` (official CLI entrypoint for validation).
2. Evaluation loop: `src/validate_blip.py` (`compute_fiq_val_metrics`, `generate_fiq_val_predictions`).
3. Dataset loader: `src/data_utils.py` (`FashionIQDataset`, `base_path`, `targetpad_transform`, `SquarePad`, `collate_fn`).
4. Feature extraction: `src/utils.py` (`extract_index_blip_features`, `collate_fn`, `device`).
5. Architecture registry: `src/lavis/models/__init__.py` (`load_model_and_preprocess`, `load_model`).
6. Model implementations:
   - Primary CIR model: `src/lavis/models/blip2_models/blip2_qformer_cir_align_prompt.py` (`Blip2QformerCirAlignPrompt`, registered as `blip2_cir_align_prompt`).
   - Rerank CIR model: `src/lavis/models/blip2_models/blip2_qformer_cir_rerank.py` (`Blip2QformerCirRerank`, registered as `blip2_cir_rerank`).
7. Base architecture: `src/lavis/models/blip2_models/blip2.py` (`Blip2Base`), `src/lavis/models/blip2_models/Qformer.py`.
8. Visual backbone: `src/lavis/models/eva_vit.py` (`create_eva_vit_g`).
9. Configuration: `src/lavis/configs/models/blip2/blip2_pretrain.yaml`.
10. Processors: `src/lavis/processors/blip_processors.py` (`BlipCaptionProcessor`).

### 2. Inspected vs Excluded
- **Inspected**:
  - `src/blip_validate.py`: validation entrypoint, CLI arguments, checkpoint loading, category loop, macro aggregation, JSON serialization.
  - `src/validate_blip.py`: prediction generation, TargetPad preprocessing, similarity calculation, ranking, Recall@10/Recall@50 computation, visualization hooks.
  - `src/validate_blip_rerank.py`: experimental two-stage re-ranking evaluator; inspected for `SPRC2` claims and import integrity.
  - `src/data_utils.py`: dataset initialization, split file loading, image loading, extension formatting, hardcoded repository paths.
  - `src/utils.py`: gallery feature extraction, batch collation, model saving logic.
  - `src/blip_fine_tune_2.py`: training loop, optimizer parameters, model saving.
  - `src/lavis/models/__init__.py`: package imports, model registry bindings.
  - `src/lavis/models/blip2_models/blip2_qformer_cir_align_prompt.py`: model initialization, Q-Former fusion, similarity scoring in `inference()`.
  - `src/lavis/models/blip2_models/blip2_qformer_cir_rerank.py`: two-stage ITM reranking, text-only first-stage retrieval.
  - `README.md`: author instructions, environment requirements, checkpoint links.
- **Excluded**:
  - `src/cirr_test_submission.py`: CIRR test server submission formatting only.
  - Unused LAVIS sub-packages: `albef_models/`, `alpro_models/`, `clip_models/`, `timesformer/`, `pnp_vqa_models/`, `img2prompt_models/`, `gpt_models/` (standard vendored Salesforce LAVIS modules irrelevant to BLIP-2 Q-Former CIR).

### 3. README vs Executable Code
1. **Entrypoint Script**: `README.md:108` documents `python src/blip_validate.py` for evaluation. This correctly targets `src/blip_validate.py` (which implements FashionIQ evaluation at line 155), whereas `src/validate_blip.py:508` only handles CIRR and silently exits on FashionIQ.
2. **Nonexistent Default Model Name**: `src/blip_validate.py:144` sets `--blip-model-name default="blip2_cir_rerank_learn"`. This model name is **not registered anywhere** in LAVIS registry (`@registry.register_model`), causing an immediate `KeyError` if run with defaults.
3. **Training Model Name vs Eval Default**: `README.md:92` trains with `--blip-model-name 'blip2_cir_align_prompt'`, whereas `blip_fine_tune_2.py:375` defaults to `blip2_cir_cat` and `blip_validate.py:144` defaults to nonexistent `blip2_cir_rerank_learn`.
4. **Image Extension Mismatch**: `README.md:35` shows images formatted as `.jpg` (`B00006M009.jpg`). Executable code in `data_utils.py` unconditionally hardcodes `.png` (`f"{image_name}.png"`) across 4 separate lines (167, 170, 179, 185).
5. **Rerank Code Availability**: `README.md:132` lists "code and pre-trained weights for rerank model" under `### Todo`, but the repo includes unmaintained `src/validate_blip_rerank.py` which crashes immediately on `from combiner import Combiner` (module absent).

---

## C. NATIVE EVALUATION

### 1. Proof of Protocol A (`fashioniq_original_split`)
SPRC strictly implements **Protocol A**:
1. **Full Ordered Validation Gallery Universe**:
   In `src/data_utils.py:153-156`:
   ```python
   self.image_names: list = []
   for dress_type in dress_types:
       with open(base_path / 'fashionIQ_dataset' / 'image_splits' / f'split.{dress_type}.{split}.json') as f:
           self.image_names.extend(json.load(f))
   ```
   When `mode == 'classic'`, `FashionIQDataset.__getitem__` (line 184) loads images directly from `self.image_names[index]`. The gallery is the complete validation split (`split.{dress_type}.val.json`), not the pair union. Gallery sizes:
   - Dress: 3,817 candidates
   - Shirt: 6,346 candidates
   - Toptee: 5,373 candidates
2. **Reference Image Remains Eligible (Zero Masking)**:
   In `src/validate_blip.py:43-56`:
   ```python
   # Compute the distances and sort the results
   distances = 1 - pred_sim
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
   There is **no masking, zeroing, or removal** of reference images (`reference_names`). Reference candidates remain in the gallery at their true similarity rank.

### 2. Evaluator Path & BLIP-2 Architecture
- Official evaluator entrypoint: `src/blip_validate.py:clip_finetune_fiq`.
- Evaluation model class:
  - Architecture: `Blip2QformerCirAlignPrompt` (`src/lavis/models/blip2_models/blip2_qformer_cir_align_prompt.py:26`), registered as `blip2_cir_align_prompt`.
  - Backbone: `eva_clip_g` (EVA-ViT-g, 1B parameters) initialized via `src/lavis/models/eva_vit.py:create_eva_vit_g` with pretrained weights `eva_vit_g.pth`.
  - Fusion Module: 12-layer BERT-based Q-Former initialized from `blip2_pretrained.pth`, with 32 learnable query tokens.

### 3. Preprocessing Pipeline
- TargetPad transform (`src/data_utils.py:63-102`):
  `targetpad_transform(1.25, 224)`:
  1. `SquarePad`: pads image borders with 0 constant padding to match target aspect ratio 1.25.
  2. `Resize((224, 224), interpolation=InterpolationMode.BICUBIC)`.
  3. `CenterCrop((224, 224))`.
  4. `ToTensor()`.
  5. `Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))` (standard CLIP normalization).
- Query Text Composition (`src/validate_blip.py:180-184`):
  ```python
  input_captions = [
      f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}"
      for i in range(0, len(flattened_captions), 2)
  ]
  input_captions = [txt_processors["eval"](caption) for caption in input_captions]
  ```
  Two relative modifier captions are concatenated deterministically with `" and "`, stripped of punctuation, capitalized, and normalized via `BlipCaptionProcessor`.

### 4. Similarity & Scoring
In `Blip2QformerCirAlignPrompt.inference` (`src/lavis/models/blip2_models/blip2_qformer_cir_align_prompt.py:313-361`):
1. Reference image features (`reference_embeds`, shape `[B, 257, 1408]`) and 32 query tokens are fed into Q-Former cross-attention with query text tokens (`max_txt_len=32`).
2. Fused query representation is extracted from BERT hidden state at index 32: `fusion_feats = F.normalize(self.text_proj(text_output.last_hidden_state[:, 32, :]), dim=-1)` (shape `[B, 256]`).
3. Gallery image features (`target_feats`, shape `[N_gallery, 32, 256]`) represent 32 visual query tokens projected into shared embedding space.
4. Token-wise dot-product matrix: `sim_t2q = torch.matmul(fusion_feats.unsqueeze(1).unsqueeze(1), target_feats.permute(0, 2, 1)).squeeze()`.
5. Max-query aggregation: `sim_i2t, _ = sim_t2q.max(-1)` yields single scalar similarity score per query-candidate pair.

### 5. Sorting, Top-K & Metric Definition
- Distance metric: `distances = 1 - pred_sim` (`src/validate_blip.py:44`).
- Sorting: `sorted_indices = torch.argsort(distances, dim=-1).cpu()` (ascending distance = descending similarity).
- Recall Definition:
  - `labels`: boolean matrix of shape `[num_queries, num_gallery]` where column index indicates matching `target_name`.
  - Assert verification: `assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())` proves exactly one unique ground truth target exists per query.
  - Denominator: total query count `len(labels)` (2,024 for dress, 2,015 for shirt, 2,024 for toptee).
  - Top-K hits:
    `recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100`
    `recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100`

### 6. Category Handling & Macro Aggregation
- Category execution order (`src/blip_validate.py:155`):
  `['dress', 'toptee', 'shirt']`
- Macro mean computation (`src/blip_validate.py:92-96`):
  ```python
  results_dict.update({
      f'average_recall_at10': mean(recalls_at10),
      f'average_recall_at50': mean(recalls_at50),
      f'average_recall': (mean(recalls_at50) + mean(recalls_at10)) / 2
  })
  ```
  Unweighted arithmetic mean of per-category R@10, unweighted arithmetic mean of per-category R@50, and overall average as `(average_recall_at50 + average_recall_at10) / 2`.

---

## D. PREPARATION & ASSETS

### 1. Dataset Layout & Hardcoded Repository Symlink
- Upstream data path resolution:
  `src/data_utils.py:12`: `base_path = Path(__file__).absolute().parents[1].absolute()`.
  Resolves to `<checkout>/`.
- The evaluator cannot accept a custom `--dataset-root` flag. It unconditionally opens:
  - `<checkout>/fashionIQ_dataset/captions/cap.{category}.val.json`
  - `<checkout>/fashionIQ_dataset/image_splits/split.{category}.val.json`
  - `<checkout>/fashionIQ_dataset/images/{image_name}.png`
- A filesystem symlink is strictly mandatory:
  `workbench/third_party/SPRC/fashionIQ_dataset -> <dataset_root>`.

### 2. Image Extension Contract
- FashionIQ images are opened as `.png` (`src/data_utils.py:167, 170, 179, 185`).
- The official FashionIQ dataset provides JPEG images (`.jpg`).
- The workbench preparation contract (`preparation_contracts.yaml:sprc_fashioniq`) requires `.png` formatted images.

### 3. Pretrained Foundation Weights
1. **BLIP-2 Pretrained Q-Former (`blip2_pretrained.pth`)**:
   - URL: `https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained.pth`
   - Destination: `~/.cache/lavis/blip2_pretrained.pth`
   - Expected Size: `746,998,955` bytes.
   - Status: Current local file is a 22-byte corrupted decoy containing `"corrupt nonempty bytes"`.
2. **EVA-ViT-g Visual Backbone (`eva_vit_g.pth`)**:
   - URL: `https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth`
   - Destination: `~/.cache/torch/hub/checkpoints/eva_vit_g.pth`
   - Expected Size: `2,025,249,237` bytes.
   - Status: Directory and file missing locally.
3. **BERT Tokenizer**:
   - HuggingFace `bert-base-uncased` tokenizer assets (`vocab.txt`).

---

## E. CHECKPOINT LOADER & WEIGHT INTEGRITY

### 1. Actual Loader Implementation
In `src/blip_validate.py:48-50`:
```python
checkpoint = torch.load(checkpoint_path, map_location=device)
msg = blip_model.load_state_dict(checkpoint[blip_model.__class__.__name__], strict=False)
print("Missing keys {}".format(msg.missing_keys))
```

### 2. State Dict Key Indexing
The loader expects the top-level PyTorch dictionary to contain a key matching `blip_model.__class__.__name__` (saved via `utils.py:218-221`):
- If `blip_model_name = "blip2_cir_align_prompt"`, class is `Blip2QformerCirAlignPrompt`.
- If `blip_model_name = "blip2_cir_rerank"`, class is `Blip2QformerCirRerank`.
- If the checkpoint file was saved under one class name and loaded under another, `checkpoint[blip_model.__class__.__name__]` raises `KeyError` immediately.

### 3. Concrete Consequences of `strict=False` Loading
`blip_model.load_state_dict(..., strict=False)` introduces severe, reproducible silent failure modes:
1. **Silent Fallback to Untrained Initializations**:
   `Blip2QformerCirAlignPrompt` contains visual encoder weights (`visual_encoder.*`), Q-Former transformer weights (`Qformer.*`), linear projection heads (`text_proj.*`, `vision_proj.*`), and sentence-level prompt parameters (`prompt_tokens`).
   If a checkpoint only contains a subset of these keys (e.g., fine-tuned Q-Former without ViT, or Q-Former without prompt tokens), `strict=False` suppresses the missing key error.
2. **Cross-Architecture Poisoning (Plausible-Looking Wrong Model)**:
   - If a checkpoint trained on `blip2_cir_align_prompt` is adapted and loaded into `Blip2QformerCirRerank` with `strict=False`:
     `Blip2QformerCirRerank.inference()` ignores `reference_embeds` completely and performs text-only retrieval against the gallery!
     This would execute without runtime exceptions, producing plausible positive Recall values (e.g. ~10–20%), but the reference image conditioning would be entirely ignored.
   - If an un-fine-tuned base BLIP-2 pretrain checkpoint is supplied, `strict=False` produces valid tensor outputs representing generic image-caption matching rather than CIR composed retrieval, silently masking checkpoint misconfiguration.

---

## F. PUBLISHED-METRIC DISCREPANCY & WORKBENCH INTEGRATION

### 1. Published Metric Discrepancy Investigation
An exhaustive comparison across primary sources reveals three distinct discrepancies:
1. **Paper Table 1 vs Internal Arithmetic Discrepancy**:
   - ICLR 2024 camera-ready Table 1, page 6 (row `Ours (SPRC)`, line 407):
     - Dress: R@10 = 49.18, R@50 = 72.43
     - Shirt: R@10 = 55.64, R@50 = 73.89
     - Toptee: R@10 = 59.35, R@50 = 78.58
     - Printed Average Block: R@10 = 54.92, R@50 = 74.97, Overall Avg. = 64.85
   - Arithmetic Verification:
     - Unweighted macro mean of R@10: `(49.18 + 55.64 + 59.35) / 3 = 164.17 / 3 = 54.7233...` -> `54.72`!
     - Printed Average R@10 cell: `54.92` (discrepancy of `+0.1967` percentage points).
     - Unweighted macro mean of R@50: `(72.43 + 73.89 + 78.58) / 3 = 224.90 / 3 = 74.9666...` -> rounds to `74.97` (exact match).
     - Mean of printed Average cells: `(54.92 + 74.97) / 2 = 64.945`.
     - Mean of recomputed category macro means: `(54.72333 + 74.96667) / 2 = 64.845` -> rounds to `64.85` (exact match).
   - **Conclusion**: The author's reported overall average of `64.85` was calculated using the recomputed category macro mean (`54.7233`), NOT the printed `54.92` cell. The printed cell `54.92` is an unresolved typographical or transcription error in the primary publication.
2. **Paper Prose vs Table 1 Attribution Conflict**:
   - Camera-ready prose at lines 435–437 states:
     *"modifying SPRC into a two-stage mechanism (termed SPRC2) achieved higher Avg. performance, i.e., Re-ranking: 62.15 vs. SPRC2: 64.85"*.
   - The prose explicitly attributes the `64.85` average to `SPRC2` (two-stage reranked mechanism), whereas Table 1 prints the identical numbers in the single-stage `Ours (SPRC)` row.
3. **Visual Backbone Discrepancy**:
   - Camera-ready text line 363 states: *"encoders are initialized from the BLIP-2 pretrained model with ViT-G"*.
   - Early preprint arXiv:2310.05473v1 states: *"with ViT-L and AdamW"*.
   - Repository `README.md:128` offers separate checkpoints for ViT-L (`sprc_fiq_vitl.pt`) and default (`sprc_fiq.pt`).

### 2. Workbench Integration & Metrics Parser
- **Adapter**: `SPRCAdapter` in `workbench/backend/adapters/models.py:119-122`:
  ```python
  class SPRCAdapter(OfficialScriptAdapter):
      model_id = "sprc"
      script = "src/blip_validate.py"
  ```
  `SPRCAdapter` does not implement `command()`, inheriting `OfficialScriptAdapter.command` which raises `NotImplementedError`.
- **Metrics Parser Reuse Legitimacy**:
  In `workbench/backend/metrics_extraction.py:408-411`:
  ```python
  if model_id == "sprc":
      # blip_validate.py:88-98 prints the same JSON key set as CSMCIR's evaluator; the csmcir
      # parser's macro-validation matches the same keys, so the shape is reused, not assumed.
      return parse_csmcir_output(stdout_text, paper_metrics)
  ```
  - **Verification**: `SPRC/src/blip_validate.py:88-98` constructs and prints `results_dict` with keys:
    `dress_recall_at10`, `dress_recall_at50`, `shirt_recall_at10`, `shirt_recall_at50`, `toptee_recall_at10`, `toptee_recall_at50`, `average_recall_at10`, `average_recall_at50`, `average_recall`.
  - This is byte-for-byte identical to `CSMCIR/src/validate_blip_csmcir.py:620-630`.
  - Reusing `parse_csmcir_output` for `sprc` is **fully legitimate and verified**.

---

## G. OBSERVATION CAPABILITY

### 1. Score & Ranking Production Point
- In `SPRC/src/validate_blip.py:compute_fiq_val_metrics`:
  - `pred_sim`: similarity matrix of shape `[num_queries, num_gallery]` computed by `generate_fiq_val_predictions`.
  - Distance: `distances = 1 - pred_sim` (line 44).
  - Ranking:
    ```python
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]
    ```
- Query alignment:
  Aligned 1-to-1 with `relative_val_dataset.triplets`:
  - Reference image: `reference_names[i]`
  - Target image: `target_names[i]`
  - Query text: `captions_all[i]`

### 2. Observation Export Feasibility
- Per-query export is **implementable** (`QUERY_EXPORT_IMPLEMENTABLE`).
- An intercept hook can extract `sorted_index_names[:, :50]` and similarity scores right after line 46 in `validate_blip.py` without modifying metric computation.
- Runtime blockers preventing execution:
  1. CLI default `--blip-model-name blip2_cir_rerank_learn` is unregistered.
  2. Adapter `command()` method is unimplemented.
  3. Checkpoint file `sprc_fiq.pt` is missing and its state_dict class key is unverified.
  4. Pretrained weights (`eva_vit_g.pth`, `blip2_pretrained.pth`) are missing or corrupted.
  5. Dataset symlink `<checkout>/fashionIQ_dataset` is not staged.

---

## H. DEFECTS

### Confirmed Bugs (`CONFIRMED_BUG`)
1. **C1 (P0) — Nonexistent Default Model Name in Evaluator**:
   - *File*: `workbench/third_party/SPRC/src/blip_validate.py:144`
   - *Issue*: Argparse default `--blip-model-name blip2_cir_rerank_learn` names a model never registered in LAVIS registry.
   - *Impact*: Running official evaluation without explicit CLI override crashes with `KeyError: "Model 'blip2_cir_rerank_learn' not found in registry!"`.
2. **C2 (P1) — Missing Upstream Module in Rerank Evaluator**:
   - *File*: `workbench/third_party/SPRC/src/validate_blip_rerank.py:17`
   - *Issue*: Evaluator attempts `from combiner import Combiner`. File `combiner.py` does not exist in the pinned repository.
   - *Impact*: Crashes immediately on import with `ModuleNotFoundError: No module named 'combiner'`.
3. **C3 (P2) — Hardcoded `.png` Image Extension**:
   - *File*: `workbench/third_party/SPRC/src/data_utils.py:167, 170, 179, 185`
   - *Issue*: Loader unconditionally opens `f"{image_name}.png"`, conflicting with official FashionIQ `.jpg` distribution shown in README.
   - *Impact*: `FileNotFoundError` unless images are converted to `.png`.
4. **C4 (P2) — Hardcoded Relative Dataset Root**:
   - *File*: `workbench/third_party/SPRC/src/data_utils.py:12`
   - *Issue*: Derives dataset path as `Path(__file__).parents[1] / 'fashionIQ_dataset'`. No CLI argument exists to specify an alternate dataset root.
   - *Impact*: Requires manual symlink `<checkout>/fashionIQ_dataset -> <dataset_root>`.
5. **C5 (P2) — Pinned Evaluator Missing FashionIQ Mode in `validate_blip.py`**:
   - *File*: `workbench/third_party/SPRC/src/validate_blip.py:508-509`
   - *Issue*: `validate_blip.py:main()` only branches on `if args.dataset == 'CIRR'`. Passing `--dataset fashionIQ` results in a silent no-op exit.
   - *Impact*: Misleads operators inspecting `validate_blip.py`; the real entrypoint is `blip_validate.py`.

### Scientific Defects / Discrepancies (`SCIENTIFIC_DEFECT`)
1. **S1 (P1) — Primary Source Table 1 Arithmetic Inconsistency**:
   - *Source*: ICLR 2024 camera-ready Table 1, page 6, line 407.
   - *Issue*: Category R@10 values (49.18, 55.64, 59.35) have arithmetic mean `54.7233`, but Table 1 prints `54.92`. Printed `Avg.` of `64.85` matches `mean(54.7233, 74.9667) = 64.845`, proving the printed `54.92` cell is internally inconsistent with both per-category scores and the macro average.
2. **S2 (P2) — Paper Prose vs Table 1 Attribution Conflict**:
   - *Source*: ICLR 2024 camera-ready prose lines 435–437 vs Table 1.
   - *Issue*: Prose attributes `64.85` to `SPRC2` (two-stage re-ranking), whereas Table 1 prints `64.85` under `Ours (SPRC)`.

### Workbench Defects (`WORKBENCH_DEFECT`)
1. **W1 (P1) — Unimplemented Adapter Command Constructor**:
   - *File*: `workbench/backend/adapters/models.py:119-122`
   - *Issue*: `SPRCAdapter` does not implement `command()`, raising `NotImplementedError` if invoked.
2. **W2 (P2) — Decoy Cache Asset File**:
   - *File*: `/home/heheboiz/.cache/lavis/blip2_pretrained.pth`
   - *Issue*: File exists but contains 22 bytes of ASCII text (`"corrupt nonempty bytes"`). Bypasses naive non-empty existence checks.

### External Blockers (`EXTERNAL_BLOCKER`)
1. **E1 (P0) — Missing Weights, Assets, and Unverified Checkpoint Mapping**:
   - `sprc_fiq.pt` checkpoint missing locally; OneDrive URL returns HTTP 403; class key mapping unverified.
   - EVA-ViT-g weights (`~/.cache/torch/hub/checkpoints/eva_vit_g.pth`) missing locally.
   - BLIP-2 pretrain weights (`~/.cache/lavis/blip2_pretrained.pth`) corrupted decoy.
   - Canonical dataset symlink missing.

---

## I. EVIDENCE

1. **Commit SHA Verification**:
   ```bash
   git -C workbench/third_party/SPRC rev-parse HEAD
   # Output: 2935a5397732260d1db6fa577e5926f963e36f0f
   ```
2. **Nonexistent Argparse Default Model**:
   `workbench/third_party/SPRC/src/blip_validate.py:144`:
   ```python
   parser.add_argument("--blip-model-name", default="blip2_cir_rerank_learn", type=str)
   ```
   AST grep across `src/lavis/models/` confirms zero `@registry.register_model` annotations matching `"blip2_cir_rerank_learn"`.
3. **Broken Combiner Import**:
   `workbench/third_party/SPRC/src/validate_blip_rerank.py:17`:
   `from combiner import Combiner` fails; `find workbench/third_party/SPRC -name "combiner*"` returns empty.
4. **Decoy Pretrain File Inspection**:
   ```bash
   cat /home/heheboiz/.cache/lavis/blip2_pretrained.pth
   # Output: corrupt nonempty bytes (22 bytes)
   ```
5. **Primary Metric Arithmetic Verification**:
   ```python
   dress_r10, shirt_r10, toptee_r10 = 49.18, 55.64, 59.35
   dress_r50, shirt_r50, toptee_r50 = 72.43, 73.89, 78.58
   mean_r10 = (dress_r10 + shirt_r10 + toptee_r10) / 3.0 # 54.723333333333336
   mean_r50 = (dress_r50 + shirt_r50 + toptee_r50) / 3.0 # 74.96666666666667
   overall_avg = (mean_r10 + mean_r50) / 2.0 # 64.845 -> rounds to 64.85
   mean_from_printed = (54.92 + 74.97) / 2.0 # 64.945 -> does NOT round to 64.85
   ```
6. **Readiness CLI Execution**:
   `python3 workbench/scripts/pc_readiness.py --model sprc --json`:
   State: `BLOCKED_IMPLEMENTATION`, Reason: `command not runnable: COMMAND_BLOCKED_MODEL_MAPPING`.

---

## J. VERDICT

| Axis | Classification | Justification |
| :--- | :--- | :--- |
| **Source Status** | **SOURCE_AUDITED** | Pinned source audited end to end. Vendored LAVIS imports are intact; evaluator entrypoint has nonexistent default model name and broken rerank import. |
| **Protocol Status** | **PROTOCOL_VERIFIED** | Full validation split gallery (`split.{cat}.val.json`); reference remains eligible in scoring and ranking without masking (Protocol A `fashioniq_original_split`). |
| **Command Status** | **COMMAND_BLOCKED_MODEL_MAPPING** | `SPRCAdapter.command()` is not implemented in workbench backend; default model name is unregistered; checkpoint state_dict class key is unverified. |
| **Checkpoint Status** | **CHECKPOINT_MAPPING_UNVERIFIED** | Local checkpoint missing; author OneDrive link returns HTTP 403; state_dict class key mapping uninspected; loads via `strict=False`. |
| **Query Export Status** | **QUERY_EXPORT_IMPLEMENTABLE** | Native evaluation calculates full distance matrix and argsort ranking at `validate_blip.py:44-46`; export hook is cleanly implementable before metric reduction. |
| **Validation Readiness** | **BLOCKED_IMPLEMENTATION** | Blocked on nonexistent default model name, unimplemented adapter command, missing checkpoint, missing EVA-ViT-g weights, corrupted BLIP-2 pretrain decoy, and dataset symlink. |
