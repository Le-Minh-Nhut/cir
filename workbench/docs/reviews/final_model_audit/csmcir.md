# CSMCIR Final Model Audit

Model registered as `csmcir` in CIR Failure Analysis Workbench.
Evaluated against pinned upstream checkout `workbench/third_party/CSMCIR` at commit `774f94e2076ff17ea91703a6239d2a08f0e1a44e`.
Evaluation Protocol: Protocol A (`fashioniq_original_split` — full validation split gallery, reference remains eligible).

---

## A. MODEL IDENTITY

- **Model ID**: `csmcir`
- **Method Name**: `CSMCIR`
- **Paper Title**: *CSMCIR: CoT-Enhanced Symmetric Alignment with Memory Bank for Composed Image Retrieval*
- **Publication & Year**: ACL 2026 Findings (camera-ready in ACL Anthology `2026.findings-acl.216`, pp. 4411–4424; arXiv:2601.03728 v3 published 2026-01-07). Upstream `README.md:4` contains the typo `(ACL206 Findings)`.
- **Paper URL**: `https://arxiv.org/abs/2601.03728` (registered in `workbench/registry/models.yaml:43`).
- **Upstream Repository**: `https://github.com/qzp2018/CSMCIR`
- **Pinned Commit SHA**: `774f94e2076ff17ea91703a6239d2a08f0e1a44e`
  - Verified: `git -C workbench/third_party/CSMCIR rev-parse HEAD` returns `774f94e2076ff17ea91703a6239d2a08f0e1a44e` (matches `workbench/registry/models.yaml:46`).
- **Checkpoint Variant**: `fashioniq` -> `fashioniq_tuned_clip_best.pt`
  - Repository URL: `https://huggingface.co/peng12138/CSMCIR/resolve/main/fashioniq_tuned_clip_best.pt`
  - Expected Size: `2,720,615,859` bytes (2.72 GB)
  - Expected SHA-256: `c8812071de0640d647c5ba2a70628b80a3289ee2c47676f7a1b6abaa313fa49e`
- **Paper Benchmark Row**:
  - Registered reference: `csmcir_fiq_val_ours_vitg` (`workbench/registry/paper_benchmarks/papers.yaml:784-844`).
  - Table 1 (`CSMCIR(Ours)` on Fashion-IQ validation set):
    - Dress: R@10 = 52.45, R@50 = 74.81
    - Shirt: R@10 = 57.70, R@50 = 75.76
    - Toptee: R@10 = 61.14, R@50 = 80.98
    - Macro Average: R@10 = 57.07, R@50 = 77.27, Overall Avg. = 67.17
  - Verified against author training log (`training_log/fashioniq_validation_metrics.csv` Epoch 9):
    - Dress: 52.355 / 75.062
    - Toptee: 61.142 / 80.979
    - Shirt: 57.704 / 75.761
    - Average: R@10 = 57.067, R@50 = 77.267, Overall = 67.167 (rounds to 57.07 / 77.27 / 67.17).

---

## B. SOURCE COVERAGE

### Tree Inventory
- Pinned tree: `workbench/third_party/CSMCIR` (857 tracked files in git, including committed bytecode and LAVIS components).
- Evaluation dependency graph:
  1. Entrypoint: `src/validate_blip_csmcir.py`
  2. Dataset loader: `src/data_utils_csmcir.py` (`FashionIQDataset`, `targetpad_transform`, `SquarePad`, `collate_fn`)
  3. Feature extraction: `src/utils_csmcir.py` (`extract_index_blip_caption_features`, `collate_fn`, `device`)
  4. Model definition: `src/lavis/models/blip2_models/blip2_qformer_cir_align_prompt_csmcir.py` (`Blip2QformerCirAlignPromptCSMCIR`)
  5. Architecture base: `src/lavis/models/blip2_models/blip2.py` (`Blip2Base`), `src/lavis/models/blip2_models/Qformer.py`
  6. Backbone: `src/lavis/models/eva_vit.py` (`create_eva_vit_g`)
  7. Registry & initialization: `src/lavis/models/__init__.py` (`load_model_and_preprocess`), `src/lavis/common/registry.py`
  8. Config: `src/lavis/configs/models/blip2/blip2_pretrain.yaml`
  9. Processors: `src/lavis/processors/blip_processors.py`

### Inspected vs Excluded
- **Inspected**:
  - `src/validate_blip_csmcir.py`: core evaluation loop, scoring, ranking, macro metric calculation.
  - `src/data_utils_csmcir.py`: dataset initialization, image loading, transforms, caption handling.
  - `src/utils_csmcir.py`: gallery caption feature extraction, target caption dictionaries, device handling.
  - `src/lavis/models/__init__.py`: package imports and model registration point.
  - `src/lavis/models/blip2_models/blip2_qformer_cir_align_prompt_csmcir.py`: model initialization, Q-Former fusion, similarity scoring in `inference()`.
  - `README.md`: author instructions and claims.
- **Excluded**:
  - `src/blip_fine_tune_2_csmcir.py`: training pipeline only.
  - `src/cirr_test_submission_csmcir.py`: CIRR test server submission formatting.
  - `src/blip_validate.py`: older generic validation script without CSMCIR memory bank.
  - `src/utils_test1_csmcir.py`: test-split utilities.
  - `data_utils_csmcir.py:223-350` (`FashionIQDataset_imageedit`): dead code, never instantiated by evaluation scripts.
  - Sub-packages `albef_models/`, `alpro_models/`, `clip_models/`, `timesformer/`, `pnp_vqa_models/`, `img2prompt_models/`: third-party LAVIS components irrelevant to BLIP-2 Q-Former CIR.
  - 80+ experimental `blip2_qformer_cir_align_prompt_target_image_caption_shiyan*.py` files: unpublished author ablation modules left uncommitted or removed.

### README vs Executable Code
1. **Entrypoint name**: `README.md:115` directs running `python src/blip_validate_csmcir.py`. This file does **not** exist in the repository; the executable script is `src/validate_blip_csmcir.py`.
2. **CLI argument**: `README.md:118` claims `--model-path {for path}`. The actual parser in `src/validate_blip_csmcir.py:505` defines `--blip-model-path`.
3. **Image extension**: `README.md:31-35` displays dataset images as `.jpg` (`B00006M009.jpg`). Executable code in `data_utils_csmcir.py` hardcodes `.png` (`f"{image_name}.png"`) across 10 sites.
4. **Caption dependency claim**: `README.md:91` claims "Our further experiments find that our model is stable for captions, and even without them can also achieve good performance". However, `src/utils_csmcir.py:100-101` unconditionally opens `COT_ours2/fashioniq/{dress_type}_cot_val.json` and raises `FileNotFoundError` if absent.

---

## C. NATIVE EVALUATION

### 1. Gallery Universe and Ordering
- In `src/validate_blip_csmcir.py:594-599`:
  ```python
  for idx, dress_type in enumerate(val_dress_types):
      idx_to_dress_mapping[idx] = dress_type
      relative_val_dataset = FashionIQDataset('val', [dress_type], 'relative', preprocess)
      classic_val_dataset = FashionIQDataset('val', [dress_type], 'classic', preprocess)
  ```
- In `src/data_utils_csmcir.py:417-420`:
  ```python
  self.image_names: list = []
  for dress_type in dress_types:
      with open(base_path / 'fashionIQ_dataset' / 'image_splits' / f'split.{dress_type}.{split}.json') as f:
          self.image_names.extend(json.load(f))
  ```
- In `src/data_utils_csmcir.py:466-470`:
  ```python
  elif self.mode == 'classic':
      image_name = self.image_names[index]
      image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{image_name}.png"
      image = self.preprocess(PIL.Image.open(image_path))
      return image_name, image
  ```
- Gallery features are extracted via `extract_index_blip_caption_features(classic_val_dataset, blip_model)` (`src/utils_csmcir.py:89, 147-167`), where the DataLoader preserves sequential split order. `index_names` equals `image_names` exactly.

### 2. Reference Eligibility (Proof of Non-Masking)
In `src/validate_blip_csmcir.py:43-56`:
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
- `pred_sim` has shape `(num_queries, num_gallery)`.
- `distances = 1 - pred_sim` is sorted across all gallery items.
- Reference image names (`reference_names`) are neither masked out with `-inf` / `1e9` nor filtered from `index_names`.
- If the reference image ranks in the top 10, it occupies a rank slot and can push the target image out of the top 10.
- Reference remains fully eligible in the candidate pool.

### 3. BLIP/LAVIS Imports
- Evaluator calls `from lavis.models import load_model_and_preprocess` (`src/validate_blip_csmcir.py:12`).
- Instantiated at line 568:
  `blip_model, _, txt_processors = load_model_and_preprocess(name=blip_model_name, model_type="pretrain", is_eval=False, device=device)`
- Fails on unpatched checkout because `src/lavis/models/__init__.py:44-151` fails to import 98 nonexistent modules.

### 4. Similarity Scoring
In `src/lavis/models/blip2_models/blip2_qformer_cir_align_prompt_csmcir.py:411-451`:
```python
    attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
    fusion_output = self.Qformer.bert(
        text_tokens.input_ids,
        query_embeds=query_tokens,
        attention_mask=attention_mask,
        encoder_hidden_states=reference_embeds,
        encoder_attention_mask=image_atts,
        return_dict=True,
    )
    fusion_feats = F.normalize(
        self.text_proj(fusion_output.last_hidden_state[:, 32, :]), dim=-1)

    sim_t2q = torch.matmul(
        fusion_feats.unsqueeze(1).unsqueeze(1), target_feats.permute(0, 2, 1)
    ).squeeze()
    sim_i2t, _ = sim_t2q.max(-1)
    return sim_i2t
```
- Query representation: Token 32 of Q-Former output projected via `self.text_proj` and L2 normalized.
- Multi-token target features: `sim_t2q.max(-1)` performs max-pooling across target query tokens.
- Output `pred_sim` is the dot-product similarity.

### 5. Sorting, Top-K, and Recall@K
- Distance: `distances = 1 - pred_sim`
- Sorting: `sorted_indices = torch.argsort(distances, dim=-1).cpu()` (ascending distance = descending similarity).
- Recall: Evaluates binary indicator `labels[:, :K]` over total queries `len(labels)`.

### 6. Category Handling & Macro Aggregation
In `src/validate_blip_csmcir.py:620-628`:
```python
    results_dict = {}
    for i in range(len(recalls_at10)):
        results_dict[f'{idx_to_dress_mapping[i]}_recall_at10'] = recalls_at10[i]
        results_dict[f'{idx_to_dress_mapping[i]}_recall_at50'] = recalls_at50[i]
    results_dict.update({
        f'average_recall_at10': mean(recalls_at10),
        f'average_recall_at50': mean(recalls_at50),
        f'average_recall': (mean(recalls_at50) + mean(recalls_at10)) / 2
    })
```
- Categories evaluated: `dress`, `toptee`, `shirt` (`val_dress_types` at line 592).
- Aggregation is the unweighted arithmetic mean across categories, and overall average is `(mean(R@10) + mean(R@50)) / 2`.

---

## D. PREPARATION & CAPTIONS

### 1. Proof of Protocol A
CSMCIR is unambiguously **Protocol A (`fashioniq_original_split`)**:
1. Gallery universe: strictly the full split (`split.{category}.val.json`), with counts 3,817 (dress), 6,346 (shirt), and 5,373 (toptee). It is NOT the union of pairs (Protocol C).
2. Reference eligibility: the reference image is NOT excluded or masked (as required by Protocol B/C).

### 2. Captions & Critical Equivalence Question
- **Query Captions**: `FashionIQDataset` (`src/data_utils_csmcir.py:408`) reads `fashionIQ_dataset/qwen_captions/{dress_type}_cot_{split}.json`. These are published on HuggingFace `peng12138/CSMCIR` and verified.
- **Gallery Captions**: `extract_index_blip_caption_features` (`src/utils_csmcir.py:100`) opens `COT_ours2/fashioniq/{dress_type}_cot_val.json`.
- **CRITICAL QUESTION: Are Qwen caption aliases PROVEN byte-identical to the author's unpublished COT files?**
  - **Verdict**: **`SOURCE_COMPATIBLE_EQUIVALENCE_UNKNOWN`**.
  - **Findings**: The author **never published** the `COT_ours2` directory (neither on HuggingFace nor on GitHub). While staging the published Qwen validation captions at `COT_ours2/fashioniq/{dress_type}_cot_val.json` satisfies the schema (`Final_Caption`), record counts, and gallery IDs, **nothing proves these files are byte-identical to the author's private files**.
  - **Workbench Policy**: The workbench does not claim official parity. It records `staged_as: SOURCE_COMPATIBLE_CAPTION_ALIAS`, emits a preflight warning, and the gate labels the run `PASS_WITHIN_5_PERCENT_VARIANT`.

### 3. Approved Compatibility Patch
- **Patch file**: `workbench/patches/csmcir_models_init_compat.patch`
- **Target file**: `src/lavis/models/__init__.py`
- **Digests**:
  - Base SHA-256: `9d8c914b551a7de614e062f9a241f44626fd591a31a0ccf1cc79f0ae23c9db01`
  - Patch SHA-256: `0e703584a7ecb5df3781574c2516fd406745c414491b6aa487fbb3e76532b9bd`
  - Derived SHA-256: `aa68fe5e20bcda5e17d978799deb8ec0ff4636e5b3b2a6ef637ce5e385b0c818`
- **Scope**: Encloses 98 missing experimental import statements within 4 `try: ... except ImportError: pass` blocks so that line 152 (`from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_csmcir import Blip2QformerCirAlignPromptCSMCIR`) can execute. It is purely packaging/import repair and does not modify any evaluation or model arithmetic.

---

## E. CHECKPOINT

- **Checkpoint File**: `fashioniq_tuned_clip_best.pt`
- **Structure**: PyTorch serialized dict mapping `checkpoint['Blip2QformerCirAlignPromptCSMCIR']` to model weights.
- **Loader**:
  `checkpoint = torch.load(blip_model_path, map_location=device)`
  `msg = blip_model.load_state_dict(checkpoint[blip_model.__class__.__name__], strict=False)` (`src/validate_blip_csmcir.py:570-571`).
- **Pretrained Backbone Requirements**:
  1. BLIP-2 pretrain weights: `blip2_pretrained.pth` (746,998,955 B from Salesforce storage).
  2. EVA-ViT-g backbone: `eva_vit_g.pth` (2,025,249,237 B from Salesforce storage).
  3. BERT tokenizer: `bert-base-uncased` vocab.txt from HuggingFace.
- **Mapping Confidence**:
  - `VERIFIED_METADATA`: Single published FashionIQ checkpoint on HuggingFace matching paper benchmark numbers and training log epoch 9.
  - Runtime execution remains unverified locally because checkpoint weights are not staged in `workbench/artifacts/checkpoints/csmcir/`.

---

## F. WORKBENCH INTEGRATION

1. **Adapter**: `CSMCIRAdapter` in `workbench/backend/adapters/models.py:28-33`:
   - Builds command: `[python, <source>/src/validate_blip_csmcir.py, "--dataset", "fashionIQ", "--blip-model-path", <checkpoint>]`.
   - Runs with cwd `source/src` (`evaluate_models.py:185`).
2. **Preparation Contract**:
   - `models.yaml:41` declares `preparation_contract: manual_standard`.
   - In `workbench/registry/preparation_contracts.yaml`, there are 0 contracts for `csmcir` (dangling layout label; handled via `auxiliary_assets.yaml` and `fashioniq_layout.py`).
3. **Runtime Preflight & Doctor**:
   - `workbench/backend/preflight.py:preflight_blockers` enforces:
     - Approved source patch state for `src/lavis/models/__init__.py`.
     - Auxiliary asset existence and digest validation.
     - Canonical dataset symlink (`CSMCIR/fashionIQ_dataset -> data/fashionIQ_dataset`).
     - Image extension compatibility (`.png`).
4. **Metrics Parser**:
   - `parse_csmcir_output` in `workbench/backend/metrics_extraction.py:219-307`:
     - Parses JSON matching `\{[^{}]*"average_recall"[^{}]*\}`.
     - Validates category keys and recomputes macro means. Discrepancies > 0.05 trigger `EVALUATION_COMPLETED_METRICS_UNPARSED`.
5. **Benchmark Gate**:
   - `workbench/backend/paper_gate.py` compares parsed metrics against `papers.yaml:csmcir_fiq_val_ours_vitg`.
   - Attaches `PASS_WITHIN_5_PERCENT_VARIANT` when run with caption aliases or patched source.
6. **UI Representation**:
   - `frontend/gate.ts`: Maps `caption_alias_staged` to badge `CAPTION THAY THẾ`.

---

## G. OBSERVATION CAPABILITY

### Ranking Production Point
- In `src/validate_blip_csmcir.py:compute_fiq_val_metrics`:
  - `pred_sim`: Shape `[num_queries, num_gallery]`.
  - `sorted_indices = torch.argsort(1 - pred_sim, dim=-1).cpu()`
  - `sorted_index_names = np.array(index_names)[sorted_indices]` (line 46).
- Query order corresponds 1-to-1 with `relative_val_dataset.triplets`:
  - Reference image: `reference_names[i]`
  - Target image: `target_names[i]`
  - Query text: `captions_all[i]` (and raw pairs in `triplets[i]['captions']`).

### Observation Hook & Feasibility
- Per-query export is **implementable** (`QUERY_EXPORT_IMPLEMENTABLE`).
- Native script currently returns only `recall_at10, recall_at50` and prints summary JSON.
- An observer hook can intercept `compute_fiq_val_metrics` right after line 46 to dump `sorted_index_names[:, :50]` and similarity scores to disk without altering metric computation.

---

## H. DEFECTS

### Confirmed Bugs (CONFIRMED_BUG)
1. **C1 (P0) — Unimportable Pinned Source**:
   - *Path*: `workbench/third_party/CSMCIR/src/lavis/models/__init__.py:44-151`
   - *Issue*: 98 unguarded imports of deleted/uncommitted modules precede line 152 where `Blip2QformerCirAlignPromptCSMCIR` is registered.
   - *Impact*: Module import fails immediately with `ModuleNotFoundError`. Evaluator cannot launch without approved patch.
2. **C2 (P0) — Unpublished Gallery Caption Dependency**:
   - *Path*: `workbench/third_party/CSMCIR/src/utils_csmcir.py:100`
   - *Issue*: Evaluator unconditionally opens `<checkout>/COT_ours2/fashioniq/{dress_type}_cot_val.json`.
   - *Impact*: Evaluator crashes with `FileNotFoundError` unless caption alias is staged.
3. **C3 (P2) — Nonexistent Entrypoint & Argument in README**:
   - *Path*: `workbench/third_party/CSMCIR/README.md:115, 118`
   - *Issue*: Documents `blip_validate_csmcir.py` (which does not exist) and `--model-path` (actual flag is `--blip-model-path`).
4. **C4 (P2) — Hardcoded `.png` Image Extension**:
   - *Path*: `workbench/third_party/CSMCIR/src/data_utils_csmcir.py:468` (and 9 other sites)
   - *Issue*: Hardcodes `.png`, conflicting with official FashionIQ `.jpg` distribution shown in `README.md:32`.
5. **C5 (P2) — Upstream Caption Collision Overwrite**:
   - *Path*: `workbench/third_party/CSMCIR/src/utils_csmcir.py:97-103`
   - *Issue*: Iterates over `['dress', 'shirt', 'toptee']` updating a single dict. Toptee overwrites shirt captions for 121 overlapping validation IDs.

### External Blockers (EXTERNAL_BLOCKER)
1. **E1 (P0) — Missing Weights & Dataset Symlink**:
   - EVA-ViT-g weights (`~/.cache/torch/hub/checkpoints/eva_vit_g.pth`) missing.
   - BLIP-2 pretrain weights (`~/.cache/lavis/blip2_pretrained.pth`) unverified.
   - Checkpoint (`workbench/artifacts/checkpoints/csmcir/fashioniq_tuned_clip_best.pt`) missing locally.
   - Canonical dataset link (`CSMCIR/fashionIQ_dataset -> data/fashionIQ_dataset`) not staged.

### Workbench Defects (WORKBENCH_DEFECT)
1. **W1 (P3) — Duplicate Entries in `source_status`**:
   - *Path*: `workbench/backend/runtime.py:2229`
   - *Issue*: Lists approved alias files twice (6 entries for 3 files) due to scanning untracked and ignored paths redundantly.
2. **W2 (P3) — Dangling Preparation Contract**:
   - *Path*: `workbench/registry/models.yaml:41`
   - *Issue*: Declares `preparation_contract: manual_standard` with 0 entries in `preparation_contracts.yaml`.

### Rejected Findings (REJECTED_FINDING)
1. **R1 — "text_cot feeds the query text"**: Disproved. `text_cot` in `blip2_qformer_cir_align_prompt_csmcir.py:411` is an unused parameter in `inference()`. CoT feeds gallery features only.
2. **R2 — "Qwen captions are proven identical to COT_ours2"**: Disproved. Author COT files were never published; equivalence is strictly unproven.

---

## I. EVIDENCE

1. `git -C workbench/third_party/CSMCIR rev-parse HEAD`:
   `774f94e2076ff17ea91703a6239d2a08f0e1a44e`
2. Missing import count in `src/lavis/models/__init__.py`:
   AST parse reveals exactly 98 missing imports spanning lines 44 to 151.
3. Patch verification:
   - Base SHA-256: `9d8c914b551a7de614e062f9a241f44626fd591a31a0ccf1cc79f0ae23c9db01`
   - Patch SHA-256: `0e703584a7ecb5df3781574c2516fd406745c414491b6aa487fbb3e76532b9bd`
   - Derived SHA-256: `aa68fe5e20bcda5e17d978799deb8ec0ff4636e5b3b2a6ef637ce5e385b0c818` (matches `models.yaml:78`).
4. Overlapping IDs between shirt and toptee:
   `|split.shirt.val ∩ split.toptee.val| = 121`. Shared IDs with identical `Final_Caption` = 0.
5. `pc_readiness.py --model csmcir --json`:
   State: `BLOCKED_SOURCE`, 6 blockers, 3 warnings for caption aliases.

---

## J. VERDICT

| Axis | Classification | Justification |
| :--- | :--- | :--- |
| **Source Status** | **BLOCKED_SOURCE** | Pinned checkout cannot import its own evaluator due to 98 missing experimental modules. Fully audited statically; requires approved compatibility patch. |
| **Protocol Status** | **PROTOCOL_VERIFIED** | Full validation split gallery; reference remains eligible in scoring and ranking (Protocol A `fashioniq_original_split`). |
| **Command Status** | **COMMAND_AUDITED** | `CSMCIRAdapter` command audited and verified against working `src/validate_blip_csmcir.py`. |
| **Checkpoint Status** | **CHECKPOINT_VERIFIED** | Metadata, HuggingFace LFS digest, and 1-to-1 paper row mapping verified; local weight file unstaged. |
| **Query Export Status** | **QUERY_EXPORT_IMPLEMENTABLE** | Native evaluation emits aggregate summary only; ranking tensor interceptable at line 46 of `compute_fiq_val_metrics`. |
| **Validation Readiness**| **BLOCKED_EXTERNAL_DEPENDENCY** | Blocked on EVA-ViT-g weights, BLIP-2 pretrain weights, checkpoint download, dataset symlink, and caption alias staging. |
