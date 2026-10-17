# Final Model Audit: limn

## A. MODEL IDENTITY

- **Model ID**: `limn` (`workbench/registry/models.yaml:900`)
- **Method Name**: `LIMN` (`workbench/registry/models.yaml:906`)
- **Research Generation**: `legacy` (`workbench/registry/models.yaml:907`)
- **Publication Year**: `2024` (`workbench/registry/models.yaml:908`)
- **Paper Title**: `Self-Training Boosted Multi-Factor Matching Network for Composed Image Retrieval` (`workbench/registry/models.yaml:909`)
- **Paper URL**: `https://doi.org/10.1109/TPAMI.2023.3346434` (`workbench/registry/models.yaml:910`)
- **Publication Venue**: IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), Vol. 46, No. 5, May 2024, pp. 3665–3678 (arXiv:2305.09979v2)
- **Upstream Repository URL**: `https://github.com/iLearn-Lab/TPAMI24-LIMN` (`workbench/registry/models.yaml:911`)
- **Upstream Pinned Commit SHA**: `7d7bc9b116f594a65ac22457491edf28a88d3c3e` (`workbench/registry/models.yaml:913`)
- **Source Directory**: `LIMN` (`workbench/registry/models.yaml:915`)
- **Source Available**: `true` (`workbench/registry/models.yaml:914`)
- **Native Protocol**: `fashioniq_val_split` (`workbench/registry/models.yaml:921`)
- **Supported Protocols**: `[fashioniq_val_split]` (`workbench/registry/models.yaml:922`)
- **Reproduction Status**: `BLOCKED_RUNTIME_PREREQUISITES` (`workbench/registry/models.yaml:930`)
- **Command Status**: `REPLAY_COMMAND_IMPLEMENTED` (`workbench/registry/models.yaml:931`)
- **Checkpoint Serialization**: `whole_model` (`workbench/registry/models.yaml:903`)
- **Category Model Policy**: `independent` (`workbench/registry/models.yaml:902`)
- **Replay Type**: `SOURCE_FAITHFUL_WORKBENCH_REPLAY` (`workbench/registry/models.yaml:904`)
- **Reported Metrics**: `reported_r10: 55.91`, `reported_r50: 77.82`, `reported_mean: 66.87` (`workbench/registry/models.yaml:927-929`)

### Pinned Commit Verification
Verification command:
```bash
git -C workbench/third_party/LIMN rev-parse HEAD
# Output: 7d7bc9b116f594a65ac22457491edf28a88d3c3e
```
The pinned commit SHA in `workbench/third_party/LIMN` exactly equals `upstream_commit_sha: 7d7bc9b116f594a65ac22457491edf28a88d3c3e` in `models.yaml:913`.

### Paper Benchmark Reference
The paper benchmark registry (`workbench/registry/paper_benchmarks/papers.yaml:1589-1659`) documents the reference row:
- **Reference ID**: `limn_fashioniq_base_iter0_table1`
- **Method Variant**: Base LIMN without iterative dual self-training (iteration 0).
- **Paper Table**: Table I ("Performance comparison on FashionIQ and Shoes with respect to R@k(%)"), page 8, arXiv:2305.09979v2.
- **Backbone**: OpenCLIP ViT-L/14, feature dimension 768.
- **Published Metrics**:
  - `dress`: R@10 = 50.72, R@50 = 74.52
  - `shirt`: R@10 = 56.08, R@50 = 77.09
  - `toptee`: R@10 = 60.94, R@50 = 81.85
  - `Avg`: R@10 = 55.91, R@50 = 77.82 (Macro arithmetic average: `(50.72 + 56.08 + 60.94) / 3 = 55.9133...`, `(74.52 + 77.09 + 81.85) / 3 = 77.82`)
  - `Mean`: 66.87 (`WORKBENCH_DERIVED`: `(55.91 + 77.82) / 2 = 66.865 -> 66.87`)
- **Non-interchangeable Variant**: Table I penultimate row publishes `LIMN+` (Avg R@10 = 57.43, R@50 = 78.60). As stated in `papers.yaml:1595-1596` and `models.yaml:955`, `LIMN+` is an iterative dual self-training enhancement with generated pseudo triplets (`LIMN+/pseudo_data/`), which is a different system and explicitly not registered under `limn`.

---

## B. SOURCE COVERAGE

- **Audit Type**: `STATIC_AUDIT` + local script validation.
- **Tree Inventory**: `workbench/third_party/LIMN/`
  ```text
  workbench/third_party/LIMN/
  ├── LICENSE
  ├── README.md
  ├── requirements.txt
  ├── correct_files/
  │   ├── correction_dict_dress.json
  │   ├── correction_dict_shirt.json
  │   ├── correction_dict_shoes.json
  │   └── correction_dict_toptee.json
  ├── LIMN/
  │   ├── Transformer.py
  │   ├── MHTransformer.py
  │   ├── cirr_test_submission.py
  │   ├── datasets.py
  │   ├── model.py
  │   ├── test.py
  │   ├── train.py
  │   └── utils.py
  └── LIMN+/
      ├── Transformer.py
      ├── MHTransformer.py
      ├── cirr_test_submission.py
      ├── datasets.py
      ├── model.py
      ├── test.py
      ├── train.py
      ├── utils.py
      └── pseudo_data/
          └── augment_data_*.json
  ```

### Reachable Evaluation Dependency Graph
1. `workbench/replay/limn.py`: Workbench replay orchestrator.
2. `workbench/third_party/LIMN/LIMN/test.py`: Pinned native evaluation entrypoint (`test()`).
3. `workbench/third_party/LIMN/LIMN/datasets.py`: Pinned native dataset loader (`FashionIQ`), image loading (`get_test_img`), caption preprocessing (`concat_text`, `correct_text`), and query/gallery construction (`get_test_data`).
4. `workbench/third_party/LIMN/LIMN/model.py`: Pinned native model classes (`LIMN`, `Backbone`, `PositionalEncoding`).
5. `workbench/third_party/LIMN/LIMN/Transformer.py`: Pinned native attention and transformer layers used in `model.py:119`.
6. `workbench/third_party/LIMN/correct_files/correction_dict_{category}.json`: Pinned spelling correction dictionaries loaded in `datasets.py:70,141`.
7. `workbench/third_party/LIMN/README.md` & `requirements.txt`: Environment and configuration contracts.

### Inspected vs Excluded Files
- **Inspected & Reachable**:
  - `LIMN/test.py`: Reachable. Evaluated line by line for query/target feature loops, reference masking, matrix dot product, argsort, and recall computation.
  - `LIMN/datasets.py`: Reachable. Evaluated line by line for dataset loading, integer indexing into `split.{cat}.val.json`, caption formatting, and pickle caching.
  - `LIMN/model.py`: Reachable. Evaluated line by line for OpenCLIP ViT-L/14 backbone configuration, tokenizer, feature extraction, and projection layers.
  - `LIMN/Transformer.py`: Reachable. Evaluated for query transformer architecture.
  - `correct_files/correction_dict_*.json`: Reachable. Verified text correction mappings.
- **Inspected but Excluded from Execution Graph**:
  - `LIMN/train.py`: Inspected. Non-evaluation script; analyzed to verify checkpoint saving mechanism (`torch.save(model, ...)`) and training parameters.
  - `LIMN/utils.py`: Inspected. Utility classes (`Params`, `RunningAverage`, `set_logger`) not called by evaluation path.
  - `LIMN/MHTransformer.py`: Inspected. Dead file in upstream repository; not imported by any module in `LIMN/` or `LIMN+/`.
  - `LIMN/cirr_test_submission.py`: Excluded. Dedicated submission generator for the CIRR benchmark; unrelated to FashionIQ evaluation.
  - `LIMN+/`: Excluded. Contains code and pseudo-triplets for the iterative dual self-training system (`LIMN+`), which is out of scope for the registered base iteration 0 model.

---

## C. NATIVE EVALUATION

- **Verdict Axis**: `PROTOCOL_VERIFIED`
- **Native Protocol**: Protocol C (`fashioniq_val_split` / Pair-Union Gallery, Reference Excluded).

### Source Trace & Quoted Implementation

#### 1. Query Construction & Ordering
In `workbench/third_party/LIMN/LIMN/datasets.py:143-156`:
```python
143:        test_queries = []
144:        for idx in range(len(ref_captions)):
145:            caption = ref_captions[idx]
146:            mod_str = self.concat_text(caption['captions'], correction_dict)
147:            candidate = caption['candidate']
148:            target = caption['target']
149:            out = {}
150:            out['source_img_id'] = images.index(candidate)
151:            out['source_img_data'] = self.get_test_img(self.name + '_' + candidate)
152:            out['target_img_id'] = images.index(target)
153:            out['target_img_data'] = self.get_test_img(self.name + '_' + target)
154:            out['mod'] = {'str': mod_str}
155:
156:            test_queries.append(out)
```
- Query ordering is strictly sequential, matching index `idx = 0` to `len(ref_captions) - 1` from `captions/cap.{category}.val.json`.
- `candidate` and `target` strings are mapped to integer positions in `images = json.load(open('image_splits/split.{category}.val.json'))`.

#### 2. Gallery Construction & Ordering
In `workbench/third_party/LIMN/LIMN/datasets.py:158-170`:
```python
158:        test_targets_id = []
159:        for i in test_queries:
160:            if i['source_img_id'] not in test_targets_id:
161:                test_targets_id.append(i['source_img_id'])
162:            if i['target_img_id'] not in test_targets_id:
163:                test_targets_id.append(i['target_img_id'])
164:        test_targets = []
165:        for i in test_targets_id:
166:            out = {}
167:            out['target_img_id'] = i
168:            out['target_img_data'] = self.get_test_img(self.name + '_' + images[i])      
169:            test_targets.append(out)
170:        return test_queries, test_targets
```
- The gallery (`test_targets`) is constructed as the first-seen ordered unique union of `source_img_id` and `target_img_id` across the query set.
- It is NOT the entire validation split file (`images`). Gallery size is 2,628 (dress), 3,089 (shirt), and 2,902 (toptee), whereas the split file contains 3,817 (dress), 6,346 (shirt), and 5,373 (toptee) images.

#### 3. Feature Extraction, Normalization & Similarity Computation
In `workbench/third_party/LIMN/LIMN/test.py:27-74`:
```python
41:                    dataset_type = torch.ones(imgs.shape[0]).cuda() * dataset_name_dict[category]
42:                    f = model.compose_query(imgs, mods, dataset_type)[0]
43:                    f = f.data.cpu().numpy()
44:                    all_queries += [f]
...
60:                    dataset_type = torch.ones(imgs.shape[0]).cuda() * dataset_name_dict[category]
61:                    imgs = model.compose_target(imgs, dataset_type)[0].data.cpu().numpy()
62:                    all_imgs += [imgs]
...
67:    for i in range(all_queries.shape[0]):
68:        all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
69:    for i in range(all_imgs.shape[0]):
70:        all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])
71:    
72:    # match test queries to target images, get nearest neighbors
74:    sims = all_queries.dot(all_imgs.T)
```
- Extracted features are L2-normalized row-wise in numpy (`all_queries[i, :] /= np.linalg.norm(...)`).
- Similarity matrix `sims` is computed via matrix dot product `all_queries.dot(all_imgs.T)` (shape `num_queries × num_gallery_targets`).

#### 4. Reference Policy & Masking Value
In `workbench/third_party/LIMN/LIMN/test.py:76-82`:
```python
76:    test_targets_id = []
77:    for i in test_targets:
78:        test_targets_id.append(i['target_img_id'])
79:    
80:    if name != 'birds':
81:        for i, t in enumerate(test_queries):
82:            sims[i, test_targets_id.index(t['source_img_id'])] = -10e10
```
- For all FashionIQ categories (`dress`, `shirt`, `toptee`), `name != 'birds'` is True.
- The reference image is masked by writing `-10e10` directly into its similarity score column.
- The reference image is NOT deleted from the gallery; it is suppressed to the bottom of the ranking.

#### 5. Sorting Semantics & Top-K Depth
In `workbench/third_party/LIMN/LIMN/test.py:85`:
```python
85:    nn_result = [np.argsort(-sims[i, :])[:50] for i in range(sims.shape[0])]
```
- Similarity values are sorted in descending order (`-sims[i, :]`).
- Sorted ranking is truncated to top 50 (`[:50]`).

#### 6. Recall@K Definition
In `workbench/third_party/LIMN/LIMN/test.py:88-97`:
```python
88:    # compute recalls
89:    out = []
90:    for k in [1, 10, 50]:
91:        r = 0.0
92:        for i, nns in enumerate(nn_result):
93:            if test_targets_id.index(test_queries[i]['target_img_id']) in nns[:k]:
94:                r += 1
95:        r = 100 * r / len(nn_result)
96:        out += [('{}_r{}'.format(name, k), r)]
97:
98:    return out
```
- A hit is registered if the column index of the target image `test_targets_id.index(test_queries[i]['target_img_id'])` appears in the top-K retrieved gallery indices `nns[:k]`.
- Recall@K is computed as `100 * hits / num_queries`.
- The native function returns only `[(f'{name}_r1', r1), (f'{name}_r10', r10), (f'{name}_r50', r50)]`.

#### 7. Category Aggregation
- Upstream `test.py` does not perform cross-category aggregation.
- The paper Table I and workbench `replay/common.py:macro_aggregate` compute the unweighted arithmetic mean over dress, shirt, and toptee for R@10 and R@50:
  $$\text{Macro } R@10 = \frac{R@10_{\text{dress}} + R@10_{\text{shirt}} + R@10_{\text{toptee}}}{3}$$
  $$\text{Macro } R@50 = \frac{R@50_{\text{dress}} + R@50_{\text{shirt}} + R@50_{\text{toptee}}}{3}$$
  $$\text{Macro Mean} = \frac{\text{Macro } R@10 + \text{Macro } R@50}{2}$$

---

## D. PREPARATION

- **Preparation ID**: `limn_fashioniq` (`workbench/registry/models.yaml:919`)
- **Preparation Contract**: `manual_ilearn_resized` (`workbench/registry/preparation_contracts.yaml:249-300`)
- **Dataset Layout**: `fashioniq_ilearn_resized_training` (`workbench/registry/models.yaml:916`)

### Raw Input Requirements
Declared in `preparation_contracts.yaml:251-264`:
- `captions/cap.{category}.train.json` (categories: dress, shirt, toptee)
- `captions/cap.{category}.val.json`
- `image_splits/split.{category}.val.json`

### Generated Artifacts
1. **Resized Images**:
   - Paths: `resized_image/dress/`, `resized_image/shirt/`, `resized_image/toptee/`
   - Image dimensions: 256 × 256 JPEG
   - Resampling filter: `LANCZOS` (`workbench/scripts/prepare_model.py:88-92`)
   - Source generator: `fashioniq_resize_images` (derived from `start_kit/resize_images.py`)
2. **Correction Dictionaries**:
   - Paths: `captions/correction_dict_{category}.json`
   - Source: Pinned directory `workbench/third_party/LIMN/correct_files/`

### Preparation Discrepancy Audit
- **Layout Discrepancy (CONFIRMED_BUG in Upstream README)**:
  `workbench/third_party/LIMN/README.md:51` instructs users to save resized images to flat `[your path]/fashion_iq_data/resized_image`. However, `datasets.py:112` and `datasets.py:121` explicitly index:
  `os.path.join(self.image_dir, image_name.split('_')[0], image_name.split('_')[1] + ".jpg")`
  which resolves to `resized_image/{category}/{image_id}.jpg`. A flat directory will crash at runtime with `FileNotFoundError`. The workbench preparation contract correctly specifies `layout: per_category`.
- **Resampling Filter Parity (REJECTED_FINDING)**:
  Prior audit notes questioned whether `LANCZOS` differed from FashionIQ `start_kit/resize_images.py`. `start_kit/resize_images.py:10` specifies `Image.ANTIALIAS`, which in modern Pillow is an alias for `Image.Resampling.LANCZOS`. The workbench's `RESAMPLING = "LANCZOS"` is therefore faithful to the upstream specification.

---

## E. CHECKPOINT

- **Verdict Axis**: `CHECKPOINT_VERIFIED`
- **Serialization Policy**: `whole_model` (`models.yaml:903`)
- **Category Model Policy**: `independent` (`models.yaml:902`)
- **Bundle ID**: `base_iter0_all_categories` (`models.yaml:985`)

### Checkpoint Inventory and Verification
All three category checkpoints are present locally in `workbench/artifacts/checkpoints/limn/`:

| Checkpoint ID | Filename | Size (bytes) | Expected SHA-256 Digest | Status |
|---|---|---|---|---|
| `base_iter0_dress` | `0_dress_best_model.pt` | 1,792,010,157 | `96a335d4eb8b1e77d35575aab9177637208e2eb6c59e12f833f67e5cb0e7fc1c` | `VERIFIED_METADATA` |
| `base_iter0_shirt` | `0_shirt_best_model.pt` | 1,792,010,157 | `9260c7f77d3017d8cc10e6652cc8cfe275dbc9704366fa2bf581a24808930d74` | `VERIFIED_METADATA` |
| `base_iter0_toptee` | `0_toptee_best_model.pt` | 1,792,010,157 | `79f663a14d89d455fdc3612b0b36d011c06d7957d1475a654755e1053a6cece2` | `VERIFIED_METADATA` |

Validation check:
```bash
.venv/bin/python -c "from pathlib import Path; from workbench.replay.limn import validate_checkpoint_bundle; print(validate_checkpoint_bundle(Path('workbench/artifacts/checkpoints/limn')))"
# Output: []
```
Result: 0 blockers. All three checkpoint files exist, are non-empty, and match expected SHA-256 hashes bit-for-bit.

### Architecture & Loader
- **Backbone**: OpenCLIP ViT-L/14 (`model.py:15-18`), initialized with `DataComp.XL-s13B-b90K` weights (`open_clip_pytorch_model.bin`).
- **Query Transformer**: 2 layers, 8 heads, hidden dimension 768 (`Transformer.py`).
- **Embeddings**: `prefix_query` parameter (`[1, 9, 768]`), `prefix_target` parameter (`[1, 9, 768]`).
- **Serialization**: Saved via `torch.save(model, ...)` in `train.py:263`. The checkpoint contains a serialized Python object tree (pickle).
- **Security Loader**: `workbench/replay/common.py:load_trusted_whole_model` verifies the SHA-256 hash against the registry before passing the weights to `torch.load(weights_path, weights_only=False)`. `limn.py:171` confirms `isinstance(model, native_model.LIMN)`.

### Provenance Confidence
- Remote Hub Repository: `https://huggingface.co/iLearn-Lab/TPAMI24-LIMN` (pinned commit `30560ad575a56c39cc39047d1a474269338c77c0`).
- The remote Git LFS pointers for `0_{dress,shirt,toptee}_best_model.pt` match the expected sizes (1,792,010,157 bytes) and SHA-256 hashes exactly. Provenance status is `VERIFIED_METADATA`.

---

## F. WORKBENCH INTEGRATION

- **Verdict Axis**: `COMMAND_AUDITED`
- **Replay Runner**: `workbench/replay/limn.py`
- **Adapter**: `LIMNAdapter` (`workbench/backend/adapters/models.py:125-145`)
- **Metrics Parser**: `parse_limn_output` (`workbench/backend/metrics_extraction.py:74-216`)
- **Paper Gate**: Validated against `limn_fashioniq_base_iter0_table1` (`workbench/backend/paper_gate.py`)

### Audit of `workbench/replay/limn.py`
1. `validate_checkpoint_bundle(checkpoint_root, categories)`:
   - Validates existence, non-emptiness, SHA-256 hash, and metadata verification status for all category models.
2. `validate_dataset(dataset_root, categories)`:
   - Validates presence of `captions/cap.{cat}.train.json`, `captions/cap.{cat}.val.json`, `captions/correction_dict_{cat}.json`, `image_splits/split.{cat}.val.json`, and `resized_image/{cat}/`.
3. `validate_environment()`:
   - Validates environment packages (`torch==1.12.1`, `torchvision==0.13.1`, `open_clip_torch==2.20.0` or adapted runtime) and verifies CUDA availability.
4. `preflight(checkpoint_root, dataset_root, source_root, categories)`:
   - Runs `validate_source()`, `validate_checkpoint_bundle()`, `validate_dataset()`, and `validate_environment()`.
5. `_run_category(category, checkpoint_root, dataset_root, source_root, observe)`:
   - Enforces pinned module imports (`model.py`, `datasets.py`, `test.py`).
   - Loads trusted whole model, verifies type `native_model.LIMN`, calls `model.cuda().eval()`.
   - Reconstructs OpenCLIP evaluation transform (`visual.image_size=224`, `mean`, `std`).
   - Isolates dataset cache via `isolated_cache_root` to prevent in-place pickle pollution.
   - Executes untouched native `native_test.test()`.
   - If `observe=True`, invokes `observe_category()` and verifies zero parity mismatches via `cross_check_with_native()`.
6. `evaluate_categories(checkpoint_root, dataset_root, source_root, categories, replay)`:
   - Runs categories and aggregates via `macro_aggregate`.
7. `main()`:
   - CLI entrypoint accepting `--checkpoint-root`, `--dataset-root`, `--source-root`, `--category`, `--preflight-only`, and `--output`.

### Upstream Reproducibility & Divergence Check

| Parameter | Pinned Native (`train.py` / `test.py`) | Workbench Replay (`limn.py`) | Divergence Status |
|---|---|---|---|
| **Batch Size** | 32 (`train.py:42`) | 32 (`limn.py:185`) | **MATCH** |
| **Dtype** | Float32 (`test.py:39`) | Float32 (`model.cuda().eval()`) | **MATCH** |
| **Image Transform** | `open_clip.image_transform` (`visual.image_size=224`, Bicubic, CenterCrop 224, Normalize) | Exact recreation via `open_clip.image_transform` (`limn.py:175-178`) | **MATCH** |
| **Device** | CUDA (`imgs.cuda()`, `test.py:39,57`) | CUDA (`model.cuda()`, `limn.py:173`) | **MATCH** |
| **Query Order** | Sequential from `cap.{category}.val.json` | Sequential (`FashionIQ.get_test_data`) | **MATCH** |
| **Masking** | `-10e10` reference score suppression | `-10e10` (native `test.py:82`) | **MATCH** |

No computational divergence exists between native `test()` and workbench replay.

---

## G. OBSERVATION CAPABILITY

- **Verdict Axis**: `QUERY_EXPORT_IMPLEMENTABLE`

### Upstream Limitation
In `workbench/third_party/LIMN/LIMN/test.py:85-97`:
- Line 85 computes `nn_result = [np.argsort(-sims[i, :])[:50] for i in range(sims.shape[0])]`.
- Line 95-97 returns ONLY `out = [('{}_r{}'.format(name, k), r)]`.
- The full similarity matrix `sims` and ranking list `nn_result` are thrown away. Upstream native code has no parameter, hook, or return value to export per-query rankings.

### Observation Point Specification
To observe the native computation without altering metric calculation:
1. **Attachment Point**: Immediately after line 85 in `test.py:test()` (after reference masking at line 82 and argsort at line 85).
2. **Variable Identities**:
   - `test_queries[i]['source_img_id']`: Integer index of reference image into `split.{category}.val.json`.
   - `test_queries[i]['target_img_id']`: Integer index of ground-truth target image into `split.{category}.val.json`.
   - `test_queries[i]['mod']['str']`: Processed modification text string (`"{c0} and {c1}"`).
   - `test_targets_id`: Ordered list of gallery integer indices into `split.{category}.val.json`, constructed in `datasets.py:158-163` as the unique union of references and targets in first-seen query order.
   - `sims[i, test_targets_id.index(t['source_img_id'])] = -10e10`: The exact reference masking statement (line 82).
3. **Annotation Index Mapping**:
   Query `i` in `test_queries` and `nn_result` maps 1:1 to annotation index `i` in `captions/cap.{category}.val.json`. `datasets.py:144` iterates `for idx in range(len(ref_captions))` sequentially without filtering or reordering.

### Canonical Image ID Mapping
- `datasets.py:137` loads `images = json.load(open('image_splits/split.{category}.val.json'))`.
- `images` is a list of canonical FashionIQ ASINs (`["B00...", ...]`).
- To recover canonical identities:
  - Reference ASIN: `images[test_queries[i]['source_img_id']]`
  - Target ASIN: `images[test_queries[i]['target_img_id']]`
  - Retrieved Gallery ASIN at rank $r$: `images[test_targets_id[gallery_index]]`

### Caption Recovery & Distinction
- `mod_str` in `test_queries[i]['mod']['str']` is computed via `concat_text`:
  1. ASCII punctuation stripped via `str.maketrans`.
  2. Lowercased and tokenized.
  3. Spell-corrected via `correction_dict_{category}.json`.
  4. Joined with `" and "`.
- The raw captions `[cap0, cap1]` are NOT stored inside `test_queries`.
- Because query index `i` maps 1:1 to annotation index `i`, raw captions are fully recoverable by loading `captions/cap.{category}.val.json` and reading `ref_captions[i]['captions']`.

### Workbench Observation Modules
The repository includes dedicated modules for per-query extraction:
- `workbench/replay/limn_observer.py`:
  - `observe_category()`: Executes identical native model calls (`model.compose_query`, `model.compose_target`), rebuilds `sims`, applies `-10e10` reference mask, and runs `rank_queries()`.
  - Attaches canonical image IDs (`_image_ids`), raw captions (`_raw_captions`), and model input text.
  - `cross_check_with_native()`: Asserts that reconstructed Recall@1, Recall@10, and Recall@50 hit counts match native `test()` metrics exactly.
- `workbench/replay/publication.py`:
  - `build_result_run()`: Assembles schema-v2 `ResultRun` with `QueryResult` records and `top_results` rankings.
- `workbench/replay/limn.py:replay_category_with_observation()`:
  - Invokes native `test()` and `observe_category()`, verifies parity, and returns observed queries.

### Integration Gap
- `workbench/scripts/evaluate_models.py:1453` hardcodes `"per_query_export_available": False`.
- `workbench/replay/limn.py:main()` does not expose an `--observe` CLI flag or invoke `replay_category_with_observation`.
- Therefore, query export capability is fully implemented and verified at the module level (`limn_observer.py`), but remains to be wired into the CLI and production evaluation pipeline.

---

## H. DEFECTS

### 1. `CONFIRMED_BUG` — Missing Trailing Slash Concatenation in `datasets.py:44`
- **Location**: `workbench/third_party/LIMN/LIMN/datasets.py:44`
- **Line**: `self.image_dir = self.path + 'resized_image'`
- **Precondition**: `path` passed without a trailing slash (e.g. `path='/data/fashioniq'`).
- **Actual Behavior**: Resolves to `/data/fashioniqresized_image`, causing `FileNotFoundError`.
- **Expected Behavior**: Path joined using `os.path.join(self.path, 'resized_image')`.
- **Workbench Mitigation**: `workbench/replay/limn.py:181` explicitly passes `path=f"{cache_root}/"`.

### 2. `CONFIRMED_BUG` — Unsafe In-Tree Pickle Caching in `datasets.py:58-64`
- **Location**: `workbench/third_party/LIMN/LIMN/datasets.py:58-64`
- **Lines**:
  ```python
  save_obj(self.test_queries, os.path.join(self.path, 'test_queries_{}.pkl'.format(self.name)))
  save_obj(self.test_targets, os.path.join(self.path, 'test_targets_{}.pkl'.format(self.name)))
  ```
- **Precondition**: `self.path` is read-only (standard in production evaluation environments) or contains unverified pre-existing pickle files.
- **Actual Behavior**: Fails with `PermissionError` when saving, or loads arbitrary unverified pickle binaries from disk via `pickle.load(f)`.
- **Workbench Mitigation**: `workbench/replay/limn.py:179` runs under `isolated_cache_root(dataset_root)` (`common.py:93-103`), symlinking only raw inputs (`captions`, `image_splits`, `resized_image`) into an ephemeral directory `/tmp/cir-replay-cache-...`. Pickles are written to temp storage and discarded on exit; canonical dataset is never written to.

### 3. `CONFIRMED_BUG` — Flat vs Per-Category Image Directory Discrepancy
- **Location**: `workbench/third_party/LIMN/README.md:51` vs `workbench/third_party/LIMN/LIMN/datasets.py:112, 121`
- **Precondition**: Operator resizes images into flat `resized_image/` following README instructions.
- **Actual Behavior**: `datasets.py:112` expects `resized_image/{category}/{image_id}.jpg`. Flat directory raises `FileNotFoundError`.
- **Workbench Mitigation**: `preparation_contracts.yaml:285` enforces `layout: per_category`.

### 4. `EXTERNAL_BLOCKER` — Historical PyTorch Pin Incompatible with sm_120 (Blackwell)
- **Location**: `workbench/registry/models.yaml:936-938`
- **Precondition**: Execution on NVIDIA RTX 50-series (Blackwell / sm_120).
- **Actual Behavior**: `torch==1.12.1` contains no sm_120 PTX/SASS kernels and cannot run on Blackwell GPUs.
- **Workbench Mitigation**: `models.yaml:945-953` defines an explicit `adapted_runtime` (`pytorch: '>=2.7,<3'`, `torchvision: '>=0.22,<1'`, `cuda: '12.8'`, `open_clip_torch>=2.24,<2.27'`).

### 5. `EXTERNAL_BLOCKER` — Unprovisioned Execution Environment on Workstation
- **Location**: `workbench/registry/models.yaml:933` (`WORKBENCH_LIMN_PYTHON`)
- **Precondition**: Running `pc_readiness` on local workstation without dedicated conda env.
- **Observed Behavior**: `python3 -m workbench.scripts.pc_readiness --model limn` outputs:
  `BLOCKED_ENVIRONMENT: execution environment not verified: WORKBENCH_LIMN_PYTHON=None [UNCONFIGURED/UNCONFIGURED]`
- **Impact**: Full runtime inference is blocked until `WORKBENCH_LIMN_PYTHON` is provisioned with required dependencies.

### 6. `BY_DESIGN` — Arbitrary Code Execution Risk in Whole-Model Checkpoint Pickles
- **Location**: `workbench/third_party/LIMN/LIMN/train.py:263` (`torch.save(model, ...)`)
- **Risk**: Deserializing unchecked PyTorch whole-model checkpoints via `torch.load` can execute arbitrary code.
- **Workbench Mitigation**: `workbench/replay/common.py:load_trusted_whole_model` computes the SHA-256 digest and verifies it against the registry before calling `torch.load`.

### 7. `REJECTED_FINDING` — Nearest-Neighbor vs LANCZOS Resampling Divergence
- **Claim**: Previous review hypothesized that workbench `prepare_model.py:92` (`RESAMPLING = "LANCZOS"`) diverged from FashionIQ `start_kit/resize_images.py`.
- **Evidence**: `start_kit/resize_images.py:10` specifies `Image.ANTIALIAS`, which PIL maps to `Image.LANCZOS`.
- **Verdict**: Rejected. The workbench resampling filter matches upstream.

---

## I. EVIDENCE

1. **Commit SHA Verification**:
   ```bash
   git -C workbench/third_party/LIMN rev-parse HEAD
   # Output: 7d7bc9b116f594a65ac22457491edf28a88d3c3e
   ```
2. **Git Cleanliness Check**:
   ```bash
   .venv/bin/python -c "from pathlib import Path; from workbench.replay.limn import validate_source; print(validate_source(Path('workbench/third_party/LIMN')))"
   # Output: []
   ```
3. **Checkpoint Bundle SHA-256 & Size Check**:
   ```bash
   .venv/bin/python -c "from pathlib import Path; from workbench.replay.limn import validate_checkpoint_bundle; print(validate_checkpoint_bundle(Path('workbench/artifacts/checkpoints/limn')))"
   # Output: []
   ```
4. **Dataset Paths Validation**:
   ```bash
   .venv/bin/python -c "from workbench.backend.operator_config import resolve_config; from workbench.replay.limn import validate_dataset; print(validate_dataset(resolve_config().FASHIONIQ_ROOT))"
   # Output: []
   ```
5. **PC Readiness Execution**:
   ```bash
   python3 -m workbench.scripts.pc_readiness --model limn
   # Output: summary: {"BLOCKED_ENVIRONMENT": 1}
   ```

---

## J. VERDICT

- **SOURCE**: `SOURCE_AUDITED`
- **PROTOCOL**: `PROTOCOL_VERIFIED`
- **COMMAND**: `COMMAND_AUDITED`
- **CHECKPOINT**: `CHECKPOINT_VERIFIED`
- **QUERY EXPORT**: `QUERY_EXPORT_IMPLEMENTABLE`
- **RELEASE READINESS**: `BLOCKED_EXTERNAL_DEPENDENCY`
