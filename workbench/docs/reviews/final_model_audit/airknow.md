# Final Model Audit: airknow

## A. MODEL IDENTITY

- **Model ID**: `airknow` (`workbench/registry/models.yaml:97`)
- **Method Name**: `Air-Know` (`workbench/registry/models.yaml:107`)
- **Research Generation**: `recent` (`workbench/registry/models.yaml:108`)
- **Publication Year**: `2026` (`workbench/registry/models.yaml:109`)
- **Paper Title**: `Air-Know: Arbiter-Calibrated Knowledge-Internalizing Robust Network for Composed Image Retrieval` (`workbench/registry/models.yaml:111`)
- **Paper URL**: `https://arxiv.org/abs/2604.19386` (`workbench/registry/models.yaml:112`; HTML: `https://arxiv.org/html/2604.19386v1`)
- **Publication Venue**: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2026) (`workbench/third_party/AirKnow/README.md:5,49`)
- **Authors**: Zhiheng Fu, Yupeng Hu, Qianyun Yang, Shiqi Zhang, Zhiwei Chen, Zixu Li (School of Software, Shandong University) (`workbench/third_party/AirKnow/README.md:7-19`)
- **Upstream Repository URL**: `https://github.com/iLearn-Lab/CVPR26-Air-Know` (`workbench/registry/models.yaml:113`)
- **Upstream Default Branch**: `main` (`workbench/registry/models.yaml:114`)
- **Upstream Pinned Commit SHA**: `6ab6fa95b746f85b08eea4cb66eb51448bce99b7` (`workbench/registry/models.yaml:115`)
- **Source Directory**: `AirKnow` (`workbench/registry/models.yaml:117`)
- **Source Available**: `true` (`workbench/registry/models.yaml:116`)
- **Native Protocol**: `fashioniq_full_gallery_ref_excluded` (`workbench/registry/models.yaml:121`)
- **Supported Protocols**: `[fashioniq_full_gallery_ref_excluded]` (`workbench/registry/models.yaml:122`)
- **Literature Split Label**: `original` (`workbench/registry/models.yaml:120`)
- **Preparation Contract**: `manual_ilearn_resized` (`workbench/registry/models.yaml:110`)
- **FashionIQ Layout**: `fashioniq_ilearn_resized` (`workbench/registry/models.yaml:118`)
- **FashionIQ Image Extensions**: `[".jpg"]` (`workbench/registry/models.yaml:119`)
- **Reported Metrics**: `reported_r10: 55.75`, `reported_r50: 75.72`, `reported_mean: 65.73` (`workbench/registry/models.yaml:123-125`)
- **Reproduction Status**: `NOT_RUN` (`workbench/registry/models.yaml:126`)
- **Strict Clean Checkpoint Available**: `false` (`workbench/registry/models.yaml:127`)

### Pinned Commit Verification
```bash
$ git -C workbench/third_party/AirKnow rev-parse HEAD
6ab6fa95b746f85b08eea4cb66eb51448bce99b7
$ git -C workbench/third_party/AirKnow remote -v
origin	git@github.com:iLearn-Lab/CVPR26-Air-Know.git (fetch)
origin	git@github.com:iLearn-Lab/CVPR26-Air-Know.git (push)
$ git -C workbench/third_party/AirKnow log -1 --oneline
6ab6fa9 Update README.md
```
The pinned commit SHA `6ab6fa95b746f85b08eea4cb66eb51448bce99b7` in `workbench/third_party/AirKnow` matches `upstream_commit_sha` in `workbench/registry/models.yaml:115` verbatim. Non-bytecode files are clean with zero diff against upstream commit.

### Checkpoint Variants
The author Hugging Face repository `https://huggingface.co/iLearn-Lab/CVPR26-Air-Know` hosts exactly two model weights:
1. `fiq_n05`:
   - Filename: `airknow_fiq_0.5.pt` (`workbench/registry/models.yaml:131`)
   - Download URL: `https://huggingface.co/iLearn-Lab/CVPR26-Air-Know/resolve/main/airknow_fiq_0.5.pt` (`workbench/registry/models.yaml:132`)
   - Source Type: `author_huggingface` (`workbench/registry/models.yaml:133`)
   - Checkpoint Training Noise Pct: `50` (`workbench/registry/models.yaml:134`)
   - Evaluation Noise Pct: `0` (`workbench/registry/models.yaml:135`)
   - Status: `NO_CLEAN_CHECKPOINT` (`workbench/registry/models.yaml:138`)
   - Mapping Status: `VERIFIED_METADATA` (`workbench/registry/models.yaml:139`)
   - Size: 2,724,056,638 bytes
   - SHA-256: `f4d25ed9e532f2ee3671f6edadb1048e73113246ec056b413bde75dfe3f79bc6`
2. `fiq_n08`:
   - Filename: `airknow_fiq_0.8.pt` (`workbench/registry/models.yaml:141`)
   - Download URL: `https://huggingface.co/iLearn-Lab/CVPR26-Air-Know/resolve/main/airknow_fiq_0.8.pt` (`workbench/registry/models.yaml:142`)
   - Source Type: `author_huggingface` (`workbench/registry/models.yaml:143`)
   - Checkpoint Training Noise Pct: `80` (`workbench/registry/models.yaml:144`)
   - Evaluation Noise Pct: `0` (`workbench/registry/models.yaml:145`)
   - Status: `NO_CLEAN_CHECKPOINT` (`workbench/registry/models.yaml:148`)
   - Mapping Status: `VERIFIED_METADATA` (`workbench/registry/models.yaml:149`)
   - Size: 2,724,056,638 bytes
   - SHA-256: `756fc0bdc43d9e0f0cae13c8a02b0279f64e62092ce465c1e60bda45e179c54e`

**Crucial Checkpoint Finding**: No clean-trained checkpoint (0% noise) is published. `strict_clean_checkpoint_available: false` is accurate.

### Paper Benchmark Reference
Four rows are committed in `workbench/registry/paper_benchmarks/papers.yaml:10-210`:
1. `airknow_fiq_0pct`: Clean condition (0% training noise), Table 1, page 6.
   - Reported: `r10: 55.75`, `r50: 75.72`, `mean: 65.73` (`papers.yaml:36-38`).
   - Category Breakdown: `dress` (51.03, 73.08), `shirt` (56.76, 75.05), `toptee` (59.47, 79.02).
   - Checkpoint: `null` (`NO_CHECKPOINT`).
   - Arithmetic: $(51.03 + 56.76 + 59.47)/3 = 55.7533 \to 55.75$; $(73.08 + 75.05 + 79.02)/3 = 75.7167 \to 75.72$; $(55.75 + 75.72)/2 = 65.735 \to 65.73$.
2. `airknow_fiq_20pct`: 20% training noise, Table 1.
   - Reported: `r10: 55.18`, `r50: 75.71`, `mean: 65.45` (`papers.yaml:94-96`).
   - Checkpoint: `null` (`NO_CHECKPOINT`).
3. `airknow_fiq_50pct`: 50% training noise, Table 1.
   - Reported: `r10: 53.19`, `r50: 74.04`, `mean: 63.61` (`papers.yaml:141-143`).
   - Category Breakdown: `dress` (47.90, 71.09), `shirt` (53.95, 73.09), `toptee` (57.71, 77.94).
   - Checkpoint ID: `fiq_n05` (`VERIFIED`).
4. `airknow_fiq_80pct`: 80% training noise, Table 1.
   - Reported: `r10: 49.23`, `r50: 70.23`, `mean: 59.73` (`papers.yaml:187-189`).
   - Category Breakdown: `dress` (42.52, 66.03), `shirt` (51.31, 70.26), `toptee` (53.86, 74.41).
   - Checkpoint ID: `fiq_n08` (`VERIFIED`).

---

## B. SOURCE COVERAGE

- **Audit Type**: `STATIC_AUDIT` + local script validation.
- **Pinned Tree Inventory** (`workbench/third_party/AirKnow/`):
  ```text
  workbench/third_party/AirKnow/
  ├── README.md                                (369 lines: claims, environment, quick start)
  ├── LICENSE                                  (Apache 2.0 license)
  ├── datasets.py                              (1041 lines: FashionIQ, FashionIQ_V, CIRR dataset loaders)
  ├── data_utils.py                            (382 lines: TargetPad, SquarePad transforms)
  ├── test_BLIP2.py                            (359 lines: FashionIQ test() function, CIRR, Lasco queries)
  ├── train_BLIP2.py                           (366 lines: CLI parser, training loop, evaluation call)
  ├── utils.py                                 (145 lines: logging, dict serialization, running average)
  ├── cirr_sub_BLIP2.py                        (133 lines: CIRR submission formatter and loader)
  ├── assets/                                  (visual assets including results-fiq.png, pipeline.png)
  └── lavis/                                   (LAVIS submodule / fork containing models, tasks, processors)
      ├── models/
      │   ├── blip2_models/
      │   │   ├── blip2.py                     (init_tokenizer, init_Qformer with hardcoded bert paths)
      │   │   └── blip2_cir.py                 (Blip2QformerCir, Proxy network, feature extractors)
      │   └── blip_models/                     (missing blip_cir.py and blip_cir_caption.py)
      └── processors/
          └── blip_processors.py               (BlipCaptionProcessor eval text processor)
  ```

### Evaluation Dependency Graph
For FashionIQ evaluation:
1. `train_BLIP2.py:298`: The training loop calls `test.test(args, model, trainset, category, txt_processors)` inside `train_and_evaluate()`. There is **no standalone CLI or evaluation entrypoint**.
2. `datasets.py:28-225`: `FashionIQ` class loads image splits, captions, correction dictionaries, initializes queries/targets, writes/reads `fashion_iq_data.json` and 6 PKL caches (`test_queries_{cat}.pkl`, `test_targets_{cat}.pkl`).
3. `data_utils.py:92-106`: `targetpad_transform(1.25, 224)` pads images with target ratio 1.25, bicubic resizes to 224, center crops to 224, and normalizes.
4. `test_BLIP2.py:9-177`: `test()` function iterates queries and targets, calls `model.extract_retrieval_compose` and `model.extract_retrieval_target`, performs matrix multiplication, squeezes, takes token maximum, masks reference to `-10e10`, sorts top-50, and calculates Recall@K.
5. `lavis/models/blip2_models/blip2_cir.py:476-528`: `Blip2QformerCir` executes query fusion feature extraction and target visual token projection.

### Inspected vs. Excluded Files
- **Inspected**:
  - `test_BLIP2.py:1-178`: Core evaluation loop, feature aggregation, similarity computation, reference masking, ranking.
  - `train_BLIP2.py:1-125, 268-366`: Training loop, args, dataset loaders, checkpoint save/restore logic.
  - `datasets.py:1-226`: `FashionIQ` dataset class, caching mechanism, `get_test_data()`, `shuffle()` noise logic.
  - `data_utils.py:1-110`: `targetpad_transform`, `squarepad_transform`, `TargetPad` implementation.
  - `cirr_sub_BLIP2.py:1-133`: Whole-model checkpoint loader check (`torch.load(..., weights_only=False)`).
  - `lavis/models/__init__.py:1-50`: Package import declarations and unresolvable dependencies.
  - `lavis/models/blip2_models/blip2.py:1-65`: Hardcoded `/root/bert-base-uncased` paths in `init_tokenizer` and `init_Qformer`.
  - `lavis/models/blip2_models/blip2_cir.py:40-65, 476-530`: `Proxy` dropout layers and feature extraction shapes.
  - `README.md`: Upstream documentation and claims compared against executable source.
- **Excluded**:
  - Unused LAVIS backbones: `alpro`, `albef`, `beats`, `ulip`, `pnp_vqa`, `timesformer`, and multimodal video/audio tasks.

### README Claims vs. Executable Source
1. **Dataset Layout**:
   - *README Claim* (`README.md:148-169`): FashionIQ data structured in flat category directories: `FashionIQ/dress/`, `FashionIQ/shirt/`, `FashionIQ/toptee/`.
   - *Executable Source* (`datasets.py:34, 155, 166`): Source code unconditionally looks for `resized_image/{category}/{id}.jpg`. Flat canonical category folders fail immediately.
2. **Correction Dictionaries**:
   - *README Claim*: Never mentions spelling correction dictionaries anywhere in the documentation.
   - *Executable Source* (`datasets.py:187`): `with open(os.path.join(self.caption_dir, 'correction_dict_{}.json'.format(name)), 'r') as f:` unconditionally opens correction dictionaries, raising `FileNotFoundError` if absent.
3. **CIRR Test Script**:
   - *README Claim* (`README.md:233`): `python src/cirr_test_submission.py checkpoints/cirr_noise0.8/`.
   - *Executable Source*: Directory `src/` does not exist. The script is located at root as `cirr_sub_BLIP2.py`.
4. **Environment Name**:
   - *README Claim* (`README.md:110`): `conda create -n conesep python=3.8` (verbatim copy-pasted from sibling repo `ConeSep`).
5. **Evaluation Command**:
   - *README Claim*: Mentions only CIRR submission generation; provides zero commands or instructions for evaluating FashionIQ.

---

## C. NATIVE EVALUATION

- **Verdict Axis**: `PROTOCOL_VERIFIED`
- **Native Protocol**: Protocol B (`fashioniq_full_gallery_ref_excluded`)

### 1. Gallery Construction Proof
Source code proof that the gallery is the FULL ordered validation split (not a pair union):
In `workbench/third_party/AirKnow/datasets.py:31`:
```python
self.split = 'original-split'
```
In `workbench/third_party/AirKnow/datasets.py:183-184`:
```python
with open(os.path.join(self.split_dir, "split.{}.{}.json".format(name, 'val')), 'r') as f:
    images = json.load(f)
```
In `workbench/third_party/AirKnow/datasets.py:218-224`:
```python
elif self.split == 'original-split':
    for id, image_name in enumerate(images):
        test_targets_id.append(id)
        out = {}
        out['target_img_id'] = id
        out['target_img_data'] = self.get_img(name + '_' + image_name)      
        test_targets.append(out)
```
`images` is loaded directly from `split.{category}.val.json`. `test_targets` iterates over every image in `images` in exact file order (`for id, image_name in enumerate(images)`). The alternate branch `val-split` (`datasets.py:206-217`) which builds a pair-union is bypassed because `self.split` is hardcoded to `'original-split'`.

### 2. Reference Exclusion Proof
Source code proof that reference images are excluded by score masking:
In `workbench/third_party/AirKnow/test_BLIP2.py:147-151`:
```python
test_targets_id = []
for i in test_targets:
    test_targets_id.append(i['target_img_id'])
for i, t in enumerate(test_queries):
    sims[i, test_targets_id.index(t['source_img_id'])] = -10e10
```
- **Masking Line**: `test_BLIP2.py:151`.
- **Masking Value**: `-10e10` (i.e., $-1 \times 10^{11}$).
- **Masking Mechanism**: The index of the source image in the gallery (`test_targets_id.index(t['source_img_id'])`) is set to `-10e10` **before** the sorting line (`test_BLIP2.py:154`):
  ```python
  nn_result = [np.argsort(-sims[i, :])[:50] for i in range(sims.shape[0])]
  ```
  The reference image is never removed from the candidate gallery; its similarity score is suppressed to guarantee exclusion from top-K.

### 3. Similarity, Scoring, and Tensor Shapes
In `test_BLIP2.py:142-144`:
```python
sims = np.matmul(all_queries, all_imgs) # all_queries.dot(all_imgs.T)
sims = sims.squeeze()
sims = sims.max(-1)
```
- `all_queries`: Output of `model.extract_retrieval_compose(imgs, mods)`, returning `(N, 1, 1, 256)` normalized feature vector (`blip2_cir.py:503`).
- `all_imgs`: Output of `model.extract_retrieval_target(imgs)`, returning `(M, 256, 32)` normalized target tokens (`blip2_cir.py:526`).
- `np.matmul(all_queries, all_imgs)` broadcasts to `(N, M, 1, 32)`.
- `.squeeze()` yields `(N, M, 32)`.
- `.max(-1)` performs max-pooling across the 32 query tokens, producing an `(N, M)` 2D matrix representing query-to-gallery similarity scores.

### 4. Sorting and Top-K Recall
In `test_BLIP2.py:168-175`:
```python
nn_result = [np.argsort(-sims[i, :])[:50] for i in range(sims.shape[0])]
for k in [1, 10, 50]:
    r = 0.0
    for i, nns in enumerate(nn_result):
        if test_targets_id.index(test_queries[i]['target_img_id']) in nns[:k]:
            r += 1
    r = 100 * r / len(nn_result)
    out += [('{}_r{}'.format(name, k), r)]
```
- **Sorting**: `-sims[i, :]` sorts in descending order using NumPy's default sorting.
- **Top-K**: Evaluated at $K \in \{1, 10, 50\}$.
- **Recall Definition**: Hit indicator checking if the target image's gallery index is within `nns[:k]`.
- **Denominator**: `len(nn_result)`, which is equal to `len(test_queries)` (the total number of queries in that category).

### 5. Category Handling & Macro Aggregation
- Evaluated category by category: `dress`, `shirt`, `toptee` (`test_BLIP2.py:82-87`).
- Native output per category is a list of tuples: `[('{name}_r1', r1), ('{name}_r10', r10), ('{name}_r50', r50)]`.
- **Native Macro Average**: The upstream code computes **no macro average**. During training (`train_BLIP2.py:300`), it simply sums `t[1][1]` (R@10) across categories for early stopping.
- **Paper Macro Average**: The paper reports the arithmetic mean across the three categories:
  $$\text{Macro R@10} = \frac{R_{10}(\text{dress}) + R_{10}(\text{shirt}) + R_{10}(\text{toptee})}{3}$$
  $$\text{AVG.} = \frac{\text{Macro R@10} + \text{Macro R@50}}{2}$$
  Workbench `src/evaluation/fashioniq.py:223-240` implements this exact aggregation.

---

## D. NOISE CONDITION AUDIT

### 1. Training Noise vs. Evaluation Noise
In `workbench/registry/models.yaml:134-135, 144-145`:
- `checkpoint_training_noise_pct`: `50` (for `fiq_n05`), `80` (for `fiq_n08`).
- `evaluation_noise_pct`: `0`.
- `literature_split_label`: `original`.

### 2. Is the Checkpoint Genuinely Trained with 50% Noise?
**YES.**
- The model file `airknow_fiq_0.5.pt` corresponds to `--noise_ratio 0.5`.
- In `train_BLIP2.py:61`: `parser.add_argument('--noise_ratio', type=float, default=0.2, help='noise_ratio')`.
- In `datasets.py:66-101` (`FashionIQ.shuffle()`):
  ```python
  shuffle_indices = random.sample(range(num_samples), int(self.noise_ratio * num_samples))
  # splits into candidates, captions, targets and randomly shuffles elements among shuffle_indices
  ```
  This synthetically corrupts 50% of the training triplets to simulate noisy triplet correspondence (NTC).

### 3. Is Evaluation Performed on Clean Validation Data?
**YES.**
- `test_BLIP2.py` does not contain any calls to `shuffle()`.
- `FashionIQ.shuffle()` only modifies `self.fashioniq_data`, `self.labels`, and `self.qlabels` (training triplets).
- `get_test_data()` (`datasets.py:181-225`) loads queries and targets directly from `split.{cat}.val.json` and `cap.{cat}.val.json`.
- Validation data is strictly clean (0% noise).

### 4. Does the Paper's Benchmark Row Correspond to that Training-Noise Condition?
**NO.**
- The paper Table 1 reports four separate blocks for FashionIQ:
  - 0% noise block: Average R@10 = 55.75, R@50 = 75.72, AVG. = 65.73.
  - 20% noise block: Average R@10 = 55.18, R@50 = 75.71, AVG. = 65.45.
  - 50% noise block: Average R@10 = 53.19, R@50 = 74.04, AVG. = 63.61.
  - 80% noise block: Average R@10 = 49.23, R@50 = 70.23, AVG. = 59.73.
- `models.yaml:123-125` transcribes `reported_r10: 55.75`, `reported_r50: 75.72`, `reported_mean: 65.73` — which is the **0% TRAINING NOISE** row!
- However, the authors published **NO 0% checkpoint** on Hugging Face (only 50% and 80% exist).
- Evaluating `fiq_n05` (50% noise) against `reported_r10: 55.75` would compare a noise-trained checkpoint against the clean-trained benchmark row.

### 5. Workbench Conflation Risk
- `workbench/scripts/evaluate_models.py:169-173` correctly uses `benchmark_references.select_reference` with `checkpoint_training_noise_pct=50`, selecting `airknow_fiq_50pct` (53.19 / 74.04 / 63.61).
- **CONFLATION TRAP in Autopilot**: `workbench/backend/autopilot.py:620-622` and `780-781` reads `model.get("reported_r10")` from `models.yaml`, and computes `measured_difference` (lines 814-828) directly against `reported_r10: 55.75` and `reported_mean: 65.73`, ignoring the checkpoint's training noise condition. Any autopilot run on `airknow_fiq_0.5.pt` would falsely report deviation from the 0% row.

---

## E. PREPARATION & DATASET CONTRACT

- **Preparation Contract**: `manual_ilearn_resized` (`models.yaml:110`)
- **Dataset Layout**: `fashioniq_ilearn_resized` (`models.yaml:118`)

### 1. Required Files and Directory Layout
Traced from `workbench/third_party/AirKnow/datasets.py:33-36, 183-188`:
- `image_splits/split.{dress,shirt,toptee}.val.json`
- `captions/cap.{dress,shirt,toptee}.val.json`
- `captions/correction_dict_{dress,shirt,toptee}.json`
- `resized_image/{dress,shirt,toptee}/{id}.jpg`
- Path concatenation caveat: `datasets.py:33-36` uses `self.image_dir = self.path + 'resized_image'`, requiring the data path parameter to end with a trailing slash `/`.

### 2. Spelling Correction Dictionaries
- Required by `datasets.py:187`: `captions/correction_dict_{category}.json`.
- Not bundled in the pinned tree (0 files match in `AirKnow/`).
- Not hosted on Hugging Face.
- Bundled copies exist in `workbench/third_party/TGCIR/correction_dict/` with identical contents and SHA-256 hashes (`9e4a5987...`, `ca29e29e...`, `3dfbf15f...`).

### 3. Resized Images
- Required by `datasets.py:155, 166`: `resized_image/{category}/{id}.jpg`.
- Pinned source contains **no resize generator script**.

### 4. Unversioned Caching Hazard
- In `datasets.py:38-62`:
  ```python
  if not os.path.exists(os.path.join(self.path, 'fashion_iq_data.json')):
      # process and save fashion_iq_data.json + 6 PKLs
  else:
      # unconditionally load fashion_iq_data.json + 6 PKLs
  ```
  If `fashion_iq_data.json` exists in `data/fashionIQ_dataset/`, the dataset constructor never re-reads the split files or correction dictionaries. Any modification to annotations is silently ignored.

### 5. Stochastic Preprocessing & Training Mode Bug
- Image transforms (`data_utils.py:92-106`) are deterministic.
- **CRITICAL EVALUATION DEFECT**: In `test_BLIP2.py:10`:
  ```python
  def test(params, model, testset, category, txt_processors):
      model.train()
      with torch.no_grad():
          if category == 'fashion200k':
              model.eval()
  ```
  For FashionIQ (`dress`, `shirt`, `toptee`), `model.eval()` is **never called**! The model remains in `model.train()` throughout FashionIQ evaluation.
  - Active dropout in Q-Former Proxy network (`blip2_cir.py:58, 61`: `nn.Dropout(p=0.1)`).
  - Active dropout in BERT Q-Former encoder layers.
  This introduces non-deterministic evaluation results and violates standard evaluation protocol.

---

## F. CHECKPOINT INVENTORY & LOADABILITY

- **Verdict Axis**: `CHECKPOINT_BLOCKED`

### 1. Published Checkpoints
- `airknow_fiq_0.5.pt` (50% noise, 2,724,056,638 bytes, SHA-256 `f4d25ed9e532f2ee3671f6edadb1048e73113246ec056b413bde75dfe3f79bc6`).
- `airknow_fiq_0.8.pt` (80% noise, 2,724,056,638 bytes, SHA-256 `756fc0bdc43d9e0f0cae13c8a02b0279f64e62092ce465c1e60bda45e179c54e`).
- Published on Hugging Face: `https://huggingface.co/iLearn-Lab/CVPR26-Air-Know`.

### 2. Checkpoint Serialization & Loader Status
- Upstream saves models using whole-model serialization:
  - `train_BLIP2.py:303`: `torch.save(model, os.path.join(args.model_dir, f'model_epoch_{epoch}.pt'))`.
  - `train_BLIP2.py:366`: `torch.save(current_model, os.path.join(args.model_dir, 'best_model.pt'))`.
- In `train_BLIP2.py:67`, `--restore_file` is defined in argparse but **never referenced in code**.
- No `torch.load` or `load_state_dict` exists in `train_BLIP2.py` or `test_BLIP2.py`.
- In `cirr_sub_BLIP2.py:125`, loading uses `torch.load(path, weights_only=False)`.
- Whole-model pickle loading of untrusted files violates workbench security constraints.

---

## G. WORKBENCH INTEGRATION AUDIT

- **Verdict Axis**: `COMMAND_BLOCKED`

### 1. Registry & Preparation Contract Bug
- `workbench/registry/models.yaml:110` declares `preparation_contract: manual_ilearn_resized`.
- `workbench/registry/preparation_contracts.yaml` has **no entry for airknow** and no definition of `manual_ilearn_resized`.
- When `workbench.scripts.prepare_model` runs:
  ```bash
  $ python3 -m workbench.scripts.prepare_model --model airknow --json
  {"automation_policy": "NONE_REQUIRED", "contract_type": "standard_fashioniq", "deterministic": "N/A", "fully_prepared": true, "model_id": "airknow", "notes": "Standard FashionIQ layout required; no custom generated artifacts.", "preparation_id": null}
  ```
  Because `preparation_contract_for_model(model)` returns `None`, `prepare_model.py` silently falls back to `standard_fashioniq`, claiming `fully_prepared: true` and `automation_policy: NONE_REQUIRED`, despite the model requiring custom resized images and correction dictionaries!

### 2. Auxiliary Asset Registry
- `workbench/registry/auxiliary_assets.yaml` has **0 rows** for `airknow`.
- The three correction dictionaries, base BLIP-2 pretrain weights (`blip2_pretrained.pth`), EVA-ViT-g weights (`eva_vit_g.pth`), and BERT tokenizer are completely unindexed.

### 3. Adapter Implementation
- `AirKnowAdapter` in `workbench/backend/adapters/models.py:54-56`:
  ```python
  class AirKnowAdapter(OfficialScriptAdapter):
      model_id = "airknow"
      script = "test_BLIP2.py"
  ```
- Does not implement `command()`. Calling `AirKnowAdapter().command()` raises `NotImplementedError: upstream evaluation command requires audit before execution` (`models.py:25`).

### 4. Doctor Output
```bash
$ python3 -m workbench.scripts.doctor --model airknow
[OK] repository: repository information
[OK] python: 3.14.4
[OK] registry: registry parsed
[OK] source:airknow: source pin ready
[BLOCKED] runtime:airknow: command unavailable
[OK] external-assets:airknow: required evaluation assets resolved
[BLOCKED] command:airknow: COMMAND_AUDITED
[OK] environment:airknow: UNCONFIGURED
[BLOCKED] doctor
```
Note: `[BLOCKED] command:airknow: COMMAND_AUDITED` is a confusing diagnostic produced because `command_status` is `None` in `models.yaml`, defaulting to `"COMMAND_AUDITED"` in `doctor.py:169`.

### 5. PC Readiness Classification
```bash
$ python3 -m workbench.scripts.pc_readiness --model airknow --json
{
  "summary": {"by_state": {"BLOCKED_SOURCE": 1}},
  "models": {
    "airknow": {
      "state": "BLOCKED_SOURCE",
      "reason": "source cannot run: no runnable evaluation path is registered for this model: the pinned evaluator is a callable with no CLI or checkpoint loader...",
      "blockers": [
        "source cannot run: no runnable evaluation path is registered...",
        "command not runnable: COMMAND_UNAUDITED",
        "checkpoint missing: .../airknow_fiq_0.5.pt",
        "checkpoint missing: .../airknow_fiq_0.8.pt"
      ]
    }
  }
}
```
`pc_readiness.py` matches needle `"source cannot run"` at the top of `BLOCKER_KINDS` and classifies `airknow` as `BLOCKED_SOURCE`. This is correct.

### 6. Metrics Parser Status
In `workbench/backend/metrics_extraction.py:402-422`:
`airknow` is not handled in `extract_aggregate_metrics()`, unconditionally falling through to:
`extraction_status: "AGGREGATE_PARSER_UNAVAILABLE"`, `parity_status: "PARSER_UNAVAILABLE"`.

---

## H. OBSERVATION CAPABILITY & PER-QUERY EXPORT

- **Verdict Axis**: `QUERY_EXPORT_IMPLEMENTABLE`

### 1. Score & Ranking Production Point
- Query and target feature extraction: `test_BLIP2.py:113, 129`.
- Matrix multiplication: `test_BLIP2.py:142` (`sims = np.matmul(all_queries, all_imgs)`).
- Token reduction: `test_BLIP2.py:144` (`sims = sims.max(-1)`).
- Reference exclusion: `test_BLIP2.py:151` (`sims[i, test_targets_id.index(t['source_img_id'])] = -10e10`).
- Ranking: `test_BLIP2.py:154` (`nn_result = [np.argsort(-sims[i, :])[:50] for i in range(sims.shape[0])]`).

### 2. Per-Query Export Feasibility
- `nn_result` at line 154 holds the top-50 gallery indices for each query.
- Each query index $i$ corresponds to `test_queries[i]['source_img_id']` and `test_queries[i]['target_img_id']`.
- Outputting a structured JSON with query ID, target rank, and top-50 retrieved candidates is directly implementable by hooking into `test_BLIP2.py:154`.
- Current status: Blocked by lack of runnable execution pipeline.

---

## I. DEFECT INVENTORY & EVIDENCE

### 1. `CONFIRMED_BUG` — Missing Modules Prevent `lavis.models` Package Import
- **Location**: `workbench/third_party/AirKnow/lavis/models/__init__.py:33-34`
- **Precondition**: `import lavis.models` or `from lavis.models import load_model_and_preprocess`.
- **Actual**: `ModuleNotFoundError: No module named 'lavis.models.blip_models.blip_cir'`. Lines 33–34 import `BlipCir` and `BlipCirCaption`, but `blip_cir.py` and `blip_cir_caption.py` were never committed to the upstream repository.
- **Expected**: Modules should be present or imports guarded. Note that sibling checkouts (`ConeSep`, `HABIT`, `INTENT`) ship both `.py` files.
- **Severity**: P0 Blocker (prevents running any code in `AirKnow`).

### 2. `CONFIRMED_BUG` — FashionIQ Evaluator Runs in Training Mode (`model.train()`)
- **Location**: `workbench/third_party/AirKnow/test_BLIP2.py:10`
- **Precondition**: Calling `test()` for FashionIQ categories (`dress`, `shirt`, `toptee`).
- **Actual**: Line 10 executes `model.train()`. Only `fashion200k` calls `model.eval()` (line 13). FashionIQ runs with active dropout in `Proxy` (`blip2_cir.py:58,61`) and BERT layers.
- **Expected**: `model.eval()` should be invoked before evaluation to ensure deterministic feature extraction.

### 3. `CONFIRMED_BUG` — Unused Argument and Missing Checkpoint Loader
- **Location**: `workbench/third_party/AirKnow/train_BLIP2.py:67`
- **Precondition**: Attempting to load an existing FashionIQ checkpoint.
- **Actual**: `--restore_file` is defined in argparse but never read. No checkpoint loading mechanism exists in `train_BLIP2.py` or `test_BLIP2.py`.
- **Expected**: Evaluator should provide a CLI that accepts `--checkpoint` and loads model weights.

### 4. `CONFIRMED_BUG` — Missing Trailing Slash Concatenation Bug
- **Location**: `workbench/third_party/AirKnow/datasets.py:33-36`
- **Precondition**: Passing a dataset path without a trailing slash (e.g. `--fashioniq_path /data/fashionIQ_dataset`).
- **Actual**: Paths are formed by string concatenation: `self.image_dir = self.path + 'resized_image'`, resolving to `/data/fashionIQ_datasetresized_image`.
- **Expected**: Use `os.path.join(self.path, 'resized_image')` or ensure trailing slash.

### 5. `CONFIRMED_BUG` — Missing Preparation Contract in Workbench Registry
- **Location**: `workbench/registry/models.yaml:110` vs `workbench/registry/preparation_contracts.yaml`
- **Precondition**: Running `python3 -m workbench.scripts.prepare_model --model airknow`.
- **Actual**: `manual_ilearn_resized` is not registered in `preparation_contracts.yaml`. `prepare_model.py` defaults to `standard_fashioniq` and falsely reports `fully_prepared: true`.
- **Expected**: Register `airknow_fashioniq` preparation contract with required resized images and correction dictionaries.

### 6. `CONFIRMED_BUG` — Autopilot Evaluates 50% Noise Checkpoints Against 0% Headline Row
- **Location**: `workbench/backend/autopilot.py:620-622, 780-781`
- **Precondition**: Running autopilot on `airknow` with checkpoint `fiq_n05`.
- **Actual**: Autopilot reads `reported_r10: 55.75` (the 0% training noise row) and computes difference against it, ignoring that `fiq_n05` is trained with 50% noise.
- **Expected**: Autopilot should bind to the noise-conditioned reference row `airknow_fiq_50pct` (53.19).

### 7. `CONFIRMED_BUG` — Missing Metrics Parser
- **Location**: `workbench/backend/metrics_extraction.py:402-422`
- **Precondition**: Evaluation completes and stdout is passed to `extract_aggregate_metrics`.
- **Actual**: Unhandled `model_id == "airknow"`, returns `AGGREGATE_PARSER_UNAVAILABLE`.
- **Expected**: Output parser for AirKnow category tuples.

### 8. `EXTERNAL_BLOCKER` — No Clean-Trained Checkpoint Published
- **Location**: Upstream Hugging Face repo `https://huggingface.co/iLearn-Lab/CVPR26-Air-Know`
- **Precondition**: Reproducing the headline Table 1 row (0% noise, 55.75 / 75.72 / 65.73).
- **Actual**: Only 50% (`airknow_fiq_0.5.pt`) and 80% (`airknow_fiq_0.8.pt`) weights are published.
- **Expected**: Authors should publish the 0% clean-trained model weights.

### 9. `EXTERNAL_BLOCKER` — Missing Correction Dictionaries from Published Repo
- **Location**: `workbench/third_party/AirKnow/datasets.py:187`
- **Precondition**: Instantiating `FashionIQ` dataset.
- **Actual**: Pinned code requires `captions/correction_dict_{cat}.json`, but files are not in the repository.
- **Expected**: Bundled dictionaries or published external asset link.

### 10. `UNVERIFIED_RISK` — Absolute Hardcoded Paths for BERT Tokenizer
- **Location**: `workbench/third_party/AirKnow/lavis/models/blip2_models/blip2.py:32, 48, 55`
- **Precondition**: Initializing Q-Former.
- **Actual**: Hardcoded path `"/root/bert-base-uncased"` crashes on systems where `/root` is inaccessible or missing Hugging Face files.
- **Expected**: Parameterized tokenizer name or standard Hugging Face model identifier `bert-base-uncased`.

---

## J. VERDICT AXES & SUMMARY

| Axis | Verdict | Key Evidence |
|---|---|---|
| **SOURCE** | **`BLOCKED_SOURCE`** | Pinned `lavis.models` package crashes on import due to uncommitted `blip_cir.py` and `blip_cir_caption.py` (`lavis/models/__init__.py:33-34`). No standalone CLI or evaluation entrypoint exists. |
| **PROTOCOL** | **`PROTOCOL_VERIFIED`** | Source `datasets.py:31, 218-224` and `test_BLIP2.py:147-151` strictly implement Protocol B (`fashioniq_full_gallery_ref_excluded`: full val split gallery, reference masked to `-10e10` before `argsort`). |
| **COMMAND** | **`COMMAND_BLOCKED`** | Evaluator exists only as a callable function in `test_BLIP2.py:9` invoked during training. `AirKnowAdapter.command()` raises `NotImplementedError`. |
| **CHECKPOINT** | **`CHECKPOINT_BLOCKED`** | No 0% clean-trained checkpoint published by authors (only 50% and 80% exist). In-repo code lacks a checkpoint loader and relies on untrusted whole-model pickle loading. |
| **QUERY EXPORT** | **`QUERY_EXPORT_IMPLEMENTABLE`** | Top-50 candidate indices produced at `test_BLIP2.py:154`, but observation is blocked because model cannot execute. |
| **RELEASE READINESS** | **`BLOCKED_EXTERNAL_DEPENDENCY`** | Blocked by missing upstream files, missing 0% checkpoint, missing correction dictionaries, missing resized images, and unconfigured environment. |
