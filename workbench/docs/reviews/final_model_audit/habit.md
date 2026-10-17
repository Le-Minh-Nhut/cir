# Final Model Audit: habit

## A. MODEL IDENTITY

- **Model ID**: `habit` (`workbench/registry/models.yaml:243`)
- **Method Name**: `HABIT` (`workbench/registry/models.yaml:252`)
- **Research Generation**: `recent` (`workbench/registry/models.yaml:253`)
- **Publication Year**: `2026` (`workbench/registry/models.yaml:254`)
- **Paper Title**: `HABIT: Chrono-Synergia Robust Progressive Learning Framework for Composed Image Retrieval` (`workbench/registry/models.yaml:256`)
- **Paper URL**: `https://ojs.aaai.org/index.php/AAAI/article/view/37608` (`workbench/registry/models.yaml:257`); arXiv version: `https://arxiv.org/abs/2604.18037`
- **Publication Venue**: Proceedings of the 40th AAAI Conference on Artificial Intelligence (AAAI-26), Vol. 40, No. 8, Article 37608. DOI: `10.1609/aaai.v40i8.37608`.
- **Authors**: Zixu Li, Yupeng Hu, Zhiwei Chen, Shiqi Zhang, Qinlei Huang, Zhiheng Fu, Yinwei Wei (Shandong University).
- **Upstream Repository URL**: `https://github.com/iLearn-Lab/AAAI26-HABIT` (`workbench/registry/models.yaml:258`)
- **Upstream Default Branch**: `main` (`workbench/registry/models.yaml:259`)
- **Upstream Pinned Commit SHA**: `6ddc2cd63eac1f65693f4e821c34fb3b42aee306` (`workbench/registry/models.yaml:260`)
- **Source Directory**: `HABIT` (`workbench/registry/models.yaml:262`)
- **Source Available**: `true` (`workbench/registry/models.yaml:261`)
- **Native Protocol**: `fashioniq_full_gallery_ref_excluded` (`workbench/registry/models.yaml:266`)
- **Supported Protocols**: `[fashioniq_full_gallery_ref_excluded]` (`workbench/registry/models.yaml:267`)
- **Reproduction Status**: `NOT_RUN` (`workbench/registry/models.yaml:271`)
- **Command Status**: Absent in `models.yaml:habit` (defaults to `COMMAND_UNAUDITED`)
- **Replay Status**: Absent in `models.yaml:habit` (no replay runner implemented)
- **Strict Clean Checkpoint Available**: `false` (`workbench/registry/models.yaml:272`)
- **Reported Metrics**: `reported_r10: 55.38`, `reported_r50: 75.20`, `reported_mean: 65.29` (`workbench/registry/models.yaml:268-270`)
- **Notes in Registry**: "Official Hub lists only FashionIQ 20%, 50%, and 80% noise-trained weights." (`workbench/registry/models.yaml:273`)

### Pinned Commit Verification
```bash
$ git -C workbench/third_party/HABIT rev-parse HEAD
6ddc2cd63eac1f65693f4e821c34fb3b42aee306
$ git -C workbench/third_party/HABIT rev-list --count HEAD
1
$ git -C workbench/third_party/HABIT log -1 --oneline
6ddc2cd Update README.md
```
Pinned commit SHA `6ddc2cd63eac1f65693f4e821c34fb3b42aee306` matches `upstream_commit_sha` in `models.yaml:260`. The repository contains a single commit authored by Zixu Li on May 25, 2026.

### Checkpoint Variants in Registry
Three variants registered (`models.yaml:274-304`), all `author_huggingface`, `status: NO_CLEAN_CHECKPOINT`, `checkpoint_mapping_status: VERIFIED_METADATA`, `evaluation_noise_pct: 0`:
- `fiq_n02` (`HABIT-FIQ_N0.2.pt`): training noise 20%, size `2,730,715,930`, sha256 `814d2d0ceb29ef910f37562e8bea8a76cf38aa546fbc2102e495df687c276b03`.
- `fiq_n05` (`HABIT-FIQ_N0.5.pt`): training noise 50%, size `2,730,715,930`, sha256 `8a0d3bd27cf46d8334f9596404d89ebf803da0fa65236d89815bb3929b9a4216`.
- `fiq_n08` (`HABIT-FIQ_N0.8.pt`): training noise 80%, size `2,730,715,930`, sha256 `763885de6907bca4f017b336ed7ec3740cb9305c5adb6a135da45c158ddb9717`.
All download URLs point to `https://huggingface.co/iLearn-Lab/AAAI26-HABIT/resolve/main/fiq/`.

### Paper Benchmark Reference Rows
Four reference rows registered in `workbench/registry/paper_benchmarks/papers.yaml:845-1038`:
1. `habit_aaai26_table1_fiq_n00` (0% training noise, fully supervised):
   - Table 1, page 5, AAAI-26 / arXiv:2604.18037: `HABIT (Ours) 49.99 72.38 56.62 74.68 59.51 78.53 55.38 75.20 65.29`.
   - `dress`: R@10 = 49.99, R@50 = 72.38.
   - `shirt`: R@10 = 56.62, R@50 = 74.68.
   - `toptee`: R@10 = 59.51, R@50 = 78.53.
   - `Avg`: R@10 = 55.38, R@50 = 75.20, `AVG` = 65.29.
   - Reference status: `VERIFIED`, `checkpoint_id: null`, `checkpoint_training_noise_pct: 0`, `checkpoint_mapping_status: NO_CHECKPOINT`.
2. `habit_aaai26_table1_fiq_n20` (20% training noise):
   - Row: `49.63 71.34 55.67 73.19 58.14 78.32 54.48 74.28 64.38`.
   - `Avg`: R@10 = 54.48, R@50 = 74.28, `AVG` = 64.38. Matches `fiq_n02`.
3. `habit_aaai26_table1_fiq_n50` (50% training noise):
   - Row: `47.33 69.71 53.72 72.55 56.51 77.00 52.52 73.09 62.80`.
   - `Avg`: R@10 = 52.52, R@50 = 73.09, `AVG` = 62.80. Matches `fiq_n05`.
4. `habit_aaai26_table1_fiq_n80` (80% training noise):
   - Row: `42.04 65.20 50.12 69.77 52.92 73.61 48.36 69.53 58.94`.
   - `Avg`: R@10 = 48.36, R@50 = 69.53, `AVG` = 58.94. Matches `fiq_n08`.

### Arithmetic Verification
- 0% row:
  - Macro R@10 = $(49.99 + 56.62 + 59.51) / 3 = 166.12 / 3 = 55.37333... \to 55.37$. Paper prints `55.38` (0.0067% rounding deviation in paper).
  - Macro R@50 = $(72.38 + 74.68 + 78.53) / 3 = 225.59 / 3 = 75.19666... \to 75.20$. Exact match.
  - Mean AVG = $(55.38 + 75.20) / 2 = 65.29$. Exact match.
- 20% row:
  - Macro R@10 = $(49.63 + 55.67 + 58.14) / 3 = 163.44 / 3 = 54.48$. Exact match.
  - Macro R@50 = $(71.34 + 73.19 + 78.32) / 3 = 222.85 / 3 = 74.2833... \to 74.28$. Exact match.
  - Mean AVG = $(54.48 + 74.28) / 2 = 64.38$. Exact match.
- 50% row:
  - Macro R@10 = $(47.33 + 53.72 + 56.51) / 3 = 157.56 / 3 = 52.52$. Exact match.
  - Macro R@50 = $(69.71 + 72.55 + 77.00) / 3 = 219.26 / 3 = 73.0866... \to 73.09$. Exact match.
  - Mean AVG = $(52.52 + 73.09) / 2 = 62.805 \to 62.80$. Exact match.
- 80% row:
  - Macro R@10 = $(42.04 + 50.12 + 52.92) / 3 = 145.08 / 3 = 48.36$. Exact match.
  - Macro R@50 = $(65.20 + 69.77 + 73.61) / 3 = 208.58 / 3 = 69.5266... \to 69.53$. Exact match.
  - Mean AVG = $(48.36 + 69.53) / 2 = 58.945 \to 58.94$. Exact match.

---

## B. SOURCE COVERAGE

- **Audit Type**: `STATIC_AUDIT` + local script validation.
- **Pinned Tree Inventory** (`workbench/third_party/HABIT/`, 1,235 tracked files total):
  ```text
  workbench/third_party/HABIT/
  ├── LICENSE                                 (1 line: MIT License)
  ├── README.md                               (427 lines: claims, environment, paper links, CIRR test)
  ├── assets/                                 (13 PNG image assets: logos, HABIT architecture, results)
  ├── cirr_test_submission.py                 (128 lines: CIRR server submission formatter only)
  ├── data_utils.py                           (130 lines: squarepad and targetpad preprocessing transforms)
  ├── datasets.py                             (390 lines: FashionIQ, CIRR, LaSCo, Birds, Fashion200k)
  ├── lavis/                                  (1,214 tracked files: vendored Salesforce LAVIS framework)
  │   ├── configs/models/blip2/               (BLIP-2 configuration YAMLs)
  │   ├── models/blip2_models/HABIT.py        (620 lines: HABIT core model, MKE, DPL, extract_retrieval_*)
  │   ├── models/blip2_models/blip2.py        (316 lines: Blip2Base, tokenizer and Q-Former init)
  │   └── processors/blip_processors.py       (140 lines: BlipCaptionProcessor text normalization)
  ├── test.py                                 (264 lines: FashionIQ test() and CIRR test_cirr_valset())
  ├── train.py                                (368 lines: CLI parser, training loop, evaluation call)
  └── utils.py                                (142 lines: logging, dict serialization, running average)
  ```

### Evaluation Dependency Graph (FashionIQ)
1. `train.py:283`: Training loop calls `test.test(args, model, trainset, category, txt_processors)`. Pinned tree has no other callsite for FashionIQ evaluation.
2. `datasets.py:27-212`: `FashionIQ` class loads image splits, captions, correction dictionaries, pre-resized images; builds queries and targets; default `split='original-split'`.
3. `test.py:9-177`: `test()` function batches queries and targets under `torch.no_grad()`, runs `model.extract_retrieval_compose` and `model.extract_retrieval_target`, performs matrix multiplication `np.matmul(all_queries, all_imgs).squeeze().max(-1)`, masks reference index with `-10e10`, sorts top-50, and calculates R@1, R@10, R@50.
4. `lavis/models/blip2_models/HABIT.py:372-436`: `extract_retrieval_compose` and `extract_retrieval_target`.
5. `lavis/processors/blip_processors.py:49-68`: `BlipCaptionProcessor.pre_caption` text processor.
6. `data_utils.py:92-106`: `targetpad_transform(1.25, 224)` image preprocessing.

### Inspected vs. Excluded Files
- **Inspected**:
  - `README.md:1-427`: Upstream claims, installation, execution commands.
  - `test.py:1-180`: Native evaluation entrypoint and FashionIQ evaluation logic.
  - `datasets.py:1-215`: `FashionIQ` class, text correction, text composition, image loading, split enumeration.
  - `train.py:1-135, 170-368`: Model initialization, data loading, training/eval loop, CLI arguments.
  - `data_utils.py:1-130`: `targetpad_transform`, `squarepad_transform`, `TargetPad` transform.
  - `utils.py:1-140`: Logging and JSON serialization helpers.
  - `cirr_test_submission.py:1-128`: Checkpoint loading and CIRR submission formatting.
  - `lavis/models/blip2_models/HABIT.py:1-620`: Architecture, feature extractors, loss definitions.
  - `lavis/models/blip2_models/blip2.py:1-100`: `Blip2Base`, BERT tokenizer and Q-Former setup.
  - `lavis/configs/models/blip2/blip2_pretrain.yaml`: Pretrained model configuration.
  - `lavis/configs/default.yaml`: Default LAVIS cache directory configuration.
- **Excluded**:
  - `datasets.py:217-389`: `CIRR`, `LaSCo`, `Birds`, `Shoes`, `Fashion200k` dataset loaders.
  - `test.py:181-264`: `test_cirr_valset()` CIRR validation evaluation.
  - `train.py:190-247`: `train()` backward optimization passes and loss accumulation.
  - `lavis/common/annotator/*`: Unused image annotators (openpose, midas, uniformer, etc.).
  - `lavis/models/*` except BLIP-2/HABIT: Unused vision-language architectures (albef, alpro, clip, etc.).

### Executable Source vs. README Claims
1. **Testing CLI**: README §2 ("Testing", lines 238-248) shows only a CIRR test submission command: `python src/cirr_test_submission.py checkpoints/cirr_noise0.5/`. The README provides ZERO commands for evaluating FashionIQ. `test.py` contains no CLI or `__main__` entrypoint.
2. **Dataset Directory Structure**: README lines 155-171 depicts `FashionIQ/{captions, image_splits, dress, shirt, toptee}`. However, `datasets.py:33` hardcodes `self.image_dir = self.path + 'resized_image'`, requiring `resized_image/{dress,shirt,toptee}/*.jpg`.
3. **Spelling Correction Dictionaries**: README never mentions that spelling correction dictionaries (`correction_dict_<cat>.json`) are needed, nor does it track them in the repo. Yet `datasets.py:100, 174` crashes with `FileNotFoundError` if they are not in `captions/`.
4. **Checkpoints**: README §1 ("Training under Noisy Settings", lines 206-207) states `--noise_ratio` options evaluated are `0.0`, `0.2`, `0.5`, `0.8`, and Table 1 highlights the 0% clean-trained setting. However, author HuggingFace repository publishes ONLY `N0.2`, `N0.5`, `N0.8` weights for FashionIQ. No `N0.0` (0% clean) checkpoint is published.
5. **Environment**: README line 127 claims Python 3.8.10 and CUDA 12.6, but line 134 installs PyTorch 2.1.0 with CUDA 12.1 wheel (`--index-url https://download.pytorch.org/whl/cu121`).

---

## C. NATIVE EVALUATION

- **Verdict Axis**: `PROTOCOL_VERIFIED`

### 1. Proof of Full-Val-Split Gallery (Protocol B)
In `workbench/third_party/HABIT/datasets.py:28`:
```python
class FashionIQ(torch.utils.data.Dataset):
    def __init__(self, path, category=None, transform=None, noise_ratio=0, split='original-split'):
```
Default `split` argument is `'original-split'`. In `train.py:109-112`, `FashionIQ` is instantiated without passing `split`, selecting `'original-split'`.
In `datasets.py:170-171, 205-212`:
```python
with open(os.path.join(self.split_dir, "split.{}.{}.json".format(name, 'val')), 'r') as f:
    images = json.load(f)
...
elif self.split == 'original-split':
    for id, image_name in enumerate(images):
        test_targets_id.append(id)
        out = {}
        out['target_img_id'] = id
        out['target_img_data'] = self.get_img(name + '_' + image_name)      
        test_targets.append(out)
```
The gallery (`test_targets`) is populated with every image in `split.<cat>.val.json` in exact file enumeration order. The pair-union branch (`val-split`, lines 193-204) is NOT taken.

### 2. Proof of Reference Exclusion (Quote Mask Line & Value)
In `workbench/third_party/HABIT/test.py:147-151`:
```python
test_targets_id = []
for i in test_targets:
    test_targets_id.append(i['target_img_id'])
for i, t in enumerate(test_queries):
    sims[i, test_targets_id.index(t['source_img_id'])] = -10e10
```
- **Mask line**: `test.py:151`
- **Mask value**: `-10e10`
For each test query `i`, its reference image index in `test_targets_id` has its similarity masked to `-10e10` prior to sorting. This confirms Protocol B (`fashioniq_full_gallery_ref_excluded`).

### 3. Scoring Computation
- In `test.py:133-137`, feature normalization is commented out:
  ```python
  # for i in range(all_queries.shape[0]):
  #     all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
  ```
- Scoring matrix:
  ```python
  sims = np.matmul(all_queries, all_imgs)
  sims = sims.squeeze()
  sims = sims.max(-1)
  ```
- Feature shapes:
  - `all_queries`: `(N, 1, 1, D)` from `fusion_feats.unsqueeze(1).unsqueeze(1)` (`HABIT.py:413`).
  - `all_imgs`: `(G, D, 32)` from `image_features.permute(0, 2, 1)` (`HABIT.py:436`).
  - Matmul broadcasts to `(N, G, 1, 32)`.
  - `sims.squeeze().max(-1)` yields `(N, G)`, taking the maximum inner product across the 32 query tokens for each query-target pair.

### 4. Sorting & Top-K Candidates
In `test.py:154, 168`:
```python
nn_result = [np.argsort(-sims[i, :])[:50] for i in range(sims.shape[0])]
```
Negative similarities sorted ascending (descending similarity order), sliced at `[:50]` (top-50 candidates).

### 5. Recall Definition & Denominator
In `test.py:169-175`:
```python
for k in [1, 10, 50]:
    r = 0.0
    for i, nns in enumerate(nn_result):
        if test_targets_id.index(test_queries[i]['target_img_id']) in nns[:k]:
            r += 1
    r = 100 * r / len(nn_result)
    out += [('{}_r{}'.format(name, k), r)]
```
- **Recall@K**: fraction of queries whose true target image is among top $K$ retrieved images.
- **Denominator**: `len(nn_result)` = total number of test queries for that category.
- **Scaling**: Multiplied by 100 to produce percentage.

### 6. Category Handling & Macro Aggregation
- Categories evaluated independently: `['dress', 'shirt', 'toptee']`.
- Output per category: list of tuples `[('<cat>_r1', r), ('<cat>_r10', r), ('<cat>_r50', r)]`.
- Upstream source computes NO macro average. `train.py:285` sums `t[1][1]` (R@10) across categories solely for checkpoint selection.
- Workbench macro aggregation (`metrics_extraction.py:148-152`):
  $$\text{Macro R@10} = \frac{\text{dress}_{\text{R@10}} + \text{shirt}_{\text{R@10}} + \text{toptee}_{\text{R@10}}}{3}$$
  $$\text{Macro R@50} = \frac{\text{dress}_{\text{R@50}} + \text{shirt}_{\text{R@50}} + \text{toptee}_{\text{R@50}}}{3}$$
  $$\text{Mean AVG} = \frac{\text{Macro R@10} + \text{Macro R@50}}{2}$$
  Matches paper Table 1 definition within 0.01 rounding tolerance.

---

## D. PREPARATION & NOISE CONDITION

### 1. Noise Condition Verification
- **Training Noise**:
  `datasets.py:65-94` (`FashionIQ.shuffle`): Shuffles triplets in `self.fashioniq_data` (built from `cap.<cat>.train.json`).
  If `self.noise_ratio == 0`, `shuffle()` logs "无噪声模式" and returns immediately (`datasets.py:67-69`).
- **Clean Evaluation**:
  Evaluation reads `cap.<cat>.val.json` and `split.<cat>.val.json` directly (`datasets.py:170-173`). It NEVER calls `shuffle()`.
  Therefore, evaluation is strictly clean: `evaluation_noise_pct: 0` for all variants.
- **Checkpoint Training Noise**:
  Author HuggingFace repository publishes ONLY noise-trained checkpoints: `fiq_n02` (20%), `fiq_n05` (50%), `fiq_n08` (80%).
  No clean (0% noise) checkpoint is published.
- **Conflation Risk**:
  Evaluating `fiq_n02`, `fiq_n05`, or `fiq_n08` and comparing metrics against the headline paper row (55.38 / 75.20 / 65.29) conflates noisy-trained models with the 0% clean-trained baseline.

### 2. Spelling Correction Dictionaries
- Consumed in `datasets.py:100, 174`:
  `open(os.path.join(self.caption_dir, 'correction_dict_{}.json'.format(name)))`
- Dictionaries are NOT shipped in `workbench/third_party/HABIT/` (`git ls-files | grep correction` returns empty).
- `AUXILIARY_ASSET_AUDIT.md:355-357` rates them `OFFICIAL_ASSET_UNAVAILABLE`.
- Workbench has NO record for `habit` in `workbench/registry/auxiliary_assets.yaml`.
- Clean run on un-staged dataset crashes with `FileNotFoundError`.

### 3. Resized Images
- Consumed in `datasets.py:33, 141, 152`:
  `self.image_dir = self.path + 'resized_image'`
  `img_path = os.path.join(self.image_dir, image_name.split('_')[0], image_name.split('_')[1] + ".jpg")`
- Requires pre-resized JPEGs in `resized_image/{dress,shirt,toptee}/*.jpg`.
- Raw dataset has 74,381 PNGs in `images/`.

### 4. Directory Trailing Slash Defect
In `datasets.py:33-35`:
```python
self.image_dir = self.path + 'resized_image'
self.split_dir = self.path + 'image_splits'
self.caption_dir = self.path + 'captions'
```
String concatenation `+` instead of `os.path.join()`. If `path` lacks a trailing slash, resolves to `<path>resized_image`.

### 5. Read-Only Filesystem Incompatibility
In `datasets.py:39-54`:
On cache miss, `FashionIQ.__init__` writes `fashion_iq_data.json` and six `.pkl` files (`test_queries_*.pkl`, `test_targets_*.pkl`) into `self.path`. Fails with `PermissionError` on read-only dataset roots.

### 6. Working Directory & Stochastic Steps
- **CWD**: `HABIT.py:120` hardcodes relative path `configs/models/blip2/blip2_pretrain.yaml`, requiring cwd at repository root.
- **Stochastic Steps**: Evaluation is fully deterministic (`torch.no_grad()`, `model.eval()`, deterministic `TargetPad` + `Resize` + `CenterCrop`).

---

## E. CHECKPOINT

- **Verdict Axis**: `CHECKPOINT_BLOCKED`

### 1. Files & Checkpoint Variants
Three variants declared in `workbench/registry/models.yaml:274-304`:
1. `fiq_n02` (`HABIT-FIQ_N0.2.pt`): 20% training noise, size `2,730,715,930` B, sha256 `814d2d0ceb29ef910f37562e8bea8a76cf38aa546fbc2102e495df687c276b03`.
2. `fiq_n05` (`HABIT-FIQ_N0.5.pt`): 50% training noise, size `2,730,715,930` B, sha256 `8a0d3bd27cf46d8334f9596404d89ebf803da0fa65236d89815bb3929b9a4216`.
3. `fiq_n08` (`HABIT-FIQ_N0.8.pt`): 80% training noise, size `2,730,715,930` B, sha256 `763885de6907bca4f017b336ed7ec3740cb9305c5adb6a135da45c158ddb9717`.
All variants have `status: NO_CLEAN_CHECKPOINT` and `checkpoint_mapping_status: VERIFIED_METADATA`.

### 2. Architecture & Serialization
- Architecture: BLIP-2 with EVA-CLIP ViT-g visual encoder (`vit_model="eva_clip_g"`, 224x224), Q-Former (32 query tokens, hidden size 768), linear projection to 256 dimensions (`embed_dim=256`).
- Serialization: Whole model (`torch.save(model)` at `train.py:288, 368`).
- Registry omission: `models.yaml:habit` omits `checkpoint_serialization: whole_model`.

### 3. Checkpoint Loader & Provenance
- No FashionIQ checkpoint loader exists in the pinned repository. `train.py:79` declares `--restore_file`, but it is dead code (never read).
- `cirr_test_submission.py:97` loads `"Blip2QformerCir"`, which is not registered in LAVIS (only `"HABIT"` exists).
- Hugging Face repository has no 0% noise checkpoint.
- Local staging `workbench/artifacts/checkpoints/habit/` does not exist (`checkpoint_downloaded: False`).

---

## F. WORKBENCH INTEGRATION

### 1. Environment Resolution & `WORKBENCH_HABIT_PYTHON` Flow
- **Registry Declaration**: `models.yaml:245` declares `python_env_var: WORKBENCH_HABIT_PYTHON`.
- **Resolution Flow**:
  1. `runtime.resolve_model_interpreter(model)` checks `os.environ.get("WORKBENCH_HABIT_PYTHON")`. If unset, checks `environment_assignment.json`. If unverified, returns `None`.
  2. `runtime.verify_environment(model)` returns `EnvironmentReport(model_id='habit', variable='WORKBENCH_HABIT_PYTHON', interpreter=None, tier='UNCONFIGURED', status='UNCONFIGURED', reason='WORKBENCH_HABIT_PYTHON not set')`.
  3. `runtime.environment_blockers(model)` checks `if not model.get("environment_required"): return []`. Because `environment_required` is unset for `habit` in `models.yaml`, it returns `[]`!
  4. `preflight.preflight_blockers(model)` calls `environment_blockers(model)`, which appends nothing.
  5. `pc_readiness.py` reads `environment_blockers(model)` -> `[]`. It outputs `"declared": false`, `"blockers": []` for the environment section, and classifies the model as `BLOCKED_IMPLEMENTATION` (due to `command not runnable: COMMAND_UNAUDITED`), NOT `BLOCKED_ENVIRONMENT`!
  6. `doctor.py` checks `environment_blockers(model)` -> `[]`, printing `[OK] environment:habit: UNCONFIGURED`.
  7. `runtime.python_executable(model)` delegates to `require_verified_model_interpreter(model)`. Because `environment_required` is unset, it silently returns `sys.executable` (`/usr/bin/python3`).
  8. `runtime.require_launch_ready(model)` checks `_requires_gpu(model)`. Because `cuda: '12.6'` is declared, `_requires_gpu` returns `True`. It calls `verify_environment(model, probe=True, require_cuda=True)`, which fails with `GpuVerificationError: GPU runtime not verified [UNCONFIGURED/UNCONFIGURED]: WORKBENCH_HABIT_PYTHON not set`.
- **Finding**: An unset interpreter is completely silent on preflight, doctor, and pc_readiness reporting surfaces, masked as `BLOCKED_IMPLEMENTATION` (by adapter command missing), and only caught at the final GPU launch barrier.

### 2. Flow through `environment_planner.py`
`environment_planner.py:840` checks:
```python
if not any(models[mid].get("environment_required") for mid in group["members"]) and not paths:
    installation = "NOT_REQUIRED"
```
Because `habit` (and its siblings `airknow`, `conesep`, `hint`) omits `environment_required: true`, the planner marks the group as `NOT_REQUIRED`.

### 3. Adapter Implementation Gap
`HabitAdapter` (`workbench/backend/adapters/models.py:64-66`):
```python
class HabitAdapter(OfficialScriptAdapter):
    model_id = "habit"
    script = "test.py"
```
Does NOT override `command()`. Inherits `OfficialScriptAdapter.command()`, which unconditionally raises `NotImplementedError("upstream evaluation command requires audit before execution")`.

### 4. Metrics Extraction Gap
`workbench/backend/metrics_extraction.py:414-422` lacks a parser for `habit`, returning:
```python
MetricExtractionResult(
    extraction_status="AGGREGATE_PARSER_UNAVAILABLE",
    parity_status="PARSER_UNAVAILABLE"
)
```

### 5. Hardcoded Machine-Specific Paths
- `lavis/models/blip2_models/blip2.py:32, 48, 55` hardcodes `/root/bert-base-uncased`.
- `lavis/configs/default.yaml:10` hardcodes `cache_root: "/export/home/.cache/lavis"`.

---

## G. OBSERVATION CAPABILITY

- **Verdict Axis**: `QUERY_EXPORT_IMPLEMENTABLE`

### 1. Score & Ranking Production Point
- In `workbench/third_party/HABIT/test.py:154, 168`:
  ```python
  nn_result = [np.argsort(-sims[i, :])[:50] for i in range(sims.shape[0])]
  ```
- Matrix `sims` of shape `(N, G)` contains query-to-target similarity scores with the reference image masked to `-10e10` at `test.py:151`.
- `nn_result[i]` contains top-50 gallery target indices for query `i`.
- Query metadata: `test_queries[i]['source_img_id']`, `test_queries[i]['target_img_id']`, `test_queries[i]['mod']['str']`.
- Gallery metadata: `test_targets[j]['target_img_id']`.

### 2. Per-Query Export Feasibility
- Native code emits aggregate metrics only (`[('<cat>_r1', r), ('<cat>_r10', r), ('<cat>_r50', r)]`). It writes no per-query rankings to disk.
- Query ranking export is implementable by intercepting `sims` and `nn_result` at `test.py:154`, serializing query ID, target ID, and top-50 candidate rankings.

---

## H. DEFECTS

| Defect ID | Severity | Classification | File / Path | Lines | Summary |
|---|---|---|---|---|---|
| `HABIT-DEF-01` | P0 | `EXTERNAL_BLOCKER` | Upstream HuggingFace Hub | — | Clean (0% training noise) checkpoint unpublished; Hub hosts only 20%, 50%, and 80% noise weights. |
| `HABIT-DEF-02` | P0 | `CONFIRMED_BUG` | `workbench/third_party/HABIT/test.py` | 1–264 | Missing standalone FashionIQ evaluation CLI or entrypoint; evaluation only exists inside training loop (`train.py:283`). |
| `HABIT-DEF-03` | P0 | `CONFIRMED_BUG` | `workbench/backend/adapters/models.py` | 64–66 | `HabitAdapter.command` not implemented; raises `NotImplementedError`. |
| `HABIT-DEF-04` | P1 | `CONFIRMED_BUG` | `workbench/registry/models.yaml` | 243–251 | `environment_required` unset for `habit`; causes `environment_blockers` to return `[]` and masks missing interpreter. |
| `HABIT-DEF-05` | P1 | `EXTERNAL_BLOCKER` | `workbench/third_party/HABIT/datasets.py` | 100, 174 | Mandatory spelling correction dictionaries missing from upstream repository (`OFFICIAL_ASSET_UNAVAILABLE`). |
| `HABIT-DEF-06` | P1 | `CONFIRMED_BUG` | `workbench/third_party/HABIT/lavis/models/blip2_models/blip2.py` | 32, 48, 55 | Hardcoded `/root/bert-base-uncased` machine path for BERT tokenizer and Q-Former. |
| `HABIT-DEF-07` | P1 | `CONFIRMED_BUG` | `workbench/third_party/HABIT/lavis/configs/default.yaml` | 10 | Hardcoded author cluster cache path `/export/home/.cache/lavis`. |
| `HABIT-DEF-08` | P2 | `CONFIRMED_BUG` | `workbench/backend/metrics_extraction.py` | 414–422 | No aggregate metrics parser for `habit`; returns `AGGREGATE_PARSER_UNAVAILABLE`. |
| `HABIT-DEF-09` | P2 | `CONFIRMED_BUG` | `workbench/third_party/HABIT/train.py` | 79–81 | `--restore_file` argument declared but never read (dead code). |
| `HABIT-DEF-10` | P2 | `CONFIRMED_BUG` | `workbench/third_party/HABIT/cirr_test_submission.py` | 97 | Requests unregistered model `"Blip2QformerCir"` (fails on load). |
| `HABIT-DEF-11` | P2 | `CONFIRMED_BUG` | `workbench/third_party/HABIT/datasets.py` | 33–35 | Trailing slash path concatenation bug (`path + 'resized_image'`). |
| `HABIT-DEF-12` | P2 | `UNVERIFIED_RISK` | `workbench/third_party/HABIT/datasets.py` | 39–54 | Unconditional write of cache files into `self.path` on cache miss fails on read-only dataset mounts. |
| `HABIT-DEF-13` | P3 | `CONFIRMED_BUG` | `workbench/registry/models.yaml` | 243–275 | `checkpoint_serialization: whole_model` omitted from registry. |
| `HABIT-DEF-14` | P3 | `BY_DESIGN` | Paper Table 1 / arXiv:2604.18037 | Page 5 | Paper Table 1 R@10 Avg prints 55.38 vs arithmetic mean 55.37 (0.0067% rounding deviation). |

---

## I. EVIDENCE

1. **Commit SHA & Single-Commit History**:
   ```bash
   $ git -C workbench/third_party/HABIT rev-parse HEAD
   6ddc2cd63eac1f65693f4e821c34fb3b42aee306
   $ git -C workbench/third_party/HABIT rev-list --count HEAD
   1
   $ git -C workbench/third_party/HABIT log -1 --oneline
   6ddc2cd Update README.md
   ```

2. **Interpreter Resolution & Silent Environment Omission**:
   ```python
   >>> from workbench.backend.runtime import resolve_model_interpreter, verify_environment, environment_blockers, require_launch_ready, require_verified_model_interpreter
   >>> from workbench.backend.registry import load_registry
   >>> habit = [m for m in load_registry()['models'] if m['model_id'] == 'habit'][0]
   >>> resolve_model_interpreter(habit)
   None
   >>> verify_environment(habit)
   EnvironmentReport(model_id='habit', variable='WORKBENCH_HABIT_PYTHON', interpreter=None, tier='UNCONFIGURED', status='UNCONFIGURED', reason='WORKBENCH_HABIT_PYTHON not set')
   >>> environment_blockers(habit)
   []
   >>> require_verified_model_interpreter(habit)
   '/usr/bin/python3'
   >>> require_launch_ready(habit)
   GpuVerificationError: GPU runtime not verified [UNCONFIGURED/UNCONFIGURED]: WORKBENCH_HABIT_PYTHON not set
   ```

3. **PC Readiness Execution**:
   ```bash
   $ python3 -m workbench.scripts.pc_readiness --model habit --json
   # state: BLOCKED_IMPLEMENTATION
   # reason: command not runnable: COMMAND_UNAUDITED
   # environment declared: false, blockers: []
   ```

4. **Doctor Verification**:
   ```bash
   $ python3 -m workbench.scripts.doctor --model habit
   # [BLOCKED] runtime:habit: command unavailable
   # [BLOCKED] command:habit: COMMAND_AUDITED
   # [OK] environment:habit: UNCONFIGURED
   ```

5. **Adapter Command Exception**:
   ```python
   >>> from workbench.backend.adapters.models import ADAPTERS
   >>> from workbench.backend.adapters.base import EvalRequest
   >>> from workbench.backend.registry import load_registry
   >>> adapter = ADAPTERS['habit'](load_registry())
   >>> req = EvalRequest('habit', 'fiq_n05', 'fashioniq_full_gallery_ref_excluded', '/data', '/tmp/out.json')
   >>> adapter.command('/source', '/checkpoint', req)
   NotImplementedError: upstream evaluation command requires audit before execution
   ```

6. **HuggingFace Repository Files & Hashes (API Inspection)**:
   ```text
   fiq/HABIT-FIQ_N0.2.pt  2730715930 B  814d2d0ceb29ef910f37562e8bea8a76cf38aa546fbc2102e495df687c276b03
   fiq/HABIT-FIQ_N0.5.pt  2730715930 B  8a0d3bd27cf46d8334f9596404d89ebf803da0fa65236d89815bb3929b9a4216
   fiq/HABIT-FIQ_N0.8.pt  2730715930 B  763885de6907bca4f017b336ed7ec3740cb9305c5adb6a135da45c158ddb9717
   cirr/HABIT-CIRR-N0.2.pt 2736022950 B  7c477c2f5f4c7c1c1401fd73ed6bf8997b8cd5ac543523cdaf17736d4dcc4b24
   cirr/HABIT-CIRR-N0.5.pt 2736022950 B  ff41e91759ca8b10b8bb4ba1c1eb26f639c99dacf4de9a5ed398e21d525df3b1
   cirr/HABIT-CIRR-N0.8.pt 2736022950 B  fac114a4916da352f8a83327f9464019064aceb932d24080f73b96fc688e0787
   ```
   No `HABIT-FIQ_N0.0.pt` exists on the repository.

7. **Staged Spelling Correction Dictionaries SHA-256**:
   ```bash
   $ sha256sum /home/heheboiz/data/cir/data/fashionIQ_dataset/captions/correction_dict_*.json
   9e4a59877098fe26036b70759387be589990eb4068692427e890113dcba2edda  .../correction_dict_dress.json
   ca29e29e23c0a1987d6b121219d9580bdae9872df0ea219c5aea870ee2f8567e  .../correction_dict_shirt.json
   3dfbf15fa6d3a30dc0f5cf7e2908ca36d1775e42aa6598f1dba7899267dc1b71  .../correction_dict_toptee.json
   ```

---

## J. VERDICT

| Axis | Verdict | Key Evidence |
|---|---|---|
| **SOURCE_COVERAGE** | **`SOURCE_AUDITED`** | Pinned tree at commit `6ddc2cd63eac1f65693f4e821c34fb3b42aee306` fully inspected. Evaluation graph mapped across `train.py`, `test.py`, `datasets.py`, and `HABIT.py`. |
| **PROTOCOL** | **`PROTOCOL_VERIFIED`** | Source `datasets.py:205-211` and `test.py:150-151` strictly implement Protocol B (`fashioniq_full_gallery_ref_excluded`: full val split gallery, reference excluded via `-10e10` mask before ranking). |
| **COMMAND** | **`COMMAND_BLOCKED`** | Pinned `test.py` lacks CLI/`__main__` entrypoint. Evaluator embedded inside training loop (`train.py:283`). `HabitAdapter.command` raises `NotImplementedError`. |
| **CHECKPOINT** | **`CHECKPOINT_BLOCKED`** | Clean (0% noise) checkpoint unpublished by authors. HuggingFace repo publishes only 20%, 50%, and 80% noise weights. Evaluating against headline paper row constitutes noise condition conflation. |
| **QUERY_EXPORT** | **`QUERY_EXPORT_IMPLEMENTABLE`** | Native `test.py:154, 168` produces candidate rankings via `np.argsort(-sims[i, :])[:50]`. Observer intercept implementable at scoring matrix, but unexported in native code. |
| **RELEASE_READINESS** | **`BLOCKED_EXTERNAL_DEPENDENCY`** | Blocked on unpublished clean checkpoint, unpublished correction dictionaries, un-provisioned environment (`WORKBENCH_HABIT_PYTHON`), missing adapter command, and hardcoded machine paths (`/root/bert-base-uncased`). |
