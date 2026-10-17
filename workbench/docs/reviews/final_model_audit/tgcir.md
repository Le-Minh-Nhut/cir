# Final Model Audit: tgcir

## A. MODEL IDENTITY

- **Model ID**: `tgcir` (`workbench/registry/models.yaml:785`)
- **Method Name**: `TG-CIR` (`workbench/registry/models.yaml:791`)
- **Research Generation**: `legacy` (`workbench/registry/models.yaml:792`)
- **Publication Year**: `2023` (`workbench/registry/models.yaml:793`)
- **Paper Title**: `Target-Guided Composed Image Retrieval` (`workbench/registry/models.yaml:794`)
- **Paper URL**: `https://arxiv.org/abs/2309.01366` (`workbench/registry/models.yaml:795`)
- **Publication Venue**: Proceedings of the 31st ACM International Conference on Multimedia (ACM MM '23), October 29–November 3, 2023, Ottawa, ON, Canada, pp. 915–923. DOI: `10.1145/3581783.3611817`.
- **Authors**: Haokun Wen, Xian Zhang, Xuemeng Song, Yinwei Wei, Liqiang Nie.
- **Upstream Repository URL**: `https://github.com/iLearn-Lab/MM23-TG-CIR` (`workbench/registry/models.yaml:796`)
- **Upstream Default Branch**: `main` (`workbench/registry/models.yaml:797`)
- **Upstream Pinned Commit SHA**: `65fa78eaf8cabe8197fcc22ccab42207c591bdee` (`workbench/registry/models.yaml:798`)
- **Source Directory**: `TGCIR` (`workbench/registry/models.yaml:800`)
- **Source Available**: `true` (`workbench/registry/models.yaml:799`)
- **Native Protocol**: `fashioniq_val_split` (`workbench/registry/models.yaml:806`)
- **Supported Protocols**: `[fashioniq_val_split]` (`workbench/registry/models.yaml:807`)
- **Reproduction Status**: `BLOCKED_COMMAND` (`workbench/registry/models.yaml:812`)
- **Command Status**: `COMMAND_BLOCKED_SOURCE_ENTRYPOINT` (`workbench/registry/models.yaml:813`)
- **Replay Status**: `PREFLIGHT_ONLY` (`workbench/registry/models.yaml:786`)
- **Replay Type**: `PREFLIGHT_ONLY` (`workbench/registry/models.yaml:789`)
- **Category Model Policy**: `shared` (`workbench/registry/models.yaml:787`)
- **Checkpoint Serialization**: `whole_model` (`workbench/registry/models.yaml:788`)
- **Reported Metrics**: `reported_r10: 51.32`, `reported_r50: 73.09`, `reported_mean: null` (`workbench/registry/models.yaml:808-810`)
- **Reported Score Note**: "Paper reports macro R@10 and R@50; 62.205 is derivable but not presented as a reproduced metric." (`workbench/registry/models.yaml:811`)

### Pinned Commit Verification
```bash
$ git -C workbench/third_party/TGCIR rev-parse HEAD
65fa78eaf8cabe8197fcc22ccab42207c591bdee
$ git -C workbench/third_party/TGCIR log -1 --oneline
65fa78e Update README.md
```
The pinned commit SHA `65fa78eaf8cabe8197fcc22ccab42207c591bdee` in `workbench/third_party/TGCIR` exactly matches `upstream_commit_sha` in `workbench/registry/models.yaml:798`.

### Checkpoint Variant
- **Variant ID**: `tgcir_archive_unresolved` (`workbench/registry/models.yaml:832`)
- **Filename**: `tgcir-fashioniq-unresolved.pt` (`workbench/registry/models.yaml:833`) — workbench placeholder.
- **Download URL**: `null` (`workbench/registry/models.yaml:834`)
- **Source URL**: `https://drive.google.com/file/d/1OdZTtJqy-RTpYXBaq5ThmH3IBnGvYCyi/view` (`workbench/registry/models.yaml:835`, matches `workbench/third_party/TGCIR/README.md:15`)
- **Source Type**: `author_google_drive` (`workbench/registry/models.yaml:836`)
- **Checkpoint Mapping Status**: `UNVERIFIED` (`workbench/registry/models.yaml:842`)
- **Status**: `CHECKPOINT_MAPPING_UNVERIFIED` (`workbench/registry/models.yaml:841`)
- **Expected SHA-256**: `null` (`workbench/registry/models.yaml:840`)
- **Expected Size Bytes**: `null` (`workbench/registry/models.yaml:839`)
- **Evaluation Noise Pct**: `0` (`workbench/registry/models.yaml:838`)
- **Checkpoint Training Noise Pct**: `null` (`workbench/registry/models.yaml:837`)

### Paper Benchmark Row
Registered in `workbench/registry/paper_benchmarks/papers.yaml:1520-1588` as `reference_id: tgcir_fashioniq_val_table1`.
- **Source Table**: Table 1 ("Performance comparison on FashionIQ and Shoes with respect to R@k(%)"), page 7, ACM MM '23 / arXiv:2309.01366v1.
- **Exact Row Quoted**:
  `TG-CIR | 45.22 69.66 | 52.60 72.52 | 56.14 77.10 | 51.32 73.09 | 25.89 63.20 85.07 58.05`
- **Metric Breakdown**:
  - `dress`: R@10 = 45.22, R@50 = 69.66
  - `shirt`: R@10 = 52.60, R@50 = 72.52
  - `toptee`: R@10 = 56.14, R@50 = 77.10
  - `Avg`: R@10 = 51.32, R@50 = 73.09
- **Arithmetic Check**:
  - Macro R@10 = $(45.22 + 52.60 + 56.14) / 3 = 51.31999... \to 51.32$
  - Macro R@50 = $(69.66 + 72.52 + 77.10) / 3 = 73.09333... \to 73.09$
  - Mean of R@10 and R@50 = $(51.32 + 73.09) / 2 = 62.205$ (`WORKBENCH_DERIVED`, not printed anywhere in the paper).
- **Reference Status**: `VERIFIED`, **Review Status**: `CONFIRMED`.

---

## B. SOURCE COVERAGE

- **Audit Type**: `STATIC_AUDIT` + local script validation.
- **Pinned Tree Inventory** (`workbench/third_party/TGCIR/`, 11 files total):
  ```text
  workbench/third_party/TGCIR/
  ├── README.md                                (94 lines: claims, environment, citations)
  ├── cirr_test_submission.py                  (101 lines: CIRR server submission formatter only)
  ├── correction_dict/                         (4 bundled spelling correction JSON dictionaries)
  │   ├── correction_dict_dress.json           (365 lines, 363 token entries)
  │   ├── correction_dict_shirt.json           (362 lines, 360 token entries)
  │   ├── correction_dict_shoes.json           (110 lines, 108 token entries)
  │   └── correction_dict_toptee.json          (393 lines, 391 token entries)
  ├── datasets.py                              (427 lines: FashionIQ, Shoes, CIRR dataset loaders)
  ├── model.py                                 (280 lines: SpatialAttention, TokenLearner, Backbone, VCG_CIR)
  ├── test.py                                  (163 lines: FashionIQ test() and CIRR test_cirr_valset())
  ├── train.py                                 (271 lines: CLI parser, training loop, evaluation call)
  └── utils.py                                 (142 lines: logging, dict serialization, running average)
  ```

### Evaluation Dependency Graph
For FashionIQ evaluation:
1. `train.py:198-204`: Training loop calls `test.test(args, model, trainset, category)`.
2. `datasets.py:20-160`: `FashionIQ` class loads resized images, train/val captions, val splits, spelling dictionaries, constructs query/target structures, and caches them to `.pkl`.
3. `test.py:6-82`: `test()` function batches queries and targets, feeds them to `model.extract_retrieval_compose` and `model.extract_retrieval_target`, normalizes features, performs matrix multiplication, masks query reference indices with `-10e10`, sorts top-50, and calculates R@1, R@10, R@50 per category.
4. `model.py:147-210`: `VCG_CIR` model executes `s_compose_feature()` and `extract_img_fea()`.
5. `model.py:38-146`: `Backbone` executes OpenAI CLIP ViT-B/16 visual encoder and token embedding transformer.

### Inspected vs. Excluded Files
- **Inspected**:
  - `test.py:6-82`: Native FashionIQ evaluation function `test()`.
  - `datasets.py:20-161`: `FashionIQ` class and helper methods `correct_text()`, `concat_text()`, `get_test_data()`, `get_test_img()`.
  - `model.py:1-211`: Model inference architecture (`SpatialAttention`, `TokenLearner`, `Backbone`, `VCG_CIR.extract_retrieval_compose`, `VCG_CIR.extract_retrieval_target`).
  - `train.py:1-125, 172-240, 242-271`: Entrypoint parser, model instantiation, training/eval hook `train_and_evaluate()`.
  - `correction_dict/*.json`: All 4 spelling dictionaries verified for structure and digest.
  - `README.md`: Upstream documentation and claims.
- **Excluded**:
  - `cirr_test_submission.py`: CIRR test server submission code only. Unreachable for FashionIQ.
  - `datasets.py:162-292` (`Shoes` class) and `datasets.py:294-427` (`CIRR` class): Other benchmark loaders.
  - `test.py:86-163` (`test_cirr_valset`): CIRR validation evaluation.
  - `train.py:137-171` (`train()` function): Backward training steps.
  - `model.py:212-280` (`compute_loss`, `mask_constraint`, `info_nce`, `orthogonal_regularization`, `kl_div`): Training distillation losses.

### Executable Source vs. README Claims
1. **Backbone Package**: `README.md:19` lists `clip==0.2.0`. On PyPI, `clip==0.2.0` is `silent1mezzo/clip` (a CLI clipboard manager), not OpenAI CLIP. OpenAI CLIP does not publish version tags on PyPI.
2. **Dataset Resizing**: `README.md:30-35` never mentions that images must be pre-resized or cropped. However, `datasets.py:25` hardcodes `self.image_dir = self.path + 'resized_image'`, requiring pre-resized images in `<fashioniq_path>/resized_image/{dress,shirt,toptee}`.
3. **Spelling Correction Dictionaries**: `README.md:34` states dictionaries are in `correction_dict/`. However, `datasets.py:59, 131` hardcodes loading from `self.caption_dir` (`<fashioniq_path>/captions/`), meaning they must be manually copied into the dataset path.
4. **Evaluation Command**: `README.md:46-70` provides ONLY training commands (`python train.py --dataset 'fashioniq' ...`). The README provides zero evaluation commands for FashionIQ. `test.py` contains no CLI or `__main__` entrypoint.

---

## C. NATIVE EVALUATION

- **Verdict Axis**: `PROTOCOL_VERIFIED`

### 1. Pair-Union Gallery Construction
Native gallery construction is implemented in `workbench/third_party/TGCIR/datasets.py:148-160` (`FashionIQ.get_test_data`):
```python
148:        test_targets_id = []
149:        for i in test_queries:
150:            if i['source_img_id'] not in test_targets_id:
151:                test_targets_id.append(i['source_img_id'])
152:            if i['target_img_id'] not in test_targets_id:
153:                test_targets_id.append(i['target_img_id'])
154:        test_targets = []
155:        for i in test_targets_id:
156:            out = {}
157:            out['target_img_id'] = i
158:            out['target_img_data'] = self.get_test_img(name + '_' + images[i])      
159:            test_targets.append(out)
```
- **Proof of Protocol C**: `test_queries` iterates over `cap.{category}.val.json`. For each query, its reference image index (`source_img_id`) is added to `test_targets_id` if not already present, followed by its target image index (`target_img_id`).
- This constructs the ordered, first-seen unique union of validation references and targets.
- Gallery sizes re-derived from local FashionIQ annotations:
  - `dress`: 2,628 target images (from 2,017 queries, 3,817 total split images)
  - `shirt`: 3,089 target images (from 2,038 queries, 6,346 total split images)
  - `toptee`: 2,902 target images (from 1,961 queries, 5,373 total split images)

### 2. Reference Exclusion Policy
Native reference exclusion is implemented in `workbench/third_party/TGCIR/test.py:63-68`:
```python
63:    test_targets_id = []
64:    for i in test_targets:
65:        test_targets_id.append(i['target_img_id'])
66:    for i, t in enumerate(test_queries):
67:        sims[i, test_targets_id.index(t['source_img_id'])] = -10e10
```
- **Masking Line**: `sims[i, test_targets_id.index(t['source_img_id'])] = -10e10`
- **Mask Value**: `-10e10` ($-10^{11}$). On cosine similarity values ranging in $[-1, 1]$, $-10^{11}$ guarantees the query's reference image is forced to the bottom of the ranking prior to sorting.

### 3. Caption Preprocessing
Native text normalization is implemented in `workbench/third_party/TGCIR/datasets.py:71-80`:
```python
71:    def correct_text(self, text, correction_dict):
72:        trans=str.maketrans({key: ' ' for key in string.punctuation})
73:        tokens = str(text).lower().translate(trans).strip().split()
74:        text = " ".join([correction_dict.get(word) if word in correction_dict else word for word in tokens])
75:        return text
76:
77:    def concat_text(self, captions, correction_dict):
78:        text = "{} and {}".format(self.correct_text(captions[0], correction_dict), self.correct_text(captions[1], correction_dict))
79:        return text
```
- Strips punctuation to spaces, lowercases, tokenizes on whitespace, performs token-by-token spelling correction via `correction_dict_{category}.json`, and joins the two captions with `"{} and {}"`.
- Tokenized in `model.py:120`: `clip.tokenize(txt).cuda()`.

### 4. Similarity Scoring and Ranking
Implemented in `workbench/third_party/TGCIR/test.py:53-70`:
```python
53:    # feature normalization
54:    for i in range(all_queries.shape[0]):
55:        all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
56:    for i in range(all_imgs.shape[0]):
57:        all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])
58:    
59:    # match test queries to target images, get nearest neighbors
60:    sims = all_queries.dot(all_imgs.T)
...
70:    nn_result = [np.argsort(-sims[i, :])[:50] for i in range(sims.shape[0])]
```
- L2-normalized feature vectors are multiplied via dot product (`all_queries.dot(all_imgs.T)`).
- References are masked with `-10e10`, then ranked via `np.argsort(-sims[i, :])[:50]`.

### 5. Metric Calculation and Aggregation
Implemented in `workbench/third_party/TGCIR/test.py:72-82`:
```python
74:    for k in [1, 10, 50]:
75:        r = 0.0
76:        for i, nns in enumerate(nn_result):
77:            if test_targets_id.index(test_queries[i]['target_img_id']) in nns[:k]:
78:                r += 1
79:        r = 100 * r / len(nn_result)
80:        out += [('{}_r{}'.format(name, k), r)]
```
- Recall@K is computed as the percentage of queries where the target image index is within the top-K retrieved items.
- Evaluated per category. Upstream `test.py` does not compute macro averages; macro aggregation is performed in the paper via equal-weight mean across dress, shirt, and toptee.

---

## D. RUNNER FIDELITY

- **Critical Check**: Detailed comparison of `workbench/replay/tgcir.py` vs. Native Evaluator (`test.py`, `datasets.py`, `model.py`).

| Dimension | Native Upstream Evaluator | Workbench Replay (`tgcir.py`) | Divergence Analysis & Classification |
|---|---|---|---|
| **Purpose & Scope** | Evaluates FashionIQ during training (`train.py:200` $\to$ `test.test()`). | Read-only preflight checker (`tgcir.py:1-5`). Never loads weights or runs inference. | **BY_DESIGN**. Documented boundary: module is explicitly preflight-only (`replay_type: PREFLIGHT_ONLY`). |
| **Execution Entrypoint** | `train.py:200` inside `train_and_evaluate()` epoch loop. `test.py` has no `__main__`. | Standalone CLI script: `python3 workbench/replay/tgcir.py --source-root ...`. | **DOCUMENTED_DIFFERENCE**. Replay provides audited CLI for preflight; native has no eval CLI. |
| **Exit / Return Behavior** | Returns metric list `[('{cat}_r1', r1), ('{cat}_r10', r10), ('{cat}_r50', r50)]`. | Emits `[BLOCKED]` lines to `sys.stderr` and returns exit code 1 (`tgcir.py:91`). | **BY_DESIGN**. Fails closed when required assets/checkpoints are unverified. |
| **Batching** | Queries: `if len(imgs) >= params.batch_size or t is test_queries[-1]:` (`test.py:28`). Targets: `test.py:44`. Default `batch_size=64`. | No batching implemented. | **BY_DESIGN**. Inference is disabled. |
| **Dtype & Precision** | `torch.float32` on GPU (`test.py:31, 47`), `.data.cpu().numpy()` float32 for features and similarities. | No tensor computation. | **DOCUMENTED_DIFFERENCE**. |
| **Device Execution** | Unconditional `.cuda()` calls (`test.py:31, 47`, `model.py:32, 41, 107, 120, 136, 244, 251`). Hard GPU requirement. | CPU-only filesystem inspections (`Path.is_file()`, `is_dir()`). | **DOCUMENTED_DIFFERENCE**. Preflight runs safely on CPU. |
| **Transforms** | OpenAI CLIP ViT-B/16 preprocess: `Resize(224, BICUBIC)`, `CenterCrop(224)`, RGB conversion, `ToTensor()`, CLIP normalization. | Preflight checks CLIP asset availability; does not execute transforms. | **DOCUMENTED_DIFFERENCE**. Matches recipe; weights unverified. |
| **Raw Inputs Check** | `datasets.py:26-27, 57, 127, 129-130`: reads train/val captions and val splits. | `tgcir.py:20-28`: `RAW_INPUTS` tuple checks train/val captions and val splits. | **EXACT_MATCH**. Both require all 6 captions and 3 val splits. |
| **Resized Images Check** | `datasets.py:25`: `self.image_dir = self.path + 'resized_image'`. | `tgcir.py:30`: `RESIZED_IMAGE_DIRS` checks `resized_image/{category}`. | **EXACT_MATCH**. |
| **Correction Dict Check** | `datasets.py:59, 131`: `correction_dict_{name}.json` in `captions/`. | `tgcir.py:29`: `CORRECTION_DICTIONARIES` checks `captions/correction_dict_{cat}.json`. | **EXACT_MATCH**. |
| **Native Cache Handling** | `datasets.py:29-53`: Trusts `fashion_iq_data.json` and `.pkl` files unconditionally if present. | `tgcir.py:31-34, 56-62`: Flags missing cache or warns on unverified cache freshness. | **EXACT_MATCH**. Captures native cache defect accurately. |
| **Checkpoint Loader** | `train.py:271`: Saves via `torch.save(current_model, ...)`. Eval would need `torch.load()`. | `tgcir.py:64-74`: Checks file presence; refuses unverified whole-model pickle loading. | **BY_DESIGN**. Enforces workbench security contract (`replay/common.py:36-39`). |
| **Metric Extraction** | Native returns `('{cat}_r{k}', r)`. | Replay does not parse stdout; workbench `metrics_extraction.py:414` lacks a TGCIR parser. | **CONFIRMED_BUG** in workbench integration layer. |

---

## E. ASSETS & PREPARATION

### 1. ViT-B/16 Asset Dependency
- **Native Dependency**: `train.py:88` and `model.py:41` call `clip.load(args.img_encoder, device='cuda', jit=False)` where default `args.img_encoder` is `'ViT-B/16'`.
- **Expected File**: `~/.cache/clip/ViT-B-16.pt`
- **Authentic Upstream Source**: OpenAI CLIP Azure public CDN (`clip/clip.py:37`):
  - URL: `https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt`
  - Digest (SHA-256): `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f` (HTTP 200, verified).
- **Registry Defect (CONFIRMED_BUG)**:
  In `workbench/registry/assets/cluster_b2_encoder_family.yaml:432, 434`:
  - `official_download_url`: `https://openaipublic.azureedge.net/clip/models/5806e77cd80cf8775b1045601c1438b6d00e063f3b21f8f47e222d83ff529a4e/ViT-B-16.pt`
  - `expected_sha256`: `5806e77cd80cf8775b1045601c1438b6d00e063f3b21f8f47e222d83ff529a4e`
  - Verification: `curl -I` on this URL returns **HTTP 404 Not Found**. The recorded digest is corrupted/invalid.
- **Local Workstation Status**: `/home/heheboiz/.cache/clip/` does not exist. Provenance remains `BLOCKED_EXTERNAL_ASSET_PROVENANCE`.

### 2. Preparation Contracts and Artifacts
- **Correction Dictionaries**:
  - Pinned tree ships all four dictionaries in `workbench/third_party/TGCIR/correction_dict/`:
    - `correction_dict_dress.json`: SHA-256 `9e4a59877098fe26036b70759387be589990eb4068692427e890113dcba2edda` (363 items)
    - `correction_dict_shirt.json`: SHA-256 `ca29e29e23c0a1987d6b121219d9580bdae9872df0ea219c5aea870ee2f8567e` (360 items)
    - `correction_dict_toptee.json`: SHA-256 `3dfbf15fa6d3a30dc0f5cf7e2908ca36d1775e42aa6598f1dba7899267dc1b71` (391 items)
    - `correction_dict_shoes.json`: SHA-256 `50451054df2d2365d732413bea40983c03c6f00ea4361ec258d16bd2c6238c24` (108 items)
  - Location mismatch: `datasets.py:59, 131` loads from `<fashioniq_path>/captions/`. They must be staged into the captions directory.
- **Native Cache Artifacts (`.pkl` and `.json`)**:
  - `datasets.py:29-53` creates `fashion_iq_data.json`, `test_queries_{cat}.pkl`, and `test_targets_{cat}.pkl`.
  - **Stale Cache Hazard**: `test_queries_*.pkl` pickles transformed image tensors (`self.get_test_img()`). If image resizing, transforms, or annotations change, `FashionIQ` unconditionally loads the stale pickle files without detecting changes. If `fashion_iq_data.json` exists but any `.pkl` is missing, it crashes with `FileNotFoundError`.
- **Resized Image Requirements**:
  - `datasets.py:25` hardcodes `self.image_dir = self.path + 'resized_image'`.
  - Expects `<fashioniq_path>/resized_image/{dress,shirt,toptee}/<id>.jpg`.
  - The local directory `data/fashionIQ_dataset/resized_image` does not exist.
  - The pinned upstream repository provides no image resizing script.
- **CWD and Path Concatenation Bug (CONFIRMED_BUG)**:
  `datasets.py:24-27`:
  ```python
  self.path = path
  self.image_dir = self.path + 'resized_image'
  self.split_dir = self.path + 'image_splits'
  self.caption_dir = self.path + 'captions'
  ```
  String concatenation without `/` requires `path` to end with a trailing slash (`/`). Without a trailing slash, paths resolve to invalid paths like `.../fashionIQ_datasetresized_image`.

---

## F. CHECKPOINT

- **Verdict Axis**: `CHECKPOINT_BLOCKED`

### Checkpoint Archive Mapping & Serialization
- **Declared Checkpoint Variant**: `tgcir_archive_unresolved` (`models.yaml:832`)
- **Upstream Location**: `README.md:15` links to Google Drive: `https://drive.google.com/file/d/1OdZTtJqy-RTpYXBaq5ThmH3IBnGvYCyi/view`.
- **Archive Structure & Member Files**:
  - The Google Drive link points to an opaque archive (`TG-CIR.zip` or similar).
  - The upstream repository never documents the internal file structure of the archive.
  - Pinned source `train.py:271` saves models via:
    `torch.save(current_model, os.path.join(args.model_dir, 'best_model.pt'))`
  - It saves a whole-model object (`VCG_CIR`), not a state dict.
- **Provenance Confidence**: `ZERO`. No author checksum or manifest exists.
- **Workbench Security Policy**: `workbench/replay/common.py:36-39` refuses whole-model unpickling without an author-verified SHA-256 digest (`load_trusted_whole_model`).
- **Official Evaluation Command**: **NONE**. Upstream provides no standalone evaluation script. `test.py` contains bare functions without a CLI or `__main__` entrypoint.

---

## G. WORKBENCH INTEGRATION

### 1. Registry & Contracts
- `workbench/registry/models.yaml:785-842`: Block registered with `replay_status: PREFLIGHT_ONLY`, `command_status: COMMAND_BLOCKED_SOURCE_ENTRYPOINT`, `reproduction_status: BLOCKED_COMMAND`.
- `workbench/registry/preparation_contracts.yaml:170-219`: `preparation_id: tgcir_fashioniq`. Accurately documents raw inputs, generated artifacts, external CLIP assets, and native cache warnings.

### 2. Adapter & Doctor
- `workbench/backend/adapters/models.py:114-116`:
  ```python
  class TGCIRAdapter(OfficialScriptAdapter):
      model_id = "tgcir"
      script = "test.py"
  ```
  Does not override `def command()`. Calling `build_command()` raises `NotImplementedError("upstream evaluation command requires audit before execution")`.
- `python3 -m workbench.scripts.doctor --model tgcir` outputs:
  - `[OK] source:tgcir: source pin ready`
  - `[BLOCKED] runtime:tgcir: command unavailable`
  - `[BLOCKED] command:tgcir: COMMAND_BLOCKED_SOURCE_ENTRYPOINT`
  - `[BLOCKED] external-assets:tgcir: required evaluation assets unresolved`

### 3. Runtime Preflight & PC Readiness
- `python3 -m workbench.scripts.pc_readiness --model tgcir --json` outputs:
  - `"state": "BLOCKED_IMPLEMENTATION"`
  - `"reason": "command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT"`
  - Blockers: missing CLIP asset provenance, blocked source entrypoint, unconfigured environment `WORKBENCH_TGCIR_PYTHON`, missing checkpoint file.

### 4. Downloader & Environment
- Downloader cannot acquire checkpoint because `download_url: null`.
- Environment `WORKBENCH_TGCIR_PYTHON` is unconfigured. Pinned requirements in `models.yaml:828` cite `clip==0.2.0` (known invalid pin).

### 5. Metrics Extraction & Paper Gate
- `workbench/backend/metrics_extraction.py:414-422`: Lacks an aggregate metric parser for `tgcir`. Emits `AGGREGATE_PARSER_UNAVAILABLE`.
- `workbench/backend/paper_gate.py`: Genuinely compares 9 metrics against Table 1, but receives `observed_metrics=None` $\to$ `FAILED_OUTPUT_VALIDATION`.

---

## H. OBSERVATION CAPABILITY

- **Verdict Axis**: `QUERY_EXPORT_IMPLEMENTABLE`

### Score & Ranking Production Point
- In `workbench/third_party/TGCIR/test.py:61-70`:
  ```python
  61:    sims = all_queries.dot(all_imgs.T)
  ...
  67:        sims[i, test_targets_id.index(t['source_img_id'])] = -10e10
  70:    nn_result = [np.argsort(-sims[i, :])[:50] for i in range(sims.shape[0])]
  ```
- Similarity matrix `sims` (shape `[M, N]`, where $M$ is query count, $N$ is gallery target count) is computed via dot product on CPU numpy array.
- Reference exclusion is applied directly at line 67.
- Top-50 ranked target indices are produced at line 70 via `np.argsort(-sims[i, :])[:50]`.

### Exposing Per-Query Rankings
- **Feasibility**: Fully implementable. The structure is equivalent to LIMN's `np.argsort(-sims)`.
- **Required Observer Payload**:
  1. Query identifier: Query index `i`, reference image ID `test_queries[i]['source_img_id']`, target ID `test_queries[i]['target_img_id']`, modifier string `test_queries[i]['mod']['str']`.
  2. Candidate rankings: Target image IDs mapped via `[test_targets_id[idx] for idx in nn_result[i]]`.
  3. Ranked similarities: `sims[i, nn_result[i]]`.
  4. Target hit indicators: `test_targets_id.index(test_queries[i]['target_img_id']) in nn_result[i][:k]`.
- **Blockers**:
  1. No runnable evaluation CLI entrypoint in source or workbench replay runner.
  2. Checkpoint unavailable and untrusted whole-model pickle loading forbidden.
  3. Missing resized images in dataset root.
  4. Missing OpenAI CLIP ViT-B/16 asset provenance.

---

## I. DEFECTS

### 1. `CONFIRMED_BUG` — Corrupted CLIP ViT-B/16 URL and SHA-256 in Asset Registry
- **Location**: `workbench/registry/assets/cluster_b2_encoder_family.yaml:432, 434`
- **Precondition**: Operator or script attempts to resolve/validate the declared CLIP ViT-B/16 asset.
- **Actual**: Declared URL `https://openaipublic.azureedge.net/clip/models/5806e77cd80cf8775b1045601c1438b6d00e063f3b21f8f47e222d83ff529a4e/ViT-B-16.pt` returns HTTP 404.
- **Expected**: Authentic OpenAI CLIP Azure URL `https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt` (HTTP 200) with digest `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f`.

### 2. `CONFIRMED_BUG` — Missing TGCIR Output Parser in Metrics Extraction
- **Location**: `workbench/backend/metrics_extraction.py:414-422`
- **Precondition**: An evaluation run finishes with returncode 0.
- **Actual**: `extract_aggregate_metrics("tgcir", ...)` falls through to `AGGREGATE_PARSER_UNAVAILABLE`, returning `observed_metrics=None`. Gate fails with `FAILED_OUTPUT_VALIDATION`.
- **Expected**: A dedicated output parser extracting `('{cat}_r{k}', value)` pairs and computing macro averages.

### 3. `CONFIRMED_BUG` — Path Concatenation Bug in Dataset Loader
- **Location**: `workbench/third_party/TGCIR/datasets.py:25-27`
- **Precondition**: Passing `args.fashioniq_path` without a trailing slash (e.g., `/path/to/fashioniq`).
- **Actual**: Resolves to `/path/to/fashioniqresized_image`, raising `FileNotFoundError`.
- **Expected**: Robust path concatenation using `os.path.join()`.

### 4. `EXTERNAL_BLOCKER` — Absence of Standalone Evaluation Entrypoint in Pinned Source
- **Location**: `workbench/third_party/TGCIR/test.py`
- **Impact**: Upstream only evaluates FashionIQ during training (`train.py:200`). There is no standalone evaluation command or CLI parser. Workbench adapter raises `NotImplementedError`.

### 5. `EXTERNAL_BLOCKER` — Unmapped Google Drive Checkpoint Archive
- **Location**: `workbench/third_party/TGCIR/README.md:15`, `workbench/registry/models.yaml:835`
- **Impact**: Checkpoint is an unmapped Google Drive URL without checksum, member names, or category mapping. Deserialization requires whole-model pickle loading, which is strictly forbidden without an author digest.

### 6. `EXTERNAL_BLOCKER` — Undocumented and Missing Image Resizing Pipeline
- **Location**: `workbench/third_party/TGCIR/datasets.py:25`
- **Impact**: Code requires pre-resized images in `<fashioniq_path>/resized_image/`, but upstream README never documents resizing and provides no script to produce them.

### 7. `UNVERIFIED_RISK` — Blind Loading of Unchecked Native Pickle Caches
- **Location**: `workbench/third_party/TGCIR/datasets.py:29-53`
- **Impact**: If `fashion_iq_data.json` exists, dataset blindly unpickles `test_queries_*.pkl` and `test_targets_*.pkl`. If transforms, images, or annotations change, stale tensors are evaluated without detection.

### 8. `BY_DESIGN` — Workbench Replay Preflight-Only Execution
- **Location**: `workbench/replay/tgcir.py:1-5, 88-91`
- **Impact**: Runner intentionally refuses inference and returns exit code 1 to prevent executing untrusted whole-model checkpoints and unverified CLIP weights.

---

## J. EVIDENCE

1. **Commit SHA & Single-Commit History**:
   ```bash
   $ git -C workbench/third_party/TGCIR rev-parse HEAD
   65fa78eaf8cabe8197fcc22ccab42207c591bdee
   $ git -C workbench/third_party/TGCIR rev-list --count HEAD
   1
   ```
2. **Preflight Runner Execution**:
   ```bash
   $ python3 workbench/replay/tgcir.py --source-root workbench/third_party/TGCIR --dataset-root data/fashionIQ_dataset --checkpoint-root workbench/artifacts/checkpoints/tgcir
   [BLOCKED] TG-CIR native cache missing: data/fashionIQ_dataset/fashion_iq_data.json
   [BLOCKED] TG-CIR native cache missing: data/fashionIQ_dataset/test_queries_dress.pkl
   [BLOCKED] TG-CIR native cache missing: data/fashionIQ_dataset/test_targets_dress.pkl
   [BLOCKED] TG-CIR native cache missing: data/fashionIQ_dataset/test_queries_shirt.pkl
   [BLOCKED] TG-CIR native cache missing: data/fashionIQ_dataset/test_targets_shirt.pkl
   [BLOCKED] TG-CIR native cache missing: data/fashionIQ_dataset/test_queries_toptee.pkl
   [BLOCKED] TG-CIR native cache missing: data/fashionIQ_dataset/test_targets_toptee.pkl
   [BLOCKED] TG-CIR checkpoint missing or empty: workbench/artifacts/checkpoints/tgcir/tgcir-fashioniq-unresolved.pt
   [BLOCKED] checkpoint mapping unresolved: https://drive.google.com/file/d/1OdZTtJqy-RTpYXBaq5ThmH3IBnGvYCyi/view does not verify FashionIQ archive member for workbench/artifacts/checkpoints/tgcir/tgcir-fashioniq-unresolved.pt; checkpoint_mapping_status=UNVERIFIED
   [BLOCKED] author-verified checkpoint SHA-256 is unavailable; whole-model pickle loading is forbidden
   [BLOCKED] OpenAI CLIP ViT-B/16 runtime asset/cache provenance and local availability are unverified
   ```
3. **PC Readiness Execution**:
   ```bash
   $ python3 -m workbench.scripts.pc_readiness --model tgcir --json
   # state: BLOCKED_IMPLEMENTATION
   # reason: command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT
   ```
4. **Doctor Verification**:
   ```bash
   $ python3 -m workbench.scripts.doctor --model tgcir
   # [BLOCKED] runtime:tgcir: command unavailable
   # [BLOCKED] command:tgcir: COMMAND_BLOCKED_SOURCE_ENTRYPOINT
   # [BLOCKED] external-assets:tgcir: required evaluation assets unresolved
   ```
5. **Asset URL 404 vs 200 Verification**:
   ```bash
   # Declared in cluster_b2_encoder_family.yaml:432:
   $ curl -I "https://openaipublic.azureedge.net/clip/models/5806e77cd80cf8775b1045601c1438b6d00e063f3b21f8f47e222d83ff529a4e/ViT-B-16.pt"
   HTTP/2 404
   # Authentic OpenAI CLIP Azure URL:
   $ curl -I "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt"
   HTTP/2 200
   ```
6. **Spelling Correction Dictionaries SHA-256 Verification**:
   ```bash
   $ sha256sum workbench/third_party/TGCIR/correction_dict/*.json
   9e4a59877098fe26036b70759387be589990eb4068692427e890113dcba2edda  correction_dict_dress.json
   ca29e29e23c0a1987d6b121219d9580bdae9872df0ea219c5aea870ee2f8567e  correction_dict_shirt.json
   50451054df2d2365d732413bea40983c03c6f00ea4361ec258d16bd2c6238c24  correction_dict_shoes.json
   3dfbf15fa6d3a30dc0f5cf7e2908ca36d1775e42aa6598f1dba7899267dc1b71  correction_dict_toptee.json
   ```

---

## K. VERDICT

| Axis | Verdict | Key Evidence |
|---|---|---|
| **SOURCE** | **`SOURCE_AUDITED`** | Full repository inspected at pinned commit `65fa78eaf8cabe8197fcc22ccab42207c591bdee`. Dependency graph completely mapped. |
| **PROTOCOL** | **`PROTOCOL_VERIFIED`** | Source `datasets.py:148-160` and `test.py:63-68` strictly implement Protocol C (`fashioniq_val_split` pair-union gallery, reference excluded with `-10e10`). |
| **COMMAND** | **`COMMAND_BLOCKED`** | Pinned `test.py` lacks CLI/`__main__`. Evaluator reachable only during training loop (`train.py:200`). Adapter command raises `NotImplementedError`. |
| **CHECKPOINT** | **`CHECKPOINT_BLOCKED`** | Opaque Google Drive archive without checksum, member names, or category mapping. Untrusted whole-model pickle loading forbidden. |
| **QUERY EXPORT** | **`QUERY_EXPORT_IMPLEMENTABLE`** | Native `test.py:70` produces top-50 candidate indices via `np.argsort(-sims[i, :])[:50]`. Observer mapping direct, but blocked by lack of runnable execution. |
| **RELEASE READINESS** | **`BLOCKED_EXTERNAL_DEPENDENCY`** | Blocked on unmapped checkpoint archive, corrupted CLIP asset URL in registry, missing resized images, and unconfigured environment. |
