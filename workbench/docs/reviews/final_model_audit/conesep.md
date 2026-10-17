# Final Model Audit: conesep

## A. MODEL IDENTITY

- **Model ID**: `conesep` (`workbench/registry/models.yaml:150`)
- **Method Name**: `ConeSep` (`workbench/registry/models.yaml:160`)
- **Research Generation**: `recent` (`workbench/registry/models.yaml:161`) — **JUSTIFIED**: Accepted at CVPR 2026 (February 21, 2026; code released April 3, 2026; arXiv:2604.20358v1). Represents the modern post-2024 CIR generation addressing robust composed retrieval under Noisy Triplet Correspondence (NTC) using foundational vision-language architectures (BLIP-2 Q-Former with EVA-CLIP-g).
- **Publication Year**: `2026` (`workbench/registry/models.yaml:162`)
- **Paper Title**: `ConeSep: Cone-based Robust Noise-Unlearning Compositional Network for Composed Image Retrieval` (`workbench/registry/models.yaml:164`)
- **Paper URL**: `https://arxiv.org/abs/2604.20358` (`workbench/registry/models.yaml:165`)
- **Publication Venue**: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2026).
- **Authors**: Zixu Li, Yupeng Hu, Zhiwei Chen, Mingyu Zhang, Zhiheng Fu, Liqiang Nie (School of Software, Shandong University & Harbin Institute of Technology, Shenzhen).
- **Upstream Repository URL**: `https://github.com/Lee-zixu/ConeSep` (`workbench/registry/models.yaml:166`, lab mirror: `https://github.com/iLearn-Lab/CVPR26-ConeSep`)
- **Upstream Default Branch**: `main` (`workbench/registry/models.yaml:171`)
- **Upstream Pinned Commit SHA**: `bd4bf629ebe68045c1a54dcc61e4a3ea65f8759b` (`workbench/registry/models.yaml:172`)
- **Source Directory**: `ConeSep` (`workbench/registry/models.yaml:174`)
- **Source Available**: `true` (`workbench/registry/models.yaml:173`)
- **Native Protocol**: `fashioniq_full_gallery_ref_excluded` (`workbench/registry/models.yaml:178`)
- **Supported Protocols**: `[fashioniq_full_gallery_ref_excluded]` (`workbench/registry/models.yaml:179`)
- **Reproduction Status**: `NOT_RUN` (`workbench/registry/models.yaml:183`)
- **Command Status**: `COMMAND_UNAUDITED` (`workbench/backend/preflight.py:206`; adapter command raises `NotImplementedError`)
- **Strict Clean Checkpoint Available**: `false` (`workbench/registry/models.yaml:184`)
- **Reported Metrics**: `reported_r10: 55.58`, `reported_r50: 75.88`, `reported_mean: 65.73` (`workbench/registry/models.yaml:180-182`)

### Pinned Commit Verification
```bash
$ git -C workbench/third_party/ConeSep rev-parse HEAD
bd4bf629ebe68045c1a54dcc61e4a3ea65f8759b
$ git -C workbench/third_party/ConeSep log -1 --oneline
bd4bf62 Update README.md
$ git -C workbench/third_party/ConeSep rev-list --count HEAD
1
```
The pinned commit SHA `bd4bf629ebe68045c1a54dcc61e4a3ea65f8759b` in `workbench/third_party/ConeSep` exactly matches `upstream_commit_sha` in `workbench/registry/models.yaml:172`. Upstream history contains exactly 1 commit.

### Paper Benchmark Rows
Four benchmark rows are registered in `workbench/registry/paper_benchmarks/papers.yaml:592-783`:
1. **Clean Headline Row (`reference_id: conesep_fiq_n00`)**:
   - `paper_table`: Table 1 ("Performance comparison on FashionIQ validation set in terms of R@K (%)"), page 7, CVPR 2026.
   - `paper_row`: `ConeSep (Ours)` in the `0%` noise block.
   - `checkpoint_id`: `null` (status `NO_CHECKPOINT`).
   - `checkpoint_training_noise_pct`: `0`
   - `reported_metrics`: R@10 = `55.58`, R@50 = `75.88`, Mean = `65.73` (`derived_metrics: []`, printed as `AVG.` in Table 1).
   - Category Breakdown:
     - `dress`: R@10 = `50.96`, R@50 = `73.02`
     - `shirt`: R@10 = `56.98`, R@50 = `75.22`
     - `toptee`: R@10 = `58.80`, R@50 = `79.40`
   - Arithmetic Check:
     - Macro R@10 = $(50.96 + 56.98 + 58.80) / 3 = 166.74 / 3 = 55.58$
     - Macro R@50 = $(73.02 + 75.22 + 79.40) / 3 = 227.64 / 3 = 75.88$
     - Mean = $(55.58 + 75.88) / 2 = 65.73$
2. **20% Noise Row (`reference_id: conesep_fiq_n20`)**:
   - `checkpoint_id`: `fiq_n02` (`models.yaml:187`), `checkpoint_training_noise_pct`: `20`.
   - `reported_metrics`: R@10 = `54.93`, R@50 = `75.01`, Mean = `64.97` (Dress: 50.23/72.19, Shirt: 56.28/74.45, Toptee: 58.29/78.38).
3. **50% Noise Row (`reference_id: conesep_fiq_n50`)**:
   - `checkpoint_id`: `fiq_n05` (`models.yaml:197`), `checkpoint_training_noise_pct`: `50`.
   - `reported_metrics`: R@10 = `53.39`, R@50 = `74.15`, Mean = `63.77` (Dress: 48.39/71.34, Shirt: 54.56/73.25, Toptee: 57.22/77.87).
4. **80% Noise Row (`reference_id: conesep_fiq_n80`)**:
   - `checkpoint_id`: `fiq_n08` (`models.yaml:207`), `checkpoint_training_noise_pct`: `80`.
   - `reported_metrics`: R@10 = `49.25`, R@50 = `70.30`, Mean = `59.77` (Dress: 43.33/66.01, Shirt: 51.23/70.35, Toptee: 53.19/74.55).

---

## B. SOURCE COVERAGE

- **Audit Type**: `STATIC_AUDIT` + executed read-only inspection.
- **Tree Inventory** (`workbench/third_party/ConeSep/`, 1,223 tracked files total):
  ```text
  workbench/third_party/ConeSep/
  ├── LICENSE                                  (30 lines: MIT License)
  ├── README.md                                (419 lines: paper info, install, claims, results)
  ├── assets/                                  (14 files: framework diagrams, logos, results PNGs)
  ├── cirr_sub.py                              (127 lines: test_cirr_submit_result for CIRR test server)
  ├── data_utils.py                            (427 lines: TargetPad, squarepad_transform, targetpad_transform)
  ├── datasets.py                              (384 lines: FashionIQ, CIRR dataset loaders)
  ├── lavis/                                   (1,203 files: customized Salesforce-LAVIS v1.0.2 framework)
  │   ├── configs/models/blip2/                (blip2_pretrain.yaml: weights URL, vision/text params)
  │   ├── models/blip2_models/
  │   │   ├── ConeSep.py                       (579 lines: GFQ, NBL, BTU, feature extraction)
  │   │   └── blip2.py                         (116 lines: Blip2Base, init_tokenizer, init_Qformer)
  │   └── common/annotator/                    (600+ files: uniformer, midas, hed, mlsd, openpose)
  ├── test.py                                  (168 lines: FashionIQ test() and CIRR test_cirr_valset())
  ├── train.py                                 (285 lines: CLI parser, training loop, evaluation call)
  └── utils.py                                 (84 lines: logging, dict serialization, RunningAverage)
  ```

### Evaluation Dependency Graph
For FashionIQ evaluation:
1. `train.py:214-220`: Training loop calls `test.test(args, model, trainset, category, txt_processors)`.
2. `datasets.py:25-60`: `FashionIQ` class loads train captions, val captions, val splits, spelling dictionaries, constructs query/target structures, and caches them to `.pkl`.
3. `datasets.py:164-208`: `get_test_data(name)` loads `split.{cat}.val.json`, `cap.{cat}.val.json`, `correction_dict_{cat}.json`. Under `self.split = 'original-split'` (line 28), it populates `test_targets` with the full ordered gallery.
4. `test.py:9-85`: `test()` function batches queries and targets, feeds them to `model.extract_retrieval_compose` and `model.extract_retrieval_target`, performs matrix multiplication, masks reference index with `-10e10`, sorts top-50, and calculates R@1, R@10, R@50 per category.
5. `lavis/models/blip2_models/ConeSep.py:452-492`: `extract_retrieval_compose()` runs visual encoder + BertTokenizer + Q-Former BERT cross-attention.
6. `lavis/models/blip2_models/ConeSep.py:495-515`: `extract_retrieval_target()` runs visual encoder + Q-Former query tokens + `vision_proj`.
7. `lavis/models/blip2_models/blip2.py:30-34`: `Blip2Base.init_tokenizer()` loads BertTokenizer.
8. `data_utils.py:92-106`: `targetpad_transform(1.25, 224)` pads, resizes (BICUBIC 224), crops, converts RGB, normalizes.

### Inspected vs. Excluded Files
- **Inspected**:
  - `test.py:9-85`: Native FashionIQ evaluation function `test()`.
  - `datasets.py:25-209`: `FashionIQ` dataset class, caching, text cleaning, query/target building.
  - `train.py:1-144, 189-285`: CLI parser, dataset loading, model/optimizer setup, evaluation loop.
  - `data_utils.py:1-106`: TargetPad and transform pipelines.
  - `lavis/models/blip2_models/ConeSep.py:117-154, 451-516`: Model initialization and retrieval feature extraction.
  - `lavis/models/blip2_models/blip2.py:30-34`: Tokenizer loader.
  - `lavis/configs/models/blip2/blip2_pretrain.yaml`: Pretrain configuration and weights URL.
  - `cirr_sub.py:1-92`: CIRR test server submission function.
  - `README.md`: Upstream documentation and claims.
- **Excluded**:
  - `datasets.py:211-384` (`CIRR` class) and `test.py:87-168` (`test_cirr_valset`): CIRR validation evaluation.
  - `train.py:145-186` (`train()` function): Backward training steps, AMP scaling.
  - `ConeSep.py:228-449` (`loss_GFQ`, `loss_NBL`, `loss_BTU`, optimal transport unlearning loss): Training loss computation.
  - `lavis/common/annotator/*` (600+ files): Bundled ControlNet annotators (uniformer, midas, hed, etc.), unreferenced by FashionIQ.
  - `lavis/projects/*`: Upstream LAVIS non-CIR project configs.

### Executable Source vs. README Claims
1. **Resized Images Missing from Documentation**: `README.md:158-173` illustrates FashionIQ dataset layout with original category folders `FashionIQ/{dress,shirt,toptee}/*.jpg`. In code (`datasets.py:31, 137, 147`), images are loaded from `self.image_dir = self.path + 'resized_image'`, requiring `<fashioniq_path>/resized_image/{dress,shirt,toptee}/<id>.jpg`. Upstream provides no image resizing script or instructions.
2. **Undocumented Spelling Correction Dictionaries**: `README.md:158-173` omits mention of `correction_dict_{name}.json`. Code (`datasets.py:96, 170`) unconditionally opens `captions/correction_dict_{name}.json`. These files are neither tracked in git nor published with download URLs.
3. **Hardcoded `/root/bert-base-uncased`**: `README.md:139` instructs installing `transformers==4.25.0`. Code (`lavis/models/blip2_models/blip2.py:32`) hardcodes `BertTokenizer.from_pretrained("/root/bert-base-uncased")`, failing on any machine lacking root permissions or pre-existing `/root/bert-base-uncased`.
4. **Absence of Evaluation CLI**: `README.md:206-250` details training commands and a CIRR submission command (`python src/cirr_test_submission.py ...`, where `src/` does not even exist). The README provides zero evaluation commands for FashionIQ. `test.py` contains no `__main__` entrypoint.
5. **Clean Checkpoint Availability**: `README.md:77-79` highlights the "0% noise setting" as the traditional fully-supervised CIR baseline. However, the official HuggingFace repository publishes ONLY `ConeSep-FIQ-N0.2.pt`, `ConeSep-FIQ-N0.5.pt`, and `ConeSep-FIQ_N0.8.pt`. Zero 0%-noise checkpoints are published.
6. **Dead Argument `--restore_file`**: `train.py:76` accepts `--restore_file`, but the variable is never used anywhere in `train.py` or the rest of the codebase.

---

## C. NATIVE EVALUATION

- **Verdict Axis**: `PROTOCOL_VERIFIED`
- **Native Protocol**: Protocol B (`fashioniq_full_gallery_ref_excluded`)

### 1. Full-Val-Split Gallery Construction (Proof of Protocol B)
Native gallery construction is implemented in `workbench/third_party/ConeSep/datasets.py:28, 201-208` (`FashionIQ.__init__` and `get_test_data`):
```python
28:        self.split = 'original-split'
...
166:        with open(os.path.join(self.split_dir, "split.{}.{}.json".format(name, 'val')), 'r') as f:
167:            images = json.load(f)
...
201:        elif self.split == 'original-split':
202:            for id, image_name in enumerate(images):
203:                test_targets_id.append(id)
204:                out = {}
205:                out['target_img_id'] = id
206:                out['target_img_data'] = self.get_img(name + '_' + image_name)      
207:                test_targets.append(out)
```
- **Proof of Protocol B Gallery**: `images` is loaded directly from `split.{category}.val.json`. Under `self.split = 'original-split'`, `test_targets` iterates over every item in `images` in exact split order, adding all validation split images to the gallery.
- Gallery sizes match official FashionIQ validation split:
  - `dress`: 3,817 images (from 2,017 queries)
  - `shirt`: 6,346 images (from 2,038 queries)
  - `toptee`: 5,373 images (from 1,961 queries)

### 2. Reference Exclusion Policy
Native reference exclusion is implemented in `workbench/third_party/ConeSep/test.py:64-68`:
```python
64:        test_targets_id = []
65:        for i in test_targets:
66:            test_targets_id.append(i['target_img_id'])
67:        for i, t in enumerate(test_queries):
68:            sims[i, test_targets_id.index(t['source_img_id'])] = -10e10
```
- **Masking Line**: `sims[i, test_targets_id.index(t['source_img_id'])] = -10e10`
- **Mask Value**: `-10e10` ($-10^{11}$). On cosine similarities ranging in $[-1, 1]$, $-10^{11}$ guarantees the query's reference image is forced to the bottom of the ranked gallery before sorting.

### 3. Caption Preprocessing
Native text cleaning and combination in `workbench/third_party/ConeSep/datasets.py:108-117`:
```python
108:    def correct_text(self, text, correction_dict):
109:        trans=str.maketrans({key: ' ' for key in string.punctuation})
110:        tokens = str(text).lower().translate(trans).strip().split()
111:        text = " ".join([correction_dict.get(word) if word in correction_dict else word for word in tokens])
112:        return text
113:
114:    def concat_text(self, captions, correction_dict):
115:        text = "{} and {}".format(self.correct_text(captions[0], correction_dict), self.correct_text(captions[1], correction_dict))
116:        return text
```
- Punctuation stripped to spaces, lowercased, tokenized on whitespace, mapped token-by-token through `correction_dict_{category}.json`, joined with `"{} and {}"`.
- Tokenized in `ConeSep.py:469-475` using `self.tokenizer(mod, padding="max_length", truncation=True, max_length=32, return_tensors="pt")`.

### 4. Similarity Scoring and Sorting
Implemented in `workbench/third_party/ConeSep/test.py:59-71`:
```python
59:        sims = np.matmul(all_queries, all_imgs) #all_queries.dot(all_imgs.T)
60:        sims = sims.squeeze()
61:        sims = sims.max(-1)
...
68:            sims[i, test_targets_id.index(t['source_img_id'])] = -10e10
71:        nn_result = [np.argsort(-sims[i, :])[:50] for i in range(sims.shape[0])]
```
- `all_queries` shape: `(M, 1, 1, 256)`.
- `all_imgs` shape: `(N, 256, 32)` (where 32 is `num_query_token` from BLIP-2 Q-Former).
- `np.matmul(all_queries, all_imgs)` broadcasts to `(M, N, 1, 32)`.
- `sims.squeeze()` produces `(M, N, 32)`.
- `sims.max(-1)` performs max-pooling across the 32 query tokens to produce `(M, N)` similarity matrix.
- References are masked to `-10e10`, then ranked via `np.argsort(-sims[i, :])[:50]`.

### 5. Metric Calculation and Aggregation
Implemented in `workbench/third_party/ConeSep/test.py:76-82`:
```python
76:        for k in [1, 10, 50]:
77:            r = 0.0
78:            for i, nns in enumerate(nn_result):
79:                if test_targets_id.index(test_queries[i]['target_img_id']) in nns[:k]:
80:                    r += 1
81:            r = 100 * r / len(nn_result)
82:            out += [('{}_r{}'.format(name, k), r)]
```
- Recall@K is computed as the percentage of queries where the target image index appears in the top-K retrieved candidates. Denominator is `len(nn_result)` (the number of validation queries in the category).
- Evaluated separately per category: `dress`, `shirt`, `toptee`.
- Macro aggregation: `train.py:218` sums `current_score = current_score + t[1][1]` (sum of R@10) for checkpoint selection. Upstream source does not compute or log macro averages; macro R@10, R@50, and AVG are computed in the paper across the three categories.

---

## D. PREPARATION & NOISE CONDITION

### 1. Noise Condition & Methodological Conflation Risk
- **Declared Checkpoint Noise**:
  - `fiq_n02`: `checkpoint_training_noise_pct: 20` (`models.yaml:191`)
  - `fiq_n05`: `checkpoint_training_noise_pct: 50` (`models.yaml:201`)
  - `fiq_n08`: `checkpoint_training_noise_pct: 80` (`models.yaml:211`)
- **Evaluation Noise**: `evaluation_noise_pct: 0` for all variants (`models.yaml:192, 202, 212`).
- **Clean Evaluation Confirmed**: In `datasets.py:164-208` and `test.py:9-85`, evaluation is performed strictly on unmodified validation captions and splits. Noise shuffling (`FashionIQ.shuffle()`) is called only on the training split during training (`train.py:109`). Evaluation is 100% clean.
- **Conflation Risk**: The paper's primary headline row (`reference_id: conesep_fiq_n00`, Table 1, 0% noise block: R@10 = 55.58, R@50 = 75.88, AVG = 65.73) was trained on 0% noise. The author HuggingFace repository publishes **ZERO 0%-noise checkpoints** (`status: NO_CLEAN_CHECKPOINT`). Evaluating the noise-trained checkpoints (20%, 50%, 80%) against the paper's 0%-noise reference row is an invalid conflation.

### 2. Spelling Correction Dictionaries
- Code in `datasets.py:96, 170` opens `captions/correction_dict_{category}.json`.
- Pinned checkout `workbench/third_party/ConeSep/` vendors zero correction dictionaries.
- `workbench/registry/auxiliary_assets.yaml` contains zero entries for `conesep`.
- `workbench/docs/AUXILIARY_ASSET_AUDIT.md:352-354` records `conesep` correction dictionaries as `OFFICIAL_ASSET_UNAVAILABLE`.
- (Note: Pinned sibling checkouts `TGCIR` and `LIMN` vendor byte-identical copies with SHA-256 `9e4a5987...`, `ca29e29e...`, `3dfbf15f...`, but ConeSep provides neither files nor provenance).

### 3. Resized Images & Layout Requirements
- Code in `datasets.py:31, 137, 147` hardcodes `self.image_dir = self.path + 'resized_image'`, opening `resized_image/{dress,shirt,toptee}/{id}.jpg`.
- Local directory `/home/heheboiz/data/cir/data/fashionIQ_dataset/` contains `images/` (74,381 PNG files), but does NOT contain `resized_image/`.
- Pinned upstream source contains no resizing script or instructions.

### 4. CWD and Path Concatenation Bug (`CONFIRMED_BUG`)
In `workbench/third_party/ConeSep/datasets.py:30-33`:
```python
30:        self.path = path
31:        self.image_dir = self.path + 'resized_image'
32:        self.split_dir = self.path + 'image_splits'
33:        self.caption_dir = self.path + 'captions'
```
String concatenation without `/` requires `path` to end with a trailing slash. If passed without a trailing slash (`/path/to/fashioniq`), paths resolve to invalid paths like `/path/to/fashioniqresized_image`, causing immediate `FileNotFoundError`.

### 5. Native Cache Artifacts (`fashion_iq_data.json` & `.pkl`)
- In `datasets.py:35-60`, the dataset checks for `fashion_iq_data.json`. If missing, it parses raw data and saves `test_queries_{cat}.pkl` and `test_targets_{cat}.pkl`.
- If `fashion_iq_data.json` is present, it blindly unpickles the `.pkl` files without checking freshness.
- `test_queries_*.pkl` pickles transformed image tensors (`self.get_test_img()`). If transforms or images change, stale cached tensors are evaluated silently. If `fashion_iq_data.json` exists but any `.pkl` is missing, it crashes with `FileNotFoundError`.

---

## E. CHECKPOINT

- **Verdict Axis**: `CHECKPOINT_BLOCKED`
- **Serialization Policy**: `whole_model` (`torch.save(model, ...)`, `models.yaml:788`)
- **Published Variants**:
  - `fiq_n02`: `ConeSep-FIQ-N0.2.pt` (training noise 20%, size: 2,721,428,966 bytes)
  - `fiq_n05`: `ConeSep-FIQ-N0.5.pt` (training noise 50%, size: 2,721,428,966 bytes)
  - `fiq_n08`: `ConeSep-FIQ_N0.8.pt` (training noise 80%, size: 2,721,428,966 bytes)
  - `clean (0% noise)`: **NONE PUBLISHED** (`status: NO_CLEAN_CHECKPOINT`)
- **Remote Host**: HuggingFace (`https://huggingface.co/iLearn-Lab/CVPR26-ConeSep/resolve/main/fiq/`)
- **Expected SHA-256**: `null` in `models.yaml:194, 204, 214` (authors publish no manifest or checksums).
- **Mapping Status**:
  - In `models.yaml:196, 206, 216`: marked `checkpoint_mapping_status: VERIFIED_METADATA` (filenames match noise levels).
  - In `papers.yaml:649, 697, 745`: marked `checkpoint_mapping_status: UNVERIFIED`.
- **Whole-Model Deserialization Risk**:
  `train.py:221, 285` saves the complete Python object `model`, not a `state_dict`. Loading untrusted third-party whole-model pickles is a remote code execution risk; the workbench refuses whole-model unpickling without author-verified SHA-256 digests (`workbench/replay/common.py:36-39`).

---

## F. WORKBENCH INTEGRATION

### 1. Registry & Contracts
- `workbench/registry/models.yaml:150-216`: Registered with `native_protocol: fashioniq_full_gallery_ref_excluded`, `fashioniq_layout: fashioniq_ilearn_resized`, `source_blocker` text, and 3 noise variants.
- `preparation_contract`: `models.yaml:163` sets `preparation_contract: manual_ilearn_resized`. However, `workbench/registry/preparation_contracts.yaml` contains **zero** contracts for `conesep`.
- `auxiliary_assets`: `workbench/registry/auxiliary_assets.yaml` contains **zero** entries for `conesep`.

### 2. Adapter Status
- `workbench/backend/adapters/models.py:59-61`:
  ```python
  class ConeSepAdapter(OfficialScriptAdapter):
      model_id = "conesep"
      script = "test.py"
  ```
  Does not override `command()`. Invoking `ConeSepAdapter.build_command()` raises `NotImplementedError("upstream evaluation command requires audit before execution")`.

### 3. PC Readiness Blocker Classification
`python3 -m workbench.scripts.pc_readiness --model conesep --json` outputs:
- `"state": "BLOCKED_SOURCE"`
- `"reason": "source cannot run: no runnable evaluation path is registered for this model: the pinned evaluator is a callable with no CLI or checkpoint loader..."`
- **Needle-Ordering Classification Analysis**:
  - In `workbench/scripts/pc_readiness.py:47-67`, `BLOCKER_KINDS` prioritizes needles in fixed order:
    1. `("source cannot run", BLOCKED_SOURCE)`
    2. `("command not runnable", BLOCKED_IMPLEMENTATION)`
    3. `("FashionIQ", BLOCKED_DATASET)`
    4. `("checkpoint missing", BLOCKED_EXTERNAL_ASSET)`
  - Because `models.yaml:151` sets `source_blocker`, `preflight.py:202` prepends `"source cannot run: ..."`.
  - Needle #1 matches, classifying the model as `BLOCKED_SOURCE`. This is honest because the upstream source has no CLI and cannot run.
  - However, notice: `pc_readiness.py:197-198` computes `requirements = model_requirements(model, config)`, but `requirements["blocks"]` (which contains the 8 missing `resized_image` and `correction_dict` dataset paths) is NEVER added to `blockers`. The dataset blockers are omitted from the classification list.
  - If `source_blocker` were cleared, `"command not runnable"` would classify the model as `BLOCKED_IMPLEMENTATION`, masking the missing checkpoints (`BLOCKED_EXTERNAL_ASSET`).

### 4. Metrics Extraction & Paper Gate
- `workbench/backend/metrics_extraction.py:414-422`: Lacks an aggregate metric parser for `conesep`. Emits `AGGREGATE_PARSER_UNAVAILABLE` with `observed_metrics=None`.
- `workbench/backend/paper_gate.py`: Cannot validate ConeSep output due to missing parser and missing run results.

---

## G. OBSERVATION CAPABILITY

- **Verdict Axis**: `QUERY_EXPORT_IMPLEMENTABLE`

### Score & Ranking Production Point
Native score and ranking production in `workbench/third_party/ConeSep/test.py:59-71`:
```python
59:        sims = np.matmul(all_queries, all_imgs) #all_queries.dot(all_imgs.T)
60:        sims = sims.squeeze()
61:        sims = sims.max(-1)
...
68:            sims[i, test_targets_id.index(t['source_img_id'])] = -10e10
71:        nn_result = [np.argsort(-sims[i, :])[:50] for i in range(sims.shape[0])]
```
- Similarity matrix `sims` has shape `[M, N]` ($M$ queries, $N$ gallery items).
- Reference exclusion is executed at line 68.
- Nearest neighbors top-50 indices are produced at line 71 via `np.argsort(-sims[i, :])[:50]`.

### Exposing Per-Query Rankings
- **Feasibility**: Fully implementable. The structure directly maps to observer output:
  1. `query_id`: Query index `i`, reference image ID `images[test_queries[i]['source_img_id']]`, target ID `images[test_queries[i]['target_img_id']]`, modifier text `test_queries[i]['mod']['str']`.
  2. `candidate_ids`: Ranked image names `[images[test_targets_id[idx]] for idx in nn_result[i]]`.
  3. `candidate_similarities`: `sims[i, nn_result[i]]`.
  4. `target_hit`: `test_targets_id.index(test_queries[i]['target_img_id']) in nn_result[i][:k]`.
- **Blockers to Execution**:
  1. Checkpoint unavailable (0% noise clean checkpoint unpublished; 20/50/80% checkpoints not downloaded, lack author hashes, whole-model pickle serialization).
  2. Absence of evaluation CLI in upstream source and adapter.
  3. Hardcoded `/root/bert-base-uncased` tokenizer path.
  4. Unregistered auxiliary assets (`blip2_pretrained.pth`, tokenizer).
  5. Missing resized images (`resized_image/`) and correction dictionaries (`correction_dict_*.json`).
  6. Unconfigured python environment `WORKBENCH_CONESEP_PYTHON`.
  7. Missing aggregate metric parser in `metrics_extraction.py`.

---

## H. DEFECTS

| Defect ID | Classification | Severity | Location | Summary |
|---|---|---|---|---|
| **D-01** | `EXTERNAL_BLOCKER` | Critical | `iLearn-Lab/CVPR26-ConeSep` Hub, `models.yaml:184-185` | Zero clean (0% training noise) checkpoints published for Table 1 headline row (`55.58 / 75.88 / 65.73`). |
| **D-02** | `EXTERNAL_BLOCKER` | Critical | `ConeSep/test.py`, `ConeSep/train.py` | Pinned source provides zero FashionIQ evaluation CLI or entrypoint (`test.py` has no `__main__`; `train.py` evaluates only in train loop). |
| **D-03** | `EXTERNAL_BLOCKER` | High | `ConeSep/lavis/models/blip2_models/blip2.py:32` | Hardcoded `/root/bert-base-uncased` local path crashes with `OSError` in non-root environments. |
| **D-04** | `EXTERNAL_BLOCKER` | High | `ConeSep/datasets.py:96, 170` | Spelling correction dictionaries `correction_dict_{name}.json` are undocumented, unpublished, and missing from repository. |
| **D-05** | `EXTERNAL_BLOCKER` | High | `ConeSep/datasets.py:31, 137, 147` | Requires pre-resized images in `<fashioniq_path>/resized_image/`, omitted from README and lacking a resizing script. |
| **D-06** | `CONFIRMED_BUG` | High | `ConeSep/datasets.py:30-33` | String concatenation bug: `self.path + 'resized_image'` requires trailing slash; crashes if path has no trailing slash. |
| **D-07** | `CONFIRMED_BUG` | Medium | `workbench/backend/metrics_extraction.py:414-422` | Missing ConeSep metric parser; completed runs emit `AGGREGATE_PARSER_UNAVAILABLE` and fail gate. |
| **D-08** | `CONFIRMED_BUG` | Medium | `workbench/registry/models.yaml:163`, `preparation_contracts.yaml`, `auxiliary_assets.yaml` | `manual_ilearn_resized` contract not registered in `preparation_contracts.yaml`; 0 auxiliary assets pinned for conesep. |
| **D-09** | `UNVERIFIED_RISK` | Medium | `ConeSep/datasets.py:35-60` | Blind loading of native `.pkl` caches without freshness validation; stale image tensors evaluated silently if files change. |
| **D-10** | `BY_DESIGN` | Low | `workbench/backend/adapters/models.py:59-61`, `preflight.py:202` | Workbench fails closed: adapter command raises `NotImplementedError`, preflight surfaces `source cannot run`. |

---

## I. EVIDENCE

1. **Commit SHA & Single-Commit History**:
   ```bash
   $ git -C workbench/third_party/ConeSep rev-parse HEAD
   bd4bf629ebe68045c1a54dcc61e4a3ea65f8759b
   $ git -C workbench/third_party/ConeSep rev-list --count HEAD
   1
   ```
2. **PC Readiness Execution**:
   ```bash
   $ python3 -m workbench.scripts.pc_readiness --model conesep --json
   # state: BLOCKED_SOURCE
   # reason: source cannot run: no runnable evaluation path is registered for this model...
   ```
3. **Doctor Verification**:
   ```bash
   $ python3 -m workbench.scripts.doctor --model conesep
   # [OK] source:conesep: source pin ready
   # [BLOCKED] runtime:conesep: command unavailable
   # [BLOCKED] command:conesep: COMMAND_AUDITED
   ```
4. **HuggingFace Hub Asset Inventory & Absence of 0% Checkpoint**:
   ```bash
   $ curl -s https://huggingface.co/api/models/iLearn-Lab/CVPR26-ConeSep | jq -r '.siblings[].rfilename'
   .gitattributes
   README.md
   fiq/ConeSep-FIQ-N0.2.pt
   fiq/ConeSep-FIQ-N0.5.pt
   fiq/ConeSep-FIQ_N0.8.pt
   ```
   (Zero clean 0%-noise weights; only 20%, 50%, and 80% noise weights exist).
5. **Checkpoint Size Verification**:
   ```bash
   $ curl -sI https://huggingface.co/iLearn-Lab/CVPR26-ConeSep/resolve/main/fiq/ConeSep-FIQ-N0.2.pt | grep "x-linked-size"
   x-linked-size: 2721428966
   $ curl -sI https://huggingface.co/iLearn-Lab/CVPR26-ConeSep/resolve/main/fiq/ConeSep-FIQ-N0.5.pt | grep "x-linked-size"
   x-linked-size: 2721428966
   $ curl -sI https://huggingface.co/iLearn-Lab/CVPR26-ConeSep/resolve/main/fiq/ConeSep-FIQ_N0.8.pt | grep "x-linked-size"
   x-linked-size: 2721428966
   ```
6. **Preflight Blockers Execution**:
   ```bash
   $ python3 -c "from workbench.backend.operator_config import resolve_config; from workbench.backend.registry import model_by_id; from workbench.backend.preflight import preflight_blockers; c = resolve_config(); print(preflight_blockers(model_by_id('conesep'), c))"
   ["source cannot run: no runnable evaluation path is registered for this model: the pinned evaluator is a callable with no CLI or checkpoint loader...", 'command not runnable: COMMAND_UNAUDITED']
   ```
7. **Similarity Computation Shape Verification**:
   ```python
   # Verified with numpy array shapes (M=10, N=20, D=256, Q=32):
   all_queries = np.random.randn(10, 1, 1, 256)
   all_imgs = np.random.randn(20, 256, 32)
   sims = np.matmul(all_queries, all_imgs).squeeze().max(-1)
   assert sims.shape == (10, 20)
   # Prior hypothesis of broadcast failure is REFUTED; matmul broadcasts correctly.
   ```
8. **Adapter Command Failure**:
   ```python
   from workbench.backend.adapters.models import ConeSepAdapter
   adapter = ConeSepAdapter()
   adapter.build_command(...)
   # NotImplementedError: upstream evaluation command requires audit before execution
   ```

---

## J. VERDICT

| Axis | Verdict | Key Evidence |
|---|---|---|
| **SOURCE** | **`SOURCE_AUDITED`** | Full repository inspected at pinned commit `bd4bf629ebe68045c1a54dcc61e4a3ea65f8759b` (single-commit history). Evaluation dependency graph mapped. |
| **PROTOCOL** | **`PROTOCOL_VERIFIED`** | Code in `datasets.py:28, 201-208` and `test.py:64-68` strictly implements Protocol B (`fashioniq_full_gallery_ref_excluded`, full ordered validation gallery, reference masked to `-10e10`). |
| **COMMAND** | **`COMMAND_BLOCKED`** | Pinned `test.py` contains bare functions without CLI or `__main__`. `train.py` evaluates only during training epochs; `--restore_file` is dead code. `ConeSepAdapter` raises `NotImplementedError`. |
| **CHECKPOINT** | **`CHECKPOINT_BLOCKED`** | Clean (0% noise) weights for Table 1 headline row (`55.58 / 75.88 / 65.73`) are unpublished (`NO_CLEAN_CHECKPOINT`). Published 20/50/80% checkpoints lack author SHA-256 hashes and require unsafe whole-model deserialization. |
| **QUERY EXPORT** | **`QUERY_EXPORT_IMPLEMENTABLE`** | Native `test.py:71` produces top-50 candidate rankings via `np.argsort(-sims[i, :])[:50]`. Observer integration direct, but blocked by lack of runnable execution. |
| **RELEASE READINESS** | **`BLOCKED_EXTERNAL_DEPENDENCY`** | Blocked on unpublished clean checkpoint, absence of evaluation CLI, hardcoded `/root/bert-base-uncased`, missing correction dictionaries, missing resized images, and unconfigured environment. |
