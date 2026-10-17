# Final Model Audit: ENCODER

## A. MODEL IDENTITY

- **Model ID**: `encoder`
- **Method Name**: `ENCODER` (`workbench/registry/models.yaml:431`)
- **Paper Title**: *ENCODER: Entity Mining and Modification Relation Binding for Composed Image Retrieval*
- **Paper Authors**: Zixu Li, Zhiwei Chen, Haokun Wen, Zhiheng Fu, Yupeng Hu, Weili Guan
- **Publication**: Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 39 No. 5 (AAAI-25), pp. 5101–5109, published April 11, 2025 (`DOI: 10.1609/aaai.v39i5.32541`).
- **Paper URL**: `https://ojs.aaai.org/index.php/AAAI/article/view/32541` (`models.yaml:436`)
- **Upstream Repository URL**: `https://github.com/iLearn-Lab/AAAI25-ENCODER` (`models.yaml:437`)
- **Upstream Default Branch**: `main` (`models.yaml:438`)
- **Pinned Commit SHA**: `29a2a31d6a56f677bf450c3be7cdaef423fb7018` (`models.yaml:439`)
  - Observed checkout HEAD: `git -C workbench/third_party/ENCODER rev-parse HEAD` -> `29a2a31d6a56f677bf450c3be7cdaef423fb7018`. **EQUALS REGISTRY**.
  - Commit message: `Update README.md` (Mon May 25 18:00:29 2026 +0800).
  - Upstream tree count: `git rev-list --all --count` = 1 (squashed export). Upstream GitHub Git Trees API confirms identical tree content at this SHA.
- **Checkpoint Variant**:
  - `checkpoint_id`: `fashioniq`
  - `filename`: `fashioniq.pt`
  - `download_url`: `null`
  - `source_url`: `https://drive.google.com/drive/folders/1QuBQVIybLwn-qFSORhpBsnXy6ebhtNr3` (Google Drive folder)
  - `source_type`: `author_google_drive`
  - `expected_sha256`: `null`
  - `expected_size_bytes`: `null`
  - `status`: `URL_UNRESOLVED`
  - `checkpoint_mapping_status`: `UNVERIFIED`
- **Paper Benchmark Row**:
  - Reference ID: `encoder_fiq_clean_val_split` in `workbench/registry/paper_benchmarks/papers.yaml:1104-1166`.
  - Paper table: Table 1, page 5, row `ENCODER(Ours)`.
  - Transcribed metrics:
    - Dresses: R@10 = `51.51`, R@50 = `76.95`
    - Shirts: R@10 = `54.86`, R@50 = `74.93`
    - Tops&Tees: R@10 = `62.01`, R@50 = `80.88`
    - FashionIQ Avg (Macro): R@10 = `56.13`, R@50 = `77.59`
    - Arithmetic Mean: `(56.13 + 77.59) / 2 = 66.86` (labeled `WORKBENCH_DERIVED`).

---

## B. SOURCE COVERAGE

### 1. Pinned Source Inventory (`workbench/third_party/ENCODER/`)
- `evaluate_model.py` (84 lines): Official evaluation entrypoint.
- `datasets.py` (672 lines): Dataset classes for FashionIQ, Shoes, CIRR, Fashion200k.
- `test.py` (230 lines): Test feature extraction, normalization, similarity calculation, ranking, and recall calculation.
- `model_try2.py` (433 lines): Model architecture (`Backbone`, `Encoder`, `ScoreNet`, `BindingDecoder`).
- `token_wise_matching.py` (72 lines): Token-level similarity and matching logic.
- `utils.py` (190 lines): Utility helpers (`save_obj`, `load_obj`, `AverageMeter`).
- `train.py` (278 lines): Official training script.
- `cirr_test_submission.py` (98 lines): CIRR evaluation server submission generator.
- `requirements.txt` (304 lines): Package environment requirements.
- `README.md` (422 lines): Documentation, setup instructions, evaluation examples.
- `LICENSE` (21 lines): MIT License.
- `assets/` (14 image files): Diagrams and result screenshots, including `assets/fashion-results.png`.

### 2. Reachable Evaluation Dependency Graph
```
evaluate_model.py (entrypoint)
├── import datasets1 [BROKEN: MISSING FILE] -> intended datasets.py
│   ├── torch.utils.data.Dataset
│   ├── PIL.Image
│   ├── string, json, os
│   └── utils.py (save_obj, load_obj)
├── import test
│   ├── numpy, torch, tqdm
│   └── model.extract_retrieval_compose / extract_retrieval_target
├── from model_try2 import Encoder
│   ├── open_clip (ViT-B-32 with ./open_clip_pytorch_model.bin) [BROKEN: MISSING BINARY]
│   ├── transformers (CLIPTextModel, AutoTokenizer)
│   ├── token_wise_matching.py
│   └── einops
└── open_clip.create_model_and_transforms('ViT-B-32', pretrained='./open_clip_pytorch_model.bin')
```

### 3. Inspected vs. Excluded Files
- **Inspected**:
  - `evaluate_model.py`: Fully inspected; execution path, arguments, dataset loading, category loop.
  - `datasets.py`: Fully inspected; `FashionIQ` class, path concatenation, cache writing, preprocessing, gallery building.
  - `test.py`: Fully inspected; feature normalization, similarity dot product, reference masking, `np.argsort`, recall evaluation.
  - `model_try2.py`: Fully inspected; OpenCLIP instantiation, checkpoint loading, cross-attention.
  - `token_wise_matching.py`: Inspected; token-level interaction math.
  - `train.py`: Inspected; verified checkpoint saving logic, broken `import model`, and FashionIQ selection score.
  - `requirements.txt`: Inspected; verified library versions (`torch==1.13.1`, `torchvision==0.14.1`, `open-clip-torch==2.24.0`).
  - `assets/fashion-results.png`: Inspected; confirmed author-rendered table matches Table 1 numbers.
- **Excluded**:
  - `cirr_test_submission.py`: Excluded from runtime audit; targets CIRR test server submission, not FashionIQ.
  - `assets/logos/*`, `assets/cirr-results.png`, `assets/fashion200k-results.png`: Visual documentation assets only.

---

## C. NATIVE EVALUATION

### 1. ONE Shared Model Across All Three Categories (Contrast with LIMN)
- **Proof of Single Shared Model**:
  In `workbench/third_party/ENCODER/evaluate_model.py:60-75`:
  ```python
  60: model = torch.load(args.ckpt_path)
  61:
  62: Encoder_model = Encoder(hidden_dim=args.hidden_dim, dropout=args.dropout_rate, local_token_num=args.P, t = args.tau_, wc = args.wc, N_p = args.N_p, weighted=args.weighted)
  63: Encoder_model.load_state_dict(model, strict=False)
  64: Encoder_model = Encoder_model.cuda()
  65:
  66: dataset_list = load_dataset()
  ...
  72: if args.dataset in ['fashioniq']:
  73:     for ci, category in enumerate(['dress', 'shirt', 'toptee']):
  74:         t = test.test(args, Encoder_model, dataset_list[0], category)
  75:         print(t)
  ```
  And in `workbench/third_party/ENCODER/train.py:186-193`:
  ```python
  186: elif args.dataset in ['fashioniq']:
  187:     for ci, category in enumerate(['dress', 'shirt', 'toptee']):
  188:         t = test.test(args, model, dataset_list[0], category)
  189:         logging.info(t)
  190:         current_score = current_score + t[1][1]
  191:     torch.save(model, os.path.join(args.model_dir, "{}_{}_model.pt".format(args.dataset, args.i)))
  ```
- **Explicit Contrast with LIMN**:
  - In **LIMN**, training is performed independently per category, producing three distinct checkpoint files (`LIMN_dress.pt`, `LIMN_shirt.pt`, `LIMN_toptee.pt`). LIMN evaluation loads a different model weights file for each category.
  - In **ENCODER**, one single model is trained across FashionIQ, evaluated on all three categories within the same loop, and saved as a single checkpoint (`fashioniq.pt`).
  - `category_model_policy: shared` in `models.yaml:429` is **100% faithful to upstream source**.

### 2. Train Data Grouping and Cache Generation
- In `datasets.py:74-88`:
  If `os.path.join(self.path, 'fashion_iq_data.json')` does not exist:
  - Calls `self.train_init_process()`, which reads `cap.{name}.train.json` and `correction_dict_{name}.json` across `['dress', 'shirt', 'toptee']` (`datasets.py:102-106`).
  - Writes root-level cache `fashion_iq_data.json`.
  - Calls `get_test_data` for dress, shirt, toptee and pickles `test_queries_{cat}.pkl` and `test_targets_{cat}.pkl` directly into `self.path`.
- If `fashion_iq_data.json` exists, it unconditionally unpickles the saved PKL files without validation (`datasets.py:90-98`).

### 3. Pair-Union Gallery Construction (Protocol C)
- In `datasets.py:185-196`:
  ```python
  185: if self.split == 'val-split':
  186:     for i in test_queries:
  187:         if i['source_img_id'] not in test_targets_id:
  188:             test_targets_id.append(i['source_img_id'])
  189:         if i['target_img_id'] not in test_targets_id:
  190:             test_targets_id.append(i['target_img_id'])
  191:     
  192:     for i in test_targets_id:
  193:         out = {}
  194:         out['target_img_id'] = i
  195:         out['target_img_data'] = self.get_img(name + '_' + images[i], stage=1)       
  196:         test_targets.append(out)
  ```
  The gallery is the ordered unique union of references (`source_img_id`) and targets (`target_img_id`) appearing in the validation query pairs, in order of first appearance.

### 4. Reference Exclusion
- In `test.py:67-68`:
  ```python
  67: for i, t in enumerate(test_queries):
  68:     sims[i, test_targets_id.index(t['source_img_id'])] = -10e10
  ```
  The reference image is excluded from the ranking by setting its similarity to `-10e10` (`-1e11` float) before sorting.

### 5. Caption Preprocessing
- In `datasets.py:117-126`:
  - `correct_text(text, correction_dict)`:
    - Replaces all ASCII punctuation with spaces (`str.maketrans({key: ' ' for key in string.punctuation})`).
    - Lowercases text and splits by whitespace.
    - Replaces words found in `correction_dict`.
  - `concat_text(captions, correction_dict)`:
    - `"{} and {}".format(self.correct_text(captions[0], correction_dict), self.correct_text(captions[1], correction_dict))`

### 6. Scoring, Sorting, and Recall@K
- **Feature Normalization**: L2 normalization on query and gallery vectors (`test.py:55-58`):
  `all_queries[i, :] /= np.linalg.norm(all_queries[i, :])`
  `all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])`
- **Scoring**: Cosine similarity via matrix dot product (`test.py:62`):
  `sims = all_queries.dot(all_imgs.T)`
- **Sorting**: Descending argsort capped at top 50 (`test.py:71`):
  `nn_result = [np.argsort(-sims[i, :])[:50] for i in range(sims.shape[0])]`
- **Recall@K Calculation** (`test.py:75-81`):
  ```python
  75: for k in [1, 10, 50]:
  76:     r = 0.0
  77:     for i, nns in enumerate(nn_result):
  78:         if test_targets_id.index(test_queries[i]['target_img_id']) in nns[:k]:
  79:             r += 1
  80:     r = 100 * r / len(nn_result)
  81:     out += [('{}_r{}'.format(name, k), r)]
  ```

### 7. Macro Aggregation
- **Upstream Source**: Does NOT perform macro aggregation across categories and does NOT compute a mean. `evaluate_model.py:75` prints `out` separately for dress, shirt, toptee.
- **Paper Table 1**: Average R@10 (`56.13`) and Average R@50 (`77.59`) are exact unweighted arithmetic means across `{dress, shirt, toptee}`:
  - Avg R@10: `(51.51 + 54.86 + 62.01) / 3 = 56.1267 -> 56.13`
  - Avg R@50: `(76.95 + 74.93 + 80.88) / 3 = 77.5867 -> 77.59`
  - Reported Mean `66.86` = `(56.13 + 77.59) / 2` (workbench derived, not printed in the paper).

---

## D. PREPARATION

### 1. Dataset Layout (`fashioniq_ilearn_resized`)
- Evaluator reads:
  - `resized_image/{category}/{image_id}.jpg` (`datasets.py:148`)
  - `image_splits/split.{category}.val.json` (`datasets.py:159`)
  - `captions/cap.{category}.val.json` (`datasets.py:161`)
  - `captions/correction_dict_{category}.json` (`datasets.py:163`)
  - On first run (empty cache): `captions/cap.{category}.train.json` (`datasets.py:103`)
- Local active dataset (`data/fashionIQ_dataset`):
  - Contains canonical `images/` (74,381 `.png` files), `captions/`, `image_splits/`.
  - MISSING: `resized_image/` directory.
  - MISSING: `captions/correction_dict_{dress,shirt,toptee}.json`.
- Root path string concatenation requirement:
  - `datasets.py:68` uses `self.path + 'resized_image'`, requiring a trailing slash in the dataset root path.

### 2. Auxiliary Assets and Dependencies
- **OpenCLIP Pretrained ViT-B-32 Weights**:
  - `evaluate_model.py:40` and `model_try2.py:135` unconditionally load `pretrained=os.path.join(clip_path, 'open_clip_pytorch_model.bin')` where `clip_path = '.'`.
  - Required binary: `open_clip_pytorch_model.bin` (corresponding to HuggingFace `laion/CLIP-ViT-B-32-laion2B-s34B-b79K`).
  - Missing from pinned checkout and NOT registered in `workbench/registry/auxiliary_assets.yaml`.
- **Preparation Contract**:
  - Registry defines `preparation_contract: manual_ilearn_resized` (`models.yaml:434`).
  - `preparation_contracts.yaml` has NO row for `encoder`. `preparation_contract_for_model(m)` resolves to `None`.

---

## E. CHECKPOINT

### 1. Identity and Provenance
- Checkpoint filename: `fashioniq.pt`
- Source: Linked in `README.md:249` as a Google Drive folder (`https://drive.google.com/drive/folders/1QuBQVIybLwn-qFSORhpBsnXy6ebhtNr3`).
- Per-file URL: `null`
- File hash / size: `expected_sha256: null`, `expected_size_bytes: null`.
- Local presence: Missing (`workbench/artifacts/checkpoints/encoder/fashioniq.pt` does not exist).
- Status: `URL_UNRESOLVED`, `checkpoint_mapping_status: UNVERIFIED`.

### 2. Architecture and Loader
- Backbone: CLIP ViT-B/32 (`open-clip-torch==2.24.0`).
- Custom modules: LFF (Local Feature Filter / `ScoreNet`), EAB (Entity-Action Binding / `BindingDecoder`), MSC (Multi-Scale Cross-modal composition).
- Tunable hyperparameters (`evaluate_model.py:24-30`):
  - Default flags: `--P 3`, `--tau_ 0.1`, `--lambda_ 1.0`, `--eta_ 1.0`, `--kappa_ 0.5`, `--wc 2`, `--N_p 2`, `--weighted True`.
  - Note: Paper §4.1 states latent factor number $P=4$ and epochs 10; code comment at line 35 states `Shoes&FashionIQ:{3,2,2}` with argparse default `--P 3`.
- Loader mechanism:
  - `model = torch.load(args.ckpt_path)` (`evaluate_model.py:60`)
  - `Encoder_model.load_state_dict(model, strict=False)` (`evaluate_model.py:63`)
  - Unconditional GPU transfer: `Encoder_model = Encoder_model.cuda()` (`evaluate_model.py:64`).

---

## F. WORKBENCH INTEGRATION

### 1. Registry & Contracts
- `workbench/registry/models.yaml:415-476`:
  - `model_id`: `encoder`
  - `category_model_policy`: `shared` (accurate)
  - `strict_clean_checkpoint_available`: `false` (accurate; correctly computes to `False`)
  - `checkpoint_training_noise_pct`: `null` (accurate; paper has no noise block)
  - `preparation_contract`: `manual_ilearn_resized` (unregistered in `preparation_contracts.yaml`)
- `workbench/registry/paper_benchmarks/papers.yaml:1104-1166`:
  - `reference_id`: `encoder_fiq_clean_val_split`
  - `paper_id`: `encoder_aaai25`
  - `paper_title`: `'ENCODER: Entity Mining and Modification Relation Binding for Composed Image Retrieval'`
  - Metrics accurately transcribed from Table 1.

### 2. Adapter
- `workbench/backend/adapters/models.py:36-41`:
  ```python
  class EncoderAdapter(OfficialScriptAdapter):
      model_id = "encoder"
      script = "evaluate_model.py"

      def command(self, source, checkpoint, request: EvalRequest) -> list[str]:
          return [_runtime.python_executable(self.model), str(source / self.script), "--dataset", "fashioniq", "--fashioniq_split", "val-split", "--fashioniq_path", f"{request.dataset_root}/", "--ckpt_path", str(checkpoint)]
  ```
  - Trailing slash `f"{request.dataset_root}/"` correctly handles `datasets.py:68` concatenation.
  - Arguments match `evaluate_model.py` argparse.
  - Working directory: `source_cwd` resolves to `workbench/third_party/ENCODER`, matching `./open_clip_pytorch_model.bin` relative path requirement.

### 3. Preflight & Readiness
- `workbench/backend/preflight.py:211-215` explicitly checks:
  - `(source / "datasets1.py").is_file()` -> blocks on `ENCODER evaluator import missing`
  - `(source / "open_clip_pytorch_model.bin").is_file()` -> blocks on `ENCODER asset missing`
- `runtime_plan_blockers` (`workbench/backend/runtime.py:2503`) reports 4 blockers:
  1. `ENCODER evaluator import missing: .../datasets1.py`
  2. `ENCODER asset missing: .../open_clip_pytorch_model.bin`
  3. `checkpoint missing: .../artifacts/checkpoints/encoder/fashioniq.pt`
  4. `checkpoint mapping unresolved`
- Classification bug in `workbench/scripts/pc_readiness.py`:
  - `classify` returns `BLOCKED_DATASET` because `"FashionIQ"` matches the word in path `.../fashioniq.pt` before reaching `"checkpoint missing"` or `"missing"`.

### 4. Metrics Extraction & Gate
- `workbench/backend/metrics_extraction.py:402-422`:
  - **No parser exists for ENCODER**.
  - `extract_aggregate_metrics('encoder', ...)` returns `AGGREGATE_PARSER_UNAVAILABLE`, `observed_metrics=None`, `category_metrics=None`, `parity_status="PARSER_UNAVAILABLE"`.
- `workbench/backend/paper_gate.py:326-328`:
  - When `observed_metrics` is `None`, `evaluate_gate` returns `FAILED_OUTPUT_VALIDATION` ("the evaluator produced nothing to compare").

---

## G. OBSERVATION CAPABILITY

### 1. Score & Ranking Production Point
- In `workbench/third_party/ENCODER/test.py:61-72`:
  - **Similarity matrix**: `sims = all_queries.dot(all_imgs.T)` (`test.py:62`).
  - **Reference exclusion masking**: `sims[i, test_targets_id.index(t['source_img_id'])] = -10e10` (`test.py:68`).
  - **Ranking**: `nn_result = [np.argsort(-sims[i, :])[:50] for i in range(sims.shape[0])]` (`test.py:71`).
  - Line 72 is the exact, unpolluted observation point.

### 2. Variable Identity & Mapping
- `test_queries[i]`:
  - `source_img_id`: integer position in `split.{category}.val.json`
  - `target_img_id`: integer position in `split.{category}.val.json`
  - `mod['str']`: processed text string (`f"{clean_cap0} and {clean_cap1}"`)
  - Canonical reference image ID: `images[test_queries[i]['source_img_id']]`
  - Canonical target image ID: `images[test_queries[i]['target_img_id']]`
- `test_targets[j]`:
  - `target_img_id`: integer position in `split.{category}.val.json`
  - Canonical gallery image ID: `images[test_targets[j]['target_img_id']]`
- `test_targets_id`: list of integer indices in gallery order.
- `nn_result[i]`: list of 50 indices into `test_targets_id`.
  - Canonical retrieved image ID at rank $r$: `images[test_targets_id[nn_result[i][r]]]`
  - Retrieved similarity score: `sims[i, nn_result[i][r]]`

### 3. Captions
- Processed caption is in `test_queries[i]['mod']['str']`.
- Raw captions are NOT retained in `test_queries`.
- Because queries are processed in identical sequential order as `captions/cap.{category}.val.json`, query index $i$ maps directly to `raw_captions[i]['captions']`.

### 4. Per-Query Export Feasibility & Blockers
- **Feasibility**: High. A wrapper around `test.py:test` can tap `sims`, `nn_result`, and `test_queries` to export canonical query IDs, reference IDs, target IDs, raw captions, and ranked retrieved lists without altering model predictions.
- **Blockers**:
  - P0: Evaluator unimportable due to missing `datasets1.py`.
  - P0: Missing backbone weights `open_clip_pytorch_model.bin`.
  - P1: Checkpoint `fashioniq.pt` unverified and missing.
  - P1: CUDA hardware mandatory.

---

## H. DEFECTS

### 1. `evaluate_model.py:7` `import datasets1` (MISSING MODULE)
- **Status**: `CONFIRMED_BUG` (upstream code defect) / `EXTERNAL_BLOCKER` (evaluation blocker).
- **Location**: `workbench/third_party/ENCODER/evaluate_model.py:7,43,47,51,55`.
- **Precondition**: Attempting to execute `evaluate_model.py`.
- **Actual**: `ModuleNotFoundError: No module named 'datasets1'`.
- **Expected**: Evaluator imports the dataset loader from the repository.
- **Evidence**: `git ls-files` shows `datasets.py` exists, but NO `datasets1.py`. GitHub API confirms `datasets1.py` is absent from tree `29a2a31d6a56f677bf450c3be7cdaef423fb7018`. In `train.py:15`, the author wrote `import datasets`. In `evaluate_model.py:8`, the author imported from `model_try2`. This indicates `datasets1` was a local scratch/renamed file mistakenly left in the squashed release commit.
- **Equivalence Analysis**:
  - `datasets.py` defines the identical class interfaces (`FashionIQ`, `Shoes`, `CIRR`, `Fashion200k`) with matching constructor signatures and test attribute names.
  - However, `datasets.py:74-76` contains a side-effect bug: `FashionIQ.__init__` calls `train_init_process()` if cache is missing, requiring training annotations (`cap.{cat}.train.json`) even during validation. It is UNVERIFIED whether `datasets1.py` had this defect or other preprocessing differences. Declaring `datasets.py` equivalent without qualification is an unproven assumption ([INFERENCE]).

### 2. `train.py:17` `import model as model` (MISSING MODULE)
- **Status**: `CONFIRMED_BUG` (upstream packaging defect).
- **Location**: `workbench/third_party/ENCODER/train.py:17`.
- **Precondition**: Attempting to execute `train.py`.
- **Actual**: `ModuleNotFoundError: No module named 'model'`.
- **Expected**: Module `model.py` present or imported from `model_try2.py`.
- **Evidence**: Pinned repository contains `model_try2.py` only.

### 3. OpenCLIP Backbone Binary Missing and Unregistered
- **Status**: `EXTERNAL_BLOCKER`.
- **Location**: `evaluate_model.py:40`, `model_try2.py:135`.
- **Precondition**: Running evaluation or instantiating `Backbone`.
- **Actual**: `FileNotFoundError: ./open_clip_pytorch_model.bin`.
- **Expected**: Backbone weights present or registered in `workbench/registry/auxiliary_assets.yaml`.
- **Evidence**: File does not exist on disk or in repository. `auxiliary_assets.yaml` has 0 rows for `encoder`.

### 4. Missing Output Parser in Workbench
- **Status**: `CONFIRMED_BUG` (workbench defect).
- **Location**: `workbench/backend/metrics_extraction.py:402-422`.
- **Precondition**: Running evaluation and extracting metrics.
- **Actual**: Returns `AGGREGATE_PARSER_UNAVAILABLE`. Paper gate returns `FAILED_OUTPUT_VALIDATION`.
- **Expected**: Parses per-category stdout `[('{cat}_r1', ...), ('{cat}_r10', ...), ('{cat}_r50', ...)]` and computes macro averages.

### 5. `pc_readiness.classify` Misclassification of Missing Checkpoints
- **Status**: `CONFIRMED_BUG` (workbench defect).
- **Location**: `workbench/scripts/pc_readiness.py:47-67`.
- **Precondition**: Running `pc_readiness` on a model whose checkpoint path contains `fashioniq` (e.g. `.../encoder/fashioniq.pt`).
- **Actual**: `classify` returns `BLOCKED_DATASET` instead of `BLOCKED_EXTERNAL_ASSET` because `"FashionIQ"` needle precedes `"checkpoint missing"`.

### 6. Unconditional CUDA Requirement
- **Status**: `BY_DESIGN` (upstream architectural requirement).
- **Location**: `evaluate_model.py:64` (`Encoder_model.cuda()`), `test.py:32,48`.
- **Impact**: Execution on CPU-only hosts is completely impossible.

### 7. Evaluation Stochasticity from Unseeded Training Transform
- **Status**: `BY_DESIGN` (upstream code property) / `UNVERIFIED_RISK` (reproducibility risk).
- **Location**: `evaluate_model.py:43`, `datasets.py:177`.
- **Impact**: Query image uses `stage=0` (`preprocess_train`, containing RandomResizedCrop) while gallery uses `stage=1` (`preprocess_val`). `evaluate_model.py` does not seed PyTorch or NumPy RNGs, causing run-to-run metric variance.

### 8. Path Concatenation Bug in Dataset Loader
- **Status**: `CONFIRMED_BUG` (upstream code defect).
- **Location**: `datasets.py:68-70`.
- **Impact**: String concatenation `self.path + 'resized_image'` requires a trailing slash on `--fashioniq_path`. Handled correctly by workbench adapter.

### 9. Missing Preparation Contract Registration
- **Status**: `CONFIRMED_BUG` (workbench registry defect).
- **Location**: `workbench/registry/models.yaml:434`, `workbench/registry/preparation_contracts.yaml`.
- **Impact**: `preparation_contract_for_model` returns `None`.

---

## I. EVIDENCE

### 1. Pinned Source Commit Verification
```bash
$ git -C workbench/third_party/ENCODER rev-parse HEAD
29a2a31d6a56f677bf450c3be7cdaef423fb7018
$ git -C workbench/third_party/ENCODER log -n 1 --oneline
29a2a31 Update README.md
```

### 2. Upstream Filesystem Check for `datasets1.py`
```bash
$ git -C workbench/third_party/ENCODER ls-files datasets1.py
(empty output)
$ ls workbench/third_party/ENCODER/datasets1.py
ls: cannot access 'workbench/third_party/ENCODER/datasets1.py': No such file or directory
```

### 3. Preflight & Readiness Execution
```bash
$ python3 -c "from workbench.backend.operator_config import resolve_config; from workbench.backend.registry import model_by_id; from workbench.backend.preflight import preflight_blockers; c = resolve_config(); print(preflight_blockers(model_by_id('encoder'), c))"
['ENCODER evaluator import missing: /home/heheboiz/data/cir/workbench/third_party/ENCODER/datasets1.py', 'ENCODER asset missing: /home/heheboiz/data/cir/workbench/third_party/ENCODER/open_clip_pytorch_model.bin']

$ python3 -c "from workbench.backend.operator_config import resolve_config; from workbench.backend.registry import model_by_id; from workbench.backend.runtime import runtime_plan_blockers; c = resolve_config(); m = model_by_id('encoder'); print(runtime_plan_blockers(m, m['checkpoint_variants'][0], c))"
['ENCODER evaluator import missing: /home/heheboiz/data/cir/workbench/third_party/ENCODER/datasets1.py', 'ENCODER asset missing: /home/heheboiz/data/cir/workbench/third_party/ENCODER/open_clip_pytorch_model.bin', 'checkpoint missing: /home/heheboiz/data/cir/workbench/artifacts/checkpoints/encoder/fashioniq.pt', 'checkpoint mapping unresolved']
```

### 4. Metrics Extraction Absence
```bash
$ python3 -c "from workbench.backend.metrics_extraction import extract_aggregate_metrics; r = extract_aggregate_metrics('encoder', \"[('dress_r1', 24.1), ('dress_r10', 51.5)]\", '', 0); print(r.extraction_status, r.parity_status)"
AGGREGATE_PARSER_UNAVAILABLE PARSER_UNAVAILABLE
```

---

## J. VERDICT

- **Source Audit**: `BLOCKED_SOURCE` (Pinned official evaluator `evaluate_model.py:7` fails on import `import datasets1` which does not exist in the source checkout or upstream repository commit).
- **Protocol Audit**: `PROTOCOL_VERIFIED` (Source implements Protocol C `fashioniq_val_split`: pair-union gallery, reference masked to `-10e10`).
- **Command Audit**: `COMMAND_BLOCKED` (Adapter command is correctly constructed against source argparse, but cannot execute due to missing `datasets1.py` and missing `open_clip_pytorch_model.bin`).
- **Checkpoint Audit**: `CHECKPOINT_BLOCKED` (Checkpoint variant `fashioniq.pt` is unverified, lacks direct URL/SHA256, is not staged locally, and requires unregistered OpenCLIP backbone binary).
- **Query Export Capability**: `QUERY_EXPORT_IMPLEMENTABLE` (Observation point confirmed at `test.py:71-72` after similarity ranking; canonical query and gallery ID mapping fully proven; currently blocked by evaluator runtime prerequisites).
- **Validation Readiness**: `BLOCKED_EXTERNAL_DEPENDENCY` (Evaluation blocked by missing upstream `datasets1.py`, missing OpenCLIP LAION weights, unverified Google Drive checkpoint, missing dataset resized images/corrections, and missing workbench metric parser).
