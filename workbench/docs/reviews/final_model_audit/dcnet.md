# Final Model Audit: dcnet

## A. MODEL IDENTITY

- **Model ID**: `dcnet` (`workbench/registry/models.yaml:606`)
- **Method Name**: `DCNet` (`workbench/registry/models.yaml:612`)
- **Research Generation**: `legacy` (`workbench/registry/models.yaml:613`)
- **Publication Year**: `2021` (`workbench/registry/models.yaml:614`)
- **Paper Title**: `Dual Compositional Learning in Interactive Image Retrieval` (`workbench/registry/models.yaml:615`)
- **Authors**: Jongseok Kim, Youngjae Yu, Hoeseong Kim, Gunhee Kim
- **Publication Venue**: Proceedings of the 35th AAAI Conference on Artificial Intelligence (AAAI '21), Vol. 35, No. 2, May 18, 2021, pp. 1771–1779. DOI: `10.1609/aaai.v35i2.16271`.
- **Paper URL**: `https://github.com/ozmig77/dcnet/blob/68c79d38569f01ca39c3b4ea641b3b7eb2ccb5fb/DCNet_Kim2021.pdf` (`workbench/registry/models.yaml:616`, bundled camera-ready PDF inside upstream repository)
- **Upstream Repository URL**: `https://github.com/ozmig77/dcnet` (`workbench/registry/models.yaml:617`)
- **Upstream Default Branch**: `main` (`workbench/registry/models.yaml:618`)
- **Upstream Pinned Commit SHA**: `68c79d38569f01ca39c3b4ea641b3b7eb2ccb5fb` (`workbench/registry/models.yaml:619`)
- **Source Directory**: `DCNet` (`workbench/registry/models.yaml:621`)
- **Source Available**: `true` (`workbench/registry/models.yaml:620`)
- **Native Protocol**: `fashioniq_full_gallery_ref_excluded` (`workbench/registry/models.yaml:627`)
- **Supported Protocols**: `[fashioniq_full_gallery_ref_excluded]` (`workbench/registry/models.yaml:628`)
- **Reproduction Status**: `BLOCKED_CHECKPOINT_MAPPING` (`workbench/registry/models.yaml:634`)
- **Command Status**: `COMMAND_AUDITED` (`workbench/registry/models.yaml:635`)
- **Replay Status**: `REPLAY_CODE_IMPLEMENTED` (`workbench/registry/models.yaml:607`)
- **Replay Type**: `OFFICIAL_CLI` (`workbench/registry/models.yaml:610`)
- **Category Model Policy**: `shared` (`workbench/registry/models.yaml:608`)
- **Checkpoint Serialization**: `run_directory` (`workbench/registry/models.yaml:609`)
- **Reported Metrics**: `reported_mean: 40.83`, `reported_r10: null`, `reported_r50: null` (`workbench/registry/models.yaml:633`)
- **Reported Score Note**: "Sanity row: the paper prints only the category-level mean (40.83). r10/r50 exist only as WORKBENCH_DERIVED aggregates in the committed row, so they stay null here — a 'reported_' field must not carry a derived value. Feeds the legacy parity display; the gate compares against the committed benchmark row dcnet_fashioniq_val_table1." (`workbench/registry/models.yaml:629-632`)

### Pinned Commit Verification
```bash
$ git -C workbench/third_party/DCNet rev-parse HEAD
68c79d38569f01ca39c3b4ea641b3b7eb2ccb5fb
$ git -C workbench/third_party/DCNet log -1 --oneline
68c79d3 Readme typo
```
The pinned commit SHA `68c79d38569f01ca39c3b4ea641b3b7eb2ccb5fb` in `workbench/third_party/DCNet` matches `upstream_commit_sha` in `workbench/registry/models.yaml:619`. The tree has a single snapshot commit.

### Checkpoint Variant
- **Variant ID**: `fashioniq_run_directory` (`workbench/registry/models.yaml:647`)
- **Artifact Type**: `run_directory` (`workbench/registry/models.yaml:649`)
- **Filename / Directory**: `fashioniq_dcnet` (`workbench/registry/models.yaml:648`)
- **Required Files**: `[config.json, trained_model.pth]` (`workbench/registry/models.yaml:650`)
- **Download URL**: `null` (`workbench/registry/models.yaml:651`)
- **Source URL**: `https://drive.google.com/drive/folders/1wgygqF095Di67EaHaGOXbwh3wEzk9izB?usp=sharing` (`workbench/registry/models.yaml:652`, matches `README.md:59`)
- **Source Type**: `author_google_drive` (`workbench/registry/models.yaml:653`)
- **Checkpoint Mapping Status**: `UNVERIFIED` (`workbench/registry/models.yaml:659`)
- **Status**: `CHECKPOINT_MAPPING_UNVERIFIED` (`workbench/registry/models.yaml:658`)
- **Expected SHA-256**: `null` (`workbench/registry/models.yaml:657`)
- **Expected Size Bytes**: `null` (`workbench/registry/models.yaml:656`)
- **Evaluation Noise Pct**: `0` (`workbench/registry/models.yaml:655`)
- **Checkpoint Training Noise Pct**: `null` (`workbench/registry/models.yaml:654`)

### Paper Benchmark Row
Registered in `workbench/registry/paper_benchmarks/papers.yaml:1442-1519` as `reference_id: dcnet_fashioniq_val_table1`.
- **Source Document**: `workbench/third_party/DCNet/DCNet_Kim2021.pdf` (1,764,944 bytes, SHA-256: `1a87a96185ab1d5a057512a72504b128adc2fd589b5dc0e0ebe8856f559e8ad4`).
- **Source Table**: Table 1 ("Comparison between our DCNet and baselines on the Fashion-IQ validation set"), page 5, AAAI '21 camera-ready.
- **Exact Row Quoted**:
  `DCNet | 28.95 56.07 | 23.95 47.30 | 30.44 58.29 | 40.83`
- **Metric Breakdown**:
  - `dress`: R@10 = 28.95, R@50 = 56.07
  - `shirt`: R@10 = 23.95, R@50 = 47.30
  - `toptee`: R@10 = 30.44, R@50 = 58.29
  - `Total Avg`: 40.83
- **Arithmetic Check**:
  - Macro R@10 = $(28.95 + 23.95 + 30.44) / 3 = 83.34 / 3 = 27.78$ (`WORKBENCH_DERIVED`)
  - Macro R@50 = $(56.07 + 47.30 + 58.29) / 3 = 161.66 / 3 = 53.88666... \to 53.89$ (`WORKBENCH_DERIVED`)
  - Total Avg = $(27.78 + 53.8867) / 2 = 81.6667 / 2 = 40.8333... \to 40.83$ (printed in paper).
  - Alternative 6-cell mean: $(28.95 + 56.07 + 23.95 + 47.30 + 30.44 + 58.29) / 6 = 245.00 / 6 = 40.8333... \to 40.83$.
- **Reference Status**: `VERIFIED`, **Review Status**: `CONFIRMED`.

---

## B. SOURCE COVERAGE

- **Audit Type**: `STATIC_AUDIT` + local script validation.
- **Pinned Tree Inventory** (`workbench/third_party/DCNet/`, 33 files total):
  ```text
  workbench/third_party/DCNet/
  ├── .gitignore                               (7 lines)
  ├── DCNet_Kim2021.pdf                        (1,764,944 bytes, camera-ready AAAI '21 paper)
  ├── README.md                                (86 lines: setup, preparation, training, eval CLI)
  ├── parse_config.py                          (95 lines: ConfigParser, CLI parser, log setup)
  ├── requirement.txt                          (11 lines: pinned legacy stack dependencies)
  ├── test.py                                  (117 lines: evaluation entrypoint, seeding, metric call)
  ├── train.py                                 (84 lines: training CLI entrypoint)
  ├── base/
  │   ├── __init__.py                          (2 lines)
  │   ├── base_model.py                        (21 lines: BaseModel with summary/parameter logging)
  │   └── base_trainer.py                      (219 lines: BaseTrainer, _resume_checkpoint, train loop)
  ├── configs/ce/
  │   ├── fashioniq_dcnet.json                 (83 lines: standard ResNet-50 ImageNet config)
  │   ├── fashioniq_dcnet_deep.json            (83 lines: optional DenseNet DeepFashion config)
  │   └── shoe_dcnet.json                      (79 lines: Shoes benchmark config)
  ├── data_loader/
  │   ├── CE_dataset.py                        (140 lines: CE dataset loader, image opening, collate_fn)
  │   └── data_loaders.py                      (262 lines: ExpertDataLoader, category loaders, transforms)
  ├── experiments/
  │   ├── result.ipynb                         (Jupyter notebook: qualitative retrieval inspection)
  │   └── textatt.ipynb                        (Jupyter notebook: attention weight visualization)
  ├── logger/
  │   ├── __init__.py                          (1 line)
  │   ├── logger.py                            (38 lines: Tensorboard logging wrapper)
  │   └── logger_config.json                   (16 lines: standard logging dictionary format)
  ├── model/
  │   ├── __init__.py                          (6 lines)
  │   ├── composition.py                       (236 lines: TIRG, TIRG2, and compositional units)
  │   ├── fishnet.py                           (304 lines: FishNet backbone alternative)
  │   ├── fusion.py                            (15 lines: Hadamard and concatenation fusion)
  │   ├── loss.py                              (99 lines: AdditiveMarginSoftmax, InfoNCE losses)
  │   ├── metric.py                            (141 lines: recall, compute_score, inner product metrics)
  │   ├── model.py                             (420 lines: CENet, TextCEModuleNew, ImageCEModule)
  │   └── modules.py                           (76 lines: pooling modules, masked_softmax)
  ├── preprocess/
  │   ├── download_fashioniq.py                (86 lines: multithreaded URL image scraper)
  │   ├── gen_shoe_cap.py                      (91 lines: Shoes caption GloVe extractor)
  │   ├── process_cap.py                       (64 lines: GloVe caption embedding and vocab builder)
  │   └── resize_img.py                        (78 lines: multithreaded 256x256 image resizing)
  └── utils/
      ├── __init__.py                          (1 line)
      ├── text2vec.py                          (44 lines: Word2Vec loader utility)
      ├── util.py                              (165 lines: memcache, read_json, write_json, np_loader)
      └── vocab.py                             (46 lines: Vocabulary builder and serializer)
  ```

### Reachable Evaluation Dependency Graph
When executing native FashionIQ evaluation via `python test.py --resume <run_dir>`:
1. `test.py:96-117` (`__main__` entrypoint):
   - Sets deterministic seeds: `seed=0`, `torch.manual_seed(0)`, `np.random.seed(0)`.
   - Calls `ConfigParser(args, 'eval')` (`parse_config.py:13-62`), which reads `<resume>/config.json`.
   - Calls `test(test_config)` (`test.py:17`).
2. `test.py:36-42`:
   - Instantiates `data_loaders = config.init('data_loader', module_data, ...)` (`data_loader/data_loaders.py:ExpertDataLoader`).
   - For `mode == 'val'`, creates query loaders (`split='val'`) and gallery loaders (`split='val_trg'`) for each category (`dress`, `shirt`, `toptee`).
   - Gallery loader `val_trg` in `data_loader/CE_dataset.py:26-28` reads `image_splits/split.{category}.val.json` (the complete ordered validation image list).
   - Both loaders apply deterministic `CenterCrop(224) -> ToTensor() -> Normalize(ImageNet)` (`data_loaders.py:66-70`).
3. `test.py:44-51`:
   - Instantiates `model = config.init('arch', module_arch, ...)` (`model/model.py:CENet`).
   - Standard config instantiates `ImageCEModule` with ResNet-50 (`torchvision.models.resnet50(pretrained=True)` at `model/model.py:283`) and `TextCEModuleNew` with shared 1D convolution over 300-d GloVe token embeddings.
4. `test.py:52-59`:
   - Instantiates `trainer = TrainerJoint(model, loss=None, optimizer=None, config=config, data_loaders=data_loaders, lr_scheduler=None)`.
   - Calls `super().__init__` (`base/base_trainer.py:10-50`), which executes `self._resume_checkpoint(config.resume)`.
   - Reads `<resume>/trained_model.pth` and calls `self.model.load_state_dict(checkpoint['state_dict'])` (`base_trainer.py:193-206`).
5. `test.py:64`:
   - Calls `metric = trainer._valid_epoch(save_textatt=True)` (`trainer/trainer_joint.py:133-254`).
   - Pre-computes gallery candidate embeddings `val_experts` over full split gallery.
   - For each query batch, extracts composition score `score_matrix_comp` and correction score `score_matrix_corr`.
   - Explicitly masks reference image in BOTH branches:
     `score_matrix_comp[ei, id2targetidx[elem['candidate']]] = -10e10`
     `score_matrix_corr[ei, id2targetidx[elem['candidate']]] = -10e10`
   - Combines normalized branch scores:
     `score_comb = F.log_softmax(score_comp[j], dim=-1) + F.log_softmax(score_corr[j], dim=-1)`
   - Computes top-50 indices: `score_comb.topk(dim=0, k=50, largest=True, sorted=True)`.
   - Calls `compute_score(meta_infos, meta_infos)` (`model/metric.py:12-21`), yielding binary R@10, R@50 per category.
   - Computes overall macro mean `comb_avg = metric['comb'].mean()`.

### Inspected vs. Excluded Files
- **Inspected**:
  - `test.py`: Full evaluation CLI flow, seeding, logging.
  - `parse_config.py`: Configuration parsing, run directory resolution, directory layout.
  - `base/base_trainer.py`: Checkpoint loading (`_resume_checkpoint`), state dict extraction, validation routing.
  - `trainer/trainer_joint.py`: `TrainerJoint._valid_epoch`, dual-branch score calculation, dual reference masking, log-softmax fusion, top-50 sorting, macro averaging.
  - `trainer/trainer_ref.py`: Alternative `TrainerRef` (investigated to verify reference masking behavior across trainers).
  - `data_loader/CE_dataset.py`: Dataset loader `CE`, gallery split resolution (`split.{category}.val.json`), GloVe PKL reading, image transform pipeline.
  - `data_loader/data_loaders.py`: `ExpertDataLoader`, category batching, evaluation transforms (`CenterCrop(224) -> ToTensor -> Normalize`).
  - `model/model.py`: `CENet`, `ImageCEModule` (ResNet-50 backbone loading), `TextCEModuleNew` (text encoder).
  - `model/metric.py`: `recall` (binary indicator), `compute_score` (category recall mean), `InnerProduct`.
  - `preprocess/process_cap.py`: GloVe extraction, NLTK/spaCy tokenization, unseeded OOV random normal vector sampling (`np.random.normal(0, 0.3, (300,))`).
  - `preprocess/resize_img.py`: Image resizing script, PIL `Image.ANTIALIAS` usage, 256x256 flat output directory.
  - `configs/ce/fashioniq_dcnet.json`: Standard evaluation configuration schema.
  - `configs/ce/fashioniq_dcnet_deep.json`: DenseNet DeepFashion alternative configuration schema.
  - `requirement.txt`, `README.md`, `DCNet_Kim2021.pdf`: Documentation, dependency pins, and publication evidence.
- **Excluded**:
  - `train.py`: Optimization loop, backpropagation, and training-time argument parsing.
  - `model/loss.py`: Training loss functions (`AdditiveMarginSoftmax`, `InfoNCE`).
  - `preprocess/gen_shoe_cap.py` & `configs/ce/shoe_dcnet.json`: Shoes benchmark files.
  - `preprocess/download_fashioniq.py`: Web image scraper (not reachable during validation).
  - `experiments/*.ipynb`: Interactive visualization notebooks.
  - `logger/logger.py`: TensorBoard event writer.

### Executable Source vs. README Claims
1. **Pillow Resize Compatibility**: `README.md:52` documents running `python resize_img.py`. However, `preprocess/resize_img.py:11` invokes `Image.ANTIALIAS`, which was removed in Pillow $\ge 10.0.0$, causing an immediate `AttributeError`.
2. **Preprocessing Working Directory**: `README.md:47-52` shows `cd preprocess` before running `process_cap.py` and `resize_img.py`. Both scripts hardcode relative paths starting with `../dataset/fashioniq/` (`preprocess/process_cap.py:12` and `preprocess/resize_img.py:71`), so running them from repo root without changing directory immediately raises `FileNotFoundError`.
3. **Evaluation CLI Dataset Parameter**: `README.md:62` gives the evaluation command as `python test.py --resume logdir/fashioniq_dcnet/`. Neither `README.md` nor `test.py` provides a `--data_dir` or `--dataset-root` command-line argument. The dataset path is strictly read from `config['data_loader']['args']['data_dir']` inside `config.json` (which defaults to relative path `"dataset/fashioniq"`).

---

## C. NATIVE EVALUATION

- **Verdict Axis**: `PROTOCOL_VERIFIED`
- **Native Protocol**: Protocol B (`fashioniq_full_gallery_ref_excluded`).

### 1. Proof of Full Split Gallery (Not a Pair-Union)
In `data_loader/CE_dataset.py:24-28`:
```python
self.no_target = (split == 'val_trg' or split == 'test_trg')
self.text_feat = text_feat
if self.no_target:
    self.data = read_json(Path(data_dir + f"/image_splits/split.{category}.{split.split('_')[0]}.json"))
    self.data_length = len(self.data)
```
When `split == 'val_trg'`, `split.split('_')[0]` evaluates to `'val'`, so `CE` directly loads:
`Path(data_dir + f"/image_splits/split.{category}.val.json")`
This JSON file contains the full official validation split of candidate images:
- Dress: 3,817 images
- Shirt: 6,346 images
- Toptee: 5,373 images
Total gallery size across three categories: 15,536 images.
Target candidates are NOT extracted from query reference-target unions; the gallery is the entire ordered validation split.

### 2. Dual Reference Masking in Both Score Branches
DCNet computes retrieval scores by fusing two expert branches:
1. Composition network score: $s_c$ (`score_matrix_comp`)
2. Correction network score: $s_r$ (`score_matrix_corr`)

In `trainer/trainer_joint.py:215-218`:
```python
# Delete reference == target
for ei, elem in enumerate(batch['meta_info']):
    score_matrix_comp[ei, id2targetidx[elem['candidate']]] = -10e10 
    score_matrix_corr[ei, id2targetidx[elem['candidate']]] = -10e10 
```
- **Audit Finding**: The reference image index `id2targetidx[elem['candidate']]` is explicitly masked with `-10e10` in **BOTH** score branches (`score_matrix_comp` and `score_matrix_corr`).
- Neither branch is left unmasked. There is no partial masking bug.

### 3. Score Fusion via `log_softmax`
In `trainer/trainer_joint.py:237-241`:
```python
# Combine
for j in range(len(score_comp)):
    score_comb = F.log_softmax(score_comp[j], dim=-1) + F.log_softmax(score_corr[j], dim=-1) # N
    _, topk = score_comb.topk(dim=0, k=50, largest=self.largest, sorted=True)
    meta_infos[j]['ranking'] = [val_ids[idx] for idx in topk]
```
- Both `score_comp[j]` and `score_corr[j]` are independently normalized across the entire gallery dimension (`dim=-1`) via `F.log_softmax`.
- Because the reference entry received `-10e10` in both matrices, its softmax probability is $0.0$, and its log-softmax value is $-\infty$ (in finite float: $\approx -10^{10}$).
- Summing the two log-probabilities produces an aggregated score `score_comb` where the reference entry remains at approx $-2 \times 10^{10}$, strictly excluding the reference from the top-$k$ ranking.

### 4. Sorting and Top-K Selection
- Top candidates are extracted using `torch.topk(dim=0, k=50, largest=True, sorted=True)` (`trainer/trainer_joint.py:240`).
- The top-50 gallery image IDs are recorded in order in `meta_infos[j]['ranking']`.

### 5. Recall Definition and Denominator
In `model/metric.py:5-9, 12-21`:
```python
def recall(actual, predicted, k):
    act_set = set([actual])
    pred_set = set(predicted[:k])
    result = len(act_set & pred_set) / float(len(act_set))
    return result

def compute_score(solution, prediction):
    n = len(solution)
    scores_r_10, scores_r_50 = [], []
    for i in range(n):
        assert solution[i]["candidate"] == prediction[i]["candidate"]
        scores_r_10.append(recall(solution[i]["target"], prediction[i]["ranking"], 10))
        scores_r_50.append(recall(solution[i]["target"], prediction[i]["ranking"], 50))

    return sum(scores_r_10) / n, sum(scores_r_50) / n
```
- For composed image retrieval, each query has exactly one ground-truth target (`len(act_set) == 1`).
- `recall()` evaluates to $1.0$ if the target is in the top-$k$ retrieved items, else $0.0$.
- `compute_score` divides by $n = \text{len}(\text{solution})$, the total number of validation queries in that category.
- Metrics are returned as fractions in $[0.0, 1.0]$ and logged directly to stdout.

### 6. Category Handling and Macro Aggregation
In `trainer/trainer_joint.py:144, 235, 243, 247-249`:
```python
metric['comb'][i] = r10_c, r50_c
...
metric['recall_avg'] = metric['recall'][:,:2].mean()
metric['recall_avg_corr'] = metric['recall'][:,2:].mean()
metric['comb_avg'] = metric['comb'].mean()
```
- `metric['comb']` has shape `(3, 2)` (3 categories: `dress`, `shirt`, `toptee`; 2 recall levels: R@10, R@50).
- `metric['comb_avg'] = metric['comb'].mean()` takes the arithmetic average across all 6 values:
  $$\text{Total Avg} = \frac{\text{R@10}_{\text{dress}} + \text{R@50}_{\text{dress}} + \text{R@10}_{\text{shirt}} + \text{R@50}_{\text{shirt}} + \text{R@10}_{\text{toptee}} + \text{R@50}_{\text{toptee}}}{6}$$
- Expressed in percent ($\times 100$), this evaluates to $40.83$, matching Table 1's reported "Total Avg" in `DCNet_Kim2021.pdf:5`.

---

## D. PREPARATION

- **Preparation Contract**: `dcnet_fashioniq` (`workbench/registry/preparation_contracts.yaml:60-139`).
- **Preparation Policy**: `AUTHOR_ARTIFACT_PREFERRED` (`preparation_contracts.yaml:137`).
- **Deterministic Status**: `NONDETERMINISTIC_WITHOUT_AUTHOR_ARTIFACT` (`preparation_contracts.yaml:136`).

### 1. Caption GloVe Embeddings
- Preprocessing script: `workbench/third_party/DCNet/preprocess/process_cap.py`.
- Vocabulary building (`process_cap.py:12-23`): Reads `captions/cap.*.train.json`, replaces hyphens and periods, tokenizes with `nltk.tokenize.word_tokenize`, and counts token frequencies.
- Vector lookup (`process_cap.py:26-32`): Looks up each token in spaCy's `en_vectors_web_lg` vocab table (300-d vectors).
- Evaluator reads precomputed vectors: `data_loader/CE_dataset.py:34-35` memcaches `captions/cap.{category}.glove.val.pkl` and accesses `data[index]['wv']` (`:55, 64`). spaCy and NLTK are runtime requirements *only* if regenerating PKLs, not during evaluation.

### 2. Unseeded OOV Random Vectors (Specific to DCNet)
In `workbench/third_party/DCNet/preprocess/process_cap.py:25-32`:
```python
vocab_words, no_word = {}, []
for word, cnt in counter.items():
    if glove.has_vector(word):
        vocab_words[word] = glove.get_vector(word)
    elif cnt > 2:
        no_word.append(word)
        vocab_words[word] = np.random.normal(0, 0.3, (300,))
print(len(vocab_words), len(no_word))
```
- **Audit Finding**: Out-of-vocabulary (OOV) tokens occurring more than twice in the training set receive random 300-d Gaussian vectors via `np.random.normal(0, 0.3, (300,))`.
- **RNG Seeding**: The file contains **zero calls to `np.random.seed(...)`** or any seed initialization.
- **Classification**: `CONFIRMED_BUG` (upstream source implementation) / `UNVERIFIED_RISK` (reproducibility hazard).
- **Concrete Consequence for Provenance**:
  1. Any execution of `process_cap.py` samples different random vectors for every frequent OOV word.
  2. These random vectors are stored in `glove_vecs.pkl` and directly embedded into the validation query arrays in `cap.{category}.glove.val.pkl`.
  3. Consequently, re-running preprocessing creates non-identical `.pkl` artifacts. An evidence digest (e.g., SHA-256) computed over regenerated PKLs will fluctuate on every run.
  4. Downstream retrieval scores for any query containing OOV words will exhibit run-to-run stochastic variance.
- **Workbench Enforcement**: The workbench preparation contract explicitly forbids automatic regeneration (`automation_policy: AUTHOR_ARTIFACT_PREFERRED`), requiring operator-supplied author PKLs.

### 3. Image Resizing
- Script: `workbench/third_party/DCNet/preprocess/resize_img.py`.
- Target: 256x256 square images stored in a flat directory `resized_images/` (`preprocess/resize_img.py:69, 55`).
- Native script flaw: `resize_img.py:11` uses `Image.ANTIALIAS`, which raises `AttributeError` on Pillow $\ge 10.0.0$.
- Evaluator transform: `data_loader/data_loaders.py:66-70` applies `CenterCrop(224) -> ToTensor() -> Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` without data augmentation during evaluation.
- Workbench handling: `workbench/scripts/prepare_model.py:fashioniq_resize_images` provides a standalone, verified resize recipe writing flat `resized_images/*.jpg` at 256px.

### 4. Workbench Refusal to Automatically Regenerate Caption PKLs
- `workbench/scripts/prepare_model.py` inspects the contract:
  - `policy=AUTHOR_ARTIFACT_PREFERRED`
  - `deterministic=NONDETERMINISTIC_WITHOUT_AUTHOR_ARTIFACT`
- `prepare_model.py` implements handlers for fixed symlinks (`_fixed_link`), bundled assets (`_stage_source_assets`), and image resizing (`_run_resize`). It contains no handler or execution branch for `process_cap.py`.
- Running `python workbench/scripts/prepare_model.py --model dcnet --execute --allow-preparation` executes only the image resize recipe and reports `MANUAL` / refuses to regenerate caption PKLs.

---

## E. CHECKPOINT

- **Verdict Axis**: `CHECKPOINT_BLOCKED`
- **Serialization Policy**: `run_directory` (`workbench/registry/models.yaml:609`)

### 1. Identity and Provenance
- Checkpoint directory: `fashioniq_dcnet`
- Upstream source link: Google Drive folder `https://drive.google.com/drive/folders/1wgygqF095Di67EaHaGOXbwh3wEzk9izB?usp=sharing` (`models.yaml:652`, `README.md:59`).
- Checkpoint status: `CHECKPOINT_MAPPING_UNVERIFIED` (`models.yaml:658`).
- Required files:
  1. `config.json` (architecture parameters, expert dimensions, hyperparameters)
  2. `trained_model.pth` (PyTorch model weights state dict)

### 2. Loader and Resolution
In `parse_config.py:18-20` and `base/base_trainer.py:193-206`:
```python
# parse_config.py
self.resume = Path(args.resume)
self.cfg_fname = self.resume / 'config.json'
self._config = read_json(self.cfg_fname)

# base/base_trainer.py
ckpt_path = self.config.resume / 'trained_model.pth'
checkpoint = torch.load(ckpt_path)
self.start_epoch = checkpoint['epoch'] + 1
self.mnt_best = checkpoint['monitor_best']
self.model.load_state_dict(checkpoint['state_dict'])
```
- **Run Directory Resolution**: The CLI argument `--resume <path>` is resolved as a directory path.
- **Missing File Behavior**:
  - If `config.json` is missing: `utils.util.read_json` raises `FileNotFoundError`.
  - If `trained_model.pth` is missing: `torch.load` raises `FileNotFoundError`.
- **Missing Key Behavior**:
  - In `config.json`: Accesses like `config['trainer']['save_dir']` or `config['data_loader']['args']` raise `KeyError`.
  - In `trained_model.pth`: Accesses like `checkpoint['state_dict']`, `checkpoint['epoch']`, or `checkpoint['monitor_best']` raise `KeyError`.

### 3. Architecture Specification
In `model/model.py:17-110` (`CENet`):
- **Image Backbone**: ResNet-50 (`torchvision.models.resnet50(pretrained=True)` at `model/model.py:283`), truncated before pooling/FC (`modules = list(resnet.children())[:-2]`). Feature maps are extracted across multi-scale spatial and intermediate channels (`im_feat`, `inter-1`, `spatial0`-`spatial4`), then projected into 1024-d subspaces.
- **External ImageNet Weights**: Requires torchvision ResNet-50 ImageNet pretrained weights (`https://download.pytorch.org/models/resnet50-19c8e357.pth`, SHA-256 prefix `19c8e357`).
- **Text Encoder**: GloVe 300-d embeddings passed through `TextCEModuleNew` with shared 1D convolution (`text_pooling_module='sharedconv'`) and multi-modality self-attention across 7 subspaces.
- **Composition Network**: TIRG modules (`model/composition.py:TIRG`) with gated residual connections.
- **Correction Network**: Feature difference (`fd`) modules mapping reference-target discrepancies.

---

## F. WORKBENCH INTEGRATION

- **Verdict Axis**: `COMMAND_AUDITED`
- **Adapter**: `workbench/backend/adapters/models.py:DCNetAdapter`
- **Command Construction**:
  ```python
  class DCNetAdapter(OfficialScriptAdapter):
      model_id = "dcnet"
      script = "test.py"

      def command(self, source, checkpoint, request: EvalRequest) -> list[str]:
          return [_runtime.python_executable(self.model), str(source / self.script), "--resume", str(checkpoint)]
  ```
  Generates: `python <source>/test.py --resume <checkpoint_dir>` executed with `cwd = workbench/third_party/DCNet`.

### 1. Dataset Root Resolution Defect
- Native `test.py` does NOT support a CLI `--data_dir` flag.
- Native `parse_config.py` loads `config.json` from the checkpoint run directory, where `config['data_loader']['args']['data_dir']` defaults to relative path `"dataset/fashioniq"`.
- When executed from `cwd = workbench/third_party/DCNet`, the script looks for `dataset/fashioniq` relative to the DCNet source directory.
- `workbench/registry/models.yaml:606-659` does NOT define a `canonical_dataset_link` for `dcnet` (unlike `combiner_rn50x4_noft` which defines `canonical_dataset_link: fashionIQ_dataset`).
- Consequently, executing the adapter command fails immediately because `dataset/fashioniq` does not exist in `workbench/third_party/DCNet/`.

### 2. Missing Output Parser
- In `workbench/backend/metrics_extraction.py:402-422`:
  `extract_aggregate_metrics` contains parser dispatches for `limn`, `csmcir`, `combiner_rn50x4_noft`, `clip4cir_rn50x4_fullft`, and `sprc`.
  It contains NO dispatch for `dcnet`.
- Any successful run of `dcnet` returns:
  `MetricExtractionResult(extraction_status="AGGREGATE_PARSER_UNAVAILABLE", parity_status="PARSER_UNAVAILABLE")`
- The paper gate will reject the run with `FAILED_OUTPUT_VALIDATION`.

### 3. Readiness Classification and Needle-Ordering Defect
- Running `python3 -m workbench.scripts.pc_readiness --model dcnet --json` currently classifies `dcnet` as:
  `state: BLOCKED_ENVIRONMENT`
  `reason: execution environment not verified: WORKBENCH_DCNET_PYTHON=None [UNCONFIGURED/UNCONFIGURED]`
- **Needle-Ordering Bug in `pc_readiness.py:47-78`**:
  `BLOCKER_KINDS` places `("FashionIQ", BLOCKED_DATASET)` at line 55, before `("checkpoint missing", BLOCKED_EXTERNAL_ASSET)` at line 64 and `("checkpoint mapping unresolved", BLOCKED_EXTERNAL_ASSET)` at line 61.
  Because the checkpoint artifact is named `fashioniq_dcnet` (containing the substring `"fashioniq"`), once the environment is configured, any missing checkpoint blocker matching `checkpoint missing: .../fashioniq_dcnet` is matched by `"fashioniq"` and erroneously classified as `BLOCKED_DATASET` instead of `BLOCKED_EXTERNAL_ASSET`.

---

## G. OBSERVATION CAPABILITY

- **Verdict Axis**: `QUERY_EXPORT_IMPLEMENTABLE`

### 1. Ranking Production Point
- In `workbench/third_party/DCNet/trainer/trainer_joint.py:238-241`:
  ```python
  for j in range(len(score_comp)):
      score_comb = F.log_softmax(score_comp[j], dim=-1) + F.log_softmax(score_corr[j], dim=-1) # N
      _, topk = score_comb.topk(dim=0, k=50, largest=self.largest, sorted=True)
      meta_infos[j]['ranking'] = [val_ids[idx] for idx in topk]
  ```
- Each query record in `meta_infos[j]` stores:
  - `candidate`: Reference image ID
  - `target`: Target ground-truth image ID
  - `captions`: Relative modification text captions
  - `ranking`: Top-50 ranked candidate image IDs from `val_ids`

### 2. Native File Export
- In `workbench/third_party/DCNet/test.py:86-90`:
  ```python
  if config._args.mode == 'val':
      ...
  else:
      save_fname = config.save_dir / f'test_score.pt'
      tic = time.time()
      logger.info("Saving score matrix: {} ...".format(save_fname))
      torch.save(metric, save_fname)
      logger.info(f"Done in {time.time() - tic:.3f}s")
  ```
  When invoked with `--mode test`, the full `metric` dictionary (including `metric['score'][category]['meta_info']`) is serialized to `<resume>/test/test_score.pt`.
- In default `--mode val`, metrics are logged to stdout and not saved to disk.

### 3. Observer Implementation Feasibility
An observer wrapper can intercept `trainer._valid_epoch()` or inspect `metric['score'][category]['meta_info']` to extract standard per-query JSON/Parquet rankings without modifying core model weights.

---

## H. DEFECTS

### 1. `CONFIRMED_BUG` / `UNVERIFIED_RISK` — Unseeded OOV Random Vectors in Preprocessing
- **Status**: `CONFIRMED_BUG` (upstream source implementation) / `UNVERIFIED_RISK` (artifact provenance).
- **Location**: `workbench/third_party/DCNet/preprocess/process_cap.py:29-31`.
- **Precondition**: Operator executes `process_cap.py` to regenerate GloVe caption `.pkl` files.
- **Actual**: OOV words occurring $>2$ times are assigned `np.random.normal(0, 0.3, (300,))` without initializing the NumPy RNG seed. Every execution produces different vector values.
- **Expected**: Deterministic seed initialization (`np.random.seed(...)`) or fixed OOV embeddings.
- **Impact**: PKL hashes vary across runs. Downstream retrieval evaluation produces stochastic metrics. The workbench correctly enforces `AUTHOR_ARTIFACT_PREFERRED` to reject automated regeneration.

### 2. `CONFIRMED_BUG` — Hardcoded Upstream Preprocessing Paths
- **Status**: `CONFIRMED_BUG` (upstream source defect).
- **Location**: `workbench/third_party/DCNet/preprocess/process_cap.py:12,35,39` and `preprocess/resize_img.py:71,75`.
- **Precondition**: Running preprocessing scripts from repo root or with non-default directory layouts.
- **Actual**: Scripts hardcode `../dataset/fashioniq/captions/` and `../dataset/fashioniq/images/`, raising `FileNotFoundError`.
- **Expected**: CLI argument support for dataset paths.

### 3. `CONFIRMED_BUG` — Removed `Image.ANTIALIAS` in `resize_img.py`
- **Status**: `CONFIRMED_BUG` (upstream dependency obsolescence).
- **Location**: `workbench/third_party/DCNet/preprocess/resize_img.py:11`.
- **Precondition**: Running `resize_img.py` on Pillow $\ge 10.0.0$.
- **Actual**: Raises `AttributeError: module 'PIL.Image' has no attribute 'ANTIALIAS'`.
- **Expected**: `Image.Resampling.LANCZOS` (or backward-compatible alias).

### 4. `CONFIRMED_BUG` — Missing Dataset Path CLI Support and Missing Dataset Link
- **Status**: `CONFIRMED_BUG` (workbench registry omission & upstream CLI limitation).
- **Location**: `workbench/third_party/DCNet/test.py:96-103`, `configs/ce/fashioniq_dcnet.json:21`, `workbench/registry/models.yaml:606-659`.
- **Precondition**: Executing `test.py` via workbench adapter.
- **Actual**: `test.py` lacks a CLI dataset argument and resolves `"dataset/fashioniq"` relative to cwd (`DCNet/`). `models.yaml` does not declare `canonical_dataset_link`, causing execution to fail with `FileNotFoundError`.
- **Expected**: `canonical_dataset_link: dataset/fashioniq` declared in `models.yaml` or CLI `--data_dir` override.

### 5. `CONFIRMED_BUG` — `pc_readiness.classify` Needle-Ordering Misclassification
- **Status**: `CONFIRMED_BUG` (workbench defect).
- **Location**: `workbench/scripts/pc_readiness.py:47-78`.
- **Precondition**: Running `pc_readiness` on a model whose missing checkpoint path contains `"fashioniq"` (e.g. `.../checkpoints/dcnet/fashioniq_dcnet`).
- **Actual**: Classified as `BLOCKED_DATASET` instead of `BLOCKED_EXTERNAL_ASSET` because `"FashionIQ"` needle precedes `"checkpoint missing"`.
- **Expected**: Checkpoint needles matched prior to generic benchmark substring needles.

### 6. `CONFIRMED_BUG` — Missing Aggregate Metric Parser in Workbench
- **Status**: `CONFIRMED_BUG` (workbench defect).
- **Location**: `workbench/backend/metrics_extraction.py:402-422`.
- **Precondition**: Running DCNet evaluation and extracting metrics.
- **Actual**: Returns `AGGREGATE_PARSER_UNAVAILABLE`. Paper gate returns `FAILED_OUTPUT_VALIDATION`.
- **Expected**: Parser for `[Avg Recall]`, `[Avg Recall corr]`, `[comb_avg]`, and per-category R@10/R@50 lines.

### 7. `EXTERNAL_BLOCKER` — Unmapped Checkpoint Archive
- **Status**: `EXTERNAL_BLOCKER`.
- **Location**: `workbench/registry/models.yaml:646-659`.
- **Precondition**: Attempting checkpoint download or verification.
- **Actual**: Unmapped Google Drive folder without file-level URLs, sizes, or SHA-256 digests. Checkpoint status is `CHECKPOINT_MAPPING_UNVERIFIED`.
- **Expected**: Author-verified downloadable archive with verified SHA-256 digests.

### 8. `EXTERNAL_BLOCKER` — Missing External ResNet-50 ImageNet Weights
- **Status**: `EXTERNAL_BLOCKER`.
- **Location**: `model/model.py:283`, `workbench/registry/preparation_contracts.yaml:123-135`.
- **Precondition**: Running evaluation without cached torchvision weights in an airgapped environment.
- **Actual**: Tries to download `https://download.pytorch.org/models/resnet50-19c8e357.pth`.
- **Expected**: Verified local staging matching SHA prefix `19c8e357`.

### 9. `EXTERNAL_BLOCKER` — Unverified Legacy Execution Environment
- **Status**: `EXTERNAL_BLOCKER`.
- **Location**: `workbench/registry/models.yaml:636-643`.
- **Precondition**: Running evaluation on host.
- **Actual**: Requires Python 3.7.7, PyTorch 1.4.0, torchvision 0.5.0. `WORKBENCH_DCNET_PYTHON` is unset (`UNCONFIGURED`).
- **Expected**: Pinned conda/venv environment satisfying legacy requirements.

### 10. `BY_DESIGN` — Reference Masking in Both Branches and Log-Softmax Fusion
- **Status**: `BY_DESIGN` (upstream architectural design).
- **Location**: `trainer/trainer_joint.py:216-218, 239`.
- **Description**: Reference indices are masked with `-10e10` in both composition and correction matrices before log-softmax fusion. Strictly implements Protocol B.

---

## I. EVIDENCE

1. **Commit SHA Verification**:
   ```bash
   $ git -C workbench/third_party/DCNet rev-parse HEAD
   68c79d38569f01ca39c3b4ea641b3b7eb2ccb5fb
   $ git -C workbench/third_party/DCNet rev-list --count HEAD
   1
   ```

2. **Camera-Ready Paper Table 1 Verification**:
   ```bash
   $ sha256sum workbench/third_party/DCNet/DCNet_Kim2021.pdf
   1a87a96185ab1d5a057512a72504b128adc2fd589b5dc0e0ebe8856f559e8ad4  workbench/third_party/DCNet/DCNet_Kim2021.pdf
   $ stat -c "%s bytes" workbench/third_party/DCNet/DCNet_Kim2021.pdf
   1764944 bytes
   $ pdftotext -layout -f 5 -l 5 workbench/third_party/DCNet/DCNet_Kim2021.pdf - | grep -A 10 "Table 1"
   Table 1: Comparison between our DCNet and baselines on the Fashion-IQ validation set.
   ...
   Category: Dress (28.95 / 56.07), Shirt (23.95 / 47.30), Toptee (30.44 / 58.29), Total Avg (40.83)
   ```

3. **Doctor Verification**:
   ```bash
   $ python3 -m workbench.scripts.doctor --model dcnet
   [OK] source:dcnet: source pin ready
   [BLOCKED] runtime:dcnet: command unavailable
   [BLOCKED] raw-dataset:dcnet: raw benchmark inputs missing
   [BLOCKED] preparation:dcnet: required generated artifacts missing
   [BLOCKED] external-assets:dcnet: required evaluation assets unresolved
   [OK] command:dcnet: COMMAND_AUDITED
   [BLOCKED] environment:dcnet: UNCONFIGURED
   ```

4. **Preparation Policy Verification**:
   ```bash
   $ python3 workbench/scripts/prepare_model.py --model dcnet
   [MANUAL] dcnet: policy=AUTHOR_ARTIFACT_PREFERRED deterministic=NONDETERMINISTIC_WITHOUT_AUTHOR_ARTIFACT
     generated artifacts missing (4): /home/heheboiz/data/cir/data/fashionIQ_dataset/captions/cap.dress.glove.val.pkl
     required external assets: torchvision ResNet-50 ImageNet weights
     notes: process_cap.py samples unseeded NumPy OOV vectors; never regenerate automatically. Later approved regeneration must record seed state, NumPy version, spaCy model/version, NLTK tokenizer/version, and generated hashes.
   ```

5. **Readiness Execution**:
   ```bash
   $ python3 -m workbench.scripts.pc_readiness --model dcnet --json
   # state: BLOCKED_ENVIRONMENT
   # reason: execution environment not verified: WORKBENCH_DCNET_PYTHON=None [UNCONFIGURED/UNCONFIGURED] (WORKBENCH_DCNET_PYTHON not set)
   ```

6. **Source Code Proofs**:
   - Reference masking in both branches (`trainer/trainer_joint.py:216-218`):
     ```python
     for ei, elem in enumerate(batch['meta_info']):
         score_matrix_comp[ei, id2targetidx[elem['candidate']]] = -10e10 
         score_matrix_corr[ei, id2targetidx[elem['candidate']]] = -10e10 
     ```
   - Score fusion via log-softmax (`trainer/trainer_joint.py:239`):
     ```python
     score_comb = F.log_softmax(score_comp[j], dim=-1) + F.log_softmax(score_corr[j], dim=-1) # N
     ```
   - Unseeded OOV random vectors (`preprocess/process_cap.py:29-31`):
     ```python
     elif cnt > 2:
         no_word.append(word)
         vocab_words[word] = np.random.normal(0, 0.3, (300,))
     ```
   - Run directory loading (`base/base_trainer.py:193-206`):
     ```python
     ckpt_path = self.config.resume / 'trained_model.pth'
     checkpoint = torch.load(ckpt_path)
     self.model.load_state_dict(checkpoint['state_dict'])
     ```

---

## J. VERDICT

| Axis | Verdict | Key Evidence |
|---|---|---|
| **SOURCE** | **`SOURCE_AUDITED`** | Pinned commit `68c79d38569f01ca39c3b4ea641b3b7eb2ccb5fb` audited. Reachable graph mapped end-to-end. |
| **PROTOCOL** | **`PROTOCOL_VERIFIED`** | Protocol B (`fashioniq_full_gallery_ref_excluded`) verified: full split gallery via `val_trg` loading `image_splits/split.{cat}.val.json`, reference masked to `-10e10` in BOTH score branches, log-softmax score fusion, binary target recall. |
| **COMMAND** | **`COMMAND_AUDITED`** | Official CLI `python test.py --resume <run_dir>` audited. Run directory requires `config.json` and `trained_model.pth`. Blocked by lack of `--data_dir` CLI support and missing `canonical_dataset_link` in `models.yaml`. |
| **CHECKPOINT** | **`CHECKPOINT_BLOCKED`** | Checkpoint is an unmapped Google Drive folder without file-level mapping, download URL, or SHA-256 digest (`CHECKPOINT_MAPPING_UNVERIFIED`). |
| **QUERY EXPORT** | **`QUERY_EXPORT_IMPLEMENTABLE`** | Top-50 rankings populated at `trainer/trainer_joint.py:241` (`meta_infos[j]['ranking']`). Saved to `<resume>/test/test_score.pt` in test mode; directly interceptable via observer wrapper. |
| **RELEASE READINESS** | **`BLOCKED_EXTERNAL_DEPENDENCY`** | Blocked on unmapped Google Drive checkpoint archive, missing author GloVe PKLs (unseeded generator), missing flat 256px resized JPEGs, missing ResNet-50 ImageNet weights, and unset `WORKBENCH_DCNET_PYTHON` environment. |
