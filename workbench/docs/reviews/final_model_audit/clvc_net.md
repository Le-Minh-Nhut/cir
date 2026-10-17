# Final Model Audit: CLVC-Net (clvc_net)

**Audit Date:** 2026-10-17  
**Model ID:** `clvc_net`  
**Protocol:** Protocol C (`fashioniq_val_split`: pair-union gallery, reference excluded)  
**Evaluation Standard:** `STATIC_AUDIT` throughout unless explicitly marked `RUNTIME_VERIFIED`. No GPU inference was executed. All AST/code/CLI checks were executed in read-only mode against HEAD (`5726d5f`) on branch `research/cir-failure-workbench`.

---

## A. MODEL IDENTITY

- **Model ID**: `clvc_net` (`workbench/registry/models.yaml:538`)
- **Method Name**: `CLVC-Net` (`workbench/registry/models.yaml:554`)
- **Paper Title**: *Comprehensive Linguistic-Visual Composition Network for Image Retrieval* (`workbench/registry/models.yaml:557`, `workbench/registry/paper_benchmarks/papers.yaml:536`)
- **Paper Authors**: Haokun Wen, Xuemeng Song, Xin Yang, Yibing Zhan, Liqiang Nie (`workbench/third_party/CLVCNet/README.md:5`)
- **Publication**: Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '21), July 11–15, 2021, pp. 1369–1378.
- **Paper DOI / URL**: `https://doi.org/10.1145/3404835.3462967` (`models.yaml:558`)
- **Upstream Repository URL**: `https://github.com/iLearn-Lab/SIGIR21-CLVC-Net` (`models.yaml:559`)
- **Upstream Default Branch**: `main` (`models.yaml:560`)
- **Pinned Commit SHA**: `bd9b6889489806cb60676597880f91d918b279cb` (`models.yaml:561`)
  - Observed checkout HEAD: `git -C workbench/third_party/CLVCNet rev-parse HEAD` -> `bd9b6889489806cb60676597880f91d918b279cb` (`RUNTIME_VERIFIED`). **EQUALS REGISTRY**.
  - Commit author/date: Haokun Wen, Sun Apr 5 18:54:18 2026 +0800 (`Update README.md`).
  - Upstream history: squashed 1-commit export (`git -C workbench/third_party/CLVCNet rev-list --all --count` = 1).
- **Checkpoint Variant**:
  - `checkpoint_id`: `author_drive_unresolved` (`models.yaml:595`)
  - `filename`: `checkpoint-unresolved.pt` (`models.yaml:596`)
  - `download_url`: `null` (`models.yaml:597`)
  - `source_url`: `https://drive.google.com/file/d/159rBhWyhkLN7sXAi8iyW_ljzFNLJinKa/view?usp=sharing` (`models.yaml:598`)
  - `source_type`: `author_google_drive` (`models.yaml:599`)
  - `checkpoint_training_noise_pct`: `null` (`models.yaml:600`)
  - `evaluation_noise_pct`: `0` (`models.yaml:601`)
  - `expected_sha256`: `null` (`models.yaml:603`)
  - `expected_size_bytes`: `null` (`models.yaml:602`)
  - `status`: `CHECKPOINT_MAPPING_UNVERIFIED` (`models.yaml:604`)
  - `checkpoint_mapping_status`: `UNVERIFIED` (`models.yaml:605`)
- **Paper Benchmark Reference Row**:
  - Reference ID: `clvc_net_sigir21_fiq_table1` in `workbench/registry/paper_benchmarks/papers.yaml:533-591`.
  - Paper table: Table 1, page 5 (PDF page 5), row `CLVC-Net`.
  - Transcribed metrics:
    - Dresses: R@10 = `29.85`, R@50 = `56.47`
    - Shirts: R@10 = `28.75`, R@50 = `54.76`
    - Tops&Tees: R@10 = `33.50`, R@50 = `64.00`
    - FashionIQ Macro Average: R@10 = `30.70`, R@50 = `58.41`
    - Arithmetic Mean: `(30.70 + 58.41) / 2 = 44.56` (labeled `WORKBENCH_DERIVED`).
  - Table 3 repeats 30.70 / 58.41 as ablation baseline.

---

## B. SOURCE COVERAGE

### 1. Pinned Source Inventory (`workbench/third_party/CLVCNet/`)
The pinned repository contains 9 files (1994 lines total):
- `README.md` (96 lines): Documentation, historical environment specs, training commands, test command assertion, and Google Drive link.
- `blocks.py` (118 lines): Neural network layers (`conv3x3`, `conv1x1`, `Inception1d`, `Inception2d`, `GeM`).
- `datasets.py` (462 lines): Dataset loaders for `FashionIQ` (lines 152–258), `Shoes` (lines 1–150), and `Fashion200k` (lines 260–462).
- `model.py` (261 lines): Model architectures (`local_conv`, `global_conv`, `img_backbone`, `LabelSmooth1`, `LabelSmooth2`, `compose_local`, `compose_global`).
- `resnet.py` (348 lines): ResNet backbone variants importing `torchvision.models.utils.load_state_dict_from_url`.
- `test.py` (148 lines): Standalone evaluation function `test(params, local_model, global_model, testset, dataname)` computing Recall@1, 10, 50.
- `text_model.py` (115 lines): Text processing (`SimpleVocab`, `TextLSTMModel`).
- `train.py` (305 lines): Official training entrypoint (`load_dataset`, `create_model_and_optimizer`, `train`, `train_and_evaluate`).
- `utils.py` (141 lines): Utility helpers (`Params`, `RunningAverage`, `save_dict_to_json`, `save_checkpoint`, `load_checkpoint`).

### 2. Reachable Evaluation Dependency Graph
```
test.py::test(params, local_model, global_model, testset, dataname)
├── testset: datasets.py::FashionIQ
│   ├── torch.utils.data.Dataset
│   ├── PIL.Image
│   ├── json, os
│   └── methods: get_img, get_test_queries, get_test_targets
├── local_model: model.py::compose_local
│   ├── text_model.py::TextLSTMModel
│   │   ├── SimpleVocab (strips brackets; fails on missing simplevocab.json fallback)
│   │   ├── torch.nn.Embedding
│   │   └── torch.nn.LSTM (2-layer, 1024-dim)
│   ├── img_backbone
│   │   ├── resnet.py::resnet50(pretrained=True)
│   │   │   └── [BROKEN IN MODERN TORCHVISION] torchvision.models.utils.load_state_dict_from_url
│   │   ├── GeM(p=3)
│   │   └── nn.Linear(2048, 1024)
│   ├── local_conv (spatial cross-attention, T=7.0)
│   │   ├── blocks.py::Inception2d (gamma, beta modulation)
│   │   └── GeM(p=3)
│   └── compose_img_text -> returns (local_out, local_feature [dim 1024], attn)
└── global_model: model.py::compose_global
    ├── text_model.py::TextLSTMModel
    ├── img_backbone -> resnet.py::resnet50(pretrained=True)
    ├── global_conv (linguistic token cross-attention, T=4.0)
    │   ├── blocks.py::Inception1d (gamma, beta modulation)
    │   └── GeM(p=3)
    └── compose_img_text -> returns (global_feature [dim 1024], attn)
```

### 3. Inspected vs. Excluded Files
- **Inspected**:
  - `README.md`: Fully inspected; environment claims, per-category training invocations, missing evaluation instructions.
  - `test.py`: Fully inspected; lines 80–149 (FashionIQ branch) and lines 6–79 (Fashion200k branch). Confirmed lack of CLI / `__main__` entrypoint.
  - `datasets.py`: Fully inspected; lines 152–258 (`FashionIQ` class, path concatenation, query extraction, pair-union gallery building).
  - `model.py`: Fully inspected; `compose_local` and `compose_global` architectures, feature dimensions, and forward paths.
  - `resnet.py`: Fully inspected; ResNet-50 definition, URL table, and legacy import mechanism.
  - `text_model.py`: Fully inspected; `SimpleVocab` tokenization, ASCII filtering, punctuation deletion, and vocabulary building.
  - `train.py`: Fully inspected; `load_dataset`, `create_model_and_optimizer`, per-category model persistence (`local_model.pt`, `global_model.pt`).
  - `utils.py`: Inspected; checkpoint serialization and dictionary saving functions.
  - `blocks.py`: Inspected; verified 1D/2D Inception modules and GeM pooling arithmetic.
- **Excluded**:
  - `datasets.py:1-150` (`Shoes`) and `datasets.py:260-462` (`Fashion200k`): Non-FashionIQ dataset loaders, unreachable on FashionIQ CIR evaluation.
  - `test.py:8-79`: Fashion200k evaluation branch, unreachable for FashionIQ.

### 4. README vs. Code Discrepancies
- **Non-executable Test Command**: `README.md:69-71` states:
  ```bash
  python test.py
  ```
  Code inspection of `test.py:1-149` reveals `test.py` defines only `def test(params, local_model, global_model, testset, dataname):`. It contains **no** `if __name__ == '__main__':` block, **no** CLI argument parser, and **no** checkpoint loading logic. Executing `python test.py` exits immediately with status 0 without performing any evaluation.
- **Environment Declaration vs. Pinned Artifacts**: `README.md:17-21` specifies `Python 3.7.6`, `PyTorch 1.6.0`, `TITAN XP`, `Ubuntu 14.04.6 LTS`. No `requirements.txt` or `environment.yml` file is provided in the repository.
- **Unmapped Checkpoint Archive**: `README.md:76` provides a Google Drive link (`CLVCNET.zip`), but provides no description, checksum, or mapping to the 6 per-category/per-state model weights required for FashionIQ.

---

## C. NATIVE EVALUATION

### 1. Proof of Pair-Union Gallery Construction
In `workbench/third_party/CLVCNet/datasets.py:153, 233-257`:
- Default initialization: `FashionIQ.__init__(..., gallery_all=False)` (`datasets.py:153`).
- `FashionIQ.get_test_targets()` implements:
  ```python
  # datasets.py:243-256
  test_targets_id = []
  queries = self.get_test_queries()
  for i in queries:
      if i['source_img_id'] not in test_targets_id:
          test_targets_id.append(i['source_img_id'])
      if i['target_img_id'] not in test_targets_id:
          test_targets_id.append(i['target_img_id'])

  self.test_targets = []
  for i in test_targets_id:
      out = {}
      out['target_img_id'] = i
      out['target_img_data'] = self.get_img(self.images[i])
      self.test_targets.append(out)   
  return self.test_targets
  ```
- **Proof**: The gallery is constructed by traversing the query list in source order and appending `source_img_id` (reference) and `target_img_id` (target) only on their first appearance. This produces the exact ordered unique union of val references + targets (`fashioniq_val_split`).

### 2. Proof of Reference Exclusion
In `workbench/third_party/CLVCNet/test.py:131-136`:
```python
# test.py:131-135
test_targets_id = []
for i in test_targets:
    test_targets_id.append(i['target_img_id'])
for i, t in enumerate(test_queries):
    sims[i, test_targets_id.index(t['source_img_id'])] = -10e10
```
- **Masking Mechanism**: The index of the reference image (`t['source_img_id']`) in the gallery (`test_targets_id`) is looked up via `.index()`.
- **Exact Mask Value**: `-10e10` (`test.py:135`). The query image's similarity is forced to negative infinity prior to sorting.

### 3. LOCAL vs. GLOBAL Model States
The CLVC-Net architecture decomposes linguistic-visual composition into two parallel representations:
1. **Local State (`local_model`, class `compose_local` in `model.py:115-185`)**:
   - Focus: Fine-grained spatial modifications.
   - Module: `local_conv` (`model.py:13-38`), temperature $T=7.0$. Computes cross-attention between the 1024-dim linguistic embedding and the $7 \times 7$ ResNet-50 visual feature map.
   - Modulation: Modulates the spatial visual tensor using 2D Inception blocks (`Inception2d`).
   - Query Output: `local_model.compose_img_text(imgs, mods)[1]` (1024-dimensional feature vector).
   - Target Output: `local_model.extract_img_feature(imgs)[1]` (1024-dimensional feature vector).
2. **Global State (`global_model`, class `compose_global` in `model.py:187-258`)**:
   - Focus: Semantic/word-level modification.
   - Module: `global_conv` (`model.py:40-67`), temperature $T=4.0$. Computes cross-attention over linguistic sequence tokens conditioned on visual features.
   - Modulation: Modulates linguistic token representations using 1D Inception blocks (`Inception1d`).
   - Query Output: `global_model.compose_img_text(imgs, mods)[0]` (1024-dimensional feature vector).
   - Target Output: `global_model.extract_img_feature(imgs)[1]` (1024-dimensional feature vector).

**Multi-State Evaluation Requirement**:
- **BOTH models are simultaneously required for evaluation.**
- In `test.py:100`:
  ```python
  f = torch.cat([
      F.normalize(local_model.compose_img_text(imgs, mods)[1]),
      F.normalize(global_model.compose_img_text(imgs, mods)[0])
  ], dim=1)
  ```
  Query feature `f` is formed by concatenating the L2-normalized 1024-dim local representation and 1024-dim global representation into a 2048-dimensional query vector.
- In `test.py:116`:
  ```python
  imgs = torch.cat([
      F.normalize(local_model.extract_img_feature(imgs)[1]),
      F.normalize(global_model.extract_img_feature(imgs)[1])
  ], dim=1).data.cpu().numpy()
  ```
  Target feature `imgs` is formed by concatenating the L2-normalized 1024-dim local target feature and 1024-dim global target feature into a 2048-dimensional gallery vector.
- **Independence & Cardinality**: Training is conducted independently per FashionIQ category (`train.py:31-44`). `train.py:274-275` saves `local_model.pt` and `global_model.pt` in `args.model_dir`. Evaluating all 3 categories requires **6 distinct whole-model checkpoint files** (3 categories $\times$ 2 states: `dress/{local,global}`, `shirt/{local,global}`, `toptee/{local,global}`).

### 4. Caption Composition
- In `datasets.py:173-175`:
  ```python
  def concat_text(self, captions):
      text = "<BOS> {} <AND> {} <EOS>".format(captions[0], captions[1])
      return text
  ```
- The two modifier sentences are concatenated with `<BOS>`, `<AND>`, and `<EOS>` marker tokens.

### 5. Tokenizer & Vocabulary
- Class `SimpleVocab` (`text_model.py:8-50`):
  - Special tokens initialized: `<UNK>`: 0, `<AND>`: 1, `<BOS>`: 2, `<EOS>`: 3 (`text_model.py:14-17`).
  - Text normalization in `tokenize_text` (`text_model.py:23-27`):
    ```python
    text = text.encode('ascii', 'ignore').decode('ascii')
    trans = str.maketrans({key: None for key in string.punctuation})
    tokens = str(text).lower().translate(trans).strip().split()
    return tokens
    ```
  - **Upstream Tokenization Defect**: `string.punctuation` contains `<` and `>`. Tokenizing `"<BOS> ... <AND> ... <EOS>"` deletes `<` and `>`, leaving `'bos'`, `'and'`, `'eos'`. Consequently:
    - Pre-registered tokens `<BOS>` (2) and `<EOS>` (3) are **never matched**.
    - `'bos'` and `'eos'` map to `<UNK>` (0) via `self.word2id.get(t, 0)` (`text_model.py:44`).
    - `'and'` matches only if "and" was encountered as a plain word in training texts.
  - **Missing Vocabulary Asset**: In `text_model.py:65-68`, if `texts_to_build_vocab=None`:
    ```python
    vocab_data = json.load(open("simplevocab.json"))
    ```
    `simplevocab.json` does not exist in the repository. Instantiating `TextLSTMModel` without training data crashes with `FileNotFoundError`. When loading whole-model checkpoints (`local_model.pt`, `global_model.pt`), `self.text_model.vocab` is deserialized from the pickle, bypassing `simplevocab.json`.

### 6. Scoring, Sorting, Top-K, and Recall
- **Feature Normalization**: In `test.py:122-125`, each 2048-dim query and target vector is L2-normalized:
  ```python
  for i in range(all_queries.shape[0]):
      all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
  for i in range(all_imgs.shape[0]):
      all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])
  ```
- **Scoring**: Matrix dot product `sims = all_queries.dot(all_imgs.T)` (`test.py:129`).
- **Sorting**: `nn_result = [np.argsort(-sims[i, :])[:50] for i in range(sims.shape[0])]` (`test.py:138`).
- **Top-K**: Evaluated at $K \in \{1, 10, 50\}$ (`test.py:142`).
- **Recall Definition & Denominator**:
  ```python
  # test.py:142-147
  for k in [1, 10, 50]:
      r = 0.0
      for i, nns in enumerate(nn_result):
          if test_targets_id.index(test_queries[i]['target_img_id']) in nns[:k]:
              r += 1
      r = 100 * r / len(nn_result)
  ```
  Denominator is `len(nn_result)` = total number of validation queries in that category.

### 7. Macro Aggregation
- Evaluator `test.py:test` evaluates one category at a time.
- Paper Table 1 macro average is the unweighted arithmetic mean:
  $$\text{Macro } R@K = \frac{R@K_{\text{dress}} + R@K_{\text{shirt}} + R@K_{\text{toptee}}}{3}$$
  - $R@10: (29.85 + 28.75 + 33.50) / 3 = 30.70$
  - $R@50: (56.47 + 54.76 + 64.00) / 3 = 58.41$
  - Mean Recall: $(30.70 + 58.41) / 2 = 44.56$ (`WORKBENCH_DERIVED`).

---

## D. PREPARATION

### 1. Resized Images
- Read location: `datasets.py:195` reads `os.path.join(self.image_dir, self.name, image_name + ".jpg")`.
- Path resolution: `datasets.py:157` (`self.image_dir = self.path + 'resized_image'`).
- **Trailing Slash Defect**: Passing `path` without a trailing slash (e.g. `/data/fashioniq`) yields `/data/fashioniqresized_image`, causing `FileNotFoundError`.
- **Expected Layout**: `resized_image/{dress,shirt,toptee}/{image_id}.jpg` (per-category layout, 256px resolution).
- **Source Availability**: Upstream `README.md:28` points to external `https://github.com/XiaoxiaoGuo/fashion-iq/tree/master/start_kit/resize_images.py`. No resize tool is bundled in `CLVCNet`.
- **Workbench Contract**: `preparation_contracts.yaml:3-59` registers `clvc_net_fashioniq` (`layout: per_category`, `image_size: 256`, `extension: .jpg`).

### 2. ResNet Backbone Assets
- Read location: `model.py:72` calls `resnet.resnet50(pretrained=True)`.
- Weight URL: `resnet.py:12` points to `https://download.pytorch.org/models/resnet50-19c8e357.pth`.
- Download handler: `resnet.py:3` (`from torchvision.models.utils import load_state_dict_from_url`).
- Modern torchvision breakage: `torchvision.models.utils` was deprecated and removed in torchvision $\ge 0.13$. In modern environments, importing `resnet.py` raises `ModuleNotFoundError: No module named 'torchvision.models.utils'`.
- Contract registration: `preparation_contracts.yaml:44-56` tracks `torchvision ResNet-50 ImageNet weights` (`cache_path_suffix: hub/checkpoints/resnet50-19c8e357.pth`, `expected_sha256_prefix: 19c8e357`).

### 3. Correction Dictionaries
- **None required or used**. CLVC-Net reads raw caption files `captions/cap.{category}.{split}.json` directly.

### 4. Tokenizer / Vocabulary Files
- `simplevocab.json` (`text_model.py:66`) is **missing from the pinned repository**.
- Recoverable from pickled whole-model weights (`local_model.pt`, `global_model.pt`), or regenerable from training captions `cap.{category}.train.json`.

### 5. Determinism & Environment
- Evaluation in `test.py` is fully deterministic: models set to `.eval()`, `torch.no_grad()`, center crop 224 applied (`train.py:92`).

---

## E. CHECKPOINT

### 1. Checkpoint Loader & Multi-State Requirement
- **No Evaluation Loader**: The pinned repository contains no test-time checkpoint loader.
- **Serialization**: `train.py:274-275` saves checkpoints as PyTorch whole models:
  ```python
  torch.save(local_model, os.path.join(model_dir, 'local_model.pt'))
  torch.save(global_model, os.path.join(model_dir, 'global_model.pt'))
  ```
- **Required Files**: 6 distinct whole-model files:
  - `dress/local_model.pt`, `dress/global_model.pt`
  - `shirt/local_model.pt`, `shirt/global_model.pt`
  - `toptee/local_model.pt`, `toptee/global_model.pt`

### 2. Checkpoint Provenance & Mapping Status
- Registry variant: `author_drive_unresolved` (`models.yaml:595-605`).
- Source URL: `https://drive.google.com/file/d/159rBhWyhkLN7sXAi8iyW_ljzFNLJinKa/view?usp=sharing` (`CLVCNET.zip`).
- Checkpoint status: `CHECKPOINT_MAPPING_UNVERIFIED`.
- Mapping status: `UNVERIFIED`. The upstream repository and README provide no member list, checksums, or file-to-category associations for `CLVCNET.zip`.

### 3. Behavior in `workbench/backend/checkpoint_selection.py`
Executing `checkpoint_selection.select_checkpoints` (`RUNTIME_VERIFIED`):
- Policy `strict_clean`:
  - Returns `status='BLOCKED'`, `reason_code='BLOCKED_CHECKPOINT'`.
  - Reason: `no checkpoint variant records a training condition, so none can be called clean under policy strict_clean. Policy include_flagged_conditions runs these with the condition recorded as unknown.`
- Policy `include_flagged_conditions`:
  - Returns `status='BLOCKED'`, `reason_code='BLOCKED_CHECKPOINT'`.
  - Reason: `checkpoint mapping unverified for author_drive_unresolved; the upstream repository does not establish which variant is the benchmark checkpoint`.

---

## F. WORKBENCH INTEGRATION

### 1. Registry & Contracts
- Declared in `workbench/registry/models.yaml:538-605`:
  - `replay_status: BLOCKED_NO_SOURCE_REPLAY`
  - `command_status: COMMAND_BLOCKED_SOURCE_ENTRYPOINT`
  - `category_model_policy: independent`
  - `category_checkpoint_structure`: `categories: [dress, shirt, toptee]`, `per_category_members: [local, global]`, `member_count_expected: 6`, `mapping_status: UNVERIFIED`.
  - `preparation_contract: manual_clvc_resized`
  - `preparation_id: clvc_net_fashioniq`
- Declared in `workbench/registry/preparation_contracts.yaml:3-59`:
  - Contract `clvc_net_fashioniq` specifies FashionIQ raw splits/captions, generated 256px resized JPEGs (`resized_image/{dress,shirt,toptee}`), and `torchvision ResNet-50 ImageNet weights`.

### 2. Adapter Implementation
In `workbench/backend/adapters/models.py:74-76`:
```python
class CLVCNetAdapter(OfficialScriptAdapter):
    model_id = "clvc_net"
    script = "test.py"
```
- `CLVCNetAdapter` does not implement `command()`. It inherits `OfficialScriptAdapter.command` (`models.py:24-25`), which raises:
  `NotImplementedError("upstream evaluation command requires audit before execution")`.

### 3. PC Readiness Classification
Running `python3 -m workbench.scripts.pc_readiness --model clvc_net` (`RUNTIME_VERIFIED`):
```
models
  clvc_net                   BLOCKED_IMPLEMENTATION
      command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT
      checkpoint author_drive_unresolved:
        - torchvision ResNet-50 ImageNet weights missing: /home/heheboiz/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth
        - external runtime asset file missing: torchvision ResNet-50 ImageNet weights (/home/heheboiz/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth)
        - command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT
        - checkpoint missing: /home/heheboiz/data/cir/workbench/artifacts/checkpoints/clvc_net/checkpoint-unresolved.pt
        - checkpoint mapping unresolved

summary: {"BLOCKED_IMPLEMENTATION": 1}
```
- **Blocker Needle-Ordering Analysis**:
  - `preflight_blockers` returns 4 blockers: ResNet-50 weights missing, external asset missing, command not runnable, and environment unconfigured (`WORKBENCH_CLVC_PYTHON`).
  - In `workbench/scripts/pc_readiness.py:48-67`, `BLOCKER_KINDS` prioritizes needles in outer loop order:
    1. `"command not runnable"` (row 6) $\rightarrow$ `BLOCKED_IMPLEMENTATION`.
    2. `"environment"` (row 7) $\rightarrow$ `BLOCKED_ENVIRONMENT`.
    3. `"missing"` (row 19) $\rightarrow$ `BLOCKED_EXTERNAL_ASSET`.
  - `"command not runnable"` matches blocker `command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT` before environment or asset needles are reached.
  - **Classification Soundness**: Sound. Because `test.py` has no CLI entrypoint and `CLVCNetAdapter.command` raises `NotImplementedError`, the model is genuinely blocked at the implementation level before environment or assets can be exercised.

### 4. Metrics Extraction
In `workbench/backend/metrics_extraction.py`:
- No parser function is registered for `clvc_net`.
- `extract_metrics` falls through to default at line 415, returning:
  `MetricExtractionResult(extraction_status="AGGREGATE_PARSER_UNAVAILABLE", parser_id=None, ...)`

---

## G. OBSERVATION CAPABILITY

### 1. Score & Ranking Production Point
In `workbench/third_party/CLVCNet/test.py:129-138`:
```python
# test.py:129
sims = all_queries.dot(all_imgs.T)

# test.py:134-135
for i, t in enumerate(test_queries):
    sims[i, test_targets_id.index(t['source_img_id'])] = -10e10

# test.py:138
nn_result = [np.argsort(-sims[i, :])[:50] for i in range(sims.shape[0])]
```
- Similarity matrix `sims`: query-gallery cosine similarity.
- Masked reference: position `test_targets_id.index(t['source_img_id'])` set to `-10e10`.
- Ranking: `np.argsort(-sims[i, :])[:50]` generates the Top-50 ranked gallery target indices for query $i$.

### 2. Per-Query Export Feasibility
- **Feasible with Replay Wrapper**:
  - Query metadata: `test_queries[i]['source_img_id']` (candidate), `test_queries[i]['target_img_id']` (target), and `test_queries[i]['mod']['str']` (caption).
  - Gallery metadata: `test_targets[idx]['target_img_id']` mapping to split image name `self.images[target_id]`.
  - Top-K scores: `sims[i, nn_result[i]]`.
  - Per-query ranking JSON can be emitted directly following line 138 in an adapted replay harness.

### 3. Blockers to Observation
1. Pinned `test.py` lacks a CLI parser, `__main__` entrypoint, and checkpoint loader.
2. Checkpoint archive `CLVCNET.zip` member files are unmapped to the 6 category/state model slots.
3. Modern torchvision environment breaks `resnet.py` (`load_state_dict_from_url`).
4. Workbench adapter raises `NotImplementedError`.

---

## H. DEFECTS

### 1. `CONFIRMED_BUG` — Missing Executable Entrypoint and Checkpoint Loader in `test.py`
- **Location**: `workbench/third_party/CLVCNet/test.py:1-149` vs `workbench/third_party/CLVCNet/README.md:70`.
- **Precondition**: Operator executes documented test command `python test.py`.
- **Actual Behavior**: Script defines only function `test(...)` with no `__main__` block, argument parser, or checkpoint loader. Exits 0 without performing evaluation.
- **Expected Behavior**: Executable CLI accepting `--dataset`, `--category`, `--checkpoint_dir`, and running evaluation.
- **Status**: `CONFIRMED_BUG` (upstream defect) / `EXTERNAL_BLOCKER` (evaluation blocker).

### 2. `CONFIRMED_BUG` — Missing Trailing Slash in Dataset Path Concatenation
- **Location**: `workbench/third_party/CLVCNet/datasets.py:157-159`.
- **Lines**:
  ```python
  self.image_dir = self.path + 'resized_image'
  self.split_dir = self.path + 'image_splits'
  self.caption_dir = self.path + 'captions'
  ```
- **Precondition**: `path` passed without trailing slash (e.g. `/data/fashioniq`).
- **Actual Behavior**: Concatenates to `/data/fashioniqresized_image`, causing `FileNotFoundError`.
- **Expected Behavior**: Paths joined with `os.path.join(self.path, ...)` or explicit trailing slash enforcement.
- **Status**: `CONFIRMED_BUG` (upstream defect).

### 3. `CONFIRMED_BUG` — Tokenizer Strips Special Marker Brackets
- **Location**: `workbench/third_party/CLVCNet/datasets.py:174` vs `workbench/third_party/CLVCNet/text_model.py:24-26, 44`.
- **Precondition**: Text formatted as `"<BOS> {} <AND> {} <EOS>"`.
- **Actual Behavior**: `SimpleVocab.tokenize_text` strips `string.punctuation` (including `<` and `>`), converting `<BOS>` to `bos`, `<AND>` to `and`, `<EOS>` to `eos`. Pre-registered tokens `<BOS>` (2) and `<EOS>` (3) are never assigned; `bos` and `eos` map to `<UNK>` (0).
- **Expected Behavior**: Punctuation filtering that preserves defined boundary tokens.
- **Status**: `CONFIRMED_BUG` (upstream defect).

### 4. `CONFIRMED_BUG` — Missing `simplevocab.json` Fallback Asset
- **Location**: `workbench/third_party/CLVCNet/text_model.py:66`.
- **Line**: `vocab_data = json.load(open("simplevocab.json"))`.
- **Precondition**: `TextLSTMModel` instantiated with `texts_to_build_vocab=None`.
- **Actual Behavior**: Crashes with `FileNotFoundError: [Errno 2] No such file or directory: 'simplevocab.json'`.
- **Expected Behavior**: Fallback JSON asset bundled in repository, or explicit descriptive error message.
- **Status**: `CONFIRMED_BUG` (upstream defect).

### 5. `EXTERNAL_BLOCKER` — Legacy Torchvision Import Incompatibility in `resnet.py`
- **Location**: `workbench/third_party/CLVCNet/resnet.py:3`.
- **Line**: `from torchvision.models.utils import load_state_dict_from_url`.
- **Precondition**: Executing in modern PyTorch/Torchvision environment (torchvision $\ge 0.13$).
- **Actual Behavior**: Crashes with `ModuleNotFoundError: No module named 'torchvision.models.utils'`.
- **Expected Behavior**: Executing under legacy pinned environment (`torchvision==0.7.0`, `torch==1.6.0`) or using compatibility shim (`from torch.hub import load_state_dict_from_url`).
- **Status**: `EXTERNAL_BLOCKER` (environment blocker).

### 6. `EXTERNAL_BLOCKER` — Unmapped Google Drive Checkpoint Archive (`CLVCNET.zip`)
- **Location**: `workbench/third_party/CLVCNet/README.md:76`, `workbench/registry/models.yaml:546-551, 598`.
- **Precondition**: Attempting FashionIQ evaluation from author-provided weights.
- **Actual Behavior**: Google Drive link points to `CLVCNET.zip` with no member listing, checksums, or category/state mappings.
- **Expected Behavior**: Explicitly mapped 6-member bundle corresponding to `{dress,shirt,toptee} \times {local,global}`.
- **Status**: `EXTERNAL_BLOCKER` (asset blocker).

### 7. `EXTERNAL_BLOCKER` — Missing Metrics Parser in Workbench
- **Location**: `workbench/backend/metrics_extraction.py`.
- **Precondition**: Running evaluation and extracting metrics.
- **Actual Behavior**: `extract_metrics` returns `AGGREGATE_PARSER_UNAVAILABLE`.
- **Expected Behavior**: Dedicated `parse_clvc_output` parser validating category R@10/R@50 and macro arithmetic mean.
- **Status**: `EXTERNAL_BLOCKER` (workbench integration blocker).

### 8. `BY_DESIGN` — Arbitrary Code Execution Risk in Whole-Model Checkpoints
- **Location**: `workbench/third_party/CLVCNet/train.py:274-275` (`torch.save(model, ...)`).
- **Risk**: Deserializing unchecked whole-model Python pickles via `torch.load` can execute arbitrary code.
- **Workbench Mitigation**: `workbench/backend/registry.py` enforces cryptographic SHA-256 verification before checkpoint loading.
- **Status**: `BY_DESIGN`.

### 9. `REJECTED_FINDING` — Prior Claim of CLIP-based Textual Branch
- **Claim**: Early workbench notes hypothesized CLVC-Net used a CLIP textual branch.
- **Evidence**: Source inspection of `model.py:119-122, 191-194` and paper §3.2.1 confirms CLVC-Net uses a two-layer LSTM (`TextLSTMModel`) with learned word embeddings. No CLIP dependency exists in CLVC-Net.
- **Verdict**: Rejected. Properly corrected in `papers.yaml:540-542` and `models.yaml:540-542`.
- **Status**: `REJECTED_FINDING`.

---

## I. EVIDENCE

1. **Pinned Source Commit SHA Verification**:
   ```bash
   git -C workbench/third_party/CLVCNet rev-parse HEAD
   # Output: bd9b6889489806cb60676597880f91d918b279cb
   ```
2. **Commit Count Verification**:
   ```bash
   git -C workbench/third_party/CLVCNet rev-list --all --count
   # Output: 1
   ```
3. **ResNet Import Incompatibility Reproduction**:
   ```bash
   .venv/bin/python -c "import sys; sys.path.insert(0, 'workbench/third_party/CLVCNet'); import resnet"
   # Output:
   # Traceback (most recent call last):
   #   File "<string>", line 1, in <module>
   #   File "/home/heheboiz/data/cir/workbench/third_party/CLVCNet/resnet.py", line 3, in <module>
   #     from torchvision.models.utils import load_state_dict_from_url
   # ModuleNotFoundError: No module named 'torchvision.models.utils'
   ```
4. **Tokenizer Punctuation Stripping Demonstration**:
   ```bash
   .venv/bin/python -c "
   import string
   text = '<BOS> red dress <AND> shorter sleeves <EOS>'
   text = text.encode('ascii', 'ignore').decode('ascii')
   trans = str.maketrans({key: None for key in string.punctuation})
   tokens = str(text).lower().translate(trans).strip().split()
   print(tokens)
   "
   # Output: ['bos', 'red', 'dress', 'and', 'shorter', 'sleeves', 'eos']
   ```
5. **Checkpoint Selection Behavior**:
   ```bash
   .venv/bin/python -c "
   from workbench.backend.checkpoint_selection import select_checkpoints, INCLUDE_FLAGGED_CONDITIONS;
   from workbench.backend.registry import load_registry;
   reg = load_registry();
   m = [x for x in reg['models'] if x['model_id'] == 'clvc_net'][0];
   print('strict_clean:', select_checkpoints(m).status, select_checkpoints(m).reason_code);
   print('flagged:', select_checkpoints(m, policy=INCLUDE_FLAGGED_CONDITIONS).status, select_checkpoints(m, policy=INCLUDE_FLAGGED_CONDITIONS).reason_code)
   "
   # Output:
   # strict_clean: BLOCKED BLOCKED_CHECKPOINT
   # flagged: BLOCKED BLOCKED_CHECKPOINT
   ```
6. **Preflight Blockers Reproduction**:
   ```bash
   .venv/bin/python -c "
   from workbench.backend.preflight import preflight_blockers;
   from workbench.backend.registry import load_registry;
   from workbench.backend.operator_config import resolve_config;
   reg = load_registry(); cfg = resolve_config();
   m = [x for x in reg['models'] if x['model_id'] == 'clvc_net'][0];
   print(preflight_blockers(m, cfg))
   "
   # Output:
   # ['torchvision ResNet-50 ImageNet weights missing: /home/heheboiz/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth',
   #  'external runtime asset file missing: torchvision ResNet-50 ImageNet weights (/home/heheboiz/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth)',
   #  'command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT',
   #  'execution environment not verified: WORKBENCH_CLVC_PYTHON=None [UNCONFIGURED/UNCONFIGURED] (WORKBENCH_CLVC_PYTHON not set)']
   ```
7. **PC Readiness Classification Execution**:
   ```bash
   .venv/bin/python -m workbench.scripts.pc_readiness --model clvc_net
   # Output summary: {"BLOCKED_IMPLEMENTATION": 1}
   ```

---

## J. VERDICT

| Axis | Verdict | Evidence Summary |
|---|---|---|
| **SOURCE** | **`BLOCKED_SOURCE`** | `test.py` lacks CLI entrypoint / `__main__`; `text_model.py:66` references missing `simplevocab.json`; `resnet.py:3` crashes in modern torchvision via removed `torchvision.models.utils`. |
| **PROTOCOL** | **`PROTOCOL_VERIFIED`** | Protocol C (`fashioniq_val_split`) verified: pair-union gallery accumulated in `datasets.py:243-256`, reference masked to `-10e10` in `test.py:135`. |
| **COMMAND** | **`COMMAND_BLOCKED`** | `test.py` contains only function `test(...)` with no loader or CLI; `CLVCNetAdapter.command` raises `NotImplementedError`; registry records `COMMAND_BLOCKED_SOURCE_ENTRYPOINT`. |
| **CHECKPOINT** | **`CHECKPOINT_BLOCKED`** | Evaluator requires 6 distinct whole-model checkpoints (3 categories $\times$ 2 states: local/global); author Google Drive archive `CLVCNET.zip` member mapping is completely unresolved; `checkpoint_selection.py` blocks under both policies. |
| **QUERY EXPORT** | **`QUERY_EXPORT_IMPLEMENTABLE`** | Top-50 ranking production point verified at `test.py:138` (`nn_result`). Query IDs, target IDs, and cosine similarities are accessible in memory. Standalone execution blocked by missing loader and unmapped weights. |
| **PC VALIDATION** | **`BLOCKED_IMPLEMENTATION`** | `pc_readiness.py` classifies model as `BLOCKED_IMPLEMENTATION` due to `COMMAND_BLOCKED_SOURCE_ENTRYPOINT`, unmapped checkpoints, missing ResNet-50 ImageNet weights, and unconfigured `WORKBENCH_CLVC_PYTHON`. |
