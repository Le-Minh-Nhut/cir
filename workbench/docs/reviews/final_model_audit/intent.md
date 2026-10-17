# Final Model Audit: INTENT

**Audit Date:** 2026-10-17  
**Model ID:** `intent`  
**Protocol:** Protocol B (`fashioniq_full_gallery_ref_excluded`: full ordered val-split gallery, reference excluded)  
**Evaluation Standard:** `STATIC_AUDIT` throughout unless explicitly marked `RUNTIME_VERIFIED`. No GPU inference was executed. All code paths, ASTs, and CLI/API checks were executed in read-only mode against HEAD (`5726d5f`) on branch `research/cir-failure-workbench`.

---

## A. MODEL IDENTITY

- **Model ID:** `intent` (`workbench/registry/models.yaml:305`)
- **Method Name:** `INTENT` (`workbench/registry/models.yaml:315`)
- **Paper Title:** "INTENT: Invariance and Discrimination-aware Noise Mitigation for Robust Composed Image Retrieval" (`workbench/registry/models.yaml:319`, `workbench/registry/paper_benchmarks/papers.yaml:1239`)
- **Publication Venue & Year:** AAAI 2026 (Proceedings of the 40th AAAI Conference on Artificial Intelligence, March 2026; DOI: `10.1609/aaai.v40i25.39181`, `publication_year: 2026` at `models.yaml:317`)
- **Paper URL:** `https://ojs.aaai.org/index.php/AAAI/article/view/39181` (`models.yaml:320`)
- **Source Repository URL:** `https://github.com/iLearn-Lab/AAAI26-INTENT` (`models.yaml:321`; canonical redirect from `https://github.com/ZivChen-Ty/INTENT`)
- **Pinned Commit SHA:** `2a75fc6f57d1a036bf3167a7fe1d2cee072f51f3` (`models.yaml:323`)
  - Verification: `git -C workbench/third_party/INTENT rev-parse HEAD` outputs `2a75fc6f57d1a036bf3167a7fe1d2cee072f51f3` (`RUNTIME_VERIFIED`). Pin matches checkout HEAD.
  - Commit Message & Date: `Update README.md`, 2026-05-25 18:15:51 +0800.
- **Paper Benchmark Reference Rows:** Registered in `workbench/registry/paper_benchmarks/papers.yaml:1236-1441` across four noise regimes (Table 2, page 6):
  1. `intent_aaai26_table2_fiq_n00` (0% noise training block, fully supervised):
     - dress: R@10 = `50.32`, R@50 = `72.10`
     - shirt: R@10 = `56.32`, R@50 = `74.93`
     - toptee: R@10 = `59.28`, R@50 = `78.45`
     - Average: R@10 = `55.31`, R@50 = `75.16`
     - AVG: `65.24`
     - Arithmetic consistency:
       - Average R@10: $(50.32 + 56.32 + 59.28) / 3 = 165.92 / 3 = 55.3067 \to 55.31$ (exact).
       - Average R@50: $(72.10 + 74.93 + 78.45) / 3 = 225.48 / 3 = 75.1600 \to 75.16$ (exact).
       - AVG: $(55.31 + 75.16) / 2 = 130.47 / 2 = 65.235 \to 65.24$ (exact with half-up rounding).
  2. `intent_aaai26_table2_fiq_n20` (20% noise training block):
     - dress: R@10 = `49.32`, R@50 = `71.43`
     - shirt: R@10 = `55.32`, R@50 = `73.57`
     - toptee: R@10 = `58.01`, R@50 = `78.46`
     - Average: R@10 = `54.22`, R@50 = `74.49`
     - AVG: `64.36`
     - Arithmetic consistency: $(49.32 + 55.32 + 58.01)/3 = 54.2167 \to 54.22$; $(71.43 + 73.57 + 78.46)/3 = 74.4867 \to 74.49$; $(54.22 + 74.49)/2 = 64.355 \to 64.36$.
  3. `intent_aaai26_table2_fiq_n50` (50% noise training block):
     - dress: R@10 = `47.99`, R@50 = `71.24`
     - shirt: R@10 = `52.78`, R@50 = `72.48`
     - toptee: R@10 = `56.79`, R@50 = `76.23`
     - Average: R@10 = `52.52`, R@50 = `73.32`
     - AVG: `62.92`
     - Arithmetic consistency: $(47.99 + 52.78 + 56.79)/3 = 52.5200 \to 52.52$; $(71.24 + 72.48 + 76.23)/3 = 73.3167 \to 73.32$; $(52.52 + 73.32)/2 = 62.920 \to 62.92$.
  4. `intent_aaai26_table2_fiq_n80` (80% noise training block):
     - dress: R@10 = `42.07`, R@50 = `65.58`
     - shirt: R@10 = `50.38`, R@50 = `69.41`
     - toptee: R@10 = `53.09`, R@50 = `73.91`
     - Average: R@10 = `48.51`, R@50 = `69.63`
     - AVG: `59.07`
     - Arithmetic consistency: $(42.07 + 50.38 + 53.09)/3 = 48.5133 \to 48.51$; $(65.58 + 69.41 + 73.91)/3 = 69.6333 \to 69.63$; $(48.51 + 69.63)/2 = 59.070 \to 59.07$.

All four reported rows are internally consistent with the paper's category columns under unweighted arithmetic macro-averaging and standard half-up rounding.

---

## B. SOURCE COVERAGE

### Pinned Tree Inventory (`workbench/third_party/INTENT`)
The pinned repository contains 1,543 git-tracked files:
- **Root Evaluation & Training Scripts:**
  1. `test.py`: Standalone callable evaluation module containing `test()` for FashionIQ, Birds, Shoes, Fashion200k, and `test_cirr_valset()` for CIRR.
  2. `train_INTENT.py`: Full training script with CLI parser (`argparse`), data initialization, training loop, and evaluation dispatch.
  3. `datasets.py`: PyTorch `Dataset` definitions for `FashionIQ` and `CIRR`, including caching logic, text normalization, and gallery split construction.
  4. `data_utils.py`: Preprocessing transforms (`TargetPad`, `SquarePad`, `targetpad_transform`, `squarepad_transform`).
  5. `utils.py`: Logging helpers, running averages, and JSON metric savers.
  6. `cirr_sub_BLIP2.py`: CIRR test-server submission generation script with hardcoded paths.
- **Documentation & Ancillary Assets:**
  7. `README.md`: Upstream project overview, setup commands, data layout instructions, and links.
  8. `LICENSE`: Apache License 2.0.
  9. `googleef10cadbf6a593ec.html`: Search console verification token.
  10. `figs/`: Documentation images, architecture diagrams (`pipeline.png`, `intervention.png`), and comparison tables (`fiq.png`, `cirr.png`).
- **Vendored Framework (`lavis/`):**
  11. Vendored Salesforce LAVIS framework (1,500+ files) customized for INTENT. Crucially includes `lavis/models/blip2_models/blip2_cir.py` (`Blip2QformerCir`), which implements the dual-stream BLIP-2 Q-Former model, causal projection, and feature extraction routines.

### Reachable Evaluation Dependency Graph
The execution path for FashionIQ evaluation requires:
`train_INTENT.py:train_and_evaluate` (or external caller) -> `test.py:test` -> `datasets.py:FashionIQ` (`get_test_data`) -> `data_utils.py:targetpad_transform` -> `lavis/models/blip2_models/blip2_cir.py:Blip2QformerCir` (`extract_retrieval_compose`, `extract_retrieval_target`).

- **Inspected Files:**
  - `test.py`: Lines 1–177 (FashionIQ evaluation path, query/target batch loops, tensor contraction, reference masking, and recall computation).
  - `datasets.py`: Lines 1–212 (`FashionIQ` dataset class, caching mechanism, `get_test_data`, `get_img`, caption spelling correction, and gallery generation).
  - `data_utils.py`: Lines 1–108 (`TargetPad`, `SquarePad`, and `targetpad_transform`).
  - `train_INTENT.py`: Lines 1–384 (CLI arguments, `load_dataset`, `create_model_and_optimizer`, `train`, `train_and_evaluate`, and `__main__` entrypoint).
  - `cirr_sub_BLIP2.py`: Lines 1–131 (Checkpoint loading pattern via `torch.load` whole-model deserialization).
  - `lavis/models/blip2_models/blip2_cir.py`: Lines 665–795 (`extract_retrieval_compose`, `extract_retrieval_target`, and Q-Former query token projections).
  - `README.md`: Lines 1–317 (Environment specifications, data directory instructions, and usage claims).
- **Excluded Files & Justification:**
  - `test.py` (lines 178–357): LaSCo and CIRR evaluation entrypoints (`lasco_query`, `lasco_target`, `test_lasco`, `test_cirr_valset`). Unreachable on FashionIQ path.
  - `datasets.py` (lines 213–385): `CIRR` dataset loader. Unreachable on FashionIQ path.
  - `cirr_sub_BLIP2.py` (lines 1–93): CIRR submission file formatter. Unreachable on FashionIQ path.
  - `lavis/` (all modules except `blip2_models/blip2_cir.py` and `models/__init__.py`): Vendored upstream tasks (VQA, captioning, video retrieval, diffusion).
  - `figs/`: Static graphical assets and paper comparison tables.

### README-vs-Code Discrepancies
1. **Dataset Layout Mismatch:** `README.md:135-155` directs users to place images in `FashionIQ/{dress,shirt,toptee}/*.jpg`. However, `datasets.py:31` hardcodes `self.image_dir = self.path + 'resized_image'`, requiring `FashionIQ/resized_image/{dress,shirt,toptee}/*.jpg`. Following the official README causes runtime crash with `FileNotFoundError`.
2. **Missing `requirements.txt`:** `README.md:123` instructs `pip install -r requirements.txt`. No `requirements.txt` exists in the repository (`git ls-files | grep -i require` yields empty).
3. **Undocumented Correction Dictionaries:** `datasets.py:98, 172` requires `captions/correction_dict_{dress,shirt,toptee}.json`. The README makes no mention of these files, and they are omitted from the source repository.
4. **Absent Standalone Evaluator:** `README.md:178-184` states:
   ```bash
   python train_INTENT.py
   ```
   "And the evaluation process is included." The README provides no command or documentation for evaluating pre-trained checkpoints on FashionIQ without executing the 20-epoch training loop.
5. **Missing 0% Noise Checkpoint:** `README.md:70-71` explicitly highlights the 0% noise setting as equivalent to traditional fully-supervised CIR. However, the author HuggingFace repository publishes only 20%, 50%, and 80% noise checkpoints; no 0% weights were released.

---

## C. NATIVE EVALUATION

### 1. Evaluator Entrypoint & CLI Interface
- Entrypoint: `test(params, model, testset, category, txt_processors)` defined at `workbench/third_party/INTENT/test.py:9`.
- **Defect:** `test.py` has **no** `if __name__ == '__main__':` block, **no** CLI argument parser (`argparse`), and **no** standalone invocation wrapper. It is a pure library callable.
- `train_INTENT.py:41-70` provides an argument parser, but contains **no** `--eval_only` or `--test_only` flag, and **no** `--checkpoint_path` or `--resume` argument. Invoking `train_INTENT.py` immediately begins a 20-epoch training run.
- Pinned source provides no executable command to evaluate pre-existing checkpoints.

### 2. Dataset Loading & Gallery Construction (Protocol B Proof)
- Implemented in `workbench/third_party/INTENT/datasets.py:166-210` (`FashionIQ.get_test_data`).
- Per category (`dress`, `shirt`, `toptee`), `get_test_data(name)` reads:
  1. `image_splits/split.{cat}.val.json` (`:168`) -> full list of validation candidate image IDs (`images`).
  2. `captions/cap.{cat}.val.json` (`:170`) -> query triplet annotations (`ref_captions`).
  3. `captions/correction_dict_{cat}.json` (`:172`) -> spell correction map (`correction_dict`).
- **Query Construction (`:174-188`):**
  - Iterates over all query triplets `idx in range(len(ref_captions))`:
    - `source_img_id = images.index(candidate)` (`:181`)
    - `source_img_data = self.get_test_img(name + '_' + candidate)` (`:182`)
    - `target_img_id = images.index(target)` (`:183`)
    - `target_img_data = self.get_test_img(name + '_' + target)` (`:184`)
    - `mod = {'str': mod_str}` (`:185`)
- **Gallery Construction (Protocol B Verification):**
  - Controlled by `self.split`, defaulting to `'original-split'` (`datasets.py:26`, `train_INTENT.py:61`):
    ```python
    elif self.split == 'original-split':
        for id, image_name in enumerate(images):
            test_targets_id.append(id)
            out = {}
            out['target_img_id'] = id
            out['target_img_data'] = self.get_img(name + '_' + image_name)      
            test_targets.append(out)
    ```
  - Under `'original-split'`, the gallery candidate set comprises **every single image in `split.{cat}.val.json` in original file order**.
  - Gallery cohort sizes on canonical FashionIQ:
    - `dress`: 3,817 images
    - `shirt`: 6,346 images
    - `toptee`: 5,373 images
  - Alternative branch `val-split` (`:191-202`) implements Protocol C (deduped query pair-union gallery), but is inactive under default configuration.

### 3. Scoring, Tensor Contraction & Reference Exclusion
- **Feature Extraction:**
  - Query composite features: `model.extract_retrieval_compose(imgs, mods)` (`test.py:112`) -> shape `(B, 1, 1, 256)`.
  - Target image features: `model.extract_retrieval_target(imgs)` (`test.py:128`) -> shape `(B, 256, 32)` (permuted from 32 query tokens of dim 256).
  - Concatenation over dataset batches yields `all_queries` of shape `(N_queries, 1, 1, 256)` and `all_imgs` of shape `(N_gallery, 256, 32)`.
- **Tensor Contraction:**
  - Lines 141–143 of `test.py`:
    ```python
    sims = np.matmul(all_queries, all_imgs) #all_queries.dot(all_imgs.T)
    sims = sims.squeeze()
    sims = sims.max(-1)
    ```
  - Broadcasting semantics: `(N_queries, 1, 1, 256)` matmul `(N_gallery, 256, 32)` contracts the inner 256 dimensions and broadcasts the leading axes, producing `(N_queries, N_gallery, 1, 32)`. Squeezing removes dimension 2, producing `(N_queries, N_gallery, 32)`. Taking `sims.max(-1)` selects the maximum dot product across the 32 Q-Former visual tokens, yielding a 2D similarity matrix `sims` of shape `(N_queries, N_gallery)`.
- **Reference Exclusion:**
  - Statement at `test.py:149-150`:
    ```python
    for i, t in enumerate(test_queries):
        sims[i, test_targets_id.index(t['source_img_id'])] = -10e10
    ```
  - Reference image index is penalized with `-10e10` ($-1 \times 10^{11}$) prior to sorting, strictly excluding it from retrieved top ranks.

### 4. Ranking, Recall & Macro Aggregation
- **Ranking:**
  `nn_result = [np.argsort(-sims[i, :])[:50] for i in range(sims.shape[0])]` (`test.py:153`, repeated at `:167`).
- **Recall@K:**
  - Evaluated for `k ∈ [1, 10, 50]` (`test.py:168-175`):
    ```python
    for k in [1, 10, 50]:
        r = 0.0
        for i, nns in enumerate(nn_result):
            if test_targets_id.index(test_queries[i]['target_img_id']) in nns[:k]:
                r += 1
        r = 100 * r / len(nn_result)
        out += [('{}_r{}'.format(name, k), r)]
    ```
  - Denominator is `len(nn_result)` = total number of test queries $N_{\text{queries}}$. Hit requires ground-truth target index in top $k$.
- **Macro Aggregation:**
  - Native `test.py` computes **no** macro averages across categories and **no** mean of R@10/R@50. It returns per-category tuples only.
  - `train_INTENT.py:286` simply sums R@10 across categories for epoch tracking (`current_score = current_score + t[1][1]`).
  - Macro numbers in Table 2 are computed externally as unweighted means.

---

## D. PREPARATION

### 1. Dataset Layout & Path Vulnerabilities
- Registry layout: `fashioniq_layout: fashioniq_ilearn_resized` (`models.yaml:326`).
- Image extensions: `fashioniq_image_extensions: [".jpg"]` (`models.yaml:327`).
- Evaluator reads:
  - `datasets.py:31-33`: `self.image_dir = self.path + 'resized_image'`, `self.split_dir = self.path + 'image_splits'`, `self.caption_dir = self.path + 'captions'`.
  - **Defect:** `self.path` is concatenated with string literals without `os.path.join`. Requires `self.path` to end with a trailing `/`.
  - Image paths: `os.path.join(self.image_dir, image_name.split('_')[0], image_name.split('_')[1] + ".jpg")` (`:139, 150`).
  - Expects images at `resized_image/{dress,shirt,toptee}/*.jpg`.

### 2. Preprocessing & Determinism
- Transform pipeline: `data_utils.py:92-106` (`targetpad_transform(target_ratio=1.25, dim=224)`):
  `TargetPad(1.25, 224)` -> `Resize(224, BICUBIC)` -> `CenterCrop(224)` -> `_convert_image_to_rgb` -> `ToTensor()` -> `Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))`.
- In `datasets.py:149-157` (`get_test_img`) and `:138-147` (`get_img`), images are transformed deterministically. Unlike PAIR, INTENT applies deterministic TargetPad to both query references and target gallery images at evaluation time.

### 3. Caption Preprocessing & Missing Dictionaries
- Captions normalized via `concat_text` (`datasets.py:117-119`):
  Calls `correct_text` (`:110-115`): lowercases string, replaces punctuation with spaces via `str.maketrans`, tokenizes, and replaces words matching keys in `correction_dict`. Joins captions with `" and "`.
- **Defect:** `captions/correction_dict_{dress,shirt,toptee}.json` are required by `datasets.py:98, 172` but are **not** present in the upstream repository, release archives, or official HuggingFace Hub.

### 4. Cache Mutation on Dataset Root
- In `datasets.py:37-52`, `FashionIQ.__init__` checks for `fashion_iq_data.json` inside `self.path`. If missing, it processes annotations and dumps:
  - `fashion_iq_data.json`
  - `test_queries_{dress,shirt,toptee}.pkl`
  - `test_targets_{dress,shirt,toptee}.pkl`
- These pickle files store preprocessed image tensors. On read-only dataset filesystems, initialization crashes immediately with `PermissionError`.

### 5. Preparation Contract Status
- `models.yaml:318` declares `preparation_contract: manual_ilearn_resized`.
- **Discrepancy:** `workbench/registry/preparation_contracts.yaml` contains **no** contract entry for `intent`.
- Running `python3 -m workbench.scripts.prepare_model --model intent` (`RUNTIME_VERIFIED`) reports:
  `[OK] intent: policy=NONE_REQUIRED deterministic=N/A`
  `notes: Standard FashionIQ layout required; no custom generated artifacts.`
- The declaration in `models.yaml` is inert.

---

## E. CHECKPOINT & NOISE CONDITIONS

### 1. Checkpoint Variants & Registry Declarations
`workbench/registry/models.yaml:338-367` registers three noise-trained checkpoint variants:
- `fiq_n02`: `filename: intent_fiq_0.2.pt`, `download_url: https://huggingface.co/iLearn-Lab/AAAI26-INTENT/resolve/main/intent_fiq_0.2.pt`, `checkpoint_training_noise_pct: 20`, `expected_size_bytes: 2722478522`, `expected_sha256: 2a504cadfd6b1c2486a2c508538b0160c5653f62c7cd11f4f3d8a9534dc26afc`, `status: NO_CLEAN_CHECKPOINT`.
- `fiq_n05`: `filename: intent_fiq_0.5.pt`, `download_url: https://huggingface.co/iLearn-Lab/AAAI26-INTENT/resolve/main/intent_fiq_0.5.pt`, `checkpoint_training_noise_pct: 50`, `expected_size_bytes: 2722478522`, `expected_sha256: 2d59ae002ecbf645a52c7e93bf8afd914e3061e2304413b97a0bf15ca43e39a9`, `status: NO_CLEAN_CHECKPOINT`.
- `fiq_n08`: `filename: intent_fiq_0.8.pt`, `download_url: https://huggingface.co/iLearn-Lab/AAAI26-INTENT/resolve/main/intent_fiq_0.8.pt`, `checkpoint_training_noise_pct: 80`, `expected_size_bytes: 2722478522`, `expected_sha256: 671bb217d3fdf32e6133fcb98eef4e80ebf831b5a203493c54507ef3c23b5cd2`, `status: NO_CLEAN_CHECKPOINT`.

Headline clean checkpoint availability: `strict_clean_checkpoint_available: false` (`models.yaml:335`).

### 2. Architecture & Serialization
- Architecture: BLIP-2 Q-Former composed network (`Blip2QformerCir` in `lavis/models/blip2_models/blip2_cir.py`).
  - Vision encoder: EVA-CLIP ViT-g/14 (`ln_vision` normalization).
  - Q-Former: BERT architecture initialized with 32 learnable query tokens. Text projection maps token 32 (`[CLS]`) to dimension $D=256$.
  - Category model policy: `shared` (single model evaluates dress, shirt, toptee).
- Serialization: Whole-model PyTorch serialization (`torch.save(model, ...)` at `train_INTENT.py:289, 339, 384`). Checkpoint loading in `cirr_sub_BLIP2.py:128` uses `torch.load(path)`.

### 3. Noise Regimes & Clean vs. Noisy Evaluation
- **Noise Injection Mechanism:**
  - Implemented in `datasets.py:63-92` (`FashionIQ.shuffle`). When `self.noise_ratio > 0`, $N \times \text{noise\_ratio}$ training triplets are shuffled (1/3 candidates, 1/3 captions, 1/3 targets). When `self.noise_ratio == 0`, `shuffle()` logs `"无噪声模式"` and returns immediately.
- **Evaluation is Strictly Clean:**
  - `FashionIQ.shuffle()` is invoked exclusively in `train_INTENT.py:101` for training triplets.
  - Validation queries and targets are loaded in `get_test_data` (`datasets.py:166-210`) directly from uncorrupted `cap.{cat}.val.json` and `split.{cat}.val.json`.
  - Evaluation inputs are never corrupted: `evaluation_noise_pct: 0` holds across all checkpoints.
- **Conflation Risk:**
  - The headline literature benchmark row (`intent_aaai26_table2_fiq_n00`, AVG `65.24`, R@10 `55.31`) corresponds to the **0% noise** training block.
  - Author HuggingFace repository publishes **no** 0% checkpoint.
  - Evaluating the 20% checkpoint (`intent_fiq_0.2.pt`) against the 0% reference row produces an apparent 0.88% metric deficit (64.36 vs 65.24), creating high risk of false-negative parity failure if noise conditions are conflated.
  - The workbench enforces this distinction in `paper_gate.py` by requiring matching `checkpoint_training_noise_pct`.

### 4. Checkpoint Mapping Discrepancy
- In `workbench/registry/models.yaml:347, 357, 367`, `checkpoint_mapping_status` is declared as `VERIFIED_METADATA`.
- In `workbench/registry/paper_benchmarks/papers.yaml:1303, 1353, 1403`, `checkpoint_mapping_status` is declared as `UNVERIFIED`.
- This discrepancy causes `paper_gate.py` to downgrade reproduction verdicts to `PASS_METRICS_ONLY_MAPPING_UNVERIFIED`.

---

## F. WORKBENCH INTEGRATION

### 1. Adapter Implementation
- `workbench/backend/adapters/models.py:69-71`:
  ```python
  class IntentAdapter(OfficialScriptAdapter):
      model_id = "intent"
      script = "test.py"
  ```
- `IntentAdapter` does not implement `command()`. It inherits `OfficialScriptAdapter.command()`, which executes:
  `raise NotImplementedError("upstream evaluation command requires audit before execution")`.

### 2. Readiness & Diagnostics
- Running `python3 -m workbench.scripts.pc_readiness --model intent` (`RUNTIME_VERIFIED`):
  ```
  intent                     BLOCKED_SOURCE
      source cannot run: no runnable evaluation path is registered for this model: the pinned evaluator is a callable with no CLI or checkpoint loader...
      checkpoint fiq_n02:
        - source cannot run: no runnable evaluation path is registered...
        - command not runnable: COMMAND_UNAUDITED
        - checkpoint missing: .../intent/intent_fiq_0.2.pt
      checkpoint fiq_n05:
        - source cannot run: no runnable evaluation path is registered...
        - command not runnable: COMMAND_UNAUDITED
        - checkpoint missing: .../intent/intent_fiq_0.5.pt
      checkpoint fiq_n08:
        - source cannot run: no runnable evaluation path is registered...
        - command not runnable: COMMAND_UNAUDITED
        - checkpoint missing: .../intent/intent_fiq_0.8.pt
  ```
- Running `python3 -m workbench.scripts.doctor --model intent` (`RUNTIME_VERIFIED`):
  `[OK] source:intent: source pin ready`
  `[BLOCKED] runtime:intent: command unavailable`
  `[BLOCKED] command:intent: COMMAND_AUDITED`

### 3. Checkpoint Selection
- Executing `select_checkpoints(model_by_id('intent'), policy='strict_clean')` (`RUNTIME_VERIFIED`):
  Returns `status='BLOCKED'`, `reason_code='BLOCKED_CHECKPOINT'`, `strict_clean_checkpoint_available=False`, `reason='no checkpoint is recorded as clean-trained under policy strict_clean: fiq_n02=20%, fiq_n05=50%, fiq_n08=80%...'`.
- Executing `select_checkpoints(model_by_id('intent'), policy='include_flagged_conditions')` (`RUNTIME_VERIFIED`):
  Returns `status='BLOCKED'`, `reason_code='BLOCKED_CHECKPOINT_SELECTION'`, `reason='more than one source-backed variant is registered and none is declared a bundle: fiq_n02, fiq_n05, fiq_n08...'`.

### 4. Metrics Extraction Parser
- `workbench/backend/metrics_extraction.py`: Contains no parser for `intent`.
- Calling `extract_aggregate_metrics("intent", stdout, stderr, return_code=0)` dispatches to default fall-through:
  Returns `extraction_status='AGGREGATE_PARSER_UNAVAILABLE'`, `parity_status='PARSER_UNAVAILABLE'`, `observed_metrics=None`.

### 5. Environment & Dependencies
- `models.yaml:308-314`: `python_env_var: WORKBENCH_INTENT_PYTHON`, `python: '3.9'`, `pytorch: '2.1.0'`, `torchvision: '0.16.0'`, `cuda: '12.6'`, packages: `[open-clip-torch==2.24.0, scikit-learn==1.3.2, transformers==4.25.0, salesforce-lavis==1.0.2, timm==0.9.16]`.
- Upstream pinned tree vendors its own `lavis/` directory containing modified modules (including `blip2_cir.py`), which shadow installed `salesforce-lavis` packages at runtime.

---

## G. OBSERVATION CAPABILITY

### 1. Score & Ranking Production Point
- In `workbench/third_party/INTENT/test.py`:
  - Line 141: `sims = np.matmul(all_queries, all_imgs)` (Query-gallery token contraction).
  - Line 142: `sims = sims.squeeze()`
  - Line 143: `sims = sims.max(-1)` (Max similarity over 32 tokens).
  - Line 150: `sims[i, test_targets_id.index(t['source_img_id'])] = -10e10` (Reference item exclusion).
  - Line 153 / 167: `nn_result = [np.argsort(-sims[i, :])[:50] for i in range(sims.shape[0])]` (Top-50 ranking indices).
- **Identity Mapping:**
  - Query index `i` maps 1:1 to annotation index in `captions/cap.{cat}.val.json`.
  - Reference image ASIN: `images[test_queries[i]['source_img_id']]` where `images` is loaded from `split.{cat}.val.json`.
  - Target ground-truth ASIN: `images[test_queries[i]['target_img_id']]`.
  - Retrieved gallery ASIN at rank $r$: `images[test_targets_id[nn_result[i][r]]]`.
  - Model input text: `test_queries[i]['mod']['str']` (preprocessed text). Raw captions recoverable from `captions/cap.{cat}.val.json[i]['captions']`.

### 2. Per-Query Export Feasibility & Status
- **Feasibility:** High. Observation interception at line 153/167 of `test.py` or via a dedicated replay observer can export `query_id`, `reference_asin`, `target_asin`, `ranked_target_asins[:50]`, and `scores[:50]`.
- **Current Blockers:** Execution blocked by absent CLI entrypoint, missing clean checkpoint, and unimplemented adapter command.
- **Export Axis Verdict:** **`QUERY_EXPORT_IMPLEMENTABLE`**.

---

## H. DEFECTS

### Confirmed Bugs (`CONFIRMED_BUG`)
1. **Unexecutable Evaluator Script & Training-Only Trainer**
   - **Path:** `workbench/third_party/INTENT/test.py:9` and `workbench/third_party/INTENT/train_INTENT.py:347-384`
   - **Precondition:** Running official FashionIQ evaluation from command line.
   - **Actual:** `test.py` defines `test(...)` with no CLI/`__main__`. `train_INTENT.py` has no evaluation-only flag or checkpoint loading argument and unconditionally runs training.
   - **Expected:** Dedicated CLI evaluation entrypoint capable of loading checkpoints and evaluating validation sets.
   - **Classification:** `CONFIRMED_BUG` (upstream).

2. **Dataset Directory Mismatch Between README and Code**
   - **Path:** `workbench/third_party/INTENT/README.md:135-155` vs `datasets.py:31`
   - **Precondition:** Organizing dataset according to official README documentation (`FashionIQ/{dress,shirt,toptee}`).
   - **Actual:** `datasets.py` fails with `FileNotFoundError` because line 31 hardcodes `self.image_dir = self.path + 'resized_image'`.
   - **Expected:** Code should match README documentation or README should specify `resized_image`.
   - **Classification:** `CONFIRMED_BUG` (upstream).

3. **Dataset Path Concatenation Missing Trailing Slash**
   - **Path:** `workbench/third_party/INTENT/datasets.py:31-33`
   - **Precondition:** Passing dataset directory path without a trailing slash (`/`).
   - **Actual:** Direct string concatenation (`self.path + 'resized_image'`) creates invalid paths (e.g., `/data/fashioniqresized_image`).
   - **Expected:** Robust path joining using `os.path.join(self.path, 'resized_image')`.
   - **Classification:** `CONFIRMED_BUG` (upstream).

4. **Hardcoded Cache Mutation on Dataset Root**
   - **Path:** `workbench/third_party/INTENT/datasets.py:37-52`
   - **Precondition:** Instantiating `FashionIQ` on a read-only dataset filesystem.
   - **Actual:** Code attempts to write `fashion_iq_data.json` and six `.pkl` files to `self.path`, raising `PermissionError`.
   - **Expected:** Read-only dataset handling or writing caches to a designated temporary/cache directory.
   - **Classification:** `CONFIRMED_BUG` (upstream).

5. **Missing `requirements.txt` in Pinned Upstream Repository**
   - **Path:** `workbench/third_party/INTENT/README.md:123`
   - **Precondition:** Setting up Python environment following official README (`pip install -r requirements.txt`).
   - **Actual:** `requirements.txt` does not exist in the repository.
   - **Expected:** Requirements file present to pin third-party package dependencies.
   - **Classification:** `CONFIRMED_BUG` (upstream).

6. **Unimplemented Adapter Command & Missing Metrics Parser**
   - **Path:** `workbench/backend/adapters/models.py:69-71` and `workbench/backend/metrics_extraction.py`
   - **Precondition:** Invoking `evaluate_models.py` or metrics extraction for `intent`.
   - **Actual:** `IntentAdapter` raises `NotImplementedError`; metrics extraction returns `AGGREGATE_PARSER_UNAVAILABLE`.
   - **Expected:** Working evaluation command adapter and stdout metric parser.
   - **Classification:** `CONFIRMED_BUG` (workbench integration).

7. **Missing Preparation Contract in Registry**
   - **Path:** `workbench/registry/models.yaml:318` vs `workbench/registry/preparation_contracts.yaml`
   - **Precondition:** Running `prepare_model --model intent`.
   - **Actual:** Script reports `policy=NONE_REQUIRED deterministic=N/A` because `preparation_contracts.yaml` has no entry for `intent`, despite `models.yaml` declaring `manual_ilearn_resized`.
   - **Expected:** Corresponding `intent_fashioniq` preparation contract entry in `preparation_contracts.yaml`.
   - **Classification:** `CONFIRMED_BUG` (workbench registry).

8. **Checkpoint Mapping Status Discrepancy Across Registries**
   - **Path:** `workbench/registry/models.yaml:347, 357, 367` vs `workbench/registry/paper_benchmarks/papers.yaml:1303, 1353, 1403`
   - **Precondition:** Checking checkpoint mapping status for `fiq_n02`, `fiq_n05`, `fiq_n08`.
   - **Actual:** `models.yaml` records `VERIFIED_METADATA` while `papers.yaml` records `UNVERIFIED`.
   - **Expected:** Synchronized mapping status across both registry files.
   - **Classification:** `CONFIRMED_BUG` (workbench registry).

### External Blockers (`EXTERNAL_BLOCKER`)
1. **Missing 0% Noise Checkpoint for Headline Supervised Benchmark:**
   - Author HuggingFace repository publishes weights only for 20%, 50%, and 80% noise. The headline supervised benchmark (Table 2 0% row: AVG 65.24) cannot be evaluated.
2. **Missing Upstream Correction Dictionaries:**
   - `captions/correction_dict_{dress,shirt,toptee}.json` are required by `datasets.py:98, 172` but omitted from the source repository and release artifacts.

### Rejected Findings (`REJECTED_FINDING`)
1. **Hypothesis: `np.matmul(all_queries, all_imgs)` fails with a tensor shape error.**
   - *Rejected.* An earlier note in `models.yaml:306` claimed a shape mismatch. Broadcasting rules in NumPy evaluate `(N_q, 1, 1, 256)` matmul `(N_g, 256, 32)` to `(N_q, N_g, 1, 32)`. After `.squeeze().max(-1)`, `sims` has shape `(N_q, N_g)`, matching the query-gallery matrix exactly.
2. **Hypothesis: INTENT uses Protocol C (`fashioniq_val_split`).**
   - *Rejected.* `datasets.py:26` defaults to `split='original-split'`, where `test_targets` includes all images from `split.{cat}.val.json`. In `test.py:150`, reference image `source_img_id` is masked out with `-10e10`. This is Protocol B (`fashioniq_full_gallery_ref_excluded`), not Protocol C.
3. **Hypothesis: INTENT suffers from stochastic data augmentations on reference images during evaluation.**
   - *Rejected.* In `datasets.py:149-158` (`get_test_img`) and `:138-147` (`get_img`), images are preprocessed with deterministic `targetpad_transform(1.25, 224)`. Reference images are not subjected to stochastic cropping at evaluation.

---

## I. EVIDENCE

1. **Commit SHA & Tree Cleanliness Verification:**
   ```bash
   git -C workbench/third_party/INTENT rev-parse HEAD
   # Output: 2a75fc6f57d1a036bf3167a7fe1d2cee072f51f3
   git -C workbench/third_party/INTENT ls-files | wc -l
   # Output: 1543
   ```
2. **Readiness Verification (`pc_readiness.py`):**
   ```bash
   python3 -m workbench.scripts.pc_readiness --model intent
   # Output:
   # intent                     BLOCKED_SOURCE
   #     source cannot run: no runnable evaluation path is registered for this model: the pinned evaluator is a callable with no CLI or checkpoint loader...
   # summary: {"BLOCKED_SOURCE": 1}
   ```
3. **Diagnostics Check (`doctor.py`):**
   ```bash
   python3 -m workbench.scripts.doctor --model intent
   # Output:
   # [OK] source:intent: source pin ready
   # [BLOCKED] runtime:intent: command unavailable
   # [BLOCKED] command:intent: COMMAND_AUDITED
   ```
4. **Preparation Contract Verification (`prepare_model.py`):**
   ```bash
   python3 -m workbench.scripts.prepare_model --model intent
   # Output:
   # [OK] intent: policy=NONE_REQUIRED deterministic=N/A
   #   notes: Standard FashionIQ layout required; no custom generated artifacts.
   ```
5. **Checkpoint Selection Policy Verification:**
   ```python
   # select_checkpoints(model_by_id('intent'), policy='strict_clean')
   # Output: CheckpointSelection(status='BLOCKED', reason_code='BLOCKED_CHECKPOINT', strict_clean_checkpoint_available=False)
   # select_checkpoints(model_by_id('intent'), policy='include_flagged_conditions')
   # Output: CheckpointSelection(status='BLOCKED', reason_code='BLOCKED_CHECKPOINT_SELECTION')
   ```
6. **Broadcasting & Arithmetic Consistency Verification:**
   ```python
   # Verified with Python 3 Decimal (ROUND_HALF_UP):
   # 0% row:   R@10 = (50.32+56.32+59.28)/3 = 55.3067 -> 55.31; R@50 = (72.10+74.93+78.45)/3 = 75.16; AVG = (55.31+75.16)/2 = 65.235 -> 65.24
   # 20% row:  R@10 = 54.2167 -> 54.22; R@50 = 74.4867 -> 74.49; AVG = 64.355 -> 64.36
   # 50% row:  R@10 = 52.5200 -> 52.52; R@50 = 73.3167 -> 73.32; AVG = 62.920 -> 62.92
   # 80% row:  R@10 = 48.5133 -> 48.51; R@50 = 69.6333 -> 69.63; AVG = 59.070 -> 59.07
   ```

---

## J. VERDICT

| Axis | Verdict | Evidence Summary |
|---|---|---|
| **SOURCE** | **`BLOCKED_SOURCE`** | Upstream repository lacks standalone CLI evaluation entrypoint, missing `requirements.txt`, missing correction dictionaries, and mutates dataset path with cache files. |
| **PROTOCOL** | **`PROTOCOL_VERIFIED`** | Source in `datasets.py:203-209` (`split='original-split'`) and `test.py:150` strictly implements Protocol B (`fashioniq_full_gallery_ref_excluded`: full val split gallery, reference excluded with `-10e10`). |
| **COMMAND** | **`COMMAND_BLOCKED`** | Pinned `test.py` lacks CLI/`__main__`; `train_INTENT.py` has no evaluation-only mode; `IntentAdapter` does not implement `command()`; `models.yaml` records `COMMAND_UNAUDITED` / `BLOCKED_SOURCE`. |
| **CHECKPOINT** | **`CHECKPOINT_BLOCKED`** | Author HuggingFace repository publishes weights only for 20%, 50%, and 80% noise. The headline 0% noise benchmark row has no published checkpoint (`NO_CLEAN_CHECKPOINT`). |
| **QUERY EXPORT** | **`QUERY_EXPORT_IMPLEMENTABLE`** | Top-50 rank production point identified at `test.py:153` (`nn_result`). Query ASINs, target ASINs, and captions map 1:1. Execution blocked by missing CLI and clean weights. |
| **PC VALIDATION** | **`BLOCKED_SOURCE`** | `pc_readiness.py` classifies `intent` as `BLOCKED_SOURCE`. Execution blocked on missing evaluation entrypoint, missing clean checkpoint, and unimplemented adapter command. |
