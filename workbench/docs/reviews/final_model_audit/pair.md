# Final Model Audit: PAIR

**Audit Date:** 2026-10-17  
**Model ID:** `pair`  
**Protocol:** Protocol C (`fashioniq_val_split`: pair-union gallery, reference excluded)  
**Evaluation Standard:** `STATIC_AUDIT` throughout unless explicitly marked `RUNTIME_VERIFIED`. No GPU inference was executed. All code paths, ASTs, and CLI/API checks were executed in read-only mode against HEAD (`5726d5f`) on branch `research/cir-failure-workbench`.

---

## A. MODEL IDENTITY

- **Model ID:** `pair` (`workbench/registry/models.yaml:477`)
- **Method Name:** `PAIR` (`workbench/registry/models.yaml:486`)
- **Paper Title:** "PAIR: Complementarity-guided Disentanglement for Composed Image Retrieval" (`workbench/registry/models.yaml:491`, `workbench/registry/paper_benchmarks/papers.yaml:1170`)
- **Publication Venue & Year:** ICASSP 2025 (IEEE International Conference on Acoustics, Speech and Signal Processing, 06–11 April 2025, Hyderabad, India; `publication_year: 2025` at `models.yaml:488`)
- **Paper URL:** `https://ieeexplore.ieee.org/document/10888153` (`DOI: 10.1109/ICASSP49660.2025.10888153`)
  - Status: Publisher PDF paywalled (`HTTP 420`, 0 bytes from `https://ieeexplore.ieee.org/iel8/10887540/10887541/10888153.pdf`). Abstract and metadata verified via Crossref and Semantic Scholar (`paperId: 1ba36b31679aa640e49e3bdcc3e1489c3ae97e61`, DBLP `conf/icassp/FuLCWSHN25`).
- **Source Repository URL:** `https://github.com/iLearn-Lab/ICASSP25-PAIR` (`models.yaml:493`; canonical redirect from `https://github.com/ZhihFu/PAIR`)
- **Pinned Commit SHA:** `bfb0b98cdb4c44f6b9190baaa84d783a77dd888f` (`models.yaml:495`)
  - Verification: `git -C workbench/third_party/PAIR rev-parse HEAD` outputs `bfb0b98cdb4c44f6b9190baaa84d783a77dd888f` (`RUNTIME_VERIFIED`). Exactly matches registry.
  - Upstream History: Shallow clone (`--is-shallow-repository` = true, 1 commit locally). All 21 remote commits on `main` inspected via GitHub Trees API (`4fa4de5622d7` to `bfb0b98cdb4c`).
- **Paper Benchmark Reference Row:** `pair_fiq_clean_val_split` (`workbench/registry/paper_benchmarks/papers.yaml:1167-1235`).
  - Claimed Benchmark Numbers (FashionIQ validation set):
    - Category Dresses: R@10 = `46.78`, R@50 = `70.93`
    - Category Shirts: R@10 = `52.60`, R@50 = `73.80`
    - Category Tops&Tees: R@10 = `58.91`, R@50 = `78.81`
    - Macro Averages: R@10 = `52.76`, R@50 = `74.51`
    - Mean Recall: `63.64` (`WORKBENCH_DERIVED`, formula `(r10 + r50) / 2`)
  - Verification Source: Table 2 image in pinned repo (`workbench/third_party/PAIR/assets/fiqshoes.png`, referenced by `README.md:75-76`), and project page figure (`https://zhihfu.github.io/PAIR.github.io/static/images/Fiq.png`). Numbers are internally consistent: macro averages equal the arithmetic means of the three category metrics (`(46.78 + 52.60 + 58.91)/3 = 52.7633 -> 52.76`; `(70.93 + 73.80 + 78.81)/3 = 74.5133 -> 74.51`).

---

## B. SOURCE COVERAGE

### Pinned Tree Inventory (`workbench/third_party/PAIR`)
The pinned repository contains 21 git-tracked files:
1. `LICENSE` (Apache-2.0 license declaration)
2. `README.md` (Project description, environment commands, CIRR test command)
3. `assets/cirr.png` (CIRR benchmark comparison table image)
4. `assets/fiqshoes.png` (Table 2 comparison on FashionIQ and Shoes validation sets)
5. `assets/pair.png` (Architecture diagram)
6. `assets/logo/*.png` (9 presentation logo images: `airknow-logo.png`, `consep-logo.png`, `encoder-logo.png`, `habit-logo.png`, `hud-logo.png`, `intent-logo.png`, `offset-logo.png`, `pair-logo.png`, `retrack-logo.png`)
7. `cirr_test_submission.py` (CLI entrypoint for generating CIRR test submission JSONs)
8. `data/datasets.py` (Dataset loaders for `FashionIQ`, `Shoes`, `CIRR`)
9. `models/MHTransformer.py` (Multi-head self/cross attention and transformer layers)
10. `models/model.py` (PAIR architecture: CLIP feature extractors and disentanglement modules)
11. `test.py` (Evaluation functions `test` for FashionIQ/Shoes, `test_cirr_valset` for CIRR)
12. `train.py` (Argument parsing, dataloader initialization, model initialization)
13. `utils/utils.py` (Logging and argument helpers)

### Reachable Evaluation Dependency Graph
The reachable path for FashionIQ evaluation requires:
`test.py::test` -> `data/datasets.py::FashionIQ` -> `models/model.py::PAIR` -> `models/MHTransformer.py` + `MHTransformer_rev.py` (MISSING!) + `open_clip_pytorch_model.bin` (ViT-B/32 backbone).

- **Inspected Files:**
  - `test.py`: Inspected lines 1–84 (`test()` for FashionIQ) and lines 85–225 (`test_cirr_valset()`).
  - `data/datasets.py`: Inspected lines 22–169 (`FashionIQ` class, `train_init_process`, `get_test_data`, `get_img`, caption cleaning).
  - `models/model.py`: Inspected lines 1–170 (`PAIR` model initialization, `visual_out`, `text_out`, `extract_retrieval_compose`, `extract_retrieval_target`).
  - `models/MHTransformer.py`: Inspected lines 1–153 (`MultiHeadAttention`, `MultiHeadAttentionRev`, `TransformerLayer`, `Transformer`).
  - `train.py`: Inspected lines 1–154 (`ArgumentParser`, `load_dataset`, `create_model_and_optimizer`, missing execution loop).
  - `cirr_test_submission.py`: Inspected lines 91–106 (`torch.load` whole-model instantiation).
  - `README.md`: Inspected lines 1–230 (environment requirements, script usage instructions).
- **Excluded Files & Justification:**
  - `cirr_test_submission.py` (lines 12–90): CIRR-specific metric formatting and submission JSON emission; unreachable on FashionIQ path.
  - `data/datasets.py` (lines 170–451): `Shoes` and `CIRR` dataset loaders; unreachable on FashionIQ path.
  - `assets/` (all images except `fiqshoes.png`): Documentation graphics and presentation logos.

---

## C. NATIVE EVALUATION

### 1. Evaluator Entrypoint & CLI Interface
- Function: `test(params, model, testset, category)` defined at `workbench/third_party/PAIR/test.py:6`.
- **Defect:** `test.py` has **no** `if __name__ == '__main__':` block, **no** CLI parser (`argparse`), and **no** invocation wrapper.
- `train.py:128-154` contains an `ArgumentParser` and sets up seeds, directories, and models, but **halts abruptly at line 154** (`logging.info("Starting train for {} epoch(s)".format(args.num_epochs))`) without calling `train()` or `test()`.
- Consequently, the pinned source provides no runnable evaluator command.

### 2. Dataset Loading & Query/Gallery Construction
- Implemented in `workbench/third_party/PAIR/data/datasets.py:22-168` (`class FashionIQ`).
- Per category (`dress`, `shirt`, `toptee`), `get_test_data(name)` (`:117-167`) reads:
  1. `image_splits/split.{cat}.val.json` (`:118`) -> list of candidate image IDs (`images`).
  2. `captions/cap.{cat}.val.json` (`:120`) -> query triplet annotations (`ref_captions`).
  3. `captions/correction_dict_{cat}.json` (`:122`) -> vocabulary correction mapping (`correction_dict`).
- **Query Construction:**
  - `source_img_id = images.index(candidate)` (`:135`)
  - `target_img_id = images.index(target)` (`:137`)
  - `source_img_data = get_img(name + '_' + candidate, stage=0)` (`:138`)
  - `target_img_data = get_img(name + '_' + target, stage=0)` (`:140`)
  - `mod = {'str': mod_str}` (`:142`)
- **Gallery Construction (Protocol C Cohort):**
  - When `split == 'val-split'` (`:148-159`), gallery IDs `test_targets_id` are built as the first-seen ordered union of query source and target IDs:
    ```python
    for i in test_queries:
        if i['source_img_id'] not in test_targets_id:
            test_targets_id.append(i['source_img_id'])
        if i['target_img_id'] not in test_targets_id:
            test_targets_id.append(i['target_img_id'])
    ```
  - For each `i` in `test_targets_id`, the gallery entry is constructed via:
    `out['target_img_data'] = self.get_img(name + '_' + images[i], stage=1)` (`:158`).
  - Deduped gallery cohort sizes (measured on canonical FashionIQ split):
    - `dress`: 2,628 images (from split of 3,817)
    - `shirt`: 3,089 images (from split of 6,346)
    - `toptee`: 2,902 images (from split of 5,373)

### 3. Preprocessing Asymmetry & Non-Determinism
- Built via OpenCLIP ViT-B/32:
  `open_clip.create_model_and_transforms('ViT-B-32', pretrained='./open_clip_pytorch_model.bin')` (`train.py:65-66`, `cirr_test_submission.py:92-93`).
- Transform list is `[preprocess_train, preprocess_val]`:
  - `stage=0` (`preprocess_train`): `RandomResizedCrop(224, scale=(0.9, 1.0), interpolation=BICUBIC)` -> `ToTensor` -> `Normalize(CLIP mean/std)`.
  - `stage=1` (`preprocess_val`): `Resize(224, interpolation=BICUBIC)` -> `CenterCrop(224)` -> `ToTensor` -> `Normalize(CLIP mean/std)`.
- **Defect:** In `get_test_data`:
  - Query source (reference) image uses `stage=0` (`data/datasets.py:138`).
  - Target gallery image uses `stage=1` (`data/datasets.py:158`).
  - Reference images are subjected to stochastic random-resized crops during evaluation!
  - `test.py` sets no random seeds, resulting in run-to-run metric variance unless deterministic seeding is explicitly added by an external harness.

### 4. Text Preprocessing & Captions
- `concat_text(captions, correction_dict)` (`data/datasets.py:84-86`):
  Calls `correct_text(text, correction_dict)` (`:77-82`):
  - Lowercases string.
  - Translates `string.punctuation` to spaces via `str.maketrans`.
  - Splits on whitespace and replaces words present in `correction_dict`.
  - Concatenates the two modifier sentences with `" and "`: `f"{cap0} and {cap1}"`.

### 5. Scoring, Reference Exclusion, Ranking & Metrics
- Query features extracted via `model.extract_retrieval_compose(imgs, mods)` (`test.py:33`).
- Gallery features extracted via `model.extract_retrieval_target(imgs)` (`test.py:49`).
- L2-normalization applied in-place to all query and gallery feature vectors (`test.py:55-58`).
- Similarity matrix computed via matrix dot product: `sims = all_queries.dot(all_imgs.T)` (`test.py:62`).
- **Reference Exclusion:**
  `sims[i, test_targets_id.index(t['source_img_id'])] = -10e10` (`test.py:68`).
  Reference index is masked to `-10e10` prior to sorting.
- **Ranking:**
  `nn_result = [np.argsort(-sims[i, :])[:50] for i in range(sims.shape[0])]` (`test.py:71`).
- **Recall@K:**
  Evaluated for `k ∈ [1, 10, 50]` (`test.py:75-81`):
  Query `i` counts as a hit if `test_targets_id.index(test_queries[i]['target_img_id']) in nns[:k]`.
  Returns `out = [('{}_r1'.format(name), r1), ('{}_r10'.format(name), r10), ('{}_r50'.format(name), r50)]`.
- **Macro Aggregation:**
  Native `test.py` does not compute macro averages across categories or mean of R@10/R@50. Upstream outputs per-category tuples only.

---

## D. PREPARATION

### 1. Dataset Layout & Paths
- Registry layout: `fashioniq_layout: fashioniq_ilearn_resized` (`models.yaml:498`).
- Image format: `fashioniq_image_extensions: [".jpg"]` (`models.yaml:499`).
- Evaluator reads:
  - `data/datasets.py:27-29`: `self.image_dir = self.path + 'resized_image'`, `self.split_dir = self.path + 'image_splits'`, `self.caption_dir = self.path + 'captions'`. **Note:** `self.path` is concatenated directly with strings; path must end with a trailing `/`.
  - Image paths: `os.path.join(self.image_dir, image_name.split('_')[0], image_name.split('_')[1] + ".jpg")` (`:107`).
  - Expects images organized as: `resized_image/{dress,shirt,toptee}/*.jpg`.

### 2. Preparation Contract Status
- Registry declares: `preparation_contract: manual_ilearn_resized` (`models.yaml:489`), `preparation_id: pair_fashioniq` (`models.yaml:490`).
- Contract definition in `workbench/registry/preparation_contracts.yaml:302-338`:
  - `source_family: PAIR`
  - `automation_policy: MANUAL_REQUIRED`
  - `deterministic_status: DETERMINISTIC_AT_EVALUATION`
  - Raw inputs: Train and val captions and splits for dress, shirt, toptee.
  - Generated artifacts: `resized_image/{dress,shirt,toptee}` using `start_kit/resize_images.py` at 256px.
  - Notes: "The PAIR README documents no resize step; this contract makes the declared manual_ilearn_resized contract effective instead of inert... The caption correction dictionaries the pinned data/datasets.py requires are operator-supplied and unverified."
- Verification via CLI (`RUNTIME_VERIFIED`):
  `python3 -m workbench.scripts.prepare_model --model pair` reports `[OK] pair: policy=MANUAL_REQUIRED deterministic=DETERMINISTIC_AT_EVALUATION`.
- Unresolved data dependencies:
  `captions/correction_dict_{dress,shirt,toptee}.json` are required by `data/datasets.py:65, 122` but are **not** present in the upstream PAIR repository or author Hub.

---

## E. CHECKPOINT

### 1. Checkpoint Variants & Registry Declarations
- `workbench/registry/models.yaml:512-537` declares two checkpoint variants:
  - `pair_b1`: `filename: pair-B1.pt`, `download_url: https://huggingface.co/ZhihengFu/PAIR/resolve/main/pair-B1.pt`, `expected_size_bytes: 694958565`, `status: CHECKPOINT_MAPPING_UNVERIFIED`, `checkpoint_mapping_status: UNVERIFIED`.
  - `pair_b2`: `filename: pair-B2.pt`, `download_url: https://huggingface.co/ZhihengFu/PAIR/resolve/main/pair-B2.pt`, `expected_size_bytes: 694872371`, `status: CHECKPOINT_MAPPING_UNVERIFIED`, `checkpoint_mapping_status: UNVERIFIED`.
- Headline reproduction status: `reproduction_status: BLOCKED_CHECKPOINT_MAPPING_UNVERIFIED` (`models.yaml:506`).
- Clean checkpoint available: `strict_clean_checkpoint_available: false` (`models.yaml:510`).

### 2. Critical Investigation: B1 vs B2 Checkpoint Identity
- **Pinned Source:** Contains zero occurrences of `pair-B1`, `pair-B2`, `pair_b1`, or `pair_b2`.
- **Pinned README:** Mentions only `checkpoints/PAIR_CIRR.pt` (`README.md:224, 227`) and `--model_dir "./checkpoints/PAIR_CIRR"` (`:202`). No mention of FashionIQ weights.
- **Upstream Commit History:** All 21 remote commits on `main` inspected; zero commits mention checkpoint filenames or release tags.
- **Author HuggingFace Repository (`iLearn-Lab/ICASSP25-PAIR` / `ZhihengFu/PAIR`):**
  - Contains exactly two checkpoint blobs: `pair-B1.pt` (694,958,565 bytes, LFS sha256 prefix `c4d861ec`) and `pair-B2.pt` (694,872,371 bytes, LFS sha256 prefix `74346fd7`).
  - The model card README states weights are evaluated on three datasets: CIRR, FashionIQ, Shoes.
  - The model card assigns **no** dataset, category, training condition, or benchmark to either file.
  - Two checkpoint files for three distinct evaluation benchmarks makes a 1:1 dataset mapping impossible to infer by convention.
- **Registry Discrepancy in `papers.yaml`:**
  - `workbench/registry/paper_benchmarks/papers.yaml:1176` assigned `checkpoint_id: pair_b1` while keeping `checkpoint_mapping_status: UNVERIFIED` (`:1178`).
  - Tested against `paper_gate.evaluate_gate` (`RUNTIME_VERIFIED`):
    - Evaluating a run with `checkpoint_id: pair_b1` yields `PASS_METRICS_ONLY_MAPPING_UNVERIFIED`.
    - Evaluating a run with `checkpoint_id: pair_b2` yields `BLOCKED_CHECKPOINT_MISMATCH` ("checkpoint_id mismatch: reference='pair_b1' run='pair_b2'").
- **Verdict on Checkpoint Identity:**
  Evidence does NOT decide whether `pair-B1.pt` or `pair-B2.pt` is FashionIQ. Any selection of B1 over B2 is arbitrary. Both checkpoints remain `UNVERIFIED` and checkpoint status is **`CHECKPOINT_BLOCKED`**.

---

## F. WORKBENCH INTEGRATION

### 1. Adapter Implementation
- `workbench/backend/adapters/models.py:49-51`:
  ```python
  class PairAdapter(OfficialScriptAdapter):
      model_id = "pair"
      script = "test.py"
  ```
- `PairAdapter` does not implement `command()`. It inherits `OfficialScriptAdapter.command()` (`:24-25`), which executes:
  `raise NotImplementedError("upstream evaluation command requires audit before execution")`.
- When `build_command` is invoked (`RUNTIME_VERIFIED`):
  1. `validate_request()` checks checkpoint file existence -> raises `FileNotFoundError: checkpoint missing`.
  2. If checkpoint file is present, `validate_request()` checks `checkpoint_mapping_status == "VERIFIED_METADATA"` -> raises `WorkbenchError: checkpoint_mapping_unverified`.
  3. If bypassed, `self.command()` raises `NotImplementedError`.

### 2. Preflight & Readiness
- In `workbench/registry/models.yaml:509`: `command_status: COMMAND_BLOCKED_SOURCE_ENTRYPOINT`.
- CLI check (`RUNTIME_VERIFIED`):
  `python3 -m workbench.scripts.pc_readiness --model pair` outputs:
  ```
  pair                       BLOCKED_IMPLEMENTATION
      command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT
      checkpoint pair_b1:
        - command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT
        - checkpoint missing: .../pair/pair-B1.pt
        - checkpoint mapping unresolved
      checkpoint pair_b2:
        - command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT
        - checkpoint missing: .../pair/pair-B2.pt
        - checkpoint mapping unresolved
  ```
- CLI check (`RUNTIME_VERIFIED`):
  `python3 -m workbench.scripts.doctor --model pair` reports:
  `[OK] source:pair: source pin ready`
  `[BLOCKED] runtime:pair: command unavailable`
  `[BLOCKED] command:pair: COMMAND_BLOCKED_SOURCE_ENTRYPOINT`

### 3. Checkpoint Selection
- Tested via `workbench.backend.checkpoint_selection.select_checkpoints` (`RUNTIME_VERIFIED`):
  - Under `strict_clean`: returns `status='BLOCKED'`, `reason_code='BLOCKED_CHECKPOINT'`, `reason='no checkpoint variant records a training condition...'`, `strict_clean_checkpoint_available=False`.
  - Under `include_flagged_conditions`: returns `status='BLOCKED'`, `reason_code='BLOCKED_CHECKPOINT'`, `reason='checkpoint mapping unverified for pair_b1, pair_b2...'`.

### 4. Metrics Extraction Parser
- `workbench/backend/metrics_extraction.py:415`:
  `extract_aggregate_metrics("pair", stdout, stderr, return_code=0)` dispatches to fall-through:
  Returns `extraction_status='AGGREGATE_PARSER_UNAVAILABLE'`, `parity_status='PARSER_UNAVAILABLE'`, `observed_metrics=None`.
  The workbench has no parser implemented for PAIR's native output format (`[('{cat}_r{k}', pct)]`).

### 5. Environment & Dependencies
- `models.yaml:478-485`: `python_env_var: WORKBENCH_PAIR_PYTHON`, `python: '3.8.10'`, `pytorch: '2.0.0'`, `cuda: '11.8'`, `packages: []`.
- Pinned tree missing `requirements.txt`: `README.md:119` directs `pip install -r requirements.txt`, but `requirements.txt` does not exist in the pinned repository (`git ls-files` shows 21 files, no requirements file).

---

## G. OBSERVATION CAPABILITY

### 1. Score & Ranking Production Point
- In `workbench/third_party/PAIR/test.py`:
  - Line 62: `sims = all_queries.dot(all_imgs.T)` (query-gallery cosine similarity matrix).
  - Line 68: `sims[i, test_targets_id.index(t['source_img_id'])] = -10e10` (reference item exclusion).
  - Line 71: `nn_result = [np.argsort(-sims[i, :])[:50] for i in range(sims.shape[0])]` (Top-50 ranking indices).
- **Identity Mapping:**
  - `test_queries[i]['source_img_id']`: Integer index of reference image in `split.{cat}.val.json`.
  - `test_queries[i]['target_img_id']`: Integer index of target ground-truth image in `split.{cat}.val.json`.
  - `test_queries[i]['mod']['str']`: Processed text caption string (`"cap0 and cap1"`).
  - `test_targets_id`: List of integer image indices corresponding to gallery items.
  - To recover canonical FashionIQ image string names: `images[test_targets_id[rank_idx]]` where `images` is loaded from `split.{cat}.val.json`.
  - Original raw captions are 1:1 aligned with `test_queries[i]` from `captions/cap.{cat}.val.json`.

### 2. Per-Query Export Feasibility & Status
- **Feasibility:** High in architecture; an observation wrapper can intercept `nn_result` at line 71 of `test.py` and serialize `query_id`, `reference_image_id`, `target_image_id`, `ranked_target_ids[:50]`, and `scores[:50]`.
- **Current Blockers:**
  - Native evaluation cannot be executed due to missing `MHTransformer_rev.py`, absent CLI entrypoint, unmapped checkpoints, and unimplemented adapter command.
- **Export Axis Verdict:** **`QUERY_EXPORT_IMPLEMENTABLE`**.

---

## H. DEFECTS

### Confirmed Bugs (`CONFIRMED_BUG`)
1. **Missing Upstream Module `MHTransformer_rev.py`**
   - **Path:** `workbench/third_party/PAIR/models/model.py:13, 47`
   - **Precondition:** Any import of `models/model.py` or attempt to instantiate `PAIR`.
   - **Actual:** `import MHTransformer_rev` raises `ModuleNotFoundError: No module named 'MHTransformer_rev'`.
   - **Expected:** Module defining `Transformer` with reversed cross-attention should be present in `models/`.
   - **Classification:** `CONFIRMED_BUG` (upstream) / `EXTERNAL_BLOCKER`.

2. **Unexecutable Evaluator Script & Dead Training Loop**
   - **Path:** `workbench/third_party/PAIR/test.py:6` and `workbench/third_party/PAIR/train.py:154`
   - **Precondition:** Running official evaluation from command line.
   - **Actual:** `test.py` defines `test(...)` with no CLI/`__main__`. `train.py` argument parser stops at line 154 without training or evaluating.
   - **Expected:** Standalone CLI entrypoint parsing arguments and evaluating checkpoints.
   - **Classification:** `CONFIRMED_BUG` (upstream).

3. **Missing `requirements.txt` in Pinned Upstream Repository**
   - **Path:** `workbench/third_party/PAIR/README.md:119`
   - **Precondition:** Setting up Python environment following official README (`pip install -r requirements.txt`).
   - **Actual:** `requirements.txt` does not exist in the repository.
   - **Expected:** Requirements file present to pin third-party package dependencies (including `open_clip_torch`, `einops`).
   - **Classification:** `CONFIRMED_BUG` (upstream).

4. **Stochastic Preprocessing on Query Reference Images During Evaluation**
   - **Path:** `workbench/third_party/PAIR/data/datasets.py:138, 140`
   - **Precondition:** Query evaluation feature extraction in `get_test_data()`.
   - **Actual:** Reference images are transformed with `stage=0` (`RandomResizedCrop(224, scale=(0.9, 1.0))`) while gallery images use `stage=1` (`Resize + CenterCrop`). No random seed is initialized in `test.py`.
   - **Expected:** Deterministic evaluation preprocessing on reference images.
   - **Classification:** `CONFIRMED_BUG` (upstream).

5. **Arbitrary Reference Checkpoint Assignment in Benchmark Registry**
   - **Path:** `workbench/registry/paper_benchmarks/papers.yaml:1176`
   - **Precondition:** Running gate evaluation against `pair_b2`.
   - **Actual:** Reference row records `checkpoint_id: pair_b1` despite `checkpoint_mapping_status: UNVERIFIED`, causing `BLOCKED_CHECKPOINT_MISMATCH` when evaluating `pair_b2`.
   - **Expected:** Reference row should reflect unverified mapping without enforcing `pair_b1` over `pair_b2`.
   - **Classification:** `CONFIRMED_BUG` (workbench registry).

6. **Unimplemented Adapter Command & Missing Metrics Parser**
   - **Path:** `workbench/backend/adapters/models.py:49-51` and `workbench/backend/metrics_extraction.py:415`
   - **Precondition:** Invoking `evaluate_models.py` or `pc_readiness` for `pair`.
   - **Actual:** Adapter raises `NotImplementedError`; metrics extraction returns `AGGREGATE_PARSER_UNAVAILABLE`.
   - **Expected:** Functioning evaluation command adapter and stdout metric parser.
   - **Classification:** `CONFIRMED_BUG` (workbench integration).

### External Blockers (`EXTERNAL_BLOCKER`)
1. **Undecidable Checkpoint Mapping (B1 vs B2)**
   - Author HuggingFace provides `pair-B1.pt` and `pair-B2.pt` without dataset or category identification across CIRR, FashionIQ, and Shoes.
2. **Missing Upstream Correction Dictionaries**
   - `captions/correction_dict_{dress,shirt,toptee}.json` are required by `data/datasets.py:65, 122` but omitted from the source repository and release artifacts.

### Rejected Findings (`REJECTED_FINDING`)
1. **Hypothesis: `pair_b1` is definitively the FashionIQ checkpoint.**
   - Rejected. No evidence in source code, commit history, README, model card, or paper text associates B1 with FashionIQ.
2. **Hypothesis: `MHTransformer.py` can be imported as a drop-in substitute for `MHTransformer_rev.py`.**
   - Rejected. `models/model.py:45-47` instantiates both `MHTransformer.Transformer` and `MHTransformer_rev.Transformer`. Line 13 explicitly imports `MHTransformer_rev`. Substituting `MHTransformer` alters network architecture (reversed attention vs normal attention) and fails state dict loading.

---

## I. EVIDENCE

1. **Commit SHA Verification:**
   ```bash
   git -C workbench/third_party/PAIR rev-parse HEAD
   # Output: bfb0b98cdb4c44f6b9190baaa84d783a77dd888f
   ```
2. **Check Missing `MHTransformer_rev`:**
   ```bash
   find workbench/third_party/PAIR -name "*MHTransformer_rev*"
   # Output: (no files found)
   ```
3. **Check Model Import Failure:**
   ```python
   # python3 -c "import sys; sys.path.insert(0, 'workbench/third_party/PAIR/models'); import model"
   # Output: ModuleNotFoundError: No module named 'MHTransformer_rev'
   ```
4. **Preflight Readiness Check:**
   ```bash
   python3 -m workbench.scripts.pc_readiness --model pair
   # Output: BLOCKED_IMPLEMENTATION, command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT
   ```
5. **Doctor Check:**
   ```bash
   python3 -m workbench.scripts.doctor --model pair
   # Output: [BLOCKED] runtime:pair: command unavailable, [BLOCKED] command:pair: COMMAND_BLOCKED_SOURCE_ENTRYPOINT
   ```
6. **Checkpoint Selection Check:**
   ```python
   # select_checkpoints(model_by_id('pair'), policy='strict_clean')
   # Output: CheckpointSelection(status='BLOCKED', reason_code='BLOCKED_CHECKPOINT', strict_clean_checkpoint_available=False)
   ```
7. **Paper Gate Comparability Check:**
   ```python
   # evaluate_gate(ref, metrics, run_meta={'checkpoint_id': 'pair_b2', ...}, real_execution=True)
   # Output: BLOCKED_CHECKPOINT_MISMATCH ("checkpoint_id mismatch: reference='pair_b1' run='pair_b2'")
   ```

---

## J. VERDICT

| Axis | Verdict | Evidence Summary |
|---|---|---|
| **SOURCE** | **`BLOCKED_SOURCE`** | Upstream repository is missing `models/MHTransformer_rev.py` (line 13 of `model.py`) and `requirements.txt`. Model cannot be imported. |
| **PROTOCOL** | **`PROTOCOL_VERIFIED`** | Code in `data/datasets.py:148-159` and `test.py:68` implements Protocol C (`fashioniq_val_split`: pair-union gallery, reference excluded with `-10e10`). Matches workbench specification. |
| **COMMAND** | **`COMMAND_BLOCKED`** | `test.py` lacks CLI/`__main__`; `train.py` halts at line 154; `PairAdapter.command` raises `NotImplementedError`; `models.yaml` records `COMMAND_BLOCKED_SOURCE_ENTRYPOINT`. |
| **CHECKPOINT** | **`CHECKPOINT_BLOCKED`** | `pair-B1.pt` and `pair-B2.pt` exist on author Hub, but cannot be mapped to FashionIQ from any code, documentation, or commit history. |
| **QUERY EXPORT** | **`QUERY_EXPORT_IMPLEMENTABLE`** | Top-50 rank production point identified at `test.py:71` (`nn_result`). Recoverable image and query IDs mapped. Execution currently blocked by source/command defects. |
| **PC VALIDATION** | **`BLOCKED_EXTERNAL_DEPENDENCY`** | Blocked on missing upstream file `MHTransformer_rev.py`, unmapped author checkpoints, and unaudited evaluation command. |
