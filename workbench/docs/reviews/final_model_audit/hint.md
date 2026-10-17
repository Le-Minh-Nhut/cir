# Final Model Audit: hint

## A. MODEL IDENTITY

- **Model ID**: `hint` (`workbench/registry/models.yaml:368`)
- **Method Name**: `HINT` (`workbench/registry/models.yaml:377`)
- **Research Generation**: `recent` (`workbench/registry/models.yaml:378`)
- **Publication Year**: `2026` (`workbench/registry/models.yaml:379`)
- **Paper Title**: `HINT: Composed Image Retrieval with Dual-Path Compositional Contextualized Network` (`workbench/registry/models.yaml:381`)
- **Paper URL**: `https://arxiv.org/abs/2603.26341` (`workbench/registry/models.yaml:382`)
- **Paper Publication Details**: Published on arXiv on 2026-03-27 (`arXiv:2603.26341v1`); authors Mingyu Zhang, Zixu Li, Zhiwei Chen, Zhiheng Fu, Xiaowei Zhu, Jiajia Nie, Yinwei Wei, Yupeng Hu (accepted to ICASSP 2026).
- **Upstream Repository URL**: `https://github.com/iLearn-Lab/ICASSP26-HINT` (`workbench/registry/models.yaml:383`)
- **Upstream Default Branch**: `main` (`workbench/registry/models.yaml:384`)
- **Upstream Commit SHA**: `bec50b6c8c19111893b617979502b948d1cea5b2` (`workbench/registry/models.yaml:385`)
- **Observed Source Pin**: Verified via `git -C workbench/third_party/HINT rev-parse HEAD` returning `bec50b6c8c19111893b617979502b948d1cea5b2`. Match is exact (STATIC_AUDIT).
- **Checkout State**: Single squashed snapshot commit (`git log --oneline --all` -> `bec50b6 Create LICENSE`). Working tree contains only untracked Python bytecode (`__pycache__/` dirs). Cleanliness passes `runtime.source_clean_and_pinned`.
- **URL Discrepancy Note**: The printed arXiv abstract page links `https://github.com/zh-mingyu/HINT`, while the registry pins `https://github.com/iLearn-Lab/ICASSP26-HINT`. Both remote repository URLs serve the exact same git commit SHA `bec50b6c8c19111893b617979502b948d1cea5b2` (not a divergence).
- **Paper Benchmark Row**: `reference_id: hint_fiq_clean_val_split` in `workbench/registry/paper_benchmarks/papers.yaml:1042-1103`.
  - **Source Excerpt**: Table 1 ("Performance comparison on FashionIQ and CIRR relative to R@k(%)"), page 4:
    `HINT (Ours) | 51.80 73.13 | 60.27 78.48 | 63.37 82.19 | 58.48 77.93`
  - **Category Metrics**: Dress R@10=51.80, R@50=73.13; Shirt R@10=60.27, R@50=78.48; Toptee R@10=63.37, R@50=82.19.
  - **Reported Aggregates**: FashionIQ Avg R@10=58.48, R@50=77.93. (Category macro: `mean(51.80, 60.27, 63.37) = 58.48`, `mean(73.13, 78.48, 82.19) = 77.9333 -> 77.93`).
  - **Derived Metric**: Overall Mean = 68.21 (`(58.48 + 77.93) / 2 = 68.205 -> 68.21`), explicitly marked `WORKBENCH_DERIVED`.

---

## B. SOURCE COVERAGE

### Tree Inventory
The pinned checkout at `workbench/third_party/HINT/` contains 1,233 tracked files:
- **Root evaluation and training scripts**:
  - `data_utils.py` (107 lines): Image pad transforms (`squarepad_transform`, `targetpad_transform`, `TargetPad`).
  - `utils.py` (68 lines): Logging, running averages, JSON helper functions.
  - `datasets.py` (311 lines): Dataset definitions (`FashionIQ`, `CIRR`, text cleaning `correct_text`, `concat_text`).
  - `train.py` (279 lines): Main training script, optimization loop, and validation call.
  - `test.py` (190 lines): Core evaluation functions (`test` for FashionIQ/shoes/lasco/birds, `test_cirr_valset`).
  - `cirr_test_submission.py` (135 lines): Script generating predictions on CIRR test set for benchmark submission.
  - `README.md` (390 lines), `LICENSE` (21 lines), `.idea/` (7 files).
  - `assets/` (10 files): Architecture diagrams, paper result screenshots, logos.
- **Vendored Framework (`lavis/`)**:
  - Complete snapshot of Salesforce LAVIS framework (1,213 tracked files).
  - HINT custom model implementation: `lavis/models/blip2_models/HINT.py` (595 lines) declaring `class HINT(Blip2Base)` with dual-path contextualized network modules.

### Reachable Evaluation Dependency Graph
When evaluating FashionIQ natively:
`train.py:198-202` (during training loop)
  └── `test.py:test(params, model, testset, category, txt_processors)` (line 11)
        ├── `lavis/models/blip2_models/HINT.py:extract_retrieval_compose` (line 401: query feature extraction)
        ├── `lavis/models/blip2_models/HINT.py:extract_retrieval_target` (line 436: candidate feature extraction)
        ├── `txt_processors["eval"]` -> LAVIS `blip_caption` processor
        ├── `testset` (`datasets.FashionIQ` in `datasets.py:24`)
        │     ├── `FashionIQ.get_test_data` (line 131: loads splits, captions, correction dicts)
        │     ├── `FashionIQ.get_test_img` (line 114: loads resized JPEG images)
        │     └── `data_utils.py:targetpad_transform` (line 92: aspect-preserving padding)
        └── `np.matmul(all_queries, all_imgs).squeeze().max(-1)` -> reference masking -> `np.argsort` -> Recall@1,10,50.

### Inspected vs Excluded
- **Inspected**: All top-level Python modules (`data_utils.py`, `utils.py`, `datasets.py`, `train.py`, `test.py`, `cirr_test_submission.py`), `lavis/models/blip2_models/HINT.py`, and `lavis/configs/models/blip2/blip2_pretrain.yaml`.
- **Excluded**: Unrelated LAVIS tasks and model families (ALBEF, ALPRO, BEATs, BLIP-Diffusion, InstructBLIP, Uniformer semantic segmentation annotators, PNP-VQA, timesformer). These are vendored framework remnants completely unreachable by the HINT architecture.

### README Claims vs Executable Source
1. **README FashionIQ Training Command**:
   `README.md:201-207` instructs:
   ```bash
   python train.py --dataset fashioniq --fashioniq_path "/path/to/FashionIQ/" --model_dir "./checkpoints/fashioniq_hint" --batch_size 128 --num_epochs 10 --lr 2e-5
   ```
   **Contradiction / Bug**: Executing this exact command raises `TypeError: __init__() got an unexpected keyword argument 'noise_ratio'` at `train.py:89` because `train.py` passes `noise_ratio=args.noise_ratio` into `datasets.FashionIQ`, but `FashionIQ.__init__` accepts only `(self, path, transform=None)`.
2. **README Testing Command**:
   `README.md:226-235` documents testing *only* for CIRR: `python src/cirr_test_submission.py checkpoints/cirr_hint/`. The path `src/` does not exist in the repository (script is in the root directory). Crucially, the README documents **no command whatsoever** for evaluating FashionIQ.

---

## C. NATIVE EVALUATION

### 1. Pair-Union Validation Gallery (Proof of Protocol C)
The validation gallery is constructed in `workbench/third_party/HINT/datasets.py:131-166` (`FashionIQ.get_test_data`):
```python
140        for idx in range(len(ref_captions)):
141            caption = ref_captions[idx]
...
143            candidate = caption['candidate']
144            target = caption['target']
145            out = {}
146            out['source_img_id'] = images.index(candidate)
147            out['source_img_data'] = self.get_test_img(name + '_' + candidate)
148            out['target_img_id'] = images.index(target)
...
152            test_queries.append(out)
153        
154        test_targets_id = []
155        for i in test_queries:
156            if i['source_img_id'] not in test_targets_id:
157                test_targets_id.append(i['source_img_id'])
158            if i['target_img_id'] not in test_targets_id:
159                test_targets_id.append(i['target_img_id'])
160        test_targets = []
161        for i in test_targets_id:
162            out = {}
163            out['target_img_id'] = i
164            out['target_img_data'] = self.get_test_img(name + '_' + images[i])      
165            test_targets.append(out)
```
- Line 140 iterates over `cap.{cat}.val.json`.
- `images` is the list loaded from `split.{cat}.val.json`.
- Lines 154-159 append `source_img_id` (the query reference) if not already seen, then `target_img_id` (the target) if not already seen.
- `test_targets` is the deduplicated, ordered union of all query references and query targets.
- Gallery sizes on standard FashionIQ val splits:
  - **Dress**: 2,628 images (split total is 3,817)
  - **Shirt**: 3,089 images (split total is 6,346)
  - **Toptee**: 2,902 images (split total is 5,373)
- This definitively proves **Protocol C (`fashioniq_val_split`)**.

### 2. Reference Exclusion (Masking Line)
In `workbench/third_party/HINT/test.py:79-84`:
```python
79        test_targets_id = []
80        for i in test_targets:
81            test_targets_id.append(i['target_img_id'])
82        for i, t in enumerate(test_queries):
83            sims[i, test_targets_id.index(t['source_img_id'])] = -10e10
```
- Line 83 locates the column corresponding to query `i`'s reference image (`t['source_img_id']`) in `test_targets_id` and overwrites its similarity score with `-10e10` ($-10^{11}$).
- This ensures the reference image can never enter the top-$K$ rankings. Reference exclusion is verified.

### 3. Caption Preprocessing
- First stage (`datasets.py:75-84`):
  ```python
  def correct_text(self, text, correction_dict):
      trans = str.maketrans({key: ' ' for key in string.punctuation})
      tokens = str(text).lower().translate(trans).strip().split()
      text = " ".join([correction_dict.get(word) if word in correction_dict else word for word in tokens])
      return text

  def concat_text(self, captions, correction_dict):
      text = "{} and {}".format(self.correct_text(captions[0], correction_dict), self.correct_text(captions[1], correction_dict))
      return text
  ```
  Text is lowercased, punctuation replaced with spaces, tokenized by whitespace, corrected via `correction_dict_{category}.json`, single-space joined, and concatenated as `"{cap0} and {cap1}"`.
- Second stage (`test.py:44`):
  `mods = [txt_processors["eval"](caption) for caption in mods]`
  Passes the string into LAVIS's `blip_caption` processor (BertTokenizer truncation, max length 32).

### 4. Category-Specific Evaluation
- Evaluated category by category in `train.py:198-202` looping over `['dress', 'shirt', 'toptee']`.
- `test.py:11` accepts `category` and switches query/target subsets (`testset.test_queries_dress`, etc.).

### 5. Similarity Scoring & Multi-Token Max-Pooling
- `test.py:74-76`:
  ```python
  sims = np.matmul(all_queries, all_imgs)
  sims = sims.squeeze()
  sims = sims.max(-1)
  ```
- `all_queries` shape: $(N_{\text{queries}}, 1, 1, D)$ from `HINT.extract_retrieval_compose` (`F.normalize(..., dim=-1).unsqueeze(1).unsqueeze(1)`).
- `all_imgs` shape: $(N_{\text{targets}}, D, 32)$ from `HINT.extract_retrieval_target` (`F.normalize(..., dim=-1).permute(0, 2, 1)`).
- `np.matmul` broadcasts to $(N_{\text{queries}}, N_{\text{targets}}, 1, 32)$.
- `.squeeze().max(-1)` takes the maximum dot product across the 32 Q-Former output tokens, yielding $(N_{\text{queries}}, N_{\text{targets}})$.

### 6. Sorting, Top-K, and Recall@K
- `test.py:86` / `test.py:100`:
  `nn_result = [np.argsort(-sims[i, :])[:50] for i in range(sims.shape[0])]`
  Descending argsort on negative similarities, sliced to top 50. Note: `np.argsort` uses quicksort by default, which is unstable on score ties.
- `test.py:101-107`:
  ```python
  for k in [1, 10, 50]:
      r = 0.0
      for i, nns in enumerate(nn_result):
          if test_targets_id.index(test_queries[i]['target_img_id']) in nns[:k]:
              r += 1
      r = 100 * r / len(nn_result)
      out += [('{}_r{}'.format(name, k), r)]
  ```
  Returns percentage of queries where the target's index in `test_targets_id` is within `nns[:k]`.

### 7. Macro Aggregation
- Source code in `test.py` does not compute macro averages across categories or overall mean; it returns only per-category tuples `[('dress_r1', ...), ('dress_r10', ...), ('dress_r50', ...)]`.
- `train.py:201` sums R@10 across categories solely for checkpoint selection.
- Paper Table 1 macro averages are reproduced by taking the unweighted mean of Dress, Shirt, and Toptee R@10 (58.48) and R@50 (77.93).

---

## D. PREPARATION

### 1. Layout Contract
- Declared layout in `models.yaml:388`: `fashioniq_layout: fashioniq_ilearn_resized`.
- Layout paths verified by `workbench/backend/fashioniq_layout.py:25-34` (`ilearn_resized_paths`):
  `captions/`, `image_splits/`, `resized_image/`, `resized_image/{dress,shirt,toptee}`, `captions/cap.{cat}.val.json`, `captions/correction_dict_{cat}.json`, `image_splits/split.{cat}.val.json`.
- Current workspace status: `captions/` and `image_splits/` exist. `resized_image/` subdirectories and `correction_dict_*.json` files are missing from `data/fashionIQ_dataset`.

### 2. Train vs Val Layout Under-Specification Bug
- In `datasets.py:33-38`:
  ```python
  if not os.path.exists(os.path.join(self.path, 'fashion_iq_data.json')):
      self.fashioniq_data = []
      self.train_init_process()
  ```
- If the cache file `fashion_iq_data.json` is absent, the constructor calls `train_init_process()`, which attempts to open `captions/cap.{cat}.train.json` (`datasets.py:61`).
- The declared `fashioniq_ilearn_resized` layout checks only `val` split captions. On any clean dataset root without pre-cached `.pkl`/`.json` files, initialization immediately crashes with `FileNotFoundError: cap.dress.train.json`.

### 3. Correction Dictionary Dependencies
- Required files: `captions/correction_dict_dress.json`, `captions/correction_dict_shirt.json`, `captions/correction_dict_toptee.json` (`datasets.py:63,137`).
- Upstream availability: **Zero files bundled**. `git -C workbench/third_party/HINT ls-files | grep correction` returns 0 hits.
- Provenance: Author HuggingFace repository does not distribute them. Upstream README does not provide a download link or generation script.
- Runtime impact: Unconditional crash (`FileNotFoundError`) at dataset instantiation if absent.

### 4. Image Resizing Contract
- Required paths: `resized_image/{dress,shirt,toptee}/{image_id}.jpg` (`datasets.py:104,115`).
- Resolution: Pinned code specifies no image size. Sibling iLearn-Lab models assume 256x256 JPEGs.
- Trailing slash requirement: `datasets.py:28-31` builds paths by string concatenation (`self.image_dir = self.path + 'resized_image'`), requiring `--fashioniq_path` to end with a directory separator (`/`).

---

## E. CHECKPOINT

### 1. Checkpoint Identity & Registry Integrity
- Registry variant (`models.yaml:400-414`):
  - `checkpoint_id: fashioniq`
  - `filename: fashioniq.pt`
  - `download_url: https://huggingface.co/iLearn-Lab/HINT/resolve/main/fashioniq.pt`
  - `expected_size_bytes: 2724451182`
  - `expected_sha256: bf9c72d42206b667ed586758c15c648970df821857e4067f26155a114a6517d4`
  - `status: URL_VERIFIED`
  - `checkpoint_mapping_status: VERIFIED_METADATA`
- Independent Hugging Face API verification (`https://huggingface.co/api/models/iLearn-Lab/HINT?blobs=true`):
  - File `fashioniq.pt` exists with exact size 2,724,451,182 bytes and LFS SHA256 `bf9c72d42206b667ed586758c15c648970df821857e4067f26155a114a6517d4`.

### 2. Architecture & Bundle Structure
- Monolithic single-checkpoint model shared across all three FashionIQ categories.
- Model architecture: BLIP-2 Q-Former with dual-path contextualized projection (`lavis/models/blip2_models/HINT.py`).
- Serialization format: Whole pickled PyTorch model (`torch.save(model, ...)` at `train.py:204`).
- Serialization registry gap: `checkpoint_serialization` is omitted in `models.yaml` for `hint`, while present for 8 sibling models.

### 3. Checkpoint Loader & Evaluation Entrypoint Gap
- **No FashionIQ loader exists in upstream source**:
  - The only `torch.load` call in the entire repository is `cirr_test_submission.py:115`, which is CIRR-specific (`datasets.CIRR`).
  - `test.py:test` expects an in-memory `model` instance and has no checkpoint loading logic.
  - `train.py:59` declares `--restore_file`, but this CLI parameter is never accessed anywhere in the codebase (0 grep hits outside the argparse declaration).
  - Therefore, `fashioniq.pt` cannot be loaded by any native FashionIQ script in the pinned checkout.

---

## F. WORKBENCH INTEGRATION

### 1. Registry & Preparation Contract
- `workbench/registry/models.yaml:380` sets `preparation_contract: manual_ilearn_resized`.
- **Defect**: No contract named `manual_ilearn_resized` exists in `workbench/registry/preparation_contracts.yaml`.
- Executed `prepare_model.py --model hint --json` returns:
  `"contract_type": "standard_fashioniq"`, `"preparation_id": null`, `"fully_prepared": true`.
  The script silently falls back to `standard_fashioniq` and claims the model is prepared, completely ignoring the missing resized images and correction dictionaries.

### 2. Adapter Implementation
- `workbench/backend/adapters/models.py:44-46`:
  ```python
  class HintAdapter(OfficialScriptAdapter):
      model_id = "hint"
      script = "test.py"
  ```
- **Defect**: `HintAdapter` does not implement `command()`. It inherits `OfficialScriptAdapter.command` (`adapters/models.py:24-25`), which unconditionally raises `NotImplementedError("upstream evaluation command requires audit before execution")`.
- Furthermore, `test.py` contains no `if __name__ == '__main__':` block. Executing `python test.py` does nothing and exits 0 immediately.

### 3. Readiness Probe & Evaluation Runner
- Executed `workbench/scripts/pc_readiness.py --model hint --json`:
  - `"state": "BLOCKED_IMPLEMENTATION"`
  - `"reason": "command not runnable: COMMAND_UNAUDITED"`
  - Blockers: `command not runnable: COMMAND_UNAUDITED`, `checkpoint missing: .../artifacts/checkpoints/hint/fashioniq.pt`.
- Executed `workbench/scripts/evaluate_models.py --model hint --dry-run`:
  `[SKIP] SKIPPED: adapter command not audited`.

### 4. Metrics Extraction
- In `workbench/backend/metrics_extraction.py:402-422`:
  - Parsers exist only for `limn`, `csmcir`, `combiner_rn50x4_noft`, `clip4cir_rn50x4_fullft`, and `sprc`.
  - For `hint`, `extract_aggregate_metrics` unconditionally returns:
    `extraction_status="AGGREGATE_PARSER_UNAVAILABLE"`, `parity_status="PARSER_UNAVAILABLE"`.

### 5. Benchmark Gate Verification
- Tested `paper_gate.evaluate_gate` with reference `hint_fiq_clean_val_split` in Python:
  - Exact paper metrics (r10=58.48, r50=77.93, mean=68.21, plus 6 category cells): `status: PASS_WITHIN_5_PERCENT`, `passed: True`, 9 comparisons at 0.00% relative error.
  - Omitted category metrics (only aggregate metrics supplied): `status: PASS_WITH_UNCHECKED_CATEGORIES`, `passed: False`. Gate strictly enforces category coverage when the reference publishes category numbers.

### 6. API and Frontend Integration
- API `GET /api/models` exposes `hint` with:
  `adapter_command_ready: False`, `runnable: False`, `blocking_reasons: ['command not runnable: COMMAND_UNAUDITED', 'checkpoint missing: ...']`.

---

## G. OBSERVATION CAPABILITY

### 1. Score & Ranking Production Point
- In `workbench/third_party/HINT/test.py:86,100`:
  `nn_result = [np.argsort(-sims[i, :])[:50] for i in range(sims.shape[0])]`
  - `sims[i]` contains similarities for query `i` across all gallery images in `test_targets`.
  - `nn_result[i]` contains the 50 candidate indices in descending order of similarity.

### 2. Per-Query Ranking & Identity Reconstruction
- **Query Index**: Index `i` in `test_queries` maps 1:1 to annotation index `i` in `cap.{category}.val.json`.
- **Query Reference Image**: `images[t['source_img_id']]` recovers canonical reference image ID string.
- **Query Target Image**: `images[t['target_img_id']]` recovers canonical target image ID string.
- **Retrieved Candidate at Rank $k$**:
  - `cand_pos = nn_result[i][k]`
  - `cand_img_idx = test_targets_id[cand_pos]`
  - `cand_image_id = images[cand_img_idx]`
  - `score = float(sims[i, cand_pos])`
- **Target Rank**: Can be computed exactly via full argsort or linear search over `sims[i]`.
- **Captions**:
  - `test_queries[i]['mod']['str']` contains corrected, lowercased string.
  - Raw captions are not retained in `test_queries`, but can be recovered directly from `cap.{category}.val.json[i]['captions']`.

### 3. Observer Attachment Feasibility & Blockers
- **Feasibility**: Fully implementable via a source-faithful replay runner (identical to `workbench/replay/limn.py`). The ranking logic is clean NumPy array operations.
- **Blockers**:
  1. No standalone runner exists in the workbench (no `workbench/replay/hint.py`).
  2. Pinned upstream evaluator requires CUDA (`.cuda()` called unconditionally at `test.py:43,60` and `train.py:122`).
  3. Upstream checkout lacks checkpoint loader for FashionIQ.
  4. Mandatory correction dictionaries and resized images must be staged.

---

## H. DEFECTS

| Defect ID | Severity | Classification | File / Path | Lines | Summary |
|---|---|---|---|---|---|
| `HINT-DEF-01` | P0 | `CONFIRMED_BUG` | `workbench/third_party/HINT/train.py` | 86–89 | `train.py` passes `noise_ratio=args.noise_ratio` to `datasets.FashionIQ`, raising `TypeError` immediately upon dataset instantiation. |
| `HINT-DEF-02` | P1 | `CONFIRMED_BUG` | `workbench/backend/adapters/models.py` | 44–46 | `HintAdapter` specifies `script = "test.py"` (which has no `__main__`) and does not implement `command()`, raising `NotImplementedError`. |
| `HINT-DEF-03` | P1 | `CONFIRMED_BUG` | `workbench/registry/models.yaml`, `preparation_contracts.yaml` | `models.yaml:380` | `models.yaml` declares `preparation_contract: manual_ilearn_resized`, which does not exist in `preparation_contracts.yaml`, causing `prepare_model.py` to silently report `fully_prepared: true`. |
| `HINT-DEF-04` | P1 | `EXTERNAL_BLOCKER` | `workbench/third_party/HINT/` | whole repo | Upstream HINT repository provides no evaluation script or checkpoint loader for FashionIQ weights (`fashioniq.pt`). |
| `HINT-DEF-05` | P1 | `EXTERNAL_BLOCKER` | `workbench/third_party/HINT/datasets.py` | 63, 137 | Pinned code unconditionally requires `correction_dict_{cat}.json`, which are neither bundled in the repository nor published on Hugging Face. |
| `HINT-DEF-06` | P2 | `CONFIRMED_BUG` | `workbench/backend/metrics_extraction.py` | 402–422 | `metrics_extraction.py` has no parser for HINT stdout, unconditionally returning `AGGREGATE_PARSER_UNAVAILABLE`. |
| `HINT-DEF-07` | P2 | `CONFIRMED_BUG` | `workbench/backend/fashioniq_layout.py` | 25–34 | `ilearn_resized_paths` omits `cap.{cat}.train.json`, causing clean cache-miss initialization to crash with `FileNotFoundError` at `datasets.py:61`. |
| `HINT-DEF-08` | P3 | `CONFIRMED_BUG` | `workbench/registry/models.yaml` | 368–414 | `checkpoint_serialization: whole_model` is missing from `models.yaml:hint`, unlike 8 sibling models. |

---

## I. EVIDENCE

1. **`train.py` Kwarg Crash Reproduction**:
   ```bash
   .venv/bin/python -c "
   from workbench.third_party.HINT import datasets
   try:
       datasets.FashionIQ(path='/tmp', noise_ratio=0)
   except TypeError as e:
       print('Observed TypeError:', e)
   "
   ```
   *Observed Output*:
   `Observed TypeError: FashionIQ.__init__() got an unexpected keyword argument 'noise_ratio'`

2. **Adapter NotImplementedError & Missing Main**:
   ```bash
   .venv/bin/python workbench/third_party/HINT/test.py
   # Exits 0 with no stdout/stderr (no __main__)
   .venv/bin/python workbench/scripts/evaluate_models.py --model hint --dry-run
   ```
   *Observed Output*:
   `[RUN] hint / fashioniq`
   `[SKIP] SKIPPED: adapter command not audited`

3. **Readiness Probe Output**:
   ```bash
   .venv/bin/python workbench/scripts/pc_readiness.py --model hint --json
   ```
   *Observed Output*:
   `"state": "BLOCKED_IMPLEMENTATION"`, `"reason": "command not runnable: COMMAND_UNAUDITED"`.

4. **Preparation Contract Fallback**:
   ```bash
   .venv/bin/python workbench/scripts/prepare_model.py --model hint --json
   ```
   *Observed Output*:
   `"contract_type": "standard_fashioniq"`, `"preparation_id": null`, `"fully_prepared": true`.

5. **Benchmark Gate Enforcement on Incomplete Category Reporting**:
   ```python
   from workbench.backend import benchmark_references, paper_gate
   ref = benchmark_references.references_for_model('hint')[0]
   v = paper_gate.evaluate_gate(
       reference=ref,
       reproduced_metrics={'r10': 58.48, 'r50': 77.93, 'mean': 68.21},
       reproduced_category_metrics=None,
       run_meta={'checkpoint_mapping_verified': True, 'protocol_id': 'fashioniq_val_split'},
       data_kind='experiment',
       real_execution=True
   )
   print('Status:', v.status, 'Passed:', v.passed)
   ```
   *Observed Output*:
   `Status: PASS_WITH_UNCHECKED_CATEGORIES Passed: False`

---

## J. VERDICT

- **SOURCE_COVERAGE**: `SOURCE_AUDITED`
- **PROTOCOL**: `PROTOCOL_VERIFIED`
- **COMMAND**: `COMMAND_BLOCKED`
- **CHECKPOINT**: `CHECKPOINT_BLOCKED`
- **QUERY_EXPORT**: `QUERY_EXPORT_IMPLEMENTABLE`
- **RELEASE_READINESS**: `BLOCKED_EXTERNAL_DEPENDENCY`
