# hint

Independent adversarial reproducibility audit of registered model `hint` (HINT), ICASSP 2026. READ-ONLY: no file under `/home/heheboiz/data/cir` was written, no GPU job, install, or checkpoint download was run. Scratch went to `/tmp/audit-final/hint/`. Every claim below is **STATIC_AUDIT** unless explicitly labelled **RUNTIME_VERIFIED**. There is exactly one RUNTIME_VERIFIED class of evidence in this document: read-only executions of the workbench's own loader/gate/readiness code and read-only HTTP fetches of the paper/Hub — never a model inference.

## MODEL ID

`hint` — `method_name: HINT` (`workbench/registry/models.yaml:373`), `source_dir: HINT` (`:383`), `source_available: true` (`:382`). Registry block spans `:364-405`. STATIC_AUDIT.

## PAPER TITLE

`HINT: Composed Image Retrieval with Dual-Path Compositional Contextualized Network` (`models.yaml:377`).

- Fetched `https://arxiv.org/abs/2603.26341` (HTTP 200): `<title>` = `[2603.26341] HINT: Composed Image Retrieval with Dual-path Compositional Contextualized Network`; `citation_date` = `2026/03/27`; `citation_author` = Zhang Mingyu, Li Zixu, Chen Zhiwei, Fu Zhiheng, Zhu Xiaowei, Nie Jiajia, Wei Yinwei, Hu Yupeng; `citation_arxiv_id` = `2603.26341`.
- PDF: `curl -sS -L https://arxiv.org/pdf/2603.26341v1` → `HTTP 200 2681236`; `sha256sum` → `fbe97798e89ef21d7f30a895a151b94ff062f708c5060ac397e494adf05f2373`. This is byte-identical to the hash recorded in the extracted bundle (`extracted/hint_icassp26.json:209`, `pdf_sha256`).
- **Contradiction found (in workbench metadata, not in the paper):** the PDF carries both a `/Pages 5` entry and a page tree that pdftoppm/pdftotext address as 6 pages (`pdfinfo` → `Pages: 5`; `file` → `6 page(s)`; `pdftotext -f 6 -l 6` → `Wrong page range given: the first page (6) can not be after the last page (5)`). The extracted bundle asserts `"PDF document, version 1.7, 6 page(s)"` (`extracted/hint_icassp26.json:201`) and `"6-page camera-ready format"` (`:217`). Prefer `pdfinfo`'s `/Pages 5`. Cosmetic; the table content is unaffected. P3.

## PINNED SOURCE COMMIT

- Registry pin `bec50b6c8c19111893b617979502b948d1cea5b2` (`models.yaml:381`).
- Observed: `git -C workbench/third_party/HINT rev-parse HEAD` → `bec50b6c8c19111893b617979502b948d1cea5b2`. **MATCH**.
- `git log --oneline --all` → one commit, `bec50b6 Create LICENSE`; `git ls-files | wc -l` → 1233. A single-commit snapshot, so no branch/tag history exists to check `upstream_default_branch: main` (`:380`) against locally.
- `git status --porcelain` → only untracked `__pycache__/` trees. `runtime.source_clean_and_pinned` executed in-process returns `(True, 'bec50b6c8c19111893b617979502b948d1cea5b2', None)`, so the checkout passes the pin gate.
- `git ls-remote https://github.com/iLearn-Lab/ICASSP26-HINT` → `bec50b6c8c…HEAD`, `…refs/heads/main`. `git ls-remote https://github.com/zh-mingyu/HINT` → the **same** SHA. The paper's printed code URL (`https://github.com/zh-mingyu/HINT`, paper p.1; `hint.txt:46`) and the registry's URL are the same repository. The prior round's "documented discrepancy" (paper personal-account URL vs lab URL) is therefore **refuted as a discrepancy** — both hosts serve one object. P3 (doc wording).

## CHECKPOINT IDENTITY

Registry variant (`models.yaml:395-405`): `checkpoint_id: fashioniq`, `filename: fashioniq.pt`, `download_url: https://huggingface.co/iLearn-Lab/HINT/resolve/main/fashioniq.pt`, `source_type: author_huggingface`, `checkpoint_training_noise_pct: 0`, `evaluation_noise_pct: 0`, `expected_size_bytes: 2724451182`, `expected_sha256: null`, `status: URL_VERIFIED`, `checkpoint_mapping_status: VERIFIED_METADATA`.

Independently re-derived from the Hub (executed):
`curl https://huggingface.co/api/models/iLearn-Lab/HINT?blobs=true` → repo `sha 19863b230c6e4e66e299e192423e9614295a7d22`, `private False`, `lastModified 2026-04-11T06:36:55Z`; siblings `.gitattributes 1519`, `README.md 6124`, `cirr.pt 2724451182` LFS `fca4657d94672571fa1dcb7615a1e8fc27a0651dd5e9ebc190943bb0acb25afd`, `fashioniq.pt 2724451182` LFS `bf9c72d42206b667ed586758c15c648970df821857e4067f26155a114a6517d4`.

1. `expected_size_bytes` is exactly the Hub's `fashioniq.pt` size. VERIFIED.
2. `expected_sha256: null` (`:403`) while the author publishes a byte-level LFS `sha256`. `checkpoint_availability` executed → `"official_sha256_known": false`. Consequence: no download could ever be verified. P2. Recommended: record `expected_sha256: bf9c72d4…`.
3. `checkpoint_mapping_status: VERIFIED_METADATA` is the registry's strongest mapping value, and `select_checkpoints(model, policy='strict_clean')` (executed) → `SELECTED ('fashioniq',)`, `strict_clean_checkpoint_available True`. But `VERIFIED_METADATA` per the registry's own header means *file metadata only*; it is not evidence the file is the paper's FashionIQ row. `registry.py:712` maps that value to `mapping_verified: true` for UI/consumption.
4. **No pinned loader consumes `fashioniq.pt` for FashionIQ.** The only `torch.load` in the tree is `cirr_test_submission.py:115`, inside a CIRR-only script (`cirr_test_submission.py:108` constructs `datasets.CIRR(path="./data/cirr_data/CIRR/")`); `test.py:test(params, model, testset, category, txt_processors)` takes an already-built model (`test.py:11`); `train.py:59` declares `--restore_file` and the string appears nowhere else in the repo (grep: one hit, the declaration). The HF card's own "Run Testing / Evaluation" step is `python src/cirr_test_submission.py checkpoints/` for CIRR and never shows a FashionIQ route; it says only that the weights are "designed to be evaluated seamlessly using the official HINT GitHub repository" (`hfcard.md:73,95-101`), and there is no such route in the pinned commit. P1 (identity of the registered artifact is not established by any executable path).
5. `checkpoint_serialization` is absent for `hint` while present for 8 other models (`encoder/clvc_net/dcnet/combiner_rn50x4_noft/clip4cir_rn50x4_fullft/tgcir/sprc/limn`, mapping `state_dict|whole_model|run_directory`). The source-loadable form is whole-model (`train.py:204,279` `torch.save(model, …)`; `cirr_test_submission.py:115`). P3.

## EVALUATION PROTOCOL

- Registry: `native_protocol: fashioniq_val_split` (`:386`), `literature_split_label: val` (`:387`), `supported_protocols: [fashioniq_val_split]` (`:388`). Protocol table entry `models.yaml:16-19` (label "FashionIQ - Val Split (Pair-Union Gallery, Reference Excluded)", `literature_split_label: val`, `evaluation_noise_pct: 0`); label agreement enforced at `registry.py:50`.
- Workbench cohort definition: `FASHIONIQ_PROTOCOLS["fashioniq_val_split"] = FashionIQProtocol(gallery="pair_union", reference_excluded=True, aliases=("fashioniq_val",))` (`src/evaluation/fashioniq.py:176-179`); dispatch `build_fashioniq_gallery` (`:204-219`).
- Source-side: the only FashionIQ evaluator is `test.py:test`, reached from `train.py:199` inside `train_and_evaluate` (`train.py:173`), which loops `['dress','shirt','toptee']` (`train.py:198`). `test.py` has no `__main__` and no `ArgumentParser` (grep: zero hits; file is 190 lines, both defs are `load_obj:8`, `test:11`, `test_cirr_valset:112`).

VERDICT: the registry's cohort assignment matches what the pinned evaluator *computes* (proved in GALLERY CONTRACT), but there is **no evaluation-only entry point upstream and none in the workbench**. Executed: `python3 workbench/scripts/evaluate_models.py --model hint --dry-run --dataset-root data/fashionIQ_dataset` → `[RUN] hint / fashioniq` / `[SKIP] SKIPPED: adapter command not audited`; `--list` → `[OK] hint / fashioniq: protocol=fashioniq_val_split command=not audited`. P1.

## DATASET CONTRACT

- Registry: `fashioniq_layout: fashioniq_ilearn_resized` (`:384`), `fashioniq_image_extensions: [".jpg"]` (`:385`), `preparation_contract: manual_ilearn_resized` (`:376`).
- Declared layout resolved in-process (`paths_for_layout(root, "fashioniq_ilearn_resized")` → `fashioniq_layout.py:68-69` → `:25-34`): `captions`, `image_splits`, `resized_image`, `resized_image/{dress,shirt,toptee}`, `cap.{cat}.val.json`, `correction_dict_{cat}.json`, `split.{cat}.val.json`. Observed existence under the configured root `/home/heheboiz/data/cir/data/fashionIQ_dataset` (explicit `FASHIONIQ_ROOT`; `workbench/config/workbench.env` does not exist, so the operator config resolves it explicitly — `operator_config.py:87`): the two directory roots, the three val captions and the three val splits **exist**; `resized_image`, its three subdirectories and the three `correction_dict_*.json` are **MISSING**.
- Source requirements traced from the evaluated constructor: `datasets.py:28-31` builds `image_dir = path + 'resized_image'`, `split_dir = path + 'image_splits'`, `caption_dir = path + 'captions'` by string concatenation (so `--fashioniq_path` must end with a separator). Fresh-constructor branch (`:33-47`) calls `train_init_process()` and `get_test_data` for all three categories and writes six pickles; `train_init_process` (`:59-73`) opens `cap.{cat}.train.json` (`:61`) and `correction_dict_{cat}.json` (`:63`); `get_test_data` (`:131-165`) opens `split.{cat}.val.json` (`:133`), `cap.{cat}.val.json` (`:135`) and `correction_dict_{cat}.json` (`:137`). Cache branch (`:49-57`) loads `fashion_iq_data.json` plus six pickles instead.
- **Gap A (P2/P3):** the declared layout omits `cap.{cat}.train.json`, so on a cache-miss root the pinned code raises `FileNotFoundError` at `datasets.py:61` before any recall is computed. The training-inclusive helper exists (`fashioniq_layout.py:37-41`) and is used only by other models. The cache branch is invisible to preflight entirely.
- **Gap B (P1 for any number):** the three `correction_dict_{cat}.json` files are mandatory (`datasets.py:63,137`) and have no provenance anywhere: `registry/assets/cluster_b2_encoder_family.yaml:66` `classification: SOURCE_EVIDENCE_MISSING`, `:75 acquisition_status: UNVERIFIED`, `:76` "Not bundled in pinned HINT checkout; no official download URL or generation script provided in repository." They are **not** in `preparation_contracts.yaml` (0 `hint` occurrences) and there is **no** resize recipe for `hint` either — all three `fashioniq_resize_images` recipes in that file belong to `clvc_net`, `dcnet`, `limn` (`:33,95,278`). Executed `prepare_model.py --model hint --json` → `"preparation_id": null`, `"contract_type": "standard_fashioniq"`, `"automation_policy": "NONE_REQUIRED"`, `raw_inputs_missing` = the 4 `resized_image` paths + 3 correction dicts, `generated_artifacts_missing: []`, `notes: "Standard FashionIQ layout required; no custom generated artifacts."` So the registry's `preparation_contract: manual_ilearn_resized` (a name legal at `registry.py:73`) has **no contract record behind it** and the status view silently falls back to a standard-layout label. P2.
- Related, model-specific: **the Hi. `resized_image` size is nowhere specified.** `hint` has no resize recipe (see above); the sibling FashionIQ `start_kit/resize_images.py` default is `--image_size 256` (fetched: `raw.githubusercontent.com/XiaoxiaoGuo/fashion-iq/master/start_kit/resize_images.py:69`), and the sibling receptor recipes in this registry use 256. The workbench labels this `OPERATOR_PROVIDED` (`cluster_b2_encoder_family.yaml:88`) with `generation_script: null`; a differently-sized staging would still pass the directory-presence check. UNKNOWN, P3.
- `workbench/registry/auxiliary_assets.yaml` has **0** `hint` rows (grep count 0). The richer cluster-B2 audit file (`registry/assets/cluster_b2_encoder_family.yaml`, and the near-duplicate `registry/audits/cluster_b_encoder_split.yaml`) is read by no runtime code (grep: no `.py` consumer). One internal inconsistency inside it: `hint_generated_test_pkls` is marked `required_for_evaluation: true` (`:103`) yet that same file's not-required list (quoted at `docs/AUXILIARY_ASSET_AUDIT.md:34`) says the same asset is not required for evaluation. No runtime effect. P3.

## PREPROCESSING CONTRACT

Image path:
- `train.py:74-80` hardcodes `transform = "targetpad"`, `input_dim = 224`, `target_ratio = 1.25` and builds `targetpad_transform(1.25, 224)`.
- `data_utils.py:92-106` → `TargetPad(1.25, 224)` (`:50-73`, aspect-ratio-preserving zero pad) → `Resize(224, BICUBIC)` (grep: `PIL.Image.BICUBIC` at `:84,101`) → `CenterCrop(224)` → RGB → `ToTensor` → `Normalize((0.48145466,0.4578275,0.40821073),(0.26862954,0.26130258,0.27577711))`.
- Applied to query references and targets by `datasets.py:114-122` (`transform` call at `:120`) and to gallery images by the same method via `get_test_data:150-162`.
- The LAVIS visual processors from `load_model_and_preprocess` (`train.py:119-121`) are not used by this dataset path (the processor call is commented out at `datasets.py:121`).
- Workbench side: nothing for HINT applies `targetpad`; the generic encoder evaluator operates on cached features (`src/evaluation/fashioniq_encoder.py:21-53`) and the layout check tests presence only. So image-preprocessing parity for a HINT replay is **NOT IMPLEMENTED**, UNKNOWN. P2 (blocks any numeric replay).

Text path:
- `datasets.py:75-80` `correct_text`: lowercase → `str.maketrans({punctuation: ' '})` over ASCII `string.punctuation` → strip/split → per-token exact dictionary substitution → single-space join. `:82-84` `concat_text`: `"{} and {}".format(...)` source order. Applied to val pairs at `:142`.
- Then `test.py:44` passes the composed string through `txt_processors["eval"]`, which `blip2_pretrain.yaml:32-36` sets to `blip_caption` (second normalisation + BERT tokenisation, `max_txt_len=32`).
- Workbench equivalent: `src/datasets/fashioniq.py:93-117` `normalize_fashioniq_caption` (byte-for-byte the same algorithm: same `string.punctuation` translate, same strip/split, same dict lookup, same single-space join) and `compose_fashioniq_caption(..., policy="normalized_ordered_and")` (`:119-172`, branch `:158-162`). Stage 1 only, and only if the identical dictionaries are supplied — which they are not. Stage 2 has no workbench counterpart. UNKNOWN.

## AUGMENTATION CONTRACT

- Training: `train.py:131-132` `model.train()` + `model.apply(set_bn_eval)`; no image augmentation beyond the deterministic `targetpad`; the `FashionIQ` dataset has no augmentation branch. Seeds set at `train.py:255-262` (`args.seed` default 42 at `:47`; `torch.backends.cudnn.deterministic = True` at `:262`).
- Evaluation: `test.py:11 model.eval()`; one deterministic forward pass per batch (`test.py:36-62`); no augmentation, no dropout.
- Therefore the registered cohort's `evaluation_noise_pct: 0` (`models.yaml:401`, protocol `:19`) is consistent with the source. VERIFIED.
- `checkpoint_training_noise_pct: 0` (`:400`): the source's `--noise_ratio` default is 0 (`train.py:56`), but the string `noise` appears in only four places in the whole repository (all `train.py:56,89,99` plus the flag name), i.e. **there is no noise-injection implementation reachable at all**; the flag is passed to constructors that reject it (see parity section). The Hub card says only that the model was "trained on the FashionIQ dataset" (`hfcard.md:65`) and the paper never mentions a training-noise condition (grep `noise` in the PDF text: 2 hits, both unrelated prose/reference). So 0 % is a *default-value inference*, not a published author statement. UNKNOWN-as-fact / consistent-with-default. P2 (labelling).

## GALLERY CONTRACT

Source — `datasets.py:FashionIQ.get_test_data` (`:131-165`):

```
140        for idx in range(len(ref_captions)):
…
146            out['source_img_id'] = images.index(candidate)
148            out['target_img_id'] = images.index(target)
…
156            if i['source_img_id'] not in test_targets_id:
157                test_targets_id.append(i['source_img_id'])
158            if i['target_img_id'] not in test_targets_id:
159                test_targets_id.append(i['target_img_id'])
…
164            out['target_img_data'] = self.get_test_img(name + '_' + images[i])
```

i.e. the ordered, first-seen union of each annotation's reference then target, with ids being integer positions into the full `split.{cat}.val.json` array; `test_targets` is that union in order.

Workbench — `build_pair_union_gallery` (`src/datasets/fashioniq.py:186-210`): iterates annotations, appends `annotation.reference_id` if unseen then `annotation.target_id` if unseen, preserving order.

| Property | HINT `get_test_data` | workbench `build_pair_union_gallery` | Match |
|---|---|---|---|
| iteration order | `cap.{cat}.val.json` order (`:140`) | annotation order (`:198`) | identical if loaded in file order |
| per-annotation order | reference (`:156`) then target (`:158`) | reference (`:201-203`) then target (`:205-207`) | identical |
| dedup key | integer `images.index(id)` (`:146,148`) | id string (`:201,205`) | identical on a duplicate-free split |
| collection | ordered list | list + set | identical |

Reference handling: source masks before sorting (`test.py:83` sets `sims[i, test_targets_id.index(t['source_img_id'])] = -10e10`, then `:86/:100` `np.argsort(-sims[i,:])[:50]`). Workbench `recall_at_k_reference_excluded` (`src/evaluation/fashioniq.py:35-76`) sorts first (`:71`) then removes the reference column (`:72`), and says so in its own docstring (`:47-52`). Same universe, same unretrievability, **different mechanism** — equivalent for the top-k hit test whenever every other score exceeds the mask, which is asserted-away only by argument, not by code. The ambiguity of `np.argsort` (unstable, `test.py:86,100`) vs `torch.argsort(..., stable=True)` (`src/evaluation/fashioniq_encoder.py:44`) is documented as unpinnable at `src/evaluation/fashioniq.py:144-146`.

Concrete cohort size, re-derived independently in this audit from the local annotations (executed, `data/fashionIQ_dataset/`):

```
dress  split_len 3817 uniq 3817 caps 2017  pair-union gallery size 2628
shirt  split_len 6346 uniq 6346 caps 2038  pair-union gallery size 3089
toptee split_len 5373 uniq 5373 caps 1961  pair-union gallery size 2902
```

These match the numbers the extracted bundle asserts (`extracted/hint_icassp26.json:65`: "dress 2628, shirt 3089, toptee 2902"). Gallery width is far below the full split, so `fashioniq_val_split` is a genuinely different cohort from the `fashioniq_full_gallery_ref_excluded` family in this registry — the assignment is not the mirror-image mistake. Also verified: the local ids carry no leading/trailing whitespace (first entries `{"target": "B0084Y8XIU", "candidate": "B005X4PL1G"}`; 0 of the sampled values differ from their `.strip()`), so the string-key vs index-key dedup cannot diverge on this data.

VERDICT: cohort construction VERIFIED as semantically equivalent, with a documented mechanism difference and a documented tie ambiguity. P3.

## METRIC CONTRACT

Source (`test.py:86-107`): `nn_result = [np.argsort(-sims[i,:])[:50] …]`; `for k in [1, 10, 50]: r += 1 if test_targets_id.index(test_queries[i]['target_img_id']) in nns[:k]`; `r = 100 * r / len(nn_result)` (denominator = query count); returns per-category `('{name}_r{k}', r)` only — no macro, no mean. The CIRR branch divides by `model.temp` (`test.py:154`) while the FashionIQ branch does not (`:74`) — recorded, not a defect here.

Workbench (`src/evaluation/fashioniq_encoder.py:18-53,56-69`): `FASHIONIQ_ENCODER_RECALL_KS = (1,10,50)` (`:18`), `100.0 * sum(hits) / len(hits)` (`:53`), gallery-uniqueness and membership asserted (`:37,43-48`). Metric *shape* matches. Macro/mean aggregation is workbench-only: `macro_average_fashioniq` (`src/evaluation/fashioniq.py:238-255`) returns `recall_at_10`, `recall_at_50` and `mean_recall = (r10 + r50)/2`. The source's own selection score is the **sum** of the three category R@10s (`train.py:201`), not this mean. Score identity (HINT's `(Q,D)×(D,T)` multivector max-pool matmul, `test.py:45,61,74-76`; `HINT.py:401-455`) vs any workbench score is UNKNOWN and no claim of score parity is made.

## SOURCE-TO-WORKBENCH PARITY

| Axis | Source evidence | Workbench evidence | Verdict |
|---|---|---|---|
| pin | `rev-parse HEAD` = `bec50b6c…` | `models.yaml:381` | MATCH |
| cohesion | `source_clean_and_pinned` → `(True, …)` | `runtime.py:2040-2052,2055+` | MATCH |
| cohort | `datasets.py:156-159` + `test.py:83` | `fashioniq.py:176-179`, `datasets/fashioniq.py:186-210`, `evaluation/fashioniq.py:35-76` | EQUIVALENT (mask-then-sort vs sort-then-remove) |
| evaluator command | `test.py:11` callable only; no CLI | `HintAdapter` (`adapters/models.py:44-46`) inherits `OfficialScriptAdapter.command` (`:24-25`) | BOTH ABSENT (no command exists to route to) |
| checkpoint consumption | no FashionIQ loader (`cirr_test_submission.py:115` is CIRR) | registry points at `fashioniq.pt` | SOURCE_EVIDENCE_MISSING (route); metadata VERIFIED (file exists/name/size) |
| image preprocessing | `targetpad_transform(1.25,224)` (`train.py:76-80`, `data_utils.py:92-106`) | none for HINT | NOT IMPLEMENTED |
| text preprocessing | `datasets.py:75-84` then `test.py:44`+`blip_caption` | `datasets/fashioniq.py:93-172` stage 1 only | INCOMPLETE / UNKNOWN |
| metric shape | percent R@1/10/50, query denominator | `fashioniq_encoder.py:18,53` | EQUIVALENT |
| metric value | needs GPU + checkpoint + dictionaries | none | UNKNOWN |
| aggregate | source emits none | `evaluation/fashioniq.py:238-255` | workbench-only; must never be called a paper metric |

Two documented mechanisms worth recording: `train.py:86-89` passes `noise_ratio=args.noise_ratio` to `datasets.FashionIQ`, whose AST-derived signature is `__init__(self, path, transform)` (`datasets.py:25`) — so `python train.py --dataset fashioniq …` raises `TypeError: __init__() got an unexpected keyword argument 'noise_ratio'` before reading data; the CIRR branch (`train.py:95-100`, `CIRR.__init__(self, path, transform, case_look)`, `datasets.py:170`) has the same defect. That documented command is also the wrong shape for *evaluation*: it trains ten epochs first (`train.py:194,275`) and evaluates the in-memory model.

## RUNTIME STATUS

- Registry `reproduction_status: NOT_RUN` (`:392`). Correct: `find workbench/artifacts -iname "*hint*"` → nothing; `workbench/artifacts/results/*/mock/` holds only mock CSMCIR/SPRC/PAIR/TGCIR/Air-Know/HABIT entries; `workbench/artifacts/logs/` holds only two `limn__ckpt__…` runs; `artifacts/checkpoints/` contains no `hint/`; `gpu_capabilities/` holds 1266 files, **zero** of which mention `hint` (executed). STATIC_AUDIT of the filesystem.
- `_requires_gpu(hint)` → `True` (executed) because `environment.cuda: '12.6'` (`:371`); `runtime.py:805-812`. The source hard-requires CUDA (`test.py:43,60` `.cuda()`, `train.py:122` `model.cuda()`). `environment_required` is unset for `hint`, so `environment_blockers(hint)` → `[]` (executed): the absent environment is silent.
- Executed `pc_readiness.py --model hint --json` → `"state": "BLOCKED_IMPLEMENTATION"`, `"reason": "command not runnable: COMMAND_UNAUDITED"`, with blockers = the 4 `resized_image` paths, the 3 correction dicts, the image-extension check, and `command not runnable: COMMAND_UNAUDITED`. Note the classification: `BLOCKER_KINDS` (`pc_readiness.py:44-60`) has `("command not runnable", BLOCKED_IMPLEMENTATION)` at position 6 and the `FashionIQ` dataset needles after it, so the implementation blocker masks a **dataset** blocker. Under the stated rule ("unverified blockers → BLOCKED_EXTERNAL_ASSET, not BLOCKED_IMPLEMENTATION") the remaining local gap (correction dictionaries, no acquisition source) is not separately surfaced. P3.
- `workbench/docs/READINESS_MATRIX.md:31` reproduces the same blocker text and state. Consistent.
- Therefore: **NOT_RUN**, and additionally not runnable as registered (no adapter command, no upstream evaluation CLI, no FashionIQ checkpoint loader, missing mandatory inputs).

## PAPER-METRIC STATUS

The paper's Table 1 was read by three independent routes in this audit:

1. `pdftotext -layout -f 4 -l 4 hint.pdf` → line `17`: ` HINT (Ours) 51.80 73.13 60.27 78.48 63.37 82.19 58.48 77.93 | 52.34 82.65 90.50 98.39 | 80.48 92.31 96.77 81.57`, under header `Method | Dresses | Shirts | Tops&Tees | Avg | R@k | Rsubset@k` with `R@10 R@50` under each category and under Avg.
2. `https://arxiv.org/html/2603.26341v1` (HTTP 200, 165220 B) → `HINT (Ours) 51.80 73.13 60.27 78.48 63.37 82.19 58.48 77.93`.
3. Rendered page image `pdftoppm -f 4 -l 4 -r 160 -png`, and `workbench/third_party/HINT/assets/results.png` (README `:78`, tracked in the pinned commit), both read visually: same cells.

Arithmetic re-checked: `mean(51.80,60.27,63.37) = 58.480`; `mean(73.13,78.48,82.19) = 77.9333` → the printed Avg cells `58.48`/`77.93`; `(58.48+77.93)/2 = 68.2067`.

- `models.yaml:389-390` `reported_r10 58.48 / reported_r50 77.93` = the paper's Avg cells. VERIFIED (PAPER_REPORTED).
- `models.yaml:391 reported_mean: 68.21` = `(58.48+77.93)/2` — the paper prints no such number. The committed row correctly withholds it from `reported_metrics` (`papers.yaml:1060-1063`, `mean: null`) and supplies it only as `derived_metrics` with `provenance: WORKBENCH_DERIVED` (`:1064-1072`). Executed in-process: `value_for('r10')=58.48 PAPER_REPORTED`, `value_for('r50')=77.93 PAPER_REPORTED`, `value_for('mean')=68.21 WORKBENCH_DERIVED`. The one place where the two are not separated is `models.yaml`'s flat `reported_r10/r50/mean` triple, which `autopilot.py:774` and `evaluate_models.py:309` read as `paper_metrics` and `autopilot.py:815-816` renders as `paper_r10/paper_r50/paper_mean` in the aggregate report. P3 (labelling; the gate itself uses the committed row, not this triple).
- Category cells also verified against the row: `papers.yaml:1074-1082` `dress 51.8/73.13`, `shirt 60.27/78.48`, `toptee 63.37/82.19`, identical to the PDF.
- Paper protocol statement (p.3, `hint.txt:226-227`): "For FashionIQ, we evaluate R@10, R@50 for each category, as well as their averages." The paper states **nothing** about the gallery, the split file, or reference exclusion; grep for `gallery|val split|validation split|split` in the PDF text yields no protocol statement. So the cohort attribution comes from the code (verified above), not the paper. The committed row says exactly this (`papers.yaml:1095-1098`). Correct.
- The paper's own improvement claim is arithmetically wrong: README `:78`/paper state "approximately 9.74% in average R@10"; the best non-HINT FashionIQ R@10 average in the same table is SPRC `(49.18+55.64+59.35)/3 = 54.7233`, so `58.48/54.7233-1 = +6.87%`, and `58.48/1.0974 = 53.29` matches no cell in the table. This is a defect in the paper, not in the workbench; the registry does not repeat the claim. P3 (recorded so it is not mistaken for a workbench error).
- **Gate behaviour, executed (RUNTIME_VERIFIED for the workbench code, not for any model run):** with the committed hint row,
  - mock kind → `NOT_RUN`;
  - no reproduced metrics → `FAILED_OUTPUT_VALIDATION ['no reproduced metrics']`;
  - paper-exact metrics + full run_meta → `PASS_WITHIN_5_PERCENT` with 9 comparisons (r10/r50/mean + 6 category cells) at 0.00 %;
  - a wrong `r10` of `57.0` (2.53 % low, inside 5 %) **and no observed category metrics** → `PASS_WITHIN_5_PERCENT`, detail admits the six category cells were "published by the paper but the run observed none". A run whose Dresses result is 8 points off therefore still passes the gate as long as the macro is inside 5 % and no category metrics were parsed;
  - a run whose `run_meta` declares 80 % training noise → `BLOCKED_CHECKPOINT_MISMATCH ['checkpoint_training_noise_pct mismatch: reference=0 run=80']` (condition refusal works);
  - with `checkpoint_training_noise_pct=None` the selector still returns `hint_fiq_clean_val_split` as an "exact condition match" (`select_reference` skips the condition check when the argument is `None`, `benchmark_references.py:170-173`).
  `extract_aggregate_metrics('hint', …)` → `AGGREGATE_PARSER_UNAVAILABLE` / `PARSER_UNAVAILABLE` for any stdout (`metrics_extraction.py:328-342`), so for HINT the gate can only ever see metrics that a human or a future parser supplied. P2.

## CRITICAL FINDINGS

**C1 (P1) — the registered artifact cannot be reproduced, and the registered pointer is not the paper's model by any executable route.** No adapter command (`adapters/models.py:44-46` + base raise `:24-25`), no upstream evaluation-only CLI (`test.py` greps clean; `train.py:199` only), no FashionIQ loader for `fashioniq.pt` (`cirr_test_submission.py:115` is CIRR), no parser (`metrics_extraction.py:334-342`), no checkpoint on disk, and mandatory inputs absent. Nothing was reproduced and nothing can be from this registration. Action: keep `NOT_RUN`; if a replay is ever wanted it must be a reviewed observation wrapper that counts as a declared variant.

**C2 (P1) — the correction dictionaries are mandatory, unpublished, unhashed, and they change every query string.** `datasets.py:63,137`; `SOURCE_EVIDENCE_MISSING` (`cluster_b2_encoder_family.yaml:66,75-76`); presence-only check (`fashioniq_layout.py:32`); locally missing. Action: keep native evaluation blocked until operator-supplied files with recorded hashes exist; never substitute a sibling repository's dictionaries.

**C3 (P1) — the pinned training entry point cannot construct its own dataset, and every workbench document describes it as if it could.** `train.py:86-89` (`noise_ratio=`) vs `datasets.py:25` (`__init__(self, path, transform)`); same defect for CIRR (`train.py:95-100` vs `datasets.py:170`). `docs/FASHIONIQ_PROTOCOL_PARITY.md:67,114`, `docs/UPSTREAM_DATA_PREPARATION_AUDIT.md:122`, `docs/audits/PHASE2_SOURCES_C.md:9` all cite `train.py:199` as a working path. Action: record the source defect; do not patch.

**C4 (P1) — the noise condition is assumed, not published.** `models.yaml:400 checkpoint_training_noise_pct: 0` descends from `train.py:56`'s default, while the flag reaches no implementation (four `noise` occurrences in the repo, all `train.py:56,89,99`), the Hub card states nothing, and the paper states nothing. Because the gate's `COMPARABILITY_FIELDS` includes this field and the committed row asserts `0`, a run under any other real condition would be refused — the value is load-bearing. Action: mark it UNKNOWN/unpublished, as was done for LIMN (`limn_fashioniq_base_iter0_table1`), or obtain an author statement.

**C5 (P2) — the gate's category control can be bypassed by not parsing categories.** The new per-category comparison is correct when both sides state a value (`paper_gate.py:35-38,340-367`), but "the run observed none" is a `detail` note, not a refusal (`:382-385`). Executed above: a 2.5 %-off macro with no categories passes while all six paper category cells go unchecked. For HINT this is the *only* possible outcome today because no parser exists. Action: when the referenced row publishes category metrics and the run reports an aggregate without them, either require the categories or downgrade the verdict out of the plain `PASS` state.

## MINOR FINDINGS

**M1 (P2) — `expected_sha256: null` while the author publishes the LFS hash** (`models.yaml:403`; Hub `bf9c72d4…`). Set it.

**M2 (P3) — `checkpoint_serialization` missing for `hint`** while eight siblings declare one; source form is whole-model (`train.py:204,279`; `cirr_test_submission.py:115`). Add `whole_model`.

**M3 (P2) — the declared layout omits the fresh-constructor train inputs and the cache artifacts.** `fashioniq_layout.py:25-34` vs `datasets.py:61,33,49-57`. Either switch to `fashioniq_ilearn_resized_training` or declare the cache artifacts.

**M4 (P2) — `preparation_contract: manual_ilearn_resized` with no `preparation_id` and no resize recipe.** `models.yaml:376`; `preparation_contract_for_model` returns `None` (`registry.py:187-194`); executed `prepare_model.py` label `standard_fashioniq` / policy `NONE_REQUIRED`. The registry cannot even declare the resize size for this model.

**M5 (P3) — `checkpoint_mapping_status` vocabulary is duplicated and inconsistent across the two registries.** `models.yaml:405` = `VERIFIED_METADATA` (registry vocabulary `{VERIFIED_METADATA, UNVERIFIED, ARTIFACT_ASSOCIATED_UNVALIDATED}`, `registry.py:29`), while the benchmark row asserts `VERIFIED` (`papers.yaml:1048`, `VALID_MAPPING = {"VERIFIED","UNVERIFIED","NO_CHECKPOINT"}`, `build_paper_registry.py:43`). The paper-registry field is **never read at runtime** (grep: no consumer in `benchmark_references.py`/`paper_gate.py`/`evaluate_models.py`). Two fields, one name, two vocabularies, one dead.

**M6 (P3) — `pc_readiness` state label is misleading.** `BLOCKED_IMPLEMENTATION` wins because `"command not runnable"` precedes the dataset needles (`pc_readiness.py:44-60`); the dataset/external-asset reality is only in the blocker text. The "command cannot be audited" state is not the model's local problem — the source has no command.

**M7 (P3) — the workbench-supplied `paper_mean` is a workbench quantity rendered next to paper fields.** `autopilot.py:774,815-816` put `mean: 68.21` in `paper_metrics`; `models.yaml:391` keeps it in a flat `reported_*` triple. The committed row separates it correctly.

**M8 (P3) — prior-doc line citations have drifted and one is wrong at the revision it claims.** `docs/reviews/models/hint.md:5,9,11` cite `models.yaml:340-381`, `:349`, `:359`, `:358`, `:353`; at `eb36aa5` (the revision that document was committed with) the block is `:360-401`, `method_name` `:369`, `source_dir` `:379`, `source_available` `:378`, `paper_title` `:373`. The doc may therefore have been written against a pre-commit tree. Its `README.md:363,381` "dead personal-URL" claim is also stale: at `eb36aa5` those lines are project-structure and related-projects content whose GitHub links are live (`github.com/lee-zixu/ConeSep`, `github.com/zhihfu/Air-Know`); no `zh-mingyu` README link exists in the pinned commit at all (grep). Action: re-cite against `eb36aa5` and drop the README-URL claim.

**M9 (P3) — the "6 page(s)" claim in the extracted bundle conflicts with the PDF's own `/Pages 5`.** `extracted/hint_icassp26.json:201,217`. Cosmetic.

**M10 (P3) — the cluster asset audit marks `hint_generated_test_pkls` `required_for_evaluation: true`** (`cluster_b2_encoder_family.yaml:103`) while a sibling file in the same registry marks the same asset not required. No runtime consumer.

**M11 (P3) — the same SHAs are recorded twice in per-model asset records with divergent notes** (`registry/assets/cluster_b2_encoder_family.yaml:57-77` vs `registry/audits/cluster_b_encoder_split.yaml:50-51`). Not read by any code.

**M12 (P3) — the paper's blog-style improvement claim (9.74 %) does not reproduce (6.87 %).** Paper-side defect.

## UNRESOLVED EVIDENCE

U1 — Whether `fashioniq.pt` was produced by this pinned commit's loop, at which epoch, in which condition. The Hub says "Trained on FashionIQ"; nothing more. UNKNOWN.
U2 — The checkpoint's actual bytes: not downloaded, not hashed locally (downloads forbidden). Only the author-published LFS pointer hash is known. UNKNOWN.
U3 — Whether the paper's Table 1 FashionIQ numbers were produced on the pair-union cohort. The paper is silent; the code computes that cohort during training. UNKNOWN.
U4 — Provenance and content of the three `correction_dict_{cat}.json` files. UNKNOWN.
U5 — The resized-image recipe actually used for the paper run (no source script, no size declaration, no recipe record). UNKNOWN.
U6 — Whether `fashioniq.pt` unpickles under the pinned environment. `HINT` is registered (`HINT.py:235`, exported `lavis/models/__init__.py:41,84`) and the LAVIS package's importability is not established for this tree, so this needs a runtime check. UNKNOWN.
U7 — Whether the two text stages compose to identical token ids under the pinned LAVIS/Transformers versions. UNKNOWN.
U8 — Whether the harness will be pointed at `data/FashionIQ` (a stale root holding only `captions/`, `image_splits/`, `images/`, `qwen_captions/` — no FashionIQ val files) or at `data/fashionIQ_dataset` (the configured, populated root). A stale env file would silently mark every model BLOCKED. UNKNOWN.
U9 — Whether any GPU-host capability record for `hint` can exist given none of the 1266 present records names it. UNKNOWN.

## EXECUTION TRACE

`paper/README/config → source dataset → preprocessing → official evaluator → workbench invocation → checkpoint → output → metric comparison`

1. **paper/README/config →** paper `https://arxiv.org/pdf/2603.26341v1` (HTTP 200, 2681236 B, sha256 `fbe97798…`): §3.1 sets the FashionIQ metric (R@10/R@50 per category and their averages, `hint.txt:226-227`), Table 1 page 4 carries the six FashionIQ cells for `HINT (Ours)`; NO gallery/split/reference statement. README `workbench/third_party/HINT/README.md:119,126,129` pins Python 3.8.10 / CUDA 12.6 / torch 2.1.0 / torchvision 0.16.0 / open-clip-torch 2.24.0 / scikit-learn 1.3.2 / transformers 4.25.0 / salesforce-lavis 1.0.2 / timm 0.9.16; README `:201-208` documents the FashionIQ training command and `:238-241` the CIRR submission test. Registry mirrors env (`models.yaml:365-372`, `confidence: VERIFIED_README` — literally true), paper (`:377-378`), checkpoint (`:395-405`).
2. **→ source dataset:** `train.py:85-89` selects `datasets.FashionIQ(path=args.fashioniq_path, transform=img_transform, noise_ratio=args.noise_ratio)`; `datasets.py:28-31` fixes the three directories by concatenation; fresh branch `:33-47`; `train_init_process` `:59-73`; `get_test_data` `:131-165`. **Broken arrow:** the `noise_ratio` kwarg (`train.py:89`) is not accepted by `datasets.py:25` (AST-checked) → `TypeError`. Also note the documented command trains before it evaluates (`train.py:194,275`).
3. **→ preprocessing:** `train.py:74-80` hardcodes `targetpad_transform(1.25, 224)`; `data_utils.py:50-73,92-106`; applied at `datasets.py:102-110,114-122`; text via `datasets.py:75-84` then `test.py:44` with `blip_caption` (`blip2_pretrain.yaml:32-36`).
4. **→ official evaluator:** `train.py:198-199` loops `['dress','shirt','toptee']` calling `test.test`; `test.py:11-110`: `extract_retrieval_compose` (`:45`) / `extract_retrieval_target` (`:61`) → `np.matmul` `.max(-1)` (`:74-76`) → reference mask `-10e10` (`:83`) → `np.argsort(-sims)[:50]` (`:86,100`) → R@1/10/50 percent on the query count (`:101-107`). Model: `HINT` at `HINT.py:235-241`, `from_config` `:565`.
5. **→ workbench invocation:** `adapters/models.py:44-46` `HintAdapter(script="test.py")`, `command` not overridden → base `raise NotImplementedError` (`:24-25`). Executed `--list` → `command=not audited`; `--dry-run` → `SKIPPED: adapter command not audited`. Registry `supported_protocols` resolves through `evaluation/fashioniq.py:176-179` + `datasets/fashioniq.py:186-210` + `evaluation/fashioniq.py:35-76`. **No invocation can be built.**
6. **→ checkpoint:** registry `fashioniq.pt`, size 2724451182 (`:398,402`); Hub confirms existence, size and LFS `bf9c72d4…`; no `hint` file under `artifacts/checkpoints/`; the only `torch.load` in the tree is CIRR. No FashionIQ load path.
7. **→ output:** none. `reproduction_status: NOT_RUN`; no `hint` artifact anywhere under `workbench/artifacts`; `extract_aggregate_metrics('hint', …)` → `AGGREGATE_PARSER_UNAVAILABLE`.
8. **→ metric comparison:** paper Table 1 (three routes) → macro `58.48 / 77.93`, derived mean `68.21`, matching `models.yaml:389-391` and the committed row `papers.yaml:1060-1080`; the committed row reproduces byte-for-byte from the bundle (`build_paper_registry.py --output <scratch>` → `diff` minus comments → IDENTICAL, `registry_version extract-ec9e596b671c`). No observed workbench metric exists, so the comparison is registry-vs-paper only.

## CHECKLIST

1. **Registered id/source_dir consistent** — VERIFIED — `models.yaml:364,383`; `model_by_id('hint')` loads (`source_available True`, `source_dir HINT`) — NONE — none.
2. **Pinned commit matches the checkout** — VERIFIED — `models.yaml:381` vs `git -C workbench/third_party/HINT rev-parse HEAD` = `bec50b6c8c19111893b617979502b948d1cea5b2` — NONE — none.
3. **Checkout clean at the pin** — VERIFIED — `git status --porcelain` only `__pycache__`; executed `source_clean_and_pinned` → `(True, 'bec50b6c…', None)` — NONE — none.
4. **Paper title/URL resolve to this model** — VERIFIED — `models.yaml:377-378`; arXiv abs/export/PDF fetches — NONE — none.
5. **`upstream_repo_url` resolves to the pinned source** — VERIFIED, and the prior round's "discrepancy" is REFUTED — `git ls-remote` on both `iLearn-Lab/ICASSP26-HINT` and the paper-printed `zh-mingyu/HINT` returns `bec50b6c…HEAD` — P3 — drop the prior doc's README-URL claim (M8).
6. **PDF page count claim** — MISMATCH — bundle says "6 page(s)" (`extracted/hint_icassp26.json:201`); `pdfinfo` says `Pages: 5` — P3 — correct the bundle.
7. **Checkpoint exists at the registered URL with the registered size** — VERIFIED — Hub API: `fashioniq.pt` size 2724451182 == `models.yaml:402` — NONE — none.
8. **Checkpoint hash pinned** — MISMATCH — `models.yaml:403 expected_sha256: null` while the Hub publishes LFS `bf9c72d42206b667ed586758c15c648970df821857e4067f26155a114a6517d4`; `checkpoint_availability` → `official_sha256_known false` — P2 — record it (M1).
9. **Checkpoint→paper-row mapping verified** — UNKNOWN — `VERIFIED_METADATA` is a metadata claim only; no source path loads `fashioniq.pt` for FashionIQ (`cirr_test_submission.py:108,115`; `train.py:59` unused; `test.py:11`) — P1 — say UNVERIFIED mapping in `notes` (C1).
10. **Checkpoint serialization declared** — MISMATCH — absent for `hint`, present for 8 others; source form whole-model — P3 — add `whole_model` (M2).
11. **Declared native protocol equals what the source computes** — VERIFIED — `models.yaml:386` vs `datasets.py:156-159` + `test.py:83` — NONE — none.
12. **Protocol table entry matches** — VERIFIED — `models.yaml:16-19` vs `evaluation/fashioniq.py:176-179` — NONE — none.
13. **`literature_split_label` agrees with the table** — VERIFIED — `models.yaml:387`; enforced `registry.py:48-51` — NONE — none.
14. **`supported_protocols` includes the native protocol** — VERIFIED — `models.yaml:386,388`; enforced `registry.py:42-47` — NONE — none.
15. **Gallery construction matches `fashioniq_val_split`** — VERIFIED — `datasets.py:156-159` vs `datasets/fashioniq.py:186-210` (reference-then-target, first-seen, order preserved) — NONE — none.
16. **Gallery construction byte-identical in procedure** — VARIANT — source dedups integer split indices (`datasets.py:146,148`), workbench dedups id strings; observed duplicate-free local splits (3817/6346/5373 unique) make them coincide — P3 — keep.
17. **Cohort size independently recomputed** — VERIFIED — executed local recomputation → dress 2628, shirt 3089, toptee 2902 — NONE — none.
18. **Reference exclusion implemented** — VERIFIED — `test.py:83`; `evaluation/fashioniq.py:70-76` — NONE — none.
19. **Reference-exclusion procedures identical** — VARIANT — mask-then-sort vs sort-then-remove; equivalence argued only (`evaluation/fashioniq.py:47-52`) — P3 — keep as documented; never assert numeric identity.
20. **Tie rule matches** — MISMATCH — `np.argsort` (`test.py:86,100`) vs `torch.argsort(..., stable=True)` (`fashioniq_encoder.py:44`); ambiguity named at `evaluation/fashioniq.py:144-146` — P3 — record for rank-level comparisons.
21. **Recall@K set matches** — VERIFIED — `test.py:101` vs `fashioniq_encoder.py:18` — NONE — none.
22. **Recall denominator matches** — VERIFIED — `test.py:106` vs `fashioniq_encoder.py:53` — NONE — none.
23. **Macro aggregate not claimed as a paper metric** — PARTIAL — `macro_average_fashioniq` (`evaluation/fashioniq.py:238-255`) vs `models.yaml:391 reported_mean: 68.21`; committed row keeps it `WORKBENCH_DERIVED` (`papers.yaml:1064-1072`) — P3 — label the flat `reported_*` triple (M7).
24. **Paper R@10 = 58.48** — VERIFIED — Table 1 macro of 51.80/60.27/63.37 = 58.480, three read routes — NONE — none.
25. **Paper R@50 = 77.93** — VERIFIED — 73.13/78.48/82.19 → 77.9333 — NONE — none.
26. **Paper mean = 68.21** — VARIANT — `(58.48+77.93)/2 = 68.2067`; not printed — P3 — keep labelled derived.
27. **Category cells match the paper** — VERIFIED — `papers.yaml:1074-1082` vs PDF line 304 — NONE — none.
28. **The registry reproduces from the bundles** — VERIFIED — executed `build_paper_registry.py --output /tmp/...` → diff IDENTICAL — NONE — none.
29. **Reported metrics are sanity references only** — VERIFIED — `reproduction_status: NOT_RUN` (`models.yaml:392`); no `hint` artifact — NONE — none.
30. **Evaluation-only CLI exists upstream** — NOT_RUN — greps on `test.py` for `__main__|argparse|ArgumentParser` → none; 190-line file; only `train.py:35-62` and `cirr_test_submission.py:111` have CLI surfaces — P1 — record as a source blocker (C1).
31. **Adapter builds a command** — NOT_RUN — `adapters/models.py:44-46`, base raise `:24-25`; executed `--dry-run` → `SKIPPED: adapter command not audited` — P1 — leave unimplemented (C1).
32. **Documented upstream FashionIQ command actually runs** — NOT_RUN (it does not) — `README.md:201-208` vs `train.py:89` → `datasets.py:25` (`TypeError`); and it would train ten epochs first (`train.py:194,275`) — P1 — record the defect (C3).
33. **Environment stack matches the README** — VERIFIED — `models.yaml:368-372` vs `README.md:119,126,129` — NONE — none.
34. **`confidence: VERIFIED_README` justified** — VERIFIED — the versions are literally in the pinned README — NONE — none.
35. **Environment actually provisioned/verified** — UNKNOWN — `WORKBENCH_HINT_PYTHON` is a name; `environment_required` unset → `environment_blockers(hint)` = `[]` (executed); no interpreter recorded — P2 — verify on the target host.
36. **CUDA requirement declared and gated** — VERIFIED — `models.yaml:371`; `_requires_gpu(hint)` → `True` (executed); source `.cuda()` (`test.py:43,60`; `train.py:122`) — NONE — none.
37. **No `hint` GPU capability record exists** — NOT_RUN — 1266 records scanned, 0 mentioning `hint` — P3 — expected under NOT_RUN.
38. **Declared layout matches what the evaluator opens** — VARIANT — `fashioniq_layout.py:25-34` vs `datasets.py:29-31,61,104,115,133-137` — P2 — add train captions or declare the cache contract (M3).
39. **Image extension matches** — VERIFIED — `models.yaml:385 [".jpg"]` vs `datasets.py:104,115` (`+ ".jpg"`) — NONE — none.
40. **Path-separator requirement** — CONCERN — `datasets.py:29-31` concatenates, so the future command must pass `<path>/`; recorded at `cluster_b2_encoder_family.yaml:99`; `EncoderAdapter` does this (`adapters/models.py:41`) — P3 — none until a command exists.
41. **Correction dictionaries required** — VERIFIED — `datasets.py:63,137` — NONE — none.
42. **Correction dictionaries available/provenance-verified** — NOT_RUN — `cluster_b2_encoder_family.yaml:66,75-76`; presence-only check; locally missing; no `hint` row in `auxiliary_assets.yaml`; no preparation contract — P1 — block until hashed files exist (C2).
43. **Data provenance of the canonical root** — UNKNOWN — `data/fashionIQ_dataset` carries no receipt/checksum file; nothing in the registry asserts its provenance — P3 — recorded.
44. **Image preprocessing reproduced** — NOT_RUN — upstream `targetpad_transform(1.25,224)` (`train.py:74-80`, `data_utils.py:92-106`); no workbench path for HINT — P2 — required for any replay.
45. **Text preprocessing reproduced** — UNKNOWN — two upstream stages (`datasets.py:75-84` + `test.py:44`); workbench has stage 1 only and only with the unverified dictionaries (`datasets/fashioniq.py:93-172`) — P2 — verify strings/token ids on the host.
46. **Evaluation-noise contract clean** — VERIFIED — `test.py` is a plain deterministic pass (`:11`, no transforms beyond `targetpad`); `evaluation_noise_pct: 0` (`models.yaml:401`, protocol `:19`) — NONE — none.
47. **Training-noise provenance established** — UNKNOWN — `models.yaml:400`; source default 0 (`train.py:56`) but no implementation; Hub and paper say nothing — P1 — mark UNKNOWN (C4).
48. **Checkpoint loaded by the pinned source for the registered protocol** — NOT_RUN — no FashionIQ loader — P1 — the fundamental gap (C1).
49. **Source pin checked mechanically** — VERIFIED — `runtime.py:2055-2077`; executed clean pass — NONE — none.
50. **Local dataset staging complete for this model** — NOT_RUN — executed layout existence scan: 8 declared paths missing; `READINESS_MATRIX.md:31` agrees — P1 — complete staging before any run.
51. **Cache artifacts declared** — NOT_RUN — `datasets.py:33,49-57`; no layout helper mentions them — P2 — declare or forbid (M3).
52. **`preparation_contract` backed by a record** — MISMATCH — `models.yaml:376` legal-name check only (`registry.py:73`); no `preparation_id`; 0 `hint` rows in `preparation_contracts.yaml`; executed `prepare_model.py` → `preparation_id: null`, label `standard_fashioniq` — P2 — add a contract or declare the fallback (M4).
53. **Gate wired to a verified row for this model** — VERIFIED (code), with a bypass — executed: `references_for_model('hint')` → one `VERIFIED` row; paper-exact → `PASS_WITHIN_5_PERCENT` (9 comparisons); wrong-but-inside-5 % macro with no categories → still `PASS_WITHIN_5_PERCENT`; 80 %-noise run → `BLOCKED_CHECKPOINT_MISMATCH` — P2 — close the category bypass (C5).
54. **Metric parser exists for this model's output** — NOT_RUN — `extract_aggregate_metrics('hint', …)` → `AGGREGATE_PARSER_UNAVAILABLE` (`metrics_extraction.py:328-342`) — P2 — a run could not be judged even if it happened.
55. **Registry honesty about state** — VERIFIED — `reproduction_status: NOT_RUN` (`:392`), `checkpoint_mapping_status: VERIFIED_METADATA` (`:405`, metadata-only claim), `confidence: VERIFIED_README` (`:367`, true) — NONE — none.
56. **Prior workbench documents consistent with the code** — MOSTLY — the methodology claims in `PHASE2_SOURCES_C.md` (HINT section), `PROTOCOL_AUDIT.md:38,42`, `UPSTREAM_DATA_PREPARATION_AUDIT.md:120-124`, `FASHIONIQ_PROTOCOL_PARITY.md:67`, `SOURCE_TO_WORKBENCH_PARITY_MATRIX.md:24` all re-derive correctly from the code I read; three are wrong or stale: `docs/reviews/models/hint.md` line citations (M8), its README-URL claim (M8), and every document that treats `train.py` as a working train-and-evaluate path (C3) — P2 — correct those.
57. **UNKNOWN never promoted to PASS** — VERIFIED — items 9, 35, 44, 45, 47, 51, U1-U9 remain UNKNOWN/NOT_RUN — NONE — none.
58. **No fabricated hash/URL/line/metric** — VERIFIED — every hash (`bf9c72d4…`, `fca4657d…`, `fbe97798…`), URL and line number above came from an observed command result or a read file — NONE — none.
59. **No unobserved runtime claim** — VERIFIED — the only RUNTIME_VERIFIED items are read-only workbench-code executions and HTTP fetches; no model was loaded or run — NONE — none.
60. **Read-only compliance** — VERIFIED — `git status --porcelain` at the end of this audit matches the pre-existing 15 modified / 3 untracked working-tree state; nothing was staged, committed, or written under `/home/heheboiz/data/cir` (all scratch in `/tmp/audit-final/hint/`) — NONE — none.

## FINAL VERDICT: VARIANT — the pin, the cohort reconstruction (pair-union gallery with the reference masked before ranking, re-derived from `datasets.py:156-159` + `test.py:83` and re-measured locally at 2628/3089/2902), and all ten paper cells are independently correct, and the committed benchmark row reproduces byte-for-byte from the extracted bundle; but nothing was reproduced and nothing can be from this registration — there is no evaluation command upstream or in the workbench, no source loader for `fashioniq.pt` for FashionIQ, a training entry point that crashes on its own dataset constructor, mandatory correction dictionaries with no provenance, an assumed rather than published 0 % training-noise condition, and no metric parser — so the model is neither VERIFIED nor BLOCKED (the registry declares `NOT_RUN` honestly) and the registered metadata is all that stands.
