# intent

## MODEL ID
`intent` (INTENT) — `workbench/registry/models.yaml:278-339`.

## PAPER TITLE
`INTENT: Invariance and Discrimination-aware Noise Mitigation for Robust Composed Image Retrieval`, `workbench/registry/models.yaml:291`. Upstream `README.md:1` carries the identical string plus the AAAI-2026 badge. STATIC_AUDIT.

Paper URL: registry `https://ojs.aaai.org/index.php/AAAI/article/view/39181` (`models.yaml:292`). **Network attempt failed**: `curl -sL -o /dev/null -w "%{http_code}"` returned `000` (no response observed), so the OJS landing page is UNVERIFIED here. Crossref independently resolves the title: `GET https://api.crossref.org/works?query.bibliographic=INTENT+Invariance+and+Discrimination-aware+Noise+Mitigation` → title exact, `DOI 10.1609/aaai.v40i25.39181`, `published 2026-03-14`.

## PINNED SOURCE COMMIT
- Registry: `upstream_commit_sha: 2a75fc6f57d1a036bf3167a7fe1d2cee072f51f3`, `upstream_default_branch: main`, `source_dir: INTENT` (`models.yaml:293-297`).
- `git -C workbench/third_party/INTENT rev-parse HEAD` → `2a75fc6f57d1a036bf3167a7fe1d2cee072f51f3`. **Pin matches checkout.**
- `git log -1 --format='%H %ci %s'` → `2a75fc6… 2026-05-25 18:15:51 +0800 Update README.md`.
- Upstream identity: `GET https://api.github.com/repos/iLearn-Lab/AAAI26-INTENT` → `full_name iLearn-Lab/AAAI26-INTENT`, `default_branch main`, `pushed_at 2026-05-25T10:15:52Z`, `fork false`. The README's own clone URL `https://github.com/ZivChen-Ty/INTENT` redirects (`curl -L -w %{url_effective}`) to `https://github.com/iLearn-Lab/AAAI26-INTENT` — one repo, two names. No remote-GitHub comparison of the commit was performed (UNKNOWN whether GitHub HEAD equals the pin today).
- Tracked tree clean: `git diff --name-only HEAD | grep -v pycache` → empty. Untracked non-source: only `__pycache__/*.pyc` (14 paths listed). `workbench/backend/runtime.py` `is_bytecode()` treats those as non-semantic. **No source file in the pinned checkout was edited by the workbench** (contrast `workbench/patches/csmcir.md` for CSMCIR).

## CHECKPOINT IDENTITY
Registry variants (`models.yaml:309-339`):

| checkpoint_id | filename | training noise | status |
|---|---|---|---|
| `fiq_n02` | `intent_fiq_0.2.pt` | 20% | `NO_CLEAN_CHECKPOINT` |
| `fiq_n05` | `intent_fiq_0.5.pt` | 50% | `NO_CLEAN_CHECKPOINT` |
| `fiq_n08` | `intent_fiq_0.8.pt` | 80% | `NO_CLEAN_CHECKPOINT` |

LIVE network evidence (STATIC_AUDIT on remote metadata, not on weights):
- `read https://huggingface.co/api/models/iLearn-Lab/AAAI26-INTENT` → `sha 16ee2d31f05576bbebbad7da0530b412f5e7a1d8`, `usedStorage 10889914088`, siblings exactly: `.gitattributes`, `README.md`, `intent_cirr_0.2.pt`, `intent_fiq_0.2.pt`, `intent_fiq_0.5.pt`, `intent_fiq_0.8.pt`. **The three registry filenames exist verbatim.**
- `curl -sIL .../resolve/main/<file>` → HTTP `200` for all three (and for `intent_cirr_0.2.pt`), with distinct `x-linked-etag`s (`2a504cad…`, `2d59ae00…`, `671bb217…`) and identical `x-linked-size: 2722478522` for every file.
- Registry `expected_size_bytes: null` and `expected_sha256: null` for all three (`models.yaml:316-317,326-327,336-337`). **No size or digest is published in the registry**, so nothing to compare; the sizes I observed are remote HTTP headers, not registry facts.
- **No 0%-noise FashionIQ checkpoint exists on the Hub.** The Hub README (`https://huggingface.co/iLearn-Lab/AAAI26-INTENT/raw/main/README.md`) never names a clean FashionIQ file; its "Usage" step says only "Download the checkpoint files (e.g. `best_model.pth`)". The paper's 0% block is the *fully-supervised setting*, and the repo README states the 0% column equals the traditional fully-supervised paradigm — but neither ships weights for it. So `strict_clean_checkpoint_available: false` (`models.yaml:307`) is **correct**.
- `checkpoint_mapping_status: VERIFIED_METADATA` (`models.yaml:319,329,339`): filename↔noise-pct mapping rests on Hub filenames only. Consistent with `UPSTREAM_DATA_PREPARATION_AUDIT.md:159` ("Hub metadata, not verified payload"). Payload identity (does `intent_fiq_0.2.pt` really encode 20% shuffling?) is **UNKNOWN** — not checked, no download performed.

## EVALUATION PROTOCOL
Registry: `native_protocol: fashioniq_full_gallery_ref_excluded`, `supported_protocols: [fashioniq_full_gallery_ref_excluded]`, `literature_split_label: original` (`models.yaml:300-302`).

Pinned source, traced STATIC_AUDIT:
- Dispatch: `train_INTENT.py:284` `t = test.test(args, model, trainset, category, txt_processors)` inside `train_and_evaluate()`, looped over `['dress','shirt','toptee']` at `train_INTENT.py:283`.
- Dataset default: `datasets.py:26` `def __init__(self, path, category=None, transform=None, noise_ratio=0, split='original-split')`; CLI default `train_INTENT.py:61` `--split default='original-split'`.
- Query set: `test.py:82-86` selects `testset.test_queries_{dress,shirt,toptee}`; built in `datasets.py:166-187` from `cap.{cat}.val.json` (`datasets.py:170`) with `source_img_id = images.index(candidate)` (`:181`) and `target_img_id = images.index(target)` (`:183`).
- Gallery: `datasets.py:203-209` (`elif self.split == 'original-split'`) enumerates **every id in `image_splits/split.{cat}.val.json` in file order**, `test_targets_id.append(id)` for `enumerate(images)`. The alternative `val-split` branch (`datasets.py:191-202`) is dead under the default.
- Reference exclusion: `test.py:150` `sims[i, test_targets_id.index(t['source_img_id'])] = -10e10` before `np.argsort(-sims[i,:])[:50]` (`test.py:153`).
- Recall: `test.py:167-174` `for k in [1, 10, 50]`; hit iff `test_targets_id.index(test_queries[i]['target_img_id']) in nns[:k]`; denominator = query count (`r = 100*r/len(nn_result)`).
- Score: `test.py:141-143` `sims = np.matmul(all_queries, all_imgs); sims = sims.squeeze(); sims = sims.max(-1)`.

Workbench side: `src/evaluation/fashioniq.py:163-165` declares `fashioniq_full_gallery_ref_excluded` with `gallery="full_split"`, `reference_excluded=True`; `build_fashioniq_gallery` (`:192`) returns `build_original_gallery` (`:109`) = the ordered split file; `recall_at_k_reference_excluded` (`:35`) removes the reference column (`ranking = ranking[ranking != reference_column]`) and searches the target. `macro_average_fashioniq` (`:223-241`) macro-averages per category and sets `mean_recall = (recall_at_10 + recall_at_50)/2`.

**Verdict: VERIFIED** for the protocol id, gallery scope, ordering, reference policy and K-set — every element traces to a line in the pinned evaluator. **VARIANT** on the exclusion *mechanism*: the source masks to `-10e10` and sorts; the workbench removes the column; the workbench's own docstring says the two are equivalent only while every other score clears the mask (`src/evaluation/fashioniq.py:41-48`). Equivalent cohort, different arithmetic — worth recording, not a correctness defect.

## DATASET CONTRACT
- `fashioniq_layout: fashioniq_ilearn_resized`, `fashioniq_image_extensions: ['.jpg']` (`models.yaml:298-299`). Layout path list: `workbench/backend/fashioniq_layout.py:25-35` — `captions/`, `image_splits/`, `resized_image/`, `resized_image/{dress,shirt,toptee}`, `cap.{cat}.val.json`, `correction_dict_{cat}.json`, `split.{cat}.val.json`. Image directory map: `fashioniq_layout.py:85-86` `"fashioniq_ilearn_resized": ("resized_image", True)`.
- Source opens exactly that: `datasets.py:31` `image_dir = self.path + 'resized_image'`, `:32` `split_dir`, `:33` `caption_dir`; `datasets.py:139` `os.path.join(self.image_dir, name.split('_')[0], name.split('_')[1] + ".jpg")` — per-category `.jpg`, matching the declaration.
- `preparation_contract: manual_ilearn_resized` (`models.yaml:290`) is a **layout name, not a registered contract**: `workbench/registry/preparation_contracts.yaml` has no `intent` entry (`grep 'preparation_id:'` → clvc_net, dcnet, clip4cir, tgcir, sprc, limn only), so `preparation_contract_for_model` (`registry.py:187-194`) returns `None` via the absent `preparation_id` and preflight falls back to `fashioniq_required_paths` → `paths_for_layout` (`registry.py:679-683`). `registry.py:73` permits the literal `manual_ilearn_resized`, so this validates.
- Local data root `/home/heheboiz/data/cir/data/fashionIQ_dataset`: `captions/`, `image_splits/`, `images/` (74381 files), `qwen_captions/` present; `resized_image/` **absent**; `correction_dict_*.json` **absent** (`ls data/fashionIQ_dataset/captions` → only `cap.*.json`). `READINESS_MATRIX.md:30` records exactly those 7 missing paths.
- OBSERVED split statistics (command: `/usr/bin/python3 -c` over the local JSON): dress gallery 3817 unique / 2017 queries; shirt 6346 / 2038; toptee 5373 / 1961; every target and every candidate is in its split, and each split is duplicate-free. This matches `workbench/docs/reviews/models/csmcir.md:44` (3817/6346/5373) and the workbench assertions `len(set(gallery_ids)) == gallery_count` (`src/evaluation/fashioniq.py:112,121`).
- **`correction_dict_{dress,shirt,toptee}.json` have no pinned provenance.** `AUXILIARY_ASSET_AUDIT.md:358-360` classifies all three `OFFICIAL_ASSET_UNAVAILABLE`; `auxiliary_assets.yaml` has **zero `intent` entries** (`grep -c 'model_id: intent'` → 0), so nothing downloads them; `UPSTREAM_AUDIT.md:23` records only DQU-CIR's Git blob IDs as a lookalike source that no inspected iLearn source names.

**Verdict: BLOCKED** (data prerequisites for the native path are absent locally), severity **P1** for the missing dictionaries because the caption string is part of the query.

## PREPROCESSING CONTRACT
- Transform: `train_INTENT.py:79-91` forces `transform = "targetpad"`, `input_dim = 224`, `target_ratio = 1.25` → `data_utils.py:92` `targetpad_transform(1.25, 224)`; pipeline at `data_utils.py:101-105` = `Resize(dim, BICUBIC)` → `TargetPad` → `ToTensor` → CLIP `Normalize((0.48145466,0.4578275,0.40821073),(0.26862954,0.26130258,0.27577711))`.
- Evaluation path applies only this transform: `datasets.py:149-157` `get_test_img` opens `PIL.Image`, `convert('RGB')`, `self.transform(img)` — no random op.
- Caption composition at eval: `datasets.py:170-177` reads `cap.{cat}.val.json`, `correction_dict_{cat}.json`, then `mod_str = self.concat_text(caption['captions'], correction_dict)`; `correct_text` (`datasets.py:110-115`) lowercases, maps `string.punctuation`→space, tokenizes, exact dictionary substitution; `concat_text` (`:117-119`) joins with `" and "`.

**Verdict: VERIFIED** for the deterministic TargetPad@1.25/224 evaluation transform.

## AUGMENTATION CONTRACT
- Registry declares no augmentation policy for `intent`; `evaluation_noise_pct: 0` for all three variants (`models.yaml:315,325,335`).
- Evaluation-side augmentation: none. `test.test` (`test.py:9`) calls `model.eval()` (`test.py:10`), and the eval feature extractors are decorated `@torch.no_grad()` (`blip2_cir.py:721,771`). The only image-level operation on the eval path is the deterministic transform above.
- Training-only machinery (never reached at eval): `colorful_spectrum_mix` (`blip2_cir.py:57`, called at `:445` inside `forward(..., after_warmup=False)`), `text_proj_cau` (`:171`), CKA/discrimination loss helpers (`blip2_cir.py:360-367`).

**Verdict: VERIFIED** (training-only).

## GALLERY CONTRACT
- Full split, file order, no sort, no dedup beyond the file's own uniqueness: `datasets.py:204-205` `for id, image_name in enumerate(images): test_targets_id.append(id)`.
- Reference excluded per query at ranking time, not by shrinking the gallery: `test.py:149-150`.
- Workbench equivalent: `src/evaluation/fashioniq.py:163-165` + `:192-205` + `:35-77`.

**Verdict: VERIFIED.** Note the source builds the full-gallery image list with `get_img` (`datasets.py:208` → `:138`, which applies `self.transform` only `if self.transform`, and does not force it), while the `val-split` branch would use `self.get_img` too (`:201`). No behavioral difference at eval because the transform is always set by `load_dataset`.

## METRIC CONTRACT
- Source returns `('{name}_r{k}', r)` for k in `[1,10,50]` (`test.py:167-174`), i.e. per-category R@1/R@10/R@50, each `mean over queries of [target in top-k] × 100`.
- Checkpoint selection metric is a **sum** of per-category R@10 (`train_INTENT.py:286` `current_score = current_score + t[1][1]`) — index `[1]` is R@10 because the list is ordered 1,10,50. The trainer never computes a macro.
- Workbench emits macro R@10/R@50 and `mean_recall` (`src/evaluation/fashioniq.py:223-241`), and `PROTOCOL_AUDIT.md`/`FASHIONIQ_PROTOCOL_PARITY.md` finding #4 states the macro is a workbench aggregate, not a source-reported metric. **That statement is correct** and I independently confirm the source has no macro.

**Verdict: VARIANT** on the mean identity only (see below); source per-category metrics themselves are VERIFIED.

## SOURCE-TO-WORKBENCH PARITY
| Element | Pinned source | Workbench | Verdict |
|---|---|---|---|
| Protocol id | `datasets.py:203` + `test.py:150` | `fashioniq_full_gallery_ref_excluded` (`models.yaml:301`; `src/evaluation/fashioniq.py:163`) | VERIFIED |
| Gallery | ordered `split.{cat}.val.json` (`datasets.py:204-205`) | `build_original_gallery` (`src/evaluation/fashioniq.py:109-123`) | VERIFIED |
| Reference rule | mask `-10e10` then argsort (`test.py:150,153`) | remove column then argsort (`src/evaluation/fashioniq.py:69-73`) | VARIANT (equivalent cohort) |
| K set | `[1,10,50]` (`test.py:168`) | `recall_at_10`, `recall_at_50` (`src/evaluation/fashioniq.py:80-107`) | VERIFIED |
| Preprocessing | targetpad 1.25/224 (`train_INTENT.py:87`) | declared `fashioniq_ilearn_resized` (.jpg, per-category) | VERIFIED |
| Evaluation command | `test.test` is a callable, no CLI (`test.py` has no `argparse`/`__main__`) | `IntentAdapter` (`workbench/backend/adapters/models.py:69-71`) inherits `OfficialScriptAdapter.command`, which raises `NotImplementedError("upstream evaluation command requires audit before execution")` (`models.py:24-25`) | NOT_RUN |
| Executable model implementation | full BLIP-2 Q-Former INTENT model | none: `src/models/` holds only `iag_srme`, `taper_mag`; no workbench implementation of INTENT exists | BLOCKED |

`run_eval.py:25` would therefore raise for `--model intent`, since `IntentAdapter().build_command()` calls the inherited stub.

## RUNTIME STATUS
**NOT_RUN / BLOCKED**, and no runtime claim is made anywhere in this document. Concretely:
1. No evaluation-ready path with the pinned source: `train_INTENT.py` trains for `--num_epochs 20` before evaluating (`train_INTENT.py:271-284`), has **no** checkpoint-restore code, and the official weights cannot be attached to it. OBSERVED: `grep -rn restore_file train_INTENT.py test.py cirr_sub_BLIP2.py` → no match, although `AUXILIARY_ASSET_AUDIT.md:203` claims `train_INTENT.py` "defines argument `--restore_file`". **The document is wrong; the registry-affecting fact (no restore path) is right.** (`--restore_file` exists in HABIT/train.py:79, ConeSep/train.py and AirKnow/train_BLIP2.py — this looks like cross-model contamination in the checklist.)
2. `cirr_sub_BLIP2.py` is CIRR-only: `datasets.CIRR(path="/root/autodl-tmp/cirr_data/CIRR/")` (`:122`) and it sweeps `torch.load` over a hardcoded `/root/autodl-tmp/BLIP/test/` directory (`:124-129`). It is not a FashionIQ evaluator. The README's documented `python cirr_sub_BLIP2.py --checkpoint_path … --output_file …` (`README.md:188-190`) **passes arguments the script has no parser for**.
3. Data assets missing locally: `resized_image/` and `correction_dict_*.json` (see DATASET CONTRACT).
4. Runtime assets: `salesforce-lavis==1.0.2` in the declared env (`models.yaml:286`) is only *part* of the story — the pinned checkout shadows it with its own in-tree `lavis/` (`36` tracked `lavis/` files, 975-line `lavis/models/blip2_models/blip2_cir.py`), and that tree hardcodes `/root/bert-base-uncased` (`lavis/models/blip2_models/blip2.py:32,48,55`) and downloads EVA-ViT-g from a Google URL (`lavis/models/eva_vit.py:442`). `auxiliary_assets.yaml` contains **no `intent` entries**, so none of these are declared, checked, or fetched for INTENT.
5. Consequence: the checkpoint identity, the reported metrics, and any numeric reproduction are **UNKNOWN**; no GPU/MPS execution was attempted (none available).

## PAPER-METRIC STATUS
Source of the numbers: `workbench/third_party/INTENT/figs/fiq.png`, "Table 2", the `0%` block, row `INTENT (Ours)` (read as an image; the paper's LaTeX source is not in the checkout and the arXiv badge in `README.md` says "Coming.Soon"). Observed cells:

- Dress R@10 50.32, R@50 72.10; Shirt 56.32 / 74.93; Toptee 59.28 / 78.45.
- Average R@10 **55.31**, R@50 **75.16**, AVG **65.24**.
- Non-zero blocks: 20% (R@10 54.22 / R@50 74.49 / AVG 64.36), 50% (52.52 / 73.32 / 62.92), 80% (48.51 / 69.63 / 59.07).

Registry `reported_r10: 55.31`, `reported_r50: 75.16`, `reported_mean: 65.24` (`models.yaml:303-305`) **match the paper's Average row exactly** (STATIC_AUDIT against the figure).

Mean identity (arithmetic, OBSERVED via `/usr/bin/python3`):
- Per-category papers values → macro R@10 = (50.32+56.32+59.28)/3 = 55.3067 → **55.31**; macro R@50 = (72.10+74.93+78.45)/3 = **75.16**. The paper's Average row *is* the macro of its own per-category columns. ✔
- `(55.31 + 75.16)/2 = 65.235`, which rounds half-up to **65.24** ✔ — so the AVG column is the arithmetic mean of the two average recalls.
- Workbench `macro_average_fashioniq` computes the same rule (`src/evaluation/fashioniq.py:239` `"mean_recall": (recall_at_10 + recall_at_50)/2`) but on **unrounded** inputs: 55.30666…, 75.16 → `mean_recall = 65.2333…`. A reproducing run would write `65.24` only after rounding to 2 dp, giving 65.23. The 0.005 gap is a rounding convention, **P3**.
- R@1 is computed by the source (`test.py:168`) but is **not** published in Table 2 (which reports R@10/R@50 only) and is **not** in the registry.

None of these numbers has been observed in a run here: they are paper values, not local reproductions.

## CRITICAL FINDINGS

1. **The pinned FashionIQ evaluator cannot complete as written.** `INTENT/lavis/models/blip2_models/blip2_cir.py:769` returns `fusion_feats.unsqueeze(1).unsqueeze(1)` → query tensor `(B, 256, 1, 1)`; `:792` returns `image_features.permute(0, 2, 1)` → gallery tensor `(B, 256, 32)`. Then `test.py:141` `np.matmul(all_queries, all_imgs)` contracts `all_queries`' last axis (1) against `all_imgs`' second axis (256). Observed with the real shapes: `np.matmul` on `(2017,256,1,1) @ (3817,256,32)` → `ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 256 is different from 1)`. The comment `#all_queries.dot(all_imgs.T)` at `test.py:141` describes the intended contraction. **Severity P0 for any claim that this evaluator produces the paper's FashionIQ numbers, and for the registry protocol mapping itself** — the protocol document describes a gallery/reference/sort order in a function that raises before `argsort` (`test.py:153`) is reached. Mirror copies exist at `test.py:321,323` (CIRR val path). *Caveat, stated honestly:* the siblings Air-Know (`test_BLIP2.py:142,144`), ConeSep (`test.py:59,61`) and HABIT (`test.py:141,143`) carry the same 4-D/3-D pairing, so either the authors never exercised the FashionIQ branch of the published trees, or an unshipped higher-dimensional-collapse step sat between extraction and `matmul`. I did not observe a passing run, so the "unreachable" verdict is **STATIC_AUDIT on tensor shapes plus one observed numpy error**, which is as strong as it gets without a GPU.

2. **No clean FashionIQ checkpoint exists, and the paper's headline row (0% noise) has no published weights.** Registry `strict_clean_checkpoint_available: false` (`models.yaml:307`) is correct; the Hub publishes only CIRR-0.2 and FashionIQ-0.2/0.5/0.8. Since `evaluation_noise_pct: 0` is required by the project and enforced at load time (`workbench/backend/registry.py:87-88` raises on any non-zero value), the category `fashioniq`/`strict_clean` is empty for `intent` by construction — `tests/test_checkpoint_selection.py:128-133` asserts INTENT is blocked under `STRICT_CLEAN`. **Severity P1**: the model can be only ever evaluated with a noise-trained checkpoint, which is a different model than the one the 0% row reports.

3. **`preparation_contract: manual_ilearn_resized` names no contract, so the declaration is inert.** The required `correction_dict_{dress,shirt,toptee}.json` and `resized_image/{cat}/*.jpg` are not generated, not downloaded, not hash-verified, and in the local root do not exist. `AUXILIARY_ASSET_AUDIT.md:358-360` classifies them `OFFICIAL_ASSET_UNAVAILABLE`. **Severity P1** for reproducibility: the caption string is a model input, and its provenance is unknown.

4. **The declared evaluation command does not exist.** `models.yaml` registers the model as `reproduction_status: NOT_RUN` with no `command_status`; `IntentAdapter` (`adapters/models.py:69-71`) has no `command()` override, so `build_command` raises `NotImplementedError` (`models.py:24-25`). The README's own invocation (`python train_INTENT.py`) trains first, and the registered Hub weights cannot be restored into that script. **Severity P1** (already reflected in the registry as `NOT_RUN`, but the registry does not record *why* — no `source_blocker`).

5. **The noise machinery itself is training-only and cannot leak into evaluation inputs** — see MINOR FINDING 2 for the line-by-line proof. This is the one thing the project's `evaluation_noise_pct=0` requirement actually needs, and it holds.

## MINOR FINDINGS

1. **Registry `notes` understates the checkpoint situation.** `models.yaml:308`: "Official Hub lists only FashionIQ 20%, 50%, and 80% noise-trained weights." True but incomplete: the Hub also ships `intent_cirr_0.2.pt` (observed in the API sibling list), and the registry's own `checkpoint_variants` record only the three FashionIQ files. P3, cosmetic for provenance, but the note reads as if the three are all the author published.
2. **Evaluation-time noise injection: impossible on the paths that matter.** `FashionIQ.shuffle()` (`datasets.py:63-92`) is the only caption/target/reference corruptor; it is invoked from `train_INTENT.py:101` (immediately after `load_dataset()` at `:95-100`) and **not** from `test.test` or from `datasets.FashionIQ.__init__`. `test.test` receives `trainset` (`train_INTENT.py:284`) and reads only `test_queries_*`/`test_targets_*`, which are produced by `get_test_data` (`datasets.py:166-210`) from `cap.{cat}.val.json` with no shuffling. Inside `shuffle()`, `noise_ratio == 0` returns before touching data (`datasets.py:65-67`). The eval feature extractors are `@torch.no_grad()` (`blip2_cir.py:721,771`) and never call `colorful_spectrum_mix` (only `forward` at `:445` does). Registry `evaluation_noise_pct: 0` is therefore an accurate description of the evaluator, **not** just a policy. VERIFIED (STATIC_AUDIT). Residual risk: if the `fashion_iq_data.json` + `test_queries_*.pkl` cache (`datasets.py:37-61`) were produced by an earlier noisy run and shipped, the val queries would still be clean (they come from `cap.*.val.json`), but `self.fashioniq_data` would carry injected mismatches; the source does not validate cache freshness (`tgcir` has the same caveat, `preparation_contracts.yaml:219`). P3.
3. **`encode('utf-8').decode('utf-8')`** at `test.py:111` (utf-8 path) and the Fashion200k branch at `test.py:31` are noise the INTENT path does not need. P3, no action.
4. **`train_INTENT.py:19` `import clip`** — the `clip` package is imported but never used in that file (`grep -n '^clip\.|clip\.load|clip\.tokenize' train_INTENT.py` → nothing). It is documented nowhere in the README's dependency list. P3 (import-time failure mode for a clean env), matters only if the trainer is ever invoked.
5. **`requirements.txt` does not exist**, though `README.md:123` tells the user to `pip install -r requirements.txt`. OBSERVED: `ls workbench/third_party/INTENT/requirements.txt` → No such file; `git ls-files | grep -i require` → empty. The env declaration in `models.yaml:279-286` is the only usable source, and its `confidence: VERIFIED_README` is correct, because the README's pip line (`README.md:119`) lists exactly the registry's five pinned packages. P2 for anyone provisioning from the README alone.
6. **`--restore_file` claim in `AUXILIARY_ASSET_AUDIT.md:203` is wrong for INTENT** (no such flag; `grep` → 0 matches, while HABIT/train.py:79 has one). The substantive claim ("no checkpoint loading routine exists") is correct. P3, doc hygiene.
7. **`README.md` links the arXiv badge to `https://arxiv.org/abs/coming soon`** (`README.md`, badge block) and the README clone URL is the *old* org name `ZivChen-Ty/INTENT`; both are cosmetic, but the registry's `paper_url` is OJS and can be cross-checked only via Crossref right now (OJS fetch returned `000`).
8. **`workbench/backend/adapters/models.py:138` line-length** and the missing blank line before `ADAPTERS` — P3, no action.
9. **Registry `environment.cuda: '12.6'`** (`models.yaml:285`) is generic: the upstream README says CUDA 12.1 wheels (`README.md:116`). P3, does not affect correctness.
10. **`literature_split_label: original`** for INTENT is correct against the source default branch, but the pinned `datasets.py` additionally ships a `val-split` branch (`:191-202`). The registry does not record that second branch; `FASHIONIQ_PROTOCOL_PARITY.md` finding #2 does. P3, documentation completeness.

## UNRESOLVED EVIDENCE

- **No run of any kind.** No checkpoint downloaded, no weights loaded, no FashionIQ inference, no GPU. Every metric statement above is from the paper figure or from arithmetic on paper numbers.
- **Checkpoint payload identity**: whether `intent_fiq_0.2/0.5/0.8.pt` encode the stated training noise levels, and whether they load into the pinned `Blip2QformerCir` (`lavis/models/blip2_models/blip2_cir.py:115-176`), is UNKNOWN. No digest is published anywhere I could read (`expected_sha256: null`; the HF API listing was fetched without `?blobs=true`, so no `lfs.sha256` was observed).
- **Whether the FashionIQ branch ever ran upstream**: UNKNOWN. The static shape analysis says no, in the pinned tree; the paper table says the numbers exist. Both can be true (results from an unshipped internal revision). I cannot adjudicate without the authors' private code.
- **`test_targets_id` for the iLearn-family FashionIQ branch**: `test.py:146-148` builds `test_targets_id` from `test_targets[...]['target_img_id']`, which for `original-split` is `range(len(split))` (`datasets.py:205-207`) — so `test_targets_id.index(...)` is the identity map. That is consistent, but it means the query→gallery lookup at `test.py:150` is index-based, not name-based. Confirmed STATIC_AUDIT; cannot be confirmed at runtime.
- **OJS page content**: fetch returned status `000`; the registry's `paper_url` is unvalidated by direct HTTP here (Crossref DOI matching is the substitute evidence).
- **GitHub remote HEAD vs pin**: not compared (would require a clone/fetch; no network clone performed).

## FINAL VERDICT
**BLOCKED** — the registry's protocol, gallery, reference rule and metric definitions are faithfully traceable to the pinned `INTENT` source (and the paper's Average row confirms `55.31 / 75.16 / 65.24`), but the pinned FashionIQ evaluator raises a shape error before ranking, no clean FashionIQ checkpoint exists, the required correction dictionaries and resized images are absent and unpinned, and `IntentAdapter` has no executable command — so nothing can be reproduced, and by the rules of this audit "unreproducible" is BLOCKED, not VARIANT.

---

# EXECUTION TRACE

`paper/README/config → source dataset → preprocessing → official evaluator → workbench invocation → checkpoint → output → metric comparison`

1. **paper/README/config →** OJS `paper_url` fetch failed (`curl -w %{http_code}` → `000`); Crossref confirms title/DOI `10.1609/aaai.v40i25.39181`. `README.md:1` reproduces the registry title (`models.yaml:291`). Metrics come from `figs/fiq.png` Table 2 (image read). Environment: `README.md:112,116,119` (`python=3.9`, `torch==2.1.0`, `open-clip-torch==2.24.0 scikit-learn==1.3.2 transformers==4.25.0 salesforce-lavis==1.0.2 timm==0.9.16`) == registry `models.yaml:279-286`. `README.md:123` references a `requirements.txt` that does not exist (`ls` → No such file). **Evidence: LIVE fetch + grep.**
2. **→ source dataset:** `datasets.FashionIQ` (`datasets.py:25`) with `path = args.fashioniq_path` (`train_INTENT.py:95-100`), reading `cap.{cat}.{train,val}.json` (`datasets.py:96,170`), `split.{cat}.val.json` (`:168`), `correction_dict_{cat}.json` (`:98,172`), `resized_image/{cat}/{id}.jpg` (`:139,150`). Local root has captions/splits/images but **no** `resized_image/` and **no** correction dicts (observed `ls`); readiness matrix line 30 lists the same 7 missing paths. **Evidence: `ls` + OBSERVED json stats (3817/6346/5373 galleries).**
3. **→ preprocessing:** `train_INTENT.py:79-91` fixed `targetpad`, `input_dim 224`, `target_ratio 1.25` → `data_utils.py:92-105`; eval applies it once in `get_test_img` (`datasets.py:149-157`). Captions normalized in `datasets.py:110-119`. **Evidence: source lines.**
4. **→ official evaluator:** `train_INTENT.py:283-284` → `test.test` (`test.py:9`); gallery `datasets.py:203-209`; mask `test.py:150`; rank `test.py:153`; recall `test.py:167-174`. Delivery **fails earlier**: `np.matmul` on `(M,256,1,1)@(N,256,32)` → observed `ValueError` (numpy 2.3.5). **Evidence: code lines + one observed command.**
5. **→ workbench invocation:** `IntentAdapter` (`adapters/models.py:69-71`) has no `command()`; inherited stub raises `NotImplementedError` (`adapters/models.py:24-25`); `run_eval.py:25` would raise; no workbench INTENT model exists (`src/models/` → `iag_srme`, `taper_mag` only). **Evidence: file listing.**
6. **→ checkpoint:** registry URLs (`models.yaml:312,322,332`) all return HTTP `200` with `x-linked-size 2722478522` (observed `curl -sIL`); registry `expected_size_bytes`/`expected_sha256` are `null`; **nothing downloaded**; zero local artifacts. **Evidence: LIVE HTTP headers.**
7. **→ output:** none. No run, no result JSON, no ranking, no per-query export. **Evidence: `workbench/artifacts/` empty for intent; no run_id.**
8. **→ metric comparison:** paper's Average row (image read) `R@10 55.31`, `R@50 75.16`, `AVG 65.24`; registry `models.yaml:303-305` reproduces them verbatim; arithmetic `(50.32+56.32+59.28)/3 = 55.307` and `(72.10+74.93+78.45)/3 = 75.16` reproduces the Average row from the same table, and `(55.31+75.16)/2 = 65.235` reproduces `65.24`. Workbench macro rule (`src/evaluation/fashioniq.py:239`) is the same function on unrounded inputs → `65.2333`, i.e. `65.23` at 2 dp. **No local metric was produced**, so the comparison is paper-vs-registry only.

---

# CHECKLIST

Format: `N. short form — verdict — evidence — severity — action`.

1. Paper title matches registry — VERIFIED — `models.yaml:291` == `INTENT/README.md:1` — NONE — none.
2. Paper URL resolves — UNKNOWN — `curl -sL -o /dev/null -w "%{http_code}" https://ojs.aaai.org/index.php/AAAI/article/view/39181` → `000`; Crossref confirms title/DOI `10.1609/aaai.v40i25.39181` — P3 — re-check OJS when reachable.
3. Upstream repo URL identity — VERIFIED — GitHub API `iLearn-Lab/AAAI26-INTENT`, `default_branch main`, `fork false`; `curl -L` of `github.com/ZivChen-Ty/INTENT` → `iLearn-Lab/AAAI26-INTENT` — NONE — none.
4. Pinned sha matches registry — VERIFIED — `models.yaml:295` == `git -C workbench/third_party/INTENT rev-parse HEAD` — NONE — none.
5. Checkout HEAD equals pin — VERIFIED — same command, output `2a75fc6f57d1a036bf3167a7fe1d2cee072f51f3` — NONE — none.
6. Tracked tree clean — VERIFIED — `git diff --name-only HEAD | grep -v pycache` → empty — NONE — none.
7. Untracked non-source in checkout — VERIFIED (benign) — `git ls-files --others --exclude-standard` → only `__pycache__/*.pyc`; `runtime.py is_bytecode()` excludes them — NONE — none.
8. `source_dir` present and consistent — VERIFIED — `source_available: true` + `source_dir: INTENT` (`models.yaml:296-297`), directory exists with `test.py`, `datasets.py`, `train_INTENT.py`, `lavis/` — NONE — none.
9. `method_name` matches README — VERIFIED — `INTENT` (`models.yaml:287`) == README title token — NONE — none.
10. Checkpoint filenames match Hub — VERIFIED — API siblings list `intent_fiq_0.2.pt`, `intent_fiq_0.5.pt`, `intent_fiq_0.8.pt` == `models.yaml:311,321,331` — NONE — none.
11. Checkpoint URLs resolve — VERIFIED — `curl -sIL` → `200` × 3 — NONE — none.
12. Published checkpoint sizes — VERIFIED (remote header only) — `x-linked-size: 2722478522` for all three; registry `null` — P3 — optional: record size in registry (a local computation stays local).
13. Published digests — UNKNOWN — no `lfs.sha256` observed (API fetched without `?blobs=true`); registry `null` — P2 — fetch `?blobs=true` and populate, or leave null and state it.
14. Registry `expected_size_bytes` — VERIFIED as `null` — `models.yaml:316,326,336` — NONE — none.
15. Registry `expected_sha256` — VERIFIED as `null` — `models.yaml:317,327,337` — NONE — none.
16. Noise coverage 20/50/80 — VERIFIED — Hub siblings + `models.yaml:314,324,334` — NONE — none.
17. Clean 0% FashionIQ checkpoint — VERIFIED absent — Hub has only `intent_cirr_0.2.pt` + 3 FIQ noise files; Hub README names no clean file — P1 — mark the 0% row as non-reproducible from published weights.
18. `strict_clean_checkpoint_available` — VERIFIED (correct value) — `models.yaml:307` false; `tests/test_checkpoint_selection.py:128-133` — NONE — none.
19. Registry `notes` claim — VARIANT — `models.yaml:308` omits `intent_cirr_0.2.pt` — P3 — extend note.
20. `checkpoint_mapping_status` — VERIFIED (metadata-level) — `models.yaml:319,329,339`; `UPSTREAM_DATA_PREPARATION_AUDIT.md:159`; payload identity UNKNOWN — P2 — mark payload identity explicitly unverified.
21. `native_protocol` id — VERIFIED — `models.yaml:301` == `src/evaluation/fashioniq.py:163-165` — NONE — none.
22. Source default split matches protocol — VERIFIED — `datasets.py:26` and `train_INTENT.py:61` default `'original-split'`; branch `datasets.py:203` — NONE — none.
23. Source has a second gallery branch — VERIFIED — `datasets.py:191-202` `val-split` (dead under default) — P3 — registry could note the branch (parity doc already does).
24. Gallery = full validation split — VERIFIED — `datasets.py:204-205` — NONE — none.
25. Gallery ordering = file order — VERIFIED — `enumerate(images)` over `split.{cat}.val.json`; workbench `build_original_gallery` (`src/evaluation/fashioniq.py:109-123`) — NONE — none.
26. Gallery uniqueness — VERIFIED — OBSERVED 3817/6346/5373 unique; `src/evaluation/fashioniq.py:112,121` assert it — NONE — none.
27. Reference exclusion present — VERIFIED — `test.py:150` — NONE — none.
28. Exclusion mechanism equivalence — VARIANT — mask `-10e10` (`test.py:150`) vs column removal (`src/evaluation/fashioniq.py:69-73`); docstring `:41-48` concedes the dependence on the score bound — P3 — none (equivalent cohort).
29. Query universe from `cap.{cat}.val.json` — VERIFIED — `datasets.py:170-187` — NONE — none.
30. Query→gallery index mapping — VERIFIED — `datasets.py:181,183` `images.index(...)`; `test_targets_id == range(len(split))` under `original-split` — NONE — none.
31. Recall@K = target in top-k — VERIFIED — `test.py:170-173` — NONE — none.
32. K values {1,10,50} — VERIFIED — `test.py:168` — NONE — none.
33. Sorting — VARIANT — `np.argsort(-sims[i,:])[:50]` (`test.py:153`) is not guaranteed stable; workbench `torch.argsort` documented as tie-ambiguous (`src/evaluation/fashioniq.py:135-140`) — P3 — none.
34. Score path shape — BLOCKED — `test.py:141` `np.matmul` on `(B,256,1,1)@(B,256,32)`; OBSERVED `ValueError` — **P0** — must be resolved by the authors or patched+documented before any FashionIQ number is cited.
35. Mean identity — VERIFIED with a rounding edge — paper `(55.31+75.16)/2 = 65.235 → 65.24`; workbench same rule unrounded → `65.2333` (`src/evaluation/fashioniq.py:239`) — P3 — note rounding convention.
36. R@10 = 55.31 — VERIFIED vs paper Average column — `figs/fiq.png` 0% row; `models.yaml:303`; macro of 50.32/56.32/59.28 = 55.3067 — NONE — none.
37. R@50 = 75.16 — VERIFIED vs paper Average column — `figs/fiq.png`; `models.yaml:304`; macro of 72.10/74.93/78.45 = 75.16 — NONE — none.
38. mean = 65.24 — VERIFIED vs paper AVG — `figs/fiq.png`; `models.yaml:305` — NONE — none.
39. Per-category paper values — VERIFIED — `figs/fiq.png` 0% block — NONE — none.
40. R@1 published — UNKNOWN — source computes it (`test.py:168`) but Table 2 has no R@1 column; registry has none — P3 — none.
41. Dataset layout matches source paths — VERIFIED — `fashioniq_layout.py:25-35` vs `datasets.py:31-33,139,150` — NONE — none.
42. Image extensions `.jpg` — VERIFIED — `models.yaml:299`; `datasets.py:139,150` hardcodes `.jpg`; `fashioniq_layout.py:85-86` — NONE — none.
43. Correction dictionaries present locally — BLOCKED — `ls data/fashionIQ_dataset/captions` → no `correction_dict_*.json`; required at `datasets.py:98,172`; `es` `fashioniq_layout.py:32` — P1 — obtain author-pinned files or do not evaluate.
44. Correction dictionary provenance — BLOCKED — `AUXILIARY_ASSET_AUDIT.md:358-360` `OFFICIAL_ASSET_UNAVAILABLE`; `auxiliary_assets.yaml` has 0 `intent` entries; `UPSTREAM_AUDIT.md:23` — P1 — record as unverified blocker (already partially done).
45. Local split sizes/uniqueness — VERIFIED — OBSERVED dress 3817/2017, shirt 6346/2038, toptee 5373/1961, all unique — NONE — none.
46. Target & reference in gallery — VERIFIED — OBSERVED `target_in_gal`/`cand_in_gal` == query count per category — NONE — none.
47. Preprocessing transform — VERIFIED — `train_INTENT.py:79-91`, `data_utils.py:92-105` — NONE — none.
48. Eval-time transform deterministic — VERIFIED — `datasets.py:149-157`; `model.eval()` (`test.py:10`); `@torch.no_grad()` (`blip2_cir.py:721,771`) — NONE — none.
49. Training-only intervention machinery — VERIFIED — `colorful_spectrum_mix` (`blip2_cir.py:57,445`), `text_proj_cau` (`:171`) reached only through `forward` — NONE — none.
50. Noise injection scope — VERIFIED (training only) — `shuffle()` (`datasets.py:63-92`) called only at `train_INTENT.py:101`; no call from `test.test` — NONE — none.
51. `evaluation_noise_pct: 0` enforced — VERIFIED — `models.yaml:315,325,335`; `workbench/backend/registry.py:87-88` raises otherwise — NONE — none.
52. Noise can leak into eval inputs — VERIFIED not to, with one caveat — `datasets.py:65-67` returns at ratio 0; val queries from `cap.*.val.json` (`:170`) are never shuffled; cache files (`:37-61`) are unvalidated for freshness — P3 — record freshness contract if a cache is ever shipped.
53. BLIP-2 base weights and hardcoded paths — BLOCKED — `lavis/configs/models/blip2/blip2_pretrain.yaml:10`, `lavis/models/eva_vit.py:442`, `/root/bert-base-uncased` at `lavis/models/blip2_models/blip2.py:32,48,55`; no `intent` entries in `auxiliary_assets.yaml` — P1 — declare these assets per model or state they are operator-supplied.
54. Environment declaration — VERIFIED — `models.yaml:279-286` == `README.md:112,116,119`; `cuda '12.6'` differs from README's 12.1 wheels — P3 — align `cuda` or mark it approximate.
55. Evaluator invocation / adapter / runnable status — BLOCKED — `IntentAdapter` (`adapters/models.py:69-71`) inherits the `NotImplementedError` stub (`:24-25`); `test.test` has no `__main__`/`argparse`; no restore code in `train_INTENT.py`; README's `train_INTENT.py` invocation trains first — P1 — no executable command exists; keep `reproduction_status: NOT_RUN` and record the reason (no `source_blocker` is currently set).



FINAL VERDICT: BLOCKED

---

---

## REFRESH (this engagement)

Appended after the work that landed on top of this audit. Earlier findings above are left
as written; where one has been superseded, the superseding fact is stated here.

- **Static benchmark reference.** 4 committed row(s):

  - `intent_aaai26_table2_fiq_n00` — `VERIFIED`, 0% training noise, protocol `fashioniq_full_gallery_ref_excluded`, values `{'r10': 55.31, 'r50': 75.16, 'mean': 65.24}`, provenance `{'r10': 'PAPER_REPORTED', 'r50': 'PAPER_REPORTED', 'mean': 'PAPER_REPORTED'}`
  - `intent_aaai26_table2_fiq_n20` — `VERIFIED`, 20% training noise, protocol `fashioniq_full_gallery_ref_excluded`, values `{'r10': 54.22, 'r50': 74.49, 'mean': 64.36}`, provenance `{'r10': 'PAPER_REPORTED', 'r50': 'PAPER_REPORTED', 'mean': 'PAPER_REPORTED'}`
  - `intent_aaai26_table2_fiq_n50` — `VERIFIED`, 50% training noise, protocol `fashioniq_full_gallery_ref_excluded`, values `{'r10': 52.52, 'r50': 73.32, 'mean': 62.92}`, provenance `{'r10': 'PAPER_REPORTED', 'r50': 'PAPER_REPORTED', 'mean': 'PAPER_REPORTED'}`
  - `intent_aaai26_table2_fiq_n80` — `VERIFIED`, 80% training noise, protocol `fashioniq_full_gallery_ref_excluded`, values `{'r10': 48.51, 'r50': 69.63, 'mean': 59.07}`, provenance `{'r10': 'PAPER_REPORTED', 'r50': 'PAPER_REPORTED', 'mean': 'PAPER_REPORTED'}`

- **Readiness now:** `BLOCKED_SOURCE` — source cannot run: no runnable evaluation path is registered for this model: the pinned evaluator is a callable with no CLI or checkpoint loader, the adapter deliberately declines to build a command, and the required correction dictionaries and resized images have no published source; the paper row is the 0%-noise row and no 0% checkpoint is published (see workbench/docs/reviews/models for the model's audit). An earlier note in this registry claimed a tensor-shape failure in the similarity computation; that claim was WRONG and is withdrawn - np.matmul broadcasts and the ranking matrix has the shape the indexing expects

- **Reference per registered checkpoint condition:**

  - `fiq_n02` (noise 20): `intent_aaai26_table2_fiq_n20` `VERIFIED` — exact condition match
  - `fiq_n05` (noise 50): `intent_aaai26_table2_fiq_n50` `VERIFIED` — exact condition match
  - `fiq_n08` (noise 80): `intent_aaai26_table2_fiq_n80` `VERIFIED` — exact condition match

- **Gate:** the run's verdict comes from `workbench/backend/paper_gate.py` against the
  committed row above, reported in the run JSON under `benchmark_gate`. A row that states no
  condition is refused (`BLOCKED_REFERENCE_INCOMPLETE`); a metric the paper never printed is
  compared only if the registry derives it, and then it is labelled `WORKBENCH_DERIVED`.
