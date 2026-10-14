# habit

**Audit type:** independent adversarial reproducibility review. **Method:** read-only static audit of `/home/heheboiz/data/cir` at `d9431fe9632b67a329771cc06567487b591afb4b`; pinned checkout `workbench/third_party/HABIT` at `6ddc2cd63eac1f65693f4e821c34fb3b42aee306`. No GPU run, no install, no download of weights, no repo write. Every claim below is **STATIC_AUDIT** unless marked otherwise.

---

## MODEL ID

`habit` — registry entry `workbench/registry/models.yaml:216-277`; `method_name: HABIT` (`:225`). Adapter `HabitAdapter` at `workbench/backend/adapters/models.py:64-66` (`script = "test.py"`), registered in `ADAPTERS` at `:138`.

## PAPER TITLE

`HABIT: Chrono-Synergia Robust Progressive Learning Framework for Composed Image Retrieval`.

- Registry `workbench/registry/models.yaml:229-230`: `paper_title` above, `paper_url: https://ojs.aaai.org/index.php/AAAI/article/view/37608`.
- Corroborated: `read https://ojs.aaai.org/index.php/AAAI/article/view/37608` failed (`socket connection was closed unexpectedly`), but `web_search` returned that URL as the paper landing page with the identical title, and `read https://arxiv.org/abs/2604.18037` returned title "HABIT: Chrono-Synergia Robust Progressive Learning Framework for Composed Image Retrieval", authors Zixu Li, Yupeng Hu, Zhiwei Chen, Shiqi Zhang, Qinlei Huang, Zhiheng Fu, Yinwei Wei, published 2026-04-20. DOI `10.1609/aaai.v40i8.37608` (search result `dl.acm.org/doi/10.1609/aaai.v40i8.37608`).
- Upstream README (`workbench/third_party/HABIT/README.md:44`) links the same OJS article id 37608.
- Title match: **VERIFIED**. OJS page body itself not fetched — noted.

## PINNED SOURCE COMMIT

- Registry: `upstream_commit_sha: 6ddc2cd63eac1f65693f4e821c34fb3b42aee306` (`workbench/registry/models.yaml:233`), `source_dir: HABIT` (`:235`).
- Checkout: `git -C workbench/third_party/HABIT rev-parse HEAD` → `6ddc2cd63eac1f65693f4e821c34fb3b42aee306`; last commit `Zixu Li | Update README.md`.
- Runtime check executed: `runtime.source_clean_and_pinned(model, Path('workbench/third_party/HABIT'))` → `(True, '6ddc2cd63eac1f65693f4e821c34fb3b42aee306', None)`.
- `git status --porcelain` shows only untracked `__pycache__/` trees (40 entries), which `runtime.is_bytecode` excludes (`workbench/backend/runtime.py:1829-1838`).
- Upstream tree contains only 8 tracked top-level files (`git ls-files` top level: `LICENSE, README.md, cirr_test_submission.py, data_utils.py, datasets.py, test.py, train.py, utils.py`) plus `assets/` and vendored `lavis/`.
- Remote: `origin git@github.com:iLearn-Lab/AAAI26-HABIT.git`.
- Note: `workbench/third_party/` is git-ignored (`.gitignore:51`); `git ls-files workbench/third_party | wc -l` = `0`, so the pin is enforced only by `runtime`, not by git. Observed.

## CHECKPOINT IDENTITY

Registry declares three variants, all `source_type: author_huggingface`, `evaluation_noise_pct: 0`, `status: NO_CLEAN_CHECKPOINT`, `checkpoint_mapping_status: VERIFIED_METADATA`, no size, no sha256 (`workbench/registry/models.yaml:247-277`):

| checkpoint_id | filename | training noise | url |
|---|---|---|---|
| `fiq_n02` | `HABIT-FIQ_N0.2.pt` | 20% | `.../resolve/main/fiq/HABIT-FIQ_N0.2.pt` |
| `fiq_n05` | `HABIT-FIQ_N0.5.pt` | 50% | `.../resolve/main/fiq/HABIT-FIQ_N0.5.pt` |
| `fiq_n08` | `HABIT-FIQ_N0.8.pt` | 80% | `.../resolve/main/fiq/HABIT-FIQ_N0.8.pt` |

Independently checked against the Hub (`read https://huggingface.co/api/models/iLearn-Lab/AAAI26-HABIT?blobs=true`, repo sha `8712993b071f4668ce5b2db6070e70c856bd9174`). The Hub has **exactly six** weight files and **no 0%-noise file**:

- `fiq/HABIT-FIQ_N0.2.pt` — 2,730,715,930 B, lfs sha256 `814d2d0ceb29ef910f37562e8bea8a76cf38aa546fbc2102e495df687c276b03`
- `fiq/HABIT-FIQ_N0.5.pt` — 2,730,715,930 B, lfs sha256 `8a0d3bd27cf46d8334f9596404d89ebf803da0fa65236d89815bb3929b9a4216`
- `fiq/HABIT-FIQ_N0.8.pt` — 2,730,715,930 B, lfs sha256 `763885de6907bca4f017b336ed7ec3740cb9305c5adb6a135da45c158ddb9717`
- `cirr/HABIT-CIRR-N0.2.pt`, `N0.5.pt`, `N0.8.pt` — 2,736,022,950 B each.

Conclusions: filenames and noise labels agree with the registry; **no clean (0%) FashionIQ checkpoint is published**; the registry's `NO_CLEAN_CHECKPOINT` labelling and `strict_clean_checkpoint_available: false` (`workbench/registry/models.yaml:245`) are truthful. The Hub LFS sha256/size values exist but are **not** recorded in the registry (`expected_sha256: null`, `expected_size_bytes: null` at `:254-255,264-265,274-275`), so `checkpoint_availability` reports `official_sha256_known: False, checkpoint_downloaded: False` (observed command). Local staging `workbench/artifacts/checkpoints/habit/` does not exist.

Upstream README's own FIQ table (paper Table 1; rendered `workbench/third_party/HABIT/assets/results-fiq.png`, sha256 `f373fb1a7c425b95889fbac6fb1136c7c01e550aa27e6fba3ebfea10f40f64ef`, 567,158 B) has a 0% column, and `README.md` states `--noise_ratio` "default options evaluated are `0.0`, `0.2`, `0.5`, `0.8`" — i.e. a 0% model was trained and reported but its weights are not released. Paper §4.1 (arXiv HTML) confirms training noise is simulated "by randomly selecting a subset of training triplets according to a noise ratio".

## EVALUATION PROTOCOL

Registry: `native_protocol: fashioniq_full_gallery_ref_excluded`, `supported_protocols: [fashioniq_full_gallery_ref_excluded]`, `literature_split_label: original` (`models.yaml:238-240`). Protocol table entry (`models.yaml:19-22`): label "FashionIQ - Original Full Gallery, Reference Excluded", `literature_split_label: original`, `evaluation_noise_pct: 0`.

Source, verified:

- Split selection: `FashionIQ.__init__(..., split='original-split')` default — `workbench/third_party/HABIT/datasets.py:28`. Launcher passes no split — `train.py:108-113`.
- Gallery construction, `original-split` branch enumerates the **whole** validation split in file order: `datasets.py:205-211` (`for id, image_name in enumerate(images): test_targets_id.append(id)`); the `val-split` pair-union branch (`:193-204`) is not taken.
- Captions from `cap.<cat>.val.json`, images from `resized_image/<cat>/<id>.jpg` — `datasets.py:170-186`, `get_test_img` at `:151-159`, `get_img` at `:140-149`.
- Scoring: `sims = np.matmul(all_queries, all_imgs)`; `sims.squeeze()`; `sims.max(-1)`; per-query L2 normalization is **commented out** — `test.py:134-143`.
- Reference exclusion: `for i, t in enumerate(test_queries): sims[i, test_targets_id.index(t['source_img_id'])] = -10e10` — `test.py:147-151`, i.e. **mask before ranking**.
- Ranking: `np.argsort(-sims[i, :])[:50]` — `test.py:154` and `:168`.
- Recall: `for k in [1, 10, 50]` with `test_targets_id.index(test_queries[i]['target_img_id']) in nns[:k]`, `r = 100 * r / len(nn_result)` — `test.py:169-175`.
- `model.eval()` at `test.py:10`; feature extraction under `torch.no_grad()` (`test.py:97`) via `extract_retrieval_compose` (`:112`) and `extract_retrieval_target` (`:128`).
- No `argparse` and no `__main__` in `test.py`: AST parse of `workbench/third_party/HABIT/test.py` yields only `load_obj`, `test`, `test_cirr_valset`. The only `__main__` in the tree that loads a checkpoint is `cirr_test_submission.py:94`, and it does `model, ... = load_model_and_preprocess(name="Blip2QformerCir", ...)` at `:97-100` — a model name that is **not registered** in the pinned `lavis` (only `@registry.register_model("HABIT")` at `lavis/models/blip2_models/HABIT.py:106`). So the CIRR submission path is broken and there is no FashionIQ CLI at all.

## DATASET CONTRACT

- Registry layout `fashioniq_ilearn_resized` (`models.yaml:236`), extensions `[".jpg"]` (`:237`).
- Expected paths — `workbench/backend/fashioniq_layout.py:25-34`: `captions/`, `image_splits/`, `resized_image/`, `resized_image/{dress,shirt,toptee}/`, `captions/cap.<cat>.val.json`, `captions/correction_dict_<cat>.json`, `image_splits/split.<cat>.val.json`. Layout dispatch `:68-69`, image-directory mapping `:85`.
- Source requires additionally, on a cache miss, `cap.<cat>.train.json` and `correction_dict_<cat>.json` (`datasets.py:96-101`), and *writes* `fashion_iq_data.json` + six PKLs (`datasets.py:39-63`) which later runs blindly trust (`:55-63`).
- Observed local state (`ls /home/heheboiz/data/cir/data/fashionIQ_dataset/`): only `captions/`, `image_splits/`, `images/`, `qwen_captions/`. There is **no `resized_image/`** and **no `correction_dict_*.json`** anywhere under `/home/heheboiz/data/cir` except in the `LIMN` and `TGCIR` checkouts (`find . -name 'correction_dict*'`), which are different methods' copies and must not be borrowed.
- `preflight.model_requirements(habit_model, resolve_config())` observed 8 blocks: `resized_image` + its three category dirs missing, three `correction_dict_*.json` missing, and `FashionIQ image extension requirement: image directory missing: .../resized_image`.
- Local captions/splits are present and are the standard FashionIQ counts (dress val 2017 annotations / 3817 gallery images; shirt 2038/6346; toptee 1961/5373), `images/` holds 74,381 `.png` files — i.e. the canonical `.png` layout, not the iLearn `resized_image/<cat>/*.jpg` layout the pinned evaluator opens (`datasets.py:141,152`).
- `preparation_contract: manual_ilearn_resized` (`models.yaml:228`); `preparation_contracts.yaml` has **no** `habit` row (`grep habit registry/preparation_contracts.yaml` → no match; contract ids are only `clvc_net_fashioniq, dcnet_fashioniq, clip4cir_fashioniq, tgcir_fashioniq, sprc_fashioniq, limn_fashioniq`). So `preflight` falls through to the layout-derived path list (`workbench/backend/preflight.py:27-30`), which is consistent with the declared manual contract but carries **no raw-input/generated-artifact provenance** for habit.

## PREPROCESSING CONTRACT

- Text: dataset-side `correct_text` lowercases, maps every `string.punctuation` char to space, tokenizes, substitutes via the correction dict (`datasets.py:112-117`); `concat_text` composes **source order** `"{cap0} and {cap1}"` (`:119-121`). Then the evaluator applies `txt_processors["eval"]` (`test.py:111`), i.e. `BlipCaptionProcessor.pre_caption` (`lavis/processors/blip_processors.py:49-68`): lowercase, replace only `[.!"()*#:;~]`, collapse whitespace, strip, truncate to 50 words.
- Tokenizer: `BertTokenizer.from_pretrained("/root/bert-base-uncased")` + `[DEC]` special token (`lavis/models/blip2_models/blip2.py:32-34`); Q-Former rebuilt from `BertConfig`/`BertLMHeadModel.from_pretrained("/root/bert-base-uncased")` (`:48,54-56`). Absolute host path — portability blocker, mirrored in `registry/audits/cluster_b_ilearn.yaml:591,604`.
- Tokenization length: `max_length=self.max_txt_len` default 32, padding max, truncation — `HABIT.py:389-395`; `max_txt_len` default in `HABIT.py:136` and `from_config` `:588`.
- Image: `targetpad_transform(1.25, 224)` selected in `train.py:94-106`; definition `data_utils.py:92-106` = `TargetPad(1.25,224)` → `Resize(224, BICUBIC)` → `CenterCrop(224)` → RGB → `ToTensor` → CLIP Normalize `(0.48145466,0.4578275,0.40821073)/(0.26862954,0.26130258,0.27577711)`.
- Base weights: LAVIS `blip2_pretrained.pth` from `lavis/configs/models/blip2/blip2_pretrain.yaml:10`, loaded by `load_checkpoint_from_config` (`lavis/models/base_model.py:81-101`) from `HABIT.from_config` (`HABIT.py:601`). `cache_root` is the author's absolute cluster path `/export/home/.cache/lavis` (`lavis/configs/default.yaml:10`), so weights staged elsewhere are not found — `docs/FIRST_GPU_EXPERIMENT.md:91-99`.
- Workbench composer `src/datasets/fashioniq.py:93-…` (`normalize_fashioniq_caption`) reproduces the punctuation-lowercase-dictionary behaviour, but per `docs/SOURCE_TO_WORKBENCH_PARITY_MATRIX.md:29` the caption stage for `habit` is only `EQUIVALENT_WITH_EVIDENCE` at the pre-LAVIS string boundary; I did not find a workbench callsite that uses it for habit (the adapter routes to the native evaluator, which owns the full pipeline). Not independently re-derived claim → see UNRESOLVED.

## AUGMENTATION CONTRACT

- `data_utils.py` defines only deterministic eval transforms; `grep -n "Random\|augment\|Flip\|Jitter"` over `data_utils.py train.py datasets.py` matched only `Compose/Resize/CenterCrop/ToTensor/Normalize` imports and two `CenterCrop(dim)` lines (`data_utils.py:85,102`). No random augmentation on either loader.
- Noise injection is training-only: `train.py:108-113` builds `datasets.FashionIQ(..., noise_ratio=args.noise_ratio)` then calls `trainset.shuffle()`; `shuffle()` rewrites only `self.fashioniq_data` (`datasets.py:65-94`), which is populated solely from `cap.<cat>.train.json` (`datasets.py:96-110`). `noise_ratio == 0` returns immediately (`:67-69`).
- Therefore the registered `evaluation_noise_pct: 0` for all three variants is consistent with the code: evaluation reads `cap.<cat>.val.json` and never calls `shuffle()`.
- Paper corroboration: noise is injected into **training** triplets (§3.1 / §4.1).

## GALLERY CONTRACT

- Gallery = full ordered validation split for the category (`datasets.py:205-211`); query set = all val annotations (`:177-189`), each carrying `source_img_id` and `target_img_id` as indices into that split (`:183,185`).
- Reference removal: masked to `-10e10` before `argsort` (`test.py:147-151`), so hidden danger is only if another score ≤ −1e10.
- Workbench cohort helper `src/evaluation/fashioniq.py:35-73` `recall_at_k_reference_excluded` **physically removes the reference column** and ranks the remainder (`ranking = ranking[ranking != reference_column]`), and its docstring at `:40-48` explicitly declares this is not the same operation as masking. `build_fashioniq_gallery` (`:192-210`) picks `full_split` for `fashioniq_full_gallery_ref_excluded` via `FASHIONIQ_PROTOCOLS` (`:150-163`), with `reference_excluded=True` (`:154-156`).
- Dispatch check: `evaluate_fashioniq(` has **no production callsite** — independent regex sweep over all non-`third_party` `*.py` found only `src/evaluation/fashioniq.py:242` (the definition) and `tests/unit/test_fashioniq_dispatch.py:1,107,303`. So this helper does not currently evaluate habit; the registered route is the native `test.py` via the adapter.
- Registry docs agree with the source on mask-before-ranking and full-split default: `docs/FASHIONIQ_PROTOCOL_PARITY.md:72`, `docs/UPSTREAM_AUDIT.md:11,27`, `docs/audits/PHASE2_CROSS_REVIEW.md:74,92` — all independently re-checked against `test.py:150-154` and `datasets.py:27-38,193-212`: **SUPPORTED**.

## METRIC CONTRACT

- Source metric set is `R@1, R@10, R@50` **per category** (`test.py:169-175`), returned as `<cat>_r<k>` percentages (`:175`, name from `:81-86`). No corpus-level macro or mean is computed anywhere in the source.
- Paper metric set for FashionIQ is `R@10, R@50` per category plus `Average R@10, R@50, AVG.` (arXiv HTML Table 1; also rendered in `assets/results-fiq.png`). Paper §4.1 says "We adopt Recall@K as the primary metric. For CIRR, we report R@{1,5,10,50}" — so the FIQ column set R@10/R@50 is what the workbench registers.
- Registry values: `reported_r10: 55.38`, `reported_r50: 75.20`, `reported_mean: 65.29` (`models.yaml:241-243`).
- Independent arithmetic (exact numbers, not the rounded ones):
  - Paper table HABIT 0% row: dress 49.99/72.38, shirt 56.62/74.68, toptee 59.51/78.53.
  - `(49.99+56.62+59.51)/3 = 55.373333` → **55.37** at 2dp, paper prints **55.38** (Δ 0.0067).
  - `(72.38+74.68+78.53)/3 = 75.196667` → **75.20** (Δ 0.0033). ✓
  - `(55.38+75.20)/2 = 65.29` exactly; `(55.373333+75.196667)/2 = 65.285` → **65.29** at 2dp (Δ 0.005).
  - Mean identity holds under both readings.
- Workbench recomputation agrees with this construction: `workbench/backend/metrics_extraction.py:148-152` computes `computed_r10_macro = (d_10+s_10+t_10)/3.0`, `computed_r50_macro` likewise, `computed_mean = (r10_macro+r50_macro)/2.0`, and enforces agreement with a declared aggregate inside `0.05` (`:169`). `macro_average_fashioniq` (`src/evaluation/fashioniq.py:223-240`) returns `mean_recall = (recall_at_10 + recall_at_50)/2`.
- Consequence: the paper's published `AVG.` column is not the 2dp average of its own printed per-category values for R@10 (55.37 vs 55.38); the registered 55.38 is a legitimate paper value but is internally inconsistent at 0.01 with the same table. Neither number can be reproduced without the 0% checkpoint.

## SOURCE-TO-WORKBENCH PARITY

| Aspect | Source | Workbench | Verdict |
|---|---|---|---|
| model_id / method | HABIT `README.md:1` | `habit` / `HABIT` | match |
| pin | `git rev-parse HEAD` | registry `:233` | match, runtime-verified `True` |
| protocol | full ordered split + mask before ranking (`datasets.py:205-211`, `test.py:150-154`) | `fashioniq_full_gallery_ref_excluded` (`:239-240`) | **EQUIVALENT_WITH_EVIDENCE** on semantics; the workbench *cohort helper* uses removal, not masking (`src/evaluation/fashioniq.py:35-73`) — see MINOR F-2 |
| gallery | full val split, source order | `build_original_gallery`/`build_fashioniq_gallery` full_split (`:110-118,192-210`) | equivalent |
| metric names | `R@1/10/50` (`test.py:169-175`) | registry reports R@10/R@50/mean | subset, consistent with paper's FIQ columns |
| caption stage | native LAVIS processor + BERT tokenizer (`test.py:111`, `blip2.py:32`) | workbench routes to native evaluator | `EXTERNAL_ASSET_REQUIRED` — matches `docs/SOURCE_TO_WORKBENCH_PARITY_MATRIX.md:29` |
| image stage | `targetpad_transform(1.25,224)` (`data_utils.py:92-106`) | native | equivalent at this boundary |
| evaluator command | `test.py:test()` called from `train.py:283`; no CLI | `HabitAdapter.command` **not overridden** → abstract (`backend/adapters/models.py:64-66`, `:24-25`) | **NOT IMPLEMENTED**; matches parity matrix "NOT IMPLEMENTED" and observed `command not runnable: COMMAND_UNAUDITED` |
| workbench replay script | — | `workbench/replay/` contains only `limn.py`, `tgcir.py` | absent for habit |

Cross-doc hypothesis checks (each re-checked against code, not taken on trust):

- `docs/audits/PHASE2_CROSS_REVIEW.md:74` "HABIT mask before ranking — `test.py:150-154`": **SUPPORTED** (`test.py:150-154`, exactly).
- `docs/audits/PHASE2_CROSS_REVIEW.md:92` "constructor defaults `split='original-split'`; launcher supplies no split; `train.py:108-113`": **SUPPORTED** (`datasets.py:28`, `train.py:108-113`). One nuance: the global argparse dataset default is CIRR, not FashionIQ (`train.py:45`), so the FIQ path only runs with `--dataset fashioniq`.
- `registry/audits/cluster_b_ilearn.yaml:561` "HABIT/test.py:1-176 near byte-identical to INTENT/test.py, differing by one blank line at HABIT/test.py:145": **SUPPORTED** — `diff <(head -176 HABIT/test.py) <(head -176 INTENT/test.py)` → `145d144 <` and `176a176 > return out`, exit 1.
- `registry/audits/cluster_b_ilearn.yaml:561` "Differs from AirKnow by setting `model.eval()` at line 10 (AirKnow calls `model.train()`), lacking outer `with torch.no_grad():` indentation": **SUPPORTED** — `AirKnow/test_BLIP2.py:9-11` is `def test(...): model.train(): / with torch.no_grad():` while `HABIT/test.py:10` is `model.eval()`.
- `docs/UPSTREAM_AUDIT.md:109` `HABIT → /export/home/.cache/lavis`: **SUPPORTED** (`lavis/configs/default.yaml:10`).
- `docs/AUXILIARY_ASSET_AUDIT.md:175-181` (no standalone eval entrypoint; `--restore_file` never implemented; no checkpoint-loading routine for the Hub files; only noisy weights published; `/root/bert-base-uncased` hardcoded; correction dicts unpublished): **SUPPORTED** — AST check on `test.py`; `--restore_file` declared at `train.py:79` and never referenced again (`grep` finds only the declaration); `grep` for `load_checkpoint` in `train.py` finds nothing, only `HABIT.load_checkpoint_from_config` (`HABIT.py:601`, base weights); Hub listing above; `blip2.py:32,48,55`; `git ls-files | grep correction` in HABIT is empty.

## RUNTIME STATUS

- `reproduction_status: NOT_RUN` (`models.yaml:244`) — consistent with everything observed.
- `runtime.environment_blockers(habit)` → `[]` (observed), because `environment_required` is not set for this model (`workbench/backend/runtime.py:1805-1808`); `WORKBENCH_HABIT_PYTHON` is not set in the environment (`python -c` check → `False`) and `workbench/envs/` contains only a `README.md` and `csmcir-import.txt`. So the declared environment (`python 3.8.10`, `torch 2.1.0`, `torchvision 0.16.0`, `cuda 12.6`, `packages: open-clip-torch==2.24.0, scikit-learn==1.3.2, transformers==4.25.0, salesforce-lavis==1.0.2, timm==0.9.16`) is **rated `VERIFIED_README` but not provisioned**, and the empty blocker list is a consequence of the missing `environment_required` flag, not of readiness.
- Full `preflight.preflight_blockers(habit, resolve_config())` observed: the 8 dataset blocks above **plus** `command not runnable: COMMAND_UNAUDITED`.
- `HabitAdapter().build_command(...)` raises `FileNotFoundError: checkpoint missing: /home/heheboiz/data/cir/workbench/artifacts/checkpoints/habit/HABIT-FIQ_N0.5.pt` (observed) — the checkpoint-presence guard fires before whatever command would have been built.
- `checkpoint_availability` for all three: `checkpoint_downloaded: False`, `official_sha256_known: False`, `adapter_command_ready: False`, `runtime_verified: False`, `runnable: False`.
- Independent checkpoint selection, executed:
  - `select_checkpoints(habit, policy='strict_clean')` → `BLOCKED / BLOCKED_CHECKPOINT`, reason "no checkpoint is recorded as clean-trained under policy strict_clean: fiq_n02=20%, fiq_n05=50%, fiq_n08=80%", `strict_clean_checkpoint_available: False`.
  - `select_checkpoints(habit, policy='include_flagged_conditions')` → also `BLOCKED / BLOCKED_CHECKPOINT_SELECTION` ("more than one source-backed variant is registered and none is declared a bundle"), because habit has no `checkpoint_bundles`.
  - An explicit `requested=['fiq_n05']` would return `SELECTED` with `policy_overridden=True` (`backend/checkpoint_selection.py:138-157`); not executed, inferred from code.
- No GPU run attempted and none possible here. Every runtime statement above is STATIC_AUDIT or a CPU-only library call.

## PAPER-METRIC STATUS

- The registered `55.38 / 75.20 / 65.29` are the paper's **0% noise** (fully-supervised) HABIT row, confirmed against the arXiv HTML Table 1 and the repo's `assets/results-fiq.png`.
- That row corresponds to a checkpoint trained with `--noise_ratio 0.0` — a setting the README says exists, but **whose weights are not published** (Hub listing has no `N0.0` file).
- Therefore the paper metric is **not reproducible with any registered checkpoint**, and there is no way to compute a meaningful `observed − paper` delta: the only obtainable weights would be trained under 20/50/80% noise while the paper row is 0%.
- Paper-metric status: **NOT REPRODUCIBLE (no clean published weights) — STATIC_AUDIT**. No runtime metric was produced; none is claimed.

## CRITICAL FINDINGS

**C-1 (P0) — No obtainable checkpoint can reproduce the registered paper row.** The registered `reported_r10/r50/mean` are the 0%-noise row (Table 1, 55.38/75.20/65.29), while the Hub publishes only 20/50/80% weights (HF API listing; registry `:247-277`). Evaluating `fiq_n05` and diffing against `55.38` would silently compare a 50%-noise-trained network to a 0%-noise-trained network. Evidence: `workbench/registry/models.yaml:241-277`; `read https://huggingface.co/api/models/iLearn-Lab/AAAI26-HABIT?blobs=true`; `assets/results-fiq.png`. Action: keep `strict_clean` refusal; if an operator explicitly requests `fiq_n0X`, the report must carry `checkpoint_training_noise_pct` in the same sentence as the delta. Currently `workbench/backend/autopilot.py:816-826` builds `measured_difference` labelled only `observed - paper` with no condition field, so an overridden run would emit an unqualified delta.

**C-2 (P0 for this model) — The declared evaluation command does not exist.** `HabitAdapter` at `workbench/backend/adapters/models.py:64-66` names `script = "test.py"` but does not override `command`; the base class raises/refuses (`:24-25`), and `preflight` observed `command not runnable: COMMAND_UNAUDITED`. The pinned `test.py` has no `argparse` and no `__main__` (AST: only `load_obj`, `test`, `test_cirr_valset`), and its single caller is the training loop (`train.py:283`), which also requires a training set. Action: implement an audited replay entrypoint (as done for LIMN in `workbench/replay/limn.py`) that constructs the model, loads a checkpoint, builds the full-split gallery, and reproduces `test.py:110-176` without importing `train.py`.

**C-3 (P1) — Dataset contract unsatisfied locally, and one input has no assigned provenance.** `preflight` observed `resized_image/` and all three `captions/correction_dict_*.json` missing. The correction dictionaries are `OFFICIAL_ASSET_UNAVAILABLE` (`docs/AUXILIARY_ASSET_AUDIT.md:355-357`; `git ls-files | grep correction` in HABIT is empty) and the workbench has **no auxiliary-asset record for habit** (`registry/auxiliary_assets.yaml` model ids: `clip4cir_rn50x4_fullft, clvc_net, combiner_rn50x4_noft, csmcir, sprc`). Action: stage `resized_image/<cat>/*.jpg` from the official FashionIQ `start_kit` resize step and record the correction-dictionary source explicitly; until then, any capsule would be unreproducible and partly unattributed.

## MINOR FINDINGS

**F-1 (P2) — The registry protocol id for habit is not the mirrored official FashionIQ protocol, despite `literature_split_label: original`.** `fashioniq_full_gallery_ref_excluded` is labelled "Original Full Gallery, Reference Excluded" (`models.yaml:19-22`), while the actual FashionIQ original-split protocol includes the reference image. Renaming the workbench cohort would not change semantics, but the label invites a reader to treat it as the mirrored benchmark. Action: keep the id (it is accurate) and ensure report badges never render it as the official protocol.

**F-2 (P2) — Workbench cohort helper and pinned evaluator perform different operations.** `src/evaluation/fashioniq.py:35-73` removes the reference column; `test.py:150-151` masks to `-10e10`. The helper's own docstring (`:40-48`) says so. Since `evaluate_fashioniq` has no production callsite (regex sweep: only `src/evaluation/fashioniq.py:242` + tests), this cannot affect a habit result today — but it will the moment habit is routed through the helper. Action: if habit is ever evaluated through the helper, either document the sentinel-bound assumption or use mask semantics.

**F-3 (P2) — `train.py` `--restore_file` is dead, and the only published-checkpoint loader points at an unregistered model name.** `--restore_file` is declared (`train.py:79`) and never read. `cirr_test_submission.py:97` requests `"Blip2QformerCir"`, which `lavis` does not register (`HABIT.py:106` registers `"HABIT"`; `grep -rn Blip2QformerCir lavis/` → no match). Evidence: `grep`/AST. Action: an audited replay must construct `HABIT` directly; do not reuse `cirr_test_submission.py`.

**F-4 (P2) — Hardcoded absolute assets.** `/root/bert-base-uncased` (`blip2.py:32,48,55`) and `cache_root: "/export/home/.cache/lavis"` (`lavis/configs/default.yaml:10`). Both are required at evaluation and both point at author hosts. Evidence: grep + `docs/FIRST_GPU_EXPERIMENT.md:91-99`. Action: provision BERT and BLIP-2 under the declared paths or patch the config (documented edit), never silently.

**F-5 (P2) — Recorded shared-file hashes in the cluster audit are wrong.** `registry/assets/cluster_b3_ilearn.yaml:13,19` (and the same values at `registry/audits/cluster_b_ilearn.yaml:347,353`) record `data_utils.py sha256 c6a90c6d7a57a8bf6ff00874ff0b497042a98f121d5c21f7dbbe03ec417430d8` and `utils.py sha256 694a5081a6eb6408d66e746a5d42188ff6e6cf1e4a5bf4d289aa841ee135cf50`. Recomputed over the four sibling checkouts (AirKnow, ConeSep, HABIT, INTENT): `data_utils.py` → `c6a90c6d7a57bbcc9604e0fb2b24875289909a6d8b727dfcb11597660e1843ca` and `utils.py` → `694a5081a6eb81ef29607d50d9a8c78335a59e43968b5e9e58ff84421833a85f`. The *sibling identity* claim is TRUE (all four files are byte-identical, 18,058 B and 4,576 B); the digests are wrong. Also `utils.py` is recorded as `lines: 145` while `wc -l` = 144, and `blip2_pretrain.yaml` is described as "Identical 38 lines" while `wc -l` = 36. My SHA-256 derivation matches the workbench's own `sha256_file` rule (raw bytes; `backend/registry.py` `_update_file_digest`), so the expected value is the observed one. Action: recompute and re-record, or drop the digests. Severity P2 because a checksum that nothing verifies is still a false statement in an evidence registry.

**F-6 (P3) — Audit YAML cites the wrong line range for the eval feature extractors.** `registry/audits/cluster_b_ilearn.yaml:822` says features are extracted by `HABIT.extract_retrieval_compose()` / `extract_retrieval_target()` "(lavis/models/blip2_models/HABIT.py:126-191)". Those functions are at `HABIT.py:372-413` and `:416-436`; `126-191` is the `__init__` signature/`robust_infoNCE`. Action: correct the range.

**F-7 (P3) — Paper's AVG. cell is not the 2dp mean of its own printed cells.** 0%-row R@10 AVG printed `55.38`; the 2dp mean of 49.99/56.62/59.51 is `55.37`. R@50 (75.20) and AVG mean (65.29) reproduce. Δ=0.0067, benign, but the registered number inherits the paper's rounding. Action: none required; note it when a parity tolerance is chosen (the workbench already tolerates 0.05 at `metrics_extraction.py:169`).

**F-8 (P3) — `environment_required` is unset for habit, so `environment_blockers` returns `[]` for an environment that does not exist.** `WORKBENCH_HABIT_PYTHON` is unset and `workbench/envs/` holds no habit environment; observed `environment_blockers(habit) == []`. Action: set `environment_required` on this model so the missing interpreter is reported rather than silently absent.

## UNRESOLVED EVIDENCE

1. **Exact checkpoint→noise mapping beyond the filename.** The Hub filename convention (`HABIT-FIQ_N0.5.pt` ⇒ 50% training noise) is corroborated by the README's `--noise_ratio` options and by the registry's `VERIFIED_METADATA`, but **checkpoint file bytes were never inspected** (2.7 GB each; no download performed). UNKNOWN whether a file's internal config records its noise level. The registry's `checkpoint_mapping_status: VERIFIED_METADATA` rests on filename + Hub metadata, not on weights.
2. **Whether the published 0% metrics were produced by `train.py` at `--noise_ratio 0.0` with this exact pinned code.** Not verifiable statically; the paper states 0% ≡ fully-supervised, and the code has a `noise_ratio == 0` early return (`datasets.py:67-69`), but no run log or seed is published.
3. **OJS landing-page body / AAAI camera-ready PDF.** `read` of `https://ojs.aaai.org/index.php/AAAI/article/view/37608` failed (socket closed); I relied on the arXiv v1 HTML (`2604.18037v1`) and the repo's own rendered table image, which agree with each other on every HABIT number. Whether the AAAI version's table differs is UNKNOWN.
4. **`resized_image` generation provenance.** The upstream README's directory listing is inconsistent (it shows `FashionIQ/{dress,shirt,toptee}` without `resized_image`, while `datasets.py:33` requires `resized_image/<cat>/<id>.jpg`); no resize script is tracked in the HABIT tree. The container `docs/AUXILIARY_ASSET_AUDIT.md:169` attributes it to FashionIQ `start_kit resize_images.py`; I did not verify that mapping against the upstream tree (it is not present there). UNKNOWN.
5. **Whether `qwen_captions/` in the local dataset is ever consumed by habit.** No reference to `qwen` in the pinned tree (not checked exhaustively). UNKNOWN; irrelevant to the registered command.
6. **Whether `sims.squeeze().max(-1)` (`test.py:141-143`) is correct for all gallery sizes.** I verified the shape algebra for a generic case (`(N,1,1,D) @ (G,D,T) → (N,G,T) → max → (N,G)`), so `G=1` collapse is not a concern for FashionIQ's 3,817–6,346-image galleries. Pathological `N=1` was not needed. Not a finding.

---

## EXECUTION TRACE

`paper/README/config → source dataset → preprocessing → official evaluator → workbench invocation → checkpoint → output → metric comparison`

1. **paper/README/config → protocol claim.** Paper (arXiv 2604.18037v1, §4.1) adopts Recall@K; Table 1 gives FIQ R@10/R@50 + AVG per noise ratio, with HABIT 0% = 49.99/72.38, 56.62/74.68, 59.51/78.53 → 55.38/75.20/65.29. README `workbench/third_party/HABIT/README.md` documents `--noise_ratio {0.0,0.2,0.5,0.8}` and prints the same table via `assets/results-fiq.png` (sha256 `f373fb1a…64ef`). Evidence: `read https://arxiv.org/html/2604.18037v1`; `workbench/third_party/HABIT/README.md`; `assets/results-fiq.png`. **STATIC_AUDIT, verified for the numbers' provenance.**
2. **→ source dataset.** `FashionIQ.__init__(path, category, transform, noise_ratio, split='original-split')`, `image_dir = path + 'resized_image'`, `caption_dir = path + 'captions'`, `split_dir = path + 'image_splits'` (`datasets.py:28-38`). Needs `cap.<cat>.{train,val}.json`, `correction_dict_<cat>.json`, `split.<cat>.val.json`, `resized_image/<cat>/*.jpg` (`:96-101,168-175,140-159`). Local: captions + splits present; `resized_image/` and correction dicts absent; `preflight` emits 8 blocks. **BLOCKED at this arrow.**
3. **→ preprocessing.** dataset-side `correct_text`/`concat_text` (`datasets.py:112-121`) → evaluator `txt_processors["eval"]` (`test.py:111`) → `BlipCaptionProcessor.pre_caption` (`blip_processors.py:49-68`) → BERT tokenizer at `/root/bert-base-uncased`, max 32 (`blip2.py:32-34`, `HABIT.py:389-395`); images via `targetpad_transform(1.25, 224)` (`train.py:94-106`, `data_utils.py:92-106`). **STATIC_AUDIT, path fully traced in code; cannot execute (arrow 2 blocked, no GPU, `/root` asset absent).**
4. **→ official evaluator.** `test.py:test(params, model, testset, category, txt_processors)`; `model.eval()` (`:10`); compose features (`:112`), target features (`:128`), `np.matmul` + `squeeze` + `max(-1)` (`:141-143`), reference mask `-10e10` (`:150-151`), `argsort(-sims)` (`:168`), R@1/10/50 (`:169-175`). Only caller is `train.py:283` inside `train_and_evaluate`. No `__main__`, no `argparse`. **STATIC_AUDIT, verified.**
5. **→ workbench invocation.** `HabitAdapter(model_id='habit', script='test.py')` (`backend/adapters/models.py:64-66`) inherits the abstract `command` (`:24-25`); `preflight.preflight_blockers` observed `command not runnable: COMMAND_UNAUDITED`; `build_command` observed to raise `FileNotFoundError: checkpoint missing: .../artifacts/checkpoints/habit/HABIT-FIQ_N0.5.pt` before reaching command construction. **BROKEN at this arrow.**
6. **→ checkpoint.** Registry `fiq_n02/fiq_n05/fiq_n08` (`models.yaml:247-277`) match Hub paths exactly; Hub has no clean file; nothing staged locally (`checkpoint_downloaded: False`); no sha256/size recorded. **BLOCKED at this arrow (no clean variant; C-1).**
7. **→ output.** No invocation → no report, no per-query ranks, no metrics. `reproduction_status: NOT_RUN` (`models.yaml:244`) is accurate. **NOT_RUN.**
8. **→ metric comparison.** No observed metric exists, so no comparison was performed and none is claimed. The only comparison machinery for FIQ aggregates is `workbench/backend/metrics_extraction.py:148-190` (recompute macro + mean, tolerance 0.05) and `backend/autopilot.py:816-826` (`observed − paper`); a habit run cannot reach either. **NOT_RUN.**

**Trace verdict:** arrows 1 and 2-partial verified, 3–4 verified statically but unexecutable, 5–8 blocked.

---

## CHECKLIST

Format: `N. question — verdict — evidence — severity — action`.

1. Registry entry exists and parses — VERIFIED — `workbench/backend/registry.py:load_registry` + `python3 -c` load of `registry/models.yaml` (habit block `:216-277`); `load_registry` validation passes — NONE — none.
2. `model_id`/`method_name` exact — VERIFIED — `models.yaml:216,225` (`habit`, `HABIT`) — NONE — none.
3. Paper title matches the real paper — VERIFIED — `models.yaml:229` vs `read https://arxiv.org/abs/2604.18037` title (identical string) — NONE — none.
4. Paper URL resolves to that paper — VARIANT — `web_search` returned `https://ojs.aaai.org/index.php/AAAI/article/view/37608` as the paper page with matching title; direct `read` of that URL failed (socket closed); arXiv alternative resolves — P3 — record the arXiv id alongside the OJS id for verifiability.
5. `upstream_repo_url` correct — VERIFIED — `models.yaml:231`; `git remote -v` in the checkout → `git@github.com:iLearn-Lab/AAAI26-HABIT.git` — NONE — none.
6. `source_available` and checkout present — VERIFIED — `models.yaml:234`; `ls workbench/third_party/HABIT` — NONE — none.
7. `upstream_commit_sha` equals checkout HEAD — VERIFIED — `models.yaml:233` = `git -C … rev-parse HEAD` = `6ddc2cd63eac1f65693f4e821c34fb3b42aee306` — NONE — none.
8. Checkout clean of tracked modifications — VERIFIED — `git status --porcelain` → only untracked `__pycache__/`; `runtime.source_clean_and_pinned` → `True` — NONE — none.
9. `source_dir` correct — VERIFIED — `models.yaml:235` `HABIT`; directory exists — NONE — none.
10. `python_env_var` declared — VERIFIED — `models.yaml:218` `WORKBENCH_HABIT_PYTHON` — NONE — none.
11. Environment confidence backed by README — VERIFIED — `VERIFIED_README` (`:219`); README "Install" section states Python 3.8.10, torch 2.1.0, torchvision 0.16.0, CUDA 12.6 compatibility and the five packages — NONE — none.
12. Declared python version matches README — VERIFIED — `'3.8.10'` (`:220`) vs README "evaluated on Python 3.8.10" — NONE — none.
13. Declared torch/torchvision match README — VERIFIED — `'2.1.0'/'0.16.0'` (`:221-222`) vs README `pip install torch==2.1.0 torchvision==0.16.0` — NONE — none.
14. Declared CUDA vs README — VARIANT — registry `'12.6'` (`:223`); README says "evaluated on … CUDA 12.6" but installs the `cu121` wheel — P3 — clarify "host CUDA 12.6 / wheel cu121".
15. Declared packages match README — VERIFIED — `:224` list equals README's core-dependency line (5 packages) — NONE — none.
16. `native_protocol` == `supported_protocols` — VERIFIED — `models.yaml:239-240`, both `fashioniq_full_gallery_ref_excluded`; enforced at `registry.py:46-47` — NONE — none.
17. Protocol id exists in the protocol table — VERIFIED — `models.yaml:19-22` (`fashioniq_full_gallery_ref_excluded`) — NONE — none.
18. Native protocol matches the evaluator's actual split default — VERIFIED — `datasets.py:28` `split='original-split'`; launcher passes no split (`train.py:108-113`); `original-split` branch at `:205-211` — NONE — none.
19. Gallery is the full ordered validation split in source — VERIFIED — `datasets.py:205-211`; query index space from `:177-189` — NONE — none.
20. Reference exclusion present in source — VERIFIED — `test.py:150-151` mask before `argsort` — NONE — none.
21. Workbench helper mask-vs-removal equivalence — VARIANT — `src/evaluation/fashioniq.py:35-73` removes the column, `:40-48` admits non-equivalence; source masks — P2 — document the sentinel bound or use mask semantics if ever routed through the helper (F-2).
22. Workbench helper is on habit's execution path — VERIFIED (it is not) — regex sweep: `evaluate_fashioniq(` only at `src/evaluation/fashioniq.py:242` + `tests/unit/test_fashioniq_dispatch.py`; adapter routes to native `test.py` — NONE — none.
23. Recall@K set in source — VERIFIED — `test.py:169-175` `[1, 10, 50]`, percent hits — NONE — none.
24. Recall@K set in paper — VERIFIED — paper §4.1 R@K; FIQ Table 1 columns R@10/R@50/AVG; CIRR R@{1,5,10,50} — NONE — none.
25. Sorting direction and determinism — VARIANT — `np.argsort(-sims)` (`test.py:154,168`); `torch.argsort`/`np.argsort` are not guaranteed stable, so tie order is unspecified on both sides; workbench states this at `src/evaluation/fashioniq.py:_assert_comparable` — P3 — none unless a tie-sensitive run is claimed exact.
26. Per-query normalization on the eval path — VERIFIED (absent, deliberately) — `test.py:133-137` normalisation block commented out; scoring at `:141` — NONE — none; do not "fix".
27. `squeeze`/`max(-1)` reduction semantics — VERIFIED — `test.py:141-143`; shape algebra reproduced for `(N,1,1,D)@(G,D,T)` → `(N,G,T)` → max → `(N,G)` — NONE — none.
28. `reported_r10` vs paper — VERIFIED — `55.38` (`models.yaml:241`) = Table 1 0% AVG R@10 — NONE — note F-7 rounding.
29. `reported_r50` vs paper — VERIFIED — `75.20` (`:242`) = Table 1 0% AVG R@50; recomputed 75.196667 — NONE — none.
30. Mean identity — VERIFIED — `65.29` (`:243`) = (55.38+75.20)/2 exactly; also (55.373333+75.196667)/2 = 65.285 → 65.29 — NONE — none.
31. Macro construction matches the source/paper notion — VARIANT — source computes no macro at all (`test.py:175` per-category only); workbench defines category-mean then mean-of-R@10/R@50 (`metrics_extraction.py:148-152`, `src/evaluation/fashioniq.py:223-240`); paper's AVG column is the same construction to within 0.01 — P3 — state the macro definition next to any habit metric.
32. `fashioniq_layout` correct — VERIFIED — `models.yaml:236` `fashioniq_ilearn_resized`; validator `backend/fashioniq_layout.py:25-34`, `:68-69`; matches `datasets.py:33` `path + 'resized_image'` — NONE — none.
33. `fashioniq_image_extensions` correct — VERIFIED — `[".jpg"]` (`:237`) vs `datasets.py:141,152` `… + ".jpg"` — NONE — none.
34. Correction dicts required, provenance assigned — BLOCKED — `datasets.py:100,174`; `git ls-files | grep correction` in HABIT empty; `docs/AUXILIARY_ASSET_AUDIT.md:355-357` `OFFICIAL_ASSET_UNAVAILABLE`; no habit row in `registry/auxiliary_assets.yaml` — P1 — record a source for the dictionaries before any run.
35. Correction dicts present locally — BLOCKED — `preflight` observed three `correction_dict_*.json` missing; only LIMN/TGCIR checkouts contain them, which must not be borrowed — P1 — stage from a declared source.
36. Resized images required and present — BLOCKED — required `datasets.py:140-141,151-152`; `preflight` observed `resized_image/` + 3 category dirs missing; local `images/` are 74,381 `.png` — P1 — produce `resized_image/<cat>/*.jpg` (C-3).
37. `preparation_contract` value is a legal id — VERIFIED — `manual_ilearn_resized` (`:228`) is in the allowed set at `registry.py:73` — NONE — none.
38. `preparation_id` resolves in `preparation_contracts.yaml` — BLOCKED — no habit row (`grep` → none; ids are clvc/dcnet/clip4cir/tgcir/sprc/limn); `preflight.py:27-30` therefore falls back to layout paths — P2 — add a habit contract or accept and document the fallback.
39. Generated caches (JSON + six PKLs) handled — VARIANT — written on cache miss (`datasets.py:39-63`), trusted on hit (`:55-63`); iLearn layout validator does not require or freshness-check them (`backend/fashioniq_layout.py:25-34`); `registry/audits/cluster_b_ilearn.yaml:712-755` classifies them `SOURCE_REGENERABLE` — P2 — refuse stale caches for a claimed reproduction (a stale `fashion_iq_data.json` silently changes the val query set).
40. Text-normalization parity (source vs workbench composer) — UNKNOWN — source `datasets.py:112-121`; workbench `src/datasets/fashioniq.py:93-…` claims to reproduce ENCODER-style normalization, but I did not execute the two on the same corpus and no habit callsite consumes the composer (`docs/SOURCE_TO_WORKBENCH_PARITY_MATRIX.md:29` rates this boundary EQUIVALENT_WITH_EVIDENCE) — P2 — diff the two implementations on the real val captions once the dictionaries exist.
41. Eval caption processor stage present and correct — VERIFIED — `test.py:111` + `blip_processors.py:49-68` (lowercase, `[.!"()*#:;~]`→space, collapse, strip, 50-word truncation) — NONE — none.
42. Tokenizer dependency pinned — VARIANT — `/root/bert-base-uncased` hardcoded (`blip2.py:32,48,55`); registry records no interpreter sha and no BERT revision (`docs/UPSTREAM_AUDIT.md:109` covers only `cache_root`) — P2 — pin the BERT revision and provision the path (F-4).
43. Image transform parity — VERIFIED — `targetpad_transform(1.25, 224)` (`data_utils.py:92-106`) selected by `train.py:94-106`; raises for unknown transform (`:105`); TargetPad/sibling files byte-identical across the four iLearn checkouts (recomputed) — NONE — note torchvision/PIL minor-version caveat.
44. Augmentation contract — VERIFIED (no train-only augmentation leaks into eval) — no `Random*` transforms anywhere in `data_utils.py`/`train.py`/`datasets.py`; eval transform is deterministic — NONE — none.
45. Training noise injection is train-only — VERIFIED — `train.py:108-113` + `datasets.shuffle()` rewrites `fashioniq_data` only (`datasets.py:65-94`), which is built from train captions (`:96-110`); eval reads `cap.<cat>.val.json` (`:172`) — NONE — none; this justifies `evaluation_noise_pct: 0`.
46. Eval path free of MKE/DPL/DBSCAN — VERIFIED — `test.py` references only `extract_retrieval_compose`/`extract_retrieval_target` (`:112,128`); `HABIT.forward` (MKE/DPL/DBSCAN at `HABIT.py:264-368`) is never called from `test.py`; `get_soft_labels`/`get_similarity_matrix` only referenced at `train.py:322-323` — NONE — none; the progressive machinery does not touch evaluation.
47. Eval model construction avoids `forward` — VERIFIED — `extract_retrieval_compose` (`HABIT.py:372-413`) and `extract_retrieval_target` (`:416-436`) are `@torch.no_grad()` and call only `visual_encoder`/`ln_vision`/`Qformer.bert`/`text_proj`/`vision_proj`; `self.temp` (`:165`) is never used on the eval path — NONE — none.
48. Unused eval-time parameters — VARIANT — `prompt_tokens` (`HABIT.py:168-171`) is trained(-initialised) but never read by `extract_retrieval_compose`; `self.sim`/`soft_margin_loss` are training-only (`:172-175,357-363`). Harmless for checkpoint loading (parameters must still exist in the state dict). — P3 — state the checkpoint key set when a loader is written, so a partial load is caught.
49. Checkpoint filenames match the Hub — VERIFIED — `models.yaml:249,259,269` equal the Hub `fiq/HABIT-FIQ_N0.{2,5,8}.pt` entries (`read …?blobs=true`) — NONE — none.
50. Clean (0%) checkpoint exists upstream — BLOCKED — Hub listing has no `N0.0` file; README documents `--noise_ratio 0.0` as an evaluated setting — P0 — this is C-1; do not present any registered weight as clean.
51. Registry noise labels are honest (no noisy checkpoint called clean) — VERIFIED — `checkpoint_training_noise_pct: 20/50/80`, `status: NO_CLEAN_CHECKPOINT` (`:252,256,262,266,272,276`), `strict_clean_checkpoint_available: false` (`:245`); enforced by `registry.py:83-103` and by observed `select_checkpoints` → `BLOCKED` — NONE — none.
52. Checkpoint digests/sizes recorded and verified — BLOCKED — `expected_sha256: null`, `expected_size_bytes: null` (`:254-255,264-265,274-275`); observed `official_sha256_known: False`, `local_sha256: None`. Hub does publish LFS sha256 + size, which are unrecorded — P2 — record the Hub LFS values.
53. Checkpoint obtainable now — BLOCKED — `checkpoint_downloaded: False`; `artifacts/checkpoints/habit/` absent; three `download_url`s resolve on the Hub (`resolve/main/...`) but downloading is outside this audit — P1 — download under authorization and verify against the recorded digest.
54. Adapter command implemented and audited — BLOCKED — `HabitAdapter` (`backend/adapters/models.py:64-66`) does not override `command`; observed `command not runnable: COMMAND_UNAUDITED`; `build_command` observed to raise `FileNotFoundError` for the missing checkpoint first; source has no CLI (`test.py` AST) — P0 — implement and audit a replay entrypoint (C-2).
55. Runtime status is honest and no runtime result is fabricated — VERIFIED — `reproduction_status: NOT_RUN` (`:244`); no run artefacts observed anywhere (`artifacts/checkpoints/habit` absent); this audit claims no runtime metric — NONE — none.

---

## FINAL VERDICT

**BLOCKED.** The registry metadata for `habit` is largely truthful and the pinned source's protocol (full ordered validation gallery, reference masked at `-10e10` before `np.argsort`, R@1/10/50) matches `fashioniq_full_gallery_ref_excluded`, and the reported 55.38 / 75.20 / 65.29 reproduce from the paper's own 0%-noise row with the mean identity holding exactly — but reproduction is impossible as registered: the upstream Hub publishes only 20/50/80%-noise weights with **no clean 0% checkpoint**, the workbench adapter has **no audited command** (`command not runnable: COMMAND_UNAUDITED`; pinned `test.py` has no `__main__`/`argparse` and is a training subroutine), the dataset contract is unsatisfied locally (`resized_image/` and the unprovenanced `correction_dict_*.json` all missing), and the only paper-comparable checkpoint does not exist.

`FINAL VERDICT: BLOCKED`

---

---

## REFRESH (this engagement)

Appended after the work that landed on top of this audit. Earlier findings above are left
as written; where one has been superseded, the superseding fact is stated here.

- **Static benchmark reference.** 4 committed row(s):

  - `habit_aaai26_table1_fiq_n00` — `VERIFIED`, 0% training noise, protocol `fashioniq_full_gallery_ref_excluded`, values `{'r10': 55.38, 'r50': 75.2, 'mean': 65.29}`, provenance `{'r10': 'PAPER_REPORTED', 'r50': 'PAPER_REPORTED', 'mean': 'PAPER_REPORTED'}`
  - `habit_aaai26_table1_fiq_n20` — `VERIFIED`, 20% training noise, protocol `fashioniq_full_gallery_ref_excluded`, values `{'r10': 54.48, 'r50': 74.28, 'mean': 64.38}`, provenance `{'r10': 'PAPER_REPORTED', 'r50': 'PAPER_REPORTED', 'mean': 'PAPER_REPORTED'}`
  - `habit_aaai26_table1_fiq_n50` — `VERIFIED`, 50% training noise, protocol `fashioniq_full_gallery_ref_excluded`, values `{'r10': 52.52, 'r50': 73.09, 'mean': 62.8}`, provenance `{'r10': 'PAPER_REPORTED', 'r50': 'PAPER_REPORTED', 'mean': 'PAPER_REPORTED'}`
  - `habit_aaai26_table1_fiq_n80` — `VERIFIED`, 80% training noise, protocol `fashioniq_full_gallery_ref_excluded`, values `{'r10': 48.36, 'r50': 69.53, 'mean': 58.94}`, provenance `{'r10': 'PAPER_REPORTED', 'r50': 'PAPER_REPORTED', 'mean': 'PAPER_REPORTED'}`

- **Readiness now:** `BLOCKED_SOURCE` — upstream source unavailable

- **Reference per registered checkpoint condition:**

  - `fiq_n02` (noise 20): `habit_aaai26_table1_fiq_n20` `VERIFIED` — exact condition match
  - `fiq_n05` (noise 50): `habit_aaai26_table1_fiq_n50` `VERIFIED` — exact condition match
  - `fiq_n08` (noise 80): `habit_aaai26_table1_fiq_n80` `VERIFIED` — exact condition match

- **Gate:** the run's verdict comes from `workbench/backend/paper_gate.py` against the
  committed row above, reported in the run JSON under `benchmark_gate`. A row that states no
  condition is refused (`BLOCKED_REFERENCE_INCOMPLETE`); a metric the paper never printed is
  compared only if the registry derives it, and then it is labelled `WORKBENCH_DERIVED`.
