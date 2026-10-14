# clvc_net

Independent adversarial reproducibility audit of registered model `clvc_net` (CLVC-Net) in the CIR Failure Analysis Workbench. Read-only: nothing under `/home/heheboiz/data/cir` was written; no GPU job, install, download, or mutation was run (scratch only under `/tmp`, plus read-only `git`/`python3` inspection). Every claim is **STATIC_AUDIT** unless labelled otherwise. This document contains **zero RUNTIME_VERIFIED** claims: no checkpoint exists locally, no environment exists, and the evaluator is CUDA-hard.

## MODEL ID

`clvc_net`

- Registry block: `workbench/registry/models.yaml:492-555` (`- model_id: clvc_net`).
- `method_name: CLVC-Net` (`models.yaml:508`), `source_dir: CLVCNet` (`models.yaml:517`), `source_available: true` (`models.yaml:516`).
- Registry validation accepts the record: `layouts` whitelist includes `fashioniq_clvc_resized` (`workbench/backend/registry.py:28`), `preparation_contract` whitelist includes `manual_clvc_resized` (`registry.py:73`), `preparation_id: clvc_net_fashioniq` ():521 is registered (`registry.py:75`).

## PAPER TITLE

`Comprehensive Linguistic-Visual Composition Network for Image Retrieval` (`workbench/registry/models.yaml:511`).

- `paper_url: https://doi.org/10.1145/3404835.3462967` (`models.yaml:512`). Fetched `https://api.crossref.org/works/10.1145/3404835.3462967`: title **"Comprehensive Linguistic-Visual Composition Network for Image Retrieval"**, authors Haokun Wen, Xuemeng Song, Xin Yang, Yibing Zhan, Liqiang Nie; `page: "1369-1378"`; `published-print` 2021-07-11; publisher ACM; `type: proceedings-article`. **VERIFIED (STATIC_AUDIT, Crossref API).**
- The pinned README carries the same title and authors (`workbench/third_party/CLVCNet/README.md:1-12`) and the same ACM DOI link (`README.md:14`). `publication_year: 2021` (`models.yaml:510`) agrees with Crossref.
- The paper PDF itself is paywalled (ACM DL); I did not obtain it, so **no table cell or metric from the paper is quoted anywhere in this document** (see PAPER-METRIC STATUS).

## PINNED SOURCE COMMIT

- Registry pin: `upstream_commit_sha: bd9b6889489806cb60676597880f91d918b279cb` (`models.yaml:515`).
- Observed: `git -C /home/heheboiz/data/cir/workbench/third_party/CLVCNet rev-parse HEAD` → `bd9b6889489806cb60676597880f91d918b279cb`. **MATCH.**
- `git -C … log -1 --format="%H %ci %s"` → `bd9b6889489806cb60676597880f91d918b279cb 2026-04-05 18:54:18 +0800 Update README.md`; `git log --oneline --all` shows a single commit (`bd9b688 Update README.md`) — the release is a squashed snapshot.
- `git -C … remote -v` → `git@github.com:iLearn-Lab/SIGIR21-CLVC-Net` (matches `models.yaml:513`).
- Cleanliness: `git -C … status --porcelain` → only `?? __pycache__/`. `_source_dirty` (`workbench/backend/runtime.py:1998-2010`) treats `__pycache__`/`*.pyc` as non-dirty, so `source_clean_and_pinned` (`runtime.py:2013-2035`) would accept this checkout. Confirmed indirectly: `preflight_blockers` for this model lists no source-cleanliness blocker (observed output, RUNTIME_STATUS below).
- Tree size 348 KB, 8 `.py` + `README.md`, no requirements file: `ls workbench/third_party/CLVCNet` → `README.md blocks.py datasets.py model.py resnet.py test.py text_model.py train.py utils.py`.

## CHECKPOINT IDENTITY

Registry variant (`models.yaml:544-555`), read back through the loader:

```
checkpoint_id: author_drive_unresolved
filename: checkpoint-unresolved.pt
download_url: null
source_url: https://drive.google.com/file/d/159rBhWyhkLN7sXAi8iyW_ljzFNLJinKa/view?usp=sharing
source_type: author_google_drive
checkpoint_training_noise_pct: null
evaluation_noise_pct: 0
expected_size_bytes: null
expected_sha256: null
status: CHECKPOINT_MAPPING_UNVERIFIED
checkpoint_mapping_status: UNVERIFIED
```

(`python3` dump of the loaded record reproduces exactly these fields; `registry.py:83-103` enforces the `status`↔`checkpoint_mapping_status` agreement.)

Findings:

1. **There is no identified checkpoint.** The filename is a literal placeholder (`checkpoint-unresolved.pt`), there is no download URL, no size, no hash. `checkpoint_path` (`registry.py:618-621`) would resolve to `<checkpoint_root>/clvc_net/checkpoint-unresolved.pt`, which does not exist: `ls workbench/artifacts/checkpoints` → directory empty/absent. This is honest registry bookkeeping, not an artifact claim. **UNKNOWN**, severity P1.
2. **The author link is real but its contents are not established.** Fetched `https://drive.google.com/file/d/159rBhWyhkLN7sXAi8iyW_ljzFNLJinKa/view?usp=sharing`: `<title>CLVCNET.zip - Google Drive</title>`, `og:title CLVCNET.zip`. No member listing, no size, no digest is exposed (Google does not publish per-file hashes without an authenticated listing, and I did not download). README.md:74 links the same file as "**Pre-trained models**".
3. **What the source actually writes (so what the archive must contain)** — STATIC_AUDIT from the pinned trainer:
   - `train.py:274-275`: `torch.save(local_model, os.path.join(model_dir, 'local_model.pt'))`, `torch.save(global_model, os.path.join(model_dir, 'global_model.pt'))` → two **whole pickled modules** per run, not a state dict. Registry agrees: `checkpoint_serialization: whole_model` (`models.yaml:506`).
   - `train.py:279-306` runs one dataset chosen by `--dataset fashioniq --name <category> --model_dir ./<category>` (README.md:31-49): three independent runs → six files for three categories. Registry records this structure exactly and marks it unresolved: `category_model_policy: independent` (`models.yaml:499`), `category_checkpoint_structure` (`:500-505`, `member_count_expected: 6`, `mapping_status: UNVERIFIED`, note "do not claim six available files").
   - `utils.py:136-140` is a `torch.load` + `load_state_dict(checkpoint['state_dict'])` helper (DCNet lineage), **never called by `train.py`/`test.py`** (`grep -n "restore_file\|load_state" train.py test.py model.py utils.py` → `train.py:62,229,306` declare/pass `--restore_file`; `utils.py:136-140` only).
4. `checkpoint_training_noise_pct: null` (`models.yaml:550`) and `strict_clean_checkpoint_available: null` (`:542`) are consistent with the unresolved archive. Selection confirms it (observed, below).

Observed checkpoint selection (`python3` against the loaded registry):

```
strict_clean            -> BLOCKED/BLOCKED_CHECKPOINT: "no checkpoint variant records a training condition,
                            so none can be called clean under policy strict_clean..."
include_flagged_conditions -> BLOCKED/BLOCKED_CHECKPOINT: "checkpoint mapping unverified for
                            author_drive_unresolved; the upstream repository does not establish which variant
                            is the benchmark checkpoint"
```

matching `checkpoint_selection.py:126-131` (clean requires `_condition(...) == 0` **and** `VERIFIED_MAPPING` **and** obtainable). So: **no path to a checkpoint is currently open, under either policy.**

## EVALUATION PROTOCOL

- Registry: `native_protocol: fashioniq_val_split` (`models.yaml:523`), `literature_split_label: val` (`:522`), `supported_protocols: [fashioniq_val_split]` (`:524`).
- Protocol table: `workbench/registry/models.yaml:16-19` — label `FashionIQ - Val Split (Pair-Union Gallery, Reference Excluded)`, `literature_split_label: val`, `evaluation_noise_pct: 0`; implementation `src/evaluation/fashioniq.py:166-168` (`gallery="pair_union"`, `reference_excluded=True`, alias `fashioniq_val`).
- Router: `src/evaluation/fashioniq.py:192-207` → `build_val_gallery` (`:120-123`) → `build_pair_union_gallery` (`src/datasets/fashioniq.py:186-209`).
- **Independent re-derivation from the actual staged dataset** (observed `python3`): for each category, walking `cap.{cat}.val.json` in file order and taking `candidate` then `target`, first occurrence retained:

  | category | val queries | pair-union gallery | full `split.{cat}.val.json` |
  |---|---|---|---|
  | dress | 2017 | 2628 | 3817 |
  | shirt | 2038 | 3089 | 6346 |
  | toptee | 1961 | 2902 | 5373 |

  Every union id is present in the full split (0 exceptions), so the union is a genuine *reduced* cohort, not the full gallery. The registry's protocol choice is therefore **not** the full-gallery family. **VERIFIED (STATIC_AUDIT + local data).**
- The source fix for which is broad and the paper statement of which is absent: the registry assigns the cohort from the code (`datasets.py:233-257`, `test.py:131-138`), not from the paper. Whether the *published numbers* were produced on this cohort is **UNKNOWN** (no run log, no paper access).
- **No runnable command exists**: `CLVCNetAdapter` (`workbench/backend/adapters/models.py:74-76`) declares `script = "test.py"` and inherits `OfficialScriptAdapter.command` (`:24-25`) which raises `NotImplementedError`. `tests/test_protocols.py:72-77` asserts exactly this for `clvc_net`. `workbench/scripts/evaluate_models.py:113-114,195-197` skips any model whose adapter command is the base one. Registry agrees: `command_status: COMMAND_BLOCKED_SOURCE_ENTRYPOINT` (`models.yaml:529`), `replay_status: BLOCKED_NO_SOURCE_REPLAY` (`:493`).

## DATASET CONTRACT

Preparation contract `clvc_net_fashioniq` (`workbench/registry/preparation_contracts.yaml:3-59`) vs the pinned reader:

| Contract statement | Pinned source | Verdict |
|---|---|---|
| raw `captions/cap.{dress,shirt,toptee}.{train,val}.json` (prep:10-15) | `datasets.py:168` opens `cap.{name}.{split}.json`; `datasets.py:206` opens `cap.{name}.train.json` for vocab | VERIFIED |
| raw `image_splits/split.{dress,shirt,toptee}.val.json` (prep:16-18) | `datasets.py:170` opens `split.{name}.{split}.json` with `split='val'` (`train.py:89`) | VERIFIED |
| raw `images` (prep:19) | not opened by the evaluator; every `get_img` call reads the *resized* tree | NOT consumed at eval (see PREPROCESSING) |
| generated `resized_image/{dress,shirt,toptee}` (prep:26-42) | `datasets.py:157,195` → `<path>/resized_image/<name>/<id>.jpg` | VERIFIED |

Record shape consumed by the evaluator, checked against the real staged files (observed `python3` on `data/fashionIQ_dataset/captions/cap.dress.val.json`): 2017 records, keys `['candidate','captions','target']`, e.g. `{"target":"B0084Y8XIU","candidate":"B005X4PL1G","captions":["is shiny and silver with shorter sleeves","fit and flare"]}`. The source indexes exactly those keys: `caption['candidate']`, `caption['target']`, `caption['captions']` (`datasets.py:184-185, 219-220`). `split.dress.val.json` holds 3817 unique ids (observed).

Path joining is string concatenation, not `os.path.join`: `datasets.py:157-159` (`self.image_dir = self.path + 'resized_image'`). The source's own default `--fashioniq_path` ends in `/` (`train.py:32`). A replay must pass a trailing separator; the layout helper is agnostic (`fashioniq_layout.py:44-52`).

Workbench side:

- Layout helper `clvc_resized_paths` (`workbench/backend/fashioniq_layout.py:44-52`) requires exactly `captions/`, `image_splits/`, `resized_image/`, `resized_image/{cat}`, `cap.{cat}.{train,val}.json`, `split.{cat}.val.json` — i.e. it **does** include the train captions the evaluator needs, unlike the iLearn val-only layouts. Consumed by `fashioniq_required_paths` (`registry.py:679-683`).
- Image-directory/extension check agrees with the evaluator: `fashioniq_clvc_resized` → `("resized_image", True)` (per-category) (`fashioniq_layout.py:79-89`, `image_extension_problems` `:92-136`), and `models.yaml:519` declares `[".jpg"]`.
- Local staging is incomplete: the observed blocker list includes `required generated artifact missing: …/resized_image/{dress,shirt,toptee}` and `FashionIQ image extension requirement: image directory missing: …/resized_image`. The canonical tree does hold `captions/`, `image_splits/`, `images/` (74381 `.png`, all 3817 dress val ids present as `.png` — observed).
- **Contract gap:** `workbench/scripts/prepare_dataset.py:17-18` `required_layout` is `standard_paths` only, so the dataset stage never validates the CLVC layout; the model-specific check happens in preflight instead. P3, documented already in `docs/audits/PHASE2_DATA_CONTRACT.md:128`.

## PREPROCESSING CONTRACT

- **Resize is mandatory and is not in the pinned repository.** `README.md:28` instructs resizing "follow the [XiaoxiaoGuo] … start_kit by resize_images.py"; the pinned tree bundles no such script (observed `ls`: 8 `.py`, none named resize). Fetched `https://raw.githubusercontent.com/XiaoxiaoGuo/fashion-iq/master/start_kit/resize_images.py` (observed): `--image_size` default `256`, output is a **flat** directory, `image.resize((s,s), Image.ANTIALIAS)`, saved with `img.format`.
- Workbench recipe: `preparation_contracts.yaml:26-42` declares `resized_image/{dress,shirt,toptee}`, `image_size: 256`, `.jpg`, `per_category`, generator `fashioniq_resize_images`, `automation_policy: MANUAL_REQUIRED` (`:58`), `deterministic_status: UNKNOWN` (`:57`). Implementation: `workbench/scripts/prepare_model.py:353-486` — reads only the ids the split names, refuses traversal ids, refuses writing into the canonical `images/` tree, writes `<id>.jpg`, hashes each derived file and binds it to the canonical source bytes (`:451-455, 487-510`).
- **Level-2 mismatch.** The documented tool produces flat, ANTLIAS-filtered, format-preserved files; the contract/implementation produce per-category RGB JPEGs via `Image.LANCZOS` (`prepare_model.py:466-478`). This is a *declared workbench-applied recipe*, not a run of the README's tool (`preparation_contracts.yaml:39-40`: "reported as a workbench-applied recipe, not as a run of an in-repo script"). Consequence at evaluation: the pinned val transform is `Resize(256) → CenterCrop(224)` (`train.py:91-92`), so both pipelines converge to a 256-long-side-style input and the residual difference is resampling/entropy-coding only — but **byte-level equality with the author's images is UNKNOWN**. Severity P2; no workbench document claims byte equality (checked: `docs/AUXILIARY_ASSET_AUDIT.md:246` classifies it `SOURCE_REGENERABLE`/`AVAILABLE`; `docs/LEGACY_SOURCE_AUDIT.md:21` notes the 256 default).
- **External weight.** `auxiliary_assets.yaml:148-161` and `preparation_contracts.yaml:44-56` declare torchvision ResNet-50 ImageNet weights, URL `https://download.pytorch.org/models/resnet50-19c8e357.pth` (identical to the filename the source hardcodes at `resnet.py:12`), `expected_sha256_prefix: 19c8e357`, `required_for_evaluation: true`, `required_for_preparation: false`, `optional: false`. Consumer verified: `model.py:72` `resnet.resnet50(pretrained=True)` → `resnet.py:218-223` `load_state_dict_from_url(model_urls[arch], …)`; `img_backbone` is built inside **both** `compose_local` (`model.py:123`) and `compose_global` (`model.py:191`).
- **Cache-path defect (observed, not a hypothesis).** `external_asset_statuses` (`registry.py:213-216`) defaults the root to `Path(os.environ.get("XDG_CACHE_HOME","~/.cache")).expanduser() / "torch"` and then appends `cache_path_suffix = checkpoints/resnet50-19c8e357.pth` (`preparation_contracts.yaml:52`) → `~/.cache/torch/checkpoints/…`. `model_requirements` (`preflight.py:99-120`, and the observed run) reports `~/.cache/torch/hub/checkpoints/…`. Both appear in the same blocker list for the same asset (observed below). `load_state_dict_from_url` uses `$TORCH_HOME/hub/checkpoints` (the second form). So the registry's *resolver* points at a path the consumer never reads, and the operator is shown two contradictory paths for one file. **Severity P2** (P1-adjacent: a hash-verified file placed at the resolver's path would be silently not found by torch, or vice versa).
- `simplevocab.json`: `text_model.py:66` reads it only when `texts_to_build_vocab is None`; both constructions pass `texts` (`model.py:119-122`, `:191-194`) and `train.py:300` supplies `trainset.get_all_texts()`. Consistent with the registry asset audit marking it not required for evaluation (`registry/assets/cluster_b4_classic.yaml:266-269`, `required_for_evaluation: false`).

## AUGMENTATION CONTRACT

- Training path (`train.py:78-85`): `Resize(256)`, `RandomCrop(224)`, `RandomHorizontalFlip()`, `ToTensor`, ImageNet `Normalize`. Validation path (`train.py:90-96`): `Resize(256)`, `CenterCrop(224)`, `ToTensor`, `Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])`. `test.py` applies no transform of its own; it consumes tensors produced by the dataset object (`test.py:94,111`).
- `test.py:81-82` sets `local_model.eval()` / `global_model.eval()` before the no-grad pass — the model is *not* left in train mode (contrast: HABIT/CSMCIR family defects recorded elsewhere). Dropout therefore inactive (`model.py:76,128`).
- Registry side: `evaluation_noise_pct: 0` for both the protocol (`models.yaml:16-19`) and the checkpoint variant (`:551`); `registry.py:84-85` refuses any non-zero evaluation noise. **VERIFIED**, no augmentation noise enters evaluation.
- `torch.backends.cudnn.deterministic = True`, `benchmark = False`, seeded (`train.py:287-293`) affect training only; evaluation determinism still depends on `np.argsort` tie behaviour (see METRIC CONTRACT).

## GALLERY CONTRACT

Source (FashionIQ branch of `test.py`):

- Queries: `testset.get_test_queries()` (`test.py:84`), built by `datasets.py:214-230`: per annotation, `source_img_id = self.images.index(candidate)`, `target_img_id = self.images.index(target)` — ids are **indices into the full split list**, not strings.
- Gallery: `testset.get_test_targets()` (`test.py:85`), and with the constructor default `gallery_all=False` (`datasets.py:153`, never overridden by `train.py:86-96`) the `else` branch runs (`datasets.py:242-257`): reference first, then target, first occurrence retained — source-order pair union.
- Reference exclusion: `test.py:131-135` builds `test_targets_id` from the gallery and sets `sims[i, test_targets_id.index(t['source_img_id'])] = -10e10` **before** `np.argsort(-sims[i,:])[:50]` (`test.py:138`). Mask, not removal. `-10e10` is below any plausible dot product of L2-normalized features, so top-k behaviour matches removal — but they are different operations and the equivalence depends on that bound (`src/evaluation/fashioniq.py:44-52` argues the same).
- Ranking depth: `[:50]` (`test.py:138`) and recall over `[1,10,50]` (`test.py:142`) — k=10/50 are inside the truncated list, so truncation is not metric-affecting here.

Workbench: `fashioniq_val_split` = `pair_union` + `reference_excluded` (`src/evaluation/fashioniq.py:166-168`); gallery built by the same first-seen reference-then-target rule but keyed by **id strings** (`src/datasets/fashioniq.py:186-209`); exclusion by **removing** the reference column after a stable sort (`fashioniq.py:35-77`, `fashioniq_encoder.py:44-45`); `recall_at_k` asserts target membership and gallery uniqueness (`fashioniq.py:9-33, 125-140`).

**Verdict:** the cohort *rule* matches (VERIFIED); the *procedure* is a documented VARIANT (int-index dedup vs id-string dedup; mask-then-sort vs sort-then-remove; `np.argsort` vs `torch.argsort(stable=True)`). Severity P3, with the standing instruction never to assert numeric identity from procedure equivalence.

## METRIC CONTRACT

- Source: `test.py:140-148` — for `k in [1,10,50]`, hit test `test_targets_id.index(test_queries[i]['target_img_id']) in nns[:k]`, `r = 100 * r / len(nn_result)`, emitted as `out += [('{}_r{}'.format(dataname, k), r)]` — i.e. `fashioniq_r1`, `fashioniq_r10`, `fashioniq_r50` for the single category under `--name`.
- Workbench extraction: `metrics_extraction.py:287-320` — only `limn` and `csmcir` have parsers; every other model returns `AGGREGATE_PARSER_UNAVAILABLE`. Observed:
  `extract_aggregate_metrics("clvc_net", "dress_r10: 30.0\n", "", 0, {…None…})` → `AGGREGATE_PARSER_UNAVAILABLE / PARSER_UNAVAILABLE / None`. So even a successful run could not be parsed today.
- Workbench aggregate: `macro_average_fashioniq` (`src/evaluation/fashioniq.py:223-240`) → `mean_recall = (recall_at_10 + recall_at_50)/2`. No CLVC source publishes that quantity (`docs/FASHIONIQ_PROTOCOL_PARITY.md:95-98`, finding 4).
- Gate: `paper_gate.py:22` (5% relative), `:24` (`r10/r50/mean`), `:40-47` comparability fields, `:156-195` ordering. Observed with the registry's own (null) paper metrics:

  ```
  evaluate_gate({r10:None,r50:None,mean:None}, {…real numbers…}, data_kind="experiment", real_execution=True)
  -> status METRIC_REFERENCE_UNVERIFIED
     detail "without a comparable published number there is nothing to reproduce"
  ```

  i.e. **the gate can never pass, fail, or compare for this model as registered** — it short-circuits at `paper_gate.py:177-180`.

## SOURCE-TO-WORKBENCH PARITY

| Component | Source evidence | Workbench | Status |
|---|---|---|---|
| Protocol/cohort | `datasets.py:242-257`, `test.py:131-138` | `models.yaml:523`, `src/evaluation/fashioniq.py:166-168` | EQUIVALENT_WITH_EVIDENCE (re-derived sizes 2628/3089/2902) |
| Gallery dedup key | ints `datasets.py:222,224,239,246-249` | id strings `src/datasets/fashioniq.py:201-207` | VARIANT (P3) |
| Reference exclusion | mask `-10e10` pre-sort `test.py:135,138` | remove column post-sort `fashioniq.py:44-52` | VARIANT (P3) |
| Ties | `np.argsort(-sims)` `test.py:138` | `torch.argsort(...,stable=True)` `fashioniq_encoder.py:44` | MISMATCH (P3) |
| Viewer | DataLoader for train (`train.py:239-243`); manual list accumulation at eval (`test.py:91-119`) | n/a (native evaluator) | N/A |
| Preparation contract | `README.md:28` external tool; tree has no script | `preparation_contracts.yaml:26-42`, `prepare_model.py:353-486` | VARIANT (P2, recipe not a run of the tool) |
| Text pipeline | markers + `SimpleVocab` (`datasets.py:174`, `text_model.py:8-27`); vocab from train captions (`train.py:300`, `datasets.py:204-212`) | local `compose_fashioniq_caption` ordered_and (`src/datasets/fashioniq.py:151-156`) — **not invoked for this model** | NOT_YET_VERIFIED / not on the path |
| Evaluator command | `test(params, local_model, global_model, testset, dataname)` `test.py:6` | adapter inherits base `command` (`adapters/models.py:24-25,74-76`) | **NO PARITY POSSIBLE — command does not exist** |
| Checkpoint consumption | `test.py:6` takes constructed models; `train.py:229` never reads `restore_file`; `utils.py:136-140` unused | `category_model_policy: independent` `models.yaml:499` | INCOMPLETE_IMPLEMENTATION (P1) |
| Metric extraction | `test.py:148` prints `fashioniq_r{k}` | `metrics_extraction.py:306-320` | NOT IMPLEMENTED (P1) |
| Paper reference | unknown (no table obtained) | `models.yaml:525-527` nulls | NO REFERENCE (P1) |

Existing documents that assert `category_model_policy: shared` for CLVC-Net are **stale**: `docs/SOURCE_TO_WORKBENCH_PARITY_MATRIX.md:25` and `:58`, `docs/UPSTREAM_DATA_PREPARATION_AUDIT.md:133`, `docs/audits/PHASE2_SOURCES_B.md:16,20,27`. The registry itself says `independent` (`models.yaml:499`), and the source is per-category (`train.py:70-96,299`; `README.md:31-49`). This audit sides with the registry; the docs need updating. P3.

## RUNTIME STATUS

Observed `python3` run of the real preflight against the real config:

```
required generated artifact missing: /home/heheboiz/data/cir/data/fashionIQ_dataset/resized_image/dress
required generated artifact missing: /home/heheboiz/data/cir/data/fashionIQ_dataset/resized_image/shirt
required generated artifact missing: /home/heheboiz/data/cir/data/fashionIQ_dataset/resized_image/toptee
FashionIQ image extension requirement: image directory missing: …/fashionIQ_dataset/resized_image
torchvision ResNet-50 ImageNet weights missing: /home/heheboiz/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth
external runtime asset file missing: torchvision ResNet-50 ImageNet weights (/home/heheboiz/.cache/torch/checkpoints/resnet50-19c8e357.pth)
command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT
execution environment not verified: WORKBENCH_CLVC_PYTHON=None [UNCONFIGURED/UNCONFIGURED]
```

- `runtime_blockers` (`runtime.py:2091-2113`) would additionally append `checkpoint missing: …/checkpoint-unresolved.pt` and `checkpoint mapping unresolved` for the one registered variant (`:2103-2106`).
- GPU: `environment.cuda: '10.2'` (`models.yaml:540`) → `_requires_gpu` True (`runtime.py:805-815`), and the evaluator calls `.cuda()` unconditionally at `test.py:28,47,99,115`, plus `text_model.py:96,108` and `model.py:178,249`. The audit host is CPU-only with no workbench torch at all (`python3 -c "import torch"` → `ModuleNotFoundError`). **This model cannot be executed in this environment, and would not run on a CUDA-less host by design.**
- `docs/READINESS_MATRIX.md:58` records `SOLVER_COMPATIBLE` environment planning and `BLOCKED` evaluation for `clvc_net`, consistent with the above.
- No result artifact exists: `find workbench/artifacts -iname "*clvc*"` → empty; only `mock/` files exist under `artifacts/results/*/`. **RUNTIME STATUS = NOT_RUN (nothing was executed, and nothing can be).**

## PAPER-METRIC STATUS

- Registry: `reported_r10: null`, `reported_r50: null`, `reported_mean: null` (`models.yaml:525-527`) — read back through `load_registry()` exactly as nulls. There is **no paper row for `clvc_net` anywhere in the registry**: `grep -n "paper_metrics|reported"` over `workbench/registry/**` finds paper values only in `models.yaml` for other models; `registry/audits/cluster_c_classic.yaml` (the CLVC section, lines 122-265) carries no reported-metric field; and `workbench/docs/**` contains no CLVC-Net R@10/R@50 number (grepped).
- I could not obtain the paper table: the DOI resolves (Crossref confirms title/authors/pages/year), but the ACM PDF is paywalled, and web search did not surface a CLVC-Net FashionIQ Recall table in any accessible secondary source (searches returned only narrative mentions, e.g. a 2025 review that describes CLVC-Net's architecture without numbers).
- Therefore: **PAPER-METRIC STATUS = UNVERIFIED / unavailable.** The gate's own vocabulary is the correct label: `METRIC_REFERENCE_UNVERIFIED` (observed). This audit deliberately **invents no row**: the honest statement is that the workbench cannot compare this model against anything, ever, until someone transcribes the published table with a citation.
- Note the consequence is not merely cosmetic: `evaluate_models.py:215` builds `paper_metrics` from these fields, `autopilot.py:620-622` writes `paper_r10/r50/mean` into every per-model run record, and `paper_gate.py:177-180` refuses comparison when they are all None. So even a hypothetically successful run would be recorded with `paper_*: None` and a gate status of `METRIC_REFERENCE_UNVERIFIED`.

## CRITICAL FINDINGS

**C1 — No source-backed evaluation entrypoint exists (P1).** `workbench/third_party/CLVCNet/test.py` is 148 lines containing exactly one function, `def test(params, local_model, global_model, testset, dataname)` (`test.py:6`). Confirmed mechanically: `python3` AST inspection → `has __main__: False`, `has argparse: False`, `defs: ['test']`; `grep -n "main\|argparse\|def \|__name__" test.py` → only `test.py:6`. The README's `python test.py` (`README.md:67-71`) therefore runs *nothing*. A replay must independently construct both `compose_local(texts, T, dropout_p)` and `compose_global(texts, T, dropout_p)` (`model.py:115-122`, `:187-194`) with `texts = trainset.get_all_texts()` as `train.py:300` does. Registry records this as `COMMAND_BLOCKED_SOURCE_ENTRYPOINT` (`models.yaml:529`) and the adapter does not implement a command (`adapters/models.py:74-76`). **Nothing on the registered path can start.** Recommended: keep it blocked; if a wrapper is ever written, it must be a reviewed `workbench/replay/clvc.py` in the style of `workbench/replay/tgcir.py`, never a patch to the pinned source.

**C2 — The checkpoint path is a placeholder, and the archive's interior is unknown (P1).** `checkpoint-unresolved.pt`, `download_url: null`, size/hash null (`models.yaml:544-555`); Drive exposes only `CLVCNET.zip` (observed page title) with no listing. Selection is BLOCKED under both policies (observed). Without the file→category→local/global mapping, one pair cannot be routed to three categories — and the registry itself insists on `category_model_policy: independent` (`models.yaml:499`). Recommended: obtain the archive, inventory it, hash every member, and record the mapping before any claim of "checkpoint available".

**C3 — No text-model loader, so a whole model is the only defensible artifact (P1).** `test.py:6` receives constructed models; `train.py:229` accepts `restore_file` and never uses it (`train.py:62-64, 306`); the only loader in the tree, `utils.py:136-140`, expects a `state_dict` and is never called. Since the text vocabulary is built from the training captions (`train.py:300`, `datasets.py:204-212`, `text_model.py:62-64`), a rebuilt vocabulary would produce different token ids and different embeddings. Any replay must `torch.load` the whole pickled module — which is exactly why the runbook says "Never deserialize a whole-model pickle until provenance is trusted" (`docs/LEGACY_REPRODUCTION_RUNBOOK.md:8`). Recommended: treat whole-model loading as a trust boundary, require the digest first.

**C4 — The paper-metric gate is structurally inoperable for this model (P1).** All three `reported_*` fields are null and no verified row exists elsewhere (see PAPER-METRIC STATUS). Observed gate output `METRIC_REFERENCE_UNVERIFIED`. Recommended: transcribe the published table with citation, or state explicitly in the registry note that this model is permanently non-comparable — never fill the fields with recalled numbers.

## MINOR FINDINGS

**M1 — External-asset path inconsistency (P2).** The registry resolver computes `~/.cache/torch/checkpoints/resnet50-19c8e357.pth` (`registry.py:213-216` appends `checkpoints/…` on top of a hardcoded `.../torch`), while preflight reports `~/.cache/torch/hub/checkpoints/…` and torch itself reads `$TORCH_HOME/hub/checkpoints`. Observed both strings in one blocker list. Either the resolver should append to `hub/checkpoints` or the contract should stop double-declaring the suffix. (`docs/LEGACY_REPRODUCTION_RUNBOOK.md:22` and `docs/LEGACY_SOURCE_AUDIT.md:22` correctly name `$TORCH_HOME/checkpoints/…`; the code is what disagrees.)

**M2 — Resize recipe is a workbench recipe, not the README's tool (P2).** Contract declares per-category `.jpg` at 256 (`preparation_contracts.yaml:26-42`); `prepare_model.py:466-478` uses `Image.LANCZOS` and RGB conversion and writes `<id>.jpg`; the fetched start_kit script writes flat, uses `Image.ANTIALIAS`, and preserves `img.format`. The evaluator's own `Resize(256)+CenterCrop(224)` (`train.py:91-92`) makes this low-impact, but the registry should not be read as asserting byte equality with the author's images — and `deterministic_status: UNKNOWN` (`preparation_contracts.yaml:57`) already admits that.

**M3 — Stale `category_model_policy: shared` claims in four documents (P3).** `docs/SOURCE_TO_WORKBENCH_PARITY_MATRIX.md:25,58`, `docs/UPSTREAM_DATA_PREPARATION_AUDIT.md:133`, `docs/audits/PHASE2_SOURCES_B.md:16,20,27`. Registry: `independent` (`models.yaml:499`, blamed to commit `70c55d84`, 2026-10-12).

**M4 — torchvision version is UNKNOWN although the source depends on a specific import path (P2).** `models.yaml:535` says `UNKNOWN`; `resnet.py:3` does `from torchvision.models.utils import load_state_dict_from_url`, a module path that was removed in later torchvision. `models.yaml:533-534` pin Python 3.7.6 / PyTorch 1.6.0 from `README.md:18-19` — the coherent torchvision for that pair is 0.7.x. Recommended: pin it explicitly before building the environment, or record the import-path constraint as the reason.

**M5 — No category-model routing code exists, unlike LIMN (P3).** LIMN has both `category_model_policy: independent` and a replay that routes it (`workbench/replay/limn.py:191`, plus hash validation). CLVC-Net records the policy but has no `workbench/replay/clvc.py` (observed: `workbench/replay/` contains `limn.py`, `tgcir.py`, `common.py` only).

**M6 — Dataset-stage validation does not cover this layout (P3).** `prepare_dataset.py:17-18` validates `standard_paths` only; CLVC-specific requirements are checked by preflight. `docs/audits/PHASE2_DATA_CONTRACT.md:128` already records this registry-wide.

**M7 — Trailing-slash coupling is recorded but unenforced (P3).** `datasets.py:157-159` concatenates; `docs/AUXILIARY_ASSET_AUDIT.md:254` records the requirement; nothing in the workbench can enforce it because no command exists. A future wrapper must pass `--fashioniq_path` with a trailing separator, as `EncoderAdapter` does (`adapters/models.py:41`).

## UNRESOLVED EVIDENCE

1. The contents of `CLVCNET.zip`: member names, count, sizes, hashes, and the file→category→local/global mapping. Cannot be resolved read-only without downloading from Drive (no per-file hash API, no listing).
2. The paper's FashionIQ table: paywalled at ACM; no accessible secondary source reproduced a CLVC-Net numeric row in the searches I ran. Therefore every paper metric for this model is UNKNOWN.
3. Whether the author's resized images are byte-identical to the workbench recipe's output (resampling filter and container differ).
4. Whether the author's published numbers were produced on the pair-union val cohort: the code computes that cohort, the paper text was not reachable to confirm it says so.
5. Whether `--restore_file` was ever functional in the authors' private tree (the pinned commit declares and passes it, but nothing reads it).
6. `WORKBENCH_CLVC_PYTHON`: no interpreter recorded or configured anywhere on this host; no environment exists to inspect.
7. torchvision/CUDA version actually used by the authors ("GPU: TITAN XP", `README.md:20`, "Ubuntu 14.04.6 LTS", `:21`; no versions).

## FINAL VERDICT

**BLOCKED.** The pin, paper identity, cohort assignment, gallery rule, metric formula and preparation contract were all re-derived and agree with the pinned source (including an independent recomputation of the pair-union gallery sizes 2628/3089/2902 against full splits 3817/6346/5373), but the registered path cannot start: the pinned evaluator has no CLI or loader, the adapter deliberately raises, the checkpoint is a placeholder with an uninspected author archive, the ResNet-50 asset has a contradictory resolved path and is absent, the resized-image tree does not exist, no aggregate parser is implemented for this model, and no verified paper row exists so the gate can only ever return `METRIC_REFERENCE_UNVERIFIED`. Nothing was reproduced; nothing here was run.

---

## EXECUTION TRACE

`paper/README/config -> source dataset -> preprocessing -> official evaluator -> workbench invocation -> checkpoint -> output -> metric comparison`

1. **paper/README/config →** Crossreref `10.1145/3404835.3462967` returns title `Comprehensive Linguistic-Visual Composition Network for Image Retrieval`, authors Wen/Song/Yang/Zhan/Nie, ACM, pages 1369-1378, 2021-07-11 — matching `models.yaml:510-512`. README `workbench/third_party/CLVCNet/README.md:1-12` states the same title/authors and `:14` the same ACM link; `:18-21` pin Python 3.7.6 / PyTorch 1.6.0 / TITAN XP / Ubuntu 14.04.6, mirrored in `models.yaml:533-534`; `:28` points at the external start-kit resize tool; `:67-71` gives the (non-functional) test command; `:74` links `CLVCNET.zip`. **Paper table: never reached (paywall) — no metric taken from it.**
2. **→ source dataset:** `train.py:70-96` `load_dataset` selects `datasets.FashionIQ(path=args.fashioniq_path, name=args.name, split=…, transform=…)` for `--dataset fashioniq`, one category per run (`README.md:31-49`). `datasets.py:152-171` fixes `image_dir = path + 'resized_image'`, `split_dir = path + 'image_splits'`, `caption_dir = path + 'captions'` and opens `cap.{name}.{split}.json` / `split.{name}.{split}.json`. Verified against the staged canonical data: `cap.dress.val.json` 2017 records with keys `candidate/captions/target`; `split.dress.val.json` 3817 unique ids; `images/` holds 74381 `.png`.
3. **→ preprocessing:** images must already be resized: `datasets.py:194-202` `get_img` opens `os.path.join(self.image_dir, self.name, image_name + ".jpg")`. The pinned tree has no resize script (`ls`), so this is the README's external tool plus the workbench recipe (`preparation_contracts.yaml:26-42`; `prepare_model.py:353-486`, 256×256, RGB/LANCZOS/JPEG, coverage receipt with per-image digests). Evaluation transform is `Resize(256) → CenterCrop(224) → ToTensor → Normalize(ImageNet)` (`train.py:90-96`). Text: `datasets.py:173-175` `concat_text` → `"<BOS> {cap0} <AND> {cap1} <EOS>"`, tokenized by `text_model.py:23-27` (ASCII-only, punctuation stripped, lowercased, split) and encoded by `:42-45` with OOV→0.
4. **→ official evaluator:** only reachable through `train.py:259` `test.test(args, local_model, global_model, dataset, args.dataset)` inside the epoch loop. `test.py:6` defines the function; the FashionIQ branch is `test.py:80-149`: features accumulated manually (`:91-119`) with `.cuda()` at `:99,115`; L2 normalisation `:121-125`; `sims = all_queries.dot(all_imgs.T)` `:129`; gallery ids from `get_test_targets()` `:85` with `gallery_all=False` default (`datasets.py:153,234-257`); reference masked `-10e10` `:134-135`; `np.argsort(-sims[i,:])[:50]` `:138`; percent R@1/10/50 `:140-148`. **There is no `__main__`, no parser, no model construction and no checkpoint load in `test.py`** (AST check; README's `python test.py` is inert).
5. **→ workbench invocation:** `workbench/backend/adapters/models.py:74-76` `CLVCNetAdapter(model_id="clvc_net", script="test.py")`; `command` not overridden → `OfficialScriptAdapter.command` `:24-25` raises `NotImplementedError`. `evaluate_models.py:113-114,195-197` skips unaudited adapters. Registry agrees: `COMMAND_BLOCKED_SOURCE_ENTRYPOINT` (`models.yaml:529`), `replay_status: BLOCKED_NO_SOURCE_REPLAY` (`:493`). Protocol resolution works (`models.yaml:523` → `src/evaluation/fashioniq.py:166-168` → `build_pair_union_gallery` `src/datasets/fashioniq.py:186-209`), and its size was independently recomputed (2628/3089/2902). **No invocation can be built.**
6. **→ checkpoint:** registry `author_drive_unresolved` / `checkpoint-unresolved.pt` / `download_url: null` / size·hash null (`models.yaml:544-555`); Drive page title `CLVCNET.zip` (observed); `checkpoint_path` (`registry.py:618-621`) resolves to a non-existent file; `runtime.py:2103-2106` would block on both "checkpoint missing" and "checkpoint mapping unresolved"; `select_checkpoints` returns BLOCKED under both policies (observed). Source writes whole `local_model.pt`/`global_model.pt` per category (`train.py:274-275`) and has no eval-time loader (`train.py:229`; `utils.py:136-140` unused).
7. **→ output:** none. `find workbench/artifacts -iname "*clvc*"` empty; only `mock/` result files exist; preflight (observed) lists missing `resized_image/{cat}` and missing ResNet-50 weights; no environment (`WORKBENCH_CLVC_PYTHON=None`). `reproduction_status: BLOCKED_CHECKPOINT_MAPPING` (`models.yaml:528`).
8. **→ metric comparison:** impossible twice over. Registry paper metrics are all null (`models.yaml:525-527`), and no verified paper row exists elsewhere in the workbench. Observed: `evaluate_gate` → `METRIC_REFERENCE_UNVERIFIED`; `extract_aggregate_metrics("clvc_net", …)` → `AGGREGATE_PARSER_UNAVAILABLE` (`metrics_extraction.py:306-320`). So there is neither a reproduced number to compare nor a published number to compare it with.

---

## CHECKLIST

1. Registered model_id and source_dir consistent with the registry — VERIFIED — `models.yaml:492` (`clvc_net`), `:517` (`source_dir: CLVCNet`), `:508` (`method_name: CLVC-Net`) — NONE — none.
2. Pinned upstream commit matches the local checkout — VERIFIED — `models.yaml:515` vs `git -C workbench/third_party/CLVCNet rev-parse HEAD` → `bd9b6889489806cb60676597880f91d918b279cb` — NONE — none.
3. Checkout is clean at the pin — VERIFIED — `git -C … status --porcelain` → only `?? __pycache__/`; `runtime.py:1998-2010` classifies that as non-dirty — NONE — none.
4. upstream_repo_url resolves to the pinned source — VERIFIED — `models.yaml:513`; `git -C … remote -v` → `git@github.com:iLearn-Lab/SIGIR21-CLVC-Net`; README author list matches — NONE — none.
5. Paper title/URL resolve to the pinned model — VERIFIED — Crossref `10.1145/3404835.3462967` title/authors/pages/year; README.md:1-14 — NONE — none.
6. Registered publication year matches the paper — VERIFIED — `models.yaml:510` `2021`; Crossref `published-print` 2021-07-11 — NONE — none.
7. Registered checkpoint is identified and byte-identifiable — MISMATCH — `models.yaml:546` `checkpoint-unresolved.pt`, `:547` `download_url: null`, `:552-553` size/hash null; `ls workbench/artifacts/checkpoints` empty — P1 — obtain the archive, record member names/sizes/hashes.
8. Checkpoint mapping to a category/benchmark condition is verified — UNKNOWN — `models.yaml:554-555` UNVERIFIED; observed `select_checkpoints` BLOCKED under both policies — P1 — keep blocked, do not invent a mapping.
9. Author archive identity verified (members, hashes) — NOT_RUN — Drive page title `CLVCNET.zip` observed only; `models.yaml:500-505` mapping UNVERIFIED with the note "do not claim six available files" — P1 — inventory the archive before any run.
10. Checkpoint serialization form is declared — VERIFIED — `models.yaml:506` `whole_model`; `train.py:274-275` `torch.save(<module>, …)` — NONE — none.
11. Per-category checkpoint policy is declared and matches the source — VERIFIED — `models.yaml:499` `independent`, `:500-505` six members; `train.py:70-96,299`; `README.md:31-49` — NONE — none.
12. A runnable checkpoint selection exists — NOT_RUN — observed `strict_clean` and `include_flagged_conditions` both `BLOCKED/BLOCKED_CHECKPOINT` (`checkpoint_selection.py:126-131, 211-221`) — P1 — requires a verified mapping plus recorded training condition.
13. Declared native protocol is the cohort the source computes — VERIFIED — `models.yaml:523` vs `datasets.py:233-257` (pair union) + `test.py:134-138` (reference masked) — NONE — none.
14. Protocol table entry matches the declared protocol — VERIFIED — `models.yaml:16-19` (`val`, `evaluation_noise_pct: 0`) vs `src/evaluation/fashioniq.py:166-168` — NONE — none.
15. literature_split_label agrees with the protocol table — VERIFIED — `models.yaml:522` `val`; enforced `registry.py:48-51` — NONE — none.
16. supported_protocols includes native_protocol — VERIFIED — `models.yaml:523-524`; enforced `registry.py:46-47` — NONE — none.
17. Gallery construction rule matches `fashioniq_val_split` semantics — VERIFIED — `datasets.py:243-249` (reference then target, first seen) vs `src/datasets/fashioniq.py:186-209` (same rule, string-keyed) — NONE — none.
18. Gallery procedure is identical, not merely equivalent — VARIANT — source dedups integer split indices (`datasets.py:222,224,239,246-249`), workbench dedups id strings (`fashioniq.py:201-207`) — P3 — record; do not assert numeric identity from this alone.
19. Gallery size re-derived from the actual registry inputs — VERIFIED — observed python: dress 2628 / shirt 3089 / toptee 2902 vs full split 3817 / 6346 / 5373, all unions ⊆ split — NONE — none.
20. Reference exclusion is implemented — VERIFIED — `test.py:134-135` sets the reference slot to `-10e10` — NONE — none.
21. Reference-exclusion procedure is identical — VARIANT — mask-then-`argsort` (`test.py:135,138`) vs sort-then-remove-column (`src/evaluation/fashioniq.py:44-52`, `fashioniq_encoder.py:44-45`) — P3 — keep documented; equivalence depends on the mask bound.
22. Tie-breaking rule matches — MISMATCH — `np.argsort(-sims[i,:])` (`test.py:138`) vs `torch.argsort(..., stable=True)` (`fashioniq_encoder.py:44`); `src/evaluation/fashioniq.py:132-135` calls ties unpinnable — P3 — relevant to any rank-level (not just R@K) comparison.
23. Recall@K values compared — VERIFIED — source `k in [1,10,50]` (`test.py:142`); workbench keys `r10/r50/mean` (`paper_gate.py:24`) — NONE — none.
24. Recall denominator matches — VERIFIED — `100 * r / len(nn_result)` (`test.py:147`, query count) vs `100.0 * sum(hits)/len(hits)` (`fashioniq_encoder.py:53`) — NONE — none.
25. Source output shape is compatible with a workbench aggregate — VARIANT — source emits `{dataname}_r{k}` for one category per run (`test.py:148`); workbench macro/mean is a workbench-defined aggregate (`src/evaluation/fashioniq.py:223-240`; `docs/FASHIONIQ_PROTOCOL_PARITY.md:95-98`) — P3 — label any aggregate as workbench-computed.
26. Evaluation-only CLI exists upstream — NOT_RUN (confirmed absent) — AST check: `has __main__: False`, `has argparse: False`, `defs: ['test']`; `test.py` is 148 lines; README's `python test.py` (`README.md:70`) is inert — P1 — record as a source blocker; never patch the pin.
27. Documented upstream test command is actually runnable — NOT_RUN — `README.md:67-71` vs `test.py` (no module-level code, no `__main__`) — P1 — a reviewed replay wrapper is the only route.
28. Adapter builds a command — NOT_RUN — `adapters/models.py:74-76` (no `command`); base raises at `:24-25`; `tests/test_protocols.py:72-77` asserts the base method for `clvc_net` — P1 — leave unimplemented until an audited wrapper exists.
29. Pinned trainer is a usable replay path — NOT_RUN — `train.py:279-306` trains from scratch; `--restore_file` declared (`train.py:62-64`) and passed (`:306`) but `train_and_evaluate` (`:229`) never reads it; `utils.py:136-140` expects a `state_dict` and is never called; artifacts are whole modules (`train.py:274-275`) — P1 — a wrapper must construct both models and load whole modules.
30. Environment stack matches the README — VERIFIED — `models.yaml:533-534` (3.7.6 / 1.6.0) vs `README.md:18-19` — NONE — none.
31. Environment confidence label is justified — VERIFIED — `models.yaml:532` `VERIFIED_README`; the two versions are literally in the pinned README — NONE — none.
32. torchvision version is declared — MISMATCH — `models.yaml:535` `UNKNOWN` while `resnet.py:3` imports `torchvision.models.utils` (path removed in later torchvision) — P2 — pin torchvision with the 1.6.0 torch before building anything.
33. CUDA requirement is declared and gated — VERIFIED — `models.yaml:540` `10.2`; `runtime.py:805-815` `_requires_gpu`; source `.cuda()` at `test.py:28,47,99,115`, `text_model.py:96,108`, `model.py:178,249` — NONE — none.
34. Evaluator can run on a CPU-only host — VERIFIED (it cannot) — unguarded `.cuda()` (above); registry comment `models.yaml:536-539`; audit host has no torch at all (`import torch` → `ModuleNotFoundError`) — P2 — blocked here by design, not a defect to "fix" by editing source.
35. Dataset layout matches what the evaluator opens — VERIFIED — `fashioniq_layout.py:44-52` == `datasets.py:157-159` + `:168,170` + `:195` (`resized_image/{cat}/{id}.jpg`, `cap.{cat}.{train,val}.json`, `split.{cat}.val.json`) — NONE — none.
36. Image extension matches the evaluator — VERIFIED — `models.yaml:519` `[".jpg"]`; `datasets.py:195` appends `".jpg"`; canonical release is `.png` (74381 observed) so resizing is mandatory — NONE — none.
37. Path-separator requirement is respected — CONCERN — `datasets.py:157-159` uses `+` so `--fashioniq_path` must end in `/` (source default does: `train.py:32`); no workbench command exists to enforce it — P3 — any future wrapper must pass a trailing separator.
38. Preparation contract is registered for the model — VERIFIED — `models.yaml:520-521`; `preparation_contracts.yaml:3` `clvc_net_fashioniq`; `registry.py:187-194` — NONE — none.
39. The declared resize artifact is reproducible from the pinned repository — PARTIAL — the pinned tree bundles no resize script (`ls`; `README.md:28` links an external tool); the workbench recipe exists (`prepare_model.py:353-486`) and only reads ids the split names — P2 — byte-level equality with the author's images remains UNKNOWN.
40. The resize recipe matches the linked tool — VARIANT — fetched start_kit: flat output, `Image.ANTIALIAS`, preserves `img.format`, default `--image_size 256`; contract/implementation: per-category, RGB, `Image.LANCZOS`, 256×256 JPEG (`preparation_contracts.yaml:26-42`; `prepare_model.py:466-478`) — P2 — declared as a workbench recipe; only defensible because the evaluator re-resizes (`train.py:91-92`).
41. External ResNet-50 weight is declared with a checksum constraint — VERIFIED — `auxiliary_assets.yaml:148-161`; `preparation_contracts.yaml:44-56` prefix `19c8e357`; matches the filename hardcoded at `resnet.py:12` — NONE — none.
42. The external asset is actually consumed by the pinned evaluator — VERIFIED — `model.py:72` `resnet.resnet50(pretrained=True)` → `resnet.py:218-223` `load_state_dict_from_url`; built in both compositors (`model.py:123,191`) — NONE — none.
43. Declared cache location matches the consumer — MISMATCH — resolver `registry.py:213-216` → `~/.cache/torch/checkpoints/resnet50-19c8e357.pth`; preflight/`load_state_dict_from_url` → `~/.cache/torch/hub/checkpoints/…`; both observed in one blocker list — P2 — harmonize the resolver or the contract suffix.
44. The `simplevocab.json` fallback is correctly classified — VERIFIED — `text_model.py:66` only when `texts_to_build_vocab is None`, which never happens here (`model.py:119-122,191-194`; `train.py:300`); `cluster_b4_classic.yaml:266-269` marks it `required_for_evaluation: false` — NONE — none.
45. Text preprocessing parity with the workbench generic composer — UNKNOWN / not on the path — source markers + `SimpleVocab` (`datasets.py:174`; `text_model.py:8-27`) vs local `ordered_and` (`src/datasets/fashioniq.py:151-156`); `docs/audits/PHASE2_SOURCES_B.md:12` notes marker spellings tokenize to `bos`/`and`/`eos`, not the reserved keys — P2 — a replay must use the native string and tokenizer; the local composer must never be substituted.
46. Vocabulary construction parity is possible — UNKNOWN — vocab is built from training captions (`train.py:300`; `datasets.py:204-212`) and baked into the saved module; rebuilding it would change token ids and embeddings — P1 — always reuse the checkpoint's own text model.
47. Augmentation/evaluation-noise contract is clean — VERIFIED — val transform is deterministic (`train.py:90-96`); randomness only in the train transform (`:78-85`); `eval()` called (`test.py:81-82`); `evaluation_noise_pct: 0` (`models.yaml:16-19,551`, enforced `registry.py:84-85`) — NONE — none.
48. Training-noise provenance for the checkpoint is established — UNKNOWN — `models.yaml:550` null; archive unverified; no source evidence — P2 — leave unrecorded until the author states it.
49. Local staging is complete for this model — NOT_RUN — observed preflight blockers: `resized_image/{dress,shirt,toptee}` missing, ResNet-50 missing (two paths), `WORKBENCH_CLVC_PYTHON` unconfigured — P1 — stage assets before any run attempt.
50. Aggregate metric extraction is implemented for this model — NOT_RUN — `metrics_extraction.py:306-320`; observed `AGGREGATE_PARSER_UNAVAILABLE / PARSER_UNAVAILABLE / None` — P1 — a parser is required before any number can be recorded.
51. The paper-metric gate can compare this model — NOT_RUN — `models.yaml:525-527` all null; no verified row elsewhere; observed `evaluate_gate` → `METRIC_REFERENCE_UNVERIFIED` (`paper_gate.py:177-180`) — P1 — transcribe the published table with citation, or declare the model non-comparable.
52. Prior workbench audit documents are consistent with the code — MOSTLY — four places assert `category_model_policy: shared` (`docs/SOURCE_TO_WORKBENCH_PARITY_MATRIX.md:25,58`; `docs/UPSTREAM_DATA_PREPARATION_AUDIT.md:133`; `docs/audits/PHASE2_SOURCES_B.md:16,20,27`) while `models.yaml:499` says `independent`; `docs/AUXILIARY_ASSET_AUDIT.md:237-254`, `docs/LEGACY_SOURCE_AUDIT.md:15-21`, `docs/LEGACY_REPRODUCTION_RUNBOOK.md:12-25`, `docs/FASHIONIQ_PROTOCOL_PARITY.md:68` all check out against the code — P3 — refresh the stale policy claims (M3).
53. UNKNOWN was never promoted to PASS — VERIFIED — items 7, 8, 9, 39, 45, 46, 48 remain UNKNOWN/PARTIAL/NOT_RUN; nothing was inferred into a pass — NONE — none.
54. No fabricated hash, URL, line number, or metric — VERIFIED — every hash/URL/line above came from an observed tool result or a pinned-file read; **no paper metric value is stated anywhere in this document because none was retrievable** — NONE — none.
55. No runtime claim was made without observation — VERIFIED — zero RUNTIME_VERIFIED claims; every runtime statement is either an observed read-only command output or explicitly labelled STATIC_AUDIT — NONE — none.

FINAL VERDICT: BLOCKED — the pin, paper identity, cohort, gallery rule, metric formula and preparation contract all check out against the pinned source, but nothing on the registered path can execute (no CLI, deliberately unimplemented adapter command, placeholder checkpoint with an uninspected author archive, contradictory/absent ResNet-50 asset, missing resized images, no metric parser) and no verified paper row exists, so the gate can only ever return `METRIC_REFERENCE_UNVERIFIED`; nothing was reproduced and nothing was run.

---

## REFRESH (this engagement)

Appended after the work that landed on top of this audit. Earlier findings above are left as
written; where one has been superseded, the superseding fact is stated here.

- **Static benchmark reference.** 1 committed row(s):

  - `clvc_net_sigir21_fiq_table1` — status `VERIFIED`, None% training noise, protocol `fashioniq_val_split`, values `{'r10': 30.7, 'r50': 58.41, 'mean': 44.56}`, provenance `{'r10': 'PAPER_REPORTED', 'r50': 'PAPER_REPORTED', 'mean': 'WORKBENCH_DERIVED'}`

- **Readiness now:** `BLOCKED_IMPLEMENTATION` — command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT

- **Reference per registered checkpoint condition:**

  - `author_drive_unresolved` (noise None): `clvc_net_sigir21_fiq_table1` status `VERIFIED` — exact condition match
