# clvc_net

Independent adversarial audit of registered model `clvc_net` (CLVC-Net) in the CIR Failure Analysis Workbench. READ-ONLY: nothing under `/home/heheboiz/data/cir` was written; no evaluator, no GPU job, no install, no checkpoint download was run. Scratch (one PDF) is under `/tmp/audit-final/clvc_net/`. Working tree audited at `eb36aa5ab0740e7dca7dd9c78515c3c49514b4e4` with `models.yaml`/`papers.yaml` modified locally (observed `git status --porcelain`). Every claim is **STATIC_AUDIT** unless labelled otherwise; this document contains **zero RUNTIME_VERIFIED** claims.

## MODEL ID

`clvc_net`.

- Registry block: `workbench/registry/models.yaml:524-587` (`- model_id: clvc_net`).
- `method_name: CLVC-Net` (`models.yaml:540`), `source_dir: CLVCNet` (`:549`), `source_available: true` (`:548`), `upstream_default_branch: main` (`:546`).
- Read back through the real loader (observed): `source_dir='CLVCNet'`, `fashioniq_layout='fashioniq_clvc_resized'`, `fashioniq_image_extensions=['.jpg']`, `literature_split_label='val'`, `native_protocol='fashioniq_val_split'`, `supported_protocols=['fashioniq_val_split']`, `checkpoint_serialization='whole_model'`, `category_model_policy='independent'`, `reproduction_status='BLOCKED_CHECKPOINT_MAPPING'`, `command_status='COMMAND_BLOCKED_SOURCE_ENTRYPOINT'`.
- Registry validation accepts the record: `layouts` includes `fashioniq_clvc_resized` (`workbench/backend/registry.py:28`), `preparation_contract` whitelist includes `manual_clvc_resized` (`registry.py:73`), `preparation_id: clvc_net_fashioniq` (`models.yaml:553`) is registered (`registry.py:75`, resolved at `:187-194` → observed `clvc_net_fashioniq`).

## PAPER TITLE

`Comprehensive Linguistic-Visual Composition Network for Image Retrieval` (`models.yaml:542`); `paper_url: https://doi.org/10.1145/3404835.3462967` (`:544`); `publication_year: 2021` (`:541`).

The paper **was obtained this round** (the previous round could not get it):

- Author-hosted camera-ready URL, from the committed extracted payload `workbench/registry/paper_benchmarks/extracted/clvc_net_sigir21.json` (`paper.paper_url`): `https://xuemengsong.github.io/SIGIR_CTI-IR.pdf`.
- Observed `curl -sL` → `http=200 size=2859422`; `sha256sum` → `4c452584eae4576c0e7b5dfbdd4356dd779c577ef258a7b9fef5086eb135af88`. This **reproduces the recorded digest and byte count exactly** (`clvc_net_sigir21.json` → `"pdf_bytes":2859422, "pdf_sha256":"4c452584…f88"`).
- `pdfinfo` (observed): title `Comprehensive Linguistic-Visual Composition Network for Image Retrieval`; author `Haokun Wen, Xuemeng Song, Xin Yang, Yibing Zhan, Liqiang Nie`; Pages 10; size 612x792. Printed page numbers 1369-1378 match `README.md:87` (`pages={1369--1378}`).
- The pinned README carries the same title and author list (`workbench/third_party/CLVCNet/README.md:1-9`) and the same DOI (`README.md:13`).

## PINNED SOURCE COMMIT

- Registry pin: `upstream_commit_sha: bd9b6889489806cb60676597880f91d918b279cb` (`models.yaml:547`).
- Observed: `git -C workbench/third_party/CLVCNet rev-parse HEAD` → `bd9b6889489806cb60676597880f91d918b279cb`. **MATCH (STATIC_AUDIT).**
- Observed `git log -1 --format="%H %ci %s"` → `bd9b688… 2026-04-05 18:54:18 +0800 Update README.md`.
- Observed `git status --porcelain` in the checkout → only `?? __pycache__/`. `runtime.py:2040-2052` (`_source_dirty`) treats `__pycache__`/`*.pyc` as non-dirty and `:2055-2079` (`source_clean_and_pinned`) accepts a clean checkout at the recorded pin. Confirmed indirectly: the observed blocker list contains no source-cleanliness or pin-mismatch entry.
- Tree: `README.md blocks.py datasets.py model.py resnet.py test.py text_model.py train.py utils.py` (observed `find`), 1994 lines total, **no requirements/lock file and no resize script**.

## CHECKPOINT IDENTITY

Single registered variant (`models.yaml:576-587`), read back as:

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

Findings:

1. **No identified, hashable checkpoint.** The filename is a literal placeholder, `download_url: null`, size and hash null. Observed `checkpoint_path('clvc_net', variant, artifacts/checkpoints)` → `/home/heheboiz/data/cir/workbench/artifacts/checkpoints/clvc_net/checkpoint-unresolved.pt`, `exists: False`; `ls -la workbench/artifacts/checkpoints` → only `.gitkeep`; `find workbench/artifacts -iname "*clvc*"` → empty. Honest bookkeeping, not an artifact claim. **UNKNOWN**, P1.
2. **What the source writes (so what the archive must contain).** `train.py:274-275`: `torch.save(local_model, …/local_model.pt)`, `torch.save(global_model, …/global_model.pt)` — two **whole pickled modules** per run. Registry agrees: `checkpoint_serialization: whole_model` (`models.yaml:537`).
3. **Six files, one pair per category.** `train.py:70-96` reads one category (`--name`) per run; `README.md:31-44` gives three separate training invocations writing `./dress`, `./shirt`, `./toptee`. Registry records exactly this and marks it unresolved: `category_model_policy: independent` (`models.yaml:531`), `category_checkpoint_structure` (`:532-537`, `member_count_expected: 6`, `mapping_status: UNVERIFIED`, note "do not claim six available files").
4. **No eval-time loader ships.** `train.py:229` declares `restore_file` as a parameter, `train.py:306` passes `args.restore_file`, and `train_and_evaluate` **never reads it** (observed `grep -n "restore_file|load_state|torch.load" train.py test.py utils.py model.py` → `train.py:62,229,306` only, plus `utils.py:136-140`). `utils.py:136-140` is a `torch.load` + `load_state_dict(checkpoint['state_dict'])` helper (DCNet lineage) that no file calls.
5. **Observed checkpoint selection** (`checkpoint_selection.select_checkpoints`, both policies):
   ```
   strict_clean               -> BLOCKED | "no checkpoint variant records a training condition, so none
                                can be called clean under policy strict_clean…"
   include_flagged_conditions -> BLOCKED | "checkpoint mapping unverified for author_drive_unresolved;
                                the upstream repository does not establish which variant is the
                                benchmark checkpoint"
   ```
   matching `checkpoint_selection.py:126-131`. `strict_clean_checkpoint_available: null` (`models.yaml:574`) is therefore consistent with the runtime computation, not with a declaration.
6. **Author archive route — new evidence this round (UNKNOWN only partially resolved).** The author's own site links a direct mirror of the same archive: observed `HEAD` on `https://237771db-2387-4fbd-9a54-dbf4c9293f62.filesusr.com/archives/be0175_d43d926b15ea442f83e1974e25cf35a1.zip?dn=CLVCNET.zip` → `HTTP/2 200`, `content-type: application/x-zip-compressed`, **`content-length: 18022`**, `last-modified: Fri, 16 Apr 2021 08:56:35 GMT`, `content-disposition: attachment; filename="CLVCNET.zip"`. That is 18 KB for an archive the registry describes as holding six whole pickled ResNet-50+LSTM models (~hundreds of MB); the plausible reading is that this mirror is **code-only** and the weights live behind the Drive link, but that is `[INFERENCE]`, not verified. **No member listing, no digest, no mapping was obtained** (I did not download or unpack).

## EVALUATION PROTOCOL

- Registry: `native_protocol: fashioniq_val_split` (`models.yaml:555`), `literature_split_label: val` (`:554`), `supported_protocols: [fashioniq_val_split]` (`:556`).
- Protocol table entry resolves to `src/evaluation/fashioniq.py:178-180` (`gallery="pair_union"`, `reference_excluded=True`, alias `fashioniq_val`).
- Router: `src/evaluation/fashioniq.py:204-219` → `build_val_gallery` (`:132-135`) → `build_pair_union_gallery` (`src/datasets/fashioniq.py:186-209`).
- **Independent re-derivation from the staged canonical dataset** (observed `python3` over `data/fashionIQ_dataset/`, walking `cap.{cat}.val.json` in file order taking `candidate` then `target`, first occurrence retained):

  | category | val queries | pair-union gallery | full `split.{cat}.val.json` | union ids absent from split |
  |---|---|---|---|---|
  | dress | 2017 | 2628 | 3817 | 0 |
  | shirt | 2038 | 3089 | 6346 | 0 |
  | toptee | 1961 | 2902 | 5373 | 0 |

  Totals: 6016 val triplets, matching the paper's "validation set of 6, 016 triplets". The union is a genuine *reduced* cohort. **VERIFIED (STATIC_AUDIT + local data).**
- The source computes this cohort via `datasets.py:233-257` (`gallery_all=False` default at `:153`, never overridden by `train.py:86-96`) plus `test.py:131-138`. **The paper never states its gallery protocol** — observed `grep -in "gallery|union|resize|candidate|6016|77,684" paper.txt` → only narrative uses of "gallery", no protocol sentence. So the registry assigns the cohort from the code, and **whether the published numbers were produced on this cohort is UNKNOWN (P2)**.
- **No runnable command exists.** `CLVCNetAdapter` (`workbench/backend/adapters/models.py:74-76`) declares `script = "test.py"` and does not override `command`, so the base `OfficialScriptAdapter.command` (`:24-25`) raises `NotImplementedError`. `evaluate_models.py:121-122` (`command_is_audited`) and `:290-291` return `["SKIPPED: adapter command not audited"]`. Observed `python3 workbench/scripts/evaluate_models.py --list | grep clvc` → `[OK] clvc_net / author_drive_unresolved: protocol=fashioniq_val_split command=not audited`. Registry agrees: `command_status: COMMAND_BLOCKED_SOURCE_ENTRYPOINT` (`models.yaml:561`), `replay_status: BLOCKED_NO_SOURCE_REPLAY` (`:525`).

## DATASET CONTRACT

Preparation contract `clvc_net_fashioniq` (`workbench/registry/preparation_contracts.yaml:3-59`) vs the pinned reader:

| Contract statement | Pinned source | Verdict |
|---|---|---|
| raw `captions/cap.{dress,shirt,toptee}.{train,val}.json` (prep:10-15) | `datasets.py:168` opens `cap.{name}.{split}.json`; `datasets.py:206` opens `cap.{name}.train.json` for vocab | VERIFIED |
| raw `image_splits/split.{dress,shirt,toptee}.val.json` (prep:16-18) | `datasets.py:170` opens `split.{name}.{split}.json` with `split='val'` (`train.py:89`) | VERIFIED |
| raw `images` (prep:19) | not opened by the evaluator; `get_img` reads the *resized* tree (`datasets.py:194-202`) | NOT consumed at eval |
| generated `resized_image/{dress,shirt,toptee}` (prep:26-42) | `datasets.py:157,195` → `<path>/resized_image/<name>/<id>.jpg` | VERIFIED |

Record shape consumed by the evaluator, checked against the real staged files (observed `python3`): `cap.dress.val.json` → 2017 records, keys `['candidate','captions','target']`. The source indexes exactly those keys (`datasets.py:184-185,219-220`). Observed disk inventory: `data/fashionIQ_dataset/images/` = 74381 files, 74381 unique ids, **all `.png`**; every val-split id (3817/6346/5373) and every train-split id (11452/19036/16121) is present on disk as `.png`, 0 missing (observed). Train triplet counts 5985+5988+6027 = **18000**, matching the paper's "training set of 18, 000 triplets".

Path joining is string concatenation, not `os.path.join`: `datasets.py:157-159`. The source's own default `--fashioniq_path` ends in `/` (`train.py:32`). A replay must pass a trailing separator; nothing in the workbench can enforce that because no command exists (P3).

Workbench side:

- Layout helper `clvc_resized_paths` (`workbench/backend/fashioniq_layout.py:44-52`) requires exactly `captions/`, `image_splits/`, `resized_image/`, `resized_image/{cat}`, `cap.{cat}.{train,val}.json`, `split.{cat}.val.json` — it **does** include the train captions a fresh constructor needs. Consumed by `fashioniq_required_paths` (`registry.py:692-696`) → `paths_for_layout` (`fashioniq_layout.py:72-73`).
- Image-directory/extension check agrees with the evaluator: layout `fashioniq_clvc_resized` → `("resized_image", True)` (per-category) (`fashioniq_layout.py:87`), `image_extension_problems` (`:92-136`), and `models.yaml:550` declares `[".jpg"]`.
- Observed local staging incomplete (real preflight against the real config):
  ```
  required generated artifact missing: data/fashionIQ_dataset/resized_image/dress
  required generated artifact missing: data/fashionIQ_dataset/resized_image/shirt
  required generated artifact missing: data/fashionIQ_dataset/resized_image/toptee
  FashionIQ image extension requirement: image directory missing: data/fashionIQ_dataset/resized_image
  ```
- **Contract gap (P3):** `workbench/scripts/prepare_dataset.py:17-18` `required_layout` is `standard_paths` only, so the dataset stage never validates the CLVC layout; the model-specific check happens in preflight instead.

## PREPROCESSING CONTRACT

- **Resize is mandatory and is not in the pinned repository.** `README.md:28` instructs resizing "follow the [XiaoxiaoGuo] … start_kit by resize_images.py"; the pinned tree bundles no such script (observed `find`: 9 files, none a resize tool).
- I fetched the linked tool (observed content of `https://raw.githubusercontent.com/XiaoxiaoGuo/fashion-iq/master/start_kit/resize_images.py`): `--image_size` default **256**, `resize_image()` uses `image.resize(size, Image.ANTIALIAS)`, output is a **flat** `--output_dir`, saved with `img.save(…, img.format)` (format preserved, no RGB conversion).
- Workbench recipe: `preparation_contracts.yaml:26-42` declares `resized_image/{dress,shirt,toptee}`, `image_size: 256`, `.jpg`, `per_category`, generator `fashioniq_resize_images`; `deterministic_status: UNKNOWN` (`:57`), `automation_policy: MANUAL_REQUIRED` (`:58`), and `:39-40` states outright it is "reported as a workbench-applied recipe, not as a run of an in-repo script".
- Implementation, re-read this round: `workbench/scripts/prepare_model.py:88-92` defines `RESAMPLING = "LANCZOS"`; `:141-144` writes `"resampling": RESAMPLING` into the receipt (context item 6 — **verified present**); `:585` does `handle.convert("RGB").resize((size, size), getattr(Image, RESAMPLING)).save(temp_file, format=save_format)` — i.e. **RGB + LANCZOS + extension-derived format**, versus the tool's preserved format and no explicit RGB conversion. It also refuses traversal ids (`:525-531`), refuses writing into the canonical `images/` tree (`:557-565`), and binds each derived file to its canonical source bytes by sha256 (`:546-550`, reuse only with a matching previous receipt at `:566-581`).
- CJK note in `prepare_model.py:460` cites `LIMN/datasets.py:112` as the reader for this layout; for CLVC-Net the equivalent reader is `datasets.py:195`. Cosmetic citation slip in a shared helper (P3).
- **Level-2 mismatch, P2:** documented tool = flat, ANTLIAS (= LANCZOS in modern PIL), format-preserving; contract/implementation = per-category, RGB, LANCZOS, JPEG. Consequence at evaluation is bounded because the pinned val transform is `Resize(256) → CenterCrop(224)` (`train.py:91-92`), but **byte equality with the author's images is UNKNOWN**.
- **External weight.** `auxiliary_assets.yaml:148-161` declares torchvision ResNet-50 ImageNet weights, `source_repo: https://download.pytorch.org/models/resnet50-19c8e357.pth`, `destination_path: hub/checkpoints/resnet50-19c8e357.pth` (`:156`), `expected_sha256_prefix: 19c8e357` (`:160`), `provenance_status: VERIFIED`, `automatic_download_available: true`. `preparation_contracts.yaml:44-56` declares the same asset (`cache_root_env_var: TORCH_HOME`, `cache_path_suffix: checkpoints/resnet50-19c8e357.pth`, `required_for_evaluation: true`, `required_for_preparation: false`, `optional: false`). Consumer verified: `model.py:72` `resnet.resnet50(pretrained=True)` → `resnet.py:218-223` `load_state_dict_from_url(model_urls[arch], …)`, with the URL hardcoded at `resnet.py:12`; `img_backbone` is built inside **both** `compose_local` (`model.py:123`) and `compose_global` (`model.py:195`).
- **Cache-path defect (observed, not a hypothesis).** `external_asset_statuses` (`registry.py:208-216`) defaults the root to `Path(os.environ.get("XDG_CACHE_HOME","~/.cache")).expanduser() / "torch"` and then appends `cache_path_suffix = checkpoints/resnet50-19c8e357.pth` → `~/.cache/torch/checkpoints/…`. `preflight.py:104-112` and `runtime_plan_blockers` report `~/.cache/torch/hub/checkpoints/…` (`auxiliary_assets.yaml:156`), and `load_state_dict_from_url` uses `$TORCH_HOME/hub/checkpoints`. Both strings appear in the same observed blocker list:
  ```
  torchvision ResNet-50 ImageNet weights missing: /home/heheboiz/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth
  external runtime asset file missing: torchvision ResNet-50 ImageNet weights (/home/heheboiz/.cache/torch/checkpoints/resnet50-19c8e357.pth)
  ```
  The registry *resolver* points at a path the consumer never reads, and the operator is shown two contradictory paths for one file. **P2.** `preparation_contracts.yaml:56` repeats the wrong location in prose.
- `simplevocab.json`: `text_model.py:66` reads it only when `texts_to_build_vocab is None`; both constructions pass `texts` (`model.py:119-122`, `:191-194`) and `train.py:300` supplies `trainset.get_all_texts()`; the only other `TextLSTMModel` mentions in the tree are its definition and those two calls (observed `grep -rn TextLSTMModel *.py`). Consistent with the asset audit marking it not required for evaluation (`registry/assets/cluster_b4_classic.yaml:266-269`).

## AUGMENTATION CONTRACT

- Training path (`train.py:78-85`): `Resize(256)`, `RandomCrop(224)`, `RandomHorizontalFlip()`, `ToTensor`, ImageNet `Normalize`. Validation path (`train.py:90-96`): `Resize(256)`, `CenterCrop(224)`, `ToTensor`, `Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])`. `test.py` applies no transform of its own; it consumes tensors produced by the dataset object (`test.py:94,111`).
- `test.py:81-82` sets `local_model.eval()` / `global_model.eval()` before the no-grad pass — the model is *not* left in train mode (contrast the HABIT/CSMCIR-family defects recorded elsewhere). Dropout therefore inactive (`model.py:128`, and the global compositor at `:200`).
- Registry side: `evaluation_noise_pct: 0` for the checkpoint variant (`models.yaml:583`); `registry.py:84-85` refuses any non-zero evaluation noise. Protocol entry carries `evaluation_noise_pct: 0` for `fashioniq_val_split`. **VERIFIED**, no augmentation noise enters evaluation.
- Evaluation determinism: `train.py:287-293` seeds torch/cuda/numpy and sets `cudnn.deterministic=True, benchmark=False`, but only on the training path. Evaluation determinism still depends on `np.argsort` tie behaviour, and `checkpoint_training_noise_pct: null` means **the registered weights' training condition is unrecorded** (P2).

## GALLERY CONTRACT

Source (FashionIQ branch of `test.py`):

- Queries: `testset.get_test_queries()` (`test.py:84`), built by `datasets.py:214-230`: per annotation, `source_img_id = self.images.index(candidate)`, `target_img_id = self.images.index(target)` — ids are **indices into the full split list**, not strings.
- Gallery: `testset.get_test_targets()` (`test.py:85`); with the constructor default `gallery_all=False` (`datasets.py:153`, never overridden by `train.py:86-96`) the `else` branch runs (`datasets.py:242-257`): reference first, then target, first occurrence retained — source-order pair union.
- Reference exclusion: `test.py:131-135` builds `test_targets_id` from the gallery and sets `sims[i, test_targets_id.index(t['source_img_id'])] = -10e10` **before** `np.argsort(-sims[i,:])[:50]` (`test.py:138`). Mask, not removal. `-10e10` is below any plausible dot product of L2-normalized features, so top-k behaviour matches removal — but they are different operations and the equivalence depends on that bound (`src/evaluation/fashioniq.py:44-52` argues the same).
- Ranking depth: `[:50]` (`test.py:138`) and recall over `[1,10,50]` (`test.py:142`) — k=10/50 sit inside the truncated list, so truncation is not metric-affecting here.

Workbench: `fashioniq_val_split` = `pair_union` + `reference_excluded` (`src/evaluation/fashioniq.py:178-180`); gallery built by the same first-seen reference-then-target rule but keyed by **id strings** (`src/datasets/fashioniq.py:195-209`); exclusion by **removing** the reference column after a sort (`src/evaluation/fashioniq.py:35-77`); `recall_at_k_reference_excluded` uses `torch.argsort(scores[row], descending=True)` with **no `stable=True`** (`:71`), while the encoder-family helper uses `stable=True` (`src/evaluation/fashioniq_encoder.py:44`). `recall_at_k` asserts gallery uniqueness and target membership (`fashioniq.py:9-33`).

**Verdict:** the cohort *rule* matches (VERIFIED); the *procedure* is a documented VARIANT (int-index dedup vs id-string dedup; mask-then-sort vs sort-then-remove; `np.argsort` vs `torch.argsort` stability). P3, with the standing instruction never to assert numeric identity from procedure equivalence.

## METRIC CONTRACT

- Source: `test.py:140-148` — for `k in [1,10,50]`, hit test `test_targets_id.index(test_queries[i]['target_img_id']) in nns[:k]`, `r = 100 * r / len(nn_result)`, emitted as `out += [('{}_r{}'.format(dataname, k), r)]` — i.e. `fashioniq_r1`, `fashioniq_r10`, `fashioniq_r50` for the single category under `--name`.
- Workbench extraction: `metrics_extraction.py:309-341` — only `limn` (`:328-329`) and `csmcir` (`:330-331`) have parsers; every other model, `clvc_net` included, returns `AGGREGATE_PARSER_UNAVAILABLE` (`:333-341`) with `parity_status="PARSER_UNAVAILABLE"`.
- Workbench aggregate: `macro_average_fashioniq` (`src/evaluation/fashioniq.py:238-255`) → `mean_recall = (recall_at_10 + recall_at_50)/2`. The paper prints an **Avg R@10 / Avg R@50 column pair** but **no (R@10+R@50)/2 number**; the registry's `mean` is therefore `WORKBENCH_DERIVED` with formula `(30.70 + 58.41) / 2` (`papers.yaml:556-566`).
- Parity annotation: `_check_parity` (`metrics_extraction.py:28-61`) now routes through `paper_gate.compare_metric` (5 % relative), as the context states — **verified by reading**. It returns `PARITY_NOT_EVALUATED` the moment any of `r10/r50/mean` is missing on either side (`:52-56`). The remaining `0.05` literals are absolute self-consistency checks inside the LIMN parser (`:186`) and the CSMCIR parser (`:275`) — declared intentional in comments at `:186-188` and `:276-279`.
- Gate: `paper_gate.py:27` (`TOLERANCE_PCT = 5.0`), `:29` (`r10/r50/mean`), `:35-38` (`dress/shirt/toptee x r10/r50`), `:163-201` (`compare_metric`), `:316-388` (ordering, category comparison).

Observed gate behaviour for `clvc_net` (arithmetic only — no run exists; I fed synthetic observed numbers to the real code to exercise the decision path):

```
select_reference('clvc_net', None, 'fashioniq_val_split') -> clvc_net_sigir21_fiq_table1 | exact condition match

run_meta = {protocol fashioniq_val_split, checkpoint author_drive_unresolved,
            checkpoint_training_noise_pct: null, evaluation_noise_pct: 0, split val, dataset fashioniq}

evaluate_gate(row, {r10:30.0, r50:58.0, mean:44.0}, …,
              reproduced_category_metrics={dress 29.8/56.4, shirt 28.7/54.7, toptee 33.4/63.9})
 -> PASS_WITHIN_5_PERCENT  (9 comparisons: r10 2.28%, r50 0.70%, mean 1.26%,
                            dress_r10 0.17%, dress_r50 0.12%, shirt_r10 0.17%, shirt_r50 0.11%,
                            toptee_r10 0.30%, toptee_r50 0.16%; provenance r10/r50 PAPER_REPORTED,
                            mean WORKBENCH_DERIVED)

evaluate_gate(row, {r10:20.0, r50:40.0, mean:30.0}, …, reproduced_category_metrics=None)
 -> ALERT_METRIC_DEVIATION  ("r10 differs by 34.8534%", "r50 differs by 31.5186%")
     detail: …; category metrics dress_r10, … toptee_r50 are published by the paper but the run
             observed none, so they were not compared
```

So category metrics are compared exactly as context item 1 states (`paper_gate.py:348-363`), and an unobserved paper category is recorded in `detail`, never as a match (`:382-384`). This is a real behavioural difference from the previous round, where this model's gate could only ever return `METRIC_REFERENCE_UNVERIFIED`.

**However, for `clvc_net` the gate is unreachable in practice:** the only producer of `extraction.observed_metrics` for this model is `extract_aggregate_metrics`, which returns `AGGREGATE_PARSER_UNAVAILABLE` with `observed_metrics=None` (`metrics_extraction.py:333-341`). `evaluate_models.py:158-170` passes `observed` straight from that object, so `evaluate_gate` hits `if not reproduced_metrics: return FAILED_OUTPUT_VALIDATION` (`paper_gate.py:301-303`) — the run is judged as "evaluator produced nothing to compare" even if the pinned evaluator printed perfect recalls, because nothing reads them.

## SOURCE-TO-WORKBENCH PARITY

| Component | Source evidence | Workbench | Status |
|---|---|---|---|
| Protocol/cohort | `datasets.py:242-257`, `test.py:131-138` | `models.yaml:555`, `src/evaluation/fashioniq.py:178-180` | EQUIVALENT_WITH_EVIDENCE (re-derived sizes 2628/3089/2902) |
| Gallery dedup key | ints `datasets.py:222,224,239,246-249` | id strings `src/datasets/fashioniq.py:195-207` | VARIANT (P3) |
| Reference exclusion | mask `-10e10` pre-sort `test.py:135,138` | remove column post-sort `fashioniq.py:70-75` | VARIANT (P3) |
| Ties | `np.argsort(-sims)` `test.py:138` | `torch.argsort(...)` no `stable` `fashioniq.py:71` (vs `stable=True` in `fashioniq_encoder.py:44`) | MISMATCH (P3) |
| Preparation contract | `README.md:28` external tool; tree has no script | `preparation_contracts.yaml:26-42`, `prepare_model.py:451-593` | VARIANT (P2, recipe not a run of the tool) |
| Text pipeline | markers + `SimpleVocab` (`datasets.py:174`, `text_model.py:23-45`) | local `compose_fashioniq_caption` — **not invoked for this model** | NOT_YET_VERIFIED / not on the path |
| Evaluator command | `test(params, local_model, global_model, testset, dataname)` `test.py:6` | adapter inherits base `command` (`adapters/models.py:24-25,74-76`) | **NO PARITY POSSIBLE — command does not exist** |
| Checkpoint consumption | `test.py:6` takes constructed models; `train.py:229` never reads `restore_file`; `utils.py:136-140` unused | `category_model_policy: independent` `models.yaml:531` | INCOMPLETE_IMPLEMENTATION (P1) |
| Metric extraction | `test.py:148` prints `fashioniq_r{k}` per category | `metrics_extraction.py:333-341` | NOT IMPLEMENTED (P1) |
| Paper reference | Table 1 page 1375 | `papers.yaml:530-588` row `VERIFIED` | MATCH (new this round) |
| Registry headline fields | same paper row | `models.yaml:557-559` **all null** | MISMATCH (P2) |

## RUNTIME STATUS

Observed real preflight + runtime blockers against the real config:

```
required generated artifact missing: /home/heheboiz/data/cir/data/fashionIQ_dataset/resized_image/dress
required generated artifact missing: …/resized_image/shirt
required generated artifact missing: …/resized_image/toptee
FashionIQ image extension requirement: image directory missing: …/fashionIQ_dataset/resized_image
torchvision ResNet-50 ImageNet weights missing: /home/heheboiz/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth
external runtime asset file missing: torchvision ResNet-50 ImageNet weights (/home/heheboiz/.cache/torch/checkpoints/resnet50-19c8e357.pth)
command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT
execution environment not verified: WORKBENCH_CLVC_PYTHON=None [UNCONFIGURED/UNCONFIGURED] (WORKBENCH_CLVC_PYTHON not set)
— plus, from runtime_plan_blockers —
checkpoint missing: /home/heheboiz/data/cir/workbench/artifacts/checkpoints/clvc_net/checkpoint-unresolved.pt
checkpoint mapping unresolved
```

- GPU: `environment.cuda: '10.2'` (`models.yaml:572`) and the evaluator calls `.cuda()` unconditionally at `test.py:28,47,99,115` plus `text_model.py:96,108` and `model.py:178,249`. Observed host state (`pc_readiness.py --model clvc_net --json`): `nvidia_smi: true`, one device `NVIDIA GeForce RTX 3050 Laptop GPU, 8.6, 4096 MiB`, but `torch: null`, `torch_error: "ModuleNotFoundError: No module named 'torch'"`. The pinned evaluator cannot run here.
- `python3 workbench/scripts/pc_readiness.py --model clvc_net --json` (observed) → `"state": "BLOCKED_IMPLEMENTATION"`, `"reason": "command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT"`, `"approved_patch": null`, `"warnings": []`, and `benchmark_references: [{checkpoint_id author_drive_unresolved, reference_status VERIFIED, reason "exact condition match"}]`. Note the first-match classifier (`pc_readiness.py:44-60`) puts `("command not runnable", BLOCKED_IMPLEMENTATION)` before the dataset/asset needles, so the implementation blocker masks the dataset and external-asset blockers that are also reported in the list — cosmetic, the list itself is complete.
- `docs/READINESS_MATRIX.md:58` records `SOLVER_COMPATIBLE` environment planning and `BLOCKED` evaluation for `clvc_net`, with the resized-image and ResNet-50 blockers — consistent with the above.
- No result artifact exists: `find workbench/artifacts -iname "*clvc*"` → empty. **RUNTIME STATUS = NOT_RUN (nothing executed, nothing can be).**

## PAPER-METRIC STATUS

**VERIFIED, first time for this model.** The committed row `clvc_net_sigir21_fiq_table1` (`workbench/registry/paper_benchmarks/papers.yaml:530-588`) is `reference_status: VERIFIED`, `review_status: CONFIRMED`, `evaluation_noise_pct: 0`, `split: val`, `gallery_protocol: fashioniq_val_split`, `checkpoint_training_noise_pct: null`, and I re-derived every number from the real paper:

- `pdftotext -layout` row (`paper.txt:675`):
  ```
  CLVC-Net   29.85  56.47  28.75  54.76  33.50  64.00  30.70  58.41  17.64  54.39  79.47
  ```
  Column headers (`paper.txt:665-666`): `FashionIQ | Dress | Shirt | Tops&Tees | Avg | Shoes`, `R@10 R@50 | R@10 R@50 | R@10 R@50 | R@10 R@50 | R@1 R@10 R@50`. So FashionIQ Dress 29.85/56.47, Shirt 28.75/54.76, Tops&Tees 33.50/64.00, Avg R@10 30.70 / R@50 58.41 — **exactly** the registry's `category_metrics` and `reported_metrics` (`papers.yaml:552-575`). Table 1 is printed on **page 1375** (PDF page 7), while the registry says `paper_page: 5` — minor localisation error (P3).
- Table 3 (`paper.txt:743`) repeats `CLVC-Net 30.70 58.41` as the ablation baseline on "FashionIQ (Avg)", printed page **1376**, matching `method_variant`'s claim (`papers.yaml:535-536`).
- Cross-check: the printed averages are exactly the category means — (29.85+28.75+33.50)/3 = 30.70; (56.47+54.76+64.00)/3 = 58.41. The registered `derived_metrics.mean` = 44.56 = (30.70+58.41)/2, labelled `WORKBENCH_DERIVED` with its formula (`papers.yaml:556-566`), and `provenance_for_all()` observed as `{'r10': 'PAPER_REPORTED', 'r50': 'PAPER_REPORTED', 'mean': 'WORKBENCH_DERIVED'}`.
- The gate can now compare this model: observed `PASS_WITHIN_5_PERCENT` (9 comparisons) and `ALERT_METRIC_DEVIATION` on synthetic paper-exact/deviating inputs respectively.
- Residual honesty limits: the paper **states no gallery protocol, no resize step, no CLIP/text-encoder variant, and no checkpoint condition** (observed: 0 occurrences of `clip` case-insensitively; no `gallery`/`union`/`resize` protocol sentence). The registry's `checkpoint_training_noise_pct: null` and `checkpoint_mapping_status: UNVERIFIED` are therefore correct.

## CRITICAL FINDINGS

**C1 — No source-backed evaluation entrypoint exists, and the adapter deliberately refuses (P1).** `workbench/third_party/CLVCNet/test.py` is 148 lines containing exactly one top-level function; AST inspection (observed) → `top-level nodes: ['Import','Import','ImportFrom','Import','FunctionDef']`, `funcs: ['test']`, `__main__ in src: False`, `argparse in src: False`. The README's `python test.py` (`README.md:67-71`) therefore runs *nothing*. A replay must independently construct both `compose_local(texts, T, dropout_p)` and `compose_global(...)` (`model.py:115-122`, `:187-194`) with `texts = trainset.get_all_texts()` as `train.py:300` does. `CLVCNetAdapter` (`adapters/models.py:74-76`) does not override `command`, so the base raises (`:24-25`); `evaluate_models.py:121-122,290-291` skips it; registry records `COMMAND_BLOCKED_SOURCE_ENTRYPOINT` (`models.yaml:561`) and preflight emits it (`preflight.py:205-206`). Recommended: keep blocked; do not patch the pin.

**C2 — The registered checkpoint is a placeholder and the author archive's interior is unresolved (P1, partially narrowed this round).** `checkpoint-unresolved.pt`, `download_url: null`, size/hash null (`models.yaml:577-587`); selection BLOCKED under both policies (observed). New: the author's own site exposes an 18,022-byte direct mirror of `CLVCNET.zip` (observed `HEAD`), far too small to contain six whole pickled models — plausible code-only, but **not verified**. Recommended: obtain the actual weights archive, inventory members, hash each, and record the file→category→local/global mapping before any "checkpoint available" claim.

**C3 — No text-model loader, so a whole-model pickle is the only defensible artifact (P1).** `test.py:6` receives constructed models; `train.py:229` accepts `restore_file` and never reads it; the only loader in the tree, `utils.py:136-140`, expects a `state_dict` and is never called. The vocabulary is built from training captions (`train.py:300`, `datasets.py:204-212`, `text_model.py:62-64`), so a rebuilt vocabulary yields different token ids and different embeddings. Any replay must `torch.load` the whole pickled module.

**C4 — The metric parser gap makes the now-working gate unreachable for this model (P1).** `metrics_extraction.py:333-341` has no `clvc_net` parser; `observed extract_aggregate_metrics("clvc_net", …)` → `AGGREGATE_PARSER_UNAVAILABLE / PARSER_UNAVAILABLE / None`. `evaluate_models.py:158-170` hands that `None` to `evaluate_gate`, which returns `FAILED_OUTPUT_VALIDATION` at `paper_gate.py:301-303`. Also: the paper publishes **per-category** metrics for this row, and the run would have to publish them too, so a parser that only scrapes a single `fashioniq_r10` line cannot satisfy `paper_gate.py:348-363`. Recommended: implement a source-backed parser for `test.py:148` output covering per-category values.

**C5 — External ResNet-50 asset absent and resolved to two contradictory paths (P2).** `registry.py:208-216` → `~/.cache/torch/checkpoints/resnet50-19c8e357.pth`; `auxiliary_assets.yaml:156` + `preflight.py:104-112` → `~/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth`; the consumer (`resnet.py:218-223`) reads the latter. Both strings observed in one blocker list.

## MINOR FINDINGS

**M1 — Registry headline `reported_r10/r50/mean` are still null although a verified paper row now exists (P2).** `models.yaml:557-559` (observed through `model_by_id`), while `papers.yaml:552-555` publishes `30.7 / 58.41` and derives `44.56`. Consequences: `evaluate_models.py:309` builds `paper_metrics` from those nulls, `_check_parity` then returns `PARITY_NOT_EVALUATED`, and `autopilot.py:620-622,772-775` write `None` into every run row. The **gate** path uses the real row, so two paper-number paths now disagree inside one system. Recommended: fill or drop the three fields.

**M2 — `backbone: "ResNet-50 visual encoder with a CLIP-based textual branch (paper §3)"` is unsupported by the paper (P2).** `papers.yaml:537-538`; observed `grep -ic clip paper.txt` → `0`. Paper §3.2.1 and the source describe two **LSTMs** (`model.py:119-122,191-194`). Recommended: replace with the source-backed string.

**M3 — Resize recipe is a workbench recipe, not the README's tool (P2).** See PREPROCESSING.

**M4 — `torchvision: UNKNOWN` although the source depends on a removed import path (P2).** `models.yaml:567`; `resnet.py:3` imports `torchvision.models.utils`. Pin torchvision with torch 1.6.0 or record the constraint as the reason.

**M5 — Stale `category_model_policy: shared` claims remain in three documents (P3).** `docs/SOURCE_TO_WORKBENCH_PARITY_MATRIX.md:25,58`, `docs/audits/PHASE2_SOURCES_B.md:16,20,27`; registry says `independent` (`models.yaml:531`).

**M6 — `OPEN_BLOCKERS.md:160,163` stale (P3).** `test.py` is reported as "148 lines" (observed `wc -l` → 148; 149 `splitlines()` because the last line lacks a trailing newline), and `:163` says preflight reports `COMMAND_UNAUDITED`, but this model reports `COMMAND_BLOCKED_SOURCE_ENTRYPOINT` (`preflight.py:206`).

**M7 — No category-model routing code exists, unlike LIMN (P3).** `workbench/replay/` = `common.py`, `limn.py`, `tgcir.py`; no `clvc.py`.

**M8 — Dataset-stage validation does not cover this layout (P3).** `prepare_dataset.py:17-18` validates `standard_paths` only.

**M9 — Trailing-slash coupling recorded but unenforceable (P3).** `datasets.py:157-159`; `registry/assets/cluster_b4_classic.yaml:244,290` records the requirement.

**M10 — `metric_provenance.paper_page: 5` is wrong (P3).** Table 1 is on printed page 1375 = PDF page 7; Table 3 on 1376.

**M11 — The committed reference row does not carry the paper URL or PDF digest (P3).** `build_paper_registry.py:27-34` `RUNTIME_FIELDS` excludes `paper_url`/`pdf_sha256`; observed the committed row keys contain neither, though `extracted/clvc_net_sigir21.json` has both. Recommended: carry them so `as_evidence()` is self-contained.

## UNRESOLVED EVIDENCE

1. The interior of the weights archive behind `https://drive.google.com/file/d/159rBhWyHK…`: member names, count, sizes, hashes, and the file→category→local/global mapping. Not resolvable read-only. The 18,022-byte `filesusr.com` mirror is a different artefact of unknown contents.
2. Whether the author's resized images are byte-identical to the workbench recipe's output.
3. Whether the published numbers were produced on the pair-union val cohort: the code computes that cohort; the paper never states a gallery protocol.
4. Whether `--restore_file` was ever functional in the authors' private tree.
5. `WORKBENCH_CLVC_PYTHON`: no interpreter recorded or configured anywhere on this host; no environment exists to inspect.
6. torchvision/CUDA version actually used by the authors (`README.md:20-21`; no versions).
7. Whether the 18 KB `filesusr.com` `CLVCNET.zip` is code-only or a stub.

## FINAL VERDICT

**BLOCKED.** The pin (`bd9b688…`), the paper identity and **every published number** — Table 1 page 1375 row `29.85 56.47 28.75 54.76 33.50 64.00 30.70 58.41` (PDF sha256 `4c452584…f88` reproduced) — plus the cohort, gallery rule, metric formula and preparation contract were all independently re-derived and agree with the pinned source (including a fresh recomputation of the pair-union gallery sizes 2628/3089/2902 against full splits 3817/6346/5373, and of 18000/6016 triplets), and the gate can now genuinely judge this model (observed `PASS_WITHIN_5_PERCENT` on paper-exact synthetic numbers, `ALERT_METRIC_DEVIATION` otherwise); but the registered path still cannot start: the pinned evaluator has no CLI or loader, the adapter deliberately raises, the checkpoint is a placeholder with an uninspected author archive, no parser exists for this model's output so the gate would report `FAILED_OUTPUT_VALIDATION` even on a perfect run, the ResNet-50 asset is absent and resolved to two contradictory paths, and the resized-image tree does not exist. Nothing was reproduced; nothing was run.

---

## EXECUTION TRACE

`paper/README/config -> source dataset -> preprocessing -> official evaluator -> workbench invocation -> checkpoint -> output -> metric comparison`

1. **paper/README/config →** `papers.yaml:530-588` row `clvc_net_sigir21_fiq_table1`, `paper_id: clvc_net_sigir21`, `VERIFIED`; registry block `models.yaml:524-587`. The paper is real and reachable: `curl -sL https://xuemengsong.github.io/SIGIR_CTI-IR.pdf` → HTTP 200, 2859422 bytes, sha256 `4c452584eae4576c0e7b5dfbdd4356dd779c577ef258a7b9fef5086eb135af88` — identical to `extracted/clvc_net_sigir21.json`'s `pdf_sha256`/`pdf_bytes`; `pdfinfo` title/authors/pages match `models.yaml:540,542`. Pinned README states the same title, authors and DOI (`README.md:1-13`), and pins Python 3.7.6 / PyTorch 1.6.0 / TITAN XP / Ubuntu 14.04.6 (`README.md:18-21`).
2. **→ source dataset:** `train.py:70-96` `load_dataset` selects `datasets.FashionIQ(path=args.fashioniq_path, name=args.name, split=…, transform=…)` for `--dataset fashioniq`, one category per run (`README.md:31-44`). `datasets.py:152-171` fixes `image_dir = path + 'resized_image'`, `split_dir = path + 'image_splits'`, `caption_dir = path + 'captions'` and opens `cap.{name}.{split}.json` / `split.{name}.{split}.json`. Verified against staged canonical data: `cap.dress.val.json` 2017 records with keys `candidate/captions/target`; `split.dress.val.json` 3817 unique ids; `images/` = 74381 `.png`, all val and train split ids present (0 missing); triplet totals 18000 train / 6016 val = the paper's counts.
3. **→ preprocessing:** images must already be resized — `datasets.py:194-202` `get_img` opens `os.path.join(self.image_dir, self.name, image_name + ".jpg")`. The pinned tree bundles no resize script, so this is the README's external tool (`README.md:28`, fetched: flat output, `Image.ANTIALIAS`, format preserved, default 256) plus the workbench recipe (`preparation_contracts.yaml:26-42`; `prepare_model.py:451-593`: 256x256, RGB, `Image.LANCZOS`, extension-derived save format, per-image sha256 binding of derived→canonical bytes, `resampling: LANCZOS` in the receipt at `:141-144`). Evaluation transform is `Resize(256) → CenterCrop(224) → ToTensor → Normalize(ImageNet)` (`train.py:90-96`). Observed locally: `resized_image/` does not exist → blocker. Text: `datasets.py:173-175` `concat_text` → `"<BOS> {cap0} <AND> {cap1} <EOS>"`, tokenized by `text_model.py:23-27` and encoded by `:42-45` with OOV→0.
4. **→ official evaluator:** reachable only through `train.py:259` `test.test(args, local_model, global_model, dataset, args.dataset)` inside the epoch loop. `test.py:6` defines the function; the FashionIQ branch is `test.py:80-149`: features accumulated manually (`:91-119`) with `.cuda()` at `:99,115`; L2 normalisation `:121-125`; `sims = all_queries.dot(all_imgs.T)` `:129`; gallery ids from `get_test_targets()` `:85` with `gallery_all=False` default (`datasets.py:153,234-257`); reference masked `-10e10` `:134-135`; `np.argsort(-sims[i,:])[:50]` `:138`; percent R@1/10/50 `:140-148`. **There is no `__main__`, no parser, no model construction and no checkpoint load in `test.py`** (AST observed; README's `python test.py` is inert).
5. **→ workbench invocation:** `adapters/models.py:74-76` `CLVCNetAdapter(model_id="clvc_net", script="test.py")`; `command` not overridden → base raises (`:24-25`). Observed `evaluate_models.py --list` → `command=not audited`; `guarded_plan` returns `["SKIPPED: adapter command not audited"]` (`:290-291`). Registry agrees: `COMMAND_BLOCKED_SOURCE_ENTRYPOINT` (`models.yaml:561`), `replay_status: BLOCKED_NO_SOURCE_REPLAY` (`:525`). Protocol resolution works (`models.yaml:555` → `src/evaluation/fashioniq.py:178-180` → `build_pair_union_gallery` `src/datasets/fashioniq.py:186-209`) and its size was independently recomputed (2628/3089/2902). **No invocation can be built.**
6. **→ checkpoint:** registry `author_drive_unresolved` / `checkpoint-unresolved.pt` / `download_url: null` / size·hash null (`models.yaml:576-587`); `checkpoint_path` resolves to a non-existent file (observed); `runtime_plan_blockers` emits `checkpoint missing:` and `checkpoint mapping unresolved`; `select_checkpoints` BLOCKED under both policies (observed); the runtime writes whole `local_model.pt`/`global_model.pt` per category (`train.py:274-275`) and has no eval-time loader (`train.py:229`; `utils.py:136-140` unused).
7. **→ output:** none. `find workbench/artifacts -iname "*clvc*"` empty; `artifacts/checkpoints` holds only `.gitkeep`; preflight (observed) lists missing `resized_image/{cat}` and missing ResNet-50 weights at two contradictory paths; no environment (`WORKBENCH_CLVC_PYTHON=None`); `reproduction_status: BLOCKED_CHECKPOINT_MAPPING` (`models.yaml:560`).
8. **→ metric comparison:** the reference side now exists and was re-derived from the real paper (row `29.85 56.47 28.75 54.76 33.50 64.00 30.70 58.41`, Table 1 p.1375; category mean arithmetic exact; `mean` derived as `(30.70+58.41)/2 = 44.56`), and the gate demonstrably compares (observed `PASS_WITHIN_5_PERCENT` with 9 comparisons on paper-exact synthetic numbers; `ALERT_METRIC_DEVIATION` on a 30 % deviation; per-category keys compared only when the run publishes them). The reproduced side cannot exist: `extract_aggregate_metrics("clvc_net", …)` → `AGGREGATE_PARSER_UNAVAILABLE / PARSER_UNAVAILABLE / None`, so `evaluate_gate` would stop at `FAILED_OUTPUT_VALIDATION` (`paper_gate.py:301-303`).

---

## CHECKLIST

1. Registered model_id / source_dir consistent — VERIFIED — `models.yaml:524,549,540`; read back through `model_by_id` — NONE — none.
2. Pinned commit matches the local checkout — VERIFIED — `models.yaml:547` vs `git -C workbench/third_party/CLVCNet rev-parse HEAD` → `bd9b6889489806cb60676597880f91d918b279cb` — NONE — none.
3. Checkout clean at the pin — VERIFIED — `git status --porcelain` → only `?? __pycache__/`; `runtime.py:2040-2052` classifies that non-dirty — NONE — none.
4. `upstream_repo_url` resolves to the pinned source — VERIFIED — `models.yaml:545`; README author list `README.md:1-9` agrees — NONE — none.
5. Paper title/DOI resolve to the pinned model — VERIFIED — paper fetched (sha256 `4c452584…f88`), `pdfinfo` title/authors match `models.yaml:540,542,544` — NONE — none.
6. Publication year matches — VERIFIED — `models.yaml:541` `2021`; SIGIR '21, printed pages 1369-1378 — NONE — none.
7. **Paper numbers re-derived from the real paper** — VERIFIED — `pdftotext -layout` `paper.txt:661-675`: `CLVC-Net 29.85 56.47 28.75 54.76 33.50 64.00 30.70 58.41`; equals `papers.yaml:552-575` exactly; `(29.85+28.75+33.50)/3=30.70`, `(56.47+54.76+64.00)/3=58.41` — NONE — none.
8. Derived `mean` provenance and value sound — VERIFIED — `papers.yaml:556-566`; observed `value_for('mean')` = 44.56, `provenance_for_all()` = `{r10 PAPER_REPORTED, r50 PAPER_REPORTED, mean WORKBENCH_DERIVED}` — NONE — none.
9. Paper page localisation in the reference row — MISMATCH — `paper_page: 5`; Table 1 is on printed page 1375 (PDF page 7) — P3 — correct the page number.
10. Paper table identity (`Table 1` / row `CLVC-Net`) — VERIFIED — `paper.txt:661,675`; Table 3 repeats `30.70 58.41` (`paper.txt:743`) — NONE — none.
11. Committed reference status is `VERIFIED` and selectable — VERIFIED — observed `select_reference('clvc_net', None, 'fashioniq_val_split')` → `clvc_net_sigir21_fiq_table1 | exact condition match` — NONE — none.
12. Reference states the comparability conditions it is judged on — VERIFIED — states `gallery_protocol`, `checkpoint_id`, `split`, `dataset`, `evaluation_noise_pct`; observed gate detail reported exactly 1 unchecked condition (`checkpoint_training_noise_pct`) and did not return `BLOCKED_REFERENCE_INCOMPLETE` (`paper_gate.py:331-334`) — NONE — none.
13. **Per-category metrics compared by the gate when both sides publish them** — VERIFIED — `paper_gate.py:348-363`; observed 9 comparisons including `dress_r10 … toptee_r50` with relative errors 0.11-0.30 % on synthetic inputs; a paper category with no observed counterpart is listed in `detail`, never a match (`:382-384`) — NONE — none.
14. **Legacy 0.05 parity check routed through `compare_metric`** — VERIFIED — `metrics_extraction.py:57-60` calls `paper_gate.compare_metric`; the remaining `0.05` literals at `:186` (LIMN) and `:275` (CSMCIR) are declared self-consistency checks — NONE — none.
15. Gate can pass/fail for this model — VERIFIED — observed `PASS_WITHIN_5_PERCENT` (9 comparisons) and `ALERT_METRIC_DEVIATION` (`r10 differs by 34.8534%`) — NONE — none.
16. Registered checkpoint is identified and byte-identifiable — MISMATCH — `models.yaml:578-585`; observed `exists: False`; `artifacts/checkpoints` holds only `.gitkeep` — P1 — obtain the archive, record member names/sizes/hashes.
17. Checkpoint mapping to a category/benchmark condition is verified — UNKNOWN — `models.yaml:586-587`; observed `select_checkpoints` BLOCKED under both policies — P1 — keep blocked; do not invent a mapping.
18. Author archive identity verified (members, hashes) — NOT_RUN — observed only the Drive filename and an 18,022-byte `filesusr.com` mirror (`HTTP/2 200`, `content-type: application/x-zip-compressed`) — P1 — inventory before any run.
19. Checkpoint serialization form declared — VERIFIED — `models.yaml:537` `whole_model`; `train.py:274-275` — NONE — none.
20. Per-category checkpoint policy declared and matching source — VERIFIED — `models.yaml:531-537`; `train.py:70-96,298-300`; `README.md:31-44` — NONE — none.
21. A runnable checkpoint selection exists — NOT_RUN — observed both policies `BLOCKED` — P1 — requires a verified mapping plus a recorded training condition.
22. Declared native protocol is the cohort the source computes — VERIFIED — `models.yaml:555` vs `datasets.py:242-257` + `test.py:131-138` — NONE — none.
23. Protocol table entry matches the declared protocol — VERIFIED — `src/evaluation/fashioniq.py:178-180` — NONE — none.
24. `literature_split_label` agrees — VERIFIED — `models.yaml:554`; `train.py:89` loads `split='val'` — NONE — none.
25. `supported_protocols` includes `native_protocol` — VERIFIED — `models.yaml:555-556`; `registry.py:46-47` — NONE — none.
26. Gallery construction rule matches `fashioniq_val_split` semantics — VERIFIED — `datasets.py:243-249` vs `src/datasets/fashioniq.py:195-207` — NONE — none.
27. Gallery procedure is identical, not merely equivalent — VARIANT — int-index dedup (`datasets.py:222,224,239,246-249`) vs id-string dedup (`src/datasets/fashioniq.py:195-207`) — P3 — do not assert numeric identity from this alone.
28. Gallery size re-derived from the actual registry inputs — VERIFIED — dress 2628 / shirt 3089 / toptee 2902 vs full split 3817 / 6346 / 5373, 0 exceptions — NONE — none.
29. Reference exclusion implemented — VERIFIED — `test.py:134-135` — NONE — none.
30. Reference-exclusion procedure is identical — VARIANT — mask-then-`argsort` (`test.py:135,138`) vs sort-then-remove (`src/evaluation/fashioniq.py:70-75`) — P3 — documented.
31. Tie-breaking rule matches — MISMATCH — `np.argsort(-sims[i,:])` (`test.py:138`) vs `torch.argsort(scores[row], descending=True)` with no `stable` (`src/evaluation/fashioniq.py:71`) while `fashioniq_encoder.py:44` passes `stable=True`; `fashioniq.py:144-147` calls ties unpinnable — P3.
32. Recall@K keys compared — VERIFIED — `test.py:142`; `paper_gate.py:29` — NONE — none.
33. Recall denominator matches — VERIFIED — `test.py:147` vs `fashioniq.py:33` — NONE — none.
34. Source output shape compatible with a workbench aggregate — VARIANT — `test.py:148` per category; workbench `mean` labelled `WORKBENCH_DERIVED` (`papers.yaml:556-566`) — P3.
35. Evaluation-only CLI exists upstream — NOT_RUN (confirmed absent) — AST: `funcs: ['test']`, `__main__ in src: False`, `argparse in src: False` — P1 — never patch the pin.
36. Documented upstream test command runnable — NOT_RUN — `README.md:67-71` vs `test.py` — P1.
37. Adapter builds a command — NOT_RUN — `adapters/models.py:74-76`, base raises `:24-25`; observed `command=not audited`, `guarded_plan` → `SKIPPED` — P1.
38. Pinned trainer is a usable replay path — NOT_RUN — `train.py:279-306`; `--restore_file` declared `:62-64`, passed `:306`, never read at `:229`; `utils.py:136-140` unused — P1.
39. Environment stack matches the README — VERIFIED — `models.yaml:565-566` vs `README.md:18-19` — NONE — none.
40. Environment confidence label justified — VERIFIED — `models.yaml:564` `VERIFIED_README` — NONE — none.
41. torchvision version declared — MISMATCH — `models.yaml:567` `UNKNOWN` while `resnet.py:3` imports `torchvision.models.utils` — P2.
42. CUDA requirement declared and gated — VERIFIED — `models.yaml:572`; unguarded `.cuda()` at `test.py:28,47,99,115`, `text_model.py:96,108`, `model.py:178,249` — NONE — none.
43. Evaluator can run on a CPU-only host — VERIFIED (it cannot) — observed host `torch: null` — P2.
44. Dataset layout matches what the evaluator opens — VERIFIED — `fashioniq_layout.py:44-52` vs `datasets.py:157-159,168,170,195` — NONE — none.
45. Image extension matches the evaluator — VERIFIED — `models.yaml:550`; `datasets.py:195`; canonical release is `.png` (74381 observed) — NONE — none.
46. Path-separator requirement respected — CONCERN — `datasets.py:157-159`; source default `train.py:32` ends in `/` — P3.
47. Preparation contract registered for the model — VERIFIED — `models.yaml:552-553`; `preparation_contracts.yaml:3`; observed `clvc_net_fashioniq` — NONE — none.
48. Declared resize artifact reproducible from the pinned repository — PARTIAL — no resize script in the tree; recipe exists (`prepare_model.py:451-593`) — P2.
49. Resize recipe matches the linked tool — VARIANT — fetched tool: flat, `Image.ANTIALIAS`, format preserved, 256; contract/impl: per-category, RGB, `Image.LANCZOS`, JPEG (`preparation_contracts.yaml:26-42`; `prepare_model.py:585`) — P2.
50. Resize receipt records the resampling kernel — VERIFIED — `prepare_model.py:92`, `:141-144` `"resampling": RESAMPLING` — NONE — none.
51. External ResNet-50 weight declared with a checksum constraint — VERIFIED — `auxiliary_assets.yaml:148-161`; `preparation_contracts.yaml:44-56`; `resnet.py:12` — NONE — none.
52. External asset actually consumed by the pinned evaluator — VERIFIED — `model.py:72` → `resnet.py:218-223`; `model.py:123,195` — NONE — none.
53. Declared cache location matches the consumer — MISMATCH — `registry.py:208-216` → `~/.cache/torch/checkpoints/…`; `auxiliary_assets.yaml:156` + consumer → `$TORCH_HOME/hub/checkpoints/…`; both observed in one blocker list — P2.
54. `simplevocab.json` fallback correctly classified — VERIFIED — `text_model.py:66` only when `texts_to_build_vocab is None`, which never happens here; `cluster_b4_classic.yaml:266-269` — NONE — none.
55. Text preprocessing parity with the workbench generic composer — UNKNOWN / not on the path — `datasets.py:174`; `text_model.py:23-45`; `PHASE2_SOURCES_B.md:12,26` — P2.
56. Vocabulary construction parity possible — UNKNOWN — vocab built from training captions (`train.py:300`) and baked into the module — P1.
57. Evaluation-noise contract clean — VERIFIED — `train.py:90-96`; `test.py:81-82`; `models.yaml:583`; `registry.py:84-85` — NONE — none.
58. Training-noise provenance of the checkpoint established — UNKNOWN — `models.yaml:582` null; the paper states no training condition — P2.
59. Local staging complete — NOT_RUN — observed blockers: `resized_image/{dress,shirt,toptee}`, image-directory check, ResNet-50 (two paths), `WORKBENCH_CLVC_PYTHON` — P1.
60. Aggregate metric extraction implemented for this model — NOT_RUN — `metrics_extraction.py:333-341`; observed `AGGREGATE_PARSER_UNAVAILABLE / PARSER_UNAVAILABLE / None` — P1 — must emit per-category values to satisfy `paper_gate.py:348-363`.
61. Run records carry gate + variant provenance — VERIFIED (wiring present, no run to exercise) — `evaluate_models.py:124-175,243-273`; `autopilot.py:846-848`; `main.py:116-117` — NONE — none.
62. Run-record paper fields agree with the committed reference — MISMATCH — `models.yaml:557-559` all null vs `papers.yaml:552-555`; `autopilot.py:620-622,772-775`; `evaluate_models.py:309`; `_check_parity` → `PARITY_NOT_EVALUATED` — P2.
63. Backbone string in the committed reference is source-backed — MISMATCH — `papers.yaml:537-538` "CLIP-based textual branch"; `grep -ic clip paper.txt` → 0; source is two LSTMs — P2.
64. Registry-backed HTTP surfaces can serve a gate verdict for this model — VERIFIED (code) — `main.py:122-132` `/api/benchmark-runs`, `/api/runs/gate` — NONE — none.
65. Readiness classification not mislabelled by an unverified blocker — VERIFIED — observed `pc_readiness.py --model clvc_net --json` → `BLOCKED_IMPLEMENTATION` with `command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT`, a genuine implementation blocker — NONE — none.
66. Documented blockers are runnable-source-backed — VERIFIED — `models.yaml:561` → `preflight.py:206`; `OPEN_BLOCKERS.md:163` still says `COMMAND_UNAUDITED` — P3.
67. Stale `category_model_policy: shared` claims — MISMATCH — `SOURCE_TO_WORKBENCH_PARITY_MATRIX.md:25,58`, `audits/PHASE2_SOURCES_B.md:16,20,27` vs `models.yaml:531` — P3.
68. Category-model routing code exists — NOT_RUN — `workbench/replay/` lacks `clvc.py`, unlike LIMN — P3.
69. `prepare_model.py` docstring cites the right reader — MISMATCH (cosmetic) — `:460` cites `LIMN/datasets.py:112`; for this layout it is `CLVCNet/datasets.py:195` — P3.
70. Dataset-stage validation covers this layout — NOT_RUN — `prepare_dataset.py:17-18` `standard_paths` only — P3.
71. Committed reference row carries its own paper URL and PDF digest — MISMATCH — `build_paper_registry.py:27-34`; observed row keys contain neither — P3.
72. `prepare_model.py` resize safely refuses escaping/canonical writes — VERIFIED (code) — `:525-531`, `:557-565`, `:546-550,566-581` — NONE — none.
73. UNKNOWN was never promoted to PASS — VERIFIED — items 17, 18, 21, 35-38, 55, 56, 58-60, 68, 70 remain UNKNOWN/NOT_RUN — NONE — none.
74. Prior audit claims re-derived, not paraphrased — VERIFIED with two corrections — the prior review's "no paper metric value is stated anywhere in this document because none was retrievable" is superseded (paper obtained; row re-derived; pdf sha256 reproduced), and its "the gate can never pass, fail, or compare for this model as registered" is now **false** (observed `PASS_WITHIN_5_PERCENT`/`ALERT_METRIC_DEVIATION`); its other 54 checklist items were re-checked and no other claim was refuted — NONE — update the prior review.
75. No fabricated hash, URL, line number, or metric — VERIFIED — every hash/URL/line above came from an observed tool result or a pinned-file read; the only `[INFERENCE]` is explicitly marked — NONE — none.
76. No runtime claim made without observation — VERIFIED — zero RUNTIME_VERIFIED claims — NONE — none.

---

**FINAL VERDICT: BLOCKED — the pin, paper identity, all published numbers (Table 1 p.1375 `29.85 56.47 28.75 54.76 33.50 64.00 30.70 58.41`, PDF sha256 `4c452584…f88` reproduced), cohort (2628/3089/2902 re-derived), gallery rule, metric formula and preparation contract all check out, and the gate now genuinely compares this model, but nothing on the registered path can execute: the pinned evaluator has no CLI or loader, the adapter deliberately raises, the checkpoint is a placeholder with an uninspected author archive, no output parser exists for this model (so even a perfect run yields `FAILED_OUTPUT_VALIDATION`), the ResNet-50 asset is absent and resolved to two contradictory paths, and the resized-image tree does not exist; nothing was reproduced and nothing was run.
