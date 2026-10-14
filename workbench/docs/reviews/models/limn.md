# limn

MODEL ID: `limn` (LIMN, base iteration 0; LIMN+ is explicitly out of scope)

PAPER TITLE: *Self-Training Boosted Multi-Factor Matching Network for Composed Image Retrieval*, IEEE TPAMI 46(5):3665-3678, 2024 — `workbench/registry/models.yaml:840`, DOI `https://doi.org/10.1109/TPAMI.2023.3346434` (`models.yaml:841`). The author project page independently names LIMN (what is evaluated here) and LIMN+ (iterative self-training) as two different systems, so "LIMN" and "LIMN+" numbers are not interchangeable (source: `https://anosite.wixsite.com/limn`, fetched; abstract text distinguishes LIMN from "LIMN+").

PINNED SOURCE COMMIT: `7d7bc9b116f594a65ac22457491edf28a88d3c3e` (`models.yaml:844`).

```
$ git -C workbench/third_party/LIMN rev-parse HEAD
7d7bc9b116f594a65ac22457491edf28a88d3c3e
$ git -C workbench/third_party/LIMN log --oneline -1
7d7bc9b Update README.md
```
Commit pin: MATCH. The checkout is **not clean** in the strict sense the replay wrapper uses:

```
$ git -C workbench/third_party/LIMN status --porcelain
?? LIMN+/__pycache__/
?? LIMN/__pycache__/
$ python -c "... workbench.replay.limn.validate_source(Path('workbench/third_party/LIMN'))"
['LIMN source pin invalid: source checkout is dirty']
$ python -c "... workbench.backend.runtime.source_clean_and_pinned(model_by_id('limn'), p)"
(True, '7d7bc9b116f594a65ac22457491edf28a88d3c3e', None)
```
Only bytecode caches are untracked (`stat` mtimes 2026-10-10 16:39, i.e. produced by a local import, not by the pin). Two workbench code paths disagree about whether that is dirty: `workbench/replay/common.py:61-66` uses raw `git status --porcelain`; `workbench/backend/runtime.py:1998-2010` deliberately excludes `__pycache__`/`*.pyc`. The replay path is the stricter one and would refuse to run today. **P2 finding.**

CHECKPOINT IDENTITY: three category-specific whole saved models, registry metadata matched against author Hub LFS metadata (STATIC_AUDIT, network fetch, no bytes downloaded):

| checkpoint_id | file | size | author LFS sha256 (registry `models.yaml:875-876,884-885,893-894`) | Hub LFS metadata (fetched) |
|---|---|---|---|---|
| `base_iter0_dress` | `0_dress_best_model.pt` | 1792010157 | `96a335d4eb8b1e77d35575aab9177637208e2eb6c59e12f833f67e5cb0e7fc1c` | same oid, same size |
| `base_iter0_shirt` | `0_shirt_best_model.pt` | 1792010157 | `9260c7f77d3017d8cc10e6652cc8cfe275dbc9704366fa2bf581a24808930d74` | same oid, same size |
| `base_iter0_toptee` | `0_toptee_best_model.pt` | 1792010157 | `79f663a14d89d455fdc3612b0b36d011c06d7957d1475a654755e1053a6cece2` | same oid, same size |

Evidence: `curl` of `https://huggingface.co/api/models/iLearn-Lab/TPAMI24-LIMN/tree/30560ad575a56c39cc39047d1a474269338c77c0?recursive=true` returned `lfs.oid` equal to the three registered hashes, and the model API returned `sha: 30560ad575a56c39cc39047d1a474269338c77c0` (the revision all three download URLs pin, `models.yaml:872,881,890`). The three files are whole pickles, not state dicts: `LIMN/train.py:263` is `torch.save(model, ...)`, and the wrapper asserts the unpickled object is `model.LIMN` (`workbench/replay/limn.py:129-130`). Registry declares `checkpoint_serialization: whole_model` (`models.yaml:834`) and the bundle `base_iter0_all_categories` with all three required (`models.yaml:897-902`). Training-condition noise is `null` for all three (`models.yaml:873,882,891`), so `strict_clean_checkpoint_available: null` (`models.yaml:867`) is correct: these are NOT verified clean checkpoints. **STATIC_AUDIT**; no bytes were downloaded or hashed locally.

EVALUATION PROTOCOL: registry `native_protocol: fashioniq_val_split` (`models.yaml:852`), defined at `models.yaml:16-19` as "Val Split (Pair-Union Gallery, Reference Excluded)". Checked against the pinned evaluator:

- Gallery construction: `LIMN/datasets.py:158-169` builds `test_targets_id` as a first-seen ordered union of each validation annotation's `source_img_id` then `target_img_id`. This is a pair union, and it is the **endpoint subset** of `image_splits/split.{cat}.val.json`, not the whole split file. Cross-checked against the local dataset: dress val split 3817 ids, endpoints 2628; shirt 6346/3089; toptee 5373/2902; all endpoints ⊆ val split, and NOT ⊆ train split (command: `python3` over `data/fashionIQ_dataset`).
- Reference exclusion: `LIMN/test.py:80-85` writes `-10e10` into the reference's column and then `argsort`s; this is score masking, not physical gallery removal, exactly as the registry protocol family and `workbench/docs/FASHIONIQ_PROTOCOL_PARITY.md:5-9,45-48` describe.
- Recall: `LIMN/test.py:88-97` computes per-category R@1/R@10/R@50 as a percentage of the category's query count over the top-50 ranking. No source-level cross-category macro is emitted anywhere in the pinned evaluator.

Protocol claim: SUPPORTED by source. Note `literature_split_label: val` (`models.yaml:851`); TPAMI-era FashionIQ evaluations usually publish the *original* split, but I did not obtain the paper's own evaluation-code statement, so I record the label as unverified against the paper (see UNRESOLVED EVIDENCE).

DATASET CONTRACT: `fashioniq_layout: fashioniq_ilearn_resized_training` (`models.yaml:847`), `preparation_contract: manual_ilearn_resized`, `preparation_id: limn_fashioniq` (`models.yaml:849-850`), contract body `workbench/registry/preparation_contracts.yaml:249-299`. Every declared path traces to a consumer:

| Declared | Consumed at |
|---|---|
| `captions/cap.{cat}.train.json` | `LIMN/datasets.py:68` (`train_init_process`) |
| `captions/cap.{cat}.val.json` | `LIMN/datasets.py:139` (`get_test_data`) |
| `image_splits/split.{cat}.val.json` | `LIMN/datasets.py:137` |
| `captions/correction_dict_{cat}.json` | `LIMN/datasets.py:70` (train) and `:141` (test) |
| `resized_image/{cat}/` | `LIMN/datasets.py:44` (`image_dir = path + 'resized_image'`) and `:112` |
| source-shipped `correct_files/correction_dict_{cat}.json` | pinned checkout, README `:52` |
| resize recipe via `start_kit/resize_images.py` | README `:51` |

Three category-specific whole-model files, resized images, correction dictionaries and base iteration 0 are therefore each grounded in source (see checklist 9-13). SAFETY: `LIMN/datasets.py:42-44` concatenates `path + 'resized_image'`, so the dataset root must end in `/`; the wrapper papers over this with `path=f"{cache_root}/"` (`limn.py:139`), which is correct because `tempfile.TemporaryDirectory` yields an absolute path with no trailing separator.

One contract-level defect: `preparation_contracts.yaml:294` cites `source_evidence: the model README documents this script and this output path`. The README (line 51) documents a **flat** `resized_image/` directory and a **256x256** output, while the contract asks for `layout: per_category` — a layout the README does not describe. The generator reconciles this by transposing the documented transform into the layout the evaluator reads (`workbench/scripts/prepare_model.py:360-367` comment; `:397-402`). The transform itself (square 256x256) does match the documented script's `resize_image(img, [size, size])`. Severity P3: honest and re-derivable, but the provenance sentence overstates what the README says.

PREPROCESSING CONTRACT (image): the wrapper rebuilds the exact factory-call semantics rather than inventing a transform — `limn.py:132-136` calls `open_clip.image_transform(visual.image_size, is_train=..., mean=visual.image_mean, std=visual.image_std)`, mirroring `open_clip v2.20.0 factory.create_model_and_transforms` (fetched: transform derived from `model.visual.image_size`, `image_mean`, `image_std`). `ViT-L-14.json` in v2.20.0 pins `image_size: 224`. `transform.py` (fetched) eval path = `Resize(224, BICUBIC)` + `CenterCrop(224)` + RGB + `ToTensor` + OpenAI mean/std; train path = `RandomResizedCrop(224, scale=(0.9,1.0), BICUBIC)`. Dataset stage assignment matches native: query image uses `self.transform[1]` = val (via `get_test_img`, `datasets.py:125`) and gallery likewise, while `self.transform[0]` (train) is only used by `get_img` for training triples (`datasets.py:117`). Preprocessing: MATCH (STATIC_AUDIT).

PREPROCESSING CONTRACT (text): `datasets.py:82-91` lowercases, maps every `string.punctuation` char to a space, strips/splits, applies the loaded correction dictionary by exact token lookup, joins with one space, and `concat_text` joins caption[0] + literal `" and "` + caption[1]. The wrapper never touches the caption: it constructs the native `FashionIQ` dataset (`limn.py:138-141`), so normalization, ordering and the 77-token OpenCLIP tokenizer (`LIMN/model.py:71`) are the source's own. Text: MATCH (STATIC_AUDIT).

AUGMENTATION CONTRACT: no training augmentation is used at evaluation. The only augmentation in the native path is the OpenCLIP `preprocess_train` `RandomResizedCrop`, and the pinned `test.py` never selects it for validation images (`datasets.py:120-127` uses index 1). The wrapper constructs both transforms for signature/source fidelity (`limn.py:133-136`) and passes `[preprocess_train, preprocess_val]` (`limn.py:140`), matching `LIMN/train.py:81` exactly. No random crop reaches an evaluated image: MATCH (STATIC_AUDIT).

GALLERY CONTRACT: gallery = first-seen ordered union of (reference, target) endpoints, one vector per endpoint id, built by `datasets.py:158-169` and consumed by `test.py:74-85`. The wrapper supplies the same dataset object, so gallery construction is native. The only wrapper-side influence is `isolated_cache_root` (`common.py:77-87`), which symlinks `captions/`, `image_splits/`, `resized_image/` into a fresh temp root so the native `{cat}_train_data.json` / `test_queries_{cat}.pkl` / `test_targets_{cat}.pkl` caches (`datasets.py:49-64`) are regenerated rather than reused. That is a data-integrity improvement, and it is also a *contract difference*: the native code trusts a pre-existing cache, the replay never reuses one. Because the caches are derived deterministically from the same JSON, the resulting gallery should be identical; I cannot verify that claim at runtime. Gallery: MATCH (STATIC_AUDIT); cache-policy VARIANT (intentional).

METRIC CONTRACT: per-category R@1/R@10/R@50 from `test.py:88-97`, unchanged. The cross-category macro is a workbench aggregate (`common.py:90-100`), applied only when all three categories are present in canonical order (`limn.py:152-158`). The parser refuses an incomplete or internally inconsistent payload (`workbench/backend/metrics_extraction.py:121-186`) and recomputes the macro independently rather than trusting the declared value. Registry records no paper metric (`models.yaml:854-856` all `null`), so the parity check is structurally `METRIC_REFERENCE_UNVERIFIED` (`workbench/backend/paper_gate.py:177-179`). Metric contract: source-faithful; the macro is correctly not attributed to `test.py`.

SOURCE-TO-WORKBENCH PARITY: `replay_type: SOURCE_FAITHFUL_WORKBENCH_REPLAY` (`models.yaml:835`) is accurate for the inference path. `replay_category` (`limn.py:100-149`) imports the pinned `model`, `datasets`, `test` modules, verifies each resolved `__file__` is the pinned file (`limn.py:118-122`), hash-verifies and unpickles the whole model (`limn.py:128`, `common.py:36-48`), type-checks it (`limn.py:129-130`), moves it to CUDA (`limn.py:131`), constructs the native dataset, and calls `native_test.test(SimpleNamespace(local_rank=0, batch_size=32), model, dataset, category)` (`limn.py:142`). No part of the encoder, the composition, the normalization, the similarity or the recall computation is reimplemented. The argument shim diff versus `LIMN/train.py:231` is exactly: `local_rank=0` (the pinned parser defaults this to `os.getenv('LOCAL_RANK', -1)`, `train.py:33`) and `batch_size=32` (pinned default, `train.py:42`). Both are the author defaults, and both affect only batching, not the metric. Preflight before the call is strict (SHA-256 per category, dataset paths, exact torch/torchvision/open_clip versions, CUDA availability, source pin). **No reimplementation found; the extra-focus P1 concern does not hold.**

RUNTIME STATUS: `BLOCKED_RUNTIME_PREREQUISITES` (`models.yaml:857`), `command_status: REPLAY_COMMAND_IMPLEMENTED` (`models.yaml:858`). Independently reproduced, read-only, with the workbench venv:

```
$ .venv/bin/python -c "... workbench.replay.limn.validate_dataset(Path('data/fashionIQ_dataset'),('dress',))"
['LIMN dataset requirement missing: .../captions/correction_dict_dress.json',
 'LIMN dataset requirement missing: .../resized_image/dress']
$ .venv/bin/python -c "... validate_environment()"
['LIMN environment version mismatch: expected torch==1.12.1, found 2.14.0+cpu',
 'LIMN environment version mismatch: expected torchvision==0.13.1, found 0.29.0+cpu',
 'LIMN environment package missing: open_clip_torch==2.20.0']
$ echo $WORKBENCH_LIMN_PYTHON   # unset
$ .venv/bin/python -c "import open_clip"
ModuleNotFoundError: No module named 'open_clip'
```

`reports/` contains no LIMN run directory and `results/` does not exist, so no evaluation has ever produced metrics. Status: CONFIRMED BLOCKED. Every claim in this document is **STATIC_AUDIT**; nothing is RUNTIME_VERIFIED.

PAPER-METRIC STATUS: `reported_r10/r50/mean` are all `null` (`models.yaml:854-856`) — the registry deliberately carries no paper row. Workbench docs name no LIMN number either (grep over `workbench/docs/**.md` for `57.43`, `LIMN+` numbers, `50.71`, `55.91` returns only prose disclaimers, e.g. `LEGACY_SOURCE_REAUDIT`-style statements at `workbench/docs/LEGACY_SOURCE_AUDIT.md:74`). I could not fetch the paywalled TPAMI table, and web search surfaced no paper table reproducing LIMN/LIMN+ FashionIQ values. So: **no verified author row exists in the registry, none was required to, and the paper's own table remains UNKNOWN to me.**

However, a *verified author artifact row does exist* and nobody in the workbench has cited it: the author Hub revision ships the evaluation logs themselves, and fetching them with `curl`/`read` returned

```
dress_0_metrics_best.json  -> dress_r1 18.542389687654932, dress_r10 50.71888943976202,  dress_r50 74.5166088249876
toptee_0_metrics_best.json -> toptee_r1 26.721060683324833, toptee_r10 60.938296787353394, toptee_r50 81.84599694033656
shirt_0_metrics_best.json  -> shirt_r1 25.858684985279687, shirt_r10 56.084396467124634, shirt_r50 77.08537782139352
```

(dress_0 etag `0f49d0e86794576ef2a212d39313d50b2bffc2a7`, file size 208; shirt_0 etag `16b2979938b618b18c124e12ff34b7c08a7b6e7e`, 210; toptee_0 etag `5fd1316c78c5703599a0e96ed159165c33b50bf4`, 213 — all from the same tree API response as the checkpoints.) These are the `args.i == 0` rows written by the pinned training loop at `LIMN/train.py:249-260` for the default invocation, i.e. exactly the experiment registered here. Arithmetic mean over categories (computed by me): R@10 = 55.9139, R@50 = 77.8160, mean = 66.8649. **This is the closest thing to an author reference row that exists in the pinned environment, and the registry leaves the field null.** Severity P2 — an omitted, freely available author baseline for the exact declared experiment, which the parity gate would otherwise be able to use as a sanity reference (the gate itself, `paper_gate.py:177-179`, treats `null` as `METRIC_REFERENCE_UNVERIFIED`, so no false parity can be claimed today).

## CRITICAL FINDINGS

1. **P2 — replay preflight refuses the current checkout although the runtime path accepts it.** `workbench/replay/limn.py:53-56` → `common.py:51-69` treats any `git status --porcelain` output as dirty; `workbench/third_party/LIMN` carries two `__pycache__/` untracked entries (observed above), and `validate_source` returns `['LIMN source pin invalid: source checkout is dirty']`. The general runtime path (`workbench/backend/runtime.py:1998-2010`, tested at `workbench/tests/test_adversarial_environment.py:139`) deliberately exempts bytecode. Consequence: `workbench/replay/limn.py --preflight-only` fails for a reason unrelated to reproduction, on a checkout whose commit is exactly pinned. Severity P2 (blocks execution, causes no wrong science). Action: make the replay use `_source_dirty`-style classification, or state and test the intent explicitly.

2. **P2 — no error handling around the native evaluator call.** `workbench/replay/limn.py:137-142` calls `native_datasets.FashionIQ(...)` and `native_test.test(...)` with no guard. Because `isolated_cache_root` (`common.py:80`) uses a `TemporaryDirectory` as the dataset root but the native code writes caches into it (`datasets.py:49-64`), a missing image or a mid-run exception can surface as a traceback instead of a structured `[BLOCKED]` line. Severity P2 (diagnostics, not correctness). Action: wrap and map to the existing blocked path, quoting the native message.

3. **P2 — the resize generator assumes the canonical images are `.jpg`.** `workbench/scripts/prepare_model.py:442-450` tries `images/{id}.jpg` and only then `.jpg` again — there is no `.png` fallback — while the declared layout table (`workbench/backend/fashioniq_layout.py:80-82`) says the iLearn family *reads* `resized_image/<cat>/{id}.jpg` and the local canonical root actually ships 74381 `.png` files. Demonstrated read-only against a scratch root containing only `.png` canonicals:

```
$ python - <<'EOF' (scratch root, .png canonicals, _run_resize dry-run=False)
png-only root -> False resize input missing: /tmp/limnprep/images/B001.jpg (and 5 more)
EOF
```

Note this is also correct-and-conservative in a different sense: the code refuses to fabricate inputs it cannot find, which is the desired failure mode. Severity P2 (blocks any model in the iLearn family on a `.png`-canonical host). Action: fall back to the model's declared canonical extensions (`.png` is already modelled for CSMCIR at `models.yaml:50`).

## MINOR FINDINGS

4. **P3 — `fashioniq_image_extensions: ['.jpg']` is checked inside the derived `resized_image/` tree.** `models.yaml:848`, enforced at `workbench/backend/preflight.py:42-44` via `image_extension_problems(root, extension, layout)`, with `_LAYOUT_IMAGE_DIRECTORY['fashioniq_ilearn_resized_training'] = ('resized_image', True)` (`fashioniq_layout.py:83-89`). Verified semantics in scratch: with `resized_image/dress/a.png` present and `.jpg` requested, the reported problem is `2 of 2 dress validation images lack .jpg in .../resized_image/dress (first: a.jpg)`. That is the *correct* check for LIMN — it tests the evaluator's own hard-coded read path (`datasets.py:112`), which is what matters — but it makes the registry field name misleading, and it means the check can never be satisfied by the canonical `.png` distribution alone.

5. **P3 — misleading `source_evidence` for the resize artifact** (`preparation_contracts.yaml:294`): README line 51 documents a flat `resized_image/`, not `per_category`; the transform (256px square) does match `resize_images.py`'s `resize_image(img, [size, size])` default.

6. **P3 — `upstream_default_branch: null`** (`models.yaml:843`) while the registry uses a non-null default branch for other models (e.g. `models.yaml:41` `main` for CSMCIR). Cosmetic; the pin is a SHA, which is what reproducibility needs.

7. **P3 — byte-identical 5,376,030,471-byte checkpoints.** All three registered files report size 1792010157 with three distinct LFS oids (fetched metadata). Plausible for three separate `torch.save(model)` pickles of the same architecture, but their equality of size is not evidence of anything and should not be used as a proxy for identity. The registry does not use it that way (it uses SHA-256); recorded only to pre-empt the inference.

## UNRESOLVED EVIDENCE

- How the TPAMI paper builds the FashionIQ gallery (course vs fine split, reference included/excluded) and its exact reported numbers: UNKNOWN. The DOI resolves to a paywall; web search returned no paper table. Registry `literature_split_label: val` (`models.yaml:851`) is consistent with the pinned evaluator but I did not verify it against the paper or against cited reproductions.
- Whether the author Hub `*_0_metrics_best.json` rows in `UNRESOLVED EVIDENCE` correspond to the *published* LIMN (not LIMN+) configuration: the parameter block (`prefix_num 8, head 8, layer 2, norm 1.0, temperature 0.1`) matches README lines 81-83 for the base FashionIQ runs, and the `_2` rows carry `interval`/`p_origin`/`p_aug` (self-training), but I did not obtain the paper's own table to confirm which row it publishes.
- Whether the resized-image set at 256x256 LANCZOS is bit-identical to what the author used: unresolvable by construction — the author publishes no resized images and no checksum (the two `*_0_metrics_best.json` rows record metrics only). Marked UNKNOWN; not a blocker for the declared evaluator, which only reads whatever resized JPEGs are staged.
- Whether the three checkpoint SHA-256 values are the file bytes' hashes: they are the Hub's LFS oids, i.e. what the author client uploaded. I did not download a byte, so local delivery is UNVERIFIED.
- Any GPU behaviour: UNKNOWN — no CUDA host, CPU-only torch 2.14.0+cpu in `.venv`.

## FINAL VERDICT

BLOCKED — the reproduction path is source-faithful and correctly refuses to run (three category models, resized images, correction dictionaries, CUDA env all absent locally, plus a preflight that rejects the current checkout over untracked `__pycache__`), and no evaluation or paper comparison has ever been executed.

# EXECUTION TRACE

`paper/README/config` → `source dataset` → `preprocessing` → `official evaluator` → `workbench invocation` → `checkpoint` → `output` → `metric comparison`.

1. **paper/README/config → declared stack.** README:20-24 states GPU 1xA100-40G, CUDA 12.4, `torch == 1.12.1`, `torchvision == 0.13.1`, `open-clip-torch == 2.20.0`. `requirements.txt:68,70,37` repeat `torch==1.12.1`, `torchvision==0.13.1`, `open-clip-torch==2.20.0`; the author project page (fetched) lists the same four plus CUDA 12.4. Registry `models.yaml:859-866` records exactly `pytorch 1.12.1`, `torchvision 0.13.1`, `cuda 12.4`, `packages: [open_clip_torch==2.20.0]`, `python: UNKNOWN`, `confidence: VERIFIED_README`. **MATCH**, including the honest `UNKNOWN` for Python (the README never states one). Note `requirements.txt:8` also pins `clip==1.0`, which is not read by the LIMN evaluator path (it uses `open_clip`) and is not, and need not be, registered.

2. **source dataset.** Registry layout `fashioniq_ilearn_resized_training` (`models.yaml:847`) → `workbench/backend/fashioniq_layout.py:37-41` (`ilearn_resized_training_paths`) → every path traced to a native read site: `captions/cap.{cat}.train.json` (`datasets.py:68`), `cap.{cat}.val.json` (`:139`), `correction_dict_{cat}.json` (`:70,:141`), `image_splits/split.{cat}.val.json` (`:137`), `resized_image/{cat}/` (`:44,:112`). **MATCH**; also verified the registry's path set is a superset of what one category needs (harmless — the wrapper validates only the selected categories, `limn.py:59-73`).

3. **preprocessing.** Wrapper `limn.py:132-136` reconstructs the transform from the *loaded* model's `visual` attributes, which is what `open_clip.factory.create_model_and_transforms` does (fetched v2.20.0); `ViT-L-14.json` (fetched) pins `image_size: 224`; `transform.py` (fetched) eval = `Resize(224,BICUBIC)+CenterCrop(224)+RGB+ToTensor+Normalize`, train = `RandomResizedCrop`. Native stage selection: validation images use index 1 only (`datasets.py:120-127`); query and gallery both go through `get_test_img`. Text path untouched by the wrapper and handled entirely by `datasets.py:82-91` plus the native tokenizer (`model.py:71`). **MATCH** (STATIC_AUDIT).

4. **official evaluator.** `LIMN/test.py:23` `test(params, model, testset, category)`; the only in-tree caller is `LIMN/train.py:231`. There is no `__main__` block in `test.py` and no standalone eval CLI anywhere in the pinned tree — `grep` over `LIMN/*.py` finds no evaluation entry point. So the workbench must supply a harness; that is a fact about the author's repository, not a workbench shortcut.

5. **workbench invocation.** `LIMNAdapter.command` (`workbench/backend/adapters/models.py:123-137`) launches `workbench/replay/limn.py` with `--checkpoint-root/--dataset-root/--source-root`, adding `--category <cat>` unless the unit is the whole bundle. The replay: source pin check (`limn.py:53-56`), per-category checkpoint SHA-256 and mapping check (`limn.py:32-51`), dataset check (`:59-73`), environment check (`:76-91`), then `importlib` of the three pinned modules with `__file__` identity enforcement (`:113-122`), untrusted-pickle refusal via hash (`:128`), `torch.load(..., weights_only=False)` only after the hash matched (`common.py:36-48`), `model.cuda().eval()` (`:131`), native dataset (`:138-141`), and `native_test.test(SimpleNamespace(local_rank=0, batch_size=32), model, dataset, category)` (`:142`). Diff versus `train.py:231`: both supplied values are the pinned parser defaults (`train.py:33,42`). **NO REIMPLEMENTATION OF THE PIPELINE** — the extra-focus P1 concern is not supported by the code.

6. **checkpoint.** Three download URLs pin revision `30560ad575a56c39cc39047d1a474269338c77c0` (`models.yaml:872,881,890`); fetched Hub tree metadata returns LFS `oid`s equal to `models.yaml:876,885,894` with equal sizes `1792010157` (= `models.yaml:875,884,893`); fetched model API confirms the revision string. Native save site `train.py:263` (`torch.save(model, "{i}_{category}_best_model.pt")`) binds iteration 0 to those filenames. **MATCH at metadata level**; bytes UNVERIFIED locally.

7. **output.** `limn.py:143-149` constructs `ReplayResult(model_id, checkpoint_id, category, protocol)` from `metrics[f"{category}_r{n}"]`, which is the key format produced by `test.py:95`. Whole-run JSON at `limn.py:190-200` carries model, protocol, per-category rows, `complete_benchmark` from `macro_aggregate` (`common.py:90-100`, applied only to the full three-category set, `limn.py:157-158`). Output path is create-only (`limn.py:203-207`). Consumer `parse_limn_output` (`metrics_extraction.py:57-186`) re-derives the macro from the category rows and marks the extraction UNPARSED on any duplicate category, missing category, out-of-range recall or declared/computed disagreement. **MATCH**, plus independent recomputation on the read side.

8. **metric comparison.** `evaluate_gate` (`paper_gate.py:164-215`) returns `METRIC_REFERENCE_UNVERIFIED` when `reported_r10/r50/mean` are all `null` — which they are (`models.yaml:854-856`) — and `NOT_RUN` when no real execution exists. Both hold. **No comparison has ever been made, and none can be made from the registry as declared.**

# CHECKLIST

1. Registered model exists and is uniquely identified — **VERIFIED** — `models.yaml:831` (`- model_id: limn`), unique among 16 entries (`python -c` over `yaml.safe_load`). — NONE — none.
2. Paper identity is a real, resolvable publication — **VERIFIED** — `models.yaml:840-841`; author project page (fetched) and `README.md:157-170` carry the same citation (TPAMI 46(5):3665-3678, 2024). — NONE — none.
3. `upstream_repo_url` resolves to the pinned checkout — **VERIFIED** — `models.yaml:842` `https://github.com/iLearn-Lab/TPAMI24-LIMN`; `git -C workbench/third_party/LIMN rev-parse HEAD` = the registry SHA; `git log -1` shows the repo's own history. — NONE — none.
4. `upstream_commit_sha` equals the local checkout HEAD — **VERIFIED** — `7d7bc9b116f594a65ac22457491edf28a88d3c3e` both at `models.yaml:844` and from `git rev-parse HEAD`. — NONE — none.
5. Checkout is clean at the pin — **VARIANT** — `git status --porcelain` = `?? LIMN+/__pycache__/`, `?? LIMN/__pycache__/`; `replay.limn.validate_source` → `['LIMN source pin invalid: source checkout is dirty']`; `runtime.source_clean_and_pinned` → `(True, '<sha>', None)`. Tracked tree unmodified; only local bytecode. — P2 — unify the dirty predicate (prefer `runtime._source_dirty`).
6. Registry and pinned source agree on the source directory name — **VERIFIED** — `models.yaml:846` `source_dir: LIMN`; the directory contains `LIMN/`, `LIMN+/`, `correct_files/`, `README.md`, `requirements.txt`; the wrapper resolves `source_root / "LIMN"` (`limn.py:113`), matching `README.md:32`. — NONE — none.
7. Registry declares the correct number and identity of checkpoints — **VERIFIED** — three variants, `models.yaml:870-896`, matching Hub siblings `0_{dress,shirt,toptee}_best_model.pt` (fetched tree API). — NONE — none.
8. Checkpoint filenames match the pinned save site — **VERIFIED** — `LIMN/train.py:263` saves `"{}_{}_best_model.pt".format(args.i, _name[1])` with `args.i` default `0` (`train.py:72`); LIMN+ uses `augment_iteration` instead (`LIMN+/train.py:44,276`), so iteration 0 is a LIMN-only concept. — NONE — none.
9. Claim "three category-specific whole-model files" — **VERIFIED** — `train.py:263` `torch.save(model, ...)` (whole object); `models.yaml:834` `checkpoint_serialization: whole_model`, `:900` `structure: category_specific_whole_models`; wrapper type-check `limn.py:129-130`. — NONE — none.
10. Claim "resized images required" — **VERIFIED** — `LIMN/datasets.py:44` `self.image_dir = self.path + 'resized_image'`; `:112,:121` open `resized_image/<cat>/<id>.jpg`; README:51 documents the generator; the wrapper requires the directory (`limn.py:67`). — NONE — none.
11. Claim "correction dictionaries required" — **VERIFIED** — consumed at `datasets.py:70` (train init) and `:141` (test data); shipped in the pinned checkout as `correct_files/correction_dict_{dress,shirt,toptee}.json` (files inspected: 363/360/391 entries); README:52; two-step preparation path (stage then read) both implemented. — NONE — none.
12. Claim "base iteration 0 required" — **VERIFIED** — `train.py:72` default `--i = 0`, `:263` writes the `0_` prefix; README:75 names `checkpoint/0_{dataset}_best_model.pt`; registry records only iteration-0 files. — NONE — none.
13. Every declared dataset/staging path has a native consumer — **VERIFIED** — table in DATASET CONTRACT; each row cites a `datasets.py` line. — NONE — none.
14. Native protocol is `fashioniq_val_split` — **VERIFIED** — `datasets.py:158-169` pair union; `test.py:80-85` reference mask; `test.py:74` dot-product similarity; registry `models.yaml:852`. — NONE — none.
15. Gallery is the pair union (not the full split) — **VERIFIED** — `datasets.py:159-163`; measured against local data: endpoints 2628/3089/2902 vs val splits 3817/6346/5373. — NONE — none.
16. Reference exclusion is a mask, not removal — **VERIFIED** — `test.py:80-85` writes `-10e10` then `argsort`; the workbench docs say the same but the code is the evidence. — NONE — none.
17. Recall denominators are the query counts, per category — **VERIFIED** — `test.py:94` `r = 100 * r / len(nn_result)` where `nn_result` is one entry per `test_queries[i]`. — NONE — none.
18. Recall is computed at K = 1, 10, 50 — **VERIFIED** — `test.py:89` `for k in [1, 10, 50]`. — NONE — none.
19. No source-level cross-category macro is being passed off as the paper's — **VERIFIED** — `test.py` returns per-category triples only; macro lives in `common.py:90-100`; output labels it `macro_*`; the parser recomputes it (`metrics_extraction.py:157-186`). — NONE — none.
20. The paper's protocol label (`val` vs `original`) is verified — **UNKNOWN** — the pinned evaluator is val-split; the paper's own evaluation protocol could not be fetched (paywalled DOI; search surfaced no table). `models.yaml:851` asserts `val`. — P3 — obtain the paper text or a cited reproduction before publishing a paper-vs-run comparison.
21. Registry declares the correct protocol family for the current evaluator — **VERIFIED (STATIC_AUDIT)** — the pinned evaluator is what the registry describes; the workbench can also express the family generically (see 22). — NONE — none.
22. The workbench's generic FashionIQ gallery builder expresses this protocol — **VERIFIED** — `workbench/docs/FASHIONIQ_PROTOCOL_PARITY.md:5-9,45-50` describes the val-split family; note the dedicated replay intentionally does *not* call it (native path is used instead). — NONE — none.
23. The workbench documents that masking ≠ removal — **VERIFIED (documentation only)** — `FASHIONIQ_PROTOCOL_PARITY.md:45-48`; not exercised here. — NONE — none.
24. Declared image extension is the one the evaluator opens — **VERIFIED** — `models.yaml:848` `['.jpg']`; `datasets.py:112` builds `+ ".jpg"`; `fashioniq_layout.py:81-82` states the same. — NONE — none.
25. The extension check inspects the directory the evaluator actually reads — **VERIFIED** — `fashioniq_layout.py:83-89` maps `fashioniq_ilearn_resized_training` to `('resized_image', True)`; scratch test showed the message `... lack .jpg in .../resized_image/dress`. — P3 — rename/annotate, since the registry field reads as if it concerned canonical `images/`.
26. Resized-image recipe matches a documented author tool — **VERIFIED** — README:51 points at `start_kit/resize_images.py`; fetched script uses `image.resize([size,size])` with default `--image_size 256`; contract records `image_size: 256`, `source_script`, `source_script_url` (`preparation_contracts.yaml:282-287`). — NONE — none.
27. Resize output layout matches the evaluator read path — **VERIFIED** — contract `layout: per_category`, `extension: .jpg` (`preparation_contracts.yaml:288-289`); generator writes `resized_image/<cat>/<id>.jpg` (`prepare_model.py:430-432,478`); `datasets.py:112` reads that path. — NONE — none.
28. Resize recipe provenance sentence is accurate — **VARIANT** — `preparation_contracts.yaml:294` says the README documents "this output path"; README:51 documents a flat directory, not `per_category`. Transform and size do match. — P3 — adjust the wording.
29. Resize generator can consume the canonical image distribution — **VARIANT** — `prepare_model.py:442-450` tries `{id}{extension}` then only `.jpg`; local canonical root is 74381 `.png`; scratch run returned `resize input missing: /tmp/limnprep/images/B001.jpg (and 5 more)`. — P2 — add the declared-canonical-extension fallback; the current failure mode is a refusal, not a silent substitution.
30. Correction dictionaries are staged byte-identically from the pinned checkout — **VERIFIED** — `prepare_model.py:299-343` (copy from `<source>/correct_files/`, refuse a differing destination, report absent declared assets); `preparation_contracts.yaml:290-299` declares `generator: pinned_source_file`, `source_directory: correct_files`. — NONE — none.
31. Correction dictionaries are the source's own (not re-derived) — **VERIFIED** — files present in the pin (`correct_files/correction_dict_{dress,shirt,toptee}.json`), 363/360/391 entries; loader `datasets.py:70,141`. — NONE — none.
32. Image transform is derived from the model, not hard-coded — **VERIFIED** — `limn.py:132-136` uses `visual.image_size`, `visual.image_mean`, `visual.image_std`; matches `open_clip.factory.create_model_and_transforms` (fetched v2.20.0). — NONE — none.
33. Declared image size is the one open_clip v2.20.0 uses for ViT-L-14 — **VERIFIED** — fetched `model_configs/ViT-L-14.json` → `vision_cfg.image_size: 224`. — NONE — none.
34. Eval transform uses no random crop — **VERIFIED** — fetched `transform.py` eval branch is `Resize`+`CenterCrop`; `datasets.py:125` uses index 1 (val) for every validation image (query and gallery). — NONE — none.
35. Text normalization reproduces the source exactly — **VERIFIED** — `datasets.py:82-91`; wrapper never rewrites captions (`limn.py:138-141` builds the native dataset). — NONE — none.
36. Caption composition order and separator match — **VERIFIED** — `datasets.py:90` `"{} and {}"`. — NONE — none.
37. Tokenizer is the source's own — **VERIFIED** — `model.py:71` `open_clip.get_tokenizer('ViT-L-14')`, reached through the unpickled model; wrapper never tokenizes. — NONE — none.
38. Augmentation contract is stated and non-empty only where it should be — **VERIFIED** — training augmentation (`RandomResizedCrop`) exists in the source but is never applied to an evaluated image; documented as the augmentation contract above. — NONE — none.
39. An `--output` file is never overwritten — **VERIFIED** — `limn.py:203-207` opens with mode `"x"` and maps `FileExistsError` to `[BLOCKED]`. — NONE — none.
40. A partial-category run cannot emit a macro — **VERIFIED** — `common.py:92-95` requires exactly the three canonical categories in order; `limn.py:157-158` gates `macro_aggregate` on `categories == CATEGORIES`; `test_replay.py:33-39,90-95` assert the same. — NONE — none.
41. The metric parser independently recomputes the macro — **VERIFIED** — `metrics_extraction.py:157-186` recomputes and rejects when the declared and computed values differ by >0.05. — NONE — none.
42. The metric parser rejects out-of-range or non-finite recalls — **VERIFIED** — `metrics_extraction.py:41-48` `_is_valid_recall`. — NONE — none.
43. The paper gate refuses to compare when no reference metric exists — **VERIFIED** — `paper_gate.py:177-179` → `METRIC_REFERENCE_UNVERIFIED`; consistent with `models.yaml:854-856` all null. — NONE — none.
44. Registry carries a verified paper row — **VERIFIED NEGATIVE** — `models.yaml:854-856`; no workbench doc names a LIMN number (grep over `workbench/docs/**.md`). The registry is honest about not having one. — NONE — none.
45. An author reference row for the exact registered experiment exists and is used — **VARIANT** — the author Hub revision ships `dress_0_metrics_best.json`, `shirt_0_metrics_best.json`, `toptee_0_metrics_best.json` (fetched; R@10 50.71888943976202 / 56.084396467124634 / 60.938296787353394; R@50 74.5166088249876 / 77.08537782139352 / 81.84599694033656; macro R@10 55.9139, R@50 77.8160, mean 66.8649 computed here), written by `train.py:249-260` for the default invocation. The registry records nothing. — P2 — record it as an author-artifact sanity reference (labelled author-evaluator, not paper) so a future run has a real comparison target.
46. Declared stack matches README and requirements — **VERIFIED** — README:20-24 and `requirements.txt:37,68,70`; `models.yaml:863-866`. — NONE — none.
47. Python version claim is honest — **VERIFIED** — registry `python: UNKNOWN` (`models.yaml:862`); the README states no Python version (grep over `README.md`). — NONE — none.
48. CUDA requirement is real, not assumed — **VERIFIED** — `LIMN/test.py:39,41,57,60` call `.cuda()` unconditionally; `replay.limn.validate_environment` turns that into a CUDA-availability blocker (`limn.py:87-90`). — NONE — none.
49. Local environment satisfies the declared stack — **VERIFIED NEGATIVE** — `validate_environment()` → `torch 2.14.0+cpu` vs `1.12.1`, `torchvision 0.29.0+cpu` vs `0.13.1`, `open_clip_torch` missing; `WORKBENCH_LIMN_PYTHON` unset. — NONE (expected on this host) — none; this is the BLOCKED reason.
50. Preflight fails closed on the local dataset — **VERIFIED** — `prepare_model.py --model limn --json` reports 6 missing generated artifacts, `image directory missing: .../resized_image`, `execution environment not verified: WORKBENCH_LIMN_PYTHON=None`, and `fully_prepared: false`. — NONE — none.
51. The replay refuses an unpinned or dirty source — **VERIFIED (strict)** — `limn.py:53-56` → `common.py:51-69`; observed message `LIMN source pin invalid: source checkout is dirty`. See item 5 for the over-strictness. — NONE — none.
52. Whole-model checkpoints are only unpickled after hash verification — **VERIFIED** — `common.py:36-48`; called at `limn.py:128` with the registry `expected_sha256`. `weights_only=False` is reached only past the equality check. — NONE — none.
53. The replay cannot load a non-pinned `model`/`datasets`/`test` module — **VERIFIED** — `limn.py:118-122` compares each resolved `__file__` to the pinned path and raises `ImportError` otherwise; `sys.path.insert(0, source_path)` at `:114`. — NONE — none.
54. The replay reimplements any part of the official evaluator (the extra-focus P1 concern) — **VERIFIED NEGATIVE** — `limn.py:142` calls native `test.test`; encoder/composition/similarity/ranking/recall all come from `LIMN/model.py` and `LIMN/test.py`; the only wrapper-authored numbers are the macro (`common.py:90-100`) and the transform factory re-derivation, which mirrors `open_clip.factory`. No approximate substitute evaluator exists in the path. — NONE — none.
55. A reproduction result exists to audit — **NOT_RUN** — no run directory under `reports/`, no `results/` directory at all; `reproduction_status: BLOCKED_RUNTIME_PREREQUISITES` (`models.yaml:857`). — NONE — none.

FINAL VERDICT: BLOCKED
