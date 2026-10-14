# sprc

Independent, adversarial, READ-ONLY audit of the `sprc` registry entry against the pinned upstream checkout and the real paper. Nothing under `/home/heheboiz/data/cir` was written; all probe output went to `/tmp/audit-final/sprc/` (`sprc_iclr.pdf`, `sprc_iclr.txt`, `sprc_arxiv_v1.html`, `page-06.png`, `t3.py`). Every claim is `STATIC_AUDIT` unless marked OBSERVED (a command I ran and its output). **No `RUNTIME_VERIFIED` claim exists for this model.** One prior-round claim is REFUTED below (§SOURCE-TO-WORKBENCH PARITY, M1).

---

## MODEL ID

`sprc` / `method_name: SPRC`.

- Registry block `workbench/registry/models.yaml:830-884` (`model_id: sprc`).
- Adapter `workbench/backend/adapters/models.py:119-121` — `SPRCAdapter`, `script = "src/blip_validate.py"`, **no `command()` override**; inherits `OfficialScriptAdapter.command`, which raises (`:24-25`).
- OBSERVED `registry.model_by_id('sprc')` resolves; the entry has **no `source_blocker` key** (OBSERVED key list).

## PAPER TITLE

`Sentence-level Prompts Benefit Composed Image Retrieval` — matches.

- Registry: `paper_title` `models.yaml:844`, `paper_url: https://arxiv.org/abs/2310.05473` (`:845`), `publication_year: 2024` (`:843`).
- OBSERVED `read https://arxiv.org/abs/2310.05473` → same title, authors Yang Bai, Xinxing Xu, Yong Liu, Salman Khan, Fahad Khan, Wangmeng Zuo, Rick Siow Mong Goh, Chun-Mei Feng; v1 2023-10-09.
- OBSERVED camera-ready: `curl -sL https://proceedings.iclr.cc/paper_files/paper/2024/file/8204f54f6cd8f0db1765a274e94ceaac-Paper-Conference.pdf` → `200 5641958`; `sha256sum` → `2c250ddb9d0251e21fdad6a2a9400f1d0f688e8427be3f536e9a8404f5409a3b`; header on every page `Published as a conference paper at ICLR 2024` (`sprc_iclr.txt:441`). Pinned `README.md:1` says `【ICLR 2024, Spotlight】`. Title/venue match.

## PINNED SOURCE COMMIT

- Registry `upstream_commit_sha: 2935a5397732260d1db6fa577e5926f963e36f0f` (`models.yaml:848`), `upstream_repo_url: https://github.com/chunmeifeng/SPRC` (`:846`), `upstream_default_branch: null` (`:847`).
- OBSERVED `git -C workbench/third_party/SPRC rev-parse HEAD` → `2935a5397732260d1db6fa577e5926f963e36f0f`. **Exact match.**
- OBSERVED `git log -1` → `2935a5397732260d1db6fa577e5926f963e36f0f|2024-04-16 21:29:29 +0800|Update README.md`.
- OBSERVED `runtime.source_clean_and_pinned(model_by_id('sprc'), config.WORKBENCH_THIRD_PARTY_ROOT/'SPRC')` → `(True, '2935a5397732260d1db6fa577e5926f963e36f0f', None)`. Not dirty. (`git status` in the checkout shows only untracked `__pycache__` build products.)
- OBSERVED `du -sh 12M` checkout dir (8.5M excluding `.git`); tree is `README.md`, `requirements.txt`, `src/` (7 scripts + vendored `src/lavis`, 158 `.py`). `UPSTREAM_SOURCE_AUDIT.md:42` records `9.0M` — consistent given build products; not a defect.

## CHECKPOINT IDENTITY

Variant `sprc_fiq_unresolved` (`models.yaml:873-884`): `filename: sprc_fiq.pt`, `download_url: null`, `source_url: https://1drv.ms/u/s!Aj0q22vyiZbabnUya4mnufIBtYI?e=n4ZVKj`, `source_type: author_onedrive`, `checkpoint_training_noise_pct: null`, `evaluation_noise_pct: 0`, `expected_size_bytes: null`, `expected_sha256: null`, `status: CHECKPOINT_MAPPING_UNVERIFIED`, `checkpoint_mapping_status: UNVERIFIED`.

- OBSERVED pinned `README.md:124` publishes exactly that URL; `README.md:128` publishes `sprc_fiq_vitl.pt` on Google Drive; `README.md:130-131` Todo: “code and pre-trained weights for rerank model”.
- OBSERVED (network) `curl -sI` → `HTTP/2 301` → `curl -sIL` → `HTTP/2 403`. No `Content-Length`, no bytes, no digest. So `expected_sha256`/`expected_size_bytes` are null by necessity.
- OBSERVED `workbench/artifacts/checkpoints/` is **empty**; `find workbench/artifacts -iname "*sprc*"` → only `workbench/artifacts/results/fashioniq_original_split/mock/mock_sprc-sprc_fiq.json`. OBSERVED `checkpoint_path('sprc', ck, root)` → `/home/heheboiz/data/cir/workbench/artifacts/checkpoints/sprc/sprc_fiq.pt` (absent).

### The mapping question is now half-answered — the default class IS an author class, but it is NOT in the pinned tree

This is the most important correction to the prior round.

- OBSERVED `src/blip_validate.py:144` `default="blip2_cir_rerank_learn"`. OBSERVED repo-wide `grep` for that string → **one** hit (that line); no `@registry.register_model("blip2_cir_rerank_learn")` anywhere. `src/lavis/models/__init__.py:35-55` imports 19 blip2 classes; `blip2_qformer_cir_rerank_learn` is not among them. `src/lavis/models/blip2_models/__init__.py` is **0 bytes** (OBSERVED `wc -c` → `0`), so it imports nothing.
- BUT the module **is shipped as tracked bytecode**: OBSERVED `git ls-files` shows `src/lavis/models/blip2_models/__pycache__/blip2_qformer_cir_rerank_learn.cpython-39.pyc` is **tracked** (`git ls-files --error-unmatch` → rc 0). OBSERVED `strings -a` on it → `blip2_cir_rerank_learn` and `Blip2QformerCirRerankLearn.inference`. Same for `blip2_cir_rerank_un`, `blip2_cir_rerank_ps`, and plain `blip2_cir_rerank`. Ten `.pyc` files under `__pycache__` have **no `.py` counterpart** (`blip2_pformer`, `blip2_qformer_cir`, `blip2_qformer_cir_bl`, `blip2_qformer_cir_catv2`, `blip2_qformer_cir_rel_fimage`, `blip2_qformer_cir_rel_fimage_input`, `blip2_qformer_cir_rerank_learn`, `blip2_qformer_cir_rerank_ps`, `blip2_qformer_cir_rerank_un`, `blip2_qformer_cir_ti`).
- Consequence (STATIC_AUDIT, no source loaded): the key it is py39 bytecode, that it is not imported by `src/lavis/models/__init__.py:50`, so it is not in `registry.mapping["model_name_mapping"]`, and therefore `registry.get_model_class("blip2_cir_rerank_learn")` returns `None` (`src/lavis/common/registry.py:236-237`) and `load_model_and_preprocess` fails at `model_cls.from_pretrained` (`src/lavis/models/__init__.py:223-226`, OBSERVED text). **So the argparse default fails on the pinned tree, and the class that would satisfy it is not registered.** Verdict on the mapping: still **UNKNOWN**, but now with a much sharper shape — “the author's rerank-learn class exists only as bytecode and is not registered by the pinned `lavis.models` package”.

## EVALUATION PROTOCOL

Registry: `native_protocol: fashioniq_original_split` (`models.yaml:855`), `supported_protocols: [fashioniq_original_split]` (`:856`), `literature_split_label: original` (`:854`), `replay_status: BLOCKED_CHECKPOINT_MAPPING` (`:837`), `command_status: COMMAND_BLOCKED_MODEL_MAPPING` (`:862`).

Native chain, line by line:

- Entrypoint `src/blip_validate.py:141-155`: `--dataset` (`:143`), `--blip-model-name` default `blip2_cir_rerank_learn` (`:144`), `--backbone` default `pretrain` (`:145`), `--model-path` (`:146`); FashionIQ route `clip_finetune_fiq(['dress','toptee','shirt'], args.blip_model_name, args.backbone, args.model_path)` (`:155`).
- `clip_finetune_fiq` (`:26`) → `load_model_and_preprocess(name=blip_model_name, model_type=backbone, is_eval=False, device=device)` (`:45`) → `torch.load(checkpoint_path)` (`:48`) → `blip_model.load_state_dict(checkpoint[blip_model.__class__.__name__], strict=False)` (`:49`).
- Per category, order **dress → toptee → shirt**: `FashionIQDataset('val',[cat],'relative',preprocess)` (`:64`) and `('classic')` (`:66`); gallery `extract_index_blip_features(classic_val_dataset, blip_model)` (`:80`); `compute_fiq_val_metrics(...)` (`:81`, defined `src/validate_blip.py:24`).
- **Gallery** = complete ordered `image_splits/split.{cat}.val.json` (`src/data_utils.py:155`), scored in declared order (`src/utils.py:~54-77`, DataLoader has no `shuffle`). No union, no dedup, no sort.
- **Reference eligible**: `compute_fiq_val_metrics` (`src/validate_blip.py:44-55`) sorts all gallery names (`:45`), builds `labels` against `target_names` only (`:50`), asserts one hit per query (`:51`), `recall_at10 = labels[:,:10].sum()/len(labels)*100` (`:54`), `recall_at50` (`:55`). No masking, no reference removal.
- **The default model class does not use the reference as query context**: `blip2_qformer_cir_rerank.py:373-397` receives `reference_embeds` and uses it only for `.to(reference_embeds.device)` (`:381`), then scores `text_proj(text CLS)` against `target_feats` (`:395`). (The alternative candidate class `blip2_qformer_cir_align_prompt.py:313-359` **does** cross-attend the reference — so this property is class-dependent and the class is unresolved.)
- Workbench protocol table `src/evaluation/fashioniq.py:172-174`: `fashioniq_original_split` = `gallery="full_split"`, `reference_excluded=False`, alias `fashioniq_original`; `build_fashioniq_gallery` (`:204-227`), `build_original_gallery` (`:121-129`, asserts uniqueness, no sort).
- Paper: OBSERVED `sprc_iclr.txt:409-410` — validation set, `Recall@K`. Table 1 caption (`sprc_iclr.txt:378` region, visual page 6 image `page-06.png`) — Fashion-IQ validation. The paper **never names the candidate/gallery set**, so `fashioniq_original_split` is an inference from the code, not from the paper (P3).

**Protocol claim VERIFIED** for gallery scope and reference eligibility, and **REFUTED** for any “reference-conditioned CIR” reading of the default rerank path.

## DATASET CONTRACT

- `canonical_dataset_link: fashionIQ_dataset` (`models.yaml:835`), `fashioniq_image_extensions: [".png"]` (`:836`), `fashioniq_layout: fashioniq_standard` (`:851`), `preparation_contract: manual_standard` (`:852`), `preparation_id: sprc_fashioniq` (`:853`).
- Prep contract `workbench/registry/preparation_contracts.yaml:220-248` (`preparation_id: sprc_fashioniq`): raw inputs = `captions/cap.{dress,shirt,toptee}.val.json`, `image_splits/split.{dress,shirt,toptee}.val.json`, `images`, under `fashioniq_root`; `generated_artifacts: []`; one external asset (LAVIS BLIP-2 pretrain, `state: BLOCKED_EXTERNAL_ASSET_PROVENANCE`, `location_env_var: WORKBENCH_SPRC_BLIP2_WEIGHTS`); `deterministic_status: DETERMINISTIC_AT_EVALUATION`; `automation_policy: MANUAL_REQUIRED`.
- Source side: **no dataset-root argument**. `base_path = Path(__file__).absolute().parents[1]` (`src/data_utils.py:12`) = the checkout; reads at `:149` (captions), `:155` (splits), `:167/:170/:179/:185` (images, `.png`). OBSERVED `(checkout/'fashionIQ_dataset').exists()` → **False**; preflight blocker “canonical dataset link missing” (OBSERVED).
- OBSERVED layout helper `workbench/backend/fashioniq_layout.py:15-22` (`standard_paths`) and `:92-132` (`image_extension_problems` iterating every split id per category).
- OBSERVED local data: `data/fashionIQ_dataset` has `captions/ image_splits/ images/ qwen_captions/`; `images` 74,381 files, **0** `.jpg`; split ids dress 3817 / shirt 6346 / toptee 5373, **all unique**, **0 missing files**; query counts 2017 / 2038 / 1961. OBSERVED `file images/245600258X.png` → `JPEG image data` (renamed copy, so the `.png` hard-code is satisfiable on this host).
- **Finding (P3, doc defect, not a workbench defect):** `PHASE2_DATA_CONTRACT.md:33` and `PHASE2_SOURCES_C.md:92` claim “no SPRC `auxiliary_assets.yaml` rows” / “`auxiliary_assets.yaml` contains no SPRC rows”. REFUTED — OBSERVED rows at `auxiliary_assets.yaml:196-208` and `:209-219`.

## PREPROCESSING CONTRACT

- Source `targetpad_transform(1.25, 224)` (`src/blip_validate.py:52-53`), defined `src/data_utils.py:91-102`: `TargetPad(target_ratio, dim)`, `Resize(dim, PIL.Image.BICUBIC)` (`:100`), `CenterCrop(dim)` (`:101`), `_convert_image_to_rgb`, `ToTensor`, `Normalize((0.48145466,0.4578275,0.40821073),(0.26862954,0.26130258,0.27577711))` (`:102`); `TargetPad.__call__` `:63-88` symmetric `F.pad(..., 'constant')`.
- Paper OBSERVED `sprc_iclr.txt:363-365`: “initialized from the BLIP-2 pretrained model with **ViT-G** … We resize the input image size to 224 × 224 and with a padding ratio of **1.25**”. **Match.**
- The workbench does not re-implement this: `SPRCAdapter` passes no transform; the evaluator is invoked unchanged. `DETERMINISTIC_AT_EVALUATION` is consistent — no generated artifact, no resize recipe.
- Text preprocessing: `flattened_captions = np.array(captions).T.flatten().tolist()` then `f"{f0.strip('.?, ').capitalize()} and {f1.strip('.?, ')}"` (`src/validate_blip.py:180-183`), then `txt_processors["eval"]`. Workbench counterpart `src/datasets/fashioniq.py:151-156` (`ordered_and`) produces the identical pre-LAVIS string — but OBSERVED it has **no callsite** outside its own module and two unit tests (`src/datasets/fashioniq.py:308`, `tests/unit/test_fashioniq_caption_policies.py`, `tests/unit/test_fashioniq_encoder.py`), so it is not on the SPRC path.

## AUGMENTATION CONTRACT

- **None on the evaluation path.** `FashionIQDataset.__getitem__` applies only `self.preprocess` (`src/data_utils.py:174-187`). No `RandomCrop`/flip/colour jitter in `src/data_utils.py`.
- Registry implies none: `checkpoint_training_noise_pct: null` (`models.yaml:879`), `evaluation_noise_pct: 0` (`:880`), `strict_clean_checkpoint_available: null` (`:871`). No noise-trained variant exists, so no clean-vs-noisy distinction is needed.

## GALLERY CONTRACT

- Native gallery = complete ordered `split.{cat}.val.json`, one category per evaluation. Sizes required to match the paper: dress 3817, shirt 6346, toptee 5373 (OBSERVED).
- Workbench `fashioniq_original_split` builds exactly this (`src/evaluation/fashioniq.py:172-174`, `:204-227`) and asserts uniqueness rather than sorting (`:128`).
- Gallery features `extract_index_blip_features` (`src/utils.py:46-77`) returns `((index_features, index_features_raw), index_names)`; `name_to_feat = dict(zip(index_names, index_features[-1]))` (`src/validate_blip.py:167`) maps names to the **raw** EVA-QFormer token tensor that `inference` scores against. Duplicate ids would silently collapse here; OBSERVED ids are unique, so no effect.
- `.png` gate: OBSERVED preflight reports **no image-extension block** for sprc, because all local files are `.png`.

## METRIC CONTRACT

- Source: `distances = 1 - pred_sim` (`src/validate_blip.py:43`), ascending `argsort` (`:45`), `labels` by target equality (`:50`), one-hit assert (`:51`), `recall_at10/50` as percentages over `len(labels)` (`:54-55`).
- Aggregation `src/blip_validate.py:92-96`: `average_recall_at10 = mean(recalls_at10)`, `average_recall_at50 = mean(recalls_at50)`, `average_recall = (mean(r50)+mean(r10))/2`; printed as one JSON object (`:98`).
- Paper OBSERVED `sprc_iclr.txt:409-410`: “Recall@K … percentage of queries whose true target ranked within the top K candidates”. Match.
- Workbench: `src/evaluation/fashioniq.py:9-33` `recall_at_k` (same top-k test, descending argsort, query-count denominator), `macro_average_fashioniq` (`:238-260`) — not called on the SPRC path.
- **No SPRC parser.** OBSERVED `metrics_extraction.extract_aggregate_metrics` dispatch (`:391-402`): limn, csmcir, combiner/clip4cir only; sprc → `AGGREGATE_PARSER_UNAVAILABLE`. OBSERVED live: feeding the pinned evaluator's exact JSON key set returns `{'extraction_status': 'AGGREGATE_PARSER_UNAVAILABLE', 'observed_metrics': None, ...}`. Note the csmcir parser's key set (`:255-267`) is byte-identical to what `blip_validate.py:88-98` prints, so this is wiring, not research.

## SOURCE-TO-WORKBENCH PARITY

| axis | upstream evidence | workbench evidence | status |
|---|---|---|---|
| protocol | full ordered `split.{cat}.val.json`, ref eligible (`data_utils.py:155`, `validate_blip.py:44-55`) | `fashioniq_original_split` = `full_split`, `reference_excluded=False` (`src/evaluation/fashioniq.py:172-174`) | EQUIVALENT_WITH_EVIDENCE |
| dataset root | `<checkout>/fashionIQ_dataset` (`data_utils.py:12`) | `canonical_dataset_link` (`models.yaml:835`) + generic link step (`prepare_dataset.py:70-77`, `:28`) | MISMATCH_CONFIRMED (link absent on this host) |
| image extension | `.png` (`data_utils.py:185`) | declared `.png` (`models.yaml:836`), checked per id (`fashioniq_layout.py:92-132`) | EQUIVALENT_WITH_EVIDENCE |
| preprocessing | `targetpad_transform(1.25,224)` (`blip_validate.py:53`) | not re-implemented; source-native | EQUIVALENT_WITH_EVIDENCE |
| caption string | `.strip('.?, ').capitalize()` + `" and "` (`validate_blip.py:180-183`) | `ordered_and` (`src/datasets/fashioniq.py:151-156`), caller-less | EQUIVALENT (dormant) |
| metric maths | target-hit R@10/50, macro mean (`validate_blip.py:54-55`, `blip_validate.py:92-96`) | `recall_at_k` + `macro_average_fashioniq` | EQUIVALENT_WITH_EVIDENCE |
| invocation | `blip_validate.py:141-155` | `SPRCAdapter` has no `command()` (`adapters/models.py:119-121`, `:24-25`) | NOT_IMPLEMENTED |
| stdout → metrics | JSON at `blip_validate.py:98` | `AGGREGATE_PARSER_UNAVAILABLE` (`metrics_extraction.py:391-402`) | NOT_IMPLEMENTED |

Positively re-verified (independently, from source):

- `prepare_dataset.py` has **no** `prepare_csmcir` symbol (OBSERVED `grep` → only `prepare_canonical_link` at `:28`); links are driven generically by `models_requiring_link()` (`:70-77`) reading `canonical_dataset_link` for **every** model. OBSERVED that function returns `{'sprc': 'fashionIQ_dataset', 'csmcir': 'fashionIQ_dataset', ...}` — so **sprc's link is covered by the generic helper**. `PHASE2_SOURCES_C.md:93` (“`prepare_dataset.py:prepare_csmcir` … does not stage SPRC's link”) and `PHASE2_DATA_CONTRACT.md:113` cite a symbol that does not exist; the behaviour they describe is right, the mechanism they name is wrong (P3).
- Unregistered-name failure mode: `registry.get_model_class` returns `None` (`src/lavis/common/registry.py:236-237`), so the failure is `AttributeError: 'NoneType' object has no attribute 'from_pretrained'` at `models/__init__.py:226`, not the `KeyError` asserted by `AUXILIARY_ASSET_AUDIT.md:290` (P3).
- `AUXILIARY_ASSET_AUDIT.md:287-288` states the entrypoint's default names an unregistered class but implies the pinned repo has no such class at all. **REFUTED**: the class exists as tracked bytecode with that exact registry name (see CHECKPOINT IDENTITY). The default still fails, but for the sharper reason that bytecode-only modules are not imported by `src/lavis/models/__init__.py:35-55`.

## RUNTIME STATUS

OBSERVED `preflight_blockers(model_by_id('sprc'), resolve_config())` → six blockers:

```
LAVIS BLIP-2 pretrain weights unverified: /home/heheboiz/.cache/lavis/blip2_pretrained.pth (operator-supplied unverified provenance)
EVA-ViT-g backbone weights unverified: /home/heheboiz/.cache/torch/hub/checkpoints/eva_vit_g.pth (operator-supplied unverified provenance)
canonical dataset link missing: .../third_party/SPRC/fashionIQ_dataset -> .../data/fashionIQ_dataset
external runtime asset provenance unresolved: LAVIS BLIP-2 pretrain weights
command not runnable: COMMAND_BLOCKED_MODEL_MAPPING
execution environment not verified: WORKBENCH_SPRC_PYTHON=None [UNCONFIGURED/UNCONFIGURED] (WORKBENCH_SPRC_PYTHON not set)
```

- OBSERVED `pc_readiness.py --model sprc --json` → `state: BLOCKED_IMPLEMENTATION`, `reason: command not runnable: COMMAND_BLOCKED_MODEL_MAPPING`. Update-5 classification fix works: the “unverified” blockers classify as `BLOCKED_EXTERNAL_ASSET` only when they are the *first* match; `BLOCKER_KINDS` order (`pc_readiness.py:39-59`) puts `command not runnable` → `BLOCKED_IMPLEMENTATION` first, which is correct here.
- OBSERVED `SPRCAdapter().command(source, checkpoint, request)` → `NotImplementedError: upstream evaluation command requires audit before execution`; OBSERVED `command_is_audited(SPRCAdapter)` → `False`; OBSERVED `build_command(request)` → `FileNotFoundError: checkpoint missing: .../artifacts/checkpoints/sprc/sprc_fiq.pt` (and behind that, `adapters/base.py:34-35` would raise `checkpoint_mapping_unverified`).
- OBSERVED `runtime.verify_environment(model_by_id('sprc'), probe=True)` → `variable=WORKBENCH_SPRC_PYTHON, interpreter=null, status=UNCONFIGURED, tier=UNCONFIGURED`.
- **Import chain, re-measured in this session with a different interpreter than the prior round:**
  - `python3 -c "import clip"` (system) and `.venv/bin/python -c "import clip"` → `ModuleNotFoundError: No module named 'clip'`; OBSERVED `/home/heheboiz/data/cir/.venv/bin/python src/blip_validate.py --dataset FashionIQ --blip-model-name blip2_cir_align_prompt --model-path /tmp/nonexistent.pt` → `ModuleNotFoundError: No module named 'clip'` at `src/blip_validate.py:8`.
  - `requirements.txt` (14 lines, OBSERVED) has **no** `clip` entry; `README.md:14-17` installs only `requirements.txt`.
  - `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c "import data_utils"` → OK.
  - `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c "import validate_blip"` → `ImportError: cannot import name 'apply_chunking_to_forward' from 'transformers.modeling_utils'` (vendored LAVIS vs modern transformers). OBSERVED installed versions: `transformers 4.57.6`, `timm 1.0.30`; `spacy` not installed.
  - The registry pins `transformers==4.36.2` (`models.yaml:875`), and the pinned `requirements.txt` says `transformers==4.36.2` — so the observed mismatch is traceable to the host, not to a missing declaration.
- Environment planner **does** derive `clip` as a required distribution: OBSERVED `model_constraints(model_by_id('sprc'))["packages"]` includes `'clip': 'UNKNOWN'` (pip_only), drawn from `_evaluator_imports` (`environment_planner.py:234-...`; OBSERVED returned set includes `clip`). So the planning view knows the module; the preflight blocker list does not mention it.
- **Decoy bytes at the declared asset paths, re-observed:** `-rw-rw-r-- 22 /home/heheboiz/.cache/lavis/blip2_pretrained.pth` (`file` → `ASCII text`); `-rw-rw-r-- 4 /home/heheboiz/.cache/torch/hub/checkpoints/eva_vit_g.pth` (contents `qwen`). Both non-empty, so `validate_auxiliary_asset` (`registry.py:417-425`) can only report `OPERATOR_SUPPLIED_UNVERIFIED`. OBSERVED both auxiliary rows have `expected_size_bytes: null`, `expected_sha256: null`.
- **Second asset inventory gap:** the prep contract lists **one** external asset (BLIP-2), while `auxiliary_assets.yaml:196-219` lists **two** (adding EVA-ViT-g). OBSERVED `external_asset_statuses(model_by_id('sprc'))` returns **one** entry (`LAVIS BLIP-2 pretrain weights`, `UNRESOLVED_PROVENANCE`, `path: null`) — it reads the contract, so the ViT-g row is declared but never surfaced as a blocker. That asymmetry is why READINESS_MATRIX shows both.
- No inference executed. No GPU job, no install, no download into the repo.

## PAPER-METRIC STATUS

Re-derived independently from the primary source in this session.

- OBSERVED camera-ready `sprc_iclr.txt:407`: `Ours (SPRC) 49.18 72.43 55.64 73.89 59.35 78.58 54.92 74.97 64.85`; duplicated at `:1099` (Appendix Table 5, `Ours+BLIP-2 (Full.)`), `:1140` (Table 7, γ=0.8) and `:1164` (Table 9, length 32). Route B: OBSERVED rendered page 6 image read visually — row `Ours (SPRC) | 49.18 72.43 | 55.64 73.89 | 59.35 78.58 | 54.92 74.97 | 64.85`; venue line `Published as a conference paper at ICLR 2024`. Route C: OBSERVED arXiv v1 HTML contains `54.92` ×4 and `64.85` ×7 (of which one is “SPRC2: 64.85”).
- Registry records R@10 54.92, R@50 74.97, mean 64.85 (`models.yaml:857-859`) — **all three match the paper exactly**, and the benchmark row carries them as `PAPER_REPORTED`: OBSERVED `benchmark_references.references_for_model('sprc')` → `sprc_fiq_val_ours_vitg`, `reference_status: VERIFIED`, `reported_metrics {r10: 54.92, r50: 74.97, mean: 64.85}`, `metric_provenance` all `PAPER_REPORTED`, `category_metrics` dress 49.18/72.43, shirt 55.64/73.89, toptee 59.35/78.58.
- Registry `reported_score_note` `models.yaml:860`: “Paper Table 1 prints R@10 54.92 while printed per-category values average 54.72” — **the arithmetic is exactly right**: OBSERVED `(49.18+55.64+59.35)/3 = 54.72333`; `(72.43+73.89+78.58)/3 = 74.96667`; and `(54.92+74.97)/2 = 64.945 ≠ 64.85` while `(54.72333+74.96667)/2 = 64.845 → 64.85`. So the printed `Avg.` is `round(mean(recomputed_category_means))`, **not** the mean of the two printed Average cells. The paper is internally consistent under that reading; the only true oddity is the 0.197pp gap in the printed Average R@10 cell.
- **The paper's own prose attributes the FashionIQ 64.85 to `SPRC2`**, a two-stage re-ranked variant (`sprc_iclr.txt:435-437`: “modifying SPRC into a two-stage mechanism (termed SPRC2) achieved higher Avg. performance, i.e., Re-ranking: 62.15 vs. SPRC2: 64.85”). The same six numbers appear in the Table 1 `Ours (SPRC)` row. Every reproduction claim and reference row must be read against a paper whose prose and Table 1 row disagree about which method produced 64.85.
- **The camera-ready says ViT-G; arXiv v1 says ViT-L.** OBSERVED `sprc_iclr.txt:363`: “encoders are initialized from the BLIP-2 pretrained model with **ViT-G**”. OBSERVED arXiv v1 HTML: “…with **ViT-L** and AdamW”; `grep -o "ViT-."` over the v1 HTML returns exactly one hit, `ViT-L`. The registry correctly pins `backbone` from the camera-ready (the benchmark row's `backbone` string cites “paper text line 363” — verified). The pinned `README.md:128` publishes `sprc_fiq_vitl.pt` as “models based on vit-l”. So which published file corresponds to the Table 1 row is genuinely ambiguous across artifacts.
- **Gate behaviour, re-measured (this session):** OBSERVED with `run_meta` exactly as `evaluate_models.py:137-148` builds it and the committed reference: `r10 ∈ {0.0, 5.0, 40.0}` → `ALERT_METRIC_DEVIATION`; `r10 = 54.92` with `r50 74.97, mean 64.85` → `PASS_WITHIN_5_PERCENT`. So the gate **does** consume R@10. OBSERVED with `reproduced_category_metrics` supplied and a deliberately broken toptee (`r10 30.0`), the comparison list is `['r10','r50','mean','dress_r10','dress_r50','shirt_r10','shirt_r50','toptee_r10','toptee_r50']` → `ALERT_METRIC_DEVIATION`. The per-category gate extension works for this model. The gate itself also requires only one of `split`/`dataset`/`protocol`/`checkpoint`/noise to be stated (`paper_gate.py:331-334`), so a row with a null noise and a stated protocol is accepted; for sprc `checkpoint_training_noise_pct` is null and `_comparability_blockers` records “not stated by the reference … not checked” rather than a match.
- **No reproduction exists.** The only sprc run artifact is `workbench/artifacts/results/fashioniq_original_split/mock/mock_sprc-sprc_fiq.json` with `data_kind: "mock"`, null `reported_paper_metrics`, `gallery_size 260`. The gate refuses it (`paper_gate.py:292-294` → `NOT_RUN`).
- Reference plumbing: OBSERVED `benchmark_references.registry_version` = `extract-1b62b520c448`, `registry_digest` = `f56c4f1bdc0887c734355f9041dd9acbbb7079527144b11c209f75fa2c63983b`, equal to OBSERVED `sha256(workbench/registry/paper_benchmarks/papers.yaml)` (86,348 bytes). `select_reference('sprc', checkpoint_training_noise_pct=None, protocol_id='fashioniq_original_split')` returns the row with reason “exact condition match” — consistent with the sprc variant's `checkpoint_training_noise_pct: null`.

## CRITICAL FINDINGS

1. **C1 — P0 — the registered entrypoint cannot be imported on any host, for a reason the workbench's blocker list does not state.** `src/blip_validate.py:8` `import clip`; `requirements.txt` declares no `clip`; `README.md:14-17` installs only `requirements.txt`. OBSERVED `ModuleNotFoundError: No module named 'clip'` from the documented command shape, on two interpreters. Independently, `src/validate_blip.py:12` → vendored LAVIS → `ImportError: cannot import name 'apply_chunking_to_forward' from 'transformers.modeling_utils'` (OBSERVED). `preflight_blockers` mentions `clip` nowhere. Action: record `clip` (and the vendored-LAVIS/transformers constraint) as a declared runtime prerequisite; the planner already computes `clip` (`environment_planner._evaluator_imports`), so the fix is to surface it.

2. **C2 — P0 — `SPRCAdapter` emits no command.** `adapters/models.py:119-121` declares only `script`; `OfficialScriptAdapter.command` raises (`:24-25`). OBSERVED `command(...)` → `NotImplementedError`, `command_is_audited` → `False`, `build_command` → `FileNotFoundError`. The blocker is honest, but every doc claim that the CLI “shape has been audited” (`UPSTREAM_AUDIT.md:38`, `UPSTREAM_SOURCE_AUDIT.md:63`, `LEGACY_SOURCE_AUDIT.md:12`) has nothing behind it.

3. **C3 — P1 — the registry note asserts a class as the evaluator's model, and the pinned repo registers no such class.** `models.yaml:872` — “Native full-gallery evaluator uses `blip2_cir_rerank_learn` with pretrain backbone”. OBSERVED: that name appears exactly once in the tree, as the argparse default (`blip_validate.py:144`); no `@registry.register_model` for it; `src/lavis/models/__init__.py:35-55` imports 19 blip2 classes and not that one; `src/lavis/models/blip2_models/__init__.py` is 0 bytes. So the sentence states an unresolved default as a fact. Action: replace with the verified facts — default names a class the pinned tree does not register; the only trace is tracked py39 bytecode under `__pycache__`; the registered alternative is `blip2_cir_rerank` (`blip2_qformer_cir_rerank.py:26`), and README Training uses `blip2_cir_align_prompt` (`README.md:92`).

4. **C4 — P1 — the checkpoint is absent, its URL returns 403, and its class/backbone mapping is unprovable from README+code.** OBSERVED 301→403 on the OneDrive URL, no size/digest; `artifacts/checkpoints/` empty; `README.md:130-131` says the rerank weights were never published. `CHECKPOINT_MAPPING_UNVERIFIED` (`models.yaml:883-884`) is the correct record.

5. **C5 — P1 — decoy bytes at both declared asset paths pass the workbench's only asset check.** OBSERVED 22-byte `blip2_pretrained.pth` (ASCII text) and 4-byte `eva_vit_g.pth` (contents `qwen`), both non-empty, both with `expected_size_bytes`/`expected_sha256` null, so `validate_auxiliary_asset` cannot reject them and the blocker reads “provenance unverified” rather than “invalid”. Action: record plausible sizes for the two published Google-Storage assets, the way CSMCIR/LIMN/encoder rows already do.

6. **C6 — P1 — the ViT-g asset is declared but invisible to the blocker list.** `auxiliary_assets.yaml:209-219` declares EVA-ViT-g; the prep contract declares only BLIP-2; OBSERVED `external_asset_statuses(model_by_id('sprc'))` returns **one** entry from the contract, so the ViT-g file is never surfaced from that path. `pc_readiness` shows it only because it reads the auxiliary registry directly. Two inventories for one dependency set.

7. **C7 — P1 — no SPRC metric parser.** OBSERVED `extract_aggregate_metrics('sprc', <pinned JSON>, '', 0, paper)` → `AGGREGATE_PARSER_UNAVAILABLE`, so a successful official run could not be compared at all. The csmcir parser's key set (`metrics_extraction.py:255-267`) is identical to `blip_validate.py:88-98`'s output, so this is wiring, not new research.

8. **C8 — P2 — the paper's prose and its Table 1 row disagree about which method produced 64.85.** OBSERVED `sprc_iclr.txt:435-437` ties 64.85 to `SPRC2` (a two-stage re-ranked variant), while `:407` prints the same six numbers in the `Ours (SPRC)` row. The benchmark row names the main-method variant (`method_variant` field) without noting the SPRC2 attribution. Action: record the attribution conflict in the reference row; it changes what a reproducing checkpoint can claim.

9. **C9 — P2 — camera-ready and arXiv v1 disagree on the backbone (ViT-G vs ViT-L), which the paper never reconciles with the two published checkpoint files.** OBSERVED `ViT-G` at `sprc_iclr.txt:363`; OBSERVED `ViT-L` in arXiv v1. `README.md:128` publishes `sprc_fiq_vitl.pt`. The registry's backbone string cites only the camera-ready. Action: state both artifacts and require the class/backbone of `sprc_fiq.pt` to be inspected alongside `sprc_fiq_vitl.pt`.

## MINOR FINDINGS

- **M1 — P3 — REFUTED prior-round/док claim.** `PHASE2_DATA_CONTRACT.md:33` and `PHASE2_SOURCES_C.md:92` say `auxiliary_assets.yaml` has no SPRC rows; OBSERVED rows exist at `:196-208` and `:209-219`.
- **M2 — P3 — `prepare_dataset.py:prepare_csmcir` does not exist** (`prepare_canonical_link` `:28`, generic driver `:70-77`); sprc **is** covered. Cited by `PHASE2_SOURCES_C.md:93`, `PHASE2_DATA_CONTRACT.md:113`. Same defect as the prior round.
- **M3 — P3 — the unregistered-name failure is `AttributeError`, not `KeyError`.** `src/lavis/common/registry.py:236-237` returns `None`; `src/lavis/models/__init__.py:226` then calls `.from_pretrained` on `None`. `AUXILIARY_ASSET_AUDIT.md:290` says `KeyError`.
- **M4 — P3 — `lavis` is listed as a pip package** in `models.yaml:875` (`packages: [transformers==4.36.2, timm==0.9.12, spacy==3.7.2, lavis]`). The pinned source vendors `src/lavis`; sibling models elsewhere in the registry spell `salesforce-lavis==1.0.2` (`models.yaml:102,155,247,310,372`). As written it can send an operator looking for a nonexistent distribution. `environment_planner`'s `lavis → salesforce-lavis` mapping (`:227`) does not fix the live import path, which is `sys.path = src/`.
- **M5 — P3 — `--backbone` is omitted by the README Evaluation block** (`README.md:108-112`) while the code defaults to `pretrain`/ViT-g (`blip_validate.py:145`). Following the README verbatim is ambiguous between ViT-g and ViT-L.
- **M6 — P3 — `README.md:124` publishes `sprc_cirr.pt` in the same sentence as `sprc_fiq.pt`** and is cited in provenance strings; harmless, but a reader can mistake the CIRR weight for the FashionIQ one.
- **M7 — P3 — dead imports in the entrypoint chain.** `blip_validate.py:8` `import clip` and `:10` `import pandas as pd` are unused on the FashionIQ path (only a commented `clip.load` at `:44`); `validate_blip.py` uses `clip` only in the CIRR-relative helpers (`:221-228`). They are still hard import failures and are the practical cause of C1.
- **M8 — P3 — `blip2_qformer_cir_z_learn_pos_align.py:314-315` calls `.cuda()` unconditionally** inside `inference` (OBSERVED), so that class cannot run on a CPU-only host. Not on the default rerank path, but relevant if the checkpoint turns out to be a z-learn file. (`blip2_qformer_cir_rerank.py` and `blip2_qformer_cir_align_prompt.py` have no unconditional `.cuda()` — OBSERVED grep empty.)
- **M9 — P3 — the LAVIS `cache_root` point is style, not a defect, for this model.** `src/lavis/configs/default.yaml:4` sets `cache_root: "/raid/temp/home/.cache/lavis"` and `src/lavis/__init__.py:27-28` registers it, but OBSERVED `cache_root` is consumed only by `src/lavis/common/utils.py:47` (path helper) and the COCO task builders (`tasks/captioning.py:86`, `tasks/dialogue.py:70`) — no SPRC/FashionIQ evaluation path. The weight-fetch path is `download_cached_file` → `timm.models.hub.get_cache_dir()` (`dist_utils.py:117-136`, used at `blip2.py:87`, `eva_vit.py:443`, `clip_vit.py:252`), which reads `TORCH_HOME`/`XDG_CACHE_HOME`, so the registry's destination resolution (`registry.py:338-359`) is correct for both published assets (OBSERVED). `UPSTREAM_AUDIT.md:105-134` presents the `cache_root` mismatch as a general consequence; for sprc it changes nothing.
- **M10 — P3 — `UPSTREAM_SOURCE_AUDIT.md:42` records 9.0M**; OBSERVED `du -sh` 12M including build products (8.5M excluding `.git`). Cosmetic.

## UNRESOLVED EVIDENCE

- **U1 — which class and backbone `sprc_fiq.pt` belongs to.** Needs the file's top-level state-dict key. Not reachable: 403 (OBSERVED), no local copy, bytecode-only `*_rerank_learn` module, README Todo says the rerank weights were never published. **UNKNOWN.**
- **U2 — whether `sprc_fiq.pt` reproduces 54.92/74.97/64.85.** No checkpoint, no GPU, no importable entrypoint. **UNKNOWN**; `reproduction_status: BLOCKED_CHECKPOINT_MAPPING` is the honest label.
- **U3 — whether the Table 1 row was produced by the ViT-G or the ViT-L file.** The camera-ready says ViT-G, arXiv v1 says ViT-L, and the README publishes both a default and a `_vitl` file. **UNKNOWN**; affects the asset set.
- **U4 — whether 64.85 is SPRC or SPRC2.** Paper Table 1 row vs paper prose (`sprc_iclr.txt:407` vs `:435-437`). **UNKNOWN** from the paper alone; relevant before treating any checkpoint as the Table 1 reproducer.
- **U5 — whether the Docker/ViT-g cache state on a future host will hold real bytes.** Only the decoy bytes were observable here. **UNKNOWN.**
- **U6 — the GitHub remote SHA.** I did not query `api.github.com`; the pin is verified against the local object only.
- **U7 — cross-document sweep.** I checked every workbench doc naming sprc: `AUTOMATION_RUNBOOK.md:167`, `AUTOPILOT_GUIDE.md:87`, `AUXILIARY_ASSET_AUDIT.md:275-291`, `FASHIONIQ_PROTOCOL_PARITY.md:77,89-91,101`, `FINAL_PC_READINESS.md:154`, `LEGACY_REPRODUCTION_RUNBOOK.md:73-80`, `LEGACY_SOURCE_AUDIT.md:12,57-64`, `PC_REPRODUCTION_HANDOFF.md:32`, `PC_TRANSFER_CHECKLIST.md:58`, `READINESS_MATRIX.md:63`, `SOURCE_TO_WORKBENCH_PARITY_MATRIX.md:34,57`, `UPSTREAM_AUDIT.md:38,101-134`, `UPSTREAM_DATA_PREPARATION_AUDIT.md:37,175-181,198`, `UPSTREAM_SOURCE_AUDIT.md:42,63`, `PROTOCOL_AUDIT.md:62`, `AUTOMATED_DATA_PREPARATION.md:58`, `PHASE2_*`, `workbench/README.md:62`, `docs/reviews/FINAL_REPRODUCIBILITY_AUDIT.md:37`, `docs/reviews/PC_RELEASE_READINESS.md:48`, `docs/reviews/models/sprc.md`, `paper_benchmarks/verification/VerifyPapersB.json`. Defects surfaced: M1, M2, M3; one prior-round claim refuted (M1 of the prior round: the `blip2_cir_rerank_learn` bytecode exists). **U7 UNKNOWN** beyond those.
- Prior `docs/reviews/models/sprc.md` refresh section says readiness is `BLOCKED_SOURCE` (“upstream source unavailable”) — that is **wrong as written** for this model: `source_clean_and_pinned` returns `(True, '2935a539…', None)` (OBSERVED) and `pc_readiness` returns `BLOCKED_IMPLEMENTATION` (OBSERVED).

## FINAL VERDICT

**BLOCKED.** The source pin is exact and clean (`2935a5397732260d1db6fa577e5926f963e36f0f`), the protocol/gallery/reference-eligibility/transform/text-composition/metric arithmetic all trace faithfully to the pinned code and to the paper's own description, and the paper row (54.92/74.97/64.85, per-category 49.18/72.43, 55.64/73.89, 59.35/78.58) is now re-derived three ways in this session and consumed correctly by the gate — but nothing can run: the registered entrypoint cannot import (`clip` missing from the declared environment; vendored LAVIS vs modern transformers), no command constructor exists, the checkpoint is absent and its URL returns 403, its class/backbone mapping is unprovable (the default names a class that exists only as tracked bytecode and is not registered), both declared asset paths on this host hold decoy bytes that the workbench's only asset check cannot reject, and there is no stdout parser for this model's output shape.

---

# EXECUTION TRACE

`paper/README/config → source dataset → preprocessing → official evaluator → workbench invocation → checkpoint → output → metric comparison`

1. **paper/README/config →** OBSERVED camera-ready PDF `200 5641958`, sha256 `2c250ddb…9a3b`; Table 1 `Ours (SPRC)` row `49.18 72.43 55.64 73.89 59.35 78.58 54.92 74.97 64.85` at `sprc_iclr.txt:407`; duplicated `:1099` (Table 5), and as the γ=0.8 row (`:1140`, Table 7) and length-32 row (`:1164`, Table 9); rendered page 6 read visually (`page-06.png`); arXiv v1 HTML agrees; prose `:363` ViT-G, `:364-365` 224×224 ratio 1.25, `:409` validation set, `:435-437` 64.85 attributed to SPRC2. Pinned `README.md:14-17` (env recipe), `:92` (training uses `blip2_cir_align_prompt`), `:97-98` (`--transform targetpad --target-ratio 1.25`), `:108-112` (evaluation argv), `:124` (OneDrive files), `:128` (`_vitl` files), `:130-131` (Todo). Registry `models.yaml:843-872` matches the README pins; `packages` includes a bogus `lavis`. **Arrow holds; three primary-source disagreements recorded (ViT-G/ViT-L, SPRC/SPRC2, Average-cell arithmetic).**
2. **→ source dataset —** `base_path = Path(__file__).absolute().parents[1]` (`src/data_utils.py:12`) ⇒ `<checkout>/fashionIQ_dataset`; reads `:149` (captions), `:155` (splits), `:167/:170/:179/:185` (`.png`). No CLI root. Registry declares the link (`models.yaml:835`) and the contract the raw paths (`preparation_contracts.yaml:220-233`). OBSERVED `(checkout/'fashionIQ_dataset').exists()` → False; local `data/fashionIQ_dataset` has 3817/6346/5373 unique ids, all present as `.png`, query counts 2017/2038/1961. **Arrow holds at the source, blocked at the link.**
3. **→ preprocessing —** `targetpad_transform(1.25, 224)` (`blip_validate.py:52-53`) → `TargetPad/Resize(BICUBIC)/CenterCrop/ToTensor/Normalize` (`data_utils.py:63-102`); caption string `validate_blip.py:180-183`. Paper agrees (`sprc_iclr.txt:363-365`). No augmentation on the eval path. **Arrow holds.**
4. **→ official evaluator —** `blip_validate.py:155` → `clip_finetune_fiq(:26)` → `load_model_and_preprocess(name=<class>, model_type=backbone, …)` (`:45`) → `torch.load` (`:48`) → `load_state_dict(..., strict=False)` (`:49`) → per-category `FashionIQDataset` (`:64,:66`) → `extract_index_blip_features` (`:80`) → `compute_fiq_val_metrics` (`:81`, defined `validate_blip.py:24`) → `blip_model.inference(...)` (`validate_blip.py:198`) → `blip2_qformer_cir_rerank.py:373-397`. **Arrow BLOCKED before any model code:** OBSERVED `ModuleNotFoundError: No module named 'clip'` at `blip_validate.py:8`; behind it the vendored-LAVIS/transformers `ImportError` (OBSERVED), the unregistered default class name, and the absent checkpoint.
5. **→ workbench invocation —** `SPRCAdapter` (`adapters/models.py:119-121`) has no `command()`; base raises (`:24-25`). OBSERVED `build_command` → `FileNotFoundError: checkpoint missing: .../sprc/sprc_fiq.pt`; OBSERVED `command_is_audited` False; OBSERVED six preflight blockers, none naming `clip`. **Arrow BROKEN.**
6. **→ checkpoint —** OBSERVED 301→403 on the OneDrive URL; no local file; `expected_sha256`/`expected_size_bytes` null by necessity (`models.yaml:881-882`). The registered class name exists only as tracked py39 bytecode (OBSERVED). **Arrow BROKEN.**
7. **→ output —** none. Only the mock (`mock_sprc-sprc_fiq.json`, `data_kind: "mock"`), which the gate refuses (`paper_gate.py:292-294`). **NOT REACHED.**
8. **→ metric comparison —** OBSERVED `extract_aggregate_metrics('sprc', …)` → `AGGREGATE_PARSER_UNAVAILABLE`, so no run could be scored through the pipeline; the gate itself is correctly wired for this reference and consumes r10 + the six per-category cells (OBSERVED exact pass, OBSERVED deviation alerts). **PARITY NOT EVALUATED.**

Nothing past step 3 is `RUNTIME_VERIFIED`; steps 4-8 are each independently blocked.

---

# CHECKLIST

1. Registry entry exists and parses — VERIFIED — `models.yaml:830-884`; OBSERVED `model_by_id('sprc')` resolves — NONE — none.
2. method_name / title / year / url consistent — VERIFIED — `models.yaml:840,843-845` vs OBSERVED arXiv abs page and ICLR camera-ready title page — NONE — none.
3. `upstream_repo_url` matches the pinned remote — VERIFIED — `models.yaml:846`; no API call made (U6) — NONE — none.
4. `source_available`/`source_dir` correct — VERIFIED — `models.yaml:849-850`; OBSERVED checkout holds README + requirements + `src/` (7 scripts, vendored `lavis`, 158 `.py`) — NONE — none.
5. Pinned commit equals checkout HEAD — VERIFIED — OBSERVED `git rev-parse HEAD` → `2935a539…36f0f` = `models.yaml:848` — NONE — none.
6. Checkout clean and pinned — VERIFIED — OBSERVED `source_clean_and_pinned` → `(True,'2935a539…',None)` — NONE — none.
7. Declared entrypoint exists — VERIFIED — `adapters/models.py:121`; file exists (155 lines) — NONE — none.
8. Entrypoint is the README's — VERIFIED — `README.md:108` vs `adapters/models.py:121` — NONE — none.
9. Entrypoint imports under the declared env — BLOCKED — OBSERVED `ModuleNotFoundError: No module named 'clip'` at `blip_validate.py:8`; `requirements.txt` has no `clip`; `README.md:14-17` installs only it — P0 — declare `clip`; surface it in preflight (C1).
10. Deeper import chain resolves — BLOCKED — OBSERVED `ImportError: cannot import name 'apply_chunking_to_forward' from 'transformers.modeling_utils'` — P1 — record and verify the vendored-LAVIS/transformers constraint (C1).
11. argparse entrypoint takes name/backbone/path — VERIFIED — `blip_validate.py:143-146` — NONE — none.
12. Default `--blip-model-name` is a **registered** LAVIS model — BLOCKED — OBSERVED the name occurs once (`:144`); no `@registry.register_model` for it; `src/lavis/models/__init__.py:35-55` does not import it; `src/lavis/models/blip2_models/__init__.py` is 0 bytes — P0 — pass an explicit registered name once the class is known (C3).
13. Author class exists only as bytecode — VARIANT (new finding) — OBSERVED `git ls-files` tracks `__pycache__/blip2_qformer_cir_rerank_learn.cpython-39.pyc`; `strings` → `blip2_cir_rerank_learn`; ten `.pyc` with no `.py` — P2 — record so the mapping is not read as “no such class exists” (C3, U1).
14. Failure mode of an unregistered name recorded correctly — VARIANT — `registry.get_model_class` returns `None` (`registry.py:236-237`), CLI path calls `.from_pretrained` (`models/__init__.py:226`) ⇒ `AttributeError`, not the `KeyError` of `AUXILIARY_ASSET_AUDIT.md:290` — P3 — correct the doc (M3).
15. Non-strict load documented, not hidden — VERIFIED — `blip_validate.py:49`; registry notes it (`models.yaml:872`, contract notes) — NONE — none.
16. Mapping provable from README+code — BLOCKED — README names no class; default names an unregistered class; Todo (`README.md:130-131`) says rerank weights unpublished; Training uses `blip2_cir_align_prompt` (`README.md:92`); paper says ViT-G — P1 — inspect the file's top-level key when supplied (C4, U1).
17. Alternative published checkpoint considered — VARIANT — `README.md:128` publishes `sprc_fiq_vitl.pt`; not registered, not analysed — P2 — resolve which file the Table 1 row used (C9, U3).
18. Backbone default matches the camera-ready — VERIFIED — `blip_validate.py:145` `pretrain` = ViT-g; `sprc_iclr.txt:363` ViT-G — NONE — none.
19. Backbone artifacts disagree across paper versions — VARIANT — OBSERVED `ViT-G` in camera-ready vs `ViT-L` in arXiv v1 — P2 — record both (C9).
20. ViT-L path assets declared — BLOCKED — `blip2_pretrain_vitL.yaml:10` → `blip2_pretrained_vitL.pth` and `clip_vit.py:251` → `clip_vit_L.pth`, **neither declared anywhere** — P2 — declare conditionally on U3.
21. Where BLIP-2 base weights load from — VERIFIED — `blip2_pretrain.yaml:10` → `base_model.load_checkpoint_from_config` (`:81-102`) → `blip2.py:85-101` `download_cached_file` → `torch.load` — NONE — none.
22. Where ViT-g weights load from — VERIFIED — `eva_vit.py:428,443-445` hard-coded URL; cache root `timm.models.hub.get_cache_dir()` (`dist_utils.py:117-136`) — NONE — none.
23. Declared asset destinations are where the code looks — VERIFIED — OBSERVED `auxiliary_destination` → `~/.cache/lavis/blip2_pretrained.pth` and `~/.cache/torch/hub/checkpoints/eva_vit_g.pth`; matches `timm` `get_cache_dir()` + `XDG_CACHE_HOME`; `TORCH_HOME`/`XDG_CACHE_HOME` unset — NONE — none (M9).
24. Assets present locally are real — BLOCKED — OBSERVED 22 bytes ASCII text and 4 bytes `qwen` — P1 — add expected sizes and reject implausible ones (C5).
25. Declared `lavis` pip package is real — VARIANT — vendored `src/lavis`; siblings use `salesforce-lavis==1.0.2` (`models.yaml:102,155,...`); `environment_planner.py:227` maps `lavis`→`salesforce-lavis` but the import path is `sys.path=src/` — P3 — relabel (M4).
26. Bert tokenizer asset declared — VARIANT — required by `blip2.py:32,48,54`; other LAVIS models declare it; sprc's two rows do not — P3 — declare or document the network fetch.
27. Two asset inventories agree — VARIANT — `auxiliary_assets.yaml:196-219` lists BLIP-2 **and** EVA-ViT-g; prep contract lists only BLIP-2; OBSERVED `external_asset_statuses` returns one entry — P1 — reconcile so a blocker is not silently absent (C6).
28. Degenerate external-asset env var used — VARIANT — contract uses `WORKBENCH_SPRC_BLIP2_WEIGHTS` (`preparation_contracts.yaml`), auxiliary rows use `XDG_CACHE_HOME`/`TORCH_HOME`; `workbench.env.example:18,25` carries both — P3 — one variable per asset, or state precedence.
29. Declared protocol id registered — VERIFIED — `models.yaml:855-856` ⊆ `protocols` — NONE — none.
30. Native protocol in supported — VERIFIED — `models.yaml:856`; enforced `registry.py:46-47` — NONE — none.
31. Protocol matches the evaluator's gallery — VERIFIED — ordered `split.{cat}.val.json` (`data_utils.py:155`, `utils.py:~54-77`) = `gallery="full_split"` (`fashioniq.py:172-174`) — NONE — none.
32. Protocol matches the reference policy — VERIFIED — no masking on the metric path (`validate_blip.py:44-55`) = `reference_excluded=False` — NONE — none.
33. Whether the reference is used as query context — VARIANT — default rerank path ignores it (`blip2_qformer_cir_rerank.py:381,395`); the alternative candidate class does use it (`blip2_qformer_cir_align_prompt.py:313-359`) — P2 — qualify any registry statement; class is unresolved (U1).
34. Protocol matches the paper — UNKNOWN — paper states validation set + Recall@K (`sprc_iclr.txt:409-410`) but never names a candidate set — P3 — annotate the inference.
35. Gallery sizes match the local benchmark — VERIFIED (local) — OBSERVED 3817/6346/5373, all unique, all files present — NONE — none.
36. Gallery order preserved — VERIFIED — `data_utils.py:155` + `utils.py` (no shuffle); `dict(zip(...))` at `validate_blip.py:167` — NONE — none.
37. Gallery is not silently a pair-union — VERIFIED — loader concatenates the split file only — NONE — none.
38. Duplicate-id hazard in `name_to_feat` — VERIFIED (no effect) — `validate_blip.py:167` collapses duplicates; OBSERVED ids unique — P3 — none.
39. Every query target/reference is in the gallery — VERIFIED — OBSERVED 0 caption ids missing from `images/` or splits — NONE — none.
40. Image extension contract matches the open call — VERIFIED — `.png` (`data_utils.py:185`), declared `.png` (`models.yaml:836`), local files `.png`, gate checks every id (`fashioniq_layout.py:92-132`) — NONE — none.
41. Preprocessing matches the paper — VERIFIED — `targetpad_transform(1.25,224)` (`blip_validate.py:52-53`) vs `sprc_iclr.txt:364-365` — NONE — none.
42. Preprocessing not re-implemented by the workbench — VERIFIED — `SPRCAdapter` passes no transform — NONE — none.
43. Augmentation absent on the eval path — VERIFIED — `data_utils.py:174-187` — NONE — none.
44. Caption composition matches the source — VERIFIED — `validate_blip.py:180-183` vs `src/datasets/fashioniq.py:151-156`; helper caller-less (OBSERVED) — NONE — none.
45. No correction dictionary applied — VERIFIED — OBSERVED no correction read in SPRC scripts; `PHASE2_CROSS_REVIEW.md:98` agrees — NONE — none.
46. Metric definition is target-hit Recall@K — VERIFIED — `validate_blip.py:44-55` — NONE — none.
47. Metric denominator is the query count — VERIFIED — `len(labels)` (`:54-55`); OBSERVED 2017/2038/1961 — NONE — none.
48. Macro aggregation is the unweighted category mean — VERIFIED — `blip_validate.py:92-96` — NONE — none.
49. Workbench-side metric helper agrees mathematically — VERIFIED — `src/evaluation/fashioniq.py:9-33`, `:238-260`; not called on the SPRC path — NONE — none.
50. A workbench parser exists for this stdout — BLOCKED — OBSERVED `extract_aggregate_metrics('sprc', …)` → `AGGREGATE_PARSER_UNAVAILABLE` (`metrics_extraction.py:391-402`) — P1 — wire the existing key set (C7).
51. Reported paper metrics recorded and correct — VERIFIED — `models.yaml:857-859` = OBSERVED `sprc_iclr.txt:407` three ways; earlier rounds' `reported_r10: null` is fixed and test-locked (`tests/test_protocols.py:54`) — NONE — none.
52. The R@10 discrepancy is characterised correctly — VERIFIED — OBSERVED in python3: `(49.18+55.64+59.35)/3 = 54.72333`; `(72.43+73.89+78.58)/3 = 74.96667`; printed Avg 64.85 = `round(mean(macro10,macro50))` = `round(64.845)`, **not** `(54.92+74.97)/2 = 64.945` — NONE — none.
53. Which method produced 64.85 — UNKNOWN — paper Table 1 row (`:407`) vs paper prose (`:435-437`) — P2 — record the SPRC2 attribution conflict (C8, U4).
54. A reproduction would actually be compared — VERIFIED (mechanism) — OBSERVED `r10 ∈ {0.0,5.0,40.0}` → `ALERT_METRIC_DEVIATION`; exact trio → `PASS_WITHIN_5_PERCENT`; OBSERVED per-category comparisons `[…]'toptee_r10','toptee_r50'` and a broken toptee → alert — NONE — none.
55. Benchmark row condition selection works — VERIFIED — OBSERVED `select_reference('sprc', checkpoint_training_noise_pct=None, protocol_id='fashioniq_original_split')` → `sprc_fiq_val_ours_vitg`, reason “exact condition match”; `_comparability_blockers` records the unstated noise as unchecked, not as a match — NONE — none.
56. Registry digest reproducible — VERIFIED — OBSERVED `registry_version extract-1b62b520c448`, `registry_digest f56c4f1b…983b` = OBSERVED `sha256(papers.yaml)` (86,348 bytes) — NONE — none.
57. Workbench can execute sprc end to end — BLOCKED — OBSERVED six preflight blockers + `FileNotFoundError` from `build_command` + `NotImplementedError` from `command()` + no importer + no parser — P0 — see C1/C2/C4/C5/C6/C7.
58. Doc claim “`auxiliary_assets.yaml` contains no SPRC rows” — VARIANT (REFUTED) — OBSERVED rows at `:196-208`, `:209-219`; asserted by `PHASE2_DATA_CONTRACT.md:33`, `PHASE2_SOURCES_C.md:92` — P3 — correct the docs (M1).
59. Doc claim `prepare_dataset.py:prepare_csmcir` stages only CSMCIR — VARIANT (REFUTED) — no such symbol; generic `prepare_canonical_link` (`:28`) driven by `models_requiring_link()` (`:70-77`), OBSERVED to include sprc — P3 — correct `PHASE2_SOURCES_C.md:93`, `PHASE2_DATA_CONTRACT.md:113` (M2).
60. Prior-round “readiness `BLOCKED_SOURCE`” refresh line — VARIANT — OBSERVED `source_clean_and_pinned` True and `pc_readiness` `BLOCKED_IMPLEMENTATION` — P3 — correct the refresh text (`docs/reviews/models/sprc.md`).
61. Any runtime claim made here / repo mutated — NONE OBSERVED / NO — no GPU run, no install, no download into the repo; all probes read-only or `/tmp/audit-final/sprc/`-scoped (`sprc_iclr.pdf`, `sprc_iclr.txt`, `sprc_arxiv_v1.html`, `page-06.png`, `t3.py`, `timm_hub_0912.py`); network reads were the ICLR PDF, arXiv HTML/abs page, and HEADs on OneDrive and Google Drive — NONE — none.

---

FINAL VERDICT: **BLOCKED** — the pin, protocol, gallery, reference policy, transform, caption string and metric arithmetic are all faithful and the paper row is re-derived three ways and correctly consumed by the gate, but the registered entrypoint cannot import (`clip` missing from the declared environment; vendored LAVIS vs modern transformers), no command constructor exists, the checkpoint is absent behind a 403 URL with an unprovable class/backbone mapping (the default names a class shipped only as tracked bytecode and never registered), both declared asset paths here hold decoy bytes the workbench's only asset check cannot reject, the ViT-g asset is invisible to the blocker list, and no parser can read this evaluator's stdout.
