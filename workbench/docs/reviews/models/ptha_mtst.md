# ptha_mtst

## MODEL ID
`ptha_mtst` ("PTHA + MTST") — `workbench/registry/models.yaml:195-215` (`model_id: ptha_mtst` at `:195`, `method_name: PTHA + MTST` at `:196`). Unique: `load_registry()['models']` has 16 entries and one `ptha_mtst`. STATIC_AUDIT.

## PAPER TITLE
`Scale Up Composed Image Retrieval Learning via Modification Text Generation` (`models.yaml:200`).

Registry `paper_url: https://arxiv.org/abs/2504.05316` (`models.yaml:201`). **Resolvable, observed:** `read https://arxiv.org/abs/2504.05316` → HTTP 200, title identical, authors `Yinan Zhou, Yaxiong Wang, Haokun Lin, Chen Ma, Li Zhu, Zhedong Zheng`, submitted 2025-02-21, categories cs.IR/cs.AI/cs.CV. arXiv `Comments: 12 pages, 8 figures` — **no code URL in the comments field** (observed). A journal version exists: search result `https://ieeexplore.ieee.org/abstract/document/11125949/` (title match, not fetched).

Method names as the paper defines them (arXiv HTML v1, §I contribution list): MTST = "Modification Text-oriented Synthetic Triplets" (the generator + two pretraining datasets, 808K nature / 579K fashion triplets); PTHA = "Prototypical Two-Hop Alignment" (the retrieval network). So `method_name: PTHA + MTST` names a *pretraining-data pipeline plus a retrieval network*, not a single released model. STATIC_AUDIT.

## PINNED SOURCE COMMIT
- Registry: `upstream_repo_url: null`, `upstream_default_branch: null`, `upstream_commit_sha: null`, `source_available: false`, `source_dir: null` (`models.yaml:202-206`).
- `ls workbench/third_party/` → `AirKnow CLIP4Cir CLVCNet CSMCIR ConeSep DCNet ENCODER HABIT HINT INTENT LIMN PAIR SPRC TGCIR` (14, none is PTHA/MTST). No checkout exists. **Registry ↔ filesystem consistent.**
- `source_provenance` therefore reports `expected_commit None`, `state: unavailable` (`.venv/bin/python -c "from workbench.scripts.doctor import source_state; ..."` → `{'path': None, 'state': 'unavailable', 'head': None, 'dirty': None, 'pin': None}`).

### REFUTED: an official author repository exists and is reachable
The registry's `null`/`source_available: false` and `workbench/registry/assets/cluster_b5_ptha_mtst.yaml:33` ("the workbench does not invent one") are **incomplete, not wrong-by-policy**: a correct official repository was found by search and fetched. Evidence (all OBSERVED network, STATIC_AUDIT):

| Fact | Command / observed output |
|---|---|
| Repository | `https://github.com/zyinan99/MTST_PTHA` — `GET https://api.github.com/repos/zyinan99/MTST_PTHA` → `full_name zyinan99/MTST_PTHA`, `default_branch main`, `created_at 2024-12-26T06:16:27Z`, `pushed_at 2025-02-09T10:52:31Z`, `size 7906`, `license MIT`, `fork false`, `stargazers 0` |
| Identity | Repo `README.md` line 1 is the **exact paper title**; `GET https://api.github.com/repos/zyinan99/MTST_PTHA/commits` → author `Yinan Zhou` (paper's first author) on all commits |
| HEAD | `GET .../branches/main` → `6a02ee4763fc2260b6abfb18be78853facb91d86`; commits `6a02ee47… 2025-02-09T10:52:27Z`, `36979a41… 2025-02-09T10:49:52Z`, `e064beb5…`, `f41ed83b…`, `3584a053… 2024-12-26` |
| Tree | `GET .../git/trees/main?recursive=1` → `truncated: false`, 433 entries, tops `MTST_PTHA/`, `README.md`, `requirements.txt`, `LICENSE`, `.gitattributes`, `.gitignore` |
| Raw fetch works | `curl https://raw.githubusercontent.com/zyinan99/MTST_PTHA/main/MTST_PTHA/src/blip_validate.py` → `200`; `.../src/data_utils.py` → `200`; `.../src/validate_blip.py` → `200`; `.../src/utils.py` → `200`; `.../src/blip_fine_tune_2.py` → `200` |
| Environment | `README.md` "Prerequisites (Following SPRC)": `conda create -n mtst -y python=3.9`, `pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2`; `requirements.txt` = `tqdm==4.66.1 omegaconf==2.3.0 iopath==0.1.10 timm==0.9.12 decord==0.6.0 webdataset==0.2.86 opencv-python transformers==4.36.2 fairscale==0.4.13 einops==0.7.0 scikit-image==0.22.0 spacy==3.7.2 pycocoevalcap==1.2` |

`workbench/registry/assets/cluster_b5_ptha_mtst.yaml:60-62` records `search_locations_checked` as exactly two places, both inside the workbench. No web or Hub search is recorded anywhere (`grep -rn "zyinan99\|MTST_PTHA\|mtst_ptha\|yinanzhou1" .` excluding `.git/` and `node_modules` → **0 matches**). The "no source" conclusion rests on a two-entry local search, and it is falsified by a GitHub repository search.

**Verdict: registry `upstream_repo_url`/`source_available` REFUTED (P1).** There is still no *pinned* commit in the registry, so there is nothing pinned to audit; the evidence above is against upstream `main` HEAD `6a02ee47…`, which is a moving target.

## CHECKPOINT IDENTITY
Registry: `checkpoint_variants: []` (`models.yaml:215`), `checkpoint_bundles` absent, `strict_clean_checkpoint_available: false` (`:213`). `select_checkpoints(model, policy='strict_clean')` → `{'status': 'BLOCKED', 'reason_code': 'BLOCKED_CHECKPOINT', 'reason': 'no checkpoint variant is declared in the registry', 'checkpoint_ids': [], 'strict_clean_checkpoint_available': False}` (executed, read-only). `workbench/artifacts/checkpoints/` contains only `.gitkeep`.

### Independent hunt for a published checkpoint (assignment focus)
`GET https://huggingface.co/api/models?search=MTST&limit=10` returned one relevant hit: **`yinanzhou1/mtst_ptha`**. Full inspection (`GET .../api/models/yinanzhou1/mtst_ptha/tree/main?recursive=true&expand=true`):

| path | size (bytes) | LFS sha256 |
|---|---|---|
| `.gitattributes` | 1754 | — |
| `README.md` | 31 (body: `---\nlicense: apache-2.0\n---`) | — |
| `cirr_modifier_generated_5p_completed_rev.json` | 298419312 | `6cb4c02170f703da5ae40ae6d988a6aae272c2fa821d30c1d090de671ddfbd4d` |
| `cirr_train_modifier_rev_long.json` | 15591788 | `b1b7e72b3416e4479e4ddb9a807e789210e8d4f9d32210da460a22882f90d15d` |
| `tuned_mtst_cirr.pt` | 2720352973 | `1c4dab86a9f66dfe10aff75b39790432eb6cc746cf27b6cfab74b66d93b7588f` |

Hub repo `sha b5e669c8ef28b0084f43398d184a5a0116903ccf`, `lastModified 2024-12-26T07:42:49Z`, `usedStorage 3034364073`, `gated false`, tags `license:apache-2.0`, `region:us`, **no model card**.

Negative checks (all OBSERVED):
- The upstream README links **exactly three** downloads, all three CIRR-related (the three files above). No FashionIQ weight is linked.
- `GET https://api.github.com/repos/zyinan99/MTST_PTHA/git/trees/main?recursive=1` → **zero** `.pt`/`.pth`/`.bin`/`.ckpt` blobs anywhere in the repo.
- `GET https://huggingface.co/api/models?author=yinanzhou1&limit=50` → six repos, only `yinanzhou1/mtst_ptha` is CIR-related (the rest are maths/nvila/frame-selection experiments).
- `curl -I .../yinanzhou1/mtst_ptha/resolve/main/{fashioniq.pt, mtst_fashioniq, fashioniq_mtst.pt, ptha_fashioniq.pt, tuned_mtst_fashioniq.pt}` → **404** for all five.
- `GET .../yinanzhou1/mtst_ptha/discussions` → `{"discussions":[],"count":0}`.
- `GET https://api.github.com/search/repositories?q=MTST+composed+image+retrieval`, `q=PTHA+composed+image+retrieval`, `q=modification+text+synthetic+triplets` → `total_count 0` each. `q=ptha+mtst` → 1, `zyinan99/MTST_PTHA`.

**Verdict: the registry's checkpoint claims are VERIFIED.** `strict_clean_checkpoint_available: false`, `reproduction_status: BLOCKED_NO_VERIFIED_FASHIONIQ_CHECKPOINT` (`models.yaml:212`) and the `notes` line "CIRR pretraining evidence is not eligible" (`:214`) all hold: the only published weight is `tuned_mtst_cirr.pt`, a CIRR MTST-pretrained model, and its companion data files are CIRR-only. No FashionIQ fine-tuned checkpoint is published. STATIC_AUDIT (metadata only; **no byte was downloaded and no weight was loaded**).

## EVALUATION PROTOCOL
Registry: `native_protocol: null`, `supported_protocols: []` (`models.yaml:207-208`). Nothing is declared.

Because no source was pinned, the protocol was never established from source. I established it from the *reachable* upstream at HEAD, STATIC_AUDIT:

- Dispatch: `MTST_PTHA/src/blip_validate.py:186-200` — `if __name__ == '__main__'`, `--dataset` required (`:188`), `--blip-model-name`, `--backbone`, `--model-path` (`:189-191`); `:199-200` `elif args.dataset.lower() == 'fashioniq': clip_finetune_fiq(['dress','toptee','shirt'], args.blip_model_name, args.backbone, args.model_path)`.
- Per category: `blip_validate.py:64` `FashionIQDataset('val', [dress_type], 'relative', preprocess)` and `:66` `'classic'`; `:80` `extract_index_blip_features(classic_val_dataset, blip_model)`; `:81` `compute_fiq_val_metrics(...)`.
- Gallery: the **complete ordered validation split file**, per category, with reference **eligible** — `validate_blip.py:44-51` builds `labels` against `target_names` only and contains **no reference mask**; contrast `validate_blip.py:258-260` and `:331-333`, where the **CIRR** path *does* build a `reference_mask`. The FashionIQ path has no such mask.
- This is exactly the workbench protocol `fashioniq_original_split`: `src/evaluation/fashioniq.py:160-162` `protocol_id="fashioniq_original_split", gallery="full_split", reference_excluded=False, aliases=("fashioniq_original",)`.

**Verdict: the correct registry value would be `native_protocol: fashioniq_original_split` + `supported_protocols: [fashioniq_original_split]`; it is `null`/`[]` — VARIANT, P1.** This is an INFERENCE from reachable upstream (no pin), so it must be re-derived against a pin before it is recorded. Note the paper gives no protocol detail beyond "R@10 and R@50 on the val set in three categories" (arXiv HTML §V-A-1), so the source is the only authority; the source is unambiguous.

## DATASET CONTRACT
Registry: `preparation_contract: unavailable` (`models.yaml:199`), `fashioniq_layout` absent, `fashioniq_image_extensions` absent. `preparation_contract_for_model(model)` → `None` (no `preparation_id`), so `preflight.model_requirements` takes the `fashioniq_required_paths` branch, and `fashioniq_required_paths(model, root)` → `()` because `fashioniq_layout is None` (`registry.py:679-683`). Consequence, executed: `model_requirements(...)` → `blocks: []`, `paths: []` for `ptha_mtst`. **The model asserts no data requirement at all**, which is why its only blockers are the source and the command.

But the `preparation_contract: unavailable` sentinel *is* a deliberate registered value: `registry.py:73` accepts it in the allow-list `{"manual_standard", "manual_ilearn_resized", "manual_clvc_resized", "manual_dcnet", "unavailable"}`, so its absence is enforced, not accidental.

What the upstream evaluator actually opens (STATIC_AUDIT, `MTST_PTHA/src/data_utils.py`, `base_path` hardcoded at `:15` as the literal string `Path("path_to_data_dir")` — the README instructs the operator to edit it):
- `:153` `<base_path>/fashionIQ_dataset/captions/cap.{cat}.val.json` (val/test)
- `:157` `<base_path>/fashionIQ_dataset/captions/cap_train_{cat}_rev.json` — **line-delimited JSON** (`for line in f: json.loads(line)`), train only, and **this file is not in the paper's or README's data layout**
- `:167` `<base_path>/fashionIQ_dataset/image_splits/split.{cat}.val.json`
- `:170`/`:206` `<base_path>/fashionIQ_dataset/images/{name}.png` — **`.png` hardcoded**
- `:169-172` every split id whose `.png` is missing is **silently removed** from `image_names`
- CIRR (not on the FashionIQ path): `:257` `<base_path>/cirr_dataset/cirr/captions/cap.rc2.{split}.json`, `:263` `cirr_dataset/cirr/captions/cirr_train_modifier_rev_long.json`, `:270` `split.rc2.{split}.json`

Local workbench dataset, OBSERVED: `data/fashionIQ_dataset/` contains `captions/ image_splits/ images/ qwen_captions/`; `captions/` holds exactly `cap.{dress,shirt,toptee}.{test,train,val}.json` (**no `cap_train_*_rev.json`**); `images/` has 74381 files, **100% `.png`**; split sizes 3817/6346/5373, each internally unique; every reference and every target present in its gallery (`refs missing in gallery: 0`, `targets missing: 0`, `ref==target count: 0` for all three categories).

**Verdict: VARIANT, P2.** The registry records no dataset contract, and the one it *could* record cannot be satisfied anyway: the resolved `fashioniq_standard`-style layout (`.png` images, `captions/cap.{cat}.val.json`, `image_splits/split.{cat}.val.json`) is present locally and the `.png` convention matches the source, but the **evaluation path is reachable only if a pinned source is ever added**; the train-only `cap_train_{cat}_rev.json` line-delimited file has no published source. No blocker is raised today because no layout is declared, so a future operator who pins the source would get **no data preflight at all** for this model. That is the real defect: the empty contract is a silent hole, not just a missing value.

## PREPROCESSING CONTRACT
- Registry: nothing.
- Upstream, STATIC_AUDIT: `MTST_PTHA/src/blip_validate.py:53` `preprocess = targetpad_transform(1.25, input_dim)` with `input_dim = 224` (`:52`); identical at `:121` and `:159` for the other two validators. `data_utils.py:52` `class TargetPad` and the `targetpad_transform(target_ratio, dim)` factory ending at `:100-106`: `Compose([TargetPad(target_ratio, dim), Resize(dim, interpolation=PIL.Image.BICUBIC), CenterCrop(dim), _convert_image_to_rgb, ToTensor, Normalize((0.48145466,0.4578275,0.40821073),(0.26862954,0.26130258,0.27577711))])` — the CLIP normalisation constants.
- `data_utils.py:187`/`:192` run `self.preprocess(PIL.Image.open(path))` on val; `:206` on classic/gallery.

**Verdict: VARIANT, P3.** A well-defined, deterministic, source-exact recipe (TargetPad@1.25 → Resize → CenterCrop → RGB → Tensor → CLIP normalise) exists and is traceable, but **nothing of it is in the registry**, and it cannot be registered until a source is pinned. Note the workbench would need `--target-ratio 1.25` hardcoded, since upstream hardcodes it rather than exposing it on the FashionIQ path.

## AUGMENTATION CONTRACT
- Registry: nothing.
- Upstream eval path: no randomised operation. `data_utils.py:187,192,201,206` apply only `self.preprocess`; the same `preprocess` object (a `Compose` of deterministic ops) is used by `blip_validate.py:53`.
- Train-time augmentation/regularisation exists elsewhere in the family (`SquarePad` `data_utils.py:30`, `TargetPad` `:52`) and `utils.py:generate_randomized_fiq_caption` (imported at `blip_validate.py:21`) — a **training-time caption augmentation**, never called from `blip_validate.py` (its only caller is `blip_fine_tune_2.py:124`).

**Verdict: VERIFIED (evaluation is augmentation-free), STATIC_AUDIT.** The registry's silence is accurate for evaluation but unregistered. Severity NONE for the code; the missing registry field is P3.

## GALLERY CONTRACT
Upstream, STATIC_AUDIT, `MTST_PTHA/src/`:
- Constructed by `data_utils.py:165-168`: for each `dress_type` in the list, `self.image_names.extend(json.load(open('fashionIQ_dataset/image_splits/split.{cat}.val.json')))` — appended in **file order**, one category per dataset instance (`blip_validate.py:66` passes `[dress_type]`).
- `data_utils.py:169-172` then filters: `for image_name in self.image_names[:]: if not (…/images/{name}.png).exists(): self.image_names.remove(image_name)`. **The gallery size is a function of the local image mirror**, silently.
- Enumerated by `utils.py:54-55` `DataLoader(dataset, batch_size=64, num_workers=2, pin_memory=True, collate_fn=collate_fn)` — `shuffle` not passed, so **file order preserved**; `utils.py:76` `index_names.extend(names)`; `utils.py:70` `blip_model.extract_target_features(images, mode="mean")`; `utils.py:80` returns `(index_features, index_features_raw), index_names`.
- Consumed at `validate_blip.py:169` `name_to_feat = dict(zip(index_names, index_features[-1]))` — a dict, so a duplicate id would collapse while `index_names` (a list) keeps both.
- **Reference is eligible.** `validate_blip.py:44-51` sorts all gallery entries and builds `labels` against targets only; no mask. References are additionally *used* as the query-side image: `validate_blip.py:190-194` `torch.stack(itemgetter(*reference_names)(name_to_feat))`.
- Category iteration order is `['dress','toptee','shirt']` (`blip_validate.py:200`), while paper Table III prints `Dress | Shirt | Toptee`; the result dict is keyed by name (`blip_validate.py:90-92`), so the averages are unaffected — cosmetic only (P3).

Local, OBSERVED: `split.{dress,shirt,toptee}.val.json` = 3817 / 6346 / 5373, all 15415 ids unique within their category, **0** ids filtered out by the `.png` existence check, `0` references or targets absent from their gallery. Duplicates within a category: none, so `name_to_feat` collapses nothing here.

**Verdict: VERIFIED against upstream HEAD (full split, file order, reference eligible); the registry has no gallery contract to check — VARIANT, P1.** The `workbench` protocol id that describes this gallery+reference policy already exists (`src/evaluation/fashioniq.py:160-162`) and is already used by four other registry models; `ptha_mtst` simply does not record it.

## METRIC CONTRACT
Upstream, STATIC_AUDIT:
- `validate_blip.py:24-56` `compute_fiq_val_metrics`: `:44` `distances = 1 - pred_sim`; `:45` `torch.argsort(distances, dim=-1)` (ascending distance = descending similarity); `:46` `sorted_index_names = np.array(index_names)[sorted_indices]`; `:49-50` `labels = sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1)`; `:51` `assert torch.equal(torch.sum(labels,dim=-1).int(), torch.ones(len(target_names)).int())`; `:54-55` `recall_at10 = (torch.sum(labels[:,:10]) / len(labels)).item() * 100`, likewise `[:, :50]`.
- Macro: `blip_validate.py:90-95` keys `f'{cat}_recall_at10'`/`at50` per category, then `f'average_recall_at10': mean(recalls_at10)`, `f'average_recall_at50': mean(recalls_at50)`, `f'average_recall': (mean(recalls_at50)+mean(recalls_at10))/2`; printed by `:98` `json.dumps(results_dict, indent=4)`.
- So `mean` = unweighted mean over the three categories of `(R@10_mean + R@50_mean)/2`. `reported_mean 65.64` (`models.yaml:211`) is exactly this convention: `(54.99 + 76.28)/2 = 65.635` → `65.64` (OBSERVED arithmetic).
- The FashionIQ path produces **no R@1/R@5/group recall**; those exist only in the CIRR/CIRCO branches (`validate_blip.py:277-285`, `:348-356`).
- No workbench parser exists for this model: `grep -n "^def parse_" workbench/backend/metrics_extraction.py` → only `parse_limn_output:57` and `parse_csmcir_output:200`. `grep -rn "ptha" workbench/backend/metrics_extraction.py workbench/backend/paper_gate.py` → 0 matches.

**Verdict: VARIANT, P2.** The metric maths is source-traceable and the registry's `reported_mean` uses the right convention, but the registry declares no metric contract, no parser exists for this output shape, and the emitted key names (`{cat}_recall_at10`, `average_recall*`) would have to be bound to `parse_*` before `paper_gate.py` could compare anything.

## SOURCE-TO-WORKBENCH PARITY
`workbench/docs/SOURCE_TO_WORKBENCH_PARITY_MATRIX.md:35`: `| ptha_mtst | no source | NOT_APPLICABLE | SOURCE_EVIDENCE_MISSING | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | SOURCE_EVIDENCE_MISSING |`.

This is **honest for the workbench**, because there is no pinned checkout to compare, and `preparation_contract: unavailable` correctly propagates. But `SOURCE_EVIDENCE_MISSING` is now partly false as a statement about the world: the evaluation source exists and is reachable (see PINNED SOURCE COMMIT). The matrix's claim is scoped to the checkout, and that scoping is not stated in the cell.

**Verdict: VARIANT, P2** — the parity cells are vacuous-truth `NOT_APPLICABLE`, and `SOURCE_EVIDENCE_MISSING` should read `NOT_PINNED_IN_WORKBENCH` to avoid implying the source does not exist.

## RUNTIME STATUS
STATIC_AUDIT. Nothing was executed and no GPU exists for this assignment. Observed, executed read-only:
- `.venv/bin/python -c "...preflight_blockers(model, cfg)"` → `['upstream source unavailable', 'command not runnable: COMMAND_UNAUDITED']` (`workbench/backend/preflight.py:131-141`, where `:133` builds `source = ... if model.get("source_dir") else None` → `None`, and `:134-135` append `"upstream source unavailable"`).
- `doctor.collect(config, all_models=True, scope='real')` for `*:ptha_mtst` → `source:ptha_mtst BLOCKED "source unavailable, dirty, or not pinned"`, `runtime:ptha_mtst BLOCKED "command unavailable"` with `reproduction_status: BLOCKED_NO_VERIFIED_FASHIONIQ_CHECKPOINT`, `command:ptha_mtst BLOCKED` (detail `COMMAND_AUDITED`, `script: null`), `environment:ptha_mtst OK` (detail `UNCONFIGURED`, `reason: "no environment variable declared"`), `external-assets:ptha_mtst OK` (`assets: []`).
- `ADAPTERS` (executed) → `['airknow','clip4cir_rn50x4_fullft','clvc_net','combiner_rn50x4_noft','conesep','csmcir','dcnet','encoder','habit','hint','intent','limn','pair','sprc','tgcir']` — 15 entries, **`ptha_mtst` absent**.
- `environment_planner.plan_environments` excludes it before any solve: `environment_planner.py:810-812` `if not _has_source_or_checkpoint(model): blocked.append({... "source unavailable and no checkpoint variants; source: registry model record for {mid} (source_available=false, checkpoint_variants empty)"})`.
- `preflight_blockers` reports **no data blocker** (see DATASET CONTRACT) — the model's silence about data means the dataset root, the `.png` extension, and the corrected-caption requirement are all unchecked.
- `workbench/artifacts/results/` contains no `*ptha*` file; `find workbench/artifacts/checkpoints -type f` → `.gitkeep` only.

**Verdict: NOT_RUN → BLOCKED.** No adapter, no checkpoint, no source, no result artifact, no environment variable. Every downstream claim is unexecuted.

## PAPER-METRIC STATUS
Registry `reported_r10: 54.99`, `reported_r50: 76.28`, `reported_mean: 65.64` (`models.yaml:209-211`).

OBSERVED, network fetch of `https://arxiv.org/html/2504.05316v1`, Table III ("Comparison on FashionIQ validation set. 'Avg.' means (Recall@10 + Recall@50)/2"). The row, extracted verbatim:

```
PTHA (Ours) BLIP2 FashionIQ_MTST  50.77 73.78  55.91 75.36  58.28 79.70  54.99 76.28  65.64
                                    |dress|      |shirt|      |toptee|    averages
```

Column order is `Dress | Shirt | Toptee | Average` with `R@10 R@50` per block; the averages block is `54.99 76.28 65.64`. **All three registry numbers match the paper table exactly.** Two adjacent rows confirm the mapping is the row the registry means, not a neighbour: `PTHA (Ours) BLIP2` (no MTST) reads `… 54.33 75.24 64.79`, and `SPRC BLIP2` reads `… 54.92 74.97 64.85`.

Arithmetic check, OBSERVED: `(54.99 + 76.28) / 2 = 65.635`, which the paper prints as `65.64` — consistent with the paper's own `Avg.` convention, and with the source's `(mean(r50)+mean(r10))/2` (`blip_validate.py:95`).

Also OBSERVED: the paper's abstract states "improvements of … +1.57 in Avg. on the FashionIQ benchmark" — consistent with the `+0.85`-per-metric deltas visible in the table (`64.79 → 65.64`), i.e. the reported row is the MTST-pretrained variant, which is the row requiring the unpublished FashionIQ MTST pretraining data.

**Verdict: reported numbers VERIFIED as a transcription (numbers are verifiable); reproduction UNKNOWN.** The numbers are verifiable — the paper is public, reachable, and the three values are exact. Whether any artifact in existence can produce them is a separate question, answered NO on current evidence (see CRITICAL FINDINGS).

## CRITICAL FINDINGS

### C1 — An official author repository exists; the registry records `null` (P1, refuted claim)
Evidence in PINNED SOURCE COMMIT. `https://github.com/zyinan99/MTST_PTHA`, HEAD `6a02ee4763fc2260b6abfb18be78853facb91d86`, MIT, authored by the paper's first author, README title identical to the paper. `models.yaml:202` `upstream_repo_url: null`, `:205` `source_available: false`, and `cluster_b5_ptha_mtst.yaml:33` "Replacing this blocker requires the operator to supply an official repository URL" are all falsified — the URL is discoverable by a public GitHub repository search.

Consequences: `REPRODUCTION_STATUS` is keyed on a **wrong** primary observation ("no source") when the accurate primary blocker is "no published FashionIQ checkpoint" — which is C3 and independently true. The registry's *outcome* (BLOCKED) survives, but its *reason* is half wrong, and the wrong half is the one an operator would act on. Also: the model is excluded from environment planning, adapter construction, data preflight, and parity work on the strength of that false premise.

Severity **P1**, not P0: nothing is corrupted by the omission; a reproduction path is foreclosed without being known to be impossible.

### C2 — The reachable upstream package cannot be imported (P0 for reproduction; STATIC_AUDIT)
`MTST_PTHA/src/lavis/models/__init__.py` line 54:
```python
from lavis.models.blip2_models.blip2_qformer_tgir_concatembedding_sim_zyn import Blip2Qformer4TGIRCESIMZYN
```
No such file exists. OBSERVED: `curl -o /dev/null -w %{http_code} https://raw.githubusercontent.com/zyinan99/MTST_PTHA/main/MTST_PTHA/src/lavis/models/blip2_models/blip2_qformer_tgir_concatembedding_sim_zyn.py` → **404**; and the file is **absent from the full tree** — I re-derived it: for every `from lavis.models.*` line in `__init__.py` (48 of them, 296-line file), I resolved the module to `MTST_PTHA/src/<path>.py` and checked membership in the 361-blob tree. Result: `MISSING: 1 — line 54 lavis.models.blip2_models.blip2_qformer_tgir_concatembedding_sim_zyn`.
- The statement is **unguarded**: the nearest `try:`/`except` in the file is far below (first `try:` at line 115-ish region; the import block 13-68 is flat — `grep -n "try:\|except"` shows only line 8-area imports and later function bodies). None of lines 13-68 sits inside a `try:`.
- It is reached by the evaluator: `blip_validate.py:16` `from lavis.models import load_model_and_preprocess`.
- The sibling that *does* exist is the one the training command needs: `blip2_qformer_tgir_concatembedding_sim.py` → `200`, and it registers `blip2_tgir_ce_sim` (`@registry.register_model("blip2_tgir_ce_sim")` at its line 38). The missing file registered `Blip2Qformer4TGIRCESIMZYN`; nothing else in the repo imports that name.

This is the **same defect class** as CSMCIR's 98 unguarded imports (`workbench/patches/csmcir.md`, `workbench/docs/reviews/models/csmcir.md` C1) — an author working copy whose `__init__.py` references an untracked module. Consequence: **`python src/blip_validate.py --dataset fashioniq …` cannot start from a clean clone at HEAD.** No `--model-path` and no checkpoint can fix it. Severity **P0**, but scoped: I audited `main` HEAD, not a registry pin, and the registry has no pin.

### C3 — No FashionIQ checkpoint is published, and the required pretraining data is not published either (P0, registry claim VERIFIED)
Registry claims reconfirmed in every direction (see CHECKPOINT IDENTITY): one author Hub repo, one weight file, and that file is CIRR. Additionally: the README documents `--dataset {'CIRRGen' or 'FashionIQGen'}` for pretraining, but `blip_fine_tune_2.py:400` accepts only `['fashioniq','cirr']` and `data_utils.py` defines **no `FashionIQGen` class** (`grep -n "FashionIQGen" data_utils.py` → empty; the only `*Gen` class is `CIRRDatasetGen` at `data_utils.py:323`). The paper's FashionIQ MTST corpus (579,114 triplets, §III-B) is therefore **neither published nor loadable**.

So the registry's `reproduction_status: BLOCKED_NO_VERIFIED_FASHIONIQ_CHECKPOINT` describes a **two-sided** blocker that the single-phrase status under-describes: the paper row is `PTHA (Ours) BLIP2 FashionIQ_MTST`, which needs (a) the 579K FashionIQ MTST triplets and (b) a FashionIQ fine-tuned PTHA weight. Neither is public. Severity **P0** for the reproduction claim; the registry's `strict_clean_checkpoint_available: false` and the `notes` line are **correct**.

### C4 — `READINESS_MATRIX.md` labels the evaluator `COMMAND_AUDITED` for a model with no adapter (P2)
`workbench/docs/READINESS_MATRIX.md:28` ends `… ; evaluator command: COMMAND_AUDITED`.
Traced: `readiness_matrix.py:114` appends `f"evaluator command: {command.get('detail')}"` when `command.get("status") != "OK"`; `doctor.py:168` computes `command_audited = command_implemented and command_status in {...}` where `command_implemented = adapter is not None and getattr(adapter, "command", None) is not OfficialScriptAdapter.command` — `ADAPTERS.get('ptha_mtst')` is `None`, so `command_audited` is `False`, and `doctor.py:169` stores `record(name, "BLOCKED", command_status or "COMMAND_AUDITED", ...)` — `command_status` is `null`, so the **detail string falls back to the literal `"COMMAND_AUDITED"`**. The matrix then prints that fallback as if it were a finding.

Effect: a reader sees the strongest possible evaluator status on a model that has no evaluator. Severity **P2** (misreporting; it is the *blocker text* that is wrong, not a metric). Action: print the `status` (`BLOCKED`) plus `script: null`, or use an explicit `COMMAND_UNAUDITED` label for the null case.

### C5 — The declared blocker states a repository does not exist; it does (P1)
Same evidence as C1, stated as a finding because it is the sentence an operator acts on: `cluster_b5_ptha_mtst.yaml:32-33` (`"No official repository has been pinned, so neither the evaluation source nor its auxiliary assets can be audited."` / `"… requires the operator to supply an official repository URL and revision; the workbench does not invent one."`), `workbench/docs/AUTOPILOT_GUIDE.md:83` ("no … eligible source integration"), `PC_REPRODUCTION_HANDOFF.md:28` ("no source integration"), `FINAL_PC_READINESS.md:151` ("no source and no checkpoint variant; blocked."), `UPSTREAM_DATA_PREPARATION_AUDIT.md:19,38,184-186` (`SOURCE_UNAVAILABLE`), `UPSTREAM_SOURCE_AUDIT.md:46-47` ("PTHA + MTST has no verified upstream repository in the registry and was not cloned"). The last one is carefully worded ("in the registry") and correct; the others are not.

## MINOR FINDINGS
- **M1 (P3) — The documented evaluation default names a model that does not exist.** `blip_validate.py:189` `parser.add_argument("--blip-model-name", default="blip2_cir_rerank_learn", type=str)`. `grep -rn "rerank_learn" lavis_models_init.py` → 0; repo-wide tree search for `rerank_learn` → 0 blobs. The registered sibling is `blip2_cir_rerank` (`blip2_qformer_cir_rerank.py:26`). The README's Evaluation section passes `{trained model name}` explicitly, so the dead default is only hit by a bare invocation — where it fails at the registry lookup, not at argparse.
- **M2 (P2) — README pre-training command is unrunnable for the fashion domain.** README documents `--dataset {'CIRRGen' or 'FashionIQGen'}`; `blip_fine_tune_2.py:400` rejects anything but `fashioniq`/`cirr` (raising `ValueError` at `:401`), and no `FashionIQGen` dataset class exists (`data_utils.py` defines `FashionIQDataset:111`, `CIRRDataset:224`, `CIRRDatasetGen:323`, `CIRCODataset:430`, `COCODataset:570`, `COCOValSubset:592`, `VAWDataset:652`, `VAWValSubset:675`). The FashionIQ MTST pipeline is documented but not shipped.
- **M3 (P3) — `data_utils.py:169-172` silently shrinks the gallery.** Any val-split id lacking `images/{id}.png` is dropped from `image_names` without a warning or a count. On the local workbench dataset this is 0/15415 (OBSERVED), but the noise is invisible on a partial mirror, and it changes Recall@K denominators per category.
- **M4 (P3) — Category iteration order differs from the paper's table order.** Source `['dress','toptee','shirt']` (`blip_validate.py:200`) vs paper `Dress | Shirt | Toptee` (Table III). Keys are name-based (`blip_validate.py:90-92`), so no numeric effect; recorded so a future log-vs-table diff is not misread as an anomaly.
- **M5 (P2) — `cluster_b5_ptha_mtst.yaml` is not machine-read.** `auxiliary_assets_for_model` reads only `registry/auxiliary_assets.yaml` (`registry.py:15,107`) and `preparation_contract_for_model` only `registry/preparation_contracts.yaml` (`:16,142`); no backend or test module references `registry/assets/*.yaml` (`grep -rn "cluster_b5\|cluster_b_ilearn\|cluster_a_encoder"` across the repo, excluding `.git/` and caches → only another review document). The file's `classification: OFFICIAL_ASSET_UNAVAILABLE` and `search_locations_checked` never reach preflight. Same structural note as the `conesep` review's MF-3.
- **M6 (P3) — `environment` block entirely absent for `ptha_mtst`,** while every other recent model records one (`HABIT` at `models.yaml:~217`, etc.). Consequence observed in `doctor`: `environment:ptha_mtst OK (UNCONFIGURED, reason: "no environment variable declared")` — reported **OK** because it is not *blocked*, which reads as a pass for a model with no environment at all.
- **M7 (P3) — `cluster_b5_ptha_mtst.yaml` records `expected_format: pt` for a checkpoint that does not exist,** alongside `classification: OFFICIAL_ASSET_UNAVAILABLE`. Harmless, but it is a fabricated detail about a non-existent artifact.

## UNRESOLVED EVIDENCE
- **U1 — Whether an upstream revision ever contained `blip2_qformer_tgir_concatembedding_sim_zyn.py`.** I inspected the current tree only (433 entries via API, `truncated: false`). `git log`/history fetch is a network mutation I did not perform. If that file existed at some commit, a pin could dodge C2. UNKNOWN.
- **U2 — Whether `tuned_mtst_cirr.pt` (sha256 `1c4dab86…`, 2720352973 B) contains anything FashionIQ-related.** Not loaded. Its name, the paper's method structure (CIRR MTST pretraining as a *distinct* stage, §V-B-1), and the two CIRR-only companion JSONs all say CIRR. UNKNOWN as to payload; the naming and companions make a FashionIQ weight unlikely, but nothing was executed.
- **U3 — Whether the paper's 54.99/76.28/65.64 row is reproducible from any public artifact.** Evidence says NO under current public trees (no FashionIQ weight; no FashionIQ MTST corpus; no `FashionIQGen` loader). Absence over all possible venues (deleted Hub revisions, private drives, author email) is not provable. Recorded as **UNKNOWN-BUT-EVIDENCE-AGAINST**.
- **U4 — Hub history / deleted uploads for `yinanzhou1`.** Only current repo lists were queried (`api/models?author=yinanzhou1`, `api/models?search=MTST`, `api/models?search=ptha`). No revision-history walk, no discussions beyond the empty list. UNKNOWN.
- **U5 — The exact revision the paper's numbers were produced at.** The repository was pushed last `2025-02-09T10:52:31Z`, and the arXiv v1 is `2025-02-21`; no tag or release is published (`/releases` not queried; the tree contains no tag-shaped artifact). UNKNOWN.
- **U6 — Whether `cap_train_{cat}_rev.json` (`data_utils.py:157`) has any published source.** The file is required for `split == 'train'` FashionIQ runs, is line-delimited JSON in a different shape from `cap.{cat}.train.json`, and is not in the README layout and not in the local dataset (`ls data/fashionIQ_dataset/captions/` → 9 `cap.*.json` files only). UNKNOWN.

---

# EXECUTION TRACE
`paper/README/config → source dataset → preprocessing → official evaluator → workbench invocation → checkpoint → output → metric comparison`

1. **paper/README/config →** Paper `arXiv:2504.05316` (v1, 2025-02-21) Table III row `PTHA (Ours) BLIP2 FashionIQ_MTST` = `50.77 73.78 55.91 75.36 58.28 79.70 54.99 76.28 65.64` (OBSERVED network fetch of the HTML v1). Paper §V-A-2 (Implementation Details) states PTHA's architecture is "the same as SPRC", vision encoder "frozen ViT-g/14 from EVA-CLIP", multimodal/text encoders "BLIP-2", batch 128, 32 query tokens, initial pretraining weights from BLIP2-pretrained. Upstream `zyinan99/MTST_PTHA@main` README documents `conda … python=3.9`, `torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2`, then `pip install -r requirements.txt` (`transformers==4.36.2`, `timm==0.9.12`, `spacy==3.7.2`, …). **Registry records none of this** (`environment` block absent). Evidence: paper HTML §V-A-2, README, `requirements.txt`. **ARROW: paper-only; workbench has no config for this model.**
2. **→ source dataset** — Upstream `data_utils.py:15` `base_path = Path("path_to_data_dir")` (a literal the operator must edit; README: "change `base_path = Path(\"path_to_data_dir\")`"). Required: `fashionIQ_dataset/captions/cap.{cat}.val.json` (`:153`), `image_splits/split.{cat}.val.json` (`:167`), `images/{id}.png` (`:170`, `:206`). Local: `data/fashionIQ_dataset/` has `captions/ image_splits/ images/ qwen_captions/`, 74381 `.png` images, val splits 3817/6346/5373 with 0 missing images and 0 missing references/targets. **ARROW: source paths and local layout agree on the evaluation-side reads; the workbench never wired this because no layout is declared (`fashioniq_required_paths → ()`, executed).**
3. **→ preprocessing** — `blip_validate.py:53` `targetpad_transform(1.25, 224)`; `data_utils.py:100-106` the exact `Compose`. Deterministic. Applied at `data_utils.py:187`/`:192` (val) and `:206` (gallery). **ARROW: fully traceable STATIC_AUDIT; unregistered (P3).**
4. **→ official evaluator** — `blip_validate.py:16` `from lavis.models import load_model_and_preprocess` → **fails** at `lavis/models/__init__.py:54` (imports an absent module; 404 and absent from a 361-blob tree). Even past that: `:186-200` dispatch `--dataset fashioniq` → `clip_finetune_fiq(['dress','toptee','shirt'], …)`; `:45` loads the model by registry name with `model_type=backbone`; `:48` `torch.load(checkpoint_path)`; `:49` `blip_model.load_state_dict(checkpoint[blip_model.__class__.__name__], strict=False)`. **ARROW: BLOCKED at the import.**
5. **→ workbench invocation** — `ADAPTERS.get('ptha_mtst')` → `None` (executed; `ADAPTERS` has 15 keys). `preflight_blockers` → `['upstream source unavailable', 'command not runnable: COMMAND_UNAUDITED']`. `doctor` `command:ptha_mtst` `BLOCKED`, `script: null`. No command is ever emitted. **ARROW: BLOCKED — no adapter, no invocation.**
6. **→ checkpoint** — Registry `checkpoint_variants: []` → `select_checkpoints` → `BLOCKED_CHECKPOINT` / `"no checkpoint variant is declared in the registry"` (executed). Public hunt: `yinanzhou1/mtst_ptha` holds `tuned_mtst_cirr.pt` (2730352973 B → wait, size `2720352973`, sha256 `1c4dab86a9f66dfe10aff75b39790432eb6cc746cf27b6cfab74b66d93b7588f`) and two CIRR JSONs; no FashionIQ weight at any probed path (5×404); no weight blob in the GitHub tree. `workbench/artifacts/checkpoints/` holds only `.gitkeep`. **ARROW: BLOCKED — nothing to evaluate.**
7. **→ output** — none. `reproduction_status: BLOCKED_NO_VERIFIED_FASHIONIQ_CHECKPOINT` (`models.yaml:212`); no `*ptha*` file under `workbench/artifacts/results/`. **ARROW: NO OUTPUT.**
8. **→ metric comparison** — not reached. No parser exists (`metrics_extraction.py` defines only `parse_limn_output:57` and `parse_csmcir_output:200`), so even a hypothetically successful run's `{cat}_recall_at10`/`average_recall*` JSON (`blip_validate.py:90-98`) would have no binding into `paper_gate.evaluate_gate`. **PARITY_NOT_EVALUATED.**

Arrows 4-8 are BLOCKED. Nothing below arrow 3 is RUNTIME_VERIFIED. Every claim in this audit is STATIC_AUDIT.

---

# CHECKLIST

*Note, recorded explicitly rather than silently:* the assignment file and the shared context both say "all 55 questions" but neither contains the list, and the list is not present anywhere under `/home/heheboiz/data/cir` (searched `*.md`/`*.py`/`*.yaml` for `55 questions`, `55-item`, `CHECKLIST`, `N. <question`). The same gap was recorded independently in `workbench/docs/reviews/models/conesep.md:177`. The 55 probes below are therefore **mine**, one per checkable fact across registry, source, checkpoint, protocol, dataset, preprocessing, augmentation, gallery, metric, parity, runtime and verdict. Each item: `N. question — verdict — evidence — severity — action`.

1. **Registry entry exists and is unique** — VERIFIED — `models.yaml:195` `model_id: ptha_mtst`; executed `load_registry()['models']` → 16 entries, one match — NONE — none.
2. **`method_name` matches the paper's own vocabulary** — VERIFIED — `models.yaml:196` `PTHA + MTST`; paper §I names "Modification Text-oriented Synthetic Triplets (MTST)" and "Prototypical Two-Hop Alignment (PTHA)" — NONE — none.
3. **`paper_title` matches the paper** — VERIFIED — `models.yaml:200` vs arXiv 2504.05316 title (OBSERVED fetch) — NONE — none.
4. **`paper_url` resolves and identifies the same work** — VERIFIED — `read https://arxiv.org/abs/2504.05316` → 200, identical title/authors, Submitted 2025-02-21 — NONE — none.
5. **`publication_year: 2025` is consistent** — VERIFIED — `models.yaml:198`; arXiv submission 2025-02-21 (OBSERVED) — NONE — none.
6. **`research_generation: recent` is consistent** — VERIFIED — `models.yaml:197`; `registry.py:66-67` allows only `legacy`/`recent`; 2025 ≥ 2020 — NONE — none.
7. **`upstream_repo_url` recorded** — **REFUTED (VARIANT)** — `models.yaml:202` is `null`; an official repository exists: `https://github.com/zyinan99/MTST_PTHA` (API: `created_at 2024-12-26`, `pushed_at 2025-02-09`, MIT, commits by `Yinan Zhou`) — P1 — add the URL and a pin.
8. **`source_available: false` is accurate** — **REFUTED (VARIANT)** — `models.yaml:205`; a public repository exists and its source files fetch over raw.githubusercontent (5×200) — P1 — set `true` once pinned.
9. **`source_dir: null` is accurate** — VERIFIED — `models.yaml:206`; `ls workbench/third_party/` → 14 dirs, none matching — NONE — none.
10. **The registry's source search was adequate** — **VARIANT** — `cluster_b5_ptha_mtst.yaml:60-62` `search_locations_checked` lists only two workbench-internal locations; a GitHub repository search (`api.github.com/search/repositories?q=ptha+mtst`) returns the author repo as the single hit — P1 — record web/Hub searches as part of the negative evidence.
11. **`upstream_commit_sha: null` is accurate** — VERIFIED — `models.yaml:204`; nothing is pinned — NONE — none.
12. **A pinned checkout exists and matches the registry** — NOT_RUN — no checkout to run `git rev-parse HEAD` against; `source_provenance` → `expected_commit None`, `state unavailable` — P1 — pin `6a02ee47…` (or a chosen revision) and re-audit.
13. **Upstream HEAD is known** — VERIFIED — `GET .../branches/main` → `6a02ee4763fc2260b6abfb18be78853facb91d86` (STATIC_AUDIT, remote metadata) — NONE — none.
14. **Upstream tree is complete as fetched** — VERIFIED — `GET .../git/trees/main?recursive=1` → `truncated: false`, 433 entries, 361 blobs — NONE — none.
15. **Upstream license recorded** — VARIANT — GitHub API `license MIT`; `models.yaml` has no license field for any model; registry schema has no such key — P3 — none (out of schema).
16. **Upstream environment recorded in the registry** — VARIANT — README pins `python 3.9 / torch 2.0.1 / torchvision 0.15.2`; `requirements.txt` adds `transformers==4.36.2, timm==0.9.12, spacy==3.7.2, opencv-python, omegaconf==2.3.0, decord==0.6.0, webdataset==0.2.86, fairscale==0.4.13, einops==0.7.0, scikit-image==0.22.0, pycocoevalcap==1.2, tqdm==4.66.1, iopath==0.1.10`; **no `environment` block exists in the registry entry** — P2 — add one (every other recent model has one).
17. **`preparation_contract` is a legal registered value** — VERIFIED — `models.yaml:199` `unavailable` is in the allow-list at `registry.py:73` — NONE — none.
18. **The `unavailable` contract leaves no silent data hole** — **VARIANT** — executed `model_requirements(ptha_mtst)` → `blocks: []`, `paths: []`, because `fashioniq_required_paths` returns `()` for `fashioniq_layout is None` (`registry.py:679-683`); `preflight_blockers` therefore reports **no** data blocker — P2 — the empty contract must be paired with an explicit "source unavailable" data gate so a future pin cannot run unverified against the raw dataset.
19. **`native_protocol` is recorded** — **VARIANT** — `models.yaml:207` is `null`; the upstream evaluator's FashionIQ gallery is the full val split with the reference eligible (`validate_blip.py:44-51` has no reference mask), i.e. protocol `fashioniq_original_split` (`src/evaluation/fashioniq.py:160-162`) — P1 — record it after pinning.
20. **`supported_protocols` is recorded** — VARIANT — `models.yaml:208` is `[]`; matches `native_protocol` being unset, so internally consistent, but empty — P1 — same action as 19.
21. **`literature_split_label` recorded** — VARIANT — absent from the entry (`:195-215`); the paper says only "the val set" (§V-A-1) and the source evaluates the standard val split, so the label is `original`, the same as the four sibling `fashioniq_original_split` models — P3 — add it with 19.
22. **`reported_r10` matches the paper** — VERIFIED — `models.yaml:209` `54.99`; paper Table III average `R@10 = 54.99` (OBSERVED fetch) — NONE — none.
23. **`reported_r50` matches the paper** — VERIFIED — `models.yaml:210` `76.28`; paper Table III average `R@50 = 76.28` — NONE — none.
24. **`reported_mean` matches the paper** — VERIFIED — `models.yaml:211` `65.64`; paper Table III `Avg. 65.64` — NONE — none.
25. **`reported_mean` equals `(reported_r10+reported_r50)/2`** — VERIFIED — executed `(54.99+76.28)/2 = 65.635 → 65.64`; paper's stated `Avg.` definition and source's `blip_validate.py:95` agree — NONE — none.
26. **The reported row is the row the registry believes it is** — VERIFIED — paper Table III: the MTST row is `54.99 76.28 65.64`; the non-MTST PTHA row is `54.33 75.24 64.79` and the SPRC row `54.92 74.97 64.85`; registry values match only the MTST row — NONE — none.
27. **The reported row's training condition is recorded** — VARIANT — the row is `PTHA (Ours) BLIP2 FashionIQ_MTST`, i.e. it *requires* the 579,114-triplet FashionIQ MTST corpus; the registry records no row-condition field — P1 — add a `paper_row_condition` (or equivalent) because this row is the one whose pretraining data is unpublished (C3).
28. **`reproduction_status` is a legal and accurate value** — VERIFIED — `models.yaml:212` `BLOCKED_NO_VERIFIED_FASHIONIQ_CHECKPOINT`; the checkpoint half is confirmed (1 Hub repo, no FashionIQ weight, 5×404 probes) — NONE — none.
29. **`reproduction_status` names the *primary* blocker** — VARIANT — the primary blocker is now (a) no published FashionIQ checkpoint **and** (b) no published FashionIQ MTST corpus **and** (c) the reachable upstream cannot import (`__init__.py:54`); the status names only (a) — P2 — the label is not wrong, it is incomplete; the source asymmetry is exactly what C1 is about.
30. **`strict_clean_checkpoint_available: false`** — VERIFIED — `models.yaml:213`; no checkpoint is published at all, so none is a clean one — NONE — none.
31. **The `notes` line is accurate** — VERIFIED — `models.yaml:214` "No author-linked FashionIQ fine-tuned checkpoint verified. CIRR pretraining evidence is not eligible."; confirmed by the Hub file set (`tuned_mtst_cirr.pt` + two CIRR JSONs, no FashionIQ file) — NONE — none.
32. **`checkpoint_variants: []` and no bundles** — VERIFIED — `models.yaml:215`; executed `len(variants)==0`, `len(bundles)==0`; `select_checkpoints` → `BLOCKED_CHECKPOINT` — NONE — none.
33. **A public author checkpoint repo exists and was checked** — VERIFIED — `yinanzhou1/mtst_ptha`, `sha b5e669c8ef28b0084f43398d184a5a0116903ccf`, 5 siblings, `lastModified 2024-12-26T07:42:49Z` (STATIC_AUDIT metadata) — NONE — none.
34. **A FashionIQ fine-tuned checkpoint is published** — VERIFIED-NEGATIVE — no such file: README links only CIRR artifacts; `curl -I` on five candidate names → 404; zero `.pt/.pth/.bin/.ckpt` blobs in the GitHub tree — NONE (registry correct) — none.
35. **An eligible FashionIQ MTST *corpus* is published** — VERIFIED-NEGATIVE — Hub holds `cirr_modifier_generated_5p_completed_rev.json` and `cirr_train_modifier_rev_long.json` only; README's `FashionIQGen` dataset has no class in `data_utils.py` and no accepted CLI value (`blip_fine_tune_2.py:400`) — P1 — record the corpus gap as its own blocker (C3).
36. **Checkpoint digests/sizes recorded where published** — VARIANT — registry has no variants; where a published file does exist the Hub exposes `tuned_mtst_cirr.pt` size `2720352973` and LFS sha256 `1c4dab86a9f66dfe10aff75b39790432eb6cc746cf27b6cfab74b66d93b7588f` — P3 — if the CIRR weight is ever registered, record size+digest (contrast the `conesep` P2 hole).
37. **Official evaluator entrypoint identified** — VERIFIED (external) — `MTST_PTHA/src/blip_validate.py:186-200`; no `__main__` exists in `validate_blip.py` (library only), so `blip_validate.py` is the evaluator — NONE — none.
38. **Official evaluator is runnable as published** — **VERIFIED-NEGATIVE (BLOCKED)** — `lavis/models/__init__.py:54` imports an absent module; `curl` → 404; my tree check found exactly 1 missing target among 48 `from lavis.models.*` imports; no guard — P0 — upstream defect; must be reported as a source blocker, not patched silently.
39. **The missing-module count/line is exact** — VERIFIED — my script output: `total import lines matched: 48`, `MISSING: 1 — line 54 lavis.models.blip2_models.blip2_qformer_tgir_concatembedding_sim_zyn`, over a 296-line file — P0 — none (correctly reported).
40. **A sibling module under a near-identical name exists (name confusion risk)** — VERIFIED — `blip2_qformer_tgir_concatembedding_sim.py` → 200, registers `blip2_tgir_ce_sim` (its line 38); the `_zyn` variant does not exist — P3 — note the near-collision so a future fix is not "rename the import".
41. **FashionIQ evaluation dispatch path traced** — VERIFIED — `blip_validate.py:199-200` → `clip_finetune_fiq(['dress','toptee','shirt'], …)`, defined `:26` — NONE — none.
42. **Gallery construction matches the protocol candidate** — VERIFIED — `data_utils.py:165-168` full split file order per category; `utils.py:54-55` DataLoader without `shuffle`; `utils.py:76` `index_names.extend(names)` — NONE — none.
43. **Reference exclusion policy matches `fashioniq_original_split`** — VERIFIED — `validate_blip.py:44-51` builds `labels` against `target_names` only, with **no** `reference_mask`; the CIRR path at `:258-260`/`:331-333` *does* mask, proving the absence in the FIQ path is deliberate — NONE — none.
44. **Gallery size is well-defined on the local dataset** — VERIFIED — executed: dress 3817, shirt 6346, toptee 5373; 0 ids dropped by the `.png` existence filter; all 15415 ids unique within their category — NONE — none.
45. **Every reference and target is inside its gallery** — VERIFIED — executed `refs missing in gallery: 0`, `targets missing: 0` for all three; `ref==target count: 0`; the `assert` at `validate_blip.py:51` (exactly one hit per query) would therefore hold — NONE — none.
46. **Gallery membership is not silently data-dependent** — **VARIANT** — `data_utils.py:169-172` removes any id lacking `images/{id}.png`, with no count and no warning; 0 occurrences locally, but on a partial mirror the gallery and the Recall@K denominator change silently — P3 — report the dropped count.
47. **Recall@K maths correct** — VERIFIED — `validate_blip.py:44` `1 - pred_sim`, `:45` `argsort`, `:46` name lookup, `:49-50` label comparison, `:51` one-hit assert, `:54-55` `sum(labels[:, :K])/len(labels)*100` — NONE — none.
48. **Macro/mean definition matches the paper** — VERIFIED — `blip_validate.py:93-95` unweighted category means, `average_recall = (mean r50 + mean r10)/2`; paper Table III caption states `(Recall@10 + Recall@50)/2` — NONE — none.
49. **K values used are 10 and 50** — VERIFIED — `validate_blip.py:54-55`; matches `reported_r10`/`reported_r50` — NONE — none.
50. **Preprocessing recipe is deterministic and traceable** — VERIFIED — `blip_validate.py:52-53` `input_dim=224`, `targetpad_transform(1.25, …)`; `data_utils.py:100-106` the exact `Compose` with CLIP normalisation — NONE — none.
51. **Evaluation is augmentation-free** — VERIFIED — eval branches apply only `self.preprocess` (`data_utils.py:187,192,206`); the randomized caption helper `generate_randomized_fiq_caption` has no callsite in `blip_validate.py` — NONE — none.
52. **Workbench adapter produces a command** — VERIFIED-NEGATIVE (BLOCKED) — `ADAPTERS.get('ptha_mtst')` → `None` (executed, 15 keys); `doctor command:ptha_mtst` `BLOCKED`, `script: null` — P1 — implement only after a pin and C2 are resolved.
53. **Workbench preflight reports the right blockers and no false ones** — VARIANT — executed `preflight_blockers` → `['upstream source unavailable', 'command not runnable: COMMAND_UNAUDITED']`; both true, but **no data blocker** is raised (see 18) and the command status string is the misleading fallback `COMMAND_AUDITED` (see 54) — P2 — fix the label and add a data gate.
54. **Blocker reporting is not misleading** — **VARIANT** — `workbench/docs/READINESS_MATRIX.md:28` prints `evaluator command: COMMAND_AUDITED`; traced to `readiness_matrix.py:114` printing `command['detail']`, which `doctor.py:169` fills with `command_status or "COMMAND_AUDITED"` while `command_implemented` is `False` — P2 — print the status or an explicit `COMMAND_UNAUDITED`.
55. **Evaluator reaches the model's protocol / a run output exists and parity was compared** — BLOCKED — `reproduction_status: BLOCKED_NO_VERIFIED_FASHIONIQ_CHECKPOINT` (`models.yaml:212`); no adapter, no checkpoint, no `*ptha*` result artifact, no parser in `metrics_extraction.py`; `SOURCE_TO_WORKBENCH_PARITY_MATRIX.md:35` cells are `NOT_APPLICABLE`/`SOURCE_EVIDENCE_MISSING` — P1 — no `ptha_mtst` number can be produced today, and the registry's "no source" premise for that conclusion is itself false.

---

**FINAL VERDICT: BLOCKED** — the model's *outcome* is right for the wrong reason: no registered checkpoint is reachable and no FashionIQ MTST corpus is published, so no number can be produced (independently confirmed in every direction I probed), but the registry's stated primary blocker — "no source" — is falsified by an official author repository (`zyinan99/MTST_PTHA`, reachable, MIT, authored by the paper's first author) whose published package cannot import anyway (`lavis/models/__init__.py:54` references an absent module), and the paper metrics 54.99 / 76.28 / 65.64 are a correct transcription of Table III verified at arXiv 2504.05316v1.
