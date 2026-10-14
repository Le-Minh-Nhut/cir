# ptha_mtst

## MODEL ID
`ptha_mtst` — registry block `workbench/registry/models.yaml:213-238` (`- model_id: ptha_mtst` at `:213`, `method_name: PTHA + MTST` at `:214`). Unique: executed `.venv/bin/python -c "from workbench.backend.registry import load_registry; ..."` → 16 models, 1 match. STATIC_AUDIT.

## PAPER TITLE
`Scale Up Composed Image Retrieval Learning via Modification Text Generation` (`models.yaml:218`), `paper_url: https://arxiv.org/abs/2504.05316` (`:219`).

Independently re-derived (STATIC_AUDIT, network OBSERVED):
- `curl -sL https://arxiv.org/pdf/2504.05316v1` → HTTP 200, 9095049 bytes, `sha256 0801d37af9471a391c0007a9ea81611c5f6459bd04619801c5ed7193f486af2a`, `pdfinfo` → `Pages: 12`. This matches the sha/size recorded in the dossier.
- `curl -sL https://arxiv.org/abs/2504.05316` → `citation_title` identical; authors `Zhou, Yinan; Wang, Yaxiong; Lin, Haokun; Ma, Chen; Zhu, Li; Zheng, Zhedong`; `citation_date 2025/02/21`; submission history has exactly one entry `[v1] Fri, 21 Feb 2025`.
- Paper full text contains **no code URL**: `'github' in <v1 HTML plain text>` → `False`; `grep -c github` on the PDF text → 0. The repository the registry now records was *not* discoverable from the paper.
- `readiness_matrix.py` executed for the model returns `reported_r10: 54.99, reported_r50: 76.28` (job `bg_1199` output).

## PINNED SOURCE COMMIT
**No pin exists. Nothing is checkoutable, so nothing can be line-audited against a pin.**

- `models.yaml:224` `upstream_repo_url: https://github.com/zyinan99/MTST_PTHA`; `:225` `upstream_default_branch: main`; `:226` `upstream_commit_sha: null`; `:227` `source_available: false`; `:228` `source_dir: null`; `:229` `source_blocker:` (long text).
- `ls workbench/third_party/` → `AirKnow CLIP4Cir CLVCNet CSMCIR ConeSep DCNet ENCODER HABIT HINT INTENT LIMN PAIR SPRC TGCIR` (14). No `MTST_PTHA`. Registry ↔ filesystem consistent.
- `preflight_blockers(model, resolve_config())` (executed) → `['source cannot run: author repository https://github.com/zyinan99/MTST_PTHA exists and is reachable (README title matches the paper, commits authored by the paper's first author Yinan Zhou, HEAD 6a02ee4763fc2260b6abfb18be78853facb91d86 at 2025-02-09)…', 'upstream source unavailable', 'command not runnable: COMMAND_UNAUDITED']` — matches `preflight.py:162,177-178,197-206`.

Remote metadata I re-derived against upstream `main` (not a pin; STATIC_AUDIT, OBSERVED):
- `curl https://api.github.com/repos/zyinan99/MTST_PTHA` → `full_name zyinan99/MTST_PTHA`, `default_branch main`, `created_at 2024-12-26T06:16:27Z`, `pushed_at 2025-02-09T10:52:31Z`, `size 7906`, `license spdx_id MIT`, `fork false`, `archived false`, `visibility public`.
- `.../branches/main` → `6a02ee4763fc2260b6abfb18be78853facb91d86`, author `Yinan Zhou <1359149292@qq.com>`, date `2025-02-09T10:52:27Z` — identical to the sha/date in `source_blocker`.
- `.../git/trees/main?recursive=1` → `truncated: false`, 433 entries, 361 blobs; tops `.gitattributes .gitignore LICENSE MTST_PTHA README.md requirements.txt`; **zero** `.pt/.pth/.bin/.ckpt` blobs.
- Blob binding of the files I actually read: `MTST_PTHA/src/blip_validate.py` = `fa5098f7eb8778f3b7fbb2535f5987fd6bf5581b` (my `git hash-object` of the downloaded bytes matched), `lavis/models/__init__.py` = `8027b1e7d018b7b3c5e9de9f691d753bb31c3e20` (matched), `data_utils.py` = `d129d83eb99a17c24572f1c73a1d82753860ed54`, `utils.py` = `a46c5da78efd6b15a524c6531c0e554471db7860`, `validate_blip.py` = `57065aaae1ad89910888e6fd88d8d9f8ba4f752d`, `blip_fine_tune_2.py` = `fbd805ea4a0773a1bfe5260e2915523cf7c3820c`. Every line number below is relative to those blob contents.
- `.../releases` → `[]`; `.../tags` → `[]`. So HEAD is a moving target with no revision marker.

## CHECKPOINT IDENTITY
- `models.yaml:238` `checkpoint_variants: []`; `:236` `strict_clean_checkpoint_available: false`; `:235` `reproduction_status: BLOCKED_NO_VERIFIED_FASHIONIQ_CHECKPOINT`; `:237` notes.
- Executed `select_checkpoints(m, policy='strict_clean')` → `status='BLOCKED', reason_code='BLOCKED_CHECKPOINT', reason='no checkpoint variant is declared in the registry'`.
- Public hunt re-derived: `https://huggingface.co/api/models/yinanzhou1/mtst_ptha/tree/main?recursive=true&expand=true` → exactly 5 files: `.gitattributes` 1754 B; `README.md` 31 B; `cirr_modifier_generated_5p_completed_rev.json` 298419312 B `lfs oid 6cb4c02170f703da5ae40ae6d988a6aae272c2fa821d30c1d090de671ddfbd4d`; `cirr_train_modifier_rev_long.json` 15591788 B `b1b7e72b3416e4479e4ddb9a807e789210e8d4f9d32210da460a22882f90d15d`; `tuned_mtst_cirr.pt` **2720352973 B** `1c4dab86a9f66dfe10aff75b39790432eb6cc746cf27b6cfab74b66d93b7588f`. Repo `sha b5e669c8ef28b0084f43398d184a5a0116903ccf`, `lastModified 2024-12-26T07:42:49Z`, `usedStorage 3034364073`, `gated false`.
- Five candidate FashionIQ filenames probed with `curl -I` → **404** for all: `fashioniq.pt`, `fashioniq_mtst.pt`, `ptha_fashioniq.pt`, `tuned_mtst_fashioniq.pt`, `mtst_fashioniq`.
- `api/models?author=yinanzhou1&limit=50` → 6 repos, one CIR-relevant (`yinanzhou1/mtst_ptha`). `api/models?search=mtst` → the same repo; nothing else CIR.
- Upstream README (`README.md:82-85`) links exactly the three CIRR artifacts above — **no FashionIQ weight and no FashionIQ corpus link** anywhere in the README.

**Verdict: registry checkpoint claims VERIFIED** (`strict_clean_checkpoint_available: false`, `checkpoint_variants: []`, notes). STATIC_AUDIT — no byte was downloaded and no weight was loaded.

## EVALUATION PROTOCOL
Registry: `native_protocol: null` (`:230`), `supported_protocols: []` (`:231`). Nothing declared for the model.

Derived from the source blobs (STATIC_AUDIT, un-pinned HEAD):
- Dispatch: `MTST_PTHA/src/blip_validate.py:186-200`; `:194-195` accepts only `fashioniq`/`cirr`; `:199-200` `elif args.dataset.lower() == 'fashioniq': clip_finetune_fiq(['dress','toptee','shirt'], args.blip_model_name, args.backbone, args.model_path)`.
- Per category: `blip_validate.py:64` `FashionIQDataset('val', [dress_type], 'relative', preprocess)`; `:66` `'classic'`; `:80` `extract_index_blip_features(classic_val_dataset, blip_model)`; `:81-82` `compute_fiq_val_metrics(...)`.
- Gallery + reference policy: `validate_blip.py:38-55` builds `labels` against `target_names` only and contains **no reference mask**, while the CIRR path at `:257-261` explicitly masks (`reference_mask = torch.tensor(sorted_index_names != np.repeat(np.array(reference_names), …))`). The absence in the FashionIQ path is deliberate, not an oversight.
- That is exactly the workbench protocol `fashioniq_original_split`: `src/evaluation/fashioniq.py:172-174` `protocol_id="fashioniq_original_split", gallery="full_split", reference_excluded=False`.

**Verdict: the correct registered value is `native_protocol: fashioniq_original_split` / `supported_protocols: [fashioniq_original_split]` (`literature_split_label: original`, `registry.py:48-51`); the registry has `null`/`[]` — VARIANT, P1.** INFERENCE from an un-pinned upstream, and it cannot be recorded without a pin because `registry.py:42-47` validates protocol ids against the registry's own protocol table (executed `load_registry()['protocols']` → three ids, `fashioniq_original_split` present).

## DATASET CONTRACT
- Registry: `preparation_contract: unavailable` (`:217`, legal per `registry.py:73` allow-list); `fashioniq_layout` absent; `preparation_id` absent.
- Consequence, executed: `fashioniq_required_paths(m, Path('data'))` → `()`, because `registry.py:692-696` returns `()` when `fashioniq_layout is None`; `preflight.py:28` then falls back to that empty tuple (`contract` is also `None`). So `model_requirements` contributes **no paths and no data blocker** — the executed `preflight_blockers` list has no data entry.
- What the source actually opens (STATIC_AUDIT, `MTST_PTHA/src/data_utils.py`, `base_path = Path("path_to_data_dir")` at `:15`, a literal the README tells the operator to edit): `:153` `<base>/fashionIQ_dataset/captions/cap.{cat}.val.json`; `:167` `<base>/fashionIQ_dataset/image_splits/split.{cat}.val.json`; `:170`/`:206` `<base>/fashionIQ_dataset/images/{id}.png` (**`.png` hardcoded**); `:157` `<base>/fashionIQ_dataset/captions/cap_train_{cat}_rev.json` — line-delimited JSON, train only, absent from the README layout.
- Name join on the val path: `:180`/`:188` (train) apply `.replace("jpg","png")`, but the val branch `:196` returns `self.triplets[index]['target']` and `:180`'s `reference_name` **without** the replace, while the gallery key (`:170`) appends `.png` to the raw split id.
- Local dataset, OBSERVED: `data/fashionIQ_dataset/` = `captions/ image_splits/ images/ qwen_captions/`; `images/` 74381 files, `{'.png': 74381}`; val splits dress 3817 / shirt 6346 / toptee 5373, each internally unique (3817/6346/5373 exact), 0 ids dropped by the `.png` existence filter; every reference and every target present in its gallery (`refs_missing 0`, `targets_missing 0`, `ref==target 0` per category). Both source reads and the local mirror agree on `.png`.

**Verdict: VARIANT, P2.** No declared contract means no preflight, and the eval-side reads happen to match the local mirror — but nothing in the workbench asserts it, so a future operator who pins the source gets **no data gate at all**. Also the target name the val path compares against still carries the FashionIQ `.jpg` extension in the official caption files; local data has no `.jpg` string in the split ids, so the join is exact only because this mirror was derived from a `.png`-named release.

## PREPROCESSING CONTRACT
- Registry: nothing. `papers.yaml:1678` `preprocessing_id: null`.
- Source: `blip_validate.py:52-53` `input_dim = 224`; `preprocess = targetpad_transform(1.25, input_dim)`; `data_utils.py:94-108` the factory → `Compose([TargetPad(target_ratio, dim), Resize(dim, interpolation=PIL.Image.BICUBIC), CenterCrop(dim), _convert_image_to_rgb, ToTensor, Normalize((0.48145466,0.4578275,0.40821073),(0.26862954,0.26130258,0.27577711))])`; `TargetPad.__call__` at `:66-75`. Applied on the eval path at `data_utils.py:192`/`:201` (val relative/test) and `:207` (classic/gallery).

**Verdict: recipe is deterministic and source-exact; unregistered — VARIANT, P3.**

## AUGMENTATION CONTRACT
- Registry: nothing. `papers.yaml:1679` `augmentation_policy: null`.
- Eval path is augmentation-free: the val/classic branches apply only `self.preprocess` (`data_utils.py:192, 201, 207`); `generate_randomized_fiq_caption` is imported at `blip_validate.py:22` but has **zero callsites** there, and its only caller is `blip_fine_tune_2.py:124` (import at `:21`).

**Verdict: VERIFIED (no augmentation at evaluation), STATIC_AUDIT, NONE.** Registry silence accurate but unrecorded (P3).

## GALLERY CONTRACT
Source (STATIC_AUDIT):
- `data_utils.py:165-168`: `for dress_type in dress_types: self.image_names.extend(json.load(open(<base>/fashionIQ_dataset/image_splits/split.{dress_type}.{split}.json)))` — appended in **file order**, one category per dataset instance (`blip_validate.py:66` passes `[dress_type]`).
- `data_utils.py:169-172`: ids whose `images/{name}.png` is missing are **removed silently**, no count and no warning.
- Enumeration: `utils.py:54-55` `DataLoader(dataset, batch_size=64, num_workers=2, pin_memory=True, collate_fn=collate_fn)` — no `shuffle`, so file order is preserved; `utils.py:76` `index_names.extend(names)`; `:70` `extract_target_features(images, mode="mean")`; `:80` returns `(index_features, index_features_raw), index_names`.
- Consumption: `validate_blip.py:169` `name_to_feat = dict(zip(index_names, index_features[-1]))` — a dict, so a duplicate id would collapse while the list keeps both; references are also used as the query-side image (`:190-193`).
- Local, OBSERVED: gallery = the full val split per category (3817/6346/5373), 0 filtered, all ids unique within category → the dict collapses nothing here.
- `papers.yaml:1675-1676` `gallery_protocol: fashioniq_original_split`, `reference_excluded: false` — agrees with the source.

**Verdict: source gallery VERIFIED (full split, file order, reference eligible); the model registry entry carries no gallery/protocol field at all — VARIANT, P2.**

## METRIC CONTRACT
Source (STATIC_AUDIT): `validate_blip.py:44` `distances = 1 - pred_sim`; `:45` `torch.argsort(distances, dim=-1)`; `:46` `sorted_index_names = np.array(index_names)[sorted_indices]`; `:49-50` `labels = sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(...)`; `:51` `assert torch.equal(torch.sum(labels,dim=-1).int(), torch.ones(len(target_names)).int())`; `:54-55` `recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100`, likewise `[:, :50]`.
Macro: `blip_validate.py:88-96` per-category keys then `average_recall_at10 = mean(recalls_at10)`, `average_recall_at50 = mean(recalls_at50)`, `average_recall = (mean(recalls_at50) + mean(recalls_at10)) / 2`; printed at `:98`.
Arithmetic re-derived: `(50.77+55.91+58.28)/3 = 54.9867 → 54.99`; `(73.78+75.36+79.70)/3 = 76.28`; `(54.99+76.28)/2 = 65.635 → 65.64`. `papers.yaml:1678-1698` records definition/aggregation/categories identically.
No parser exists for this output shape: `grep -n "^def parse_" workbench/backend/metrics_extraction.py` → `parse_limn_output` (:74) and `parse_csmcir_output` (:219) only; the dispatch at `:328-341` falls through to `extraction_status="AGGREGATE_PARSER_UNAVAILABLE", parity_status="PARSER_UNAVAILABLE"`.

**Verdict: source metric maths VERIFIED; registry entry declares no metric contract and no parser binds `{cat}_recall_at10`/`average_recall*` — VARIANT, P2.**

## SOURCE-TO-WORKBENCH PARITY
`workbench/docs/SOURCE_TO_WORKBENCH_PARITY_MATRIX.md:35`: `| ptha_mtst | no source | NOT_APPLICABLE | SOURCE_EVIDENCE_MISSING | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | SOURCE_EVIDENCE_MISSING |`. The cells are honest *for the workbench* (no checkout → nothing to compare) but `no source` / `SOURCE_EVIDENCE_MISSING` is now contradicted by `models.yaml:224,229` of the same working tree, and `workbench/docs/UPSTREAM_DATA_PREPARATION_AUDIT.md:19` ("`ptha_mtst` has no source and is `SOURCE_UNAVAILABLE`"), `:38` (`| ptha_mtst | — | none |`) and `:184-186` ("### ptha_mtst — no source / `SOURCE_UNAVAILABLE`. No repository, no checkpoint mapping, no evaluator. Not auditable.") assert no repository exists.
**Verdict: VARIANT, P1** — the parity cells should read `NOT_PINNED_IN_WORKBENCH`, and the four doc strings above are falsified by the registry's own `upstream_repo_url`.

## RUNTIME STATUS
STATIC_AUDIT. No GPU inference was run; no checkpoint was downloaded. Executed read-only:
- `doctor.py --model ptha_mtst` → `[BLOCKED] source:ptha_mtst: source unavailable, dirty, or not pinned`; `[BLOCKED] runtime:ptha_mtst: command unavailable`; `[OK] external-assets:ptha_mtst: required evaluation assets resolved`; `[BLOCKED] command:ptha_mtst: COMMAND_AUDITED`; `[OK] environment:ptha_mtst: UNCONFIGURED`; overall `[BLOCKED] doctor`.
- `readiness_matrix.py --json --model ptha_mtst` (job `bg_1199`) → every readiness flag `false` (`SOURCE_VERIFIED`, `ENVIRONMENT_PLANNED`, `ENVIRONMENT_INSTALLED`, `RUNTIME_VERIFIED`, `GPU_VERIFIED`, `CHECKPOINT_READY`, `EVALUATOR_READY`, `SCIENTIFICALLY_REPRODUCED`), `checkpoint_policy_status: BLOCKED`, blockers `['source unavailable, dirty, or not pinned', 'environment: source unavailable and no checkpoint variants; …', 'evaluator command: COMMAND_AUDITED']`.
- `pc_readiness.py --model ptha_mtst` → `ptha_mtst  BLOCKED_SOURCE` with reason `source cannot run: author repository https://github.com/zyinan99/MTST_PTHA exists and is reachable …`; `gpu : nvidia-smi=yes torch=2.14.0+cpu cuda_available=False`; summary `{"BLOCKED_SOURCE": 1}`.
- `ADAPTERS.get('ptha_mtst')` → `None` (`workbench/backend/adapters/models.py:139` `ADAPTERS = {...}` has 15 model ids; `ptha_mtst` absent). `environment_planner.py:682-686, 810-812` excludes it before any solve.
- No `*ptha*` artifact under `workbench/artifacts/`; `reproduction_status: BLOCKED_NO_VERIFIED_FASHIONIQ_CHECKPOINT` (`models.yaml:235`).

**Verdict: NOT_RUN → BLOCKED.** No source, no adapter, no checkpoint, no environment, no output.

## PAPER-METRIC STATUS
`models.yaml:232-234` `reported_r10: 54.99`, `reported_r50: 76.28`, `reported_mean: 65.64`. Re-derived **four independent routes**, all agreeing (STATIC_AUDIT):
1. `curl https://arxiv.org/html/2504.05316v1` (HTTP 200, 303805 B) → parsed table 11 row by row: `PTHA (Ours) | BLIP2 | FashionIQMTST | 50.77 73.78 | 55.91 75.36 | 58.28 79.70 | 54.99 76.28 | 65.64`.
2. `pdftotext -layout paper.pdf paper.txt` → line 474 `PTHA (Ours)              BLIP2        FashionIQMTST      50.77    73.78     55.91     75.36    58.28      79.70      54.99    76.28   65.64`.
3. `pdftotext -f 7 -l 7 -layout` → page 7, line 54, identical row; caption at page-7 line 30: `TABLE III: Comparison on FashionIQ validation set. “Avg.” means (Recall@10 + Recall@50)/2.`; `paper.txt:450` carries the same caption.
4. Arithmetic (above) — the printed Average/Avg. columns are the macro of the printed categories.
The immediately preceding data row is the control: `PTHA (Ours) BLIP2 - 49.54 72.81 55.50 73.97 57.96 78.96 54.33 75.24 64.79` (`paper.txt:473`). Registry values match the MTST row and no other.

**Verdict: transcription VERIFIED. Reproducibility UNKNOWN** (no artifact can produce these numbers; see C2/C3).

## CRITICAL FINDINGS

### C1 — Two committed reference rows bind the identical condition set, so `select_reference` returns `None` for **every** query (P1, NEW this round)
`papers.yaml:1657-1722` (`ptha_mtst_fiq_table3_fashiqmtst_blip2`) and `:1723-1771` (`ptha_mtst_fiq_no_pretrain_table3`) both carry `model_id: ptha_mtst`, `checkpoint_id: null` (`:1667`, `:1732`), `checkpoint_training_noise_pct: 0` (`:1668`, `:1733`), `gallery_protocol: fashioniq_original_split` (`:1675`, `:1738`), `reference_excluded: false`, `split: val`, `dataset: fashioniq`, `reference_status: VERIFIED`.
Executed:
```
select_reference('ptha_mtst')                                                    -> None | more than one verified reference matches this run: ['ptha_mtst_fiq_no_pretrain_table3', 'ptha_mtst_fiq_table3_fashiqmtst_blip2']
select_reference('ptha_mtst', checkpoint_training_noise_pct=0)                   -> None | same
select_reference('ptha_mtst', checkpoint_training_noise_pct=0, protocol_id='fashioniq_original_split') -> None | same
```
`benchmark_references.py:168-184` is the cause: `matches()` only filters on `checkpoint_training_noise_pct` and `gallery_protocol`; both rows pass both, and the ambiguity branch (`:184`) refuses rather than picking. There is no `method_variant`/`pretraining_data` criterion, so the disambiguating field the rows themselves carry (row `Pretraining Data` = `FashionIQMTST` vs `-`) is not part of selection.
Worse, forcing either row through the gate gives the same verdict for the same run — executed `evaluate_gate(reference, {'r10':54.99,'r50':76.28,'mean':65.64}, run_meta={protocol_id:'fashioniq_original_split', checkpoint_id:None, checkpoint_training_noise_pct:0, evaluation_noise_pct:0, split:'val', dataset:'fashioniq'}, data_kind='experiment', real_execution=True)` → `PASS_WITHIN_5_PERCENT` for **both** `ptha_mtst_fiq_table3_fashiqmtst_blip2` and `ptha_mtst_fiq_no_pretrain_table3`. The 65.64 row and the 64.79 row are 1.3 % apart, so no near-miss discriminates them either.
Consequence today (executed `pc_readiness.py --json --model ptha_mtst`): `benchmark_references[0] = {reference_id: null, reason: "more than one verified reference matches this run: …"}` — the model is surfaced as having *no* usable reference. Severity **P1** (the previously-correct single-row binding is replaced by a state in which a future run cannot be judged at all); recommended action: give selection a discriminator that the paper actually states (the pretraining-data condition) or mark all but one row as non-selectable, and re-register under a new `registry_version`.

### C2 — The published upstream package cannot be imported (P0, external, STATIC_AUDIT)
`MTST_PTHA/src/lavis/models/__init__.py` line 54 (blob `8027b1e7d018b7b3c5e9de9f691d753bb31c3e20`):
```python
from lavis.models.blip2_models.blip2_qformer_tgir_concatembedding_sim_zyn import Blip2Qformer4TGIRCESIMZYN
```
Re-derived, not inherited: I enumerated all `from lavis.models…` lines in that 296-line file, resolved each module to `<repo>/MTST_PTHA/src/<path>.py` or `__init__.py`, and tested membership in the 361-blob tree. Output: `total from-lavis.models lines: 48`, `MISSING: 1 [(54, 'lavis.models.blip2_models.blip2_qformer_tgir_concatembedding_sim_zyn')]`. `curl -o /dev/null -w %{http_code}` on the raw path → **404**. Unguarded: no `try:`/`except` anywhere in lines 1–70. Reached by the evaluator: `blip_validate.py:16` `from lavis.models import load_model_and_preprocess` (also used at `:45`, `:104`, `:150`). The near-identical sibling that does exist is `blip2_qformer_tgir_concatembedding_sim.py` (registers `blip2_tgir_ce_sim`) plus `blip2_qformer_cir_rerank.py` — the `_zyn` variant is in none of them.
Additional independent fact (resolves the previous round's U1): I fetched the recursive tree of **all 8 commits** (`a6efd886…`, `22548343…`, `3584a053…`, `f8bcca37…`, `f41ed83b…`, `36979a41…`, `e064beb5…`, `6a02ee47…`) — **no revision ever contained a `*_zyn*` path**. So no pin can dodge this by choosing another revision. Severity **P0** for reproduction; not patchable silently.

### C3 — The registered row needs a corpus that is neither published nor loadable (P0, external)
The row is `PTHA (Ours) | BLIP2 | FashionIQ_MTST`, i.e. it requires the paper's FashionIQ MTST franchise: `579,114 triplets` from `dress/shirt/toptee` (paper Table I, v1 HTML: `FashionIQ_{MTST} | Fashion | FashionIQ (train) | 579,114 | 26,048`). That corpus is (a) not on the author Hub (only the two CIRR JSONs exist, above) and (b) **not loadable by the published code**: `README.md:93` documents `--dataset {'CIRRGen' or 'FashionIQGen'}` for the pretraining stage, but `blip_fine_tune_2.py:400` rejects anything outside `['fashioniq','cirr']` (`:401` raises), and `data_utils.py` defines no `FashionIQGen` class — its dataset classes are `FashionIQDataset:111`, `CIRRDataset:224`, `CIRRDatasetGen:323`, `CIRCODataset:430`, `COCODataset:570`, `COCOValSubset:592`, `VAWDataset:652`, `VAWValSubset:675`. Paired with no published FashionIQ weight, `reproduction_status: BLOCKED_NO_VERIFIED_FASHIONIQ_CHECKPOINT` (`models.yaml:235`) is **true but one-sided**: it names the checkpoint half only. Severity **P0** for the reproduction claim; recommend recording the corpus gap explicitly.

### C4 — Both rows assert `checkpoint_training_noise_pct: 0`, a training condition the paper never states (P2)
The word "noise" occurs **0 times** in the paper's body text: `grep -c noise paper.txt` → 0, and `'noise' count` over the v1 HTML plain text → the only hits are two bibliography strings ("noisy text supervision", "noisy annotations"). The paper describes no noise-injection protocol for PTHA at all. `papers.yaml:1668` and `:1733` nevertheless declare `checkpoint_training_noise_pct: 0`.
Consequence in the gate, executed on `paper_gate.py:211-239`: `_reference_expected` returns that field as a **stated** condition, so `_comparability_blockers` (`:241-255`) treats it as a hard comparison and `_reference_constraint_count` (`:229-238`) counts it toward the "states something" threshold. A condition the paper does not state acquires the authority of one it does. The registry's own policy for exactly this situation is `null` (`models.yaml:403,461,515,526,595,649,705…` for hint/encoder/pair-style variants, per context item 11). Severity **P2** — unsupported condition claim; recommend `null` (or paper evidence).

### C5 — The "no source" claim survives in four documents that the registry now contradicts (P1)
`models.yaml:224,229` record a reachable author repository and state the earlier "no source exists" note was incomplete. Still asserting otherwise in the same working tree: `workbench/docs/UPSTREAM_DATA_PREPARATION_AUDIT.md:19` ("`ptha_mtst` has no source and is `SOURCE_UNAVAILABLE`"), `:38` (`| ptha_mtst | — | none |`), `:184-186` ("### ptha_mtst — no source / `SOURCE_UNAVAILABLE`. No repository, no checkpoint mapping, no evaluator. Not auditable."), `workbench/docs/SOURCE_TO_WORKBENCH_PARITY_MATRIX.md:35` (`no source`), `workbench/docs/FINAL_PC_READINESS.md:151` ("no source and no checkpoint variant; blocked."), `workbench/registry/assets/cluster_b5_ptha_mtst.yaml:3,32-33,56`. `workbench/docs/UPSTREAM_AUDIT.md:10` uses `Source pin: unavailable` and `n/a` — accurate. Severity **P1**; the same defect class as `OPEN_BLOCKERS.md:198-200` (B13) which was fixed only in the registry.

### C6 — `READINESS_MATRIX.md` prints `COMMAND_AUDITED` for a model with no adapter (P2, re-confirmed this round)
`workbench/docs/READINESS_MATRIX.md:28` ends `… ; evaluator command: COMMAND_AUDITED`. Trace, executed: `doctor.py:167` `command_implemented = adapter is not None and getattr(adapter, "command", None) is not OfficialScriptAdapter.command` → `False` here; `doctor.py:169` then stores `record(f"command:{model_id}", "BLOCKED", command_status or "COMMAND_AUDITED", …)` — `command_status` is `None`, so the **detail** falls back to the literal `COMMAND_AUDITED`; `readiness_matrix.py:113-114` prints that detail. My own `doctor.py --model ptha_mtst` run reproduces it verbatim: `[BLOCKED] command:ptha_mtst: COMMAND_AUDITED` with no script. Severity **P2**; print the status (`BLOCKED`) or `COMMAND_UNAUDITED`.

### C7 — The model asserts no data contract at all, and preflight therefore raises no data blocker (P2)
Executed: `fashioniq_required_paths(m, root)` → `()`, `model_requirements` → no paths/blocks, `preflight_blockers` → only source + command. Combined with `preparation_contract: unavailable` (`:217`), a future pin plus a raw un-resized dataset would pass preflight. There is also no `fashionIQ_dataset` link step, no `.png` extension assertion, and no `cap_train_{cat}_rev.json` requirement — that file is opened by `data_utils.py:157` on the train path and is absent from both the README layout and the local mirror (`ls data/fashionIQ_dataset/captions/` → the 9 `cap.*.{train,val,test}.json` only). Severity **P2**.

## MINOR FINDINGS
- **M1 (P3)** — Dead evaluation default: `blip_validate.py:189` `parser.add_argument("--blip-model-name", default="blip2_cir_rerank_learn", type=str)`; `rerank_learn` occurs in no blob of the HEAD tree (only `blip2_qformer_cir_rerank.py` exists). Only a bare invocation hits it, then fails at the registry lookup.
- **M2 (P2)** — No `environment` block in the `ptha_mtst` entry (`:213-238`) while the README pins `python=3.9` / `torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2` and `requirements.txt` adds `transformers==4.36.2, timm==0.9.12, spacy==3.7.2, pycocoevalcap==1.2, …`. Consequence, executed: `environment:ptha_mtst OK (UNCONFIGURED)` — printed as OK for a model with no environment.
- **M3 (P3)** — `blip_validate.py:200` iterates `['dress','toptee','shirt']` while Table III prints `Dress | Shirt | Toptee`; keys are name-based (`:90-91`) so no numeric effect.
- **M4 (P3)** — `data_utils.py:169-172` silently shrinks the gallery with no count; 0 occurrences locally, but on a partial mirror the Recall@K denominator changes per category.
- **M5 (P2)** — `workbench/registry/assets/cluster_b5_ptha_mtst.yaml` is not machine-read: `grep -rn "cluster_b5" --include=*.py .` → 0 matches; only review documents reference it. Its `classification: OFFICIAL_ASSET_UNAVAILABLE` and `search_locations_checked` (only two workbench-internal entries, `:58-60`) never reach preflight.
- **M6 (P3)** — Same file records `expected_format: pt` (`:41`) for a checkpoint that does not exist, and `:33` still says "the workbench does not invent one [a repository URL]" while `models.yaml:224` now carries one.
- **M7 (P3)** — `UPSTREAM_AUDIT.md:10` records `Training noise: n/a` / `Literature label: n/a` / `Exact native protocol: n/a` for this model; the protocol is in fact determinable from source (see EVALUATION PROTOCOL) and the `papers.yaml` row already records it.

## UNRESOLVED EVIDENCE
- **U1 — The revision the paper's numbers were produced at.** `.../releases` → `[]`, `.../tags` → `[]`; HEAD `6a02ee47…` was pushed `2025-02-09`, arXiv v1 is `2025-02-21`. No revision marker exists. UNKNOWN.
- **U2 — Whether any revision ever shipped the `_zyn` module.** **RESOLVED-NEGATIVE**: all 8 commit trees fetched; no `*_zyn*` path in any of them. (Previously UNKNOWN.)
- **U3 — What `tuned_mtst_cirr.pt` contains.** Not downloaded, not loaded. Name, the paper's separate CIRR-MTST stage (§V-B-1) and the two CIRR-only companion JSONs say CIRR; the payload is UNKNOWN.
- **U4 — Whether 54.99/76.28/65.64 is reproducible from any public artifact.** Evidence says no (no FashionIQ weight, no FashionIQ MTST corpus, no `FashionIQGen` loader, evaluator cannot import). Absence across deleted Hub revisions / private mirrors is not provable. UNKNOWN-BUT-EVIDENCE-AGAINST.
- **U5 — Author Hugging Face revision history / deleted uploads.** Only current repo lists queried; no revision walk. UNKNOWN.
- **U6 — Source of `cap_train_{cat}_rev.json`** (`data_utils.py:157`). Absent from the README layout and the local mirror; UNKNOWN.

---

# EXECUTION TRACE
`paper/README/config → source dataset → preprocessing → official evaluator → workbench invocation → checkpoint → output → metric comparison`

1. **paper/README/config →** `arXiv:2504.05316v1` (submitted 2025-02-21; PDF 9095049 B, `sha256 0801d37a…`, 12 pages). Table III row `PTHA (Ours) BLIP2 FashionIQMTST` = `50.77 73.78 | 55.91 75.36 | 58.28 79.70 | 54.99 76.28 | 65.64` (`paper.txt:474`, page-7 line 54, HTML table 11). Config: paper §V-A-2 — "the same as SPRC", frozen `ViT-g/14` from EVA-CLIP, BLIP-2 encoders, batch 128, 32 query tokens; README (`zyinan99/MTST_PTHA@main`, blob-bound) `python=3.9`, `torch==2.0.1`, `requirements.txt`. **ARROW: consistent; the registry records the numbers but none of the config (no `environment` block).**
2. **→ source dataset** — `data_utils.py:15` `base_path = Path("path_to_data_dir")` (operator-edited literal). Reads `captions/cap.{cat}.val.json` (`:153`), `image_splits/split.{cat}.val.json` (`:167`), `images/{id}.png` (`:170`, `:206`). Local mirror: 74381 `.png`, splits 3817/6346/5373, 0 missing, 0 refs/targets absent from gallery. **ARROW: paths and mirror agree; the workbench wires nothing (`fashioniq_required_paths → ()`, executed).**
3. **→ preprocessing** — `blip_validate.py:52-53` `targetpad_transform(1.25, 224)`; `data_utils.py:94-108` exact `Compose` + CLIP normalisation; applied `:192/:201/:207`. **ARROW: deterministic and traceable; unregistered (P3).**
4. **→ official evaluator** — `blip_validate.py:16` `from lavis.models import load_model_and_preprocess` → **fails** at `lavis/models/__init__.py:54` (absent module, 404, absent from every one of the 8 commit trees, unguarded). Past that: `:186-200` dispatch `--dataset fashioniq`, `:45` model load, `:48` `torch.load`, `:49` `load_state_dict(strict=False)`. **ARROW: BLOCKED at the import (P0, external).**
5. **→ workbench invocation** — `ADAPTERS.get('ptha_mtst')` → `None` (executed); `preflight_blockers` → source + command blockers only; `doctor command:ptha_mtst BLOCKED`, `script: null`; `pc_readiness` → `BLOCKED_SOURCE`. **ARROW: BLOCKED — no adapter is ever registered for this model.**
6. **→ checkpoint** — `checkpoint_variants: []` (`:238`) → `select_checkpoints` → `BLOCKED_CHECKPOINT` (executed). Public: `tuned_mtst_cirr.pt` (CIRR) + 2 CIRR JSONs only; 5 candidate FashionIQ names → 404; no weight blob in the tree; no releases/tags. **ARROW: BLOCKED — nothing to evaluate, and the paper row additionally needs the unpublished 579,114-triplet corpus.**
7. **→ output** — none. No `*ptha*` file under `workbench/artifacts/`. **ARROW: NO OUTPUT.**
8. **→ metric comparison** — unreachable. No parser for the `{cat}_recall_at10`/`average_recall*` shape (`metrics_extraction.py:328-341` → `PARSER_UNAVAILABLE`); and even for a hypothetical run the committed rows now bind to the *same* conditions, so `select_reference` returns `None` (`benchmark_references.py:184`, executed). **ARROW: BLOCKED / PARITY_NOT_EVALUATED.**

Arrows 4–8 are BLOCKED. Nothing below arrow 3 is RUNTIME_VERIFIED; every claim here is STATIC_AUDIT.

---

# CHECKLIST

1. **Registry entry exists and is unique** — VERIFIED — `models.yaml:213`; executed `load_registry()` → 16 models, 1 match — NONE — none.
2. **`method_name` matches the paper's vocabulary** — VERIFIED — `models.yaml:214` `PTHA + MTST`; paper §I/§IV names "Modification Text-oriented Synthetic Triplets (MTST)" and "Prototypical Two-Hop Alignment (PTHA)" — NONE — none.
3. **`paper_title` matches the paper** — VERIFIED — `models.yaml:218` vs arXiv `citation_title` (OBSERVED) — NONE — none.
4. **`paper_url` resolves to the same work** — VERIFIED — `curl -sL https://arxiv.org/abs/2504.05316` → 200, same title/authors, `citation_date 2025/02/21` — NONE — none.
5. **`publication_year: 2025`** — VERIFIED — `models.yaml:216`; submission history `[v1] Fri, 21 Feb 2025` — NONE — none.
6. **`research_generation: recent`** — VERIFIED — `models.yaml:215`; `registry.py:66-67` allows `legacy|recent`; 2025 ≥ 2020 — NONE — none.
7. **`upstream_repo_url` recorded** — VERIFIED — `models.yaml:224` `https://github.com/zyinan99/MTST_PTHA`; API confirms public, MIT, `fork false`, author's own commits — NONE — none. *(This was REFUTED in the previous round as `null`; the working tree fixed it.)*
8. **`source_available: false` is accurate** — VARIANT — `models.yaml:227`; `third_party/` really has no checkout (14 dirs), so "unavailable *here*" is true, but combined with `:224` the field reads as "no source exists" — P3 — rename/re-document as "not synced/pinned".
9. **`source_dir: null` is accurate** — VERIFIED — `:228`; `ls workbench/third_party/` → 14 dirs, none matching — NONE — none.
10. **`source_blocker` text is factually correct** — VERIFIED — `:229` vs API: HEAD `6a02ee4763fc2260b6abfb18be78853facb91d86` at `2025-02-09`, authors `Yinan Zhou`, README line 1 = the paper title — NONE — none.
11. **`upstream_commit_sha: null` accurate** — VERIFIED — `:226`; nothing is pinned — NONE — none.
12. **A pinned checkout exists to audit against** — NOT_RUN — `third_party/MTST_PTHA` absent; `preflight_blockers` reports `upstream source unavailable` — P1 — pin `6a02ee47…` (or another revision) and re-audit; note all 8 revisions share the C2 defect.
13. **Upstream HEAD is known and recorded** — VERIFIED — `:229` vs `.../branches/main` → `6a02ee47…` — NONE — none.
14. **Upstream tree completeness** — VERIFIED — `.../git/trees/main?recursive=1` → `truncated: false`, 433 entries, 361 blobs; tops match the record — NONE — none.
15. **Upstream license recorded** — VARIANT — GitHub API `license MIT`; no license field in the registry schema — P3 — none (out of schema).
16. **Environment recorded in the registry** — VARIANT — README + `requirements.txt` pin the stack; entry has no `environment` block — P2 — add one (every sibling recent model has one).
17. **`preparation_contract` is a legal value** — VERIFIED — `:217` `unavailable` ∈ allow-list `registry.py:73` — NONE — none.
18. **The empty contract leaves no silent data hole** — VARIANT — executed `fashioniq_required_paths(m, Path('data'))` → `()`; `preflight.py:28` falls back to it; no data blocker raised — P2 — pair the sentinel with an explicit data gate.
19. **`native_protocol` is recorded** — VARIANT — `:230` is `null`; source gallery/reference policy = `fashioniq_original_split` (`validate_blip.py:38-55`, `src/evaluation/fashioniq.py:172-174`) — P1 — record after pinning.
20. **`supported_protocols` recorded** — VARIANT — `:231` `[]`, internally consistent with 19 but empty — P1 — same action.
21. **`literature_split_label` recorded** — VARIANT — absent from `:213-238`; correct value `original` (`registry.py:50` ties it to the protocol) — P3 — add with 19.
22. **`reported_r10` matches the paper** — VERIFIED — `:232` `54.99`; Table III Average R@10 = 54.99 (HTML table 11, `paper.txt:474`) — NONE — none.
23. **`reported_r50` matches the paper** — VERIFIED — `:233` `76.28`; Table III Average R@50 — NONE — none.
24. **`reported_mean` matches the paper** — VERIFIED — `:234` `65.64`; Table III `Avg.` column — NONE — none.
25. **`reported_mean = (r10+r50)/2`** — VERIFIED — computed `(54.99+76.28)/2 = 65.635 → 65.64`; paper caption + `blip_validate.py:95` agree — NONE — none.
26. **The row is the row the registry means** — VERIFIED — MTST row `54.99/76.28/65.64` vs control row `54.33/75.24/64.79` (`paper.txt:473`) and SPRC `54.92/74.97/64.85` — NONE — none.
27. **The row's training condition is recorded** — VARIANT — the row is `… | FashionIQMTST`, i.e. it needs the 579,114-triplet corpus; no row-condition field exists in `models.yaml` — P1 — record it (C3).
28. **`reproduction_status` is legal and accurate** — VERIFIED — `:235` `BLOCKED_NO_VERIFIED_FASHIONIQ_CHECKPOINT`; checkpoint half confirmed in every direction probed — NONE — none.
29. **`reproduction_status` names the primary blocker** — VARIANT — (a) no FashionIQ weight, (b) no FashionIQ MTST corpus, (c) upstream cannot import, (d) no source is pinned; the status names only (a) — P2 — incomplete, not wrong.
30. **`strict_clean_checkpoint_available: false`** — VERIFIED — `:236`; no checkpoint published at all → none is clean — NONE — none.
31. **`notes` accurate** — VERIFIED — `:237`; Hub holds `tuned_mtst_cirr.pt` + two CIRR JSONs only — NONE — none.
32. **`checkpoint_variants: []`, no bundles** — VERIFIED — `:238`; executed `select_checkpoints` → `BLOCKED_CHECKPOINT` / `"no checkpoint variant is declared in the registry"` — NONE — none.
33. **Author Hub repo exists and was inspected** — VERIFIED — `api/models/yinanzhou1/mtst_ptha` → `sha b5e669c8…`, 5 siblings with sizes and LFS oids — NONE — none.
34. **A FashionIQ fine-tuned checkpoint is published** — VERIFIED-NEGATIVE — README links CIRR artifacts only; 5×404 probes; 0 weight blobs in the tree; `releases`/`tags` empty — NONE (registry correct) — none.
35. **An eligible FashionIQ MTST corpus is published** — VERIFIED-NEGATIVE — Hub has only the two CIRR JSONs; README's `FashionIQGen` has no dataset class and no accepted CLI value (`blip_fine_tune_2.py:400-401`) — P1 — record as its own blocker (C3).
36. **Digests/sizes recorded where published** — VARIANT — registry has no variants; the published CIRR weight exposes `2720352973` B + `sha256 1c4dab86…` — P3 — record if ever registered.
37. **Evaluator entrypoint identified** — VERIFIED — `blip_validate.py:186-200` (`__main__` at `:186`, dispatch `:194-200`). Note `validate_blip.py` also has a `__main__` (`:558`) but only dispatches CIRR (`main()` at `:497-509`) — cosmetically misleading, not an alternative evaluation path — NONE — none.
38. **Evaluator runnable as published** — VERIFIED-NEGATIVE (BLOCKED) — `lavis/models/__init__.py:54` absent module; 404; unguarded lines 1–70; reached via `blip_validate.py:16` — P0 — external blocker, do not patch silently.
39. **Missing-module count/line exact** — VERIFIED — my script over the blob-bound `__init__.py`: `total from-lavis.models lines: 48`, `MISSING: 1 (54, 'lavis.models.blip2_models.blip2_qformer_tgir_concatembedding_sim_zyn')`, file 296 lines — P0 — none.
40. **Name-confusion sibling exists** — VERIFIED — `blip2_qformer_tgir_concatembedding_sim.py` exists (registers `blip2_tgir_ce_sim`); `blip2_qformer_cir_rerank.py` exists; the `_zyn` name appears in exactly one blob (`__init__.py` itself) — P3 — a future fix must not be "rename the import".
41. **FashionIQ dispatch traced** — VERIFIED — `blip_validate.py:199-200` → `clip_finetune_fiq(['dress','toptee','shirt'], …)` defined `:26` — NONE — none.
42. **Gallery construction matches the protocol** — VERIFIED — `data_utils.py:165-168` full split, file order; `utils.py:54-55` DataLoader without `shuffle`; `:76` `index_names.extend(names)` — NONE — none.
43. **Reference exclusion policy matches the row** — VERIFIED — `validate_blip.py:44-55` no reference mask; CIRR path masks at `:257-261` — NONE — none.
44. **Gallery size well-defined locally** — VERIFIED — 3817/6346/5373, 0 ids dropped, all unique — NONE — none.
45. **Every reference and target in gallery** — VERIFIED — `refs_missing 0`, `targets_missing 0`, `ref==target 0` per category; the `assert` at `validate_blip.py:51` would hold — NONE — none.
46. **Gallery membership not silently data-dependent** — VARIANT — `data_utils.py:169-172` removes ids with no `.png`, no count/warning — P3 — report the dropped count.
47. **Recall@K maths correct** — VERIFIED — `validate_blip.py:44,45,46,49-51,54-55` — NONE — none.
48. **Macro/mean matches the paper's definition** — VERIFIED — `blip_validate.py:93-95`; Table III caption `(Recall@10 + Recall@50)/2` — NONE — none.
49. **K values 10 and 50** — VERIFIED — `validate_blip.py:54-55`; matches `reported_r10`/`r50` — NONE — none.
50. **Preprocessing deterministic and traceable** — VERIFIED — `blip_validate.py:52-53`; `data_utils.py:94-108` — NONE — none.
51. **Evaluation augmentation-free** — VERIFIED — eval branches use only `self.preprocess` (`data_utils.py:192,201,207`); `generate_randomized_fiq_caption` has no callsite in `blip_validate.py`/`validate_blip.py`, only `blip_fine_tune_2.py:124` — NONE — none.
52. **Workbench adapter produces a command** — VERIFIED-NEGATIVE (BLOCKED) — `ADAPTERS.get('ptha_mtst')` → `None` (15 keys at `adapters/models.py:139`) — P1 — implement only after a pin and C2.
53. **Preflight reports blockers and no false ones** — VARIANT — executed `['source cannot run: …MTST_PTHA…', 'upstream source unavailable', 'command not runnable: COMMAND_UNAUDITED']`: all true, but **no data blocker** (18) and the source line is the model's own long `source_blocker` text rather than a classified reason — P2 — classify it and add the data gate.
54. **Blocker reporting not misleading** — VARIANT — `READINESS_MATRIX.md:28` prints `evaluator command: COMMAND_AUDITED`; my `doctor.py --model ptha_mtst` reproduces `[BLOCKED] command:ptha_mtst: COMMAND_AUDITED`; cause `doctor.py:169` fallback with `command_implemented=False` (`:167`), printed by `readiness_matrix.py:113-114` — P2 — print the status or `COMMAND_UNAUDITED`.
55. **A run exists / parity was compared / the reference binds** — BLOCKED — no adapter, no checkpoint, no result artifact, no parser (`metrics_extraction.py:328-341`), and executed `select_reference('ptha_mtst', checkpoint_training_noise_pct=0, protocol_id='fashioniq_original_split')` → `None`, `"more than one verified reference matches this run: ['ptha_mtst_fiq_no_pretrain_table3', 'ptha_mtst_fiq_table3_fashiqmtst_blip2']"`; `SOURCE_TO_WORKBENCH_PARITY_MATRIX.md:35` cells are `NOT_APPLICABLE`/`SOURCE_EVIDENCE_MISSING` — P1 — the reference ambiguity (C1) must be fixed before any future run can be judged.

---

**FINAL VERDICT: BLOCKED** — no number can be produced today and the path is blocked at four independent points (no pinned checkout, no published FashionIQ checkpoint, the required 579,114-triplet FashionIQ MTST corpus is neither published nor loadable, and the published evaluator cannot import at `lavis/models/__init__.py:54`), while the paper metrics 54.99 / 76.28 / 65.64 are a correct four-route transcription of Table III at arXiv 2504.05316v1 and the round's newest defect is that the two committed reference rows bind an identical condition set, so `select_reference` refuses every `ptha_mtst` query and neither row can be distinguished from the other by the gate.
