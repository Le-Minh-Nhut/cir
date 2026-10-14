# csmcir

Independent adversarial, READ-ONLY audit. Nothing under `/home/heheboiz/data/cir` was written; scratch only in `/tmp`. Every claim is `STATIC_AUDIT` unless marked `OBSERVED` (a command I ran and its output) or `RUNTIME_VERIFIED` (an evaluator execution — **none exists for this model**: no GPU environment, no real checkpoint on disk).

---

## MODEL ID

`csmcir` / method_name `CSMCIR`.
- Registry: `workbench/registry/models.yaml` (model_id: csmcir).
- Adapter: `workbench/backend/adapters/models.py:28-33` — `CSMCIRAdapter`, `script = "src/validate_blip_csmcir.py"`.
- `source_dir: CSMCIR`, `source_available: true`.

## PAPER TITLE

`CSMCIR: CoT-Enhanced Symmetric Alignment with Memory Bank for Composed Image Retrieval`.
- Registry field `paper_url: null` (no URL), `publication_year: null`.
- OBSERVED: pinned `README.md:1-4` states the same title and claims "**(ACL206 Findings)**" (literal typo in the repo for ACL 2026 Findings).
- OBSERVED (network): `https://arxiv.org/abs/2601.03728` resolves to that exact title, authors Zhipeng Qian et al., submitted 2026-01-07, v3 2026-05-08; `web_search` also returns ACL Anthology `2026.findings-acl.216` ("Findings of the ACL 2026, pages 4411-4424").
- Registry records no paper URL although the paper is public → **P3**, `paper_url` should be `https://arxiv.org/abs/2601.03728` plus the ACL Anthology link.

## PINNED SOURCE COMMIT

- Registry: `upstream_commit_sha: 774f94e2076ff17ea91703a6239d2a08f0e1a44e`.
- OBSERVED: `git -C workbench/third_party/CSMCIR rev-parse HEAD` → `774f94e2076ff17ea91703a6239d2a08f0e1a44e`. **Match.**
- OBSERVED (network): `api.github.com/repos/qzp2018/CSMCIR/commits/774f94e2...` → sha confirmed, committer date `2026-04-27T08:30:50Z`, message `'Update README.md'`.
- OBSERVED: `source_clean_and_pinned(model, src)` from `workbench/backend/runtime.py:2013` returns `(True, '774f94e2...', None)`; `source_status` → `{tracked: 0, unauthorized: 0, approved: 6, invalid: 0}`. The 6 approved entries are declared runtime artifacts (`source_runtime_artifacts`, `registry.py:563`): the three staged `COT_ours2/fashioniq/*_cot_val.json` plus the `fashionIQ_dataset` link expectation. Checkout is pinned and not dirty.

## CHECKPOINT IDENTITY

Registry variant: `fashioniq_tuned_clip_best.pt`,
`download_url https://huggingface.co/peng12138/CSMCIR/resolve/main/fashioniq_tuned_clip_best.pt`,
`expected_sha256 c8812071de0640d647c5ba2a70628b80a3289ee2c47676f7a1b6abaa313fa49e`,
`expected_size_bytes 2720615859`, `checkpoint_training_noise_pct: 0`, `status: URL_VERIFIED`, `checkpoint_mapping_status: VERIFIED_METADATA`.

- OBSERVED (network, live): `GET https://huggingface.co/api/models/peng12138/CSMCIR?blobs=true` → repo `sha cf0c19bb346266c295b4b5772ebf03bd1c4f0467`, 22 siblings; entry `fashioniq_tuned_clip_best.pt` → `size 2720615859`, `lfs.sha256 c8812071de0640d647c5ba2a70628b80a3289ee2c47676f7a1b6abaa313fa49e`. **Size and digest match the registry exactly.**
- The HF repo contains exactly three `*_tuned_clip_best.pt` (fashioniq, cirr arithmetic, LaSCo, shoes) — no ambiguity about which file maps to FashionIQ.
- OBSERVED: `workbench/artifacts/checkpoints/` is **empty**; the only file named `fashioniq_tuned_clip_best.pt` on this host is `/tmp/tmpj82v7wks/csmcir/fashioniq_tuned_clip_best.pt`, **1 byte**, `sha256 2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881` — a placeholder, not the checkpoint.
- Consequence: the artifact-level mapping is `VERIFIED_METADATA` (bytes published by the author), but **no checkpoint was loaded or executed here**. Whether the weights actually reach the reported number is `UNKNOWN`.
- Registry `notes` describes the digest as "Hugging Face LFS oid" — OBSERVED correct for this file (the LFS `sha256` equals the registry `expected_sha256`).

## EVALUATION PROTOCOL

Registry: `native_protocol: fashioniq_original_split`, `supported_protocols: ['fashioniq_original_split']`, `literature_split_label: original`.

Workbench protocol table: `src/evaluation/fashioniq.py:161-168` — `fashioniq_original_split` = `gallery="full_split"`, `reference_excluded=False`, alias `fashioniq_original`.

Native evaluator (`workbench/third_party/CSMCIR/src/validate_blip_csmcir.py`):
- `blip_validate_fashioniq` at `:565`.
- Gallery: for each of `['dress','toptee','shirt']` (`:591`) it builds `FashionIQDataset('val', [dress_type], 'relative', preprocess)` (`:596`) and `FashionIQDataset('val', [dress_type], 'classic', preprocess)` (`:598`), then `index_features, index_names = extract_index_blip_caption_features(classic_val_dataset, blip_model)` (`:612`).
- `compute_fiq_val_metrics(relative_val_dataset, blip_model, index_features, index_names, txt_processors)` at `:615`, defined at `:27`.
- Gallery = the complete ordered `image_splits/split.{cat}.val.json` list (`src/data_utils_csmcir.py:416-419`), no pair union, no dedup beyond the file's own contents.
- **Reference stays eligible.** `compute_fiq_val_metrics` builds `labels` only against `target_names` (`:46-49`) and never masks `reference_names`; `generate_fiq_val_predictions` (`:149`) does use the reference — it *adds* the reference embedding to the query (`name_to_feat` + `blip_model.inference`) but never removes it from the candidate pool (`:196-198`).
- So the registry's `reference_excluded: false` claim is **correct for the native path**. The generic workbench helper `evaluate_fashioniq` (`src/evaluation/fashioniq.py:242`) implements the same protocol, but OBSERVED: `grep -rn "evaluate_fashioniq(" src workbench --include=*.py` → only its definition at `src/evaluation/fashioniq.py:242`; no callsite. The native evaluator is the only executing path for `csmcir`.

**Verdict: VERIFIED** (protocol id, gallery scope, reference policy, category set all trace to source).

## DATASET CONTRACT

- `fashioniq_layout: fashioniq_standard`; `preparation_contract: manual_standard`; `canonical_dataset_link: fashionIQ_dataset`; OBSERVED `FASHIONIQ_ROOT` resolves to `/home/heheboiz/data/cir/data/fashionIQ_dataset`.
- `preparation_contract_for_model(csmcir)` returns **None** (OBSERVED): `preparation_contracts.yaml` contains **zero** `csmcir` entries (`grep -c csmcir workbench/registry/preparation_contracts.yaml` → 0). The registry declares `manual_standard`, a layout name, not a registered contract id — so preflight falls back to `fashioniq_required_paths` / `paths_for_layout` (`registry.py:679-683`).
- Raw inputs the source actually opens (all relative to `base_path = checkout_root`, `src/data_utils_csmcir.py:16`):
  - `fashionIQ_dataset/captions/cap.{cat}.val.json` (`:405`)
  - `fashionIQ_dataset/image_splits/split.{cat}.val.json` (`:416-419`)
  - `fashionIQ_dataset/images/{id}.png` (`:468` classic, `:449/:452` train relative)
  - `fashionIQ_dataset/qwen_captions/{cat}_cot_{split}.json` (`:408`)
  - `COT_ours2/fashioniq/{cat}_cot_val.json` (`src/utils_csmcir.py:100-101`)
- OBSERVED: `data/fashionIQ_dataset/{captions,image_splits,images,qwen_captions}` all present; `images/` has 74381 files, `100%` `.png` — consistent with the declared `fashioniq_image_extensions: ['.png']`.
- Split sizes OBSERVED: dress 3817, shirt 6346, toptee 5373 (all unique), union 15415 (sum 15536 ⇒ 121 duplicate ids, see GALLERY CONTRACT).
- `fashioniq_standard` layout check only covers `captions`/`image_splits`/`images` and the 6 val files (`fashioniq_layout.py:20-27`); the `qwen_captions`/`COT_ours2` requirements are carried by the auxiliary-asset registry, not by the layout list.

**Verdict: VARIANT** — the declared layout matches the source, but the registry's `preparation_contract: manual_standard` names no registered contract, so the two caption directories that the run genuinely needs are not part of the layout contract (they are block-protected separately through `auxiliary_assets.yaml`). That works, but it is two mechanisms for one requirement. Severity **P2**.

## PREPROCESSING CONTRACT

- Adapter command (`adapters/models.py:32-33`): `python src/validate_blip_csmcir.py --dataset fashionIQ --blip-model-path <checkpoint>`. It omits `--blip-model-name`, but OBSERVED `validate_blip_csmcir.py:504` sets `default="blip2_cir_align_prompt_csmcir"` — the same string the class registers under — so the omissible flag is harmless. Note the flag is `--blip-model-path`, while pinned `README.md:117` documents `--model-path`; the adapter uses the working spelling.
- Transform: `--transform` default `targetpad` (`:506-507`), `--target-ratio` default `1.25` (`:506`) → `targetpad_transform(1.25, 224)` (`:577-580`); `input_dim = 224`.
- Eval path applies only the transform: classic branch `__getitem__` returns `self.preprocess(PIL.Image.open(path))` (`data_utils_csmcir.py:466-470`) with no randomised op.

**Verdict: VERIFIED** (preprocessing is deterministic TargetPad@1.25/224 in evaluation; the adapter omits a flag whose default equals the needed value).

## AUGMENTATION CONTRACT

- Training-only augments exist in the module (`SquarePad:31`, `TargetPad:53`, `RandomMask:95`, `RandomPerspective:144`, `GaussianBlur:162`) but are constructed in `__init__` only for `split == 'train'` paths; the validation `classic`/`relative` branches never apply them.
- Registry declares no augmentation policy; evaluation noise `0`.

**Verdict: VERIFIED / NONE** — evaluation is augmentation-free (STATIC_AUDIT; not executed).

## GALLERY CONTRACT

- Gallery per category = the full ordered val split, `len(split.{cat}.val.json)` **including the reference** (`data_utils_csmcir.py:416-419, 466-470`; `validate_blip_csmcir.py:598, 612`).
- `index_names` order equals split order; `generate_fiq_val_predictions` builds `name_to_feat = dict(zip(index_names, index_features[-1]))` (`validate_blip_csmcir.py:168`) — a dict, so a duplicate id silently collapses to one entry while `index_names` (a list) still holds two, and `sorted_index_names` in `compute_fiq_val_metrics` retains the duplicate lengths.
- **121-id shirt/toptee overlap (independently recomputed OBSERVED):** `set(split.shirt.val) ∩ set(split.toptee.val) = 121`. The same 121 ids appear in both published caption files.
- Dict-merge effect: `FashionIQDataset.__init__` merges per-category caption dicts with `dict.update` in `dress, shirt, toptee` order (`data_utils_csmcir.py:404-410`); `extract_index_blip_caption_features` always loads **all three** `COT_ours2` files in the same order and `update`s (`utils_csmcir.py:97-103`). Therefore for those 121 ids the **toptee caption wins**, even inside the shirt gallery; OBSERVED: of the 121 shared ids, **0** have identical `Final_Caption` across shirt and toptee (e.g. `B00018AA74`: shirt 'The shirt is black and has a simple, folded style with a crew neck.' vs toptee 'The top is black in color with a simple, classic crew neck style.').
- Consequence: 121/6346 (1.9%) of shirt gallery entries are captioned as tops, not shirts. This is upstream behaviour, present regardless of the alias. It shifts shirt gallery features — a real, if small, reproduction variance. Severity **P2** (does not invalidate the protocol, but must be reported, and it is the reason a per-category caption check must expect exactly 121 deltas for shirt).
- The workbench's own `build_fashioniq_gallery` (`src/evaluation/fashioniq.py:192`) would return the full split for this protocol, but it has no callsite (see EVALUATION PROTOCOL).

**Verdict: VERIFIED** for gallery composition and reference policy; the 121-id overlap is a documented, source-level property (**P2** note).

## METRIC CONTRACT

- `compute_fiq_val_metrics` (`validate_blip_csmcir.py:27-57`):
  - `distances = 1 - pred_sim` (`:41`); `sorted_indices = torch.argsort(distances, dim=-1)` (`:42`); `sorted_index_names = np.array(index_names)[sorted_indices]` (`:43`).
  - `labels = sorted_index_names == repeat(target_names, len(index_names))` (`:45-47`); `assert torch.equal(sum(labels,dim=-1).int(), ones(len(target_names)).int())` (`:48-49`) — one target hit per query in the gallery.
  - `recall_at10 = sum(labels[:, :10]) / len(labels) * 100` (`:52`); `recall_at50 = sum(labels[:, :50]) / len(labels) * 100` (`:53`).
  - **No reference masking** anywhere in the function.
- Macro: `average_recall_at10 = mean(recalls_at10)`, `average_recall_at50 = mean(recalls_at50)`, `average_recall = (mean(r50) + mean(r10)) / 2` (`:620-627`), over `['dress','toptee','shirt']` order (`:591`) — an unweighted category mean, matching the registry's `reported_mean` definition.
- Score path: `blip_model.inference(reference_image_features, feature_curr, input_captions, captions_cot)` (`:191`), defined at `blip_qformer_cir_align_prompt_csmcir.py:411-451`; it fuses reference+text through Q-Former and returns `sim_t2q.max(-1)`.
- Caption text used by `inference` is the **dataset's** caption (`input_captions` built from `relative_val_dataset`, `:181-185`); the gallery `captions_cot` is passed but the parameter is **never read** in `inference` (OBSERVED: `text_cot` appears only at `:411` and `:454` in the whole model file). OBSERVED consequence: `COT_ours2/fashioniq/*_cot_val.json` drives the **gallery features only** (through `extract_index_blip_caption_features` → `extract_target_caption_features`, `:498-535`), and `fashionIQ_dataset/qwen_captions/*` drives the **query text** (through `FashionIQDataset.__getitem__`). This is a genuine second dependency, and it is dead-weight in the query path — an upstream design detail, not a workbench error. Severity **P3** (documentation-level: any doc implying COT affects query text is wrong).
- Workbench parser `parse_csmcir_output` (`workbench/backend/metrics_extraction.py:200-281`) requires all six per-category keys plus `average_recall_at10/at50/average_recall`, validates 0-100, recomputes the macro means and rejects disagreement > 0.05, then `_check_parity(observed, paper, tolerance=0.05)` (`:25-42`). Matches the source's key names (`validate_blip_csmcir.py:620-627`). **VERIFIED**.
- `reported_mean 67.17` is the paper's `(R@10 + R@50)/2` convention, not `(R@1+R@10+R@50)/3` (that is the Shoes branch, `:678-681`). Consistent.

**Verdict: VERIFIED** (STATIC_AUDIT; arithmetic traced to source, not executed).

## SOURCE-TO-WORKBENCH PARITY

| Item | Upstream source | Workbench | Verdict |
|---|---|---|---|
| entrypoint | `src/validate_blip_csmcir.py` (`README.md:115`) | adapter `script` = same (`adapters/models.py:30`) | VERIFIED |
| model name | `blip2_cir_align_prompt_csmcir` (`:504` default; `blip2_qformer_cir_align_prompt_csmcir.py:27`) | adapter omits flag → default used | VERIFIED |
| dataset flag | `--dataset fashionIQ` | adapter passes it | VERIFIED |
| checkpoint flag | `--blip-model-path` (`:505`) | adapter passes it | VERIFIED (README's `--model-path` is the wrong spelling — **P3** doc bug) |
| protocol | full split, reference eligible | `fashioniq_original_split`, `reference_excluded=False` | VERIFIED |
| gallery builder | native code path | `evaluate_fashioniq` exists but has no callsite | VERIFIED (native runs) |
| metric keys | `{cat}_recall_at10/at50`, `average_recall*` | parser expects exactly those | VERIFIED |
| importability | `lavis.models` executes 98 unguarded absent imports before registration | workbench reports the blocker, does not patch | VERIFIED (blocker is real) |

## RUNTIME STATUS

- `reproduction_status: NOT_RUN`; `command_status: null`; no result artifact exists. OBSERVED: only `workbench/artifacts/results/fashioniq_original_split/mock/mock_csmcir-fiq_a.json` (a mock).
- OBSERVED preflight (`preflight_blockers`) blockers, in order:
  1. `source cannot run: pinned lavis.models package cannot be imported (98 unguarded imports of absent modules)`
  2. `CSMCIR LAVIS BLIP-2 pretrain weights unverified: ~/.cache/lavis/blip2_pretrained.pth`
  3. `CSMCIR EVA-ViT-g backbone weights unverified: ~/.cache/torch/hub/checkpoints/eva_vit_g.pth`
  4. `CSMCIR bert-base-uncased tokenizer unverified: ~/.cache/huggingface/hub/models--bert-base-uncased/vocab.txt`
  5. `canonical dataset link missing: workbench/third_party/CSMCIR/fashionIQ_dataset -> data/fashionIQ_dataset`
- OBSERVED warning (not a block): the three `COT_ours2` files are `staged as a source-compatible caption alias` with the published Qwen digests — the deliberate variant path.
- No GPU, no staged checkpoint, and the source cannot import ⇒ **NOT_RUN**. Every number below is untested at runtime.

**Verdict: NOT_RUN** (BLOCKED for official reproduction).

## PAPER-METRIC STATUS

- Registry: `reported_r10 57.07`, `reported_r50 77.27`, `reported_mean 67.17`.
- OBSERVED (network): the arXiv v3 HTML Table 1 "Quantitative comparison on the Fashion-IQ validation set" row `CSMCIR(Ours)` reads `52.45 | 74.81 | 57.70 | 75.76 | 61.14 | 80.98 | 57.07 | 77.27 | 67.17` (dress/shirt/toptee R@10/R@50, then averages). **All three registry numbers match the paper exactly.**
- OBSERVED (network): `training_log/fashioniq_validation_metrics.csv` epoch 9 = 52.35498547554016 / 75.06197094917297 (dress), 61.142271757125854 / 80.97909092903137 (toptee), 57.70363211631775 / 75.7605493068695 (shirt), averages 57.066963/<wbr>77.26720372835796/67.16708342234294 → rounds to **57.07 / 77.27 / 67.17**. The published CSV, the paper table and the registry agree to two decimals.
- Caveat that limits this: the paper's own table is a **dress-first** ordering (`Dress | Shirt | Toptee`) while the CSV is `dress, toptee, shirt`; the two agree on the averages, so the registry mapping is not affected.
- Mapping evidence chain (checkpoint → reported number): paper table = training-log row = registry numbers, and the HF repo exposes exactly one FashionIQ checkpoint whose LFS digest matches the registry. That is *metadata* evidence. **No run was executed**, so "this checkpoint reproduces 57.07/77.27/67.17" is `UNKNOWN`, not PASS.
- Paper verifiability: **verifiable** (arXiv abs + HTML v3 + ACL Anthology 2026.findings-acl.216 all reachable and consistent), with a repo-side typo (`README.md:4` "ACL206 Findings").

**Verdict: VERIFIED (numbers), UNKNOWN (reproduction).**

## CRITICAL FINDINGS

### C1 — The pinned package cannot import, and the failure precedes the registration (P0, independently reconfirmed)

OBSERVED reproduction of the workbench claim, run by me against the pinned checkout:

```
$ cd workbench/third_party/CSMCIR && python3 - <<'PY'
import re; from pathlib import Path
init = Path('src/lavis/models/__init__.py'); base = init.parent
bad = [i for i, l in enumerate(init.read_text().splitlines(), 1)
       if (m := re.match(r'from (lavis\.models\.[\w.]+) import (\w+)', l.strip()))
       and not (base / (m.group(1).replace('lavis.models.','').replace('.','/') + '.py')).is_file()]
print(len(bad), min(bad), max(bad))
PY
98 44 151
```

Supporting observations:
- `src/lavis/models/__init__.py` has 402 lines; lines 44-151 contain exactly 98 uncommented `from lavis.models....` imports, and the missing-module line list is `[44..61, 66..73, 79..128, 130..151]` — every line in 44-151 except 62-65 and 74-78 and 129.
- `any try/except in 40-155: []` — **none** of the failing imports is guarded, so the first one (`caption_query`) raises `ModuleNotFoundError` during import.
- `line 152: from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_csmcir import Blip2QformerCirAlignPromptCSMCIR` — the import the evaluator needs sits **after** all 98 failures.
- The class exists and is registered: OBSERVED `blip2_qformer_cir_align_prompt_csmcir.py:27` → `@registry.register_model("blip2_cir_align_prompt_csmcir")`, `:28` → `class Blip2QformerCirAlignPromptCSMCIR(Blip2Base)`.
- Entrypoint import: `src/validate_blip_csmcir.py:12` → `from lavis.models import load_model_and_preprocess`.
- Network corroboration OBSERVED: `raw.githubusercontent.com/qzp2018/CSMCIR/774f94e2.../src/lavis/models/blip2_models/blip2_qformer_cir_align_prompt_csmcir.py` → **200**; `..._target_image_caption_shiyan50.py` → **404**; `..._shiyan88.py` → **404**.
- Extra defect found: `src/lavis/models/blip2_models/.ipynb_checkpoints/blip2_qformer_cir_align_prompt_target_image_caption_shiyan88-checkpoint.py:28` defines a class that the live `__init__.py` shows commented out; it is not on the import path (its directory is a checkpoint dir, not a module), so it does not change the count. It confirms the author's working copy diverged from the public revision.
- Workbench response: reported first, not hidden (`preflight.py:137-141` intentional ordering; `workbench/patches/csmcir.md`; test `workbench/tests/test_csmcir_source_blocker.py`). The tests **assert the same facts** I recomputed; per the rules they are evidence that someone asserted this, and my independent count is the evidence that it is true.

Impact: the official evaluator cannot start; Recall@10/@50 cannot be produced from the pinned source as published; and because the registration never executes, `load_model_and_preprocess(name="blip2_cir_align_prompt_csmcir")` would fail even if the imports were bypassed. **Verdict: VERIFIED, P0.**

### C2 — The gallery caption input the evaluator reads was never published (P0)

- Executing reader: `src/utils_csmcir.py:100-101` opens `<checkout>/COT_ours2/fashioniq/{cat}_cot_val.json` unconditionally whenever the dataset is a `FashionIQDataset` (`:96-103`; all three categories loaded regardless of which category the dataset holds).
- Called from the official path: `validate_blip_csmcir.py:612` (metrics) and `:636` (visualisation) — the first is on the Recall path.
- These files have **no declared source**: `auxiliary_assets.yaml` `family: COT_ours2 captions` has `source_type: none`, `source_repo: null`, `source_revision: null`, `source_path: null`, `automatic_download_available: false`, `provenance_status: UNVERIFIED`, `expected_sha256: null`.
- OBSERVED (network): the author's HF repo (22 siblings, listed at the pinned revision `cf0c19bb...`) contains `fashioniq_qwen_captions/`, `cirr_qwen_captions/`, `shoes_captions/`, `cirr_test_files/`, `training_log/` and four `*.pt` — **no `COT_ours2` path at all**, no URL, no checksum anywhere.
- The workbench conclusion "the author never published them" is therefore **independently confirmed for the public trees queried** (HF repo at the pinned revision; the GitHub checkout contains only an untracked local `COT_ours2/` — OBSERVED `git status --porcelain` lists `COT_ours2/` as untracked and `git ls-files COT_ours2` is empty).

**Verdict: VERIFIED, P0.** Cannot be scoped beyond the public trees (HF history / deleted uploads not examined).

### C3 — No checkpoint, no GPU, no run (P0 for reproduction)

See CHECKPOINT IDENTITY and RUNTIME STATUS. A 1-byte placeholder at `/tmp/tmpj82v7wks/csmcir/fashioniq_tuned_clip_best.pt` is the only artefact on disk. `reproduction_status: NOT_RUN` is honest.

**Verdict: NOT_RUN.**

## MINOR FINDINGS

- **M1 (P2) — Two caption paths, correctly distinguished but documented only in a workbench doc.** `src/data_utils_csmcir.py:408` reads `fashionIQ_dataset/qwen_captions/{cat}_cot_{split}.json`; `src/utils_csmcir.py:100` reads `COT_ours2/fashioniq/{cat}_cot_val.json`. Both are real and both execute in one run. Independently confirmed; `workbench/docs/COT_OURS2_AUDIT.md:16-24` states the same and its line numbers check out (`data_utils_csmcir.py:408`, `utils_csmcir.py:100`, `data_utils_csmcir.py:223,226`, `validate_blip_csmcir.py:565`, `:596/:598`, `:612` all verified by me).
- **M2 (P3) — `workbench/docs/COT_OURS2_AUDIT.md:36-37` misattributes the defective handler.** It says "the FashionIQ branch has an `except:` at `src/utils_csmcir.py:166` whose handler re-executes the *same* expression". OBSERVED: the defect is at **`utils_csmcir.py:156-158`** (`except:` → `target_text.append(target_image_captions[name]['Final_Caption']); continue`) inside the `for name in names:` loop at `:152`; `:166` is `index_features_raw.append(image_embeds_frozen)`. The conclusion (a missing id re-raises `KeyError` instead of being skipped, so coverage must be complete) is **correct**; the line number is wrong. Fix the citation to `:153-158`.
- **M3 (P2) — `FashionIQDataset_imageedit` is dead code.** OBSERVED: `grep -rn "FashionIQDataset_imageedit"` across the whole checkout returns only its own definition at `src/data_utils_csmcir.py:175`; no instantiation anywhere, including the workbench. Its reads (`:216` `captions/keywords_in_mods_{dress_type}.json`, `:223` `COT_ours2/bert_captions/fashioniq/cap.*`, `:226` `COT_ours2/fashioniq/{cat}_cot_{split}.json`) are therefore never required at evaluation. The claim is **VERIFIED**.
- **M4 (P3) — `--blip-model-name` omitted by the adapter relies on the argparse default.** Correct behaviour, but a stricter command would pass the name explicitly so a future upstream default change cannot silently alter which class loads.
- **M5 (P3) — Pinned `README.md:117` documents `--model-path`; the code's flag is `--blip-model-path` (`:505`).** Following the README verbatim fails. The adapter uses the correct spelling.
- **M6 (P3) — Pinned `README.md:4` says "ACL206 Findings"**; the venue is ACL 2026 Findings. Cosmetic.
- **M7 (P2) — `manual_standard` is not a registered preparation contract.** `preparation_contract: manual_standard` on a model with no entry in `preparation_contracts.yaml`. Preflight silently falls back to the layout list. It works, but the registry field is a dangling reference — either drop the field or register the contract.
- **M8 (P3) — `enabled` image extension check is the only guard that the `.png` hard-code is satisfiable.** Local data is 100% `.png`, so no block was raised. On a `.jpg`-only FashionIQ mirror the run would die on the first image; the workbench catches this in preflight (`fashioniq_layout.py:90-120`, registry `fashioniq_image_extensions`). Correct design.
- **M9 (P3) — `text_cot` is a dead parameter** in `blip2_qformer_cir_align_prompt_csmcir.py:411` (and `:454`). Anything documenting "COT captions feed the query" would be wrong; COT feeds gallery features only.

## UNRESOLVED EVIDENCE

- **U1 — Whether `COT_ours2/fashioniq/{cat}_cot_val.json` held the published Qwen bytes.** No copy reachable; HF repo (pinned revision) has no such path; GitHub tracks none. `workbench/docs/COT_OURS2_AUDIT.md` reaches `SOURCE_COMPATIBLE_EQUIVALENCE_UNKNOWN`; I agree with that verdict on the evidence available to me. The staged alias is a labelled variant: OBSERVED all three staged files are **real files, not symlinks**, and their digests equal the published Qwen digests (`b7129baf...`, `bbe6f689...`, `acdacc3d...`) which also equal `auxiliary_assets.yaml` `official_sha256` and the manifest `downloaded_sha256`.
- **U2 — Whether `fashioniq_tuned_clip_best.pt` reproduces 57.07/77.27/67.17.** Metadata matches the paper and the author's training log; the weights were never loaded. `UNKNOWN`.
- **U3 — Whether the paper's table is fully reproducible from the pinned code in another respect** (e.g. the paper's query-time caption use). Not testable without a run; the paper says captions are pre-generated offline for target images (arXiv HTML §3.2), which matches the gallery-feature reading I traced, but the query-text path in the code uses the dataset's `cap.{cat}.val.json` captions, not COT. Whether the paper's official number used the aliased caption set is unproven.
- **U4 — HF history / deleted uploads.** Not examined; the "never published" claim is scoped to current public trees.
- **U5 — Cross-document claims.** I did not exhaustively re-derive every statement in every workbench doc; I checked the ones that name csmcir. Documents read: `COT_OURS2_AUDIT.md`, `AUXILIARY_ASSET_AUDIT.md` (§csmcir), `FASHIONIQ_PROTOCOL_PARITY.md`, `AUTOMATED_DATA_PREPARATION.md`, `FINAL_PC_READINESS.md`, `FIRST_GPU_EXPERIMENT.md`, `workbench/patches/csmcir.md`, plus registry and code.

## FINAL VERDICT

**BLOCKED.** Independently reconfirmed: the pinned `lavis.models` package executes 98 unguarded imports of modules the author never committed (lines 44-151, before the registration at 152), so the registered entrypoint cannot import; the FashionIQ gallery caption input it then needs (`COT_ours2/fashioniq/{cat}_cot_val.json`) was never published; the checkpoint URL/digest/size are correct but no checkpoint exists locally and no GPU run was performed. The registry entry, protocol, gallery, reference policy, and metric maths are all faithful to the source, and the reported numbers match the paper; reproduction itself is unproven.

---

# EXECUTION TRACE

`paper/README/config → source dataset → preprocessing → official evaluator → workbench invocation → checkpoint → output → metric comparison`

1. **paper/README/config →** paper `arXiv:2601.03728v3` Table 1 gives CSMCIR 57.07/77.27/67.17 (OBSERVED network fetch); pinned `README.md:99-120` documents the FashionIQ train/eval commands (with the `--model-path` typo M5); README Prerequisites pin `python 3.9`, `torch 2.0.1`, `torchvision 0.15.2` (OBSERVED, matches registry `environment`). Evidence: `README.md:7-16, 90-120`; registry `environment`.
2. **→ source dataset** — `base_path = Path(__file__).absolute().parents[1]` (`data_utils_csmcir.py:16`, `utils_csmcir.py:18`) ⇒ `<checkout>/fashionIQ_dataset` plus `<checkout>/COT_ours2`. Registry declares `canonical_dataset_link: fashionIQ_dataset`; OBSERVED the link does **not** exist yet (preflight blocker 5) and `FASHIONIQ_ROOT` is `/home/heheboiz/data/cir/data/fashionIQ_dataset` (dress 3817 / shirt 6346 / toptee 5373, all `.png`).
3. **→ preprocessing** — `targetpad_transform(1.25, 224)` (`validate_blip_csmcir.py:575-580`), applied per image in `__getitem__` (`data_utils_csmcir.py:466-470`). No augmentation on the eval path.
4. **→ official evaluator** — `validate_blip_csmcir.py:515` dispatches `--dataset fashionIQ` → `blip_validate_fashioniq(:565)`; loads the model with `load_model_and_preprocess(name="blip2_cir_align_prompt_csmcir")` (`:568`) via `lavis.models` — **this is the import that fails** (C1). Gallery built at `:594-598`, gallery caption features at `:612`, metrics at `:615`.
5. **→ workbench invocation** — `CSMCIRAdapter.command` (`adapters/models.py:32-33`) → `python src/validate_blip_csmcir.py --dataset fashionIQ --blip-model-path <checkpoint>`; interpreter from `WORKBENCH_CSMCIR_PYTHON` through `runtime.python_executable` (`runtime.py:1818`); guarded by `preflight_blockers`, which returns 5 blockers and stops before a command is emitted. Evidence: OBSERVED preflight output above.
6. **→ checkpoint** — registry URL resolves; HF LFS sha256 `c8812071...` and size `2720615859` match the registry (OBSERVED network). No local file: `workbench/artifacts/checkpoints/` empty; only a 1-byte placeholder at `/tmp/tmpj82v7wks/csmcir/` (OBSERVED).
7. **→ output** — none. `reproduction_status: NOT_RUN`; the only csmcir result artefact is a mock (`workbench/artifacts/results/fashioniq_original_split/mock/mock_csmcir-fiq_a.json`).
8. **→ metric comparison** — not reached. The parser that would do it (`metrics_extraction.py:200-281` + `_check_parity:25-42`) is source-correct but has never seen this model's stdout. **PARITY_NOT_EVALUATED.**

Every arrow from step 4 onward is `BLOCKED` at the import. Nothing downstream is `RUNTIME_VERIFIED`.

---

# CHECKLIST

Each item: `N. question — verdict — evidence — severity — action`.

1. **Registry entry exists and is unique** — VERIFIED — `models.yaml` model_id `csmcir`; `registry.model_by_id('csmcir')` resolves — NONE — none.
2. **method_name / paper title recorded** — VERIFIED — registry `method_name: CSMCIR`; title matches arXiv 2601.03728 — NONE — none.
3. **paper_url resolvable** — VARIANT — registry `paper_url: null`; paper is public at `arxiv.org/abs/2601.03728` and ACL Anthology `2026.findings-acl.216` — P3 — populate `paper_url`.
4. **upstream_repo_url recorded** — VERIFIED — `https://github.com/qzp2018/CSMCIR`; pinned README footer + arXiv abstract both point there — NONE — none.
5. **Pinned commit matches the checkout** — VERIFIED — `git rev-parse HEAD` → `774f94e2076ff17ea91703a6239d2a08f0e1a44e`; equals `upstream_commit_sha` — NONE — none.
6. **Checkout clean and pinned** — VERIFIED — `source_clean_and_pinned` → `(True, '774f94e2...', None)`; `source_status` tracked 0 / unauthorized 0 / approved 6 — NONE — none.
7. **Entrypoint file exists** — VERIFIED — `workbench/third_party/CSMCIR/src/validate_blip_csmcir.py`; adapter `script` matches (`adapters/models.py:30`) — NONE — none.
8. **Entrypoint imports the model package** — VERIFIED — `validate_blip_csmcir.py:12` `from lavis.models import load_model_and_preprocess` — NONE — none.
9. **98 unguarded absent-module imports (own count)** — VERIFIED — my script printed `98 44 151`; missing lines `[44..61,66..73,79..128,130..151]`; `wc -l` = 402 — P0 — none (upstream defect, correctly reported).
10. **Are those imports guarded?** — VERIFIED (no) — `any try/except in 40-155: []` — P0 — none.
11. **Do the failures precede the needed registration?** — VERIFIED — registration import at line 152; `max(missing)=151 < 152` — P0 — none.
12. **The class the evaluator needs is present** — VERIFIED — `blip2_qformer_cir_align_prompt_csmcir.py:28` `class Blip2QformerCirAlignPromptCSMCIR(Blip2Base)` — NONE — none.
13. **It is registered under the expected name** — VERIFIED — `:27` `@registry.register_model("blip2_cir_align_prompt_csmcir")` — NONE — none.
14. **The adapter asks for that name** — VERIFIED — adapter omits `--blip-model-name`; argparse default `:504` = `blip2_cir_align_prompt_csmcir` — P3 — pass the flag explicitly (M4).
15. **Patch `csmcir_models_init_compat.patch` applies cleanly to a COPY** — VERIFIED — `patch -p1 --dry-run < patch` on `/tmp/csm_audit/src/lavis/models/__init__.py` → `checking file ...` `EXIT=0`; real application also succeeded, patched file parses via `ast.parse` — NONE — none.
16. **Patch preserves every original import** — VERIFIED — orig import count 147, new 147; `imports lost: []`, `imports gained: []` — NONE — none.
17. **Patch leaves no missing-module import unguarded** — VERIFIED — `still-unguarded missing-module imports: []`; `guarded missing imports: 98` — NONE — none.
18. **Patch does not change evaluator semantics** — VERIFIED — non-import/non-comment content diff is the empty set; the 98 statements are wrapped in two `try/except ImportError: pass` blocks; the CSMCIR import moves 152 → 168 and stays unconditional. Only `lavis.models`'s own registration side-effects are affected — P2 (operational) — record that applying it *is* an implementation modification; the workbench's policy of not applying it is correct, and the patch is a deliberate human step.
19. **Do the removed names matter anywhere else?** — VERIFIED (no) — `grep -rn 'Blip2QformerCirAlignPromptCaptionQuery|...DaulContrastive|...AlignLoss|...Attentionpooling|...Gatepool|...Aggregator|...Logsumexp|TargetImageCaptionshiyan'` across `*.py` (excluding `models/__init__.py`) returns exactly one hit: `.ipynb_checkpoints/..._shiyan88-checkpoint.py:28` — NONE — none.
20. **Caption path A — dataset reads `qwen_captions`** — VERIFIED — `data_utils_csmcir.py:408` `base_path/'fashionIQ_dataset'/'qwen_captions'/f'{dress_type}_cot_{split}.json'` — NONE — none.
21. **Caption path B — gallery extractor reads `COT_ours2`** — VERIFIED — `utils_csmcir.py:100-101` `os.path.join(base_path,'COT_ours2','fashioniq',f'{dress_type}_cot_val.json')`; called at `validate_blip_csmcir.py:612` — NONE — none.
22. **`FashionIQDataset_imageedit` is dead code** — VERIFIED — defined `data_utils_csmcir.py:175`; `grep -rn FashionIQDataset_imageedit` returns only that definition — NONE — none.
23. **Caption merge order** — VERIFIED — `dress, shirt, toptee` in both consumers (`data_utils_csmcir.py:404-410` via the `dress_types` argument `['dress']` per dataset but **all three** loaded in the extractor at `utils_csmcir.py:97-103`); later file wins on collision — NONE — none.
24. **121 shirt/toptee gallery-id overlap and its effect** — VERIFIED — OBSERVED `|shirt∩toptee| = 121`; `dict.update` in toptee-last order ⇒ those 121 shirt-gallery images get the **toptee** caption; OBSERVED 0/121 have identical `Final_Caption` across the two files — P2 — record in the audit/notes; expect exactly 121 shirt deltas in any per-category caption check.
25. **Gallery construction** — VERIFIED — full ordered `split.{cat}.val.json` (`data_utils_csmcir.py:416-419`), per category (`validate_blip_csmcir.py:594-598, 612`) — NONE — none.
26. **Reference-exclusion policy matches `fashioniq_original_split`** — VERIFIED — `compute_fiq_val_metrics:27-57` never masks the reference; `src/evaluation/fashioniq.py:161-168` has `reference_excluded=False` for this id — NONE — none.
27. **Is the reference kept in the gallery by the code?** — VERIFIED (yes) — `labels` compares only against targets (`:45-49`), and `index_names` is the untrimmed split; the reference is additionally *used* as the query image at `:183-191` — NONE — none.
28. **Recall@K maths** — VERIFIED — `distances = 1 - pred_sim` (`:41`), `argsort` (`:42`), `labels` (`:45-47`), one-hit assert (`:48-49`), `labels[:, :10]`/`labels[:, :50]` (`:52-53`) — NONE — none.
29. **Macro average definition** — VERIFIED — unweighted mean of the three categories; `average_recall = (mean(r50)+mean(r10))/2` (`:620-627`) — NONE — none.
30. **K values used are 10 and 50** — VERIFIED — `:52-53`; matches registry `reported_r10/r50` — NONE — none.
31. **Gallery order / `index_names` alignment** — VERIFIED — `extract_index_blip_caption_features` `index_names.extend(names)` in DataLoader order (`utils_csmcir.py:167`), loader `shuffle` default False (`:88-89`) — NONE — none.
32. **Duplicate ids collapse in `name_to_feat`** — VERIFIED (upstream quirk) — `validate_blip_csmcir.py:168` `dict(zip(index_names, index_features[-1]))`; duplicates within one category are absent here (each split is unique, OBSERVED), so no effect for this model — P3 — none.
33. **One-target-per-query invariant holds** — VERIFIED (static) — `assert torch.equal(sum(labels,dim=-1).int(), ones(len(target_names)).int())` (`:48-49`) would raise otherwise; requires every target id to be in the gallery — NONE — none.
34. **Caption coverage is complete** — VERIFIED (for the staged alias) — OBSERVED `set(qwen_captions/{cat}_cot_val.json) == set(image_splits/split.{cat}.val.json)` for all three categories; may or may not hold for the author's unpublished COT files — P2 — keep the alias labelled a variant.
35. **Missing-caption handler behaviour** — VARIANT — `utils_csmcir.py:153-158`: `except:` re-executes the *same* expression, so a missing id re-raises `KeyError` rather than being skipped; a doc cites this at `:166`, which is wrong (M2) — P3 — fix the citation in `COT_OURS2_AUDIT.md:36-37`.
36. **`except:` bare clauses** — VERIFIED (present) — `utils_csmcir.py:156`, `validate_blip_csmcir.py:648-650` (`vis` block), `data_utils_csmcir.py:480` — P3 — upstream style; if a patched evaluator is ever produced, only the metric-path clause matters.
37. **Image extension contract** — VERIFIED — registry `fashioniq_image_extensions: ['.png']`; source hard-codes `.png` at `data_utils_csmcir.py:449,452,462,468,474` — NONE — none.
38. **Images present with the declared extension** — VERIFIED (filesystem) — `data/fashionIQ_dataset/images` has 74381 files, all `.png`; preflight raised no image-extension block — NONE — none.
39. **Dataset root resolution** — VERIFIED — `base_path` = `__file__.parents[1]` → checkout root (`data_utils_csmcir.py:16`, `utils_csmcir.py:18`); preflight requires `canonical_dataset_link` at that root (currently missing) — NONE — create the link (operator action, not an audit action).
40. **Canonical dataset link** — VARIANT — declared (`canonical_dataset_link: fashionIQ_dataset`) but absent; preflight blocker 5 OBSERVED — NONE — operator stage.
41. **Preprocessing transform** — VERIFIED — `targetpad_transform(1.25, 224)` (`validate_blip_csmcir.py:575-580`), `input_dim = 224` (`:572`) — NONE — none.
42. **Target ratio default** — VERIFIED — `--target-ratio` default `1.25` (`:506`); adapter does not override — NONE — none.
43. **Augmentation in evaluation** — VERIFIED (none) — eval branches apply only `self.preprocess` (`data_utils_csmcir.py:466-470`); `RandomMask/RandomPerspective/GaussianBlur` are train-only objects — NONE — none.
44. **Qwen caption integrity** — VERIFIED — OBSERVED sha256 `b7129baf...` dress (2817386 B), `bbe6f689...` shirt (5015979 B), `acdacc3d...` toptee (3929303 B) on the local files; identical to `auxiliary_assets.yaml`, to the manifest `downloaded_sha256`, and to my recomputation; the HF API lists the same sizes — NONE — none.
45. **COT_ours2 provenance** — UNKNOWN — registry `provenance_status: UNVERIFIED`, `source_type: none`; my HF enumeration (22 siblings) found no `COT_ours2` path; `git ls-files COT_ours2` empty — P0 — keep the model blocked unless an operator supplies bytes with recorded provenance.
46. **Alias staging semantics** — VERIFIED (as a labelled variant) — OBSERVED files at `COT_ours2/fashioniq/*.json` are regular files (not symlinks) whose bytes equal the Qwen digests; preflight emits them as **warnings** "staged as a source-compatible caption alias ... — not official parity"; `source_runtime_artifacts` calls them out as declared runtime artifacts — P2 — no result from an aliased run may be labelled official reproduction.
47. **Checkpoint download URL** — VERIFIED — `https://huggingface.co/peng12138/CSMCIR/resolve/main/fashioniq_tuned_clip_best.pt`; HF API lists the path — NONE — none.
48. **Checkpoint sha256** — VERIFIED — registry `c8812071de0640d647c5ba2a70628b80a3289ee2c47676f7a1b6abaa313fa49e` == HF `lfs.sha256` (OBSERVED) — NONE — none.
49. **Checkpoint size** — VERIFIED — registry `2720615859` == HF `size`/`lfs.size` (OBSERVED) — NONE — none.
50. **Checkpoint present locally / loaded** — BLOCKED — `workbench/artifacts/checkpoints/` empty; only a 1-byte placeholder (`sha256 2d711642...`) at `/tmp/tmpj82v7wks/csmcir/` — P0 — download with digest verification before any run.
51. **Checkpoint → reported metric mapping** — UNKNOWN — provenance chain is metadata-only (paper table = training-log row = registry numbers; one FashionIQ checkpoint published) — P1 — must be shown by a completed run.
52. **Reported metrics vs paper Table 1** — VERIFIED — arXiv v3 Table 1 `CSMCIR(Ours)` = 57.07 / 77.27 / 67.17 (OBSERVED network fetch) — NONE — none.
53. **Reported metrics vs author training log** — VERIFIED — `training_log/fashioniq_validation_metrics.csv` epoch 9 averages 57.066964/77.267204/67.167083 → rounds to the registry values (OBSERVED network fetch) — NONE — none.
54. **Environment spec matches the README** — VERIFIED — registry python 3.9 / torch 2.0.1 / torchvision 0.15.2 == `README.md:7-14` (OBSERVED); `cuda: UNKNOWN` in the registry — P3 — `cuda` unresolved is honest but should be filled once a host is fixed (the registry records `WORKBENCH_CSMCIR_PYTHON`, `confidence: VERIFIED_README`).
55. **Workbench metric parser correctness** — VERIFIED — `parse_csmcir_output` (`metrics_extraction.py:200-281`) expects exactly the six category keys and the three averages the evaluator prints (`validate_blip_csmcir.py:620-627`), re-derives the macro and rejects >0.05 disagreement, then `_check_parity(tolerance=0.05)` — NONE — none.

---

FINAL VERDICT: BLOCKED — the pinned package cannot import (98 unguarded imports of absent modules ahead of the registration, independently reconfirmed), the gallery caption input the evaluator requires was never published, and no checkpoint or GPU run exists, so motion is impossible while the registry, protocol, gallery, metric maths and reported numbers are all faithful to the source.

---

---

## REFRESH (this engagement)

Appended after the work that landed on top of this audit. Earlier findings above are left
as written; where one has been superseded, the superseding fact is stated here.

- **Static benchmark reference.** 1 committed row(s):

  - `csmcir_fiq_val_ours_vitg` — `VERIFIED`, 0% training noise, protocol `fashioniq_original_split`, values `{'r10': 57.07, 'r50': 77.27, 'mean': 67.17}`, provenance `{'r10': 'PAPER_REPORTED', 'r50': 'PAPER_REPORTED', 'mean': 'PAPER_REPORTED'}`

- **Readiness now:** `BLOCKED_SOURCE` — source checkout is dirty: src/data_utils_csmcir.py

- **Reference per registered checkpoint condition:**

  - `fashioniq` (noise 0): `csmcir_fiq_val_ours_vitg` `VERIFIED` — exact condition match

- **Approved source patch:** applied=`True` — src/lavis/models/__init__.py matches the recorded post-patch digest

- **Gate:** the run's verdict comes from `workbench/backend/paper_gate.py` against the
  committed row above, reported in the run JSON under `benchmark_gate`. A row that states no
  condition is refused (`BLOCKED_REFERENCE_INCOMPLETE`); a metric the paper never printed is
  compared only if the registry derives it, and then it is labelled `WORKBENCH_DERIVED`.
