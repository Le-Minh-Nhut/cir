# csmcir

Independent adversarial, READ-ONLY audit of the model registered as `csmcir` in the CIR Failure
Analysis Workbench. Nothing under `/home/heheboiz/data/cir` was written: every mutation in this audit
went to `/tmp/audit-final/csmcir/` (scratch clone of the pinned `src/lavis/models/__init__.py`, a
`cp -a` variant root, a synthetic manifest). Every prior audit document was treated as a hypothesis;
each is re-derived from the pinned source below, and three prior claims turned out to be wrong and are
corrected in MINOR FINDINGS.

Labels: every claim is `STATIC_AUDIT` unless it is a command I actually ran (`OBSERVED`) or an
execution of the official evaluator (`RUNTIME_VERIFIED` — **there are none for this model**: no
checkpoint bytes, no `torch` in the repo interpreter, no GPU run).

---

## MODEL ID

`csmcir` / `method_name: CSMCIR`.
- Registry: `workbench/registry/models.yaml:21-92` — unique entry (`grep -c 'model_id: csmcir'` → `1`).
- Adapter: `workbench/backend/adapters/models.py:28-33` — `CSMCIRAdapter`, `script = "src/validate_blip_csmcir.py"`, command `[python_executable, <source>/script, "--dataset", "fashionIQ", "--blip-model-path", <checkpoint>]`.
- `source_dir: CSMCIR` (`models.yaml:44`), `source_available: true` (`:45`).
- Auxiliary-asset entries: 9 in `workbench/registry/auxiliary_assets.yaml` (Qwen ×3, COT_ours2 ×3, BLIP-2 pretrain, EVA-ViT-g, bert-base-uncased).
- `preparation_contract: manual_standard` (`models.yaml:37`); OBSERVED `grep -c csmcir workbench/registry/preparation_contracts.yaml` → `0` (dangling, P3).

## PAPER TITLE

`CSMCIR: CoT-Enhanced Symmetric Alignment with Memory Bank for Composed Image Retrieval`
(`models.yaml:38`, byte-identical to the pinned README title).
- Repository `paper_url: null` (`models.yaml:39`), `publication_year: null` (`:36`).
- OBSERVED (network): `https://arxiv.org/abs/2601.03728` resolves to that exact title, authors Zhipeng Qian et al., published 2026-01-07. The committed row records the camera-ready: ACL Anthology `2026.findings-acl.216`, pp. 4411-4424 (`papers.yaml:780-868`).
- OBSERVED: pinned `README.md:4` claims "**(ACL206 Findings)**" — a repo-side typo for ACL 2026 Findings.
- Action: fill `paper_url` (`https://arxiv.org/abs/2601.03728`). Severity P3.

## PINNED SOURCE COMMIT

- Declared: `upstream_commit_sha: 774f94e2076ff17ea91703a6239d2a08f0e1a44e` (`models.yaml:42`), `upstream_default_branch: main` (`:41`).
- OBSERVED: `git -C workbench/third_party/CSMCIR rev-parse HEAD` → `774f94e2076ff17ea91703a6239d2a08f0e1a44e`. **Match.**
- OBSERVED (network): `api.github.com/repos/qzp2018/CSMCIR/commits/774f94e2…` → sha confirmed, committer date `2026-04-27T08:30:50Z`, message `'Update README.md'`. So `qzp2018/CSMCIR` is a fork at least as recent as the camera-ready, not an archival snapshot.
- OBSERVED: `runtime.source_clean_and_pinned(model, source)` → `(True, '774f94e2…', None)`; `runtime.source_status(source)` → `{tracked: 0, unauthorized: 0, approved: 6, invalid: 0}`.
- OBSERVED: no file anywhere under `workbench/` hashes to the registered **derived** digest (`aa68fe5e…`); only the **base** digest exists (`9d8c914b…` at the registered checkout). The registry records a variant that is not staged on this host.
- **The checkout is unpatched.** `approved_patch_state` → `applied: false`, detail `"src/lavis/models/__init__.py is the unmodified published file"`.

## CHECKPOINT IDENTITY

`fashioniq_tuned_clip_best.pt` (`models.yaml:78-92`), URL `https://huggingface.co/peng12138/CSMCIR/resolve/main/fashioniq_tuned_clip_best.pt`, `expected_sha256 c8812071de0640d647c5ba2a70628b80a3289ee2c47676f7a1b6abaa313fa49e`, `expected_size_bytes 2720615859`, `checkpoint_training_noise_pct: 0`, `status: URL_VERIFIED`, `checkpoint_mapping_status: VERIFIED_METADATA`, `strict_clean_checkpoint_available: true`.

- OBSERVED (network, live at revision `cf0c19bb346266c295b4b5772ebf03bd1c4f0467`): `fashioniq_tuned_clip_best.pt` → `size 2720615859`, `lfs.sha256 c8812071de0640d647c5ba2a70628b80a3289ee2c47676f7a1b6abaa313fa49e`. **Both match the registry exactly.**
- The HF repo holds exactly three `*_tuned_clip_best.pt` (fashioniq / LaSCo / shoes), so no ambiguity about which file is FashionIQ.
- OBSERVED: `workbench/artifacts/checkpoints/` contains only `.gitkeep`. `evaluate_models.py --model csmcir --dry-run` →
  `[BLOCKED] checkpoint missing: /home/heheboiz/data/cir/workbench/artifacts/checkpoints/csmcir/fashioniq_tuned_clip_best.pt`.
- Consequence: `VERIFIED_METADATA` (author-published bytes), but **no checkpoint was loaded or executed here**. Whether these weights produce 57.07/77.27/67.17 is `UNKNOWN`, not PASS.

## EVALUATION PROTOCOL

Registry: `native_protocol: fashioniq_original_split` (`models.yaml:52`), `supported_protocols: [fashioniq_original_split]` (`:53`), `literature_split_label: original` (`:51`).

Pinned evaluator (`workbench/third_party/CSMCIR/src/validate_blip_csmcir.py`):
- `main()` at `:501`; the `--dataset fashionIQ` branch at `:515-516` dispatches to `blip_validate_fashioniq` (defined `:565`).
- `val_dress_types = ['dress', 'toptee', 'shirt']` (`:592`); per category it builds `FashionIQDataset('val', [dress_type], 'relative', preprocess)` (`:596`) and `('classic')` (`:598`).
- Gallery = the complete ordered `image_splits/split.{cat}.val.json` list (`src/data_utils_csmcir.py:419`), the same list the dataset's `classic` branch indexes (`:468`). No pair union, no dedup beyond the file.
- **Reference stays eligible.** `compute_fiq_val_metrics` (`:24`) builds labels only against `target_names` (`:46-47`) and never masks the reference; `generate_fiq_val_predictions` (`:149`) *adds* the reference embedding to the query (`:191`) without removing it from the candidate pool.

Verdict: registry protocol id, gallery scope, reference policy and category set all trace to source. **VERIFIED** for the declared protocol.

## DATASET CONTRACT

- `fashioniq_layout: fashioniq_standard` (`models.yaml:46-48`), `fashioniq_image_extensions: [".png"]` (`:50`), `canonical_dataset_link: fashionIQ_dataset` (`:33`).
- Source roots are derived from `__file__`, not from a CLI argument: `base_path = Path(__file__).absolute().parents[1].absolute()` (`data_utils_csmcir.py:16`, `utils_csmcir.py:18`) ⇒ `<checkout>/fashionIQ_dataset` and `<checkout>/COT_ours2`.
- Paths the source actually opens: `fashionIQ_dataset/captions/cap.{cat}.val.json` (`:405`), `fashionIQ_dataset/qwen_captions/{cat}_cot_{split}.json` (`:408`), `fashionIQ_dataset/image_splits/split.{cat}.val.json` (`:419`), `fashionIQ_dataset/images/{id}.png` (`:449,:452,:462,:468`, 10 hard-coded `.png` sites listed by `grep -n '\.png'`), `COT_ours2/fashioniq/{cat}_cot_val.json` (`utils_csmcir.py:100`).
- The pinned README's own layout example ships `.jpg` (`README.md:31-33`); the executable opens `.png` (`:468`). The registry reports this honestly rather than hiding it, and `fashioniq_image_extensions` (`models.yaml:50`) drives `fashioniq_layout.image_extension_problems` (`:92-116`), which fails on the laptop instead of on the GPU host. **VERIFIED (mismatch reported, not suppressed).**
- OBSERVED (filesystem): `data/fashionIQ_dataset/{captions,image_splits,images,qwen_captions}` all present; `images/` = 74381 files, 100% `.png`; splits dress 3817 / shirt 6346 / toptee 5373. So the `.png` hard-code is satisfiable on this host.
- OBSERVED: `preflight.model_requirements(csmcir).blocks` includes `"canonical dataset link missing: …/CSMCIR/fashionIQ_dataset -> …/data/fashionIQ_dataset"`. The link is declared but **not staged**. Staging is `prepare_dataset.py:65 destination.symlink_to(root, …)`, gated on the declaration at `registry.py:600-609` (`kind: canonical_dataset_link`), and verified by `runtime._verify_declared_artifact` (`:1989-2000`) which requires a symlink resolving exactly to `FASHIONIQ_ROOT`. **VARIANT** (declared, correct mechanism, not yet staged) — P2.
- Frontend/API consistency check: `main.py:261` targets an explicit in-checkout path for csmcir only (`config.WORKBENCH_THIRD_PARTY_ROOT / "CSMCIR" / "fashionIQ_dataset"`), which is precisely the declared link. No `/CSMCIR/CSMCIR` double-join anywhere on this path. **VERIFIED.**

## PREPROCESSING CONTRACT

- Adapter command omits `--blip-model-name`; `validate_blip_csmcir.py:504` sets `default="blip2_cir_align_prompt_csmcir"`, the same string the class registers under (`blip2_qformer_cir_align_prompt_csmcir.py:27`). Harmless.
- `--transform` default `targetpad` (`:507`), `--target-ratio` default `1.25` (`:506`), `input_dim = 224` (`:574`) → `targetpad_transform(1.25, 224)` (`:580`).
- Eval path applies only that transform: `classic` branch returns `self.preprocess(PIL.Image.open(path))` (`data_utils_csmcir.py:466-470`); no randomised op.
- Registry row and `papers.yaml:802-804` agree: `TargetPad target_ratio=1.25 at 224x224`.
- **VERIFIED** (STATIC_AUDIT; deterministic transform, adapter overrides nothing).

## AUGMENTATION CONTRACT

- Training-only augment objects exist in the module, but the validation `classic`/`relative` branches never apply them.
- Registry declares no augmentation policy (`papers.yaml:805 augmentation_policy: null`) and `evaluation_noise_pct: 0` (`models.yaml:86`).
- **VERIFIED / NONE** (STATIC_AUDIT; not executed).

## GALLERY CONTRACT

- Per category the gallery is the full ordered val split **including the reference** (`data_utils_csmcir.py:419`, `:468`; `validate_blip_csmcir.py:598, :612`).
- `index_names` order equals split order; `generate_fiq_val_predictions` builds `name_to_feat = dict(zip(index_names, index_features[-1]))` (`:169`) — a dict, so a duplicate id inside one category collapses. OBSERVED each split is internally unique, so no effect for this model.
- **121-id shirt/toptee overlap — re-derived independently, OBSERVED:** `|set(split.shirt.val) ∩ set(split.toptee.val)| = 121`, and of those, **0** have an identical `Final_Caption` across the two published Qwen files. `extract_index_blip_caption_features` loads all three `COT_ours2` files in `dress, shirt, toptee` order and merges with `dict.update` (`utils_csmcir.py:97-103`), so toptee wins on collision and 121 shirt-gallery images are captioned as tops. Upstream behaviour, present with or without the alias.
- **VERIFIED** for gallery composition and reference policy; the 121-id effect is a documented source-level property (**P2** note — it must be reported, and any per-category caption check must expect exactly 121 shirt deltas).

## METRIC CONTRACT

- `compute_fiq_val_metrics` (`:24-53`): `distances = 1 - pred_sim` (`:41`), `argsort` (`:42`), `labels` vs `target_names` (`:45-47`), one-hit-per-query assert (`:48-49`), `labels[:, :10]` / `labels[:, :50]` over `len(labels)` (`:52-53`). No reference masking.
- Macro emitted as JSON at `:620-626` (`{cat}_recall_at10/at50_`, `average_recall_at10`, `average_recall_at50`, `average_recall = (mean(r50)+mean(r10))/2`) and printed (`:628`).
- Workbench parser `parse_csmcir_output` (`workbench/backend/metrics_extraction.py:219-303`) requires exactly those six category keys plus the three averages, validates 0–100, re-derives the macro and rejects disagreement > 0.05. OBSERVED: that 0.05 is an **internal self-consistency** check (the run's own declared aggregate vs its own categories), not a paper comparison — the comment at `:277-280` says so. It cannot contradict the gate.
- `_check_parity` (`:28-62`) now routes through `paper_gate.compare_metric` (5 percent **relative**) and refuses a partial paper row, so `parity_status` can no longer disagree with `benchmark_gate`. **VERIFIED** (matches the re-check list item 2).
- Recall convention sanity: the paper's printed `Average` column is the mean of the category R@10 and R@50 columns (`papers.yaml:809-811` documents this), so `reported_mean = (r10+r50)/2` is the right reading and not a `(R@1+R@10+R@50)/3` Shoes-style mean.
- **VERIFIED** (STATIC_AUDIT; arithmetic traced to source, never executed).

## SOURCE-TO-WORKBENCH PARITY

| Item | Upstream source | Workbench | Verdict |
|---|---|---|---|
| entrypoint | `src/validate_blip_csmcir.py` (adapter `script`, `adapters/models.py:30`) | adapter passes it | VERIFIED |
| **README entrypoint name** | `README.md:115` documents `blip_validate_csmcir.py`, which **does not exist**; the shipped file is `validate_blip_csmcir.py` | adapter uses the file that exists | VERIFIED (upstream doc bug; M1) |
| model name | `validate_blip_csmcir.py:504` default = `blip2_cir_align_prompt_csmcir` = registration at `blip2_qformer_cir_align_prompt_csmcir.py:27` | adapter omits the flag → default used | VERIFIED |
| checkpoint flag | `--blip-model-path` (`:505`); README says `--model-path` (`README.md:118`) | adapter uses the working spelling | VERIFIED (README typo) |
| protocol | full split, reference eligible | `fashioniq_original_split` | VERIFIED |
| metric keys | `:620-626` | parser expects exactly those | VERIFIED |
| caption path A (query text) | `data_utils_csmcir.py:408` `qwen_captions/` | registry `auxiliary_assets.yaml` Qwen ×3 | VERIFIED |
| caption path B (gallery features) | `utils_csmcir.py:100` `COT_ours2/fashioniq/` | registry COT_ours2 ×3, `provenance_status: UNVERIFIED` | VERIFIED (unpublished input, correctly blocked) |
| importability | 98 unguarded imports of absent modules before the registration at `:152` | workbench reports and does not patch in place | VERIFIED (defect real; see CRITICAL) |
| `text_cot` parameter | declared `blip2_qformer_cir_align_prompt_csmcir.py:411`, never read inside `inference` | no doc claims otherwise | VERIFIED (dead parameter; M9) |

## RUNTIME STATUS

- `reproduction_status: NOT_RUN` (`models.yaml:56`); no result artifact except a mock
  (`workbench/artifacts/results/fashioniq_original_split/mock/mock_csmcir-fiq_a.json`); `workbench/artifacts/` has no `reports/` directory.
- OBSERVED `preflight.preflight_blockers(csmcir)` — five blockers, first one first:
  1. `source cannot run: pinned lavis.models package cannot be imported as published (98 unguarded imports of absent modules sit before the registration); an approved patch is declared in approved_source_patches and removes only that blocker; src/lavis/models/__init__.py is the unmodified published file; apply workbench/patches/csmcir_models_init_compat.patch to a derived checkout of the pinned revision and register that checkout as the model source`
  2. `CSMCIR LAVIS BLIP-2 pretrain weights unverified: ~/.cache/lavis/blip2_pretrained.pth`
  3. `CSMCIR EVA-ViT-g backbone weights unverified: ~/.cache/torch/hub/checkpoints/eva_vit_g.pth`
  4. `CSMCIR bert-base-uncased tokenizer unverified: ~/.cache/huggingface/hub/models--bert-base-uncased/vocab.txt`
  5. `canonical dataset link missing: …/CSMCIR/fashionIQ_dataset -> …/data/fashionIQ_dataset`
- OBSERVED warnings (non-blocking): all three `COT_ours2` files `staged as a source-compatible caption alias: … — not official parity`.
- OBSERVED `pc_readiness.py --model csmcir --json`: `state: BLOCKED_SOURCE`, plus
  `gpu: {"torch": null, "torch_error": "ModuleNotFoundError: No module named 'torch'", "devices": ["NVIDIA GeForce RTX 3050 Laptop GPU, 8.6, 4096 MiB"]}` — the repo interpreter has no torch, so no evaluator could run from here even with weights.
- OBSERVED `evaluate_models.py --model csmcir --dry-run`: `[BLOCKED]` ×3 (source, link, checkpoint), exit 1.
- OBSERVED: the only csmcir auxiliary receipt is `workbench/artifacts/receipts/auxiliary-assets_csmcir.json` → `return_code: 0` for `download_auxiliary_assets.py --model csmcir --stage-caption-aliases`.

## PAPER-METRIC STATUS

- Registry: `reported_r10 57.07`, `reported_r50 77.27`, `reported_mean 67.17` (`models.yaml:54-56`); category row in `papers.yaml:812-826` (dress 52.45/74.81, shirt 57.70/75.76, toptee 61.14/80.98).
- OBSERVED (network): arXiv v3 HTML table row extraction →
  `['52.45','74.81','57.70','75.76','61.14','80.98','57.07','77.27','67.17']`; the `<td>` cells sit in the row `CSMCIR(Ours)` under the printed header `Dress | Shirt | Toptee | Average`, columns `R@10 R@50 R@10 R@50 R@10 R@50 R@10 R@50 Avg.`. The paper's per-column means are 57.097 / 77.183 (the authors' own rounding); the printed Average row is what the registry transcribes.
- OBSERVED (network): author training log `training_log/fashioniq_validation_metrics.csv`, epoch 9 → dress 52.35498547554016/75.06197094917297, toptee 61.142271757125854/80.97909092903137, shirt 57.70363211631775/75.7605493068695, averages 57.06696311632792/77.26720372835796/67.16708342234294 → rounds to **57.07 / 77.27 / 67.17**.
- The committed row's condition (`papers.yaml:790-801`): checkpoint `fashioniq`, training noise 0, evaluation noise 0, protocol `fashioniq_original_split`, split `val`, `reference_excluded: false`, `reference_status: VERIFIED`. `benchmark_references.select_reference` returns it with `reason: "exact condition match"` (OBSERVED).
- **Verdict: VERIFIED (the numbers and the row's condition), UNKNOWN (reproduction).** No run exists, so "this checkpoint reproduces 57.07/77.27/67.17" is unproven.

---

## CRITICAL FINDINGS

### C1 — The pinned package cannot register its own model (P0, independently reconfirmed)

Recomputed against the pinned checkout, not read from a document:

- OBSERVED: 98 uncommented `from lavis.models.…` imports in `src/lavis/models/__init__.py` whose target `.py` does not exist; line range **44–151**; the file is 402 lines. Independently cross-checked against the GitHub recursive tree at the pin (948 entries): 98 of those imports are absent there too, 27 present.
- OBSERVED: `any(try:/except in lines 40-155)` → none. The first missing module raises `ModuleNotFoundError`.
- OBSERVED: line 152 is `from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_csmcir import Blip2QformerCirAlignPromptCSMCIR` — the import the evaluator needs sits **after** all 98 failures.
- OBSERVED: the class exists and is registered: `blip2_qformer_cir_align_prompt_csmcir.py:27` → `@registry.register_model("blip2_cir_align_prompt_csmcir")`; `validate_blip_csmcir.py:12` → `from lavis.models import load_model_and_preprocess`, called at `:568`.
- Impact: the official evaluator cannot start; and because the registration never executes, `load_model_and_preprocess(name="blip2_cir_align_prompt_csmcir")` would fail even if the imports were bypassed. **VERIFIED, P0** — upstream defect, correctly reported, not hidden.

### C2 — The gallery caption input the evaluator reads was never published (P0)

- Executing reader: `utils_csmcir.py:100-103` opens `<checkout>/COT_ours2/fashioniq/{cat}_cot_val.json` unconditionally for any `FashionIQDataset`, loading all three categories regardless of which one the dataset holds.
- Called from the Recall path: `validate_blip_csmcir.py:612` (metrics; also `:636` for the visualisation block).
- Registry declares `source_type: none`, `source_repo: null`, `source_revision: null`, `source_path: null`, `automatic_download_available: false`, `provenance_status: UNVERIFIED`, `expected_sha256: null` (`auxiliary_assets.yaml`, `family: COT_ours2 captions`).
- OBSERVED: the author's HF repo at the pinned revision has 22 siblings and **no `COT_ours2` path**; the GitHub tree at the pin has no `COT_ours2` path; `git ls-files COT_ours2` in the checkout is empty (the local directory is untracked).
- **VERIFIED, P0.** Scoped to the public trees queried (HF current revision, GitHub pin); HF history/deleted uploads were not examined. The workbench keeps the model blocked by default for exactly this reason.

### C3 — No checkpoint, no torch, no run (P0 for reproduction)

`workbench/artifacts/checkpoints/` holds only `.gitkeep`; the repo interpreter has no `torch`; the only csmcir result file is a mock. `reproduction_status: NOT_RUN` is honest. **NOT_RUN.**

---

## MINOR FINDINGS

- **M1 (P2) — Corrected path instruction is right, and two docs still carry the old one.** The corrected instruction is `PC_TRANSFER_CHECKLIST.md:127-138`: build `cp -a workbench/third_party/CSMCIR /path/to/variant-root/CSMCIR`, apply the patch there, and point the config's `third_party` at the **parent directory** (`/path/to/variant-root`), because the registry declares `source_dir: CSMCIR` and every resolver computes `<third_party>/<source_dir>`. OBSERVED: `registry.py:563-609` emits the declaration, and my simulated variant root at `/tmp/audit-final/csmcir/sim/variant-root/CSMCIR` was accepted with `applied: true` — the parent-directory form is the working one. Still stale: `PC_RELEASE_READINESS.md:38` ("apply the approved patch to a derived checkout and point the config at it" — does not say *parent*) and `PC_TRANSFER_CHECKLIST.md:53`, which still describes the COT path as "author CoT captions, no published URL" in the manual-action table while `:142-154` says the same path is available as an opt-in alias. Fix both. P2.
- **M2 (P2) — "Tracked edit" is stricter than the rule as written.** `runtime.source_clean_and_pinned` (`:2055-2078`, `:2060-2070`) returns dirty with the message `"source checkout is dirty: …"`. For an approved-patch checkout this is reached only when the *same* file hash-mismatches (already reported as invalid by `runtime.py:2002-2015` and `preflight.approved_patch_state:154-157`), so no information is lost. But the message an operator reads if a sibling file is ever tracked-edited says "dirty", not "unauthorized/unapproved", which is the wrong instruction. P2 (message quality, not semantics).
- **M3 (P3) — `source_status` lists each approved path twice.** OBSERVED on the live checkout: `approved` = the three `COT_ours2` files **twice each** (6 entries for 3 files). Cause: `runtime.py:1857-1907` scans tracked lines and untracked+ignored lines, while `_porcelain("--untracked-files=all")` already includes ignored-adjacent untracked paths; and `_recorded_asset_digest` (`:1938-1960`) matches by resolved destination, so the same manifest entry is found either way. Cosmetic count inflation in an audit field — fix by de-duplicating the final list. P3.
- **M4 (P2) — The patch doc and the real patch differ.** `workbench/patches/csmcir.md` shows `@@ -41,114 +41,130 @@` and describes deleting lines 44-151 "keeping line 152". The shipped file is stricter: it uses four `try/except ImportError: pass` blocks, preserves every original import line verbatim inside them, and moves the registration import from line 152 to line 168. Consequences: (a) the doc's header line numbers are the document's own, not the patch's; (b) `workbench/configs/pc_autopilot.pilot_csmcir.yaml:14-16` cites `extract_index_blip_caption_features (src/validate_blip_csmcir.py:618)` — the real call is at **:612**. P2.
- **M5 (P2) — Applying the patch is always an implementation modification, and the workbench says so.** OBSERVED warnings on a patched variant: `"source repaired by the approved patch … — the published-source blocker no longer applies and the run is an approved compatibility variant, not official parity"` and `preflight.py:50-53` `notes.append(...)`. OBSERVED `pc_readiness.py:229-234`: an otherwise-ready patched model becomes `READY_WITH_DECLARED_VARIANT`, never `READY_FOR_FULL_EVALUATION`. Correct and consistent. Recorded here rather than filed as a defect.
- **M6 (P3) — The variant instruction in `PC_TRANSFER_CHECKLIST.md:127-130` begins with `git -C workbench/third_party/CSMCIR checkout --detach …`.** That mutates the **pinned** checkout (harmless while the SHA is unchanged, but the doc's own comment two lines above says "The pinned tree stays untouched"). Prefer `git -C … worktree add --detach <path> 774f94e2…` or clone. P3.
- **M7 (P2) — `READINESS_MATRIX.md:25` is stale.** It reports csmcir's blocker chain as "Qwen captions missing" ×3 plus "COT_ours2 captions missing" ×3, but OBSERVED the Qwen files exist and verify (`status: VERIFIED`) and the staged aliases resolve the COT path (now a warning). The matrix's rendered Markdown is a snapshot with no generator timestamp, while `pc_readiness --json` (`state: BLOCKED_SOURCE`, five blockers, all three aliases `OPERATOR_SUPPLIED_UNVERIFIED` + warning) is current. Regenerate. P2.
- **M8 (P3) — `data_utils_csmcir.py:223,226` (`FashionIQDataset_imageedit`) is dead code.** It is the only reader of `COT_ours2/bert_captions/fashioniq/` and `captions/keywords_in_mods_*`; nothing instantiates it, so those reads are never required at evaluation. The prior review's M3 claim is correct.
- **M9 (P3) — `text_cot` is declared and never read** in `blip2_qformer_cir_align_prompt_csmcir.py:411` (and `:454`). Any sentence claiming "COT captions feed the query" would be wrong: COT drives gallery features only, while the query text comes from the dataset's `cap.{cat}.val.json` captions. The existing docs get this right.
- **M10 (P3) — `paper_url: null`** with a public, verifiable paper (arXiv abs + ACL Anthology camera-ready). Populate it.
- **M11 (P3) — `preparation_contract: manual_standard` (`models.yaml:37`) names no registered contract** (`grep -c csmcir workbench/registry/preparation_contracts.yaml` → 0). Preflight silently falls back to the layout list (`preflight.py:22-31`), while the two caption directories that a real run needs are enforced through `auxiliary_assets.yaml` instead. Works, but it is two mechanisms for one requirement; either register the contract or drop the field.
- **M12 (P3) — The alias copy is safe against symlink and operator-file replacement.** `download_auxiliary_assets.py:202-213`: refuses a symlinked destination, refuses to overwrite a file the manifest does not own, copies to a temp file, re-hashes, then `os.replace`. `registry.caption_alias_evidence:514-556` additionally requires the staged bytes to equal the registry's published digest, so a hand-written manifest cannot relabel arbitrary bytes as a variant. **Verified design, no defect** — recorded because the alias is the one input whose identity is genuinely unknown.

---

## UNRESOLVED EVIDENCE

- **U1 — Whether the author's own `COT_ours2/fashioniq/{cat}_cot_val.json` held the published Qwen bytes.** No copy reachable; not in the HF tree at the pinned revision, not in the GitHub tree at the pin, not committed locally. The verdict `SOURCE_COMPATIBLE_EQUIVALENCE_UNKNOWN` is the correct one on the evidence available. OBSERVED: the three staged files are regular files (not symlinks) whose digests equal the published Qwen digests, which equal the registry's alias `expected_sha256` and the manifest's `downloaded_sha256`. No construction argument was found that would tie them to the author's bytes without executing the evaluator.
- **U2 — Whether `fashioniq_tuned_clip_best.pt` reproduces 57.07/77.27/67.17.** Metadata agrees with the paper and with the author's training log; the weights were never loaded (none on disk, no torch here). `UNKNOWN`.
- **U3 — Whether the paper's reported number used the qwen-caption (aliased) set.** The training script's validation path does call `extract_index_blip_caption_features` (`blip_fine_tune_2_csmcir.py:268`), i.e. a path that reads `COT_ours2`; the CSV carries no path, revision or hash metadata. Not decidable from the repository.
- **U4 — HF history / deleted uploads.** Not examined; C2's "never published" claim is scoped to the current public trees.
- **U5 — One-off `.pyc` hygiene.** The pinned checkout contains `__pycache__` (498 committed `.pyc` per `runtime.py:1880-1883`, and 243 untracked/ignored entries OBSERVED at the checkout) and the public tree still ships committed `.pyc`. `runtime.source_status` deliberately exempts bytecode. It is a real risk class (a tampered `.pyc` over an unchanged `.py`), reported as engineering debt; no tampering was observed. UNKNOWN.
- **U6 — Cross-document completeness.** I re-derived every csmcir-specific statement in `COT_OURS2_AUDIT.md`, `AUXILIARY_ASSET_AUDIT.md` (§csmcir), `UPSTREAM_AUDIT.md`, `PROTOCOL_AUDIT.md`, `PC_TRANSFER_CHECKLIST.md`, `FIRST_GPU_EXPERIMENT.md`, `FINAL_PC_READINESS.md`, `PC_RELEASE_READINESS.md`, `FINAL_REPRODUCIBILITY_AUDIT.md`, `OPEN_BLOCKERS.md`, `workbench/patches/csmcir.md`, `UPSTREAM_DATA_PREPARATION_AUDIT.md`, `SOURCE_TO_WORKBENCH_PARITY_MATRIX.md`, `RESULT_SCHEMA.md`, `workbench/README.md`, `workbench/configs/pc_autopilot.pilot_csmcir.yaml`, and the two registries. Statements about *other* models were not re-derived.

---

## FINAL VERDICT

**VARIANT.** The model's reproduction path is a correctly engineered, correctly labelled declared variant: the pinned package genuinely cannot import (98 unguarded imports of modules absent from both the local checkout and the public tree, all before the registration at line 152), the registry declares exactly one base/derived-digest-checked patch for that, preflight accepts the derived file and refuses anything else, the unpublished `COT_ours2` gallery-caption input is substituted only through an explicitly opted-in, digest-bound, manifest-recorded, regular-file alias, and both `variant_provenance` and the gate then label any result accordingly (`PASS_WITHIN_5_PERCENT_VARIANT`, never bare `PASS_WITHIN_5_PERCENT`) while a deviation still fails. No completed run exists and no document claims official parity, so the verdict is not VERIFIED; the model is not BLOCKED in the workbench sense either, because the source defect has a reviewed, provenance-tracked repair path that I exercised end to end in scratch. The residual gaps are real but honestly recorded: the author's COT caption bytes are unidentifiable, the checkpoint has never been loaded, and the corrected parent-directory instruction in `PC_TRANSFER_CHECKLIST.md:127-138` is not yet propagated to the two docs that still say "point the config at it".

---

# EXECUTION TRACE

`paper/README/config → source dataset → preprocessing → official evaluator → workbench invocation → checkpoint → output → metric comparison`

1. **paper/README/config →** registry row `models.yaml:21-92`; committed benchmark row `papers.yaml:780-868` (`reference_status: VERIFIED`, `review_status: CONFIRMED`); OBSERVED paper HTML v3 row `CSMCIR(Ours)` = `52.45 74.81 57.70 75.76 61.14 80.98 57.07 77.27 67.17`; OBSERVED author training-log epoch 9 averages = `57.066963…/77.267204…/67.167083…`. Pinned `README.md:7-16` (python 3.9 / torch 2.0.1 / torchvision 0.15.2) matches registry `environment` (`models.yaml:22-30`).
2. **→ source dataset** — `base_path = Path(__file__).absolute().parents[1]` (`data_utils_csmcir.py:16`, `utils_csmcir.py:18`) ⇒ `<checkout>/fashionIQ_dataset` + `<checkout>/COT_ours2`; declared link `models.yaml:33` **not staged** (preflight block 5); canonical root OBSERVED at `data/fashionIQ_dataset` with 74381 `.png` images and splits 3817/6346/5373; `/api/evaluate` uses the same in-checkout path (`main.py:261`).
3. **→ preprocessing** — `targetpad_transform(1.25, 224)` (`validate_blip_csmcir.py:506,:574,:580`), applied per image in `__getitem__` (`data_utils_csmcir.py:466-470`); no augmentation on the eval path.
4. **→ official evaluator** — `main()` `:501`, `--dataset fashionIQ` → `blip_validate_fashioniq` `:515-516`/`:565`; loads the model at `:568` through `from lavis.models import …` (`:12`) — **this import fails** (C1). Gallery `:594-599`, gallery caption features `:612` (reading `COT_ours2/…` at `utils_csmcir.py:100`), metrics `:613`/`compute_fiq_val_metrics:24-53`, JSON at `:620-628`.
5. **→ workbench invocation** — `CSMCIRAdapter.command` (`adapters/models.py:32-33`) → `python src/validate_blip_csmcir.py --dataset fashionIQ --blip-model-path <checkpoint>`; cwd `source/src` (`evaluate_models.py:185-186`); `variant_provenance` computed at `evaluate_models.py:243-273` and attached at `:357`; gate `run_meta` at `:137-148`. Blocked at `preflight_blockers` (`preflight.py:162`) and by `runtime.runtime_blockers`.
6. **→ checkpoint** — HF API: `size 2720615859`, `lfs.sha256 c8812071…` == registry `expected_sha256`/`expected_size_bytes` (`models.yaml:89-90`). No local file: `artifacts/checkpoints/` is empty; dry run reports `checkpoint missing`.
7. **→ output** — none. `reproduction_status: NOT_RUN`; only `results/fashioniq_original_split/mock/mock_csmcir-fiq_a.json` exists; no `artifacts/reports/`.
8. **→ metric comparison** — not reached. The machinery is source-correct and I exercised it with synthetic inputs: `paper_gate.evaluate_gate` returned `PASS_WITHIN_5_PERCENT` plain, `PASS_WITHIN_5_PERCENT_VARIANT` with `caption_alias_staged` or `declared_variant`, and `ALERT_METRIC_DEVIATION` (`passed: False`) on a 60%+ deviation even with both variant keys set. **PARITY_NOT_EVALUATED** for the real model.

Every arrow from step 4 onward is blocked at the import plus the missing local assets. Nothing downstream is RUNTIME_VERIFIED. The variant path was exercised only in `/tmp` scratch (`/tmp/audit-final/csmcir/sim/`), never against the registered checkout.

---

# CHECKLIST

1. **Registry entry exists and is unique** — VERIFIED — `models.yaml:21-92`; `grep -c 'model_id: csmcir'` → 1 — NONE — none.
2. **method_name / paper title recorded** — VERIFIED — `models.yaml:35,38`; identical to arXiv 2601.03728 — NONE — none.
3. **paper_url resolvable** — VARIANT — `models.yaml:39` is `null`; paper is public (`arxiv.org/abs/2601.03728`; ACL `2026.findings-acl.216`) — P3 — populate `paper_url`.
4. **upstream_repo_url recorded and live** — VERIFIED — `models.yaml:40`; GitHub API returns the pin — NONE — none.
5. **Pinned commit matches the checkout** — VERIFIED — `git rev-parse HEAD` → `774f94e2076ff17ea91703a6239d2a08f0e1a44e` — NONE — none.
6. **Checkout clean and pinned (workbench's own check)** — VERIFIED — `source_clean_and_pinned` → `(True, '774f94e2…', None)`, `source_status` tracked 0 / unauthorized 0 / invalid 0 — NONE — none.
7. **Entrypoint file exists** — VERIFIED — `third_party/CSMCIR/src/validate_blip_csmcir.py`; adapter `script` matches (`adapters/models.py:30`) — NONE — none.
8. **Pinned README documents the same entrypoint** — VARIANT — `README.md:115` names `blip_validate_csmcir.py`, which does **not** exist; the shipped file is `validate_blip_csmcir.py` — P3 — note the upstream doc bug (M1).
9. **Entrypoint imports the model package** — VERIFIED — `validate_blip_csmcir.py:12` — NONE — none.
10. **98 unguarded absent-module imports (own count)** — VERIFIED — my script: `MISSING 98 … missing range 44 151`; `wc -l` = 402 — P0 — none (upstream defect, correctly reported).
11. **Are any of those imports guarded?** — VERIFIED (no) — lines 44-151 contain zero `try:`/`except` — P0 — none.
12. **Do the failures precede the needed registration?** — VERIFIED — registration import at line 152; `max(missing)=151 < 152` — P0 — none.
13. **Absent modules are also absent upstream** — VERIFIED — GitHub recursive tree at the pin (948 entries): 98 of those imports have no target, 27 do — P0 — none.
14. **The class the evaluator needs is present and registered** — VERIFIED — `blip2_qformer_cir_align_prompt_csmcir.py:27-28` — NONE — none.
15. **The adapter asks for that name** — VERIFIED — adapter omits `--blip-model-name`; default at `:504` is the registered string — P3 — pass it explicitly (M4 class).
16. **Patch file digest matches the registry** — VERIFIED — `sha256sum workbench/patches/csmcir_models_init_compat.patch` → `0e703584…` == `models.yaml:72 patch_sha256` — NONE — none.
17. **Registered base digest matches the pinned file** — VERIFIED — `sha256sum src/lavis/models/__init__.py` → `9d8c914b…` == `models.yaml:73 base_sha256` — NONE — none.
18. **Registered derived digest is reproducible from base + patch** — VERIFIED — `patch derived__init__.py < p.patch` → `aa68fe5e…` == `models.yaml:74 derived_sha256`; repeated through the documented `cp -a` + `patch -p1 --directory <variant>/CSMCIR` flow with the same result — NONE — none.
19. **Patch changes only the declared import registration set** — VERIFIED — AST: base has 98 top-level lavis.models imports inside the patched region, derived has 98 inside `try:` bodies and 0 at top level; `guarded-set == removed-set` True; non-import/non-comment statements 332 → 344 (the 4 `try:` + 4 `except` + 4 `pass`, minus 12 commented lines now counted) — P2 — record that applying it *is* an implementation modification.
20. **Patch preserves every original import statement** — VERIFIED — every one of the 98 is preserved verbatim inside the guards; nothing added outside guards; the CSMCIR registration import moves 152 → 168 — NONE — none.
21. **Do the removed names matter elsewhere?** — VERIFIED (no) — no importer of those 98 classes outside `models/__init__.py` — NONE — none.
22. **Unpatched checkout is refused with the publishing explanation** — VERIFIED — `approved_patch_state` `applied: false`, detail `"…is the unmodified published file; apply … to a derived checkout"`; blocker 1 of `preflight_blockers` — NONE — none.
23. **Anything other than base or derived is refused** — VERIFIED — `runtime.py:2002-2015` returns `"modified source in an approved-patch checkout: the file matches neither the published base nor the recorded post-patch digest"`; `preflight.py:154-157` reports the same — NONE — none.
24. **Patched derived checkout is accepted as a variant, not as parity** — VERIFIED (OBSERVED, scratch) — `WORKBENCH_THIRD_PARTY_ROOT=<variant-root>`: `approved_patch_state.applied: true`; blocker 1 disappears; warning `"…approved compatibility variant, not official parity"`; `pc_readiness` → `READY_WITH_DECLARED_VARIANT` when otherwise ready — NONE — none.
25. **Corrected path instruction: variant checkout at `<third_party_root>/CSMCIR`** — VERIFIED — `PC_TRANSFER_CHECKLIST.md:127-138` says `cp -a … /path/to/variant-root/CSMCIR`, patch there, and point `third_party` at the **parent**; `registry.py:563-609` + `registry.py:600-609` emit `source_dir: CSMCIR`; my scratch variant root at `variant-root/CSMCIR` was the form accepted — P2 — propagate the same wording to `PC_RELEASE_READINESS.md:38` and fix stale `PC_TRANSFER_CHECKLIST.md:53`.
26. **A derived checkout exists in the registered location** — BLOCKED — only the base digest exists under `workbench/`; `approved_patch_state.applied: false` — P0 — build the variant root before any run.
27. **Alias staged at the destination the evaluator reads** — VERIFIED — `COT_ours2/fashioniq/{dress,shirt,toptee}_cot_val.json` present (`utils_csmcir.py:100`) — NONE — none.
28. **Alias files are real files, not symlinks** — VERIFIED — `os.path.islink` → False for all three; `record_alias` writes via `os.replace` after a re-hash (`download_auxiliary_assets.py:202-213`) — NONE — none.
29. **Alias bytes equal the published digest** — VERIFIED — `b7129baf…` (2817386 B), `bbe6f689…` (5015979 B), `acdacc3d…` (3929303 B); equal to the Qwen files, to `auxiliary_assets.yaml` `caption_alias_from.expected_sha256`, and to the manifest `downloaded_sha256` — NONE — none.
30. **Alias provenance recorded as a labelled variant** — VERIFIED — manifest entries carry `staged_as: SOURCE_COMPATIBLE_CAPTION_ALIAS`, `alias_source_path`, `alias_source_sha256`, `source_revision cf0c19bb…`, `official_hash_available: false`, `official_sha256: null`; `registry.caption_alias_evidence:514-556` re-verifies both the manifest digest and the published digest and returns `equivalence: SOURCE_COMPATIBLE_EQUIVALENCE_UNKNOWN` — NONE — none.
31. **Alias is reported as a warning, never as official parity** — VERIFIED — OBSERVED three warnings `"…staged as a source-compatible caption alias: … — not official parity"`; no result may be labelled official (documented `COT_OURS2_AUDIT.md:9`, `PC_TRANSFER_CHECKLIST.md:137`, `workbench/README.md:83`) — NONE — none.
32. **An operator's own file is never accepted as an alias** — VERIFIED — `caption_alias_evidence` returns None without a manifest record, and `preflight.py:82-83` then blocks with `"operator-supplied unverified provenance"`; `stage_caption_alias:204-206` refuses to overwrite a file the manifest does not own — NONE — none.
33. **Changing the staged bytes stops the variant** — VERIFIED — `registry.py:534-545` requires `actual == recorded` and `actual == published`; the alias is inside the stage fingerprint (`pipeline.model_eval_inputs`) so resume is invalidated — NONE — none.
34. **Variant provenance reaches evaluation reports** — VERIFIED — `evaluate_models.py:243-273` (`_variant_provenance`), attached at `:357`, serialized at `:1427`; gate `run_meta` carries `caption_alias_staged`/`declared_variant` (`:146-147`) — NONE — none.
35. **Gate labels a variant pass as a variant** — VERIFIED (OBSERVED, synthetic) — `caption_alias_staged` → `PASS_WITHIN_5_PERCENT_VARIANT`; `declared_variant` → same; plain → `PASS_WITHIN_5_PERCENT`; `paper_gate.py:83,370-376` — NONE — none.
36. **Gate still fails a deviation in variant mode** — VERIFIED (OBSERVED, synthetic) — 60%+ deviation → `ALERT_METRIC_DEVIATION`, `passed: False`, `variant: True`, blockers naming each metric — NONE — none.
37. **Rows carry the gate and variant provenance** — VERIFIED — `autopilot.py:844-845` and `:961-962`; `/api/benchmark-runs` + `/api/runs/gate` serve them (`main.py:122-133`); frontend maps `caption_alias_staged` → amber `CAPTION THAY THẾ` (`frontend/gate.ts:46-49`) — NONE — none.
38. **A metric deviation never demotes run status** — VERIFIED — `autopilot.py` rows carry `benchmark_gate` separately from `status`; OBSERVED the gate returns `passed: False` with `variant: True` rather than rewriting the run state — NONE — none.
39. **Caption path A (query text) reads `qwen_captions`** — VERIFIED — `data_utils_csmcir.py:408`; files present and `status: VERIFIED` in preflight — NONE — none.
40. **Caption path B (gallery features) reads `COT_ours2`** — VERIFIED — `utils_csmcir.py:100`; called at `validate_blip_csmcir.py:612` — NONE — none.
41. **COT_ours2 has no published source** — VERIFIED — `auxiliary_assets.yaml` `source_type: none`/`provenance_status: UNVERIFIED`; HF tree at the pin has no such path; GitHub tree at the pin has none — P0 — keep the model blocked unless an operator supplies bytes with recorded provenance.
42. **Published caption coverage is exact** — VERIFIED — OBSERVED `missing 0 extra 0` and `Final_Caption` non-empty for all of dress 3817 / shirt 6346 / toptee 5373 against the declared splits — NONE — none.
43. **121 shirt/toptee overlap and its effect** — VERIFIED — OBSERVED `|shirt ∩ toptee| = 121`, `shared ids with identical Final_Caption: 0`; toptee-last `dict.update` (`utils_csmcir.py:97-103`) ⇒ 121 shirt-gallery images captioned as tops — P2 — report it; expect exactly 121 shirt deltas in any per-category caption check.
44. **Gallery construction** — VERIFIED — full ordered `split.{cat}.val.json` (`data_utils_csmcir.py:419`), per category (`validate_blip_csmcir.py:596-598, :612`) — NONE — none.
45. **Reference-exclusion policy matches `fashioniq_original_split`** — VERIFIED — `compute_fiq_val_metrics:24-53` never masks the reference; `reference_excluded: false` in `papers.yaml:800` — NONE — none.
46. **Recall@K maths** — VERIFIED — `:41-53` — NONE — none.
47. **Macro definition** — VERIFIED — unweighted mean of the three categories; `average_recall = (mean(r50)+mean(r10))/2` (`:624-626`) — NONE — none.
48. **Evaluator output shape matches the parser** — VERIFIED — JSON keys at `:620-626`; `parse_csmcir_output` (`metrics_extraction.py:219-303`) requires exactly those — NONE — none.
49. **Parser internal-consistency tolerance stays internal** — VERIFIED — `metrics_extraction.py:277-280` comments it and uses 0.05 only for the run's own aggregate vs its own categories; `_check_parity:28-62` now goes through `paper_gate.compare_metric` (5% relative) — NONE — none.
50. **Image extension contract** — VERIFIED — `models.yaml:50` `[".png"]`; 10 `.png` sites in `data_utils_csmcir.py`; the official release ships `.jpg` (`README.md:31-33`) — NONE — none.
51. **Images present with the declared extension** — VERIFIED (filesystem) — 74381 files, 100% `.png`; no extension block raised — NONE — none.
52. **Canonical dataset link** — VARIANT — declared `models.yaml:33`, absent (preflight block 5); `prepare_dataset.py:65` stages it; `runtime.py:1989-2000` verifies it as a symlink to `FASHIONIQ_ROOT` — P2 — operator stages it.
53. **Preprocessing transform** — VERIFIED — `targetpad_transform(1.25, 224)` (`:506,:574,:580`) — NONE — none.
54. **Augmentation in evaluation** — VERIFIED (none) — eval branches apply only `self.preprocess` (`data_utils_csmcir.py:466-470`) — NONE — none.
55. **Qwen caption integrity** — VERIFIED — three digests/sizes match `auxiliary_assets.yaml`, the manifest, the live HF API sizes, and my own hashing — NONE — none.
56. **Checkpoint download URL** — VERIFIED — HF API lists the path; registry URL resolves — NONE — none.
57. **Checkpoint sha256** — VERIFIED — `c8812071…` == HF `lfs.sha256` — NONE — none.
58. **Checkpoint size** — VERIFIED — `2720615859` == HF `size` — NONE — none.
59. **Checkpoint present locally / loaded** — BLOCKED — `artifacts/checkpoints/` holds only `.gitkeep`; dry run reports `checkpoint missing` — P0 — download with digest verification before any run.
60. **Checkpoint → reported metric mapping** — UNKNOWN — metadata-only chain (paper row = training-log row = registry numbers; one FashionIQ checkpoint published) — P1 — must be shown by a completed run.
61. **Reported metrics vs paper** — VERIFIED — HTML v3 row cells `52.45 74.81 57.70 75.76 61.14 80.98 57.07 77.27 67.17` — NONE — none.
62. **Reported metrics vs author training log** — VERIFIED — epoch 9 averages 57.066963/77.267204/67.167083 → the registry triplet — NONE — none.
63. **Benchmark row states the experiment's conditions** — VERIFIED — `papers.yaml:790-801`; `select_reference` → `exact condition match`, `reference_status: VERIFIED`; `_reference_constraint_count > 0` so `BLOCKED_REFERENCE_INCOMPLETE` cannot apply — NONE — none.
64. **A run with no variant declared is not labelled a variant** — VERIFIED — `_variant_provenance` derives from preflight evidence only, never from the model id (`evaluate_models.py:243-248`) — NONE — none.
65. **Environment spec matches the README** — VERIFIED — python 3.9 / torch 2.0.1 / torchvision 0.15.2 == `README.md:7-14`; `cuda: UNKNOWN` is honest — P3 — fill once a host is fixed.
66. **`source_runtime_artifacts` allowlist is bounded** — VERIFIED — `registry.py:563-612` returns only the declared auxiliary destinations, the declared link, and the declared patch (each with its digests); anything else is `unauthorized`/`invalid` (`runtime.py:1857-1907`) — NONE — none.
67. **`source_status` approved list is de-duplicated** — VARIANT — 6 entries for 3 files (OBSERVED) because untracked+ignored paths are scanned twice (`runtime.py:1874-1877`) — P3 — de-duplicate (M3).
68. **Docs never claim official parity for the variant** — VARIANT — `COT_OURS2_AUDIT.md:9`, `PC_TRANSFER_CHECKLIST.md:137`, `workbench/README.md:83`, `PC_REPRODUCTION_HANDOFF.md:39`, and the two registries all say variant/unverified. Residual defects: `PC_TRANSFER_CHECKLIST.md:53` still lists the COT path as a bare manual action, and `PC_RELEASE_READINESS.md:38` omits the parent-directory correction — P2 — fix both (M1).
69. **`READINESS_MATRIX.md` reflects current state** — VARIANT — `:25` still reports "Qwen captions missing" and "COT_ours2 captions missing" while `pc_readiness --json` (OBSERVED) reports 3 verified Qwen + 3 unverified aliases + 5 blockers, `BLOCKED_SOURCE` — P2 — regenerate (M7).
70. **Workbench csmcir tests encode true facts** — VERIFIED — `.venv/bin/python -m pytest workbench/tests/test_approved_source_patch.py workbench/tests/test_csmcir_source_blocker.py -q` → `13 passed in 2.64s`; the assertions match my independent recomputation (98 imports, 44-151, before line 152, digests) — NONE — none.
71. **Runtime execution observed** — NOT_RUN — no checkpoint, no torch in the repo interpreter, no GPU run; all claims above are STATIC_AUDIT/UNKNOWN — P0 — a GPU run is the only route to a runtime verdict.

---

FINAL VERDICT: **VARIANT** — the reproduction path is a faithfully derived, digest-pinned, explicitly labelled variant: an unimportable pinned package repaired by exactly one base/derived-checked patch plus a manifest-bound caption alias that the workbench labels `SOURCE_COMPATIBLE_CAPTION_ALIAS`/`SOURCE_COMPATIBLE_EQUIVALENCE_UNKNOWN` and whose gate verdict can never be bare `PASS_WITHIN_5_PERCENT`, while no run, no checkpoint load, and no official-parity claim exists anywhere.
