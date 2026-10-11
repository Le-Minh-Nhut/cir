# Upstream data-preparation audit

Every registered model, audited against its pinned upstream checkout. Source-level evidence only:
no dataset, checkpoint, environment or GPU evaluation was run for this document, and no upstream
file was modified.

Detailed per-component matrices with severity, downstream impact and required correction live in
`workbench/docs/audits/PHASE2_SOURCES_{A,B,C,D}.md`; the cross-method data contract is in
`PHASE2_DATA_CONTRACT.md`; the adversarial cross-check of these findings is in
`PHASE2_CROSS_REVIEW.md`. This document consolidates them.

**No model is `EXACT_MATCH`.** Adapters mostly route to the official evaluator, and routed
execution plus artifact parity have not been reproduced. `EQUIVALENT_WITH_EVIDENCE` means the
named component agrees with the source at that boundary only.

## Pin state

All fourteen source repositories are present and clean at the SHA recorded in
`workbench/registry/models.yaml`; `ptha_mtst` has no source and is `SOURCE_UNAVAILABLE`.

| Model | Repository | Pinned SHA (12) |
|---|---|---|
| csmcir | qzp2018/CSMCIR | `774f94e2076f` |
| limn | iLearn-Lab/TPAMI24-LIMN | `7d7bc9b116f5` |
| encoder | iLearn-Lab/AAAI25-ENCODER | `29a2a31d6a56` |
| pair | iLearn-Lab/ICASSP25-PAIR | `bfb0b98cdb4c` |
| hint | iLearn-Lab/ICASSP26-HINT | `bec50b6c8c19` |
| clvc_net | iLearn-Lab/SIGIR21-CLVC-Net | `bd9b68894898` |
| tgcir | iLearn-Lab/MM23-TG-CIR | `65fa78eaf8ca` |
| airknow | iLearn-Lab/CVPR26-Air-Know | `6ab6fa95b746` |
| conesep | iLearn-Lab/CVPR26-ConeSep | `bd4bf629ebe6` |
| habit | iLearn-Lab/AAAI26-HABIT | `6ddc2cd63eac` |
| intent | iLearn-Lab/AAAI26-INTENT | `2a75fc6f57d1` |
| dcnet | ozmig77/dcnet | `68c79d38569f` |
| combiner_rn50x4_noft | ABaldrati/CLIP4Cir | `dfed9f748a8a` |
| clip4cir_rn50x4_fullft | ABaldrati/CLIP4Cir | `dfed9f748a8a` |
| sprc | chunmeifeng/SPRC | `2935a5397732` |
| ptha_mtst | — | none |

## The iLearn family shares one data contract

Air-Know, ConeSep, HABIT, INTENT, HINT, ENCODER, PAIR, LIMN, CLVC-Net and TG-CIR are one code
lineage. Their dataset classes agree on the following, traced in each repository's own
`datasets.py`:

- **Root**: a string joined by concatenation (`path + 'resized_image'`), so the caller's path MUST
  end with a separator. The workbench appends one.
- **Layout**: `resized_image/{category}/{id}.jpg`, `captions/cap.{category}.{train,val}.json`,
  `captions/correction_dict_{category}.json`, `image_splits/split.{category}.val.json`, and — on a
  cache miss — a fresh-constructor path that also reads the *train* annotations.
- **Cache**: when `fashion_iq_data.json` is absent the constructor builds it and writes six
  `test_queries_{category}.pkl` / `test_targets_{category}.pkl` files; when present it loads them.
  No inspected source validates the cache against the JSON, so a stale cache is trusted.
- **Text**: `correct_text()` lowercases, translates every `string.punctuation` character to a
  space, whitespace-tokenises, applies an exact per-token dictionary lookup, then `concat_text()`
  joins the two captions in source order with ` and `. Downstream the string goes through the
  LAVIS `BlipCaptionProcessor` (lowercase, replace `[.!"()*#:;~]`, collapse whitespace, 50-word
  limit) and then a BERT tokenizer with `max_length=32`, truncating.
- **Images**: `targetpad_transform(1.25, 224)` — symmetric zero-pad when the aspect ratio is at
  least 1.25, bicubic short-edge resize to 224, centre crop 224, RGB, tensor, CLIP mean/std
  `(0.48145466, 0.4578275, 0.40821073)` / `(0.26862954, 0.26130258, 0.27577711)`.
- **Noise**: `FashionIQ.shuffle()` is a *training* hook applied at `noise_ratio`. No evaluator
  calls it. Training noise is never evaluation noise.

Two consequences matter scientifically:

1. **The evaluation preprocessing is a two-stage composition**, not one normalisation. The dataset
   `correct_text` stage and the LAVIS processor stage are separate and not equivalent; a workbench
   path that applies only one of them does not reproduce the model's input.
2. **These repositories require a correction dictionary per category** that no upstream pins.
   The workbench checks only that the file exists; it cannot verify provenance.

## Per-model findings

### csmcir — `fashioniq_original_split`

- Dataset entry `CSMCIR/src/data_utils_csmcir.py:FashionIQDataset.__init__` (class at line 363,
  the one `validate_blip_csmcir.py:225/227/596/598` constructs). It reads
  `captions/cap.{category}.{split}.json`, `qwen_captions/{category}_cot_{split}.json` and
  `image_splits/split.{category}.{split}.json`.
- `utils_csmcir.py:extract_index_blip_caption_features` additionally opens
  `COT_ours2/fashioniq/{category}_cot_val.json`.
- Images are opened as `images/{id}.png` (lines 279, 314, 320, 337, 343, 449, 452, 462) while the
  official release and the repository's own README show `.jpg`.
- Auxiliary assets: `qwen_captions/*` (author HF, verified revision, no SHA) and
  `COT_ours2/fashioniq/*_cot_val.json` (no pinned provenance) — both recorded in
  `workbench/registry/auxiliary_assets.yaml`. The LAVIS base `blip2_pretrained.pth` is required and
  **not** recorded for this model.
- **Mismatch, fixed here**: the image extension. The registry now declares
  `fashioniq_image_extensions: [".png"]` and both preflight views check every validation id.
- **Open**: LAVIS base weight provenance; COT_ours2 provenance; torchvision version.

### limn — `fashioniq_val_split`

- `LIMN/LIMN/datasets.py:FashionIQ.get_test_data` and `LIMN/LIMN/test.py:test`. Pair-union gallery
  in first-seen order; reference masked before ranking.
- `workbench/replay/limn.py` drives the pinned `test()` per category with an isolated temp root and
  a trailing separator, and constructs the transform from the checkpoint's own image size/mean/std
  while checking package versions. This is the one model with a dedicated native replay path.
- **Verified**: cohort, reference eligibility, ranking, and the explicit workbench macro (which is
  labelled as a workbench aggregate, not a source metric).
- **Open**: none identified beyond GPU-host execution.

### encoder — `fashioniq_val_split`

- `ENCODER/datasets.py:FashionIQ`, `evaluate_model.py`. Pair union; reference masked.
- Requires `open_clip_pytorch_model.bin` in the upstream root and `datasets1.py`, both checked as
  hard blockers by the runtime.
- **Open**: the base OpenCLIP binary has no pinned provenance; the fresh-constructor train-caption
  and cache requirement is not in the layout check. Local `fashioniq_encoder.py` is *not* the
  upstream score model and must never be presented as reproducing ENCODER.

### pair — `fashioniq_val_split`

- Pinned `test.py` defines a callable only; there is no source-supported CLI. The registry leaves
  the command blocked and both `pair-B1.pt` / `pair-B2.pt` mapping `UNVERIFIED`, which is correct:
  nothing in the source establishes either as the FashionIQ benchmark model.
- **Open (blocking)**: no runnable command, unverified checkpoint identity.

### hint — `fashioniq_val_split`

- `HINT/datasets.py:FashionIQ.get_test_data` and `test.py:test`. Ordered pair union; reference
  masked before ranking.
- **Open (blocking)**: no evaluation-only command or checkpoint load in the pinned source; the
  val-only layout check misses the fresh-constructor train inputs.

### clvc_net — `fashioniq_val_split`

- `CLVCNet/datasets.py:FashionIQ.get_test_targets`, `test.py:test`. Source-order endpoint union;
  reference masked before ranking. Captions become `<BOS> {cap0} <AND> {cap1} <EOS>` through a
  `SimpleVocab` tokenizer — a different pipeline from the LAVIS/BERT line.
- **Mismatch, open**: the source trains separate whole models per category while the registry sets
  `category_model_policy: shared` and carries one unresolved variant. One pair cannot cover three
  categories.

### tgcir — `fashioniq_val_split`

- `TGCIR/datasets.py`, `test.py`. Source-order endpoint union; reference masked.
- **Open**: correction dictionaries must be placed at the loader's expected path and have no
  pinned provenance; no runnable command; no archive identity.

### airknow / conesep / habit / intent — `fashioniq_full_gallery_ref_excluded`

- `datasets.py:FashionIQ.get_test_data` for each, with `test_BLIP2.py:test` (Air-Know) or
  `test.py:test`. Full ordered validation split; reference masked to `-10e10` before ranking.
- Air-Know's `test()` calls `model.train()` and its FashionIQ branch never switches to `eval()`;
  ConeSep's calls `model.eval()`. This is a real difference between the two evaluators and is
  recorded as such.
- INTENT's `datasets.py` has both an `original-split` (default) and a `val-split` gallery branch;
  the registry protocol describes the default.
- **Mismatch, fixed here**: all four named a protocol the workbench evaluator could not build.
- **Open**: LAVIS base BLIP-2 weights and `/root/bert-base-uncased` are required but not tracked
  per model; correction dictionaries have no pinned provenance; no runnable command; the
  noisy-training checkpoint filename → noise-level mapping is Hub metadata, not verified payload.

### dcnet — `fashioniq_full_gallery_ref_excluded`

- `DCNet/data_loader/CE_dataset.py:CE.__init__` reads `captions/cap.{category}.glove.val.pkl` and
  the ordered `split.{category}.val.json` as `val_trg`;
  `trainer/trainer_joint.py:TrainerJoint._valid_epoch` masks each reference in both branches and
  fuses `log_softmax(score_comp) + log_softmax(score_corr)` before ranking.
- Requires generated GloVe PKLs, pre-existing `resized_images/`, ResNet-50 ImageNet weights, and —
  to *regenerate* the PKLs — spaCy `en_vectors_web_lg` with unseeded OOV vectors. Author-generated
  assets must not be silently regenerated and called equivalent.
- **Mismatch, fixed here**: named a protocol the evaluator could not build.

### combiner_rn50x4_noft / clip4cir_rn50x4_fullft — `fashioniq_original_split`

- `CLIP4Cir/src/validate.py`. Full ordered split; reference eligible. Both variants share one data
  contract but have distinct checkpoints.
- **Mismatch, open**: the validator hardcodes `<checkout>/fashionIQ_dataset` and never uses the
  request's dataset root, while the workbench only links that path for CSMCIR. A canonical root
  alone does not satisfy this evaluator.

### sprc — `fashioniq_original_split`

- `SPRC/src/validate_blip.py:compute_fiq_val_metrics`, `data_utils.py:FashionIQDataset`. Full
  ordered split; reference eligible.
- **Mismatch, open**: `blip2_qformer_cir_rerank.py:inference` accepts `reference_embeds` and never
  reads it — the official score is text-to-gallery. The workbench retrieval path is
  reference-conditioned, so local SPRC-shaped scores cannot stand in for the official ones. The
  CLI's default model name is also unregistered in the pinned source.

### ptha_mtst — no source

`SOURCE_UNAVAILABLE`. No repository, no checkpoint mapping, no evaluator. Not auditable.

## Changes made in this pass

| Change | Reason |
|---|---|
| `src/evaluation/fashioniq.py` implements `fashioniq_full_gallery_ref_excluded` | Five registered models named a protocol the evaluator raised on |
| The two older families accept their registry protocol ids, not only the internal aliases | A caller holding `fashioniq_original_split` or `fashioniq_val_split` could not build either cohort |
| `recall_at_k_reference_excluded` removes the reference from the ranking | Matches the upstream evaluators instead of relying on mask equivalence |
| `evaluate_fashioniq_recall` requires references exactly when exclusion is active | A protocol mix-up cannot be silently evaluated as a plain one |
| Registry gains `fashioniq_image_extensions`, validated and checked in both preflight views | CSMCIR opens `.png`; the official release ships `.jpg` |
| `tests/unit/test_fashioniq_protocols.py`, `test_fashioniq_caption_policies.py` | Pin the laptop-testable protocol and caption semantics |

## Tests executed

`python -m pytest tests/unit workbench/tests -q` — 565 passing, including the new protocol,
caption and image-extension tests. No test in this pass reproduces a model score, and none claims
to.

## Pending on the GPU host

Per model: fetch and hash the author checkpoint, build a source-compatible environment, run the
native evaluator, capture per-category and per-query rankings, and compare against the
paper-reported numbers as sanity references only. For the ten models whose evaluator is a callable
with no CLI, reaching a runnable command under the pinned source is itself unresolved and is
recorded as such rather than invented.
