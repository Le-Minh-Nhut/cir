# FashionIQ protocol audit

`fashioniq_original_split`, `fashioniq_full_gallery_ref_excluded`, and `fashioniq_val_split` are incompatible analysis cohorts. Every registered workbench protocol declares clean FashionIQ evaluation: `evaluation_noise_pct: 0`. Training noise belongs to checkpoint provenance and does not change this evaluation setting.

## `fashioniq_original_split` — Full Gallery, Reference Eligible

- Literature label: `original`.
- Categories: `dress`, `shirt`, `toptee`.
- Annotation source: `captions/cap.{category}.val.json`.
- Gallery source/order: complete `image_splits/split.{category}.val.json` order.
- Target must occur once in gallery.
- Reference stays eligible; no reference exclusion.
- Recall: target occurs within the first K descending-score results.
- Aggregate: macro category R@10/R@50; mean is their arithmetic mean.

Evidence: `validate_blip_csmcir.py:blip_validate_fashioniq` creates per-category validation `FashionIQDataset` instances in `relative` and `classic` modes. `FashionIQDataset` reads `fashionIQ_dataset/captions`, `image_splits`, and `images`, plus linked dataset-root `qwen_captions/{dress,shirt,toptee}_cot_val.json`. The direct index call at `validate_blip_csmcir.py:612` reaches `utils_csmcir.py:extract_index_blip_caption_features`, which reads source-root `COT_ours2/fashioniq/{dress,shirt,toptee}_cot_val.json`. The evaluator accepts no ordinary dataset-root CLI; preparation maintains the fixed source-root link. It emits aggregate metrics only and does not establish a per-query export or local reproduction.


## `fashioniq_full_gallery_ref_excluded` — Full Gallery, Reference Excluded

- Literature label: `original`.
- Categories, annotation source, and gallery order: identical to `fashioniq_original_split`.
- Reference is excluded from each query ranking before Recall@K.
- Air-Know, ConeSep, HABIT, and INTENT select `original-split`, enumerate full validation gallery order, and set reference similarity to `-10e10` before R@1/R@10/R@50.
- DCNet `CE_dataset.py:CE.__init__` reads `image_splits/split.{category}.val.json` directly as ordered `val_trg`; `TrainerJoint._valid_epoch` masks each reference in both branches before ranking. It fuses `F.log_softmax(score_comp)` and `F.log_softmax(score_corr)` before the final rank. This establishes this cohort's gallery/reference semantics; it does not certify DCNet checkpoint, generated caption artifacts, environment, command, or metric reproduction.

This differs from `fashioniq_original_split`: target rank and Recall@K can change after removing reference. It also differs from `fashioniq_val_split`, whose gallery is annotation pair-union. Cross-protocol analysis remains invalid.


## `fashioniq_val_split` — FashionIQ Val Split

- Categories: `dress`, `shirt`, `toptee`.
- Annotation source: `captions/cap.{category}.val.json`.
- Gallery: first-seen ordered union of validation annotation reference then target IDs.
- Reference image is removed from each query's ranking before Recall@K.
- Recall: target occurs within the first K non-reference descending-score results.

Evidence: local `src/datasets/fashioniq.py:build_pair_union_gallery` implements ordered reference/target union. Local `src/evaluation/fashioniq_encoder.py` removes reference after a stable descending sort. ENCODER and PAIR official datasets use this pair-union gallery in default `val-split`; each evaluation zeros reference similarity before ranking. HINT builds the same union and excludes reference before ranking.

Air-Know, ConeSep, HABIT, and INTENT map to `fashioniq_full_gallery_ref_excluded`. Their commands remain unready until separately audited; mapping preserves evaluator semantics, not runtime readiness.

All inspected iLearn sources (`Air-Know`, ConeSep, HABIT, INTENT, HINT, ENCODER, and PAIR) require `captions/correction_dict_{dress,shirt,toptee}.json` in addition to their image layout. They lowercase a caption, translate `string.punctuation` to spaces, whitespace-tokenize, make exact dictionary substitutions, then compose source-order captions with ` and `. DQU-CIR pins files with matching names and behavior, but no inspected iLearn source pins those blobs. Workbench therefore only checks local presence for iLearn layouts; it does not download, verify, or attribute these files.

ENCODER requires an external `open_clip_pytorch_model.bin` ViT-B-32 asset in its upstream root in addition to its FashionIQ checkpoint. Its direct checkpoint file URL is unresolved. Those prerequisites block execution; they do not alter val-split semantics.

## Analysis and adapter rules

- An adapter declares supported workbench protocol IDs; method names never imply a protocol.
- Upstream owns preprocessing, caption composition, score computation, gallery, ranking, and official metrics.
- Cross-protocol aggregate, consensus, Jaccard, rank-disagreement, common-distractor, and cohort requests fail closed.
- Same-protocol comparison still requires exact query-ID universe and matching canonical category, annotation index, reference ID, target ID, and raw captions. `model_input_text` may differ.
- Top-K set analysis requires every selected artifact to store requested depth. Target-rank analysis does not claim unavailable retrieval rows.

## Audited legacy assignments

| Model | Assignment | Evidence boundary |
| --- | --- | --- |
| CLVC-Net | `fashioniq_val_split` | Source-order annotation queries; first-seen reference/target union; reference score set to `-10e10` before descending ranking. |
| DCNet | `fashioniq_full_gallery_ref_excluded` | Ordered `val_trg` directly from `split.{category}.val.json`; reference score masked in composition and correction branches before log-softmax fusion/rank. Generated GloVe PKLs and checkpoint/config pairing remain blocked. |
| Combiner RN50x4 noft / CLIP4Cir RN50x4 fullft | `fashioniq_original_split` | Full ordered validation split gallery; reference remains eligible. Artifact variants remain unverified. |
| TG-CIR | `fashioniq_val_split` | Source-order annotation endpoint union and source mask before descending ranking. Correction dictionaries must be manually placed at loader's expected dataset path. |
| SPRC | `fashioniq_original_split` | Full ordered validation gallery; reference remains eligible. |
| LIMN base iteration 0 | `fashioniq_val_split` | Source-order annotation endpoint union and source mask before descending ranking. Do not generalize to LIMN+ or later iterations. |

These assignments establish analysis compatibility only. They do not certify a checkpoint, preparation artifact, environment, command, native aggregate, or instrumentation parity. NEUCORE remains outside every registered cohort: its two caption-order queries per annotation are a different query universe.
