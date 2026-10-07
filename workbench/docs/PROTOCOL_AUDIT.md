# FashionIQ protocol audit

`fashioniq_original_split` and `fashioniq_val_split` are incompatible analysis cohorts. Every registered workbench protocol declares clean FashionIQ evaluation: `evaluation_noise_pct: 0`. Training noise belongs to checkpoint provenance and does not change this evaluation setting. A method with a different reference policy is BLOCKED from result ingestion until a protocol ID is explicitly approved; it is never coerced into either mode.

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


## `fashioniq_val_split` — FashionIQ Val Split

- Categories: `dress`, `shirt`, `toptee`.
- Annotation source: `captions/cap.{category}.val.json`.
- Gallery: first-seen ordered union of validation annotation reference then target IDs.
- Reference image is removed from each query's ranking before Recall@K.
- Recall: target occurs within the first K non-reference descending-score results.

Evidence: local `src/datasets/fashioniq.py:build_pair_union_gallery` implements ordered reference/target union. Local `src/evaluation/fashioniq_encoder.py` removes reference after a stable descending sort. ENCODER and PAIR official datasets use this pair-union gallery in default `val-split`; each evaluation zeros reference similarity before ranking. HINT builds the same union and excludes reference before ranking.

Air-Know, ConeSep, HABIT, and INTENT were independently audited as full-gallery evaluators that mask their references before recall. That behavior is neither `fashioniq_original_split` nor `fashioniq_val_split`, so these models have no workbench protocol assignment and cannot enter a result cohort.

All inspected iLearn sources (`Air-Know`, ConeSep, HABIT, INTENT, HINT, ENCODER, and PAIR) require `captions/correction_dict_{dress,shirt,toptee}.json` in addition to their image layout. They lowercase a caption, translate `string.punctuation` to spaces, whitespace-tokenize, make exact dictionary substitutions, then compose source-order captions with ` and `. DQU-CIR pins files with matching names and behavior, but no inspected iLearn source pins those blobs. Workbench therefore only checks local presence for iLearn layouts; it does not download, verify, or attribute these files.

ENCODER requires an external `open_clip_pytorch_model.bin` ViT-B-32 asset in its upstream root in addition to its FashionIQ checkpoint. Its direct checkpoint file URL is unresolved. Those prerequisites block execution; they do not alter val-split semantics.

## Analysis and adapter rules

- An adapter declares supported workbench protocol IDs; method names never imply a protocol.
- Upstream owns preprocessing, caption composition, score computation, gallery, ranking, and official metrics.
- Cross-protocol aggregate, consensus, Jaccard, rank-disagreement, common-distractor, and cohort requests fail closed.
- Same-protocol comparison still requires exact query-ID universe and matching canonical category, annotation index, reference ID, target ID, and raw captions. `model_input_text` may differ.
- Top-K set analysis requires every selected artifact to store requested depth. Target-rank analysis does not claim unavailable retrieval rows.
