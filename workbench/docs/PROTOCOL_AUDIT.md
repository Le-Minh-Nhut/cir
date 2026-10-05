# FashionIQ protocol audit

`fashioniq_original_split` and `fashioniq_val_split` are incompatible analysis cohorts. Every workbench protocol declares clean FashionIQ evaluation: `evaluation_noise_pct: 0`. Training noise belongs to checkpoint provenance and does not change this evaluation setting.

## `fashioniq_original_split` — FashionIQ Original Split

- Categories: `dress`, `shirt`, `toptee`.
- Annotation source: `captions/cap.{category}.val.json`.
- Gallery source/order: complete `image_splits/split.{category}.val.json` order.
- Target must occur once in gallery.
- Reference stays eligible; no reference exclusion.
- Recall: target occurs within the first K descending-score results.
- Aggregate: macro category R@10/R@50; mean is their arithmetic mean.

Evidence: local `src/evaluation/fashioniq.py` loads `split.{category}.val.json`, ranks complete score rows, and directly checks target positions. CSMCIR `src/validate_blip_csmcir.py` extracts validation index features from the split list and computes target labels without reference removal.

CSMCIR's official evaluator is fixed-root/cwd-sensitive, accepts no ordinary dataset-root CLI, requires source-root `fashionIQ_dataset` and qwen/COT auxiliary caption files, and emits aggregate metrics only. This describes official phase behavior; it does not establish a per-query export or local reproduction. Any future observation-only ranking export must follow verified aggregate-metric parity without changing evaluation semantics.

## `fashioniq_val_split` — FashionIQ Val Split

- Categories: `dress`, `shirt`, `toptee`.
- Annotation source: `captions/cap.{category}.val.json`.
- Gallery: first-seen ordered union of validation annotation reference then target IDs.
- Reference image is removed from each query's ranking before Recall@K.
- Recall: target occurs within the first K non-reference descending-score results.

Evidence: local `src/datasets/fashioniq.py:build_pair_union_gallery` implements ordered reference/target union. Local `src/evaluation/fashioniq_encoder.py` removes reference after a stable descending sort. ENCODER and PAIR official datasets use this pair-union gallery in default `val-split`; each evaluation zeros reference similarity before ranking. HINT builds the same union and excludes reference before ranking.

ENCODER requires an external `open_clip_pytorch_model.bin` ViT-B-32 asset in its upstream root in addition to its FashionIQ checkpoint. Its direct checkpoint file URL is unresolved. Those prerequisites block execution; they do not alter val-split semantics.

## Analysis and adapter rules

- An adapter declares supported workbench protocol IDs; method names never imply a protocol.
- Upstream owns preprocessing, caption composition, score computation, gallery, ranking, and official metrics.
- Cross-protocol aggregate, consensus, Jaccard, rank-disagreement, common-distractor, and cohort requests fail closed.
- Same-protocol comparison still requires exact query-ID universe and matching canonical category, annotation index, reference ID, target ID, and raw captions. `model_input_text` may differ.
- Top-K set analysis requires every selected artifact to store requested depth. Target-rank analysis does not claim unavailable retrieval rows.
