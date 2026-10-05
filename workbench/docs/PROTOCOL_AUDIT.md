# FashionIQ protocol audit

`fashioniq_original_split` and `fashioniq_val_split` are incompatible analysis cohorts. Every evaluation uses clean FashionIQ data: `evaluation_noise_pct: 0`.

## `fashioniq_original_split` — FashionIQ Original Split

- Categories: `dress`, `shirt`, `toptee`.
- Annotation source: `captions/cap.{category}.val.json`.
- Gallery source and ordering: full `image_splits/split.{category}.val.json` order.
- Target must occur once in gallery.
- No reference exclusion.
- Recall: target appears in top K of descending scores.
- Aggregate: macro average category R@10 and R@50; mean is their arithmetic mean.

Evidence: local `src/evaluation/fashioniq.py` loads `split.{category}.val.json`, ranks complete score rows, and directly checks target positions. CSMCIR `src/validate_blip_csmcir.py` extracts classic validation index features from the split list and computes target labels without reference removal.

## `fashioniq_val_split` — FashionIQ Val Split

- Categories: `dress`, `shirt`, `toptee`.
- Annotation source: `captions/cap.{category}.val.json`.
- Gallery: ordered unique union, for validation annotation order, of reference then target IDs.
- Reference image is removed from each query ranking before Recall@K.
- Deterministic gallery order comes from first occurrence.
- Recall: target appears in first K non-reference results of descending scores.

Evidence: local `src/datasets/fashioniq.py:build_pair_union_gallery` implements ordered reference/target union. Local `src/evaluation/fashioniq_encoder.py` removes each reference after a stable descending sort. ENCODER and PAIR official datasets use this pair-union gallery in their default `val-split`; each evaluation zeros reference similarity before ranking. HINT builds the same union and excludes reference before ranking.

## Adapter rules

- A model adapter declares compatible workbench protocol IDs; model names never determine protocol IDs.
- Upstream command, preprocessing, caption composition, score computation, gallery and rank semantics remain upstream-owned.
- Instrumentation may only export existing rankings after official reproduction is proven metric-identical.
- Cross-protocol aggregate, consensus, Jaccard, rank-disagreement and common-distractor requests fail closed.
