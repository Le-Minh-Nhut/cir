# Result JSON schema v2

Canonical source artifact: `workbench/artifacts/results/{protocol_id}/{model_id}/{checkpoint_id}.json`. DuckDB is a rebuildable serving layer, never source of truth.

```json
{
  "schema_version": 2,
  "run": {
    "run_id": "immutable unique evaluation artifact ID",
    "dataset": "FashionIQ",
    "split": "val",
    "protocol_id": "fashioniq_original_split",
    "evaluation_noise_pct": 0,
    "model_id": "csmcir",
    "method_name": "CSMCIR",
    "checkpoint_id": "fashioniq",
    "checkpoint_training_noise_pct": 0,
    "top_k_saved": 200,
    "gallery_size": 12345,
    "data_kind": "experiment",
    "reported_paper_metrics": {"r10": 57.07, "r50": 77.27, "mean": 67.17},
    "reproduced_metrics": {"r10": 0, "r50": 0, "mean": 0}
  },
  "queries": [{
    "query_id": "dress:123:reference:target",
    "category": "dress",
    "annotation_index": 123,
    "reference_id": "reference",
    "target_id": "target",
    "raw_captions": ["caption one", "caption two"],
    "model_input_text": "model-specific composed caption",
    "target_rank": 37,
    "top_results": [{"rank": 1, "image_id": "image", "score": 0.8123}]
  }]
}
```

`run_id`, not `model_id`, is analysis identity. Different checkpoints or training-noise conditions of one model must have distinct `run_id` values. `query_id` derives from canonical category, annotation index, reference ID and target ID; `model_input_text` may differ.

v2 requires `checkpoint_id`, `top_k_saved`, and `gallery_size`. Stored ranks are contiguous and unique; `len(top_results) <= min(top_k_saved, gallery_size)`, while `top_k_saved` may intentionally exceed gallery size. Target rank cannot exceed gallery size. Top-K set operations require every selected run to save requested depth. Target-rank metrics remain valid without top-K retrieval rows.

Schema v1 files are intentionally rejected. Regenerate result JSON from verified instrumentation rather than silently guessing missing v2 metadata. Mock fixtures use `data_kind: "mock"`; no UI result from them is research evidence.
