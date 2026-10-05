# Result JSON schema v1

Canonical source artifact: `workbench/artifacts/results/{protocol_id}/{model_id}/{checkpoint_id}.json`.

```json
{
  "schema_version": 1,
  "run": {
    "run_id": "unique run ID",
    "dataset": "FashionIQ",
    "split": "val",
    "protocol_id": "fashioniq_original_split",
    "evaluation_noise_pct": 0,
    "model_id": "csmcir",
    "method_name": "CSMCIR",
    "checkpoint_training_noise_pct": 0,
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
    "model_input_text": "Model-specific composed caption",
    "target_rank": 37,
    "top_results": [{"rank": 1, "image_id": "image", "score": 0.8123}]
  }]
}
```

`query_id` derives from category, annotation index, reference ID and target ID. It is independent of `model_input_text`. `top_results` ranks must be unique, contiguous from 1, and no deeper than configured save depth. `target_rank` is mandatory even when it exceeds stored `top_results` depth.

Mock fixtures set `run.data_kind` to `mock`; UI must never interpret them as experiments.
