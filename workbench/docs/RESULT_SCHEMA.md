# Result JSON schema v2

Canonical result JSON is source of truth. DuckDB is a rebuildable serving index. Conventional placement is:

```text
workbench/artifacts/results/{protocol_id}/{model_id}/{checkpoint_id}.json
```

Discovery is deliberately recursive: `result_files()` loads every `*.json` below the selected result root except `.gitkeep`. Placement may therefore be nested differently, but each artifact must be schema v2 and `run_id` must be globally unique within that root.
The JSON below is an illustrative shape only; its IDs, ranks, and metric values are not an evaluation artifact or reproduction claim.


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

## Identity and protocol contract

`run_id`, not `model_id`, is analysis identity. Different checkpoints, source revisions, environments, or training-noise conditions require distinct IDs. `query_id` derives from canonical category, annotation index, reference ID, and target ID. Same-protocol comparison requires matching canonical identities and raw captions; `model_input_text` may differ.

`fashioniq_original_split` and `fashioniq_val_split` cannot be compared, aggregated, or combined into cohorts. Preserve the producing protocol exactly; do not relabel an artifact to fit a comparison.

## Rank and retrieval contract

v2 requires `checkpoint_id`, `top_k_saved`, and `gallery_size`. Stored ranks are contiguous and unique. `len(top_results) <= min(top_k_saved, gallery_size)`; `top_k_saved` may exceed gallery size. `target_rank` cannot exceed gallery size. Top-K operations require each selected run to contain requested depth; target-rank metrics remain valid when retrieval rows were not saved.

The model's official evaluation remains authoritative. For CSMCIR, official evaluator output is aggregate-only; no claim is made that official output supplies this per-query schema. Canonical per-query artifacts require a future reviewable observation-only export after official aggregate-metric parity is demonstrated.

## Provenance and data kinds

Use `data_kind: "mock"` only for deterministic development fixtures. Mock UI output is not research evidence. An experiment should retain source pin/checkpoint/environment/command evidence; `validate_results.py --strict-real` requires non-empty `upstream_commit`, `checkpoint_sha256`, `command_digest`, and `environment_digest` for non-mock artifacts.

Reported paper metrics are references, not reproduced local metrics. Do not add a checksum or direct asset URL that was not independently verified. Schema v1 artifacts are rejected, not migrated: regenerate from verified instrumentation rather than guessing v2 metadata.

## Validation and index behavior

```bash
python workbench/scripts/validate_results.py --file PATH [--strict-real]
python workbench/scripts/validate_results.py --root PATH [--strict-real]
python workbench/scripts/validate_results.py --all [--strict-real]
python workbench/scripts/rebuild_index.py --results-root PATH --check-only
python workbench/scripts/rebuild_index.py --results-root PATH --database PATH [--validate-first]
```

Load/validation rejects malformed JSON/schema, duplicate `run_id`, and invalid result invariants. Rebuild creates a temporary DuckDB file and atomically replaces target database only after all canonical artifacts load. A failed rebuild leaves existing index intact.
