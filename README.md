# CIR Failure Analysis Workbench

Local research tool for reproducible, per-query FashionIQ retrieval failure analysis. It preserves canonical FashionIQ dataset and evaluation utilities; it does not implement a new CIR model.

## Executable structure

```text
src/datasets/      FashionIQ and CIRR annotation/loading utilities
src/evaluation/    FashionIQ original-split and ENCODER-compatible metrics
workbench/         pinned-upstream registry, result schema, DuckDB API, local UI
```

The current workflow lives in [workbench/README.md](workbench/README.md). External upstream source trees, datasets, checkpoints, generated results, DuckDB indexes, and frontend builds are ignored. Historical research context remains in Git history and tracked documentation.

## Safe laptop workflow

```bash
python workbench/scripts/load_mock_results.py
python workbench/scripts/rebuild_index.py
python -m pytest -q
cd workbench/frontend && npm run build
```

These commands create deterministic mock artifacts only. Do not download weights or run official model evaluation on this checkout.
