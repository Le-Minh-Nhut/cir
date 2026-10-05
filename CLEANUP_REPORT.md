# Cleanup history

Date: 2026-10-06.

The former TAPER/CSMCIR training implementation and its direct entrypoints were removed. This repository now contains canonical dataset/evaluation utilities and CIR Failure Analysis Workbench.

Current executable scope:

- `src/datasets/`, `src/cache/`, `src/evaluation/`: FashionIQ and CIRR loading/evaluation utilities.
- `workbench/`: registry-driven upstream orchestration, result schema, DuckDB serving index, local FastAPI API, and React UI.

Removed surface is recorded with rationale in [workbench/docs/LEGACY_CLEANUP_AUDIT.md](workbench/docs/LEGACY_CLEANUP_AUDIT.md). Historical implementation details remain available through Git history; this file intentionally contains no runnable commands for removed paths.
