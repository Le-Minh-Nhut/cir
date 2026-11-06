# Cleanup history

Date: 2026-10-06.

The former TAPER/CSMCIR training implementation and its direct entrypoints were removed. This repository now contains canonical dataset/evaluation utilities and CIR Failure Analysis Workbench.

Current executable scope:

- `src/datasets/`, `src/cache/`, `src/evaluation/`: FashionIQ and CIRR loading/evaluation utilities.
- `workbench/`: registry-driven upstream orchestration, result schema, DuckDB serving index, local FastAPI API, and React UI.

Removed surface is recorded with rationale in [workbench/docs/LEGACY_CLEANUP_AUDIT.md](workbench/docs/LEGACY_CLEANUP_AUDIT.md). Historical implementation details remain available through Git history; this file intentionally contains no runnable commands for removed paths.

## 2026-11-07 — checkpoint and cache sweep

Authorised as a deliberate restart of the old training workflow. Classification was by **file
content**, not by extension or directory name: every candidate was opened with `torch.load`, and a
dict carrying `model`/`optimizer`/`state_dict` (or `.weight`/`.bias` keys) is a trained model, while a
raw tensor or a dict of `sample_ids`/`features`/`timesteps` is a measurement artifact. Twelve `.pt`
files turned out to be feature dumps and were kept.

**Deleted — 160 files, 115.68 GiB (124 212 078 490 B):**

- `outputs/` — 142 checkpoints of the removed TAPER/`iag_srme` stack.
- `runs/` — 8 `.ckpt` files from the same stack; the metrics/config JSON beside them were kept.
- 10 duplicate or wrong-dataset CLVC and TG-CIR weights. The six CLVC duplicates were verified
  byte-for-byte and sha256 against `selected_bundle_digests.json` before removal, so the canonical
  copies under `checkpoints/clvc/` are provably the same bytes.

**Also deleted earlier in the same sweep — 78.80 GiB of legacy feature caches:**
`features/cirr/csmcir`, `features/fashioniq/csmcir`, `features/fashioniq/fgclip2-base`. Their
producers (`src/precompute_cirr.py`, `src/precompute_csmcir_stage1.py`,
`src/precompute_taper_e2e_text.py`) were removed on 2026-10-06, and no live code references the paths.

**Kept, verified after the sweep:** all 22 cohort checkpoints (bytes + sha256 re-checked against the
acquisition manifest), TME and ENCODER base weights, the regenerated TME feature cache, datasets,
environments, result metadata (`metrics_val.jsonl`, `config_resolved.yaml`, logs), reports, and the
feature/audit dumps under `teacher/outputs/**` and `outputs/diagnostics/**`.

**Free space:** 24.70 GiB → 201.92 GiB. Net 177.22 GiB, less than the 194.48 GiB removed because the
TME feature cache was regenerated (20.94 GiB) during verification.

**Out of scope, untouched:** `data/CIRR/*.pkl` (dataset-adjacent, keep-class), `data/stage1_relation_probe`,
`data/FashionIQ/captions/*.pkl` (referenced by the DCNet contract).

Full detail with per-group tables is in
[CHECKPOINT_SELECTION_REPORT.md §3.17](CHECKPOINT_SELECTION_REPORT.md).
