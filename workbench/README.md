# CIR Failure Analysis Workbench

Local research tool for reproducing public FashionIQ checkpoints, saving per-query retrievals, comparing methods, and finding systematic CIR failures. It does not reimplement models: an adapter invokes each pinned official repository later on a GPU host.

> **Laptop development policy:** this checkout ships code and mock data only. No checkpoints, backbone assets, feature caches, datasets, or model inference are required or performed.

## Quickstart — laptop development without checkpoints

```bash
cd ~/data/cir
git switch research/cir-failure-workbench
python3 -m venv .venv-workbench
source .venv-workbench/bin/activate
pip install -r workbench/backend/requirements.txt
python workbench/scripts/load_mock_results.py
python workbench/scripts/rebuild_index.py
uvicorn workbench.backend.main:app --host 127.0.0.1 --port 8000 --reload
```

In another terminal:

```bash
cd ~/data/cir/workbench/frontend
npm install
npm run dev
```

Open <http://localhost:5173>. Mock runs are visibly marked `mock`; they are not experiments.

## Quickstart — later on GPU host

```bash
python workbench/scripts/sync_upstreams.py --all
python workbench/scripts/download_checkpoints.py --list
python workbench/scripts/download_checkpoints.py --all --dry-run
python workbench/scripts/download_checkpoints.py --model csmcir
python workbench/scripts/download_checkpoints.py --verify-only --all
python workbench/scripts/run_eval.py --model csmcir --checkpoint fashioniq --protocol fashioniq_original_split --dataset-root "$CIR_DATA_ROOT/FashionIQ" --output workbench/artifacts/results/fashioniq_original_split/csmcir/fashioniq.json
python workbench/scripts/rebuild_index.py
```

`run_eval.py` only validates and prints an official command in this laptop phase. Execute reviewed command only in model-specific GPU environment after official evaluation behavior is verified.

## Protocols

| Workbench ID | UI label | Gallery and rank semantics |
| --- | --- | --- |
| `fashioniq_original_split` | FashionIQ — Original Split | Full ordered `image_splits/split.{category}.val.json`; reference remains eligible. |
| `fashioniq_val_split` | FashionIQ — Val Split | Ordered unique validation reference/target union; reference removed before Recall@K. |

These are separate research cohorts. Never aggregate recall, failure Jaccard, consensus failures, common distractors, rank disagreement, heatmaps, or saved cohorts across them. API rejects cross-protocol analytics.

Full audit: [docs/PROTOCOL_AUDIT.md](docs/PROTOCOL_AUDIT.md).

## Layout

- `backend/` — FastAPI API, schemas, adapters, indexing and analysis.
- `frontend/` — React/Vite local UI.
- `registry/models.yaml` — checked-in source/checkpoint provenance.
- `scripts/` — source sync, checkpoint download, mock loader, derived index rebuild, future evaluation command builder.
- `envs/` — model-specific environment specifications; add only from upstream requirements.
- `patches/` — reviewable observation-only patches after official metric reproduction.
- `docs/` — audits, result schema and workflow.
- `artifacts/checkpoints/` — downloaded checkpoints; ignored except marker file.
- `artifacts/results/` — canonical run JSON; ignored except marker file.
- `artifacts/workbench.duckdb` — rebuildable derived index; ignored.
- `third_party/` — synced official source trees; ignored.

## Installation

Ubuntu laptop requirements: Python 3.11+, Node 20+, npm. Backend dependencies are isolated from all model environments:

```bash
python3 -m venv .venv-workbench
source .venv-workbench/bin/activate
pip install -r workbench/backend/requirements.txt
cd workbench/frontend
npm install
```

The backend starts with no checkpoint installed. Models then show `CHECKPOINT NOT DOWNLOADED`; run control remains disabled. Browser never downloads weights.

`registry/models.yaml` records official/author-linked source URL, exact repository commit, required `source_dir`, native protocol, paper sanity scores, and checkpoint variants. `source_dir` is shared by sync and adapters; no method-name-derived path exists.

```yaml
checkpoint_id: fiq_n02
filename: HABIT-FIQ_N0.2.pt
download_url: https://huggingface.co/iLearn-Lab/AAAI26-HABIT/resolve/main/fiq/HABIT-FIQ_N0.2.pt
checkpoint_training_noise_pct: 20
evaluation_noise_pct: 0
expected_sha256: null
status: NO_CLEAN_CHECKPOINT
```

`checkpoint_training_noise_pct` and `evaluation_noise_pct` are different. A 20%-noise-trained model evaluated at 0% is **not** a clean-trained or “0% checkpoint.” `expected_sha256: null` means authors did not publish a hash; later local computed hashes go only into ignored `artifacts/checkpoints/download_manifest.json`.

Audit details: [docs/UPSTREAM_AUDIT.md](docs/UPSTREAM_AUDIT.md).

## External source repositories

Source sync never downloads checkpoints and never updates a pin silently:

```bash
python workbench/scripts/sync_upstreams.py --list
python workbench/scripts/sync_upstreams.py --model encoder
python workbench/scripts/sync_upstreams.py --all
python workbench/scripts/sync_upstreams.py --all --fetch
```

It clones missing repositories beneath `workbench/third_party/`, then detached-checks out recorded commit and verifies `HEAD`. Source sync and checkpoint download are separate operations.

Each model needs its own official dependency environment. CSMCIR upstream documents Python 3.9, Torch 2.0.1 and torchvision 0.15.2. ENCODER documents its own `requirements.txt`; HINT, Air-Know, ConeSep, HABIT and INTENT document Python/Torch/LAVIS combinations in their READMEs. These are runtime-unverified here; preserve upstream versions rather than combining model dependencies.

## Checkpoints

List records without network download:

```bash
python workbench/scripts/download_checkpoints.py --list
python workbench/scripts/download_checkpoints.py --model csmcir --dry-run
python workbench/scripts/download_checkpoints.py --all --dry-run
```

`--dry-run` performs no network access or artifact mutation. Actual download uses `filename.part`; resumes only after an HTTP `206` with a matching `Content-Range`. A server ignoring a range request (`200`) restarts safely from byte zero. It atomically renames after completion, computes local SHA-256, refuses different existing content unless `--force-redownload`, and writes local manifest only.
Later individual downloads:

```bash
# Original split
python workbench/scripts/download_checkpoints.py --model csmcir
python workbench/scripts/download_checkpoints.py --model airknow --checkpoint fiq_n05
python workbench/scripts/download_checkpoints.py --model conesep --checkpoint fiq_n02
python workbench/scripts/download_checkpoints.py --model habit --checkpoint fiq_n02
python workbench/scripts/download_checkpoints.py --model intent --checkpoint fiq_n02

# Val split
python workbench/scripts/download_checkpoints.py --model hint
# ENCODER needs registry direct Google Drive file URL before this command can download.
python workbench/scripts/download_checkpoints.py --model encoder

# PAIR mapping remains unresolved; do not choose either variant automatically.
python workbench/scripts/download_checkpoints.py --model pair --checkpoint pair_b1
python workbench/scripts/download_checkpoints.py --model pair --checkpoint pair_b2
```

PTHA + MTST intentionally has no FashionIQ download command: no official FashionIQ fine-tuned checkpoint is verified. `--all` skips blocked models and unresolved PAIR mappings; run `--all --dry-run` first. Noise-trained checkpoints retain training-noise metadata.

Verify local files without network:

```bash
python workbench/scripts/download_checkpoints.py --verify-only --all
python workbench/scripts/download_checkpoints.py --model hint --verify-only
```

Expected layout later:

```text
workbench/artifacts/checkpoints/
├── csmcir/fashioniq_tuned_clip_best.pt
├── encoder/fashioniq.pt
├── hint/fashioniq.pt
└── pair/pair-B1.pt
```

## Later evaluation flow

1. Sync one pinned official source.
2. Build its documented isolated environment on GPU host.
3. Download and verify exact checkpoint.
4. Run upstream evaluation unmodified; capture command, environment, stdout, stderr, commit, hash and local metrics.
5. Compare paper metrics as sanity only.
6. Add observation-only ranking export patch only after metric parity.
7. Write canonical per-query JSON and rebuild index.

Future request validation:

```bash
python workbench/scripts/run_eval.py \
  --model encoder --checkpoint fashioniq \
  --protocol fashioniq_val_split \
  --dataset-root "$CIR_DATA_ROOT/FashionIQ" \
  --output workbench/artifacts/results/fashioniq_val_split/encoder/fashioniq.json \
  --top-k 200
```

Adapters reject unsupported protocol/checkpoint pairs and missing checkpoints. A manually installed verified-mapping checkpoint can be runnable even when no automatic direct download URL exists. Availability reports source metadata, local source sync, automatic download availability, local SHA, official SHA status, mapping status, command readiness, runtime verification, and runnable state separately. No adapter imports model code into backend process.
CSMCIR later requires an audited upstream working-directory/data layout because its official evaluator has no ordinary dataset-root CLI. ENCODER additionally requires the upstream `./open_clip_pytorch_model.bin` ViT-B-32 backbone asset; its FashionIQ checkpoint alone is insufficient. Neither asset is downloaded by this workbench.

## Results and index

Canonical result files live below:

```text
workbench/artifacts/results/
├── fashioniq_original_split/csmcir/fashioniq.json
└── fashioniq_val_split/encoder/fashioniq.json
```

Each stores run provenance, training/evaluation noise, paper and local metrics, canonical query identity, raw captions, model input text, exact target rank and top results. Schema v2 adds `run_id` artifact identity, `checkpoint_id`, `top_k_saved`, and `gallery_size`. v1 is rejected rather than silently migrated. See [docs/RESULT_SCHEMA.md](docs/RESULT_SCHEMA.md).

DuckDB is a rebuildable serving index with normalized `runs`, `queries`, and `top_results` tables. Normal API browsing uses SQL metadata queries and paginated query retrieval; canonical JSON remains source of truth. Cross-run analytics load only aligned selected-run fields and requested Top-K rows.

```bash
python workbench/scripts/rebuild_index.py
```

## UI guide

- **Sample Explorer** — reference/target local image endpoint with `.png`, `.jpg`, and `.jpeg` known FashionIQ layouts; safe image IDs only; graceful browser placeholders when unavailable. It paginates queries and supports top 1/5/10/20/50/100/200.
- **Compare Models** — same canonical query across compatible selected runs; run identity remains `run_id`.
- **Common Failures** — consensus fail fraction at K, median/mean/worst rank, common distractors counted per run, and top-K Jaccard.
- **Disagreement / Failure Analytics** — rank range/std, universal failure/success and one-run-win inspection; run-level Jaccard matrix.
- **Annotations** — multi-label dataset-query notes keyed by `(protocol_id, query_id)`; never mutate raw run JSON.
- **Hypotheses** — save same-protocol cohort definition, selected run IDs and query IDs; JSON/CSV/Markdown exports include cohort metrics.

## Research workflow example

1. Choose **FashionIQ — Original Split**.
2. Select completed compatible original-split runs.
3. Filter **all selected models fail @10**.
4. Sort by median target rank.
5. Inspect same top-ranked distractors.
6. Annotate `preservation_failure` where evidence supports it.
7. Save `H01 — Common failure at R@10` with query IDs and exact run IDs.
8. Export JSON/CSV/Markdown and compare cohort R@10 against full benchmark.

This converts visual observations into a reproducible hypothesis, not a conclusion.

## Scientific Integrity Guards

1. Cross-run analysis requires one protocol, exact query-ID universe, and identical canonical category, annotation index, reference ID, target ID, and raw captions. `model_input_text` is intentionally allowed to differ.
2. Analysis uses `run_id`, never `model_id`; checkpoint variants cannot overwrite one another.
3. Top-K retrieval analyses reject saved depth below requested K. Target-rank cohort metrics do not pretend top-K rows exist.
4. Rebuild rejects duplicate `run_id` values and writes a temporary DuckDB before atomic replacement. A malformed result file leaves prior index intact.
5. Structured API guard failures expose `cross_protocol`, `query_alignment_mismatch`, `canonical_query_mismatch`, `insufficient_top_k_depth`, and checkpoint mapping state.
6. Image lookup accepts only benchmark image IDs and known FashionIQ `.png`/`.jpg`/`.jpeg` locations; HTTP paths never select arbitrary filesystem files.

## Scientific warnings

1. Never mix Original Split and Val Split metrics or analysis cohorts.
2. Paper scores are not locally reproduced scores.
3. Noise-trained checkpoint plus clean evaluation is not a clean-trained model.
4. Missing checkpoint provenance excludes strict comparison.
5. Common failure is evidence to inspect, not proof of architectural flaw.
6. Manually examine ambiguous ground truth and dataset defects.
7. Never guess checkpoint mapping; PAIR remains unresolved until authors establish mapping.

## Script help

All scripts support `--help`:

```bash
python workbench/scripts/sync_upstreams.py --help
python workbench/scripts/download_checkpoints.py --help
python workbench/scripts/load_mock_results.py --help
python workbench/scripts/rebuild_index.py --help
python workbench/scripts/run_eval.py --help
```
