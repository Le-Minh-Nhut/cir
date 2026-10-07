# CIR Failure Analysis Workbench

Local operator tooling for FashionIQ composed-image-retrieval (CIR) failure analysis. It keeps official upstream evaluators authoritative: this repository records pinned-source/checkpoint facts, prepares guarded commands, validates canonical result artifacts, builds a local DuckDB serving index, and presents those artifacts in a browser. It does **not** implement or substitute a CIR model.

> **Status boundary:** no official model evaluation or reproduction has run in this checkout. Paper metrics are registry sanity references, never local metrics. Mock artifacts are development data, never research evidence.

## Operating rules

- Use one exact protocol per run, comparison, cohort, and export. `fashioniq_original_split`, `fashioniq_full_gallery_ref_excluded`, and `fashioniq_val_split` are incompatible cohorts; shared literature label `original` does not establish evaluator compatibility.
- Keep model environments isolated from workbench backend/frontend dependencies. Do not install upstream model requirements into the workbench environment.
- Sync to registry pins; do not update a source tree to an arbitrary revision.
- Download only registry-declared, author-linked direct URLs. A locally computed hash records local evidence; it does not become an official hash.
- Run each official evaluator unmodified and establish metric parity before adding observation-only ranking export. Official evaluation and future per-query export are distinct phases.
- Treat `run_id`, not `model_id`, as artifact identity. Distinct checkpoints or training conditions need distinct run IDs.

## Protocols and guards

| ID | Exact label | Literature split label | Gallery and target rank |
| --- | --- | --- | --- |
| `fashioniq_original_split` | FashionIQ — Full Gallery, Reference Eligible | `original` | Full ordered `image_splits/split.{category}.val.json`; reference remains eligible. CSMCIR. |
| `fashioniq_full_gallery_ref_excluded` | FashionIQ — Full Gallery, Reference Excluded | `original` | Full ordered `image_splits/split.{category}.val.json`; reference is excluded before Recall@K. Air-Know, ConeSep, HABIT, INTENT. |
| `fashioniq_val_split` | FashionIQ — Val Split | `val` | First-seen ordered union of validation reference/target IDs; remove reference before Recall@K. |

All use `dress`, `shirt`, and `toptee`, `captions/cap.{category}.val.json`, and clean evaluation (`evaluation_noise_pct: 0`). The literature `original` label names a gallery family, not identical ranking semantics. Full evidence: [docs/PROTOCOL_AUDIT.md](docs/PROTOCOL_AUDIT.md).

CSMCIR consumes standard FashionIQ `images`; iLearn model sources consume `resized_image/{category}` plus category correction dictionaries. Dictionary files are **local, manual, and UNVERIFIED** for iLearn methods: no automatic download or DQU-CIR substitution occurs. `prepare_dataset.py` validates only standard FashionIQ because CSMCIR link preparation must stay independent; strict model preflight validates each adapter's native layout.

## Repository tree and data ownership

```text
workbench/
├── backend/                 tracked API, schemas, registry reader, guards, index logic
├── config/workbench.env.example
├── docs/                    tracked audits and result contract
├── envs/                    tracked upstream environment notes; no environments
├── frontend/                tracked local React/Vite UI
├── registry/models.yaml     tracked source, checkpoint, protocol metadata
├── scripts/                 tracked operator entry points
├── third_party/             ignored pinned upstream clones
└── artifacts/
    ├── checkpoints/         ignored model assets; .gitkeep tracked
    ├── results/             ignored canonical JSON artifacts; .gitkeep tracked
    ├── workbench.duckdb     ignored, rebuildable serving index
    ├── annotations.json     ignored local annotations
    ├── cohorts.json         ignored local cohort definitions
    └── logs/                ignored generated logs
```

Repository-level `data/`, `*.pt`, `*.pth`, `*.ckpt`, frontend `node_modules/`, `frontend/dist/`, local `workbench.env`, source clones, downloaded checkpoints, result JSON, and derived artifacts are ignored. Keep generated material out of commits; canonical result JSON is the source of truth even though it is local/ignored.

## No-install environment policy

This workbench provides orchestration only. It does not create virtual environments, install Python packages, run `npm install`, download assets, or combine upstream dependencies. Use already-provisioned environments:

- **Workbench environment:** backend/UI dependencies only, if already installed.
- **One GPU environment per upstream:** use that pinned upstream's documented requirements. Do not merge model stacks.
- **Browser:** `serve_workbench.py` starts only dependencies already present and reports missing frontend dependencies.

Known upstream guidance is recorded in [envs/README.md](envs/README.md), not validated runtime compatibility.

## Local configuration

Copy `workbench/config/workbench.env.example` to ignored `workbench/config/workbench.env`, then set only needed `KEY=VALUE` entries. Precedence is explicit CLI option, process environment, `workbench/config/workbench.env`, then built-in default. Relative paths resolve against `CIR_REPO_ROOT`.

```dotenv
CIR_REPO_ROOT=
CIR_DATA_ROOT=data
FASHIONIQ_ROOT=data/FashionIQ
WORKBENCH_HOST=127.0.0.1
WORKBENCH_BACKEND_PORT=8000
WORKBENCH_FRONTEND_PORT=5173
WORKBENCH_CHECKPOINT_ROOT=workbench/artifacts/checkpoints
WORKBENCH_RESULTS_ROOT=workbench/artifacts/results
WORKBENCH_THIRD_PARTY_ROOT=workbench/third_party
```

All listed roots are operational: `CIR_DATA_ROOT` supplies the default `FASHIONIQ_ROOT`; the checkpoint, result, and third-party roots select their corresponding local stores. `doctor.py` resolves and reports this configuration; scripts with explicit root options use their passed paths. Do not put secrets in this file.

## Mock and manual workflows

### Deterministic mock UI data

```bash
python workbench/scripts/pipeline.py mock
python workbench/scripts/pipeline.py mock --serve
```

`pipeline.py mock` runs workbench preflight, generates mock results, validates them, and rebuilds the index. It skips serving unless `--serve` is supplied. With `--serve`, preflight also checks Node, npm, and frontend `node_modules`. `load_mock_results.py` removes only existing `**/mock` directories beneath its output root, then writes deterministic schema-v2 files. Default result root is `workbench/artifacts/results`; `--output-root PATH` changes it. The UI labels mock runs. Never present mock metrics, cohorts, or screenshots as experimental evidence.

`serve_workbench.py` sets `WORKBENCH_BACKEND_URL` to the loopback URL for its selected backend port. Vite reads that value and proxies `/api` for development and preview; use the same selected ports for backend and frontend.

### Manually imported real artifacts

After verified instrumentation creates canonical JSON, place it anywhere below the result root, not only the conventional path. Result discovery is recursive for every `*.json` except `.gitkeep`; each must parse as schema v2 and no two files may carry the same `run_id`.

```bash
python workbench/scripts/validate_results.py --file PATH [--strict-real]
python workbench/scripts/validate_results.py --root PATH [--strict-real]
python workbench/scripts/validate_results.py --all [--strict-real]
python workbench/scripts/rebuild_index.py --results-root PATH --check-only
python workbench/scripts/rebuild_index.py --results-root PATH --database PATH [--validate-first]
```

`--strict-real` makes missing `upstream_commit`, `checkpoint_sha256`, `command_digest`, or `environment_digest` block non-mock artifacts. Rebuild writes a temporary database and atomically replaces the target only after all artifacts load.

## Safe real-evaluation workflow

Use this sequence on a GPU host only. It is a guardrail and recordkeeping procedure, not a claim that any model is ready.

1. Inspect availability without mutation: `python workbench/scripts/doctor.py --all` or `--model MODEL_ID`; add `--json` for machine output.
2. Inspect pins with `python workbench/scripts/sync_upstreams.py --list`. Its missing clones are informational; use `--model MODEL_ID --verify-only` when absence, pin mismatch, or a dirty checkout must fail. Use `--model MODEL_ID --dry-run` before clone/fetch; existing repos change only with `--update-existing`.
3. Build that upstream's own documented GPU environment outside workbench. Do not claim compatibility from these notes.
4. Validate FashionIQ layout: `python workbench/scripts/prepare_dataset.py --dataset-root PATH --check-only`. For CSMCIR, satisfy its dedicated preparation below.
5. Inspect checkpoint records: `python workbench/scripts/download_checkpoints.py --list`; use `--model MODEL_ID --dry-run` before a download. Verify installed files with `--verify-only`.
6. For CSMCIR, download verified Qwen captions with `python workbench/scripts/download_auxiliary_assets.py --model csmcir`. `COT_ours2/fashioniq` remains manual because its exact author download source is unverified.
7. Run the unmodified official evaluator and retain exact command, source pin, environment, checkpoint digest, stdout/stderr, and resulting aggregate metrics. Compare paper scores only as a sanity check.
8. Only after official metric parity is established, make a reviewable **observation-only** instrumentation change that exports the evaluator's existing per-query rankings. Validate its schema-v2 artifact and rebuild the index.

`evaluate_models.py` can print/run only audited official commands after source, pin, checkpoint, protocol, dataset, and model-specific prerequisites pass. It does not generate result JSON or prove reproduction. Use `--dry-run` until the reviewed environment is ready; dry run announces its plan and creates no evaluation log directory.

## Master pipeline

CSMCIR has one canonical guarded setup/evaluation command. It validates canonical FashionIQ, syncs pinned source, creates its fixed dataset link, downloads recorded checkpoint, downloads verified Qwen text, runs strict CSMCIR runtime preflight, then starts unmodified official aggregate evaluation only if every prerequisite passes:

```bash
python workbench/scripts/pipeline.py real \
  --model csmcir \
  --dataset-root "$FASHIONIQ_ROOT" \
  --sync-sources \
  --download-checkpoints \
  --download-auxiliary-assets \
  --evaluate
```

Use `--dry-run` first. The pipeline never installs environments, changes evaluator semantics, or treats aggregate official metrics as schema-v2 result JSON. `--rebuild-index` remains explicit and only belongs after an instrumented evaluation has written canonical result JSON.

Automatic CSMCIR work: pinned source checkout, `fashionIQ_dataset -> $FASHIONIQ_ROOT` link, registry checkpoint, and Qwen captions from immutable author Hugging Face revision `cf0c19bb346266c295b4b5772ebf03bd1c4f0467`. COT_ours2 captions remain the sole manual blocker: exact author acquisition source is unverified. Pipeline reports this explicitly at strict preflight; do not substitute or invent files.

Troubleshooting commands remain available: `doctor.py --scope real --model csmcir`, `sync_upstreams.py --model csmcir --dry-run`, `prepare_dataset.py --dataset-root "$FASHIONIQ_ROOT" --model csmcir --dry-run`, and `download_auxiliary_assets.py --model csmcir --dry-run`.

## Registry-derived checkpoint matrix

This matrix describes registry metadata and blockers, not present local files, runtime compatibility, command readiness, or successful reproduction. See [docs/UPSTREAM_AUDIT.md](docs/UPSTREAM_AUDIT.md).

| Model | Exact protocol | Literature label | Variants | Direct download state | Evaluation state |
| --- | --- | --- | --- | --- |
| CSMCIR | `fashioniq_original_split` | original | `fashioniq` clean | author-linked URL recorded | command audited; fixed-root/auxiliary-file limits apply |
| Air-Know | `fashioniq_full_gallery_ref_excluded` | original | `fiq_n05`, `fiq_n08` | author-linked URLs recorded | command not audited; only noise-trained variants |
| ConeSep | `fashioniq_full_gallery_ref_excluded` | original | `fiq_n02`, `fiq_n05`, `fiq_n08` | author-linked URLs recorded | command not audited; only noise-trained variants |
| PTHA + MTST | unavailable | n/a | none | no verified FashionIQ checkpoint | blocked |
| HABIT | `fashioniq_full_gallery_ref_excluded` | original | `fiq_n02`, `fiq_n05`, `fiq_n08` | author-linked URLs recorded | command not audited; only noise-trained variants |
| INTENT | `fashioniq_full_gallery_ref_excluded` | original | `fiq_n02`, `fiq_n05`, `fiq_n08` | author-linked URLs recorded | command not audited; only noise-trained variants |
| HINT | `fashioniq_val_split` | val | `fashioniq` clean | author-linked URL recorded | command not audited |
| ENCODER | `fashioniq_val_split` | val | `fashioniq` clean | Google Drive folder known; direct file URL unresolved | command audited; checkpoint and OpenCLIP asset both required |
| PAIR | `fashioniq_val_split` | val | `pair_b1`, `pair_b2` | no verified direct URL or FashionIQ mapping | blocked |

A checkpoint trained with 20%, 50%, or 80% noise remains that training condition when evaluated on clean FashionIQ data. It is not a clean checkpoint. `expected_sha256: null` means no official hash is in registry; local hashes belong in ignored download manifests.

## CSMCIR guide

CSMCIR is the only full-gallery/reference-eligible model with an audited command constructor. Its upstream evaluator is cwd/root-sensitive and does **not** accept an ordinary dataset-root argument. `evaluate_models.py` runs its command from `workbench/third_party/CSMCIR/src`; it requires `--dataset-root workbench/third_party/CSMCIR/fashionIQ_dataset` because upstream data must appear there.

Before evaluation, provide four independent prerequisites:

- **Base FashionIQ** under the canonical root: `captions/cap.{dress,shirt,toptee}.val.json`, `image_splits/split.{dress,shirt,toptee}.val.json`, and `images/*.png`.
- **Author-provided Qwen text** under that same canonical root: `qwen_captions/{dress,shirt,toptee}_cot_val.json`. This is CSMCIR auxiliary text, not standard FashionIQ.
- **Author-provided COT text** under the source root: `workbench/third_party/CSMCIR/COT_ours2/fashioniq/{dress,shirt,toptee}_cot_val.json`. Direct validation call graph confirms reads of these source-root files.
- **CSMCIR `fashioniq` model checkpoint** at its registry-selected local path.

`download_auxiliary_assets.py --model csmcir` downloads verified author-hosted Qwen files from immutable revision `cf0c19bb346266c295b4b5772ebf03bd1c4f0467`:

```text
https://huggingface.co/peng12138/CSMCIR/resolve/cf0c19bb346266c295b4b5772ebf03bd1c4f0467/fashioniq_qwen_captions/qwen_captions/{dress,shirt,toptee}_cot_val.json
  -> <FASHIONIQ_ROOT>/qwen_captions/{dress,shirt,toptee}_cot_val.json
```

It records local SHA-256 values in ignored `workbench/artifacts/auxiliary/download_manifest.json`; those are not official hashes. `COT_ours2/fashioniq` has no verified exact author download URL, remains required, and must be manually placed with known provenance. `--verify-only` requires both asset families.

Prepare/check its fixed dataset link without changing evaluator semantics:

```bash
python workbench/scripts/prepare_dataset.py --dataset-root PATH --model csmcir --check-only
python workbench/scripts/prepare_dataset.py --dataset-root PATH --model csmcir --dry-run
python workbench/scripts/prepare_dataset.py --dataset-root PATH --model csmcir
python workbench/scripts/download_auxiliary_assets.py --model csmcir --verify-only
```

The official phase emits aggregate metrics only. It is **not** a per-query result export and cannot populate canonical JSON alone. Establish official aggregate metric parity first; only then design and review an observation-only export patch.

## ENCODER guide

ENCODER uses `fashioniq_val_split`; its audited command passes `--fashioniq_split val-split`, `--fashioniq_path`, and `--ckpt_path`. The upstream loader concatenates the supplied root with child paths, so the adapter appends its required trailing `/`. It requires both a local FashionIQ checkpoint and the exact upstream-root `open_clip_pytorch_model.bin` asset used by `open_clip.create_model_and_transforms('ViT-B-32', pretrained='./open_clip_pytorch_model.bin')`.

The registry records only an official Google Drive folder for ENCODER's FashionIQ checkpoint. Its direct file URL, official hash, and downloaded file are unresolved/unverified. Pinned `evaluate_model.py` also imports missing `datasets1.py`; `doctor.py` and `evaluate_models.py` block it rather than assuming an untracked module. Do not invent a URL, substitute a backbone or module, or claim the checkpoint alone is sufficient. Only a reviewed acquisition with provenance can clear these blockers.

## Other model blockers

- Air-Know, ConeSep, HABIT, INTENT, and HINT have registry/checkpoint evidence but no audited official command constructor. Do not execute guessed commands.
- Every inspected iLearn model needs `resized_image/{dress,shirt,toptee}` and `captions/correction_dict_{dress,shirt,toptee}.json`; these correction assets are presence-only and unverified. Strict `doctor.py --scope real --model MODEL_ID` reports exact missing paths.
- PTHA + MTST has no verified author-linked FashionIQ fine-tuned checkpoint or eligible source integration.

## Results, provenance, and index

Schema v2 contract: [docs/RESULT_SCHEMA.md](docs/RESULT_SCHEMA.md). A result needs immutable `run_id`, exact protocol identity, separate literature split label, model/checkpoint identity, training/evaluation noise, gallery size, saved depth, paper references when available, locally reproduced metrics, and per-query canonical identity, raw captions, model input text, target rank, and contiguous unique ranked results.

Use `data_kind: "mock"` only for development fixtures. For an experiment, retain `upstream_commit`, `checkpoint_sha256`, `command_digest`, and `environment_digest`; `validate_results.py --strict-real` enforces these. Schema v1 is rejected rather than guessed/migrated. DuckDB normalizes runs, queries, and top results for serving only; rebuild it from JSON whenever artifacts change.

## UI pages

- **Dashboard:** registry, experiment-run, and mock-run counts; paper-score and protocol warnings.
- **Models:** registry status plus checkpoint-local, automatic-download, mapping, command, and runtime states.
- **Evaluation Runs:** choose compatible runs by immutable `run_id`.
- **Sample Explorer:** paginate/filter one run's queries; inspect reference, target, rank, captions, and saved retrievals.
- **Compare Models:** align selected compatible runs on one canonical query and inspect retrieved overlap.
- **Common Failures:** filter all-fail/threshold/winner/easy sets, inspect consensus failure and common distractors, save a cohort.
- **Disagreement:** inspect rank range and selected-run disagreement.
- **Failure Analytics:** inspect failure-set Jaccard and query-by-run target-rank matrix.
- **Annotations:** save local multi-label notes keyed by `(protocol_id, query_id)` without mutating raw result JSON.
- **Hypotheses:** list saved same-protocol cohorts. A cohort preserves selected run IDs, query IDs, and filter definition; it is evidence organization, not a conclusion.

Image lookup accepts benchmark image IDs only and known FashionIQ `.png`, `.jpg`, and `.jpeg` layouts. Missing local images show safely as unavailable.

## Research example

1. Select only completed runs from one exact protocol, for example `fashioniq_original_split`; never combine it with `fashioniq_full_gallery_ref_excluded` only because both retain literature label `original`.
2. Choose Top-K 10, then filter **all runs fail**.
3. Sort/inspect queries by median target rank and repeated distractors.
4. Review reference, target, raw captions, model input text, and actual rankings before labeling a failure.
5. Store supported labels/notes in **Annotations**.
6. Save cohort `H01-common-failure-r10` with its run IDs and query IDs.
7. Compare that cohort with its full benchmark population, while reporting protocol, checkpoint conditions, saved depth, and provenance.

This tests a reproducible hypothesis. It does not prove an architectural cause.

## Troubleshooting

| Symptom | Meaning and action |
| --- | --- |
| `BLOCKED` from `doctor.py` | Read reported prerequisite; it performs no repair. Fix only verified local paths/assets. |
| Source pin missing/mismatch/dirty | `sync_upstreams.py --list` reports missing clones without failure. Use `sync_upstreams.py --model ID --verify-only` for strict local verification; `--update-existing` is explicit. |
| FashionIQ layout missing | Point `--dataset-root` at root containing `captions`, `image_splits`, and `images`; do not point at one subdirectory. |
| CSMCIR layout/auxiliary blocker | Supply base FashionIQ, canonical-root `qwen_captions`, source-root `COT_ours2/fashioniq`, and `fashioniq` checkpoint at their exact paths. Use `download_auxiliary_assets.py --model csmcir --verify-only`; it does not download files. |
| ENCODER asset blocker | Both `fashioniq.pt` and `open_clip_pytorch_model.bin` are required. Direct checkpoint URL remains unresolved; do not substitute one. |
| Checkpoint download blocked | Registry lacks direct URL/mapping or local file conflicts. Inspect with `--list`; use `--force-redownload` only for intentional replacement. |
| `adapter command not audited` | No verified command exists. Do not derive flags from a guessed README command. |
| Evaluation logs needed | A real evaluator run writes `command.json`, `stdout.log`, and `stderr.log` in ignored `workbench/artifacts/logs/<UTC timestamp>_<model>_<checkpoint>/`. Dry runs create none. |
| Duplicate run ID or schema failure | Correct/regenerate artifact; do not edit derived database to hide it. |
| Top-K analysis blocked | Selected artifacts did not save requested depth. Lower K or regenerate after verified instrumentation. |
| Frontend dependency/port blocker | Use pre-provisioned frontend dependencies or select free distinct ports. Script never installs packages. |

## Adding a model

1. Record source ownership, pinned commit, source directory, supported protocol, checkpoint evidence, training noise, and mapping status in `registry/models.yaml`.
2. Update [docs/UPSTREAM_AUDIT.md](docs/UPSTREAM_AUDIT.md) with evidence and unresolved facts. Do not add invented URL/hash/command claims.
3. Add an adapter only after auditing exact official evaluator semantics and its protocol. Keep it unaudited rather than guessing.
4. Add model-specific guards for required non-checkpoint assets or fixed layouts.
5. Run official aggregate evaluation unmodified in its isolated environment and record provenance/metric evidence.
6. After parity, add observation-only per-query export, write schema-v2 artifacts, and use validation/rebuild.

## Safe updates and cleanup

Before source maintenance, inspect with `sync_upstreams.py --list` and preserve recorded pins. `--fetch` contacts remotes; `--update-existing` changes a clean existing checkout. Neither should be used as an implicit update path. Update registry pin and audit evidence together only after review.

Preview/remove only supported generated outputs:

```bash
python workbench/scripts/clean_generated.py --all-generated --dry-run
python workbench/scripts/clean_generated.py --mock-results
python workbench/scripts/clean_generated.py --database
python workbench/scripts/clean_generated.py --logs
python workbench/scripts/clean_generated.py --all-generated
```

Cleanup can remove mock result directories, DuckDB index, and logs only. It never removes experiment result JSON, checkpoints, source clones, or dataset files.

## Exact command reference

All operator scripts use `python workbench/scripts/NAME.py ...`. These are current flags; bracketed values are optional.

```text
# doctor.py
[--model MODEL_ID | --all] [--json]

# pipeline.py
`mock` | `prepare` | `real` | `serve` | `status`
[--dataset-root PATH] [--sync-sources] [--download-checkpoints]
[--download-auxiliary-assets] [--evaluate] [--rebuild-index] [--serve]
[--dry-run] [--continue-on-error] [--model MODEL_ID] [--protocol PROTOCOL_ID]
[--checkpoint CHECKPOINT_ID] [--top-k POSITIVE_INTEGER]

# prepare_dataset.py
--dataset-root PATH [--check-only] [--model MODEL_ID] [--dry-run]

# sync_upstreams.py
(--list | --model MODEL_ID | --all) [--fetch] [--update-existing] [--dry-run] [--verify-only] [--output-root PATH]

# download_checkpoints.py
(--list | --model MODEL_ID | --all) [--checkpoint CHECKPOINT_ID] [--dry-run] [--verify-only] [--force-redownload] [--output-root PATH]

# download_auxiliary_assets.py
(--list | --model csmcir) [--dry-run] [--verify-only] [--force-redownload]

# evaluate_models.py
(--list | --model MODEL_ID | --all-runnable) [--checkpoint CHECKPOINT_ID]
[--protocol fashioniq_original_split|fashioniq_full_gallery_ref_excluded|fashioniq_val_split] [--dataset-root PATH]
[--top-k POSITIVE_INTEGER] [--dry-run] [--continue-on-error]

# run_eval.py — legacy command preview; pipeline uses guarded evaluate_models.py
--model MODEL_ID --checkpoint CHECKPOINT_ID
--protocol fashioniq_original_split|fashioniq_full_gallery_ref_excluded|fashioniq_val_split --dataset-root PATH --output PATH
[--top-k INTEGER]

# validate_results.py
(--file PATH | --root PATH | --all) [--strict-real]

# rebuild_index.py
[--results-root PATH] [--database PATH] [--check-only] [--validate-first]

# serve_workbench.py
[--backend-only | --frontend-only] [--host HOST] [--backend-port PORT]
[--frontend-port PORT] [--open-browser] [--production-frontend] [--dry-run]

# clean_generated.py — select one or more
[--mock-results] [--database] [--logs] [--all-generated] [--dry-run]

# load_mock_results.py
[--output-root PATH]
```

`sync_upstreams.py --verify-only` cannot be combined with `--fetch`; `serve_workbench.py --production-frontend` requires frontend operation; backend and frontend ports must differ when both run. Use script `--help` for argparse wording, not for readiness claims.
