# Baseline — honest starting state

Recorded 2026-11-08, local 11:29–11:32 (+0700), UTC 04:29–04:32, by an agent that is one of
several editing this repo concurrently. Every number below came from a command run in this session;
the raw outputs are the files listed at the end.

**Read this first: the working tree is dirty and it is not this agent's doing.** Six other agents
were writing to the repo while these measurements were taken. Byte counts, `git status`, and API
run counts are point-in-time snapshots, not stable properties.

---

## 1. Git

```console
$ git rev-parse --abbrev-ref HEAD
research/cir-failure-workbench

$ git rev-parse HEAD
6421188f4f93574b7fe093a804dd235d39e630e8

$ git rev-parse origin/research/cir-failure-workbench
6421188f4f93574b7fe093a804dd235d39e630e8

$ git status --short
?? workbench/docs/probe_out.txt
?? workbench/docs/reviews/20261108T042920Z-baseline/
?? workbench/docs/reviews/20261108T042931Z/
```

- HEAD **equals** the remote-tracking ref: local `6421188f4f93574b7fe093a804dd235d39e630e8`,
  commit date `2026-11-08 11:19:31 +0700`, subject *"State that the named SHA precedes the commit
  carrying it"*.
- Remote is `origin git@github.com:Le-Minh-Nhut/cir.git` (fetch and push).
- The tree is **dirty with untracked files only** — no tracked file was modified at snapshot time.
  Two of the three untracked entries are other agents' output directories; the third,
  `workbench/docs/probe_out.txt`, is also not mine. No stash entries exist (`git stash list` → 0).
- Because siblings were mid-write, this snapshot is already stale: re-running `git status` later in
  the session showed the same three entries, but there is no guarantee a later run does.

## 2. Service actually serving

Command:

```console
$ ss -ltnp | grep -E ':8000|:5173'
$ pgrep -af 'serve_workbench|uvicorn|vite preview'
```

| PID | PPID | Role | Command line |
|---|---|---|---|
| 3213589 | 21761 (`omp` worker daemon) | launcher | `workbench/envs/ctl312h/bin/python workbench/scripts/serve_workbench.py --production-frontend --host 127.0.0.1 --backend-port 8000 --frontend-port 5173` |
| 3213598 | 3213589 | backend | `/home/heheboiz/data/cir/workbench/envs/ctl312h/bin/python -m uvicorn workbench.backend.main:app --host 127.0.0.1 --port 8000` |
| 3213599 | 3213589 | frontend supervisor | `npm run preview --host 127.0.0.1 --port 5173 --strictPort` |
| 3213610 | 3213599 | shell | `sh -c vite preview --host 127.0.0.1 --port 5173 --strictPort` |
| 3213611 | 3213610 | frontend | `node /home/heheboiz/data/cir/workbench/frontend/node_modules/.bin/vite preview --host 127.0.0.1 --port 5173 --strictPort` |

- All five started `Sun Nov 8 09:23:11 2026`; cwd of both Python processes is `/home/heheboiz/data/cir`.
- Listening sockets: `127.0.0.1:8000` → pid 3213598 (uvicorn); `127.0.0.1:5173` → pid 3213611 (vite).
  Both bound to loopback only. Nothing is on `0.0.0.0:8000`.
- Supervisor is the `omp` worker daemon (pid 21761), **not** systemd: no `systemctl --user` unit
  matches serve/uvicorn/vite. So the service is owned by a previous agent session, not by init.
- Production frontend is in use: `serve_workbench.py` line 52 maps `--production-frontend` to
  `npm run preview`, which serves the built `workbench/frontend/dist/`.

### Frontend bundle served

```
workbench/frontend/dist/index.html         271 bytes   2026-11-08 09:32:17 +0700
workbench/frontend/dist/assets/index-CsxrZyq6.js   277 236 bytes
workbench/frontend/dist/assets/index-uzz8fv-H.css   12 034 bytes
```

`dist/index.html` references `/assets/index-CsxrZyq6.js` and serves that exact reference:

```console
$ curl -s http://127.0.0.1:5173/            # HTTP 200, same markup as dist/index.html
$ curl -s -o /dev/null -w '%{http_code} %{size_download}' \
    http://127.0.0.1:5173/assets/index-CsxrZyq6.js
200 277236
```

Verified byte-identical, not just name-identical:

```
sha256(dist/assets/index-CsxrZyq6.js) = fe38b77d252440c3a158070bc2dbfff50c1c41be0a21c0e1ebf0663df1ea169e
sha256(served /assets/index-CsxrZyq6.js) = fe38b77d252440c3a158070bc2dbfff50c1c41be0a21c0e1ebf0663df1ea169e
```

So the browser gets the bundle currently sitting in `dist/`. Note the ordering: the server started at
09:23, `dist/` was written at 09:32, and the bytes served now equal the 09:32 file — the only
consistent explanation is that `vite preview` reads `dist/` from disk per request rather than
snapshotting it at launch `[INFERENCE from those two facts]`. Anyone editing `workbench/frontend/src`
still has to rebuild `dist/` for the running server to pick changes up.

## 3. Entrypoints

### `workbench/scripts/serve_workbench.py`

```console
$ workbench/envs/ctl312h/bin/python workbench/scripts/serve_workbench.py --help
usage: serve_workbench.py [-h] [--backend-only | --frontend-only]
                          [--config CONFIG] [--host HOST]
                          [--backend-port BACKEND_PORT]
                          [--frontend-port FRONTEND_PORT] [--open-browser]
                          [--production-frontend] [--dry-run]

Launch workbench backend and frontend without installing dependencies.
```

`--config` is autopilot YAML whose resolved paths override `workbench.env` for the served backend.
Live invocation uses `--production-frontend --host 127.0.0.1 --backend-port 8000 --frontend-port 5173`.

### Evaluation scripts

`workbench/scripts/evaluate_models.py` — guarded official-reproduction runner:

```console
usage: evaluate_models.py [-h] (--list | --model MODEL | --all-runnable)
                          [--checkpoint CHECKPOINT]
                          [--protocol {fashioniq_full_gallery_ref_excluded,fashioniq_original_split,fashioniq_val_split}]
                          [--dataset-root DATASET_ROOT]
                          [--canonical-dataset-root CANONICAL_DATASET_ROOT]
                          [--top-k TOP_K] [--run-id RUN_ID] [--dry-run]
                          [--verify-gpu] [--verify-gpu-only] [--keep-temp]
                          [--continue-on-error]
```

`workbench/scripts/composition_eval.py` — opt-in CIRCUS composition CLI:

```console
usage: composition_eval.py [-h] [--config CONFIG] [--models MODELS]
                           [--subsets SUBSETS] [--resume] [--smoke [SMOKE]]
                           [--plan] [--prepare-data] [--seed SEED]
```

`--subsets` accepts `full,sf,validated`; `--smoke` limits query count while always preserving the
full gallery; `WORKBENCH_CIRCUS_SYNTHETIC_FIXTURE=1` is documented test-only.

`workbench/scripts/pipeline.py` — orchestrator wrapping the above:

```console
usage: pipeline.py [-h] [--dataset-root DATASET_ROOT] [--sync-sources]
                   [--download-checkpoints] [--download-auxiliary-assets]
                   [--stage-caption-aliases] [--evaluate] [--rebuild-index]
                   [--serve] [--dry-run] [--continue-on-error] [--resume]
                   [--force-stage FORCE_STAGE] [--apply] [--allow-network]
                   [--allow-large-downloads] [--allow-env-install]
                   [--allow-preparation] [--allow-gpu-eval]
                   [--allow-gpu-preparation] [--verify-gpu]
                   [--allow-index-write] [--model MODEL] [--all-models]
                   [--protocol PROTOCOL] [--checkpoint CHECKPOINT]
                   [--top-k TOP_K] [--report REPORT]
                   {mock,prepare,real,serve,status,setup,reproduce,analyze,all,report}
```

`workbench/scripts/run_eval.py` — validates a request and *prints* the official command; never
executes it (`--model/--checkpoint/--protocol/--dataset-root/--output`, `--top-k`).

## 4. Roots

From `workbench/config/workbench.env` (secrets redacted; the file contained no credential-shaped
keys — only paths):

```
CIR_REPO_ROOT=/home/heheboiz/data/cir
CIR_DATA_ROOT=/home/heheboiz/data/cir/data
FASHIONIQ_ROOT=/home/heheboiz/data/cir/data/FashionIQ
WORKBENCH_ARTIFACTS_ROOT=/home/heheboiz/data/cir/workbench/artifacts
WORKBENCH_CHECKPOINT_ROOT=/home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads
WORKBENCH_RESULTS_ROOT=/home/heheboiz/data/cir/workbench/artifacts/results
WORKBENCH_THIRD_PARTY_ROOT=/home/heheboiz/data/cir/workbench/third_party
WORKBENCH_CIRCUS_RELEASE_ROOT=/home/heheboiz/data/cir/workbench/artifacts/circus_release
```

### Live process environment

```console
$ tr '\0' '\n' < /proc/3213598/environ | grep -E '^(WORKBENCH_|CIR_|FASHIONIQ)'
WORKBENCH_CIRCUS_RELEASE_ROOT=/home/heheboiz/data/cir/workbench/artifacts/circus_release
```

Same single variable for pid 3213589; pid 3213611 (vite) additionally carries
`WORKBENCH_BACKEND_URL=http://127.0.0.1:8000` and `NODE_ENV`-adjacent node vars.

**The backend does not inherit the rest of `workbench.env` as environment variables.** Only
`WORKBENCH_CIRCUS_RELEASE_ROOT` was exported when it was launched. The remaining values are read
from the file at request time by `workbench/backend/model_settings.py` (documented precedence:
process environment, then `workbench.env`, then the persisted
`<artifacts>/environment_assignment.json`). Two consequences worth stating plainly:

- Reading the file and reading the live env give different answers for every other root. The file is
  authoritative for the running backend, and `print_roots.py` confirms the effective resolution.
- `<artifacts>/environment_assignment.json` **does not exist** on disk, so the third precedence tier
  contributes nothing.

`workbench/scripts/print_roots.py` (interpreter `workbench/envs/ctl312h/bin/python`), rc=0:

```
autopilot_yaml.paths.registry=/home/heheboiz/data/cir/workbench/registry/models.yaml
autopilot_yaml.paths.dataset_root=/home/heheboiz/data/cir/data/FashionIQ
autopilot_yaml.paths.artifacts=/home/heheboiz/data/cir/workbench/artifacts
autopilot_yaml.paths.checkpoints=/home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads
autopilot_yaml.paths.results=/home/heheboiz/data/cir/workbench/artifacts/results
autopilot_yaml.paths.third_party=/home/heheboiz/data/cir/workbench/third_party
standalone.dataset_root=/home/heheboiz/data/cir/data/FashionIQ
standalone.artifacts=/home/heheboiz/data/cir/workbench/artifacts
standalone.checkpoints=/home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads
standalone.results=/home/heheboiz/data/cir/workbench/artifacts/results
standalone.third_party=/home/heheboiz/data/cir/workbench/third_party
CIR_REPO_ROOT=/home/heheboiz/data/cir
resolved_registry=/home/heheboiz/data/cir/workbench/registry/models.yaml
verdict=MATCH
```

`verdict=MATCH`: the standalone resolver and the autopilot YAML agree.

## 5. Datasets

Byte counts are `du -sb` (apparent bytes, includes directory entry overhead).

| Path | Bytes | Notes |
|---|---:|---|
| `data/` (`CIR_DATA_ROOT`) | 25 424 200 455 | |
| `data/FashionIQ` | 1 983 245 924 | |
| `data/FashionIQ/images` | 991 916 036 | **74 381 files** — the gallery |
| `data/FashionIQ/captions` | 403 551 063 | 22 files |
| `data/FashionIQ/resized_image` | 288 392 173 | 3 files |
| `data/FashionIQ/resized_images` | 286 225 672 | 15 415 files |
| `data/FashionIQ/qwen_captions` | 11 762 668 | |
| `data/FashionIQ/image_splits` | 1 398 312 | |
| `data/CIRR` | 23 195 580 630 | dominant cost of `data/` |
| `data/cirr` | 29 106 482 | |
| `data/stage1_relation_probe` | 216 267 419 | |
| `workbench/third_party` | 26 260 038 415 | vendored upstreams |

The 74 381-image gallery is `data/FashionIQ/images` — the count matches the `gallery_count` the API
reports for every composition run, and `gallery_identity_sha256` in those runs is
`1db68c59d96f014cdbecf936e108653de98abef0bd9a98a8abc2fe08b5dec4b4`.

### M-BEIR / CIRCUS release

`WORKBENCH_CIRCUS_RELEASE_ROOT` = `workbench/artifacts/circus_release`, total **34 708 451 bytes**.

| Bytes | File |
|---:|---|
| 11 889 854 | `mbeir_fashioniq_task7_cand_pool.jsonl` |
| 1 926 189 | `mbeir_fashioniq_task7_test.jsonl` |
| 19 404 614 | `shortcut_audit/retrieval_data_fashioniq_task7_aggregated_best_rank_mm_any.json` |
| 1 298 794 | `shortcut_free_subsets/fashioniq/shortcut_free/mbeir_fashioniq_task7_test_multimodal.jsonl` |
| 118 644 | `final_dataset/query_jsonl/fashioniq/validated_solved/mbeir_fashioniq_task7_test_validated_solved.jsonl` |
| 70 356 | `final_dataset/query_jsonl/fashioniq/validated_unsolved/mbeir_fashioniq_task7_test_validated_unsolved.jsonl` |

Worth flagging: the stored release does not carry its own image set — it points at the 74 381
FashionIQ gallery above — and the API itself records the discrepancy
`paper states 18,853; pinned released audit artifact contains 74,381`. The paper number and the
release number differ; the runs use the release.

## 6. Checkpoints

`WORKBENCH_CHECKPOINT_ROOT` =
`workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads`, total
**25 076 335 372 bytes (24 GiB)** across 9 model directories, 11 weight files.

| Model dir | File | Bytes |
|---|---|---:|
| `csmcir` | `fashioniq_tuned_clip_best.pt` | 2 720 615 859 |
| `conesep` | `ConeSep-FIQ-N0.2.pt` | 2 721 428 966 |
| `ptha_mtst` | `tuned_mtst_cirr.pt` | 2 720 352 973 |
| `habit` | `HABIT-FIQ_N0.2.pt` | 2 730 715 930 |
| `intent` | `intent_fiq_0.2.pt` | 2 722 478 522 |
| `hint` | `fashioniq.pt` | 2 724 451 182 |
| `encoder` | `fashioniq.pt` | 631 879 365 |
| `limn` | `0_dress_best_model.pt`, `0_shirt_best_model.pt`, `0_toptee_best_model.pt` | 1 792 010 157 each |
| `tme` | `fashioniq_0_best_model.pth` | 2 728 382 104 |

Related trees: `workbench/artifacts` total **57 721 221 314 bytes (55 GiB)**;
`workbench/artifacts/checkpoints` 2 720 615 859 bytes; `workbench/artifacts/composition`
782 626 872 bytes; `workbench/artifacts/results` **0 bytes** (empty directory).

## 7. Interpreters: declared vs configured

Each model declares a Python version in `workbench/registry/models.yaml`; the configured path comes
from `WORKBENCH_<MODEL>_PYTHON` in `workbench/config/workbench.env`. Both columns below are literal.
"Actual" is `<path> -V`.

| Model | Env var | Declared | Configured path | Exists | Actual |
|---|---|---|---|---|---|
| csmcir | `WORKBENCH_CSMCIR_PYTHON` | 3.9 | *(unset)* | – | – |
| airknow | `WORKBENCH_AIRKNOW_PYTHON` | 3.8.10 | *(unset)* | – | – |
| conesep | `WORKBENCH_CONESEP_PYTHON` | 3.8.10 | `envs/airknow-adapted/bin/python` | yes | 3.11.15 |
| ptha_mtst | `WORKBENCH_PTHA_MTST_PYTHON` | 3.9 | `envs/airknow-adapted/bin/python` | yes | 3.11.15 |
| habit | `WORKBENCH_HABIT_PYTHON` | 3.8.10 | `envs/airknow-adapted/bin/python` | yes | 3.11.15 |
| intent | `WORKBENCH_INTENT_PYTHON` | 3.9 | `envs/airknow-adapted/bin/python` | yes | 3.11.15 |
| hint | `WORKBENCH_HINT_PYTHON` | 3.8.10 | `envs/airknow-adapted/bin/python` | yes | 3.11.15 |
| encoder | `WORKBENCH_ENCODER_PYTHON` | 3.9 | `envs/pair-adapted/bin/python` | yes | 3.11.15 |
| pair | `WORKBENCH_PAIR_PYTHON` | 3.8.10 | *(unset)* | – | – |
| clvc_net | `WORKBENCH_CLVC_PYTHON` | 3.7.6 | *(unset)* | – | – |
| dcnet | `WORKBENCH_DCNET_PYTHON` | 3.7.7 | *(unset)* | – | – |
| combiner_rn50x4_noft | `WORKBENCH_CLIP4CIR_PYTHON` | 3.8 | *(unset)* | – | – |
| clip4cir_rn50x4_fullft | `WORKBENCH_CLIP4CIR_PYTHON` | 3.8 | *(unset)* | – | – |
| tgcir | `WORKBENCH_TGCIR_PYTHON` | UNKNOWN | *(unset)* | – | – |
| sprc | `WORKBENCH_SPRC_PYTHON` | 3.9 | *(unset)* | – | – |
| limn | `WORKBENCH_LIMN_PYTHON` | UNKNOWN | `envs/limn-adapted/bin/python` | yes | 3.11.15 |
| tme | `WORKBENCH_TME_PYTHON` | 3.10 | `envs/tme-adapted/bin/python` | yes | 3.10.22 |

Control interpreter `workbench/envs/ctl312h/bin/python` → `Python 3.12.11` (matches AGENTS.md).

**8 configured paths, 8 exist, all runnable.** But only `tme` matches its declared version. Five
`3.8.10`/`3.9` declarations are served by `envs/airknow-adapted` at 3.11.15, and `encoder` (declares
3.9) points at `pair-adapted` 3.11.15. This is not an inference from the table alone — the served API
reports the same mismatch as a per-run failure on the runs that used those paths (see §11).

One runnable path is **not** in `workbench.env`: `ptha_mtst` composition runs record
`/home/heheboiz/data/cir/workbench/artifacts/environments/cir-005ef4e1b02999a5/bin/python`
(`Python 3.10.22`, status `RUNTIME_READY`, verified), supplied through the process environment of
whatever launched that run rather than the config file.

## 8. Disk and GPU

```console
$ df -h / /home
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme0n1p5  630G  461G  138G  78% /
/dev/nvme0n1p5  630G  461G  138G  78% /
```

`/` and `/home` are the **same filesystem** (`/dev/nvme0n1p5`, same device id 66309), so the 138 GB
free is one shared pool, not two.

```console
$ nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader
0, NVIDIA GeForce RTX 5070 Ti, 580.173.02, 16303 MiB, 1048 MiB, 14 %, 48, 26.23 W
```

CUDA 13.0 driver, compute mode Default, no ECC. Memory in use (1 048 MiB) is desktop processes only
(Xorg 320 MiB, gnome-shell 116 MiB, xdg-desktop-portal-gnome 5 MiB, firefox 408 MiB);
`nvidia-smi --query-compute-apps` returned **empty** — no CUDA compute process is running. The 14–26 %
utilization sampled across two reads is desktop/firefox compositing, not evaluation work. Host has
20 cores, 31 GiB RAM (24 GiB available).

## 9. Registry counts

```console
$ workbench/envs/ctl312h/bin/python -c "import yaml; d=yaml.safe_load(open('workbench/registry/models.yaml'))['models']; ..."
```

- Models in `workbench/registry/models.yaml`: **17** —
  `csmcir, airknow, conesep, ptha_mtst, habit, intent, hint, encoder, pair, clvc_net, dcnet,
  combiner_rn50x4_noft, clip4cir_rn50x4_fullft, tgcir, sprc, limn, tme`
- With a `composition_adapter`: **9** — `csmcir, conesep, ptha_mtst, habit, intent, hint, encoder,
  limn, tme`
- With a staged checkpoint directory under `WORKBENCH_CHECKPOINT_ROOT`: **9** —
  exactly the 9 adapter models. Every adapter model has weights staged; no non-adapter model does.
- With results under `workbench/artifacts/composition/summary/`: **9 directories total** —
  **7 real** + **2 synthetic fixtures**. The real ones cover **5 distinct models**: `encoder` (2),
  `limn` (2), `habit` (1), `ptha_mtst` (1), `tme` (1). The two fixtures are
  `SYNTHETIC_encoder_fashioniq_fixture001` and `SYNTHETIC_habit_fiq_n02_fixture002`; they must be
  excluded from any "models with results" count or the tally inflates to 6.

So: 9 of 17 models have an adapter; 9 of 9 adapter models have weights; only 5 of 9 adapter models
have ever produced a composition summary, and 2 of the 9 summary directories are synthetic fixtures.
Four adapter models — `csmcir`, `conesep`, `intent`, `hint` — have staged weights and a working
adapter but **zero** composition results. Note `habit` counts as "has a summary" while its only run
is a `FAILED`/`MODEL_UNAVAILABLE` one, so "has a summary directory" is not the same as "has a usable
result" — of the 5, only 3 (`encoder`, `ptha_mtst`, `tme`) reached `COMPLETED`.

**Two of the seven real summary directories are orphans** — they exist on disk but the API does not
serve them. Every summary directory name ends in a 24-hex run id; cross-checking against the five
`run_id` values in `/api/composition/runs`:

| Summary directory | Run id | Served by API? |
|---|---|---|
| `encoder_fashioniq_6a528e877c328772e2c7bb28` | `6a528e877c328772e2c7bb28` | yes |
| `encoder_fashioniq_eda30a37c0bc17ac43844f09` | `eda30a37c0bc17ac43844f09` | **no — orphan** |
| `habit_fiq_n02_e0dd04a46ef7a58334b9e353` | `e0dd04a46ef7a58334b9e353` | yes |
| `limn_base_iter0_all_categories_10173dc45dea8c90fd2a284b` | `10173dc45dea8c90fd2a284b` | yes |
| `limn_base_iter0_all_categories_3f86f58483b368377e2c151e` | `3f86f58483b368377e2c151e` | **no — orphan** |
| `ptha_mtst_cirr_transfer_315dd590eeb03d2c78d7937a` | `315dd590eeb03d2c78d7937a` | yes |
| `tme_fashioniq_0_ad477312aa57c1b9114f9e51` | `ad477312aa57c1b9114f9e51` | yes |

The two orphans are older builds of `encoder` and `limn` (mtimes 2026-11-07 23:38 and 2026-11-08
02:42, both earlier than the served counterparts at 00:29 and 04:51). So `workbench/artifacts/` holds
7 real composition result sets while the UI shows 5 — a naive on-disk count overstates what the
product displays by 2. `workbench/artifacts/composition/runs/` is separate and was not reconciled
here.

## 10. API — what the server actually answers

```console
$ curl -s -o body -w '%{http_code} %{size_download} %{time_total}' http://127.0.0.1:8000<path>
```

| Endpoint | Code | Bytes (1st read) | Body |
|---|---:|---:|---|
| `/api/composition/runs` | 200 | 117 041 | 15 runs (body grew to 117 133 bytes on the later read, same 15 `run_id`s) |
| `/api/runs` | 200 | 2 | `[]` |
| `/api/benchmark-runs` | 200 | 23 025 | 6 runs at first read, **8** on re-read |
| `/api/models` | 200 | 86 553 | 17 models |
| `/api/cohorts` | 200 | 2 | `[]` |
| `/api/analysis` | 422 | 93 | (requires query parameters) |

All three requested endpoints return **HTTP 200**. Raw bodies are saved as
`api_composition_runs.json`, `api_runs.json`, `api_benchmark_runs.json`.

Correction on provenance, so nobody over-trusts a filename: `api_benchmark_runs.json` and
`snap_a_benchmark.json` / `snap_b_benchmark.json` **all hold the later 8-run body** (27 153 bytes).
The first 6-run read (23 025 bytes) was overwritten when the same path was re-targeted for the
snapshot pass; it survives only as the size line in `api_raw.txt`, not as a body. The 6→8 count change
is real — it came from two independent `curl` invocations — but the 6-run JSON itself is gone.

### `/api/composition/runs` — 15 runs, 5 distinct runs × 3 subsets

| Model | Checkpoint | Subsets (query counts) | Status |
|---|---|---|---|
| `encoder` | `fashioniq` | full 6003, sf 4069, validated 586 | COMPLETED ×3 |
| `ptha_mtst` | `cirr_transfer` | full 6003, sf 4069, validated 586 | COMPLETED ×3 |
| `tme` | `fashioniq_0` | full 6003, sf 4069, validated 586 | COMPLETED ×3 |
| `limn` | `base_iter0_all_categories` | full 6003, sf 4069, validated 586 | **PARTIAL ×3** |
| `habit` | `fiq_n02` | full 6003, sf 4069, validated 586 | **FAILED ×3** |

Status split: 9 `COMPLETED`, 3 `PARTIAL`, 3 `FAILED`. Every run reports `is_synthetic: false`,
`gallery_count: 74381`, `protocol_id: circus_fashioniq_release_v1`.

- `limn` PARTIAL: all three conditions (`multimodal`, `image_only_empty_text`,
  `text_only_black_image`) report `category_scope_failure_count: 119`, `unscored_count: 119`. Its
  recorded interpreter is `envs/limn-adapted/bin/python`, status `DEPENDENCY_VERSION_MISMATCH`,
  `verified: false` (`torch: expected ==1.12.1, found 2.7.1+cu128`).
- `habit` FAILED: `RuntimeError: execution environment not verified
  [BLOCKED/PYTHON_VERSION_MISMATCH]: WORKBENCH_HABIT_PYTHON=…/envs/airknow-adapted/bin/python
  (expected Python 3.8.10, found 3.11.15)`; all 6003 queries `unscored`, condition status
  `MODEL_UNAVAILABLE`.

### `/api/benchmark-runs` — 6 at first read, 8 thirty seconds later

The count moved **during this session**. Another agent was writing runs. Latest read, 8 runs:

| Model | Runs | Parity | Gate |
|---|---:|---|---|
| `csmcir` | 2 | `PARITY_VERIFIED` | `PASS_WITHIN_5_PERCENT_VARIANT` |
| `synthetic_model` | 6 | `PARSER_UNAVAILABLE` | `FAILED_OUTPUT_VALIDATION` (blocker: `no reproduced metrics`) |

Both `csmcir` rows pass the gate only as a **variant** (`variant: true`, detail *"every comparable
metric is within 5 percent relative error"*), reference `csmcir_fiq_val_ours_vitg`, paper
`csmcir_acl26`. The six `synthetic_model` rows are evaluations that produced nothing to compare —
they are noise in the run count, not evidence.

### `/api/runs` — empty

`[]`. Zero rows. `/api/cohorts` is also `[]`. So the legacy runs surface is empty while the
composition and benchmark surfaces carry real data; anything that counts "runs" off `/api/runs`
reports 0.

## 11. Composition runtime interpreter per run (from the API payload)

| Model | Interpreter recorded | Verification |
|---|---|---|
| `habit` | `envs/airknow-adapted/bin/python` | `PYTHON_VERSION_MISMATCH` — expected 3.8.10, found 3.11.15 |
| `encoder` | `envs/pair-adapted/bin/python` | `PYTHON_VERSION_MISMATCH` — expected 3.9, found 3.11.15 |
| `limn` | `envs/limn-adapted/bin/python` | `DEPENDENCY_VERSION_MISMATCH` — torch expected ==1.12.1, found 2.7.1+cu128 |
| `ptha_mtst` | `artifacts/environments/cir-005ef4e1b02999a5/bin/python` | `RUNTIME_READY`, verified, Py 3.10.22 |
| `tme` | `envs/tme-adapted/bin/python` | `GPU_RUNTIME_READY`, verified, Py 3.10.22 |

Only `ptha_mtst` and `tme` ran on verified interpreters. `habit` is blocked and refuses to score.
`encoder` and `limn` ran anyway with unverified interpreters — `encoder` still produced COMPLETED
results with every condition at 0 unscored, which is the misleading combination: a green status on an
interpreter the system itself marks as mismatched. Controller interpreter for all runs was
`3.12.11 (main, Oct 7 2025, 15:34:39) [Clang 20.1.4]` on `NVIDIA GeForce RTX 5070 Ti`.

## 12. Model execution audit — was only DeepSeek in use?

`/home/heheboiz/.omp/agent/config.yml` read verbatim (not modified):

```yaml
modelRoles:
  advisor: 9router/wdb/deepseek-ai/DeepSeek-V4.1-Flash-1017:max
  gemini: 9router/ag/gemini-3.8-flash-high
  default: 9router/wdb/deepseek-ai/DeepSeek-V4.1-Flash-1017:high
task:
  perCallModelAllow:
    - 9router/ag/gemini-3.8-flash-high
    - 9router/wdb/deepseek-ai/DeepSeek-V4.1-Flash-1017
  perCallModel: "on"
  agentModelOverrides:
    task: 9router/wdb/deepseek-ai/DeepSeek-V4.1-Flash-1017:high
    sonic: 9router/wdb/deepseek-ai/DeepSeek-V4.1-Flash-1017:high
```

Classification:

| Setting | Literal value | DeepSeek? |
|---|---|---|
| `modelRoles.advisor` | `9router/wdb/deepseek-ai/DeepSeek-V4.1-Flash-1017:max` | **yes** |
| `modelRoles.default` | `9router/wdb/deepseek-ai/DeepSeek-V4.1-Flash-1017:high` | **yes** |
| `modelRoles.gemini` | `9router/ag/gemini-3.8-flash-high` | **no — Gemini** |
| `task.agentModelOverrides.task` | `9router/wdb/deepseek-ai/DeepSeek-V4.1-Flash-1017:high` | **yes** |
| `task.agentModelOverrides.sonic` | `9router/wdb/deepseek-ai/DeepSeek-V4.1-Flash-1017:high` | **yes** |
| `task.perCallModelAllow[0]` | `9router/ag/gemini-3.8-flash-high` | **no — Gemini** |
| `task.perCallModelAllow[1]` | `9router/wdb/deepseek-ai/DeepSeek-V4.1-Flash-1017` | **yes** |

### The ID sessions are actually spawned with

**`9router/wdb/deepseek-ai/DeepSeek-V4.1-Flash-1017`** at thinking level `high`, provider `9router`,
wire API `openai-completions`.

How it was determined — from the session transcript, not from intent. The parent session file is
`/home/heheboiz/.omp/agent/sessions/-data-cir/2026-11-05T17-22-31-104Z_01a1acbb-cc40-76fa-bfa0-7cde2257b89e/baseline.jsonl`
(also exposed to this process as `PI_SESSION_FILE`). Its records:

```json
{"type":"session", ..., "parentSession":".../2026-11-05T17-22-31-104Z_01a1acbb-....jsonl"}
{"type":"model_change", "model":"9router/wdb/deepseek-ai/DeepSeek-V4.1-Flash-1017",
 "resolvedModelIsFallback":false}
{"type":"thinking_level_change", "thinkingLevel":"high", "configured":null}
```

and every assistant message record carries the provider fields directly:

```json
"api":"openai-completions", "provider":"9router", "model":"wdb/deepseek-ai/DeepSeek-V4.1-Flash-1017"
```

Counting all records in that file: **one** `model_change` (the DeepSeek ID above,
`resolvedModelIsFallback: false` — so it is the configured model, not a fallback), and **85 of 85**
assistant messages report `provider: 9router`, `model: wdb/deepseek-ai/DeepSeek-V4.1-Flash-1017`
(count taken at the end of this session and still growing — the tail of the file is this agent's own
turns). No other provider/model pair appears anywhere in the transcript. `thinkingLevel` was `high`
for the whole session. The `:high` suffix in `modelRoles.default` is the thinking level, not part of
the wire model id — which is why the spawned string and the transcript's `model` field differ by that
suffix only.

### Verdict on Job 2

- Work in **this** session was executed **entirely by DeepSeek** — the single spawn-path model is
  `9router/wdb/deepseek-ai/DeepSeek-V4.1-Flash-1017`, thinking `high`, no fallback engaged, zero
  non-DeepSeek assistant messages.
- **A non-DeepSeek model is configured and reachable**, however: `modelRoles.gemini` and
  `task.perCallModelAllow[0]` both name `9router/ag/gemini-3.8-flash-high`, and `task.perCallModel`
  is `"on"`, so a per-call override to Gemini is permitted by configuration. That model appears
  **nowhere** in this session's spawn path or transcript. The truthful statement is "all work in this
  session ran on DeepSeek V4.1 Flash; Gemini is allowlisted for per-call use but was not used here."
- No model-level claim can be made about the *other* concurrent agents from this evidence — their
  transcripts are separate session files and were not read.

---

## Evidence files (all in this directory)

| File | Contents |
|---|---|
| `git.txt`, `git_info.json` | branch, HEAD, origin ref, status, remotes, commit date |
| `service_raw.txt`, `service_ps.txt` | `ss -ltnp`, `pgrep -af`, process table |
| `frontend.txt`, `frontend2.txt`, `frontend_raw.txt` | `dist/` listing, `index.html`, served HTML, asset codes, sha256 compare |
| `api_raw.txt`, `api_codes.txt` | per-endpoint code/size/time |
| `api_composition_runs.json` | raw `/api/composition/runs` body (15 runs) |
| `api_runs.json` | raw `/api/runs` body (`[]`) |
| `api_benchmark_runs.json` | raw `/api/benchmark-runs` body (later read, 8 runs — see the provenance correction in §10) |
| `snap_a_*.json`, `snap_b_*.json` | two consecutive snapshots, 30 ms apart; identical content, count already at 8 when they were taken |
| `api_models.json`, `api_cohorts.json`, `api_analysis.json` | supporting endpoints |
| `composition_summary.json`, `benchmark_summary.json` | trimmed per-run fields |
| `runtime_interpreters.json` | interpreter + verification status per composition run |
| `config_redacted.txt` | `workbench.env` with any credential-shaped value redacted |
| `live_env.txt` | `WORKBENCH_*`/`CIR_*` from `/proc/<pid>/environ` |
| `print_roots.txt` | `print_roots.py` output, `verdict=MATCH` |
| `interpreters.json`, `model_python.json`, `model_matrix.json` | declared vs configured interpreters; per-model adapter/staged-checkpoint/summary matrix |
| `help_serve.txt`, `help_evaluate_models.py.txt`, `help_composition_eval.py.txt`, `help_pipeline.py.txt`, `help_run_eval.py.txt` | `--help` surfaces |
| `disk_gpu.txt` | `du`/`df`/`nvidia-smi` reads |
| `data1.txt`, `models_head.txt` | dataset listing, registry header |
