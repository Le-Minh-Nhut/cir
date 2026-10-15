# Root-Cause Matrix — final hardening (HEAD e9cafaf)

Status: living document. Findings A.md–D.md (subagent audits) fold in as they land.

## Confirmed P0 (main-agent source trace, reproduced by reading actual flow)

### P0-A — Environment planner ignores the selected runtime adaptation
- **Root cause**: `workbench/backend/environment_planner.py::model_constraints()` derives
  `torch/torchvision/cuda/packages` solely from the historical `environment` block. The registry
  declares `environment.adapted_runtime` (limn: torch>=2.7,<3, torchvision>=0.22,<1, CUDA 12.8,
  open_clip_torch>=2.24) and `runtime.runtime_adaptation()` (runtime.py:765) resolves the
  operator selection (`WORKBENCH_RUNTIME_ADAPTATION=sm_120`), but the planner never calls it.
  Two components therefore operate on different dependency contracts: the verifier
  (`runtime.py:1217` uses `_adapted_packages` when adaptation selected) accepts torch>=2.7,
  while the installer plans torch==1.12.1. On a Blackwell host the installed historical torch
  cannot run sm_120 kernels at all.
- **Boundary affected**: planner ↔ solver ↔ installer ↔ verifier ↔ evaluator.
- **Fix (one canonical mechanism)**: a single resolved-contract helper in `runtime.py`
  (`resolved_runtime_environment(model)`) that applies the selected adaptation to the
  environment block; `model_constraints()` consumes it. Environment identity
  (`_environment_digest`) already hashes the constraint dict, so historical vs adapted plans get
  distinct `environment_id`s and never share a cache entry or env dir. The wheel index for the
  CUDA build matching the adapted CUDA version is declared in the registry's `adapted_runtime`
  block (`pip_index_url`) and used by the installer only under that profile — one
  registry-declared field, no new config mechanism.
- **Isolation**: adaptation applies only to models declaring the matching
  `target_compute_capability`; unrelated models are untouched (runtime_adaptation raises for a
  model without a matching declaration, so the planner treats it as "no adaptation").
- **Verification**: resolved install specs differ per profile; torch/torchvision pairing check
  still enforced; interpreter assignment per profile; verify_environment passes only against the
  matching profile.
- **Regression tests** (must FAIL on e9cafaf):
  1. `WORKBENCH_RUNTIME_ADAPTATION=sm_120` + limn → `model_constraints` returns adapted
     torch/torchvision/cuda/open_clip constraints.
  2. Same env + unrelated model (no adapted_runtime) → unchanged historical constraints.
  3. Unknown adaptation value → planner blocks with the registry-declared refusal.
  4. `pip_specs_for_group` under adaptation includes the declared wheel index.
  5. Plan/environment_id differs between profiles (cache invalidation).

### P0-B — Evaluator child never receives the runtime-adaptation selection
- **Root cause**: `workbench/scripts/evaluate_models.py::_evaluator_environment()` builds the
  untrusted-evaluator child env from a fixed allowlist that omits
  `WORKBENCH_RUNTIME_ADAPTATION`. The LIMN official command is `workbench/replay/limn.py`
  (adapters/models.py:132), whose `validate_environment()` (replay/limn.py:84) reads the same
  env var via `runtime_adaptation()` **inside the child process**. Orchestrator plans an adapted
  runtime; the child validates historical pins — mismatched contracts in the same run.
- **Fix**: add `WORKBENCH_RUNTIME_ADAPTATION` to `_EVALUATOR_ENV_ALLOWLIST` (a validated
  operator profile name, not a secret; deny-prefix logic unchanged; no blanket forwarding).
- **Regression tests** (must FAIL on e9cafaf): child env built by the production
  `_evaluator_environment` with the var set contains it; unrelated credentials still absent;
  unset → child env also unset (no accidental adoption).

## Confirmed P1

### P1-C — GPU doctor probes a CPU-only throwaway container
- **Root cause**: `workbench/docker/run.sh::cmd_doctor()` runs the nvidia-smi probe via
  `docker compose -f docker-compose.yml` (base file only), even when the stack was started with
  `WORKBENCH_ENABLE_GPU=1` + the GPU override. The probe therefore describes the wrong
  configuration, not the running stack.
- **Fix**: doctor uses the same resolved `COMPOSE_FILES` as every other command and probes the
  **actual running container** (`docker exec` when up), falling back to the resolved
  `compose run` only when nothing is running; nonzero exit when a CUDA-declaring model needs
  the GPU and it is not verified.
- **Regression test**: `run.sh doctor` with GPU override resolved execs the running container /
  includes `-f docker-compose.gpu.yml`.

### P1-D — Launcher never reads `.env`; two sources of truth
- **Root cause**: compose reads `--env-file .env` but run.sh's own decisions
  (`WORKBENCH_BACKEND_PORT`, `WORKBENCH_FRONTEND_PORT`, `WORKBENCH_ENABLE_GPU`, dataset paths)
  come from the shell only. `.env`-only configuration yields wrong BASE URL and no GPU override.
- **Fix**: one resolution point at the top of run.sh — parse `.env` with a strict parser (no
  eval), precedence explicit shell override > validated `.env` > defaults; all commands
  (`up/doctor/prepare/pilot/status/logs/stop`) consume the resolved values.
- **Regression tests**: `.env`-only ports produce correct BASE; shell override wins; malformed
  `.env` fails loudly; `WORKBENCH_ENABLE_GPU=1` in `.env` selects the GPU override.

### P1-E — Release verifier overstates evidence
- **Root cause**: `release_verify.sh` mixes dimensions: J5 proves the *gate contract* by
  source-string inspection, not a runtime probe; J7 certifies a synthetic single-category
  report as "complete-report rule"; J8 passes on an empty job list; overall verdict collapses
  all dimensions into one boolean, so `RELEASE_READY` can print while GPU/model execution was
  never exercised.
- **Fix**: split the report into explicit dimensions (STATIC_SOURCE, BUILD, DEPLOYMENT,
  CONFIGURATION, DATA_PREPARATION, MODEL_ENVIRONMENT, GPU_RUNTIME, MODEL_INFERENCE,
  SCIENTIFIC_OUTPUT, FRONTEND, RESEARCH_ANALYSIS), each PASS/FAIL/PENDING/NOT_APPLICABLE with
  evidence; verdict engine maps mandatory-dimension pendings to `GPU_VALIDATION_PENDING` /
  `ENGINEERING_READY` instead of `RELEASE_READY`.
- **Regression test**: verifier on a no-GPU host yields a verdict that is NOT RELEASE_READY;
  per-dimension statuses present in the JSON report.

### P1-F — Verifier restarts the production container
- **Root cause**: `release_verify.sh` J11 `docker restart`s the live workbench container. An
  active research evaluation is killed mid-run.
- **Fix**: J11 persistence check moves to a disposable container started from the same image
  with a temporary artifacts directory (isolated compose project name); the production stack is
  never touched. PENDING when docker/image unavailable.
- **Regression test**: verifier run leaves a running job untouched.

## Audit-wave fold-in (2026-10-16) — dispositions

Slice A (deployment, A.md):
- A-01 (P1, CONFIRMED) `.dockerignore` `*.env` failed to exclude `workbench/docker/.env` (later negations killed the recursive pattern); md5 of host file matched `/repo/workbench/docker/.env` in the image. FIXED: `**/.env` with example negations kept.
- A-02 (P1, CONFIRMED) `run.sh` used Python f-string syntax `${2!r}` → bash bad-substitution abort with wrong rc. FIXED: plain `${2}`; repro `WORKBENCH_BACKEND_PORT=80a0` now exits rc 2 with the documented message.
- A-07 (P1, EXTRA facet) J5 asserted `inspect.getsource` substrings — replaced by a behavior check (CUDA-declaring model must never reach TIER_GPU_READY unprobed). J8 passed on an empty job list — now `PENDING` ("no jobs observed") instead of PASS. J7 now also asserts the checkpoint-bundle validator refuses an empty bundle root (3-category enforcement).
- A-03 (P2) container ran as uid 0 and wrote root-owned files into host bind mounts (artifacts uid histogram {0: 25589}). FIXED: compose `user: "${WORKBENCH_UID:-1000}:${WORKBENCH_GID:-1000}"` + `.env.example` keys.
- A-04 (P2) doctor/launcher curls had no timeout (71 s `/api/models` → `000` false negative); pilot called the endpoint twice. FIXED: explicit `--max-time` on every curl; pilot resolves protocol+checkpoint in one fetch.
- A-05 (P2) queue was RAM-only; restart lost jobs and left `running` phantoms. FIXED: `queue_state.json` snapshots on every transition + `SingleGpuQueue.recover_interrupted()` at backend startup marks in-flight records `failed` with the restart reason (smoke: recovered record flips to failed).
- A-08 (P2) acceptance B's same-line regex missed `Promise.all` across lines/template literals (0/4). FIXED: balance-scan of every `Promise.all(...)` block (negative test: fabricated Promise.all over `/api/models`+`/api/runs` caught).
- Acceptance C asserted only model count. FIXED: served model ids/protocols/environments must equal the registry.
- A-09 (P2) mutation M3 never mutated anything (asserted the healthy behavior); `mutate()` used CWD-relative `.venv/bin/python`; the "startup restore" comment described code that did not exist. FIXED: M3 is a real mutant (compare_metric mutated to the old BUG-7 semantics; assert the gate verdict FLIPS — smoke: `KILLED: mutant flips the verdict`), `mutate()` uses `${REPO}/.venv/bin/python`, crashed-run backups are restored before the fresh run, and M1/M2 now run against a deterministic stub server instead of assuming a live stack (smoke: M1 mutant → `malformed JSON`, no prepare; M2 mutant → dress-only checkpoint).
- A-13 (P2) `--filter name=workbench` is a substring match that can catch the wrong container. FIXED: `--filter name=^cir-workbench$`.
- A-06 (P3) header claimed pilot passes `--verify-gpu` (it does not; the backend launch guard owns that). FIXED comment.
- A-11 (P3) launcher `.env` parsing diverged from compose (quotes) and `WORKBENCH_DATASET_ROOT` was shell-only. FIXED: quote stripping mirrors compose, CR stripped correctly (`$'\r'`), dataset root via `env_value`; per-model interpreter env vars documented in `.env.example`.
- Verified honest, no change: A-10 (first-boot `index_unavailable`), A-14 (GPU override fail-closed), A-15 (RO mounts kernel-enforced).

Slice B (runtime, B.md):
- Extra facet (P1): CSMCIR declared `cuda: UNKNOWN` although its evaluator hard-requires CUDA (`torch.cuda.empty_cache()` in `blip_validate.py` raises without a device). FIXED: registry declares `cuda: '11.7'` (the runtime bundled by the pinned PyPI torch 2.0.1 wheel); `_requires_gpu` now gates it.
- Q1/Q6 resolved by the P0 wave: `_environment_contract_digest` includes `runtime_adaptation`, so environment ids, fingerprints and GPU-capability keys all differ per profile.
- Q4 residual-1 already fixed (`--extra-index-url`, not a replacement).
- Q8 FIXED: adapted LIMN package floor pinned to the tested minor (`open_clip_torch>=2.24,<2.27`; YAML flow scalar quoted so the range stays one declaration, and the adaptation tests updated to the new constraint).
- G1 FIXED: schema-v2 `RunMetadata` gains optional `runtime_adaptation` (None = historical environment).
- G2 FIXED: planner group dict carries `runtime_adaptation`; readiness matrix exposes `environment_adapted` instead of letting `RUNTIME_VERIFIED` read as an exact-reproduction claim.
- G3 FIXED: aggregate report carries `runtime_adaptation` in the environment block, not only in `variant_provenance`.

Slice C (scientific, C.md):
- C-5 (P1) fixed: a reference that publishes no category table can no longer certify a full pass — the gate demotes to `PASS_WITH_UNCHECKED_CATEGORIES` (or `PASS_METRICS_ONLY_MAPPING_UNVERIFIED`) because the per-category check is unchecked for every category. All 28 committed VERIFIED rows publish full tables, so real data keeps full-pass semantics; fixture updated and new refusal semantics covered by `test_a_run_category_with_no_paper_counterpart_is_not_compared`.
- C-2 (P1) fixed: `evaluate_models.py` fed the gate `data_kind="experiment"` even for registry-declared synthetic sources, so a synthetic aggregate could mint `PASS_WITHIN_5_PERCENT`. Synthetic runs now declare `data_kind="synthetic"` and the gate refuses them as not-a-real-experiment.
- C-4 (P1) fixed: `/api/benchmark-runs` and `/api/runs/gate` served ANY parseable `*_aggregate.json`. Both now require evidence-ledger corroboration (entry exists for the run, `report_digest` equals the file's sha256, identity fields agree); uncorroborated reports are isolated with a stderr reason. Smoke: forged report refused; tampered report (post-ledger edit) refused; real ledger-backed report served. New API test `test_a_report_without_ledger_corroboration_is_not_served`.
- C-3 (P1) fixed: conesep reference rows named `fiq_n20/50/80` while the registry's canonical checkpoint ids are `fiq_n02/05/08` (0.2/0.5/0.8 fractions) → every conesep run BLOCKED_CHECKPOINT_MISMATCH. `papers.yaml` now references the registry ids; training-noise percentages (the actual condition match) unchanged.
- Withdrawn/verified: C-1 (caption paths resolve through the canonical dataset link — directory symlink), `_check_parity` ±0.05 claim (already migrated to the gate's relative rule), CSMCIR `observed["mean"]` (reference aggregation explicitly defines mean = (avg R@10 + avg R@50)/2, matching the parser).
- C-16/C-18 remain UNKNOWN (no data inspected; not falsifiable here).
- Single-category LIMN diagnostics can no longer be written into the official reports directory (`limn.py --output` refuses a parent named `reports`).

Slice D (API/frontend, D.md):
- D-2 (P1) fixed: the frontend refetches `/api/runs` when a job settles (`completed`/`failed`/`blocked`), so API-queued runs appear without a manual reindex; `POST /api/index/rebuild` stays as the operator fallback.
- D-3 (P2) fixed (terminal handling): `blocked` is treated as settled by the frontend. `completed_at` remains unset for `blocked` by design — the job never started, so a completed timestamp would lie.
- D-4 (P2) fixed: `POST /api/jobs` refuses an identical model+checkpoint+protocol job already queued/preparing/running (409 `duplicate_job`, names the existing job id). Finished jobs never block re-running.
- D-5 (P2) fixed: same mechanism as A-05 (persisted job state + startup recovery).
- D-6 (P3) fixed: `index_unavailable` now returns 503 (server state), other `WorkbenchError`s stay 400.
- D-10 (P3) fixed: a held workflow lock exits rc 3 with one `[BLOCKED]` line instead of a traceback.
- D-1/D-7 recorded as accepted gaps: D-1's mock-only surface is closed by the D-2/A-05 chain end-to-end (queue → evaluator → ledger → index rebuild), D-7 is UI polish deferred.

Verification: full non-e2e suite 923 passed (parallel-run lock contention caused 4 spurious failures that pass in isolation); targeted suites for gate/evaluator/API all green after the changes.

## Second review wave (A2/B2/C2/D2) — responses

Reports: `A2.md` (deployment), `B2.md` (runtime), `C2.md` (scientific), `D2.md` (API/frontend/jobs).

Slice A (A2.md, BLOCKED):
- A2-1 `.dockerignore **/.env` verified fixed with positive+negative build-context controls.
- A2 P1 compose `user: 1000:1000` bricked the deployment (vite-temp EACCES, micromamba root prefix, caches under `HOME=/`). FIXED in Dockerfile: `appuser` uid 1000 with a real `HOME=/home/appuser`, `MAMBA_ROOT_PREFIX=/artifacts/mamba`, `node_modules` chowned to 1000; compose sets `XDG_CONFIG_HOME=/artifacts/config`. Verified live: container as uid 1000 boots backend + frontend, serves 16 models, `config`/`node_modules`/`mamba` all writable.
- A2 P2 mutation-restore loop used a no-op path derivation. FIXED: backup names are underscore-encoded paths; restore decodes and `mkdir -p`s the parent. Smoke-tested on a fake backup tree.
- A2 P2 `--filter name=workbench` collision → `^cir-workbench$` (GPU doctor + J2).
- A2 P2 J1 hardcoded `len==16` → compares the served set to `models.yaml`; jobs assertion replaced with enum validation (long-lived stacks may legitimately have jobs).
- A2 P2 J4(b) grep matched any mention of bundles → greps the live selection expression (defeats the M2 mutant).
- A2 P2 `acceptance.sh` first-boot runs code 400 → 503 (matches `main.py`).
- A2 P2 recovery was file-rollback only: recovered records are now adopted into the queue (`JOBS.adopt_recovered`) and served via `/api/jobs`.

Slice B (B2.md, VARIANT):
- B2 confirmed `cuda: '11.7'` is evidence-backed (torch 2.0.1+cu117 wheel `version.py`), `_requires_gpu` gating works end-to-end, and the quoted YAML scalar parses as one declaration everywhere.
- B2-3 (P1) `_runtime_model` rebuilt packages by hand → duplicated unversioned entries, no version floor. FIXED: it now uses the canonical `runtime.resolved_runtime_environment` contract; `_required_packages` keys by canonical distribution name via `packaging.Requirement` (inline specs like `open_clip_torch>=2.24,<2.27` survive as version constraints; unusable constraints fall back to bare-name presence checks → `VERSION_CONSTRAINT_UNVERIFIED`, not `DEPENDENCY_MISSING`).
- B2-2/B2-4 (P2) `environment_adapted` always False → final group dict and readiness rows now carry `runtime_adaptation` (smoke: limn row reports `sm_120`).
- B2-1 recorded: `RunMetadata.runtime_adaptation` has a writer only in the adapted-report path; round-trip verified intact.

Slice C (C2.md, BLOCKED):
- F-01 (P1) evaluator child could read the ledger key via `HOME`. FIXED: `HOME` (and `XDG_CONFIG_HOME`) are never forwarded to the child; caches reach it through `TORCH_HOME`/`HF_HOME`/`TRANSFORMERS_CACHE`/`XDG_CACHE_HOME`. Reproduced: child now fails with `LedgerKeyError`.
- F-02 (P1) corroboration keyed by run_id → keyed by report digest; identity fields must match the file's own claims; duplicate files for one run_id serve one run. Reproduced: copied report not served twice.
- F-03 (P1) `return_code`/`invocation_status` unchecked → corroboration requires `return_code == 0` and `INVOCATION_COMPLETED`. Reproduced: failed-run ledger entry does not corroborate.
- F-04 (P1) container `HOME=/` made key creation fail → key lives under `/artifacts/config` (bind-mount persistent across rebuilds).
- F-05 (P2) average-only reference demoted full-coverage runs → demotion now keyed on the run's own coverage: complete per-category evidence passes; partial coverage still demotes. Both directions covered by tests.
- F-06 (P2) reference's own `checkpoint_mapping_status` now in `as_evidence` and caps the verdict: any recorded mapping claim other than `VERIFIED_METADATA`/`NO_CHECKPOINT` forces `PASS_METRICS_ONLY_MAPPING_UNVERIFIED` (conesep exact-match no longer certifies).
- F-07 (P2) limn reports-dir guard compares the resolved parent directory name.
- F-08 (P3) synthetic `NOT_RUN` reason renamed to "pipeline self-test".

Slice D (D2.md, BLOCKED):
- D2 P0 corroboration demanded a nonexistent `EXECUTION_OK` status → now `INVOCATION_COMPLETED` (the real producer value); fixture updated.
- D2 P1 `queued` jobs never persisted (log dir created only at dispatch) and queued-cancel not persisted → `_persist` creates the directory; `cancel` persists; worker `mkdir` relaxed to `exist_ok=True`. Tests cover enqueue-persistence, cancel-persistence, dispatch re-check still `blocked`.
- D2 P2 duplicate-job TOCTOU → the authoritative check is atomic inside `JOBS.enqueue` (`refuse_duplicate=True`, `DuplicateJobError` → 409); handler pre-check stays as fast path.
- D2 P2 stale `dist/` → frontend rebuilt from current `App.tsx`.
- D2 P2 tautological digest test replaced with the real P0-A contract: `_runtime_model` output has no duplicate package declarations and preserves the version floor.
- D2 P3 recovery root resolved at import → moved to the FastAPI startup hook.
- D2 P2(3) ledger key vs rebuild: compose points `XDG_CONFIG_HOME` at the artifacts bind mount, so the key survives image rebuilds.

Verification: full non-e2e suite 916 passed after the wave-2 fixes (single flake fixed and re-run green); `release_verify.sh --offline` verdict `GPU_VALIDATION_PENDING` with no unmet criteria; `mutation_checks.sh` all mutations detected; docker image rebuilt and boot-verified as uid 1000 (backend 200, frontend 200, ledger config writable).
