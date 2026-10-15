# Slice D — API / Frontend / Operations audit (HEAD e9cafaf)

Auditor: subagent D (read-only). Scope: `workbench/backend/main.py`, `workbench/backend/jobs.py`,
`workbench/backend/storage.py`, `workbench/backend/index.py`, `workbench/backend/analysis.py`,
`workbench/backend/paper_gate.py` (read-only), `workbench/frontend/**` (source + `gate.ts` + built
`dist/`), `workbench/scripts/pipeline.py` (state/lock), `workbench/scripts/serve_workbench.py`.
No GPU inference was run. Nothing in the repo was written except this file. All commands were
read-only; in-process reproductions used `.venv/bin/python` with `WORKBENCH_ARTIFACTS_ROOT=/tmp/...`
where the repo index would otherwise be touched (never written).

Every claim below is marked STATIC_AUDIT or RUNTIME_VERIFIED. "RUNTIME_VERIFIED" here means
*observed in-process* (queue/API/DuckDB), never a GPU or container run.

---

## D-1 (P0) The served UI can only ever show mock data: no official run is ever indexed

**Verdict: CONFIRMED (STATIC_AUDIT + RUNTIME_VERIFIED in-process).**

Evidence:

- The official evaluation path writes an aggregate report and nothing else:
  `workbench/scripts/evaluate_models.py:1453` hardcodes `"per_query_export_available": False` in the
  report it writes to `<artifacts>/reports/<model>_<ckpt>_<run_id>_aggregate.json`
  (`run_report_path`, `evaluate_models.py:1307-1308`, `reports_root` 1303-1304).
- No production code writes a schema-v2 `ResultRun`. `EvalRequest.output_path` exists
  (`workbench/backend/adapters/base.py:17`) but **no adapter ever passes it to its command**
  (grep `output_path` → only the dataclass field; the LIMN replay has its own `--output`,
  `workbench/replay/limn.py:212`, and the LIMN adapter never supplies it,
  `workbench/backend/adapters/models.py:129-139`). The only writer of `"schema_version": 2` in the
  whole repo is the mock generator: `workbench/tests/mock_data.py:26` →
  `workbench/scripts/load_mock_results.py` (`parser.add_argument("--output-root", ...)`, line 18).
- The index consumes *every* `*.json` under the results root except aggregate reports
  (`workbench/backend/index.py:39-50`) and validates it as `ResultRun` (index.py:53-63) whose schema
  requires `schema_version: Literal[2]` (`workbench/backend/schemas/results.py:90`). So the index is
  empty unless someone runs the mock stage or an export that does not exist yet.
- The doc says the same: `workbench/docs/RESULT_SCHEMA.md:54` ("Canonical per-query artifacts require
  a future reviewable observation-only export"), `workbench/docs/PC_REPRODUCTION_HANDOFF.md:20`.
- Observed, against the real artifact tree (no env override):
  `main.runs()` → 6 runs, all `data_kind: "mock"`; `main.run_gates()` → `{}`;
  `main.benchmark_runs()` → `[]`. `ls workbench/artifacts/reports/` → empty; the DuckDB index holds
  exactly the 6 mock rows (`select run_id,data_kind from runs`, venv duckdb).
- Consequence in the UI: Sample Explorer, Compare Models, Common Failures, Disagreement, Failure
  Analytics all read `/api/queries` + `/api/analysis`, i.e. DuckDB only → mock-only. The two gate
  labels (`CAPTION THAY THẾ`, `LỆCH PAPER x%`) only render for `benchmarkRuns`/`gates`, both of which
  are empty for any artifact tree with no aggregate reports, and the frontend comments the UI has "no
  error boundary" (main.py:149-152), so an empty gate set degrades to silence, not to a warning.
- Honesty mitigation that *does* exist: the global banner at `App.tsx:33`
  ("MOCK DEVELOPMENT DATA — no real evaluation results are loaded") and the per-run `MOCK` badge
  (`App.tsx:37`).

**Severity: P0** (research claim path: the UI is presented as the analysis surface, and every
analysis page is mock-only with no per-page statement of that fact).

**Recommended action:** pick one and make it explicit —
(a) implement the observation-only per-query export and thread `request.output_path` (or an
auto-derived path) into each adapter command so the evaluator/réplay writes schema-v2 JSON; or
(b) declare aggregate-only in the product: disable/hide the per-query pages (or render a blocking
notice inside each) when `runs` contains no `experiment` row, instead of a single global banner.

---

## D-2 (P1) The serving index is never updated by the API/UI; `POST /api/index/rebuild` has no caller

**Verdict: CONFIRMED (STATIC_AUDIT).**

Evidence:

- Rebuild happens only in the pipeline (`workbench/scripts/pipeline.py:1346-1358` validation-index
  stage, `pipeline.py:1412-1444` `run_validation_index`, `pipeline.py:1910` mock branch) and in the
  HTTP route `workbench/backend/main.py:287-289`. `App.tsx` contains **0** occurrences of `rebuild`
  / `index/rebuild`; the only other mention in the repo is a doc line
  (`workbench/docs/PC_DOCKER_QUICKSTART.md:90`).
- `/api/runs` (DuckDB) is fetched exactly once, at mount: `App.tsx:20-21` `useEffect(()=>{refresh();},[])`.
  What *is* polled is only the report filesystem (`App.tsx:22`, `:27` `loadReports` every 10 s) and
  `/api/jobs` every 2 s (`App.tsx:27`). So new evaluation *reports* appear automatically while new
  *per-query runs* never do.
- `POST /api/index/rebuild` is an unauthenticated state-mutating route: the FastAPI app installs only
  CORS and a `WorkbenchError` handler (`main.py:33-34`, `:57-59`); there is no `Depends`, no API key,
  no capability check anywhere in `main.py` (grep `Depends|api_key|token|capabilit` → only a comment).
  Implication accepted for a single-user laptop deployment (`WORKBENCH_HOST` default `127.0.0.1`,
  `operator_config.py:70`), but the shipped compose publishes to the host and the container binds
  `0.0.0.0` (`workbench/docker/Dockerfile:59`, `docker-compose.yml:18-20`), so on a shared host any
  reachable client can rebuild the index and queue jobs.

**Severity: P1** (staleness is a genuine user-journey defect; the missing auth is a deployment
property — if slice A does not own it, treat it as P1 there).

**Recommended action:** refresh `/api/runs` on the same settle/poll trigger as `loadReports`, and
surface a rebuild affordance (or rebuild automatically after a job completes) — the queue already
knows the run finished. Gate the mutating routes behind the same operator capability vocabulary the
pipeline uses (`allow_index_write`), or bind the server to `127.0.0.1` in the shipped compose.

---

## D-3 (P2) `blocked` is a terminal state the frontend never treats as terminal, and `completed_at` is unset for it

**Verdict: CONFIRMED (STATIC_AUDIT + RUNTIME_VERIFIED in-process).**

Evidence:

- `workbench/backend/jobs.py:263` sets `job.status = "blocked"` when the dispatch-time launch guard
  fails. `SingleGpuQueue._next_job` has already stamped `started_at` (`jobs.py:152-153`) but the
  `blocked` branch sets no `completed_at` (`jobs.py:258-264`), and `return_code` stays `None`.
- Reproduced in-process with a raising `launch_guard`: `status: blocked | return_code: None |
  completed_at set: False`, `stderr_tail: [BLOCKED] gpu capability invalidated`.
- Frontend: the settle effect only advances on `completed`/`failed`
  (`App.tsx:25-26`), so a `blocked` job never triggers `loadReports`; the row renders the raw status
  word and, because `job.error` is `None` for this path, the reason is only visible if the operator
  inspects `/api/jobs` JSON (`jobs.py:258-264` appends to `stderr_tail`, and `App.tsx` renders only
  `job.error`, not `stderr_tail` — grep: `stderr_tail` count in `App.tsx` is 0).

**Severity: P2** (status is visible, cause is not; a workaround exists — read the JSON).

**Recommended action:** in the `blocked` branch also set `job.error` (or render `stderr_tail` for
non-queued jobs) and include `blocked` in the terminal set that re-fetches reports.

---

## D-4 (P2) Duplicate identical jobs are neither refused nor deduplicated — the work is duplicated on the single GPU

**Verdict: CONFIRMED (RUNTIME_VERIFIED in-process).**

Evidence:

- `SingleGpuQueue.enqueue` always mints a fresh `uuid4` and appends (`jobs.py:100-105`); there is no
  key on (model, checkpoint, protocol) and no "already queued/running" check anywhere in `jobs.py`
  or `main.py` (grep `already|duplicate|dedupe` → only unrelated comments).
- Observed: two enqueues of the same model/checkpoint/protocol with a 0.6 s command produced two
  distinct job ids, both `queued`, both `completed`, elapsed 1.25 s (serialized), two distinct log
  directories.
- The operator-side guard cannot catch it: it refuses a *repeated run id* ("refusing to overwrite
  existing run manifest/report", `evaluate_models.py:1351`, `:519`), but the API generates a fresh
  `run_id` per request — timestamp + `uuid4().hex[:8]` (`main.py:341-343`) passed as `--run-id`
  (`evaluate_models.py:414`, `:1860-1864`), so no collision is possible and the second run silently
  re-does the whole evaluation.

**Severity: P2** (laptop/single-operator: costs GPU hours, does not corrupt evidence — artifacts are
run-scoped and immutable).

**Recommended action:** refuse (HTTP 409 with the existing job id) when a `queued|preparing|running`
job already matches model+checkpoint+protocol, or make the button idempotent in the UI.

---

## D-5 (P2) Job state is in-memory only; a backend restart mid-job loses the job and orphans the evaluator

**Verdict: CONFIRMED for the state loss (STATIC_AUDIT); CONFIRMED-WITHOUT-RUNTIME for the orphan
(PROBABLE — the container case was not exercised).**

Evidence:

- `jobs.py:90` `self._jobs: list[Job] = []` and `jobs.py:91` `self._processes`; nothing persists or
  reloads the list (`JOBS = SingleGpuQueue()` at import, `main.py:34`). The only on-disk trace of a
  job is `<artifacts>/queue_jobs/<job_id>/queue_command.json` written by the worker
  (`jobs.py:266-275`), which nothing reads back (grep `queue_jobs` → `main.py:351`, `jobs.py:266`
  only). After a restart `/api/jobs` returns `[]` and the UI shows no history; a job that was
  `running` simply disappears.
- Orphan path: the child is started with `start_new_session=True` (`jobs.py:293`) precisely so
  cancellation can `killpg` it. Nothing else reaps it. The container's PID 1 is
  `python workbench/scripts/serve_workbench.py` (`Dockerfile:70`), which only signals **its own direct
  children** in a `finally` (`serve_workbench.py:88-104`, used at `:167`/`:184`) — it never signals
  the queue's python process group parented to the *uvicorn worker* process, which is itself only a
  child of serve_workbench. So a `docker stop`/restart of the backend can leave the GPU-holding
  evaluator running while the queue is gone. STATIC_AUDIT: reading the process tree construction, not
  observed.
- Related: the compose service declares no `init: true` and no `--init`-equivalent
  (`docker-compose.yml:12-48`), so once the backend exits, its reparented children are never reaped
  (zombies) while the container stays up.
- Existing tests: `test_process_group_cancellation.py` (6 tests) and
  `test_phase1_acceptance_scenarios.py::test_scenario_d/g` cover *cancellation*, not backend restart;
  no test greps for restart/orphan in `workbench/tests` (only an unrelated hit in
  `test_release_blocker_evaluation_output.py`).

**Severity: P2** (the workbench is a single-operator tool and a restart is an operator action; the
GPU process leak is the part with real cost).

**Recommended action:** persist job rows (the `queue_command.json` file already exists — read it back
at startup and mark unknown-status jobs `interrupted`), and add `init: true` plus a container-level
`stop_grace_period` so a stopped container takes the whole process group down.

---

## D-6 (P3) Missing index returns HTTP 400, not 503

**Verdict: CONFIRMED (RUNTIME_VERIFIED in-process).**

Evidence: `workbench/backend/storage.py:13-17` raises
`WorkbenchError("index_unavailable", ...)`; the handler maps **every** `WorkbenchError` to status 400
(`main.py:57-59`). Reproduced: `storage.list_runs(database_path=Path('/tmp/nope.duckdb'))` →
`index_unavailable` payload; `main.runs()` through the app path yields the same error object.
Degradation elsewhere is honest: `/api/benchmark-runs` and `/api/runs/gate` read the report
filesystem, not DuckDB, and returned `[]` / `{}` with no index present (observed with
`WORKBENCH_ARTIFACTS_ROOT=/tmp/sliceD/artifacts`), and `/api/models`, `/api/jobs`, `/api/cohorts`,
`/api/annotations` all succeeded. So "startup without DuckDB" is **honest degradation**, not a 500 —
only the status code is wrong (`client.ts:6` reads `payload.message`, so the UI shows the right
sentence).

**Severity: P3.** **Action:** map `index_unavailable` to 503 (it is a server-state precondition, not
a bad request).

---

## D-7 (P3) `blocked`/`preparing` are the only job states the slider/config UI distinguishes; `INCOMPLETE_METRIC_EVIDENCE` is only visible after expanding

**Verdict: CONFIRMED (STATIC_AUDIT).**

Evidence: `GateDetail` (`App.tsx:49-66`) renders `verdict.status` verbatim, then
`blockers` (joined), then the comparison table; `INCOMPLETE_METRIC_EVIDENCE` produces comparisons
whose `relative_error_pct` is `None` (`paper_gate.py:193`, `:390-392`), so `reflag`/`warningLabels`
(gate.ts:43-46, 55-70) correctly never flag it — that part is right and matches the backend's stated
intent (`paper_gate.py:101-105`). The gap is presentation: the state is a collapsed `<details>`
summary with no badge, so a run with no numerical deviation and incomplete coverage looks like a run
with no findings until it is opened.

**Severity: P3.** **Action:** give non-PASS statuses a badge next to `GateDetail` (the status string
is already in the payload).

---

## D-8 Contract checks that PASS (recorded so they are not re-audited)

1. **Frontend ↔ backend field contract — PASS (RUNTIME_VERIFIED in-process).**
   `/api/models`: all fields the `Model`/checkpoint types read are emitted
   (`types.ts:4`; observed: `missing: []` for
   `checkpoint_id, automatic_download_available, checkpoint_downloaded, mapping_verified,
   adapter_command_ready, runtime_verified, runnable`; `method_name/native_protocol/
   literature_split_label/reproduction_status` all present; 16 models).
   `/api/jobs` = `dataclasses.asdict(Job)` (`jobs.py:107-109`) matching `types.ts:7` exactly.
   `/api/queries` items carry `top_results` (observed: `{rank, image_id, score}`).
2. **Warning labels / threshold wiring present in the built bundle — PASS (STATIC_AUDIT).**
   `gate.ts:52,58,66` defines `CAPTION THAY THẾ` and `LỆCH PAPER …% (metric)`; the slider/number input
   are `min=0 max=20 step=0.5` with `localStorage` key `cir-workbench.thresholdPct`
   (`gate.ts:3-12`, `App.tsx:34`). `dist/assets/index-C_KtNoc6.js` contains all three strings and
   `min:0,max:20,step:.5` twice.
   `reflag` arithmetic is byte-for-byte the backend's `deviations()` predicate
   (`gate.ts:43-46` `relative_error_pct > thresholdPct` vs `paper_gate.py:53-55`).
3. **Built bundle is current — PASS (STATIC_AUDIT, and a negative finding).**
   `workbench/frontend/App.tsx` is identical to committed `767ae3d`
   (`diff <(git show 767ae3d:App.tsx) App.tsx` → empty), and `dist` contains the 767ae3d-only strings
   (`preparing` inside the cancel condition, `` `runs: ${e.message}` ``). `dist/` is gitignored
   (`.gitignore:65`) and rebuilt at image build (`Dockerfile:11-16`, copied at `:39`), so no committed
   stale-artifact defect exists. (Marker check: `v!=null` is absent from dist because esbuild
   rewrote `v!=null` to `v?…`, not because the code is older.)
4. **Frontend is served in production, just not by the backend — PASS (STATIC_AUDIT).** No
   `StaticFiles`/`mount` in `main.py` or `serve_workbench.py`; `vite preview` proxies `/api` to
   `WORKBENCH_BACKEND_URL` (`vite.config.mts:4-10`, injected at `serve_workbench.py:150`, default
   `http://127.0.0.1:8000` from `Dockerfile:64`). Port drift is prevented by the compose env
   override (`docker-compose.yml:24-28`).
5. **Analysis endpoints do not fabricate — PASS (STATIC_AUDIT).** `/api/analysis` and
   `/api/analysis/failure-jaccard` fail closed on cross-protocol (`storage.py:150-152`), mixed
   `data_kind` (`:153-155`), non-identical query universes (`:166-175`), and insufficient top-k depth
   (`:156-161`). `/api/benchmark-runs` serves only well-formed reports
   (`main.py:129-175`, `:98-123`) and returns `[]` — never a synthesized row — when the reports
   directory is absent/empty (observed). Cohort export computes R@k/mean/median from stored target
   ranks only (`storage.py:192-204`); it does not extrapolate.
   Minor: two empty failure sets yield `jaccard: 1.0` (`analysis.py:82`) — documented convention, but
   "1.0" reads as perfect agreement; P3 at most.
6. **Result JSON → API → frontend full path — PASS for mock data (RUNTIME_VERIFIED in-process, no
   HTTP/GUI).** `workbench/artifacts/results/<protocol>/mock/*.json` (schema-v2, written by
   `load_mock_results.py`) → `rebuild_index` (`index.py:75-110`, atomic `os.replace`) → DuckDB
   (`runs/queries/top_results`) → `list_runs`/`query_page`/`validate_analysis_selection`/`_rows`
   (`storage.py`, `analysis.py`) → `main.runs()/queries()/analysis()/overlap()`. Observed: 6 runs;
   `queries(run_id=mock_airknow-fiq_n05, limit=2) → total 24`, item keys and `top_results` as typed;
   `analysis(run_ids="mock_airknow-fiq_n05,mock_habit-fiq_n05", k=10) → 24 rows` with all fields the
   frontend `Analysis` type reads; `overlap(...)` → diagonal 1.0, off-diagonal 0.8696. For **real**
   runs the same path terminates at D-1: there is nothing to index.

---

## D-9 Job lifecycle / cancellation — PASS (STATIC_AUDIT + RUNTIME_VERIFIED in-process)

**Verdict: VERIFIED for the paths that were exercised; no out-of-order transition found.**

Reachable transitions (all in `jobs.py`): `queued`(27) → `preparing`(154) → `running`(294) →
`completed`|`failed`(323, 332); `queued|preparing` → `cancelled`(125-126); `preparing` → `blocked`(263);
`running` → `cancelled`(134). Guards:

- Cancelling a `queued` job only mutates state (`jobs.py:124-128`) and the worker skips it before
  starting anything (`:245-247`); a test asserts this (`test_process_group_cancellation.py:155`).
- A cancellation arriving while the worker is inside the (lock-free) launch guard is re-checked under
  the lock immediately before `Popen` (`jobs.py:281`), so `preparing`+no-process cannot be escalated
  to `running`; the process handle, group id and status transition are all taken under the same lock
  (`:285-301`).
- Completion publication is suppressed for a cancelling job (`jobs.py:322-323`, `:331-333`), and
  cancellation removes the handle under the lock *before* signalling, so the worker cannot publish
  `completed` for a job being torn down (`jobs.py:130-135` docstring + code).
- Cancellation is real, not cosmetic: `_terminate_group` sends SIGTERM to the whole group for
  `GRACE_SECONDS = 10` (`jobs.py:184-186`), then SIGKILL, then reports survivors truthfully
  (`jobs.py:139-144`, `error = "cancellation could not stop every process in the job's group"`).
  **RUNTIME_VERIFIED in-process**: an operator that ignores SIGTERM and starts a SIGTERM-ignoring
  grandchild → `cancel` returned `cancelled, error: None`, and both the operator pid and the
  grandchild pid were dead 0.5 s later (`os.kill(pid,0)` → ProcessLookupError). A job that exits
  while `_cancelling` is not reported as `completed`.
- Group capture is taken at launch (`os.getpgid`, `jobs.py:298-301`) with the documented reason
  (the leader may exit first) — correct.

**Severity: NONE.** One residual note folded into D-3/D-5: `blocked`/`interrupted` are not rendered
as terminal, and nothing survives a restart.

---

## D-10 (P3) `pipeline_state.json` locking — correct, but a held lock surfaces as a traceback

**Verdict: CONFIRMED (STATIC_AUDIT).**

Evidence: `acquire_workflow_lock` uses `fcntl.flock(LOCK_EX|LOCK_NB)` with a live-owner timeout and
releases on process death (`pipeline.py:692-720`); `save_pipeline_state` writes
`pipeline_state.json.tmp.<pid>` then `replace` (`:748-754`); `run_stages` acquires at `:1974` and
always releases in `finally` (`:2237-2239`); `load_pipeline_state` swallows corrupt JSON to `{}`
(`:738-745`). Autopilot is a second caller with the same semantics
(`autopilot.py:525-531`, `:264-270`). The failure mode: `WorkflowLockError` is raised out of
`run_stages` and out of `pipeline.main()` with no catch anywhere in the repo (grep: only the
definition and the two raise sites), so two concurrent workflows produce a Python traceback and exit
1 instead of a `[BLOCKED]` line. No state corruption path was found (all state writes happen under
the lock).

**Severity: P3.** **Action:** catch `WorkflowLockError` in `pipeline.main()`/autopilot and print a
`[BLOCKED]` message with the resolved state path.

---

## D-11 Outline facet confirmed for P0-B (API-queued runs are exactly where it bites)

**Verdict: CONFIRMED (STATIC_AUDIT) — extends the known P0-B, do not duplicate it.**

- The queue forwards the full parent environment minus deny-prefixes (`jobs.py:57-83`), so a worker
  launched from the UI *does* carry `WORKBENCH_RUNTIME_ADAPTATION` to the operator.
- The HTTP route admits the job using the **adapted** readiness gate
  (`main.py:334` `runtime.require_launch_ready(model)`, re-checked at dispatch by
  `main.py:37-54`), while the operator's child allowlist drops the variable (the known P0-B), so the
  operator's own `guarded_plan`/report path (`evaluate_models.py:162` reads
  `variant_provenance.adapted_runtime`) can disagree with the replay child that
  `workbench/replay/limn.py:84` validates.
- Extra facet specific to this slice: the two endpoints that a UI operator uses to *start* a job and
  to *read its gate* (`POST /api/jobs`, `GET /api/benchmark-runs`) both resolve their model/view
  through `load_registry()` + `runtime.require_launch_ready` (adapted) and then display a
  `benchmark_gate` that was computed with the historical pins — so a UI-queued adapted run can be
  admitted and labelled `PASS_WITHIN_5_PERCENT_VARIANT` while the executing child never saw the
  adaptation. The single fix in `_evaluator_environment`'s allowlist closes both.

**Severity: P0** (owned by the main agent as P0-B).

---

## Execution trace (commands actually run, read-only)

```
git log --oneline -3                      # HEAD e9cafaf, worktree has main-agent edits (see below)
read workbench/backend/main.py, jobs.py, storage.py, index.py, analysis.py, paper_gate.py
grep -n 'output_path|schema_version|per_query' workbench/**  # → only aggregate writer + mock producer
.venv/bin/python -c "main.runs()/run_gates()/benchmark_runs()"        # → 6 mock runs, {} , []
.venv/bin/python - <<EOF  (SingleGpuQueue in-memory reproductions) EOF
   duplicate enqueue        → 2 distinct ids, both completed, 1.25s serial
   launch_guard raises      → status blocked, return_code None, completed_at None
   cancel running group     → operator + SIGTERM-ignoring grandchild both dead
.venv/bin/python -c "storage.list_runs('/tmp/nope.duckdb')"           # → index_unavailable payload
WORKBENCH_ARTIFACTS_ROOT=/tmp/sliceD/artifacts ... main.*()           # → honest [] / {} degradation
diff <(git show 767ae3d:workbench/frontend/App.tsx) workbench/frontend/App.tsx   # → empty
grep -c 'preparing|runs: ${e.message}' workbench/frontend/dist/assets/index-C_KtNoc6.js
```

Not run, by rule: no pytest, no docker, no GPU inference, no network. `git status` shows the main
agent's in-flight edits to `environment_planner.py`, `runtime.py`, `docker/run.sh`,
`registry/models.yaml`, `evaluate_models.py` — none of this report depends on those diffs, and none
of my line references are in those files except `evaluate_models.py`, where I cited only lines that
exist in HEAD (`git show HEAD:...)` semantics).

## What is UNKNOWN

- Container restart mid-job leaving an orphaned evaluator under `start_new_session=True`: PROBABLE
  from the process-tree reading, **UNKNOWN** as behaviour (needs a live container; not exercised).
- Zombie reaping in the container once the backend exits: **UNKNOWN** (no `init`, but the base image
  may supply one — not verified).
- Whether any *real* evaluation can ever produce schema-v2 JSON through some path I did not grep:
  every producer search (`schema_version`, `ResultRun(`, `per_query_export_available`,
  `output_path`) points to mock-only, and the docs agree, but a not-yet-written GPU-host script would
  not be visible in this tree — **UNKNOWN** in the strict sense, CONFIRMED-as-of-HEAD.

## Final verdict

**VARIANT.** The API/queue machinery in this slice is solid and honestly degrades: cancellation is a
real process-group kill (observed), the state machine has no out-of-order transition, the analysis
endpoints fail closed rather than fabricate, error bodies are structured, and the frontend↔backend
field contract and the two warning labels/threshold wiring check out against the built bundle.

But the *product* claim does not hold end-to-end: **D-1 (P0)** — no official evaluation result can
reach `/api/runs`, `/api/queries`, `/api/analysis`, `/api/runs/gate` or `/api/benchmark-runs`, so the
five per-query analysis pages and both gate labels can only ever render mock development data; that
is exactly the reverse of the failure-analysis workflow the workbench advertises. With that, plus
D-2 (index never refreshed by the API/UI), D-4 (silent duplicated GPU work) and D-5 (job state and
GPU process leak across a restart), the slice is not release-ready for a research claim, though it is
not dishonest about it in the places it does report (mock banner, `per_query_export_available: false`,
`PASS_WITH_UNCHECKED_CATEGORIES`).
