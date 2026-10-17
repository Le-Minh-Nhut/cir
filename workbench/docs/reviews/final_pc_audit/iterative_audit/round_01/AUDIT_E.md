# AUDIT_E — Round 01 (slice E+C frontend half)

Target: `workbench/backend/{main,jobs,index,storage,analysis}.py`, `workbench/frontend/` (TS sources).
Branch `research/cir-failure-workbench`; working tree WITH the uncommitted A–F fix wave. HEAD = `92ecbd94`.
Method: read the current source; drove the ASGI app, the queue and the ledger helpers directly in throwaway
`/tmp` scratch (`/tmp/api_probe`, `/tmp/evprobe`, plus inline probes). No docker, no downloads, no repo writes.
Targeted pytest only: `test_http_api_layer.py` 29 passed, `test_concurrency_ownership.py` 11 passed,
`test_process_group_cancellation.py`+`test_only_synthetic.py` 9 passed, `test_phase1_acceptance_scenarios.py` 8 passed,
`test_pc_launcher.py` 37 passed.
Every claim below is STATIC_AUDIT unless the row says otherwise. No GPU inference was run and none is claimed.

Legend: CONFIRMED / PROBABLE / FALSE_POSITIVE / EXPECTED_BEHAVIOR / EXTERNAL_DEPENDENCY_PENDING / KNOWN_LIMITATION; P0–P3.

---

## 1. Shutdown / dispatch race — the just-landed fix

### Q1. Does `job.status == "cancelled" or self._stopping` still let a legitimate queued job run when no shutdown is underway?
**VERIFIED (fix correct).**
- `jobs.py:541` `if job.status == "cancelled" or self._stopping: continue`, evaluated under `self._condition`.
- `jobs.py:286` sets `_stopping = True` under the same lock; `jobs.py:348-351` `_next_job` refuses to hand out a job once `_stopping`.
- A normal enqueue→dispatch with no shutdown was exercised in the same local pattern the new tests use
  (`test_concurrency_ownership.py:245-352`, 11/11 passed) and in my own probes; the predicate is only taken when a
  shutdown or a cancellation actually happened, so a legitimately queued job is not refused.
- Severity NONE. No action.

### Q2. Does a queued/preparing job settle honestly on shutdown?
**VERIFIED for the state transition; VARIANT on the label it chooses.**
- `jobs.py:297-302` settles every `queued`/`preparing` job as `cancelled` with `completed_at`, a `[STOP]` stderr line and a persisted snapshot.
- Probe (STATIC_AUDIT + local queue): `before: [('A','running'), ('B','queued')] → survivors: [] → after: [('A','cancelled'), ('B','cancelled')]`, both `queue_state.json` read back `cancelled`. No completion was published; a job already settled stays `completed` (`D completed / E running → D completed / E cancelled`).
- The precedented terminal status for a job stopped before dispatch is what `cancel` writes (`jobs.py:246-249`), so the choice is consistent. See F6 for the misclassification risk.

### Q3. Cancellation precedence
**VERIFIED.** `cancel()` writes `cancelled` for `queued`/`preparing` under the lock (`jobs.py:246-249`), records `_cancelling` for a running one (`jobs.py:257-258`), and the worker refuses to publish a completion while `job.status == "cancelled"` or the id is in `_cancelling` (`jobs.py:586`). The slow launch-guard window is covered by the same-lock decision at `jobs.py:541`. The blocked branch also refuses to overwrite a cancellation (`jobs.py:513-517`).

### Q4. Worker lifecycle
**VERIFIED.** `shutdown()` sets `_stopping`, `notify_all()` (`jobs.py:286-287`), `_next_job` returns `None`, `_work` returns; `shutdown()` joins the worker with a 10 s bound outside the lock (`jobs.py:314-316`). After shutdown `q._worker.is_alive()` was `False` in my probe.

### Q5. Persistence and restart recovery
**VERIFIED (STATIC_AUDIT).** `_persist` writes `queue_jobs/<job_id>/queue_state.json` (`jobs.py:443-464`); `recover_interrupted` globs exactly that path and rewrites in-flight records to `failed` (`jobs.py:466-491`); `main.py:96-100` runs it on startup and `adopt_recovered` re-surfaces each record (`jobs.py:322-343`). Probe: persisted state read back after dispatch and after shutdown matched the in-memory status.

### Q6. Duplicate jobs
**VERIFIED.** The authoritative refusal rides the queue lock (`jobs.py:214-223`), the handler's pre-check is documented as a fast path only (`main.py:790-798`); `test_http_api_layer.py::test_a_duplicate_submission_is_refused_atomically_by_the_queue` passes. See F13 for the bundle-vs-member identity gap (P3).

### Q7. Does the new API certification path ever refuse a legitimate corroborated run (FALSE FAIL)?
**CONFIRMED — yes, unconditionally, in the deployment. See F1 (P0).**

### Q8. Can the certification cache serve a stale verdict after a report or ledger change?
**VARIANT.** Key = `(report digest, ledger mtime:size, proof mtime:size)` (`main.py:439-443`). A report rewrite changes the digest (new key, no staleness). A ledger append moves mtime+size. What the key does NOT cover is a proof that is *removed* while the ledger is unchanged: the current request re-reads the report and derives `"absent"` → fresh key → fresh receipt, but a receipt already cached under the proof-present key is kept and re-served if the proof is restored byte-identically at the same size — and, worse, if the proof is deleted the degraded receipt is *also* cached, so a later restore of the identical proof re-uses the degraded (development) receipt. See F5 (P1). Severity of the plain mtime:size key: P3 (no reachable replay of a false `certified=true`).

---

## 2. API certification — false PASS and false FAIL

### Q9. Do `/api/benchmark-runs` and `/api/runs/gate` present an uncertified report as research evidence?
**VARIANT — they do not certify without evidence, but in the deployment they certify nothing at all (F1), while in a development process they certify without a proof (F4).**

### Q10. Does the new dimension set make any satisfiable dimension permanently PENDING?
**CONFIRMED — yes, three of them, and one by construction.** `MODEL_INFERENCE`/`MODEL_EVALUATION` have `"checks": []` (`release_verify.sh:523-524`), so `statuses` is empty, `statuses and all(...)` is False and the vacuous `all(s in ("PASS","PENDING"))` branch marks them `PENDING` (`release_verify.sh:527-536`). They can only move to `PASS` inside the block at `release_verify.sh:746-752`, which requires `_accepted_evidence`. With F1 making the API refuse every report and F2 making the ledger unreadable on this host, that set is empty on every run → `mandatory_pending` → verdict `GPU_VALIDATION_PENDING` permanently, and `SCIENTIFIC_OUTPUT` (a `PASS` from J7 before this fix) is now downgraded to `PENDING` whenever no accepted evidence exists (`release_verify.sh:771`). Severity P1. Action: fix F1/F2 first; then decide whether J7-only `SCIENTIFIC_OUTPUT` should have a floor.

---

## 3. Findings

### F1 — CONFIRMED, P0 — deployment refuses every report: `/api/benchmark-runs` and `/api/runs/gate` are unconditionally uncertified
- Location: `main.py:208-236` (`_development_receipt_refusal`), short-circuited first in `main.py:348-350`; the authoritative verifier path `main.py:352-366` is therefore unreachable in production.
- Repro (STATIC_AUDIT, executed): with `PYTEST_CURRENT_TEST`, `WORKBENCH_DEVELOPMENT_MODE` and `WORKBENCH_ALLOW_DEVELOPMENT_CERTIFICATES` unset, and a report with its completion proof present:
  `proof stamp = 1792169590940456029:56` → `_certify_report` returned
  `{'verified': False, 'is_research_evidence': False, 'reasons': [], 'evidence_reasons': ['this workbench process is not the evaluator deployment ...'], 'certified': False, 'development_mode': False}` and the authoritative verifier was called **0** times.
- Expected vs actual: the docstring says a deployment serving reports it produced itself validates the proof here, lazily (`main.py:216-219`). Actual: the branch is pre-empted in every non-pytest, non-`WORKBENCH_DEVELOPMENT_MODE` process, and neither flag is set by `Dockerfile`/`docker-compose.yml`/`docker-compose.gpu.yml`/`.env.example` — `grep -rn` over `workbench/docker workbench/config pc.sh` finds no occurrence. So the container, which serves exactly `Dockerfile`-built reports, reports `certified=false`.
- Blast radius: `_gate_fields` forwards the receipt (`main.py:501-506`) into `/api/runs/gate` and `/api/benchmark-runs`; `release_verify.sh:737` uses `is_research_evidence` for J7-adjacent dimensions; `pc.sh:1118` and `pc.sh:1313` refuse to reuse or summarise a run without it.
- Corrective action: delete the blanket refusal. Gate only the *path that cannot validate a proof* (the `_proof_stamp == "absent"` branch, i.e. `main.py:367`), and keep the pytest flag as a test-only relaxation inside that branch instead of a global precondition.

### F2 — CONFIRMED on this host, P0 — the evidence ledger cannot be authenticated, so no report is served at all
- Location: `evaluate_models.py:1143-1163` (key resolution), `main.py:396` + `400` (`read_evidence_ledger`, `ledger_by_digest`).
- Repro (command + observed output):
  - `cmp .workbench_ledger_key ~/.config/cir/workbench_ledger_key` → `differ: char 1, line 1`; mtimes `2026-10-12 09:59` vs `2026-10-13 12:18`.
  - `.venv/bin/python -c "from workbench.scripts.evaluate_models import ledger_hmac_key, LedgerKeyError; ..."` → `LedgerKeyError: two different ledger keys exist (...); keep one and remove the other, or set WORKBENCH_LEDGER_KEY explicitly to choose`; `read_evidence_ledger(config) -> None`.
  - Neither key verifies the existing `workbench/artifacts/evidence_ledger.jsonl` (HMAC recomputed over each line with both keys → `digest mismatch`; total 226 bytes).
- Root cause: the same repo has accumulated both candidate key files (`evaluate_models.py:1155-1176`), and the container cannot see either: compose pins `XDG_CONFIG_HOME=/artifacts/config` (`docker-compose.yml:41`) while the ledger lives at `/artifacts/evidence_ledger.jsonl` (`evaluate_models.py:1108-1113`); `workbench/artifacts/config/cir` does not exist on this host.
- Expected vs actual: with `ledger is None`, `_aggregate_reports` skips every report ("no evidence-ledger entry for its run", `main.py:418-420`) — the API silently serves `[]` while `/api/runs` still lists the indexed run, so the two views disagree about the same run. Classification: EXTERNAL_DEPENDENCY_PENDING as to the operator's key hygiene, P0 as to the observable behaviour (the deployable stack cannot serve its own evidence).
- Corrective action: fail loudly — surface `LedgerKeyError` on `/api/benchmark-runs` (a 503 with the two paths named) instead of an empty list; and stop emitting two key locations in a fresh clone (keep the config-home key only).

### F3 — CONFIRMED, P1 — the documented LIMN selection is permanently unlaunchable through the API
- Location: `main.py:684-693` (bundle selection folds a declared bundle onto its *anchor* member variant), `evaluate_models.py:328` (bundle-incomplete blockers stripped only when `checkpoint_id != bundle_id`), `runtime.py:2451-2452` (re-adds them), `main.py:719-721` (`guarded_plan` → 409 `evaluation_blocked`, no re-enqueue path), `pc.sh:1043-1050` (the launcher prefers `bundles[0].bundle_id`).
- Repro (STATIC_AUDIT): `checkpoint_availability` for the folded bundle selection (`checkpoint_id=base_iter0_all_categories`, registry filename of the anchor `0_dress_best_model.pt`) → `runtime_blockers(...)` returned `['execution environment not verified: ...']` on this host, and in the real "dress installed, shirt/toptee not yet" state the anchor row is runnable while `checkpoint_bundle_missing_paths` (`registry.py:444-458`) yields `checkpoint bundle incomplete: base_iter0_all_categories: <path>` — the strip condition `checkpoint_id != checkpoint.get("bundle_id")` is False whenever the caller already sent the bundle id, and `runtime.py:2451-2452` re-adds the blocker unconditionally a few lines later.
- Expected vs actual: the workbench's own preferred first task (the complete three-category macro) is refused with `evaluation_blocked`, while the operator sees READY. Classification CONFIRMED (false FAIL), severity P1.
- Corrective action: when the selection is already the bundle id, drop the bundle-incomplete blocker exactly as the member-selection path does (or mark readiness as per-member after acquisition).

### F4 — CONFIRMED, P1 — a report with NO completion proof is certified through the aggregate API
- Location: `main.py:352` (proof branch taken only when `_proof_stamp != "absent"`), `main.py:367` → `_development_certification` (`main.py:304-330`).
- Repro (STATIC_AUDIT, executed): with `WORKBENCH_ALLOW_DEVELOPMENT_CERTIFICATES=1`, a report with **no** `*_completion.json` and a valid LIMN category/macro shape → `development_mode: False, certified: True, is_research_evidence: True` and the verifier was *not* called (called 1 time only when a proof file existed). Note the receipt labels itself `development_mode: False` even on the relaxed path it is named for.
- Impact: the opt-out is per-process, not per-report: J7-style digging aside, any report in `/artifacts/reports` that the ledger corroborates is served as certified research evidence provided it is not synthetic — without the proof `pc.sh:1118`, `verify_run.py` and the queue all require.
- Secondary: `_proof_stamp` (`main.py:254-262`) re-derives the proof filename instead of calling `completion_proof_path` (`evaluate_models.py:1364-1365`), so any naming drift between the two silently downgrades a real proof run to the development path.
- Corrective action: make certification conditional on the proof, not on the environment, and have `_proof_stamp` call `em.completion_proof_path`.

### F5 — CONFIRMED, P1 — the receipt cache can re-serve a development receipt for a proof-backed report
- Location: `main.py:439-443` (`cache_key`), `main.py:246` (process-global `_CERTIFICATION_CACHE`, cleared nowhere — the only clears are in tests).
- Repro (STATIC_AUDIT, by construction): request while the proof is absent → key `(d, ledger, "absent")` → `_development_certification`, certifying at ledger level. Delete/restore the proof so that `(d, ledger, stamp)` recurs → the development-mode receipt is returned for a report that now has an authoritative proof.
- Root cause: two different certification strengths are keyed by inputs whose degradation is invisible to the key; the report itself is never an input (reports are read every request and re-attach the receipt, so a *same-bytes* rewrite also re-uses it).
- Corrective action: include the certification channel (`"proof"`/`"ledger"`) in the key, or refuse to cache the development receipt.

### F6 — CONFIRMED, P1 — shutdown rewrites jobs it never owned, and does not mark them cancellable-but-queued
- Location: `jobs.py:297-302`. Every `queued`/`preparing` job becomes `cancelled`, monotonically and irreversibly.
- Repro (STATIC_AUDIT, executed): `before: [('A','running'), ('B','queued')]` → `shutdown()` → both `cancelled`, both persisted `cancelled`, `error = None`.
- Contradiction with the rest of the tree: `pc.sh:781-789` and `pc.sh:843-860` treat `queued`/`preparing`/`running` as *active work* that must never be torn down, while the queue turns a merely deferred queued job into a terminal `cancelled`. `cancel()` then raises `ValueError("cannot cancel cancelled job")` (`jobs.py:251-252`), and `queue_state.json` has no field that records *which* disappearance produced the status (contrast `recover_interrupted`, which writes `error = "workbench restarted while the job was in flight"`).
- Practical effect on the target PC: `docker restart cir-workbench` (or `run.sh up` replacing a stuck container — `pc.sh:903-914`) with N jobs queued silently discards N−1 evaluations and reports all N as `cancelled`, without an operator-visible reason beyond the `[STOP]` stderr line.
- Corrective action: settle queued/preparing jobs as `failed` with the shutdown reason (or keep `queued` and let the next boot's `recover_interrupted` handle them); either way record a reason. Severity P1.
- Related (FALSE_POSITIVE): `blocked` is *not* in the `{"queued","preparing"}` set, so a job parked in `blocked` and never re-enqueued stays non-terminal forever. That is a real gap but it is the pre-existing "blocked needs operator action" design, not a shutdown regression — recorded, not counted.

### F7 — CONFIRMED, P1 — an enqueue after shutdown is accepted and parks forever
- Location: `jobs.py:212-231` (`enqueue` never consults `_stopping`), `jobs.py:345-351` (`_next_job` returns `None` once stopping).
- Repro (STATIC_AUDIT, executed): `shutdown() -> []`; `enqueue(...) -> status queued`; after 1 s `in-memory = queued | persisted = queued | worker alive = False`; a second `shutdown()` finally flips it to `cancelled`.
- Impact: the record is left `queued` on disk for the next boot, which `recover_interrupted` (`jobs.py:483`) turns into `failed("workbench restarted while the job was in flight")` and `pc.sh`'s persisted-active probe (`pc.sh:795-828`) reads as permission to block a restart. Reachable in-process for any ASGI app whose lifespan re-runs `_startup`/`_shutdown` without a fresh import (`main.py:95-118`).
- Corrective action: refuse the enqueue once `_stopping` (409/`cancelled`), one line.

### F8 — PROBABLE, P1 — `release_verify.sh` can report MODEL_INFERENCE PASS from an already-refuted run
- Location: `release_verify.sh:731` `_has_proof = not any("completion proof" in r for r in _reasons)` (a substring proxy), `:743-744` `if not _is_synthetic and _rep.get("data_kind") == "experiment" and _has_proof: _inference_evidence.append(...)`, `:746-748` diminishing to `MODEL_INFERENCE = PASS`.
- Reachability: needs a report carrying `data_kind == "experiment"`. `grep -n "data_kind" workbench/scripts/evaluate_models.py` returns only the gate argument (`:185`), so evaluator-written reports never carry the field and the guard currently blocks this. A fixture, a migrated report, or any future schema carrying the field reaches it: the run would be listed under "real model execution observed" although `verify` refused it.
- Corrective action: use the verifier's own answer for the execution question — `verdict["execution"]` / `verdict["verified"]` — instead of a substring over English reasons.

### F9 — CONFIRMED, P2 — two guard paths in `pc.sh` take different decisions on the same host state
- Location: `pc.sh:1081-1128` (reuse branch: "Phát hiện kết quả đánh giá đã hoàn thành và được sổ cái chứng thực") tests `is_research_evidence` per candidate run, then falls through to `pc.sh:1161-1165` (`POST /api/jobs`), which queues a fresh evaluation. The reuse gate reads `candidate_runs` from `/api/runs` + `/api/benchmark-runs` only (`pc.sh:1077-1111`), i.e. precisely the two views F2 desynchronises: with `read_evidence_ledger -> None` an aggregate-only run disappears from `benchmark-runs`, the candidate set is empty or unverifiable, and the launcher re-runs the whole evaluation instead of stating "a run exists and its evidence could not be authenticated".
- Severity: P2. Fail-closed in the sense that no conclusion is drawn from unauthenticated evidence, but the operator gets a silent full re-run of an expensive evaluation instead of the real reason. Corrective action: distinguish "no run" from "run present, evidence unauthenticated" and fail loudly in the second case.
- Corrective action: route both through the same "is this run research evidence" helper.

### F10 — CONFIRMED, P2 — the receipt is honest in the API and invisible everywhere else
- STATIC_AUDIT: `grep -rn "is_research_evidence\|certified" --include=*.ts --include=*.tsx workbench/frontend` → only `gate.ts:20-22` type declarations; no consumer. `App.tsx:36-37` renders any served report by its `benchmark_gate`, and `App.tsx:58` appends "(not a certified pass)" only when `verdict.passed === false`.
- Consequence: a served report with `benchmark_gate.passed = true` and `verification.certified = false` renders as `PASS_WITHIN_5_PERCENT` with no caveat, and the refused evidence verdict never reaches the operator. `pc.sh` likewise never reads `verification`.
- Corrective action: render `verification.certified === false` as a refusal badge in `GateDetail` (one condition) — the receipt is already in the payload.

### F11 — CONFIRMED, P3 — report/mock assets cannot be told apart in the served payload
- Location: `main.py:528-534` (`/api/benchmark-runs` does not project `data_kind`; real aggregate reports do not carry it — see F8's grep), `App.tsx:37` (`{b.is_synthetic && "SYNTHETIC SOURCE"}`, which is the only provenance badge on an aggregate-only row).
- Effect: a report written by a mock/development loader is served with `is_synthetic=false` and no distinguishing field, unlike a per-query `Run` which carries `data_kind`. P3, no false certification (the macro/extraction contract still applies).

### F12 — CONFIRMED, P3 — dead weight in the audited files
- `index.py:30-31` freezes `DATABASE_PATH`/`RESULTS_ROOT` at import; only tests read them (`test_artifact_root_configuration.py:87`), while every live call site uses the functions.
- `analysis.py:research_interest` (3 references, all its own definition/docstring), `analysis.py:get_query_universe` and `storage.py:get_query_universe` (0 external callers) are unused.
- Corrective action: delete. No behaviour is load-bearing on them.

### F13 — PROBABLE, P3 — bundle-vs-member selections are distinct duplicate keys
- Location: `jobs.py:214-223` compares `(model_id, checkpoint_id, protocol_id)`; `main.py:684-693` maps a bundle selection to `checkpoint_id = bundle_id` while a member selection keeps the member id.
- Repro (STATIC_AUDIT, executed): with an anchor job already `running`, `enqueue("limn","m1",...)` against a queued `("limn","bundle_all",...)` was **ACCEPTED** (`no duplicate detected`).
- Impact: the same weights can be queued twice (bundle path and member path), duplicating expensive GPU work. The single-slot queue serialises them, so this is not GPU double-booking; severity P3.

---

## 4. False-PASS / false-FAIL summary

| direction | mechanism | file | verdict |
|---|---|---|---|
| FALSE FAIL | every report uncertified in any non-pytest process | `main.py:208`, `:348` | CONFIRMED P0 (F1) |
| FALSE FAIL | unreadable ledger → empty `/api/benchmark-runs`, launcher refuses a completed run | `main.py:396`, `evaluate_models.py:1143` | CONFIRMED P0 on this host (F2) |
| FALSE FAIL | documented LIMN bundle can never launch with ≥1 member present | `main.py:684`, `evaluate_models.py:328`, `runtime.py:2451` | CONFIRMED P1 (F3) |
| FALSE FAIL | shutdown irreversibly cancels deferred queued jobs | `jobs.py:297` | CONFIRMED P1 (F6) |
| FALSE PASS | report with no proof certified at ledger level | `main.py:352`, `:367` | CONFIRMED P1 (F4) |
| FALSE PASS | API parity verdict rendered without the refused-evidence verdict | `App.tsx:58` | CONFIRMED P2 (F10) |
| FALSE PASS | `MODEL_INFERENCE=PASS` from a substring proxy over refutation reasons | `release_verify.sh:731,743` | PROBABLE P2 (F8) |
| not reachable | duplicate GPU job via concurrent POST | `jobs.py:214` | FALSE_POSITIVE (atomic refusal holds) |
| not reachable | stale `certified=true` from a rewritten ledger | `main.py:264-277` | FALSE_POSITIVE (HMAC chain breaks; new entry moves mtime+size) |

Prior-audit hypotheses re-checked against the current tree (not paraphrased):
- "Fresh clone startup 503 / no index" — **FALSE_POSITIVE now**: `main.py:62-88` `_ensure_index()` builds asynchronously when the database is absent and `_startup` (`main.py:96-100`) calls it; `_check_index_ready` (`main.py:180-187`) returns 503 only while that build is genuinely in flight.
- "Shutdown orphan during `preparing`" — **FALSE_POSITIVE now**: the same-lock `_stopping` decision (F Q1) plus 11 passing race tests; my probe saw `_processes == {}` and no dispatch.
- "Cohort markdown exports crash on `None`" — **not reproduced**: the table cells are built through `_format_metric` (`main.py:619-623`), which maps `None`/NaN to `"N/A"`; `storage.cohort_metrics` (`storage.py:192-204`) yields `None` only for empty ranges and needs no change.
- "UI cannot queue the LIMN bundle" — **FALSE_POSITIVE**: `App.tsx:36` renders `checkpoint_bundles` with per-member readiness and posts `bundle_id`; `types.ts:1-3` declares the shape.
- "Job completion never rebuilds the DuckDB index" — **EXPECTED_BEHAVIOR**: rebuild is an explicit operator step (`pc.sh:1277-1278`, `main.py:655-664`) and `App.tsx:24-26` documents that a settled job refreshes reports, not the run index.

---

## 5. Frontend checklist (TS read, not compiled)

| question | verdict | evidence |
|---|---|---|
| bundle selection payload | VERIFIED | `App.tsx:36` sends `b.bundle_id`, disables unless every member `runnable` and `native_protocol` present; members/blockers rendered |
| blocked/failed terminal handling | VERIFIED | `App.tsx:37` renders any status string verbatim, `error` and `stderr_tail` in `<pre>`, cancel offered only for `queued`/`preparing`/`running` |
| result refresh | VERIFIED | `App.tsx:26` refreshes reports+runs once per settled job id (`useRef` set), `App.tsx:22`/`:27` initial load and 10 s poll |
| error isolation | VERIFIED | each loader catches its own failure (`App.tsx:21`, `:26-27`, `:29`); `_gate_fields` (`main.py:478-517`) normalizes every shape the components dereference, which is the app's substitute for an error boundary (`App.tsx` has none — `components.tsx` is 10 lines) |
| receipt surfaced | **NOT VERIFIED / defect** | F10: no component reads `verification`; `gate.ts:20-22` declares only `observed_metrics`/`execution_status`/`is_synthetic` |

---

## 6. Evidence not obtained

- No GPU inference, no docker build, no container; `release_verify.sh` and `pc.sh` were read, not executed against a live stack. Every `release_verify.sh`/`pc.sh` statement above is STATIC_AUDIT.
- The two `conflicting ledger keys` state was observed on this audit host; whether the target PC starts from the same state is UNKNOWN.
- `workbench/backend/autopilot.py`, `environment_planner.py`, `registry.py`, `paper_gate.py`, `benchmark_references.py`, `metrics_extraction.py`, `preflight.py`, the `scripts/` tree and the `dist/` bundle were opened only where a cited line required it, and were not audited as such.

## 7. Can this fail on the real first run on an RTX 5070 Ti PC?

**Yes — before any GPU question is reached.** Three independent first-run blockers, in order:
1. P0 F2: the ledger key state on this host makes `read_evidence_ledger` raise `LedgerKeyError`, so `/api/benchmark-runs` silently returns `[]` and `pc.sh`'s reuse gate can no longer corroborate any completed run — it falls through to queueing a fresh evaluation of an already-completed run. The container additionally cannot see either key file (compose pins `XDG_CONFIG_HOME=/artifacts/config`, the ledger sits at `/artifacts/evidence_ledger.jsonl`; `workbench/artifacts/config/cir` does not exist here).
2. P0 F1: even with a healthy ledger, every report the network serves is `certified=false` — the certification branch that would validate a real proof is unreachable outside a development process, and the release verifier reads exactly that field.
3. P1 F3: the launcher's own preferred LIMN selection (the three-category bundle) is refused with `evaluation_blocked` as soon as the anchor member is installed and a sibling is not.

Beyond those, the shutdown/queue chain itself now behaves: no orphan is spawned after shutdown, cancellation wins, the worker exits, and a settled job keeps its status (probes above). The unaudited remainder is the GPU/evaluator half (slice B/C), not this slice.
