# REV6 — Job settlement: can a job still reach `completed` without a validated proof?

Adversarial, read-only review of the release-fix wave at baseline `b45c068`.
Scope: `workbench/backend/jobs.py`, `workbench/scripts/evaluate_models.py`,
`workbench/backend/main.py`, `workbench/tests/test_release_blocker_invocation_proof.py`.
All probes are throwaway scripts under `/tmp/rev6/`; nothing in the repository was modified.

**Verdict: UNSOUND.** The intended guarantee holds for the *operator-mediated* attack (no verdict
⇒ no completion) and for the silent/failed/`PROOF_REJECTED` cases, but the completion decision is
made by reading **one unauthenticated, self-declared string** out of a directory the untrusted
evaluator child also writes. Three reproductions below settle a job `completed` with no
corroboration at all, including one where the operator crashed and the HMAC ledger was never
written, and one where the queue contradicts its own `return_code` evidence. `verify_run.py`
already rejects exactly these; `jobs.py` does not call it.

Reproduction environment: `.venv/bin/python` (3.13.12) — the repository's only interpreter with
`pytest`/`yaml`. Probes reuse the fixtures of the reviewed suite.

## 1. What was verified as correct

Baseline test run of the reviewed suite (all 14 pass):

```
$ .venv/bin/python -m pytest workbench/tests/test_release_blocker_invocation_proof.py -q
14 passed in 2.31s
$ .venv/bin/python -m pytest workbench/tests/test_process_group_cancellation.py \
      workbench/tests/test_concurrency_ownership.py -q
12 passed in 63.60s
```

| Attribute | Evidence | Result |
|---|---|---|
| No run identity ⇒ no verdict ⇒ not `completed` | probe A (`jobs.py:143-144`) | `failed`, `return_code=0` preserved |
| Verdict naming another run id | probe B1 (`jobs.py:120`) | `failed` |
| No verdict file at all | probe A, reviewed test `test_a_dispatch_with_no_recorded_verdict_never_completes` | `failed` |
| Verdict `NO_OBSERVABLE_OUTPUT` / `PROOF_REJECTED` / `EXECUTION_FAILED` | probe I, reviewed tests (b)(d) | `failed` with the operator's own reason |
| Genuine validated run | probe D, E1 | `completed` |
| Verdict written *during* the child's run is seen (read happens after `wait()`) | probe D (`jobs.py:520-525`) | `completed` |
| Cancelled job stays cancelled | `test_a_cancelled_job_never_reports_completion`, `test_cancellation_while_queued_starts_nothing` | `cancelled` |
| Signal-killed child | probe C1, E2 | `failed`, native `rc=-9` preserved in all four artifacts |
| Backend-reported run directory == operator-computed run directory | probe F (`main.py:460-461` vs `evaluate_models.py:1021-1031`) | `AGREE: True` |
| Stale run directory is refused, not reused | probe G (`evaluate_models.py:518-528`) | `RuntimeError`, no verdict written |
| Reason + artifact path remain inspectable | probe I | `error`, `stderr_tail`, `queue_state.json`, `queue_command.json`, verdict file all readable |
| Native exit code survives everywhere | probe E1/E2 | `command.json`, report, ledger entry, verdict all carry the same code |

## 2. Findings

### P0 — `completed` is decided by an unauthenticated string the evaluated child can write
`workbench/backend/jobs.py:102-123`, `:138-139`, `:525`

`invocation_verdict()` reads `<run_directory>/invocation_outcome.json`, checks only
`payload.get("run_id") == run_id` (`jobs.py:120`) and then `settled_status()` returns
`("completed", None)` on the bare string `verdict == "PROOF_VALIDATED"` (`jobs.py:138-139`).
Nothing is checked against the HMAC ledger, the report, the completion proof or the exit code.

**Reproduction (probe K, `/tmp/rev6/probe_k_child_forges.py`).** A rogue evaluator — the
*untrusted* process the operator starts via `_evaluator_environment` — writes a
`PROOF_VALIDATED` verdict into its own run directory and an aggregate report into the reports
root, then makes the operator's own report write fail. The audited operator therefore never
writes a verdict and **never appends a ledger entry**. Dispatched through the real
`SingleGpuQueue` with the real `evaluate_models.py` operator:

```
RESULT job status: completed | return_code: 1 | error: None
child's forged verdict file still present: True
operator stderr tail: [BLOCKED] refusing to overwrite existing run report: .../limn_ckpt_v1_child-forged-run_aggregate.json
ledger entries for run: []
```

The job is published `completed` while (a) the child exited 1, (b) the operator reported
`[BLOCKED]` on stderr, (c) no evidence-ledger entry exists for the run, and (d) the report it
"completed" is `{}` written by the model code.

**Same defect without any hostile intent (probe H, `/tmp/rev6/probe_h_stale_verdict.py`).** Run
directories are claimed atomically (`evaluate_models.py:518-528`), but a *fresh* run directory
left by an earlier invocation of the same run id already contains that invocation's verdict.
When the new child fails immediately, the operator is not even reached and the stale verdict is
read as this dispatch's:

```
H child failed rc=1, stale PROOF_VALIDATED present -> completed | return_code: 1 | error: None
```

Note the self-contradiction: `status=completed` with `return_code=1`.

**Same defect against authenticated evidence (probe J,
`/tmp/rev6/probe_j_ledger_contradiction.py`).** With a genuine HMAC-signed ledger entry for the
run recording `return_code=137, invocation_status=INVOCATION_FAILED`:

```
ledger witness: return_code= 137 invocation_status= INVOCATION_FAILED
RESULT: job status = completed | error = None
```

The queue publishes success while the out-of-band witness — the one mechanism explicitly
described in `jobs.py:37-45` as the reason a job needs `WORKBENCH_LEDGER_KEY` — says the run
failed. `main.py:150-210` refuses to serve the corresponding report for exactly this reason
("disagrees with its evidence-ledger entry"); the job status does not.

`workbench/scripts/verify_run.py` already answers precisely this question, in the same tree,
using the evaluator's own ledger reader and proof validator (its docstring, `verify_run.py:1-16`),
and `jobs.py` does not use it. `test_release_blocker_invocation_proof.py` pins the positive and
negative *verdict* cases but never pins that the queue cross-checks the ledger, so the suite
cannot fail on this.

### P1 — `test_release_blocker_invocation_proof.py` writes the queue's premise by hand, so the fix's core edge is untested
`workbench/tests/test_release_blocker_invocation_proof.py:164-187`

`queue_for()` (`:171-187`) dispatches a **no-op** child (`[sys.executable, "-c", "pass"]`) whose provenance
already points at the verdict file a *previous* `execute()` call wrote:

```python
job = queue.enqueue(MODEL, CHECKPOINT, PROTOCOL, [sys.executable, "-c", "pass"], ...,
                    {"run_id": run_id, "run_directory": str(em.log_dir(...))})
```

No test dispatches a real operator through the queue and asserts that the verdict the queue reads
is the one *that* invocation produced. That is the assumption the P0 reproductions break
(probe H is the same shape, minus the earlier `run_operator()` call, with the opposite outcome).
`test_a_dispatch_with_no_recorded_verdict_never_completes` (`:189-209`) is the same hand-written
shape. The suite is therefore evidence that the *branch* works, not that the *contract* holds.

### P2 — a self-declared `PROOF_VALIDATED` outranks the queue's own exit-code evidence
`workbench/backend/jobs.py:138-146`

When `verdict == _COMPLETION_VERDICT` the function returns before looking at `code` (`:138`), and
the `code != 0 → failed` branch is unreachable for any run that declared a verdict (`:145`).
Probes H and K rest on this: `status=completed` with `return_code: 1`. Whether a nonzero child is
allowed to settle `completed` is a judgement call, and here the code has already decided
`failed` for bare bookkeeping dispatches on the same fact — the inconsistency is the bug, not the
choice.

### P2 — non-string verdicts are decided by Python's type rules, not by the schema
`workbench/backend/jobs.py:122`

`str(payload.get("verdict") or "")` turns a JSON `true` into `"True"`, which happens to miss
`_COMPLETION_VERDICT`. Probes L1-L3 all settle `failed` (`{"verdict": true, ...}` → `failed`);
the behaviour is correct only by string-formatting accident, and no field other than
`verdict`/`run_id` (`:120`) is validated — `completed`, `return_code` and `invocation_status` in
the file are ignored. Probe L1 (`completed: false, return_code: 137, invocation_status:
INVOCATION_FAILED`) settles `failed` only because its `verdict` string was not the magic one, not
because the file was rejected as self-contradictory.

### P3 — shutdown's bounded worker join discards the verdict read's unbounded disk I/O
`workbench/backend/jobs.py:268-270`, `:523-525`

The comment at `:523-524` (and the `shutdown` docstring `:242-243`) justify reading the verdict
before taking the lock so that "disk I/O must not be held under the queue's condition". The read
is still unbounded in `shutdown`'s 10-second worker join: a run directory whose
`invocation_outcome.json` is a FIFO stalls the reader thread inside `Path.read_text()`.
Probe M (`/tmp/rev6/probe_m_deadlock.py`):

```
M list() during blocked verdict read: 0.0 ms
M enqueue() during blocked verdict read: 0.3 ms
M job status while blocked: running
M shutdown() returned in 10.0s, survivors=[]
M worker alive after shutdown: True (a daemon thread; the process still exits)
```

No deadlock and no stall of `list()`/`cancel()` — the stated design goal holds. The residual
cost is that a queued job never settles and `shutdown` burns its full 10 s join, which is
acceptable for a daemon thread; reported only because the file is now read from a
caller-influenced path with no `O_NONBLOCK`/timeout.

### False-failure audit (item (e)): none found

No legitimate run was wrongly rejected in probes A-M or in the 26 tests executed. Specifically
checked: a verified run completes (D, E1); a run whose verdict appears mid-run completes (D);
verdict re-read as `remaining` alignment between `main.py` and `log_dir()` is exact (F); the
signal-killed evaluator fails rather than being mistaken for an unfinished run (C1, E2); the four
identity fields `verify_run.py` demands (`report run/model/checkpoint/protocol`,
`verify_run.py:69-74`) are the same four `main.py:443-461` puts in provenance.

## 3. Probe index

| Probe | File | Question |
|---|---|---|
| A | `/tmp/rev6/probe_a_noverdict.py` | dispatch naming a run with no verdict |
| B | `/tmp/rev6/probe_b_forged.py` | cross-run verdict / hand-forged verdict / report deleted |
| C | `/tmp/rev6/probe_c_signal.py` | SIGKILLed child; silent child; no-model dispatch |
| D | `/tmp/rev6/probe_d_order.py` | verdict written mid-run |
| E | `/tmp/rev6/probe_e_evidence.py` | native exit code in all four artifacts |
| F | `/tmp/rev6/probe_f_naming.py` | `main.py` run directory == `log_dir()` |
| G | `/tmp/rev6/probe_g_staledir.py` | operator refuses a pre-existing run directory |
| H | `/tmp/rev6/probe_h_stale_verdict.py` | stale verdict replayed by a failing child |
| I | `/tmp/rev6/probe_i_inspect.py` | reason + artifact path inspectable after a rejection |
| J | `/tmp/rev6/probe_j_ledger_contradiction.py` | verdict vs. HMAC ledger disagreement |
| K | `/tmp/rev6/probe_k_child_forges.py` | child forges the verdict; operator crashed, no ledger entry |
| L | `/tmp/rev6/probe_l_fields.py` | contradictory / non-string verdict payloads |
| M | `/tmp/rev6/probe_m_deadlock.py` | lock discipline while the verdict read blocks |

## 4. Findings against the questions asked

- (a) `completed` without a validated proof — **CONFIRMED**, probes H/J/K. Exit code, report,
  proof, ledger and reason are all ignored; one string in one file decides.
- (b) a success claim without corroboration — the queue (a) above; `pc.sh`
  `validate_and_summarize()` itself does call `verify_run.py` and `fail_stage`s on failure
  (`pc.sh:1054-1104`), and `pc.sh` runs under `set -euo pipefail` (`pc.sh:5`) with `fail_stage`
  calling `exit 1` (`pc.sh:48-64`), so no success path was reproduced there.
- (c) forged exit 0 from `verify_run.py` — **NOT REPRODUCED** in this slice. It reads the report
  digest from the HMAC ledger (`verify_run.py:85-89`), the ledger's `return_code` and
  `invocation_status` (`:92-98`), and the evaluator's own `validate_run_manifest`
  (`:103-110`); the verdict payload it prints is not what its exit code is derived from.
- (d) `release_verify.sh` — out of this slice.
- (e) false failures — **none found**.
- (f) evidence loss — the child exit code, the rejection reason and the artifact path all remain
  inspectable (probe I, probe E); the one integrity defect is that the queue's *own*
  `return_code` field and its `status` can contradict each other (probes H, K).

## 5. Verdict

**UNSOUND.** Item (a) — "a path where a job can still reach `completed` without a validated
proof" — is reproduced four times, once with a hand-written file and a no-op child, once with a
stale file and a failing child, once with a hostile child under a crashed operator and no ledger
entry, and once against a signed ledger entry that says the run failed. The fix correctly removes
the fabricated report from `pc.sh` and correctly fails closed on *missing* verdicts, but it
replaced "trust the exit code" with "trust a string the evaluated child can write", and the test
suite was written in the one shape (verdict written by hand, noop child) that cannot observe the
difference.

Minimal close (not a feature request): have the queue consume
`workbench/scripts/verify_run.py`'s verdict — or at least require the verdict to agree with the
run's HMAC ledger entry — instead of the bare `PROOF_VALIDATED` string, and add one queue-level
test that dispatches the real operator through the real queue.
