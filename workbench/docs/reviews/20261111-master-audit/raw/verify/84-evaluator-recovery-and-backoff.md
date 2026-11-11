# 84 — evaluator recovery and backoff (verify)

Base commit `46baf126`, branch `research/cir-failure-workbench`. Control interpreter
`workbench/envs/ctl312h/bin/python` (CPython 3.12.11). LIVE service (supervisor PID 999204, ports
8000/5173) was never touched; every process experiment ran in `/tmp` on port 8411.

## 1. Task 1 — the exponent was clamped AFTER the multiply

`/tmp/backoff_repro.py` loads `workbench/scripts/serve_workbench.py` exactly the way
`workbench/tests/test_service_supervision.py` does (`importlib.util.spec_from_file_location`).

Before the fix, measured on CPython 3.12.11:

```text
python 3.12.11
restart_delay(1) = 0.2
restart_delay(2) = 0.4
restart_delay(3) = 0.8
restart_delay(4) = 1.6
restart_delay(5) = 3.2
restart_delay(6) = 6.4
restart_delay(7) = 8.0
restart_delay(8) = 8.0
restart_delay(1025) RAISED OverflowError: int too large to convert to float
Traceback (most recent call last):
  File "/tmp/backoff_repro.py", line 14, in <module>
    print(f"restart_delay({n}) = {module.restart_delay(n)!r}")
                                  ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/heheboiz/data/cir/workbench/scripts/serve_workbench.py", line 128, in restart_delay
    return min(BACKOFF_BASE * 2 ** max(0, failures - 1), BACKOFF_CAP)
               ~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~
OverflowError: int too large to convert to float
restart_delay(1000000) RAISED OverflowError: int too large to convert to float
Traceback (most recent call last):
  File "/tmp/backoff_repro.py", line 14, in <module>
    print(f"restart_delay({n}) = {module.restart_delay(n)!r}")
                                  ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/heheboiz/data/cir/workbench/scripts/serve_workbench.py", line 128, in restart_delay
    return min(BACKOFF_BASE * 2 ** max(0, failures - 1), BACKOFF_CAP)
               ~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~
OverflowError: int too large to convert to float
```

**OverflowError: yes**, first at `failures = 1025`. `2 ** 1024` is computed as an `int`, then
`float.__rmul__` tries to convert it and fails; `min()` never runs. N.B. the *supervisor* cannot
reach 1025 restarts in one boot quickly, but nothing in the function bounded the argument, and the
whole point of a backoff is to survive an argument that grows without bound.

### The fix

`workbench/scripts/serve_workbench.py`:

```python
#: Smallest exponent whose value already reaches BACKOFF_CAP. Clamping *here* keeps the
#: multiply away from an oversized power: `2 ** 1024` is a big int that cannot be converted
#: to a float at all, so the unclamped form raised OverflowError instead of returning the cap.
BACKOFF_SATURATING_EXPONENT = next(
    exponent for exponent in range(64) if BACKOFF_BASE * 2 ** exponent >= BACKOFF_CAP)

def restart_delay(failures: int) -> float:
    """Capped exponential backoff: a service that dies on startup cannot spin."""
    exponent = min(max(0, failures - 1), BACKOFF_SATURATING_EXPONENT)
    return min(BACKOFF_BASE * 2 ** exponent, BACKOFF_CAP)
```

`BACKOFF_BASE` (0.2) and `BACKOFF_CAP` (8.0) are unchanged. The saturating exponent is derived from
them (`BACKOFF_SATURATING_EXPONENT == 6`), not hard-coded, so changing either constant keeps the
clamp exact.

### Measured ladder after the fix

| `failures` | `restart_delay` |
|---|---|
| 0 | 0.2 |
| -1 | 0.2 |
| 1 | 0.2 |
| 2 | 0.4 |
| 3 | 0.8 |
| 4 | 1.6 |
| 5 | 3.2 |
| 6 | 6.4 |
| 7 | 8.0 |
| 8 | 8.0 |
| 9 | 8.0 |
| 1025 | 8.0 |
| 10**6 | 8.0 |
| 10**12 | 8.0 |

Timing: 1000 calls at `failures = 10**6` → 0.000130 s; the whole `1..2000` sweep → 0.000260 s.
No giant-int arithmetic happens at all.

Regressions added to `workbench/tests/test_service_supervision.py` (the 3 pre-existing tests are
untouched and still pass):

* `test_backoff_ladder_is_exact` — the `0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 8.0` sequence for 1..7;
  `failures <= 0` is the first step.
* `test_backoff_saturates_without_giant_int_arithmetic` — `8.0` at 7, 8, 1025, 10**6, 10**12,
  inside a 1 s budget, plus the invariant that the clamp is the *smallest* saturating exponent.
* `test_backoff_is_monotone_over_a_long_crash_loop` — non-decreasing over 1..2000.

## 2. Task 2 — crash loop with an occupied port

Setup: `python -m http.server 8411 --bind 127.0.0.1` (pid 1213355) held the backend port. The real
launcher was then started against the same port, in `/tmp`, with
`WORKBENCH_ARTIFACTS_ROOT=/tmp/work2/artifacts`.

Two separate observations, because they answer different questions:

**(a) The real entrypoint refuses early.** `serve_workbench.py --backend-only --backend-port 8411`
printed one line and exited:

```text
[BLOCKED] backend port unavailable: 127.0.0.1:8411
```

rc = 2. That is `main()`'s `port_is_available` preflight; it never reaches `Supervisor`. So the
"occupied port" crash loop is reachable only in the race the supervisor documents — the port is
taken between the preflight and the child's `bind`.

**(b) The supervisor's behaviour in that race**, driven through the real `serve.main()` with
`port_is_available` patched to `True` (everything else real: argv parsing, `build_commands`, the
`Supervisor`, real `python -m uvicorn` children, real log lines). 10 s window, then SIGINT. Verbatim
`/tmp/work2/cli_loop_held.log`:

```text
[RUN] (backend) /home/heheboiz/data/cir/workbench/envs/ctl312h/bin/python -m uvicorn workbench.backend.main:app --host 127.0.0.1 --port 8411
[SUPERVISOR] 2026-11-11T23:38:03+0700 backend started PID 1213893
INFO:     Started server process [1213893]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
ERROR:    [Errno 98] error while attempting to bind on address ('127.0.0.1', 8411): [errno 98] address already in use
INFO:     Waiting for application shutdown.
INFO:     Application shutdown complete.
[SUPERVISOR] 2026-11-11T23:38:04+0700 backend PID 1213893 died at 2026-11-11T23:38:04+0700 (exit status 3); restart 1 in 0.20s
[SUPERVISOR] 2026-11-11T23:38:04+0700 backend started PID 1213895
...
[SUPERVISOR] 2026-11-11T23:38:04+0700 backend PID 1213895 died at 2026-11-11T23:38:04+0700 (exit status 3); restart 2 in 0.40s
[SUPERVISOR] 2026-11-11T23:38:05+0700 backend started PID 1213903
...
[SUPERVISOR] 2026-11-11T23:38:05+0700 backend PID 1213903 died at 2026-11-11T23:38:05+0700 (exit status 3); restart 3 in 0.80s
[SUPERVISOR] 2026-11-11T23:38:06+0700 backend started PID 1213914
...
[SUPERVISOR] 2026-11-11T23:38:06+0700 backend PID 1213914 died at 2026-11-11T23:38:06+0700 (exit status 3); restart 4 in 1.60s
[SUPERVISOR] 2026-11-11T23:38:08+0700 backend started PID 1213943
...
[SUPERVISOR] 2026-11-11T23:38:08+0700 backend PID 1213943 died at 2026-11-11T23:38:08+0700 (exit status 3); restart 5 in 3.20s
[SUPERVISOR] 2026-11-11T23:38:12+0700 backend started PID 1213967
...
[SUPERVISOR] 2026-11-11T23:38:12+0700 backend PID 1213967 died at 2026-11-11T23:38:12+0700 (exit status 3); restart 6 in 6.40s
[SKIP] interrupted; stopping children
[DRIVER] main returned 130
```

(The `...` elisions are the child's own `Started server process` / `Waiting for application
startup.` / `Application startup complete.` / `ERROR: [Errno 98] ...` / `Waiting for application
shutdown.` / `Application shutdown complete.` block, identical every cycle; the full 52-line file is
reproducible from the procedure above.)

* **restart count over the 10 s window: 6** (`restarts=6, failures=6`), not the ~50 an unthrottled
  0.2 s poll would produce.
* **Observed inter-restart gaps:** start→death ≈ 0–1 s each (the child fails to bind immediately),
  and the waits between a death and the next start were 0.20, 0.40, 0.80, 1.60, 3.20, 6.40 s —
  exactly the ladder. Timestamps are second-resolution, which is why the sub-second steps look
  identical there; the `in X.XXs` field of each line carries the exact value.
* **Every death line is readable in the log file**: each death is one `[SUPERVISOR] ... died at
  ... (exit status 3); restart N in X.XXs` line, and the child's own `Errno 98` line precedes it.
  Both stdout and stderr are inherited, so they interleave in the same append-only file — the log
  is readable, but it is *not* line-atomic between the two streams when a child dies mid-write (see
  §7).

Note: the child exits with **status 3** (uvicorn's startup-failure code), not `signal`; `describe_exit`
reports it as `exit status 3`.

**(c) Normal start after release.** Holding process killed, `ss -ltnp` showed no listener on 8411,
then the same real entrypoint:

```text
[SUPERVISOR] 2026-11-11T23:38:20+0700 backend started PID 1214053
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8411 (Press CTRL+C to quit)
[DRIVER] free-start code 130
```

One start, no restarts. Cleaned up: holder pid 1213355 killed, every child reaped by
`supervisor.stop()`; `ps` afterwards showed only the LIVE service (999208/999204) and the
pre-existing `http-server` (2480984), neither of which this task started.

## 3. Task 3 — `jobs.py` startup recovery

Startup hook: `workbench/backend/main.py:52 _recover_interrupted_jobs()` →
`SingleGpuQueue.recover_interrupted([.../queue_jobs])` (`jobs.py:502`) → `JOBS.adopt_recovered`
(`jobs.py:340`) for each record. Called from the FastAPI `startup` event (`main.py:111`).

### Finding 1 — in-flight records become `failed`, artifacts untouched

Verified with real fixtures for `queued`, `preparing`, `running`: each `queue_state.json` is
rewritten to `status: failed`, `error: "workbench restarted while the job was in flight"`, a fresh
`completed_at`; `queue_stdout.log`, `queue_stderr.log` and a nested `artifacts/result.json` are
byte-identical (directory snapshot compared before/after, only `queue_state.json` removed from both
sides). Recovery never opens the artifact tree for writing.

### Finding 2 — a `completed` record is not touched at all

Recovery returns `[]` for it, and both `mtime_ns` and `inode` are unchanged (the code path is a
`status in (...)` test before any write). Proven by `test_finished_record_is_not_touched`.

### Finding 3 — missing metadata: what happened today, and the minimal fix

Measured before the fix (`/tmp/probe3/probe.py`):

* A record that is a **JSON list** (`[{"job_id": ...}]`) or scalar (`"hello"`, `7`) crashed the
  whole scan: `AttributeError: 'list' object has no attribute 'get'`, raised out of
  `recover_interrupted`. Because `_recover_interrupted_jobs` wraps the call in a bare
  `except Exception: pass`, the effect in production was silent: **every record sorted after the
  bad one was skipped**, and startup looked normal.
* A record with **no `job_id`** was rewritten to `failed` and adopted, and `adopt_recovered`
  synthesised a **bogus entry with a random uuid** and `log_directory: ""` (confirmed:
  `('9af4c121-…', 'failed', '', 'workbench restarted while the job was in flight')`). Nothing on
  disk can ever be matched to that id, and it pollutes `/api/jobs` forever.
* A record with `job_id` but **no `log_directory`** is legitimate-ish (the field is lost from a
  hand-edited snapshot): it is still recovered and adopted with `log_directory: ""`. That is
  harmless — recovery writes through the path it globbed, not through the record's field.
* **Malformed JSON** and a **missing `status`** were already handled (skip / no-op).

Minimal fix (`jobs.py:509-515`), two guards only:

```python
if not isinstance(record, dict) or not record.get("job_id"):
    continue
```

Non-dict JSON no longer aborts the scan; an unnamed record is no longer turned into a phantom job.

### Finding 4 — recovery never reruns a job

`subprocess.Popen` monkeypatched to raise `AssertionError`; `recover_interrupted` + `adopt_recovered`
over four records (three in-flight, one completed) run clean. Structurally, recovery only reads/writes
`queue_state.json`: no `Popen`, no `_work` dispatch, and adopted jobs are terminal (`failed`), so
`_next_job` (which selects `status == "queued"`) can never pick them up.

### Finding 5 — ownership audit of every kill reachable at startup / recovery

| site | kind of id | provably owned? |
|---|---|---|
| `jobs.py:392` `os.killpg(group, 0)` | group id for a **live** dispatched job, `os.getpgid(process.pid)` captured at dispatch (`jobs.py:626`), or the in-memory `_process_groups[job_id]` | **yes** — signal 0 is a probe, never delivers |
| `jobs.py:438` `os.killpg(group, signum)` (SIGTERM/SIGKILL) | same, from `cancel()`/`shutdown()` only | **yes** — captured from a live child this process started; never read from disk |
| `jobs.py:429` `os.getpgid(process.pid)` | live `Popen` handle | **yes** (fallback only) |
| `jobs.py:440` `process.send_signal`, `jobs.py:452` `process.kill()` | live `Popen` handle | **yes** |
| `serve_workbench.py:86,95` `os.killpg(os.getpgid(child.pid), …)` | live `Popen` handle | **yes** |
| `serve_workbench.py:245` `os.killpg(process.pid, SIGKILL)` | live `Popen` child, `start_new_session=True` ⇒ pid == pgid | **yes** |
| `serve_workbench.py:333` `os.kill(process.pid, SIGTERM)` | live `Popen` handle, in the stop handler | **yes** |
| `jobs.py:385-402` (comments only) | — | no code |
| `pipeline.py:2137,2147,2152,2164` `os.killpg(proc.pid, …)` | live `Popen` under `run_command`, `start_new_session=True` | **yes** — and not reachable from recovery (interactive CLI path) |

No `pkill`, no `killall`, no `os.kill` by a pid read from a file, no `os.killpg` on a group id read
from disk. `recover_interrupted` itself contains **no** signal call at all. `grep` for
`.pid`-file reads under `workbench/backend` and `workbench/scripts` found none.

Proven with real processes (`/tmp/foreign_probe2.py`): after `recover_interrupted` +
`adopt_recovered` + a blocked dispatch + `shutdown()`, all three survived — (a) a child started by
the probe (pid 1217175), (b) a `sleep 300` started by the probe in its own session
(pid 1217176), (c) a process the probe did **not** start, the pre-existing `http-server`
pid 2480984, whose `/proc/<pid>/stat` state and `starttime` were identical before and after
(`S` / `26970850`).

### Finding 6 — the GPU guarantee, and the guard

**`recover_interrupted` does NOT guarantee the GPU is free, and before this change nothing blocked a
new dispatch.** The record it reads contains `job_id`, `model_id`, `checkpoint_id`, `protocol_id`,
`status`, `started_at`, `completed_at`, `return_code`, `error` — **no pid and no pgid**. A job runs
`start_new_session=True`, so after a hard kill of the backend the operator's evaluator child keeps
its own session and can keep holding the single GPU indefinitely. Recovery marked the record
`failed`, the in-memory queue forgot the job entirely (nothing adopted it as live), and the next
`POST /api/jobs` + worker dispatch would start a **second** evaluation on the same device. That is
the blocker.

Minimal honest guard (no kill, ever — the group id was never persisted, so there is no id that can
be proven to refer to the crashed job):

* `recover_interrupted` marks `gpu_may_be_held: true` on records whose status was `preparing` or
  `running` (a `queued` record never had a process). Persisted in `queue_state.json`, so the flag
  survives further restarts and is visible to an operator.
* `adopt_recovered` puts those job ids in `self._gpu_orphans`; `_persist` carries the flag.
* The worker, immediately before dispatch, refuses any job while `_gpu_orphans` is non-empty:
  `status = "blocked"`, `error = "a job recovered from a crash may still hold the GPU (<ids>);
  release the device and restart the workbench before dispatching"`, plus a `[BLOCKED]` line in
  `stderr_tail` (so it shows up in the status API, like every other named refusal).

Measured (`/tmp/foreign_probe2.py`):

```text
dispatch while orphan recorded -> blocked | a job recovered from a crash may still hold the GPU (job-foreign); release the device and restart the workbench before dispatching
job command ran? False
```

The refusal is deliberately escape-free by *restart*: confirming the device is free is an operator
act (check `nvidia-smi`), after which "restart the workbench" is the documented release. It is not
self-clearing, because a self-clearing flag would be the same blind dispatch with extra steps.

### New tests — `workbench/tests/test_job_recovery_ownership.py` (11 tests)

`tmp_path` fixtures only; no ports, no GPU, no network:
in-flight→failed + byte-identical artifacts (1), completed untouched mtime+inode (1),
six incomplete/malformed shapes parametrised without raising (6), recovery spawns nothing (1),
foreign processes survive (1), dispatch refused while an orphan may hold the GPU (1).

## 4. Task 4 — the required test files

```text
$ workbench/envs/ctl312h/bin/python -m pytest workbench/tests/test_service_supervision.py \
    workbench/tests/test_job_recovery_ownership.py \
    workbench/tests/test_process_group_cancellation.py \
    workbench/tests/test_operator_runtime.py workbench/tests/test_pipeline.py -q
..............................................................           [100%]
62 passed in 78.11s (0:01:18)
```

`test_service_supervision.py`: 6 passed (3 pre-existing + 3 new).
`test_job_recovery_ownership.py`: 11 passed.

## 5. Blast radius

`git diff --stat` = `workbench/backend/jobs.py +44 −1`, `workbench/scripts/serve_workbench.py`
(the backoff function + one derived constant), `workbench/tests/test_service_supervision.py`
(appended tests), plus the new test and this file. `_persist` gained the `gpu_may_be_held` key; every
existing consumer reads named keys (`test_concurrency_ownership._persisted_job`, `test_pc_launcher`,
`test_final_hardening_regressions`), so the extra key is additive. `WORKBENCH_*` records already on
disk from released versions have no such key and read as `False`.

## 6. What is still unknown

* **Who historically sent the SIGTERM that started this investigation remains unknown.** No log line,
  journal entry, or code path in this checkout identifies a sender. Everything above describes what
  the code does *when* a stop arrives (launcher forwards SIGTERM to the backend, backend reaps job
  groups), not who sent the original signal. Only `docker stop` / an operator / the OOM killer are
  candidates; none is evidenced.
* Whether the real crash loop ever reached `failures ≥ 1025` in production: no such log was found.
  The overflow is a proven latent defect of the function, not a proven past incident.
* Whether an orphan evaluator actually survived any real backend crash: the guard added here is
  preventative, and no concrete orphaned process was observed on this host.
* The `gpu_may_be_held` refusal stays until restart by design. A future feature could verify device
  state (`nvidia-smi` compute-apps against a persisted `(pid, starttime)` pair) and clear it safely;
  that requires persisting verifiable process identity at dispatch time, which is not done here.

## 7. Log-readability caveat observed

In §1(b) the child's uvicorn lines and the supervisor's lines share one file. In the earlier
`/tmp/work2/sup_held.log` run (a different driver) several `ERROR: [Errno 98]` lines were duplicated
and one death line lost its neighbours, because two processes append to the same fd with independent
buffers. The supervisor's own lines are written with `flush=True` and were always readable; the
interleaving is a property of inherited stdout+stderr, unchanged by this work and not fixed here.

## 8. Porcelain: three entries, two of them not this task's

After the commit, `git status --porcelain` is:

```text
?? CLEANUP_REPORT.md.save
?? workbench/systemd/
?? workbench/tests/test_systemd_user_services.py
```

`CLEANUP_REPORT.md.save` is the pre-existing entry the task expected. The other two appeared
**during** this session (mtimes `2026-11-11 23:40:11` … `23:43:26`, vs. session start before
`23:37`): `workbench/systemd/{__init__,render_units}.py`, two `*.service.in` templates, and
`workbench/tests/test_systemd_user_services.py`. They are another writer's in-flight work — a
systemd user-unit renderer with its own test file, unrelated to backoff or job recovery. They were
neither written nor staged by this task and are left untouched (deleting unrelated work is out of
scope); nothing in the commit above references them.
