# 80 — Service supervision and recovery

Branch `research/cir-failure-workbench`, base HEAD `a58b51db3529696519b2ce878ab302c5b0e3e837`.
Control interpreter `workbench/envs/ctl312h/bin/python`. Collected 2026-11-11 19:16 +0700.

## Mechanism

`workbench/scripts/serve_workbench.py` now owns a `Supervisor` that keeps every child it starts:

- **Detection.** One pass every 0.2 s calls `child.poll()`. Any exit — clean status or signal —
  is an unexpected death; the launcher no longer returns on the first child exit.
- **Restart.** The dead child's process group is `SIGKILL`ed (so a surviving `npm` shell, `vite`,
  or uvicorn worker cannot hold the port), then the same argv is respawned with
  `start_new_session=True`.
- **Backoff.** `restart_delay(failures) = min(0.2 * 2**(failures-1), 8.0)` → 0.20, 0.40, 0.80,
  1.60, … 8.00 s. A child that stays up 30 s (`STABILITY_WINDOW`) resets the counter, so a long
  healthy stint never inherits old backoff. A crash loop therefore cannot spin.
- **Logging.** Every start and every death is one stderr line with the local timestamp, the exit
  status or signal name, and both PIDs:
  `[SUPERVISOR] <ts> backend PID 995987 died at <ts> (signal SIGKILL); restart 1 in 0.20s`.
  Children inherit the launcher's stdout/stderr, so all their output lands in the same file.
- **Log preservation.** The launcher only ever appends. `rotate_logs` inspects the inherited
  stderr target; once it exceeds 16 MiB, a fresh `serve.log.<stamp>` segment is opened
  `O_APPEND` and `dup2`ed onto fd 1/2, and the previous segment is left byte-for-byte intact
  (never truncated, never renamed onto the writer's path).
- **Shutdown.** SIGTERM/SIGHUP still forward SIGTERM to the children (so the backend's shutdown
  handshake reaps queued job process groups) and set `stopping`, which suppresses restarts for
  what the operator deliberately stopped.

Nothing here claims a sender for the historical deaths: the SIGTERM signature in
`raw/service-death-log*.txt` remains **sender unknown**. Supervision makes the sender irrelevant
to availability, it does not identify it.

## Isolated proof (:8200 backend, :5300 frontend, `/tmp/workbench_supervise_8200.log`)

Launcher: `serve_workbench.py --production-frontend --host 127.0.0.1 --backend-port 8200
--frontend-port 5300`, log via `>>`. Supervisor PID 998038, unchanged through all three cases.
Prerequisite state before kills: `/api/benchmark-runs` 200, `:5300/` 200,
`/api/composition/runs/10173dc45dea8c90fd2a284b?subset=full` 200.

Kill command per case (pid and pgid of the supervised child, read from `ps --ppid <supervisor>`):
`kill -9 <pid> && kill -9 -- -<pgid>`.

| case | killed (service, pid/pgid) | recovered PID | ms → backend 200 | ms → frontend 200 | ms → circus run 200 |
|---|---|---|---|---|---|
| backend | backend 998039 / 998039 | 998420 | **4274** | 10 (untouched) | **5480** |
| frontend | frontend 998040 / 998040 | 998513 | 9 (untouched) | **471** | 15 (untouched) |
| both | backend 998420, frontend 998513 | 998607 / 998608 | **4530** | **669** | **5563** |

Recovery time is measured from the `kill -9` syscall return to the first HTTP 200 while polling
all three endpoints in one loop at 50 ms intervals. The existing circus run
`10173dc45dea8c90fd2a284b` (`circus_fashioniq_release_v1`, `subset=full`) was re-read after every
restart and returned its metrics each time (`coverage`, `composition_gap_ndcg`,
`gap_{text_only,image_only,multimodal}_ndcg`), i.e. restarts preserve artifact readability.

Supervisor log, verbatim (the backoff ladder is visible across the four backend deaths):

```
[SUPERVISOR] 2026-11-11T19:17:33+0700 backend started PID 998039
[SUPERVISOR] 2026-11-11T19:17:33+0700 frontend started PID 998040
[SUPERVISOR] 2026-11-11T19:17:42+0700 backend PID 998039 died at 2026-11-11T19:17:42+0700 (signal SIGKILL); restart 1 in 0.20s
[SUPERVISOR] 2026-11-11T19:17:42+0700 backend started PID 998420
[SUPERVISOR] 2026-11-11T19:17:50+0700 frontend PID 998040 died at 2026-11-11T19:17:50+0700 (signal SIGKILL); restart 1 in 0.20s
[SUPERVISOR] 2026-11-11T19:17:50+0700 frontend started PID 998513
[SUPERVISOR] 2026-11-11T19:17:52+0700 backend PID 998420 died at 2026-11-11T19:17:52+0700 (signal SIGKILL); restart 2 in 0.40s
[SUPERVISOR] 2026-11-11T19:17:52+0700 frontend PID 998513 died at 2026-11-11T19:17:52+0700 (signal SIGKILL); restart 2 in 0.40s
[SUPERVISOR] 2026-11-11T19:17:53+0700 backend started PID 998607
[SUPERVISOR] 2026-11-11T19:17:53+0700 frontend started PID 998608
```

Log continuity: the file begins with the `[RUN]` line written before any kill and still contains
it after all restarts (`head -1 /tmp/workbench_supervise_8200.log` = `[RUN] (backend) …--port
8200`); 3 backend "Application startup complete" and 3 frontend `Local:` banners accumulated in
one file across the initial start plus two restarts each.

Two earlier rounds on the same ports reproduced the same behaviour and additionally exercised
the backoff ladder 0.20 → 0.40 → 0.80 → 1.60 s and the 30 s stability reset (`frontend PID
976731 stable 30s; backoff reset`, supervisor PID 974450). Logs preserved at
`/tmp/workbench_supervise_8200.round1.log` and `/tmp/workbench_supervise_8200.round2.log`; their
recovery was 9896/473/11047 ms and 4207/622/5158 ms respectively — the first round was slower
only because the backend was still warming cold caches.

## Live rollout (:8000 / :5173)

1. The previous live launcher (PID 821787, pre-supervision code) was stopped with `kill -TERM
   821787`; it forwarded SIGTERM, logged `[SKIP] interrupted; stopping children` and
   `Finished server process [821788]` into the existing `/tmp/workbench_serve.log`, which was
   reopened with `>>` (never `>`) — the four committed death signatures are still in it
   (`grep -c "Finished server process" /tmp/workbench_serve.log` = 5, including the new
   deliberate stop).
2. Restarted as supervisor: same argv, `--production-frontend --host 127.0.0.1 --backend-port
   8000 --frontend-port 5173`, `WORKBENCH_CIRCUS_RELEASE_ROOT=/home/heheboiz/data/cir/workbench/
   artifacts/circus_release`, `PYTHONUNBUFFERED=1`, `>> /tmp/workbench_serve.log`.
   Supervisor PID **999204**, backend **999208**, frontend **999209** (the first supervised live
   start, PID 987539, was replaced once the isolated suite had run against the final code; its
   `[SKIP] interrupted; stopping children` line and the new `[SUPERVISOR] … started` lines are
   both in the same append-only log).

Live checks (all three after rollout):

```
5173/                                                         200 0.001559s
8000/api/benchmark-runs                                       200 0.007817s
8000/api/composition/runs/10173dc45dea8c90fd2a284b?subset=full 200 0.005718s
8000/api/runs/parity-gate                                     200 0.003810s
```

`/tmp/workbench_serve.log` after the rollout: 516690 bytes, 5 `Finished server process` lines
(the four historical deaths plus the new deliberate stop), first line still the pre-rollout
`[RUN] (backend) …--port 8000`, and both supervised starts present — append-only, never
truncated.

Served bundle:

```
curl -s http://127.0.0.1:5173/ | grep -o 'assets/index-[^"]*\.js'   -> assets/index-BLH2yome.js
sha256sum workbench/frontend/dist/assets/index-BLH2yome.js
702ed2e93a1bf86ba2094fe62791d4272dcdbf36e8caf7f757f08cf96805d681  workbench/frontend/dist/assets/index-BLH2yome.js
```

Match. `dist/` was not rebuilt; no frontend source or asset was touched.

## Stale :8765

Identified by port and route behaviour only, never by PID: `:8765` answered **404** for
`/api/runs/parity-gate` while `:8000` answered **200** for the same path — the same app on a
different port, stale since the earlier session (it also had a dev `vite` twin on `:5765`).

```
before: 8765=404  8000=200
after : 8765=000  8000=200     (connection refused; nothing listening on 8765 or 5765)
```

`kill -TERM` was sent to the process that `ss -ltnp` showed listening on `:8765`, plus the
orphaned `npm run dev --port 5765` twin of the same session. `:8000` and `:5173` were unaffected
throughout.

## Tests

```
workbench/envs/ctl312h/bin/python -m pytest workbench/tests/test_service_supervision.py \
    workbench/tests/test_operator_runtime.py -q
..........                                                               [100%]
10 passed in 1.99s
```

New file `workbench/tests/test_service_supervision.py` — 3 tests, real processes, no ports:

1. `test_child_death_is_restarted_and_logged` — `SIGKILL` a supervised stub, assert a new PID,
   `restarts == 1`, and that the log carries the timestamped death line, the observed
   `signal SIGKILL`, and the replacement's start line.
2. `test_backoff_is_exponential_capped_and_bounded` — the delay ladder is monotone, starts at
   0.20 s and saturates at 8.00 s; a 1.5 s crash loop (`SystemExit(3)`) yields 1–6 restarts
   rather than dozens.
3. `test_log_rotation_keeps_previous_segment` — with `>>`-style fd 1/2 on a real file, rotation
   opens a new segment, the earlier bytes stay in the original file, and the post-rotation write
   lands only in the new one (`O_APPEND`, never a truncation).

Also `workbench/tests/test_pipeline.py` (32 passed) still asserts the launcher's dry-run contract,
and `test_operator_runtime.py`'s 4 launcher tests pass unchanged (7 total with the new file's 3,
i.e. the 10 above).
