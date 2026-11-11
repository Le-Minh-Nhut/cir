# 86 — systemd user services for the workbench

Scope: replace nothing. These two units are the *installable* form of the same servers the
supervised launcher starts, proven on isolated ports **8401 / 5401** with an isolated artifacts
root under `/tmp/cir_systemd_iso/`. The live 8000/5173 stack (supervisor PID 999204) was not
stopped, restarted, or signalled at any point; the only interaction was read-only HTTP GET and
`journalctl`/`systemctl show`.

Host facts reused (not re-measured): systemd 255, PID 1 is systemd, `Linger=yes`, unified
hierarchy, no prior cir/workbench user unit.

## Delivered files

| Path | Purpose |
|---|---|
| `workbench/systemd/render_units.py` | renderer: reads `resolve_config()` + `build_commands`, installs, `--dry-run`, `--verify` |
| `workbench/systemd/cir-workbench-backend.service.in` | backend unit template |
| `workbench/systemd/cir-workbench-frontend.service.in` | frontend unit template |
| `workbench/systemd/__init__.py` | package marker so the renderer is importable from tests |
| `workbench/tests/test_systemd_user_services.py` | 9 tests, 7 of them with no systemd manager needed |

Templates are not hand-edited in production: the renderer substitutes `@EXEC_START@`,
`@WORKING_DIRECTORY@`, `@ENVIRONMENT@`, `@DESCRIPTION@` and `@BACKEND_UNIT@` from the repo's own
resolver. No path in this document was transcribed by hand from a shell — every absolute path
below came out of the renderer.

Why the resolver and not a copied `serve_workbench.py` argv: `build_commands` builds the backend
as `[sys.executable, "-m", "uvicorn", "workbench.backend.main:app", "--host", …, "--port", …]`
with `cwd = root.parent`, and `resolve_config` folds `workbench.env` under env overrides. Both are
reused verbatim, so a change to either lands in the units on the next `--install`.

## Backend unit, rendered (`--dry-run`, live ports)

```ini
[Unit]
Description=CIR workbench backend (uvicorn on 127.0.0.1:8000)
# A crash loop must not spin: 5 starts inside 60 s, then the unit stays failed until an operator
# resets it. RestartSec below is the between-attempt delay; this pair is the ceiling on attempts.
StartLimitIntervalSec=60
StartLimitBurst=5

[Service]
Type=simple
# uvicorn is the main process: no shell, no wrapper, so MainPID is the server itself and signals
# reach the process that runs the shutdown hook.
ExecStart=/home/heheboiz/data/cir/workbench/envs/ctl312h/bin/python -m uvicorn workbench.backend.main:app --host 127.0.0.1 --port 8000
WorkingDirectory=/home/heheboiz/data/cir
Environment="CIR_DATA_ROOT=/home/heheboiz/data/cir/data"
Environment="CIR_REPO_ROOT=/home/heheboiz/data/cir"
Environment="FASHIONIQ_ROOT=/home/heheboiz/data/cir/data/FashionIQ"
Environment="WORKBENCH_ARTIFACTS_ROOT=/home/heheboiz/data/cir/workbench/artifacts"
Environment="WORKBENCH_CHECKPOINT_ROOT=/home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads"
Environment="WORKBENCH_CIRCUS_RELEASE_ROOT=/home/heheboiz/data/cir/workbench/artifacts/circus_release"
Environment="WORKBENCH_RESULTS_ROOT=/home/heheboiz/data/cir/workbench/artifacts/results"
Environment="WORKBENCH_THIRD_PARTY_ROOT=/home/heheboiz/data/cir/workbench/third_party"

Restart=always
# 2 s is long enough for a port released by the previous instance to be bindable again and short
# enough that an operator sees the recovery promptly.
RestartSec=2
# mixed: SIGTERM to the main process only (so uvicorn's shutdown hook runs and reaps job groups),
# then SIGKILL to anything left in the cgroup -- including a job child that called setsid() and
# left the session. control-group would SIGTERM the GPU-holding job directly, which skips the
# queue's truthful "failed" bookkeeping.
KillMode=mixed
# The shutdown hook may wait GRACE_SECONDS (10 s) + 5 s per live job group and a 10 s worker join;
# 30 s covers one stalled group without letting a wedged stop hang forever.
TimeoutStopSec=30
# Never start a second instance while the old one still holds the port.
KillSignal=SIGTERM
SendSIGKILL=yes

[Install]
WantedBy=default.target
```

## Frontend unit, rendered (`--dry-run`, live ports)

```ini
[Unit]
Description=CIR workbench frontend (vite preview on 127.0.0.1:5173)
# Ordering only: the proxy reconnects per request, so the frontend may come up while the backend
# is still starting; it must, however, be stopped after the backend when both are stopped.
After=cir-workbench-backend.service
Wants=cir-workbench-backend.service
StartLimitIntervalSec=60
StartLimitBurst=5

[Service]
Type=simple
# npm (Type=simple) is the main process; it spawns `sh -c vite preview`, which spawns node.
# KillMode=mixed is measured (see the 20261111 audit evidence): SIGTERM to npm, then SIGKILL to
# the rest of the cgroup, otherwise the vite process keeps the port and every restart fails with
# EADDRINUSE.
ExecStart=/home/heheboiz/.nvm/versions/node/v25.9.0/bin/npm run preview -- --host 127.0.0.1 --port 5173 --strictPort
WorkingDirectory=/home/heheboiz/data/cir/workbench/frontend
Environment="PATH=/home/heheboiz/.nvm/versions/node/v25.9.0/bin:/usr/local/bin:/usr/bin:/bin"
Environment="WORKBENCH_BACKEND_URL=http://127.0.0.1:8000"

Restart=always
RestartSec=2
KillMode=mixed
TimeoutStopSec=10
KillSignal=SIGTERM
SendSIGKILL=yes

[Install]
WantedBy=default.target
```

`PATH` is rendered because a systemd unit inherits no shell: the live launcher's `npm` comes from
`~/.nvm/versions/node/v25.9.0/bin`, which is not in systemd's default `PATH`. This is the only
reason a frontend env line exists — the backend needs no shim. `npm` is still the `ExecStart`
target, so `MainPID` is npm (which is the process whose death the unit restarts). Node's real
work happens in a grandchild; that is why `KillMode` matters (below).

## MainPID proof (backend)

Test unit `cir-wb-backend-test.service`, rendered with `--backend-port 8401` and isolated roots:

```
MainPID=1219496
comm=python
exe=/home/heheboiz/.local/share/uv/python/cpython-3.12.11-linux-x86_64-gnu/bin/python3.12
argv:
  /home/heheboiz/data/cir/workbench/envs/ctl312h/bin/python
  -m
  uvicorn
  workbench.backend.main:app
  --host
  127.0.0.1
  --port
  8401
cwd=/home/heheboiz/data/cir
cgroup=0::/user.slice/user-1000.slice/user@1000.service/app.slice/cir-wb-backend-test.service
ControlGroup=/user.slice/user-1000.slice/user@1000.service/app.slice/cir-wb-backend-test.service
release_root=WORKBENCH_CIRCUS_RELEASE_ROOT=/home/heheboiz/data/cir/workbench/artifacts/circus_release
### confirm uvicorn is in-process (no reload child)
children=
```

`comm` is `python` because the venv's `bin/python` is a symlink to `python3.12`; `/proc/<pid>/exe`
resolves through the same symlink the live launcher's interpreter does. `/proc/<pid>/cmdline` is
exactly the uvicorn argv — no `sh -c`, no wrapper, and the unit has no children, so `MainPID` is
the server. **No env-shim was needed and none was used**, so no `exec` PID transfer appears
anywhere. `serve_workbench.py` builds this same argv and cwd from the same resolver.

Resolved unit properties (`systemctl --user show`):

```
Type=simple
Restart=always
RestartUSec=2s
TimeoutStopUSec=30s
KillMode=mixed
KillSignal=15
StartLimitIntervalUSec=1min
StartLimitBurst=5
ControlGroup=/user.slice/user-1000.slice/user@1000.service/app.slice/cir-wb-backend-test.service
```

## Numbers and why

- `RestartSec=2` — the port must be bindable again. Measured: SIGTERM released the socket at
  +150 ms and the replacement was listening ≈2 s later; 2 s also keeps a fast crash loop visible
  to an operator instead of hiding it behind a long delay.
- `StartLimitIntervalSec=60`, `StartLimitBurst=5` — five attempts in a minute. Measured on a stub
  unit with an `exit 1` `ExecStart` and the same pair: it reached `failed` after **10 s** and 5
  restarts, with `Start request repeated too quickly.` The unit stays failed (no infinite spin)
  until an operator runs `reset-failed`/`start`; a real backend that crashed on startup settles in
  the same bound.
- `TimeoutStopSec=30` — the shutdown hook's own ceiling is `GRACE_SECONDS` (10 s, `jobs.py`) + 5 s
  SIGKILL escalation per live job group + a 10 s worker join. 30 s covers one stalled group.
  Measured graceful stop with no live job: 172 ms (23:45:37.842425 → 23:45:38.014936).
- `KillMode=mixed` — see the containment section.

## Case A — `kill -TERM <MainPID>`

```
MainPID_before=1216470
before MainPID=1216470 at 23:43:57
sent SIGTERM to 1216470 at +0 ms
after MainPID=1217967 ready in 3092 ms (old pid gone at 151 ms, 51 polls)
MainPID_after=1217967 comm_after=python exe=…/cpython-3.12.11-linux-x86_64-gnu/bin/python3.12
bench=200 comp=200
```

Journal:

```
Nov 11 23:43:57.060816 heheboiz-nhut-pc python[1216470]: INFO:     Shutting down
Nov 11 23:43:57.161706 heheboiz-nhut-pc python[1216470]: INFO:     Finished server process [1216470]
Nov 11 23:43:59.252212 heheboiz-nhut-pc systemd[1735]: cir-wb-backend-test.service: Scheduled restart job, restart counter is at 2.
Nov 11 23:43:59.261988 heheboiz-nhut-pc systemd[1735]: Started cir-wb-backend-test.service - CIR workbench backend (uvicorn on 127.0.0.1:8401).
Nov 11 23:43:59.514481 heheboiz-nhut-pc python[1217967]: INFO:     Started server process [1217967]
Nov 11 23:43:59.515251 heheboiz-nhut-pc python[1217967]: INFO:     Uvicorn running on http://127.0.0.1:8401 (Press CTRL+C to quit)
```

Recovered. **Methodology note, measured:** a naive "poll until `/api/benchmark-runs` answers 200"
produced a false pass — the *dying* instance kept answering 200 for 9 ms after the SIGTERM (first
attempt logged `9 ms (1 polls)` with `after=1215527`, i.e. the old PID). Every number here requires
the old PID's `/proc` entry to be gone before the 200 is credited, because only one process can hold
the listening socket.

## Case B — `kill -KILL <MainPID>`

```
before MainPID=1217967 at 23:44:00
sent SIGKILL to 1217967 at +0 ms
after MainPID=1218038 ready in 2881 ms (old pid gone at 50 ms, 47 polls)
MainPID_after=1218038 comm=python
bench=200 comp=200
```

Journal:

```
Nov 11 23:44:00.986184 heheboiz-nhut-pc systemd[1735]: cir-wb-backend-test.service: Main process exited, code=killed, status=9/KILL
Nov 11 23:44:00.986247 heheboiz-nhut-pc systemd[1735]: cir-wb-backend-test.service: Failed with result 'signal'.
Nov 11 23:44:03.002130 heheboiz-nhut-pc systemd[1735]: cir-wb-backend-test.service: Scheduled restart job, restart counter is at 3.
Nov 11 23:44:03.012012 heheboiz-nhut-pc systemd[1735]: Started cir-wb-backend-test.service - CIR workbench backend (uvicorn on 127.0.0.1:8401).
Nov 11 23:44:03.263587 heheboiz-nhut-pc python[1218038]: INFO:     Started server process [1218038]
Nov 11 23:44:03.264374 heheboiz-nhut-pc python[1218038]: INFO:     Uvicorn running on http://127.0.0.1:8401 (Press CTRL+C to quit)
```

`status=9/KILL` then `Failed with result 'signal'`, `Restart=always` restarts anyway. Recovered.

## Case C — `systemctl --user stop` (must stay stopped), then start

```
CASE_C_before: MainPID=1219086 NRestarts=0 at 23:45:37.839895224
stop returned, journal: 23:45:37.842425 Stopping … → 23:45:38.014936 Stopped (172 ms)
CASE_C_after10s: active=inactive sub=dead MainPID=0 NRestarts=0 port_listeners=0 curl=000 old_pid_alive=NO
CASE_C_restart: active=active MainPID=1219268 bench=200 comp=200
```

Held inactive for 10 s (`NRestarts=0`, no `Scheduled restart job`, port free, old PID gone) — the
unit does **not** restart itself after an operator stop, which is the difference between
`Restart=always` and a stop. `start` brought it back in ≈1.3 s and both readiness endpoints answered.

Also proven during the exercise: `systemctl start` after a *failed* `start` (an earlier stop probe)
starts cleanly; no stale `failed` state blocked it.

## cgroup containment for a detached child (backend)

Stub job, standing in for an evaluator: a process joined to the unit's cgroup (the backend's own
children inherit that membership) that forks a child which calls `os.setsid()`. The child's cgroup
membership is never written — it must inherit.

Containment **before** the kill:

```
detached_child=1219438 at 23:46:06.869239732
child_cgroup=0::/user.slice/user-1000.slice/user@1000.service/app.slice/cir-wb-backend-test.service
child_sid=1219438 backend_sid=1219268 child_pgid=1219438
unit_ControlGroup=/user.slice/user-1000.slice/user@1000.service/app.slice/cir-wb-backend-test.service
cgroup_members=1219268:python 1219437:python 1219438:python
child_ps_before=1219438 Ss python
```

The child is a session *and* process-group leader (`sid == pgid == 1219438`, `Ss`), i.e. it left
the backend's session (1219268) entirely, yet `systemctl --user show -p ControlGroup` and
`/proc/1219438/cgroup` both place it inside the unit's cgroup. Containment: **yes**.

Ordered timeline across the kill of the backend:

```
sent_SIGTERM_to_1219268 at 23:46:06.946587033
child_absent_at=23:46:07.153723576 (poll 5, 50ms)
child_exists_after=NO
child_ps_after=            # ps -p 1219438 empty
unit stopped: active=activating mainpid=0 at 23:46:07.369683698
cgroup_dir_exists=NO members=EMPTY
sleeper_count=0
replacement_ready_at=23:46:10.69443064 mainpid=1219496 bench=200
```

Journal for the same window:

```
Nov 11 23:46:07.015814 python[1219268]: INFO:     Shutting down
Nov 11 23:46:07.116408 python[1219268]: INFO:     Waiting for application shutdown.
Nov 11 23:46:07.116932 python[1219268]: INFO:     Application shutdown complete.
Nov 11 23:46:07.116932 python[1219268]: INFO:     Finished server process [1219268]
Nov 11 23:46:07.123918 systemd[1735]: cir-wb-backend-test.service: Killing process 1219437 (python) with signal SIGKILL.
Nov 11 23:46:07.123933 systemd[1735]: cir-wb-backend-test.service: Killing process 1219438 (python) with signal SIGKILL.
Nov 11 23:46:09.252117 systemd[1735]: cir-wb-backend-test.service: Scheduled restart job, restart counter is at 1.
```

The detached child was dead at 23:46:07.15, the unit finished stopping at 23:46:07.37, and the
replacement could not accept work before 23:46:10.69 — so no replacement had a chance to dispatch
anything while the old child still lived. `KillMode=mixed` was sufficient: SIGTERM went to the main
process alone (its shutdown hook ran and logged the handshake), then systemd SIGKILLed the cgroup
survivors. `KillMode=control-group` was **not** needed and was not adopted; had it been adopted the
GPU-holding job would receive SIGTERM directly, bypassing the queue's "shutdown could not stop…"
bookkeeping in `jobs.py`.

Honest limit: the stub child was not registered in `SingleGpuQueue`, so it was reaped by systemd's
cgroup SIGKILL rather than by `JOBS.shutdown()`. Both paths exist and both end at SIGKILL; a real
job additionally gets a truthful `cancelled`/`failed` record from the hook.

## Frontend: restart and the kill mode needed

Kill the *port-holding* process (node, grandchild):

```
frontend MainPID(npm)=1215557 node=1215569
sent SIGKILL to 1215569
after MainPID=1216559 ready in 2432 ms (old pid gone at 50 ms, 49 polls)
fe_root=200 fe_api=200
journal: npm[1215568] Killed / Main process exited, status=137/n/a / Scheduled restart … / Started …
```

Kill the MainPID (npm):

```
npm=1216771 node=1216783
sent SIGKILL to 1216771
after MainPID=1216962 ready in 2381 ms (old pid gone at 50 ms, 48 polls)
oldnode_alive=NO   root=200 proxy=200
journal: Main process exited, code=killed, status=9/KILL
         cir-wb-frontend-test.service: Killing process 1216782 (sh) with signal SIGKILL.
         cir-wb-frontend-test.service: Killing process 1216783 (node-MainThread) with signal SIGKILL.
         Scheduled restart job, restart counter is at 1. / Started …
```

Which kill mode is required — measured, not assumed. With `KillMode=mixed`, `systemctl stop`
releases the port and leaves nothing behind:

```
before npm=1216559 node=1216571 members=1216559 1216570 1216571
state=inactive port_listeners=0
npm_alive=NO node_alive=NO
leftover_node_procs=
```

With `KillMode=process` (patched in, tested, then reverted to `mixed`):

```
npm=1216962 node=1216974
state=inactive port_listeners=1
leftover_node=YES_1216974
journal: cir-wb-frontend-test.service: Unit process 1216974 (node-MainThread) remains running after unit stopped.
```

So `process` leaves node holding 5401 — the next start fails on `--strictPort` — and `mixed` does
not. **`KillMode=mixed` is kept for the frontend.** (`systemd` prints `Unit process … remains
running after unit stopped` in some `mixed` stops too; in the measured mixed stop above the PID was
absent and the port free 2 s later, so that line is journal ordering, not a surviving process. The
leftover process was killed by hand before reverting the unit.)

## Readiness through the frontend proxy

After a restart, with the backend on 8401:

```
fe_state=active mainpid=1216771
fe_root=200 fe_proxy=200 fe_proxy_comp=200
be_state=active mainpid=1216470
```

`vite.config.mts` proxies `/api` to `WORKBENCH_BACKEND_URL`, which the unit renders as
`http://127.0.0.1:8401` for the test units (and `:8000` for the production render). A proxied
`/api/composition/runs/10173dc45dea8c90fd2a284b?subset=full` through 5401 returned 200, so the
frontend unit is not a separate fiction: it serves the real CIRCUS run through the real backend.

Backend readiness used everywhere in this document:

```
bench8401=200
comp8401=200    # /api/composition/runs/10173dc45dea8c90fd2a284b?subset=full
fe5401=200
feapi5401=200
fecomp5401=200
```

## Journal readability (`journalctl --user -u <unit> --no-pager -n 40`)

Startup + each restart are visible with the PID that took over, filtered here to the lines that
matter (the unfiltered stream also carries per-request `GET` lines and the ledger's
`skipping aggregate report` warnings, both of which come from the RPC layer, not from systemd):

```
Nov 11 23:40:58.522054 systemd[1735]: Started cir-wb-backend-test.service - CIR workbench backend (uvicorn on 127.0.0.1:8401).
Nov 11 23:40:58.768311 python[1215527]: INFO:     Started server process [1215527]
Nov 11 23:40:58.769009 python[1215527]: INFO:     Uvicorn running on http://127.0.0.1:8401 (Press CTRL+C to quit)
Nov 11 23:41:18.255911 python[1215527]: INFO:     Shutting down
Nov 11 23:41:18.356820 python[1215527]: INFO:     Finished server process [1215527]
Nov 11 23:41:20.502244 systemd[1735]: cir-wb-backend-test.service: Scheduled restart job, restart counter is at 1.
Nov 11 23:41:20.511872 systemd[1735]: Started cir-wb-backend-test.service - CIR workbench backend (uvicorn on 127.0.0.1:8401).
Nov 11 23:41:20.753605 python[1215834]: INFO:     Started server process [1215834]
Nov 11 23:41:20.754183 python[1215834]: INFO:     Uvicorn running on http://127.0.0.1:8401 (Press CTRL+C to quit)
…
Nov 11 23:41:44.370559 systemd[1735]: Stopping cir-wb-backend-test.service - CIR workbench backend (uvicorn on 127.0.0.1:8401)...
Nov 11 23:41:44.455110 python[1216051]: INFO:     Shutting down
Nov 11 23:41:44.555873 python[1216051]: INFO:     Finished server process [1216051]
Nov 11 23:41:44.562865 systemd[1735]: Stopped cir-wb-backend-test.service - CIR workbench backend (uvicorn on 127.0.0.1:8401).
Nov 11 23:41:50.604825 systemd[1735]: Started cir-wb-backend-test.service - CIR workbench backend (uvicorn on 127.0.0.1:8401).
…
Nov 11 23:42:22.616642 systemd[1735]: cir-wb-backend-test.service: Killing process 1216354 (python) with signal SIGKILL.
Nov 11 23:42:22.616660 systemd[1735]: cir-wb-backend-test.service: Killing process 1216355 (python) with signal SIGKILL.
Nov 11 23:42:24.752227 systemd[1735]: cir-wb-backend-test.service: Scheduled restart job, restart counter is at 1.
```

Frontend (`-n 40`) shows the same shape plus npm's two script banner lines and vite's
`➜ Local: http://127.0.0.1:5401/`:

```
Nov 11 23:40:59 systemd[1735]: Started cir-wb-frontend-test.service - CIR workbench frontend (vite preview on 127.0.0.1:5401).
Nov 11 23:40:59 npm[1215557]: > cir-failure-workbench@0.1.0 preview
Nov 11 23:40:59 npm[1215557]: > vite preview --host 127.0.0.1 --port 5401 --strictPort
Nov 11 23:40:59 npm[1215569]:   ➜  Local:   http://127.0.0.1:5401/
Nov 11 23:42:29 npm[1215568]: Killed
Nov 11 23:42:29 systemd[1735]: Main process exited, code=exited, status=137/n/a
Nov 11 23:42:31 systemd[1735]: Scheduled restart job, restart counter is at 1.
Nov 11 23:42:31 systemd[1735]: Started cir-wb-frontend-test.service - …
```

Both units are readable without `LogsDirectory=`/journald configuration: uvicorn and npm inherit
the journal stream, so the startup line and every restart are in the default journal.

## Install commands (the reusable product)

```bash
cd /home/heheboiz/data/cir
workbench/envs/ctl312h/bin/python workbench/systemd/render_units.py --dry-run   # print, write nothing
workbench/envs/ctl312h/bin/python workbench/systemd/render_units.py --install   # write + daemon-reload
systemctl --user enable --now cir-workbench-backend.service cir-workbench-frontend.service
workbench/envs/ctl312h/bin/python workbench/systemd/render_units.py --verify    # MainPID/ControlGroup/journal
```

`--dry-run` was proven to write nothing (`test_dry_run_prints_units_and_writes_nothing`: the target
directory does not exist afterwards). Overrides: `--backend-port/--frontend-port/--host`,
`--python`, `--npm`, `--backend-unit/--frontend-unit`, `--dev-frontend`, and repeated
`--set KEY=VALUE` (used to point `WORKBENCH_ARTIFACTS_ROOT`, `WORKBENCH_RESULTS_ROOT`,
`WORKBENCH_CHECKPOINT_ROOT` at `/tmp/cir_systemd_iso/artifacts` for the test deployment; the release
root stayed the real read-only one, which the renderer emits whenever the operator re-renders).

## Cutover (NOT executed here — for the lead)

Order matters: the supervisor owns the ports until it is gone, so stop it before enabling units.

```bash
cd /home/heheboiz/data/cir
# 1. Stop the supervised launcher (its children are stopped by its own signal path).
kill -TERM 999204
for _ in $(seq 1 100); do kill -0 999204 2>/dev/null || break; sleep 0.2; done
# 2. Confirm the ports are free before enabling anything that binds them.
ss -lnt | grep -E ':(8000|5173)\b'   # must print nothing
# 3. Install and enable the units (they carry the same interpreter, cwd, env, ports).
workbench/envs/ctl312h/bin/python workbench/systemd/render_units.py --install
systemctl --user enable --now cir-workbench-backend.service cir-workbench-frontend.service
# 4. Verify.
systemctl --user show -p MainPID -p ControlGroup -p KillMode cir-workbench-backend.service
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/api/benchmark-runs
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:5173/
curl -s -o /dev/null -w '%{http_code}\n' 'http://127.0.0.1:8000/api/composition/runs/10173dc45dea8c90fd2a284b?subset=full'
journalctl --user -u cir-workbench-backend.service --no-pager -n 20
```

Rollback (restores the supervised launcher exactly as it ran; the release root must be in the
environment, and the log is append-only):

```bash
systemctl --user disable --now cir-workbench-frontend.service cir-workbench-backend.service
systemctl --user reset-failed cir-workbench-backend.service cir-workbench-frontend.service 2>/dev/null
cd /home/heheboiz/data/cir
WORKBENCH_CIRCUS_RELEASE_ROOT=/home/heheboiz/data/cir/workbench/artifacts/circus_release \
  nohup workbench/envs/ctl312h/bin/python workbench/scripts/serve_workbench.py \
    --production-frontend --host 127.0.0.1 --backend-port 8000 --frontend-port 5173 \
    >> /tmp/workbench_serve.log 2>&1 &
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/api/benchmark-runs   # 200
```

`>>` (never `>`) keeps the existing `/tmp/workbench_serve.log` — the supervisor preserves and only
appends, and the current process has it open as fd 1 and 2.

## Tests

```
$ workbench/envs/ctl312h/bin/python -m pytest workbench/tests/test_systemd_user_services.py -q
.........                                                                [100%]
9 passed in 0.19s
```

Seven tests need no manager: they assert `ExecStart`/`WorkingDirectory`/`Environment`/`Restart`/
`RestartSec`/`StartLimit*`/`KillMode`/`TimeoutStopSec` from a fixture `WorkbenchConfig`, that
`ExecStart` contains no shell or wrapper, that the frontend `ExecStart`/proxy target follow
`build_commands`, that `--dev-frontend` switches to `run dev`, that `--dry-run` writes nothing, and
that `install` writes both unit files with no `@PLACEHOLDER@` left. The two manager-dependent
tests are `@pytest.mark.skipif(not Path("/run/systemd/system").exists())`.

## Still unproven / not done

- **The live 8000/5173 cutover was not performed and must not be performed by this worker.** The
  sequence above is written, not run. Consequently the production-rendered units were verified only
  by `--dry-run` and by `systemd-analyze verify` on the rendered test pair (clean, no output);
  they were never started on 8000/5173.
- `systemd-analyze verify` was run on the test-rendered files only. It printed nothing (no warnings,
  no unknown keys).
- A real GPU evaluator job was deliberately not queued (no GPU work allowed). The containment proof
  uses a `setsid` stub, not a live evaluator; `JOBS.shutdown()`'s own group-kill path therefore ran
  with zero live jobs, so the 30 s `TimeoutStopSec` was never actually saturated — the worst case
  (a job that ignores SIGTERM for its full 10 s grace) is bounded by arithmetic, not measured.
- Long-run behaviour: the unit was never left up for hours, so journal growth and the DuckDB index
  thread under repeated restarts are unmeasured. The unit has no log rotation of its own; journald
  is expected to handle it.
- `Restart=always` means an operator cannot leave the unit down by killing the process — only
  `systemctl stop` does that. That is the intended semantic but worth stating: a `kill -9` of the
  backend is *never* a stop.
- Two runs of the isolated backend share `workbench.duckdb` (copied into
  `/tmp/cir_systemd_iso/artifacts`), and the ledger key stayed the operator's, so the log carries
  `skipping aggregate report … disagreeing with its evidence-ledger entry` for reports whose
  contents do not match *this* copy. That is an artifact of the isolated copy, not of the units.

## Historical SIGTERM sender: still unanswered

The units do not answer it. `80-supervision-recovery.md` records the sender of the historical
backend `SIGTERM` deaths as unknown; nothing measured here identifies it. What *is* now known is
that systemd's own machinery never sends a bare SIGTERM to a running main process: it signals only
on `stop`/`restart`/`reset-failed`+`start`, each of which is an operator action and each of which
the journal records (every case above has a `Stopping …`/`Scheduled restart` line naming systemd).
So if the live stack ever moves to these units and a `SIGTERM` death reappears in
`/tmp/workbench_serve.log`, it is *not* the init system — but that inference is for a future
investigation, not evidence about the past. **The historical sender remains unidentified.**
