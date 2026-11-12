# ops-gaps lane — Task 3 measurements (verbatim)

Control interpreter `workbench/envs/ctl312h/bin/python`. No GPU. Shipped units on :8000/:5173
were **read only** (`systemctl show`, `systemctl cat`) and never started, stopped or signalled.

## Unit properties (quoted, `systemctl --user cat` + `show`)

### `cir-workbench-backend.service`

```
Type=simple
ExecStart=/home/heheboiz/data/cir/workbench/envs/ctl312h/bin/python -m uvicorn workbench.backend.main:app --host 127.0.0.1 --port 8000
WorkingDirectory=/home/heheboiz/data/cir
Restart=always
RestartUSec=2s
KillMode=mixed
KillSignal=15
SendSIGKILL=yes
TimeoutStopUSec=30s          (unit file: TimeoutStopSec=30)
ExecStop=                    (none declared)
```
`StartLimitIntervalSec=60`, `StartLimitBurst=5`.

### `cir-workbench-frontend.service`

```
Type=simple
ExecStart=/home/heheboiz/.nvm/versions/node/v25.9.0/bin/npm run preview -- --host 127.0.0.1 --port 5173 --strictPort
WorkingDirectory=/home/heheboiz/data/cir/workbench/frontend
After=cir-workbench-backend.service
Wants=cir-workbench-backend.service
Restart=always
RestartUSec=2s
KillMode=mixed
KillSignal=15
SendSIGKILL=yes
TimeoutStopUSec=10s          (unit file: TimeoutStopSec=10)
ExecStop=                    (none declared)
```

## Isolated measurement, spare ports, own session (`task3-measure.sh`)

Same argv, same cwd, same `Environment=` as the units; spare ports 8795 (backend) / 8796
(frontend). Signal delivered to the process that the unit's `ExecStart` makes MainPID; "stop_ms"
is signal → main process gone. "restart_same_port" launches the same command again on the SAME
port immediately.

```
== backend :8795 ==
be-term sig=TERM stop_ms=170  children_at_signal=[] children_surviving=[] port_owner_after=none restart_same_port=YES
be-int  sig=INT  stop_ms=168  children_at_signal=[] children_surviving=[] port_owner_after=none restart_same_port=YES
== frontend :8796 ==
fe-term sig=TERM stop_ms=22   children_at_signal=[2388412] children_surviving=[] port_owner_after=2388413 restart_same_port=NO
fe-int  sig=INT  stop_ms=50086 children_at_signal=[2391124] children_surviving=[2391124] port_owner_after=2391125 restart_same_port=NO
== cleanup proof ==
ports 8795/8796 free
no processes on spare ports
```

A second, independent pass (counter-based, `/proc`-derived children, 89 s wall) agreed on the
backend and reproduced the frontend asymmetry, including the `node` grandchild outliving `npm`:

```
frontend-SIGTERM-8792: exit_code=-15 stop_seconds=0.007 children_at_signal=2
                       all_children_died=false children_surviving=[2379269]
                       port_released=false port_release_wait_s=10.0 restart_became_ready=yes*
```
`*` that `restart_became_ready=yes` is a **false positive** of the first harness: the restart's
readiness probe connected to the ORIGINAL node process, which still owned the port. The
counter-based second harness attributes the listening socket to a pid and reports NO.
Raw: `task3-signal-direct.json`.

## Re-measurement after the unit-template fix (2026-11-13, second pass)

The frontend defect above was repaired in the template (`KillMode=control-group`,
`TimeoutStopSec=5`). Re-measured with the same method — isolated instance, spare port 8797, same
argv/cwd/`Environment=`, own session, instance killed afterwards. `mixed` emulates "SIGTERM to
MainPID only"; `control-group` emulates "SIGTERM to every cgroup member at once", which is what
systemd does under that setting:

| KillMode | npm exit | node exit | port free | node alive after | restart on same port |
|---|---|---|---|---|---|
| `mixed` (**old**, `TimeoutStopSec=10`) | 22 ms | **32506 ms** | 66266 ms | no | yes, but only after the cgroup SIGKILL |
| `control-group` (**new**, `TimeoutStopSec=5`) | 22 ms | **24 ms** | **51 ms** | no | **yes, immediately** |

Raw: `fe-mixed mode=mixed npm_ms=22 node_ms=32506 port_free_ms=66266 node_alive_after=NO
restart_same_port=YES` / `fe-group mode=group npm_ms=22 node_ms=24 port_free_ms=51
node_alive_after=NO restart_same_port=YES`.

**The finding, before → after:** under `mixed` the `node` grandchild outlived `npm` by **32.5 s** and
held :5173 until systemd's cgroup SIGKILL — every normal stop burnt the entire 10 s budget and then
some (the measured 66 s port-free includes the emulated timeout wait). Under `control-group` the
whole group dies on the first signal: **npm 22 ms, node 24 ms, port free at 51 ms**, restart
succeeds immediately.

**New `TimeoutStopSec=5` and its margin:** measured normal-case stop is **51 ms**, so the new budget
is a **~98x margin** (was: 0x, because the full 10 s was consumed by design). 5 s still bounds a
wedged `node` (SIGSTOPped or in uninterruptible I/O) to half the old wait. The backend deliberately
keeps `mixed` and `TimeoutStopSec=30` — its job children carry queue state that a direct group signal
would skip, which is why the two units are not homogenised.

Still not proven by any of this: `systemctl stop` on the real shipped unit (cgroup mechanics,
`Restart=always` bookkeeping and timeout enforcement itself remain unit-manager behaviour), and the
backend worst case with a live stubborn job (`GRACE_SECONDS=10` + 5 s + `join(10)` = 25 s vs 30 s)
stays arithmetic.

## What this proves and what it does not

**Proves** (measured, not inferred):

1. Backend: idle stop is **170 ms (SIGTERM)** / **168 ms (SIGINT)**, well inside `TimeoutStopSec=30`.
   No children, port released immediately, restart on the same port succeeds. The shutdown hook
   (`main.py` `_shutdown` → `JOBS.shutdown()`) has no live job to reap here, so this measures the
   cheap path only.
2. Frontend: the `npm` main process exits in **22 ms** on SIGTERM but **`node` (the vite preview
   listener) survives and keeps the port**. That is exactly the pathology the unit's own comment
   names, and exactly what `KillMode=mixed` is for: after `TimeoutStopSec=10` systemd SIGKILLs the
   rest of the cgroup, which is what actually frees :5173.
3. SIGINT to the frontend is NOT equivalent to SIGTERM: `node` ignores it and the port stays held
   for the full 50 s observation window. An operator "Ctrl-C to restart the preview" does not work.

**Does not prove** (stated plainly):

- It does **not** measure `systemctl stop` on the real unit. It measures the same process tree
  receiving the same signal systemd would deliver first (`KillSignal=15` to MainPID). The
  systemd-only behaviour not exercised here is the cgroup SIGKILL phase, cgroup containment of
  the surviving `node`, `Restart=always` bookkeeping, and `TimeoutStopSec` enforcement itself.
  Those are unit-manager mechanics, and the surviving-child result above is the input they act on.
- The **worst case is still not measured**: `TimeoutStopSec=30` is sized for `GRACE_SECONDS=10`
  (SIGTERM ignored by a job) + 5 s SIGKILL escalation + `join(timeout=10.0)` on the queue worker
  (`jobs.py:325,404,433`). No live job existed in the isolated instance, so 10+5+10=25 s remains
  arithmetic, not a measurement. **I could not falsify or confirm the 25 s bound.** It is 5 s
  inside the configured 30 s — thin, but not exceeded on paper.
- ~~The frontend's 10 s `TimeoutStopSec` **is** effectively consumed every stop~~ — **SUPERSEDED by
  the fix below.** That was true under `KillMode=mixed`; the template now uses `control-group` with
  `TimeoutStopSec=5` and the measured stop is 51 ms (~98x margin). Original observation, kept for the
  record: the main process dies in 22 ms and the port stays owned by `node` until the cgroup SIGKILL,
  so the full 10 s was spent — **no margin at all** between the configured timeout and the
  normal-case stop duration. See "Re-measurement after the unit-template fix" above.
