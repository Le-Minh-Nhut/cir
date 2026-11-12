# ops-gaps lane — Task 4: the historical SIGTERM sender

## Verdict: `UNDETERMINED` — and this pass added two facts and removed nothing

The prior report's row stands. I did not find a culprit, and I did not invent one.

## The question, precisely

Four `serve_workbench.py`-supervised stacks died on 2026-11-11 (02:25:51, ~03:06, 03:53:06, 07:29:33),
each with the identical signature in the committed excerpts
(`raw/service-death-log{,-2,-3,-4}.txt`):

```
[WARN] npm exited with status -15
INFO:     Shutting down
INFO:     Waiting for application shutdown.
INFO:     Application shutdown complete.
INFO:     Finished server process [<pid>]
```

`-15` = SIGTERM. The receive path is known (`serve_workbench.py` `forward_stop`: SIGTERM/SIGHUP →
`os.kill(child.pid, SIGTERM)` per child → `raise_signal(SIGINT)`). What is unknown is **who sent the
SIGTERM to the supervisor**. Previously excluded: `jobs.py`, tests/scripts in-repo, session reaping.

## What I checked, and what each returned

| probe | command | result |
|---|---|---|
| Unit journal for both workbench units, whole retained history | `journalctl --user -u cir-workbench-backend.service --no-pager -o short-iso` | first entry **2026-11-11T23:48:44** — i.e. the units were created *after* all four deaths. **The workbench had no unit then**; it was a hand-launched `serve_workbench.py` under a terminal. So the unit journal cannot name the sender, and its absence is not evidence. |
| User journal **in** each of the four death windows | `journalctl --user --since … --until …` (293 / 323 / 434 / 442 entries per window) | **zero** entries mentioning `cir-workbench`, `serve`, `uvicorn`. The windows are full of `heteroes-worker.service` crash-loop noise and Chromium scopes; the workbench stack left no user-journal record because it was not a unit. |
| Any killing signal in those windows | grep for `kill|signal|term|oom` over the same windows | only `heteroes-worker` and one `Killing process … (nvidia-smi) with signal SIGKILL` from *that* unit. Nothing against the workbench. |
| `systemd-oomd` (the plausible non-human SIGKILL/SIGTERM source) | `systemctl --user is-active systemd-oomd`; grep whole user journal for `oomd`/`oom-kill`/`Killed process` | `inactive`; **0** entries 2026-11-11→11-13. Not the sender, and it would send SIGKILL anyway, not SIGTERM. |
| logind session teardown | `journalctl -u systemd-logind --since 2026-11-11 02:00 --until 08:00` | 4 entries, all "Watching system buttons on /dev/input/eventN". **No** session close, no logout, no `Removed session` in any window. |
| terminal scope closure | grep for `vte-spawn|scope.*(Stopped|Consumed|Failed)` per window | only `app-org.chromium.…scope: Consumed …` lines. No `vte-spawn-…scope` stop in any of the four windows. |
| shell history for a manual `kill` | `~/.bash_history` (2000 lines, 107 724 B) | **0** entries matching `kill`/`pkill`/`systemctl stop`/`serve_workbench`. History is truncated to the last 2000 commands, so this excludes a *recent manual* kill only — it cannot cover 11-11 if the buffer has rolled over. |
| kernel audit (the instrument that would settle it) | `systemctl is-active auditd`; `which auditctl` | `inactive` / **not installed**. No syscall audit trail exists for those windows and none can be created retroactively. |
| upstream `kill` accounting | `journalctl -k` for the window | no OOM, no `Killed process` for a workbench pid; kernel log shows only UFW/NVRM noise. |

## New facts this pass contributes

1. **The retained unit journal covers none of the four deaths** (units began 23:48:44; deaths were
   02:25–07:29). Any future "check the journal" for these four deaths will always return nothing —
   that is a coverage fact, not a negative result about the sender. It is worth recording so the
   question is not re-opened on a false premise.
2. **Both plausible service-level senders are measurably excluded**: `systemd-oomd` is inactive with
   zero entries, and logind logged no session close in any window. That leaves a plain `kill` from a
   process whose credentials were never recorded.

## What would settle it

Exactly one of these, in order of cost:

1. **Re-enable session/boot syscall auditing before the next occurrence**: `auditd` (currently absent)
   with a rule on kill-family syscalls plus `-a always,exit -F arch=b64 -S kill,tkill,tgkill -k sig`.
   Then the audit record names `pid`, `comm`, `exe` and `ppid` of the sender. This is the only
   instrument that can answer "who" after the fact, and it does not exist on this host.
2. **Give the service a decidable identity before it can die unattended**: run the stack as the
   (already-written) `cir-workbench-backend.service` / `cir-workbench-frontend.service` rather than a
   hand-launched supervisor. Then *every* stop — human, logind, oomd, operator — is a journal line
   (`Stopping …`, `Main process exited, code=…, status=…`), which is what the four deaths lack and
   what the current instance has.
3. **Keep the supervisor's own signal record**: `serve_workbench.py` `forward_stop` could log
   `signum` and `os.getpid()` of the caller (via `SO_PEERCRED` only for socket paths; for signals a
   `sigaction` handler can at least record `signum` + timestamp). It cannot name the sender, but it
   would pin the signal and time in the same file the child output lands in.

Absent one of those, this stays `UNDETERMINED`. The four committed excerpts are the complete primary
evidence, and they record the *receiver's* behaviour, never the sender's identity.
