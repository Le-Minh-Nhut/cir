# 82 — Authoritative serial sweep on a stationary tree (`96d86010`)

Status: **RUNNING**. Tree proven stationary before start. Prior run (`c25d67f2`) was killed
at 31% because production mutated mid-run — see `81-sweep-running.md`. This file replaces it
as the authoritative sweep of record.

No source file edited, `dist/` not rebuilt, web service untouched.

## Identity

| Field | Value |
|---|---|
| Branch | `research/cir-failure-workbench` |
| HEAD (full) | `96d86010e85048d82f2dacf26e17d7b7206d31ab` |
| HEAD (sha8) | `96d86010` |
| origin/research/cir-failure-workbench | `96d86010e85048d82f2dacf26e17d7b7206d31ab` |
| HEAD == origin | yes |
| Start timestamp | `2026-11-11T19:38:19+07:00` (12:38:19Z) |
| Sweep PID | `1046852` (setsid, detached) |
| Log path | `/tmp/sweep_96d86010.txt` |
| Basetemp | `/tmp/sweep_basetemp_96d86010` |
| Bundle | `workbench/frontend/dist/assets/index-DYYSN7oG.js` |
| Bundle sha256 | `74e4b4f86ce8b4f5061456138ce1ff79aa084d7662c54c24a123e5bb679d651f` |
| Expected wall time | ~55–58 min |

Command actually run:

```
mkdir -p /tmp/sweep_basetemp_96d86010
setsid nohup workbench/envs/ctl312h/bin/python -m pytest workbench/tests -q \
  -p no:cacheprovider -rfE --tb=short \
  --basetemp=/tmp/sweep_basetemp_96d86010 \
  > /tmp/sweep_96d86010.txt 2>&1 < /dev/null &
```

## Preflight — 8/8 PASS

### 1. `pgrep -af "pytest workbench/tests"` — PASS (empty)

```
(no output)
exit=1
```

### 2. `git status --porcelain` — PASS (only the expected untracked file)

```
?? CLEANUP_REPORT.md.save
```

### 3. HEAD vs origin — PASS (equal)

```
HEAD=96d86010e85048d82f2dacf26e17d7b7206d31ab
ORIGIN=96d86010e85048d82f2dacf26e17d7b7206d31ab
SHORT=96d86010
EQUAL=yes
```

### 4. `git diff --stat -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'` — PASS (empty)

```
(no output)
exit=0
```

No production file differs from HEAD: the tree the tests will import is exactly the
committed tree.

### 5. dist mtime invariant — PASS

Newest frontend source vs the built bundle:

```
2026-11-11 19:35:15.964357463 +0700  workbench/frontend/QueryInspector.tsx   <- newest src
2026-11-11 19:35:53.302376698 +0700  workbench/frontend/dist/assets/index-DYYSN7oG.js

newest src epoch = 1794400515.9643574630
dist js  epoch   = 1794400553.3023766980
```

`1794400515.96 <= 1794400553.30` → source is older than `dist` by ~37 s, so
`test_frontend_staleness.py:245-248` (`mtime(dist) >= mtime(newest ts/tsx)`) holds.

### 6. `ls workbench/frontend/dist/assets/` — PASS (exactly one `.js` + one `.css`)

```
total 412
-rw-rw-r-- 1 heheboiz heheboiz 389252 Nov 11 19:35 index-DYYSN7oG.js
-rw-rw-r-- 1 heheboiz heheboiz  26552 Nov 11 19:35 index-DecQnta9.css

js count = 1
css count = 1

74e4b4f86ce8b4f5061456138ce1ff79aa084d7662c54c24a123e5bb679d651f  workbench/frontend/dist/assets/index-DYYSN7oG.js
```

### 7. Disk / basetemp — PASS

```
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme0n1p5  630G  500G   99G  84% /

ls: cannot access '/tmp/sweep_basetemp_*': No such file or directory
```

99 GiB avail ≥ 45 GiB; target basetemp absent.

### 8. `nvidia-smi --query-compute-apps` — PASS (no compute process)

```
pid, process_name
```

## Liveness evidence

```
$ pgrep -af "pytest workbench/tests"
1046852 workbench/envs/ctl312h/bin/python -m pytest workbench/tests -q -p no:cacheprovider -rfE --tb=short --basetemp=/tmp/sweep_basetemp_96d86010
```

At +83 s (`2026-11-11T12:39:42Z`) the log had grown 49 B → 219 B (0 → 2 lines), progress 4%:

```
........................................................................ [  4%]
...........................................................
```

## MID-RUN STATIONARY RE-PROOF — required, do not skip

A sweep is only authoritative if the tree did not move while it ran. This must be proven
twice more, and both results recorded here:

1. **At ~+25 min** (around `2026-11-11T20:03+07:00` / 13:03Z):
   ```
   git rev-parse HEAD
   git status --porcelain
   git diff --stat 96d86010e85048d82f2dacf26e17d7b7206d31ab -- . \
     ':(exclude)workbench/docs' ':(exclude)workbench/tests'
   git diff --stat -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'
   ls workbench/frontend/dist/assets/ ; sha256sum workbench/frontend/dist/assets/*.js
   ```
2. **At the end**, immediately after the summary line appears — the same four commands.

Both checkpoints must show:

- **source tree** still identical to the swept commit. HEAD itself is allowed to advance
  past `96d86010e85048d82f2dacf26e17d7b7206d31ab`, because committing this very report
  already moved it (to `5455f0e`). The correct test is therefore:
  ```
  git diff --stat 96d86010e85048d82f2dacf26e17d7b7206d31ab -- . \
    ':(exclude)workbench/docs' ':(exclude)workbench/tests'
  ```
  which must stay **empty** — i.e. any new commit is docs-only.
- porcelain shows nothing outside `workbench/docs` (docs-only edits are tolerable;
  anything else is not)
- the `:(exclude)` working-tree diff still empty
- the bundle still `index-DYYSN7oG.js` with sha256 `74e4b4f8…9d651f`

If any checkpoint differs, the run is **not** usable for `TESTED_SOURCE_SHA` and must be
repeated — same rule that killed `c25d67f2` at 31%.

Checkpoint results are appended under "Stationary checkpoints" below.

## Stationary checkpoints

| When | HEAD | porcelain | non-docs diff | bundle | verdict |
|---|---|---|---|---|---|
| start (19:38:19 +07) | `96d86010…31ab` | `?? CLEANUP_REPORT.md.save` | empty | `index-DYYSN7oG.js` / `74e4b4f8…9d651f` | stationary |

Checkpoint rule: the swept tree is `96d86010`. HEAD may advance by docs-only commits
(including this report's own commit `5455f0e`); the authoritative test is
`git diff --stat 96d86010 -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'`
staying empty, plus an unchanged bundle digest.
| ~+25 min | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |
| end | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |

## POST-RUN CHECKLIST — do not execute until the log shows a summary line

Run only when `/tmp/sweep_96d86010.txt` ends with a `... passed ... in NNNN.NNs` summary line
and the pytest PID is gone.

1. **No collection errors:** `grep -c '^ERROR ' /tmp/sweep_96d86010.txt` must be `0`.
2. **No disk-full errors:** `grep -c "Errno 28" /tmp/sweep_96d86010.txt` must be `0`.
3. **Accounting:** from a *separate* `--collect-only` run, confirm
   `passed + failed + skipped == collected`. `collected` must equal the summary total.
4. **Failure-set diff** against baseline
   `workbench/docs/reviews/20261110-e2e-repair/full_suite_68ad75e.txt`
   (that baseline records `18 failed, 2701 passed, 47 skipped` → 2766 tests, 18 `^FAILED ` rows):
   ```
   grep '^FAILED ' baseline.txt            | sort > /tmp/base_failed.txt
   grep '^FAILED ' /tmp/sweep_96d86010.txt | sort > /tmp/new_failed.txt
   diff /tmp/base_failed.txt /tmp/new_failed.txt
   ```
   New rows are regressions; vanished rows need an explanation. Note that a later sweep
   recorded 24 FAILED / 2793 passed — if the count lands there instead, diff against that
   set too and say which baseline applies.
5. **Cleanup:** `rm -rf /tmp/sweep_basetemp_96d86010` (~7–30 GB) after results are captured,
   then confirm with `df -h /` and `ls -d /tmp/sweep_basetemp_*`.
6. **NEVER** grep `'^E '` — `--tb=short` traceback bodies inflate that count into noise.
   Use only `'^ERROR '` and `'^FAILED '` anchors.

## Notes

- Serial run, single process, no xdist — authoritative.
- Web service untouched: supervisor `999204` (`serve_workbench.py --production-frontend`)
  was not restarted, killed, or rebuilt during this step.
- Only file created by this step is this report. `CLEANUP_REPORT.md.save` is pre-existing
  and unmodified.
