# 81 — Authoritative serial `workbench/tests` sweep (running)

Status: **RUNNING** (started, not finished). No files edited, `dist/` not rebuilt, web service not touched.

## Identity

| Field | Value |
|---|---|
| Branch | `research/cir-failure-workbench` |
| HEAD (full) | `c25d67f2873806be3e28aca69b7a2ad9c9aacb7f` |
| HEAD (sha8) | `c25d67f2` |
| origin/research/cir-failure-workbench | `c25d67f2873806be3e28aca69b7a2ad9c9aacb7f` |
| HEAD == origin | yes |
| Start timestamp | `2026-11-11T12:29:28Z` (19:29:28 +07) |
| Sweep PID | `1021047` (setsid, detached) |
| Log path | `/tmp/sweep_c25d67f2.txt` |
| Basetemp | `/tmp/sweep_basetemp_c25d67f2` |
| Expected wall time | ~55–58 min (prior full sweep: 3318.19 s = 55:18) |

Command actually run:

```
mkdir -p /tmp/sweep_basetemp_c25d67f2
setsid nohup workbench/envs/ctl312h/bin/python -m pytest workbench/tests -q \
  -p no:cacheprovider -rfE --tb=short \
  --basetemp=/tmp/sweep_basetemp_c25d67f2 \
  > /tmp/sweep_c25d67f2.txt 2>&1 < /dev/null &
```

## Preflight (all 7 items PASS)

### 1. `pgrep -af "pytest workbench/tests"` — PASS (empty)

```
(no output)
EXIT=1
```

### 2. `nvidia-smi --query-compute-apps=pid,process_name --format=csv` — PASS (no compute processes)

```
pid, process_name
```

### 3. `df -h /` — PASS (99 GiB avail ≥ 45 GiB)

```
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme0n1p5  630G  499G   99G  84% /
```

### 4. `ls /tmp/sweep_basetemp_*` — PASS (target absent)

```
ls: cannot access '/tmp/sweep_basetemp_*': No such file or directory
EXIT=2
```

### 5. `ls workbench/frontend/dist/assets/*.js` — PASS (exactly one file)

```
workbench/frontend/dist/assets/index-CBR6lHgq.js
COUNT=1
acfb5fbbced77e88b86ee542e5f261c6df71c0e0e8039d753965e1fc873ac21a  workbench/frontend/dist/assets/index-CBR6lHgq.js
```

Bundle: `index-CBR6lHgq.js`
SHA-256: `acfb5fbbced77e88b86ee542e5f261c6df71c0e0e8039d753965e1fc873ac21a`

### 6. `git status --porcelain` — PASS (expected single untracked file)

```
?? CLEANUP_REPORT.md.save
```

### 7. HEAD vs origin — PASS (equal)

```
HEAD   = c25d67f2873806be3e28aca69b7a2ad9c9aacb7f
ORIGIN = c25d67f2873806be3e28aca69b7a2ad9c9aacb7f
SHORT  = c25d67f2
BRANCH = research/cir-failure-workbench
```

## Liveness evidence

```
$ pgrep -af "pytest workbench/tests"
1021047 workbench/envs/ctl312h/bin/python -m pytest workbench/tests -q -p no:cacheprovider -rfE --tb=short --basetemp=/tmp/sweep_basetemp_c25d67f2
```

At +60 s (`2026-11-11T12:30:31Z`) the log had grown to 2 lines / 219 bytes and progress
reached 4%:

```
ssssssssss.s...............sssssssssssssssss......s..................... [  2%]
........................................................................ [  4%]
```

## POST-RUN CHECKLIST — do not execute until the log shows a summary line

Run only when `/tmp/sweep_c25d67f2.txt` contains a trailing
`... passed ... in NNNN.NNs` summary line (i.e. the pytest process is gone).

1. **No collection errors:** `grep -c '^ERROR ' /tmp/sweep_c25d67f2.txt` must be `0`.
2. **No disk-full errors:** `grep -c "Errno 28" /tmp/sweep_c25d67f2.txt` must be `0`.
3. **Accounting:** from a *separate* `--collect-only` run, confirm
   `passed + failed + skipped == collected`. Collected count must match the summary
   totals (prior sweep: 24 failed + 2793 passed + 47 skipped = 2864).
4. **Failure-set diff** against baseline
   `workbench/docs/reviews/20261110-e2e-repair/full_suite_68ad75e.txt`
   (extract `^FAILED ` rows from both, sort, `diff`). Baseline has 24 FAILED rows.
   Any new row is a regression; any vanished row must be explained.
5. **Cleanup:** delete the basetemp afterwards (~30 GB):
   `rm -rf /tmp/sweep_basetemp_c25d67f2` — only after capturing results.
6. **NEVER** grep `'^E '` — `--tb=short` traceback bodies inflate that count and make it
   meaningless. Use `'^ERROR '` and `'^FAILED '` anchors only.

## Notes

- Serial run (`-p no:cacheprovider`, no `-n`/xdist) — authoritative, single process.
- Web service untouched: no restart, no kill, no rebuild of `dist/`.
- Only file created by this step is this report (plus the pre-existing untracked
  `CLEANUP_REPORT.md.save`, which was not modified).

## KẾT THÚC — KHÔNG DÙNG ĐƯỢC

Sweep này bị dừng giữa đường. **Kết quả KHÔNG dùng được** — không được dùng cho
`TESTED_SOURCE_SHA`, không được đem so với baseline.

### Lý do: cây làm việc đã biến đổi giữa lúc chạy (mutating tree)

Production thay đổi khi sweep đang chạy:

- Commit `3ab8d0d` — `fix(frontend): compare table separates peers by composition label (A5)` —
  sửa `workbench/frontend/QueryInspector.tsx`.
- `dist/` đang được rebuild trong lúc sweep chạy. Bằng chứng mtime:

  ```
  2026-11-11 19:35:15 +0700  workbench/frontend/QueryInspector.tsx   (source, đổi giữa run)
  2026-11-11 19:35:53 +0700  workbench/frontend/dist/assets/index-DYYSN7oG.js  (rebuild giữa run)
  ```

- `workbench/tests/test_frontend_staleness.py:245-248` khẳng định
  `mtime(dist) >= mtime(mới nhất của frontend *.ts/*.tsx)`. Nguồn và bundle cùng bị ghi
  trong lúc pytest đang chạy, nên bất biến của cây bị phá tại đúng thời điểm đo.
- Bundle đã đổi tên trong lúc chạy: lúc preflight là `index-CBR6lHgq.js`
  (sha256 `acfb5fbb…ac21a`), sau khi dừng là `index-DYYSN7oG.js`. Đây là bằng chứng trực tiếp
  rằng artefact được test không còn là artefact đã preflight.

Quy tắc của campaign này: **sweep trên cây đang biến đổi thì không đo được gì**
(a sweep on a mutating tree measures nothing). Vì vậy run bị hủy thay vì thu thập.

### Mốc thời gian và tiến độ

| Mục | Giá trị |
|---|---|
| Bắt đầu | `2026-11-11T19:29:28+07:00` (12:29:28Z) |
| Dừng (SIGKILL) | `2026-11-11T19:36:14+07:00` (12:36:14Z) |
| Thời gian chạy | ~6 min 46 s trên dự kiến 55–58 min |
| Tiến độ đạt được | **31%** |
| PID | `1021047` |
| Log giữ lại | `/tmp/sweep_c25d67f2.txt` (13 dòng, 1070 byte) |

Dòng log cuối cùng, nguyên văn (chưa kết thúc bằng newline, pytest bị cắt giữa dòng):

```
.......................................F................................ [ 31%]
............................s.
```

Lưu ý: đã có một `F` ở dòng 31% — nhưng vì cây biến đổi, kể cả failure này cũng không
quy được cho source nào một cách đáng tin.

### Lệnh dừng

```bash
pgrep -af "pytest workbench/tests"     # -> 1021047 ... --basetemp=/tmp/sweep_basetemp_c25d67f2
kill -TERM 1021047                     # graceful, chờ 10 s
kill -KILL 1021047                     # vẫn còn sống sau SIGTERM -> kill cứng
```

Sau đó `pgrep -af "pytest workbench/tests"` trả về rỗng. Không đụng tới web service:
supervisor `999204` (`serve_workbench.py --production-frontend --host 127.0.0.1`) và các
con của nó vẫn chạy bình thường trong suốt quá trình.

### Dọn dẹp

```bash
du -sh /tmp/sweep_basetemp_c25d67f2   # 7.5G
rm -rf /tmp/sweep_basetemp_c25d67f2   # exit=0
df -h /                               # 630G, 500G used, 99G avail, 84%
```

Basetemp đã xóa (`ls /tmp/sweep_basetemp_*` → không tồn tại). Log
`/tmp/sweep_c25d67f2.txt` **được giữ lại** làm bằng chứng, không xóa.

### Việc phải làm tiếp

Phải chạy một **sweep serial mới, sạch**, tại **HEAD cuối cùng**, trên một cây **đứng yên**
(stationary): không commit, không rebuild `dist/`, không sửa `workbench/frontend/**` trong
suốt thời gian chạy. Trước khi start lại phải lặp lại đủ 7 mục preflight, và ghi lại
`sha256` của bundle tại thời điểm start để chứng minh artefact không đổi giữa run.

Checklist post-run ở phần trên vẫn giữ nguyên và chỉ áp dụng cho run mới đó.
