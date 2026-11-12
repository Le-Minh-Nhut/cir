# BÁO CÁO GIAO HÀNG — audit tổng 20261111 (viết 2026-11-12)

Đây là báo cáo **giao hàng**. Mọi con số dưới đây được **chép nguyên** từ một file bằng chứng có đường
dẫn trích ngay trong dòng. Chỗ nào không có số trong file bằng chứng thì ghi `chưa đo`, không suy diễn.

Phần "số liệu sống" (git, systemd, `pgrep`, GPU) được chụp bằng lệnh ghi rõ trong §0; bản chụp
`04:43+07` nằm ở `/tmp/audit-facts/05-live.txt` (ngoài repo, chỉ để đối chiếu). **§0.1 là lần chụp lại
lúc giao báo cáo** (`08:46+07`) và là trạng thái đúng để đọc; §0 là lịch sử.

Bản sửa lần này (commit nối tiếp trên `d0e0eb7`, chỉ chạm file báo cáo này) chỉ đổi **những câu đã sai vì cả hai run Full giờ đều tồn tại**, cộng các
mục mà một commit sau đã vá. Danh sách sửa từng dòng: `raw/verify/102-report-staleness.md`. Nguồn chuẩn
cho hai run Full: `raw/verify/98-tgcir-full-run.md`. Mọi lệnh dùng để kiểm: §9. Số liệu nào **đo lại**
trong lượt này (§0.1, §9) thì ghi rõ "đo lại"; còn lại là chép từ file bằng chứng.

§1-§4 là **bốn phán quyết riêng biệt**, không gộp; §5 tách **đã vá / còn**, §6-§9 là bảng tra (file
bằng chứng, phán quyết, `run_id`, lệnh kiểm).

---

## 0. Số liệu sống tại thời điểm viết (2026-11-12T04:43+07, tức 2026-11-11T21:43Z)

Khối dưới đây là **bản chụp có mốc thời gian `2026-11-12T04:43+07`**, **không** phải trạng thái hiện
tại, và **không** được đọc như cùng một thời điểm với §0.1: hai khối cách nhau vài giờ và nhiều sự kiện.
Hai tiến trình `composition_eval.py` trong khối (`1366356`, `1366364`) **đã thoát**; xem §4.4 (đã sửa)
và `raw/verify/98-tgcir-full-run.md` §1. **Nguồn duy nhất cho trạng thái hiện tại là §0.1** — HEAD
`a0c08c0`, đã `== origin`, backend `1510910`, frontend `1221148`, porcelain một dòng.

```
## git rev-parse HEAD
04f6f4c8147d35f956e43f5a00f04d1028d8252d
## git rev-parse origin/research/cir-failure-workbench
04f6f4c8147d35f956e43f5a00f04d1028d8252d
## git rev-parse --abbrev-ref HEAD
research/cir-failure-workbench

## systemctl --user show -p MainPID -p ActiveEnterTimestamp -p NRestarts
MainPID=1221379
NRestarts=1
ActiveEnterTimestamp=Wed 2026-11-11 23:48:58 +07

MainPID=1221148
NRestarts=0
ActiveEnterTimestamp=Wed 2026-11-11 23:48:44 +07

## pgrep -af composition_eval.py
1366356 bash /home/heheboiz/data/cir/workbench/scripts/gpu_job.sh workbench/envs/ctl312h/bin/python workbench/scripts/composition_eval.py --config /home/heheboiz/cir_rerun_20261109.yaml --models clvc_net --subsets full --topk-depth 100
1366364 workbench/envs/ctl312h/bin/python workbench/scripts/composition_eval.py --config /home/heheboiz/cir_rerun_20261109.yaml --models clvc_net --subsets full --topk-depth 100

## ps -o pid,etime,%cpu,nlwp,stat -p 1366364
1366364  20:23  767  19  Sl

## nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader
NVIDIA GeForce RTX 5070 Ti, 580.173.02, 839 MiB, 16303 MiB
```

Thứ tự hai dòng `systemctl` theo đúng thứ tự tham số trên lệnh: dòng đầu là
`cir-workbench-backend.service`, dòng sau là `cir-workbench-frontend.service`.

`git status --porcelain` **tại `2026-11-12T04:43+07`, TRƯỚC khi ba đường dẫn sau được commit**
(nguyên văn — bản chụp lịch sử, **không** phải trạng thái hiện tại):

```
?? CLEANUP_REPORT.md.save
?? workbench/docs/reviews/20261111-master-audit/raw/verify/91-browser-smoke-after-cutover.md
?? workbench/docs/reviews/20261111-master-audit/raw/verify/97-api-surface-tgcir-run.md
?? workbench/docs/reviews/20261111-master-audit/web_evidence/e2e_post_cutover/
```

Ba đường dẫn `raw/verify/91-…`, `raw/verify/97-…` và `web_evidence/e2e_post_cutover/` đã được commit ở
**`23d847e`** ("evidence: commit the post-cutover browser smoke, the tgcir API surface, and the E2E
screenshots"), nên chúng **không còn** untracked. Trạng thái hiện tại ở §0.1.

### 0.1 Số liệu sống **đo lại** lần cuối (2026-11-12T10:0x+07)

Chụp bằng đúng các lệnh in kèm; không suy diễn từ khối §0 ở trên. **Đây là nguồn duy nhất** cho trạng
thái sống; §9 chỉ lặp lại các giá trị này.

```
$ date -Iseconds
2026-11-12T10:0x:xx+07:00

$ git rev-parse HEAD
a0c08c0e658bb9ece6112352f4b2de90e07d3ad9

$ git rev-parse origin/research/cir-failure-workbench
a0c08c0e658bb9ece6112352f4b2de90e07d3ad9

$ git rev-list --left-right --count origin/research/cir-failure-workbench...HEAD
0       0

$ systemctl --user show -p MainPID -p ActiveEnterTimestamp -p NRestarts \
    cir-workbench-backend.service cir-workbench-frontend.service
MainPID=1510910
NRestarts=0
ActiveEnterTimestamp=Thu 2026-11-12 09:34:21 +07

MainPID=1221148
NRestarts=0
ActiveEnterTimestamp=Wed 2026-11-11 23:48:44 +07
```

`git status --porcelain` **tại thời điểm chụp §0.1** (nguyên văn, khi đó đúng **một** dòng):

```
$ git status --porcelain
?? CLEANUP_REPORT.md.save
```

Dòng đó là file của operator và **cố ý không bao giờ được stage** (`CLEANUP_REPORT.md.save`, không
phải một phần của repo). Mọi đường dẫn bằng chứng từng untracked lúc `04:43` đã vào `23d847e`.

**Lưu ý về một bản chụp:** `git status` là trạng thái **sống**, nên nó đổi khi người khác đang sửa
việc. Ngay sau lần chụp này, hai file frontend (`workbench/frontend/SamplesPage.tsx`,
`workbench/frontend/routes.ts`) xuất hiện ở dạng ` M` — **không** phải thay đổi của báo cáo này và
**không** nằm trong commit nào của nó. Vì vậy khối trên là ảnh chụp có mốc thời gian, không phải một
lời hứa rằng porcelain luôn có đúng một dòng; muốn biết hiện trạng thì chạy lại lệnh.

Đọc theo thứ tự tham số như §0: dòng đầu là backend (**`MainPID=1510910`**, vào `active` lúc
`09:34:21 +07` — đã restart thêm một lần sau bản chụp `1476678` của §0.1 trước), dòng sau là frontend
(**`MainPID=1221148`**, vẫn là tiến trình của cutover `23:48:44 +07`, `NRestarts=0` — unit này **không**
bị chạm suốt chiến dịch). Cả hai `NRestarts=0`.

**HEAD là `a0c08c0e658bb9ece6112352f4b2de90e07d3ad9` và đã `== origin`** (`0 behind / 0 ahead`), tức
công việc của chiến dịch **đã được push** — 36 commit kể từ `82cb1044`. Mọi kết luận trong báo cáo này
nói về cây làm việc tại `a0c08c0`.

Không còn tiến trình composition nào chạy, và GPU trống:

```
$ pgrep -af composition_eval.py
(none)

$ nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader
NVIDIA GeForce RTX 5070 Ti, 580.173.02, 373 MiB, 16303 MiB

$ nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
(none)
```

Lần chạy từng bay ở bản chụp `08:46` (PID wrapper `1477254`, controller `1477261`) **đã xong**: nó là
`da0a2162be0e159a1a456cec`, summary `…da0a2162…/full.json` (17 956 B, mtime `09:15:18.964 +0700`), số
ở **§4.4b**, nguồn chuẩn `raw/verify/104-clvc-full-image-only.md`. Cả hai PID đó đã thoát.

---

## 1. Phục hồi dịch vụ web

**Phán quyết: ĐẠT MỘT PHẦN** — hai unit và ca cutover thật đã đo đầy đủ; ca `systemctl stop` trên
unit thật và worst-case `TimeoutStopSec` thì chưa đo.

### 1.1 Hai unit systemd user

Hai unit `cir-workbench-backend.service` (uvicorn `127.0.0.1:8000`) và
`cir-workbench-frontend.service` (vite preview `127.0.0.1:5173`) được render từ
`workbench/systemd/render_units.py`, không sửa tay; mọi đường dẫn tuyệt đối trong unit do renderer
sinh ra (`raw/verify/86-systemd-user-services.md` §"Backend unit, rendered" và §"Frontend unit,
rendered").

`MainPID` của backend là chính uvicorn, **không có shell/wrapper**:

```
MainPID=1219496
comm=python
argv: …/ctl312h/bin/python -m uvicorn workbench.backend.main:app --host 127.0.0.1 --port 8401
children=
```

(`raw/verify/86-systemd-user-services.md` §"MainPID proof (backend)"). `children=` rỗng nghĩa là
không có process con, nên `MainPID` là server và tín hiệu đi thẳng tới process chạy shutdown hook.

Thuộc tính unit, đọc lại bằng `systemctl --user show` (`raw/verify/86-systemd-user-services.md`
§"Resolved unit properties"):

```
Type=simple
Restart=always
RestartUSec=2s
TimeoutStopUSec=30s
KillMode=mixed
KillSignal=15
StartLimitIntervalUSec=1min
StartLimitBurst=5
```

Tức `Restart=always`, `RestartSec=2`, `StartLimitIntervalSec=60` / `StartLimitBurst=5`,
`KillMode=mixed`, `TimeoutStopSec=30`. Lý do từng tham số (`raw/verify/86-systemd-user-services.md`
§"Numbers and why"): `RestartSec=2` vì socket được nhả ở +150 ms và bản thay thế nghe lại sau ≈2 s;
`StartLimitIntervalSec=60`/`Burst=5` vì đo trên stub unit đạt `failed` sau **10 s** và 5 lần restart
với `Start request repeated too quickly.`; `TimeoutStopSec=30` phủ `GRACE_SECONDS` (10 s) + 5 s
SIGKILL mỗi job group + 10 s worker join.

### 1.2 Ba ca kill và độ trễ phục hồi (đo trên cổng cách ly 8401/5401)

| ca | thao tác | thời gian phục hồi | nguồn |
|---|---|---|---|
| A | `kill -TERM <MainPID>` | **3092 ms** (PID cũ mất ở 151 ms, 51 lần poll) | 86 §"Case A" |
| B | `kill -KILL <MainPID>` | **2881 ms** (PID cũ mất ở 50 ms, 47 lần poll) | 86 §"Case B" |
| C | `systemctl --user stop` | **không tự restart** — giữ `inactive/dead` 10 s, `NRestarts=0`, port trống; `start` đưa lại sau **≈1.3 s** | 86 §"Case C" |

Ca A có ghi chú phương pháp: một vòng poll ngây thơ "chờ `/api/benchmark-runs` trả 200" cho kết quả
dương tính giả, vì instance đang chết vẫn trả 200 trong 9 ms sau SIGTERM (lần thử đầu ghi
`9 ms (1 polls)` với `after=1215527`, chính là PID cũ). Mọi số ở trên chỉ được tính sau khi
`/proc/<pid cũ>` biến mất (`raw/verify/86-systemd-user-services.md` §"Case A").

### 1.3 Cô lập cgroup cho con `setsid` tách phiên

Con stub gọi `os.setsid()` nên trở thành leader của cả session và process group
(`sid == pgid == 1219438`, `Ss`), tức đã rời session của backend (1219268); nhưng
`/proc/1219438/cgroup` và `ControlGroup` của unit vẫn đặt nó **trong** cgroup của unit
(`raw/verify/86-systemd-user-services.md` §"cgroup containment for a detached child"):

```
detached_child=1219438 at 23:46:06.869239732
child_sid=1219438 backend_sid=1219268 child_pgid=1219438
cgroup_members=1219268:python 1219437:python 1219438:python
```

Timeline có thứ tự khi kill backend:

```
sent_SIGTERM_to_1219268   at 23:46:06.946587033
child_absent_at=23:46:07.153723576 (poll 5, 50ms)   child_exists_after=NO
unit stopped: active=activating mainpid=0 at 23:46:07.369683698  cgroup_dir_exists=NO
replacement_ready_at=23:46:10.69443064 mainpid=1219496 bench=200
```

Nghĩa là con tách phiên chết lúc 23:46:07.15, unit dừng xong lúc 23:46:07.37, và bản thay thế không
thể nhận việc trước 23:46:10.69 — không có bản thay thế nào dispatch được khi con cũ còn sống
(`raw/verify/86-systemd-user-services.md` §"cgroup containment"). Giới hạn trung thực: stub này không
đăng ký vào `SingleGpuQueue`, nên nó bị SIGKILL của systemd thu, không phải bởi `JOBS.shutdown()`
(cùng file, cùng mục).

### 1.4 Frontend: số lần restart và vì sao cần `mixed`

Kill process giữ port (node, cháu) → `after MainPID=1216559 ready in **2432 ms**` (PID cũ mất ở
50 ms, 49 lần poll). Kill `MainPID` (npm) → `after MainPID=1216962 ready in **2381 ms**`
(`raw/verify/86-systemd-user-services.md` §"Frontend: restart and the kill mode needed").

`KillMode=mixed` là **bắt buộc**, không phải lựa chọn thẩm mỹ — đo trực tiếp, không giả định:

- với `mixed`: `state=inactive port_listeners=0`, `npm_alive=NO node_alive=NO`, `leftover_node_procs=`;
- với `process` (vá vào, thử, rồi hoàn nguyên): `state=inactive **port_listeners=1**`,
  `leftover_node=YES_1216974`, journal `Unit process 1216974 (node-MainThread) remains running after
  unit stopped.`

Vì `npm` chỉ là `Type=simple` cha của `sh -c vite preview` → `node`, `process` để node giữ port 5401 và
lần start sau chết vì `--strictPort`. Nên `mixed` được giữ cho frontend
(`raw/verify/86-systemd-user-services.md` §"Frontend: restart and the kill mode needed").

### 1.5 Cutover thật (2026-11-11 23:48:4x +07)

Stack 8000/5173 chạy dưới systemd. Rollback script `/tmp/cutover_rollback.sh` (`chmod +x`, `bash -n`
sạch, `md5 04fbd5ecfdd978d4b3996ffc173ddf2c`, đọc lại trước khi chạm supervisor); **rollback không cần
dùng** (`raw/verify/86-systemd-user-services.md` §"Cutover — EXECUTED").

MainPID trước → sau (`raw/verify/86-systemd-user-services.md` §"MainPIDs, before → after"):

| unit | trước cutover | sau cutover |
|---|---|---|
| backend | `999204` (supervisor) → con `999208` | **`1221147`**, rồi `1221379` sau ca kill |
| frontend | con `999209` (npm), node `999221` | **`1221148`** (npm), node `1221161` |

Năm endpoint sau cutover đều 200: `direct_bench`, `direct_fe_root`, `direct_comp`, `proxied_bench`,
`proxied_comp`; bundle phục vụ không đổi (`assets/index-DYYSN7oG.js` trước và sau — tên bundle **lịch
sử** của lần cutover `23:48:4x`; bundle đang phục vụ hiện nay là `assets/index-DDviexaW.js`, xem §9)
(`raw/verify/86-systemd-user-services.md` §"Endpoint verification").

Ca `kill -KILL` trên **unit thật** (`raw/verify/86-systemd-user-services.md` §"Live `kill -KILL`
recovery case on the REAL unit"):

```
LIVE_KILL: MainPID_before=1221147 NRestarts_before=0 at 23:48:56.213893937
MainPID_after=1221379  NRestarts_after=1  active=running
kill (23:48:56.236) → Scheduled restart job 23:48:58.252316 → Started 23:48:58.263926
                    → Started server process 23:48:58.512972 → Uvicorn running 23:48:58.514069
=> ≈ 2.28 s từ kill tới process phục vụ; ≤ 3.0 s tới một 200 từ bản thay thế.
```

Trạng thái **lúc cutover** khớp: backend `MainPID=1221379 NRestarts=1 ActiveEnterTimestamp=Wed
2026-11-11 23:48:58 +07`, frontend `MainPID=1221148 NRestarts=0 …23:48:44 +07` (§0, bản chụp `04:43`).
Trạng thái **sau đó** (bản chụp §0.1 trước): backend đã restart thêm một lần → `MainPID=1476678`,
`NRestarts=0`, `ActiveEnterTimestamp=Thu 2026-11-12 08:39:33 +07`; frontend **không đổi**
(`1221148`, `23:48:44 +07`, `NRestarts=0`) (§0.1). Việc restart đó là của lượt vá cache (`raw/verify/100-…md`
§"Live confirmation after restart"), **không** phải của báo cáo này.
Trạng thái **hiện tại** (§0.1, nguồn duy nhất): backend `MainPID=1510910`, `NRestarts=0`,
`ActiveEnterTimestamp=Thu 2026-11-12 09:34:21 +07`; frontend vẫn `1221148` / `23:48:44 +07` /
`NRestarts=0` — unit này **không** bị chạm suốt chiến dịch.

### 1.6 Chưa đo / còn giữ nguyên

- `systemctl --user stop` trên **unit backend thật**: **chưa đo** — ca "giữ nguyên trạng thái dừng" chỉ
  chứng minh trên bản cách ly (case C). Unit thật mới chứng minh `Restart=always` sau SIGKILL thật
  (`raw/verify/86-systemd-user-services.md` §"Still unproven / not done").
- Worst-case `TimeoutStopSec=30` khi một job sống phớt lờ SIGTERM hết 10 s grace: **chưa đo**, chỉ bị
  chặn bằng số học (cùng mục).
- Tăng trưởng journal và thread DuckDB dưới restart lặp lại: **chưa đo** (unit mới chạy vài phút ở
  thời điểm đo; cùng mục).
- **Người gửi SIGTERM lịch sử vẫn chưa xác định.** `raw/verify/86-systemd-user-services.md`
  §"Historical SIGTERM sender: still unanswered" nói rõ: systemd không bao giờ gửi SIGTERM trần tới
  main process đang chạy, nhưng "that inference is for a future investigation, not evidence about the
  past. **The historical sender remains unidentified.**" `raw/verify/84-evaluator-recovery-and-backoff.md`
  §6 lặp lại cùng kết luận.

---

## 2. Dọn tiến trình con (evaluator cleanup)

**Phán quyết: ĐẠT MỘT PHẦN** — backoff, crash-loop, `jobs.py` và guard GPU đều đã vá và đo, và cache
không khoá của `/api/models` **nay đã vá** (§2.4); phần chưa đóng là guard `gpu_may_be_held` (vẫn từ
chối, đường release là thủ tục — §5.2 mục 9) và các mục `chưa đo` ở §2.5.

### 2.1 Tràn backoff: tái hiện được, đã vá

Trước khi vá, đo trên CPython 3.12.11 (`raw/verify/84-evaluator-recovery-and-backoff.md` §1):

```
restart_delay(1025) RAISED OverflowError: int too large to convert to float
restart_delay(1000000) RAISED OverflowError: int too large to convert to float
```

Nguyên nhân: `2 ** 1024` được tính như `int`, rồi `float.__rmul__` cố chuyển và thất bại; `min()`
không bao giờ chạy (cùng file, §1). Ngưỡng đầu tiên là `failures = 1025`.

Cách vá — clamp **số mũ trước** phép nhân (`workbench/scripts/serve_workbench.py`):

```python
BACKOFF_SATURATING_EXPONENT = next(
    exponent for exponent in range(64) if BACKOFF_BASE * 2 ** exponent >= BACKOFF_CAP)

def restart_delay(failures: int) -> float:
    exponent = min(max(0, failures - 1), BACKOFF_SATURATING_EXPONENT)
    return min(BACKOFF_BASE * 2 ** exponent, BACKOFF_CAP)
```

`BACKOFF_BASE` (0.2) và `BACKOFF_CAP` (8.0) không đổi; số mũ bão hoà suy ra từ chúng,
`BACKOFF_SATURATING_EXPONENT == 6`, không hard-code (`raw/verify/84-…md` §1 "The fix").

Thang đo sau khi vá (`raw/verify/84-…md` §"Measured ladder after the fix"): 0→0.2, 1→0.2, 2→0.4,
3→0.8, 4→1.6, 5→3.2, 6→6.4, 7→8.0, 8→8.0, 9→8.0, 1025→8.0, 10**6→8.0, 10**12→8.0; 1000 lần gọi tại
`failures = 10**6` mất 0.000130 s.

### 2.2 Crash loop khi port bị chiếm

Setup: `python -m http.server 8411` giữ port, launcher thật chạy cùng port trong `/tmp`
(`raw/verify/84-evaluator-recovery-and-backoff.md` §2).

- Đường vào thật từ chối sớm: `[BLOCKED] backend port unavailable: 127.0.0.1:8411`, rc = 2 — đây là
  preflight `port_is_available`, chưa tới `Supervisor`. Vậy crash loop chỉ với tới được trong race
  giữa preflight và `bind` của con.
- Trong race đó (patch `port_is_available` = `True`, mọi thứ khác thật): **6 lần restart trong cửa sổ
  10 s** (`restarts=6, failures=6`), không phải ~50 lần như poll 0.2 s không hãm; khoảng chờ giữa chết
  và start kế tiếp: **0.20, 0.40, 0.80, 1.60, 3.20, 6.40 s** — đúng thang. Con thoát **status 3**
  (mã startup-failure của uvicorn), không phải `signal` (`raw/verify/84-…md` §2(b)).
- Sau khi nhả port: một start, không restart nào (`raw/verify/84-…md` §2(c)).

### 2.3 `jobs.py` — phục hồi lúc khởi động

Năm phát hiện, mỗi cái đo bằng fixture thật (`raw/verify/84-evaluator-recovery-and-backoff.md` §3):

1. **Bản ghi in-flight → `failed`, artifact không bị chạm.** `queue_state.json` được ghi lại thành
   `status: failed`, `error: "workbench restarted while the job was in flight"`; `queue_stdout.log`,
   `queue_stderr.log` và `artifacts/result.json` lồng nhau **byte-identical** (snapshot thư mục so
   trước/sau, chỉ loại `queue_state.json` ở cả hai phía). Recovery không mở cây artifact để ghi.
2. **Bản ghi `completed` không bị chạm.** Recovery trả `[]`; `mtime_ns` và `inode` không đổi.
3. **Thiếu metadata.** Trước khi vá, một bản ghi là JSON list (`[{"job_id": …}]`) hoặc scalar
   (`"hello"`, `7`) làm sập cả lượt quét: `AttributeError: 'list' object has no attribute 'get'`,
   ném ra khỏi `recover_interrupted`; vì `_recover_interrupted_jobs` bọc trong
   `except Exception: pass`, hậu quả ở production là **im lặng**: mọi bản ghi xếp sau bản ghi hỏng bị
   bỏ qua. Guard tối thiểu (`jobs.py:509-515`): `if not isinstance(record, dict) or not
   record.get("job_id"): continue`.
4. **Recovery không bao giờ chạy lại job.** `subprocess.Popen` bị monkeypatch để ném
   `AssertionError`; `recover_interrupted` + `adopt_recovered` trên 4 bản ghi (3 in-flight, 1
   completed) chạy sạch. Job được adopt là terminal (`failed`), nên `_next_job` (chọn
   `status == "queued"`) không bao giờ bốc lại.
5. **Kiểm toán quyền sở hữu của mọi chỗ kill.** Mọi `os.killpg`/`os.kill`/`process.kill()` đều nhận
   group id hoặc `Popen` **của tiến trình con đang sống do chính process này spawn** (bảng đầy đủ
   trong `raw/verify/84-…md` §3 "Finding 5"); `recover_interrupted` **không chứa một lời gọi tín hiệu
   nào**; không có `pkill`, không `killall`, không đọc pid từ file. Chứng minh bằng tiến trình thật
   (`/tmp/foreign_probe2.py`): cả ba sống sót — con do probe spawn (pid 1217175), `sleep 300` tự tạo
   session riêng (pid 1217176), và `http-server` pid 2480984 có sẵn từ trước với `/proc/<pid>/stat`
   state + `starttime` y nguyên (`S` / `26970850`).

Guard `gpu_may_be_held` (`raw/verify/84-…md` §3 "Finding 6"): bản ghi recovery **không chứa pid và
pgid**, và job chạy `start_new_session=True`, nên sau một cú kill cứng của backend, con evaluator của
operator có thể vẫn giữ GPU vô thời hạn. Vì không có id nào chứng minh được là của job đã chết, guard
**từ chối dispatch** thay vì kill:

```
dispatch while orphan recorded -> blocked | a job recovered from a crash may still hold the GPU (job-foreign); release the device and restart the workbench before dispatching
job command ran? False
```

Cờ này không tự xoá; xác nhận thiết bị trống là việc của operator (`nvidia-smi`), sau đó restart.

### 2.4 Env-cache: backend **không** bị cũ (đã đo), và `/api/models` **đã vá** (trước đó thì có)

`raw/verify/94-backend-env-cache-restart.md` §Verdict: cơ chế có thật nhưng **hậu quả không xảy ra**,
và **không restart** (restart trên tiền đề sai sẽ là gián đoạn không cần thiết cho unit đang chạy).

Lý do: `file_settings()` là `@lru_cache(maxsize=1)` nhưng được nạp **lazy ở lần dùng đầu**, không phải
lúc import. File `workbench.env` sửa lúc `00:11:54`, process start lúc `23:48:58`, và request
`/api/models` **đầu tiên** của process là `02:43:37` — hơn hai giờ sau khi file được sửa — nên lần đọc
đầu tiên đã thấy cả hai dòng mới (`raw/verify/94-…md` §Verdict). Đo đối chứng: (c) API sống và (d)
process mới **khớp từng blocker**, cả hai đều resolve interpreter; bản mô phỏng stale trong process
ném đi mới ra dạng `execution environment not verified: WORKBENCH_CLVC_PYTHON=None
[UNCONFIGURED/UNCONFIGURED] (WORKBENCH_CLVC_PYTHON not set)` — API sống **không** trả dạng đó
(`raw/verify/94-…md` §(c) vs (d), §"Counterfactual").

**`/api/models` từng có cache module không khoá — nay đã vá.** Bản cũ: `main._MODELS_CACHE` là list mức
module, không có key nào (`main.py:170-176`), nên `GET /api/models` phục vụ tài liệu **trước khi sửa**
suốt đời process; đo được listing vẫn báo file ResNet-50 của `clvc_net` là missing dù file đã có, trong
khi `POST /api/jobs` (tính lại `preflight_blockers` mỗi request) **đã** bỏ blocker đó
(`raw/verify/95-external-asset-unblock.md` §"Did the API pick up the registry edit without a restart?").

Bản hiện tại key theo `_models_state_stamp(registry, config)` (`main.py:240`): 73 entry, **chỉ**
`stat`/`setting` (không hash, không probe), gồm thống kê document registry/auxiliary/preparation-contract,
file env của operator, mọi setting mà registry nêu tên, mọi đường dẫn checkpoint (và member nếu là
bundle), mọi đường dẫn external asset bắt buộc, và thư mục checkout của từng model. Chi phí stamp
**0.016 s** (warm 0.006 s); `/api/models` **lạnh 98.43 s** → **67.28 s** trên process sống, **0.026 s /
0.017 s** khi ấm (`raw/verify/100-api-cache-and-ui-blocked.md` §2, §"Live confirmation after restart"
mục 1). Badge đổi **trong cùng process** theo cả hai chiều (thiếu → sai checksum → thiếu), và gate
`POST /api/jobs` **vẫn** 409 — cache không được "vá" bằng cách nới lỏng từ chối (cùng file §2). Quy tắc
vận hành còn lại: `workbench.env` sửa thì vẫn nên restart backend, không phải vì cache mà vì
`file_settings()` (`raw/verify/94-…md` §"Why an edit to `workbench.env` still requires a restart").

### 2.5 Chưa đo

- Có orphan evaluator thật nào từng sống sót qua một cú crash backend thật hay không: **chưa đo** —
  guard là phòng ngừa, chưa quan sát được orphan cụ thể trên host này (`raw/verify/84-…md` §6).
- Crash loop thật trong production có từng chạm `failures ≥ 1025` hay không: **chưa đo** — không tìm
  thấy log nào như vậy; đây là khuyết điểm tiềm ẩn đã chứng minh của hàm, không phải sự cố quá khứ đã
  chứng minh (`raw/verify/84-…md` §6).

---

## 3. Đối chiếu với upstream (native parity)

**Phán quyết: ĐẠT MỘT PHẦN** — parity ở mức **mẫu 5 query / 20 ảnh mỗi category** đạt trên mọi trường
đo; **không** phải chứng minh mức metric, và **một chứng minh parity ở quy mô Full KHÔNG tồn tại**.

Hai điều phải nói thẳng, vì bản trước để người đọc tự suy:

1. **Đây là mẫu, không phải Full.** Mọi bảng ở §3 đo trên **5 query × 20 ảnh gallery**, một category mỗi
   lần. Chưa có phép so native↔composition nào chạy trên 6 003 query hay gallery 74 381 ảnh.
2. **Con số Full mới của composition KHÔNG phải chứng minh parity, và không bao giờ được trình bày như
   vậy.** §4.1b có `tgcir` composition R@10 `23.33833083458271`; đó là **một** protocol đo trên **một**
   đường (composition). Nó không so với bất kỳ lần chạy native nào trên cùng input, nên nó **không** nói
   gì về việc hai đường có khớp nhau ở quy mô Full. Một Full run chỉ chứng minh composition **chạy
   được** và **cho ra số**; parity vẫn chỉ có ở mức mẫu.
3. **Native ≠ composition.** Native macro R@10 **49.7208** (`raw/tgcir_run3cat_gpu.json`) và composition
   **23.3383** là **hai protocol khác nhau** — khác câu query (release sentence đơn vs hai caption train
   join), khác candidate pool (toàn bộ gallery release vs split theo category) — nên **không** được đặt
   chung một cột, không được trừ cho nhau, và không được đọc thành "composition kém hơn một nửa"
   (`raw/verify/98-tgcir-full-run.md` §"Why the CIRCUS composition number is not the native number").

### 3.1 CLVC-Net: từng model, từng trường

Phương pháp: hai đường nhận input byte-identical — mini FashionIQ 3 file mỗi category với symlink tới
ảnh thật, nên không ảnh nào bị encode lại; phía native là chính evaluator đã pin
(`datasets.FashionIQ(gallery_all=True)` + `test.test(...)` qua `workbench/replay/clvc_net.py`), phía
composition là `_run(adapter, ...)` trên worker thường trú. Cả hai ma trận nhận **cùng** chính sách
reference (hàng reference bị mask `-10e10`, đúng như upstream làm) trước khi tính hạng
(`raw/verify/87-clvc-composition.md` §2).

`max |Δ|` theo từng trường (float32, 5 query × gallery 20) — `raw/verify/87-clvc-composition.md` §2:

| so sánh | dress | shirt | toptee |
|---|---|---|---|
| nửa local 1024-d | **0.0** | **0.0** | **0.0** |
| nửa global 1024-d | **0.0** | **0.0** | **0.0** |
| hàng query ghép 2048-d | **0.0** | **0.0** | **0.0** |
| hàng target ghép 2048-d | **0.0** | **0.0** | **0.0** |
| ma trận score đã mask | **8.94e-08** | **1.19e-07** | **1.79e-07** |

Bốn trong năm trường **đúng bằng 0**; trường score không bit-identical vì là chỗ duy nhất hai đường
dùng số học khác nhau (upstream chia từng hàng cho norm L2 của nó rồi dot, `test.py:122-129`); dư là
1 ULP của giá trị gần 1 (`raw/verify/87-…md` §2).

Vector hạng **giống hệt** theo từng category, và cả danh sách xếp hạng đầy đủ (vị trí 0-9 của
`argsort(-row)`) cũng giống, không chỉ hạng metric (`raw/verify/87-…md` §2 "Rank vector"):

| category | native ranks | composition ranks | giống |
|---|---|---|---|
| dress | `[1, 1, 10, 1, 1]` | `[1, 1, 10, 1, 1]` | **có** |
| shirt | `[1, 2, 2, 1, 2]` | `[1, 2, 2, 1, 2]` | **có** |
| toptee | `[1, 2, 1, 1, 1]` | `[1, 2, 1, 1, 1]` | **có** |

Đồng nhất thức tương đương score, đo trên embedding thật: `score(row_q, row_g)` vs
`(dot_local + dot_global) / 2` — `max abs difference` **1.1920929e-07** ở cả ba category; norm L2 hàng
query là `1.4142135` (tức `sqrt(2)` tới độ chính xác float32), đúng như đồng nhất thức đòi hỏi. Đây là
**1 ULP**, không phải bitwise; docstring nói "exactly" nên được ghi lại là "float32 identity to 1 ULP,
not a bitwise one" (`raw/verify/87-…md` §"Score-equivalence identity").

### 3.2 Batch 1 vs batch N: **không** giống nhau, và đó là hành vi của chính upstream

Đây là chỗ kỳ vọng của brief và phép đo lệch nhau, nên ghi đúng như đo
(`raw/verify/87-clvc-composition.md` §3b):

| đại lượng | giá trị |
|---|---|
| hàng query, batch 20 vs batch 1 | **không bằng**, max abs Δ **1.4638714492321014e-02** |
| hàng target, batch 20 vs batch 1 | **không bằng**, max abs Δ **6.810575723648071e-05** |
| hàng query, cùng batch size lặp lại | **0.0** (tất định) |
| hàng query, edit **cùng độ dài**, batch N vs 1 | max abs Δ **8.05e-05** |
| **hàng query native `test.py`, batch N vs 1** | max abs Δ **1.4638714492321014e-02** — giống hệt worker |
| **hàng target native `test.py`, batch N vs 1** | max abs Δ **6.810575723648071e-05** — giống hệt worker |

Hai nguyên nhân, cả hai thuộc upstream, cả hai **tái hiện chính xác** bằng evaluator đã pin trên cùng
input (`raw/verify/87-…md` §3b):

1. **Padding văn bản là một phần của input.** `TextLSTMModel.forward_encoded_texts` dựng tensor
   `(max_len_in_batch, N)` (`text_model.py:96-98`), nên số cột pad của một edit phụ thuộc các edit
   *khác* trong batch; `local_conv`/`global_conv` cộng theo trục đó. Năm edit dress dài 13, 14, 16, 14,
   14 token, nên batch 1 vs batch 5 đổi số pad của bốn hàng. Làm mọi edit cùng độ dài thu Δ từ
   1.46e-02 xuống 8.05e-05 — đây là phép cô lập nguyên nhân này.
2. **cuDNN chọn thuật toán theo shape.** Phần còn lại (≈1e-5..1e-4) thấy được trên hàng **chỉ ảnh**,
   nơi không có văn bản nào.

Hệ quả được nói thẳng: **hàng feature CLVC-Net phụ thuộc batch size, và hàng native của nó cũng vậy**,
nên batch size là một phần của run identity cho model này. Trong mẫu đo, phán quyết metric không đổi
(hạng `[1, 1, 10, 1, 1]` / `[1, 2, 2, 1, 2]` / `[1, 2, 1, 1, 1]` giống ở cả batch 1 và 32), nhưng
top-10 list thì **không** giống (vị trí 6-9 đảo; toptee 4-9 đảo)
(`raw/verify/87-…md` §3b, bảng "category | ranks @ batch 1 | ranks @ batch 32").

### 3.3 TG-CIR: 12 mảng bit-identical

Cùng input, cùng checkpoint, cùng transform, reference được mask ở cả hai phía; tolerance khai báo
**trước** khi đo: target `atol=0, rtol=0`, PASS ≤ 1e-6 (`raw/verify/88-tgcir-composition.md` §2).

| category | query rows max |Δ| | target rows max |Δ| | score vector max |Δ| | masked score max |Δ| |
|---|---|---|---|---|---|
| dress | **0.0** | **0.0** | **0.0** | **0.0** |
| shirt | **0.0** | **0.0** | **0.0** | **0.0** |
| toptee | **0.0** | **0.0** | **0.0** | **0.0** |

Bit-identical trên **cả 12 mảng** (15 hàng query + 60 hàng target). Vector hạng giống hệt phía native
trên cả 15 query (`rank_vectors_equal: true`), `Adapter.rank()` bằng `_ranks_from_scores` áp lên ma
trận score native, và thứ tự candidate đầy đủ 20 mức giống nhau ở mọi category
(`raw/verify/88-…md` §3).

Batch 1 vs batch N (5 query dress, một hàng mỗi lần gọi vs một lần gọi 5 hàng, cùng worker): max |Δ|
mỗi hàng `2.898e-06, 1.611e-06, 2.027e-06, 1.714e-06, 1.211e-06`, tổng **2.898e-06**; thứ tự hàng
giống (`order_identical: true`), hạng giống (`1,1,10,1,1` cả hai). Đây là nondeterminism theo shape
của cuDNN, không phải khác biệt plumbing — cùng ngưỡng xuất hiện giữa hai shape batch trên đường
native (`raw/verify/88-…md` §5).

### 3.4 Sửa caption đúng **một lần** (đo bằng digest)

Hợp đồng: `texts` đưa cho `encode_queries` là caption **RAW** ở dạng protocol join
(`cap0 + " and " + cap1`); adapter chuyển nguyên; **worker** áp correction đã pin đúng một lần ngay
trước forward pass (`raw/verify/88-…md` §6).

Đo qua worker thật, cùng query `7:1` (`raw/verify/96-staged-path-identity.md` §3c):

```
multimodal            corrected_first='is shiny and silver with shorter sleeves and fit and flare'
                      digest=3174ba7f60c00d56554a629f80c34599f39a0a768ead99cfa27ee06376f4dbd1
text_only_black_image corrected_first='is shiny and silver with shorter sleeves and fit and flare'
                      digest=3174ba7f60c00d56554a629f80c34599f39a0a768ead99cfa27ee06376f4dbd1
image_only_empty_text corrected_first=''
                      digest=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
```

Hai điều kiện không rỗng chia cùng digest (cùng text, chỉ khác pixel); điều kiện rỗng có
`corrected_first == ''` và digest khác — text rỗng tới model **như rỗng**, đúng một lần. Bằng chứng
số học về "đúng một lần" trên embedding thật (`toptee/B00C9NQNSY`, hàng val không idempotent duy nhất,
`raw/verify/88-…md` §6):

```
native(pinned_corrected) vs adapter(raw_in)      max |Δ| = 0.0        ← đúng một lần
native(raw)              vs adapter(raw_in)      max |Δ| = 0.04678698629140854
adapter(raw_in)          vs adapter(corrected_in) max |Δ| = 0.021270141005516052
adapter(corrected_in)    vs adapter(twice_in)     max |Δ| = 0.0        ← lần thứ hai là no-op
```

Áp lần hai sẽ thấy được (`0.0213`), nên phép đo này không phải tautology. Sửa cả chuỗi đã join khác
**3 trên 6016** hàng val (toàn bộ có caption đầu rỗng), nên nó không phải hàm tương đương và không được
dùng cho hàng annotation (`raw/verify/88-…md` §6).

### 3.5 Không kế thừa mask reference — cố ý

CIRCUS **không** kế thừa `test.py:63-67` (mask reference); đây là khác biệt có tên, không phải lỗi
(`raw/verify/88-tgcir-composition.md` §4 mục 1). Đo được: không hạng nào đổi trên mẫu này (reference
đã là top-1 thì vẫn ở hạng 1; reference không phải top-1 thì nằm dưới một candidate *khác*, nên bỏ nó
không đổi `1 + #greater`). Cái **đổi** là cửa sổ candidate: `reference_is_top1_when_unmasked` đúng với
**6 trên 15** query, nên với những query đó reference chiếm một slot top-1 mà CIRCUS giao cho một
candidate thật — đúng chính sách `reference_eligible: true` đã ghi, đo chứ không phải khẳng định.

### 3.6 Phạm vi — nói rõ

Mọi số ở §3 là **parity 5 query / 20 ảnh mỗi category** — **một mẫu**, không phải chứng minh mức
metric, và **không** có parity quy mô Full. `raw/verify/87-clvc-composition.md` §"Còn chặn" mục 1 nói
đúng ở phạm vi của tài liệu đó: "No numbers here are a metric: every measurement is a 5-query, 20-image
parity sample …". `raw/verify/88-tgcir-composition.md` §"Còn chặn" mục 4 nói cùng ý cho TG-CIR.

Phần **đã đổi** so với bản trước: câu "the per-condition aggregate and R@1/10/50 for `clvc_net` remain
unmeasured" **không còn đúng ở mức run** — CLVC Full đã có R@10/R@50 theo điều kiện (§4.4) và R@1 suy ra
từ ranks (`raw/verify/101-metrics-r1-and-fallback.md` §3b). Cái **vẫn** chưa có là **parity** ở quy mô
đó: một phép so native↔composition trên cùng 6 003 query / 74 381 ảnh. Con số composition Full **không**
thay thế được phép so đó.

Số native của campaign (`raw/tgcir_run3cat_gpu.json`: macro R@10 49.7208 / R@50 71.1023 / mean 60.4115)
**không** bị công việc này chạm tới và vẫn là mục tiêu so sánh cho một phép parity Full sau này
(`raw/verify/89-…md` §"Còn chặn" mục 1). Đặt nó cạnh `23.3383` của composition là **sai protocol**
(§3 đầu mục, `raw/verify/98-…md`).

---

## 4. Kết quả composition trên Full

**Phán quyết: ĐẠT MỘT PHẦN** — **cả hai** run Full giờ đều tồn tại. TG-CIR Full
`7b1fa85007bffdab931f7f19` là **`COMPLETED`** trên **6003 query** với gap ba điều kiện
**`0.43853620852911723`**; CLVC Full `eab10106b2df8e97fb7a5ce5` (lần chạy `-retry-2`) là **`PARTIAL`**
với lý do **đã khai báo** (`image_only_empty_text` là ablation `ABLATION_UNSUPPORTED`, cộng 119 query
nhập nhằng annotation như bản release gốc giao), nên `composition_gap_ndcg` của CLVC là `null` —
`raw/verify/98-tgcir-full-run.md` §1-§3.

Một run CLVC **mới hơn** dưới profile khai báo (`--conditions
multimodal,text_only_black_image,clvc_image_encoder_only`) **đã xong** và **đã có số**:
`da0a2162be0e159a1a456cec`, `PARTIAL`, 6001/6003 chấm được, gap ba điều kiện
**`0.33061712371309204`** trên cohort 6001 — gap đầu tiên của CLVC tính được; trên cohort
caption-free 5884 (không có edit text nào chạm routing) con số là **`0.33400015944039774`** (§4.4b,
`raw/verify/104-clvc-full-image-only.md`). Nhưng điều kiện ảnh của nó là
`clvc_image_encoder_only`, **không phải** `image_only_empty_text` của protocol, nên đây là gap **biến
thể** — và **không** được đặt chung cột với `0.43853620852911723` của TG-CIR (§4.2, §4.4b).

### 4.1 TG-CIR: smoke 5 query **và** một Full run `COMPLETED` 6003 query

Có **hai** run, không phải một. Full: `7b1fa85007bffdab931f7f19` (§4.1b). Smoke:
`1eee4c0adbc93f7708205a5e` — vẫn là smoke, và không phải tất cả những gì đang có.

Run `1eee4c0adbc93f7708205a5e`, `--subsets full --smoke 5 --topk-depth 100`
(`raw/verify/97-api-surface-tgcir-run.md` §1; `raw/verify/96-staged-path-and-tgcir-worker-exit.md` §4).

Cả ba điều kiện `COMPLETED`, `unscored_count` 0, `fallback_used` false,
`category_scope_failure_count` 0 — và `topk_export` đầy đủ độ sâu 100:
`{"depth": 100, "missing_query_count": 0, "partial": false, "status": "TOPK_EXPORT_COMPLETE"}`
(`raw/verify/97-…md` §1, §1a).

Metric của **smoke 5 query** này (`raw/verify/97-…md` §1, bảng "Per-condition metrics"):

| condition | R@10 | R@50 | mrr | ndcg_full | scored/query |
|---|---|---|---|---|---|
| `multimodal` | 20.0 | 40.0 | 0.044481057352368764 | 0.18061210310965697 | 5/5 |
| `text_only_black_image` | 0.0 | 0.0 | 0.002228127690261614 | 0.10082312072434303 | 5/5 |
| `image_only_empty_text` | 0.0 | 0.0 | 0.00019029794641650075 | 0.07693918292263288 | 5/5 |

`composition_gap_ndcg` **0.4417698538002782**, `gap_cohort_count` 5, `gap_unavailable_reason` `null`;
`shortcut_outcomes` `{"both": 0, "composition_required": 1, "image_only": 0, "neither_scored": 0,
"text_only": 0, "unresolved": 4}` (`raw/verify/97-…md` §1).

`target_rank` của query `7:1` theo từng điều kiện: **23** (multimodal) / **207**
(text_only_black_image) / **3697** (image_only_empty_text), JSONL = API ở cả ba
(`raw/verify/97-…md` §3, §4). R@10/R@50 ở trên **tái tính được** từ ranks artifact
`[23, 6, 14404, 91, 832]` / `[207, 249, 31976, 450, 24900]` / `[3697, 2469, 34272, 66684, 4314]`
(`raw/verify/97-…md` §1, "Cross-check").

**R@1 không có trên artifact của smoke này — phạm vi: chỉ `1eee4c0a`.** `metrics.<cond>.recall_at_1` là
`<ABSENT>` trên artifact **và** API của `1eee4c0a`, và `derived_metrics.recall_at_1.status` là
**`MISSING`** (không có record suy ra cho run này), nên không có R@1 nào được phục vụ **cho nó**
(`raw/verify/97-…md` §1, "R@1, stated as fact"; đo lại qua
`GET /api/composition/runs/1eee4c0adbc93f7708205a5e?subset=full`, 2026-11-12T08:5x).

Câu **"R@1 không tồn tại ở cả hai phía"** chỉ còn đúng cho **ba** run: smoke `1eee4c0a` và **hai** run
Full cũ `7b1fa850` (tgcir) / `eab10106…-retry-2` (clvc) — cả ba đều **không** mang key này trong
`full.json` của chính chúng; R@1 của hai run Full cũ chỉ tồn tại ở file derived
(`raw/verify/101-metrics-r1-and-fallback.md` §1, §7). Nó **không** còn đúng cho run
`da0a2162be0e159a1a456cec`, nơi `metrics.multimodal.recall_at_1 = 1.3997667055490752` **nằm ngay trong
`full.json`** (§4.4b) — run đó được chấm **sau** khi `metrics.py` có `recall_at_1`
(`workbench/composition/metrics.py:48`, công thức `100.0 * sum(hit_at_k(rank, 1) …) / scored_count` tại
`:77`). Hiện trạng đếm được: **1/36** file `summary/*/full.json` mang key (§9). Vì vậy R@1 **không bao
giờ** được đọc thành 0, và mỗi lần trích phải nói nó đến từ `full.json` hay từ file derived.

**Nói thẳng: đây là smoke 5 query, không phải Full run, và recall của 5 query không phải metric công
bố được.** Chính payload mang marker `smoke.limited: true`, `smoke.query_limit: 5`,
`smoke.label: "SMOKE QUERY LIMIT; full released gallery preserved"` trong khi `gallery_count` vẫn là
gallery release đầy đủ `74381` (`raw/verify/97-…md` §2). Trong 60 dòng phục vụ với `include_smoke=true`,
**10** dòng mang `smoke.limited: true` và **50** dòng không; không có `include_smoke` thì run này
**vắng mặt** khỏi 50 dòng (`raw/verify/97-…md` §2).

### 4.1b TG-CIR Full: `7b1fa85007bffdab931f7f19`, `COMPLETED`, 6003 query

Summary: `workbench/artifacts/composition/summary/tgcir_tgcir_archive_unresolved_7b1fa85007bffdab931f7f19/full.json`
(**109 793 B**, mtime `2026-11-12 05:16:29.257 +0700`; đọc lại nguyên file tại HEAD `a0c08c0`).

| trường | giá trị |
|---|---|
| `execution_status` | **`COMPLETED`** |
| `query_count` | **6003** (`sf` 4069, `validated` 586) |
| `smoke` | `{"label": null, "limited": false, "query_limit": null}` — Full thật, không `--smoke` |
| `gallery_count` | 74381 (`gallery_identity_sha256` `1db68c59d96f014cdbecf936e108653de98abef0bd9a98a8abc2fe08b5dec4b4`, `hash_verification: matched`) |
| `model_id` / `checkpoint_id` | `tgcir` / `tgcir_archive_unresolved` |
| wall time | **556.9 s (9 min 17 s)** — log birth `05:07:12.401` → summary `05:16:29.257` |
| ảnh đã encode | **92 390 — ĐÚNG** (74 381 gallery + 18 009 dòng query, một gallery `global` dùng chung; xác nhận từ sidecar `gallery_global.npy` `[74381, 512]`, một file duy nhất) → **0.00603 s/ảnh**, 165.9 ảnh/s. Ở mọi chỗ: mẫu số là **lượt encode ảnh**, không phải số ảnh gallery (`104` §0.2a) |
| GPU | RTX 5070 Ti, `tgcir39-la` (Python 3.9.25, torch 2.7.1+cu128); 22 mẫu `nvidia-smi`: util **97-100 %**, memory.used **2172 MiB** (min = max), power **235.02-260.04 W** |

Ba điều kiện đều **`COMPLETED`, `unscored_count: 0`, `fallback_used: false`,
`category_scope_failure_count: 0`** (`raw/verify/98-tgcir-full-run.md` §2):

| condition | R@10 | R@50 | mrr | ndcg_full | scored |
|---|---|---|---|---|---|
| `multimodal` | 23.33833083458271 | 42.11227719473597 | 0.12098114437415375 | 0.24702051966534347 | 6003/6003 |
| `text_only_black_image` | 6.546726636681659 | 13.65983674829252 | 0.03674930344151411 | 0.1386930775424115 | 6003/6003 |
| `image_only_empty_text` | 2.798600699650175 | 7.163085124104614 | 0.010877930517005432 | 0.10452130761687346 | 6003/6003 |

Gap **ba điều kiện có thật** cho model này: `composition_gap_ndcg` **`0.43853620852911723`**,
`mrr_gap` `0.08423184093263963`, `gap_cohort_count` **6003**, `gap_unavailable_reason` `null`. Cơ chế
(`metrics.py:126-148`): cả ba điều kiện đều có hạng cho mọi qid nên `common` = toàn bộ 6003, và
`gap = 1 - max(gap_text, gap_image) / gap_mm` = `1 - 0.1386930775/0.2470205197` = **0.4385362085**,
tái lập tới 1e-12 (`raw/verify/98-…md` §2). `shortcut_outcomes`:
`{"both": 9, "composition_required": 956, "image_only": 159, "neither_scored": 0, "text_only": 384,
"unresolved": 4495}`. `topk_export`: `TOPK_EXPORT_COMPLETE`, depth **100**, ba điều kiện,
`missing_query_count` **0**, `partial: false`.

R@1 của run này **có** — nhưng ở **file suy ra**, không ở `full.json`: `multimodal`
**6.3135099117108116**, `text_only_black_image` **1.949025487256372**, `image_only_empty_text`
**0.08329168748958854** (`workbench/artifacts/composition/derived/recall_at_1/clvc_net_…-retry-2+tgcir_7b1fa85007bffdab931f7f19_full.json`;
`raw/verify/101-metrics-r1-and-fallback.md` §3a). Ba số này **không** nằm trong `full.json` và chỉ được
phục vụ qua overlay `derived_metric` (`raw/verify/100-api-cache-and-ui-blocked.md` §5, §"Live
confirmation after restart" mục 2: `derived_metrics.recall_at_1.status = "VERIFIED"`).

**Composition không phải native.** R@10 `23.33833083458271` ở đây là protocol composition (xếp hạng
từng query trên toàn bộ gallery release 74 381 ảnh, dùng câu release đơn), còn **49.7208** là native
macro R@10 (`raw/tgcir_run3cat_gpu.json`, hai caption train join lại, gallery chia theo category). Hai
số **không** so sánh được và **không** được đặt chung một cột (`raw/verify/98-…md` §"Why the CIRCUS
composition number is not the native number").

### 4.2 CLVC-Net: `ABLATION_UNSUPPORTED` cho `image_only_empty_text`

Điều kiện `image_only_empty_text` bị **khai báo không hỗ trợ** chứ không chạy rồi fail từng query. Lý do
là thuộc tính của text tower: `TextLSTMModel.forward` tokenise về 0 token rồi index `x[0][0]`
(`text_model.py:84-86`), nên ném `IndexError` cho mọi hàng — điều kiện này **không biểu diễn được**
trong text tower của model này (`raw/verify/90a-condition-degradation-fix.md` §2; lý do nguyên văn nằm
trong `condition_status` của summary, §4.3 dưới).

`mode_supported` và `status().per_mode` dùng **cùng một vị từ** nên không thể mâu thuẫn; đo trên
adapter dựng với config của operator (`raw/verify/90a-…md` §2):

```
multimodal              -> (True, None)
text_only_black_image   -> (True, None)
image_only_empty_text   -> (False, "CLVC-Net cannot encode an empty edit: TextLSTMModel.forward … (text_model.py:84-86) …")
```

Runner **hỏi** adapter chứ không chạy mọi điều kiện: `active_conditions` lọc theo `supported`
(`runner.py:1920-1921`), điều kiện không hỗ trợ được ghi `ABLATION_UNSUPPORTED` kèm lý do của adapter
(`runner.py:1923-1925`), gallery chỉ encode `if active_conditions:` (`runner.py:1928`), và bộ ghi Top-K
cũng lọc theo cùng cách (`export_conditions`, `runner.py:1835-1837`) nên điều kiện không hỗ trợ không
bị tính là thiếu export (`raw/verify/90a-…md` §1).

**`ABLATION_UNSUPPORTED` vẫn đúng nguyên văn cho `image_only_empty_text` — run mới không đổi điều đó.**
`TextLSTMModel.forward` vẫn tokenise về 0 token rồi index `x[0][0]` (`text_model.py:84-86`), và không
có gì trong công việc này chạm vào text tower của CLVC (`raw/verify/104-clvc-full-image-only.md`
"Còn chặn" mục 5).

**Nhưng vai "ảnh" trong run mới không phải điều kiện đó.** Nó được điền bởi một điều kiện **có tên
riêng**, `clvc_image_encoder_only`, khai báo qua `ABLATION_CONDITIONS` (không đụng `CONDITIONS`):
hàng query lấy **nguyên biểu thức target row** của upstream, `test.py:116`
(`cat(normalize(local.extract_img_feature(img)[1]), normalize(global.extract_img_feature(img)[1]))`),
**không có text tower nào được gọi** — đo bằng cách patch `compose_img_text` ném lỗi ở cả hai nửa mà
vẫn trả `(2, 2048)`, và hai caption khác nhau cho tensor **giống hệt** (`raw/verify/99-…md` §6.2, §6.3:
tolerance khai trước `1e-6`, đo được `6.705522537231445e-08`). Vì vậy nó là một **thứ khác** với điều
kiện protocol bị khai không hỗ trợ, và **cấm** viết thành "CLVC giờ làm được image-only": câu đúng là
"CLVC có một ablation ảnh **có tên**, không phải điều kiện `image_only_empty_text` của protocol". Hai
điều kiện này không thay thế nhau trong bất kỳ bảng nào, và gap của chúng không được đặt chung cột.

### 4.3 CLVC-Net: 119/6003 (1.98 %) query không chấm được — giữ có chủ ý, và đã tách được 117 + 2

`raw/verify/90-circus-composition-runs.md` §4.1: **119 trên 6003 query (1.98 %)** không được chấm vì
annotation của release nêu **hai** category cho các qid đó; adapter từ chối đoán và runner ghi lý do
theo từng query (`category_scope_failure_count: 119`). Đây là **chính sách đã khai báo, không phải
khuyết điểm của CLVC**: release tự báo `category_resolution.ambiguous_count: 119` (trên 6003 xét,
5884 resolve).

**Bản sửa quan trọng: 119 không phải một khối không thể tách.** Dry run trên **cả 6003 query**
(`raw/verify/99-clvc-scope-and-image-only.md` §2, §3; file `/tmp/clvc_scope_dryrun.json`, ngoài repo)
đọc ra:

```
loader_scoped                          5884
resolver_agrees_with_loader            5884
resolver_disagrees_with_loader            0
recovered_from_annotation               117     (shirt 46, toptee 71)
still_unscoped                            2
```

Tức **117 query tìm lại được category từ chính file annotation đã pin**, và **đúng 2 query** còn lại là
nhập nhằng thật: `7:3038` (stem `B004TMGDME`, caption `is darker and has a different graphic and is
darker`) và `7:3790` (stem `B0079K0YBO`, caption `is grey and has a less whimsical graphic and is
lighter`) — cả hai nằm trong **cả** `shirt` và `toptee`, lý do nguyên văn
`unresolved:annotation_names_two_categories` (`raw/verify/99-…md` §3). Chính sách giữ nguyên: 5 884 +
117 + 2 = 6 003, và hai qid kia **vẫn không được chấm** — nhưng từ nay đây là **hai** query nhập nhằng
được nêu tên, không phải "119 query không rõ". Chỉ dùng stem sẽ **tệ hơn**: trong 119, **26** reference
nằm ở cả hai split, nên stem một mình sẽ chọn im lặng cho 26 query mà annotation giải được bằng text
(cùng mục). Resolver được **tái dùng nguyên văn** từ `model_limn.py` (0 dòng sửa) nên hai model vẫn phủ
cùng một tập query.

**Đã đo — không còn là kỳ vọng.** Run mới `da0a2162be0e159a1a456cec` (§4.4b) cho đúng hệ quả này,
và đọc từ artifact chứ không từ dự đoán (`raw/verify/104-clvc-full-image-only.md` §3, §4, §7):

| đại lượng | run cũ `eab10106…-retry-2` | run mới `da0a2162…` |
|---|---|---|
| `adapter_query_scope_resolution.still_unscoped` | 119 | **2** (`unscoped_qids ["7:3038","7:3790"]`) |
| `adapter_query_scope_resolution.adapter_routed` | — (resolver chưa có) | **6001** |
| `unscored_count` mỗi điều kiện | 119 | **2** |
| `scored_count` mỗi điều kiện | 5884 | **6001** |
| `coverage` | `0.980176578377478` | **`0.9996668332500417`** |
| `topk_export.missing_query_count` | 238 (119 × 2 điều kiện) | **6** (2 × **3** điều kiện) |

Tức 117 query **thật sự được chấm**, không chỉ được đếm: mọi metric của mọi điều kiện **đều đổi**
(R@10 multimodal `7.256968048946295` → `7.198800199966672`; text-only `1.3936097892590076` →
`1.3997667055490752`), và `scored_count` là 6001 ở cả ba (`104` §9). Một metric giữ nguyên bit-for-bit
sẽ là dấu hiệu các hàng đó vẫn bị ghi `null` — không có chuyện đó.

**Rò rỉ modality — phán quyết (a) cho 5884, và KHÔNG phải (a) cho 117.** Bản trước viết gọn thành một
câu, và câu đó **sai** ở phần 117. Đầu vào routing **không** chỉ là stem. Đọc đúng hàm
`resolve_query_scope` (`workbench/composition/model_limn.py:189-227`, được `model_clvc_net.py` gọi
nguyên văn):

* **5884 query — category lấy từ loader.** `category = getattr(query, "category", None)`; nếu khác
  `None` thì `record["reason"] = "loader"` và **không** đụng tới văn bản. Đây **đúng là** metadata của
  release, lấy được không cần byte ảnh nào, không suy từ pixel/target/score. Phán quyết **(a)** đúng
  cho nhóm này.
* **117 query — category lấy từ `(stem, caption sửa)`, tức CÓ dùng văn bản.**
  `holders = index.get((stem, normalize_query_text(query.query_txt)), …)` — khoá là **cặp** (stem,
  caption đã chuẩn hoá), không phải stem đơn. Nhóm này được route bằng chính **edit caption**.
* **2 query — không có category duy nhất** (`len(holders) > 1` → `unresolved:annotation_names_two_categories`),
  nên không được chấm.

**Hệ quả, nói thẳng:** với **117/6001** query đã chấm, một điều kiện **loại bỏ** edit caption
(`clvc_image_encoder_only`, và cả `text_only_black_image`) vẫn **dùng** edit caption để chọn category.
Nhãn routing của nhóm đó là **hỗn hợp**: metadata release **cộng** văn bản mà điều kiện kia nói là đã
bỏ. Vì vậy **cấm** hai câu: "routing chỉ dùng stem, không cần văn bản" và "(a) đúng cho toàn bộ cohort".
Phán quyết đúng là: **(a) cho 5884; với 117 thì đó là nhãn routing trộn metadata với edit text.**

Điều này **đã được khai trong code**, không còn là suy luận của báo cáo:
`IMAGE_ENCODER_ONLY_LEAKS_SCOPE` (`workbench/composition/model_clvc_net.py:150`) nói nguyên văn
"category routing reads the release loader's assignment (reference image stem) for 5884 queries and,
for the 117 the loader left unscoped, (reference stem, edit caption text) resolved against the pinned
caption annotation; under text_only_black_image and clvc_image_encoder_only the routing input is
therefore a release DATASET LABEL plus, for those 117, the edit text that the condition otherwise
removes", kèm dạng một dòng `IMAGE_ENCODER_ONLY_LEAKS_CAPTION` cho trường `claim_narrowed`. Chữ ký
`Adapter.query_scope(self, query, candidate_names)` vẫn chặn việc nhãn phụ thuộc `condition`/gallery/
score — nhưng nó **không** chặn việc nhãn phụ thuộc `query.query_txt`, và đó chính là chỗ rò rỉ.

**Kích thước của rò rỉ đã đo, không phải ước lượng:** gap trên cohort A (6001, có 117 hàng nhãn routing
phụ thuộc caption) là `0.33061712371309204`; trên cohort B (5884, `caption_free: true`, không edit text
nào chạm routing) là **`0.33400015944039774`** — lệch `0.0033830357273053` (≈1.02 % của A). Nguồn:
`workbench/artifacts/composition/derived/category_routing_inputs/clvc_net_da0a2162be0e159a1a456cec_full.json`
(`raw/verify/99-clvc-scope-and-image-only.md` §4.3); chi tiết và cách dùng ở §4.4b.

**Việc còn mở, nêu tên:** quy tắc "một điều kiện image-only không được dùng văn bản trừ khi miễn trừ
đó được ghi trong một **run identity riêng**" **chưa** được đáp ứng. Đo chính xác, cả hai phạm vi:

```
$ grep -rn reference_metadata_seen workbench/composition workbench/backend workbench/scripts workbench/tests
(0 hit)                                    ← không có trong CODE
$ grep -rn reference_metadata_seen workbench
workbench/docs/reviews/20261111-master-audit/raw/verify/99-clvc-scope-and-image-only.md:125
workbench/docs/reviews/20261111-master-audit/raw/verify/99-clvc-scope-and-image-only.md:314
(2 dòng, cả hai trong CÙNG một file đề xuất)   ← 4 hit khi tính cả file .md và bản trùng
```

Tức trường `reference_metadata_seen` **chỉ tồn tại trong văn bản đề xuất** (`raw/verify/99-…md` §4,
§7.6), **không** trong code chạy: chưa ship, vì nó mint run id mới cho mọi run hiện có. Ghi nhận là
**open item có tên**, không phải đã đóng — và nếu ai chạy `grep` trên cả `workbench` sẽ thấy hit, nên
câu "0 hit" chỉ đúng khi giới hạn ở **thư mục code**, đúng như lệnh in ở trên.

**So với LIMN — nay nói chính xác, không suy diễn.** Run LIMN `10173dc45dea8c90fd2a284b` báo cùng
hình dạng ở mức summary: `ambiguous_count: 119`, `category_scope_failures: {multimodal: 119,
text_only_black_image: 119, image_only_empty_text: 119}`, `PARTIAL`, `execution_status PARTIAL` trên
6003 query (`raw/verify/90-…md` §4.1; đọc lại summary hôm nay).

Nhưng **rank artifact** của LIMN trả lời câu hỏi cụ thể hơn, và câu trả lời là **LIMN cũng để hai qid
đó unscored**. Đọc trực tiếp
`workbench/artifacts/composition/runs/10173dc45dea8c90fd2a284b/ranks/limn_base_iter0_all_categories.json`:
cả ba điều kiện có **5884 non-null / 119 null** trên 6003, và bốn qid được tra đều **`null` ở cả ba
điều kiện** — không chỉ `7:3038`/`7:3790` mà **cả** `7:2179` và `7:4066`, tức LIMN **chưa** có resolver
nên vẫn bỏ toàn bộ 119:

| qid | LIMN `multimodal` | LIMN `text_only_black_image` | LIMN `image_only_empty_text` |
|---|---|---|---|
| `7:3038` | **`null`** | **`null`** | **`null`** |
| `7:3790` | **`null`** | **`null`** | **`null`** |
| `7:2179` | **`null`** | **`null`** | **`null`** |
| `7:4066` | **`null`** | **`null`** | **`null`** |

Hệ quả chính xác cho tính so sánh chéo, nói đúng mức: CLVC mới và LIMN **trùng nhau ở đúng 2 query**
(`7:3038`, `7:3790` — cả hai đều unscored) và **khác nhau ở 117 query** còn lại, nơi CLVC mới **có**
hạng còn LIMN **không**. Vậy hai model **không còn** phủ cùng một tập query: cohort chung là **5884**
(nếu giao theo qid có hạng ở cả hai) chứ không phải 6001. Đây là hệ quả của một quyết định có chủ ý —
resolver được **tái dùng nguyên văn** từ `model_limn.py` (0 dòng sửa) nhưng **chỉ CLVC gọi nó** trong
một run — nên muốn so sánh chéo trên 6001 query thì LIMN phải chạy lại qua cùng resolver; báo cáo này
**không** làm việc đó và **không** ghi đè số LIMN. `PARTIAL` vẫn là trạng thái đúng cho cả hai run.

### 4.4 CLVC Full: hai run dưới **hai profile khác nhau** — `4.4a` run cũ (protocol triple), `4.4b` run mới (có gap ba điều kiện)

#### 4.4a Run cũ `eab10106b2df8e97fb7a5ce5` (`-retry-2`) — protocol triple, `image_only_empty_text` là ablation

Run Full đã hoàn tất trước đó: `eab10106b2df8e97fb7a5ce5`, `execution_status` **`PARTIAL`**, wall
**4489 s (1 h 14 m 56 s)**, 223 143 gallery forward + 12 006 query forward, batch 32, peak VRAM
**2897 MiB / 16303 MiB**, GPU util **6–10 %** ở **39.9–47.9 W** của 285 W
(`raw/verify/90-circus-composition-runs.md` §4.1). `condition_status` nguyên văn (cùng mục):

```json
{"image_only_empty_text": {"status": "ABLATION_UNSUPPORTED",
   "reason": "CLVC-Net cannot encode an empty edit: … text_model.py:84-86 …"},
 "multimodal":            {"status": "PARTIAL", "unscored_count": 119,
                           "category_scope_failure_count": 119, "fallback_used": false},
 "text_only_black_image": {"status": "PARTIAL", "unscored_count": 119,
                           "category_scope_failure_count": 119, "fallback_used": false}}
```

Metric theo điều kiện, mẫu số 6003 (`raw/verify/90-…md` §4.1; **đọc lại** từ
`workbench/artifacts/composition/summary/clvc_net_author_drive_unresolved_eab10106b2df8e97fb7a5ce5-retry-2/full.json`
— **17 029 B**, mtime `2026-11-12 04:53:22.955 +0700`). Bản `…_eab10106…/full.json` (không có hậu tố)
là lần chạy **cũ đã bị thay thế**; các con số giống nhau ở cả hai lần, nhưng lần **hoàn tất** là
`-retry-2` (`raw/verify/98-tgcir-full-run.md` §1):

| condition | scored | coverage | R@10 | R@50 | nDCG_full | MRR |
|---|---|---|---|---|---|---|
| `multimodal` | 5884 | 0.980176578377478 | 7.256968048946295 | 18.303874915023794 | 0.1495349414930845 | 0.03594722775284061 |
| `text_only_black_image` | 5884 | 0.980176578377478 | 1.3936097892590076 | 4.46974847042828 | 0.09954839186850512 | 0.006375664231695533 |
| `image_only_empty_text` | 0 | 0.0 | null | null | null | null |

`composition_gap_ndcg`: **`null`** với lý do `"no queries have exact ranks in all three conditions"`
(`raw/verify/90-…md` §4.1) — nên **không** có gap cho CLVC. `topk_export`:
`{"depth": 100, "requested": true, "status": "TOPK_EXPORT_COMPLETE_PARTIAL", "partial": true,
"missing_query_count": 238, ...}` với 238 = 119 × 2 điều kiện được export (cùng mục).

**Trạng thái lần chạy lại — ĐÃ XONG; PID `1366364` đã thoát.** Lần chạy `-retry-2` do PID `1366364`
(wrapper `gpu_job.sh` `1366356`) điều khiển **kết thúc** và ghi ra
`workbench/artifacts/composition/summary/clvc_net_author_drive_unresolved_eab10106b2df8e97fb7a5ce5-retry-2/full.json`
(**17 029 B**, mtime `04:53:22.955 +0700`), `execution_status` **`PARTIAL`**
(`raw/verify/98-tgcir-full-run.md` §1). Các số của chính lần chạy đó:

| đại lượng | giá trị | nguồn |
|---|---|---|
| wall time | **1867.2 s (31 min 7 s)** — log birth `04:22:15.752` → summary mtime `04:53:22.956` | 98 §1 |
| ảnh đã encode | **số cũ SAI — không dùng.** `92 390` đếm gallery một lần thay vì ba scope, **và** attempt `-retry-2` này phục vụ gallery từ **cache** (`feature_storage.projected_bytes` = 609 329 280 một lần cho mỗi scope = nhánh cache-hit; không file feature nào được ghi trong cửa sổ 04:22–04:53), nên `1867.2 s` chỉ mua **11 768** dòng query → **0.1587 s/dòng (query-only)**. Attempt **có** encode là attempt-1 `eab10106` (01:21:13→02:35:43): **234 911 = 223 143 gallery (3 scope × 74 381) + 11 768 query**, **4469.8 s → 0.01903 s/ảnh, 52.56 ảnh/s**. Mẫu số là **lượt encode ảnh**, không phải số ảnh gallery — xem `104` §0.2a, §9.1 | 98 §1; 104 §9.1 |
| `smoke` | `{"label": null, "limited": false, "query_limit": null}` — **không** dùng `--smoke` | 98 §1 |
| `query_count` / `gallery_count` | 6003 / 74381 | 98 §1 |
| interpreter | `clvc39-la` (Python 3.9.25, torch 2.7.1+cu128), RTX 5070 Ti | 98 §1 |
| peak VRAM | **chưa đo** — không có mẫu `nvidia-smi` nào được lấy trong lúc run chạy (run đã bay trước khi phiên đo bắt đầu) | 98 §1, §5 |

Peak VRAM **không** được gán cho run này: con số `2897 MiB` ở đầu mục thuộc lần chạy **trước**
(`eab10106…`, `raw/verify/90-…md` §4.1) và **không** được chuyển sang `-retry-2`. Câu cũ "**không** được
gán bất kỳ con số Full nào cho nó" nay **hết hiệu lực**: các con số trong bảng trên **là** số của nó.

R@1 của CLVC Full cũng **có**, ở file suy ra chứ không ở `full.json`: `multimodal`
**1.4106050305914344**, `text_only_black_image` **0.10197144799456152**, `image_only_empty_text`
**`null`** với lý do `"no queries have an exact rank"` — **không** ghi thành 0.0
(`raw/verify/101-metrics-r1-and-fallback.md` §3b; phục vụ sống với
`derived_metrics.recall_at_1.status = "VERIFIED"`, `raw/verify/100-api-cache-and-ui-blocked.md` §5).

#### 4.4b Run mới `da0a2162be0e159a1a456cec` — profile khai báo, gap ba điều kiện **tính được lần đầu**

Run đang bay lúc §0.1 được chụp **đã xong** và **đã ghi summary**. Nguồn chuẩn:
`raw/verify/104-clvc-full-image-only.md`.

Summary: `workbench/artifacts/composition/summary/clvc_net_author_drive_unresolved_da0a2162be0e159a1a456cec/full.json`
(**17 956 B**, mtime `2026-11-12 09:15:18.964 +0700`); ranks
`runs/da0a2162be0e159a1a456cec/ranks/clvc_net_author_drive_unresolved.json` (388.4 KB). Đọc lại nguyên
các trường dưới đây tại HEAD `a0c08c0` + commit báo cáo này.

| trường | giá trị |
|---|---|
| `execution_status` | **`PARTIAL`** |
| `query_count` | **6003** (`full` 6003, `sf` 4069, `validated` 586) |
| `smoke` | `{"label": null, "limited": false, "query_limit": null}` — Full thật |
| `conditions` | `["multimodal", "text_only_black_image", "clvc_image_encoder_only"]` |
| `image_only_empty_text` | **VẮNG** khỏi `conditions` **và** `condition_status` — **không phải** `ABLATION_UNSUPPORTED`, mà là **không được yêu cầu** |
| `condition_status` | cả ba điều kiện `PARTIAL`, `unscored_count` **2**, `category_scope_failure_count` 2, `fallback_used: false` |
| `feature_batch_fallback` | `partition: canonical_call_window`, `fallback_qids: []`, `runner_batch_size` **32**, `adapter_batch_size` **32**, `call_window` **4096**, `adapter_calls` 3 |
| `adapter_query_scope_resolution` | `loader_scoped` **5884**, `adapter_routed` **6001**, `still_unscoped` **2**, `unscoped_qids` `["7:3038","7:3790"]` |
| wall time | **2068.4 s (34 min 28 s)** — `08:40:50.565` → `09:15:18.964` |
| ảnh đã encode | **241 146** = **223 143** gallery (3 scope × 74 381 — ba sidecar `[74381, 2048]`, ba digest khác nhau) + **18 003** dòng query (3 điều kiện × 6001; 2 query unscored giữ hàng NaN) → **2068.4 s / 241 146 = 0.00858 s/ảnh, 116.6 ảnh/s**. Mẫu số là **lượt encode ảnh** (scope × ảnh gallery + dòng query), **không** phải số ảnh gallery; bản cũ `92 390 → 0.02239` đếm gallery một lần và thấp hơn 2.61× (`104` §0.2a) |
| GPU | 22 mẫu `nvidia-smi`: util **min 8 % / max 99 % / mean 90.7 %** (20/22 mẫu ở 96-99 %), `memory.used` **1701 MiB** (hằng số — **không** phải peak toàn run), power **116.55-217.35 W**, mean **205.7 W** |
| `topk_export` | `TOPK_EXPORT_COMPLETE_PARTIAL`, depth **100**, **ba** điều kiện, `missing_query_count` **6** (= 2 × 3 manifest), `partial: true` |

Metric theo điều kiện, **6001/6003 chấm được**, coverage **`0.9996668332500417`** (đọc từ
`metrics.<role>`, với `metrics.image_only.condition == "clvc_image_encoder_only"`):

| condition | R@1 | R@10 | R@50 | mrr | ndcg_full |
|---|---|---|---|---|---|
| `multimodal` | 1.3997667055490752 | 7.198800199966672 | 18.180303282786202 | 0.03564730414020548 | 0.14921544731411218 |
| `text_only_black_image` | 0.11664722546242293 | 1.3997667055490752 | 4.49925012497917 | 0.006595896893970061 | 0.09988226530955799 |
| `clvc_image_encoder_only` | 0.11664722546242293 | 1.1998000333277787 | 2.9495084152641224 | 0.005437399476599689 | 0.08644584977266369 |

**Nhưng gap này là gap CÓ ĐIỀU KIỆN — đọc kèm §4.3 trước khi trích nó.** Cohort 6001 ở trên không
đồng nhất về nguồn nhãn routing: **5884** query lấy category từ loader (không dùng văn bản), **117**
query lấy category từ `(stem, caption sửa)` đối chiếu file caption đã pin — tức routing của 117 query đó
**dùng chính edit caption** mà `clvc_image_encoder_only` nói là đã bỏ (`model_limn.py:213`;
`IMAGE_ENCODER_ONLY_LEAKS_SCOPE`, `workbench/composition/model_clvc_net.py:150`). Vì vậy
`0.33061712371309204` là gap trên **cohort A (6001)** — cohort mà **117/6001 hàng có nhãn routing lấy
một phần từ văn bản**.

**Con số strict caption-free đã đo: cohort B.** Cùng một rank table, cùng công thức, chỉ khác **tập
qid**: bỏ 117 query mà resolver route bằng `(stem, caption)` và giữ 5884 query loader-scoped
(`query_count 5884`, `caption_free: true`):

| cohort | query | `composition_gap_ndcg` | mm | text | image | `caption_free` |
|---|---|---|---|---|---|---|
| **A — toàn bộ query đã chấm (bản đã công bố)** | **6001** | **0.33061712371309204** | 0.14921544731411218 | 0.09988226530955799 | 0.08644584977266369 | `false` |
| **B — chỉ loader-scoped** | **5884** | **0.33400015944039774** | 0.14947221054602272 | 0.09954846839174243 | 0.08655137339831104 | **`true`** |

Hai số lệch nhau **0.0033830357273053** (≈**1.02 %** của A). Nguồn:
`workbench/artifacts/composition/derived/category_routing_inputs/clvc_net_da0a2162be0e159a1a456cec_full.json`
(4 623 B; khoá `cohort_a_all_scored`, `cohort_b_loader_scoped`), sinh bởi
`workbench/scripts/derive_routing_cohort_gap.py`, ghi lại ở `raw/verify/99-clvc-scope-and-image-only.md`
§4.3. Công cụ **replay** resolver trên CPU và **từ chối** nếu replay không khớp
`adapter_query_scope_resolution` của artifact — ở đây `replay_agrees_with_artifact: true`
(5 884 / 6 001 / 2, đúng hai qid `7:3038`/`7:3790`) — và **từ chối ghi đè** file derived đã tồn tại.
`cohort_a_all_scored.reproduces_published: true`.

**Dùng số nào:** khi một khẳng định đòi "không có edit text nào chạm vào routing" thì **phải** trích
**B = `0.33400015944039774` trên 5884**. Khi nói về run như nó đã công bố thì trích **A =
`0.33061712371309204` trên 6001**, và nói rõ A là gap của một cohort có 117 hàng nhãn routing phụ thuộc
caption.

**Đây là phụ thuộc routing, không phải lỗi metric.** Hai số đến từ **cùng** rank table
(`runs/da0a2162be0e159a1a456cec/ranks/…`), **cùng** `metrics.compose`, **cùng** công thức
`1 - max(text, image)/mm`; chúng chỉ khác **tập qid được đưa vào**. A tái lập artifact tới **1e-12**
(`104` §5, §6; `reproduces_published: true`). Không có hàng nào bị tính sai; cái từng sai là **mô tả
cohort phụ thuộc vào gì** (§4.3), và đó là thứ đã sửa.

**Run đã xong KHÔNG ghi lại điều này, và không được nói ngược lại.** Summary `da0a2162…/full.json` là
**bất biến** và **không** mang khối `category_routing_inputs`; file derived ở trên là artifact **riêng**,
không nằm trong summary. Một run **sau** sẽ mang nó (patch spec: `raw/verify/99-…md` §7.7) — xem §5.2
mục 12.

**Đây là lần đầu gap ba điều kiện của CLVC tính được.** `composition_gap_ndcg`
**`0.33061712371309204`**, `gap_cohort_count` **6001**, `mrr_gap` `0.029051407246235415`,
`gap_unavailable_reason` **`null`**; `gap_conditions` =
`["multimodal","text_only_black_image","clvc_image_encoder_only"]` và `gap_role_reason` khác `null`:
"image role filled by the adapter's named ablation 'clvc_image_encoder_only': the protocol image
condition 'image_only_empty_text' carries no ranks in this run". Công thức tự kiểm:
`1 - max(0.09988226530955799, 0.08644584977266369) / 0.14921544731411218` =
`0.33061712371309204`, khớp **chính xác** (`104` §5). `shortcut_outcomes`:
`{"both": 1, "composition_required": 382, "image_only": 71, "neither_scored": 2, "text_only": 83,
"unresolved": 5464}`.

**Run này KHÔNG thay số của run cũ, và không được gộp với nó.** Nó thay **coverage**: 5884 → **6001**
chấm được, `still_unscoped` 119 → **2**, `missing_query_count` 238 → **6**. Số của
`eab10106…-retry-2` (§4.4a) vẫn là số của **run đó**, trên **cohort 5884** — không viết lại, không
trộn. Và **gap `0.3306` không phải gap protocol**: điều kiện ảnh ở đây là `clvc_image_encoder_only`
(**thay thế**, ghi theo tên), không phải `image_only_empty_text`; nên **không** được đặt chung cột với
gap `0.4385` của TG-CIR (`104` "Còn chặn" mục 4).

**Ba giới hạn trung thực của run này, ghi nguyên từ `104`:**

1. **Lệnh như văn bản không chạy được; controller phải là `ctl312h`.** Gọi
   `clvc39-la/bin/python … composition_eval.py` bị từ chối trong **20 s, trước mọi encode**:
   `"environment gate would refuse the run [BLOCKED/UNSAFE_INTERPRETER]: cannot use active workbench
   environment"`, `"status": "SELECTED_BUT_BLOCKED"` — cổng `check_environment_isolation`
   (`workbench/backend/runtime.py:257`) không cho dùng chính venv của model làm môi trường điều phối.
   Đo trực tiếp: `clvc39-la` → `(False, 'cannot use active workbench environment')`;
   `ctl312h` → `(True, 'ISOLATION_VERIFIED')`. Run này (và **cả hai** run Full trước) dùng `ctl312h`
   làm controller, model vẫn chạy trên `clvc39-la` (`verification_status: GPU_RUNTIME_READY`). Cổng
   **không** bị sửa; đây là báo cáo, không phải workaround (`104` §0.1, "Còn chặn" mục 1).
2. **`backfill_recall_at_1.py` từng từ chối backfill run này — đã vá sau đó, và đây là cả hai nửa.**
   Lúc `104` được viết, script **từ chối**: `METRIC_BLOCKS` chỉ map ba tên protocol, nên
   `clvc_image_encoder_only` rơi vào chính nó, không thấy `metrics["clvc_image_encoder_only"]` (khối
   thật là `metrics["image_only"]`), so với `None` rồi từ chối —
   `clvc_image_encoder_only.scored_count: recomputed 6001 vs published None` (và ba trường nữa), exit 1,
   **không** ghi file `derived/` (`104` §6.1, "Còn chặn" mục 3). Commit `b887c7e` ("read the
   condition->metric-block role mapping from the artifact") đã sửa đúng điểm đó: chạy lại
   `backfill_recall_at_1.py --run clvc_net:da0a2162be0e159a1a456cec` nay trả **rc = 0** và ghi record
   với `multimodal` 1.3997667055490752 / `text_only_black_image` 0.11664722546242293 trên
   `scored_count 6001` — khớp `full.json` (đo lại trong lượt này; file `derived/` sinh ra bởi phép kiểm
   đó đã được xoá, vì run này **không cần** nó: R@1 đã có sẵn trong `full.json`). Ghi chú phương pháp:
   script **không** đọc `gap_conditions`, nên câu hỏi "có nhận gap_conditions phi protocol không" trả
   lời **có**; chỗ chặn thật là map tên điều kiện, và đó là chỗ đã vá.
3. **VRAM 1701 MiB là cửa sổ lấy mẫu, không phải peak.** 22 mẫu trong pha encode/query; một đỉnh
   thoáng qua ngoài cửa sổ sẽ không xuất hiện — con số này là **sàn** cho pha đó (`104` "Còn chặn"
   mục 7).

**Không còn `composition_eval.py` nào đang chạy** sau run này: wrapper `1477254` và controller
`1477261` đều đã thoát khi run hoàn tất (`104` §0.2).

---

## 5. Khuyết điểm: cái gì đã **VÁ** và cái gì **CÒN**

Mục này thay cho danh sách "CHƯA vá" của bản trước, vì nhiều mục trong đó **đã vá và đã đo** ở những
commit sau. Mỗi mục dưới đây mang file bằng chứng; chỗ chưa đo thì ghi `chưa đo`.

### 5.1 Đã VÁ (9 mục) — mỗi mục có bằng chứng đo lại, không phải "đã sửa" suông

1. **Tràn backoff** (trước: `restart_delay(1025)` ném `OverflowError: int too large to convert to
   float`). Vá bằng clamp **số mũ trước** phép nhân: `BACKOFF_SATURATING_EXPONENT == 6`, thang
   0.2/0.4/0.8/1.6/3.2/6.4/8.0, bão hoà 8.0 tại 8, 9, 1025, 1e6, 1e12; 1000 lần gọi tại
   `failures = 10**6` mất 0.000156 s (`raw/verify/102-queue-gpu-and-blocker-inventory.md` §5;
   `raw/verify/84-evaluator-recovery-and-backoff.md` §1).
2. **`jobs.py` recovery lúc khởi động** — 5 phát hiện đều vá và đo trên fixture thật: bản ghi in-flight
   → `failed` với artifact **byte-identical**, bản ghi `completed` không bị chạm, metadata hỏng
   (`list`/scalar) không còn làm sập lượt quét (`jobs.py:509-515`), recovery không bao giờ chạy lại job,
   và kiểm toán quyền sở hữu kill (không `pkill`, không `killall`, không pid từ file)
   (`raw/verify/84-…md` §3). Test hồi quy qua **queue thật**: 4 test, `4 passed in 0.80s`
   (`raw/verify/102-…-inventory.md` §6).
3. **`_MODELS_CACHE` không khoá** — nay key theo `_models_state_stamp` (`main.py:240`, 73 entry,
   `stat`/`setting` only): stamp 0.016 s (warm 0.006 s), `/api/models` **lạnh 98.43 s** → **67.28 s**
   trên process sống, **0.026 s / 0.017 s** khi ấm; badge đổi **trong cùng process** theo cả hai chiều
   (thiếu → sai checksum → thiếu), và gate `POST /api/jobs` **vẫn** 409
   (`raw/verify/100-api-cache-and-ui-blocked.md` §2, §"Live confirmation after restart" mục 1).
4. **`statistics.query_count`** phục vụ `6003` cạnh mẫu số `5` trên smoke — nay suy từ số query **thực
   sự được xếp hạng**: smoke `6003 → 5`, và run Full vẫn báo đúng cả cohort (3/3). Đo lại sống trên
   `1eee4c0a`: `statistics.query_count: 5`, ba denominator đều `5`, `statistics_whole_run.query_count: 5`
   (`raw/verify/100-…md` §3, §"Live confirmation after restart" mục 3).
5. **`subset_provenance` phụ thuộc thứ tự gọi** — `read_summary_file` nay trả `copy.deepcopy`; detail
   **byte-identical ở cả hai thứ tự gọi**, cùng `80 174 B` và cùng sha256
   `bb7406d633bad67e0f4f6daca0bed01e921fafd5c2e0bf1da8adcbda6e359e57` (đo sống, qua HTTP; trong
   process là `80 104 B` / `bc165e5…`) (`raw/verify/100-…md` §4, §"Live confirmation after restart" mục 4).
6. **R@1 từng vắng mặt** — `recall_at_1` đã có trong `ConditionMetrics` (`workbench/composition/metrics.py:48`)
   và được tính bằng `100.0 * sum(hit_at_k(rank, 1) …) / scored_count` (`:77`) trên **cùng cohort và cùng
   mẫu số** với R@10/R@50; 7 test biên (rỗng → `None` + lý do, không bao giờ `0`), backfill 3 guard
   (cohort equality, reproducibility ở 1e-12, immutability) và **136/136** digest artifact **không đổi**
   (`raw/verify/101-metrics-r1-and-fallback.md` §1, §2, §3, §7). Số suy ra: tgcir 6.3135099117108116 /
   1.949025487256372 / 0.08329168748958854; clvc 1.4106050305914344 / 0.10197144799456152 / `null`
   "no queries have an exact rank" (§3a, §3b). **Giới hạn còn lại:** hai file `full.json` **đã công bố
   trước đó không bị ghi lại** (bị từ chối theo tên), nên R@1 của **chúng** chỉ tồn tại ở file suy ra +
   overlay API (`raw/verify/101-…md` "Còn chặn" mục 2). Run `da0a2162…` thì **khác**: nó được chấm sau khi
   `metrics.py` có `recall_at_1`, nên mang `metrics.multimodal.recall_at_1 = 1.3997667055490752` **ngay
   trong `full.json`** (§4.4b) — hiện là **1/36** file `summary/*/full.json` có key này (§9). Mỗi lần
   trích R@1 phải nói rõ nó đến từ `full.json` hay từ file derived.
7. **Fallback encode không được ghi lại** — `runner.py` nay ghi partition theo từng điều kiện
   (`partition`, `fallback_used`, **`fallback_qids`**, `fallback_reason`, `call_window`,
   `runner_batch_size`, `adapter_batch_size`, `adapter_calls`, `runner.py:657-700`) và khoá memmap cache
   theo `_feature_partition_fingerprint` (`runner.py:669`), nên cache sinh từ partition trộn **không**
   thể đội lốt cache canonical: `canonical-key cache HIT?: False` sau khi fallback, và 6 test khoá hành
   vi (`raw/verify/101-…md` §5, §6). **Giới hạn:** hai run Full đã công bố có `work/` đã prune nên
   `partition` của chúng là **không thể biết từ đĩa** (`raw/verify/101-…md` "Còn chặn" mục 3) — đã vá
   cho run **sau** commit, không hồi tố.
8. **Cohort save → reopen → export** (trước: `400 run_not_found`, index DuckDB rỗng) — nhánh
   composition trong `storage.py` (`c:<run_id>`) làm tròn vòng: `PUT` 200, `GET /api/cohorts` mở lại
   đúng id, `export/{json,csv,markdown}` 200/200/200, **và** cùng đường đó trong Chromium
   (`cohort-20261112T014331`, 6 003 qid, `r1 6.3135099117108116`, `r10 23.33833083458271` — trùng
   metric của chính run). Năm đường từ chối đều 400 và nêu tên nguyên nhân; cohort cũ (DuckDB) vẫn nạp
   (`raw/verify/100-…md` §9, §"Live confirmation after restart" mục 6, 6b).
9. **CLVC 119-query coverage** — không còn là khối mù: dry run 6 003 query tách **117 tìm lại được**
   (shirt 46, toptee 71) + **2 nhập nhằng thật** (`7:3038`, `7:3790`), resolver **tái dùng nguyên văn**
   từ `model_limn.py` (0 dòng sửa) và 15 test mới khoá hành vi; seam của runner cho
   `routed 6001 / unscoped 2`, scope digest đổi từ `8a5dd517…` sang `3a1c5a67…` nên lần chạy lại mint
   run id mới thay vì tái dùng ranks all-null (`raw/verify/99-clvc-scope-and-image-only.md` §2, §3, §5).
   Hệ quả metric của việc này **đã đo** trên run mới: coverage `0.980176578377478` →
   **`0.9996668332500417`**, và mọi metric đều đổi (§4.3, §4.4b).

### 5.2 CÒN (12 mục) — mỗi mục nêu đúng thứ còn thiếu

1. **Nguồn upstream của PAIR** — `BLOCKED_MISSING_SOURCE_CLASSES`, không phải "blocked" chung chung:
   `iLearn-Lab/ICASSP25-PAIR` tại `bfb0b98cdb4c44f6b9190baaa84d783a77dd888f` có **21** commit và **31**
   đường dẫn từng tồn tại, **không** chứa `model_vssp` (`Encoder`, `Backbone`, `FeatureWiseAffine`) hay
   `MHTransformer_dis` (`Transformer`, `TransformerLayer`, `MultiHeadAttention`, `MlpTransformer`) mà
   các pickle checkpoint yêu cầu (`raw/verify/102-…-inventory.md` §7.2 dòng `PAIR-SRC`).
2. **15 model `NOT_RUN` cho G1/G2** (13 active + 2 deferred `csmcir`, `sprc`) — không có artifact
   differential native↔adapter; chỉ `encoder` và `limn` có, và cả hai là `MANDATORY_SUBCHECK_FAILED`
   (`encoder.failure_count 16`, `limn` 3 `batch_1_vs_n.max_rel`) nên `passed_models: []`
   (`raw/verify/102-…-inventory.md` §7.2 dòng `G1G2-15`).
3. **5 ca chạy lại** — `clip4cir_rn50x4_noft` (chỉ roll-up, `artifact_backed: false`), `airknow` (3 dòng
   counted, `fiq_n05` có hai dòng xung đột `52.54661498980954` vs `52.924086629607395`), `clvc_net`
   (hai dòng counted cùng `protocol_id`, đã truy provenance: **cùng byte checkpoint**
   `bd9b6889…79cb`, khác biệt 1-2 query do tie-break — nên nhãn là tie-break, không phải provenance),
   `pair` (4 dòng B1/B2 × H_normal/H_rev), `habit e0dd04a4` (`FAILED` vì interpreter `airknow-adapted`
   3.11.15 ≠ 3.8.10; env nay đã đúng). ENCODER **không** nằm trong 5
   (`raw/verify/102-…-inventory.md` §7.2 dòng `RERUN-5`, `CLVC-NUM`).
4. **LIMN `7:3038` / `7:3790`** — nay nêu chính xác là **hai** query nhập nhằng còn lại (không phải
   "119"), mỗi cái nằm trong **cả** `shirt` và `toptee` với cùng reference identity và cùng cặp caption;
   release không có khoá category nào để phân xử, và mọi luật phân xử (tra ảnh target, so score, quyết
   định curation) đều phá luật "chỉ gán từ đúng một category được nêu"
   (`raw/verify/99-…md` §3; `raw/verify/102-…-inventory.md` §7.2 dòng `LIMN-2`). Cần metadata upstream,
   không phải vá code.
5. **46/50 dòng metric dùng đơn vị suy từ độ lớn** — `unit_declared 4`, `unit_sibling_key 2`,
   `unit_inferred 46`, `unit_not_declared 46` trên `blocks_with_metrics 50`; đã soát từng dòng, **0**
   khai báo bị bỏ sót, và các key `*_pct` mô tả đại lượng khác nên không phải khai báo thang macro. Sửa
   được **chỉ ở producer** của từng artifact, không sửa được ở tầng API
   (`raw/verify/102-…-inventory.md` §7.2 dòng `UNIT-46`).
6. **`TESTED_SOURCE_SHA` đã nâng — phần còn mở là 20 test fail, không phải pin.** Pin cũ
   `96d86010e85048d82f2dacf26e17d7b7206d31ab` đã được **thay bằng `117fc17220d0e35651c6778c1afb62e271c8d944`**
   sau một lượt sweep **dừng** (stationary), trong **cả hai** release document; sweep đó đo
   **`3123 passed, 20 failed, 47 skipped = 3190 collected`**, wall **`3586.55 s`**, record
   `raw/verify/106-sweep-117fc172.md`; guard `test_handoff_commands.py` **8 passed**. Lần nâng phủ
   **29 file production, +3505 / −246** kể từ pin cũ (trước đó báo cáo này ghi 14 file / 1 805 / 155
   ở pin `96d86010`, và con số đó là **lịch sử**, không phải hiện trạng).
   **Chưa đóng:** **20 test vẫn fail** — **18** id baseline đã ghi (blip ×4, `circus_cli`,
   `tme_lifecycle` ×2, `tme_precision`, `f08_dcnet`, `handoff_commands` SHA guard,
   `release_blocker_gpu_records` ×7, `prepare_model`) cộng **2** id **chưa từng nằm trong baseline 18**
   (`test_doctor.py::test_dcnet_runtime_asset_blocks_after_preparation_is_present`,
   `test_evaluate_models.py::test_dcnet_audited_command_reports_non_command_blockers`), cả hai cũng fail
   ở pin trước (`106` §3, §4). Ba trong số đó phụ thuộc trạng thái host nên pass trong worktree trần và
   fail trong cây thật (`106` §6) — ghi nhận, không sửa.

   **Vì sao "20 fail" và "guard 8 passed" cùng đúng — không mâu thuẫn.** Một trong 20 id đó là
   `test_handoff_commands.py::test_no_release_document_points_at_a_superseded_source_sha`. Nó fail
   trong sweep **vì** pin chưa được nâng: guard đó **từ chối** một release document trỏ tới SHA đã bị
   thay, nên ở thời điểm `117fc172` được đo, chính pin còn lệch cây của nó và guard đỏ **có chủ ý**.
   Sau khi `c998504` nâng pin, cùng guard đó **pass** (`8 passed`, đo lại sau mọi commit về sau —
   `106` §5). Vậy tập fail ghi trong sweep chứa **một** test chỉ đỏ khi pin còn lệch.

   Hệ quả, **suy ra chứ không đo**: chạy lại toàn suite tại HEAD hiện tại *nên* cho **19 fail / 3124
   passed / 47 skipped**, vì đúng một id đã chuyển sang xanh. Đây là **phép suy** từ triple đã ghi cộng
   trạng thái guard đã kiểm, **không** phải một lượt sweep mới — báo cáo này không chạy lại suite
   ~60 phút và **không** được trích `19`/`3124` như một kết quả đo. Triple **đã đo** vẫn là
   **`3123 passed, 20 failed, 47 skipped = 3190`** và **không** bị viết lại: nó là kết quả của sweep ở
   `117fc172`, không phải của HEAD.
7. **Người gửi SIGTERM lịch sử vẫn chưa xác định** — đường nhận đã biết
   (`serve_workbench.py:152-166`, `forward_stop` → `os.kill(child.pid, SIGTERM)` → `raise_signal(SIGINT)`),
   `jobs.py` bị loại (`start_new_session=True`), test/script trong repo bị loại, session reaping bị loại;
   bốn chữ ký chết đã commit (`raw/service-death-log{,-2,-3,-4}.txt`, PID 3691345 / 4040574 / 4062957 /
   4082632, đều `[WARN] npm exited with status -15`). Không thể đóng bằng suy luận: "that inference is
   for a future investigation, not evidence about the past"
   (`raw/verify/102-…-inventory.md` §7.2 dòng `SIGTERM-SENDER`; `raw/verify/86-…md` cùng mục).
8. **Worst-case `TimeoutStopSec=30` chưa đo** — chỉ bị chặn bằng **số học** (`GRACE_SECONDS` 10 s + 5 s
   SIGKILL mỗi job group + 10 s worker join ≤ 30 s); chưa có ca nào cho một job sống phớt lờ SIGTERM hết
   grace (`raw/verify/86-systemd-user-services.md` §"Still unproven / not done"; §1.6 trên).
9. **`gpu_may_be_held`: release là thủ tục, không phải kiểm chứng** — guard được **GIỮ**, không xoá:
   con `setsid` **sống** sau SIGKILL của chủ queue trên supervisor trần (`child_alive_after_owner_sigkill
   = True`), **chết** trong unit `KillMode=mixed` ở t+0.050 s, trước bản thay thế 1.09 s — nhưng
   `jobs.py` **không** quan sát được mình đang ở đường nào. Release chạy đúng (`blocked` → restart →
   `completed`) nhưng `/tmp/q102/restart_clears.log` cho thấy restart **vẫn nhận dispatch khi orphan còn
   sống** (`ORPHAN_STILL_ALIVE=True`, `dispatch_status=completed`): "release the device and restart" chỉ
   được bảo đảm bởi sự trung thực của operator, không phải bởi code
   (`raw/verify/102-…-inventory.md` §1-§4, §8).
10. **Export cohort CSV/Markdown thiếu cột `reference`** — `composition_cohort_metrics` cố ý trả
    `run_id` chứ không trả reference `c:<run_id>`, vì bộ cột của export **dùng chung** với nhánh DuckDB
    (`csv.DictWriter(..., fieldnames=list(metrics[0]))` tại `workbench/backend/main.py:1530`, và header
    Markdown cố định tại `:1535`), nên thêm cột sẽ đổi hình dạng của **mọi** export
    (`workbench/backend/storage.py:339-342`). Hệ quả: một export chỉ có `run_id` **không tự nói** nó là
    composition hay index, và reference chỉ còn trong provenance của cohort. Ghi nhận, **chưa vá** —
    sửa cần một quyết định về hợp đồng cột (thêm cột cho cả hai nhánh, hay tách schema).
11. **Peak VRAM của CLVC Full (run CŨ) chưa đo — và run MỚI thì có mẫu, nhưng không phải peak.**
    Hai câu riêng, không được trộn:
    * **Run cũ `eab10106…-retry-2`: `chưa đo`.** Không có mẫu `nvidia-smi` nào được lấy trong lúc nó
      chạy; `2897 MiB` thuộc lần chạy `eab10106…` **trước đó** và **không** được chuyển sang
      (`raw/verify/98-tgcir-full-run.md` §1, §5).
    * **Run mới `da0a2162…`: CÓ mẫu — 22 mẫu, `memory.used` `1701 MiB` hằng số, util mean 90.7 % (20/22
      mẫu ≥ 96 %), power mean 205.7 W.** Nhưng đó là **cửa sổ lấy mẫu**, không phải peak toàn run: một
      đỉnh thoáng qua ngoài cửa sổ sẽ không xuất hiện, nên **1701 MiB là sàn** cho pha đó
      (`raw/verify/104-clvc-full-image-only.md` §0.3, "Còn chặn" mục 7). **Không** con số nào ở đây
      được gán cho run cũ.
    * TG-CIR vẫn là run duy nhất có mẫu GPU **toàn run** đọc được: 22 mẫu, util 97-100 %, 2172 MiB,
      235.02-260.04 W (`raw/verify/98-…md` §2).
12. **Run đã công bố không mang `category_routing_inputs`; một run sau thì sẽ.** Summary
    `da0a2162…/full.json` là **bất biến** và **không** có khối nào nói rằng 117/6001 query có nhãn
    routing lấy từ `(stem, caption)`. Số đo hiện nằm ở một artifact **riêng**
    (`derived/category_routing_inputs/clvc_net_da0a2162be0e159a1a456cec_full.json`, 4 623 B) — đủ để
    trích, nhưng **không** phải một phần của run. Patch spec để run **sau** tự mang khối đó:
    `raw/verify/99-clvc-scope-and-image-only.md` §7.7. Hệ quả phải nói kèm: bất kỳ ai đọc
    `full.json` mà không đọc file derived sẽ thấy `0.3306` **không kèm** cảnh báo cohort, nên câu
    "run này đã ghi lại việc routing dùng caption" là **sai** cho tới khi §7.7 ship.

### 5.3 Hai ghi chú còn mở, giữ nguyên nhưng nay có thêm dữ kiện

* **Gallery encode `0.0138 s/ảnh` so với `0.0027 s/ảnh` độc lập (~5×) — vẫn chưa giải thích.** Quy cho
  memmap 609 MB flush nhưng **chưa chứng minh** (`raw/verify/96-staged-path-and-tgcir-worker-exit.md`
  §6.2). Dữ kiện mới **thu hẹp** khoảng, với mẫu số đã sửa (**lượt encode ảnh**, không phải số ảnh gallery):
  `clvc_net` `da0a2162` **0.00858 s/ảnh** (241 146 / 2068.4 s) và `tgcir` **0.00603 s/ảnh** (92 390 /
  556.9 s — mẫu số này **đúng**, TG-CIR chỉ có **một** gallery `global`; xem `104` §0.2a và
  `raw/verify/98-…md` §2). Con số `0.0202 s/ảnh` của `clvc_net` **đã bị rút**: nó đếm gallery một lần
  và attempt `-retry-2` còn phục vụ gallery từ cache (`104` §9.1), nên **không** dùng nó làm mốc.
  Hai số còn lại (0.00858, 0.00603) đều **không** tái hiện `0.0027`, nên phép đọc standalone càng ít
  được xác nhận hơn — mục vẫn **mở**, và vẫn cần per-scope instrumentation trong **một** run.
* **Guard first-job của TG-CIR `adapter_worker_exited exit_status=1`** — guard **đã ship** (`be9de58` +
  hunk `model_tgcir.py:653-663`) và chỉ được chứng minh trên **CPU** (trước: `exit_status: 1, replies:
  0`; sau: `exit_status: 0, replies: 2`); thay đổi latch thì **KHÔNG** ship và phải **giữ nguyên**
  reverted (`raw/verify/96-…md` §2.4, §5). Câu "chưa có bằng chứng từ run thật" nay có bạn đồng hành,
  **không** phải mâu thuẫn: TG-CIR Full `7b1fa850` chạy **`COMPLETED` 6003/6003 query**, tức một run
  thật **không** đi vào guard — đúng như thiết kế. Đọc "chưa chứng minh trên run thật" thành "có thể
  hỏng" là đọc sai (`raw/verify/98-…md` §2).

## 6. Bảng file bằng chứng

| đường dẫn | chứng minh điều gì |
|---|---|
| `raw/verify/84-evaluator-recovery-and-backoff.md` | `restart_delay(1025)` ném `OverflowError`; cách vá clamp số mũ (`BACKOFF_SATURATING_EXPONENT == 6`) và thang 0.2→8.0; crash loop 6 restart/10 s với gap 0.2/0.4/0.8/1.6/3.2/6.4; `jobs.py` recovery 5 phát hiện + kiểm toán quyền sở hữu kill + guard `gpu_may_be_held`; SIGTERM lịch sử vẫn chưa rõ người gửi |
| `raw/verify/86-systemd-user-services.md` | hai unit user; `MainPID` = uvicorn không wrapper; `Restart=always`/`RestartSec=2`/`StartLimit 60,5`/`KillMode=mixed`/`TimeoutStopSec=30`; ba ca kill (3092 ms / 2881 ms / giữ dừng); cô lập cgroup cho con `setsid` + timeline; frontend 2432 ms và 2381 ms + lý do `mixed` vs `process`; cutover 23:48:4x, MainPID trước→sau, kill thật ≈2.28 s; `/tmp/cutover_rollback.sh` md5 `04fbd5ec…` |
| `raw/verify/89-adapted-runtime-and-gpu-capability.md` | `adapted_runtime` sm_120 cho `clvc_net`/`tgcir`; tier `BLOCKED` → `GPU_VERIFICATION_PENDING` → `GPU_RUNTIME_READY`; bản ghi GPU (capability 12.0, torch 2.7.1+cu128, 16303 MiB); `command_status` giữ `COMMAND_BLOCKED_SOURCE_ENTRYPOINT` |
| `raw/verify/90-circus-composition-runs.md` | CLVC Full `eab10106` §4.1 (PARTIAL, 4489 s, metrics, `gap null`, 119/6003) — **lần chạy bị thay thế**, xem `98`; §4.2 s/img 0.0027 vs 0.0603; §4.3 TG-CIR `9f115afe` mất 2 điều kiện; danh sách "Còn chặn" |
| `raw/verify/98-tgcir-full-run.md` | **nguồn chuẩn cho CẢ HAI run Full**: `tgcir` `7b1fa85007bffdab931f7f19` `COMPLETED`, 6003 q, gap `0.43853620852911723`, `TOPK_EXPORT_COMPLETE` missing 0 (§2); `clvc_net` `eab10106…-retry-2` `PARTIAL`, 1867.2 s, **s/ảnh cũ SAI — gallery lấy từ cache**, xem `104` §9.1, VRAM `chưa đo` (§1); side-by-side (§3); cross-check API 200/71 954 B và JSONL=API (§4); "composition ≠ native" (§ cuối) |
| `raw/verify/99-clvc-scope-and-image-only.md` | dry run 6 003 query: `loader_scoped 5884`, `recovered_from_annotation 117` (shirt 46, toptee 71), `still_unscoped 2` = `7:3038`/`7:3790` (§2, §3); phán quyết rò rỉ modality (a) + điều cấm phát biểu (§4) — **lưu ý:** (a) đúng cho 5884, còn 117 được route bằng `(stem, caption)` nên (a) **không** đúng nguyên trạng cho chúng (xem §4.3; khai trong `model_clvc_net.py:150`); `clvc_image_encoder_only` đo trên CPU: tolerance `1e-6`, đo được `6.705522537231445e-08` (§6.3); 15 test (§5) |
| `raw/verify/100-api-cache-and-ui-blocked.md` | bốn nguyên nhân "blocked" tách biệt với field path (§1); `_MODELS_CACHE` theo state stamp — cold 98.43 s → 67.28 s sống, warm 0.026/0.017 s (§2, live §1); `statistics.query_count` 6003→5 (§3); `subset_provenance` byte-identical hai thứ tự gọi, sha256 `bb7406d6…` (§4); R@1 phục vụ từ derived record (§5); cohort save→reopen→export HTTP **và** browser (§6, §6b) |
| `raw/verify/101-metrics-r1-and-fallback.md` | R@1: định nghĩa + cùng mẫu số với R@10/50 (§1), 7 test biên (§2), backfill 3 guard + số suy ra (tgcir 6.3135099117108116 / 1.949025487256372 / 0.08329168748958854; clvc 1.4106050305914344 / 0.10197144799456152 / `null`) (§3), 136/136 digest artifact **không đổi** (§7); partition encode + cache key (§5, §6) |
| `raw/verify/102-queue-gpu-and-blocker-inventory.md` | **21 blocker** inventoried: 4 nhãn cũ nay SAI (§7.1) + 17 còn mở (§7.2), mỗi dòng nêu đúng thứ thiếu; `gpu_may_be_held` **GIỮ** với chứng minh hai đường (§1, §2) và giới hạn release là thủ tục (§4); backoff PASS (§5); 4 test queue thật (§6) |
| `raw/verify/103-image-only-plumbing.md` | `clvc_image_encoder_only` được khai báo (`ABLATION_CONDITIONS`, không đụng `CONDITIONS`) và có đường qua runner/metrics (§1-§3); gap triple giải theo **vai** + `gap_conditions`/`gap_role_reason` (§4); ba `run_id` khác nhau theo danh sách điều kiện (§6); `--conditions` CLI + `_plan` theo danh sách thật (§8.1, §8.2); kỳ vọng của run mới (§8.5) |
| `raw/verify/104-clvc-full-image-only.md` | **run CLVC mới** `da0a2162be0e159a1a456cec`: `PARTIAL`, 6003 q, ba điều kiện `mm`/`text`/`clvc_image_encoder_only` (§1, §2); `unscored_count 2` mỗi điều kiện, resolver `6001 routed / 2 unscoped` (§3, §4); gap **`0.33061712371309204`** trên 6001, tự kiểm tới 0 khớp (§5); R@1/10/50 + tự tính lại từ ranks (§6); backfill từng **từ chối** run này vì `METRIC_BLOCKS` (§6.1), nay đã vá bởi `b887c7e`; `missing_query_count 6` = 2 × 3 manifest (§7); `partition: canonical_call_window`, batch 32 (§8); before/after với `eab10106` (§9); `7:2179` = 1580/4549/2100, `7:4066` = 59/287/6054, `7:3038` null cả ba (§10); API 200/22 514 B (§11); controller phải là `ctl312h` (§0.1)
| `raw/verify/90a-condition-degradation-fix.md` | `ABLATION_UNSUPPORTED` cho `image_only_empty_text` của CLVC (`text_model.py:84-86`); vị từ dùng chung `mode_supported`/`per_mode`; runner hỏi adapter (`runner.py:1920-1925`); sửa `Corrections.correct` cho text rỗng của TG-CIR |
| `raw/verify/91-browser-smoke-after-cutover.md` | bundle phục vụ `assets/index-DYYSN7oG.js` (sha256 `74e4b4f8…`) sau cutover — tên **lịch sử**, không phải bundle hiện tại (xem §9); deep link, run picker 20 option, query `7:1242`, Top-K 10 vs 50 ở 2560×1440 |
| `raw/verify/94-backend-env-cache-restart.md` | `file_settings()` lazy nên backend **không** cũ (đo (c) vs (d)); counterfactual stale; **không** restart; MainPID **lúc đó** `1221379` (nay `1476678`, §0.1); quy tắc `models.yaml` sửa-được vs `workbench.env` phải restart |
| `raw/verify/95-external-asset-unblock.md` | ResNet-50 `19c8e357…` tải và verify prefix; `main._MODELS_CACHE` **không khoá** nên `/api/models` phục vụ tài liệu cũ — **đã vá**, xem `100` §2; `POST /api/jobs` tính lại nên thấy ngay; addendum hợp đồng CLIP `SOURCE_HASH_AVAILABLE` |
| `raw/verify/96-staged-path-identity.md` | staged path phải giữ **stem** của ảnh reference (`<stem>.png`) không phải qid; 3/6003 query chỉ resolve được bằng stem; smoke `1eee4c0a` ba điều kiện `COMPLETED`; digest `corrected_digest` từng điều kiện |
| `raw/verify/96-staged-path-and-tgcir-worker-exit.md` | chế độ hỏng thứ hai `adapter_worker_exited exit_status=1` (§2); guard first-job đã ship + repro CPU (§2.4, §3); latch phân tích **không** ship (§5); §6.2 lệch 0.0138 vs 0.0027 s/img (~5×) chưa giải thích; §6.4 ghi run `1366364` **lúc đó đang chạy** — nay đã thoát và đã có `-retry-2/full.json` (`98` §1) |
| `raw/verify/97-api-surface-tgcir-run.md` | **smoke** `1eee4c0a` (5 query): 61/61 trường artifact↔API khớp; ba điều kiện `COMPLETED`, `unscored` 0, `topk_export` sâu 100; gap `0.4417698538002782`; R@10 20/0/0, R@50 40/0/0; `target_rank` 23/207/3697; R@1 vắng ở artifact smoke (§7a `statistics.query_count` 6003 và §7b `subset_provenance` 20 053 vs 20 841 B — **cả hai đã vá**, xem `100` §3, §4) |
| `raw/verify/87-clvc-composition.md` | parity CLVC **5 query × 20 ảnh**: bốn trường `0.0`, masked score 8.94e-08/1.19e-07/1.79e-07; hạng giống hệt; đồng nhất thức score 1.1920929e-07 (1 ULP); batch 1 vs N **khác** (1.4638714492321014e-02) và native tái hiện y hệt; fallback `_encode_queries` **đã vá** cho run sau `5c6c781` (`101` §5, §6) |
| `raw/verify/88-tgcir-composition.md` | parity TG-CIR 12 mảng `0.0`, hạng giống; batch 1 vs N `2.898e-06`; không kế thừa mask reference (6/15 query reference là top-1 khi bỏ mask); correction đúng một lần + digest |
| `/tmp/audit-facts/05-live.txt` (ngoài repo) | bản chụp §0: `git rev-parse HEAD`/`origin`, `git status --porcelain`, `systemctl --user show`, `pgrep -af composition_eval.py`, `ps` PID `1366364`, handle `.tmp` đang mở, `nvidia-smi` |

---

## 7. Tổng hợp bốn phán quyết

| # | hạng mục | phán quyết | lý do (1 câu) |
|---|---|---|---|
| 1 | Phục hồi dịch vụ web | **ĐẠT MỘT PHẦN** | Cutover thật và phục hồi sau SIGKILL thật đã đo (≈2.28 s), nhưng ca `systemctl stop` trên unit thật và worst-case `TimeoutStopSec` chưa đo. |
| 2 | Dọn tiến trình con | **ĐẠT MỘT PHẦN** | Backoff, crash-loop, `jobs.py` recovery, `_MODELS_CACHE`, `query_count`, `subset_provenance`, R@1, partition encode và cohort round-trip **đã vá và đo** (§5.1), nhưng guard `gpu_may_be_held` vẫn **từ chối** dispatch và đường release của nó là **thủ tục**, không phải kiểm chứng thiết bị (§5.2 mục 9). |
| 3 | Đối chiếu với upstream | **ĐẠT MỘT PHẦN** | Parity ở mức **mẫu 5 query / 20 ảnh** đạt trên mọi trường (rows `0.0`, hạng giống hệt), nhưng **chứng minh parity quy mô Full không tồn tại**, và con số composition Full `23.3383` **không phải** parity proof (native macro 49.7208 là protocol khác). |
| 4 | Kết quả composition trên Full | **ĐẠT MỘT PHẦN** | Cả hai model đều có run Full: TG-CIR `7b1fa85007bffdab931f7f19` **`COMPLETED`** trên 6003 query, gap ba điều kiện protocol `0.43853620852911723`; CLVC có **hai** run — `eab10106…-retry-2` (`PARTIAL`, `image_only_empty_text` là `ABLATION_UNSUPPORTED`, 119 query không chấm, gap `null`) và `da0a2162be0e159a1a456cec` (`PARTIAL`, 6001/6003 chấm được, gap **`0.33061712371309204`** trên 6001 tính được lần đầu; cohort caption-free 5884 cho **`0.33400015944039774`**) — nhưng gap đó chạy trên **profile biến thể đã khai báo**: vai ảnh do `clvc_image_encoder_only` điền, **không phải** `image_only_empty_text`, nên **điều kiện thứ ba của protocol vẫn chưa được đo cho CLVC** và `0.3306`/`0.3340` **không** được đặt chung cột với `0.4385` của TG-CIR. Thêm một giới hạn nữa: cohort 6001 có **117** query mà nhãn routing lấy từ `(stem, caption)` — nên nếu một khẳng định đòi "không edit text nào chạm routing" thì phải dùng `0.33400015944039774` trên 5884, và run đã công bố **không** tự ghi điều đó (§5.2 mục 12). |

---

## 8. Bảng artifact/run — mọi `run_id` xuất hiện trong báo cáo

Trạng thái đọc từ `workbench/artifacts/composition/summary/<dir>/full.json` tại HEAD
`a0c08c0e658bb9ece6112352f4b2de90e07d3ad9` (**`== origin`**), đọc lại **2026-11-12T10:0x+07**. `smoke` = `smoke.limited`; "—" = không phải run composition.

| run_id / attempt | model | `execution_status` | query | smoke | nghĩa một dòng | nguồn |
|---|---|---|---|---|---|---|
| `7b1fa85007bffdab931f7f19` | `tgcir` | **`COMPLETED`** | 6003 | `false` | **TG-CIR Full duy nhất**, 3 điều kiện `COMPLETED`, gap `0.43853620852911723` | `98` §2; §4.1b |
| `1eee4c0adbc93f7708205a5e` | `tgcir` | `COMPLETED` | 5 | `true` | smoke 5 query, `gallery_count` vẫn 74381; gap `0.4417698538002782` trên 5 qid | `97` §1; §4.1 |
| `6396454ce29c2bea9ebfbe00` | `tgcir` | `COMPLETED` | 5 | `true` | smoke trước đó, 3 điều kiện `COMPLETED` — không dùng làm metric | summary dir |
| `9f115afeb58eb1adf75d88ba` | `tgcir` | `PARTIAL` | 6003 | `false` | Full **hỏng**: `text_only_black_image` và `image_only_empty_text` = `MODEL_UNAVAILABLE` | summary dir; `90` §4.3 |
| `9f115afeb58eb1adf75d88ba-retry-1` | `tgcir` | `PARTIAL` | 6003 | `false` | retry của lần trên, cùng dạng; nơi thấy `adapter_worker_exited exit_status=1` | `96` §2 |
| `eab10106b2df8e97fb7a5ce5` | `clvc_net` | `PARTIAL` | 6003 | `false` | CLVC Full **lần đầu**, 4489 s, VRAM 2897 MiB (số VRAM **chỉ** thuộc run này) | `90` §4.1 |
| `eab10106b2df8e97fb7a5ce5-retry-1` | `clvc_net` | `PARTIAL` | 6003 | `false` | retry thứ nhất, cùng metrics, bị `-retry-2` thay thế | summary dir |
| `eab10106b2df8e97fb7a5ce5-retry-2` | `clvc_net` | **`PARTIAL`** | 6003 | `false` | **CLVC Full có hiệu lực**: 1867.2 s, **`0.0202 s/ảnh` đã rút** (gallery cache; `104` §9.1), gap `null` (`no queries have exact ranks in all three conditions`), 5884/6003 mỗi điều kiện | `98` §1; §4.4 |
| `889bc330d60e8cbab956944f` | `clvc_net` | `PARTIAL` | 5 | `true` | smoke; `image_only_empty_text` = `MODEL_UNAVAILABLE` (trước khi có `ABLATION_UNSUPPORTED`) | summary dir |
| `d541e40cde1a949026f98854` | `clvc_net` | `PARTIAL` | 5 | `true` | smoke; `image_only_empty_text` = `ABLATION_UNSUPPORTED` | summary dir |
| `10173dc45dea8c90fd2a284b` | `limn` | `PARTIAL` | 6003 | `false` | run LIMN so sánh: `ambiguous_count: 119`, `PARTIAL` cùng lý do `unresolved:annotation_names_two_categories` | `90` §4.1; §4.3 |
| `3f86f58483b368377e2c151e` | `limn` | `COMPLETED` | 8 | `true` | smoke LIMN (8 query) | summary dir |
| `9e4918a6b5c243412cbc58c9` | `limn` | `PARTIAL` | 6003 | `false` | run LIMN khác, cùng lớp `PARTIAL` | summary dir |
| `4bc9893a977e501b82378e76` | `hint` | `COMPLETED` | 6003 | `false` | Full có `candidate_topk.depth = 50` — ca kiểm UI "menu không mời 100" | `100` §8 |
| `ad477312aa57c1b9114f9e51` | `tme` | `COMPLETED` | 6003 | `false` | Full **không** export Top-K → `TOPK_EXPORT_NOT_REQUESTED` | `100` §8 |
| `e0dd04a46ef7a58334b9e353` | `habit` | `FAILED` | 6003 | `false` | `FAILED` vì interpreter `airknow-adapted` 3.11.15 ≠ 3.8.10 — một trong 5 ca chạy lại | `102` §7.2 `RERUN-5` |
| `62e362bf911fb5f97c6e497b` | fixture | — | — | — | `run_id` của 2 điều kiện, chỉ để chứng minh danh sách điều kiện vào identity | `103` §6 |
| `8ef7596c0e1fd42a1910e766` | fixture | — | — | — | `run_id` của protocol triple | `103` §6 |
| `e877a310f9e70ea9d6df29af` | fixture | — | — | — | `run_id` khi thay điều kiện ảnh bằng `clvc_image_encoder_only` | `103` §6 |
| `da0a2162be0e159a1a456cec` | `clvc_net` | **`PARTIAL`** | 6003 | `false` | **run CLVC dưới profile khai báo**: ba điều kiện `mm`/`text`/`clvc_image_encoder_only` (không có `image_only_empty_text`), 6001/6003 chấm được, gap `0.33061712371309204` trên cohort 6001 (A) và **`0.33400015944039774`** trên cohort caption-free 5884 (B) — gap ba điều kiện **đầu tiên** của CLVC, **biến thể**, không phải gap protocol; 117/6001 nhãn routing dùng `(stem, caption)` | `104` §1-§7; `99` §4.3; §4.4b |
| `1366364` / `1366356` | — | — | — | — | **PID** (không phải run_id) của lần chạy `-retry-2`; đã thoát | `98` §1 |
| `1477261` / `1477254` | — | — | — | — | **PID** của run profile mới (controller / wrapper), đã thoát khi run xong; run của chúng là `da0a2162…` | §0.1; `104` §0.2 |

Ba `run_id` fixture (`62e362bf…`, `8ef7596c…`, `e877a310…`) là **CPU fixture**, không phải run GPU và
không có metric công bố — chúng ở đây chỉ để chứng minh danh sách điều kiện được hash vào identity.
`da0a2162be0e159a1a456cec` **đã** có `summary/` (17 956 B, mtime `09:15:18.964 +0700`) nên là một run
công bố được; `eab10106…-retry-2` và nó là **hai** run khác nhau, khác `conditions` trong identity.

---

## 9. Bảng lệnh kiểm chứng — lệnh dùng để kiểm từng khẳng định

Chạy tại `/home/heheboiz/data/cir`, HEAD `a0c08c0e658bb9ece6112352f4b2de90e07d3ad9` (**`== origin`**,
0 behind / 0 ahead — chiến dịch đã được push, 36 commit kể từ `82cb1044`). Không lệnh nào ghi vào
artifact, không lệnh nào chạm unit/GPU/lock. Các giá trị sống dưới đây lặp lại **§0.1**, là nguồn duy
nhất; nếu hai chỗ lệch nhau thì §0.1 đúng. Cột cuối ghi rõ nguồn của kết quả: **đo lại** = chạy trong lượt tích hợp này
(2026-11-12T08:4x-08:5x+07); **theo `NN`** = chép từ file bằng chứng ghi kèm, không chạy lại (một số
selection cần GPU hoặc lâu hơn ngân sách của lượt này).

| khẳng định | lệnh | kết quả mong đợi / đã thấy |
|---|---|---|
| HEAD và khoảng cách với `origin` (§0.1) | `git rev-parse HEAD` · `git rev-parse origin/research/cir-failure-workbench` · `git rev-list --left-right --count origin/research/cir-failure-workbench...HEAD` · `git rev-list --count 82cb1044..HEAD` | **đo lại:** `a0c08c0e658bb9ece6112352f4b2de90e07d3ad9` · cùng sha · `0<TAB>0` · `36` — đã push |
| Hai unit sống đúng MainPID (§0.1, §1.5) | `systemctl --user show -p MainPID -p ActiveEnterTimestamp -p NRestarts cir-workbench-backend.service cir-workbench-frontend.service` | **đo lại:** backend `1510910` / `Thu 2026-11-12 09:34:21 +07` / `0` · frontend `1221148` / `Wed 2026-11-11 23:48:44 +07` / `0` |
| Không còn run composition nào chạy (§0.1, §4.4b) | `pgrep -af composition_eval.py` | **đo lại:** rỗng `(none)` — mọi run đã thoát |
| GPU đang bận bao nhiêu (§0.1) | `nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader` · `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader` | **đo lại:** `… 373 MiB, 16303 MiB` · không có compute app |
| TG-CIR Full là `COMPLETED` 6003 (§4.1b, §7) | `python3 -c "import json;d=json.load(open('workbench/artifacts/composition/summary/tgcir_tgcir_archive_unresolved_7b1fa85007bffdab931f7f19/full.json'));print(d['execution_status'],d['query_count'],d['smoke'],d['metrics']['composition_gap_ndcg'])"` | `COMPLETED 6003 {'label': None, 'limited': False, 'query_limit': None} 0.43853620852911723` |
| CLVC Full là `PARTIAL` và gap `null` có lý do (§4.4, §7) | cùng dạng trên `…/clvc_net_author_drive_unresolved_eab10106b2df8e97fb7a5ce5-retry-2/full.json` | `PARTIAL 6003 … None no queries have exact ranks in all three conditions` |
| Run Full được phục vụ đúng qua API (§4.1b, §4.4) | `curl -s "http://127.0.0.1:8000/api/composition/runs/7b1fa85007bffdab931f7f19?subset=full"` (và `…/eab10106b2df8e97fb7a5ce5?subset=full`) | `200`, `72958 B` / `22011 B`, `COMPLETED` / `PARTIAL`, `recall_at_1_multimodal` `6.3135099117108116` / `1.4106050305914344` |
| R@1 đến từ derived record, không từ `full.json` (§4.1b, §5.1 mục 6) | `python3 -c "import json;d=json.load(open('workbench/artifacts/composition/derived/recall_at_1/clvc_net_eab10106b2df8e97fb7a5ce5-retry-2+tgcir_7b1fa85007bffdab931f7f19_full.json'));[print(r['attempt'],[(c,(m or {}).get('recall_at_1')) for c,m in r['conditions'].items()]) for r in d['runs']]"` | `7b1fa850…` 6.3135099117108116 / 1.949025487256372 / 0.08329168748958854; `…-retry-2` 1.4106050305914344 / 0.10197144799456152 / `None` |
| R@1 có trong **đúng một** artifact đã công bố (§4.1, §5.1 mục 6) | `grep -L '"recall_at_1"' workbench/artifacts/composition/summary/*/full.json \| wc -l` · `ls workbench/artifacts/composition/summary/*/full.json \| wc -l` · `grep -l '"recall_at_1"' workbench/artifacts/composition/summary/*/full.json` | **đo lại:** `35` và `36` — tức **1/36**. File duy nhất mang key là `clvc_net_author_drive_unresolved_da0a2162be0e159a1a456cec/full.json` (`metrics.multimodal.recall_at_1 = 1.3997667055490752`). **Điều kiện:** chỉ run được chấm **sau** khi `metrics.py` có `recall_at_1`; hai run Full cũ (tgcir `7b1fa850`, clvc `eab10106…-retry-2`) **không** có key này trong `full.json` của chúng và R@1 của chúng chỉ tồn tại ở file derived |
| Smoke không còn báo `query_count` 6003 (§5.1 mục 4) | `curl -s "http://127.0.0.1:8000/api/composition/runs/1eee4c0adbc93f7708205a5e/queries?subset=full&condition=multimodal&limit=1&top_k=10"` | `statistics.query_count: 5`, ba denominator `5`, `statistics_whole_run.query_count: 5` |
| 117 + 2 tách đúng (§4.3, §5.1 mục 9) | `workbench/envs/ctl312h/bin/python -m pytest workbench/tests/test_clvc_scope_and_image_only.py -p no:cacheprovider --basetemp=/tmp/r9 -q` | **đo lại:** `15 passed in 33.63s` (test `test_the_119_split_into_117_recovered_and_2_genuinely_ambiguous` nêu tên `7:3038`, `7:3790`) |
| R@1 đúng định nghĩa và đúng biên (§5.1 mục 6) | `… -m pytest workbench/tests/test_metrics_recall_at_1.py workbench/tests/test_metrics_backfill_recall_at_1.py -q` | **đo lại:** `7 passed` + `4 passed` (backfill 3 guard) |
| Partition encode được ghi và khoá cache (§5.1 mục 7) | `… -m pytest workbench/tests/test_runner_feature_partition.py workbench/tests/test_runner_batch_provenance.py -q` | **đo lại:** `6 passed` + `2 passed` |
| `_MODELS_CACHE` hết cũ, gate vẫn đỏ đúng chỗ (§5.1 mục 3) | `… -m pytest workbench/tests/test_api_models_cache_staleness.py -q` | **đo lại:** `3 passed in 154.22s` |
| `query_count` / `subset_provenance` / R@1 overlay (§5.1 mục 4, 5) | `… -m pytest workbench/tests/test_composition_api_contracts.py -q` | **đo lại:** `7 passed in 1.65s` |
| `gpu_may_be_held` vẫn từ chối và release đúng thủ tục (§5.2 mục 9) | `… -m pytest workbench/tests/test_job_gpu_guard_through_queue.py -p no:cacheprovider --basetemp=/tmp/r9q -q` | **đo lại:** `4 passed in 0.85s` |
| `--conditions` chấp nhận/từ chối đúng danh sách (§4.4) | `… -m pytest workbench/tests/test_runner_cli_conditions.py -q` | **đo lại:** `18 passed` |
| Gap triple giải theo vai (§4.4) | `… -m pytest workbench/tests/test_runner_gap_identity.py workbench/tests/test_runner_gap_roles.py -q` | **đo lại:** `3 passed` + `11 passed` |
| `TESTED_SOURCE_SHA` **đã được nâng** (§5.2 mục 6) | `grep -n "TESTED_SOURCE_SHA" workbench/docs/reviews/FINAL_PC_RELEASE_VERDICT.md workbench/docs/PC_PILOT_HANDOFF.md` · `git diff --stat 96d86010e85048d82f2dacf26e17d7b7206d31ab HEAD -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'` | Pin nay là **`117fc17220d0e35651c6778c1afb62e271c8d944`**, sweep **`3123 passed, 20 failed, 47 skipped = 3190 collected`**, wall **`3586.55 s`**, record `raw/verify/106-sweep-117fc172.md`; guard `test_handoff_commands.py` **8 passed**. Khoảng cách từ pin **CŨ** `96d86010…` tới HEAD: **29 file, +3505 / −246** — đó là khối lượng mà lần nâng pin này đã phủ. Triple trên là **kết quả đo của sweep tại `117fc172`** và **không** bị viết lại. **Chưa đóng:** **20 test vẫn fail** trong sweep đó, gồm **18 id baseline đã ghi** (blip ×4, `circus_cli`, `tme_lifecycle` ×2, `tme_precision`, `f08_dcnet`, `handoff_commands` SHA guard, `release_blocker_gpu_records` ×7, `prepare_model`) và **2 id chưa từng nằm trong baseline 18** (`test_doctor.py::test_dcnet_runtime_asset_blocks_after_preparation_is_present`, `test_evaluate_models.py::test_dcnet_audited_command_reports_non_command_blockers`) — cả hai cũng fail ở pin trước (`106` §3, §4). **Một trong 20 id đó nay đã xanh:** `test_handoff_commands.py::test_no_release_document_points_at_a_superseded_source_sha` đỏ **vì pin chưa nâng**, và pass sau `c998504` (§5.2 mục 6). Vì vậy **19 fail / 3124 passed / 47 skipped** là con số **suy ra** cho HEAD hiện tại, **không** phải một lượt đo — đừng trích nó như kết quả sweep |
| Guard mở rộng: `git checkout` **và** `must print` được kiểm trong **MỌI** release document (§5.2 mục 6) | `workbench/envs/ctl312h/bin/python -m pytest workbench/tests/test_handoff_commands.py -q -p no:cacheprovider --basetemp=/tmp/pin_rep` · `sed -n '/^RELEASE_DOCS/,/])/p' workbench/tests/test_handoff_commands.py` | **đo lại:** `8 passed in 13.36s`; `RELEASE_DOCS` gồm **5** file (`PC_PILOT_HANDOFF.md`, `PC_ONE_COMMAND_GUIDE.vi.md`, `FINAL_PC_RELEASE_VERDICT.md`, `FINAL_DEFECT_LEDGER.md`, `AUTOMATION_RUNBOOK.md`), và test lặp qua **từng** file cho cả `^git checkout <sha>` lẫn `must print <sha>` (`test_handoff_commands.py:505-535`). **Bite proof** (`106` §8.3): regress verdict về `git checkout 96d86010…` → `FAILED … tells the operator to check out 96d86010…, whose PRODUCTION source differs from the tested source 117fc172…`; sửa lại → `1 passed`; cả file `8 passed`. Trước khi mở rộng, extraction chỉ đọc handoff nên verdict sai vẫn xanh | `grep -n "would silently change the shape" -B3 -A1 workbench/backend/storage.py` · `grep -n "fieldnames" workbench/backend/main.py` · `grep -n "header = \"| Run" workbench/backend/main.py` | `storage.py:339-342`; `main.py:1530` (CSV lấy khoá từ `metrics[0]`) và `:1535` (header Markdown cố định, không có cột reference) |
| Run mới CLVC có gap ba điều kiện (§4.4b, §7) | `python3 -c "import json;d=json.load(open('workbench/artifacts/composition/summary/clvc_net_author_drive_unresolved_da0a2162be0e159a1a456cec/full.json'));m=d['metrics'];print(d['execution_status'],d['query_count'],d['conditions'],m['composition_gap_ndcg'],m['gap_cohort_count'],m['gap_conditions'],m['image_only']['condition'])"` | **đo lại:** `PARTIAL 6003 ['multimodal', 'text_only_black_image', 'clvc_image_encoder_only'] 0.33061712371309204 6001 [ba tên như trên] clvc_image_encoder_only` |
| Run mới phục vụ đúng qua API (§4.4b) | `curl -s -o /tmp/r104.json -w "%{http_code} %{size_download}\n" "http://127.0.0.1:8000/api/composition/runs/da0a2162be0e159a1a456cec?subset=full"` | **đo lại:** `200 22514` |
| `7:2179` được chấm thật ở điều kiện ảnh, Top-K sâu 100 (§4.3, §4.4b) | `curl -s "http://127.0.0.1:8000/api/composition/runs/da0a2162be0e159a1a456cec/queries?subset=full&condition=clvc_image_encoder_only&query_id=7:2179&top_k=100&limit=1&offset=0"` | **đo lại:** `200`, 14 695 B, `target_rank 2100`, `candidates.depth 100`, `top_results` dài **100** |
| LIMN cũng để `7:3038`/`7:3790` unscored (§4.3) | `python3 -c "import json;d=json.load(open('workbench/artifacts/composition/runs/10173dc45dea8c90fd2a284b/ranks/limn_base_iter0_all_categories.json'));r=d['ranks'];[print(c,{q:r[c].get(q) for q in ('7:3038','7:3790','7:2179','7:4066')},sum(1 for v in r[c].values() if v is not None)) for c in r]"` | **đo lại:** cả ba điều kiện `5884` non-null / `119` null; bốn qid đều `None` ở cả ba |
| Hai cohort gap A/B khớp file derived (§4.4b) | `python3 -c "import json;d=json.load(open('workbench/artifacts/composition/derived/category_routing_inputs/clvc_net_da0a2162be0e159a1a456cec_full.json'));print(d['routing']['resolver_key'],d['routing']['loader_scoped_count'],d['routing']['resolver_routed_count'],d['routing']['still_unscoped_count'],d['routing']['replay_agrees_with_artifact']);[print(c,d[c]['gap_cohort_count'],d[c]['composition_gap_ndcg'],d[c]['caption_free'],d[c].get('reproduces_published')) for c in ('cohort_a_all_scored','cohort_b_loader_scoped')]"` | **đo lại:** `reference_stem+edit_caption 5884 117 2 True`; `cohort_a_all_scored 6001 0.33061712371309204 False True`; `cohort_b_loader_scoped 5884 0.33400015944039774 True None` |
| Run đã công bố **không** mang khối routing (§5.2 mục 12) | `python3 -c "import json;print('category_routing_inputs' in json.load(open('workbench/artifacts/composition/summary/clvc_net_author_drive_unresolved_da0a2162be0e159a1a456cec/full.json')))"` | `False` — summary bất biến, số đo nằm ở artifact derived riêng |
| Bundle đang phục vụ (§1.5) | `curl -s http://127.0.0.1:5173/ \| grep -o 'assets/index-[^"]*\.js'` · `sha256sum workbench/frontend/dist/assets/index-DDviexaW.js` · `curl -s http://127.0.0.1:5173/assets/index-DDviexaW.js \| sha256sum` | **đo lại:** trang phục vụ trỏ `assets/index-DDviexaW.js`; bản trong `dist/` và bản qua proxy **cùng** sha256 `58eed6d48640fa1f93d2a64a96d87812a044a3b07b324fb872f6126162bf5369` (392 261 B, mtime `09:34:18.594`). `index-DYYSN7oG.js` và `index-Lne9CV3w.js` là tên **lịch sử**, không phải hiện tại |
| Backfill nhận run biến thể **sau** `b887c7e` (§4.4b giới hạn 2) | `workbench/envs/ctl312h/bin/python workbench/scripts/backfill_recall_at_1.py --run clvc_net:da0a2162be0e159a1a456cec; echo rc=$?` | **đo lại:** `rc=0`, record có `multimodal` 1.3997667055490752 / `text_only_black_image` 0.11664722546242293 / `scored_count 6001`, khớp `full.json`. Trước `b887c7e` thì exit 1 với `clvc_image_encoder_only.scored_count: recomputed 6001 vs published None` (`104` §6.1) |

Chỗ nào lệnh **không** chạy được ở đây (cần GPU, cần run mới, hoặc cần một artifact chưa tồn tại) thì
báo cáo ghi `chưa đo` tại chỗ đó thay vì thay bằng một con số suy ra.
