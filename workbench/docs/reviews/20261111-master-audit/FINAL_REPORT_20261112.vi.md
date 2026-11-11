# BÁO CÁO GIAO HÀNG — audit tổng 20261111 (viết 2026-11-12)

Đây là báo cáo **giao hàng**, không đo gì mới. Mọi con số dưới đây được **chép nguyên** từ một file
bằng chứng có đường dẫn trích ngay trong dòng. Chỗ nào không có số trong file bằng chứng thì ghi
`chưa đo`, không suy diễn.

Phần "số liệu sống" (git, systemd, `pgrep`, GPU) được chụp bằng lệnh ghi rõ trong §0; bản chụp nằm ở
`/tmp/audit-facts/05-live.txt` (ngoài repo, chỉ để đối chiếu).

Bốn phần dưới đây là **bốn phán quyết riêng biệt**, không gộp.

---

## 0. Số liệu sống tại thời điểm viết (2026-11-12T04:43+07, tức 2026-11-11T21:43Z)

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

`git status --porcelain` (nguyên văn):

```
?? CLEANUP_REPORT.md.save
?? workbench/docs/reviews/20261111-master-audit/raw/verify/91-browser-smoke-after-cutover.md
?? workbench/docs/reviews/20261111-master-audit/raw/verify/97-api-surface-tgcir-run.md
?? workbench/docs/reviews/20261111-master-audit/web_evidence/e2e_post_cutover/
```

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
`proxied_comp`; bundle phục vụ không đổi (`assets/index-DYYSN7oG.js` trước và sau)
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

Trạng thái hiện tại khớp: backend `MainPID=1221379 NRestarts=1 ActiveEnterTimestamp=Wed 2026-11-11
23:48:58 +07`, frontend `MainPID=1221148 NRestarts=0 …23:48:44 +07` (§0).

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

**Phán quyết: ĐẠT MỘT PHẦN** — backoff, crash-loop, `jobs.py` và guard GPU đều đã vá và đo; riêng
cache không khoá của `/api/models` vẫn có thể phục vụ tài liệu cũ.

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

### 2.4 Env-cache: backend **không** bị cũ (đã đo), nhưng `/api/models` thì có

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

Nhưng **`/api/models` có cache module không khoá**: `main._MODELS_CACHE` là list mức module, không có
key nào (`main.py:170-176`), nên `GET /api/models` phục vụ tài liệu **trước khi sửa** suốt đời process
(`raw/verify/95-external-asset-unblock.md` §"Did the API pick up the registry edit without a restart?").
Đo được: listing vẫn báo file ResNet-50 của `clvc_net` là missing dù file đã có và đã verify; trong khi
`POST /api/jobs` (tính lại `preflight_blockers` mỗi request) **đã** bỏ blocker asset của `clvc_net`.
Đối lập: `registry._load_yaml_cached` / `load_auxiliary_assets` key theo `(path, size, mtime_ns)` nên
thấy ngay (`raw/verify/95-…md`, cùng mục). Quy tắc vận hành rút ra: `models.yaml` — sửa và chạy tiếp;
`workbench.env` — sửa rồi restart backend (`raw/verify/94-…md` §"Why an edit to `workbench.env` still
requires a restart").

### 2.5 Chưa đo

- Có orphan evaluator thật nào từng sống sót qua một cú crash backend thật hay không: **chưa đo** —
  guard là phòng ngừa, chưa quan sát được orphan cụ thể trên host này (`raw/verify/84-…md` §6).
- Crash loop thật trong production có từng chạm `failures ≥ 1025` hay không: **chưa đo** — không tìm
  thấy log nào như vậy; đây là khuyết điểm tiềm ẩn đã chứng minh của hàm, không phải sự cố quá khứ đã
  chứng minh (`raw/verify/84-…md` §6).

---

## 3. Đối chiếu với upstream (native parity)

**Phán quyết: ĐẠT MỘT PHẦN** — parity ở mức mẫu 5 query / 20 ảnh đạt trên mọi trường đo; **không**
phải chứng minh mức metric và chưa có parity ở quy mô Full.

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

Mọi số ở §3 là **parity 5 query / 20 ảnh mỗi category**, **không phải** chứng minh mức metric.
`raw/verify/87-clvc-composition.md` §"Còn chặn" mục 1: "No numbers here are a metric: every measurement
is a 5-query, 20-image parity sample … Full run counts (6016 release queries, 74381-image gallery), the
per-condition aggregate and R@1/10/50 for `clvc_net` remain unmeasured" — ở phạm vi của tài liệu đó.
`raw/verify/88-tgcir-composition.md` §"Còn chặn" mục 4 nói cùng ý cho TG-CIR. Số native của campaign
(`raw/tgcir_run3cat_gpu.json`: macro R@10 49.7208 / R@50 71.1023 / mean 60.4115) **không** bị công
việc này chạm tới và vẫn là mục tiêu so sánh cho một full run sau này (`raw/verify/89-…md` §"Còn
chặn" mục 1).

---

## 4. Kết quả composition trên Full

**Phán quyết: CHƯA ĐẠT** — TG-CIR Full chưa tồn tại; CLVC Full đã có nhưng là `PARTIAL` với một điều
kiện không đo được và `composition_gap_ndcg` là `null`.

### 4.1 TG-CIR: cái đang có là smoke 5 query, **không phải** Full

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

**R@1 không tồn tại ở cả hai phía.** `metrics.<cond>.recall_at_1` là `<ABSENT>` trên artifact **và**
API, nên không có R@1 nào được phục vụ (`raw/verify/97-…md` §1, "R@1, stated as fact").

**Nói thẳng: đây là smoke 5 query, không phải Full run, và recall của 5 query không phải metric công
bố được.** Chính payload mang marker `smoke.limited: true`, `smoke.query_limit: 5`,
`smoke.label: "SMOKE QUERY LIMIT; full released gallery preserved"` trong khi `gallery_count` vẫn là
gallery release đầy đủ `74381` (`raw/verify/97-…md` §2). Trong 60 dòng phục vụ với `include_smoke=true`,
**10** dòng mang `smoke.limited: true` và **50** dòng không; không có `include_smoke` thì run này
**vắng mặt** khỏi 50 dòng (`raw/verify/97-…md` §2). Số **Full** của TG-CIR: **chưa đo** — không có
summary Full nào cho `tgcir` trong bằng chứng của đợt này.

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

### 4.3 CLVC-Net: 119/6003 (1.98 %) query không chấm được — giữ có chủ ý

`raw/verify/90-circus-composition-runs.md` §4.1: **119 trên 6003 query (1.98 %)** không được chấm vì
annotation của release nêu **hai** category cho các qid đó; adapter từ chối đoán và runner ghi lý do
theo từng query (`category_scope_failure_count: 119`). Đây là **chính sách đã khai báo, không phải
khuyết điểm của CLVC**: release tự báo `category_resolution.ambiguous_count: 119` (trên 6003 xét,
5884 resolve).

Nó **khớp chính xác với LIMN**: run LIMN `10173dc4` báo cùng hình dạng — `ambiguous_count: 119`,
`category_scope_failures: {multimodal: 119, text_only_black_image: 119, image_only_empty_text: 119}`,
`PARTIAL` — vì `model_limn.py` cố ý từ chối với `unresolved:annotation_names_two_categories`
(`raw/verify/90-…md` §4.1). Giải chúng bằng suy luận riêng cho CLVC sẽ làm hai model phủ hai tập query
khác nhau và phá tính so sánh chéo, nên **không làm**; `PARTIAL` là trạng thái đúng.

### 4.4 CLVC Full: đã có một run hoàn tất, và một lần chạy lại đang chạy

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

Metric theo điều kiện, mẫu số 6003 (`raw/verify/90-…md` §4.1; đọc lại từ
`workbench/artifacts/composition/summary/clvc_net_author_drive_unresolved_eab10106b2df8e97fb7a5ce5/full.json`):

| condition | scored | coverage | R@10 | R@50 | nDCG_full | MRR |
|---|---|---|---|---|---|---|
| `multimodal` | 5884 | 0.980176578377478 | 7.256968048946295 | 18.303874915023794 | 0.1495349414930845 | 0.03594722775284061 |
| `text_only_black_image` | 5884 | 0.980176578377478 | 1.3936097892590076 | 4.46974847042828 | 0.09954839186850512 | 0.006375664231695533 |
| `image_only_empty_text` | 0 | 0.0 | null | null | null | null |

`composition_gap_ndcg`: **`null`** với lý do `"no queries have exact ranks in all three conditions"`
(`raw/verify/90-…md` §4.1) — nên **không** có gap cho CLVC. `topk_export`:
`{"depth": 100, "requested": true, "status": "TOPK_EXPORT_COMPLETE_PARTIAL", "partial": true,
"missing_query_count": 238, ...}` với 238 = 119 × 2 điều kiện được export (cùng mục).

**Trạng thái lần chạy lại hiện tại — ĐANG CHẠY, chưa xong.** `pgrep` (§0) trả hai tiến trình: PID
`1366364` là `composition_eval.py --models clvc_net --subsets full --topk-depth 100`, PID `1366356` là
`gpu_job.sh` cha. Đo thêm: `ps` cho `1366364` ở `Sl`, `etime 20:23`, 19 thread, ~767 % CPU; nó đang mở
hai handle chỉ-ghi vào
`runs/eab10106b2df8e97fb7a5ce5/candidates/clvc_net_author_drive_unresolved/.multimodal.jsonl.1366364….tmp`
(**15 997 926 B**, mtime `04:40:38`) và `.text_only_black_image.jsonl.1366364….tmp` (**0 B**, mtime
`04:22:26`). Kích thước file thứ nhất **không tăng** giữa hai lần đo `04:40:38` và `04:42:39`, nên
không thể kết luận nó đang ghi liên tục; cái kết luận được là **chưa hoàn tất**: thư mục `ranks/` có
mtime `03:46:27` và các `full.json` hiện có của run_id này là `02:35:43`
(`…_eab10106…/full.json`) và `03:46:30` (`…_eab10106…-retry-1/full.json`) — **tất cả đều trước** lúc
process này start (`04:22:15`), nên lần chạy này chưa ghi lại ranks và chưa có summary mới. Vì vậy
**không** được gán bất kỳ con số Full nào cho nó. Tiến trình này **không bị chạm** trong suốt quá trình
soạn báo cáo.

Không có con số Full nào khác: **chưa đo**.

---

## 5. Khuyết điểm đã tìm ra và **CHƯA** vá

1. **`statistics.query_count` phục vụ `6003` bên cạnh mẫu số từng điều kiện là `5` trên một smoke.**
   `workbench/backend/composition.py:1372` truyền `len(subset_queries)` — kích thước subset `full` của
   release — trong khi khối `conditions` ngay bên cạnh chỉ tính trên các qid thực sự được xếp hạng (5).
   Trong cùng payload, `statistics_whole_run.query_count` là `5` và `query_count` của run-detail cũng là
   `5`. Người đọc sẽ hiểu như thể 6003 query đã được chấm trong smoke này. Artifact **không** có khối
   `statistics`, nên đây không phải mismatch artifact↔API và **không bị chấm là FAIL** trong 97 — ghi ở
   `raw/verify/97-api-surface-tgcir-run.md` §7(a) với mức **OBSERVATION (not scored)**.
2. **`subset_provenance` chỉ xuất hiện sau khi route list làm nóng cache summary dùng chung.** Lần
   fetch detail **lạnh** 20 053 B **không** có key này; hai lần detail sau và lần fetch qua proxy
   20 841 B **có**. Nguyên nhân đọc từ code: `read_summary_file` cache dict đã parse theo path và trao
   **cùng object** cho mọi caller, còn `list_composition_summaries` sửa nó tại chỗ qua
   `decorate_run_rows` — route nào làm nóng cache quyết định block có mặt hay không. Khi có mặt thì nó
   đúng (`consistent: true`) (`raw/verify/97-…md` §7(b)).
3. **Fallback encode theo từng query của runner có thể trộn batch size trong một điều kiện mà không ghi
   lại** (`runner.py` `_encode_queries`, `:700-709`). Với CLVC-Net điều này **đổi feature**, nên một run
   mà feature query được sinh một phần bởi fallback **không so sánh được** với run không dùng fallback
   (`raw/verify/87-clvc-composition.md` §3b, §"Còn chặn" mục 2). Ngược lại, fallback ở đường **SCORING**
   thì **có** ghi: `condition_status.<cond>.fallback_used` xuất hiện trong artifact và API (ví dụ
   `fallback_used: false` ở `raw/verify/97-…md` §1a và `raw/verify/90-…md` §4.1). **Chưa vá.**
4. **Gallery encode của `CLVC-Net` đo được 0.0138 s/ảnh trong run, so với 0.0027 s/ảnh đo độc lập
   (90 §4.2) — lệch ~5×.** Quy cho memmap 609 MB flush trong lúc đo, nhưng **chưa chứng minh**:
   `raw/verify/96-staged-path-and-tgcir-worker-exit.md` §6.2 ghi rõ "Recording the discrepancy rather
   than explaining it away … Both readings are real; they measure different code around the same window.
   Separating them needs per-scope instrumentation … which this section did not do." Ghi nhận là
   **chưa giải thích được**.
5. **Chế độ hỏng thứ hai của TG-CIR: `adapter_worker_exited exit_status=1`, thấy đúng một lần ở
   `9f115afe-retry-1`, không tái hiện được ở quy mô 5 query.** Cơ chế: job đầu tiên tại `model_tgcir.py:653`
   là job **duy nhất** chạy ngoài `try`, nên một `ValueError` ở đó thoát process với exit 1; `_exchange`
   thấy EOF rồi **latch** `_worker_failed` (`model_tgcir.py:317`), khiến mọi batch sau — ở mọi điều kiện
   — short-circuit với cùng chuỗi, và `runner._encode_queries` re-raise vô điều kiện (`runner.py:713-714`)
   nên cả điều kiện chết với 6003 unscored (`raw/verify/96-staged-path-and-tgcir-worker-exit.md` §2).
   **Guard first-job ĐÃ ship** (`be9de58` + hunk 653-663), kèm **repro trên CPU không GPU không weights**:
   trước guard `exit_status: 1, replies: 0`; sau guard `exit_status: 0, replies: 2` với job 2 được phục
   vụ bình thường (`raw/verify/96-…md` §2.4). **Thay đổi latch thì KHÔNG ship**: `raw/verify/96-…md` §5
   ghi rõ nó đã được phân tích, **reverted**, và hành vi latch/stop gốc được phục hồi nguyên văn —
   "A semantics change to `_worker_failed` justified only by a scenario that no longer reproduces is a
   precaution, not a fix". Guard này cũng **chưa có bằng chứng từ run thật**, chỉ có repro CPU, và không
   đường nào của run thật đi vào nó khi 6003/6003 query đều resolve (`raw/verify/96-…md` §"Còn chặn"
   mục 3).

---

## 6. Bảng file bằng chứng

| đường dẫn | chứng minh điều gì |
|---|---|
| `raw/verify/84-evaluator-recovery-and-backoff.md` | `restart_delay(1025)` ném `OverflowError`; cách vá clamp số mũ (`BACKOFF_SATURATING_EXPONENT == 6`) và thang 0.2→8.0; crash loop 6 restart/10 s với gap 0.2/0.4/0.8/1.6/3.2/6.4; `jobs.py` recovery 5 phát hiện + kiểm toán quyền sở hữu kill + guard `gpu_may_be_held`; SIGTERM lịch sử vẫn chưa rõ người gửi |
| `raw/verify/86-systemd-user-services.md` | hai unit user; `MainPID` = uvicorn không wrapper; `Restart=always`/`RestartSec=2`/`StartLimit 60,5`/`KillMode=mixed`/`TimeoutStopSec=30`; ba ca kill (3092 ms / 2881 ms / giữ dừng); cô lập cgroup cho con `setsid` + timeline; frontend 2432 ms và 2381 ms + lý do `mixed` vs `process`; cutover 23:48:4x, MainPID trước→sau, kill thật ≈2.28 s; `/tmp/cutover_rollback.sh` md5 `04fbd5ec…` |
| `raw/verify/89-adapted-runtime-and-gpu-capability.md` | `adapted_runtime` sm_120 cho `clvc_net`/`tgcir`; tier `BLOCKED` → `GPU_VERIFICATION_PENDING` → `GPU_RUNTIME_READY`; bản ghi GPU (capability 12.0, torch 2.7.1+cu128, 16303 MiB); `command_status` giữ `COMMAND_BLOCKED_SOURCE_ENTRYPOINT` |
| `raw/verify/90-circus-composition-runs.md` | CLVC Full `eab10106` §4.1 (PARTIAL, 4489 s, metrics, `gap null`, 119/6003); §4.2 s/img 0.0027 vs 0.0603; §4.3 TG-CIR `9f115afe` mất 2 điều kiện; danh sách "Còn chặn" |
| `raw/verify/90a-condition-degradation-fix.md` | `ABLATION_UNSUPPORTED` cho `image_only_empty_text` của CLVC (`text_model.py:84-86`); vị từ dùng chung `mode_supported`/`per_mode`; runner hỏi adapter (`runner.py:1920-1925`); sửa `Corrections.correct` cho text rỗng của TG-CIR |
| `raw/verify/91-browser-smoke-after-cutover.md` | bundle phục vụ `assets/index-DYYSN7oG.js` (sha256 `74e4b4f8…`) sau cutover; deep link, run picker 20 option, query `7:1242`, Top-K 10 vs 50 ở 2560×1440 |
| `raw/verify/94-backend-env-cache-restart.md` | `file_settings()` lazy nên backend **không** cũ (đo (c) vs (d)); counterfactual stale; **không** restart; MainPID giữ `1221379`; quy tắc `models.yaml` sửa-được vs `workbench.env` phải restart |
| `raw/verify/95-external-asset-unblock.md` | ResNet-50 `19c8e357…` tải và verify prefix; `main._MODELS_CACHE` **không khoá** nên `/api/models` phục vụ tài liệu cũ; `POST /api/jobs` tính lại nên thấy ngay; addendum hợp đồng CLIP `SOURCE_HASH_AVAILABLE` |
| `raw/verify/96-staged-path-identity.md` | staged path phải giữ **stem** của ảnh reference (`<stem>.png`) không phải qid; 3/6003 query chỉ resolve được bằng stem; smoke `1eee4c0a` ba điều kiện `COMPLETED`; digest `corrected_digest` từng điều kiện |
| `raw/verify/96-staged-path-and-tgcir-worker-exit.md` | chế độ hỏng thứ hai `adapter_worker_exited exit_status=1` (§2); guard first-job đã ship + repro CPU (§2.4, §3); latch phân tích **không** ship (§5); §6.2 lệch 0.0138 vs 0.0027 s/img (~5×) chưa giải thích; §6.4 run `1366364` đang chạy |
| `raw/verify/97-api-surface-tgcir-run.md` | 61/61 trường artifact↔API khớp; ba điều kiện `COMPLETED`, `unscored` 0, `topk_export` sâu 100; gap `0.4417698538002782`; R@10 20/0/0, R@50 40/0/0; `target_rank` 23/207/3697; R@1 vắng cả hai phía; `statistics.query_count` 6003 (§7a) và `subset_provenance` 20 053 vs 20 841 B (§7b) |
| `raw/verify/87-clvc-composition.md` | parity CLVC 5 query × 20 ảnh: bốn trường `0.0`, masked score 8.94e-08/1.19e-07/1.79e-07; hạng giống hệt; đồng nhất thức score 1.1920929e-07 (1 ULP); batch 1 vs N **khác** (1.4638714492321014e-02) và native tái hiện y hệt; fallback `_encode_queries` chưa vá |
| `raw/verify/88-tgcir-composition.md` | parity TG-CIR 12 mảng `0.0`, hạng giống; batch 1 vs N `2.898e-06`; không kế thừa mask reference (6/15 query reference là top-1 khi bỏ mask); correction đúng một lần + digest |
| `/tmp/audit-facts/05-live.txt` (ngoài repo) | bản chụp §0: `git rev-parse HEAD`/`origin`, `git status --porcelain`, `systemctl --user show`, `pgrep -af composition_eval.py`, `ps` PID `1366364`, handle `.tmp` đang mở, `nvidia-smi` |

---

## 7. Tổng hợp bốn phán quyết

| # | hạng mục | phán quyết | lý do (1 câu) |
|---|---|---|---|
| 1 | Phục hồi dịch vụ web | **ĐẠT MỘT PHẦN** | Cutover thật và phục hồi sau SIGKILL thật đã đo (≈2.28 s), nhưng ca `systemctl stop` trên unit thật và worst-case `TimeoutStopSec` chưa đo. |
| 2 | Dọn tiến trình con | **ĐẠT MỘT PHẦN** | Backoff, crash-loop, `jobs.py` và guard GPU đã vá và đo, nhưng cache không khoá của `/api/models` vẫn có thể phục vụ tài liệu cũ. |
| 3 | Đối chiếu với upstream | **ĐẠT MỘT PHẦN** | Parity ở mức 5 query / 20 ảnh đạt trên mọi trường (rows `0.0`, hạng giống hệt), nhưng đây không phải chứng minh mức metric và chưa có parity quy mô Full. |
| 4 | Kết quả composition trên Full | **CHƯA ĐẠT** | TG-CIR Full chưa tồn tại (chỉ có smoke 5 query), còn CLVC Full là `PARTIAL` với `composition_gap_ndcg` `null`. |
