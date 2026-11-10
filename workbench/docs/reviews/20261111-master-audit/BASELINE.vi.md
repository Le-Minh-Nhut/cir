# BASELINE — mốc audit 2026-11-11 (workbench CIR)

Mọi số dưới đây kèm đúng lệnh sinh ra nó. Lệnh chạy tại `/home/heheboiz/data/cir`.
Không sửa repo, không commit, không đụng `CLEANUP_REPORT.md.save`.

## 1. Git: branch, HEAD, remote, worktree, 6 commit cuối

Lệnh (một lần, ghi ra `/tmp/gitfacts.txt` rồi đọc lại):

```
{ git branch --show-current; git rev-parse HEAD;
  git rev-parse origin/research/cir-failure-workbench;
  git status --porcelain; git worktree list;
  git log -6 --format='%h %ci %s'; } > /tmp/gitfacts.txt 2>&1
```

Output thô:

```
== branch ==
research/cir-failure-workbench
== head ==
92ed66de4f359ef479da18bf8939723b6f9313e5
== remote ==
92ed66de4f359ef479da18bf8939723b6f9313e5
== status ==
?? CLEANUP_REPORT.md.save
== worktrees ==
/home/heheboiz/data/cir  92ed66d [research/cir-failure-workbench]
/tmp/cir1716             1716e81 (detached HEAD)
/tmp/cirb0               b0f38bd (detached HEAD)
/tmp/wtc                 b0f38bd (detached HEAD)
== log6 ==
92ed66d 2026-11-10 11:44:30 +0700 Report the true final state: three commits, sweep at 68ad75e, R1/R5 closed
d850f98 2026-11-10 10:44:10 +0700 Advance TESTED_SOURCE_SHA to 68ad75e with the serial sweep's own numbers
68ad75e 2026-11-10 09:54:50 +0700 Fix the R5 producer's target DID and allow cited protocol_source correction
3828163 2026-11-10 09:00:31 +0700 Close the E2E repair campaign: gate, protocol labels, LIMN guards, web scope
2e2e2d2 2026-11-10 07:11:38 +0700 Final report reflects 11/15 native targets and 2/15 G1/G2
323be19 2026-11-10 04:43:49 +0700 Two more native evaluations; encoder G1/G2; matrices and gates updated
```

- HEAD local == remote ref: **khớp** (`92ed66de…`).
- Working tree: sạch, chỉ có **một** file untracked của operator là `CLEANUP_REPORT.md.save` — không stage, không đọc ghi.
- Ba worktree phụ đang detached ở `/tmp/cir1716` (1716e81), `/tmp/cirb0` và `/tmp/wtc` (b0f38bd). Chúng **không** phải HEAD; đừng lấy số từ đó.

**Ghi chú về độ trôi của working tree.** Lúc chụp mốc, tree chỉ có `?? CLEANUP_REPORT.md.save`. Ngay sau đó, các worker khác bắt đầu ghi file nên `git status --porcelain` lúc đóng báo cáo đã thành:

```
?? .scratch/
?? CLEANUP_REPORT.md.save
?? raw/
?? scripts/
?? workbench/docs/reviews/20261111-master-audit/
?? workbench/scripts/setup_blocked_model_envs.sh
?? workbench/tests/test_gate_contract_negatives.py
```

Đó là output của 5 worker còn lại, không phải thay đổi của bản baseline này. HEAD vẫn `92ed66d` và không có file tracked nào bị sửa — nên mốc vẫn giá trị. Ai đọc sau phải phân biệt "untracked do chiến dịch" với "untracked của operator" (`CLEANUP_REPORT.md.save`).

## 2. Data/artifact root thực tế

### 2a. `workbench/config/workbench.env` — chỉ key không bí mật (file không chứa secret)

```
cat workbench/config/workbench.env
```

Output thô (đã lược 10 dòng `WORKBENCH_*_PYTHON` chỉ chứa đường dẫn interpreter, xem 2c):

```
CIR_REPO_ROOT=/home/heheboiz/data/cir
CIR_DATA_ROOT=/home/heheboiz/data/cir/data
FASHIONIQ_ROOT=/home/heheboiz/data/cir/data/FashionIQ
WORKBENCH_ARTIFACTS_ROOT=/home/heheboiz/data/cir/workbench/artifacts
WORKBENCH_CHECKPOINT_ROOT=/home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads
WORKBENCH_RESULTS_ROOT=/home/heheboiz/data/cir/workbench/artifacts/results
WORKBENCH_THIRD_PARTY_ROOT=/home/heheboiz/data/cir/workbench/third_party
WORKBENCH_CIRCUS_RELEASE_ROOT=/home/heheboiz/data/cir/workbench/artifacts/circus_release
...
WORKBENCH_RUNTIME_ADAPTATION=sm_120
```

Không có key nào là mật khẩu/token; toàn bộ là đường dẫn + tên interpreter.

### 2b. Root backend resolve lúc chạy

`workbench/backend/operator_config.py:8` đọc `parents[1]/"config"/"workbench.env"`, nên root là cái file trên khai, không phải default. Kiểm bằng cách gọi chính module:

```
workbench/envs/ctl312h/bin/python -c "
import sys; sys.path.insert(0,'.')
from workbench.backend import operator_config as oc
c=oc._CONFIG
for k in ['CIR_REPO_ROOT','CIR_DATA_ROOT','FASHIONIQ_ROOT','WORKBENCH_ARTIFACTS_ROOT',
          'WORKBENCH_CHECKPOINT_ROOT','WORKBENCH_RESULTS_ROOT','WORKBENCH_THIRD_PARTY_ROOT',
          'WORKBENCH_CIRCUS_RELEASE_ROOT']: print(k,'=',getattr(c,k,'<absent>'))"
```

Output thô:

```
CIR_REPO_ROOT = /home/heheboiz/data/cir
CIR_DATA_ROOT = /home/heheboiz/data/cir/data
FASHIONIQ_ROOT = /home/heheboiz/data/cir/data/FashionIQ
WORKBENCH_ARTIFACTS_ROOT = /home/heheboiz/data/cir/workbench/artifacts
WORKBENCH_CHECKPOINT_ROOT = /home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads
WORKBENCH_RESULTS_ROOT = /home/heheboiz/data/cir/workbench/artifacts/results
WORKBENCH_THIRD_PARTY_ROOT = /home/heheboiz/data/cir/workbench/third_party
WORKBENCH_CIRCUS_RELEASE_ROOT = /home/heheboiz/data/cir/workbench/artifacts/circus_release
```

Kiểm process đang phục vụ không override gì ngoài một biến:

```
tr '\0' '\n' < /proc/2820650/environ | grep -E 'WORKBENCH_|CIR_DATA|FASHIONIQ_ROOT'
```
```
WORKBENCH_CIRCUS_RELEASE_ROOT=/home/heheboiz/data/cir/workbench/artifacts/circus_release
```

Biến đó trùng giá trị trong file → **không có lệch root**. `cwd` của cả hai uvicorn là `/home/heheboiz/data/cir` (`ls -l /proc/1227919/cwd /proc/2820650/cwd`).

Ghi chú quan trọng: `WORKBENCH_CHECKPOINT_ROOT` trỏ vào `…/cohort/downloads`, **không** phải `workbench/artifacts/checkpoints`. Hai cây này cùng tồn tại; script nào hardcode `artifacts/checkpoints` sẽ nhìn thấy bộ file khác.

### 2c. Interpreter theo model (đường dẫn, không phải secret)

```
WORKBENCH_ENCODER_PYTHON=…/envs/encoder310-adapted/bin/python
WORKBENCH_LIMN_PYTHON=…/envs/limn310-adapted/bin/python
WORKBENCH_HABIT_PYTHON=…/envs/ilearn39-adapted/bin/python
WORKBENCH_INTENT_PYTHON=…/envs/ilearn39-adapted/bin/python
WORKBENCH_HINT_PYTHON=…/envs/ilearn39-adapted/bin/python
WORKBENCH_CONESEP_PYTHON=…/envs/ilearn39-adapted/bin/python
WORKBENCH_PTHA_MTST_PYTHON=…/artifacts/environments/cir-005ef4e1b02999a5/bin/python
WORKBENCH_TME_PYTHON=…/envs/tme-adapted/bin/python
WORKBENCH_CSMCIR_PYTHON=…/artifacts/environments/cir-005ef4e1b02999a5/bin/python
```

`clvc_net`, `tgcir`, `sprc` **không** có biến interpreter (khớp registry: không khai env).

## 3. Service đang chạy

```
{ pgrep -af serve_workbench.py; echo --uvicorn--; pgrep -af uvicorn;
  echo --ports--; ss -ltnp | grep -E ':(5173|8000|8765)'; } > /tmp/svc.txt
```

Output thô:

```
2820649 workbench/envs/ctl312h/bin/python workbench/scripts/serve_workbench.py --production-frontend --host 127.0.0.1 --backend-port 8000 --frontend-port 5173
--uvicorn--
1227919 workbench/envs/ctl312h/bin/python -m uvicorn workbench.backend.main:app --host 127.0.0.1 --port 8765
2820650 /home/heheboiz/data/cir/workbench/envs/ctl312h/bin/python -m uvicorn workbench.backend.main:app --host 127.0.0.1 --port 8000
--ports--
LISTEN 0 511  127.0.0.1:5173 0.0.0.0:* users:(("node-MainThread",pid=2820663,fd=21))
LISTEN 0 2048 127.0.0.1:8000 0.0.0.0:* users:(("python",pid=2820650,fd=7))
LISTEN 0 2048 127.0.0.1:8765 0.0.0.0:* users:(("python",pid=1227919,fd=7))
```

Ba cổng đều listen trên loopback. **Có hai backend**: 8000 (con của `serve_workbench.py`, đang được web dùng) và 8765 (uvicorn độc lập, PID 1227919). Khi đối chiếu số API phải nói rõ cổng nào; mặc định trong báo cáo này là **8000**.

Frontend bundle đang serve:

```
cat workbench/frontend/dist/index.html
ls -l --time-style=full-iso workbench/frontend/dist/assets/
```

```
<script type="module" crossorigin src="/assets/index-DjAje2d2.js"></script>
<link rel="stylesheet" crossorigin href="/assets/index-DecQnta9.css">
```

```
-rw-rw-r-- 1 heheboiz heheboiz  26552 2026-11-10 08:54:36.404852802 +0700 index-DecQnta9.css
-rw-rw-r-- 1 heheboiz heheboiz 367327 2026-11-10 08:54:36.403659333 +0700 index-DjAje2d2.js
```

Bundle = `index-DjAje2d2.js`, **367 327 byte**, mtime `2026-11-10 08:54:36 +0700`. Đã xác minh file server trả về **đúng** file trên đĩa:

```
curl -s http://127.0.0.1:5173/assets/index-DjAje2d2.js -o /tmp/served.js
sha256sum /tmp/served.js workbench/frontend/dist/assets/index-DjAje2d2.js
```
```
74a9319ed6f05ce0508cf93424d5a0dcd8909e9a0116dcec4e87a1f24d50f867  /tmp/served.js
74a9319ed6f05ce0508cf93424d5a0dcd8909e9a0116dcec4e87a1f24d50f867  workbench/frontend/dist/assets/index-DjAje2d2.js
```

Lưu ý: `workbench/frontend/dist/index.html` **không** được git track (`git ls-files --error-unmatch` → rc=1).

Sửa lại một cách nói sai trước đây: build **tái lập được từ HEAD**, không phải "không tái lập được".
Chỉ `workbench/frontend/dist/` và `workbench/frontend/node_modules/` bị gitignore; còn
`package.json`, `package-lock.json`, `tsconfig.json`, `vite.config.mts` và toàn bộ nguồn `.ts`/`.tsx`
đều **tracked**, và `tsc` (7.0.2) + `vite` đã cài trong `node_modules`. Nên
`npm run build --prefix workbench/frontend` cho lại đúng bundle đó.

Điều **đúng** cần nói: **artifact build không được track**, nên byte đang phục vụ phải được **pin bằng
`sha256` tại mỗi lần đo** — một bản build từ cây khác vẫn phục vụ được mà không có lỗi nào ở phía
server, và đó chính là cách W-05 phát hiện bundle hỏng. Pin hiện hành nằm ở
`WEB_SURFACE_VERIFICATION.vi.md` §"Service được phục vụ".

Cũng vì vậy con số `index-DjAje2d2.js` ghi bên trên là **superseded**: bundle đang phục vụ là
`assets/index-DWpK0yeS.js`, sha256 `7e34724ebf55092e36524b98980cca55e71f3ddbaa62489c65577f92e18a6a16`.

Backend trả lời (dùng `/api/benchmark-runs` làm readiness, **không** dùng `/api/models`):

```
curl -s -o /tmp/br.json -w "http_code=%{http_code} time=%{time_total}\n" --max-time 30 \
  http://127.0.0.1:8000/api/benchmark-runs
```
```
http_code=200 time=0.011570
```
54 761 byte JSON. Backend **sẵn sàng**.

## 4. Runtime

```
workbench/envs/ctl312h/bin/python -VV
```
```
Python 3.12.11 (main, Oct  7 2025, 15:39:34) [Clang 20.1.4 ]
```

Đo trong cùng interpreter (script ghi ra `/tmp/rt.txt`):

```
workbench/envs/ctl312h/bin/python - <<'EOF'
import os, sys
print("cpu_count", os.cpu_count()); print("affinity", len(os.sched_getaffinity(0)))
for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"]:
    print("env", k, os.environ.get(k))
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(),
      "cuda", torch.version.cuda, "device", torch.cuda.get_device_name(0))
try:
    import threadpoolctl; print(threadpoolctl.threadpool_info())
except Exception as e: print("threadpoolctl_error", type(e).__name__, e)
EOF
```

Output thô:

```
cpu_count 20
affinity 20
env OMP_NUM_THREADS None
env OPENBLAS_NUM_THREADS None
env MKL_NUM_THREADS None
env NUMEXPR_NUM_THREADS None
env VECLIB_MAXIMUM_THREADS None
torch 2.15.0+cu132
cuda_available True
cuda_version 13.2
device NVIDIA GeForce RTX 5070 Ti
threadpoolctl_error ModuleNotFoundError No module named 'threadpoolctl'
```

- **20 CPU**, affinity cũng **20** (không bị cpuset bó).
- Không đặt biến pool nào → mọi pool lấy default = số CPU khả dụng. Vì `threadpoolctl` không có trong env, đo trực tiếp qua torch:

```
workbench/envs/ctl312h/bin/python -c "import torch; print(torch.get_num_threads(), torch.get_num_interop_threads(), torch.version.cuda, torch.cuda.get_device_capability(0), torch.cuda.is_bf16_supported())"
```
```
torch.get_num_threads 20
torch.get_num_interop_threads 20
torch.version.cuda 13.2
cudnn 92600
device_cap (12, 0)
bf16 True
```

→ pool hiệu dụng **20 intra-op / 20 inter-op**. Đây là lý do rule "tối đa 4 thread" phải set tay (`OMP_NUM_THREADS=4`), vì default không tự giới hạn.

cgroup (cgroup v2, `/proc/self/cgroup` = `0::/user.slice/…/vte-spawn-….scope`):

```
cpu.max    = max 100000        (/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/app.slice/cpu.max)
memory.max = max               (cùng thư mục)
memory.current = 29027979264   (~27.0 GiB)
```

`/sys/fs/cgroup/cpu.max` không tồn tại (cgroup v2 đặt file ở thư mục con theo scope); cả `cpu.max` lẫn `memory.max` đều **không giới hạn**. Đây là host thật, không container hoá.

Torch import được → **không** cần chế độ "nếu torch không import". Vẫn ghi GPU:

```
nvidia-smi --query-gpu=driver_version,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```
```
580.173.02, NVIDIA GeForce RTX 5070 Ti, 16303 MiB, 1092 MiB, 14715 MiB, 12 %, 45
```
```
(không có dòng nào — không process compute nào đang giữ GPU)
```

Driver **580.173.02**, driver báo CUDA **13.0** (torch build với cu132), card **RTX 5070 Ti** 16 303 MiB, trống **14 715 MiB**, util 12 %, 45 °C. Không có job GPU nào chạy → mốc này là GPU rảnh, dùng được làm baseline.

## 5. Disk

```
df -h /
for d in data workbench/artifacts workbench/third_party …; do du -sh "$d"; done
```

```
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme0n1p5  630G  446G  152G  75% /
25G   /home/heheboiz/data/cir/data
84G   /home/heheboiz/data/cir/workbench/artifacts
25G   /home/heheboiz/data/cir/workbench/third_party
34M   /home/heheboiz/data/cir/workbench/artifacts/circus_release
4.0K  /home/heheboiz/data/cir/workbench/artifacts/results
24G   /home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads
```

Còn **152 G** trống. Lịch sử có một sweep chết vì ENOSPC (`…/20261109-native15-composition-limn/sweep_36ae9e2_ABORTED_ENOSPC.txt`) — hiện tại không còn là blocker, nhưng đừng giả định dung lượng là vô hạn: `artifacts` 84 G, riêng checkpoint cohort 24 G.

## 6. Registry inventory (số quyết định của chiến dịch)

Nguồn: `workbench/registry/models.yaml`, `yaml.safe_load(...)['models']` → **17 entry**, `protocols` có 3 định nghĩa. Chi tiết từng dòng: `INVENTORY.csv` cùng thư mục.

Query đếm:

```
workbench/envs/ctl312h/bin/python -c "
import yaml; d=yaml.safe_load(open('workbench/registry/models.yaml'))
ms=d['models']; print(len(ms)); print(len({m['upstream_repo_url'] for m in ms}))"
```

- **17 cấu hình model** — khớp kỳ vọng.
- **16 upstream repo** — khớp kỳ vọng. Repo duy nhất có 2 model: `https://github.com/ABaldrati/CLIP4Cir` → `combiner_rn50x4_noft` + `clip4cir_rn50x4_fullft`, cùng pin `dfed9f748a8a`.

### Cohort "15 native FashionIQ"

17 trừ 2 deferred (`csmcir`, `sprc`) còn **15**: `airknow, conesep, ptha_mtst, habit, intent, hint, encoder, pair, clvc_net, dcnet, combiner_rn50x4_noft, clip4cir_rn50x4_fullft, tgcir, limn, tme`.

### Hai deferred

`csmcir`, `sprc`. Registry **không** có field `deferred`/`status: deferred` — nó chỉ ghi `reproduction_status`:

- `csmcir` → `NOT_RUN`, kèm `source_blocker` (package lavis pin không import được) và 2 `approved_source_patches`. Trạng thái "deferred" sống ở văn bản, không ở yaml: `workbench/docs/reviews/20261109-native15-composition-limn/EVIDENCE_INDEX.md:17-18` ("17 minus the 2 user-deferred (`csmcir`, `sprc`)"), `UNFINISHED.vi.md:61-62` (`USER-DEFERRED`), `GATES.csv` dòng `Scope/accounting`.
- `sprc` → `BLOCKED_CHECKPOINT_MAPPING`, và là model **duy nhất không có `composition_adapter`** (16/17 có).

### Không có alias / roll-up trong registry

`grep -i 'defer\|alias\|rollup' workbench/registry/models.yaml` → **không match**. Cả 17 entry đều là model độc lập; không entry nào là `alias_of`/`rollup_of` model khác.

"Roll-up" chỉ tồn tại ở **artifact**, không ở registry — `workbench/backend/native_measurements.py:206` suy ra rollup khi một file JSON có ≥2 block với owner khác nhau. Các file đó là `all_measured_summary.json` (`airknow_n05/airknow_n08/clvc/tgcir`), `clip4cir_parsed.json` (`clip4cir_noft`/`clip4cir_fullft`), `selection_decisions.json`, `pair_compat/*/summary.json` (`B1/H_normal` …). Đây là *cách file tổng hợp được đọc*, không phải model thứ 18.

### Đối chiếu với fixture `workbench/tests/fixtures/pc_launcher/models.json`

Cả hai bên đều 17 entry, cùng tập `model_id` (không thiếu/thừa). Lệch **nhãn protocol**:

| model_id | models.yaml | fixture models.json |
|---|---|---|
| `tme` | `fashioniq_original_split` | `fashioniq_val_split_ref_included` |
| `ptha_mtst` | `fashioniq_original_split` | `null` (và `supported_protocols: []`) |

`tme` đúng là ca đã biết. `ptha_mtst` là ca **thứ hai**, cùng bản chất: fixture còn để `upstream_commit_sha: null`, `source_available: false`, `reproduction_status: BLOCKED_NO_VERIFIED_FASHIONIQ_CHECKPOINT`, còn yaml đã pin `6a02ee4763fc…` và đặt `SELECTED_TRANSFER_ONLY`. Fixture đang **cũ hơn** yaml.

Lệch khác (không phải protocol nhưng cùng gốc "fixture cũ"): `fashioniq_layout` của 6 model iLearn-Lab (`airknow, conesep, habit, intent, hint, pair`) là `fashioniq_ilearn_resized_training` trong yaml vs `fashioniq_ilearn_resized` trong fixture; `encoder.reproduction_status` yaml `METRICS_MEASURED_MAPPING_UNVERIFIED` vs fixture `NOT_RUN`; `tme.literature_split_label` yaml `original` vs fixture `val`.

## 7. Cái gì thực sự có trên đĩa (và cái bẫy)

### Bẫy phải nhớ

Native measurement **không** nằm ở `workbench/artifacts/logs/`. Nó nằm trong cây selection:

```
workbench/artifacts/checkpoint_selection/20261106T002617+0700/measurements/          (17 file)
workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/measurements/   (11 file)
workbench/artifacts/checkpoint_selection/20261106T002617+0700/logs/                  (18 file, chỉ log)
```

Ai chỉ nhìn `workbench/artifacts/logs/` sẽ kết luận sai rằng `tgcir`, `clvc_net`, `airknow` chưa chạy. Thực tế cả ba **có** artifact native hoàn chỉnh:

```
measurements/tgcir_run1.json  → aggregate.complete = true (R@10 49.7208 / R@50 71.1023)
measurements/clvc_run1.json   → aggregate.complete = true (R@10 29.9707 / R@50 58.6217)
measurements/airknow_fiq_n05_run1.json + airknow_fiq_n08_run1.json (2 biến thể nhiễu)
```

Ngoài ra còn hai cây selection khác mà cùng bẫy: `20261106T233409+0700` (hint/tme/ptha/dcnet bổ sung) và `20261107T021500+0700` (csmcir). Tổng `files_scanned = 71` trong `/api/native-measurements`.

### (a) Native measurement artifact

Nguồn: `GET /api/benchmark-runs` → `GET /api/native-measurements`, đếm `metrics` có và `rollup == false`:

```
curl -s http://127.0.0.1:8000/api/native-measurements -o /tmp/api_native-measurements.json
workbench/envs/ctl312h/bin/python -c "
import json; d=json.load(open('/tmp/api_native-measurements.json'))
print(d['counts'])
print(sorted({r['model_id'] for r in d['measurements'] if r.get('metrics') and not r.get('rollup')}))"
```

```
{'files_scanned': 71, 'files': 32, 'blocks_with_metrics': 43, 'with_metrics': 21,
 'rollup_rows': 22, 'unreadable': 2, 'signposted_blockers': 4,
 'models_with_metrics': 15, 'models_rollup_only': 13, 'verified': 0}
['airknow','clvc_net','conesep','csmcir','dcnet','encoder','habit','hint',
 'intent','limn','pair','ptha_mtst','sprc','tgcir','tme']
```

**15/17** model có measurement trực tiếp (không qua roll-up). Thiếu: `combiner_rn50x4_noft`, `clip4cir_rn50x4_fullft` — hai model này chỉ xuất hiện **bên trong** file roll-up `clip4cir_parsed.json`/`selection_decisions.json`, tức `has_native_measurement` phải đọc là "có số nhưng chỉ qua roll-up, không có file measurement riêng".

Cảnh báo mạnh: `verified: 0`. Không measurement nào có completion proof / evidence-ledger (`native_measurements.py:150-157`: `state: UNVERIFIED_NATIVE_MEASUREMENT`). "Có artifact" **không** bằng "đã verify".

### (b) G1/G2 parity artifact

Nguồn: `GET /api/runs/parity-gate`, root `workbench/docs/reviews/20261109-native15-composition-limn/` (`parity_gate.py:74`), tên file `g1_g2.{category}.{native,adapter}.json` (+ tiền tố `encoder.`):

```
curl -s http://127.0.0.1:8000/api/runs/parity-gate -o /tmp/api_runs_parity-gate.json
workbench/envs/ctl312h/bin/python -c "
import json; d=json.load(open('/tmp/api_runs_parity-gate.json'))
print(d['passed_models'])
for k,v in d['models'].items(): print(k, v['status'], len(v['checks']))"
```

```
passed_models []
limn    PARITY_MANDATORY_SUBCHECK_FAILED  48 checks
encoder PARITY_MANDATORY_SUBCHECK_FAILED  96 checks
```

**2/15** có artifact (`limn`, `encoder`); **0/15 PASS**. Con số "2/15" chỉ là *có bằng chứng*, không phải *đạt* — đừng trích thành "2 model pass G1/G2".

Riêng `hint` có `20261106T233409+0700/measurements/hint_parity_top50.json` (top-50 set/order equality, 0 mismatch trên 3 category) nhưng **không** nằm trong `EVIDENCED_MODELS = ("limn","encoder")` của `parity_gate.py:77` → gate không đếm nó. Đây là chỗ dễ tranh cãi "2 hay 3"; theo gate hiện hành, đáp án là **2**.

### (c) CIRCUS composition run

Nguồn: `GET /api/composition/runs` (35 record = 22 thư mục summary × subset), lấy tập `model_id` distinct:

```
curl -s http://127.0.0.1:8000/api/composition/runs -o /tmp/api_composition_runs.json
workbench/envs/ctl312h/bin/python -c "
import json; print(sorted({r.get('model_id') for r in json.load(open('/tmp/api_composition_runs.json')) if r.get('model_id')}))"
```

```
['conesep','encoder','habit','hint','intent','limn','ptha_mtst','tme']
```

**8/15**. Khớp `GATES.csv` dòng `Composition correctness` ("8 selected CIRCUS runs"). Trạng thái: `COMPLETED` ×7, `limn` = `PARTIAL` (6001/6003 query, 2 unscored).

Bẫy thứ hai ở đây: `workbench/artifacts/composition/summary/` có **22** thư mục nhưng chỉ **8** là run được chọn. Phần còn lại là run cũ (nhiều run_id cho cùng model, ví dụ `encoder` có 3, `habit` có 3 gồm 1 `FAILED`), 2 synthetic (`SYNTHETIC_habit_fiq_n02_fixture002`) và 1 release audit (`released_audit_task7_mm_any_…`). Đếm thư mục = 22 sẽ ra "22 model", sai hoàn toàn.

### Bảng tổng hợp 3 cột

| | có artifact | thiếu |
|---|---|---|
| native measurement (trực tiếp) | 15/17 | `combiner_rn50x4_noft`, `clip4cir_rn50x4_fullft` (chỉ roll-up) |
| G1/G2 parity | 2/15 (0 PASS) | 13 model |
| CIRCUS composition | 8/15 | `airknow, pair, clvc_net, dcnet, combiner_rn50x4_noft, clip4cir_rn50x4_fullft, tgcir, sprc` (sprc còn thiếu cả adapter) |

## Kết luận baseline

HEAD `92ed66d` khớp remote, tree sạch (chỉ file operator), service 8000/5173/8765 đều sống và bundle khớp đĩa, GPU rảnh 14.7 GiB, đĩa còn 152 G. Ba số phải nhớ: **17 cấu hình / 16 repo / 15 native cohort** — tất cả khớp kỳ vọng. Không có lệch nào ở ba số này. Lệch duy nhất cần báo là **nhãn protocol giữa yaml và fixture**: `tme` (đã biết) **và** `ptha_mtst` (mới), cùng một nguyên nhân: fixture chưa được cập nhật theo yaml.
