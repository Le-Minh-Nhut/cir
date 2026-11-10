# BASELINE — snapshot đầu lượt (2026-11-10)

Mọi dòng dưới đây do đúng lệnh ghi kèm sinh ra; output thô nằm ở `raw/`.
Control interpreter: `workbench/envs/ctl312h/bin/python` (CPython 3.12.11).

> **Cảnh báo phạm vi:** trong lượt này một worker khác đã sửa
> `workbench/backend/limn_diagnostic.py` (218 dòng tại HEAD → 615 dòng trong working tree,
> `git diff --stat` = `458 insertions(+), 61 deletions(-)`). Vì vậy **mọi kết luận về R5/R6
> dưới đây là về HEAD `2e2e2d2`, không phải working tree**. Bằng chứng đo được ghi ở
> `raw/r6_guard.txt` (chạy trên bản HEAD) và `raw/r6_working_tree.txt`.

## 1. Git

| mục | giá trị | lệnh |
|---|---|---|
| HEAD | `2e2e2d299e0cd3f27d6bcaff09fdf64d7c778dc6` | `git rev-parse HEAD` |
| nhánh | `research/cir-failure-workbench` | `git rev-parse --abbrev-ref HEAD` |
| upstream | `origin/research/cir-failure-workbench` = `2e2e2d2…` | `git rev-parse origin/…` |
| commit chưa push | **0** (`git rev-list --count origin/…..HEAD` = 0, `git log` rỗng) | `git log --oneline origin/…..HEAD` |
| `git status --short` | `?? CLEANUP_REPORT.md.save`, `?? workbench/docs/reviews/20261110-e2e-repair/` | `git status --short` |
| `git diff --stat` | *rỗng tại thời điểm chụp*; sau đó thành `workbench/backend/limn_diagnostic.py \| 519 ++++---` | `git diff --stat` |

Output: `raw/baseline_git.txt`. Bản chụp git ở trên là **trước** khi worker kia ghi vào
`limn_diagnostic.py`; xem §9.

## 2. Provenance upstream/vendor

Nguồn: chính `workbench/third_party/<repo>/provenance.json` của từng repo (không suy đoán).
Script `raw/baseline_provenance.py` đọc `upstream_commit`, `local_fixes`, và **tự băm lại
toàn bộ `file_manifest`** để so với digest đã ghi. Output: `raw/baseline_provenance.txt`.

| repo | upstream_commit | ngày commit | model_id | local_fixes (khai báo) | manifest |
|---|---|---|---|---|---|
| AirKnow | `6ab6fa95b746f85b08eea4cb66eb51448bce99b7` | 2026-09-28 | airknow | `data_utils.py`, `lavis/models/__init__.py` (model_recovery) | OK (1218 file) |
| CLIP4Cir | `dfed9f748a8a4d05abf613164e06d59b22dbdf13` | 2023-09-05 | combiner_rn50x4_noft, clip4cir_rn50x4_fullft | – | OK (14) |
| CLVCNet | `bd9b6889489806cb60676597880f91d918b279cb` | 2026-04-05 | clvc_net | `resnet.py` | OK (9) |
| CSMCIR | `774f94e2076ff17ea91703a6239d2a08f0e1a44e` | 2026-04-27 | csmcir | – | **2 file lệch** |
| ConeSep | `bd4bf629ebe68045c1a54dcc61e4a3ea65f8759b` | 2026-05-25 | conesep | – | OK (1221) |
| DCNet | `68c79d38569f01ca39c3b4ea641b3b7eb2ccb5fb` | 2021-03-15 | dcnet | – | **1 file lệch** |
| ENCODER | `29a2a31d6a56f677bf450c3be7cdaef423fb7018` | 2026-05-25 | encoder | `datasets1.py`, `evaluate_model.py` | OK (26) |
| HABIT | `6ddc2cd63eac1f65693f4e821c34fb3b42aee306` | 2026-05-25 | habit | – | OK (1234) |
| HINT | `bec50b6c8c19111893b617979502b948d1cea5b2` | 2026-04-09 | hint | – | OK (1225) |
| INTENT | `2a75fc6f57d1a036bf3167a7fe1d2cee072f51f3` | 2026-05-25 | intent | – | OK (1228) |
| LIMN | `7d7bc9b116f594a65ac22457491edf28a88d3c3e` | 2026-04-05 | limn | – | OK (33) |
| MTST_PTHA | `6a02ee4763fc2260b6abfb18be78853facb91d86` | 2025-02-09 | ptha_mtst | `.gitattributes`, `src/data_utils.py`, `src/lavis/models/__init__.py` | OK (360) |
| PAIR | `bfb0b98cdb4c44f6b9190baaa84d783a77dd888f` | 2026-04-05 | pair | `models/MHTransformer_rev.py` | OK (22) |
| SPRC | `2935a5397732260d1db6fa577e5926f963e36f0f` | 2024-04-16 | sprc | – | OK (354) |
| TGCIR | `65fa78eaf8cabe8197fcc22ccab42207c591bdee` | 2026-04-09 | tgcir | – | OK (11) |
| TME | `394aad79df2c104feb32a2c96132582012765062` | 2026-10-19 | tme | `precompute.sh`, `src/utility.py` | **2 file lệch** |

### 2.1 Năm file "lệch manifest" — **không phải drift**, là patch đã được duyệt

`raw/baseline_provenance_drift.txt` cho thấy cả 5 file **không có** modification nào so với HEAD
(`git status --short` rỗng) và digest thực tế **trùng khít `derived_sha256`** trong
`workbench/registry/models.yaml`:

| file | `base_sha256` (= digest trong provenance) | digest thực tế trên đĩa = `derived_sha256` |
|---|---|---|
| `CSMCIR/src/lavis/models/__init__.py` | `9d8c914b…` | `0ec13443…` (models.yaml:108) |
| `CSMCIR/src/validate_blip_csmcir.py` | `aa7a4395…` | `a057e8f3…` (models.yaml:113) |
| `DCNet/base/base_trainer.py` | `3edd9836…` | `6917e18f…` (models.yaml:1644) |
| `TME/src/precompute.py` | `c3da9c9d…` | `97bdbc14…` (models.yaml:1649) |
| `TME/src/precompute_evaluation.py` | `0f6a0573…` | `1f13dbd5…` (models.yaml:1654) |

Đây là `approved_source_patches` — cơ chế ở `workbench/backend/vendored_source.py:224-253`
(so `live[path]` với `derived_sha256` **hoặc** `base_sha256`, chứ không chỉ manifest).
**Hệ quả cho báo cáo của campaign:** nếu ai kiểm tra provenance chỉ bằng `file_manifest` thì sẽ
báo sai 5 "local modification" không tồn tại. Kiểm tra đúng phải đi qua `vendored_source` /
`approved_source_patches`.

## 3. Service config đang chạy

| mục | giá trị | lệnh |
|---|---|---|
| cmdline | `workbench/envs/ctl312h/bin/python workbench/scripts/serve_workbench.py --production-frontend --host 127.0.0.1 --backend-port 8000 --frontend-port 5173` (PID 2759507) | `pgrep -af serve_workbench.py` |
| backend thật | `…/ctl312h/bin/python -m uvicorn workbench.backend.main:app --host 127.0.0.1 --port 8000` (PID 2759510) | `pgrep -af python` |
| bind | `127.0.0.1:8000` (python 2759510), `127.0.0.1:5173` (node 2759525) | `ss -ltnp` |
| backend port | 8000 (`/docs` → 200, `/` → 404) | `curl -o /dev/null -w '%{http_code}'` |
| env export **thật** | chỉ **một** biến: `WORKBENCH_CIRCUS_RELEASE_ROOT=/home/heheboiz/data/cir/workbench/artifacts/circus_release` (ở cả 2759507 và 2759510). Không có `WORKBENCH_*_PYTHON` nào được export | `tr '\0' '\n' < /proc/<pid>/environ` |
| cwd | `/home/heheboiz/data/cir` (cả hai) | `readlink /proc/<pid>/cwd` |
| start | Tue Nov 10 07:24:34 2026 | `ps -o lstart` |
| log | `/tmp/workbench_serve.log` (fd 1 và 2 của **cả hai** process trỏ vào đây; 61 859 byte) | `ls -l /proc/<pid>/fd/1` |
| có service thứ hai | `…/ctl312h/bin/python -m uvicorn workbench.backend.main:app --host 127.0.0.1 --port 8765` (PID 1227919) — **cũng đang nghe**, không rõ chủ | `ss -ltnp` |

Output: `raw/baseline_services.txt`, `raw/baseline_env.txt`, `raw/baseline_log.txt`.

Config trên đĩa (đọc, không sửa): `workbench/config/workbench.env`, 18 key, snapshot ở
`raw/baseline_config_env.txt`. Đáng chú ý: nó đặt **9** `WORKBENCH_*_PYTHON` (csmcir, encoder,
limn, habit, intent, hint, conesep, ptha_mtst, tme) và `WORKBENCH_RUNTIME_ADAPTATION=sm_120`.
`resolve_config()` đọc file này ở **call time** (`operator_config.py:100-125`), nên các biến
Python **có hiệu lực** dù không nằm trong `/proc/<pid>/environ`.

## 4. Interpreter mỗi model dùng

`raw/baseline_interpreters.py` → `raw/baseline_interpreters.txt`. Cột "upstream" = `environment.python`
trong `workbench/registry/models.yaml` (bản README upstream pin); "adapted" = `environment.adapted_runtime.python`;
"resolved" = giá trị `python_env_var` trong `workbench.env`, hoặc interpreter mặc định của workbench
khi biến không được khai; "reported" = `<interpreter> -c "import sys;print(sys.version.split()[0])"`.

| model | upstream | adapted | resolved | reported |
|---|---|---|---|---|
| csmcir | 3.9 | 3.10 | `artifacts/environments/cir-005ef4e1b02999a5/bin/python` | **3.10.22** |
| airknow | 3.8.10 | – | *(default workbench)* | **3.12.11** |
| conesep | 3.8.10 | 3.9 | `envs/ilearn39-adapted/bin/python` | **3.9.25** |
| ptha_mtst | 3.9 | 3.10 | `artifacts/environments/cir-005ef4e1b02999a5/bin/python` | **3.10.22** |
| habit | 3.8.10 | 3.9 | `envs/ilearn39-adapted/bin/python` | **3.9.25** |
| intent | 3.9 | 3.9 | `envs/ilearn39-adapted/bin/python` | **3.9.25** |
| hint | 3.8.10 | 3.9 | `envs/ilearn39-adapted/bin/python` | **3.9.25** |
| encoder | 3.9 | 3.10 | `envs/encoder310-adapted/bin/python` | **3.10.20** |
| pair | 3.8.10 | – | *(default workbench)* | **3.12.11** |
| clvc_net | 3.7.6 | – | *(default workbench)* | **3.12.11** |
| dcnet | 3.7.7 | – | *(default workbench)* | **3.12.11** |
| combiner_rn50x4_noft | 3.8 | – | *(default workbench)* | **3.12.11** |
| clip4cir_rn50x4_fullft | 3.8 | – | *(default workbench)* | **3.12.11** |
| tgcir | UNKNOWN | – | *(default workbench)* | **3.12.11** |
| sprc | 3.9 | – | *(default workbench)* | **3.12.11** |
| limn | UNKNOWN | 3.10 | `envs/limn310-adapted/bin/python` | **3.10.20** |
| tme | 3.10 | – | `envs/tme-adapted/bin/python` | **3.10.22** |

Hai điều đọc được từ bảng này, cả hai đều **có sẵn trong code** chứ không phải suy đoán:

1. **7/17 model không có `python_env_var`** trong `workbench.env` (airknow, pair, clvc_net, dcnet,
   hai entry CLIP4Cir, tgcir, sprc). Với chúng, `require_verified_model_interpreter`
   (`workbench/backend/runtime.py:2301-2303`) trả `sys.executable`, tức **interpreter của workbench
   (3.12.11)** — không phải bản upstream pin (3.7.6 … 3.9).
2. `csmcir`, `ptha_mtst` **dùng chung một** interpreter `cir-005ef4e1b02999a5` (3.10.22); `habit`,
   `intent`, `hint`, `conesep` **dùng chung** `ilearn39-adapted` (3.9.25). Chia sẻ env không sai,
   nhưng nghĩa là "mỗi model một môi trường" không đúng như tên gọi.

## 5. Bundle được phục vụ

`curl -s http://127.0.0.1:5173/` trả **đúng 271 byte**, chỉ có 3 asset:

| asset trong `index.html` | phục vụ | trên đĩa `workbench/frontend/dist/assets/` |
|---|---|---|
| `/assets/index-BgedtZDd.js` | HTTP 200, 353 254 byte | có, 353 254 byte |
| `/assets/index-DAZAlgSy.css` | HTTP 200, 26 003 byte | có, 26 003 byte |

`index.html` **không** chứa `<div id="root">` có nội dung, chỉ một `<div id="root"></div>` rỗng —
đúng như SPA. Không có asset nào bị thiếu. Output: `raw/baseline_bundle.txt`, `raw/baseline_index.txt`.

## 6. Data / artifact / cache root

| root | đường dẫn | kích thước / digest |
|---|---|---|
| repo | `/home/heheboiz/data/cir` | – |
| artifacts | `workbench/artifacts` | 84 G |
| data | `data` | 25 G |
| release root | `workbench/artifacts/circus_release` | 34 M |
| native run dir (limn) | `workbench/artifacts/composition/runs/9e4918a6b5c243412cbc58c9` | 860 M |
| composition summary | `workbench/artifacts/composition/summary` (22 thư mục) | 1.1 M |
| `summary/limn_base_iter0_all_categories_9e4918a6b5c243412cbc58c9/full.json` | – | sha256 `c446acc4f554d2e995e346ab837e553ea122068d1180387dcbf87426ea800bef` |
| checkpoint root | `workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads` | (theo `workbench.env`) |

Output: `raw/baseline_roots.txt`, `raw/baseline_disk.txt`.

## 7. Job đang chạy / GPU

- `pgrep -af python`: chỉ có (a) hai uvicorn/serve ở §3, (b) 4 process
  `.venv/bin/python -u /tmp/omp-python-runner-1000/runner-2gcnasrh1f37g.py` — chính là các
  kernel eval của phiên này, (c) `/usr/bin/python3 … unattended-upgrade-shutdown` và
  `gnome-terminal --wait` của hệ điều hành.
- `nvidia-smi`: RTX 5070 Ti, **430 MiB / 16 303 MiB**, 32 W, GPU-Util 29 %, **không có process
  compute nào** trong bảng Processes (chỉ Xorg, gnome-shell, xdg-desktop-portal-gnome).
  ⇒ **Không có job GPU nào đang chạy.**
- Output: `raw/baseline_jobs.txt`.

## 8. Đĩa

```
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme0n1p5  630G  442G  156G  74% /
```
(`df -h /home/heheboiz/data/cir`, `raw/baseline_disk.txt`). Còn 156 G — không có nguy cơ `ENOSPC`.

## 9. Ranh giới thay đổi trong lượt

`workbench/backend/limn_diagnostic.py` đã bị một worker khác sửa **sau** khi tôi chụp git baseline:
HEAD 218 dòng → working tree 615 dòng (`git diff --stat` = `458 insertions(+), 61 deletions(-)`,
mtime `2026-11-10 07:57:23`). Bản sửa đó thêm `target_did_not_own_query`,
`export_duplicate_candidate`, `summary_rank_invalid`, `checkpoint_summary_vs_export`,
`export_target_not_flagged` và một manifest/digest cache — tức **vá đúng R5 và R6**.

Vì vậy:
- `raw/r6_guard.txt` = hành vi **HEAD** (chép `limn_diagnostic.py` của HEAD ra scratch rồi chạy).
- `raw/r6_working_tree.txt` = hành vi **working tree** (cùng bộ input).
- Kết luận "CONFIRMED trên HEAD" **không** đồng nghĩa "còn lỗi hôm nay". Xem cột status của
  `FINDINGS_AND_FIXES.md`, nơi R5/R6 được ghi là **ALREADY_FIXED_IN_WORKING_TREE** kèm cả hai bằng chứng.
