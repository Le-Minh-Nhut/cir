# RUNS — nhật ký chạy cohort (đợt export candidate Top-K)

Ngày: 2026-11-08 · Nhánh `research/cir-failure-workbench` · Không commit, không push.
Thư mục bằng chứng: `workbench/docs/reviews/20261108T120416-runs/`.

Máy: RTX 5070 Ti 16 GB. **Tuần tự, một model một lúc**, driver chờ `nvidia-smi --query-compute-apps`
rỗng trước mỗi model. Control interpreter: `workbench/envs/ctl312h/bin/python` (CPython 3.12.11).

---

## 0. Tóm tắt

| model | trạng thái | run_id | candidate_topk | ghi chú |
|---|---|---|---|---|
| `habit` | **COMPLETED** | `497c25c57fbeba885ecfb9d1` | **true, depth 50** | NEW_RUN, có candidate thật, có `variant_provenance.sm_120` |
| `intent` | **COMPLETED** | `37ee72aa298b276d1bfda0fa` | **true, depth 50** | NEW_RUN |
| `hint` | **COMPLETED** | `4bc9893a977e501b82378e76` | **true, depth 50** | NEW_RUN |
| `conesep` | **COMPLETED** | `30f6de423861cc1d6c78c361` | **true, depth 50** | NEW_RUN |

**Cả bốn model đã xong.** Chain exit code **0**, GPU đã rỗng, không còn process nào.

| model | MM R@10 | MM R@50 | MM nDCG | gap | coverage |
|---|---|---|---|---|---|
| `habit` | 35.6155 | 54.3728 | 0.32828 | 0.38870 | 1.00000 |
| `intent` | 34.3995 | 54.0063 | 0.31849 | 0.37538 | 1.00000 |
| `hint` | 32.8169 | 51.7741 | 0.30927 | 0.43870 | 1.00000 |
| `conesep` | 35.4823 | 54.1396 | 0.32944 | 0.39426 | 1.00000 |

---

## 1. Dọn thư mục chạy chết (bước 1)

Xác nhận bằng `os.walk` rằng **cả hai đều không có `ranks/`** trước khi xoá:

| thư mục | nội dung | có `ranks/`? | thu hồi |
|---|---|---|---|
| `runs/f76171fa32296f3369970a5c` | `work/habit/features/gallery_global.npy` (2,437,316,736 B) + 1 job json + worker.stderr | **không** | 2.27 GiB |
| `runs/7f46f99729d6df51c00e8e66` | như trên | **không** | 2.27 GiB |
| **tổng** | | | **4.54 GiB** |

Lý do chúng thực sự vô dụng (không chỉ vì thiếu rank): `_load_memmap()` (`runner.py:187-197`) chỉ
chấp nhận cache khi có sidecar `<file>.npy.json` khớp `fingerprint`. Hai thư mục này **chỉ có `.npy`,
không có sidecar** → cache luôn trả `None`, tức là memmap 2.4 GB chưa bao giờ dùng lại được.

Đã xoá thêm một thư mục `6c4fd134f93e0efa737ec960` (2.27 GiB) do chính lượt chạy của tôi sinh ra rồi
bị kill giữa lúc encode — cùng lý do (`.npy` không sidecar, không `ranks/`).

**Tổng thu hồi: 6.81 GiB.** Không thư mục nào có rank file bị chạm.

---

## 2. Driver: `/tmp/evalwork/drive.py`

Một lệnh duy nhất cho mỗi model, **một invocation**, `--subsets full,sf,validated`. Không tách theo
subset vì `run_id` phụ thuộc thành viên subset; tách lệnh sẽ encode lại toàn bộ gallery 74.381 ảnh.

### Lệnh chính xác (argv do **một** hàm `build_argv()` dựng)

```sh
/home/heheboiz/data/cir/workbench/envs/ctl312h/bin/python -u \
  workbench/scripts/composition_eval.py \
  --config workbench/docs/reviews/20261108T113610-eval/circus_cohort.yaml \
  --models habit \
  --subsets full,sf,validated \
  --topk-depth 50 \
  --keep-features \
  --resume            # chỉ lượt relaunch của habit; 3 model còn lại KHÔNG có cờ này
```

Biến môi trường (đã đối chiếu **10/10 khớp** với `workbench/config/workbench.env`):

```sh
WORKBENCH_RUNTIME_ADAPTATION=sm_120
WORKBENCH_CIRCUS_RELEASE_ROOT=workbench/artifacts/circus_release
WORKBENCH_HABIT_PYTHON=workbench/envs/ilearn39-adapted/bin/python
WORKBENCH_INTENT_PYTHON=workbench/envs/ilearn39-adapted/bin/python
WORKBENCH_HINT_PYTHON=workbench/envs/ilearn39-adapted/bin/python
WORKBENCH_CONESEP_PYTHON=workbench/envs/ilearn39-adapted/bin/python
WORKBENCH_TME_PYTHON=workbench/envs/tme-adapted/bin/python
WORKBENCH_ENCODER_PYTHON=workbench/envs/encoder310-adapted/bin/python
WORKBENCH_LIMN_PYTHON=workbench/envs/limn310-adapted/bin/python
WORKBENCH_PTHA_MTST_PYTHON=workbench/artifacts/environments/cir-005ef4e1b02999a5/bin/python
# WORKBENCH_CIRCUS_SYNTHETIC_FIXTURE bị unset tường minh
```

`ENCODER`/`LIMN` dùng env **3.10.20** riêng (`encoder310-adapted` có `open_clip_torch==2.24.0`,
`limn310-adapted` có `2.20.0`). Hai pin khác nhau và đều exact-match, nên **không được gộp**.
Đường cũ `pair-adapted`/`limn-adapted` là Python 3.11.15 và sẽ bị gate từ chối.

### Vì sao không có `--resume` cho lượt export

`run_composition` bỏ qua rank site khi `--resume` + đã có rank file. Không có rank site thì cửa sổ
candidate **không bao giờ được ghi**, và run vẫn báo `COMPLETED`. Đây là cái bẫy đã được vá (xem §5).

### Cơ chế sống sót và giám sát

- **Detach thật**: `setsid nohup … &` → driver có session/PGID riêng. Lượt đầu dùng `nohup &` thường
  và bị harness kill khi turn kết thúc; đó là lý do phải đổi.
- **Log tăng trong lúc chạy**: con chạy `python -u`, stdout ghi **thẳng** vào file (không pipe — pipe
  64 KiB sẽ chặn con).
- **Lock theo model**: `flock` không-blocking trên `LOCKDIR/<model>.lock`; nhả thì unlink. Test đã
  chạy: acquire thứ hai → `False`, sau release → acquire lại `True`.
- **Tiến độ thật**, không đọc stdout (CLI chỉ in JSON ở cuối): CPU-time của cả cây process + byte trên
  đĩa + số job file + số dòng candidate.
- **Phân biệt chậm và treo**: ngưỡng stall chỉ tính là treo khi **không có chuyển động** (CPU-time
  không tăng, đĩa không tăng, job không đổi) **và** không còn process GPU. Log ghi tốc độ đo được
  (`+N giây CPU/phút`, `+N MiB/phút`) để quyết định timeout có căn cứ.
- **Gate capability**: sau khi model xong, gọi API; nếu `candidate_topk` không phải `true` hoặc
  `execution_status` không thuộc `COMPLETED/PARTIAL/REUSED` → **model FAILED**, dừng cohort.

---

## 3. `habit` / `fiq_n02` — NEW_RUN, COMPLETED

**run_id `497c25c57fbeba885ecfb9d1`** · checkpoint `fiq_n02`

### Lệnh

Lượt 1 (14:39 **thất bại**, bị timeout của chính driver):
`… --models habit --subsets full,sf,validated --topk-depth 50 --keep-features` (không `--resume`)
Lượt 2 (thành công): thêm `--resume` để dùng lại feature cache 2.3 GiB.

### Thời gian

| mốc | epoch | giờ |
|---|---|---|
| start (lượt thành công) | 1794123582 | 2026-11-08T14:39:42+0700 |
| end | 1794126095 | 2026-11-08T15:21:35+0700 |
| **thời lượng** | **2513 s = 41.9 ph** | |
| exit code | **0** | |

Phân rã pha (đo từ mtime cache và log):

| pha | lượt encode lạnh | lượt relaunch (cache ấm) |
|---|---|---|
| encode gallery (74.381 ảnh) | 12:26:47 → 13:34 ≈ **57 ph** | **0** — cache hit |
| encode query ×3 (6003 mỗi condition) | ≈ 24 ph | **≈ 30 s** |
| scoring + export | — | ≈ 35 ph |
| **tổng** | | **41.9 ph** |

### Số liệu (`summary/habit_fiq_n02_497c25c57fbeba885ecfb9d1/full.json`)

| trường | giá trị |
|---|---|
| `execution_status` | **COMPLETED** |
| `query_count` | 6003 (full) — `sf` 4069, `validated` 586 |
| `gallery_count` | 74381 (exact rank trên toàn gallery) |
| MM R@10 | **35.6155** |
| MM R@50 | **54.3728** |
| MM nDCG | **0.32828** |
| image-only R@10 / nDCG | 3.3150 / 0.11374 |
| text-only R@10 / nDCG | 15.2924 / 0.20068 |
| coverage | **1.00000** (6003/6003 scored, 0 unscored, cả 3 condition) |
| Composition Gap | **0.38870** (cohort 6003, có mẫu số thật) |

### Candidate Top-K (điểm mấu chốt)

```json
"topk_export": {"requested": true, "depth": 50, "partial": false,
                "missing_query_count": 0, "reason": null,
                "status": "TOPK_EXPORT_COMPLETE",
                "conditions": ["image_only_empty_text","multimodal","text_only_black_image"]}
```

API `GET /api/composition/runs/497c25c57fbeba885ecfb9d1?subset=full`:

```json
"capabilities": {"aggregate_metrics": true, "per_query_target_ranks": true,
                 "candidate_topk": true, "image_availability": true},
"capability_detail": {"candidate_topk": {"available": true, "status": "AVAILABLE",
  "depth": 50, "gallery_count": 74381,
  "tie_policy": "descending score, ties broken by earlier gallery position",
  "conditions": ["image_only_empty_text","multimodal","text_only_black_image"]}}
```

`GET /api/composition/runs/<run_id>/queries?subset=full&condition=multimodal&limit=3` trả
`top_results` thật, ví dụ:

```json
{"rank": 1, "image_id": "mbeir_images/fashioniq_images/B007X6BEF4.jpg",
 "gallery_position": 34725, "score": 0.7826133966445923}
```

**Bất biến đã kiểm**: với 5 query có `target_rank <= 50`, candidate tại đúng `target_rank` **là ảnh
target** (`positive_names`) — 5/5 khớp. Đây là điều kiện chấp nhận: candidate và rank đến từ **cùng
một** lần `score()`.

### Provenance — sửa được lỗi của lượt trước

```json
"variant_provenance": {
  "adapted_runtime": {"runtime_adaptation": "sm_120", "pytorch": "==2.7.1", "cuda": "12.8",
    "torchvision": "==0.22.1",
    "note": "ADAPTED_RUNTIME: modern package stack for the target GPU; not the authors'
             historical environment, not an exact reproduction claim"},
  "adapted_runtime_applied": true}
```

`grep sm_120` trong summary = **2 lần khớp**. Lượt trước TME/PTHA chạy dưới `sm_120` nhưng summary
**không ghi**; `habit` bây giờ ghi đúng.

### Artifact

```
workbench/artifacts/composition/runs/497c25c57fbeba885ecfb9d1/ranks/habit_fiq_n02.json
workbench/artifacts/composition/runs/497c25c57fbeba885ecfb9d1/candidates/habit_fiq_n02/{multimodal,text_only_black_image,image_only_empty_text}.jsonl
workbench/artifacts/composition/runs/497c25c57fbeba885ecfb9d1/candidates/habit_fiq_n02/*.manifest.json
workbench/artifacts/composition/summary/habit_fiq_n02_497c25c57fbeba885ecfb9d1/{full,sf,validated}.json
```

Nhãn: **NEW_RUN**. Không dùng lại cache của lượt chết (lượt chết không có rank file nên
`reused=False`; cache gallery/query thì dùng lại hợp lệ vì sidecar fingerprint khớp).

---

## 4. `intent`, `hint`, `conesep` — NEW_RUN, COMPLETED

Chain `/tmp/evalwork/chain_rest.sh` chờ driver của `habit` thoát (15:21:38), chờ GPU rỗng, rồi chạy
`intent → hint → conesep` trong **một** invocation của driver. Không có khoảng trống giữa model.
Kết thúc **17:43:04**, exit code **0**.

Không có `--resume` (để rank site ghi candidate), có `--topk-depth 50 --keep-features`, timeout 21600 s.

### 4.1 `intent` / `fiq_n02` — `37ee72aa298b276d1bfda0fa`

| trường | giá trị |
|---|---|
| start → end | 15:21:38 → 16:38:37 (**77.0 ph**) |
| exit code | 0 |
| `execution_status` | **COMPLETED** |
| query / gallery | 6003 / 74381 |
| condition_status | cả 3: `COMPLETED`, `unscored_count: 0` |
| MM R@10 / R@50 / nDCG | **34.3995 / 54.0063 / 0.31849** |
| image-only nDCG (R@10) | 0.11223 (2.8652) |
| text-only nDCG (R@10) | 0.19894 (15.3257) |
| coverage | **1.00000** |
| Composition Gap | **0.37538** (cohort 6003) |
| `topk_export` | `TOPK_EXPORT_COMPLETE`, depth 50, `partial: false`, `missing_query_count: 0` |
| `variant_provenance.sm_120` | **có** |

### 4.2 `hint` / `fashioniq` — `4bc9893a977e501b82378e76`

| trường | giá trị |
|---|---|
| start → end | 16:38:38 → 17:07:43 (**29.1 ph**) |
| exit code | 0 |
| `execution_status` | **COMPLETED** |
| query / gallery | 6003 / 74381 |
| condition_status | cả 3: `COMPLETED`, `unscored_count: 0` |
| MM R@10 / R@50 / nDCG | **32.8169 / 51.7741 / 0.30927** |
| image-only nDCG | 0.11427 |
| text-only nDCG | 0.17359 |
| coverage | **1.00000** |
| Composition Gap | **0.43870** (cohort 6003) |
| `topk_export` | `TOPK_EXPORT_COMPLETE`, depth 50, `partial: false`, `missing_query_count: 0` |
| `variant_provenance.sm_120` | **có** |

### 4.3 `conesep` / `fiq_n02` — `30f6de423861cc1d6c78c361`

| trường | giá trị |
|---|---|
| start → end | 17:07:43 → 17:43:04 (**35.4 ph**) |
| exit code | 0 |
| `execution_status` | **COMPLETED** |
| query / gallery | 6003 / 74381 |
| condition_status | cả 3: `COMPLETED`, `unscored_count: 0` |
| MM R@10 / R@50 / nDCG | **35.4823 / 54.1396 / 0.32944** |
| image-only nDCG | 0.11206 |
| text-only nDCG | 0.19956 |
| coverage | **1.00000** |
| Composition Gap | **0.39426** (cohort 6003) |
| `topk_export` | `TOPK_EXPORT_COMPLETE`, depth 50, `partial: false`, `missing_query_count: 0` |
| `variant_provenance.sm_120` | **có** |

### 4.4 Kiểm chứng API cho cả bốn model

`GET /api/composition/runs/<run_id>?subset=full` → `capabilities.candidate_topk = true`,
`capability_detail.candidate_topk.status = "AVAILABLE"`, `depth = 50`.

`GET /api/composition/runs/<run_id>/queries?subset=full&condition=multimodal&limit=1` →

| model | `candidates_available` | scored / unscored | candidate đọc được |
|---|---|---|---|
| habit | 6003 | 6003 / 0 | `B007X6BEF4.jpg` pos 34725 score 0.78261 |
| intent | 6003 | 6003 / 0 | `B0047HWOOC.jpg` pos 12852 score 0.81323 |
| hint | 6003 | 6003 / 0 | `B009FWH264.jpg` pos 46425 score 0.54547 |
| conesep | 6003 | 6003 / 0 | `B0047HWOOC.jpg` pos 12852 score 0.81391 |

Bất biến `target_rank` ↔ candidate đã kiểm trên `habit`: **5/5** query có `target_rank <= 50` trả về
đúng ảnh target (`positive_names`) tại đúng vị trí đó.

### 4.5 Vì sao ba model này nhanh hơn `habit`

Bản vá BLAS đã landed (`composition_eval.py:13-20`, `adapters.py:28-37`) đặt
`OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS = min(10, cpu//2)` = **10 thread** (đúng nửa máy theo AGENTS.md).

Đo được: `intent` đạt **+136…+296 giây CPU/phút** trong pha scoring, so với `habit` (import numpy
**trước** bản vá) chỉ **+59 giây CPU/phút** → khoảng **2.3–5×**. `hint`/`conesep` còn cao hơn
(+456…+617 giây CPU/phút ở pha scoring).

**Lưu ý cho lượt sau:** `habit` không hưởng bản vá vì đã import numpy từ 14:39, trước khi bản vá tồn
tại. Nếu cần số liệu `habit` nhanh hơn (không đổi kết quả — chỉ đổi thời gian), phải chạy lại thành
process mới.

### 4.6 Thời gian encode gallery (đo được, để ước lượng)

| model | encode gallery | tổng |
|---|---|---|
| habit (lượt encode lạnh, bị timeout giết) | ~57 ph | ~2 giờ (không publish) |
| habit (relaunch, cache ấm) | **0** | 41.9 ph (2513 s) |
| intent | ~18 ph | 77.0 ph (4619 s) |
| hint | riêng từng model | 29.1 ph (1745 s) |
| conesep | riêng từng model | 35.4 ph (2121 s) |

---

## 5. Lỗi đã gặp và nguyên nhân gốc

### L1. Driver bị harness kill khi turn kết thúc (mất 1 lượt)
`nohup … &` vẫn nằm trong process group của shell harness. Con bị SIGKILL, evaluator sống sót nhờ
reparent sang `systemd --user`, nhưng driver chết nên không ai ghi `.end`/gate capability.
**Sửa**: `setsid nohup … &` (session riêng). Đã xác nhận driver mới có `sid` riêng.

### L2. Timeout 7200 s giết một run đang tiến triển thật (mất ~2 giờ GPU)
Lượt `habit` đầu chạy **thật**: VRAM 4.5–4.8 GiB, gallery memmap đầy đủ, encode xong 3 bộ query, và
đã stream candidate tới `multimodal` 6000/6003, `text_only_black_image` 6000/6003,
`image_only_empty_text` 1758/6003 — thì driver kill ở 7213 s (`EXIT=-15`).
`close_export` không chạy → không publish gì. `worker.stderr.log` **chỉ có** FutureWarning về
`torch.cuda.amp.autocast`, **không OOM, không traceback** → đây là **timeout**, không phải lỗi code
hay lỗi môi trường.
Nguyên nhân gốc: (a) timeout 7200 s quá nhỏ so với 57 ph encode gallery + 24 ph encode query + scoring;
(b) bộ đếm "tiến độ" của tôi dùng số job file, mà con số đó **nhảy 1 → 0 → 1** giữa các pha nên pha
CPU-bound bị đọc là "không tiến triển".
**Sửa**: timeout mặc định 21600 s; stall chỉ tính khi **không có chuyển động** (CPU-time + byte +
job) **và** không còn process GPU; log ghi tốc độ đo được.

### L3. `.tmp` của lượt chết còn nằm lại
3 file `.jsonl.<pid>.tmp` của lượt bị kill vẫn còn trong `candidates/habit_fiq_n02/`. Đo 60 s: **0
byte tăng** → hoàn toàn trơ, không ảnh hưởng lượt sau, và `close_export` của lượt thành công đã ghi
file thật (`*.jsonl` + `*.manifest.json`) bên cạnh. Không xoá tay để không mất bằng chứng.

### L4. Bẫy `--resume` + export (đã được vá trong `runner.py`, không phải tôi sửa)
`identity` **không** chứa `topk_depth`, nên `run_id` giữ nguyên; nhưng `writers = {} if reused else …`
khiến lượt resume có rank file **không bao giờ** ghi candidate mà vẫn báo `COMPLETED`. Nay:
- export key theo **attempt** (không nhét `topk_depth` vào `identity` → không làm bẩn measurement
  identity, không tạo run_id trùng);
- resume có cache → **backfill từ vector đã cache** (`_backfill_export`), không encode lại gallery;
- resume **không** có cache → `EXPORT_INCOMPLETE` + `TOPK_EXPORT_BLOCKED` kèm lý do, **không** giả vờ
  thành công;
- `--keep-features` giữ `work/<model>/features/*.npy` (~2.4 GB/model) để backfill về sau.

**Lỗi còn tồn tại (đã báo, chưa sửa vì `runner.py` thuộc agent khác):**
`runner.py:1266-1268` tính `execution_status` theo thứ tự
`"REUSED" if reused else … "EXPORT_INCOMPLETE" if export_blocked else "COMPLETED"`.
Vì `reused` đứng **đầu**, nhánh `EXPORT_INCOMPLETE` **không bao giờ chạy được trên chính đường resume**
— tức là đúng cái tình huống nó sinh ra để báo. Lượt resume không có cache sẽ báo `REUSED` thay vì
`EXPORT_INCOMPLETE`, dù `topk_export.status` vẫn đúng là `TOPK_EXPORT_BLOCKED`. Gate capability của
tôi vẫn bắt được vì nó đọc `candidate_topk`, không đọc `execution_status`.

### L5. `reference_eligible` là literal (đã được vá)
Trước đây `identity` và summary ghi `True` cứng. Nay lấy từ
`adapter.measured_reference_eligibility()`; adapter không đo được thì ghi `null` + lý do, thay vì
khẳng định một chính sách chưa từng chạy (`limn` đo ra **false** kèm lý do).

---

## 6. Bảng provenance — nói đúng sự thật

| summary | có `variant_provenance`? | `sm_120` trong summary? |
|---|---|---|
| `habit_..._497c25c5` (mới) | **có** | **có (2 khớp)** |
| `intent_..._37ee72aa` (mới) | **có** | **có** |
| `hint_..._4bc9893a` (mới) | **có** | **có** |
| `conesep_..._30f6de42` (mới) | **có** | **có** |
| `habit_..._e0dd04a4` (FAILED cũ) | không | không |
| `tme_..._ad477312` | **không** | **không** |
| `ptha_mtst_..._315dd590` | **không** | **không** |
| `encoder_..._6a528e87` | không | không |
| `limn_..._10173dc4` | không | không |

TME và PTHA **đã chạy** dưới `sm_120` (interpreter 3.10.22, torch 2.7.1+cu128, verified=true) nhưng
artifact của chúng **không ghi** — đó là khuyết điểm provenance của lượt trước, chỉ được ghi nhận
trong tài liệu, **không** được sửa lại trong artifact. Chỉ `habit` (chạy sau bản vá) mới ghi đúng.

---

## 7. Đĩa

`--keep-features` giữ lại gallery/query memmap để lần resume sau có thể backfill cửa sổ candidate.
Đây là **chủ ý**, không phải rác:

| model | `work/<model>/features` |
|---|---|
| `conesep` | 2.9 GB |
| `intent` | 2.3 GB |
| `habit` | 2.3 GB |
| `hint` | 2.3 GB |
| **tổng features** | **≈ 9.8 GB** |

Export candidate thật (thứ **không được** xoá): **116 MB** tổng cộng
(`habit` 43 MB — còn 3 file `.tmp` trơ của lượt bị kill; `conesep`/`intent` 24 MB, `hint` 25 MB).

`df -h /home`: **97 GB còn trống** (84% dùng). Đủ chỗ cho cả bốn feature cache, nhưng nếu chạy thêm
nhiều model thì dọn feature trước:

```sh
rm -rf workbench/artifacts/composition/runs/<run_id>/work/<model>/features
```

Sau khi dọn, `--resume --topk-depth 50` trên run đó sẽ báo `EXPORT_INCOMPLETE` (đúng sự thật: không
còn cache để dựng lại cửa sổ) — export đã có sẵn trên đĩa thì vẫn đọc được bình thường.

---

## 8. Cách resume nếu cohort bị dừng

```sh
cd /tmp/evalwork
# model chưa xong, còn cache features:
DRIVE_RESUME=0 DRIVE_TIMEOUT_S=21600 setsid nohup \
  /home/heheboiz/data/cir/workbench/envs/ctl312h/bin/python -u drive.py <model> \
  >> /tmp/evalwork/cohort_rest.log 2>&1 &
```

`DRIVE_RESUME=0` là **bắt buộc** khi cần export candidate. Chỉ dùng `DRIVE_RESUME=1` khi rank file đã
có và chỉ muốn backfill cửa sổ từ cache.

Kiểm tra trạng thái: `/tmp/evalwork/<model>.start`, `/tmp/evalwork/<model>.end`,
`/tmp/evalwork/drive_summary.json`, `/tmp/evalwork/cohort_rest.log`.
