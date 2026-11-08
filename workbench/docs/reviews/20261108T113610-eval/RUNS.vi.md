# RUNS — nhật ký chạy thực tế

Mọi lệnh dưới đây chạy trên `/home/heheboiz/data/cir`, GPU RTX 5070 Ti (16 GB), **tuần tự một model
một lúc**. Control interpreter: `workbench/envs/ctl312h/bin/python` (CPython 3.12.11).

Biến môi trường chung cho cả phiên chạy:

```sh
export WORKBENCH_RUNTIME_ADAPTATION=sm_120
export WORKBENCH_HABIT_PYTHON=/home/heheboiz/data/cir/workbench/envs/ilearn39-adapted/bin/python
export WORKBENCH_INTENT_PYTHON=$WORKBENCH_HABIT_PYTHON
export WORKBENCH_HINT_PYTHON=$WORKBENCH_HABIT_PYTHON
export WORKBENCH_CONESEP_PYTHON=$WORKBENCH_HABIT_PYTHON
export WORKBENCH_PTHA_MTST_PYTHON=/home/heheboiz/data/cir/workbench/artifacts/environments/cir-005ef4e1b02999a5/bin/python
export WORKBENCH_ENCODER_PYTHON=/home/heheboiz/data/cir/workbench/envs/pair-adapted/bin/python
export WORKBENCH_LIMN_PYTHON=/home/heheboiz/data/cir/workbench/envs/limn-adapted/bin/python
export WORKBENCH_TME_PYTHON=/home/heheboiz/data/cir/workbench/envs/tme-adapted/bin/python
unset WORKBENCH_CIRCUS_SYNTHETIC_FIXTURE
```

> `WORKBENCH_RUNTIME_ADAPTATION=sm_120` là **bắt buộc** cho `habit`/`intent`/`hint`/`conesep`:
> không có nó thì contract lịch sử (py 3.8.10 / torch 2.1.0) được áp dụng và interpreter 3.9 bị từ chối.
> Mọi run dưới đây vì vậy là **ADAPTED RUNTIME**, không phải reproduction nguyên bản của paper.

## 0. Profile dùng để chạy

`workbench/docs/reviews/20261108T113610-eval/circus_cohort.yaml`

Lý do phải có profile riêng: `workbench/configs/pc_autopilot.all_available.yaml` khai
`paths.checkpoints: ../artifacts/checkpoints`, thư mục này **chỉ chứa `csmcir/`**. Trọng số thật
nằm ở `artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads/` (đúng
`WORKBENCH_CHECKPOINT_ROOT` trong `workbench/config/workbench.env`). Chạy bằng profile shipped thì
8/9 model báo `checkpoint not staged` — đó là lỗi cấu hình đường dẫn, không phải lỗi model.

## 1. GPU probe chính thức (`--verify-gpu-only`)

Bắt buộc trước khi chạy thật, vì `verify_environment` chỉ đạt `GPU_VERIFICATION_PENDING` cho tới khi
probe này chạy. Cả bốn model dùng chung một env nên dependency fingerprint khác nhau theo contract.

| model | checkpoint | lệnh | exit | `compute_capability` | tensor thật | `capability_verified` | log |
|---|---|---|---|---|---|---|---|
| habit | fiq_n02 | `evaluate_models.py --model habit --checkpoint fiq_n02 --verify-gpu-only` | 0 | 12.0 | có (`computed: true`) | true | `logs/gpu_probe_habit.log` |
| intent | fiq_n02 | `… --model intent --checkpoint fiq_n02 --verify-gpu-only` | 0 | 12.0 | có | true | `logs/gpu_probe_intent.log` |
| hint | fashioniq | `… --model hint --checkpoint fashioniq --verify-gpu-only` | 0 | 12.0 | có | true | `logs/gpu_probe_hint.log` |
| conesep | fiq_n02 | `… --model conesep --checkpoint fiq_n02 --verify-gpu-only` | 0 | 12.0 | có | true | `logs/gpu_probe_conesep.log` |

Trường ghi nhận: `workbench/artifacts/gpu_capabilities/gpu_capability.json` +
`workbench/artifacts/gpu_capabilities/<sha256>.json`.

Lần chạy đầu của `intent` exit 137 vì **tôi** kill nhầm cả cây process khi dừng một vòng probe quá
rộng (agent provisioning đã cảnh báo phải kill cả cây). Chạy lại riêng lẻ: exit 0. Không phải lỗi model.

## 2. Run thật

Cú pháp mỗi model — **một lệnh duy nhất** cho cả ba subset, không tách lệnh (vì `run_id` phụ thuộc
thành viên subset, tách lệnh sẽ mã hoá lại gallery 74.381 ảnh mỗi lần):

```sh
workbench/envs/ctl312h/bin/python workbench/scripts/composition_eval.py \
    --config workbench/docs/reviews/20261108T113610-eval/circus_cohort.yaml \
    --models <MODEL> --subsets full,sf,validated --resume
```

`--smoke` **không** được chạy trước, vì smoke dùng query hash khác → `run_id` khác → mất memmap
gallery đã encode.

### Trạng thái từng model

| model | trạng thái | ghi chú |
|---|---|---|
| `habit` | **RUNNING** | xem §3 |
| `intent` | QUEUED | trong queue tuần tự, chạy sau `habit` |
| `hint` | QUEUED | |
| `conesep` | QUEUED | |
| `tme` | QUEUED | sẽ tạo NEW_RUN, ghi đè nhãn PRE_EXISTING nếu hoàn tất |
| `ptha_mtst` | QUEUED | giữ nhãn TRANSFER |
| `encoder` | **BLOCKED** | gate từ chối: py 3.11.15 vs contract 3.10 (đã xác nhận, BLOCKERS §B2) |
| `limn` | **BLOCKED** | cùng lý do py; agent khác xử lý 119 query unscored |
| `csmcir` | **BLOCKED** | adapter tự từ chối (BLOCKERS §B1) |

**Queue tuần tự**: `/tmp/evalwork/queue.sh` (PID ghi ở `/tmp/evalwork/queue_pid.txt`), chạy
`intent → hint → conesep → tme → ptha_mtst`, mỗi model một lần, log gộp ở `/tmp/evalwork/queue.log`.
Queue chờ GPU lock nên không có hai model chạy đồng thời.

## 3. Run đang chạy khi báo cáo

**`habit` / `fiq_n02`**

- Lệnh: `/tmp/evalwork/run_model.sh habit`, tương đương lệnh ở §2 với `--models habit`
- Driver PID: `3749218` (evaluator) / worker `3754439`; run trước đó `3719035` bị kill nhầm — xem dưới.
- GPU đã cấp: 3654 MiB (`nvidia-smi --query-compute-apps`)
- Log: `/tmp/evalwork/run_habit.log`
- Bản sao log trong evidence: chép vào `logs/` khi run kết thúc
- Thời điểm bắt đầu: `$(cat /tmp/evalwork/habit.start)` (epoch), kết thúc ghi vào `/tmp/evalwork/habit.end`
- Thời lượng dự kiến: ~40 phút cho gallery encoding + 3 condition pass
- **Cách resume**: chạy lại đúng lệnh trên. `--resume` dùng lại feature cache nếu hash + cấu hình khớp.
  KHÔNG chạy `--smoke` trước (khác query hash → mất cache).

### Sự cố trong phiên: tự kill mất run, đã resume

Run `habit` **bị tôi kill nhầm** lúc `11:53` khi đang dọn một queue chạy song song. Đây đúng là lỗi
`AGENTS.md` cảnh báo: kill cả cây process thì evaluator chết theo.

Thiệt hại và cách khắc phục — **không mất dữ liệu encode**:

```
work/runs/<run_id>/work/habit/features/gallery_global.npy = 2,437,316,736 bytes
74381 x 32 x 256 x 4 = 2,437,316,736  → khớp chính xác, memmap gallery ĐÃ HOÀN TẤT
```

Lúc bị kill, gallery đã encode xong; nó chết ở bước query/rank. Restart lúc `11:53:29` với `--resume`
dùng lại memmap nên không phải encode 74.381 ảnh lần hai. GPU worker mới PID `3754439` (3 654 MiB).

Bài học ghi lại để không lặp: **không** kill theo pattern `composition_eval` khi nhiều model đang ở
cùng cây process; dừng đúng PID của driver.

### Caveat F7 áp dụng cho chính run này

Run `habit` đang chạy **trước khi** `runner.py` được sửa, nên `summary/full.json` của nó sẽ mang
`"reference_eligible": true` dạng **literal** ở `runner.py:1007` (và trong identity ở `:751`).
Đó **không** phải giá trị đo được — xem BLOCKERS §B7 và `workbench/docs/reviews/eval-runner-requests.md`.

Giá trị **đo được** cho `habit` thì đã có và đúng: `measured: true, reference_eligible: true,
reference_rank: 1` (adapter `score_dimension = 256`). Nhưng nó nằm ở phía adapter, **không** tự chảy
vào artifact cho tới khi agent sở hữu `runner.py` áp bản sửa.

→ Khi đọc artifact của bất kỳ run nào trong phiên này: `reference_eligible: true` là **khai báo**,
chưa phải **đo**. Đừng trích nó như bằng chứng reference-eligible.

## 4. Artifact cũ — KHÔNG phải NEW_RUN

Các thư mục dưới `workbench/artifacts/composition/summary/` có từ commit `6421188`:

| model | run_id | execution_status | query_count | MM R@10 | MM R@50 | MM nDCG | gap |
|---|---|---|---|---|---|---|---|
| tme / fashioniq_0 | `ad477312aa57c1b9114f9e51` | COMPLETED | 6003 | 34.0163 | 53.9564 | 0.32039 | 0.35860 |
| ptha_mtst / cirr_transfer | `315dd590eeb03d2c78d7937a` | COMPLETED | 6003 | 19.4736 | 34.4494 | 0.22493 | 0.24362 |
| encoder / fashioniq | `6a528e877c328772e2c7bb28` | COMPLETED | 6003 | 31.0178 | 49.5752 | 0.29586 | 0.47230 |
| limn / base_iter0_all_categories | `10173dc45dea8c90fd2a284b` | PARTIAL | 6003 | 30.0646 | 48.7593 | 0.29084 | 0.54174 |
| habit / fiq_n02 | `e0dd04a46ef7a58334b9e353` | FAILED | 6003 | — | — | — | — |

**Hai artifact dễ đọc nhầm là full run** (query_count = 8 → SMOKE-scale, loại khỏi mọi số liệu):
`encoder_fashioniq_eda30a37c0bc17ac43844f09`, `limn_base_iter0_all_categories_3f86f58483b368377e2c151e`.

Nhãn đúng cho cả bảng: **PRE_EXISTING_UNVERIFIED** — chưa có run mới nào trong phiên này xác nhận lại.
