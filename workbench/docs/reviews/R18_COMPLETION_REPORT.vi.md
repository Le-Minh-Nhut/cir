# R18 — Báo cáo hoàn thành

Ngày: 2026-10-23 → 2026-10-24. Nhánh `research/cir-failure-workbench`.

## 1. Revision, push, đồng bộ remote

| Mục | Giá trị |
|---|---|
| **Revision đã kiểm thử** | `ca10235` (production tree; verify bằng production-tree equality) |
| **HEAD đã push** | `506cb13` |
| `origin/research/cir-failure-workbench` | `506cb13` — **khớp** |
| `main` | `8ebcc74` — **không đổi** |
| Working tree | sạch |
| Số commit trong R18 | 30 (từ `daab57f`) |
| Fresh checkout | clone lại `506cb13`, thấy **17 model / 4 protocol** — đã kiểm bằng lệnh, không phải suy đoán |

Revision production cuối cùng là `ca10235`; các commit sau chỉ đổi docs (đã verify bằng
`git diff ca10235 HEAD -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'` → rỗng).

## 2. Các defect đã xác nhận — nguyên nhân, fix, kiểm chứng

Mỗi cái đều **reproduce trước**, rồi mới sửa, rồi **kiểm lại bằng lệnh chạy thật**.

### D1 — Doctor từ chối source vendored hợp lệ
**Nguyên nhân:** `doctor.source_state` gọi `git rev-parse HEAD` trong checkout vendored (không có
`.git` lồng nhau). Git đi lên repo CIR và trả commit của CIR; doctor so với pin upstream và từ chối.
**Bằng chứng:** LIMN pin `7d7bc9b` bị báo `3fcf11f`/`dirty`, đổ lỗi `src/train_cirr.py` — file ở
**gốc CIR**.
**Fix:** route qua `vendored_source.source_provenance` (authority sẵn có). **Kiểm:** doctor và
authority khớp trên LIMN/HABIT/ENCODER.

### D2 — `NO_CLEAN_CHECKPOINT` chặn checkpoint nhiễu đã chọn
**Nguyên nhân:** marker đó nghĩa "không có bản clean", nhưng doctor đọc thành "variant này không
dùng được". **Fix:** đánh giá theo fact của variant; chỉ kiểm variant được chọn. **Kiểm:** chọn 20%
mà không cần cài 50%/80%.

### D3 — Tiến trình con observer không có `TMPDIR` riêng
**Nguyên nhân:** `isolated_cache_root` dùng `tempfile` ⇒ cache ~11.9 GiB rơi vào `/tmp` (tmpfs RAM).
**Bằng chứng:** `observer exited -9`; di chứng `/tmp/cir-replay-cache-*` **5.3 GB**.
**Fix:** cấp `TMPDIR` 0700 theo run.

### D4 — Gate observation so với chuỗi bundle của LIMN
**Nguyên nhân:** ENCODER có một checkpoint và không có bundle nên không bao giờ khớp ⇒ observer bị
bỏ im lặng. **Fix:** khai `per_query_checkpoint` trong registry. **Kiểm:** ENCODER export chạy thật.

### D5 — `environment_required` thiếu cho model khai `python_env_var`
**Nguyên nhân:** `python_executable` trả interpreter workbench. **Bằng chứng:** ENCODER chết
`No module named 'open_clip'`. **Fix:** khai đúng, **chỉ cho model có bằng chứng chạy thật**.

### D6 — `pipeline.all_stages` tham chiếu `model_obj` không tồn tại
**Bằng chứng:** 3 suite fail. **Do subagent đưa vào; chỉ lượt chạy full accounting bắt được.**

### D7 — Asset LAION thiếu digest
**Fix:** ghi `lfs.oid` **do publisher công bố**; local hash = registry = HF oid.

### D8–D10 (FixB) — authority vendored mượn commit repo ngoài; interpreter; readiness
FixB tự bắt regression của chính nó rồi sửa, giữ `ready=False` trung thực.

### **D11 — Query image khác nhau giữa aggregate và exporter** (lớn nhất)
**Nguyên nhân:** constructor iLearn nướng tensor query lúc dựng dataset qua `get_img(..., stage=0)`,
mà replay truyền `[preprocess_train, preprocess_val]` ⇒ query qua `RandomResizedCrop` (stochastic);
gallery `stage=1` (deterministic).
**Bằng chứng:** hai lần dựng dress khác **3994/4034** tensor query; gallery giống hệt.
**Giải thích cũ của chính tôi — "cuDNN nondeterministic" — SAI, đã bỏ khỏi docs.**
**Fix:** `seeded_dataset_construction` (seed + khôi phục RNG state) ở cả 4 nơi dựng dataset.
**Kiểm:** 0/4034 khác.

### D12 — Cache native dựng hai lần mỗi run
**Bằng chứng:** `[Errno 28] No space left on device` trong `save_obj`; cache để lại **10.4 GB**.
**Fix:** `run_cache_root` + `WORKBENCH_REPLAY_CACHE_ROOT` (một cache/run) + xoá sau khi dùng.
**Kiểm:** log dir 3 MB thay vì 10.4 GB.

### **D13 — Lượt đánh giá thứ hai không tái lập được lượt thứ nhất** (quyết định)
**Bằng chứng đo được:** hai tiến trình, cùng seed, cùng input, cuDNN đã ghim deterministic và
`CUBLAS_WORKSPACE_CONFIG` đã set ⇒ **feature hash khác nhau** (`82dc55fc…` vs `af784fa2…`).
**Kết luận:** không cấu hình nào làm pass thứ hai khớp pass thứ nhất ⇒ observer **không được** tự
chạy pass riêng.
**Fix:** aggregate ghi **chính feature** nó đã tính (`WORKBENCH_OBSERVATION_OUT`); export xếp hạng
trên đó. `FeatureRecorder` bọc chính method của model nên preprocessing/batching/dtype không đổi.

### D14 — Số học quanh feature cũng là một phần của interface
**Bằng chứng:** xếp hạng bằng `float64` thay vì `float32` làm normalize round khác ⇒ R@1 lệch 1 hit
ở dress và toptee. Tie order cũng phải đúng `np.argsort(-sims)` như native.
**Fix:** đúng dtype + đúng lời gọi native.

### D15 — PTHA: layout source lồng nhau
Checkout giữ thư mục gốc repo bên trong ⇒ code ở `<checkout>/MTST_PTHA/src`, nhưng cả hai chỗ gọi
trỏ `<checkout>/src` (không tồn tại). **Fix:** một resolver `source_tree()`. **Kiểm:** preflight
`NO PROBLEMS` khi có opt-in, **từ chối** khi không.

### D16 — PTHA: không khai environment
Chạy bằng interpreter workbench (Python 3.13) ⇒ `ImportError: cannot import name
'apply_chunking_to_forward' from 'transformers.modeling_utils'`. Checkout **có** `requirements.txt` +
README pin version ⇒ khai từ chính nguồn đó. `manage_environment.py --create` nay thấy spec.

### D17 — TME cần protocol riêng
`precompute_evaluation.py:83` chỉ mask reference khi `category == 'CIRR'` ⇒ trên FashionIQ reference
**vẫn là ứng viên**, gallery là toàn bộ split val theo category. Không khớp 3 protocol cũ ⇒ khai
`fashioniq_val_split_ref_included`.

### D18 — Frontend thiếu protocol mới (defect thật)
Selector lọc theo `run.protocol_id`; protocol không có trong UI ⇒ run TME **vô hình**. Đã thêm vào
`protocols.ts` và build lại (guard staleness bắt được ngay).

### D19 — Nhiều test pin cứng danh sách thay vì pin quy tắc
`test_protocols`, `test_replay`, `test_limn_query_export_protocols`, `test_frontend_staleness`,
`test_pc_launcher`, `test_adversarial_orchestration`, `test_execution_graph_and_dependencies`,
`test_fresh_clone_delivery`, hai env-contract. Sửa theo hướng **đọc registry** hoặc **assert bất biến**,
có comment giải thích. Đã nhờ subagent review độc lập xem cái nào là *masking* — kết luận: không cái
nào che mất bảo vệ.

## 3. Bảng 17 model — variant, điều kiện, trạng thái thực thi

| # | model | variant chọn | điều kiện | đã chạy thật | blocker còn lại |
|---|---|---|---|---|---|
| 1 | `limn` | `base_iter0_all_categories` | eval noise 0%, train unrecorded | **FULL_VAL_VERIFIED** + parity | — |
| 2 | `encoder` | `fashioniq` | eval noise 0% | **3 category GPU thật**, R@10 56.076 vs paper 56.13 | `PAPER_PARITY_CHECKED` chưa chạy |
| 3 | `hint` | `fashioniq` | — | chưa | env + đĩa |
| 4 | `pair` | `pair_b1`/`pair_b2` | training dataset **unverified**, exploratory | **NOT RUN** | pickle cần `model_vssp` — **không có ở mọi revision công khai** |
| 5 | `clvc_net` | — | — | chưa | ckpt + env |
| 6 | `tgcir` | — | — | chưa | ckpt + env |
| 7 | `habit` | **`fiq_n02`** | **train noise 20% / eval 0%** | **READY**, chưa chạy | **đĩa: cần 11.58 GiB** |
| 8 | `intent` | **`fiq_n02`** | **train noise 20% / eval 0%** | ckpt tải + hash khớp | env + đĩa |
| 9 | `conesep` | **`fiq_n02`** | **train noise 20% / eval 0%** | metadata verify | chưa tải ckpt + env |
| 10 | `airknow` | — | **20% HOÃN** | — | chỉ có 50%/80%; **không** thay bằng 50% |
| 11 | `dcnet` | — | — | chưa | ckpt |
| 12 | `csmcir` | `fashioniq` | — | chưa | env + đĩa |
| 13 | `combiner_rn50x4_noft` | — | — | chưa | ckpt |
| 14 | `clip4cir_rn50x4_fullft` | — | — | chưa | ckpt |
| 15 | `sprc` | `sprc_fiq_unresolved`, `sprc_fiq_vitl` | — | chờ file tay | hướng dẫn đã sẵn sàng |
| 16 | `ptha_mtst` | **`cirr_transfer`** | **CIRR→FashionIQ, cross-dataset transfer, exploratory** | preflight PASS | env py3.9/torch2.0.1 (~5 GB) |
| 17 | **`tme`** | `fashioniq_0` | **train noise 0% / eval 0%**; **20% `CHECKPOINT_UNAVAILABLE`** | source vendored, wrapper + adapter | tải ckpt Kaggle (2.5 GB) + env |

Năm model ngoài registry (RankVR, Css-Net, MAPNet, MPCIR, ReCon): **không có checkpoint công khai đã
xác lập**. MPCIR: tài nguyên công khai tìm được là **dataset**, không phải checkpoint. Đã ghi lý do,
**không** dựng stub.

## 4. PAIR / PTHA / TME riêng

### PAIR B1 và B2 — **NOT RUN**
Cả hai tải thật và **khớp publisher oid**: B1 `c4d861ec…5fbf2` (694 958 565 B), B2 `74346fd7…eb3f03`
(694 872 371 B). Nên **không** phải lỗi tải hỏng.
`torch.load(..., weights_only=True)` dừng ở `GLOBAL model_vssp.Encoder`. Lỗi xảy ra **trước** khi có
state_dict ⇒ **không tồn tại** báo cáo key/shape — không bịa.
Kiểm độc lập: `models/` của repo tác giả chỉ có `MHTransformer.py` + `model.py`; `commits?path=`
cho `model_vssp.py` rỗng; bản phát hành nguồn đầu tiên `5991d554` cũng vậy ⇒ **không phải pin cũ**.
Khác biệt cấu trúc có bằng chứng: `transformer_c`/`transformer_d`, `backbone.weight_chanelImg`,
`weight_chanelImg_t`, `cross_t2i` chỉ có trong pickle; `_a` pickle 4 head/2 layer vs pinned 8/1;
`_b` pickle `MHTransformer_dis` vs pinned `MHTransformer_rev` (bản pinned là reconstruction của
workbench). `local_weight=(8,)` **khớp** `local_token_num=6` nên **không** dùng làm bằng chứng.
Operator path cũng từ chối: `COMMAND_BLOCKED_SOURCE_ENTRYPOINT`. **Remedy**: tác giả phải công bố
`model_vssp` + `MHTransformer_dis`.

### PTHA+MTST — checkpoint tải thật, 2 defect sửa xong, chưa chạy
Checkpoint **2 720 352 973 B**, sha256 **khớp publisher** `1c4dab86…`. Inspect an toàn: dict 2 key
`['epoch','Blip2QformerCirCat']`, epoch 9, **857 tensor**; `visual_encoder` 511, `Qformer` 336.
**Blocker**: cần env Python 3.9 + torch 2.0.1 (~5 GB).
CIRR native sanity-check: **dữ liệu CIRR không có** trong data root ⇒ blocker ngoài riêng.

### TME — đã tích hợp, chưa chạy
Entry thứ 17, pin `394aad79df2c104feb32a2c96132582012765062`, vendored 379 file, protocol riêng,
wrapper `replay/tme.py`, `TMEAdapter`. `doctor --scope real --model tme` → `source: OK`,
`command: REPLAY_COMMAND_IMPLEMENTED`.
**20% yêu cầu: `CHECKPOINT_UNAVAILABLE`** — Kaggle chỉ có `fashioniq-0.0-best` và `cirr-0.0-best`.
**Không** thay bằng 0%, **không** tự làm nhiễu val, **không** mượn ckpt model khác.
**Giấy phép**: GitHub metadata `license: null`, cây gốc không có `LICENSE` ⇒ **không** khai.

## 5. Dữ liệu/asset đã tải thật (không commit)

| Asset | Kích thước | Kiểm chứng |
|---|---|---|
| HABIT `HABIT-FIQ_N0.2.pt` | 2 730 715 930 B | sha256 **khớp** `814d2d0c…` |
| INTENT `intent_fiq_0.2.pt` | 2 722 478 522 B | sha256 **khớp** `2a504cad…` |
| PTHA `tuned_mtst_cirr.pt` | 2 720 352 973 B | sha256 **khớp** `1c4dab86…` |
| PAIR `pair-B1.pt` / `pair-B2.pt` | 694 958 565 / 694 872 371 B | sha256 **khớp** publisher oid |
| ENCODER `fashioniq.pt` | 631 879 365 B | khớp registry (từ trước) |

Metadata ConeSep 20% (size + sha + revision) **verify độc lập qua Hub API** và ghi vào registry;
HABIT/INTENT nay **pin theo revision** thay vì `resolve/main`.

## 6. Test — số lượng, skip, fail, frontend, E2E

**1833 passed + 1 failed + 18 skipped = 1852 collected** — **khớp tuyệt đối** với `--collect-only`.
Chạy chính: single process 37:29 (loại `gpu_records`) + 35 test `gpu_records` đo riêng.

| Loại | Kết quả |
|---|---|
| Full suite | **1833 passed, 1 failed, 18 skipped** |
| Failure duy nhất | guard đếm test của chính trang docs — **không phải defect production** |
| Frontend build | ✓ (`npm run build`) |
| TypeScript | ✓ (`tsc --noEmit`, rc=0) |
| Software E2E | doctor/preflight/verify_run/index/API trên đường operator thật |
| Real GPU E2E | **ENCODER 3 category** (R@10 56.076), feature-hash determinism test, GPU capability record |
| Đĩa | Hai lượt single-process **bị cắt bởi đĩa đầy**; errors là disk-full, **không** tính |

## 7. Lệnh operator còn lại

```sh
# HABIT (READY) — cần ≥12 GB trên volume chứa TMPDIR
TMPDIR=<ext4 ≥12GB> WORKBENCH_HABIT_PYTHON=$PWD/workbench/envs/habit/bin/python \
  .venv/bin/python -m workbench.scripts.evaluate_models \
  --model habit --checkpoint fiq_n02 --protocol fashioniq_full_gallery_ref_excluded \
  --dataset-root $PWD/data/fashionIQ_dataset --run-id <id>

# SPRC — đặt file tay rồi verify
cp ~/Downloads/<main> workbench/artifacts/checkpoints/sprc/sprc_fiq.pt
.venv/bin/python workbench/scripts/download_checkpoints.py \
  --model sprc --checkpoint sprc_fiq_unresolved --verify-only

# PTHA transfer — dựng env rồi chạy
uv venv --python 3.9 workbench/envs/ptha-mtst && uv pip install --python ... \
  --index-url https://download.pytorch.org/whl/cu118 torch==2.0.1 torchvision==0.15.2
export WORKBENCH_PTHA_MTST_PYTHON=$PWD/workbench/envs/ptha-mtst/bin/python
```

Công thức env đầy đủ: `workbench/docs/MODEL_ENV_RECIPES.md`.

## 8. Subagent — model/effort thật, kết quả

| Agent | Model / effort | Việc | Kết quả |
|---|---|---|---|
| `FixBAndC` | `cx/gpt-6-luna` **xhigh** | Fix B+C | 164 passed; **tự bắt regression của chính nó** rồi sửa |
| `TmeRecon` | `cx/gpt-6-luna` **xhigh** | Recon TME | pin, data, assets, **20% không tồn tại** |
| `PairWork` | `cx/gpt-6-luna` **xhigh** | PAIR B1/B2 | **NOT RUN** + bằng chứng; tự sửa 2 lần overstatement |
| `SprcAndPtha` | `cx/gpt-6-luna` **xhigh** | SPRC doc + PTHA wrapper | tự sửa **2 lần** sai (kể cả "checkout missing") |
| `MaskingReview` | `reviewer` `cx/gpt-6-luna` **xhigh** | Review test đã sửa có che mất bảo vệ không | Không cái nào masking; xác nhận fixture theo quy ước cũ |

Không dùng DeepSeek, không dùng Gemini. Mỗi kết luận tôi **tự kiểm lại** — kể cả tự tìm ra
`model_vssp` và dtype bug.

## 9. Blocker ngoài — có bằng chứng

1. **Đĩa**: cần ~12 GB cho một lượt iLearn 3-category; volume 173 GB đã dùng ~97%.
2. **PAIR**: thiếu định nghĩa lớp `model_vssp`/`MHTransformer_dis` từ tác giả.
3. **TME 20%**: chưa từng được công bố.
4. **PTHA**: cần env ~5 GB; CIRR dataset không có ở local.
5. **SPRC**: chờ bạn tải file tay.

Không blocker nào được mô tả như "đã hoàn thành".
