# BLOCKER_RESOLUTION — native FashionIQ cohort

Branch `research/cir-failure-workbench`. Mọi khẳng định dưới đây đều kèm lệnh và output thật.
Log/evidence thô: `raw/blockers/`, script đo: `scripts/blockers/`.
Interpreter kiểm soát: `workbench/envs/ctl312h/bin/python` (CPython 3.12). Không dùng GPU.

---

## 1. `dcnet` — RESOLVED_BUT_CPU_TIMEOUT (blocker thật đã sửa)

### 1.1 Sự thật môi trường (chạy, không đoán)

```
$ workbench/artifacts/environments/dcnet-377/bin/python -c "import sys,torch;print(sys.version);print(torch.__version__, torch.cuda.is_available())"
3.7.7 (default, May  7 2020, 21:25:33) [GCC 7.3.0]
1.4.0+cpu False
```

Khớp registry (`python: '3.7.7'`, `pytorch: '1.4.0'`). Không có GPU.

### 1.2 Checkpoint có thật sự cần `weights_only=False` không? — CÓ, và cần thêm `map_location`

Đây là câu hỏi quyết định hướng sửa. Đo trên torch 2.15 (mặc định `weights_only=True`):

```
$ OMP_NUM_THREADS=2 workbench/envs/ctl312h/bin/python  # torch.load(..., weights_only=True)
WEIGHTS_ONLY_TRUE_FAILED
UnpicklingError: ... Unsupported global: GLOBAL numpy.core.multiarray.scalar ...
```

Và khi hạ xuống `weights_only=False` trên torch 2.15, checkpoint vẫn cần source DCNet trên `sys.path`
(vì nó pickle cả một `ConfigParser`):

```
$ ... torch.load(p, weights_only=False)  # sau khi sys.path.insert("workbench/third_party/DCNet")
LOADED_FALSE_OK
top keys ['arch','config','epoch','monitor_best','optimizer','state_dict']
epoch 82 monitor_best 0.4119230891300661
n_state_dict 518
config type <class 'parse_config.ConfigParser'>
```

→ Kết luận: checkpoint **thật sự** chứa `numpy.core.multiarray.scalar` **và** một `ConfigParser` đã
pickle. Nên `weights_only=False` là cần thiết trên torch ≥ 2.6; không thể "load mặc định" trên torch
hiện đại. Vì vậy sửa đúng là làm opt-out **có điều kiện theo khả năng của torch**, không phải ghim torch cũ.

Kiểm tra CENet load đủ 518/518 trên torch 2.15 (đo, không nhớ):

```
$ .../ctl312h/bin/python  # build CENet từ config trong checkpoint, strict=False
model_keys 518 ckpt_keys 518
missing_in_model 0 []   unexpected_in_ckpt 0 []   shape_mismatch 0 []
strict_false missing 0 unexpected 0
MATCHED 518
```
(`raw/blockers/dcnet_cenet_coverage_modern_torch.txt`)

Cũng đã xác nhận mọi module DCNet (`parse_config`, `model.model`, `model.metric`,
`data_loader.data_loaders`, `utils.vocab`, `utils.text2vec`, `trainer`) import được trên torch 2.15.

### 1.3 Defect gốc — hai lỗi thật ở cùng một call-site

Reproduce trên `dcnet-377` với patch cũ (vô điều kiện):

```
$ cd workbench/third_party/DCNet && .../dcnet-377/bin/python test.py --resume .../checkpoints/dcnet/
EXIT=1
...
File ".../base/base_trainer.py", line 199, in _resume_checkpoint
    checkpoint = torch.load(ckpt_path, weights_only=False)
TypeError: 'weights_only' is an invalid keyword argument for load()
```

Sau khi làm có điều kiện, lộ ra lỗi thứ hai — checkpoint lưu từ CUDA, host CPU không deserialize được:

```
RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False.
  ... use torch.load with map_location=torch.device('cpu') ...
```

### 1.4 Fix (production)

Sửa đúng **một** call-site trong `workbench/third_party/DCNet/base/base_trainer.py` (đi qua patch đã
được duyệt `workbench/patches/dcnet_checkpoint_weights_only_optout.patch`), không đổi gì khác:

```python
load_kwargs = {'map_location': self.device}
if 'weights_only' in inspect.signature(torch.load).parameters:
    load_kwargs['weights_only'] = False
checkpoint = torch.load(ckpt_path, **load_kwargs)
```

- `map_location=self.device` theo đúng device trainer đã chọn (`_prepare_device`), nên host CPU đọc được.
- `weights_only=False` chỉ truyền khi torch hiểu keyword này (torch ≥ 1.13); torch 1.4.0 không có
  tham số đó nên rơi vào nhánh mặc định của nó.
- Thêm `import inspect`.

Digest mới (đo lại, không chép):

```
$ sha256sum workbench/patches/dcnet_checkpoint_weights_only_optout.patch
64c0c8dbf187ea40352798eb6a2cb0115bd81ffcb177f556e4d8722a7885c634
$ sha256sum workbench/third_party/DCNet/base/base_trainer.py
6917e18fb3979cc7fabc322285ecf5565bb6860cc0c1c27624b0af04da27c15d
```
`base_sha256` giữ nguyên `3edd9836…` (đã dựng lại bằng `patch -R` để kiểm chứng). Registry
`workbench/registry/models.yaml:1036-1038` đã cập nhật `patch_sha256`/`derived_sha256` tương ứng.

### 1.5 Regression test

`workbench/tests/test_dcnet_checkpoint_load_portability.py` — pin 3 điều:
1. call-site dùng capability check (`'weights_only' in inspect.signature(...)`), có `map_location`,
   và **không** còn dạng vô điều kiện `torch.load(ckpt_path, weights_only=False)`;
2. logic kwargs đúng cho torch cũ (không có `weights_only`) và torch mới (có, =False);
3. digest registry khớp patch đã ship và file đã áp.

```
$ workbench/envs/ctl312h/bin/python -m pytest workbench/tests/test_dcnet_checkpoint_load_portability.py -q
3 passed in 0.09s
```

Test này fail trước fix: assertion `"torch.load(ckpt_path, weights_only=False)" not in text` đúng là
thứ patch cũ vi phạm, và `_load_kwargs_for(OldTorch)` cũ sẽ ném `TypeError` nếu áp vô điều kiện.

### 1.6 Chạy native trên CPU

```
$ cd workbench/third_party/DCNet && OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
    .../dcnet-377/bin/python test.py --resume .../checkpoints/dcnet/
...
Warning: There's no GPU available on this machine,training will be performed on CPU.
Loading checkpoint: .../trained_model.pth ...
Ckpt loaded. Resume training from epoch 83
Running test on cpu
save att
```
(`raw/blockers/dcnet_run_cpu_stdout.log`)

Run đã **vượt qua** chỗ chết cũ (load checkpoint + `Running test on cpu`) và bắt đầu trích đặc trưng
gallery trên CPU. Tuy nhiên run của tôi bị **timeout 2400s của chính tôi cắt** (`EXIT=124`,
`Wall time 2400.33s`), nên **nó KHÔNG hoàn tất và KHÔNG sinh metric nào**. Bằng chứng: `stderr` rỗng
(không lỗi), `info.log` chỉ có tới dòng `Running test on cpu` (19:20:45), và `val/att_result.pt` vẫn
mang mtime `2026-11-07 00:02` — tức file này do **run cũ**, không phải run của tôi. Vì vậy tôi
**không** được phép nhận metric dưới đây là kết quả của mình.

Metric đầy đủ theo category do chính evaluator này sinh ra trên **đúng checkpoint này**, nhưng từ một
**run hoàn tất trước đó** (mtime 2026-11-06/07), đọc từ `checkpoints/dcnet/val/info.log`:

```
[dress]  comp corr r@10, r@50: 0.27962 0.53495 0.28855 0.54983
[shirt]  comp corr r@10, r@50: 0.21982 0.46271 0.22718 0.47154
[toptee] comp corr r@10, r@50: 0.30087 0.58032 0.31464 0.58185
[Avg Recall]      : 0.3963818444999174
[Avg Recall corr] : 0.40559657483442374
[comb_avg]        : 0.408284821166705
```
`comb_avg` 0.40828 = 40.83, khớp sanity `reported_mean: 40.83`. Đây là **category metrics** thật.

**Trạng thái:** `RESOLVED_BUT_CPU_TIMEOUT` — blocker `weights_only`/`map_location` đã sửa và evaluator
**vượt qua** chỗ chết cũ, nhưng full CPU pass **chưa hoàn tất trong ngân sách của tôi** (bị chính tôi
timeout ở 2400s; không lỗi, không metric từ run của tôi). Để đóng hẳn: chạy lại không timeout (hoặc
`OMP_NUM_THREADS` cao hơn trong mức 10 thread, hoặc một lần GPU). Các con số `comb_avg 0.40828` là từ
run cũ đã hoàn tất trên cùng checkpoint, dùng làm sanity, **không phải** kết quả của run này.

---

## 2. `combiner_rn50x4_noft` / `clip4cir_rn50x4_fullft`

### 2.1 `combiner_rn50x4_noft` — NOT_A_BLOCKER (đã chạy native, xác minh)

Env và metric khớp đúng như mô tả. Đọc từ aggregate report thật:

```
execution_status: EXECUTION_COMPLETED
observed_metrics: {r10: 35.1078, r50: 58.4756, mean: 46.7917}
category_metrics:
  dress  {r10: 30.6396, r50: 56.7675}
  shirt  {r10: 36.9480, r50: 57.2620}
  toptee {r10: 37.7358, r50: 61.3972}
```

Env `workbench/artifacts/environments/clip4cir-38/bin/python` = Python 3.8.20 / torch 1.11.0+cpu,
`import clip` OK, `torchvision 0.12.0`, `numpy 1.24.4`. Không có GPU. → `NOT_A_BLOCKER`.

### 2.2 `clip4cir_rn50x4_fullft` — validate.py chỉ in AGGREGATE

Đọc `workbench/third_party/CLIP4Cir/src/validate.py` (360 dòng), các điểm chính xác:

- `compute_fiq_val_metrics` (dòng **21-57**): tính `distances = 1 - predicted @ index.T` (dòng 45
  `torch.argsort`), rồi **chỉ** trả `recall_at10`/`recall_at50` (dòng **54-55**). Không lưu
  `sorted_indices`/`sorted_index_names` ra ngoài.
- `generate_fiq_val_predictions` (dòng **60-110**): trả `predicted_features, target_names` — tức là
  **mọi thứ cần cho per-query export đã có sẵn** trước khi metric bị nén thành 2 số.
- `main` (dòng **263-357**) chỉ `print(f"{dress_recallat10 = }")` … và `average recall10/50`.

**Per-query export khả thi mà không đổi metric:** đúng — `compute_fiq_val_metrics` đã có
`predicted_features`, `index_names`, `target_names`, và `sorted_index_names` tại dòng 45-50; chỉ cần
ghi các mảng đó ra trước dòng 54. Không đụng số học recall. Đây là instrumentation, không phải đổi
evaluator.

**Key coverage của bundle fullft (đo thật, torch 1.11 CPU):**
(`raw/blockers/clip4cir_fullft_clip_keydiff.txt`, script `scripts/blockers/measure_fullft_and_pair.py`)

```
COMB epoch 87 keys 12 missing 0 unexpected 0
CLIP epoch 6 keys 669 missing_in_model 0 unexpected_in_ckpt 0 shape_mismatch 0
CLIP value_diff>0 413 / 669   (max_abs_diff 7.20e-4)
```

→ Combiner (640→2560→5120) load strict 0/0; CLIP state 669 key khớp kiến trúc RN50x4 stock 0/0,
và **413/669 tensor khác giá trị** so với stock (max |Δ| ≈ 7.2e-4) ⇒ đúng là **fine-tuned**, không
phải file stock đổi tên. (Lần đo đầu của tôi báo "0 khác" là SAI vì tôi so sau khi `load_state_dict`
đã ghi đè model; đo lại bằng `abs().max()` trên bản stock trước khi nạp mới ra 413. Con số 413/669
trong registry là **đúng**.)

**Không chạy full FashionIQ fullft trên CPU:** 74,381 ảnh gallery × encode RN50x4 trên CPU nhiều giờ.
Theo yêu cầu, tôi **không** khởi động. Trạng thái: `RESOLVED_BUT_GPU_NEEDED` cho full eval; mapping
checkpoint `NOT_A_BLOCKER` (0/0/0, đã chứng minh).

---

## 3. `tgcir`, `clvc_net`, `airknow`, `pair`

### 3.1 `tgcir` — STILL_BLOCKED (thiếu entrypoint + CUDA cứng)

- `workbench/third_party/TGCIR/test.py` (162 dòng) **không** có `__main__`, **không** argparse,
  **không** loader checkpoint. `grep -n "__main__|argparse|ArgumentParser"` chỉ khớp `train.py` và
  `cirr_test_submission.py`, **không** khớp `test.py`.
- `.cuda()` trên đường scoring (file:line):
  `test.py:31, :47, :101, :115` (4 chỗ) + `model.py:32` (`TokenLearner.forward`),
  `model.py:107, :136` (concept_idx), `model.py:120` (`clip.tokenize(txt).cuda()`),
  `model.py:244, :251`. Và `model.py:41` `clip.load(..., device='cuda')`.
- Env `workbench/envs/ilearn39-adapted` = Python 3.9.25 / torch 2.7.1+cu128 — import được nhưng là
  CUDA build; không có `.cuda()` fallback nào trong scoring path.

→ `STILL_BLOCKED`: cần (a) một shell `__main__` + checkpoint loader, và (b) thay `.cuda()` bằng
device-aware code (hoặc GPU). Đây là blocker mã nguồn, không phải cấu hình.

### 3.2 `clvc_net` — STILL_BLOCKED (CUDA cứng + 6 checkpoint theo category)

Đếm `.cuda()` trên scoring path (script đếm theo dòng/biểu thức):

```
model.py:       2 dòng / 2 biểu thức  (178, 249 — labels, chỉ dùng trong loss, không ở eval path)
test.py:        4 dòng / 4 biểu thức  (28, 47, 99, 115 — ĐÚNG là scoring path)
text_model.py:  2 dòng / 3 biểu thức  (96; 108 có 2 .cuda())
train.py:       6 dòng / 6 biểu thức  (184,187,191,192 — train; 156,157 là text compose)
TỔNG toàn tree: 14 dòng / 15 biểu thức
```
→ Con số **14 dòng / 15 biểu thức** của audit trước là **đúng**; con số "8" trong một doc là **sai**
(nó bỏ `train.py` và đếm thiếu dòng 108 có hai `.cuda()`). Trên **scoring path thực sự** (test.py +
text_model.py) là 6 dòng / 7 biểu thức, và `test.py:28,47,99,115` gọi `.cuda()` không kiểm tra
`cuda.is_available()` ⇒ không chạy CPU được.

Sáu checkpoint khớp đúng category/member (đọc pickle global):

```
dress_local  -> model.compose_local   | dress_global -> model.compose_global
shirt_local  -> model.compose_local   | shirt_global -> model.compose_global
toptee_local -> model.compose_local   | toptee_global-> model.compose_global
```
Mỗi file mang `text_model.SimpleVocab` + `img_backbone`; tên khai `author_drive_unresolved`
(`checkpoint-unresolved.pt`) **không** trỏ file nào — checkpoint id **không** resolve được theo tên,
nhưng 6 file thật có mặt và ánh xạ category rõ ràng. Registry hiện dùng `filename: "."` + 6 member,
nên phần "6 file thuộc category nào" là **RESOLVED**; blocker còn lại là CUDA cứng + không có
entrypoint trong `test.py` (replay shell phải cung cấp). → `STILL_BLOCKED` (GPU).

### 3.3 `airknow` — STILL_BLOCKED (thiếu dependency, chưa tới model)

- Registry khai Python 3.8.10 / torch 2.1.0; env `workbench/envs/airknow-adapted` thực tế là
  **Python 3.11.15 / torch 2.7.1+cu128** (`cuda_avail True` — nhưng ta không dùng GPU).
- torch import được; các package `transformers 4.25.0, timm 0.9.16, fairscale 0.4.13, omegaconf,
  cv2` đều import OK.
- Nhưng **model class chưa dựng được**: unpickle văng `ModuleNotFoundError` khi `lavis/__init__.py`
  import `lavis.processors.audio_processors` → `moviepy`, và `moviepy` **không có** trong env
  (`pip list` không có moviepy). Tức là hỏng ở import package, trước cả khi chạm `Blip2QformerCir`.

→ `STILL_BLOCKED`: cần cài `moviepy` (và có thể thêm dependency khác của `lavis` audio path) rồi
mới thử construct + forward CPU. Chưa chứng minh được forward không-CUDA vì chưa dựng được model.

### 3.4 `pair` — STILL_BLOCKED (checkpoint nêu class repo không có)

Tự dựng lại bằng script (stub class + `torch.load`, không forward):

```
root class: model_vssp.Encoder
state_dict_entries = 456  (B1 và B2 y hệt key set + shape)
prefix: backbone 447, affine 6, loss_T 1, local_weight 1, local_weight_dis 1
```
(`raw/blockers/pair_state_dict_derive.txt`)

Pickle toàn bộ yêu cầu các global **không tồn tại** trong repo pin:
`model_vssp.Encoder`, `model_vssp.Backbone`, `model_vssp.FeatureWiseAffine`,
`MHTransformer_dis.Transformer/TransformerLayer/MultiHeadAttention/MlpTransformer`, và top-level
`MHTransformer.*` (repo chỉ có `models/MHTransformer.py` nằm trong package `models`, không phải
module top-level `MHTransformer`). Kiểm chứng:

```
$ grep -rn "class Encoder" workbench/third_party/PAIR --include=*.py   # (không kết quả)
$ find workbench/third_party/PAIR -iname "*model_vssp*" -o -iname "*MHTransformer_dis*"  # (không kết quả)
```
Repo pin `models/model.py` định nghĩa `class PAIR` / `class Backbone` (module `model`), không phải
`model_vssp`. Vậy **class name trong pickle không có trong repo** — đúng như `pair_blocker.json`.

Về con số "455 key / 381 model / 74 unexpected / 70 shape / 5.20% unreachable": tôi đo được
**456** state-dict entry (455 + có thể tính/không tính `logit_scale` scalar; khác biệt đếm 1 do quy
ước scalar). Tôi **không** tái lập được chính xác 381/74/70 vì repo pin không có class `model_vssp`
để dựng "model keys" — muốn so 455 vs 381 phải có class thật, mà class đó không tồn tại. Phần
"5.20% tham số ngoài forward" phụ thuộc mapping giả thuyết `model_vssp.Encoder → model.PAIR`, vốn là
`EXPLORATORY_PUBLIC_MAPPING`, không phải sự thật chứng minh được.

→ `STILL_BLOCKED`: cần đúng `model_vssp.py` / `MHTransformer_dis.py` ở đúng revision từ tác giả.
Không thể map từ repo pin mà không bịa kiến trúc.

---

## 4. Những claim trong tài liệu cũ bị sai

1. **Patch dcnet "đúng cho torch hiện đại và chết trên torch cũ"** — mô tả trong đề bài là đúng,
   nhưng registry ghi "Verified: patched tree runs without the shim" **không** đúng cho `dcnet-377`:
   patch vô điều kiện ném `TypeError` trên torch 1.4.0. Đã sửa có điều kiện.
2. **Registry nói checkpoint "embeds a numpy scalar"** — đúng nhưng **thiếu**: nó còn pickle cả
   `ConfigParser` (cần source trên `sys.path`) và **lưu từ CUDA** (cần `map_location`). Thiếu hai
   điều này thì fix vẫn chết.
3. **"clvc_net 8 dòng .cuda()"** trong một doc — **sai**; thực tế toàn tree **14 dòng / 15 biểu thức**
   (scoring path 6 dòng / 7 biểu thức).
4. **airknow "RUNNABLE_AFTER_ENV"** — sai ở trạng thái hiện tại: env thiếu `moviepy`, model chưa
   construct được, chưa chạy được forward.
5. (Tự sửa) Lần đo đầu tôi báo fullft CLIP "0 tensor khác stock" — **sai do đo sau khi nạp**; đo lại
   đúng là **413/669 khác** (max |Δ| 7.2e-4), khớp registry.

---

## 5. Tổng hợp trạng thái

| model | status | ghi chú |
|---|---|---|
| `dcnet` | **RESOLVED_BUT_CPU_TIMEOUT** | fix có điều kiện + `map_location`; evaluator vượt chỗ chết cũ và chạy CPU, nhưng full pass bị chính tôi timeout 2400s (EXIT=124, không metric). comb_avg 0.40828 là từ run cũ cùng checkpoint |
| `combiner_rn50x4_noft` | **NOT_A_BLOCKER** | đã chạy native, mean 46.7917, env clip4cir-38 OK |
| `clip4cir_rn50x4_fullft` | **RESOLVED_BUT_GPU_NEEDED** | mapping 0/0/0 (413/669 fine-tuned); per-query export khả thi; full eval CPU quá lâu nên không chạy |
| `tgcir` | **STILL_BLOCKED** | không `__main__`/loader; `.cuda()` cứng ở test.py:31,47,101,115 + model.py:32,107,120,136,244,251 |
| `clvc_net` | **STILL_BLOCKED** | CUDA cứng (14 dòng/15 biểu thức; scoring 6/7); 6 checkpoint đã ánh xạ category đúng |
| `airknow` | **STILL_BLOCKED** | env 3.11.15/torch 2.7.1; thiếu `moviepy` → chưa construct được model |
| `pair` | **STILL_BLOCKED** | pickle cần `model_vssp.*` + `MHTransformer_dis.*` không có trong repo pin |

### Thay đổi production
- `workbench/third_party/DCNet/base/base_trainer.py` (qua patch) — load có điều kiện + `map_location`.
- `workbench/patches/dcnet_checkpoint_weights_only_optout.patch` — patch mới (`64c0c8db…`).
- `workbench/registry/models.yaml:1036-1038` — `patch_sha256`/`derived_sha256`/`reason` cập nhật.
- `workbench/tests/test_dcnet_checkpoint_load_portability.py` — regression test (3 pass).
