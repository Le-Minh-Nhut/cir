# UNFINISHED — inventory đầy đủ những gì chiến dịch này CHƯA hoàn thành

Ngày: 2026-11-09. Branch `research/cir-failure-workbench`.
Phạm vi: chỉ tài liệu. Không chạy GPU, không chạy test suite, không sửa production code.
Mọi khẳng định dưới đây kèm đường dẫn file hoặc lệnh tái lập. Con số nào tôi không đọc được từ file
thì ghi rõ là **KHÔNG xác minh được**, không suy đoán thay.

---

## 1. Mở đầu — chiến dịch kết thúc ở đâu, và "chưa hoàn thành" nghĩa là gì

### 1.1 Điểm dừng thật của chiến dịch

| trục | đã đóng | còn mở |
|---|---|---|
| Native FashionIQ aggregate | **9 / 15** target active có run hoàn tất | **6** model không có aggregate hoàn tất |
| G1 input parity | **1 / 15** (`limn`) | **14** model `NOT_RUN` |
| G2 forward/scoring parity | **1 / 15** (`limn`, kèm một sub-check FAIL có khai báo) | **14** model `NOT_RUN` |
| LIMN ambiguous queries | 0 / 2 giải được canonical (giữ `PARTIAL 6001/6003`) | 2 query vẫn unscored, không có đường giải |
| Performance | chỉ đo producer stage (decode + transform) | end-to-end throughput **chưa đo lần nào** |
| Test suite | 18 failure tồn tại trước, id set ổn định qua 4 sweep | không có sweep nào cho thay đổi tài liệu của chiến dịch này |

Nguồn đếm: `NATIVE_EVAL_MATRIX.csv` (17 hàng, cột `native_executed` / `native_verified` / `g1_status` /
`g2_status`), `GATES.csv` (hàng `Native completeness` = `FAIL`, `G1 input parity` = `NOT_RUN`,
`G2 forward/scoring` = `NOT_RUN`), `INPUT_FORWARD_PARITY.vi.md`.

### 1.2 Định nghĩa "chưa hoàn thành" dùng trong file này

Ba loại khác nhau, **không được trộn**:

1. **THIẾU ARTIFACT** — đường thực thi đã biết và đã từng chạy được, nhưng lần chạy đúng chưa xảy ra
   (hoặc chưa xảy ra trong ngân sách). Ví dụ: `dcnet` (blocker đã sửa, run bị timeout 2400 s),
   `clip4cir_rn50x4_fullft` (bundle load sạch, chưa chạy eval).
2. **THIẾU ĐIỀU KIỆN ĐỂ CHẠY** — không thể khởi động evaluator vì thiếu môi trường, thiếu
   entrypoint, hoặc code cứng CUDA. Đây là blocker **đo được từ source/env**, không phải phỏng đoán.
   Ví dụ: `clvc_net`, `tgcir`, `airknow`.
3. **KHÔNG THỂ KẾT LUẬN TỪ DỮ LIỆU** — không có đường nào để biết đáp án đúng mà không bịa. Ví dụ:
   `pair` (class không tồn tại trong repo pin), hai query LIMN mơ hồ category.

Một mục **không** thuộc "chưa hoàn thành": lỗi đã biết và đã sửa (`dcnet` `weights_only` +
`map_location`), hoặc khác biệt đã reconcile và không phải defect (455 vs 456 key của `pair`).
Những thứ đó nằm ở §4.

---

## 2. Bảng tổng hợp

| # | Mục | Trạng thái | Khối lượng còn lại | Điều kiện gỡ |
|---|---|---|---|---|
| 1 | `pair` native run | `BLOCKED_CHECKPOINT_MAPPING` (UNVERIFIED) | 1 model × 3 category native | tác giả publish `model_vssp` / `MHTransformer_dis`, hoặc mapping per-key trên class thật |
| 2 | `airknow` native run | `BLOCKED_ENV` | 1 model × 3 category native | cài `moviepy` (hoặc làm chuỗi import lavis audio-optional) + xác nhận lại contract Python |
| 3 | `clvc_net` native run | `BLOCKED_ENV` + `BLOCKED_DEVICE` | 1 model × 6 checkpoint (2 member × 3 category) | venv py3.7/torch1.6 (hoặc contract adapt `sm_120`) **và** device-guard cho 6 dòng `.cuda()` trên scoring path |
| 4 | `tgcir` native run | `BLOCKED_COMMAND` + `BLOCKED_ENV` + `BLOCKED_DEVICE` | 1 launch shell + 1 venv + device guard | viết `__main__`/argparse/loader (shell `workbench/replay/tgcir.py` mới một phần) + interpreter + device guard |
| 5 | `clip4cir_rn50x4_fullft` native eval | `READY_NOT_RUN` | 1 full FashionIQ eval | cấp ngân sách cho một lượt eval đầy đủ (hoặc GPU); per-query export là tuỳ chọn riêng |
| 6 | `dcnet` native eval | `RESOLVED_BUT_CPU_TIMEOUT` | 1 full CPU pass (hoặc GPU pass) | ngân sách > 2400 s, hoặc chạy GPU, rồi ghi metric |
| 7 | G1/G2 cho 13 model | `NOT_RUN` | 13 differential native-vs-adapter | differential trên một mẫu thật cố định, layer 1→5, tolerance khai báo trước khi đo. `limn` + `encoder` đã xong ⇒ **2/15** |
| 8 | LIMN 2 query mơ hồ | `PARTIAL 6001/6003` | 2 query canonical | metadata category per-query do tác giả publish, hoặc defect report lên maintainer release |
| 9 | Performance end-to-end | `NOT_MEASURED` | 1 benchmark isolated | đo một lượt eval đầy đủ trên box rảnh, có lặp |
| 10 | Test suite cho thay đổi của chiến dịch | `NOT_RUN` | 1 sweep serial | chạy sweep trên tree sạch sau khi tài liệu đóng băng |
| 11 | UI defect `.topbar` che `thead` | `KNOWN_PREEXISTING`, không sửa | 1 CSS fix + test | quyết định phạm vi thiết kế chung (không phải bản vá riêng panel LIMN) |
| 12 | `csmcir` native | `USER-DEFERRED` | — | người dùng bỏ hoãn |
| 13 | `sprc` native | `USER-DEFERRED` + `BLOCKED_CHECKPOINT_MAPPING` | — | người dùng bỏ hoãn **và** mapping checkpoint + adapter composition |

13 mục. Trong đó 6 mục là native run (1–6), 2 mục là verification (7–8), 5 mục là chất lượng đo
lường / hạ tầng / defer (9–13).

---

## 3. Chi tiết từng mục

### 3.1 `pair` — checkpoint pickle các class không tồn tại trong repo pin

**Đang thiếu gì.** Không có native run, và không có mapping checkpoint nào được xác minh. Trạng thái
registry: `checkpoint_mapping_status: UNVERIFIED`, `reproduction_status: NOT_RUN`
(`workbench/registry/models.yaml`, block `model_id: pair`).

**Bằng chứng (đo được, không suy ra).**

Pickle thô, decode `GLOBAL` opcode — `workbench/docs/reviews/20261109-native15-composition-limn/scripts/pair_pickle_globals.out`:

```
GLOBAL_COUNT 38
GLOBAL model_vssp Encoder
GLOBAL model_vssp Backbone
GLOBAL MHTransformer_dis Transformer
GLOBAL MHTransformer_dis TransformerLayer
GLOBAL MHTransformer_dis MultiHeadAttention
GLOBAL MHTransformer_dis MlpTransformer
GLOBAL model_vssp FeatureWiseAffine
MODULES: ['MHTransformer', 'MHTransformer_dis', '__builtin__', '_codecs', 'collections',
          'model_vssp', 'open_clip.model', 'open_clip.tokenizer', 'open_clip.transformer',
          'regex._regex', 'torch', 'torch._utils', ...]
```

Bảy class thuộc hai module `model_vssp` và `MHTransformer_dis`. Kiểm tra repo pin:

```
$ find workbench/third_party/PAIR -iname "model_vssp*" -o -iname "MHTransformer_dis*" | wc -l
0
```

Script `scripts/pair_rederive.py` cũng in ra cùng kết luận (`scripts/pair_rederive.out`):

```
model_vssp files in pinned repo: ABSENT
MHTransformer_dis files in pinned repo: ABSENT
```

Hệ quả: checkpoint **chỉ mở được** qua một alias loader thăm dò
(`workbench/artifacts/checkpoint_selection/20261106T002617+0700/pair_compat/`, có
`BUNDLE_MANIFEST.json`, `scripts/`), tức là một đường **không** phải đường production.

Số liệu inventory (`scripts/pair_rederive.out`, đo trên alias loader):

```
ROOT_CLASS: model.PAIR
STATE_DICT_ENTRIES: 455
UNIQUE_TENSOR_OBJECTS_IN_STATE_DICT: 455
NAMED_PARAMETERS: 455 NAMED_BUFFERS: 1 SUM: 456
PREFIX_COUNTS: {'loss_T': 1, 'local_weight': 1, 'local_weight_dis': 1, 'backbone': 446, 'affine': 6} SUM: 455
FRESH_MODEL_CLASS: model_public.PAIR
FRESH_MODEL_KEY_COUNT: 381
MISSING_IN_CKPT: 0 []
UNEXPECTED_IN_CKPT: 74
SHAPE_MISMATCH: 70
PARAMS_TOTAL: 172814185
PARAMS_BYPASSED: 8986448
PARAMS_UNCLASSIFIED: 0
PCT_BYPASSED_PLUS_UNCLASS: 5.2001
```

**455 vs 456 đã reconcile, không phải defect.** `scripts/pair_probe.out`:

```
BUFFER backbone.clip.attn_mask shape (77, 77) in_state_dict False
MOD backbone.clip BUF attn_mask non_persistent True
```

456 = 455 parameter + 1 non-persistent buffer `backbone.clip.attn_mask`; `state_dict()` loại buffer
này, nên con số 455 là đúng và không có tensor thứ hai bị mất.

**Đã thử gì.** (a) decode `GLOBAL` opcode trực tiếp từ pickle; (b) `find`/`glob` tên file trong repo
pin; (c) dựng `model_public.PAIR` từ `third_party/PAIR/models/model.py` và so `state_dict` hai chiều;
(d) đếm `named_parameters` + `named_buffers` để giải thích chênh lệch 1; (e) phân loại parameter theo
prefix để tính phần trăm bypass.

**Vì sao dừng.** Không có class thật để so. `BLOCKER_RESOLUTION.vi.md` §3.4 nói thẳng: "Tôi **không**
tái lập được chính xác 381/74/70 vì repo pin không có class `model_vssp` để dựng 'model keys' — muốn
so 455 vs 381 phải có class thật, mà class đó không tồn tại."

**Điều kiện gỡ.** Tác giả publish module `model_vssp.py` / `MHTransformer_dis.py` ở đúng revision
`bfb0b98cdb4c44f6b9190baaa84d783a77dd888f`, **hoặc** một mapping per-key được xác lập trên class thật
(không phải alias loader). Chỉ khi đó mới được gọi mapping là VERIFIED.

**Mức chắc chắn.**
- Các class thiếu trong repo: **đo được** (decode pickle + `find` trả 0).
- 455 entry / 456 raw / 1 non-persistent buffer: **đo được**.
- 381 key / 74 unexpected / 70 shape mismatch / 5.2001 %: **đo được, nhưng đối chiếu với một class
  giả thuyết** (`model_public.PAIR` dựng từ `third_party/PAIR/models/model.py`), **không** phải class
  của tác giả. Vì vậy con số này **không chứng nhận** điều gì về checkpoint thật — nó chỉ nói rằng
  nếu mapping giả thuyết đúng thì 5.2001 % parameter không đi qua forward. Mapping giữ **UNVERIFIED**.

---

### 3.2 `airknow` — không dựng được model, chết ở chuỗi import `lavis`

**Đang thiếu gì.** Không có native run; chưa từng construct được model. Registry:
`python: '3.8.10'`, `pytorch: '2.1.0'`, `reproduction_status: NOT_RUN`.

**Bằng chứng (đo lại trong lượt này, không chép).**

```
$ workbench/envs/airknow-adapted/bin/python -c "import sys,torch; print(sys.version); print(torch.__version__, torch.cuda.is_available())"
VER 3.11.15 (main, Jun 11 2026, 15:20:16) [GCC 14.3.0]
TORCH 2.7.1+cu128 CUDA True
```

Registry khai Python **3.8.10** / torch **2.1.0**; env thực tế **3.11.15** / **2.7.1+cu128**. Đây là
mismatch contract, không phải sai số ghi chép.

```
$ cd workbench/third_party/AirKnow && .../airknow-adapted/bin/python -c "import lavis.models"
LAVIS FAIL ModuleNotFoundError No module named 'moviepy'
```

Cùng kết quả trong `scripts/probe_airknow2.out`:

```
lavis.models FAIL ModuleNotFoundError("No module named 'moviepy'")
datasets OK
test_BLIP2 OK
```

Nghĩa là `datasets` và `test_BLIP2` import được; chỉ chuỗi `lavis.models` chết. `moviepy` là
transitive dependency mà registry **không** khai (`packages:` của airknow không có `moviepy`;
xem `workbench/registry/models.yaml`, block `model_id: airknow`).

**Đã thử gì.** Import từng tầng: `torch`, `datasets`, `test_BLIP2` OK; `lavis.models` FAIL. Ghi rõ
`lavis` là bản checkout của repo (được đặt lên `sys.path` bởi wrapper), không phải wheel PyPI —
registry có ghi chú điều này trong comment `packages:`.

**Vì sao dừng.** Model class chưa dựng được, nên chưa tới được bước forward; không có gì để chạy
native. Không thể nói "forward CPU chạy được" khi chưa construct được model.

**Điều kiện gỡ.** Cài `moviepy` (và kiểm tra các dependency audio khác của `lavis.processors`) **hoặc**
làm chuỗi import audio-optional; sau đó **xác nhận lại contract Python 3.8.10** — hoặc sửa registry
cho khớp 3.11.15, hoặc dựng env đúng 3.8.10. Cả hai việc đều phải xong trước khi gọi là "gỡ".

**Mức chắc chắn.** Version và lỗi import: **đo được**. Việc `moviepy` là mắt xích duy nhất còn thiếu:
**đo được ở mức import** (`lavis.models` là module đầu tiên fail); chưa chứng minh được rằng sau khi
cài `moviepy` sẽ không có lỗi kế tiếp — đó là **giả thuyết**, và phải nói rõ như vậy.

---

### 3.3 `clvc_net` — không có interpreter, và scoring path cứng CUDA

**Đang thiếu gì.** Không có venv nào cho model này; registry khai `python: '3.7.6'`,
`pytorch: '1.6.0'`, `cuda: '10.2'`. `NATIVE_EVAL_MATRIX.csv` ghi `interpreter` =
`none (no interpreter declared in registry)`.

**Bằng chứng (đo được).**

```
$ ls -d workbench/envs/*clvc*
ls: cannot access 'workbench/envs/*clvc*': No such file or directory
```

Không có env. Đếm `.cuda()` không có guard, toàn tree `workbench/third_party/CLVCNet`:

| file | dòng | số biểu thức |
|---|---|---|
| `test.py` | 28, 47, 99, 115 | 4 |
| `model.py` | 178, 249 | 2 |
| `text_model.py` | 96, 108 (dòng 108 có **hai** `.cuda()`) | 2 dòng / 3 biểu thức |
| `train.py` | 156, 157, 184, 187, 191, 192 | 6 |
| **tổng** | | **14 dòng / 15 biểu thức** |

Trên **scoring path thực sự** (`test.py` + `text_model.py`): `test.py:28,47,99,115` và
`text_model.py:96,108` = **6 dòng / 7 biểu thức**. `model.py:178,249` chỉ dùng trong loss, `train.py`
là train.

Sáu checkpoint **không** phải blocker: `workbench/artifacts/checkpoint_selection/20261106T002617+0700/checkpoints/clvc/`
có đúng `dress_local_model.pt`, `dress_global_model.pt`, `shirt_local_model.pt`,
`shirt_global_model.pt`, `toptee_local_model.pt`, `toptee_global_model.pt`, và registry ghi mapping
`VERIFIED_METADATA` với root class `model.compose_local` / `model.compose_global` cho từng category.

**Đã thử gì.** Đếm `.cuda()` theo dòng/biểu thức trên cả bốn file; liệt kê 6 file checkpoint; đối
chiếu registry block `clvc_net` (`checkpoint_id: author_drive_unresolved`, `filename: "."`,
`required_files` 6 member).

**Vì sao dừng.** Hai blocker độc lập, cả hai đều cứng: không có interpreter để chạy, và kể cả có
interpreter thì `test.py:28` gọi `.cuda()` mà không hỏi `torch.cuda.is_available()`.

**Điều kiện gỡ.** (a) dựng venv py3.7 / torch 1.6 (hoặc một contract adapt `sm_120` với torch mới),
**và** (b) thêm device-guard cho 6 dòng `.cuda()` trên scoring path. Thiếu một trong hai thì vẫn
không chạy được.

**Mức chắc chắn.** Số dòng `.cuda()` và việc không có env: **đo được**. Việc checkpoint map đúng
category/member: **đo được** (root class đọc từ pickle). Việc py3.7/torch1.6 là **bắt buộc** (không
thể dùng torch mới): **suy ra từ source + registry**, chưa thử dựng env mới để chứng minh.

---

### 3.4 `tgcir` — evaluator pin không khởi động được

**Đang thiếu gì.** Ba thứ cùng lúc: entrypoint, interpreter, device path.

**Bằng chứng (đo lại trong lượt này).**

```
$ cd workbench/third_party/TGCIR
$ grep -c "__main__" test.py       -> 0
$ grep -c "argparse" test.py       -> 0
$ grep -c "load_state_dict" test.py -> 0
$ grep -c "torch.load" test.py     -> 0
```

`test.py` là một hàm trần `def test(params, model, testset, category)`. Không có `__main__`, không có
CLI, không có loader.

`.cuda()` không guard:

```
test.py:31   imgs = torch.stack(imgs).float().cuda()
test.py:47   imgs = torch.stack(imgs).float().cuda()
test.py:101  imgs = torch.stack(imgs).float().cuda()
test.py:115  imgs = torch.stack(imgs).float().cuda()

model.py:32    Z = torch.Tensor(B, self.S, C).cuda()
model.py:41    self.clip, _ = clip.load(img_encoder, device='cuda', jit=False)
model.py:107   concept_idx = concept_idx.cuda()
model.py:120   text = clip.tokenize(txt).cuda()
model.py:136   concept_idx = concept_idx.cuda()
model.py:244   y = torch.ones_like(mask).float().cuda()
model.py:251   labels = torch.arange(query.shape[0]).long().cuda()
```

CLIP backbone bị load thẳng `device='cuda'` tại `model.py:41`.

Không có interpreter: registry `python: UNKNOWN`, `NATIVE_EVAL_MATRIX.csv` ghi
`none (no interpreter declared; registry python UNKNOWN)`; `ls -d workbench/envs/*tgcir*` không có
kết quả.

Checkpoint **không** phải blocker: `workbench/artifacts/checkpoint_selection/20261106T002617+0700/downloads/tgcir/x/checkpoints/Fashioniq.pt`
tồn tại (607,928,637 byte), sha `c37768f374c75bebff1299e870518c23bf6db3dc8a75c4d1f729ccd6dd02e966`,
và registry ghi `VERIFIED_METADATA` với strict load 489/489.

**Đã thử gì.** `grep -c` bốn token trong `test.py`; liệt kê `.cuda()` theo file:line; kiểm tra
checkpoint trên đĩa; kiểm tra `workbench/replay/tgcir.py` (3,802 byte — shell **một phần**, không
đầy đủ).

**Vì sao dừng.** Không có `__main__` thì không có gì để gọi; không có interpreter thì không có gì để
chạy; `.cuda()` cứng thì kể cả hai cái trên có cũng chết trên host CPU.

**Điều kiện gỡ.** Viết launch shell đầy đủ (checkpoint loader + argparse) — `workbench/replay/tgcir.py`
là điểm bắt đầu nhưng chưa đủ; **và** provision interpreter; **và** thêm device-guard (hoặc chạy trên
GPU thật).

**Mức chắc chắn.** Bốn `grep -c` bằng 0 và danh sách `.cuda()`: **đo được**. Checkpoint tồn tại và
map đúng: **đo được**. Việc `workbench/replay/tgcir.py` "chưa đủ": **đọc source** — shell chỉ 3.8 KB
so với `clvc_net.py` 9.7 KB / `airknow.py` 12.8 KB, và registry ghi `replay_status: PREFLIGHT_ONLY`.

---

### 3.5 `clip4cir_rn50x4_fullft` — bundle load sạch, chưa chạy eval, upstream chỉ export aggregate

**Đang thiếu gì.** Không có aggregate artifact. `NATIVE_EVAL_MATRIX.csv`: `native_executed = N`,
`last_native_run_id` rỗng.

**Bằng chứng (đo được).**

Bundle load sạch — `raw/blockers/clip4cir_fullft_clip_keydiff.txt`:

```
keys compared 669
n_diff>0 413  n_exact_equal 256  max_abs_diff 0.0007199421525001526
example transformer.resblocks.10.mlp.c_proj.weight
```

`BLOCKER_RESOLUTION.vi.md` §2.2 ghi thêm:

```
COMB epoch 87 keys 12 missing 0 unexpected 0
CLIP epoch 6 keys 669 missing_in_model 0 unexpected_in_ckpt 0 shape_mismatch 0
CLIP value_diff>0 413 / 669   (max_abs_diff 7.20e-4)
```

413/669 tensor khác stock CLIP với `max |Δ| ≈ 7.2e-4` ⇒ đây là fine-tune thật, không phải file stock
đổi tên. (Con số "0 khác" trong một lần đo đầu là SAI do đo sau khi `load_state_dict` đã ghi đè; đã
đo lại.)

Upstream chỉ in aggregate — `workbench/third_party/CLIP4Cir/src/validate.py`, hàm
`compute_fiq_val_metrics` (dòng 21–57):

```python
distances = 1 - predicted_features @ index_features.T
sorted_indices = torch.argsort(distances, dim=-1).cpu()
sorted_index_names = np.array(index_names)[sorted_indices]
...
recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
return recall_at10, recall_at50
```

`main` (dòng ~344–354) chỉ `print` per-category `recallat10`/`recallat50` cộng macro average. Không
có per-query row.

**Per-query export khả thi mà không đổi metric**: `predicted_features`, `index_names`,
`target_names`, `sorted_index_names` đều tồn tại **trước** dòng tính recall (dòng 45–50). Chỉ cần ghi
các mảng đó ra; số học recall không đổi. Đây là instrumentation, không phải sửa evaluator.

**Đã thử gì.** Load cả hai bundle member và đối chiếu key/shape; đo `abs().max()` trên bản stock
**trước** khi nạp; đọc `validate.py` để xác định điểm chèn export.

**Vì sao dừng.** Không có ngân sách cho một lượt full FashionIQ trên CPU. Con số tham chiếu:
`combiner_rn50x4_noft` (cùng kiến trúc, cùng env) mất **~81 phút** CPU — xem §3.13 về nguồn con số.

**Điều kiện gỡ.** Cấp ngân sách cho một lượt eval đầy đủ (CPU ~ giờ, hoặc GPU), rồi ghi aggregate.
Nếu muốn per-query inspection thì instrument ranking path **mà không** đổi metric.

**Mức chắc chắn.** Key/shape coverage và 413/669: **đo được**. Hình dạng output upstream: **đọc
source**. Khả thi per-query export: **suy ra từ source** (các biến tồn tại trước khi tính metric),
chưa instrument thực tế.

---

### 3.6 `dcnet` — blocker đã sửa, nhưng chưa có run hoàn tất

**Đang thiếu gì.** Một lượt chạy tới hết. Không có metric nào từ chiến dịch này.

**Bằng chứng (đo được).**

Patch hiện tại trong `workbench/third_party/DCNet/base/base_trainer.py` (dòng ~190–207):

```python
ckpt_path = self.config.resume / 'trained_model.pth'
...
load_kwargs = {'map_location': self.device}
if 'weights_only' in inspect.signature(torch.load).parameters:
    load_kwargs['weights_only'] = False
checkpoint = torch.load(ckpt_path, **load_kwargs)
```

Đúng cho cả hai phía: torch 1.4.0 không có keyword `weights_only` (nhánh mặc định), torch ≥ 2.6 nhận
`weights_only=False`.

Checkpoint **thật sự** cần opt-out trên torch hiện đại — `BLOCKER_RESOLUTION.vi.md` §1.2:

```
WEIGHTS_ONLY_TRUE_FAILED
UnpicklingError: ... Unsupported global: GLOBAL numpy.core.multiarray.scalar ...
LOADED_FALSE_OK
top keys ['arch','config','epoch','monitor_best','optimizer','state_dict']
config type <class 'parse_config.ConfigParser'>
```

Nó pickle cả `numpy.core.multiarray.scalar` **và** một `ConfigParser`.

`CENet` load strict 518/518 trên torch 2.15 — `raw/blockers/dcnet_cenet_coverage_modern_torch.txt`:

```
model_keys 518 ckpt_keys 518
missing_in_model 0 []   unexpected_in_ckpt 0 []   shape_mismatch 0 []
MATCHED 518
```

Evaluator vượt chỗ chết cũ — `raw/blockers/dcnet_run_cpu_stdout.log` (84 dòng), dòng cuối:

```
Ckpt loaded. Resume training from epoch 83
Running test on cpu
save att
```

**Vì sao dừng.** Run bị **chính người chạy** cắt ở ngân sách 2400 s (`EXIT=124`), không sinh metric.
`raw/blockers/dcnet_run_cpu_stdout_TIMEOUT124.log` là bản log của lượt bị cắt, giống hệt bản kia tới
cùng dòng cuối. `val/info.log` dừng ở `Running test on cpu` (timestamp `2026-11-09 19:20:45`).

**Cảnh báo artifact cũ.** `workbench/artifacts/checkpoint_selection/20261106T002617+0700/checkpoints/dcnet/val/att_result.pt`
có mtime **`2026-11-07 00:02:45`** — tức từ một run trước, **không phải** output của chiến dịch này.
Không được trích nó như kết quả của lượt này. Metric `comb_avg 0.40828` trong `BLOCKER_RESOLUTION.vi.md`
cũng đến từ run cũ đó (mtime 2026-11-06/07), dùng làm sanity, không phải kết quả mới.

**Điều kiện gỡ.** Chạy lại với ngân sách dài hơn (hoặc trên GPU) tới khi evaluator in metric, rồi ghi
metric kèm `run_id`/mtime của chính lượt đó.

**Mức chắc chắn.** Patch đúng cho cả hai torch: **đo được** (regression test
`workbench/tests/test_dcnet_checkpoint_load_portability.py`, 3 pass). Checkpoint cần opt-out:
**đo được**. Nguyên nhân dừng: **đo được** (`EXIT=124`, stderr rỗng, info.log dừng ở một dòng). Việc
"chạy dài hơn sẽ xong": **giả thuyết** — chưa có bằng chứng thời gian chạy đầy đủ là bao lâu.

---

### 3.7 G1/G2 cho 13 model còn lại — `NOT_RUN`, không phải FAIL

**Đang thiếu gì.** Không có differential native-vs-adapter nào cho 13 model ngoài `limn` và `encoder`.
`GATES.csv` hàng `G1 input parity` = `NOT_RUN`, hàng `G2 forward/scoring` = `NOT_RUN`.

**`encoder` đã có, nên bộ đếm G1/G2 chuyển từ 1/15 lên 2/15** — nhưng phải đọc đúng: **G1 của
`encoder` là PASS, G2 là PASS-kèm-2-failure-đã-khai-báo**, không phải PASS sạch. Đây đúng là khuôn mẫu
của `limn` (một gate PASS, một gate PASS có sub-check FAIL được ghi trung thực), nên **không được làm
tròn thành "2/15 PASS"**. Hai failure của `encoder`: (a) **2 rank change không phải tie** (dress 1, gap
6.57e-04; shirt 1, gap 1.60e-04) ở hạng ≫ 50 và không đổi hit count nào — ghi là FAIL theo đúng luật
tie `1e-6` đã khai báo trước; (b) **sub-check `max_rel` của batch 1-vs-N FAIL** (tới 1.165e-02 so với
khai báo `1e-3`; `max_abs` 3.28e-07 vẫn đạt `1e-4`), cùng nguyên nhân mẫu số gần 0 như `limn`.
Nguồn: `ENCODER_PARITY.vi.md` §5.2 và §4.2.

**Bằng chứng — `limn` có đủ 5 tầng, là khuôn mẫu; `encoder` theo cùng khuôn.**

Từ `INPUT_FORWARD_PARITY.vi.md` và `g1_g2.<cat>.{native,adapter}.json`:

| tầng | kết quả `limn` | nguồn |
|---|---|---|
| G1 input parity | **PASS** — gallery order, ref/target ids, model text khớp tuyệt đối 2017/2038/1961 | `INPUT_FORWARD_PARITY.vi.md` §3.1 |
| G2 forward (cùng batch composition) | **PASS bitwise** — `max_abs = 0.0` cả ba category, query và target | §4, `g1_g2.dress.adapter.json` → `layer2_4_forward.sample_forward.query.max_abs = 0.0` |
| G2 scoring | **PASS** — `max_abs ≤ 5.364e-07` (khai báo `1e-6`) | §5.1 |
| G2 ranking | **PASS** — **0** rank difference trên toàn population | §5.2 |
| G2 batch 1 vs 16 | **FAIL tolerance tự khai** — `max_rel` 1.1399e-2 / 2.698e-3 / 7.509e-3 vs khai báo `1e-3` | §4.1 |

Sub-check FAIL là **thật và không được giấu**. Đo được từ `g1_g2.dress.adapter.json`:

```
layer2_4_forward/batch_size_1_vs_n/max_abs = 1.7508864402770996e-07
layer2_4_forward/batch_size_1_vs_n/max_rel = 0.011399774792037631
layer2_4_forward/batch_size_1_vs_n/min_cosine = 0.9999999999991949
layer2_4_forward/batch_size_1_vs_n/within_tolerance = False
```

Nguyên nhân **đo được, không phải giả thuyết**: `max_rel` chia elementwise cho `|reference|`, và một
embedding 768 chiều chuẩn hoá đơn vị thường có thành phần gần 0; một sai số tuyệt đối 1.75e-07 trên
một thành phần cỡ 1.5e-05 đọc ra 1.1e-02 tương đối. Sai số tuyệt đối 3 bậc dưới ngưỡng `1e-4`, cosine
≥ 1 − 1.1e-12, và **ranking giống hệt** giữa batch-1 và batch-16 ở cả ba category.

Tolerance **không bị dịch**. Đây là điểm quan trọng: một sub-check FAIL được khai báo trung thực có
giá trị hơn một PASS bằng cách nới ngưỡng.

**Vì sao 13 model kia `NOT_RUN`.** Chưa từng chạy differential. `NATIVE_EVAL_MATRIX.csv` có 9 hàng
`native_executed = Y`, và trong file đó **mọi hàng trừ `limn`** vẫn ghi `native_verified = N` với lý do
nguyên văn "a completed run is not a verified reproduction". **Hàng `encoder` trong CSV đó chưa được
cập nhật theo lượt này** — file do worker khác sở hữu, nên nó vẫn còn đọc:

> `native_verified=N because G1/G2 were not established for this model (a completed run is not a verified reproduction).`

Đó là trạng thái **của file**, không phải của bằng chứng: `g1_g2.encoder.<cat>.{native,adapter}.json`
và `ENCODER_PARITY.vi.md` đã tồn tại cho cả ba category. Ai đọc CSV phải biết điều này cho tới khi
`NATIVE_EVAL_MATRIX.csv` được chủ sở hữu của nó cập nhật.

**Một native run hoàn tất KHÔNG phải một reproduction đã xác minh.** Đây là câu phải đọc đúng: 9
aggregate tồn tại, và **2 trong 9** (`limn`, `encoder`) giờ có bằng chứng đường adapter tính đúng cùng
hàm với đường native; 7 còn lại thì chưa.

**Điều kiện gỡ.** Với từng model: một differential trên **một mẫu thật cố định**, đi qua các tầng
1→5 (identity → pixels → text → forward → score/rank), với tolerance **khai báo trước khi đo** như
`INPUT_FORWARD_PARITY.vi.md` §0 đã làm cho `limn` và `ENCODER_PARITY.vi.md` §0 đã làm cho `encoder`.

**Mức chắc chắn.** `limn` và `encoder` PASS/FAIL ở trên: **đo được**, có file JSON từng category.
13 model còn lại `NOT_RUN`: **đo được** (không có file `g1_g2.*` nào khác trong thư mục). Không có
model nào trong 13 bị coi là FAIL — `NOT_RUN` nghĩa là chưa có dữ liệu, và điều đó khác với "đã đo và
trượt".

---

### 3.8 LIMN — hai query mơ hồ category, 0/2 giải được canonical

**Đang thiếu gì.** `7:3038` và `7:3790` không được chấm trong metric canonical. Canonical status giữ
`PARTIAL`, `scored_count = 6001`, `unscored_count = 2` (`limn_dual_scope.json` →
`canonical_status`).

**Bằng chứng — category field chưa từng tồn tại trong pipeline release.**

Upstream `TIGER-AI-Lab/UniIR`, `src/data/preprocessing/fashioniq_data_preprocessor.py`, hàm
`fashioniq_to_mbeir_entry` dựng mỗi entry bằng đúng **7 khoá**: `qid`, `query_txt`, `query_img_path`,
`query_modality`, `query_src_content`, `pos_cand_list`, `neg_cand_list`; `query_src_content` chỉ chứa
`{"candidate_img_id": ...}`. Không có `category`, `subtask`, `source_file`.

Rebuild đầy đủ pipeline đó: **6003/6003 dòng khớp tuyệt đối**, 0 mismatch (`LIMN_SCOPE_AUDIT.vi.md`
§1.4). Nghĩa là release không bị cắt xén khi import — upstream cũng không có trường đó.

Cả hai query có **cùng cặp ảnh tham chiếu + caption chuẩn hoá** trong **cả hai** file:

| query | `cap.shirt.val.json` | `cap.toptee.val.json` |
|---|---|---|
| `7:3038` | hàng 1027 | hàng 1595 |
| `7:3790` | hàng 1780 | hàng 1426 |

Caption **byte-identical** giữa hai category (`LIMN_SCOPE_AUDIT.vi.md` §1.2). Vì khoá của
`caption_scope_index` là `(stem ảnh, text chuẩn hoá)`, mỗi cặp ánh xạ vào
`frozenset({"shirt","toptee"})` có **hai** phần tử → `resolve_query_category` trả `None`. Đây là hành
vi đúng (từ chối thay vì đoán).

**Đã thử gì.** 12 diagnostic ranking đã chạy thật (2 query × 2 scope × 3 condition) trên nguyên gallery
74,381, mỗi ô có Top-100 JSONL. Chúng được **dán nhãn diagnostic only** và **không** được cộng vào
canonical. Cross-check quy ước hạng: 36 hạng canonical chấm lại, **36/36 khớp tuyệt đối**.

**Vì sao dừng.** Không có fix-at-source vì field không tồn tại để phục hồi. Chọn một scope bằng tay
là **đoán**, hoặc tệ hơn là **ranking-target leakage**. Cả hai bị cấm. Không có regression test
fail-trước vì không có hành vi sai để sửa.

**Điều kiện gỡ.** Tác giả publish metadata category per-query cho bản release FashionIQ→M-BEIR,
**hoặc** một defect report được release maintainer chấp nhận. Chỉ khi có nguồn ngoài mới được gán
scope.

**Mức chắc chắn.** 7 khoá và rebuild 6003/6003: **đo được**. Caption trùng ở hai category: **đo được**
(hàng index cụ thể). 12 diagnostic ranking: **đo được**. Việc "không có scope nào đúng hơn scope kia"
theo nghĩa khoa học: **kết luận từ dữ liệu** — không scope nào tốt hơn ở cả hai query (`7:3038` #3
shirt / #43 toptee; `7:3790` #65 shirt / #43 toptee ở MM), nên "chọn scope tốt hơn" cũng không cho câu
trả lời nhất quán.

---

### 3.9 Performance — chỉ producer stage được đo, end-to-end CHƯA đo

**Đang thiếu gì.** Không có throughput end-to-end của một lượt evaluation. `GATES.csv` hàng
`Performance` = `NOT_RUN`.

**Đường cong đo được (producer stage: decode + transform).** Nguồn:
`workbench/docs/reviews/20261109-clean-rerun/PERFORMANCE.vi.md` (sinh tự động từ `perf_raw.jsonl` bởi
`gen_performance_doc.py`; không ô nào gõ tay).

| workers | steady trials | median img/s | range (min–max) | cold img/s | peak RSS MiB | parity |
|---:|---:|---:|---|---:|---:|---|
| 0 | 3 | 554.5 | 548.1–554.6 | 548.0 | 231.0 | IDENTICAL |
| 2 | 3 | 1032.8 | 998.1–1041.7 | 988.4 | 309.9 | IDENTICAL |
| 4 | 3 | 1756.2 | 1724.5–1826.1 | 1752.7 | 398.2 | IDENTICAL |
| 6 | 3 | 2282.1 | 2240.8–2288.9 | 2342.5 | 504.3 | IDENTICAL |
| 8 | 3 | 2688.1 | 2580.2–2688.9 | 2784.3 | 602.9 | IDENTICAL |

Đường cong **chưa phẳng** ở setting cao nhất: 8 nhanh hơn 6 **18 %**, nên đây là điểm cao nhất **đã
thử**, không phải điểm tối ưu.

**Phạm vi tuyệt đối phải giữ.** `PERFORMANCE.vi.md` nói nguyên văn: phép đo này "đo **producer stage:
decode + transform** … Nó **không** đo model forward pass, và **không** phải thời gian end-to-end của
một lượt chạy đầy đủ." Tám lượt composition run **không có timing** vì chạy dưới tải đồng thời, và
wall-time ghi `not_measured (concurrent load)`.

Trong chiến dịch này, `PERFORMANCE.csv` chỉ có 4 hàng, và cả 4 là **native test + observer**, không
phải producer-only:

```
native dress: 2017 queries x 2628 gallery, CLIP-L whole model, 231.44 s (native 123.08, observer 108.36)
native shirt: 2038 queries x 3089 gallery, CLIP-L whole model, 275.48 s (native 156.44, observer 119.04)
native toptee: 1961 queries x 2902 gallery, CLIP-L whole model, 241.03 s (native 127.95, observer 113.08)
dual-scope diagnostic: 2 queries x 2 scopes x 3 conditions, 60.47 s
```

**Số producer-only KHÔNG được trình bày như tốc độ end-to-end.** Chúng đo một đoạn của pipeline, trên
một sample 9 ảnh, với BLAS=1; chúng không nói gì về thời gian một lượt eval 6,003 query × 74,381 gallery.

**Điều kiện gỡ.** Đo một lượt eval đầy đủ trên box rảnh, có warmup + lặp, ghi wall-time và tách được
thời gian decode / forward / scoring. Hiện `wait time` được ghi là **unknown** ngay trong harness.

**Mức chắc chắn.** Bảng img/s và RSS: **đo được** (`perf_raw.jsonl`). Việc chưa có end-to-end:
**đo được** (không có file nào ghi wall-time sạch). "workers=8 là tối ưu": **không** — chính tài liệu
nói đường cong chưa phẳng.

---

### 3.10 Test suite — 18 failure tồn tại trước; không có sweep cho thay đổi tài liệu của chiến dịch

**Đang thiếu gì.** Không có full-suite sweep nào chạy cho các thay đổi **tài liệu** của chiến dịch này.
`GATES.csv` hàng `Regression/release` = `NOT_RUN`.

**Bằng chứng.** Bốn log trong `workbench/docs/reviews/20261109-clean-rerun/`:

| log | revision | summary | số `FAILED` |
|---|---|---|---|
| `full_suite_0857744.txt` | `085774488449e774df22e20fe410f30ad008f9d1` | 18 failed, 2558 passed, 47 skipped | 18 |
| `full_suite_2a832bc.txt` | `2a832bc0efe482d1536e466232a3e2803929d1d8` | 21 failed, 2510 passed, 47 skipped | 21 |
| `full_suite_7a58ce6.txt` | `7a58ce6` | 18 failed, 2581 passed, 51 skipped | 18 |
| `full_suite_cd97068.txt` | `cd97068` | 18 failed, 2581 passed, 51 skipped | 18 |

**Id set của 18 failure là byte-identical** giữa `0857744`, `7a58ce6`, `cd97068` (diff rỗng khi so
từng `FAILED <id>`). Sweep `2a832bc` có **21**, trong đó 18 cũ + 3 mới; ba cái mới là test-vs-code
drift sau `80842661` (hai test hard-code `== 9` adapter trong khi registry đã thành 10 sau commit
`20a406a`; một test pin chuỗi `"NGUỒN"` mà `87ff419` đã đổi copy) — đã sửa test-only trong `3d576bf`,
và sweep `0857744` xác nhận chúng không còn fail.

18 failure tồn tại là **pre-existing**, không phải do chiến dịch này. Ví dụ đầu danh sách:

```
workbench/tests/test_circus_adapters_blip.py::test_resident_worker_failure_uses_shared_error_and_stays_failed[workbench.composition.model_conesep-conesep]
workbench/tests/test_circus_cli.py::test_cli_3d_full_gallery_resume_and_gallery_invalidation
workbench/tests/test_handoff_commands.py::test_no_release_document_points_at_a_superseded_source_sha
workbench/tests/test_standalone_operator_commands.py::test_prepare_model_executes_only_declared_recipe_steps
```

**Vì sao dừng.** Sweep mất ~50–60 phút và chiến dịch này chỉ thay đổi tài liệu. `SWEEP.md` ghi rõ
"Not fixed here: this sweep ran on a frozen tree". Không có sweep mới sau khi đóng tài liệu.

**Điều kiện gỡ.** Một sweep serial trên tree sạch, sau khi tài liệu đóng băng, với log có header ghi
revision / command / interpreter / start / end / exit code (đúng format 4 log hiện có).

**Mức chắc chắn.** 18 id và việc chúng identical: **đo được** (đếm `^FAILED ` và so set). Việc không
có sweep cho thay đổi tài liệu: **đo được** (không có log nào sau `cd97068`).

---

### 3.11 UI defect — sticky `.topbar` che `thead` khi cuộn tới đỉnh viewport

**Đang thiếu gì.** Defect **tồn tại, đã biết, cố ý không sửa** trong chiến dịch này.

**Bằng chứng (đo được, viewport 390×844, cuộn bảng vào `block:'start'`)** — `WEB_VERIFICATION.vi.md`
§6 D2:

```
topbar_bottom = 126 px, thead_top = 0 px  ->  che 125 px
elementFromPoint tại đỉnh thead = HEADER.topbar
```

Tại 1920×1080, tiêu đề card ở `top=24` trong khi `topbar_bottom=89` → che 65 px
(`elementFromPoint` trả `H1`).

Tái lập trên **hai bảng khác**: `composition-table` che 103 px, `inventory-table` che 125 px.

**Có từ trước, không do panel LIMN.** `.topbar { position: sticky }` nằm ở `workbench/frontend/style.css`
dòng 158 hiện tại (dòng 109 là `.sidebar`). Không có `scroll-margin-top` nào trong CSS của repo
(`grep -rc "scroll-margin-top" workbench/frontend/` chỉ khớp `node_modules`, không khớp file nguồn).

Về mốc "có từ trước": commit được nêu là `b0f38bd` (2026-11-08). **Lưu ý một điểm tôi xác minh được
và phải nói chính xác:** tại `b0f38bd`, `style.css` **chưa** có rule `.topbar` (chỉ có
`position: sticky` cho cột đầu của `table.composition-table`), và `App.tsx` tại commit đó **chưa**
render `<header className="topbar">`. Rule `.topbar { position: sticky; z-index: 20 }` xuất hiện từ
commit `4e61f2f` (2026-11-08 17:55, "Split the frontend into pages…"). Vì vậy phát biểu đúng là:
defect **predates công việc của chiến dịch này** (mọi thay đổi UI của chiến dịch đều sau `4e61f2f`),
nhưng **không** phải "bắt đầu từ `b0f38bd`" — nó bắt đầu từ `4e61f2f`.

**Đề xuất sửa (chưa làm).** `scroll-margin-top: calc(var(--topbar-h) + 8px)` cho `table`, `.card`,
hoặc anchor `:target` / `[data-testid]`; hoặc hạ `z-index` của `.topbar`. Đây là phạm vi thiết kế
chung, không phải bản vá riêng panel LIMN.

**Mức chắc chắn.** Số px bị che và `elementFromPoint`: **đo được** (ảnh chụp trong `raw/ui/`). Commit
gốc của rule `.topbar`: **đo được bằng `git log -S`** — và kết quả đó **không khớp** mốc `b0f38bd`
được nêu trong brief, xem ghi chú cuối file.

---

### 3.12 `csmcir` — người dùng hoãn

**Đang thiếu gì.** Không nằm trong 15 target active. `NATIVE_EVAL_MATRIX.csv`:
`USER-DEFERRED (not deleted)`.

**Bằng chứng.** `GATES.csv` hàng `Scope/accounting`: "17 declared, 15 active, 2 deferred (csmcir,
sprc)". Lịch sử: một native run **đã hoàn tất** (R@10 57.0507 / R@50 77.2986,
`last_native_run_id = 20261107T054933277898Z_csmcir_fashioniq_f03887d2`), nhưng trạng thái hiện tại là
hoãn theo quyết định người dùng, nên **không** được đếm vào 15.

**Vì sao dừng.** Quyết định người dùng, không phải blocker kỹ thuật. File không bị xoá.

**Điều kiện gỡ.** Người dùng bỏ hoãn.

**Mức chắc chắn.** Trạng thái hoãn và run lịch sử: **đo được** từ registry + matrix.

---

### 3.13 `sprc` — người dùng hoãn, cộng thêm hai blocker kỹ thuật

**Đang thiếu gì.** Ba lớp: (a) user-deferred, (b) `BLOCKED_CHECKPOINT_MAPPING`, (c) **không có
composition adapter**.

**Bằng chứng.** Registry `workbench/registry/models.yaml`, block `model_id: sprc`:
`reproduction_status: BLOCKED_CHECKPOINT_MAPPING`; notes ghi evaluator's argparse default
`--blip-model-name blip2_cir_rerank_learn` "names a class the pinned repo does not register (only an
argparse default; no @registry.register_model); the checkpoint class-key mapping remains unverified and
loads non-strictly"; `checkpoint_mapping_status: UNVERIFIED`.

Đếm adapter: `grep -c "composition_adapter" workbench/registry/models.yaml` → **16**. `sprc` **không**
nằm trong 16 block có `composition_adapter` (danh sách model có adapter: csmcir, airknow, conesep,
ptha_mtst, habit, intent, hint, encoder, pair, clvc_net, dcnet, combiner_rn50x4_noft,
clip4cir_rn50x4_fullft, tgcir, limn, tme). `GATES.csv`: "16/17 have composition_adapter (sprc lacks)".

**Điều kiện gỡ.** Người dùng bỏ hoãn **và** mapping checkpoint được xác minh trên class thật **và**
một composition adapter được viết.

**Mức chắc chắn.** Thiếu adapter: **đo được** (grep). Mapping UNVERIFIED: **đo được** (registry).

---

## 4. Những gì KHÔNG phải blocker

Mục này để người đọc không nhầm một khác biệt đã giải thích thành một lỗ hổng.

| mục | vì sao không phải blocker | nguồn |
|---|---|---|
| `pair` 455 vs 456 key | 456 = 455 parameter + 1 **non-persistent buffer** `backbone.clip.attn_mask`; `state_dict()` loại buffer này nên 455 là đúng. **Không** có tensor thứ hai bị mất. | `scripts/pair_probe.out`, `scripts/pair_rederive.out` |
| `full.json` thiếu `adapter_module_source_hash` | Chỉ là **thiếu sót ghi chép của summary**, không phải bằng chứng rằng hash không đổi. Hai trường này `null` trong `full.json` (đã kiểm: `'adapter_module_source_hash' in d` → `False`) nhưng có giá trị thật trong `runs/<run_id>/ranks/<stem>.json` → `identity.*`. | `COMPOSITION_VERIFICATION.vi.md` §3.3; `workbench/artifacts/composition/summary/encoder_fashioniq_5dee7425dd46c5f154808aac/full.json` |
| `ptha_mtst` có native run hoàn tất | Run **chạy được** nhưng là **TRANSFER**: checkpoint CIRR → FashionIQ, không phải reproduction FashionIQ. `SELECTED_TRANSFER_ONLY`; paper FashionIQ context R@10 54.99 vs observed 37.90 = −17.09 gap. Được đếm trong 9 aggregate nhưng phải đọc kèm nhãn. | `NATIVE_EVAL_MATRIX.csv` hàng `ptha_mtst`; `COMPOSITION_MATRIX.csv` dòng 8 |
| `combiner_rn50x4_noft` có aggregate | Run hoàn tất **trước** chiến dịch (`run_id …020a70b0`, `completion_status: COMPLETED`, `return_code: 0`), được **xác minh lại**, không chạy lại. Vẫn không có G1/G2. | `workbench/artifacts/reports/combiner_…020a70b0_completion.json` |
| `dcnet` patch từng sai | Đã sửa: patch cũ vô điều kiện `weights_only=False` ném `TypeError` trên torch 1.4.0; patch hiện tại capability-check. Đây là **lỗi đã biết và đã đóng**, không phải mục mở. | `BLOCKER_RESOLUTION.vi.md` §1.3–1.4 |
| `limn` G2 PASS nhưng batch sub-check FAIL | Không phải "PASS bị hạ" hay "FAIL bị nâng": cả hai sự thật đều được ghi, tolerance không dịch, nguyên nhân mẫu số đo được. | `INPUT_FORWARD_PARITY.vi.md` §4.1 |
| 12 diagnostic ranking của LIMN | **Diagnostic only**, không phải metric canonical, không được cộng vào nDCG/Recall/coverage. | `limn_dual_scope.json` → `canonical_status.note` |
| `dcnet` `comb_avg 0.40828` | Từ run **cũ** (mtime 2026-11-06/07) trên cùng checkpoint, dùng làm sanity. **Không** phải kết quả chiến dịch này. | `BLOCKER_RESOLUTION.vi.md` §1.6; `att_result.pt` mtime 2026-11-07 00:02:45 |
| **`encoder`: `learnable_relation` không nằm trong checkpoint** — **đã chẩn đoán xong, là phát hiện nghiêm trọng, nhưng KHÔNG blocking** | Defect của **pinned upstream artifact**, không phải của workbench: `model_try2.py:93` gán `nn.Parameter(torch.randn(...)).cuda()`, tức kết quả là `torch.Tensor` thường nên không được đăng ký ⇒ checkpoint có **417** key, không key nào chứa `relation`; `state_dict()` có **415** entry và `'learnable_relation' in state_dict()` = **False**; mỗi process tự rút một relation `[2,512]` riêng. **Một dòng hệ quả:** hai lần chạy native cùng code/cùng pixel đổi hạng **540/2017 · 503/2038 · 375/1961** (max delta 24/28/41) nhưng metric chỉ dịch **−0.05…+0.10**, nên `raw/native/encoder.json` (56.2908 / 77.6334) và CIRCUS run `5dee7425` **không thể replay bit-identical bởi bất kỳ ai, kể cả tác giả**, trừ khi RNG state lúc `Encoder()` dựng tình cờ trùng — số đã công bố vẫn là measurement hợp lệ của một lần rút. **Không đưa vào bảng blocker**: nó không chặn việc gì, chỉ là giới hạn của artifact; seed trước khi dựng model là pin đủ. | `ENCODER_PARITY.vi.md` §4.0.1; `raw/encoder_parity/relation_proof_construct.out`; `raw/encoder_parity/relation_proof_native_pair.txt`; `FINAL_REPORT.vi.md` §6c |

---

## 5. Một attribution đã sửa — ghi lại vì nó từng tồn tại

**Khẳng định SAI (đã bị bác bỏ):** "encoder đổi 12,727/18,009 hạng là do BLAS pinning."

**Nguyên nhân đúng, theo thứ tự nhân quả (đo được):**

1. **Hàm `rank` của adapter đã đổi.** Hai run không tính cùng một hàm:

| trường (trong rank payload `identity`) | old `6a528e87…` | new `5dee7425…` |
|---|---|---|
| `adapter_module_source_hash` | `1a8b0dd25827ca0ba90f504d81f2fa190cede1ee3fb3bfe335e70ab359005823` | `bf60d855be3d72dd2ff0abc6d8b9dc12dbb5fad4a50aa87aaa774454f65a76ab` |
| `adapter_source_hashes.rank` | `5f828b13fef94a3be4e63806cf383d4f8d67a92561f136005bd6cf50315eb978` | `f0914b12834acd0bf7542913384b9b08bd9a742af91816056f1bd73022cf3d4a` |
| `adapter_source_hashes.score` | `0ab4fe1dd028b3a803e79617d6bf24f0d8de410553219a87e98125833847d2cd` | *(giống hệt — không đổi)* |

`score` **không đổi** nhưng `rank` **đổi**: cùng một ma trận điểm, xếp hạng bằng thủ tục khác.

2. **Interpreter đổi.** `runtime.model_interpreter` old = `workbench/envs/pair-adapted/bin/python`
   py **3.11.15**, `verification_status: PYTHON_VERSION_MISMATCH`, `verified: false`; new =
   `workbench/envs/encoder310-adapted/bin/python` py **3.10.20**, `verified: true`.
   `checkpoint_identity` và `gallery_identity` **giống nhau** ở cả hai, nên không phải checkpoint/gallery.

3. **BLAS KHÔNG phải nguyên nhân.** A/B có kiểm soát, cùng vector, chỉ đổi threads 1 vs 10 —
   `workbench/docs/reviews/20261108T-throughput/d3b_encoder_aggregate_scan.txt`:

```
condition                     R@10 d     R@50 d      dNDCG  nchg  maxdR  xK10  xK50  xK100
multimodal                    0.0000    0.00000   2.18e-05     1      1     0     0      0
text_only_black_image         0.0000    0.00000   2.18e-05     1      1     0     0      0
image_only_empty_text         0.0000    0.00000   2.18e-05     1      1     0     0      0
```

`changed: 1`, `r10_delta: 0.0`, `r50_delta: 0.0` mỗi condition.

**Corroboration hình dạng metric:** R@10 **tăng** 31.0178 → 31.0345 trong khi R@50 **giảm**
49.5752 → 49.4753. Hai chiều ngược nhau — một thay đổi số thread không tạo ra mẫu đó.

Quy mô: 12,727/18,009 (`changed = 5214 + 2819 + 4694`, `max|Δrank| = 1706`) so với **3/18,009** của
BLAS-only — chênh ~4,000 lần.

**Trạng thái:** đã sửa trong `COMPOSITION_MATRIX.csv`, `RERUN_DECISIONS.csv`, `MODEL_MATRIX.csv`,
`RUN_LOG.csv` và các tài liệu liên quan. Đây là lỗi **đã đóng**, nhưng được ghi lại vì nó từng tồn tại
và vì bất kỳ ai đọc bản cũ vẫn có thể gặp câu sai.

**Mức chắc chắn.** Ba hash và hai interpreter: **đo được** từ `runs/<id>/ranks/encoder_fashioniq.json`
và `summary/<id>/full.json`. A/B BLAS: **đo được**. Kết luận nhân quả: **suy ra từ ba mệnh đề độc lập
cùng hướng** (rank hash đổi, score hash không đổi, A/B sạch chỉ đổi 1 query), không phải giả thuyết.

---

## 6. Cách tái lập từng khẳng định

Chạy từ `/home/heheboiz/data/cir`. Interpreter kiểm soát: `workbench/envs/ctl312h/bin/python`.
Trên các artifact JSON của chiến dịch này, **selector lồng object trả `null` GIẢ** — đọc từng scalar
một hoặc `json.load` trong Python.

### §3.1 `pair`
```bash
find workbench/third_party/PAIR -iname "model_vssp*" -o -iname "MHTransformer_dis*"   # rỗng
grep -E "^GLOBAL (model_vssp|MHTransformer_dis)" workbench/docs/reviews/20261109-native15-composition-limn/scripts/pair_pickle_globals.out
workbench/envs/ctl312h/bin/python workbench/docs/reviews/20261109-native15-composition-limn/scripts/pair_rederive.py
```

### §3.2 `airknow`
```bash
workbench/envs/airknow-adapted/bin/python -c "import sys,torch; print(sys.version); print(torch.__version__, torch.cuda.is_available())"
cd workbench/third_party/AirKnow && ../../../envs/airknow-adapted/bin/python -c "import lavis.models"
```

### §3.3 `clvc_net`
```bash
ls -d workbench/envs/*clvc*                                        # không tồn tại
grep -n "\.cuda()" workbench/third_party/CLVCNet/{test.py,model.py,text_model.py,train.py}
grep -c "\.cuda()" workbench/third_party/CLVCNet/{test.py,model.py,text_model.py,train.py}
ls workbench/artifacts/checkpoint_selection/20261106T002617+0700/checkpoints/clvc/
```

### §3.4 `tgcir`
```bash
cd workbench/third_party/TGCIR
for p in __main__ argparse load_state_dict "torch.load"; do printf "%s: " "$p"; grep -c "$p" test.py; done
grep -n "\.cuda()" test.py
grep -n "\.cuda()\|device='cuda'" model.py
ls -l workbench/artifacts/checkpoint_selection/20261106T002617+0700/downloads/tgcir/x/checkpoints/Fashioniq.pt
```

### §3.5 `clip4cir_rn50x4_fullft`
```bash
cat workbench/docs/reviews/20261109-native15-composition-limn/raw/blockers/clip4cir_fullft_clip_keydiff.txt
sed -n '21,57p' workbench/third_party/CLIP4Cir/src/validate.py
sed -n '336,360p' workbench/third_party/CLIP4Cir/src/validate.py
```

### §3.6 `dcnet`
```bash
sed -n '190,210p' workbench/third_party/DCNet/base/base_trainer.py
cat workbench/docs/reviews/20261109-native15-composition-limn/raw/blockers/dcnet_cenet_coverage_modern_torch.txt
tail -5 workbench/docs/reviews/20261109-native15-composition-limn/raw/blockers/dcnet_run_cpu_stdout_TIMEOUT124.log
stat -c "%y %n" workbench/artifacts/checkpoint_selection/20261106T002617+0700/checkpoints/dcnet/val/att_result.pt
```

### §3.7 G1/G2
```bash
ls workbench/docs/reviews/20261109-native15-composition-limn/g1_g2.*.json      # chỉ có dress/shirt/toptee của limn
workbench/envs/ctl312h/bin/python - <<'PY'
import json
d=json.load(open('workbench/docs/reviews/20261109-native15-composition-limn/g1_g2.dress.adapter.json'))
b=d['layer2_4_forward']['batch_size_1_vs_n']
print(b['max_abs'], b['max_rel'], b['within_tolerance'])
print(d['tolerances']['batch_max_rel'])
PY
```

### §3.8 LIMN scope
```bash
workbench/envs/ctl312h/bin/python - <<'PY'
import json
d=json.load(open('workbench/docs/reviews/20261109-native15-composition-limn/limn_dual_scope.json'))
print(d['canonical_status'])
PY
grep -n "fashioniq_to_mbeir_entry" -A20 workbench/third_party/UniIR/src/data/preprocessing/fashioniq_data_preprocessor.py
```

### §3.9 Performance
```bash
cat workbench/docs/reviews/20261109-native15-composition-limn/PERFORMANCE.csv
cat workbench/docs/reviews/20261109-clean-rerun/PERFORMANCE.vi.md
wc -l workbench/docs/reviews/20261109-clean-rerun/perf_raw.jsonl
```

### §3.10 Test suite
```bash
for f in workbench/docs/reviews/20261109-clean-rerun/full_suite_*.txt; do echo "### $f"; grep -c "^FAILED " "$f"; done
workbench/envs/ctl312h/bin/python - <<'PY'
import re,glob
s={}
for f in sorted(glob.glob('workbench/docs/reviews/20261109-clean-rerun/full_suite_*.txt')):
    s[f]=set(re.findall(r'^FAILED (\S+)', open(f,errors='replace').read(), re.M))
ks=sorted(s)
for k in ks[1:]: print(k.split('/')[-1], 'new:', s[k]-s[ks[0]], 'missing:', s[ks[0]]-s[k])
PY
```

### §3.11 UI `.topbar`
```bash
grep -n -A3 "^\.topbar" workbench/frontend/style.css
grep -rn "scroll-margin-top" workbench/frontend/*.css      # rỗng
git log --format="%h %ad %s" --date=short -S".topbar {" -- workbench/frontend/style.css
git show b0f38bd:workbench/frontend/style.css | grep -c "position: sticky"
```

### §5 Attribution encoder
```bash
grep -n "changed\|r10_delta\|r50_delta" workbench/docs/reviews/20261108T-throughput/d3b_encoder_aggregate_scan.txt
workbench/envs/ctl312h/bin/python - <<'PY'
import json
for r in ['6a528e877c328772e2c7bb28','5dee7425dd46c5f154808aac']:
    d=json.load(open(f'workbench/artifacts/composition/runs/{r}/ranks/encoder_fashioniq.json'))
    i=d['identity']
    print(r, i['adapter_module_source_hash'][:8], i['adapter_source_hashes']['rank'][:8], i['adapter_source_hashes']['score'][:8])
PY
```

---

## 7. Ghi chú về mức độ chắc chắn — tổng hợp

| loại | mục |
|---|---|
| **Đo được** (lệnh + file + số) | `pair` class thiếu + inventory 455/456; `airknow` version + `moviepy`; `clvc_net` 14/15 + 6/7 `.cuda()` + thiếu env; `tgcir` 4×`grep -c` = 0 + 11 dòng `.cuda()` + thiếu env; `clip4cir` 413/669 + 0/0/0; `dcnet` patch + 518/518 + EXIT=124 + mtime `att_result.pt`; `limn` G1/G2 + sub-check FAIL; LIMN 6003/6003 rebuild + hàng caption; performance 5 mức workers; 18 failure id; `.topbar` px + `git log -S`; encoder 3 hash + interpreter + A/B |
| **Suy ra từ source** (đọc code, chưa chạy) | `clip4cir` per-query export khả thi (biến tồn tại trước dòng 54); `tgcir` shell chưa đủ; `clvc_net` cần py3.7/torch1.6; `limn` không scope nào đúng hơn |
| **Giả thuyết** (phải nói rõ) | cài `moviepy` là đủ cho `airknow` (chưa thử sau khi cài); `dcnet` chạy dài hơn sẽ xong (chưa có ước lượng thời gian); `pair` mapping giả thuyết `model_vssp.Encoder → model.PAIR` — **không** chứng nhận gì |

---

## 8. Những điểm trong brief tôi KHÔNG xác minh được hoặc xác minh ra khác

Ghi thẳng, không làm mềm:

1. **`b0f38bd` là mốc của UI defect `.topbar` — KHÔNG khớp.** Tại `b0f38bd`, `workbench/frontend/style.css`
   không có rule `.topbar`, và `App.tsx` không render `<header className="topbar">`. Rule
   `.topbar { position: sticky; top: 0; z-index: 20 }` xuất hiện từ `4e61f2f` (2026-11-08 17:55).
   Kết luận "defect có từ trước chiến dịch này" vẫn đúng; mốc commit thì phải là `4e61f2f`, không
   phải `b0f38bd`.
2. **"`combiner_rn50x4_noft` mất ~81 phút" — con số này KHÔNG có trong artifact của run.** Hai mốc
   thời gian đọc được (`run_id` timestamp `2026-11-08T06:41:37Z` và `completed_at`
   `2026-11-08T07:20:28.984136Z`) cho **~38.9 phút**. Con số "~81 phút" chỉ xuất hiện trong tài liệu
   viết tay `workbench/docs/reviews/20261108T140435+0700-native-runs/NATIVE_RUNS.vi.md:177`
   (`start → end 2026-11-08T06:41:37Z → ~08:02Z`), và mốc `~08:02Z` đó **không khớp** `completed_at`
   `07:20:28Z`. Tôi **không** xác minh được 81 phút; ước lượng từ artifact là ~39 phút. Vì vậy mục §3.5
   ghi con số kèm cảnh báo, không dùng 81 phút như một sự thật đã đo.
3. **`pair` "5.2001 % parameter bypassed" — đúng số, nhưng không chứng nhận gì.** Đo được trên class
   giả thuyết `model_public.PAIR`; không phải class tác giả. Đã ghi rõ ở §3.1.
4. **`airknow` registry khai Python 3.8.10 nhưng matrix ghi 3.11.15** — cả hai đều đúng theo nguồn
   riêng của chúng; đây là mismatch contract, không phải một trong hai sai.
5. **`raw/native/ptha_mtst.json` không parse được như JSON** (`JSONDecodeError: Expecting value: line 1
   column 1`). Bằng chứng `ptha_mtst` trong tài liệu này lấy từ `NATIVE_EVAL_MATRIX.csv` và
   `raw/native/ptha_mtst.result.json`, không từ file đó.
6. **`encoder.json` và `tme.json`** dùng schema `results`/`aggregate` thay vì `categories[]` như 5 model
   kia — không phải lỗi, chỉ là schema không đồng nhất giữa các artifact native.

---

## 9. Tóm tắt một câu

Chiến dịch đóng được 9/15 native aggregate và **2/15 G1-G2** (`limn`, và `encoder` thêm ở lượt này —
G1 PASS, G2 PASS-kèm-failure-đã-khai-báo); phần còn mở chia thành **6 model thiếu
native run** (mỗi cái có một nguyên nhân đo được riêng: class thiếu, dependency thiếu, env thiếu, CLI
thiếu, ngân sách thiếu, hoặc timeout), **13 model thiếu differential G1/G2**, **2 query không thể
route canonical**, và **ba khoảng trống đo lường** (end-to-end throughput, sweep cho thay đổi tài liệu,
một UI defect tồn tại trước). Không mục nào trong số đó được gọi là "hoàn thành", và không mục nào bị
đóng bằng cách nới ngưỡng hay chọn đáp án theo hướng có lợi.
