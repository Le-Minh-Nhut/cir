# CKPT MAPPING — 8 model native: checkpoint ↔ evaluator, xác lập từ chính bytes

Session: `20261108T114646+0700`. Branch `research/cir-failure-workbench`.
Không chạy GPU, không load model lên GPU, không sửa git state, không commit/push.
Mọi `torch.load` đều `map_location='cpu'`, `weights_only` ghi rõ ở từng phép đo.

## 0. TL;DR

| hạng mục | kết quả |
|---|---|
| variant native trước đây `UNVERIFIED` | **9** (pair×2, clvc, dcnet, combiner, clip4cir, tgcir, sprc×2) |
| `MAPPING_CONFIRMED` | **6** variant — `clvc_net`, `dcnet`, `combiner_rn50x4_noft`, `clip4cir_rn50x4_fullft`, `tgcir`, `sprc_fiq_vitl` |
| `MAPPING_CONTRADICTED` | **2** variant — `pair_b1`, `pair_b2` |
| `MAPPING_UNDETERMINED` | **1** variant — `sprc_fiq_unresolved` (file không tồn tại ở đâu trên đĩa) |
| model chạy được ngay | **0** — mapping không còn là blocker của 6 variant; cái chặn là `sm_120`/torch cũ, env chưa wire, asset provenance, preflight dataset, một adapter (§5) |
| defect đường dẫn root | **thật**, đã khoanh đúng một điểm nghẽn (`checkpoint_path`) và **đã áp dụng**: native 0/11 → **10/11** row resolve được, CIRCUS 11/11 không đổi |
| sha256 đo được | **19** file checkpoint thật (mục §4) |
| stack web còn 200 | **có** — `http://127.0.0.1:8000/api/runs` 200, `/api/benchmark-runs` 200, `http://127.0.0.1:5173/` 200 |

**Điều "CONFIRMED" nghĩa là gì, và không nghĩa là gì.** Nó nghĩa: file này **load strict vào đúng
class mà evaluator pinned dựng lên**, với đúng hình học mà checkpoint tự khai — tức là checkpoint
thuộc về kiến trúc của evaluator này. Nó **không** nghĩa là "tác giả xác nhận đây là checkpoint sinh
ra con số trong paper": hầu hết các file không có checksum upstream. Nhãn đúng là
`LOADS_STRICTLY_INTO_PINNED_CLASS` + provenance đo được.

---

## 1. Defect đường dẫn: `WORKBENCH_CHECKPOINT_ROOT` và cách sửa theo từng model

### 1.1 Cơ chế (đọc từ code, không suy đoán)

`workbench/backend/registry.py:1837-1858`:

```python
def checkpoint_path(model_id, checkpoint, root=None) -> Path:
    base = (root or resolve_config().WORKBENCH_CHECKPOINT_ROOT) / model_id
    if checkpoint.get("bundle_id"):
        return base
    return base / checkpoint["filename"]
```

Đây là **điểm nghẽn duy nhất**: `checkpoint_missing_paths` / `checkpoint_is_present`
(`registry.py:898`, `:1025`), `checkpoint_availability` (`:1968`), `runtime_blockers`
(`runtime.py:3077`), `build_command` (`adapters/models.py:36`), `preflight.py:314` — tất cả đều đi qua
hàm này. Không có consumer nào tự ghép `<root>/<model_id>` (đã grep: chỉ
`evaluate_models.py:2548` cho bundle manifest, và nó cũng lấy đường dẫn từ `checkpoint_path`).

Quy ước cứng `<root>/<model_id>/<filename>` là **một** quy ước, còn staging thực tế có **ba** hình
dạng trong cùng một session:

```
…/20261106T002617+0700/cohort/downloads/<model>/…     ← 9 model CIRCUS (root đang cấu hình)
…/20261106T002617+0700/checkpoints/<dir>/…             ← weights native, tên thư mục KHÔNG trùng model_id
…/20261106T002617+0700/downloads/tgcir/x/checkpoints/  ← member bóc từ archive tác giả
```

Đo được (in-memory, `/tmp/ckm/prove_per_model_root.py`): dưới root đang cấu hình, **5/11** row native
là "present" nếu chỉ đổi root sang `…/checkpoints`; nhưng khi đó **toàn bộ** 9 model CIRCUS thành
missing. Đây là lý do không được sửa root toàn cục: hai cohort nằm ở hai cây khác nhau.

### 1.2 Đề xuất: một field tuỳ chọn trên variant, đọc ở đúng một chỗ

```python
# workbench/backend/registry.py — checkpoint_path()
configured = root or resolve_config().WORKBENCH_CHECKPOINT_ROOT
relative = checkpoint.get("checkpoint_base_relative")   # MỚI
# Không .resolve(): presence check cố tình từ chối symlink (_regular_file/_regular_directory),
# chuẩn hoá đường dẫn trước sẽ vô hiệu hoá chính guard đó.
base = (configured / relative) if relative else configured / model_id
```

`checkpoint_base_relative` là đường dẫn **tương đối so với root đang cấu hình**, nên nó:
- không phải sửa `workbench/config/workbench.env` (bất khả xâm phạm),
- không phải sửa `WORKBENCH_CHECKPOINT_ROOT` toàn cục (không phá CIRCUS),
- thay **toàn bộ** base `<root>/<model_id>`, nên `clvc_net` đọc được `…/checkpoints/clvc` (tên thư
  mục không trùng `model_id`),
- đi cùng `filename` nên `dcnet` đọc `…/checkpoints/dcnet` với `filename: "."` và
  `artifact_type: run_directory`,
- sống sót khi session dir bị đổi tên, khác với một đường dẫn tuyệt đối.

### 1.3 Trạng thái: **ĐÃ ÁP DỤNG**

Quyền sở hữu `workbench/backend/registry.py` được chuyển lại cho session này sau khi agent trước
hoàn tất, nên thay đổi đã được **ghi vào file thật**, không còn là đề xuất.

Đã sửa (`workbench/backend/registry.py`):

1. `checkpoint_path()` đọc `checkpoint_base_relative`; giá trị là đường dẫn **tương đối so với root
   đang cấu hình** và **thay** base `<root>/<model_id>` (không nối thêm).
2. Một guard trong `load_registry()`: `checkpoint_base_relative` phải **relative và khác rỗng**. `..`
   được phép (cần thiết — cây native là sibling của cây download), giá trị tuyệt đối bị từ chối vì nó
   ghim layout của một máy vào registry.

**Một điều chỉnh thiết kế so với bản đề xuất ban đầu, phát hiện bằng test.** Bản đầu *override*
vô điều kiện. Điều đó làm một **root bị redirect bị bỏ qua**: `test_pipeline.py::
test_selected_checkpoint_bytes_mutation_changes_fingerprint_unselected_does_not` dựng checkpoint
dưới `tmp_path` rồi gọi `checkpoint_path` với root đó, và bản override đọc vào cây repo thay vì
`tmp_path` ⇒ test đỏ. Sửa thành **fallback**: vị trí quy ước được thử trước, vị trí khai báo chỉ
được dùng khi vị trí quy ước **không tồn tại**. Deployment thật không đổi hành vi (cây native không
bao giờ có `<root>/<model_id>`, cây CIRCUS không bao giờ có relocation), nhưng root do caller truyền
vào vẫn là authoritative. Toàn bộ 175 test của 10 file liên quan xanh sau sửa này.

Đo lại bằng **chính các hàm thật** sau khi sửa (`.raw/` — `verify_live.py`), không phải mô phỏng:

| model | trước (root CIRCUS) | sau khi áp dụng |
|---|---|---|
| `airknow` fiq_n05 / fiq_n08 | missing / missing | **PRESENT / PRESENT** |
| `pair` pair_b1 / pair_b2 | missing / missing | **PRESENT / PRESENT** |
| `sprc` sprc_fiq_vitl | missing | **PRESENT** |
| `sprc` sprc_fiq (OneDrive) | missing | missing (file không có ở đâu) |
| `clvc_net` | missing | **PRESENT** (6/6 member) |
| `dcnet` | missing | **PRESENT** (`config.json` + `trained_model.pth`) |
| `combiner_rn50x4_noft` | missing | **PRESENT** |
| `clip4cir_rn50x4_fullft` | missing | **PRESENT** (2/2 member) |
| `tgcir` | missing | **PRESENT** |
| native tổng | 0/11 | **10/11** |
| **toàn bộ variant CIRCUS** | 11 present | **11 present — không đổi một row nào** |

Các consumer đều đi qua `checkpoint_path()` và đã được gọi thật để xác nhận chứ không suy luận:

- `checkpoint_is_present` / `checkpoint_missing_paths` → bảng trên.
- `checkpoint_availability` → 10/11 native `checkpoint_downloaded=True`; `mapping_verified=True`
  cho 6 variant confirmed (+2 airknow); `runnable=False` cho **tất cả** (đúng — xem §5).
- `runtime_blockers` → blocker "checkpoint missing" **biến mất** khỏi 10 row native; cái còn lại là
  environment, dataset-link, external-asset, và `mapping unresolved` đúng cho `pair` + `sprc_fiq`.
- `build_command` (`ADAPTERS[...].build_command`) → đi qua `checkpoint_path`; `tgcir`/`clvc_net`/
  `dcnet`/`combiner` nhận đúng đường dẫn mới.
- `preflight.py:314` (`binding_mismatch`) → cùng hàm, cùng kết quả.

### 1.4 Stack web

Trước và sau đều 200, và **composition identity không đổi** (§5.4 có bảng đo):

```
8000/api/runs            200  0.018s
8000/api/benchmark-runs  200  0.008s
8000/api/composition/runs 200  — 15 row, 5 run_id, identical trước/sau
5173/                    200  0.001s
```

Lưu ý đo được, không phải suy đoán: `GET /api/models` là endpoint **đắt** — lần gọi nguội mất
**216 s** (nó hash mọi checkpoint đã staged, 29 row), các lần sau trả cache. `/api/runs` và
`/api/benchmark-runs` không đi qua đường đó nên vẫn nhanh. Nếu ai thấy `/api/models` "treo", đó là
hashing, không phải deadlock.

---

## 2. Từng model: dấu vân tay cấu trúc và phán quyết

Cách đo dùng chung: đọc `data.pkl` trong zip của `torch.save` bằng `pickletools` để liệt kê
**pickle globals** (tên module.class đã tạo ra weights) — không cấp phát tensor, không cần torch,
2.7 GB chỉ tốn vài KB đọc. Sau đó `torch.load(map_location='cpu', weights_only=False)` trên CPU,
dựng lại **đúng class của source pinned** rồi `load_state_dict(strict=True)`.

Script: `.raw/pickle_classes.py`, `.raw/fingerprint2.py`, và các proof riêng trong `.raw/`.

### 2.1 `clvc_net` — **MAPPING_CONFIRMED**

Pickle globals (6/6 file): `model compose_local` / `model compose_global`, `model img_backbone`,
`model local_conv` / `global_conv`, `blocks Inception2d`/`Inception1d`/`GeM`, `resnet ResNet`,
`text_model TextLSTMModel`, `text_model SimpleVocab`, `__builtin__ set`.
Root class: `model.compose_local` (3 file `*_local_model.pt`) và `model.compose_global`
(3 file `*_global_model.pt`). 415 tensor / 54.2 M (local) và 44.1 M (global) tham số.

Bằng chứng quyết định — **strict load vào class của source pinned**, dựng bằng chính vocab lấy từ
checkpoint (`/.raw/clvc_load_proof.json`):

```
dress local   STRICT_OK missing 0 unexpected 0 shapes 0
dress global  STRICT_OK missing 0 unexpected 0 shapes 0
shirt local   STRICT_OK missing 0 unexpected 0 shapes 0
shirt global  STRICT_OK missing 0 unexpected 0 shapes 0
toptee local  STRICT_OK missing 0 unexpected 0 shapes 0
toptee global STRICT_OK missing 0 unexpected 0 shapes 0
```

Bằng chứng **category identity** — đây là phần trước đây bị gọi là "mapping chưa xác lập". Mỗi
category có vocab riêng, và tôi dựng lại vocab từ `data/FashionIQ/captions/cap.<cat>.train.json`
bằng đúng `SimpleVocab.tokenize_text` của pinned source:

| file | vocab size | = số token khác nhau trong `cap.<cat>.train.json` | token của train-caption có trong vocab | vocab có mà train không có |
|---|---|---|---|---|
| `dress_*` | 1602 (1598 + 4 special) | **1598** ✔ | 643/643 | 0 |
| `shirt_*` | 2294 (2290 + 4) | **2290** ✔ | 675/675 | 0 |
| `toptee_*` | 2152 (2148 + 4) | **2148** ✔ | 757/757 | 0 |

Và: `local` + `global` của **cùng** một category chia sẻ **cùng một vocab** (`_same_vocab_dress/shirt/
toptee = true`), ba category có **ba vocab khác nhau** (`_all_six_share_one_vocab = false`). Đó là
chữ ký số học của "mỗi category train một cặp model" — đúng điều `train.py:load_dataset` làm và đúng
`category_checkpoint_structure` mà registry mô tả.

Suy luận: kiến trúc khớp, tên lớp khớp, category khớp, và cặp local/global khớp đúng chiều.
**Kết luận: CONFIRMED.** Điều *không* xác lập được bằng bytes: đây có phải checkpoint sinh ra con số
Table 1 của paper hay không (không có checksum tác giả). Con số đo được trước đó (29.97/58.62) lệch
paper 0.51 pp — phù hợp với "đúng checkpoint", nhưng lệch nhỏ không phải bằng chứng.

### 2.2 `dcnet` — **MAPPING_CONFIRMED**

Root: `dict` với `{arch, epoch, state_dict, monitor_best, config, optimizer}` — đúng hình dạng
`base_trainer.py:158-168` lưu.

| trường | giá trị đọc từ chính file |
|---|---|
| `arch` | `'CENet'` |
| `epoch` | `82` |
| `monitor_best` | `0.4119230891300661` (= `comb_avg` trong `val/info.log`) |
| `config.arch.type` | `CENet`, `backbone: resnet`, `backbone_pretrain: imagenet`, `composition: tirg`, `correction: fd`, `fusion: hadamard` |
| `config.data_loader.args.categories` | `['dress','shirt','toptee']` |
| `config.data_loader.args.data_dir` | `dataset/fashioniq` (author gốc; bản staged đổi thành `/…/data/FashionIQ`) |
| `state_dict` | 518 tensor, 179.9 M tham số |

Dựng `CENet` bằng `parse_config` semantics + **config nhúng trong checkpoint**, rồi load strict
(`.raw/dcnet_load_proof.json`):

```
class model.model.CENet   n_model 518  n_ckpt 518
missing 0   unexpected 0   verdict STRICT_OK
```

Đối chiếu với log evaluator: `checkpoints/dcnet/val/info.log` ghi `[comb_avg] : 0.408284821166705`,
khớp `monitor_best` 0.4119 và khớp paper Avg 40.83 (D6 = 0.88 pp, đã đo trước đó). **CONFIRMED.**
Không xác lập: checksum tác giả (không tồn tại).

### 2.3 `combiner_rn50x4_noft` — **MAPPING_CONFIRMED**

Pickle globals chỉ 4: `collections OrderedDict`, `torch._utils _rebuild_tensor_v2`,
`torch FloatStorage`, `torch LongStorage` — tức đây là **state_dict thuần**, không có class model nào.
Top-level: `{'epoch': 165, 'Combiner': OrderedDict}`.

12 tensor, 59 003 521 tham số, hình học:

```
text_projection_layer.weight  [2560, 640]
image_projection_layer.weight [2560, 640]
combiner_layer.weight         [5120, 5120]
output_layer.weight           [640, 5120]
dynamic_scalar.0.weight       [5120, 5120]
dynamic_scalar.3.weight       [1, 5120]
```

`validate.py:305-307` dựng `Combiner(feature_dim, args.projection_dim, args.hidden_dim)` rồi
`load_state_dict(state_dict["Combiner"])`. Với `--projection-dim 2560 --hidden-dim 5120` và
`feature_dim = clip_model.visual.output_dim = 640` (RN50x4), dựng đúng class pinned và load:

```
Combiner(640, 2560, 5120) strict load OK, epoch 165     ← .raw/clip4cir_load_proof.json
```

Hình học 640/2560/5120 **chỉ** khớp RN50x4 (RN50 = 1024/…, ViT-L = 768/…). **CONFIRMED.**
Lưu ý trung thực: đây là state_dict nên "kiến trúc" được xác lập bằng hình học + strict load, không
bằng tên lớp trong pickle (§2.8 nói rõ giới hạn này).

### 2.4 `clip4cir_rn50x4_fullft` — **MAPPING_CONFIRMED**

Hai member, mỗi member tự khai role:

| member | top-level | nội dung |
|---|---|---|
| `fiq_comb_RN50x4_fullft.pt` | `{'epoch': 87, 'Combiner': …}` | 12 tensor, hình học **giống hệt** noft (2560/5120/640) |
| `fiq_clip_RN50x4_fullft.pt` | `{'epoch': 6, 'CLIP': …}` | 669 tensor, 178.4 M |

Cả hai load **strict** vào đúng thứ `validate.py` dựng lên (`:285` đọc `saved_state_dict["CLIP"]`,
`:306` đọc `state_dict["Combiner"]`):

```
noft   Combiner strict OK epoch 165
fullft Combiner strict OK epoch 87
CLIP   strict OK; input_res 288 output_dim 640 conv1 40 stages [4,6,10,6]
official RN50x4: input_res 288 output_dim 640 conv1 40 stages [4,6,10,6]
key diff vs official: missing 0 extra 0
```

Và nó thực sự là **fine-tuned**, không phải bản stock đổi tên: so 669 tensor với base RN50x4 của
OpenAI, **413 tensor khác giá trị (61.73 %)**. Với "noft" (frozen encoders) thì ngược lại: chỉ có
`Combiner` — đúng như tên gọi và đúng như adapter (`--combiner-path` không kèm `--clip-model-path`).
**CONFIRMED.**

### 2.5 `tgcir` — **MAPPING_CONFIRMED**

Pickle globals: `model VCG_CIR`, `model Backbone`, `model TokenLearner`, `model SpatialAttention`,
`clip.model CLIP` + `LayerNorm`/`QuickGELU`/`ResidualAttentionBlock`/`Transformer`/`VisionTransformer`.
Root là **whole model** `model.VCG_CIR` (489 tensor, 238.1 M), không phải state_dict.

Trường tự khai trong module: `local_token_num=8`, `global_token_num=4`, `local_weight` shape `[12]`,
`t=0.1`, `clip.visual.input_resolution=224`, `output_dim=512`, `conv1` `[768,3,16,16]` → **ViT-B/16**.

Đây đúng là kiến trúc mà `train.py:124` dựng
(`VCG_CIR(img_encoder, hidden_dim=args.feature_dim, local_token_num=args.Q, global_token_num=args.P)`)
với default `--feature_dim 512 --P 4 --Q 8` (`train.py:60,68,69`). Kiểm bằng cách dựng lại và load
(`.raw/tgcir_load_proof.json`):

```
hidden512_Q8_P4 : missing 0  unexpected 0  → STRICT_OK
hidden512_Q4_P8 : missing 0  unexpected 8  → FAIL
hidden1024_Q8_P4: shape mismatch            → FAIL
```

Tức 3/4 hoán vị sai bị loại bằng tensor, chỉ **một** tổ hợp khớp: đúng tổ hợp mà checkpoint tự khai
và đúng default của source pinned. Thêm: archive tác giả chứa `Fashioniq_metrics.json` cùng thư mục
với `Fashioniq.pt`, ghi `dress_r10 45.2157 / shirt 52.6006 / toptee 56.1448` → macro 51.32/73.09 =
đúng hàng paper. Bản đo trước đó được 49.72/71.10 (D6 1.80 pp). **CONFIRMED.**

Một chi tiết vận hành đáng ghi: pickle tên `torch.nn.modules.linear._LinearWithBias`, class đã bị
torch ≥ 2.6 xoá. Alias **chỉ tên lớp** (không tensor nào) là đủ; đó là cùng cách workbench đã ghi
trong report trước.

### 2.6 `sprc` / `sprc_fiq_vitl` — **MAPPING_CONFIRMED**

Top-level `{'epoch': 7, 'Blip2QformerCirAlignPrompt': OrderedDict}` — khoá chính là
`blip_model.__class__.__name__`, đúng cách `blip_validate.py:49` index.

628 tensor, 497.2 M tham số, mix `float32` 279.4 M + **`float16` 217.8 M** (khớp `vit_precision: fp16`
và `freeze_vit: True` của config).

Dấu vân tay backbone: `visual_encoder.conv1.weight [1024,3,14,14]`,
`positional_embedding [257,1024]`, `class_embedding [1024]`, `resblocks.*.attn.in_proj_weight [3072,1024]`
→ **CLIP ViT-L/14**, không phải EVA-CLIP ViT-g. Q-Former: `hidden 768`, `query_tokens [1,32,768]`,
`prompt_tokens [1,32,768]`, `vision_proj [256,768]`, `text_proj [256,768]`, `itm_head [2,768]`.

Load strict vào `blip2_cir_align_prompt` với **cả hai** backbone (`/.raw/sprc_vitl_load_proof.json`):

```
pretrain_vitL : n_model 628  n_ckpt 628  missing 0  unexpected 0  → STRICT_OK
                conv1 [1024,3,14,14]  positional_embedding [257,1024]  qformer_hidden 768
pretrain (ViT-g): missing 511  unexpected 281  → STRICT_FAIL
```

481 tensor lệch giữa hai backbone, và chỉ một backbone khớp tuyệt đối. Vì vậy `backbone:
pretrain_vitL` mà registry khai là **đúng**, và nó được xác lập bằng shape chứ không bằng tên file.
**CONFIRMED** — kèm đúng nhãn cũ: đây là `scientific_variant: true` (ViT-L thay ViT-g của paper),
nên kết quả từ nó **không** được so với hàng paper `reported_r10/r50`.

### 2.7 `pair` / `pair_b1`, `pair_b2` — **MAPPING_CONTRADICTED**

Đây là ca duy nhất mà bytes **phản bác** mapping. Pickle globals (cả B1 và B2, 38 global):

```
model_vssp Encoder / Backbone / FeatureWiseAffine      ← module repo pinned KHÔNG có
MHTransformer_dis Transformer / TransformerLayer / MultiHeadAttention / MlpTransformer  ← cũng KHÔNG có
MHTransformer Transformer / TransformerLayer / MultiHeadAttention / MlpTransformer      ← có
open_clip.model CLIP / open_clip.transformer VisionTransformer / ResidualAttentionBlock ← có
```

Root: `model_vssp.Encoder`, 455 tensor, **172 814 185** tham số. Zip member: B1 =
`model_epoch_1_ShuffleTrue`, B2 = `model_epoch_4_ShuffleFalse`.

Dựng class công khai `PAIR()` rồi load strict (`/.raw/pair_mapping_proof.json`):

| | B1 | B2 |
|---|---|---|
| ckpt state keys | 455 | 455 |
| public `PAIR()` state keys | 381 | 381 |
| missing trong ckpt | 0 | 0 |
| **unexpected key trong model** | **74** | **74** |
| **shape mismatch** | **70** | **70** |
| verdict | **STRICT_FAIL** | **STRICT_FAIL** |

Khác biệt hình học, đo trực tiếp từ module tree đã khôi phục:

| nhánh | checkpoint | `third_party/PAIR/models/model.py` |
|---|---|---|
| `transformer_a` | **2 layer × 4 head** | **1 layer × 8 head** (`model.py:46`) |
| `transformer_b` | 2 × 4 (`MHTransformer.Transformer`) | 2 × 4 (`MHTransformer_rev.Transformer`) ✔ |
| `transformer_c`, `transformer_d` | **có**, 2 × 4 mỗi cái | **không tồn tại** |
| `weight_chanelImg`, `weight_chanelImg_t`, `cross_t2i` | **có** | **không tồn tại** |

`weight_chanelImg` / `weight_chanelImg_t` / `cross_t2i` chỉ tồn tại trong implementation **khác**
(`workbench/third_party/ENCODER/model_try2.py:145,152,158`), không phải trong PAIR. Khối lượng tham
số không được forward của source pinned chạm tới: **8 986 448 / 172 814 185 = 5.20 %**, thuộc 5 module
`transformer_c, transformer_d, weight_chanelImg, weight_chanelImg_t, cross_t2i`.

Suy luận: checkpoint được tạo bởi một bản `model_vssp`/`MHTransformer_dis` chưa công bố, khác hình học
với `third_party/PAIR`. Ngay cả `transformer_a` — nhánh **có được forward** — cũng sai 2×4 so với
1×8, nên không thể nói "chỉ có module chết bị bỏ". Điểm đo được trước đó (B1 35.01/57.58 so paper
52.76/74.51) là hệ quả phù hợp, nhưng không phải bằng chứng nhân quả; **bằng chứng nhân quả ở đây là
shape**.

**Kết luận: CONTRADICTED.** Không được nâng status. Muốn đổi thành CONFIRMED cần một trong hai:
(a) tác giả phát hành `model_vssp.py` + `MHTransformer_dis.py` đúng revision, hoặc (b) một revision
thứ hai của PAIR trong đó `PAIR.__init__` khai `transformer_a` 2×4 và có 5 module nói trên. Không có
suy luận nào từ tên file hay từ B1/B2 hay điểm số có thể thay hai điều kiện đó.

### 2.8 `sprc` / `sprc_fiq_unresolved` — **MAPPING_UNDETERMINED**

`sprc_fiq.pt` (OneDrive, backbone ViT-g) **không tồn tại ở đâu trên máy**. Đã `find /` toàn filesystem
(trừ `/proc`): 0 kết quả. Không có bytes để đọc, nên không có gì để xác lập. **UNDETERMINED**, và
đúng lý do: *file chưa được tải về*. Cái sẽ settle nó: tải file từ OneDrive rồi chạy đúng phép
`strict=False` key-diff mà §2.6 đã chạy cho ViT-L (dự kiến ra `conv1 [1408,3,14,14]`,
`positional_embedding [257,1408]` cho ViT-g). **Không** dùng ViT-L thay thế.

### 2.9 `airknow` (ngoài phạm vi "unverified") — dùng làm đối chứng

Hai variant đã `VERIFIED_METADATA` từ trước; tôi chỉ kiểm chéo cho nhất quán: pickle globals của
`airknow_fiq_0.5.pt` / `0.8.pt` gồm `lavis.models.blip2_models.blip2_cir Blip2QformerCir`,
`lavis.models.eva_vit VisionTransformer`, `lavis.models.blip2_models.Qformer BertLMHeadModel`,
`transformers.models.bert.configuration_bert BertConfig`, `Proxy`. Tức là **LAVIS Blip2QformerCir +
EVA-ViT** — khớp `replay/airknow.py` (import `test_BLIP2`, `data_utils`) và khớp `Proxy` trong source.
Không cần và không được nâng status ở đây: mapping đã verified, và hai file này phủ 100 % bytes
(46 global, không có global lạ).

---

## 3. Ba defect phụ do scoping pass báo — cái nào thật

### 3.1 Filename khai báo sai — THẬT, sửa được bằng dữ kiện

Mỗi dòng dưới đây là **tên file thật** trên đĩa và **cái evaluator thực sự mở**, không phải lựa chọn
chính sách:

| model | old (khai báo) | new (đo được) | evaluator mở cái gì |
|---|---|---|---|
| `clvc_net` ✅ | `filename: checkpoint-unresolved.pt`<br>`artifact_type: file` | `filename: "."`<br>`artifact_type: bundle_directory`<br>`required_files:` 6 file `{dress,shirt,toptee}_{local,global}_model.pt` | `replay/clvc_net.py:EXPECTED_CATEGORY_FILES` đọc **cả 6** trong `--checkpoint-root` |
| `dcnet` ✅ | `filename: fashioniq_dcnet` | `filename: "."`<br>`artifact_type: run_directory`<br>`required_files: [config.json, trained_model.pth]` | `test.py --resume <dir>`; `base_trainer.py:191` mở `<dir>/trained_model.pth` |
| `combiner_rn50x4_noft` ✅ | `rn50x4-noft-unresolved.pt` | `fiq_comb_RN50x4_noft.pt` | `validate.py --combiner-path <file>` → `state_dict["Combiner"]` |
| `clip4cir_rn50x4_fullft` ✅ | `rn50x4-fullft-unresolved.pt`<br>`bundle_placeholder`<br>`required_files: [combiner_state.pt, clip_state.pt]` | `filename: "."`<br>`artifact_type: bundle_directory`<br>`required_files: [fiq_comb_RN50x4_fullft.pt, fiq_clip_RN50x4_fullft.pt]` | adapter đọc `<bundle>/combiner_state.pt` + `<bundle>/clip_state.pt` — **tên nội bộ**, xem ghi chú |
| `tgcir` ✅ | `tgcir-fashioniq-unresolved.pt` | `Fashioniq.pt` @ `downloads/tgcir/x/checkpoints` | `replay/tgcir.py:65` `checkpoint_root / variant["filename"]` |
| `sprc_fiq_unresolved` | `sprc_fiq.pt` | **không đổi** — file không tồn tại | — (đúng: không có bytes) |

Ghi chú `clip4cir`: adapter hard-code tên member `combiner_state.pt` / `clip_state.pt`. Nếu registry
khai đúng tên tác giả, adapter phải đọc member theo **khai báo** chứ không theo literal — nếu không
sẽ lại là "registry đúng, command trỏ sai". Hai cách, chọn một, không trộn: (a) giữ tên literal →
copy/rename 2 file thành `combiner_state.pt` / `clip_state.pt` (tạo bản sao 950 MB, **không** làm vì
đĩa và vì bản sao làm mờ provenance), hoặc (b) sửa `CLIP4CirFullftAdapter.command` đọc
`finalize` `required_files` từ variant. Đề xuất **(b)**, kèm đề xuất `filename: "."` để
`checkpoint_path` trả về đúng thư mục member.

`dcnet` có thêm một dòng dữ kiện: run dir staged đã bị sửa 3 giá trị so với bản tác giả —
`data_dir` (`/home/heheboiz/data/cir/data/FashionIQ`), `num_workers` 6→8, `trainer.save_dir`. Bản
gốc 1982 B (`downloads/dcnet/config.json`, sha `80b33d43…`) vs bản staged 1734 B
(`checkpoints/dcnet/config.json`, sha `fa5fca78…`). Đây là **sửa vận hành đã biết**, không phải
tráo file; nhưng nó có nghĩa `config.json` là artifact *đã biến đổi*, nên `expected_sha256` của nó
(nếu điền) phải là hash của bản staged, không phải bản tác giả.

### 3.2 `expected_sha256` null — THẬT, **đã điền**; bảng old→new ở §4.1

Không có hash nào do tác giả công bố cho 9/11 variant native, nên "null" từng là **trung thực**.
Nhưng giờ có một sự thật đo được: hash của đúng bytes đang staged. Phân biệt hai loại, và ghi rõ
loại nào là loại nào:

- **sha của bytes staged** (đo được, đề xuất điền) — chặn tráo file tại chỗ;
- **sha do tác giả công bố** (không tồn tại) — không thể bịa.

Với `airknow` hai file và `sprc_fiq_vitl`, hash staged **trùng** metadata LFS / size tác giả công bố,
nên điền là hoàn toàn chính đáng.

### 3.3 `resnet50-19c8e357` vs `0676ba61` — THẬT nhưng **không** chặn evaluation; và lỗi thật nằm ở chỗ khác

Ba dữ kiện đo được:

1. torchvision hiện tại: `ResNet50_Weights.IMAGENET1K_V1.url = …/resnet50-0676ba61.pth`;
   `DEFAULT/IMAGENET1K_V2 = …/resnet50-11ad3fa6.pth`. `19c8e357` là entry legacy mà
   `third_party/CLVCNet/resnet.py:20` tự hard-code.
2. `external_asset_statuses` (`registry.py:406-432`): `cache_shaped = bool(asset["cache_path_suffix"]);
   location = None if cache_shaped else (configured if configured else expected_local_location)`.
   Tức với asset **cache-shaped**, `location_env_var` bị **bỏ qua hoàn toàn**. Đây là hành vi có chủ
   đích (comment trong code giải thích đúng lý do: cache root khác file), nhưng hệ quả là
   `WORKBENCH_CLVC_RESNET50_WEIGHTS` / `WORKBENCH_DCNET_RESNET50_WEIGHTS` **khai báo mà không có tác
   dụng** cho đúng hai asset này. Đó là defect thật: *declaration chết*. Sửa: hoặc (a) cho
   `location_env_var` thắng trước khi rơi vào cache resolution (nó là file-level override, đúng ngữ
   nghĩa của tên), hoặc (b) bỏ `location_env_var` khỏi hai asset này để declaration không nói dối.
   Không sửa ở đây: `preparation_contracts.yaml` không thuộc quyền sửa của tôi.
3. **Nhưng cả CLVC-Net và DCNet đều không thực sự cần file ImageNet đó để chấm điểm.** Cả hai lưu
   *whole model*, và mọi tham số backbone đều nằm trong checkpoint:
   - CLVC: `img_model.img_model.*` chứa **318/320** tensor của `resnet50`; 2 cái thiếu đúng là
     `fc.weight`/`fc.bias`, mà `model.py:78` đã thay bằng `nn.Sequential()` nên bị loại khỏi
     `state_dict`.
   - DCNet: `image_encoder.backbone.*` có 318 tensor, child index `0,1,4,5,6,7` =
     `conv1,bn1,layer1..4`; `fc` bị bỏ bởi `list(resnet.children())[:-2]` (`model.py:284`).

   Nghĩa là `pretrained=True` chỉ phục vụ **dựng module** trước khi `load_state_dict` ghi đè 100 %
   trọng số được dùng. Vậy verdict đúng: *metadata lỗi thời + một override chết*, **không** phải
   "thiếu asset chặn khoa học".

**Tác dụng phụ cần khai báo:** phép chứng minh strict-load §2.1 khởi tạo `compose_local/global`, mà
`img_backbone.__init__` gọi `resnet.resnet50(pretrained=True)`, nên **nó đã tải**
`resnet50-19c8e357.pth` (97.8 MB, sha `19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097`)
vào `~/.cache/torch/hub/checkpoints/`. Đó là thay đổi ngoài repo (cache người dùng), không phải sửa
chữa gì của tôi, và nó **không** làm cho declaration đúng: file `0676ba61` vẫn là file mà
torchvision hiện tại cấp, file `19c8e357` chỉ tồn tại vì pinned source tự yêu cầu nó.

---

## 4. sha256 đo được (staged bytes) và registry diff — cái gì đã áp dụng

Đo bằng `sha256sum` trên đĩa, chỉ đọc (`.raw/sha256_all.txt`). `…/` =
`workbench/artifacts/checkpoint_selection/20261106T002617+0700/`.

| variant | file (dưới `…/`) | bytes | sha256 |
|---|---|---|---|
| `airknow/fiq_n05` | `checkpoints/airknow/airknow_fiq_0.5.pt` | 2 724 056 638 | `f4d25ed9e532f2ee3671f6edadb1048e73113246ec056b413bde75dfe3f79bc6` |
| `airknow/fiq_n08` | `checkpoints/airknow/airknow_fiq_0.8.pt` | 2 724 056 638 | `756fc0bdc43d9e0f0cae13c8a02b0279f64e62092ce465c1e60bda45e179c54e` |
| `pair/pair_b1` | `checkpoints/pair/pair-B1.pt` | 694 958 565 | `c4d861ecd0baa3f1427fc7a853943f5614baf87def1cdd9c4bb7583ee925fbf2` |
| `pair/pair_b2` | `checkpoints/pair/pair-B2.pt` | 694 872 371 | `74346fd7de6efdafe8dde70458316286a60ceb2d7b419efed55193418aeb3f03` |
| `clvc_net` | `checkpoints/clvc/dress_local_model.pt` | 217 168 235 | `33d44186efd6a25ca0761794115610016182b9f53ff6413083f31b6979769a07` |
| | `checkpoints/clvc/dress_global_model.pt` | 176 437 419 | `be41c2ba81b803fd392c2c2517bf29cf4dd58205af7777f2becb4f84a80f5c7c` |
| | `checkpoints/clvc/shirt_local_model.pt` | 218 603 243 | `2ba8e320b08ebb97d6558eb599cde2be228b5488250de7cb4cde8a138ff2c681` |
| | `checkpoints/clvc/shirt_global_model.pt` | 177 872 427 | `78b1d660227ab6928e3a11a7384f9fff2aa7ab2a39baec1a45ac59db35e28bcf` |
| | `checkpoints/clvc/toptee_local_model.pt` | 218 308 843 | `d1fa9170496f73a008736723fe5329c93ff710305d0bcc79ff628822c0a91f4c` |
| | `checkpoints/clvc/toptee_global_model.pt` | 177 578 027 | `c78a58c4becafee3bffe719bef743d42c4495c3a91096a4b5935c364449ed11d` |
| `dcnet` | `checkpoints/dcnet/trained_model.pth` | 2 877 310 172 | `130f031e0f7808ccf5f15630a691bd7af95ad9fadbf94f64acfce64e7d8b850f` |
| | `checkpoints/dcnet/config.json` (đã sửa) | 1 734 | `fa5fca78790a33063a4e73b379393f4ffbbf1b251b3c27058aed481136984ba0` |
| `combiner_rn50x4_noft` | `checkpoints/clip4cir/FashionIQ/RN50x4_noft/fiq_comb_RN50x4_noft.pt` | 236 017 921 | `b31882fe6f7e3e194cb9cfda048926844ee0ccc9d4ebc5483fb3bacfa5972955` |
| `clip4cir_rn50x4_fullft` | `…/RN50x4_fullft/fiq_comb_RN50x4_fullft.pt` | 236 017 921 | `7c7c0a10fb4c995cb1ded9ad70a58891d50c622263c54131523bfb232b493f8e` |
| | `…/RN50x4_fullft/fiq_clip_RN50x4_fullft.pt` | 713 874 695 | `4111ac139cf96ac0cf31b30bf8f55b654247b06f0f5fbee4fe8dcc8c1c7cfd2f` |
| `tgcir` | `downloads/tgcir/x/checkpoints/Fashioniq.pt` | 607 928 637 | `c37768f374c75bebff1299e870518c23bf6db3dc8a75c4d1f729ccd6dd02e966` |
| `sprc_fiq_vitl` | `checkpoints/sprc/sprc_fiq_vitl.pt` | 1 459 756 541 | `19386104271c01ea03ee2177575c0e55f98edd3ba61a116d78f2e61ab5dbea08` |

Hash phụ (đối chứng / cần cho dòng `config.json`): `downloads/dcnet/config.json` (bản tác giả,
1982 B) = `80b33d43f6f49c2eeb8ddf948a205f33a78a4c19cdf25975af033ad3be1f320a`;
`checkpoints/dcnet/val/config.json` = `6c4c7849c6882b6ccb8d733bad86fdbcbb4a582d9723bcc602405c806f755b0b`;
`downloads/tgcir/x/checkpoints/Fashioniq_metrics.json` = `bd1dac4dc81f0ac5…`.

### 4.1 Registry: **ĐÃ ÁP DỤNG** những dòng có file thật để trỏ vào

`workbench/registry/models.yaml` đã được sửa. Nguyên tắc chỉ áp dụng: **chỉ những row mà tôi có
artifact staged để trỏ tới**. `sprc_fiq.pt` để nguyên `UNVERIFIED` vì file không tồn tại. `pair`
giữ `UNVERIFIED` vì bytes **phản bác**. Không row `VERIFIED_*` nào tôi không kiểm bị chạm.

Bảng old → new, kèm trạng thái:

| model / variant | field | old | new | áp dụng |
|---|---|---|---|---|
| `airknow/fiq_n05` | `expected_sha256` | `null` | `f4d25ed9…bc6` | ✅ |
| | `expected_size_bytes` | `null` | `2724056638` | ✅ |
| | `checkpoint_base_relative` | — | `../../checkpoints/airknow` | ✅ |
| `airknow/fiq_n08` | `expected_sha256` | `null` | `756fc0bd…54e` | ✅ |
| | `expected_size_bytes` | `null` | `2724056638` | ✅ |
| | `checkpoint_base_relative` | — | `../../checkpoints/airknow` | ✅ |
| `pair/pair_b1` | `expected_sha256` | `null` | `c4d861ec…bf2` (đúng LFS oid tác giả) | ✅ |
| | `checkpoint_base_relative` | — | `../../checkpoints/pair` | ✅ |
| | `checkpoint_mapping_status` | `UNVERIFIED` | **giữ `UNVERIFIED`** | ✅ (không đổi) |
| `pair/pair_b2` | `expected_sha256` | `null` | `74346fd7…f03` | ✅ |
| | `checkpoint_base_relative` | — | `../../checkpoints/pair` | ✅ |
| `clvc_net` | `filename` | `checkpoint-unresolved.pt` | `"."` | ✅ |
| | `artifact_type` | `file` | `bundle_directory` | ✅ |
| | `required_files` | — | 6 file `{dress,shirt,toptee}_{local,global}_model.pt` | ✅ |
| | `expected_sha256` (per member) | — | 6 digest đo được | ✅ |
| | `status` | `CHECKPOINT_MAPPING_UNVERIFIED` | `URL_VERIFIED` | ✅ |
| | `checkpoint_mapping_status` | `UNVERIFIED` | `VERIFIED_METADATA` | ✅ |
| | `category_checkpoint_structure.mapping_status` | `UNVERIFIED` | `VERIFIED_METADATA` | ✅ |
| | `checkpoint_base_relative` | — | `../../checkpoints/clvc` | ✅ |
| `dcnet` | `filename` | `fashioniq_dcnet` | `"."` | ✅ |
| | `bundle_members` | — | 2 member + digest (`trained_model.pth`, `config.json`) | ✅ |
| | `status` | `CHECKPOINT_MAPPING_UNVERIFIED` | `URL_VERIFIED` | ✅ |
| | `checkpoint_mapping_status` | `UNVERIFIED` | `VERIFIED_METADATA` | ✅ |
| | `checkpoint_base_relative` | — | `../../checkpoints/dcnet` | ✅ |
| `combiner_rn50x4_noft` | `filename` | `rn50x4-noft-unresolved.pt` | `fiq_comb_RN50x4_noft.pt` | ✅ |
| | `expected_sha256` / `expected_size_bytes` | `null` | `b31882fe…955` / `236017921` | ✅ |
| | `status` | `CHECKPOINT_MAPPING_UNVERIFIED` | `URL_VERIFIED` | ✅ |
| | `checkpoint_mapping_status` | `UNVERIFIED` | `VERIFIED_METADATA` | ✅ |
| | `checkpoint_base_relative` | — | `../../checkpoints/clip4cir/FashionIQ/RN50x4_noft` | ✅ |
| `clip4cir_rn50x4_fullft` | `filename` | `rn50x4-fullft-unresolved.pt` | `"."` | ✅ |
| | `artifact_type` | `bundle_placeholder` | `bundle_directory` | ✅ |
| | `required_files` | `[combiner_state.pt, clip_state.pt]` | `[fiq_comb_RN50x4_fullft.pt, fiq_clip_RN50x4_fullft.pt]` | ✅ |
| | `bundle_members` | — | 2 member + digest | ✅ |
| | `status` | `CHECKPOINT_MAPPING_UNVERIFIED` | `URL_VERIFIED` | ✅ |
| | `checkpoint_mapping_status` | `UNVERIFIED` | `VERIFIED_METADATA` | ✅ |
| | `checkpoint_base_relative` | — | `../../checkpoints/clip4cir/FashionIQ/RN50x4_fullft` | ✅ |
| `tgcir` | `filename` | `tgcir-fashioniq-unresolved.pt` | `Fashioniq.pt` | ✅ |
| | `expected_sha256` / `expected_size_bytes` | `null` | `c37768f3…966` / `607928637` | ✅ |
| | `status` | `CHECKPOINT_MAPPING_UNVERIFIED` | `URL_VERIFIED` | ✅ |
| | `checkpoint_mapping_status` | `UNVERIFIED` | `VERIFIED_METADATA` | ✅ |
| | `checkpoint_base_relative` | — | `../../downloads/tgcir/x/checkpoints` | ✅ |
| `sprc/sprc_fiq_unresolved` | `checkpoint_mapping_status` | `UNVERIFIED` | **giữ `UNVERIFIED`** | ✅ (không đổi) |
| | `filename` | `sprc_fiq.pt` | **giữ nguyên** — file không có | ✅ (không đổi) |
| | `checkpoint_base_relative` | — | `../../checkpoints/sprc` | ✅ |
| `sprc/sprc_fiq_vitl` | `expected_sha256` | `null` | `19386104…a08` | ✅ |
| | `status` | `CHECKPOINT_MAPPING_UNVERIFIED` | `URL_VERIFIED` | ✅ |
| | `checkpoint_mapping_status` | `UNVERIFIED` | `VERIFIED_METADATA` | ✅ |
| | `scientific_variant` | `true` | **giữ `true`** | ✅ (không đổi) |
| | `checkpoint_base_relative` | — | `../../checkpoints/sprc` | ✅ |

Ba dòng **chưa** áp dụng, và lý do không áp dụng:

1. **`dcnet` `expected_sha256` ở mức variant vẫn `null`.** Đúng như dự đoán: variant là
   `run_directory`, nên `checkpoint_missing_paths` so **từng member** với `bundle_members[].expected_sha256`
   chứ không đọc `expected_sha256` của variant. Tôi đã điền digest vào `bundle_members` (được kiểm
   thật) chứ không điền vào chỗ không ai đọc — điền vào đó sẽ là một con số trông như đang được
   enforce mà thực tế không.
2. **`clvc_net` / `clip4cir` `expected_sha256` ở mức variant vẫn `null`**, cùng lý do: digest nằm ở
   `bundle_members`, là nơi consumer đọc.
3. **Danh sách `required_files` của `sprc`** không đổi: file không tồn tại thì không có gì để khai.

### 4.2 Một prerequisite còn hở, đã ghi ngay trong registry

`clip4cir_rn50x4_fullft`: adapter `adapters/models.py:CLIP4CirFullftAdapter.command` **hard-code**
`combiner_state.pt` / `clip_state.pt` thay vì đọc `required_files`. Với tên tác giả được khai đúng,
adapter đó sẽ truyền hai đường dẫn không tồn tại vào `validate.py`. Tôi **không** sửa adapter (file
thuộc `workbench/backend/**` vừa được trả, nhưng đây là thay đổi hành vi command cần chủ riêng
duyệt), và **không** khai tên bịa của adapter vào registry — làm vậy là tái tạo đúng defect ban đầu.
Thay vào đó tôi ghi prerequisite này thành comment ngay trong record, để row không thể "trông như
chạy được" khi adapter còn lệch. Hiện tại nó bị chặn ở tầng environment trước, nên chưa có rủi ro
chạy sai.

---

## 5. Sáu model `MAPPING_CONFIRMED`: còn thiếu gì để **thực sự chạy**

Trả lời thẳng câu hỏi: **không cái nào chạy được ngay hôm nay**, và điều còn thiếu **không** phải
Python version.

### 5.1 Python version KHÔNG phải blocker — đã kiểm bằng dữ kiện

Prompt nói rõ: khác biệt version Python không được gọi là blocker. Đúng, và tôi đã kiểm cụ thể thay
vì giả định. Tra PyPI cho từng bản torch mà contract đòi:

| model | contract | cpXX có wheel manylinux x86_64 | kết luận |
|---|---|---|---|
| `clvc_net` | py3.7.6 / torch 1.6.0 | torch 1.6.0 có `cp36, cp37, cp38` | **tạo env được** |
| `dcnet` | py3.7.7 / torch 1.4.0 | torch 1.4.0 có `cp27, cp35, cp36, cp37, cp38` | **tạo env được** |
| `tgcir` | py UNKNOWN / torch 1.7.0 | torch 1.7.0 có `cp36, cp37, cp38` | **tạo env được** |
| `airknow` | py3.8.10 / torch 2.1.0 | torch 2.1.0 có `cp38, cp39, cp310, cp311` | **tạo env được** |
| `pair` | py3.8.10 / torch 2.0.0 | torch 2.0.0 có `cp38, cp39, cp310, cp311` | **tạo env được** |
| `sprc` | py3.9 / torch 2.0.1 | torch 2.0.1 có `cp38, cp39, cp310, cp311` | **tạo env được**, và `ilearn39-adapted` đã là py3.9.25 |
| `combiner` / `clip4cir` | py3.8 / torch 1.11.0 | tương tự — cp38 nhánh còn | **tạo env được** |

Vậy: **mọi contract native đều dựng được bằng conda/pip hôm nay.** Python 3.7/3.8 EOL nhưng wheel
vẫn phục vụ; không có gì phải biên dịch. `ilearn39-adapted` (py3.9.25 + torch 2.7.1) thoả **Python**
cho `sprc` và (`hp39` wheel) cho `airknow`/`pair`, nhưng **không** thoả torch pin — nên nó là môi
trường *dùng được để dựng*, không phải để chấm.

### 5.2 Blocker thật: GPU Blackwell `sm_120` vs torch cũ

Đây mới là điều không thể giải bằng "tạo thêm một env Python". `clvc_net`, `tgcir`, `pair` gọi
`.cuda()` **không có guard** trên đường chấm (4 / 4 / 6 lời gọi trong `test.py`, `guard=0`) — nên
chúng **buộc** phải có GPU. Nhưng torch 1.6/1.7/2.0 wheel chỉ ship kernel tới `sm_86`; host là
**compute capability 12.0** (RTX 5070 Ti). Đo trên env đang có:

```
torch 2.7.1+cu128 arch list: ['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120','compute_120']
```

Tức chỉ **torch mới** mới có kernel cho card này. Đây là lý do 9 model CIRCUS đã phải khai
`adapted_runtime` (sm_120 + torch 2.7.1) — và **không** model native nào có block đó. Đó là quyết
định registry của owner, không phải việc session này tự quyết.

Phân loại chính xác từng model:

| model | `.cuda()` không guard trên đường chấm | chạy CPU được? | thiếu gì |
|---|---|---|---|
| `clvc_net` | **4** | **không** | GPU có kernel khớp, tức contract phải là torch ≥ 2.7 (adapted_runtime) |
| `tgcir` | **4** | **không** | như trên |
| `pair` | **6** | **không** | như trên — nhưng **còn** bị `MAPPING_CONTRADICTED`, nên kể cả có GPU cũng không được chạy |
| `combiner_rn50x4_noft` | 0 | **có** (device guarded: `cuda if available else cpu`) | env (py3.8/torch1.11 hoặc adapted) |
| `clip4cir_rn50x4_fullft` | 0 | **có** | env **+** adapter phải đọc member theo `required_files` (§4.2) |
| `dcnet` | 0 trong `test.py`/trainer; `.cuda()` chỉ ở `model/loss.py`, mà `test.py:54` truyền `loss=None` | **có** | env; canonical link `dataset/fashioniq` |
| `sprc/sprc_fiq_vitl` | **0** trên đường FashionIQ (`blip_validate.py`, `validate_blip.py`, `utils.py`, `data_utils.py`) | **có** | env + `LAVIS BLIP-2 pretrain weights` provenance + `COMMAND_BLOCKED_MODEL_MAPPING` |

### 5.3 Tóm lại, cho từng model confirmed

| model | mapping | weight | interpreter | còn thiếu để chạy |
|---|---|---|---|---|
| `clvc_net` | ✅ | ✅ 6/6 | ✗ | env có torch ≥ 2.7 (sm_120) **hoặc** GPU khác; `WORKBENCH_CLVC_PYTHON`; `command_status` đang `COMMAND_BLOCKED_SOURCE_ENTRYPOINT` |
| `dcnet` | ✅ | ✅ 2/2 | ✗ | `WORKBENCH_DCNET_PYTHON` + canonical link `third_party/DCNet/dataset/fashioniq` |
| `combiner_rn50x4_noft` | ✅ | ✅ | ✗ | `WORKBENCH_CLIP4CIR_PYTHON`; preflight `.jpg`/`.png` (hai rule đang mâu thuẫn, dữ liệu **có** đủ) |
| `clip4cir_rn50x4_fullft` | ✅ | ✅ 2/2 | ✗ | như trên **+ adapter đọc member theo registry** |
| `tgcir` | ✅ | ✅ | ✗ | env torch ≥ 2.7 (sm_120); `WORKBENCH_TGCIR_PYTHON`; ViT-B/16 asset provenance; `command_status` `COMMAND_BLOCKED_SOURCE_ENTRYPOINT` |
| `sprc/sprc_fiq_vitl` | ✅ | ✅ | ✗ | `WORKBENCH_SPRC_PYTHON`; BLIP-2 provenance; `COMMAND_BLOCKED_MODEL_MAPPING`; preflight `.png` link |

Không model nào trong bảng này bị chặn bởi **mapping** nữa. Cái chặn là: (a) chưa có interpreter
nào được wire **và** chưa model nào khai `adapted_runtime`, (b) ba asset ngoài chưa admit được
provenance, (c) hai xung đột preflight dataset, (d) một `command_status` chưa audit xong, (e) một
adapter cần sửa. Tất cả đều là việc kỹ thuật đã biết, **không** phải câu hỏi khoa học.

### 5.4 Composition identity không đổi sau khi sửa registry

Đo trước/sau bằng `GET /api/composition/runs`:

```
trước: 15 row, distinct run_id = ['10173dc45dea8c90fd2a284b','315dd590eeb03d2c78d7937a',
                                  '6a528e877c328772e2c7bb28','ad477312aa57c1b9114f9e51',
                                  'e0dd04a46ef7a58334b9e353']
sau  : 15 row, cùng 5 run_id
identical run_id set: True   same rows: True
```

Đây là phép kiểm quan trọng nhất sau khi sửa một file mà cả CIRCUS phụ thuộc: nếu edit làm identity
của composition run trôi thì đó là hỏng nặng hơn defect gốc. Nó không trôi.

---

## 6. Kiểm chứng sau khi áp dụng

| phép kiểm | kết quả |
|---|---|
| `import workbench.backend.registry` + `load_registry()` | OK, 17 model |
| `pytest workbench/tests/test_ckpt_mapping_fingerprints.py` | **11 passed** |
| 10 file test bị ảnh hưởng (replay, doctor, protocols, evaluate_models, encoder_pair_clvc, preflight_consistency, pipeline, ckpt_mapping, verified_declarations, backend_config_roots) | **175 passed**, exit 0 |
| full suite `-n 10` | **10 failed, 2357 passed, 47 skipped** |
| 10 failure đó có phải của tôi? | **Không.** Đo baseline bằng cách `git checkout` toàn bộ 8 file tôi sửa rồi chạy lại đúng 3 file chứa chúng: **cùng 10 failure, cùng tên, same set: True**. Chúng là `test_circus_tme_lifecycle` (env unset), `test_handoff_commands` (source-sha gate), `test_release_blocker_gpu_records` (server/offline), có sẵn từ trước |
| `GET /api/runs`, `/api/benchmark-runs`, `/api/composition/runs`, `:5173/` | **200**, composition 15 row / 5 run_id không đổi |

**Một sự cố tự gây, đã sửa và khai báo:** phép chứng minh strict-load CLVC khởi tạo
`img_backbone`, mà nó gọi `resnet.resnet50(pretrained=True)`, nên **đã tải**
`resnet50-19c8e357.pth` (97.8 MB) vào `~/.cache/torch/hub/checkpoints/`. File đó làm hai test
(`test_dcnet_runtime_asset_blocks_after_preparation_is_present`, ...) đỏ vì chúng giả định asset
**vắng mặt**. Tôi đã **xoá** file vừa tải để trả host về đúng trạng thái trước session. Đây cũng là
bằng chứng phụ cho §3.3: pinned source thực sự đòi `19c8e357`, trong khi torchvision hiện tại cấp
`0676ba61` — cả hai giờ đều có mặt trong cache.

---

## 7. Những gì tôi **cố ý không** thay đổi

1. `workbench/config/workbench.env` — bất khả xâm phạm; và đề xuất §1 không cần nó. `WORKBENCH_CHECKPOINT_ROOT` vẫn trỏ `cohort/downloads`.
2. `workbench/backend/adapters/models.py` — §4.2: adapter `clip4cir` cần sửa, nhưng đó là thay đổi
   hành vi command và cần chủ riêng duyệt. Đã ghi thành prerequisite trong registry.
3. `workbench/registry/preparation_contracts.yaml` — đề xuất ở §3.3 (bỏ hoặc hợp thức hoá
   `location_env_var` cho hai asset ResNet-50), không sửa.
4. `third_party/**` — không patch. Alias `_LinearWithBias → Linear` chỉ tồn tại **trong tiến trình
   đo**, không ghi vào source.
5. `workbench/replay/**`, `workbench/frontend/**`, `workbench/composition/**`,
   `workbench/artifacts/composition/**` — không chạm.
6. Không `checkpoint_mapping_status` nào được nâng cho `pair` (contradicted) hay `sprc_fiq_unresolved`
   (undetermined). Không row `VERIFIED_*` nào tôi không kiểm bị sửa.
7. Không chuyển/rename/copy checkpoint nào (đĩa ~129 G; copy `clip4cir` tốn ~1 GB vô ích và làm mờ
   provenance).
8. Không commit, không push, không `git add`/`reset`/`stash`.

---

## 8. Bảng verdict trước/sau

| | trước | sau | ghi chú |
|---|---|---|---|
| variant có mapping xác lập (`VERIFIED_METADATA`) | 2 (airknow ×2) | **8** | +6: clvc, dcnet, combiner, clip4cir, tgcir, sprc_vitl |
| `MAPPING_CONFIRMED` (nhãn của báo cáo này) | 0 | **6** | cộng 2 airknow đã verified sẵn = 8/11 |
| `MAPPING_CONTRADICTED` | 0 | **2** | pair_b1, pair_b2 |
| `MAPPING_UNDETERMINED` | 7 (`UNVERIFIED`) | **1** | sprc_fiq_unresolved (file không có) |
| `BLOCKED_CHECKPOINT_MISSING` (path) | 11 | **1** | sprc_fiq_unresolved (đã áp dụng §1.3) |
| `RUNNABLE_NOW` | 0 | **0** | environment vẫn chặn cả 8 |

Trạng thái runnable, nói thẳng:

- **Chạy được ngay: 0/11.** Nhưng lý do **không** phải Python version: mọi contract native đều có
  wheel cho đúng cpXX của nó, nên dựng env là việc làm được hôm nay. Blocker thật là `sm_120` +
  `.cuda()` không guard (ba model), env chưa wire, ba external asset chưa admit provenance, hai
  xung đột preflight dataset, một `command_status` chưa audit, và một adapter cần sửa. Chi tiết
  từng model ở **§5**, có bảng "còn thiếu gì".
- **Mapping không còn là blocker** cho `clvc_net, dcnet, combiner_rn50x4_noft, clip4cir_rn50x4_fullft,
  tgcir, sprc_fiq_vitl` (sau khi áp §4). Nghĩa là 6 variant này đã sẵn sàng về bytes + đường dẫn +
  khoa học; còn đúng một tầng chặn.
- Command, khi tầng env được gỡ (dạng chuẩn, chưa chạy vì không có interpreter hợp lệ):

```bash
cd /home/heheboiz/data/cir
workbench/envs/ctl312h/bin/python workbench/scripts/evaluate_models.py \
  --model clvc_net --checkpoint author_drive_unresolved \
  --protocol fashioniq_val_split \
  --dataset-root /home/heheboiz/data/cir/data/FashionIQ --top-k 200
```

`--model` / `--checkpoint` đổi theo model; `checkpoint_id` giữ nguyên như registry (§4 không đổi id
nào, chỉ đổi filename/path/sha, để report cũ vẫn tham chiếu được).

Danh sách còn chặn, một dòng lý do mỗi cái:

| model / variant | lý do một dòng |
|---|---|
| `pair_b1`, `pair_b2` | **bytes phản bác mapping**: pickle cần `model_vssp` + `MHTransformer_dis` (repo pinned không có), `transformer_a` là 2×4 vs 1×8, 74 key lạ + 70 shape lệch, 5.20 % tham số nằm ngoài forward |
| `sprc_fiq_unresolved` | `sprc_fiq.pt` **không tồn tại** ở đâu trên đĩa (link OneDrive chưa tải) → không có bytes để đọc |
| cả 8 model | environment: không interpreter nào khớp contract pinned, và không model native nào khai `adapted_runtime` |
| `combiner`, `clip4cir` (+`sprc`) | preflight `.jpg`/`.png`: `fashioniq_standard` khai `.jpg` trong khi `images/` là `.png`; view `.jpg` chỉ có dưới `prepared/` và link canonical trỏ vào đó → hai rule preflight mâu thuẫn nhau (không phải thiếu dữ liệu) |
| `dcnet` | thiếu canonical link `third_party/DCNet/dataset/fashioniq -> data/FashionIQ` (link là part of contract, chưa staged) |
| `tgcir` | `OpenAI CLIP ViT-B/16 weights/cache` ở trạng thái `BLOCKED_EXTERNAL_ASSET_PROVENANCE` — file có (`ViT-B-16.pt`, sha `5806e77c…`) nhưng không có sha upstream để admit |
| `sprc` (cả 2) | `LAVIS BLIP-2 pretrain weights` provenance unresolved + `COMMAND_BLOCKED_MODEL_MAPPING` |

---


## 9. Evidence index (tất cả trong `.raw/`)

| file | nội dung |
|---|---|
| `pickle_classes.py` | liệt kê pickle globals từ `data.pkl` — không torch, không tensor |
| `fingerprint2.py` | driver: root type, module tree, state_dict prefix/shape/numel, alias + stub có ghi lại |
| `classes_*.json` | 12 file: globals của pair-B1/B2, 2 clvc, dcnet, tgcir, sprc, clip fullft, 2 airknow |
| `fp2_*.json` | 21 file: fingerprint đầy đủ (module tree, scalar, shape) |
| `clvc_load_proof.json` | 6/6 `STRICT_OK` vào `compose_local`/`compose_global` |
| `dcnet_load_proof.json` | 518/518 `STRICT_OK` vào `CENet` từ config nhúng |
| `clip4cir_load_proof.json` | `Combiner` ×2 strict OK; `CLIP` strict OK; key-diff vs RN50x4 official = 0 |
| `tgcir_load_proof.json` | 4 hoán vị (hidden, Q, P); chỉ `hidden512_Q8_P4` STRICT_OK |
| `sprc_vitl_load_proof.json` | `pretrain_vitL` 0/0 STRICT_OK; `pretrain` 511 missing / 281 unexpected |
| `pair_mapping_proof.json` | B1/B2: 74 key lạ, 70 shape lệch, hình học 5 nhánh, 5.20 % tham số chết |
| `resnet50_necessity_proof.json` | 318/320 (CLVC, thiếu chỉ `fc`) và 318/318 dùng được (DCNet) → asset không đổi số nào |
| `clvc_vocab_probe.json` | vocab 3 category + `_same_vocab_*` + `_all_six_share_one_vocab: false` |
| `dcnet_embedded_config.json` | config nhúng: arch, categories, data_dir, trainer |
| `sha256_all.txt` / `sha256_flat.tsv` | 28 file, gồm 19 checkpoint/asset |
| `native_blockers_live.txt` | 62 dòng `blocking_reasons` của 11 row native, đọc từ `GET /api/models` |

Test chạy được để giữ kết luận khỏi trôi: `workbench/tests/test_ckpt_mapping_fingerprints.py`
(**11 passed**, 0.10 s) — assert rằng mỗi pickle names đúng module/class pinned, rằng PAIR pickle
names hai module mà repo không ship, và rằng mọi artifact confirmed còn đúng byte-count. Test chỉ đọc
`data.pkl` (vài KB), không load tensor, nên chạy được trong CI không có GPU.
