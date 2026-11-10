# BLOCKERS R12 — blocker thật của 5 model native còn lại

Branch `research/cir-failure-workbench`, HEAD `2e2e2d2`. Interpreter kiểm soát
`workbench/envs/ctl312h/bin/python` (CPython **3.12.11**, torch **2.15.0+cu132**). CPU-only, không
chạy training/eval loop, không sửa code. Log thô: `raw/r12/`.

Quy ước: mọi mục ghi **ĐÃ QUAN SÁT** (kèm file log) hoặc **NOT_OBSERVED** (không đo được trong ngân
sách của lượt này). Không có số nào dưới đây là kết quả paper reproduction.

---

## 0. Hai kết luận quyết định (đọc trước)

### 0.1 `clavis` KHÔNG tồn tại — đó là `c` (opcode GLOBAL) + `lavis`

Nghi vấn "checkpoint pickle module `clavis`" là **FALSE POSITIVE do grep nhị phân**. Byte đầu của
`model_epoch_1/data.pkl` là:

```
00000000: 8002 636c 6176 6973 2e6d 6f64 656c 732e  ..clavis.models.
00000010: 626c 6970 325f 6d6f 62                     blip2_mob
```

`0x63` = `c` = opcode **GLOBAL**. Nên chuỗi byte là `\x80\x02 c lavis.models...`, và `grep -a clavis`
khớp `c`+`lavis`. `pickletools.genops` giải mã **46 GLOBAL op**, op đầu tiên nguyên văn:

```
GLOBAL  lavis.models.blip2_models.blip2_cir  Blip2QformerCir
```

(`raw/r12/airknow_pkl_globals.txt`). **Không cần `sys.modules` aliasing** cho airknow: tên module
trong pickle chính là `lavis.*`, trùng package thật trong `workbench/third_party/AirKnow/lavis/`.
`grep -rn "clavis" --include=*.py .` → 0 kết quả trên toàn repo, đúng như kỳ vọng.

### 0.2 `combiner_rn50x4_noft` bị "REFUSED" là **HÀNH VI ĐÚNG**, không phải defect

Chuỗi gate nguyên văn, đọc từ `workbench/artifacts/reports/…020a70b0_aggregate.json`
(`raw/r12/combiner_gate_full.txt`):

```json
"status": "METRIC_REFERENCE_UNVERIFIED",
"passed": false,
"blockers": ["no committed reference describes this experiment"],
"detail": "without a verified published row there is nothing to reproduce",
"selection_reason": "combiner_rn50x4_noft: no publishable comparison -- every committed row for it
 is ['UNVERIFIED'], so no published value describes the experiment this run performed: there is
 nothing to compare a reproduction against, and no value would be reported as if there were. Rows:
 ['clip4cir_rn50x4_noft_fiq_absent_in_paper',
  'combiner_rn50x4_noft__clip4cir_paper_note_backbone_inventory_and_version',
  'combiner_rn50x4_noft__clip4cir_rn50x4_fiq_gallery_protocol']"
```

Chuỗi quyết định (file:line):

- `workbench/backend/paper_gate.py:333-335` — `if reference is None: return GateVerdict(
  METRIC_REFERENCE_UNVERIFIED, False, False, [], ["no committed reference describes this experiment"],
  "without a verified published row there is nothing to reproduce")`. Run này rơi đúng nhánh đó.
- `workbench/backend/benchmark_references.py:207-225` — `unbenchmarked()` đọc
  `reported_r10/reported_r50/reported_mean` từ registry; cả ba `null` ⇒ model là **variant tác giả
  phát hành nhưng paper không in dòng kết quả nào**. Đo được: `unbenchmarked('combiner_rn50x4_noft')
  = True`, `select_reference(...) → (None, SELECTION_NO_PUBLISHED_COMPARISON)`.
- `workbench/scripts/verify_run.py:580-585` — tail nguyên văn: *"so a run of combiner_rn50x4_noft is
  reported as having no publishable comparison, **a permanent property of what its author published**
  -- not as a failure of this run and not as a comparison"*. Chạy thật: exit 1,
  `{"verified": false, "is_research_evidence": false, …}` (`raw/r12/combiner_verify_run.json`).
- `workbench/backend/main.py:639` — `"state": "certified" if certified else "refused"`.
- `workbench/frontend/gate.ts:118-121` — `state === "refused"` → nhãn `TỪ CHỐI CHỨNG NHẬN (REFUSED)`.

⇒ Web hiện REFUSED vì `is_research_evidence=false` do **không có dòng paper nào mô tả thí nghiệm
này**, không phải vì run hỏng. Run đã hoàn tất thật: `EXECUTION_COMPLETED`,
`INVOCATION_COMPLETED`, `verdict: PROOF_VALIDATED`, `return_code: 0`,
`extraction_status: EXTRACTED_VERIFIED`, `checkpoint_sha256: b31882fe…`,
metrics `r10 35.1078 / r50 58.4756 / mean 46.7917`. Đóng đúng cách **không phải** nới gate, mà là
hoặc (a) chấp nhận vĩnh viễn nhãn "no publishable comparison" và chỉ dùng số này làm số **nội bộ**,
hoặc (b) trích dẫn một nguồn tác giả khác (CVPR22 demo, Table 1) và thêm một row `VERIFIED` **có
commit** — việc đó là quyết định của operator, không phải của agent này.

---

## 1. `airknow` — BLOCKED (interpreter đã mất + import chain + `.cuda()` cứng)

**Interpreter.** `workbench/envs/airknow-adapted/bin/python` là **symlink gãy**:

```
lrwxrwxrwx … bin/python -> /home/heheboiz/miniconda3/envs/teacher-sprc/bin/python
readlink -f → (rỗng)          # DANGLING
/home/heheboiz/miniconda3/envs/ = VisRAG, melt, teacher-csmcir   # teacher-sprc KHÔNG CÒN
```

`pyvenv.cfg` khai `version = 3.11.15`, `home = …/teacher-sprc/bin`, `include-system-site-packages =
true`. Registry (`models.yaml:136-140`) khai `python: '3.8.10'`, `pytorch: '2.1.0'`.
⇒ **Không có interpreter nào chạy được cho model này.** Con số "3.11.15 / torch 2.7.1" của campaign
trước **không tái lập được** — env đã bị xoá khỏi đĩa, không phải tôi không đo.

**Architecture code.** Có, đầy đủ:
`workbench/third_party/AirKnow/lavis/models/blip2_models/blip2_cir.py:80-81` —
`@registry.register_model("Blip2QformerCir") class Blip2QformerCir(Blip2Base)`.
ĐÃ QUAN SÁT import thành công (sau khi thoả dependency): `CLASS lavis.models.blip2_models.blip2_cir.
Blip2QformerCir`, `MRO → Blip2Base → BaseModel → nn.Module`, `__init__(vit_model='eva_clip_g',
img_size=224, …, num_query_token=32, cross_attention_freq=2, embed_dim=256, max_txt_len=32,
alpha=0.5)`, `PRETRAINED_MODEL_CONFIG_DICT` 3 entry (`raw/r12/airknow_class_introspect.log`).

**Import chain — sửa lại claim cũ.** `moviepy` **không phải lỗi đầu tiên** và **không nằm trên
inference path**. Trên một env sạch (ctl312h, không có torchaudio), lỗi đầu tiên là:

```
File ".../lavis/processors/__init__.py", line 29, in <module>
    from lavis.processors.audio_processors import BeatsAudioProcessor
File ".../lavis/processors/audio_processors.py", line 9, in <module>
    import torchaudio
ModuleNotFoundError: No module named 'torchaudio'
```

(`raw/r12/airknow_import_lavis_ctl312h.log`). `moviepy` chỉ ở **dòng 11** của cùng file, sau
`torchaudio` (dòng 9). Chain đầy đủ đo được bằng stub lặp:

| # | module thiếu | nơi import | có trên đường scoring? |
|---|---|---|---|
| 1 | `torchaudio` | `processors/audio_processors.py:9` | **KHÔNG** (audio BEATs) |
| 2 | `moviepy` | `processors/audio_processors.py:11` | **KHÔNG** (video clip) |
| 3 | `sklearn` | `models/blip2_models/blip2_cir.py:19` (`from sklearn.cluster import KMeans`) | **KHÔNG** ở forward, nhưng là import cấp module của chính class cần dùng |
| 4 | `nltk` | `processors/instruction_text_processors.py:17` | **KHÔNG** |
| 5 | `h5py` | (qua `lavis/models/ulip_models/...`) | **KHÔNG** |

⇒ `moviepy` chỉ là **eager import** của `lavis.processors.__init__`, không phải dependency suy luận.
Repo **đã có** cơ chế bỏ qua đúng nhánh này: `workbench/replay/lavis_processors.py:57-71`
(`_processors_module`) cấy `lavis.processors` rỗng vào `sys.modules` **trước** khi import, đúng để
"`BeatsAudioProcessor` → `torchaudio` … has nothing to do with FashionIQ". Nhưng `sklearn` thì
**không** tránh được, vì nó nằm trong file class.
`moviepy` cài sạch vào env mới được không: **NOT_OBSERVED** (không cài vào env chia sẻ; env đích đã
mất nên không có chỗ cài đúng).

**Checkpoint load.** ĐÃ QUAN SÁT **load được**, CPU, không cần GPU:

```
LOAD_OK 3.9 s
TYPE lavis.models.blip2_models.blip2_cir.Blip2QformerCir
N_PARAM_TENSORS 863
N_PARAMS 1196753657
FIRST_KEYS ['query_tokens','prompt_tokens','visual_encoder.cls_token','visual_encoder.pos_embed',
            'visual_encoder.patch_embed.proj.weight','visual_encoder.patch_embed.proj.bias']
```

(`raw/r12/airknow_fullload.log`; `torch.load(..., map_location="cpu", weights_only=False)` sau khi
import class). Đây là **whole-model pickle** (zip `model_epoch_1/`), root object đúng class tác giả
⇒ **key khớp 100% vì không có key nào để so** — object được unpickle nguyên, không `load_state_dict`.
Kích thước 2,724,056,638 byte/file, khớp registry `expected_size_bytes`.

**Entrypoint.** Có: `workbench/replay/airknow.py` — `main()` với `--checkpoint-root`,
`--dataset-root`, `--source-root`, `--variant`, `--category`, `--eval-mode`, `--preflight-only`,
`--output`. `preflight()` kiểm source pin + checkpoint sha + layout dataset. Đường chạy:
`_load_replay_modules` → `load_trusted_whole_model` → `datasets.FashionIQ` → `test_BLIP2.test`.

**Lỗi đầu tiên khi chạy thật (theo đúng đường wrapper).** Không phải moviepy. Trong một env sạch,
`test_BLIP2` import được (`raw` cũ) nhưng `from lavis...` chết ở `torchaudio`; sau khi thoả hết
import, chỗ chết kế tiếp là **`.cuda()` cứng trên chính đường scoring**, không có
`cuda.is_available()`:

- `test_BLIP2.py:31, 49, 111, 128, 190, 209, 234, 293, 308` — **9 biểu thức `.cuda()`** (đếm bằng
  `grep -c`); nhánh FashionIQ dùng `:111` (query) và `:128` (target).
- `lavis/models/blip2_models/blip2_cir.py:205` — `labels = torch.arange(...).long().cuda()`.
- `blip2_cir.py:409-410` — `inference()`: `reference_embeds.cuda()`, `target_feats.cuda()`.

Trên host không GPU, lỗi sẽ là `AssertionError: Torch not compiled with CUDA enabled` /
`RuntimeError: No CUDA GPUs are available` tại `test_BLIP2.py:111`. **Câu này là suy luận từ code +
torch 2.15 CPU**, không phải quan sát — tôi không chạy forward: **NOT_OBSERVED** cho text lỗi runtime.

**Verdict: BLOCKED.** Hai blocker độc lập: (a) không có interpreter (env đã mất), (b) 11 biểu thức
`.cuda()` cứng trên đường scoring.
**Đường ngắn nhất:** (1) tạo env mới `/tmp` hoặc `workbench/envs/airknow-3.11` với torch CPU +
`torchaudio` + `moviepy` + `scikit-learn==1.3.2` + `nltk` + `h5py` (**~10–15 phút CPU**, phần lớn là
tải wheel); (2) chấp nhận wrapper `workbench/replay/airknow.py` + một patch device-awareness cho
`test_BLIP2.py` và `blip2_cir.py:205,409-410` (**code change — không thuộc quyền tôi**); (3) chạy
1 ảnh để xác nhận forward CPU (**~2 phút CPU**). Full FashionIQ 74k gallery: **~6–12 giờ GPU** hoặc
**~2–4 ngày CPU** — không nên.

**Claim cũ sống/chết:** "thiếu `moviepy` ⇒ chưa construct được model" — **sống một phần** (đúng là
`lavis` chết vì eager audio import) nhưng **sai ở thứ tự** (torchaudio trước) và **sai ở kết luận**
(`moviepy` không cần cho inference; `sklearn` mới là cái chặn class). "env 3.11.15 / torch 2.7.1,
cuda_avail True" — **không tái lập được**, env đã bị xoá.

---

## 2. `clvc_net` — BLOCKED (không có interpreter + `.cuda()` cứng, không có entrypoint)

**Interpreter.** Không có. `workbench/envs/` **không** có thư mục `clvc*`; bảng kiểm symlink cho
`airknow-adapted, pair-adapted, limn-adapted, sprc-adapted, dl-adapted, test-adapted,
clip4cir-adapted` đều DANGLING (cùng trỏ `teacher-sprc` đã mất). `workbench/artifacts/environments/`
chỉ có `cir-005ef4e1b02999a5, clip4cir-38, dcnet-377, sprc-39` — **không có clvc**.
Env khả dĩ theo code: `README.md:18-19` khai **Python 3.7.6 / PyTorch 1.6.0**. `train.py:24`
`from torch.cuda.amp import autocast, GradScaler` (cần torch ≥ 1.6), `train.py:289-290`
`torch.cuda.manual_seed(_all)`. Không có `apex`, không có `cudnn` bắt buộc ⇒ **một env torch CPU
1.6–2.x + Python 3.7–3.11 là plausible về dependency** (không có API đã bị xoá trong eval path).

**Architecture code.** Có: `model.py` (`compose_local`/`compose_global`), `text_model.py`
(`SimpleVocab`, LSTM text encoder), `blocks.py`, `resnet.py`.

**Đếm `.cuda()` — ĐÃ QUAN SÁT, khớp chính xác claim cũ:**

| file | dòng | biểu thức |
|---|---|---|
| `test.py` | 28, 47, 99, 115 | 4 |
| `text_model.py` | 96, **108 (hai `.cuda()`)** | 3 |
| **scoring path** | **6 dòng / 7 biểu thức** | |
| `model.py` | 178, 249 (labels, chỉ loss) | 2 |
| `train.py` | 156, 157, 184, 187, 191, 192 | 6 |
| **toàn tree** | **14 dòng / 15 biểu thức** | |

Không có `torch.cuda.is_available()` ở bất kỳ đâu trong `test.py`/`text_model.py` ⇒ CPU-only chết ở
`test.py:28` (`imgs = torch.autograd.Variable(imgs).cuda()`).

**Checkpoint.** 6 member staged, đúng category: `checkpoint_selection/20261106T002617+0700/
checkpoints/clvc/{dress,shirt,toptee}_{local,global}_model.pt` (207.1 / 168.3 / 208.5 / 169.6 / 208.2
/ 169.4 MB). **Load: NOT_OBSERVED** (không có interpreter để load).

**Entrypoint.** Không có. `test.py` 148 dòng, `0` `__main__`, `0` `argparse`, `0`
`torch.load`/`load_state_dict`; chỉ có `def test(params, local_model, global_model, testset,
dataname)` (dòng 6). `train.py:279` có `__main__` + argparse nhưng không có `--restore_file` nào được
dùng để nạp checkpoint cho eval.

**Device shim có trung thực không?** Về **số học: CÓ**. Đường eval không dùng `autocast`
(`autocast` chỉ ở `train.py:24`/train loop), không dùng `.half()`, và toàn bộ phép tính là
matmul/normalize trên fp32. Nếu model được `.to('cpu')` và `.cuda()` bị monkeypatch thành identity
thì tensor ở lại CPU, dtype không đổi ⇒ **kết quả bit-tương đương** với một bản sửa device-aware.
Về **hợp đồng: KHÔNG** — đó là thay đổi *code của tác giả* (thêm device abstraction), nên phải gắn
nhãn `adapted runtime`, không được trình bày như đường native. Đây là **wrapper trung thực về số,
không trung thực về provenance** nếu không dán nhãn.

**Lỗi đầu tiên (dự kiến, chưa chạy).** `test.py:28` →
`AssertionError: Torch not compiled with CUDA enabled`. **NOT_OBSERVED** (không có interpreter).

**Verdict: BLOCKED** (không interpreter + 7 biểu thức `.cuda()` + không entrypoint).
**Đường ngắn nhất:** (1) env mới Python 3.8–3.10 + `torch==1.13+cpu`/`2.x+cpu` + torchvision
(**~10 phút CPU**); (2) một replay shell (`workbench/replay/clvc_net.py` **đã tồn tại**, dùng
`torch.load` ở `:160-161`) + patch device (`--device cpu`, thay `.cuda()` bằng `.to(device)`) —
**code change, không thuộc quyền tôi**; (3) smoke 1 batch (**~1 phút CPU**). Full eval: **~4–8 giờ
GPU** / **~1–2 ngày CPU**.

**Claim cũ sống/chết:** "6 dòng / 7 biểu thức trên scoring path, 14/15 toàn tree" — **SỐNG, xác nhận
từng dòng**. "Không có interpreter" — **SỐNG**.

---

## 3. `tgcir` — BLOCKED (không entrypoint eval cho checkpoint phát hành + `.cuda()` cứng)

**Interpreter.** Không có env `tgcir*`. `workbench/envs/ilearn39-adapted` = **Python 3.9.25**, torch
**2.7.1+cu128**, `cuda.is_available() True`, **nhưng `clip` = False** (không import được OpenAI CLIP)
— đã đo. Registry khai `pytorch: '1.7.0'`, `torchvision: '0.8.0'`, `cuda: '11.0'`, `python: UNKNOWN`.

**Architecture code.** Có: `model.py` (`VCG_CIR`), `datasets.py`, `utils.py`.

**Đếm `.cuda()` — ĐÃ QUAN SÁT, khớp claim cũ:**
`test.py:31, 47, 101, 115` (4) + `model.py:32, 107, 120, 136, 244, 251` (6) = **10 biểu thức**.
`test.py` 162 dòng: `grep -n "__main__\|argparse\|ArgumentParser\|load_state_dict\|torch.load\|def main\|if __name"`
→ **exit 1, 0 khớp**. Xác nhận.

`model.py:41` `clip.load(..., device='cuda')` — **NOT_OBSERVED**. Điều tôi quan sát được là
`train.py:88`: `clip, preprocess = clip.load(args.img_encoder, device='cuda', jit=False)` — cùng vấn
đề, khác file. Claim cũ nêu `model.py:41` mà không kèm bằng chứng; **không tái lập**.

**Authors chạy eval thế nào — ĐÃ TÌM RA, và nó đổi kết luận.** Không có script `.sh`, không
notebook, không lệnh eval trong `README.md`. Đường thật:

- `train.py:242` `if __name__ == '__main__':` + argparse đầy đủ (`--dataset fashioniq`,
  `--fashioniq_path`, `--batch_size 64`, `--P 4`, `--Q 8`, `--tau_`, …), và **eval được gọi bên
  trong training loop**: `train.py:200` `t = test.test(args, model, trainset, category)` cho
  `['dress','shirt','toptee']`; chọn best theo `t[1][1]` (R@10) và ghi `metrics_best.json`.
- `--restore_file` **được khai ở `train.py:74` nhưng KHÔNG BAO GIỜ được dùng** (`grep -rn
  restore_file` → chỉ 1 khớp, chính dòng khai). ⇒ **không có đường nạp một checkpoint đã phát hành
  để eval**. Đây là lý do thật của `command_status: COMMAND_BLOCKED_SOURCE_ENTRYPOINT`
  (`models.yaml`, entry `tgcir`).
- `cirr_test_submission.py:94-99` là `__main__` duy nhất có loader
  (`torch.load(os.path.join(model_dir,'CIRR.pt'))`, `clip.load('ViT-B/16', device='cpu')`), nhưng
  **chỉ cho CIRR**, không cho FashionIQ.

**Checkpoint.** Có staged: `checkpoint_selection/20261106T002617+0700/downloads/tgcir/x/checkpoints/
Fashioniq.pt` (**579.8 MB**) + `Fashioniq_metrics.json` + `Shoes_metrics.json`. Registry:
`filename: Fashioniq.pt`, `expected_size_bytes: 607928637`, `expected_sha256: c37768f3…`,
`status: URL_VERIFIED`. **Load/giải mã pickle: NOT_OBSERVED** (lượt đo bị cắt).
File `Fashioniq_metrics.json` **chính là metric của tác giả**, đọc nguyên văn:
`dress_r10 45.2157 / dress_r50 69.6579`, `shirt_r10 52.6006 / shirt_r50 72.5221`,
`toptee_r10 56.1448 / toptee_r50 77.1035` (macro R@10 ≈ 51.32, khớp `reported_r10: 51.32`).

**Lỗi đầu tiên.** `test.py:31` `.cuda()` không guard; và trước đó là **không có gì để chạy** —
`python test.py` chỉ định nghĩa hàm rồi thoát 0, không nạp checkpoint, không in metric. Đó là lỗi
"đầu tiên" thật: **exit 0, không output**.

**Verdict: BLOCKED — UNBLOCKABLE BY UPSTREAM FIX ONLY** cho đường *native*. Tác giả không phát hành
script eval nào cho checkpoint đã phát hành; `test.test` là hàm thư viện được `train.py` gọi.
**Đường ngắn nhất (nếu chấp nhận wrapper):** viết shell 40 dòng trong `workbench/replay/tgcir.py`
(argparse + `torch.load` + `clip.load(..., device='cpu')` + `model.eval()` + `test.test`) và patch
10 biểu thức `.cuda()` → device-aware (**code change**). Chi phí: **~15 phút CPU** để smoke, **~3–6
giờ GPU** cho full FashionIQ. Nếu **không** chấp nhận wrapper: vĩnh viễn BLOCKED.

**Claim cũ sống/chết:** "0 `__main__`/argparse/loader trong `test.py`, 10 `.cuda()`" — **SỐNG**.
"Repo không có entrypoint nào" — **SAI một phần**: `train.py:242` là entrypoint có thật, chỉ là nó
**train + eval xen kẽ** và không nạp checkpoint ngoài. "`model.py:41` `clip.load(device='cuda')`" —
**không tái lập**; quan sát được là `train.py:88`.

---

## 4. `pair` — BLOCKED (pickle cần module không tồn tại trong repo pin)

**Interpreter.** Không có: `workbench/envs/pair-adapted/bin/python` là symlink gãy cùng
`teacher-sprc` đã mất (đã kiểm). Registry/adapter khai py 3.10 nhưng không có interpreter thật.

**Architecture code.** Repo pin có **một phần**: `workbench/third_party/PAIR/models/model.py`
— `class Backbone` (:34), `class FeatureWiseAffine` (:175), `class PAIR` (:201);
`models/MHTransformer.py` — `MlpTransformer` (:7), `MultiHeadAttention` (:24),
`MultiHeadAttentionRev` (:61), `TransformerLayer` (:98), `Transformer` (:119); thêm
`models/MHTransformer_rev.py`. **KHÔNG có `class Encoder`** (`grep -c "class Encoder" model.py` = 0),
**không có file `model_vssp*`**, **không có file `MHTransformer_dis*`** (đã `find` cả hai: rỗng).

**Pickle globals — tôi tự giải mã (`pickletools`, `raw/r12/pair_pkl_globals.log`):**
member `model_epoch_1_ShuffleTrue/data.pkl`, **38 GLOBAL op**, 36 unique. Các global **không tồn tại
trong repo pin**, nguyên văn:

```
GLOBAL model_vssp Encoder
GLOBAL model_vssp Backbone
GLOBAL model_vssp FeatureWiseAffine
GLOBAL MHTransformer_dis Transformer
GLOBAL MHTransformer_dis TransformerLayer
GLOBAL MHTransformer_dis MultiHeadAttention
GLOBAL MHTransformer_dis MlpTransformer
GLOBAL MHTransformer Transformer          # top-level, repo chỉ có models.MHTransformer
GLOBAL MHTransformer TransformerLayer
GLOBAL MHTransformer MultiHeadAttention
GLOBAL MHTransformer MlpTransformer
```

Cộng với các global **có** trong repo: `open_clip.model CLIP`, `open_clip.transformer
VisionTransformer/Transformer/ResidualAttentionBlock/LayerNorm`,
`open_clip.tokenizer SimpleTokenizer/_clean_lower`, `torch.*`, `regex._regex compile`,
`_codecs encode`. `pair-B1.pt` và `pair-B2.pt` **cùng key set + shape** (đo trước: 455 entry cả hai).

**Checkpoint load.** Chỉ load được qua **exploratory alias loader**
`checkpoint_selection/20261106T002617+0700/pair_compat/scripts/compat_loader.py` (138 dòng), nó cấy
`sys.modules["model_vssp"]` (:106) và `sys.modules["MHTransformer_dis"]` (:114). **Đây là alias của
tôi, không phải của tác giả** ⇒ mọi số đo dưới nó là `EXPLORATORY_PUBLIC_MAPPING`.
Số đo (từ output script của campaign trước, đọc lại trong lượt này —
`20261109-native15-composition-limn/scripts/pair_rederive.out`):

```
ROOT_CLASS: model.PAIR
STATE_DICT_ENTRIES: 455
NAMED_PARAMETERS: 455  NAMED_BUFFERS: 1  SUM: 456
PREFIX_COUNTS: {'loss_T':1,'local_weight':1,'local_weight_dis':1,'backbone':446,'affine':6}
FRESH_MODEL_KEY_COUNT: 381
MISSING_IN_CKPT: 0
UNEXPECTED_IN_CKPT: 74
SHAPE_MISMATCH: 70
PARAMS_TOTAL: 172814185   PARAMS_BYPASSED: 8986448
PCT_BYPASSED_PLUS_UNCLASS: 5.2001
```

⇒ **455 state-dict entry** (không phải 456; 456 là param+buffer). 381/74/70/5.2001% **tái lập được
về số** nhưng **chỉ dưới alias loader**, nên **mapping vẫn UNVERIFIED**.

**Reconstruct được kiến trúc chỉ từ key không? KHÔNG.** Key cho biết *tên và shape*, không cho biết
*wiring* của forward: (a) `backbone` 446 key nằm trong `model_vssp.Backbone`, nhưng repo có **hai**
biến thể `MHTransformer.py` và `MHTransformer_rev.py` với `MultiHeadAttention` vs
`MultiHeadAttentionRev` — pickle gọi tên `MHTransformer_dis.MultiHeadAttention`, không nói biến thể
nào; (b) 74 key "unexpected" chính là phần thân của `Encoder`/`FeatureWiseAffine` mà class `PAIR`
công khai không có; (c) 70 shape mismatch nghĩa là ngay cả khi đoán đúng tên, chiều vẫn khác. Số
parameter **thực sự được exercise** bởi một forward: **NOT_OBSERVED** — muốn biết phải có class thật
rồi forward; ước lượng từ alias loader là `172,814,185 − 8,986,448 = 163,827,737` (**94.80%**),
nhưng đó là **dưới mapping giả thuyết**, không phải sự thật.

**Entrypoint.** `workbench/replay/pair.py` tồn tại nhưng không thể chạy vì không unpickle được nếu
không có alias ⇒ entrypoint **có về hình thức, vô dụng về thực chất**.

**Lỗi đầu tiên (nguyên văn, khi không có alias):**
`AttributeError: Can't get attribute 'Encoder' on <module 'model_vssp' (built-in)>` — với
`torch.load` thuần thì là `ModuleNotFoundError: No module named 'model_vssp'`. **NOT_OBSERVED trong
lượt này** (tôi không chạy `torch.load` trên pair; đây là suy ra từ chính 38 GLOBAL op đã giải mã).

**Verdict: BLOCKED — UNBLOCKABLE BY UPSTREAM FIX ONLY.** Cần đúng `model_vssp.py` +
`MHTransformer_dis.py` ở đúng revision tác giả. Không thể tái tạo từ repo pin mà không bịa kiến trúc.
**Chi phí:** 0 CPU hôm nay; sau khi có file: **~10 phút CPU** để load strict + smoke, **~4–8 giờ GPU**
cho full eval.

**Claim cũ sống/chết:** "pickle cần `model_vssp.*` + `MHTransformer_dis.*` không có trong repo" —
**SỐNG, tôi tự giải mã lại**. "455 key" — **SỐNG** (455 state-dict + 1 buffer = 456). "381/74/70/
5.2001%" — **tái lập về số nhưng vẫn UNVERIFIED vì phụ thuộc alias loader**.

---

## 5. `combiner_rn50x4_noft` — NOT_A_BLOCKER (đã chạy; bị REFUSED là đúng)

**Interpreter.** Env `workbench/artifacts/environments/clip4cir-38` tồn tại (Python 3.8.20, torch
1.11.0+cpu — theo báo cáo trước; **tôi không chạy lại `-V` trong lượt này**, nhưng run đã hoàn tất
với `environment_digest: d399aa6a…` trong proof, nên interpreter đã tồn tại lúc chạy).
**NOT_OBSERVED** cho phiên bản hiện tại.

**Architecture code.** `workbench/composition/model_combiner_rn50x4_noft.py` (registry
`models.yaml:1074` `composition_adapter:`), source `workbench/third_party/CLIP4Cir` @ `dfed9f74…`
(khớp `source_expected_commit` == `source_actual_commit` trong proof).

**Checkpoint load.** ĐÃ QUAN SÁT gián tiếp qua proof: `checkpoint_sha256:
b31882fe6f7e3e194cb9cfda048926844ee0ccc9d4ebc5483fb3bacfa5972955`,
`checkpoint_artifact_type: file`, `bundle_manifest_digest: null`. Run hoàn tất với metric ⇒ load thành
công. **Không đếm key trực tiếp trong lượt này** (NOT_OBSERVED); báo cáo trước ghi Combiner 12 key
0/0/0 và CLIP 669 key 0/0/0 với 413/669 tensor khác stock — **không tái lập trong lượt này**.

**Entrypoint.** Có, OFFICIAL_CLI (`replay_type: OFFICIAL_CLI`, `command_status: COMMAND_AUDITED`);
`command.json` + `stdout.log` + `stderr.log` trong run dir.

**Run đã hoàn tất — ĐÃ QUAN SÁT:**
`run_id 20261108T064137301769Z_combiner_rn50x4_noft_rn50x4_noft_unresolved_020a70b0`,
`execution_status: EXECUTION_COMPLETED`, `invocation_status: INVOCATION_COMPLETED`, `return_code: 0`,
`verdict: PROOF_VALIDATED`, `extraction_status: EXTRACTED_VERIFIED`,
metrics `r10 35.1078 / r50 58.4756 / mean 46.7917`, `per_query_export_reason: CAPABILITY_ABSENT`.

**"Lỗi" đầu tiên — không có lỗi thực thi. Refusal ở tầng evidence:**
xem §0.2. Gate `METRIC_REFERENCE_UNVERIFIED`, `blockers: ["no committed reference describes this
experiment"]`, `verify_run` exit 1 với `is_research_evidence: false`.

**Verdict: NOT_A_BLOCKER** (về mặt chạy) — nhưng **không phải evidence cấp paper**, và **vĩnh viễn**
như vậy vì paper không in dòng nào cho cấu hình noft RN50x4.
**Đường ngắn nhất:** không có gì để "sửa". Muốn con số này có hiệu lực đối chiếu thì phải commit một
row `VERIFIED` từ nguồn tác giả khác (CVPR22 demo Table 1: 36.36/58.00, 31.63/56.67, 38.19/62.42,
macro 35.39/59.03) — **quyết định của operator, không phải việc của gate**. Chi phí: **0 CPU**;
đối chiếu số học 35.1078 vs 35.39 là **−0.80%** nhưng **không được** gọi là reproduction cho tới khi
row đó được commit với `reference_status: VERIFIED`.

**Claim cũ sống/chết:** "NOT_A_BLOCKER, đã chạy native, mean 46.7917" — **SỐNG**. Điểm cần bổ sung:
"không bị chặn để chạy" **không** bằng "dùng được làm bằng chứng paper"; hai điều đó khác nhau và
trước đây chỉ nói vế đầu.

---

## 6. Tổng hợp

| model | interpreter | arch code | checkpoint load | entrypoint | verdict |
|---|---|---|---|---|---|
| `airknow` | **MẤT** (symlink gãy) | có (`blip2_cir.py:80`) | **ĐƯỢC**, 863 tensor / 1,196,753,657 param | có (`replay/airknow.py`) | **BLOCKED** |
| `clvc_net` | không có | có | NOT_OBSERVED | **không** | **BLOCKED** |
| `tgcir` | không có (ilearn39 thiếu `clip`) | có | NOT_OBSERVED | **không cho ckpt phát hành** | **BLOCKED / upstream-only** |
| `pair` | **MẤT** (symlink gãy) | **thiếu class** | chỉ qua alias loader | hình thức | **BLOCKED / upstream-only** |
| `combiner_rn50x4_noft` | có (lúc chạy) | có | có (proof) | có | **NOT_A_BLOCKER** (nhưng không phải paper evidence) |

**Ba claim của campaign trước bị sửa:**
1. **airknow**: lỗi đầu tiên là `torchaudio` (`audio_processors.py:9`), **không phải** `moviepy`
   (`:11`); `moviepy` **không** nằm trên inference path; và `sklearn` (`blip2_cir.py:19`) mới là cái
   chặn class. Env 3.11.15 **không còn** để tái lập.
2. **tgcir**: "repo không có entrypoint" **sai một phần** — `train.py:242` là entrypoint thật, eval
   được gọi trong training loop ở `:200`; cái **thiếu** là đường nạp checkpoint phát hành
   (`--restore_file` khai ở `:74`, không dùng). Claim `model.py:41 clip.load(device='cuda')`
   **không tái lập**; quan sát được `train.py:88`.
3. **combiner**: "NOT_A_BLOCKER" đúng về chạy nhưng thiếu vế "không thể trở thành paper evidence vì
   `unbenchmarked()==True`" (`benchmark_references.py:207`).

**Claim không tái lập được (ghi rõ, không suy diễn):** env airknow 3.11.15/torch 2.7.1; `moviepy`
cài được vào env đó; load/inspect `clvc` 6 checkpoint; giải mã pickle `Fashioniq.pt` của tgcir;
đếm key combiner 12/669 và 413/669 khác stock; số parameter thực sự được forward của `pair` ngoài
alias loader.
