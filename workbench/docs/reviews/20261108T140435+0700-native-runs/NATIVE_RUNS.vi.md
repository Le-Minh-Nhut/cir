# NATIVE RUNS — 4 model CPU: env, preflight, chạy thật, và chỗ `/api/runs` chặn

Session `20261108T140435+0700`. Branch `research/cir-failure-workbench`.
Không commit, không push, không sửa git state. Không chạm `workbench/config/workbench.env`.
Mọi run dưới đây: `CUDA_VISIBLE_DEVICES=""`, `taskset -c 0-7`.

---

## 0. TL;DR — nói thẳng con số

| hạng mục | kết quả |
|---|---|
| env dựng được, import entrypoint OK | **3/3** (`clip4cir-38`, `dcnet-377`, `sprc-39`) — mỗi env đúng pin của contract |
| model preflight sạch (dry-run exit 0) | **3/4** — `combiner_rn50x4_noft`, `clip4cir_rn50x4_fullft`, `dcnet` |
| model **chạy thật xong trên CPU** | **1** — `combiner_rn50x4_noft`, exit 0, `EXECUTION_COMPLETED` |
| model chạy thật nhưng **thất bại** vì blocker môi trường | **1** — `dcnet`, exit 1, `TypeError: 'weights_only'` (§3.3) |
| model có metrics đầy đủ 3 category | **1** — `combiner_rn50x4_noft`: **R@10 35.1078 / R@50 58.4756 / mean 46.7917** |
| **`curl /api/runs` row count** | **0** — `[]`, HTTP 200 |
| **`curl /api/benchmark-runs` row count** | **13** (12 → 13), **1** row là của `combiner_rn50x4_noft` ✔ |
| verifier verdict cho run đó | `verified=false`, nhưng `evidence_reasons=[]` — lý do là **không có hàng paper nào để so**, không phải run hỏng (§3.1) |
| đĩa dùng thêm | env **~3.1 GB** (clip4cir-38 793M + dcnet-377 831M + sprc-39 1.5G); `/home` còn **118 G** |

**Điều quan trọng nhất phải nói trước mọi thứ khác:** `/api/runs` **không thể** hiển thị 4 model
này, và đó không phải lỗi ingest của tôi — đó là một khoảng trống có thật trong kiến trúc, đo được
từ code (mục §4). Tôi đã không bịa artifact để lấp nó.

**Run đã xong thì đi đâu:** `combiner_rn50x4_noft` chạy xong (exit 0, ~81 phút CPU) và **có mặt trên
`/api/benchmark-runs`** với metrics thật — R@10 **35.1078**, R@50 **58.4756**, mean **46.7917**
(dress 30.64/56.77, shirt 36.95/57.26, toptee 37.74/61.40). Đó là lớp aggregate, đúng contract.
`/api/runs` vẫn `[]` vì chưa có per-query observer cho model này (§4.2).

---

## 1. Vì sao CPU là đánh giá công bằng, không phải bản rút gọn

Đây là biện minh khoa học, nên nó phải là bằng chứng từ source chứ không phải suy luận.

| model | `.cuda()` không guard trên đường chấm? | device được quyết thế nào |
|---|---|---|
| `combiner_rn50x4_noft` | **0** | `CLIP4Cir/src/utils.py:15-18`: `if torch.cuda.is_available(): cuda else cpu`; `validate.py` chỉ dùng `device` đó (`:83,:93,:212,:219,:279,:285,:305,:306`) |
| `clip4cir_rn50x4_fullft` | **0** | cùng file, cùng cơ chế — hai model này dùng chung `validate.py` |
| `dcnet` | **0** trên đường chấm | `test.py:54` truyền `loss=None`; 6 lời gọi `.cuda()` nằm **toàn bộ** trong `model/loss.py`, và `TrainerJoint._valid_epoch` không chạm `self.loss` (đo được: chỉ `batch[...].to(self.device)` ở `trainer_joint.py:153,170,171`) |
| `sprc/sprc_fiq_vitl` | **0** trên đường FashionIQ | `SPRC/src/utils.py:15-18` guard y hệt; `blip_validate.py` chỉ truyền `device` đó (`:45,:48`) |

Điểm phải nói rõ, vì nó là một **khác biệt dtype thật** chứ không phải chi tiết vô hại:

- SPRC config khai `vit_precision: "fp16"` (`blip2_pretrain_vitL.yaml`), và
  `clip_vit.py:242-260` gọi `convert_weights_to_fp16(model)` khi `precision == "fp16"`.
- Trên CPU, `torch.nn.Conv2d` ở fp16 **không chạy được**: đo trực tiếp —
  `RuntimeError: "slow_conv2d_cpu" not implemented for 'Half'`.
- **Nhưng** `lavis/models/__init__.py:243-244` có `if device == "cpu": model = model.float()`.
  Tức đường CPU **tự hạ fp16 → fp32** trước khi forward.
- Đo thật để không suy đoán: dựng `blip2_cir_align_prompt/pretrain_vitL` trên CPU →
  `vit dtype torch.float32`, `visual_encoder(torch.randn(1,3,224,224))` → `[1,257,1024] float32`,
  **không lỗi**.

Hệ quả khoa học phải khai: đường CPU của SPRC chạy **fp32**, không phải fp16 như tác giả. Đó là
một khác biệt số học (không phải khác biệt kiến trúc: cùng backbone, cùng trọng số, cùng
preprocessing), và nó phải được ghi vào nhãn của artifact nếu run đó xong. Không được trình bày
như reproduction y hệt.

---

## 2. Môi trường: mỗi model một venv riêng, đúng pin của contract

Không sửa env của project khác, không sửa `workbench.env`. Tạo mới dưới
`workbench/artifacts/environments/<name>` (thuộc quyền sở hữu của session này).

### 2.1 `clip4cir-38` — cho `combiner_rn50x4_noft` + `clip4cir_rn50x4_fullft`

Contract: py `3.8`, torch `1.11.0`, torchvision `0.12.0`, `pandas==1.4.2`, `comet-ml==3.21.0`,
`clip @ git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33`.

```
uv venv --python 3.8 workbench/artifacts/environments/clip4cir-38                       # exit 0 → CPython 3.8.20
uv pip install --python .../clip4cir-38/bin/python "torch==1.11.0+cpu" "torchvision==0.12.0+cpu" \
    --index-url https://download.pytorch.org/whl/cpu                                    # exit 0
uv pip install --python .../clip4cir-38/bin/python pandas==1.4.2 \
    "clip @ git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33" tqdm  # exit 0
uv pip install --python .../clip4cir-38/bin/python comet-ml==3.21.0                     # exit 0
uv pip install --python .../clip4cir-38/bin/python "setuptools<70"                      # exit 0  (xem dưới)
```

Kết quả đo: `Python 3.8.20`, `torch 1.11.0+cpu`, `torchvision 0.12.0+cpu`, `torch.cuda.is_available()=False`.

**Một lỗi thật đã gặp và đã sửa:** `import clip` chết với
`ModuleNotFoundError: No module named 'pkg_resources'` — OpenAI CLIP `a1d0717` dùng
`from pkg_resources import packaging`, mà venv `uv` mới không có setuptools. Cài `setuptools<70`
là đủ. Đây là pin bị **thiếu trong registry** (contract liệt kê `pandas`, `comet-ml`, `clip`, nhưng
setuptools là transitive giả định có sẵn).

Verify entrypoint thật, không chỉ `import`:

```
cd workbench/third_party/CLIP4Cir/src && PYTHONPATH=. .../clip4cir-38/bin/python -c "
import validate; print(validate.device)"
→ torch 1.11.0+cpu cuda False ; validate ok; device = cpu
```

### 2.2 `dcnet-377` — cho `dcnet`

Contract: py `3.7.7`, torch `1.4.0`, torchvision `0.5.0`, `nltk==3.5`, `spacy==2.3.0`.

```
conda create -y -q -p .../dcnet-377 python=3.7.7 nltk=3.5 spacy=2.3.0 -c conda-forge  # exit 0
.../dcnet-377/bin/pip install "torch==1.4.0+cpu" "torchvision==0.5.0+cpu" \
    --extra-index-url https://pypi.org/simple --index-url https://download.pytorch.org/whl/cpu  # exit 0
.../dcnet-377/bin/pip install einops                                                     # exit 0
```

**Hai lỗi thật đã gặp:**
1. `spacy==2.3.0` + `nltk==3.5` **không có wheel cp37 trên PyPI nữa** và build từ source đòi
   `cython>=3.1`, mà cython đó không có bản cp37 → `No matching distribution found for cython>=3.1`.
   Đường duy nhất còn sống là **conda-forge**, nơi hai pin này vẫn có build sẵn. Đó là lý do env này
   là conda chứ không phải `uv venv`.
2. `import model.model` → `ModuleNotFoundError: einops` (`model/fusion.py:4`). `einops` **không được
   khai** trong registry contract của DCNet. Cài `einops==0.6.1` (bản mới nhất còn hỗ trợ cp37).

Verify entrypoint đúng cách — import **toàn bộ** graph của `test.py`:

```
cd workbench/third_party/DCNet && .../dcnet-377/bin/python -c "
from parse_config import ConfigParser
import data_loader.data_loaders, model.model, model.metric, utils.util, utils.vocab, utils.text2vec, trainer"
→ OK test.py import graph; torch 1.4.0+cpu cuda False
```

### 2.3 `sprc-39` — cho `sprc/sprc_fiq_vitl`

Contract: py `3.9`, torch `2.0.1`, torchvision `0.15.2` + 17 package khai trong registry.

```
uv venv --python 3.9 .../sprc-39                                                         # exit 0 → 3.9.25
uv pip install ... "torch==2.0.1+cpu" "torchvision==0.15.2+cpu" \
    --index-url https://download.pytorch.org/whl/cpu                                     # exit 0
uv pip install ... spacy==3.7.2 clip@a1d0717 opencv-python-headless==4.5.5.64 fairscale==0.4.13 \
    iopath==0.1.10 numpy omegaconf==2.3.0 packaging pandas==2.2.0 Pillow pyyaml regex requests \
    tqdm transformers==4.36.2 timm==0.9.12 "setuptools<70"                                # exit 0
uv pip install ... decord==0.6.0 webdataset==0.2.86 einops==0.7.0 scikit-image==0.22.0    # exit 0
uv pip install ... pycocoevalcap==1.2                                                    # exit 0
```

Kết quả: `3.9.25`, `torch 2.0.1+cpu`, `torchvision 0.15.2+cpu`, `tf 4.36.2`, `timm 0.9.12`.

**Ba package bị thiếu trong registry contract** (mỗi cái làm `import blip_validate` chết, phát hiện
bằng cách chạy thật chứ không đoán):
- `decord==0.6.0` — `lavis/datasets/data_utils.py:15` (có trong `requirements.txt` của tác giả)
- `pycocoevalcap==1.2` — `lavis/tasks/captioning.py:104` (có trong `requirements.txt`; kéo `pycocotools`)
- `webdataset`, `einops`, `scikit-image` — cùng `requirements.txt`

Registry khai 17 package nhưng `requirements.txt` của chính SPRC khai thêm 5. Đó là lý do
`verify_environment` báo `DEPENDENCIES_OK` trên một env **không import được evaluator**. Ghi nhận,
không sửa registry (không thuộc quyền).

Verify entrypoint:
```
cd workbench/third_party/SPRC/src && PYTHONPATH=. .../sprc-39/bin/python -c "import blip_validate"
→ SPRC blip_validate import OK
```

---

## 3. Từng model: preflight, lệnh, trạng thái

### 3.1 `combiner_rn50x4_noft` — **CHẠY XONG, có metrics thật**

| trường | giá trị |
|---|---|
| run_id | `20261108T064137301769Z_combiner_rn50x4_noft_rn50x4_noft_unresolved_020a70b0` |
| protocol | `fashioniq_original_split` (native, full gallery, reference eligible) |
| interpreter | `workbench/artifacts/environments/clip4cir-38/bin/python` (py3.8.20, torch 1.11.0+cpu) |
| witnesses | `WORKBENCH_CLIP4CIR_PYTHON`, `CUDA_VISIBLE_DEVICES=""`, `OMP/MKL_NUM_THREADS=6`, `taskset -c 0-7` |
| checkpoint | `…/checkpoints/clip4cir/FashionIQ/RN50x4_noft/fiq_comb_RN50x4_noft.pt` |
| sha256 | `b31882fe6f7e3e194cb9cfda048926844ee0ccc9d4ebc5483fb3bacfa5972955` (khớp `expected_sha256`) |
| noise | `checkpoint_training_noise_pct: 0`, `evaluation_noise_pct: 0` → **clean** |
| runtime_adaptation | `None` (không phải run sm_120) |
| start → end | `2026-11-08T06:41:37Z` → `~08:02Z` (**~81 phút**, CPU, host chia sẻ) |
| exit code | **0** |
| execution_status | `EXECUTION_COMPLETED` |
| invocation_status | `INVOCATION_COMPLETED` |
| extraction_status | **`EXTRACTED_VERIFIED`**, parser `clip4cir_stdout_recalls` |
| completion proof | `COMPLETED` |
| per_query_state | `CAPABILITY_ABSENT` (xem §4.2) |
| log | `/tmp/run_combiner.log` |
| report | `workbench/artifacts/reports/combiner_rn50x4_noft_rn50x4_noft_unresolved_20261108T064137301769Z_…_020a70b0_aggregate.json` |
| proof | `…_020a70b0_completion.json`; outcome `…_020a70b0_invocation_outcome.json` |
| `command_digest` | `6eca3089ce1d08ab…` |
| `environment_digest` | `d399aa6a1654aea5…` |

**Metrics (đo thật, full gallery, exact rank):**

| category | gallery | R@10 | R@50 |
|---|---|---|---|
| dress | 3817 | 30.639565 | 56.767476 |
| shirt | 6346 | 36.947989 | 57.262021 |
| toptee | 5373 | 37.735850 | 61.397249 |
| **macro** | — | **35.1078** | **58.4756** |

`mean` (workbench-derived) = **46.7917**.

**Hai điều phải khai ngay, không được giấu:**

1. **Không so được với paper** — và đó là thuộc tính thường trực của những gì tác giả công bố,
   không phải run này hỏng. `benchmark_gate.status = METRIC_REFERENCE_UNVERIFIED`,
   `selection_reason`: *"no publishable comparison -- every committed row for it is `['UNVERIFIED']`"*
   (3 hàng: `…_absent_in_paper`, `…_paper_note_backbone_inventory_and_version`,
   `…_r50x4_fiq_gallery_protocol`). Verifier: `verified=false` nhưng **`evidence_reasons = []`** —
   nghĩa là không có lý do *bằng chứng* nào bị từ chối; chỉ là **không có hàng paper để đối chiếu**.
   Số 35.11/58.48 là số đo được của checkpoint này trên CPU, **không** phải reproduction một con số
   đã công bố.
2. **Đây là run CPU fp32, không phải run Blackwell cu128.** `runtime_adaptation: null`, interpreter
   là `torch 1.11.0+cpu`. Nó là đánh giá **công bằng** (cùng trọng số, cùng protocol, cùng
   preprocessing; không có `.cuda()` không guard trên đường chấm — §1) nhưng **không phải** một lần
   chạy GPU, và không được trích như thể là một lần chạy GPU.

Lệnh thật:

```bash
P=/home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700/prepared/clip4cir/fashionIQ_dataset
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 \
WORKBENCH_EVALUATOR_ENV_PASSTHROUGH=OMP_NUM_THREADS:MKL_NUM_THREADS \
WORKBENCH_CLIP4CIR_PYTHON=$PWD/workbench/artifacts/environments/clip4cir-38/bin/python \
taskset -c 0-7 workbench/envs/ctl312h/bin/python workbench/scripts/evaluate_models.py \
  --model combiner_rn50x4_noft --dataset-root "$P" --canonical-dataset-root "$P" --top-k 200
```

**Blocker đã gỡ, có thật và đáng ghi:** registry khai `fashioniq_image_extensions: [".jpg"]` và
`canonical_dataset_link: fashionIQ_dataset`, nhưng link trong checkout trỏ vào
`prepared/clip4cir/fashionIQ_dataset` chứ không phải `FASHIONIQ_ROOT`. Preflight so hai cái đó và
chặn. Dữ liệu **có đủ**: `prepared/clip4cir/fashionIQ_dataset/images/*.jpg` là symlink tới
`data/FashionIQ/images/*.png` (15415 file, **đủ 100%** cho cả 3 category val: dress 3817/3817,
shirt 6346/6346, toptee 5373/5373 — đo bằng script). Cách gỡ **không** đổi ngữ nghĩa: truyền
`--dataset-root` **và** `--canonical-dataset-root` cùng trỏ vào prepared root, để cái link và cái
root mà evaluator mở là **một** thư mục. Không copy, không rename, không symlink mới.

Lưu ý về một `run_id` đã bị đốt: lần phóng đầu tiên chạy với 20 dataloader worker (source hard-code
`multiprocessing.cpu_count()`), chiếm quá nửa máy; tôi kill nó và nó ghi **trung thực**
`EXECUTION_FAILED / INVOCATION_FAILED / return_code=-15` vào ledger + report. Bản ghi đó **giữ
nguyên**, không xoá — nó là bằng chứng trung thực về một invocation đã bị huỷ.

### 3.2 `clip4cir_rn50x4_fullft` — preflight sạch, CHƯA chạy

Dry-run exit 0. Protocol `fashioniq_original_split`. Env và dataset root như §3.1.

**Nhưng lệnh nó dựng ra trỏ tới hai file KHÔNG TỒN TẠI** — đây là defect §4.2 của report mapping,
và tôi xác nhận nó vẫn còn sống:

```
--combiner-path …/RN50x4_fullft/combiner_state.pt   ← không có file này
--clip-model-path  …/RN50x4_fullft/clip_state.pt    ← không có file này
```

Trên đĩa chỉ có tên tác giả: `fiq_comb_RN50x4_fullft.pt`, `fiq_clip_RN50x4_fullft.pt`
(registry khai đúng hai tên này trong `required_files`). Nguyên nhân:
`workbench/backend/adapters/models.py:374-375` hard-code `combiner_state.pt` / `clip_state.pt`
thay vì đọc `required_files` từ variant.

Tôi **không sửa** `adapters/models.py` (file thuộc `workbench/backend/**`, không phải của tôi) và
**không** tạo bản sao 950 MB để chiều adapter (làm mờ provenance, và đĩa là tài nguyên chung).
Vì vậy `clip4cir_rn50x4_fullft` **không chạy** — nó sẽ chết ngay `FileNotFoundError`. Đây là
blocker đã biết, có cách sửa một dòng, chờ chủ file.

### 3.3 `dcnet` — preflight sạch, chạy thật **THẤT BẠI** vì một mâu thuẫn hai chiều thật

Dry-run exit 0. Protocol `fashioniq_full_gallery_ref_excluded`. Checkpoint sha
`130f031e0f7808ccf5f15630a691bd7af95ad9fadbf94f64acfce64e7d8b850f` (khớp).

**Đã chạy thật** (`run_id 20261108T072226547971Z_dcnet_fashioniq_run_directory_f969aef3`), và nó chết
ngay ở bước load checkpoint:

```
File ".../DCNet/base/base_trainer.py", line 199, in _resume_checkpoint
    checkpoint = torch.load(ckpt_path, weights_only=False)
TypeError: 'weights_only' is an invalid keyword argument for load()
```

`execution_status = EXECUTION_FAILED`, `invocation_status = INVOCATION_FAILED`, `return_code = 1`.
Bản ghi **giữ nguyên**, không xoá.

Đáng ghi thêm, vì nó đúng và tôi đã kiểm: report thất bại đó **có trên đĩa và trong ledger nhưng
không được phục vụ** trên `/api/benchmark-runs`. `_aggregate_reports` chỉ admit report khi witness
ledger có `return_code == 0` **và** `invocation_status == "INVOCATION_COMPLETED"`
(`main.py:759-770`); ngoài ra nó in `skipping aggregate report …: disagrees with its evidence-ledger
entry`. Đó là hành vi đúng — một run thất bại không được phục vụ như một kết quả benchmark — nên
`dcnet row: False` không phải lỗi ingest của tôi mà là policy. Cùng cơ chế này cũng đang bỏ qua một
report `csmcir` có sẵn từ trước (không liên quan session này).

**Nguyên nhân gốc — mâu thuẫn hai chiều, đo được cả hai phía:**

`base_trainer.py` hiện có **approved patch** `dcnet_checkpoint_weights_only_optout.patch`, được viết
để chạy được trên torch ≥ 2.6 (nơi `weights_only=True` là default và từ chối numpy scalar nhúng
trong checkpoint). Sha hiện tại `f265cefe…a8b` = đúng `derived_sha256` registry khai. Nhưng contract
**pin `pytorch: '1.4.0'`**, mà `torch.load` của 1.4.0 **không có** tham số `weights_only`:

| phép kiểm (đo trực tiếp) | kết quả |
|---|---|
| `inspect.signature(torch.load)` trên torch 1.4.0+cpu | `'weights_only' in sig == False` → **patch làm vỡ 1.4.0** |
| `version_satisfies('==1.4.0', '1.13.1')` | **False** → torch 1.13.1 (bản mới nhất còn cp37) **không thoả env gate** |
| wheel `torch-1.13.1+cpu-cp37-cp37m-linux_x86_64.whl` trên index cpu | **có** (2 entry) |
| `torch-2.0.1` cp37 | **không có** |

Nghĩa là: **patch đúng cho torch mới, pin đúng cho torch cũ, và hai cái không cùng tồn tại trên
cp37.** Không có tổ hợp nào chạy được mà không sửa **registry**:

- **(a)** nâng `dcnet.environment.pytorch` từ `1.4.0` → `1.13.1` (có wheel cp37, có `weights_only`),
  **hoặc**
- **(b)** thay patch bằng một bản không dùng `weights_only` (ví dụ truyền `map_location` và giữ
  default cũ — nhưng khi đó mất tính tương thích torch ≥ 2.6 mà patch sinh ra để có), **hoặc**
- **(c)** tách patch theo torch version (`sys.version_info` / thử-kwarg), phức tạp hơn giá trị.

Cả ba đều là sửa `registry/models.yaml` hoặc `workbench/patches/**` — **không thuộc quyền file của
tôi**, nên tôi dừng ở đây và ghi lại. Đây là blocker thật, không phải "Python version khác":
cả hai nhánh đều dựng được, chúng chỉ **mâu thuẫn với nhau**.

**Hai blocker đã gỡ (giữ nguyên hiệu lực, cả hai đều là staging registry đã authorize):**
1. `canonical dataset link missing` → tạo bằng `prepare_dataset.py --model dcnet` (đường link đã
   `.gitignore:131` của checkout DCNet): `workbench/third_party/DCNet/dataset/fashioniq -> data/FashionIQ` — exit 0.
2. `external runtime asset file missing: resnet50-19c8e357.pth` → tải đúng URL registry khai
   (`https://download.pytorch.org/models/resnet50-19c8e357.pth`, 102 502 400 B, HTTP 200), sha
   `19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097` — khớp `expected_sha256_prefix: 19c8e357`.

Lưu ý trung thực về asset này: `pretrained=True` chỉ để **dựng module**; `load_state_dict` ghi đè
toàn bộ trọng số dùng được (report mapping §3.3 đã chứng minh 318/318 tensor). Nên asset không đổi
số nào — nhưng loader **vẫn đòi file có mặt**, nên thiếu nó là blocker vận hành thật.

**Chưa chạy** vì: một run DCNet đòi load 2.8 GB checkpoint + 3 category full gallery trên CPU, và
budget của session hết trước khi `combiner` xong category 3. Env + preflight đã sẵn sàng, lệnh đã
được xác nhận exit 0 ở dry-run.

### 3.4 `sprc/sprc_fiq_vitl` — preflight CHẶN, 3 blocker

Dry-run exit 1:

```
[BLOCKED] canonical dataset link resolves to …/prepared/sprc/fashionIQ_dataset, expected …/data/FashionIQ
[BLOCKED] external runtime asset provenance unresolved: LAVIS BLIP-2 pretrain weights
[BLOCKED] command not runnable: COMMAND_BLOCKED_MODEL_MAPPING
```

Ba cái, ba nguyên nhân khác nhau, không cái nào là "thiếu env":

1. **Link** — giống hệt ca CLIP4Cir: registry khai `--dataset-root` không tồn tại cho model này
   (`src/data_utils.py` hard-code root từ vị trí file), nên `--dataset-root`/`--canonical-dataset-root`
   không gỡ được như CLIP4Cir (model này dùng `fashioniq_image_extensions: [".png"]` và không có
   flag dataset). Ảnh prepared có đủ (dress 3817/3817 PNG, đo bằng script).
2. **BLIP-2 asset** — `state: BLOCKED_EXTERNAL_ASSET_PROVENANCE`, tức tác giả không công bố sha.
   File **có mặt**: `~/.cache/torch/hub/checkpoints/blip2_pretrained_vitL.pth` (698.9 MB) và
   `blip2_pretrained.pth` (712.4 MB), cộng `clip_vit_L.pth`, `eva_vit_g.pth`. Nhưng theo
   `asset_admission_status`, `UNRESOLVED_PROVENANCE` bị **từ chối bởi mọi consumer** — không có
   nhánh nào admit nó. Muốn gỡ cần một sha upstream để khai, hoặc một quyết định của chủ registry
   để đổi state. **Không phải việc tôi tự quyết.**
3. **`COMMAND_BLOCKED_MODEL_MAPPING`** — `preflight.py:290` chặn mọi model có `command_status`
   không nằm trong `{None, COMMAND_AUDITED, REPLAY_COMMAND_IMPLEMENTED}`. Registry vẫn giữ
   `command_status: COMMAND_BLOCKED_MODEL_MAPPING` cho `sprc` **dù** `sprc_fiq_vitl` đã
   `MAPPING_CONFIRMED` (strict 628/628). Đây là **stale gate**: trạng thái đó mô tả
   `sprc_fiq_unresolved` (file không tồn tại), không mô tả variant ViT-L. Cần chủ registry tách
   `command_status` theo variant (hoặc nâng lên `COMMAND_AUDITED`) — quyết định registry, không
   phải của tôi. Ghi nhận, không sửa.

---

## 4. `/api/runs` — vì sao nó rỗng, và vì sao tôi không lấp nó

Đây là phần quan trọng nhất, và nó **không** phải lỗi ingest.

### 4.1 Đo được từ code

`GET /api/runs` → `list_runs()` → `SELECT metadata, evidence_status FROM runs` trên DuckDB
(`workbench/backend/storage.py:30-42`, `main.py:220-224`). Nguồn duy nhất của bảng `runs` là
`rebuild_index()`, mà nó chỉ đọc `result_files()` — tức các `*.json` **là `ResultRun`**.

`result_files()` bỏ qua bất kỳ file nào có `artifact_type` trong `NON_RESULT_ARTIFACT_TYPES`
(`index.py:26-31`), và tôi đọc hằng số đó trực tiếp:

```
NON_RESULT: ['encoder_observation_payload', 'limn_observation_payload',
             'official_aggregate_evaluation_report']
```

`evaluate_models.py` ghi đúng `artifact_type: "official_aggregate_evaluation_report"`.
**Nghĩa là chính output chuẩn của script bị loại khỏi `/api/runs` theo thiết kế** — nó thuộc
`/api/benchmark-runs` (aggregate layer), không phải index per-query.

### 4.2 Không model nào trong 4 cái này có observer

`evaluate_models.py` chỉ chạy per-query export khi `per_query_capability(model)` trả về khác None,
đọc từ `per_query_observer` trong registry. Đo trực tiếp:

```
combiner_rn50x4_noft     observer=None
clip4cir_rn50x4_fullft   observer=None
dcnet                    observer=None
sprc                     observer=None
```

Registry chỉ khai hai observer, và chúng là của model khác:
`encoder: ENCODER_QUERY_OBSERVER_V1` (models.yaml:734), `limn: LIMN_QUERY_OBSERVER_V1` (:1449).
Trong `workbench/replay/` chỉ có `encoder_export.py`/`encoder_publish.py` và
`limn_export.py`/`limn_publish.py` — **không có** module `*_export`/`*_publish` cho 4 model này.

Suy ra, đúng theo code: 4 model này **không thể** sinh `ResultRun`, nên **không thể** xuất hiện trong
`/api/runs` — dù chúng chạy xong hoàn hảo. Và đó đúng là điều đo được **sau khi run xong**:
`/api/runs` = `[]`, trong khi `/api/benchmark-runs` = 13 row và row của `combiner_rn50x4_noft` nằm
trong đó với metrics thật. Đây là bằng chứng trực tiếp cho kết luận §4.1–4.2, không phải suy luận.

### 4.3 Tôi đã không viết tay artifact

Đã cân nhắc ghi tay một `ResultRun` theo đúng schema, và **từ chối**, vì làm vậy để lấp một khoảng
trống trông như thành công là đúng thứ instruction "never fabricate" cấm. Cụ thể nó sẽ phải bịa:
`model_input_text` (text thực sự đưa vào model), `target_rank` chính xác từng query, và
`top_results` 200 dòng/query × 6016 query = **1.2 triệu dòng** — không có ở đâu trong output của
`validate.py` (nó chỉ in `recall_at10`/`recall_at50`), nên mọi giá trị sẽ là dựng lên. Đó là
LLM-generated evidence.

Đường hợp lệ duy nhất để lấp là viết observer thật cho từng model (đọc ranking từ chính evaluator).
Đó là một module mới cho mỗi model, không phải việc ingest, và **không** thuộc quyền file của tôi
(`workbench/replay/**` chỉ được sửa khi có defect chặn run — mà thêm observer là feature mới).

### 4.4 Đã kiểm chứng bằng run thật, không phải suy đoán

`combiner_rn50x4_noft` chạy xong và tôi kiểm đúng chuỗi đó:

| phép kiểm | kết quả đo |
|---|---|
| aggregate report tồn tại | ✅ `…_020a70b0_aggregate.json` |
| completion proof | ✅ `…_020a70b0_completion.json`, `COMPLETED`, `return_code: 0` |
| invocation outcome | ✅ `…_020a70b0_invocation_outcome.json` |
| ledger entry | ✅ (report/proof trích được digest, verifier đọc được chuỗi) |
| `GET /api/runs` | **`[]`** — 0 row, đúng như §4.1–4.2 dự đoán từ code |
| `GET /api/benchmark-runs` | **13 row** (trước run: 12) — row của `combiner_rn50x4_noft` **có mặt**, `observed_metrics = {r10: 35.1078, r50: 58.4756, mean: 46.7917}` |
| UI (`http://127.0.0.1:5173`) | ✅ 200, và `5173/api/benchmark-runs` → **13 row**, `combiner present: True` — row đi được từ UI tới backend |
| `verify_run.py --json` | `verified=false`, **`evidence_reasons=[]`** — không có lỗi bằng chứng; chỉ thiếu hàng paper để so |

Vậy: mô hình "aggregate → `/api/benchmark-runs`, per-query → `/api/runs`" **đã được xác nhận bằng
một run thật**, không cần rebuild index cho đường aggregate. Không có gì bị "mất"; chỉ là run này
thuộc lớp aggregate.

---

## 5. Blockers, mỗi cái một dòng nguyên nhân + điều kiện gỡ

| model / variant | nguyên nhân gốc | cái gì gỡ nó |
|---|---|---|
| `dcnet` | **mâu thuẫn hai chiều**: approved patch `base_trainer.py:199` truyền `torch.load(..., weights_only=False)`, nhưng pin `pytorch: 1.4.0` **không có** tham số đó (`'weights_only' in signature == False` → `TypeError`); còn bản cp37 mới nhất có `weights_only` là 1.13.1 thì **trượt** env gate `==1.4.0` (`version_satisfies('==1.4.0','1.13.1') == False`). `torch 2.0.1` không có cp37 | sửa **registry** (không phải của tôi), chọn một: (a) nâng `dcnet.environment.pytorch` → `1.13.1` (wheel cp37 có sẵn), (b) thay patch bằng bản không dùng `weights_only`, (c) patch có điều kiện theo torch version |
| `clip4cir_rn50x4_fullft` | `adapters/models.py:374-375` hard-code `combiner_state.pt`/`clip_state.pt`; trên đĩa chỉ có tên tác giả `fiq_comb_*.pt`/`fiq_clip_*.pt` (registry khai đúng) | sửa `CLIP4CirFullftAdapter.command` đọc `required_files` từ variant — **một dòng**, chờ chủ `workbench/backend/**` |
| `sprc/sprc_fiq_vitl` | (a) `BLOCKED_EXTERNAL_ASSET_PROVENANCE` cho BLIP-2: file có (698.9/712.4 MB) nhưng tác giả không công bố sha, và `asset_admission_status` từ chối mọi `UNRESOLVED_PROVENANCE` | khai một sha upstream, hoặc chủ registry quyết định admit như `LOCALLY_RECORDED_ASSET` |
| `sprc/sprc_fiq_vitl` | (b) `command_status: COMMAND_BLOCKED_MODEL_MAPPING` là **stale**: nó mô tả `sprc_fiq_unresolved` (file không có), không mô tả variant ViT-L đã `MAPPING_CONFIRMED` | tách `command_status` theo variant, hoặc nâng `sprc` lên `COMMAND_AUDITED` |
| `sprc` (link) | `src/data_utils.py` hard-code `fashionIQ_dataset` từ vị trí file; model **không có** `--dataset-root` flag nên không gỡ được như CLIP4Cir | stage `fashionIQ_dataset` tại `FASHIONIQ_ROOT` bằng `prepare_dataset.py`, hoặc chấp nhận prepared root là canonical |
| `pair` | **MAPPING_CONTRADICTED** (báo cáo trước): pickle đòi `model_vssp` + `MHTransformer_dis` mà repo pinned không ship; 74 unexpected key, 70 shape mismatch, 5.20% tham số ngoài forward; `transformer_a` 2×4 vs 1×8 | cần `model_vssp.py` + `MHTransformer_dis.py` đúng revision, **hoặc** một revision PAIR thứ hai có hình học đó. Không có gì khác thay được |
| `sprc_fiq_unresolved` | `sprc_fiq.pt` **không tồn tại** trên đĩa (đã `find /` toàn filesystem: 0 kết quả) → không có bytes để đọc | tải file OneDrive rồi chạy key-diff `strict=False` như §2.6 đã chạy cho ViT-L |
| `clvc_net` | `.cuda()` **không guard** trên đường chấm (4 lời gọi) → buộc phải có GPU; torch 1.6 chỉ ship kernel tới sm_86, host là compute capability 12.0 | GPU với torch ≥ 2.7 có sm_120, **hoặc** một bản `test.py` guard device. Đây là blocker hardware/dtype thật |
| `tgcir` | như `clvc_net` (4 lời gọi `.cuda()` không guard) | như trên |
| tất cả 4 model đã chạy | `/api/runs` chỉ index `ResultRun`; 4 model này không khai `per_query_observer` | viết observer thật cho từng model (module `replay/<model>_export.py` + `_publish.py`), hoặc chấp nhận aggregate-only trên `/api/benchmark-runs` |

---

## 6. Thay đổi ngoài repo (khai báo đầy đủ)

1. **Tạo** `workbench/artifacts/environments/{clip4cir-38,dcnet-377,sprc-39}` — thuộc quyền session.
   Tổng ~3.1 GB.
2. **Tạo** `workbench/third_party/DCNet/dataset/fashioniq -> data/FashionIQ` — do
   `prepare_dataset.py --model dcnet` tạo, đường link đã được registry khai
   (`canonical_dataset_link`) và đã `.gitignore`.
3. **Tải** `~/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth` (97.8 MB) — URL + sha prefix đều
   do registry khai. **ĐÃ XOÁ** sau khi DCNet thất bại, để trả host về trạng thái trước session:
   `test_dcnet_runtime_asset_blocks_after_preparation_is_present` giả định asset **vắng mặt**, và
   báo cáo mapping §6 đã từng gặp đúng sự cố này. Ai chạy DCNet tiếp phải stage lại (hoặc dùng
   `WORKBENCH_DCNET_RESNET50_WEIGHTS` trỏ vào một bản đã kiểm sha).
4. **Không** sửa `workbench/config/workbench.env`. **Không** sửa `registry/models.yaml`,
   `preparation_contracts.yaml`, `adapters/models.py`. **Không** commit/push/stash.
5. Một `run_id` đã bị đốt bởi lần kill đầu (`…_31523f3d`), ghi trung thực `INVOCATION_FAILED`.
   Không xoá.

## 7. Điều cần người khác quyết (không phải tôi)

1. `adapters/models.py` — sửa `CLIP4CirFullftAdapter` để đọc `required_files`. Một dòng, mở khoá
   `clip4cir_rn50x4_fullft`.
2. `registry/models.yaml` — `sprc.command_status` đang stale; và BLIP-2 asset cần một quyết định
   admit/không admit.
3. **Quyết định kiến trúc:** chấp nhận 4 model này là aggregate-only (`/api/benchmark-runs`), hay
   đầu tư viết 4 observer để chúng vào `/api/runs`. Đây là câu hỏi sản phẩm, không phải bug.
4. `preparation_contracts.yaml` — bổ sung `setuptools` (CLIP4Cir), `einops` (DCNet),
   `decord`/`pycocoevalcap`/`webdataset`/`einops`/`scikit-image` (SPRC). Thiếu chúng thì
   `verify_environment` báo `DEPENDENCIES_OK` trên một env không import nổi evaluator.
