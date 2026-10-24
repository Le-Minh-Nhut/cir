# R19 — Báo cáo hoàn thành

Ngày: 2026-10-24. Nhánh `research/cir-failure-workbench`.

Phạm vi: 10 finding của audit độc lập trước khi chuyển sang PC
(`CIR_R19_PC_READINESS_AUDIT_2026-10-24.md`), cộng phần bàn giao PC và phần "còn lại của registry".

## 1. Revision, delivery

| Mục | Giá trị |
|---|---|
| Revision production đã kiểm thử | `4dd827f` (production tree; verify bằng `git diff <rev> HEAD -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'`) |
| `main` | `8ebcc74` — **không đổi** |
| Baseline R19 | `afe18fa` (audit độc lập kiểm), production tree khi đó = `ca10235` |
| Working tree | chỉ còn `src/train_cirr.py` + `uv.lock` (untracked, **không phải** việc của R19) |

## 2. Mười finding: reproduce → nguyên nhân → fix → kiểm

Mỗi finding đều được **tái hiện trước** khi sửa. Không finding nào bị bỏ qua, không finding nào bị
"đã sửa rồi" mà không có bằng chứng.

### R19-01 — Adapter khai sai đường dẫn evaluator

**Đã sửa.** Tái hiện: `TMEAdapter.script = "precompute_test.py"` và
`PthaMtstAdapter.script = "src/validate_blip.py"`, trong khi file thật ở
`TME/src/precompute_test.py` và `MTST_PTHA/MTST_PTHA/src/validate_blip.py`. `guarded_plan()` kiểm
`source / adapter.script` nên **resolver đúng của wrapper không cứu được** phép kiểm này: cả hai bị
từ chối `official evaluator missing` dù preflight của wrapper tìm thấy file.

Sửa: khai đúng đường dẫn tương đối tính từ checkout. Kiểm bằng chính `guarded_plan()` với registry và
checkpoint thật: sau khi sửa, blocker chuyển từ `official evaluator missing` sang blocker **thật**
(thiếu asset/env), tức phép kiểm đường dẫn đã qua.

### R19-02 — Checkpoint root bị thêm model id hai lần

**Đã sửa.** Tái hiện: cả hai wrapper tính `checkpoint_root / MODEL["model_id"] / filename`, thành
`checkpoints/tme/tme/best_model.pth` và `checkpoints/ptha_mtst/ptha_mtst/tuned_mtst_cirr.pt`, trong khi
file nằm một tầng trên. `_replay_command` đã truyền **member root** qua `checkpoint_member_root`.

Sửa: theo đúng quy ước `checkpoint_root / <filename>` (wrapper LIMN đã làm vậy ở `limn.py:50,180`).
Kiểm bằng lệnh thật trên checkpoint thật: adapter tạo
`--checkpoint-root workbench/artifacts/checkpoints/tme`, wrapper mở
`.../tme/fashioniq_0_best_model.pth`, file **tồn tại**.

### R19-03 — PTHA nhận cờ parser không định nghĩa + danh tính lệch

**Đã sửa.** Tái hiện: subprocess thật exit **2**,
`ptha_mtst.py: error: unrecognized arguments: --checkpoint-id cirr_transfer`, trước cả inference.

Sửa: parser nhận `--checkpoint-id`, **validate** theo `checkpoint_variants` của registry (id lạ bị từ
chối và liệt kê id hợp lệ), và **registered id là danh tính duy nhất** dùng trong command, output và
provenance — trước đây output đổi thành `ptha_mtst_cirr_transfer_exploratory` nên không cái nào khớp
nhau. Nhãn exploratory vẫn hiện rõ. Kiểm: `test_r19_ptha_wrapper.py` build command **theo đúng cách
adapter làm** rồi chạy trong subprocess (không chỉ `--help`, vì argparse thoát trước khi kiểm
invocation lỗi).

### R19-04 — Planner nhầm module trong source; TME thiếu env

**Đã sửa.** Tái hiện: cả hai adapter khai `launcher_roots = ()`, nên planner sinh yêu cầu
`data`, `utility`, `precompute-evaluation`, `salesforce-lavis` (TME) và `data-utils`, `utils`,
`validate-blip`, `salesforce-lavis` (PTHA) — đều là module/cây LAVIS **có trong source**. Đồng thời
các pin thật (`transformers`, `timm`) bị bỏ khỏi plan.

Sửa: `TMEAdapter.launcher_roots = ("src",)`,
`PthaMtstAdapter.launcher_roots = ("MTST_PTHA/src",)`. TME được khai environment từ **chính README đã
pin**: Python 3.10, torch 2.1.2, torchvision 0.16.2, CUDA 12.1, cộng `requirements.txt`.

Hai điều chỉnh registry **có bằng chứng** khi provisioning thật:
- **NumPy phải `<2`**: torch 2.1.2 không có C ABI cho NumPy 2.x; nếu không, import torch cảnh báo
  `Failed to initialize NumPy: _ARRAY_API not found` và cầu tensor↔ndarray chết.
- **Bỏ pin `mkl==2023.1.0` của tác giả**: không tồn tại module `mkl` import được (`import mkl` →
  `ModuleNotFoundError`), nên chính requirement đó **chặn** môi trường mà nó thuộc về. torch bundle
  mộtMKL; đã chứng minh stack chạy đủ sau khi bỏ (matmul, numpy bridge, `nn.Linear`).

Kiểm: env `workbench/envs/tme` (Python 3.10.13) **verified**, gồm **probe CUDA thật** trong đúng
interpreter đó.

### R19-05 — TME đọc sai tầng metric, chạy category 3 lần

**Đã sửa.** Tái hiện: `evaluate_features()` **đã** tự chạy đủ ba category và trả dict lồng
(`{"dress": {"recall_at1/10/50"}, ...}`), nhưng wrapper gọi nó **trong** vòng lặp ba category rồi đọc
`metrics.get("recall_at10")` ở cấp ngoài. Kết quả: **9 lần tính** thay vì 3, mọi `ReplayResult` có
`r1/r10/r50 = None`, `complete=False`, macro null.

Sửa: gọi native **một lần**, chuẩn hóa từng sub-dict theo category, macro tính lại; payload native
khuyết hoặc non-finite bị **từ chối** chứ không được lấy trung bình. Test dùng **đúng cấu trúc native**
và đếm số lần gọi.

### R19-06 — Downloader lưu archive nén dưới tên weight

**Đã sửa.** Tái hiện: probe endpoint Kaggle cho thấy 302 → `.../bundle/archive.tar.gz` và body
`1f8b0800`; downloader ghi thẳng bytes container vào file tên như weight, exit 0, "đã tải". Hai variant
còn khai **cùng `filename`** nên cùng trỏ một path.

Sửa: thêm `archive_member` + `archive_size_bytes` (tuỳ chọn, các model khác không đổi): container
stream → trích **đúng** member đã khai ra file tạm → verify (size + hash nếu có) → publish atomic.
`filename` hai variant **khác nhau** (`fashioniq_0_best_model.pth`, `cirr_0_best_model.pth`). Manifest
ghi size/hash **container và member riêng biệt**; hash local **không** được gọi là checksum tác giả.
Presence check **từ chối bytes container** tại đích của weight và **enforce size đã khai**.

**Đã tải thật:** container 2 533 586 661 B → member **2 728 382 104 B** (đúng số audit đo), sha256
local `fea7fc08…`, torch-zip 862 entry / 860 tensor.

### R19-07 — Thiếu feature cache và base assets

**Đã sửa.** Tái hiện: preflight chỉ kiểm annotation + checkpoint; native loader chết
`FileNotFoundError: ./features/FashionIQ/val_images.pt`; doctor báo external assets OK với **danh sách
rỗng**.

Sửa: thêm preparation contract `tme_fashioniq` (raw inputs, generated artifact `features/FashionIQ`,
external assets) và preflight **báo** file cache, **số byte cần** (suy từ tokens/width/dtype đã khai
nhân số ảnh gallery thật) và **dung lượng trống tại đích**. Hai điều chỉnh metadata từ việc làm tử tế:
- dtype là **fp16**, không phải fp32 (`precompute.py:37` dùng `autocast`, `create_eva_vit_g` mặc định
  `precision="fp16"`), nên yêu cầu **10.47 GiB** chứ không phải 20.94 GiB.
- **BLIP-2 pretrain weights BẮT BUỘC.** Một revision trước của chính tôi ghi là không bắt buộc, với lý
  do `load_from_pretrained` không được gọi. **Lý do đó SAI** và đã được ghi lại để không ai suy luận
  lại lần nữa: `blip2_pretrain.yaml` đặt `load_finetuned: False` **và** khai một `pretrained` URL, nên
  `load_checkpoint_from_config` (`base_model.py:89-102`) đi vào nhánh `load_pretrained` và **thật sự
  gọi** `load_from_pretrained`, nơi `blip2.py:87-89` trỏ URL đó về file local.
  Phép kiểm thực nghiệm trước đó bị **đọc sai**: dựng model khi thiếu base weight chết **trước tiên** ở
  `weight/eva_vit_g.pth` vì `create_eva_vit_g` chạy trong `init_vision_encoder` **sớm hơn** lần load
  pretrained — chết ở file EVA **không** chứng minh file BLIP-2 không dùng.
  Cả hai file nay được khai `expected_local_location` là **đúng đường dẫn cwd-relative** mà native mở
  (`third_party/TME/weight/...`), nên preflight kiểm chính file loader đọc.

### R19-08 — Không có parser aggregate

**Đã sửa.** Tái hiện: document JSON đầy đủ, hợp lệ, exit 0 → vẫn `AGGREGATE_PARSER_UNAVAILABLE`.

Sửa: đăng ký cả hai, parse **đúng shape thật** mà wrapper phát ra, validate category đầy đủ/duy nhất,
Recall hữu hạn trong 0..100, macro khớp các hàng, từ chối document ngoại lai/khuyết với lý do **phân
biệt được**, và mang **request identity** (`expected_checkpoint_id`, `expected_protocol_id`): document
mô tả checkpoint/protocol khác bị từ chối. So sánh paper của PTHA vẫn chỉ là **context**.

### R19-09 — Mapping chặn nhầm cả lựa chọn đã được cho phép

**Đã sửa.** Tái hiện: `select_checkpoints(..., policy="include_flagged_conditions", requested=[...])`
trả BLOCKED cho cả `fashioniq_0` lẫn `cirr_transfer`.

Sửa: **tách hai điều khác nhau**. Thứ được ghi: artifact **đúng là bản phát hành của tác giả** (TME:
endpoint do tác giả link, container một member; PTHA: repo Hub liệt đúng file đó dưới
`apache-2.0`, digest local khớp **byte-for-byte** với LFS oid đã công bố). Thứ **không** được ghi: rằng
nó tái lập hàng FashionIQ của paper — `reproduction_status` giữ nguyên câu trả lời "không"
(`SELECTED_TRANSFER_ONLY`). Registry còn khai `checkpoint_selection_preference: fashioniq_0` để weight
CIRR **không thể** trở thành mặc định FashionIQ do thứ tự.

### R19-10 — Lý do protocol sai + consumer không đồng bộ

**Đã sửa.** Tái hiện: test core **fail** với
`Unsupported FashionIQ protocol: 'fashioniq_val_split_ref_included'`, và annotation API trả HTTP 400
`unknown_protocol`.

Lý do R18 cho protocol mới — "`fashioniq_original_split` dùng gallery train split" — **sai**. Họ đó
luôn là **toàn bộ `split.<category>.val.json` theo thứ tự file, reference ELIGIBLE**
(`FASHIONIQ_PROTOCOL_PARITY.md`). TME dựng **đúng** cohort đó: `data.py:175-188` nạp split val vào
`current_image_names`, và `precompute_evaluation.py:83` chỉ mask reference khi `category == 'CIRR'`.
Hai cohort **trùng nhau**.

Sửa: **xoá** protocol thừa khỏi registry và frontend, TME trỏ về `fashioniq_original_split`, giữ id cũ
làm **alias** trong `src/evaluation/fashioniq.py` để record cũ đọc được thay vì lỗi, và schema/API
**dẫn xuất từ registry** thay vì hard-code.

## 3. Ngoài 10 finding

- **Download budget**: tính theo **container** khi variant là archive, không theo member đã trích; và
  profile `all_available` nêu một con số hữu hạn suy từ registry (**37 580 963 840 B**) thay vì con số
  nhỏ hơn chính artifact nó nhận là phải phủ.
- **6 test cũ phải sửa fixture**: chúng khai size/hash thật (hoặc dùng root model-id lồng) trong khi
  ghi placeholder byte. Sửa bằng cách **làm fixture trung thực**, **không** nới bất kỳ phép kiểm nào.
  Đây là bằng chứng phép kiểm size mới hoạt động: nó bắt được cả test của chính repo.
- **Release verifier báo `ENGINEERING_INCOMPLETE`**, 5 journey PENDING (cần live stack). Verifier exit 0
  **không** phải `RELEASE_READY`.
- **`acceptance.sh`: ACCEPTANCE PASSED** (4 live check SKIP vì không có server).

## 4. Trạng thái 17 model

| # | model | protocol native | variant chọn | đã chạy thật | blocker còn lại |
|---|---|---|---|---|---|
| 1 | `limn` | `fashioniq_val_split` | bundle 3 category | full val + per-query parity | — |
| 2 | `encoder` | `fashioniq_val_split` | `fashioniq` | 3 category GPU + per-query parity | — |
| 3 | `hint` | `fashioniq_val_split` | `fashioniq` | chưa | env + đĩa |
| 4 | `pair` | `fashioniq_val_split` | B1/B2 exploratory | **NOT RUN** | `model_vssp`/`MHTransformer_dis` không tồn tại ở mọi revision công khai |
| 5 | `clvc_net` | `fashioniq_val_split` | drive unresolved | chưa | map 6 file local/global |
| 6 | `tgcir` | `fashioniq_val_split` | archive | preflight-only | member `TG-CIR.zip` chưa xác minh |
| 7 | `habit` | `fashioniq_full_gallery_ref_excluded` | **`fiq_n02`** | READY | đĩa ~12 GiB |
| 8 | `intent` | `fashioniq_full_gallery_ref_excluded` | **`fiq_n02`** | ckpt hash khớp | env + đĩa |
| 9 | `conesep` | `fashioniq_full_gallery_ref_excluded` | **`fiq_n02`** | metadata verify | tải ckpt + env |
| 10 | `airknow` | `fashioniq_full_gallery_ref_excluded` | 20% **hoãn** | — | chỉ có 50%/80%; **không** thay thế |
| 11 | `dcnet` | `fashioniq_full_gallery_ref_excluded` | — | chưa | run directory tác giả |
| 12 | `csmcir` | `fashioniq_original_split` | `fashioniq` | chưa | env + đĩa |
| 13 | `combiner_rn50x4_noft` | `fashioniq_original_split` | — | chưa | member archive + mapping |
| 14 | `clip4cir_rn50x4_fullft` | `fashioniq_original_split` | — | chưa | cặp CLIP/Combiner khớp config |
| 15 | `sprc` | `fashioniq_original_split` | ViT-L variant | chờ file người dùng | OneDrive tải tay |
| 16 | `ptha_mtst` | `fashioniq_original_split` | **`cirr_transfer`** (CIRR→FashionIQ, exploratory) | ckpt hash khớp, preflight PASS | env py3.9/torch2.0.1 (~5 GB) |
| 17 | **`tme`** | `fashioniq_original_split` | `fashioniq_0` (0%); **20% `CHECKPOINT_UNAVAILABLE`** | ckpt **đã tải + trích**, env **verified + GPU probe thật** | **đĩa: ≥10.47 GiB** feature cache + 2 025 249 237 B EVA |

Không ô nào gọi "asset có sẵn" là "đã chạy".

## 5. TME: số đo thật

| Mục | Giá trị |
|---|---|
| Container Kaggle | `archive.tar.gz`, 2 533 586 661 B, magic `1f8b0800` |
| Member | `best_model.pth` **2 728 382 104 B** |
| Đích trên đĩa | `checkpoints/tme/fashioniq_0_best_model.pth`, **đúng 2 728 382 104 B** |
| sha256 local | `fea7fc08760dad1f5a66184ea7e65b45388f9781157cbd33a3b64db74103968a` — **hash LOCAL**, Kaggle không công bố checksum |
| Cấu trúc | torch-zip, 862 entry / 860 tensor, `visual_encoder.*` 511 + `Qformer.*` 336 |
| Env | `workbench/envs/tme`, Python 3.10.13, torch 2.1.2+cu121, torchvision 0.16.2+cu121 |
| GPU trong interpreter đó | RTX 3050 Laptop, `sm_86`, cuda 12.1, `allocated_tensor: true`, `computed: true` |
| Feature cache cần | **10.47 GiB** (15 536 ảnh × 257 × 1408 × 2) |
| EVA-ViT-g | 2 025 249 237 B, **bắt buộc** (constructor đọc sớm hơn BLIP-2) |
| Tokenizer | HF hub cache `models--bert-base-uncased`, **đã có** trên máy này |
| BLIP-2 pretrain | 746 998 955 B, **bắt buộc** (xem dưới) |
| 0% vs 20% | 0% **đã có**; 20% **`CHECKPOINT_UNAVAILABLE`** (tác giả chỉ phát hành `fashioniq-0.0-best`, `cirr-0.0-best`) |

**Chưa chạy end-to-end trên máy này**: 10.47 GiB + 2 025 249 237 B > dung lượng trống. Preflight báo
đúng cả hai kèm số byte và dung lượng trống tại đích — đây là kết quả **trung thực** của host này,
không phải một blocker giả.

## 6. Kiểm chứng

| Phép kiểm | Kết quả |
|---|---|
| Test R19 mới | **38 passed** (4 file) |
| `tests/unit` | **59 passed** |
| Toàn bộ suite `workbench/tests` | **1852 passed, 1 failed, 29 skipped = 1882** — khớp tuyệt đối `--collect-only`. Failure duy nhất là guard đếm test của chính trang docs, theo thiết kế fail trên chính revision nó mô tả. Số đo: một lượt single-process 39:32 cho mọi file trừ hai file chậm, cộng `test_release_blocker_gpu_records.py` + `test_verified_declarations.py` đo riêng (54 passed, 25:13). 4 shard song song trước đó cho 1919 passed / 6 failed, nhưng 10 case fail dưới shard đã pass trong lượt single-process ⇒ đó là tranh chấp tài nguyên, **không** tính là failure |
| Frontend production build | PASS; bundle chỉ còn **3 protocol**, id đã xoá **vắng mặt** |
| `pc.sh --self-check` | PASS |
| `acceptance.sh` | **ACCEPTANCE PASSED** (4 live check SKIP, không có server) |
| Offline release verifier | exit 0, **ENGINEERING_INCOMPLETE**, 5 PENDING |

Fixture và inference thật được tách rõ: 38 test R19 thay **duy nhất** phép tính neural đắt tiền bằng
fixture, còn source điều khiển, adapter, downloader, parser, registry và API được gọi **thật**. Phép
đo GPU nêu ở §5 là CUDA thật trong interpreter của model.

## 7. Lệnh trên PC

```sh
git fetch origin && git checkout 1848fb1      # revision đã kiểm thử
export WORKBENCH_LEDGER_KEY="$(cat ~/.config/cir/workbench_ledger_key)"

# 1) Asset/checkpoint: GitHub KHÔNG chứa dataset, checkpoint, venv hay base weight.
.venv/bin/python workbench/scripts/download_checkpoints.py --model limn --checkpoint base_iter0_dress \
  --checkpoint base_iter0_shirt --checkpoint base_iter0_toptee
export WORKBENCH_LIMN_PYTHON=$PWD/workbench/envs/limn-historical/bin/python

# 2) GPU đích, trong ĐÚNG interpreter của model
.venv/bin/python workbench/scripts/doctor.py --scope real --model limn --verify-only

# 3) Pilot
.venv/bin/python -m workbench.scripts.evaluate_models --model limn \
  --checkpoint base_iter0_all_categories --protocol fashioniq_val_split \
  --dataset-root <FashionIQ> --run-id <id>
```

**TME trên PC** (cần ≥10.47 GiB trống cho feature cache):

```sh
export WORKBENCH_TME_PYTHON=$PWD/workbench/envs/tme/bin/python
export WORKBENCH_TME_ASSET_EVA_VIT_G=/path/to/eva_vit_g.pth
mkdir -p workbench/third_party/TME/weight     # native đọc theo cwd của evaluator
workbench/envs/tme/bin/python workbench/replay/tme.py \
  --checkpoint-root workbench/artifacts/checkpoints/tme \
  --dataset-root <FashionIQ> --source-root workbench/third_party/TME \
  --checkpoint-id fashioniq_0 --preflight-only
```

**SPRC** (file tải tay):
`cp <OneDrive>.pth workbench/artifacts/checkpoints/sprc/sprc_fiq.pt` rồi
`.venv/bin/python workbench/scripts/download_checkpoints.py --model sprc --verify-only`.
Đặt đúng chỗ và hash local **không** chứng minh mapping của tác giả — cần kiểm schema/key/shape.

Chưa có bước nào được xác minh trên GPU đích. Laptop là `sm_86`; PC là `sm_120` (Blackwell) và **chưa
được đo**. Chỉ LIMN có block `adapted_runtime`; profile khai báo **không phải** phép tính CUDA thật.

## 8. Subagent

| Agent | Model / effort | Việc | Đóng góp |
|---|---|---|---|
| `TmeReplay` | `cx/gpt-6-luna` **xhigh** | R19-02/05/07 (TME) | phát hiện dtype native là **fp16** chứ không phải fp32 — sửa registry thay vì code quanh nó |
| `PthaWrapper` | `cx/gpt-6-luna` **xhigh** | R19-02/03 (PTHA) | sửa danh tính về registered id; test chạy subprocess thật |
| `ArchiveInstall` | `cx/gpt-6-luna` **xhigh** | R19-06 | extract + verify + publish atomic; ship kèm **một regression `import urllib.request` bị xoá** mà tôi phát hiện khi tải thật và đã sửa + yêu cầu test phủ đường thật |
| `AggregateParser` | `cx/gpt-6-luna` **xhigh** | R19-08 | tự phát hiện `EvaluationPlan` dùng field `.protocol` (không phải `.protocol_id`) và báo lại |
| `ProtocolDedup` | `cx/gpt-6-luna` **xhigh** | R19-10 | xác nhận cohort trùng nhau từ source; cập nhật schema/API theo registry |
| `ScienceReview` | `reviewer` `cx/gpt-6-luna` **xhigh** | nhãn khoa học, protocol, test bị nới | **bác** kết luận BLIP-2 của tôi (đúng), xác nhận cohort trùng nhau và xác nhận không test nào bị nới |
| `WiringReview` | `reviewer` `cx/gpt-6-luna` **xhigh** | wiring operator, fake readiness, asset thiếu | **5 finding CONFIRMED**: allowlist env, scope cache, path tokenizer, BLIP-2, `click` |

Không dùng Gemini, không dùng `ag/...`. Tôi tự kiểm từng kết luận, tự sửa 3 lỗi của chính subagent và
2 lỗi metadata của chính mình.

## 8b. Năm lỗi review độc lập tìm thêm — đã sửa

Hai reviewer (`reviewer`, `cx/gpt-6-luna` xhigh) đọc source đã đổi đối chiếu native code và trả **5 kết luận
CONFIRMED**. Mỗi cái đã sửa và commit riêng:

1. **Env var asset không tới được tiến trình con.** `_evaluator_environment` dựng env theo allowlist **cố
   định**, nên `WORKBENCH_TME_ASSET_EVA_VIT_G` resolve được ở parent nhưng tới child lại unset, và check asset
   của child chặn một run mà operator **đã** cấp input. Nay allowlist forward thêm tên do **registry khai**
   (`python_env_var` + `location_env_var` của từng asset bắt buộc).
2. **Feature cache khai sai scope.** Khai dưới dataset root trong khi native loader mở theo **cwd của
   evaluator**. Readiness kiểm path mà run không hề mở ⇒ input thiếu vẫn pass. Nay `preparation_paths`
   resolve theo `scope` đã khai (`fashioniq_root` / `evaluator_cwd`).
3. **Tokenizer trỏ sai chỗ.** `~/.cache/torch/hub/...` là path loader **không** dùng; `BertTokenizer.from_pretrained`
   đọc `HF_HOME` (mặc định `~/.cache/huggingface`), và đó là **thư mục**, không phải file. Nay khai
   `location_type: directory` + hub suffix; checker chấp nhận thư mục có nội dung.
4. **BLIP-2 BẮT BUỘC — kết luận cũ của chính tôi SAI.** `blip2_pretrain.yaml` đặt `load_finetuned: False`
   **và** khai `pretrained` URL ⇒ `load_checkpoint_from_config` đi nhánh `load_pretrained` và **thật sự
   gọi** `load_from_pretrained`. Phép kiểm thực nghiệm trước đó bị **đọc sai**: dựng model thiếu base weight
   chết **trước tiên** ở `weight/eva_vit_g.pth` vì `create_eva_vit_g` chạy sớm hơn. Chết ở EVA không chứng
   minh file BLIP-2 không dùng. Cả hai nay khai đúng path cwd-relative mà native mở.
5. **`click` thiếu trong env declaration.** LAVIS pinned import `spacy`, mà CLI của spacy làm
   `from click import NoSuchOption`, nên chặn click làm `import lavis.models` fail. Khai vì code cần, không
   vì requirements của tác giả có.

Reviewer khoa học cũng **bác** kết luận protocol của tôi: alias `fashioniq_val_split_ref_included` chỉ
resolve ở phía **evaluator**; một result document đã persist mang id cũ vẫn bị `RunMetadata` từ chối. Comment
đã sửa để nói đúng phạm vi đó thay vì nói quá rằng artefact cũ "đọc được".

Sửa xong: **193 test** trên các file liên quan + `tests/unit` pass.

## 9. Blocker ngoài

| Blocker | Bằng chứng | Việc nhỏ nhất tiếp theo |
|---|---|---|
| Đĩa cho TME | 10.47 GiB feature + 2 025 249 237 B EVA > trống | giải phóng ~13 GB rồi chạy preflight |
| PAIR | `model_vssp`/`MHTransformer_dis` vắng ở mọi revision công khai | tác giả công bố 2 file đó |
| TME 20% | tác giả chỉ công bố 0% | — (giữ `CHECKPOINT_UNAVAILABLE`) |
| Air-Know 20% | chỉ có 50%/80% | tác giả công bố 20% |
| sprC ViT-g | OneDrive 403 | tải tay bản ViT-L |
| Đĩa cho LIMN/HABIT trên PC | cache iLearn ~12 GB/lượt | dùng volume riêng cho `WORKBENCH_ARTIFACT_ROOT` |

Không blocker nào được mô tả như "đã hoàn thành".
