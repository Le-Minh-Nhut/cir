# Native runs bổ sung — `dcnet` và `clip4cir_rn50x4_fullft`

Tài liệu này chỉ ghi **hai** lượt chạy native FashionIQ do workstream này thực hiện. Sáu model
còn lại của danh sách 15 (`airknow`, `pair`, `clvc_net`, `tgcir`, và các target bị chặn khác) **không
bị đụng tới**. Cả hai lượt dưới đây là **reproduction native thật**: entrypoint gốc của upstream, trên
val split gốc của FashionIQ, dùng checkpoint đã có trong `checkpoint_selection/20261106T002617+0700/`.

Không có transfer, không có adapter, không có số liệu suy ra. Mọi con số dưới đây đọc trực tiếp từ
stdout của lần chạy được ghi lại trong `raw/native/*.stdout.log` và đối chiếu chéo với
`raw/native/*.json`.

---

## 1. `clip4cir_rn50x4_fullft` — HOÀN TẤT, EXIT 0

### 1.1 Lệnh chạy

Entrypoint: `workbench/third_party/CLIP4Cir/src/validate.py` (upstream commit
`dfed9f748a8a4d05abf613164e06d59b22dbdf13`, **không patch** — `git status` của upstream sạch).

```bash
cd /home/heheboiz/data/cir/workbench/third_party/CLIP4Cir
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
TMPDIR=/tmp/cir-native15-runs/clip4cir_fullft \
/home/heheboiz/data/cir/workbench/artifacts/environments/clip4cir-38/bin/python src/validate.py \
  --dataset fashionIQ \
  --combining-function combiner \
  --combiner-path /home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700/checkpoints/clip4cir/FashionIQ/RN50x4_fullft/fiq_comb_RN50x4_fullft.pt \
  --projection-dim 2560 \
  --hidden-dim 5120 \
  --clip-model-name RN50x4 \
  --clip-model-path /home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700/checkpoints/clip4cir/FashionIQ/RN50x4_fullft/fiq_clip_RN50x4_fullft.pt \
  --target-ratio 1.25 \
  --transform targetpad
```

Script đầy đủ: `raw/native/clip4cir_fullft.command.sh`.

Khác biệt duy nhất so với lượt `combiner_rn50x4_noft` đã hoàn tất (`run_id
20261108T064137301769Z_..._020a70b0`): thêm `--clip-model-path`, để tower CLIP đã fine-tune thay
cho RN50x4 stock. Mọi flag còn lại (`--projection-dim 2560`, `--hidden-dim 5120`,
`--target-ratio 1.25`, `--transform targetpad`, cùng `cwd`) giữ nguyên.

### 1.2 Kết quả (phần trăm, đúng như upstream in ra)

| Category | R@10 | R@50 |
|---|---|---|
| shirt | 44.50441598892212 | 65.4563307762146 |
| dress | 39.06792402267456 | 64.20426368713379 |
| toptee | 47.83273935317993 | 71.29015922546387 |
| **macro** | **43.8016931215922** | **66.98358456293742** |

Macro do upstream tự tính (`validate.py:351-352`, `statistics.mean` trên ba category) và
workstream này tính lại độc lập, hai giá trị khớp tuyệt đối.

**Đơn vị:** phần trăm. `validate.py::compute_fiq_val_metrics` nhân `* 100` trước khi trả về, nên
`44.504…` nghĩa là 44.50 %. Khác `dcnet` ở §2 (dùng phân số 0–1) — không được trộn hai thang.

### 1.3 Số query — suy ra từ annotation, không hard-code

| Category | Query (triplets) | Gallery ảnh |
|---|---|---|
| dress | 2017 | 3817 |
| shirt | 2038 | 6346 |
| toptee | 1961 | 5373 |
| **tổng** | **6016** | 15536 |

Nguồn: `prepared/clip4cir/fashionIQ_dataset/captions/cap.{category}.val.json` (số triplet) và
`…/image_splits/split.{category}.val.json` (số ảnh gallery), đọc bằng Python trong chính
workstream này. 6016 tình cờ bằng con số thường được trích, nhưng ở đây nó là **kết quả đo**, không
phải giả định.

Về con số gallery: `15536` là **tổng số entry index trên ba category** (mỗi category dựng index
riêng trên `split.{category}.val.json`), không phải số ảnh unique. Đo lại trên đúng các file split:
`3817 + 6346 + 5373 = 15536` entry, trong đó **15415 ảnh unique** (dress và shirt/toptee dùng chung
một số ảnh). Cả 15415 ảnh đều tồn tại trong `prepared/…/fashionIQ_dataset/images/` (0 file thiếu),
và đúng bằng số file trong `data/FashionIQ/resized_images/`. Vì vậy cột "Gallery ảnh" ở bảng trên là
**số entry index mỗi category**, không cộng dồn thành gallery chung.

### 1.4 Tài nguyên và thời gian

| Hạng mục | Giá trị |
|---|---|
| Interpreter | `/home/heheboiz/data/cir/workbench/artifacts/environments/clip4cir-38/bin/python` |
| Python / torch | 3.8.20 / 1.11.0+cpu |
| `torch.cuda.is_available()` | `False` |
| Thiết bị thực tế | cpu |
| Wall time | **2246 s** (37:26) |
| User / sys CPU | 6558.38 s / 1860.50 s |
| Song song hiệu dụng | 374 % |
| Peak RSS | 3139724 KiB ≈ **2.99 GiB** |
| Exit code | **0** |
| stderr | 40425 B, chỉ là thanh tiến trình tqdm — **không có traceback** |

Thời gian khớp với ước lượng đặt ra trước khi chạy (sibling `combiner_rn50x4_noft` khoảng 39 phút
theo artifact của chính nó; lượt này 37:26).

### 1.5 Checkpoint

| File | Size (B) | SHA-256 (workstream này tự tính) |
|---|---|---|
| `fiq_comb_RN50x4_fullft.pt` | 236017921 | `7c7c0a10fb4c995cb1ded9ad70a58891d50c622263c54131523bfb232b493f8e` |
| `fiq_clip_RN50x4_fullft.pt` | 713874695 | `4111ac139cf96ac0cf31b30bf8f55b654247b06f0f5fbee4fe8dcc8c1c7cfd2f` |

Cả hai digest khớp bản ghi trước đó ở
`workbench/docs/reviews/20261108T113348+0700-native-scope/checkpoint_sha256.tsv` (dòng 2 và 3) —
tức bundle không bị đổi giữa hai lần đo.

Bundle load trong chính lượt chạy này: stdout có `CLIP model loaded successfully` ngay sau
`Trying to load the CLIP model`, và `combiner.load_state_dict(...["Combiner"])` không ném lỗi. Đó là
load `strict=True` trên cả hai file.

### 1.6 Tác dụng phụ

Không. `validate.py` chỉ in ra stdout; không ghi file, không sửa checkpoint, không tạo artifact nào
trong `checkpoint_selection/`.

### 1.7 Vì sao trước đây bị coi là chưa chạy

Registry khai một bundle `rn50x4-fullft-unresolved.pt` cần `combiner_state.pt` **và** `clip_state.pt`,
trong khi đĩa thực tế có `fiq_comb_RN50x4_fullft.pt` + `fiq_clip_RN50x4_fullft.pt`. Đó là **ánh xạ
tên**, không phải thiếu checkpoint: đúng cặp file tồn tại, load được, và cho ra metric ở trên. Không
có blocker kỹ thuật nào bị vượt qua ở đây — chỉ là chưa từng có lượt chạy eval đầy đủ.

---

## 2. `dcnet` — HOÀN TẤT, EXIT 0

### 2.1 Lệnh chạy

Entrypoint: `workbench/third_party/DCNet/test.py` (upstream `https://github.com/ozmig77/dcnet`,
commit `68c79d38569f01ca39c3b4ea641b3b7eb2ccb5fb`).

```bash
cd /home/heheboiz/data/cir/workbench/third_party/DCNet
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
TMPDIR=/tmp/cir-native15-runs/dcnet \
/home/heheboiz/data/cir/workbench/artifacts/environments/dcnet-377/bin/python \
  test.py --resume /home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700/checkpoints/dcnet/ \
  --mode val
```

Script đầy đủ: `raw/native/dcnet.command.sh`.

`--resume` là tham số duy nhất `test.py` cần (`test.py:100-103`); `--mode val` là mặc định nhưng
được ghi tường minh để cố định ngữ nghĩa (nhánh đó in metric ra log, nhánh `test` thì ghi
`test_score.pt`).

**Patch nguồn duy nhất liên quan:** `workbench/third_party/DCNet/base/base_trainer.py::_resume_checkpoint`,
đã có sẵn từ trước workstream này — `map_location=self.device`, và `weights_only=False` chỉ được
truyền khi `'weights_only' in inspect.signature(torch.load).parameters`. Không sửa file nào khác
trong `third_party/DCNet`. Không có timeout: lượt trước bị cắt ở 2400 s (EXIT=124), lần này chạy
đến hết.

### 2.2 Kết quả

`trainer_joint.py::_valid_epoch` in ra sáu cột cho mỗi category: R@10/R@50 của nhánh `comp`,
R@10/R@50 của nhánh `corr`, rồi `comb`. Cột dùng làm kết quả cuối là **`comb`** (hợp
`log_softmax(comp) + log_softmax(corr)`), vì đó là cột mà checkpoint được chọn theo
(`trainer.monitor = "max comb_avg"`).

| Category | comp R@10 | comp R@50 | corr R@10 | corr R@50 | **comb R@10** | **comb R@50** |
|---|---|---|---|---|---|---|
| dress | 0.28011898859692613 | 0.5349529003470501 | 0.28805156172533464 | 0.5493306891422906 | **0.2915220624690134** | **0.5458601883986118** |
| shirt | 0.2198233562315996 | 0.46270853778213933 | 0.22718351324828262 | 0.471540726202159 | **0.2281648675171737** | **0.4764474975466143** |
| toptee | 0.3003569607343192 | 0.5803161652218256 | 0.31463539010708824 | 0.5818459969403366 | **0.31820499745028047** | **0.590515043345232** |
| **macro (comb)** | | | | | **0.27929730914548917** | **0.5376075764301528** |

Tức **27.93 % / 53.76 %** nếu quy về phần trăm. Thang của `dcnet` là **phân số 0–1**, không phải
phần trăm — ngược với `clip4cir` ở §1. Ghi sai thang là sai gấp trăm lần.

Ba dòng `upstream_reported` cũng được lưu trong JSON:

| Khoá upstream | Giá trị | Nghĩa thật |
|---|---|---|
| `recall_avg` | 0.3963794848189767 | mean của **sáu** ô `recall[i][:2]` (comp+corr, cả 3 category) |
| `recall_avg_corr` | 0.4054313128942486 | mean của **sáu** ô `recall[i][2:]` (corr) |
| `comb_avg` | 0.40845244278782095 | mean của **sáu** ô `comb` (3 category × 2 metric) |

**Không** dùng ba số này làm macro: `comb_avg = 0.4085` là mean của R@10 **và** R@50 gộp lại, không
phải macro R@50. Macro đúng theo từng metric là 0.27930 / 0.53761 như bảng trên. Đây là chỗ rất dễ
trích sai từ `info.log`.

Không có R@1: `_valid_epoch` chỉ tính R@10/R@50 cho FashionIQ, nên `r1` để `null` thay vì bịa.

### 2.3 Số query — suy ra từ annotation, và đối chiếu bằng artifact của chính lần chạy

| Category | Query | Gallery |
|---|---|---|
| dress | 2017 | 3817 |
| shirt | 2038 | 6346 |
| toptee | 1961 | 5373 |
| **tổng** | **6016** | |

Nguồn query: `data/FashionIQ/captions/cap.{category}.glove.val.pkl` (số phần tử = 2017 / 2038 / 1961),
trùng khớp với `cap.{category}.val.json` cùng thư mục. Đây là file mà `CE_dataset.py` thực sự đọc
(`text_feat == "glove"`), nên nó là nguồn đúng hơn JSON.

Đối chiếu độc lập: `att_result.pt` do chính lượt này ghi ra, giải nén bằng chính interpreter của
`dcnet-377`:

| Category | Số batch attention | Tổng query trong batch |
|---|---|---|
| dress | 253 | 2017 |
| shirt | 255 | 2038 |
| toptee | 246 | 1961 |

Khớp tuyệt đối với số suy từ annotation, nên không có query nào bị rơi và gallery được dựng đủ.

### 2.4 `att_result.pt` — đã bị ghi lại

Đây là điểm cần ghi rõ, vì file này từng chứa metric của một lượt **cũ**:

| | Giá trị |
|---|---|
| Trước lượt này | mtime `2026-11-07 00:02:45.026598380 +0700`, 5442051 B |
| Sau lượt này | mtime **`2026-11-10 04:22:45.358250989 +0700`**, 5340335 B |
| SHA-256 mới | `6d776ffad3168e3f62105e07edfb954eb7ff320206cbd62a79c061db25160438` |

Vậy file **đã bị ghi lại** bởi lần chạy này. Metric ở §2.2 là của lượt này; metric cũ 2026-11-07
**không** được dùng ở bất kỳ đâu trong báo cáo này.

`val/info.log` cũng bị append (lượt này ghi từ `2026-11-10 02:15:17` đến `04:22:45`) và
`val/config.json` bị `ConfigParser` ghi lại với nội dung không đổi — hành vi gốc của upstream.

### 2.5 Kiểm tra thứ tự — so với lượt CUDA 2026-11-07

Chỉ để xác nhận lần chạy này không sinh ra số vô lý, **không** phải nguồn metric:

| Category | comb R@10 cũ (CUDA) | comb R@10 lần này (CPU) | comb R@50 cũ | comb R@50 lần này |
|---|---|---|---|---|
| dress | 0.2915220624690134 | 0.2915220624690134 | 0.5453644025780863 | 0.5458601883986118 |
| shirt | 0.2281648675171737 | 0.2281648675171737 | 0.4764474975466143 | 0.4764474975466143 |
| toptee | 0.31820499745028047 | 0.31820499745028047 | 0.5900050994390617 | 0.590515043345232 |

Bốn trong sáu ô trùng khít từng bit; hai ô R@50 lệch ở chữ số thập phân thứ tư (≤ 5.5e-4). Đó là
hành vi bình thường khi đổi đường tính float giữa CUDA và CPU ở vài hạng tử biên. Không ô nào lệch
đủ để đổi kết luận.

### 2.6 Tài nguyên và thời gian

| Hạng mục | Giá trị |
|---|---|
| Interpreter | `/home/heheboiz/data/cir/workbench/artifacts/environments/dcnet-377/bin/python` |
| Python / torch | 3.7.7 / 1.4.0+cpu |
| `torch.cuda.is_available()` | `False` |
| Thiết bị thực tế | cpu (`Running test on cpu`) |
| Wall time | **7656 s** = 2:07:35 |
| User CPU | 27975.41 s (7.77 h CPU) |
| Song song hiệu dụng | 376 % |
| Peak RSS | 13082084 KiB ≈ **12.48 GiB** |
| Exit code | **0** |
| stderr | 0 B — hoàn toàn trống |

Ước lượng đưa ra **trước khi chạy** là ~2.4 h (đo `FusDiff` 7 modality trên N=3817/6346/5373, cộng
ResNet-50 forward 15536 ảnh ≈ 6.5 phút); thực tế 2.13 h. Không vượt mốc 3 h nên không phải báo
trước khi chạy target 2, và thứ tự chạy (clip4cir trước, dcnet sau, không song song) vẫn giữ.

Nút thắt là `model.py::get_correction`, chạy vòng `for bi in range(B)` trên từng query với
`N × D` phần tử mỗi modality — chi phí tuyến tính theo số query chứ không vector hoá. Đó là lý do
12.5 GiB RSS và 2 giờ CPU, không phải do tràn bộ nhớ.

---

## 3. Tổng hợp

| Model | Trạng thái | Macro R@10 | Macro R@50 | Đơn vị | Wall | Exit |
|---|---|---|---|---|---|---|
| `clip4cir_rn50x4_fullft` | HOÀN TẤT | 43.8017 | 66.9836 | phần trăm | 2246 s | 0 |
| `dcnet` | HOÀN TẤT | 27.9297 | 53.7608 | phân số → 27.93 / 53.76 % | 7656 s | 0 |

Cả hai là native reproduction trên `fashioniq_original_split` / val split gốc, không có transfer,
không có adapter. Số model native đã hoàn tất tăng từ 9 lên **11 trên 15** mục tiêu active; sáu
model bị chặn thật (`airknow`, `pair`, `clvc_net`, `tgcir`, và hai target khác) **không** bị đụng
tới trong workstream này.

Không có lượt nào thất bại, nên không có lý do lỗi verbatim nào cần ghi.

### 3.1 Kỷ luật tài nguyên đã giữ

- `nvidia-smi` và `pgrep -af "python|pytest"` kiểm tra trước mỗi lần chạy: không có eval nào khác
  đang chạy; GPU ở 741 MiB (chỉ desktop), hai lượt đều CPU.
- Tổng thread ≤ 10: mỗi lượt export `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
  NUMEXPR_NUM_THREADS=4`, và hai lượt **không** chạy đồng thời.
- Đĩa: 157 GiB trống trước khi chạy (yêu cầu ≥ 25), 157 GiB sau khi dọn — không tăng.
- `TMPDIR` = `/tmp/cir-native15-runs/{clip4cir_fullft,dcnet}`, mỗi script `rm -rf` chính nó sau khi
  xong; thư mục cha `/tmp/cir-native15-runs` đã được `rmdir` và xác nhận không còn tồn tại.
- Không có `OSError: [Errno 28]` ở bất kỳ lượt nào.
- `workbench/config/workbench.env` không bị sửa; không có `git reset`/`clean`/force-push.

### 3.2 File đã ghi

| File | Nội dung |
|---|---|
| `raw/native/dcnet.json` | metric 3 category + sáu cột gốc, macro, checkpoint SHA, lệnh, runtime, side-effect `att_result.pt` |
| `raw/native/clip4cir_fullft.json` | metric 3 category, macro, checkpoint SHA, lệnh, runtime |
| `raw/native/dcnet.command.sh` | script chạy lại được, đường dẫn tuyệt đối |
| `raw/native/clip4cir_fullft.command.sh` | script chạy lại được, đường dẫn tuyệt đối |
| `raw/native/dcnet.stdout.log` | stdout gốc (6.2 KiB) |
| `raw/native/dcnet.stderr.log` | trống (0 B) |
| `raw/native/dcnet.time.log` | `/usr/bin/time -v` |
| `raw/native/clip4cir_fullft.stdout.log` | stdout gốc (1.2 KiB) |
| `raw/native/clip4cir_fullft.stderr.log` | chỉ thanh tiến trình tqdm (40.4 KiB) |
| `raw/native/clip4cir_fullft.time.log` | `/usr/bin/time -v` |
| `NATIVE_RUNS_EXTRA.vi.md` | tài liệu này |

Không sửa `FINAL_REPORT.vi.md`, `UNFINISHED.vi.md`, `NATIVE_EVAL_MATRIX.csv`, `COMPOSITION_MATRIX.csv`,
`GATES.csv`, `EVIDENCE_INDEX.md` hay bất kỳ file nào khác ngoài danh sách trên.
