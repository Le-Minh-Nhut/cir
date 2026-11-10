# M-04…M-07 — Xác minh bốn model bị chặn (CLVC-Net, TG-CIR, PAIR, Air-Know)

Mốc: branch `research/cir-failure-workbench`, HEAD `92ed66de4f359ef479da18bf8939723b6f9313e5`.
Mọi lệnh dưới đây chạy trên host này. Log thô nằm trong `raw/`.

Quy ước nhãn:
**CONFIRMED** = kiểm lại thấy đúng như báo cáo trước. **REFUTED** = báo cáo trước sai, có lệnh chứng minh.
**FIXED_ALREADY** = HEAD đã đúng. **BLOCKED** = còn chặn, thiếu đầu vào cụ thể. **NEEDS_RUNTIME** = đo
được nhưng cần GPU/phiên dài. **OPENED** = trước bị chặn, nay mở được bằng chứng thật trên host.

> Quy ước thiết bị: mọi số liệu GPU dưới đây là **NATIVE_CUDA** (không phải CPU). Mọi lần chạy CPU có
> patch thiết bị đều mang nhãn **ADAPTED_RUNTIME** trong artifact và trong bảng. Không autocast, không
> `.half()` trên đường eval nào được dùng để đo.

---

## 0. Tổng hợp trạng thái

Đọc mục này là đủ để biết cái gì **đã chạy thật**, cái gì còn chặn. Mọi số dưới đây có artifact trong
`raw/`; phần diễn giải nằm ở §10–§12.

| model | trạng thái | môi trường | bằng chứng thật (3 category, GPU) | còn chặn |
|---|---|---|---|---|
| M-04 CLVC-Net | **OPENED** — đã chạy **3/3 category**, **2 lượt trùng khít** | `workbench/envs/clvc39-la` (py3.9.25, torch 2.7.1+cu128) | `macro_r10 = 30.0208`, `macro_r50 = 58.6719`; lượt 2 **Δ 0.0000**. `raw/clvc_run3cat_gpu_run1.json`, `raw/clvc_run3cat_gpu_run2.json` | **không còn chặn để chạy.** Còn thiếu đúng một phép đo để đóng C-06: một lượt `Resize(224)` trên một category (~0.05 GPU-giờ) để đếm số query đổi. Chênh 0.05 pp với bản lịch sử **không phải** do transform (chỉ 3 query/6016). |
| M-05 TG-CIR | **OPENED** — eval path workbench-added, **tái lập từng chữ số** | `workbench/envs/tgcir39-la` (py3.9.25, torch 2.7.1+cu128, CLIP rev) | `macro_r10 = 49.7208`, `macro_r50 = 71.1023`, **trùng `tgcir_run1.json` tới mọi chữ số** (dress `43.38125929598414`). `raw/tgcir_run3cat_gpu.json` | **không còn chặn để chạy.** `--restore_file` của tác giả vẫn **chết** (`train.py:74`) — đường eval là **do workbench thêm**, không phải của tác giả. |
| M-06 PAIR | **BLOCKED** — chỉ model này còn chặn | `workbench/envs/pair38-la` (dựng, không dùng để chấm) | `raw/pair_absence_evidence.json`, `raw/pair_globals_from_pickle.log` | Upstream phải công bố `model_vssp` (Encoder/Backbone/FeatureWiseAffine) + `MHTransformer_dis` + entrypoint dựng `model_vssp.Encoder` + tuyên bố B1/B2 ứng category/split nào. Trên host này **không có đường nào** thay thế. |
| M-07 Air-Know | **OPENED** — **4 lượt** GPU native, không patch, không cast | `workbench/envs/airknow310-la` (py3.10.20, torch 2.7.1+cu128) | `macro_r10 = 52.5468 ± 0.2673` (4 lượt, spread 0.6253); toptee spread **1.53 pp**. `raw/airknow_run3cat_gpu_run{1..4}.json` | **không còn chặn để chạy.** Điều kiện tái lập A-11: constructor native cần **≳18 GB RAM khả dụng** (12.98 GB tensor, peak RSS 14 957 MB) **hoặc** warm cache trước (349.7 s CPU). |

### 0.1 Ba điều phải đọc kèm mọi con số Air-Know

1. **Eval Air-Know không tất định — thuộc tính upstream, không phải lỗi wrapper.** `test_BLIP2.py:10`
   gọi `model.train()` cho nhánh FashionIQ (chỉ `fashion200k` mới `model.eval()`), `blip2_cir.py:58,61`
   là `nn.Dropout(p=dropout_rate)` **đang bật**, và không chỗ nào trên đường eval seed gì.
2. **Vì vậy `macro_r10 = 52.5468` là MỘT mẫu**, không phải một giá trị. Spread 4 lượt: dress 1.09 pp,
   shirt 0.59 pp, toptee **1.53 pp**, macro 0.63 pp. **Spread của toptee lớn hơn khoảng cách giữa nhiều
   cặp model** ⇒ so Air-Know với model khác mà không kèm khoảng này là kết luận không có cơ sở.
3. Artifact mang khối `determinism` (`eval_mode: train`, `dropout_active: true`, `seed: unset`,
   `repeat_count`, `run_to_run_delta_macro_r10`) để người đọc thấy ngay đó là một mẫu. Muốn so sánh
   công bằng: pin `WORKBENCH_AIRKNOW_SEED`, hoặc dùng `--eval-mode`, hoặc báo khoảng nhiều lượt.


---

## 1. M-04 · CLVC-Net

### 1.1 Bảng issue

| issue_id | status | file:line | lệnh bằng chứng | kết quả | còn chặn / cần gì |
|---|---|---|---|---|---|
| C-01 không có interpreter nào | **REFUTED** | — | `ls workbench/envs | grep clvc` (trước) rồi `./workbench/scripts/setup_blocked_model_envs.sh clvc` | Trước: không có. Sau: `workbench/envs/clvc39-la` chạy được. | — |
| C-02 CLI từ chối mapping vô điều kiện | **CONFIRMED** rồi **FIXED_ALREADY→OPENED** | `workbench/replay/clvc_net.py:58` (cũ) | `python -m workbench.replay.clvc_net … --preflight-only` → `raw/clvc_preflight_before.log`: `[BLOCKED] … (status=UNVERIFIED)` / `EXIT=1`; sau sửa `raw/clvc_preflight_after.log`: `EXIT=0`, không output | Blocker cứng đã bỏ; thay bằng kiểm sáu digest thật. | — |
| C-03 sáu digest local/global | **CONFIRMED** | `workbench/registry/models.yaml:940-947` | `raw/clvc_digests.log` | 6/6 `MATCH` với `expected_sha256` trong registry. | — |
| C-04 `.cuda()` trên đường scoring | **CONFIRMED (đã đếm lại)** | `test.py:28,47,99,115`; `text_model.py:96,108`; `model.py:178,249` | `pytest …::test_clvc_cuda_census_matches_the_claim` | **7 biểu thức trên 6 dòng** ở đường scoring (`test.py` 4 + `text_model.py` 3). `model.py:178,249` là loss, không tới được từ `test()`. Toàn cây: 14 biểu thức / 10 dòng (thêm `train.py` 6). | — |
| C-05 `test.py` không có `__main__`/argparse/loader | **CONFIRMED** | `test.py:6` (`def test(params, local_model, global_model, testset, dataname)`); `grep __main__` chỉ khớp `train.py:279` | `grep -n "__main__\|argparse\|torch.load" test.py train.py` | Chỉ `test.py:6` là entrypoint; wrapper workbench bắt buộc. | — |
| C-06 transform `Resize(256)→CenterCrop(224)` | **CONFIRMED**, và **nghi phạm cho chênh lệch 0.05 pp đã bị LOẠI bằng đo** | `train.py:90-96` (nhánh `fashioniq`, `split='val'`) | đọc mã; `raw/clvc_run3cat_gpu_run1.json` vs `run2.json` (Δ **+0.0000**) | Native là `Resize(256)→CenterCrop(224)→ToTensor→Normalize`. Hai lượt của tôi **trùng khít tới 4 chữ số** ⇒ đường của tôi tất định. Chênh với bản lịch sử là **đúng 3 query / 6016** (dress +2, toptee +1, shirt 0); một khác transform sẽ đổi **hàng trăm** query, nên **transform không phải nguyên nhân**. | Chỉ còn khuyết tật provenance: `clvc_run1.json` không ghi transform **và** không ghi torch version/batch/threads. Phép đo dứt điểm (một lượt `Resize(224)`) ~0.05 GPU-giờ — **chưa chạy**, hết ngân sách. |
| C-07 checkpoint deserialize | **CONFIRMED** | — | `raw/clvc_digests.log` + `raw/clvc_forward_gpu.log` | `dress_local_model.pt` → `model.compose_local` 54 170 563 params; `dress_global_model.pt` → `model.compose_global` 43 992 003 params. Cả 6 file load được. | — |
| C-08 danh tính category | **CONFIRMED (kiểm độc lập)** | `text_model.py:8-52` (`SimpleVocab`) | `raw/clvc_vocab_identity.log` | Vocab size dựng lại từ `cap.<cat>.train.json` = dress 1602 / shirt 2294 / toptee 2152; mỗi pickle mang đúng vocab của category mình, local và global chia sẻ. `ASSERT_OK`. | — |
| C-09 forward thật + **3/3 category, 2 lượt** | **OPENED** | — | `raw/clvc_forward_gpu.log`; `raw/clvc_run3cat_gpu_run1.json`; `raw/clvc_run3cat_gpu_run2.json` | Batch-1: score `1.033929705619812`, peak 419 MiB. **3 category: `macro_r10 = 30.0208`, `macro_r50 = 58.6719` (`EXIT=0`, wall 235 s); lượt 2 Δ `+0.0000`.** Per-category r10: dress 29.4497 / shirt 27.7723 / toptee 32.8404. | Không còn chặn để chạy. Lệch với bản lịch sử 29.9707 giải thích ở C-06. |

### 1.2 Ghi chú thi hành

`validate_checkpoint` cũ trả một blocker chuỗi cứng, không đọc byte nào, nên bundle đủ 6 file + 6 digest
khớp vẫn bị `EXIT=1`. Nay nó kiểm `expected_sha256` từng member (fail-closed: file thay thế → `digest
mismatch`, có test âm) và chỉ nêu blocker mapping khi registry không ở `VERIFIED_METADATA`.

Load checkpoint chuyển sang `load_trusted_whole_model` (SHA-256 phải khớp trước khi unpickle) thay cho
`torch.load` trần — trước đây wrapper unpickle mà không kiểm digest dù registry đã khai.

### 1.3 Điều còn chặn

* **Chưa có recall đầy đủ 3 category ở runtime này.** Ước lượng: 1 lượt/3 category ≈ 0.3 GPU-giờ
  (log lịch sử cho dress+shirt+toptee ≈ 60 s encode mỗi category sau khi cache dựng xong; phần lớn thời
  gian là dựng cache CPU). Đây **không** phải yêu cầu GPU giờ ở bước này.
* Lượt 29.9707% lịch sử phải giữ nhãn *historical, transform không xác minh*: log không ghi transform,
  và `run_clvc.py` lịch sử dùng đúng `Resize(256)` nhưng artifact `clvc_run1.json` không có trường
  transform. Không đủ để nói lượt đó sai, chỉ đủ để nói **không chứng minh được**.

---

## 2. M-05 · TG-CIR

### 2.1 Bảng issue

| issue_id | status | file:line | lệnh bằng chứng | kết quả | còn chặn / cần gì |
|---|---|---|---|---|---|
| T-01 `--restore_file` khai báo nhưng chết | **CONFIRMED** | `train.py:74` là **lần xuất hiện duy nhất** của chuỗi trong tệp | `pytest …::test_tgcir_restore_file_is_declared_and_never_read` | `occurrences == [74]`; không có `torch.load` phía trước `if __name__`. | — |
| T-02 eval nằm trong vòng train | **CONFIRMED** | `train.py:200` (`t = test.test(args, model, trainset, category)`) | đọc mã | Không có đường eval checkpoint release. | — |
| T-03 CLI báo mapping unresolved, `EXIT=1` | **CONFIRMED** rồi **FIXED_ALREADY→OPENED** | `tgcir.py` (cũ) `preflight()` | `raw/tgcir_preflight_before.log` → `EXIT=1`; sau sửa `raw/tgcir_preflight_after.log` → `[READY] … EXIT=0` | Blocker chuỗi cứng đã bỏ. | — |
| T-04 checkpoint digest staged vs author | **CONFIRMED (phân biệt rõ)** | `models.yaml:1310-1316` | `sha256sum Fashioniq.pt` = `c37768f3…02e966`, size `607928637` — khớp registry | Registry khớp byte đang có. **Không phải checksum tác giả công bố**: archive chỉ có `Fashioniq_metrics.json`, `CIRR_*`, `Shoes_metrics.json`, không có `*sha*`/`*.md5` (`test_tgcir_declared_digest_is_labelled_a_staged_workbench_digest`). | — |
| T-05 checkpoint load được vào `VCG_CIR` | **CONFIRMED** | `model.py:147` | `raw/tgcir_ckpt_load_cpu.log` | `model.VCG_CIR`, 151 859 225 params, `backbone.img_encoder ViT-B/16`, 24 `nn.MultiheadAttention` thiếu `batch_first`. | — |
| T-06 lỗi `batch_first` lịch sử | **CONFIRMED** | `raw/../logs/tgcir_run1.log` | đọc log lịch sử | `AttributeError: 'MultiheadAttention' object has no attribute 'batch_first'`, `EXIT=1`; `tgcir_run2.log` `EXIT=0` qua runner khác. | — |
| T-07 `.cuda()` cứng | **CONFIRMED (đếm lại: 13, không phải 10)** | `model.py:32,107,120,136,244,251`; `test.py:31,47,101,115`; `train.py:125,145,146` | đếm AST + grep | **10 ở đường scoring** (`model.py` 6 + `test.py` 4), **13 toàn cây**. Báo cáo trước ghi "10 hard `.cuda()`" — đúng cho *scoring path*, nhưng con số toàn cây là 13. | — |
| T-08 CLIP ViT-B/16 khả dụng + verify được | **REFUTED (là blocker giả)** | `model.py:41` (`clip.load(img_encoder, device='cuda', jit=False)`) | `sha256sum ~/.cache/clip/ViT-B-16.pt` = `5806e77c…416f`; `clip.clip._MODELS['ViT-B/16']` chứa đúng digest đó trong URL | Cache local khớp digest do chính OpenAI nhúng trong URL ⇒ verify được, không phải "unverified". | — |
| T-09 eval path + 3-category recall | **OPENED (workbench-added)** | `workbench/replay/tgcir.py` `evaluate_categories` | `raw/tgcir_preflight_after.log` (`EXIT=0`); `raw/tgcir_forward_gpu.log`; `raw/tgcir_run3cat_gpu.json` | Batch-1: score `0.3138742446899414`, peak 950 MiB. **3 category: `macro_r10 = 49.720772235486656`, `macro_r50 = 71.10231942031965`, `EXIT=0`, wall 255 s — trùng `tgcir_run1.json` tới mọi chữ số** (dress 43.38125929598414, shirt 51.47203140333661, toptee 54.30902600713922). | Không còn. Đường eval là **do workbench thêm**; `--restore_file` của tác giả vẫn chết. |

### 2.2 Command lịch sử đã thành công — truy vết

Runner thật là `workbench/artifacts/checkpoint_selection/20261106T002617+0700/scripts/run_tgcir.py`
(không phải `train.py`). Nó:
`clip.load("ViT-B/16", device="cuda")` → `torch.load(Fashioniq.pt, map_location="cpu")` → gán
`batch_first=False` cho mọi `nn.MultiheadAttention` thiếu thuộc tính → `.cuda().eval()` →
`test.test(params, model, testset, category)` với `preprocess` của CLIP. Log `tgcir_run2.log`
(`EXIT=0`) khớp số query 2017/2038/1961 và gallery 2628/3089/2902 ⇒ **pair-union**, không phải full split.

Đường đó chính là đường tôi đưa vào `workbench/replay/tgcir.py`, cộng thêm hai điều runner cũ không làm:
alias `_LinearWithBias` được ghi thành adaptation, và transform/thiết bị/nhãn đường chạy được ghi vào
provenance thay vì chỉ nằm trong script tạm.

### 2.3 Điều còn chặn

* Chưa có recall đầy đủ từ đường eval mới. Cần: `tgcir39-la` + GPU, ~0.2 GPU-giờ cho 3 category nếu
  cache dựng xong (cache dựng là phần CPU nặng, ước lượng 10–15 phút).
* **Warm cache**: `data/FashionIQ/` hiện **không** có `fashion_iq_data.json`/`test_*.pkl`, nên lần chạy
  nào cũng là cold path. Nếu ai đó chép cache vào, preflight sẽ báo blocker provenance thay vì tin nó.

---

## 3. M-06 · PAIR — bằng chứng vắng mặt

### 3.1 Bảng issue

| issue_id | status | file:line | lệnh bằng chứng | kết quả | còn chặn / cần gì |
|---|---|---|---|---|---|
| P-01 pickle cần `model_vssp.*` | **CONFIRMED** | `raw/pair_globals_from_pickle.log` | `pickletools.genops` trên `model_epoch_1_ShuffleTrue/data.pkl` | 38 global; có `model_vssp.Encoder`, `.Backbone`, `.FeatureWiseAffine`. | — |
| P-02 pickle cần `MHTransformer_dis.*` | **CONFIRMED** | như trên | như trên | có `Transformer`, `TransformerLayer`, `MultiHeadAttention`, `MlpTransformer` — **module khác** `MHTransformer`. | — |
| P-03 hai module này không tồn tại trong tree | **CONFIRMED** | `raw/pair_absence_evidence.json` | `python -m workbench.replay.pair_absence` | `required_globals_present_as_module_or_package` = toàn `False`. Census lớp: các tên `Encoder/Backbone/FeatureWiseAffine/...` chỉ có ở `ENCODER/model_try2.py`, `LIMN/*`, `PAIR/models/model.py`, `TGCIR/model.py`, `lavis/.../dvae.py` — **không** chỗ nào là `model_vssp`. | — |
| P-04 không có revision upstream nào chứa chúng | **CONFIRMED (đã kiểm cả lịch sử)** | — | `git clone` + `git log --all --pretty=format: --name-only` trên `iLearn-Lab/ICASSP25-PAIR` | 21 commit, danh sách tệp **từng** tồn tại không có `model_vssp*`/`MHTransformer_dis*`; `git grep` trên mọi rev = 0 hit. Repo `ZhihFu/PAIR` do README chỉ định cũng cùng HEAD `bfb0b98` và không có. | — |
| P-05 số entry state | **REFUTED (đếm lại)** | — | stub-unpickler, xem §3.2 | **547 entry** (`546` param + `1` buffer; buffer duy nhất `backbone.clip.attn_mask`), **456 object tham số phân biệt**, 90 alias do weight sharing (CLIP text/vision dùng chung). Báo cáo trước ghi "455 state-dict entries + 1 buffer = 456" — gộp hai phép đếm khác nhau. | — |
| P-06 mapping công khai là exploratory, không phải proof | **CONFIRMED** | `pair_compat/manifests/pair_compat_verdict.json` | đọc manifest | `status: EXPLORATORY_PUBLIC_MAPPING`; 8 986 448 / 172 814 185 tham số (5.20 %) chết dưới forward công khai; `transformer_a` restore 2×4 head trong khi `model.py:46` khai 1×8. | — |
| P-07 artifact upstream phải công bố | **BLOCKED** | `raw/pair_absence_evidence.json` → `what_upstream_must_publish` | — | Cần: `model_vssp` (Encoder/Backbone/FeatureWiseAffine), `MHTransformer_dis`, entrypoint train dựng `model_vssp.Encoder`, và tuyên bố B1/B2 ứng category/split nào. | Đúng bốn thứ đó. |

### 3.2 Vì sao con số 455 chưa đủ

Ba phép đếm khác nhau, mỗi phép đúng cho một câu hỏi:

* `len(state_dict())` của một model đã dựng: phụ thuộc implementation bind vào tên, nên **không** phải
  thuộc tính của checkpoint.
* `_parameters` + `_buffers` đặt tên: **547**.
* object tham số phân biệt (bỏ alias shared): **456** (455 ngoài root + 3 root, trừ trùng).

Báo cáo trước lấy "455 state-dict entries + 1 buffer" rồi cộng thành 456 — trùng số cuối nhưng sai
đường đi, và không nói rõ 90 alias. Đã sửa thành phát biểu theo từng phép đếm.

---

## 4. M-07 · Air-Know

### 4.1 Bảng issue

| issue_id | status | file:line | lệnh bằng chứng | kết quả | còn chặn / cần gì |
|---|---|---|---|---|---|
| A-01 interpreter là symlink gãy | **CONFIRMED** | `workbench/envs/airknow-adapted/bin/python -> /home/heheboiz/miniconda3/envs/teacher-sprc/bin/python` | `test -e … || echo DANGLING`; `ls miniconda3/envs` | `DANGLING`; `teacher-sprc` không còn tồn tại. Env mới: `workbench/envs/airknow310-la`. | — |
| A-02 lỗi import đầu là `torchaudio` | **CONFIRMED (đường khác)** | `lavis/processors/__init__.py:29` → `audio_processors.py:9` | đọc mã | `__init__` import `BeatsAudioProcessor` ⇒ `torchaudio`, rồi `moviepy.editor` (line 11). **Nhưng** `workbench/replay/lavis_processors.py` đã dựng skeleton `lavis.processors` để bỏ qua đúng import này, nên đường replay **không** đi qua `torchaudio`/`moviepy` — đã chạy được `import test_BLIP2` mà không cài hai gói đó. | — |
| A-03 `moviepy` không trên đường inference | **CONFIRMED** | `audio_processors.py:11` | như trên | Cùng file audio, không liên quan FashionIQ. | — |
| A-04 `sklearn` chặn class | **CONFIRMED** | `blip2_cir.py:19` (`from sklearn.cluster import KMeans`) | `raw/airknow_ckpt_load_cpu.log` | Cần và đã cài `scikit-learn==1.3.2`. | — |
| A-05 checkpoint deserialize trên CPU | **CONFIRMED (số liệu khớp)** | — | `raw/airknow_ckpt_load_cpu.log` | `lavis.models.blip2_models.blip2_cir.Blip2QformerCir`, **863** state-dict entry, **1 196 753 657** param, `device cpu`, load 3.2 s. | — |
| A-06 "11 hard `.cuda()` trên scoring path" | **REFUTED** | `test_BLIP2.py` (AST) + `blip2_cir.py` (AST) | `pytest`/script AST | Đường FashionIQ thực dùng: `test_BLIP2.py:111,128` (nhánh `else`, dress/shirt/toptee) = **2**, cộng `blip2_cir.py:205` nằm trong `robust_infoNCE` (**loss, không tới được từ `extract_retrieval_*`**). Tổng file là 12 biểu thức / 9 dòng: `test.py` 9 dòng (4 ở `test()`, 5 ở `lasco_*`/`test_lasco`/`test_cirr_valset`), `blip2_cir.py` 3 (`:409,410` trong `inference()` — hàm này **không được gọi** từ `test()`). | Con số 11 là tổng hai tệp, không phải "trên scoring path". |
| A-07 eval 0% / train 50% noise | **CONFIRMED (theo registry)** | `models.yaml:216-222` | đọc registry | `notes: … lists only FashionIQ 50% and 80% noise-trained weights; clean evaluation remains 0%`. | Không có checkpoint sạch; mọi số đo là transfer/noise-condition. |
| A-08 import + text processor OK | **OPENED** | `workbench/replay/lavis_processors.py` | `raw/airknow_ckpt_load_cpu.log` | `prepare()` OK; `text_processors(..., 'configs/models/blip2/blip2_pretrain.yaml')` OK qua `lavis.models`; `datasets`, `data_utils`, `test_BLIP2` import OK. | — |
| A-09 forward thật + **3/3 category × 4 lượt (NATIVE có GPU)** | **OPENED** | `lavis/models/blip2_models/blip2_cir.py:463,506` | `raw/airknow_forward_cpu_adapted.log` (CPU); `raw/airknow_forward_gpu_native.log`; `raw/airknow_run3cat_gpu_run{1..4}.json` | CPU ADAPTED: score 0.557740, 11.5 s. GPU NATIVE (không patch, không cast, peak 2634 MiB): `macro_r10 = 52.5468 ± 0.2673` trên 4 lượt (min 52.2988, max 52.9241); per-category mean dress 47.5830 / shirt 54.2566 / toptee 55.8006. | Không còn chặn để chạy. Đọc kèm §0.1: số này là **một mẫu**, không tất định (A-12). |
| A-10 checkpoint là **mixed precision**; CPU autocast là `nullcontext` nên forward CPU nổ dtype | **CONFIRMED (phát hiện mới)** | `blip2.py:36-44` (`maybe_autocast`: `enable = self.device != cpu`), `eva_vit.py:204` | `raw/airknow_forward_cpu_adapted.log` (dòng `ADAPTED: model.float()`) | 275/860 tensor là **float16** (tháp EVA-CLIP), 585 float32. Trên GPU `maybe_autocast()` bọc vision trong autocast half; trên CPU trả `nullcontext` ⇒ `RuntimeError: Input type (float) and bias type (c10::Half) should be the same`. `.float()` là **lossless** cho weights nhưng đổi precision số học của tháp vision so với GPU. | Không phải lỗi — là ràng buộc: forward CPU không thể "native" cho checkpoint này. |
| A-11 constructor native bị SIGKILL trên host 31 GB | **CONFIRMED + ĐÃ ĐỊNH LƯỢNG** | `datasets.py:95-107` | `raw/airknow_ctor_sigkill.log` (`EXIT=137`), `raw/airknow_constructor_rss_ceiling.log` | `datasets.FashionIQ(...)` vật chất hoá **cả ba category** trong một lần gọi: **21 552** input tensor, **12 976 717 824 B (12.98 GB)** tensor bytes. Với checkpoint 2.6 GB đã load, RSS vượt ngưỡng khả dụng ⇒ SIGKILL ở ~770 s. Đo riêng phần dữ liệu (không model, CPU): peak RSS **14 957 MB**, `MemAvailable` tụt còn **6 232 MB**, wall **1 130 s**; dress 5 450 MB/77 s, shirt 12 673 MB/606 s, toptee 14 957 MB/1 130 s. | Ngưỡng tái lập: **cần ≥ 15 GB RSS + model**, tức ~18 GB khả dụng. Máy 16 GB sẽ chết sớm hơn. |

### 4.2 Ghi chú

`model.train()` ở `test_BLIP2.py:10` là hành vi native — wrapper đã phản ánh đúng (`--eval-mode` là biến
thể thí nghiệm, có nhãn). Không sửa.

---

## 5. Môi trường đã dựng (script + lệnh + version)

Script: `workbench/scripts/setup_blocked_model_envs.sh`. Interpreter do `uv` quản lý, tách khỏi
`workbench/envs/ctl312h`.

| env | python | torch | torchvision | gói chính | log |
|---|---|---|---|---|---|
| `workbench/envs/clvc39-la` | 3.9.25 | 2.7.1+cu128 (CUDA 12.8) | 0.22.1+cu128 | `numpy<2`, Pillow, tqdm, pyyaml, packaging | `raw/env_clvc_install.log` |
| `workbench/envs/tgcir39-la` | 3.9.25 | 2.7.1+cu128 | 0.22.1+cu128 | `numpy==1.22.3`, `pandas==1.4.2`, `clip @ git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33` | `raw/env_tgcir_install.log` |
| `workbench/envs/airknow310-la` | 3.10.20 | 2.7.1+cu128 | 0.22.1+cu128 | `transformers==4.25.0`, `timm==0.9.16`, `scikit-learn==1.3.2`, `omegaconf==2.3.0`, `decord==0.6.0`, `einops==0.7.0`, … | `raw/env_airknow_install.log` |
| `workbench/envs/pair38-la` | 3.8.20 | 2.7.1+cu128 | 0.22.1+cu128 | `open_clip_torch==2.24.0` | (dựng, không dùng để chấm) |

Vì sao *adapted*: CLVC-Net khai Python 3.7.6/Torch 1.6.0, TG-CIR Torch 1.7.0, Air-Know 3.8.10/Torch
2.1.0 — không bản nào có kernel `sm_120`, GPU duy nhất của host là RTX 5070 Ti (compute 12.0). Đây là
cùng một cặp runtime mà registry đã ghi cho nhóm LAVIS (`torch 2.7.1 / torchvision 0.22.1 / cu128`).

Bằng chứng env "chạy được" (không phải chỉ `pip list`):

* `clvc39-la`: load 6 checkpoint, forward batch-1 (VRAM 419 MiB), và **3/3 category hai lượt trùng khít**
  (`macro_r10 30.0208`) — xem C-09.
* `tgcir39-la`: load `Fashioniq.pt` vào `VCG_CIR`, `preflight --preflight-only` → `EXIT=0`.
* `airknow310-la`: load checkpoint 2.6 GB vào `Blip2QformerCir`, `text_processors` + `test_BLIP2` import.

---

## 6. Regression tests

`workbench/tests/test_blocked_models_verification.py` (11 test) + 5 tệp test cũ của bốn model:
`29 passed, 60 warnings in 10.42s` (`raw/tests_blocked_models.log`). Hai test cũ
(`test_clvc_net_preflight_fails_on_empty_paths`, `test_tgcir_preflight_blocks_empty_environment`) đã được
cập nhật vì hợp đồng cũ **đòi** blocker chuỗi cứng — chính là defect C-02/T-03.

---

## 7. Yêu cầu GPU (chờ go-ahead)

Đã chạy hai forward batch-1 (CLVC 166 s wall / peak 419 MiB; TG-CIR 389 s wall / peak 950 MiB). Chưa chạy
job recall đầy đủ nào. GPU trước mỗi lần chạy: `nvidia-smi` chỉ có một compute app lạ `pid 3160688`/`3164557` là
chính job của tôi; không có worker khác.

Đề nghị chạy (một GPU job tại một thời điểm):

```
# CLVC-Net, 3 category, GPU native
workbench/envs/clvc39-la/bin/python -m workbench.replay.clvc_net \
  --checkpoint-root workbench/artifacts/checkpoint_selection/20261106T002617+0700/checkpoints/clvc \
  --dataset-root data/FashionIQ --source-root workbench/third_party/CLVCNet \
  --batch-size 32 --output /tmp/clvc_run_gpu.json

# TG-CIR, 3 category, workbench-added eval path
workbench/envs/tgcir39-la/bin/python -m workbench.replay.tgcir \
  --source-root workbench/third_party/TGCIR --dataset-root data/FashionIQ \
  --checkpoint-root workbench/artifacts/checkpoint_selection/20261106T002617+0700/downloads/tgcir/x/checkpoints \
  --batch-size 16 --output /tmp/tgcir_run_gpu.json
```

Ước lượng: CLVC ≈ 0.3 GPU-giờ (chủ yếu dựng cache/transform 3817–6346 ảnh mỗi category);
TG-CIR ≈ 0.25 GPU-giờ (forward batch-1 đã đo 389 s wall, phần lớn là dựng cache/import). Tổng **≈ 0.6 GPU-giờ**, VRAM dự kiến < 4 GB (weights CLVC ~2×210 MB, TG-CIR
~600 MB + CLIP ViT-B/16). Air-Know forward thật cần thêm **≈ 0.1 GPU-giờ** cho batch 1 một category
(2.6 GB weights) nếu được duyệt.

---

## 8. Ghi chú vận hành bổ sung (sau review)

### 8.1 Interpreter kiểm thử

Toàn bộ test chạy lại bằng **`workbench/envs/ctl312h/bin/python`** (CPython 3.12.11, `Python.h` có mặt)
— không phải `ctl312`. Kết quả: `33 passed, 60 warnings in 8.25s`
(`raw/tests_blocked_models_ctl312h.log`). Chạy chéo bằng `ctl312` cho cùng 23 test của ba tệp đầu:
`23 passed` — không có test nào chỉ pass dưới một env.

### 8.2 Hai test cũ không bị nới lỏng

`test_clvc_net_preflight_fails_on_empty_paths` và `test_tgcir_preflight_blocks_empty_environment` nay
khẳng định **hợp đồng thay thế**, không chỉ "không còn fail": mỗi tệp có thêm hai test âm —
`s3_preflight_fails_closed_on_a_wrong_digest` (file có mặt, byte sai ⇒ `digest mismatch`) và
`..._still_blocks_when_the_registry_is_not_verified` (digest khớp nhưng `checkpoint_mapping_status` khác
`VERIFIED_METADATA` ⇒ vẫn `mapping unresolved`). Tổng test của hai tệp tăng từ 9 lên 13.

### 8.3 GPU lock

`workbench/scripts/gpu_job.sh`: giữ `flock -n` trên `/tmp/cir_gpu.lock` suốt thời gian chạy, **bên trong
lock** mới kiểm `nvidia-smi --query-compute-apps=pid` (rỗng mới chạy), và **không** chạy unlocked, không
kill job khác. Đã test cả hai nhánh: acquire → `INSIDE_LOCK_OK`; có người giữ lock → in
`lock still held after 4s; not starting`, exit 75, không chạy lệnh.

### 8.4 Nhãn artifact Air-Know CPU

Artifact `raw/airknow_forward_cpu_adapted.log` mang nhãn **`ADAPTED_RUNTIME_CPU_PRECISION_CHANGED`**:
`model.float()` đổi precision số học của tháp vision so với autocast-half trên GPU. Con số
`0.557739794254303` **không được** so từng chữ số với một lượt GPU.

## 9. Còn treo tại thời điểm giao

TG-CIR forward batch-1 đã **hoàn tất** sau khi viết mục 2: `raw/tgcir_forward_gpu.log`, `EXIT=0`,
score `0.3138742446899414`, peak VRAM 950 MiB, wall 388.96 s. T-09 nay là OPENED với bằng chứng thật.

Air-Know A-09 nay **OPENED** nhưng ở nhãn **ADAPTED_RUNTIME (CPU)**: `raw/airknow_forward_cpu_adapted.log`,
score `0.557739794254303` trên dress qid#0, 11.5 s forward. Hai adaptation đã ghi trong log:
`torch.Tensor.cuda` neutralised và `model.float()`. Đường CPU **không thể** là native với checkpoint này
(A-10: half weights + `maybe_autocast` trả `nullcontext` trên CPU). Recall đầy đủ vẫn cần GPU.

---

## 10. Kết quả GPU đầy đủ (đã chạy, có lock)

Ba lượt chạy dưới `workbench/scripts/gpu_job.sh` (`flock -n /tmp/cir_gpu.lock`, kiểm
`nvidia-smi --query-compute-apps` **bên trong** lock). Cache ảnh được làm nóng **trước**, CPU-only,
bằng `workbench/scripts/warm_replay_cache.py`, để wall time của lượt GPU không bị tính công dựng cache.

### 10.1 Cache cold vs warm (CPU-only, không dùng GPU)

| model | cold (đo trong lần chạy đầu) | warm | số input materialise |
|---|---|---|---|
| TG-CIR | 389 s (đo ở forward batch-1) | **1 067.5 s** cho lượt warm đầy đủ 3 category, 14 635 input | 14 635 |
| CLVC-Net | tương tự, chưa đo cold riêng | **347.6 s**, 14 635 input | 14 635 |

Lưu ý: "warm" ở đây là **dựng cache**, không phải đọc cache đã có — `warm_replay_cache.py` tạo
`fashion_iq_data.json` + `test_*.pkl` để lượt GPU sau đó đi nhánh warm của constructor pinned. Thời gian
14 635 input × decode + transform ≈ 6–18 phút CPU; đây là lý do wall time GPU thật nhỏ hơn nhiều.

### 10.2 TG-CIR — NATIVE_CUDA, 3 category

`raw/tgcir_run3cat_gpu.json`, `EXIT=0`, **wall 255 s**, batch 16, cache warm.

| category | R@1 | R@10 | R@50 |
|---|---|---|---|
| dress | 14.3282 | 43.3813 | 68.3689 |
| shirt | 22.4239 | 51.4720 | 69.8234 |
| toptee | 22.5905 | 54.3090 | 75.1147 |

`macro_r10 = 49.7208`, `macro_r50 = 71.1023`.
Nhãn trong artifact: `runtime_label = NATIVE_CUDA`, `evaluation_path = WORKBENCH_ADDED_EVAL_PATH`.
**Khớp từng chữ số** với `tgcir_run1.json` lịch sử (macro 49.720772235486656) — đường eval mới tái lập
được lượt cũ, kể cả sau khi thay `isolated_cache_root` bằng `run_cache_root` có cache warm.

### 10.3 CLVC-Net — NATIVE_CUDA, 3 category

`raw/clvc_run3cat_gpu.json`, `EXIT=0`, **wall 235 s**, batch 32, cache warm.

| category | R@1 | R@10 | R@50 |
|---|---|---|---|
| dress | 6.8418 | 29.4497 | 56.7675 |
| shirt | 6.4769 | 27.7723 | 55.2502 |
| toptee | 7.8531 | 32.8404 | 63.9980 |

`macro_r10 = 30.0208`, `macro_r50 = 58.6719`.
So với lượt lịch sử `clvc_run1.json` (macro_r10 **29.9707**, r50 58.6217): chênh **+0.0501 r10** /
**+0.0502 r50**. Nguyên nhân đã truy được: lượt này dùng `Resize(256)→CenterCrop(224)` (native,
`train.py:90-96`), còn runner lịch sử `run_clvc.py` cũng khai transform đó nhưng **không** ghi vào
artifact nên không đối chứng được. Mức lệch ~0.05 pp là nhiễu số học GPU (BLAS/batch 32 vs 32, không
autocast hai bên), **không** phải khác transform — nếu khác transform thì lệch phải cỡ vài pp.
Vì vậy: lượt lịch sử nay **không còn cần nhãn "transform không xác minh"** để so sánh; nó chỉ cần nhãn
*historical, transform không được ghi lại*. Đây là kết luận "OPENED" cho C-06.

### 10.4 Air-Know — NATIVE (GPU, không cast, không patch)

`raw/airknow_forward_gpu_native.log`, `EXIT=0`, **wall 6 s**, peak VRAM **2 634 MiB**.

Trả lời câu hỏi bắt buộc: **có**, trọng số half load và chạy **không cần cast** trên GPU.
`dtypes_after_to_cuda {'torch.float32': 585, 'torch.float16': 275}` — `.to("cuda")` **không** đổi dtype
nào; `maybe_autocast_enabled True`, và `torch.cuda.amp.autocast(dtype=float16)` được bật đúng như
`blip2.py:36-44` thiết kế. Không có adaptation nào (`ADAPTATIONS []`).

* dress qid#0, `score_native_matmul_max = 0.5348384976387024`, `hash_q 30441dbb323b58b6`, forward 0.42 s.

**So sánh có kiểm soát với lượt CPU (A-09):** CPU `0.557739794254303` vs GPU `0.5348384976387024`,
lệch **0.0229** (~4.3 %). Đây là bằng chứng định lượng cho A-10: hai nhãn precision khác nhau cho ra số
khác nhau ở mức **không thể so từng chữ số**, đúng như cảnh báo. Artifact CPU giữ nhãn
`ADAPTED_RUNTIME_CPU_PRECISION_CHANGED`; artifact GPU mang nhãn `NATIVE`.

### 10.5 Tổng hợp ngân sách GPU đã dùng

| lượt | wall | ghi chú |
|---|---|---|
| CLVC forward batch-1 | 166 s | probe |
| TG-CIR forward batch-1 | 389 s | probe (cold) |
| TG-CIR 3 category | 255 s | dưới lock |
| CLVC 3 category | 235 s | dưới lock |
| Air-Know native GPU | 6 s | dưới lock |
| **tổng** | **~17.5 phút ≈ 0.29 GPU-giờ** | trong ngân sách 0.75 GPU-giờ |

Không lượt nào vượt ước lượng 2×, nên không phải kill lượt nào. Không giành GPU với worker
`limn-encoder`: mỗi lần launch đều qua lock và đều thấy `compute-apps` rỗng.

### 10.6 Điều vẫn chặn

* **PAIR**: không đổi — thiếu `model_vssp` / `MHTransformer_dis`. Không có đường nào trên host thay được.
* **PAIR**: không đổi.
* ~~Air-Know recall đầy đủ~~ — **đã chạy được**, xem §11. Điều kiện tái lập A-11 vẫn giữ nguyên
  (≳18 GB khả dụng cho constructor native, hoặc dùng cache warm như §11).

---

## 11. Air-Know 3 category — NATIVE (GPU, không patch, không cast)

### 11.1 Cách mở được A-11

Constructor native cần dựng cache cả ba category trước forward đầu tiên. Cache được dựng **trước**,
CPU-only, bằng `warm_replay_cache.py --model airknow` (**349.7 s**, 21 552 input), rồi lượt GPU chạy
trên root đó qua `run_cache_root`. Điều này **không** thay đổi đường native: vẫn
`datasets.FashionIQ(...)` pinned, `test_BLIP2.py:test`, `model.train()`, transform
`targetpad_transform(1.25, 224)`, checkpoint `fiq_n05` (SHA khớp). Nhãn: `runtime_label` suy ra NATIVE,
`model_mode = train (native behavior)`.

### 11.2 Kết quả (lượt 1 và lượt 2, cùng cấu hình, `batch_size 64`)

`raw/airknow_run3cat_gpu_run1.json`, `raw/airknow_run3cat_gpu_run2.json`, `EXIT=0`, wall **430 s** và
**433 s**.

| category | R@1 | R@10 | R@50 | R@10 (lượt 2) | Δ r10 |
|---|---|---|---|---|---|
| dress | 18.9886 | 47.7938 | 70.3024 | 46.9509 | -0.8428 |
| shirt | 23.9941 | 54.1708 | 72.6693 | 54.6614 | +0.4907 |
| toptee | 25.9052 | 56.8078 | 76.1856 | 55.7369 | -1.0709 |

`macro_r10` = **52.9241** (lượt 1) / **52.4497** (lượt 2);
`macro_r50` = 73.0524 / 72.9517.
So với số paper `reported_r10 = 55.75`: thiếu **−2.83 pp**, với điều kiện noise 50 % trong checkpoint
(registry ghi rõ không có checkpoint sạch). Không được trình bày đây là tái lập paper.

### 11.3 Phát hiện mới A-12 — eval Air-Know **không tất định**

Hai lượt y hệt lệch `macro_r10` **-0.4743 pp**, cá biệt toptee
**-1.0709 pp**. Nguyên nhân, đọc mã:

* `test_BLIP2.py:10` gọi `model.train()` cho FashionIQ (hành vi native, giữ nguyên theo yêu cầu);
* `blip2_cir.py:58,61` có `nn.Dropout(0.1)` và **đường eval không seed gì** — không `torch.manual_seed`,
  không `np.random.seed`, không `random.seed` trong `test_BLIP2.py`, `data_utils.py` hay
  `replay/airknow.py`.

⇒ Mọi con số Air-Know là **một mẫu**, không phải một giá trị tất định. Muốn so sánh công bằng phải
hoặc pin seed trước `test()`, hoặc chạy eval-mode (`--eval-mode`, đã có và có nhãn), hoặc báo khoảng
qua nhiều lượt. Đây là **thuộc tính của upstream**, không phải lỗi wrapper; wrapper hiện **không** che nó.

### 11.4 Ngân sách GPU cập nhật

Thêm 430 s + 433 s = **863 s ≈ 0.24 GPU-giờ** cho Air-Know 3 category ×2. Tổng chiến dịch
**≈0.53 GPU-giờ / 0.75**. Không lượt nào vượt 2× ước lượng.

---

## 12. Sửa theo review: variance, lặp lại, và đóng câu hỏi transform

### 12.1 A-12 nay nằm trong artifact, không chỉ trong báo cáo

`workbench/replay/airknow.py` sinh thêm khối `determinism` (cả ở top-level payload và trong
`provenance` từng category):

```json
{"eval_mode": "train", "dropout_active": true, "seed": "unset", "repeat_count": 4,
 "run_to_run_delta_macro_r10": 0.4744, "measured_two_run_macro_r10": [52.9241, 52.4497],
 "four_run_macro_r10_mean": 52.5468, "four_run_macro_r10_sd": 0.2673,
 "four_run_macro_r10_spread": 0.6253}
```

`seed` do `WORKBENCH_AIRKNOW_SEED` điều khiển (mặc định `unset` — ghi ra chứ không giả định). Khi seed
được đặt, wrapper seed `torch`/`random`/`numpy` **và** ghi giá trị vào artifact; khi không, artifact nói
`unset` và `dropout_active: true`. Bốn artifact `raw/airknow_run3cat_gpu_run{1..4}.json` đều mang khối
này; run1–3 được backfill từ chính số đo bốn lượt, có trường `_backfilled` nói rõ, metric không đổi.
`--repeat-count` thêm vào CLI để artifact tự khai nó đại diện bao nhiêu lượt.

### 12.2 Bốn lượt Air-Know: trung bình ± độ tán (per category)

`raw/airknow_run3cat_gpu_run{1..4}.json`, mỗi lượt 428–433 s dưới lock.

| category | mean R@10 | sd | min | max | spread |
|---|---|---|---|---|---|
| dress | 47.5830 | 0.4675 | 46.9509 | 48.0416 | **1.0907** |
| shirt | 54.2566 | 0.2728 | 54.0726 | 54.6614 | 0.5888 |
| toptee | 55.8006 | 0.6997 | 55.2779 | 56.8078 | **1.5298** |
| macro | **52.5468** | **0.2673** | 52.2988 | 52.9241 | 0.6253 |

Đúng như lo ngại: **spread của toptee (1.53 pp) lớn hơn khoảng cách giữa nhiều cặp model**. Mọi so sánh
Air-Know với model khác phải kèm khoảng này, hoặc phải pin seed / dùng `--eval-mode`.

### 12.3 CLVC: hai lượt của tôi **trùng khít**, nên câu hỏi transform đóng bằng đo, không bằng suy đoán

`raw/clvc_run3cat_gpu_run1.json` vs `run2.json`, cùng cache warm, batch 32, `EXIT=0`, wall 235 s / 155 s:

| category | lượt 1 | lượt 2 | Δ (tôi) | historical | Δ queries vs historical |
|---|---|---|---|---|---|
| dress | 29.4497 | 29.4497 | +0.0000 | 29.3505 | +2.00 |
| shirt | 27.7723 | 27.7723 | +0.0000 | 27.7723 | +0.00 |
| toptee | 32.8404 | 32.8404 | +0.0000 | 32.7894 | +1.00 |

`macro_r10` **30.0208 vs 30.0208, Δ = +0.0000** (r50 cũng +0.0000). Tức đường của tôi **tất định**.

⇒ Kết luận phải sửa so với bản trước: chênh lệch với historical **không phải nhiễu của tôi**. Nó là
**đúng 3 query** trên 6 016 (dress +2, toptee +1, shirt 0) — một khác biệt rời rạc, thật. Nguyên nhân
khả dĩ nhất là **môi trường số học của lượt lịch sử**, không phải transform: `run_clvc.py` lịch sử không
ghi torch version, batch size hay thread count vào artifact (`clvc_run1.json` thiếu cả ba), nên không
đối chứng được. Một khác transform `Resize(224)` vs `Resize(256)` sẽ đổi **hàng trăm** query, không phải
3.

Trạng thái đúng của C-06: **transform của lượt lịch sử vẫn chưa được ghi lại** (đó là khuyết tật
provenance thật), nhưng **không còn là nghi phạm** cho chênh lệch 0.05 pp — vì 3 query là quá nhỏ cho
một khác transform, và lượt của tôi tái lập được tuyệt đối. Bằng chứng dứt điểm còn thiếu: một lượt
CLVC với `Resize(224)` trên **một** category (≈0.05 GPU-giờ) để đếm số query đổi. **Chưa chạy** — ngân
sách GPU đã dùng hết (xem §12.5).

### 12.4 TG-CIR: tái lập từng chữ số — kết quả fidelity mạnh nhất của chiến dịch

| nguồn | dress r10 | shirt r10 | toptee r10 | macro_r10 | macro_r50 |
|---|---|---|---|---|---|
| historical `tgcir_run1.json` | 43.38125929598414 | 51.47203140333661 | 54.30902600713922 | 49.720772235486656 | 71.10231942031965 |
| **workbench eval entrypoint, 3 category, GPU** | 43.38125929598414 | 51.47203140333661 | 54.30902600713922 | **49.720772235486656** | 71.10231942031965 |

Trùng **tới từng chữ số**, không phải "gần bằng". Đường sinh ra nó: `workbench/replay/tgcir.py`
`evaluate_categories` — entrypoint **do workbench thêm**, gọi `test.test()` pinned với
`datasets.FashionIQ` pinned và `clip.load('ViT-B/16').preprocess`; `--restore_file` của tác giả vẫn
**chết** (`train.py:74`, lần xuất hiện duy nhất). Đây là bằng chứng wrapper trung thành **dù** entrypoint
của tác giả không có đường eval: cùng checkpoint, cùng annotation, cùng transform, cùng gallery
pair-union ⇒ cùng rank ⇒ cùng recall. Nhãn artifact: `runtime_label = NATIVE_CUDA`,
`evaluation_path = WORKBENCH_ADDED_EVAL_PATH`.

### 12.5 Ngân sách GPU — đã vượt

| lượt | wall |
|---|---|
| CLVC batch-1 probe | 166 s |
| TG-CIR batch-1 probe (cold) | 389 s |
| TG-CIR 3 cat | 255 s |
| CLVC 3 cat (lượt 1) | 235 s |
| Air-Know native GPU probe | 6 s |
| Air-Know 3 cat ×4 | 430 + 433 + 428 + 430 s |
| CLVC 3 cat (lượt 2) | 155 s |
| **tổng** | **2 927 s ≈ 0.81 GPU-giờ** (hạn mức 0.75) |

Vượt 0.06 GPU-giờ, do bốn lượt Air-Know được yêu cầu thêm. Không lượt nào vượt 2× ước lượng riêng nên
không phải kill. Mọi lượt đều qua `gpu_job.sh` (lock + kiểm `compute-apps` bên trong lock).

### 12.6 Registry: KHÔNG sửa; diff đề xuất cho `ptha_mtst`

`workbench/tests/fixtures/pc_launcher/models.json` lệch registry ở **11 trường**, không phải hai. Trường
nghiêm trọng là protocol truth và trạng thái nguồn:

```diff
--- a/workbench/tests/fixtures/pc_launcher/models.json
+++ b/workbench/tests/fixtures/pc_launcher/models.json
@@ -339,8 +339,12 @@
     "model_id": "ptha_mtst",
-    "preparation_contract": "unavailable",
+    "preparation_contract": "manual_standard",
+    "preparation_id": "ptha_mtst_fashioniq",
     "upstream_default_branch": "main",
-    "upstream_commit_sha": null,
-    "source_available": false,
-    "source_dir": null,
+    "upstream_commit_sha": "6a02ee4763fc2260b6abfb18be78853facb91d86",
+    "source_available": true,
+    "source_dir": "MTST_PTHA",
@@ -352,4 +356,7 @@
-    "native_protocol": null,
-    "supported_protocols": [],
+    "native_protocol": "fashioniq_original_split",
+    "supported_protocols": ["fashioniq_original_split"],
+    "literature_split_label": "original",
+    "fashioniq_layout": "fashioniq_standard",
+    "fashioniq_image_extensions": [".png"],
@@ -356,2 +363,2 @@
-    "reproduction_status": "BLOCKED_NO_VERIFIED_FASHIONIQ_CHECKPOINT",
+    "reproduction_status": "SELECTED_TRANSFER_ONLY",
```

Khuyến nghị: regenerated fixture từ `load_registry()` thay vì sửa tay, vì
`test_pc_launcher.py:1744-1780` hiện chỉ so `model_id` order và `checkpoints`, **không** so
`native_protocol`/`supported_protocols` — nên các trường lệch này đang sai mà **không test nào bắt**.
Thêm một assertion so `native_protocol`/`supported_protocols` cho mọi model sẽ biến lệch này thành
failure thay vì im lặng.

### 12.7 Test A-12 tự bắt lỗi của chính tôi

Khi thêm seed opt-in vào `replay/airknow.py`, test `test_airknow_eval_is_nondeterministic_by_upstream_design`
**fail** (`assert 'torch.manual_seed' not in ... replay/airknow.py`). Đó là hành vi đúng của một guard.
Test được cập nhật để phân biệt hai phía:

* **upstream** (`test_BLIP2.py`, `data_utils.py`) phải **không** seed — seed vào đó là adaptation;
* **wrapper** được seed, nhưng chỉ khi `WORKBENCH_AIRKNOW_SEED` được đặt, và phải khai
  `repeat_count` + `run_to_run_delta_macro_r10` trong artifact.

Cả hai phía đều được assert, nên một default-on seed trong tương lai sẽ fail test thay vì làm lệch số
đã công bố. Kết quả: `34 passed` (`raw/tests_blocked_models_ctl312h.log`).

---

## 13. Ingest: ba lượt mới có nên thay số đang được serve?

### 13.1 Trạng thái hiện tại (đo, không suy đoán)

`workbench/backend/native_measurements.py:616` quét `checkpoint_selection/**/*.json`, bỏ qua
`SKIP_PARTS = ("/downloads/", "/environments/", "/conda-meta/", "/site-packages/")`. Thư mục
`docs/reviews/.../raw/` **không** nằm dưới `checkpoint_selection/`, nên bốn artifact của tôi **không được
scan**. Kiểm bằng chính module đó:

| model | nguồn đang counted | macro_r10 | ghi chú |
|---|---|---|---|
| `clvc_net` | `measurements/clvc_run1.json` | 29.9707 | `artifact_backed: True`, `counted: True` |
| `tgcir` | `measurements/tgcir_run1.json` | 49.720772235486656 | `artifact_backed: True` |
| `airknow` | `measurements/airknow_fiq_n05_run1.json` | 52.5466 | `artifact_backed: True`; `fiq_n08` = 50.6417 |

Và `counts`: `blocks_with_metrics 43`, `with_metrics 21`, `unbacked_counted 0`,
`models_with_metrics 17`.

### 13.2 Khuyến nghị: **cài ba lượt mới vào `measurements/`, giữ file cũ, đánh dấu cũ là superseded**

Không phải vì số mới "đúng hơn" — mà vì **ba lượt mới là bằng chứng tốt hơn về cùng một đại lượng**, và
vì TG-CIR/CLVC mới có thứ cũ thiếu.

**Lý do theo từng model:**

1. **TG-CIR — thay, không tranh cãi.** Số mới **trùng từng chữ số** với cũ (49.720772235486656). Không
   có nguy cơ đổi nghĩa; cái mới chỉ thêm provenance mà cũ thiếu: `runtime_label`, `evaluation_path`,
   commit nguồn, đường dẫn transform, và việc `--restore_file` chết. Đây là upgrade thuần.
2. **CLVC — thay, và nó mang một sự thật mới.** 30.0208 vs 29.9707 (**+0.0501 pp = 3 query**). Cái mới
   có **lượt thứ hai Δ 0.0000** — tức nó tự chứng minh tính tất định, điều mà `clvc_run1.json` không
   làm được (file cũ không ghi transform, torch version, batch hay threads). Nếu giữ cũ làm canonical,
   ta đang serve một số **không thể tái lập từ artifact của chính nó**, trong khi bản mới tái lập được.
3. **Air-Know — thay, và đây là chỗ quan trọng nhất.** Xem §13.3.

**Cái KHÔNG nên làm:** xoá hoặc ghi đè `clvc_run1.json`, `tgcir_run1.json`,
`airknow_fiq_n05_run1.json`. Chúng là lịch sử có thật; xoá chúng là phá bằng chứng của chính lập luận
"3 query khác nhau". Giữ nguyên file, thêm file mới, và ghi vào file cũ một trường
`superseded_by` + `superseded_reason` (theo §1 mục "giữ nguyên artifact lịch sử và ghi superseded").

**Hai điều kiện kỹ thuật phải kiểm trước khi cài** (chưa làm, vì đây là quyết định của bạn):

* `_backing_reports` (dòng 447) chỉ nhận **hai** nguồn: `artifacts/reports/*_aggregate.json`, và
  `docs/reviews/*/raw/native/*.json` có `replay_type` + `aggregate`. Artifact của tôi hiện ở
  `raw/` chứ không phải `raw/native/`, và schema của chúng là `results` (CLVC/TG-CIR) hoặc `categories`
  (Air-Know) — `_categories` chấp nhận cả `results` (dòng 205-206), nhưng **chưa** kiểm là một file
  trong `raw/native/` có được `artifact_backed` hay không khi thiếu `reports/*_aggregate.json`.
* **SỬA (§13.4): câu "bản cũ sẽ thắng theo path order" ở trên là SAI** — tôi đã đo lại và nó không
  đúng. Xem §13.4 để có số.

### 13.3 Nếu chỉ serve một điểm ước lượng thì mất gì

Đây là lập luận quyết định, và nó **không** phải về con số:

* Số Air-Know đang serve (52.5466) và trung bình 4 lượt (52.5468) lệch nhau **+0.0001** — về mặt số học,
  thay hay không thay **không đổi gì người đọc thấy**. `52.5466` vốn dĩ **cũng là một mẫu** từ cùng phân
  phối đó (nó là một lượt chạy thật).
* Cái mất là **công bố phương sai**. Khoảng tin cậy 95 % của trung bình 4 lượt là
  **[52.1215, 52.9720]** — rộng **0.85 pp**. Một điểm ước lượng trần trụi khiến người đọc so
  `52.55` với, ví dụ, ConeSep `55.58` và kết luận "kém 3 pp", trong khi spread toptee của chính Air-Know
  là **1.53 pp** — tức một phần đáng kể của "khoảng cách giữa hai model" có thể chỉ là một lần rút thăm.
* Vì vậy: cài artifact mới **kèm** khối `determinism` (đã có: `repeat_count: 4`,
  `four_run_macro_r10_sd: 0.2673`, `spread: 0.6253`, `dropout_active: true`, `seed: unset`). Nếu API chỉ
  render một số, thì **ít nhất** phải render kèm nhãn "một mẫu, sd 0.27, dropout bật" — nếu không, việc
  cài bản mới là **vô nghĩa về mặt thông tin** và chỉ tạo cảm giác chính xác không có thật.
* Ngược lại, giữ bản cũ làm canonical **mà không** nói nó là một mẫu còn tệ hơn: người đọc thấy một số
  chính xác đến 4 chữ số thập phân cho một đại lượng có sd 0.27.

**Tóm lại (đã bị §13.4–13.6 sửa bằng phép đo):** điều kiện tiên quyết là **ingest phải có khái niệm
supersede**, và tầng serve phải mang được phương sai. Thiếu một trong hai thì **không nên cài** — xem
bảng cuối ở §13.6, đó là khuyến nghị có hiệu lực.

### 13.4 Đã ĐO cơ chế ingest trên một cây nháp — và bản trước của tôi sai

Dựng `/tmp/ingest-probe2/checkpoint_selection/20261106T002617+0700/measurements/` chứa ba file cũ
(`clvc_run1.json`, `tgcir_run1.json`, `airknow_fiq_n05_run1.json`) cộng **một** file mới
`clvc_run2.json` (chính lượt 3-category của tôi, thêm `replay_type`). Chạy
`list_native_measurements(Path("/tmp/ingest-probe2"))`. **Không ghi gì vào cây thật.**

Kết quả:

| | trước (cây thật) | sau (cây nháp, +1 file mới) |
|---|---|---|
| `with_metrics` | 19 | 4 (chỉ 4 file trong cây nháp) |
| `blocks_with_metrics` | 43 | 4 |
| `duplicate_rows` | 24 | **0** |
| clvc rows | `clvc_run1` counted | `clvc_run1` **counted** *và* `clvc_run2` **counted** |

⇒ Hai điều:

1. **`_deduplicate` không giữ bản cũ.** Khoá dedup là
   `(model_id, protocol_id, tuple(round(value,4)))` — tức **giá trị là một phần của identity**. Hai lượt
   khác nhau 0.05 pp là **hai identity khác nhau**, nên **cả hai cùng `counted: True`**. Bản trước của
   tôi nói "bản cũ thắng theo path order" là **sai**: path order chỉ quyết định khi identity *trùng*.
2. **Hệ quả thật lại ngược lại và nghiêm trọng hơn:** cài file mới sẽ làm
   `counts.with_metrics` của `clvc_net` thành **2 lượt cùng được đếm**, và `duplicate_rows` **không** tăng.
   `test_native_measurements_census.py:145` chỉ cấm **cùng identity** bị đếm hai lần —
   `len(identities) == len(set(identities))` — nên **test này vẫn xanh** dù một model có hai số counted
   khác nhau. Không có bất biến nào cấm "hai lượt khác nhau của cùng model × protocol đều counted".

**Vì vậy việc cài đặt không phải "thêm file".** Phải chọn một trong hai, và đây là quyết định thiết kế
chứ không phải thao tác:

* **(a) `superseded_by` trên bản cũ + ingest hiểu trường đó.** Cần sửa `native_measurements.py` để một
  row có `superseded_by` bị `counted: False`. Đây là **thay đổi backend**, không làm được trong lúc sweep
  chạy.
* **(b) Đặt bản mới vào `reports/*_aggregate.json`** (đường `_backing_reports` thứ nhất) và để bản cũ
  trong `measurements/`. Nhưng bản cũ vẫn `counted: True` (nó là artifact và không phải rollup), nên vẫn
  hai số — trừ khi bản cũ được xoá hoặc sửa, mà xoá thì mất bằng chứng "3 query".

⇒ **Khuyến nghị cuối cùng, có tính đến việc ingest hiện không có khái niệm supersede:** cài **TG-CIR**
ngay (số trùng từng chữ số, nên identity *trùng* và dedup xử lý đúng — đây là trường hợp duy nhất an
toàn). Với **CLVC** và **Air-Know**, **giữ nguyên số đang serve** và ghi ba lượt mới vào
`raw/native/` như **corroboration có nhãn**, **cho đến khi** `native_measurements.py` biết
`superseded_by`. Lý do: thay số mà không có cơ chế supersede sẽ làm dashboard hiển thị hai giá trị cho
cùng một model mà không nói cái nào là cái nào — đúng loại nhầm lẫn mà chiến dịch này đang sửa.

Điều này **đổi so với §13.2** ở trên. §13.2 dựa trên một giả định về `_deduplicate` mà phép đo bác bỏ.

### 13.5 Xác nhận thứ hai: TG-CIR thì dedup ĐÚNG, CLVC thì không

Thêm `tgcir_run2.json` (chính lượt 3-category, số trùng từng chữ số) vào cùng cây nháp:

| model | row | `counted` | `duplicate_of` |
|---|---|---|---|
| tgcir | `tgcir_run1.json` | **True** | — |
| tgcir | `tgcir_run2.json` | **False** | `…/tgcir_run1.json` |
| clvc_net | `clvc_run1.json` | **True** | — |
| clvc_net | `clvc_run2.json` | **True** | — |

`duplicate_rows` = 1 (chỉ TG-CIR). Đúng như phân tích: dedup khoá theo **giá trị**, nên

* **TG-CIR cài được ngay và an toàn** — file mới tự nhận là duplicate của file cũ, `with_metrics`
  không tăng, và **số được serve vẫn là `tgcir_run1.json`** với provenance cũ. Đây là bằng chứng đo được
  rằng "cài TG-CIR" **không** đổi con số nào; nó chỉ thêm một file được đánh dấu là bản lặp. Nếu mục tiêu
  là *provenance tốt hơn* (transform, `evaluation_path`, `--restore_file` chết) thì phải **thay** nội dung
  `tgcir_run1.json` hoặc thêm cơ chế supersede — thêm file mới **không** đạt mục tiêu đó, vì bản cũ vẫn
  là bản được serve.
* **CLVC/Air-Know thì không** — hai giá trị khác nhau ⇒ hai identity ⇒ cả hai `counted: True` ⇒ dashboard
  có hai số cho một model, và `test_native_measurements_census.py:145` **vẫn xanh** vì nó chỉ cấm trùng
  identity. Đã kiểm: `counted rows: 4 | distinct identities: 4 | test PASS: True`.

### 13.6 Trả lời cuối cho câu hỏi "cài hay không cài"

| model | cài file mới? | số được serve sau khi cài | vì sao |
|---|---|---|---|
| TG-CIR | **Được, nhưng vô ích nếu chỉ thêm file** | vẫn `tgcir_run1.json` (49.720772235486656) | file mới bị dedup thành duplicate; muốn provenance mới thì phải supersede/thay |
| CLVC | **Không, cho đến khi có supersede** | hai số 29.9707 *và* 30.0208 cùng counted | identity khác nhau, không cơ chế nào chọn hộ |
| Air-Know | **Không, cho đến khi có supersede** | hai số 52.5466 *và* 52.5468 cùng counted, **mất phương sai** | như trên |

Điều kiện tiên quyết cho cả ba: `native_measurements.py` phải hiểu một trường supersede (hoặc ingest phải
chọn theo `recorded_at` mới nhất trong cùng `(model, protocol)`), **và** payload phải mang khối phương sai
sang tầng serve. Cho đến lúc đó, khuyến nghị: **giữ số đang serve**, đặt ba lượt mới dưới
`docs/reviews/20261111-master-audit/raw/native/` như corroboration có nhãn (đường `_backing_reports` thứ
hai, dòng 494, nhận `replay_type` + `aggregate` — cả ba artifact của tôi đều có, CLVC/TG-CIR cần thêm
`replay_type` khi copy).

Đây là **đảo ngược có đo** so với §13.2, và §13.2 đã được sửa tại chỗ để không còn hai khuyến nghị mâu
thuẫn trong cùng một báo cáo.

### 13.7 Bẫy thứ ba: `raw/native/` **không** phải nơi "chỉ để corroboration" — nó THAY số đang serve

§13.6 khuyến nghị đặt ba lượt mới dưới `docs/reviews/.../raw/native/` coi như "ghi chú tham chiếu". Đo
lại thì **sai**: `_backing_reports` (dòng 494) coi mỗi file ở đó là **nguồn có thể BACK**, và
`_backed_by_artifact` (dòng 558-563) **ghi đè** `metrics` của row bằng số của report, chỉ để bản gốc trong
`raw_metrics`.

Cây nháp `/tmp/ingest-probe3` (`measurements/airknow_fiq_n05_run1.json` cũ + `raw/native/airknow.json` mới):

```
metrics      : {'r10': 52.9241, ...}      <- số MỚI, đang được serve
raw_metrics  : {'r10': 52.54661498980954} <- số cũ, chỉ còn để đối chứng
backed_by    : .../raw/native/airknow.json
counted      : True
```

⇒ Đặt file vào `raw/native/` là **đổi số đang serve**, không phải thêm ghi chú. Với Air-Know, nó sẽ lấy
`52.9241` — **một lượt cụ thể (lượt 1)**, không phải trung bình 4 lượt `52.5468`, và **không kèm** khối
`determinism` ở tầng serve. Đó là kết quả **tệ hơn cả hai lựa chọn** đã bàn: số vẫn là một mẫu, lại là
mẫu *khác*, và người đọc mất luôn khả năng so với báo cáo cũ.

**Hệ quả cho khuyến nghị:** đừng đặt artifact mới ở `raw/native/` trừ khi **muốn** nó thay số. Ba vị trí,
ba hành vi khác nhau, đã đo:

| vị trí | hành vi ingest | dùng khi |
|---|---|---|
| `measurements/*.json` (artifact thường) | một row, `artifact_backed: True`, `counted: True`; identity trùng ⇒ dedup, identity khác ⇒ **hai số cùng counted** | không dùng để "thêm lượt" |
| `measurements/` + số trùng từng chữ số | dedup đúng: bản sau thành `duplicate_of`, `counted: False` | TG-CIR (an toàn, nhưng không đổi provenance) |
| `docs/reviews/*/raw/native/*.json` | **ghi đè** số đang serve (`metrics` ← report), giữ `raw_metrics` | chỉ khi muốn thay số |
| `docs/reviews/*/raw/*.json` (không `native/`) | **không được scan**; không ảnh hưởng gì | nơi đúng cho "corroboration, không serve" |

⇒ Khuyến nghị có hiệu lực, thay §13.6: **giữ nguyên** `measurements/`, **giữ** ba lượt mới ở
`raw/` (không `native/`) như hiện tại — đó chính là trạng thái hiện tại và nó **không** ảnh hưởng số đang
serve. Muốn thay số thật thì phải sửa `native_measurements.py` (supersede, hoặc chọn theo `recorded_at`)
và mang phương sai sang tầng serve; cả hai đều là thay đổi backend, **không** làm trong lúc sweep chạy.
