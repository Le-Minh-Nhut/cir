# Báo cáo cuối — native FashionIQ, LIMN dual-scope, composition CIRCUS

Ngày: 2026-11-09. Branch `research/cir-failure-workbench`.
Thư mục: `workbench/docs/reviews/20261109-native15-composition-limn/`.

Quy ước nguồn: mỗi con số dưới đây ghi kèm file trong **thư mục này** đã chứa nó. Con số nào không
truy được về file thì bị loại khỏi báo cáo, không được thay bằng ước lượng. Các bảng tổng hợp
`NATIVE_EVAL_MATRIX.csv`, `COMPOSITION_MATRIX.csv`, `GATES.csv` do worker `native15-inventory` sở hữu;
báo cáo này đối chiếu chúng với artifact thô và ghi lại kết quả đối chiếu ở §2.5 và §7.

---

## 1. Trạng thái tổng quan

| hạng mục | đạt / tổng | nghĩa |
|---|---|---|
| Native FashionIQ có aggregate hoàn tất | **9 / 15** | 8 chạy trong chiến dịch này (7 ở workstream A + `limn`), cộng `combiner_rn50x4_noft` có run hoàn tất **từ trước** chiến dịch và được xác minh lại ở đây |
| G1 + G2 | **1 / 15** | chỉ `limn`; và chính nó mang **một sub-check fail thật** (§3.4) |
| Composition CIRCUS (8 run được chọn) | **8 / 8** | 7 `COMPLETED`, 1 `PARTIAL` (limn 6001/6003) |
| LIMN: 2 query mơ hồ được resolve canonical | **0 / 2** | provenance không cho phép; canonical giữ `PARTIAL 6001/6003` |
| LIMN: chẩn đoán 2 scope | **12 / 12 ranking** | 2 query × 2 scope × 3 điều kiện |

**Không phải 15/15.** 6 model còn thiếu native (§6). **Không** tuyên bố chiến dịch hoàn tất: G1/G2
mới có 1/15, và 6 blocker còn nguyên.

Danh sách 15 model active = 17 khai báo − 2 deferred (`csmcir`, `sprc`) theo `GATES.csv` dòng
`Scope/accounting`. Đếm lại trên `NATIVE_EVAL_MATRIX.csv` (17 dòng) khớp 17, trừ 2 deferred còn 15.

---

## 2. Native FashionIQ — theo protocol, không trộn bảng

Ba protocol là ba định nghĩa gallery/denominator khác nhau, nên **không** được xếp chung một
leaderboard và **không** được lấy trung bình chéo protocol. `macro R@k` trong mỗi bảng là trung bình
không trọng số trên 3 category của **chính model đó**, không phải trung bình giữa các model.

### 2.1 `fashioniq_val_split` (gallery = target mà query tham chiếu, đúng split val)

| model | R@10 | R@50 | nguồn |
|---|---|---|---|
| `hint` | **59.8034** | **80.1308** | `raw/native/hint.json` |
| `encoder` | **56.2908** | **77.6334** | `raw/native/encoder.json` |
| `limn` | **55.7919** | **77.6378** | `limn_native.json` |

`limn` thêm R@1 = **23.3394** (`limn_native.json` → `aggregate.macro_r1`).

### 2.2 `fashioniq_full_gallery_ref_excluded`

| model | R@10 | R@50 | nguồn |
|---|---|---|---|
| `habit` | **54.8733** | **74.5034** | `raw/native/habit.json` |
| `intent` | **54.8662** | **75.4319** | `raw/native/intent.json` |
| `conesep` | **54.7613** | **75.2151** | `raw/native/conesep.json` |

### 2.3 `fashioniq_original_split`

| model | R@10 | R@50 | ghi chú | nguồn |
|---|---|---|---|---|
| `tme` | **54.9229** | **74.8384** | — | `raw/native/tme.json` |
| `ptha_mtst` | **37.8978** | **59.8364** | **TRANSFER CIRR→FashionIQ, KHÔNG phải tái lập** | `raw/native/ptha_mtst.result.json` |
| `combiner_rn50x4_noft` | **35.1078** | **58.4756** | run hoàn tất **trước** chiến dịch; xác minh lại, không chạy lại | `workbench/artifacts/reports/combiner_…_020a70b0_aggregate.json` |

`ptha_mtst` mang `label: "PTHA+MTST | CIRR checkpoint -> FashionIQ | cross-dataset transfer |
exploratory"` và `train_dataset: CIRR` trong chính JSON của nó. Con số 37.8978 **không** so sánh được
với các dòng `fashioniq_*` như một kết quả tái lập; nó là bằng chứng transfer.

### 2.4 Đếm chính xác 9/15

8 chạy mới: `encoder`, `hint`, `habit`, `intent`, `conesep`, `tme`, `ptha_mtst` (7, đủ 3 category,
`complete = true` trong từng JSON) + `limn` (3 category, `instrumentation_parity = VERIFIED` cả ba,
`limn_native.json` → `all_parity_verified = true`).
1 có sẵn từ trước: `combiner_rn50x4_noft` (`execution_status: EXECUTION_COMPLETED`,
`reproduction_status: EXTRACTED_VERIFIED`, `metric_extraction_available: true`).
**Tổng 9/15.** `csmcir` cũng có `native_executed=Y` nhưng bị user defer, nên **không** được tính
vào 9 (§2.5).

### 2.5 `NATIVE_EVAL_MATRIX.csv` khớp với artifact thô

Sau lượt hiệu chỉnh của worker `native15-inventory`, matrix ghi `native_executed = Y` cho
`encoder`, `hint`, `habit`, `intent`, `conesep`, `tme`, `ptha_mtst`, `limn`, `combiner_rn50x4_noft`
(và `csmcir`, model deferred), với metric macro và đường dẫn `raw/native/*.json` / `limn_native.json`
làm bằng chứng — khớp §2.1–§2.3 của báo cáo này và khớp `aggregate.complete = true` trong các JSON.

Khác biệt `verified` là **phân biệt đúng, không phải defect**: 8 model executed giữ
`native_verified = N` và `g1/g2 = NOT_RUN` vì chưa có parity nào được đo cho chúng; chỉ `limn` có
`native_verified = Y` cùng `g1 = PASS`, `g2 = PASS-with-caveat`. Đó chính là điều §1 báo cáo (native
9/15 nhưng G1/G2 chỉ 1/15) — một lần chạy hoàn tất không tự động là một lần chạy đã xác minh.

`csmcir` có `native_executed = Y` nhưng bị user defer nên **không** được tính vào 9 — và metric của
nó không được dùng ở báo cáo này.

---

## 3. G1 / G2 — chi tiết LIMN

Chỉ `limn` có G1/G2. Tolerance được **khai trước khi đo**, trong `INPUT_FORWARD_PARITY.vi.md` §0.
File kết quả: `g1_g2.{dress,shirt,toptee}.{native,adapter}.json`.

### 3.1 G1 input parity — **PASS**

| kiểm | dress | shirt | toptee |
|---|---|---|---|
| gallery order = dựng lại từ `cap.<cat>.val.json` | ✔ | ✔ | ✔ |
| reference/target ids = annotation | ✔ | ✔ | ✔ |
| model text = `correct_text(c1) + " and " + correct_text(c2)` | 2017/2017 | 2038/2038 | 1961/1961 |

Token/transform hai phía giống nhau vì cùng dựng từ checkpoint: `open_clip.tokenizer.tokenize`,
vocab 49408, context 77; `Resize(224,bicubic)→CenterCrop(224)→convert RGB→ToTensor→Normalize`
(mean/std lấy từ `visual` của checkpoint). Tensor ảnh đã decode **bit-identical**.

Một khác biệt code có thật được tìm và **đo**, không suy đoán: `datasets.get_test_img` của upstream
gọi `img.convert('RGB')`, worker adapter không gọi. Với sample native, mọi ảnh là PIL `RGB` nên
no-op; và `image_transform(is_train=False)` của open_clip có `_convert_to_rgb` bên trong nên việc
convert vẫn xảy ra. Ghi chú phụ có bằng chứng: 32/74,381 ảnh gallery CIRCUS là mode `L`, nên phát
biểu "mọi ảnh đều RGB" **chỉ** đúng cho sample native, không đúng cho toàn gallery.

### 3.2 G2 forward — PASS bitwise, có một điều kiện

`max_abs = 0.0` và bitwise-equal cho **cả query lẫn target**, ở **cả ba** category. Restart worker
(cache tắt) cũng `max_abs = 0.0`.

### 3.3 G2 scoring + ranking — PASS

Score `max_abs ≤ 5.364e-07` (khai `≤ 1e-6`). Ranking: **0 khác biệt** so với hạng native của chính
aggregate, trên 2017 / 2038 / 1961 query, `max_abs_rank_delta = 0`.

Khác biệt tie: 27 / 7 / 13 hạng, và **tất cả** là tie float32 **chính xác** (gap score lớn nhất
trong mỗi ca = `0.0`), `unexplained = 0`. Ví dụ cặp `(argsort, adapter)`: `(1,3)`, `(12,4)`,
`(1554,1555)` — không phải lệch ±1 cố định, nên một luật "cho phép ±1" sẽ sai.

### 3.4 Sub-check **FAIL** — batch size 1 vs 16, và tôi không nâng tolerance

| category | max_abs | max_rel đo | ngưỡng khai | verdict |
|---|---|---|---|---|
| dress | 1.751e-07 | **1.1399e-02** | `1e-3` | **FAIL** |
| shirt | 1.974e-07 | **2.698e-03** | `1e-3` | **FAIL** |
| toptee | 1.714e-07 | **7.509e-03** | `1e-3` | **FAIL** |

Nguyên nhân đã đo: `max_rel` chia từng phần tử cho `|reference|`, và vector 768 chiều chuẩn hoá đơn
vị thường có thành phần gần 0; sai số tuyệt đối 1.7e-07 trên thành phần cỡ 1.5e-05 đọc thành 1.1e-02
tương đối. Sai số tuyệt đối thấp hơn ngưỡng tuyệt đối 3 bậc; cosine ≥ 1 − 1.1e-12; **hạng không đổi**.
Nhưng đúng như đã khai trước, sub-check này **fail**, và báo cáo giữ nguyên chữ fail. Không có
"nới ngưỡng để gate xanh".

### 3.5 G2 là **native parity**, không phải CIRCUS parity

`text_only_black_image` và `image_only_empty_text` là định nghĩa của workbench, không có tương
đương upstream (không code upstream nào đưa ảnh zero vào). G1/G2 đo đường native. Không có kết quả
nào ở §3 nói gì về CIRCUS.

---

## 4. LIMN dual-scope

### 4.1 Phán quyết provenance: **không resolve được**, và không phải "bị làm rơi"

Category **chưa từng tồn tại** trong pipeline release M-BEIR task-7:

* `TIGER-AI-Lab/UniIR`, `src/data/preprocessing/fashioniq_data_preprocessor.py` — `fashioniq_to_mbeir_entry`
  ghi đúng 7 khoá (`qid, query_txt, query_img_path, query_modality, query_src_content,
  pos_cand_list, neg_cand_list`); `query_src_content` chỉ `{"candidate_img_id": …}`. Category bị xoá
  ngay tại bước dựng list.
* Tôi dựng lại pipeline đó (dress→shirt→toptee, `format_fashioniq_sentence` + `format_string`, nối
  `" and "`, giữ hàng đầu theo khoá `(ảnh, text)`) và so với release: **6003/6003 khớp** qid, text,
  reference, target; **0 mismatch**. Biên block `dress[0,2010) shirt[2010,4047) toptee[4047,6003)`
  trùng category mà loader suy ra.
* Cả 4 file release (full, SF, validated_solved, validated_unsolved) đều 8 khoá, không có `category`.
* Audit artifact có `category` nhưng là **shortcut label** (`either`, `multimodal_needed`), không
  phải garment category.
* File release local có sha256 trùng `lfs.oid` upstream HF (`b601ef6a…`), nên không phải bản import
  bị cắt.

Hai query này là ca **duy nhất** mà annotation nói hai category như nhau:

| query | ảnh tham chiếu | `cap.shirt.val.json` | `cap.toptee.val.json` | kết quả chuẩn hoá |
|---|---|---|---|---|
| `7:3038` | `B004TMGDME` | hàng 1027 `["is darker and has a different graphic","is darker"]` | hàng 1595 **y hệt** | `is darker and has a different graphic and is darker` |
| `7:3790` | `B0079K0YBO` | hàng 1780 `["Is grey and has a less whimsical graphic","is lighter"]` | hàng 1426 **y hệt** | `is grey and has a less whimsical graphic and is lighter` |

Vì khoá index là `(stem ảnh, text chuẩn hoá)`, mỗi cặp ánh xạ vào `frozenset({"shirt","toptee"})` hai
phần tử → resolver trả `None`. Đó là hành vi đúng, đã được test giữ.

**Không sửa production code ở đường routing.** Không có mapping error để sửa; một "fix" chỉ có thể là
mặc định một scope (bịa) hoặc chọn theo hạng đích (bị cấm). Vì vậy không có test fail-trước để viết
cho routing. Canonical status giữ **PARTIAL 6001/6003**.

### 4.2 12 ranking chẩn đoán

Gallery **74,381** ảnh cho mọi scope (không lọc category). Hạng đích (1-based) / score đích:

| query | scope | multimodal | text-only (ảnh đen) | image-only (text rỗng) |
|---|---|---|---|---|
| `7:3038` | `shirt` | **3** / 0.570377 | **88** / 0.422086 | **152** / 0.367792 |
| `7:3038` | `toptee` | **43** / 0.567825 | **389** / 0.343617 | **67** / 0.454051 |
| `7:3790` | `shirt` | **65** / 0.514070 | **6891** / 0.217340 | **2200** / 0.301990 |
| `7:3790` | `toptee` | **43** / 0.619016 | **4730** / 0.272491 | **4047** / 0.299975 |

Nguồn: `limn_dual_scope.json` → `queries.<qid>.scopes.<scope>.<condition>`; Top-100 + vị trí gallery
+ đường dẫn ảnh ở 6 file `limn_dual_scope.<scope>.<condition>.jsonl`.

Checkpoint sha256 đo lại từ file, khớp `expected_sha256` của registry:
`shirt` = `9260c7f77d3017d8cc10e6652cc8cfe275dbc9704366fa2bf581a24808930d74`,
`toptee` = `79f663a14d89d455fdc3612b0b36d011c06d7957d1475a654755e1053a6cece2`.

Tính đúng đắn của phép chấm được chứng minh trước khi chấm hai query: **36 hạng canonical đã ghi**
được tái lập từ chính feature cache của run, bằng `Adapter.score`/`Adapter.rank` — 36/36 khớp. Cả 8
feature cache chỉ được dùng sau khi **tính lại `content_digest`** và khớp sidecar; lệch là abort.

Không có tie ở ô nào (margin nhỏ nhất 0.0835, lớn hơn `1e-6` bốn bậc), nên hạng không phụ thuộc quy
ước tie.

### 4.3 Điều chẩn đoán **không** làm

Không cộng vào canonical (metrics run `9e4918a6…` không đổi), không chọn scope, không trộn embedding
hai scope, không rút mẫu số, không chấm trùng.

---

## 5. Composition CIRCUS — 8 run được chọn

Nguồn: `COMPOSITION_MATRIX.csv`. Đây là metric **R@10/R@50 multimodal** trên cohort CIRCUS, **không**
phải metric native; không so trực tiếp với §2.

| model | checkpoint | run_id | R@10 | R@50 | trạng thái | validation |
|---|---|---|---|---|---|---|
| `habit` | `fiq_n02` | `c6eb9d5775ae636a3b8addbc` | **35.6155** | **54.3728** | COMPLETED | NOT_RUN |
| `conesep` | `fiq_n02` | `a4f786d8af4d46ac2a85ab9c` | **35.4823** | **54.1396** | COMPLETED | NOT_RUN |
| `intent` | `fiq_n02` | `4901da139ac40b874d2363c5` | **34.3995** | **54.0063** | COMPLETED | NOT_RUN |
| `tme` | `fashioniq_0` | `cd1a696d8bca0fc4bce6925b` | **34.0163** | **53.9564** | COMPLETED | NOT_RUN |
| `hint` | `fashioniq` | `f461f391c6d4696011be8067` | **32.8169** | **51.7741** | COMPLETED | NOT_RUN |
| `encoder` | `fashioniq` | `5dee7425dd46c5f154808aac` | **31.0345** | **49.4753** | COMPLETED | METRICS_MEASURED_MAPPING_UNVERIFIED |
| `limn` | `base_iter0_all_categories` | `9e4918a6b5c243412cbc58c9` | **30.0283** | **48.6752** | **PARTIAL 6001/6003** | BLOCKED_RUNTIME_PREREQUISITES |
| `ptha_mtst` | `cirr_transfer` | `f54a77450ebd2126590d4385` | **19.4736** | **34.4494** | COMPLETED | SELECTED_TRANSFER_ONLY |

7 `COMPLETED` + 1 `PARTIAL` = 8/8 có metric thật. `ptha_mtst` là **TRANSFER CIRR→FashionIQ**, không
phải tái lập — nên thứ hạng thấp của nó là kỳ vọng, không phải "model kém".

`limn` PARTIAL: mỗi điều kiện 6001/6003 chấm được, 2 query ở §4 không chấm. nDCG gap
`0.5391438513723383`, `execution_status: PARTIAL`
(`workbench/artifacts/composition/summary/limn_base_iter0_all_categories_9e4918a6b5c243412cbc58c9/full.json`).

---

## 6. Blocker còn lại — mỗi cái kèm bằng chứng đo

Nguồn chính: `BLOCKER_RESOLUTION.vi.md`, log `raw/blockers/`.

| model | trạng thái | bằng chứng đo |
|---|---|---|
| `pair` | **STILL_BLOCKED** | Pickle cần global `model_vssp.Encoder/Backbone/FeatureWiseAffine` và `MHTransformer_dis.*`; `grep "class Encoder" workbench/third_party/PAIR` và `find -iname "*model_vssp*"` = **không kết quả**. Chỉ load được qua alias loader thăm dò. Đo được **456** state-dict entry = 455 param + 1 buffer không persistent `backbone.clip.attn_mask`; so với class công khai giả thuyết: 381 model key, **74 unexpected**, **70 shape mismatch**, **5.2001%** tham số không đi qua forward. Mapping giữ `UNVERIFIED`. |
| `airknow` | **STILL_BLOCKED** | Env thật **Python 3.11.15 / torch 2.7.1+cu128**, registry khai 3.8.10. `import lavis.models` → `ModuleNotFoundError: moviepy`; `pip list` không có moviepy. Hỏng ở import, **trước** khi chạm `Blip2QformerCir`. |
| `clvc_net` | **STILL_BLOCKED** | Không có interpreter trong registry. `.cuda()` cứng: **14 dòng / 15 biểu thức** toàn tree; **6 dòng / 7 biểu thức** trên scoring path (`test.py:28,47,99,115`; `text_model.py:96,108`). Không có `cuda.is_available()` guard ⇒ không chạy CPU. 6 checkpoint đã ánh xạ category đúng (RESOLVED phần đó). |
| `tgcir` | **STILL_BLOCKED** | `third_party/TGCIR/test.py` (162 dòng): **0** `__main__`, **0** argparse, **0** `load_state_dict`, **0** `torch.load`. `.cuda()` cứng `test.py:31,47,101,115` + `model.py:32,107,120,136,244,251`, và `model.py:41` `clip.load(..., device='cuda')`. Không interpreter. |
| `dcnet` | **RESOLVED_BUT_CPU_TIMEOUT** | Blocker đã sửa thật: `weights_only` có điều kiện + `map_location=self.device` (`base_trainer.py`). Chạy lại **vượt** chỗ chết cũ và tới `Running test on cpu`, nhưng **bị timeout 2400 s của chính tôi cắt** (`EXIT=124`), **không metric nào** từ run đó. `comb_avg 0.40828` là của **run cũ** cùng checkpoint (mtime 2026-11-06/07), dùng làm sanity, **không** phải kết quả run này. Vì vậy `dcnet` **không** được tính trong 9/15. `GATES.csv` gọi lần đó là `EXECUTION_FAILED`; nguyên
nhân bên dưới là timeout 2400 s của chính tôi (`EXIT=124`), không phải lỗi model — hai cách ghi này
không xung đột, nhưng nên đọc cùng nhau. |
| `clip4cir_rn50x4_fullft` | **RESOLVED_BUT_GPU_NEEDED** | Bundle load sạch: Combiner **12 key, 0 missing, 0 unexpected**; CLIP **669 key, 0 missing, 0 unexpected**, và **413/669** tensor khác stock CLIP (max \|Δ\| 7.20e-4) ⇒ thật là fine-tuned. Chưa chạy full eval (74,381 ảnh × RN50x4 trên CPU quá lâu); upstream `validate.py` chỉ in aggregate (`compute_fiq_val_metrics` dòng 21-57) nhưng `generate_fiq_val_predictions` (dòng 60-110) đã có đủ mảng cho per-query export ⇒ export khả thi mà không đổi metric. |
| `csmcir`, `sprc` | **USER-DEFERRED** | Không xoá, không tính vào 15 active. `sprc` còn thiếu `composition_adapter` và registry `BLOCKED_CHECKPOINT_MAPPING`. |

Ba claim cũ **sai** đã đính chính bằng đo, ghi ở `BLOCKER_RESOLUTION.vi.md` §4: patch dcnet vô điều
kiện chết trên torch 1.4.0; registry mô tả checkpoint dcnet thiếu `ConfigParser` + nguồn CUDA; và
"clvc_net 8 dòng `.cuda()`" thực tế là 14/15.

---

## 6b. Đính chính: 12.727 không phải do BLAS

**Claim sai.** Báo cáo này (và mọi tài liệu clean-rerun trước đó) đã ghi rằng **12.727/18.009 hạng**
của `encoder` đổi *vì BLAS thread pin*. Attribution đó **sai**. Sáu tài liệu khác cùng
`COMPOSITION_MATRIX.csv` đã được sửa trong cùng một lượt.

### Nguyên nhân, theo thứ tự đóng góp

**1. Chính yếu — hàm `rank` của adapter đã đổi.** Đọc từ
`workbench/artifacts/composition/runs/<run_id>/ranks/encoder_fashioniq.json`:

| trường | run CŨ `6a528e877c328772e2c7bb28` | run MỚI `5dee7425dd46c5f154808aac` |
|---|---|---|
| `adapter_module_source_hash` | `1a8b0dd2…` | `bf60d855…` |
| `adapter_source_hashes.rank` | `5f828b13…` | `f0914b12…` |
| `adapter_source_hashes.score` | `0ab4fe1d…` | **`0ab4fe1d…` (giống hệt)** |

`score` giống hệt, `rank` khác. Cùng một ma trận điểm, khác hàm sắp thứ tự — đó là phát biểu sạch
nhất về nguyên nhân: hạng đổi vì **cách đếm hạng** đổi, không phải vì số học điểm đổi.

**2. Thứ yếu — interpreter đổi.** Cũ: `envs/pair-adapted` Python 3.11.15, `verified: false`
(`PYTHON_VERSION_MISMATCH`). Mới: `envs/encoder310-adapted` Python 3.10.20, `verified: true`.
Cùng torch 2.7.1+cu128 nhưng khác bản dựng Python/extension.

**3. BLAS pin KHÔNG phải nguyên nhân.** A/B có kiểm soát trong
`workbench/docs/reviews/20261108T-throughput/d3b_encoder_aggregate_scan.txt` — cùng vector, chỉ đổi
threads 1 vs 10 — báo **`changed: 1`, `max_drank: 1`, `r10_delta: 0.0`, `r50_delta: 0.0`,
`ndcg_delta: 2.181e-05`** ở cả ba điều kiện. Tức **3/18.009**, không phải 12.727/18.009.

**4. Chứng cứ phụ trợ.** R@10 **tăng** 31.0178 → 31.0345 trong khi R@50 **giảm** 49.5752 → 49.4753.
Ngược chiều nhau. Một thay đổi thread pin không thể làm hai chỉ số đi hai hướng khác nhau; một hàm
sắp thứ tự khác thì có thể.

### Hệ quả

**`6a528e87` và `5dee7425` KHÔNG phải cùng một phép tính, nên không bao giờ được đem so với nhau như
một đo BLAS-sensitivity, và bất kỳ phát biểu nào trước đây nói "BLAS làm đổi 12.727 hạng" đều sai.**
Muốn đo BLAS thì phải chạy A/B trên cùng vector trong cùng interpreter — đúng như tệp
`d3b_encoder_aggregate_scan.txt` đã làm, và tệp đó cho 1 hạng.

Không metric, không `run_id`, không giá trị gate nào thay đổi trong lượt sửa này: chỉ **attribution**
đổi. `COMPOSITION_MATRIX.csv` vẫn ghi `encoder` R@10 **31.0345** / R@50 **49.4753**, `blas_pin`
`requested=1 effective=1 (verified)`; chuỗi trong cột `blas_pin` của matrix đã bỏ phần quy sai cho
BLAS và giữ lại phần đo được.

---

## 7. Gates

`GATES.csv` do worker `native15-inventory` sở hữu, và sau lượt hiệu chỉnh của nó file **đã khớp**
với artifact thô. Bảng dưới ghi trạng thái **hiện tại trong file** và phần bằng chứng mà lượt này
đóng góp:

| gate | trạng thái trong `GATES.csv` | lượt này đóng góp gì | bằng chứng |
|---|---|---|---|
| Scope/accounting | PASS | không đổi | `GATES.csv` |
| G1 input parity | NOT_RUN (gate as a whole) | **PASS cho `limn`** — file ghi rõ "LIMN is the ONLY model with a G1 PASS"; 14 model còn lại không có differential nào nên gate tổng vẫn NOT_RUN | `g1_g2.*.native.json` |
| G2 forward/scoring | NOT_RUN (gate as a whole) | **PASS cho `limn`** + caveat batch-relative đã ghi trong file | `g1_g2.*.adapter.json` |
| Native completeness | **FAIL** (gate = "all 15") | file đã ghi **9/15** với đúng 9 tên và nguồn; FAIL vì tiêu chí là *toàn bộ* 15, không phải *phần lớn* | `raw/native/*.json`, `limn_native.json`, report `combiner_…020a70b0` |
| Composition correctness | PASS | 7 COMPLETED + 1 PARTIAL, 8/8 | `COMPOSITION_MATRIX.csv` |
| **LIMN dual-scope** | **PASS** | 12 record, mỗi record có JSONL Top-100; `rank_convention_crosscheck all_agree = true` trên 36 | `limn_dual_scope.json` + 6 JSONL |
| Data/cache integrity | PASS | 8/8 feature cache digest recomputed & matched | `limn_dual_scope.json` |
| Numerical reproducibility | NOT_RUN | **một phần**: G1/G2 đóng cho `limn`; BLAS-sensitivity đã đo ở nhiều model | `COMPOSITION_MATRIX.csv` cột `blas_pin` |
| Performance | NOT_RUN | **một phần**, số đo thật | `PERFORMANCE.csv`, `raw/native/*.meta.txt` |
| Browser usability | NOT_RUN | **PASS cho panel mới** (1920/2560/390, không tràn ngang) | `raw/ui/report.json` |
| Regression/release | NOT_RUN | **một phần**: 34 test liên quan pass; **không** chạy full suite | `RUN_LOG.csv` |

`Native completeness` vẫn **FAIL** và đó là trạng thái đúng: 6 model chưa có run hoàn tất (§6), nên
một gate định nghĩa là "cả 15" không được xanh chỉ vì 9 model xong. Ba gate còn `NOT_RUN`
(`Numerical reproducibility`, `Performance`, `Regression/release`) cũng đúng như vậy — lượt này đo
được một phần nhưng không đủ để đóng.

---

## 8. Cái gì đã chứng minh, cái gì chưa

### Đã chứng minh (có file)

1. 9/15 model active có ít nhất một aggregate native hoàn tất; 8 trong số đó chạy trong chiến dịch này.
2. `limn` đạt G1 và G2 native parity: ids/order/text exact, tensor ảnh bit-identical, feature bitwise
   `0.0`, score `≤ 5.4e-07`, **0** khác biệt hạng so với native aggregate.
3. 12 ranking chẩn đoán cho 2 query mơ hồ, đủ 2 scope × 3 điều kiện, trên nguyên gallery 74,381.
4. Category của `7:3038`/`7:3790` **không** xác định được, và tôi chỉ ra được vì sao ở mức code
   upstream + dữ liệu (không phải "không tìm ra").
5. 8/8 run CIRCUS được chọn có metric thật; 1 run PARTIAL đúng 2 query, không giấu. **Con số CIRCUS
   đứng vững như đã đo — nhưng lời giải thích từng gắn vào thay đổi của chúng thì không** (§6b:
   quy cho BLAS là sai; nguyên nhân thật là hàm `rank` của adapter đổi).
6. `dcnet` blocker đã sửa và evaluator **vượt** chỗ chết cũ (không metric vì timeout của chính tôi).
7. `clip4cir_rn50x4_fullft` bundle load sạch 0/0, và 413/669 tensor CLIP thật sự fine-tuned.

### Chưa chứng minh / còn mở

1. **6 model chưa có native run**: `pair`, `airknow`, `clvc_net`, `tgcir`, `dcnet` (timeout),
   `clip4cir_rn50x4_fullft` (chưa chạy full).
2. **G1/G2 cho 14 model còn lại**: chỉ `limn` có. Không có suy luận nào từ `limn` sang model khác.
3. **`limn` sub-check batch-size fail** theo ngưỡng tương đối đã khai — chưa đóng, và tôi không nới
   ngưỡng.
4. **Canonical cho 2 query mơ hồ**: vẫn 0/2. Sẽ chỉ đóng được nếu tác giả FashionIQ/M-BEIR công bố
   mapping category per-query, hoặc nếu chấp nhận một tiên nghiệm có khai báo (hiện không được phép).
5. **Không** chạy full test suite (chỉ 34 test liên quan, pass). Không chạy lại toàn cohort.
6. **Hai CSV không còn mâu thuẫn**: `NATIVE_EVAL_MATRIX.csv` và `GATES.csv` đã được worker
   `native15-inventory` hiệu chỉnh và hiện khớp artifact thô (§2.5, §7). Việc đó **không** do lượt
   này làm, và báo cáo này không nhận công cho nó.
7. Không có metric native nào được chứng minh là **tái lập paper**; tất cả là chạy lại trên dữ liệu
   thật với runtime đã adapted (`sm_120`, torch 2.7.1), trừ ghi chú riêng cho `combiner`/`sprc`/`csmcir`
   vốn dùng stack cũ hơn.
8. **Attribution của thay đổi CIRCUS `encoder` chưa từng được kiểm soát cho tới lượt sửa này.** Con số
   R@10 31.0345 / R@50 49.4753 là đo được và giữ nguyên; nhưng lời giải thích cũ ("BLAS làm đổi
   12.727 hạng") là **sai** và đã bị rút (§6b). `6a528e87` và `5dee7425` **không phải cùng một phép
   tính** nên không được dùng làm cặp so sánh độ nhạy BLAS. Đo BLAS đúng cách cho 1 hạng
   (`d3b_encoder_aggregate_scan.txt`).

### Cái báo cáo này cố ý không làm

Không xếp 9 model native thành một leaderboard (ba protocol khác nhau). Không lấy trung bình chéo
protocol. Không dùng metric CIRCUS để nói về native. Không dùng `ptha_mtst` như tái lập FashionIQ.
Không gán metric của run cũ cho run mới (như ca `dcnet`). Không tuyên bố 15/15.
