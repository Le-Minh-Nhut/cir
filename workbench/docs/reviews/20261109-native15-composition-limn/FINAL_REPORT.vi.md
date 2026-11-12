# Báo cáo cuối — native FashionIQ, LIMN dual-scope, composition CIRCUS

Ngày: 2026-11-09. Branch `research/cir-failure-workbench`.
Thư mục: `workbench/docs/reviews/20261109-native15-composition-limn/`.

Quy ước nguồn: mỗi con số dưới đây ghi kèm file trong **thư mục này** đã chứa nó. Con số nào không
truy được về file thì bị loại khỏi báo cáo, không được thay bằng ước lượng. Các bảng tổng hợp
`NATIVE_EVAL_MATRIX.csv`, `COMPOSITION_MATRIX.csv`, `GATES.csv` do worker `native15-inventory` sở hữu;
báo cáo này đối chiếu chúng với artifact thô và ghi lại kết quả đối chiếu ở §2.5 và §7. Tường thuật
chi tiết của hai native run thêm trong lượt này (`dcnet`, `clip4cir_rn50x4_fullft`) nằm ở
`NATIVE_RUNS_EXTRA.vi.md`; bằng thô ở `raw/native/dcnet.json` và `raw/native/clip4cir_fullft.json`.
Bằng chứng parity của `encoder` ở `ENCODER_PARITY.vi.md` + `g1_g2.encoder.*.json` + `raw/encoder_parity/`.

---

## 1. Trạng thái tổng quan

| hạng mục | đạt / tổng | nghĩa |
|---|---|---|
| Native FashionIQ có aggregate hoàn tất | **11 / 15** | 10 chạy trong chiến dịch này (7 ở workstream A + `limn` + `dcnet` + `clip4cir_rn50x4_fullft`), cộng `combiner_rn50x4_noft` có run hoàn tất **từ trước** chiến dịch và được xác minh lại ở đây |
| G1 + G2 | **2 / 15** | `limn` và `encoder`; **cả hai** mang failure đã khai báo một cách trung thực (§3.4, §3.6) |
| Composition CIRCUS (8 run được chọn) | **8 / 8** | 7 `COMPLETED`, 1 `PARTIAL` (limn 6001/6003) |
| LIMN: 2 query mơ hồ được resolve canonical | **0 / 2** | provenance không cho phép; canonical giữ `PARTIAL 6001/6003` |
| LIMN: chẩn đoán 2 scope | **12 / 12 ranking** | 2 query × 2 scope × 3 điều kiện |

**Không phải 15/15.** **4** model còn thiếu native (`pair`, `airknow`, `clvc_net`, `tgcir` — §6).
**Không** tuyên bố chiến dịch hoàn tất: G1/G2 mới có **2/15** (`limn`, và `encoder` thêm ở lượt này —
G1 PASS, G2 PASS-kèm-failure-đã-khai-báo, xem §3.6 và §6c), và **4** blocker còn nguyên.

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
| `dcnet` | **0.27930** (27.93%) | **0.53761** (53.76%) | `raw/native/dcnet.json` |
| `clip4cir_rn50x4_fullft` | **43.8017** | **66.9836** | `raw/native/clip4cir_fullft.json` |

**Đơn vị của hai dòng mới KHÁC nhau, và đây là cái bẫy phải đọc kỹ.** `raw/native/dcnet.json` →
`aggregate.macro_r10 = 0.27929730914548917` / `macro_r50 = 0.5376075764301528`: đây là **phân số**,
không phải phần trăm — 0.27930 nghĩa là **27.93%**, không phải 0.279%. `raw/native/clip4cir_fullft.json`
→ `aggregate.macro_r10 = 43.8016931215922` / `macro_r50 = 66.98358456293742`: đây là **phần trăm**
(upstream `validate.py:351-352` dùng `statistics.mean` trên recall đã nhân 100). Vì vậy **hai cột này
tuyệt đối không được so trực tiếp với nhau**, và không dòng nào được đem trung bình với dòng kia.

Thêm hai cảnh báo về `dcnet`, cả hai đều từ chính JSON của nó:

* `comb_avg 0.40845` trong `info.log` là **trung bình trên sáu ô R@10/R@50** (hoặc trên cặp
  per-branch), **KHÔNG phải macro** trên cột comb; `aggregate.upstream_aggregate_note` nói nguyên văn
  điều đó, và macro đúng của báo cáo này là trung bình 3 category của **riêng cột comb** =
  **0.27930 / 0.53761**. `comb_avg` **không bao giờ được trích như một macro**.
* `r1 = null` ở cả ba category là **đúng theo upstream**: evaluator FashionIQ của DCNet chỉ tính
  R@10/R@50, nên R@1 **không tồn tại** — không phải thiếu sót của lượt chạy này.

`clip4cir_rn50x4_fullft` nằm ở bảng này vì **chính file bằng chứng của nó khai protocol đó**:
`protocol: "fashioniq_full_gallery_ref_excluded"`, `protocol_source: "third_party/CLIP4Cir/src/validate.py
::compute_fiq_val_metrics — index_features over the whole split.{category}.val.json gallery; the
reference is excluded by masking its own index entry rather than removing it from the gallery"`. Đây
**đính chính** cách xếp trước đó của báo cáo dưới `fashioniq_original_split`.

### 2.3 `fashioniq_original_split`

| model | R@10 | R@50 | ghi chú | nguồn |
|---|---|---|---|---|
| `tme` | **54.9229** | **74.8384** | — | `raw/native/tme.json` |
| `ptha_mtst` | **37.8978** | **59.8364** | **TRANSFER CIRR→FashionIQ, KHÔNG phải tái lập** | `raw/native/ptha_mtst.result.json` |
| `combiner_rn50x4_noft` | **35.1078** | **58.4756** | run hoàn tất **trước** chiến dịch; xác minh lại, không chạy lại | `workbench/artifacts/reports/combiner_…_020a70b0_aggregate.json` |

`ptha_mtst` mang `label: "PTHA+MTST | CIRR checkpoint -> FashionIQ | cross-dataset transfer |
exploratory"` và `train_dataset: CIRR` trong chính JSON của nó. Con số 37.8978 **không** so sánh được
với các dòng `fashioniq_*` như một kết quả tái lập; nó là bằng chứng transfer.

### 2.4 Đếm chính xác 11/15

10 chạy trong chiến dịch này:

* `encoder`, `hint`, `habit`, `intent`, `conesep`, `tme`, `ptha_mtst` — 7 model, đủ 3 category,
  `complete = true` trong từng JSON.
* `limn` — 3 category, `instrumentation_parity = VERIFIED` cả ba; `limn_native.json` →
  `all_parity_verified = true`.
* `dcnet` — hoàn tất **sau khi blocker được sửa**, `exit_code 0`, `wall_human "2:07:35"` (7656 s),
  `raw/native/dcnet.json` → `complete_benchmark = true`, `aggregate.complete = true`. Run này **ghi lại**
  `checkpoints/dcnet/val/att_result.pt`: `side_effects.att_result_pt.previous_mtime`
  `2026-11-07 00:02:45`, `new_mtime` **`2026-11-10 04:22:45`**, sha256 mới `6d776ffa…60438`. Vì vậy
  **số cũ của run 2026-11-07 (`comb_avg 0.40828`) KHÔNG phải output của run này**, và không được trích
  như vậy.
* `clip4cir_rn50x4_fullft` — `exit_code 0`, `wall_human "0:37:26"` (2246 s),
  `raw/native/clip4cir_fullft.json` → `complete_benchmark = true`; `side_effects.wrote_nothing_outside
  = true` (eval chỉ in ra stdout, không ghi file nào).

1 có sẵn từ trước: `combiner_rn50x4_noft` (`execution_status: EXECUTION_COMPLETED`,
`reproduction_status: EXTRACTED_VERIFIED`, `metric_extraction_available: true`).

**Tổng 11/15.** `csmcir` cũng có `native_executed=Y` nhưng bị user defer, nên **không** được tính
vào 11 (§2.5).

### 2.5 `NATIVE_EVAL_MATRIX.csv` khớp với artifact thô

Đọc trực tiếp file (17 dòng, 17 khai báo): `native_executed = Y` cho **12** model —
`conesep`, `ptha_mtst`, `habit`, `intent`, `hint`, `encoder`, `dcnet`, `combiner_rn50x4_noft`,
`clip4cir_rn50x4_fullft`, `limn`, `tme`, `csmcir`. Trừ `csmcir` (user defer) còn **11/15** — khớp
§2.1–§2.3. Cột `g1_status`/`g2_status` hiện ghi:

| model | `g1_status` | `g2_status` |
|---|---|---|
| `limn` | `PASS` | `PASS-with-caveat` |
| `encoder` | `PASS` | `PASS_WITH_DECLARED_FAILURES` |
| 10 model còn lại | `NOT_RUN` | `NOT_RUN` |

Hàng `encoder` **đã được worker `native15-inventory` cập nhật** theo `ENCODER_PARITY.vi.md`, và
`dcnet`/`clip4cir_rn50x4_fullft` đã được đặt `native_executed = Y`. Mô tả cũ trong báo cáo này
("hàng `encoder` vẫn `NOT_RUN`") **không còn đúng** và đã bị thay bằng bảng trên.

Khác biệt `verified` là **phân biệt đúng, không phải defect**: 10 model executed vẫn giữ
`native_verified = N` vì chưa có parity nào được đo cho chúng; chỉ `limn` có `native_verified = Y`, và
`encoder` có G1/G2 đo được nhưng cột `native_verified` vẫn `N` cho tới khi chủ sở hữu quyết định có
nâng nó theo `g1/g2` hay không — báo cáo này không tự nâng. Đó chính là điều §1 báo cáo (native 11/15,
G1/G2 **2/15**) — một lần chạy hoàn tất không tự động là một lần chạy đã xác minh.

`csmcir` có `native_executed = Y` nhưng bị user defer nên **không** được tính vào 11 — và metric của
nó không được dùng ở báo cáo này.

---

## 3. G1 / G2 — chi tiết `limn` và `encoder`

**Hai** model có G1/G2: `limn` (§3.1–§3.5) và `encoder` (§3.6). Tolerance của **mỗi** model được
**khai trước khi đo**: `INPUT_FORWARD_PARITY.vi.md` §0 cho `limn`,
`ENCODER_PARITY.vi.md` §0 cho `encoder`. File kết quả: `g1_g2.{dress,shirt,toptee}.{native,adapter}.json`
(`limn`) và `g1_g2.encoder.{dress,shirt,toptee}.{native,adapter}.json` (`encoder`).

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

### 3.6 `encoder` — G1 PASS, G2 PASS với **hai failure đã khai báo**

Toàn bộ chi tiết ở `ENCODER_PARITY.vi.md`; file kết quả
`g1_g2.encoder.{dress,shirt,toptee}.{native,adapter}.json`; log + npz ở `raw/encoder_parity/`.
Protocol `fashioniq_val_split`, cùng checkpoint `fashioniq.pt`
(sha256 `0f31c501…9173`), mẫu thật cố định chọn trước khi đo: index 0, hàng đầu có caption 1 viết hoa,
hàng có dấu câu/apostrophe, hàng cuối.

**G1 — PASS.** ids / thứ tự gallery / thứ tự annotation / model text **exact**: 2017 / 2038 / 1961 hàng,
**0 mismatch**; gallery dựng lại từ `cap.<cat>.val.json` khớp tuyệt đối và không có ảnh trùng.
Tokenizer `open_clip.tokenizer.SimpleTokenizer`, vocab **49408**, context **77**, `attn_mask` `[77,77]`
float32. Pixel: biểu thức decode của adapter (`Image.open` + `preprocess_val`) **bitwise** bằng decode
`preprocess_val` của native (`max_abs = 0.0`), và feature gallery hai phía **bitwise**.

**G2 forward.** Mẫu cố định: **bitwise `0.0`** cho cả query và target ở cả ba category, **qua hai
process khác nhau**. Toàn population: query `max_abs` **1.10e-06 / 1.28e-06 / 9.87e-07**, gallery
**1.13e-06 / 1.65e-06 / 9.98e-07** — nằm trong `1e-5` đã khai; `min_cosine` ≥ 0.999999999916 (khai
`1 − 1e-7`). **`max_rel` của population FAIL** (tới 7.85) vì mẫu số gần 0 — báo là FAIL, không dời
ngưỡng.

**G2 scoring.** `max_abs = 1.19e-07` (khai `1e-6`) ở cả ba category.

**G2 ranking.** Không đổi hit count nào, nhưng **2 rank change KHÔNG phải tie**, báo là **FAIL đã khai**:
dress 1 (gap `6.57e-04`), shirt 1 (gap `1.60e-04`), toptee 0; đều ở hạng ≫ 50 nên không ảnh hưởng
R@1/R@10/R@50. So sánh tie-convention trên **cùng** ma trận: 47 khác biệt, **tất cả gap `0.0`**, 0
non-tie.

**G2 batch 1-vs-N.** `max_abs` **3.20e-07 / 2.81e-07 / 3.28e-07** nằm trong `1e-4`; nhưng `max_rel`
**1.165e-02** (toptee) / 7.795e-03 (dress) / 3.131e-03 (shirt) so với khai `1e-3` ⇒ **FAIL đã khai**,
cùng nguyên nhân mẫu số gần 0.

**Hai failure trung thực, không nới ngưỡng, gate KHÔNG được nâng thành PASS sạch.** Đây đúng khuôn
mẫu của `limn`.

**Hạn chế đã khai (protocol, không phải bug).** Native val-split đưa **ảnh query** qua
`preprocess_train` (`RandomResizedCrop`), adapter dùng `preprocess_val` cho mọi input. Ảnh hưởng đo
được: ΔR@10 / ΔR@50 tối đa **0.2975 / 0.1530**, dấu không nhất quán ⇒ là nhiễu của crop ngẫu nhiên,
không phải thiên lệch. Vì vậy nó được ghi là **protocol divergence**, không phải lỗi mapping.

**Một defect upstream được phát hiện, ghi ở §6c**: `BindingDecoder.learnable_relation` không nằm trong
checkpoint và không được đăng ký, nên mỗi process rút một giá trị khác nhau ⇒ ENCODER **không thể
replay bit-identical** trừ khi seed trước khi dựng model. Mọi số ở §3.6 đo với relation được **pin ở
cả hai phía**.

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
**Còn đúng 4 model bị chặn**; hai model từng nằm ở đây đã chạy xong trong lượt này, ghi ở §6a ngay dưới.

| model | trạng thái | bằng chứng đo |
|---|---|---|
| `pair` | **STILL_BLOCKED** | Pickle cần global `model_vssp.Encoder/Backbone/FeatureWiseAffine` và `MHTransformer_dis.*`; `grep "class Encoder" workbench/third_party/PAIR` và `find -iname "*model_vssp*"` = **không kết quả**. Chỉ load được qua alias loader thăm dò. Đo được **456** state-dict entry = 455 param + 1 buffer không persistent `backbone.clip.attn_mask`; so với class công khai giả thuyết: 381 model key, **74 unexpected**, **70 shape mismatch**, **5.2001%** tham số không đi qua forward. Mapping giữ `UNVERIFIED`. |
| `airknow` | **STILL_BLOCKED** | Env thật **Python 3.11.15 / torch 2.7.1+cu128**, registry khai 3.8.10. `import lavis.models` → `ModuleNotFoundError: moviepy`; `pip list` không có moviepy. Hỏng ở import, **trước** khi chạm `Blip2QformerCir`. |
| `clvc_net` | **STILL_BLOCKED** | Không có interpreter trong registry. `.cuda()` cứng: **14 dòng / 15 biểu thức** toàn tree; **6 dòng / 7 biểu thức** trên scoring path (`test.py:28,47,99,115`; `text_model.py:96,108`). Không có `cuda.is_available()` guard ⇒ không chạy CPU. 6 checkpoint đã ánh xạ category đúng (RESOLVED phần đó). |
| `tgcir` | **STILL_BLOCKED** | `third_party/TGCIR/test.py` (162 dòng): **0** `__main__`, **0** argparse, **0** `load_state_dict`, **0** `torch.load`. `.cuda()` cứng `test.py:31,47,101,115` + `model.py:32,107,120,136,244,251`, và `model.py:41` `clip.load(..., device='cuda')`. Không interpreter. |
| `csmcir`, `sprc` | **USER-DEFERRED** | Không xoá, không tính vào 15 active. `sprc` còn thiếu `composition_adapter` và registry `BLOCKED_CHECKPOINT_MAPPING`. |

### 6a. Đã gỡ trong lượt này — hai model từng nằm trong bảng trên

| model | kết quả đo | bằng chứng |
|---|---|---|
| `dcnet` | **Đã chạy xong, không còn timeout.** Blocker được sửa thật (`weights_only` có điều kiện + `map_location=self.device` trong `base_trainer.py`), rồi run hoàn tất: `exit_code 0`, `wall_human "2:07:35"` (7656 s), 3 category, macro R@10 **0.27930** / R@50 **0.53761** (**phân số** — tức 27.93% / 53.76%). Ghi lại `val/att_result.pt`: mtime cũ `2026-11-07 00:02:45` → mới **`2026-11-10 04:22:45`**, sha256 `6d776ffa…60438`; `prior_run_metrics_used: false`. | `raw/native/dcnet.json`, `raw/native/dcnet.stdout.log`, `raw/native/dcnet.time.log` |
| `clip4cir_rn50x4_fullft` | **Đã chạy full, không còn "GPU needed".** `exit_code 0`, `wall_human "0:37:26"` (2246 s) trên CPU (torch 1.11.0+cpu), 3 category, macro R@10 **43.8017** / R@50 **66.9836** (**phần trăm**). `side_effects.wrote_nothing_outside = true` — eval chỉ in stdout, không ghi file nào. | `raw/native/clip4cir_fullft.json`, `.stdout.log`, `.time.log` |

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

## 6c. Defect upstream của ENCODER: tham số không nằm trong checkpoint

**Đây là defect của artifact upstream, KHÔNG phải defect của workbench.** Không file nào trong
`workbench/composition/`, `workbench/replay/`, `workbench/backend/` gây ra nó, và không sửa gì ở đó
có thể sửa được nó. Nó chỉ được phát hiện **vì** đã chạy G1/G2 cho `encoder`: trước lượt đó, mọi so
sánh đều dừng ở mức metric aggregate và một tham số ngẫu nhiên không lộ ra.

### Cơ chế

`workbench/third_party/ENCODER/model_try2.py:93`:

```python
self.learnable_relation = nn.Parameter(torch.randn(N_p, dim, dtype=torch.float32)).cuda()
```

`.cuda()` gọi trên `nn.Parameter` **trả về một `torch.Tensor` thường**, và kết quả đó mới là thứ được
gán vào attribute. Vì vậy `self.learnable_relation` **không bao giờ được đăng ký làm parameter**: nó
không vào `state_dict()`, không vào `named_parameters()`, và do đó không có trong file checkpoint.

### Hệ quả — đo được, từng dòng có file thô

Nguồn: `ENCODER_PARITY.vi.md` §4.0.1, `raw/encoder_parity/relation_proof_construct.out`,
`raw/encoder_parity/relation_proof_native_pair.txt`.

| khẳng định | số đo |
|---|---|
| checkpoint `fashioniq.pt` (sha256 `0f31c501…9173`) | **417** key; **không** key nào chứa `relation`; **không** key nào chứa `learnable_relation` |
| `model.state_dict()` của model dựng mới | **415** entry; `'learnable_relation' in state_dict()` → **False**; `named_parameters()` khớp `relation` → `[]` |
| attribute có tồn tại? | `hasattr` → **True**, nhưng `type` → **`torch.Tensor`** (không phải `Parameter`), shape `[2, 512]` |
| hai lần dựng trong cùng process | `(a − b).abs().max()` = **4.2088** (một lần đo khác: **5.1108**) — mỗi process tự rút một relation khác nhau |
| đặt seed **trước khi dựng model**, hai process riêng | `max_abs = 0.0`, `torch.equal` → **True** — đây là bằng chứng nhân quả: nguồn duy nhất của khác biệt là phép rút RNG này |

### Ảnh hưởng lên metric: hai lần chạy NATIVE, cùng code, cùng pixel

Hai process chạy **chính code native**, cùng pixel đã bitwise-identical, cùng toàn bộ cờ determinism
(`cudnn.deterministic=True`, `use_deterministic_algorithms(True)`, seed cố định, TF32 off):

| | dress | shirt | toptee |
|---|---|---|---|
| hạng thay đổi | **540 / 2017** | **503 / 2038** | **375 / 1961** |
| max rank delta | **24** | **28** | **41** |
| query feature `max_abs` | 1.33e-02 | 2.39e-02 | 1.81e-02 |
| score matrix `max_abs` | 2.33e-02 | 4.06e-02 | 3.36e-02 |
| ΔR@1 / ΔR@10 / ΔR@50 | −0.0496 / +0.0992 / −0.0496 | 0.0000 / +0.0491 / −0.0491 | +0.1020 / −0.0510 / 0.0000 |

**Rank churn lớn, ảnh hưởng metric thì không — và con số rank churn TUYỆT ĐỐI KHÔNG được trích như
một sai số metric.** 20–27% query đổi hạng, nhưng các hạng bị đổi nằm sâu và triệt tiêu lẫn nhau, nên
mọi Recall chỉ dịch trong khoảng **−0.05 … +0.10**. Đây cũng là toàn bộ biên độ của "độ lệch giữa các
lần chạy ENCODER" từng bị quy nhầm cho BLAS (§6b): nó là **nhiễu do phép rút relation**, không phải
defect mapping, không phải BLAS.

### Hệ quả cho tính tái lập — nói thẳng

- **`raw/native/encoder.json` (macro R@10 56.2908 / R@50 77.6334) không thể được tái lập bit-for-bit
  bởi bất kỳ ai, kể cả chính tác giả**, trừ khi RNG state tại thời điểm `Encoder()` được dựng tình cờ
  trùng. Con số đã công bố vẫn là **một measurement hợp lệ của một lần rút cụ thể**; chỉ việc replay
  chính xác là không thể. Lượt đo lại có pin cho macro R@10 **56.2398** / R@50 **77.6157**
  (Δ −0.051 / −0.018).
- **Run CIRCUS `encoder` `5dee7425` chịu đúng hệ quả đó**: không thể replay bit-identical, cùng một lý
  do. Số của nó vẫn là measurement hợp lệ của lần rút đã dùng để tính nó.
- Cách pin **đủ** để so sánh hai bên: đặt seed **trước** khi dựng `Encoder(...)`. Điều này giải thích
  vì sao `seeded_dataset_construction` sẵn có **không** che được lỗi: nó seed quanh lúc **dựng
  dataset**, tức là **sau** khi `Encoder(...)` đã rút xong relation.

**Không sửa.** Sửa tối thiểu (bỏ `.cuda()` để parameter được đăng ký) không cung cấp giá trị mà
checkpoint thiếu, nên nếu không retrain hoặc can thiệp checkpoint thì "sửa" chỉ là thay một tensor
ngẫu nhiên bằng một tensor ngẫu nhiên khác — và sẽ **đổi ngữ nghĩa đánh giá** mà không nói ra. Mọi số
trong `ENCODER_PARITY.vi.md` §4–§5 đều ghi rõ là đo với relation được pin ở **cả hai phía**.

---

## 7. Gates

`GATES.csv` do worker `native15-inventory` sở hữu, và sau lượt hiệu chỉnh của nó file **đã khớp**
với artifact thô. Bảng dưới ghi trạng thái **hiện tại trong file** và phần bằng chứng mà lượt này
đóng góp:

| gate | trạng thái trong `GATES.csv` | lượt này đóng góp gì | bằng chứng |
|---|---|---|---|
| Scope/accounting | PASS | không đổi | `GATES.csv` |
| G1 input parity | NOT_RUN (gate as a whole) | **PASS cho `limn` VÀ `encoder`** — gate tổng vẫn NOT_RUN vì 13 model còn lại không có differential nào | `g1_g2.*.native.json`, `g1_g2.encoder.*.native.json` |
| G2 forward/scoring | NOT_RUN (gate as a whole) | **PASS cho `limn`** + caveat batch-relative đã ghi trong file; **`encoder`: PASS-kèm-2-failure-đã-khai-báo** (§6c, `ENCODER_PARITY.vi.md` §4.2/§5.2) | `g1_g2.*.adapter.json`, `g1_g2.encoder.*.adapter.json` |
| Native completeness | **FAIL** (gate = "all 15") | file đã ghi **11/15** với đúng 11 tên và nguồn; FAIL vì tiêu chí là *toàn bộ* 15, không phải *phần lớn* | `raw/native/*.json`, `raw/native/dcnet.json`, `raw/native/clip4cir_fullft.json`, `limn_native.json`, report `combiner_…020a70b0` |
| Composition correctness | PASS | 7 COMPLETED + 1 PARTIAL, 8/8 | `COMPOSITION_MATRIX.csv` |
| **LIMN dual-scope** | **PASS** | 12 record, mỗi record có JSONL Top-100; `rank_convention_crosscheck all_agree = true` trên 36 | `limn_dual_scope.json` + 6 JSONL |
| Data/cache integrity | PASS | 8/8 feature cache digest recomputed & matched | `limn_dual_scope.json` |
| Numerical reproducibility | NOT_RUN | **một phần**: G1/G2 đóng cho `limn`; BLAS-sensitivity đã đo ở nhiều model | `COMPOSITION_MATRIX.csv` cột `blas_pin` |
| Performance | NOT_RUN | **một phần**, số đo thật | `PERFORMANCE.csv`, `raw/native/*.meta.txt` |
| Browser usability | NOT_RUN | **PASS cho panel mới** (1920/2560/390, không tràn ngang) | `raw/ui/report.json` |
| Regression/release | NOT_RUN | **một phần**: 34 test liên quan pass; **không** chạy full suite | `RUN_LOG.csv` |

`Native completeness` vẫn **FAIL** và đó là trạng thái đúng: **4** model chưa có run hoàn tất (§6), nên
một gate định nghĩa là "cả 15" không được xanh chỉ vì 11 model xong. Ba gate còn `NOT_RUN`
(`Numerical reproducibility`, `Performance`, `Regression/release`) cũng đúng như vậy — lượt này đo
được một phần nhưng không đủ để đóng.

---

## 8. Cái gì đã chứng minh, cái gì chưa

### Đã chứng minh (có file)

1. 11/15 model active có ít nhất một aggregate native hoàn tất; 10 trong số đó chạy trong chiến dịch này.
2. **Hai** model đạt G1, và G2 đo được với failure đã khai báo: `limn` (ids/order/text exact, tensor ảnh
   bit-identical, feature bitwise `0.0`, score `≤ 5.4e-07`, **0** khác biệt hạng so với native
   aggregate) và `encoder` (§3.6: G1 PASS, sample forward bitwise `0.0` qua hai process, population
   `max_abs ≤ 1.65e-06`, score `1.19e-07`, **0** thay đổi hit count — nhưng **2 rank change non-tie** và
   `max_rel` batch **FAIL**, cả hai báo là FAIL đã khai). **Cập nhật:** nay **0 rank change non-tie**
   (`non_tie_count 0`, `min_gap_abs 0.0` ở cả ba category); `max_rel` batch vẫn FAIL. Hai con số `2`
   ở trên là bản ghi của phép đo trên đường chưa seed — `20261112-parity-repair/ENCODER_RELATION.md` §4.
3. 12 ranking chẩn đoán cho 2 query mơ hồ, đủ 2 scope × 3 điều kiện, trên nguyên gallery 74,381.
4. Category của `7:3038`/`7:3790` **không** xác định được, và tôi chỉ ra được vì sao ở mức code
   upstream + dữ liệu (không phải "không tìm ra").
5. 8/8 run CIRCUS được chọn có metric thật; 1 run PARTIAL đúng 2 query, không giấu. **Con số CIRCUS
   đứng vững như đã đo — nhưng lời giải thích từng gắn vào thay đổi của chúng thì không** (§6b:
   quy cho BLAS là sai; nguyên nhân thật là hàm `rank` của adapter đổi).
6. `dcnet` blocker đã sửa **và run đã hoàn tất**: `exit_code 0`, 2 h 07 m, macro R@10 **0.27930** /
   R@50 **0.53761** (phân số ⇒ 27.93% / 53.76%), `att_result.pt` được ghi lại với mtime
   **2026-11-10 04:22:45** — nên metric của run 2026-11-07 **không** dùng cho run này.
7. `clip4cir_rn50x4_fullft` bundle load sạch 0/0, **413/669** tensor CLIP thật sự fine-tuned, **và** full
   eval đã chạy: `exit_code 0`, 37 m 26 s, macro R@10 **43.8017** / R@50 **66.9836** (phần trăm).

### Chưa chứng minh / còn mở

1. **4 model chưa có native run**: `pair`, `airknow`, `clvc_net`, `tgcir` (§6) — mỗi cái có nguyên nhân
   đo được riêng, không cái nào là "chưa thử".
2. **G1/G2 cho 13 model còn lại**: `limn` và `encoder` đã có (**2/15**), 13 model kia chưa. Không có
   suy luận nào từ hai model này sang model khác.
3. **Sub-check FAIL đã khai báo, không đóng bằng cách nới ngưỡng**: `limn` batch-size `max_rel`; và với
   `encoder` là **2 rank change không phải tie** + `max_rel` của batch 1-vs-N (§6c).
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

Không xếp 11 model native thành một leaderboard (ba protocol khác nhau, và `dcnet` báo phân số trong khi
`clip4cir_rn50x4_fullft` báo phần trăm). Không lấy trung bình chéo
protocol. Không dùng metric CIRCUS để nói về native. Không dùng `ptha_mtst` như tái lập FashionIQ.
Không gán metric của run cũ cho run mới (như ca `dcnet`: `comb_avg 0.40845` là trung bình trên sáu ô
R@10/R@50, **không phải** macro, và số 2026-11-07 **không** phải output của run 2026-11-10).
Không tuyên bố 15/15.
