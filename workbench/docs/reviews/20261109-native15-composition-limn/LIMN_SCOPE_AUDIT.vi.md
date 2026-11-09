# LIMN — hai query CIRCUS mơ hồ category: provenance, và chẩn đoán hai scope

Ngày: 2026-11-09. Lượt canonical được phân tích: `9e4918a6b5c243412cbc58c9`
(`limn_base_iter0_all_categories`, CIRCUS `full`, gallery 74,381, 6,003 query).
Mọi con số dưới đây đọc từ file trên đĩa; nguồn ghi kèm từng bảng.

---

## 0. Kết luận ngắn

**Provenance KHÔNG giải được category.** Pipeline sinh release M-BEIR task-7 không mang trường
category nào, ở bất kỳ tầng nào, cho bất kỳ query nào — không phải "bị làm rơi", mà **chưa từng có**.
Category chỉ tồn tại trong FashionIQ gốc, dưới dạng *file annotation* (`cap.<cat>.val.json`), và với
hai query này cả hai file đều chứa đúng cặp caption đó. Vì vậy canonical status **giữ nguyên
6001/6003**, và tôi không sửa gì ở đường routing.

**Chẩn đoán hai scope thì chạy được thật.** Cả hai query được chấm dưới cả `shirt` và `toptee`, đủ
ba điều kiện CIRCUS, trên nguyên gallery 74,381 — 12 bảng xếp hạng, lưu riêng từng scope, không
scope nào được chọn làm "đáp án".

| Query | scope | MM | Text-only | Image-only |
|---|---|---|---|---|
| `7:3038` | `shirt` | **3** (0.5704) | **88** (0.4221) | **152** (0.3678) |
| `7:3038` | `toptee` | **43** (0.5678) | **389** (0.3436) | **67** (0.4541) |
| `7:3790` | `shirt` | **65** (0.5141) | **6891** (0.2173) | **2200** (0.3020) |
| `7:3790` | `toptee` | **43** (0.6190) | **4730** (0.2725) | **4047** (0.3000) |

Nguồn: `limn_dual_scope.json` (mục `queries.<qid>.scopes.<scope>.<condition>`), 6 file
`limn_dual_scope.<scope>.<condition>.jsonl` cho Top-100 + đường dẫn ảnh.

---

## 1. Task 1 — tái lập hai query và truy provenance

### 1.1 Record trong release (đọc từ file release)

`workbench/artifacts/circus_release/mbeir_fashioniq_task7_test.jsonl`, sha256
`b601ef6a5411bf9385cff41e64bea863871ab1094cd09d78683de6a5141165ef` (khớp hash M-BEIR upstream,
đối chiếu ở §1.4):

| | `7:3038` | `7:3790` |
|---|---|---|
| dòng | 3038 (index 3037) | 3790 (index 3789) |
| `query_img_path` | `mbeir_images/fashioniq_images/B004TMGDME.jpg` | `mbeir_images/fashioniq_images/B0079K0YBO.jpg` |
| `query_txt` | `Is darker and has a different graphic and is darker.` | `Is grey and has a less whimsical graphic and is lighter.` |
| `pos_cand_list` | `["7:3416"]` | `["7:42304"]` |
| đích | `B004D8QCN4` | `B00E3WVAD2` |
| `query_src_content` | `{"candidate_img_id": "B004TMGDME"}` | `{"candidate_img_id": "B0079K0YBO"}` |
| vị trí đích trong gallery | 13614 | 70410 |

`pos_cand_list` được resolve qua `mbeir_fashioniq_task7_cand_pool.jsonl` (74,381 dòng, sha256
`121ec284832c33289ee13bfac19d5d59dc3f8aeddf672d784fd5eb8bcdf79eef`); chính vị trí đó trùng với
`target_gallery_index` của audit artifact (`13614`, `70410`), tức gallery của workbench là cùng
một gallery, cùng thứ tự với release. Nguồn: `raw/t1_provenance.json`.

### 1.2 Caption gốc — cả hai category đều có, nguyên văn

`data/FashionIQ/captions/cap.<cat>.val.json`, quét mọi hàng có `candidate` = ảnh tham chiếu:

| query | category | hàng (0-based) | `captions` (nguyên văn) | nối `" and "` | khớp text release |
|---|---|---|---|---|---|
| `7:3038` | **shirt** | **1027** | `["is darker and has a different graphic", "is darker"]` | `is darker and has a different graphic and is darker` | ✔ |
| `7:3038` | **toptee** | **1595** | `["is darker and has a different graphic", "is darker"]` | *(giống hệt)* | ✔ |
| `7:3790` | **shirt** | **1780** | `["Is grey and has a less whimsical graphic", "is lighter"]` | `Is grey and has a less whimsical graphic and is lighter` | ✔ |
| `7:3790` | **toptee** | **1426** | `["Is grey and has a less whimsical graphic", "is lighter"]` | *(giống hệt)* | ✔ |

Hàng không khớp của cùng ảnh tham chiếu, để thấy việc "có hàng caption" không phải là bằng chứng:
`cap.shirt.val.json[217]` = `["is lighter", "it is blue"]` (target `B000WNEK7Y`) — ảnh
`B004TMGDME` xuất hiện **3** lần trong `shirt`, chỉ **một** hàng tái dựng đúng text query.

`split.{shirt,toptee}.val.json` đều chứa **cả** ảnh tham chiếu và ảnh đích; `split.<cat>.test.json`
không chứa cả hai trong bất kỳ category nào. Nghĩa là: cả hai query đến từ **cùng một cặp annotation
được nộp hai lần** dưới hai category, không phải từ hai annotation khác nhau.

### 1.3 Vì sao chuẩn hoá làm chúng trùng

`normalize_query_text` (casefold, bỏ dấu câu, gộp khoảng trắng) áp lên text nối:

```
7:3038  shirt  ["is darker and has a different graphic", "is darker"]
        toptee ["is darker and has a different graphic", "is darker"]
        -> cùng một chuỗi: "is darker and has a different graphic and is darker"

7:3790  shirt  ["Is grey and has a less whimsical graphic", "is lighter"]
        toptee ["Is grey and has a less whimsical graphic", "is lighter"]
        -> cùng một chuỗi: "is grey and has a less whimsical graphic and is lighter"
```

Khoá của `caption_scope_index` là `(stem ảnh, text chuẩn hoá)`, nên mỗi cặp ánh xạ vào
`frozenset({"shirt","toptee"})` có **hai** phần tử → `resolve_query_category` trả `None`. Đây là
đúng hành vi mong muốn (từ chối thay vì đoán), không phải lỗi.

Với 117 query mơ hồ còn lại, cơ chế ngược lại: ảnh nằm trong hai val-split nhưng chỉ **một**
category có hàng caption tái dựng ra text, nên index trả về một phần tử và routing giải được.

### 1.4 Category có bị làm rơi ở đâu trong pipeline release không?

Trả lời: **không có trường nào để rơi.** Bằng chứng theo thứ tự từ nguồn phát sinh:

1. **Code sinh release, đọc từ upstream.** `TIGER-AI-Lab/UniIR`,
   `src/data/preprocessing/fashioniq_data_preprocessor.py` — hàm `fashioniq_to_mbeir_entry` dựng
   mỗi entry bằng đúng bảy khoá: `qid`, `query_txt`, `query_img_path`, `query_modality`,
   `query_src_content`, `pos_cand_list`, `neg_cand_list`; `query_src_content` chỉ chứa
   `{"candidate_img_id": ...}`. Ba file caption mỗi split được đọc **lần lượt** rồi
   `extend` vào một list chung, `aggregate_candidates_for_mbeir_format_dataset` gộp theo khoá
   `f"{query_txt}-{query_img_path}-{query_modality}"`, rồi qid được cấp lại tuần tự
   `7:{i+1}`. **Không** có `category`, `subtask`, `source_file` hay biến thể nào của chúng —
   identification duy nhất của category bị xoá ngay tại bước dựng danh sách.
2. **Đo lại giả thuyết đó trên dữ liệu thật.** Tôi dựng lại đúng pipeline (đọc
   `dress → shirt → toptee`, chuẩn hoá qua `format_fashioniq_sentence` + `format_string`, nối
   `" and "`, giữ hàng xuất hiện đầu tiên theo khoá `(ảnh, text)`) và so **6003/6003** dòng với
   release: khớp tuyệt đối `qid`, text, ảnh tham chiếu, ảnh đích — **0 mismatch**. Số hàng sau
   gộp: `dress 2010`, `shirt 2037`, `toptee 1956` = 6003, và biên block trùng đúng
   `dress[0,2010)`, `shirt[2010,4047)`, `toptee[4047,6003)` như category mà loader suy ra được.
   Nghĩa là category **có** tồn tại trong dữ liệu, nhưng chỉ dưới dạng *vị trí dòng*, và vị trí
   dòng không phân biệt được hai query này vì cả hai nằm trong block `shirt` (3037 và 3789 đều
   `< 4047`).
3. **Bốn file release đều không có trường category.** Quét key của dòng đầu trong
   `mbeir_fashioniq_task7_test.jsonl`, `..._test_multimodal.jsonl` (SF), `validated_solved`,
   `validated_unsolved`: cả bốn chỉ có 8 khoá như §1.1. Không có `category`.
4. **Audit artifact cũng không có.** `shortcut_audit/retrieval_data_fashioniq_task7_aggregated_best_rank_mm_any.json`
   (6,003 sample) có `query_idx`, `query_text`, `query_image_path`, `target_image_path`,
   `category` — nhưng `category` ở đây là **shortcut category** (`either` / `multimodal_needed`,
   tức nhãn của audit về việc query có giải được bằng một modality), không phải garment category.
   Hai query này mang `either` và `multimodal_needed`. Không dùng được.
5. **File release trong repo đúng là byte của upstream.** sha256 của `mbeir_fashioniq_task7_test.jsonl`
   (`b601ef6a…`) trùng `lfs.oid` mà HuggingFace API báo cho
   `query/test/mbeir_fashioniq_task7_test.jsonl` của `TIGER-Lab/M-BEIR`; candidate pool
   (`121ec284…`) là file 74,381 dòng tương ứng `cand_pool/local/mbeir_fashioniq_cand_pool.jsonl`.
   Vậy đây không phải bản sao bị cắt bớt khi import — upstream cũng không có trường đó.

Kết luận có bằng chứng: không tồn tại "per-query category field tồn tại upstream nhưng thiếu trong
release". Nguồn duy nhất phân biệt được category là **file annotation**, và với hai query này nó
nói hai category như nhau. Vì vậy **không có fix-at-source**, và không có regression test
fails-before vì không có hành vi sai để sửa: một "fix" ở đây chỉ có thể là chọn một scope bằng
tiên nghiệm, tức đúng thứ bị cấm.

### 1.5 Cái *đã* từng sai, và đã được sửa trước lượt này

Bản phân tích 2026-11-08 (`20261108T114545+0700-limn/LIMN_UNSCORED.vi.md`) ghi 119 query bị bỏ,
trong đó 117 có thể giải từ annotation. Lượt canonical hiện tại
(`9e4918a6…`) đã áp dụng: 6001 chấm, 2 bỏ. Hai query còn lại là đúng hai query này. Vậy "one
regression from 119 to 2" là lịch sử, không phải việc còn tồn.

---

## 2. Task 2 — chẩn đoán hai scope

### 2.1 Giao thức đã dùng

| Thành phần | Giá trị |
|---|---|
| script | `scripts/t2_dual_scope.py` (chạy lại được, idempotent) |
| controller | `workbench/envs/ctl312h/bin/python` 3.12.11; torch 2.15.0+cu132 |
| model interpreter | `workbench/envs/limn310-adapted/bin/python` 3.10.20; torch 2.7.1+cu128 |
| adapter | `workbench.composition.model_limn.Adapter` — `encode_queries`, `encode_targets`, `score`, `rank` (nguyên production path) |
| staging input | `runner._stage_query_path` → `inputs.adapt_query_inputs` (black image / empty text đúng định nghĩa CIRCUS) |
| gallery | 74,381 ảnh, protocol `circus_fashioniq_release_v1`, **không lọc category** |
| điều kiện | `multimodal`, `text_only_black_image`, `image_only_empty_text` |
| batch / thread | `WORKBENCH_LIMN_BATCH_SIZE=2` (2 query/job), `WORKBENCH_DECODE_WORKERS=2`, OMP/OPENBLAS/MKL/NUMEXPR = 2 |
| tie/rank | `Adapter.rank` → `1 + count(row > v) + count(row[:i] == v)`, đúng evaluator |
| thời gian | 60.5 s (bao gồm 6 lần encode 2 query + 36 lần verify `content_digest` của cache) |

### 2.2 Checkpoint, sha256 đo lại từ file

| scope | checkpoint | sha256 | size | mapping |
|---|---|---|---|---|
| `shirt` | `base_iter0_shirt` / `0_shirt_best_model.pt` | `9260c7f77d3017d8cc10e6652cc8cfe275dbc9704366fa2bf581a24808930d74` | 1,792,010,157 | `VERIFIED_METADATA` |
| `toptee` | `base_iter0_toptee` / `0_toptee_best_model.pt` | `79f663a14d89d455fdc3612b0b36d011c06d7957d1475a654755e1053a6cece2` | 1,792,010,157 | `VERIFIED_METADATA` |
| `dress` | không dùng | — | — | — |

Băm được tính lại trên file và **so với `expected_sha256` của registry** trước khi dùng; lệch là
abort. `dataset_type` index upstream: `shirt=1`, `toptee=2`.

### 2.3 Gallery cache dùng lại — fail-closed

Chỉ dùng `gallery_{shirt,toptee}.npy` và 6 `query_<cond>_{shirt,toptee}.npy` của chính run
canonical, sau khi:

* đọc sidecar `.npy.json`, so `shape` với file;
* **tính lại `content_digest`** (sha256 trên dtype + shape + toàn bộ bytes, chunk 8192 dòng) và so
  với digest ghi trong sidecar. Cả 8 file: khớp. Lệch → `SystemExit`, không ranking.

| file | shape | digest (đo lại) | verify_mode |
|---|---|---|---|
| `gallery_shirt.npy` | (74381, 768) | `1896231c9dbbbfe32c5205eaf33a3cd24430e56da63163db1b4c692830d9f576` | full |
| `gallery_toptee.npy` | (74381, 768) | `fac3e6d7d3644ece8e742eaac86060452d973da0ccc647b1f68026c4b86e6a9f` | full |

Kiểm tra bổ sung: 2048 hàng đầu của mỗi ma trận có norm ∈ `[1 − 8.9e-8, 1 + 8.9e-8]`, tức đúng ma
trận target đã L2-normalize của `compose_target` — không phải ma trận của run khác.

### 2.4 Quy ước hạng đã đối chứng bằng run thật (không phải giả định)

Trước khi chấm hai query, tôi chấm lại **36 hạng canonical đã ghi** (12 query × 3 điều kiện,
query nằm trong scope `shirt`/`toptee`, không phải hai query mơ hồ) từ chính cache của lượt, bằng
`Adapter.score` + `Adapter.rank`. **36/36 khớp tuyệt đối** với `ranks/limn_base_iter0_all_categories.json`.
Đó là bằng chứng cùng lúc cho: gallery ORDER đúng, scoring path đúng, và tie rule đúng.
(`limn_dual_scope.json` → `rank_convention_crosscheck`.)

### 2.5 Kết quả 2 × 2 × 3

Hạng đích (1-based, trên 74,381 ảnh), điểm đích, điểm #1:

| Query | scope | condition | rank | target score | #1 score |
|---|---|---|---|---|---|
| `7:3038` | shirt | multimodal | **3** | 0.570377 | 0.814774 |
| `7:3038` | shirt | text_only_black_image | **88** | 0.422086 | 0.740477 |
| `7:3038` | shirt | image_only_empty_text | **152** | 0.367792 | 0.738514 |
| `7:3038` | toptee | multimodal | **43** | 0.567825 | 0.817536 |
| `7:3038` | toptee | text_only_black_image | **389** | 0.343617 | 0.689705 |
| `7:3038` | toptee | image_only_empty_text | **67** | 0.454051 | 0.723991 |
| `7:3790` | shirt | multimodal | **65** | 0.514070 | 0.644287 |
| `7:3790` | shirt | text_only_black_image | **6891** | 0.217340 | 0.601059 |
| `7:3790` | shirt | image_only_empty_text | **2200** | 0.301990 | 0.786004 |
| `7:3790` | toptee | multimodal | **43** | 0.619016 | 0.702563 |
| `7:3790` | toptee | text_only_black_image | **4730** | 0.272491 | 0.605579 |
| `7:3790` | toptee | image_only_empty_text | **4047** | 0.299975 | 0.720134 |

Không có tie ở bất kỳ ô nào: `target_score < score_at_rank_1` ở cả 12, và margin nhỏ nhất là
0.0835 (`7:3790`/toptee/MM) — lớn hơn `tie_epsilon = 1e-6` mười nghìn lần, nên hạng không phụ
thuộc quy ước tie.

Nhận xét đọc được từ bảng (nêu ra, không dùng để chọn scope):

* `7:3038` đạt hạng rất tốt ở `shirt` (MM #3) nhưng chỉ #43 ở `toptee`; chiều ngược lại với
  `7:3790` ở điều kiện MM (#65 `shirt` vs #43 `toptee`). **Không** có scope nào tốt hơn ở cả hai
  query, nên "chọn scope tốt hơn" cũng không cho một câu trả lời nhất quán — đúng lý do để không
  chọn.
* Text-only tệ hơn image-only ở cả hai query, mọi scope: phù hợp với việc checkpoint được huấn luyện
  trên ảnh + text, và với việc ảnh đen là input ngoài phân phối.
* `7:3790` ở text-only `shirt` là #6891 — ngoài @10 rất xa; nếu scope này là scope đúng thì query là
  một retrieval failure thật. Chẩn đoán không được phép biến điều đó thành "hit".

### 2.6 Điều chẩn đoán **không** làm

* Không scope nào được cộng vào canonical: `metrics` của run `9e4918a6…` không đổi, coverage vẫn
  `6001/6003 = 0.99966683`, `execution_status` vẫn `PARTIAL`.
* Không trộn embedding giữa hai scope, không ghép gallery của scope này với query của scope kia.
* Không mẫu số rút gọn (vẫn 74,381 candidate mỗi lần), không chấm trùng một query hai lần trong
  một condition, không bịa điểm miss.
* Không đảo ngược chiều quyết định theo hạng: bảng được tính trước khi nhìn hạng, và không có nhánh
  nào trong script đọc `target_rank` để chọn scope.

### 2.7 Một bug của chính script này, đã sửa (ghi lại để không ai đọc nhầm artifact cũ)

Lần chạy đầu mở file JSONL per-scope **bên trong** vòng lặp query, nên mỗi file chỉ còn dòng của
query cuối (`7:3790`); hạng trong `limn_dual_scope.json` vẫn đúng nhưng Top-100 của `7:3038` mất.
Đã sửa (một lần mở, ghi cả hai dòng) và chạy lại toàn bộ; lần chạy lại cho hạng **giống hệt**
lần đầu trên cả 12 ô, và mỗi file JSONL có 2 dòng. Artifact trên đĩa là bản đã sửa.

---

## 3. Vì sao không sửa production code cho routing

Tiêu chí của task: chỉ sửa nếu tìm được mapping error phục hồi được, và phải kèm test fail-trước.
Không có mapping error: §1.4 chứng minh field không tồn tại để mà phục hồi. Thêm một nhánh
"nếu mơ hồ thì lấy scope có hạng tốt hơn" sẽ là chọn bằng target — bị cấm tuyệt đối. Thêm
"mặc định `toptee`" hay "`shirt`" chỉ là thiên vị không có bằng chứng, và làm mất luôn thông tin
rằng dữ liệu không quyết định được.

Vì vậy: **0 thay đổi ở đường routing/evaluator**, và `workbench/tests/test_limn_scope_routing.py`
vẫn giữ nguyên hai assertion bảo vệ trạng thái này —
`test_text_filed_under_two_categories_stays_unscored` và
`test_all_119_unscored_queries_are_classified` (kỳ vọng `{7:3038, 7:3790}` vẫn `None`).

---

## 4. Task 5 — frontend hiển thị cả hai scope

### 4.1 Cái đã làm

`workbench/frontend/LimnScopeDiagnostic.tsx`, render trong `SamplesPage` (view `explorer`), tiêu đề
**"Ambiguous category — diagnostic comparison"**.

| thành phần | nội dung |
|---|---|
| badge ở card head | `Canonical: 6,001/6,003 scored` |
| notice (luôn hiện) | "Đây là chẩn đoán, không phải metric canonical", nêu đích danh `7:3038`, `7:3790`, coverage `6,001/6,003`, và câu "không số nào ở đây được cộng vào nDCG, Recall hay coverage canonical" |
| bảng | 2 query × 3 điều kiện × 2 cột scope (`shirt`, `toptee`), mỗi ô: hạng đích, score, `trên 74,381 ảnh`, `cách #1 <margin>`; header cột ghi `checkpoint_id` + sha256 (title) |
| disclosure | vì sao mơ hồ (cùng cặp caption dưới hai category), ảnh tham chiếu/đích + vị trí gallery từng query, gallery dùng cho mọi scope, và 36 hạng canonical đã tái lập khớp toàn bộ |
| probe | chọn query × scope × điều kiện → bảng Top-20 từ đúng file JSONL của scope đó, kèm ghi chú "diagnostic ranking … not a canonical per-query row" |

Backend: `workbench/backend/limn_diagnostic.py` +
`GET /api/composition/diagnostics/limn-dual-scope` và
`GET /api/composition/diagnostics/limn-dual-scope/candidates?qid=&scope=&condition=&depth=`.

Các quy tắc được **cơ chế hoá**, không chỉ bằng lời:

* endpoint trả `canonical_status` trong **cùng payload** với so sánh, nên không view nào render được
  so sánh mà thiếu trạng thái canonical;
* thiếu **một** scope, hoặc thiếu **một** điều kiện của một scope → raise
  `DiagnosticEvidenceError` → HTTP 503, vì một so sánh một-phía sẽ được đọc như "đã resolve được
  category";
* `canonical_status.unscored_qids` phải đúng `{7:3038, 7:3790}`, nếu không thì refuse;
* mỗi scope đọc **chỉ** file của mình; không có code path nào trộn, lấy max, hay trung bình hai scope;
* không render ảnh ứng viên trong panel này (một tile ảnh cạnh metric canonical dễ bị đọc như một
  hàng canonical); đường dẫn ảnh nằm trong JSONL.

### 4.2 Kiểm chứng trên app đang chạy

Service được restart (không có GPU job nào chạy lúc đó) để phục vụ endpoint mới + bundle mới.
`/api/benchmark-runs` → 200. Screenshot: `raw/ui/diagnostic-{1920x1080,2560x1440,390x844}.png`,
crop full-res `raw/ui/diagnostic-card-1920.png`, DOM facts `raw/ui/report.json`.

| viewport | card có mặt | số hàng bảng | `body.scrollWidth` / `innerWidth` | badge |
|---|---|---|---|---|
| 1920×1080 | 1 | 6 | 1920 / 1920 | `Canonical: 6,001/6,003 scored` |
| 2560×1440 | 1 | 6 | 2560 / 2560 | `Canonical: 6,001/6,003 scored` |
| 390×844 | 1 | 6 | 390 / 390 | `Canonical: 6,001/6,003 scored` |

Ở 390 px, lần đo đầu cho bảng rộng 478 px **tràn ra ngoài** (page scroll ngang) — đó là defect thật,
không phải false positive từ thumbnail: `table_box.width = 478.06` với `innerWidth = 390`. Đã sửa
bằng CSS (`[data-testid="limn-diagnostic-table"] { min-width: 560px }` trong `.table-wrap` cuộn),
rebuild, và đo lại: `body.scrollWidth == innerWidth == 390`. Bảng vẫn phải cuộn **trong** hộp của nó
ở mobile (cố ý: 4 cột số liệu không đọc được nếu ép vào 390 px), nhưng document không còn tràn.

Kiểm tra nội dung trên crop full-res (không đọc từ thumbnail): hạng hiển thị `#3 / #43`,
`#88 / #389`, `#152 / #67`, `#65 / #43`, `#6,891 / #4,730`, `#2,200 / #4,047` — khớp
`limn_dual_scope.json` từng ô; dòng đích ở probe `7:3038 / shirt / MM` là `#3`, đúng vị trí gallery
`13,614`, khớp `target_gallery_index` của audit artifact.

### 4.3 Production code đã đổi

| file | thay đổi | lý do |
|---|---|---|
| `workbench/backend/limn_diagnostic.py` | mới | reader fail-closed cho evidence chẩn đoán |
| `workbench/backend/main.py` | +2 endpoint | phục vụ evidence đó |
| `workbench/frontend/LimnScopeDiagnostic.tsx` | mới | panel so sánh hai scope |
| `workbench/frontend/SamplesPage.tsx` | +1 dòng import, +1 `<LimnScopeDiagnostic />` | render panel |
| `workbench/frontend/style.css` | +4 dòng | chặn tràn ngang ở mobile |
| `workbench/tests/test_limn_dual_scope_diagnostic.py` | mới, 10 test | 10 passed |

**Không** đổi: `composition/model_limn.py`, `composition/data.py`, `composition/runner.py`,
evaluator, ngưỡng, hay bất kỳ metric nào. Canonical vẫn `6001/6003`, `PARTIAL`.
