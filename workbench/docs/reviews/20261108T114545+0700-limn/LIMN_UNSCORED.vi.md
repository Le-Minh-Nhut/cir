# LIMN — 119 query không được chấm trên cohort CIRCUS FashionIQ

Thời điểm điều tra: 2026-11-08. Phạm vi: phân tích artifact trên đĩa + semantics adapter. **Không chạy GPU eval.**

## 0. Kết luận ngắn

**Giải thích cũ sai một nửa.** Đúng phần "có 119 query bị bỏ" và đúng phần "danh tính *split* là mơ hồ".
Sai phần then chốt: **category hoàn toàn xác định được từ dữ liệu chính thức** cho **117/119**. Chỉ **2/119**
là mơ hồ thật.

Bằng chứng quyết định: text query M-BEIR task-7 **chính là hai caption validation của ảnh tham chiếu ghép
bằng `" and "`** — 6003/6003 query tái dựng được đúng như vậy từ `data/FashionIQ/captions/cap.<cat>.val.json`.
Nghĩa là hàng caption sinh ra text đó nằm trong file của một category cụ thể, và đó là category mà tác giả
FashionIQ đã gán. Split membership mơ hồ; annotation thì không.

Sau khi sửa: **6001/6003 chấm được, 2 không chấm được** (thay vì 5884/6003). Coverage 0.999667 thay vì 0.980176.
Hai query còn lại (`7:3038`, `7:3790`) có cặp caption **giống hệt nhau** được nộp dưới cả `shirt` lẫn `toptee` —
đây là mơ hồ thật, và code **không đoán**.

## 1. Tái lập hiện tượng từ artifact thật

Run được ghi trong báo cáo: `run_id = 10173dc45dea8c90fd2a284b`. **Vẫn tồn tại**, khớp từng byte với summary:

| Trường | Giá trị |
|---|---|
| summary | `workbench/artifacts/composition/summary/limn_base_iter0_all_categories_10173dc45dea8c90fd2a284b/full.json` |
| ranks | `workbench/artifacts/composition/runs/10173dc45dea8c90fd2a284b/ranks/limn_base_iter0_all_categories.json` |
| `checkpoint_id` | `base_iter0_all_categories` |
| `adapter` | `workbench.composition.model_limn.Adapter` |
| `category_scopes` | `['dress', 'shirt', 'toptee']` |
| `query_count` / `gallery_count` | 6003 / 74381 |
| `execution_status` | `PARTIAL` |
| scored / coverage (cả 3 condition) | 5884 / `0.980176578377478` |

Tập bị bỏ lấy trực tiếp từ `query_failures` của ranks artifact, **không** lấy từ danh sách `ambiguous_qids`
trong summary. Hai nguồn **bằng nhau tuyệt đối** ở cả ba condition:

```
ambiguous_count = 119, unique = 119
image_only_empty_text   ranks=6003  failures=119  failed == ambiguous: True
multimodal              ranks=6003  failures=119  failed == ambiguous: True
text_only_black_image   ranks=6003  failures=119  failed == ambiguous: True
failures identical across conditions: True
```

Toàn bộ 119 message có **cùng một template**, không có ngoại lệ:

```
query <qid>: category unresolved has no adapter embedding scope ('dress', 'shirt', 'toptee');
batch_indices=(<i>,); qids=['<qid>']
```

(357 = 119 × 3 condition, một loại message duy nhất.)

## 2. Nhân quả: mỗi query chỉ có **một** nguyên nhân, và đó là nguyên nhân đã ghi

Đã kiểm tra từng khả năng thay thế trên toàn bộ 6003, không chỉ trên 119:

| Nguyên nhân khác | Số query | Cách kiểm |
|---|---|---|
| Ảnh tham chiếu thiếu trên đĩa | **0** | `runner.image_resolver` resolve 6003/6003 |
| Positive target không có trong gallery | **0** | map qua 74381 gallery id của aggregated audit |
| Record malformed (qid/text/pos thiếu) | **0** | `load_release` raise nếu có |
| Gallery path trùng / thiếu | **0** | 74381 id, 74381 unique basename |
| Category nằm ngoài scope adapter | **0** | chỉ có `None` hoặc ∈ 3 category |
| `None` category | **119** | đúng 119 |

→ **Không phải đa nguyên nhân.** Câu chuyện một nguyên nhân là đúng *về hình thức*; điều sai là hệ quả
"không xác định được category" mà nó suy ra.

## 3. Phạm vi checkpoint — bằng chứng gốc, không phải ký ức

`workbench/registry/models.yaml` (model `limn`, dòng 1374–1407), khớp với bundle record nạp bằng
`registry.checkpoint_by_id(model_by_id("limn"), "base_iter0_all_categories")`:

```
bundle_id            = base_iter0_all_categories
artifact_type        = bundle_placeholder
required_checkpoint_ids = [base_iter0_dress, base_iter0_shirt, base_iter0_toptee]
mapping_status       = VERIFIED_METADATA
```

| checkpoint_id | filename | sha256 (16 ký tự đầu) | size |
|---|---|---|---|
| `base_iter0_dress` | `0_dress_best_model.pt` | `96a335d4eb8b1e77` | 1792010157 |
| `base_iter0_shirt` | `0_shirt_best_model.pt` | `9260c7f77d3017d8` | 1792010157 |
| `base_iter0_toptee` | `0_toptee_best_model.pt` | `79f663a14d89d455` | 1792010157 |

Ba file có mặt thật, mỗi file 1709.0M, tại
`workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads/limn/`.
Đo độc lập từng category (cùng file checkpoint, cùng hash) trong
`.../cohort/measurements/limn_base_iter0.json`: dress r10=51.1651, shirt r10=55.9863, toptee r10=60.2244,
`complete: true`.

Ánh xạ category → `dataset_type` là **hằng số của upstream**, không phải suy đoán:

- `workbench/third_party/LIMN/LIMN/datasets.py:17-31` — `dataset_name_dict = {'dress': 0, 'shirt': 1, 'toptee': 2, ...}`
- `workbench/composition/model_limn.py` worker — `dataset_ids = {"dress": 0, "shirt": 1, "toptee": 2}`

→ **Đúng 3 scope, không có scope thứ tư, không có scope mặc định.** Một query không thuộc 3 scope này thì
không có checkpoint nào để chấm — đó là lý do *kỹ thuật* của việc bỏ, và nó vẫn đúng.

## 4. Category có xác định được không? — Có, cho 117/119

### 4.1 Cơ chế: text query tái dựng được từ caption validation

M-BEIR task-7 dựng text query bằng cách ghép hai caption validation của ảnh tham chiếu. Chuẩn hoá
(casefold, bỏ dấu câu, gộp khoảng trắng) rồi ghép `" and "` cho ra **khớp chính xác 6003/6003** query:

```
dress/val 2017 hàng · shirt/val 2038 hàng · toptee/val 1961 hàng  (6016 hàng, 5973 text phân biệt)
recon over val: {(1,('dress',)):2010, (1,('shirt',)):2035, (2,('shirt','toptee')):2, (1,('toptee',)):1956}
misses = 0
```

Cho cả 6003 query, so với category mà release loader đã gán: **0 lệch** trong 5884 query đã gán.
Nghĩa là quy tắc này **không mâu thuẫn** với quy tắc split hiện có ở bất kỳ query nào — nó chỉ trả lời
được thêm 117 query mà split không trả lời được. Đó là điều kiện để coi nó là mở rộng hợp lệ, không phải
một quy ước thứ hai.

### 4.2 Ba nguyên nhân, phân loại từng query

`A` = loader thấy ảnh trong **một** val-split, nhưng **test-split của category khác** cũng chứa nó →
loader gán mơ hồ, dù val-split nói rõ một hướng.

`B` = ảnh nằm trong **cả hai** val-split (`shirt` và `toptee`), nhưng chỉ **một** category có hàng caption
sinh ra đúng text này.

`C` = cặp caption **giống hệt** được nộp dưới cả hai category → mơ hồ thật.

| Nhóm | Số query | → `shirt` | → `toptee` | không xác định |
|---|---|---|---|---|
| **A.** Lệch split val/test | **34** | 14 | 20 | 0 |
| **B.** Ảnh trong cả hai val-split, một hàng annotation | **83** | 32 | 51 | 0 |
| **C.** Cặp caption trùng khít dưới cả hai category | **2** | 0 | 0 | **2** |
| **Tổng** | **119** | **46** | **71** | **2** |

Cross-tab thô (val tags, test tags, test-caption tags) → quyết định:

```
 51  (('shirt','toptee'), (),        (),        ) -> toptee
 32  (('shirt','toptee'), (),        (),        ) -> shirt
 20  (('toptee',),        ('shirt',), ('shirt',)) -> toptee
 14  (('shirt',),         ('toptee',),('toptee',)) -> shirt
  2  (('shirt','toptee'), (),        (),        ) -> KHÔNG XÁC ĐỊNH
```

Ví dụ cụ thể mỗi nhóm:

- **A** — `7:2179`, ảnh `B0020Q11LE`. `split.shirt.val.json` chứa ảnh này; `split.toptee.test.json` cũng chứa.
  Text: *"Has writing on the t-shirt and is gray with black lettering."* = `cap.shirt.val.json` hàng
  `["has writing on the t-shirt", "is gray with black lettering"]`. → `shirt`.
- **B** — `7:4066`, ảnh `B004D8QCN4`. Có trong cả `split.shirt.val.json` và `split.toptee.val.json`.
  Nhưng chỉ `cap.toptee.val.json` có hàng `["is navy blue with a mustang on it", "is a lighter color"]`;
  `cap.shirt.val.json` **không có hàng nào** cho ảnh này. → `toptee`.
- **C** — `7:3038`, ảnh `B004TMGDME`. `cap.shirt.val.json` có `["is darker and has a different graphic", "is darker"]`
  và `cap.toptee.val.json` có **đúng cặp đó**. Không có bằng chứng nào phân biệt. → **không chấm**.

Một ca B khó hơn, vẫn xác định: `7:2486`, ảnh `B006VJXGAU` — `shirt` có 2 hàng, `toptee` có 3 hàng,
nhưng chỉ một hàng `shirt` tái dựng ra đúng text → `shirt`. Việc "category kia cũng có hàng caption" **không**
làm mất tính xác định; cái quyết định là hàng nào sinh ra text này.

### 4.3 Điều tra thêm: một phần "mơ hồ" là lỗi của loader, không phải của dữ liệu

Loader hiện tại (`composition/data.py:assign_garment_categories`) tính danh tính category bằng
`val-split ∪ ảnh có mặt trong cap.<cat>.test.json`. Thay bằng **val-split thuần** thì resolved tăng 5884 → 5918,
và **34 query mới** đều khớp 100% với kết luận từ annotation (34/34 agree, 0 disagree). Đó đúng bằng nhóm A.

Kết luận: nhóm A là hệ quả của việc trộn split val với test-caption; nhóm B mới là ca "ảnh thật sự nằm trong
hai category". Cả hai đều **không** cản việc xác định category từ chính annotation.

## 5. Thay đổi code

Chỉ trong `workbench/composition/model_limn.py` (file tôi sở hữu). **Không sửa** `runner.py`, `data.py`,
`registry/*.yaml`, hay ngưỡng đánh giá nào.

### 5.1 Thêm ba hàm thuần

```python
normalize_query_text(text)            # casefold + bỏ dấu câu + gộp space
caption_scope_index(fashioniq_root)   # {(image stem, text ghép) -> frozenset(categories)}
resolve_query_category(scopes, index, query, candidate_names)  # -> str | None
```

`caption_scope_index` khoá theo **cả ảnh và text**, không chỉ text: hai ảnh khác category có thể trùng caption
nguyên văn (13 text như vậy tồn tại trong release), và danh tính ảnh tham chiếu là thứ phân biệt chúng.

`resolve_query_category` theo đúng thứ tự ưu tiên:

1. Loader **đã** gán category → trả lại nguyên văn nếu thuộc scope, còn không thì `None` (thiếu checkpoint là
   lỗ hổng adapter, không phải cớ để lấy annotation đè lên). **Không bao giờ diễn giải lại 5884 query đã gán.**
2. Chưa gán → tra `(stem, text)`; chỉ nhận khi **đúng một** category.
3. Còn lại (`0` hoặc `≥2` category, không có stem, không có candidate table) → `None`.

### 5.2 Thêm seam trên adapter

```python
Adapter.candidate_names()   # lazy load cand_pool.jsonl -> {did: img_path}
Adapter.query_scope(query, candidate_names=None) -> str | None
```

Seam có thể tự lấy candidate table từ `WORKBENCH_CIRCUS_RELEASE_ROOT`, nên runner không phải truyền thêm gì.

### 5.3 Vì sao không đổi ngữ nghĩa đánh giá

`encode_queries`, `encode_targets`, `score`, `rank` **không bị chạm một dòng nào**. Cụ thể:

- **Cùng checkpoint**: mỗi query vẫn vào đúng một trong ba whole-model đã có; không thêm, không thay file nào.
- **Cùng tiền xử lý**: transform lấy từ `model.backbone.clip.visual` như cũ; `dataset_type` vẫn là hằng số
  upstream `{dress:0, shirt:1, toptee:2}`.
- **Cùng tie policy**: `rank()` dùng nguyên biểu thức
  `1 + count(score > s) + count(score[:i] == s)`; không sửa.
- **Cùng gallery**: vẫn nguyên 74381 ảnh cho mỗi scope, không lọc candidate theo category, không trộn ma trận
  giữa các scope.
- **Không nhìn positive target**: quyết định chỉ dùng text query + danh tính ảnh tham chiếu. Test
  `test_annotation_never_contradicts_the_loader_on_the_other_5884` chứng minh tính extend-only.

Cái duy nhất đổi là **nhóm `scope -> [(query, ...)]`**, tức là một hàng query được chấm bằng ma trận gallery
nào trong ba ma trận có sẵn. Đúng phạm vi được yêu cầu, không hơn.

### 5.4 Còn phải sửa ở `runner.py` (file của agent khác)

`runner._scope_for_query` hiện đọc thẳng `query.category`. Tôi **không sở hữu** file đó nên để yêu cầu cụ
thể ở `RUNNER_REQUESTS.md` cùng thư mục. Hai điểm, điểm thứ hai dễ bỏ sót:

1. Cho `_scope_for_query` ưu tiên `adapter.query_scope(query)` nếu có.
2. **Đưa scope đã resolve vào identity của run.** `_query_fingerprint` khoá theo `query.category` (giá trị của
   loader, vẫn `None`), nên fingerprint **không đổi**; ranks cũ vẫn khớp; và vì reason nằm trong
   `_STABLE_FAILURE_MARKERS`, resume sẽ báo `REUSED` với đúng 119 null cũ — **bản sửa sẽ vô hình**. Phải thêm
   digest của `query_scopes` vào identity để `run_id` đổi.

Cho đến khi (1) và (2) được áp, artefact trên đĩa vẫn là 5884; con số 6001 là kết quả của đường routing, đã
được test chứng minh, chưa được chấm lại.

## 6. Test

`workbench/tests/test_limn_scope_routing.py` — **13 test, tất cả pass** (1.84 s), không cần GPU, không cần
checkpoint:

```
13 passed in 1.84s
```

Chạy cùng bộ liên quan (`test_limn_scope_routing`, `test_circus_resume_stable`, `test_component_boundaries`,
`test_circus_data`, `test_circus_adapters_blip`):

```
74 passed in 125.42s
```

Nội dung:

| Test | Điều nó chặn |
|---|---|
| `test_normalize_query_text_joins_like_the_release` | chuẩn hoá đúng kiểu release |
| `test_caption_index_keys_on_image_and_text` | cùng text khác ảnh; cặp trùng dưới 2 category → frozenset 2 |
| `test_caption_index_skips_missing_and_malformed_annotation_files` | file thiếu/hỏng → index rỗng, không crash |
| `test_loader_assigned_category_is_never_reinterpreted` | loader gán `shirt` thì annotation `toptee` **không** được đè |
| `test_annotation_resolves_the_gap_the_loader_left` | 2 ca resolve đúng `shirt`/`toptee` |
| `test_text_filed_under_two_categories_stays_unscored` | **không mặc định** khi mơ hồ |
| `test_unknown_reference_or_text_stays_unscored` | stem lạ / text lạ / không có candidate table → `None` |
| `test_text_shared_by_two_images_follows_its_own_reference` | text trùng giữa dress và shirt không rò category |
| `test_scope_outside_declared_scopes_is_refused` | category lạ (`'shoes'`) → `None` |
| `test_adapter_seam_is_pure_routing` | seam trả lời chỉ từ file annotation; index cache 1 lần |
| `test_all_119_unscored_queries_are_classified` | **đúng 119**, **117 xác định**, `{7:3038, 7:3790}` không xác định, tập scope = `{shirt, toptee}` |
| `test_annotation_never_contradicts_the_loader_on_the_other_5884` | **0** query đã chấm nào đổi scope |
| `test_rescoring_arithmetic_closes_at_6003` | `scored=6001, unscored=2, tổng=6003`, không bỏ ngầm |

Hai test cuối là test số học kế toán — thứ báo cáo cần. Nếu dữ liệu release/annotation vắng mặt, chúng
`pytest.skip` (artifact ngoài repo) chứ không fail giả; 11 test đầu chạy trên fixture tự dựng nên luôn chạy.

## 7. Kế toán chính xác cho báo cáo

Cohort: `fashioniq` `full`, 6003 query, gallery 74381, 3 condition, cùng một tập query cho cả 3.

| | Trước | Sau khi routing (đã test, **chưa** chấm lại) |
|---|---|---|
| scored | 5884 | **6001** |
| unscored | 119 | **2** |
| coverage | 0.980176578377478 | **0.9996668…** |
| `execution_status` dự kiến | `PARTIAL` | `PARTIAL` (vẫn còn 2) |

Phân rã 119 theo nguyên nhân, kèm một qid mẫu mỗi loại:

| Nguyên nhân | Số | Ví dụ | Sau khi sửa |
|---|---|---|---|
| A. Lệch split val/test (lỗi loader, không phải mơ hồ dữ liệu) | 34 | `7:2179` | chấm, scope `shirt` |
| B. Ảnh trong cả hai val-split, annotation chỉ một hàng | 83 | `7:4066` | chấm, scope `toptee` |
| C. Cặp caption trùng khít dưới cả `shirt` và `toptee` | 2 | `7:3038` | **vẫn không chấm** |

Danh sách đầy đủ từng query: `CLASSIFICATION_119.md` (bảng) và `classification_119.json` (máy đọc) trong
cùng thư mục. Không query nào trong 119 bị bỏ vì lý do khác (ảnh thiếu, record hỏng, id ngoài gallery: tất cả = 0).

## 8. Còn gì phải chạy GPU mới xác nhận được

1. **Rank thật của 117 query.** Routing được chứng minh; **giá trị rank** thì chưa. Chưa ai chấm 117 query này
   bằng 3 checkpoint. nDCG/mRR/R@k sau sửa là **chưa biết** — không suy ra được từ test.
2. **Coverage thực tế = 6001/6003.** Đây là số học của routing; chỉ một lần chạy thật mới xác nhận là cả 117
   query encode được và không rơi vào failure khác (ví dụ feature non-finite) ở bước sau.
3. **`run_id` mới.** Sau khi `runner.py` thêm digest scope vào identity, phải chạy lại để sinh run dir mới;
   artifact cũ `10173dc45dea8c90fd2a284b` phải được giữ nguyên như bản ghi lịch sử của hành vi cũ.
4. **Ảnh hưởng lên metrics tổng.** 117 query mới, phân bố 46 `shirt` / 71 `toptee`, sẽ dịch `composition_gap_ndcg`,
   `coverage`, `gap_cohort_count` (5884 → 6001). Độ lớn và dấu **chưa biết**.
5. **Nhóm C.** Hai query đó vẫn unscored sau sửa. Muốn chấm chúng thì phải **chọn** một category — đó là đoán,
   và tôi không làm. Nếu về sau có nguồn chính thức khác (ví dụ nhãn category của M-BEIR task-7 gốc), thêm vào
   index là đủ, không phải sửa logic.

## 9. Tóm tắt trả lời

- **Giải thích cũ: SAI** ở kết luận "không xác định được category". Đúng ở chỗ "119 query bị bỏ" và "split mơ hồ".
- **117/119 xác định được** từ annotation chính thức; **2/119** mơ hồ thật.
- **Đã sửa**: resolver thuần + seam trên adapter trong `model_limn.py`; ngữ nghĩa chấm điểm giữ nguyên.
- **Test**: 13/13 pass (74/74 cùng bộ liên quan) — `scored=6001, unscored=2, tổng=6003`, `0` query cũ đổi scope.
- **Còn lại**: `runner.py` phải gọi seam và đưa scope vào run identity; sau đó một lần chạy GPU để lấy rank thật
  và metrics của 117 query mới.
