# M-01 / M-02 / M-03 — xác minh độc lập và sửa lỗi (master audit)

Repo `/home/heheboiz/data/cir`, nhánh `research/cir-failure-workbench`, HEAD lúc bắt đầu
`92ed66de4f359ef479da18bf8939723b6f9313e5` (không commit). Nguồn yêu cầu:
`/home/heheboiz/Downloads/CIR_MASTER_FIX_AND_UPSTREAM_VERIFICATION.vi.md` §4 hàng M-01/M-02/M-03 và
§2 (quy tắc dataset/protocol). Toàn bộ bằng chứng thô nằm cạnh file này:

```
raw/m01/   forward_delta.json, forward_delta_runs.txt, proc_a.txt, proc_b.txt,
           tests_failing_first.txt, tests_encoder.txt, source_hash.txt
raw/m02/   m02_resolver.txt (HEAD), m02_resolver_after.txt, m02_recount.json, qid_scope_map.json,
           m02_upstream.json, probe_resolver_before_after.txt, tests_failing_first.txt,
           tests_passing_after.txt
raw/m03/   cells_recompute.txt, cells_did_dump.txt, leafdiff.txt, generations.txt,
           validator_probe_before.txt, validator_probe_after.txt, intersection.txt,
           tests_failing_first.txt, tests_limn.txt
scripts/   m01/*.py  m02/*.py  m03/*.py
```

Probe CPU-only `OMP_NUM_THREADS=1-2`. Một job GPU **đã** chạy (ENCODER rerun được duyệt) và một job
CPU dài (transform control) chạy sau khi job GPU kết thúc; chi tiết lock ở §A.0.
Máy: RTX 5070 Ti (16 303 MiB), `nvidia-smi` không có compute process suốt lượt chạy.

---

## Bảng tổng hợp

| issue_id | status | file:line | failing-first | fix | regression | affected_run_ids |
|---|---|---|---|---|---|---|
| **M-03-a** — export khai `depth=100` nhưng lưu ít hàng hơn (truncation) | **FIXED_VERIFIED** | `workbench/backend/limn_diagnostic.py:297-307` | `raw/m03/validator_probe_before.txt`: cắt `candidates[:1]` với `depth=100` vẫn `DIAGNOSTIC_COMPLETE`, 0 defect; `raw/m03/tests_failing_first.txt` (2 failed trên HEAD) | thêm `export_depth_truncated`, biên trên là `min(depth, gallery_size)` của chính hàng đó | `raw/m03/tests_limn.txt` → 74 passed | không đổi số của run nào (validator chỉ đọc); run đã xuất bản không bị sửa |
| **M-03-b** — hai hàng cùng một `qid` trong một export bị bỏ im lặng | **FIXED_VERIFIED** | `workbench/backend/limn_diagnostic.py:179-192` | `raw/m03/validator_probe_before.txt` case `duplicate_qid_row`: thêm hàng thứ hai (rank +1000) vẫn `COMPLETE`, `next(...)` lấy hàng đầu | thêm `export_duplicate_qid`, đếm theo `(scope, condition)` trên mọi file | `raw/m03/tests_limn.txt` → 74 passed | như trên |
| **M-03-c** — 12 ô + r5 patch | **CONFIRMED (đúng như tuyên bố)** | `workbench/docs/reviews/20261109-native15-composition-limn/limn_dual_scope.json` | — | không sửa: dữ liệu đã đúng | `raw/m03/cells_recompute.txt`, `raw/m03/leafdiff.txt` | `9e4918a6b5c243412cbc58c9` |
| **M-03-d** — generation mixing Full/SF/Validated | **FIXED_VERIFIED (derive, không GPU; đã publish + API 200; status theo option (b))** | `workbench/artifacts/composition/summary/limn_base_iter0_all_categories_9e4918a6b5c243412cbc58c9/` | `raw/m03/generations.txt` | `scripts/m03/derive_subset_runs.py` → `raw/m03/derived_runs/<id>/` | `raw/m03/derived_subset_runs.txt`, `raw/m03/derived_subset_runs.json` | `9e4918a6…` (parent) → derived `6a96c5c1…` (sf), `aee2e8d3…` (validated) |
| **M-03-e** — `6001/6003` là coverage, gap trên giao ba condition | **CONFIRMED (đã đúng), wording đã ghi rõ** | `.../full.json:metrics.gap_cohort_count` | `raw/m03/intersection.txt` | không sửa code; ghi lại định nghĩa trong báo cáo này | — | `9e4918a6…` |
| **M-02-a** — resolver chọn category khi nguồn annotation thiếu/hỏng | **FIXED_VERIFIED** | `workbench/composition/model_limn.py:59-108` (`CaptionScopeIndex.unreadable`), `:140-141` | `raw/m02/probe_resolver_before_after.txt`: HEAD trả `'shirt'`/`'shirt'`/`'toptee'`; `raw/m02/tests_failing_first.txt` (2 failed trên HEAD) | `caption_scope_index` ghi tên file không đọc được; `resolve_query_category` trả `None` khi index thiếu nguồn | `raw/m02/tests_passing_after.txt` → 74 passed | `9e4918a6…` (không đổi: 117 vẫn giải được vì cả ba file đọc được) |
| **M-02-b** — 117 query phục hồi lịch sử | **CONFIRMED (đúng), xác minh độc lập** | `raw/m02/qid_scope_map.json` | — | không sửa | `raw/m02/m02_recount.json` | `9e4918a6…` |
| **M-02-c** — upstream không có khoá garment category | **CONFIRMED (đã đọc source upstream)** | `TIGER-AI-Lab/UniIR@98dc570c` `src/data/preprocessing/fashioniq_data_preprocessor.py` | — | không sửa | `raw/m02/m02_upstream.json` | — |
| **M-01-a** — `learnable_relation` sinh không seed, không vào checkpoint | **`FIXED_VERIFIED`** — code đúng (pin + persist đã chạy thật) **và** đã có run canonical mang cả identity lẫn tensor | `workbench/composition/model_encoder.py:42-63` (hằng số), `:148-157` (nơi persist), `:262` (`_adopt_scratch_relation`), `:340-380` (pin + persist) | `raw/m01/tests_failing_first.txt` (4 failed trên HEAD); `raw/m01/forward_delta.json` | pin `torch.manual_seed(20261111)` ngay trước `Encoder(...)`, ghi tensor + sha256 vào `<run>/construction_state/` (bản đầu dưới `work/` bị cleanup xoá — xem §A; guard persist bỏ run đầu vì handshake là job đầu — đóng ở `34aaa10`) | 64 passed (`raw/m01/tests_encoder.txt`); 2 process sạch cùng seed → gallery/query feature **bit-identical**; canonical `acc9b20e1aea48487f65f944`: `relation_state.sha256` `20db0c27…f9ce92` = `expected_sha256`, `matches_pinned_digest true`, **và** `runs/acc9b20e…/construction_state/relation-seed20261111.npy` (+`.json`) | `ce89a9a27c4cceadeaaf30b0` đo được số (giống hệt `acc9b20e`) nhưng id hash `relation_state: not_recorded`; `9b806c22…` recorded nhưng thiếu tensor; `5dee7425dd46c5f154808aac`, `6a528e877c328772e2c7bb28` (relation drawn from an unpinned generator), `eda30a37c0bc17ac43844f09` — tất cả superseded (§A; census §0.6) |
| **M-01-b** — train transform cho query, val cho gallery | **CONFIRMED; đo như adaptation sensitivity, KHÔNG đổi production** | `workbench/composition/model_encoder.py:390-400` (`preprocess_val` cho cả hai) | `raw/m01/` không có; số đo cũ ≤0.3 R@10, dấu không nhất quán (`ENCODER_PARITY.vi.md:376-382`) | đề xuất diff, xem §M-01.2 | — | `5dee7425…`, `6a528e87…` |
| **M-01-c** — cold start / warm-cold cache | **CONFIRMED; rerun mới cold + `cache_verify_mode=full`** | `workbench/composition/runner.py:1612-1618` (`seed_applied: False`) | §M-01.4 | xem §M-01.4 | — | `5dee7425…` (cache `full`), `6a528e87…` (cache field thiếu) |

Ghi chú trạng thái: `FIXED_VERIFIED` = có lệnh fail trước / pass sau trên HEAD thật; `CONFIRMED` =
xác minh được bằng dữ liệu, không có gì phải sửa; `BLOCKED` = thiếu artifact trên host, ghi rõ thiếu gì.

---

## M-03 — artifact và subset của LIMN

### M-03.1 Mười hai ô `qid × scope × condition` (recompute độc lập)

Nguồn join: `data.gallery.image_ids` (thứ tự release từ `shortcut_audit/…aggregated_best_rank_mm_any.json`),
`data.candidate_names` (bảng candidate), và `positive_ids` của **chính** qid đó từ
`workbench/composition/data.py::load_release` — join theo **candidate ID**, không theo position, không
theo vector.

```text
7:3038  release positives ('7:3416',)      asset B004D8QCN4.jpg  pos 13614
7:3790  release positives ('7:42304',)     asset B00E3WVAD2.jpg  pos 70410

7:3038  shirt   multimodal rank   3 | text_only_black_image 88 | image_only_empty_text 152
7:3038  toptee  multimodal rank  43 | text_only_black_image 389| image_only_empty_text  67
7:3790  shirt   multimodal rank  65 | text_only_black_image 6891| image_only_empty_text 2200
7:3790  toptee  multimodal rank  43 | text_only_black_image 4730| image_only_empty_text 4047
```

Đã kiểm từng ô (`raw/m03/cells_recompute.txt`, 12/12 PASS):

- `target_did` của mỗi ô là positive **của chính** qid đó; `target_gallery_id` map về đúng did đó;
- `target_gallery_position` khớp `gallery_paths` của release (join theo ID) và nằm trong
  `target_position_in_gallery` của bản ghi query;
- với mọi hàng trong cửa sổ: `gallery_paths[gallery_position] == gallery_id` (0 mismatch), rank liên
  tục 1..100, score không tăng, không trùng `gallery_id`, không trùng `gallery_position`,
  `image_path == gallery_id`;
- `target_rank`/`target_score`/`target_margin_vs_rank1` của summary khớp hàng `is_target` khi target
  nằm trong cửa sổ (6 ô), và nằm ngoài cửa sổ đúng như khai khi rank > 100 (6 ô);
- `score_at_rank_1` khớp hàng rank 1; `checkpoint_sha256` của export khớp `checkpoints[scope].sha256`
  (`9260c7f7…` shirt, `79f663a1…` toptee).

Không thể recompute **score** từ đầu trên host: hàng của hai qid này là NaN trong cả 6 cache query của
run `9e4918a6…` (đúng như thiết kế — chúng không được chấm canonical). Recompute rank từ ma trận
gallery cần query embedding, tức cần GPU ⇒ **không chạy**, xem §GPU.

### M-03.2 Sửa r5 (6 ô của `7:3038`)

Leaf diff có cấu trúc giữa `limn_dual_scope.json` và `limn_dual_scope.json.pre-r5`
(`raw/m03/leafdiff.txt`):

```text
leaves old/new 560/616 (bản của worker M-03: 302->314 sau khi bỏ 2 block lớn)
added 56   removed 0   changed 6
changed = đúng 6 leaf target_did của 7:3038 (2 scope × 3 condition), 7:42304 -> 7:3416
added   = 56 leaf của riêng block provenance_correction_r5 (không leaf nào ngoài block này)
sha256(pre-r5) = 04c4419e44ff555b909ad92ed0a8596520a0cb5cdcfc0467986baabf77bfabf0
                 == provenance_correction_r5.artifact_sha256_before  (khớp)
sha256(hiện tại) = d1b4aadb6735b8b3c57e9fbfe234b1e12c15a13b977ace23500916af94473e3f
```

Không byte nào của rank/score/position/asset thay đổi; `changed` chỉ có `target_did`. Bản gốc còn
nguyên (`limn_dual_scope.json.pre-r5`, 27 311 byte).

### M-03.3 Guard còn thiếu (đã sửa, failing-first)

Hai lỗ mà guard cũ bỏ lọt, cả hai **tái hiện được** và nay là defect có mã:

| mã defect mới | kịch bản | HEAD | sau sửa |
|---|---|---|---|
| `export_depth_truncated` | `depth=100`, `candidates[:1]` | `COMPLETE`, 0 defect | `DIAGNOSTIC_EVIDENCE_DEFECT` |
| `export_duplicate_qid` | thêm hàng thứ hai cùng qid (rank +1000) | `COMPLETE` (lấy hàng đầu) | `DIAGNOSTIC_EVIDENCE_DEFECT` |

`raw/m03/validator_probe_before.txt` / `_after.txt` là cặp before/after. Regression:
`test_a_window_shorter_than_its_declared_depth_is_a_defect`,
`test_a_duplicate_export_row_for_one_qid_is_a_defect` (2 failed trên HEAD:
`raw/m03/tests_failing_first.txt`; 74 passed sau sửa). Fixture dương tính của hai file test cũ khai
`depth=100` trong khi chỉ lưu 1–4 hàng — bản thân nó là lỗi truncation, nên fixture được sửa để khai
đúng số hàng nó lưu (`test_limn_diagnostic_record.py:117-126,156`,
`test_limn_dual_scope_diagnostic.py:97`).

Các guard đã có sẵn và vẫn đúng: `export_depth_short`, `export_rank_not_sequential`,
`export_scores_not_descending`, `export_duplicate_candidate`, `export_duplicate_position`,
`export_multiple_targets`, `target_did_not_own_query`, `checkpoint_summary_vs_export`,
`rank_summary_vs_export`, `score_summary_vs_export`, `export_row_identity`, `per-scope export missing`
(refuse, không phải defect).

### M-03.4 Manifest/digest, validator dùng chung, duplicate, depth

```text
validator (load_diagnostic)  status=DIAGNOSTIC_COMPLETE complete=True defects=0 standard=limn-diagnostic-evidence/v2
manifest digest              9d1018524a8931…: 7 file, sha256 từng file khớp bytes trên đĩa (worker recompute độc lập)
depth                        12/12 hàng khai depth=100, lưu 100, 0 truncated, 0 over-long
top100_export                6 tên phân biệt, đúng bộ mong đợi, mỗi file 2 hàng, không file nào thiếu
duplicate qid trong export   0 (mỗi file đúng 2 hàng); 0 trùng gallery_id; 0 trùng gallery_position
summary vs candidates        cùng một `_load_record`, cùng manifest digest ⇒ không thể lệch nhau
```

### M-03.5 Generation mixing Full/SF/Validated

```text
9e4918a6b5c243412cbc58c9  full.json 6003 query / 6001 chấm / mm R@10 30.0283   PARTIAL  <- canonical
                          sf.json, validated.json  KHÔNG TỒN TẠI
10173dc4dea8c90fd2a284b  full 6003/5884  sf 4069/4000  validated 586/581     PARTIAL  (run id khác)
3f86f58483b368377e2c151e  smoke (full 8, sf 6, validated 0)                   COMPLETED
```

- `source_run_id` trong `limn_dual_scope.json` == `run_id` của `full.json` == `9e4918a6…`; cohort
  (`full`, 6003, 6001, 2, 30.0283) khớp từng trường.
- Cùng `checkpoint_identity 12cbbf95…` và `gallery_identity 1db68c59…` ở cả ba generation, nên "trộn
  đời" ở đây **không** phải trộn checkpoint/gallery — nhưng **có** trộn run id: sf/validated của
  model LIMN chỉ tồn tại ở `10173dc4…`, một run khác với canonical `9e4918a6…`.
- Vì vậy một trang hiển thị `sf`/`validated` của LIMN **không** phải là subset của run canonical.
  Đây là `BLOCKED`: canonical run chỉ có `full`; để có sf/validated canonical phải chạy lại
  `sf,validated` trên `9e4918a6…` (GPU, xem §GPU). Không suy subset từ parent khác.

### M-03.6 Coverage và metric

```text
6001/6003 = coverage của full cohort (99.9667%); 2 qid không chấm (7:3038, 7:3790)
giao ba condition = 6001 (union = 6001); không condition nào loại thêm qid nào
metrics.gap_cohort_count = 6001  -> gap ba condition được tính trên đúng 6001, không phải 6003
per condition: query_count 6003, scored_count 6001, coverage 0.9996668…
```

Cách viết đúng: **"coverage 6.001/6.003; metric tính trên 6.001 query đã chấm; gap multimodal ↔
image-only ↔ text-only tính trên giao ba condition = 6.001"**. Không được viết "độ phủ metric
6.001/6.003" như thể mẫu số của metric là 6.003, và không được bịa một scope cho hai qid để lấp
khoảng trống.

---

## M-02 — routing LIMN

### M-02.1 Ba file được pin

| file | sha256 |
|---|---|
| `data/FashionIQ/captions/cap.dress.val.json` | `5e5117d45695df9c3ca91fac3e0bc49e6422ae9c8b3793f0ac9b3ab83adb7de9` |
| `data/FashionIQ/captions/cap.shirt.val.json` | `7b7ca3797b85dddfd83e74cdb4454cb63e3e7c09acef2590155efbacea6d7feb` |
| `data/FashionIQ/captions/cap.toptee.val.json` | `b4e09f428e6c255cc4b4ec46e77908305ab83075d82840c960be81af15cdc5eb` |

Release query file (đối chiếu membership): `mbeir_fashioniq_task7_test.jsonl`
`b601ef6a5411bf9385cff41e64bea863871ab1094cd09d78683de6a5141165ef` (= `EXPECTED_SHA256`, loader báo
`hash_verification: matched`). File hỗ trợ cho `loader_identity` (6 `split.*.val.json` +
`cap.*.test.json`) cũng có digest trong `raw/m02/qid_scope_map.json`.

### M-02.2 Fixture trùng ref+caption — defect đã xác nhận và đã sửa

```text
                                    HEAD                              sau sửa
a_both_present (shirt+toptée)       None  (đúng)                     None
b_toptée absent                     'shirt'  <- DEFECT                None
c_toptée corrupt                    'shirt'  <- DEFECT                None
d_shirt absent                      'toptee' <- DEFECT                None
```

Nguyên nhân: `caption_scope_index` bắt `OSError/ValueError` rồi `continue`, nên "file kia không có
hàng" và "file kia chưa từng được đọc" là cùng một trạng thái; index còn lại một category ⇒ resolver
coi là duy nhất. Sửa: index mang `unreadable` (`CaptionScopeIndex`), và `resolve_query_category`
trả `None` cho query **chưa được loader gán** khi `unreadable` khác rỗng. Query đã có category từ
loader vẫn giữ nguyên category đó (không đọc annotation), nên 5 884 query không đổi.

Ảnh hưởng thật (đo bởi worker M-02): nếu `cap.shirt.val.json` không đọc được → chỉ 73 query giải được
và cả hai qid mơ hồ thật bị gán sai sang `toptee`; nếu `cap.toptee.val.json` không đọc được → 48
query, hai qid bị gán sang `shirt`. Sau sửa: 0 query được route từ một bộ nguồn không đầy đủ.

### M-02.3 117 query phục hồi — xác minh độc lập

```text
ambiguous (query.category is None)      119
giải được đúng một scope                117   shirt 46 / toptee 71 / dress 0
còn lại                                  2   ['7:3038', '7:3790']
tái lập tuyên bố 117 = YES; hai qid còn lại đúng = YES
rederive từ bytes thô cap.<cat>.val.json cho từng qid: 117/117 PASS_<cat>_only
  (file quyết định có cặp (stem, text chuẩn hoá), hai file còn lại KHÔNG có)
parser độc lập == caption_scope_index trên cả 119
```

Mapping đầy đủ 6 003 query: `raw/m02/qid_scope_map.json` —
`loader_identity` 5 884 (dress 2 010 / shirt 1 989 / toptee 1 885),
`caption_annotation` 117 (shirt 46 / toptee 71), `ambiguous_reference_and_caption` 2.
Mỗi bản ghi mang `reason`, `source_files`, `caption_file_sha256`, `decided_by`; file ghi rõ scope nào
do loader quyết định và scope nào do annotation.

Điều **còn thiếu** (nói rõ, không lấp): không tồn tại metadata category per-query ở upstream, nên
"category mà tác giả gán cho query này" **không có nguồn**; cái đang có chỉ là category suy ra từ vị
trí ảnh trong split/annotation. Với `7:3038`/`7:3790` hai nguồn đó nói hai category như nhau ⇒ không
có scope canonical, canonical giữ `PARTIAL 6001/6003`. Hai scope diagnostic vẫn tách riêng, không
scope nào được chọn làm "đáp án", và không có ngưỡng score/rank nào tham gia.

### M-02.4 Upstream có khoá garment category không? — **REFUTED** (đã đọc source upstream)

```text
repo    TIGER-AI-Lab/UniIR
commit  98dc570c2645fec8e1fd42c7d4ac9853f46051fd
file    src/data/preprocessing/fashioniq_data_preprocessor.py (428 dòng, blob sha1 d815957c…)
sha256 của bytes đã đọc: 17091804658779d71c66049f5c12250123740c993ed4fde7b63315b0ed7e53b2

fashioniq_to_mbeir_entry() dựng đúng các khoá:
  qid, query_txt, query_img_path, query_modality, query_src_content, pos_cand_list, neg_cand_list
  query_src_content = {"candidate_img_id": ...}
generate_fashioniq_candidate_pool() dựng: did, txt, img_path, modality, src_content{img_id}
shipped row = các khoá trên + task_id (gán sau, trong mbeir_data_generator.py)
KHÔNG có: category, garment, subtask, source_file
```

Kết luận: pipeline release upstream **chưa bao giờ ghi** khoá garment category ⇒ per-query category
metadata **không tồn tại upstream**; `category` chỉ là suy luận của loader downstream từ vị trí ảnh.
Không có "fix-at-source".

---

## M-01 — ENCODER

### M-01.1 `learnable_relation`: sinh ở đâu, có seed không, vào đâu

```text
sinh       third_party/ENCODER/model_try2.py:93
           self.learnable_relation = nn.Parameter(torch.randn(N_p, dim, dtype=torch.float32)).cuda()
           shape [2, 512] float32, N_p=2, hidden_dim=512
loại       nn.Parameter(...).cuda() -> Tensor thường, không đăng ký (đo: raw/m01/parameter_cuda_type.txt):
           state_dict 415 entry, checkpoint 417 key, missing_keys [] , unexpected 2
           [backbone.binding_decoder.fc.0.weight, .fc.0.bias]  (model không đọc)
           named_parameters/buffers: không có 'relation'
vào đâu    binding() tại model_try2.py:119 -> dùng cho CẢ HAI:
             gallery  extract_img_fea -> binding (model_try2.py:375) -> extract_retrieval_target
             query    extract_img_fea_patch_selection -> binding cho image (:410) và text (:426)
                      -> compose_feature (:483) -> extract_retrieval_compose
seed       KHÔNG có seed nào trên đường composition: model_encoder.py không có manual_seed;
           runner ghi seed_applied: False (runner.py:1612-1618, 2112).
           replay/common.py:372-376 (seeded_dataset_construction) seed SAU construction và không
           được worker gọi. Thứ tự: Encoder(...) draw trước mọi bước khác của worker
           (model_encoder.py:340-344 nằm trước load_state_dict/transform).
```

Cơ chế "not registered" đo trực tiếp trên host (`raw/m01/parameter_cuda_type.txt`, torch 2.7.1+cu128,
CUDA available, không cần model):

```text
Parameter.cuda()        -> Tensor          (KHÔNG phải Parameter)
Parameter.to('cuda')    -> Tensor
class M(nn.Module): self.attr = nn.Parameter(...).cuda()
named_parameters        -> []              ('attr' không được đăng ký)
'attr' in state_dict    -> False
isinstance(attr, nn.Parameter) -> False
```

Tức `.cuda()` trả về `Tensor` thường, và `Module.__setattr__` chỉ đăng ký khi giá trị là `Parameter`
⇒ tensor không vào `state_dict()` (415 entry) và không thể được checkpoint (417 key) khôi phục;
2 key `backbone.binding_decoder.fc.0.{weight,bias}` trong checkpoint là key model không đọc.

Đo trên host (`raw/m01/forward_delta.json`, 4-5 process sạch, CPU bằng cách vô hiệu hoá `.cuda()`):

```text
unseeded, 2 process sạch:  relation max|Δ| 4.571284
                           gallery feature max|Δ| 0.000223288
                           query  feature max|Δ| 0.01039352
seeded (20261111), 2 process: relation 0.0, gallery 0.0, query 0.0, BIT-IDENTICAL
cùng seed, khác process: digest relation 20db0c27…ce92 ở cả hai (khớp RELATION_EXPECTED_DIGEST)
```

Construction probe (`raw/m01/proc_a.txt`, `proc_b.txt`, 4 process sạch): unseeded same-process
max|Δ| **5.012038230895996** và **4.727078914642334**; seeded same-process 0.0; seeded cross-process
cùng digest `20db0c27…ce92`.

Đo cũ được xác nhận lại: 4.923079013824463 (unseeded, `20261110-e2e-repair/raw/r4_determinism_p1.json`),
0.0 cùng seed, 4.862130165100098 khác seed, 417 key checkpoint / 415 state_dict / 2 key lạ — tất cả
khớp với probe mới. Số của lượt này **không** phải 4.923 (delta phụ thuộc draw, đúng bản chất): hai
lượt của tôi cho 4.11787748336792 / 4.744387149810791 (construction) và 4.571284 (đo trên feature).

Hai run đã xuất bản chứng minh hậu quả: `5dee7425…` vs `6a528e87…` khác nhau **2 819 / 6 003 rank**
multimodal (R@10 31.0345 vs 31.0178), 4 694 rank text-only, 5 214 rank image-only — không phải do
checkpoint (`0f31c501…` giống nhau) mà do relation khác nhau.

### M-01.2 Chọn cách xử lý

Chọn **pin trước construction + persist tensor kèm digest** (không chỉ pin, không "tuyên bố
non-reproducible"):

- `RELATION_SEED = 20261111`, `torch.manual_seed(...)` đặt ngay trước `Encoder(...)` trong worker;
- tensor đã draw được lưu nguyên bytes tại **`runs/<run_id>/construction_state/relation-seed20261111.npy`**
  (đường dẫn cuối cùng, sau sửa — `model_encoder.py:148-157`), cùng sidecar JSON (`shape`, `dtype`,
  `seed`, `sha256`, `seed_placement`, `in_state_dict: false`, `source`, `permutation_scope`,
  `transform_policy`). Đường dẫn ban đầu `<run>/work/encoder/construction_state/` **đã bị bỏ** vì
  cleanup keep-features của runner xoá nó; ai đọc bản cũ mà tra dưới `work/` sẽ không thấy gì và
  tưởng evidence bị dọn — xem §A;
- worker trả `relation_digest`/`relation_seed` trong reply; adapter ghi `self.relation_state` với
  `matches_pinned_digest` (so với digest pin). Lệch digest **không** bị ép thành pass: nó là một
  measurement mới và phải mint run id mới.

Không dùng `arange`/hằng số để "tái lập" — đó không phải weight đã phát hành. Không sửa vendored
`model_try2.py` (giữ nguyên byte của tác giả); pin nằm ở worker.

Về run id: `_module_source_hash` = sha256 của file adapter (`runner.py:1336-1342`), nên sau sửa
`model_encoder.py` đổi từ `bf60d855…` sang `df688248…` ⇒ mọi run ENCODER mới mint id mới. Hai artifact
`5dee7425…`, `6a528e87…` vì vậy là **superseded** cho mục đích tái lập, và **không** nên bị trộn với
run sau sửa.

### M-01.3 Transform: query dùng train, gallery dùng val

Upstream đã kiểm lại trong source vendor:

```text
evaluate_model.py:58   transform=[preprocess_train, preprocess_val]
datasets.py:147-153    get_img(name, stage) -> self.transform[stage]
datasets.py:177        source_img_data = get_img(..., stage=0)      # query  -> TRAIN
datasets.py:195,202    target_img_data = get_img(..., stage=1)      # gallery-> VAL
```

Composition adapter dùng `preprocess_val` cho **cả hai** (`model_encoder.py:390-400`). Đây là adaptation
đã khai (`ENCODER_PARITY.vi.md:47,428`), tách khỏi pin relation. **Không sửa trong lượt này** vì:
`runner.py` thuộc chủ khác, và thay đổi transform phải mint run id mới + đo lại toàn bộ (train
transform có `RandomResizedCrop` nên một reproduction lặp lại trong cùng process sẽ khác nhau trừ khi
khoá seed cho transform). Đề xuất (cho worker sở hữu `runner.py`/`native_measurements.py`):

```diff
--- a/workbench/composition/model_encoder.py
+++ b/workbench/composition/model_encoder.py
-        _, _, preprocess_val = open_clip_module.create_model_and_transforms(
-            "ViT-B-32", pretrained=str(source_root / "open_clip_pytorch_model.bin"))
+        preprocess_train, _, preprocess_val = open_clip_module.create_model_and_transforms(
+            "ViT-B-32", pretrained=str(source_root / "open_clip_pytorch_model.bin"))
...
-                    images.append(preprocess_val(image.copy()))
+                    transform = preprocess_train if current["phase"] == "queries" else preprocess_val
+                    images.append(transform(image.copy()))
```

Đi kèm bắt buộc: (a) transform là input số học nên phải vào run identity; (b) vì `preprocess_train`
ngẫu nhiên, seed cho nó phải được pin cùng chỗ với relation và ghi vào summary; (c) phải chạy lại
native control cùng pixels trước khi so số. Số đo cũ: ≤ 0.3 R@10, dấu không nhất quán ⇒ không được
"chọn cái cao hơn".

### M-01.4 Cold start và warm/cold cache

```text
5dee7425…  cache_verify_mode 'full'  reused_query_result_count 0  (cold, verify đầy đủ)
6a528e87…  cache_verify_mode None    reused_query_result_count 0  (artifact cũ trước khi có field)
```

- **Cold start**: cả hai run đều cold (`reused_query_result_count = 0`), nên chênh lệch 2 819 rank
  không thể do cache — nó do relation khác nhau.
- **Warm/cold cache**: trường `cache_verify_mode` chỉ tồn tại từ một revision; `6a528e87…` thiếu nó,
  nên không thể khẳng định run đó được verify `full` hay không. Đây là `BLOCKED` ở tầng dữ liệu lịch
  sử: **không backfill** giá trị chưa từng được ghi.
- Warm run sau sửa: cần chạy lại ENCODER (GPU). Vì `relation_state` giờ là một phần của trạng thái
  model chứ không phải cache, warm run phải kiểm `matches_pinned_digest` trước khi tái dùng feature;
  hiện `runner.py` (không thuộc lượt này) chưa so digest này — đề xuất đưa `relation_state` vào
  `identity` của runner để warm run không tái dùng feature của một draw khác.

### M-01.5 GPU

Không chạy job GPU nào. `nvidia-smi` suốt lượt: RTX 5070 Ti, 0 compute process, ~1 000 MiB do desktop.
Việc cần GPU (đề nghị, chờ duyệt):

| việc | ước lượng | căn cứ |
|---|---|---|
| ENCODER `full,sf,validated` × 3 condition sau khi pin relation (đo tái lập thật, 2 process) | **~10 phút GPU** (1 run full ≈ 4.75 phút + khởi động) | mtime run `5dee7425…`: birth 1794169855 → write cuối 1794170140 = 285 s; gallery 74 381 ảnh 201 s, 3×6 003 query 48 s |
| LIMN `full,sf,validated` × 3 condition × 3 scope để đóng M-03-d | **~2.3 giờ GPU** | mtime run `9e4918a6…`: birth 1794176766 → write cuối 1794182632 = 5 866 s cho `full` (gallery 3 scope 3 569 s + query 2 294 s); thêm 4 655 query ⇒ query stage ×1.776 ≈ 4 074 s, tổng ≈ 7 643 s |

Tổng đề nghị ≈ **2.5 giờ GPU**, tuần tự, một job một lúc. **Chưa chạy** — chờ duyệt.
Lưu ý: `subset_memberships` nằm trong `identity` (`runner.py:1608`), nên không thể chạy riêng `sf`/`validated`
rồi gắn vào `9e4918a6…`; phải chạy lại cả ba subset trong CÙNG một run (xem
`raw/m03/subset_rerun_cost.txt`).

---

## Bổ sung sau khi chủ dự án duyệt (2026-11-10)

### A.0 Kỷ luật GPU / lock

Job ENCODER rerun được khởi động **trước** khi quy tắc `flock /tmp/cir_gpu.lock` được ban hành, nên nó
chạy **không** giữ lock; đúng chỉ dẫn "nếu đã mid-run thì chạy xong rồi báo", nó được để chạy tới hết
(`START 2026-11-10T19:54:04+07:00`, GPU process duy nhất trong suốt thời gian chạy, `nvidia-smi`
`gpu_procs=0` cả trước lẫn sau). Mọi job GPU **về sau** phải:

```bash
flock -n /tmp/cir_gpu.lock -c '<command>'   # giữ lock suốt job; nếu bị giữ thì chờ, không kill job kia
nvidia-smi --query-compute-apps=pid --format=csv,noheader   # phải rỗng, kiểm TRONG lock
```

Không job GPU nào khác của tôi chạy trong lượt này; transform control là CPU và chạy sau khi GPU rảnh.

### A. ENCODER rerun — pin đã chạy thật

Run **đo lần đầu** (không phải canonical — xem cuối mục): **`ce89a9a27c4cceadeaaf30b0`** (`full,sf,validated` × 3 condition, `cache=full`,
`reused=0`, `COMPLETED`, `topk-depth 100`). Canonical nay là **`acc9b20e1aea48487f65f944`**, cùng số
này tới từng digit (delta `0.0`, 0/18009 rank khác). Số (`raw/m01/rerun_pinned_final.txt`):

| subset | condition | scored | coverage | R@10 | R@50 | nDCG |
|---|---|---|---|---|---|---|
| full | multimodal | 6003 | 1.0 | **31.0178** | 49.5586 | 0.295680 |
| full | text_only | 6003 | 1.0 | 8.5124 | 18.0243 | 0.155412 |
| full | image_only | 6003 | 1.0 | 2.6487 | 7.5129 | 0.107327 |
| sf | multimodal | 4069 | 1.0 | 18.7270 | 36.3234 | 0.218785 |
| validated | multimodal | 586 | 1.0 | 34.4710 | 55.2901 | 0.296207 |

`composition_gap_ndcg` = **0.474394** trên `gap_cohort_count` **6003** — đúng như
`summary/encoder_fashioniq_ce89a9a27c4cceadeaaf30b0/full.json`: `composition_gap_ndcg
0.47439372489324116`, `gap_cohort_count 6003`, `query_count 6003`; `raw/m01/rerun_pinned_final.txt:7`.
Con số `0.539144 / 6001` **không** thuộc ENCODER: nó là của LIMN `9e4918a6…`
(`raw/m03/derived_subset_runs.json:56,67`), dán nhầm sang mục này (census R12.1). So với run trước pin
`6a528e87…` (gap `0.472302`, cohort 6003), pin đổi gap `0.472302 → 0.474394`; cohort giữ 6003. Câu cũ
"không đổi so với run canonical cũ: pin chỉ làm feature xác định, không đổi mẫu số" **đã bỏ** — sai
theo raw.

**Run này là measurement hợp lệ với identity KHÔNG partition.** `ce89a9a2…` được mint **trước**
`f856338` (lúc runner còn hash `relation_state` trước handshake), nên identity/summary của nó ghi
`relation_state: not_recorded` — kiểm trực tiếp: `ranks/encoder_fashioniq.json` → `identity` **không**
có key `relation_state` (`runner.py:1219-1239` `_relation_identity` trả `not_recorded` cho payload
thiếu block). Vì vậy id **không** partition tensor relation: hai run draw relation khác nhau vẫn có
thể trùng id. Pin thì vẫn chắc **cho chính run này** — seed `20261111` áp trong worker
(`RELATION_SEED`) và tensor + sha256 `20db0c27…f9ce92` được persist dưới
`runs/ce89a9a2…/construction_state/`, nên số đo quy được về một tensor đã biết. Kết luận: **valid
measurement, unpartitioned identity — không phải run canonical.** Canonical là `acc9b20e…` (dưới);
năm id encoder còn lại đều superseded-by-design (`ce89a9a27c4cceadeaaf30b0`, `5dee7425dd46c5f154808aac`,
`6a528e877c328772e2c7bb28`, `eda30a37c0bc17ac43844f09` — lý do chung: id hash với
`relation_state: not_recorded`; và `9b806c22221e944ff10e514e` — identity recorded nhưng thiếu tensor).

Trạng thái construction đã persist, **bên ngoài** `work/` — run canonical:

```text
runs/acc9b20e1aea48487f65f944/construction_state/relation-seed20261111.npy(+.json)
  shape [2,512] dtype torch.float32 seed 20261111
  sha256 20db0c27c979e9a0938c71af5142458dfa2070468d0ad3fcb6f24134c2f9ce92  (== RELATION_EXPECTED_DIGEST)
  in_state_dict false
  file digest .npy (gồm header numpy) = 6ce23bbc6a367c34f9536a612c9e32e3b9cec213687c7bd73378a16001ccf244
```

Bản y hệt cũng có ở `runs/ce89a9a27c4cceadeaaf30b0/construction_state/` (byte-identical; run đó được
phục hồi bằng script re-derive sau khi bản gốc bị xoá). Pin là trên **byte tensor**, không phải trên
file `.npy` — hai digest khác nhau là đúng, đừng lẫn.

**Defect phát hiện khi chạy thật, đã sửa:** bản đầu ghi record dưới
`work/encoder/encoder/construction_state/`, và cleanup keep-features của runner
(`runner.py:2263-2271`) xoá mọi child của `work/<model>/` trừ `features/` ⇒ record bị xoá đúng lúc
run kết thúc, tức artifact cần cho hậu kiểm bị mất. Đã chuyển sang
`runs/<run_id>/construction_state/` qua `Adapter.construction_state_root()`
(`model_encoder.py:148-157,194,255`) — cùng cấp `ranks/`, `candidates/`, nên sống qua cleanup. Record
của run vừa chạy được phục hồi bằng digest (`scripts/m01/restore_relation_record.py`): tensor được
draw lại từ đúng seed đã pin, **refuse nếu digest khác**, và sidecar ghi rõ provenance là re-derive
sau khi bản gốc bị xoá (digest bằng giá trị run đã in trước khi xoá).

Bốn run encoder đầu `ce89a9a27c4cceadeaaf30b0`, `5dee7425dd46c5f154808aac`, `6a528e877c328772e2c7bb28`,
`eda30a37c0bc17ac43844f09` giữ nguyên byte, đánh dấu **superseded**: `5dee7425`/`6a528e87` với lý do
"relation drawn from an unpinned generator", cả bốn với lý do chung "id hashed with
`relation_state: not_recorded`" — mint trước `f856338`.

Run thứ năm **`9b806c22221e944ff10e514e`** mint sau `f856338`: `relation_state` **đã** `recorded`
(`sha256 20db0c27…f9ce92`, `matches_pinned_digest: true`), nhưng `construction_state/` **vắng** — defect
thật (guard persist bỏ qua vì handshake là job đầu), đóng ở `34aaa10`
(`model_encoder.py:262` `_adopt_scratch_relation`). Superseded vì thiếu tensor, **không** vì identity.

Run thứ sáu **`acc9b20e1aea48487f65f944`** (mint sau `34aaa10`) có **cả hai** ⇒ **canonical** (xem
khối `construction_state` ở trên). Superseded list của encoder nay là **năm** id trên.

### B. LIMN sf/validated — suy ra từ parent, không dùng GPU

`raw/m03/derived_subset_runs.json` + `scripts/m03/derive_subset_runs.py`. Chọn hàng theo qid trong
membership đã khai của release, tính lại metric bằng đúng `metrics.compose` của runner:

```text
parent            9e4918a6b5c243412cbc58c9  (ranks sha256, summary sha256 ghi trong artifact)
full              6003 khai / 6001 chấm / coverage 0.999667  mm R@10 30.0283  (khớp parent=True)
sf                4069 khai / 4068 chấm / coverage 0.999754  mm R@10 18.3874
validated          586 khai /  586 chấm / coverage 1.000000  mm R@10 30.0341

derived run id (mới, KHÔNG sửa parent; hàm của parent identity + membership sha256 + schema):
  sf        6a96c5c199526b61c1470b0e
  validated aee2e8d39214d877a84d7c08
  membership_sha256: full b1b1f21290b5995c… | sf 45e19ff14165e4de… | validated 52e0cf3138a4eb9e…
```

**Đã publish vào cây artifact thật** (`scripts/m03/publish_derived_runs.py`), đúng layout runner ghi,
nên các route hiện có đọc được mà không cần route mới:

```text
workbench/artifacts/composition/summary/limn_base_iter0_all_categories_6a96c5c1…/sf.json
workbench/artifacts/composition/summary/limn_base_iter0_all_categories_aee2e8d3…/validated.json
workbench/artifacts/composition/results/circus_fashioniq_release_v1/limn/base_iter0_all_categories/<derived>/<subset>/<condition>.jsonl
workbench/artifacts/composition/runs/<derived>/ranks/limn_base_iter0_all_categories.json
```

Mỗi manifest tự mang `derived_from` = `{schema circus-derived-subset/v1, parent_run_id,
parent_rank_artifact_sha256, parent_summary_sha256, membership_definition, membership_sha256,
membership_count, declared_query_count, coverage, scored_count, denominator, denominator_note,
gpu_used false, re_encoded false, rescored false}` — người đọc không phải mở file review nào.

Parent `9e4918a6…` chỉ được ĐỌC (`parent_bytes_touched: false`); derived run id khác parent; không
ảnh nào được decode, không forward pass, không score nào bị tính lại — chỉ select + re-aggregate.

Kiểm qua API sống (`raw/m03/publish_api_check.json`; **không** restart service):

```text
GET /api/composition/runs                                   -> 2 hàng derived, đúng số
  6a96c5c199526b61c1470b0e  subset sf         mm R@10 18.3874139626352  scored 4068  coverage 0.9997542
  aee2e8d39214d877a84d7c08  subset validated  mm R@10 30.034129692832764 scored  586  coverage 1.0
GET /api/composition/runs/<id>?subset=<subset>               -> 200, `derived_from.denominator` 4068 / 586
   status: sf DERIVED_PARTIAL (4068/4069), validated DERIVED_COMPLETE (586/586)
  (gọi không kèm ?subset thì 404 vì mặc định subset=full và derived run chỉ có subset của nó —
   đúng hành vi hiện có của route, không phải lỗi publish)
```

Không qid nào thiếu artifact per-query (`missing_per_query_artifacts: []`). Mỗi subset ghi
`parent_run_id`, sha256 của artifact parent, định nghĩa membership, `membership_sha256`, coverage và
mẫu số thực dùng.

### C. Transform — đo như control, KHÔNG đổi production

`raw/m01/transform_control.txt`, `scripts/m01/m01_transform_control.py`; cùng pixels, cùng weight, cùng
relation đã pin; chỉ transform của query khác. Có thêm phép kiểm rằng hai nhánh thật sự nhận pixel
khác nhau (`pixel_identical_val_vs_train`) — nếu không, delta 0 là control hỏng chứ không phải kết quả.

Kết quả (`raw/m01/transform_control.txt`, shirt, gallery 1 000 ảnh, 60 query, relation đã pin):

```text
pixel_max_abs_val_vs_train      3.605811   pixel_identical=false  (control THẬT SỰ khác input)
val_both                        R@10 70.0  ranks [4,1,6,5,10,3,10,2,14,3,84,…]
train_query/val_gallery         R@10 70.0  ranks [4,1,6,4,7,3,11,3,12,5,88,…]
recall_at_10_delta              0.0
ranks_changed                   26 / 60
max_abs_rank_delta              33
val lặp lại                     identical  (không nhiễu)
train lặp lại cùng seed         identical  (nhiễu augmentation đã pin)
```

Câu ghi vào artifact (`runs/ce89a9a2…/construction_state/transform_policy.json`), nguyên văn:

> **the transform asymmetry moves individual ranks materially while a saturated top-10 on a
> 1000-image window cannot see it**

Kèm `gallery_size: 1000`, `queries: 60`, `saturation_observed: true`, `ranks_changed: 26/60`,
`max_abs_rank_delta: 33`, `recall_at_10_delta: 0.0`, `pixel_identical_val_vs_train: false`
(pixel max|Δ| 3.6058111 — control thật sự đổi input), và `upgrade_path`:
"full 74,381-image gallery re-measurement; needs GPU, not approved". **Không** báo "R@10 delta 0" một
mình: chính dữ liệu rank bác bỏ cách đọc "không có ảnh hưởng".

`transform_policy` cũng nằm trong chính job payload/adapter
(`model_encoder.py:80-96,223,273,419`) với `query_transform: preprocess_val` và ghi chú native dùng
`preprocess_train` cho query, nên adaptation hiện diện ngay tại nơi số liệu sống.

CIRCUS policy giữ nguyên `preprocess_val` cho cả hai phía, ghi trong report là adaptation kèm
`query_transform: preprocess_val` và ghi chú native dùng `preprocess_train` cho query
(`datasets.py:177` vs `:195,202`). **Không** đổi production.

## M-03-f — status của derived run (phát hiện sau khi publish)

Trạng thái cũ: publisher copy `parent_summary["execution_status"]` ⇒ **cả hai** hàng derived trả
`PARTIAL`, kể cả `validated` có coverage 1.0 (586/586 ở cả ba condition). Đây là case **(a)**: status
không được tính cho derived run, và `PARTIAL` ở phần còn lại của hệ thống nghĩa là "có query chưa
chấm", nên nhãn đó vừa under-claim vừa làm mất phân biệt giữa "parent có 2 query chưa chấm" và "subset
này có query chưa chấm".

Sửa (`scripts/m03/publish_derived_runs.py`):

- `execution_status` **tính từ coverage per-condition của chính subset**: `DERIVED_COMPLETE` khi mọi
  condition `scored == query_count`, ngược lại `DERIVED_PARTIAL`;
- `condition_status` cũng tính lại, `unscored_count = membership − scored` cho từng condition;
- parent giữ nguyên trong `derived_from`: `parent_execution_status`, `parent_coverage`,
  `parent_unscored_qids`, `parent_unscored_qids_in_this_subset`.

Kết quả sau sửa (API sống, `raw/m03/publish_api_check.json`):

```text
sf        6a96c5c199526b61c1470b0e  DERIVED_PARTIAL   4068/4069  parent PARTIAL  unscored_here ['7:3790']
validated aee2e8d39214d877a84d7c08  DERIVED_COMPLETE   586/586   parent PARTIAL  unscored_here []
```

Hai qid chưa chấm của parent: **`7:3038`** và **`7:3790`**.
`7:3790` **nằm trong** `sf` (nên sf partial thật), `7:3038` **không** nằm trong `sf`;
**cả hai đều không** nằm trong `validated` (nên validated complete thật).

Regression: `workbench/tests/test_derived_subset_status.py` (4 test) — failing-first khi publisher
inherit status (`raw/m03/tests_derived_status_failing_first.txt`: `sf is labelled PARTIAL`),
pass sau sửa (`raw/m03/tests_derived_status_after.txt`).

## M-03-g — bản derived tự mâu thuẫn (scope blocks) + quyết định status enum

Publisher làm `summary = dict(parent_summary)` nên `validated.json` thừa hưởng nguyên block của
parent: `category_scope_failures: {mm: 2, …}` và
`adapter_query_scope_resolution.still_unscoped: 2` với `unscoped_qids: ['7:3038','7:3790']`, trong khi
`condition_status` của chính nó nói `unscored_count: 0` — hai câu phủ định nhau trên cùng một hàng, và
hai qid đó **đã chứng minh** nằm ngoài membership `validated`.

Sửa (`scripts/m03/publish_derived_runs.py`), chọn cách **tính lại** thay vì đổi tên:

```text
category_resolution            considered/resolved/ambiguous/unresolved tính lại trên membership subset
category_scope_failures        = |unscoped ∩ membership| cho từng condition
adapter_query_scope_resolution still_unscoped + unscoped_qids tính lại trên membership subset
unresolved_category_query_count  theo subset
parent_category_scope_failures / parent_adapter_query_scope_resolution / category_resolution.parent_…
                               giữ NGUYÊN giá trị của parent, dưới tên nói rõ là của parent
```

Thêm **assert** để hàng không thể tự phủ định: `still_unscoped` phải bằng `unscored_count` của cả ba
condition, sai thì publisher `REFUSING` chứ không ghi.

Kết quả sau sửa (`raw/m03/publish_api_check.json`, API sống):

```text
sf        DERIVED_PARTIAL   still_unscoped 1  unscoped_qids ['7:3790']  cond_unscored {mm:1, tob:1, ioe:1}
validated DERIVED_COMPLETE  still_unscoped 0  unscoped_qids []          cond_unscored {mm:0, tob:0, ioe:0}
parent (cả hai)             parent_category_scope_failures {mm:2, tob:2, ioe:2} — vẫn hiển thị, dưới tên parent_*
```

**Lỗi thứ hai phát hiện trong lúc sửa:** parent ghi khoá `unscoped_qids` ("un-**scoped**": không
chọn được checkpoint), tôi đọc `unscored_qids` ("un-**scored**": chưa chấm) ⇒ luôn rỗng, khiến `sf`
báo `still_unscoped: 0` trong khi `condition_status` đếm 1. Chính assert ở trên bắt được nó.

## Quyết định enum status: **option (b) đã áp dụng**

`execution_status` của hàng derived **kế thừa nguyên giá trị của parent**, nên luôn nằm trong
`COMPLETED|PARTIAL|REUSED|FAILED`. Độ đầy đủ của **chính subset** nằm trong `derived_from`:

```text
derived_from.own_coverage        coverage thật của subset (sf 0.9997542…, validated 1.0)
derived_from.own_scored_count    số query đã chấm (4068 / 586)
derived_from.own_unscored_count  số query chưa chấm (1 / 0)
derived_from.own_complete        own_unscored_count == 0
derived_from.own_status_label    'DERIVED_COMPLETE' / 'DERIVED_PARTIAL' — chỉ là NHÃN trong block này,
                                 không phải giá trị của enum
derived_from.parent_execution_status / parent_coverage / parent_unscoped_qids /
                parent_unscoped_qids_in_this_subset   lý do parent là PARTIAL, tách khỏi subset
```

Vì sao **không** thêm giá trị mới vào enum (phương án a bị loại): `execution_status` là một enum mà
bốn chỗ khác đang đọc theo nghĩa "vòng chạy này có hoàn tất không" —
`workbench/frontend/scope.ts:151`, `workbench/frontend/OverviewPage.tsx:251`,
`workbench/frontend/EvaluationPage.tsx:259`, và comment schema ở
`workbench/backend/composition.py:189`. Mở rộng contract đó chỉ để có thêm một nhãn là đổi thứ người
khác phụ thuộc; còn `derived_from` vốn đã là field nói "hàng này không phải measurement mới", và UI
bỏ qua key lạ.

Kết quả đọc được trên API sống (`raw/m03/publish_api_check.json`):

```text
sf        execution_status PARTIAL (kế thừa)  own_unscored 1  own_coverage 0.9997542
          parent PARTIAL, parent_coverage 0.9996668, unscoped trong subset ['7:3790']
validated execution_status PARTIAL (kế thừa)  own_unscored 0  own_coverage 1.0
          parent PARTIAL, parent_coverage 0.9996668, unscoped trong subset []
```

Nhờ đó phân biệt được "parent có 2 query chưa chấm" với "subset này có 0 query chưa chấm" mà không
cần giá trị enum mới.

### Regression test đang đỏ — chờ hết sweep (không sửa `workbench/tests` lúc này)

`workbench/tests/test_derived_subset_status.py` được viết theo contract tạm thời
(`execution_status = DERIVED_COMPLETE/DERIVED_PARTIAL`), nên sau khi chuyển sang option (b) nó fail
đúng như dự kiến: `assert 'PARTIAL' == 'DERIVED_PARTIAL'` — kỳ vọng của test mới là cái sai. Ba test
còn lại vẫn xanh. Bằng chứng: `raw/m03/tests_derived_status_option_b_pending_update.txt`, kèm
`raw/m03/tests_derived_status_option_b_note.txt` ghi chính xác dòng cần sửa (test phải assert
`derived_from.own_status_label`, `own_unscored_count`, `own_coverage` thay vì `execution_status`).
Sửa được ngay khi sweep xong; **không** sửa bây giờ vì `workbench/tests` đang bị đo.

### Known UI gap — **chưa có** marker "DERIVED"

`derived_from` hiện **không** được đọc ở đâu trong `workbench/frontend/` hay `workbench/backend/`
(`grep -rn derived_from` chỉ khớp một file trong `node_modules`). Nghĩa là: một hàng derived đang hiện
badge `PARTIAL` y như parent, không có nhãn "DERIVED", và người đọc không có cách nào từ UI biết nó là
subset suy ra thay vì một lần đo mới. Đây là **gap đã biết, chưa sửa** (frontend thuộc worker khác và
sweep đang chạy). Hai việc còn thiếu, ghi rõ để không ai đọc báo cáo này thành "UI đã xong":

1. chip "DERIVED" cạnh badge khi `row.derived_from` tồn tại;
2. hiển thị `own_unscored_count` / `own_coverage` thay cho (hoặc cạnh) `execution_status`, để badge
   `PARTIAL` của `validated` không bị đọc thành "subset này thiếu query".

## Việc bàn giao cho chủ file khác

1. `runner.py`: thêm `relation_state` (seed + sha256 + `matches_pinned_digest`) vào `identity`, để
   warm run không tái dùng feature từ draw khác; và thêm transform policy vào identity nếu M-01-b
   được duyệt.
2. `parity_gate.py`: cổng nên đọc `relation_state.matches_pinned_digest` và fail-closed khi thiếu.
3. Frontend: với M-03-d, không render số `sf`/`validated` của LIMN như subset của canonical run
   `9e4918a6…` khi hai file đó không tồn tại; hiển thị "canonical run chỉ có full".
4. Docs cũ (`ENCODER_PARITY.vi.md`, `LIMN_SCOPE_AUDIT.vi.md`) nên trỏ sang báo cáo này cho số sau sửa;
   các số 4.923/0.0/417 vẫn đúng như đo tại thời điểm của chúng.

## Kiểm chứng đã chạy (không chạy full suite)

```text
workbench/tests/test_limn_scope_routing.py test_limn_scope_runner.py
workbench/tests/test_limn_dual_scope_diagnostic.py test_limn_dual_scope_producer.py
workbench/tests/test_limn_diagnostic_record.py                                  -> 74 passed
workbench/tests/test_encoder_relation_seed.py test_encoder_replay.py
workbench/tests/test_audit_fix_encoder_helpers_imported.py
workbench/tests/test_audit_fix_f4_encoder_mapping_status.py
workbench/tests/test_encoder_pair_clvc_verification.py
workbench/tests/test_derived_subset_status.py (mới, M-03-f)                     -> 142 passed
```
