# Kiểm toán đường ống đầu vào — upstream → adapter → CIRCUS

Ngày: 2026-11-09. Revision khi viết: `0857744` (branch `research/cir-failure-workbench`).

**Mức bằng chứng dùng trong tài liệu này là mức thấp nhất đủ để nói một câu, không phải mức cao nhất
có thể biện hộ.** Bốn mức, dùng đúng như sau:

| mức | nghĩa | ở đâu |
|---|---|---|
| `static review` | đọc code, đối chiếu biểu thức/đường dẫn; **không** chạy | gần như toàn bộ tài liệu |
| `synthetic differential` | probe BLAS trên ma trận tổng hợp, so 1 vs 20 thread | §2.1 (bảng nhạy BLAS) |
| `real differential` | so rank từng query giữa hai lượt chạy thật, có file trong `rank_diffs/` | 8 model đã rerun |
| `NOT VERIFIED` | không có bằng chứng nào | các mục còn lại |

**G1 và G2 vẫn là `NOT_RUN`.** Tài liệu này kiểm toán *đường đi* của dữ liệu (file nguồn nào, hàm nào,
biểu thức nào), không chứng minh đầu ra của bất kỳ lượt forward nào khớp native. Một dòng ghi
`static review` **không** phải là chứng nhận parity. `real differential` chỉ nói rằng hai lượt chạy
*trên máy này* khớp nhau, không nói rằng chúng khớp upstream.

Điểm chung của mọi model: `runner._stage_query_path` (`composition/runner.py:619`) gọi
`inputs.adapt_query_inputs` (`composition/inputs.py:19`) **trước khi** adapter được gọi. Do đó
adapter không bao giờ tự suy ra điều kiện; nó nhận ảnh/text đã được áp dụng. Bảng ba điều kiện:

| điều kiện | ảnh adapter nhận | text adapter nhận |
|---|---|---|
| `multimodal` | ảnh gốc | caption gốc |
| `text_only_black_image` | PNG đã stage, pixel toàn 0 | caption gốc |
| `image_only_empty_text` | ảnh gốc | chuỗi rỗng |

Gallery bất biến theo điều kiện (`test_circus_inputs.py::test_gallery_objects_are_invariant_to_query_condition`).

---

## 1. Sáu model đã rerun với export Top-100

### 1.1 ENCODER — `/fashioniq`

- **Upstream:** `workbench/third_party/ENCODER/`, toán tử ghép nối nằm trong adapter (`model_encoder.py`).
- **Adapter `score()`** (`model_encoder.py:225-238`): chuẩn hoá L2 cả hai phía, `queries @ gallery.T`,
  chia theo lô 8192 cột gallery. **Lệch có chủ ý:** mask native ở `test.py:67-68` **không** kế thừa —
  ghi rõ trong docstring hàm. Reference giữ eligible.
- **Tiền xử lý:** transform native sau can thiệp PIL (black-image stage trước transform, đúng thứ tự).
- **Checkpoint:** `checkpoint_path()` từ registry; single file, `VERIFIED_METADATA`.
- **Bằng chứng:** `real differential` — `rank_diffs/encoder_6a528e87_vs_5dee7425.json`, 12727/18009
  rank đổi (max |Δ| 1706).
- **Đánh giá:** `static review` cho biểu thức, `real differential` cho tính ổn định giữa hai lượt.

### 1.2 CONESEP — `/fiq_n02`

- **Upstream:** `third_party/ConeSep/lavis/models/blip2_models/ConeSep.py`.
- **Adapter `score()`** (`model_conesep.py:72-79`): `einsum("nd,mdk->nmk", optimize=True).max(-1)` trên
  layout 32×256, `[:, 0, :]` cho query.
- **Checkpoint:** single file `ConeSep-FIQ-N0.2.pt`, `VERIFIED_METADATA`, `NO_CLEAN_CHECKPOINT`
  (chỉ có bản huấn luyện nhiễu 20%).
- **Bằng chứng:** `real differential` — `rank_diffs/conesep_30f6de42_vs_a4f786d8.json`, 17/18009 rank
  đổi, max |Δ| 1. Cơ chế: near-tie 1.2e-7 tại multimodal `7:3744`.
- **Đánh giá:** `static review` + `real differential`.

### 1.3 HABIT / INTENT / HINT — `/fiq_n02` và `/fashioniq`

- **Adapter `score()`**: cả ba dùng `np.einsum("nd,gqd->ngq", ...).max(-1)` — **không có đường BLAS**
  (vòng lặp numpy thuần). `model_habit.py:258`, `model_intent.py:248`, `model_hint.py:250`.
- **Checkpoint:** single file mỗi model; HABIT/INTENT `NO_CLEAN_CHECKPOINT` (nhiễu 20%), HINT
  `VERIFIED_METADATA`.
- **Bằng chứng:** `real differential` — cả ba đều 0/18009 rank đổi
  (`rank_diffs/habit_*, intent_*, hint_*`), metric bit-identical.
- **Đánh giá:** `static review` + `real differential`. Việc thiếu pin BLAS ở đây là **lỗ provenance
  lịch sử**, không phải lỗi số.

### 1.4 TME — `/fashioniq_0`

- **Adapter `score()`**: `model_tme.py:252`, cùng dạng `einsum("nd,gqd->ngq")` như HABIT.
- **Checkpoint:** single file `fashioniq_0_best_model.pth`; registry ghi FashionIQ 0% là mặc định
  workflow nên checkpoint CIRR không thể lặng lẽ thay thế.
- **Bằng chứng:** `real differential` — `rank_diffs/tme_ad477312_vs_cd1a696d.json`, 0/18009 đổi.
- **Đánh giá:** `static review` + `real differential`.

### 1.5 LIMN — `/base_iter0_all_categories`

- **Adapter `score()`** (`model_limn.py:370-408`): tính theo từng query, mỗi query dùng đúng không
  gian category của nó; chuẩn hoá query rồi `query @ gallery[start:stop].T` theo lô 8192.
  **Lệch có chủ ý:** yêu cầu scope-to-full-gallery matrices; ném `ExactRankUnavailable` nếu không có.
- **Tiền xử lý:** transform riêng theo category (`dress`/`shirt`/`toptee`), cộng
  `correct_files/` staged từ nguồn (`prepare_model.py` BUNDLED_ASSET_GENERATOR).
- **Checkpoint:** **bundle ba member** (`0_dress/shirt/toptee_best_model.pt`), mỗi member sha riêng.
- **Bằng chứng:** **không có** `real differential` — hai lượt chấm cohort khác nhau (5884 vs 6001),
  nên `rank_diffs/limn_cohort_note.json` chỉ ghi cohort và lý do không so được.
- **Coverage:** 6001/6003. Hai query `7:3038`, `7:3790` **không chấm được**, có lý do trong artifact:
  *"category unresolved has no adapter embedding scope"*. Đây là unscored-có-lý-do, không phải miss.
- **Đánh giá:** `static review`; **`real differential`: không có** (cohort khác).

### 1.6 PTHA+MTST — `/cirr_transfer` — **TRANSFER, không phải dòng FashionIQ của paper**

- **Upstream:** `third_party/MTST_PTHA/` (vendored, pin `6a02ee4`).
- **Adapter `score()`** (`model_ptha_mtst.py:80-87`): `einsum("nd,mkd->nmk", optimize=True).max(-1)`,
  query lấy `[:, 0, :]` của layout 32×256.
- **Checkpoint:** single file `tuned_mtst_cirr.pt` — **huấn luyện trên CIRR**, đánh giá trên FashionIQ.
  `checkpoint_transfer` đọc là `CIRR→FashionIQ transfer` trong cả hai artifact.
- **Runtime:** env `cir-005ef4e1b02999a5` **chỉ có MKL**, không có OpenBLAS. `mkl_set_num_threads`
  segfault interpreter đó (exit 139) qua mọi đường đã thử; chỉ `MKL_NUM_THREADS` env-var hoạt động
  nhưng không đặt được giữa tiến trình. Đường **encode** vì vậy **không pin được**; đường **chấm**
  chạy trong `ctl312h` (OpenBLAS thật) nên pin được và verify được.
- **Bằng chứng:** `real differential` — `ptha_mtst_315dd590_vs_f54a7745.json`, 46/18009 đổi, max |Δ| 2.
- **Đánh giá:** `static review` + `real differential`, **nhưng** không tương đương một lượt pin
  OpenBLAS. Không được trình bày như một dòng của paper FashionIQ.

---

## 2. Bốn adapter mới — chưa chạy trên GPU lần nào

Cả bốn đều tự báo `ready=False` với lý do thật. Kiểm tra trực tiếp ở revision này:

| model | `status().ready` | lý do nó tự báo |
|---|---|---|
| `pair` | **False** | `checkpoint mapping is UNVERIFIED for pair_b1: the released pickle names classes the pinned repository does not contain` |
| `combiner_rn50x4_noft` | **False** | `no interpreter environment resolved for WORKBENCH_CLIP4CIR_PYTHON` |
| `clip4cir_rn50x4_fullft` | **False** | `no interpreter environment resolved for WORKBENCH_CLIP4CIR_PYTHON` |

### 2.1 PAIR — `/pair_b1`, `/pair_b2`

- **Upstream scoring:** `third_party/PAIR/test.py:62` `sims = all_queries.dot(all_imgs.T)`;
  `:71` `np.argsort(-sims[i,:])[:50]`.
- **Adapter `score()`** (`model_pair.py:112-...`): chuẩn hoá cosine cả hai phía rồi một dot product,
  **chia lô theo gallery** để không materialise bản float32 thứ hai của ma trận 74k dòng.
- **Lệch có chủ ý, và đây là lệch lớn nhất trong tài liệu này:** upstream mask reference ở
  `test.py:62-64` (`sims[i, ...] = -10e10`). Adapter **không kế thừa mask đó** — CIRCUS yêu cầu
  gallery đầy đủ có reference eligible. `reference_eligible` khai `True` và được **đo** lúc chạy
  (probe score+rank trên gallery tổng hợp). Ghi rõ ở `model_pair.py:20-23,116`.
- **Checkpoint route:** `checkpoint_path()` cho đường dẫn, `_declared_member_paths()` cho bundle;
  adapter không viết tên file nào.
- **Blocker tự báo:** mapping `UNVERIFIED` với bằng chứng **dương tính**: 455 khoá checkpoint so với
  381 khoá model, 74 khoá thừa, 70 lệch shape, **5.20% tham số không reachable** từ forward pass đã
  pin. Vì vậy không lượt nào từ các byte này được gọi là tái lập PAIR.
- **Tiền xử lý:** registry khai `fashioniq_ilearn_resized_training`, `.jpg`.
- **Bằng chứng:** `static review` cho biểu thức; `test_composition_adapters_missing.py:108`
  kiểm **biểu thức tự khai** trên ma trận tổng hợp — **không** phải parity với upstream.
  `real differential`: **không có** (chưa chạy được).

### 2.2 combiner_rn50x4_noft — `/rn50x4_noft_unresolved`

- **Upstream:** `third_party/CLIP4Cir/src/combiner.py:33-48` — `Combiner.forward` trả
  `logit_scale * predicted_features @ target_features.T`, target chuẩn hoá ở `:45`;
  `:50 combine_features` dựng vector ghép.
- **Adapter `score()`** (`model_clip4cir.py:123-...`): chuẩn hoá cả hai phía rồi một dot product.
  Docstring ghi nguồn `validate.py:41,44,107`.
- **Lệch có chủ ý, quan trọng:** upstream tính **khoảng cách** `1 - predicted @ index.T`
  (`validate.py:44`) rồi `argsort` **tăng dần**. Runner xếp hạng **giảm dần** theo score. Adapter
  trả **độ tương đồng**, nên không được chép nguyên dòng `1 - …` vào đây — chép vào sẽ đảo ngược
  toàn bộ truy hồi trong khi trông vẫn hợp lý.
- **Checkpoint route:** **bundle** hai member, resolve theo `role` (`combiner`/`clip`) qua
  `_declared_member_paths()`. Tên `combiner_state.pt`/`clip_state.pt` **từng** là lỗi của adapter và
  không file nào tồn tại dưới tên đó.
- **Blocker tự báo:** chưa có interpreter `WORKBENCH_CLIP4CIR_PYTHON` trên host này.
- **Bằng chứng:** `static review`; `real differential`: **không có**.

### 2.3 clip4cir_rn50x4_fullft — `/rn50x4_fullft_unresolved`

- Cùng biểu thức và cùng adapter (`model_clip4cir.py`) như 2.2.
- **Khác biệt:** bundle có **hai** member — `fiq_comb_RN50x4_fullft.pt` (role `combiner`) và
  `fiq_clip_RN50x4_fullft.pt` (role `clip`). Cả hai đã stage với sha riêng, mapping
  `VERIFIED_METADATA`. Nửa CLIP là **fine-tuned** (413/669 tensor khác bản stock), nên **không** thay
  thế được bằng encoder đóng băng của bản noft.
- **Blocker tự báo:** cùng lý do interpreter.
- **Hạn chế cần nói rõ:** validator upstream **chỉ phát metric tổng hợp** R@10/R@50
  (`validate.py:44-56` tính distance, sort, trả hai scalar). Rank từng query và export Top-K là
  **việc mới** trong adapter này, không phải chép lại. Số học khớp biểu thức upstream, nhưng upstream
  chưa từng sinh ra dòng nào để đối chiếu.
- **Bằng chứng:** `static review`; `real differential`: **không có**.

---

## 3. Ba model có adapter nhưng chưa rerun

| model | adapter | `score()` | trạng thái |
|---|---|---|---|
| `airknow` | `model_airknow.py:99` | `einsum("nd,mgd->nmg").max(-1)` trên 32 token | `ready=False` — không có interpreter env; checkpoint `NO_CLEAN_CHECKPOINT` |
| `csmcir` | `model_csmcir.py:108-116` | `einsum("nd,mkd->nmk", optimize=True).max(-1)` | DEFERRED theo yêu cầu người dùng |
| `sprc` | **không có** | — | DEFERRED theo yêu cầu người dùng |

Cả ba: `static review` only. Không có `real differential`.

---

## 4. Những gì KHÔNG trace được

Ghi `unknown` thay vì suy đoán:

- **`score_dimension` thật của PAIR** — phụ thuộc backbone load được; registry không khai, và adapter
  chưa chạy nên giá trị chưa được xác nhận trên byte thật.
- **Việc `correction_dict/` của TGCIR có tham gia chấm hay không** — `test.py:61,129` là dot product
  thuần, nhưng `model.py` chưa được đọc để loại trừ.
- **`l2renorm` thực tế của DCNet** — `metric.py:24 sharded_cross_view_inner_product(...)` nhận cờ này;
  config trong `checkpoints/dcnet/config.json` chưa được đối chiếu. Và `dcnet-prepare` **không có
  torch**, nên không thể kiểm bằng chạy.
- **CLVCNet dùng interpreter nào** — registry khai py 3.7.6/torch 1.6.0, nhưng không env nào trong
  `workbench/envs/` khớp; **unknown**.
- **TGCIR dùng interpreter nào** — registry khai python `UNKNOWN`, nên không thể chọn env theo phiên bản.
- **Parity native↔adapter cho bất kỳ model nào** — G1/G2 `NOT_RUN`. Không có differential đầu vào
  hay forward nào được chạy trong lượt này.

---

## 5. Kết luận trung thực

- **8 model** có artifact thật dưới policy BLAS được ghi lại, export Top-100, invariant PASS.
- **7 trong 8** có `real differential` giữa hai lượt chạy (limn không, vì cohort khác).
- **0 model** có differential native↔adapter. G1 và G2 giữ `NOT_RUN`.
- **4 adapter mới** viết xong, test contract pass, **chưa chạy GPU lần nào**, tất cả `ready=False`.
- Các con số 12727 rank đổi của encoder **KHÔNG phải do BLAS** (đính chính 2026-11-09: A/B có kiểm soát
  `d3b_encoder_aggregate_scan.txt` chỉ đổi 1 rank/điều kiện, ΔR@50=0; 12727 là do **đổi rank method của adapter**
  (hash module/rank đổi, hash score giữ nguyên) cộng **đổi interpreter** py3.11.15→py3.10.20; R@10 tăng trong khi
  R@50 giảm — thread-count không làm được vậy). Chúng là **bằng chứng về tính bất định của phép tính**,
  không phải bằng chứng rằng số mới đúng hơn số cũ về mặt khoa học — chỉ rằng số cũ được đo dưới một
  policy không được ghi lại.
