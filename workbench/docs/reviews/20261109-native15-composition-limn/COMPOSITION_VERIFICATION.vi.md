# COMPOSITION VERIFICATION — kiểm tra độc lập 8 model CIRCUS

Ngày: 2026-11-09. Người thực hiện: workstream verify độc lập.
Phạm vi: chỉ đọc artifact trên đĩa. **Không chạy GPU, không chạy job mới.**
Interpreter dùng để đọc: `workbench/envs/ctl312h/bin/python`.

Mọi con số dưới đây đều đọc bằng `json.load` trong Python (không dùng `?q=` selector),
vì selector trên các file JSON này có thể trả null giả.

---

## 0. Kết luận nhanh (TL;DR)

| Hạng mục | Kết quả |
|---|---|
| 8/8 `run_id` khớp danh sách giao | **KHỚP TOÀN BỘ** |
| 16 ô `metrics_r10` / `metrics_r50` trong `COMPOSITION_MATRIX.csv` | **KHỚP TOÀN BỘ** (16/16) |
| Các ô số khác (query_count, export_depth, coverage_ratio, subset) | **KHỚP TOÀN BỘ** |
| Top-K depth 100 đầy đủ | **ĐẠT** 7/8; `limn` đúng là `partial: true`, thiếu 2 query |
| `file_sha256` của JSONL | **KHỚP** 8/8 (tính lại từ byte trên đĩa) |
| Mất query âm thầm | **KHÔNG CÓ**. `limn` có 2 query lỗi, đều có REASON |
| `identity_schema_version` / `policy_observed` trong `full.json` | **VẮNG MẶT ở cả 8 file** (xem §1.2) |
| **Lỗi phát hiện** | Cột `blas_pin` của `encoder` quy `12727/18009` cho BLAS — nguyên nhân thật là **đổi hàm `rank`** (chính) + **đổi interpreter** (phụ); xem §3.3 và danh sách file sai ở §3.3b |

---

## 1. Xác minh từng summary, từng trường

### 1.1 Bảng trường bắt buộc (giá trị nguyên văn)

Đường dẫn nguồn: `workbench/artifacts/composition/summary/<dir>/full.json`.

| model | run_id trên đĩa | khớp | execution_status | failed_expectations | hash_verification | query_count | gallery_count |
|---|---|---|---|---|---|---|---|
| encoder | `5dee7425dd46c5f154808aac` | ✅ | `COMPLETED` | `[]` | `matched` | 6003 | 74381 |
| conesep | `a4f786d8af4d46ac2a85ab9c` | ✅ | `COMPLETED` | `[]` | `matched` | 6003 | 74381 |
| habit | `c6eb9d5775ae636a3b8addbc` | ✅ | `COMPLETED` | `[]` | `matched` | 6003 | 74381 |
| intent | `4901da139ac40b874d2363c5` | ✅ | `COMPLETED` | `[]` | `matched` | 6003 | 74381 |
| hint | `f461f391c6d4696011be8067` | ✅ | `COMPLETED` | `[]` | `matched` | 6003 | 74381 |
| limn | `9e4918a6b5c243412cbc58c9` | ✅ | **`PARTIAL`** | `[]` | `matched` | 6003 | 74381 |
| ptha_mtst | `f54a77450ebd2126590d4385` | ✅ | `COMPLETED` | `[]` | `matched` | 6003 | 74381 |
| tme | `cd1a696d8bca0fc4bce6925b` | ✅ | `COMPLETED` | `[]` | `matched` | 6003 | 74381 |

**Không có `run_id` nào lệch.** Cả 8 đều trùng đúng chuỗi được giao.

`query_count_per_subset` giống nhau ở cả 8: `{"full": 6003, "sf": 4069, "validated": 586}`.

### 1.2 `topk_export`, `condition_status`, `cache_verify_mode`, `identity_schema_version`, `policy_observed`

| model | topk_export (depth / missing / partial / status) | condition_status (3 điều kiện) | cache_verify_mode | identity_schema_version | policy_observed |
|---|---|---|---|---|---|
| encoder | 100 / 0 / `false` / `TOPK_EXPORT_COMPLETE` | cả 3 `COMPLETED`, unscored 0 | `full` | **VẮNG** | **VẮNG** |
| conesep | 100 / 0 / `false` / `TOPK_EXPORT_COMPLETE` | cả 3 `COMPLETED`, unscored 0 | `full` | **VẮNG** | **VẮNG** |
| habit | 100 / 0 / `false` / `TOPK_EXPORT_COMPLETE` | cả 3 `COMPLETED`, unscored 0 | `full` | **VẮNG** | **VẮNG** |
| intent | 100 / 0 / `false` / `TOPK_EXPORT_COMPLETE` | cả 3 `COMPLETED`, unscored 0 | `full` | **VẮNG** | **VẮNG** |
| hint | 100 / 0 / `false` / `TOPK_EXPORT_COMPLETE` | cả 3 `COMPLETED`, unscored 0 | `full` | **VẮNG** | **VẮNG** |
| limn | 100 / **6** / **`true`** / `TOPK_EXPORT_COMPLETE_PARTIAL` | cả 3 **`PARTIAL`**, unscored **2** | `full` | **VẮNG** | **VẮNG** |
| ptha_mtst | 100 / 0 / `false` / `TOPK_EXPORT_COMPLETE` | cả 3 `COMPLETED`, unscored 0 | `full` | **VẮNG** | **VẮNG** |
| tme | 100 / 0 / `false` / `TOPK_EXPORT_COMPLETE` | cả 3 `COMPLETED`, unscored 0 | `full` | **VẮNG** | **VẮNG** |

`condition_status` có 3 khoá cố định: `multimodal`, `text_only_black_image`, `image_only_empty_text`;
mỗi khoá gồm `{status, unscored_count, fallback_used, category_scope_failure_count}`.
`fallback_used = false` ở cả 8 model, cả 3 điều kiện.

#### Vì sao `identity_schema_version` và `policy_observed` VẮNG — và đây KHÔNG phải lỗi che giấu

Quét đệ quy toàn bộ 8 `full.json` và 8 rank payload: **0 lần xuất hiện** của cả hai khoá
(`identity_schema_version`, `measurement_identity_schema_version`, `policy_observed`).

Nguyên nhân đã truy ra bằng mtime + git, không phải suy đoán:

| Bằng chứng | Giá trị |
|---|---|
| mtime rank payload `encoder` | `2026-11-09 03:35:38 +0700` |
| mtime `full.json` `encoder` | `2026-11-09 03:35:40 +0700` |
| commit thêm `policy_observed` (`d6bb1d3`) | `2026-11-09 09:17:13 +0700` |
| commit sửa reset-order (`0857744`) | `2026-11-09 09:21:06 +0700` |

⇒ Cả 8 artifact được ghi **trước** commit thêm trường `policy_observed` khoảng 5 giờ 42 phút.
Đây đúng là trường hợp "legacy artifact" mà chính source mô tả:

- `workbench/composition/runner.py:1155` — `policy_observed_from_payload()` trả
  `{"status": NOT_RECORDED, "reason": "this artifact predates the policy_observed field; ..."}`.
- `workbench/composition/runner.py:1056` — sentinel `NOT_RECORDED =
  "not_recorded (no readback captured for this run)"`.

Nghĩa là: **không có giá trị nào bị bịa ra, và không có giá trị nào bị backfill.**
Nhưng cũng có nghĩa: **8 artifact này không tự chứng minh được BLAS policy của chính chúng.**
Đây là open defect đã được ghi nhận ở `20261109-clean-rerun/BLAS_IDENTITY_FINDINGS.md`
("the artifact cannot prove its own BLAS policy").

`identity_schema_version` không nằm trong artifact; nó được **tính lúc serve** bởi
`workbench/backend/composition.py:720,729,804` (`measurement_identity()`).
Tôi đã gọi chính hàm đó trên 8 cặp (summary, rank identity):

```
encoder   schema=2 group=c43c108bd782ff2820c628df conf=full missing=[]
conesep   schema=2 group=042ae823b101f49630f066d6 conf=full missing=[]
habit     schema=2 group=53391a2fe528718e1fd6c1bb conf=full missing=[]
intent    schema=2 group=ac9e88ad1eff7fc2665e0d86 conf=full missing=[]
hint      schema=2 group=26077707346eddc848ab9ec2 conf=full missing=[]
limn      schema=2 group=1c4139f3fe80dcc1ad0b9afa conf=full missing=[]
ptha_mtst schema=2 group=c4c547c7dacede144321ac2f conf=full missing=[]
tme       schema=2 group=749cc7c8b57cb74c77bd19e7 conf=full missing=[]
```

⇒ Giá trị **suy ra** là `2` với `confidence=full` cho cả 8 (đủ 3 marker v2:
`blas_threads_pinned`, `query_scope_resolution`, `query_order_identity`).
Nhưng **giá trị nguyên văn trong `full.json` là VẮNG**. Báo cáo này ghi đúng cả hai sự kiện.

### 1.3 Ba điều kiện — metric đầy đủ

Đơn vị: Recall là **PHẦN TRĂM**, nDCG và MRR là **PHÂN SỐ** (theo `CIRCUS_COMPOSITION_SPEC.md:70`).

| model | điều kiện | R@10 | R@50 | nDCG | MRR | scored | coverage |
|---|---|---:|---:|---:|---:|---:|---:|
| encoder | multimodal | 31.0345 | 49.4753 | 0.296068 | 0.169984 | 6003 | 1.000000 |
| encoder | text_only_black_image | 8.7456 | 18.2742 | 0.156955 | 0.047610 | 6003 | 1.000000 |
| encoder | image_only_empty_text | 2.6653 | 7.4629 | 0.107320 | 0.011075 | 6003 | 1.000000 |
| conesep | multimodal | 35.4823 | 54.1396 | 0.329441 | 0.206441 | 6003 | 1.000000 |
| conesep | text_only_black_image | 15.0925 | 26.8699 | 0.199557 | 0.087284 | 6003 | 1.000000 |
| conesep | image_only_empty_text | 3.1984 | 9.6119 | 0.112064 | 0.012699 | 6003 | 1.000000 |
| habit | multimodal | 35.6155 | 54.3728 | 0.328283 | 0.204968 | 6003 | 1.000000 |
| habit | text_only_black_image | 15.2924 | 27.0531 | 0.200678 | 0.088180 | 6003 | 1.000000 |
| habit | image_only_empty_text | 3.3150 | 10.3115 | 0.113736 | 0.013568 | 6003 | 1.000000 |
| intent | multimodal | 34.3995 | 54.0063 | 0.318491 | 0.192747 | 6003 | 1.000000 |
| intent | text_only_black_image | 15.3257 | 26.7366 | 0.198935 | 0.086034 | 6003 | 1.000000 |
| intent | image_only_empty_text | 2.8652 | 9.1288 | 0.112233 | 0.012596 | 6003 | 1.000000 |
| hint | multimodal | 32.8169 | 51.7741 | 0.309269 | 0.184097 | 6003 | 1.000000 |
| hint | text_only_black_image | 11.2111 | 21.5892 | 0.173591 | 0.064388 | 6003 | 1.000000 |
| hint | image_only_empty_text | 3.3983 | 9.9284 | 0.114274 | 0.014000 | 6003 | 1.000000 |
| limn | multimodal | 30.0283 | 48.6752 | 0.290394 | 0.166757 | **6001** | **0.999667** |
| limn | text_only_black_image | 6.3823 | 14.3809 | 0.133830 | 0.029294 | **6001** | **0.999667** |
| limn | image_only_empty_text | 2.4996 | 6.9322 | 0.099570 | 0.010266 | **6001** | **0.999667** |
| ptha_mtst | multimodal | 19.4736 | 34.4494 | 0.224934 | 0.104523 | 6003 | 1.000000 |
| ptha_mtst | text_only_black_image | 10.8113 | 21.0728 | 0.170135 | 0.060938 | 6003 | 1.000000 |
| ptha_mtst | image_only_empty_text | 1.3327 | 4.6477 | 0.094981 | 0.006189 | 6003 | 1.000000 |
| tme | multimodal | 34.0163 | 53.9564 | 0.320388 | 0.195808 | 6003 | 1.000000 |
| tme | text_only_black_image | 16.0753 | 28.0360 | 0.205497 | 0.092925 | 6003 | 1.000000 |
| tme | image_only_empty_text | 2.6987 | 9.2287 | 0.111706 | 0.011584 | 6003 | 1.000000 |

Composition gap (`composition_gap_ndcg`) và `gap_cohort_count`:

| model | gap | mrr_gap | cohort |
|---|---:|---:|---:|
| encoder | 0.469868 | 0.122374 | 6003 |
| conesep | 0.394255 | 0.119156 | 6003 |
| habit | 0.388705 | 0.116788 | 6003 |
| intent | 0.375383 | 0.106712 | 6003 |
| hint | 0.438704 | 0.119709 | 6003 |
| limn | 0.539144 | 0.137463 | **6001** |
| ptha_mtst | 0.243622 | 0.043585 | 6003 |
| tme | 0.358601 | 0.102883 | 6003 |

### 1.4 Tôi đã tính lại metric từ rank payload, không chỉ đọc lại summary

Đọc `runs/<run_id>/ranks/<stem>.json` → `ranks[condition]`, tính lại trực tiếp
`R@10 = 100 * #{r ≤ 10} / n`, `R@50 = 100 * #{r ≤ 50} / n`,
`nDCG = mean(1 / log2(r+1))`, rồi so với `full.json`.

Kết quả: **24/24 dòng (8 model × 3 điều kiện) khớp đến 1e-9** ở cả ba metric.
Ví dụ encoder multimodal: tính lại 31.034483 / 49.475262 / 0.29606758 — summary 31.034483 / 49.475262 / 0.296068. ✅

Gap cũng tính lại bằng `1 - max(I, T) / MM` sau khi lấy trung bình trên cùng cohort:
**8/8 khớp tuyệt đối (diff = 0.00e+00)**, đúng thứ tự thao tác mà spec yêu cầu
(`CIRCUS_COMPOSITION_SPEC.md:70`: "Compute the ratio after averaging, not by averaging per-query ratios").

---

## 2. Top-K export — depth 100 và tính đầy đủ

### 2.1 Manifest (multimodal) — 8/8

Đường dẫn: `workbench/artifacts/composition/runs/<run_id>/candidates/*/multimodal.manifest.json`.

| model | depth | record_count | expected_record_count | missing_query_count | partial | gallery_count | file_sha256 (đầu) |
|---|---:|---:|---:|---:|---|---:|---|
| encoder | 100 | 6003 | 6003 | 0 | `false` | 74381 | `74257d25eb5d71b8` |
| conesep | 100 | 6003 | 6003 | 0 | `false` | 74381 | `346e9ef66ba43ce0` |
| habit | 100 | 6003 | 6003 | 0 | `false` | 74381 | `8f92318c887dfde1` |
| intent | 100 | 6003 | 6003 | 0 | `false` | 74381 | `107b0fc048bb7216` |
| hint | 100 | 6003 | 6003 | 0 | `false` | 74381 | `11260fb23f6a31ed` |
| **limn** | 100 | **6001** | **6003** | **2** | **`true`** | 74381 | `8d2a01b6255c8315` |
| ptha_mtst | 100 | 6003 | 6003 | 0 | `false` | 74381 | `c9ebea0bbb12a187` |
| tme | 100 | 6003 | 6003 | 0 | `false` | 74381 | `f09d2af0d1e608b7` |

`limn` đúng như kỳ vọng: `partial: true`, thiếu 2.
`missing_query_examples` của limn = `["7:3038", "7:3790"]` — đúng 2 query đã biết.

Cả 3 điều kiện (`multimodal`, `text_only_black_image`, `image_only_empty_text`) đều có manifest riêng,
và **tất cả** đều cùng trạng thái: 7 model × 3 = 21 manifest đầy đủ; limn 3 manifest đều 6001/6003 partial.

### 2.2 `file_sha256` — tính lại từ byte trên đĩa

Tôi băm lại `sha256` của chính file JSONL và so với `file_sha256` trong manifest,
đồng thời so `file_bytes` với `os.path.getsize`:

**8/8 KHỚP**, kể cả `file_bytes`. Ví dụ encoder: manifest `74257d25…f07d3d`, tính lại
`74257d25…f07d3d`, bytes 16440072 = 16440072. ✅

Đây là kiểm tra độc lập thật: nếu JSONL bị sửa sau khi manifest ghi, bước này sẽ fail.

### 2.3 Spot-check một dòng JSONL — cả 8 model (không chỉ 3)

Schema một dòng: `{"qid": str, "target_rank": int|null, "topk": [[image_id, score], ...]}`.
`topk` là list các cặp `[int, float]`, không phải list dict.

| model | qid dòng 1 | target_rank | len(topk) | giảm dần | id duy nhất | tổng dòng | dòng ≠ 100 |
|---|---|---:|---:|---|---:|---:|---:|
| encoder | 7:1 | 11 | 100 | ✅ | 100 | 6003 | 0 |
| conesep | 7:1 | 26 | 100 | ✅ | 100 | 6003 | 0 |
| habit | 7:1 | 15 | 100 | ✅ | 100 | 6003 | 0 |
| intent | 7:1 | 26 | 100 | ✅ | 100 | 6003 | 0 |
| hint | 7:1 | 5 | 100 | ✅ | 100 | 6003 | 0 |
| **limn** | 7:1 | 15 | 100 | ✅ | 100 | **6001** | 0 |
| ptha_mtst | 7:1 | 167 | 100 | ✅ | 100 | 6003 | 0 |
| tme | 7:1 | 15 | 100 | ✅ | 100 | 6003 | 0 |

"Giảm dần" = `score[i] >= score[i+1]` với mọi i. Không dòng nào vi phạm.
Không dòng nào có ít/ nhiều hơn 100 ứng viên (quét **toàn bộ** file, không lấy mẫu).
`image_id` nằm trong `[0, 74381)` ở cả 8, và 100 id đều phân biệt.

Ví dụ encoder dòng 1: đầu `[47046, 0.6823006868362427]`, cuối `[15147, 0.46935802698135376]`.
ptha_mtst dùng thang score khác (`[47205, 33.9124…]` … `[40905, 28.5592…]`) — vẫn giảm dần.

### 2.4 Cross-check `target_rank` giữa hai artifact

So `target_rank` trong JSONL ứng viên với `ranks["multimodal"][qid]` trong rank payload,
trên **toàn bộ** dòng: **0 mismatch ở cả 8 model**. Nghĩa là cửa sổ Top-K và hạng metric
được sinh từ cùng một lần chấm điểm, đúng như `adapters.py:392` cam kết
(`rank_and_topk`: "the score matrix that produced the rank also produced the window").

---

## 3. Đối chiếu `COMPOSITION_MATRIX.csv` với `full.json`

Nguồn: `workbench/docs/reviews/20261109-native15-composition-limn/COMPOSITION_MATRIX.csv`.
Tôi đọc từng ô, mở `evidence_path` của chính dòng đó, so giá trị.

### 3.1 `metrics_r10` / `metrics_r50` — 16/16 KHỚP

| model | matrix r10 | json `recall_at_10` | matrix r50 | json `recall_at_50` | khớp |
|---|---:|---:|---:|---:|---|
| encoder | 31.0345 | 31.0344827586 | 49.4753 | 49.4752623688 | ✅ |
| conesep | 35.4823 | 35.4822588706 | 54.1396 | 54.1395968682 | ✅ |
| habit | 35.6155 | 35.6155255705 | 54.3728 | 54.3728135932 | ✅ |
| intent | 34.3995 | 34.3994669332 | 54.0063 | 54.0063301682 | ✅ |
| hint | 32.8169 | 32.8169248709 | 51.7741 | 51.7741129435 | ✅ |
| limn | 30.0283 | 30.0283286119 | 48.6752 | 48.6752207965 | ✅ |
| ptha_mtst | 19.4736 | 19.4735965351 | 34.4494 | 34.4494419457 | ✅ |
| tme | 34.0163 | 34.0163251707 | 53.9564 | 53.9563551558 | ✅ |

Tất cả khớp đúng ở 4 chữ số thập phân (sai số làm tròn < 5e-5).
Không có ô nào lệch. Không có ô nào bị đảo giữa hai model.

### 3.2 Các ô số/chuỗi khác — KHỚP

| ô | kết quả |
|---|---|
| `query_count` (8 dòng) | 6003 = `full.json.query_count` ✅ |
| `export_depth` (8 dòng) | 100 = `topk_export.depth` ✅ |
| `coverage_ratio` (8 dòng) | khớp `metrics.multimodal.coverage` ✅ (limn 0.9996668332500417) |
| `subset` (8 dòng) | `full` = `full.json.subset` ✅ |
| `topk_available` (8 dòng) | khớp `topk_export.status` ✅ (limn `TOPK_EXPORT_COMPLETE_PARTIAL`) |
| `validation_status` (8 dòng) | khớp `full.json.reproduction_status` ✅ |
| `run_id` (8 dòng) | khớp ✅ |

### 3.3 ❌ LỖI PHÁT HIỆN — cột `blas_pin` của `encoder` quy sai nguồn

Ô của encoder ghi:

> `requested=1 effective=1 (verified; BLAS-sensitive: 12727/18009 ranks changed vs unpinned)`

Con số `12727/18009` là **thật**, nhưng **quy cho BLAS pinning là sai**.

### Nguyên nhân, theo đúng thứ tự nhân quả

**1. NGUYÊN NHÂN CHÍNH — hàm `rank` của adapter đã đổi.** Hai run **không tính cùng một hàm**:

| trường (trong rank payload `identity`) | old `6a528e87…` | new `5dee7425…` |
|---|---|---|
| `adapter_module_source_hash` | `1a8b0dd25827ca0ba90f504d81f2fa190cede1ee3fb3bfe335e70ab359005823` | `bf60d855be3d72dd2ff0abc6d8b9dc12dbb5fad4a50aa87aaa774454f65a76ab` |
| `adapter_source_hashes.rank` | `5f828b13fef94a3be4e63806cf383d4f8d67a92561f136005bd6cf50315eb978` | `f0914b12834acd0bf7542913384b9b08bd9a742af91816056f1bd73022cf3d4a` |
| `adapter_source_hashes.score` | `0ab4fe1dd028b3a803e79617d6bf24f0d8de410553219a87e98125833847d2cd` | *(giống hệt — không đổi)* |

File: `runs/6a528e877c328772e2c7bb28/ranks/encoder_fashioniq.json` và
`runs/5dee7425dd46c5f154808aac/ranks/encoder_fashioniq.json`.

`score` **không đổi**, nhưng `rank` **đổi** — nghĩa là cùng một ma trận điểm được xếp hạng
bởi một thủ tục khác. Một mình điều này đủ để giải thích đảo thứ tự quy mô lớn, **và** giải thích
hình dạng metric: **R@10 đi LÊN (31.0178 → 31.0345) trong khi R@50 đi XUỐNG (49.5752 → 49.4753)** —
không có thay đổi số thread nào tạo ra mẫu đó.

**2. NGUYÊN NHÂN PHỤ — interpreter đổi.** `runtime.model_interpreter` trong hai summary, nguyên văn:

```json
// OLD — summary/encoder_fashioniq_6a528e877c328772e2c7bb28/full.json
"model_interpreter": {
  "environment_fingerprint": "8f0a3a15c90d746a2104ecbfdf8ff8932a0a0c94f5d8e566c2d89a1b52b8d210",
  "path": "/home/heheboiz/data/cir/workbench/envs/pair-adapted/bin/python",
  "source": "process_environment",
  "verification_reason": "expected Python 3.9, found 3.11.15",
  "verification_status": "PYTHON_VERSION_MISMATCH",
  "verified": false,
  "version": "Python 3.11.15; torch 2.7.1+cu128"
}

// NEW — summary/encoder_fashioniq_5dee7425dd46c5f154808aac/full.json
"model_interpreter": {
  "environment_fingerprint": "9abbe9047d6d581959a324d9dad121c4d6520a22785204c2fdedcfe7da08b79a",
  "path": "/home/heheboiz/data/cir/workbench/envs/encoder310-adapted/bin/python",
  "source": "workbench_env",
  "verification_reason": "CUDA capability must be verified on the GPU host with verify_gpu_runtime() before this model can be launched",
  "verification_status": "GPU_VERIFICATION_PENDING",
  "verified": true,
  "version": "Python 3.10.20; torch 2.7.1+cu128"
}
```

Run cũ chạy `pair-adapted` Python **3.11.15** — bị khai `PYTHON_VERSION_MISMATCH`, `verified: false`;
run mới chạy `encoder310-adapted` Python **3.10.20**, `verified: true`.
`checkpoint_identity` (`0f31c501…`) và `gallery_identity` (`1db68c59…`) **giống nhau** ở cả hai,
nên không phải checkpoint hay gallery gây ra khác biệt.

**3. BLAS pin — KHÔNG phải nguyên nhân.** Thí nghiệm A/B có kiểm soát
(`workbench/docs/reviews/20261108T-throughput/d3b_encoder_aggregate_scan.json`, mã
`d3b_encoder_aggregate_scan.py`): encode vector **một lần**, rồi `rank()` hai lần ở BLAS=1 và BLAS=10
trên **cùng vector, cùng gallery**:

```
multimodal            changed=1  max_drank=1  ndcg_delta=2.18e-05  r10_delta=0.0  r50_delta=0.0
text_only_black_image changed=1  max_drank=1  ndcg_delta=2.18e-05  r10_delta=0.0  r50_delta=0.0
image_only_empty_text changed=1  max_drank=1  ndcg_delta=2.18e-05  r10_delta=0.0  r50_delta=0.0
```

⇒ BLAS-only làm đổi **1 query mỗi điều kiện** (tổng 3/18009), `max|Δrank| = 1`,
**R@10/R@50 đổi đúng 0.0**.

**Quy mô so sánh:** `12727/18009` (`rank_diffs/encoder_6a528e87_vs_5dee7425.json`,
`changed = 5214 + 2819 + 4694`, `max|Δrank| = 1706`) so với **3/18009** của BLAS-only.
Chênh ~4000 lần. BLAS pin không thể là nguyên nhân.

### Metric đã đổi (đọc từ hai summary)

| chỉ số | old | new | Δ |
|---|---:|---:|---:|
| multimodal R@10 | 31.0178 | 31.0345 | **+0.0167** |
| multimodal R@50 | 49.5752 | 49.4753 | **−0.0999** |
| image_only R@50 | 7.4796 | 7.4629 | −0.0167 |
| text_only R@10 | 8.5790 | 8.7456 | +0.1666 |
| text_only R@50 | 18.1576 | 18.2742 | +0.1166 |
| `composition_gap_ndcg` | 0.472302 | 0.469868 | −0.002434 |

### ⚠️ BẪY ĐỌC FILE — `null` GIẢ từ selector dạng object

Trường `adapter_module_source_hash` và `adapter_source_hashes` **là `null` trong `full.json`**
(summary không mang hai trường này — một lỗ hổng tài liệu của summary, **không** phải bằng chứng
là chúng không đổi). Chúng chỉ có giá trị trong **rank payload** `runs/<run_id>/ranks/<stem>.json`.

Nguy hiểm hơn: trên chính các file rank payload này, **selector dạng object trả `null` GIẢ**
dù scalar có thật. Đã tái lập được trên `runs/6a528e877c328772e2c7bb28/ranks/encoder_fashioniq.json`:

```
?q={identity:{adapter_module_source_hash,adapter_source_hashes,blas_threads_pinned}}
  -> {"identity":{"adapter_module_source_hash":null,"adapter_source_hashes":null,"blas_threads_pinned":null}}

?q=.identity.adapter_module_source_hash
  -> 1a8b0dd25827ca0ba90f504d81f2fa190cede1ee3fb3bfe335e70ab359005823

?q=.identity.adapter_source_hashes.rank
  -> 5f828b13fef94a3be4e63806cf383d4f8d67a92561f136005bd6cf50315eb978
```

**Cùng một file: selector object → `null` cả ba; selector scalar → giá trị thật.**
Bẫy này đã thực sự đánh lừa một lần trong quá trình làm báo cáo: một bản nháp trước đã rút lại
bằng chứng source-hash vì đọc ra `null`, rồi phải khôi phục lại sau khi kiểm bằng scalar.
**Quy tắc cho người đọc kế tiếp: trên các artifact này, đọc từng scalar một, không bao giờ
qua selector lồng object.** (Trong Python, `json.load` rồi truy cập từng khoá là an toàn.)

**Kết luận §3.3:** `12727/18009` là khác biệt giữa hai run **tính hai hàm `rank` khác nhau**
(`adapter_source_hashes.rank` `5f828b13…` → `f0914b12…`), cộng thêm **đổi interpreter**
(py3.11.15 unverified → py3.10.20 verified). **Không phải** do BLAS thread pin.

Đề nghị sửa ô này (tôi **không** sửa — file không thuộc quyền tôi) thành dạng:
`requested=1 effective=1 (BLAS pin KHÔNG phải nguyên nhân: A/B sạch đổi 1 rank/điều kiện, ΔR@50=0; 12727/18009 là do đổi hàm rank + đổi interpreter, xem §3.3)`.

> **Hệ quả — bắt buộc sửa:** bất kỳ tài liệu nào viết *"12,727 hạng đổi vì BLAS pinning"*
> đều **SAI** và phải được sửa lại. Con số 12,727 là thật nhưng nguyên nhân là **đổi hàm `rank`
> và đổi interpreter**, không phải BLAS. Danh sách file đang nói sai — xem §3.3b.

### 3.3b Danh sách file đang nói SAI "12,727 hạng đổi vì BLAS"

Tôi grep `12727` trong `workbench/docs/` (14 file khớp). Phân loại:

**A. Nói SAI — quy 12,727 cho BLAS pinning, cần sửa:**

| file:dòng | câu đang nói |
|---|---|
| `docs/reviews/20261109-native15-composition-limn/COMPOSITION_MATRIX.csv:2` | `blas_pin` = `requested=1 effective=1 (verified; BLAS-sensitive: 12727/18009 ranks changed vs unpinned)` — "vs unpinned" quy thẳng cho BLAS |
| `docs/reviews/20261109-clean-rerun/RERUN_DECISIONS.csv:16` | `"rerun under pinned BLAS; 12727/18009 ranks moved vs 6a528e87 ..."` — "under pinned BLAS" đặt con số trong ngữ cảnh pin |
| `docs/reviews/20261109-clean-rerun/MODEL_MATRIX.csv:16` | `"rerun under pinned BLAS; 12727/18009 ranks moved vs 6a528e87 (max |drank| 1706) ..."` |
| `docs/reviews/20261109-clean-rerun/RUN_LOG.csv:2` | `"BLAS-sensitive as measured: 12727 of 18009 query ranks changed ..."` — mở đầu bằng "BLAS-sensitive as measured" |
| `docs/reviews/20261109-clean-rerun/BLAS_IDENTITY_FINDINGS.md:115` | `"Genuinely BLAS-sensitive: 12727 of 18009 ranks moved ..."` — khẳng định trực tiếp |
| `docs/reviews/20261109-clean-rerun/GPU_LANE_REPORT.md:70` | `"encoder — the only model with material movement: 12727 of 18009 ranks changed ..."` (mục "What the pinned rerun changed") |
| `docs/reviews/20261109-clean-rerun/INPUT_PIPELINE_AUDIT.vi.md:209` | `"Các con số BLAS-sensitive (encoder 12727 rank đổi) ..."` — gọi thẳng là "BLAS-sensitive" |

**B. Trung tính / chỉ báo số, không quy nguyên nhân (không bắt buộc sửa):**

| file:dòng | ghi chú |
|---|---|
| `docs/reviews/20261109-clean-rerun/EVIDENCE_INDEX.md:51` | bảng đếm thuần: `18009 | 12727 | 805/852/1706` |
| `docs/reviews/20261109-clean-rerun/INPUT_PIPELINE_AUDIT.vi.md:44` | chỉ dẫn chứng `12727/18009` từ file rank_diff |
| `docs/reviews/20261109-clean-rerun/BLAS_IDENTITY_FINDINGS.md:47` | tổng kết `changed_rank_queries` |
| `docs/reviews/20261109-native15-composition-limn/raw/composition_verify/allranks.txt:2` | log của chính tôi |

**C. Không liên quan — `12727` là image id trong dữ liệu, không phải con số rank:**

| file:dòng | ghi chú |
|---|---|
| `docs/reviews/20261108T153045-audit-equivalence/query_manifest.json:15709` | `"7:5706": [12727]` — index ảnh |
| `limn_native.{dress,shirt,toptee}.jsonl` | chuỗi số trong score/JSONL |

Lưu ý: chính `BLAS_IDENTITY_FINDINGS.md` **đã** ghi đúng cơ chế ở chỗ khác (§"Mechanism: conesep",
và open defect về policy) — nhưng dòng 115 vẫn khẳng định "Genuinely BLAS-sensitive" cho encoder,
và `d3b_encoder_aggregate_scan.json` (A/B sạch) **không** hỗ trợ khẳng định đó.

**Ghi chú:** các ô `blas_pin` còn lại **không** bị lỗi này:
- conesep `17/18009`: cặp run có `adapter_source_hashes.rank` **giống nhau**, `score` giống,
  interpreter giống, checkpoint/gallery giống → nhiễu đã được kiểm soát. ✅
- ptha_mtst `46/18009`: cùng kiểm tra, `rank` + `score` + interpreter + ckpt + gallery đều giống. ✅
  (ô này đã ghi đúng caveat "ENCODE path UNPINNABLE").
- habit / intent / hint / tme `0 ranks changed`: khớp `rank_diffs` (0/18009, `bit_identical=true`). ✅
- limn `plain einsum`: **chưa** kiểm chứng được bằng A/B sạch. `rank_diffs/limn_cohort_note.json`
  nói rõ không so được (5884 vs 6001 query). Nhưng `model_limn.py:370-399` dùng `query @ gallery.T`
  (gemv/gemm), **không phải** naive einsum như habit/intent/hint/tme.
  Nhãn "plain einsum" trong matrix **không khớp source**. Xem §3.4.

### 3.4 ❌ LỖI THỨ HAI — nhãn `plain einsum` của `limn` sai so với source

`COMPOSITION_MATRIX.csv` dòng limn ghi: `requested=1 effective=1 (verified; plain einsum)`.

Nhưng `workbench/composition/model_limn.py:400` (hàm `score`, vòng lặp khối 8192 dòng):

```python
scores[row, start:stop] = query @ np.asarray(gallery[start:stop]).T
```

Đây là **phép nhân ma trận** (`@`) trên từng khối 8192 dòng — tức đường BLAS gemm/gemv,
**không** phải naive einsum. Naive einsum là dạng `np.einsum("nd,gqd->ngq", ...)` không
`optimize`, thấy ở habit/intent/hint/tme:

```
habit  model_habit.py:258   np.einsum("nd,gqd->ngq", query_values, targets).max(axis=-1)
intent model_intent.py:248  np.einsum("nd,gqd->ngq", query_values, target_values).max(axis=-1)
hint   model_hint.py:250    np.einsum("nd,gqd->ngq", query_values, target_values).max(axis=-1)
tme    model_tme.py:252     np.einsum("nd,gqd->ngq", query_values, target_values).max(axis=-1)
```

Còn conesep/ptha_mtst dùng `np.einsum(..., optimize=True)` → cũng hạ xuống gemm:

```
conesep   model_conesep.py:79   np.einsum("nd,mdk->nmk", query, target, optimize=True).max(axis=-1)
ptha_mtst model_ptha_mtst.py:87 np.einsum("nd,mkd->nmk", query, target_tokens, optimize=True).max(axis=-1)
```

Theo phân loại của chính `20261108T-throughput/BLAS-POLICY.vi.md:85`, limn nằm trong nhóm
**GEMM-path** (encoder, limn, csmcir, ptha_mtst, conesep), không phải nhóm naive-einsum.
Vậy nhãn "plain einsum" cho limn là **mâu thuẫn với source và với tài liệu BLAS nội bộ**.
Không ảnh hưởng metric, nhưng là một khẳng định sai về cơ chế.

---

## 4. Giới hạn so sánh — CIRCUS ≠ native, không bao giờ chung một bảng

### 4.1 CIRCUS là gì (đã xác minh bằng file, không chỉ đọc doc)

Protocol `circus_fashioniq_release_v1`. Đếm trực tiếp và băm lại:

| artifact | đường dẫn | số dòng | sha256 (tính lại) |
|---|---|---:|---|
| Query test | `workbench/artifacts/circus_release/mbeir_fashioniq_task7_test.jsonl` | **6003** | `b601ef6a5411bf9385cff41e64bea863871ab1094cd09d78683de6a5141165ef` |
| Candidate pool | `workbench/artifacts/circus_release/mbeir_fashioniq_task7_cand_pool.jsonl` | **74381** | `121ec284832c33289ee13bfac19d5d59dc3f8aeddf672d784fd5eb8bcdf79eef` |

Hai hash này trùng đúng `release_artifact_hashes.mbeir_query` / `.mbeir_candidates`
trong cả 8 `full.json`, và `hash_verification = "matched"`, `failed_expectations = []`.
`gallery_count = 74381` ở cả 8 summary.

Thành phần subset (từ `query_count_per_subset`, giống nhau ở 8 model):
`full = 6003`, `sf = 4069`, `validated = 586`.

### 4.2 Native FashionIQ là gì (đã đếm file thật)

Annotation native nằm ở `data/FashionIQ/captions/cap.<category>.val.json`:

| file | số annotation |
|---|---:|
| `cap.dress.val.json` | 2017 |
| `cap.shirt.val.json` | 2038 |
| `cap.toptee.val.json` | 1961 |
| **tổng** | **6016** |

⇒ Khớp `expected_query_count` trong `NATIVE_EVAL_MATRIX.csv` ("6016 (measured dress 2017 + shirt 2038 + toptee 1961)").

Gallery native **không** phải 74381:

| cách dựng gallery native | số ảnh |
|---|---:|
| val-split pair-union (union `candidate` ∪ `target` của cả 3 category) | **8582** |
| theo từng category: dress / shirt / toptee | 2628 / 3089 / 2902 |
| `image_splits/split.{cat}.val.json` (dress+shirt+toptee) | 3817+6346+5373 = **15536** |

Cả ba đều **khác 74381**. Đây là các cohort khác nhau về bản chất.

### 4.3 Sáu protocol native, không phải một — trích từ `NATIVE_EVAL_MATRIX.csv`

| model | `native_protocol` |
|---|---|
| conesep | `fashioniq_full_gallery_ref_excluded` |
| habit | `fashioniq_full_gallery_ref_excluded` |
| intent | `fashioniq_full_gallery_ref_excluded` |
| ptha_mtst | `fashioniq_original_split` |
| tme | `fashioniq_original_split` |
| hint | `fashioniq_val_split` |
| encoder | `fashioniq_val_split` |
| limn | `fashioniq_val_split` |

Ba protocol này khác nhau ở **candidate universe** và **reference policy**
(`workbench/docs/FASHIONIQ_PROTOCOL_PARITY.md`, bảng "The three protocol families"):
`original_split` giữ reference trong ranking; `full_gallery_ref_excluded` loại reference;
`val_split` thu gallery về pair-union. Tài liệu đó nói thẳng: *"These are three different cohorts.
A target's rank — and therefore Recall@K — changes when the reference is removed, and changes again
when the gallery shrinks to the pair union."*

### 4.4 TUYÊN BỐ DỨT KHOÁT

> **Metric CIRCUS và metric native KHÔNG THỂ so sánh. Tuyệt đối không được đặt chung một leaderboard.**

Lý do, mỗi lý do đều đủ để phá so sánh:

1. **Gallery khác**: CIRCUS 74,381 ảnh; native 8,582 (val-split pair-union) hoặc 15,536 (full split).
   Xác suất trúng ngẫu nhiên R@10 chênh nhau ~8.7× (10/8582 vs 10/74381).
2. **Query khác**: CIRCUS 6,003 (M-BEIR test); native 6,016 (FashionIQ val annotation).
   Hai tập query không đồng nhất, không phải superset của nhau.
3. **Reference policy khác**: CIRCUS **reference-eligible** (reference được xếp hạng bình thường);
   native **mask reference** bằng `-10e10` trước `argsort`. Spec nói rõ việc không kế thừa mask
   là **cố ý**: *"The native `-10e10` reference mask from ENCODER/LIMN val-split replays is
   deliberately not inherited"* (`CIRCUS_COMPOSITION_SPEC.md:11`).
4. **Cohort khác**: 6 protocol native khác nhau giữa các model (§4.3); CIRCUS dùng một protocol duy nhất cho cả 8.
5. **Đơn vị/định nghĩa**: CIRCUS báo `composition_gap_ndcg = 1 - max(I,T)/MM` — một đại lượng
   **không tồn tại** trong evaluation native nào. Trộn nó vào bảng native là trộn hai định nghĩa.

Hệ quả thực hành: bảng `COMPOSITION_MATRIX.csv` (CIRCUS) và `NATIVE_EVAL_MATRIX.csv` (native)
phải đứng **riêng**. Mọi câu kiểu "model X giỏi hơn model Y vì R@10 CIRCUS cao hơn" là vô nghĩa
nếu hai số lấy từ hai bảng khác nhau. Trong chính bảng CIRCUS, so sánh giữa 8 model là hợp lệ
(cùng protocol, cùng gallery, cùng cohort) — trừ `ptha_mtst` vốn là **CIRR→FashionIQ transfer**,
không phải FashionIQ reproduction, nên không phải "paper row" của FashionIQ.

### 4.5 Preprocessing của composition adapter: khớp native, hay là adaptation cố ý?

**Trả lời: là ADAPTATION CỐ Ý, không phải mô phỏng native.** Đọc `CIRCUS_COMPOSITION_SPEC.md`,
vị trí khai báo rõ ràng:

- `:5` — *"This **opt-in** evaluation asks whether specialized CIR models benefit from composing
  the reference image and modification text on the released CIRCUS subsets"* — đây là câu hỏi
  nghiên cứu **mới**, không phải tái lập native.
- `:11` — *"This workbench uses the released-artifact protocol. It does **not** claim exact
  reproduction of the paper's stated 18,853-image setting. No 18,853-image subset is invented, and
  the local native gallery is **not silently substituted**."*
- `:11` — mask reference của native bị **cố ý không kế thừa**.
- `:23` — can thiệp black-image thay **RAW image trước transform của model**, đúng
  `subset_evaluation/eval_subset.py::apply_ablation`; *"A zero tensor after normalization or a
  zeroed feature vector is a different intervention and is not used here."*
- `:5` — cảnh báo diễn giải: *"Report any difference as **input-ablation dependence under this
  protocol**, not proof of intrinsic compositional reasoning. Specialized models were not trained
  on blank images or empty text, so a large gap may partly reflect distribution shift."*

Trường `preprocessing` trong cả 8 `full.json` ghi đúng hai câu đó:

```json
{"policy": "adapter-native transform after path-based raw-image intervention",
 "ablation_policy": "black PIL replacement is staged before adapter transform; empty text for image-only"}
```

Nghĩa là: **transform ảnh là của chính model** (adapter-native), nhưng **gallery, query set,
reference policy và ba điều kiện can thiệp là của CIRCUS**, không phải của evaluator native.
Đây là thiết kế có chủ đích và có khai báo, **không** phải nỗ lực tái lập native bị sai.
Vì vậy càng không thể so metric CIRCUS với metric native: chúng trả lời hai câu hỏi khác nhau.

---

## 5. Kiểm tra mất query âm thầm

### 5.1 Đẳng thức phải giữ

Với mỗi model, mỗi điều kiện:
`query_count (6003) − scored_count == số query có failure reason được ghi`.

Đọc từ `runs/<run_id>/ranks/<stem>.json`:
`len(ranks[cond])`, đếm `value is None`, và `len(query_failures[cond])`.

| model | điều kiện | rank entries | null (unscored) | failure có reason | 6003 − scored | khớp |
|---|---|---:|---:|---:|---:|---|
| encoder | cả 3 | 6003 | 0 | 0 | 0 | ✅ |
| conesep | cả 3 | 6003 | 0 | 0 | 0 | ✅ |
| habit | cả 3 | 6003 | 0 | 0 | 0 | ✅ |
| intent | cả 3 | 6003 | 0 | 0 | 0 | ✅ |
| hint | cả 3 | 6003 | 0 | 0 | 0 | ✅ |
| **limn** | cả 3 | 6003 | **2** | **2** | **2** | ✅ |
| ptha_mtst | cả 3 | 6003 | 0 | 0 | 0 | ✅ |
| tme | cả 3 | 6003 | 0 | 0 | 0 | ✅ |

Quan trọng: `ranks[cond]` của limn vẫn có **6003 entry** (2 entry là `None`), **không** bị cắt bớt.
Nghĩa là không có query nào "biến mất" khỏi payload — nó ở đó với hạng `None` và lý do kèm theo.

### 5.2 Mọi failure đều có REASON, không có drop câm

Kiểm tra: mọi value trong `query_failures[cond]` là chuỗi không rỗng.
Kết quả: **0 failure nào thiếu reason** ở cả 8 model.

Limn — 2 query, cùng một lý do, ghi ở **cả 3 điều kiện**:

```
"7:3038": "query 7:3038: category unresolved has no adapter embedding scope ('dress', 'shirt', 'toptee'); batch_indices=(3037,), qids=['7:3038']"
"7:3790": "query 7:3790: category unresolved has no adapter embedding scope ('dress', 'shirt', 'toptee'); batch_indices=(3789,), qids=['7:3790']"
```

Đây là lý do **tất định** (deterministic), đúng nhóm "never retry" trong
`CIRCUS_COMPOSITION_SPEC.md:280` ("has no adapter embedding scope"). Không phải lỗi tạm thời.

Đối chiếu chéo 3 nơi cho limn, tất cả nhất quán:

| nơi ghi | giá trị |
|---|---|
| `ranks/*.json → query_failures` | 2 qid × 3 điều kiện |
| `summary/full.json → condition_status[*].unscored_count` | 2 (mỗi điều kiện) |
| `summary/full.json → adapter_query_scope_resolution.still_unscoped` | 2, `unscoped_qids = ["7:3038","7:3790"]` |
| `candidates/*/multimodal.manifest.json → missing_query_count` | 2, `partial: true`, examples đúng 2 qid |
| `metrics[*].scored_count` | 6001 |

Và **không** có mâu thuẫn với con số 119: `unresolved_category_query_count = 119` là **fact cấp
release** (số query ambiguous category), còn `adapter_query_scope_resolution` là **fact cấp adapter**
(limn route được 6001, còn 2 không scope nào). Runner ghi cả hai và chú thích rõ ở
`runner.py:2093-2095` rằng đây là hai đại lượng khác nhau, để "6001 scored" không bị đọc thành
mâu thuẫn. Xác nhận: `loader_scoped = 5884`, `adapter_routed = 6001`, `still_unscoped = 2`.

### 5.3 `topk_export.missing_query_count = 6` của limn — không phải lỗi

Trong `full.json`, limn có `topk_export.missing_query_count = 6`, nhưng mỗi manifest chỉ `2`.
Đây là **tổng trên 3 điều kiện**: `runner.py:2005-2006` cộng
`sum(int(item.get("missing_query_count") or 0) for item in manifests)` = 2 + 2 + 2 = 6.
Đúng theo thiết kế. Số **theo từng điều kiện** là 2, khớp manifest.

---

## 6. Tổng hợp phát hiện

| # | Phát hiện | Mức | Bằng chứng |
|---|---|---|---|
| 1 | 8/8 `run_id` khớp | ✅ OK | §1.1 |
| 2 | 16/16 ô `metrics_r10`/`metrics_r50` khớp | ✅ OK | §3.1 |
| 3 | Metric tính lại từ rank payload khớp 24/24 | ✅ OK | §1.4 |
| 4 | Top-K depth 100 đầy đủ 7/8; limn đúng partial 2 | ✅ OK | §2.1, §2.3 |
| 5 | `file_sha256` khớp 8/8 | ✅ OK | §2.2 |
| 6 | Không mất query âm thầm; limn có reason | ✅ OK | §5 |
| 7 | `identity_schema_version` **vắng** trong 8 `full.json` | ⚠️ ghi nhận | §1.2 |
| 8 | `policy_observed` **vắng** trong 8 artifact (predates commit) | ⚠️ ghi nhận | §1.2 |
| 9 | **Cột `blas_pin` của encoder quy `12727/18009` cho BLAS là SAI** — nguyên nhân chính là **đổi hàm `rank`** (`adapter_source_hashes.rank` `5f828b13…`→`f0914b12…`, `adapter_module_source_hash` `1a8b0dd2…`→`bf60d855…`), phụ là **đổi interpreter** (py3.11.15 `PYTHON_VERSION_MISMATCH` → py3.10.20 `verified`); A/B BLAS sạch chỉ đổi 1 rank/điều kiện, ΔR@50 = 0 | ❌ **LỖI** | §3.3 |
| 9b | 7 file đang nói sai "12,727 hạng đổi vì BLAS" cần sửa | ❌ **LỖI** | §3.3b |
| 10 | **Nhãn `plain einsum` của limn sai** — source dùng `@` (gemm), không phải naive einsum | ❌ **LỖI** | §3.4 |

**Không có metric nào bị bịa.** Hai lỗi (9, 10) đều nằm ở **cột mô tả cơ chế**, không ở cột số
đo lường. Metric vẫn đứng vững; điều sai là *lời giải thích* tại sao metric đổi.

---

## 7. Tái lập

Mọi con số trong báo cáo này đọc bằng lệnh dạng:

```bash
cd /home/heheboiz/data/cir
./workbench/envs/ctl312h/bin/python - <<'PY'
import json
j = json.load(open("workbench/artifacts/composition/summary/"
                   "encoder_fashioniq_5dee7425dd46c5f154808aac/full.json"))
print(j["execution_status"], j["query_count"], j["topk_export"])
PY
```

Log kiểm tra thô (nếu có) nằm ở `raw/composition_verify/`.
