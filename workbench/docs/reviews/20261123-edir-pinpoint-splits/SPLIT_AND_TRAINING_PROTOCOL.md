# Giao Thức Chia Tập (`train / val / test`) Chống Rò Rỉ & Quy Ước Huấn Luyện Cho EDIR và PinPoint

- **Ngày lập:** 2026-11-23
- **Seed cố định:** `42` | **Tỷ lệ mục tiêu:** `70% train / 10% val / 20% test`
- **Script sinh split tái lập:** [`workbench/scripts/build_edir_pinpoint_splits.py`](file:///home/heheboiz/data/cir/workbench/scripts/build_edir_pinpoint_splits.py)
- **Manifest bất biến (đã lưu phiên bản trong repo & bản sao trong `data/`):**
  - **EDIR:** [`edir_70_10_20_seed42.json`](file:///home/heheboiz/data/cir/workbench/docs/reviews/20261123-edir-pinpoint-splits/manifests/edir_70_10_20_seed42.json) (`SHA256: 9f4c320e0c16482063da1b36a5bf52267b95becacc36ba7a2c5cfd2405d88eda`)
  - **PinPoint:** [`pinpoint_70_10_20_seed42.json`](file:///home/heheboiz/data/cir/workbench/docs/reviews/20261123-edir-pinpoint-splits/manifests/pinpoint_70_10_20_seed42.json) (`SHA256: ee48900aa11798298e88029075d87fb084732e041641584a19ac2580c50635b4`)

---

## 1. Động Cơ Phương Pháp Nghiên Cứu

1. **Vấn đề khi eval zero-shot trực tiếp:** Toàn bộ ~15 mô hình CIR trong repo (`CLIP4Cir`, `SPRC`, `TME`, `CLVC-Net`, `TG-CIR`, `ENCODER`, `DCNet`, `LIMN`, `HINT`, `HABIT`, `Air-Know`, `ConeSep`, `INTENT`, `CSMCIR`, `PTHA+MTST`) đều là mô hình **supervised**, với checkpoint hiện có đã khớp chặt vào **FashionIQ** (miền quần áo) hoặc **CIRR**. Nếu mang trực tiếp checkpoint FashionIQ/CIRR đi đánh giá trên **EDIR** và **PinPoint**, lỗi thu được sẽ bị **nhiễu bởi domain shift** — không chứng minh được mô hình yếu do **giới hạn kiến trúc** hay do **lệch miền dữ liệu**.
2. **Mục tiêu của split in-domain:** Cả `EDIR-BENCH/EDIR` (`5,000` query) và `PinPoint` (`7,635` query) đều là benchmark đánh giá thuần (không phát hành sẵn `train / val / test`). Việc xây dựng bộ split **cố định, có kiểm toán chống rò rỉ dữ liệu** cho phép huấn luyện công bằng toàn bộ ~15 mô hình trên cùng một dữ liệu in-domain, từ đó tách bạch rõ **lỗi cố hữu của kiến trúc** (*intrinsic bottleneck*: phủ định/`remove`, `shape`, `viewpoint`, `spatial`, bẫy hard-negative, nhạy cảm paraphrase) khỏi lỗi lệch miền.
3. **Lưu ý so sánh số liệu:** Điểm số trên split `70/10/20` dùng để **so sánh công bằng nội bộ giữa các kiến trúc CIR** trong nghiên cứu này (không so sánh trực tiếp với con số full-test trong paper gốc của EDIR/PinPoint).

---

## 2. Kiểm Toán Đồ Thị Dữ Liệu & Chiến Lược Split Chống Rò Rỉ

### 2.1. EDIR (`data/EDIR`: `5,000` query, `28,645` ảnh corpus, `15` subcategory)

#### Kết quả kiểm toán cấu trúc dữ liệu
- **Mọi ảnh đích `pos` (`5,000/5,000`) và mọi ảnh hard-negative `neg` (`24,939/24,939`)** đều là ảnh chỉnh sửa sinh từ chính ảnh gốc `ref` (cùng `stem_base`, ví dụ `523_007655595.jpg` → `523_007655595_1.png`…`_6.png`).
- Đồ thị liên thông toàn cục trên tập (`ref`, `pos`, `neg`, `stem_base`) tách thành đúng **4,784 cụm hoàn toàn độc lập**: `4,574` cụm cỡ 1 query, `204` cụm cỡ 2 query, `6` cụm cỡ 3 query (`max = 3`). Không có bất kỳ ảnh nào dùng chung giữa 2 cụm khác nhau.
- Phân bố 15 subcategory: `complex` có `800` query; 14 subcategory còn lại (`color`, `material`, `shape`, `texture`, `addition`, `remove`, `replace`, `cardinality`, `spatial`, `action`, `viewpoint`, `style`, `time`, `weather`) có đúng `300` query mỗi loại.

#### Quy tắc chia & kết quả kiểm chứng
- **Chiến lược:** Gom cụm theo `stem_base` (4,784 cụm) + phân tầng đều (*stratified*) theo 15 subcategory với `seed = 42`.
- **Phân bố kích thước:**

| Split | Số query | Tỷ lệ | Số cụm `stem_base` độc lập |
|---|---:|---:|---:|
| **train** | **3,500** | 70.0% | 3,347 |
| **val** | **497** | 9.9% | 479 |
| **test** | **1,003** | 20.1% | 958 |
| **Tổng** | **5,000** | 100.0% | 4,784 |

- **Kiểm chứng chống rò rỉ (0% tuyệt đối ở mọi tầng):**

| Cặp split | Trùng `ref` | Trùng `pos` | Trùng `neg` | Trùng `stem_base` |
|---|---:|---:|---:|---:|
| `train` vs `val` | **0** | **0** | **0** | **0** |
| `train` vs `test` | **0** | **0** | **0** | **0** |
| `val` vs `test` | **0** | **0** | **0** | **0** |

- **Gallery khi đánh giá `test`:** Truy hồi trên **toàn bộ `28,645` ảnh corpus** (giữ nguyên độ khó chuẩn paper; vì `pos`/`neg` của tập `test` hoàn toàn không trùng tập `train`, không có rò rỉ ground-truth). Có thể báo cáo kèm chỉ số phụ trên test-only gallery (`~5,700` ảnh) để phân tích độ nhạy kích thước gallery.

---

### 2.2. PinPoint (`data/PinPoint`: `7,635` query tổng = `6,740` single-image + `895` multi-image, `109,598` ảnh local)

#### Kết quả kiểm toán cấu trúc dữ liệu (2 bẫy rò rỉ ẩn đã phát hiện)
1. **Bẫy 1 — Siêu hub ảnh gốc (`query_image_signature`):** Trong `6,740` query single-image (`1,081` ảnh `ref` duy nhất), có **5 ảnh `ref` siêu hub** được tái sử dụng cho lượng query rất lớn:
   - `391a1672ebd6c82b9926ef9980e96d84`: **754 query** (11.2% toàn tập single-image, trải trên 5 category)
   - `f8add0f0e359e6886b9762291aa621a4`: **113 query**
   - `5b36b46e4623165a200b76a2b8647adc`: **89 query**
   - `269fdb892d9a4decbf0ada9b4576d6d4`: **77 query**
   - `e1d009f37b767777ae5fd49e7bd2914a`: **53 query**
   - Nếu chia ngẫu nhiên, cùng 1 ảnh gốc và 6 bản paraphrase của nó sẽ rơi vào cả `train` và `test`. Nếu gom nhóm nhưng để hub `754` rơi vào `val` (~674 query), toàn bộ `val` sẽ chỉ gồm 1 ảnh gốc duy nhất.
2. **Bẫy 2 — Dùng chung ảnh ứng viên (`pos`/`neg`) trên kho Pinterest:** Trong tập single-image, **44.3%** ảnh `pos` (`4,215/9,518`) và **60.3%** ảnh `neg` (`15,404/25,542`) được chia sẻ giữa nhiều nhóm `ref` khác nhau (vì nhiều query cùng chủ đề `Women's Fashion`, `Home Decor`, `Art` trỏ về chung tập ảnh đích). Đồ thị `ref + pos` kết thành một khối khổng lồ `5,368/6,740` query (80%), nên **về mặt toán học không thể vừa bảo toàn phân bố category vừa đạt `0%` trùng ảnh `pos`**.

#### Quy tắc chia & kết quả kiểm chứng
- **Phạm vi split chính:** `6,740` query single-image (`is_single = True`) dùng chung cho toàn bộ ~15 mô hình CIR.
- **Tập đánh giá phụ đa ảnh (`multi_image_eval`):** `895` query multi-image (`is_single = False`, `894` thuộc `Composition` + `1` thuộc `Complement`) giữ tách riêng hoàn toàn cho các mô hình hỗ trợ 2 ảnh đầu vào.
- **Chiến lược chống leak (`seed = 42`):**
  1. **Bắt buộc `0%` trùng ảnh `ref` & `0%` rò rỉ paraphrase:** Gom toàn bộ query cùng `query_image_signature` vào chung một split.
  2. **Ghim 5 siêu hub `ref` (`≥ 50` query/ảnh, tổng `1,086` query) vào `train`:** Đảm bảo `val` (`118` ảnh `ref` khác nhau) và `test` (`224` ảnh `ref` khác nhau) có độ đa dạng ảnh gốc cao, không bị chi phối bởi một vài ảnh lặp.
  3. **Gán tham lam phân tầng kết hợp phạt trùng `pos` (`min pos-overlap`):** Giữ đúng tỷ lệ `70/10/20` theo từng `query_category` đồng thời tối thiểu hóa lượng ảnh `pos` trùng chéo giữa các split.

- **Phân bố kích thước & category:**

| Split | Tổng query | Tỷ lệ | Số ảnh `ref` duy nhất | `Swap` | `Context Fit` | `Negation` | `Complement` | `Explore` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **train** | **4,724** | 70.1% | 739 | 2,386 | 1,750 | 243 | 206 | 139 |
| **val** | **679** | 10.1% | 118 | 339 | 262 | 32 | 30 | 16 |
| **test** | **1,337** | 19.8% | 224 | 673 | 508 | 53 | 70 | 33 |
| **Tổng (single)** | **6,740** | 100.0% | 1,081 | 3,398 | 2,520 | 328 | 306 | 188 |
| *`multi_image_eval`* | *895* | — | *179 cặp* | — | — | — | *1* | *894 (`Composition`)* |

- **Kiểm chứng chống rò rỉ & minh bạch độ trùng ứng viên:**

| Cặp split | Trùng `ref` / paraphrase | Trùng `pos` (% tập sau) | Trùng `neg` (% tập sau) |
|---|---:|---:|---:|
| `train` vs `val` | **0 (0.00%)** | 157 (**15.33%**) | 1,563 (43.66%) |
| `train` vs `test` | **0 (0.00%)** | 338 (**15.45%**) | 2,799 (42.29%) |
| `val` vs `test` | **0 (0.00%)** | 29 (**1.33%**) | 503 (7.60%) |

> Nhờ tối ưu phạt trùng `pos`, tỷ lệ trùng ảnh `pos` giữa `train` và `test` giảm gần **3 lần** so với mức trùng tự nhiên `44.3%` của đồ thị gốc, xuống chỉ còn **15.45%**, trong khi vẫn đạt **0% rò rỉ ảnh `ref` và 0% rò rỉ paraphrase**.

- **Tác động của 2 ảnh lỗi `HTTP 403` (`data/PinPoint/failed_signatures.txt`):**
  - 2 ảnh không tải được (`1269714db704dc89eb1fcfe18fbc8c4e`, `a5b5625b26942bd4d5cd5513b235bc83`) chỉ xuất hiện ở `negative_candidates` (`1/50` hard negative) của 18 query single-image (`train: 12`, `val: 6`, **`test: 0`**) và 5 query trong `multi_image_eval`.
  - **Tập `test` (`1,337` query) có 0 query bị ảnh hưởng — đủ 100% ảnh `ref`, `pos` và `neg`.**

---

## 3. Quy Ước Khởi Tạo Trọng Số & Huấn Luyện Chuẩn Cho ~15 Mô Hình

### 3.1. Nguyên tắc cốt lõi: Khởi tạo từ **Pretrained VLM / Backbone gốc của từng paper** (KHÔNG dùng checkpoint FashionIQ / CIRR)
- **Đúng như cách các paper gốc huấn luyện trên FashionIQ và CIRR:**
  1. **Giữ nguyên trọng số pretrained gốc** của Vision-Language / Vision backbone theo đúng thiết kế từng paper (CLIP, BLIP-2, hoặc ResNet-ImageNet + GloVe).
  2. **Khởi tạo mới (from scratch)** toàn bộ phần **CIR composition head / combiner / projector / adapter / prompt tokens** của từng mô hình.
  3. **Huấn luyện trực tiếp trên `train` của EDIR hoặc PinPoint**, chọn checkpoint tốt nhất bằng **early stopping trên `val`**, và chỉ đánh giá cuối cùng đúng 1 lần trên `test`.
- **Vì sao quy ước này công bằng nhất?**
  - **Không** khởi tạo random toàn bộ CLIP/BLIP-2 (bất khả thi ở quy mô vài nghìn query).
  - **Không** fine-tune nối tiếp từ checkpoint đã train trên FashionIQ hoặc CIRR (tránh mang theo thiên kiến miền quần áo/NLVR2 và tránh bất công giữa các mô hình có/không có checkpoint CIRR).
  - **Tín hiệu học đủ giàu ở quy mô 70% train:**
    - `PinPoint train` (`4,724` query × ~`9` positive/query) cung cấp **~42,000 cặp `(query, pos)`** + `50` hard negative/query (giàu tín hiệu hơn cả CIRR).
    - `EDIR train` (`3,500` query × `5` hard negative cùng ảnh gốc/query) cung cấp tín hiệu contrastive rất sắc theo đúng 15 trục edit.

### 3.2. Phân nhóm khởi tạo backbone cho 15 paper trong repo

| Nhóm backbone khởi tạo gốc | Các mô hình trong repo | Lưu ý huấn luyện trên EDIR / PinPoint |
|---|---|---|
| **BLIP-2 (Q-Former + ViT-G/L, `lavis`)** | `SPRC`, `TME`, `CSMCIR`, `PTHA+MTST`, `HINT`, `HABIT`, `Air-Know`, `ConeSep`, `INTENT`, `ENCODER`, `LIMN` | Khởi tạo từ checkpoint pretrained BLIP-2 gốc (`blip2_pretrained` / `blip2_feature_extractor`); train phần composition/adapter của từng paper theo đúng optimizer/LR schedule chuẩn + early stopping trên `val`. |
| **CLIP (`RN50x4` / `ViT-L/14`)** | `CLIP4Cir` (`Combiner` / `FullFT`), `TG-CIR` | Khởi tạo từ trọng số OpenAI CLIP gốc; với nhánh `FullFT` / unfreeze backbone cần theo dõi sát `val` và dùng LR nhỏ + weight decay để chống overfit trên 3.5k query EDIR. |
| **Classic CNN + RNN/GloVe (`ResNet` + `LSTM`/`GloVe`)** | `DCNet`, `CLVC-Net` | Khởi tạo từ `ResNet` ImageNet + `GloVe 300d`; xây bộ từ vựng (vocab) chỉ trên `train` (không đụng `val`/`test`). Kỳ vọng lộ rõ điểm yếu OOV và data-efficiency — đây là đối chứng giá trị khi so với nhóm VLM. |

### 3.3. Quy tắc đánh giá thống nhất
- **Không chạm vào `test` trong quá trình tuning hyperparameter** (chỉ dùng `val` để chọn epoch/checkpoint tốt nhất).
- **Báo cáo phân rã theo trục lỗi (*failure breakdown*):**
  - Trên **EDIR `test` (`1,003` query)**: báo cáo `Recall@1`, `Recall@5`, `Recall@10` tổng thể và **riêng từng nhóm trong 15 subcategory** (`remove`, `shape`, `viewpoint`, `spatial`, `cardinality`, `color`, …).
  - Trên **PinPoint `test` (`1,337` query)**: chạy chính `pinpoint-dataset/evaluate.py` để báo cáo `Precision@1`, `Precision@10`, `mAP@10`, **`NegRecall@10`**, `mAP@10_noNeg`, `delta_mAP@10_noNeg`, và độ nhạy paraphrase (**`ling_sens_range`**, **`ling_sens_std`**) theo từng `query_category` (`Swap`, `Context Fit`, `Negation`, `Complement`, `Explore`).
- **Đối chứng tùy chọn (tách domain-shift vs lỗi cố hữu):** Chạy thêm 1 lượt eval **trước khi train** (bằng backbone gốc hoặc checkpoint CIRR/FashionIQ) trên đúng cùng tập `test` để đo chính xác độ cải thiện *before vs after in-domain training* trên từng loại lỗi.
