# Giao Thức Chia Tập (`train / val / test`) Chống Rò Rỉ & Quy Ước Huấn Luyện Cho EDIR và PinPoint

- **Ngày lập:** 2026-11-23
- **Seed cố định:** `42` | **Tỷ lệ mục tiêu:** `70% train / 10% val / 20% test`
- **Script sinh split tái lập:** [`workbench/scripts/build_edir_pinpoint_splits.py`](file:///home/heheboiz/data/cir/workbench/scripts/build_edir_pinpoint_splits.py)
- **Manifest bất biến (đã lưu phiên bản trong repo & bản sao trong `data/`):**
  - **EDIR:** [`edir_70_10_20_seed42.json`](file:///home/heheboiz/data/cir/workbench/docs/reviews/20261123-edir-pinpoint-splits/manifests/edir_70_10_20_seed42.json) (`SHA256: 659e9329bc12d4ca93e6b7cfe65c954334067c6a669aa5f7ad23a6aa28eb7c75`)
  - **PinPoint:** [`pinpoint_70_10_20_seed42.json`](file:///home/heheboiz/data/cir/workbench/docs/reviews/20261123-edir-pinpoint-splits/manifests/pinpoint_70_10_20_seed42.json) (`SHA256: 15a25e30cd76d6fc9cc0f1fab985feea46022b5571bd0024b68d04a214896f3c`)

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

- **Kiểm toán toàn vẹn ảnh & làm sạch dữ liệu bẩn (đã tự động xử lý trong bản export chuẩn):**
  - **100% ảnh đọc hợp lệ bằng PIL (`0` ảnh hỏng / `0-byte`)**: `33,429/33,429` ảnh EDIR và `109,598/109,598` ảnh PinPoint.
  - **5 lỗi dữ liệu bẩn phát hiện trong `pinpoint_licensed.parquet` gốc (và cách xử lý kép):**
    1. `6,729` phần tử `None/null` trong mảng `negative_candidates` → loại bỏ khỏi `negative_ids` dùng khi train.
    2. `2,719` bản ghi trùng lặp trong `positive_candidates` và `8,794` bản ghi trùng lặp trong `negative_candidates` → khử trùng lặp theo thứ tự gốc (còn đúng **`39,886`** cặp `(query, pos)` sạch duy nhất trong `PinPoint train`).
    3. `476` trường hợp cùng một ảnh nằm đồng thời ở cả `positive_candidates` và `negative_candidates` (`310` query) → loại khỏi `negative_ids` khi train để contrastive loss không đẩy ảnh đích ra xa.
    4. `23` tham chiếu tới 2 ảnh `HTTP 403` trong `negative_candidates` → loại khỏi `negative_ids` và `corpus_full.jsonl` (`109,598` ảnh tồn tại trên đĩa).
    5. `111` dòng có `is_single=True` nhưng cột `query_image_signature2` khác `None` → giữ `ref2_id=None` khi train/infer single-image, đồng thời bảo toàn `query_image_signature2_raw` trong `metadata` để khớp `ling_sens` bit-exact với `pinpoint-dataset/evaluate.py`.
  - **Bảo toàn song song `positive_ids_raw` / `negative_ids_raw`:** Khi đánh giá bằng `evaluate_pinpoint(..., use_raw_gt=True)`, mọi chỉ số khớp **bit-exact (`≤ 1e-12`)** với `pinpoint-dataset/evaluate.py` gốc.
  - **Lưu ý quy ước gallery PinPoint:** Ảnh `ref` nằm sẵn trong `index_signatures.txt` và là hard-negative tường minh cho `4,217` query — giống `pinpoint-dataset/run_retrieval.py`, **không** mask `ref` khi truy hồi PinPoint.

---

## 3. Quy Ước Khởi Tạo Trọng Số & Huấn Luyện Chuẩn Cho ~15 Mô Hình

### 3.1. Nguyên tắc cốt lõi: Khởi tạo từ **Pretrained VLM / Backbone gốc của từng paper** (KHÔNG dùng checkpoint FashionIQ / CIRR)
- **Đúng như cách các paper gốc huấn luyện trên FashionIQ và CIRR:**
  1. **Giữ nguyên trọng số pretrained gốc** của Vision-Language / Vision backbone theo đúng thiết kế từng paper (CLIP, BLIP-2, hoặc ResNet-ImageNet + GloVe).
  2. **Khởi tạo mới (from scratch)** toàn bộ phần **CIR composition head / combiner / projector / adapter / prompt tokens** của từng mô hình.
  3. **Giữ nguyên cơ chế freeze / unfreeze của từng repo gốc** (repo nào code gốc cho unfreeze/train tiếp Q-Former hoặc CLIP thì vẫn train tiếp đúng như code gốc; repo nào freeze backbone thì giữ nguyên freeze).
  4. **Huấn luyện trực tiếp trên `train` của EDIR hoặc PinPoint**, chọn checkpoint tốt nhất bằng **early stopping trên `val`**, và chỉ đánh giá cuối cùng đúng 1 lần trên `test`.
- **Vì sao quy ước này công bằng nhất?**
  - **Không** khởi tạo random toàn bộ CLIP/BLIP-2 (bất khả thi ở quy mô vài nghìn query).
  - **Không** fine-tune nối tiếp từ checkpoint đã train trên FashionIQ hoặc CIRR (tránh mang theo thiên kiến miền quần áo/NLVR2 và tránh bất công giữa các mô hình có/không có checkpoint CIRR).
  - **Tín hiệu học đủ giàu ở quy mô 70% train:**
    - `PinPoint train` (`4,724` query) cung cấp đúng **`39,886` cặp `(query, pos)` sạch duy nhất** + hard-negative tường minh (giàu tín hiệu hơn cả CIRR).
    - `EDIR train` (`3,500` query × `5` hard negative cùng ảnh gốc/query) cung cấp tín hiệu contrastive rất sắc theo đúng 15 trục edit.

### 3.2. Phân nhóm khởi tạo backbone cho 15 paper trong repo

| Nhóm backbone khởi tạo gốc | Các mô hình trong repo | Lưu ý huấn luyện trên EDIR / PinPoint |
|---|---|---|
| **BLIP-2 (Q-Former + ViT-G/L, `lavis`)** | `SPRC`, `TME`, `CSMCIR`, `PTHA+MTST`, `HINT`, `HABIT`, `Air-Know`, `ConeSep`, `INTENT`, `ENCODER`, `LIMN` | Khởi tạo từ checkpoint pretrained BLIP-2 gốc (`blip2_pretrained` / `blip2_feature_extractor`); giữ nguyên cơ chế unfreeze Q-Former/adapter theo code tác giả + early stopping trên `val`. |
| **CLIP (`RN50x4` / `ViT-L/14`)** | `CLIP4Cir` (`Combiner` / `FullFT`), `TG-CIR` | Khởi tạo từ trọng số OpenAI CLIP gốc; giữ nguyên freeze (`Combiner noft`) hoặc unfreeze (`FullFT`, `TG-CIR`) theo code gốc, theo dõi sát `val` để chống overfit trên 3.5k query EDIR. |
| **Classic CNN + RNN/GloVe (`ResNet` + `LSTM`/`GloVe`)** | `DCNet`, `CLVC-Net` | Khởi tạo từ `ResNet` ImageNet + `GloVe 300d`; xây bộ từ vựng (vocab) chỉ trên `train` (không đụng `val`/`test`). Kỳ vọng lộ rõ điểm yếu OOV và data-efficiency — đây là đối chứng giá trị khi so với nhóm VLM. |

### 3.3. Quy tắc đánh giá thống nhất
- **Không chạm vào `test` trong quá trình tuning hyperparameter** (chỉ dùng `val` để chọn epoch/checkpoint tốt nhất).
- **Báo cáo phân rã theo trục lỗi (*failure breakdown*):**
  - Trên **EDIR `test` (`1,003` query)**: báo cáo `Recall@1`, `Recall@5`, `Recall@10`, `Recall@50`, `MRR` tổng thể, **macro-average** và **riêng từng nhóm trong 15 subcategory** (`remove`, `shape`, `viewpoint`, `spatial`, `cardinality`, `color`, …).
  - Trên **PinPoint `test` (`1,337` query)**: báo cáo `precision@1`, `precision@10`, `mAP@10`, **`NegRecall@10`**, `mAP@10_noNeg`, `delta_mAP@10_noNeg`, `ling_sens_range`, `ling_sens_std` tổng thể và theo từng `query_category` (`Swap`, `Context Fit`, `Negation`, `Complement`, `Explore`).
- **Đối chứng tùy chọn (tách domain-shift vs lỗi cố hữu):** Chạy thêm 1 lượt eval **trước khi train** (bằng backbone gốc hoặc checkpoint CIRR/FashionIQ) trên đúng cùng tập `test` để đo chính xác độ cải thiện *before vs after in-domain training* trên từng loại lỗi.

---

## 4. Hạ Tầng Dùng Chung Cho Toàn Bộ ~15 Repo (`workbench/benchmarks/`)

Để **mọi repo** (dù chạy Python `3.9`, `3.10` hay `3.12`, không cần cài `duckdb`/`pyarrow`/`pandas`) đều đọc đúng cùng một bộ split bất biến với SHA256 kiểm chứng:

- **Danh sách ID chuẩn theo dõi bằng git** (`workbench/docs/reviews/20261123-edir-pinpoint-splits/manifests/`):
  - `edir_train_ids.txt` (`3,500`), `edir_val_ids.txt` (`497`), `edir_test_ids.txt` (`1,003`), `edir_70_10_20_seed42.json`
  - `pinpoint_train_ids.txt` (`4,724`), `pinpoint_val_ids.txt` (`679`), `pinpoint_test_ids.txt` (`1,337`), `pinpoint_multi_image_eval_ids.txt` (`895`), `pinpoint_70_10_20_seed42.json`
- **Bản export JSONL tự chứa kèm `manifest.json` (SHA256 mọi file):**
  - `data/EDIR/splits/70_10_20_seed42/` (`train/val/test.jsonl`, `*_ids.txt`, `corpus_full.jsonl`, `corpus_{train,val,test}.jsonl`)
  - `data/PinPoint/splits/70_10_20_seed42/` (`train/val/test/multi_image_eval.jsonl`, `*_ids.txt`, `corpus_full.jsonl`, `corpus_*.jsonl`)
- **Module Python dùng chung (`workbench/benchmarks/`):**
  - [`schema.py`](file:///home/heheboiz/data/cir/workbench/benchmarks/schema.py): `verify_split_integrity()`, `load_split_ids()`, `load_split_bundle()`
  - [`datasets.py`](file:///home/heheboiz/data/cir/workbench/benchmarks/datasets.py): `CIRTrainingDataset` (`positive_mode="random" | "expand" | "first"`, `num_hard_negatives`), `CIRQueryEvalDataset`, `CIRCorpusDataset`, `build_dataloaders()`
  - [`evaluators.py`](file:///home/heheboiz/data/cir/workbench/benchmarks/evaluators.py): `rank_embeddings()` (deterministic `kind="stable"`), `evaluate_edir()`, `evaluate_pinpoint()` (khớp bit-exact `pinpoint-dataset/evaluate.py`)
  - Test kiểm chứng: [`workbench/tests/test_edir_pinpoint_benchmarks.py`](file:///home/heheboiz/data/cir/workbench/tests/test_edir_pinpoint_benchmarks.py)

```python
from workbench.benchmarks import build_dataloaders, rank_embeddings, evaluate_edir, evaluate_pinpoint

loaders = build_dataloaders(
    dataset="edir",                  # hoặc "pinpoint"
    train_image_transform=preprocess_train,
    eval_image_transform=preprocess_val,
    batch_size=32,
    eval_batch_size=64,
    num_workers=4,
    positive_mode="expand",          # "random" (1 pos/query) hoặc "expand" (tất cả cặp pos)
    num_hard_negatives=0,            # >0 nếu loss của repo dùng hard negative tường minh
    corpus_scope="full",             # "full" (mặc định) hoặc "split"
    verify_checksums=True,           # tự kiểm tra SHA256 manifest trước khi train
)
```
