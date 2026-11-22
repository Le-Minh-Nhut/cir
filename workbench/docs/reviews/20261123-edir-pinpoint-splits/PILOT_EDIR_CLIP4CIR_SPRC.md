# Kết Quả Pilot Trên EDIR (`70/10/20`, `seed=42`): `CLIP4Cir (RN50x4)` vs `SPRC (BLIP-2)`

- **Ngày hoàn tất:** 2026-11-23
- **Split bất biến:** [`edir_70_10_20_seed42.json`](file:///home/heheboiz/data/cir/workbench/docs/reviews/20261123-edir-pinpoint-splits/manifests/edir_70_10_20_seed42.json) (`3,500 train / 497 val / 1,003 test`, **0% leak**)
- **Đánh giá cuối cùng trên `test`:** toàn bộ **`28,645` ảnh `corpus_full`**, `strict=True`, `kind="stable"`.
- **Script huấn luyện chuẩn (khởi tạo từ pretrained VLM gốc, không dùng checkpoint FashionIQ/CIRR):**
  - [`train_clip4cir.py`](file:///home/heheboiz/data/cir/workbench/benchmarks/trainers/train_clip4cir.py)
  - [`train_sprc.py`](file:///home/heheboiz/data/cir/workbench/benchmarks/trainers/train_sprc.py)
  - [`common.py`](file:///home/heheboiz/data/cir/workbench/benchmarks/trainers/common.py)

---

## 1. Cấu Hình Thực Chạy & Kiểm Chứng Kỹ Thuật

| Mô hình | Backbone gốc | Micro-BS / Accum | Peak VRAM | Best Epoch (chọn theo `val`) | Early Stop | Scorer Self-Check |
|---|---|---|---:|---|---|---|
| **CLIP4Cir (`RN50x4`)** | OpenAI CLIP `RN50x4` | Stage 1: `32 / 1`; Stage 2: `256 / 1` | `11,711 MiB` | S1: Ep `7` / S2: Ep `5` | S1: Ep `11` / S2: Ep `11` | Exact cosine `rank_embeddings` |
| **SPRC (`pretrain`)** | BLIP-2 `EVA-ViT-g` (frozen) + Q-Former (`186.7M` trainable) | **`48 / 1`** (auto-probe `64→48`) | `11,336 MiB` | Ep **`17`** (`val=0.7498`) | Ep **`22`** (`patience=5`) | `max_abs` vs `blip_model.inference()` = **`1.49e-08`** |

---

## 2. Kết Quả Tổng Hợp Trên EDIR `test` (`1,003` query, `28,645` corpus)

| Mô hình / Mốc | Overall `R@1` | Overall `R@5` | Overall `R@10` | Overall `R@50` | `MRR` | Macro-15 `R@1` | Macro-15 `R@10` |
|---|---:|---:|---:|---:|---:|---:|---:|
| Zero-Shot CLIP `RN50x4` (trước train) | 28.32% | 72.28% | 84.25% | 95.71% | 0.4636 | 28.14% | 82.74% |
| CLIP4Cir Stage 1 (`CLIP` fine-tuned) | 27.92% | 80.76% | 93.42% | 98.70% | 0.4815 | 28.55% | 93.08% |
| **CLIP4Cir Final (`+Combiner`)** | **27.02%** | **82.35%** | **95.51%** | **98.90%** | **0.4785** | **27.39%** | **95.22%** |
| Zero-Shot BLIP-2 (`SPRC` Ep 0) | 3.19% | 10.77% | 14.96% | 25.62% | 0.0666 | 2.62% | 13.70% |
| **SPRC Final (`BLIP-2 Q-Former`)** | **36.19%** | **85.84%** | **96.51%** | **99.20%** | **0.5516** | **37.29%** | **96.26%** |

---

## 3. Phân Rã Theo 15 Subcategory (`test`, sắp theo `SPRC R@1` tăng dần)

| Subcategory | N | `CLIP4Cir R@1` | `SPRC R@1` | Chênh (`SPRC − CLIP4Cir`) | `CLIP4Cir R@10` | `SPRC R@10` |
|---|---:|---:|---:|---:|---:|---:|
| **remove** | 62 | 19.35% | **17.74%** | -1.61% | 90.32% | 95.16% |
| **shape** | 61 | 34.43% | **22.95%** | -11.48% | 96.72% | 96.72% |
| **viewpoint** | 60 | 11.67% | **25.00%** | +13.33% | 96.67% | 98.33% |
| **complex** | 160 | 24.38% | **27.50%** | +3.12% | 98.12% | 99.38% |
| **texture** | 61 | 32.79% | **27.87%** | -4.92% | 96.72% | 96.72% |
| **spatial** | 61 | 26.23% | **32.79%** | +6.56% | 96.72% | 96.72% |
| **cardinality** | 57 | 26.32% | **36.84%** | +10.52% | 96.49% | 98.25% |
| **material** | 61 | 22.95% | **37.70%** | +14.75% | 93.44% | 91.80% |
| **time** | 60 | 30.00% | **40.00%** | +10.00% | 95.00% | 96.67% |
| **weather** | 60 | 15.00% | **40.00%** | +25.00% | 91.67% | 95.00% |
| **style** | 64 | 25.00% | **40.62%** | +15.62% | 98.44% | 98.44% |
| **action** | 59 | 37.29% | **45.76%** | +8.47% | 100.00% | 98.31% |
| **color** | 58 | 41.38% | **48.28%** | +6.90% | 89.66% | 89.66% |
| **replace** | 62 | 29.03% | **54.84%** | +25.81% | 93.55% | 91.94% |
| **addition** | 57 | 35.09% | **61.40%** | +26.31% | 94.74% | 100.00% |

---

## 4. Nhận Định Nghiên Cứu Bước Đầu Từ 2 Pilot

1. **Tách bạch hoàn toàn domain-shift vs lỗi cố hữu:** Cả hai mô hình sau khi train in-domain đều đạt **`95.5% – 96.5% Recall@10`** (`99.4%` ở `complex`), chứng minh khả năng truy hồi đúng cụm ảnh gốc rất mạnh; khoảng trống lớn giữa `R@10 (~96%)` và `R@1 (27%–36%)` phản ánh trực tiếp **khó khăn phân biệt tinh giữa 6 biến thể edit trên cùng một cảnh gốc**.
2. **Sức mạnh của early cross-attention (`SPRC / BLIP-2`) vs late fusion (`CLIP4Cir`):** `SPRC` vượt `CLIP4Cir` tới **`+9.90%` Macro `R@1`** (`+26.3%` ở `addition`, `+25.8%` ở `replace`, `+25.0%` ở `weather`, `+15.6%` ở `style`).
3. **Bất đối xứng phủ định cực đoan (`addition 61.4%` vs `remove 17.7%`):** Kể cả khi `SPRC` (1.17B params) đã được train in-domain 17 epoch và đạt `61.40%` ở `addition`, nó vẫn **sụp xuống `17.74%` ở `remove`** (ngang mức đoán ngẫu nhiên `1/6 = 16.7%`, thấp hơn cả `CLIP4Cir 19.35%`) — cho thấy lỗ hổng cố hữu của cơ chế contrastive chuẩn khi đối tượng cần *xóa* vẫn xuất hiện tường minh trong văn bản.
4. **Hành vi khác biệt ở `shape` & `texture`:** `CLIP4Cir` giữ `R@1` cao hơn `SPRC` ở `shape` (`34.43%` vs `22.95%`) và `texture` (`32.79%` vs `27.87%`), là tín hiệu thú vị cần kiểm chứng tiếp trên các mô hình BLIP-2 và CLIP còn lại.

---

## 5. Artifact Đã Lưu
- `CLIP4Cir`: `workbench/artifacts/benchmark_runs/edir/70_10_20_seed42/clip4cir_rn50x4/{config.json,history.jsonl,test_results.json,test_predictions_top50.json}`
- `SPRC`: `workbench/artifacts/benchmark_runs/edir/70_10_20_seed42/sprc_pretrain/{config.json,history.jsonl,test_results.json,test_predictions_top50.json}`
