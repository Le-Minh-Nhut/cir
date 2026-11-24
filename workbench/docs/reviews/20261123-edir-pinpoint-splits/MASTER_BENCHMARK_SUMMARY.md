# Báo Cáo Tổng Hợp Thực Nghiệm Toàn Diện: EDIR & PinPoint (In-Domain vs Cross-Benchmark Transfer)

- **Ngày hoàn tất:** `2026-11-25` | **Split chuẩn duy nhất:** `70_10_20_seed42` (zero data leakage)
- **Tổng số lượt đánh giá hoàn tất:** **`73 / 73` (`0` lỗi, `16` model CIR)** — tất cả đều đã kiểm chứng độc lập bit-exact từ `test_predictions_top50.json`: **`max_abs_diff = 0.00e+00`**
  - **EDIR In-Domain:** `16 / 16` ([LEADERBOARD_EDIR.md](LEADERBOARD_EDIR.md))
  - **PinPoint In-Domain:** `16 / 16` ([LEADERBOARD_PINPOINT.md](LEADERBOARD_PINPOINT.md))
  - **FashionIQ → PinPoint Transfer:** `15 / 15` ([TRANSFER_AND_INDOMAIN_PINPOINT.md](TRANSFER_AND_INDOMAIN_PINPOINT.md))
  - **CIRR → EDIR Transfer:** `11 / 11` ([TRANSFER_AND_INDOMAIN_EDIR.md](TRANSFER_AND_INDOMAIN_EDIR.md))
  - **FashionIQ → EDIR Transfer:** `15 / 15` ([TRANSFER_AND_INDOMAIN_EDIR.md](TRANSFER_AND_INDOMAIN_EDIR.md))
- **Ghi chú checkpoint & tái lập QuRe (ICML 2025, `jaehyun-kwak/QuRe`, commit `6a50a27`):**
  - Hai thư mục Google Drive checkpoint FashionIQ & CIRR của tác giả đều trống (`jaehyun-kwak/QuRe#3`), do đó đã train tái lập chuẩn nguyên bản từ `blip2_pretrained_vitL` trên dữ liệu gốc với kết quả khớp tuyệt đối paper:
    - **FashionIQ (`30` ep, best `Ep 20`):** `Avg Recall = 62.96%` (`R@10 = 52.29`, `R@50 = 73.63`) vs paper **`63.04%`** (`Δ = -0.08%`).
    - **CIRR (`50` ep, best `Ep 43`):** `mean(R@5 + R_s@1) = 80.66%` (`R@1 = 53.38`, `R@5 = 83.04`, `R@10 = 90.48`, `R_s@1 = 78.28`) vs paper **`80.52%`** (`Δ = +0.14%`).
- **Minh bạch checkpoint không phát hành chính chủ:**
  - *FashionIQ (thiếu 1/16):* `mtst_ptha` (tác giả chỉ phát hành checkpoint CIRR).
  - *CIRR (thiếu 5/16):* `tme` (link Kaggle `cirr-0.0-best` trả HTTP 404), `airknow` & `conesep` (tác giả không đăng bản CIRR trên HuggingFace), `clvcnet` & `dcnet` (paper năm 2021 trước khi có CIRR).
  - *Ghi chú biến thể:* 4 model iLearn dùng bản noise-trained chính chủ (`habit`, `intent`, `conesep`: `N0.2`; `airknow`: `N0.5`); `limn` & `clvcnet` trên FashionIQ dùng 3-head ensemble (`dress + shirt + toptee`).

---

## 1. Năm Phát Hiện Khoa Học Cốt Lõi

### 1.1. Nghịch lý Fine-Tuning trên EDIR (`R@1` vs `R@10`) & Sức mạnh Transfer của `QuRe`
- **Trên `Recall@10`**, fine-tune **In-Domain EDIR** thắng áp đảo trên toàn bộ các model (`94.6% – 98.4%` ở nhóm CLIP/BLIP-2 so với `70.6% – 86.5%` của CIRR Transfer).
- **Nhưng trên `Recall@1`**, **9 / 11 checkpoint `CIRR Transfer` lại vượt chính bản `In-Domain EDIR`** — điển hình:
  - `INTENT`: CIRR **`40.88%`** vs In-Domain `26.52%` (`-14.36%`)
  - **`QuRe`:** CIRR **`39.58%`** (#2/11 toàn bảng!) & FIQ **`36.09%`** (#2/15 toàn bảng!) vs In-Domain `29.11%` (#4/16, `R@10 = 97.91%`, `R@50 = 99.60%`)
  - `CSMCIR`: CIRR **`39.18%`** vs In-Domain `25.82%` (`-13.36%`)
  - `HABIT`: CIRR **`38.88%`** vs In-Domain `27.32%` (`-11.57%`)
  - Chỉ **2 model** cải thiện đồng thời cả `R@1` lẫn `R@10` khi fine-tune in-domain: **`MTST-PTHA`** (`36.49% → 38.38%`, `+1.89%`) và **`SPRC`** (`30.51% → 36.19%`, `+5.68%`).
- **Cơ chế giải thích:** Mỗi cảnh EDIR gồm `6` biến thể chỉnh sửa gần nhau (`1` đích + `5` hard siblings). Tập train EDIR nhỏ (`3,510` query) khiến phần lớn model học rất nhanh việc khoanh đúng cụm cảnh (`R@10 ≈ 98%`) nhưng đánh mất năng lực phân biệt thuộc tính tinh vi bên trong 6 ảnh của cụm (trong khi `CIRR` với `~28k` cặp ảnh đa miền và cơ chế lọc false negative của `QuRe` giữ được biểu diễn ngữ nghĩa rất sắc nét: `QuRe` đứng #1 trên `cardinality` `38.6%`, đồng #1 `replace` `54.8%`, #1 `remove` `21.0%` ở CIRR Transfer và #1 `style` `42.2%` ở FIQ Transfer).

### 1.2. Điểm gãy phổ quát trên phép chỉnh sửa `remove` (EDIR)
- **Toàn bộ `16/16` In-Domain (`9.7% – 21.0%`), `11/11` CIRR Transfer (`9.7% – 21.0%`), và `15/15` FashionIQ Transfer (`6.5% – 17.7%`) đều sụp đổ trên `remove`** — bám sát sàn ngẫu nhiên `1/6 = 16.7%` dù `Recall@10` đạt `95–98%`.
- Điều này chứng minh lỗi **không thể xóa vật thể / phủ định sự hiện diện (`remove`)** là điểm mù kiến trúc & tiền huấn luyện cốt lõi của cả 3 thế hệ CIR (`Classic`, `CLIP`, `BLIP-2`), kể cả phương pháp hard-negative mining mới nhất như `QuRe` (ICML 2025).

### 1.3. PinPoint: Nhóm BLIP-2 chống nhiễu nhãn vượt chuẩn GPT-5 reranked (`0.1837`)
- Khác với EDIR, trên PinPoint (`109,598` ảnh, `4,724` query train), **In-Domain fine-tuning thắng FashionIQ Transfer trên `13 / 15` model** (`+0.0130` đến `+0.0591` `mAP@10`; chỉ `HABIT` FIQ `N0.2` `0.1737` vs `0.1670` và `QuRe` FIQ `0.1480` vs `0.1401` có bản FIQ Transfer nhỉnh nhẹ nhờ tập train FashionIQ lớn hơn ~4×).
- **Top 4 model In-Domain đều vượt baseline công bố GPT-5 reranked (`0.1837`) của paper PinPoint:**
  1. **`ConeSep`:** **`0.2128`** `mAP@10` (`+0.0501` vs FIQ Transfer `0.1626`)
  2. **`CSMCIR`:** **`0.1963`** `mAP@10` (`+0.0591` vs FIQ Transfer `0.1372`)
  3. **`INTENT`:** **`0.1929`** `mAP@10` (`+0.0183` vs FIQ Transfer `0.1745`)
  4. **`AirKnow`:** **`0.1891`** `mAP@10` (`+0.0145` vs FIQ Transfer `0.1746`)
- Đặc biệt, ngay cả ở chế độ **Transfer (chưa thấy 1 ảnh PinPoint nào)**, top 3 checkpoint `FashionIQ` tốt nhất trên PinPoint cũng là bộ ba chống nhiễu nhãn **`AirKnow` (`0.1746`), `INTENT` (`0.1745`), `HABIT` (`0.1737`)**, còn **`QuRe`** (`0.1480`) nổi bật ở lát cắt **`Fashion-Context Fit = 0.1129` (#1 / 15 toàn bảng FIQ Transfer)** nhờ không đẩy quá mạnh các ảnh hợp ngữ cảnh (false-negative-free).

### 1.4. Phân rã lát cắt `PinPoint-Fashion` (`351` queries) vs `Non-Fashion` (`986` queries)
- Dưới **FashionIQ Transfer**, nhóm model mạnh đạt `Fashion-Swap` (`197` queries) rất tốt (`0.1741 – 0.1907`, ngang `Non-Fashion`), nhưng **gãy mạnh trên `Fashion-Context Fit` (`0.06 – 0.11`, cao nhất là `QuRe` `0.1129`)** và **`Fashion-Negation` (`0.02 – 0.04`)**.
- Sau khi **fine-tune In-Domain PinPoint**, **`Fashion-Context Fit` tăng gấp ~3 lần** (vd `AirKnow`: `0.0914 → 0.2351`, `ConeSep`: `0.0715 → 0.2259`, `SPRC`: `0.0595 → 0.2130`), trong khi `Fashion-Negation` (`0.02–0.08`) và `Fashion-Complement` (`0.04–0.11`) vẫn là điểm nghẽn khó nhất.

### 1.5. Khoảng cách thế hệ kiến trúc (`BLIP-2` >> `CLIP` >> `Classic ResNet50`)
- Nhóm **Classic 2021 (`CLVC-Net`, `DCNet`)** sụp đổ hoàn toàn trên quy mô lớn của PinPoint (`0.0025 – 0.0082` `mAP@10`) và EDIR Transfer (`2.59% – 4.39%` `R@1`).
- Nhóm **CLIP (`LIMN`, `ENCODER`, `TG-CIR`, `CLIP4Cir`)** đạt mức trung bình (`0.086 – 0.149` PinPoint; `22% – 30.6%` CIRR→EDIR) nhưng có tỷ lệ `Ref Copy` & `NegRecall@10` cao hơn đáng kể (`0.14 – 0.22`) so với **BLIP-2** (`0.10 – 0.16`, riêng `QuRe` In-Domain `0.1244`).

---

## 2. Bảng Tóm Tắt Nhanh Top-5 Theo Từng Chế Độ (`16` model)

| Hạng | EDIR · In-Domain (`R@1`) | EDIR · CIRR Transfer (`R@1`) | EDIR · FIQ Transfer (`R@1`) | PinPoint · In-Domain (`mAP@10`) | PinPoint · FIQ Transfer (`mAP@10`) |
|---:|---|---|---|---|---|
| **#1** | **MTST-PTHA** (`38.38%`) | **INTENT** (`40.88%`) | **HINT** (`36.79%`) | **ConeSep** (`0.2128`) | **AirKnow** (`0.1746`) |
| **#2** | **SPRC** (`36.19%`) | **QuRe** (`39.58%`) | **QuRe** (`36.09%`) | **CSMCIR** (`0.1963`) | **INTENT** (`0.1745`) |
| **#3** | **TME** (`34.90%`) | **CSMCIR** (`39.18%`) | **INTENT** (`35.89%`) | **INTENT** (`0.1929`) | **HABIT** (`0.1737`) |
| **#4** | **QuRe** (`29.11%`) | **HABIT** (`38.88%`) | **TME** (`34.90%`) | **AirKnow** (`0.1891`) | **TME** (`0.1651`) |
| **#5** | **HINT** (`28.22%`) | **MTST-PTHA** (`36.49%`) | **HABIT** (`34.10%`) | **TME** (`0.1781`) | **ConeSep** (`0.1626`) |

---

## 3. Tài Liệu & Công Cụ Khám Phá Trực Quan
- Bảng đầy đủ EDIR In-Domain (`16/16`): [LEADERBOARD_EDIR.md](LEADERBOARD_EDIR.md)
- Bảng đầy đủ PinPoint In-Domain (`16/16`): [LEADERBOARD_PINPOINT.md](LEADERBOARD_PINPOINT.md)
- Đối chiếu 4 cột & ma trận 15 phép sửa EDIR (`16` model): [TRANSFER_AND_INDOMAIN_EDIR.md](TRANSFER_AND_INDOMAIN_EDIR.md)
- Đối chiếu 3 cột & 8 lát cắt Fashion PinPoint (`16` model): [TRANSFER_AND_INDOMAIN_PINPOINT.md](TRANSFER_AND_INDOMAIN_PINPOINT.md)
- Giao diện Web nghiên cứu trực quan (live):
  - `❖ EDIR & PinPoint` (4 tab bảng, heatmap, 8 lát cắt, inspector): `http://127.0.0.1:5173/?area=benchmarks&view=bm_overview`
  - `⊘ Lỗi chung nhanh` (mặc định FashionIQ, tách 3 dataset, split-pane & trang riêng Top-50 tất cả model): `http://127.0.0.1:5173/?area=common_failures`
