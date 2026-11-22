# Báo Cáo Sửa & Kiểm Chứng G1/G2 Parity (16 Cấu Hình FashionIQ — Căn Theo Verdict `parity_gate`)

> [!IMPORTANT]
> **Quy tắc nhãn:** Trong báo cáo này, **chỉ** cấu hình đã chạy đủ 10 role V3 + `score-ranks.jsonl` + qua `parity_gate.evaluate_model(..., standard="parity-absolute-relative/v3") == PARITY_PASS` mới được gắn nhãn **`PARITY_PASS`** và publish vào `workbench/artifacts/parity/v3/CURRENT.json`. Mọi cấu hình chưa chạy đủ 10-role gate V3 được ghi đúng nhãn **`DIAGNOSTIC_ONLY (NOT_PUBLISHED)`**.
>
> **Ngưỡng giữ nguyên:** `score_atol = 1e-4`, `score_rtol = 1e-4`, `tie_epsilon = 1e-6` (`parity-absolute-relative/v3`).
> **Hai chốt protocol FashionIQ (opt-in, không ảnh hưởng CIRCUS):**
> 1. **ENCODER:** `preprocess_train` cho query, `preprocess_val` cho gallery ([`model_encoder.py`](file:///home/heheboiz/data/cir/workbench/composition/model_encoder.py)).
> 2. **CSMCIR:** gộp CoT caption theo `dress → shirt → toptee` như upstream ([`model_csmcir.py`](file:///home/heheboiz/data/cir/workbench/composition/model_csmcir.py)).
> 3. **CLIP4Cir (`fullft` & `noft`):** tái dùng feature ảnh tham chiếu chưa chuẩn hóa từ gallery do chính adapter mã hóa trong run đó ([`model_clip4cir.py`](file:///home/heheboiz/data/cir/workbench/composition/model_clip4cir.py)).
> 4. **PTHA+MTST:** giữ nguyên nhãn **CIRR-to-FashionIQ transfer (`cirr_transfer`)**.

---

## 1. Bảng Tổng Hợp 16 Cấu Hình (Theo Đúng Verdict `parity_gate`)

- **Bảng đối chiếu từng query bị lệch (14/14 query ranh giới Top-10 + DCNet `comp_scores`/`corr_scores`):** [`TOP10_BOUNDARY_AND_DCNET_DIFFS.md`](file:///home/heheboiz/data/cir/workbench/docs/reviews/20261122-g1-g2-parity-verification/TOP10_BOUNDARY_AND_DCNET_DIFFS.md)
- **Bằng chứng G1 pixel tensor + text/token thực tế (worker adapter thật vs native độc lập, 9 model đầu):** [`g1_real_worker_9models.json`](file:///home/heheboiz/data/cir/workbench/artifacts/isolated-runs/g1_g2_reports/g1_real_worker_9models.json)
- **Phạm vi đo chẩn đoán G2:** `target_rank`, `Top-10`, `Top-50`, `R@10`, `R@50` tính từ **ranking trên toàn bộ gallery ở cả hai bên** (`3,817/6,346/5,373` hoặc `2,454/2,901/2,612`). Sai lệch giá trị score (`max_abs`/`max_rel`/`v3_viol`) trong bảng chẩn đoán đo trên **Top-10 ứng viên native mỗi query (`60,160` cặp)**. Riêng **LIMN** đã chạy **đủ 10 role V3 + `score-ranks.jsonl` toàn gallery** qua `parity_gate`.

| # | Model / Cấu hình | Protocol | G1 thực tế (worker vs native) | Rank đổi (tổng / non-tie) | Top-10 vào/ra | Top-50 vào/ra | ΔR@10 (pp) | ΔR@50 (pp) | `max_abs` | `max_rel` | V3 viol (/60,160) | Verdict `parity_gate` V3 |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | **LIMN** (`base_iter0_all`) | `val_split` | **PASS** (`q_px=0.0, g_px=0.0, text=8/8, tok=True`) | `2` / **`0`** | `+0/-0` | `+0/-0` | `0.0000` | `0.0000` | `6.56e-07` | `9.50e-07` | `0` | **`PARITY_MANDATORY_SUBCHECK_FAILED`** (1 failure: `shirt.metrics.hit_count_equality` ở `r1`: `516` vs `517`; đã gỡ publish) |
| 2 | **Air-Know** (`fiq_n05`) | `full_gallery_ref_excluded` | **PASS** (`q_px=0.0, g_px=0.0, text=8/8`) | `0` / **`0`** | `+0/-0` | `+0/-0` | `0.0000` | `0.0000` | `0.00e+00` | `0.00e+00` | `0` | `DIAGNOSTIC_ONLY (NOT_PUBLISHED)` — G2 bit-exact, chưa chạy 10-role V3 |
| 3 | **ConeSep** (`fiq_n02`) | `full_gallery_ref_excluded` | **PASS** (`q_px=0.0, g_px=0.0, text=8/8`) | `0` / **`0`** | `+0/-0` | `+0/-0` | `0.0000` | `0.0000` | `7.15e-07` | `1.28e-06` | `0` | `DIAGNOSTIC_ONLY (NOT_PUBLISHED)` — G2 exact, chưa chạy 10-role V3 |
| 4 | **CLVC-Net** | `val_split` | **PASS** (`q_px=0.0, g_px=0.0, text=8/8`) | `0` / **`0`** | `+0/-0` | `+0/-0` | `0.0000` | `0.0000` | `3.58e-07` | `4.51e-07` | `0` | `DIAGNOSTIC_ONLY (NOT_PUBLISHED)` — G2 exact, chưa chạy 10-role V3 |
| 5 | **ENCODER** (`fashioniq`) | `val_split` | **PASS** (`q_px=0.0, g_px=0.0, text=8/8, tok=True`) | `0` / **`0`** | `+0/-0` | `+0/-0` | `0.0000` | `0.0000` | `1.79e-07` | `2.69e-07` | `0` | `DIAGNOSTIC_ONLY (NOT_PUBLISHED)` — G2 exact, chưa chạy 10-role V3 |
| 6 | **INTENT** (`fiq_n02`) | `full_gallery_ref_excluded` | **PASS** (`q_px=0.0, g_px=0.0, text=8/8`) | `1` / **`0`** | `+0/-0` | `+0/-0` | `0.0000` | `0.0000` | `6.56e-07` | `1.26e-06` | `0` | `DIAGNOSTIC_ONLY (NOT_PUBLISHED)` — G2 exact, chưa chạy 10-role V3 |
| 7 | **HINT** (`fashioniq`) | `val_split` | **PASS** (`q_px=0.0, g_px=0.0, text=8/8`) | `2` / **`0`** | `+0/-0` | `+0/-0` | `0.0000` | `0.0000` | `5.36e-07` | `1.51e-06` | `0` | `DIAGNOSTIC_ONLY (NOT_PUBLISHED)` — G2 exact, chưa chạy 10-role V3 |
| 8 | **TG-CIR** | `val_split` | **PASS** (`q_px=0.0, g_px=0.0, text=8/8, tok=True`) | `2` / **`0`** | `+0/-0` | `+0/-0` | `0.0000` | `0.0000` | `2.98e-06` | `1.04e-05` | `0` | `DIAGNOSTIC_ONLY (NOT_PUBLISHED)` — G2 exact, chưa chạy 10-role V3 |
| 9 | **HABIT** (`fiq_n02`) | `full_gallery_ref_excluded` | **PASS** (`q_px=0.0, g_px=0.0, text=8/8`) | `8` / **`0`** | `+0/-0` | `+0/-0` | `0.0000` | `0.0000` | `7.15e-07` | `1.50e-06` | `0` | `DIAGNOSTIC_ONLY (NOT_PUBLISHED)` — G2 exact, chưa chạy 10-role V3 |
| 10 | **CSMCIR** (`fashioniq`) | `original_split` | Structural PASS (chưa thu tensor) | `47` / **`0`** | `+1/-0` | `+0/-0` | `+0.0165` | `0.0000` | `5.96e-07` | `7.61e-07` | `0` | `DIAGNOSTIC_ONLY (NOT_PUBLISHED)` — 1/1 Top-10 swap là exact tie `0.0` (trùng MD5) |
| 11 | **PTHA+MTST** (`cirr_transfer`, **transfer**) | `original_split` | Structural PASS (chưa thu tensor) | `48` / **`0`** | `+2/-0` | `+0/-0` | `+0.0331` | `0.0000` | `3.05e-05` | `9.00e-07` | `0` | `TRANSFER_DIAGNOSTIC_ONLY (NOT_PUBLISHED)` — 2/2 Top-10 swap là exact tie `0.0` (trùng MD5) |
| 12 | **SPRC** (`sprc_fiq_vitl`) | `original_split` | Structural PASS (chưa thu tensor) | `49` / **`0`** | `+2/-0` | `+0/-0` | `+0.0331` | `0.0000` | `5.07e-07` | `1.23e-06` | `0` | `DIAGNOSTIC_ONLY (NOT_PUBLISHED)` — 2/2 Top-10 swap là exact tie `0.0` (trùng MD5) |
| 13 | **TME** (`fashioniq_0`) | `original_split` | Structural PASS (chưa thu tensor) | `54` / **`0`** | `+2/-1` | `+0/-0` | `+0.0165` | `0.0000` | `1.25e-06` | `3.03e-06` | `0` | `DIAGNOSTIC_ONLY (NOT_PUBLISHED)` — 3/3 Top-10 swap là exact tie `0.0` (trùng MD5) |
| 14 | **CLIP4Cir (`fullft`)** | `original_split` | Structural PASS (chưa thu tensor) | `57` / **`0`** | `+3/-2` | `+0/-0` | `+0.0157` | `0.0000` | `9.54e-07` | `1.83e-06` | `0` | `DIAGNOSTIC_ONLY (NOT_PUBLISHED)` — 5/5 Top-10 swap là exact tie `0.0` (trùng MD5) |
| 15 | **CLIP4Cir (`noft`)** | `original_split` | Structural PASS (chưa thu tensor) | `57` / **`0`** | `+1/-0` | `+0/-0` | `+0.0165` | `0.0000` | `7.15e-07` | `1.90e-06` | `0` | `DIAGNOSTIC_ONLY (NOT_PUBLISHED)` — 1/1 Top-10 swap là exact tie `0.0` (trùng MD5) |
| 16 | **DCNet** (`tf32=False, trg=100, q=8`) | `full_gallery_ref_excluded` | Structural PASS + 4/4 embedding batch 0 bit-exact `0.0` | `13` / `7` | `+0/-0` | `+0/-0` | `0.0000` | `0.0000` | `3.81e-05` | `2.40e-04` | `0` | `DIAGNOSTIC_ONLY (NOT_PUBLISHED)` — 0 vi phạm V3, `ΔR@10=0, ΔR@50=0`; 7 non-tie ngoài Top-50 do chunk 512 cột trong `model_dcnet.py:554` |

---

## 2. Kết Luận Kỹ Thuật Chính Từ Đợt Kiểm Chứng Này

1. **9/9 cấu hình đầu đạt G1 bit-exact `0.0e+00`** khi thu trực tiếp từ worker adapter thật so với tiến trình native độc lập (`q_px = 0.0`, `g_px = 0.0`, `text = 8/8`, `token_ids` khớp tuyệt đối).
2. **14/14 query vào/ra Top-10 trên 6 cấu hình `original_split`** (`CLIP4Cir FullFT/NoFT`, `CSMCIR`, `SPRC`, `PTHA+MTST`, `TME`) đều có **`|s10 - s11| = 0.00e+00`** ở cả hai bên và **`dup_md5 = True`** (gallery `images/*.png` có 50 nhóm ảnh trùng byte ở `dress`, 33 ở `shirt`, 15 ở `toptee`). Không có bất kỳ lệch ranking thực sự nào.
3. **DCNet (`trg=100, q=8, allow_tf32=False`):**
   - Tất cả 4 bước embedding đầu tiên (`gallery_emb`, `query_src_emb`, `query_txt_emb`, `query_cmp_emb`) đều **bit-exact `0.0`**.
   - Bước đầu tiên xuất hiện sai số là `comp_scores` (`1.53e-5` dress, `0.0` shirt, `1.14e-5` toptee) và `corr_scores` (`2.29e-5`–`3.05e-5`) tại [`model_dcnet.py:554`](file:///home/heheboiz/data/cir/workbench/composition/model_dcnet.py#L554) (chunk gallery 512 cột), với `ΔR@10 = 0.0000`, `ΔR@50 = 0.0000`, `0` vi phạm V3 và chỉ còn 7 query non-tie đều nằm ngoài Top-50.
4. **Gate V3 thật (`parity_gate.py`) trên LIMN & giới hạn dung lượng đĩa (`/` còn 21 GB trống):**
   - `LIMN` vượt toàn bộ G1 + 10 role V3 + `r10`/`r50` nhưng fail 1 check duy nhất `shirt.metrics.hit_count_equality` ở **`r1` (`516` vs `517`, 1 tie-only swap hạng 1↔2)** → verdict **`PARITY_MANDATORY_SUBCHECK_FAILED`** (đã gỡ khỏi `CURRENT.json`).
   - Mỗi lần chạy full 10-role V3 cho 1 model tiêu tốn ~27 phút GPU và ~4 GB đĩa (~1.1 GB cho generation publish). Với 21 GB trống hiện tại trên `/`, cần tiếp tục chạy tuần tự có dọn scratch sau từng model (ưu tiên 4 model `0` rank change: `CLVC-Net`, `ENCODER`, `ConeSep`, `Air-Know` vì chắc chắn không có tie-flip ở `r1`).
