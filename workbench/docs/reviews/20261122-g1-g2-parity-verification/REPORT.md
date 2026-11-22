# Báo Cáo Sửa & Kiểm Chứng G1/G2 Parity (16 Cấu Hình FashionIQ — Căn Theo Verdict `parity_gate`)

> [!IMPORTANT]
> **Quy tắc nhãn:** Trong báo cáo này, **chỉ** cấu hình đã chạy đủ 10 role V3 + `score-ranks.jsonl` + qua `parity_gate.evaluate_model(..., standard="parity-absolute-relative/v3") == PARITY_PASS` mới được gắn nhãn **`PARITY_PASS`** và publish vào `workbench/artifacts/parity/v3/CURRENT.json`. Mọi cấu hình chưa qua hoặc bị chặn ở gate V3 được ghi rõ verdict thật và blocker cụ thể.
>
> **Ngưỡng giữ nguyên:** `score_atol = 1e-4`, `score_rtol = 1e-4`, `tie_epsilon = 1e-6` (`parity-absolute-relative/v3`).
> **Chốt protocol FashionIQ (opt-in, không ảnh hưởng CIRCUS):**
> 1. **ENCODER:** `preprocess_train` cho query, `preprocess_val` cho gallery ([`model_encoder.py`](file:///home/heheboiz/data/cir/workbench/composition/model_encoder.py)).
> 2. **CSMCIR:** gộp CoT caption theo `dress → shirt → toptee` như upstream ([`model_csmcir.py`](file:///home/heheboiz/data/cir/workbench/composition/model_csmcir.py)).
> 3. **CLIP4Cir (`fullft` & `noft`):** tái dùng feature ảnh tham chiếu chưa chuẩn hóa từ gallery do chính adapter mã hóa trong run đó ([`model_clip4cir.py`](file:///home/heheboiz/data/cir/workbench/composition/model_clip4cir.py)).
> 4. **DCNet:** opt-in `score_target_chunk_size = None` (chấm full gallery như `trainer_joint.py:188-214`, giữ mặc định `512` cho CIRCUS) ([`model_dcnet.py`](file:///home/heheboiz/data/cir/workbench/composition/model_dcnet.py)).
> 5. **PTHA+MTST:** giữ nguyên nhãn **CIRR-to-FashionIQ transfer (`cirr_transfer`)**.

---

## 1. Bảng Tổng Hợp 16 Cấu Hình (Sau Sửa Triệt Để — Căn Theo Verdict Thật `parity_gate` V3)

- **Phép thử đối chứng 4 sorter (6 cấu hình `original_split`) & DCNet `comp_scores`/`corr_scores`:** [`TOP10_BOUNDARY_AND_DCNET_DIFFS.md`](file:///home/heheboiz/data/cir/workbench/docs/reviews/20261122-g1-g2-parity-verification/TOP10_BOUNDARY_AND_DCNET_DIFFS.md)
- **Bằng chứng G1 pixel tensor + text/token thực tế (worker adapter thật vs native độc lập, 9 model đầu):** [`g1_real_worker_9models.json`](file:///home/heheboiz/data/cir/workbench/docs/reviews/20261122-g1-g2-parity-verification/g1_real_worker_9models.json)
- **Phạm vi đo chẩn đoán G2:** `target_rank`, `Top-10`, `Top-50`, `R@10`, `R@50` tính từ **ranking trên toàn bộ gallery ở cả hai bên**. Sau đợt sửa này, **16/16 cấu hình đều đạt `0` non-tie rank change, `+0/-0` Top-10, `+0/-0` Top-50, `ΔR@10 = 0.0000`, `ΔR@50 = 0.0000`, `0/60.160` vi phạm V3**.

| # | Model / Cấu hình | Protocol | G1 thực tế (worker vs native) | Rank đổi (tổng / non-tie) | Top-10 vào/ra | Top-50 vào/ra | ΔR@10 (pp) | ΔR@50 (pp) | `max_abs` | `max_rel` | V3 viol | Verdict thật `parity_gate` V3 & Blocker |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | **CLVC-Net** | `val_split` | **PASS** (`0.0`, 8/8) | `0` / **`0`** *(V3 full: 1 tie ở `dress` #2↔#3, `0` ở `shirt`/`toptee`)* | `+0/-0` | `+0/-0` | `0.0000` | `0.0000` | `1.13e-06` *(toàn gallery)* | `4.51e-07` | `0` | **`PARITY_EVIDENCE_MALFORMED` (NOT_PUBLISHED)** — đã chạy đủ 10 role × 3 cat (`metrics_equal` khớp r1/r10/r50, `max_abs ≤ 1.13e-6`), nhưng bị chặn bởi schema `prepare_clvc_tme_v3_plans.py`: `v3 identity.checkpoint must be a SHA256 digest of verified checkpoint bytes` |
| 2 | **DCNet** (`tf32=False, trg=100, q=8, full-gallery`) | `full_gallery_ref_excluded` | Structural PASS + 4/4 embedding batch 0 bit-exact `0.0` | **`0`** / **`0`** | `+0/-0` | `+0/-0` | `0.0000` | `0.0000` | **`0.00e+00`** *(comp/corr/fused)* | **`0.00e+00`** | `0` | **`PARITY_NOT_RUN` (DIAGNOSTIC_ONLY)** — G2 **bit-exact `0.0`** trên toàn bộ 6.016 query sau sửa `score_target_chunk_size=None`; chưa có V3 publisher `CURRENT.json` |
| 3 | **Air-Know** (`fiq_n05`) | `full_gallery_ref_excluded` | **PASS** (`0.0`, 8/8) | **`0`** / **`0`** | `+0/-0` | `+0/-0` | `0.0000` | `0.0000` | `0.00e+00` | `0.00e+00` | `0` | **`PARITY_NOT_RUN` (DIAGNOSTIC_ONLY)** — G1+G2 bit-exact `0.0`; chưa có V3 publisher `CURRENT.json` |
| 4 | **ConeSep** (`fiq_n02`) | `full_gallery_ref_excluded` | **PASS** (`0.0`, 8/8) | **`0`** / **`0`** | `+0/-0` | `+0/-0` | `0.0000` | `0.0000` | `7.15e-07` | `1.28e-06` | `0` | **`PARITY_EVIDENCE_MISSING` (DIAGNOSTIC_ONLY)** — G1+G2 exact; chưa có V3 publisher `CURRENT.json` |
| 5 | **ENCODER** (`fashioniq`) | `val_split` | **PASS** (`0.0`, 8/8, tok=True) | **`0`** / **`0`** | `+0/-0` | `+0/-0` | `0.0000` | `0.0000` | `1.79e-07` | `2.69e-07` | `0` | **`PARITY_MANDATORY_SUBCHECK_FAILED` (artifact cũ trước sửa)** — bản sửa `preprocess_train` đạt G1+G2 exact `0` rank change; cần chạy lại V3 publisher |
| 6 | **LIMN** (`base_iter0_all`) | `val_split` | **PASS** (`0.0`, 8/8, tok=True) | `2` / **`0`** | `+0/-0` | `+0/-0` | `0.0000` | `0.0000` | `6.56e-07` | `9.50e-07` | `0` | **`PARITY_MANDATORY_SUBCHECK_FAILED` (đã gỡ publish)** — vượt G1 + 10 role V3 + `r10`/`r50`, fail 1 check duy nhất `shirt.metrics.hit_count_equality` ở `r1` (`516` vs `517`, 1 tie-only swap #1↔#2) |
| 7 | **INTENT** (`fiq_n02`) | `full_gallery_ref_excluded` | **PASS** (`0.0`, 8/8) | `1` / **`0`** | `+0/-0` | `+0/-0` | `0.0000` | `0.0000` | `6.56e-07` | `1.26e-06` | `0` | **`PARITY_EVIDENCE_MISSING` (DIAGNOSTIC_ONLY)** — G1+G2 exact; chưa có V3 publisher `CURRENT.json` |
| 8 | **HINT** (`fashioniq`) | `val_split` | **PASS** (`0.0`, 8/8) | `2` / **`0`** | `+0/-0` | `+0/-0` | `0.0000` | `0.0000` | `5.36e-07` | `1.51e-06` | `0` | **`PARITY_MANDATORY_SUBCHECK_FAILED` (artifact cũ)** — G1+G2 exact trên source hiện tại; cần V3 publisher |
| 9 | **TG-CIR** | `val_split` | **PASS** (`0.0`, 8/8, tok=True) | `2` / **`0`** | `+0/-0` | `+0/-0` | `0.0000` | `0.0000` | `2.98e-06` | `1.04e-05` | `0` | **`PARITY_MANDATORY_SUBCHECK_FAILED` (artifact cũ)** — G1+G2 exact trên source hiện tại; cần V3 publisher |
| 10 | **HABIT** (`fiq_n02`) | `full_gallery_ref_excluded` | **PASS** (`0.0`, 8/8) | `8` / **`0`** | `+0/-0` | `+0/-0` | `0.0000` | `0.0000` | `7.15e-07` | `1.50e-06` | `0` | **`PARITY_MANDATORY_SUBCHECK_FAILED` (artifact cũ)** — G1+G2 exact trên source hiện tại; cần V3 publisher |
| 11 | **PTHA+MTST** (`cirr_transfer`, **transfer**) | `original_split` | Structural PASS *(chưa thu tensor)* | **`0`** / **`0`** *(trước sửa: 48)* | **`+0/-0`** | `+0/-0` | **`0.0000`** | `0.0000` | `3.05e-05` | `9.00e-07` | `0` | **`PARITY_NOT_RUN` (`TRANSFER_DIAGNOSTIC_ONLY`)** — sau sửa sorter `stable`: `0/6.016` rank đổi, `ΔR@10=0`; giữ nhãn transfer |
| 12 | **CSMCIR** (`fashioniq`) | `original_split` | Structural PASS *(chưa thu tensor)* | **`2`** / **`0`** *(trước sửa: 47)* | **`+0/-0`** | `+0/-0` | **`0.0000`** | `0.0000` | `5.96e-07` | `7.61e-07` | `0` | **`PARITY_NOT_RUN` (DIAGNOSTIC_ONLY)** — sau sửa sorter `stable`: `+0/-0` Top-10, `ΔR@10=0`; chưa có V3 publisher `CURRENT.json` |
| 13 | **SPRC** (`sprc_fiq_vitl`) | `original_split` | Structural PASS *(chưa thu tensor)* | **`2`** / **`0`** *(trước sửa: 49)* | **`+0/-0`** | `+0/-0` | **`0.0000`** | `0.0000` | `5.07e-07` | `1.23e-06` | `0` | **`PARITY_NOT_RUN` (DIAGNOSTIC_ONLY)** — sau sửa sorter `stable`: `+0/-0` Top-10, `ΔR@10=0`; chưa có V3 publisher `CURRENT.json` |
| 14 | **TME** (`fashioniq_0`) | `original_split` | Structural PASS *(chưa thu tensor)* | **`2`** / **`0`** *(trước sửa: 54)* | **`+0/-0`** | `+0/-0` | **`0.0000`** | `0.0000` | `1.25e-06` | `3.03e-06` | `0` | **`PARITY_NOT_RUN` (DIAGNOSTIC_ONLY)** — sau sửa sorter `stable`: `+0/-0` Top-10, `ΔR@10=0`; bị chung blocker schema `identity.checkpoint` ở `prepare_clvc_tme_v3_plans.py` |
| 15 | **CLIP4Cir (`fullft`)** | `original_split` | Structural PASS *(chưa thu tensor)* | **`6`** / **`0`** *(trước sửa: 57)* | **`+0/-0`** | `+0/-0` | **`0.0000`** | `0.0000` | `9.54e-07` | `1.83e-06` | `0` | **`PARITY_NOT_RUN` (DIAGNOSTIC_ONLY)** — sau sửa sorter `stable`: `+0/-0` Top-10, `ΔR@10=0`; chưa có V3 publisher `CURRENT.json` |
| 16 | **CLIP4Cir (`noft`)** | `original_split` | Structural PASS *(chưa thu tensor)* | **`6`** / **`0`** *(trước sửa: 57)* | **`+0/-0`** | `+0/-0` | **`0.0000`** | `0.0000` | `7.15e-07` | `1.90e-06` | `0` | **`PARITY_NOT_RUN` (DIAGNOSTIC_ONLY)** — sau sửa sorter `stable`: `+0/-0` Top-10, `ΔR@10=0`; chưa có V3 publisher `CURRENT.json` |

---

## 2. Phân Loại Rõ: Đã Sửa Triệt Để vs Hành Vi Vốn Có vs Blocker V3 Publish

1. **Đã sửa triệt để tại nơi sinh lỗi (đều đã có test hồi quy và kiểm chứng lại đủ 3 category):**
   - **DCNet (`comp_scores`/`corr_scores` lệch `1.5e-5`–`3.1e-5`, 13 rank đổi / 8 non-tie → `0.00e+00`, `0` rank đổi):** sửa [`model_dcnet.py`](file:///home/heheboiz/data/cir/workbench/composition/model_dcnet.py) thêm `score_target_chunk_size: int | None = 512` (full gallery cho FashionIQ parity) + đưa reference masking vào `torch.inference_mode()`.
   - **6 cấu hình `original_split` (14/14 hit@10 vào/ra → `+0/-0`, `ΔR@10 = 0.0000`):** sửa [`verify_g1_g2_adapter.py`](file:///home/heheboiz/data/cir/workbench/artifacts/isolated-runs/_launchers/verify_g1_g2_adapter.py) từ `np.argsort(..., kind="quicksort")` sang `kind="stable"` (khớp CUDA `torch.argsort(1 - sim, dim=-1)` trên upstream và `workbench/composition/topk.py`).
   - **CLVC-Net replay observer (3 lỗi harness phát hiện khi chạy 10 role V3 thật):** sửa [`workbench/replay/clvc_net.py`](file:///home/heheboiz/data/cir/workbench/replay/clvc_net.py) (thiếu CLI `--batch-size`, hook `extract_img_feature` bị gọi lồng trong `compose_img_text`, và lưu `-sims` chưa đảo dấu) + [`workbench/replay/clvc_tme_parity.py`](file:///home/heheboiz/data/cir/workbench/replay/clvc_tme_parity.py) (thiếu `plan_sha256` ở `AdapterArtifactSink`).
2. **Hành vi vốn có đã chứng minh bằng đối chứng:**
   - **Nhóm ảnh trùng MD5 trong FashionIQ `images/*.png`:** các tie tuyệt đối `0.00e+00` chỉ bị xáo trộn nếu dùng unstable `quicksort`; dùng `stable` sort (hoặc CUDA `torch.argsort`) tái hiện 100% tie-breaking upstream trên toàn bộ 14/14 query ranh giới Top-10.
   - **2–6 tie-only rank swaps còn lại ngoài Top-10/50** (hoặc `LIMN` `shirt` #1↔#2 làm lệch `r1` `516` vs `517`): phát sinh từ chênh lệch FP32 `~1e-7` trên các cặp điểm hòa (`< tie_epsilon = 1e-6`).
3. **Blocker hạ tầng V3 publish (cần gỡ trước khi đưa vào `CURRENT.json`):**
   - **`clvc_net` / `tme`:** `prepare_clvc_tme_v3_plans.py` xuất `identity.checkpoint` không phải chuỗi hex SHA256 duy nhất nên `parity_gate` trả `PARITY_EVIDENCE_MALFORMED` (dù toàn bộ 30 role của `clvc_net` đạt `max_abs ≤ 1.13e-6` và `metrics_equal` khớp tuyệt đối ở r1/r10/r50).
   - **13 cấu hình còn lại:** chỉ mới có `limn_v3_parity.py` và `clvc_tme_v3_parity.py` được nối với `publish_v3_generation` (`CURRENT.json`); 13 cấu hình kia chưa có V3 publisher nối `CURRENT.json` và 7 cấu hình nhóm sau chưa thu G1 tensor 2-tiến-trình độc lập.
