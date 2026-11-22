# Báo Cáo Sửa & Kiểm Chứng G1/G2 Parity (16 Cấu Hình FashionIQ)

> [!IMPORTANT]
> **Ngưỡng giữ nguyên (không nới lỏng):** `score_atol = 1e-4`, `score_rtol = 1e-4`, `tie_epsilon = 1e-6` (`parity-absolute-relative/v3`).
> **Hai chốt protocol FashionIQ đã áp dụng đúng phạm vi (không ảnh hưởng CIRCUS):**
> 1. **ENCODER:** `preprocess_train` cho query, `preprocess_val` cho gallery ([`model_encoder.py`](file:///home/heheboiz/data/cir/workbench/composition/model_encoder.py)).
> 2. **CSMCIR:** gộp CoT caption theo thứ tự `dress → shirt → toptee` như upstream ([`model_csmcir.py`](file:///home/heheboiz/data/cir/workbench/composition/model_csmcir.py)).

---

## 1. Bảng Tổng Hợp G1 & G2 Toàn Bộ 16 Cấu Hình (6,016 queries / cấu hình)

Tất cả báo cáo JSON chi tiết nằm tại [`workbench/artifacts/isolated-runs/g1_g2_reports/`](file:///home/heheboiz/data/cir/workbench/artifacts/isolated-runs/g1_g2_reports/) và tổng hợp G1 tại [`g1_summary_all_16.json`](file:///home/heheboiz/data/cir/workbench/artifacts/isolated-runs/g1_g2_reports/g1_summary_all_16.json).

| # | Model / Cấu hình | Protocol | G1 (2 đường chạy độc lập) | Rank đổi (tổng / non-tie) | Top-10 vào/ra | Top-50 vào/ra | ΔR@10 (pp) | ΔR@50 (pp) | `max_abs` | `max_rel` | V3 viol (/60,160) | Kết luận G2 V3 |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | **Air-Know** (`fiq_n05`) | `full_gallery_ref_excluded` | PASS (`6016/6016`) | `0` / **`0`** | `+0/-0` | `+0/-0` | `0.0000` | `0.0000` | `0.00e+00` | `0.00e+00` | `0` | **PASS (bit-exact)** |
| 2 | **ConeSep** (`fiq_n02`) | `full_gallery_ref_excluded` | PASS (`6016/6016`) | `0` / **`0`** | `+0/-0` | `+0/-0` | `0.0000` | `0.0000` | `7.15e-07` | `1.28e-06` | `0` | **PASS (exact)** |
| 3 | **CLVC-Net** | `val_split` | PASS (`6016/6016`) | `0` / **`0`** | `+0/-0` | `+0/-0` | `0.0000` | `0.0000` | `3.58e-07` | `4.51e-07` | `0` | **PASS (exact)** |
| 4 | **ENCODER** (`fashioniq`) | `val_split` | PASS (`6016/6016`) | `0` / **`0`** | `+0/-0` | `+0/-0` | `0.0000` | `0.0000` | `1.79e-07` | `2.69e-07` | `0` | **PASS (exact)** |
| 5 | **INTENT** (`fiq_n02`) | `full_gallery_ref_excluded` | PASS (`6016/6016`) | `1` / **`0`** | `+0/-0` | `+0/-0` | `0.0000` | `0.0000` | `6.56e-07` | `1.26e-06` | `0` | **PASS (exact)** |
| 6 | **HINT** (`fashioniq`) | `val_split` | PASS (`6016/6016`) | `2` / **`0`** | `+0/-0` | `+0/-0` | `0.0000` | `0.0000` | `5.36e-07` | `1.51e-06` | `0` | **PASS (exact)** |
| 7 | **LIMN** (`base_iter0_all`) | `val_split` | PASS (`6016/6016`) | `2` / **`0`** | `+0/-0` | `+0/-0` | `0.0000` | `0.0000` | `6.56e-07` | `9.50e-07` | `0` | **PASS (exact)** |
| 8 | **TG-CIR** | `val_split` | PASS (`6016/6016`) | `2` / **`0`** | `+0/-0` | `+0/-0` | `0.0000` | `0.0000` | `2.98e-06` | `1.04e-05` | `0` | **PASS (exact)** |
| 9 | **HABIT** (`fiq_n02`) | `full_gallery_ref_excluded` | PASS (`6016/6016`) | `8` / **`0`** | `+0/-0` | `+0/-0` | `0.0000` | `0.0000` | `7.15e-07` | `1.50e-06` | `0` | **PASS (exact)** |
| 10 | **CSMCIR** (`fashioniq`) | `original_split` | PASS (`6016/6016`) | `47` / **`0`** | `+1/-0` | `+0/-0` | `+0.0165` | `0.0000` | `5.96e-07` | `7.61e-07` | `0` | **PASS (tie-only)** |
| 11 | **PTHA+MTST** (`cirr_transfer`) | `original_split` | PASS (`6016/6016`) | `48` / **`0`** | `+2/-0` | `+0/-0` | `+0.0331` | `0.0000` | `3.05e-05` | `9.00e-07` | `0` | **PASS (tie-only)** |
| 12 | **SPRC** (`sprc_fiq_vitl`) | `original_split` | PASS (`6016/6016`) | `49` / **`0`** | `+2/-0` | `+0/-0` | `+0.0331` | `0.0000` | `5.07e-07` | `1.23e-06` | `0` | **PASS (tie-only)** |
| 13 | **TME** (`fashioniq_0`) | `original_split` | PASS (`6016/6016`) | `54` / **`0`** | `+2/-1` | `+0/-0` | `+0.0165` | `0.0000` | `1.25e-06` | `3.03e-06` | `0` | **PASS (tie-only)** |
| 14 | **DCNet** (`tf32=False`) | `full_gallery_ref_excluded` | PASS (`6016/6016`) | `27` / `3` | `+0/-2` | `+0/-0` | `-0.0331` | `0.0000` | `7.53e-05` | `2.40e-04` | `0` | **SCORE V3 PASS** (3 non-tie do batch_size ResNet50) |
| 15 | **CLIP4Cir (`fullft`)** | `original_split` | PASS (`6016/6016`) | `58` / `1` | `+3/-2` | `+0/-0` | `+0.0157` | `0.0000` | `7.17e-05` | `1.33e-04` | `0` | **SCORE V3 PASS** (1 non-tie do TF32 RN50x4 conv) |
| 16 | **CLIP4Cir (`noft`)** | `original_split` | PASS (`6016/6016`) | `62` / `4` | `+1/-0` | `+0/-0` | `+0.0165` | `0.0000` | `2.38e-04` | `6.22e-04` | `21` | **FAIL NHẸ** (21/60160 cặp vượt `1e-4` do TF32 RN50x4 conv re-encode theo query batch order) |

> [!NOTE]
> - **13/16 cấu hình đạt G2 V3 hoàn toàn (`v3_viol = 0` và `non_tie_rank_changes = 0`)**, trong đó 9 cấu hình có `ΔR@10 = 0.0000, ΔR@50 = 0.0000`, và 4 cấu hình thuộc `fashioniq_original_split` (`CSMCIR`, `PTHA+MTST`, `SPRC`, `TME`) chỉ có tie-breaking trong `tie_epsilon = 1e-6` (giữa `torch.argsort` GPU vs `np.argsort` CPU trên ảnh trùng lặp trong gallery `dress`).
> - Không có cấu hình nào rơi vào trường hợp "chỉ `max_rel` fail mà ranking giữ nguyên" trên các model đã qua sửa đổi (`only_max_rel_failed_ranking_unchanged = false` ở 16/16).

---

## 2. Phân Loại Lỗi Đã Tái Hiện & Sửa Đúng Chỗ (Adapter / Harness / Báo Cáo)

### A. Lỗi ở Adapter (đã sửa trực tiếp trên source local)
1. **TME** ([`model_tme.py`](file:///home/heheboiz/data/cir/workbench/composition/model_tme.py)):
   - *Tái hiện:* `KeyError: 'dataset_root'` khi `_run_job` không gửi `"dataset_root"` nhưng `_prepare_worker_cwd` đọc `job["dataset_root"]`.
   - *Sửa:* Bổ sung `"dataset_root"` vào payload và kiểm tra `if job.get("dataset_root"):`.
2. **CSMCIR** ([`model_csmcir.py`](file:///home/heheboiz/data/cir/workbench/composition/model_csmcir.py)):
   - *Tái hiện:* `121/6346` ảnh gallery `shirt` có `Final_Caption` khác giữa đọc đơn lẻ `shirt_cot_val.json` và gộp `dress → shirt → toptee` của upstream (`utils_csmcir.py:98-103`).
   - *Sửa:* Thêm thuộc tính opt-in `target_caption_merge_order` (mặc định `None` giữ nguyên CIRCUS; FashionIQ parity đặt `("dress", "shirt", "toptee")`) + `_load_merged_caption_map` + cập nhật `prepare_identity()` để tránh đụng cache giữa hai protocol. Kết quả `shirt` sau sửa: `ΔR@10 = 0.0000, ΔR@50 = 0.0000, non_tie = 0`.
3. **ENCODER** ([`model_encoder.py`](file:///home/heheboiz/data/cir/workbench/composition/model_encoder.py)):
   - *Tái hiện:* Worker chỉ tạo `preprocess_val` cho cả query và gallery thay vì `preprocess_train` cho query như upstream (`ENCODER/datasets.py:177`).
   - *Sửa:* Thêm `query_transform_mode` (mặc định `"preprocess_val"` cho CIRCUS; FashionIQ parity đặt `"preprocess_train"` + `query_transform_seed`) và cập nhật `transform_policy`. Kết quả sau sửa trên hai đường chạy độc lập khớp `RELATION_SEED=20261111` và `seed=0`: `0/6016` đổi hạng, `max_abs = 1.79e-7`.
4. **CLIP4Cir (`fullft`)** ([`model_clip4cir.py`](file:///home/heheboiz/data/cir/workbench/composition/model_clip4cir.py)):
   - *Tái hiện:* Gọi `clip_model.float().eval()` trước `clip_model.load_state_dict(saved["CLIP"])`, khác thứ tự fp16-roundtrip của upstream `validate.py:279,286,124`.
   - *Sửa:* Đưa `clip_model = clip_model.float().eval()` xuống sau `load_state_dict`.
5. **DCNet** ([`model_dcnet.py`](file:///home/heheboiz/data/cir/workbench/composition/model_dcnet.py)):
   - *Tái hiện:* `mode_supported("image_only_empty_text")` trả `(True, None)` nhưng worker tạo mảng `(0,)` gây crash khi text rỗng.
   - *Sửa:* Chuẩn hóa `.reshape(-1, 300)` và fallback `(1, 300)` zero-vector khi chuỗi token rỗng (không ảnh hưởng bất kỳ query có text nào).
6. **Air-Know** ([`model_airknow.py`](file:///home/heheboiz/data/cir/workbench/composition/model_airknow.py)):
   - *Tái hiện:* `hasattr(Adapter(), "seed") == False` trước khi gọi `prepare_identity()`, và thiếu hook `_capture_spec` cho G1.
   - *Sửa:* Khởi tạo `self.seed = 0` và `self._capture_spec = None` trong `__init__`, bổ sung `_capture_spec` trong `_worker_main`. Kết quả trên hai đường chạy độc lập: `max_abs = 0.00e+00`, `0/6016` đổi hạng.

### B. Lỗi ở Harness / Protocol Prep (đã xác định & sửa đúng trong harness)
1. **`correction_dict_<cat>.json` (298/6,016 queries: 102 dress, 103 shirt, 93 toptee):** Thuộc tầng dataset/harness của 7 model (`HINT`, `HABIT`, `INTENT`, `ConeSep`, `Air-Know`, `ENCODER`, `LIMN`); khi harness truyền đúng text đã qua `correction_dict_<cat>.json`, cả 7/7 model đạt `0` non-tie rank change và `ΔRecall = 0.0000`.
2. **CLVC-Net `<BOS> ... <AND> ... <EOS>`:** Thuộc tầng dataset/harness (`CLVCNet/datasets.py:173-175`); khi truyền đúng format này, `clvc_net` đạt `0/6016` đổi hạng, `max_abs = 3.58e-7`.
3. **DCNet TF32 (`4.44e-2` → `7.53e-5`):** Native `run_dcnet.py` để mặc định `cudnn.allow_tf32=True` trên RTX 5070 Ti trong khi `model_dcnet.py` khóa `allow_tf32=False`. Khi hai đường chạy độc lập cùng khóa `allow_tf32=False`, `v3_viol` giảm từ `51,759` xuống `0/60,160`.
4. **CLIP4Cir PEP-610 environment:** Env `clip4cir310-sm120` bị gate chặn do metadata `file://` của `uv`; chuyển sang env chuẩn PEP-610 sẵn có `clip4cir310-sm120-pep610` thì qua gate hợp lệ.

---

## 3. Kiểm Tra Hồi Quy (Unit Tests)
- Lệnh: `workbench/envs/ctl312h/bin/python -m pytest -n 10 -q -p no:cacheprovider workbench/tests/test_parity_absolute_relative_v2.py workbench/tests/test_csmcir_composition_adapter.py workbench/tests/test_airknow_composition_adapter.py workbench/tests/test_clip4cir_composition_adapter.py workbench/tests/test_r21_tme_lifecycle.py workbench/tests/test_encoder_relation_seed.py workbench/tests/test_dcnet_checkpoint_load_portability.py`
- Kết quả: **134 passed, 0 failed** (5.03s). Đã dọn sạch `/tmp/pytest-of-heheboiz`, `/tmp/g1g2_*`, `/tmp/ctrl_*`.
