# Đối Chiếu Từng Query Lệch Ranh Giới Top-10 (6 Cấu Hình `original_split`) & DCNet `comp_scores`/`corr_scores`

## Phần 1: 6 Cấu Hình Lệch Hit@10 — Nguyên Nhân Gốc & Phép Thử Đối Chứng 4 Bộ Sắp Xếp (Sorter Ablation)

### 1.1. Kết luận nguyên nhân gốc (đã kiểm chứng bằng phép thử đơn yếu tố)
- **Tất cả 14/14 query vào/ra Top-10** ở 6 cấu hình `original_split` đều có **`|s10 - s11| = 0.00e+00`** ở cả hai bên và **`dup_md5 = True`** (ảnh PNG trùng byte trong FashionIQ gallery: 50 nhóm ở `dress`, 33 ở `shirt`, 15 ở `toptee`).
- **Bước đầu tiên gây lệch:** Không phải lỗi score (`max_abs ≤ 3.05e-5`, `0/60.160` vi phạm V3) và không phải lỗi đổi sang distance (`dist_merges = 0` trên 14/14 query này), mà là **lỗi chọn bộ sắp xếp không ổn định (`np.argsort(..., kind="quicksort")`)** trong harness kiểm chứng [`verify_g1_g2_adapter.py`](file:///home/heheboiz/data/cir/workbench/artifacts/isolated-runs/_launchers/verify_g1_g2_adapter.py).
- **Đối chứng với upstream:** Cả 6 repo gốc (`CLIP4Cir/src/validate.py:45`, `CSMCIR/src/validate_blip_csmcir.py:45`, `SPRC/src/validate_blip.py:45`, `TME/src/precompute_evaluation.py:94`, `MTST_PTHA/src/validate_blip.py:45`) đều dùng **`torch.argsort(1 - sim, dim=-1)` trên CUDA** (CUB radix sort — **ổn định**, `cuda_det = True`, giữ thứ tự index gallery nhỏ hơn trước khi hòa điểm tuyệt đối, trùng khớp hoàn toàn với `np.argsort(..., kind="stable")` và [`workbench/composition/topk.py`](file:///home/heheboiz/data/cir/workbench/composition/topk.py#L76-L102)).

### 1.2. Bảng phép thử đối chứng đơn yếu tố (cùng ma trận score & thứ tự gallery, chỉ đổi đúng bộ sắp xếp)
(Artifact JSON: [`sorter_ablation_6models.json`](file:///home/heheboiz/data/cir/workbench/artifacts/isolated-runs/g1_g2_reports/sorter_ablation_6models.json))

| Model : Cat | `dist_merges` (top-20) | `cuda_det` | `np_qs` (`1-s`, quicksort — trước sửa) | `np_stable` (`-s`, stable) | `torch_cuda_1m` (`1-s`, CUDA — upstream) | `torch_cuda_neg` (`-s`, CUDA) |
|---|---:|---|---|---|---|---|
| `csmcir:dress` | `0` | `True` | `rchg=22, R10=+1/-0` | **`rchg=0, R10=+0/-0`** | **`rchg=0, R10=+0/-0`** | **`rchg=0, R10=+0/-0`** |
| `sprc:dress` | `0` | `True` | `rchg=28, R10=+2/-0` | **`rchg=0, R10=+0/-0`** | **`rchg=0, R10=+0/-0`** | **`rchg=0, R10=+0/-0`** |
| `ptha_mtst:dress` | `0` | `True` | `rchg=26, R10=+2/-0` | **`rchg=0, R10=+0/-0`** | **`rchg=0, R10=+0/-0`** | **`rchg=0, R10=+0/-0`** |
| `tme:dress` | `0` | `True` | `rchg=30, R10=+2/-1` | **`rchg=1, R10=+0/-0`** | **`rchg=1, R10=+0/-0`** | **`rchg=1, R10=+0/-0`** |
| `combiner_rn50x4_noft:dress` | `1` | `True` | `rchg=27, R10=+1/-0` | **`rchg=0, R10=+0/-0`** | **`rchg=0, R10=+0/-0`** | **`rchg=0, R10=+0/-0`** |
| `clip4cir_rn50x4_fullft:dress` | `0` | `True` | `rchg=31, R10=+1/-1` | **`rchg=2, R10=+0/-0`** | **`rchg=2, R10=+0/-0`** | **`rchg=2, R10=+0/-0`** |
| `clip4cir_rn50x4_fullft:shirt` | `1` | `True` | `rchg=14, R10=+2/-0` | **`rchg=3, R10=+0/-0`** | **`rchg=2, R10=+0/-0`** | **`rchg=3, R10=+0/-0`** |
| `clip4cir_rn50x4_fullft:toptee` | `0` | `True` | `rchg=12, R10=+0/-1` | **`rchg=2, R10=+0/-0`** | **`rchg=2, R10=+0/-0`** | **`rchg=2, R10=+0/-0`** |

- **Tách riêng trường hợp `1 - s` gộp score (`dist_merges`):** Chỉ xảy ra 1 query ở `clip4cir_rn50x4_fullft:shirt` (`np_stable rchg=3` vs `cuda_1m rchg=2`) và 1 query ở `combiner_rn50x4_noft:dress`, đều không nằm ở ranh giới Top-10/50.
- **Sau khi sửa harness sang `kind="stable"` và chạy lại đủ 3 category (6.016 query/cấu hình):** Cả **6/6 cấu hình đều đạt `Top-10 +0/-0`, `Top-50 +0/-0`, `ΔR@10 = 0.0000`, `ΔR@50 = 0.0000`, `0` non-tie**.

### 1.3. Bảng 14 query ranh giới Top-10 trước sửa (đều đã về `+0/-0` sau khi đổi sang `kind="stable"`)

| Model | Cat | Query ID | Hướng (trước sửa) | Rank Nat→Adp (trước sửa) | `|s10 - s11|` Nat | `|s10 - s11|` Adp | Trùng MD5 tại 10/11 | Native #10 (score, md5) | Native #11 (score, md5) |
|---|---|---|---|---|---:|---:|---|---|---|
| `clip4cir_rn50x4_fullft` | `dress` | `dress:311:B009TM8IG8:B00BEJ34Q8` | `LEFT_R10` | `6→15` | `0.00e+00` | `0.00e+00` | `True` | `B00BQX3Z0M` (0.569089, `ca7a44f4a5dd`) | `B00BQX461E` (0.569089, `ca7a44f4a5dd`) |
| `clip4cir_rn50x4_fullft` | `dress` | `dress:1980:B00BQX3Z0M:B00CIQD2MW` | `ENTERED_R10` | `13→7` | `0.00e+00` | `0.00e+00` | `True` | `B00AYXA5D0` (0.622646, `ca7a44f4a5dd`) | `B00BDGE5DI` (0.622646, `ca7a44f4a5dd`) |
| `clip4cir_rn50x4_fullft` | `shirt` | `shirt:424:B0094KCP96:B009PQRI82` | `ENTERED_R10` | `11→10` | `0.00e+00` | `0.00e+00` | `True` | `B00AFZ38RM` (0.479366, `81bc1b9122d3`) | `B009PQRI82` ★ (0.479366, `81bc1b9122d3`) |
| `clip4cir_rn50x4_fullft` | `shirt` | `shirt:1736:B007HRO9B0:B005DIJKXW` | `ENTERED_R10` | `11→10` | `0.00e+00` | `0.00e+00` | `True` | `B005DIJJBK` (0.506905, `1ed4b3d70ad5`) | `B005DIJKXW` ★ (0.506905, `1ed4b3d70ad5`) |
| `clip4cir_rn50x4_fullft` | `toptee` | `toptee:592:B00APE63FM:B00EK6N7XM` | `LEFT_R10` | `10→11` | `0.00e+00` | `0.00e+00` | `True` | `B00EK6N7XM` ★ (0.498795, `b3f974a840e1`) | `B00AFVH8P4` (0.498795, `b3f974a840e1`) |
| `combiner_rn50x4_noft` | `dress` | `dress:1980:B00BQX3Z0M:B00CIQD2MW` | `ENTERED_R10` | `13→7` | `0.00e+00` | `0.00e+00` | `True` | `B00AYXA5D0` (0.475097, `ca7a44f4a5dd`) | `B00BDGE5DI` (0.475097, `ca7a44f4a5dd`) |
| `csmcir` | `dress` | `dress:1980:B00BQX3Z0M:B00CIQD2MW` | `ENTERED_R10` | `13→3` | `0.00e+00` | `0.00e+00` | `True` | `B00AYXA5D0` (0.915630, `ca7a44f4a5dd`) | `B00BDGE5DI` (0.915630, `ca7a44f4a5dd`) |
| `sprc` | `dress` | `dress:1362:B00A16KWZ6:B00AYXA5D0` | `ENTERED_R10` | `11→7` | `0.00e+00` | `0.00e+00` | `True` | `B00ALSQOYW` (0.665480, `ca7a44f4a5dd`) | `B00AYXA5D0` ★ (0.665480, `ca7a44f4a5dd`) |
| `sprc` | `dress` | `dress:1980:B00BQX3Z0M:B00CIQD2MW` | `ENTERED_R10` | `13→6` | `0.00e+00` | `0.00e+00` | `True` | `B00AYXA5D0` (0.589123, `ca7a44f4a5dd`) | `B00BDGE5DI` (0.589123, `ca7a44f4a5dd`) |
| `ptha_mtst` | `dress` | `dress:1362:B00A16KWZ6:B00AYXA5D0` | `ENTERED_R10` | `13→7` | `0.00e+00` | `0.00e+00` | `True` | `B00BQX461E` (34.940876, `ca7a44f4a5dd`) | `B00AKR44XM` (34.940876, `ca7a44f4a5dd`) |
| `ptha_mtst` | `dress` | `dress:1980:B00BQX3Z0M:B00CIQD2MW` | `ENTERED_R10` | `13→3` | `0.00e+00` | `0.00e+00` | `True` | `B00AYXA5D0` (34.105255, `ca7a44f4a5dd`) | `B00BDGE5DI` (34.105255, `ca7a44f4a5dd`) |
| `tme` | `dress` | `dress:331:B00A16KWZ6:B00CAKLNY0` | `LEFT_R10` | `6→14` | `0.00e+00` | `0.00e+00` | `True` | `B00BQX461E` (0.561485, `ca7a44f4a5dd`) | `B00AKR44XM` (0.561485, `ca7a44f4a5dd`) |
| `tme` | `dress` | `dress:611:B00CFXUI4S:B00B2JHXOE` | `ENTERED_R10` | `11→10` | `0.00e+00` | `0.00e+00` | `True` | `B00BZTBYQO` (0.490015, `fc775d791ef1`) | `B00B2JHXOE` ★ (0.490015, `fc775d791ef1`) |
| `tme` | `dress` | `dress:1980:B00BQX3Z0M:B00CIQD2MW` | `ENTERED_R10` | `13→5` | `0.00e+00` | `0.00e+00` | `True` | `B00AYXA5D0` (0.514943, `ca7a44f4a5dd`) | `B00BDGE5DI` (0.514943, `ca7a44f4a5dd`) |

---

## Phần 2: Chẩn Đoán & Sửa Triệt Để DCNet (`comp_scores` / `corr_scores` → Bit-Exact `0.00e+00`)

### 2.1. Định vị bước đầu tiên bị lệch & Phép thử đơn yếu tố
- Cả 4 bước embedding đầu tiên (`gallery_emb`, `query_src_emb`, `query_txt_emb`, `query_cmp_emb`) khi chạy `trg=100, q=8, allow_tf32=False` đều **bit-exact `0.00e+00`**.
- **Bước đầu tiên xuất hiện lệch:** `model_dcnet.py:554` chia gallery thành các chunk **512 cột** (`for target_start in range(0, len(t), 512)`), trong khi upstream `DCNet/trainer/trainer_joint.py:188-214` truyền **toàn bộ gallery `val_experts` (`N x D`)** vào `get_score_matrix` và `get_correction`, khiến cuBLAS chọn kernel GEMM tiling khác nhau (`comp_max = 1.53e-5`, `corr_max = 3.05e-5`).

| Phép thử (đều cố định `trg=100, q=8, allow_tf32=False`) | `comp_max_abs` | `corr_max_abs` | `fused_max_abs` | `rank_changed` (/6.016) | `non_tie` (/6.016) | Top-10 / Top-50 vào-ra |
|---|---:|---:|---:|---:|---:|---|
| Trước sửa: `t_chunk = 512` (`model_dcnet.py` cũ) | `1.53e-05` | `3.05e-05` | `3.81e-05` | `13` | `8` | `+0/-0` / `+0/-0` |
| **Sau sửa (đổi đúng 1 yếu tố): `score_target_chunk_size = None` (full gallery, sản xuất qua `model_dcnet.Adapter`)** | **`0.00e+00`** | **`0.00e+00`** | **`0.00e+00`** | **`0`** | **`0`** | **`+0/-0` / `+0/-0`** |

- **Bản sửa đã triển khai tại [`workbench/composition/model_dcnet.py`](file:///home/heheboiz/data/cir/workbench/composition/model_dcnet.py):**
  - Thêm thuộc tính opt-in `score_target_chunk_size: int | None = 512` (mặc định `512` giữ nguyên an toàn VRAM cho CIRCUS 74k gallery và không đổi schema job mặc định; khi đặt `None` cho FashionIQ parity sẽ chấm toàn bộ gallery khớp `trainer_joint.py:188-214`).
  - Đưa `_mask_reference_scores` và `F.log_softmax` vào trong `with torch.inference_mode():` để tránh `RuntimeError: Inplace update to inference tensor outside InferenceMode`.
  - Bổ sung test hồi quy tại [`workbench/tests/test_dcnet_replay.py`](file:///home/heheboiz/data/cir/workbench/tests/test_dcnet_replay.py) (`26 passed in 1.15s`).
  - Artifact kiểm chứng toàn bộ 3 category: [`dcnet_fixed_full_gallery_parity.json`](file:///home/heheboiz/data/cir/workbench/artifacts/isolated-runs/g1_g2_reports/dcnet_fixed_full_gallery_parity.json).
