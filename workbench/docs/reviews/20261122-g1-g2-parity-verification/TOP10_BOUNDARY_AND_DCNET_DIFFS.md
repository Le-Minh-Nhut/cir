# Đối Chiếu Từng Query Lệch Ranh Giới Top-10 (6 Cấu Hình `original_split`) & DCNet `comp_scores`/`corr_scores`

## Phần 1: 6 Cấu Hình Lệch Hit@10 (14/14 Query Đều Là Exact Tie `0.00e+00` Do Ảnh PNG Trùng MD5)

| Model | Cat | Query ID | Hướng | Rank Nat→Adp | `|s10 - s11|` Nat | `|s10 - s11|` Adp | Trùng MD5 tại 10/11 | Native #10 (score, md5) | Native #11 (score, md5) | Adapter #10 (score, md5) | Adapter #11 (score, md5) |
|---|---|---|---|---|---:|---:|---|---|---|---|---|
| `clip4cir_rn50x4_fullft` | `dress` | `dress:311:B009TM8IG8:B00BEJ34Q8` | `LEFT_R10` | `6→15` | `0.00e+00` | `0.00e+00` | `True` | `B00BQX3Z0M` (0.569089, `ca7a44f4a5dd`) | `B00BQX461E` (0.569089, `ca7a44f4a5dd`) | `B00ALSQOYW` (0.569089, `ca7a44f4a5dd`) | `B00CIQD2MW` (0.569089, `ca7a44f4a5dd`) |
| `clip4cir_rn50x4_fullft` | `dress` | `dress:1980:B00BQX3Z0M:B00CIQD2MW` | `ENTERED_R10` | `13→7` | `0.00e+00` | `0.00e+00` | `True` | `B00AYXA5D0` (0.622646, `ca7a44f4a5dd`) | `B00BDGE5DI` (0.622646, `ca7a44f4a5dd`) | `B00BDGE5DI` (0.622646, `ca7a44f4a5dd`) | `B00ANOMAUG` (0.622646, `ca7a44f4a5dd`) |
| `clip4cir_rn50x4_fullft` | `shirt` | `shirt:424:B0094KCP96:B009PQRI82` | `ENTERED_R10` | `11→10` | `0.00e+00` | `0.00e+00` | `True` | `B00AFZ38RM` (0.479366, `81bc1b9122d3`) | `B009PQRI82` ★ (0.479366, `81bc1b9122d3`) | `B009PQRI82` ★ (0.479366, `81bc1b9122d3`) | `B00AFZ38RM` (0.479366, `81bc1b9122d3`) |
| `clip4cir_rn50x4_fullft` | `shirt` | `shirt:1736:B007HRO9B0:B005DIJKXW` | `ENTERED_R10` | `11→10` | `0.00e+00` | `0.00e+00` | `True` | `B005DIJJBK` (0.506905, `1ed4b3d70ad5`) | `B005DIJKXW` ★ (0.506905, `1ed4b3d70ad5`) | `B005DIJKXW` ★ (0.506905, `1ed4b3d70ad5`) | `B005DIJJBK` (0.506905, `1ed4b3d70ad5`) |
| `clip4cir_rn50x4_fullft` | `toptee` | `toptee:592:B00APE63FM:B00EK6N7XM` | `LEFT_R10` | `10→11` | `0.00e+00` | `0.00e+00` | `True` | `B00EK6N7XM` ★ (0.498795, `b3f974a840e1`) | `B00AFVH8P4` (0.498795, `b3f974a840e1`) | `B00AFVH8P4` (0.498795, `b3f974a840e1`) | `B00EK6N7XM` ★ (0.498795, `b3f974a840e1`) |
| `combiner_rn50x4_noft` | `dress` | `dress:1980:B00BQX3Z0M:B00CIQD2MW` | `ENTERED_R10` | `13→7` | `0.00e+00` | `0.00e+00` | `True` | `B00AYXA5D0` (0.475097, `ca7a44f4a5dd`) | `B00BDGE5DI` (0.475097, `ca7a44f4a5dd`) | `B00CIQE758` (0.475097, `ca7a44f4a5dd`) | `B00BDGE5DI` (0.475097, `ca7a44f4a5dd`) |
| `csmcir` | `dress` | `dress:1980:B00BQX3Z0M:B00CIQD2MW` | `ENTERED_R10` | `13→3` | `0.00e+00` | `0.00e+00` | `True` | `B00AYXA5D0` (0.915630, `ca7a44f4a5dd`) | `B00BDGE5DI` (0.915630, `ca7a44f4a5dd`) | `B00BDGE5DI` (0.915630, `ca7a44f4a5dd`) | `B00ALSQOYW` (0.915630, `ca7a44f4a5dd`) |
| `sprc` | `dress` | `dress:1362:B00A16KWZ6:B00AYXA5D0` | `ENTERED_R10` | `11→7` | `0.00e+00` | `0.00e+00` | `True` | `B00ALSQOYW` (0.665480, `ca7a44f4a5dd`) | `B00AYXA5D0` ★ (0.665480, `ca7a44f4a5dd`) | `B00BDGE5DI` (0.665480, `ca7a44f4a5dd`) | `B00BQX461E` (0.665480, `ca7a44f4a5dd`) |
| `sprc` | `dress` | `dress:1980:B00BQX3Z0M:B00CIQD2MW` | `ENTERED_R10` | `13→6` | `0.00e+00` | `0.00e+00` | `True` | `B00AYXA5D0` (0.589123, `ca7a44f4a5dd`) | `B00BDGE5DI` (0.589123, `ca7a44f4a5dd`) | `B00BDGE5DI` (0.589123, `ca7a44f4a5dd`) | `B00AYXA5D0` (0.589123, `ca7a44f4a5dd`) |
| `ptha_mtst` | `dress` | `dress:1362:B00A16KWZ6:B00AYXA5D0` | `ENTERED_R10` | `13→7` | `0.00e+00` | `0.00e+00` | `True` | `B00BQX461E` (34.940876, `ca7a44f4a5dd`) | `B00AKR44XM` (34.940876, `ca7a44f4a5dd`) | `B00ALSQOYW` (34.940876, `ca7a44f4a5dd`) | `B00CIQE758` (34.940876, `ca7a44f4a5dd`) |
| `ptha_mtst` | `dress` | `dress:1980:B00BQX3Z0M:B00CIQD2MW` | `ENTERED_R10` | `13→3` | `0.00e+00` | `0.00e+00` | `True` | `B00AYXA5D0` (34.105255, `ca7a44f4a5dd`) | `B00BDGE5DI` (34.105255, `ca7a44f4a5dd`) | `B00A16DXWK` (34.105255, `ca7a44f4a5dd`) | `B00BDGE5DI` (34.105255, `ca7a44f4a5dd`) |
| `tme` | `dress` | `dress:331:B00A16KWZ6:B00CAKLNY0` | `LEFT_R10` | `6→14` | `0.00e+00` | `0.00e+00` | `True` | `B00BQX461E` (0.561485, `ca7a44f4a5dd`) | `B00AKR44XM` (0.561485, `ca7a44f4a5dd`) | `B00CIQE758` (0.561485, `ca7a44f4a5dd`) | `B00A16DXWK` (0.561485, `ca7a44f4a5dd`) |
| `tme` | `dress` | `dress:611:B00CFXUI4S:B00B2JHXOE` | `ENTERED_R10` | `11→10` | `0.00e+00` | `0.00e+00` | `True` | `B00BZTBYQO` (0.490015, `fc775d791ef1`) | `B00B2JHXOE` ★ (0.490015, `fc775d791ef1`) | `B00B2JHXOE` ★ (0.490015, `fc775d791ef1`) | `B00BZTBYQO` (0.490015, `fc775d791ef1`) |
| `tme` | `dress` | `dress:1980:B00BQX3Z0M:B00CIQD2MW` | `ENTERED_R10` | `13→5` | `0.00e+00` | `0.00e+00` | `True` | `B00AYXA5D0` (0.514943, `ca7a44f4a5dd`) | `B00BDGE5DI` (0.514943, `ca7a44f4a5dd`) | `B00AKR44XM` (0.514943, `ca7a44f4a5dd`) | `B00BEJ34Q8` (0.514943, `ca7a44f4a5dd`) |

---

## Phần 2: Chẩn Đoán DCNet (`trg batch_size=100`, `query batch_size=8`, `allow_tf32=False`)

### Định vị bước đầu tiên bị lệch (Stage-by-stage max_abs)

| Bước | `max_abs` (Native vs Adapter) | Kết luận |
|---|---:|---|
| `get_image_emb` (gallery batch 0, 100 ảnh) | `0.00e+00` | **Bit-exact `0.0`** |
| `get_image_emb` (query src batch 0, 8 ảnh) | `0.00e+00` | **Bit-exact `0.0`** |
| `get_text_emb` (query text batch 0, 8 câu) | `0.00e+00` | **Bit-exact `0.0`** |
| `get_composition` (query composed batch 0) | `0.00e+00` | **Bit-exact `0.0`** |
| `comp_scores` (`dress` / `shirt` / `toptee`) | `1.53e-05` / `0.00e+00` / `1.14e-05` | **Bước đầu tiên xuất hiện lệch `~1.5e-5`** (do `model_dcnet.py:554` chunk gallery 512 cột) |
| `corr_scores` (`dress` / `shirt` / `toptee`) | `2.29e-05` / `3.05e-05` / `2.29e-05` | **Lệch `~2.3–3.1e-5`** (cùng bước chunk gallery 512 cột trong `get_correction` + `get_score_matrix`) |
| `fused_scores` (`log_softmax` tổng) | `3.43e-05` / `3.81e-05` / `3.43e-05` | Dưới ngưỡng `1e-4` (`0` vi phạm V3, `ΔR@10 = 0.0000`, `ΔR@50 = 0.0000`) |

### Bảng đối chiếu 7 query non-tie còn lại (khi dùng đúng `trg=100, q=8`; tất cả đều ngoài Top-50)

| Cat | Query idx | Ref ID | Target ID | Rank Nat→Adp | `target_comp` (Nat / Adp / `|Δ|`) | `target_corr` (Nat / Adp / `|Δ|`) | `row_comp_max_abs` | `row_corr_max_abs` | `target_fused_diff` |
|---|---:|---|---|---|---|---|---:|---:|---:|
| `dress` | `230` | `B0036DE0R2` | `B00A16DXWK` | `24→27` | `65.13988` / `65.13988` / `0.00e+00` | `66.43436` / `66.43436` / `0.00e+00` | `0.00e+00` | `1.53e-05` | `9.54e-07` |
| `dress` | `355` | `B007ZDYK2E` | `B00BZTBYQO` | `51→50` | `65.59364` / `65.59364` / `0.00e+00` | `66.71666` / `66.71667` / `7.63e-06` | `0.00e+00` | `1.53e-05` | `1.14e-05` |
| `dress` | `1737` | `B006WM4LYG` | `B00BG3NQE2` | `161→162` | `61.90994` / `61.90994` / `0.00e+00` | `62.15228` / `62.15228` / `0.00e+00` | `0.00e+00` | `1.53e-05` | `5.72e-06` |
| `dress` | `1987` | `B00BG3HJKY` | `B00CG5QJF2` | `53→52` | `59.68966` / `59.68966` / `0.00e+00` | `62.75541` / `62.75542` / `7.63e-06` | `0.00e+00` | `1.53e-05` | `9.54e-06` |
| `shirt` | `1335` | `B001O883U6` | `B001O84MJC` | `2→3` | `75.60271` / `75.60271` / `0.00e+00` | `72.46407` / `72.46405` / `1.53e-05` | `0.00e+00` | `1.53e-05` | `9.54e-06` |
| `shirt` | `1736` | `B007HRO9B0` | `B005DIJKXW` | `159→158` | `59.62511` / `59.62511` / `0.00e+00` | `64.31442` / `64.31442` / `0.00e+00` | `0.00e+00` | `1.53e-05` | `3.81e-06` |
| `toptee` | `717` | `B00AL1QEXU` | `B00C7AFSIA` | `1→2` | `82.86243` / `82.86243` / `0.00e+00` | `84.18752` / `84.18752` / `0.00e+00` | `0.00e+00` | `1.53e-05` | `1.43e-06` |
