# MODEL RUN MATRIX — CIRCUS cohort

Phạm vi báo cáo này: **9 model có `composition_adapter`** (đường CIRCUS 3-condition). 8 model còn lại không có adapter composition và **do agent khác phụ trách** theo đường native — chúng vẫn có hàng trong `MODEL_INVENTORY.md` nhưng không có hàng ở đây.

Nhãn execution:

- **NEW_RUN** — chạy trong phiên này, artifact mới, feature cache có thể dùng lại.
- **RUNNING** — đang chạy, xem `RUNS.vi.md` §3 để resume.
- **PRE_EXISTING_UNVERIFIED** — artifact từ commit `6421188`, **chưa** được tái lập trong phiên này.
- **NOT_RUN** — chưa chạy, có lý do.
- **BLOCKED_ADAPTER** — adapter tự từ chối, không phải lỗi môi trường.

## Ma trận

| model | checkpoint | protocol | condition | execution | coverage | MM R@10 | MM R@50 | MM nDCG | CompGap | run_id | artifact |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `tme` | `fashioniq_0`<br>0% | `circus_fashioniq_release_v1` | `multimodal` | **PRE_EXISTING_UNVERIFIED** | 1.00000 | 34.0163 | 53.9564 | 0.32039 | 0.35860 | `ad477312aa57c1b9114f9e51` | `summary/tme_fashioniq_0_ad477312aa57c1b9114f9e51` |
| `tme` | `fashioniq_0`<br>0% | `circus_fashioniq_release_v1` | `text_only_black_image` | **PRE_EXISTING_UNVERIFIED** | — | — | — | — | — | `ad477312aa57c1b9114f9e51` | `summary/tme_fashioniq_0_ad477312aa57c1b9114f9e51` |
| `tme` | `fashioniq_0`<br>0% | `circus_fashioniq_release_v1` | `image_only_empty_text` | **PRE_EXISTING_UNVERIFIED** | — | — | — | — | — | `ad477312aa57c1b9114f9e51` | `summary/tme_fashioniq_0_ad477312aa57c1b9114f9e51` |
| `encoder` | `fashioniq`<br>unstated · eval 0% | `circus_fashioniq_release_v1` | `multimodal` | **PRE_EXISTING_UNVERIFIED** | 1.00000 | 31.0178 | 49.5752 | 0.29586 | 0.47230 | `6a528e877c328772e2c7bb28` | `summary/encoder_fashioniq_6a528e877c328772e2c7bb28` |
| `encoder` | `fashioniq`<br>unstated · eval 0% | `circus_fashioniq_release_v1` | `text_only_black_image` | **PRE_EXISTING_UNVERIFIED** | — | — | — | — | — | `6a528e877c328772e2c7bb28` | `summary/encoder_fashioniq_6a528e877c328772e2c7bb28` |
| `encoder` | `fashioniq`<br>unstated · eval 0% | `circus_fashioniq_release_v1` | `image_only_empty_text` | **PRE_EXISTING_UNVERIFIED** | — | — | — | — | — | `6a528e877c328772e2c7bb28` | `summary/encoder_fashioniq_6a528e877c328772e2c7bb28` |
| `ptha_mtst` | `cirr_transfer`<br>CIRR→FashionIQ TRANSFER | `circus_fashioniq_release_v1` | `multimodal` | **PRE_EXISTING_UNVERIFIED** | 1.00000 | 19.4736 | 34.4494 | 0.22493 | 0.24362 | `315dd590eeb03d2c78d7937a` | `summary/ptha_mtst_cirr_transfer_315dd590eeb03d2c78d7937a` |
| `ptha_mtst` | `cirr_transfer`<br>CIRR→FashionIQ TRANSFER | `circus_fashioniq_release_v1` | `text_only_black_image` | **PRE_EXISTING_UNVERIFIED** | — | — | — | — | — | `315dd590eeb03d2c78d7937a` | `summary/ptha_mtst_cirr_transfer_315dd590eeb03d2c78d7937a` |
| `ptha_mtst` | `cirr_transfer`<br>CIRR→FashionIQ TRANSFER | `circus_fashioniq_release_v1` | `image_only_empty_text` | **PRE_EXISTING_UNVERIFIED** | — | — | — | — | — | `315dd590eeb03d2c78d7937a` | `summary/ptha_mtst_cirr_transfer_315dd590eeb03d2c78d7937a` |
| `limn` | `base_iter0_all_categories`<br>unstated · eval 0% | `circus_fashioniq_release_v1` | `multimodal` | **PRE_EXISTING_UNVERIFIED** | 0.98018 | 30.0646 | 48.7593 | 0.29084 | 0.54174 | `10173dc45dea8c90fd2a284b` | `summary/limn_base_iter0_all_categories_10173dc45dea8c90fd2a284b` |
| `limn` | `base_iter0_all_categories`<br>unstated · eval 0% | `circus_fashioniq_release_v1` | `text_only_black_image` | **PRE_EXISTING_UNVERIFIED** | — | — | — | — | — | `10173dc45dea8c90fd2a284b` | `summary/limn_base_iter0_all_categories_10173dc45dea8c90fd2a284b` |
| `limn` | `base_iter0_all_categories`<br>unstated · eval 0% | `circus_fashioniq_release_v1` | `image_only_empty_text` | **PRE_EXISTING_UNVERIFIED** | — | — | — | — | — | `10173dc45dea8c90fd2a284b` | `summary/limn_base_iter0_all_categories_10173dc45dea8c90fd2a284b` |
| `habit` | `fiq_n02`<br>train 20% · eval 0% | `circus_fashioniq_release_v1` | `multimodal` | **RUNNING** — RUNNING — encoding gallery; run_id `f76171fa32296f3369970a5c` | 0.00000 | — | — | — | — | `e0dd04a46ef7a58334b9e353` | `summary/habit_fiq_n02_e0dd04a46ef7a58334b9e353` |
| `habit` | `fiq_n02`<br>train 20% · eval 0% | `circus_fashioniq_release_v1` | `text_only_black_image` | **RUNNING** — RUNNING — encoding gallery; run_id `f76171fa32296f3369970a5c` | — | — | — | — | — | `e0dd04a46ef7a58334b9e353` | `summary/habit_fiq_n02_e0dd04a46ef7a58334b9e353` |
| `habit` | `fiq_n02`<br>train 20% · eval 0% | `circus_fashioniq_release_v1` | `image_only_empty_text` | **RUNNING** — RUNNING — encoding gallery; run_id `f76171fa32296f3369970a5c` | — | — | — | — | — | `e0dd04a46ef7a58334b9e353` | `summary/habit_fiq_n02_e0dd04a46ef7a58334b9e353` |
| `intent` | `fiq_n02`<br>train 20% · eval 0% | `circus_fashioniq_release_v1` | `multimodal` | **NOT_RUN** — NOT_RUN — xếp sau habit | — | — | — | — | — | — | — |
| `intent` | `fiq_n02`<br>train 20% · eval 0% | `circus_fashioniq_release_v1` | `text_only_black_image` | **NOT_RUN** — NOT_RUN — xếp sau habit | — | — | — | — | — | — | — |
| `intent` | `fiq_n02`<br>train 20% · eval 0% | `circus_fashioniq_release_v1` | `image_only_empty_text` | **NOT_RUN** — NOT_RUN — xếp sau habit | — | — | — | — | — | — | — |
| `hint` | `fashioniq`<br>unstated · eval 0% | `circus_fashioniq_release_v1` | `multimodal` | **NOT_RUN** — NOT_RUN — xếp sau | — | — | — | — | — | — | — |
| `hint` | `fashioniq`<br>unstated · eval 0% | `circus_fashioniq_release_v1` | `text_only_black_image` | **NOT_RUN** — NOT_RUN — xếp sau | — | — | — | — | — | — | — |
| `hint` | `fashioniq`<br>unstated · eval 0% | `circus_fashioniq_release_v1` | `image_only_empty_text` | **NOT_RUN** — NOT_RUN — xếp sau | — | — | — | — | — | — | — |
| `conesep` | `fiq_n02`<br>train 20% · eval 0% | `circus_fashioniq_release_v1` | `multimodal` | **NOT_RUN** — NOT_RUN — xếp sau | — | — | — | — | — | — | — |
| `conesep` | `fiq_n02`<br>train 20% · eval 0% | `circus_fashioniq_release_v1` | `text_only_black_image` | **NOT_RUN** — NOT_RUN — xếp sau | — | — | — | — | — | — | — |
| `conesep` | `fiq_n02`<br>train 20% · eval 0% | `circus_fashioniq_release_v1` | `image_only_empty_text` | **NOT_RUN** — NOT_RUN — xếp sau | — | — | — | — | — | — | — |
| `csmcir` | `fashioniq`<br>0% | `circus_fashioniq_release_v1` | `multimodal` | **BLOCKED_ADAPTER** — BLOCKED_ADAPTER — xem BLOCKERS §B1 | 0.00000 | — | — | — | — | — | — |
| `csmcir` | `fashioniq`<br>0% | `circus_fashioniq_release_v1` | `text_only_black_image` | **BLOCKED_ADAPTER** — BLOCKED_ADAPTER — xem BLOCKERS §B1 | 0.00000 | — | — | — | — | — | — |
| `csmcir` | `fashioniq`<br>0% | `circus_fashioniq_release_v1` | `image_only_empty_text` | **BLOCKED_ADAPTER** — BLOCKED_ADAPTER — xem BLOCKERS §B1 | 0.00000 | — | — | — | — | — | — |
## Năng lực (capabilities) theo model

| model | MM | text-only | image-only | category-scoped | reference_eligible (khai báo) | reference_eligible (đo) | gate F1 |
|---|---|---|---|---|---|---|---|
| `tme` | YES | YES | YES | no | true | **true — đo được, rank 1** | YES (giữ nguyên) |
| `encoder` | YES | YES | YES | no | true | true (rank 1) | **NO→YES (đã sửa)** |
| `ptha_mtst` | YES | YES | YES | no | true | **true — đo được, rank 1** | **NO→YES (đã sửa)** |
| `limn` | YES | YES | YES | **yes** (dress/shirt/toptee) | true | `measured:false` — scorer cần Mapping gallery thật | **NO→YES (đã sửa)** |
| `habit` | YES | YES | YES | no | true | **true — đo được, rank 1** | YES (giữ nguyên) |
| `intent` | YES | YES | YES | no | true | **true — đo được, rank 1** | YES (giữ nguyên) |
| `hint` | YES | YES | YES | no | true | **true — đo được, rank 1** | YES (giữ nguyên) |
| `conesep` | YES | YES | YES | no | true | đo sau khi khai `score_dimension=8192` | YES (giữ nguyên) |
| `csmcir` | NO (adapter từ chối) | NO | NO | no | true | **true — đo được, rank 1** | **NO→YES (đã sửa)** |

### Ghi chú về từng cột

- **MM / text-only / image-only**: kết quả `adapter.mode_supported()` đọc trực tiếp, không suy đoán. `csmcir` từ chối cả ba (lý do: BLOCKERS §B1).
- **category-scoped**: chỉ `limn` cần gán category theo reference image; 119/6003 query không gán được (BLOCKERS §B4).
- **reference_eligible (khai báo)** trước đây là literal `True` ở cả 9 adapter (defect F7).
- **reference_eligible (đo)**: `CompositionAdapter.measured_reference_eligibility()` chạy `score`+`rank` thật của adapter trên gallery tổng hợp nơi reference là exact match.
- **gate F1**: trước fix chỉ 5/9 adapter gọi gate môi trường. Nay 9/9.

## Tổng hợp

- Model có `composition_adapter`: **9**.
- Chạy được về môi trường (env + checkpoint staged + adapter nhận): **8** (trừ `csmcir`).
- Có artifact full-run cũ (6003 query): **5** — trong đó 4 COMPLETED, 1 PARTIAL, 1 FAILED.
- NEW_RUN trong phiên này: **1 đang chạy** (`habit`), 0 hoàn tất (hết budget phiên).
- Chưa từng chạy: **intent, hint, conesep** (môi trường mới có, xếp sau `habit`).

> **Không** có hàng nào ở đây được gọi là NEW_RUN khi chưa có artifact do phiên này sinh.