# MODEL RUN MATRIX — CIRCUS cohort

> **Audit liên quan (đọc trước khi trích bất kỳ số nào ở đây):**
> [`workbench/docs/reviews/20261108T173736+0700/INPUT_PIPELINE_AND_INFERENCE_AUDIT.vi.md`](../20261108T173736+0700/INPUT_PIPELINE_AND_INFERENCE_AUDIT.vi.md)
> — kiểm toán độc lập pipeline đọc → inference → scoring. **Nó xác lập** danh tính repository/checkpoint
> (G0 PASS, G1 PASS 17/17), decode bit-identical nơi đã đo (G2 PASS), và rằng fast path bị từ chối vì
> đổi **207/18,009** rank (G3b FAIL). **Nó KHÔNG xác lập**: forward-feature bit-identity (**G2-fwd
> NOT_RUN**, GPU bận suốt cửa sổ), batch/tail/cold-warm axes (**G4 PARTIAL**), cache identity
> (**G5 DEFECT, đã đóng một phần** — 7 input vắng, nay còn 5), hay throughput (**G6 PARTIAL**, một phần
> chỉ có trong prose). Không con số nào ở đây là paper reproduction.

Phạm vi báo cáo này: **9 model có `composition_adapter`** (đường CIRCUS 3-condition). 8 model còn lại không có adapter composition và **do agent khác phụ trách** theo đường native — chúng vẫn có hàng trong `MODEL_INVENTORY.md` nhưng không có hàng ở đây.

Nhãn execution:

- **COMPLETED** — artifact đầy đủ, đọc trực tiếp từ `summary/<run>/full.json`.
- **PARTIAL** — chạy xong nhưng coverage < 1.0 (có query không chấm được).
- **FAILED** — chạy và thất bại; xem log.
- **BLOCKED_ADAPTER** — adapter tự từ chối, không phải lỗi môi trường.
- **SUPERSEDED_NOT_RERUN** — số cũ còn hợp lệ lịch sử nhưng **không phải** kết quả sẽ có dưới code hiện
  tại; run id đã đổi.

Cột **`scoring_impl`** lấy từ chính backend (`scoring_impl` của `/api/composition/runs`), tức đọc thân
`score()` của module adapter trong registry — **không** suy từ hash. Cột **BLAS policy** ghi policy hiệu
dụng của run đó; mọi run ở đây sinh **trước** khi pin ra đời (`bf6a448`) nên 5 model GEMM-path ghi
`unknown (predates the pin)`, còn 4 model naive-einsum ghi `BLAS-neutral` vì contraction của chúng không
có đường BLAS (đo phẳng ở 1/2/4/8/10 thread, delta đúng 0.0).

**Cảnh báo khoa học:** bảng này xếp 8 model cạnh nhau, nhưng chúng **không** chấm bằng cùng một cách.
`einsum` naive, `einsum(..., optimize=True)` và `matmul` là ba thứ tự cộng dồn khác nhau, nên rank của
chúng **không so sánh được ở mức từng query** (audit §9). Chênh lệch aggregate dưới đây là chênh lệch
metric, không phải chứng minh model này "tốt hơn" model kia.

## Ma trận

| model | checkpoint | protocol | condition | execution | coverage | MM R@10 | MM R@50 | MM nDCG | CompGap | `scoring_impl` | BLAS policy | run_id | artifact |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `habit` | `fiq_n02`<br>train 20% · eval 0% | `circus_fashioniq_release_v1` | `multimodal` | **COMPLETED** | 1.00000 | 35.61552557054806 | 54.3728135932034 | 0.32828280509551133 | 0.388704727214254 | `einsum` | BLAS-neutral (Δ 0.0) | `497c25c57fbeba885ecfb9d1` | `summary/habit_fiq_n02_497c25c57fbeba885ecfb9d1` |
| `habit` | `fiq_n02`<br>train 20% · eval 0% | `circus_fashioniq_release_v1` | `text_only_black_image` | **COMPLETED** | 1.00000 | 15.292353823088456 | 27.053140096618357 | 0.2006777268917305 | — | `einsum` | BLAS-neutral (Δ 0.0) | `497c25c57fbeba885ecfb9d1` | `summary/habit_fiq_n02_497c25c57fbeba885ecfb9d1` |
| `habit` | `fiq_n02`<br>train 20% · eval 0% | `circus_fashioniq_release_v1` | `image_only_empty_text` | **COMPLETED** | 1.00000 | 3.315009162085624 | 10.31151091121106 | 0.11373588650412819 | — | `einsum` | BLAS-neutral (Δ 0.0) | `497c25c57fbeba885ecfb9d1` | `summary/habit_fiq_n02_497c25c57fbeba885ecfb9d1` |
| `intent` | `fiq_n02`<br>train 20% · eval 0% | `circus_fashioniq_release_v1` | `multimodal` | **COMPLETED** | 1.00000 | 34.39946693320007 | 54.00633016824921 | 0.3184910253798891 | 0.37538250032494136 | `einsum` | BLAS-neutral (Δ 0.0) | `37ee72aa298b276d1bfda0fa` | `summary/intent_fiq_n02_37ee72aa298b276d1bfda0fa` |
| `intent` | `fiq_n02`<br>train 20% · eval 0% | `circus_fashioniq_release_v1` | `text_only_black_image` | **COMPLETED** | 1.00000 | 15.32567049808429 | 26.73663168415792 | 0.19893506794173196 | — | `einsum` | BLAS-neutral (Δ 0.0) | `37ee72aa298b276d1bfda0fa` | `summary/intent_fiq_n02_37ee72aa298b276d1bfda0fa` |
| `intent` | `fiq_n02`<br>train 20% · eval 0% | `circus_fashioniq_release_v1` | `image_only_empty_text` | **COMPLETED** | 1.00000 | 2.865234049641846 | 9.128768948858903 | 0.11223296860976952 | — | `einsum` | BLAS-neutral (Δ 0.0) | `37ee72aa298b276d1bfda0fa` | `summary/intent_fiq_n02_37ee72aa298b276d1bfda0fa` |
| `hint` | `fashioniq`<br>unstated · eval 0% | `circus_fashioniq_release_v1` | `multimodal` | **COMPLETED** | 1.00000 | 32.816924870897886 | 51.77411294352824 | 0.30926916692748224 | 0.43870420817223343 | `einsum` | BLAS-neutral (Δ 0.0) | `4bc9893a977e501b82378e76` | `summary/hint_fashioniq_4bc9893a977e501b82378e76` |
| `hint` | `fashioniq`<br>unstated · eval 0% | `circus_fashioniq_release_v1` | `text_only_black_image` | **COMPLETED** | 1.00000 | 11.211061136098618 | 21.58920539730135 | 0.17359148193847487 | — | `einsum` | BLAS-neutral (Δ 0.0) | `4bc9893a977e501b82378e76` | `summary/hint_fashioniq_4bc9893a977e501b82378e76` |
| `hint` | `fashioniq`<br>unstated · eval 0% | `circus_fashioniq_release_v1` | `image_only_empty_text` | **COMPLETED** | 1.00000 | 3.3983008495752123 | 9.928369148758954 | 0.11427446535657035 | — | `einsum` | BLAS-neutral (Δ 0.0) | `4bc9893a977e501b82378e76` | `summary/hint_fashioniq_4bc9893a977e501b82378e76` |
| `conesep` | `fiq_n02`<br>train 20% · eval 0% | `circus_fashioniq_release_v1` | `multimodal` | **COMPLETED** | 1.00000 | 35.48225887056472 | 54.13959686823255 | 0.3294414713812836 | 0.3942551173824983 | `einsum-optimized` | `unknown (predates the pin)` | `30f6de423861cc1d6c78c361` | `summary/conesep_fiq_n02_30f6de423861cc1d6c78c361` |
| `conesep` | `fiq_n02`<br>train 20% · eval 0% | `circus_fashioniq_release_v1` | `text_only_black_image` | **COMPLETED** | 1.00000 | 15.092453773113442 | 26.869898384141262 | 0.19955748541119267 | — | `einsum-optimized` | `unknown (predates the pin)` | `30f6de423861cc1d6c78c361` | `summary/conesep_fiq_n02_30f6de423861cc1d6c78c361` |
| `conesep` | `fiq_n02`<br>train 20% · eval 0% | `circus_fashioniq_release_v1` | `image_only_empty_text` | **COMPLETED** | 1.00000 | 3.1984007996001997 | 9.611860736298517 | 0.11206403391502913 | — | `einsum-optimized` | `unknown (predates the pin)` | `30f6de423861cc1d6c78c361` | `summary/conesep_fiq_n02_30f6de423861cc1d6c78c361` |
| `tme` | `fashioniq_0`<br>0% | `circus_fashioniq_release_v1` | `multimodal` | **COMPLETED** | 1.00000 | 34.01632517074796 | 53.95635515575545 | 0.32038840538867464 | 0.3586012581618244 | `einsum` | BLAS-neutral (Δ 0.0) | `ad477312aa57c1b9114f9e51` | `summary/tme_fashioniq_0_ad477312aa57c1b9114f9e51` |
| `tme` | `fashioniq_0`<br>0% | `circus_fashioniq_release_v1` | `text_only_black_image` | **COMPLETED** | 1.00000 | 16.075295685490588 | 28.035982008995504 | 0.20549672011583525 | — | `einsum` | BLAS-neutral (Δ 0.0) | `ad477312aa57c1b9114f9e51` | `summary/tme_fashioniq_0_ad477312aa57c1b9114f9e51` |
| `tme` | `fashioniq_0`<br>0% | `circus_fashioniq_release_v1` | `image_only_empty_text` | **COMPLETED** | 1.00000 | 2.6986506746626686 | 9.22871897384641 | 0.1117059272471363 | — | `einsum` | BLAS-neutral (Δ 0.0) | `ad477312aa57c1b9114f9e51` | `summary/tme_fashioniq_0_ad477312aa57c1b9114f9e51` |
| `encoder` | `fashioniq`<br>unstated · eval 0% | `circus_fashioniq_release_v1` | `multimodal` | **COMPLETED** | 1.00000 | 31.01782442112277 | 49.5752123938031 | 0.29586071325251023 | 0.47230210227676583 | `matmul` | `unknown (predates the pin)` | `6a528e877c328772e2c7bb28` | `summary/encoder_fashioniq_6a528e877c328772e2c7bb28` |
| `encoder` | `fashioniq`<br>unstated · eval 0% | `circus_fashioniq_release_v1` | `text_only_black_image` | **COMPLETED** | 1.00000 | 8.579043811427619 | 18.1575878727303 | 0.15612507640224627 | — | `matmul` | `unknown (predates the pin)` | `6a528e877c328772e2c7bb28` | `summary/encoder_fashioniq_6a528e877c328772e2c7bb28` |
| `encoder` | `fashioniq`<br>unstated · eval 0% | `circus_fashioniq_release_v1` | `image_only_empty_text` | **COMPLETED** | 1.00000 | 2.6653339996668333 | 7.4795935365650505 | 0.10727509765806255 | — | `matmul` | `unknown (predates the pin)` | `6a528e877c328772e2c7bb28` | `summary/encoder_fashioniq_6a528e877c328772e2c7bb28` |
| `ptha_mtst` | `cirr_transfer`<br>CIRR→FashionIQ TRANSFER | `circus_fashioniq_release_v1` | `multimodal` | **COMPLETED** | 1.00000 | 19.473596535065802 | 34.44944194569382 | 0.22493423438490692 | 0.24362219946484542 | `einsum-optimized` | `unknown (predates the pin)` | `315dd590eeb03d2c78d7937a` | `summary/ptha_mtst_cirr_transfer_315dd590eeb03d2c78d7937a` |
| `ptha_mtst` | `cirr_transfer`<br>CIRR→FashionIQ TRANSFER | `circus_fashioniq_release_v1` | `text_only_black_image` | **COMPLETED** | 1.00000 | 10.811261036148592 | 21.0727969348659 | 0.17013526146911484 | — | `einsum-optimized` | `unknown (predates the pin)` | `315dd590eeb03d2c78d7937a` | `summary/ptha_mtst_cirr_transfer_315dd590eeb03d2c78d7937a` |
| `ptha_mtst` | `cirr_transfer`<br>CIRR→FashionIQ TRANSFER | `circus_fashioniq_release_v1` | `image_only_empty_text` | **COMPLETED** | 1.00000 | 1.3326669998334166 | 4.6476761619190405 | 0.09498086792403913 | — | `einsum-optimized` | `unknown (predates the pin)` | `315dd590eeb03d2c78d7937a` | `summary/ptha_mtst_cirr_transfer_315dd590eeb03d2c78d7937a` |
| `limn` | `base_iter0_all_categories`<br>unstated · eval 0% | `circus_fashioniq_release_v1` | `multimodal` | **SUPERSEDED_NOT_RERUN** (PARTIAL, 5884/6003) | 0.980176578377478 | 30.064581917063222 | 48.759347382732834 | 0.2908449032087251 | 0.5417445849251903 | `matmul` | `unknown (predates the pin)` | `10173dc45dea8c90fd2a284b` → expected `2a4b19f335f5ec281375d09f` | `summary/limn_base_iter0_all_categories_10173dc45dea8c90fd2a284b` |
| `limn` | `base_iter0_all_categories`<br>unstated · eval 0% | `circus_fashioniq_release_v1` | `text_only_black_image` | **SUPERSEDED_NOT_RERUN** (PARTIAL, 5884/6003) | 0.980176578377478 | 6.3052345343303875 | 14.208021753908906 | 0.13328125184230716 | — | `matmul` | `unknown (predates the pin)` | `10173dc45dea8c90fd2a284b` → expected `2a4b19f335f5ec281375d09f` | `summary/limn_base_iter0_all_categories_10173dc45dea8c90fd2a284b` |
| `limn` | `base_iter0_all_categories`<br>unstated · eval 0% | `circus_fashioniq_release_v1` | `image_only_empty_text` | **SUPERSEDED_NOT_RERUN** (PARTIAL, 5884/6003) | 0.980176578377478 | 2.4983004758667575 | 6.985044187627464 | 0.09970453632376666 | — | `matmul` | `unknown (predates the pin)` | `10173dc45dea8c90fd2a284b` → expected `2a4b19f335f5ec281375d09f` | `summary/limn_base_iter0_all_categories_10173dc45dea8c90fd2a284b` |
| `csmcir` | `fashioniq`<br>0% | `circus_fashioniq_release_v1` | `multimodal` | **BLOCKED_ADAPTER** — `encode_targets` raises `_TARGET_REASON`; xem BLOCKERS §B1 | 0.00000 | — | — | — | — | `einsum-optimized` | n/a (không có artifact) | — | — |
| `csmcir` | `fashioniq`<br>0% | `circus_fashioniq_release_v1` | `text_only_black_image` | **BLOCKED_ADAPTER** — xem BLOCKERS §B1 | 0.00000 | — | — | — | — | `einsum-optimized` | n/a (không có artifact) | — | — |
| `csmcir` | `fashioniq`<br>0% | `circus_fashioniq_release_v1` | `image_only_empty_text` | **BLOCKED_ADAPTER** — xem BLOCKERS §B1 | 0.00000 | — | — | — | — | `einsum-optimized` | n/a (không có artifact) | — | — |

### Ghi chú về các hàng vừa sửa

- **`habit`** trước đây ghi `RUNNING — run_id f76171fa32296f3369970a5c`, coverage `0.00000`. Run đó là
  một lần thử **trước khi relaunch** và đã chết; nó không có trong `summary/`. Kết quả thật là
  `497c25c57fbeba885ecfb9d1`, COMPLETED, coverage 1.0, `topk_export` `TOPK_EXPORT_COMPLETE` depth 50.
- **`intent`, `hint`, `conesep`** trước đây ghi `NOT_RUN` với artifact trống. Cả ba **đã COMPLETED**,
  coverage 1.0, `candidate_topk` depth 50. Số ở trên đọc trực tiếp từ `full.json` của từng run.
- **`limn`** trước đây trình bày `30.0646 / 0.98018` như hiện hành. Nó **bị superseded** bởi thay đổi
  scope-resolver: `identity` nay gồm `query_scope_resolution`, nên run id đổi thành
  `2a4b19f335f5ec281375d09f` và **chưa được chạy lại**. Con số `30.06%` vì thế là số **5884-mẫu số cũ**;
  nó **không** phải giá trị sẽ có sau khi wire resolver (6001/6003 query resolve, 2 query thực sự nhập
  nhằng `7:3038`, `7:3790`). Đừng trích nó như số hiện tại.

---

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
- **category-scoped**: chỉ `limn` cần gán category theo reference image. Sau khi wire resolver: 6001/6003 query resolve, **2 genuinely ambiguous** (`7:3038`, `7:3790`) — xem BLOCKERS §B4.
- **reference_eligible (khai báo)** trước đây là literal `True` ở cả 9 adapter (defect F7).
- **reference_eligible (đo)**: `CompositionAdapter.measured_reference_eligibility()` chạy `score`+`rank` thật của adapter trên gallery tổng hợp nơi reference là exact match.
- **gate F1**: trước fix chỉ 5/9 adapter gọi gate môi trường. Nay 9/9.

## Tổng hợp

- Model có `composition_adapter`: **9**.
- Chạy được về môi trường (env + checkpoint staged + adapter nhận): **8** (trừ `csmcir`).
- Có artifact full-run (6003 query, `subset=full`): **8 run** — `habit` ×2 (một COMPLETED `497c25c`, một
  **FAILED** `e0dd04a`), `intent`, `hint`, `conesep`, `tme`, `encoder`, `ptha_mtst` (tất cả COMPLETED), và
  `limn` (**PARTIAL**, 5884/6003, superseded).
- **Chưa từng chạy**: `csmcir` (adapter tự từ chối) — và `limn` **chưa chạy lại** dưới run id mới.
- **Không** model nào ở bảng trên là paper reproduction. `conesep` mang nhãn `fiq_n02` = train noise 20%;
  `ptha_mtst` là CIRR→FashionIQ **transfer**, không phải FashionIQ training.

> **Không** hàng nào ở đây được gọi là NEW_RUN khi chưa có artifact do phiên này sinh.
