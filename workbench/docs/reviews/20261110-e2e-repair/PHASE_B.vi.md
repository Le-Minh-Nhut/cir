# Phase B — RUN_MATRIX (báo cáo ngắn)

`RUN_MATRIX.csv`: **61 dòng** = **50 dòng composition** (22 run × subset) + **11 dòng native**,
một cột cho mỗi surface. Sinh bằng `scripts/pb_run_inventory.py` → `raw/run_inventory.json` →
`scripts/pb_run_matrix.py`. Mọi con số dưới đây đọc lại từ chính CSV, không chép tay.

## Tách cột là có lý do — chúng fail độc lập

| surface | đo được | nghĩa |
|---|---|---|
| execution | 37 COMPLETED, 4 PARTIAL, 3 FAILED (`habit_fiq_n02_e0dd04a4…`), 6 dòng fixture | chạy xong **không** bằng đo được |
| validation | 41 dòng `NOT_RUN` (không receipt nào trong `workbench/artifacts/receipts/`, `validation_status` vắng ở **mọi** summary); 6 `N/A - fixture`; 3 `NOT_APPLICABLE` (`released_audit`) | validation **trống toàn bộ**, không phải "pass" |
| export | 12 `EXPORTED` (3 `.jsonl` + 3 manifest, depth 50, `record_count=6003`, `missing_query_count=0`); 8 `COVERED BY RUN EXPORT` (subset sf/validated nằm trong export 6003 của chính run); 21 `NOT_EXPORTED`; 9 `N/A` | export thiếu = thiếu artifact, không phải ranking sai |
| serving | 41 `RANKS_PRESENT`, 6 `NO_RANKS` (fixture), 3 `IMPORTED_ARTIFACT` (`released_audit`) | phục vụ được ≠ đã đo |
| cache | `resume_key` + `gallery_identity_sha256` (`1db68c59…`, 74381) có trên mọi run thật | cache identity tách khỏi metric identity |
| rerun | 12 `KEEP`, 17 `EXPORT ONLY`, 8 `DO_NOT_RERUN`, 6 `NEVER` (fixture), 4 `RESUME`, 3 `RERUN` | hành động là cột riêng, không phải phán đoán ngầm |

## Cặp trùng xác định bằng số, không bằng tên

Cùng `model_id` + `checkpoint_id` + `subset` + `query_count` + `gap_cohort_count` **và** `gap` trùng
tới chữ số cuối, chỉ khác `run_id`:

* `habit_fiq_n02_497c25c5…` ≡ `habit_fiq_n02_c6eb9d57…` — gap `0.388704727214254`
* `hint_fashioniq_4bc9893a…` ≡ `hint_fashioniq_f461f391…` — gap `0.43870420817223343`
* `intent_fiq_n02_37ee72aa…` ≡ `intent_fiq_n02_4901da13…` — gap `0.37538250032494136`
* `tme_fashioniq_0_ad477312…` ≡ `tme_fashioniq_0_cd1a696d…` — gap `0.3586012581618244`

Không cặp nào bị tự động xoá: matrix chỉ gắn `DO_NOT_RERUN` + tên bản trùng. Chọn bản giữ lại thuộc
người tổng hợp, vì hai bản có thể khác `adapter_module_source_status`.

**Không** trùng (đã kiểm, đừng gộp) — ba run ENCODER khác `resume_key` và khác gap:
`5dee7425…` → `0.469868464471065`; `6a528e87…` → `0.47230210227676583`;
`eda30a37…` → `0.5329370636334637` (q=8, smoke). Cùng `checkpoint_identity
0f31c50150a7…` nhưng khác attempt/cache identity ⇒ ba dòng riêng, không phải bản sao.

## Nối Phase A

`clip4cir_rn50x4_fullft` chỉ xuất hiện ở nhánh native; không run composition nào của nó tồn tại.
Vì vậy migration nhãn ở Phase A không chạm dòng nào trong `RUN_MATRIX`. Native: 11 dòng, trong đó
`ptha_mtst.json` là JSON không đọc được (`raw/native/ptha_mtst.json`, 4715 byte, lỗi cú pháp) —
dòng đó ghi `UNREADABLE_JSON` và **không** được dùng làm bằng chứng; `ptha_mtst.result.json` mới là
bản đọc được.

## Chưa làm

Phase C (R2, R3, R4, R10, R11) **chưa bắt đầu**; chưa chạy job GPU/CPU nào. `PARITY.vi.md` sẽ được
tạo ở Phase C kèm verdict từng finding và lệnh chạy lại.
