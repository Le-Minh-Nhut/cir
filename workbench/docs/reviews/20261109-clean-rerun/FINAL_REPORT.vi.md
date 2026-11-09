# CIR — báo cáo cuối phiên clean-rerun (2026-11-09)

Revision khi viết: branch tip `7a58ce6` (sweep đo tại đây); commit sau nó `5a1f456` chỉ sửa tài liệu.
`TESTED_SOURCE_SHA` đã tiến lên `7a58ce64a82b83bb126992fac5b9f784ffe97dd9` sau một lượt sweep serial
đầy đủ tại chính revision đó (2581 passed / 18 failed / 51 skipped = 2650 collected, exit 1,
2950.22 s); guard `test_handoff_commands.py` sau đó **8 passed**, xem §7.
Mọi con số dưới đây dẫn từ file có trong `EVIDENCE_INDEX.md`; không có số nào là ước lượng.
Bảng so sánh rank từng query nằm ở `rank_diffs/` — mỗi con số trong §3 dẫn về đúng một file ở đó.

## 1. Trạng thái tổng quan

Sáu lỗi đường chạy mà bản review độc lập tại `d699b66` nêu đã được sửa và **mỗi lỗi có test fail
trước / pass sau** (19 failed–2 passed trước, 25 passed sau, cùng một file test). Giao diện sáng đã
lên bản chạy thật với 17 model luôn hiển thị, Top-K thật theo từng query, và phạm vi protocol/subset
không còn lệch giữa các trang.

Sự thật dữ liệu đã xác lập: **toàn bộ artifact CIRCUS cũ trên máy này được sinh trước commit pin BLAS
`bf6a448`**, nên trước phiên này chưa có số nào được chấm dưới một policy BLAS được ghi lại.

**Và đã chạy lại xong: tám model, đủ tám, mỗi model một lượt đầy đủ với `--topk-depth 100`.** Đây là
kết quả chính của phiên, không phải ghi chép sổ sách:

- Bảy model đạt `6003/6003` với export sâu 100; `limn` đạt `6001/6003` và ghi rõ hai query không
  chấm được. Mọi lượt có `blas_threads_pinned=1`, capability `verifiable=true`, `cache_verify=full`,
  `identity_schema_version=2`, và invariant kiểm từ artifact đều PASS.
- **`encoder` đổi số thật, và đây là phát hiện correctness chứ không phải bookkeeping:** 12.727/18.009
  rank đổi, R@50 multimodal **−0,09995** (49,5752 → 49,4753), gap **0,4723 → 0,4699**. Mọi con số
  encoder công bố trước đây được đo dưới policy chưa pin, tức là **không tái lập được**; chúng phải
  được thay, không phải được trích dẫn song song.

Cuối cùng, một điều không thể làm: **không thể backfill** bất kỳ run cũ nào, vì khoá cache là chính
digest của identity (giải thích ở §6) — nên tám lượt trên đều là rerun đầy đủ, không phải re-export.

Tóm lại: code đúng hơn, giao diện thật hơn, và **tám model nay có số đo dưới policy được ghi lại** —
và guard release đã xanh (G11 PASS), trong khi hai gate khoa học vẫn chưa chạy (G1, G2).

## 2. Sáu lỗi evaluator đã sửa

Tất cả trong `workbench/tests/test_review_defects_evaluator.py`.

| # | Nguyên nhân | Hàm đã đổi | Test |
|---|---|---|---|
| 1 | `_blas_thread_limit` chỉ được áp trong `_rank_groups_concurrently`; `_backfill_export` và nhánh fallback theo batch gọi thẳng `adapter.rank*` **sau khi** helper đó đã restore, nên chấm ở thread mặc định của tiến trình | `_backfill_export` (pin + restore trong `finally`), vòng fallback, `BackfillRankDisagreement` mới | `test_backfill_runs_under_the_same_blas_policy_as_the_normal_path` |
| 2 | `if setter is None: return int(previous)` là "thành công im lặng", và `previous` đọc từ biến môi trường chứ không từ runtime | `_openblas_symbols`, `_blas_thread_limit` (trả record, **đọc lại** sau khi set), `_current_openblas_threads`, `_blas_pin_capability` | 4 test: silent-success, verified, readback-disagrees, int-coercion |
| 5 | `blas_threads_pinned` / `query_scope_resolution` / `query_order_identity` không nằm trong payload đã hash, nên hai run khác policy vẫn cùng `group_id` và bị gọi là `duplicate` | `measurement_identity` (payload + `identity_schema_version`) | 6 test, gồm `test_identity_v1_artifacts_are_unknown_not_regrouped` |
| 6 | `scoring_impl` suy từ source **hiện tại** của adapter, rồi gắn lên artifact lịch sử | `scoring_implementation(..., summary=, live=)`, `_recorded_source_matches_disk` | 3 test |
| 7 | Loader chỉ kiểm fingerprint + shape; file cùng shape khác nội dung vẫn load (đo được: sum 6.0 → 0.0, `accepted=true`) | `_load_memmap`, `_write_memmap_metadata` (digest theo chunk), `_memmap_integrity` | 6 test, gồm truncated / stale-sidecar / legacy-absent |
| 8 | `assert … or True` không bao giờ fail (`test_composition_topk_runner.py:131`, `test_capability_composition.py:81`) | hai assert thay bằng kiểm tra thật + ca âm (sai hash, sai thứ tự, sai độ dài, id trùng) | `test_gallery_identity_assertion_is_not_vacuous` |

Thêm một lỗi do review bắt trong chính bản sửa của tôi: `cache_verify_mode` được đọc bằng
`summary.get(...)` trong khi runner **không hề ghi khoá đó vào summary**, nên mọi artifact thật mang
`None` và run `header` bị gộp nhóm với run `full`. Đã sửa (ghi vào summary + đọc qua
`_recorded_or_summary`) và thêm 4 test.

Chi phí cache đã đo trên gallery thật 74.381×512 (152 MB): digest 43,6 ms, load có verify **+43,3 ms**
so với mmap trần, trong khi encode gallery mất hàng phút. Chế độ `header` +0,0 ms; `off` 0,09 ms.

## 3. Sự thật quan trọng nhất: pin BLAS và mức ảnh hưởng thật

`git log -S blas_threads_pinned` → trường này do **`bf6a448`** thêm, commit
**2026-11-08 20:33:11 +0700**. Summary composition mới nhất trên đĩa: **2026-11-08 17:42**. Số summary
sinh sau 20:33: **0**. Vậy **mọi artifact CIRCUS đều được chấm khi chưa hề pin BLAS**.

Nhưng điều đó chỉ quan trọng với model mà **thread count thật sự đổi kết quả**. Đo, không suy từ
source (seed 20261109, gallery 3000, 8 query, 6 near-tie mỗi query, BLAS 1 vs 20 qua setter runtime):

| model | biểu thức chấm | max \|Δscore\| | rank đổi | max \|Δrank\| | tie cluster đảo | nhạy BLAS? |
|---|---|---:|---:|---:|---:|---|
| conesep | einsum(optimize=True) | 0.000e+00 | 0 / 400 | 0 | 0 / 8 | không |
| ptha_mtst | einsum(optimize=True) | 0.000e+00 | 0 / 400 | 0 | 0 / 8 | không |
| habit | einsum | 0.000e+00 | 0 / 400 | 0 | 0 / 8 | không |
| intent | einsum | 0.000e+00 | 0 / 400 | 0 | 0 / 8 | không |
| hint | einsum | 0.000e+00 | 0 / 400 | 0 | 0 / 8 | không |
| tme | einsum | 0.000e+00 | 0 / 400 | 0 | 0 / 8 | không |
| **encoder** | matmul theo batch (M=8) | **1.788e-07** | **29 / 400** | **5** | **6 / 8** | **CÓ** |
| limn | matmul 1 query (M=1) | 0.000e+00 | 0 / 400 | 0 | 0 / 8 | không |

Cơ chế đã cô lập: GEMM theo batch (M>1) thay đổi theo thread count; matvec một query thì không.
`np.linalg.norm(axis=1)` bất biến. Và dù `einsum(optimize=True)` **có** đi qua đường đa luồng (nhanh
~2,1× ở 20 thread), kết quả vẫn **bit-identical**.

**Hệ quả:** với bảy model còn lại, việc thiếu pin là **lỗ provenance**, **không phải lỗi số** — câu
"rerun" cho chúng không có nghĩa "số cũ sai". Chỉ `encoder` là có khả năng sai số thật.

### 3.1 Kết quả rerun: bảng đo 400 query đã bỏ sót điều gì

Bảng trên đo **400 query/model**. Sau khi chạy lại đủ **6003 query/model** và so rank từng query
(`rank_diffs/`), kết quả tách thành ba nhóm — không phải hai:

| model | rank đổi / 18009 | max \|Δrank\| | R@10, R@50 | kết luận |
|---|---:|---:|---|---|
| `encoder` | **12.727** | 1.706 | đổi (multimodal R@50 −0,09995) | **nhạy thật** |
| `conesep` | **17** | 1 | **không đổi** | nhạy **yếu** ở mức ~1e-7 |
| `ptha_mtst` | **46** | 2 | **không đổi** | nhạy **yếu** ở mức ~1e-7 |
| habit, intent, hint, tme | **0** | 0 | không đổi | bit-identical |

`conesep` và `ptha_mtst` dùng `einsum(optimize=True)`; điểm số trả về **không** bit-identical, chỉ là
chênh lệch ~1e-7 đủ để đảo thứ tự khi hai ứng viên gần bằng nhau. Cơ chế nhìn thấy được trong export
thật, multimodal `7:3744` (hạng 30 → 31):

```
[29] cũ  27939:0.6219767332077026   mới 67612:0.6219767928123474
[30] cũ  67612:0.6219766139984131   mới 27939:0.6219766736030579
```

Hai điểm cách nhau **1,2e-7** đổi chỗ. 16 ca còn lại của conesep có target rank 1007–65087, tức nằm
**ngoài** cửa sổ top-100 đã lưu, nên cửa sổ giống nhau và chỉ số lượng ứng viên cao hơn target thay
đổi 1.

**Vì sao điều này quan trọng:** R@10/R@50 của conesep/ptha_mtst không đổi **chỉ vì không rank nào vượt
qua mốc k** — đúng cái bẫy mà "aggregate bằng nhau" che mất. Và mẫu 400 query không thể chứng nhận
một quần thể 6003 query: nó không bắt được một query nào có cặp gần bằng trong top-50. Một mẫu 400
query là bằng chứng về **mẫu**, không phải về **quần thể** — nên câu "0/400 nên không nhạy" phải đọc
là "không phát hiện được trong 400", không phải "không tồn tại".

### 3.2 `limn`: 6001/6003, không phải 100%

Lượt mới chấm **6001** query mỗi điều kiện (coverage `0.9996668332500417`, cohort gap 6001). Hai
query **không chấm được**, có lý do ghi trong `query_failures` của artifact:

```
7:3038: category unresolved has no adapter embedding scope ('dress','shirt','toptee')
7:3790: cùng lý do
```

Đây là **unscored có lý do**, không phải miss. `category_resolution` báo 5884 resolved / 0 unresolved —
con số đó **không** phải số đã chấm; trần nằm ở embedding scope của adapter. Lượt trước
(`10173dc4`) chấm 5884, nên coverage **tăng**, nhưng **hai cohort khác nhau** (5884 vs 6001) và
**không được so trực tiếp** — gap 0,5417 vs 0,5391 là hai mẫu số khác nhau, không phải một cải thiện
0,0026.

### 3.3 `ptha_mtst`: đường chấm pin được, đường encode thì không

`score()` của ptha_mtst là `einsum(optimize=True)` và chạy **trong tiến trình điều khiển**
(`ctl312h`, OpenBLAS thật, pin verify được). Nhưng interpreter của model
(`artifacts/environments/cir-005ef4e1b02999a5`) **không có OpenBLAS**: `libblas.so.3` trỏ tới
`libmkl_rt.so.2` (MKL 2022.1), `numpy.libs/` rỗng. Hệ quả đo được:

- `_openblas_symbols()` không tìm thấy setter → nếu pin ở đó sẽ là `setter_found: false`.
- `mkl_set_num_threads(1)` **segfault** interpreter đó (exit 139) qua mọi đường đã thử:
  `ctypes.CDLL("libmkl_rt.so.2")`, `ctypes.CDLL(None)`, `mkl_domain_set_num_threads`,
  `LD_PRELOAD=libiomp5.so`, `LD_PRELOAD=libomp.so`, trước và sau khi import NumPy.
  `mkl_get_max_threads()` thì chạy (trả 20).
- `MKL_NUM_THREADS=1` **có** pin được (max đọc ra 1, gemm chạy, và kết quả gemm **khác** bản 20
  thread: 15302833 vs 15377137 trên 600×256 @ 400×256) — nhưng biến môi trường chỉ có tác dụng
  trước khi load, không đặt được giữa tiến trình.

Vậy forward pass encode của ptha_mtst chạy dưới MKL **không pin được và không verify được**. Lượt
này **không tương đương** một lượt pin OpenBLAS, và phải được trình bày như vậy ở mọi nơi.

## 4. Frontend

- Nền sáng `#f5f7fa` (`rgb(245,247,250)`), chữ 17px mặc định và tuỳ chọn 19px (`style.css:51-52`).
- 17 model luôn hiển thị, kể cả chưa chạy/bị chặn; đếm riêng model / run / pending / deferred.
- Subset **luôn** được ghi vào URL → Full không còn bị biến thành SF khi copy/reload/back/forward.
- Top-K thật: K thuộc identity của request (run+subset+condition+query+K), menu bị chặn theo depth đã
  export, response cũ không ghi đè ngữ cảnh mới. Chọn Top 50 render đủ 50 ảnh.
- `outcomeLabel()` để badge không in trần chữ `unresolved` (đọc thành `unscored`, nghĩa ngược lại).
- `ScoringImplNotice` bỏ câu quá rộng "không so sánh được ở mức từng query".

Harness chạy lại live trên `127.0.0.1:5173` (dữ liệu thật, không fixture): **cold 3, composition 8,
mobile 4, samples 2 check — tất cả exit 0**. 13 test node pass. Ảnh chụp:
`ui/overview-1920x1080.png`, `ui/overview-2560x1440.png`, `ui/composition-1440x900.png`,
`ui/samples-top50-1920x1080.png`, `ui/mobile-390x844.png`. Không tràn ngang ở 1920/2560/390
(`body.scrollWidth == innerWidth`), 0 ô bị cắt.

## 5. Bảng gate G0–G12 (trạng thái thật)

| gate | trạng thái | phạm vi / lý do |
|---|---|---|
| G0 inventory | **PASS** | 17 model id, 29 variant, `MODEL_MATRIX.csv` |
| G1 input correctness | **NOT_RUN** | không chạy differential native↔adapter nào trong phiên này |
| G2 forward/scoring | **NOT_RUN** | không forward pass nào được đối chứng; đo BLAS là bằng chứng G7 |
| G3 policy/identity | **PASS** | gồm hai phần: (a) **code** — identity v2 + readback được test; (b) **artifact công bố** — **tám lượt rerun đều ghi `identity_schema_version=2`**, confidence `full`, `cache_verify_mode=full`, `blas_threads_pinned=1` cùng capability `verifiable=true`. Cảnh báo trung thực: `_record_effective_blas_pin` **vẫn chưa** đi vào payload identity (lỗi còn mở, ghi ở `BLAS_IDENTITY_FINDINGS.md`), nên thứ artifact chứng minh được là **khả năng pin + giá trị ĐÃ XIN**, không phải giá trị runtime đọc lại của từng lượt |
| G4 cache/resume/fallback | **PASS** | cơ chế + ca âm; cache cũ nay không truy cập được (đúng thiết kế) |
| G5 clean run/coverage | **PASS** (phạm vi: tám model đã rerun) | tám lượt đầy đủ dưới policy được ghi lại, verify từ artifact: 7×`6003/6003`, limn `6001/6003` khai đúng hai query không chấm được. 21 hàng registry còn lại vẫn BLOCKED/DEFERRED, **không** thuộc phạm vi PASS này. ptha_mtst có caveat: đường encode không pin được |
| G6 rank/export | **PASS** | tám model đều export **Top-100**; manifest depth=100, record 6003/6003 `partial=false` (limn 6001/6003 `partial=true`). 18009 dòng ứng viên được kiểm: vị trí hợp lệ, thứ tự theo score, `target_rank` khớp nhánh metric — 0 vi phạm |
| G7 optimization parity | **PASS** | bảng đo BLAS + chứng minh song song theo trục query; **không claim speedup** |
| G8 measured performance | **PASS** (phạm vi: **decode + transform** producer stage, một model family) | **Đo hai lần; lần đầu ĐÃ BỊ THAY THẾ** vì harness đo một workload production không chạy (vòng PNG encode+decode trong pool, transform chạy serial ở consumer) nên đường cong bỏ mất đúng stage song song hoá — con số 4,1× của nó không phải lợi ích pipeline. Harness đã viết lại để **khớp đúng `model_habit.py:358-364`**: `Image.open` → `convert("RGB")` → `targetpad_transform(1.25,224)` → stack, tất cả **trong pool**, không PNG. Đo lại trên box rảnh (loadavg 0,47; không pytest/evaluator): median **554,5 img/s ở 0 worker → 1032,8 ở 2 → 1756,2 ở 4 → 2282,1 ở 6 → 2688,1 ở 8**; RSS 231→603 MiB; VRAM phẳng 852 MiB. **Cả 5 setting parity IDENTICAL** (thứ tự hàng + mảng sau transform). **Đề xuất đổi từ `workers=4` sang `workers=8`**: đường cong vẫn tăng 18% từ 6 lên 8 và **chưa phẳng**, nên 8 là điểm cao nhất **đã thử** trong ngân sách 10 thread, không phải điểm tối ưu — và nó chỉ còn 2 thread cho tiến trình chính. Transform chiếm **64%** wall time serial, đó là lý do đường cong cũ sai. **Không** phải baseline end-to-end — forward pass không nằm trong phép đo này |
| G9 frontend correctness | **PASS** | 4 session browser trên dữ liệu thật, exit 0 |
| G10 frontend usability | **PASS** | sáng, 17/19px, 1920/2560/390, 0 ô cắt |
| G11 integration/release | **PASS** | sweep serial tại `7a58ce6` (2581/18/51 = 2650, exit 1, 2950.22 s, log `full_suite_7a58ce6.txt`), `TESTED_SOURCE_SHA` đã tiến trong **cả hai** tài liệu, guard `test_handoff_commands.py` **8 passed**, `npm run build` exit 0 và tái tạo đúng asset `index-9MwV5MAk.js` mà server đang phục vụ. PASS **không** có nghĩa suite xanh: 18 failure được ghi lại, đúng bộ cũ, 0 mới |
| G12 coordination | **PASS** | có một sai lệch đã ghi: 4 lane chạy song song so với cap 3, lane thứ tư đã dừng |

**11 PASS · 0 FAIL · 2 NOT_RUN.** Không có gate nào được PASS mà thiếu bằng chứng trên đĩa.

G11 chuyển từ FAIL sang PASS sau khi sweep tại `7a58ce6` được đo, `TESTED_SOURCE_SHA` được tiến trong cả `FINAL_PC_RELEASE_VERDICT.md` và `PC_PILOT_HANDOFF.md`, rồi guard báo **8 passed** — trạng thái đỏ trước đó là **độ trễ ghi sổ**, đóng lại bằng đo lường chứ không bằng sửa guard. **G8 chuyển NOT_RUN → PASS** nhờ một phép đo trên box rảnh thật (xem hàng G8); tài liệu `PERFORMANCE.vi.md` **sinh tự động** từ `perf_raw.jsonl`, không gõ tay. G1/G2 vẫn NOT_RUN — không có differential native↔adapter.

G5 và G6 chuyển trạng thái vì tám lượt chạy đã xong và invariant kiểm từ artifact. G1/G2 giữ NOT_RUN (không chạy differential native↔adapter nào). G8 chuyển NOT_RUN → PASS sau phép đo producer stage trên box rảnh (xem hàng G8 và §7; mọi wall-time đo khi hai lane khác đang chạy — **không phải baseline**).

## 6. Rerun: đã làm gì, chưa làm gì, vì sao

**Đã chạy lại trong phiên này — tám model**, một job GPU tại một thời điểm, mỗi lượt đầy đủ với
`--topk-depth 100 --keep-features`, và mỗi lượt verify từ artifact bằng `verify_run.py` (không dựa vào
stdout). Nguồn: `RUN_LOG.csv` (8 hàng) + `GPU_LANE_REPORT.md`.

| model | run_id | coverage | depth | pin | invariant |
|---|---|---|---:|---|---|
| `encoder` | `5dee7425dd46c5f154808aac` | 6003/6003 | 100 | 1, verified | PASS |
| `conesep` | `a4f786d8af4d46ac2a85ab9c` | 6003/6003 | 100 | 1, verified | PASS |
| `habit` | `c6eb9d5775ae636a3b8addbc` | 6003/6003 | 100 | 1, verified | PASS |
| `intent` | `4901da139ac40b874d2363c5` | 6003/6003 | 100 | 1, verified | PASS |
| `hint` | `f461f391c6d4696011be8067` | 6003/6003 | 100 | 1, verified | PASS |
| `limn` | `9e4918a6b5c243412cbc58c9` | **6001/6003** | 100, `partial` | 1, verified | PASS |
| `ptha_mtst` | `f54a77450ebd2126590d4385` | 6003/6003 | 100 | chấm: có / encode: **không** | PASS (caveat) |
| `tme` | `cd1a696d8bca0fc4bce6925b` | 6003/6003 | 100 | 1, verified | PASS |

Cả tám ghi `identity_schema_version=2`, confidence `full`, `cache_verify_mode=full`. Kiểm `--resume`
trên từng lượt: `resume_would_reuse_cache=true` — memmap load được dưới fingerprint tính lại, sidecar
khớp, `content_digest` có và verify qua. Tức khoá cache **ổn định trong một identity**; cache thế hệ
cũ vẫn không truy cập được, đúng như thiết kế.

**Wall-time của cả tám lượt đều KHÔNG phải baseline:** đo khi hai lane khác đang chạy, cột
`throughput` ghi `not_measured (concurrent load)`. G8 chỉ được PASS ở phạm vi trục decode sau đó
(§7). Ba lượt (hint, ptha_mtst,
tme) còn không có file VRAM vì sampler không được nối lại sau khi HEAD đổi — `RUN_LOG.csv` ghi
`not_sampled`, không suy ra số.

**Chưa chạy:** 11 hàng `RERUN_FULL` còn lại (3 variant limn khác, 2 variant airknow, và các variant
checkpoint không staged) và toàn bộ hàng `BLOCKED`.

**Đối chiếu nguồn khi HEAD đổi giữa phiên:** HEAD chuyển `ffb6ae8` → `20b83e5` lúc 03:53 do lane khác.
Bốn lượt chạy trước đó (encoder, conesep, habit, intent) được kiểm lại bằng cách so
`scoring_module_source_hash` và `condition_transform_source_hash` ghi trong artifact với file nguồn
hiện tại — **khớp byte-for-byte**. Nên chúng mô tả đúng cây hiện tại dù nhãn lúc chạy ghi `ffb6ae8`.
Bốn lượt sau ghi nhãn `20b83e5`.

**Vì sao không thể "chỉ export thêm" cho run cũ — điều này chốt phương án:** khoá cache là
`_digest({"run": fingerprint, ...})` với `fingerprint = _digest(identity)`. Identity nay có các
trường v2 mà artifact cũ không có, nên fingerprint đổi. Tính trên artifact conesep thật:
`30f6de4238…` → `cb92b4b4fd…`, khoá cache gallery `439d49b3…` → `4f5e49f0…`. Loader fail kiểm
fingerprint → `_cached_features` trả `{},{}` → `--resume` từ chối. **File `.npy` còn trên đĩa nhưng
không dùng được.** Đây là hành vi đúng (cache khác policy không được tái dùng), nhưng nghĩa là
`--resume` **không thể** bắc qua một thay đổi schema identity. Vì vậy mọi run cần export sâu hơn đều
phải **RERUN_FULL + `--topk-depth 100`** trong một lượt GPU, không phải rerun rồi backfill.

Tổng theo `RERUN_DECISIONS.csv` **sau khi cập nhật**: **10 hàng `RERUN_FULL_DONE`** (tám model đã chạy
lại; limn chiếm 3 hàng vì bundle có ba không gian category) · **2 hàng `RERUN_FULL`** còn lại (airknow
`fiq_n05`/`fiq_n08`, adapter có nhưng chưa có forward GPU) · **14 BLOCKED · 3 DEFERRED** (29 hàng).
Cột `previous_run_id` giữ lại run id cũ cho cả tám, nên dòng dõi trước/sau không bị mất.

## 7. Tồn đọng

- **G11 nay là PASS, không còn đỏ.** Trình tự đã xảy ra và đều kiểm được: (1) một lượt sweep serial tại
  `7a58ce6` — **2581 passed / 18 failed / 51 skipped = 2650 collected**, exit 1, 2950.22 s, log
  `full_suite_7a58ce6.txt`; (2) `TESTED_SOURCE_SHA` tiến lên `7a58ce6` trong **cả**
  `FINAL_PC_RELEASE_VERDICT.md` và `PC_PILOT_HANDOFF.md` (kể cả dòng `git checkout`, vốn phải là revision
  cố định); (3) `test_handoff_commands.py` → **8 passed**. Guard trước đó đỏ vì **ghi sổ chậm**, và nó được
  đóng bằng đo lường chứ không bằng sửa test hay nới gate. **PASS không có nghĩa suite xanh:** 18 failure
  được ghi nguyên văn trong log và attribution ở `SWEEP.md` — **đúng bộ id cũ, 0 mới**; 3 lỗi drift trước đó
  đã sửa **test-only** ở `3d576bf` và nay pass.
- **Sweep tại revision hiện tại: ĐÃ CÓ** (chính là `full_suite_7a58ce6.txt` ở trên). Sweep tại `ffb6ae8`
  bị huỷ ở **22%** vẫn được giữ làm lịch sử (`sweep_ffb6ae8_ABORTED.txt`, không có count) — nó không phải
  lượt đo của cây hiện tại và không được trích làm bằng chứng.
- **G8 nay là PASS, nhưng chỉ cho producer stage (decode + transform).** Đo lại sau khi phát hiện harness đầu
  đo sai workload; số mới 554,5 → 2688,1 img/s từ 0 lên 8 worker, cả 5 setting parity-identical, và **lần đo
  đầu đã bị thay thế** (xem hàng G8). Đề xuất đổi sang `workers=8`, kèm cảnh báo đường cong chưa phẳng và
  ngân sách thread. Chi tiết ở `PERFORMANCE.vi.md`, sinh tự động từ `perf_raw.jsonl`. **Tám lượt rerun vẫn không có timing** — chúng chạy khi bốn lane khác đang hoạt động,
  nên wall-time của chúng vẫn ghi *"measured under concurrent load — not a baseline"* và `throughput`
  `not_measured`. Một phép đo producer stage không biến chúng thành baseline.
- **Đã sửa: artifact nay tự chứng minh được policy BLAS của nó.** `policy_observed` được ghi vào
  **rank payload** (`runner.py`, dict `_atomic_json(rank_path, {...})`) và vào **summary**, chứa
  `status`, `source` (`rank_pass` / `backfill` / `backfill+rank_pass` / `not_recorded`) và record
  readback thật (`requested`, `effective`, `previous`, `setter_found`, `getter_found`, `verified`,
  `source`). Nó **cố ý nằm ngoài identity đã hash**: identity được dựng trước khi scoring chạy, nên đưa
  readback vào đó sẽ khiến `run_id` phụ thuộc lịch sử process — đúng lỗi mà comment `runner.py:1481-1486`
  ghi lại và đã từng xảy ra. Vì vậy **không nâng schema v3, không đổi `run_id`**: tám artifact của phiên
  này đọc `not_recorded (predates policy_observed)`, trung thực và không bị regroup.
  Một lỗi ordering trong chính bản sửa này đã được tìm và sửa: `_reset_effective_blas_pin()` nằm sau
  nhánh backfill nên một lượt resume+backfill **đã pin thật rồi bị xoá** và báo `not_recorded`.
  Nay readback của backfill được snapshot trước khi reset, và `source` nói rõ phase nào quan sát được.
  Xem `BLAS_IDENTITY_FINDINGS.md`.
- **BLOCKED (14 hàng), lý do cụ thể trong `MODEL_MATRIX.csv`:** thiếu byte checkpoint (conesep n0.5/n0.8,
  habit n0.5/n0.8, intent n0.5/n0.8, tme cirr_0); adapter **đã có** cho pair ×2, clvc_net, dcnet, combiner_rn50x4_noft,
  clip4cir_rn50x4_fullft và tgcir — tất cả báo `ready=False` với lý do riêng trong `MODEL_MATRIX.csv`.
  Điều còn thiếu là **byte checkpoint hoặc interpreter được khai báo**, không phải file adapter.
  `dcnet` đặc biệt: `CENet` import được và checkpoint load **strict 518/518 tensor** dưới torch 2.7.1;
  `weights_only` TypeError là do torch 1.4.0 thiếu tham số, không phải checkpoint hỏng. Chỉ `sprc`
  là model duy nhất **không có** adapter (deferred).
- **DEFERRED (3 hàng):** `csmcir`, `sprc` — theo yêu cầu người dùng.
- **`airknow`:** adapter đã có và contract score/rank **chỉ verify trên input tổng hợp**; **chưa có
  forward GPU**, layout target chưa xác nhận với checkpoint thật, `status()` trả `ready=False`
  ("no interpreter environment resolved"), cả hai variant `NO_CLEAN_CHECKPOINT`.
- **`limn`:** lượt mới coverage **6.001/6.003** (`0.9996668332500417`), không phải 100%. Hai query
  không chấm được là `7:3038`, `7:3790` — lý do *"category unresolved has no adapter embedding scope"*
  ghi trong `query_failures` của artifact, không phải miss. Lượt trước chấm 5.884 nên coverage tăng,
  nhưng hai cohort khác nhau và **không được so trực tiếp** (§3.2). Không lượt nào được phép lặng lẽ
  đổi mẫu số.
- **Lỗi ordering có sẵn, không sửa:** `test_circus_cli.py::test_cli_3d_full_gallery_resume_and_gallery_invalidation`
  fail `'REUSED' != 'COMPLETED'` khi `test_composition_topk_runner.py` chạy trước. Đã chứng minh
  **có trước** thay đổi này (stash hai file production vẫn fail). Thuộc defect #5 của sweep
  `20261108T212500`, là lỗi isolation của test — không thuộc phạm vi lane này.

## 8. Commit, push, dịch vụ

**Danh sách commit có thẩm quyền là lịch sử git, không phải bảng trong file này.** Một file được
track không thể đặt tên commit chứa chính nó — đó đúng là lý do `FINAL_PC_RELEASE_VERDICT.md` dùng
field `FINAL_HEAD_SHA = git rev-parse HEAD` thay vì chép SHA vào văn bản. Xem:

```bash
git log --oneline 80842661..HEAD       # toàn bộ lượt này
git log --oneline -1                   # commit hiện tại
git rev-parse HEAD                     # SHA đầy đủ
```

Các mốc của lượt này, để đối chiếu với `git log` (đây là mô tả nội dung, không phải nguồn có thẩm
quyền về SHA):

| nội dung | commit |
|---|---|
| sáu lỗi evaluator + test (19 failed/2 passed → 25 passed) | `7ef4cc4` |
| CLIP4Cir: member theo `role` từ registry + test | `f6796b3` |
| bằng chứng lane A + `RERUN_IMPACT.md` + sweep bị huỷ | `97f9d35` |
| frontend light workspace (15 file) | `87ff419` |
| adapter Air-Know (ghi rõ chưa sẵn sàng) | `20a406a` |
| evidence pack: matrix, decisions, gates, index, báo cáo | `20b83e5` |
| rerun tám model dưới policy BLAS được ghi lại + export Top-100 | `92e337c` |
| ảnh chụp frontend làm bằng chứng G9/G10 | `2a832bc` |
| sweep serial tại `2a832bc` + tiến `TESTED_SOURCE_SHA` | `a2d9c45` |
| sửa ba assertion test bị drift (test-only) | `3d576bf` |
| cân lại G3/G11 + làm mới matrix/decisions theo kết quả rerun | *(commit này — xem `git log -1`)* |

Các SHA cũ hơn (`80842661`, `1129c58`, `c2d8cdeb`) chỉ còn xuất hiện như **mốc lịch sử** trong báo
cáo này và trong `SWEEP.md`; chúng không mô tả cây hiện tại.

Dịch vụ: `http://127.0.0.1:5173` (200) và `http://127.0.0.1:8000` (200); asset đang phục vụ
`index-9MwV5MAk.js` khớp đúng `dist/assets/`.

Push và trạng thái remote: xem mục cuối của `GATE_RESULTS.json` (G11) và xác nhận
`git rev-parse HEAD` == `git rev-parse origin/research/cir-failure-workbench` sau khi push.

**Không** claim "tất cả đã sạch": G1, G2 chưa chạy (không có differential đầu vào/forward); G5 PASS chỉ
trong phạm vi tám model đã rerun, không phải toàn cohort (14 hàng BLOCKED, 3 DEFERRED); G8 PASS chỉ cho
trục decode của một model family, **không** phải baseline end-to-end, và tám lượt rerun vẫn không có
timing; G11 PASS **không** có nghĩa suite xanh — 18 failure đã biết vẫn được ghi nguyên văn.
