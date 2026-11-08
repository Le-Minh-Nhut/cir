# Defects D1–D3: sửa, chứng minh, và hai mục ghi nhận nhưng KHÔNG áp dụng

Ngày 2026-11-08. Nhánh `research/cir-failure-workbench`.

## D1 — `prefetch_rows` không chặn đúng như docstring nói (ĐÃ SỬA)

### Lỗi

`ThreadPoolExecutor.map` **submit toàn bộ future ngay lập tức** — cả ~2.327 lô. `islice` chỉ giới
hạn số *kết quả kéo về mỗi lần gọi*, không giới hạn số tensor đã decode đang nằm trong bộ nhớ.
Decode đo được 2.089 img/s còn forward ~84 img/s, nên pool chạy trước ~25× và giữ các lô đã xong
trong future của chúng: khoảng **45 GB** cho toàn gallery — đúng cái blow-up mà docstring nói là
đã ngăn.

### Sửa

Cửa sổ trượt thật: chỉ `workers * 2` future đang mở tại một thời điểm; yield future cũ nhất, rồi
submit thêm đúng một lô.

### Test và chứng minh ngược

`test_parallel_decode_is_bounded` trước đây ghi `live["peak"]` mà **không assert** — pass ở mọi độ
sâu. Nay nó assert `peak <= workers * 2` và đếm số lô "đã submit nhưng chưa consume" (đó mới là thứ
chiếm bộ nhớ: future đã xong vẫn giữ tensor).

Chứng minh ngược — khôi phục thân `pool.map` cũ rồi chạy lại (raw: `d1_revert_proof.txt`):

```
FAILED test_parallel_decode_is_bounded
AssertionError: prefetch_rows held 12 batches in flight for 64 batches;
a bounded pipeline must never exceed workers*2 = 8
assert 12 <= (4 * 2)
```

12 > 8: test bắt đúng lỗi. Bản đã sửa: pass.

## D2 — nhóm query phụ thuộc thread count, thiếu trong `identity` (ĐÃ SỬA)

### Lỗi

`rank_in_query_chunks` đặt `size = ceil(total / workers)`. Call site cũ truyền `vectors` tối đa
`batch_size = 32`, nên ở mặc định 8 thread nhóm là **4 query/lần gọi `adapter.score`**, còn ở
`WORKBENCH_SCORE_THREADS=1` là **32**. Đó là cách gộp phép cộng khác nhau trong đường publishable,
điều khiển bằng env var, mà `run_id` không đổi — cùng loại lỗi với F1.

### Sửa

Kích thước nhóm do **caller** quyết định (`chunk_size`), không phải thread count. Call site truyền
`batch_size`, và việc gộp nhóm của vòng lặp được giữ nguyên hình dạng `batch_size`; song song hoá
nằm **giữa các nhóm** (`_rank_groups_concurrently`). Nên `WORKBENCH_SCORE_THREADS` chỉ đổi thời
gian, không bao giờ đổi một con số publishable.

### Chứng minh ở đúng độ chi tiết production

Chạy đường đã ship (`_rank_groups_concurrently`, `chunk_size=32`, threads=8) trên **toàn bộ
6003 query × 3 condition**, so từng rank với artifact `habit_fiq_n02.json` đã publish:

| condition | queries | thời gian | q/s | mismatch |
|---|---|---|---|---|
| multimodal | 6003 | 183.0 s | 32.8 | **0** |
| text_only_black_image | 6003 | 196.0 s | 30.6 | **0** |
| image_only_empty_text | 6003 | 184.3 s | 32.6 | **0** |
| **tổng** | **18,009** | **563 s (9.4 phút)** | — | **0** |

**PASS — đường đã ship tái lập chính xác rank đã publish trên cả 18,009 cặp.**

Chi phí thật của gate này: **9.4 phút, chạy single-core theo thiết kế** (threads=8 nhưng mỗi lần
gọi `einsum` vẫn là một core, và `OPENBLAS_NUM_THREADS=1`). Lần thử đầu mất ~12 phút rồi bị huỷ vì
nó tính lại cả baseline serial ~20.8 phút/condition (≈3 giờ) — baseline đó không cần thiết vì
artifact đã publish **chính là** kết quả serial.

Test: `test_group_size_does_not_depend_on_thread_count` — chứng minh ngược bằng cách khôi phục
`size = ceil(total/workers)`: thất bại với group size 64 thay vì 16.

## D3 — xung đột thread BLAS (ĐÃ SỬA)

### Lỗi

Song song theo query nhân với BLAS đa luồng sẽ oversubscribe: 8 nhóm × 10 thread = 80 thread trên
20 core. `composition_eval.py` đặt `OPENBLAS_NUM_THREADS`/`OMP_NUM_THREADS` = 10.

### ĐÍNH CHÍNH phân loại — thay hẳn cách chia cũ

Cách chia cũ trong tài liệu này ("4 `@` vs 4 einsum", và "`encoder`/`limn` thừa hưởng
`CompositionAdapter.score`") là **SAI** và đã bị thay thế:

- Cả `model_encoder.py` và `model_limn.py` đều **override** `score()`, không thừa hưởng bản nền.
  ENCODER (`model_encoder.py:225-238`) chia gallery theo block 8192 và làm `queries @ gallery.T` —
  **gemm theo batch**. LIMN (`model_limn.py:370-400`) chấm **từng row** (`query @ gallery[...].T`) —
  **gemv với M=1**. Hai thứ này chia phép rút gọn theo trục K khác nhau.
- Ba adapter `optimize=True` cũng **không** trơ: `np.einsum(..., optimize=True)` hạ phép tính xuống
  BLAS gemm và **có** nhạy với số thread.

**Phân loại đúng — 5 GEMM-path vs 4 naive-einsum:**

| họ | adapter | vì sao |
|---|---|---|
| **GEMM-path (5)** | encoder, limn, csmcir, ptha_mtst, conesep | `@` (encoder: batch gemm; limn: gemv) hoặc `np.einsum(..., optimize=True)` → hạ xuống gemm |
| **naive-einsum (4)** | habit, intent, hint, tme | `np.einsum("nd,gqd->ngq")` **không** `optimize` → không có đường BLAS nào |

### Cơ chế ĐÚNG (sửa lại, không phải gemv-vs-gemm)

Với group shape `batch_size=32`, phép gọi đã là gemm từ trước, nên cơ chế **không phải** gemv-vs-gemm
(đó là cơ chế batch-1 của F8). Cơ chế thật là: **OpenBLAS dispatch một kernel đa luồng và chia lại
phép rút gọn theo trục K**. Đó cũng là lý do **bản thân delta thay đổi theo số thread** (3.576e-07 ở
1/2/4/8 thread so với 2.384e-07 ở 10) chứ không phải một giá trị cố định — và là lý do `np.dot`
giữ delta 0.0 ở 1/2/4/8 thread rồi mới nhảy 6.104e-05 ở 20 thread.

**Cảnh báo về bằng chứng D3 cũ:** phép đo D3 ban đầu (`d3_blas_collision_proof.py`) khởi tạo
`CompositionAdapter.__new__` và gọi **score của lớp nền** — tức là không đo đúng scorer của encoder
hay limn. Số của nó vẫn dùng được cho câu hỏi *tốc độ/oversubscription* (vì nó là một phép `@` thật),
nhưng **không** dùng được để khẳng định ảnh hưởng rank của encoder/limn. Bằng chứng thay thế là
`d3b_encoder_aggregate_scan` (encode thật) và `conesep` qua `Adapter.score` thật.

### Sửa

Khi `score_threads > 1`, ép BLAS về 1 thread cho **toàn bộ đoạn song song** — bật/tắt **một lần**,
không phải trong từng worker.

**Đây là lỗi tôi tự tạo rồi tự bắt:** phiên bản đầu toggle trong từng worker qua
`openblas_set_num_threads`. Chạy thật thì **rank sai ở 4 và 8 thread** — một worker khôi phục số
thread cũ trong khi worker khác đang tính. Đó là race, và nó biểu hiện đúng như một lỗi số học.
Đã sửa thành một toggle duy nhất bao quanh cả đoạn.

### Chứng minh (scorer `@` thật, gallery 74,381 × 512, BLAS ngoài = 10)

| cấu hình | q/s | so với serial | rank giống |
|---|---|---|---|
| serial, BLAS=1 | 269.8 | 1.00× | chuẩn |
| serial, BLAS=10 | 355.3 | 1.32× | — |
| threads=2 (BLAS ép 1) | 508.7 | 1.89× | ✅ |
| threads=4 (BLAS ép 1) | 707.4 | 2.62× | ✅ |
| **threads=8 (BLAS ép 1)** | **1097.8** | **4.07×** | ✅ |

Nhanh hơn cả serial-BLAS-10 (355 → 1098 q/s) và **không đổi một rank nào**. Nên không cần
auto-disable cho adapter dùng `@`.

Test: `test_concurrent_groups_are_identical_to_serial_for_matmul` (rank giống ở 2/4/8 thread) và
`test_concurrent_group_failure_does_not_abort_siblings`.

## D3b — pin BLAS chỉ ở nhánh concurrent (ĐÃ SỬA; cùng lớp lỗi với F1)

### Lỗi

`_rank_groups_concurrently` đặt `_blas_thread_limit(1)` **sau** nhánh `if threads <= 1`, nên nhánh
serial chạy với BLAS bằng giá trị env — và `composition_eval.py` đặt mặc định **10**.

Hệ quả: với scorer dùng `@` (encoder, limn, `CompositionAdapter.score` nền),
`WORKBENCH_SCORE_THREADS=1` tính với **BLAS=10** còn `=8` tính với **BLAS=1**. Số thread BLAS
**không trung tính về số học** — chính `ein_threads.txt` đo delta khác nhau (3.576e-07 ở 1 thread so
với 2.384e-07 ở 10). Nên knob này có thể đổi số publishable với `run_id` không đổi. D2 trên HABIT
không phát hiện được vì `np.einsum("nd,gqd->ngq")` không đi qua BLAS.

### Sửa

Hoist `_blas_thread_limit(1)` lên **trước** cả hai nhánh, bọc toàn bộ hàm trong `try/finally`. Một
toggle duy nhất cho mỗi lần gọi — vẫn nằm ngoài worker, vì toggle trong worker đã đo được là gây
race làm đổi rank ở 4 và 8 thread.

### Chứng minh với scorer `@` thật (gallery 74,381 × 512, env BLAS=10)

| cấu hình | q/s | rank giống threads=1 |
|---|---|---|
| threads=1 (BLAS ép 1) | 210.9 | chuẩn |
| serial KHÔNG pin (BLAS=10) | 301.8 | — |
| threads=2 (BLAS ép 1) | 310.6 | ✅ |
| threads=4 (BLAS ép 1) | 549.9 | ✅ |
| threads=8 (BLAS ép 1) | 556.5 | ✅ |

**PASS** — rank giống hệt ở 1/2/4/8. Đây là điều D2 trên HABIT không thể chứng minh.

### Bằng chứng còn thiếu: BLAS=1 và BLAS=10 có thật sự khác nhau?

Trước đó bảng chỉ so thread 2/4/8 với baseline BLAS=1, và để `—` ở dòng serial-BLAS=10 — tức là
**chưa chứng minh** vế duy nhất còn thiếu: hai mức BLAS có khác nhau không. Đo lại (cùng scorer `@`,
cùng gallery, `d3_serial_blas_compare.py`):

```
serial BLAS=1 vs serial BLAS=10: queries=512
  ranks_identical = False
  queries_changed_rank = 11
  max_abs_drank = 1
  score_max_abs_delta = 4.172e-07
```

Ví dụ: query 77 rank 38550 → 38549; 112 39674 → 39673; 127 7510 → 7509; 272 58976 → 58977.

**Kết luận: pin là LOAD-BEARING, không phải belt-and-braces.** Với scorer dùng `@`, hai mức BLAS
cho **rank khác nhau** (11/512 query, max |Δrank| 1). Nghĩa là pin không phải "cho chắc" — nó là
**thứ duy nhất** khiến knob `WORKBENCH_SCORE_THREADS` hiện không thể đổi số publishable. Nếu bỏ pin,
kết quả publish sẽ phụ thuộc biến môi trường.

Đây là lựa chọn ngữ nghĩa, phải ghi lại chứ không được giả định: hiện code **ép một giá trị cố định
(BLAS=1)** cho mọi cấu hình, nên kết quả tái lập được — nhưng "tái lập được" ở đây nghĩa là "tái lập
được dưới policy BLAS=1", không phải "giống hệt cách tính trước đây". Với adapter `einsum`
(habit/intent/hint/tme) điều này vô hại vì chúng không đi qua BLAS — đã đo: `OPENBLAS_NUM_THREADS`
1 vs 10 không đổi throughput (5.6–6.5 q/s). Với 7 adapter còn lại thì đây là **thay đổi so với
cách tính cũ** và phải coi như một policy mới, không phải một tối ưu thuần tuý.

### Đánh đổi, nói thẳng

Nhánh serial **chậm đi 30%** khi bị pin BLAS=1: 301.8 → 210.9 q/s. Đó là giá của tính tái lập được.
Vẫn nên pin: kết quả publish phải tái lập được bất kể knob, kể cả khi đường serial chậm hơn. Ở
threads ≥ 4, song song theo query bù lại nhiều hơn phần mất (549.9 > 301.8).

### Test và chứng minh ngược

`test_both_serial_and_concurrent_branches_pin_blas` (assert có pin 1, restore đúng giá trị cũ, và
**đúng một lần** mỗi lần gọi) + `test_blas_pin_applies_with_a_single_group`.

Khôi phục lỗi (pin chỉ ở nhánh concurrent):

```
FAILED test_both_serial_and_concurrent_branches_pin_blas
  assert 1 in calls, "threads=1 never pinned BLAS to 1: []"
  assert 1 in []
FAILED test_blas_pin_applies_with_a_single_group
  IndexError: list index out of range
```

`calls == []` nghĩa là nhánh serial **không pin lần nào** — đúng lỗi được mô tả.

## Lưu trữ bằng chứng

Mọi script chứng minh và output thô đều nằm trong thư mục này (không chỉ `/tmp`):

| file | nội dung |
|---|---|
| `d2_gate_full_6003x3.py` / `.txt` / `.json` | D2: 6003×3, 0 mismatch |
| `d2_gate_first_attempt_superseded.py` | lần thử đầu (~3 h), bị huỷ — giữ để đối chiếu |
| `d2_result.json`, `d2_revert_proof.txt` | kết quả D2 + bằng chứng ngược D2 (group size 64≠16) |
| `d1_revert_proof.txt` | D1: 12 lô in-flight > 8 |
| `d3b_hoist_revert_proof.txt` | D3b: `calls == []`, serial không pin |
| `d3_serial_blas_compare.py` / `.txt` / `.json` | BLAS=1 vs 10: 11/512 rank đổi — pin là load-bearing |
| `d3_blas_collision_proof.py` / `.txt` | D3: sau khi sửa race (rank giống, 4.07×). Lưu ý: script in nhãn "BLAS forced 1 inside" từ phiên bản toggle-trong-worker; ở lần chạy này pin đã là một toggle bao ngoài — nhãn cũ, số liệu đúng |
| `d3_hoist_proof.py` / `.txt` / `.json` | D3b: rank giống ở 1/2/4/8 |
| `batch.txt`, `ein_threads.txt`, `vdecode.txt`, `query.txt`, `cohort_full.txt` | số đo gốc |
| `eq_fast.txt`, `eq_opt.txt`, `metric_decision_optimize_true.txt`, `metric_gate_*` | bằng chứng từ chối `optimize=True` |

## D3c — HỆ QUẢ METRIC của BLAS pin: nDCG CÓ đổi

Câu hỏi quyết định: pin BLAS có đổi **metric công bố** hay chỉ đổi rank từng query? Đã đo trên dữ
liệu thật.

### LIMN — không đổi gì

LIMN `score()` chấm **từng query một** (`query @ gallery[start:stop].T` trong vòng lặp theo row),
nên mỗi row là một phép nhân vector với ma trận — BLAS không có chiều M để đổi. Đo 600 query thật ×
3 condition, `d3b_limn_aggregate_scan.{py,txt,json}`:

| condition | rank đổi | max Δrank | vượt K10 | vượt K50 | ΔR@10 | ΔR@50 | ΔnDCG |
|---|---|---|---|---|---|---|---|
| multimodal | **0** | 0 | 0 | 0 | 0 | 0 | 0 |
| text_only_black_image | **0** | 0 | 0 | 0 | 0 | 0 | 0 |
| image_only_empty_text | **0** | 0 | 0 | 0 | 0 | 0 | 0 |

### ENCODER — nDCG ĐỔI

ENCORER là adapter `@` chấm theo **batch** (`queries @ gallery.T` trên block 8192 row) — đúng hình
dạng mà kernel phụ thuộc chiều M. Encode thật: gallery 74381 ảnh trong 1.8 phút.

> **ĐÍNH CHÍNH — bảng 3 dòng cũ đã bị thay.** Bảng trước đây báo ba dòng **giống hệt nhau** (mỗi
> dòng 1 query đổi rank, ΔnDCG 2.181e-05). Đó **không** phải ba phép đo độc lập: script cũ truyền
> **cùng một ảnh tham chiếu và cùng một text gốc ở mọi vòng lặp** và **không gọi
> `adapt_query_inputs`**, nên cả ba condition chấm **cùng một input**. Ba dòng đó là **một** phép đo.
>
> `run_conesep` trong cùng script **không** mắc lỗi này (nó đọc `query_<cond>_global.npy` do runner
> tạo sau ablation), nên dòng conesep là per-condition thật.

Bản sửa dùng đúng helper production `runner._stage_query_path` (nguồn duy nhất của
`adapt_query_inputs` + quy tắc staging) để áp intervention **trước** khi encode, cộng assertion kiểm
tra intervention đã thực sự xảy ra (image_only phải rỗng text; text_only phải stage ảnh đen;
multimodal phải truyền ảnh gốc). Kết quả **ba condition khác nhau thật** (`blas_encoder_v2.txt`):

| condition | dscore | query đổi rank | max Δrank | vượt K=10 | vượt K=50 | ΔR@10 | ΔR@50 | **ΔnDCG** |
|---|---|---|---|---|---|---|---|---|
| multimodal | 2.38e-07 | 2 | 2 | **0** | 0 | **0** | **0** | 3.42e-06 |
| text_only_black_image | 2.38e-07 | 4 | 2 | **0** | 0 | **0** | **0** | 1.30e-08 |
| image_only_empty_text | 2.09e-07 | 1 | 1 | **0** | 0 | **0** | **0** | **2.18e-05** |

**Phát hiện phụ, và nó đáng ghi hơn cả con số:** chạy hai lần cùng script, condition **multimodal**
cho ΔnDCG **8.33e-05** (lần 1) và **3.42e-06** (lần 2) — khác nhau **24 lần**; text_only đổi từ 3
sang 4 query. Nghĩa là phía **BLAS=10 không tất định giữa các lần chạy**, không chỉ giữa các mức
thread. Đây chính là lý do phải pin: nếu không pin, cùng một máy, cùng một input, hai lần chạy cho
nDCG khác nhau. Nên **không** được đọc các con số ΔnDCG ở trên như một hằng số vật lý — chúng là
**bậc độ lớn** của độ nhạy (1e-8 … 8e-5), không phải một giá trị tái lập được.

**Đúng condition, đúng query** (đọc thẳng từ `blas_encoder_v2.json`):

- `image_only_empty_text`: query **`7:5555`** đổi rank **2 → 3**. Vì 2 và 3 **đều nằm trong top-10**,
  R@10 và R@50 không đổi. Nhưng nDCG dùng `1/log2(rank+1)`: 1/log2(3) = 0.6309 so với 1/log2(4) =
  0.5. Một query đổi từ hạng 2 xuống 3 làm nDCG của nó giảm 0.1309, chia 6003 query ⇒ cỡ **1e-5**.
  Đây là nguồn của con số ΔnDCG lớn nhất (2.18e-05).
- `multimodal`: query **`7:4506`** (18 → 16) và **`7:4764`** (16 → 14) — cả hai ở hạng 14–18, tức là
  **ngoài K=10 và trong K=50**, nên không vượt biên nào (đo được: 0 vượt K=10, 0 vượt K=50) và
  R@10/R@50 giữ nguyên; đóng góp nDCG nhỏ nhưng không bằng 0 (3.42e-06).
- `text_only_black_image`: **`7:2651`** (10588 → 10587), `7:4506` (790 → 788), `7:4764` (788 → 786),
  `7:5555` (4943 → 4944) — toàn hạng sâu, ΔnDCG 1.30e-08.

Lưu ý `7:5555` xuất hiện ở **hai** condition với hai mức rank khác nhau (2→3 ở image_only, và
4943→4944 ở text_only) — cùng một query, hai không gian điểm khác nhau.

### Phán quyết

**R@10 và R@50 giống hệt. nDCG ĐỔI, và vượt ngưỡng 1e-6** (2.18e-05, gấp ~22 lần ngưỡng).

Nên không thể nói "pin chỉ đổi rank từng query, metric tổng không đổi". Phát biểu đúng là:

- **R@10 / R@50**: không đổi trên dữ liệu này (0 query vượt biên K=10/K=50).
- **nDCG**: **đổi**, ở **cả ba condition**, trong dải **1.3e-08 … 2.18e-05**; lớn nhất là
  image_only_empty_text (2.18e-05). Nguyên nhân: query rơi từ hạng 2 xuống 3 — vẫn trong top-10
  nhưng làm nặng hình phạt log. Con số **không tái lập giữa hai lần chạy** (multimodal: 8.33e-05 so
  với 3.42e-06), nên chỉ nên đọc như bậc độ lớn.
- **0 query vượt biên K=10/K=50/K=100** là **quan sát trên mẫu này, không phải bảo đảm** — query nào
  ở đúng hạng 9/10/11 sẽ vượt biên chỉ vì 1 ULP.
- **encoder** (và bất kỳ adapter `@` chấm theo batch) chịu ảnh hưởng; **limn** thì không; **ptha /
  csmcir** dùng `einsum` nên không đi qua BLAS.

**Hệ quả:** dòng `encoder` đã publish phải được coi là **tính dưới một policy BLAS khác** với code
hiện tại. nDCG của nó **không tái lập chính xác** dưới policy BLAS=1 (lệch 2.18e-05). R@10/R@50 thì
tái lập được.

**Lỗi trong chính lần đo này, phải nói rõ:** script cũ **bỏ qua `adapt_query_inputs`** — tức là không áp dụng condition transform
(`text_only_black_image` vẫn dùng ảnh gốc thay vì ảnh đen, v.v.). Vì vậy **giá trị tuyệt đối** trong
bảng trên (R@10 ≈ 26.37) **không** phải số published của encoder (31.02), và cột `published_*` trong
JSON không dùng để so trực tiếp được. Phần **còn giá trị** là phép so **BLAS=1 vs BLAS=10 trên cùng
một tập vector**: 1 query đổi rank 2→3 và ΔnDCG = 2.18e-05. Đó là câu trả lời cho câu hỏi đặt ra,
và nó không phụ thuộc việc condition transform có được áp dụng hay không — vì cả hai vế đều dùng
cùng một vector.

## D3d — Policy BLAS của từng run đã publish, và lỗ resume

> Câu hỏi "upstream không đặt gì, sao mình lại đặt?" được trả lời đầy đủ trong
> **`BLAS-POLICY.vi.md`** cùng thư mục: upstream dùng `all_queries.dot(all_imgs.T)` trần
> (`ENCODER/test.py:62,130,215`; `LIMN/LIMN/test.py:74,146,240`) và không eval path nào đặt BLAS
> thread, nên số của họ phụ thuộc máy. Chúng ta buộc phải chọn một giá trị để artifact của chính
> mình so sánh và resume được.

### Mốc thời gian quyết định

Khối BLAS env trong `composition_eval.py` (`_BLAS_THREADS` + vòng `os.environ.setdefault`) được
đưa vào bởi commit **`3d0e2b89fb3a86d5eedd8fa7f122d8c1a9dcb502`, 2026-11-08 17:55:36 +0700**
(`git log -S'_BLAS_THREADS'`, không dùng mtime). Pin `_blas_thread_limit` trong `runner.py` nằm ở
commit **`bf6a448`** (trước đó `git show 3d0e2b8:workbench/composition/runner.py` không có chuỗi
này). Bảng policy bên dưới vẫn giữ nguyên vì mọi run trong đó kết thúc trước `bf6a448`.

### Bảng policy từng run

Mtime của rank artifact = thời điểm run kết thúc. **Không run nào chạy sau 17:55:36**, nên **không
run nào từng thấy khối env BLAS**, và pin thì chưa từng tồn tại khi chúng chạy.

| run | kết thúc | khối env? | pin? | policy hiệu dụng |
|---|---|---|---|---|
| encoder `6a528e8` | 11-08 00:00 | không | không | **unknown** |
| limn `10173dc` | 11-08 04:51 | không | không | **unknown** |
| habit `e0dd04a` | 11-08 05:00 | không | không | **unknown** |
| tme `ad47731` | 11-08 06:14 | không | không | **unknown** |
| ptha `315dd59` | 11-08 06:43 | không | không | **unknown** |
| habit `497c25c` | 11-08 15:21 | không | không | **unknown** |
| intent `37ee72a` | 11-08 16:38 | không | không | **unknown** |
| hint `4bc9893` | 11-08 17:07 | không | không | **unknown** |
| conesep `30f6de4` | 11-08 17:42 | không | không | **unknown** |

**"unknown" ở đây là kết luận, không phải thiếu sót.** `identity` của các artifact đó có 61 khoá và
**0** khoá thread/blas/omp — artifact không ghi lại policy, và khối env chưa tồn tại. Suy ra "10" vì
code hôm nay sẽ đặt 10 là **suy diễn sai**: khi ấy giá trị là do shell/ambient quyết định.

Hệ quả trực tiếp: **mệnh đề "pin=10 sẽ tái lập mọi artifact lịch sử" là SAI.** Khi không biết giá trị
cũ thì không giá trị pin nào — kể cả 10 — tái lập được chúng.

### Lỗ resume đã đóng

`_rank_groups_concurrently` pin BLAS trong `runner.py`, mà `_module_source_hash` **không bao giờ**
hash file này. Nên trước khi sửa, `fingerprint` không đổi khi pin đổi, và một rank file ghi dưới
BLAS=10 vẫn khớp fingerprint dưới code BLAS=1 → `--resume` tái dùng im lặng. Cùng dạng với lỗi LIMN
scope.

Đã thêm một trường tường minh vào `identity`:

```python
"blas_threads_pinned": BLAS_THREADS_PINNED,   # = 1
```

Kèm hằng `BLAS_THREADS_PINNED = 1` dùng chung cho cả trường identity **và** lời gọi
`_blas_thread_limit(...)`, nên nhãn và hành vi không thể lệch nhau. Không hash toàn bộ `runner.py` —
làm vậy sẽ đổi run id vô cớ cho mọi sửa đổi không liên quan.

**Chứng minh resume bị chặn** (`blas_resume_proof.py`):

```
pinned=1   fingerprint: 95bb975daaac39028e77dace
absent     fingerprint: 05dafbbbfed84c2dfc3236a3
pinned=10  fingerprint: 8f7997872334f49defcd3855
pin != absent : True
pin != 10     : True

published encoder identity has blas_threads_pinned: False
published encoder fingerprint (as recorded): 6a528e877c32872...
new-code fingerprint would be:               95bb975daaac390...
=> resume cannot reuse it
```

**Test** (3 test mới, tổng 27 pass):
- `test_blas_policy_is_a_recorded_identity_field` — trường phải có trong identity.
- `test_changing_the_blas_policy_changes_the_fingerprint` — pin≠absent và pin1≠pin10.
- `test_both_branches_execute_under_the_pinned_value` — cả nhánh serial và concurrent đều chạy dưới
  **đúng** `BLAS_THREADS_PINNED` và đều restore.

## Ghi nhận, KHÔNG áp dụng

### G1 — `require_batch_fits` under-reserve

Projection hiện tại: `batch_size * per_image_bytes` so với `total * 0.85`, **bỏ qua floor cố định
~3.96 GiB** (trọng số + context) và bộ nhớ đã cư trú.

Số đo thật (slice 2000 ảnh, `habit`, threads=20, fp32):

| batch | peak alloc | peak reserved |
|---|---|---|
| 64 | 3,484 MiB | 3,958 MiB |
| 128 | 4,350 MiB | 5,066 MiB |
| 256 | 6,082 MiB | 7,306 MiB |
| 512 | 9,544 MiB | 11,788 MiB |

Hồi quy tuyến tính: **~3.96 GiB cố định + ~17.5 MiB/ảnh** (marginal). Guard hiện tại dùng
14 MiB/ảnh và không trừ floor, nên ở **bs ≈ 989** nó sẽ duyệt một batch thực tế cần **~20.1 GiB** —
vượt 16.3 GiB card. Nói cách khác guard có thể duyệt một batch sẽ OOM.

Đề xuất (chưa áp dụng):
```python
projected = FIXED_FLOOR_BYTES + batch_size * MARGINAL_BYTES_PER_IMAGE   # 3.96 GiB + 17.5 MiB
```
hoặc dùng `torch.cuda.mem_get_info()` lấy free bytes trừ trọng số đã cư trú. Nên làm cả hai: floor
cho trường hợp tĩnh, `mem_get_info` cho trường hợp đã có run khác trên card.

**Quan trọng:** guard hiện chỉ được nối trong `model_habit.py`. `--batch-size` **không được bảo vệ**
ở bảy adapter còn lại.

### G2 — `measurement_identity` bỏ sót hai hash

`workbench/backend/composition.py` (~520–530) không có `scoring_module_source_hash` và
`adapter_module_source_hash` trong `measurement_identity`. Hệ quả: hai run chỉ khác nhau ở đường
scoring sẽ hash ra cùng `group_id` và bị gắn nhãn `duplicate` — **sai**.

Đề xuất một dòng (file do agent khác sở hữu, KHÔNG sửa):
```python
"scoring_module_source_hash": identity.get("scoring_module_source_hash"),
"adapter_module_source_hash": identity.get("adapter_module_source_hash"),
```

## Test

`workbench/tests/test_throughput_knobs.py` — **22 test, pass**. Ba test mới khoá đúng ba lỗi:
boundedness có assert, group size độc lập thread count, và `@`-scorer bit-identical khi song song.
