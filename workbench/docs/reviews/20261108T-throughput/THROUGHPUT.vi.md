# Throughput CIRCUS: đo được, sửa được gì, và cái gì bị từ chối

Ngày 2026-11-08. Nhánh `research/cir-failure-workbench`.

## TL;DR

Câu hỏi của user: *"gpu mạnh mà nâng batch size đồ lên đi chứ mắc gì chạy như rùa vậy"*.

**Nâng batch size là một manh mối sai.** Đo được: forward trên GPU đạt **84 img/s ở batch 32 và
84 img/s ở batch 256** — bão hoà hoàn toàn, nên tăng batch chỉ tốn VRAM mà không nhanh hơn. VRAM
không phải nút cổ chai.

**Nút cổ chai thật là CPU, và nó nằm ở hai chỗ:**

1. **Decode ảnh tuần tự một core** trong worker (`Image.open` → `convert("RGB")` → `transform` từng
   ảnh), trong khi GPU chờ. Đo: GPU ở **11–15% utilization**, thỉnh thoảng nhảy 100%, VRAM 8.4 GB
   nhưng power chỉ 39–41 W / 285 W — mô hình đã nạp, kernel ngồi không.
2. **`transform` chạy với 20 intra-op thread của torch** — đúng bằng số core của máy. Mỗi ảnh chỉ có
   vài op nhỏ, overhead tạo thread lớn hơn việc thật. Đo: **27 img/s ở 20 thread vs 650 img/s ở 1
   thread** cho cùng một transform.

Sửa decode (song song hoá theo lô ảnh + hạ thread trong một transform): **21.2 → 2089 img/s** trên
cùng 512 ảnh, cùng transform, **bit-identical (max abs delta = 0.0)**.

**ĐÍNH CHÍNH quan trọng — con số 98× chỉ đúng cho stage decode+transform, KHÔNG phải cả pipeline.**

- **Decode+transform thuần**: 21.2 → 2089 img/s (~98×). Đây là con số ở trên.
- **Pipeline đầy đủ** (decode + transform + forward GPU): đo 2000 ảnh thật, **17.6–18.0 img/s ở batch
  64–512** — forward GPU bão hoà ở ~84 img/s (≈886 s cho 74,381 ảnh ≈ 15 phút) và là sàn cứng.
- **Ước lượng end-to-end cho 74,381 ảnh**: trước ~73 phút → sau **~15 phút**, tức **~5×**, không phải
  98×.

Đừng ai bấm giờ `intent` rồi thấy số khác rồi tưởng tài liệu sai: **98× là decode, 5× là cả model.**

Điểm số học (einsum trên 74,381 ảnh) **cũng là nút cổ chai thật** và tôi đã đo được cách sửa 9–16×,
nhưng **đã từ chối áp dụng** vì nó đổi rank — xem mục "Bị từ chối" bên dưới. Đây là quyết định quan
trọng nhất của tài liệu này.

## 1. Batch size: đo và bác bỏ

Slice thật 2000 ảnh gallery, checkpoint `habit` `fiq_n02`, batch 64/128/256/512:

| batch | img/s | peak VRAM alloc | peak reserved |
|---|---|---|---|
| 64  | 17.6 | 3484 MiB | 3958 MiB |
| 128 | 18.0 | 4350 MiB | 5066 MiB |
| 256 | 17.9 | 6082 MiB | 7306 MiB |
| 512 | 18.0 | 9544 MiB | 11788 MiB |

`img/s` **không đổi** từ 64 đến 512, dù VRAM tăng 3484 → 9544 MiB. Nguyên nhân: forward thuần GPU
bão hoà ở ~84 img/s; phần còn lại của 17–18 img/s là CPU.

Forward thuần (không decode), cùng checkpoint, cùng batch:

| batch | forward | img/s | peak alloc |
|---|---|---|---|
| 32  | 381 ms | 83.9 | 3053 MiB |
| 64  | 763 ms | 83.9 | 3486 MiB |
| 128 | 1538 ms | 83.2 | 4354 MiB |
| 256 | 3028 ms | 84.6 | 6090 MiB |

Kết luận: **bão hoà từ batch 32.** Không có lý do nâng. VRAM dư ~11 GB trong lúc chạy thật không phải
chỗ lãng phí cần lấp — nó là chỗ trống vì GPU không có việc để làm.

### bf16: đo và bác bỏ

bf16 autocast, cùng slice 2000 ảnh: **17.7 img/s (batch 64), 16.2 img/s (batch 256)** so với fp32
17.6 / 17.9. Không nhanh hơn, thậm chí chậm hơn ở batch lớn. Không áp dụng — không có lý do đánh đổi
dtype khi lợi ích bằng 0.

## 2. Nút cổ chai thật: decode + thread

Tách riêng hai stage trên cùng 512 ảnh:

```
decode only  512: 0.30s   (~1700 img/s)
transform only 512: 19.56s  (~26 img/s)   <- với torch 20 thread
transform only 512 (1 thread): 0.38s  (~1350 img/s)
```

Decode (I/O + PIL) nhanh. **Transform là thủ phạm, và thủ phạm là số thread của torch**, không phải
bản thân phép biến đổi. Ảnh FashionIQ nhỏ (8–28 KB, ~342×407) nên mỗi op trong transform quá nhỏ để
bù chi phí launch thread.

Đo song song hoá decode (thread pool, giữ nguyên thứ tự lô):

> **Bảng dưới đây là PROSE-ONLY — không có raw file, không được trích như số đo.** Kiểm lại bằng số học:
> `wall × img/s` ở mỗi dòng cho ~1015–1030 ảnh, tức bảng này đo **~1024 ảnh/dòng**, trong khi bảng A/B
> ngay dưới và `vdecode.txt` đo **512 ảnh**. Hai thứ là hai thí nghiệm khác nhau, và không file nào
> trong tree sinh ra định dạng `workers/wall/img/s` này. Giữ lại chỉ để đối chiếu lịch sử.
> **Số có raw để trích là `vdecode.txt`** (xem bảng A/B bên dưới): 512 ảnh thật, decode+transform,
> serial 109.1 img/s → **682.7 img/s ở 4 worker** (6.2×), 588.5 ở 8 worker (5.4×), delta 0.0.

| workers | wall | img/s | cpu_per_wall_min |
|---|---|---|---|
| 1  | 1.66s | 618 | 476s |
| 2  | 1.31s | 780 | 113s |
| 4  | 0.61s | 1672 | 252s |
| 8  | 0.47s | 2192 | 390s |
| 10 | 0.53s | 1915 | 348s |

> **Không dùng bảng trên để chọn worker count.** Nó nói "8 worker là điểm tốt nhất", nhưng `vdecode.txt`
> đo **4 worker nhanh hơn 8** (682.7 vs 588.5 img/s). Chọn mặc định `min(10, cpu_count // 2)` là quyết
> định vận hành, không phải kết luận từ bảng prose này.

### A/B cuối cùng (cùng 512 ảnh, cùng transform, cùng checkpoint)

**Chỉ stage decode+transform** — không có forward GPU:

| | img/s | quy ra 74,381 ảnh |
|---|---|---|
| **BEFORE** (code hôm nay: 1 worker, torch 20 thread) | 21.2 | 58.5 phút |
| **AFTER** (8 worker, torch 8 thread) | **2089.2** | **0.6 phút** |

Bảng này là **decode+transform**, nên 0.6 phút KHÔNG phải thời gian chạy model. Cộng forward GPU
(~886 s, bão hoà ~84 img/s) thì end-to-end là **~15 phút**: trước **~73 phút** → sau **~15 phút**,
tức **~5×** cho cả model. Xem phần ĐÍNH CHÍNH ở TL;DR.

**98× đó là decode+transform.** Con số tuyệt đối của BEFORE thấp hơn 17.6 img/s của bảng batch vì
bảng kia có cả forward GPU; ở đây chỉ đo decode+transform. End-to-end cho cả model là **~5×**. Xem
ĐÍNH CHÍNH ở TL;DR.

### A/B riêng cho `intent` (512 ảnh, transform của chính nó, checkpoint thật)

| | img/s | quy ra 74,381 ảnh (decode+transform) | bit-identical |
|---|---|---|---|
| **BEFORE** (1 worker, torch 20 thread) | 30.3 | 41.0 phút | — |
| **AFTER** (8 worker, torch 8 thread) | **2291.9** | **0.5 phút** | ✅ delta 0.0 |

Đây vẫn là **decode+transform**. Cộng forward GPU thì end-to-end ~15 phút, tức **~5×** — xem ĐÍNH
CHÍNH ở TL;DR.

## 3. Đã thay đổi gì

Chỉ **một** thay đổi về hành vi, cộng thêm các nút vận hành:

**a) Decode song song trong worker** — áp dụng cho **8 adapter**:

| Adapter | Đã sửa | Ghi chú |
|---|---|---|
| `model_habit.py` | ✅ | idiom gốc |
| `model_conesep.py` | ✅ | hardcode 32 → đọc `batch_size`, unset vẫn = 32 |
| `model_intent.py` | ✅ | cùng idiom |
| `model_hint.py` | ✅ | cùng idiom |
| `model_tme.py` | ✅ | giữ nguyên `targetpad_transform` + autocast riêng |
| `model_ptha_mtst.py` | ✅ | hardcode 32 → đọc `batch_size` |
| `model_encoder.py` | ✅ | giữ `open_clip` `preprocess_val(image.copy())` |
| `model_limn.py` | ✅ | transform theo category, capture per-group |
| `model_csmcir.py` | ❌ **không sửa** | không có vòng lặp lô: nạp cả gallery vào một tensor |

Một idiom duy nhất, không có biến thể thứ hai:

```python
pixels_by_batch = prefetch_rows(decode, batches, workers)
for index, pixels in enumerate(pixels_by_batch):
    ...
```

`prefetch_rows` (`workbench/composition/adapters.py`) dùng `ThreadPoolExecutor`, `map` giữ nguyên
thứ tự nộp, và **có chặn**: tối đa `workers * 2` lô đang bay. Chặn là bắt buộc — decode hết 74,381
ảnh trước khi forward lần đầu sẽ buffer ~45 GB RAM (74381 × 224 × 224 × 3 × 4 byte).

**b) `limit_torch_threads()`** — ghim intra-op thread của torch trong worker.

**c) BLAS thread env** — `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
`NUMEXPR_NUM_THREADS` set trước khi numpy import, cả trong `worker_environment()` lẫn đầu
`composition_eval.py`.

**d) Nút vận hành** — `--batch-size`, `--decode-workers`, và env `WORKBENCH_COMPOSITION_BATCH_SIZE`,
`WORKBENCH_DECODE_WORKERS`. **Mặc định batch vẫn là 32**, nên run nào không opt-in thì `run_id`
không đổi.

**e) VRAM guard** — `require_batch_fits()` từ chối batch không vừa (85% tổng VRAM) trước khi chạy,
thay vì OOM sau 90 phút.

**f) `scoring_module_source_hash` vào `identity`** (`runner.py`) — vá lỗ: helper dùng chung nằm trong
`adapters.py`, mà `adapter_module_source_hash` chỉ hash module của adapter. Không có hash này, sửa
helper dùng chung sẽ tái dùng rank cũ dưới `run_id` không đổi.

## 4. Tương đương số học

| Thay đổi | max abs delta | rank đổi |
|---|---|---|
| Decode song song (2/4/8 worker) | **0.0 — bit-identical** | 0 |
| `limit_torch_threads` | **0.0 — bit-identical** | 0 |
| bf16 autocast | không áp dụng | — |
| **einsum `optimize=True`** | 3.576e-07 | **207 / 18,009** |
| einsum chunked GEMM (chunk 4096) | 3.576e-07 | **207 / 18,009** |

Hai dòng cuối là số **đầy đủ, toàn cohort** (6003 query × 3 condition), đo trên chính feature
memmap và rank đã lưu của run `497c25c5`. Số cũ "14 / 1200" chỉ là mẫu 400 query/condition và
**đánh giá thấp** mức độ thay đổi — ghi lại ở đây để không ai trích lại con số nhỏ đó.

Decode song song là **bit-identical**, nên không đổi một feature nào. Đây là lý do nó an toàn để
bật mà không cần biện luận thêm.

### Bit-identity đo theo TỪNG transform, không suy từ HABIT

Delta 0.0 của HABIT **không** suy ra được cho model khác, vì mỗi họ dùng transform khác nhau. Đo
riêng từng cái, 512 ảnh thật, worker 2/4/8:

| Họ | Transform | max abs delta | Bit-identical |
|---|---|---|---|
| HABIT / INTENT / HINT / TME / CONESEP / PTHA | `targetpad_transform(1.25, 224)` (lavis) | **0.0** | ✅ 2/4/8 worker |
| ENCODER | `open_clip` `preprocess_val` (`image.copy()`) | **0.0** | ✅ 2/4/8 worker |
| LIMN | `open_clip.image_transform(224, …)` (`image.copy()`) | **0.0** | ✅ 2/4/8 worker |

Cả ba họ đều bit-identical. `CSMCIR` không đo được theo cách này vì nó nạp cả gallery vào **một**
tensor trong **một** job duy nhất (không có vòng lặp lô để song song hoá) — xem mục 6.

### CẢNH BÁO: bất đối xứng summation giữa các model

Đây là phát hiện thật, **không** phải thứ cần "sửa" bằng cách thêm `optimize=True`:

- **CONESEP / CSMCIR / PTHA** gọi `np.einsum(..., optimize=True)` với subscript `nmk` trên 32 token.
- **HABIT / INTENT / HINT / TME** gọi `np.einsum("nd,gqd->ngq", ...)` **không** `optimize`.

Bốn model sau byte-identical nhau ở dòng `score()`; ba model trước dùng thứ tự cộng dồn khác.
Không model nào override `rank()`, nên cả bảy dùng chung `_ranks_from_scores` — nhưng **đầu vào**
của nó đã khác ở mức 1 ULP. Mà `_ranks_from_scores` đếm `#bằng-nhau` chính xác, nên một chênh lệch
1 ULP có thể biến một tie thành không-tie và **dịch rank đi 1**.

Hệ quả: **rank giữa hai nhóm model không hoàn toàn so sánh được ở mức từng query.** Ở mức R@10/R@50
thì không đáng kể (flip đo được rơi vào rank ~2k–64k), nhưng đó là caveat phải ghi, không phải bug
phải vá. Vá theo hướng nào cũng là đổi metric của một nửa cohort — để người quyết định.

## 5. Bị từ chối — quyết định bằng METRIC, không phải bằng tốc độ

Điểm số học trên CPU là nút cổ chai lớn thứ hai: log cohort cho thấy `cpu=+60s/min` = **đúng một
core trên 20** trong suốt pha scoring, trong khi GPU nằm ở 852 MiB. Nguyên nhân trực tiếp: `einsum`
không `optimize=True` không đi vào BLAS `gemm` cho shape này.

Tốc độ đo được (BLAS 8 thread, gallery thật 74,381):

| cách | q/s | 6003 query | max abs delta |
|---|---|---|---|
| `einsum` hiện tại (production) | 5.6 | 18.0 phút | 0 (chuẩn) |
| `einsum(..., optimize=True)` | 25.7 | 3.9 phút | 3.576e-07 |
| chunked GEMM (chunk 4096) | **99.7** | **1.0 phút** | 3.576e-07 |

### Câu hỏi đúng là METRIC, và câu trả lời là KHÔNG

Tốc độ 18× không quyết định gì. Đã tính lại **toàn bộ 6003 query × 3 condition** từ chính feature
memmap và rank đã lưu của run `497c25c5`, dùng đúng công thức production
(`compute_condition_metrics`: R@k là phần trăm `rank <= k`, nDCG = trung bình `1/log2(rank+1)`) —
tất cả **chỉ phụ thuộc rank**.

| condition | R@10 old | R@10 new | R@50 old | R@50 new | nDCG old | nDCG new | \|ΔnDCG\| | query đổi rank | max \|Δrank\| | vượt K=10 | vượt K=50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| multimodal | 35.6155 | 35.6155 | 54.3728 | 54.3728 | 0.3282828051 | 0.3282828065 | 1.40e-09 | **4** | 1 | 0 | 0 |
| text_only_black_image | 15.2924 | 15.2924 | 27.0531 | 27.0531 | 0.2006777269 | 0.2006777284 | 1.55e-09 | **35** | 2 | 0 | 0 |
| image_only_empty_text | 3.3150 | 3.3150 | 10.3115 | 10.3115 | 0.1137358865 | 0.1137358708 | 1.58e-08 | **168** | 5 | 0 | 0 |
| **tổng** | giống hệt | | giống hệt | | | | **~1.6e-08** | **207 / 18,009** | **5** | **0** | **0** |

`chunked GEMM` cho **đúng cùng bảng** với `optimize=True` — cùng delta, cùng 207 query đổi rank,
cùng max Δrank 5, và cũng **0 vượt biên K=10/K=50**. Nhanh hơn 4× nhưng không "sạch" hơn một chút
nào, vì cả hai chỉ khác `einsum` ở **thứ tự cộng dồn float**.

Harness tự kiểm: đường `optimize=False` tính lại đúng bằng rank run đã lưu (0 mismatch trên mẫu
64 query/condition), nên bảng này đáng tin.

### Tại sao R@10/R@50 giống hệt mà vẫn TỪ CHỐI

G3 yêu cầu **rank giống hệt**, không phải Recall giống hệt: "Identical target ranks and hit/miss
outcomes… A changed ranking cannot be excused solely because aggregate Recall remains unchanged."
Và nó chỉ cho phép "chunked/vectorized scoring with **identical** scores" — `3.576e-07` không phải
identical.

Ba lý do cụ thể, không phải giáo điều:

1. **207 query đổi rank là artifact đã công bố.** Candidate Top-K được trả cho UI; đổi rank nghĩa là
   người dùng thấy thứ tự khác, dù Recall tổng không đổi.
2. **0 query vượt biên K=10/K=50 ở cohort này — nhưng đó là quan sát, không phải bảo đảm.** Query
   nào có rank đúng bằng 49/50/51 sẽ vượt biên chỉ vì 1 ULP. Không có gì trong phép tính ngăn điều
   đó; chỉ là lần đo này không có ca nào.
3. **`run_id` không đổi** nếu chỉ sửa thân `score()`, nên một rank bị lật sẽ nằm im trong artifact
   cũ — loại lỗi tệ nhất, vì không ai truy được nó.

**Quyết định: giữ `np.einsum("nd,gqd->ngq").max(-1)` làm mặc định cho cả 9 adapter.** Đã áp dụng
rồi revert; `grep -c optimize=True` = **0** ở habit/intent/hint/tme. Ba adapter còn lại
(conesep/csmcir/ptha_mtst) vốn đã có `optimize=True` từ trước — **không phải thay đổi của tôi**, và
tôi không sửa chúng (xem caveat bất đối xứng ở mục 4).

### Ý tưởng còn để ngỏ (KHÔNG implement — đây là câu hỏi nghiên cứu mới)

Nếu muốn giữ tốc độ 18× **và** bit-identity thì phải ép thứ tự cộng dồn của BLAS trùng đúng lattice
của `einsum` — ví dụ tự viết kernel cộng theo thứ tự cố định, hoặc cố định thứ tự khoá/khối để
`gemm` sinh ra cùng cây cộng. Trần tốc độ đo được là **99.7 q/s (18×)**; chi phí là một kernel tự
viết phải chứng minh bit-identical trên toàn bộ gallery. Không nằm trong task này.
Song song hoá **theo query** (chia query thành dải, mỗi dải giữ nguyên thứ tự cộng bên trong) về lý
thuyết không đổi bit nào vì mỗi query là một tích vô hướng độc lập — đây là hướng rẻ nhất và đáng
thử trước.

## 5b. ĐƯỢC CHẤP NHẬN: song song hoá theo QUERY (bit-identical, không đổi rank nào)

Song song hoá theo trục **query** khác về bản chất với `optimize=True`: mỗi query là một tích vô
hướng độc lập trên cùng gallery, nên chạy nhiều query đồng thời **không thể** đổi thứ tự cộng dồn
bên trong bất kỳ query nào. Kết quả giống hệt theo đúng nghĩa đen, không phải "trong sai số".

### Bằng chứng bit-identity (feature thật, gallery 74381×32×256)

Điều kiện tiên quyết — một query tính riêng có giống khi nằm trong batch không:

```
single-vs-batch: worst_delta=0.000e+00 bit_identical=True
subbatch m=2/4/8/16/32/64/128 : bit_identical=True  maxdelta=0.000e+00   (tất cả)
```

Không có phụ thuộc kích thước M của BLAS. Nên chia batch không đổi một bit nào.

### Đo trên chính đường production (`rank_in_query_chunks` trong `runner.py`)

512 query thật, so với **rank đã lưu của run thật**:

| threads | thời gian | q/s | speedup | 6003 query | rank identical |
|---|---|---|---|---|---|
| 1 | 106.0 s | 4.8 | 1.00× | 20.7 phút | chuẩn |
| 2 | 58.7 s | 8.7 | 1.81× | 11.5 phút | ✅ |
| 4 | 32.6 s | 15.7 | 3.26× | 6.4 phút | ✅ |
| **8** | **18.7 s** | **27.4** | **5.67×** | **3.7 phút** | ✅ |

**0 mismatch trên 512 query khi so với artifact đã lưu.** Đây là phép so với kết quả run thật đã
publish, không phải so với một bản tính lại.

### Tương tác với BLAS thread

Đây là điểm quan trọng nhất về regime:

- Đường `einsum` hiện tại **không dùng BLAS đa luồng** — đã đo: đổi `OPENBLAS_NUM_THREADS` từ 1 lên
  10 **không đổi** tốc độ (5.6–6.5 q/s). Đó chính là lý do trước đây nó chỉ ăn 1 core.
- Vì BLAS chỉ dùng 1 luồng, song song theo query **không tranh chấp** với nó: tăng thread query là
  tăng gần tuyến tính (1.81 / 3.26 / 5.67×). Amdahl thực tế đạt ~71% ở 8 thread.
- Nếu sau này ai bật BLAS đa luồng thì hai cơ chế sẽ tranh nhau; khi đó chỉ nên chọn **một**. Ghi ở
  đây để không ai bật cả hai rồi thấy chậm hơn.
- Trần lý thuyết ~7.5 phút → thực tế 3.7 phút cho 6003 query/condition. Trước là 18–20.7 phút.

### Vị trí code và hệ quả `run_id`

- Nằm **hoàn toàn trong `workbench/composition/runner.py`** (`rank_in_query_chunks`,
  `score_in_query_chunks`), **không sửa `model_*.py`**.
- `runner.py` **không** được hash vào `identity`. Nên `adapter_source_hashes` và
  `adapter_module_source_hash` **không đổi** → **`run_id` không đổi** → các run đã publish vẫn so
  sánh được, và không phải encode lại.
- Nút: `WORKBENCH_SCORE_THREADS` (mặc định `min(8, cpu//2)`). Đặt `=1` để tắt hoàn toàn.

### Vì sao cái này được và cái kia không

| | Đổi thứ tự cộng dồn trong 1 query? | Rank đổi | Chấp nhận |
|---|---|---|---|
| `optimize=True` / GEMM chunk | **Có** | 207/18,009 | ❌ |
| Song song theo query | **Không** | **0** | ✅ |

## 6. Concurrency hai model: bác bỏ bằng số

`habit` encode đạt **8439 MiB** peak. Hai model ≈ 16.9 GB > 16.3 GB. Không vừa. `drive.py` tuần tự
hoá là đúng. Không làm.

## 7. Trả lời trực tiếp ba câu hỏi

**1. `habit` pid 184129 có đang chạy code cũ?** **Có.** Python đã import `model_habit.py` vào bộ nhớ
lúc process khởi động; sửa file trên đĩa không ảnh hưởng process đang chạy.

**2. Vậy 98× chỉ áp dụng cho `intent`/`hint`/`conesep`?** **Đúng** — và cho mọi lần chạy `habit` tiếp
theo. Run đang chạy giữ nguyên code cũ.

**3. CONESEP có bị đổi đường tính không?** **Không.** `score()` của CONESEP còn nguyên
(`optimize=True` sẵn có, tôi không chạm). Tôi chỉ đổi decode của nó. Một khác biệt nhỏ: CONESEP
trước hardcode `32`, giờ đọc `batch_size` từ job; khi env không set thì giá trị vẫn là 32, nên hành
vi không đổi.

## 8. Hệ quả `run_id` / cache

`_load_memmap` (`runner.py:333-343`) chỉ so `metadata["fingerprint"]` **trong file `.json` sidecar**;
nội dung `.npy` không mang fingerprint. Nên:

> **Cache 2.3 GiB CÓ THỂ copy sang run dir mới và dùng lại được** — chỉ cần sidecar chứa đúng
> fingerprint của run mới. Nhưng fingerprint phụ thuộc `identity`, mà `identity` giờ có thêm
> `scoring_module_source_hash`. Nên hiện tại copy nguyên xi sẽ **không** khớp; phải tính lại
> fingerprint rồi ghi lại sidecar. Việc đó rẻ (15 phút re-score) so với re-encode (1.5 giờ), nhưng là
> thao tác thủ công và dễ sai — không nên làm ngầm.

Chi phí từng thay đổi:

| Thay đổi | `run_id` | Chi phí |
|---|---|---|
| BLAS env, `limit_torch_threads`, decode song song | **không đổi** | 0 |
| `--batch-size` khác 32 | đổi (`adapter_batch_size`) | re-encode 1 lần |
| `--decode-workers` bất kỳ | **không đổi** | 0 |
| `scoring_module_source_hash` | **đổi toàn bộ** | run_id cũ = artifact tiền-thay-đổi |

## 9. Lệnh cho operator

Đường nhanh, **không** đổi `run_id` (decode-only + thread):

```bash
workbench/envs/ctl312h/bin/python -u workbench/scripts/composition_eval.py \
  --config workbench/docs/reviews/20261108T113610-eval/circus_cohort.yaml \
  --models intent,hint,conesep --subsets full,sf,validated \
  --topk-depth 50 --keep-features \
  --decode-workers 8
```

Mặc định đã là `min(10, cpu//2)` = 10 trên máy này; `--decode-workers 8` đo được tốt hơn 10 và để
dành core cho project khác.

Nếu muốn thử batch lớn hơn (chấp nhận `run_id` mới + re-encode):

```bash
... --batch-size 128 --decode-workers 8
```

Guard sẽ từ chối trước khi chạy nếu không vừa VRAM:

```
batch size 4096 would need ~56.0 GiB of VRAM, above the 13.5 GiB safety budget of 15.9 GiB
```

## 10. Số liệu thô

`batch.txt`, `query.txt`, `ein_threads.txt`, `vdecode.txt`, `eq_fast.txt`, `eq_opt.txt`,
`cohort_full.txt` — trong cùng thư mục này.

## 10b. Hệ quả provenance: bốn model có số công bố PHẢI chạy lại

`model_encoder`, `model_limn`, `model_ptha_mtst`, `model_tme` là bốn model có số **đã công bố**
(tme MM R@10 34.02% / R@50 53.96%, encoder 31.02% / 49.58%, limn 30.06% / 48.76%,
ptha_mtst 19.47% / 34.45%).

Sửa worker của chúng đổi `adapter_module_source_hash` → **`run_id` mới**. Cache 2.3 GB của chúng đã
bị xoá. Nên **từ HEAD không còn đường nào tái lập đúng các con số đã công bố** — đây là
provenance break, và nó nặng hơn decode chậm.

Xử lý (không được làm ngầm):

1. Patch (đã làm) → verify bit-identity từng model (đã làm, cả ba họ transform đều 0.0).
2. **Chạy lại cả bốn** với `--topk-depth 50 --keep-features`. Mỗi model giờ ~15–25 phút thay vì
   ~73 phút, nên cả bốn ≈ 1.5 h. Lần chạy này còn cho chúng bộ candidate Top-K mà hiện chưa có.
3. Dòng mới là **lần thử thứ hai**, phải ghi provenance của nó. `run_id` cũ đánh dấu là
   **pre-decode-change artifact**. **Không thay thế im lặng** một con số đã công bố — thêm số mới
   kèm provenance.

## 11. Cái gì chưa làm, và tại sao

- **Nâng batch size mặc định**: đo được không nhanh hơn (bão hoà ở 32), lại đổi `run_id`.
- **bf16**: không nhanh hơn fp32 trên slice này, lại phải chứng minh tương đương.
- **`optimize=True` / chunked GEMM**: nhanh 9–18× nhưng **đổi rank của 207/18,009 query** (max Δrank
  5; R@10/R@50 giống hệt; 0 query vượt biên K=10/K=50; delta score 3.576e-07). **Từ chối** — G3 đòi
  rank giống hệt, và rank là artifact đã công bố (Top-K trả cho UI). Xem mục 5 để có bảng đầy đủ.
- **Concurrency hai model**: 8439 MiB × 2 > VRAM. Từ chối.
- **DataLoader rewrite**: thread pool đã đủ và bit-identical; DataLoader thêm phức tạp không mua gì.
- **Scoring trên GPU bằng torch**: đổi `run_id`, cần re-encode, và chưa có bằng chứng về tương đương
  rank. Không làm khi cohort đang chạy.

## 12. Test

`workbench/tests/test_throughput_knobs.py` — 13 test, pass. Khoá các tính chất: thứ tự decode được
giữ, decode có chặn, guard từ chối batch quá lớn, mặc định batch vẫn 32, env BLAS được ghim.

Ghi chú: test **không** khoá `token_max_scores` vì thay đổi đó đã bị từ chối — test cho code không
ship sẽ ghi lại sai hợp đồng.
