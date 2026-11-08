# Vì sao lại pin BLAS, khi upstream không đặt gì?

Câu hỏi: *"mấy repo kia họ không đặt gì hết? vậy sao mình lại đặt?"*

## 1. Upstream im lặng = không tất định, KHÔNG phải trung tính

Lời gọi trong eval path của upstream là numpy trần:

- `workbench/third_party/ENCODER/test.py:62`, `:130`, `:215` — `sims = all_queries.dot(all_imgs.T)`
- `workbench/third_party/LIMN/LIMN/test.py:74`, `:146`, `:240` — `sims = all_queries.dot(all_imgs.T)`

Không file eval nào đặt BLAS thread. Lần duy nhất có gọi thread là **`torch.set_num_threads(2)` và
chỉ trong file train**: `ENCODER/cirr_test_submission.py:13`, `ENCODER/train.py:29`,
`LIMN/LIMN/train.py:29`, `LIMN/LIMN+/train.py:32`. Tìm `OMP_NUM_THREADS`/`OPENBLAS` trong cả hai
repo: **không có kết quả** ngoài `torch.set_num_threads`.

OpenBLAS tự chọn kích thước pool theo số core **lúc import**. Nên cùng một checkpoint:

- máy 8 core → một chuỗi cộng dồn
- máy 20 core → chuỗi khác

**Hệ quả phải nói thẳng: rank đã publish của upstream phụ thuộc vào máy đã chạy nó, nên việc tái lập
bit-exact một con số của upstream là điều KHÔNG AI làm được** — kể cả chính tác giả, trên máy khác.

### Đo được: `np.dot` theo số thread

Cùng ma trận (256 query × 512 chiều × 8192 gallery), cùng numpy, chỉ đổi BLAS thread:

| threads | score max abs delta (so với 1) | rank đổi |
|---|---|---|
| 1 | 0.000e+00 | 0 |
| 2 | 0.000e+00 | 0 |
| 4 | 0.000e+00 | 0 |
| 8 | 0.000e+00 | 0 |
| **20** | **6.104e-05** | 0 |

Hai điều rút ra:

1. **Sự khác biệt là thật nhưng xuất hiện ở ngưỡng cao** (20 thread), không phải ở 2. Đây là
   threaded-kernel **chia lại phép rút gọn theo trục K**, không phải gemv-vs-gemm.
2. **Pin=1 nằm giữa một vùng phẳng ổn định (1–8 thread đều cho delta 0.0).** Nên chọn 1 không phải
   chọn tuỳ tiện một điểm dị thường — nó là điểm rẻ nhất trong vùng ổn định.

## 1b. Lý do MẠNH NHẤT để pin: BLAS=10 không tất định NGAY TRÊN CÙNG MỘT MÁY

Đây là lập luận mạnh hơn cả lập luận số thread, và nó thuộc về quyết định policy nên phải nằm ở đây.

Chạy **hai lần cùng một script**, cùng máy, cùng input, cùng `OPENBLAS_NUM_THREADS=10` (encoder,
gallery 74381 ảnh, 6003 query):

| condition | ΔnDCG lần 1 | ΔnDCG lần 2 | chênh |
|---|---|---|---|
| multimodal | **8.33e-05** | **3.42e-06** | **24×** |
| text_only_black_image | 1.30e-08 | 1.30e-08 | — (nhưng số query đổi: 3 → 4) |
| image_only_empty_text | — | 2.18e-05 | — (lần 1 không đo được condition này) |

**BLAS=10 không tái lập được ngay cả giữa hai lần chạy trên cùng host.** Không chỉ "đổi máy mới đổi
kết quả" — chỉ cần chạy lại là đã khác. Nghĩa là:

- Không pin thì artifact **không resumable**: `--resume` có thể tái dùng rank cũ trong khi một lần
  chạy mới sẽ sinh rank khác, và không có gì trong artifact phân biệt hai thứ đó.
- Không pin thì "tái lập lại kết quả" **không có nghĩa** — không tồn tại một kết quả cố định để tái lập.
- Vì vậy pin là điều kiện **cần** để artifact có nghĩa, không phải một tinh chỉnh hiệu năng.

Hệ quả khi đọc số: các ΔnDCG đo được chỉ nên đọc như **bậc độ lớn** (1e-8 … 8e-5), **không** phải
một hằng số. Đừng trích một giá trị đơn lẻ như thể nó tái lập được.

## 2. Ba lựa chọn, không phải hai

| lựa chọn | throughput | tái lập được? |
|---|---|---|
| **Không đặt gì** (theo upstream) | nhanh nhất trên máy nhiều core | **KHÔNG** — đổi máy là đổi last bit |
| **Pin=1** (đã chọn) | serial chậm hơn ~30% (301.8 → 210.9 q/s) ở bước chấm điểm; bù lại từ 4 thread trở lên nhờ song song theo query (549.9 q/s) | **CÓ**, và ghi vào `identity` |
| Pin=10 | nhanh hơn serial-BLAS-1 nhưng 8 nhóm × 10 thread = 80 thread / 20 core → oversubscribe; đúng lỗi D3 quay lại | có, nhưng chậm hơn thực tế vì tranh chấp |

**"Không đặt gì" không tồn tại như một lựa chọn tái lập được.** Nó chỉ có nghĩa "để máy quyết
định", tức là kết quả phụ thuộc máy — chính là tính chất mà artifact khoa học không được có.

## 2b. HAI cơ chế khác nhau — đừng gộp

Đây là chỗ một comment cũ trong repo nói sai, và gộp hai hiện tượng này là nguyên nhân:

| | **Threaded-kernel K-reduction partitioning** | **Không có đường BLAS** |
|---|---|---|
| áp dụng cho | GEMM-path: encoder, limn, csmcir, ptha_mtst, conesep | naive einsum: habit, intent, hint, tme |
| vì sao | `optimize=True` hạ phép rút gọn xuống gemm, hoặc thân hàm dùng `@` | `np.einsum("nd,gqd->ngq")` không `optimize` chạy vòng lặp numpy tự viết |
| đổi số thread BLAS thì | **KẾT QUẢ đổi** (OpenBLAS chia lại phép cộng theo trục K) | **KHÔNG đổi gì** — không có gì để chia |
| bằng chứng | encoder: 1 query đổi rank 2→3, ΔnDCG 2.181e-05 | `ein_threads.txt` `old`: 6.5/6.0/6.2/5.6/6.2 q/s ở 1/2/4/8/10 thread — **phẳng** |
| kết luận | phải pin, nếu không kết quả phụ thuộc máy | tăng thread **không** làm nó nhanh hơn; speedup của nó đến từ song song theo query |

**Phẳng trên cả 5 mức thread là bằng chứng loại trừ lỗi "pool sizing muộn".** Nếu chỉ vì OpenBLAS
đã lấy kích thước pool trước khi numpy import, thì đặt env **trước** khi import sẽ thay đổi kết quả —
nó không thay đổi. Nguyên nhân thật là contraction đó **không có đường BLAS nào cả**.

Comment cũ trong `workbench/scripts/composition_eval.py` nói ngược lại ("the scoring einsum runs at
6.2 q/s regardless of the env because OpenBLAS is baked in at import"). Đã sửa: env vẫn phải đặt
trước khi import (đúng, và cần thiết cho **GEMM path**), nhưng **lý do** đã được nêu đúng.

## 3. Sai lệch, ghi theo từng họ adapter

| họ | adapter | ảnh hưởng |
|---|---|---|
| **GEMM-path** (5) | encoder, limn, csmcir, ptha_mtst, conesep | nDCG **lệch 2.181e-05**; R@10/R@50 **không đổi**; **0** query vượt biên K=10/50/100 ở mẫu đo |
| **naive-einsum** (4) | habit, intent, hint, tme | `array_equal`, delta **đúng 0.0** — không bị ảnh hưởng |

Phân loại này thay cho cách chia sai trước đó ("4 `@` vs 4 einsum"). Điểm phân biệt không phải có
`@` hay không, mà là **phép rút gọn có bị hạ xuống BLAS gemm** hay không:
`np.einsum(..., optimize=True)` **có** hạ xuống (nên conesep/csmcir/ptha nằm trong nhóm GEMM), còn
`np.einsum("nd,gqd->ngq")` naive thì **không**.

**Đây là policy được khai báo, không phải một tối ưu.** Giá của nó (nhánh serial chậm ~30%) được
chấp nhận để đổi lấy tính tái lập.

## 4. Bác bỏ "pin=10 sẽ tái lập mọi artifact lịch sử"

Khối BLAS env trong `composition_eval.py` (`_BLAS_THREADS`, vòng `os.environ.setdefault`) được đưa
vào bởi commit **`3d0e2b89fb3a86d5eedd8fa7f122d8c1a9dcb502`, 2026-11-08 17:55:36 +0700** (xác định
bằng `git log -S'_BLAS_THREADS'`, không dùng mtime). Pin `_blas_thread_limit` trong `runner.py` được
commit trong **`bf6a448`** (cùng commit với `BLAS_THREADS_PINNED` và `_rank_groups_concurrently`);
trước đó `git show 3d0e2b8:workbench/composition/runner.py` không có `_blas_thread_limit` lần nào.
**Bảng dưới đây không đổi vì thế**: mọi run trong đó kết thúc trước `bf6a448`, nên chưa run nào chạy
dưới pin hiện tại.

Thời điểm kết thúc của từng run (mtime rank artifact):

| run | kết thúc | khối env tồn tại? | pin tồn tại? | policy hiệu dụng |
|---|---|---|---|---|
| encoder `6a528e8` | 2026-11-08 00:00 | **không** | không | **unknown** |
| limn `10173dc` | 2026-11-08 04:51 | **không** | không | **unknown** |
| habit `e0dd04a` | 2026-11-08 05:00 | **không** | không | **unknown** |
| tme `ad47731` | 2026-11-08 06:14 | **không** | không | **unknown** |
| ptha `315dd59` | 2026-11-08 06:43 | **không** | không | **unknown** |
| habit `497c25c` | 2026-11-08 15:21 | **không** | không | **unknown** |
| intent `37ee72a` | 2026-11-08 16:38 | **không** | không | **unknown** |
| hint `4bc9893` | 2026-11-08 17:07 | **không** | không | **unknown** |
| conesep `30f6de4` | 2026-11-08 17:42 | **không** | không | **unknown** |

**Không một run nào chạy sau 17:55:36**, tức là **không run nào từng thấy khối env BLAS**. Nên tiền
đề "pin=10 tái lập mọi artifact lịch sử" là **SAI**: các run đó chạy dưới BLAS do shell/ambient
quyết định, mà artifact **không ghi lại** (identity có 61 khoá, **0** khoá thread/blas/omp). Khi
không biết giá trị cũ thì không có giá trị pin nào — kể cả 10 — tái lập được chúng.

Mọi dòng trong bảng trên phải được ghi nhãn **`unknown (predates the BLAS env block)`**, không được
để ngụ ý chúng cùng policy với code hôm nay.

## 5. Một đoạn để bảo vệ

Upstream không đặc tả BLAS threading ở bất kỳ eval path nào, nên số của họ phụ thuộc host. Chúng ta
buộc phải chọn một giá trị để artifact của chính mình so sánh được và resume được, nên ta pin một
giá trị, ghi nó vào `identity`, và dán nhãn mọi artifact trước-policy là được tính dưới một policy
khác và phần lớn không xác định.
