# CONSOLIDATED MODEL MATRIX

16 model đã đăng ký. Mỗi dòng là kết quả audit **độc lập trên source đã ghim**, với bằng chứng
`file:line`. Cột "Protocol" là protocol **chứng minh được từ code**, không phải nhãn trong bài báo.

| model_id | Source audited | Protocol | Checkpoint | Preparation | Lệnh chính thức | Semantics evaluator | Per-query | Verdict |
|---|---|---|---|---|---|---|---|---|
| `limn` | SOURCE_AUDITED (`7d7bc9b1`) | C ✓ | CHECKPOINT_VERIFIED (3 model theo category, digest khớp từng byte) | OK | COMMAND_AUDITED | `np.argsort(-sims)`, mask `-10e10` | **SUPPORTED — đã hiện thực** | READY_FOR_PC_VALIDATION |
| `csmcir` | BLOCKED_SOURCE (checkout chưa patch không import được) | A ✓ | CHECKPOINT_VERIFIED (metadata LFS) | OK (caption alias khai báo) | COMMAND_AUDITED | full gallery, reference eligible | IMPLEMENTABLE | BLOCKED_EXTERNAL_DEPENDENCY |
| `hint` | SOURCE_AUDITED (`bec50b6c`) | C ✓ | CHECKPOINT_BLOCKED | thiếu correction dict | **COMMAND_BLOCKED** | pair-union + exclude | IMPLEMENTABLE | BLOCKED_EXTERNAL_DEPENDENCY |
| `encoder` | SOURCE_AUDITED (`29a2a31d`) | C ✓ | CHECKPOINT_BLOCKED | `datasets1.py` vắng | COMMAND_BLOCKED | 1 model dùng chung 3 category | IMPLEMENTABLE | BLOCKED_EXTERNAL_DEPENDENCY |
| `pair` | BLOCKED_SOURCE (`bfb0b98c`) | C ✓ | CHECKPOINT_BLOCKED (**B1/B2 không quyết được từ evidence**) | OK | COMMAND_BLOCKED | pair-union + exclude | IMPLEMENTABLE | BLOCKED_EXTERNAL_DEPENDENCY |
| `ptha_mtst` | **BLOCKED_SOURCE** (repo chưa sync/pin) | **UNRESOLVED** (bài báo không nêu) | CHECKPOINT_BLOCKED (0 weight FashionIQ công khai) | unavailable | COMMAND_BLOCKED | không có | BLOCKED | BLOCKED_EXTERNAL_DEPENDENCY |
| `airknow` | BLOCKED_SOURCE (`6ab6fa95`) | B ✓ | CHECKPOINT_BLOCKED (không có weight clean 0%) | thiếu correction dict | COMMAND_BLOCKED | full val split, mask `-10e10` | IMPLEMENTABLE | BLOCKED_EXTERNAL_DEPENDENCY |
| `conesep` | SOURCE_AUDITED (`bd4bf629`) | B ✓ | CHECKPOINT_BLOCKED (không có weight clean) | thiếu correction dict | COMMAND_BLOCKED | matmul→max(-1) trên 32 token | IMPLEMENTABLE | BLOCKED_EXTERNAL_DEPENDENCY |
| `habit` | SOURCE_AUDITED (`6ddc2cd6`) | B ✓ | CHECKPOINT_BLOCKED (không có weight clean) | thiếu correction dict | COMMAND_BLOCKED | full val split, mask `-10e10` | IMPLEMENTABLE | BLOCKED_EXTERNAL_DEPENDENCY |
| `intent` | SOURCE_AUDITED (`2a75fc6f`) | B ✓ | CHECKPOINT_BLOCKED (không có weight clean) | hợp đồng **chưa đăng ký** | COMMAND_BLOCKED | full val split, mask `-10e10` | IMPLEMENTABLE | BLOCKED_EXTERNAL_DEPENDENCY |
| `tgcir` | SOURCE_AUDITED (`65fa78ea`) | C ✓ | CHECKPOINT_BLOCKED (Drive không map được) | thiếu dict + asset URL hỏng | COMMAND_BLOCKED | pair-union + exclude | IMPLEMENTABLE | BLOCKED_EXTERNAL_DEPENDENCY |
| `clvc_net` | BLOCKED_SOURCE (`bd9b6889`) | C ✓ | CHECKPOINT_BLOCKED (cần 6 file: 3 cat × 2 state) | `simplevocab.json` vắng | COMMAND_BLOCKED | local+global concat 2048-d | IMPLEMENTABLE | BLOCKED_IMPLEMENTATION |
| `dcnet` | SOURCE_AUDITED (`68c79d38`) | B ✓ | CHECKPOINT_BLOCKED (Drive không map) | PKL OOV **không seed** | COMMAND_AUDITED | mask **cả hai** nhánh điểm, fuse `log_softmax` | IMPLEMENTABLE | BLOCKED_EXTERNAL_DEPENDENCY |
| `combiner_rn50x4_noft` | SOURCE_AUDITED (`dfed9f74`) | A ✓ | CHECKPOINT_BLOCKED (Drive không digest) | OK | COMMAND_AUDITED | full split, reference eligible | IMPLEMENTABLE | BLOCKED_EXTERNAL_DEPENDENCY |
| `clip4cir_rn50x4_fullft` | SOURCE_AUDITED (`dfed9f74`) | A ✓ | CHECKPOINT_BLOCKED (**bundle 2 file**) | OK | COMMAND_AUDITED | full split, reference eligible | IMPLEMENTABLE | BLOCKED_EXTERNAL_DEPENDENCY |
| `sprc` | SOURCE_AUDITED (`2935a539`) | A ✓ | CHECKPOINT_BLOCKED (OneDrive 403) | OK | COMMAND_BLOCKED | full split, reference eligible | IMPLEMENTABLE | BLOCKED_IMPLEMENTATION |

## Điểm chung đã kiểm chứng

* **Ba protocol được bảo toàn.** 16/16 model có `native_protocol` thuộc đúng ba tên đã khai, hoặc là
  `null`. Không model nào bị gán một protocol thứ tư. `ptha_mtst` là `null` (sau khi sửa F-01/F-02).
* **Chính sách reference được chứng minh bằng code**, không bằng nhãn: Protocol A không mask (bốn model
  `csmcir`/`combiner`/`clip4cir`/`sprc`), B và C mask bằng đúng sentinel `-10e10` của tác giả.
* **Gallery được chứng minh**: A và B đọc `split.{cat}.val.json` theo thứ tự file; C dựng union duy
  nhất theo thứ tự xuất hiện. Không model nào ghi "original" mà thực chất dùng union.
* **10/16 model không có entrypoint CLI** trong source gốc — workbench ghi `COMMAND_BLOCKED`, không
  tự bịa lệnh.
* **Điều kiện nhiễu**: với họ iLearn (`airknow`, `conesep`, `habit`, `intent`), evaluator luôn chạy
  trên validation **sạch**, nhưng weight công khai chỉ có 20/50/80% nhiễu. Đối chiếu chúng với dòng
  headline 0% của bài báo là **không hợp lệ** — được ghi riêng, không bị trộn.
* **`QUERY_EXPORT_IMPLEMENTABLE` cho 15/16**: điểm sinh ranking được chỉ ra cụ thể (`file:line`) trong
  từng báo cáo. Chỉ `ptha_mtst` không có, vì không có source.
* **Chỉ `limn` đủ điều kiện xuất per-query thật** trong lượt này: source ghim được, checkpoint có trên
  máy với digest khớp, replay đã kiểm chứng **không có sai lệch thực thi** so với `test()` gốc.

## Vì sao không model nào ở trạng thái `READY_FOR_PC_VALIDATION` ngoài `limn`

Đây không phải thiếu sót của workbench mà là thực tế của artifact công khai:

1. Thiếu checkpoint (Drive/OneDrive không digest, không map được member, hoặc weight clean 0% không
   tồn tại).
2. Thiếu asset phụ thuộc (backbone, tokenizer, correction dictionary, resized images) mà tác giả
   không công bố kèm script sinh.
3. Thiếu entrypoint CLI — phải viết adapter mới có lệnh chạy được.
4. Source ghim có file thiếu (import chết ngay), tức **không thể chạy đúng như tác giả** kể cả khi có
   đủ dữ liệu.

Mọi mục trên đều là `EXTERNAL_BLOCKER`, không phải bug của workbench. Viết adapter/parser cho chúng
mà không có output thật để đối chiếu sẽ tạo ra chính loại bằng chứng giả mà dự án đang chống.
