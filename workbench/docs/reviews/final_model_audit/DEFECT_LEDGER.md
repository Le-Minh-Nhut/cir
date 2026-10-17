# DEFECT LEDGER — LƯỢT SIẾT CHẶT TÍNH ĐÚNG ĐẮN (F-07 … F-16)

Baseline: `4658986`. Mỗi mục dưới đây đều được **tái hiện** trước khi sửa, và có test hồi quy
**cả hai chiều** (đầu vào hợp lệ phải qua). Không mục nào là suy đoán: mục nào không dựng được
đường vi phạm đều bị hạ xuống `REJECTED_FINDING` hoặc `UNREACHABLE_BY_DESIGN`.

## A. Bug đã xác nhận và đã sửa

| ID | Mức | Tái hiện | Nguyên nhân gốc | Sửa |
|---|---|---|---|---|
| **F-07** | **P0** | `index.evidence_status("experiment")` → `PER_QUERY_VERIFIED`. Một file JSON hợp lệ schema tự chứng thực tính khoa học của chính nó; ai ghi được file thì ghi được `"experiment"`. | Đồng nhất **khai báo** (`data_kind`) với **trạng thái bằng chứng đã xác thực**. | Tách đôi. `schemas/results.admission()` là chính sách duy nhất, trả về `PER_QUERY_UNVERIFIED` cho experiment chưa ai chứng thực. `promote()` chỉ chuyển sang `PER_QUERY_VERIFIED` khi **verifier** nói vậy, và chỉ cho **đúng artifact đó**. |
| **F-08** | **P1** | `./pc.sh` → `evaluate_models.execute()` chạy evaluator rồi ghi `"per_query_export_available": False`. Không chỗ nào gọi observer; `limn_export.py` chưa từng được gọi. | Export là entry point rời, không nằm trong transaction của lượt chạy. | `execute()` gọi observer như tiến trình con dưới interpreter của model, truyền `--run-id` và digest của chính invocation, rồi **parent** kiểm artifact và ghi `per_query_artifact_digest` vào report. Thất bại ở đây không đổi kết quả aggregate. |
| **F-09** | **P1** | `run_id_for(model, checkpoint, protocol, source, digest[:16])`: hai lượt chạy khác thời điểm/máy/lệnh nhưng cùng cấu hình → **cùng một `run_id`**. | `run_id` được suy từ *cấu hình*, không phải từ *lượt thực thi*. | `build_result_run` **bắt buộc** `run_id` do caller truyền (`evaluate_models.make_run_id`: thời gian + model + checkpoint + uuid). Bỏ `run_id_for`, thay bằng `development_run_id` chỉ cho fixture. Experiment mang id `dev-` bị từ chối. |
| **F-10** | **P1** | `publish()` kết thúc bằng `os.replace` → công bố artifact B cùng `run_id` **ghi đè** artifact A, mất bytes cũ. | `os.replace` không điều kiện. | `os.link` — tạo tên hoặc lỗi, không có cửa sổ giữa "kiểm tra" và "thay". Công bố lại **cùng bytes** là idempotent; cùng id nhưng **khác bytes** bị từ chối. Temp luôn bị xoá. |
| **F-11** | **P1** | `limn_export.export()` tạo artifact `data_kind="experiment"` mà thiếu `command_digest`/`environment_digest`. | Validator strict của chính nó từ chối artifact nó vừa tạo; hai chính sách khác nhau. | Parent truyền digest của **invocation aggregate** xuống (observer chạy lệnh khác nên không tự suy được). Schema/admission từ chối experiment thiếu provenance. |
| **F-12** | **P1** | Exporter gắn digest của **checkpoint dress** cho cả bundle ba category. | Một scalar không biểu diễn được ba model độc lập. | Bundle manifest đầy đủ (tên + digest từng member), `bundle_manifest_digest` tính bằng đúng quy tắc `evaluate_models.manifest_digest`, cộng `category_checkpoint_ids`; admission từ chối nếu một checkpoint phục vụ hai category hoặc thiếu category. |
| **F-13** | **P1** | `validate_file(strict_real=True)` **chặn** artifact với `run_id` viết tay, nhưng `validate_root` **cho qua** (không gọi `verify_run_identity`), còn `rebuild_index` thì nạp vào DuckDB. Ba entry point, ba kết luận. | Mỗi file áp một luật riêng, không có chủ sở hữu. | Một hàm `admission()` cho tất cả. `validate_file`, `validate_root`, `validate_index` đều gọi nó; `validate_root` thêm `verify_run_identity`. |
| **F-14** | **P2** | `publication.PROTOCOL_LITERATURE_LABELS` nói `fashioniq_full_gallery_ref_excluded → "val"`; schema nói `"original"`. Hai bản sao, mỗi consumer đọc bản của mình. | Bản đồ bị chép đôi. | Schema là chủ sở hữu duy nhất; `publication` import lại. Test quét cây để bắt bản sao thứ hai trong tương lai. |
| **F-15** | **P2** | Pipeline native giữ cột reference bị mask `-10e10`; nếu `top_k >= gallery_size` thì reference xuất hiện như một "ảnh được truy hồi". | Đồng nhất *xếp hạng native* (giữ cột) với *ứng viên hợp lệ* (loại reference). | Giữ nguyên xếp hạng native. Độ sâu lưu bị chặn bởi `gallery_size - 1`; schema dùng `category_eligible_counts` làm biên, nên rank chỉ tồn tại nhờ slot của reference bị từ chối. |
| **F-16** | **P2** | `rank_queries` với vector zero-norm/NaN/inf vẫn **sinh ra xếp hạng**: NaN similarity, và `np.argsort` đặt reference bị mask lên **rank 1**. | Không kiểm hữu hạn trước khi xếp hạng. | `NonFiniteRanking` — từ chối, không thay 0, không kẹp, không bịa rank. Không bao giờ được promote thành parity. |

## B. Bug bổ sung phát hiện trong lượt này (từ audit độc lập)

| ID | Mức | Mô tả | Sửa |
|---|---|---|---|
| **F-17** | **P0** | `promote()` chỉ khớp `run_id`, không khớp **artifact**. Artifact bịa ranking mượn `run_id` của một lượt đã xác thực → được promote thành `PER_QUERY_VERIFIED`. | Receipt của verifier mang `per_query_artifact_digest`; promotion phải khớp **digest của bytes trên đĩa** (do reader tính, không do artifact tự khai) **và** `model_id`/`checkpoint_id`. |
| **F-18** | **P1** | `storage.list_runs`/`get_runs` `SELECT metadata` và **bỏ rơi** cột `evidence_status` của DuckDB → API phục vụ run không có trạng thái bằng chứng, frontend quay về tin `data_kind`. | Đọc kèm cột và merge vào bản ghi trả về. |
| **F-19** | **P1** | Coverage: artifact chỉ có **một** category vẫn qua được strict validation và tự nhận là benchmark đầy đủ. | `admission()` từ chối experiment không phủ đủ ba category. |
| **F-20** | **P1** | UI hiện `1–0 of 0` khi không có per-query, đọc thành "0 truy vấn thất bại" trong khi sự thật là "không có dữ liệu". | Banner và pager nói rõ trạng thái không khả dụng. |
| **F-21** | **P1** | Query identity: dataset native cache `.pkl` được nạp **vô điều kiện**, không kiểm nguồn → cache lệch/cũ làm hỏng danh tính mà Recall vẫn hợp lý. | `assert_alignment()` kiểm từng index: vị trí reference/target khớp annotation, và `model_input_text` khớp composition từ chính caption + correction dict trên đĩa. |
| **F-22** | **P1** | **Do lượt này gây ra**: import `schemas.results` (pydantic) vào đường bị truy vết của `evaluate_models` → planner đòi interpreter cho pydantic → cold start hỏng. | Parent kiểm bằng JSON thuần; verdict admission thuộc về index/API (không nằm trên đồ thị). Test chốt `pydantic` không được lọt vào tập import bị truy vết của LIMN. |

## C. Bác bỏ sau khi kiểm chứng

| Cáo buộc | Kết luận | Bằng chứng |
|---|---|---|
| "Mask `-10e10` lọt vào Top-K vì điểm có thể ≤ -1e11" | **BẤT KHẢ QUA ĐƯỜNG NÀY** | LIMN chuẩn hoá L2 mọi hàng trước `dot`, nên mọi similarity ∈ [-1,1]; mask luôn xếp cuối. Giá trị thô `-2e11` sụp về `-1` sau chuẩn hoá (đã chạy để xác nhận). Guard hữu hạn vẫn giữ như phòng thủ nhiều lớp, và được ghi rõ là *defensive*. |
| "Thứ tự tie có thể đổi vì cột mask còn nằm trong khoảng so sánh" | **BÁC BỎ** | Mask xếp cuối, nên thứ tự giữa các ứng viên do chính chúng quyết định. Test với trường tie toàn phần xác nhận reference luôn cuối. |
| "`release_verify` exit 0 khi `ENGINEERING_INCOMPLETE` là lỗi" | **BY_DESIGN** | Chỉ `RELEASE_BLOCKED` mới exit khác 0; trạng thái `INCOMPLETE`/`PENDING` là thông tin cho người vận hành. |
| "`acceptance.sh` SKIP nhóm live khi stack xuống" | **BY_DESIGN**, được nêu tên | SKIP trung thực, không được đọc thành PASS. |

## D. Giới hạn bên ngoài (không phải lỗi phần mềm)

* Không có RTX 5070 Ti, không có checkpoint FashionIQ, không chạy suy luận thật.
* 11/16 model vẫn `AGGREGATE_PARSER_UNAVAILABLE`: cần output thật của evaluator.
* `WORKBENCH_LIMN_PYTHON` chưa đặt trên máy này.

Tất cả đều fail-closed và được nêu tên trong output, không bị che.
