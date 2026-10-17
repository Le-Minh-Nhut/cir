# CIR WORKBENCH — FINAL DEFECT LEDGER

Cây đóng băng: **`5726d5f`** (baseline) → các thay đổi của lượt này trên `research/cir-failure-workbench`.
Mọi mục `CONFIRMED_BUG` dưới đây đều có đường dẫn, điều kiện kích hoạt, hành vi thực tế, hành vi mong đợi
và bài kiểm thử hồi quy đi kèm. Mục nào chỉ là giả thuyết thì nằm ở `UNVERIFIED_RISK`, không được nâng
thành bug.

## A. Bug đã xác nhận và đã sửa trong lượt này

| ID | Mức | Vị trí | Nguyên nhân gốc | Sửa | Kiểm thử hồi quy |
|---|---|---|---|---|---|
| **F-01** | P0 | `workbench/registry/paper_benchmarks/papers.yaml` (2 dòng `ptha_mtst`), `extracted/ptha_mtst_scaleup_mtst.json` | Hai dòng benchmark gán `gallery_protocol: fashioniq_original_split` + `reference_excluded: false` cho một bài báo **không hề nêu** cách dựng gallery hay chính sách reference. Bằng chứng được viện dẫn là `validate_blip.py:51` của repo tác giả — repo này **chưa được sync/pin/audit** (`models.yaml`: `upstream_commit_sha: null`, `source_available: false`). Đây là suy luận từ một file chưa đọc, và nó **mâu thuẫn với chính entry registry** (`native_protocol: null`). | Đặt `gallery_protocol: null`, `reference_excluded: null`; viết lại `verification_evidence` nói rõ protocol là UNKNOWN và vì sao bằng chứng cũ không hợp lệ. | `test_an_unresolved_reference_protocol_never_wildcard_matches_a_run` |
| **F-02** | P0 | `workbench/backend/benchmark_references.py` `matches()` | `gallery_protocol is None` được coi là **wildcard**: một dòng không ai xác lập protocol sẽ khớp với **mọi** protocol. Hệ quả: run bị gán một reference mà protocol chưa từng được chứng minh, và có thể so sánh hai candidate universe khác nhau. | Giá trị vắng mặt **không khớp**; dòng vẫn được báo cáo (không giấu), chỉ không được chọn làm reference. Thêm `unresolved_protocol_rows()` để refusal nói đúng sự thật ("publish metrics but declare no protocol") thay vì "không có dòng nào". | `test_an_unresolved_reference_protocol_never_wildcard_matches_a_run`, `test_a_legitimate_ptha_run_is_refused_because_no_protocol_was_ever_established`, `test_the_verifier_never_claims_no_row_publishes_a_metric_when_one_does` |
| **F-03** | P1 | `workbench/backend/runtime.py` `_gpu_cache_key` vs `workbench/scripts/evaluate_models.py` `ledger_hmac_key` | Hai bộ phân giải khoá ledger cùng tồn tại với **ưu tiên khác nhau**: `runtime` chọn `<repo>/.workbench_ledger_key` khi repo ghi được, còn `ledger_hmac_key` ưu tiên `$XDG_CONFIG_HOME/cir/...`. Trên host repo ghi được, mỗi bên tạo một file khoá **khác nhau** → lượt sau chết với `two different ledger keys exist`. Nguy hiểm hơn: bản ghi capability được ký bằng khoá mà bên đọc ledger sẽ từ chối. | `runtime._gpu_cache_key` **uỷ quyền** cho `ledger_hmac_key` (chủ sở hữu duy nhất, vì nó có yêu cầu mạnh hơn: ledger ký bằng khoá kia sẽ không kiểm chứng được), in nguyên văn thông điệp từ chối, trả `b""` = "không tin được" thay vì "đã ký". | `test_the_capability_key_and_the_ledger_key_never_disagree`, `test_the_gpu_capability_key_reports_the_same_conflict`, `test_the_fallback_key_is_created_so_a_capability_can_be_signed` |
| **F-04** | P1 | `workbench/scripts/pc_readiness.py` `classify()` / `BLOCKER_KINDS` | Bảng needle được duyệt **theo thứ tự vị trí**: `"FashionIQ"` đứng trước `"checkpoint missing"`, nên `"checkpoint missing: /data/FashionIQ/fashioniq_dcnet"` bị phân loại `BLOCKED_DATASET` — bảo người vận hành đi sửa dataset trong khi checkpoint mới là thứ thiếu. Hai audit model độc lập cùng tái hiện. | Xếp lại theo **ưu tiên ngữ nghĩa**: mọi cụm *nguyên nhân* đứng trước các danh từ *vị trí/vague*; xét **toàn bộ danh sách** blocker (không phải blocker đầu tiên khớp) để kết quả không phụ thuộc thứ tự thông điệp của preflight. | `test_classify_is_not_decided_by_where_a_word_sits_in_the_table`, `test_classify_is_order_independent_across_the_blocker_list`, `test_classify_reports_the_blocker_that_decided_the_state` |
| **F-05** | P1 | `workbench/scripts/validate_results.py` | Validator không kiểm tra **tính nhất quán của danh tính**: `run_id` được dẫn xuất từ provenance, nhưng không ai tính lại. Một artifact bị sửa `run_id` sang của run khác vẫn **hợp lệ schema** và consumer khoá theo `run_id` sẽ gán số của run này cho run kia. | Thêm `verify_run_identity()` (tính lại `run_id` từ chính provenance) và gọi ở cả `publish()` lẫn `validate_file()`. Chỉ áp dụng cho `data_kind == "experiment"` — mock/synthetic là dữ liệu phát triển viết tay, và nó không bao giờ trở thành bằng chứng. | `test_a_relabelled_run_id_is_refused_for_an_experiment`, `test_swapped_checkpoint_identity_changes_the_run_id`, `test_publication_refuses_an_inconsistent_identity`, `test_a_mock_artifact_is_not_held_to_derived_identity` |

## B. Lỗi thượng nguồn đã xác nhận (không sửa được trong workbench — ghi nhận chính xác)

Các mục này là defect trong **source của tác giả**, được audit viên tái hiện, và workbench phải
fail-closed quanh chúng. Không sửa source thượng nguồn.

| ID | Model | Vị trí | Hành vi |
|---|---|---|---|
| **U-01** | `ptha_mtst` | repo tác giả `src/lavis/models/__init__.py:54` | Import module không tồn tại (`blip2_qformer_tgir_concatembedding_sim_zyn`) — module vắng trong **cả 8 commit**. Evaluator chết ngay khi import. |
| **U-02** | `encoder` | `evaluate_model.py:7` | `import datasets1` nhưng file **không tồn tại** trong cả checkout lẫn upstream. Dùng `datasets.py` thay thế là suy luận chưa được chứng minh. |
| **U-03** | `pair` | `models/model.py:13` | `import MHTransformer_rev` — file vắng trong **cả 21 commit**. |
| **U-04** | `airknow` | `lavis/models/__init__.py:33-34` | Import `blip_cir.py`, `blip_cir_caption.py` không được commit. |
| **U-05** | `csmcir` | `src/lavis/models/__init__.py:44-151` | 98 import không được bảo vệ làm hỏng package. (Workbench có **patch tương thích đã duyệt**; patch chỉ về khả năng import, digest base/patch/derived đều được ghim.) |
| **U-06** | `sprc` | `src/blip_validate.py:144`, `src/validate_blip_rerank.py:17` | Tên model mặc định không được đăng ký trong LAVIS; `validate_blip_rerank.py` import module `combiner` không tồn tại. |
| **U-07** | `hint`, `habit`, `conesep`, `intent`, `airknow`, `dcnet` | `test.py`/`train.py` | **Không có entrypoint CLI đánh giá**. `--restore_file` là dead code. |
| **U-08** | `combiner_rn50x4_noft`, `clip4cir_rn50x4_fullft`, `sprc`, `csmcir` | `data_utils.py` | Hardcode đuôi `.jpg`; dataset canonical trên máy này có `.png`. |
| **U-09** | tất cả model iLearn | `datasets.py` | Ghép chuỗi thiếu dấu `/` (`self.path + 'resized_image'`) → crash nếu path không có trailing slash. |
| **U-10** | `dcnet` | `preprocess/process_cap.py:29-31` | Vector OOV lấy `np.random.normal` **không seed** → PKL không tái lập được; digest thay đổi mỗi lần chạy. Workbench từ chối tự sinh lại PKL của tác giả (`AUTHOR_ARTIFACT_PREFERRED`). |
| **U-11** | `clvc_net` | `text_model.py:66`, `resnet.py:3` | `simplevocab.json` vắng; `torchvision.models.utils` đã bị xoá ở torchvision hiện đại. |
| **U-12** | `tgcir` | `cluster_b2_encoder_family.yaml:432,434` | URL/digest ViT-B/16 hỏng (404). |
| **U-13** | `airknow` | `test_BLIP2.py:10` | FashionIQ eval chạy dưới `model.train()` → dropout hoạt động trong suy luận. |
| **U-14** | `limn`, `tgcir`, `encoder`, `hint`, `pair`, `clvc_net`, `airknow`, `intent`, `habit`, `conesep` | `datasets.py` | Dataset ghi cache `.pkl` vào chính `self.path` → chết trên mount read-only. (Workbench đã cách ly qua `isolated_cache_root`.) |

## C. Sai lệch số liệu công bố (báo cáo, không đoán)

| Model | Vấn đề | Bằng chứng |
|---|---|---|
| `sprc` | Bảng 1 in Average R@10 = **54.92**, nhưng trung bình của ba category in ngay trên đó là **54.7233**. Con số 54.85 ở cột Avg. khớp với `round((54.7233+74.9667)/2)`, chứng tỏ ô 54.92 là lỗi đánh máy của bài báo. Thêm nữa, phần prose gán 64.85 cho `SPRC2` trong khi bảng gán cho `Ours (SPRC)`. | `sprc.md`; không được resolve bằng cách đoán |
| `combiner_rn50x4_noft` | Bài TOMM 2023 **không có dòng nào** cho RN50x4 noft. Số CVPR 2022 Demo (35.39/59.03) không được đăng ký trong `papers.yaml`. | `combiner_rn50x4_noft.md` → `clip4cir_rn50x4_noft_fiq_absent_in_paper`, metrics null |
| `ptha_mtst` | Hai dòng verified (54.99/76.28 và 54.33/75.24) — dòng thứ hai là **control row** không pretrain | sau F-01/F-02 cả hai đều không thể bind |

## D. Bác bỏ sau khi kiểm chứng (`REJECTED_FINDING`)

| Cáo buộc | Kết luận | Bằng chứng |
|---|---|---|
| "`models.yaml` mô tả sai shape matmul của Air-Know/INTENT/ConeSep/HABIT" | **BÁC BỎ** | NumPy broadcast `(N,1,1,256) @ (G,256,32)` → `(N,G)` đúng như thiết kế. Audit viên đã verify bằng cách chạy. |
| "Bộ đệm capability GPU có thể bị tái sử dụng cho máy đã đổi" | **BÁC BỎ** | `_gpu_capability_key()`/`_host_identity()` bind node, arch, driver version, visible devices và torch contract; đổi host/driver là mất hiệu lực. |
| "Xác thực viên từ chối artifact có `run_id` viết tay (mock) là bug" | **BÁC BỎ / BY_DESIGN** | Fixture mock là dữ liệu phát triển khai báo; `verify_run.py` từ chối mọi report không phải `experiment`, nên danh tính dẫn xuất không bảo vệ thêm gì. |
| "`strict=False` khi load LAVIS là bug" (một phần) | **BY_DESIGN**, có rủi ro ghi riêng | Là hành vi thượng nguồn; workbench bù bằng cách ghim digest checkpoint. Rủi ro "trông hợp lý nhưng sai" được ghi ở `UNVERIFIED_RISK` (`sprc.md`). |
| "Verifier bỏ qua `benchmark_gate.passed` là lỗ hổng" | **BY_DESIGN** | Tính toàn vẹn thực thi được tách khỏi độ khớp điểm paper một cách có chủ ý; cả hai đều hiển thị riêng. |

## E. `UNVERIFIED_RISK` — không phải bug, không được nâng cấp

| Mục | Vì sao chưa xác nhận |
|---|---|
| 11/16 model trả `AGGREGATE_PARSER_UNAVAILABLE` (`airknow`, `conesep`, `ptha_mtst`, `habit`, `intent`, `hint`, `encoder`, `pair`, `clvc_net`, `dcnet`, `tgcir`) | Viết parser cần **output thật** của evaluator, mà không model nào chạy được trên máy này. Viết parser từ phỏng đoán định dạng sẽ tạo ra chính xác loại "xác thực giả" mà dự án đang chống. Được ghi là blocker trung thực, không phải bug đã sửa. |
| `parse_csmcir_output` / `parse_combiner_output` phản chiếu lại số của report (chỉ kiểm tra lệch ≤0.05pp) | Có chủ ý: parser không thể tái dẫn xuất macro vì report là nguồn duy nhất. Chỉ `parse_limn_output` tái dẫn xuất độc lập. Ghi nhận là điểm yếu đã biết, không phải phát hiện mới. |
| `_non_limn_reasons` chỉ cần **một** metric Recall hợp lệ | Một run chỉ có R@10 mà thiếu R@50 vẫn qua được kiểm tra bằng chứng. Chưa tái hiện được đường đi thực tế tạo ra report như vậy. |
| Dataset root + `captions/` mount `rw` trong compose | Chỉ `/images` và `/image_splits` là `ro`. Đã có `isolated_cache_root` bảo vệ đường ghi cache, nhưng mount rộng hơn mức cần thiết. |
| `release_verify.sh` exit 0 khi `ENGINEERING_INCOMPLETE`/`GPU_VALIDATION_PENDING` | Là thiết kế có chủ ý (chỉ `RELEASE_BLOCKED` mới exit 3), nhưng nghĩa là CI có thể xanh mà chưa chạy mục tiêu phát hành. Cần quyết định của người vận hành, không phải sửa ngầm. |
| `acceptance.sh` bỏ qua nhóm live khi stack xuống và vẫn exit 0 | Cùng loại: SKIP trung thực, nhưng không nên đọc thành PASS. |
