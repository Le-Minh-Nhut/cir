# R9 — trạng thái sau review độc lập

Hai review độc lập đã chạy trên cây đang sửa (`R9_ADVERSARIAL_FIXES.md`, `R9_INTEGRATION_REVIEW.md`).
Cả hai **tìm ra lỗi thật**. Dưới đây là từng phát hiện và cách xử lý, kèm bằng chứng chạy lại.

| # | Phát hiện | Nguồn | Xử lý | Bằng chứng |
|---|---|---|---|---|
| 1 | `preflight_blockers` gọi `binding_mismatch` **thiếu `evaluator_cwd=source`** ⇒ resolve theo CWD của orchestrator và **từ chối oan** symlink canonical đặt đúng trong checkout | `R9_INTEGRATION_REVIEW` #1, `R9_ADVERSARIAL_FIXES` claim 1 | truyền `evaluator_cwd=source` | `test_the_binding_refusal_reaches_the_shared_preflight_list` |
| 2 | `dataset_root is None` ⇒ check DCNet **bỏ qua** ở `pc_readiness`, `doctor`, `prepare_model` | `R9_INTEGRATION_REVIEW` #2, `R9_ADVERSARIAL_FIXES` claim 1 | `requested` mặc định là `config.FASHIONIQ_ROOT`; điều kiện chỉ còn phụ thuộc `source` | `python workbench/scripts/pc_readiness.py --json` — 16 model, không blocker binding giả |
| 3 | `encoder.py` **không** được nối vào `constructor_input_blockers` ⇒ root dress-only vẫn pass rồi chết ở `cap.shirt.train.json` | `R9_INTEGRATION_REVIEW` #3, `R9_ADVERSARIAL_FIXES` claim 4 | nối ENCODER (7/7 wrapper) | `.venv/bin/python -c "from workbench.replay import encoder; print(len(encoder.validate_dataset(Path('/tmp/ctor2'), ('dress',))))"` → **10** blocker |
| 4 | Bốn khai báo `fashioniq_layout` (`habit`, `intent`, `conesep`, `airknow`) **bỏ `cap.*.train.json`** trong khi constructor native vẫn mở ⇒ layout check mâu thuẫn runtime check | `R9_ADVERSARIAL_FIXES` claim 4 | nâng lên `fashioniq_ilearn_resized_training` | `paths_for_layout` nay trả 3 train path cho cả 7 model |
| 5 | Target là **symlink** ⇒ `sha256_file` raise `ValueError` thoát khỏi readiness report | `R9_ADVERSARIAL_FIXES` claim 3 | nhánh tường minh, trả `applied=False` kèm lý do | kiểm bằng thực thi: `A symlink target -> False` |
| 6 | Khai báo có `base_sha256 == derived_sha256` ⇒ file **base** bị coi là `applied=True` | `R9_ADVERSARIAL_FIXES` claim 3 | nhánh tường minh, coi là **lỗi registry** | `B degenerate digests -> False` |
| 7 | Test "no second model" chỉ đọc **tên** hàm ⇒ `from lavis.models import load_model_and_preprocess as loader` lọt qua | `R9_ADVERSARIAL_FIXES`, mục "Tests Passing Despite Defect Presence" | siết: so **module** được import và mọi attribute/literal | kiểm 4 dạng bypass: alias/module-import/importlib đều bị bắt, dạng sạch không |
| 8 | `MODEL_STATUS_SPEC.md` trỏ `benchmarks/lavis_processor_parity.py` và `workbench/docs/reviews/R9_FINAL_REPORT.md` (không tồn tại) | `R9_INTEGRATION_REVIEW` #5 | sửa đường dẫn | `grep -rn "benchmarks/lavis_processor_parity" workbench/` → chỉ còn trong chính báo cáo review |
| 9 | `conesep`/`airknow` ghi "hash VERIFIED" nhưng registry để `expected_sha256: null` | `SS_06_SPEC.md` §2 | sửa nhãn thành "hash **chưa có**" | `model_by_id('conesep')` → `sha=NULL` |
| 10 | `tgcir`/`ptha_mtst` ghi `PREFLIGHT_PASS` dù đang bị chặn đúng | `SS_06_SPEC.md` §7 | nhãn mới `PREFLIGHT_IMPLEMENTED_BLOCKED`, định nghĩa trong bảng §1 | `MODEL_STATUS_SPEC.md:27` |
| 11 | Handoff còn nói "10 model cần wrapper" trong khi `ADAPTERS` có 15 và `replay/` có 16 wrapper | `SS_06_SPEC.md` §5 | sửa handoff, nêu lý do thật theo từng model | `PC_PILOT_HANDOFF.md:15-23` |
| 12 | `acceptance.sh`: nhóm C fail vì backend đang chạy là **container cũ**, nhóm A-runs/I-gate fail vì **ledger cục bộ hỏng**, không liên quan repo | điều tra của lead | chứng minh bằng fresh clone + backend từ source hiện tại | fresh clone: **ACCEPTANCE PASSED** |

## Điều review **không** bác bỏ được

`R9_ADVERSARIAL_FIXES` thử bác bỏ claim 2 (thứ tự category) và claim 5 (processor) bằng: thứ tự lặp,
thứ tự trùng, thiếu category, config kiểu tuple/None/khác tập, gọi `text_processors` hai lần, stub rò
vào `sys.modules` — **không** dạng nào bác bỏ được. Cả hai review xác nhận không có thay đổi ngữ nghĩa
khoa học của evaluator, và reproduction transpose của `DC_02_PARSER.md` nay trả đúng giá trị từng
category.

## Một điểm còn treo, ghi rõ

`lavis_processors.py:_load_preprocess_function` có hai nhánh: import thật `lavis.models`, và compile
`load_preprocess` từ chính source pinned khi import đó thất bại. Trên host này **chỉ nhánh thứ hai**
chạy được (`lavis.models` cần `transformers.modeling_utils.apply_chunking_to_forward`, bản
`transformers` ở đây mới hơn bản pinned). `R9_INTEGRATION_REVIEW` gọi nhánh thứ nhất là dead code.

Giữ cả hai là **có chủ ý**: trên runtime khớp environment khai báo (`transformers==4.25.0`) nhánh
import thật chạy, và nó là đường chính — nhánh compile là fallback cho host lệch version. Xoá nhánh
import sẽ biến một fallback thành đường duy nhất, tức khai báo rằng bản pinned không bao giờ import
được, trong khi điều đó chỉ đúng ở host này. Nhánh nào đã chạy được **ghi lại** trong provenance mỗi
run (`text_processor.loaded_via`), nên không có chỗ nào che mất biên đó.
