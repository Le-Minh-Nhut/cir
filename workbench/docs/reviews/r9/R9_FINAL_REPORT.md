# R9 FINAL REPORT — bốn lỗi tái hiện được, F06, và phần kiểm chứng độc lập

Lượt này bắt đầu từ `914703ecd42dbb6a0054bc8a5811dcb4dae88cbd` (đầu nhánh
`research/cir-failure-workbench`, không có thay đổi chưa commit). Bốn lỗi đã được **tái hiện bằng
thực thi** trước khi sửa, và mỗi lỗi có một test hồi quy chạy trên entry point production.

Sáu điều tra đọc-only và hai review độc lập nằm cạnh báo cáo này:

| File | Nội dung |
|---|---|
| `DC_01_BINDING.md` | đường launch thật của DCNet, CWD của evaluator, `data_dir` resolve ở đâu |
| `DC_02_PARSER.md` | tái hiện transpose category + chứng minh `comb_avg` không phát hiện được |
| `CS_03_PATCHSET.md` | `approved_patch_state` return trong vòng lặp; chỉ CSMCIR có >1 patch |
| `DX_04_CONSTRUCTOR.md` | constructor native mở gì, wrapper nào kiểm gì, production caller nào truyền subset |
| `RS_05_TESTS.md` | test nào chỉ kiểm bản sao; danh sách test vẫn pass khi lỗi quay lại |
| `SS_06_SPEC.md` | spec vs registry vs code; claim cũ trong handoff |
| `R9_ADVERSARIAL_FIXES.md` | review đối kháng năm claim của lượt này |
| `R9_INTEGRATION_REVIEW.md` | review tích hợp diff + fresh-clone + status spec |

---

## 1. Bốn lỗi

### A. DCNet dataset binding không nằm trên đường launch

**Tái hiện.** `workbench/replay/dcnet.py:validate_dataset_binding` **không** được gọi từ bất kỳ
entry point production nào: `DCNetAdapter` (`workbench/backend/adapters/models.py:172-178`) launch
`test.py --resume <run_dir>` trực tiếp và không đi qua `_replay_command`, nên `dcnet.preflight` chỉ
chạy khi người vận hành gọi module bằng tay. Bản cũ chỉ từ chối **relative** path, và bất kỳ path
**absolute** nào cũng được chấp nhận — kể cả `/data/fiq-other` trong khi `--dataset-root` là
`/data/fiq-selected`.

**Gốc.** Hai lỗi cùng lúc: (1) kiểm tra nằm trong helper mà production không gọi; (2) luật quá yếu —
`CE_dataset.py:22,27,34` join `data_dir` vào **CWD của tiến trình**, mà CWD đó
(`evaluate_models.py:214-215,594`) là checkout; không có `chdir` nào trong DCNet.

**Sửa.** `configured_data_dir`/`binding_mismatch` resolve theo CWD và **so với `--dataset-root` cho
cả absolute lẫn relative**; symlink canonical (cách preparation stage dataset) pass, thư mục thường
trong checkout hoặc absolute trỏ nơi khác bị **refuse**. Kiểm tra chạy trong
`backend/preflight.py:preflight_blockers` — danh sách mà `doctor`, `pc_readiness`, `prepare_model`,
`evaluate_models` đều đọc — với `evaluator_cwd=source` và `dataset_root` mặc định là canonical root.
Override tường minh `WORKBENCH_DCNET_DATA_DIR` cho operator có CWD khác; file tác giả không bị sửa.
Run directory chép `config.json` vào evidence dir để `verify_run` re-parse được.

**Kiểm.** `test_r9_defect_regressions.py` A1–A8: mismatch absolute, relative resolve đúng, relative
trong checkout, thiếu field, override, đường preflight thật, và `preflight_blockers` trên model dict
thật.

### B. DCNet parser transpose category trong im lặng

**Tái hiện.** Order `toptee, dress, shirt`, `comb: 0.7 0.9 0.2 0.4 0.3 0.5` ⇒ parser trả `dress`
70/90, `shirt` 20/40, `toptee` 30/50, `extraction_status=EXTRACTED_VERIFIED`. Aggregate 40/60, mean
50 — **không** phát hiện được, vì `[comb_avg]` là tổng của sáu ô và tổng bất biến với hoán vị.

**Gốc.** `test.py:74-84` in cả `recall` lẫn `comb` bằng **cùng** một vòng `trainer.categories`
(`base_trainer.py:32` = `config['data_loader']['args']['categories']`). Sáu ô `comb:` không có tên,
và parser map chúng theo tuple cứng `("dress","shirt","toptee")` sau khi lọc theo thứ tự đó.

**Sửa.** Thứ tự lấy từ **chính stream** (`[cat] comp corr r@10` — thứ tự xuất hiện là thứ tự đã in,
và hai block đến từ cùng một vòng lặp). Config là **nhân chứng thứ hai**: nếu khai báo ba category
FashionIQ mà lệch với stream ⇒ `EVALUATION_COMPLETED_METRICS_UNPARSED`; config khai báo tập khác
(ví dụ có thêm `shoes`) thì không nói gì về thứ tự ba category này và không ràng buộc. Đơn vị
fraction→% giữ đúng một lần; composition/correction vẫn tách riêng.

**Kiểm.** `test_r9_defect_regressions.py` B1–B6, gồm cả giá trị từng category theo thứ tự không mặc
định, config lệch, config khác tập, và phép nhân 100 đúng một lần.

### C. CSMCIR chỉ kiểm patch đầu tiên

**Tái hiện.** `approved_patch_state` **return ngay trong vòng lặp** (`preflight.py:140,145,149,154`),
nên checkout chỉ có patch #1 vẫn cho `applied=True`. CSMCIR là model **duy nhất** khai báo nhiều hơn
một patch (2). Không consumer nào phân biệt partial với complete.

**Gốc.** `for patch in patches:` với `return` bên trong — hợp lệ khi mỗi model có nhiều nhất một
patch, và sai ngay khi có hai. Patch #1 (import compat) đủ để package import; patch #2 (visualization
opt-in) mới là cái giữ benchmark shell exit 0 **sau khi** đã in metric JSON.

**Sửa.** Hàm kiểm **mọi** patch đã khai báo, trả thêm `patches[]` theo từng mục, và `detail` nói rõ
"*N* of *M* required patches are not applied". `model_requirements` phát note `source repair
incomplete` cho trạng thái nửa vời; `source_blocker` vẫn còn hiệu lực. Provenance ghi cả tập patch
đã áp. Hai trường hợp biên được xử lý thay vì raise: target là symlink (báo, không crash) và khai
báo có `base_sha256 == derived_sha256` (không được coi là applied). `--visualize` vẫn off mặc định.

**Kiểm.** `test_r9_defect_regressions.py` C1–C6: pristine, chỉ patch #1, chỉ patch #2, cả hai, edit
sau cả hai, và note partial trong `model_requirements`.

### D. Validator một category không phủ input của constructor

**Tái hiện.** `validate_dataset(root, categories=('dress',))` trả `[]` trên root chỉ có dress. Cả
sáu wrapper (và ENCODER) gọi `datasets.FashionIQ(...)`, mà `__init__` chạy `train_init_process()`
rồi `get_test_data()` cho **cả ba** category — không có tham số category nào trong signature. Vì
`isolated_cache_root` luôn cho cache rỗng, nhánh cold **luôn** chạy: checkpoint được load xong mới
chết ở `cap.shirt.train.json`.

**Gốc.** Hai câu hỏi khác nhau bị gộp làm một: "category nào được **chấm**" và "file nào constructor
**mở**". `--category` chỉ trả lời câu đầu.

**Sửa.** `workbench/replay/common.py:constructor_input_blockers` trả lời câu thứ hai, dùng chung
cho bảy wrapper (`hint`, `habit`, `intent`, `conesep`, `airknow`, `pair`, `encoder` — ENCODER do
review tích hợp chỉ ra). Bốn khai báo `fashioniq_layout` (`habit`, `intent`, `conesep`, `airknow`)
được nâng lên `fashioniq_ilearn_resized_training`: layout cũ bỏ `cap.*.train.json` mà constructor
vẫn mở, tức layout check và runtime check mâu thuẫn nhau.

**Kiểm.** `test_r9_defect_regressions.py` D1–D3 (7 wrapper × 3 bài), cộng bài đọc chính source
native để khẳng định tập input, thay vì tin vào danh sách viết trong test.

---

## 2. F06 — processor BLIP-2

**Lỗi.** Năm wrapper (`hint`, `habit`, `intent`, `conesep`, `airknow`) gọi
`load_model_and_preprocess(name=..., model_type=..., is_eval=True, device=...)` chỉ để lấy
`txt_processors`. Hàm đó dựng **model BLIP-2 thứ hai** (`lavis/models/__init__.py:210`) trước khi
đọc preprocess config (`:216-220`), và `Blip2Base.init_tokenizer`/`init_Qformer` đọc
`/root/bert-base-uncased` (`blip2.py:32,48,55`) — cần asset BERT cục bộ và trọng số base chỉ để trả
về một text processor.

**Vì sao trước đây hoãn.** Lượt R8 kết luận `lavis.processors` kéo `torchaudio` nên không kiểm được
trên host này, và mọi assertion sẽ là test một stub. Kết luận đó **không đầy đủ**: import
`torchaudio` là **một dòng chết** trong `lavis/processors/audio_processors.py:9`, chỉ tới được qua
`lavis/processors/__init__.py:29`, và processor không cần gì từ nó. torchaudio không nằm trong môi
trường khai báo của **bất kỳ** model nào ở đây.

**Sửa.** `workbench/replay/lavis_processors.py` lấy processor từ **đúng config mà model class công
bố** (`PRETRAINED_MODEL_CONFIG_DICT`), resolve như `get_abs_path` resolve, đăng ký `library_root`
như `lavis/__init__.py:24`, và chạy `load_preprocess` của chính repo. `BlipCaptionProcessor` là
class pinned thật, `from_config` thật. Chỉ module `audio_processors` bị bỏ khỏi import graph và
`sys.modules` được phục hồi sau lời gọi.

**Biên kiểm chứng (ghi rõ trong docstring và trong provenance mỗi run qua `text_processor.loaded_via`):**
đường `load_model_and_preprocess` **không** chạy được trên host này vì `lavis.models` cần
`transformers.modeling_utils.apply_chunking_to_forward` (bản `transformers` ở đây mới hơn bản pinned),
nên so sánh trực tiếp hai đường là **không** thực hiện được. Cái chứng minh được, và đã chứng minh:
processor trả về là class pinned, và output là hành vi native (`'  A dog, running!  '` →
`'a dog, running'`, `'Dress.'` → `'dress'`, whitespace gộp, dấu câu bỏ). Chạy được:
`workbench/scripts/lavis_processor_parity.py` (rc=0, 5 model).

**Không** cần tạo `/root` trên PC.

---

## 3. Review độc lập tìm ra gì

Hai review chạy sau khi sửa và **đã tìm ra lỗi thật** — đây là lý do tin phần còn lại:

| Nguồn | Phát hiện | Xử lý |
|---|---|---|
| `R9_INTEGRATION_REVIEW` #1 + `R9_ADVERSARIAL_FIXES` claim 1 | `preflight_blockers` gọi `binding_mismatch` **thiếu `evaluator_cwd=source`** ⇒ symlink canonical đúng bị từ chối oan; và `dataset_root=None` làm check **bỏ qua** ở `pc_readiness`/`doctor`/`prepare_model` | sửa: truyền `evaluator_cwd=source`, và `dataset_root` mặc định là canonical root; kiểm **mọi** variant |
| `R9_INTEGRATION_REVIEW` #3 + `R9_ADVERSARIAL_FIXES` claim 4 | `encoder.py` **không** được nối vào `constructor_input_blockers` ⇒ vẫn pass trên root dress-only | sửa: nối ENCODER (7/7 wrapper) |
| `R9_ADVERSARIAL_FIXES` claim 4 | bốn khai báo `fashioniq_layout` bỏ `train.json` trong khi constructor vẫn mở ⇒ layout check mâu thuẫn runtime check | sửa: nâng lên `[..._training]` |
| `R9_ADVERSARIAL_FIXES` claim 3 | target là symlink ⇒ `sha256_file` raise `ValueError` thoát khỏi readiness report; `base==derived` ⇒ base file bị coi là applied | sửa: hai nhánh tường minh, đã kiểm bằng thực thi |
| `R9_ADVERSARIAL_FIXES` test hardening | test "no second model" chỉ đọc **tên** hàm, nên `from lavis.models import load_model_and_preprocess as loader` lọt qua | sửa: so cả **module được import** (`lavis.models`) và mọi literal/attribute; đã kiểm bốn dạng bypass |
| `R9_INTEGRATION_REVIEW` #5 | spec trỏ `R9_FINAL_REPORT.md` và `benchmarks/lavis_processor_parity.py` (không tồn tại) | sửa đường dẫn |
| `R9_INTEGRATION_REVIEW` #5 | spec nhận `doctor`/`pc_readiness`/`prepare_model` chạy check DCNet khi chúng không truyền `dataset_root` | chính là lỗi #1; nay đã đúng |
| `SS_06_SPEC.md` (điều tra) | `conesep`/`airknow` ghi "hash VERIFIED" nhưng registry để `expected_sha256: null` | sửa nhãn trong bảng §2 |
| `SS_06_SPEC.md` | handoff còn nói "10 model cần wrapper" trong khi `ADAPTERS` có 15 và `replay/` có 16 wrapper | sửa handoff |
| `R9_INTEGRATION_REVIEW` #4 | file mới chưa track ⇒ fresh clone không có | giải quyết bằng commit của lượt này |

`R9_ADVERSARIAL_FIXES` **không** bác bỏ được claim 2 (parser) và claim 5 (F06) sau khi thử các
đường falsification: thứ tự lặp, thứ tự trùng, thiếu category, config kiểu tuple/None/khác tập, gọi
`text_processors` hai lần, stub rò vào `sys.modules`. Đó là phần có giá trị nhất của review.

Cả hai review đều xác nhận: **không** thay đổi ngữ nghĩa khoa học của evaluator (gallery, mask
reference, ranking, transform, đơn vị metric, protocol), và reproduction transpose của
`DC_02_PARSER.md` nay trả đúng giá trị từng category.

---

## 4. Test mới thay test chỉ kiểm bản sao

`RS_05_TESTS.md` liệt kê test nào vẫn pass khi lỗi quay lại — chủ yếu là
`test_f05_f08_f09_verification.py`, nơi **định nghĩa lại** validator/parser trong chính file test
(`validate_dataset_with_train`, `validate_dcnet_data_dir_alignment`, `parse_dcnet_native_output`).
`test_r9_defect_regressions.py` gọi thẳng production: `parse_dcnet_output`,
`extract_aggregate_metrics`, `approved_patch_state`, `preflight_blockers`,
`module.validate_dataset`, `common.constructor_input_blockers`, `lavis_processors.text_processors`,
`hint._load_replay_modules`, `EvaluationPlan`/`read_executed_config`.

Đã kiểm **bằng cách revert**: `R9_ADVERSARIAL_FIXES` revert từng sửa trong một bản sao repo và xác
nhận test tương ứng **fail** cho cả bốn lỗi. Riêng bài "no second model" thì lọt qua dạng alias —
đã siết lại và kiểm bốn dạng bypass.

---

## 5. Phạm vi bằng chứng

| Đã chạy thật | Không chạy được ở đây |
|---|---|
| pytest trên suite workbench (kết quả ở commit) | inference trên GPU thật |
| constructor native `datasets.FashionIQ` trên fixture (cold cache) | checkpoint thật của bất kỳ model nào (không có file local) |
| `BlipCaptionProcessor` pinned thật, output thật | `load_model_and_preprocess` (bản pinned cần `transformers` cũ hơn) |
| `patch -p1` thật cho cả hai patch CSMCIR | dataset FashionIQ canonical |
| `git worktree` collection ở revision đã đặt tên | recall / parity khoa học |

Không dòng nào ở đây là "đã chạy model". Mọi số trong registry vẫn là **số paper**, không phải số đo.

---

## 6. Việc còn lại

1. Chạy thật trên RTX 5070 Ti, bắt đầu từ LIMN (3 checkpoint, mapping VERIFIED, có per-query export).
2. Stage asset theo `python workbench/scripts/download_checkpoints.py --all --dry-run` trước.
3. `csmcir`: apply **cả hai** patch rồi kiểm `approved_patch.state.applied == true` trước khi chạy.
4. `dcnet`: stage run directory của tác giả, rồi để preflight xác nhận binding thay vì sửa config.
5. Các blocker tác giả (PAIR B1/B2, CLVC-Net 6 file, TG-CIR archive member, PTHA FashionIQ weight)
   vẫn cần tác giả cung cấp — không có đường kỹ thuật nào thay thế được.
