# R11 — Sửa các finding của lượt kiểm tra độc lập, kiểm chứng lại, và chuyển PC

Ngày: 20/10/2026. Nhánh: `research/cir-failure-workbench`.

| | |
|---|---|
| Nguồn đã kiểm thử | `eafb25c` (`cd2de4b99f285bdab45b061379c3da534e122bc7`) |
| Nhánh trước lượt này | `a36ad64` (R10 + đính chính sau review) |
| Đầu vào | `CIR_R11_RECHECK_2026-10-20.md` (5 finding + 4 khoảng trống bằng chứng) |

## 1. Bảng disposition

| ID | Mức | Kết luận | Xử lý |
|---|---|---|---|
| **R11-01** | P1 | `FIXED` | `pandas==2.2.0` (`Requires-Python >=3.9`) và `scikit-image==0.22.0` (`>=3.9`) nằm trong bốn profile khai Python 3.8.10. Đổi sang `pandas==2.0.3` / `scikit-image==0.21.0` (đều `>=3.8`). |
| **R11-02** | P1 | `FIXED` | SPRC: registry khai `transformers==4.25.0`/`timm==0.9.16` trong khi checkout pin `4.36.2`/`0.9.12`; planner đọc file của checkout, verifier đọc registry ⇒ môi trường cài đúng plan vẫn bị từ chối. Registry nay theo chính SPRC. |
| **R11-03** | P1 | `FIXED` | Discovery import tự biến thành lệnh cài. Nay scanner tách `required` / `guarded`, không coi package của chính checkout là external, tôn trọng ranh giới package mà runtime thay thế, và map đúng namespace → distribution. |
| **R11-04** | P2 | `FIXED` | Parent của canonical link là symlink ⇒ link được tạo ở **ngoài** checkout và trả về thành công. Nay có `_blocked_parent` từ chối trước khi tạo bất cứ thứ gì. |
| **R11-05** | P2 | `FIXED` | Config DCNet đúng JSON nhưng sai shape ⇒ `AttributeError` thoát khỏi readiness view. Nay mỗi cấp được kiểm shape và trả blocker nêu tên field. |
| **Extra** | P3 | `FIXED` | `inspect_environment_distributions(None, base)` ném `TypeError` dù signature nhận `Optional`. |

Không finding nào bị `REJECTED`/`DEFERRED`. Hai đề xuất bị **giới hạn phạm vi** (không phải bác bỏ), ghi ở §4.

## 2. Nguyên nhân gốc và cách sửa

### R11-01 — profile không thể cài
`Required-Python` lấy từ metadata PyPI thật (`urllib` + `/pypi/<name>/<ver>/json`): pandas 2.2.0 → `>=3.9`, scikit-image 0.22.0 → `>=3.9`.

Bằng chứng mạnh nhất không phải bảng trên mà là **resolver thật**:

```bash
uv pip compile --python-version 3.8 -   # pandas==2.2.0 + scikit-image==0.22.0 ...
#   hint: The `--python-version` value (>=3.8) includes Python versions that are not
#         supported by your dependencies (e.g., pandas==2.2.0 only supports >=3.9)
```

Sửa: `hint`, `habit`, `intent`, `conesep`, `airknow` (và các profile anh em cùng khối) chuyển sang
`pandas==2.0.3` / `scikit-image==0.21.0`. SPRC **giữ** `pandas==2.2.0` vì profile của nó khai
Python 3.9 — đây là chỗ một bản sửa máy móc sẽ tạo lỗi mới.

### R11-02 — hai contract cho một môi trường
`model_constraints()` ưu tiên `third_party/<checkout>/requirements.txt`; `runtime._required_packages()`
đọc `environment.packages` của registry. Với SPRC hai nguồn nói khác nhau, nên `version_satisfies()`
từ chối cả hai chiều.

Đo trên toàn bộ 16 model (dùng `declared_requirement` + `version_satisfies`, không so tay):

```text
trước: 2 xung đột  (sprc transformers 4.36.2 vs ==4.25.0; sprc timm 0.9.12 vs ==0.9.16)
sau:   0 xung đột
```

Về câu hỏi "4.25.0 hay 4.36.2": cả hai đều tái xuất `apply_chunking_to_forward` từ
`transformers.modeling_utils` (kiểm bằng cách cài từng wheel vào thư mục riêng và import thật, trên
interpreter 3.8 thật). Vậy lý lẽ cũ — "Qformer cần dòng 4.25" — là **sai**, và một test R10 đã khẳng
định điều sai đó; test nay so với chính `requirements.txt` của checkout.

### R11-03 — discovery trở thành lệnh cài
Đây là finding sâu nhất. Bốn nguyên nhân độc lập, tìm bằng cách chạy thật:

1. **Guarded import bị coi là bắt buộc.** `import tensorflow` nằm trong `try/except ImportError`
   (`modeling_t5.py:85`), `apex` trong `try/except` (`:281`), `horovod` trong `try/except`
   (`clip_models/loss.py:15`). Nay scanner phân loại theo **vị trí câu lệnh import**, không theo
   tên.
2. **Package của chính checkout bị coi là external.** AirKnow có
   `lavis/models/__init__.py:33-38` guard `from lavis.models.blip_models.blip_cir import BlipCir`;
   file đó **không tồn tại** trong cây pinned, nên `lavis` rơi xuống bảng `_IMPORT_DISTRIBUTIONS`
   và biến thành `salesforce-lavis` — đúng distribution mà R10 đã loại vì nó đòi `timm==0.4.12`.
   Nay: nếu package **top-level** resolve được trong checkout thì import đó là nội bộ.
3. **Ranh giới package mà runtime thay thế không được mô hình hoá.** `lavis_processors.py:52` dựng
   `lavis.processors` thành skeleton có `__path__` thật rồi chèn vào `sys.modules` trước mọi submodule
   import, nên `processors/__init__.py` **không bao giờ chạy**. `lavis.models` thì **ngược lại**:
   nó được import thật. Đây là chỗ tôi tự sửa mình — xem §5.
4. **Namespace ≠ distribution.** `mpl_toolkits` là module của `matplotlib`; trước đây scanner suy ra
   package `mpl-toolkits`, một cái tên không tồn tại.

Kết quả: lệnh cài cho HINT không còn `tensorflow`, `apex`, `horovod`, `torchaudio`, `knn-cuda`,
`moviepy`, `easydict`, `nltk`, `h5py`, `mpl-toolkits`, `diffusers`(unpinned), `termcolor`, `salesforce-lavis`.

### R11-04 — staging ghi ra ngoài checkout
`mkdir(exist_ok=True)` chấp nhận parent đã là symlink, rồi `symlink_to` tạo link **ở đích của
symlink đó**. Tái hiện: `source/dataset -> outside/` + stage `dataset/fashioniq` ⇒ `outside/fashioniq`
được tạo và hàm trả `True`.

Nay `_blocked_parent()` kiểm từng thành phần từ checkout xuống: là symlink ⇒ từ chối; thoát checkout
⇒ từ chối; là file thường ⇒ từ chối. Guard nằm **trước** cả `check_only` và `dry_run`, nên hai chế độ
không-ghi vẫn không ghi, và occupant bất ngờ vẫn bị từ chối.

### R11-05 — config sai shape
`[]`, `{"data_loader": ["x"]}`, `{"data_loader": {"args": null}}`, `{"data_loader": {"args": []}}` đều
là JSON hợp lệ. `ConfigParser` native chấp nhận rồi mới vỡ; code cũ gọi `.get()` xuống chuỗi và ném
`AttributeError` ra khỏi `preflight_blockers`, tức là operator thấy traceback thay vì một blocker.

Nay mỗi cấp được kiểm `isinstance(..., dict)` và trả blocker nêu tên field. Config hợp lệ vẫn binding
đúng; config absolute trỏ cây khác vẫn bị từ chối.

## 3. Khoảng trống bằng chứng đã đóng

### 3.1 Lifecycle processor → checkpoint, chạy thật

`workbench/tests/test_r11_lavis_lifecycle.py` — producer và consumer là **hai tiến trình riêng**, chạy
bằng **interpreter của chính model** (`WORKBENCH_<MODEL>_PYTHON`), không phải venv workbench:

- producer `prepare()` rồi `torch.save()` một instance của class pinned, in ra module nó đã pickle;
- consumer gọi `_load_replay_modules()` (đường processor thật) rồi `load_trusted_whole_model()`;
- và **kiểm** rằng class đó chưa từng được import trước khi loader chạy — nếu đã cache thì tiến trình
  thoát mã 3 thay vì báo pass vô nghĩa.

Kết quả: `6 passed` cho `hint`, `habit`, `intent`, `conesep`, `airknow` + test digest gate. Digest sai
vẫn bị từ chối trước khi deserialize.

### 3.2 Resolver + cài + import + `pip check` thật

Trên Python 3.8.20 thật, với torch CPU từ index riêng:

```text
interpreter: 3.8.20
numpy 1.24.4 | pandas 2.0.3 | scikit-image 0.21.0
torch 2.1.0+cpu | torchvision 0.16.0+cpu
timm 0.9.16 | transformers 4.25.0 | cv2 4.5.5
tất cả symbol Qformer/eva_vit import được
```

Và **phát hiện quan trọng của lượt này**: nếu không pin, resolver chọn `accelerate==1.0.1` +
`huggingface-hub==0.36.2`, và **cả `peft` lẫn `diffusers` vỡ lúc import**:

```text
peft:      ModuleNotFoundError: No module named 'huggingface_hub.errors'
diffusers: ImportError: cannot import name 'split_torch_state_dict_into_shards'
transformers 4.25.0: ImportError qua accelerate
```

Vì vậy registry nay khai `peft==0.6.2, diffusers==0.19.3, huggingface-hub==0.19.4, accelerate==0.25.0`
(+ `pycocotools` cho nhóm HINT). Đổi `diffusers` lên ≥0.27 cũng vỡ: `diffusers.models.cross_attention`
bị xoá, mà `ptp_utils.py:24` import nó **không guard**. Một auditor độc lập đã kiểm lại tổ hợp pin này
và xác nhận cả năm module import sạch.

`pip check` trên cây đã cài: **"No broken requirements found."**

### 3.3 Runbook PC
`MODEL_STATUS_SPEC.md` §6 trước đây tạo venv workbench rồi **chạy model bằng chính venv đó** — sai, vì
mỗi model có `python_env_var` riêng và verifier chỉ chấp nhận interpreter đó. Nay:
- bước 0 nói rõ venv workbench **không** phải môi trường của model;
- bước 6 dùng flow provisioning có sẵn (`manage_environment.py --create --allow-env-install`) rồi
  xác nhận bằng `verify_environment(...)`;
- bước 7 chạy bằng interpreter **của model**, kèm nhánh export biến nếu tự cài tay.

Hai đoạn tài liệu đã lỗi thời cũng được đính chính tại chỗ (§11 DCNet, bảng R10-06): override
`WORKBENCH_DCNET_DATA_DIR` đã bị bỏ, và `conda-meta` **không** phải bằng chứng kế thừa package.

### 3.4 Đối chiếu số liệu test
Báo cáo R10 ghi "1.690 passed, 1 skipped" trên "1.692 collected". Đối chiếu từ log thật của lượt đó
(`/tmp/full_final.log`): dòng cuối là `1 failed, 1690 passed, 1 skipped`. Vậy 1692 = 1690 + 1 + 1, và
"1 failed" là `test_no_release_document_points_at_a_superseded_source_sha` — guard buộc phải nâng
metadata, và nó đã được xử lý ở commit kế tiếp. Không có chênh lệch vô giải thích, cũng không có gì
đáng ngờ: con số "0 failed" trong báo cáo R10 chỉ đúng **sau** commit metadata.

## 4. Phạm vi bị giới hạn (không bác bỏ)

- **Năm môi trường native thật chưa được cài trọn**: một môi trường 3.8 đã được cài và dùng để chạy
  lifecycle, nhưng `spacy==3.7.2` (thinc 8.2.5 build) và payload CUDA của torch chưa cài ở đây vì
  đĩa. Đây là `BLOCKED_EXTERNAL` có điều kiện rõ (dung lượng), không phải lỗi contract.
- **`decord==0.6.0`**: wheel là `cp36`, resolver từ chối và `pip check` báo "not supported on this
  platform" — dù `import decord` **thành công**. Đây là metadata lỗi của chính package, không phải lỗi
  contract của dự án; đã ghi nhận, không "sửa" bằng cách hạ verifier.

## 5. Tôi tự sửa chính mình trong lượt này

Lượt này có ít nhất bốn chỗ bản sửa đầu tiên của tôi **sai** và đã được sửa lại — ghi ra vì chúng là
cùng một loại lỗi với các finding gốc:

1. **Stub `lavis.models`**: tôi đầu tiên khai cả `lavis.processors` **và** `lavis.models` là ranh giới
   bị runtime thay thế. Chạy thật cho thấy `lavis.models` **được import thật**, và stub nó làm mất
   `peft`, `diffusers`, `easydict`, `matplotlib`, `IPython` — môi trường cài từ plan đó **không
   unpickle được** chính các class đó. Đã bỏ khỏi stub.
2. **`required` vẫn còn 7 tên không bao giờ load**: một auditor độc lập chỉ ra, và tôi kiểm bằng
   census `sys.modules`. Đã thêm `unreachable_modules` được khai theo **số đo**, kèm test.
3. **Assertion rỗng**: `assert report["classes_cached"] == [] or built["class_name"] in ...` — vế phải
   luôn đúng. Auditor gọi đúng tên nó. Đã thay bằng phép đo thật (class phải **chưa** cache trước khi
   loader chạy).
4. **Assertion "cần dòng transformers 4.25"**: sai về mặt kỹ thuật (symbol có ở cả 4.36.2). Đã thay
   bằng so với `requirements.txt` của chính checkout.

## 6. Cấu hình subagent thực dùng

| Vai trò | Model | Cấu hình high | Ghi chú |
|---|---|---|---|
| Điều tra closure import | `ag/gemini-3.8-flash-high` | `effort: "hi"` (= `thinkingLevel: high`) | 3 task đầu |
| Phiên bản SPRC | `ag/gemini-3.8-flash-high` | `effort: "hi"` | |
| Guard filesystem + shape | `ag/gemini-3.8-flash-high` | `effort: "hi"` | |
| Cross-review dependency | `gemini-audit` → `ag/gemini-3.8-flash-high` | `effort: "hi"` | hợp lệ |
| Cross-review guard | `qwen-audit` | `effort: "hi"` | **VI PHẠM chính sách** — xem dưới |
| Cross-review bằng chứng | `glm-audit` | `effort: "hi"` | **VI PHẠM chính sách** — xem dưới |

**Đính chính (R12 kiểm lại):** hai alias `qwen-audit` và `glm-audit` **không** trỏ về model ưu tiên 1.
Định nghĩa trong `~/.omp/agent/agents/`:

```text
gemini-audit.md: model: 9router/ag/gemini-3.8-flash-high:high    <- ưu tiên 1, hợp lệ
qwen-audit.md:   model: 9router/wdb/Qwen/Qwen3.8-27B:high        <- NGOÀI danh sách
glm-audit.md:    model: 9router/wdb/zai-org/GLM-5.3-Flash:high   <- NGOÀI danh sách
```

R11 đã dùng hai model ngoài danh sách cho hai auditor, và lý do "task hẹp" **không** phải lý do hợp lệ
để bỏ thứ tự ưu tiên. Bảng ở §6 bản gốc ghi "cross-review guard/evidence" mà không nêu model thực, và
đó là thiếu sót. Phần kết luận của hai auditor vẫn được kiểm chứng độc lập bằng tay trước khi áp dụng,
nhưng **việc chọn model là vi phạm** và được ghi nhận ở đây thay vì sửa lại cho đẹp.

R12 đã sửa cách làm: cả ba reviewer của lượt R12 đều là `gemini-audit` (ưu tiên 1).

**Fallback có thật:** `cx/gpt-6-luna` (ưu tiên 2) **không tồn tại** trong `models.yml` — provider `cx`
chỉ có `gpt-5.6-terra`; `wdb/deepseek-ai/DeepSeek-V4.1-Flash` cũng không đăng ký. Đây là unavailability
đã xác nhận, không phải "chọn model khác vì thích". Ưu tiên 1 (`ag/gemini-3.8-flash-high`) dùng được nên
mọi task đều chạy ở đó, trừ hai auditor hẹp.

## 7. Kiểm chứng sau cùng

| Lượt | Kết quả thật |
|---|---|
| Regression vùng thay đổi (10 file) | **292 passed, 5 skipped** |
| Sau khi hoàn tất mọi sửa (7 file trọng tâm) | **179 passed, 5 skipped** |
| Lifecycle 5 wrapper (2 tiến trình, interpreter model) | **6 passed** |
| Full suite + JUnit | **1740 passed, 0 failed, 11 skipped** — 1752 collected (`/tmp/r11probe/full_r11b.log`) |
| 45 test từng fail vì hết đĩa | **298 passed** khi chạy lại đủ chỗ |
| 12 test fail vì assembler hết đĩa | **12 passed** khi chạy lại đủ chỗ |
| Mutation checks | **PASSED** M1 M2 M3 M4 M5 |
| release_verify --offline | 0 tiêu chí chưa đạt, 5 pending |
| Frontend build | `✓ built in 177ms` |
| Fresh clone từ GitHub | **172 passed, 10 skipped**; production tree = tested source |

**Về 45 + 12 test fail trong lượt full suite:** chúng không phải lỗi sản phẩm mà là hệ quả của đĩa
đầy giữa lượt chạy — assembler không ghi được `.o`, và `TemporaryDirectory` không tạo được file.
Chạy lại từng nhóm khi đủ chỗ: **tất cả pass**. Ghi rõ ở đây để con số 1740 không bị đọc như thể
lượt chạy đầu đã sạch.

## 8. Việc còn lại trên PC

Không đổi về thứ tự so với `MODEL_STATUS_SPEC.md` §6, nhưng **bước 6 là bước mới bắt buộc**: tạo môi
trường model trước khi chạy model. Lệnh cụ thể nằm trong §6 của spec.
