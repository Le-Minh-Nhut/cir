# R13 — Kiểm tra độc lập, sửa, và chuyển PC

Ngày 21/10/2026. Nhánh `research/cir-failure-workbench`.

| | |
|---|---|
| Baseline đầu lượt | `9658cca` (đúng revision R13 đã review), cây sạch |
| **Tested source SHA** | `87978b0521c1e2990e52b6bdb88b2a63111afcf2` |
| Full suite trên tested source | **1786 collected = 1773 passed + 1 failed + 12 skipped** |
| Failure còn lại | đúng **một**: guard `test_no_release_document_points_at_a_superseded_source_sha`, chỉ pass sau khi metadata trỏ đúng revision này |

## 1. Disposition

| ID | Kết luận | Nguyên nhân gốc | Bằng chứng |
|---|---|---|---|
| **R13-01A** | `FIXED` | `request.partition("@")[2].partition("@")` + `removeprefix("git+")`: với cách viết chuẩn `clip @ git+URL@rev`, URL ra `' git+https://…'` — còn khoảng trắng, còn prefix — nên **nguồn ĐÚNG bị từ chối** | Parser nay structural + chuẩn hoá; 4 cách viết (chuẩn, 2 space, không space, không revision) cho **cùng identity**; cài thật `clip @ git+…@a1d0717` được **accept** |
| **R13-01B** | `FIXED` | `runtime._required_packages` giữ `Requirement.name` + `.specifier` và **bỏ `Requirement.url`**, nên direct reference co lại thành presence check: PyPI `clip==0.2.0` ⇒ `RUNTIME_READY` ⇒ `_reusable_interpreter` trả lại interpreter ⇒ planning ghi `RUNTIME_VERIFIED` | `verify_environment` nay kiểm nguồn qua `direct_url.json`; **đúng nguồn → RUNTIME_READY**, **sai nguồn → BLOCKED/DEPENDENCY_SOURCE_MISMATCH**, reuse → `None` |
| **R13-02** | `FIXED` | Hai entry CLIP4Cir khai `clip` trần ⇒ pip chọn `clip==0.2.0`, không có `clip.model.CLIP` mà `src/validate.py:8,12` cần | Cả hai nay khai source; install spec sinh ra đúng URL. **README upstream không pin revision** nên `a1d0717` được ghi là **adaptation có lý do**, không phải pin tác giả. Mapping checkpoint vẫn là blocker riêng |
| **R13-03** | `FIXED` | Nhánh miễn lỗi lọc stdout rồi **tiếp tục khi rỗng**, nên interpreter **không có pip** (stderr `No module named pip`, stdout rỗng) ⇒ *"All checks passed"* | Tái hiện bằng venv `--without-pip` thật: trước = `True/All checks passed`, sau = **refuse** với "no diagnostics". Nonzero mặc định block; exception duy nhất (decord) phải được nhận diện **positively**, stderr rỗng, và module import được |
| **R13-04** | `FIXED` | `_add_package` so `@ref:` đã lưu với request đã decode ⇒ cùng một nguồn khai hai lần luôn khác nhau | Qua `model_constraints()`: khai trùng ⇒ **idempotent**; nguồn khác ⇒ vẫn raise |
| **R13-05** | `FIXED` | Verdict còn lệnh diff của R10, đoạn "what was tested" ghi SHA cũ, hai dòng blocker nói "no CLI entrypoint"/"no pinned source at all", và câu kết khẳng định không còn lỗi trong chính vùng mà lượt này tìm ra 5 lỗi | Đã sửa từng chỗ; hàng số liệu R11 đánh dấu **HISTORICAL-SUPERSEDED** ngay tại chỗ đọc (log nó dẫn chứng kết bằng `1 failed`); claim "spec no longer exempt" của R12 sửa lại cho đúng phạm vi |
| **Giả thuyết A** | `FIXED` | `Path(path).resolve() == Path(sys.executable).resolve()`: venv hợp lệ có `bin/python` symlink tới binary nền bị **từ chối là "system Python"** | Reuse nay so **environment identity** (`sys.prefix`). Venv thật tạo từ cùng binary ⇒ **accept**; interpreter nền trần ⇒ vẫn refuse |
| **Giả thuyết B** | `VERIFIED CLOSED` | "Nguồn bị chặn có mượn được readiness từ record GPU cũ?" | Cache khoá theo `_environment_contract_digest`, **có `packages`**; đổi source ⇒ đổi digest ⇒ record cũ không dùng được. Đo trực tiếp: digest khác nhau |

## 2. Lỗi do chính lượt này gây ra và đã sửa

| Lỗi | Bắt bởi | Xử lý |
|---|---|---|
| Một edit thay (thay vì thêm) dòng import, làm `WorkbenchError` **không định nghĩa** ⇒ mọi refusal runtime ném `NameError` | `test_runtime_adaptation.py` (7 test) | Khôi phục import; commit riêng |
| Metadata probe đọc `report.prefix` (một **thư mục**) trực tiếp ⇒ chạy bằng interpreter của orchestrator, tìm thấy **sai** ở cả hai chiều | Probe hai chiều | Thêm `_environment_interpreter()` để lấy interpreter trong prefix |
| Hai test R12 assert **chuỗi xuất hiện trong source** (`inspect.getsource`) thay vì hành vi — pass khi hành vi sai, fail khi hành vi được viết lại cho **chặt hơn** | Sweep R13 | Thay bằng assert trên **quyết định**, với metadata thật |

## 3. Neighboring paths đã kiểm

- **Đường reorder/fallback**: `_constraints` xử lý reference-vs-version ở **cả hai thứ tự** (đã có regression từ R12); idempotence nay đúng cả hai chiều.
- **Cold start vs reuse**: `_reusable_interpreter` và `_group_status` đều dựa trên `verify_environment`, nên cả hai nay thấy source; đo bằng fixture interpreter thật.
- **GPU evidence**: record cũ không vượt được contract digest; GPU_RUNTIME vẫn `PENDING` khi không có target.
- **Cancellation/failure**: nhóm E2E 18 test (release-chain, true-master, acceptance, standalone) **pass** trên host này.

## 4. Kiểm chứng đã chạy

| Lượt | Kết quả thật |
|---|---|
| Full suite, disk sạch | **1773 passed, 1 failed, 12 skipped**; 1786 collected (`/tmp/r13/full2.log`) |
| R13 + R12 regression | **32 passed, 1 skipped** |
| E2E set (release-chain, true-master, acceptance, standalone) | **18 passed** |
| release_verify --offline | `ENGINEERING_INCOMPLETE`, **0 tiêu chí chưa đạt**, 5 pending |
| Frontend | pass |
| Source provenance | 16 entry / 15 checkout verified |

**Về lượt full suite đầu (10 failed):** đĩa xuống còn ~4 GB giữa lượt; 8 failure thuộc `release_verify`/`gpu_records`, và **tất cả pass khi chạy riêng** (35 passed, 68 passed, 152 passed). Lượt chạy lại trên đĩa sạch: **1 failed duy nhất**. Hai lượt được ghi riêng, không cộng thành một tổng.

## 5. Blocker bên ngoài

- **`clang` không có trên host này**, và **không có passwordless sudo** ⇒ không cài được. Đây là prerequisite đúng như reviewer nêu cho 2 E2E failure ở host của họ. Trên host này nhóm E2E liên quan **pass** vì fixture không cần clang.
- **Không có checkpoint tác giả, GPU, FashionIQ đầy đủ** ⇒ không có inference/paper parity.
- **`conda` có trên host** nhưng solve thật vượt budget planning ⇒ plan dựng bằng injected solver; **không** coi đó là solve/install thành công.

## 6. Subagent — thực tế

| Mục | Kết quả |
|---|---|
| `cx/gpt-6-luna` (ưu tiên 1) | **KHÔNG tồn tại** trong `models.yml` |
| `wdb/deepseek-ai/DeepSeek-V4.1-Flash` (ưu tiên 2) | **KHÔNG tồn tại** trong `models.yml` |
| Gemini | **bị vô hiệu hoá** theo yêu cầu |

⇒ **Không có subagent nào được gọi** trong lượt này. Không có cross-review độc lập; mọi kết luận do main agent tự kiểm chứng. Đây là khoảng trống bằng chứng được ghi nhận, không được thay bằng mô tả.

## 7. Giới hạn đã biết

`pip check` trên profile native vẫn báo decord (wheel metadata khai tag `cp36`); đây là lỗi packaging upstream được chấp nhận **có điều kiện và có ghi tên**, không phải raw pass. Python 3.8.20 fixture **không** phải profile 3.8.10 đã khai; INTENT khai 3.9 là contract riêng. CPU torch không chứng nhận CUDA.
