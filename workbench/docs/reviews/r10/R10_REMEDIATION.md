# R10 REMEDIATION — sáu finding, cách xử lý, và phạm vi bằng chứng

Bắt đầu từ `e46ad65aa792b01461cc5319b2b8511b4714df41` (production tree = `23dca02`), sạch.
Mỗi finding được **tái hiện trước khi sửa**, và mỗi sửa có một test chạy qua ranh giới production.

Báo cáo điều tra đi kèm trong cùng thư mục:

| File | Nội dung |
|---|---|
| `R10_LAVIS_DEPS.md` | closure import thật của 6 model, metadata wheel LAVIS, luận điểm về `timm`/`transformers` |
| `R10_TEST_PORTABILITY.md` | 9 failure của lượt review ngoài: nguyên nhân, phân loại, cách sửa nhỏ nhất |
| `R10_E2E_SURFACE.md` | sản phẩm hứa gì, đường launch thật, cái gì có test và cái gì không |

---

## 1. Bảng disposition

| ID | Nguyên nhân thật | Phương án đề xuất | Quyết định | Thay đổi | Bằng chứng |
|---|---|---|---|---|---|
| **R10-06** | `inspect_environment_distributions` đọc `base_prefix/site-packages` **bất cứ khi nào** `base_prefix != prefix`. Venv cô lập cũng có quan hệ đó, nhưng `site.py` của nó không thêm base vào `sys.path` ⇒ verifier cấp `RUNTIME_READY` cho package interpreter không import được | "Chỉ merge base inventory khi cấu hình/đường import thật cho phép" | **FIXED** (đúng như đề xuất) | `runtime.py:_inherits_base_site_packages` + gate trong `inspect_environment_distributions`; đọc `pyvenv.cfg:include-system-site-packages`, `conda-meta`, và **mặc định là không kế thừa** khi quan hệ không giải thích được | `test_r10_remediations.py`: venv thật + `python -I -c find_spec` làm ground truth; `packaging` biến mất khỏi inventory của venv cô lập, còn nguyên ở venv `--system-site-packages` |
| **R10-04** | Registry còn `source_blocker` nói "no runnable evaluation path is registered" cho `airknow`/`conesep`/`intent`, trong khi `ADAPTERS` có adapter thật và `replay/` có wrapper | "Gỡ đúng blocker lỗi thời; giữ blocker asset/variant/runtime thật" | **FIXED** (đúng như đề xuất) | Xoá 3 field `source_blocker` đó | `preflight_blockers` thật: airknow từ 3 → **2** blocker (BLIP-2 weights, EVA-ViT-g), conesep/intent từ 1 → **0**; không blocker nào còn nhắc "no runnable evaluation path" |
| **R10-05** | `salesforce-lavis==1.0.2` khai `timm==0.4.12` + `transformers>=4.25,<4.27`, không thể đồng thời thoả `timm==0.9.16` mà vendored `eva_vit.py:15-16` cần. **Thêm**: `environment_planner._evaluator_imports` quét `adapter.script` — với 5 wrapper đó là `test.py`, tức code **training thượng nguồn**, không phải chương trình chạy thật | "Bỏ distribution/pin nếu không cần; khai báo dependency thật của bản vendored" | **FIXED**, và **nguyên nhân gốc sâu hơn đề xuất**: không chỉ pin sai mà **scanner đọc sai chương trình** | Bỏ `salesforce-lavis` khỏi 6 profile (wheel cũng thiếu file checkout này có, nên không phải drop-in); thêm `Adapter.launcher`/`launcher_roots` **khai báo** chương trình chạy; `_evaluator_imports` đi theo khai báo đó | Wheel sha256 `489782ef…` khớp PyPI JSON (kiểm 2 lần); scanner trước/sau: `hint` từ `['numpy','torch','tqdm']` thành tới `transformers`/`timm`/`omegaconf` — tức chính các package bị bỏ sót |
| **R10-01** | `WORKBENCH_DCNET_DATA_DIR` đổi **validator tin gì**, nhưng native `ConfigParser` không đọc biến đó ⇒ validator pass trong khi `CE_dataset` mở root khác | "Bỏ override khỏi quyết định binding nếu nó không đổi tiến trình native" | **FIXED** (đúng như đề xuất) | Xoá override khỏi `configured_data_dir`; quyết định chỉ đọc `config.json` + CWD evaluator; `preflight` mặc định CWD = checkout | Chạy **ConfigParser thật** (không phải bản sao): link trỏ `selected` → đọc `selected`; link trỏ `wrong` → đọc `['WRONG']`; với override đặt về `selected`, validator vẫn **refuse** và nêu đúng `wrong` |
| **R10-02** | Staging đúng cách lại bị integrity gate từ chối: DCNet không khai `canonical_dataset_link`, và stager không tạo được thư mục cha `dataset/` trên checkout mới | "Khai báo link; cho stager tạo thư mục cha; truyền contract đầy đủ" | **FIXED** (đúng như đề xuất) + 2 chỗ đề xuất chưa nêu | Khai `canonical_dataset_link: dataset/fashioniq`; nới validation registry sang path lồng 1 cấp; stager tạo parent **chỉ khi ghi thật**; `validate_source` nhận `MODEL`; `preflight` dùng CWD checkout | Trên bản sao checkout: verified `True` **trước và sau** khi stage, `unauthorized_untracked == []`; check-only/dry-run **không tạo gì** (kể cả parent); người chiếm chỗ thật và link trỏ nơi khác đều bị refuse |
| **R10-03** | `text_processors()` đăng ký `library_root` rồi **xoá package root khỏi `sys.modules`** nhưng giữ submodule + registry. Checkpoint là module pickle nên `find_class` import lại `lavis/__init__.py`, file này **đăng ký lại** `library_root`, mà `registry.register_path` **raise KeyError** khi tên đã có | "Giữ import context nhất quán xuyên suốt processor → checkpoint trong subprocess hiện có" | **FIXED** (đúng như đề xuất) | `prepare()` cài context **một lần và giữ nguyên** cho đời process; `register_path` thành idempotent **chỉ khi cùng giá trị** (khác giá trị vẫn raise); `reset()` tường minh cho tool so nhiều checkout; `prepare` từ chối checkout thứ hai trong cùng process | Fixture có `lavis/__init__.py` đăng ký như bản pinned + checkpoint **torch.save thật** ghi GLOBAL path: `load_trusted_whole_model` **thành công**, digest sai vẫn bị refuse; trên registry pinned thật: cùng root ⇒ chấp nhận, root khác ⇒ KeyError |

### Điểm khác với đề xuất, và lý do

- **R10-05**: đề xuất nói cần "xác định dependency thật của bản vendored". Đo được điều đó cần một thay đổi nữa mà đề xuất không nêu — sửa **scanner** trước, vì nó đang trả lời về một file khác. Đã làm cả hai, theo đúng thứ tự đó.
- **R10-02**: đề xuất nói thêm khai báo link; thực tế còn cần (a) nới validation registry vốn chỉ cho tên một cấp, và (b) tạo thư mục cha — nếu không thì `dataset/` chưa tồn tại trên checkout mới và `symlink_to` fail.
- **R10-01**: xoá override làm mất một đường "thoát" mà R9 từng thêm. Không có consumer production nào đọc biến đó (đã grep), nên việc xoá không phá tính năng nào đang dùng — nó chỉ xoá một cách **nói dối validator**.

### Không làm, và lý do

- **Không** hạ/bump version hàng loạt cho 16 model. Chỉ sáu profile LAVIS-family/SPRC được sửa, và chỉ ở chỗ chứng minh được mâu thuẫn.
- **Không** dùng `--no-deps`, không `pip install` im lặng bỏ qua conflict.
- **Không** khai 35 distribution mà đường chạy thật chạm tới: danh sách khai báo là **core do tác giả pin**, còn tập reachable do `_evaluator_imports` tính. Viết cả hai vào registry là hai nguồn sự thật sẽ lệch nhau — đúng điều lượt review cảnh báo.
- **Không** dựng loader service / plugin framework mới cho R10-03. Sửa nằm trong module sẵn có.

---

## 2. Phạm vi bằng chứng

| Đã chạy thật trên máy này | Fixture một phần | Không chạy được ở đây |
|---|---|---|
| pytest toàn suite (số ở §4) | cây `lavis` thu nhỏ + checkpoint `torch.save` thật | inference GPU |
| **ConfigParser native thật** của DCNet | venv thật (`venv --without-pip`, `--system-site-packages`) | checkpoint tác giả đầy đủ |
| `venv` thật + `python -I` làm ground truth | bản sao checkout DCNet | dataset FashionIQ canonical |
| `git ls-tree`/`worktree`, verify wheel sha256 với PyPI | `conda` (không có binary) | sáu môi trường cài thật |
| `prepare_canonical_link` ghi/xoá thật | `docker` (không có daemon) | recall / paper parity |

**Fixture không phải bằng chứng khoa học.** Checkpoint fixture chứng minh **đường class resolution** sống qua bước processor; nó không chứng minh HINT suy luận đúng.

---

## 3. Việc còn phải làm trên PC

Xem `MODEL_STATUS_SPEC.md` §6 cho danh sách lệnh đầy đủ và artifact cần stage.
