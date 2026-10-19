# PC PILOT HANDOFF — CIR FAILURE ANALYSIS WORKBENCH

Tài liệu này là hợp đồng bàn giao sang máy RTX 5070 Ti. Nó nói **chính xác** cái gì đã được kiểm chứng
trên máy dev, cái gì chỉ có thể kiểm trên máy đích, và cách đọc kết quả.

**Chưa từng chạy trên RTX 5070 Ti.** Không có chứng thư GPU nào trong repo, chưa từng có suy luận
thật trên GPU và chưa từng tái lập Recall — mọi khẳng định về LIMN trong tài liệu này là mức tĩnh
hoặc mức fixture.

## Phán quyết

**`PC_PILOT_HANDOFF_READY`** — đường pilot chạy được trên máy đích.
**`GPU_SCIENTIFIC_VALIDATION_PENDING`** — toàn bộ phần khoa học còn chờ GPU.

**Chỉ LIMN sẵn sàng cho lượt PC pilot đầu tiên.** 15 model còn lại **chưa** sẵn sàng, nhưng **không
phải vì thiếu wrapper**: `workbench/backend/adapters/models.py:294` đăng ký 15 adapter và
`workbench/replay/` có 16 wrapper — mọi model trừ `ptha_mtst` đều có entrypoint chạy được. Lý do
thật, theo từng model: `encoder` thiếu asset LAION ViT-B/32;
`clip4cir_rn50x4_noft`/`clip4cir_rn50x4_fullft` cần đúng member của archive và cặp CLIP/Combiner khớp;
`dcnet` thiếu run directory của tác giả; `tgcir` chỉ có preflight (chưa xác minh member của
`TG-CIR.zip`); `sprc` chỉ có ViT-L public (variant khác paper); `clvc_net` cần map sáu file
local/global; `pair` cần nhãn B1/B2 từ tác giả; `ptha_mtst` không có weight FashionIQ.
Ma trận đầy đủ: `workbench/docs/MODEL_STATUS_SPEC.md` §2.

**Trạng thái 16 model (bảng chuẩn, bằng chứng + điều kiện chuyển trạng thái):**
[`workbench/docs/MODEL_STATUS_SPEC.md`](MODEL_STATUS_SPEC.md)

**SHA ĐÃ KIỂM THỬ (nguồn sản xuất): `cd2de4b`** — mọi test, acceptance, mutation và release_verify mô
tả trên trang này đã chạy trên đúng cây đó (1754 passed, 0 failed, 11 skipped; acceptance, mutation và release_verify đều
chạy lại trên cùng commit). `119df5f` đã bị **thay thế**: lượt R9 sửa production (DCNet binding và
thứ tự category, tập patch CSMCIR, tập input của constructor, processor BLIP-2), nên số test và kết
quả gate ghi ở các bản trước không còn mô tả cây này.

Lưu ý: các bản ghi cũ `59949ff`, `9a53ee1` và `5567bc7` đã bị **thay thế**. Công việc vendoring và phục hồi
model đã thay đổi file production (`workbench/backend/runtime.py`, `workbench/replay/common.py`,
`workbench/backend/adapters/models.py`, `workbench/registry/*`, `workbench/replay/*`), nên số test và
kết quả gate của các bản ghi cũ không còn mô tả cây hiện tại.

Nhóm C của acceptance chỉ fail khi chạy vào container `cir-workbench` **dựng trước** thay đổi này
(image cũ vẫn phục vụ registry cũ: `ptha_mtst source_available: false`). Chạy lại với backend khởi
động từ source hiện tại thì nhóm C PASS. Cần rebuild image trước khi chạy lại acceptance.

Trước đây trang này trỏ tới `11ba3df`. Commit đó nay là **nguồn đã lỗi thời**: `git diff 11ba3df
2bc0cd2` thay đổi `pc.sh`, `workbench/replay/limn_export.py`, `workbench/replay/limn_publish.py`,
`workbench/scripts/evaluate_models.py` và `workbench/frontend/evidence.ts`. Cụ thể, bản `pc.sh` ở
`11ba3df` không hề chứa `sys.exit(7)` hay `REGISTRY_UNKNOWN` (mã thoát 7 được định nghĩa ngay trên
trang này), nên checkout theo nó sẽ cài một launcher KHÔNG THỂ in ra mã thoát mà chính tài liệu này
mô tả. Test `test_no_release_document_points_at_a_superseded_source_sha` nay so sánh **cây nguồn sản
xuất theo từng byte**, không phải quan hệ tổ tiên.

Đây là một SHA bất biến, không phải đầu nhánh: nhánh `research/cir-failure-workbench` sẽ tiếp tục tiến
lên, nên **không** checkout theo tên nhánh nếu bạn muốn đúng cây đã kiểm thử.

Lấy cây đã kiểm thử:

```bash
git clone git@github.com:Le-Minh-Nhut/cir.git && cd cir
git fetch origin research/cir-failure-workbench
# Checkout tường minh. KHÔNG dùng `git reset --hard`: trên máy đã có việc đang làm, nó xoá sạch
# không hỏi. Muốn đúng cây đã kiểm thử thì checkout đúng commit đó.
# Nguồn đã kiểm thử (revision cố định; đây là cái đã chạy 1609 test):
git checkout cd2de4b
# Nhánh đang phát triển thêm tài liệu. Dùng nó nếu muốn bản mới nhất:
#   git checkout research/cir-failure-workbench
```

Nhánh này sẽ tiếp tục tiến lên. Mốc bất biến để đối chiếu là `TESTED_SOURCE_SHA` trong
`FINAL_PC_RELEASE_VERDICT.md`; commit sau mốc đó chỉ được phép sửa tài liệu, và một test so sánh
cây nguồn sản xuất theo từng byte sẽ thất bại nếu điều đó không còn đúng.

Bản đầy đủ của lượt review cuối (ma trận 16 model, coverage, sổ lỗi, đối chiếu độc lập, verdict) nằm
trong `workbench/docs/reviews/FINAL_*.md` và `workbench/docs/reviews/r8/R8_FINAL_REPORT.md` (báo cáo
của lượt R8: git state, đóng lỗi, kết quả test, phán quyết).

## 1. Lệnh chính

```bash
./pc.sh
```

Đây là lệnh duy nhất của người vận hành. Nó: kiểm host → dựng/kiểm container → đồng bộ source → kiểm
dataset → chuẩn bị LIMN checkpoint bundle → lập kế hoạch môi trường → xác minh GPU → chạy đánh giá
chính thức → **quan sát per-query** → phân tích metric aggregate → kiểm parity thiết bị đo → tạo
completion proof → ghi ledger có HMAC → validate strict → dựng chỉ mục DuckDB → phục vụ API → hiển thị
UI.

Cờ đã tồn tại (đọc từ `pc.sh`): `--model`, `--force`, `--doctor`, `--verify-only`, `--no-gpu`,
`--dataset`, `--checkpoint`, `--timeout`, `--help`.

### Xuất per-query: tự động, không cần cờ

Với LIMN, export per-query là **tự động** trong `./pc.sh`. Không có cờ nào để bật.

Cơ chế là **hai tầng**, và lý do là kỹ thuật chứ không phải kiến trúc cho vui:

1. **Trình quan sát** (`workbench.replay.limn_export`) chạy dưới **interpreter của model**. Nó nạp
   checkpoint qua loader gốc, tái dẫn xuất xếp hạng, đối chiếu với aggregate native, rồi ghi **một
   payload JSON thuần**. Nó KHÔNG import schema canonical, vì môi trường model được cài từ
   `requirements` của chính model (torch, open_clip, numpy) và **không có pydantic** — import pydantic
   ở đó làm observer chết ngay khi khởi động, khiến aggregate đạt mà mọi export thất bại.
2. **Trình xuất bản** (`workbench.replay.limn_publish`) chạy dưới **interpreter workbench** (nơi có
   pydantic), validate payload, kiểm chính sách admission, và công bố artifact canonical **atomic**.

Cả hai chạy dưới **cùng một `run_id`** do orchestrator sinh, và cùng mang digest lệnh/môi trường của
lượt aggregate. Cha (`evaluate_models.execute`) chạy observer như tiến trình con, rồi chạy publisher
như tiến trình con, rồi **tự đọc lại bytes** để tính digest — con không bao giờ tự khai digest của mình.

**Lượt chạy trên API/UI**: chỉ mục DuckDB chứa bản ghi per-query canonical, nên chỉ lượt chạy của
model có observer mới xuất hiện ở `/api/runs` và tab **Evaluation Runs**. Lượt của 15 model
aggregate-only nằm ở `/api/benchmark-runs` và tab **Models**. `./pc.sh` đọc **cả hai** endpoint và
kết luận theo đúng khả năng mà chính report của lượt chạy khai (`AGGREGATE_ONLY_VERIFIED`), nên một
lượt aggregate-only hợp lệ vẫn đạt exit 0 — nó **không** bị đòi một artifact per-query không tồn tại.
Xem mục 7 để có lệnh kiểm chứng từng bước.

**Trạng thái per-query** (`per_query_state` trong report aggregate):

| Trạng thái | Nghĩa | Exit code của `./pc.sh` |
|---|---|---|
| `CAPABILITY_ABSENT` | model không khai báo observer — **aggregate-only theo thiết kế**, không phải lỗi | 0 |
| `CAPABILITY_ABSENT_FOR_DIAGNOSTIC` | observer chỉ bao phủ bundle đầy đủ, còn lượt chạy này chủ động chọn MỘT danh mục | 0 |
| `PER_QUERY_EXPORT_FAILED` | model CÓ khai báo observer và nó không xuất được gì | **3** |
| `PER_QUERY_EXPORTED_UNVERIFIED` | đã xuất artifact, verifier chưa chứng thực | **4** |
| `PER_QUERY_VERIFIED` | đã xuất + verifier đã chứng thực | 0 |
| (không đọc được report) | launcher không đọc được điều lượt chạy đã ghi — **không** được coi là aggregate-only | **5** |
| `PROOF_UNAVAILABLE_AND_*` | tệp completion proof không còn: full verification không chạy được. Lượt gọi vẫn được xác thực (`execution_authenticated`) và báo cáo tổng hợp vẫn được phục vụ, nhưng một artifact per-query **không bao giờ** được promote lên `PER_QUERY_VERIFIED` — nên đây **không** phải pilot đầy đủ | **4** nếu mô hình khai báo observer |
| `AGGREGATE_ONLY_EXECUTION_AUTHENTICATED_PROOF_UNAVAILABLE` | mô hình aggregate-only, sổ cái xác thực LƯỢT GỌI, nhưng **thiếu completion proof** nên **không** chứng thực được khoa học | **7** |

**Xác thực lượt gọi KHÔNG phải chứng thực khoa học.** `execution_authenticated` nghĩa là sổ cái xác
nhận lượt gọi này đã chạy và đã ghi đúng các byte này, ở đúng danh tính này. `certified` nghĩa là
completion proof hợp lệ và các con số tái lập được từ chính output của lượt chạy. Thiếu proof thì điều
thứ nhất đúng, điều thứ hai **sai**, và hai điều đó **không bao giờ** dùng chung một nhãn hay một exit
code. Exit code 7 là "đã ghi nhận trung thực, chưa chứng thực" — không phải thành công.

Exit code 5 nghĩa là launcher **không biết** — nó không được phép tuyên bố "mô hình này không có trình
quan sát" khi sự thật là nó không đọc được report. Kiểm tra đường dẫn `HOST_WORKBENCH_ARTIFACTS_ROOT`.

Exit code 3 nghĩa là: *tổng hợp đạt và hợp lệ, nhưng pilot ĐẦY ĐỦ chưa sẵn sàng* — phân tích theo
truy vấn chưa dùng được. Kết quả tổng hợp, sổ cái và completion proof **vẫn được giữ nguyên**.

Trình quan sát và trình xuất bản nhận **môi trường đã lọc** giống hệt evaluator chính thức: không có
`WORKBENCH_LEDGER_KEY`, `OPENAI_API_KEY`, `GITHUB_TOKEN` hay credential nào của người vận hành.

## 2. Điều kiện tiên quyết trên máy đích

| Mục | Trạng thái |
|---|---|
| NVIDIA driver + container runtime | `NOT_EXECUTED_ON_TARGET_PC` |
| `WORKBENCH_LIMN_PYTHON` (hoặc env tương đương được planner chọn) | phải đặt, nếu không `pc_readiness` báo `BLOCKED_ENVIRONMENT` |
| FashionIQ tại `FASHIONIQ_ROOT` | bắt buộc; kiểm bằng `./pc.sh --doctor` |
| LIMN bundle ba checkpoint (`0_{dress,shirt,toptee}_best_model.pt` + SHA256) | bắt buộc; thiếu một member → từ chối **nêu tên member** |
| `WORKBENCH_LEDGER_KEY` hoặc đúng **một** file khoá | hai file khoá khác nhau → từ chối và nêu đường dẫn |

## 3. Hành vi từ chối (fail-closed)

| Tình huống | Kết quả |
|---|---|
| Thiếu một trong ba checkpoint | từ chối, nêu tên member thiếu; **không** rút xuống một category |
| `--checkpoint base_iter0_dress` một mình | chạy như **diagnostic**, không bao giờ là kết quả benchmark đầy đủ |
| Parity thiết bị đo lệch | artifact per-query bị **giữ lại**, aggregate native vẫn được xuất, exit khác 0 |
| Experiment thiếu provenance | không được promote; `validate_results --strict-real` từ chối |
| `run_id` đã tồn tại với bytes khác | từ chối ghi đè |
| Ledger không đọc được | chỉ mục vẫn dựng, mọi experiment ở trạng thái **chưa xác thực** (không promote) |

## 4. Kết quả ở đâu

| Artifact | Đường dẫn |
|---|---|
| Report aggregate chính thức | `<artifacts>/reports/<model>_<checkpoint>_<run_id>_aggregate.json` |
| Completion proof | `<artifacts>/reports/<model>_<checkpoint>_<run_id>_completion.json` |
| Log lượt chạy | `<artifacts>/logs/<model>__<checkpoint>__<run_id>/{stdout,stderr}.log` |
| Log quan sát | cùng thư mục: `observation.{stdout,stderr}.log` |
| Artifact per-query canonical | `<results>/<protocol>/<model>/<run_id>.json` |
| Ledger | `<artifacts>/evidence_ledger.jsonl` |
| Chỉ mục | `<artifacts>/workbench.duckdb` |

## 5. Bộ lọc thất bại theo mọi Top-K và chọn lượt đang xem

Sample Explorer có một bộ chọn **Active model/run** riêng: đây là lượt mà ảnh Top-K đang hiển thị
thuộc về, và nhãn của nó (model, checkpoint, protocol, run ID, trạng thái bằng chứng) hiện ngay trên
chi tiết truy vấn. Chọn nhiều lượt vẫn dùng cho **Compare Models**, **Failure Analytics**, **Common
Failures** và **Disagreement** — chúng là hai việc khác nhau.

**Bộ lọc kết quả** có đủ mọi ngưỡng:

```
All · fail @1 · fail @5 · fail @10 · fail @20 · fail @50 · fail @100 · fail @200
                          pass @1 · pass @5 · pass @10 · pass @20 · pass @50 · pass @100 · pass @200
```

Ngữ nghĩa (rank 1-based, `target_rank` là vị trí trong **toàn bộ** xếp hạng ứng viên hợp lệ):

```
SUCCESS@K  ⟺  target_rank <= K
FAILURE@K  ⟺  target_rank >  K
```

`fail @K` và `pass @K` **loại trừ lẫn nhau** (gửi cả hai → HTTP 400 `conflicting_rank_filter`), và
giá trị ngoài khoảng → 400 `invalid_rank_threshold`. Lọc chạy **trước** phân trang, nên `total` luôn
là số của tập đã lọc chứ không phải số của trang.

`target_rank` **không** phụ thuộc độ sâu đã lưu: một truy vấn rank 500 vẫn là thất bại ở mọi K dù
Top-K đã lưu chỉ có 50 ảnh. Phân loại thất bại và khả năng xem ảnh là hai chuyện khác nhau — khi K
vượt độ sâu đã lưu, giao diện **nói rõ** ("Yêu cầu Top-200 — chỉ có Top-50 được lưu"), không im lặng
hiển thị 50 ảnh dưới nhãn Top-200. Đổi K chỉ đổi phần hiển thị; **không** chạy lại suy luận.

---

## 6. Trạng thái bằng chứng — cách đọc

`evidence_status` **không** đến từ `data_kind`. Nó do chính sách admission quyết, và chỉ `verifier`
được promote lên `PER_QUERY_VERIFIED`:

| Trạng thái | Nghĩa |
|---|---|
| `SYNTHETIC_OR_MOCK` | dữ liệu phát triển; không bao giờ là bằng chứng |
| `PER_QUERY_PARITY_PENDING` | chưa ghi nhận parity thiết bị đo |
| `PER_QUERY_PARITY_FAILED` | xếp hạng quan sát không tái lập native; artifact bị giữ |
| `PER_QUERY_UNVERIFIED` | khai là experiment nhưng chưa ai chứng thực |
| `PER_QUERY_VERIFIED` | verifier đã chứng thực, và receipt **khớp digest artifact này** |

## 7. Cách xác minh và dựng lại chỉ mục

`--verify-only` **chỉ** kiểm chứng năng lực GPU của môi trường mô hình rồi dừng. Nó **không** xác
thực một lượt chạy đã có, **không** dựng lại chỉ mục DuckDB, và **không** tải checkpoint. (Trước đây
một số thông báo lỗi hướng dẫn chạy `--verify-only` cho những việc đó; các thông báo ấy đã được sửa.)

Muốn **xác thực một lượt chạy đã có rồi dựng lại chỉ mục**, chạy chính lệnh thường — nó **tái sử dụng**
kết quả đã chứng thực thay vì đánh giá lại:

```bash
./pc.sh                       # nhận diện lượt đã chứng thực -> xác thực -> dựng lại chỉ mục
```

Từng bước, chạy trực tiếp trong container:

```bash
# 1. Kiểm chứng năng lực GPU (hardware/runtime), rồi dừng:
./pc.sh --verify-only

# 2. Kiểm tra pipeline trên mọi cờ trong pc.sh (không bịa lệnh nào):
./pc.sh --self-check

# 3. Kiểm chứng runtime GPU bên trong container cho một model cụ thể (lệnh thật của image):
docker exec cir-workbench python -c \
  "import sys, workbench.scripts.pipeline as p; sys.exit(p.probe_gpu_capability(sys.argv[1]))" limn

# 4. Verdict của verifier cho MỘT lượt chạy. --model/--checkpoint/--protocol là BẮT BUỘC:
docker exec cir-workbench python3 workbench/scripts/verify_run.py \
  --run-id <run_id> --model limn \
  --checkpoint base_iter0_all_categories --protocol fashioniq_val_split --json

# 5. Kiểm tra mọi artifact canonical trong results root theo schema:
python -m workbench.scripts.validate_results --all --strict-real

# 6. Dựng lại chỉ mục DuckDB (thay cho việc khởi động lại backend):
curl -sS -X POST http://127.0.0.1:8000/api/index/rebuild
```

**Lượt chạy của mô hình aggregate-only** (15/16 model) **không** xuất hiện trong `/api/runs`: chỉ mục
chỉ chứa bản ghi per-query canonical, mà chỉ LIMN có observer. Lượt chạy của chúng nằm ở
`/api/benchmark-runs` và hiển thị trong UI ở tab **Models**. Đây là đúng theo thiết kế, không phải lỗi;
`./pc.sh` báo `AGGREGATE_ONLY_VERIFIED` cho tình huống này và **không** đòi một artifact per-query
không tồn tại.

**Xác nhận một lượt chạy hiện trên UI**: mở `http://127.0.0.1:${WORKBENCH_FRONTEND_PORT}/` (mặc định
5173) → tab **Models** (báo cáo tổng hợp, mọi model) và tab **Evaluation Runs** (lượt có per-query).
Nếu lượt chạy của LIMN không có ở tab Evaluation Runs, chạy lại bước 4 rồi bước 6 ở trên.

## 7b. Ranh giới tin cậy của evaluator (đọc trước khi tin vào "cách ly")

Evaluator mô hình chạy như tiến trình con **cùng UID/GID** với orchestrator, trên **cùng filesystem**
bind-mount. Những gì đã làm:

* Môi trường của con được **allowlist**, không copy `os.environ`: `WORKBENCH_LEDGER_KEY`,
  `HF_TOKEN`, `GITHUB_TOKEN`, `OPENAI_API_KEY`, `HOME`, `XDG_*` và `WORKBENCH_SECRET_*` **không** được
  truyền, nên con không suy ra được đường dẫn khoá ký.
* Khoá HMAC được hạn chế ở 0600 và ưu tiên `$XDG_CONFIG_HOME/cir/` (trong deployment là `/artifacts/config`).

Những gì **chưa** làm, và không được tuyên bố là đã làm:

* Không có **cách ly OS** (không user riêng, không namespace, không seccomp cho evaluator).
* Vì cùng UID trên cùng volume, code upstream **có thể đọc mọi tệp mà người vận hành đọc được**,
  kể cả `/artifacts/config/cir/workbench_ledger_key`.

**Kết luận thẳng thắn**: mô hình tin cậy hiện tại là **(A) code upstream đã ghim và checkpoint đã xác
minh là TIN CẬY**. Hệ thống không tuyên bố chống được (B) source upstream thù địch hay (C)
deserialization checkpoint thực thi code. Nếu cần chạy source chưa được review, phải thêm ranh giới thật
(user OS riêng cho evaluator + signer chạy ở tiến trình tách biệt) trước khi tin vào chứng thực.

## 8. Trạng thái GPU mong đợi lúc bàn giao

```
GPU_RUNTIME          NOT_EXECUTED_ON_TARGET_PC
MODEL_INFERENCE      NOT_EXECUTED_ON_TARGET_PC
MODEL_EVALUATION     NOT_EXECUTED_ON_TARGET_PC
SCIENTIFIC_OUTPUT    NOT_EXECUTED_ON_TARGET_PC
PER_QUERY_PARITY     PENDING  (chỉ kiểm được sau suy luận thật)
```

`GPU_SCIENTIFIC_VALIDATION_READY` **không** đạt được trên máy dev, và không có gì trong repo tuyên bố
ngược lại.

## 9. Lỗi thường gặp thật (không phải lỗi phần mềm)

| Triệu chứng | Nguyên nhân |
|---|---|
| `BLOCKED_ENVIRONMENT` cho LIMN | `WORKBENCH_LIMN_PYTHON` chưa đặt |
| `two different ledger keys exist` | còn cả `<repo>/.workbench_ledger_key` và `$XDG_CONFIG_HOME/cir/workbench_ledger_key`; giữ một |
| `checkpoint bundle incomplete` | thiếu member category; tên member được nêu trong thông báo |
| `QUERY_EXPORT_PARITY_MISMATCH` | xếp hạng quan sát lệch native — **điều tra, không bỏ qua**; aggregate native vẫn đúng |
| `AGGREGATE_PARSER_UNAVAILABLE` | model chưa có parser metric; cần output thật của evaluator |
