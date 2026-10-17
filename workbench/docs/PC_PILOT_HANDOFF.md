# PC PILOT HANDOFF — CIR FAILURE ANALYSIS WORKBENCH

Tài liệu này là hợp đồng bàn giao sang máy RTX 5070 Ti. Nó nói **chính xác** cái gì đã được kiểm chứng
trên máy dev, cái gì chỉ có thể kiểm trên máy đích, và cách đọc kết quả.

**Chưa từng chạy trên RTX 5070 Ti.** Không có chứng thư GPU nào trong repo.

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

Với LIMN, export per-query là **tự động** trong `./pc.sh`: `evaluate_models.execute()` gọi observer như
tiến trình con dưới interpreter của model và truyền thẳng `--run-id <run_id>` — chính danh tính mà
orchestrator đã dùng cho lượt này (đường vào `execute()` cũng đã chở nó từ `POST /api/jobs`). Vì cha
truyền xuống con đúng chuỗi đó, **không thể** có hai danh tính cho một lượt chạy.

Chạy exporter độc lập (không qua `pc.sh`) vẫn được, nhưng khi đó **không có** danh tính orchestrator:
artifact được đánh dấu `data_kind="mock"` và **không bao giờ** thành bằng chứng nghiên cứu.

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

## 5. Trạng thái bằng chứng — cách đọc

`evidence_status` **không** đến từ `data_kind`. Nó do chính sách admission quyết, và chỉ `verifier`
được promote lên `PER_QUERY_VERIFIED`:

| Trạng thái | Nghĩa |
|---|---|
| `SYNTHETIC_OR_MOCK` | dữ liệu phát triển; không bao giờ là bằng chứng |
| `PER_QUERY_PARITY_PENDING` | chưa ghi nhận parity thiết bị đo |
| `PER_QUERY_PARITY_FAILED` | xếp hạng quan sát không tái lập native; artifact bị giữ |
| `PER_QUERY_UNVERIFIED` | khai là experiment nhưng chưa ai chứng thực |
| `PER_QUERY_VERIFIED` | verifier đã chứng thực, và receipt **khớp digest artifact này** |

## 6. Cách xác minh và dựng lại chỉ mục

```bash
python -m workbench.scripts.validate_results --all --strict-real
python -m workbench.scripts.verify_run --run-id <run_id>      # verdict của verifier
# dựng lại chỉ mục: POST /api/index/rebuild, hoặc khởi động backend (tự dựng nếu chỉ mục thiếu)
```

## 7. Trạng thái GPU mong đợi lúc bàn giao

```
GPU_RUNTIME          NOT_EXECUTED_ON_TARGET_PC
MODEL_INFERENCE      NOT_EXECUTED_ON_TARGET_PC
MODEL_EVALUATION     NOT_EXECUTED_ON_TARGET_PC
SCIENTIFIC_OUTPUT    NOT_EXECUTED_ON_TARGET_PC
PER_QUERY_PARITY     PENDING  (chỉ kiểm được sau suy luận thật)
```

`GPU_SCIENTIFIC_VALIDATION_READY` **không** đạt được trên máy dev, và không có gì trong repo tuyên bố
ngược lại.

## 8. Lỗi thường gặp thật (không phải lỗi phần mềm)

| Triệu chứng | Nguyên nhân |
|---|---|
| `BLOCKED_ENVIRONMENT` cho LIMN | `WORKBENCH_LIMN_PYTHON` chưa đặt |
| `two different ledger keys exist` | còn cả `<repo>/.workbench_ledger_key` và `$XDG_CONFIG_HOME/cir/workbench_ledger_key`; giữ một |
| `checkpoint bundle incomplete` | thiếu member category; tên member được nêu trong thông báo |
| `QUERY_EXPORT_PARITY_MISMATCH` | xếp hạng quan sát lệch native — **điều tra, không bỏ qua**; aggregate native vẫn đúng |
| `AGGREGATE_PARSER_UNAVAILABLE` | model chưa có parser metric; cần output thật của evaluator |
