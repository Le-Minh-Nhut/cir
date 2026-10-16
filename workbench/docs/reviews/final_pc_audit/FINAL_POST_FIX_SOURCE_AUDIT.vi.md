# FINAL_POST_FIX_SOURCE_AUDIT.vi.md

Báo cáo audit nguồn sau khi sửa — CIR Failure Analysis Workbench
Baseline: `a517cf9` · **Commit đã kiểm thử: `46d8e59`** · Nhánh: `research/cir-failure-workbench`

Toàn bộ bằng chứng ở mục 6 được thu trên cây nguồn tại `46d8e59` (giống hệt nội dung đã push;
`origin/research/cir-failure-workbench` = `46d8e59`). Tệp này được cập nhật SHA sau khi commit đó,
nên bản thân nó nằm ở commit kế tiếp — nội dung nguồn được kiểm thử không đổi.

## 0. Phạm vi và cách kiểm

| | |
|---|---|
| File nguồn first-party được lập manifest | **114** (`SOURCE_COVERAGE_MANIFEST.md`) |
| Audit độc lập chạy | 4 slice (A deploy, B runtime, C models, D science) + 3 slice E/F bổ sung |
| Subagent | một phần harness subagent hỏng (lỗi chọn model) giữa chừng → các slice A/B/C được **main agent** tự audit có bằng chứng chạy thật, ghi rõ trong từng báo cáo |
| Nguyên tắc | “không bằng chứng = không phải defect”; tìm cả hai hướng (false PASS **và** false FAIL) |

Báo cáo slice: `AUDIT_A_DEPLOY.md`, `AUDIT_B_RUNTIME.md`, `AUDIT_C_MODELS.md`,
`AUDIT_D_SCIENCE.md`, `AUDIT_E_BACKEND.md`, `AUDIT_F_ADVERSARIAL.md`.
Ma trận nguyên nhân gốc hợp nhất: `ROOT_CAUSE_MATRIX.md`.

## 1. Đã sửa trong đợt này

### Phase 1 — bốn defect được giao

| ID | Nguyên nhân gốc | Trạng thái |
|---|---|---|
| P0-1 | `pc.sh` chạy `verify_run.py` bằng `python3` **host** (không có phụ thuộc dự án); host còn có hai khóa ledger xung đột nên việc xác minh ledger do container ghi là **bất khả thi về mật mã**, không chỉ sai môi trường | ĐÃ SỬA — `run_verifier_in_container` dùng `docker exec -T` và truyền tường minh môi trường khai báo của container (`Config.Env`); container không chạy → `fail_stage` có tên, không fallback host |
| P0-2 | `verify_run.py` chỉ ràng buộc category cho LIMN; không kiểm extraction đã chạy, không đối chiếu macro với category, không từ chối synthetic | ĐÃ SỬA — `metric_evidence_reasons` + `_extraction_reasons` + `_macro_reasons` + `_evidence_kind_reasons`; mọi model khác phải có so sánh publishable |
| P1-3 | `pc.sh` tái dùng container mà không kiểm mount thật → có thể đọc dataset khác | ĐÃ SỬA — so `docker inspect .Mounts` với `HOST_FASHIONIQ_ROOT`/`HOST_WORKBENCH_CHECKPOINT_ROOT`/`HOST_WORKBENCH_ARTIFACTS_ROOT` (realpath); thiếu mount → lý do có tên; không đọc được → UNKNOWN → fail closed |
| P2-4 | Nhánh sửa cấu hình gọi `run.sh up` hai lần | ĐÃ SỬA — một lần; test đếm đúng 1 |

### Phase 2–4 — defect do audit độc lập tìm thấy

| ID | Mức | Nguyên nhân gốc | Trạng thái |
|---|---|---|---|
| A-1/B-1 | P1 | Khóa ledger và khóa ký capability ưu tiên `<CIR_REPO_ROOT>/.workbench_ledger_key` — thư mục checkout **root-owned, không thuộc bind mount** → uid operator không ghi được và khóa **mất khi container được tạo lại** → mọi entry ledger cũ mất khả năng xác minh | ĐÃ SỬA — khóa mới tạo ở `$XDG_CONFIG_HOME/cir/` (bind mount, bền vững); legacy chỉ được dùng khi nó **đã tồn tại** (nó đã ký ledger hiện có); hai khóa khác bytes → từ chối. Chứng minh 3 case + container thật: khóa nằm ở `/artifacts/config/cir/workbench_ledger_key` |
| A-2/B-2 | P2 | `XDG_CONFIG_HOME` chỉ khai trong compose `environment:`, không có trong image `ENV` → `docker exec` không thấy → xác minh có thể dùng khóa khác | ĐÃ SỬA — khai trong image `ENV`; xác minh `docker exec` thấy `/artifacts/config` |
| E-1 | P1 | Boot lần đầu không có `workbench.duckdb` (gitignored) → `/api/runs` 503 vĩnh viễn; đúng tình trạng PC mới | ĐÃ SỬA — build index nền khi thiếu; index có sẵn giữ nguyên; build lỗi → 503 trung thực kèm lý do thật |
| E-2 | P1 | Export cohort Markdown format số không guard `None` → HTTP 500 | ĐÃ SỬA — render `N/A`; không bịa số |
| D-1 | HIGH | Hợp đồng model không phải LIMN yếu hơn hẳn (bỏ qua extraction/category/macro, dùng `any(...)`) | ĐÃ SỬA — dùng chung helper với LIMN |
| D-2 | HIGH | `select_reference` trả `None` khi hai row VERIFIED khớp cùng điều kiện → **mọi** run `ptha_mtst` hợp lệ bị từ chối, lý do không hành động được | ĐÃ SỬA — outcome có tên (`AMBIGUOUS`), nêu cả hai row và cách khắc phục; registry không bị sửa |
| D-3 | MED | `schemas/results.py` `Literal["experiment","mock"]` nhưng pipeline phát `data_kind="synthetic"` | ĐÃ SỬA — thêm `synthetic`; gate + verifier + `validate_results --strict-real` vẫn từ chối nó là evidence |
| D-4 | MED | `combiner_rn50x4_noft` đã đăng ký nhưng mọi reference UNVERIFIED → không phân biệt “không có gì để so sánh” với “xác minh thất bại” | ĐÃ SỬA — `NO_PUBLISHED_COMPARISON` là điều kiện có tên, tách khỏi thất bại |
| F-1 | P2 | `mutation_checks.sh` M4 đọc exit code `docker compose config`; daemon down → non-zero bị hiểu là “đã phát hiện” → PASS giả | ĐÃ SỬA — kiểm CLI/daemon trước; không chạy được → SKIP NOT VERIFIED |
| E-A1 | P1 | Container chạy nhưng API không trả 200 → `pc.sh` bỏ qua kiểm tra active-job → `run.sh up` có thể phá job đang chạy | ĐÃ SỬA — nhánh API-không-trả-lời hỏi cùng câu hỏi, đọc thêm `queue_state.json` (bind mount host đọc được); có job → từ chối; không đọc được → fail closed |
| D-1b | P1 (hồi quy nội bộ) | Hợp đồng D-1 từ chối cả run synthetic hợp lệ của harness vì gộp “execution có bằng chứng” và “có phải bằng chứng nghiên cứu” vào một field | ĐÃ SỬA — tách `verified` (execution được chứng thực) và `is_research_evidence`; `pc.sh` gate trên `is_research_evidence`; fixture harness in đúng schema contract + khai `synthetic_source` |

## 2. VERIFIED — hành vi đã kiểm bằng thực thi

- **Chuỗi xác thực hai chiều.** Report sửa sau ledger → từ chối; proof replay sang run khác → từ chối;
  ledger entry replay → từ chối; HMAC giả → chuỗi không verify; macro không khớp category → từ chối
  (LIMN); extraction chưa chạy → từ chối; NaN/ngoài khoảng/bool → từ chối; synthetic → không phải
  research evidence. Chiều ngược: run hợp lệ với Recall đúng biên **0.0** và **100.0** được chấp
  nhận; run LIMN khớp paper chứng nhận; `csmcir` có so sánh thật được chấp nhận.
- **Contract runtime thống nhất.** `WORKBENCH_RUNTIME_ADAPTATION=sm_120`: limn → `pytorch>=2.7,<3`
  (digest `042baec6…`), csmcir → `2.0.1` lịch sử (digest `f7f829a3…`). Biến toàn cục không ép model
  không khai báo.
- **Ràng buộc danh tính capability.** Hai interpreter → fingerprint khác; record của interpreter A bị
  từ chối khi B đang cấu hình; record hợp lệ tái dùng mà không probe lại.
- **Advisory vs authoritative.** Advisory: 0 lần hash nội dung; authoritative: hash đầy đủ, rewrite
  cùng size+mtime vẫn bị phát hiện. `checkpoint_availability(limn)` 11.06 s → 0.97 s; `sha256_file`
  129.751 → 0 lần.
- **Mount provenance.** `docker inspect .Mounts` khớp cấu hình; dataset `images/` và `image_splits/`
  read-only được kernel thực thi; mount sai/thiếu/không đọc được đều bị từ chối.
- **Không bịa dữ liệu per-query.** Không producer thật nào ghi `top_results`; chỉ mock loader.
- **UI không thể trình bày run không hợp lệ như bằng chứng đã chứng thực.**

## 3. PENDING — cần phần cứng đích

| Mục | Lý do chính xác |
|---|---|
| Xác thực GPU trên RTX 5070 Ti | máy này là RTX 3050 Laptop (cc 8.6) và **thiếu `nvidia-container-toolkit`**; `./pc.sh --doctor` báo PENDING đúng lý do |
| Thực thi kernel CUDA sm_120 trong môi trường LIMN cách ly | cần máy đích; danh sách arch neo trên nguồn torch, chưa có bản wheel hiện đại để đối chiếu thực nghiệm |
| Cài môi trường máy đích | `torch==1.12.1` lịch sử không có trên PyPI thường → cần index hoặc profile adapted |
| Suy luận LIMN thật + đánh giá FashionIQ thật | cần máy đích |
| `test_e2e14` (đường synthetic có GPU đã xác thực) | **fail ở cả commit phát hành `a517cf9`** — hồi quy cũ, không do đợt này; nằm trong 3 file e2e nặng vốn được deselect |

## 4. BLOCKED

Không có. Mọi P0/P1 đã xác nhận trong đợt này đều đã sửa và có test hồi quy.

## 5. KNOWN LIMITATIONS

1. **LIMN không xuất xếp hạng Top-K theo từng truy vấn.** Đường đánh giá thật chỉ sinh Recall theo
   category + macro. Tầng tiêu thụ (schema `ResultRun` v2, bảng `queries`/`top_results`, `/api/queries`,
   `/api/analysis`) đã xây xong, **thiếu nhà sản xuất thật**. Phân tích thất bại ở mức truy vấn
   **chưa vận hành được**. Không có kết quả truy hồi nào bị bịa. Chi tiết + 4 bước triển khai tối thiểu:
   `PER_QUERY_EXPORT_STATUS.md`; đã ghi rõ trong §15 của guide.
2. `checkpoint_by_id` resolve bundle về member đầu để giữ xử lý path/hash đồng nhất; caller chỉ đọc
   `filename` mà bỏ qua `bundle_members` sẽ vô tình đánh giá một category. Đường sản xuất hiện tại đúng.
   Rủi ro cấu trúc, chưa phải lỗi đang hoạt động.
3. `_published_reasons` bỏ qua so sánh khi report không mang metric nào mà reference công bố — nhưng
   điều đó giờ **bị từ chối** với lý do có tên, không phải bỏ qua im lặng.
4. Chưa có test nào chạy kernel sm_120/Blackwell thật; mọi test capability đều mock sm_89.

## 6. Bằng chứng acceptance (gắn với commit)

| Kiểm tra | Kết quả |
|---|---|
| Full pytest serial (loại 3 e2e nặng) | **1030 passed, 0 failed** (27 deselected) |
| `test_pc_launcher.py` | 34 passed |
| `test_release_blocker_invocation_proof.py` | 35 passed |
| `test_phase1_acceptance_scenarios.py` | 8 passed |
| `test_final_hardening_regressions.py` | 33 passed |
| Kiểm cú pháp Python toàn bộ first-party | `compileall` rc=0 |
| Kiểm cú pháp Bash (`pc.sh`, acceptance, release_verify, mutation) | `bash -n` rc=0 |
| TypeScript (`tsc -b`) | rc=0 |
| Docker Compose config | VALID |
| Docker image build | OK |
| Docker smoke: `docker exec` thấy `XDG_CONFIG_HOME`; khóa ledger tạo trên bind mount | `/artifacts/config/cir/workbench_ledger_key`, uid 1000 ghi được |
| `acceptance.sh` (live) | **ACCEPTANCE PASSED** |
| `release_verify.sh --offline` | không tiêu chí chưa đạt (5 pending) |
| `mutation_checks.sh` | **6/6 DETECTED** |
| `./pc.sh --self-check`, `--doctor`, `--verify-only`, `--help` | hoạt động; `--verify-only` trả rc=1 fail-closed khi thiếu toolkit |

## 7. Phán quyết

**ENGINEERING_READY / GPU_VALIDATION_PENDING.**

- Không còn P0/P1 đã xác nhận nào chưa sửa.
- Mọi test kỹ thuật và toàn vẹn khoa học sẵn có đều pass.
- Xác thực GPU thật trên RTX 5070 Ti **chưa** thực hiện — PENDING, không gọi là PASS.
- Lệnh duy nhất trên PC: `./pc.sh`
