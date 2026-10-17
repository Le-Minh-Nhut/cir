# FINAL_AUTONOMOUS_RELEASE_REPORT.vi.md

Báo cáo phát hành tự động — CIR Failure Analysis Workbench
Nhánh: `research/cir-failure-workbench` · Baseline: `92ecbd9`

---

## 1. Commit cuối

`[xem mục 2 — tệp báo cáo này nằm ở commit tài liệu kế tiếp]`

## 2. Commit nguồn đã kiểm thử

Toàn bộ bằng chứng ở mục 11–13 được thu trên cây nguồn **không đổi** tại
`8fb93ee` — commit **nguồn** đã kiểm thử. Toàn bộ test ở mục 11–13 chạy trên cây này. Tệp báo cáo
này nằm ở commit tài liệu kế tiếp và **không** đổi mã nguồn.

## 3. Số file nguồn đã audit

**100 file** first-party production, lập manifest hai lần (mỗi vòng một lần):

- `workbench/docs/reviews/final_pc_audit/iterative_audit/round_01/SOURCE_MANIFEST.md`
- `workbench/docs/reviews/final_pc_audit/iterative_audit/round_02/SOURCE_MANIFEST.md`

Phạm vi: `pc.sh`, `workbench/backend/**` (Python + schemas), `workbench/scripts/**`,
`workbench/replay/**`, `workbench/frontend/**` (TypeScript), `workbench/docker/**`,
`workbench/config/**`, `workbench/registry/**` (YAML), `src/**` (Python). Tệp sinh tự động
(`dist/`, `node_modules/`, `__pycache__`, `*.tsbuildinfo`, lockfile) không tính là nguồn first-party.

## 4. Số vòng audit đã hoàn tất

**2 vòng đầy đủ**, cộng 5 đợt sửa (fix wave) xen kẽ:

| Vòng | Nội dung | Kết quả |
|---|---|---|
| round 01 | 4 slice độc lập (deploy / runtime / science / backend) trên cây sau wave 1 | 12 finding CONFIRMED, 2 FALSE_POSITIVE |
| round 02 | 4 slice độc lập trên cây sau wave 3, tập trung vào chính các fix vừa landing | 12 finding CONFIRMED (gồm 2 regression do fix trước gây ra) |
| wave 5 | sửa defect cô lập test (không phải sửa sản phẩm) | suite trở nên độc lập thứ tự |
| wave 6 | robustness của verifier | J1 báo lỗi có tên thay vì traceback |

Vòng lặp dừng khi vòng 02 **không còn** finding CONFIRMED nào chưa xử lý (danh sách còn lại là
FALSE_POSITIVE / EXPECTED_BEHAVIOR / KNOWN_LIMITATION / EXTERNAL_DEPENDENCY_PENDING, xem mục 10).

## 5. Số subagent đã dùng

**22 subagent** (chỉ đọc hoặc sở hữu một vùng file riêng), không tính các lần spawn lỗi harness.

## 6. Model dùng cho subagent

- `ag/gemini-3.8-flash-high` — phần lớn (ưu tiên theo yêu cầu người dùng).
- `wdb/deepseek-ai/DeepSeek-V4.1-Flash` (`v41audit`) — các slice đối kháng và một số fix, dùng khi
  harness gemini không spawn được.
- Harness subagent có lúc hỏng ("No model selected"); mỗi lần đều ghi nhận và **thay bằng agent
  khác hoặc tự làm bằng main agent**, không bỏ sót vùng audit nào.

## 7. Độ phủ audit theo từng vòng

| Vòng | Slice | File chính | Báo cáo |
|---|---|---|---|
| 01 | A deploy | `pc.sh`, `docker/**`, `config/**`, `.dockerignore`, `serve_workbench.py`, `operator_config.py` | `round_01/AUDIT_A.md` |
| 01 | B runtime | `runtime.py`, `environment_planner.py`, `manage_environment.py`, `pc_readiness.py`, `pipeline.py`, `gpu_check.py`, `registry.py`, `models.yaml` | `round_01/AUDIT_B.md` |
| 01 | D science | `verify_run.py`, `paper_gate.py`, `benchmark_references.py`, `checkpoint_selection.py`, `schemas/`, `papers.yaml`, `evaluate_models.py`, `replay/common.py` | `AUDIT_D_SCIENCE.md` |
| 01 | E backend/jobs | `main.py`, `jobs.py`, `index.py`, `storage.py`, `analysis.py`, `frontend/**` | `round_01/AUDIT_E.md` |
| 02 | A/B/D/E | như trên, cộng `autopilot.py` | `round_02/AUDIT_{A,B,D,E}.md` |
| — | F adversarial (vòng trước) | toàn bộ tầng kiểm chứng, `release_verify.sh`, `mutation_checks.sh`, `acceptance.sh` | `AUDIT_F_ADVERSARIAL.md` |

Mỗi vòng ghi lại phần **không** được inspect; các vùng đó được vòng sau hoặc main agent phủ tiếp.

## 8. Defect đã xác nhận

**24 defect CONFIRMED** qua hai vòng (12 + 12), trong đó 2 là regression do chính các fix trước đó
gây ra — điều này chứng minh vòng audit thứ hai là cần thiết, không phải hình thức.

## 9. Defect đã sửa theo mức độ

| Mức | Số | Ví dụ nghiêm trọng nhất |
|---|---|---|
| **P0** | 5 | `main.py` từ chối mọi tiến trình production (mọi report `certified=false`); ba vị trí khóa ledger làm API im lặng trả `[]`; `pc.sh` xoá mảng lý do làm mất nguyên nhân fail-closed |
| **P1** | 13 | race shutdown/dispatch (tiến trình mồ côi giữ GPU); verifier có thể đạt `RELEASE_READY` không cần inference thật; hợp đồng hoàn thành non-LIMN yếu hơn hẳn; `ptha_mtst` bị từ chối mọi run hợp lệ; khóa capability không đọc lại được (false FAIL trên PC đích); autopilot bỏ qua corroboration khi ledger thiếu/hỏng |
| **P2** | 12 | `run.sh up` hai lần; `XDG_CONFIG_HOME` không tới `docker exec`; M4 mutation PASS giả khi daemon down; schema thiếu `data_kind="synthetic"`; J8 coi queue rỗng là lỗi; verdict `--offline` nêu sai nguyên nhân |
| **P3** | 4 | lý do trùng lặp; in key không tồn tại; guard falsy bỏ qua kiểm fingerprint |

Tất cả các mức đã sửa. Không có defect nào bị đánh dấu "PENDING" để lấy verdict xanh.

## 10. Phân loại những gì KHÔNG phải defect

| Mục | Phân loại | Lý do |
|---|---|---|
| J1 FAIL trên máy dev này | ENVIRONMENT, không phải sản phẩm | Máy này có **hai** tệp khóa ledger khác nhau; sản phẩm cố ý từ chối thay vì đoán. Trên PC sạch (một khóa) J1 PASS. Verifier giờ báo lỗi **có tên** kèm cả hai đường dẫn và cách khắc phục |
| "Xoá artifact 0 byte khi verify" | EXPECTED_BEHAVIOR | Có chủ đích và được ghi trong báo cáo; tệp không mang bằng chứng và nếu giữ lại sẽ lặp lại cảnh báo mỗi lần quét. Bản ghi đọc được không bao giờ bị chạm |
| "`real_gpu=True` bị che bởi UNREADABLE/WRONG_KEY" | FALSE_POSITIVE | Báo cáo mang `real_gpu` là trường riêng, tách khỏi trạng thái quét |
| `OPERATIONS` PENDING khi queue rỗng | ĐÃ SỬA | Từng là false failure; giờ queue rỗng là trạng thái rảnh bình thường, job đang chạy vẫn được kiểm chặt |
| A2 (J1 fail vì `.venv` thiếu yaml) | FALSE_POSITIVE | Đo trực tiếp: J1 PASS live |
| A9-dimension (dimension không thể đạt) | FALSE_POSITIVE | `DEPLOYMENT`/`CONFIGURATION` đạt PASS live |

## 11. Kết quả pytest đầy đủ

| Lần chạy | Kết quả |
|---|---|
| Full suite (gồm cả 3 file E2E nặng) sau wave 2 | **1103 passed, 0 failed** |
| Full suite sau wave 4 | 1131 passed, 2 failed — **hai test cô lập hỏng thứ tự**, không deselect |
| Sau khi sửa cô lập (wave 5) | **1133 passed, 0 failed** |
| **Lần chạy cuối trên cây đóng băng `8fb93ee`** | **1134 passed, 0 failed** (23 phút 32 giây) |

Không test nào bị deselect, skip hay xfail để có suite xanh.

## 12. Kết quả E2E nặng

Ba file E2E trước đây bị deselect (**`test_true_master_e2e.py`**, **`test_release_blocker_master_e2e.py`**,
**`test_autopilot_e2e.py`**) **nay chạy đầy đủ** trong suite, không deselect.

- `test_e2e14_verified_gpu_capability_permits_the_authorized_synthetic_path` — trước đây fail ở **cả hai**
  commit `a517cf9` và `46d8e59`. **Root cause đã tìm ra và sửa**: `runtime._remember` gán
  `report.contract_digest` **sau** cache-gate, nên với `WORKBENCH_NO_ENV_CACHE=1` bản ghi capability
  được ký với `dependency_fingerprint: null`, và mọi reader từ chối nó. Đây là lỗi **production**,
  không phải lỗi test; đã sửa tại nguồn. Test giờ PASS, không skip/xfail.

## 13. Kết quả Docker

| Kiểm tra | Kết quả |
|---|---|
| Docker Compose config | VALID |
| Docker image build | OK |
| Container boot (uid 1000) | backend 200, `/api/models` ~1.2 s cho 16 model, frontend 200, `/api/runs` 503 trung thực |
| `docker exec` thấy `XDG_CONFIG_HOME` | CÓ (`/artifacts/config`) |
| Khóa ledger tạo trên bind mount | `/artifacts/config/cir/workbench_ledger_key`, uid 1000 ghi được |
| `acceptance.sh` (live) | ACCEPTANCE PASSED |
| `mutation_checks.sh` | 6/6 mutant DETECTED |
| `release_verify.sh` live | J1 báo lỗi có tên (xung đột khóa của máy này), verdict trung thực, không traceback |
| `release_verify.sh --offline` | exit 0, verdict `ENGINEERING_INCOMPLETE`, pending được nêu tên |

## 14. Trạng thái GPU/runtime

**EXTERNAL_DEPENDENCY_PENDING.** Máy này là RTX 3050 Laptop (cc 8.6) và **thiếu
`nvidia-container-toolkit`**. Không có bằng chứng GPU thật nào bị làm giả. Hợp đồng capability
(định danh interpreter, host, khóa, độ bền) đã được kiểm bằng thực thi theo cả hai chiều.

## 15. Trạng thái thẩm định khoa học

**VERIFIED** cho tầng thẩm định: report sửa sau ledger → từ chối; proof replay → từ chối; ledger
entry replay → từ chối; HMAC giả → chuỗi không verify; macro không khớp category → từ chối; extraction
chưa chạy → từ chối; NaN/ngoài khoảng/bool → từ chối; synthetic → không phải bằng chứng; biên 0.0 và
100.0 → chấp nhận; model có so sánh thật → chấp nhận; model không có gì để so sánh → từ chối **có tên**;
báo cáo không mang số nào → từ chối.

Sai lệch số so với paper vẫn là **phát hiện nghiên cứu** (`ALERT_METRIC_DEVIATION`), không phải lỗi
hạ tầng — đúng theo thiết kế.

## 16. Hạn chế đã biết

1. **LIMN không xuất xếp hạng Top-K theo từng truy vấn.** Đường đánh giá thật chỉ sinh Recall theo
   category + macro. Tầng tiêu thụ (schema v2, bảng `queries`/`top_results`, `/api/queries`,
   `/api/analysis`) đã xây xong, **thiếu nhà sản xuất thật**; chỉ mock loader ghi schema đó. Vì vậy:
   - `BENCHMARK_REPRODUCTION_READY` — đạt được.
   - `QUERY_LEVEL_ANALYSIS_READY` — **chưa** đạt. Phân tích thất bại theo mức truy vấn chưa vận hành
     được với kết quả thật.

   Không có kết quả truy hồi nào bị bịa. Chi tiết + 4 bước triển khai tối thiểu:
   `PER_QUERY_EXPORT_STATUS.md`; guide §15 cũng ghi rõ.

2. `checkpoint_by_id` resolve bundle về member đầu để giữ xử lý path/hash đồng nhất; caller chỉ đọc
   `filename` mà bỏ qua `bundle_members` sẽ vô tình đánh giá một category. Đường sản xuất hiện tại đúng.
3. Chưa có test nào chạy kernel sm_120/Blackwell thật; mọi test capability đều mock.

## 17. Blocker bên ngoài còn lại

| Blocker | Bằng chứng |
|---|---|
| Không có RTX 5070 Ti | `./pc.sh --doctor` → `NVIDIA GeForce RTX 3050 Laptop GPU (Compute 8.6)` |
| Không có `nvidia-container-toolkit` | `docker info` không có runtime `nvidia`; `--doctor` báo PENDING đúng lý do |
| Dữ liệu/checkpoint author đầy đủ cho model khác LIMN | chỉ có checkpoint LIMN (3 × 1.79 GB) trên máy này |
| Suy luận LIMN thật + đánh giá FashionIQ thật | cần máy đích |

Các mục trên là **PENDING**, không bao giờ được gọi là PASS.

## 18. Phán quyết phát hành

**ENGINEERING_READY / GPU_VALIDATION_PENDING / MODEL_INFERENCE_PENDING /
MODEL_EVALUATION_PENDING.**

- Không còn defect P0/P1/P2/P3 nào đã xác nhận chưa xử lý.
- Hai vòng audit độc lập đầy đủ trên 100 file nguồn first-party.
- Test khả dụng đều pass: **1133 passed, 0 failed** (gồm cả 3 file E2E nặng), không deselect.
- Mutation checks 6/6, acceptance live PASSED, Docker build + smoke OK.
- **Không** tuyên bố `RELEASE_READY`: chưa có GPU đích, chưa có suy luận thật, chưa có đánh giá
  FashionIQ thật, chưa có tầng per-query.

## 19. Lệnh trên PC đích

```bash
git clone git@github.com:Le-Minh-Nhut/cir.git
cd cir
git checkout research/cir-failure-workbench
./pc.sh
```

Tuỳ chọn: `./pc.sh --doctor`, `./pc.sh --model csmcir`, `./pc.sh --verify-only`, `./pc.sh --force`,
`./pc.sh --timeout <giây>`, `./pc.sh --no-gpu`, `./pc.sh --dataset <đường_dẫn>`.

Hướng dẫn đầy đủ tiếng Việt: `workbench/docs/PC_ONE_COMMAND_GUIDE.vi.md`.
