# FINAL_RELEASE_FIX_REPORT.vi.md

Báo cáo bằng chứng phát hành — đợt sửa defect chặn phát hành
Nhánh: `research/cir-failure-workbench` · Baseline: `b45c068`

## 1. Defect đã sửa

| ID | Vị trí | Bản chất | Trạng thái |
|---|---|---|---|
| P0-1 | `pc.sh::validate_and_summarize` | Bịa `report_data` với `execution_status=EXECUTION_COMPLETED` khi thiếu báo cáo; in `HOÀN TẤT`/`CORROBORATED` không có chứng thực; chọn report bằng glob chuỗi con | ĐÃ SỬA |
| P0-2 | `evaluate_models.execute` + `jobs.py` | `INVOCATION_NO_OBSERVABLE_OUTPUT` và proof bị từ chối vẫn `return 0` → job publish `completed` | ĐÃ SỬA |
| P0-3 | `pc.sh::execute_evaluation` | Tái dùng mọi run khớp model/checkpoint/protocol, kể cả mock/synthetic/thiếu chứng thực, không kiểm provenance | ĐÃ SỬA |
| P1-4 | `release_verify.sh` | Tin field JSON thô của record capability; record tự viết tay đủ để báo PASS | ĐÃ SỬA |
| P1-5 | `pc.sh` container reuse | Chỉ cần HTTP 200 là tái dùng; không kiểm GPU override/adaptation/mount; không chặn restart khi có job đang chạy | ĐÃ SỬA |
| P1-6 | `PC_ONE_COMMAND_GUIDE.vi.md` | Khuyến nghị `nvidia-driver-550` — sai cho Blackwell | ĐÃ SỬA |
| P2-7 | `release_verify.sh` J1 | Run mock có nhãn bị coi là giả mạo → false failure | ĐÃ SỬA |

### Defect phát hiện thêm bởi vòng review độc lập (đã sửa)

| ID | Bản chất | Trạng thái |
|---|---|---|
| R-P0 | Queue tin chuỗi `PROOF_VALIDATED` trong file mà **evaluator child không tin cậy** tự viết được → job `completed` không có bằng chứng | ĐÃ SỬA — queue tự xác minh độc lập qua `verify_run.py` |
| R-P1 | `interpreter_fingerprint` tồn tại nhưng **không bao giờ được so sánh** (reader dùng field mà đường advisory để trống) → capability của interpreter này thoả mãn interpreter khác | ĐÃ SỬA |
| R-P1 | `bundle_manifest_digest` băm **đường dẫn tuyệt đối** → container ghi `/checkpoints/...`, host tính `<artifacts>/checkpoints/...` → run LIMN hợp lệ bị từ chối là "tampered" | ĐÃ SỬA |
| R-P2 | `pc.sh` active-job check fail-open: API lỗi → coi như "0 none" → restart container đang chạy job | ĐÃ SỬA |
| R-P2 | J1/J5–J9/J12 chỉ bắt stdout → `detail` rỗng khi fail, không có lý do | ĐÃ SỬA |

## 2. Cơ chế sửa

**Một nguồn xác thực dùng chung.** `workbench/scripts/verify_run.py` (mới) là nơi duy nhất trả lời
"run này có phải bằng chứng thật không". Exit 0 chỉ khi đồng thời: report tồn tại và parse được;
identity run/model/checkpoint/protocol khớp **chính xác**; sổ cái HMAC xác minh và có entry cho run
với `report_digest` khớp sha256 file, `return_code == 0`, `invocation_status == INVOCATION_COMPLETED`;
completion proof qua `validate_run_manifest` (checksum + environment identity); và với LIMN, đủ 3
category với R@10/R@50 hữu hạn trong [0, 100].

Ba nơi gọi cùng hàm đó: `pc.sh` (in kết quả), `jobs.py` (quyết định trạng thái job), và
`release_verify.sh` (bản ghi GPU). Không nơi nào tự định nghĩa lại luật yếu hơn.

**Child code giữ nguyên.** `evaluate_models.execute` vẫn `return code` (mã thoát gốc của evaluator)
như bằng chứng, nhưng ghi thêm verdict có máy đọc được
(`<model>_<checkpoint>_<run_id>_invocation_outcome.json` + trong run directory):
`PROOF_VALIDATED` / `NO_OBSERVABLE_OUTPUT` / `PROOF_REJECTED` / `EXECUTION_FAILED`. Queue đọc verdict
như đường nhanh rồi **tự xác minh lại** trước khi publish `completed`.

**Kiểm soát truy cập ghi/đọc capability.** Record capability lưu fingerprint **authoritative** — cùng
giá trị mà capability key được suy ra từ đó — nên cả người ghi và mọi người đọc tính ra cùng một
giá trị; record của interpreter khác bị từ chối. `release_verify.sh` không đọc field thô nữa mà gọi
`runtime.verify_environment(authoritative=True)` + loader record của runtime.

**False failure được coi ngang hàng false pass.** `manifest_digest` bỏ đường dẫn tuyệt đối khỏi
identity (giữ filename + sha256) để cùng một artifact không bị từ chối chỉ vì mount prefix khác.
J1 tách "clean first boot" / "labelled non-evidence only" / "existing valid results" và chỉ FAIL khi
có run **trình bày như bằng chứng thật mà thiếu chứng thực**.

## 3. Bằng chứng OLD → FAIL / FIXED → PASS

| Defect | Lệnh | OLD | FIXED |
|---|---|---|---|
| P0-1/02/03/05 (adv01–07,11) | `pytest workbench/tests/test_release_blocker_adversarial_final.py -q` | **12 failed** (12/12) | **12 passed** |
| P0-2 | `pytest workbench/tests/test_release_blocker_invocation_proof.py -q` | **14 failed** | **15 passed** |
| P1-4/P2-7 | `pytest workbench/tests/test_release_blocker_gpu_records.py -q` | **7 failed / 2 passed** | **9 passed** |
| R-P0 | probe verdict giả, không report/ledger/proof | job `completed` | job `failed`, lý do: `run … does not verify against its recorded evidence` |
| R-P1 | hai interpreter khác nhau | `fingerprint(a) == fingerprint(b)` → record bị dùng chéo | fingerprint khác nhau; record lạ bị từ chối |
| R-P1 (manifest) | cùng bytes, hai mount prefix | digest khác → run hợp lệ bị từ chối | digest bằng nhau; đổi bytes/đổi member set vẫn đổi |
| R-P2 | `docker info` lỗi | coi như "0 none" → restart | `UNKNOWN` → `fail_stage CONTAINER_REUSE` |

Bằng chứng P1-4 trên máy này: record viết tay → `[WARN] capability record(s) that are not verified
evidence: forge.json: the record is not signed with this deployment's operator key`,
verdict `GPU_VALIDATION_PENDING` (không phải PASS).

## 4. Kết quả suite

| Kiểm tra | Kết quả |
|---|---|
| Full serial (loại 3 file e2e nặng) | **994 passed, 0 failed** (27 deselected) |
| `test_pc_launcher.py` | **25 passed** |
| `test_release_blocker_adversarial_final.py` | **12 passed** |
| `test_release_blocker_invocation_proof.py` | **15 passed** |
| `test_release_blocker_gpu_records.py` | **9 passed** |
| `acceptance.sh` (live) | **ACCEPTANCE PASSED (all groups A–I executed)** |
| `release_verify.sh --offline` | không tiêu chí chưa đạt |
| `release_verify.sh` (live) | **J1 PASS** `boot state=labelled non-evidence only: 6 run(s), 0 corroborated real, 6 labelled non-evidence` |
| `mutation_checks.sh` | **6/6 mutant DETECTED** |
| Docker build + smoke | backend 200, `/api/models` **1.23 s** (16 model), `/api/runs` 503 trung thực, frontend 200 |
| `verify_run.py` trong image | run không tồn tại → `verified: false` + lý do, **rc=1** |

## 5. Phân tích theo truy vấn — trạng thái trung thực

**LIMN không xuất xếp hạng Top-K theo từng truy vấn.** Đường đánh giá thật chỉ sinh Recall theo
category và macro aggregate. Tầng tiêu thụ (schema `ResultRun` v2, bảng `queries`/`top_results` trong
DuckDB, `/api/queries`, `/api/analysis`) đã xây xong nhưng **thiếu nhà sản xuất thật**: chỉ mock
loader ghi schema đó.

Vì vậy: phân tích thất bại ở mức truy vấn **chưa vận hành được** với kết quả LIMN thật, và điều này
đã được ghi rõ trong guide (`PC_ONE_COMMAND_GUIDE.vi.md` §15) và trong
`PER_QUERY_EXPORT_STATUS.md` (kèm 4 bước triển khai tối thiểu). Không có kết quả truy hồi nào bị bịa.

## 6. Trạng thái phân loại

**VERIFIED:** đường xác thực dùng chung; quyết định trạng thái job; bản ghi GPU; J1 hai trạng thái
boot; contract container reuse; guide; 994 test; mutation checks; docker build + smoke.

**PENDING:** xác thực GPU thật trên RTX 5070 Ti; cài môi trường máy đích; suy luận LIMN thật; đánh
giá FashionIQ thật; tương đương số liệu paper.

**BLOCKED:** không có.

## 7. Phán quyết

**GPU_VALIDATION_PENDING** — toàn bộ test kỹ thuật sẵn có đều pass, nhưng thực thi trên RTX 5070 Ti
chưa được xác thực. Không có P0/P1 nào đã xác nhận còn tồn tại. PENDING không bị gọi là PASS.

## 8. Lệnh trên PC

```bash
git clone git@github.com:Le-Minh-Nhut/cir.git
cd cir
git checkout research/cir-failure-workbench
./pc.sh
```
