# BÁO CÁO PHÁT HÀNH ĐỐI KHÁNG — CIR FAILURE ANALYSIS WORKBENCH

Kết quả của chiến dịch hardening: **sửa → kiểm thử → xác thực hợp đồng thật → tấn công đối kháng →
audit toàn bộ nguồn → review độc lập → lặp lại**.

Mục tiêu không phải một báo cáo ấn tượng. Mục tiêu là phần mềm sống sót khi bị cố tình phá.

---

## A. SHA nguồn cuối cùng

| | |
|---|---|
| **TESTED_SOURCE_SHA** | `20afc55` (cây sạch, 0 file chưa commit) |
| Baseline đầu chiến dịch | `cf52311` |
| Mốc audit vòng 01 | `a2861b2` |
| Cây trước khi viết báo cáo | `5f36116` — toàn bộ suite: **1169 passed, 0 failed** (24:36) |

Nguồn sản xuất **không đổi** sau lần chạy acceptance cuối: báo cáo này chỉ sửa tài liệu.

## B. SHA commit phát hành

| Commit | Nội dung |
|---|---|
| `a2861b2` | Wave A — 4 defect đã nêu tên (A/B/C/D) |
| `04ae2ad` | Wave B — 22 defect từ audit vòng 01 |
| `26ca86b` | Ma trận bất biến I01–I20 dạng test chạy được |
| `5047e64` | Wave C — 6 defect từ audit vòng 02 |
| `5f36116` | Wave D — 5 defect từ slice định danh vòng 02 |
| `1a2b875` | Wave E — 6 defect từ red-team |
| `20afc55` | Wave F — **verifier tự trích xuất lại metric** (defect nghiêm trọng nhất) |

## C. Manifest nguồn

`workbench/docs/reviews/final_pc_audit/iterative_audit/round_01/SOURCE_MANIFEST.md` (bản đóng băng
`a2861b2`) và `.../round_02/SOURCE_MANIFEST.md` (cây cuối). **110 file sản xuất first-party**, mỗi
file có sha256. Regenerate bằng script đã dùng — không file nào được bỏ khỏi kiểm kê.

## D. Sử dụng subagent

**16 subagent** trên 4 model khác nhau (`gemini-scout`, `v41audit`, `v41review`, `gemini`), 8 + 4 + 4
theo ba vòng. Mỗi slice nhận brief chung (`local://AUDIT_BRIEF.md`): read-only, phải tái hiện trước
khi kết luận, phân biệt `CONFIRMED_BUG` với `UNVERIFIED_RISK`, không được tin docstring/comment/test.

**Mọi phát hiện do subagent báo đều được tôi tái hiện độc lập trước khi chấp nhận.** Điều này không
phải hình thức: **4 phát hiện bị bác bỏ**, và **2 phát hiện chỉ ra rằng chính bản sửa trước đó của
tôi sai**.

## E. Defect đã xác nhận: 43

### 4 defect đã nêu tên (wave A)

| | Kết quả |
|---|---|
| **A** Cache chứng thực cũ | Chứng thực còn hiệu lực sau khi checkpoint bị đổi. **Tái hiện**: sửa 1 member, giữ nguyên report/ledger/proof → `certified=true`; xoá cache → verifier từ chối |
| **B** Hợp đồng `pc.sh` ↔ `/api/models` | `pc.sh` kiểm `member.get('available')` — field API chưa bao giờ có → **mọi member bị coi là thiếu**, lệnh mặc định âm thầm rơi xuống một danh mục |
| **C** Định danh bằng chứng phát hành | `GPU_RUNTIME`/`MODEL_EVALUATION`/`SCIENTIFIC_OUTPUT` PASS trên bằng chứng của **bất kỳ model nào** |
| **D** Banner frontend sai | Evaluation aggregate-only đã chứng thực bị công bố là "no real evaluation results are loaded" |

### 22 defect từ audit vòng 01 (8 slice, cây đóng băng `a2861b2`)

Hai defect **phá chính bản sửa wave A** — đây là giá trị thật của vòng audit:

- **R01-1 (P0)** Token cache dùng `size:mtime` của member — cả hai đều **ghi được** bởi người sửa
  file. Ghi đè cùng độ dài + `os.utime` → chứng thực giả vẫn còn. **Tái hiện.** Token giờ là **digest
  nội dung**, memo theo `(ctime, mtime, size, inode)`; `ctime` không thể đặt lùi bằng userspace.
- **R01-4 (P0)** Token hash qua `sha256_file` của module `main.py` — seam mà test HTTP patch thành
  hằng số. Hash qua verifier, không qua seam.
- **R01-2 (P0)** Token giải root từ `checkpoint_variants[0]` → mọi checkpoint đơn file hash ra
  `"unresolved"`, khoá đó đóng băng vĩnh viễn.
- **R01-3 (P0)** Ledger hỏng → mọi endpoint trả `200 []`, không phân biệt được với workspace rỗng
  (I02/I17). Giờ raise `ledger_corrupt`; ledger **thiếu** vẫn là câu trả lời rỗng hợp lệ.
- **R01-5 (P0)** Verdict `RELEASE_BLOCKED` **exit 0** — bộ đếm bash chỉ đi J1–J12, `TARGET_BINDING`
  không có J-check (I20).
- **R01-6 (P0)** `checkpoint_by_id` lưu `bundle_members` là **id** trong khi 5 consumer đọc như
  **record**; `completion_validator` nuốt `TypeError` trong `except` trần và rơi về `Path('.')` →
  proof của bundle được xác thực **theo thư mục làm việc**.
- **R01-7 (P0)** `/api/jobs` tự dựng lại record bundle và **bỏ hết member fact** → bundle LIMN **đầy
  đủ** bị từ chối là thiếu.
- **R01-8..R01-22 (P1/P2)** Verifier **xoá** artifact của operator; `TARGET_BINDING` PASS với
  checkpoint/protocol hư cấu; tuỳ chọn CLI bị bỏ im lặng; cache môi trường thiếu host/driver/device
  (I08/I13); interpreter chỉ đóng góp `size:mtime`; `run.sh` parse `.env` khác `pc.sh`; fallback
  `run_id` lấy report mới nhất bất kể định danh (I05/I06/I07); 6 defect frontend (banner, nhãn gate,
  `ready` mặc định `true` với bundle rỗng…); fixture `pc_launcher` **bịa field `available`** và thiếu
  9 field thật — đây là lý do launcher hỏng mà vẫn pass test suốt một release.

### 4 phát hiện bị BÁC BỎ (ghi lại vì sao)

| Cáo buộc | Lý do bác |
|---|---|
| `/api/models` dùng key `bundles`, nên `pc.sh` đọc list rỗng | **Sai.** `main.py:167-175` ghi `entry["checkpoint_bundles"]` có `members`. Harness của auditor dựng payload dưới key khác. |
| Gate LIMN không bao giờ chứng thực được run LIMN | **Sai với đường thật.** `run_meta` do `evaluate_models.py:148-170` dựng khớp reference và đạt `ALERT_METRIC_DEVIATION` không có blocker checkpoint. Auditor dùng `dataset: "FashionIQ"` trong khi sản xuất ghi `"fashioniq"`. |
| `MACRO_TOLERANCE_PP` là slack của một phép so khớp chính xác | **Chấp nhận một phần, chỉ là lỗi tài liệu.** Số học của auditor đúng; giá trị bound không đổi (là bound của chính parser), rationale đã viết lại cho đúng sự thật. |
| Report `checkpoint_sha256` bị sửa vẫn được chấp nhận (F2) | **Sai.** Sửa field đó đổi byte của report → digest trong ledger không còn khớp → bị bắt. |

### Defect vòng 02 + red-team: 17

Đáng chú ý nhất, theo mức độ:

- **Verifier không bao giờ trích xuất lại metric** (P0, red-team A). `verify_run` chỉ hỏi chính
  report xem nó đã trích xuất chưa — `extraction_status` là field **run tự viết về mình**. Tái hiện:
  report khai `EXTRACTED_VERIFIED` mang số của paper, `stdout.log` **không có dòng metric nào** →
  vẫn được phục vụ với `is_research_evidence: True`. Mọi consumer của cờ đó (claim thành công của
  `pc.sh`, `SCIENTIFIC_OUTPUT` của verifier, badge frontend) đang đọc **lời tự khai**. Đã sửa: verifier
  parse lại chính log đã được `evidence_digest` ràng buộc, so với số report khai.
- **`SCIENTIFIC_OUTPUT` vượt qua check của chính nó** (P0). Dimension được set PASS chỉ vì có evidence,
  bỏ qua J7 → với J7 PENDING, verdict đạt `RELEASE_READY` khi câu hỏi khoa học chưa được trả lời (I20).
- **`_proof_stamp`/`_ledger_stamp` vẫn dùng `size:mtime`** (P0) — đúng lớp tấn công đã bị giết hai
  lần ở chỗ khác nhưng còn sống ở proof. Sửa 1 ký tự trong proof + khôi phục mtime → `certified=true`
  trong khi verifier từ chối. Giờ là digest nội dung.
- **Presence/digest shortcut tin vào size** (P0, hai chỗ còn lại). Ghi đè cùng độ dài + mtime khôi phục
  → đọc là "present" **và** sinh `official_sha256_match: True`, không hash byte nào, trong khi verifier
  từ chối cùng file. Giờ yêu cầu thêm `ctime`.
- **`official_sha256_match` FALSE trên bundle LIMN nguyên vẹn** (P1/I20). Record bundle thừa hưởng
  digest **của anchor member**, so với digest **thư mục** → FALSE, và `pc.sh` từ chối đúng field đó →
  bản tải đúng bị chặn.
- **`NVIDIA_VISIBLE_DEVICES` không nằm trong host identity** (P0/I13). Chỉ có cách viết CUDA, nên
  capability ghi với "mọi device" vẫn `AUTHORIZED` khi bị giới hạn về một device hoặc không device nào.
- **Runtime adaptation không nằm trong target binding** (P1/I07). Run chạy trên stack lịch sử chứng
  thực target đã adapt `sm_120`.
- **Blocker "bundle incomplete" áp lên cả checkpoint đơn danh mục** (P1) → `./pc.sh --checkpoint` không
  dùng được; nút per-checkpoint của frontend bị disable.
- **Chỉ số ledger keyed theo digest đơn** (P1/I18) → hai report trùng byte bị từ chối ở API trong khi
  verifier chấp nhận.
- **`candidates()` dựng lại bundle record bằng tay** ở hai chỗ (P1) → `evaluate_models.py --checkpoint
  base_iter0_all_categories` báo "checkpoint missing" trên bundle đầy đủ.
- **`POST /api/jobs` rò `ValueError`** (P1/I17) → 500 trần không lý do.
- **Parser `.env`**: quoted value + comment bị từ chối; bare key không `=` bị đọc thành giá trị của
  chính nó (**regression do tôi gây ra** và do red-team bắt); resolver thư mục queue cắt quote mà
  không cắt comment → báo "0 job đang chạy" (I10).

## F. Nguyên nhân gốc và cách sửa

Một mẫu lặp lại chi phối gần hết các defect: **một sự thật được tính lại ở hai nơi, và hai nơi bất
đồng.** Cách sửa luôn là quy về **một nguồn duy nhất**, không phải thêm một lần kiểm nữa.

| Nguyên nhân gốc | Sửa |
|---|---|
| `size:mtime` được tin như bằng chứng về nội dung | Digest nội dung, memo theo `ctime` (thứ userspace không đặt lùi được) |
| Record bundle được dựng lại bằng tay ở 4 nơi | Một resolver `checkpoint_by_id`; cả 4 nơi route qua nó |
| Bundle path suy từ `.parent` ở 3 nơi | `checkpoint_path` là nguồn duy nhất cho path của artifact |
| Report tự khai đã trích xuất metric | Verifier parse lại log đã được digest ràng buộc |
| Dimension PASS độc lập với check của nó | Promotion chỉ khi check của chính nó PASS |
| Fixture bịa schema API (`available`) | Fixture sinh từ **producer thật** + test so **key set** |
| `pc.sh`/`run.sh` tự viết lại parser `.env` | Cùng thứ tự comment-trước-quote như compose |
| Verdict và exit code tính ở hai nơi | Report block sở hữu exit code |

## G. Test hồi quy

| Test | Bảo vệ |
|---|---|
| `test_http_api_layer.py::test_a_checkpoint_edit_invalidates_an_already_issued_certificate` | R01-1/I08 — sửa member → từ chối; khôi phục → chứng thực lại (cả hai chiều) |
| `...::test_an_unchanged_checkpoint_does_not_re_enter_the_verifier_per_request` | Chiều FALSE-FAIL: bằng chứng không đổi không được vào lại verifier |
| `...::test_the_checkpoint_change_token_covers_every_declared_member` | Mọi member của bundle, không chỉ member đầu |
| `test_release_invariants.py::test_i01b_the_verifier_re_extracts_metrics_from_the_runs_own_output` | Defect nghiêm trọng nhất: report khai số, output phải khớp |
| `test_release_blocker_gpu_records.py::test_a_dimension_may_not_pass_when_its_own_check_did_not` | J7 PENDING/FAIL không thể thành PASS |
| `...::test_a_verified_record_for_another_model_does_not_certify_the_target` | I05 |
| `...::test_evidence_for_another_identity_cannot_certify_the_declared_target` | I05/I06/I07 |
| `...::test_a_declared_target_that_is_not_in_the_registry_blocks_the_release` | I20 + exit code |
| `...::test_frontend_banner_distinguishes_aggregate_only_from_absent_evidence` | 8 trạng thái UI |
| `...::test_frontend_gate_fails_closed_and_distinguishes_refusal` | Fail-closed + `refused` riêng biệt |
| `test_pc_launcher.py::test_the_launcher_fixture_carries_the_real_api_schema` | Fixture bịa schema API sẽ fail |
| `...::test_a_limn_bundle_with_a_missing_member_is_refused_not_degraded` | I04: từ chối, không thoái hoá |
| `...::test_env_quoted_port_with_a_trailing_comment_is_valid` | `.env` hợp lệ không bị từ chối |
| `...::test_env_bare_word_without_an_assignment_is_skipped` | Regression của chính tôi |
| `test_e2e_release_chain.py` | Chuỗi thật đầu-cuối + tamper→refused→restore→re-certified |

## H. Ma trận hợp đồng liên module

Kiểm bằng **producer thật → consumer thật**, không mock tự viết:

| Biên | Producer | Consumer | Kết quả |
|---|---|---|---|
| `/api/models` → `pc.sh` | `checkpoint_availability` | snippet chọn trong `pc.sh` | VERIFIED — 7 hình dạng (đầy đủ, thiếu member, không runnable, sha mismatch, thiếu key `members`, chọn đơn danh mục, id lạ) |
| `checkpoint_by_id` → 6 consumer | registry resolver | `checkpoint_missing_paths`, `checkpoint_availability`, `guarded_plan`, `completion_validator`, adapter LIMN, `candidates()` | VERIFIED sau wave C/D |
| `/api/models` → frontend | route thật | `App.tsx`, `types.ts` | VERIFIED (`ptha_mtst` thiếu `literature_split_label` — P3, ghi nhận) |
| `/api/benchmark-runs` → gate/evidence | receipt 4 câu hỏi | `gate.ts`, `evidence.ts` | VERIFIED |
| Fixture `models.json` → availability thật | producer thật | fixture | VERIFIED — 0 thiếu, 0 thừa, không còn `"available"` |
| `run.sh` → `/api/jobs` → `checkpoint_by_id` | route thật | resolver | VERIFIED |
| Evaluator → parser → report → proof → ledger → verifier → API → UI | chuỗi thật | | VERIFIED (`test_e2e_release_chain.py`) |

## I. Ma trận bất biến I01–I20

`workbench/tests/test_release_invariants.py` — **22 test, mỗi bất biến một test có tên**, và chính
tính đầy đủ của ma trận cũng là một test (thiếu I nào, hoặc test mồ côi, đều fail).

**Hai mục UNVERIFIED, ghi rõ lý do — không mục nào bị biến thành PASS:**

| | Lý do |
|---|---|
| **I13 (chiều dương)** | Máy này không có NVIDIA container toolkit và không có GPU Blackwell → không thể tạo capability probe thật. Chiều âm (record giả mạo bị từ chối) **có** được kiểm. |
| **I11 (chiều dương)** | Cần docker daemon thật với compose stack để mount-inspect container sống; parser và đường từ chối được kiểm trên output inspect tổng hợp. |

## J. Kết quả pytest

| Lần chạy | Cây | Kết quả |
|---|---|---|
| Cuối, đầy đủ | `5f36116` | **1169 passed, 0 failed** (24:36) |
| Sau wave F (re-extraction) | `20afc55` | xem `M` — 5 fixture builder phải bắt đầu phát output thật |
| Heavy E2E (`*_master_e2e`, `autopilot_e2e`) | gồm trong lần chạy đầy đủ | pass — không deselect, không skip, không xfail |
| `test_release_invariants.py` | `20afc55` | 22 passed |

**Không test nào bị deselect/skip/xfail để giấu lỗi.** Hai test trong
`test_final_hardening_regressions.py` khẳng định verifier **xoá** artifact đã được sửa để khẳng định
đúng hợp đồng (báo cáo, không xoá) — đây là **đổi kỳ vọng vì hợp đồng đã được sửa**, không phải để
test pass.

## K. Kết quả tích hợp thật

`test_e2e_release_chain.py` — chuỗi mission yêu cầu, trên component thật:

```
/api/models THẬT → snippet chọn pc.sh THẬT → queue THẬT → evaluator có kiểm soát
  → completion proof THẬT → evidence ledger THẤT (HMAC) → verify_run THẬT
  → /api/benchmark-runs THẬT → module frontend THẬT
  → SỬA 1 member checkpoint → phải bị TỪ CHỐI → khôi phục → chứng thực LẠI
```

Chỉ thay đúng hai thứ không có trên máy này: tiến trình evaluator, và hai điều kiện triển khai
(verdict môi trường model, và pin nguồn). **Đường chứng thực hoàn toàn thật.**

`acceptance.sh`: **ACCEPTANCE PASSED** (một nhóm SKIP trung thực: hàng đợi rỗng không chứng minh được
enum status — I19).
`release_verify.sh`: verdict `ENGINEERING_INCOMPLETE`, **0 check fail**, exit 0 — verdict đúng cho máy này.

## L. Mutation & fault injection

`mutation_checks.sh`: **MUTATION CHECKS PASSED** — M1 (JSON lỗi), M2 (**chọn bundle thoái hoá về
`checkpoints[0]`**), M3 (metric thiếu không mang nhãn deviation), M4 (compose từ chối host path rỗng;
mount RO bảo vệ dữ liệu canonical), M5 (GPU reservation khai báo).

Tấn công cache do tôi chạy trên cây cuối: ghi đè cùng độ dài **có** khôi phục mtime → từ chối; đổi
member → từ chối; xoá member → từ chối; khôi phục → chứng thực lại; 400 revision → cache giữ đúng
bound 256; 200 request không đổi → **0 lần hash lại**.

## M. Defect chưa giải quyết

| | |
|---|---|
| `ptha_mtst` thiếu `literature_split_label` trong `/api/models` (P3) | `types.ts` khai `"original" \| "val" \| null`; API trả `undefined`. Frontend không dereference field này cho model đó nên **không có hành vi sai**, nhưng hợp đồng type lỏng hơn thực tế. Không sửa: sửa registry là thay đổi dữ liệu không thuộc phạm vi release này. |

**Không còn defect nào ở mức P0/P1/P2 đã xác nhận mà chưa sửa.**

## N. Phụ thuộc bên ngoài còn chờ

| | |
|---|---|
| GPU đích RTX 5070 Ti (Blackwell sm_120) | Không có trên máy này (RTX 3050 Laptop, cc 8.6; thiếu nvidia-container-toolkit) → `GPU_RUNTIME` PENDING |
| Suy luận model thật | Cần checkpoint 1.8 GB + GPU đích → `MODEL_INFERENCE` PENDING |
| Đánh giá FashionIQ thật | → `MODEL_EVALUATION`, `SCIENTIFIC_OUTPUT` PENDING |
| Dữ liệu Top-K per-query | LIMN không xuất → `QUERY_LEVEL_ANALYSIS_PENDING` |

## O. Phán quyết phát hành

**ENGINEERING_READY** — mọi test kỹ thuật khả dụng pass, các hợp đồng liên module chính đã được xác
thực bằng producer/consumer thật, không còn defect P0/P1/P2 đã xác nhận.

**GPU_VALIDATION_PENDING** · **MODEL_INFERENCE_PENDING** · **MODEL_EVALUATION_PENDING** ·
**QUERY_LEVEL_ANALYSIS_PENDING**

**Không** `RELEASE_READY`: verdict đó đòi bằng chứng thực thi thật đã xác thực trên GPU đích, và máy
này không có GPU đích, không có suy luận thật, không có đánh giá thật. Đó là hành vi trung thực.

**ENVIRONMENT_BLOCKED** trên chính máy này: hai file khoá ledger xung đột khiến ledger sống không
dùng được, nên live acceptance đầy đủ không thể PASS ở đây (nhóm G bị SKIP, không được tính là PASS).

## P. Lệnh chạy chính xác trên RTX 5070 Ti

```bash
git clone git@github.com:Le-Minh-Nhut/cir.git
cd cir
git checkout research/cir-failure-workbench
git reset --hard 20afc55          # TESTED_SOURCE_SHA
./pc.sh
```

Kiểm tra trước khi chạy dài:

```bash
./pc.sh --doctor                  # preflight chỉ đọc
./pc.sh --verify-only             # probe GPU trong container, không đánh giá
```

Kỳ vọng trên máy đích, nếu mọi thứ đúng: `./pc.sh` chạy **bundle LIMN đầy đủ ba danh mục**
(`base_iter0_all_categories`) và **từ chối nêu tên member** nếu thiếu, chứ không âm thầm chạy một
danh mục. Nếu muốn chủ động chạy một danh mục (KHÔNG phải tái lập đầy đủ):

```bash
./pc.sh --checkpoint base_iter0_dress
```
