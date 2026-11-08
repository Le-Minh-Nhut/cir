# TEST_TRIAGE — 33 lỗi tại `b0f38bd`: khoanh vùng, tìm nguyên nhân gốc, sửa

Phạm vi: toàn bộ 33 lỗi trong
`workbench/docs/reviews/20261108T015603+0700/tests/full_suite_b0f38bd.txt`
(lượt serial tại `b0f38bd`: 33 failed, 2186 passed, 47 skipped = 2266 collected, exit 1).

Cây làm việc trong suốt quá trình: nhánh `research/cir-failure-workbench`, `HEAD = 6421188`.
**Cây không hề sạch.** Bốn agent khác đang sửa `workbench/backend/**`, `workbench/composition/**`,
`workbench/frontend/**`, `workbench/artifacts/**`, `workbench/envs/**` suốt thời gian này.
`git status --porcelain -uall` có 261 mục lúc chạy sweep; ảnh chụp ở
`tree_state_before_sweep.txt`. Không con số full-suite nào dưới đây mô tả một cây release sạch.

Không commit, không push, không stage. File untracked `workbench/config/workbench.env` của người
vận hành được tạm chuyển đi để đo rồi khôi phục **nguyên byte**
(`md5 b0d9fb7260e8032cc4c361d53c9e62d8`, đã kiểm sau từng lần probe).

---

## 1. Tổng kết

| kết cục | số lượng |
|---|---|
| lỗi do trạng thái máy này (file operator / env export / checkpoint đã stage) | **30** |
| lỗi do thiếu tiền đề build (frontend `dist`) | 0 |
| lỗi do defect thật trong code sản phẩm | 0 |
| lỗi là guard release-metadata, đúng theo hợp đồng của nó | **1** (`#04` ở mức file, tự nó pass) |
| tái hiện được khi chạy riêng tại `HEAD` | 32 / 33 |
| pass khi chạy riêng tại `HEAD` | 1 / 33 (`#04`) |

**Cả 33 hiện đều pass** khi chạy đúng tập từng lỗi (`all33_after_fix.txt`: `33 passed`).
Tất cả đều là nhiễm bẩn từ môi trường/trạng thái máy, hoặc là guard release-metadata — không có
lỗi nào là defect thật trong code sản phẩm, và không chỗ nào phải nới assertion, xoá test, hay
thêm skip.

## 2. Nguyên nhân gốc thật — một cơ chế, ba hình dạng

`operator_config.resolve_config()` đọc hai nguồn, theo thứ tự ưu tiên:

1. biến môi trường của process (`CIR_*`, `WORKBENCH_*`)
2. `workbench/config/workbench.env` — **untracked**, file của riêng máy này

Phần lớn suite không cô lập cả hai, nên trạng thái riêng của máy quyết định kết quả theo ba hình
dạng:

**Dạng A — checkpoint đã stage làm `Adapter.status().ready` lật.**
File operator đặt
`WORKBENCH_CHECKPOINT_ROOT=workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads`,
nơi máy này đã stage weight của ConeSep / PTHA-MTST / HABIT / INTENT / HINT / TME. Test adapter
assert `status.ready is False` — đó là đáp án của một bản clone mới; trên máy này checkpoint thật
sự có mặt nên `ready` là `True` và assertion fail. Test không sai: nó assert một thuộc tính độc
lập với máy, nhưng lại đang đọc đầu vào phụ thuộc máy.

**Dạng B — `WORKBENCH_ARTIFACTS_ROOT` tuyệt đối cướp output của subprocess.**
Cùng file đó đặt `WORKBENCH_ARTIFACTS_ROOT=/home/heheboiz/data/cir/workbench/artifacts`.
Mọi test spawn `pipeline.py` / `evaluate_models.py` / `verify_run.py` đều truyền
`CIR_REPO_ROOT=<sandbox>` để artifacts root *mặc định* rơi vào `<sandbox>/workbench/artifacts` —
đúng — nhưng tiến trình con đọc lại file operator, và giá trị tuyệt đối trong đó thắng. Lượt chạy
sau đó ghi ledger, report, completion proof và lock `pipeline_state.json` vào checkout của người
phát triển. Hệ quả quan sát được:

- `workbench/artifacts/logs/` trong sandbox không bao giờ được tạo → `FileNotFoundError`
  (`#33`, `#17`)
- completion proof mà test assert không nằm trong sandbox → `#21`
- `verify_run.py`, chạy bên trong shim Docker của `pc.sh`, tìm report của sandbox dưới checkout
  thật, không thấy, rồi từ chối → `pc.sh` exit 1 trong một sandbox đã stage lượt chạy đúng
  (`#05`–`#16`)
- hai case chạy song song đụng lock `pipeline_state.json` thật của nhau →
  `another workflow owns .../pipeline_state.json (lock held)` (`#29`, `#30`)
- `_recorded_asset_digest` của `test_release_blocker_source_integrity` đọc
  `<artifacts>/auxiliary/download_manifest.json` từ checkout thật, nên digest ghi vào sandbox
  tmp không bao giờ được dùng tới → `#31`, `#32`

**Dạng C — nhiễm bẩn một artifacts root dùng chung, bị gitignore.**
`workbench/artifacts/gpu_capabilities/` trên máy này có **504** record, **368** trong đó do test
ghi với `probe_source: override` ("a synthetic test probe, not a real probe").
`release_verify.sh`, được `test_release_blocker_gpu_records.py` gọi, quét đúng thư mục thật đó vì
`gpu_capability_directory()` đi theo `WORKBENCH_ARTIFACTS_ROOT`. 12 lỗi sau khi sửa ở §6 thuộc
dạng này; chúng **có từ trước và không do việc này tạo ra**.

### Probe từng biến một — biến nào lật test nào

Dạng lệnh (mỗi khóa tiêm riêng vào process pytest; bỏ conftest, file operator có/không như ghi
chú):

```bash
env "$KEY=$VALUE" workbench/envs/ctl312h/bin/python -m pytest <files> -q -p no:cacheprovider --tb=no
```

Khi đã có `conftest.py` của tôi thì không khóa nào lật được nữa — đó chính là mục đích của bản sửa:

```
BASELINE (không tiêm gì)                          48 passed
WORKBENCH_CIRCUS_RELEASE_ROOT=...                 48 passed
WORKBENCH_THIRD_PARTY_ROOT=...                    48 passed
WORKBENCH_ARTIFACTS_ROOT=...                      48 passed
WORKBENCH_CHECKPOINT_ROOT=...                     48 passed
CIR_DATA_ROOT=...                                 48 passed
WORKBENCH_RESULTS_ROOT=...                        48 passed
```
(`single_key_env_probe_after_fix.txt`)

Trên cây **chưa sửa**, file phải cung cấp *cả hai* nửa — checkpoint đã stage *và* interpreter
dùng được — vì `_status()` chỉ báo lý do checkpoint sau khi đã resolve được interpreter. Đó là lý
do tiêm riêng `WORKBENCH_CHECKPOINT_ROOT` không đủ:

```
A. nguyên file operator                                   3 failed, 27 passed
B. CHỈ WORKBENCH_CHECKPOINT_ROOT                        30 passed
C. CHỈ các khóa *_PYTHON (interpreter)                  30 passed
D. CHECKPOINT_ROOT + interpreter CONESEP/PTHA_MTST        2 failed, 28 passed
E. file operator rỗng                                   30 passed
```
(`operator_file_key_probe.txt`)

Probe tiêm từng biến môi trường trên cây chưa sửa (đã chuyển file operator đi):
`WORKBENCH_ARTIFACTS_ROOT`, `WORKBENCH_CHECKPOINT_ROOT`, `CIR_DATA_ROOT` tiêm riêng đều **không
lật** gì với các file adapter (30 passed mỗi lần, `single_key_env_probe_unfixed.txt`) — vì dạng A
cần thêm các khóa interpreter từ cùng file. Thứ mang bệnh là *file*, không phải một biến đơn lẻ.

Dạng source-integrity, chứng minh trực tiếp (`source_integrity_probe.txt`):

```
không conftest, không file operator                               18 passed
không conftest, CÓ file operator (ARTIFACTS_ROOT tuyệt đối rò ra)   2 failed
không conftest, không file, ARTIFACTS_ROOT -> thư mục scratch     2 failed
```

Tức `#31`/`#32` fail hoặc vì đọc manifest thật thay cho manifest trong sandbox, hoặc vì dưới root
được tiêm vào thì không có manifest nào. Ghim root là thoả cả hai.

## 3. Cả 33, chạy riêng từng cái

Lệnh tái hiện cho mỗi test:

```bash
workbench/envs/ctl312h/bin/python -m pytest "<node id>" -q -p no:cacheprovider --tb=long
```

Traceback đầy đủ: `01.txt` … `33.txt`. Thời lượng/exit từng test: `individual_repro_results.txt`.

| # | test id (file::test) | kết quả riêng @ HEAD | nhóm nguyên nhân | cách sửa | kiểm chứng |
|---|---|---|---|---|---|
| 01 | `test_circus_adapters_blip.py::test_adapter_import_and_unready_status[conesep]` | FAIL (`assert True is False`) | trạng thái máy — checkpoint đã stage (A) | scrub trong conftest | PASS |
| 02 | `test_circus_adapters_blip.py::test_adapter_import_and_unready_status[ptha_mtst]` | FAIL (như trên) | trạng thái máy (A) | scrub trong conftest | PASS |
| 03 | `test_circus_adapters_rest.py::test_status_is_honest_and_reports_every_mode` | FAIL (`ready=True` cho habit) | trạng thái máy (A) | scrub trong conftest | PASS |
| 04 | `test_handoff_commands.py::test_the_release_documents_quote_the_test_count_they_actually_ran` | **PASS** khi riêng; FAIL ở mức file | guard release-metadata (đúng hợp đồng) | không cần — xem §4 | PASS trong tập |
| 05 | `test_http_api_layer.py::test_a7_launcher_reuses_ledger_corroborated_run_...` | FAIL (`pc.sh` exit 1) | trạng thái máy — artifacts root tuyệt đối (B) | ghim root trong `setup_sandbox` | PASS |
| 06 | `test_pc_launcher.py::test_existing_completed_corroborated_run_...` | FAIL (`assert 1 == 0`) | trạng thái máy (B) | ghim root | PASS |
| 07 | `test_pc_launcher.py::test_matching_container_is_reused_positive_control` | FAIL | trạng thái máy (B) | như trên | PASS |
| 08 | `test_pc_launcher.py::test_verification_attempted_inside_container_not_on_host` | FAIL | trạng thái máy (B) | như trên | PASS |
| 09 | `test_pc_launcher.py::test_limn_bundle_selection_submits_bundle_id_with_native_protocol` | FAIL | trạng thái máy (B) | như trên | PASS |
| 10 | `test_pc_launcher.py::test_a_container_built_from_another_commit_is_not_reused` | FAIL | trạng thái máy (B) | như trên | PASS |
| 11 | `test_pc_launcher.py::test_the_summary_does_not_claim_the_index_holds_an_aggregate_only_run` | FAIL | trạng thái máy (B) | như trên | PASS |
| 12 | `test_pc_launcher.py::test_the_index_gate_refuses_a_per_query_run_that_is_not_verified` | FAIL | trạng thái máy (B) | như trên | PASS |
| 13 | `test_pc_launcher.py::test_the_index_gate_refuses_a_per_query_run_the_index_never_saw` | FAIL | trạng thái máy (B) | như trên | PASS |
| 14 | `test_pc_launcher.py::test_the_index_gate_refuses_an_aggregate_only_run_the_api_does_not_certify` | FAIL | trạng thái máy (B) | như trên | PASS |
| 15 | `test_pc_launcher.py::test_the_index_gate_refuses_a_run_the_api_cannot_certify` | FAIL | trạng thái máy (B) | như trên | PASS |
| 16 | `test_pc_launcher.py::test_a_completed_job_naming_its_run_uses_exactly_that_run` | FAIL | trạng thái máy (B) | như trên | PASS |
| 17 | `test_phase1_acceptance_scenarios.py::test_scenario_a_api_to_evaluator_success` | FAIL (`FileNotFoundError .../workspace/workbench`) | trạng thái máy (B) | ghim root trong phase1 + fixture master | PASS |
| 18 | `test_phase1_acceptance_scenarios.py::test_scenario_b_..._pointer_unchanged` | FAIL (thiếu `evidence_ledger.jsonl`) | trạng thái máy (B) | như trên | PASS |
| 19 | `test_phase1_acceptance_scenarios.py::test_scenario_d_cancellation_kills_the_evaluator_tree` | FAIL (`operator never started`) | trạng thái máy (B) | như trên | PASS |
| 20 | `test_phase1_acceptance_scenarios.py::test_scenario_g_grandchild_ignoring_sigterm_...` | FAIL (như trên) | trạng thái máy (B) | như trên | PASS |
| 21 | `test_release_blocker_master_e2e.py::test_e2e01_full_synthetic_evaluation_succeeds` | FAIL (ledger không verify) | trạng thái máy (B) | ghim root trong fixture master | PASS |
| 22 | `...::test_e2e02_resume_skips_verified_completed_run` | FAIL (`assert 1 == 0`) | trạng thái máy (B) | như trên | PASS |
| 23 | `...::test_e2e03_checkpoint_modification_invalidates_evaluation` | FAIL | trạng thái máy (B) | như trên | PASS |
| 24 | `...::test_e2e04_dataset_modification_invalidates_evaluation` | FAIL | trạng thái máy (B) | như trên | PASS |
| 25 | `...::test_e2e04b_dataset_modification_invalidates_only_metadata_size_changes` | FAIL | trạng thái máy (B) | như trên | PASS |
| 26 | `...::test_e2e05_exit_zero_without_new_output_fails_evaluation` | FAIL | trạng thái máy (B) | như trên | PASS |
| 27 | `...::test_e2e06_failed_rerun_preserves_previous_successful_experiment` | FAIL | trạng thái máy (B) | như trên | PASS |
| 28 | `...::test_e2e07_recovery_after_failure_succeeds_and_advances` | FAIL | trạng thái máy (B) | như trên | PASS |
| 29 | `...::test_e2e11_cli_refuses_weak_input_fingerprint` | FAIL | trạng thái máy (B) | như trên | PASS |
| 30 | `...::test_e2e14_verified_gpu_capability_permits_the_authorized_synthetic_path` | FAIL | trạng thái máy (B) | như trên | PASS |
| 31 | `test_release_blocker_source_integrity.py::test_src07_declared_auxiliary_asset_accepted_only_with_provenance` | FAIL (`True is False`) | trạng thái máy (B, root của manifest) | scrub trong conftest | PASS |
| 32 | `test_release_blocker_source_integrity.py::test_src08_modified_auxiliary_asset_rejected` | FAIL (`True is False`) | trạng thái máy (B, root của manifest) | scrub trong conftest | PASS |
| 33 | `test_true_master_e2e.py::test_true_master_pipeline_subprocess_e2e_complete_flow` | FAIL (thiếu `logs`) | trạng thái máy (B) | ghim root trong env của test | PASS |

Ghi chú về A/B trước đó: kết luận cũ rằng file operator giải thích cả 33 là **đúng hướng nhưng
nói quá** — con số 221/0 của nó chỉ phủ 4–8 file. Ở đây tái hiện được: nó cũng giải thích
`test_pc_launcher.py`, `test_phase1_acceptance_scenarios.py`, `test_http_api_layer.py` và
`test_true_master_e2e.py`, những file A/B cũ không phủ, và giải thích chúng qua
`WORKBENCH_ARTIFACTS_ROOT` chứ không phải chỉ qua checkpoint root.

Ghi chú về `test_pc_launcher.py`: nó dùng **shim PATH cho Docker**, không phải daemon thật, nên 11
lỗi của nó chưa bao giờ là "không có daemon". Chúng là do artifacts root tuyệt đối đánh bại
`verify_run.py` bên trong shim. Xác nhận bằng bản sửa: 101 passed mà không cần daemon nào.

## 4. `#04` — cái duy nhất không phải trạng thái máy

`test_the_release_documents_quote_the_test_count_they_actually_ran` **pass khi chạy riêng** tại
`HEAD` (nó đọc số lượng của *revision được nêu tên* trong một worktree tạm). Nó chỉ fail khi
`test_handoff_commands.py` chạy ở mức file, vì test anh em
`test_no_release_document_points_at_a_superseded_source_sha` fail ở đó:

```
TESTED_SOURCE_SHA b0f38bd... does not contain the production code currently on disk
```

Test anh em đó là guard release-metadata và nó **đúng**: verdict ghi `b0f38bd` là bản đã test,
trong khi 22 file production tracked dưới `workbench/backend/**` và `workbench/composition/**` đang
có sửa đổi chưa commit từ các agent chạy song song. Guard tồn tại chính là để nói điều đó. Nó
không thể pass khi việc của agent khác còn chưa commit, và làm nó pass sẽ phải hoặc commit việc của
họ (bị cấm) hoặc nới guard (bị cấm). Kỳ vọng của nó không lỗi thời; chỉ là cây này không phải cây
release. Báo cáo, không "sửa".

## 5. Đã thay đổi gì

| file | thay đổi |
|---|---|
| `workbench/tests/conftest.py` (**mới**) | autouse `isolate_operator_state`: xoá mọi `CIR_*`/`WORKBENCH_*` khỏi env process, trỏ `operator_config._CONFIG_PATH` vào đường dẫn không bao giờ tồn tại, và clear `model_settings.file_settings` + `runtime.forget_environment()` ở setup |
| `workbench/tests/test_pc_launcher.py` | `setup_sandbox` đặt tên `CIR_DATA_ROOT`, `FASHIONIQ_ROOT`, `WORKBENCH_ARTIFACTS_ROOT/CHECKPOINT/RESULTS/THIRD_PARTY/CIRCUS_RELEASE_ROOT` nằm trong sandbox |
| `workbench/tests/test_release_blocker_master_e2e.py` | fixture `master` ghim `WORKBENCH_ARTIFACTS_ROOT` và `WORKBENCH_CIRCUS_RELEASE_ROOT` |
| `workbench/tests/test_phase1_acceptance_scenarios.py` | cả hai danh sách khóa env thêm `WORKBENCH_ARTIFACTS_ROOT` + `WORKBENCH_CIRCUS_RELEASE_ROOT` |
| `workbench/tests/test_true_master_e2e.py` | fixture ghim `WORKBENCH_ARTIFACTS_ROOT` + `WORKBENCH_CIRCUS_RELEASE_ROOT` |

Conftest chỉ clear ở **setup**. Clear ở teardown chạy *trước* khi `monkeypatch` khôi phục bản vá
`file_settings` của chính test, nên nó sẽ gọi `cache_clear()` trên lambda thay thế của
`test_a36_assigned_identity`. Lỗi hồi quy đó đã quan sát được (6 errors) và đã bỏ.

Không test nào bị xoá, bị skip, hay bị làm yếu. Không hằng số assertion nào bị nới. Không counter
hay tài liệu nào bị sửa để giấu một test fail.

### Kiểm hồi quy trên các test nhạy cảm cấu hình

```
workbench/tests/test_operator_config.py test_a31_effective_interpreter.py test_a33_interpreter_alias.py
test_a34_effective_target.py test_a36_assigned_identity.py test_circus_worker_env.py
test_artifact_root_configuration.py test_manage_environment_a30.py test_r20_shared_seams.py
test_backend_config_roots.py test_index_workspace_configuration.py
=> 51 passed, 33 skipped   (config_sensitive_regression.txt)
```

## 6. Trước / sau

### Tập từng lỗi — cả 33

| | kết quả | lệnh |
|---|---|---|
| trước | 33 failed (trên 33) | xem §3 |
| sau | **33 passed in 130.37s, exit 0** | `workbench/envs/ctl312h/bin/python -m pytest $(cat ids.txt) -q -p no:cacheprovider --tb=line` |

`all33_after_fix.txt`.

### Các file bị ảnh hưởng

| file | trước | sau |
|---|---|---|
| `test_circus_adapters_blip.py` | 2 failed, 16 passed | 18 passed |
| `test_circus_adapters_rest.py` | 1 failed, 11 passed | 12 passed |
| `test_handoff_commands.py` | 1 failed, 7 passed | 1 failed, 7 passed (guard, §4) |
| `test_http_api_layer.py` | 1 failed, 44 passed | 45 passed |
| `test_pc_launcher.py` | 11 failed, 90 passed | 101 passed |
| `test_phase1_acceptance_scenarios.py` | 5 failed, 3 passed | 8 passed |
| `test_release_blocker_master_e2e.py` | 10 failed, 6 passed | 16 passed |
| `test_release_blocker_source_integrity.py` | 2 failed, 16 passed | 18 passed |
| `test_true_master_e2e.py` | 1 failed, 2 passed | 3 passed |

(`per_file_before.txt`, `per_file_after.txt`)

### Full suite

```
workbench/envs/ctl312h/bin/python -m pytest workbench/tests -q -p no:cacheprovider -rfE --tb=no -n 10
```

| | passed | failed | skipped | collected | exit | thời lượng |
|---|---|---|---|---|---|---|
| `b0f38bd` serial (tham chiếu) | 2186 | **33** | 47 | 2266 | 1 | 2211.38 s |
| `6421188` + việc này, `-n 10` | 2321 | **12** | 47 | 2380 | 1 | 588.40 s |

Log: `full_suite_6421188_after_isolation.txt`. SHA `6421188f4f93574b7fe093a804dd235d39e630e8`.

Số collected tăng 2266 → 2380 vì các agent khác thêm file test của họ (`test_frontend_*`,
`test_composition_topk_*`, `test_capability_*`, `test_circus_*_*`). **So sánh này không phải
cây-sạch với cây-sạch:** 2321/12 mô tả worktree đang trộn việc, không phải một revision phát hành
được. Con số quy được cho việc này là 33 → 0 trong tập từng lỗi.

## 7. 12 lỗi còn lại, và vì sao không cái nào là của tôi

`new_failures_attribution.txt` — cả 12 fail y hệt khi bỏ `conftest.py`, và đều fail trên cây
`b0f38bd` vì lý do riêng của chúng, tức **không cái nào do việc cô lập này tạo ra**. Không cái nào
nằm trong 33 ban đầu.

| test | traceback (dòng đầu) | vì sao chưa sửa ở đây |
|---|---|---|
| `test_circus_tme_lifecycle.py::test_encoder_and_limn_failed_child_latch_and_preserve_log[encoder]` | `RuntimeError: execution environment not verified [UNCONFIGURED/UNCONFIGURED]: WORKBENCH_ENCODER_PYTHON=None` | cần interpreter encoder thật; là **thiếu tiền đề**, không phải cô lập. Thuộc agent composition (`require_verified_interpreter` là gate mới của họ — xem dưới). |
| `...[limn]` | như trên, `WORKBENCH_LIMN_PYTHON=None` | như trên |
| `test_release_blocker_gpu_records.py::test_defect_b_g_with_no_target_gpu_verdict_is_pending_not_pass` | `RELEASE_BLOCKED` với 368 `probe_source=override ... synthetic test probe` | nhiễm bẩn artifacts dùng chung (§2 dạng C) |
| `...::test_offline_verdict_names_deployment_reason` | như trên | như trên |
| `...::test_release_verify_documented_invocations_produce_parseable_report[<lambda>0..4]` (5 test) | `Invocation failed` | như trên |
| `test_handoff_commands.py::test_no_release_document_points_at_a_superseded_source_sha` | `TESTED_SOURCE_SHA b0f38bd... does not contain the production code currently on disk` | guard đúng; 22 file production chưa commit bởi agent khác (§4) |

Hai mục cần nói rõ:

- **`test_circus_tme_lifecycle`** fail vì agent composition chạy song song đã thêm
  `CompositionAdapter.require_verified_interpreter()` mà helper `_encoder_adapter` / `_limn_adapter`
  của test không bật chốt đó. Bản sửa ở file anh em `test_circus_adapters_blip.py` của họ (thêm
  `adapter._interpreter_verified = True`, mtime 11:40, **không phải sửa của tôi**) cho thấy cách sửa
  dự kiến; áp đúng ba dòng đó vào hai helper lifecycle là việc trong file của họ.
- **`test_release_blocker_gpu_records`** chính là defect dạng C đã đo ở trên: máy này có 504 record
  trong `workbench/artifacts/gpu_capabilities/`, 368 trong đó là synthetic-probe, và
  `release_verify.sh` quét đúng thư mục thật đó. Cách sửa đúng là cho hai lời gọi `release_verify.sh`
  ghim `WORKBENCH_ARTIFACTS_ROOT` vào một sandbox `tmp_path`, giống case
  `test_offline_...` ngay bên cạnh đã làm ở dòng 795 — nhưng tôi **không** áp dụng, vì probe không
  thể kết luận dứt điểm trong khi các agent khác đang liên tục ghi lại thư mục dùng chung bị
  gitignore đó (một lần `mv` đã đụng giữa chừng; thư mục đã được khôi phục về 504 record phẳng và
  đã kiểm). Sửa một assertion đang sống mà chưa kiểm chứng thì tệ hơn báo cáo chính xác.
  **Đây là phần việc bàn giao duy nhất.**

Cảnh báo cho người làm tiếp: chuyển `workbench/artifacts/gpu_capabilities/` đi chỗ khác trong lúc
bốn agent đang chạy test sẽ gây race. Hãy cô lập phía *đọc* (ghim `WORKBENCH_ARTIFACTS_ROOT` cho
hai subprocess), tuyệt đối không chuyển thư mục dùng chung đó.

## 8. File đã thay đổi

- `workbench/tests/conftest.py` (mới)
- `workbench/tests/test_pc_launcher.py`
- `workbench/tests/test_release_blocker_master_e2e.py`
- `workbench/tests/test_phase1_acceptance_scenarios.py`
- `workbench/tests/test_true_master_e2e.py`
- `workbench/docs/reviews/20261108T120100+0700-tests/**` (báo cáo này và log thô)

Không do tôi sửa nhưng bị các agent chạy song song thay đổi trong cây trong lúc làm, và thấy được
trong lượt sweep: `workbench/tests/test_circus_adapters_blip.py`, `test_frontend_staleness.py`,
cùng 22 file production dưới `workbench/backend/**` / `workbench/composition/**` và 8 file frontend
mới. `workbench/config/workbench.env` chỉ được chuyển đi để probe, không hề bị sửa, và đã kiểm là
nguyên byte sau đó.
