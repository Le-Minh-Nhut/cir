# NGHIỆM THU CUỐI — workbench CIR (2026-11-13)

Tài liệu bàn giao. Mọi con số dưới đây chép từ một record có đường dẫn ngay trong dòng; cái nào là
suy luận thì ghi `[SUY LUẬN]`. Không chạy GPU, không restart service, không sửa file production/test.

## Trạng thái

**COMPLETE.**

Lý do: cả năm dòng nghiệm thu đều có lệnh chứng minh và kết quả đo trong record; `acceptance.sh` trả
`ACCEPTANCE PASSED (skipped: G-status-unverified)` rc 0; sweep tại pin đúng 3330/3330 thu
(3275/8/47), 0 lỗi mới so sweep trước; guard release 8 passed sau khi nâng pin; 8 id sweep đỏ và 19
failure gate đều đã phân lớp, **0** thuộc lớp `absent_evidence` hay `derived_invariant`. Không có
check nào "coi như xong" mà thiếu số đo: hai dòng từng `UNVERIFIED` (L09, L22) đã được đo lại trong
lượt này (xem §Năm dòng).

## Mốc bàn giao

| mục | giá trị | nguồn |
|---|---|---|
| HEAD | **`git rev-parse HEAD` là thẩm quyền** — không chốt SHA ở đây, vì tài liệu này được chính commit chứa nó đưa vào cây. HEAD **lúc đo** = `a53044f7cf5607a08975e135f7698444986e855e`; hai commit chỉ-tài-liệu sau đó (`6ee1618`, `d2edc02`) không chạm cây production | `git rev-parse HEAD` |
| `origin/research/cir-failure-workbench` | bằng HEAD tại mọi thời điểm đo; `git rev-list --left-right --count HEAD...origin/research/cir-failure-workbench` = `0	0` **lúc đo** (và lại `0 0` sau khi push hai commit tài liệu) | `git rev-list --left-right --count` |
| nhánh | `research/cir-failure-workbench` | `git branch --show-current` |
| `TESTED_SOURCE_SHA` | `90f7b2c1e1f6b8474570961ae5a2e08ecf2ee36c` (có trong **cả hai** release doc) | `FINAL_PC_RELEASE_VERDICT.md:22`, `PC_PILOT_HANDOFF.md` |
| cây production tại pin == HEAD | `git diff --quiet 90f7b2c HEAD -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'` → exit **0**. Đây là **bất biến**, không phải một SHA đóng băng: nó đúng ở `a53044f`, ở `6ee1618`, ở `d2edc02`, và sẽ đúng ở mọi commit chỉ-sửa-`workbench/docs` tiếp theo | đo lại lượt này, và sau mỗi commit tài liệu |
| bundle đã deploy | `dist/assets/index-8klMGIWd.js` **399 398 B**, sha256 `43b2618c56bf2a355dda893c774d8160ab64fcbb1df88623d84159baea284d87` | `sha256sum`, và `curl :5173/assets/…` cho **cùng** digest |
| shell | `dist/index.html` **271 B**, sha256 `696c23fcaec29e809f997063e3a758afd3665859d397b9741e9c76fcc6ea0d04`; served == built | đo lại lượt này |
| `dist` | gitignored (`.gitignore:95`, 0 path tracked) → **deploy mà không commit**, có chủ ý | `SWEEP_90f7b2c.md` §1 |
| backend | PID **2816638**, `ActiveEnterTimestamp=Fri 2026-11-13 10:10:07 +07`, `NRestarts=0`, `KillMode=mixed`, `TimeoutStopUSec=30s` | `systemctl --user show` (đo lại) |
| frontend | PID **2816640**, cùng timestamp, `NRestarts=0`, `KillMode=control-group`, `TimeoutStopUSec=5s` | `systemctl --user show` (đo lại) |
| runtime roots | `CIR_DATA_ROOT=/home/heheboiz/data/cir/data` · `FASHIONIQ_ROOT=…/data/FashionIQ` · `WORKBENCH_ARTIFACTS_ROOT=…/workbench/artifacts` · `WORKBENCH_CHECKPOINT_ROOT=…/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads` · `WORKBENCH_RESULTS_ROOT=…/artifacts/results` · `WORKBENCH_THIRD_PARTY_ROOT=…/workbench/third_party` · `WORKBENCH_CIRCUS_RELEASE_ROOT=…/artifacts/circus_release` | `workbench/config/workbench.env:1-8` |
| artifact roots | sweep/test log `/tmp/sweep_90f7b2c.txt`; evidence campaign `workbench/docs/reviews/20261113-final-acceptance/raw/**`; artifact native/CIRCUS `workbench/artifacts/**` | record từng lane |
| interpreters | control `workbench/envs/ctl312h/bin/python` = CPython **3.12.11** (`uv` cpython-3.12.11-linux-x86_64-gnu); model envs: `encoder310/limn310-adapted`→cpython-3.10.20, `ilearn39/clvc39-la/tgcir39-la`→3.9.25, `tme-adapted`→3.10, `artifacts/environments/cir-005ef4e1b02999a5`→3.10 | `-VV`; `raw/00-state/06-interpreters.txt`; `executability.json` |
| đĩa | `/` **545G dùng / 53G trống (92%)** — trên sàn 45 GiB | `df -h /` (đo lại lượt này; sweep ghi 84 GiB free lúc chạy) |
| GPU | RTX 5070 Ti, 748 MiB / 16 303 MiB, **0 compute app** — không job nào đang chạy | `nvidia-smi` (đo lại) |

## Năm dòng nghiệm thu

| # | dòng | lệnh chứng minh | kết quả đo |
|---|---|---|---|
| 1 | **PAIR deferred** | `curl -s :8000/api/models \| jq length` · `curl -s :8000/api/models/publication` · `curl -s -o /dev/null -w '%{http_code}' -XPOST :8000/api/jobs -d '{"model_id":"pair","checkpoint_id":"pair_b1","protocol_id":"fashioniq_val_split"}'` | `16`, không có `pair`; publication `{"registry_count":17,"published_count":16,"deferred_count":1}`; POST → **409** `model_deferred_from_web`; model lạ → **404** `model_not_found`; `/api/native-measurements` 51 dòng, **0** `pair`; by-id `61ccfc4d63fb69eb` → 200 `EXPLORATORY_PUBLIC_MAPPING` + `web_publication.status=DEFERRED`; 3 deep link `?view=native&model=pair` đều hiện notice, 0 console/page/request error |
| 2 | **gate trustworthy** | `curl -s :8000/api/runs/parity-gate` (và `raw/parity-coverage/gate.json` từ `parity_gate.evaluate_all()` in-process) | `passed_models: []`; `standard` là **chuỗi** `parity-relative/v1` (+ `standards` phân nhóm); 7 model được phán; 19 failure: encoder 9 / habit 2 / hint 2 / limn 3 / tgcir 3 — **giống hệt** `evaluate_all()`; 19/19 lớp `measured_disagreement`, **0** `absent_evidence`, **0** `derived_invariant`; `conesep`/`intent` = `PARITY_EVIDENCE_MISSING` (0 failure, không phải pass) |
| 3 | **CLVC usable** | `curl -s ':8000/api/composition/diagnostics/clvc-dual-scope?qid=7:3790'` · `curl -s ':8000/api/composition/diagnostics/limn-dual-scope?qid=7:3038'` · probe browser 12 ô | cả hai route **200** `DIAGNOSTIC_COMPLETE`, `complete:true`, `defects:[]`; 12 ô browser vs `INVENTORY.md` = **0 mismatch**, mỗi ô 100 tile, 4/4 ô ablation có banner self-match, 12/12 target "ngoài cửa sổ Top-100" kèm rank tuyệt đối |
| 4 | **parity/benchmark scope accounted** | `pairs_on_disk.json` · `executability.json` · `unprobed_contracts.json` · `gate.json` | 5 pair (encoder, habit, hint, limn, tgcir); 10 EXECUTABLE / 5 `ENV_VAR_UNSET` / 1 `ADAPTER_NOT_READY` / 1 `NO_ADAPTER`; mỗi model không đo được có **record blocker tái chạy được** (lệnh + lỗi verbatim), 4 model EXECUTABLE không pair có contract trích từ code pin |
| 5 | **release reproducible** | `pytest workbench/tests/test_handoff_commands.py` · `bash workbench/docker/acceptance.sh` · sweep serial | guard **8 passed** (15.22 s, đo lại lượt này); `ACCEPTANCE PASSED (skipped: G-status-unverified)` rc 0; sweep `90f7b2c` **3275 passed / 8 failed / 47 skipped = 3330**, 3808.40 s, `^ERROR` 0, `Errno 28` 0, **0 new / 0 gone / 8 identical** |

Ghi chú trung thực cho dòng 5: lần chạy acceptance **đầu tiên sau restart** báo sai
`SKIP A/C/G/I live checks (no server)` vì `/api/models` còn nguội (phân giải
checkpoint-availability ~1 phút). Đó là false negative của `live()` (một probe duy nhất), không phải
stack hỏng; chạy lại sau khi warm thì mọi nhóm live PASS — `raw/live-verify/REPORT.md` §9. Lượt này
tôi chạy lại `acceptance.sh` trên stack đã deploy: **12 PASS / 1 SKIP / 0 FAIL**, kết thúc
`ACCEPTANCE PASSED (skipped: G-status-unverified)`, **rc 0**, 373.47 s (`/tmp/acc_final.txt`). Nhóm G
SKIP vì hàng đợi rỗng — "hàng đợi trống không phải bằng chứng enum status đúng", nên nó được báo SKIP
chứ không PASS. Nhóm E đo được `12 GPU-requiring models; 8 carry a verified GPU_RUNTIME_READY probe`,
nhóm H `the served bundle carries both display-only labels (CAPTION THAY THẾ, LỆCH PAPER)`, nhóm I
`/api/runs/gate` trả 200 (không 500).

## Số đếm chính xác

Không con số nào dưới đây được dùng thay cho con số khác.

| đại lượng | giá trị | đơn vị / cách đếm | nguồn |
|---|---|---|---|
| model trong registry | **17** | `yaml['models']` | `executability.json:registry_count` |
| model công bố trên web | **16** | registry − deferred (`pair`) | `/api/models`, `/api/models/publication` |
| model bị defer khỏi web | **1** | `pair` | `registry.py:web_deferred_models` |
| có **native aggregate** trong `raw/native/` | **9** | file JSON parse được: clip4cir_fullft, conesep, dcnet, encoder, habit, hint, intent, ptha_mtst(.result), tme | `raw/native/*.json` |
| có **pair đo được** (native+adapter, đủ 3 category) | **5** | `pairs_on_disk.json:complete_pair_count` | encoder, habit, hint, limn, tgcir |
| **PASS** gate | **0** | `gate.json:passed_models` = `[]` | `gate.json` |
| được gate **phán** | **7** | 5 có verdict + conesep/intent ở `EVIDENCE_MISSING` | `gate.json:discovered_models` |
| failure của gate | **19** | 9+2+2+3+3 | `gate.json` |
| … lớp `measured_disagreement` | **19** | tất cả | `gate.json` |
| … lớp `absent_evidence` | **0** | habit's 3 rank-check đã đóng ở vòng F1 | `COVERAGE.md` §F1 |
| … lớp `derived_invariant` | **0** | — | `run_gate.py` classifier |
| `metrics_parity` **missing** | **3 model** | `encoder`, `habit`, `tgcir` — `evaluate_all()`: encoder có 3 check nhưng đều `ok: false` với `counts: null`; habit/tgcir **0** check. `conesep`/`intent` là `PARITY_EVIDENCE_MISSING`, **không** có dimension → **unjudged**, không phải missing | `evaluate_all()` đo lại lượt này; `FINAL_TABLE.md` §Counts |
| `metrics_parity` **đo được và PASS** | **2 model** | `hint`, `limn` — 12 check mỗi model (4 × 3 category), tất cả `ok: true`; hit count hai phía bằng nhau ở mọi cutoff, gate **tự suy** rồi so flag | `evaluate_all()`; `g1_g2.{hint,limn}.*.native.json` |
| **EXECUTABLE** | **10** | 4 điều kiện cùng đúng | `executability.json:verdicts` |
| **BLOCKED** | **6** | 5 `ENV_VAR_UNSET` + 1 `ADAPTER_NOT_READY` (csmcir) | `executability.json` |
| … cộng `pair` (deferred, cũng chặn) | **7** | đếm riêng, không gộp vào 6 | `FINAL_TABLE.md` §Counts |
| `NO_ADAPTER` | **1** | `sprc` (không có `model_sprc.py`) | `executability.json` |
| **SCOPE_DECISION** | **1** | `ptha_mtst` (`SELECTED_TRANSFER_ONLY`) | `unprobed_contracts.json` |
| **specified-not-built** | **2** | `clvc_net`, `tme` (code có, hook thiếu) | `unprobed_contracts.json` |
| **G1_ONLY** | **2** | `conesep`, `intent` | `pairs_on_disk.json` |
| CIRCUS run **thật** | **34** | 36 `full.json` − 2 synthetic (`fixture_encoder`, `fixture_habit`); `released_audit` là run thật, được đếm | `workbench/artifacts/composition/**/full.json` |
| sweep passed / failed / skipped / collected | **3275 / 8 / 47 / 3330** | một lượt serial | `SWEEP_90f7b2c.md` §3 |
| id sweep đỏ còn lại | **8** | danh sách §Danh sách lỗi còn đỏ | `SWEEP_90f7b2c.md` §4 |
| id baseline đã đóng bởi chiến dịch | **12** | 4 blip + tme_precision + J10 ×7 | `raw/ops-gaps/REPAIRS.md` |
| test pytest đã collect | **3330** | `--collect-only`, không đổi từ `e833eda` | `SWEEP_90f7b2c.md` §3 |

Bốn đại lượng từng bị dùng lẫn trong `MODEL_MATRIX.md` (25 key / 17 model / 23 dòng / 27 trùng —
`LEDGER.md` L29): **không** đại lượng nào ở bảng trên là "đã đánh giá". "Native-eval 9" là *có
aggregate*, không phải *đã tái lập*.

## Danh sách lỗi còn đỏ

### 8 id sweep (đo tại `90f7b2c`, y nguyên tập của `e833eda`)

| id | lớp | lý do |
|---|---|---|
| `test_circus_cli.py::test_cli_3d_full_gallery_resume_and_gallery_invalidation` | order dependency | `_module_source_hash` hash module **đã import**, nên cùng-process `--resume` sau khi sửa fixture thấy byte khác nhau tuỳ thứ tự chạy (`TASK2_TRIAGE.md` §2) |
| `test_circus_tme_lifecycle.py::…[encoder]` | ENVIRONMENT | `conftest.py::isolate_operator_state` xoá mọi `WORKBENCH_*`; adapter **từ chối** interpreter chưa xác thực — đúng thiết kế |
| `test_circus_tme_lifecycle.py::…[limn]` | ENVIRONMENT | như trên |
| `test_doctor.py::test_dcnet_runtime_asset_blocks_after_preparation_is_present` | EXTERNAL | assertion cũ vs asset **tốt hơn** trên host (`assert 'OK' == 'BLOCKED'`) |
| `test_evaluate_models.py::test_dcnet_audited_command_reports_non_command_blockers` | EXTERNAL | cùng asset |
| `test_f05_f08_f09_verification.py::test_f08_dcnet_config_data_dir_resolution` | ENVIRONMENT | cần symlink FashionIQ do host stage |
| `test_handoff_commands.py::test_no_release_document_points_at_a_superseded_source_sha` | by construction | **đỏ trong chính sweep** vì sweep chạy trước lệnh nâng pin; xanh sau (`8 passed`) |
| `test_standalone_operator_commands.py::test_prepare_model_executes_only_declared_recipe_steps` | ENVIRONMENT | link `dataset` trỏ cây thật |

### 19 failure gate (0 pass)

| model | failures | check | measured vs declared |
|---|---|---|---|
| encoder | **9** | 3× `population.features_vs_native_val_control.max_rel`, 3× `population.gallery_vs_native.max_rel`, 3× `batch_1_vs_n.max_rel` | 0.75 / 10.46 / 0.00318 … vs `1e-4`/`1e-3`; `max_abs` worst **1.65e-06**, `min_cosine ≥ 0.99999999998` |
| tgcir | **3** | 3× `batch_1_vs_n.max_rel` | 0.2403 / 0.0705 / 0.0295 vs `1e-3`; `max_abs` ≤ **2.90e-06** vs atol `1e-4` |
| limn | **3** | 3× `batch_1_vs_n.max_rel` | 0.0114 / 0.00270 / 0.00751 vs `1e-3` |
| habit | **2** | 2× `batch_1_vs_n.max_rel` | 0.005096 / 0.006168 vs `1e-3`; `max_abs` 1.94e-07 / 1.63e-07 |
| hint | **2** | 2× `batch_1_vs_n.max_rel` | 0.005050 / 0.04214 vs `1e-3`; `max_abs` 1.53e-07 |

**Lớp:** 19/19 `measured_disagreement` — tất cả là `max_rel` trên mẫu số tiến về 0, có `max_abs`
trong atol và `min_cosine ≥ 0.9999999`. **Không** failure nào thuộc lớp `absent_evidence` (kiểu "không
có bằng chứng") hay `derived_invariant` (kiểu gate tự suy ra rồi tự mâu thuẫn). Không tolerance nào bị
dời sau khi đọc số (`LEDGER.md` L10).

## Lỗi đã sửa trong chiến dịch

Mỗi dòng kèm bằng chứng **trước khi sửa**.

| # | sửa | bằng chứng trước fix |
|---|---|---|
| 1 | **v2 preview-PASS**: preview `chunked` không bao giờ được `ok`; offender đã ghi mà không emit = `MALFORMED` | `repro_a.py` A2b: offender rows xoá + `mismatch_count`/`offenders` zeroed, `worst_normalized 166.67` để nguyên → **`PARITY_PASS`, `passed: True`** (`repro_a.observed.txt`) |
| 2 | **hit-count flag**: gate **tự suy** `hits` rồi so với flag; flag không bằng suy luận = `MALFORMED`; flag không kèm count = `INCOMPLETE` | `repro_b2.py`: `flag=true`, detail native 375 / adapter 376, `equality.r1=false` → **`PARITY_PASS`**; B2b flag trần không count → `PARITY_PASS`; `grep native_metrics_parity_detail workbench/backend/` = **0 hit** |
| 3 | **11 dạng record v2 lọt PASS** (chunk gap / thứ tự chunk / chunk thay thế / digest cũ / nhãn `coverage` / tổng bị khai nhỏ / head thiếu / chunk 1 trong 4146 / overlap / đứt tail / `shape` sai) | `repro_a3/a3b/a4.observed.txt`: **P1..P6, P2/P3/P10/P11, A4** đều `PARITY_PASS`; vd `P10 chunked: 1 of 4146 → PARITY_PASS`, `P5 sha256 = 'a'*64 → PARITY_PASS`, `A4 chunked→full relabel 512/512 → PARITY_PASS` |
| 4 | **gallery digest hai công thức**: `_plan` và blocked-report tái dùng `_gallery_fingerprint`; payload **không** đổi để artifact cũ vẫn verify | plan `0bfd43b9cc01…` vs runner `1db68c59d96f…`, `equal: False`; fix → cả ba nhánh `1db68c59…`, đổi thứ tự gallery thì digest đổi (negative control) |
| 5 | **scope-resolution 6003**: `still_unscoped` suy từ **reason** trả về, thêm `global_scoped`/`category_agnostic`, giữ nghĩa hẹp của `adapter_routed` | `7b1fa850…`: `still_unscoped: 6003`, `adapter_routed: 0`, `unscored_count` cả 3 condition = 0; `_scope_for_query((None,))` → `(None, None)` |
| 6 | **conditions `None` vs `()`**: ba trạng thái phân biệt + `conditions_declaration` phục vụ trên cả 2 route | `run_conditions({"conditions": []})` → `None` và `displayed` = bộ ba protocol; `absent is empty -> True` (`task1-before-after.txt`) |
| 7 | **hai-module hash guard**: `backend/composition.py` so **hai** hash của **hai** module | `da0a2162…` adapter hash `3b8e1bb5…` ≠ đĩa `7f150b6a…` nhưng served vẫn `scoring_impl: live` |
| 8 | **PAIR publication**: `web_publication: DEFERRED` + `web_deferred_models()` + filter ở mỗi surface + `/api/models/publication` (17/16/1) + notice deep link | `/api/models` **17** gồm `pair`; `/api/native-measurements` **56** dòng gồm 5 dòng `pair`; POST job pair → **404 `model_not_found`** (lời nói dối: entry có trong registry) |
| 9 | **CLVC diagnostic reader hardwired LIMN**: descriptor hóa `DiagnosticModel`, thêm router CLVC và **mount** router | `[r.path for r in app.routes if 'diagnostics' in r.path]` chỉ có 2 path LIMN; `curl :8000/…/clvc-dual-scope` → **404**; `grep -c include_router main.py` = **0** |
| 10 | **checkpoint-digest âm thầm bỏ qua**: digest thiếu là **defect**, không phải skip | CLVC `checks[scope]["sha256"] = ""` → so sánh bị bỏ, gate vẫn xanh; nay bóc digest → `DIAGNOSTIC_EVIDENCE_DEFECT`, `complete:false`, **12 defect** `checkpoint_summary_digest_missing` |
| 11 | **blip fixture**: 4 id đỏ do fixture dựng adapter bằng `__new__` | `AttributeError: 'Adapter' object has no attribute 'batch_size'`; 4 failed / 14 passed → **18 passed**; **không** thêm default class-level (batch size là input của measurement identity) |
| 12 | **tme stub**: fake `torch` thiếu `set_num_threads` | `CIRCUS_WORKER:{"ok": false, … "module 'torch' has no attribute 'set_num_threads'"}` → `FileNotFoundError output.npy`; 1 failed → **1 passed** |
| 13 | **J10 group**: grep copy trang trí đã bị xoá có chủ ý → thay bằng "bundle được shell nêu có tồn tại và mang cả 2 nhãn gate" | J10 grep `phán quyết audit chuẩn`, chuỗi `4e61f2f` đã xoá; 7 id `release_blocker_gpu_records` đỏ; mutation-proof M1/M2/M3 đều FAIL đúng, restored PASS |
| 14 | **acceptance.sh group H**: mirror J10 (bundle + 2 nhãn), không grep copy chết | H trên `43bfc9e` không thể PASS; test mới trong `test_final_hardening_regressions.py` (3 thêm, 48 trong file, xanh) |
| 15 | **acceptance.sh group C**: kỳ vọng = registry − deferral, và **deferred mà vẫn served = FAIL** | C đòi served **bằng** cả 17 entry trong khi `393de24` cố ý giữ 16 → bất khả thi |
| 16 | **frontend unit KillMode**: `control-group` + `TimeoutStopSec=5`; backend giữ `mixed`/30 s có chủ ý + negative control | `mixed`: npm 22 ms, **node 32 506 ms**, port free 66 266 ms, restart cùng port thất bại; `control-group`: 22 / **24** / **51 ms**, restart ngay. Test `test_frontend_stop_contract.py` (4) mutation-proof: revert → 3 failed |
| 17 | **`model` bị rơi khỏi location model**: `model` là field của `Location`, serialize cuối; URL nêu view mà không nêu area thì lấy area sở hữu view | `?area=evaluation&view=native&model=pair` sau load → `model=` **biến mất**, notice không hiện; `?view=native` trần rơi về dashboard. Nay `node --test tests/model-param.test.cjs` **9 passed**; 3 deep link đều hiện notice |

12 id baseline đóng bởi #11, #12, #13 (×7) và `circus_cli` tự xanh (`TASK2_TRIAGE.md` bảng tổng).

## Blocker với bằng chứng, tác động, lệnh chạy tiếp theo

### 1. `clvc_net` — cần hook **hai model** (không phải thêm tham số)

```
$ workbench/envs/ctl312h/bin/python workbench/docs/reviews/20261112-parity-repair/scripts/third_model_parity.py \
    --model clvc_net --phase identity --category dress
this producer cannot measure 'clvc_net'; it measures ['conesep', 'habit', 'hint', 'intent', 'tgcir'].
A model absent here has no capture hook, which is a missing implementation, not a measurement
```
Cơ chế, trích code pin: `third_party/CLVCNet/test.py:48,116` gọi
`torch.cat([F.normalize(local_model.extract_img_feature(imgs)[1]), F.normalize(global_model.extract_img_feature(imgs)[1])], dim=1)`
→ một hàng 2048-d từ **hai** mạng; contract chung là `native_features(paths, texts)` trên **một** mạng.
**Tác động:** dòng "CLVC usable" đứng trên 12 ô diagnostic (đã PASS), nhưng **không** có pair
native↔adapter → không có verdict G1/G2. **Chạy tiếp:**
```
workbench/envs/ctl312h/bin/python workbench/docs/reviews/20261109-native15-composition-limn/scripts/t4_parity.py --help   # contract writer
# seam: một ClippedModelCapture dạng hai-model (local + global), transform_source='targetpad', population_scope='sample'
workbench/envs/ctl312h/bin/python workbench/scripts/gpu_job.sh workbench/envs/clvc39-la/bin/python -u \
  workbench/docs/reviews/20261112-parity-repair/scripts/third_model_parity.py --model clvc_net --phase identity --category dress
```

### 2. `tme` — không có API một-category

```
$ …/third_model_parity.py --model tme --phase identity --category dress
this producer cannot measure 'tme'; it measures ['conesep', 'habit', 'hint', 'intent', 'tgcir'].
```
Và chính evaluator từ chối subset, `workbench/replay/tme.py:352-356`:
`ValueError: TME native evaluator only supports the full FashionIQ category set`. Dataset đọc qua
global module (`tme_data.FashioniqPath`, `:365`). **Tác động:** hình dạng artifact mỗi pair đang dùng
(theo category) không biểu diễn được cho `tme`. **Chạy tiếp:** cần một artifact dạng ba-category
(`population_scope: 'full'`) rồi mới tới lệnh trên; đo trước bằng
`workbench/envs/tme-adapted/bin/python -u workbench/replay/tme.py --help`.

### 3. `ptha_mtst` — quyết định phạm vi, không phải lỗi producer

`models.yaml:392 reproduction_status: SELECTED_TRANSFER_ONLY`; checkpoint `cirr_transfer` (CIRR→FashionIQ).
**Tác động:** một pair ở đây sẽ đo một cấu hình mà chiến dịch **không** coi là tái lập FashionIQ, nên
nó sẽ dán nhãn sai. **Chạy tiếp:** quyết định của acceptance owner — hoặc ghi "không pair, transfer-only"
vào chuẩn, hoặc đo và ghi rõ `transfer_only: true` trong artifact. Không có lệnh nào tự quyết.

### 4. Năm model `ENV_VAR_UNSET` + `sprc` (thiếu adapter class + thiếu ViT-g)

```
$ grep -n -E 'WORKBENCH_(AIRKNOW|PAIR|DCNET|CLIP4CIR|SPRC)_PYTHON' workbench/config/workbench.env
(exit=1, không output)
$ workbench/envs/ctl312h/bin/python workbench/scripts/evaluate_models.py --model airknow \
    --checkpoint fiq_n05 --protocol fashioniq_full_gallery_ref_excluded --dry-run
[BLOCKED] official command construction failed: execution environment not verified [UNCONFIGURED/UNCONFIGURED]:
          WORKBENCH_AIRKNOW_PYTHON=None (WORKBENCH_AIRKNOW_PYTHON not set)
```
`dcnet`, `combiner_rn50x4_noft`, `clip4cir_rn50x4_fullft`, `sprc` cho cùng dạng.
`sprc` thêm hai blocker riêng: `workbench/composition/` **không có** `model_sprc.py` (class không tồn
tại) và `sprc_fiq.pt` (ViT-g của paper) **vắng toàn home** — `find /home/heheboiz -name 'sprc_fiq.pt'`
= **0 hit**, trong khi `sprc_fiq_vitl.pt` **1 459 756 541 B** có mặt và đã đo.
**Tác động:** 6/17 model không thể chạy; native row của chúng có (airknow 3 counted, csmcir 2,
ptha_mtst 2, sprc 1, clip4cir_fullft 1, combiner 1) nhưng không có phép đo mới nào kiểm chứng lại.
**Chạy tiếp:** điền biến trong `workbench/config/workbench.env` (operator), rồi
`workbench/envs/ctl312h/bin/python workbench/scripts/manage_environment.py --list` → `--model <id> --create --allow-env-install --dry-run`.
`sprc`: phải có `workbench/composition/model_sprc.py` **và** `sprc_fiq.pt` trước.

### 5. Ba symlink interpreter treo **ngoài** khai báo (5 khai báo + 3 thừa = 7 trên đĩa)

```
$ readlink -f workbench/envs/airknow-adapted/bin/python   ; echo $?
1
$ ls /home/heheboiz/miniconda3/envs/
VisRAG  melt  teacher-csmcir          # teacher-sprc KHÔNG CÒN
```
5 symlink **có khai báo** cùng trỏ `/home/heheboiz/miniconda3/envs/teacher-sprc/bin/python` (đã mất):
`airknow-adapted`, `pair-adapted`, `sprc-adapted`, `clip4cir-adapted`, `limn-adapted`
(`blockers.txt` §INTERPRETER_BROKEN_SYMLINK). Trên đĩa còn **3** symlink treo nữa mà registry và
`workbench.env` **không** tham chiếu: `dl-adapted`, `test-adapted` — và `limn-adapted` là alias thừa
(LIMN thật chạy `limn310-adapted`, đang tốt). **Tác động:** không model nào trong 17 bị chặn *chỉ* vì
alias thừa; nhưng một reader dùng `workbench/envs/<model>-adapted` sẽ nhận exit 1 không rõ nguyên nhân.
**Chạy tiếp:** `find workbench/envs -maxdepth 3 -name python -xtype l` để liệt kê, rồi tạo env thật
hoặc xoá alias; không có lệnh nào trong repo tự sửa (đúng: không âm thầm trỏ lại link).

### 6. `pair` — deferred, hai blocker độc lập

```
$ workbench/envs/ctl312h/bin/python workbench/scripts/evaluate_models.py --model pair \
    --checkpoint pair_b1 --protocol fashioniq_val_split --dry-run
[RUN] pair / pair_b1
[BLOCKED] command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT
[BLOCKED] checkpoint mapping unresolved
```
Cộng `WORKBENCH_PAIR_PYTHON=None` và symlink `pair-adapted` treo. **Tác động:** native row chỉ còn ở
mức `EXPLORATORY_PUBLIC_MAPPING`; mọi số PAIR là exploratory, không phải reproduction.
**Chạy tiếp:** cần source class của pickle (ngoài repo) → khi mapping verify được, đặt biến rồi chạy
lại đúng lệnh trên **không** `--dry-run`.

### 7. `max_rel` — quyết định **lớp**, chưa chốt

19/19 failure còn lại là `max_rel` trên mẫu số ~1e-5. `max_abs` worst: encoder 1.65e-06, tgcir
2.90e-06, habit 1.94e-07, hint 1.53e-07 — đều trong atol `1e-4`. **Tác động:** không model nào PASS,
và **0/19** là lỗi khoa học đã chứng minh; nhưng cũng **không** được phép dời tolerance sau khi đọc
số. **Chạy tiếp:** chọn một trong hai, ghi thành quyết định chuẩn (không phải tweak từng model):
(a) scope `max_abs`-only; (b) `max_rel` có **sàn mẫu số** khai báo **trước** khi đo. Cả hai đổi
`V2_STANDARD.md` + `t4_parity.TOL` và sẽ lật 19 failure trong một lần — thuộc acceptance owner.
Đo bổ trợ rẻ nhất (11 s GPU, đọc-only với cây): hai lượt batch-N khác N, cùng số component so sánh —
nếu `max_abs` giữ ~1e-6 mà `max_rel` đổi thì mẫu số là nguyên nhân (`TGCIR_BATCH.md` §5).

### 8. Ba file `population_habit_*.npz` chưa track

```
$ git status --porcelain --untracked-files=all | grep npz
?? workbench/docs/reviews/20261112-parity-repair/raw/third_model/population_habit_dress.npz   121.3 MiB
?? …population_habit_shirt.npz   200.3 MiB
?? …population_habit_toptee.npz  169.8 MiB
```
`git check-ignore` → **không** ignore (chỉ untracked). `CLEANUP_REPORT.md.save` (2.9 KiB) cũng untracked,
không ai đọc. 21 `.npz` khác trong cùng thư mục **đã** track (782 MiB tổng thư mục).
**Tác động:** sweep và mọi commit đều phải né chúng; một `git add -A` sẽ nhét thêm 491 MiB vào repo
và guard `test_handoff_commands` sẽ so byte ở một cây khác. **Chạy tiếp:** quyết định của owner —
xoá (`rm` ba file) hoặc thêm vào `.gitignore`; **không** commit.

### 9. Người gửi SIGTERM lịch sử — `UNDETERMINED`

Bốn stack chết 2026-11-11 (02:25:51, ~03:06, 03:53:06, 07:29:33). Journal của unit **không** phủ được:
entry đầu `2026-11-11T23:48:44`, tức unit được tạo **sau** cả bốn lần chết (hồi đó là
`serve_workbench.py` chạy tay, không có unit journal → vắng mặt là fact về coverage, không phải bằng
chứng). Hai nghi phạm bị loại bằng đo: `systemd-oomd` `inactive`, 0 entry 11-11→11-13;
`systemd-logind` không có session close trong cả bốn cửa sổ. `auditd` **không cài** → không có và
không thể tạo audit trail hồi tố. **Tác động:** không quy được trách nhiệm, và một lần nữa sẽ không
quy được. **Chạy tiếp:** (a) `sudo auditctl -a always,exit -F arch=b64 -S kill -S tkill -S tgkill -k cir-kill`
trước lần lặp sau; hoặc (b) chạy stack bằng đúng unit đã viết để mọi stop thành một dòng journal;
hoặc (c) log `signum` trong `forward_stop`. Không bịa nghi phạm.

## Lệnh tái lập và lệnh tiếp tục

```bash
# 0. cây đã kiểm thử (KHÔNG dùng git reset --hard; PC_PILOT_HANDOFF.md:60)
git clone git@github.com:Le-Minh-Nhut/cir.git && cd cir
git checkout 90f7b2c1e1f6b8474570961ae5a2e08ecf2ee36c     # = TESTED_SOURCE_SHA
git rev-parse HEAD                                        # phải in đúng SHA trên
git diff --quiet 90f7b2c HEAD -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'; echo $?   # 0

# 1. sweep (một tiến trình, ~64 phút; cây phải đứng yên)
mkdir -p /tmp/sweep_basetemp_90f7b2c
setsid nohup workbench/envs/ctl312h/bin/python -m pytest workbench/tests \
  -q -p no:cacheprovider -rfE --tb=short --basetemp=/tmp/sweep_basetemp_90f7b2c \
  > /tmp/sweep_90f7b2c.txt 2>&1 < /dev/null &

# 2. gate in-process (không HTTP — service :8000 không --reload nên có thể cũ)
workbench/envs/ctl312h/bin/python \
  workbench/docs/reviews/20261113-final-acceptance/raw/parity-coverage/run_gate.py

# 3. acceptance (chạy sau khi /api/models đã warm, nếu không sẽ false "no server")
curl -s -o /dev/null http://127.0.0.1:8000/api/models
bash workbench/docker/acceptance.sh            # kỳ vọng: ACCEPTANCE PASSED (skipped: G-status-unverified)
bash workbench/docker/release_verify.sh --report /tmp/rv.json
workbench/envs/ctl312h/bin/python -m pytest workbench/tests/test_handoff_commands.py -q   # 8 passed

# 4. probe chẩn đoán (đọc-only, không GPU)
curl -s 'http://127.0.0.1:8000/api/composition/diagnostics/clvc-dual-scope?qid=7:3790' | jq '{status,complete,defects}'
curl -s 'http://127.0.0.1:8000/api/composition/diagnostics/limn-dual-scope?qid=7:3038' | jq '{status,complete,defects}'
curl -s 'http://127.0.0.1:8000/api/composition/runs/7b1fa85007bffdab931f7f19' | jq '.adapter_query_scope_resolution'
curl -s 'http://127.0.0.1:8000/api/runs/parity-gate' | jq '{standard,passed_models,f:([.models[]|.failures|length])}'
workbench/envs/ctl312h/bin/python \
  workbench/docs/reviews/20261113-final-acceptance/raw/live-verify/probe_live_verify.py

# 5. rollback — thủ tục của repo (không có lệnh "rollback" riêng; đây là nó)
git checkout 90f7b2c1e1f6b8474570961ae5a2e08ecf2ee36c
(cd workbench/frontend && npm run build)      # dist gitignored → PHẢI build lại, không checkout được
workbench/envs/ctl312h/bin/python workbench/systemd/render_units.py --install
systemctl --user daemon-reload
systemctl --user restart cir-workbench-backend.service cir-workbench-frontend.service
systemctl --user show cir-workbench-frontend -p KillMode -p TimeoutStopUSec -p NRestarts
```
Ghi chú rollback: `workbench/frontend/dist/` nằm trong `.gitignore:95` (0 path tracked), nên checkout
pin **không** khôi phục bundle; nếu bỏ qua `npm run build` thì server tiếp tục phục vụ bundle mới
trong khi source cũ — trạng thái tệ nhất, vì `index.html` cũ vẫn trỏ tới file mới còn tồn tại.

## Giới hạn của bằng chứng

Những điều **không** được chứng minh, nói thẳng:

1. **Không model nào PASS gate.** `passed_models: []`. Bảy model được phán, 0 đạt. Đây là kết luận từ
   artifact, lớp `measured_disagreement`, không phải lỗi tải hay lỗi parse.
2. **Bốn trong năm pair hỏng chỉ vì số hạng tương đối.** encoder, habit, hint, limn (và tgcir) fail
   trên `max_rel`; **mọi** `max_abs` nằm trong atol (`≤ 2.9e-06` vs `1e-4`) và `min_cosine ≥
   0.9999999`. "Hỏng" ở đây nghĩa là *không đạt ngưỡng tương đối đã khai*, không phải *sai số đo*.
3. **tgcir: pair là sample-scope, `INCOMPLETE`.** `population_scope: "sample"` → verdict nói về mẫu
   **5 query / 20 ảnh**, không phải quần thể 2017/2038/1961. `metrics_parity` **missing**. `max_rel`
   0.2403 là do **mẫu số**: `max_abs` 2.898e-06 chia cho component cỡ **1.206e-05**; atol `1e-4` được
   thoả **34×**. Không có reduction nào trên trục batch trong scoring path (`model.py:203,209`, chỉ
   `dim=1`), nên khác biệt là thứ tự cộng float32.
4. **Hai query mơ hồ vẫn chưa chấm được theo canonical.** `7:3038`, `7:3790`: 119 ambiguous, **117**
   resolved, 2 còn lại unscoped; `canonical_status` = `PARTIAL` (6001/6003) cho **cả** CLVC và LIMN.
   Không luật trọng tài nào được phép gán chúng vào một category mà release không nêu tên
   (`LEDGER.md` L13 = `EXTERNAL`, không đổi).
5. **`image_only_empty_text` không hỗ trợ cho CLVC.** `text_model.py:84-86` IndexError với 0 token →
   trạng thái `ABLATION_UNSUPPORTED` có lý do, **không** được nặn ra rank. Vì vậy UI CLVC hiển thị
   **ba** condition (`multimodal`, `text_only_black_image`, `clvc_image_encoder_only`), không phải bộ
   ba của LIMN với condition thứ ba bị thay lặng lẽ.
6. **Caveat 117 query caption-routing.** Dưới cả ba condition, category của **117/6001** query được
   chọn từ `(reference stem, edit caption)` — tức một condition *bỏ* caption vẫn dùng nó để route;
   5884 query còn lại do loader gán (cohort B, `caption_free: true`). Chênh lệch nDCG đo được:
   `0.33061712371309204` vs `0.33400015944039774` (**0.0033830357273053**, ≈1.02%). **Không** được
   nói "routing chỉ dùng reference stem" cho cả cohort. Cả hai query diagnostic **không** nằm trong
   117, nên caveat không đổi 12 rank — nó đổi cách mô tả aggregate.
7. **Không có parity native↔composition ở quy mô đầy đủ 6003 query.** Mọi pair hiện có là sample
   (5 query / 20 ảnh) trừ phần population encode của encoder/habit. Con số 6003 chỉ là **kích thước
   cohort**, không phải số component đã so. `component_parity` full chỉ ở mức 2560 component/mẫu.
8. **Chưa đo `systemctl stop` trên unit thật.** Đo trên instance cô lập cùng argv/cwd/`Environment=`
   (port 8795/8796/8797). Chưa tập: pha cgroup SIGKILL, containment của `node` sống sót, bookkeeping
   `Restart=always`, và bản thân việc enforce `TimeoutStopSec`. Worst case backend (`GRACE_SECONDS=10`
   + 5 s + `join(10)` = 25 s vs 30 s) vẫn là **số học**, chưa falsify — không có job sống trong
   instance cô lập.
9. **Acceptance sau restart có thể false-negative.** `live()` chỉ probe một lần; `/api/models` nguội
   (~1 phút phân giải checkpoint-availability) ⇒ `SKIP A/C/G/I (no server)`. Phải warm trước khi đọc
   kết quả. Lượt chạy đầy đủ của tôi (sau warm) là **12 PASS / 1 SKIP / 0 FAIL, rc 0, 373.47 s** —
   đúng bằng `raw/live-verify/REPORT.md` §9, và là quan sát trực tiếp chứ không phải trích dẫn.
10. **`release_verify.sh` lần chạy gần nhất ghi `verdict: RELEASE_BLOCKED`** (report
    `generated_at 2026-11-11T08:28:55Z`, trước pin hiện tại). Không có report mới trong campaign; đừng
    trích report cũ như trạng thái hiện tại — chạy lại là việc của lượt sau.
11. **Native-eval "9" ≠ đã tái lập.** Đó là "có aggregate JSON"; `verified: 0` trong census
    measurement. `raw/native/encoder.json` và `.../tme.json` không có `replay_type` nên **không** là
    row cũng **không** là backing report — loại trừ có predicate, không phải lỗ hổng.
