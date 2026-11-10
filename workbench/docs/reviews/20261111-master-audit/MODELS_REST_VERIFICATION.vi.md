# M-08 · M-09 · M-10 — Xác minh upstream và artifact (đợt 2026-11-11)

Phạm vi: mục 4 của `CIR_MASTER_FIX_AND_UPSTREAM_VERIFICATION.vi.md`, các ca **M-08** (CLIP4Cir
FullFT, Combiner NoFT), **M-09** (DCNet, TME, PTHA+MTST), **M-10** (ConeSep, HABIT, INTENT, HINT,
CSMCIR, SPRC).

Trạng thái dùng trong file: `CONFIRMED` (đã kiểm bằng lệnh/`file:line`), `FIXED_ALREADY` (defect cũ,
mã hiện tại đã đúng), `REFUTED` (khẳng định cũ sai), `BLOCKED` (thiếu input/quyền), `NEEDS_RUNTIME`
(chỉ phân xử được bằng một lượt chạy), `UNVERIFIED` (chưa đủ bằng chứng).

Môi trường đọc: interpreter `workbench/envs/ctl312h/bin/python`. Service sống ở `127.0.0.1:8000`
(uvicorn pid 2820650, cha `serve_workbench.py` pid 2820649, khởi động `2026-11-10 08:47:09`);
tiến trình này **không** bị restart trong đợt audit. Mọi probe dùng `/api/benchmark-runs` và
`/api/native-measurements`; **không** gọi `/api/models` (154 s cold).

Digest sha256 tự tính trong đợt này (`sha256sum`, không chép từ artifact):

| file | sha256 |
|---|---|
| `checkpoints/clip4cir/FashionIQ/RN50x4_fullft/fiq_comb_RN50x4_fullft.pt` | `7c7c0a10fb4c995cb1ded9ad70a58891d50c622263c54131523bfb232b493f8e` |
| `checkpoints/clip4cir/FashionIQ/RN50x4_fullft/fiq_clip_RN50x4_fullft.pt` | `4111ac139cf96ac0cf31b30bf8f55b654247b06f0f5fbee4fe8dcc8c1c7cfd2f` |
| `checkpoints/clip4cir/FashionIQ/RN50x4_noft/fiq_comb_RN50x4_noft.pt` | `b31882fe6f7e3e194cb9cfda048926844ee0ccc9d4ebc5483fb3bacfa5972955` |
| `cohort/downloads/ptha_mtst/tuned_mtst_cirr.pt` | `1c4dab86a9f66dfe10aff75b39790432eb6cc746cf27b6cfab74b66d93b7588f` |
| `checkpoints/dcnet/trained_model.pth` | `130f031e0f7808ccf5f15630a691bd7af95ad9fadbf94f64acfce64e7d8b850f` |
| `cohort/downloads/habit/HABIT-FIQ_N0.2.pt` | `814d2d0ceb29ef910f37562e8bea8a76cf38aa546fbc2102e495df687c276b03` |
| `cohort/downloads/conesep/ConeSep-FIQ-N0.2.pt` | `5a864ee5a78c1d249320c4f997cc7d14b529b329b2efba5577c82dbe019ed9d2` |
| `cohort/downloads/intent/intent_fiq_0.2.pt` | `2a504cadfd6b1c2486a2c508538b0160c5653f62c7cd11f4f3d8a9534dc26afc` |
| `cohort/downloads/hint/fashioniq.pt` | `bf9c72d42206b667ed586758c15c648970df821857e4067f26155a114a6517d4` |
| `cohort/ckpt/tme/fashioniq_0_best_model.pth` | `fea7fc08760dad1f5a66184ea7e65b45388f9781157cbd33a3b64db74103968a` |
| `checkpoints/sprc/sprc_fiq_vitl.pt` | `19386104271c01ea03ee2177575c0e55f98edd3ba61a116d78f2e61ab5dbea08` |
| `artifacts/checkpoints/csmcir/fashioniq_tuned_clip_best.pt` | `c8812071de0640d647c5ba2a70628b80a3289ee2c47676f7a1b6abaa313fa49e` |

Ba digest cuối **khớp** giá trị đã khai (`models.yaml:1427`, `csmcir_blocker.json.checkpoint_sha256`);
các digest còn lại khớp trường `checkpoint_sha256` / `checkpoint.sha256` trong artifact tương ứng —
tức không checkpoint nào bị thay sau khi đo.

---

## M-08a · CLIP4Cir FullFT — `clip4cir_rn50x4_fullft`

- **Upstream:** `https://github.com/ABaldrati/CLIP4Cir` @ `dfed9f748a8a4d05abf613164e06d59b22dbdf13`
  (`workbench/registry/models.yaml:1093`; vendor **byte-identical** với clone — dữ kiện có sẵn).
- **Workbench path:** `workbench/third_party/CLIP4Cir/src/validate.py`; artifact
  `workbench/docs/reviews/20261109-native15-composition-limn/raw/native/clip4cir_fullft.json`.
- **Checkpoint:** `fiq_clip_RN50x4_fullft.pt` `4111ac13…`, `fiq_comb_RN50x4_fullft.pt` `7c7c0a10…`
  (hai member của một bundle; registry `BLOCKED_CHECKPOINT_BUNDLE`, `models.yaml:1193`).

- **Protocol = `fashioniq_original_split`. CONFIRMED.** `validate.py:21-57` `compute_fiq_val_metrics`
  **không** có dòng mask nào; mask duy nhất trong file là `:162-166`, nằm trong
  `compute_cirr_val_metrics`. Gallery = toàn bộ `split.<cat>.val.json` (`:127-131`,
  `src/data_utils.py:144-147`); `gallery_count` 3817/6346/5373 = full split.
- **Ranking. CONFIRMED.** `distances = 1 - predicted_features @ index_features.T` rồi
  `torch.argsort(distances, dim=-1)` **tăng dần** (`:44-45`); `:51` `assert` đúng một positive mỗi
  query; recall `* 100` (`:54-55`) ⇒ **phần trăm**.
- **Ổ SỐ KHÔNG CÓ ARTIFACT ĐỠ — defect nặng nhất của đợt này.** Ba nguồn, ba bộ số:

  | nguồn | macro R@10 | macro R@50 | dress R@10 | toptee R@50 |
  |---|---|---|---|---|
  | `raw/native/clip4cir_fullft.json` (đo thật) | 43.8016931215922 | 66.98358456293742 | 39.06792402267456 | 71.29015922546387 |
  | `measurements/clip4cir_parsed.json` (roll-up) | **43.81821850935618** | **66.95005893707275** | **39.11750018596649** | **71.23916139404** |
  | `measurements/selection_decisions.json` (roll-up) | 43.8182 | 66.9501 | — | — |

  `shirt_*` và `toptee_r10` trùng khít giữa hai bộ; `dress_*` và `toptee_r50` khác ⇒ **hai lượt
  chạy khác nhau**, không phải sai số đọc. Không file nào dưới `checkpoint_selection/` chứa lượt đã
  sinh ra 43.8182: `/api/native-measurements` phục vụ **cả hai** (`clip4cir_fullft` đường
  `clip4cir_fullft`, và `clip4cir_rn50x4_fullft` đường
  `selected_2026-11-06.clip4cir_rn50x4_fullft.metrics`), tức một con số hiện hai lần dưới hai tên,
  và con số đó **không** tái lập được từ artifact native đang có.
- **Nhãn artifact sai và migration chưa xong.** `raw/native/clip4cir_fullft.json` khai
  `protocol: "fashioniq_full_gallery_ref_excluded"` ở cấp document **và** trong từng
  `categories[].protocol`, trong khi `protocol_source` của chính nó mô tả cơ chế "mask index entry"
  của một scorer **không** mask. Bản `workbench/docs/reviews/20261110-e2e-repair/raw/clip4cir_fullft.migrated.json`
  đã sửa cấp document thành `fashioniq_original_split` nhưng `categories[].protocol` vẫn còn
  `fashioniq_full_gallery_ref_excluded`. ⇒ **FIXED_ALREADY một nửa**, còn defect ở cấp category.
- **Units:** đường này đã là %, catalog suy `percent`; không có ca lệch thang.

Trạng thái: **CONFIRMED** protocol/ranking/units; **CONFIRMED (defect mở)** roll-up 43.8182 vô
artifact + `categories[].protocol` chưa migrate.

Neo: `awk 'NR>=21&&NR<=57' workbench/third_party/CLIP4Cir/src/validate.py`;
`python -c` đọc ba file JSON ở bảng trên; `curl :8000/api/native-measurements`.

## M-08b · Combiner NoFT — `combiner_rn50x4_noft`

- **Upstream:** cùng repo/SHA với M-08a (`models.yaml:1091-1093`); khác entrypoint train và
  `--combining_function`. **Không bao giờ gộp** hai cấu hình.
- **Checkpoint:** `fiq_comb_RN50x4_noft.pt` `b31882fe…`; `checkpoint_mapping_status: UNVERIFIED`,
  `reproduction_status: BLOCKED_CHECKPOINT_MAPPING` (`models.yaml:1073-1074`;
  `selection_decisions.json` ghi `EMPIRICALLY_SELECTED_MAPPING_UNVERIFIED`).
- **Số:** roll-up `selection_decisions.json` = 35.1407 / 58.4436 / 46.7922; run thật
  `20261108T064137301769Z_combiner_rn50x4_noft_rn50x4_noft_unresolved_020a70b0` đo
  **35.1078 / 58.4756 / 46.7917**, `dress_r10` 30.639564990997314 so với roll-up 30.689141154289246
  ⇒ lại là *cùng tên, khác lượt chạy*, cùng pattern với M-08a. Trạng thái: **CONFIRMED**.

### Chuỗi `METRIC_REFERENCE_UNVERIFIED` — kiểm từng mắt

1. `workbench/backend/paper_gate.py:333-336`: `if reference is None: return GateVerdict(
   METRIC_REFERENCE_UNVERIFIED, … "without a verified published row there is nothing to
   reproduce")`. Đây đúng là giá trị phục vụ: run trên có
   `benchmark_gate.status = "METRIC_REFERENCE_UNVERIFIED"`,
   `blockers = ["no committed reference describes this experiment"]`.
2. `reference is None` đến từ `benchmark_references.py:258-275`: có 3 row cho model này nhưng
   `verified` rỗng ⇒ `SELECTION_NO_PUBLISHED_COMPARISON` + `ReferenceSelection(None, …)`. Ba row:
   `clip4cir_rn50x4_noft_fiq_absent_in_paper` (`papers.yaml:293`),
   `combiner_rn50x4_noft__clip4cir_rn50x4_fiq_gallery_protocol` (`:356`),
   `combiner_rn50x4_noft__clip4cir_paper_note_backbone_inventory_and_version` (`:446`) — tất cả
   `UNVERIFIED`.
3. `benchmark_references.py:207-225` `unbenchmarked()` đọc **registry**: `True` vì `models.yaml:1073`
   không khai `reported_r10/r50/mean`. Có test khoá hành vi:
   `workbench/tests/test_release_blocker_invocation_proof.py:898-899`
   (`combiner_rn50x4_noft is True`, `clip4cir_rn50x4_fullft is False`).
4. Không hàng paper nào mô tả thí nghiệm: `papers.yaml:297-303` nói rõ release có folder
   `FashionIQ/RN50x4_noft` nhưng paper **không in dòng FashionIQ** cho cấu hình đó.

⇒ **Đây là hành vi ĐÚNG, không phải defect.** Từ chối in metric reference khi không có nguồn để
trích là đúng; nới gate để con số xuất hiện mới là defect. Trạng thái: **CONFIRMED (behaviour
correct)**. Chỉ có một điểm cần nói: `variants` không phân biệt hai cấu hình của cùng một paper, nên
FullFT (có row verified) và Combiner (chỉ có row mô tả) phải đọc kèm `reference_evidence` — hiện
payload có trường đó.

---

## M-09a · DCNet — `dcnet`

- **Upstream:** `https://github.com/ozmig77/dcnet` @ `68c79d38569f01ca39c3b4ea641b3b7eb2ccb5fb`
  (`models.yaml:989`). Vendor **1 file patch** (`base/base_trainer.py`: `map_location` +
  `weights_only=False` có điều kiện; `models.yaml:1035`), không drift khác.
- **Workbench path:** `third_party/DCNet/test.py` → `trainer/trainer_joint.py::_valid_epoch`;
  artifact `docs/reviews/20261109-native15-composition-limn/raw/native/dcnet.json`;
  measurement `artifacts/checkpoint_selection/20261106T002617+0700/measurements/dcnet_run1.json`.
- **Checkpoint:** `checkpoints/dcnet/trained_model.pth` `130f031e…`, 2 877 310 172 B, epoch 82
  ("Resume training from epoch 83").

- **Protocol `fashioniq_full_gallery_ref_excluded`; mask áp cho CẢ HAI ma trận. CONFIRMED.**
  `trainer_joint.py:216-218` gán `-10e10` vào `score_matrix_comp[ei, idx(candidate)]` **và**
  `score_matrix_corr[ei, …]`. Gallery = `image_splits/split.<cat>.val.json` qua loader `val_trg`
  (`data_loader/CE_dataset.py:22-26`), batch 100.
- **Ranking: `torch.topk`, ba lần. CONFIRMED.** `:228` (comp), `:232` (corr), `:240` (comb),
  `dim=0` (ma trận `(N_gallery, N_query)`), `largest=true` từ
  `configs/ce/fashioniq_dcnet.json:78`. Không phải `argsort`; tie policy là implementation-defined.
- **Nhánh sinh ra số được phục vụ = `comb`. CONFIRMED.** `:238-243`
  `score_comb = F.log_softmax(score_comp[j]) + F.log_softmax(score_corr[j])` rồi topk lần ba,
  `metric['comb'][i] = r10_c, r50_c`. Không phải suy đoán: checkpoint được chọn theo
  `configs/ce/fashioniq_dcnet.json:74` `"monitor": "max comb_avg"`. Artifact ghi cùng kết luận
  (`reported_column: "comb (log-softmax(comp) + log-softmax(corr))"`), và hai cột `comp`/`corr` được
  lưu nguyên văn nên kiểm lại được.
- **`r1 = null`** vì `_valid_epoch` chỉ tính R@10/R@50 cho FashionIQ (`model/metric.py:12-21`).

### Đơn vị — API phục vụ thang nào

| nơi | macro R@10 | thang |
|---|---|---|
| `raw/native/dcnet.json` → `aggregate.macro_r10` | `0.27929730914548917` | **phân số** |
| cùng file → `aggregate.macro_r10_percent` | `27.929730914548916` | phần trăm (khoá riêng) |
| `$D1/measurements/dcnet_run1.json` → `aggregate.macro_r10` | `27.929730914548923` | **phần trăm** |
| `selection_decisions.json` → `selected_2026-11-06.dcnet.metrics.macro_r10` | `27.9297` | phần trăm |

- **API phục vụ PHẦN TRĂM, hai dòng cùng một số:** `dcnet` (roll-up, đường
  `selected_2026-11-06.dcnet.metrics`, từ `selection_decisions.json`) và `dcnet` (đo trực tiếp,
  đường `aggregate`, từ `dcnet_run1.json`), cả hai `27.9297…`.
- **File phân số không được phục vụ — vì vị trí, không vì cơ chế.** `raw/native/dcnet.json` nằm
  dưới `workbench/docs/reviews/`, ngoài `WORKBENCH_ARTIFACTS_ROOT`, nên
  `list_native_measurements()` không quét tới. Kiểm: trong 49 dòng phục vụ, **0 dòng** có mọi giá
  trị trong `0..1`.
- **UI in cùng thang. FIXED_ALREADY.** `format.ts:51-57` `fmtRecall(value, unit, digits)`:
  `fraction → *100`, `percent → giữ nguyên`, `null → số thô + "(đơn vị không ghi)"`. Hai nơi in
  recall native (`OverviewPage.tsx:127`, `EvaluationPage.tsx:613-614`) đều truyền `row.unit`.
- **Lỗ hổng tiềm ẩn (đang sống trong code):** `unit` **không đến từ artifact** mà suy theo độ lớn
  (`native_measurements.py:105-120`). Không file nào khai unit: `counts.unit_declared = 0`,
  `unit_inferred = 43`, `unit_undeclared = 0`. `UNIT_KEYS = ("metric_unit","units","unit",
  "recall_unit")` (`:81`) — `dcnet_run1.json` và `raw/native/dcnet.json` **đều không** có khoá nào
  trong bốn khoá đó, dù cả hai có sẵn `macro_r10_percent` / `macro_r50_percent`. Hệ quả: block phần
  trăm nào có **mọi** giá trị ≤ 1 (model dưới 1 % ở cả ba category) sẽ bị đọc là `fraction` và in
  lớn gấp 100 lần. Đây là chỗ duy nhất trong M-08..M-10 còn trộn thang được một cách âm thầm.
  Patch đề xuất (không sửa production trong đợt này): thêm cặp `*_percent` vào nhận diện, cho block
  có sibling `macro_r10_percent` khai `percent` với `unit_source="sibling_key"`.
- **Ảnh:** `CE_dataset.py:19` đọc `resized_images/` (số nhiều); eval transform
  `CenterCrop(224) → ToTensor → Normalize(ImageNet)`, **không Resize**
  (`data_loader/data_loaders.py:66-70`). Trên đĩa `data/FashionIQ/resized_images/` có 15 415 jpg,
  mẫu 40/40 `256×256`; manifest `measurements/resized_images.json`:
  `"recipe": "LANCZOS 256x256 JPEG q95"`. ⇒ "pre-resized to 256" **đúng ở mức artifact của lượt
  này**, **không** được source khai (`UNVERIFIED` ở mức source).

Trạng thái: **CONFIRMED** protocol/mask/nhánh-`comb`/đơn-vị-phục-vụ; **FIXED_ALREADY** cho thang đo
trên UI; **CONFIRMED (defect tiềm ẩn)** cho suy unit theo độ lớn.

## M-09b · TME — `tme` (`fashioniq_0`)

- **Upstream:** `https://github.com/li-shuxian/TME` @ `394aad79df2c104feb32a2c96132582012765062`.
  Vendor **4 file patch**, gồm `workbench/patches/tme_eval_batch_override.patch` (commit `98c0eeb`,
  2026-11-07 01:47).
- **Workbench path:** `third_party/TME/src/precompute_evaluation.py`, runner
  `cohort/scripts/tme_session_runner.py`; measurement `cohort/measurements/tme_fashioniq_0.json`
  (mtime 23:08:49), log `cohort/logs/tme_run5.log`.
- **Checkpoint:** `cohort/ckpt/tme/fashioniq_0_best_model.pth` `fea7fc08…`.

- **Protocol `fashioniq_original_split`, reference ĐƯỢC TÍNH trong gallery. CONFIRMED.**
  `provenance.reference_policy: "reference_included"`; mask chỉ có trong nhánh CIRR.
- **Ranking:** `argsort` trên `1 - similarity`, tăng dần, do code pinned gọi; R@1/5/10/50, phần trăm
  (`* 100` ở `:102-105`).
- **Hai dòng trùng khít trong catalog:** `tme_fashioniq_0.json` và `tme_normal_path_full.json` cùng
  `54.922901590665184 / 74.83842174212138`, chỉ khác `feature_cache_provenance`
  (`UNVERIFIED` vs `VERIFIED_PRIOR_RUN`) ⇒ một model, hai dòng.

### Batch size — env var nào, và lượt chạy thật đã dùng batch nào

- Env var: **`WORKBENCH_TME_QFORMER_BS`** (ảnh) và **`WORKBENCH_TME_QUERY_BS`** (query), đọc **bên
  trong** hàm: `precompute_evaluation.py:45`
  `Qformer_bs = int(os.environ.get("WORKBENCH_TME_QFORMER_BS", 256))`, `:65`
  `query_bs = int(os.environ.get("WORKBENCH_TME_QUERY_BS", 32))`. Default tác giả giữ nguyên 256/32.
- **Hai env var đó KHÔNG có tác dụng trên lượt đã ghi số.** Progress bar trong `tme_run5.log`
  (lượt sinh ra `tme_fashioniq_0.json`):

  | category | gallery | số batch trong log | batch suy ra |
  |---|---|---|---|
  | dress | 3817 | `15/15` | **256** |
  | shirt | 6346 | `25/25` | **256** |
  | toptee | 5373 | `21/21` | **256** |

  query: dress 2017 với `64` batch ⇒ **32**; toptee 1961 với `62` ⇒ **32**. Nếu
  `Qformer_bs=64` thì dress phải 60 batch và `query_bs=8` thì phải 253 batch.
  ⇒ **lượt đã ghi dùng 256/32 (default tác giả), không phải 64/8.**
- **Vì sao log lại in 64/8:** `tme_session_runner.py:305-323` `_apply_batch_overrides()` gán
  `qe.Qformer_bs = 64` / `qe.query_bs = 8` lên **module attribute**, rồi in
  `[SESSION-NOTE] evaluator batches: Qformer_bs=64 query_bs=8` (log dòng 3). Nhưng hai tên đó là
  **local** trong `compute_metric_features`, và bản checkout lúc chạy còn hardcode
  `Qformer_bs = 256` / `query_bs = 32` (`git show 3655b89:workbench/third_party/TME/src/precompute_evaluation.py:39,59`).
  Gán module attribute là **no-op**; dòng `SESSION-NOTE` chỉ in lại giá trị vừa gán, không in giá
  trị thật. ⇒ (a) lượt chạy dùng default 256/32; (b) tồn tại một đường override chết.
- **Artifact không ghi batch nào.** `tme_fashioniq_0.json` không có khoá `Qformer`, `query_bs`,
  `batch`, `WORKBENCH_TME`. `CHECKPOINT_SELECTION_REPORT.md:742-743` lại khai 64/8 như điều kiện
  **bắt buộc** để tái lập ⇒ tài liệu mô tả một cấu hình **không phải** cấu hình đã chạy.

Trạng thái: **CONFIRMED** protocol/ranking/units/hai-dòng-trùng; **REFUTED** cho "env var đổi batch
của lượt đã ghi" (được set, được in, không có tác dụng); **NEEDS_RUNTIME** nếu muốn một lượt 64/8
thật (chạy sau `98c0eeb` và ghi batch vào artifact).

## M-09c · PTHA+MTST — `ptha_mtst` (transfer)

- **Upstream:** `https://github.com/zyinan99/MTST_PTHA` @ `6a02ee4763fc2260b6abfb18be78853facb91d86`.
  Vendor **3 file patch**, không drift lạ.
- **Workbench path:** `workbench/replay/ptha_mtst.py` → `validate_blip.py::compute_fiq_val_metrics`;
  artifact `cohort/measurements/ptha_transfer_run1.json` (240 s, `EXITCODE=0`).
- **Checkpoint:** `tuned_mtst_cirr.pt` `1c4dab86…`, top-level key `Blip2QformerCirCat`,
  `strict_load: true`, 0 missing / 0 unexpected.

- **Nhãn transfer có trên artifact VÀ trong payload API. CONFIRMED.** Artifact:
  `"label": "PTHA+MTST | CIRR checkpoint -> FashionIQ | cross-dataset transfer | exploratory"`,
  `train_dataset: "CIRR"`, `evaluation_dataset: "FashionIQ"` (sinh từ
  `replay/ptha_mtst.py:35` `TRANSFER_LABEL`, gán ở `:380`). Dòng `ptha_mtst`/`cirr_transfer` trong
  `/api/native-measurements` trả đúng chuỗi đó ở `label`.
- **Nhưng UI KHÔNG in `label` của dòng native — defect hiển thị.** `NativeMeasurementTable`
  (`EvaluationPage.tsx:600-628`) in `model_id`, `metric_path`, `source.path`, `unit`,
  `mapping_status`; `NativeMeasurementDetail` (`:643-668`) liệt kê 11 khoá, **không** có `label`.
  `OverviewPage.tsx:122-135` cũng không. `transferLabel()` (`catalog.ts:238-242`) chỉ áp cho run
  composition. ⇒ 37.90 hiện ra ở bảng native **không** kèm chữ "transfer".
- **Projection ở biên: có trong artifact, KHÔNG có trong văn bản cạnh con số.**
  `ptha_transfer_run1.json` `target_feature_contract.{dress,shirt,toptee}`:
  `action: "projected_at_boundary_via_inference_target"`, `before_shape [N,32,768] → after_shape
  [N,32,256]`, `projection_parameters_reused: 196864`, `new_parameters_introduced: 0`,
  `equivalence_vs_direct_projection_max_abs_diff: 0.0`, `double_projection_possible: false`; cài đặt
  `replay/ptha_mtst.py:246-265`. Nhưng `grep -i "inference_target\|boundary" ` trên
  `20261109-native15-composition-limn/EVIDENCE_INDEX.md`, `FINAL_REPORT.vi.md`,
  `NATIVE_RUNS_EXECUTED.vi.md` **không** trả dòng nào thuộc PTHA ⇒ yêu cầu "projection được ghi ở
  nơi con số sống" chỉ đạt một nửa.
- **Processed ID có tính filter PNG thiếu. CONFIRMED.** Upstream lọc gallery bằng
  `if not file_path.exists(): self.image_names.remove(image_name)`
  (`third_party/MTST_PTHA/MTST_PTHA/src/data_utils.py:174-177`), và `__getitem__` bọc
  `except Exception: print(...)` không raise (`:217-218`). Trên host này filter **không bỏ gì**:
  hợp ba `split.*.val.json` = 15 415 ID, `data/FashionIQ/images` có 74 381 png, thiếu **0**; shape
  sau trích xuất đúng `[3817]/[6346]/[5373]`. ⇒ processed ID khớp full split. Lưu ý: artifact
  **không** ghi trường "0 filtered"; kết luận này suy từ shape + phép đếm ở trên.
- **Rác catalog:** `cohort/measurements/ptha_transfer_run2.json` là **log văn bản** đuôi `.json`
  ⇒ một dòng `unreadable: JSONDecodeError` (1 trong 2 dòng unreadable của catalog).

Trạng thái: **CONFIRMED** nhãn-nơi-lưu / checkpoint / shape / filter; **CONFIRMED (defect UI)** nhãn
transfer không hiển thị; **UNVERIFIED** "projection được ghi cạnh con số" ở prose.

---

## M-10a · ConeSep — `conesep`

- **Upstream:** `https://github.com/iLearn-Lab/CVPR26-ConeSep` @ `bd4bf629ebe68045c1a54dcc61e4a3ea65f8759b`
  (`models.yaml` mục `model_id: conesep`); vendor **byte-identical**.
- **Workbench path:** `third_party/ConeSep/test.py`, `datasets.py`, `data_utils.py`; measurement
  `cohort/measurements/conesep_fiq_n02.json`.
- **Checkpoint:** `ConeSep-FIQ-N0.2.pt` `5a864ee5…` (registry khai cùng digest,
  `checkpoint_mapping_status: VERIFIED_METADATA`, `status: NO_CLEAN_CHECKPOINT`).

- **Split = original, gallery full category.** `datasets.py:28` `split='original-split'` là default
  của constructor; nhánh gallery `:201-217`. Reference **bị loại tại scorer**: `test.py:68`
  `sims[i, test_targets_id.index(t['source_img_id'])] = -10e10`, đặt **trước** `argsort`
  (`:71`, `:75`).
- **Điểm gộp KHÔNG phải cosine. CONFIRMED.** `test.py:59-61`
  `sims = np.matmul(all_queries, all_imgs); sims = sims.squeeze(); sims = sims.max(-1)` — max theo
  trục token của inner product. Mô tả "cosine similarity" là sai. Nhánh FashionIQ (`:44-72`)
  **không có** một dòng normalize nào; block normalize duy nhất trong file là `:123-127`, đã bị
  comment out, và nằm trong nhánh **CIRR** (`test_cirr_valset`). Đã đọc nguyên văn.
- **Đơn vị:** phần trăm, `100 * r / len(...)` (`test.py:76-80`).
- **Số:** macro `54.76130615726771 / 75.21514359343843 / 64.98822487535307`; R@1/10/50 từng
  category được lưu (`dress` 20.823004462072383 / 49.62816063460585 / 72.68220128904314).
- **Ô `dress_r1` trùng giữa ConeSep và INTENT — ĐÃ PHÂN XỬ, không phải copy.**
  `conesep` và `intent` cùng ghi `dress_r1 = 20.823004462072383`. Quy về số đếm:
  `20.823004462072383 × 2017 / 100 = 420.0` cho **cả hai**; các ô còn lại thì khác
  (`conesep` dress `r10 = 1001` hit, `r50 = 1466`; `intent` `1007` / `1479`) và `shirt_r1` cũng khác
  (`conesep` 552 vs `intent` 531). Vậy giá trị trùng **chính là cùng một phân số `420/2017`**, không
  phải một ô bị chép: R@1 là số đếm nguyên thô, hai model BLIP-2 họ hàng trên cùng gallery đạt đúng
  420 query ở top-1 là trùng hợp bình thường. Không còn nghi vấn dữ liệu.

Trạng thái: **CONFIRMED** split/mask/max-pool/units/normalize-comment/`dress_r1`.
Không còn mục `UNVERIFIED` trong ca này.

## M-10b · HABIT — `habit` (`fiq_n02`)

- **Upstream:** `https://github.com/iLearn-Lab/AAAI26-HABIT` @ `6ddc2cd63eac1f65693f4e821c34fb3b42aee306`;
  vendor **byte-identical**.
- **Workbench path:** `third_party/HABIT/test.py`, `datasets.py`, `data_utils.py`; adapter
  `workbench/composition/model_habit.py`; measurement `cohort/measurements/habit_fiq_n02.json`.
- **Checkpoint:** `HABIT-FIQ_N0.2.pt` `814d2d0c…` (khớp `provenance.checkpoint_sha256`).

- **Split:** `datasets.py:28` `split='original-split'` default; launcher không truyền `split`
  ⇒ nhánh gốc (`:205-217`). **CONFIRMED.**
- **Mask + argsort:** `test.py:151` gán `-10e10` cho reference, **rồi** `:154`
  `np.argsort(-sims[i, :])[:50]` ⇒ **giảm dần trên điểm đã mask**. Khác CLIP4Cir (tăng dần trên
  `1-cos`): cùng kết quả, khác biểu thức.
- **Điểm gộp:** `:141-143` `np.matmul → squeeze → max(-1)`; normalize `:133-137` **bị comment out**
  (đọc trực tiếp). ⇒ không phải cosine. **CONFIRMED.**
- **Đơn vị:** phần trăm, `test.py:160-170` `100*r/len(nn_result)`, k∈{1,10,50}.
  Số: `54.87332239571265 / 74.50337396905957 / 64.68834818238611`; R@1 có mặt
  (`dress` 21.2692117…).

### Saved-vector probe — nó chứng minh gì, và KHÔNG chứng minh gì

Probe: `workbench/scripts/audit/equivalence_fastpath.py` neo trên **artifact lưu sẵn** của run
`497c25c57fbeba885ecfb9d1`, đọc `work/habit/features/gallery_global.npy` shape
**`(74381, 32, 256)`** float32 và `query_multimodal_global.npy` `(6003, 256)`, rồi tính lại
`score_old = np.einsum("nd,gqd->ngq", q, t).max(axis=-1)` cho mẫu ngẫu nhiên và so với
`ranks/habit_fiq_n02.json`. Kết quả trong
`docs/reviews/20261108T153045-audit-equivalence/fastpath_vs_stored.json` →
`anchor_verification`: **12 mẫu / condition, 0 mismatch, cả 3 condition (36/18 009 dòng)**.

**Chứng minh:** (a) biểu thức scorer của adapter `np.einsum("nd,gqd->ngq", …).max(axis=-1)`
(`composition/model_habit.py:258`) tái tạo **đúng** thứ hạng đã lưu trên vector đã lưu; (b) artifact
ranks tự nhất quán với features của chính nó; (c) max-pool theo token + inner product (không
normalize) là ngữ nghĩa đang chạy, không phải cosine.

**KHÔNG chứng minh:** (1) đường production sinh ra các vector đó **hôm nay** — gate
`G2 forward features` trong `EQUIVALENCE.md` ghi **`NOT_RUN`** ("GPU held by the live `intent`
cohort; pixel equality does not imply feature equality"), và sidecar `gallery_global.npy.json` chỉ
có `dtype`/`shape`/`fingerprint` của file, không có identity của checkpoint/transform đã sinh nó;
(2) độ phủ toàn cohort — chỉ 36/18 009 dòng; (3) đường **native**: gallery của probe là
74 381 ảnh (view của CIRCUS composition), không phải gallery full-split 3817/6346/5373 của
`fashioniq_full_gallery_ref_excluded` mà dòng native `habit` phục vụ ⇒ probe này **không** là bằng
chứng cho số 54.87.
Trạng thái: **CONFIRMED (phạm vi hẹp, đúng như trên)**; **không** được đọc thành "production đã
đúng" (đúng như cảnh báo của master doc).

## M-10c · INTENT — `intent` (`fiq_n02`)

- **Upstream:** `https://github.com/iLearn-Lab/AAAI26-INTENT` @ `2a75fc6f57d1a036bf3167a7fe1d2cee072f51f3`;
  vendor **byte-identical**.
- **Workbench path:** `third_party/INTENT/test.py`, `train_INTENT.py`; wrapper
  `workbench/replay/intent.py`; measurement `cohort/measurements/intent_fiq_n02.json`.
- **Checkpoint:** `intent_fiq_0.2.pt` `2a504cad…` (khớp registry `expected_sha256`).

- **Ranking/mask: CONFIRMED bằng đọc trực tiếp.** `test.py:150`
  `sims[i, test_targets_id.index(t['source_img_id'])] = -10e10` (mask reference) rồi `:153`
  `nn_result = [np.argsort(-sims[i, :])[:50] for i in range(sims.shape[0])]` — giảm dần trên điểm đã
  mask, cùng dạng HABIT.
- **Điểm gộp:** `:141-143` `matmul → squeeze → max(-1)`; normalize `:134-137` **comment out** (đọc
  trực tiếp) ⇒ không phải cosine.
- **Đơn vị:** phần trăm; `54.86624335289025 / 75.43191079283865 / 65.14907707286446`.
- **Eval-from-checkpoint KHÔNG có trong source tác giả — CONFIRMED, và workbench tự bù.**
  `train_INTENT.py:380-384` chỉ `torch.save(current_model, …'best_model.pt')`; **không** có
  `torch.load`/`load_state_dict` trong file đó, và `test.py` cũng không có loader (grep rỗng). Đường
  chạy là wrapper của workbench: `replay/intent.py:147`
  `load_trusted_whole_model(model_path, variant["expected_sha256"], torch)` →
  `replay/common.py:45-57`, hàm này **bắt buộc** `expected_sha256` trùng và dùng
  `weights_only=False` chỉ sau khi khớp ⇒ không phải `strict=False`, không phải unpickle mù.
  Nhưng nó vẫn là **hành vi do workbench định nghĩa**, không phải của tác giả: parity của
  "load whole-model rồi gọi `test()`" chưa được đo trên GPU. Trạng thái: **CONFIRMED** (upstream
  không có loader) + **NEEDS_RUNTIME** (parity của đường tự bù).

## M-10d · HINT — `hint`

- **Upstream:** `https://github.com/iLearn-Lab/ICASSP26-HINT` @ `bec50b6c8c19111893b617979502b948d1cea5b2`;
  vendor **byte-identical**.
- **Workbench path:** `third_party/HINT/test.py`, `datasets.py`; measurements
  `cohort/measurements/hint_fashioniq.json` và `20261106T233409+0700/measurements/hint_doc_command.json`.
- **Checkpoint:** `fashioniq.pt` `bf9c72d4…` (registry `expected_sha256` khớp, `URL_VERIFIED`).

- **Protocol `fashioniq_val_split` = pair-union, reference loại. CONFIRMED và nhất quán.**
  `datasets.py:25-31` không có tham số `split`; gallery dựng từ `test_targets` = **hợp** reference +
  target của tập query (`:154-160`), đúng định nghĩa registry `models.yaml:23-26`
  ("Val Split (Pair-Union Gallery, Reference Excluded)"). Row native khai đúng
  `protocol_id: fashioniq_val_split`.
- **Điểm gộp:** `test.py:74-76` `matmul → squeeze → max(-1)`; normalize `:66-70` comment out;
  chia `temp` ở `:154` (một temperature của model, **không** phải cosine); mask `:162`;
  `argsort(-sims)` `:165`.
- **Đơn vị:** phần trăm (`:106` `100*r/len(nn_result)`); `59.80338494972219 / 80.13076097500765`.
- **Hai dòng cho một model:** `hint_fashioniq.json` và `hint_doc_command.json` có **431 khoá lá
  giống hệt**, chỉ khác `.timestamp` (`2026-11-06T15:23:57…` vs `2026-11-06T17:18:51…`) — diff
  từng khoá lá bằng Python, 1 khoá khác. ⇒ một lượt chạy bị ingest hai lần.

## M-10e · CSMCIR — `csmcir`, trạng thái tạm gác

- **Upstream:** `https://github.com/qzp2018/CSMCIR` @ `774f94e2076ff17ea91703a6239d2a08f0e1a44e`;
  vendor **2 file patch** (`workbench/patches/csmcir_models_init_compat.patch`,
  `csmcir_validate_visualization_opt_in.patch`), đã khai digest post-patch và cả hai khớp.
- **Checkpoint:** `fashioniq_tuned_clip_best.pt` `c8812071…`, 2 720 615 859 B — **có mặt và khớp**
  registry; không phải blocker.
- **Được hiển thị khác zero? `CONFIRMED` (về mặt UI).** `frontend/ModelInventory.tsx:16-19`
  `DEFERRED_MODELS = { csmcir: "Tạm gác theo yêu cầu: thiếu dữ liệu phụ trợ", sprc: "Tạm gác theo
  yêu cầu: chưa tải checkpoint" }`; dòng có `status = "Tạm gác"` + lý do, **không** in 0; bộ lọc
  `deferred` tách riêng khỏi `measured`/`pending` (`:42-44`, `:71-73`). Nhưng lưu ý: cùng lúc
  `/api/native-measurements` **có** số của `csmcir` (57.0507) ⇒ trạng thái "tạm gác" và một số đã
  đo cùng tồn tại; đó là chủ ý (deferred là quyết định phạm vi, không phải kết luận đo).
- **Blocker `BLOCKED_MISSING_AUTHOR_CAPTION_ASSETS` — REFUTED bằng thực thi.**
  `cohort/measurements/csmcir_blocker.json` khai thiếu
  `COT_ours2/bert_captions/cap.*.val.json` (do `src/data_utils_csmcir.py:223`) và
  `data/FashionIQ/captions/keywords_in_mods_*.json`. Nhưng probe bắt `builtins.open` trên **đúng
  call path** của evaluator (`20261107T021500+0700/csmcir/call_path_probe.txt`):
  `reads_bert_captions: []`, `reads_keywords_in_mods: []`, `imageedit_instantiated: false`; 12 file
  thực mở chỉ gồm `COT_ours2/fashioniq/*_cot_val.json`, `fashionIQ_dataset/captions/cap.*.val.json`,
  `image_splits/split.*.val.json`, `qwen_captions/*_cot_val.json`. Ba file `COT_ours2` **có trong
  git** (`git ls-files`) và digest khớp bản ghi. ⇒ blocker đó là **artifact-only**, không chặn
  đường chạy; một lớp chết (`FashionIQDataset_imageedit`) mới là reader của `bert_captions`.
- **Số đã có và đã certify:** hai run `20261107T054343…f564570e` và `20261107T054933…f03887d2`,
  `57.05073873202006 / 77.29859352111816 / 67.17466612656911`, gate
  `PASS_WITHIN_5_PERCENT_VARIANT` (`variant: true`), `verification.state = "certified"`,
  `is_research_evidence: true`. Variant vì hai patch đã duyệt: `models_init_compat` (bản pin không
  import được chính nó) và `validate_visualization_opt_in`.
- **Hai run trùng:** cùng số, cùng checkpoint, **hai** `report_digest` khác nhau
  (`542b617a…`, `7d381b30…`), mỗi run `duplicate_status: "unique"` ⇒ một thí nghiệm bị đếm hai lần
  trong `/api/benchmark-runs` (13 run, 2 certified — cả hai là CSMCIR).
- **Vì sao vẫn ngoài đợt chạy mặc định:** quyết định phạm vi của người vận hành, cộng hai patch
  nguồn khiến kết quả là "approved compatibility variant", không phải parity chính thức; và
  `equivalence_to_historical_cot_input` trong `measurement.json` ghi **`NOT PROVEN`**.

## M-10f · SPRC — `sprc`, trạng thái tạm gác

- **Upstream:** `https://github.com/chunmeifeng/SPRC` @ `2935a5397732260d1db6fa577e5926f963e36f0f`;
  vendor **byte-identical**.
- **Hai variant checkpoint, và chỉ một có byte:**
  - `sprc_fiq.pt` (ViT-g, **khớp dòng paper**) — `status: CHECKPOINT_MAPPING_UNVERIFIED`,
    `expected_sha256: null`, không có byte trên đĩa (`models.yaml:1401-1417`, ghi chú "a
    filesystem-wide search found no `sprc_fiq.pt`").
  - `sprc_fiq_vitl.pt` (CLIP ViT-L/14, **scientific variant**) — `status: URL_VERIFIED`,
    `expected_sha256: 1938610427…`, **có trên đĩa, digest khớp** (tự tính), nạp strict 628/628
    (`models.yaml:1418-1448`).
- **Đã đo và có số:** `measurements/sprc_vitl_run1.json` macro
  `51.038571198781334 / 73.32633932431538 / 62.182455261548355`, khai
  `protocol: fashioniq_original_split`, `runtime.note: "save_memory=True index/target features
  staged on CPU; identical tensors"`. Row native `sprc`/`sprc_fiq_vitl` phục vụ đúng số này.
- **Không vào `/api/benchmark-runs`:** 0 run cho `sprc` (quét 13 run) ⇒ không có aggregate report,
  không gate, chỉ có dòng native chưa xác minh.
- **Lý do tạm gác, và một câu SAI trong UI:** registry để
  `reproduction_status: BLOCKED_CHECKPOINT_MAPPING` cho cả model; lý do thật là **không có
  checkpoint khớp cấu hình paper (ViT-g)**. Nhưng `ModelInventory.tsx:18` in
  "chưa tải checkpoint" — hàm ý không có checkpoint nào, trong khi `sprc_fiq_vitl.pt` đã tải, đã đo.
  Lý do đúng phải là "chưa có checkpoint ViT-g khớp paper; ViT-L đã đo nhưng không so được".
  Đồng thời `models.yaml:1447` ghi rõ kết quả ViT-L **không được** so với dòng paper
  (`reported_r10: 54.92`) — nên việc giữ nó ngoài đợt so sánh là đúng, chỉ nhãn là sai.
- **Đơn vị:** phần trăm; `gallery` 3817/6346/5373, `queries` 2017/2038/1961 trong artifact.

Trạng thái: **CONFIRMED** deferred-không-phải-zero; **CONFIRMED (defect chữ)** lý do deferred của
`sprc` mô tả sai cái gì đang thiếu; **CONFIRMED** digest + tính không so sánh được với paper.

---

## Roll-up và alias bị đếm như model độc lập

Đo trên `/api/native-measurements` (49 dòng): `counts.models_with_metrics = 15`,
`counts.models_rollup_only = 13`, `counts.rollup_rows = 22`, `rollup_sources` = 5 file. Nhưng 17
model trong registry; và tập "model" xuất hiện trong catalog bao gồm **tên không có trong registry**:

```
roll-up-only ids (13): B1/H_normal, B1/H_rev, B2/H_normal, B2/H_rev,
                       airknow_n05, airknow_n08, airknow_variant_80,
                       clip4cir_fullft, clip4cir_noft, clip4cir_rn50x4_fullft,
                       clvc, combiner_rn50x4_noft, sprc_vitl
không có trong registry.models: B1/H_*, B2/H_*, airknow_n05, airknow_n08, airknow_variant_80,
                                clip4cir_fullft, clip4cir_noft, clvc, sprc_vitl  (11/13)
```

Diễn giải từng nhóm — tất cả đều là **alias của model đã có**, trừ khi nói khác:

| nguồn (file) | entry | thực chất là |
|---|---|---|
| `clip4cir_parsed.json` | `clip4cir_fullft`, `clip4cir_noft` | cùng upstream CLIP4Cir, tên khác registry (`clip4cir_rn50x4_fullft`, `combiner_rn50x4_noft`); **hai cấu hình của một upstream, không được gộp** |
| `selection_decisions.json` | `clip4cir_rn50x4_fullft`, `combiner_rn50x4_noft`, `dcnet`, `clvc_net`, `tgcir`, `sprc_vitl`, `airknow`, `airknow_variant_80` | alias/dup của model đã đếm |
| `all_measured_summary.json` | `airknow_n05`, `airknow_n08`, `clvc`, `tgcir` | alias của `airknow` (`n05` = checkpoint được chọn, `n08` = variant 80 %), `clvc_net`, `tgcir` |
| `pair_compat_verdict.json` + `pair_compat/measurements/summary.json` | `B1/H_normal`, `B1/H_rev`, `B2/H_normal`, `B2/H_rev` | **4 cấu hình checker của MỘT model `pair`**; đồng thời bị ingest **hai lần** (mỗi entry xuất hiện ở cả hai file) |

- **Cơ chế đã đúng ở chỗ quan trọng:** dòng roll-up có `rollup: true`, `rollup_entry`, và
  `counts.with_metrics` / `models_with_metrics` **loại** chúng; response có `note` nói rõ
  "must not be read as a leaderboard position"; `EvaluationPage.tsx:607` in
  "roll-up: lặp lại số của file khác". ⇒ chúng **không** bị cộng vào số model đo.
- **Chỗ vẫn đếm như model — `models_rollup_only = 13`:** con số này được phục vụ và được đọc như
  "13 model chỉ có số trong roll-up", nhưng 11/13 không phải model (4 biến thể `pair`, 3 alias
  airknow, 2 tên CLIP4Cir, `clvc`, `sprc_vitl`). Muốn đếm đúng phải chuẩn hoá theo identity
  (`checkpoint_id` + upstream SHA), không theo khoá JSON.
- **File mang số mà UI đếm như model, nêu đích danh:** `selection_decisions.json` (8 entry, cả 8 là
  alias/dup) và `clip4cir_parsed.json` (2 entry, cả 2 là tên ngoài registry). Cả hai nằm trong
  `rollup_sources` của API.
- **Hệ quả tới con số 15/17 của chiến dịch:** `counts.models_with_metrics = 15` **không** phải "15
  model": trong 15 id đó có `csmcir` (đang deferred) và 4 biến thể `pair`
  (`pair_b1_H_normal`, `pair_b1_H_rev`, `pair_b2_H_normal`, `pair_b2_H_rev` — mỗi file một dòng,
  cùng `model_id: "pair"`), tức số **model** thật nhỏ hơn 15; cùng lúc 3 model registry
  (`combiner_rn50x4_noft`, `clip4cir_rn50x4_fullft`, và `pair` khi tính theo id registry) chỉ có mặt
  qua roll-up. Con số "15 model có số đo" **không** được dùng làm mốc bàn giao.
- **Không nơi nào gộp FullFT với NoFT:** tách theo `checkpoint_id`/tên key, đúng yêu cầu. Cũng không
  có chỗ nào lấy `clip4cir_fullft` 43.8182 để so với `combiner_rn50x4_noft` 35.1407 như hai model
  khác nhau — nhưng cả hai **đều là alias ngoài registry**, nên một roll-up chứa chúng (=
  `clip4cir_parsed.json`) là file cần đổi tên key trước khi đếm.

---

## Kết luận ngắn

| ca | trạng thái | điều còn mở |
|---|---|---|
| M-08a CLIP4Cir FullFT | CONFIRMED (protocol/rank/unit); **CONFIRMED defect** | roll-up 43.8182 **không có artifact**; `categories[].protocol` vẫn `ref_excluded` |
| M-08b Combiner NoFT | CONFIRMED | `METRIC_REFERENCE_UNVERIFIED` là **đúng**; số roll-up 35.1407 ≠ run 35.1078 |
| M-09a DCNet | CONFIRMED | nhánh `comb`; API phục vụ %, file phân số ngoài root; suy unit theo độ lớn là hazard |
| M-09b TME | REFUTED (env var) | `WORKBENCH_TME_*_BS` là **no-op**; lượt ghi dùng 256/32; artifact không ghi batch |
| M-09c PTHA+MTST | CONFIRMED | nhãn transfer có ở API, **không** hiện ở UI; projection chỉ ghi trong artifact |
| M-10a ConeSep | CONFIRMED | `dress_r1` trùng INTENT = cùng phân số 420/2017, không phải copy |
| M-10b HABIT | CONFIRMED (hẹp) | saved-vector probe ≠ bằng chứng production; `G2 forward` NOT_RUN |
| M-10c INTENT | CONFIRMED + NEEDS_RUNTIME | upstream không có loader; wrapper workbench bù, parity chưa đo |
| M-10d HINT | CONFIRMED | pair-union đúng; một lượt bị ingest hai lần |
| M-10e CSMCIR | CONFIRMED (deferred hiển thị đúng) + **REFUTED blocker** | `bert_captions` không nằm trên call path; 2 run certified trùng |
| M-10f SPRC | CONFIRMED (deferred) + defect chữ | lý do UI "chưa tải checkpoint" sai: ViT-L đã tải và đã đo |

**Defect hệ quả lớn nhất:** hai trong ba con số CLIP4Cir đang được phục vụ (43.8182 và 35.1407)
đến từ **roll-up của một lượt chạy không còn artifact nào trong cây** — artifact native của cùng
thí nghiệm ghi 43.8016931215922 / 66.98358456293742 — và chúng được phục vụ dưới **hai tên
CLIP4Cir khác nhau**, tức vừa là số không tái lập được, vừa là alias có thể bị đếm thành model.
Sửa: hoặc ingest lại lượt đã sinh 43.8182, hoặc bỏ roll-up đó khỏi catalog và chỉ phục vụ dòng có
artifact; đồng thời migrate `categories[].protocol` của artifact FullFT.

**Việc chưa làm (và vì sao):** `UNVERIFIED` — chỉ còn (1) chưa chạy lại TME với
`WORKBENCH_TME_QFORMER_BS=64` / `WORKBENCH_TME_QUERY_BS=8` sau commit `98c0eeb` để chứng minh
override mới thật sự có tác dụng — cần GPU, ngoài phạm vi "No GPU work"; (2) parity của đường
load-whole-model do workbench tự bù cho INTENT (`replay/intent.py:147`) chưa được đo — cũng cần
GPU. Hai mục đã đóng trong đợt này và không còn treo: block normalize của ConeSep (đọc trực
tiếp `test.py:123-127`, comment-out, thuộc nhánh CIRR) và dòng mask của INTENT (đọc trực tiếp
`test.py:150`).
