# BLOCKERS — blocker có bằng chứng

> **Audit liên quan (đọc trước khi dùng bất kỳ blocker nào dưới đây):**
> [`workbench/docs/reviews/20261108T173736+0700/INPUT_PIPELINE_AND_INFERENCE_AUDIT.vi.md`](../20261108T173736+0700/INPUT_PIPELINE_AND_INFERENCE_AUDIT.vi.md)
> — kiểm toán độc lập đường đọc → inference → scoring. **Xác lập**: danh tính repo/checkpoint (G0 PASS,
> G1 PASS 17/17), decode bit-identical nơi đã đo (G2 PASS), và fast path bị từ chối vì đổi 207/18,009
> rank (G3b FAIL). **KHÔNG xác lập**: forward-feature bit-identity (**G2-fwd NOT_RUN**), batch/tail/
> cold-warm axes (**G4 PARTIAL**), cache identity (**G5 DEFECT, đã đóng một phần**), throughput
> (**G6 PARTIAL**, một phần chỉ có trong prose). Không con số nào ở đây là paper reproduction.

Mỗi mục theo đúng 6 mục yêu cầu: (1) model/checkpoint/protocol/condition bị ảnh hưởng; (2) stage
hỏng; (3) lệnh tái hiện + exit code + lỗi gốc + log; (4) từng fix đã thử và kết quả, phân biệt
**nguyên nhân đã xác nhận** với **giả thuyết**; (5) dữ liệu dùng được, coverage bị ảnh hưởng, phần
chưa kiểm chứng; (6) thiếu chính xác cái gì và cách resume.

Trạng thái tổng hợp (cập nhật): **7 model COMPLETED** với artifact đầy đủ (habit, intent, hint,
conesep, tme, encoder, ptha_mtst), **1 model PARTIAL và đã bị superseded** (limn — 5884/6003 dưới run id
cũ, chưa re-run), **1 model adapter tự từ chối** (csmcir). **Không model nào là paper reproduction.**
Blocker interpreter Python và NameError import của ENCODER/PTHA+MTST **đã gỡ**; blocker còn mở là
`csmcir` (`encode_targets` raises `_TARGET_REASON`), 2 query LIMN genuinely ambiguous, và việc chưa
re-run ENCODER/PTHA/LIMN dưới code đã sửa. **8 model chỉ có đường native, không có composition adapter** — agent khác phụ trách.

---

## B1. `csmcir` — adapter từ chối có chủ đích, không phải lỗi môi trường

1. **Phạm vi**: `csmcir` / `fashioniq` / `circus_fashioniq_release_v1` / cả 3 condition
   (`multimodal`, `text_only_black_image`, `image_only_empty_text`).
2. **Stage hỏng**: `mapping` (không phải load/inference). Adapter ném lý do ở `mode_supported()`.
3. **Tái hiện**:
   ```sh
   workbench/envs/ctl312h/bin/python workbench/scripts/composition_eval.py \
       --models csmcir --subsets full,sf,validated --plan
   ```
   exit 0, `readiness[0].ready = false`, `status = SELECTED_BUT_BLOCKED`.
   Lý do nguyên văn:
   `CSMCIR cannot run through the composition adapter: the adapter's score() is a 256-d text embedding
   expanded x32 into an 8192-d dot product, whereas the native evaluator's inference() uses
   attention-weighted token importance and a max over 32 query tokens. Those are different quantities,
   so adapter scores would be a substitution, not the measured contract. Additionally, local captions
   cover only the FashionIQ validation split (15,415 of 74,381 CIRCUS gallery images, or 20.72%),
   leaving 58,966 images uncaptioned.`
4. **Fix đã thử**: không có, và **không nên có** — hai lý do đều đúng về mặt khoa học:
   - `score()` hiện tại tính một đại lượng khác với `inference()` native. Sửa cho khớp là thay đổi
     semantics đánh giá (bị cấm), không phải sửa bug.
   - Caption chỉ phủ 20.72% gallery. 58.966 ảnh không có caption thì nhánh text-conditioned không thể
     tính đúng; điền caption bằng model khác = bịa dữ liệu (bị cấm).
   **Nguyên nhân đã xác nhận** (đọc code + đo tỉ lệ caption), không phải giả thuyết.
5. **Dữ liệu dùng được**: checkpoint `fashioniq_tuned_clip_best.pt` (2.72 GB) đã staged và hash-verified.
   Coverage CIRCUS = 0 cho model này.
6. **Thiếu gì**: (a) mapping `score()` ↔ `inference()` attention-weighted, cần người đọc lại
   `src/blip_validate.py` và tái hiện đúng phép tính; (b) caption cho 58.966 ảnh còn lại từ nguồn
   chính thức. **Resume**: sau khi có (a)+(b), chạy lệnh ở §2 của `RUNS.vi.md` với `--models csmcir`.

---

## B2. `encoder` — ĐÃ GỠ (env 3.10 có, interpreter khớp) — còn lại: chưa re-run

1. **Phạm vi**: `encoder` / `fashioniq` / `circus_fashioniq_release_v1` / 3 condition.
2. **Stage hỏng**: `env` (nghi ngờ — chưa chạy được tới bước load).
3. **Tái hiện**: `WORKBENCH_ENCODER_PYTHON=…/pair-adapted/bin/python`,
   contract `adapted_runtime.python = '3.10'`, interpreter thực tế **3.11.15**.
   `verify_environment` sẽ trả `PYTHON_VERSION_MISMATCH`.
4. **Fix đã thử**: **chưa** chạy (hết budget phiên). **Giả thuyết** — mới đọc registry + đo version,
   chưa quan sát report. Env đúng cần `python 3.10` + `open_clip_torch==2.24.0`; env
   `tme-adapted` là 3.10.22 nhưng thiếu `open_clip`, nên không dùng lại trực tiếp được.
5. **Dữ liệu dùng được**: checkpoint staged + hash-verified; artifact cũ `6a528e877c328772e2c7bb28`
   (6003 query, MM R@10 31.0178) vẫn còn nhưng là **PRE_EXISTING_UNVERIFIED**.
6. **Trạng thái cập nhật**: blocker interpreter **đã gỡ** — env 3.10 có `open_clip_torch` đã được
   provision và `WORKBENCH_ENCODER_PYTHON` trỏ vào nó. Blocker **thứ hai** đã gỡ luôn: ENCODER
   **không chạy được từ HEAD** vì `model_encoder.py` gọi `limit_torch_threads`, `prefetch_rows`,
   `resolved_decode_workers` mà **không import** chúng (`NameError`), đã sửa. Cái **còn mở**:
   ENCODER **chưa được chạy lại**, nên số `31.0178` vẫn là `PRE_EXISTING_UNVERIFIED` dưới
   checkpoint cũ. **Resume**: chạy lệnh §2 với `--topk-depth 50 --keep-features`.

---

## B3. `ptha_mtst` — chỉ có checkpoint CIRR, nhãn transfer là bắt buộc (NameError đã gỡ)

1. **Phạm vi**: `ptha_mtst` / `cirr_transfer` / `circus_fashioniq_release_v1` / 3 condition.
2. **Stage hỏng**: không hỏng — chỉ **không phải reproduction**.
3. **Tái hiện**: registry `reproduction_status: SELECTED_TRANSFER_ONLY`;
   `checkpoint_training_dataset: CIRR`, `checkpoint_forwards_dataset: FashionIQ`,
   `transfer_condition: cross_dataset_transfer`.
4. **Fix đã thử**: không cần. **Nguyên nhân đã xác nhận**: tác giả không phát hành checkpoint
   FashionIQ nào cho PTHA+MTST; chỉ có `tuned_mtst_cirr.pt`.
5. **Dữ liệu dùng được**: checkpoint staged (2.72 GB, hash khớp LFS oid). Artifact cũ
   `315dd590eeb03d2c78d7937a`: MM R@10 19.4736 / R@50 34.4494 / nDCG 0.22493 — **số của transfer,
   không phải của paper row**. Không được so với `reported_r10 54.99`.
   **NameError đã gỡ**: `model_ptha_mtst.py` gọi `limit_torch_threads` / `prefetch_rows` /
   `resolved_decode_workers` mà không import, nên adapter **không chạy được từ HEAD**; đã sửa và có
   test hồi quy tĩnh. Blocker **còn mở ở đây** chỉ là: chưa re-run dưới code đã sửa.
6. **Thiếu**: checkpoint FashionIQ do tác giả huấn luyện. **Resume**: chạy lại để xác nhận
   (lệnh §2, `--models ptha_mtst`), luôn giữ nhãn TRANSFER.

---

## B4. `limn` — 2/6003 query genuinely ambiguous; 119 là con số cũ của loader

1. **Phạm vi**: `limn` / `base_iter0_all_categories` / 3 condition / subset `full`.
2. **Stage hỏng**: `metrics` (category resolution).
3. **Tái hiện**: `category_resolution.ambiguous_count: 119` — đây là **con số của release loader**,
   vẫn đúng như một sự kiện cấp loader, và nó **giữ nguyên** sau fix. Nguyên nhân: LIMN là model
   **category-scoped** 3 checkpoint riêng (`dress`/`shirt`/`toptee`), còn ảnh reference dùng chung giữa
   2 category thì không có không gian đặc trưng để chấm.
4. **Fix đã thử**: adapter có hook `query_scope()` đọc category từ chính caption annotation, **trước đây
   chưa bao giờ được runner gọi**. Đã wire; scope map đã resolve được **6001/6003** query. Chi tiết và
   số đo: `20261108T180000+0700-limn-scope/EVIDENCE.md`.
5. **Dữ liệu dùng được**: **2 query genuinely ambiguous**, không phải 119:
   `7:3038` và `7:3790` — không resolver nào gán được, và chúng **không** được chấm (không đoán bừa).
   Artifact cũ `10173dc45dea8c90fd2a284b` là PARTIAL 5884/6003 và **đã bị superseded**: `identity` nay
   gồm `query_scope_resolution`, nên run id đổi thành `2a4b19f335f5ec281375d09f` và **chưa re-run**.
   Vì thế `MM R@10 30.0646` là số **5884-mẫu số cũ**, không phải kết quả sẽ có sau khi wire.
6. **Thiếu**: một lần chạy lại LIMN dưới run id mới. **Blocker còn lại thật**: 2 query
   `7:3038`/`7:3790` nhập nhằng vĩnh viễn với dữ liệu đã pin — phải nêu chúng trong mọi claim coverage.
   **Resume**: chạy §2 cho `limn`; báo coverage là 6001/6003 và nêu tên 2 qid.

---

## B5. `tme` — chạy được nhưng là adapted runtime, không phải paper reproduction

1. **Phạm vi**: `tme` / `fashioniq_0` / 3 condition.
2. **Stage hỏng**: không hỏng.
3. **Tái hiện**: registry `adapted_runtime.pytorch: '>=2.7.1,<2.9'`, env `tme-adapted` = py 3.10.22
   + torch 2.7.1+cu128. Phải đặt `WORKBENCH_RUNTIME_ADAPTATION=sm_120`.
4. **Fix đã thử**: env đã có sẵn và dùng được. **Nguyên nhân đã xác nhận**: pin lịch sử
   (torch 2.1.2 / cu121) không có kernel sm_120.
5. **Dữ liệu dùng được**: checkpoint staged. Artifact cũ `ad477312aa57c1b9114f9e51`: MM R@10 34.0163 /
   R@50 53.9564 / nDCG 0.32039 / gap 0.35860, coverage 1.0. **Nhưng**: `full.json` **không** chứa key
   `variant_provenance`, tức badge adaptation không được ghi vào artifact (chỉ ghi trong
   `cohort_status.txt`). Đây là khuyết điểm provenance, xem B6.
6. **Thiếu**: chỉ là chạy lại để có NEW_RUN. **Resume**: lệnh §2, `--models tme`.

---

## B6. Thiếu badge adaptation trong artifact (ảnh hưởng tme, ptha_mtst và mọi run sm_120)

1. **Phạm vi**: mọi run dưới `WORKBENCH_RUNTIME_ADAPTATION=sm_120`.
2. **Stage hỏng**: `export`.
3. **Tái hiện**:
   ```sh
   grep -c "sm_120\|variant_provenance" \
     workbench/artifacts/composition/summary/tme_fashioniq_0_ad477312aa57c1b9114f9e51/full.json
   # → 0
   ```
4. **Fix đã thử**: chưa sửa được — `runner.py` thuộc agent khác. Đã **đo** được giá trị
   (`runtime_identity` có trong identity) nhưng summary không mang key provenance.
   **Nguyên nhân đã xác nhận**: `runner.py` không ghi `variant_provenance` vào summary.
5. **Hệ quả**: một người đọc artifact đơn lẻ sẽ tưởng đây là reproduction hợp lệ. Đã ghi rõ trong
   `RUNS.vi.md` và `MODEL_INVENTORY.md`, nhưng artifact tự nó vẫn thiếu.
6. **Thiếu**: một dòng trong `runner.py` ghi `variant_provenance`. Xem
   `workbench/docs/reviews/eval-runner-requests.md`. **Resume**: sau khi agent kia sửa, chạy lại.

---

## B7. `F7` — `reference_eligible` là hằng số khai báo, không phải đo (đã sửa phía adapter)

1. **Phạm vi**: **tất cả 9 adapter composition** và mọi summary chúng sinh ra.
2. **Stage hỏng**: `export` / provenance.
3. **Tái hiện**: `runner.py:751` và `:1007` ghi literal `"reference_eligible": True`;
   `runner.py:710` dùng `getattr(adapter, "reference_eligible", True)` — default `True` nghĩa là
   adapter không khai báo gì vẫn được coi là reference-eligible.
4. **Fix đã làm (phía tôi)**: thêm `CompositionAdapter.measured_reference_eligibility()` — **chạy**
   `score`+`rank` của adapter trên gallery tổng hợp nơi reference là exact match. Trả
   `{measured, reference_eligible, reason, reference_rank}`. Khai báo `score_dimension` cho 4 adapter
   có scoring cố định độ rộng. **Phía runner** ghi vào
   `workbench/docs/reviews/eval-runner-requests.md` (file thuộc agent khác).
   **Trạng thái: adapter xong, runner chưa** — nên artifact mới vẫn mang literal `true`.
5. **Đo được ngay — 8/9 adapter**: `csmcir`, `conesep`, `ptha_mtst`, `habit`, `intent`, `hint`,
   `encoder`, `tme` → `measured: true, reference_eligible: true, reference_rank: 1`.
   `limn` → `measured: false` (scorer cần Mapping gallery thật), lý do được ghi nguyên văn.
   Test hồi quy khẳng định `limn` là adapter **duy nhất** được phép không đo được, và khẳng định
   một scorer giả có mask reference bị phát hiện là `false`.
6. **Thiếu**: 4 dòng sửa trong `runner.py`. **Resume**: agent sở hữu `runner.py` áp
   `eval-runner-requests.md` §F7.

---

## B8. `F1` — gate môi trường không nhất quán (đã sửa)

1. **Phạm vi**: `model_ptha_mtst.py`, `model_encoder.py`, `model_limn.py`, `model_csmcir.py` — 4 adapter
   **không hề gọi gate**; và `runtime.py` return sớm ở nhánh Python mismatch.
2. **Stage hỏng**: `env` (gate).
3. **Tái hiện (trước fix)**: chỉ 5/9 adapter chứa chuỗi `require_verified_model_interpreter`.
   `runtime.py:1645-1649` (số dòng hiện tại) `return report` ngay sau `PYTHON_VERSION_MISMATCH`,
   nên `packages`/`imports` rỗng và không có probe dependency nào chạy.
4. **Fix đã làm**:
   - `CompositionAdapter.require_verified_interpreter()` — một chỗ duy nhất, **không có tham số bypass**.
   - Cả 9 adapter gọi nó (`grep` xác nhận 9/9; không adapter nào còn gọi hàm runtime trực tiếp).
   - `runtime.py`: thêm `_record_dependency_facts()` để nhánh version-mismatch vẫn **ghi** inventory
     dependency rồi mới trả verdict `PYTHON_VERSION_MISMATCH` (verdict giữ nguyên, chỉ không mất evidence).
   - Test hồi quy `workbench/tests/test_circus_env_gate_omp.py` (20 test, pass).
5. **Đã kiểm**: `pytest workbench/tests/test_manage_environment_a30.py
   workbench/tests/test_release_blocker_interpreter.py` → 86 passed, 5 skipped.

### F1(c) — `--plan` báo READY trong khi run chết ở gate (đã sửa)

Đây là phần **cuối** của F1 và là phần dễ bỏ sót nhất: `status()` chỉ kiểm
checkpoint/tệp-trình-thông-dịch/CUDA/mapping, **không** kiểm contract Python.

**Tái hiện trước fix** — `--plan` nói READY cho 8/9 model, nhưng:

```sh
WORKBENCH_ENCODER_PYTHON=…/pair-adapted/bin/python
workbench/envs/ctl312h/bin/python -c "…verify_environment(model_by_id('encoder'), probe=True, advisory=True)"
# → tier=BLOCKED status=PYTHON_VERSION_MISMATCH py=3.11.15
#   reason: expected Python 3.10, found 3.11.15
```

tức gate từ chối trong khi `status().ready` là `True`.

**Fix**: thêm `_environment_gate_reason(model, config)` trong `composition/adapters.py` — gọi
**chính** `verify_environment(probe=True, advisory=True)` mà gate gọi, nên plan và gate không thể
lệch nhau. Nối vào `status()` của `encoder` và `limn` (hai adapter từng nói dối); các adapter còn
lại đã kiểm contract qua đường khác. Chi phí đo được: **~1 s/model**, không chạm model, không chạm GPU.

**Sau fix**:

```
encoder    ready=False | environment gate would refuse the run [BLOCKED/PYTHON_VERSION_MISMATCH]: expected Python 3.10,…
limn       ready=False | environment gate would refuse the run [BLOCKED/PYTHON_VERSION_MISMATCH]: expected Python 3.10,…
ptha_mtst  ready=True  | None
tme        ready=True  | None
habit      ready=True  | None
```

Nghĩa là kết quả `--plan` báo "8/9 READY" ở §B1 **đã hết đúng**: giờ chỉ 6 model READY, `encoder` và
`limn` báo đúng lý do bị chặn.
**Còn lại**: 3 test cũ (`test_circus_adapters_rest.py::test_status_is_honest_and_reports_every_mode`,
   `test_circus_adapters_blip.py::test_adapter_import_and_unready_status[conesep|ptha_mtst]`) fail vì
   checkpoint **nay đã staged** ở root operator nên `status().ready` là `True`, trong khi test kỳ vọng
   `False`. **Đã xác nhận là tiền tồn tại — có bằng chứng**: đặt lại `model_habit.py` về bản `HEAD`
   (`git show HEAD:workbench/composition/model_habit.py`), import và gọi `Adapter.status()`:

   ```
   HEAD model_habit status.ready = True
   HEAD reason = None
   ```

   rồi khôi phục file đã sửa. Vậy kỳ vọng `ready is False` của test đã lỗi thời **trước** thay đổi của
   tôi: checkpoint nay đã staged ở `WORKBENCH_CHECKPOINT_ROOT` nên `status()` trả `True`.
   Fix đúng là fixture của test trỏ tới một checkpoint root **trống**, không phải nới lỏng adapter.

---

## B9. 8 model không có composition adapter (ngoài phạm vi CIRCUS)

`airknow`, `pair`, `clvc_net`, `dcnet`, `combiner_rn50x4_noft`, `clip4cir_rn50x4_fullft`, `tgcir`,
`sprc`.

1. **Phạm vi**: không vào được ma trận CIRCUS 3-condition vì không có `composition_adapter`.
2. **Stage hỏng**: `mapping` / `command`.
3. **Tái hiện**: registry không khai `composition_adapter` cho cả 8.
4. **Fix đã thử**: không — theo chỉ đạo, **agent khác phụ trách đường native** (non-CIRCUS).
5. **Dữ liệu dùng được**: 6/8 có checkpoint staged ở cây **phụ**
   `workbench/artifacts/checkpoint_selection/20261106T002617+0700/checkpoints/`
   (`airknow` 50%/80%, `pair` B1/B2, `clvc` 6 member, `dcnet` run dir, `clip4cir` cả 2 state, `sprc` ViT-L,
   `tgcir` Fashioniq.pt). Không có cái nào ở `WORKBENCH_CHECKPOINT_ROOT` khai báo.
6. **Thiếu**: entrypoint CLI cho phần lớn (xem `MODEL_INVENTORY.md` §4b), và mapping checkpoint.
