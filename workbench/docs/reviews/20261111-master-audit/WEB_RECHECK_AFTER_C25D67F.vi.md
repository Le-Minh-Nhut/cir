# WEB RECHECK sau `c25d67f` — 2026-11-11 (chỉ đọc, service thật)

Repo `/home/heheboiz/data/cir`, nhánh `research/cir-failure-workbench`.
Frontend `http://127.0.0.1:5173`, API `http://127.0.0.1:8000`. Viewport **2560×1440, deviceScaleFactor 1**, ảnh chụp full-page đọc lại ở kích thước gốc.
Không restart, không kill, không build, không sửa config, không ghi gì vào backend. Mọi request tới API trong lượt này là **GET**.

## Bundle mà toàn bộ bằng chứng này mô tả

| mục | giá trị |
|---|---|
| bundle phục vụ | **`assets/index-CBR6lHgq.js`** |
| sha256 | **`acfb5fbbced77e88b86ee542e5f261c6df71c0e0e8039d753965e1fc873ac21a`** |
| đối chứng | `curl -s http://127.0.0.1:5173/ \| grep -o 'assets/index-[^"]*\.js'` → `assets/index-CBR6lHgq.js`; `sha256sum workbench/frontend/dist/assets/index-CBR6lHgq.js` → `acfb5fbb…c21a` → **khớp** |
| `dist/assets` | chứa **đúng** `index-CBR6lHgq.js` + `index-DecQnta9.css` (2 tệp, không tệp lạ) |
| đo lại tự động | `recheck_probe.py:bundle_check()` đọc lại cả 3 điều trên ở **mỗi** lần chạy → một lần rebuild sau đó làm bằng chứng này mất hiệu lực |

Mọi kết luận dưới đây chỉ đúng cho digest `acfb5fbb…c21a`.

## Cách chạy lại

```
workbench/envs/ctl312h/bin/python \
  workbench/docs/reviews/20261111-master-audit/web_evidence/e2e_recheck/recheck_probe.py
```

Kết quả máy đọc được: `web_evidence/e2e_recheck/recheck_results.json` (mỗi check: URL, DOM excerpt, expectation, đúng/sai, tên ảnh, các request API đã dùng). Lượt chạy ghi trong báo cáo: **20/21 PASS**, chạy 79,36 s.

## Bảng A–E

| # | điều phải chứng minh | kết quả | ảnh | API dùng để đối chiếu số |
|---|---|---|---|---|
| **A** | URL lỗi #1 cũ giờ phải có TME (hoặc cả 2 lượt đã chọn) làm cột so sánh, và picker cho chọn peer tùy ý | **PASS** | `A1_compare_before_single.png`, `A2_compare_after_two_selected.png`, `A3_compare_arbitrary_peer.png`, `A4_compare_peer_clear.png`, `A5_compare_two_runs_one_model.png` | `GET /api/composition/runs?subset=full` (19 dòng cùng protocol+subset → picker có 19 lựa chọn); `GET /api/composition/runs/cd1a696d…/queries?subset=full&condition=multimodal&query_id=7%3A1242&top_k=10&limit=1&offset=0` → `ranks.multimodal=65`; cùng URL cho `f461f391…` → `98` |
| **B** | Nhãn run phân biệt được 2×`hint / fashioniq` và 2×`tme / fashioniq_0`, và nhãn dự đoán được lượt nào có ứng viên; mở `ad477312…` khớp nhãn | **PASS** | `B1_run_labels_distinct.png`, `B1_run_labels.txt`, `B2_noexport_run_matches_label.png`, `B3_sibling_run_has_candidates.png` | `GET /api/composition/runs/ad477312aa57c1b9114f9e51/queries?subset=full&condition=multimodal&query_id=7%3A1242&top_k=10&limit=1&offset=0` → `candidates.status=TOPK_EXPORT_NOT_REQUESTED`, `reason="no candidate export directory for this run"`, `depth=null`, `candidates_available=0` |
| **C** | Các bước đã xác minh trước đó không bị fix làm hỏng | **PASS** | `C1_reference_caption_target.png`, `C2_MM_top10.png`, `C2_MM_top50.png`, `C2_Text-only_top10.png`, `C2_Text-only_top50.png`, `C2_Image-only_top10.png`, `C2_Image-only_top50.png`, `C3_three_conditions_differ.png`, `C4_bad_qid_no_results.png` | `GET /api/composition/runs/f461f391…/queries?subset=full&condition={multimodal\|text_only_black_image\|image_only_empty_text}&query_id=7%3A1242&top_k={10\|50}&limit=1&offset=0` → 10/50 dòng, id+thứ tự khớp DOM; `ranks={multimodal:98, text:226, image:2370}`; 3 điều kiện, ứng viên đầu: MM `B009CCA04I.jpg`, Text `B00CSY2XWO.jpg`, Image `B00CQAJI7S.jpg` |
| **D** | Dòng CSMCIR hiện câu "FashionIQ native đã đo · CIRCUS composition tạm gác" + link tới kết quả native | **PASS** | `D1_csmcir_row_text.png`, `D2_csmcir_native_link_opens_row.png` | `GET /api/native-measurements` → dòng `protocol_id=fashioniq_original_split`, `measurement_id=6812926dbe6394ce`, `r10=57.0507`, `r50=77.2986` (khớp `57.05% / 77.30%` trên UI) |
| **E** | Cột dẫn xuất/lineage hiện cho lượt LIMN dẫn xuất; subset Validated đủ **586/586** trong khi parent vẫn partial | **PASS** | `E1_derived_lineage_validated.png`, `E2_parent_stays_partial.png`, `F1_default_subset_landing.png` | `GET /api/composition/runs?subset=validated` → `aee2e8d3…` có `derived_from.own_scored_count=586`, `own_unscored_count=0`, `own_status_label=DERIVED_COMPLETE`, `parent_run_id=9e4918a6b5c243412cbc58c9`, `execution_status=PARTIAL` (kế thừa); `GET /api/composition/runs?subset=full` → `9e4918a6…` `execution_status=PARTIAL` |

## Chi tiết từng mục (chuỗi render thật)

### A — peer so sánh đã với tới được

*Trước* (URL lỗi #1 cũ, `runs=` chỉ 1 lượt): `hint / fashioniq (đang xem) #98`, `limn #193`, `conesep #193`, `ptha_mtst #1.235`, `intent #111` — đúng 4 peer `slice(0,4)` cũ. Picker **đã có** và liệt kê **19** lượt.
*Sau* (thêm `%2Cc%3Acd1a696d8bca0fc4bce6925b` vào `runs=`): `hint / fashioniq (đang xem) #98`, **`tme #65`**. Cohort line: `Cohort: 2 lượt, cùng protocol circus_fashioniq_release_v1 và subset full`.
Chọn peer tùy ý `a4f786d8…` (conesep) trong `[data-testid=compare-peer-select]` → bảng thành 3 cột `hint #98 / tme #65 / conesep #193`, URL đổi **chỉ** `runs=` (`run=`, `query=`, `protocol=`, `subset=` giữ nguyên) — đúng hợp đồng `changePeers`.
`[data-testid=compare-peer-clear]` → `runs=` còn đúng lượt đang xem, peer vừa bỏ chọn lại được, `query=7%3A1242` không mất.

### B — nhãn run phân biệt và dự đoán được

`#run-select` (nguyên văn, `B1_run_labels.txt`):

```
hint / fashioniq · 4bc9893a · Top-K 50
hint / fashioniq · f461f391 · Top-K 100
tme / fashioniq_0 · ad477312 · không có Top-K
tme / fashioniq_0 · cd1a696d · Top-K 100
```

`ad477312…` (`tme / fashioniq_0 · ad477312 · không có Top-K`) mở ra: `data-empty-kind=capability`, tiêu đề `Lượt này chưa có danh sách ứng viên`, chi tiết mở rộng `no candidate export directory for this run` — khớp đúng `candidate_status` API. Lượt anh em `cd1a696d…` (`Top-K 100`) có **10/10** ứng viên. Nhãn dự đoán đúng trước khi bấm.

### C — không hỏng bước đã xác minh

- `7:1242` / HINT: caption `Is shorter skirt with brighter colors and is colorful with sleeves.`, ảnh đích `B00B79JENC.jpg` trong `[data-testid=query-targets]`, hạng MM **#98**; `[data-testid=target-outside-window]`: `Ảnh đích ở hạng #98 theo điều kiện Ảnh + văn bản (MM) — ngoài cửa sổ Top-10 đang hiện (10 ứng viên được serve, export lưu 100 dòng)…`. Đích **không** có trong 10 ứng viên đang hiện.
- Top-10 → Top-50 (MM): 10 rồi **50** dòng, id + thứ tự khớp API, thumbnail `50/50` tải thật. Text-only 10/10, Image-only 10/10, tất cả `ids_and_order_match=true`.
- Ba điều kiện → ba thứ tự khác nhau, khớp API ở cả 3 (ứng viên đầu MM `B009CCA04I.jpg`, Text `B00CSY2XWO.jpg`, Image `B00CQAJI7S.jpg`).
- `7:99999` → `data-empty-kind=no-results`, câu `Không tìm thấy đúng query 7:99999 ở lượt này hoặc subset đang chọn. Không tự thay bằng query khác.`, **0** `[data-testid=query-inspector]`, **0** nút thử lại.

### D — CSMCIR (chuỗi chính xác)

Trong `[data-testid=model-row-csmcir]`, `[data-testid=deferred-csmcir]` render **nguyên văn**:

```
FashionIQ native đã đo · CIRCUS composition tạm gác
```

kèm link `[data-testid=deferred-native-csmcir-fashioniq_original_split]`:

```
Mở số đo native · FashionIQ — Original Split (Full Gallery) · 6
```

Bấm link → `?area=evaluation&view=native&protocol=fashioniq_original_split&subset=full`, bảng native 6 dòng, dòng `native-row-6812926dbe6394ce` hiện `57.05% / 77.30%` (csmcir, `measurement.json`, `unit=percent`, `inferred_from_magnitude`, `verified=0`).

### E — lineage dẫn xuất

Ở subset **Validated**, `[data-testid=derived-lineage-aee2e8d39214d877a84d7c08]` render:

```
DERIVED  Lượt dẫn xuất từ 9e4918a6 (subset validated): tái tổ hợp hạng của parent, không chạy lại GPU —
chính subset này: 586/586 đã chấm · DERIVED_COMPLETE. execution_status là của parent, không phải của subset này.
```

Badge tone `ok`. Ở subset **Full**, dòng parent `9e4918a6…` vẫn `PARTIAL` `6,001 / 6,003` và **không** mang lineage (không có `[data-testid=derived-lineage-9e4918a6…]`). Hai trạng thái đọc được cùng lúc, đúng R09.

## Còn sai

### 1. Bảng so sánh vẫn đặt nhãn peer bằng `model_id` trần — hai lượt cùng model không phân biệt được (MỚI, chưa sửa)

Fix #2 đã sửa `#run-select` và option của peer picker, nhưng **cột trong bảng** so sánh vẫn lấy `model_id` (`QueryInspector.tsx:519-521`: `peers.find(...)?.model_id ?? row.key`). Hệ quả: hai peer cùng model `limn` hiện **y hệt** `limn#193` / `limn#193`, dù một lượt không có ứng viên và một lượt có.

Tái hiện:

```
http://127.0.0.1:5173/?area=samples&view=compare&protocol=circus_fashioniq_release_v1&subset=full
  &runs=c%3Af461f391c6d4696011be8067%2Cc%3A10173dc45dea8c90fd2a284b%2Cc%3A9e4918a6b5c243412cbc58c9
  &run=f461f391c6d4696011be8067&query=7%3A1242
```

DOM đọc được: `[["hint / fashioniq (đang xem)", "#98"], ["limn", "#193"], ["limn", "#193"]]` (`A5_compare_two_runs_one_model.png`, check `A5_compare_two_runs_one_model` = FAIL).
API: `10173dc45dea8c90fd2a284b` → `candidate_status=TOPK_EXPORT_NOT_REQUESTED`; `9e4918a6b5c243412cbc58c9` → `AVAILABLE`, `depth=100`. Cùng hạng `193` nên bảng không có tín hiệu nào khác để phân biệt.

Đúng ra cột phải dùng cùng hàm `compositionLabel(peer)` đã dùng cho picker (đã có sẵn, đã đúng), hoặc ít nhất thêm `checkpoint_id` như dòng `(đang xem)`.

### 2. (Không phải lỗi của fix này, còn nguyên) Mặc định subset — nay đã tự sửa

`routes.defaultLocation()` vẫn đặt `subset: "sf"`, nhưng `App.tsx` tự sửa scope khi URL không nêu (`chooseInitialScope`): vào `?area=samples&view=explorer` không có `subset=` → URL tự thành `subset=full` (`F1_default_subset_landing.png`). Lỗi #4 cũ **không còn** ở đường này; chỉ còn khác biệt là URL bị `replaceState` một lần.

### 3. Chưa chạm trong lượt này (trung thực là chưa xác minh)

- Lỗi #3 (lưu cohort trên dữ liệu CIRCUS) và lỗi #5 (thời gian nạp run list): **không** nằm trong phạm vi "đã sửa" của `c25d67f` nên lượt này không kiểm lại. Trạng thái của chúng vẫn là FAIL như `WEB_WORKFLOW_E2E_20261111.vi.md`.
- `ad477312…` ở subset **Validated/Full** có 2 dòng khác nhãn (`không có Top-K`); chưa thử subset `sf` cho run này.

## Tệp trong thư mục này

`recheck_probe.py` (harness, read-only), `recheck_results.json` (kết quả máy đọc), `B1_run_labels.txt` (nhãn `#run-select` nguyên văn), 22 ảnh `*.png` 2560×1440.
