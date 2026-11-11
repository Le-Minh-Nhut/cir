# WEB WORKFLOW E2E — 2026-11-11 (read-only, service thật)

Repo `/home/heheboiz/data/cir`, nhánh `research/cir-failure-workbench`, HEAD lúc đo `a58b51db3529696519b2ce878ab302c5b0e3e837`.
Frontend `http://127.0.0.1:5173`, API `http://127.0.0.1:8000`. Viewport **2560×1440, deviceScaleFactor 1**.
Không restart, không kill, không build, không sửa code. Ghi chú này **chỉ đọc**.

## Bundle mà bằng chứng này mô tả

| mục | giá trị |
|---|---|
| bundle phục vụ **tại lượt đo này** | **`assets/index-BLH2yome.js`** |
| sha256 | **`702ed2e93a1bf86ba2094fe62791d4272dcdbf36e8caf7f757f08cf96805d681`** |
| đối chứng | `sha256(serve từ :5173)` == `sha256(workbench/frontend/dist/assets/index-BLH2yome.js)` → khớp |
| **hiệu lực** | một lần rebuild/restart sau đó **thay thế** bundle này; mọi kết luận dưới đây chỉ đúng cho digest trên |
| **trạng thái nay** | bundle trên **đã superseded**: frontend được build lại sau lượt này, và bundle **đang phục vụ bây giờ** là `index-CBR6lHgq.js` / `acfb5fbbced77e88b86ee542e5f261c6df71c0e0e8039d753965e1fc873ac21a`. Bảng PASS/FAIL dưới đây là **bản ghi lịch sử** của `index-BLH2yome.js` và **không** được đọc lại như kết quả đo trên bundle mới — muốn kết luận cho bundle hiện tại thì phải chạy lại harness trên nó |

## Phạm vi dữ liệu dùng làm chuẩn (không đoán)

| vai | run_id | model / checkpoint | subset | `candidate_topk.depth` |
|---|---|---|---|---|
| HINT | `f461f391c6d4696011be8067` | hint / fashioniq | full | 100 |
| TME | `cd1a696d8bca0fc4bce6925b` | tme / fashioniq_0 | full | 100 |

Query chung đã chấm: **`7:1242`** — rank MM **98** (HINT) / **65** (TME), ảnh đích `B00B79JENC.jpg`,
ảnh tham chiếu `B00CQAJI7S.jpg`; caption *"Is shorter skirt with brighter colors and is color…"*.
Đích nằm **ngoài Top-10 và Top-50**, chỉ vào cửa sổ ở Top-100 → dùng đúng cho bước 3.
`/api/images/{dress,shirt,toptee}/B00B79JENC.jpg` → 200 (thumbnail có thật để tải).

## Bảng bước 1–7

| # | bước | kết quả | bằng chứng |
|---|---|---|---|
| 1 | Khám phá phiên mới: tìm HINT + TME, mở run có sẵn **không đoán** protocol/subset | **PASS** | `e2e/s1_discovery_no_guessing.png`, `e2e/s1_discovery/01_landing.png`…`04_first_query_open.png`; URL sau khi chọn: `/?area=samples&view=explorer&protocol=circus_fashioniq_release_v1&subset=full&runs=c%3Af461f391…&run=f461f391…` |
| 2 | Cùng query đã chấm trên cả hai run: ảnh tham chiếu + caption + ảnh đích, đích **ngoài** cửa sổ ứng viên | **PASS** | `e2e/s2_same_query_hint.png`, `e2e/s2_same_query_tme.png`; API: `.../runs/f461f391…/queries?subset=full&condition=multimodal&query_id=7%3A1242` → `ranks.multimodal=98`, `.../runs/cd1a696d…` → `65`; DOM `[data-testid=query-targets] img` chứa `B00B79JENC.jpg` và `[data-testid=target-outside-window]` nói "ngoài cửa sổ Top-… đang hiện" |
| 3 | Top-10 → Top-50 → Top-10 × MM → Text-only → Image-only, đối chiếu số/id/thứ tự với API + `naturalWidth>0` | **PASS** | `e2e/s3_MM_top10.png`, `e2e/s3_MM_top50.png`, `e2e/s3_Text-only_top10.png`, `e2e/s3_Text-only_top50.png`, `e2e/s3_Image-only_top10.png`, `e2e/s3_Image-only_top50.png`, `e2e/s3_topk_and_condition_differ.png`. API cùng query: `top_k=10`→10 dòng, `top_k=50`→50 dòng, `top_k=100`→100 dòng (depth 100); 3 điều kiện trả 3 thứ tự khác nhau (MM `B009CCA04I…`, Text `B00CSY2XWO…`, Image `B00CQAJI7S…`). UI khớp id+thứ tự ở cả 9 trạng thái, mọi thumbnail `naturalWidth>0` |
| 4 | Search mã query · chip scored/unscored/hit/trượt · reset · pagination · đổi run nhanh · giữ context khi chuyển trang | **PASS** | `e2e/s4_query_id_search.png`, `s4_unscored_chip.png`, `s4_hit_chip.png`, `s4_miss_chip.png`, `s4_pagination.png`, `s4_context_survives_navigation.png`, `s4_rapid_run_switch.png`. Counter khớp API (`hit@10=1.970`, `unscored=0` trên `7:1242`/HINT full); pager `1–20 / 6.003` → `offset=20` trong URL → Back trả lại `1–20` và query đầu khác; 3 lần đổi run trong 0,4 s → DOM cuối là TME, không còn "hint" |
| 5 | Ba trạng thái phân biệt được: loading vs lỗi API vs rỗng thật (4xx/404) | **PASS** | `e2e/s5_loading.png` (`data-empty-kind=loading`), `e2e/s5_api_failure.png` + `s5_state_api_failure.png` (503 cưỡng bức ở riêng request ứng viên → `data-empty-kind=error`, `role=alert`, có `[data-testid=error-retry]`), `e2e/s5_empty_bad_qid.png` (`7:99999` → `no-results`, nói tên qid, "Không tự thay bằng query khác", **không** dựng inspector, **không** nút thử lại), `e2e/s5_404_response.png` / `s5_404_visible.png` (404 thật `composition_run_not_found` hiện nguyên văn + thử lại) |
| 6 | So sánh 2 model cùng query · mở kết quả từ Phân tích lỗi · lưu **một** ghi chú `e2e-check-20261111`, reload, kiểm đúng context | **FAIL (một phần)** | *PASS*: `s6_failure_analysis_to_inspector.png`, `s6_jumped_from_failures.png`, `s6_note_saved.png`, `s6_note_after_reload.png` — `PUT /api/annotations` → `GET /api/annotations` trả `{"query_id":"7:3143","protocol_id":"circus_fashioniq_release_v1","note":"e2e-check-20261111"}`, reload vẫn còn. *FAIL*: **không so sánh được HINT với TME** — `s6_compare_view.png`, `s6_compare_two_models.png` chỉ có `hint / fashioniq (đang xem) #98`, `limn #193`, `conesep #193`, `ptha_mtst #1.235`, `intent #111`. *FAIL*: cohort `e2e-check-20261111` không tạo được (xem lỗi #3) |
| 7 | Đo thời gian: điều hướng → danh sách run dùng được; `/api/models` cold vs warm | **FAIL (thiếu cold)** | `nav → #run-select có >1 option`: **0,87 s / 3,854 s / 1,731 s** (3 lần, `wait_until=domcontentloaded`) — lệnh: `workbench/envs/ctl312h/bin/python /tmp/final_probe.py`. `/api/models` **warm**: `for i in 1 2 3 4 5; do curl -sS -o /dev/null -w "models %{http_code} total=%{time_total}s size=%{size_download}\n" http://127.0.0.1:8000/api/models; done` → **3,205 / 2,288 / 2,208 / 2,211 / 2,092 ms**, 87.139 byte. **COLD: NOT_VERIFIED** — `_MODELS_CACHE` là cache toàn tiến trình (`workbench/backend/main.py:172-176`) và tôi bị cấm restart, nên nhiệt độ "cold" không tạo được trong lượt này |

## Bước chưa chạm tới (báo cáo trung thực là chưa xác minh)

- **7 (phần cold)**: chưa đo được `/api/models` cold — cần một lần restart của người khác; không tự tạo.
- **6 (phần cohort)**: chỉ ghi nhận UI từ chối; **không** tạo được cohort tên `e2e-check-20261111` trên lượt CIRCUS, nên "cohort lưu được" là chưa xác minh ở đường này (ghi chú thì đã lưu và đọc lại được).
- **3**: chưa thử run có export **mỏng hơn** yêu cầu trên UI (đã đo ở API: HINT `4bc9893a…` depth 50 → `top_k=100` serve **50**, tức UI sẽ báo "API serve 50"), nhưng chưa bấm Top-100 trên chính run đó.

## Lỗi thấy được (xếp theo mức cản nghiên cứu hằng ngày)

### 1. Bảng "So sánh mô hình" không bao giờ với tới TME (cản đúng việc của bước 6)
`QueryInspector.tsx` → `ComparePanel`: peer lấy `/api/composition/runs?subset=${run.subset}` rồi `.filter(protocol).filter(run_id≠).slice(0, 4)` — cắt 4 dòng **theo thứ tự API trả về**, nên 4 peer luôn là `limn, conesep, ptha_mtst, intent`, TME (và 13 run còn lại) không bao giờ xuất hiện.

Tái hiện: mở `/?area=samples&view=compare&protocol=circus_fashioniq_release_v1&subset=full&runs=c%3Af461f391c6d4696011be8067&run=f461f391c6d4696011be8067&query=7%3A1242` → bảng chỉ có `hint / fashioniq (đang xem) #98`, `limn #193`, `conesep #193`, `ptha_mtst #1.235`, `intent #111`. Không có ô chọn peer, không có "xem thêm".

Hệ quả: hai run đã chọn (`runs=` có 2 giá trị, header "Lượt chọn 2") **không** dùng để dựng cột so sánh; chọn nhiều run không mở rộng được bảng.

### 2. Hai run khác nhau mang nhãn y hệt nhau — chọn nhầm là mất luôn danh sách ứng viên
`#run-select` in `${model_id} / ${checkpoint_id}` (`SamplesPage.tsx`, `catalog.ts` `label`). Trong `subset=full` có **hai** dòng "hint / fashioniq" (`4bc9893a…`, `f461f391…`) và **hai** dòng "tme / fashioniq_0" (`ad477312…`, `cd1a696d…`). Chỉ một trong mỗi cặp có Top-K: `ad477312…` trả `candidate_status=TOPK_EXPORT_NOT_REQUESTED`, `depth=null`, 0 ứng viên.

Tái hiện: chọn dòng "tme / fashioniq_0" đầu tiên → khối ứng viên rỗng kèm lý do "no candidate export directory for this run"; đổi sang dòng thứ hai (nhãn giống hệt) → có ứng viên. Nhãn không có run id rút gọn, nên không phân biệt được trước khi bấm.

### 3. Không lưu được cohort trên dữ liệu CIRCUS (đường "lưu tập đã lọc" là ngõ cụt)
`FailuresPage.tsx` `saveCohort()` tự `throw` khi mọi run là `c:` ("API cohort chỉ nhận run của chỉ mục per-query…"). Và `/api/cohorts` (`main.py:1340`) validate qua `validate_analysis_selection` với `/api/runs` — endpoint này trả **`[]`** trên deployment hiện tại, nên ngay cả khi UI gửi đi cũng không có run hợp lệ.

Tái hiện: `/?area=failures&view=analytics&protocol=circus_fashioniq_release_v1&subset=full&runs=c%3Af461f391…%2Cc%3Acd1a696d…` → bấm `[data-testid=cohort-save]` ("Lưu tập đã lọc (…)") → `[data-testid=save-feedback][data-status=error]`: "Lưu thất bại: API cohort chỉ nhận run của chỉ mục per-query…"; `GET /api/cohorts` → `[]`. Ảnh `e2e/s6_cohort_save_final.png`.

### 4. Trang mặc định mở subset SF, trong khi dữ liệu người dùng cần lại ở Full
`routes.defaultLocation()` đặt `subset: "sf"`; vào `/` rồi "Khám phá mẫu" thì danh sách run là cohort SF. Muốn Full phải tự đổi ở `[data-testid=subset-select]`. Không mất dữ liệu, nhưng là một bước thừa mỗi phiên (và là lý do bước 1 cần 3 click thay vì 2).

### 5. Thời gian vào danh sách run dao động 4,4×
0,87 s → 3,854 s cho cùng một deep link, cùng bundle. Chưa tách được nguyên nhân (mạng nội bộ vs `/api/composition/runs` vs `/api/evidence/summary`); nêu ra vì 3,9 s là mức người dùng cảm nhận được.

## Đã đúng sẵn (không cần sửa)

- **Không thay query khác**: `7:99999` → khối `no-results` nói tên đúng qid, "Không tự thay bằng query khác", không dựng inspector (`selectExactQuery`, không phải `items[0]`).
- **Ba trạng thái rỗng tách bạch**: `data-empty-kind` = `loading` / `error` / `no-results`, lỗi có `role=alert` + `[data-testid=error-retry]`; lỗi hiện **trong vùng dữ liệu**, không phải khoảng trắng.
- **Top-K là request thật**: đổi Top-10 ↔ Top-50 tạo request mới (`candidateRequest.ts`), số dòng đổi thật (10 ↔ 50), id và thứ tự khớp API từng điều kiện; menu không mời số vượt depth.
- **Ảnh đích tách khỏi cửa sổ ứng viên**: `[data-testid=target-outside-window]` nêu hạng + điều kiện + "ngoài cửa sổ Top-N đang hiện" + số ứng viên được serve so với số dòng export — đúng cả khi hạng thay đổi theo điều kiện (MM #98, Text #226, Image #2.370 trên cùng query).
- **Thumbnail tải thật**: mọi `[data-testid=query-image]` có `naturalWidth > 0` ở 9 trạng thái Top-K × điều kiện; ảnh thiếu sẽ thành tile `data-asset-state=missing` có tên file, không phải ô trống.
- **Context sống qua điều hướng**: sang "Đánh giá" rồi về "Khám phá mẫu" vẫn giữ run + subset + query trong URL và trong inspector; đổi run liên tiếp 3 lần/0,4 s không để answer cũ vẽ đè (DOM cuối là TME).
- **Ghi chú gắn đúng danh tính**: lưu theo cặp `(query_id, protocol_id)`, reload vẫn còn, `GET /api/annotations` xác nhận; thông điệp lỗi của server được hiện nguyên văn, không bị nuốt.
- **Counter và chip trạng thái khớp API**: `hit@10=1.970`, `retrieval_failure=4.033`, `unscored=0` cho `7:1242`/HINT full; chip đổi tập và reset được.

## Cách chạy lại

```
workbench/envs/ctl312h/bin/python \
  workbench/docs/reviews/20261111-master-audit/web_evidence/e2e/e2e_probe.py
```
Harness dùng playwright + chromium có sẵn của repo (`workbench/envs/ctl312h/.playwright-browsers`), selector ưu tiên `data-testid`, mỗi check đối chiếu với `/api/composition/runs/{run}/queries?…` cho **đúng** run + subset + condition + K. Ghi chú `e2e-check-20261111` là bản ghi dùng một lần do chính lượt này tạo (`7:3143`), có thể xoá bằng `PUT /api/annotations` với `note: ""`.

## Ghi nhận sự cố service trong lúc đo

Trong lúc chạy, **một** lần kết nối bị đóng giữa chừng (`RemoteDisconnected` khi gọi `/api/models`) — đúng hiện tượng "agent khác restart service". Lần thử lại ngay sau đó: API 200, FE 200. Không restart gì từ phía tôi. Các lần đo trong báo cáo đều **sau** lần thử lại này và bundle digest vẫn là `702ed2e9…d681` (digest của
lượt đo này; **nay superseded** — xem bảng đầu báo cáo).

---

# PHỤ LỤC 2026-11-11 (lượt đo lại) — không viết lại các hàng gốc ở trên

Phần trên là **bản ghi lịch sử** của `index-BLH2yome.js` / `702ed2e9…d681`. Phần này là trạng thái **hiện tại**,
đo trên bundle **đang phục vụ**, và nó **thay thế** phần trên ở đúng những điểm được nêu tên dưới đây.

## Bundle của phụ lục này

| mục | giá trị |
|---|---|
| served js | **`assets/index-DYYSN7oG.js`** |
| sha256 | **`74e4b4f86ce8b4f5061456138ce1ff79aa084d7662c54c24a123e5bb679d651f`** |
| served css | `index-DecQnta9.css` (không đổi so với lượt trước) |
| đối chứng | `curl -s http://127.0.0.1:5173/ \| grep -o 'assets/index-[^\"]*\.js'` → `assets/index-DYYSN7oG.js`; `sha256sum workbench/frontend/dist/assets/index-DYYSN7oG.js` → `74e4b4f8…d651f`; `recheck_results.json` → `bundle.served_js` / `bundle.sha256` trùng |
| harness | `web_evidence/e2e_recheck/recheck_probe.py`, viewport 2560×1440×1, **21 check** |
| sửa trong lượt này | chỉ hai literal digest của chính harness (`js_expected` + docstring) từ `index-CBR6lHgq.js` / `acfb5fbb…ac21a` → `index-DYYSN7oG.js` / `74e4b4f8…d651f`; không sửa assertion nào |

## Kết quả probe: 21/21 PASS

`web_evidence/e2e_recheck/recheck_results.json` → `{"passed": 21, "failed": []}`.

| nhóm | check | nay |
|---|---|---|
| A — peer so sánh | `A1_compare_before_single`, `A2_compare_after_two_selected`, `A3_compare_arbitrary_peer`, `A4_compare_peer_clear`, `A5_compare_two_runs_one_model` | **PASS ×5** |
| B — nhãn run | `B1_run_labels_distinct`, `B2_noexport_run_matches_label`, `B3_sibling_run_has_candidates` | **PASS ×3** |
| C — inspector/Top-K | `C1_reference_caption_target`, `C2_MM_top10/top50`, `C2_Text-only_top10/top50`, `C2_Image-only_top10/top50`, `C3_three_conditions_differ`, `C4_bad_qid_no_results` | **PASS ×9** |
| D — CSMCIR | `D1_csmcir_row_text`, `D2_csmcir_native_link_opens_row` | **PASS ×2** |
| E — derived lineage | `E1_derived_lineage_validated`, `E2_parent_stays_partial` | **PASS ×2** |

**A5 — verdict: PASS.** Đây là check duy nhất đỏ ở lượt trước và bản sửa `3ab8d0d` đã ở trong bundle phục vụ.
Bằng chứng: `A5_compare_two_runs_one_model.png` + `columns.rows` của `recheck_results.json` —
`limn / base_iter0_all_categories · 10173dc4 · không có Top-K #193` và
`limn / base_iter0_all_categories · 9e4918a6 · Top-K 100 #193` là **hai hàng khác nhau**, dù cùng `model_id`;
`api_candidate_status` của hai run lần lượt là `TOPK_EXPORT_NOT_REQUESTED` và `AVAILABLE`, đúng như nhãn in ra.
Không còn check nào đỏ.

## Lỗi nào nay đã đóng, bởi commit nào, trên bundle nào

| lỗi ở phần trên | trạng thái | commit | bằng chứng trên `index-DYYSN7oG.js` / `74e4b4f8…d651f` |
|---|---|---|---|
| **#1** ComparePanel không bao giờ với tới TME (`slice(0,4)` trên thứ tự API, không có ô chọn peer) | **ĐÃ ĐÓNG** | `c25d67f` "reachable comparison peers, unique run labels, derived lineage, CSMCIR native link" | `A2_compare_after_two_selected`: URL `runs=c%3Af461f391…%2Cc%3Acd1a696d…` dựng bảng **2 hàng** — `hint / fashioniq · f461f391 · Top-K 100 (đang xem) #98` + `tme / fashioniq_0 · cd1a696d · Top-K 100 #65`, `cohortLine` "Cohort: 2 lượt…". Có `[data-testid=compare-peer-picker]` / `compare-peer-select` liệt kê **20** lựa chọn (**không** cắt 4): `A3` chọn được `conesep a4f786d8` mà bản cũ chưa từng hiện; `A4` bỏ lượt đã thêm vẫn chọn lại được. Ảnh `A1_compare_before_single.png`, `A2…`, `A3_compare_arbitrary_peer.png`, `A4_compare_peer_clear.png` |
| **#2** hai run khác nhau mang nhãn y hệt (`hint / fashioniq`, `tme / fashioniq_0`), chọn nhầm là mất Top-K | **ĐÃ ĐÓNG** | `c25d67f` (nhãn run có run-id stub + trạng thái Top-K); `3ab8d0d` (bảng so sánh dùng đúng nhãn đó thay vì `model_id` trần) | `B1_run_labels.txt` — cả 20 dòng đều phân biệt được, ví dụ `hint / fashioniq · 4bc9893a · Top-K 50`, `hint / fashioniq · f461f391 · Top-K 100`, `tme / fashioniq_0 · ad477312 · không có Top-K`, `tme / fashioniq_0 · cd1a696d · Top-K 100`; `B1_run_labels_distinct.png`. `B2_noexport_run_matches_label`: chọn dòng "không có Top-K" thì khối ứng viên nói đúng "Lượt này chưa có danh sách ứng viên … no candidate export directory for this run" — nhãn và dữ liệu khớp. `B3`: dòng anh em cùng model thì **có** ứng viên |
| **A5 (bảng so sánh in trùng tên peer)** | **ĐÃ ĐÓNG** | `3ab8d0d` | `A5_compare_two_runs_one_model` PASS; nhãn peer nay là nhãn composition đầy đủ (xem verdict ở trên) |

Hai commit này **không** đóng lỗi #3 và #5. Lỗi #4 hoá ra **không phải lỗi hiện hành** — nó đã được sửa từ trước lượt đo đầu (xem hàng #4 dưới đây).

## Lỗi nào VẪN CÒN, và một hàng gốc phải rút lại (không nằm trong phạm vi giờ này)

| # | lỗi | nay vẫn tái hiện được? | ghi chú |
|---|---|---|---|
| **#3** | Lưu cohort là ngõ cụt trên dữ liệu CIRCUS | **CÒN** | `FailuresPage.tsx` `saveCohort()` vẫn tự `throw` khi mọi run là `c:`; và `/api/runs` vẫn trả **`[]`**, nên `PUT /api/cohorts` (validate qua `validate_analysis_selection`) không có run hợp lệ. `GET /api/cohorts` → `[]`. Đo lại trên `index-DYYSN7oG.js` (`F3_cohort_still_dead.png`): nút "Lưu tập đã lọc (6,003 query)" → `[data-testid=save-feedback][data-status=error]`, "Lưu thất bại: API cohort chỉ nhận run của chỉ mục per-query; CIRCUS release run chưa có endpoint cohort…"; `GET /api/runs` → `[]`; `GET /api/cohorts` → `[]`. **Không** có cohort `e2e-check-20261111` nào được tạo |
| **#4** | Trang mặc định mở subset `sf` trong khi dữ liệu cần ở `full` | **KHÔNG CÒN — hàng gốc là đọc sai của tôi, không phải quan sát** | `routes.defaultLocation()` vẫn ghi `subset: "sf"` (đó là giá trị *trước khi* resolve), nhưng `App.tsx:148-166` gọi `chooseInitialScope` / `chooseSubsetFor` rồi `replaceState` sang subset có nhiều dòng đo được nhất — bản sửa này ở commit `3828163` (2026-11-10), tức **đã có trước cả lượt đo đầu tiên của tôi**, nên hàng #4 ở phần trên là suy luận từ việc đọc `defaultLocation()` chứ không phải kết quả đo. Đo lại trên `index-DYYSN7oG.js`: `GET /` trần → `subset-select = full`, URL `…&subset=full`; `/?protocol=circus_fashioniq_release_v1` (không nêu subset) → `full`; `&subset=sf` nêu rõ → giữ `sf`. Ảnh `F2_default_subset_after_addendum.png`. Số click của bước 1 vì vậy là **2**, không phải 3 như hàng gốc ghi |
| **#5** | Độ trễ vào danh sách run dao động mạnh | **CÒN (chưa đo lại)** | số ở lượt trước: **0,87 s / 3,854 s / 1,731 s** cho cùng một deep link. Lượt này **không** đo lại, nên con số vẫn là của `index-BLH2yome.js`; chưa có bằng chứng nó đã được xử lý |
| **bước 7 — `/api/models` cold** | chưa đo được | **NOT_VERIFIED** | `_MODELS_CACHE` là cache toàn tiến trình (`workbench/backend/main.py:172-176`); lượt này **không được** restart service nên không tạo được trạng thái cold. Đo được chỉ có warm: **2,092–3,205 ms** (5 lần, 87.139 byte) |

## Bước 6 của bảng gốc: đọc lại cho đúng hiện trạng

- **So sánh hai model cùng query** — nay **PASS** (trước là FAIL): HINT `#98` và TME `#65` nằm chung một bảng, chứng minh bằng `A2_compare_after_two_selected.png`.
- **Mở kết quả từ Phân tích lỗi** — vẫn PASS như lượt trước (`s6_failure_analysis_to_inspector.png`, `s6_jumped_from_failures.png`).
- **Một ghi chú `e2e-check-20261111`** — vẫn PASS như lượt trước: `GET /api/annotations` trả
  `{"query_id":"7:3143","protocol_id":"circus_fashioniq_release_v1","note":"e2e-check-20261111"}` (`s6_note_after_reload.png`).
- **Cohort `e2e-check-20261111`** — vẫn **không tạo được** (lỗi #3). Đây là phần duy nhất của bước 6 còn FAIL.

## Chạy lại phụ lục này

```
workbench/envs/ctl312h/bin/python \
  workbench/docs/reviews/20261111-master-audit/web_evidence/e2e_recheck/recheck_probe.py
```
Probe tự kiểm digest bundle ở đầu mỗi lần chạy; nếu frontend được build lại lần nữa, `bundle_check` sẽ đỏ **trước** mọi
assertion khác — đó là tín hiệu phải cập nhật `js_expected`, không phải lỗi sản phẩm.
Không chạy `pytest` trong lượt này (một sweep tuần tự đang chạy, PID 1046852); không rebuild; không restart/kill service.
