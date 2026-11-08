# Báo cáo kiểm chứng — ENCODER, CIRCUS FashionIQ composition run `6a528e877c328772e2c7bb28`

Reviewer độc lập, read-only. Toàn bộ số học dưới đây là tính tay của tôi (`math.log2`, vòng lặp dict thuần). Tôi **không** import helper metric nào từ `workbench.composition.metrics`.

Nguồn dữ liệu:
- Rank artifact: `workbench/artifacts/composition/runs/6a528e877c328772e2c7bb28/ranks/encoder_fashioniq.json` (6003 qid × 3 condition)
- Summary: `workbench/artifacts/composition/summary/encoder_fashioniq_6a528e877c328772e2c7bb28/{full,sf,validated}.json`
- SF cohort: `workbench/artifacts/circus_release/shortcut_free_subsets/fashioniq/shortcut_free/mbeir_fashioniq_task7_test_multimodal.jsonl` (4069 dòng)
- Validated cohort: `workbench/artifacts/circus_release/final_dataset/query_jsonl/fashioniq/validated_solved/mbeir_fashioniq_task7_test_validated_solved.jsonl` (368) + `.../validated_unsolved/mbeir_fashioniq_task7_test_validated_unsolved.jsonl` (218)
- Full cohort: `workbench/artifacts/circus_release/mbeir_fashioniq_task7_test.jsonl` (6003)

## 0. Định nghĩa nDCG dùng trong báo cáo này

Định nghĩa tôi dùng (khớp `workbench/composition/metrics.py:27` và `workbench/docs/CIRCUS_COMPOSITION_SPEC.md:60`): một item relevant duy nhất, gain = 1, discount = log2(rank+1), không cắt cutoff.

```text
ndcg_full(r)  = 1 / log2(r + 1)          # r = rank 1-based chính xác trên toàn bộ 74,381
R@K           = 100 * #{r <= K} / n      # PHẦN TRĂM, 0-100
MRR           = trung bình(1/r)
composition_gap_ndcg = 1 - max(mean_I, mean_T) / mean_MM
                       # cohort = các query có đủ 3 rank; trung bình từng condition TRƯỚC rồi mới chia
```

Tôi cũng đã kiểm tra độ nhạy của gap theo các định nghĩa khác, để không kết luận sai là lệch: nếu dùng `nDCG@10` cắt cutoff thì gap là 0.7378 / 0.9557 / 0.9624; dùng `nDCG@50` thì 0.6953 / 0.9012 / 0.9180; dùng chênh lệch R@10 (`1 - max(T,I)/MM` theo R@10) thì 0.7234 / 0.9447 / 0.9458. **Summary dùng `ndcg_full` (toàn catalogue), không dùng @K.** Vì vậy bảng dưới so sánh đúng cùng định nghĩa, và kết quả khớp tuyệt đối. Nếu ai đó so 0.4723 với một con số "nDCG@10 gap" thì đó là so sai định nghĩa, không phải summary sai.

Sai số số học: mọi ô dưới đây có abs diff đúng **0.0** (so sánh `float` bằng `==`, không phải làm tròn).

## 1. Bảng kiểm chứng theo subset

### full (6003 query, 74381 gallery)

| metric | claimed | recomputed | abs diff | rel diff | verdict |
|---|---|---|---|---|---|
| MM R@10 (percent) | 31.0178 | 31.01782442112277 | 0.0 | 0 | PASS |
| MM R@50 (percent) | 49.5752 | 49.5752123938031 | 0.0 | 0 | PASS |
| MM nDCG (fraction) | 0.29586 | 0.29586071325251023 | 0.0 | 0 | PASS |
| text nDCG (fraction) | 0.15613 | 0.15612507640224627 | 0.0 | 0 | PASS |
| image nDCG (fraction) | 0.10728 | 0.10727509765806255 | 0.0 | 0 | PASS |
| composition gap | 0.47230 | 0.47230210227676583 | 0.0 | 0 | PASS |
| gap_cohort_count | 6003 | 6003 | 0 | 0 | PASS |
| coverage | 1.0 | 1.0 (6003/6003 × 3 condition) | 0.0 | 0 | PASS |

Giá trị không được nêu trong bảng claim nhưng tôi cũng tính và khớp: MM MRR `0.16970724586506772`, text MRR `0.04665721885560417`, image MRR `0.010994057357674506`, `mrr_gap 0.12305002700946355`, `gap_multimodal_ndcg/gap_text_only_ndcg/gap_image_only_ndcg` = 0.29586071325251023 / 0.15612507640224627 / 0.10727509765806255.

### sf (4069 query)

| metric | claimed | recomputed | abs diff | rel diff | verdict |
|---|---|---|---|---|---|
| MM R@10 | 18.6778 | 18.677807815188007 | 0.0 | 0 | PASS |
| MM R@50 | 36.3234 | 36.32342098795773 | 0.0 | 0 | PASS |
| MM nDCG | 0.21886 | 0.21885803276804192 | 0.0 | 0 | PASS |
| text nDCG | 0.10775 | 0.10774821200217893 | 0.0 | 0 | PASS |
| image nDCG | 0.09765 | 0.09765477702407051 | 0.0 | 0 | PASS |
| composition gap | 0.50768 | 0.5076798843550945 | 0.0 | 0 | PASS |
| gap_cohort_count | 4069 | 4069 | 0 | 0 | PASS |
| coverage | 1.0 | 1.0 (4069/4069 × 3) | 0.0 | 0 | PASS |

Phụ: MM MRR `0.0913768513661978`, text MRR `0.006780447420925458`, image MRR `0.003859940208176203`, `mrr_gap 0.08459640394527235`.

### validated (586 query = 368 + 218)

| metric | claimed | recomputed | abs diff | rel diff | verdict |
|---|---|---|---|---|---|
| MM R@10 | 34.6416 | 34.641638225255974 | 0.0 | 0 | PASS |
| MM R@50 | 55.2901 | 55.2901023890785 | 0.0 | 0 | PASS |
| MM nDCG | 0.29646 | 0.2964578771690486 | 0.0 | 0 | PASS |
| text nDCG | 0.11611 | 0.11610761614305669 | 0.0 | 0 | PASS |
| image nDCG | 0.10776 | 0.10776004500996128 | 0.0 | 0 | PASS |
| composition gap | 0.60835 | 0.608350375939416 | 0.0 | 0 | PASS |
| gap_cohort_count | 586 | 586 | 0 | 0 | PASS |
| coverage | 1.0 | 1.0 (586/586 × 3) | 0.0 | 0 | PASS |

Phụ: MM MRR `0.16194676212924766`, text MRR `0.008826964174594368`, image MRR `0.006147699330819014`, `mrr_gap 0.1531197979546533`.

### Kiểm tra tính toàn vẹn cohort (độc lập)

- validated_solved ∩ validated_unsolved = 0; union = 586 (không trùng).
- `validated ⊂ sf ⊂ full` theo qid: 0 phần tử thiếu ở cả hai phép bao hàm.
- sf có 4069 qid phân biệt; full có 6003 dòng / 6003 qid phân biệt.
- Tập qid trong rank file **bằng đúng** tập qid của full release (không thừa, không thiếu).
- Cross-check artifact thứ hai: cả 18 file per-query jsonl dưới `workbench/artifacts/composition/results/circus_fashioniq_release_v1/encoder/fashioniq/6a528e877c328772e2c7bb28/{full,sf,validated}/{multimodal,text_only_black_image,image_only_empty_text}.jsonl` khớp rank table trên **mọi** qid (0 mismatch, 9 phép so sánh).

### Đơn vị đo

Recall lưu dạng **phần trăm (0-100)**: `metrics.py:73-74` nhân `100.0`, và bytes summary là `31.0178…` không phải `0.310178`. nDCG, MRR, gap là **fraction**. Không có việc trộn đơn vị trong summary.

## 2. Shortcut outcome bins tại K=10

| subset | text_only | image_only | both | composition_required | unresolved | neither_scored | verdict |
|---|---|---|---|---|---|---|---|
| full | 510 | 155 | 5 | 1282 | 4051 | 0 | MATCH |
| sf | 42 | 22 | 0 | 716 | 3289 | 0 | MATCH |
| validated | 11 | 6 | 0 | 188 | 381 | 0 | MATCH |

Tổng = số query của từng subset (6003 / 4069 / 586), khớp `query_count`. Quy tắc bin tôi dùng giống `metrics.py:127-146`: `both` nếu cả text và image hit; `text_only` nếu chỉ text; `image_only` nếu chỉ image; `composition_required` nếu MM hit mà cả hai unimodal đều trượt; `unresolved` nếu cả ba trượt; `neither_scored` nếu thiếu rank nào đó.

## 3. Findings theo mức độ nghiêm trọng

### F1 — REAL DEFECT: runner chạy một environment mà chính verifier của nó đánh BLOCKED, và bỏ qua toàn bộ bước kiểm dependency

Bằng chứng: `workbench/artifacts/composition/runs/6a528e877c328772e2c7bb28/ranks/encoder_fashioniq.json` → `identity.runtime_identity`:

```json
{"verification_status": "PYTHON_VERSION_MISMATCH", "verified": false,
 "verification_reason": "expected Python 3.9, found 3.11.15",
 "version": "Python 3.11.15; torch 2.7.1+cu128",
 "path": ".../workbench/envs/pair-adapted/bin/python",
 "environment_fingerprint": "8f0a3a15c90d746a2104ecbfdf8ff8932a0a0c94f5d8e566c2d89a1b52b8d210"}
```

Tôi tái hiện trực tiếp bằng `workbench/envs/ctl312h/bin/python`:

```text
unset   -> status PYTHON_VERSION_MISMATCH tier BLOCKED reason expected Python 3.9,  found 3.11.15
sm_120  -> status PYTHON_VERSION_MISMATCH tier BLOCKED reason expected Python 3.10, found 3.11.15
contract (unset):   python 3.9,  pytorch 1.13.1
contract (sm_120):  python 3.10, pytorch ==2.7.1
```

Nguồn gate: `workbench/backend/runtime.py:1645-1652` — khi `python_version_satisfies` trả `False`, hàm **return ngay** với `tier = TIER_BLOCKED`. Do return sớm, khối dependency verification bên dưới (`runtime.py:1655+`) **không bao giờ chạy**: `report.packages == {}`, `content_verified == False`, `validated_environment_fingerprint is None`, và `_BLOCKED_STATUSES` (`runtime.py:1148`) chặn cả GPU probe (`READY_TIERS` = `runtime.py:1829`).

Nhưng run vẫn chạy tới `COMPLETED`. Cửa hậu là đường composition: `workbench/composition/model_encoder.py:65-105` (`Adapter.status()`) chỉ kiểm checkpoint, **sự tồn tại** của interpreter, CUDA khả dụng, và mapping — **không** kiểm Python version, **không** gọi `verify_environment`. `workbench/scripts/composition_eval.py:64-78` (`_path_ready`) + `:278-290` (`adapter_status.ready`) là toàn bộ gate trước khi chạy. Vì vậy status mismatch được **ghi lại** nhưng không bao giờ được **thi hành** trên đường composition.

Hệ quả chính xác: xem mục 5.

### F2 — REAL DEFECT (đang ngủ, chưa gây sai số hiện tại): dashboard hardcode badge CERTIFIED

`workbench/frontend/App.tsx:69` render cho **mọi** report benchmark được phục vụ:

```jsx
<span className="badge badge-pass">CERTIFIED</span>
```

không hề đọc `b.verification`. Trong khi đó backend **có** tính và phục vụ receipt thật: `workbench/backend/main.py:860-900` (`_gate_fields`) đưa ra `verification` gồm `state`, `reasons`, `evidence_reasons`, kèm nhánh fallback `state: "uncorroborated"`, `reasons: ["no evidence ledger corroboration for this run"]` cho report không có ledger entry. Trang Benchmark làm **đúng**: `workbench/frontend/BenchmarkView.tsx:47-50` dùng `isCertifiedEvidence(run)`.

Hiện tại cả hai report CSMCIR đều thật sự `verification.state == "certified"` (tôi truy vấn `/api/benchmark-runs`), nên chưa có pixel sai. Nhưng badge là vô điều kiện: một report uncertified được API phục vụ sẽ bị hiển thị là CERTIFIED. Defect thật, chưa kích hoạt.

### F3 — REAL DEFECT: `PASS_WITHIN_5_PERCENT_VARIANT` và lý do "variant" không được hiển thị nơi có số

Cả hai report có `benchmark_gate.variant = true`, `status = PASS_WITHIN_5_PERCENT_VARIANT`, vì `declared_variant` true (approved source patch / adapted runtime) theo `workbench/backend/paper_gate.py:403` + `workbench/scripts/evaluate_models.py:161-170`.

Nơi hiển thị:
- Benchmark card: chỉ in `gate.status` **bên trong** `<details>` đóng mặc định — `workbench/frontend/BenchmarkView.tsx:163-166`.
- Dashboard card: không in gì về variant.
- Chuỗi "variant: …" chỉ xuất hiện ở `GateDetail` của trang Evaluation-runs — `workbench/frontend/App.tsx:99`.

Thêm nữa, hai report CSMCIR là **cùng một kết quả**: `observed_metrics` giống hệt (57.0507 / 77.2986 / 67.1747), `category_metrics` giống hệt; chỉ khác `run_id`, `run_directory`, `environment_digest`, `evidence_digest`, `TMPDIR`, `replay-cache` path, và `evaluator_environment_knobs` (knobs chỉ có ở report `f03887d2`). Dashboard hiển thị chúng thành **hai card** với nhãn "2 certified research results" → trình bày một kết quả hai lần, có trọng số gấp đôi.

### F4 — REAL DEFECT: run_id không tồn tại vẫn trả 200 với payload trông hợp lệ

`workbench/backend/composition.py:260-375`: `get_composition_run_queries` load release queries bất kể rank file có tồn tại hay không. Kết quả đo được:

```text
GET /api/composition/runs/no_such_run_xyz/queries?subset=full
 -> HTTP 200, total = 6003, items[0] = qid "7:1" with all ranks null,
    queries_unavailable_reason = "Per-query exact ranks not found on disk ..."
GET /api/composition/runs/3f86f58483b368377e2c151e/queries?subset=full -> same shape
```

Không thể bị nhầm thành kết quả (mọi rank null + reason string), nhưng `total: 6003` gây hiểu nhầm và đúng ra phải là 404. Endpoint chi tiết `get_composition_run_detail` **có** trả 404 đúng — chỉ endpoint queries thiếu kiểm tồn tại.

### F5 — REAL DEFECT (đã khoanh vùng): hai summary `SYNTHETIC_*` tự mâu thuẫn; và trên UI chúng không thể truy cập

`workbench/artifacts/composition/summary/SYNTHETIC_habit_fiq_n02_fixture002/full.json`:

- Khai `composition_gap_ndcg = 0.2856`, nhưng chính các nDCG của nó (`mm 0.2881 / t 0.1520 / i 0.1130`) suy ra `1 - 0.1520/0.2881 = 0.4724`. Lệch **0.1868**.
- Bins cộng lại = **6001** trong khi `query_count = 6003`.
- `coverage 0.98` nhưng `common_cohort_count 6003`.
- `smoke.query_limit = 8` nhưng `query_count = 6003`.

`SYNTHETIC_encoder_fashioniq_fixture001/full.json`: bins cộng lại = 5881 ≠ 6003.

Nguồn: literal hardcode trong `workbench/backend/composition.py:400-460`. Các flag `is_synthetic / synthetic / synthetic_fixture = true` và `reproduction_status: FIXTURE_DEVELOPMENT_ONLY` là trung thực, nhưng **số bên trong tự phủ định nhau**. Về API: chỉ xuất hiện với `include_fixtures=true`, xếp sau row thật. Về UI: **không thể tới được** vì `workbench/frontend/CompositionView.tsx:83` gọi `/api/composition/runs?subset=${subset}` không kèm `include_fixtures`. Chính QA của repo ghi nhận defect này:

```text
workbench/docs/reviews/20261108T015603+0700/browser_qa/flows.json
  composition_switch / 1440x900 + 768x1024 + 390x844:
  "full|sf|validated: two SYNTHETIC_* fixture rows visible and labelled FIXTURE"
  -> ok: false, detail: "0 FIXTURE rows in the table", defect: true
```

Endpoint chi tiết cũng phục vụ fixture với `query_count 6003, coverage 1.0, queries total 6003, all unscored`, không reason.

### F6 — ACCEPTABLE: run LIMN đang chạy không thể bị phục vụ như đã hoàn tất

`workbench/artifacts/composition/runs/3f86f58483b368377e2c151e/` chỉ có `work/limn/{features,limn/worker-08d7f476612a4f95b5a0cf7ede773184/{job-2521.json,worker.stderr.log}}` + lock 0 byte `.limn_base_iter0_all_categories.lock`. `find` không thấy `ranks/`, không thấy `summary/`. Đo trực tiếp:

```text
GET /api/composition/runs                       -> không liệt kê 3f86f584… (chỉ 6a528e87…)
GET /api/composition/runs/3f86f584…?subset=full -> 404
GET .../3f86f584…/queries                       -> 6003 unscored + reason "pending run" (xem F4)
```

Trong lúc review có tiến trình `composition_eval.py --models limn --smoke 8` và worker `limn-adapted` đang chạy; không có gì được ghi vào `summary/`. Kết luận: không thể bị phục vụ nhầm là hoàn tất.

### F7 — REAL DEFECT (nhỏ nhưng là gốc rễ): `reference_eligible` được **assert**, không được **đo**

`workbench/composition/runner.py:707-708`:

```python
if not getattr(adapter, "reference_eligible", True):
    raise ValueError("adapter excludes the reference; CIRCUS requires reference-eligible ranks")
```

`getattr(..., True)` mặc định là **True** nếu adapter quên khai báo — permissive. Rồi `runner.py:748` (rank identity) và `runner.py:1004` (summary) ghi literal:

```python
"category_scopes": list(scopes), "reference_eligible": True,
"reference_eligible": True,
```

Không có chỗ nào suy ra field này từ kết quả ranking hay từ mask thực tế đã áp.

Với run này giá trị **đúng trên thực tế**, tôi đã kiểm chứng độc lập:
- 6003/6003 reference image nằm trong gallery (so khớp exact path và basename).
- 0 query có positive không resolve được vào gallery.
- `workbench/composition/adapters.py:167-168` tính rank bằng `row > row[index]` + tie-break theo gallery order trên **toàn bộ** vector 74381, **không** áp mask `-10e10`.
- 0 query có positive trùng reference image.

Nhưng nếu một adapter tương lai lặng lẽ mask reference, summary vẫn sẽ ghi `true`. Đây là assertion, không phải measurement.

### F8 — REAL DEFECT / UI: R@50 không được hiển thị ở đâu trong Composition view

Claim "web UI shows these values" đúng cho MM nDCG, MM R@10, text nDCG, image nDCG, comp gap, coverage — nhưng **sai cho R@50**. `workbench/frontend/CompositionView.tsx` không có cột R@50; `recall_at_50_multimodal` tồn tại trong `workbench/frontend/types.ts:17` nhưng **không dùng**. R@50 chỉ có trong JSON/API.

### F9 — DOCUMENTATION GAP: spec và các con trỏ bằng chứng đã cũ

`workbench/docs/CIRCUS_COMPOSITION_SPEC.md:99` mô tả ENCODER **chỉ** như một smoke 8 query, dẫn attempt `1410522b57d4fe79083c60a6` và `17ef444e30b829ac22f2d3c8`. **Cả hai không tồn tại trên đĩa**:

```text
summary/encoder_fashioniq_1410522b57d4fe79083c60a6  -> NO
runs/1410522b57d4fe79083c60a6                       -> NO
summary/encoder_fashioniq_17ef444e30b829ac22f2d3c8  -> NO
runs/17ef444e30b829ac22f2d3c8                       -> NO
```

Run full thật `6a528e877c328772e2c7bb28` **không** được nhắc ở đâu trong `docs/` (0 hit `6a528e` trong spec). Spec vẫn ghi "Full composition evaluation remain[s] TARGET_GPU_PENDING" cho ENCODER trong khi đã có đánh giá full 6003 query, coverage 100%. Ngược lại `A37/A38` dẫn run `1410522b57…` như bằng chứng production (`workbench/docs/reviews/a37/A37_EVIDENCE_INDEX.md:100`), cũng không còn trên đĩa.

### F10 — DOCUMENTATION GAP: `reproduction_status` và cảnh báo runtime không được đặt cạnh con số

Summary mang `reproduction_status: METRICS_MEASURED_MAPPING_UNVERIFIED`, `checkpoint_mapping_status: VERIFIED_METADATA`, `checkpoint_training_noise_pct: null` ("unstated"). Composition view chỉ render `training_or_transfer_label` = "train noise unstated · eval noise 0%" (cột Condition / Transfer). Grep `reproduction_status|runtime|verification_status` trong `workbench/frontend/CompositionView.tsx` → **0 match**. Cả hai trường này có trong API JSON và hiện trên trang Models (`App.tsx:70`), nhưng **không** hiện nơi có con số composition. Điều này trái với câu trong spec:97 ("ENCODER's checkpoint-mapping uncertainty remains visible").

### F11 — ACCEPTABLE nhưng phải nêu: gap bị pha loãng bởi sàn ngẫu nhiên

Sàn ngẫu nhiên đều trên 74,381: `E[1/log2(r+1)] = 0.06874`.

| subset | MM nDCG | ×sàn | text | image | gap thô | gap hiệu chỉnh sàn |
|---|---|---|---|---|---|---|
| full | 0.2959 | 4.30× | 0.1561 (2.27×) | 0.1073 (**1.56×**) | 0.4723 | 0.6153 |
| sf | 0.2189 | 3.18× | 0.1077 (1.57×) | 0.0977 (**1.42×**) | 0.5077 | 0.7402 |
| validated | 0.2965 | 4.31× | 0.1161 (1.69×) | 0.1078 (**1.57×**) | 0.6084 | 0.7920 |

Gap hiệu chỉnh = `1 - (max(T,I) - E) / (MM - E)`. Vì image-only chỉ hơn sàn 1.4–1.6×, mẫu số của gap thô mang phần lớn là "nhiễu trên nền sàn"; so 0.472 với 0.608 giữa các subset chủ yếu là so mức độ pha loãng.

Bootstrap 2000 mẫu (seed 0, percentile 95%) cho gap: full `[0.4608, 0.4848]`, sf `[0.4942, 0.5213]`, validated `[0.5820, 0.6320]`. Chênh lệch full→sf và full→validated nằm ngoài nhiễu; ưu thế của validated so với sf cũng vậy, nhưng n=586.

Về `sf`: text-only R@10 = 1.03%, image-only R@10 = 0.54% — sát sàn. Gap composition trên sf nghĩa là "MM hoạt động yếu, ablation không hoạt động", chứ không phải bằng chứng về composition. full có 68% query là sf, nên headline 0.2959 được nâng bởi 1934 query ngoài sf (`validated_solved` một mình: R@10 50.8%). Đây là diễn giải, không phải defect: spec:5 quy định đúng cách trình bày.

### F12 — ACCEPTABLE: protocol integrity đạt, với một lưu ý

Tất cả đều tôi tính lại từ bytes:

| hạng mục | giá trị recomputed | khớp |
|---|---|---|
| `gallery_identity_sha256` từ audit `gallery_paths` + candidate table + protocol_id | `1db68c59d96f014cdbecf936e108653de98abef0bd9a98a8abc2fe08b5dec4b4` | exact |
| `gallery_image_identity.resolved_paths_sha256` (74381 ảnh resolve) | `b71e9a190c4b524ccab98f726d8076d482175926de5490aef1ff5a6ae2e08189` | exact |
| `gallery_image_identity.content_digest` (sha256 nội dung 74381 ảnh) | `6d590a5b05fd819ba414e249d34238dbcc8d65ef897cc16a351922bb53fb837a` | exact |
| `checkpoint_identity` (sha256 `encoder/fashioniq.pt`, 631,879,365 B) | `0f31c50150a7bda3495e3378df13b4cec7924b52efd23b5e5dd9731459da9173` | exact |
| `adapter_module_source_hash` | `1a8b0dd25827ca0ba90f504d81f2fa190cede1ee3fb3bfe335e70ab359005823` | exact |
| 5 method-source hash (`encode_queries`, `encode_targets`, `score`, `rank`, `mode_supported`) | tất cả khớp hash đã ghi | exact |
| `condition_transform_source_hash` (`inputs.py`) | khớp | exact |
| `model_provenance` (`upstream_commit 29a2a31d…7018`, `vendored_source_digest e878c0b1…f3f6`, `provenance_digest e7a6bf88…3a4f`) | tính lại từ registry + `workbench/third_party/ENCODER` | exact |
| 6 release artifact hash | khớp `data.py:15-22` và bảng spec:42-47 | exact |
| `hash_verification` / `failed_expectations` | `matched` / `[]` | exact |
| `is_synthetic` / `synthetic` / `synthetic_fixture` | `false` / `false` / `false` | exact |
| `gallery_count` | 74381 (cả 3 summary) | exact |
| `feature_storage.projected_bytes.global` = 74381·512·4 + 3·6003·512·4 | 189,214,720 | exact |

Ranks là **exact trên toàn gallery**, không phải Top-K cắt: max rank quan sát = 74,147 < 74,381 nên không có bão hòa ở cutoff; quy tắc rank đếm trên toàn bộ vector 74,381 (`adapters.py:148-169`); phân bố top-8 của `image_only` = 70,730…74,147 không có pile-up; số lượng giá trị phân biệt bình thường.

`gap_cohort_count` = 6003 / 4069 / 586: khớp, vì coverage = 1.0 trên cả ba condition ở cả ba subset (không có qid nào unscored), nên "cohort có đủ 3 rank" = toàn bộ cohort.

### F13 — ACCEPTABLE (cần biết): spec ghi paper 18,853 ảnh, release 74,381

`summary.paper_vs_release_gallery` = `{paper_stated_images: 18853, released_images: 74381, discrepancy: "paper states 18,853; pinned released audit artifact contains 74,381", identity_normalization: "removed <path_to_M-BEIR>/; retained released order"}`. Summary tự khai báo sự lệch này; spec:11 quy định không được gọi đây là tái lập chính xác paper. Đúng cách xử lý, không phải defect. (Lưu ý: các con số ENCODER ở đây **không** so sánh trực tiếp được với R@10 56.13 của paper ENCODER trên split/ gallery khác — native protocol là `fashioniq_val_split`, xem `workbench/registry/models.yaml:744-748`.)

## 4. Hai finding từ addendum (số dòng chính xác)

### F2 (chi tiết dòng)

- `workbench/frontend/App.tsx:69` — dashboard card render `<span className="badge badge-pass">CERTIFIED</span>` **vô điều kiện**, cùng với header cứng `<h3>Certified Aggregate Benchmark Evidence</h3>`.
- `workbench/backend/main.py:860-900` (`_gate_fields`) **có** tính receipt thật: `verification` với `state`, `reasons`, `evidence_reasons`, kèm nhánh fallback `state: "uncorroborated"`, `reasons: ["no evidence ledger corroboration for this run"]`, `evidence_reasons: ["the report carries no verification receipt"]` cho report không có ledger entry.
- `workbench/frontend/BenchmarkView.tsx:47-50` dùng đúng: `const isCertified = isCertifiedEvidence(run);` rồi `run.verification?.state?.toUpperCase() || (isCertified ? "CERTIFIED" : "UNCERTIFIED")`.

Kết luận: dữ liệu để điều kiện hóa badge **đã có sẵn và bị dashboard bỏ qua**. Defect thật, hiện đang ngủ (hai report đang phục vụ đều certified thật).

### F7 (chi tiết dòng)

- `workbench/composition/runner.py:707-708` — `if not getattr(adapter, "reference_eligible", True): raise …`. Mặc định **True** khi adapter không khai báo.
- `workbench/composition/runner.py:748` — `"category_scopes": list(scopes), "reference_eligible": True,` (rank identity).
- `workbench/composition/runner.py:1004` — `"reference_eligible": True,` (summary).
- Không có chỗ nào derive từ kết quả ranking hay mask đã áp.

Đúng trên thực tế với run này: 6003/6003 reference trong gallery, 0 positive thiếu, 0 query có positive trùng reference, `workbench/composition/adapters.py:167-168` không áp mask `-10e10`. Nhưng là **asserted, not measured**.

## 5. Phán xét về `PYTHON_VERSION_MISMATCH`

**Nó KHÔNG invalidate:**
- Số học rank và mọi metric ở mục 1, 2. Rank được tính từ feature của model; tôi đã tính lại độc lập từ rank file, và cross-check với 18 file per-query jsonl.
- Gallery: 74,381 ảnh, đúng thứ tự pinned — tôi tính lại identity từ audit + candidate table và digest nội dung của cả 74,381 file.
- Checkpoint: sha256 `0f31c501…9173` tôi băm lại từ `encoder/fashioniq.pt`, đúng byte và đúng kích thước registry.
- Source: hash module adapter, 5 method, `inputs.py`, vendored ENCODER digest — tôi tính lại từ file sống, tất cả khớp.
- Tính tái lập/định danh: `runtime_identity` (gồm cả `verification_status`, `verification_reason`, `environment_fingerprint`) nằm **trong** digest run-id. Tôi tính lại `sha256(json(identity, sort_keys))` = `6a528e877c328772e2c7bb28f8b17477c3d6fd1197b3b2a5f224eba58c6b09cc` = `fingerprint` trong rank file = `resume_key` trong summary, và `[:24]` = `run_id`/tên thư mục. Nên bất kỳ thay đổi environment hay status nào cũng tạo run id khác: **so sánh với run tương lai vẫn an toàn**.

**Nó invalidate:**
- Bất kỳ claim rằng run này tuân thủ **contract runtime đã khai**. Nó không đạt contract lịch sử (`python 3.9`, `pytorch 1.13.1`) và cũng không đạt contract adapted đã khai (`python 3.10`, `pytorch ==2.7.1`, `cuda 12.8`) — xem `workbench/registry/models.yaml:697-716`. Interpreter là 3.11.15.
- Bất kỳ claim rằng **tập dependency đã được verify**. Verifier return trước khối dependency, nên `open_clip_torch==2.24.0`, `torchvision==0.22.1` chưa bao giờ được nó kiểm. Tôi có tự kiểm `pair-adapted` bằng `pip list` (numpy 1.26.4, open-clip-torch 2.24.0, timm 0.9.12, torch 2.7.1+cu128, torchvision 0.22.1+cu128) — nhưng đó là phép kiểm của **tôi**, không phải của run.
- **So sánh với paper**: run này là kết quả trên một software stack khác tác giả (3.11 + torch 2.7.1 thay vì 3.9 + 1.13.1). Theo tiền lệ của chính repo, một adapted runtime là *declared variant* và verdict phải được gắn nhãn variant (`workbench/scripts/evaluate_models.py:161-170`, `paper_gate.py:403`) chứ không phải reproduction chính thức. Với composition, không có verdict tương đương — và đó chính là F3/F10.
- `environment_fingerprint` = `8f0a3a15c90d746a2104ecbfdf8ff8932a0a0c94f5d8e566c2d89a1b52b8d210` — tôi tính lại bằng `_record_environment_fingerprint` và khớp, nhưng nó là fallback path (`validated_environment_fingerprint is None`), tức identity môi trường được suy ra từ contract + interpreter prefix, **không** phải từ verify byte-exact của site-packages. Nó ổn định, nhưng yếu hơn một fingerprint đã validate.

Tóm lại một câu: **run là một phép đo thật; nó không phải một reproduction tuân thủ contract**. Việc nó chạy được cho thấy gate của đường composition (F1) không thi hành kết quả của verifier.

## 6. Fixture có bị nhầm thành kết quả thật không? LIMN có bị phục vụ như đã xong không?

### Fixture (`SYNTHETIC_encoder_fashioniq_fixture001`, `SYNTHETIC_habit_fiq_n02_fixture002`)

- Qua **API**: `list_composition_summaries(..., include_fixtures=False)` là mặc định của endpoint; fixture chỉ xuất hiện khi `include_fixtures=true`, và luôn xếp **sau** row thật (tier 2 trong `composition.py:196-206`). Trong policy đó: **không thể** bị nhầm.
- Qua **UI**: `CompositionView.tsx:83` không truyền `include_fixtures`, nên fixture **không thể** tới được UI. Vậy là an toàn khỏi bị nhầm — nhưng đây là do defect (QA của repo đánh dấu là defect thật), không phải do thiết kế.
- Đọc **JSON trực tiếp**: `is_synthetic/synthetic/synthetic_fixture = true` và `reproduction_status: FIXTURE_DEVELOPMENT_ONLY` là trung thực, **nhưng** các metric bên trong tự mâu thuẫn (F5: gap declared 0.2856 vs implied 0.4724; bins cộng 6001/5881 ≠ 6003; coverage 0.98 vs common_cohort 6003; smoke 8 vs query_count 6003). Ai chỉ đọc `metrics` mà bỏ qua flag thì bị dẫn sai.
- Qua **endpoint chi tiết**: `GET /api/composition/runs/SYNTHETIC_encoder_fashioniq_fixture001?subset=full` trả `query_count 6003, coverage 1.0` và `.../queries` trả `total 6003, all unscored, reason: null`. Response chi tiết **có** `is_synthetic: true` và `reproduction_status: FIXTURE_DEVELOPMENT_ONLY`, nên người đọc cẩn thận vẫn phân biệt được — nhưng `reason: null` ở queries là thiếu.

Kết luận: **không thể bị nhầm qua UI hoặc qua danh sách API mặc định**; **có thể bị dẫn sai nếu đọc thẳng file JSON metrics** hoặc nếu gọi endpoint với `include_fixtures=true` mà bỏ qua flag.

### LIMN `3f86f58483b368377e2c151e`

**Không thể** bị phục vụ như đã hoàn tất. Không có file summary nào (đã `find` toàn bộ cây), không có `ranks/`, không được liệt kê trong `/api/composition/runs`, và endpoint chi tiết trả 404. Endpoint queries trả 6003 unscored kèm lý do "pending run" — chấp nhận được, tuy F4 (thiếu 404) là defect riêng.

## 7. Những gì tôi KHÔNG thể verify

1. **Không chạy lại được phép tính model.** Không được chạy GPU job trong review này, nên tôi không thể tái tạo feature, score, hay rank từ checkpoint. Tôi verify rank từ **rank artifact** — nghĩa là tôi chứng minh số học trên các rank đó đúng, **không** chứng minh rank đó đúng với model. Đây là giới hạn cốt lõi: nếu chính quá trình encode/score sai, báo cáo này không phát hiện được.
2. **Không verify được dependency set bằng verifier.** Vì F1, `report.packages == {}`. Tôi kiểm `pair-adapted` bằng `pip list` (thấy `open-clip-torch 2.24.0`, `torch 2.7.1+cu128`, `torchvision 0.22.1+cu128`, `numpy 1.26.4`, `timm 0.9.12`) nhưng đó là kiểm thủ công ở thời điểm review, không phải trạng thái tại thời điểm run, và không chứng minh byte-exact.
3. **Không verify byte-exact site-packages.** `validated_environment_fingerprint is None`; tôi chỉ tái tạo được fingerprint ở nhánh fallback (`8f0a3a15…`), không phải fingerprint authoritative.
4. **Không kiểm chứng được siêu dữ liệu upstream commit của ENCODER.** Registry ghi `upstream_commit_sha: 29a2a31d…7018`, và `model_provenance_identity` tính lại khớp — nhưng `workbench/third_party/ENCODER` **không phải** repo git riêng (`git -C … rev-parse --show-toplevel` trả `/home/heheboiz/data/cir`); việc khớp chứng minh **registry và digest nội bộ nhất quán**, không chứng minh nội dung đó đúng là commit `29a2a31…` trên GitHub. Tôi không truy cập mạng để so.
5. **Không chứng minh hai report CSMCIR là hai lần chạy độc lập.** Chúng có cùng `observed_metrics` và `category_metrics`, khác `run_id`/`environment_digest`/`evidence_digest`/`TMPDIR`. Tôi không kiểm log/`command.json` của từng run để xác nhận chúng thật sự chạy riêng hai lần, nên chỉ báo cáo là "trình bày trùng một kết quả" (F3), không khẳng định là copy.
6. **Không đối chiếu với paper ENCODER.** Registry ghi `reported_r10 56.13 / reported_r50 77.59` trên `fashioniq_val_split`; composition run dùng `circus_fashioniq_release_v1` (74,381 gallery). Hai thứ không cùng protocol, nên tôi không so và không kết luận run composition "khớp/lệch paper".
7. **Số liệu screenshot là ảnh chụp tại thời điểm khác nhau.** `docs/reviews/20261107T234628+0700/composition/*.png` bị ghi lại trong lúc tôi review (mtime 02:00–02:01, trong khi `native/*.png` là 23:55). Tôi đối chiếu nội dung với JSON/rank table hiện tại và khớp, nhưng không xác nhận được trạng thái server tại đúng thời điểm từng ảnh được chụp; tôi dựa vào `browser_qa/flows.json` của lần QA sau (20261108T015603+0700) để xác nhận DOM == API.

## Kết luận

Về số: **21/21 ô claim của bảng, 18/18 outcome bin, và toàn bộ cohort/coverage đều khớp chính xác (abs diff 0.0)** — tính lại độc lập từ rank file thô, không dùng helper của runner. Protocol integrity đạt (gallery, hash, checkpoint, source, exact rank, không synthetic).

Về lỗi, theo thứ tự cần xử lý:

1. **F1** — đường composition bỏ qua kết quả verifier (BLOCKED vẫn chạy; dependency không được kiểm) — REAL DEFECT.
2. **F2** — dashboard hardcode CERTIFIED — REAL DEFECT (đang ngủ).
3. **F3** — variant không được gắn nhãn; một kết quả hiển thị thành hai — REAL DEFECT.
4. **F4** — run_id không tồn tại trả 200 với `total: 6003` — REAL DEFECT.
5. **F5** — fixture JSON tự mâu thuẫn; không thể tới được trên UI — REAL DEFECT.
6. **F8** — R@50 không hiển thị — REAL DEFECT (UI).
7. **F7** — `reference_eligible` assert chứ không đo — REAL DEFECT (gốc rễ).
8. **F9** — spec cũ, dẫn hai run không tồn tại — DOCUMENTATION GAP.
9. **F10** — mapping/runtime status không đặt cạnh con số — DOCUMENTATION GAP.
10. **F6, F11, F12, F13** — ACCEPTABLE (đã kiểm chứng hoặc là diễn giải đúng phạm vi).

---

# ADDENDUM — 2026-11-08: F15 (đính chính phạm vi của F2)

> Phần này **thêm mới**; không sửa finding nào ở trên. F2 ở mục 3 và mục 4 giữ nguyên văn bản gốc.

### F15 — REAL DEFECT (đã sửa): badge CERTIFIED hardcode nằm ở **HAI** chỗ render, không phải một

**Đính chính F2.** F2 chỉ nêu Dashboard (`App.tsx:69`) và vì vậy mô tả defect là "dashboard hardcode
badge CERTIFIED". Điều đó **đúng nhưng không đầy đủ**: cùng một defect tồn tại độc lập ở **hai** chỗ
render, và sửa `:69` một mình là một **fix chưa trọn** — trang Evaluation Runs vẫn tiếp tục khẳng định
certification mà receipt API không nói.

Bảng trạng thái từng chỗ render (đã đọc trực tiếp source):

| chỗ render | trạng thái trước addendum | xử lý |
|---|---|---|
| `App.tsx:69` Dashboard | **đã fix** — dùng `isCertifiedEvidence(b)` + nhãn duplicate-provenance | không đổi |
| `App.tsx:70` Models | **không hề có badge CERTIFIED** — chỉ render `aggregate-only (CAPABILITY_ABSENT)` | **không cần sửa gì** |
| `App.tsx:71` Evaluation Runs, nhánh aggregate | **STILL BROKEN** — `<span className="badge badge-pass">CERTIFIED</span>` vô điều kiện, không đọc `b.verification.state`, không có nhãn duplicate-provenance | **đã fix trong addendum này** |
| `BenchmarkView.tsx` | **đã đúng từ trước** — `isCertifiedEvidence(run)` (`:32`) + `run.verification?.state` (`:57`) | không đổi |

**Về `App.tsx:70` (Models) — nêu rõ để không ai đi tìm một defect không tồn tại:** trang Models
**chưa bao giờ** render badge CERTIFIED. Nó chỉ render badge `aggregate-only (CAPABILITY_ABSENT)`.
Không có gì để sửa ở đó, và không có finding nào về trang này.

`BenchmarkView.tsx` đã đúng từ đầu, nên F2 vốn không áp dụng cho trang Benchmark.

**Fix ở `App.tsx:71`.** Nhánh `safeBenchmarks.filter(b=>b.protocol_id===protocol&&!compatible.some(...))`
giờ tính hai biến trước khi render và dùng đúng pattern của Dashboard:

```jsx
const isCertified=isCertifiedEvidence(b);const isDuplicateWeak=!b.evaluator_environment_knobs;
```

- Badge: `<span className={`badge ${isCertified?"badge-pass":"badge-warn"}`}>{b.verification?.state?.toUpperCase()||(isCertified?"CERTIFIED":"UNCERTIFIED")}</span>`
  — tức là in **state thật do API phục vụ**; chỉ khi receipt không mang `state` mới suy ra từ
  `isCertifiedEvidence`.
- Nhãn provenance trùng: `{isDuplicateWeak&&<small className="duplicate-note" ...>duplicate attempt — provenance knobs missing</small>}`
  — cùng chuỗi, cùng style amber, cùng điều kiện `!b.evaluator_environment_knobs` như Dashboard, nên
  hai trang không thể nói khác nhau.
- Mọi thứ còn lại của nhánh giữ nguyên: badge `aggregate-only — per-query Top-K unavailable
  (CAPABILITY_ABSENT)`, badge `SYNTHETIC SOURCE` (khi `b.is_synthetic`), preview measured-vs-paper,
  `GateDetail`.

**Bằng chứng data-driven (kiểm bằng cách chặn response, không phải suy luận).**

`isCertifiedEvidence` (`frontend/evidence.ts:21`) đọc **`run.verification`**: `certified === true`,
`is_research_evidence === true`, `development_mode !== true`, `certification_mode !== "development"`.
Vì vậy cả hai nhánh đều được kiểm: `state` (nhãn hiển thị) và `certified` (quyết định `badge-pass`).

API thật (`GET /api/benchmark-runs`, 2 row, cả hai `certification_mode: "deployment"`,
`execution_authenticated: true`, `is_research_evidence: true`, `certified: true`):

| run_id | state | `evaluator_environment_knobs` |
|---|---|---|
| `20261107T054343058424Z_csmcir_fashioniq_f564570e` | `certified` | `null` → duplicate note |
| `20261107T054933277898Z_csmcir_fashioniq_f03887d2` | `certified` | có |

Playwright, viewport 1440x900, trang Evaluation Runs, protocol `fashioniq_original_split`:

| kịch bản | badge text quan sát được | class |
|---|---|---|
| dữ liệu thật | `CERTIFIED` (cả 2 row) + `duplicate attempt — provenance knobs missing` chỉ ở row `f564570e` | `badge badge-pass` |
| chặn `**/api/benchmark-runs*`, rewrite `verification.state = "uncorroborated"` | `UNCORROBORATED` (cả 2 row); **chuỗi `CERTIFIED` không còn xuất hiện trong body** | `badge badge-pass` (flags vẫn `certified:true`) |
| chặn, rewrite `state = null` + `certified = false` | `UNCERTIFIED` (cả 2 row) + badge gate đỏ `CHƯA CHỨNG NHẬN (UNCERTIFIED)` | `badge badge-warn` |

Hàng thứ hai là phép kiểm quyết định: cùng một bundle JS production, cùng DOM, chỉ response đổi →
badge đổi theo. Badge không còn hardcode. Hàng thứ ba chứng minh `isCertifiedEvidence` cũng là
driver của màu badge: `certified:false` → `badge-warn`.

Ảnh: `native/evaluation_runs_badge_1440x900.png`, `native/evaluation_runs_badge_390x844.png`,
`native/evaluation_runs_badge_390x844_rows.png`,
`native/evaluation_runs_badge_stub_uncorroborated_1440x900.png`,
`native/evaluation_runs_badge_stub_uncertified_1440x900.png`.

**Trạng thái test:** `npm run build --prefix workbench/frontend` pass;
`pytest workbench/tests/test_composition_api.py workbench/tests/test_csmcir_aggregate_api_parity.py -q`
→ **11 passed**, 5 warnings, 3.04s.

**Bài học cho review này:** "đã fix ở `App.tsx:69`" **không** đồng nghĩa "defect badge đã hết". Cùng
một lớp defect có thể có nhiều chỗ render; khi một finding nêu một dòng, phải grep toàn bộ các site
cùng loại trước khi đóng finding.
