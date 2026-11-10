# GUARDS_FIXES — R5 / R6 / R8 / R9

Ngày: 2026-11-10. Branch `research/cir-failure-workbench`, HEAD **`2e2e2d2`**.
Thư mục bằng chứng: `workbench/docs/reviews/20261110-e2e-repair/raw/guards/`.
Interpreter: `workbench/envs/ctl312h/bin/python` (CPython 3.12). Không commit gì.

Quy ước: mỗi kết luận dưới đây ghi kèm file **trong thư mục guards** đã chứa số đo. Số nào không
truy được về file thì không được đưa vào. Không có số nào bị làm lại, không có tolerance nào bị dời,
không có lần chấm điểm nào bị chạy lại.

---

## 0. Tổng hợp

| # | Verdict | Một dòng |
|---|---|---|
| **R5** | **CONFIRMED** | Cả 12 ô ghi `target_did = 7:42304` (positive của `7:3790`); rank/score/asset/candidate **đúng**; chỉ trường identity sai. |
| **R6** | **CONFIRMED** | `depth=0`/`-5` bị nuốt thành cửa sổ **1 dòng** với HTTP 200; `depth=1000` trả 100 dòng im lặng; summary và candidates đọc file **độc lập** nên có thể nói khác nhau. |
| **R8** | **CONFIRMED** | Response **không** có image URL, không category, không label theo scope, không object target riêng ⇒ target hạng 6891 **không thể render**. |
| **R9** | **CONFIRMED** | Không tồn tại gate nào trong code: `encoder` có **13** subcheck bắt buộc FAIL trong artifact (3× `batch_1_vs_n.max_rel` — 1.165e-2 / 7.795e-3 / 3.131e-3 vs `1e-3`; 2 non-tie rank change; 6 `population.*.max_rel`) mà mọi surface vẫn đọc "G2 PASS". |

Bốn finding đều **reproduce được** trên HEAD. Không có NOT_REPRODUCED.

---

## 1. R5 — LIMN ghi sai target

### 1.1 Verdict: **CONFIRMED**

Đối chiếu đủ **12 ô** (2 query × 2 scope × 3 condition) giữa: `qid` → positive DID của chính query
(đọc từ release) → asset trong gallery → rank/score/position trong summary **và** trong export row
**và** trong candidate được gắn `is_target`.

Bằng chứng: `raw/guards/r5_reconcile.json`, `raw/guards/r5_reconcile.out`.
Script: `raw/guards/r5_reconcile.py`.

Kết quả: `combinations: 12`, `verdict: R5_CONFIRMED`, **`problems: 6`** — và cả 6 đều là
`did_matches_own_query` trên `7:3038`. Chín kiểm tra còn lại (`asset_matches_own_positive`,
`target_flag_consistent_with_window`, `rank_summary_eq_export`, `score_summary_eq_export`,
`position_summary_eq_export`, `checkpoint_summary_eq_export`, `no_duplicate_candidate_ids`,
`export_depth_is_candidate_count`, `at_most_one_target_row`) **đạt ở cả 12/12 ô**. Đây là bằng chứng
định lượng cho câu "ranking đúng, chỉ identity sai".

Lưu ý về cửa sổ: 6/12 ô có target **ngoài** Top-100 (`152`, `389`, `2200`, `4047`, `4730`, `6891`)
nên không có candidate nào mang `is_target` — đúng như thiết kế, không phải thiếu sót; script kiểm
`target_flag_consistent_with_window` cho từng ô thay vì đòi một cờ `is_target` luôn tồn tại.

**Cái sai:**

| | `7:3038` | `7:3790` |
|---|---|---|
| positive id **của chính query** | `7:3416` | `7:42304` |
| asset tương ứng | `B004D8QCN4` | `B00E3WVAD2` |
| `target_did` **đã ghi** ở cả 6 ô | `7:42304` ❌ | `7:42304` ✅ |

**Cái đúng (đã kiểm từng ô, không suy đoán):** `target_rank`, `target_score`,
`target_gallery_position`, `target_gallery_id` của query, `checkpoint_sha256`, `depth`,
`is_target` flag, và danh sách candidate id — tất cả khớp giữa summary, export row và release
mapping. Không có candidate id trùng trong bất kỳ cửa sổ nào.

Ví dụ ô `7:3038 / shirt / multimodal`: summary rank 3, score 0.570377; export row rank 3, score
0.570377; candidate được flag nằm ở `gallery_position 13614`, `B004D8QCN4.jpg` — đúng asset của
positive `7:3416`. **Ranking đúng; chỉ nhãn identity sai.**

### 1.2 Nguyên nhân (lớp producer, đã đọc source)

`workbench/docs/reviews/20261109-native15-composition-limn/scripts/t2_dual_scope.py:273`:

```python
"target_did": query.positive_ids[0],
```

`query` ở đây là biến của vòng lặp `for qid in QIDS` **bên ngoài** (dòng 237), đã tiến tới phần tử
cuối (`7:3790`) trước khi vòng lặp trong (`for row, qid in enumerate(QIDS)`, dòng 264) ghi từng row.
Đúng như nghi ngờ: **một biến scope-query bị dùng lại trong vòng lặp serialize**. Sáu row của
`7:3038` vì thế mang positive id của `7:3790`.

### 1.3 Sửa ở lớp nào

Hai lớp, và cả hai đều đã làm:

**(a) Lớp phục vụ — `workbench/backend/limn_diagnostic.py` (đã sửa).** `target_did` được lấy từ
**record của chính query** lúc serialize, không đọc giá trị dùng chung. Kèm kiểm chứng: nếu giá trị
đã ghi **không** nằm trong positive id của chính query đó, nó bị báo là defect
`target_did_not_own_query` và giá trị đúng của query được phục vụ. Một producer sau này không thể
tái đưa biến dùng chung vào reader.

**(b) Artifact — sinh lại **bản sao đã sửa**, không ghi đè bản gốc.** Script
`raw/guards/r5_correct_summary.py` → `raw/guards/limn_dual_scope.corrected.json`.

* đổi **6** record (`7:3038` ở cả 2 scope × 3 condition): `7:42304` → `7:3416`;
* **không** đổi rank / score / position / asset / export path / candidate — danh sách
  `unchanged_fields` nằm trong `provenance_correction` của file;
* file gốc `limn_dual_scope.json` **không bị chạm** (sha256 `04c4419e…abf0`, ghi lại trong
  `raw/guards/r5_correction.json` cùng `"source_untouched": true`);
* digest của bản sửa: `7efb49dfb92be774f06ea13996e9081ca19c3dd7974e7dd8e07a3e069bbec9aa`;
* **verify**: đặt bản sửa vào một repo root tạm → `load_diagnostic` trả
  `DIAGNOSTIC_COMPLETE`, `complete: true`, **0 defect**, đúng `7:3416` / `7:42304`. Digest của
  manifest: `4849b5b995118279…`.

**Không rescore, không rerank.** Ranking đã được chứng minh đúng trước khi sửa (12/12 ô khớp export
+ release), nên theo đúng remedy đã đề ra, chỉ regenerate summary từ bằng chứng đúng và verify
digest.

### 1.4 Rerun

```bash
cd /home/heheboiz/data/cir
workbench/envs/ctl312h/bin/python workbench/docs/reviews/20261110-e2e-repair/raw/guards/r5_reconcile.py
workbench/envs/ctl312h/bin/python workbench/docs/reviews/20261110-e2e-repair/raw/guards/r5_correct_summary.py
```

---

## 2. R6 — diagnostic guard quá dễ dãi

### 2.1 Verdict: **CONFIRMED**

Probe trên **service thật** (`127.0.0.1:8000`, HEAD `2e2e2d2`): `raw/guards/r6_probe.json`,
script `raw/guards/r6_probe.py`.

| Input | HEAD | Đúng phải thế nào |
|---|---|---|
| payload rỗng (summary không có file) | `DIAGNOSTIC_NOT_PUBLISHED`, có reason | ✅ đã đúng |
| sai qid (`7:9999`) | 404 + "not one of the ambiguous queries" | ✅ đã đúng |
| sai scope (`dress`) | 404 + "not a diagnostic scope" | ✅ đã đúng |
| sai condition (`audio_only`) | 404 + "not a supported condition" | ✅ đã đúng |
| **`depth=0`** | **HTTP 200, cửa sổ 1 dòng, không thông báo** | ❌ **bị nuốt** |
| **`depth=-5`** | **HTTP 200, cửa sổ 1 dòng, không thông báo** | ❌ **bị nuốt** |
| **`depth=1000`** | **HTTP 200, im lặng trả 100 dòng** | ❌ **cắt âm thầm** |
| rank/score mâu thuẫn nhau | *không có đường nào phát hiện*: summary và candidates parse **độc lập** | ❌ |
| candidate id trùng | không kiểm | ❌ |
| depth hỏng (`depth=2` nhưng lưu 4 dòng) | không kiểm | ❌ |
| checkpoint không khớp giữa summary và export | không kiểm | ❌ |

**Summary vs candidates cho cùng một request:** đo đủ 12 cặp. HEAD trả **12/12 mismatch**: summary
khai `export: "limn_dual_scope.<scope>.<condition>.jsonl"` còn candidates **không có trường `export`
nào cả** (`raw/guards/r6_probe.json` → `summary_vs_candidates.mismatches`). Hai endpoint mô tả cùng
một ô bằng hai hợp đồng khác nhau, nên không có cách nào phát hiện chúng lệch nhau về số.

### 2.2 Sửa

`workbench/backend/limn_diagnostic.py` — **một record đã validate, dùng chung cho cả hai endpoint**:

* `_load_record()` dựng record một lần, đọc **7 file** (summary + 6 export) và tính **manifest**
  gồm `name`/`sha256`/`bytes` của từng file + `digest` trên toàn bộ. Response nào cũng mang manifest
  và digest này;
* cache **theo content identity**: key là sha256 của manifest. File bị sửa/cắt/hoán → key khác →
  validate lại. Không bao giờ phục vụ kết quả cache cho byte khác;
* mỗi ô thành một `combination` được validate: rank/score/position/asset của summary phải khớp
  export row; candidate phải có rank 1..N liên tục, score không tăng, **không trùng id và không
  trùng position**; `depth` phải ≥ số dòng lưu; checkpoint của summary phải khớp export; nếu target
  nằm trong cửa sổ thì phải có **đúng một** candidate `is_target` và hạng/score của nó phải khớp;
* mọi sai lệch trả về **lý do cụ thể** (`code` + `qid`/`scope`/`condition` + `detail`), **không**
  xoá bằng chứng: rank/score/cửa sổ vẫn được phục vụ, chỉ `complete` thành `false` và `status` thành
  `DIAGNOSTIC_EVIDENCE_DEFECT`;
* `depth` được **validate**, không clamp: `< 1` → từ chối; `> 200` → từ chối; nhỏ hơn depth đã lưu
  thì phục vụ đúng lượng yêu cầu và báo `depth_truncated`;
* `complete` chỉ `true` khi **không** còn defect nào.

Đo lại sau khi sửa trên instance scratch (`127.0.0.1:8011`):

| Input | Sau |
|---|---|
| `depth=0` | 404 `"depth must be at least 1, got 0; a zero or negative depth cannot return a window and silently returning one row would misstate the request"` |
| `depth=-5` | 404 cùng lý do |
| `depth=1000` | 404 `"depth 1000 exceeds the 200-row limit this endpoint serves"` |
| `depth=20` | 200, `depth_served 20` / `depth_stored 100` |
| `7:9999` | 404, lý do cụ thể |
| summary (artifact thật) | `DIAGNOSTIC_EVIDENCE_DEFECT`, `complete: false`, **6 defect** `target_did_not_own_query`, manifest digest `9fd200bd…` trên 7 file |
| summary vs candidates | cùng `manifest_digest`, cùng rank/score/checkpoint/target_did cho 12/12 ô |

Bằng chứng: `raw/guards/after_summary.json`, `raw/guards/after_candidates_7_3038.json`,
`raw/guards/after_candidates_7_3790.json`.

### 2.3 Test

`workbench/tests/test_limn_diagnostic_record.py` — **28 test, 28 pass** (gồm 6 test lặp tham số).
Cụ thể cho R6: `test_out_of_range_depth_is_refused_not_clamped[0/-1/-5/201/10000]`,
`test_depth_beyond_the_stored_window_reports_the_truncation`,
`test_summary_and_candidates_describe_the_same_combination`,
`test_the_validation_is_cached_by_content_identity`,
`test_rank_contradicting_the_export_is_a_defect`, `test_score_contradicting_the_export_is_a_defect`,
`test_duplicate_candidate_ids_are_a_defect`, `test_scores_that_do_not_descend_are_a_defect`,
`test_broken_depth_is_a_defect`, `test_mismatched_checkpoint_is_a_defect_not_a_silent_comparison`,
`test_empty_payload_details_are_named_not_empty`,
`test_wrong_identity_components_are_refused_with_a_reason[4 ca]`.

### 2.4 Rerun

```bash
cd /home/heheboiz/data/cir
workbench/envs/ctl312h/bin/python -m pytest workbench/tests/test_limn_diagnostic_record.py \
  workbench/tests/test_limn_dual_scope_diagnostic.py -q -p no:cacheprovider
# probe lại trên service:
workbench/envs/ctl312h/bin/python workbench/docs/reviews/20261110-e2e-repair/raw/guards/r6_probe.py
```

---

## 3. R8 — chỉ có text và Top-20

### 3.1 Verdict: **CONFIRMED**

Đo trên response thật (`raw/guards/r6_probe.json` → `cases`, và `r6_probe.py`):

* `depth` của client là **literal `20`** (`workbench/frontend/LimnScopeDiagnostic.tsx:288`,
  `&depth=20`), không đọc từ export depth nào;
* response candidate ở HEAD chỉ có `rank, gallery_position, gallery_id, image_path, score, is_target`.
  **Không có** `image_url`, không `category`, không `image_stem` ⇒ không gallery nào render được ảnh;
* **không** có object `target` riêng. Ô `7:3790 / shirt / text_only_black_image` có target rank
  **6891**, nằm ngoài mọi cửa sổ Top-100, nên **ảnh đích không xuất hiện ở bất kỳ đâu** trong
  response — đúng cái finding mô tả;
* `diagnostic` không mang label theo scope; cửa sổ K thay đổi không kèm thông tin gì về việc bị cắt;
* frontend ghi rõ "Ảnh không được render ở đây" (`LimnScopeDiagnostic.tsx:344-347`) trong khi
  Inspector dùng `Shot` + `/api/images/...` — **hai gallery khác nhau**, đúng như yêu cầu phải hợp
  nhất.

### 3.2 Sửa (backend, trong quyền sở hữu)

`limn_diagnostic.load_candidates` giờ trả:

* `identity` = `{qid, scope, condition, standard, manifest_digest}` — **scope là một phần của định
  danh request và response**, không phải tham số trang trí;
* `label` theo scope **và** condition: `"diagnostic — <scope> scope, <condition> (not canonical)"`,
  giữ ở cả summary (mỗi ô) và candidates;
* mỗi candidate có `image_url` (đã kiểm là **resolve được thật**, HTTP 200 `image/png` qua
  `/api/images/<category>/<stem>.jpg`), `image_stem`, `image_category`;
* `target` **luôn tách riêng** khỏi cửa sổ: `{rank, score, gallery_id, gallery_position, in_window,
  delta_to_rank_1, image_url, image_category, image_stem}`, cộng `target_did` / `target_asset` ở
  cấp payload;
* K lấy từ **export depth thật**: `depth_requested` / `depth_served` / `depth_stored` /
  `depth_truncated`. Không còn hằng số 20 nào ở backend.

Verify: `raw/guards/after_candidates_7_3790.json` → `target.rank 6891`, `target.in_window false`,
`target.image_url "/api/images/shirt/B00E3WVAD2.jpg"`, `depth_served 100`, `depth_stored 100`;
`/api/images/shirt/B00E3WVAD2.jpg` trả **200**. Test:
`test_the_target_is_returned_separately_from_the_window`,
`test_a_target_inside_the_window_is_flagged_once`,
`test_every_candidate_carries_an_image_identity`,
`test_k_follows_the_export_depth_not_a_hardcoded_constant`,
`test_scope_and_condition_are_echoed_as_request_identity`.

### 3.3 Phần R8 **không** sửa được ở đây — patch giao lại

`workbench/frontend/**` không thuộc quyền. Patch chính xác cần áp:

**(1) `workbench/frontend/LimnScopeDiagnostic.tsx` — kiểu dữ liệu.** Thay interface
`DiagnosticCandidates` (dòng ~74-84) bằng:

```ts
interface DiagnosticCandidate {
  rank: number;
  gallery_position: number;
  gallery_id: string;
  image_path: string;
  image_stem: string;
  image_url: string | null;      // đã resolve sẵn bởi backend
  image_category: string | null;
  score: number;
  is_target: boolean;
}

interface DiagnosticCandidates {
  qid: string;
  scope: string;
  condition: string;
  label: string;                 // "diagnostic — <scope> scope, <condition> (not canonical)"
  identity: { qid: string; scope: string; condition: string;
              standard: string; manifest_digest: string };
  complete: boolean;
  defects: Array<{ code: string; qid: string | null; scope: string | null;
                   condition: string | null; detail: string }>;
  target: { rank: number; score: number; gallery_id: string; gallery_position: number;
            in_window: boolean; delta_to_rank_1: number | null;
            image_url: string | null; image_category: string | null; image_stem: string };
  target_did: string | null;
  target_asset: string | null;
  target_rank: number;
  target_score: number;
  checkpoint_sha256: string | null;
  gallery_size: number | null;
  model_text: string;
  depth_requested: number;
  depth_served: number;
  depth_stored: number;
  depth_truncated: boolean;
  candidates: DiagnosticCandidate[];
  staged_query_image_is_evaluator_input: boolean;
  canonical_status_note: string;
}
```

**(2) cùng file — thay `&depth=20` (dòng 288) bằng K lấy từ export depth:**

```ts
// backend báo depth_stored; request K = giá trị đã lưu, không phải hằng số đóng trong UI.
+ `&depth=${encodeURIComponent(String(depthStored ?? 100))}`
```

với `depthStored` lấy từ lần gọi summary (`payload.queries[qid].scopes[scope][condition]` không
mang depth; nếu chưa có, gọi không tham số `depth` và đọc `depth_stored` từ response đầu tiên rồi
giữ lại — backend mặc định là depth đã lưu, nên cách gọn nhất là **bỏ hẳn tham số `depth`** khỏi URL
và để backend trả toàn bộ cửa sổ đã lưu).

**(3) cùng file — thay `<table>` candidate (dòng ~316-350) bằng gallery dùng chung với Inspector.**
`QueryInspector.tsx` đã có component `Shot` (`function Shot({category, id, label, target, compact})`,
dòng 278) và nó dựng `src` từ `/api/images/<category>/<id>`. Cần:

* export `Shot` từ `QueryInspector.tsx` (`export function Shot(...)`) và import vào
  `LimnScopeDiagnostic.tsx` — **một** gallery, không phải bản thứ hai;
* render `data.target` **trước**, luôn luôn, bằng `Shot` với `target={true}`, kể cả khi
  `target.in_window === false` (khi đó thêm nhãn "ngoài cửa sổ Top-K, hạng N");

```tsx
<Shot category={d.target.image_category ?? d.scope}
      id={d.target.gallery_id}
      label={`ảnh đích ${d.target_did ?? ""} (hạng ${d.target.rank})`}
      target />
{d.target.in_window
  ? null
  : <p className="hint">Ảnh đích ở hạng {fmtRank(d.target.rank)}, ngoài cửa sổ {d.depth_served} ứng viên đang hiện.</p>}
```

* render `d.candidates` bằng `Shot` với `category={c.image_category ?? d.scope}` và
  `id={c.gallery_id}`;
* hiển thị `d.defects` khi `d.complete === false` (một badge + danh sách lý do), và `d.label` làm
  tiêu đề khối; **không** render như một hàng canonical.

**(4) `workbench/frontend/LimnScopeDiagnostic.tsx:344-347` — xoá đoạn văn "Ảnh không được render ở
đây: đây là chẩn đoán…"**, vì lý do nó nêu (một tile ảnh cạnh metric canonical dễ bị đọc như hàng
canonical) đã được xử lý bằng `label` + `canonical_status` + `complete/defects`, chứ không bằng cách
bỏ ảnh.

**(5) `LimnScopeDiagnostic.tsx` — panel tổng hợp (dòng ~264):** đọc thêm `payload.complete` và
`payload.defects`; khi `complete === false` thì hiện cảnh báo "bằng chứng chẩn đoán tự mâu thuẫn"
cùng danh sách `code`/`detail`. Không đổi bất kỳ số nào.

Backend đã sẵn sàng cho cả 5 điểm này; không cần sửa gì thêm ở `main.py` (route hiện tại đã
forward `depth` và các exception đúng).

### 3.4 Rerun

```bash
cd /home/heheboiz/data/cir
workbench/envs/ctl312h/bin/python -m pytest workbench/tests/test_limn_diagnostic_record.py -q -p no:cacheprovider
curl -s "http://127.0.0.1:8011/api/composition/diagnostics/limn-dual-scope/candidates?qid=7:3790&scope=shirt&condition=text_only_black_image&depth=100"
```

---

## 4. R9 — gate báo PASS trong khi subcheck bắt buộc FAIL

Đây là finding quan trọng nhất, và verdict là **CONFIRMED** — nhưng không phải theo hình thức "gate
tính sai". **Không tồn tại gate nào trong code để tính.** Toàn bộ phán quyết G1/G2 sống ở văn bản
viết tay (`GATES.csv`, `ENCODER_PARITY.vi.md`, `FINAL_REPORT.vi.md`), và không có gì nối chúng với
artifact bên dưới. Đó chính là cơ chế khiến "PASS" in cạnh bằng chứng FAIL.

### 4.1 Gate hiện tại nói gì cho `encoder`, và subcheck thực sự làm gì

Đọc thẳng artifact `g1_g2.encoder.{dress,shirt,toptee}.{native,adapter}.json`:

| Subcheck bắt buộc | Declared | Đo được | Artifact tự ghi |
|---|---|---|---|
| `batch_size_1_vs_n.max_rel` (dress) | `batch_max_rel 1e-3` | **7.795e-03** | `within_tolerance: false` |
| `batch_size_1_vs_n.max_rel` (shirt) | `1e-3` | **3.131e-03** | `within_tolerance: false` |
| `batch_size_1_vs_n.max_rel` (toptee) | `1e-3` | **1.165e-02** | `within_tolerance: false` |
| `rank_differences_vs_native_val_control.non_tie_count` (dress) | 0 non-tie | **1** (gap 6.572e-04) | `explained_by_tie_epsilon: false` |
| `…non_tie_count` (shirt) | 0 non-tie | **1** (gap 1.604e-04) | `explained_by_tie_epsilon: false` |
| `population.features_vs_native_val_control.max_rel` (dress/shirt/toptee) | `forward_max_rel 1e-4` | **3.355 / 0.752 / 7.850** | `within_tolerance: false` |
| `population.gallery_vs_native.max_rel` | **harness KHÔNG test `max_rel` cho key này** (`max_abs` only) | **1.364 / 7.750 / 3.898** | `within_tolerance: true` (chỉ vì `max_abs` đạt) |

Các subcheck **đạt**: G1 identity (gallery order, reference/target ids, model text — exact,
2017/2038/1961 hàng, 0 mismatch); pixel decode bitwise (`max_abs = 0.0`); `forward max_abs`
≤ 1.65e-06 (declared 1e-5); `min_cosine` ≥ 0.999999999982; `score max_abs` 1.192e-07
(declared 1e-6); tie-convention 0 non-tie.

### 4.1b Harness dùng **hai chuẩn khác nhau cho cùng một loại đại lượng**

Đây là điểm làm R9 nặng hơn, và nó nằm trong chính harness
(`scripts/encoder_parity.py`), không phải ở artifact:

* `:745-747` — `layer4_forward_sample.query_native_tensors.within_tolerance` chỉ kiểm
  `max_abs <= TOL["forward_max_abs"]`; **không** kiểm `max_rel`.
* `:778-781` — `population.features_vs_native_val_control` và `features_vs_native_train_query` kiểm
  **cả** `max_abs` **và** `max_rel <= TOL["forward_max_rel"]`.
* `:783` — `population.gallery_vs_native.within_tolerance` chỉ kiểm `max_abs`; **không** kiểm
  `max_rel`.

Hệ quả đo được trên cùng một file `g1_g2.encoder.<cat>.adapter.json`: feature **gallery** có
`max_rel` **1.364 / 7.750 / 3.898** vẫn đọc `within_tolerance: true`, trong khi feature **query**
cùng shape, cùng dtype (`[N, 512]` float32) đọc `false` trên `max_rel` **0.752 / 7.850 / 3.355**.
Đó **không** phải một phân biệt khoa học — cùng đại lượng, cùng ngưỡng đã khai, cùng hiện tượng mẫu
số gần 0 — mà là sự không nhất quán ở chỗ **mỗi check gọi một comparator khác nhau**. Kéo theo:
câu tít "population `max_rel` FAIL" **phụ thuộc vào việc bạn nhìn phía nào** của cùng một artifact;
nếu chỉ đọc cờ `within_tolerance` thì population *gallery* trông sạch, còn population *query* thì
không.

Phát biểu cho đúng: **vấn đề mẫu số gần 0 (đã chứng minh ở §4.2) là thật, và harness đáp lại nó
một cách không nhất quán thay vì khai báo một chính sách.** Không có chỗ nào trong harness nói "với
đại lượng loại này ta bỏ `max_rel`"; chỉ có ba chỗ gọi comparator khác nhau.

**Hướng sửa (không làm trong lượt này, vì harness không thuộc quyền và sửa nó là đổi cách đo):**
một comparator đã khai duy nhất, áp cho **mọi** sub-check cùng loại; các cờ cũ giữ nguyên dưới nhãn
`v1`. Gate mới đã đi đúng hướng đó ở lớp phán quyết: nó áp bảng tolerance **đã khai** cho mọi
comparison layer-4 (kể cả `gallery_vs_native`), và ghi lại chỗ harness lệch dưới
`label_disagreements` — `{comparison, subcheck, declared, measured, harness_recorded,
gate_verdict, why}` — đồng thời giữ cờ cũ trên từng check
(`harness_recorded_within_tolerance`). Ba mục lệch hiện có, đều là `gallery_vs_native.max_rel`:
`1.364`, `7.750`, `3.898` với `harness_recorded: true`.

Test: `test_the_harness_tests_max_rel_on_some_keys_and_not_others`.

Còn surface nói gì: `GATES.csv` dòng `G2 forward/scoring` = **`NOT_RUN`** với phần chữ
*"encoder: G2 PASS with TWO honestly-declared failures"*; `ENCODER_PARITY.vi.md:399` =
**"G2 forward parity | PASS (absolute), one declared sub-check FAILED"**; `FINAL_REPORT.vi.md:27` =
*"G1 PASS, G2 PASS-kèm-failure-đã-khai-báo"*; và cột `G1 + G2` trong bảng tổng quan đếm `encoder`
là **đạt**. Vậy: một chữ **PASS** đứng ngay trước bằng chứng subcheck bắt buộc đã FAIL, và không có
code nào phản đối được.

**Phân biệt bốn thứ dễ lẫn — mỗi thứ giữ nguyên `dimension` riêng trong gate mới:**

| Dimension | Nội dung | `encoder` |
|---|---|---|
| `observer_parity` | pipeline quan sát tái tạo đúng macro của chính nó (`native_metrics_match_feature_rerank`) | ✅ |
| `input_parity` | G1: ids, gallery order, model text, token/mask, decode expression | ✅ |
| `forward_parity` | G2: cùng batch composition, cross-process (`max_abs`, `min_cosine`) | ✅ (`max_abs` đạt; `max_rel` FAIL vì mẫu số) |
| `batch_parity` | G2: batch 1 vs N — **subcheck actually fails** | ❌ |
| `scoring_parity` | G2: score và rank trên cùng feature | ❌ (2 non-tie change) |
| `metrics_parity` | hit-count equality | ✅ |

Không có chỗ nào lấy trung bình giữa các dimension, và `observer` **không bao giờ** đại diện cho
forward parity.

### 4.2 Chuẩn sai số tương đối gần-0 — tái lập đúng

`compare()` của harness
(`scripts/encoder_parity.py:73-85`) tính `max_rel = max(|a-b| / max(|b|, 1e-12))`. Trên vector
unit-norm 512 chiều có thành phần ~1e-8, nên một sai số tuyệt đối 1e-7 đọc thành 7.85.

Tái lập trên dữ liệu thô (`raw/encoder_parity/{adapter,native}.{dress,shirt,toptee}.npz`), không
chạy lại model:

| category | pair | max_abs | max_rel | abs tại điểm max_rel | ref tại điểm đó | số thành phần \|ref\| < 1e-5 | max_abs/atol |
|---|---|---|---|---|---|---|---|
| dress | population (val control) | 1.0999e-06 | 3.3548 | 9.439e-08 | **-2.814e-08** | 229 / 1,032,704 | 0.0110 |
| dress | gallery | 1.1325e-06 | 1.3636 | 4.325e-08 | **3.172e-08** | 248 / 1,345,536 | 0.0113 |
| shirt | population | 1.2778e-06 | 0.7520 | 7.359e-08 | **9.786e-08** | 195 / 1,043,456 | 0.0128 |
| shirt | gallery | 1.6484e-06 | 7.7500 | 9.057e-08 | **-1.169e-08** | 297 / 1,581,568 | 0.0165 |
| toptee | population | 9.872e-07 | 7.8500 | 1.764e-07 | **2.247e-08** | 180 / 1,004,032 | 0.0099 |
| toptee | gallery | 9.984e-07 | 3.8976 | 2.439e-07 | **-6.258e-08** | 293 / 1,485,824 | 0.0100 |

Kết luận: `max_rel` bị chi phối **hoàn toàn** bởi mẫu số ~1e-8, trong khi `max_abs` chỉ bằng ~1%
ngưỡng tuyệt đối. Đúng bản chất đã mô tả. **Không dời ngưỡng.**

### 4.3 Sửa: `workbench/backend/parity_gate.py` (module mới)

Hợp đồng: **một subcheck bắt buộc FAIL thì `passed` luôn `False`.** `PASS` là giá trị duy nhất khẳng
định mọi subcheck bắt buộc đã giữ; mọi trạng thái khác đều tường minh
(`PARITY_MANDATORY_SUBCHECK_FAILED`, `PARITY_NOT_RUN`, `PARITY_EVIDENCE_MISSING`,
`PARITY_EVIDENCE_MALFORMED`).

* **Đọc từ artifact, không từ văn bản.** Quyết định suy ra từ `max_abs`/`max_rel`/`min_cosine`/
  `non_tie_count`/`within_tolerance` và **tolerance do chính artifact khai** (`tolerances` block);
  không có ngưỡng nào được viết lại trong gate.
* **Lịch sử không bị viết lại.** Gate không ghi file nào. `historical` giữ nguyên
  `declared_tolerances`, `recorded_within_tolerance`, `recorded_non_tie_rank_changes` của từng
  category, đúng như artifact. Test `test_the_historical_record_survives_untouched` so **byte** của
  hai artifact trước/sau khi gọi gate. Không xoá, không sửa verdict cũ trong `GATES.csv` /
  `ENCODER_PARITY.vi.md` / `FINAL_REPORT.vi.md`.
* **`observer` / `input` / `forward` / `batch` / `scoring` / `metrics` tách dimension**, mỗi cái một
  boolean riêng (`verdict.dimensions`).
* **Protocol divergence không bị tính là FAIL.** `population.features_vs_native_train_query` (so
  với tensor `preprocess_train`) là divergence đã khai (D2), được ghi lại nhưng không phải subcheck
  bắt buộc — chính xác để gate không tự tạo thêm FAIL ngoài những cái đã khai.
* **Chuẩn `abs(a-b) <= atol + rtol*abs(b)`**: nếu cần, nó là `parity-absolute-relative/v2`, khai
  **trước** khi đo, và chỉ áp cho artifact **tự khai** chuẩn đó. Trên dữ liệu v1 hiện có, v2 được
  báo là `ADVISORY_ONLY` với `declared_before_measurement: false`, `applies: false`, và **không
  đóng góp gì vào `passed`**. Test `test_v2_is_advisory_on_v1_data_and_cannot_certify` khoá điều
  này: dưới v2, `encoder` **vẫn** `PARITY_MANDATORY_SUBCHECK_FAILED`, kết quả v1 vẫn nằm nguyên
  trong `checks[]`.

### 4.4 Kết quả gate mới trên HEAD

`raw/guards/r9_after.txt` (sinh bởi `raw/guards/r9_after.py`):

```
encoder  PARITY_MANDATORY_SUBCHECK_FAILED  passed=False  failures=13 / 96 checks
  dims {input_parity: True, forward_parity: False, batch_parity: False,
        scoring_parity: False, metrics_parity: True}
limn     PARITY_MANDATORY_SUBCHECK_FAILED  passed=False  failures=3  / 48 checks
  dims {input_parity: True, forward_parity: True, batch_parity: False, scoring_parity: True}
hint     PARITY_NOT_RUN                    passed=False  failures=0
ALL standard=parity-relative/v1 passed_models=[]
```

13 failure của `encoder`: 3 `batch_1_vs_n.max_rel`, 4 `ranks.*` (2 `non_tie_changes` +
2 `min_gap_vs_tie_epsilon`), 6 `population.*.max_rel`. Đúng bằng tập subcheck đã khai là FAIL, không
thêm không bớt. `limn` cũng không PASS (3 `batch_1_vs_n.max_rel`: 1.140e-02 / 2.698e-03 /
7.509e-03 vs `1e-3`) — nhất quán với `INPUT_FORWARD_PARITY.vi.md:258`, chỗ trước đây vẫn đọc "PASS".

**Coverage: `True` nghĩa là "mọi check đang tồn tại đều đạt", không phải "đã kiểm đủ".** Gate trả
`verdict.coverage[dimension] = {measured, not_measured}`. Với `limn`, `forward_parity: True` đi kèm
`not_measured: ["no population-level forward comparison recorded; only the fixed sample was
compared", "no cosine similarity recorded"]`; `metrics_parity` và `observer_parity` không có check
nào (artifact LIMN không ghi) và được nêu đúng như vậy. `encoder` có `forward_parity` với
`not_measured: []` — tức đã đo tới population. Không dimension nào được phép trông đầy đủ hơn thực
tế. Test: `test_a_passing_dimension_names_what_it_never_saw`.

Lưu ý trung thực: **chưa có surface sản phẩm nào tiêu thụ module này.** Nó là gate ở đúng lớp
(backend), có test, và trả lời được câu "mandatory FAIL có thể thành PASS không" = **không**. Việc
nối nó vào `/api/runs/gate` hoặc một endpoint riêng nằm ở `main.py` — **không thuộc quyền sở hữu**,
xem §5.

### 4.5 Test

`workbench/tests/test_gate_parity_subchecks.py` — **21 test, 21 pass**. Trong đó:

* `test_encoder_gate_is_not_pass_on_head` — `encoder` không PASS; 3 giá trị `max_rel` đúng bằng
  artifact (0.0031308995258445375 / 0.0077950176161260265 / 0.011646653843520266);
* `test_the_declared_tolerance_is_what_the_artifact_declares` — `batch_max_rel == 1e-3` v.v., đọc từ
  artifact;
* `test_limn_gate_is_also_not_pass_on_head` — LIMN không được miễn;
* `test_the_historical_record_survives_untouched` — so byte artifact;
* `test_the_dimensions_are_not_conflated`, `test_a_model_without_evidence_is_not_run_never_pass`,
  `test_a_partial_evidence_set_is_not_a_pass`;
* `test_a_failing_batch_relative_subcheck_cannot_pass`,
  `test_a_non_tie_rank_change_cannot_pass`,
  `test_a_rank_change_inside_the_tie_epsilon_is_allowed`, `test_a_clean_artifact_passes`,
  `test_a_near_zero_denominator_relative_error_still_fails_v1`,
  `test_v2_is_advisory_on_v1_data_and_cannot_certify`, `test_v1_remains_the_default_standard`,
  `test_an_unknown_standard_is_refused`, `test_a_broken_identity_subcheck_cannot_pass`,
  `test_the_abs_rel_rule_is_the_declared_form`,
  `test_the_harness_tests_max_rel_on_some_keys_and_not_others`,
  `test_a_passing_dimension_names_what_it_never_saw`,
  `test_limn_rank_checks_are_read_from_its_own_record_shape`,
  `test_evaluate_all_reports_no_passing_model_on_head`.

### 4.6 Encoder replay và defect upstream

Gate **không** replay model, nên không khẳng định gì về bit-identical. Defect đã chứng minh
(`third_party/ENCODER/model_try2.py:93` → `nn.Parameter(...).cuda()` trả plain Tensor, không đăng
ký; checkpoint 417 key, 0 key chứa `relation`; hai construction lệch ~4.2) được ghi trong
`parity_gate` như một sự thật nền tảng của artifact: mọi số liệu `encoder` được đo với relation được
pin ở cả hai phía, và gate không bao giờ kết luận "bit-identical replay đạt được" từ các số đó.

### 4.7 Rerun

```bash
cd /home/heheboiz/data/cir
workbench/envs/ctl312h/bin/python -m pytest workbench/tests/test_gate_parity_subchecks.py -q -p no:cacheprovider
workbench/envs/ctl312h/bin/python -c "
import sys; sys.path.insert(0,'.')
from workbench.backend.parity_gate import evaluate_all
print(evaluate_all()['standard'], evaluate_all()['passed_models'])"
```

---

## 5. Patch giao lại (file không thuộc quyền)

### 5.1 `workbench/frontend/LimnScopeDiagnostic.tsx` (R8)

Xem §3.3 — 5 điểm, đầy đủ mã. Điểm quan trọng nhất: xuất `Shot` từ `QueryInspector.tsx` và dùng
**cùng** component cho diagnostic, để hai panel không còn hai gallery khác nhau.

### 5.2 `workbench/backend/main.py` (R9 — nối gate vào surface)

Route hiện có `GET /api/runs/gate` trả gate paper-parity. Route mới, đặt cạnh nó:

```python
@app.get("/api/runs/parity-gate")
def parity_gates():
    """G1/G2 parity per model, derived from the artifacts. A mandatory FAIL is never PASS.

    The hand-written surfaces (GATES.csv, ENCODER_PARITY.vi.md) are evidence a reader consults;
    this endpoint is the coded verdict, and it cannot read PASS while a declared subcheck failed.
    """
    from workbench.backend.parity_gate import evaluate_all

    return evaluate_all()
```

Không cần sửa gì khác ở `main.py`: `load_diagnostic` / `load_candidates` đã tự xử lý lỗi và trả
`DiagnosticEvidenceError`, và route hiện tại đã map exception đó sang 503/404 đúng.

### 5.3 `workbench/docs/reviews/20261109-native15-composition-limn/` (R5)

Nếu muốn bản đã sửa thành artifact chính thức, thay `limn_dual_scope.json` bằng
`raw/guards/limn_dual_scope.corrected.json` (digest `7efb49df…`) và giữ file cũ dưới tên
`limn_dual_scope.json.pre-r5` — **không xoá**. Cho tới khi đó, reader báo `DIAGNOSTIC_EVIDENCE_DEFECT`
với 6 defect, là câu trả lời trung thực cho byte hiện có.

---

## 6. File đã thay đổi

| File | Việc |
|---|---|
| `workbench/backend/limn_diagnostic.py` | viết lại: một record validate dùng chung, manifest + digest, cache theo content, depth validate, `target_did` theo từng query, `target` tách riêng, `image_url`, label theo scope, defect có mã |
| `workbench/backend/parity_gate.py` | **mới**: gate G1/G2 suy ra từ artifact, 6 dimension, v1/v2, `PASS` chỉ khi không có mandatory FAIL |
| `workbench/tests/test_limn_diagnostic_record.py` | **mới**: 28 test (R5/R6/R8) |
| `workbench/tests/test_gate_parity_subchecks.py` | **mới**: 21 test (R9) |
| `workbench/tests/test_limn_dual_scope_diagnostic.py` | 1 dòng fixture: `checkpoints.sha256` `0…` → `a…` để khớp export của chính nó (fixture trước đây tự mâu thuẫn mà vẫn đòi `COMPLETE`) |
| `raw/guards/r5_reconcile.py` / `.json` / `.out` | bằng chứng R5 trước khi sửa |
| `raw/guards/r5_correct_summary.py` / `r5_correction.json` / `r5_correction.out` / `limn_dual_scope.corrected.json` | bản sửa R5 + digest |
| `raw/guards/r6_probe.py` / `r6_probe.json` | bằng chứng R6/R8 trước khi sửa (trên `:8000` HEAD) |
| `raw/guards/after_summary.json`, `after_candidates_7_3038.json`, `after_candidates_7_3790.json` | response sau khi sửa (trên `:8011`) |
| `raw/guards/r9_after.py`, `r9_after.txt` | gate mới trên HEAD (sinh lại được) |

**Không** chạm: `workbench/backend/main.py`, `workbench/frontend/**`, `workbench/registry/**`,
`workbench/composition/**`, `workbench/config/workbench.env`, mọi artifact trong
`20261109-native15-composition-limn/`, `CLEANUP_REPORT.md.save`. Không commit.

---

## 7. Ảnh hưởng run / cache

* **Không chạy lại model nào.** Không encode, không forward, không rescore, không rerank. Mọi số
  trong tài liệu này đọc từ artifact đã có trên đĩa hoặc từ response HTTP.
* **Không đổi số nào.** Rank/score/Recall/`max_abs`/`max_rel` y như campaign trước. R5 chỉ sửa một
  trường identity; R9 chỉ đọc artifact.
* **Cache:** `limn_diagnostic` có một cache trong process, key = sha256 của manifest 7 file. Đọc lần
  đầu ~1 lần 7 file (summary 12 KB + 6 JSONL ~3 MB); sau đó là hit cho tới khi byte đổi. Chỉ giữ
  **một** entry (clear rồi set), nên không tích luỹ bộ nhớ. `_resolvable` cache 8192 entry cho việc
  resolve ảnh. Endpoint vẫn không ghi gì xuống đĩa.
* **Chi phí endpoint:** `GET .../limn-dual-scope` giờ validate đủ 12 ô (~36 lần so sánh + sha256 7
  file) thay vì chỉ đọc summary — đo được vẫn dưới 1 s trên máy này; đây là giá của việc "một câu
  trả lời đã kiểm" thay vì hai câu trả lời chưa kiểm.
* **Service:** instance đang chạy `:8000` được start 07:24 nên **chưa** nạp code mới (không
  `--reload`). Tôi verify trên instance scratch `:8011` cùng lệnh
  (`python -m uvicorn workbench.backend.main:app --host 127.0.0.1 --port 8011`), `/api/benchmark-runs`
  trả 200. Để áp dụng cho `:8000`, restart bằng đúng lệnh cũ của nó; `/api/benchmark-runs` phải trả
  200 sau đó.
* **Test:** chỉ chạy 3 file được đặt tên: `test_limn_dual_scope_diagnostic.py` (12),
  `test_limn_diagnostic_record.py` (28), `test_gate_parity_subchecks.py` (21) = **61 pass, 0 fail**.
  Không chạy full suite.
