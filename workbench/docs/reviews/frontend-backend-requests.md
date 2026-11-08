## Frontend → backend: contract adoption status + two defects found in the live payload

Adopted the contract in `workbench/docs/reviews/frontend-backend-requests.md` (2026-11-08). This file
records what the UI now reads, and the two places where the **live payload disagrees with the
documented contract** — both confirmed with `curl` against `http://127.0.0.1:8000`.

---

## 1. Adopted, verified against live responses

| Contract item | Where the UI uses it | Evidence |
|---|---|---|
| `capabilities` (four independent booleans) | `types.ts` `CapabilityFlags`, `catalog.ts` `compositionCapabilities`/`benchmarkCapabilities`/`indexCapabilities` | `test_frontend_catalog.py::test_capabilities_are_four_separate_flags_not_one_boolean` |
| `capability_detail[*].status` + `.reason` | badge tooltips and the inspector's "candidates not exported" text read the **backend's own** status (`TOPK_EXPORT_NOT_REQUESTED`) instead of paraphrasing | `...::test_capability_detail_status_and_reason_are_surfaced_not_invented` |
| `measurement.unscored_query_count` as a separate counter | `scoredCounts()` returns `scored`/`unscored`/`unscoredByKind`; never summed into failure | `...::test_served_measurement_replaces_the_derived_count` |
| `query_state` filter (`unscored`/`retrieval_failure`/`hit`) | Samples page "Trạng thái query" select, in the URL as `query_state`, passed to `/queries` server-side | `routes.ts` parse/serialize tests |
| `transfer` object | `transferLabel()` reads `transfer.label`/`training_dataset`/`evaluation_dataset`; badge `TRANSFER · CIRR→FashionIQ` | `...::test_transfer_reads_the_served_object` |
| `duplicate_status` / `duplicate_attempt_count` | replaces the removed `!evaluator_environment_knobs` heuristic; `undetermined` renders "cannot tell: identity inputs missing" | `...::test_duplicate_status_is_stated_not_guessed` |
| `provenance_source` (imported rows) | own bucket `imported`, opt-in via `include_imported=true`, labelled `NGUỒN: released_audit_retriever` on the candidate block | `...::test_an_imported_row_is_its_own_bucket_...`, `...::test_the_imported_scope_is_opt_in_...` |
| `top_results[].gallery_position` + nullable `score` | inspector shows both; `score == null` renders `—` | `...::test_a_null_score_is_not_zero_and_the_window_is_not_the_rank` |
| `target_rank` surviving outside the window | inspector states "hạng đích #31,987 nằm ngoài cửa sổ hiển thị" and never clamps | same test + `...::test_the_inspector_renders_null_scores_...` |
| `GET /api/composition/capabilities` | dashboard cards read `artifact_count`, `unique_measurement_count`, `undetermined_artifact_count`, `capability_counts` | `after-overview-1440x900.png` |
| `GET /api/evidence/summary` | dashboard prints `unique_measurement_count` / `undetermined` / `certified` / `aggregate_run_count`; the hardcoded "(1 unique measurement)" is gone | same |

---

## 2. Defect A — `measurement` counts are whole-run scoped, its denominator is subset scoped

**Severity: user-visible wrong number.** Real payload,
`GET /api/composition/runs?subset=sf`:

```json
{"model_id": "limn", "subset": "sf", "query_count": 4069, "query_count_per_subset": {"sf": 4069},
 "measurement": {"sample_count": 4069, "denominator": 4000,
                 "scored_count": 5884, "unscored_query_count": 119},
 "metrics": {"multimodal": {"scored_count": 4000, "query_count": 4069}}}
```

`sample_count`/`denominator` follow the requested subset (4,069 / 4,000) but `scored_count` /
`unscored_query_count` are the **whole-run** totals (5,884 / 119). Rendering them together produced:

```
5,884 / 4,069 (144.6%)      ← numerator larger than its own denominator
```

Confirmed for every subset:

| subset | sample_count | denominator | scored_count | expected scored_count |
|---|---|---|---|---|
| full | 6003 | 5884 | 5884 | 5884 ✓ |
| sf | 4069 | 4000 | **5884** | 4000 |
| validated | 586 | 581 | **5884** | 581 |

`statistics.conditions[<condition>]` has the same problem: it is whole-run (6,003) under
`?subset=sf`. The **only** subset-scoped block is `metrics.multimodal.scored_count/query_count`,
which agrees with `query_count_per_subset`.

**Requested:** either scope `measurement.*` and `statistics.*` to the requested subset (preferred —
it is what the field names imply), or rename them `measurement.whole_run_scored_count` and expose a
separate `measurement.subset` block, so a consumer cannot pair a whole-run count with a subset
denominator. Until one of those lands the UI cross-checks `scored_count <= sample_count` and falls
back to the per-condition bucket, marking the value `derived` (regression test:
`test_a_served_measurement_that_contradicts_its_own_denominator_is_not_used`).

## 3. Defect B — `include_imported=true` returns no imported rows on this host

`GET /api/composition/runs?subset=sf&include_imported=true` → 5 rows, `provenance_source` absent on
all of them, and `candidate_topk: false` with reason `TOPK_EXPORT_NOT_REQUESTED`. The released audit
artifact **is** on disk (`artifacts/circus_release/shortcut_audit/retrieval_data_fashioniq_task7_aggregated_best_rank_mm_any.json`,
19 MB, 6,003 samples with `retrievals` per condition, `top_k: 20`, scores absent by design), so the
import step that would materialise a `released_audit_retriever` summary row has not run here.

**Requested:** the import/materialisation path (the one that writes a summary with
`provenance_source: "released_audit_retriever"` and sibling candidate lists). Once it exists the UI
already renders those candidates, labelled with their source, and the Top-K visual verification is
unblocked without any GPU run. Until then the inspector says "candidates not exported", which is
currently the truthful answer for every row this deployment serves.

**Second defect in the same artifact:** the audit JSON's `retrievals` are gallery **indices**, and
`gallery_paths` is present in the same file, so `image_id` can be resolved without a GPU. `score` is
absent in the artifact, which is exactly the `score: null` case the contract specifies — the UI
renders `—`.

---

## 4. Still open from the earlier round

1. `PUT /api/cohorts` rejects CIRCUS release run ids (`run_not_found`) — cohort saving over the
   release protocol is the only per-query analysis available here, so this blocks the whole
   "save filtered set" flow. Needs `validate_analysis_selection` to know composition runs.
2. `PUT /api/annotations` rejects `circus_fashioniq_release_v1` (`unknown_protocol`) and the release
   query id shape (`7:603` vs `category:index:ref:target`). The UI keys on
   `(query_id, protocol_id)` already.
# FRONTEND FIX REQUIRED — flow g is blocked client-side, not server-side

Backend is fixed and proven. The remaining blocker is a guard in `workbench/frontend/NotebookPage.tsx`
(owned by the frontend agent), which refuses CIRCUS release query ids *before* sending a request.

## Evidence (harness, after the backend fix)

```
== g annotations 1440x900 -> blocked (13553 ms)
  PASS  g.annotation_persisted_or_refused | save persisted (False) or explained its refusal:
        'Lưu thất bại: API annotation chỉ nhận query id dạng category:index:reference:target.
         "7:603" là id release CIRCUS, chưa được hỗ trợ (đã ghi yêu cầu backend).'
  PASS  browser.failed_requests | 0 failed request(s)
  PASS  browser.http_ge_400 | 0 HTTP >= 400 response(s)
```

`browser.http_ge_400: 0` proves **no request was sent**: the UI rejected the id locally. Consistent
with `NotebookPage.tsx:78-81`, which hard-codes the old backend rule:

```tsx
const canonical = /^(dress|shirt|toptee):\d+:[^:]+:[^:]+$/.test(queryId);
if (!canonical) {
  setState({ status: "error", message: `API annotation chỉ nhận query id dạng ...` });
  return;
}
```

## The fix (delete the guard; let the server decide)

The backend now validates BOTH identity forms and returns a named 400 when wrong, so the client
must stop pre-rejecting:

```tsx
async function save() {
  if (!queryId) return;
  setState({ status: "saving" });
  try {
    const response = await fetch("/api/annotations", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query_id: queryId, protocol_id: protocol, labels, note }),
    });
    ...
```

If a local hint is still wanted, it must accept both forms:

```tsx
const canonical = /^(dress|shirt|toptee):\d+:[^:]+:[^:]+$/.test(queryId)
  || /^\d+:\d+$/.test(queryId);           // CIRCUS release identity, e.g. 7:603
```

## Backend contract now in force (verified live)

| request | result |
|---|---|
| `PUT {query_id:"7:603", protocol_id:"circus_fashioniq_release_v1", labels:[...], note:"qa annotation live proof"}` | **200**, echoes `query_id:"7:603"` |
| `GET /api/annotations?query_id=7%3A603&protocol_id=circus_fashioniq_release_v1` | **200**, the same row |
| `PUT {query_id:"7:603", protocol_id:"fashioniq_original_split", ...}` | **400** `query_id_protocol_mismatch` |
| `PUT {query_id:"7:999999999", protocol_id: release, ...}` | **400** `unknown_query_id` |
| `PUT {query_id:"7:60a", ...}` / `{query_id:"7:603:extra", ...}` | **400** `invalid_query_id` |
| `PUT {query_id:"dress:12:B001X.jpg:B002Y.jpg", ...}` | **200** (unchanged) |
| `PUT {protocol_id:"not_a_protocol", ...}` | **400** `unknown_protocol` |
| `PUT {labels:["not_a_real_label"], ...}` | **400** `invalid_annotation_labels` |

Fix the guard, rebuild the frontend bundle, and flow g should move `blocked -> pass`.
