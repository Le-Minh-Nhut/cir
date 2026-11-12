# 97 — API surface for the `tgcir` smoke run `1eee4c0adbc93f7708205a5e`

Read-only verification. No edit, commit, build, restart, signal, or GPU use. The CLVC full run
(PID 1366364, `clvc_net --subsets full --topk-depth 100`) was executing throughout; it was not
touched, `composition_eval.py` was not run, `/tmp/cir_gpu.lock` was not taken. All web access was
HTTP GET against the live systemd user units (`127.0.0.1:8000` backend, `127.0.0.1:5173` frontend);
`systemctl` was never invoked. JSON was parsed with `json.load` and walked with plain dict access —
no nested `?q=` selectors.

Run under test: `model_id` `tgcir`, `checkpoint_id` `tgcir_archive_unresolved`, subset `full`,
5-query smoke, `--topk-depth 100`.

Artifact side:
`workbench/artifacts/composition/summary/tgcir_tgcir_archive_unresolved_1eee4c0adbc93f7708205a5e/full.json`
Candidate artifact side:
`workbench/artifacts/composition/runs/1eee4c0adbc93f7708205a5e/candidates/tgcir_tgcir_archive_unresolved/`
(`multimodal.jsonl`, `text_only_black_image.jsonl`, `image_only_empty_text.jsonl` — 5 rows each,
`depth: 100` and `missing_query_count: 0` in each manifest).

---

## 1. `GET http://127.0.0.1:8000/api/composition/runs/1eee4c0adbc93f7708205a5e?subset=full` → **HTTP 200**

Body captured: 20 053 bytes cold, 20 841 bytes warm (see §7b). Top-level identity:

| field | value |
|---|---|
| `model_id` | `tgcir` |
| `checkpoint_id` | `tgcir_archive_unresolved` |
| `execution_status` | `COMPLETED` |
| `query_count` | `5` |
| `conditions` | `["multimodal", "text_only_black_image", "image_only_empty_text"]` |
| `protocol_id` | `circus_fashioniq_release_v1` |
| `gallery_count` | `74381` |
| `gallery_identity_sha256` | `1db68c59d96f014cdbecf936e108653de98abef0bd9a98a8abc2fe08b5dec4b4` |

Per-condition status (identical in artifact and API):

| condition | `status` | `unscored_count` | `fallback_used` | `category_scope_failure_count` |
|---|---|---|---|---|
| `multimodal` | `COMPLETED` | 0 | false | 0 |
| `text_only_black_image` | `COMPLETED` | 0 | false | 0 |
| `image_only_empty_text` | `COMPLETED` | 0 | false | 0 |

Per-condition metrics (identical in artifact and API):

| condition | R@1 | R@10 | R@50 | mrr | ndcg_full | coverage | scored/query |
|---|---|---|---|---|---|---|---|
| `multimodal` | **not reported** | 20.0 | 40.0 | 0.044481057352368764 | 0.18061210310965697 | 1.0 | 5/5 |
| `text_only_black_image` | **not reported** | 0.0 | 0.0 | 0.002228127690261614 | 0.10082312072434303 | 1.0 | 5/5 |
| `image_only_empty_text` | **not reported** | 0.0 | 0.0 | 0.00019029794641650075 | 0.07693918292263288 | 1.0 | 5/5 |

Gap fields (identical in artifact and API): `composition_gap_ndcg` `0.4417698538002782`,
`mrr_gap` `0.04225292966210715`, `gap_cohort_count` `5`, `gap_multimodal_ndcg`
`0.18061210310965697`, `gap_text_only_ndcg` `0.10082312072434303`, `gap_image_only_ndcg`
`0.07693918292263288`, `gap_unavailable_reason` `null`. `shortcut_outcomes`:
`{"both": 0, "composition_required": 1, "image_only": 0, "neither_scored": 0, "text_only": 0, "unresolved": 4}`.

**R@1, stated as fact and not smoothed over:** neither the artifact nor the API carries a
`recall_at_1` field for any condition. `metrics.<cond>.recall_at_1` is `<ABSENT>` on **both** sides,
so this row is equal-by-absence, not a served value. There is no R@1 to compare.

### 1a. Field-by-field artifact vs API

61 fields compared; **0 mismatches**. `<ABSENT>` means the key does not exist on either side.

| field | artifact | API | match |
|---|---|---|---|
| `model_id` | `"tgcir"` | `"tgcir"` | MATCH |
| `checkpoint_id` | `"tgcir_archive_unresolved"` | `"tgcir_archive_unresolved"` | MATCH |
| `protocol_id` | `"circus_fashioniq_release_v1"` | `"circus_fashioniq_release_v1"` | MATCH |
| `execution_status` | `"COMPLETED"` | `"COMPLETED"` | MATCH |
| `query_count` | `5` | `5` | MATCH |
| `conditions` | `["multimodal", "text_only_black_image", "image_only_empty_text"]` | `["multimodal", "text_only_black_image", "image_only_empty_text"]` | MATCH |
| `condition_status.multimodal.status` | `"COMPLETED"` | `"COMPLETED"` | MATCH |
| `condition_status.multimodal.unscored_count` | `0` | `0` | MATCH |
| `condition_status.multimodal.fallback_used` | `false` | `false` | MATCH |
| `condition_status.multimodal.category_scope_failure_count` | `0` | `0` | MATCH |
| `condition_status.text_only_black_image.status` | `"COMPLETED"` | `"COMPLETED"` | MATCH |
| `condition_status.text_only_black_image.unscored_count` | `0` | `0` | MATCH |
| `condition_status.text_only_black_image.fallback_used` | `false` | `false` | MATCH |
| `condition_status.text_only_black_image.category_scope_failure_count` | `0` | `0` | MATCH |
| `condition_status.image_only_empty_text.status` | `"COMPLETED"` | `"COMPLETED"` | MATCH |
| `condition_status.image_only_empty_text.unscored_count` | `0` | `0` | MATCH |
| `condition_status.image_only_empty_text.fallback_used` | `false` | `false` | MATCH |
| `condition_status.image_only_empty_text.category_scope_failure_count` | `0` | `0` | MATCH |
| `metrics.multimodal.recall_at_1` | `"<ABSENT>"` | `"<ABSENT>"` | MATCH |
| `metrics.multimodal.recall_at_10` | `20.0` | `20.0` | MATCH |
| `metrics.multimodal.recall_at_50` | `40.0` | `40.0` | MATCH |
| `metrics.multimodal.mrr` | `0.044481057352368764` | `0.044481057352368764` | MATCH |
| `metrics.multimodal.ndcg_full` | `0.18061210310965697` | `0.18061210310965697` | MATCH |
| `metrics.multimodal.coverage` | `1.0` | `1.0` | MATCH |
| `metrics.multimodal.scored_count` | `5` | `5` | MATCH |
| `metrics.multimodal.query_count` | `5` | `5` | MATCH |
| `metrics.multimodal.unavailable_reason` | `null` | `null` | MATCH |
| `metrics.text_only.recall_at_1` | `"<ABSENT>"` | `"<ABSENT>"` | MATCH |
| `metrics.text_only.recall_at_10` | `0.0` | `0.0` | MATCH |
| `metrics.text_only.recall_at_50` | `0.0` | `0.0` | MATCH |
| `metrics.text_only.mrr` | `0.002228127690261614` | `0.002228127690261614` | MATCH |
| `metrics.text_only.ndcg_full` | `0.10082312072434303` | `0.10082312072434303` | MATCH |
| `metrics.text_only.coverage` | `1.0` | `1.0` | MATCH |
| `metrics.text_only.scored_count` | `5` | `5` | MATCH |
| `metrics.text_only.query_count` | `5` | `5` | MATCH |
| `metrics.text_only.unavailable_reason` | `null` | `null` | MATCH |
| `metrics.image_only.recall_at_1` | `"<ABSENT>"` | `"<ABSENT>"` | MATCH |
| `metrics.image_only.recall_at_10` | `0.0` | `0.0` | MATCH |
| `metrics.image_only.recall_at_50` | `0.0` | `0.0` | MATCH |
| `metrics.image_only.mrr` | `0.00019029794641650075` | `0.00019029794641650075` | MATCH |
| `metrics.image_only.ndcg_full` | `0.07693918292263288` | `0.07693918292263288` | MATCH |
| `metrics.image_only.coverage` | `1.0` | `1.0` | MATCH |
| `metrics.image_only.scored_count` | `5` | `5` | MATCH |
| `metrics.image_only.query_count` | `5` | `5` | MATCH |
| `metrics.image_only.unavailable_reason` | `null` | `null` | MATCH |
| `metrics.composition_gap_ndcg` | `0.4417698538002782` | `0.4417698538002782` | MATCH |
| `metrics.mrr_gap` | `0.04225292966210715` | `0.04225292966210715` | MATCH |
| `metrics.gap_cohort_count` | `5` | `5` | MATCH |
| `metrics.gap_multimodal_ndcg` | `0.18061210310965697` | `0.18061210310965697` | MATCH |
| `metrics.gap_text_only_ndcg` | `0.10082312072434303` | `0.10082312072434303` | MATCH |
| `metrics.gap_image_only_ndcg` | `0.07693918292263288` | `0.07693918292263288` | MATCH |
| `metrics.gap_unavailable_reason` | `null` | `null` | MATCH |
| `metrics.shortcut_outcomes` | `{"both": 0, "composition_required": 1, "image_only": 0, "neither_scored": 0, "text_only": 0, "unresolved": 4}` | `{"both": 0, "composition_required": 1, "image_only": 0, "neither_scored": 0, "text_only": 0, "unresolved": 4}` | MATCH |
| `topk_export.depth` | `100` | `100` | MATCH |
| `topk_export.status` | `"TOPK_EXPORT_COMPLETE"` | `"TOPK_EXPORT_COMPLETE"` | MATCH |
| `topk_export.missing_query_count` | `0` | `0` | MATCH |
| `topk_export.partial` | `false` | `false` | MATCH |
| `topk_export.requested` | `true` | `true` | MATCH |
| `topk_export.conditions` | `["image_only_empty_text", "multimodal", "text_only_black_image"]` | `["image_only_empty_text", "multimodal", "text_only_black_image"]` | MATCH |
| `smoke.limited` | `true` | `true` | MATCH |
| `smoke.query_limit` | `5` | `5` | MATCH |
| `smoke.label` | `"SMOKE QUERY LIMIT; full released gallery preserved"` | `"SMOKE QUERY LIMIT; full released gallery preserved"` | MATCH |
| `gallery_count` | `74381` | `74381` | MATCH |
| `gallery_identity_sha256` | `"1db68c59d96f014cdbecf936e108653de98abef0bd9a98a8abc2fe08b5dec4b4"` | `"1db68c59d96f014cdbecf936e108653de98abef0bd9a98a8abc2fe08b5dec4b4"` | MATCH |

A full recursive walk of both documents (every nested key, both directions) reports **0 value
mismatches, 0 artifact-only keys, 30 API-only keys**. The API-only keys are server-side decorations
absent from the stored summary: `metrics.ndcg_multimodal`, `metrics.recall_at_10_multimodal`,
`metrics.recall_at_50_multimodal`, `metrics.coverage`, `metrics.ndcg_text_only`,
`metrics.ndcg_image_only`, `metrics.outcomes`, `is_failed`, `training_or_transfer_label`,
`interpreter_verification_status`, `available_subsets`, `_recorded_identity`,
`scoring_module_source_hash`, `adapter_module_source_hash`, `is_model_run`, `provenance_label`,
`scoring_impl`, `capabilities`, `capability_detail`, `statistics`, `statistics_whole_run`,
`transfer`, `measurement`, `measurement_identity`, `measurement_group_id`,
`measurement_identity_confidence`, `measurement_identity_missing_inputs`,
`measurement_identity_schema_version`, `duplicate_attempt_count`, `duplicate_status`.
No artifact field is missing from the API.

Cross-check against the per-query ranks artifact
(`runs/1eee4c0adbc93f7708205a5e/ranks/tgcir_tgcir_archive_unresolved.json`), recomputed here:
multimodal R@10 20.0 / R@50 40.0 from ranks `[23, 6, 14404, 91, 832]`; text_only R@10 0.0 / R@50 0.0
from `[207, 249, 31976, 450, 24900]`; image_only R@10 0.0 / R@50 0.0 from
`[3697, 2469, 34272, 66684, 4314]` — all four served recall figures reproduce exactly.

---

## 2. `GET http://127.0.0.1:8000/api/composition/runs?include_smoke=true` → **HTTP 200**

60 rows. The run appears **once**, and it carries its smoke marker. Row, verbatim:

```json
{
 "run_id": "1eee4c0adbc93f7708205a5e",
 "model_id": "tgcir",
 "checkpoint_id": "tgcir_archive_unresolved",
 "subset": "full",
 "execution_status": "COMPLETED",
 "query_count": 5,
 "smoke": {
  "label": "SMOKE QUERY LIMIT; full released gallery preserved",
  "limited": true,
  "query_limit": 5
 },
 "is_failed": false,
 "topk_export": {
  "conditions": [
   "image_only_empty_text",
   "multimodal",
   "text_only_black_image"
  ],
  "depth": 100,
  "missing_query_count": 0,
  "partial": false,
  "reason": null,
  "requested": true,
  "status": "TOPK_EXPORT_COMPLETE"
 },
 "duplicate_status": "unique",
 "measurement_identity_confidence": "full"
}
```

The smoke marker is what lets a reader tell this from a full run: `smoke.limited: true`,
`smoke.query_limit: 5`, `smoke.label: "SMOKE QUERY LIMIT; full released gallery preserved"`, while
`gallery_count` stays the full released `74381` and `query_count` is `5`. Of the 60 rows served with
`include_smoke=true`, **10** carry `smoke.limited: true` and **50** do not.

Control: `GET .../api/composition/runs` (no `include_smoke`) → **HTTP 200**, **50** rows, and
`1eee4c0adbc93f7708205a5e` is **absent**. The marker and the opt-in are both real.

---

## 3. Candidates — `multimodal`, qid `7:1`

`GET http://127.0.0.1:8000/api/composition/runs/1eee4c0adbc93f7708205a5e/queries?subset=full&condition=multimodal&query_id=7%3A1&top_k=100&limit=1&offset=0` → **HTTP 200**

`candidates.status` = `AVAILABLE`, `candidates.depth` = `100`, `candidate_depth` = `100`, `candidate_source` = `run`. Returned `top_results` = **100**, JSONL `topk` rows = **100**, distinct ids = **100**, rows with `score: null` = **0**.

| rank | API `image_id` | API `gallery_position` | API `score` | JSONL `[position, score]` | match |
|---|---|---|---|---|---|
| 1 | `mbeir_images/fashioniq_images/B004YHIRJG.jpg` | 17777 | 0.6184659004211426 | `[17777, 0.6184659004211426]` | id+rank+pos+score ✅ |
| 2 | `mbeir_images/fashioniq_images/B005FUWFFS.jpg` | 20706 | 0.6091134548187256 | `[20706, 0.6091134548187256]` | id+rank+pos+score ✅ |
| 3 | `mbeir_images/fashioniq_images/B008M0M9BS.jpg` | 40918 | 0.6019251346588135 | `[40918, 0.6019251346588135]` | id+rank+pos+score ✅ |
| 100 | `mbeir_images/fashioniq_images/B004O6LVY0.jpg` | 15722 | 0.42433810234069824 | `[15722, 0.42433810234069824]` | id+rank+pos+score ✅ |

Full-window comparison over all 100 rows: positions equal `True`, ids equal `True`, ranks equal `True`, scores equal `True`. `target_rank`: JSONL `23` = API `23`. API `ranks` map: `{"multimodal": 23, "text_only_black_image": 207, "image_only_empty_text": 3697}`. `outcome` `unresolved`, `query_state` `retrieval_failure`, `unscored` `false`, `unscored_kind` `None`.


---

## 4. Candidates — the two conditions that were `MODEL_UNAVAILABLE` / 6003 unscored before the fix

This is the check that matters most: `text_only_black_image` and `image_only_empty_text` had no
readable API surface before the condition-degradation fix, so nothing below had ever been read.

`GET .../queries?subset=full&condition=text_only_black_image&query_id=7%3A1&top_k=100&limit=1&offset=0` → **HTTP 200**

`candidates.status` = `AVAILABLE`, `candidates.depth` = `100`, `candidate_depth` = `100`, `candidate_source` = `run`. Returned `top_results` = **100**, JSONL `topk` rows = **100**, distinct ids = **100**, rows with `score: null` = **0**.

| rank | API `image_id` | API `gallery_position` | API `score` | JSONL `[position, score]` | match |
|---|---|---|---|---|---|
| 1 | `mbeir_images/fashioniq_images/B00C5KF74W.jpg` | 62017 | 0.5972601175308228 | `[62017, 0.5972601175308228]` | id+rank+pos+score ✅ |
| 2 | `mbeir_images/fashioniq_images/B00AGSE9TE.jpg` | 51848 | 0.5716893076896667 | `[51848, 0.5716893076896667]` | id+rank+pos+score ✅ |
| 3 | `mbeir_images/fashioniq_images/B0071OIJ74.jpg` | 28424 | 0.5705326199531555 | `[28424, 0.5705326199531555]` | id+rank+pos+score ✅ |
| 100 | `mbeir_images/fashioniq_images/B0094E3G90.jpg` | 44639 | 0.4310251474380493 | `[44639, 0.4310251474380493]` | id+rank+pos+score ✅ |

Full-window comparison over all 100 rows: positions equal `True`, ids equal `True`, ranks equal `True`, scores equal `True`. `target_rank`: JSONL `207` = API `207`. API `ranks` map: `{"multimodal": 23, "text_only_black_image": 207, "image_only_empty_text": 3697}`. `outcome` `unresolved`, `query_state` `retrieval_failure`, `unscored` `false`, `unscored_kind` `None`.


`GET .../queries?subset=full&condition=image_only_empty_text&query_id=7%3A1&top_k=100&limit=1&offset=0` → **HTTP 200**

`candidates.status` = `AVAILABLE`, `candidates.depth` = `100`, `candidate_depth` = `100`, `candidate_source` = `run`. Returned `top_results` = **100**, JSONL `topk` rows = **100**, distinct ids = **100**, rows with `score: null` = **0**.

| rank | API `image_id` | API `gallery_position` | API `score` | JSONL `[position, score]` | match |
|---|---|---|---|---|---|
| 1 | `mbeir_images/fashioniq_images/B005X4PL1G.jpg` | 23741 | 0.2625194489955902 | `[23741, 0.2625194489955902]` | id+rank+pos+score ✅ |
| 2 | `mbeir_images/fashioniq_images/B00C181SNM.jpg` | 61402 | 0.2062666118144989 | `[61402, 0.2062666118144989]` | id+rank+pos+score ✅ |
| 3 | `mbeir_images/fashioniq_images/B00B2JJ15S.jpg` | 55358 | 0.20514549314975739 | `[55358, 0.20514549314975739]` | id+rank+pos+score ✅ |
| 100 | `mbeir_images/fashioniq_images/B00BLY37T0.jpg` | 58801 | 0.16747361421585083 | `[58801, 0.16747361421585083]` | id+rank+pos+score ✅ |

Full-window comparison over all 100 rows: positions equal `True`, ids equal `True`, ranks equal `True`, scores equal `True`. `target_rank`: JSONL `3697` = API `3697`. API `ranks` map: `{"multimodal": 23, "text_only_black_image": 207, "image_only_empty_text": 3697}`. `outcome` `unresolved`, `query_state` `retrieval_failure`, `unscored` `false`, `unscored_kind` `None`.


### 4a. Evidence that no window was silently emptied

| probe | multimodal | text_only_black_image | image_only_empty_text |
|---|---|---|---|
| `candidates.status` | `AVAILABLE` | `AVAILABLE` | `AVAILABLE` |
| `candidates.depth` | 100 | 100 | 100 |
| returned `top_results` | 100 | 100 | 100 |
| JSONL `topk` length | 100 | 100 | 100 |
| distinct `image_id` | 100 | 100 | 100 |
| rows with `score: null` | 0 | 0 | 0 |
| `rank` sequence 1…100 | yes | yes | yes |
| JSONL scores non-increasing | yes | yes | yes |
| `unscored` / `unscored_kind` | false / null | false / null | false / null |
| `query_state` | `retrieval_failure` | `retrieval_failure` | `retrieval_failure` |
| `target_rank` (JSONL = API) | 23 = 23 | 207 = 207 | 3697 = 3697 |

All 15 stored query rows (5 qids × 3 conditions) were checked: every one has depth 100, a distinct
id set, non-increasing scores, and a `target_rank` equal to the ranks artifact's value for that qid
and condition.

The three conditions are also not accidentally the same window: top-1 differs
(multimodal `B004YHIRJG.jpg`, text_only `B00C5KF74W.jpg`, image_only `B005X4PL1G.jpg`), and top-100
overlap is multimodal∩text_only = 34, multimodal∩image_only = 2, text_only∩image_only = 0. The
`image_only_empty_text` top-1 is the query's own reference image (`B005X4PL1G.jpg`, position 23741,
score 0.2625194489955902) — the expected degenerate identity hit when the text is emptied and only
the reference image remains, not an empty window. Its `target_rank` is still a real 3697.

### 4b. The id equality is not tautological

`workbench/backend/composition.py::_resolve_candidates` maps a stored `[gallery_position, score]`
pair to an image name through the release's gallery tuple. I re-derived that mapping independently
of the server (`workbench/composition/data.py::load_release`, release root
`workbench/artifacts/circus_release`, `gallery_len 74381`, `candidate_names 74381`) and rebuilt all
100 rows per condition from the JSONL alone. Against the served rows: positions equal `True`, ids
equal `True`, ranks equal `True`, scores equal `True` for all three conditions. The served `image_id`
is the release candidate filename at that stored position, so the id agreement checks the served
string rather than restating the same lookup.

---

## 5. Served frontend reaches the same run through the proxy

`GET http://127.0.0.1:5173/api/composition/runs/1eee4c0adbc93f7708205a5e?subset=full` → **HTTP 200**

`execution_status` = `COMPLETED`, `run_id` = `1eee4c0adbc93f7708205a5e`, `query_count` = `5` — same
as the backend. The proxied payload is identical in content to a warm backend fetch
(`proxy == warm backend` → `True`); the only difference against the *cold* backend capture is the
API-only `subset_provenance` block described in §7b.

---

## 6. Served bundle

`GET http://127.0.0.1:5173/` → **HTTP 200**

```
curl -s http://127.0.0.1:5173/ | grep -o 'assets/index-[^"]*\.js'
assets/index-DYYSN7oG.js
```

Served JS bundle: **`index-DYYSN7oG.js`** — the expected name. The stylesheet referenced alongside
it is `assets/index-DecQnta9.css`.

---

## 7. Secondary observations (outside the six checks; reported because this surface shows them)

**(a) `statistics.query_count` reads `6003` on a 5-query smoke run, in the same payload whose
per-condition denominators read `5`.** From the `multimodal` queries response, verbatim:

```json
"statistics": {
  "conditions": {
    "image_only_empty_text": {"denominator": 5, "scored_count": 5, "unscored_count": 0, "unscored_by_kind": {}, "coverage": 1.0},
    "multimodal":            {"denominator": 5, "scored_count": 5, "unscored_count": 0, "unscored_by_kind": {}, "coverage": 1.0},
    "text_only_black_image": {"denominator": 5, "scored_count": 5, "unscored_count": 0, "unscored_by_kind": {}, "coverage": 1.0}
  },
  "query_count": 6003
}
```

`statistics_whole_run.query_count` in the same body is `5`, and the run-detail endpoint's
`query_count` is `5` and agrees with the artifact. The `6003` comes from
`workbench/backend/composition.py:1372`, which passes `len(subset_queries)` — the size of the
release's `full` subset — while the conditions block beside it is scoped to the qids actually ranked
(5). It contradicts the `denominator: 5` printed one level above it and would read to an operator as
if 6003 queries had been scored in this smoke run. It is not an artifact↔API mismatch (the artifact
carries no `statistics` block at all), so it is not scored as a FAIL in §8.

**(b) `subset_provenance` is present or absent depending on which route warmed the cache.** The
first cold fetch of the run detail (20 053 bytes) had **no** `subset_provenance` key; two later
detail fetches and the proxy fetch (20 841 bytes) had it. Cause, read from the code:
`read_summary_file` caches the parsed dict by path and hands the *same object* to every caller, and
`list_composition_summaries` mutates it in place through `decorate_run_rows`; whichever route warms
the cache decides whether the detail response carries the block. When present it is correct
(`consistent: true`, `metric_check: {"metric": "multimodal.recall_at_10", "served": 20.0,
"served_scored_count": 5, "recomputed": 20.0, "agrees": true, "denominator": "scored queries of
this subset's own per-query artifact"}`).

---

## Kết luận

| # | check | PASS/FAIL | evidence |
|---|---|---|---|
| 1 | `GET :8000/api/composition/runs/1eee4c0adbc93f7708205a5e?subset=full` | **PASS** | HTTP 200. `model_id` `tgcir`, `checkpoint_id` `tgcir_archive_unresolved`, `execution_status` `COMPLETED`, `query_count` 5; all 3 conditions `COMPLETED` with `unscored_count` 0; 61/61 compared fields match the artifact, **0 mismatches**; recursive walk 0 value mismatches, 0 artifact-only keys. `recall_at_1` is absent on both sides — no R@1 is served at all. |
| 1b | metrics artifact == API | **PASS** | R@10 20.0 / 0.0 / 0.0; R@50 40.0 / 0.0 / 0.0; mrr and ndcg identical to the last digit; gaps `0.4417698538002782`, `0.04225292966210715`, `0.18061210310965697`, `0.10082312072434303`, `0.07693918292263288` identical. All four recall figures reproduce from the ranks artifact. |
| 2 | `GET :8000/api/composition/runs?include_smoke=true` | **PASS** | HTTP 200, 60 rows, run present once with `smoke: {"label": "SMOKE QUERY LIMIT; full released gallery preserved", "limited": true, "query_limit": 5}` and `query_count: 5`; absent without `include_smoke` (50 rows). |
| 3 | `GET .../queries?condition=multimodal&query_id=7%3A1&top_k=100` | **PASS** | HTTP 200. `candidates.depth` 100, 100 candidates returned, first three `1 B004YHIRJG.jpg`, `2 B005FUWFFS.jpg`, `3 B008M0M9BS.jpg`; id-for-id and rank-for-rank (also position- and score-for-score) equal to the JSONL across all 100 rows. |
| 4 | `condition=text_only_black_image` | **PASS** | HTTP 200. `candidates.depth` 100, 100 candidates, `target_rank` 207 = JSONL 207; first three `1 B00C5KF74W.jpg`, `2 B00AGSE9TE.jpg`, `3 B0071OIJ74.jpg`; full 100-row id/rank equality. Not empty: 100 distinct ids, 0 null scores, non-increasing scores. |
| 4 | `condition=image_only_empty_text` | **PASS** | HTTP 200. `candidates.depth` 100, 100 candidates, `target_rank` 3697 = JSONL 3697; first three `1 B005X4PL1G.jpg` (the reference itself), `2 B00C181SNM.jpg`, `3 B00B2JJ15S.jpg`; full 100-row id/rank equality. Not empty: 100 distinct ids, 0 null scores. |
| 4b | three-condition confirmation, no empty windows | **PASS** | All three conditions `status` `COMPLETED`, `unscored_count` 0, `fallback_used` false, `category_scope_failure_count` 0 in artifact and API. 15/15 stored query rows have depth 100, distinct ids, non-increasing scores, `target_rank` equal to the ranks artifact. Top-1 differs per condition; text_only∩image_only overlap 0. |
| 5 | `GET :5173/api/composition/runs/...?subset=full` (proxy) | **PASS** | HTTP 200, `execution_status` `COMPLETED`, payload identical to a warm backend fetch. |
| 6 | served bundle | **PASS** | `GET :5173/` HTTP 200, `assets/index-DYYSN7oG.js` — matches the expected name. |
| 7a | `statistics.query_count` = 6003 on a 5-query smoke run | **OBSERVATION** (not scored) | Served `statistics.query_count: 6003` beside `denominator: 5` per condition; `statistics_whole_run.query_count: 5`; run-detail `query_count: 5`. Artifact has no `statistics` block, so this is not an artifact↔API mismatch. |
| 7b | `subset_provenance` only after the list route warms the cache | **OBSERVATION** (not scored) | Cold detail fetch 20 053 B without the key; warm detail and proxy fetch 20 841 B with it. Shared mutable summary cache (`read_summary_file` + `decorate_run_rows`). When present the block is `consistent: true`. |

**No artifact↔API metric mismatch was found.** Every value in §1a matches exactly, unrounded. The
two items in §7 are defects in what the API adds around the artifact, not in the numbers it serves
from it.
