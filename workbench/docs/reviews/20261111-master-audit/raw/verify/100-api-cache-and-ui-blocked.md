# 100 — API cache, scoped `query_count`, provenance aliasing, and "blocked" as four separate facts

Scope: `workbench/backend/composition.py`, `workbench/backend/main.py`, new tests
`workbench/tests/test_api_*` / `test_composition_api_*`. Live stack read-only over HTTP GET; no unit
restarted, no evaluator started. HEAD before this pass: `82cb104` (two other workers landed
`d16a1ba`/`88dd3a1` mid-pass; the working tree was re-read before every edit).

---

## 1. "Blocked" from the API to the UI: four causes, four field paths

The web UI has exactly three renderers that can say "blocked/unavailable", and each reads a different
field. They must never collapse into one label.

### (a) Native command blocked — `COMMAND_BLOCKED_SOURCE_ENTRYPOINT`

This is TRUE of the pinned upstream `test.py` and says **nothing** about the composition path. Its
field path is `models[].checkpoints[].blocking_reasons[]` (a string), sourced from
`models[].command_status`.

`GET http://127.0.0.1:8000/api/models` → **HTTP 200**, 87 149 B, 17 entries.

```json
tgcir  reproduction_status = BLOCKED_COMMAND
  checkpoints[0].checkpoint_id            = "tgcir_archive_unresolved"
  checkpoints[0].checkpoint_downloaded    = true
  checkpoints[0].local_sha256             = "c37768f374c75bebff1299e870518c23bf6db3dc8a75c4d1f729ccd6dd02e966"
  checkpoints[0].official_sha256_match    = true
  checkpoints[0].adapter_command_ready    = false
  checkpoints[0].blocking_reasons = [
    "external runtime asset provenance unresolved: OpenAI CLIP ViT-B/16 weights/cache",
    "command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT"]

clvc_net  reproduction_status = BLOCKED_CHECKPOINT_MAPPING
  checkpoints[0].checkpoint_id = "author_drive_unresolved"
  checkpoints[0].blocking_reasons = [
    "torchvision ResNet-50 ImageNet weights missing: /home/heheboiz/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth",
    "external runtime asset file missing: torchvision ResNet-50 ImageNet weights (...)",
    "command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT"]
```

False reading to forbid: both models HAVE a completed composition run (below), so
`COMMAND_BLOCKED_SOURCE_ENTRYPOINT` is a statement about the pinned native entrypoint, not about
`workbench/composition`. The registry itself says so: `composition_adapter` is
`"workbench.composition.model_tgcir"` / `"workbench.composition.model_clvc_net"` for both.

UI renderers of this cause: `ModelInventory.tsx` column "Chặn bởi" (`checkpoints[].blocking_reasons`)
and `OverviewPage.tsx` "Registry mô hình · bị chặn" (counts `reproduction_status` starting `BLOCKED`).

### (b) One condition unsupported — `ABLATION_UNSUPPORTED`

Field path: `runs[].condition_status.<condition>.status` and `.reason`.

`GET /api/composition/runs/eab10106b2df8e97fb7a5ce5?subset=full` → **HTTP 200**, 21 738 B,
`model_id clvc_net`, `checkpoint_id author_drive_unresolved`,
`protocol circus_fashioniq_release_v1`, `run_id eab10106b2df8e97fb7a5ce5`,
`execution_status PARTIAL`, `query_count 6003`.

```json
"condition_status": {
  "image_only_empty_text": {"status": "ABLATION_UNSUPPORTED",
    "reason": "CLVC-Net cannot encode an empty edit: TextLSTMModel.forward tokenises to zero tokens and indexes x[0][0] (text_model.py:84-86), ... image_only_empty_text is unrepresentable in this model's text tower"},
  "multimodal":            {"status": "PARTIAL", "unscored_count": 119, "category_scope_failure_count": 119, "fallback_used": false},
  "text_only_black_image": {"status": "PARTIAL", "unscored_count": 119, "category_scope_failure_count": 119, "fallback_used": false}}
```

This is per-condition only; the run is `PARTIAL`, not `FAILED` (`_FAILED_EXECUTION_STATUSES =
{"FAILED"}`, `composition.py:193`).

### (c) A `PARTIAL` run that still has numbers

Same run, `runs[].execution_status` + `measurement.scored_count`:

```json
"execution_status": "PARTIAL",
"measurement": {"sample_count": 6003, "scored_count": 5884, "coverage": 0.980176578377478,
                "unscored_by_kind": {"category_unresolved": 119, "other": 6003}},
"metrics": {"multimodal": {"recall_at_1": 1.4106050305914344, "recall_at_10": 7.256968048946295,
                           "scored_count": 5884, "coverage": 0.980176578377478}}
```

5884/6003 MM queries scored → real numbers. The `other: 6003` kind is `image_only_empty_text`, which
is (b): the same run carries both facts at once and they must render separately.

Contrast — `tgcir` Full, `GET /api/composition/runs/7b1fa85007bffdab931f7f19?subset=full` →
**HTTP 200**, 72 732 B, `execution_status COMPLETED`, `query_count 6003`, all three conditions
`COMPLETED` with `unscored_count 0`.

### (d) GPU queue / dispatch refusal

Field paths: the `POST /api/jobs` 409 body (`detail.error`) and a job row's `status`
(`"blocked"`) + `error`.

Live, from `raw/verify/95-external-asset-unblock.md` §"POST codes and bodies" (same backend, re-read
today, unchanged by this pass):

```
POST /api/jobs {tgcir, tgcir_archive_unresolved, fashioniq_val_split} → 409
{"detail":{"error":"evaluation_blocked","message":"Official evaluation is not ready.",
  "details":{"blockers":["external runtime asset provenance unresolved: OpenAI CLIP ViT-B/16 weights/cache",
                          "command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT"]}}}
```

Dispatch refusal is `job.status = "blocked"` + `job.error` (`jobs.py:565-566`, `:582-583`), rendered by
`ExperimentsPage.tsx` `QueueView` (`job.status === "blocked" ? "warn"`, `job.error` shown verbatim).
`GET /api/jobs` → **HTTP 200**, `[]` (no job queued now, so no live blocked row exists to quote).

**These four are structurally distinct in the payload and stay distinct:** (a) is a registry fact about
a model's command, (b) is a condition status on a run, (c) is a run's execution status plus counters,
(d) is a request/job outcome. Nothing in the code merges them; the assertion below locks that in.

---

## 2. `_MODELS_CACHE` staleness — reproduced, then fixed

### Before

`main.py:170-176` was `_MODELS_CACHE: list[dict] | None = None` with no key: the first request of a
process pinned the document for that process's life. Measured live in `95-external-asset-unblock.md`
§"Did the API pick up the registry edit without a restart?": after the pinned ResNet-50 file was
acquired and verified, `GET /api/models` still listed `clvc_net`'s file as missing while
`POST /api/jobs` — which recomputes `preflight_blockers` per request — had already dropped that
blocker.

### After

`_MODELS_CACHE` is keyed on `_models_state_stamp(registry, config)`: registry / auxiliary /
preparation-contract document stats, the operator env file, every setting value the registry names
(interpreter vars, asset location/cache-root vars, `XDG_CACHE_HOME`/`HF_HOME`/`TORCH_HOME`), every
checkpoint path and (for a bundle) its members, every required external-asset path, and each model's
checkout directory. 73 entries, **`stat`/`setting` only — no hash, no probe** (that constraint is why:
a full rebuild measured **98.43 s** cold, 51.31 s warm, dominated by `preflight_blockers` = 44.62 s
of it; `tme` alone is 20.88 s warm).

```
stamp cost           0.016 s   (73 entries)
stamp cost (warm)    0.006 s   (identical value)
/api/models cold    98.43 s
/api/models cached   0.0141 s  (same object returned)
```

The badge now moves in the same process, both ways, proven by
`test_a_newly_present_asset_changes_the_badge_in_the_same_process`:
* asset absent → `"external runtime asset file missing: torchvision ResNet-50 ..."`
* asset created → `"external runtime asset checksum mismatch: ..."` (a DIFFERENT status, so the
  assertion cannot pass on an incidental reword)
* asset deleted → `"external runtime asset file missing: ..."` again

### The gates stay

`test_the_launch_gate_still_refuses_a_blocked_model`: `POST /api/jobs` → **409** with `blockers`, and
`main._job_launch_guard(_Job())` raises. `test_a_run_with_results_stays_openable_while_a_new_run_is_refused`:
in ONE process, `get_composition_run_detail` returns `execution_status COMPLETED` with real metrics
for a model that `POST /api/jobs` refuses — both directions, so the cache cannot be fixed by
weakening the refusal, and a blocked model cannot hide a run that already has numbers.

---

## 3. `statistics.query_count` — 6003 beside a denominator of 5

### Before

`composition.py:1372` passed `len(subset_queries)` (the release's `full` subset size) as
`query_count`. Live, `GET /api/composition/runs/1eee4c0adbc93f7708205a5e/queries?subset=full&condition=multimodal&limit=1&top_k=10`
→ HTTP 200, 2 081 B:

```json
"statistics": {"conditions": {"image_only_empty_text": {"denominator": 5, "scored_count": 5, ...},
                              "multimodal":            {"denominator": 5, "scored_count": 5, ...},
                              "text_only_black_image": {"denominator": 5, "scored_count": 5, ...}},
               "query_count": 6003}
"statistics_whole_run": {..., "query_count": 5}
```

### After

`query_count` is now derived from the queries actually ranked, so it cannot contradict the
denominators it sits beside. Measured against stubbed artifacts in
`test_statistics_query_count_describes_the_queries_actually_ranked` (5 ranked of 6003 in `full`) and
`test_a_full_run_still_reports_its_whole_cohort` (3 of 3):

| shape | `statistics.query_count` before | after | `statistics_whole_run.query_count` |
|---|---|---|---|
| 5-query smoke, release cohort 6003 | 6003 | **5** | 5 |
| 3-query full run | (would be 3) | **3** | 3 |

---

## 4. `subset_provenance` aliasing — both call orders, bytes + sha256

### Before

`read_summary_file` cached the parsed dict by path and returned the SAME object to every caller, while
`list_composition_summaries` decorated it in place (`decorate_summary` → `annotate_duplicate_attempts`
→ `decorate_run_rows`). Reproduced in one process through the real route handlers
(`main.composition_run` / `main.composition_runs`), run id `7b1fa85007bffdab931f7f19`:

```
1 cold detail    (79123, 'a0e4e9917aff4f3d9673d4c59a723a61dbad6bd2857887df9ec91308e20bf657')  prov: False
2 list rows      52, every row carries subset_provenance
3 warm detail    (79946, '4635de32b3297ad84127f0ee8ea270d6ec1a5f21c7b4df7893640f38ff2caf61')  prov: True
cold is warm object: True          <-- the defect, stated as identity
```

Difference: **+823 B**, `sha256` `a0e4e99…` → `4635de3…`.

### After

`read_summary_file` returns `copy.deepcopy` of the cached document (and stores an undecorated copy), so
each caller mutates only its own. Same process, same routes, same run:

```
cold detail (no list yet)   (80104, 'bc165e509d3f027b015c9ea56c4042aaca99302cb69a6b36cb793742abf6c62a')  prov: False
after list                  (80104, 'bc165e509d3f027b015c9ea56c4042aaca99302cb69a6b36cb793742abf6c62a')  prov: False
list first, then detail     (80104, 'bc165e509d3f027b015c9ea56c4042aaca99302cb69a6b36cb793742abf6c62a')  prov: False
second detail (same order)  (80104, 'bc165e509d3f027b015c9ea56c4042aaca99302cb69a6b36cb793742abf6c62a')  identical: True
```

All four measurements are the same **80 104 B / `bc165e3…`**. `subset_provenance` is absent from the
detail payload in every order — it is a LIST-row field (`decorate_run_rows` is called only from the
list route), and absence is now a property of the route, not of request history. The 80 104 B figure
is larger than the pre-fix 79 123 B because `decorate_summary` now also publishes
`derived_metrics`/`conditions_display` (§5), which is the same on every route.

Asserted by `test_detail_is_byte_identical_in_both_call_orders` (order A: detail → list → detail;
order B: list → detail), comparing `(len, sha256)` of the canonical JSON, not the key's presence.

---

## 5. R@1 served from a verified derived record

Another worker added `recall_at_1` to `workbench/composition/metrics.py` and the derived file
`workbench/artifacts/composition/derived/recall_at_1/clvc_net_eab10106b2df8e97fb7a5ce5-retry-2+tgcir_7b1fa85007bffdab931f7f19_full.json`.
Verified independently (not trusted): both records' `derived_from.ranks.sha256` and
`derived_from.summary.sha256` match the on-disk files byte for byte (4/4 `True`).

Mechanism (`composition.py`): `derived_metric(artifacts_root, metric, run_id, attempt)` finds the
record by `(run_id, attempt)`, then **re-hashes the rank artifact** and refuses on any mismatch. The
attempt id is read from the rank artifact itself (`ranks_data["attempt"]` =
`eab10106b2df8e97fb7a5ce5-retry-2`), because the summary carries no attempt field — its `run_id` is
the undecorated `eab10106b2df8e97fb7a5ce5`, so matching on the summary's own fields alone would find
nothing.

`recall_at_1` is also flattened into `metrics.recall_at_1_multimodal` the same way R@10 is, and added
to `_METRIC_KEYS` so a summary carrying only R@1 is not read as "no numbers".

Live `GET /api/composition/runs/...?subset=full`, after:

| run | MM | text | image | record status |
|---|---|---|---|---|
| tgcir `7b1fa85007bffdab931f7f19` | **6.3135099117108116** | **1.949025487256372** | **0.08329168748958854** | VERIFIED |
| clvc `eab10106b2df8e97fb7a5ce5` | **1.4106050305914344** | **0.10197144799456152** | `null` — `"no queries have an exact rank"` | VERIFIED |

All six non-null values equal the derived file's `conditions[*].recall_at_1` exactly. A refusal is a
named absence: `test_a_stale_derived_record_is_refused` rewrites the rank artifact and asserts the
block flips to `REFUSED` with `"changed since"` in the reason and the number is **not** left behind.
`test_an_artifact_recorded_recall_at_1_is_never_overwritten` asserts a published value wins.

**Bug found and fixed while verifying this**: the first cut looked the record's condition name up
directly in `metrics`, but `metrics` buckets by ROLE (`text_only`, `image_only`) while the record keys
by CONDITION (`text_only_black_image`, `image_only_empty_text`) — so 2 of 3 buckets silently received
nothing. The overlay now routes through each bucket's own `condition` field, and the test asserts all
three buckets are filled, not just `multimodal`.

---

## 6. The fourth condition: `clvc_image_encoder_only` displays

`workbench/composition/inputs.py` declares `ABLATION_CONDITIONS = ("clvc_image_encoder_only",)`
(label `"Image only (image encoder)"`). `CONDITIONS` keeps its exact meaning — `determine_query_outcome`
reads `image_only_empty_text` positionally, and widening the tuple would change every run id.

What the backend now does (`displayed_conditions`, only when the run actually scored the name):
* `conditions_display: [{id, label}]` on the run detail — protocol triple first, then the ablation.
* `conditions: [{id, label}]` on the `/queries` page payload, and `items[].ranks` carries the ablation
  key alongside the triple.
* `gap_conditions` + `gap_condition_labels` + `gap_role_reason` when `metrics.gap_conditions` exists,
  so a variant gap cannot be read as the protocol gap. Absent → no keys, byte-identical to today.

Fixtures were **not** touched: no fixture invents a condition no real run has.

Asserted by `test_an_ablation_condition_renders_its_own_labelled_row` and
`test_a_run_without_a_derived_record_or_ablation_is_unchanged`.

---

## 7. Cold/warm time to a usable run list

`GET /api/composition/runs` (the list endpoint; `main.composition_runs`), 52 rows, profiled in-process:

| | wall | bytes |
|---|---|---|
| cold (fresh process, all caches empty) | **3.216 s** | 2 129 225 |
| warm (same process, second call) | **0.286 s** | same |
| live HTTP, warm | 0.355 s / 0.338 s | |

Dominating call graph node: **`subset_provenance.decorate_run_rows` → `_read_subset`, 0.820 s of the
3.216 s (25 %)**, of which `json.loads` is 0.718 s over 283 773 decode calls across 126 per-query
files — one full JSONL parse per (run, subset) row, 78 `_subset_row` calls. Second:
`decorate_summary` 2.367 s cumulative, of which `composition_capabilities` 1.818 s (`image_availability`
1.749 s) and `load_cached_release` 1.027 s (first load). Third: `_ranks_statistics` 0.418 s.

So the total is dominated by **decoration, not by the summary scan** (`list_composition_summaries`
2.396 s cumulative including everything it calls). It is cached per file stamp, which is why warm is
7.4× faster; nothing here was changed in this pass — it is recorded as the number this defect needs
before a fix is worth designing.

---

## 8. UI checks

### Top-K 100 on a run that stored depth 50 — `4bc9893a977e501b82378e76` (HINT, Full)

`capability_detail.candidate_topk.depth = 50`, `status = "AVAILABLE"`, all three conditions.

Backend, `GET .../queries?subset=full&condition=multimodal&query_id=7%3A1&top_k=100&limit=1` → HTTP
200: `candidates.depth 50`, `top_results` length **50**, `candidate_depth 50`. Same through the vite
proxy (:5173) → 50 rows. Requesting more than was exported does not fabricate a window.

UI: **the Top-K menu never offers 100.** `candidateRequest.ts::candidateDisplayOptions(depth)` filters
the offer list to `n <= depth`, so on this run the options are `[1, 5, 10, 20, 50]`; and
`effectiveCandidateK(k, depth)` caps at `min(k, depth, 100) = 50`. The label printed is
`Đang hiển thị {shown.length} ứng viên · export lưu {depth} dòng`, and the ranks grid says
`ngoài {depth} dòng export đã lưu` when a target rank exceeds it.

**Honest: yes.** The number 100 cannot be selected where the export holds 50, and the depth is printed
on the row that would have been wrong.

Observed in a real browser (Chromium via `browser_qa.py`'s playwright, `?area=samples&view=explorer&
runs=c:4bc9893a977e501b82378e76`, 1440x900), reading the live `<select data-testid="topk">`:

```
Top-K options on depth-50 run: ['Top 1', 'Top 5', 'Top 10', 'Top 20', 'Top 50']
offers 100: False
```

### `TOPK_EXPORT_NOT_REQUESTED` — `ad477312aa57c1b9114f9e51` (TME, Full)

`GET .../runs/ad477312aa57c1b9114f9e51?subset=full`:

```json
"capability_detail": {"candidate_topk": {"available": false, "status": "TOPK_EXPORT_NOT_REQUESTED",
  "reason": "this run was produced without candidate Top-K export (enable it with --topk-depth on composition_eval)",
  "depth": null, "conditions": [], "source": null, "path": null}}
"capabilities": {"aggregate_metrics": true, "per_query_target_ranks": true, "candidate_topk": false, "image_availability": true}
```

UI: `canRequest` is false, the selector renders `<option value="unavailable">Chưa có ứng viên</option>`
and is disabled, and the panel renders `EmptyState kind="capability"` titled
`"Lượt này chưa có danh sách ứng viên"` with the backend's own reason inside a `Detail`. No placeholder
images. **Honest: yes** — a named status and a next step, never an empty success.

Observed in the browser on `runs=c:ad477312aa57c1b9114f9e51`:

```
selector count: 1   disabled: True
options: ['Chưa có ứng viên']
contains 'chưa có danh sách ứng viên': True
contains 'Chưa có ứng viên': True
```

(The literal `TOPK_EXPORT_NOT_REQUESTED` is **not** in the DOM text: the panel shows the backend's
human reason, and the status word lives in the `Detail` body. That is the intended split — the status
is vocabulary, the reason is the sentence a reader acts on.)

---

## 9. Cohort save → reopen → export (BEFORE the fix — kept as the reproduction)

The cohort API is scoped to the per-query **index** (`storage.validate_analysis_selection` →
`get_runs` over `runs`/`queries` DuckDB tables), which holds **0 rows** on this deployment
(`select count(*) from runs` → `0`), and the frontend says so rather than sending a request it knows
will 400 (`FailuresPage.tsx:84-87`).

Live probe with a CIRCUS run id, throwaway cohort id `api-probe`:

```
PUT /api/cohorts {cohort_id:"api-probe", protocol_id:"circus_fashioniq_release_v1",
                  query_ids:["7:1"], run_ids:["7b1fa85007bffdab931f7f19"], ...}
→ 400 {"error":"run_not_found","message":"One or more run IDs were not found.",
       "details":{"requested_run_ids":["7b1fa85007bffdab931f7f19"],
                  "missing_run_ids":["7b1fa85007bffdab931f7f19"]}}
GET /api/cohorts → 200 []
```

Nothing was written (`cohorts.json` does not exist; `GET /api/cohorts` returns `[]`), so there is no
partial state to clean up and no throwaway cohort to delete.

**Round-trip verdict: NOT EXERCISED, and the blocker is outside my file ownership.** A
save→reopen→export round-trip on this HEAD needs a run the cohort API can see, and no such run exists
(the index is empty; composition runs live in artifacts, not DuckDB). Fixing that would mean editing
`workbench/backend/storage.py` / the index builder, which I may not touch.

**What a browser would add over this HTTP probe:** nothing for the save path — the frontend refuses
composition-only cohorts client-side before any request, so the browser's only additional information
would be the rendered Vietnamese error text and the disabled state, which I can read from the source
(`FailuresPage.tsx:84-87`, `NotebookPage.tsx` `CohortsView` empty state) but cannot observe live
because no cohort exists to reopen.

**Patch spec for the owner of `storage.py`** (not applied): give `validate_analysis_selection` a
composition arm that resolves `c:<run_id>` keys against `backend/composition.load_run_ranks` +
the release's qids instead of the index, and have `cohort_metrics` read ranks from the same artifact;
then `FailuresPage.saveCohort`'s `compositionOnly` early return can be removed. Until then the
client-side refusal is the correct behaviour, not a bug.

---

## 9a. The four causes observed separately in one page load

`?area=experiments&view=models`, 1440x1400, real Chromium, reading `model-row-<id>`:

```
tgcir     status badge: "2 lượt trong phạm vi, đã có số đo — chọn một"   blockers: 1
          reasons: "2 lý do | external runtime asset provenance unresolved: OpenAI CLIP ViT-B/16
                    weights/cache | command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT"
clvc_net  status badge: "Chấm thiếu"                                    blockers: 1
          reasons: "3 lý do | torchvision ResNet-50 ImageNet weights missing: ... |
                    external runtime asset file missing: ..."
```

This is the separation the task asks for, on one screen:
* the **status badge** describes the RUN in scope — tgcir "có số đo" (a measured run), clvc "Chấm
  thiếu" (partial coverage, cause (c));
* the **blockers list** carries cause (a) verbatim, and it does NOT turn the row's status into
  "blocked" — tgcir is `BLOCKED_COMMAND` yet its row reads as measured-with-numbers;
* clvc's ResNet line is still `missing` here because this process predates the cache fix (§2) — the
  file is present and verified on disk, and the listing cannot see it until the unit restarts.
  That is the defect §2 removes, caught in the act.

The `R@1` column, same page load (frontend `96014e9`, backend still pre-fix):

```
line: R@1 —        (×8 rows, each carrying the em dash and a title with the reason)
```

Correct for the current process: the live JSON has no `recall_at_1` yet, and the cell says so with an
em dash instead of `0.0`. Confirmed against the live payload in the same run:

```
GET /api/composition/runs/7b1fa85007bffdab931f7f19?subset=full → 200
recall_at_1_multimodal present: False     derived_metrics present: False     conditions_display present: False
```

## 9b. Frontend: R@1 beside R@10, nulls with a reason

`EvaluationPage.tsx` renders `R@1 {fmtPercentValue(recall_at_1_multimodal)}` as a sub-line inside the
R@10 cell (not a new column — see item 4), with a `title` that carries
`derived_metrics.recall_at_1.reason` when the value is null. The expanded row detail prints
`R@1 … · R@10 … · R@50 …` plus `R@1 từ derived_record (VERIFIED)`.

A null is `fmtPercentValue`'s em dash with the reason attached, never `0.0`:
* clvc `image_only_empty_text` → `"no queries have an exact rank"` (the condition is
  `ABLATION_UNSUPPORTED`, so it has no ranks at all);
* a refused derived record → `"the rank artifact changed since this recall_at_1 record was derived …"`.

`dist/` rebuilt: `index-Lne9CV3w.js` (390.03 kB / 113.86 kB gzip), stylesheet unchanged
(`index-DecQnta9.css`). `test_frontend_staleness` (which asserts `dist/` is newer than the sources and
carries the current-source literals) plus the layout, catalog, and format suites → **44 passed**.

## 10. Tests run (CPU only, targeted)

```
workbench/envs/ctl312h/bin/python -m pytest \
  workbench/tests/test_api_models_cache_staleness.py \
  workbench/tests/test_composition_api_contracts.py \
  -p no:cacheprovider --basetemp=/tmp/api_<unique> -q
→ 3 passed   (models cache: add/remove in-process, launch gate, run-openable-while-refused)
→ 7 passed   (query_count smoke + full, both provenance orders, stale-refusal, artifact-wins,
              ablation row, unchanged-when-absent)

workbench/envs/ctl312h/bin/python -m pytest \
  workbench/tests/test_capability_subset_scoping.py workbench/tests/test_composition_topk_api.py \
  workbench/tests/test_composition_api.py workbench/tests/test_capability_composition.py \
  -p no:cacheprovider --basetemp=/tmp/api_u7 -q
→ 44 passed
```

---

## Live confirmation after restart

One restart, at the end of the pass, as scheduled: `systemctl --user restart cir-workbench-backend.service`
→ **MainPID 1476678**, `ActiveState active`, first `/api/models` 200 after ~2 polls (the first request
pays the cold rebuild). The frontend unit was **not** touched.

**The GPU CLVC Full run survived.** It was not visible when this pass began, launched while the
restart was being prepared, and is alive afterwards:

```
1477254  1735    03:01  bash workbench/scripts/gpu_job.sh … composition_eval.py --config /home/heheboiz/cir_rerun_20261109.yaml
                        --models clvc_net --subsets full --topk-depth 100
                        --conditions multimodal,text_only_black_image,clvc_image_encoder_only
1477261  1477254 03:01  workbench/envs/ctl312h/bin/python workbench/scripts/composition_eval.py …
nvidia-smi: 99 %, 1701 MiB
```

`ppid 1735` is the operator shell that started `gpu_job.sh`, not the backend, so the restart could
not reach it — confirmed by the PIDs being unchanged across the restart, not just inferred.

### 1. `/api/models` — a present ResNet-50 is no longer reported missing

```
first request after restart : 200, 86 658 B, 67.28 s   (cold rebuild)
second                      : 200, 86 658 B,  0.026 s
third                       : 200, 86 658 B,  0.017 s   (byte-identical to the second)
clvc_net checkpoints[0].blocking_reasons = ["command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT"]
tgcir    checkpoints[0].blocking_reasons = ["command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT"]
```

Both asset lines are **gone**: `torchvision ResNet-50 ImageNet weights missing: …` and
`external runtime asset file missing: …` no longer appear, and tgcir's CLIP provenance line is gone
too. What remains is exactly (a) from §1 — the pinned entrypoint — which is true and is not a claim
about composition. This is the live defect from `95-external-asset-unblock.md` closed.

Stamp cost, measured in-process on the same code: **0.016 s / 73 entries** (warm 0.006 s). The
per-request cost of the live endpoint is the second/third call above (0.017–0.026 s), which is that
stamp plus JSON serialisation — the 98 s rebuild is not on the request path.

### 2. Run detail — `recall_at_1`, `derived_metrics`, `conditions_display`

`GET /api/composition/runs/7b1fa85007bffdab931f7f19?subset=full` → **200**, 72 958 B, 1.77 s:

```
recall_at_1_multimodal: 6.3135099117108116
per-bucket: multimodal 6.3135099117108116 · text_only 1.949025487256372 · image_only 0.08329168748958854
derived_metrics.recall_at_1.status = "VERIFIED", source = "derived_record"
conditions_display = [{multimodal, "Image + text"}, {text_only_black_image, "Text only"},
                      {image_only_empty_text, "Image only"}]
execution_status COMPLETED · query_count 6003
```

`GET …/runs/eab10106b2df8e97fb7a5ce5?subset=full` → **200**, `execution_status PARTIAL`:

```
clvc r1: multimodal 1.4106050305914344 · text_only 0.10197144799456152 · image_only null
         reason "no queries have an exact rank" · derived status VERIFIED
```

All six non-null values equal `derived/recall_at_1/…json` exactly, live.

### 3. `statistics.query_count` on a smoke

`GET …/runs/1eee4c0adbc93f7708205a5e/queries?subset=full&condition=multimodal&limit=1&top_k=10` →
**200**:

```
statistics.query_count: 5      (was 6003)
denominators: image_only_empty_text 5 · multimodal 5 · text_only_black_image 5
statistics_whole_run.query_count: 5
```

### 4. `subset_provenance` — identical in both call orders

```
detail before list: (80174, 'bb7406d633bad67e0f4f6daca0bed01e921fafd5c2e0bf1da8adcbda6e359e57')  prov: False
detail after  list: (80174, 'bb7406d633bad67e0f4f6daca0bed01e921fafd5c2e0bf1da8adcbda6e359e57')  prov: False
identical: True
```

One sha256 pair, live, through HTTP. (80 174 B rather than the in-process 80 104 B because the live
payload goes through FastAPI's JSON encoder, which does not sort keys; within this process the two
orders are byte-identical, which is the property.)

### 5. A run with results still opens while a new launch is refused

```
GET  /api/composition/runs/7b1fa85007bffdab931f7f19?subset=full  → 200, COMPLETED, query_count 6003
POST /api/jobs {tgcir, tgcir_archive_unresolved, fashioniq_val_split, top_k 10} → 409
{"detail":{"error":"evaluation_blocked","message":"Official evaluation is not ready.",
  "details":{"blockers":["command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT"]}}}
```

Both directions in the same live process: the number is readable, the launch is refused.

### 6. Cohort round-trip — save → reopen → export, HTTP **and** browser

The composition arm in `storage.py` (`c:<run_id>` references) makes this work for the runs this
deployment actually has. Live, over HTTP:

```
PUT /api/cohorts {cohort_id:"rt-live-100", protocol_id:"circus_fashioniq_release_v1",
                  query_ids:["7:1","7:2","7:603"], run_ids:["c:7b1fa85007bffdab931f7f19"],
                  definition:{preset:"all",k:10,subset:"full"}}                  → 200
  saved run_ids ["c:7b1fa85007bffdab931f7f19"] · protocol circus_fashioniq_release_v1 · qids ["7:1","7:2","7:603"]
  provenance: family composition · protocol circus_fashioniq_release_v1 ·
              qid_subsets ["full","sf","validated"] ·
              conditions {image_only_empty_text 6003, multimodal 6003, text_only_black_image 6003}
              ranks sha256 188d84febd9a1ebc…
GET  /api/cohorts                                        → 200, row reopens with the same ids
GET  /api/cohorts/rt-live-100/export/{json,csv,markdown}  → 200 / 200 / 200
  json metrics: run_id 7b1fa85007bffdab931f7f19 · query_count 3 · scored_count 3 · r10 33.3333
```

Then the same path through the actual UI (Chromium, `?area=failures&view=consensus&
runs=c:7b1fa85007bffdab931f7f19`), which exercises the frontend change too:

```
save button      "Lưu tập đã lọc (6,003 query)" → feedback "Đã lưu", no console errors
saved cohort     cohort-20261112T014331 · run_ids ["c:7b1fa85007bffdab931f7f19"] · 6003 qids · subset full
reopen           ?area=notebook&view=cohorts → card `cohort-20261112T014331`,
                 "6,003 query · 1 lượt", detail shows protocol circus_fashioniq_release_v1, preset all, cutoff @K 10
                 no console errors, no failed requests
export (proxy)   :5173/api/cohorts/cohort-20261112T014331/export/{json,csv,markdown} → 200/200/200
                 json: run_ids ["c:7b1fa85007bffdab931f7f19"], protocol circus_fashioniq_release_v1,
                       6003 qids, subset full, r1 6.3135099117108116, r10 23.33833083458271
```

`r10 23.33833083458271` equals the run's own `metrics.recall_at_10` (23.33833083458271) and
`r1 6.3135099117108116` equals its served R@1 — the cohort's numbers come from the same artifact the
run reports, on 6003 real queries. **The round trip is closed end to end in the browser, not only in
curl.**

Refusal paths, live (all 400, each naming its cause):

```
unknown run   → run_not_found            {run_id: deadbeef…}
unranked qid  → query_alignment_mismatch {unknown_query_ids: ["9999:1"]}
mixed family  → mixed_data_kind          {index_run_ids, composition_references}
wrong proto   → cross_protocol           {cohort_protocol, run_protocol}
stale artifact→ cohort_artifact_changed  {saved_rank_artifact_sha256: 000…, current: 188d84f…}
```

Old cohort files still load: `test_a_cohort_saved_before_the_composition_arm_still_loads` writes a
pre-change record (no `definition.composition`) and asserts it reaches `cohort_metrics` with its
original arguments — the DuckDB arm, unchanged.

Both throwaway cohorts (`rt-live-100`, `cohort-20261112T014331`) were deleted; `GET /api/cohorts`
now returns `[]` and `cohorts.json` holds `[]`.

### 6b. The export control that did not exist

The cohort card listed saved cohorts and offered **no export at all**: `GET
/api/cohorts/{id}/export/{json,csv,markdown}` was a backend feature reachable only by curl. For the
person who saved the cohort that is indistinguishable from a missing feature, so the card now carries
the control.

`NotebookPage.tsx` `CohortRow`: three buttons (JSON / CSV / MD), each fetching the route it names and
handing the bytes to the browser as `<cohort_id>.json|csv|md`. A refusal is printed with the backend's
own sentence (`cohort_artifact_changed` and friends), not a status code. The provenance is rendered
too — the reference, `sha256` prefix, `qid_subsets` and the ranked conditions — because a cohort whose
artifact cannot be named is a number without a source.

Verified in Chromium on a throwaway `ui-export-probe`, reading the real download:

```
card text: ui-export-probe | 3 query | 1 lượt | CIRCUS composition | Xuất: | JSON | CSV …
detail has 'artifact đã kiểm': True        'sha256 188d84febd9a1ebc': True
detail has 'qid thuộc subset': True        'điều kiện đã xếp hạng': True
button json/csv/markdown: present and enabled
downloaded file: ui-export-probe.json, 1499 B
  run_ids ['c:7b1fa85007bffdab931f7f19'] · protocol circus_fashioniq_release_v1 · qids ['7:1','7:2','7:603']
  r10 33.333333333333336
console errors: none
```

Probe deleted afterwards; `cohorts.json` is `[]`. Frontend suite + cohort tests: **44 passed**.

## The outcome triple: a variant run classified every query "unscored"

**Reproduced first, live** on `da0a2162be0e159a1a456cec` (CLVC Full, conditions
`[multimodal, text_only_black_image, clvc_image_encoder_only]`),
`GET …/queries?subset=full&condition=clvc_image_encoder_only&query_id=7:1`:

```
outcome: unscored | target_rank: 12796 | unscored: False | state: retrieval_failure
counters: {'scored': 1, 'unscored': 0, …}        state_counts: {'hit': 0, 'retrieval_failure': 1, 'unscored': 0}
ranks keys: ['clvc_image_encoder_only', 'image_only_empty_text', 'multimodal', 'text_only_black_image']
```

One row, two contradictory statements about the same qid, and the counters already said `scored: 1`.
Cause: `determine_query_outcome` read `ranks.get("image_only_empty_text")` literally, got `None`,
and returned `"unscored"` — for every one of the 6,003 queries.

### The fix

`determine_query_outcome(ranks, k=10, conditions=CONDITIONS)`: the (multimodal, text, image) triple
is a PARAMETER. The default keeps every existing run's output byte-identical (asserted).

`outcome_conditions(ranks_by_condition, summary)` picks that triple, three sources strongest first:

1. `metrics.gap_conditions` on the run's summary — what the scorer itself computed the gap over;
2. otherwise the protocol triple with an ablation substituted for the image role **only when that role
   has no non-null rank at all**, by the same rule `runner.resolve_gap_conditions` applies;
3. otherwise `CONDITIONS` unchanged.

No ablation name is hard-coded: the substitution scans `ABLATION_CONDITIONS`, and a test passes a
different recorded name (`some_other_ablation`) to prove it.

Inside the function, a role **absent from the row** is a MISS (`False`), never "unknown"; a role
present but null for this qid is the one true `unscored`. Written as a tri-state `hit()` so `-1 <= k`
cannot be mistaken for a hit.

**One decision, reported under both names.** `row_unscored = q_outcome == "unscored"`; the row's
`unscored`, `unscored_kind` and the `counters.scored/unscored` pair all read that one boolean.

**One source for the filter.** `outcome_counts` on the page payload counts each row with the same
predicate that wrote `items[].outcome`, and the `outcome` filter selects on `q_outcome`, so the menu,
the filter and the rows cannot disagree.

**Only ranked conditions are served.** `served_conditions` listed the protocol names regardless of
whether the run had a table for them, so the payload carried
`ranks: {image_only_empty_text: null}` and the inspector rendered "Chưa chấm" for a condition the run
never measured. It now lists the protocol names the run has a table for, then any ablation table.

### Frontend

* `SamplesPage.tsx` badge and filter already read `item.outcome` — no local recompute, so they were
  right once the backend was. The outcome menu now also prints `outcome_counts` (a menu that listed
  all 6,003 under "Unscored" was the same defect on a second surface).
* `inspector.ts` `conditionsOf` DID recompute locally: it mapped the hard-coded `CONDITIONS` and
  looked each id up in `ranks`, so a run with no `image_only_empty_text` key got three "Chưa chấm"
  cells. It now reads the ROW's own `ranks` keys (protocol order first, then extras), and
  `clvc_image_encoder_only` has its own label.
* `EvaluationPage.tsx:394` prints `run.metrics.outcomes`, which the scorer computed over the resolved
  triple — already correct.

### Measured after (live, two restarts total: MainPID 1507997, then 1510910 for this last change)

Outcome histogram for `da0a2162be0e159a1a456cec`, live over HTTP:

```json
"outcome_counts": {"both": 1, "composition_required": 382, "image_only": 71, "text_only": 83,
                   "unresolved": 5464, "unscored": 2}
```

**6,001 scored across the five scored classes + 2 unscored = 6,003**, matching
`counters.scored 6001 / unscored 2` and the scorer's own `metrics.outcomes` (`neither_scored: 2`).
Before the fix every one of those 6,003 was in `unscored`.

One row, live:

```json
{"query_id": "7:1", "outcome": "unresolved", "target_rank": 12796,
 "unscored": false, "unscored_kind": null, "query_state": "retrieval_failure"}
"ranks": {"multimodal": 223, "text_only_black_image": 3197, "clvc_image_encoder_only": 12796}
"conditions": ["multimodal", "text_only_black_image", "clvc_image_encoder_only"]
```

Browser, `?area=samples&…&runs=c:da0a2162…&condition=clvc_image_encoder_only&query=7:1`:

```
badge: "Unresolved — mọi hạng ngoài @10"
cells: multimodal #223 · text-only #3,197 · image-only (image encoder) #12,796   (no phantom cell)
outcome menu: Composition required (382) · Unresolved (5,464) · Text-only (83) · Image-only (71)
              · Both (1) · Unscored (pipeline/mapping) (2)
no console errors
```

### `ui_smoke.py`: the assertion now demonstrably bites on this run

The first claim made here — "its run id is in the served set the session walks" — was **wrong**:
`samples` walks `usable[:6]` and `da0a2162be0e159a1a456cec` is 7th or later among 27 eligible runs, so
the assertion was passing on peer runs and had never seen the defect. Checked, not assumed.

The harness gained `--run-id` (check that run first, and only it) and `--query-id` (open that card).
**The assertion itself is unchanged**; only which run it bites on became selectable. Without
`--query-id` the pinned run's first row is an unscored qid (`7:3038`), so the scored half would still
not fire — the second attempt skipped for exactly that reason, which is why the query pin exists.

```
python workbench/frontend/tests/ui_smoke.py --url http://127.0.0.1:5173 \
    --session samples --run-id da0a2162be0e159a1a456cec --query-id 7:1
exit 0
[DOM-VERIFIED] samples Query 7:1 (clvc_net): outcome badge 'Unresolved — mọi hạng ngoài @10';
               ranks ['#223', '#3,197', '#12,796']; 3 numeric; 10 candidates
[DOM-VERIFIED] samples Query 7:1: badge 'Unresolved — mọi hạng ngoài @10' agrees with
               3 numeric ranks and 10 candidates
```

Before the fix this card rendered `outcome: unscored` → badge "Chưa chấm" beside three numbered
ranks, which is the exact contradiction the assertion exists to catch. The unpinned `--session
samples` run still passes (exit 0, 5 DOM-VERIFIED notes), so the default path is unchanged.

`--session all` still times out in its `fixture` session on this stack: that session waits for
`fashioniq_val_split` in `#protocol-select` and this deployment serves only the CIRCUS protocol. It
requires `--build-fixture`, which is what that session is for; the failure is not from this fix.

Tests: `workbench/tests/test_api_outcome_triple.py` (9), including the invariant on a variant-triple
AND a protocol-triple run, the filter returning exactly `outcome_counts[class]`, and
`test_only_ranked_conditions_are_served`. Regressions: 38 + 58 passed.

## Còn chặn

1. ~~**Cohort round-trip on this HEAD is not exercisable** (§9).~~ **CLOSED** — the composition arm
   landed in `storage.py` and the round trip is proven over HTTP and in the browser (§ "Live
   confirmation after restart" item 6). The §9 text is kept as the reproduction of the defect.
2. ~~**The live backend still serves the pre-fix code.**~~ **CLOSED** — restarted once at MainPID
   1476678; every number in the "Live confirmation after restart" section is from the live process.
   The paragraph below records what was true before that restart.
   **The live backend still served the pre-fix code.** `cir-workbench-backend.service` was not
   restarted (explicitly out of scope), so every "after" number above is from the source tree in a
   fresh process / the test suite, not from :8000. The live payloads quoted in §1, §3-before and §8 are
   the old code and remain what the running process returns until the lead schedules a restart.
   `GET /api/models` on the running process will keep serving its first document until then.
3. **`_models_state_stamp` cost scales with the checkpoint tree**: 73 entries / 0.016 s today. A
   directory carrying thousands of members would make the per-request stamp the dominant cost; the
   bundle arm iterates members on purpose (a member swap must be visible), so this is a real ceiling
   with a known upgrade path (stamp the directory's own mtime and only enumerate on change).
4. **Frontend R@1 landed (`96014e9`), but the served bundle on :5173 is still the old one.**
   `workbench/frontend/dist/assets/index-Lne9CV3w.js` is built from the current sources (the vite
   preview process serves whatever is in `dist/`, so it picks this up on its next page load — but the
   *live* backend serving :8000 still returns the pre-fix JSON, which has no `recall_at_1`, so a page
   load today renders the em dash for every run). The display is correct and tested against the real
   payload shape; it needs the backend restart in item 2 to show numbers. R@1 sits inside the R@10
   track: a twelfth declared column would push the table past its 1148 px container, and the minimum
   width is derived from that declared list (`test_frontend_layout_provenance` would fail).
5. **List-route cold cost is dominated by `subset_provenance`** (§7, 0.820 s of 3.216 s, 283 773
   JSON decodes over 126 files). Not changed; recorded so the next pass starts from a number.
