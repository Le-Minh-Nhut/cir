# 90 — CIRCUS composition runs for `clvc_net` and `tgcir`: plan, smoke, full runs, API surface

Branch `research/cir-failure-workbench`, 2026-11-11/12. Owner of this file and of the run artifacts
under `workbench/artifacts/composition/` that the runner wrote. Control interpreter
`workbench/envs/ctl312h/bin/python`; model interpreters `workbench/envs/{clvc39-la,tgcir39-la}` (the
registry `adapted_runtime` blocks + `workbench.env` pins from 89).

Every GPU command ran under `workbench/scripts/gpu_job.sh` with `MAX_WAIT=3600`, one job at a time.
The live service on 8000/5173 was never restarted, never `systemctl`'d, never bound; only HTTP GET.
Protocol: **CIRCUS FashionIQ Release v1 Full** (`circus_fashioniq_release_v1`), released gallery
74 381 identities, 6003 release queries.

The profile used for every run here is `~/cir_rerun_20261109.yaml` (task-local config, outside the
repo): `checkpoint_policy: include_flagged_conditions`, absolute paths, `allow_gpu_eval: true`. It is
NOT the minimal default: without it, both models plan as `NOT_READY` (see §0.1).

A defect found while preparing the full runs — `tgcir` would have ABORTED its whole run at condition
3 after paying for the gallery encode — was fixed and committed separately; its evidence is
`90a-condition-degradation-fix.md` in this directory, commit `5cb0103`.

---

## 0.1 Step 0 — plan readiness, verbatim

Command, exactly:

```
workbench/envs/ctl312h/bin/python workbench/scripts/composition_eval.py \
  --models <model> --plan --topk-depth 100 --config ~/cir_rerun_20261109.yaml
```

`clvc_net`:

```json
{
  "adapter": "workbench.composition.model_clvc_net",
  "checkpoint_id": "author_drive_unresolved",
  "model_id": "clvc_net",
  "ready": true,
  "reason": null,
  "runtime": {
    "model_interpreter": {
      "environment_fingerprint": null,
      "path": "/home/heheboiz/data/cir/workbench/envs/clvc39-la/bin/python",
      "source": "workbench_env",
      "verification_reason": "environment verification not run in plan mode",
      "verification_status": "NOT_PROBED",
      "verified": false,
      "version": "Python 3.9.25; torch 2.7.1+cu128"
    }
  },
  "status": "READY"
}
```

`tgcir`:

```json
{
  "adapter": "workbench.composition.model_tgcir",
  "checkpoint_id": "tgcir_archive_unresolved",
  "model_id": "tgcir",
  "ready": true,
  "reason": null,
  "runtime": {
    "model_interpreter": {
      "environment_fingerprint": null,
      "path": "/home/heheboiz/data/cir/workbench/envs/tgcir39-la/bin/python",
      "source": "workbench_env",
      "verification_reason": "environment verification not run in plan mode",
      "verification_status": "NOT_PROBED",
      "verified": false,
      "version": "Python 3.9.25; torch 2.7.1+cu128"
    }
  },
  "status": "READY"
}
```

Neither model reported `SELECTED_BUT_BLOCKED`, so nothing had to be skipped.

Both plans also reported, identically:

```json
"gallery_count": 74381,
"gallery_identity_sha256": "0bfd43b9cc0162885f06c60cfc751602ee9f3bed2fb4d206ffcebe3f7b443462",
"protocol_id": "circus_fashioniq_release_v1",
"missing_assets": {"status": "extension_mismatch", ...}
```

`extension_mismatch` is the plan's own accounting of how it resolved image files, not a missing
asset: `extension_resolution_counts` = `{".png": 58966, "exact": 15415}`,
`local_resolved_by_stem_count` = 74381, `missing_gallery_by_model: {}`, `unresolvable_gallery_count:
0`. It is recorded because a reader who sees the word "mismatch" should see the counts beside it.

### The same plan WITHOUT the profile

```
"readiness": [],
"selected_models": [],
"unavailable": [{"model_id": "clvc_net", "status": "NOT_READY",
                 "reason": "no checkpoint variant records a training condition, so none can be
                            called clean under policy strict_clean. Policy include_flagged_conditions
                            runs these with the condition recorded as unknown."}]
```

i.e. the default policy refuses both models for the recorded reason (the checkpoints' training noise
is unrecorded), which is why every number below is a **flagged-condition** run, not a strict-clean one.

## 0.2 Step 1 — smoke, `clvc_net`, 5 queries, full 74 381-image gallery

```
MAX_WAIT=3600 workbench/scripts/gpu_job.sh workbench/envs/ctl312h/bin/python \
  workbench/scripts/composition_eval.py --config ~/cir_rerun_20261109.yaml \
  --models clvc_net --smoke 5 --topk-depth 100 --subsets full
```

| field | value |
|---|---|
| run_id | `889bc330d60e8cbab956944f` |
| start / end (UTC) | 2026-11-11T17:26:57Z → 2026-11-11T18:19:54Z (wall 3176.87 s, of which the 74 381-image gallery encode dominates) |
| exit code | **1** (PARTIAL by design, see `condition_status`) |
| summary | `workbench/artifacts/composition/summary/clvc_net_author_drive_unresolved_889bc330d60e8cbab956944f/full.json` |
| per-condition JSONL | `workbench/artifacts/composition/results/circus_fashioniq_release_v1/clvc_net/author_drive_unresolved/889bc330d60e8cbab956944f/full/{multimodal,text_only_black_image,image_only_empty_text}.jsonl` |
| candidates | `workbench/artifacts/composition/runs/889bc330d60e8cbab956944f/candidates/clvc_net_author_drive_unresolved/` |
| queries / gallery | 5 / 74381 |
| execution_status | `PARTIAL` |
| smoke block | `{"label": "SMOKE QUERY LIMIT; full released gallery preserved", "limited": true, "query_limit": 5}` |

Numbers, verbatim from the summary's `metrics`:

| condition | rows | scored | coverage | R@10 | R@50 | nDCG_full | MRR |
|---|---|---|---|---|---|---|---|
| `multimodal` | 5 | 5 | 1.0 | 0.0 | 40.0 | 0.14946719000742667 | 0.024285594499071063 |
| `text_only_black_image` | 5 | 5 | 1.0 | 20.0 | 20.0 | 0.26460011115611787 | 0.20022174697847878 |
| `image_only_empty_text` | 5 | **0** | **0.0** | null | null | null | null |

`composition_gap_ndcg`: `null`, `gap_unavailable_reason`: `"no queries have exact ranks in all three
conditions"`. `shortcut_outcomes`: `{"both": 0, "composition_required": 0, "image_only": 0,
"neither_scored": 5, "text_only": 0, "unresolved": 0}` — `neither_scored: 5` is the empty-text
condition's absence, counted as excluded rather than as misses.

**The three known facts, verified rather than assumed:**

1. **`image_only_empty_text` is refused per query by name, not by crashing.** 5 rows, 5 failures, one
   distinct failure kind:
   `Refused: adapter_empty_edit_unrepresentable: model=clvc_net rows=[0] count=1: … (batch:
   adapter_empty_edit_unrepresentable: model=clvc_net rows=[0, 1, 2, 3, 4] count=5: …)`.
   `condition_status.image_only_empty_text` = `{"status": "MODEL_UNAVAILABLE", "unscored_count": 5,
   "category_scope_failure_count": 0, "reason": "<the same text>"}`. The other two conditions are
   `COMPLETED` with `unscored_count: 0` and `fallback_used: false`. This smoke ran under code
   PREDATING `5cb0103`, so it reports 5 named per-query failures; after that commit the condition is
   declared unsupported and the runner skips it as `ABLATION_UNSUPPORTED` instead.
2. **CLVC native edits are `"<BOS> cap0 <AND> cap1 <EOS>"`, the release ships one sentence per
   query** (87 §5b). The adapter transports the text verbatim, so these numbers are NOT native
   feature vectors. Recorded as a property of the run; the adapter was not edited to paper over it.
3. **Batch-size dependence is real and upstream-native** (87 §3b). This run pinned it explicitly:
   `--batch-size 32`, and the recorded identity carries `adapter_batch_size: 32` / `batch_size: 32`
   beside `blas_threads_pinned: 1` (`_recorded_identity`, read from the API). The run id covers
   those values: they are hashed into `identity` (`runner.py:1755-1758`), so the same gallery at a
   different batch size mints a different run id rather than looking like a duplicate.

**Runner-fallback finding (the thing the brief asked to detect):** not observed. Both completed
conditions report `fallback_used: false`, and the per-condition JSONL has no `(batch:` suffix on any
row (that suffix appears only on a per-query fallback failure). So no batch size was mixed inside
this run — for the empty-text condition the failure is the adapter's own named refusal, raised
before any worker job is written.

## 0.3 Step 1 — smoke surface cross-check (artifact vs API vs UI data path)

| check | result |
|---|---|
| candidates present | yes — `candidates/<model>/multimodal.jsonl` 5 rows × depth 100 |
| ranks present | yes — every row carries `target_rank`; the candidate list's `rank` is 1..100 |
| candidate ids are gallery ids | yes — `image_id` resolves through the release's `candidate_names`; `7:1`'s first candidate `mbeir_images/fashioniq_images/B009ND97G8.jpg` is in the released gallery identity set, and all 100 resolve |
| summary numbers vs API | **all 11 shared `metrics` sub-objects equal**; the API's `metrics` additionally carries flattened conveniences (`recall_at_10_multimodal`, `coverage`, `outcomes`, `ndcg_*`) so the two dicts are not `==` as whole objects — recorded rather than glossed |
| query used | **`7:1`** (multimodal, `target_rank` **234**) |
| artifact vs API positions | identical, all 100 |
| artifact vs API scores | identical, all 100 |
| UI data path (`candidateRequest.ts` → `/queries?…&top_k=100&limit=1`) | positions/scores identical to artifact; `top_results` length 100 |

Exact URLs and codes (HTTP GET only):

```
GET /api/composition/runs                                      -> 200 (46 rows; the smoke run is
                                                                       correctly ABSENT: smoke is opt-in)
GET /api/composition/runs?include_smoke=true                   -> 200 (53 rows; run present)
GET /api/composition/runs/889bc330d60e8cbab956944f?subset=full -> 200 (execution_status PARTIAL)
GET /api/composition/runs/889bc330d60e8cbab956944f/queries?subset=full&condition=multimodal
    &query_id=7%3A1&top_k=100&limit=1&offset=0                 -> 200 (100 candidates)
GET /api/composition/runs/889bc330d60e8cbab956944f/candidates?subset=full
    &condition=multimodal&query_id=7%3A1&limit=100             -> 200
```

The default `/api/composition/runs` hiding the run is correct behaviour, not a defect: smoke rows are
opt-in (`include_smoke=true`), so the measured-runs list is not polluted by a 5-query limit.

`topk_export` on the smoke: `{"depth": 100, "requested": true, "status":
"TOPK_EXPORT_COMPLETE_PARTIAL", "partial": true, "missing_query_count": 5, "conditions":
["image_only_empty_text", "multimodal", "text_only_black_image"]}` — the 5 missing are exactly the
empty-text condition's rows, and the partial status names that rather than reporting a clean export.

## 0.4 Pre-flight proof that the fix makes condition 3 real (before paying for a full run)

Committed fix `5cb0103`. Proven through the REAL worker under the GPU lock, 2 query images, no
gallery (18 s):

```
ENCODED ok shape: (2, 512)
finite: True
rows unit-norm: [1.0, 1.0]
empty-text row differs from the caption row: True
worker reply corrected_first: ''
worker reply corrected: 2
corrected_digest: 83ce6a2479d8e9df016b90ece5f27037b2b0cbeecb9bedae5aaf370b4c3d06f6
```

and in the model's own interpreter, `clip.tokenize([""])` = `(1, 77)`, ids `[49406, 49407, 0, …]`,
i.e. a valid `(SOT, EOT)` sequence. So the empty edit is a real input to TG-CIR, not a substituted
one, and it no longer aborts the run.

---

## 1. Step 2 — the full runs (sequential, one GPU at a time)

Pinned for both: `--subsets full --topk-depth 100 --batch-size 32`, launched detached through
`gpu_job.sh` with `MAX_WAIT=3600`. `clvc_net` first; `tgcir` started by a chain watcher that waits for
the `clvc_net` control process to exit and refuses to start unless a `full.json` summary exists.

### `clvc_net`

| field | value |
|---|---|
| PID (gpu_job wrapper / control python) | 1272483 / 1272491 |
| start (UTC) | 2026-11-11T18:21:13Z |
| run_id | `eab10106b2df8e97fb7a5ce5` |
| log | `/tmp/cirrus_clvc_net.log` |
| argv | `--config ~/cir_rerun_20261109.yaml --models clvc_net --subsets full --topk-depth 100 --batch-size 32` |

### `tgcir`

Set to launch from `/tmp/c90/chain.sh` once `clvc_net` exits; its stamp is
`/tmp/cirrus_tgcir.stamp`, its log `/tmp/cirrus_tgcir.log`.

**Status at the time this section was written: `clvc_net` IN FLIGHT, `tgcir` NOT STARTED.** Both are
reported as in flight in `Còn chặn`, not as completed.

---

## 2. Step 3 — what the runs produced

Filled in §4 for each run that finished; §Còn chặn names any that did not.

### Why a CIRCUS-composition number is NOT the native number

The native campaign numbers for these two models are:

| model | source | dress R@1/R@10/R@50 | shirt | toptee | macro R@10 | macro R@50 | macro mean |
|---|---|---|---|---|---|---|---|
| `tgcir` | `raw/tgcir_run3cat_gpu.json` | 14.3282 / 43.3813 / 68.3689 | 22.4239 / 51.4720 / 69.8234 | 22.5905 / 54.3090 / 75.1147 | 49.720772 | 71.102319 | 60.411546 |
| `clvc_net` | `raw/clvc_run3cat_gpu.json` | 6.8418 / 29.4497 / 56.7675 | 6.4769 / 27.7723 / 55.2502 | 7.8531 / 32.8404 / 63.9980 | 30.020797 | 58.671894 | 44.346346 |

(`clvc_net`'s `measurements/clvc_run1.json` is the earlier single-run record of the same campaign;
`raw/clvc_run3cat_gpu.json` is the 3-category aggregate quoted above.)

A composition number differs from these for FOUR named reasons, none of them a bug:

1. **Different gallery.** The native evaluator ranks each query over the annotation pair-union of its
   category's val queries; CIRCUS ranks over the released 74 381-identity gallery. A larger gallery
   can only lower recall at fixed K. 87 §5b measured this alone to move a dress query from rank 10 to
   rank 2526.
2. **Different text.** Native edits are `"<BOS> cap0 <AND> cap1 <EOS>"` built from both annotation
   captions (`datasets.py:174`); the release ships one `query_txt` sentence per query. For TG-CIR the
   workbench additionally applies the pinned correction once inside the worker; for CLVC-Net the
   string is transported verbatim and is therefore not the string the model was trained to read.
3. **Different reference policy.** CIRCUS is reference-ELIGIBLE (the reference image may be returned);
   the native evaluators mask the reference row. That can only change ranks where the reference is
   competitive, and it changes the candidate window on every query in 88 §4's measurement.
4. **Three conditions vs one.** CIRCUS reports `multimodal`, `text_only_black_image` and
   `image_only_empty_text`; the native campaign reports the multimodal number only. The composition
   gap is a CIRCUS-only quantity with no native counterpart.

Therefore: **the native numbers are the reference point for "did we reproduce the paper", and the
composition numbers are a different experiment.** Putting them in the same table without these four
labels would be the error; they are labelled instead.

---

## 3. Step 4 — API cross-check for the full runs

Recorded in §4 per run, with exact URLs, codes and the query id used.

---

## 4. Results per completed run

### 4.1 `clvc_net` full run — COMPLETED (PARTIAL)

| field | value |
|---|---|
| run_id | `eab10106b2df8e97fb7a5ce5` |
| PID (gpu_job / control) | 1272483 / 1272491 |
| start / end (UTC) | 2026-11-11T18:21:13Z → 2026-11-11T19:36:09Z |
| wall time | **4489 s (1 h 14 m 56 s)** |
| images encoded | 74381 × 3 scopes = **223 143 gallery forwards**, plus 6003 × 2 conditions = **12 006 query forwards** |
| end-to-end s/img (gallery) | **0.0603 s/img** (4489 s / 74 381) — the number the campaign was missing |
| peak VRAM | **2897 MiB / 16303 MiB** (nvidia-smi, sampled during encode) |
| GPU util / power during encode | **6–10 %**, **39.9–47.9 W** of 285 W |
| batch size | default **32**, `adapter_batch_size: 32` in `_recorded_identity`, `blas_threads_pinned: 1` |
| execution_status | `PARTIAL` |
| exit code | **1 — by declared ablation, not by error**: `runner.py` treats `ABLATION_UNSUPPORTED` as `has_failure` → `PARTIAL`, and `composition_eval.py` treats `PARTIAL` as failure for exit purposes |
| queries / gallery | 6003 / 74381 |

`condition_status`, verbatim:

```json
{"image_only_empty_text": {"status": "ABLATION_UNSUPPORTED",
   "reason": "CLVC-Net cannot encode an empty edit: … text_model.py:84-86 …"},
 "multimodal":            {"status": "PARTIAL", "unscored_count": 119,
                           "category_scope_failure_count": 119, "fallback_used": false},
 "text_only_black_image": {"status": "PARTIAL", "unscored_count": 119,
                           "category_scope_failure_count": 119, "fallback_used": false}}
```

Per-condition metrics (summary `metrics`, denominator 6003):

| condition | scored | coverage | R@10 | R@50 | nDCG_full | MRR |
|---|---|---|---|---|---|---|
| `multimodal` | 5884 | 0.980176578377478 | 7.256968048946295 | 18.303874915023794 | 0.1495349414930845 | 0.03594722775284061 |
| `text_only_black_image` | 5884 | 0.980176578377478 | 1.3936097892590076 | 4.46974847042828 | 0.09954839186850512 | 0.006375664231695533 |
| `image_only_empty_text` | 0 | 0.0 | null | null | null | null |

`composition_gap_ndcg`: `null` — `"no queries have exact ranks in all three conditions"`, because
condition 3 is declared unsupported.

`topk_export`: `{"depth": 100, "requested": true, "status": "TOPK_EXPORT_COMPLETE_PARTIAL",
"partial": true, "missing_query_count": 238, "conditions": ["multimodal",
"text_only_black_image"]}` — 238 = 119 unscored queries × 2 exported conditions; the unsupported
condition is not listed at all.

Feature caches were removed after ranking (`feature_storage.arrays_removed_after_rank: true`).

**The 119 unscored queries (1.98 %) are declared policy, not a CLVC defect.** The release's own
`category_resolution.ambiguous_count: 119` (of 6003 considered, 5884 resolved) names TWO categories
for those qids; the adapter refuses to guess and the runner records a per-query reason
(`category_scope_failure_count: 119`, reasons in `query_failures`). The LIMN run `10173dc4` reports
the identical shape — `ambiguous_count: 119`, `category_scope_failures: {multimodal: 119,
text_only_black_image: 119, image_only_empty_text: 119}`, `PARTIAL` — because `model_limn.py`
deliberately refuses with `unresolved:annotation_names_two_categories`. Resolving them by inference
for CLVC alone would make the two models cover different query sets and break cross-model
comparability, so it is not done. `PARTIAL` is the correct status.

## 4.2 The decode pool never engages: a measured runner limitation (NOT fixed here)

`runner._encode_gallery:598-600` slices the gallery into `batch_size` chunks and calls
`adapter.encode_targets(...)` once per chunk, so each call receives exactly `batch_size` paths:

```
598:    for start in range(0, len(image_paths), batch_size):
599:        paths = image_paths[start:start + batch_size]
600:        encoded = _as_array(adapter.encode_targets([str(path) for path in paths], [scope] * len(paths)))
```

Each adapter worker then re-slices its (already `batch_size`-sized) input
(`model_clvc_net.py:548`, `model_tgcir.py:618`, `model_encoder.py:574`), producing
`len(batches) == 1`, and `adapters.prefetch_rows:141` takes its serial branch:

```
141:    if workers <= 1 or len(batches) <= 1:
142:        for batch in batches:
143:            yield map_fn(batch)
144:        return
```

So `WORKBENCH_DECODE_WORKERS` never has effect for gallery or query encoding in ANY of the three
adapters — the degeneracy is universal in the runner, not adapter-specific. That is the measured cause
of 6–10 % GPU utilisation at 40 W with one core decoding 74 381 images. The fix (a runner call window,
e.g. 4096 paths per `encode_targets` call with the worker still batching at 32 internally) was
specified but **not implemented or committed**: the `tgcir` full run held the GPU lock and a
run-identity-affecting edit to the runner was not landed against an in-flight run. It is the open item.

### API cross-check for the full run

Recorded from the served API (HTTP GET only) for `eab10106b2df8e97fb7a5ce5`:

```
GET /api/composition/runs/eab10106b2df8e97fb7a5ce5?subset=full                   -> 200
GET /api/composition/runs/<run>/queries?...&top_k=100&limit=1&offset=0           -> 200 (100 candidates)
GET /api/composition/runs?include_smoke=true                                     -> 200
```


## Còn chặn

1. **`tgcir` full run is IN FLIGHT at yield time.** PID 1304819 (gpu_job wrapper) / 1304827 (control),
   run_id not yet visible (the directory appears after the gallery encode begins), started
   2026-11-11T19:36:09Z, log `/tmp/cirrus_tgcir.log`, stamp `/tmp/cirrus_tgcir.stamp`. How to check:
   `pgrep -af composition_eval`, `tail /tmp/cirrus_tgcir.log`, then
   `ls -d workbench/artifacts/composition/summary/tgcir_*`.
2. **The runner call-window fix is NOT implemented.** `runner.py` is untouched (`git log -1 --
   workbench/composition/runner.py` = `f856338`, predating this session); `prefetch_rows:141` still
   short-circuits to serial. So §4.2's 0.0603 s/img is the UNIMPROVED number, and 6–10 % GPU
   utilisation remains unexplained by anything except this limitation. This is the main open item.
3. **`clvc_net`'s full-run numbers are the pre-window-fix numbers.** When the window lands, the run
   must be re-measured; the regression anchor is the gallery digest, not a sample comparison:
   `889bc330`'s `gallery_dress.npy.json` records `content_digest =
   a9bfa6ed4853bf999eb468379bdf5b760ef6dc3ba81861f3f0020ea509c30680`, shape `[74381, 2048]`, and
   `eab10106` reproduced that exact digest at stride 32. Any window change must reproduce it exactly.
4. **`clvc_net`'s empty-text condition remains unmeasurable** (`text_model.py:84-86`, 87 §5c).
   Declared `ABLATION_UNSUPPORTED` since `5cb0103`; the composition gap is therefore `null` for this
   model and cannot be filled.
5. **119 of 6003 queries (1.98 %) are unscored** because the release annotation names two categories
   for them. This matches LIMN exactly and is retained deliberately (§4.1).
6. **`--smoke` sets `seed_applied: false` and `seed: 0`.** No step of the composition path draws
   randomness, so the seed is provenance only; recorded so a reader does not read it as a control.
