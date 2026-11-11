# 98 — Both CIRCUS composition Full runs: `clvc_net` (PARTIAL) and `tgcir` (COMPLETED)

First-ever `tgcir` **Full** run, plus the CLVC Full run that finished just before it. Both are
`--subsets full --topk-depth 100` under `/home/heheboiz/cir_rerun_20261109.yaml`, protocol
`circus_fashioniq_release_v1`, gallery `74381` (`gallery_identity_sha256`
`1db68c59d96f014cdbecf936e108653de98abef0bd9a98a8abc2fe08b5dec4b4`), `hash_verification: matched`,
`cache_verify_mode: full`. Every number below was read from the summary JSON and the candidate
JSONL, then cross-checked against the live API (§4). The live systemd units on 8000/5173 were only
read over HTTP GET; no unit was touched, no process was signalled, no lock was forced.

---

## 1. `clvc_net` Full — run `eab10106b2df8e97fb7a5ce5`, attempt `-retry-2`

Summary: `workbench/artifacts/composition/summary/clvc_net_author_drive_unresolved_eab10106b2df8e97fb7a5ce5-retry-2/full.json`
(17 029 B, mtime `2026-11-12 04:53:22.955 +0700`).

| field | value |
|---|---|
| `execution_status` | **`PARTIAL`** |
| `query_count` | 6003 (`full`; `sf` 4069, `validated` 586) |
| `smoke` | `{"label": null, "limited": false, "query_limit": null}` — a real Full run |
| `gallery_count` | 74381 |
| `model_id` / `checkpoint_id` | `clvc_net` / `author_drive_unresolved` |
| GPU | RTX 5070 Ti, model interpreter `clvc39-la` (Python 3.9.25; torch 2.7.1+cu128) |
| controller PID | 1366364 (wrapper `gpu_job.sh` 1366356); exited before 2026-11-12T05:02:31+07:00 |
| wall time | log birth `04:22:15.752` → summary mtime `04:53:22.956` = **1867.2 s (31 min 7 s)** |
| images encoded | 92 390 = 74 381 gallery + 3 × 6003 queries → **0.0202 s/image, 49.5 image/s** |
| peak VRAM | **not sampled** — no `nvidia-smi` sample was taken during this run (it was already in flight when this session started) |

Condition status (artifact and API identical):

| condition | `status` | `unscored_count` | `fallback_used` | `category_scope_failure_count` |
|---|---|---|---|---|
| `multimodal` | `PARTIAL` | 119 | false | 119 |
| `text_only_black_image` | `PARTIAL` | 119 | false | 119 |
| `image_only_empty_text` | `ABLATION_UNSUPPORTED` | — | — | — |

`image_only_empty_text` reason, verbatim: "CLVC-Net cannot encode an empty edit:
TextLSTMModel.forward tokenises to zero tokens and indexes x[0][0] (text_model.py:84-86), which
raises IndexError for every row, so image_only_empty_text is unrepresentable in this model's text
tower rather than merely unsupported by the adapter".

Metrics:

| condition | R@1 | R@10 | R@50 | mrr | ndcg_full | coverage | scored/query |
|---|---|---|---|---|---|---|---|
| `multimodal` | **not produced** | 7.256968048946295 | 18.303874915023794 | 0.03594722775284061 | 0.1495349414930845 | 0.980176578377478 | 5884/6003 |
| `text_only_black_image` | **not produced** | 1.3936097892590076 | 4.46974847042828 | 0.006375664231695533 | 0.09954839186850512 | 0.980176578377478 | 5884/6003 |
| `image_only_empty_text` | — | `null` | `null` | `null` | `null` | 0.0 | 0/6003 |

The 119 unscored = 1.98 % of 6003. `adapter_query_scope_resolution` names them:
`adapter_routed 5884`, `still_unscoped 119`, `unscoped_qids` starting `7:2179, 7:2211, …` — the same
ambiguous-annotation class reported for LIMN (`category_resolution.unresolved_count` is 0, so these
are scope-routing gaps, not unresolvable identities). `shortcut_outcomes`:
`{"both": 0, "composition_required": 0, "image_only": 0, "neither_scored": 6003, "text_only": 0, "unresolved": 0}`
— with one condition represented by no ranks at all, no qid can carry all three, so every qid lands
in `neither_scored`.

Gap fields: `composition_gap_ndcg` `null`, `mrr_gap` `null`, `gap_cohort_count` 0,
`gap_multimodal_ndcg`/`gap_text_only_ndcg`/`gap_image_only_ndcg` all `null`,
`gap_unavailable_reason` **"no queries have exact ranks in all three conditions"**. That reason is
literally true and is the mechanism, not a phrasing: `metrics.py:126` builds the gap cohort as
`common = [qid for qid in qids if mm_ranks[qid] is not None and text_ranks[qid] is not None and
image_ranks[qid] is not None]`; `image_only_empty_text` has zero ranks, so `common` is empty,
`metrics.py:135-136` sets that reason, and `gap`/`mrr_gap` stay `None`. **A three-condition gap does
not exist for `clvc_net` because `image_only_empty_text` is a declared `ABLATION_UNSUPPORTED`, not
because the metric failed.** The two-condition difference is visible directly in nDCG:
0.1495 (multimodal) vs 0.0995 (text-only).

`topk_export`: `TOPK_EXPORT_COMPLETE_PARTIAL`, depth 100, conditions
`["multimodal", "text_only_black_image"]`, `missing_query_count` **238**, `partial: true`.
Arithmetic verified against the manifests and row counts: `multimodal.jsonl` 5884 rows of
`expected_record_count` 6003 (missing 119), `text_only_black_image.jsonl` 5884 of 6003 (missing 119);
119 + 119 = **238**. The missing windows are exactly the unscored queries — not a write failure.
`ranks/clvc_net_author_drive_unresolved.json` 430 706 B;
`candidates/` 16 003 378 B + 15 936 807 B, each row `topk` length 100.

---

## 2. `tgcir` Full — run `7b1fa85007bffdab931f7f19` (first ever tgcir Full)

Summary: `workbench/artifacts/composition/summary/tgcir_tgcir_archive_unresolved_7b1fa85007bffdab931f7f19/full.json`
(109 793 B, mtime `2026-11-12 05:16:29.257 +0700`).

| field | value |
|---|---|
| `execution_status` | **`COMPLETED`** |
| `query_count` | 6003 (`sf` 4069, `validated` 586) |
| `smoke` | `{"label": null, "limited": false, "query_limit": null}` |
| `gallery_count` | 74381 |
| `model_id` / `checkpoint_id` | `tgcir` / `tgcir_archive_unresolved` |
| GPU | RTX 5070 Ti, model interpreter `tgcir39-la` (Python 3.9.25; torch 2.7.1+cu128) |
| PID | controller **1386987**, wrapper `gpu_job.sh` **1386980**, launched by `setsid nohup` |
| start | `2026-11-12 05:07:12.401 +0700` (log birth) |
| exit / summary | `2026-11-12 05:16:29.257 +0700` |
| wall time | **556.9 s (9 min 17 s)** |
| images encoded | 92 390 (74 381 gallery + 3 × 6003 queries, one shared `global` projection) → **0.00603 s/image, 165.9 image/s** |
| peak VRAM | 2172 MiB **at the sampled window** (constant across all 22 samples; not a whole-run peak) |
| GPU sample | `timeout 22 nvidia-smi -l 1`, 22 samples: util **97–100 %** (mean 99), memory.used **2172 MiB** (min = max), power **235.02–260.04 W** (mean 257 W) |

`--batch-size` was not passed and `MAX_WAIT=3600` was used; the lock was free and the GPU empty at
launch, so `gpu_job.sh` printed `lock held, GPU empty, launching` and took the lock itself.

Condition status: **all three `COMPLETED`, `unscored_count: 0`, `fallback_used: false`,
`category_scope_failure_count: 0`.**

Metrics:

| condition | R@1 | R@10 | R@50 | mrr | ndcg_full | coverage | scored/query |
|---|---|---|---|---|---|---|---|
| `multimodal` | **not produced** | 23.33833083458271 | 42.11227719473597 | 0.12098114437415375 | 0.24702051966534347 | 1.0 | 6003/6003 |
| `text_only_black_image` | **not produced** | 6.546726636681659 | 13.65983674829252 | 0.03674930344151411 | 0.1386930775424115 | 1.0 | 6003/6003 |
| `image_only_empty_text` | **not produced** | 2.798600699650175 | 7.163085124104614 | 0.010877930517005432 | 0.10452130761687346 | 1.0 | 6003/6003 |

Gap fields — **not null for this model**: `composition_gap_ndcg` **0.43853620852911723**,
`mrr_gap` 0.08423184093263963, `gap_cohort_count` **6003**, `gap_multimodal_ndcg`
0.24702051966534347, `gap_text_only_ndcg` 0.1386930775424115, `gap_image_only_ndcg`
0.10452130761687346, `gap_unavailable_reason` `null`. Mechanism (`metrics.py:126-148`): all three
conditions have a rank for every qid, so `common` = all 6003; `gap = 1.0 - max(gap_text,
gap_image) / gap_mm` (line 148) = `1 - 0.1386930775/0.2470205197` = **0.4385362085**, reproduced to
1e-12. `shortcut_outcomes`: `{"both": 9, "composition_required": 956, "image_only": 159,
"neither_scored": 0, "text_only": 384, "unresolved": 4495}` — the four bins sum to 6003 − 4495
unresolved, and no qid is excluded.

`topk_export`: `TOPK_EXPORT_COMPLETE`, depth 100, all three conditions, `missing_query_count` **0**,
`partial: false`. Manifests agree: 6003/6003 rows per condition, `reference_eligible: true`.
`ranks/tgcir_tgcir_archive_unresolved.json` 386 801 B. Feature cache retained under
`workbench/artifacts/composition/runs/7b1fa85007bffdab931f7f19/work/tgcir/features/`
(`gallery_global.npy` 152 332 416 B + three 12 294 272 B query arrays, `global` projection
189 214 720 B) because `--topk-depth` implies `--keep-features`.

---

## 3. Side by side

| model | condition | R@1 | R@10 | R@50 | mrr | scored | status |
|---|---|---|---|---|---|---|---|
| `clvc_net` | `multimodal` | not produced | 7.256968 | 18.303875 | 0.035947 | 5884/6003 | `PARTIAL` |
| `clvc_net` | `text_only_black_image` | not produced | 1.393610 | 4.469748 | 0.006376 | 5884/6003 | `PARTIAL` |
| `clvc_net` | `image_only_empty_text` | — | — | — | — | 0/6003 | `ABLATION_UNSUPPORTED` |
| `tgcir` | `multimodal` | not produced | 23.338331 | 42.112277 | 0.120981 | 6003/6003 | `COMPLETED` |
| `tgcir` | `text_only_black_image` | not produced | 6.546727 | 13.659837 | 0.036749 | 6003/6003 | `COMPLETED` |
| `tgcir` | `image_only_empty_text` | not produced | 2.798601 | 7.163085 | 0.010878 | 6003/6003 | `COMPLETED` |

Three-condition gap: `clvc_net` not computable (ablation), `tgcir` **0.4385** over 6003 queries.

**R@1, stated as fact and not as a zero:** R@1 is **not computed at all** by this pipeline.
`ConditionMetrics` (`workbench/composition/metrics.py:38-49`) declares `recall_at_10` and
`recall_at_50` only, and `compute_condition_metrics` (lines 60-78) returns exactly those two
Recall fields; no `recall_at_1` exists anywhere in `workbench/composition/`. The one
`recall_at_1_multimodal` in the tree is `workbench/backend/composition.py:1470`, inside the
synthetic **fixture** builder (`reproduction_status: FIXTURE_DEVELOPMENT_ONLY`, value
`r10 / 2.0`), i.e. an invented placeholder for layout fixtures, not a measurement. So R@1 is
absent for both models for a structural reason and must not be read as 0.

### Why the CIRCUS composition number is not the native number

`tgcir` native macro R@10 is **49.7208** (`raw/tgcir_run3cat_gpu.json`) against **23.3383** in
composition; neither is wrong and they are not comparable, because the composition protocol ranks
each query over the whole 74 381-image CIRCUS release gallery using the single release sentence,
while the native evaluation joins the two FashionIQ training captions into one query and scores over
the native per-category split — different query text, different candidate pool, different protocol.

---

## 4. API cross-check (HTTP GET only)

| request | code | bytes | `execution_status` |
|---|---|---|---|
| `http://127.0.0.1:8000/api/composition/runs/7b1fa85007bffdab931f7f19?subset=full` | **200** | 71 954 | `COMPLETED` |
| `http://127.0.0.1:5173/api/composition/runs/7b1fa85007bffdab931f7f19?subset=full` | **200** | 71 954 | `COMPLETED` |

Both hosts return the same 71 954 bytes; per-condition `condition_status` through the API is
`{"image_only_empty_text": "COMPLETED", "multimodal": "COMPLETED", "text_only_black_image":
"COMPLETED"}`, matching the artifact. For the CLVC run,
`/api/composition/runs/eab10106b2df8e97fb7a5ce5?subset=full` answers **200** on both `:8000` and
`:5173` (21 738 B each) with `execution_status` `PARTIAL`.

One query in `multimodal`, `top_k=100`, `limit=1`, `offset=0`, qid `7:1`:

| request | code | bytes |
|---|---|---|
| `…:8000/api/composition/runs/7b1fa85007bffdab931f7f19/queries?subset=full&condition=multimodal&query_id=7:1&top_k=100&limit=1&offset=0` | **200** | 14 241 |
| `…:5173/api/composition/runs/7b1fa85007bffdab931f7f19/queries?…` (identical query) | **200** | 14 241 |

`candidates` envelope: `{"available": true, "status": "AVAILABLE", "depth": 100, "source":
"run", "source_run_id": "7b1fa85007bffdab931f7f19", "parent_run_id": null, "reason": null,
"lineage_verified": null}` — **`candidates.depth == 100`** confirmed. `items[0]`:
`query_id 7:1`, `target_rank 23`, `candidate_status AVAILABLE`, `query_state
retrieval_failure`, `unscored false`, `top_results` length **100**.

### JSONL vs API, same qid

First row of `workbench/artifacts/composition/runs/7b1fa85007bffdab931f7f19/candidates/tgcir_tgcir_archive_unresolved/multimodal.jsonl`:

```
{"qid":"7:1","target_rank":23,"topk":[[17777,0.6184663772583008], …100 entries… ]}
```

First element of `items[0].top_results` from the API (identical on `:8000` and `:5173`):

```
{"rank":1,"image_id":"mbeir_images/fashioniq_images/B004YHIRJG.jpg","gallery_position":17777,"score":0.6184663772583008}
```

| fact | JSONL | API | match |
|---|---|---|---|
| qid | `7:1` | `7:1` | yes |
| candidate id (first) | gallery position `17777` | `gallery_position` `17777` (`B004YHIRJG.jpg`) | **yes** |
| candidate rank (first) | position 0 of `topk` ⇒ rank 1 | `rank 1` | **yes** |
| score | 0.6184663772583008 | 0.6184663772583008 | yes |
| depth | 100 | `candidates.depth` 100, `top_results` 100 | yes |
| target rank | 23 | `target_rank 23` | yes |

`target_rank 23` is consistent with the qid's absence from the Top-10 but presence in the Top-50
window; it is a scored query, not an unscored one.

---

## 5. Còn chặn

- **Neither run is in flight anymore.** CLVC `1366364` and TG-CIR `1386987` both exited; check with
  `pgrep -af "composition_eval.py"` and `nvidia-smi --query-compute-apps=pid,used_memory
  --format=csv,noheader` (both currently empty), and `/tmp/cir_gpu.lock` is unlocked.
- **`image_only_empty_text` for `clvc_net` is `ABLATION_UNSUPPORTED`**, so no three-condition
  composition gap exists for that model. `gap_unavailable_reason` says so in the artifact; the
  CLVC Full result is legitimately two-condition, `execution_status: PARTIAL`, and cannot be
  raised to `COMPLETED` without teaching CLVC-Net's text tower to encode an empty edit.
- **CLVC `PARTIAL` is a real 119-query hole (1.98 %)**: `7:2179, 7:2211, 7:2218, …` are routed to no
  category, so `multimodal` and `text_only_black_image` are each scored on 5884 of 6003. Those 238
  missing candidate windows are a consequence, not a truncated write.
- **R@1 is not served and is not computed** (`ConditionMetrics` has only R@10/R@50). If R@1 is
  wanted, it has to be added to `workbench/composition/metrics.py` and the runs re-ranked from the
  stored `ranks/*.json`; the current artifacts cannot supply it, and its absence is not a zero.
- **CLVC peak VRAM was never sampled** — the run predates this session's sampling window. Only
  TG-CIR has GPU evidence (97–100 % util, 2172 MiB, mean 257 W over 22 samples).
- **Composition ≠ native.** `tgcir` 23.3383 R@10 (composition) vs 49.7208 (native macro) are
  different protocols; any table that places them in one column is wrong.
