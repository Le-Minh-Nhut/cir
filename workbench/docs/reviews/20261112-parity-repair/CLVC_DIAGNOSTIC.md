# CLVC-Net dual-scope DIAGNOSTIC — `7:3038`, `7:3790`

**Status: DIAGNOSTIC ONLY. Not canonical. Never a metric.**

CLVC-Net now has the same dual-scope diagnostic LIMN already had, for the two queries the release
cannot scope. Producer: `scripts/clvc_dual_scope.py`. Raw: `raw/clvc_diag/`. Neither path is read by
`workbench/composition/metrics.py`, the runner, or any canonical artifact (proof in §5).

## 1. Reused schema (LIMN's, cited, not reinvented)

Source of truth: `workbench/docs/reviews/20261109-native15-composition-limn/scripts/t2_dual_scope.py`
and its output `limn_dual_scope.json` (12 records = 2 qids × 2 scopes × 3 conditions).

| Element | LIMN lines | What is reused |
|---|---|---|
| report skeleton | `t2_dual_scope.py:120-146` | `artifact_type`, `schema_version: 1`, `purpose`, `canonical_status{status,scored_count:6001,unscored_count:2,unscored_qids,note}`, `source_run_id`, `gallery{count,protocol_id,note}`, `conditions[]`, `scopes[]`, `queries{}` |
| per-query block | `:274-291` | `qid`, `release{index,query_txt,reference_id,positive_ids,loader_category,matched_subsets}`, `resolved_query_image_path`, `target_position_in_gallery`, `target_gallery_id`, `scopes{}` |
| per-cell block | `:305-327` | `scope`, `condition`, `query_embedding_norm`, `staged_image_path`, `model_text`, `target_rank`, `target_score`, `target_gallery_position`, `target_did`, `gallery_size_scored`, `score_min`, `score_max`, `score_at_rank_1`, `target_margin_vs_rank1`, `encode_seconds`, `top100_export` |
| top-100 export row | `:330-346` | `qid`, `scope`, `condition`, `model_text`, `query_image_path`, checkpoint sha256, `gallery_size`, `target_rank`, `target_score`, `depth`, `candidates[{rank, gallery_position, gallery_id, image_path, score, is_target}]` |
| qid identity recorded | `:105-134` | `release` block is authoritative; `target_did` looked up **by qid**, never from an outer loop's leftover `query` |
| target-is-own-query guard | `:66-83` (per cell), `:270-273`, `:349-352` (whole-report sweep) | both reused; a whole-report sweep runs before any file is written |
| rank crosscheck | `:230-268` | recompute recorded ranks of already-scored queries with the adapter's own `score`/`rank` |
| depth | `:40` `TOP_K = 100` | export depth 100; the source run's candidate manifests also declare `depth: 100` |
| file naming | `:348-352` | `<artifact>.<scope>.<condition>.jsonl`, one file per scope+condition holding **both** qids (a per-row open truncated it — that defect is not repeated) |

Qid identity is recorded in `queries.<qid>.release` plus `qid_identity_verification.<qid>` (new
block, additive: release row, resolved paths, gallery-id-at-position agreement, which conditions the
source run actually ranked, and whether the source run's cached query row is finite).

Naming/scope reused verbatim: `diagnostic`/`canonical` flags and `canonical_status` note are
**added** on top (task requirement); every other key keeps LIMN's name and meaning.

## 2. What was produced

12 cells: 2 qids × 2 scopes (`shirt`, `toptee`) × CLVC's three conditions — `multimodal`,
`text_only_black_image`, and the named ablation `clvc_image_encoder_only`
(`workbench/composition/inputs.py:11`, `:9-11`).

| qid | scope | condition | kind | target_did | target rank | score@target | score@rank 1 | margin |
|---|---|---|---|---|---|---|---|---|
| 7:3038 | shirt | multimodal | protocol | 7:3416 | 848 | 0.463620 | 0.736086 | 0.272467 |
| 7:3038 | shirt | text_only_black_image | protocol | 7:3416 | 304 | 0.477407 | 0.654882 | 0.177476 |
| 7:3038 | shirt | clvc_image_encoder_only | ablation | 7:3416 | 8702 | 0.303293 | 1.000000 | 0.696707 |
| 7:3038 | toptee | multimodal | protocol | 7:3416 | 1253 | 0.565981 | 0.736116 | 0.170134 |
| 7:3038 | toptee | text_only_black_image | protocol | 7:3416 | 1549 | 0.458470 | 0.666364 | 0.207893 |
| 7:3038 | toptee | clvc_image_encoder_only | ablation | 7:3416 | 1267 | 0.592141 | 1.000000 | 0.407859 |
| 7:3790 | shirt | multimodal | protocol | 7:42304 | 412 | 0.571143 | 0.795686 | 0.224543 |
| 7:3790 | shirt | text_only_black_image | protocol | 7:42304 | 1466 | 0.444272 | 0.757220 | 0.312948 |
| 7:3790 | shirt | clvc_image_encoder_only | ablation | 7:42304 | 5481 | 0.259081 | 1.000000 | 0.740919 |
| 7:3790 | toptee | multimodal | protocol | 7:42304 | 259 | 0.743001 | 0.842818 | 0.099816 |
| 7:3790 | toptee | text_only_black_image | protocol | 7:42304 | 568 | 0.618858 | 0.747598 | 0.128739 |
| 7:3790 | toptee | clvc_image_encoder_only | ablation | 7:42304 | 4535 | 0.445611 | 1.000000 | 0.554389 |

Every cell carries: `raw_edit_caption` (the unmodified released sentence), `resolved_category`
(the diagnostic scope) plus `resolved_category_reason` (why it is a diagnostic and not a routing
decision), `reference image id/path`, `target image id/path`, `target_score`, `score_at_rank_1`,
`rank_1_gallery_id`, `target_margin_vs_rank1`, ordered Top-100 with ids and scores, and
`diagnostic: true`, `canonical: false`, `canonical_status` naming the unscoped 6001/6003 status.
Same three flags on all 6 export files and all 12 export rows.

`target_within_export_depth` is `false` in every cell: no target is inside the Top-100 (ranks
259…8702), which is exactly why the exact rank is stored separately and never inferred from the
window (`workbench/composition/topk.py:11-16`).

### `clvc_image_encoder_only` vs `image_only_empty_text`

Kept in separate columns, never merged:

* `clvc_image_encoder_only` — COMPUTED, in its own condition column, `condition_kind: ablation`.
  Its rank-1 is the reference image itself with score ≈ 1.0 (`rank_1_gallery_id` equals the
  reference in all 4 ablation cells) — the expected signature of ranking a reference-eligible
  gallery with the reference image's own row.
* `image_only_empty_text` — **`ABLATION_UNSUPPORTED`**, recorded only under
  `unsupported_conditions`, never as a cell. Reason (verbatim from the artifact): CLVC's text tower
  tokenises an empty edit to zero tokens and indexes `x[0][0]` (`third_party/CLVCNet/text_model.py:84-86`),
  raising `IndexError` for every row (`workbench/composition/model_clvc_net.py:418-435`,
  `EMPTY_TEXT_REASON` at `:96`). Not faked.

## 3. QID identity verification — agrees

Checked per qid against three sources and reported in `qid_identity_verification`:

| Check | `7:3038` | `7:3790` |
|---|---|---|
| release row (`workbench/artifacts/circus_release/mbeir_fashioniq_task7_test.jsonl`) `query_img_path` | `B004TMGDME.jpg` | `B0079K0YBO.jpg` |
| release `pos_cand_list` | `["7:3416"]` | `["7:42304"]` |
| loader index | 3037 | 3789 |
| resolved reference path | `data/FashionIQ/resized_image/shirt/B004TMGDME.jpg` | `.../shirt/B0079K0YBO.jpg` |
| target gallery position | 13614 | 70410 |
| gallery id at that position | `.../B004D8QCN4.jpg` ✓ | `.../B00E3WVAD2.jpg` ✓ |
| target image file resolves | yes | yes |
| source run's cached query row finite | **no** (NaN by design, all 3 conditions × 2 scopes) | **no** (same) |
| source run never ranked this qid | confirmed (`ranks[…][qid] is None`; failure recorded as `category unresolved`) | confirmed |

`agrees_with_release_row: true` for both. The reference JPEG bytes are identical between the shirt
and toptee directories (same sha256), so the resolved path's category segment does not change pixels.

**Finding (no disagreement):** the two qids are absent from the run's `ranks` and `candidates` — the
candidate manifests state `record_count: 6001`, `partial: true`,
`missing_query_examples: ["7:3038","7:3790"]`. So nothing could be copied from the run: **every cell
was computed** (task item 5 needed no `NOT_COMPUTED_GPU_PENDING` — see §4).

**Export depth:** source run depth 100 (`candidates/…/clvc_image_encoder_only.manifest.json`), our
depth 100, `export_depth_provenance.matches: true`. The run holds depth 100, so this is depth 100 and
says so.

## 4. Computation, device, and the measured CPU/GPU delta — no GPU pair pending

CLVC's image path is CPU-capable; the text tower hardwires `.cuda()` at `text_model.py:96,108`. This
run was CPU-only, so those two device moves were neutralised to identity (the same seam
`docs/reviews/20261111-master-audit/scripts/m01/m01_forward_delta.py:36-37` uses). The row
expression itself is never re-implemented: `model_clvc_net.composed_rows` — the one function the GPU
worker and the CPU fixture share (`:669-684`) — produces every row, and scoring/ranking go through the
adapter's own `score`/`rank_and_topk`.

Because CPU and cuDNN kernels round differently, the artifact MEASURES the delta rather than claiming
equivalence. Recomputing six already-scored, loader-scoped queries' ranks from the cached gallery:

* `max_abs_row_delta` = **9.32e-05** (CPU image-only row vs the same image's cached GPU gallery row —
  the same expression per `test.py:116`)
* `max_rank_delta` = **10** on a 74,381-image gallery (ranks 37 515→37 525, 21 136→21 137,
  69 917→69 921, 53 248→53 240, 12 678→12 680; one query agreed exactly at rank 26)
* batch-1 vs batch-2 CPU rows differ by ≤1.2e-07, i.e. the movement is the DEVICE, not batch
  composition.

Interpretation: gallery ORDER and the tie rule reproduce (rows agree to float noise); the residual
rank movement is device arithmetic. That is precisely why this artifact is diagnostic and no cell is
compared to a canonical number. The diagnostic cells are all `device: cpu`,
`cuda_neutralised: true`, with `batch_composition.forward_batch_rows: 2`.

**`NOT_COMPUTED_GPU_PENDING`: none.** All 12 cells computed on CPU. If a future reader wants the
GPU-row twin of these cells for a like-for-like device comparison, the command would be the
production runner with the two qids given a scope — deliberately NOT written here, because that
requires a scope decision this diagnostic must not make.

## 5. Leakage proof

(a) **Outputs live under `raw/clvc_diag/` and are named distinctly.** Files:
`clvc_dual_scope.json`, `clvc_dual_scope.{shirt,toptee}.{multimodal,text_only_black_image,clvc_image_encoder_only}.jsonl`
— no `clvc_net_author_drive_unresolved` prefix, no path under `workbench/artifacts/composition/`.

(b) **Nothing reads them.** Measured, not asserted:

| grep | result |
|---|---|
| `clvc_diag` across `workbench/` excluding this directory and its producer | 0 hits (only the producer's own `__pycache__`) |
| `clvc_dual_scope` in `workbench/composition/`, `workbench/backend/`, `workbench/tests/` | 0 hits |
| `raw/` or `docs/reviews` in `workbench/composition/metrics.py` | 0 hits |
| `_review_run_rows` glob `docs/reviews/*/raw/*.json` vs our file | does **not** match (`raw/clvc_diag/x.json` is one level deeper) |
| `native_measurements._review_run_payload(our payload)` | `None` — the ingestion predicate refuses it (no `model_id`/`replay_type`/`aggregate`), so it cannot become a served measurement row either |
| rows returned by `_review_run_rows()` containing `clvc_diag` | `[]` |

(c) **Every summary/aggregate is marked.** `diagnostic: true`, `canonical: false` at top level, in
all 12 cells, and in all 12 export rows; every cell carries a `canonical_status` string stating the
qid remains unscoped in the canonical 6001/6003 metric; `leakage_guard` names the directory, the
naming, and that nothing reads it.

**No leakage risk found.** The only reachable consumer of anything under `docs/reviews/*/raw/` is
`backend/native_measurements.py`, and it is out of scope twice over (glob depth, and a structural
predicate that refuses this payload).

## 6. Provenance

* source run `da0a2162be0e159a1a456cec`, checkpoint `author_drive_unresolved`
  (`checkpoint_mapping_status: VERIFIED_METADATA`), `shirt_local_model.pt` sha256 recomputed and
  equal to the registry `expected_sha256`.
* gallery matrices are the source run's own caches, re-verified by recomputed content digest:
  `shirt fd8b18bc…b6a341`, `toptee 55a5b990…19000`. Nothing re-encoded 74,381 images.
* source run's own provenance: `hash_verification: matched`, gallery `1db68c59…4c4b4`, count 74,381.
* producer interpreter `3.9.25`, torch `2.7.1+cu128`, `device: cpu`, elapsed 44 s.
* runnable check: re-running the producer re-verifies every gallery digest, both checkpoint
  digests, both qids' release identity, all 12 cells' `target_did` (whole-report sweep before any
  write), and each export file's line count.
