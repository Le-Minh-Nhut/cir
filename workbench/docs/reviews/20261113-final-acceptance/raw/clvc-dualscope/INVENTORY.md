# CLVC-Net dual-scope diagnostic — evidence inventory for the final-acceptance check

Lane: CLVC (`clvc_net`, checkpoint `author_drive_unresolved`). Branch
`research/cir-failure-workbench`, HEAD `b4d25c7`. **No production code edited, no commit made.**

**Verdict: 12/12 cells already exist as raw evidence and are reused as-is. 0 captured, 0 blocked, no
GPU job launched.** The substance of this lane is therefore the receipt: a copy of every artifact
with re-computed digests, a machine-readable manifest, a validated 12-cell inventory, and the
measured list of what the next lane must change in the API/UI before the check can pass through it.

| | |
|---|---|
| Raw evidence (authoritative, single copy) | `workbench/docs/reviews/20261112-parity-repair/raw/clvc_diag/` — 7 files, 292 499 bytes. **No copy is kept in this directory** |
| Manifest | `raw/clvc-dualscope/manifest.json` |
| Producer (copied for provenance, not re-run) | `scripts/clvc_dual_scope.py` (sha256 `b14292329185b907ca001120e4580b55eefc47bbfa59ad7b5274e3f517b88bb9`) |
| Reader probe (new, this lane) | `scripts/probe_clvc_reader.py` → prints the five refusals below |
| GPU job launched | **none** — nothing was missing, so nothing to regenerate |

`workbench/docs/reviews/20261113-final-acceptance/LEDGER.md` does **not** exist at this HEAD
(`ls` → no such file); the only pre-existing content of that directory is
`raw/parity-repro/` from another lane. Nothing here edited or depended on it.

---

## 1. What exists — inventory

Producer: `workbench/docs/reviews/20261112-parity-repair/scripts/clvc_dual_scope.py`.
Origin artifacts: `workbench/docs/reviews/20261112-parity-repair/raw/clvc_diag/`.
**There is exactly one copy of this evidence**, at the location the producing record names. This lane
briefly carried a duplicate under `raw/clvc-dualscope/evidence/`; all seven files were verified
byte-identical (sha256, both sets below), the duplicate was deleted, and the CLVC descriptor and this
manifest now point at the authoritative path. Two copies of one artifact is how a reader ends up citing
the one that will drift.

| file | bytes | sha256 |
|---|---|---|
| `clvc_dual_scope.json` | 40 004 | `00baa12f6ca054dd2c9c7a4eacdd3bd4b74ad257342cbbc1e5c0204bdd45e214` |
| `clvc_dual_scope.shirt.multimodal.jsonl` | 42 084 | `c3ad1ce11b9a22bbc5837e5094b71788601312364e9d2379d60c69a55a301402` |
| `clvc_dual_scope.shirt.text_only_black_image.jsonl` | 42 118 | `60f027209433deb109edbb412fcb8719dbbe9224ad54ced47c7f1ddd71eca27d` |
| `clvc_dual_scope.shirt.clvc_image_encoder_only.jsonl` | 42 013 | `aa840e8682a31f7a7e27f29f8261ed6452b476071c775c9178c6d2d35f11a968` |
| `clvc_dual_scope.toptee.multimodal.jsonl` | 42 118 | `ae1ed1212e6f579bb97b0cac67fae40dddca18c496e3fafdb9b423e0588e5665` |
| `clvc_dual_scope.toptee.text_only_black_image.jsonl` | 42 123 | `47e909b85b0fe3ae28b360a5be8218eda2b791653901e270be3dfdf207458abc` |
| `clvc_dual_scope.toptee.clvc_image_encoder_only.jsonl` | 42 039 | `0a514244159fa87ff41c1e3eeff064d1d0c74c8a94be1235df6e7ed373dc3aa5` |

### Axes actually stored

| axis | values | where declared |
|---|---|---|
| queries | `7:3038`, `7:3790` | `queries{}` keys; `canonical_status.unscored_qids` |
| scopes | `shirt`, `toptee` | `scopes: ["shirt","toptee"]` |
| conditions | `multimodal`, `text_only_black_image`, `clvc_image_encoder_only` | `conditions[]` (length 3) |
| cutoffs / Top-K | exact `target_rank` on the full 74 381-image gallery, **plus** a stored Top-100 window per cell | `runtime.top_k = 100`; per row `depth: 100`, `candidates[]` length 100 |
| runs | `da0a2162be0e159a1a456cec` only | `source_run_id` |
| device / precision | `cpu`, `cuda_neutralised: true`, float32 everywhere | `runtime.device`, `runtime.torch` |

Exact cell keys: `queries["7:3038"].scopes["shirt"]["clvc_image_encoder_only"]` etc. — 2 × 2 × 3 = 12
leaf objects, each with the 28 keys quoted below.

Cell keys (**28**, verbatim from the artifact, counted in the copied bytes):
`scope, condition, diagnostic, canonical, canonical_status, condition_kind, raw_edit_caption,
resolved_category, resolved_category_reason, query_embedding_norm, staged_image_path, model_text,
device, cuda_neutralised, batch_composition, target_rank, target_score, target_gallery_position,
target_did, gallery_size_scored, score_min, score_max, score_at_rank_1, rank_1_gallery_id,
target_margin_vs_rank1, encode_seconds, target_within_export_depth, top100_export`.

Window row keys (20): `qid, scope, condition, diagnostic, canonical, model_text, raw_edit_caption,
query_image_path, reference_id, reference_image_path, target_did, target_image_path,
checkpoint_sha256, gallery_content_digest, gallery_size, target_rank, target_score,
score_at_rank_1, depth, candidates[]`; each candidate: `rank, gallery_position, gallery_id,
image_path, score, is_target`.

**Stored Top-100 window per cell: YES, all 12.** Every cell's `top100_export` names one of the six
`.jsonl` files, each file holds exactly the 2 rows of its scope+condition (one per qid), each row
holds 100 candidates at ranks 1..100. `target_within_export_depth` is `false` in all 12 cells
(ranks 259…8702), which is why the exact rank is stored separately — never inferred from the window
(`workbench/composition/topk.py:11-16`).

### The 12 cells

| # | query | scope | condition | kind | target_did | rank | score@target | score@rank1 | margin | enc s |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `7:3038` | `shirt` | `multimodal` | protocol | `7:3416` | 848 | 0.463620 | 0.736086 | 0.272467 | 4.2 |
| 2 | `7:3038` | `shirt` | `text_only_black_image` | protocol | `7:3416` | 304 | 0.477407 | 0.654882 | 0.177476 | 4.388 |
| 3 | `7:3038` | `shirt` | `clvc_image_encoder_only` | ablation | `7:3416` | 8702 | 0.303293 | 0.99999982 | 0.696707 | 0.503 |
| 4 | `7:3038` | `toptee` | `multimodal` | protocol | `7:3416` | 1253 | 0.565981 | 0.736116 | 0.170134 | 2.244 |
| 5 | `7:3038` | `toptee` | `text_only_black_image` | protocol | `7:3416` | 1549 | 0.458470 | 0.666364 | 0.207893 | 6.011 |
| 6 | `7:3038` | `toptee` | `clvc_image_encoder_only` | ablation | `7:3416` | 1267 | 0.592141 | 0.99999964 | 0.407859 | 4.577 |
| 7 | `7:3790` | `shirt` | `multimodal` | protocol | `7:42304` | 412 | 0.571143 | 0.795686 | 0.224543 | 4.2 |
| 8 | `7:3790` | `shirt` | `text_only_black_image` | protocol | `7:42304` | 1466 | 0.444272 | 0.757220 | 0.312948 | 4.388 |
| 9 | `7:3790` | `shirt` | `clvc_image_encoder_only` | ablation | `7:42304` | 5481 | 0.259081 | 0.99999970 | 0.740919 | 0.503 |
| 10 | `7:3790` | `toptee` | `multimodal` | protocol | `7:42304` | 259 | 0.743001 | 0.842818 | 0.099816 | 2.244 |
| 11 | `7:3790` | `toptee` | `text_only_black_image` | protocol | `7:42304` | 568 | 0.618858 | 0.747598 | 0.128739 | 6.011 |
| 12 | `7:3790` | `toptee` | `clvc_image_encoder_only` | ablation | `7:42304` | 4535 | 0.445611 | 0.99999994 | 0.554389 | 4.577 |

(`clvc_image_encoder_only`'s `score_at_rank_1` is 0.9999998211860657 / 0.9999996423721313 /
0.9999997019767761 / 0.9999999403953552 — the reference's own row, not exactly 1.0. Rounded above.)

### Validation performed this lane on the copied bytes (all pass)

* 12/12 `(qid, scope, condition)` keys exist in the summary; exactly one `.jsonl` row per key; 12
  rows total, 2 per file.
* Per cell: ranks contiguous `1..100`; scores non-increasing; 100 unique `gallery_position`s;
  window rank-1 score `==` summary `score_at_rank_1` **bit-exactly**; row `target_rank`/`target_score`
  `==` summary; `target_did` `==` that query's own `release.positive_ids[0]`; `target_score` inside
  `[score_min, score_max]`; `gallery_size == 74381`; `depth == 100`.
* **1200/1200** candidate gallery ids resolve through the served resolver
  (`workbench/backend/main.py:1684 resolve_image`) against `FASHIONIQ_ROOT` — so every thumbnail the
  UI would draw exists on disk. (A naive `resized_image/<scope>/<stem>.jpg` probe resolves only 279;
  the resolver's multi-root fallback is what makes it 1200. Wiring must use the resolver, not a path
  convention.)
* Checkpoint `local` digests recomputed on disk this lane and equal to the registry:
  shirt `2ba8e320b08ebb97d6558eb599cde2be228b5488250de7cb4cde8a138ff2c681`,
  toptee `d1fa9170496f73a008736723fe5329c93ff710305d0bcc79ff628822c0a91f4c`.

---

## 2. The acceptance contract, resolved from the schema

The check text: *"Use actual evidence for `7:3038` and `7:3790`, both `shirt` and `toptee`, all three
supported conditions, and each stored Top-100 window. Display reference, caption, target, rank,
score, ordered thumbnails and the diagnostic status. Verify all 12 combinations through the API."*

### Resolved 12-cell definition

```
( query,       scope,   condition )                    × 12
  7:3038  × { shirt, toptee } × { multimodal,
  7:3790                          text_only_black_image,
                                  clvc_image_encoder_only }
```

Tuple, one per cell:

1. `(7:3038, shirt,  multimodal)`
2. `(7:3038, shirt,  text_only_black_image)`
3. `(7:3038, shirt,  clvc_image_encoder_only)`
4. `(7:3038, toptee, multimodal)`
5. `(7:3038, toptee, text_only_black_image)`
6. `(7:3038, toptee, clvc_image_encoder_only)`
7. `(7:3790, shirt,  multimodal)`
8. `(7:3790, shirt,  text_only_black_image)`
9. `(7:3790, shirt,  clvc_image_encoder_only)`
10. `(7:3790, toptee, multimodal)`
11. `(7:3790, toptee, text_only_black_image)`
12. `(7:3790, toptee, clvc_image_encoder_only)`

**It is 2 queries × 2 scopes × 3 conditions = 12.** Grounded in the artifact, not preference:

* `scopes: ["shirt","toptee"]` — 2, and the per-cell key is `queries.<qid>.scopes.<scope>.<condition>`,
  so scope is an axis of the cell's identity.
* `conditions[]` — 3, and `runtime.top_k = 100` is a single global depth, not a per-cell axis.
* `queries{}` — 2 keys.
* 2 × 2 × 3 = 12 = the number of `.jsonl` rows and of leaf cells. There is no 13th.

### Where the ambiguity is — and why `category` cannot be a free axis

The phrase *"both `shirt` and `toptee`"* can be misread as *"both categories"*, which invites
`2 queries × 2 categories × 3 conditions`. That reading is **not** usable here, and the schema is
what forbids it:

1. **For these two queries, `category` and `scope` are the same axis.** They are precisely the two
   queries the release cannot route (their reference image + normalised caption pair appear in BOTH
   the `shirt` and the `toptee` FashionIQ validation captions — `canonical_status.note`,
   `tests/data/../CLVC_DIAGNOSTIC.md` §1). The adapter therefore writes per-cell
   `resolved_category` **equal to** the scope, with `resolved_category_reason: "diagnostic scope: the
   release resolver names BOTH shirt and toptee for this query and the canonical run refuses to
   pick one"`. Verified: `resolved_category == scope` in all 12 cells. There is no second category
   to cross with.
2. **`2 × 2 × 3 = 12` is already consumed**, so a third axis would yield 24. The artifact stores 12.
3. **A category axis exists elsewhere and is excluded on purpose.** The canonical run's cohorts are
   `category_scopes: ["dress","shirt","toptee"]`, and the 117 caption-routed queries (§5) DO carry a
   real single category. But those are *other* queries: the canonical run scores 6001 of them and
   these two are the residual 2 (`adapter_query_scope_resolution.unscoped_qids`). Mixing them in
   would replace the subject of the check.
4. **The parallel LIMN artifact uses the identical shape** — `limn_dual_scope.json` is also
   `2 queries × 2 scopes × 3 conditions = 12`, with the same `scopes`/`conditions` axes and the same
   `queries.<qid>.scopes.<scope>.<condition>` path. Two models agreeing on the shape is the schema
   evidence; it is not re-derived here.

**One residual ambiguity, stated rather than hidden:** *"all three supported conditions"* names
`clvc_image_encoder_only` for CLVC, but the third condition differs per model (LIMN's third is
`image_only_empty_text`). The artifact settles this explicitly: `conditions[]` lists
`clvc_image_encoder_only`, `unsupported_conditions.image_only_empty_text` exists and is
`ABLATION_UNSUPPORTED`, and `condition_kind` marks the computed one `ablation` / the other two
`protocol`. So for CLVC the third condition is the named ablation; `image_only_empty_text` is **not**
a 13th cell and cannot be produced (§4).

### Per-cell display contract (what "display" must mean, from the keys)

| field the check names | artifact source (exact key) |
|---|---|
| reference | `queries.<qid>.resolved_query_image_path`, `…release.reference_id`, per row `reference_image_path` |
| caption | `…release.query_txt`; per cell `raw_edit_caption` (the unmodified released sentence) and `model_text` (what the tower actually received). Measured: `raw_edit_caption == release.query_txt` in all 12 cells; `model_text == raw_edit_caption` in the 8 `multimodal`/`text_only_black_image` cells and `model_text == ""` in the 4 `clvc_image_encoder_only` cells — that condition transports no text (`workbench/composition/inputs.py:9-11,53-59`, `model_clvc_net.py:519`). A UI must show the reference caption AND, for the ablation, that the encoded text was empty |
| target | cell `target_did`, `target_gallery_position`, row `target_image_path` |
| rank | cell `target_rank` (exact, full gallery) — **not** the window position |
| score | cell `target_score`, plus `score_at_rank_1`, `target_margin_vs_rank1`, `score_min/max` |
| ordered thumbnails | row `candidates[0..99]` (`rank`, `gallery_id`, `score`, `is_target`) |
| diagnostic status | top-level `diagnostic: true`, `canonical: false`, `canonical_status{status: PARTIAL, scored_count: 6001, unscored_count: 2, unscored_qids}` — and the same two flags on every cell and every row |

---

## 3. The gap between (1) and (2)

**Cells missing: none.** All 12 exist, have a 100-row stored window each, and pass every
consistency check in §1. Nothing was regenerated — the "reuse, never regenerate" rule is what
decides this, and there is no missing cell to justify a run.

**What is genuinely missing is not evidence, it is the serving path.** Measured, not asserted, by
`scripts/probe_clvc_reader.py` against the current `workbench/backend/limn_diagnostic.py`:

| stage | what changes | observed result |
|---|---|---|
| S1 | `EVIDENCE_RELATIVE` → `raw/clvc_diag`, file names untouched | `available=False`, `DIAGNOSTIC_NOT_PUBLISHED` — `…/limn_dual_scope.json` not found |
| S2 | files renamed to `limn_dual_scope.*` | **REFUSED** `DiagnosticEvidenceError: dual-scope evidence has an unexpected artifact type` (`artifact_type` is `clvc_dual_scope_diagnostic`) |
| S3 | + `artifact_type` rewritten | **REFUSED** `per-scope export(s) missing: ['limn_dual_scope.shirt.image_only_empty_text.jsonl', 'limn_dual_scope.toptee.image_only_empty_text.jsonl']` — `CONDITIONS` is hardwired to LIMN's three |
| S4 | + `canonical_cohort` added | still REFUSED with the same missing-export line (the file set is checked before the cohort) |
| S5 | + third condition relabelled to `image_only_empty_text`, cohort still **absent** | **REFUSED** `dual-scope evidence does not declare which cohort its canonical counts belong to; a count that does not name its population reads as belonging to whatever page shows it` |
| S6 | + cohort + relabel + scope-level `sha256` in `checkpoints` | `status=DIAGNOSTIC_COMPLETE`, `complete=true`, `defects=0`; `load_candidates` serves `depth_stored=100`, 100 rows, 0 defects |

So the CLVC payload is structurally valid and passes the reader's full validation once the reader is
generalised; **the reader is the gap, not the artifact**. Two CLVC-specific schema deltas drive the
differences:

* `conditions` differs (a named ablation, not the protocol's `image_only_empty_text`) — the hardwired
  `CONDITIONS` tuple, the `_evidence_files` expected-name list, and `artifact_type` are LIMN-specific.
* `checkpoints.<scope>` has no single `sha256`; it records `members{local,global}` +
  `local_sha256_recomputed`. The reader's `checkpoint_summary_vs_export` comparison reads
  `checks[scope]["sha256"]`, which is absent, and the check is skipped when absent (measured: S5
  passes with 0 defects) — i.e. checkpoint agreement is **not** currently enforced for CLVC. That is
  a follow-up defect for the wiring lane, not a blocker for display: the per-cell
  `checkpoint_sha256` is present and equals `local_sha256_recomputed` (shirt) / `toptee` local digest
  in all 12 rows.

### What each notional missing cell would have needed (none was needed)

Recorded so a future lane does not re-derive it, from `CLVC_NATIVE.md` §2 and the artifact:

| item | value |
|---|---|
| runner | `workbench/docs/reviews/20261112-parity-repair/scripts/clvc_dual_scope.py` (no CLI args) |
| checkpoint pair | `author_drive_unresolved`: shirt `shirt_local_model.pt` + `shirt_global_model.pt`; toptee `toptee_local_model.pt` + `toptee_global_model.pt`, under `workbench/artifacts/checkpoint_selection/20261106T002617+0700/checkpoints/clvc/` |
| inputs | release `mbeir_fashioniq_task7_test.jsonl`; the run's cached `gallery_{shirt,toptee}.npy` (74 381 × 2048 float32); the two query JPEGs |
| device | **CPU** (gallery is cached, 2 queries encoded; text tower `.cuda()` neutralised to identity) |
| wall time | 44.07 s total recorded in `runtime.elapsed_seconds`; per-cell `encode_seconds` 0.503–6.011 s. Compare `CLVC_NATIVE.md`: a full 3-category native pass at batch 32 is 334.3 s of GPU work — two orders of magnitude more, because that re-encodes galleries |
| batch | 32 is the declared baseline and is what the source run used (`batch_partition.adapter_batch_size = 32`); batch 48 is **not** used here and is not comparable (batch 1 vs 32 moves 1548/2017 dress ranks) |
| GPU needed? | **No.** A GPU twin would change only device arithmetic — measured, not assumed: `device_delta.max_abs_cpu_vs_gpu_gallery_row` 6.3e-05…8.5e-05, and `rank_convention_crosscheck.max_rank_delta = 10` with `all_agree: false`. Those 6 cross-checked queries are the reason the artifact labels itself diagnostic |

---

## 4. Capture — nothing captured, nothing faked

**No GPU job was launched.** Pre-launch checks were run anyway and are recorded:

* `nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv` → header only, 0 compute apps.
* `pgrep -af 'gpu_job|pytest|composition_eval'` → no processes.
* `/tmp/cir_gpu.lock` exists and is unlocked (`flock -n` would acquire); lock was never bypassed,
  never deleted, and `workbench/config/workbench.env` was never touched.

**Blockers: none for the 12-cell evidence.** One condition is unproducible and is recorded as such
rather than faked:

> `image_only_empty_text` — **ABLATION_UNSUPPORTED**, not a cell.
> Reason (verbatim from `unsupported_conditions.image_only_empty_text.reason`): *"CLVC-Net cannot
> encode an empty edit: TextLSTMModel.forward tokenises to zero tokens and indexes `x[0][0]`
> (`text_model.py:84-86`), which raises `IndexError` for every row, so `image_only_empty_text` is
> unrepresentable in this model's text tower rather than merely unsupported by the adapter"*.
> Code path: `workbench/composition/model_clvc_net.py:418-435`, `EMPTY_TEXT_REASON` at `:96`.
> No failing command is quoted because **no run was attempted**: the artifact already carries the
> refusal from the source campaign, and re-running the producer would not produce a different answer
> for a structurally impossible encode. This is stated as a limitation of the record, not as a pass.

The producer was **not** re-run. Its output already equals the evidence copied here, and re-running
would (a) violate "reuse and never regenerate cells that already exist" and (b) risk byte drift with
no gain. The producer's own runnable check (re-verify galleries, checkpoints, qid identity, all 12
`target_did`, export line counts) was reproduced independently in §1.

### Reproduce / recapture commands (for the record)

```bash
# the 12 cells (CPU, ~44 s, no GPU): writes workbench/docs/reviews/20261112-parity-repair/raw/clvc_diag/
workbench/envs/clvc39-la/bin/python \
  workbench/docs/reviews/20261112-parity-repair/scripts/clvc_dual_scope.py

# what the current API reader does with them (measured refusals in §3)
workbench/envs/ctl312h/bin/python \
  workbench/docs/reviews/20261113-final-acceptance/raw/clvc-dualscope/scripts/probe_clvc_reader.py

# re-validate this lane's copy (12 cells, 0 issues) -- the check embedded in §1
workbench/envs/ctl312h/bin/python -c "
import json, os
ev='workbench/docs/reviews/20261112-parity-repair/raw/clvc_diag'
d=json.load(open(ev+'/clvc_dual_scope.json'))
for f in sorted(os.listdir(ev)):
    if not f.endswith('.jsonl'): continue
    for line in open(os.path.join(ev,f)):
        r=json.loads(line); c=d['queries'][r['qid']]['scopes'][r['scope']][r['condition']]
        assert r['target_rank']==c['target_rank'] and r['target_score']==c['target_score']
        assert [x['rank'] for x in r['candidates']]==list(range(1,101))
        assert r['candidates'][0]['score']==c['score_at_rank_1']
print('12 cells ok')"
```

Interpreter split (resolved from `workbench/registry/models.yaml` → `clvc_net.environment.python_env_var
= WORKBENCH_CLVC_PYTHON` → `workbench/envs/clvc39-la/bin/python`, version `3.9.25`, torch
`2.7.1+cu128`, confirmed this lane by running it): the controller is `ctl312h` (3.12.11) for
inventory/validation; the model env owns checkpoint unpickling and would own any recapture.

---

## 5. Also recorded, from the artifacts

### 5.1 Reference eligible in the diagnostic gallery — and it ranks 1st in the ablation

Yes, eligible: gallery is the FULL released 74 381-image set with no category filtering
(`gallery.note`), and the run measured reference eligibility (`reference_eligible: true`,
`reference_eligibility.method: "score+rank on a synthetic gallery whose reference is the exact
match"`, `reference_rank: 1`).

Consequence, measured per cell in the stored windows:

| condition | reference in Top-100 | rank of reference | score at that rank |
|---|---|---|---|
| `clvc_image_encoder_only` | 4/4 cells | **rank 1 in all 4** (positions 16809 `B004TMGDME.jpg`, 29872 `B0079K0YBO.jpg`) | 0.99999982 / 0.99999964 / 0.99999970 / 0.99999994 |
| `multimodal` | 2/4 cells | rank 17 (`7:3038`/shirt, 0.663807), rank 13 (`7:3038`/toptee, 0.706146); absent from the other two windows | — |
| `text_only_black_image` | 0/4 cells | absent from all four windows | — |

So the earlier ablation observation is reproduced exactly: **`clvc_image_encoder_only`'s rank-1 is
the reference image itself, score ≈ 1.0, in all four cells.** A UI that renders the ordered thumbnails
must not present that rank-1 tile as a successful retrieval; the artifact's `rank_1_gallery_id` names
it and `is_target` is `false` for it (0 `is_target` flags in every stored window, because every
target is far outside the Top-100).

### 5.2 Canonical coverage 6001/6003 — identical for CLVC and LIMN

| model | considered | scored | unscored | unscored qids | source |
|---|---|---|---|---|---|
| CLVC | 6003 | **6001** | 2 | `7:3038`, `7:3790` | `canonical_status`; run summary `adapter_query_scope_resolution{adapter_routed: 6001, still_unscoped: 2}`; `execution_status: PARTIAL` |
| LIMN | 6003 | **6001** | 2 | `7:3038`, `7:3790` | `limn_dual_scope.json` → `canonical_cohort{query_count: 6003, scored_count: 6001, unscored_count: 2}` |

Per-subset (both models, from the run summary / LIMN `canonical_cohort`): `full 6003`, `sf 4069`,
`validated 586`. The diagnostic never changes these counts — `canonical_status.note` says so and no
diagnostic number is folded into a metric. `canonical_status` is `PARTIAL` for both, and neither
artifact selects a scope as "the" answer.

### 5.3 The 117-query caption-routed cohort caveat

Measured, from `workbench/artifacts/composition/derived/category_routing_inputs/clvc_net_da0a2162be0e159a1a456cec_full.json`
(sha256 `72ba8d0d63430118e5d2863fde6eadbc4bf20358ec8807a89c73bd01e1bc67dc`):

* routing split: **117** queries got their category from `(reference stem, edit caption)`
  (`routing.resolver_key: "reference_stem+edit_caption"`, `resolver_routed_count: 117`);
  **5884** came from the release loader (`artifact_loader_scoped: 5884`); **2** remain unscoped —
  these two. `replay_agrees_with_artifact: true`.
* Consequence for the conditions: *"under `multimodal`, `text_only_black_image`,
  `clvc_image_encoder_only` the category of 117 of 6001 scored queries was chosen from (reference
  stem, edit caption), so a condition that removes the edit still used it for routing; cohort B
  (5884 loader-scoped queries) does not"* (`claim_narrowed`, verbatim). Code constants:
  `IMAGE_ENCODER_ONLY_LEAKS_SCOPE` (`workbench/composition/model_clvc_net.py:150`) and
  `IMAGE_ENCODER_ONLY_LEAKS_CAPTION` (`:157`).
* Size of the leak, measured not estimated: cohort A (6001) gap nDCG **0.33061712371309204**;
  cohort B (5884, `caption_free: true`) **0.33400015944039774** — a difference of
  **0.0033830357273053** (≈1.02 % of A).
* Scope of the caveat for THIS check: **neither of the two diagnostic queries is in the 117.** Both
  are the residual unscoped pair, and each cell's `resolved_category_reason` says the resolver names
  BOTH categories — i.e. they were *excluded* by the rule that routed the 117. So the caveat does
  not change any of the 12 ranks above; it changes how the *aggregate* gap may be described. A UI or
  report MUST NOT claim "routing uses only the reference stem, no text" for the whole cohort.

---

## 6. What the next lane must wire

Evidence is complete; the API/UI is not. Concretely, to pass "verify all 12 combinations through the
API":

1. **Generalise the diagnostics reader** (`workbench/backend/limn_diagnostic.py`, 655 lines) —
   currently one module hardwired to LIMN: `EVIDENCE_RELATIVE`, `CONDITIONS`, `_evidence_files()`'s
   expected file names, `artifact_type == "limn_dual_scope_diagnostic"`, and
   `canonical_cohort` required-but-absent-in-CLVC. Two defensible shapes: parameterise the module by
   a model descriptor, or add a CLVC sibling that reuses the same validation body. Do **not** copy the
   655 lines; the S6 probe shows the existing validation accepts the CLVC payload unchanged once the
   descriptor is right (0 defects, `complete: true`).
2. **Add the CLVC routes** — `main.py:1267` and `:1284` serve only
   `/api/composition/diagnostics/limn-dual-scope{,/candidates}`. CLVC needs its own pair returning the
   same payload shape, with the third condition named `clvc_image_encoder_only`.
3. **Add `canonical_cohort` to the CLVC artifact, or have the reader supply it.** The reader refuses
   a payload that does not name its cohort (measured). The counts are not in the CLVC artifact; the
   authoritative source is the run summary (`query_count_per_subset`, `query_count: 6003`,
   `adapter_query_scope_resolution`). Regenerating the artifact is the cleaner fix but was out of
   scope here (no production edits); supplying it from the run summary at serve time is the additive
   option.
4. **Handle the checkpoint block shape.** `checkpoints.<scope>` has no `sha256`; either map
   `local_sha256_recomputed` for comparison or add the key. Right now the checkpoint cross-check
   silently passes for CLVC (`checks[scope]["sha256"]` is `""` → comparison skipped), which is a
   *missing* validation, not a passing one.
5. **Render from the resolver, not a path.** All 1200 thumbnails resolve via
   `main.resolve_image(category, image_id, FASHIONIQ_ROOT)`; a `resized_image/<scope>/<stem>.jpg`
   convention resolves only 279/1200. Frontend `LimnScopeDiagnostic.tsx` already fetches
   `/api/images/{category}/{stem}` per candidate — reuse it.
6. **Carry the three display facts the check names that are easy to drop:** `reference_eligibility`
   is gallery-wide (`reference_eligible: true`) and the ablation's rank-1 IS the reference at
   score ≈1.0 (all 4 cells) — label it, do not hide it; `target_within_export_depth` is `false` in
   all 12 cells, so the exact rank must come from `target_rank` and never be inferred from the
   window; and `image_only_empty_text` is `ABLATION_UNSUPPORTED`, so the UI must show **three**
   conditions for CLVC, not LIMN's three with a different third silently substituted.
7. **Keep the diagnostic framing non-negotiable:** `diagnostic: true` / `canonical: false` on the
   payload, every cell and every row; `canonical_status` rendered beside the comparison; no scope
   chosen, averaged, or folded into 6001/6003.

---

## 7. Provenance summary (also machine-readable in `manifest.json`)

| field | value |
|---|---|
| model / checkpoint id | `clvc_net` / `author_drive_unresolved` (`VERIFIED_METADATA`, `reproduction_status: BLOCKED_CHECKPOINT_MAPPING` — metadata-level identity, not a reproduction claim) |
| upstream commit | `bd9b6889489806cb60676597880f91d918b279cb` (`workbench/registry/models.yaml`) |
| source run | `da0a2162be0e159a1a456cec`, `subset: full`, `protocol_id: circus_fashioniq_release_v1`, `execution_status: PARTIAL` |
| gallery | 74 381 images, identity sha256 `1db68c59d96f014cdbecf936e108653de98abef0bd9a98a8abc2fe08b5dec4b4`; matrices float32 `[74381, 2048]`, shirt digest `fd8b18bc…b6a341`, toptee digest `55a5b990…19000`, both re-verified `verify_mode: full` |
| precision / device | float32; `cpu`; `cuda_neutralised: true`; `cuda_available_but_unused: true` |
| batch | 32 (adapter and runner; `call_window: 4096`; 3 adapter calls per condition); diagnostic cells encode 2 rows per forward pass (`batch_composition.forward_batch_rows: 2`) |
| dataset order | queries: release/loader order, ascending index; gallery: released gallery order |
| ranking policy | `adapter.score` (L2-normalised dot) + `adapter.rank` (evaluator tie rule); Top-100 by that rank |
| top-K | 100 (= the source run's own candidate depth, `export_depth_provenance.matches: true`) |
| producer runtime | python 3.9.25, torch 2.7.1+cu128, 44.07 s |
| leakage | `leakage_guard.read_by: "nothing"`; 0 references to `clvc_diag` outside its own producer, verified in `CLVC_DIAGNOSTIC.md` §5 and re-checked (`grep -rn clvc_dual_scope workbench/ --include=*.py|*.ts|*.tsx` outside docs → 0 hits) |
