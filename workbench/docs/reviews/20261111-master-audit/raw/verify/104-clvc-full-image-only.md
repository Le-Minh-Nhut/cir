# 104 — CLVC-Net Full under the new profile: three conditions, a computed gap, and the 117 recovered

Branch `research/cir-failure-workbench`, base HEAD `283f985` (not pushed). This file is the only
write besides nothing else: the backfill script refused (§6), so no `derived/` file was produced.
`workbench/composition/**` was frozen for the whole run and is byte-identical before/after (§0).
No unit was restarted, no lock touched or removed, no lock holder killed, no `git reset`/`clean`/
`stash`, no `--batch-size`.

Run id: **`da0a2162be0e159a1a456cec`**.

---

## 0. Preconditions, and one deviation from the written command

| check | result |
|---|---|
| `pgrep -af composition_eval.py` | empty before launch |
| `fuser /tmp/cir_gpu.lock` | free (no holder) |
| `nvidia-smi --query-compute-apps=pid,used_memory` | empty |
| `df -h /` | 90 GiB available (≥ 45 required) |
| `git status --porcelain` | the four expected untracked paths only (`CLEANUP_REPORT.md.save`, `91-*`, `97-*`, `web_evidence/e2e_post_cutover/`); none staged |
| composition freeze | 24 files under `workbench/composition` hashed before and after: identical digest `99eac514928df2a2214a71a8b9c4b60dae05477d0061645386ee38ef26651a51`, `diff` empty |

### 0.1 The command was run with `ctl312h` as the controller, not `clvc39-la` — and why

The command as written (`workbench/envs/clvc39-la/bin/python workbench/scripts/composition_eval.py …`)
is **refused in 20 s, before any encode**, with:

```
"reason": "environment gate would refuse the run [BLOCKED/UNSAFE_INTERPRETER]:
           cannot use active workbench environment",
"status": "SELECTED_BUT_BLOCKED"
```

That is not flakiness, and it is not a defect of the model: it is the interpreter gate
(`workbench/backend/runtime.py`, `check_environment_isolation`, line 257) refusing to use a
venv as *both* the orchestrator environment and the model target. Measured directly:

| invocation interpreter | target `clvc39-la/bin/python` | verdict |
|---|---|---|
| `clvc39-la/bin/python` | same path | `(False, 'cannot use active workbench environment')` — `sys.prefix` == prefix |
| `ctl312h/bin/python` | absolute path | `(True, 'ISOLATION_VERIFIED')` |

Both published Full runs in this series used the controller this way:
`/tmp/cirrus_clvc_net.log` and `/tmp/cirrus_tgcir_full.log` both open with
`[gpu_job] … workbench/envs/ctl312h/bin/python workbench/scripts/composition_eval.py …`. So the
correct controller for this repo is `ctl312h` (Python 3.12) and the model still runs on
`clvc39-la` (Python 3.9.25, torch 2.7.1+cu128) — which the artifact confirms
(`model_interpreter.path = …/envs/clvc39-la/bin/python`,
`verification_status: GPU_RUNTIME_READY`, `verified: true`, environment_fingerprint
`6b10cb3f66cb8bb68e32fcf5b51c7230fbb769ec7170a43914169a9ec6fb16d3`).

**The gate itself was not touched** (owned by another worker); this is a report, not a workaround of
a failing model. If the lead wants the literally-written command to work, the gate's same-environment
refusal has to be relaxed for the controller role — that is a `workbench/backend/**` change, out of
my ownership.

Second gate finding worth recording: `check_environment_isolation` is **cwd-relative** in its
prefix comparison. Passed a *relative* path (`workbench/envs/clvc39-la/bin/python`) it fails
`interpreter executable does not belong to its claimed prefix`; passed the same interpreter as an
**absolute** path it returns `ISOLATION_VERIFIED`. The registry stores an absolute path, so the run is
unaffected — but a relative `WORKBENCH_CLVC_PYTHON` would be refused for a reason that has nothing to
do with the environment.

The plan/selection JSON also prints `gallery_identity_sha256 0bfd43b9…` while the run summary prints
`1db68c59…`. These are two different formulas over the same gallery, not two galleries: the CLI's
`_plan` hashes `{"protocol_id", "image_ids"}`, the runner hashes
`{"protocol_id", "ordered_gallery", "candidate_names"}` (`runner.py:_gallery_fingerprint`). Verified
by computing both on one loaded release: `0bfd43b9…` and `1db68c59…` respectively, `gallery_count`
74 381 both times. `1db68c59…` equals the identity every published run carries.

### 0.2 Launch

```bash
setsid nohup env MAX_WAIT=3600 workbench/scripts/gpu_job.sh \
  workbench/envs/ctl312h/bin/python workbench/scripts/composition_eval.py \
  --config /home/heheboiz/cir_rerun_20261109.yaml --models clvc_net --subsets full \
  --topk-depth 100 --conditions multimodal,text_only_black_image,clvc_image_encoder_only \
  > /tmp/cirrus_clvc_newprofile.log 2>&1 < /dev/null &
```

* PID recorded to `/tmp/clvc_newprofile.pid`: wrapper `gpu_job.sh` **1477254**, controller
  **1477261** (the file holds 1477261; both exited at completion).
* `/tmp/cir_gpu.lock` was taken by `gpu_job.sh` itself; it printed
  `lock held, GPU empty, launching`.
* `--batch-size` **never passed**; the recorded batch is the adapter's own 32.
* start `2026-11-12 08:40:50.565 +0700`; summary mtime `2026-11-12 09:15:18.964 +0700` →
  **2068.4 s (34 min 28 s)** wall.
* images encoded: 74 381 gallery + 3 × 6003 queries = **92 390** →
  **0.02239 s/image, 44.67 image/s**.

### 0.3 GPU sample series (first time this run family has one)

`timeout 22 nvidia-smi -l 1 --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader`,
22 samples, taken during the gallery encode:

| quantity | min | max | mean |
|---|---|---|---|
| GPU util | 8 % (two samples: the between-call gaps) | 99 % | 90.7 % |
| memory used | 1701 MiB | 1701 MiB | 1701 MiB (constant — not a whole-run peak) |
| power draw | 116.55 W | 217.35 W | 205.7 W |

20 of 22 samples are 96–99 % util at 1701 MiB / ~216 W. The previous CLVC Full (`eab10106`) never had
a sample; TG-CIR's was 2172 MiB / 257 W, so this model is the lighter of the two on VRAM.

---

## 1. Headline fields — artifact, not the log

Summary: `workbench/artifacts/composition/summary/clvc_net_author_drive_unresolved_da0a2162be0e159a1a456cec/full.json`
(17 956 B). Ranks: `workbench/artifacts/composition/runs/da0a2162be0e159a1a456cec/ranks/clvc_net_author_drive_unresolved.json`
(388.4 KB).

| field | value | expected? |
|---|---|---|
| `execution_status` | `PARTIAL` | yes — 2 queries unscored, by design |
| `query_count` | **6003** (`full` 6003, `sf` 4069, `validated` 586) | yes |
| `smoke` | `{"label": null, "limited": false, "query_limit": null}` | yes, a real Full |
| `subset` | `full`; `synthetic: false` | yes |
| `gallery_count` / `gallery_identity_sha256` | 74381 / `1db68c59…` | yes, identical to `eab10106` |
| `hash_verification` / `cache_verify_mode` | `matched` / `full` | yes |
| `protocol_id` | `circus_fashioniq_release_v1` | yes |
| `failed_expectations` | `[]` | yes |

## 2. `conditions` — exactly the three requested, the protocol image name absent

```json
"conditions": ["multimodal", "text_only_black_image", "clvc_image_encoder_only"]
```

`image_only_empty_text` is **absent from `conditions` and from `condition_status`** — not
`ABLATION_UNSUPPORTED`. It was never requested and was not inherited from the protocol tuple. The
condition list is in the hashed identity (`identity["conditions"]`), so this run cannot be mistaken
for the protocol-triple run.

## 3. `condition_status` — PARTIAL, `unscored_count: 2`, no fallback

| condition | status | unscored_count | category_scope_failure_count | fallback_used |
|---|---|---|---|---|
| `multimodal` | `PARTIAL` | **2** | 2 | `false` |
| `text_only_black_image` | `PARTIAL` | **2** | 2 | `false` |
| `clvc_image_encoder_only` | `PARTIAL` | **2** | 2 | `false` |

Not 119: the scope resolver ran. Not more than 2: the two qids are `7:3038` and `7:3790` (§10), and
`category_scope_failures` is `{each condition: 2}`. The `partial` status is the mechanism, not a
decline — a run that refuses two queries is `PARTIAL` by the runner's own rule.

## 4. `adapter_query_scope_resolution` — the resolver ran inside the run

```json
{"adapter_routed": 6001, "loader_scoped": 5884, "still_unscoped": 2,
 "resolver": "workbench.composition.model_clvc_net.Adapter.query_scope",
 "unscoped_qids": ["7:3038", "7:3790"]}
```

Matches the dry run of 99 §2 exactly: 5 884 loader-scoped + 117 recovered from the caption
annotations = 6 001 routed, 2 genuinely ambiguous. The run's identity carries
`query_scope_resolution: 3a1c5a6763154870536f3586491d5bea9d87b73a9836363dc298d2accdd34d87`, the
resolver-aware digest (not the loader-only `8a5dd517…`), which is what forces a new run id instead
of `--resume` reusing `eab10106`'s all-null ranks for those 117.

Annotation provenance is in the summary and `complete: true`: the three `cap.*.val.json` files at
their pinned digests (`cap.shirt.val.json` `7b7ca379…`, matching 99 §2) plus the three
`split.*.val.json`.

## 5. `metrics` — three-condition gap, hand-recomputed

| field | value |
|---|---|
| `gap_conditions` | `["multimodal", "text_only_black_image", "clvc_image_encoder_only"]` |
| `gap_role_reason` | `"image role filled by the adapter's named ablation 'clvc_image_encoder_only': the protocol image condition 'image_only_empty_text' carries no ranks in this run"` (non-null) |
| `composition_gap_ndcg` | **0.33061712371309204** (not null) |
| `gap_cohort_count` | **6001** |
| `gap_multimodal_ndcg` | 0.14921544731411218 |
| `gap_text_only_ndcg` | 0.09988226530955799 |
| `gap_image_only_ndcg` | 0.08644584977266369 |
| `mrr_gap` | 0.029051407246235415 |
| `gap_unavailable_reason` | `null` |
| `shortcut_outcomes` | `{"both": 1, "composition_required": 382, "image_only": 71, "neither_scored": 2, "text_only": 83, "unresolved": 5464}` |
| `metrics.image_only.condition` | `clvc_image_encoder_only` |

**Arithmetic, recomputed by me from the published fields:**

```
1 - max(gap_text_only_ndcg, gap_image_only_ndcg) / gap_multimodal_ndcg
= 1 - max(0.09988226530955799, 0.08644584977266369) / 0.14921544731411218
= 1 - 0.09988226530955799 / 0.14921544731411218
= 1 - 0.6693828762869080
= 0.33061712371309204
published = 0.33061712371309204          difference = 0.0 (exact)
```

So the text-only condition is the binding one (`max` picks it), and the multimodal condition is
worth 0.33 nDCG over the best single-modality condition on the 6 001 common queries. The image-only
role is a *substitution* recorded by name — a reader cannot mistake this for the protocol gap,
because `gap_conditions` and `gap_role_reason` say which condition filled the image role.

`shortcut_outcomes` is computed over the same triple (2 unscored → `neither_scored: 2`; 5464 qids
where neither single-modality condition retrieves the target but multimodal does not either =
`unresolved`).

## 6. Per-condition metrics, and R@1

R@1 **is** present in this summary (`ConditionMetrics` gained `recall_at_1` before this run). No
backfill was required, and none was written.

| condition | scored / query | coverage | R@1 | R@10 | R@50 | mrr | ndcg_full |
|---|---|---|---|---|---|---|---|
| `multimodal` | 6001/6003 | 0.9996668332500417 | 1.3997667055490752 | 7.198800199966672 | 18.180303282786202 | 0.03564730414020548 | 0.14921544731411218 |
| `text_only_black_image` | 6001/6003 | 0.9996668332500417 | 0.11664722546242293 | 1.3997667055490752 | 4.49925012497917 | 0.006595896893970061 | 0.09988226530955799 |
| `clvc_image_encoder_only` | 6001/6003 | 0.9996668332500417 | 0.11664722546242293 | 1.1998000333277787 | 2.9495084152641224 | 0.005437399476599689 | 0.08644584977266369 |

**Independent recomputation from the rank table** (`hit_at_k` = `rank <= k`, over the 6001 non-null):

| condition | my R@1 | my R@10 | my R@50 | matches summary |
|---|---|---|---|---|
| `multimodal` | 1.3997667055 | 7.1988002 | 18.1803032828 | yes |
| `text_only_black_image` | 0.1166472255 | 1.3997667055 | 4.499250125 | yes |
| `clvc_image_encoder_only` | 0.1166472255 | 1.1998000333 | 2.9495084153 | yes |

R@1 ≤ R@10 ≤ R@50 holds in every condition. Both non-multimodal conditions hit R@1 exactly once
(R@1 equal to each other at 0.1166 = 7/6001); that is the same 7 queries, coincidentally, not a copy
bug — R@10 and R@50 differ in both.

### 6.1 The backfill script: refuses this run, for a name-mapping reason

```
$ workbench/envs/ctl312h/bin/python workbench/scripts/backfill_recall_at_1.py --run clvc_net:da0a2162be0e159a1a456cec
clvc_net/da0a2162be0e159a1a456cec: summary is not reproducible from the rank table, refusing the backfill:
  clvc_image_encoder_only.scored_count: recomputed 6001 vs published None
  clvc_image_encoder_only.coverage: recomputed 0.9996668332500417 vs published None
  clvc_image_encoder_only.recall_at_10: recomputed 1.1998000333277787 vs published None
  clvc_image_encoder_only.recall_at_50: recomputed 2.9495084152641224 vs published None
```

The question asked was whether it accepts a summary whose `gap_conditions` is non-protocol. It
does: the script never reads `gap_conditions` (`grep -c gap_conditions` → 0). What it cannot do is
map a condition name to a metrics block: `METRIC_BLOCKS` is the fixed protocol triple
(`multimodal`→`multimodal`, `text_only_black_image`→`text_only`,
`image_only_empty_text`→`image_only`), so `clvc_image_encoder_only` falls through to itself, finds
no `metrics["clvc_image_encoder_only"]` block (the role field is `metrics["image_only"]`), compares
against `None`, and refuses — correctly, by its own reproducibility rule. The two protocol conditions
would have passed. **No `derived/recall_at_1/` file was written** (the script exits 1 before the
write), and none is needed: R@1 is published in the summary.

This is a real, named residual for the backfill tooling (see §Còn chặn 3): a variant-triple run
cannot be backfilled without teaching `METRIC_BLOCKS` about `metrics.image_only.condition`.

## 7. `topk_export` — three conditions, 6 missing windows, arithmetic on the manifests

```json
{"status": "TOPK_EXPORT_COMPLETE_PARTIAL", "depth": 100, "partial": true, "requested": true,
 "conditions": ["clvc_image_encoder_only", "multimodal", "text_only_black_image"],
 "missing_query_count": 6, "reason": null}
```

`missing_query_count` is a **sum over manifests**, so I read the manifests rather than assuming:

| manifest | `expected_record_count` | `record_count` | `missing_query_count` | `missing_query_examples` |
|---|---|---|---|---|
| `clvc_image_encoder_only` | 6003 | 6001 | 2 | `7:3038`, `7:3790` |
| `multimodal` | 6003 | 6001 | 2 | `7:3038`, `7:3790` |
| `text_only_black_image` | 6003 | 6001 | 2 | `7:3038`, `7:3790` |
| **sum** | 18009 | 18003 | **6** | |

`2 + 2 + 2 = 6` = the published `missing_query_count`. Independent row count:
`wc -l` on the three `.jsonl` files is 6001 / 6001 / 6001 = **18003**. Every manifest also carries
`depth: 100`, `partial: true`, `reference_eligible: true`, `gallery_identity_sha256: 1db68c59…`
identical to the summary's, and `source: "model_run"`.

The third condition **did** get a writer: `topk_export.conditions` has three entries and
`clvc_image_encoder_only.jsonl` exists (15.6 MB). By 103 §8.5's own discriminant this is the
"all three exported" case, so the expected value is 6, not 4 — and 6 is what is published.

## 8. `batch_partition` — canonical, no fallback, batch 32

Top-level `batch_partition` and each condition's `feature_batch_fallback` agree:

| field | value (identical in all three conditions) |
|---|---|
| `partition` | **`canonical_call_window`** |
| `fallback_qids` | **`[]`** |
| `fallback_used` | `false`; `fallback_reason: null`, `reason: null` |
| `runner_batch_size` | **32** |
| `adapter_batch_size` | **32** |
| `call_window` | **4096** |
| `adapter_calls` | 3 |

`partition_policy`: "one adapter call per call_window queries; a failing call falls back to one query
at a time". No call fell back, so every query row came from one batched adapter call — the numbers
are canonical, not diagnostic. `--batch-size` was never passed; 32 is the adapter's declared value.

## 9. Before/after against `eab10106b2df8e97fb7a5ce5` (`-retry-2`)

| quantity | `eab10106` (old profile) | `da0a2162` (this run) | moved? |
|---|---|---|---|
| `execution_status` | `PARTIAL` | `PARTIAL` | same (2 unscored, by design) |
| `conditions` | mm, text, `image_only_empty_text` | mm, text, `clvc_image_encoder_only` | **yes** |
| scored / query | 5884 / 6003 | **6001 / 6003** | +117 |
| coverage | 0.980176578377478 | **0.9996668332500417** | +0.01949 |
| `still_unscoped` | 119 | **2** | −117 |
| `topk_export.missing_query_count` | 238 | **6** | −232 |
| `topk_export.conditions` | 2 | **3** | +1 |
| `composition_gap_ndcg` | `null` | **0.33061712371309204** | **computed for the first time** |
| `gap_cohort_count` | 0 | 6001 | +6001 |
| `gap_unavailable_reason` | "no queries have exact ranks in all three conditions" | `null` | resolved |

Per-condition metrics, old → new. **Every condition's numbers moved**, which is the check that the
117 queries were genuinely scored rather than silently dropped:

| condition | metric | old | new | delta |
|---|---|---|---|---|
| `multimodal` | R@10 | 7.256968048946295 | 7.198800199966672 | **−0.05816784897962268** |
| | R@50 | 18.303874915023794 | 18.180303282786202 | **−0.12357163223759215** |
| | mrr | 0.03594722775284061 | 0.03564730414020548 | **−0.0002999236126351307** |
| | ndcg_full | 0.1495349414930845 | 0.14921544731411218 | **−0.0003194941789723127** |
| `text_only_black_image` | R@10 | 1.3936097892590076 | 1.3997667055490752 | **+0.006156916290067649** |
| | R@50 | 4.46974847042828 | 4.49925012497917 | **+0.02950165455088971** |
| | mrr | 0.006375664231695533 | 0.006595896893970061 | **+0.00022023266227452765** |
| | ndcg_full | 0.09954839186850512 | 0.09988226530955799 | **+0.0003338734410528732** |
| `clvc_image_encoder_only` | R@10 | — (`null`, the old image role was an unsupported ablation) | 1.1998000333277787 | first measured |
| | R@50 | — | 2.9495084152641224 | first measured |
| | mrr | — | 0.005437399476599689 | first measured |

Direction is the honest one: multimodal *falls* slightly and text-only *rises* slightly, because the
117 recovered queries are ones the loader could not scope and the caption annotation assigns to
`shirt` (46) or `toptee` (71) — on them the caption alone retrieves better than the composed query
does. A metric identical to the old value would have meant those rows were written as `null`; none
is identical, and `scored_count` is 6001 in all three.

R@1 did not exist in the old run's summary, so there is no old value to compare — its absence there
is not a zero (98 §3).

## 10. Two queries read through the artifact

| qid | why | `multimodal` | `text_only_black_image` | `clvc_image_encoder_only` |
|---|---|---|---|---|
| **`7:2179`** | recovered by the resolver, `annotation:shirt` (dry run: stem `B0020Q11LE`, holders `["shirt"]`, `scope: "shirt"`) | **1580** | **4549** | **2100** |
| **`7:4066`** | recovered, `annotation:toptee` | **59** | **287** | **6054** |
| **`7:3038`** | still ambiguous | **`null`** | **`null`** | **`null`** |

* `7:2179` has a **non-null rank in all three conditions** — the recovery is real in the ranks, not
  only in the scope counter. It was one of the 119 all-null qids in `eab10106`.
* `7:4066` likewise: non-null in all three (and its multimodal rank 59 is a Top-50-window hit, which
  is part of why multimodal's R@50 *fell* — the recovered cohort's composed-query performance is
  weaker than the average of the 5884).
* `7:3038` is `null` in all three, and the reason is recorded per condition in the rank artifact's
  `query_failures`:

  ```
  query 7:3038: category unresolved has no adapter embedding scope ('dress', 'shirt', 'toptee');
  batch_indices=(3037,), qids=['7:3038']
  ```

  Same for `7:3790` (`batch_indices=(3789,)`), identical text in all three conditions. The
  **upstream** reason is the ambiguous annotation — `7:3038` and `7:3790` each name two categories
  for one (reference identity, joined caption text), per 99 §3 — and both appear in
  `category_resolution.ambiguous_qids` (which lists all 119 loader-ambiguous qids) while
  `ambiguous_examples` (the first 20) shows `7:2179` alongside them with the *same* two holders
  `["shirt","toptee"]`. That is the point of the resolver: identical holders, but `7:2179`'s caption
  text matches exactly one annotation row, so it is routed, while the other two match two rows and
  stay unscored. `category_resolution.unresolved_count` is `0` — no *identity* was unresolvable.

## 11. Live API (`http://127.0.0.1:8000`) — answers, and agrees

The backend was mid-restart by another worker earlier; at read time it answered. Three GETs:

| request | code | bytes |
|---|---|---|
| `/api/composition/runs/da0a2162be0e159a1a456cec?subset=full` | **200** | 22 514 |
| `…/queries?subset=full&condition=clvc_image_encoder_only&query_id=7:2179&top_k=100&limit=1&offset=0` | **200** | 14 695 |
| `…/queries?subset=full&condition=multimodal&query_id=7:3038&top_k=100&limit=1&offset=0` | **200** | 2 778 |

Run detail vs artifact — identical: `execution_status PARTIAL`; `conditions` the three requested;
`condition_status` each `PARTIAL` / `unscored_count 2` / `fallback_used false`; `metrics.gap_conditions`
the three names; `composition_gap_ndcg 0.33061712371309204`; `gap_cohort_count 6001`; per-role
`recall_at_1/10/50`, `mrr`, `scored_count 6001`, `coverage 0.9996668332500417` — every value equal to
the summary's, including the R@1 the API now serves.

Query `7:2179`, `top_k=100`:

| fact | API |
|---|---|
| `candidates` | `{"available": true, "status": "AVAILABLE", "depth": 100, "source": "run", "source_run_id": "da0a2162be0e159a1a456cec", …}` |
| `items[0]` | `query_id 7:2179`, `target_rank 2100`, `candidate_status AVAILABLE`, `query_state retrieval_failure`, `unscored false` |
| `top_results` | length **100**; rank 1 = `mbeir_images/fashioniq_images/B0020Q11LE.jpg`, `gallery_position` 5338, score 0.9999999403953552 |

`target_rank 2100` equals the rank artifact's `ranks["clvc_image_encoder_only"]["7:2179"]` = 2100, and
rank 1 is the reference image itself (`B0020Q11LE.jpg` — the query's own reference stem), which is
correct: the reference is eligible, and under image-only the reference image's own row is the query
row, so it scores ~1.0 against itself. That is the documented behaviour of the image-encoder-only
path (99 §6.3), visible here as the API's own first candidate.

Query `7:3038`: `target_rank null`, `candidate_status "NO_CANDIDATES_FOR_QUERY"`,
`query_state "unscored"`, `unscored true`, `top_results` length 0 — the artifact's `null` rank
surfaced as an explicit unscored state, not as an empty success.

## Còn chặn

1. **The literal command in the assignment cannot run.** `clvc39-la/bin/python` as the *controller*
   is refused by design (`cannot use active workbench environment`, `runtime.py:257`): the gate
   will not accept the model venv as the orchestrator environment. This run used `ctl312h` as the
   controller, exactly as both prior Full runs did (`/tmp/cirrus_clvc_net.log`,
   `/tmp/cirrus_tgcir_full.log`). Making the literal command work needs a `workbench/backend/**`
   change (controller-role exemption), which is not mine. **Reported, not worked around.**
2. **`check_environment_isolation` is path-shape dependent.** Relative path →
   `interpreter executable does not belong to its claimed prefix`; the same interpreter as an
   absolute path → `ISOLATION_VERIFIED`. Harmless today because the registry stores an absolute path;
   a relative `WORKBENCH_CLVC_PYTHON` would be refused for the wrong reason.
3. **`backfill_recall_at_1.py` cannot backfill a variant-triple run.** Its `METRIC_BLOCKS` maps only
   the protocol names, so `clvc_image_encoder_only` is compared against a non-existent
   `metrics["clvc_image_encoder_only"]` and the script refuses (exit 1, no file). It does not read
   `gap_conditions`, so the question "does it accept a non-protocol gap_conditions" is answered *yes
   — but the condition name mapping is the real blocker*. Fix (not mine, `workbench/scripts/`):
   resolve each condition's role block through `metrics.<role>.condition` rather than a fixed name
   map. Not needed for this run — the summary already publishes R@1.
4. **The image role here is a substitution, by name.** `composition_gap_ndcg = 0.3306` is CLVC's
   gap over `(multimodal, text_only_black_image, clvc_image_encoder_only)`. It is **not** the protocol
   gap and must never be placed in one column with TG-CIR's 0.4385 (which uses
   `image_only_empty_text`). `gap_conditions` and `gap_role_reason` say so in the artifact; any
   report that drops them is wrong.
5. **`image_only_empty_text` for CLVC remains `ABLATION_UNSUPPORTED`.** Nothing here weakened that
   reason (the empty-edit `IndexError` in `text_lstm` is untouched). The comparison with TG-CIR is
   therefore across *different* image conditions, not the same one under two models.
6. **`PARTIAL` is not a defect to raise to `COMPLETED`.** The two unscored qids are ambiguous
   annotations; a `COMPLETED` status would require inventing a category for one of them. The run
   publishes `unscored_count: 2`, `still_unscoped: 2`, `missing_query_count: 6`, and
   `shortcut_outcomes.neither_scored: 2` — four independent statements of the same two queries.
7. **VRAM 1701 MiB is the sampled window, not a peak.** 22 samples during the query/encode phase; a
   transient spike outside the window would not appear. The number is a floor for that phase.
