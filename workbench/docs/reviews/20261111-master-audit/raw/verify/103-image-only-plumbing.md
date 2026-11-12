# 103 — `clvc_image_encoder_only` plumbing, and the gap triple resolved by role

Branch `research/cir-failure-workbench`, base `88dd3a1` (HEAD at start, `== origin`).
Control interpreter `workbench/envs/ctl312h/bin/python`. **No GPU job was started**, no `--smoke`, no
composition run; the live units and `/tmp/cir_gpu.lock` were not touched.

Files touched (all mine): `workbench/composition/inputs.py`, `workbench/composition/metrics.py`,
`workbench/composition/runner.py`, plus new tests
`test_runner_image_only_condition.py`, `test_metrics_gap_conditions.py`,
`test_runner_gap_roles.py`, `test_runner_gap_identity.py`.
`workbench/backend/**`, `workbench/frontend/**`, `workbench/composition/model_*.py`,
`workbench/composition/data.py` and the registry YAML were **not** edited.

---

## 1. `inputs.py` — the ablation name is declared, not added to `CONDITIONS`

* `CONDITIONS` is byte-identical: `("multimodal", "text_only_black_image", "image_only_empty_text")`.
  Deliberately. Widening it would key a fourth `ABLATION_UNSUPPORTED` row into every other model's
  `mode_support`/`condition_status` and change every run id for nothing.
* `ABLATION_CONDITIONS = ("clvc_image_encoder_only",)` — opt-in only, never defaulted.
* `CONDITION_LABELS["clvc_image_encoder_only"] = "Image only (image encoder)"` — the single label
  source the frontend worker asked for.
* `CONDITION_ENCODERS = {"clvc_image_encoder_only": "encode_image_only_queries"}` with
  `DEFAULT_CONDITION_ENCODER = "encode_queries"`, exposed as `encoder_method(condition)`.
* `adapt_query_inputs` accepts the new name and returns `(image, "")` **the same PIL object**. The
  text is empty because the mapped encoder has no text parameter at all, so the string is never
  encoded; the untouched image is what makes the row equal the reference image's own gallery row up
  to batch composition.

## 2. `runner.py`

| change | where |
|---|---|
| validation accepts `CONDITIONS + ABLATION_CONDITIONS`, same message style, anything else still refuses | `run_composition`, `runner.py:1727` |
| `_condition_encoder(adapter, condition)` resolves the method from `CONDITION_ENCODERS` via `getattr`, returns `(callable, transports_text)` | new, above `_encode_queries` |
| missing method → named `AdapterEncoderMissing("adapter_encoder_missing: condition=… requires Adapter.encode_image_only_queries … refusing rather than falling back to encode_queries …")`, raised BEFORE any encode | same, and a pre-flight check in the `supported` loop |
| `encode_queries` path unchanged: same three positional arguments, same fallback call | `encode_batch` closure |
| image-only path gets `(paths, categories)` — no texts | same |
| `_stage_query_path` leaves the condition on the ORIGINAL file | already true (it only stages for `text_only_black_image`); now pinned by a test |
| condition list already in `identity`; verified it still is | `runner.py:1933` `"conditions": list(conditions)` |

The pre-flight check matters for cost: without it a run would pay for the 74 381-image gallery encode
and only then discover that condition 3 has no encoder.

## 3. `metrics.py` — parameterised, not forked

`compose(ranks, query_count=None, conditions=_CONDITIONS)` and
`shortcut_outcomes(ranks, k=SHORTCUT_K, conditions=_CONDITIONS)`:

* the SAME rule applies over the supplied triple, in `(multimodal, text, image)` role order —
  `common` = qids with a non-null rank in all three, `gap = 1 - max(second, third)/first`, and the
  three existing reason strings (`empty query cohort`, `missing condition(s): …`,
  `no queries have exact ranks in all three conditions`, `multimodal mean nDCG is zero`).
* a triple that is not three distinct names raises `ValueError`.
* `CompositionResult.gap_conditions: tuple[str, ...]` records the names, defaulting to the protocol
  triple. The role FIELDS keep their names (`multimodal`/`text_only`/`image_only`), so
  `metrics.image_only.condition` is what says which condition filled the image role.
* `shortcut_outcomes` follows the triple it is GIVEN. Left alone would have been wrong: the bins are
  defined per qid across all three roles, so a variant run reporting protocol bins would mix two
  cohorts (and `neither_scored` would double-count). Pinned by
  `test_shortcut_outcomes_follow_the_triple_they_are_given`.

### 3a. Default-triple equality — the numbers

Captured from `workbench/composition/metrics.py` as it stood at `88dd3a1` (pre-change), on the fixed
table

```python
{"multimodal":            {"a": 1, "b": 10, "c": None, "d": 400},
 "text_only_black_image": {"a": 2, "b": 1,  "c": 30,   "d": 400},
 "image_only_empty_text": {"a": 50,"b": 50, "c": None, "d": None}}
```

and asserted **as literals** after the change
(`test_default_triple_is_bit_identical_to_the_pre_change_function`):

| field | before == after |
|---|---|
| `composition_gap_ndcg` | `-0.26520382860036595` |
| `gap_cohort_count` | `2` |
| `gap_multimodal_ndcg` | `0.6445324131589439` |
| `gap_text_only_ndcg` | `0.8154648767857288` |
| `gap_image_only_ndcg` | `0.17629143438888212` |
| `mrr_gap` | `-0.19999999999999996` |
| `shortcut_outcomes` | `{"both": 0, "composition_required": 0, "image_only": 0, "neither_scored": 2, "text_only": 2, "unresolved": 0}` |
| `multimodal` | coverage `0.75`, mrr `0.3675`, ndcg `0.46823524691863566`, R@1 `33.333333333333336`, R@10 `66.66666666666667`, R@50 `66.66666666666667`, scored `3`, query `4` |
| `text_only` | coverage `1.0`, mrr `0.38395833333333335`, ndcg `0.4871049386478941`, R@1 `25.0`, R@10 `50.0`, R@50 `75.0`, scored `4`, query `4` |
| `image_only` | coverage `0.5`, mrr `0.02`, ndcg `0.17629143438888212`, R@1 `0.0`, R@10 `0.0`, R@50 `100.0`, scored `2`, query `4` |

Also checked by running the pre-change and post-change functions on the same five cases
(`fixed`, `cohort4`, `cohort6`, `empty`, `missing_cond`) plus `aggregate_subsets`: the `asdict` diff
is **empty except for the new `gap_conditions` key**. `compose(t, conditions=PROTOCOL) ==
compose(t)` on every field (`test_explicit_default_triple_matches_the_implicit_one`).

### 3b. Variant gap = 0.5 on a crafted cohort

Ranks chosen so each condition's mean nDCG is exact — `nDCG(7)=1/3`, `nDCG(15)=1/4`,
`nDCG(255)=1/8`:

| condition | ranks | mean nDCG |
|---|---|---|
| `multimodal` | `a=1, b=7, c=7, d=7` | **0.5** |
| `text_only_black_image` | all `15` | **0.25** |
| `clvc_image_encoder_only` | all `255` | **0.125** |

`compose(..., conditions=VARIANT)` → `composition_gap_ndcg = 1 - max(0.25, 0.125)/0.5 =` **0.5**,
`gap_cohort_count 4`, `gap_conditions == VARIANT`, `gap_unavailable_reason None`, and
`mrr_gap = (1 + 3/7)/4 - max(1/15, 1/255)`. All asserted.

## 4. The gap triple is resolved from what was actually scored, BY ROLE

`resolve_gap_conditions(scored_conditions) -> (triple, reason)` (`runner.py`, beside `GAP_ROLES` and
`PROTOCOL_GAP_CONDITIONS`). Roles, not positions; a conditional, not "the ablation wins".

### 4a. Resolved-triple matrix

| conditions actually scored | resolved triple | gap | `gap_role_reason` |
|---|---|---|---|
| `mm, text, image_only_empty_text` (TG-CIR today) | protocol triple | unchanged, `-0.26520382860036595` on the table above | `null` |
| `mm, text, clvc_image_encoder_only` (CLVC new profile) | `(mm, text, clvc_image_encoder_only)` | **computed** (0.5 on the crafted cohort) | "image role filled by the adapter's named ablation 'clvc_image_encoder_only': the protocol image condition 'image_only_empty_text' carries no ranks in this run" |
| `mm, text` (CLVC's old profile) | protocol triple | `null` | "role 'image' has no scored condition in this run (would be 'image_only_empty_text')" |
| all four scored | protocol triple | as protocol | `null` — **the ablation never replaces a scored protocol number** |
| `mm, clvc_image_encoder_only` | `(mm, text, clvc_image_encoder_only)` | `null` | "image role filled by …; role 'text' has no scored condition in this run (would be 'text_only_black_image')" |
| nothing | protocol triple | `null` | three "role '<role>' has no scored condition" clauses |

A condition present in the table but with every rank `None` does **not** count as scored
(`any(rank is not None …)` at the call site, and `test_only_conditions_with_a_rank_count_as_scored`).
A refused condition therefore cannot fill the image role — which is what the last test in §5 shows.

### 4b. Where it is recorded

`summary["metrics"]` now carries, beside the existing fields:

```json
"gap_conditions": ["multimodal", "text_only_black_image", "clvc_image_encoder_only"],
"gap_role_reason": "image role filled by the adapter's named ablation ..."
```

`gap_role_reason` is written even when the gap is null: *why* the gap is absent is a fact, and the
protocol reason string (`gap_unavailable_reason`) is kept unmodified so every existing consumer keeps
reading the same thing. `metrics.image_only.condition` names the condition behind the image role.

## 5. End-to-end proof on a CPU fixture

`test_runner_gap_identity.py` — a NumPy adapter whose image-only row differs from its caption row, so
a routing mistake would be visible in the ranks.

CLVC-shaped run, conditions `[mm, text, clvc_image_encoder_only]`:

| quantity | value |
|---|---|
| `condition_status.clvc_image_encoder_only` | `COMPLETED` |
| `conditions` in the summary | the three requested — `image_only_empty_text` is NOT inherited |
| `metrics.gap_conditions` | `["multimodal", "text_only_black_image", "clvc_image_encoder_only"]` |
| `metrics.gap_role_reason` | "image role filled by …" |
| `metrics.composition_gap_ndcg` | **`-0.5849625007211559`**, hand-computed below |

Hand computation for that fixture: reference row `[1,0,0,0]`, positive `g1=[0,1,0,0]`, gallery
`g0=[1,0,0,0], g1, g2`, reference-eligible. multimodal and text-only both produce the caption row →
`g0` ranks above `g1` → rank 2 → nDCG `1/log2(3) = 0.6309297535714574`. image-only produces
`[0,1,0,0]` → `g1` first → rank 1 → nDCG `1.0`. So
`gap = 1 - max(0.6309297535714574, 1.0)/0.6309297535714574 = 1 - 1.584962500721156 =`
**`-0.584962500721156`**, matching the published value to float noise.

Refusal shape, same fixture with `encode_image_only_queries = None`:

| quantity | value |
|---|---|
| `condition_status.clvc_image_encoder_only` | `ABLATION_UNSUPPORTED`, reason contains `adapter_encoder_missing` and the condition name |
| `mm`, `text` | `COMPLETED` — the refusal is per condition |
| `metrics.gap_conditions` | protocol triple (a refused condition carries no rank, so it does not fill a role) |
| `metrics.composition_gap_ndcg` | `null`, `gap_unavailable_reason == "missing condition(s): image_only_empty_text"` |

## 6. Run identity

`identity["conditions"] = list(conditions)` (`runner.py:1933`) was already there and is unchanged, so
the list is hashed. Proven by three real `run_composition` calls on one fixture, differing only in the
condition list:

| conditions | run_id |
|---|---|
| `[multimodal, text_only_black_image]` | `62e362bf911fb5f97c6e497b` |
| `[multimodal, text_only_black_image, image_only_empty_text]` | `8ef7596c0e1fd42a1910e766` |
| `[multimodal, text_only_black_image, clvc_image_encoder_only]` | `e877a310f9e70ea9d6df29af` |

Three distinct ids: a longer list and a substituted image condition each mint a new run, so a variant
gap can never be mistaken for the protocol gap at the artifact level either
(`test_condition_list_is_part_of_the_run_identity`).

## 7. Tests

```
workbench/envs/ctl312h/bin/python -m pytest \
  workbench/tests/test_runner_image_only_condition.py workbench/tests/test_metrics_gap_conditions.py \
  workbench/tests/test_runner_gap_roles.py workbench/tests/test_runner_gap_identity.py \
  workbench/tests/test_circus_cli.py workbench/tests/test_circus_metrics.py \
  workbench/tests/test_metrics_recall_at_1.py workbench/tests/test_metrics_backfill_recall_at_1.py \
  workbench/tests/test_runner_feature_partition.py workbench/tests/test_runner_batch_provenance.py \
  -q -p no:cacheprovider --basetemp=/tmp/metrics_b10
→ 78 passed
```

Wider affected sweep (24 files: CLI, composition API/topk/provenance/parity, CLVC scope+image-only,
derived, cache/identity, adversarial metrics) → **298 passed**; then CLVC + API again after the last
edit → **50 passed**.

New tests: 9 (`image_only_condition`) + 7 (`gap_conditions`) + 11 (`gap_roles`) + 3
(`gap_identity`) = **30**.

Artifacts: 136/136 digests under `composition/summary/**/full.json`, `runs/**/ranks/*.json`,
`runs/**/candidates/**/*.jsonl` **unchanged** (`diff` of the before/after `sha256sum` listings empty).

## Còn chặn

1. **`composition_eval.py` cannot request the new condition — the GPU command below will not run it
   as written.** The CLI has no `--conditions` flag; `run_composition` defaults to `CONDITIONS`, so a
   Full run launched today scores exactly the protocol triple and the ablation name is never
   requested. Patch spec (`workbench/scripts/composition_eval.py`, not mine):

   - in `_parser()`, after `--keep-features` (line ~80):
     ```python
     parser.add_argument("--conditions", default=None,
                         help="comma-separated condition names; defaults to the protocol triple. "
                              "Adapter-declared ablations (e.g. clvc_image_encoder_only) must be "
                              "named explicitly and change the run_id.")
     ```
   - at the `run_composition(...)` call (line ~381), add:
     ```python
     conditions=(tuple(item.strip() for item in args.conditions.split(",") if item.strip())
                 if args.conditions else CONDITIONS),
     ```
     with `CONDITIONS` imported from `workbench.composition.inputs` (the module the runner validates
     against), and reject an empty value with the CLI's existing `print(json.dumps({"error": …}))`
     style.
   - `_plan`'s `outputs` loop (line 261) iterates `composition_data.CONDITIONS`; it should iterate the
     same resolved list so the plan prints the `.jsonl` path the run will actually write.

   Until that lands, a Full run with the ablation needs a one-off script that calls `run_composition`
   with `conditions=(...)` — the runner side is ready and proven (§5).

2. **No GPU numbers exist for `clvc_image_encoder_only`.** §5 is a CPU fixture; §6 of 99 is the CPU
   row expression. Nothing here claims a GPU parity or a metric.

3. **The backend still needs the name** to display it: `backend/composition.py` `CONDITIONS` (line 18)
   and the two fixture literals (lines ~1451-1505) are owned by the worker running now. My side now
   *emits* `conditions`, `condition_status.<name>` and `metrics.gap_conditions` containing the
   ablation, so a backend that hard-codes the protocol triple will simply not show the row — it will
   not break.

4. **`_write_failed_results` still calls `compose(failed_ranks, query_count=…)` on the default
   triple.** For a failed run every rank is `None`, so the gap is null either way and the reason
   string stays the protocol one; making the failed-run summary name a resolved triple would report a
   role selection for a run that measured nothing. Deliberately left.

5. **`image_only_empty_text` for CLVC stays `ABLATION_UNSUPPORTED`.** This workstream did not weaken
   that reason and a test pins it (99 §5). The new name is the supported image-only path.

6. **Exact GPU command I would now run (NOT run here), assuming §1's flag lands:**

   ```
   MAX_WAIT=3600 workbench/scripts/gpu_job.sh \
     workbench/envs/clvc39-la/bin/python workbench/scripts/composition_eval.py \
     --config /home/heheboiz/cir_rerun_20261109.yaml \
     --models clvc_net --subsets full --topk-depth 100 \
     --conditions multimodal,text_only_black_image,clvc_image_encoder_only
   ```

   What it should publish, so the result can be checked rather than believed: three conditions
   present in `conditions`; `unscored_count` 2 per active condition (the two genuinely ambiguous qids
   `7:3038`, `7:3790`), coverage `0.99967`; `condition_status.clvc_image_encoder_only.status ==
   COMPLETED`; `metrics.gap_conditions` = the triple above with `gap_role_reason` non-null; and
   `composition_gap_ndcg` **not null**. A gap that is null with
   `gap_unavailable_reason == "missing condition(s): clvc_image_encoder_only"` would mean the
   condition produced no ranks and the run must not be published as a variant-gap measurement.
