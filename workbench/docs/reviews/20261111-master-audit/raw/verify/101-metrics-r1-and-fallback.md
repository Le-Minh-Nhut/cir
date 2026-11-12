# 101 — R@1 from stored exact ranks, and the ENCODE batch-partition record

Branch `research/cir-failure-workbench`, HEAD at start `82cb1044` (`== origin`).
Control interpreter `workbench/envs/ctl312h/bin/python`. CPU only: no GPU job was started, no
`--smoke` run was made, the live systemd units and `/tmp/cir_gpu.lock` were not touched.

Owned files touched: `workbench/composition/metrics.py`, `workbench/composition/runner.py`, new
`workbench/scripts/backfill_recall_at_1.py`, new tests `test_metrics_recall_at_1.py`,
`test_metrics_backfill_recall_at_1.py`, `test_runner_feature_partition.py`,
`test_runner_batch_provenance.py`. Nothing else in the tree was edited, so no patch spec was needed
for the owned side; the API/UI patch spec is in `Còn chặn` §1.

---

## 1. R@1 — definition

`workbench/composition/metrics.py:48` adds `recall_at_1` to `ConditionMetrics`;
`metrics.py:77` computes it as

```python
100.0 * sum(hit_at_k(rank, 1) for rank in scored) / scored_count
```

so it is the **same cohort and the same denominator** as R@10 (`metrics.py:78`) and R@50
(`:79`): `scored` is the list of non-`None` ranks, unscored queries move `coverage` only, and a
rank of exactly 1 is the only hit. R@1 is the exact rank being 1 — it is not `hit_at_k(rank, 1)` on a
truncated candidate list, so a stored `target_rank` of 23 contributes 0 regardless of the export
depth.

Tie policy is untouched: `compute_condition_metrics` consumes ranks and never re-ties. A rank of 1
that the producer reached through a tie (`CompositionAdapter._ranks_from_scores`:
`1 + #strictly-greater + #earlier-equal`, tie broken by earlier gallery position) counts as a hit;
that is asserted, not assumed, in `test_rank_1_from_a_producer_tie_is_a_hit_and_is_not_re_tied`
(`test_metrics_recall_at_1.py`).

## 2. Boundary and null cases — test results

`test_metrics_recall_at_1.py`, 7 tests, all pass:

| case | assertion | result |
|---|---|---|
| ranks `[1, 4, null]` | `recall_at_1 == 50.0`, `coverage == 2/3`, `scored_count 2`, `query_count 3` | pass |
| rank exactly 1 vs rank exactly 2 | `hit_at_k(1,1)==1`, `hit_at_k(2,1)==0` | pass |
| producer-resolved tie at 1 | R@1 = 50 % on `{exact_tie:1, behind_tie:2}` | pass |
| empty cohort `{}` | `recall_at_1 is None`, reason `empty query cohort` | pass |
| zero ranks `{a: None, b: None}` | `recall_at_1 is None`, reason `no queries have an exact rank` | pass |
| condition absent from the table | per-condition `None` + reason, other conditions unaffected | pass |
| incomplete rank table (`query_count` 3, 1–2 rows per condition) | own denominator per condition; R@1 50/100/0 with coverage 2/3, 1/3, 2/3 | pass |
| monotonicity over `[1,1,2,3,9,10,11,50,51,74381]` | R@1 20 ≤ R@10 60 ≤ R@50 80 | pass |

`null` is never emitted as `0`: a condition with no scorable rank reports `recall_at_1 = None`
**and** `unavailable_reason`, which is how `clvc_net`'s `image_only_empty_text` is reported in §3.

## 3. Backfill of the published Full runs — no recomputation, no GPU

`workbench/scripts/backfill_recall_at_1.py` reads only `runs/<run_id>/ranks/*.json` plus the matching
`summary/<run_dir>/full.json`. It re-scores nothing, re-encodes nothing and calls no adapter.

Three guards, all enforced in code:

1. **Cohort equality** — the rank table's qid set must equal the summary's `query_count`, else the
   derivation refuses rather than compare two cohorts (test
   `test_backfill_refuses_a_cohort_mismatch`).
2. **Reproducibility** — R@10, R@50, `scored_count` and `coverage` are recomputed with the same
   `metrics.compose` and compared to the published summary at `1e-12`; a mismatch refuses the whole
   backfill (test `test_backfill_refuses_when_the_summary_is_not_reproducible`).
3. **Immutability** — writing into a published `summary/**/{full,sf,validated}.json` is refused by
   name (test `test_backfill_refuses_to_write_into_a_published_summary`).

Output (new file, nothing existing rewritten):

```
workbench/artifacts/composition/derived/recall_at_1/clvc_net_eab10106b2df8e97fb7a5ce5-retry-2+tgcir_7b1fa85007bffdab931f7f19_full.json
```

Command:

```
workbench/envs/ctl312h/bin/python workbench/scripts/backfill_recall_at_1.py \
  --run tgcir:7b1fa85007bffdab931f7f19 --run clvc_net:eab10106b2df8e97fb7a5ce5
```

### 3a. `tgcir` Full — run `7b1fa85007bffdab931f7f19` (ranks sha256 `188d84fe…`)

| condition | rank==1 | scored | **R@1** | R@10 | R@50 |
|---|---|---|---|---|---|
| `multimodal` | 379 | 6003 | **6.313510** | 23.338331 | 42.112277 |
| `text_only_black_image` | 117 | 6003 | **1.949025** | 6.546727 | 13.659837 |
| `image_only_empty_text` | 5 | 6003 | **0.083292** | 2.798601 | 7.163085 |

### 3b. `clvc_net` Full — attempt `eab10106b2df8e97fb7a5ce5-retry-2` (ranks sha256 `9ad5399e…`)

| condition | rank==1 | scored | **R@1** | R@10 | R@50 |
|---|---|---|---|---|---|
| `multimodal` | 83 | 5884 | **1.410605** | 7.256968 | 18.303875 |
| `text_only_black_image` | 6 | 5884 | **0.101971** | 1.393610 | 4.469748 |
| `image_only_empty_text` | 0 | 0 | **`null`** — `no queries have an exact rank` | `null` | `null` |

`image_only_empty_text` stays `null` with its reason and was NOT written as 0.0: the condition is
`ABLATION_UNSUPPORTED` in that run and has zero ranks.

### 3c. Consistency check

* **Recomputation.** Every published R@10/R@50/`scored_count`/`coverage` on both runs reproduces from
  the rank table at `1e-12` — the tool refuses the backfill otherwise, and it did not fire.
  Reproduced numbers, exactly: tgcir `23.33833083458271 / 42.11227719473597`, `6.546726636681659 /
  13.65983674829252`, `2.798600699650175 / 7.163085124104614`; clvc `7.256968048946295 /
  18.303874915023794`, `1.3936097892590076 / 4.46974847042828`.
* **Monotonicity.** R@1 ≤ R@10 ≤ R@50 holds for all six scorable condition-rows (table above).
* **By hand, one condition each.** Counted from the candidate JSONL, which is an independent
  serialisation of the same ranking:

  | source | rows | `target_rank == 1` | R@1 | R@10 | R@50 |
  |---|---|---|---|---|---|
  | `tgcir/.../multimodal.jsonl` | 6003 | 379 | 6.3135099117108116 | 23.33833083458271 | 42.11227719473597 |
  | `clvc/.../multimodal.jsonl` | 5884 | 83 | 1.4106050305914344 | 7.256968048946295 | — |
  | `clvc/.../text_only_black_image.jsonl` | 5884 | 6 | 0.10197144799456152 | 1.3936097892590076 | — |

  JSONL and rank table agree to the last digit on all three, i.e. R@1 is not a second ranking.

## 4. ENCODE fallback — reproduction before the fix

`runner._encode_queries` retried a failed batch call ONE QUERY AT A TIME and mixed that shape with
the group shape inside one condition, while nothing on disk said so. Reproduced on a CPU fixture
adapter that raises on any multi-row call (`/tmp/omp_partition_repro.py`; test adapter
`BatchStrictAdapter` in `test_runner_feature_partition.py`):

```
BEFORE (pristine worktree at 82cb1044):
adapter call shapes: [3, 1, 1, 1]
memmap sidecar: canonical-fingerprint
partition record: ABSENT (nothing on disk says these 3 rows were encoded one at a time)
canonical-key cache HIT?: True          <-- a later canonical run would have loaded these rows
```

So the defect was two-sided: the mixed partition was unrecorded, **and** its memmap carried the
canonical key, so a canonical re-run (`--resume`, or the same configuration) would have taken a cache
hit on features produced at batch 1. For CLVC-Net that is a different measurement: query rows move by
up to `1.4639e-02` between batch 1 and 20 and the pinned upstream evaluator moves identically
(87 §3b).

## 5. What is now recorded — AFTER

Same script, same fixture, after the change:

```
AFTER (this branch):
adapter call shapes: [3, 1, 1, 1]
memmap sidecar: 60b52d6eacf5ae9ab9838b445b83948b14a9a14bb6ec1d7836cdf90445b62b87
partition record: {"adapter_batch_size": null, "adapter_calls": 4,
  "batch_order": "query index order, ascending; fallback rows re-encoded alone in the same order",
  "cache_key": "60b52d6e…b62b87", "call_window": 4096,
  "fallback_qids": ["7:1", "7:2", "7:3"],
  "fallback_reason": "RuntimeError: fixture batch rejection", "fallback_used": true,
  "partition": "per_query_fallback", "runner_batch_size": 2}
canonical-key cache HIT?: False
```

Where it is published (all in `runner.py`):

* `condition_status.<cond>.feature_batch_fallback` (`runner.py:2280`, `:2298`) — per condition:
  `partition` (`canonical_call_window` | `per_query_fallback` | `not_encoded`), `fallback_used`,
  the **named `fallback_qids`**, `fallback_reason` (the exception that triggered the retry),
  `call_window`, `runner_batch_size`, `adapter_batch_size`, `adapter_calls`, and a `reason` when
  nothing on disk records a partition.
* `condition_status.<cond>.fallback_used` keeps its **unchanged** meaning: the SCORING pass fell back
  to one query at a time. The two keys are deliberately distinct — encode fallback changes features,
  scoring fallback changes the matrix M dimension of the contraction.
* Summary-level `batch_partition` (`runner.py:2488`) — `runner_batch_size`, `adapter_batch_size`,
  `call_window`, the `query_order` actually used ("release/loader query order, ascending index;
  scoring groups of runner_batch_size in that same order"), the `partition_policy` in words, and the
  per-condition partition block. A reader can therefore tell a canonical run from a diagnostic one
  from the artifact alone.
* Policy statement in code: `runner.py:664` (`FEATURE_PARTITION_CANONICAL`/`_MIXED`), with the
  measured `1.4639e-02` cited at the constant.

`_query_feature_partition(..., not_encoded_reason=...)` (`runner.py:852`) reports `not_encoded` for a
condition that never encoded (an `ABLATION_UNSUPPORTED` mode, a refused model) instead of claiming
"unknown", so `clvc_net`'s empty-text condition does not read as lost provenance.

## 6. Cache-key proof

The partition record is written beside the memmap (`<feature>.npy.feature-fallback.json`) and the
memmap sidecar's fingerprint is the **derived** key
`_feature_partition_fingerprint(fingerprint, record)` (`runner.py:668`), which digests
`{cache, feature_partition, fallback_qids}` and returns the plain fingerprint only when no qid fell
back. Consequences, each pinned by a test in `test_runner_feature_partition.py` (6 tests, all pass):

| scenario | assertion | result |
|---|---|---|
| same inputs, different partition (fallback first, canonical second) | fallback key ≠ canonical key; `_load_memmap(dest, canonical)` returns `None`; the canonical encode RUNS (`calls == [3]`) and rewrites the sidecar to the canonical key | pass |
| same configuration twice (canonical both times) | second call does not reach the adapter (`calls == []`) — a HIT | pass |
| key depends on the affected qids | `{7:1}` ≠ `{7:2}` ≠ canonical, and `{7:1}` twice is equal | pass |
| resume path (`_cached_features`) | reads the recorded `cache_key` first, so a fallback cache is not served under the canonical digest | pass (`test_circus_cli.py` resume tests) |

End-to-end (CPU fixture CLI, `test_runner_batch_provenance.py`, 2 tests, both pass): a run whose
encode falls back publishes `condition_status.multimodal.feature_batch_fallback.fallback_qids`
non-empty with the triggering reason, `batch_partition.conditions.multimodal` agreeing with it, and
`fallback_used: false` still meaning "scoring did not fall back"; a normal run publishes
`partition: canonical_call_window`, `fallback_qids: []`, `adapter_batch_size: 2`.

Refusal of `--batch-size` changes: none. `adapter_batch_size` is recorded, never raised, never
"normalised"; the baseline pointer `runner.py:668` docstring states that the two batch fields are
already hashed into the run identity and only the partition is added here.

## 7. Published artifacts: unchanged

136 files under `workbench/artifacts/composition/summary/**/full.json`,
`runs/**/ranks/*.json` and `runs/**/candidates/**/*.jsonl` were digested before any edit and after the
last test run. `diff` of the two `sha256sum` listings is empty: **136/136 identical**.

Spot values:

| artifact | sha256 (before == after) |
|---|---|
| `summary/tgcir_tgcir_archive_unresolved_7b1fa85007bffdab931f7f19/full.json` | `0e20e46c2dc6963c432841bcfc833ea4cb0d04be10387bc1c36e8c5e9ffbc87d` |
| `summary/clvc_net_author_drive_unresolved_eab10106b2df8e97fb7a5ce5-retry-2/full.json` | `48783dbce99ab8ea0f9e6ff80696a47abacf9e359c71e9a714ed59cfa362678a` |
| `runs/7b1fa85007bffdab931f7f19/ranks/tgcir_tgcir_archive_unresolved.json` | `188d84febd9a1ebcf57b3a3f27e8fdfcbe91733feddba036ed44bb55df849cf0` |
| `runs/eab10106b2df8e97fb7a5ce5/ranks/clvc_net_author_drive_unresolved.json` | `9ad5399e20f927ce9a6c410156bc937e6ddd06bfa4de7534b3007f647d1f7e55` |

The only artifact written is the new derived file named in §3.

## 8. Tests run

```
workbench/envs/ctl312h/bin/python -m pytest \
  workbench/tests/test_circus_cli.py workbench/tests/test_circus_metrics.py \
  workbench/tests/test_composition_cache_and_identity_lock.py workbench/tests/test_composition_backfill_refusal.py \
  workbench/tests/test_composition_topk_runner.py workbench/tests/test_audit_fix_f3_gallery_cache_shape.py \
  workbench/tests/test_review_defects_evaluator.py workbench/tests/test_limn_scope_runner.py \
  workbench/tests/test_derived_lineage_r09.py workbench/tests/test_derived_subset_status.py \
  workbench/tests/test_composition_api.py workbench/tests/test_composition_topk_api.py \
  workbench/tests/test_composition_topk_provenance.py workbench/tests/test_capability_composition.py \
  workbench/tests/test_composition_topk_export.py workbench/tests/test_composition_topk_rank_agreement.py \
  workbench/tests/test_clvc_net_composition_parity.py workbench/tests/test_tgcir_composition_parity.py \
  workbench/tests/test_composition_adapters_missing.py workbench/tests/test_composition_provenance_identity.py \
  workbench/tests/test_composition_condition_transform_identity.py workbench/tests/test_composition_blas_pool_isolation.py \
  workbench/tests/test_adversarial_metrics.py \
  workbench/tests/test_metrics_recall_at_1.py workbench/tests/test_metrics_backfill_recall_at_1.py \
  workbench/tests/test_runner_feature_partition.py workbench/tests/test_runner_batch_provenance.py \
  -q -p no:cacheprovider --basetemp=/tmp/metrics_t8a
→ 302 passed, 5 warnings in 15.47s
```

Pre-existing, NOT caused by this change: `test_circus_adapters_blip.py` has 4 failures
(`AttributeError: 'Adapter' object has no attribute 'batch_size'`, from
`model_ptha_mtst.py:188` / `model_conesep.py`) on a **pristine worktree at `82cb1044`** as well —
`4 failed, 14 passed` both before and after. Reported, not touched: those files are outside this
workstream's ownership.

## Còn chặn

1. **API/UI do not serve R@1 yet — patch spec, files I may not edit.** The metric exists in the
   summary of NEW runs (`metrics.multimodal.recall_at_1` etc.) but the two published Full artifacts
   are immutable, and neither the API nor the UI reads the derived file.

   *`workbench/backend/composition.py`*

   - line 123 (`read_summary_file`, inside the `mm` block), add beside `recall_at_10_multimodal`:
     ```python
     if "recall_at_1_multimodal" not in metrics and "recall_at_1" in mm:
         metrics["recall_at_1_multimodal"] = mm["recall_at_1"]
     ```
   - line 194, extend the metric-key set so a summary whose ONLY number is R@1 is not read as failed:
     ```python
     _METRIC_KEYS = frozenset({"ndcg_full", "recall_at_1", "recall_at_10", "recall_at_50", "mrr"})
     ```
   - `decorate_summary` (`:905`), after the `metrics`/`statistics` block, overlay the derived file so
     an already-published run can serve R@1 without rewriting its summary:
     ```python
     derived = load_recall_at_1(artifacts_root, summary.get("run_id"), summary.get("attempt"))
     if derived:
         metrics.setdefault("recall_at_1_multimodal",
                            (derived["conditions"].get("multimodal") or {}).get("recall_at_1"))
         summary["recall_at_1_source"] = derived["derived_from"]
     ```
     with a new module-level helper (new function; no existing behaviour changes):
     ```python
     def load_recall_at_1(artifacts_root: Path, run_id: str | None, attempt: str | None) -> dict[str, Any] | None:
         """R@1 derived from stored exact ranks, keyed by (model, attempt); None when absent."""
     ```
     reading `composition/derived/recall_at_1/*.json`, matching on `attempt` (the file records both
     `attempt` and `run_id`), and refusing a record whose `derived_from.ranks.sha256` no longer
     matches `runs/<run_id>/ranks/*.json` — the derivation must name the bytes it describes.
     `attempt` is not on the summary dict today; read it from the ranks payload already loaded in
     `decorate_summary` (`ranks_data["attempt"]`) rather than adding a summary field.

   *`workbench/frontend/types.ts`* — line 72 already declares `recall_at_1_multimodal?: number | null`;
   also add `recall_at_1?: number | null;` to each of the three nested blocks at lines 63-69 so the
   per-condition value is typable.

   *`workbench/frontend/EvaluationPage.tsx`* — the R@10 column at line 280; add the R@1 value in the
   same cell (or a new column) and in the detail row at line 367:
   ```tsx
   <td className="num">{run.is_failed ? "—" : fmtPercentValue(run.metrics?.recall_at_1_multimodal)}</td>
   ...
   ["recall", <span key="r">R@1 {fmtPercentValue(run.metrics?.recall_at_1_multimodal)} · R@10 …
   ```
   and, if a sortable column is wanted, a `case "r1": return run.metrics?.recall_at_1_multimodal ?? null;`
   beside `case "r10"` (line 150).

   **A `null` must render as `—`/`n/a` with the `unavailable_reason`, never as `0.0`** — that is the
   whole point of the null distinction in §2, and `clvc_net`'s `image_only_empty_text` is exactly
   that case.

2. **The two published Full runs still carry no R@1 in their own `metrics` block.** The derived file
   is the only place their R@1 exists. Making it appear in their summary requires either the API
   overlay above or a new run; rewriting `full.json` is refused by design.

3. **A fallback-partition run is now marked but not prevented.** Nothing stops a diagnostic run from
   being read as comparable if a consumer ignores `feature_batch_fallback`. The `partition` field is
   `null` (not `canonical`) for any run whose feature cache was deleted or which predates the field —
   including both Full runs above, whose `work/` directories were pruned, so their ENCODE partition
   is now **unknowable from disk**. Their `ranks/*.json` predates this field too; the low-level
   evidence that they were canonical is the absence of a batch-fallback suffix in the per-query
   results (`grep -c "(batch:" …jsonl` = 0 on all nine CLVC files), which is recorded in 90 §4 rather
   than re-derived here.

4. **R@1 was not added to `subset_provenance` or the bootstrap/gap path.** `paired_bootstrap_gap` and
   `CompositionResult`'s gap fields are unchanged; adding an R@1 gap would need a `gap_recall_at_1`
   definition the brief did not ask for.
