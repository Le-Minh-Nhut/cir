# R09 — derived lineage: failure map + parent candidates

Run: 2026-11-11. Branch research/cir-failure-workbench.

## Publisher (`scripts/m03/publish_derived_runs.py`)

`query_failures` is now built per condition over the derived membership only:

```
parent_failures = ranks_payload.get("query_failures") or {}
for condition in parent_ranks:
    reason = reasons.get(qid)
    if reason is None and rank is None: reason = PARENT_RECORDED_NO_REASON  # "parent_recorded_no_reason"
```

Refusals (SystemExit) when an unscored qid has no carried reason or a carried qid is outside the membership.
`derived_from` gained `parent_topk_export` (requested/status/depth/conditions/partial) + `parent_topk_note`.

## Reader (`backend/composition.py`)

- `_derived_parent_candidates`: verifies the claimed `parent_rank_artifact_sha256` against the parent rank file, then reads the PARENT export.
- statuses: `NOT_REQUESTED`, `NOT_EXPORTED`, `PARTIAL`, `COMPLETE`; refusal is `parent_topk_unavailable` with the reason (digest mismatch, no parent rank artifact, no claimed digest).
- per-item `candidate_status` carries the parent status verbatim (not relabelled `AVAILABLE`), plus `candidate_source` = `parent_run` and `candidate_source_run_id`.
- page `candidates` block gained `source`, `source_run_id`, `parent_run_id`, `lineage_verified`; new `lineage` block.

## On-disk artifacts patched (ranks + summary of both derived runs)

Ranks/metrics proven identical before and after (asserted in the patch script, and by
`test_on_disk_derived_runs_still_re_aggregate_to_their_published_metrics`).

pre-r9 sha256:

| file | sha256 |
|---|---|
| runs/6a96c5c1.../ranks/limn_base_iter0_all_categories.json | 635350d5de128924223e18e5b57c5e4664803037ca62501ed1a4dd54ccfc3643 |
| summary/..._6a96c5c1.../sf.json | aefcacb45ad496ebda9bbd427b05363483ce41679467e8b78323f90f0eb71eb1 |
| runs/aee2e8d3.../ranks/limn_base_iter0_all_categories.json | 63c4c78c4d2f9fd8e5d39b591348bb1c7fdd78f903085a9259efb90e21d01bb1 |
| summary/..._aee2e8d3.../validated.json | 91d5db938b1655e2b618bae57d76ee564ede23b5d09b42b8a9166df7cc4c7ab3 |

Copies: `<name>.pre-r9` beside each file (gitignored artifact tree).

Leaf diff (only additions, no value changed):

```
ranks 6a96c5c1   + query_failures.{multimodal,text_only_black_image,image_only_empty_text}."7:3790"
                + derived_from.parent_topk_export
                + derived_from.parent_topk_note
                + identity.derived_from.parent_topk_export
                + identity.derived_from.parent_topk_note
summary 6a96c5c1 + derived_from.parent_topk_export
                + derived_from.parent_topk_note
ranks aee2e8d3   + derived_from.parent_topk_export / parent_topk_note
                + identity.derived_from.parent_topk_export / parent_topk_note
summary aee2e8d3 + derived_from.parent_topk_export / parent_topk_note
```

`aee2e8d3` carries zero failure entries: its 586-qid membership excludes both parent-unscored qids
(`7:3038`, `7:3790`), so there is nothing to carry and nothing to invent.

## Probe (/tmp/derived_reason_probe_r09.py, imports repointed)

before:

```
published_failures = {multimodal: {}, text_only_black_image: {}, image_only_empty_text: {}}
reader: failure=null, unscored_kind=null, candidate_status="TOPK_EXPORT_NOT_REQUESTED"
metrics: sf denominator 0 coverage 0.0; validated 1 / 1.0 / R@10=100
```

after:

```
published_failures = {multimodal: {"q1": "query q1: category unresolved ..."}, ...3 conditions}
reader: failure="query q1: category unresolved has no adapter embedding scope"
        unscored_kind="category_unresolved"
        candidate_status="NOT_REQUESTED"  (parent has no export in the probe fixture; honest)
        candidate_source="parent_run", candidate_source_run_id="parent"
metrics: identical block (sf: denominator 0, coverage 0.0, R@10 null; validated: 1, 1.0, 100.0)
```

## Tests (`workbench/tests/test_derived_lineage_r09.py`)

| test | before | after |
|---|---|---|
| test_failure_map_is_carried_by_membership | FAIL KeyError q1 | pass |
| test_unscored_qid_without_a_parent_reason_says_so | FAIL None != parent_recorded_no_reason | pass |
| test_reader_serves_the_parents_candidates_with_the_parents_status | FAIL TOPK_EXPORT_NOT_REQUESTED != COMPLETE | pass |
| test_reader_refuses_when_the_parent_digest_does_not_match | FAIL NOT_REQUESTED != parent_topk_unavailable | pass |
| test_reader_reports_the_parents_own_export_status | FAIL | pass |
| test_on_disk_derived_runs_carry_the_failure_map | FAIL KeyError source | pass |
| test_metrics_and_denominators_do_not_move | pass (guard) | pass |
| test_on_disk_derived_runs_still_re_aggregate_to_their_published_metrics | pass (guard) | pass |

pytest:

```
FAILED workbench/tests/test_circus_adapters_blip.py::test_resident_worker_failure_uses_shared_error_and_stays_failed[workbench.composition.model_conesep-conesep] - AttributeError: 'Adapter' object has no attribute 'batch_size'
FAILED workbench/tests/test_circus_adapters_blip.py::test_resident_worker_failure_uses_shared_error_and_stays_failed[workbench.composition.model_ptha_mtst-ptha_mtst] - AttributeError: 'Adapter' object has no attribute 'batch_size'
4 failed, 249 passed, 1 skipped, 2617 deselected, 8 warnings in 9.72s
```

`pytest workbench/tests -k "derived or composition" -q --basetemp=/tmp/r09_bt`:

```
4 failed, 251 passed, 1 skipped, 2617 deselected
```

The 4 failures are `test_circus_adapters_blip.py`
(`test_resident_worker_reuse_and_cleanup`, `test_resident_worker_failure_uses_shared_error_and_stays_failed`
x 2 model ids), all `AttributeError: 'Adapter' object has no attribute 'batch_size'`. They fail
identically with `backend/composition.py` reverted to HEAD, so they are not caused by R09 and were
not touched.
