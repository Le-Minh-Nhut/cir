# 60 — Collector identity: R01 / R07 / R08

Owner file: `workbench/backend/native_measurements.py` (only production file touched).
New tests: `workbench/tests/test_native_measurements_identity.py`.

## The one shared parser

`metric_values(block)` is now the ONLY metric/unit reader; `served_metrics()` adds the two fallbacks;
`_report_metrics()` delegates to it and `_blocks()`/`_rows_for_file()` call it directly.

It iterates `METRIC_ALIASES`, and per metric tries `UNIT_SUFFIX_ORDER` (`_percent`, `_pct`, bare,
`_fraction`), taking the first key that is a real number — so the value and the declared unit always
describe the SAME key. A suffix only counts when appended to THAT metric's own alias, so
`dead_parameters_pct` declares nothing about recall. It returns
`(values, unit, unit_source, per_metric_units)`; `per_metric_units` names the suffix each metric's
value came from, so a mixed-scale block is visible instead of collapsed to one guess.

`_suffix_declared_unit()` is deleted (subsumed; nothing referenced it outside the row path).
`unit_source` vocabulary unchanged: `declared`, `sibling_key`, `inferred_from_magnitude`,
`undeclared`; `DECLARED_UNIT_SOURCES` still means "the file stated the scale".

One addition, required by an existing census assertion: `_agreed_suffix()` accepts a suffix of the
metric's OWN aliases only when it states the same number on another scale
(`macro_r10: 0.6` + `macro_r10_fraction: 0.6` ⇒ `fraction`); if it carries a different number it
declares nothing (`macro_r10: 12.0` + `macro_r10_fraction: 0.12` ⇒ undeclared).

## R01 — join on real identity

`IDENTITY_JOIN_FIELDS = ("model_id", "protocol_id", "checkpoint_id")`.

Fields a row records that a report also really carries:

| field | row side | report side | compared |
|---|---|---|---|
| `model_id` | `model_id`/`model`/`method_id`/roll-up entry key | `model_id` on both report shapes | yes |
| `protocol_id` | `protocol_id`/`protocol` | `protocol_id` (aggregate report), `protocol` (native replay) | yes, when both state it |
| `checkpoint_id` | `checkpoint_id`/`variant`/`checkpoint_name`/`label` | `checkpoint_id` on aggregate reports; **absent** on native replays | yes, when the report states it |

Not included, because one side never carries it: `dataset`/`dataset_id` (neither side, on any of the
17 files under `workbench/artifacts/reports/` or the 8 replays under `docs/reviews/*/raw/native/`),
`benchmark`/`benchmark_reference` (reports only, and `null` on every report in the tree), `run_id`
(reports only).

Rules, as implemented:

* exact identity match ⇒ replacement allowed;
* `by_model_any_protocol` fallback **deleted**; the only relaxation is a row with
  `protocol_id is None` and exactly ONE candidate report ⇒ `identity_join: "single_candidate_protocol_unknown"`;
* 2+ candidates ⇒ no replacement, `identity_ambiguous: [paths]`;
* **policy: refuse, never relabel.** A report that differs on a field the row states may not replace
  the row's number. The row keeps its own number, and `artifact_backed`/`counted` keep their
  previous value (a direct artifact is still `artifact_backed: true` — `backed_by` points at itself,
  because that is what backs it; the unrelated report does not);
* when a replacement does happen, `raw_model_id`, `raw_checkpoint_id`, `raw_protocol_id`,
  `raw_source_path` are stored beside `raw_metrics`/`raw_unit`/`raw_unit_source`.

## R08 — dedup by identity

`duplicate_of` is now written in exactly two ancestry situations, documented in the docstring:

1. a roll-up and another row share the SAME backing artifact for the same model+protocol (one run
   stated twice: `clip4cir_parsed.json` / `selection_decisions.json`);
2. a roll-up restates an in-tree artifact's number (decided by source KIND `rollup:*` vs `artifact`;
   value equality only chooses WHICH artifact).

Anything else with equal numbers keeps `counted: true` and records `same_metrics_as: <path>`.
The stated `sha256` + `metric_path` identity needs no rule: each block is yielded once per path.

## with_metrics 21 → 23: which two rows, and why they are two measurements

Verdict: **neither pair is a second measurement.** The +2 was wrong and has been reverted; `hint` and
`tme` are each ONE run written twice, and each pair now carries `duplicate_of` again — this time on
stated ancestry (identical `checkpoint_sha256`), not on rounded scores.

| model | file A | file B | A sha256 | B sha256 | metrics (both) | checkpoint (both) | protocol (both) | verdict |
|---|---|---|---|---|---|---|---|---|
| hint | `checkpoint_selection/20261106T002617+0700/cohort/measurements/hint_fashioniq.json` (4333 B, `2026-11-06T15:23:57Z`) | `checkpoint_selection/20261106T233409+0700/measurements/hint_doc_command.json` (4333 B, `2026-11-06T17:18:51Z`) | `131e4a35f6cc…acc3acd6` | `0355a0c27849…c06f94` | r10 59.80338494972219, r50 80.13076097500765, mean 69.96707296236491 | `fashioniq` / `fashioniq` | `fashioniq_val_split` / same | **DUPLICATE** — one run re-exported |
| tme | `checkpoint_selection/20261106T002617+0700/cohort/measurements/tme_fashioniq_0.json` (11483 B, `16:08:56Z`) | `checkpoint_selection/20261106T233409+0700/measurements/tme_normal_path_full.json` (11372 B, `18:53:49Z`) | `35b605786602…9641d0c` | `0239f181e8e2…03345a9` | r10 54.922901590665184, r50 74.83842174212138, mean 64.88066166639328 | `fashioniq_0` / `fashioniq_0` | `fashioniq_original_split` / same | **DUPLICATE** — one run re-exported |

Different files, different `sha256` — but both members of each pair record the SAME weights:
`provenance.checkpoint_sha256 = bf9c72d42206b667ed586758c15c648970df821857e4067f26155a114a6517d4`
(hint) and `fea7fc08760dad1f5a66184ea7e65b45388f9781157cbd33a3b64db74103968a` (tme), with the same
`checkpoint_id`. The file pair also agrees on `replay_type`/`model_id`/`protocol`/`complete_benchmark`
and differs only in `timestamp` (hint) or in feature-cache provenance (tme:
`UNVERIFIED` → `VERIFIED_PRIOR_RUN`). That is one measurement at two precisions of bookkeeping, not
two checkpoints.

Fix applied: `_checkpoint_fingerprint()` reads the digest a document states (top-level, else unanimous
across `categories`/`results` provenance) into `row["checkpoint_fingerprints"]`, and `_mark_duplicates`
rule 2 demotes a same-model/same-protocol/same-digest/same-number group. The keeper is the latest
`recorded_at`, and the group size is recorded as `checkpoint_ancestry_count`. Result: `with_metrics`
back to **21**, `duplicate_rows` back to **22**; `hint_doc_command.json` and `tme_normal_path_full.json`
are the counted rows, the earlier exports are `duplicate_of` them.

This rule cannot merge two genuinely different checkpoints: `airknow_n05` (`f4d25ed9…`) and
`airknow_n08` (`756fc0bd…`) have different digests and stay two counted rows. Equal scores alone still
never demote anything.

## R01 side door (found in review) — closed

`single_candidate_protocol_unknown` used to skip `_identity_matches` entirely, so a row with no
protocol but `checkpoint_id="checkpoint_A"` was replaced by a `checkpoint_B` report and kept label A.
Now the relaxation applies only when the join already matched (`len(matches) == 1`), and it is a
LABEL for the `row.protocol_id is None ∧ one candidate ∧ candidate states a protocol` case. A row
naming a different checkpoint refuses like every other mismatch and lists the candidate in
`identity_ambiguous`. Enforced, not documented: after the if/elif chain,

```python
if report is not None and identity_join != "exact" and not _identity_matches(row, report):
    raise AssertionError(...)
```

`test_a_single_candidate_cannot_back_a_row_that_names_another_checkpoint` covers it
(`r10 == 12.0`, `raw_metrics is None`, `backed_by == own source`, `identity_ambiguous` non-empty).

## csmcir

`identity_ambiguous` holds both `reports/csmcir_fashioniq_20261107T054343058424Z_…f564570e_aggregate.json`
and `reports/csmcir_fashioniq_20261107T054933277898Z_…f03887d2_aggregate.json`. Nothing was
substituted, and `r10 = 57.05073873202006` is still served under its own label `checkpoint_id=fashioniq`
/ `protocol_id=fashioniq_original_split`, with `backed_by` = its own source file.

## Verification

New tests: `workbench/tests/test_native_measurements_identity.py` — 13 passed.
Fail-before is proved by the bundled probe, which imports the `06e30e8c` snapshot and still shows
every defect (see below); each new test quotes the pre-fix output in its docstring.

`pytest workbench/tests/test_native_measurements_census.py workbench/tests/test_native_measurements_identity.py -q` ⇒ **44 passed**.

`test_a_rollup_entry_is_not_double_counted_as_a_second_measurement` asserted ranked identity as
`(model_id, protocol_id, round(value, 4))` uniqueness — the R08 defect spelled as a requirement. It
now asserts uniqueness of `(source sha256, metric_path)`, i.e. of the source block, and its docstring
states why value equality is not identity (two checkpoints with equal recall). Everything else in the
test is unchanged. No other test in the repo used the old rule
(`grep -rn "round(value, 4)\|round(value,4)" workbench/tests/` now matches only that docstring).

Real catalog, before → after (`counts`):

* `with_metrics` 21 (unchanged: the +2 candidate pair is one run each — see the table above)
* `duplicate_rows` 22 (unchanged)
* everything else unchanged: `blocks_with_metrics 43`, `rollup_rows 18`, `unbacked_rows 18`,
  `unbacked_counted 0`, `unit_declared 4`, `unit_sibling_key 2`, `unit_inferred 39`,
  `models_with_metrics 17`, `verified 0`
* `csmcir` is the one identity-ambiguous row in the real tree: two identical aggregate reports for
  `fashioniq_original_split`; it keeps its own 57.0507 and states both paths.

## Probes

`/tmp/after_collector.txt` — bundled `collector_probe.py`, which loads the **snapshot** at
`review_06e30e8c/repo/`, so it reproduces the defects unchanged (`r10=88.0` from `unrelated_run_B`,
`0.2793` labelled `percent`, `duplicate_of` on equal values). Kept as the failing-before baseline.

`/tmp/after_real_collector.txt` — `/tmp/verify_native_identity.py`, which imports
`/home/heheboiz/data/cir/workbench/backend/native_measurements.py` and runs on the real artifacts
root plus the same synthetic fixtures:

| case | result |
|---|---|
| R01 A row + unrelated B report | `r10=12.0`, `ck=checkpoint_A`, `artifact_backed=true` with `backed_by` = itself, no `backed_by_run_id` |
| R01 positive control (no B) | `r10=12.0`, unchanged |
| R01 matched identity | `r10=12.5`, `raw_metrics.r10=12.0`, `identity_join="exact"`, `raw_checkpoint_id=checkpoint_A` |
| R07a `0.2793` + `_percent: 27.93` | `r10=27.93`, `unit=percent/sibling_key`, `per_metric_units={r10:_percent, r50:_percent}` |
| R07b only `dead_parameters_pct` | `r10=0.2793`, `unit=fraction/inferred_from_magnitude`, `per_metric_units={r10:"", r50:""}` |
| R08 two checkpoints, equal metrics | both `counted=true`, `duplicate_of=None`, mutual `same_metrics_as` |
