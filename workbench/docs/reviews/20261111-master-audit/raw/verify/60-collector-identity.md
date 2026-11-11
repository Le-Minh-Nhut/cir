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

## Verification

New tests: `workbench/tests/test_native_measurements_identity.py` — 13 passed.
Fail-before is proved by the bundled probe, which imports the `06e30e8c` snapshot and still shows
every defect (see below); each new test quotes the pre-fix output in its docstring.

`pytest workbench/tests/test_native_measurements_census.py -q` ⇒ **1 failed, 27 passed**.

`test_a_rollup_entry_is_not_double_counted_as_a_second_measurement` fails at
`assert len(identities) == len(set(identities))` (23 vs 21). That assertion is R08 itself: it builds
ranked-table identity as `(model_id, protocol_id, round(value, 4))` and demands uniqueness, which is
exactly the rule that demoted a distinct checkpoint to a duplicate. The two rows it now flags are
`hint` (`hint_fashioniq.json` vs `hint_doc_command.json`) and `tme` (`tme_fashioniq_0.json` vs
`tme_normal_path_full.json`) — different files, same checkpoint, no shared backing artifact, both
`counted: true` by design. The test encodes the wrong contract and is left unmodified.

Also updated earlier in this change and now green: `test_a_suffixed_sibling_key_declares_the_unit`
(its `fx_fraction_block` case now passes via `_agreed_suffix`).

Real catalog, before → after (`counts`):

* `with_metrics` 21 → 23 (the two demoted rows above are now counted)
* `duplicate_rows` 22 → 20
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
