# `parity-absolute-relative/v2` — the recorded shape, written before the code

Status: **contract**. This file defines the evidence shape, so the writer (`t4_parity.py`,
`encoder_parity.py`) and the reader (`workbench/backend/parity_gate.py`) are built against one
written rule instead of two guesses. No artifact on disk was regenerated for this document.

## Why v2 is a contract change, not a relabel

`parity-relative/v1` judges a comparison from two aggregates:

    fail if  max_abs > atol   OR   max_rel > rtol

`max_rel` divides by the reference component, so a unit-norm embedding with a component near `1e-8`
turns a `1e-7` absolute error into `7.8`. The declared v2 rule does not collapse:

    fail if  abs(adapter - native) > atol + rtol * abs(native)   for ANY component

Native is the REFERENCE side (`b`); adapter is the measured side (`a`). That rule is defined per
component, so it cannot be evaluated from `max_abs`/`max_rel` alone. Until now the gate refused
every artifact declaring v2 (`_declared_standard` raised `_Malformed`) and `evaluate_abs_rel` had no
production caller. Both are now wired; neither tolerance and no v1 check was moved.

## Additive by construction

* The block is a NEW top-level key, `component_parity`, on the paired artifacts. A v1 reader that
  ignores unknown keys reads the artifact exactly as before.
* The version is DECLARED, never inferred: the artifact carries top-level
  `"standard": "parity-absolute-relative/v2"` (both sides, same string). An artifact with no
  `standard` key is still judged `parity-relative/v1`, unchanged.
* An artifact that declares v2 and carries no `component_parity` block stays `_Malformed`. A bare
  label over v1 aggregates is never a v2 result.

## Shape

```json
"standard": "parity-absolute-relative/v2",
"component_parity": {
  "standard": "parity-absolute-relative/v2",
  "declared_rule": "abs(adapter - native) <= atol + rtol * abs(native)",
  "reference_side": "native",
  "chunk_rule": "full when components_total <= 4096, else the first 512 components plus every component whose normalized violation exceeds 1",
  "scopes": {
    "<scope name>": {
      "dimension": "forward_parity" | "batch_parity",
      "shape": [4, 512],
      "components_total": 2048,
      "components_emitted": 2048,
      "coverage": "full" | "chunked",
      "indices": [0, 1, 2047],
      "reference": [0.1, 0.2, 0.3],
      "actual":    [0.1, 0.2, 0.3],
      "sha256": {"reference": "<64 hex>", "actual": "<64 hex>", "delta": "<64 hex>"},
      "mismatch_count": 0,
      "max_abs": 1.2e-07,
      "worst_normalized": 0.41,
      "offenders": [{"index": 2047, "reference": 0.3, "actual": 0.31, "normalized": 1.7}]
    }
  }
}
```

* `<scope name>` is the artifact's own comparison name, so a reader can join the component record to
  the aggregate block it belongs to: `sample_forward.query`, `sample_forward.target`,
  `layer4_forward_sample.query_val_pixels`, `layer4_forward_sample.target`,
  `population.features_vs_native_val_control`, `population.gallery_vs_native`. The scope set is
  whatever that run actually compared — and exactly the subset whose reference array is native.
* `dimension` names the verdict dimension the scope feeds. The componentwise rule is recorded only
  for `forward_parity`, because the rule names NATIVE as the reference and that is the only dimension
  where both producers hold a native array aligned against an adapter array. `batch_parity`
  (`batch 1 vs N`) and `cache_off_restart` compare the adapter against its OWN output; their v1
  aggregate checks judge them exactly as before, and labelling their left-hand array "native" would
  misdescribe the measurement. Any other `dimension` is malformed.
* `indices` are FLAT indices into `shape`, ascending, unique, `< components_total`. `reference[i]`
  and `actual[i]` are the values at `indices[i]`, so a chunk is self-describing.
* `sha256` is over the FULL arrays, canonical little-endian float64, in flat order
  (`np.ascontiguousarray(a, dtype="<f8").tobytes()`). A reviewer holding the run's arrays can prove
  the emitted chunk came from them; the chunk itself is not a new measurement.

## The four outcomes the reader derives

The gate re-runs `evaluate_abs_rel` over the emitted `(reference, actual)` pairs. It never derives a
verdict from `max_abs`, `max_rel`, `mismatch_count` or `worst_normalized`; those are cross-checks
against the derivation.

| derivation | verdict for the scope |
|---|---|
| every emitted pair satisfies `abs(a-b) <= atol + rtol*abs(b)` | check ok |
| any emitted pair violates it | check FAILS, `measured.offending_coordinates` names it |
| block missing while `standard: v2` is declared | pair is `_Malformed` |
| any emitted value non-finite | pair is `_Malformed` |

`ok` needs ALL of: zero derived offenders; `mismatch_count` equal to the derived count; `offenders`
holding exactly those indices; `components_emitted == len(indices) == len(reference) == len(actual)`
and `<= components_total`; `coverage == "full"` only when `components_emitted == components_total`;
all three digests present and 64 hex characters.

A check that cannot be evaluated from the recorded data is malformed, never ok. Concretely:
`components_total <= 0`, a shape whose product disagrees with it, an index out of range, a duplicated
index, a non-numeric or non-finite value, an `offenders` list whose length disagrees with
`mismatch_count`, an offender index that is not among the emitted indices, a `coverage` value outside
`{"full", "chunked"}`, a digest that is not 64 hex — all raise, and the pair becomes
`PARITY_EVIDENCE_MALFORMED`.

Tolerances are NOT read from the block. `atol`/`rtol` come from the pair's own declared
`tolerances` table, selected by `dimension`: `forward_parity` → `forward_max_abs`, `forward_max_rel`.
A block therefore cannot loosen the tolerance it is judged against.

## Mixed standards, and where the block may sit

An artifact with no `standard` key is a v1 result (that is what makes the change additive). So ONE
side v1-by-default and the other declaring v2 is a MIXED pair: the two sides were measured under
different rules, one verdict cannot be reported under both, and the pair is `_Malformed`. Both sides
absent is not a disagreement — it is the unchanged v1 record.

The block is written by the phase that holds BOTH arrays (the adapter, in both producers), so it
normally sits on one side only. A block on either side is accepted; a block on BOTH sides is accepted
only when every scope they share is identical, because two records of one comparison that disagree
are an artifact contradicting itself.

## Aggregates are never sufficient

`max_abs` / `max_rel` are still recorded, still judged as v1 checks, and are NEVER the source of a
componentwise verdict:

* the gate derives its own `max_abs` and worst normalized violation
  (`max(|a-b| / (atol + rtol*|b|))`) over the emitted values;
* when `coverage == "full"` the recorded `max_abs`/`worst_normalized` must MATCH the derivation
  (`math.isclose`, rel `1e-9`) or the artifact contradicts itself and is malformed;
* when `coverage == "chunked"` the recorded values describe the full array, so they are only
  required to be at least the derived values — never used to declare a scope ok.

## Chunk rule

Full arrays are emitted when `components_total <= 4096`. Above that the dump is the first 512
components PLUS every violating component, ascending, deduplicated. This keeps a full-population
scope (~1.5 M components) inside a reviewable file while still emitting every coordinate a failure
verdict could name. `coverage: "chunked"` is part of the record, so a reader can never mistake a
chunk for the whole array.

## What stays exactly as it was

* v1 artifact pairs: judged under `parity-relative/v1`, same checks, same tolerances, same verdict.
  No `standard` key, no block, no change.
* A v2 REQUEST against v1 artifacts: still advisory only, `applies: False`.
* The two sides declaring different standards: still `_Malformed`.
* **Known ceiling, deliberate:** the v1 aggregate checks remain mandatory for a v2-declaring pair, so
  a v2 pair whose aggregate `max_rel` exceeds `rtol` still fails even when every component is inside
  the combined rule. Deleting a declared v1 check for v2 artifacts would relax a check and is NOT
  done here; it is a separate, explicit decision (a v2-only dimension set) with its own record.
  Upgrade path: vote on that decision, then gate the v1 aggregate check on
  `applied_standard == STANDARD_RELATIVE_V1`.

## Where it is implemented

* Writer: `component_scope(...)` in each producer; emitted only by the phase that holds BOTH arrays
  (the adapter phase), lifted to the document's top level beside `relation_state`.
* Reader: `_component_scopes`, `_component_checks` and the v2 branch of `_declared_standard` in
  `workbench/backend/parity_gate.py`; `evaluate_abs_rel` is called per emitted component.
* Tests: `workbench/tests/test_parity_absolute_relative_v2.py`.

## Tests updated

`workbench/tests/test_gate_parity_subchecks.py` pinned four MEASURED values from `g1_g2.encoder.*`.
Seeding the ENCODER relation moved those measurements legitimately (`non_tie_count` 1 → 0; several
`max_rel` ratios changed), so the tests failed on incidental values, not on a regression. Each was
replaced by the invariant the check exists to protect. No check was deleted and no float re-pinned.

| test | was | now | why the invariant is stronger |
|---|---|---|---|
| `test_encoder_gate_is_not_pass_on_head` | `assert any("non_tie_changes" …)` in the failures, plus the three exact `max_rel` floats `0.0031308995258445375 / 0.0077950176161260265 / 0.011646653843520266` | asserts `non_tie_changes` is **absent** from failures, one failing batch check per category each `measured > declared == 1e-3`, and the CAUSE: `relation_state.sha256` byte-equal on both sides and `non_tie_count == 0` in all three categories | the old form asserted a measurement, so it flipped when the measurement improved. The new form asserts the declared-tolerance relation (holds at any magnitude) and the condition that explains a clean rank dimension. A re-pinned `1` would have re-encoded a defect the seeding fixed |
| `test_the_historical_record_survives_untouched` | `recorded_non_tie_rank_changes == 1`, `recorded_within_tolerance["batch_size_1_vs_n"] is False` | each echoed field `==` the artifact's own recorded value (`non_tie_count`, both `within_tolerance` labels, the whole `declared_tolerances` table) | `historical` is an ECHO, so the invariant is equality with its source. Literals pinned a reading that must track legitimate re-measurement; equality holds for every future run and fails if the gate ever stops echoing faithfully |
| `test_the_dimensions_are_not_conflated` | `dimensions["scoring_parity"] is False` (a snapshot of one run's outcome) | for every dimension, `dimensions[d] is all(check.ok for mandatory checks in d)`; every failure's dimension is False; `batch_parity`/`forward_parity` fail only on `.max_rel`; `input_parity` True; `observer_parity` never equals `forward_parity` | asserts the DERIVATION between `checks` and `dimensions` instead of which dimensions happen to fail. It is strictly stronger: it still catches a dimension carried by another dimension's checks (the `other_categories_empty` defect) while surviving a legitimately clean `scoring_parity` |
| `test_the_harness_tests_max_rel_on_some_keys_and_not_others` | `gallery_vs_native.max_rel > 1.0`, `features_vs_native_val_control.max_rel > 0.5` (one run's ratios) | both `> declared forward_max_rel`, both with `max_abs <= declared forward_max_abs`; gallery labelled `within_tolerance: True` while the query label is `False`; the gate names `("shirt.population.gallery_vs_native", "max_rel")` as a disagreement and records `gallery.declared == declared_rel` | the test's subject is WHICH comparator each producer calls, not the magnitude. Comparing to the declared tolerance states the incoherence (gallery exceeds the same `max_rel` yet reads within tolerance; the absolute delta is inside atol, so a near-zero denominator drives it) at any ratio. The old `> 1.0` threshold recorded one run and broke at 0.53 |

Mutation proof, so the surviving assertions can still fail:

* `non_tie_count: 1` in a `/tmp` copy of the shirt adapter → `test_encoder_gate_is_not_pass_on_head`
  fails at the cause assertion and `test_the_historical_record_survives_untouched` fails at the echo
  equality.
* `gallery_vs_native.max_rel: 1e-06` in a `/tmp` copy (wiring change) →
  `test_the_harness_tests_max_rel_on_some_keys_and_not_others` fails at
  `gallery max_rel > declared_rel`.
* A gate copy with `verdict.dimensions[dimension] = True` injected → both
  `test_the_dimensions_are_not_conflated` and the harness test fail.

Runs: `pytest workbench/tests/test_gate_parity_subchecks.py
workbench/tests/test_parity_absolute_relative_v2.py -q` → **46 passed**.
`evaluate_all()` on the real evidence is byte-identical to the pre-change gate
(`sha256 bd87d8b5…f9bd9`), so the test edit changed no gate behaviour.
