# 61 — parity_gate contract hardening

Scope: `workbench/backend/parity_gate.py` only (`native_measurements.py` untouched), plus
`workbench/tests/test_gate_contract_negatives.py` and
`workbench/tests/test_gate_parity_subchecks.py` (fixture).

Reproduced by `/tmp/cir-review-06e30e8c/review_06e30e8c/gate_probe.py` (exit 0 on HEAD): five of six
negative fixtures returned `passed=true, complete=true, failure_count=0`.

## Rule 1 — per-category scoping

`verdict.missing_by_category` and `verdict.mandatory_by_category` are filled per category inside the
artifact loop. `verdict.missing` is now the UNION over categories of required dimensions that lack a
mandatory check in that category, so one well-measured category can no longer carry a bare one.
`verdict.categories` is set before the loop so a pass can be asserted against the judged set.

Fixture that now fails correctly: `other_categories_empty` → `PARITY_MANDATORY_SUBCHECK_FAILED`,
`missing = [forward_parity, batch_parity, scoring_parity]`, per-category gaps recorded for `dress`
and `toptee` (empty for `shirt`). A category with no artifacts at all still yields
`PARITY_EVIDENCE_MISSING` (unchanged; `test_a_partial_evidence_set_is_not_a_pass`).

## Rule 2 — identity cross-check

New `_identity_cross_checks(native, adapter, category)` reads `IDENTITY_FIELDS` from BOTH top-level
documents — `model_id`, `checkpoint_id`, `checkpoint_identity`, `protocol_id`, `protocol`,
`query_order_sha256`, `gallery_order_sha256` — and emits one mandatory `input_parity` check per field
present on at least one side, `name=f"{category}.input.identity_agreement.{field}"`,
`measured={"native": ..., "adapter": ...}`. Mismatch or one-sided presence ⇒ `ok=False`. No payload
boolean is consulted. Field names come from the real artifacts: `g1_g2.encoder.*` carry `model_id`
and `protocol`; `docs/reviews/20261108T015603+0700/composition/**` carry `checkpoint_id`,
`checkpoint_identity`, `protocol_id`; the probe supplies `query_order_sha256`. The check is skipped
only when the adapter body is EMPTY: an adapter that measured nothing makes no identity claim, and
`INCOMPLETE` is the honest verdict there (the probe's `empty_adapter_control` control).

Fixture: `contradictory_identity` → `PARITY_MANDATORY_SUBCHECK_FAILED`, 4 identity failures naming
both values (model_id / checkpoint_id / protocol_id / query_order_sha256) plus the relation failure.

## Rule 3 — relation mandatory for a relation-drawing model

`_relation_checks` now takes `model_id`. `RELATION_DRAWING_MODELS = ("encoder",)`; a pair is also
treated as relation-drawing when EITHER document carries a `relation_state` key at all (including
`not_recorded`). Those produce `mandatory=True, ok=False`. A model with no `relation_state` on
either side keeps `mandatory=False, ok=False`, so `relation_parity` is False rather than absent.

Fixtures: `complete_fixture_no_relation` → `PARITY_MANDATORY_SUBCHECK_FAILED` (1 failure,
`relation.<cat>.digest`).

## Rule 3a — absent evidence is not a failed claim (per category)

A check that reports ABSENCE (`ParityCheck.absence=True`: an unrecorded relation digest, "no rank
comparison recorded") measures nothing, so it must not be reported as a failed claim on an artifact
that made no claim. The rule is decided PER CATEGORY, from `bool(adapter_body)`:

* adapter body empty ⇒ absence checks are demoted to non-mandatory (`_unclaimable`), the dimension
  reports `missing`, and the verdict is `INCOMPLETE` with `failure_count` 0;
* adapter body has any content ⇒ absence checks stay mandatory, so a category that claims forward or
  scoring results but records no relation digest is `MANDATORY_FAILED` beside its real claims.

`empty_adapter_control` (encoder, `{}`) → `INCOMPLETE`, `passed=false`, `failure_count` 0, `missing`
= `[forward_parity, batch_parity, scoring_parity]`, `relation_parity` False. A GLOBAL version of this
rule is wrong and `other_categories_empty` proves it: shirt claims results and lacks a digest
(1 genuine failure), dress/toptee claim nothing (missing), and the verdict is `MANDATORY_FAILED` —
a global "all failures are absence ⇒ INCOMPLETE" test would have thrown shirt's real failure away.

## Rule 4 — scoring needs rank evidence

`_scoring_evidence(checks, category)` inspects the dimension's checks after scores and ranks were
emitted. Scores without ranks ⇒ mandatory
`name=f"{category}.ranks.evidence_present", ok=False, measured="no rank comparison recorded"`.
Ranks without scores ⇒ the rank checks are kept, `scoring_parity` is removed from that category's
mandatory set, so rule 1 reports it `missing` ⇒ `INCOMPLETE`.

Fixtures: `score_only_no_rank_evidence` → `MANDATORY_FAILED` (`shirt.ranks.evidence_present`); the
reverse case is `test_rank_evidence_without_scores_is_missing_not_satisfied` → `INCOMPLETE`.

## Rule 5 — one tolerance source

`_score_checks`: both top-level `score_max_abs` and body `score_max_abs_declared` present and the
body is LOOSER ⇒ `_Malformed` naming both numbers ("a looser body tolerance cannot override a
stricter declared one"). Body STRICTER ⇒ body applied. One-sided ⇒ that one applied, `declared`
recorded as before. `harness_recorded` untouched, so no artifact's recorded `within_tolerance`
boolean is retro-changed.

Fixture: `local_tolerance_overrides_global_and_no_rank_evidence` → `PARITY_EVIDENCE_MALFORMED`
(detail names `1.0` and `1e-06`). No historical artifact is affected: LIMN's adapter carries no
`score_max_abs_declared`; ENCODER's body `1e-06` equals its top level.

## Rule 6 — pass invariant

`_check_invariants(verdict)` is called from `ParityVerdict.to_dict()` and asserts, for any
`passed=True`: `status == PASS`, no failures, no missing, at least one mandatory check, and every
required dimension present in `mandatory_by_category` for EVERY judged category. A future edit that
reintroduces a false pass raises instead of reporting.

## Probe before → after (same six fixtures)

| fixture | before | after | fails |
|---|---|---|---|
| complete_fixture_no_relation | PARITY_PASS | PARITY_MANDATORY_SUBCHECK_FAILED | 0→1 |
| empty_adapter_control | PARITY_INCOMPLETE | PARITY_INCOMPLETE (unchanged, correctly) | 0→0 |
| contradictory_identity | PARITY_PASS | PARITY_MANDATORY_SUBCHECK_FAILED | 0→5 |
| local_tolerance_overrides_global_and_no_rank_evidence | PARITY_PASS | PARITY_EVIDENCE_MALFORMED | 0→0 |
| score_only_no_rank_evidence | PARITY_PASS | PARITY_MANDATORY_SUBCHECK_FAILED | 0→2 |
| other_categories_empty | PARITY_PASS | PARITY_MANDATORY_SUBCHECK_FAILED | 0→1 (missing non-empty) |

Every `passed` moved to False where it was True. Probe run: copied to `/tmp/gate_probe_real.py`, its
import repointed at `/home/heheboiz/data/cir/workbench/backend/parity_gate.py` and its historical root
at the repo's `20261109-native15-composition-limn`. Its trailing
`assert results[case]['passed'] is True` block encodes the defect under test, so it was dropped from
the copy; the bundled probe itself is unmodified.

## Historical verdicts before → after

| model | before | after |
|---|---|---|
| encoder | PARITY_MANDATORY_SUBCHECK_FAILED, 13 failures, missing [] | same status, 16 failures, missing [] |
| limn | PARITY_MANDATORY_SUBCHECK_FAILED, 3 failures, missing [metrics_parity] | unchanged |

The +3 on encoder are exactly `relation.{dress,shirt,toptee}.digest` (rule 3: the artifacts record no
`relation_state` and ENCODER draws `learnable_relation`). Nothing else moved: identity cross-checks
agree (`model_id` = `encoder`, `protocol` = `fashioniq_val_split` on both sides for all three
categories), tolerance sources agree (body `1e-06` == top-level `1e-06`), and both models' scoring
already carried rank evidence.

## Fixture change (tests only)

`test_gate_parity_subchecks._minimal` now writes an identical `relation_state.sha256` on both sides
by default (`relation_sha=...`), because a clean ENCODER fixture must be able to show both sides drew
the same tensor. `relation_sha=None` omits the key, which the new negative tests use.
`test_an_empty_adapter_artifact_does_not_pass_and_is_not_complete` now accepts
`INCOMPLETE` OR `MANDATORY_FAILED` (rule 3 makes the encoder case the latter) while still asserting
`missing` names the three unmeasured dimensions.

## Verification commands

```
workbench/envs/ctl312h/bin/python -m pytest \
  workbench/tests/test_gate_contract_negatives.py workbench/tests/ \
  -k parity_gate -q --basetemp=/tmp/r02_gate_bt
```

`-k parity_gate` selects nothing in this repo (no test name carries that substring), so the gate
suites were run explicitly:
`workbench/tests/test_gate_contract_negatives.py workbench/tests/test_gate_parity_subchecks.py`
→ **57 passed**. The named command deselects everything (2908 deselected, 0 selected), which is a
selection artefact, not a pass — recorded here so it is not mistaken for coverage.

The wider selection `-k "gate"` over all of `workbench/tests/` was also run against the same commit:

```
workbench/envs/ctl312h/bin/python -m pytest workbench/tests/ \
  workbench/tests/test_gate_contract_negatives.py -k "gate" -q --basetemp=/tmp/r02_gate_bt
→ 240 passed, 1 skipped, 2670 deselected in 107s
```

Nothing outside `parity_gate.py`'s own contract regressed.

A wider selection (`-k "gate or verdict"`, excluding four modules another agent currently has
mid-edit — `test_airknow_replay/test_clvc_net_replay/test_tgcir_replay` fail collection on
`cache_requirement` missing from `workbench/replay/common.py`) reached ~90% with one failure in a
module unrelated to this gate (`test_pc_launcher.py`); every gate-owned module was run individually
and passes: `test_gate_contract_negatives`, `test_gate_parity_subchecks`,
`test_gate_r3_mask_join`, `test_csmcir_aggregate_api_parity`, `test_final_hardening_regressions`
(110 passed), `test_release_gate_scenarios` + `test_r19_aggregate_parsers` (24 passed),
`test_circus_env_gate_omp` (27), `test_paper_gate` (65).
