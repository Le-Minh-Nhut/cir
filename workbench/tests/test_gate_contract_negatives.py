"""G-01 on HEAD: the seven negatives the parity gate must produce, each as a fixture.

Why this file exists
--------------------
`test_gate_parity_subchecks.py` proves the gate reads the campaign artifacts correctly. It does not
prove what the gate does with evidence that is EMPTY, malformed, self-contradicting or declared under
a different standard. Those are the cases where a verdict silently becomes `PARITY_PASS`:

* an adapter artifact that is `{}` -- no forward measurement, no batch measurement, no scoring --
  currently passes and reports `complete: True`, so `PARITY_PASS` means "nothing was checked";
* a `scoring_ranking` block whose `score_within_tolerance: true` sits beside a `score_max_abs` of
  `0.1` against a declared `1e-6` -- the verdict is read from the harness boolean, not the numbers;
* a model id outside the two hardcoded ones is `PARITY_NOT_RUN` even when its artifacts are on disk,
  so a third model can never be gated at all;
* a malformed document (a list where an object belongs) raises `AttributeError` out of the gate;
  an infinite tolerance accepts any measured value;
* `None` for an unmeasured subcheck, and an artifact missing whole dimensions, still read as PASS.

Each test below is one of the seven required negatives. Written first: every one of them fails on
HEAD for the reason named in its docstring.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend import parity_gate as gate

CATEGORY = "shirt"

TOLERANCES = {
    "identity": "exact, zero tolerance",
    "forward_max_abs": 1e-05,
    "forward_max_rel": 1e-04,
    "forward_min_cosine": 0.9999999,
    "batch_max_abs": 1e-04,
    "batch_max_rel": 1e-03,
    "score_max_abs": 1e-06,
    "tie_epsilon": 1e-06,
    "metrics": "exact hit-count equality",
}


def _native_identity(**overrides) -> dict:
    layer = {
        "query_count_native": 10, "query_count_annotations": 10,
        "gallery_count_native": 20, "gallery_count_rebuilt_from_annotations": 20,
        "gallery_order_identical": True, "reference_ids_identical": True,
        "target_ids_identical": True, "model_text_identical_to_author_composition": True,
        "gallery_has_duplicates": False,
    }
    layer.update(overrides)
    return layer


#: The identifiers a hit-count comparison needs on BOTH sides before it can certify anything.
PROVENANCE = {"protocol": "fashioniq_val_split", "checkpoint_id": "testmodel_checkpoint",
              "query_order_sha256": "a" * 64, "gallery_order_sha256": "b" * 64,
              "model_id": "testmodel"}


def _metrics_detail(*, native_hits: dict | None = None, adapter_hits: dict | None = None,
                    query_count: int = 10, identical_set: bool = True) -> dict:
    native_hits = native_hits or {"r1": 4, "r10": 7, "r50": 9}
    adapter_hits = adapter_hits or dict(native_hits)
    return {"cutoffs": [1, 10, 50],
            "native": {"hits": dict(native_hits), "query_count": query_count},
            "adapter": {"hits": dict(adapter_hits), "query_count": query_count},
            "eligible_query_set_identical": identical_set,
            "equality": {k: native_hits[k] == adapter_hits[k] for k in native_hits}}


def _native(*, tolerances: dict | None = None, metrics: bool = True) -> dict:
    body = {"layer1_identity": _native_identity()}
    document = {"category": CATEGORY, "phase": "native", "model_id": "testmodel",
                "tolerances": dict(tolerances or TOLERANCES), "native": body}
    if metrics:
        body["native_metrics_match_feature_rerank"] = True
        document["native_metrics_parity_detail"] = _metrics_detail()
        document.update(PROVENANCE)
    return document


def _forward(value: float = 0.0, *, flag: bool | None = True, cosine: float = 1.0) -> dict:
    block = {"max_abs": value, "max_rel": value, "min_cosine": cosine, "bitwise_equal": False,
             "shape": [2, 4]}
    if flag is not None:
        block["within_tolerance"] = flag
    return block


def _adapter(*, tolerances: dict | None = None, forward: dict | None = None,
             scoring: dict | None = None, batch: dict | None = None,
             population: bool = True, metrics_present: bool = True) -> dict:
    """A complete adapter body. Every knob defaults to a value that would pass on its own."""
    body: dict = {
        "layer2_pixels": {"adapter_decode_vs_native_val_decode": {
            "max_abs": 0.0, "max_rel": 0.0, "bitwise_equal": True, "min_cosine": 1.0,
            "shape": [2, 4]}},
        "layer4_forward_sample": {
            "query_val_pixels": forward if forward is not None else _forward(),
            "target": _forward(),
        },
        "scoring_ranking": scoring if scoring is not None else {
            "score_max_abs_delta_vs_dot": 1.19e-07, "score_pairs_compared": 8,
            "score_max_abs_declared": 1e-06, "score_within_tolerance": True,
            "adapter_vs_argsort_unmasked": {"non_tie_count": 0},
            "rank_differences_vs_native_val_control": {"count": 0, "non_tie_count": 0,
                                                       "min_gap_abs": 0.0},
        },
    }
    if population:
        body["population"] = {
            "features_vs_native_val_control": _forward(1e-06),
            "gallery_vs_native": _forward(1e-06),
        }
    if batch is not None:
        body["batch_size_1_vs_n"] = batch
    elif metrics_present:
        body["batch_size_1_vs_n"] = {"max_abs": 3e-07, "max_rel": 1e-06, "min_cosine": 0.9999999,
                                     "bitwise_equal": False, "shape": [2, 4],
                                     "within_tolerance": True}
    document = {"category": CATEGORY, "phase": "adapter", "model_id": "testmodel",
                "tolerances": dict(tolerances or TOLERANCES), "adapter": body}
    # Provenance travels with the comparison it identifies: a pair that records no hit-count
    # comparison at all declares none of these either, exactly like the real LIMN pair did.
    if metrics_present:
        document.update(PROVENANCE)
    return document


def _names(model: str) -> tuple[str, str]:
    return (f"g1_g2.{model}.{CATEGORY}.native.json", f"g1_g2.{model}.{CATEGORY}.adapter.json")


def _write(root: Path, model: str, native: object, adapter: object) -> Path:
    native_name, adapter_name = _names(model)
    for name, body in ((native_name, native), (adapter_name, adapter)):
        (root / name).write_text(body if isinstance(body, str) else json.dumps(body),
                                 encoding="utf-8")
    return root


def _evaluate(root: Path, model: str = "testmodel", **kwargs) -> gate.ParityVerdict:
    return gate.evaluate_model(model, root=root, categories=(CATEGORY,), **kwargs)


# --------------------------------------------------------- negative 1: empty evidence is not a pass

def test_an_empty_adapter_artifact_does_not_pass_and_is_not_complete(tmp_path: Path) -> None:
    """`{}` plus G1 identity is not evidence: no forward, batch, scoring or metric measurement."""
    _write(tmp_path, "testmodel", _native(), {"category": CATEGORY, "phase": "adapter",
                                              "tolerances": dict(TOLERANCES), "adapter": {}})
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.INCOMPLETE, verdict.detail
    assert verdict.passed is False
    assert verdict.complete is False, "an artifact with no measurement must not report complete"
    # An empty adapter records NO CLAIM, so nothing was disproved: `failure_count` must be 0 and the
    # gaps are reported through `missing`. (The encoder relation digest is absent here too, and it is
    # absent evidence, not a failed claim -- `complete_fixture_no_relation` covers the case where the
    # same absences sit beside real claims and do fail.)
    assert verdict.failures == [], verdict.failures
    # `metrics_parity` is MISSING too, and for a reason the fix introduced on purpose: the native side
    # records hit counts and the adapter side records neither the counts nor the provenance a hit-count
    # comparison needs (checkpoint, protocol, order digests). A comparison only one side identified is
    # not a comparison, so the dimension is unmeasured rather than satisfied. It stays an ABSENCE here
    # -- the adapter claims nothing at all, so nothing about it was disproved.
    assert set(verdict.missing) == {"forward_parity", "batch_parity", "scoring_parity",
                                    "metrics_parity"}, verdict.missing


def test_an_artifact_missing_one_mandatory_dimension_is_incomplete_not_pass(tmp_path: Path) -> None:
    """No batch comparison and no hit-count equality: two dimensions were never measured."""
    _write(tmp_path, "testmodel", _native(metrics=False), _adapter(metrics_present=False))
    verdict = _evaluate(tmp_path)
    assert verdict.status != gate.PASS
    assert verdict.passed is False and verdict.complete is False
    assert verdict.status == gate.INCOMPLETE
    assert "batch_parity" in verdict.missing and "metrics_parity" in verdict.missing


# ------------------------------------------------- negative 2: the number, never the flag

def test_a_self_declared_within_tolerance_flag_cannot_override_the_numbers(tmp_path: Path) -> None:
    """`0.1` against a declared `1e-6`, with the artifact claiming `within_tolerance: true`."""
    scoring = {"score_max_abs_delta_vs_dot": 0.1, "score_pairs_compared": 8,
               "score_max_abs_declared": 1e-06, "score_within_tolerance": True,
               "adapter_vs_argsort_unmasked": {"non_tie_count": 0},
               "rank_differences_vs_native_val_control": {"count": 0, "non_tie_count": 0,
                                                          "min_gap_abs": 0.0}}
    _write(tmp_path, "testmodel", _native(), _adapter(scoring=scoring))
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MANDATORY_FAILED, verdict.detail
    assert verdict.passed is False
    failed = [check for check in verdict.failures if "scores.max_abs_vs_dot" in check.name]
    assert failed, [check.name for check in verdict.failures]
    assert failed[0].declared == 1e-06 and failed[0].measured == 0.1
    assert failed[0].harness_recorded is True, "the self-declared flag must be kept as the record"


def test_a_self_declared_flag_also_cannot_override_a_failing_forward_delta(tmp_path: Path) -> None:
    _write(tmp_path, "testmodel", _native(), _adapter(forward=_forward(0.1, flag=True)))
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MANDATORY_FAILED
    assert any("max_abs" in check.name for check in verdict.failures)


def test_a_missing_declared_tolerance_cannot_be_read_as_a_pass(tmp_path: Path) -> None:
    """A comparison the artifact declares no tolerance for is not silently satisfiable."""
    tolerances = dict(TOLERANCES)
    tolerances.pop("score_max_abs")
    scoring = {"score_max_abs_delta_vs_dot": 0.1, "score_pairs_compared": 8,
               "score_within_tolerance": True}
    _write(tmp_path, "testmodel", _native(tolerances=tolerances),
           _adapter(tolerances=tolerances, scoring=scoring))
    verdict = _evaluate(tmp_path)
    assert verdict.status != gate.PASS, verdict.detail


# --------------------------------------------------------- negative 3: a third model is gated

def test_a_third_model_with_artifacts_is_gated_not_not_run(tmp_path: Path) -> None:
    """`EVIDENCED_MODELS` is a discovery seed, never the set of models the gate will judge."""
    _write(tmp_path, "habit", _native(), _adapter())
    verdict = _evaluate(tmp_path, model="habit", **{})
    assert verdict.status != gate.NOT_RUN, verdict.detail
    assert verdict.status == gate.PASS
    assert verdict.passed is True


def test_a_third_model_with_failing_evidence_fails(tmp_path: Path) -> None:
    _write(tmp_path, "pair", _native(), _adapter(forward=_forward(0.1)))
    verdict = gate.evaluate_model("pair", root=tmp_path, categories=(CATEGORY,))
    assert verdict.status == gate.MANDATORY_FAILED
    assert verdict.passed is False


def test_evaluate_all_discovers_models_beyond_the_hardcoded_two(tmp_path: Path) -> None:
    """Discovery is by filename; a model name the campaign never listed must still be judged."""
    for model in ("encoder", "habl"):
        for category in ("dress", "shirt", "toptee"):
            native_name, adapter_name = (f"g1_g2.{model}.{category}.native.json",
                                         f"g1_g2.{model}.{category}.adapter.json")
            (tmp_path / native_name).write_text(json.dumps(_native()), encoding="utf-8")
            (tmp_path / adapter_name).write_text(json.dumps(_adapter()), encoding="utf-8")
    result = gate.evaluate_all(root=tmp_path)
    assert "habl" in result["models"], sorted(result["models"])
    assert result["models"]["habl"]["status"] == gate.PASS
    assert "habl" in result["passed_models"], result["passed_models"]


# ---------------------------------------- negative 4: malformed, non-finite, contradictory

def test_a_malformed_document_is_malformed_not_a_crash(tmp_path: Path) -> None:
    _write(tmp_path, "testmodel", "[1, 2, 3]", _adapter())
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MALFORMED, verdict.detail
    assert verdict.passed is False
    assert "object" in verdict.detail.lower() or "json" in verdict.detail.lower()


def test_an_infinite_tolerance_cannot_accept_any_measurement(tmp_path: Path) -> None:
    tolerances = dict(TOLERANCES)
    tolerances["forward_max_abs"] = math.inf
    _write(tmp_path, "testmodel", _native(tolerances=tolerances),
           _adapter(tolerances=tolerances, forward=_forward(999.0)))
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MALFORMED, verdict.detail
    assert verdict.passed is False


def test_a_nan_measurement_fails_rather_than_passing(tmp_path: Path) -> None:
    _write(tmp_path, "testmodel", _native(),
           _adapter(forward=_forward(math.nan, cosine=math.nan)))
    verdict = _evaluate(tmp_path)
    assert verdict.status in {gate.MALFORMED, gate.MANDATORY_FAILED}, verdict.status
    assert verdict.passed is False


def test_a_non_numeric_measurement_is_malformed_not_silently_dropped(tmp_path: Path) -> None:
    forward = _forward()
    forward["max_abs"] = "0.1"
    _write(tmp_path, "testmodel", _native(), _adapter(forward=forward))
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MALFORMED, verdict.detail
    assert verdict.passed is False


def test_a_negative_tolerance_is_refused(tmp_path: Path) -> None:
    tolerances = dict(TOLERANCES)
    tolerances["batch_max_rel"] = -1e-03
    _write(tmp_path, "testmodel", _native(tolerances=tolerances),
           _adapter(tolerances=tolerances))
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MALFORMED, verdict.detail


def test_non_object_native_body_is_malformed(tmp_path: Path) -> None:
    native = _native()
    native["native"] = [1, 2, 3]
    _write(tmp_path, "testmodel", native, _adapter())
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MALFORMED, verdict.detail


# ---------------------------------- negative 5: four distinct verdicts, None is never a pass

def test_the_four_verdicts_are_distinct_values() -> None:
    assert len({gate.PASS, gate.INCOMPLETE, gate.MANDATORY_FAILED, gate.NOT_RUN}) == 4


def test_none_for_an_unmeasured_subcheck_is_not_a_pass(tmp_path: Path) -> None:
    """`non_tie_count: null` is an unmeasured comparison, not zero non-tie changes."""
    scoring = {"score_max_abs_delta_vs_dot": 1.19e-07, "score_max_abs_declared": 1e-06,
               "score_within_tolerance": True,
               "rank_differences_vs_native_val_control": {"count": None,
                                                          "non_tie_count": None}}
    _write(tmp_path, "testmodel", _native(), _adapter(scoring=scoring))
    verdict = _evaluate(tmp_path)
    assert verdict.status != gate.PASS, verdict.detail
    assert verdict.passed is False
    assert all(check.ok is not None for check in verdict.checks)


def test_a_bool_measurement_is_not_a_number(tmp_path: Path) -> None:
    forward = _forward()
    forward["max_abs"] = False  # `False == 0` would pass a numeric comparison
    _write(tmp_path, "testmodel", _native(), _adapter(forward=forward))
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MALFORMED, verdict.detail


# -------------------- negative 6: the applied standard is the pinned one, advisory is separate

def test_the_applied_standard_is_the_one_the_artifact_declares(tmp_path: Path) -> None:
    native = _native()
    native["standard"] = gate.STANDARD_RELATIVE_V1
    adapter_body = _adapter()
    adapter_body["standard"] = gate.STANDARD_RELATIVE_V1
    _write(tmp_path, "testmodel", native, adapter_body)
    verdict = _evaluate(tmp_path, standard=gate.STANDARD_ABS_REL_V2)
    assert verdict.applied_standard == gate.STANDARD_RELATIVE_V1
    assert verdict.requested_standard == gate.STANDARD_ABS_REL_V2
    assert all(check.standard == gate.STANDARD_RELATIVE_V1 for check in verdict.checks)
    assert verdict.advisory and verdict.advisory[0]["applies"] is False


def test_a_v1_failure_is_not_relabelled_by_a_v2_request(tmp_path: Path) -> None:
    _write(tmp_path, "testmodel", _native(),
           _adapter(forward=_forward(0.1, flag=False)))
    verdict = _evaluate(tmp_path, standard=gate.STANDARD_ABS_REL_V2)
    assert verdict.status == gate.MANDATORY_FAILED
    assert verdict.applied_standard == gate.STANDARD_RELATIVE_V1
    assert verdict.passed is False


def test_evaluate_all_separates_applied_from_requested(tmp_path: Path) -> None:
    _write(tmp_path, "encoder", _native(), _adapter())
    result = gate.evaluate_all(root=tmp_path, standard=gate.STANDARD_ABS_REL_V2)
    assert result["standard"] == gate.STANDARD_RELATIVE_V1, "applied standard, never the request"
    assert result["requested_standard"] == gate.STANDARD_ABS_REL_V2


def test_an_artifact_declaring_v2_but_recording_only_maxima_is_not_applied_as_v2(
        tmp_path: Path) -> None:
    native = _native()
    native["standard"] = gate.STANDARD_ABS_REL_V2
    adapter_body = _adapter()
    adapter_body["standard"] = gate.STANDARD_ABS_REL_V2
    _write(tmp_path, "testmodel", native, adapter_body)
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MALFORMED, verdict.detail
    assert verdict.passed is False


def test_the_two_sides_disagreeing_on_the_declared_standard_is_malformed(tmp_path: Path) -> None:
    native = _native()
    native["standard"] = gate.STANDARD_RELATIVE_V1
    adapter_body = _adapter()
    adapter_body["standard"] = gate.STANDARD_ABS_REL_V2
    _write(tmp_path, "testmodel", native, adapter_body)
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MALFORMED, verdict.detail


# ------------------ negative 7: differing declared tolerances are refused, never reconciled

def test_native_and_adapter_tolerances_that_disagree_are_refused(tmp_path: Path) -> None:
    """The looser side must never silently win: the gate refuses and names both values."""
    loose = dict(TOLERANCES)
    loose["forward_max_abs"] = 1.0
    _write(tmp_path, "testmodel", _native(),
           _adapter(tolerances=loose, forward=_forward(0.5, flag=False)))
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MALFORMED, verdict.detail
    assert "forward_max_abs" in verdict.detail
    # Both declared values are named, so a reader can see which side was looser.
    assert "1.0" in verdict.detail and "1e-05" in verdict.detail


def test_identical_tolerances_on_both_sides_are_not_refused(tmp_path: Path) -> None:
    _write(tmp_path, "testmodel", _native(), _adapter())
    assert _evaluate(tmp_path).status == gate.PASS


def test_a_tolerance_declared_on_one_side_only_is_accepted_and_recorded(tmp_path: Path) -> None:
    """One-sided declaration is not a conflict: nothing disagrees, and the value is recorded."""
    adapter_tolerances = dict(TOLERANCES)
    adapter_tolerances.pop("tie_epsilon")
    _write(tmp_path, "testmodel", _native(), _adapter(tolerances=adapter_tolerances))
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.PASS, verdict.detail

# --------------- negative 8: the five defects the bundled probe reproduces (2026-11-11)

def _identity(**overrides) -> dict:
    """The identity-bearing fields an artifact MAY carry at its top level."""
    fields = {"model_id": "testmodel", "checkpoint_id": "fashion_iq", "protocol_id": "val_split"}
    fields.update(overrides)
    return fields

def test_a_complete_artifact_without_a_relation_digest_cannot_pass(tmp_path: Path) -> None:
    """ENCODER draws `learnable_relation`; an unrecorded digest is a mandatory failure for it."""
    _write(tmp_path, "encoder", _native(), _adapter())
    verdict = gate.evaluate_model("encoder", root=tmp_path, categories=(CATEGORY,))
    assert verdict.status == gate.MANDATORY_FAILED, verdict.detail
    failed = [check for check in verdict.failures if "relation" in check.name]
    assert failed and failed[0].mandatory is True
    assert verdict.dimensions["relation_parity"] is False

def test_a_model_with_no_relation_at_all_reports_the_dimension_false_not_absent(
        tmp_path: Path) -> None:
    """No `relation_state` on either side and a model that draws none: not mandatory, still False."""
    _write(tmp_path, "testmodel", _native(), _adapter())
    verdict = _evaluate(tmp_path)
    relation = [check for check in verdict.checks if check.dimension == "relation_parity"]
    assert relation and relation[0].mandatory is False and relation[0].ok is False
    assert verdict.dimensions["relation_parity"] is False
    assert verdict.dimensions["input_parity"] is True

def test_a_relation_state_key_recorded_as_not_recorded_is_mandatory(tmp_path: Path) -> None:
    """Carrying the key at all says a relation exists; `not_recorded` is then a failure."""
    native, adapter = _native(), _adapter()
    native["relation_state"] = {"sha256": None, "expected_sha256": "abc"}
    adapter["relation_state"] = {"sha256": None, "expected_sha256": "abc"}
    _write(tmp_path, "testmodel", native, adapter)
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MANDATORY_FAILED, verdict.detail
    assert any("relation" in check.name for check in verdict.failures)

def test_contradictory_identity_fields_cannot_pass(tmp_path: Path) -> None:
    """The native side alone cannot say whether the adapter loaded the same checkpoint."""
    _write(tmp_path, "testmodel", {**_native(), **_identity(model_id="model_A")},
           {**_adapter(), **_identity(model_id="model_B")})
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MANDATORY_FAILED, verdict.detail
    failed = [check for check in verdict.failures
              if check.name == f"{CATEGORY}.input.identity_agreement.model_id"]
    assert failed, [check.name for check in verdict.failures]
    assert failed[0].measured == {"native": "model_A", "adapter": "model_B"}
    assert verdict.dimensions["input_parity"] is False

def test_an_identity_field_on_one_side_only_cannot_pass(tmp_path: Path) -> None:
    _write(tmp_path, "testmodel", {**_native(), **_identity()}, _adapter())
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MANDATORY_FAILED, verdict.detail
    assert any("identity_agreement" in check.name for check in verdict.failures)

def test_a_required_dimension_measured_in_one_category_only_is_incomplete(tmp_path: Path) -> None:
    """Shirt is fully measured; dress and toptee carry an empty adapter body."""
    _write(tmp_path, "testmodel", _native(), _adapter())
    for category in ("dress", "toptee"):
        for side, body in (("native", _native()), ("adapter",
                           {"category": category, "phase": "adapter",
                            "tolerances": dict(TOLERANCES), "adapter": {}})):
            (tmp_path / f"g1_g2.testmodel.{category}.{side}.json").write_text(
                json.dumps({**body, "category": category}), encoding="utf-8")
    verdict = gate.evaluate_model("testmodel", root=tmp_path,
                                  categories=("dress", "shirt", "toptee"))
    assert verdict.status == gate.INCOMPLETE, verdict.detail
    assert {"forward_parity", "batch_parity", "scoring_parity"} <= set(verdict.missing)
    assert verdict.missing_by_category["dress"], verdict.missing_by_category
    assert verdict.missing_by_category["shirt"] == [], verdict.missing_by_category

def test_scores_without_rank_evidence_cannot_satisfy_scoring_parity(tmp_path: Path) -> None:
    """A score delta says the values agree, not that the ORDER still matches."""
    scoring = {"score_max_abs_delta_vs_dot": 1.19e-07, "score_pairs_compared": 8,
               "score_max_abs_declared": 1e-06, "score_within_tolerance": True}
    _write(tmp_path, "testmodel", _native(), _adapter(scoring=scoring))
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MANDATORY_FAILED, verdict.detail
    failed = [check for check in verdict.failures if check.name.endswith("ranks.evidence_present")]
    assert failed, [check.name for check in verdict.failures]
    assert failed[0].measured == "no rank comparison recorded"

def test_rank_evidence_without_scores_is_missing_not_satisfied(tmp_path: Path) -> None:
    """The reverse gap is `missing`, so the dimension is `INCOMPLETE`, never silently satisfied."""
    scoring = {"rank_differences_vs_native_val_control": {"count": 0, "non_tie_count": 0,
                                                          "min_gap_abs": 0.0}}
    _write(tmp_path, "testmodel", _native(), _adapter(scoring=scoring))
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.INCOMPLETE, verdict.detail
    assert "scoring_parity" in verdict.missing, verdict.missing

def test_a_looser_body_tolerance_cannot_override_a_stricter_declared_one(tmp_path: Path) -> None:
    """Body `1.0` against a declared `1e-6`, delta `0.1`: refused, both numbers named."""
    scoring = {"score_max_abs_delta_vs_dot": 0.1, "score_max_abs_declared": 1.0,
               "score_within_tolerance": True,
               "rank_differences_vs_native_val_control": {"count": 0, "non_tie_count": 0,
                                                          "min_gap_abs": 0.0}}
    _write(tmp_path, "testmodel", _native(), _adapter(scoring=scoring))
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MALFORMED, verdict.detail
    assert "1e-06" in verdict.detail and "1.0" in verdict.detail
    assert "looser body tolerance" in verdict.detail

def test_a_stricter_body_tolerance_is_allowed_and_applied(tmp_path: Path) -> None:
    """A narrower claim is not an override: the body's tighter number is what gets used."""
    scoring = {"score_max_abs_delta_vs_dot": 5e-07, "score_max_abs_declared": 1e-07,
               "score_within_tolerance": True,
               "rank_differences_vs_native_val_control": {"count": 0, "non_tie_count": 0,
                                                          "min_gap_abs": 0.0}}
    _write(tmp_path, "testmodel", _native(), _adapter(scoring=scoring))
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MANDATORY_FAILED, verdict.detail
    failed = [check for check in verdict.failures if "scores.max_abs_vs_dot" in check.name]
    assert failed and failed[0].declared == 1e-07 and failed[0].measured == 5e-07

def test_a_pass_carries_a_mandatory_check_in_every_required_dimension_every_category(
        tmp_path: Path) -> None:
    """The invariant itself: `passed` is impossible on a partly-measured mandatory set."""
    _write(tmp_path, "testmodel", _native(), _adapter())
    verdict = _evaluate(tmp_path)
    if not verdict.passed:
        assert verdict.status != gate.PASS
        return
    for category in verdict.categories:
        assert set(verdict.mandatory_by_category[category]) == set(gate.REQUIRED_DIMENSIONS)

def test_absent_evidence_is_incomplete_while_the_same_absence_beside_claims_fails(
        tmp_path: Path) -> None:
    """The distinction is exactly "did the artifact claim anything for this category".

    Both fixtures lack a relation digest. With an empty adapter body nothing was claimed, so nothing
    was disproved: `INCOMPLETE` with `failure_count` 0. Beside real forward/scoring claims the same
    absence contradicts the artifact's own claims and is a mandatory failure.
    """
    _write(tmp_path, "encoder", _native(),
           {"category": CATEGORY, "phase": "adapter", "tolerances": dict(TOLERANCES), "adapter": {}})
    silent = gate.evaluate_model("encoder", root=tmp_path, categories=(CATEGORY,))
    assert silent.status == gate.INCOMPLETE, silent.detail
    assert silent.failures == []
    assert "relation_parity" not in silent.missing  # the dimension exists, it is just False

    loud_root = tmp_path / "claims"
    loud_root.mkdir()
    _write(loud_root, "encoder", _native(), _adapter())
    loud = gate.evaluate_model("encoder", root=loud_root, categories=(CATEGORY,))
    assert loud.status == gate.MANDATORY_FAILED, loud.detail
    assert [check.name for check in loud.failures] == [f"relation.{CATEGORY}.digest"]
    assert loud.failures[0].absence is True

# ------------------------------- negative 8: a count compared to ITSELF can never fail

def _native_with_identity(**overrides) -> dict:
    native = _native()
    native["native"]["layer1_identity"] = _native_identity(**overrides)
    return native

def _identity_checks(verdict: gate.ParityVerdict) -> dict[str, gate.ParityCheck]:
    return {check.name: check for check in verdict.checks
            if check.name.startswith(f"{CATEGORY}.identity.")
            and check.name.endswith("_agreement")}

@pytest.mark.parametrize("pair, native_key, declared_key, native_value, declared_value", [
    ("query_count", "query_count_native", "query_count_annotations", 6001, 6003),
    ("gallery_count", "gallery_count_native", "gallery_count_rebuilt_from_annotations", 6001, 6003),
])
def test_a_count_that_disagrees_across_the_two_sides_fails(
        tmp_path: Path, pair: str, native_key: str, declared_key: str,
        native_value: int, declared_value: int) -> None:
    """The old code compared each count to ITSELF (`query_count_annotations == .query_count_annotations`).

    Two of the four mandatory G1 count checks could therefore never fail, while being reported as if
    they certified agreement. Here the native side scored a different number than the annotations
    declare, which is exactly what the checks exist to catch.
    """
    _write(tmp_path, "testmodel",
           _native_with_identity(**{native_key: native_value, declared_key: declared_value}),
           _adapter())
    verdict = _evaluate(tmp_path)
    check = _identity_checks(verdict)[f"{CATEGORY}.identity.{pair}_agreement"]
    assert check.ok is False, "a self-comparison passed: the tautology is back"
    assert check.mandatory is True and check.gate == "G1"
    assert check.dimension == "input_parity"
    assert check.declared == declared_value
    assert check.measured == {"native": native_value, "annotations": declared_value}
    assert check in verdict.failures
    assert verdict.status == gate.MANDATORY_FAILED and verdict.passed is False
    assert verdict.dimensions["input_parity"] is False

def test_agreeing_counts_pass_and_name_both_values(tmp_path: Path) -> None:
    _write(tmp_path, "testmodel", _native(), _adapter())
    verdict = _evaluate(tmp_path)
    pairs = _identity_checks(verdict)
    assert set(pairs) == {f"{CATEGORY}.identity.query_count_agreement",
                          f"{CATEGORY}.identity.gallery_count_agreement"}, sorted(pairs)
    for check in pairs.values():
        assert check.ok is True
        assert check.measured == {"native": check.declared,
                                  "annotations": check.declared}, check.measured

@pytest.mark.parametrize("pair, key, value", [
    ("query_count", "query_count_native", None),
    ("query_count", "query_count_annotations", None),
    ("query_count", "query_count_native", "2017"),
    ("gallery_count", "gallery_count_native", None),
    ("gallery_count", "gallery_count_rebuilt_from_annotations", None),
    ("gallery_count", "gallery_count_rebuilt_from_annotations", True),
])
def test_a_missing_or_non_integer_count_fails_and_names_the_missing_side(
        tmp_path: Path, pair: str, key: str, value: object) -> None:
    """An absent count is not agreement: it fails, and the note says which side is missing."""
    _write(tmp_path, "testmodel", _native_with_identity(**{key: value}), _adapter())
    verdict = _evaluate(tmp_path)
    check = _identity_checks(verdict)[f"{CATEGORY}.identity.{pair}_agreement"]
    assert check.ok is False
    assert check.mandatory is True
    assert key in check.note, check.note
    assert "no count recorded" in check.note, check.note
    assert check in verdict.failures
    assert verdict.status == gate.MANDATORY_FAILED

def test_an_absent_count_field_fails_not_just_a_null(tmp_path: Path) -> None:
    """Field omitted entirely: same failure class, still names the absent side."""
    native = _native()
    del native["native"]["layer1_identity"]["query_count_annotations"]
    _write(tmp_path, "testmodel", native, _adapter())
    verdict = _evaluate(tmp_path)
    check = _identity_checks(verdict)[f"{CATEGORY}.identity.query_count_agreement"]
    assert check.ok is False and "query_count_annotations" in check.note
    assert verdict.status == gate.MANDATORY_FAILED
