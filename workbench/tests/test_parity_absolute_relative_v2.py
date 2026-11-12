"""`parity-absolute-relative/v2`: the writer, the reader, and the six legacy-compatibility cases.

The contract is written down in
`workbench/docs/reviews/20261112-parity-repair/V2_STANDARD.md` BEFORE either side existed, and this
file tests the pair of them:

* the WRITER (`component_scope`/`component_block` in `t4_parity.py` and `encoder_parity.py`) records
  the per-component aligned values each producer already holds, bound to the full arrays by sha256;
* the READER (`_component_checks` -> `_component_scope_check` in `workbench/backend/parity_gate.py`)
  derives ok / mismatch count / worst normalized violation from those values with
  `evaluate_abs_rel`, and never from the aggregate `max_abs`/`max_rel`.

Six legacy-compatibility outcomes are pinned by name: v1 stays v1, v2 is applied and recorded, a
mixed pair is malformed, a synthetic mismatch fails with its coordinate, NaN/Inf is malformed not a
pass, and a v2 label over aggregates only stays malformed.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from workbench.backend import parity_gate as gate

SCRIPTS = REPO / "workbench/docs/reviews/20261109-native15-composition-limn/scripts"
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

def _producer(name: str):
    """The producer module by path: neither is importable as a package member."""
    spec = importlib.util.spec_from_file_location(f"producer_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

@pytest.fixture(scope="module")
def t4():
    return _producer("t4_parity")

@pytest.fixture(scope="module")
def encoder():
    return _producer("encoder_parity")

# ------------------------------------------------------------------- fixture artifacts

def _native() -> dict:
    return {
        "category": CATEGORY, "phase": "native", "model_id": "testmodel",
        "tolerances": dict(TOLERANCES),
        "native": {
            "layer1_identity": {
                "query_count_native": 10, "query_count_annotations": 10,
                "gallery_count_native": 20, "gallery_count_rebuilt_from_annotations": 20,
                "gallery_order_identical": True, "reference_ids_identical": True,
                "target_ids_identical": True, "model_text_identical_to_author_composition": True,
                "gallery_has_duplicates": False,
            },
            "native_metrics_match_feature_rerank": True,
        },
    }

def _forward(value: float = 0.0, *, cosine: float = 1.0) -> dict:
    return {"max_abs": value, "max_rel": value, "min_cosine": cosine, "bitwise_equal": False,
            "shape": [2, 4], "within_tolerance": value <= TOLERANCES["forward_max_abs"]}

def _adapter(components: dict | None = None) -> dict:
    body: dict = {
        "layer2_pixels": {"adapter_decode_vs_native_val_decode": {
            "max_abs": 0.0, "max_rel": 0.0, "bitwise_equal": True, "min_cosine": 1.0,
            "shape": [2, 4]}},
        "layer4_forward_sample": {"query_val_pixels": _forward(), "target": _forward()},
        "batch_size_1_vs_n": {"max_abs": 3e-07, "max_rel": 1e-06, "min_cosine": 0.9999999,
                              "bitwise_equal": False, "shape": [2, 4], "within_tolerance": True},
        "scoring_ranking": {
            "score_max_abs_delta_vs_dot": 1.19e-07, "score_pairs_compared": 8,
            "score_max_abs_declared": 1e-06, "score_within_tolerance": True,
            "adapter_vs_argsort_unmasked": {"non_tie_count": 0},
            "rank_differences_vs_native_val_control": {"count": 0, "non_tie_count": 0,
                                                       "min_gap_abs": 0.0},
        },
        "population": {
            "features_vs_native_val_control": _forward(1e-06),
            "gallery_vs_native": _forward(1e-06),
        },
    }
    document = {"category": CATEGORY, "phase": "adapter", "model_id": "testmodel",
                "tolerances": dict(TOLERANCES), "adapter": body}
    if components is not None:
        document["component_parity"] = components
    return document

def _write(root: Path, native: dict, adapter: dict) -> Path:
    for name, body in ((f"g1_g2.testmodel.{CATEGORY}.native.json", native),
                       (f"g1_g2.testmodel.{CATEGORY}.adapter.json", adapter)):
        (root / name).write_text(json.dumps(body), encoding="utf-8")
    return root

def _evaluate(root: Path, **kwargs) -> gate.ParityVerdict:
    return gate.evaluate_model("testmodel", root=root, categories=(CATEGORY,), **kwargs)

def _reference_actual(t4, *, delta: float = 1e-8, n: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """A reference array with a component near zero, plus an `actual` inside the v2 rule."""
    reference = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    reference[3] = 0.0
    actual = reference + np.float32(delta)
    return actual, reference

def _block(t4, actual: np.ndarray, reference: np.ndarray, *, name: str = "sample_forward.query"):
    return t4.component_block({name: t4.component_scope(
        name, "forward_parity", actual, reference,
        atol=TOLERANCES["forward_max_abs"], rtol=TOLERANCES["forward_max_rel"])})

# ------------------------------------------------------ the six compatibility outcomes, by name

def test_a_a_v1_artifact_pair_is_judged_under_v1_exactly_as_today(tmp_path: Path) -> None:
    """(a) No `standard` key, no component block: v1, the same checks, the same verdict."""
    _write(tmp_path, _native(), _adapter())
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.PASS
    assert verdict.applied_standard == gate.STANDARD_RELATIVE_V1
    assert not any(check.name.startswith(f"{CATEGORY}.components.") for check in verdict.checks)
    assert all(check.standard == gate.STANDARD_RELATIVE_V1 for check in verdict.checks)
    # A v2 REQUEST is still advisory on this pair and still cannot relabel it.
    requested = _evaluate(tmp_path, standard=gate.STANDARD_ABS_REL_V2)
    assert requested.applied_standard == gate.STANDARD_RELATIVE_V1
    assert requested.advisory and requested.advisory[0]["applies"] is False

def test_b_a_v2_pair_is_judged_under_v2_and_records_the_applied_standard(tmp_path: Path, t4) -> None:
    """(b) Both sides declare v2, the block is present, the verdict records `applied_standard`."""
    actual, reference = _reference_actual(t4)
    native, adapter = _native(), _adapter(_block(t4, actual, reference))
    native["standard"] = adapter["standard"] = gate.STANDARD_ABS_REL_V2
    _write(tmp_path, native, adapter)
    verdict = _evaluate(tmp_path, standard=gate.STANDARD_ABS_REL_V2)
    assert verdict.applied_standard == gate.STANDARD_ABS_REL_V2
    assert verdict.status == gate.PASS, verdict.detail
    derived = [check for check in verdict.checks if check.name.startswith(f"{CATEGORY}.components.")]
    assert derived, [check.name for check in verdict.checks]
    assert all(check.standard == gate.STANDARD_ABS_REL_V2 for check in derived)
    assert derived[0].measured["mismatch_count"] == 0
    assert derived[0].measured["components_compared"] == derived[0].measured["components_total"] == 8

def test_c_a_mixed_v1_v2_pair_is_malformed(tmp_path: Path, t4) -> None:
    """(c) One side v1, the other v2: one verdict cannot be reported under two standards."""
    actual, reference = _reference_actual(t4)
    native, adapter = _native(), _adapter(_block(t4, actual, reference))
    adapter["standard"] = gate.STANDARD_ABS_REL_V2
    _write(tmp_path, native, adapter)
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MALFORMED, verdict.detail
    assert "different standards" in verdict.detail

def test_d_a_synthetic_component_mismatch_fails_and_names_the_coordinate(
        tmp_path: Path, t4) -> None:
    """(d) One component outside `atol + rtol*|reference|`: the check FAILS at that index."""
    actual, reference = _reference_actual(t4)
    actual = actual.copy()
    actual[5] = np.float32(reference[5] + 1e-2)   # far outside 1e-5 + 1e-4*|ref|
    native, adapter = _native(), _adapter(_block(t4, actual, reference))
    native["standard"] = adapter["standard"] = gate.STANDARD_ABS_REL_V2
    _write(tmp_path, native, adapter)
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MANDATORY_FAILED, verdict.detail
    failed = [check for check in verdict.failures if check.name.startswith(f"{CATEGORY}.components.")]
    assert len(failed) == 1
    assert failed[0].measured["mismatch_count"] == 1
    assert 5 in failed[0].measured["offending_coordinates"]
    assert failed[0].measured["offending_values"][0]["index"] == 5
    assert failed[0].measured["worst_normalized"] > 1.0

def test_e_a_non_finite_component_is_malformed_not_a_pass(tmp_path: Path, t4) -> None:
    """(e) NaN/Inf in the block: malformed evidence, never `ok`."""
    for poison in (math.nan, math.inf, -math.inf):
        actual, reference = _reference_actual(t4)
        block = _block(t4, actual, reference)
        block["scopes"]["sample_forward.query"]["actual"][2] = poison
        native, adapter = _native(), _adapter(block)
        native["standard"] = adapter["standard"] = gate.STANDARD_ABS_REL_V2
        _write(tmp_path, native, adapter)
        verdict = _evaluate(tmp_path)
        assert verdict.status == gate.MALFORMED, (poison, verdict.status, verdict.detail)
        assert verdict.passed is False
        assert "non-finite" in verdict.detail

def test_f_a_v2_label_with_only_aggregates_stays_malformed(tmp_path: Path) -> None:
    """(f) `standard: v2` with no component block: the label alone is never a v2 result."""
    native, adapter = _native(), _adapter()
    native["standard"] = adapter["standard"] = gate.STANDARD_ABS_REL_V2
    _write(tmp_path, native, adapter)
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MALFORMED, verdict.detail
    assert gate.COMPONENT_KEY in verdict.detail
    # And the label on ONE side only is refused for the same reason.
    native["standard"] = gate.STANDARD_ABS_REL_V2
    adapter["standard"] = None
    native.pop("standard")
    adapter["standard"] = gate.STANDARD_ABS_REL_V2
    _write(tmp_path, native, adapter)
    assert _evaluate(tmp_path).status == gate.MALFORMED

# ------------------------------------------------------------------ the writer

def test_the_writer_records_aligned_components_with_full_digests(t4) -> None:
    actual, reference = _reference_actual(t4)
    scope = t4.component_scope("sample_forward.query", "forward_parity", actual, reference,
                               atol=TOLERANCES["forward_max_abs"],
                               rtol=TOLERANCES["forward_max_rel"])
    assert scope["coverage"] == "full"
    assert scope["components_total"] == scope["components_emitted"] == 8
    assert scope["indices"] == list(range(8))
    assert scope["mismatch_count"] == 0
    assert scope["shape"] == [8]
    assert set(scope["sha256"]) == {"reference", "actual", "delta"}
    assert all(len(digest) == 64 for digest in scope["sha256"].values())
    # The digests bind the FULL arrays: recomputing them from the reference proves the record.
    import hashlib
    expected = hashlib.sha256(
        np.ascontiguousarray(reference, dtype="<f8").tobytes()).hexdigest()
    assert scope["sha256"]["reference"] == expected
    # `max_abs` is the recorded aggregate over the full array, not a claim about the rule.
    assert scope["max_abs"] == float(np.abs(
        actual.astype(np.float64) - reference.astype(np.float64)).max())

def test_the_writer_chunks_a_large_array_and_keeps_every_offender(t4) -> None:
    """Above the limit: the head plus every violating coordinate, with `coverage: chunked`."""
    total = t4.COMPONENT_FULL_LIMIT + 100
    reference = np.full(total, 0.5, dtype=np.float32)
    actual = reference + np.float32(1e-9)
    offender = total - 1                       # outside the emitted head on purpose
    actual[offender] = np.float32(0.5 + 1e-2)
    scope = t4.component_scope("population.features_vs_native_val_control", "forward_parity",
                              actual, reference, atol=TOLERANCES["forward_max_abs"],
                              rtol=TOLERANCES["forward_max_rel"])
    assert scope["coverage"] == "chunked"
    assert scope["components_total"] == total
    assert scope["components_emitted"] == t4.COMPONENT_HEAD + 1
    assert offender in scope["indices"], "a violating component must never be dropped"
    assert scope["mismatch_count"] == 1
    assert [item["index"] for item in scope["offenders"]] == [offender]
    position = scope["indices"].index(offender)
    assert scope["reference"][position] == pytest.approx(0.5)
    assert scope["actual"][position] == pytest.approx(0.5 + 1e-2)

def test_the_writer_refuses_misaligned_arrays(t4) -> None:
    with pytest.raises(SystemExit, match="not aligned"):
        t4.component_scope("x", "forward_parity", np.zeros(4), np.zeros(5), atol=1e-5, rtol=1e-4)

def test_the_two_producers_emit_the_same_declared_contract(t4, encoder) -> None:
    """Two interpreters, two files, one contract: the block and its rule must be byte-comparable."""
    assert t4.STANDARD_V2 == encoder.STANDARD_V2 == gate.STANDARD_ABS_REL_V2
    assert t4.COMPONENT_FULL_LIMIT == encoder.COMPONENT_FULL_LIMIT
    assert t4.COMPONENT_HEAD == encoder.COMPONENT_HEAD
    actual, reference = _reference_actual(t4)
    left = t4.component_scope("sample_forward.query", "forward_parity", actual, reference,
                              atol=1e-5, rtol=1e-4)
    right = encoder.component_scope("sample_forward.query", actual, reference, atol=1e-5, rtol=1e-4)
    assert left == right, "the v2 record must not depend on which producer wrote it"
    assert t4.component_block({"s": left})["declared_rule"] == gate.COMPONENT_RULE

# ------------------------------------------------------------------ the reader's refusals

@pytest.mark.parametrize("mutate, needle", [
    (lambda block: block.pop("scopes"), "scopes"),
    (lambda block: block["scopes"]["sample_forward.query"].update(components_total=0),
     "components_total"),
    (lambda block: block["scopes"]["sample_forward.query"].update(shape=[3, 3]), "shape"),
    (lambda block: block["scopes"]["sample_forward.query"].update(indices=[0, 1]),
     "index"),
    (lambda block: block["scopes"]["sample_forward.query"].update(coverage="partial"), "coverage"),
    (lambda block: block["scopes"]["sample_forward.query"].update(mismatch_count=3),
     "mismatch_count"),
    (lambda block: block["scopes"]["sample_forward.query"]["sha256"].update(reference="abc"),
     "sha256"),
    (lambda block: block["scopes"]["sample_forward.query"].update(dimension="scoring_parity"),
     "dimension"),
    (lambda block: block["scopes"]["sample_forward.query"].update(reference_side="adapter"),
     "reference_side"),
])
def test_a_block_that_cannot_be_evaluated_is_malformed(tmp_path: Path, t4,
                                                       mutate, needle: str) -> None:
    """Every shape that cannot yield a componentwise answer refuses instead of passing."""
    actual, reference = _reference_actual(t4)
    block = _block(t4, actual, reference)
    block["reference_side"] = "native"
    mutate(block)
    if needle == "reference_side":
        block["reference_side"] = "adapter"
    native, adapter = _native(), _adapter(block)
    native["standard"] = adapter["standard"] = gate.STANDARD_ABS_REL_V2
    _write(tmp_path, native, adapter)
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MALFORMED, (needle, verdict.status, verdict.detail)
    assert needle in verdict.detail

def test_the_aggregate_cannot_infer_the_componentwise_result(tmp_path: Path, t4) -> None:
    """A block whose `mismatch_count` claims zero beside violating values is malformed, not ok."""
    actual, reference = _reference_actual(t4)
    actual = actual.copy()
    actual[6] = np.float32(reference[6] + 1e-2)
    block = _block(t4, actual, reference)
    scope = block["scopes"]["sample_forward.query"]
    # The writer recorded the mismatch; now a hand-edit erases it, keeping the values.
    scope["mismatch_count"] = 0
    scope["offenders"] = []
    native, adapter = _native(), _adapter(block)
    native["standard"] = adapter["standard"] = gate.STANDARD_ABS_REL_V2
    _write(tmp_path, native, adapter)
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MALFORMED, verdict.detail
    assert "contradicts the values it carries" in verdict.detail

def test_a_v2_pair_whose_aggregate_fails_still_fails(tmp_path: Path, t4) -> None:
    """The v1 aggregate checks are NOT deleted for a v2 pair (the known ceiling in V2_STANDARD.md)."""
    actual, reference = _reference_actual(t4)
    native, adapter = _native(), _adapter(_block(t4, actual, reference))
    native["standard"] = adapter["standard"] = gate.STANDARD_ABS_REL_V2
    adapter["adapter"]["layer4_forward_sample"]["query_val_pixels"] = _forward(0.1)
    _write(tmp_path, native, adapter)
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MANDATORY_FAILED, verdict.detail
    assert any("max_abs" in check.name for check in verdict.failures)

def test_a_v2_block_on_the_native_side_alone_is_accepted(tmp_path: Path, t4) -> None:
    """The block travels with whichever phase held both arrays; either side may carry it."""
    actual, reference = _reference_actual(t4)
    native, adapter = _native(), _adapter()
    native["component_parity"] = _block(t4, actual, reference)
    native["standard"] = adapter["standard"] = gate.STANDARD_ABS_REL_V2
    _write(tmp_path, native, adapter)
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.PASS, verdict.detail
    assert verdict.applied_standard == gate.STANDARD_ABS_REL_V2

def test_two_sides_recording_different_components_is_malformed(tmp_path: Path, t4) -> None:
    actual, reference = _reference_actual(t4)
    left = _block(t4, actual, reference)
    shifted = actual.copy()
    # float32 spacing near 1.0 is ~1.2e-7, so the shift must exceed it or both records round to the
    # SAME bytes and the test proves nothing.
    shifted[0] = np.float32(reference[0] + 1e-5)
    right = _block(t4, shifted, reference)
    assert left != right, "the fixture must produce two genuinely different records"
    native, adapter = _native(), _adapter(left)
    native["component_parity"] = right
    native["standard"] = adapter["standard"] = gate.STANDARD_ABS_REL_V2
    _write(tmp_path, native, adapter)
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MALFORMED, verdict.detail
    assert "different components" in verdict.detail

def test_evaluate_abs_rel_has_a_production_caller() -> None:
    """The rule is not merely defined: the v2 branch of the gate reaches it per component."""
    import inspect
    source = inspect.getsource(gate._component_scope_check)
    assert "evaluate_abs_rel(" in source

def test_a_chunked_block_is_judged_and_never_read_as_full(tmp_path: Path, t4) -> None:
    """`coverage: chunked` is accepted, judged on what it carries, and reported as a chunk."""
    total = t4.COMPONENT_FULL_LIMIT + 50
    reference = np.full(total, 0.5, dtype=np.float32)
    actual = reference + np.float32(1e-9)
    actual[total - 1] = np.float32(0.5 + 1e-2)      # the only violating component, outside the head
    block = _block(t4, actual, reference, name="population.features_vs_native_val_control")
    native, adapter = _native(), _adapter(block)
    native["standard"] = adapter["standard"] = gate.STANDARD_ABS_REL_V2
    _write(tmp_path, native, adapter)
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MANDATORY_FAILED, verdict.detail
    check = next(check for check in verdict.checks
                 if check.name == f"{CATEGORY}.components.population.features_vs_native_val_control")
    assert check.measured["coverage"] == "chunked"
    assert check.measured["components_compared"] == t4.COMPONENT_HEAD + 1
    assert check.measured["components_total"] == total
    assert check.measured["offending_coordinates"] == [total - 1]

# ------------------------------------------------- aggregate label: one string, plus a breakdown

def _model_pair(root: Path, model_id: str, category: str, *, standard: str | None,
                block: dict | None = None) -> None:
    """Write one model's artifact pair (native + adapter) under `root`, from the shared fixtures."""
    native, adapter = _native(), _adapter(block)
    for document in (native, adapter):
        document["category"] = category
        document["model_id"] = model_id
        if standard is not None:
            document["standard"] = standard
    for side, document in (("native", native), ("adapter", adapter)):
        (root / f"g1_g2.{model_id}.{category}.{side}.json").write_text(
            json.dumps(document), encoding="utf-8")

def test_a_two_standard_evidence_set_keeps_standard_a_string(tmp_path: Path, t4) -> None:
    """Two standards on disk: `standard` stays the ONE label, `standards` holds the grouping.

    The shipped bug: `standard` held `applied[0] if len(applied) == 1 else applied`, so a second
    standard turned the slot into a LIST. The frontend types that field as `string` and renders
    `Chuẩn ${gate.standard}`, so a mixed directory printed an array into a sentence. The label is now
    the standard of the FIRST model in sorted order that has a verdict -- stable as artifacts arrive
    -- and the per-standard grouping lives in its own field.
    """
    actual, reference = _reference_actual(t4)
    block = _block(t4, actual, reference)
    for category in ("dress", "shirt", "toptee"):
        _model_pair(tmp_path, "encoder", category, standard=None)
        _model_pair(tmp_path, "aaa_v2", category, standard=gate.STANDARD_ABS_REL_V2, block=block)

    result = gate.evaluate_all(root=tmp_path)
    assert isinstance(result["standard"], str), result["standard"]
    assert result["standard"] in gate.STANDARDS
    # Sorted first model id ("aaa_v2") wins the single label, and it really is judged under v2.
    assert result["standard"] == gate.STANDARD_ABS_REL_V2
    assert result["models"]["aaa_v2"]["applied_standard"] == gate.STANDARD_ABS_REL_V2
    assert result["models"]["encoder"]["applied_standard"] == gate.STANDARD_RELATIVE_V1
    assert result["standards"] == {gate.STANDARD_ABS_REL_V2: ["aaa_v2"],
                                   gate.STANDARD_RELATIVE_V1: ["encoder"]}
    # Nothing is left out of the breakdown: every judged model belongs to exactly one standard.
    grouped = sorted(model for models in result["standards"].values() for model in models)
    judged = sorted(model for model, verdict in result["models"].items()
                    if verdict["status"] != gate.NOT_RUN)
    assert grouped == judged
    # A request for the other standard cannot move either answer.
    requested = gate.evaluate_all(root=tmp_path, standard=gate.STANDARD_RELATIVE_V1)
    assert requested["standard"] == gate.STANDARD_ABS_REL_V2
    assert requested["requested_standard"] == gate.STANDARD_RELATIVE_V1

def test_the_aggregate_label_is_v1_when_nothing_was_judged(tmp_path: Path) -> None:
    """No artifact at all: the label is v1 and the breakdown is empty, never a list or None."""
    result = gate.evaluate_all(root=tmp_path)
    assert result["standard"] == gate.STANDARD_RELATIVE_V1
    assert result["standards"] == {}
    assert result["passed_models"] == []
