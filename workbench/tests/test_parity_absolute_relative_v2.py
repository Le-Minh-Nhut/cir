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

import base64
import importlib.util
import json
import time
import os
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

#: The identifiers a hit-count comparison must carry before it can certify anything. `metrics_parity`
#: is derived from per-side counts plus these, so a fixture that passes needs both -- a bare
#: `native_metrics_match_feature_rerank` boolean is only a cross-check now.
PROVENANCE = {"protocol": "fashioniq_val_split", "checkpoint_id": "testmodel_checkpoint",
              "query_order_sha256": "a" * 64, "gallery_order_sha256": "b" * 64}


def _metrics_detail(native_hits: dict | None = None, adapter_hits: dict | None = None) -> dict:
    native_hits = native_hits or {"r1": 4, "r10": 7, "r50": 9}
    adapter_hits = adapter_hits or dict(native_hits)
    return {"cutoffs": [1, 10, 50],
            "native": {"hits": dict(native_hits), "query_count": 10},
            "adapter": {"hits": dict(adapter_hits), "query_count": 10},
            "eligible_query_set_identical": True,
            "equality": {k: native_hits[k] == adapter_hits[k] for k in native_hits}}


def _native() -> dict:
    document = {
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
        "native_metrics_parity_detail": _metrics_detail(),
    }
    document.update(PROVENANCE)
    return document

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
    document.update(PROVENANCE)
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

def test_the_writer_emits_every_component_of_a_large_array_as_ordered_chunks(t4) -> None:
    """Above the limit: EVERY component, in chunks that tile the array. No preview, no head budget.

    The record used to emit the first `COMPONENT_HEAD` components plus the offenders, which left the
    gate unable to say anything about the rest. A tail offender is now emitted because the tail is.
    """
    total = t4.COMPONENT_FULL_LIMIT + 100
    reference = np.full(total, 0.5, dtype=np.float32)
    actual = reference + np.float32(1e-9)
    offender = total - 1                       # the tail: never inside a 512-component head
    actual[offender] = np.float32(0.5 + 1e-2)
    scope = t4.component_scope("population.features_vs_native_val_control", "forward_parity",
                              actual, reference, atol=TOLERANCES["forward_max_abs"],
                              rtol=TOLERANCES["forward_max_rel"])
    assert scope["coverage"] == "chunks"
    assert scope["components_total"] == scope["components_emitted"] == total
    assert len(scope["chunks"]) == math.ceil(total / t4.COMPONENT_CHUNK_SIZE)
    covered: list[int] = []
    for chunk in scope["chunks"]:
        reference_values = gate._decode_components(chunk["reference"], "reference", where="chunk")
        covered.extend(range(chunk["start"], chunk["start"] + len(reference_values)))
        actual_values = gate._decode_components(chunk["actual"], "actual", where="chunk")
        assert len(reference_values) == len(actual_values)
        # Each chunk digests ITS OWN bytes, so the reader can recompute it.
        assert chunk["sha256"]["reference"] == gate.component_digest(reference_values)
        assert chunk["sha256"]["actual"] == gate.component_digest(actual_values)
    assert covered == list(range(total)), "the chunks must tile the array exactly"
    assert scope["mismatch_count"] == 1
    assert [item["index"] for item in scope["offenders"]] == [offender]
    assert scope["worst_normalized"] > 1.0

def test_the_writer_declares_the_threshold_and_chunk_size_it_used(t4) -> None:
    """The constant that decides full-vs-chunks is named in the record, not inferred."""
    block = t4.component_block({})
    assert str(t4.COMPONENT_FULL_LIMIT) in block["chunk_rule"]
    assert str(t4.COMPONENT_CHUNK_SIZE) in block["chunk_rule"]
    assert "no gap and no overlap" in block["chunk_rule"]

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

def test_a_chunks_block_covering_a_large_scope_is_judged_over_every_component(
        tmp_path: Path, t4) -> None:
    """The `chunks` form: complete coverage of a scope larger than the `full` limit.

    This is the POSITIVE CONTROL for the coverage fix. A preview of the same array could never earn a
    verdict; the chunks form can, because it carries every component and the reader recomputes every
    digest and applies the rule to all of them.
    """
    total = t4.COMPONENT_FULL_LIMIT + 50
    reference = np.full(total, 0.5, dtype=np.float32)
    actual = reference + np.float32(1e-9)          # every component inside atol + rtol*|b|
    block = _block(t4, actual, reference, name="population.features_vs_native_val_control")
    native, adapter = _native(), _adapter(block)
    native["standard"] = adapter["standard"] = gate.STANDARD_ABS_REL_V2
    _write(tmp_path, native, adapter)
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.PASS, verdict.detail
    check = next(check for check in verdict.checks
                 if check.name == f"{CATEGORY}.components.population.features_vs_native_val_control")
    assert check.measured["coverage"] == "chunks"
    assert check.measured["complete_coverage"] is True
    assert check.measured["components_compared"] == check.measured["components_total"] == total
    assert check.measured["mismatch_count"] == 0
    assert check.ok is True

def test_a_chunks_block_with_one_tail_offender_fails_and_names_it(tmp_path: Path, t4) -> None:
    """A violating component in the LAST chunk is found, because the last chunk is emitted."""
    total = t4.COMPONENT_FULL_LIMIT + 50
    reference = np.full(total, 0.5, dtype=np.float32)
    actual = reference + np.float32(1e-9)
    actual[total - 1] = np.float32(0.5 + 1e-2)      # the tail chunk, outside any head sample
    block = _block(t4, actual, reference, name="population.features_vs_native_val_control")
    native, adapter = _native(), _adapter(block)
    native["standard"] = adapter["standard"] = gate.STANDARD_ABS_REL_V2
    _write(tmp_path, native, adapter)
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MANDATORY_FAILED, verdict.detail
    check = next(check for check in verdict.checks
                 if check.name == f"{CATEGORY}.components.population.features_vs_native_val_control")
    assert check.measured["coverage"] == "chunks"
    assert check.measured["offending_coordinates"] == [total - 1]
    assert check.measured["worst_normalized"] > 1.0

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


# ------------------------------------------------ the coverage rule and its regression battery

#: One large scope's arrays, reused by the coverage tests. 4146 = the scope the audit measured: bigger
#: than `COMPONENT_FULL_LIMIT`, with room for an offender well past any head sample.
LARGE_TOTAL = 4146
LARGE_OFFENDER = 3600



def _encode(values, t4) -> str:
    """The writer's own transport, so a test never invents a second encoding."""
    return t4._encode_components(np.ascontiguousarray(np.array(values), dtype="<f8"))


def _decoded(chunk: dict, key: str) -> list[float]:
    return gate._decode_components(chunk[key], key, where="test")


def _as_preview(scope: dict, *, keep: list[int], t4=None) -> dict:
    """Rewrite a `chunks` scope as the preview the OLD writer emitted: `keep` components, chunked."""
    values = {}
    for chunk in scope["chunks"]:
        reference = _decoded(chunk, "reference")
        actual = _decoded(chunk, "actual")
        for offset in range(len(reference)):
            values[chunk["start"] + offset] = (reference[offset], actual[offset])
    scope.pop("chunks", None)
    scope.update({"coverage": "chunked", "components_emitted": len(keep), "indices": list(keep),
                  "reference": [values[i][0] for i in keep],
                  "actual": [values[i][1] for i in keep]})
    return scope


def _large_pair(t4, *, offender: int | None = LARGE_OFFENDER,
                delta: float = 1e-2) -> tuple[dict, dict]:
    """A native/adapter pair whose single component scope holds `LARGE_TOTAL` components."""
    reference = np.full(LARGE_TOTAL, 0.5, dtype=np.float32)
    actual = reference + np.float32(1e-9)
    if offender is not None:
        actual[offender] = np.float32(0.5 + delta)
    block = t4.component_block({"sample_forward.query": t4.component_scope(
        "sample_forward.query", "forward_parity", actual, reference,
        atol=TOLERANCES["forward_max_abs"], rtol=TOLERANCES["forward_max_rel"])})
    native, adapter = _native(), _adapter(block)
    native["standard"] = adapter["standard"] = gate.STANDARD_ABS_REL_V2
    return native, adapter


def _scope_of(verdict) -> object:
    return next(check for check in verdict.checks if check.name.startswith(f"{CATEGORY}.components."))


def test_the_tail_offender_in_a_complete_record_fails_the_scope(tmp_path: Path, t4) -> None:
    """An offender past the old head sample is judged, because the whole array is emitted."""
    native, adapter = _large_pair(t4)
    _write(tmp_path, native, adapter)
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MANDATORY_FAILED, verdict.detail
    scope = _scope_of(verdict)
    assert scope.measured["offending_coordinates"] == [LARGE_OFFENDER]
    assert scope.measured["worst_normalized"] > 1.0


def test_a_preview_record_can_never_pass_even_when_every_emitted_row_is_clean(
        tmp_path: Path, t4) -> None:
    """THE DEFECT. A 512-component preview of 4146, every row clean, digest intact: INCOMPLETE.

    The defect was that `ok` was `not offenders` over the rows the artifact chose to emit, so a
    smaller preview was a smaller claim and an easier pass. Partial coverage is now reported as
    unmeasured, and the dimension is `missing`, never True.
    """
    native, adapter = _large_pair(t4, offender=None)
    scope = adapter["component_parity"]["scopes"]["sample_forward.query"]
    # Turn the complete record into exactly the shape the old writer emitted: the head, previewed.
    head = t4.COMPONENT_HEAD
    _as_preview(scope, keep=list(range(head)))
    _write(tmp_path, native, adapter)
    verdict = _evaluate(tmp_path)
    assert verdict.status != gate.PASS, verdict.detail
    assert verdict.passed is False
    assert verdict.status == gate.INCOMPLETE
    assert verdict.missing == ["forward_parity"], verdict.missing
    check = _scope_of(verdict)
    assert check.ok is False
    assert check.measured["complete_coverage"] is False
    assert check.measured["components_compared"] == head
    assert check.measured["components_total"] == LARGE_TOTAL
    assert "PREVIEW ONLY" in check.note
    assert f"{head} of {LARGE_TOTAL}" in check.note


def test_a_preview_whose_offender_rows_were_deleted_is_malformed_not_a_pass(
        tmp_path: Path, t4) -> None:
    """The adversarial variant: offender rows removed, counts silenced, `worst_normalized` kept."""
    native, adapter = _large_pair(t4)
    scope = adapter["component_parity"]["scopes"]["sample_forward.query"]
    # Reduce to a preview the way the old writer would have, then delete the offender's row and
    # silence the two fields that would otherwise name it. `worst_normalized` and `max_abs` stay:
    # they describe the FULL array, as they always did.
    _as_preview(scope, keep=list(range(t4.COMPONENT_HEAD)))
    scope.update({"mismatch_count": 0, "offenders": []})
    assert scope["worst_normalized"] > 1.0
    _write(tmp_path, native, adapter)
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MALFORMED, verdict.detail
    assert "worst_normalized" in verdict.detail
    assert "no offending component is recorded" in verdict.detail


def test_a_preview_naming_an_offender_it_does_not_emit_is_malformed(tmp_path: Path, t4) -> None:
    """A preview's offender list is checked against the components it actually carries."""
    native, adapter = _large_pair(t4, offender=None)
    scope = adapter["component_parity"]["scopes"]["sample_forward.query"]
    _as_preview(scope, keep=list(range(t4.COMPONENT_HEAD)))
    scope.update({"mismatch_count": 1,
                  "offenders": [{"index": LARGE_OFFENDER, "reference": 0.5, "actual": 0.51,
                                 "normalized": 166.6}]})
    _write(tmp_path, native, adapter)
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MALFORMED, verdict.detail
    # The count and the list are checked against the values carried, so a claimed offender the record
    # does not emit is caught as a contradiction -- there is no way to assert one you cannot show.
    assert "contradicts the values it carries" in verdict.detail


def _chunks_pair(t4) -> tuple[dict, dict, list[dict]]:
    native, adapter = _large_pair(t4, offender=None)
    chunks = adapter["component_parity"]["scopes"]["sample_forward.query"]["chunks"]
    return native, adapter, chunks


def test_a_chunk_gap_is_malformed(tmp_path: Path, t4) -> None:
    """A dropped chunk leaves a hole, and the union no longer equals the scope."""
    native, adapter, chunks = _chunks_pair(t4)
    del chunks[1]
    _write(tmp_path, native, adapter)
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MALFORMED, verdict.detail
    assert "not complete coverage" in verdict.detail or "gap and no overlap" in verdict.detail


def test_a_short_final_chunk_is_malformed(tmp_path: Path, t4) -> None:
    """The tail truncated by one component: the array is not fully covered."""
    native, adapter, chunks = _chunks_pair(t4)
    last = chunks[-1]
    for key in ("reference", "actual"):
        last[key] = _encode(_decoded(last, key)[:-1], t4)
    # Its own digest still describes the longer values, so the truncation is caught either way.
    _write(tmp_path, native, adapter)
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MALFORMED, verdict.detail
    assert "does not bind these values" in verdict.detail or "tail was never emitted" in verdict.detail


def test_a_substituted_chunk_is_malformed_by_its_own_digest(tmp_path: Path, t4) -> None:
    """One chunk's values replaced: ITS digest no longer binds them, whatever the full-array one says.

    This is the check the old reader could not make at all -- it validated 64 hex characters and
    nothing else, so a substituted preview was indistinguishable from a faithful one.
    """
    native, adapter, chunks = _chunks_pair(t4)
    adjusted = _decoded(chunks[0], "actual")
    adjusted[7] = 0.5009765625                     # a change below the rule's tolerance
    chunks[0]["actual"] = _encode(adjusted, t4)
    _write(tmp_path, native, adapter)
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MALFORMED, verdict.detail
    assert "does not bind these values" in verdict.detail


def test_a_stale_chunk_digest_is_malformed(tmp_path: Path, t4) -> None:
    """The values are honest but the digest was copied from another chunk."""
    native, adapter, chunks = _chunks_pair(t4)
    chunks[0]["sha256"] = dict(chunks[1]["sha256"])
    _write(tmp_path, native, adapter)
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MALFORMED, verdict.detail
    assert "does not bind these values" in verdict.detail


def test_chunks_in_the_wrong_order_are_malformed(tmp_path: Path, t4) -> None:
    native, adapter, chunks = _chunks_pair(t4)
    chunks[0], chunks[1] = chunks[1], chunks[0]
    _write(tmp_path, native, adapter)
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MALFORMED, verdict.detail
    assert "must tile" in verdict.detail


def test_overlapping_chunks_are_malformed(tmp_path: Path, t4) -> None:
    """A chunk re-emitting components the previous one already covered."""
    native, adapter, chunks = _chunks_pair(t4)
    # A chunk that starts one component EARLY, re-covering its predecessor's last index.
    chunks[1]["start"] = chunks[1]["start"] - 1
    _write(tmp_path, native, adapter)
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MALFORMED, verdict.detail
    assert "must tile" in verdict.detail


def test_a_chunk_starting_past_the_tail_is_malformed(tmp_path: Path, t4) -> None:
    """A last chunk that runs past `components_total`."""
    native, adapter, chunks = _chunks_pair(t4)
    chunks[-1]["start"] = LARGE_TOTAL - 10
    _write(tmp_path, native, adapter)
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MALFORMED, verdict.detail
    assert "past the" in verdict.detail or "runs to flat index" in verdict.detail


def test_a_chunk_with_a_non_finite_component_is_malformed(tmp_path: Path, t4) -> None:
    for poison in (math.nan, math.inf, -math.inf):
        native, adapter, chunks = _chunks_pair(t4)
        poisoned = _decoded(chunks[-1], "actual")
        poisoned[3] = poison
        chunks[-1]["actual"] = _encode(poisoned, t4)
        _write(tmp_path, native, adapter)
        verdict = _evaluate(tmp_path)
        assert verdict.status == gate.MALFORMED, (poison, verdict.status)
        assert "non-finite" in verdict.detail


def test_a_duplicated_component_across_chunks_is_malformed(tmp_path: Path, t4) -> None:
    """A chunk whose span re-covers a component its predecessor already emitted."""
    native, adapter, chunks = _chunks_pair(t4)
    chunks[1]["start"] = chunks[1]["start"] - 5
    _write(tmp_path, native, adapter)
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MALFORMED, verdict.detail
    assert "gap and no overlap" in verdict.detail


def test_a_declared_indices_list_must_match_its_start(tmp_path: Path, t4) -> None:
    """`indices` is optional, but when present it must be the run `start` implies."""
    native, adapter, chunks = _chunks_pair(t4)
    chunks[0]["indices"] = list(range(1, 1 + len(_decoded(chunks[0], "reference"))))
    _write(tmp_path, native, adapter)
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MALFORMED, verdict.detail
    assert "contiguous run" in verdict.detail


def test_a_chunk_count_that_disagrees_with_the_scope_is_malformed(tmp_path: Path, t4) -> None:
    """`mismatch_count` is derived over ALL chunks, so it cannot be owned by one of them."""
    native, adapter, chunks = _chunks_pair(t4)
    scope = adapter["component_parity"]["scopes"]["sample_forward.query"]
    scope["mismatch_count"] = 3
    _write(tmp_path, native, adapter)
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MALFORMED, verdict.detail
    assert "contradicts the values it carries" in verdict.detail


def test_a_chunks_label_on_a_partial_scope_is_malformed(tmp_path: Path, t4) -> None:
    """Claiming complete coverage while carrying one chunk fewer."""
    native, adapter, chunks = _chunks_pair(t4)
    chunks.pop()
    _write(tmp_path, native, adapter)
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MALFORMED, verdict.detail


def test_an_unknown_coverage_value_is_malformed(tmp_path: Path, t4) -> None:
    native, adapter = _large_pair(t4, offender=None)
    adapter["component_parity"]["scopes"]["sample_forward.query"]["coverage"] = "sampled"
    _write(tmp_path, native, adapter)
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MALFORMED, verdict.detail
    assert "coverage" in verdict.detail


def test_a_shape_that_disagrees_with_the_total_is_malformed(tmp_path: Path, t4) -> None:
    native, adapter = _large_pair(t4, offender=None)
    adapter["component_parity"]["scopes"]["sample_forward.query"]["shape"] = [LARGE_TOTAL + 1]
    _write(tmp_path, native, adapter)
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MALFORMED, verdict.detail
    assert "shape" in verdict.detail


def test_a_complete_record_whose_summary_understates_the_values_is_malformed(
        tmp_path: Path, t4) -> None:
    """`max_abs` smaller than a component actually carried: the summary is not a summary."""
    native, adapter = _large_pair(t4)
    scope = adapter["component_parity"]["scopes"]["sample_forward.query"]
    scope["max_abs"] = 1e-12
    _write(tmp_path, native, adapter)
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MALFORMED, verdict.detail
    assert "max_abs" in verdict.detail


def test_a_complete_record_whose_worst_understates_the_values_is_malformed(
        tmp_path: Path, t4) -> None:
    native, adapter = _large_pair(t4)
    scope = adapter["component_parity"]["scopes"]["sample_forward.query"]
    scope["worst_normalized"] = 0.5
    _write(tmp_path, native, adapter)
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MALFORMED, verdict.detail
    assert "worst_normalized" in verdict.detail


def test_the_complete_form_recomputes_digests_with_one_shared_serialization(t4) -> None:
    """The reader's digest and the writer's are the same eight bytes per component, or nothing holds."""
    import hashlib
    values = [0.5, -0.0, 1.0 / 3.0, float(np.float32(1e-9))]
    expected = hashlib.sha256(
        np.ascontiguousarray(np.array(values), dtype="<f8").tobytes()).hexdigest()
    assert gate.component_digest(values) == expected
    assert gate.COMPONENT_DIGEST_DTYPE == "<f8"


# ------------------------------------------------------ hit counts: derived, never a flag

def _metrics_pair(*, native_hits=None, adapter_hits=None, flag: bool | None = True,
                  native_count: int = 2017, adapter_count: int = 2017,
                  identical_set: bool = True, provenance: dict | None = None,
                  detail: dict | None = None) -> tuple[dict, dict]:
    """A pair whose metric evidence is one hit-count block plus a recorded flag."""
    native_hits = native_hits or {"r1": 375, "r10": 1032, "r50": 1498}
    adapter_hits = adapter_hits or dict(native_hits)
    native, adapter = _native(), _adapter()
    block = detail if detail is not None else {
        "cutoffs": [1, 10, 50],
        "native": {"hits": dict(native_hits), "query_count": native_count},
        "adapter": {"hits": dict(adapter_hits), "query_count": adapter_count},
        "eligible_query_set_identical": identical_set,
        "equality": {k: native_hits.get(k) == adapter_hits.get(k) for k in native_hits}}
    native["native_metrics_parity_detail"] = block
    if flag is None:
        native.pop("native_metrics_match_feature_rerank", None)
        native["native"].pop("native_metrics_match_feature_rerank", None)
    else:
        native["native"]["native_metrics_match_feature_rerank"] = flag
    if provenance is not None:
        native.update(provenance)
        adapter.update(provenance)
    return native, adapter


def _metrics_of(verdict) -> list:
    return [check for check in verdict.checks if check.dimension == "metrics_parity"]


def test_a_true_flag_beside_differing_hit_counts_is_malformed(tmp_path: Path) -> None:
    """THE DEFECT: `native r1 = 375`, `adapter r1 = 376`, flag still `true`, used to PASS.

    The gate read the boolean AS the verdict and never opened the block beside it, so the mismatch
    was recorded and ignored. The flag is now a cross-check against the counts, and a disagreement
    makes the artifact self-contradictory rather than certified.
    """
    native, adapter = _metrics_pair(native_hits={"r1": 375, "r10": 1032, "r50": 1498},
                                    adapter_hits={"r1": 376, "r10": 1032, "r50": 1498},
                                    flag=True)
    _write(tmp_path, native, adapter)
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MALFORMED, verdict.detail
    assert verdict.passed is False
    assert "native_metrics_match_feature_rerank is True" in verdict.detail
    assert "'r1': 375" in verdict.detail and "'r1': 376" in verdict.detail


def test_agreeing_counts_with_a_false_flag_are_malformed_too(tmp_path: Path) -> None:
    """Contradiction runs both ways: a `false` flag beside equal counts is equally wrong."""
    native, adapter = _metrics_pair(flag=False)
    _write(tmp_path, native, adapter)
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MALFORMED, verdict.detail
    assert "native_metrics_match_feature_rerank is False" in verdict.detail


def test_a_true_flag_with_no_counts_anywhere_is_incomplete_not_a_pass(tmp_path: Path) -> None:
    """`flag=true` with nothing behind it. Used to PASS; now the dimension is unmeasured."""
    native, adapter = _native(), _adapter()
    native.pop("native_metrics_parity_detail", None)
    native["native"]["native_metrics_match_feature_rerank"] = True
    _write(tmp_path, native, adapter)
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.INCOMPLETE, verdict.detail
    assert verdict.passed is False
    assert "metrics_parity" in verdict.missing
    check = _metrics_of(verdict)[0]
    assert check.ok is False and check.absence is True
    assert "not evidence" in check.note


def test_a_false_flag_without_counts_still_cannot_pass(tmp_path: Path) -> None:
    native, adapter = _native(), _adapter()
    native.pop("native_metrics_parity_detail", None)
    native["native"]["native_metrics_match_feature_rerank"] = False
    _write(tmp_path, native, adapter)
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.INCOMPLETE, verdict.detail


def test_equal_counts_with_full_provenance_pass_and_record_the_derivation(tmp_path: Path) -> None:
    """The positive control for the metric dimension: counts + denominators + provenance."""
    native, adapter = _metrics_pair()
    _write(tmp_path, native, adapter)
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.PASS, verdict.detail
    check = next(check for check in _metrics_of(verdict)
                 if check.name.endswith(".metrics.hit_count_equality"))
    assert check.ok is True and check.mandatory is True
    assert check.measured["derived_equal"] is True
    assert check.measured["per_cutoff"] == {"r1": True, "r10": True, "r50": True}
    assert check.measured["native"] == {"r1": 375, "r10": 1032, "r50": 1498}
    assert check.measured["recorded_flag"] is True
    assert check.harness_recorded is True
    assert verdict.dimensions["metrics_parity"] is True


def test_equal_counts_over_different_denominators_do_not_certify(tmp_path: Path) -> None:
    """Same-size reordered sets: equal hits over different eligible counts is not equality."""
    native, adapter = _metrics_pair(native_count=2017, adapter_count=2016)
    _write(tmp_path, native, adapter)
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MANDATORY_FAILED, verdict.detail
    check = next(check for check in verdict.failures
                 if check.name.endswith(".metrics.eligible_denominators"))
    assert check.measured == {"native": 2017, "adapter": 2016}
    assert "different denominators" in check.note


def test_an_unasserted_eligible_query_set_does_not_certify(tmp_path: Path) -> None:
    native, adapter = _metrics_pair(identical_set=False)
    _write(tmp_path, native, adapter)
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MANDATORY_FAILED, verdict.detail
    assert any(check.name.endswith(".metrics.eligible_query_set_identical")
               for check in verdict.failures)


def test_a_metrics_block_whose_own_equality_contradicts_its_counts_is_malformed(
        tmp_path: Path) -> None:
    """The block's OWN per-cutoff verdict is cross-checked too, not just the single flag."""
    document = {
        "cutoffs": [1, 10, 50],
        "native": {"hits": {"r1": 375, "r10": 1032, "r50": 1498}, "query_count": 2017},
        "adapter": {"hits": {"r1": 376, "r10": 1032, "r50": 1498}, "query_count": 2017},
        "eligible_query_set_identical": True,
        "equality": {"r1": True, "r10": True, "r50": True}}   # r1 is a lie
    native, adapter = _metrics_pair(detail=document, flag=True)
    _write(tmp_path, native, adapter)
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MALFORMED, verdict.detail
    assert ".equality says" in verdict.detail


def test_missing_provenance_is_a_named_mandatory_failure(tmp_path: Path) -> None:
    """Counts without the identifiers that make them comparable: a failure naming each gap."""
    native, adapter = _metrics_pair(provenance=None)
    for document in (native, adapter):
        for field in ("protocol", "protocol_id", "checkpoint_id", "query_order_sha256",
                      "gallery_order_sha256"):
            document.pop(field, None)
    _write(tmp_path, native, adapter)
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MANDATORY_FAILED, verdict.detail
    check = next(check for check in verdict.failures if check.name.endswith(".metrics.provenance"))
    assert check.ok is False
    for field in ("protocol", "checkpoint_id", "query_order_sha256", "gallery_order_sha256"):
        assert f"`{field}`" in check.note, (field, check.note)


def test_provenance_on_one_side_only_is_a_failure(tmp_path: Path) -> None:
    native, adapter = _metrics_pair()
    adapter.pop("query_order_sha256")
    _write(tmp_path, native, adapter)
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MANDATORY_FAILED, verdict.detail
    check = next(check for check in verdict.failures if check.name.endswith(".metrics.provenance"))
    assert "one side only" in check.note


def test_a_missing_cutoff_in_one_count_vector_is_malformed(tmp_path: Path) -> None:
    native, adapter = _metrics_pair(adapter_hits={"r1": 375, "r10": 1032})
    _write(tmp_path, native, adapter)
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MALFORMED, verdict.detail
    assert "'r50'" in verdict.detail


def test_a_non_integer_hit_count_is_malformed(tmp_path: Path) -> None:
    native, adapter = _metrics_pair(adapter_hits={"r1": 375.0, "r10": 1032, "r50": 1498})
    _write(tmp_path, native, adapter)
    verdict = _evaluate(tmp_path)
    assert verdict.status == gate.MALFORMED, verdict.detail
    assert "not an integer hit count" in verdict.detail


def test_the_shipped_limn_pair_reports_no_false_metric_pass() -> None:
    """The LIVE LIMN artifacts: whatever they record, `metrics_parity` is never True on a flag alone."""
    review = (Path(__file__).resolve().parents[1] / "docs" / "reviews"
              / "20261109-native15-composition-limn")
    if not (review / "g1_g2.dress.native.json").is_file():
        pytest.skip("the LIMN artifacts are not on this host")
    verdict = gate.evaluate_model("limn")
    checks = [check for check in verdict.checks if check.dimension == "metrics_parity"]
    assert checks, "LIMN records a metric comparison, so the dimension must carry checks"
    for check in checks:
        if not check.name.endswith(".metrics.hit_count_equality"):
            continue
        measured = check.measured
        assert measured.get("derived_equal") is True, (
            f"{check.name}: enabled only by a recorded boolean, not by a derivation")
        assert measured.get("native") == measured.get("adapter"), check.name
    assert verdict.status != gate.PASS


# ---------------------------------------------- the aggregate memo: exactness, not speed

def _stamp_fixture(tmp_path: Path, t4) -> Path:
    """A complete evidence directory the memo can stamp: one model, all three categories, 6 files."""
    for category in ("dress", "shirt", "toptee"):
        native, adapter = _large_pair(t4, offender=None)
        for document in (native, adapter):
            document["category"] = category
        for side, document in (("native", native), ("adapter", adapter)):
            (tmp_path / f"g1_g2.testmodel.{category}.{side}.json").write_text(
                json.dumps(document), encoding="utf-8")
    return tmp_path


def _count_derivations(monkeypatch) -> list[int]:
    """Count entries into the UNCACHED derivation, which is what a warm hit must avoid."""
    calls: list[int] = []
    original = gate.evaluate_all_uncached

    def counting(**kwargs):
        calls.append(1)
        return original(**kwargs)

    monkeypatch.setattr(gate, "evaluate_all_uncached", counting)
    monkeypatch.setattr(gate, "_EVALUATE_ALL_CACHE", [])
    return calls


def test_a_warm_request_derives_once_and_returns_the_same_payload(tmp_path: Path, t4,
                                                                  monkeypatch) -> None:
    """Two requests, one derivation. Asserted on CALLS, never on elapsed time."""
    root = _stamp_fixture(tmp_path, t4)
    calls = _count_derivations(monkeypatch)
    first = gate.evaluate_all(root=root)
    second = gate.evaluate_all(root=root)
    assert len(calls) == 1, calls
    assert first == second


def test_touching_an_artifact_re_derives(tmp_path: Path, t4, monkeypatch) -> None:
    """The stamp is the whole safety property: a rewritten artifact MUST re-derive."""
    root = _stamp_fixture(tmp_path, t4)
    calls = _count_derivations(monkeypatch)
    before = gate.evaluate_all(root=root)
    assert len(calls) == 1
    target = root / "g1_g2.testmodel.shirt.adapter.json"
    os.utime(target, ns=(time.time_ns(), time.time_ns()))
    after = gate.evaluate_all(root=root)
    assert len(calls) == 2, calls
    assert after == before, "a touch with identical bytes must change the CACHE, not the verdict"


def test_rewriting_an_artifact_is_never_served_from_the_cache(tmp_path: Path, t4,
                                                              monkeypatch) -> None:
    """Same-size rewrite with a new mtime: the stamp sees it and the verdict follows the bytes."""
    root = _stamp_fixture(tmp_path, t4)
    calls = _count_derivations(monkeypatch)
    first = gate.evaluate_all(root=root)
    assert first["models"]["testmodel"]["status"] == gate.PASS, first["models"]["testmodel"]["detail"]
    adapter_path = root / "g1_g2.testmodel.shirt.adapter.json"
    document = json.loads(adapter_path.read_text())
    scope = document["component_parity"]["scopes"]["sample_forward.query"]
    # Publish the offender the record was built without: the verdict must stop being a PASS.
    scope["mismatch_count"] = 1
    scope["offenders"] = [{"index": 0, "reference": 0.5, "actual": 0.51, "normalized": 166.6}]
    adapter_path.write_text(json.dumps(document))
    second = gate.evaluate_all(root=root)
    assert len(calls) == 2, calls
    assert second["models"]["testmodel"]["status"] != gate.PASS


def test_a_caller_mutating_the_payload_cannot_poison_the_next_request(tmp_path: Path, t4,
                                                                     monkeypatch) -> None:
    root = _stamp_fixture(tmp_path, t4)
    calls = _count_derivations(monkeypatch)
    first = gate.evaluate_all(root=root)
    first["models"].clear()
    first["passed_models"].append("POISON")
    second = gate.evaluate_all(root=root)
    assert len(calls) == 1, "a mutation is not a cache miss"
    assert "POISON" not in second["passed_models"]
    assert second["models"], "the cached payload must be deep-copied, not handed out"


def test_the_memoised_payload_is_byte_identical_to_the_derivation(tmp_path: Path, t4,
                                                                  monkeypatch) -> None:
    """A memoised payload that differs from the derivation is worse than a slow one."""
    root = _stamp_fixture(tmp_path, t4)
    _count_derivations(monkeypatch)
    memoised = gate.evaluate_all(root=root)
    derived = gate.evaluate_all_uncached(root=root)
    assert json.dumps(memoised, sort_keys=True) == json.dumps(derived, sort_keys=True)


def test_the_requested_standard_is_part_of_the_key(tmp_path: Path, t4, monkeypatch) -> None:
    """A v2 request reports a different advisory block, so it cannot share the v1 entry."""
    root = _stamp_fixture(tmp_path, t4)
    calls = _count_derivations(monkeypatch)
    v1 = gate.evaluate_all(root=root)
    v2 = gate.evaluate_all(root=root, standard=gate.STANDARD_ABS_REL_V2)
    assert len(calls) == 2, calls
    assert v1["requested_standard"] != v2["requested_standard"]
    assert v1["standard"] == v2["standard"], "the APPLIED standard still comes from the artifacts"


def test_the_cache_holds_one_entry_only(tmp_path: Path, t4, monkeypatch) -> None:
    root = _stamp_fixture(tmp_path, t4)
    _count_derivations(monkeypatch)
    gate.evaluate_all(root=root)
    gate.evaluate_all(root=root, standard=gate.STANDARD_ABS_REL_V2)
    gate.evaluate_all(root=root)
    assert len(gate._EVALUATE_ALL_CACHE) == 1, gate._EVALUATE_ALL_CACHE


def test_an_unlistable_evidence_root_derives_without_caching(tmp_path: Path) -> None:
    """A directory that cannot be read must never be remembered as 'no artifacts'."""
    missing = tmp_path / "not-there"
    gate._EVALUATE_ALL_CACHE.clear()
    result = gate.evaluate_all(root=missing)
    # Nothing to derive, and nothing cached: an unreadable root is not a remembered "no evidence".
    assert gate._EVALUATE_ALL_CACHE == []
    assert all(verdict["status"] in (gate.NOT_RUN, gate.EVIDENCE_MISSING)
               for verdict in result["models"].values()), result["models"]
    assert "not-there" not in json.dumps(result.get("evidence", ""))
