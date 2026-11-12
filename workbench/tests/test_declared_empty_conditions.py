"""A declared-EMPTY condition list must not be served as the protocol triple.

`run_conditions` collapsed three states into two: a summary with no `conditions` field and one that
declared `conditions: []` both returned ``None``, and ``None`` means "unknown, fall back to the
protocol triple". So a run that declared it measured no condition was served
`(multimodal, text_only_black_image, image_only_empty_text)` -- three conditions it never measured.

The contract now:

    field absent  -> None  -> `conditions_declaration: UNKNOWN`        -> protocol triple stands in
    `[]`          -> ()    -> `conditions_declaration: DECLARED_EMPTY` -> served as an empty list
    `[...]`       -> names -> `conditions_declaration: DECLARED`       -> those conditions

Every on-disk summary on this host has a non-empty list (36/36), so this is a latent defect: the
last test pins the real CLVC Full run so the fix cannot change a real payload.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from workbench.backend import composition as C

CONDITIONS = ("multimodal", "text_only_black_image", "image_only_empty_text")
VARIANT = ("multimodal", "text_only_black_image", "clvc_image_encoder_only")
VARIANT_RUN = "da0a2162be0e159a1a456cec"

@pytest.fixture(autouse=True)
def _fresh_caches():
    C._SUMMARY_FILE_CACHE.clear()
    C._RANKS_CACHE.clear()
    C._DERIVED_FILE_CACHE.clear()
    yield
    C._SUMMARY_FILE_CACHE.clear()

def _summary(root: Path, run_id: str, *, drop_conditions: bool, conditions=()):
    payload = {
        "schema_version": 1, "model_id": "m", "checkpoint_id": "c", "run_id": run_id,
        "subset": "full", "protocol_id": "circus_fashioniq_release_v1",
        "query_count": 1, "gallery_count": 2, "execution_status": "COMPLETED",
        "metrics": {"coverage": 1.0},
    }
    if drop_conditions:
        payload.pop("conditions", None)
    else:
        payload["conditions"] = list(conditions)
    path = root / "composition" / "summary" / f"m_c_{run_id}" / "full.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path

# --------------------------------------------------------------------------- the three states

def test_the_field_absent_is_unknown_not_empty():
    """Absent the field, the run said nothing -- which is not the same as saying "none"."""
    assert C.run_conditions({}) is None
    assert C.run_conditions(None) is None
    assert C.run_conditions({"metrics": {}}) is None
    assert C.conditions_declaration({}) == C.CONDITIONS_UNKNOWN
    # An unknown list falls back to the protocol triple: that is a stand-in, and the declaration
    # key is what tells a reader so.
    assert C.displayed_conditions({}) == CONDITIONS

def test_a_declared_empty_list_is_an_empty_tuple_not_none():
    assert C.run_conditions({"conditions": []}) == ()
    assert C.run_conditions({"conditions": ()}) == ()
    assert C.conditions_declaration({"conditions": []}) == C.CONDITIONS_DECLARED_EMPTY
    # The defect: this served all three protocol names for a run that measured none of them.
    assert C.displayed_conditions({"conditions": []}) == ()
    assert C.displayed_conditions({"conditions": [], "condition_status": {}}) == ()

def test_a_declared_non_empty_list_is_those_conditions():
    assert C.run_conditions({"conditions": list(VARIANT)}) == VARIANT
    assert C.conditions_declaration({"conditions": list(VARIANT)}) == C.CONDITIONS_DECLARED
    assert C.displayed_conditions({"conditions": list(VARIANT)}) == VARIANT
    # Protocol names first even when the declaration lists them in another order.
    assert C.run_conditions({"conditions": ["clvc_image_encoder_only", "multimodal"]}) == (
        "multimodal", "clvc_image_encoder_only")

def test_the_three_states_are_pairwise_distinct():
    """One assertion that the conflation is gone, stated as the invariant it is."""
    absent = C.run_conditions({})
    empty = C.run_conditions({"conditions": []})
    declared = C.run_conditions({"conditions": ["multimodal"]})
    assert absent is not empty, (absent, empty)
    assert empty != declared and absent != declared
    assert (absent, empty, declared) == (None, (), ("multimodal",))
    assert len({C.conditions_declaration({}), C.conditions_declaration({"conditions": []}),
                C.conditions_declaration({"conditions": ["multimodal"]})}) == 3

def test_a_non_list_conditions_value_is_unknown_rather_than_empty():
    """A malformed field is not a declaration of zero conditions."""
    for malformed in ("multimodal", {"a": 1}, 3, None):
        assert C.run_conditions({"conditions": malformed}) is None, malformed
        assert C.conditions_declaration({"conditions": malformed}) == C.CONDITIONS_UNKNOWN

# ------------------------------------------------------------------------ the served payloads

def test_the_detail_payload_serves_a_declared_empty_run_as_an_empty_list(tmp_path):
    """Read through `get_composition_run_detail`, the route the UI calls -- not the helper."""
    _summary(tmp_path, "empty1", drop_conditions=False, conditions=())

    detail = C.get_composition_run_detail("empty1", "full", tmp_path)
    assert detail is not None
    assert detail["conditions_declaration"] == C.CONDITIONS_DECLARED_EMPTY
    assert detail["conditions_display"] == [], detail["conditions_display"]

    _summary(tmp_path, "variant1", drop_conditions=False, conditions=VARIANT)
    declared = C.get_composition_run_detail("variant1", "full", tmp_path)
    assert declared is not None
    assert declared["conditions_declaration"] == C.CONDITIONS_DECLARED
    assert [item["id"] for item in declared["conditions_display"]] == list(VARIANT)

def test_a_summary_the_reader_refuses_is_not_served_under_any_declaration(tmp_path):
    """The absent-field state cannot reach the DETAIL route: `validate_summary_dict` requires the
    field, so a file without it is refused rather than served as the protocol triple. That is why
    the `UNKNOWN` path is only reachable where `summary_for_run` itself is None (next test)."""
    _summary(tmp_path, "absent1", drop_conditions=True)
    assert C.get_composition_run_detail("absent1", "full", tmp_path) is None

def test_the_queries_payload_serves_a_declared_empty_run_as_an_empty_list(tmp_path, monkeypatch):
    """The selector's own route: `conditions` empty AND declared, with no rank table to fall back on."""
    _summary(tmp_path, "empty2", drop_conditions=False, conditions=())

    class _Query:
        qid = "7:1"
        subsets = ("full",)
        category = "dress"
        query_txt = "clip"
        reference_id = 0
        positive_ids = [1]

    class _Release:
        queries = [_Query()]
        candidate_names = {0: "ref.jpg", 1: "target.jpg"}

    monkeypatch.setattr(C, "load_cached_release", lambda *a, **k: _Release())
    monkeypatch.setattr("workbench.composition.topk.read_candidates",
                        lambda *a, **k: {"available": False, "status": "TOPK_EXPORT_NOT_REQUESTED",
                                         "reason": None, "rows": {}, "manifest": None})
    page = C.get_composition_run_queries("empty2", subset="full", artifacts_root=tmp_path)
    assert page["conditions_declaration"] == C.CONDITIONS_DECLARED_EMPTY
    assert page["conditions"] == [], page["conditions"]
    assert page["items"][0]["ranks"] == {}

    # A summary the reader refuses is not served with any declaration either: `summary_for_run`
    # returns None, which is `UNKNOWN` -- never `DECLARED_EMPTY`.
    _summary(tmp_path, "absent2", drop_conditions=True)
    absent = C.get_composition_run_queries("absent2", subset="full", artifacts_root=tmp_path)
    assert absent["conditions_declaration"] == C.CONDITIONS_UNKNOWN
    assert absent["conditions_declaration"] != C.CONDITIONS_DECLARED_EMPTY

# ------------------------------------------------------------- the real run must not have moved

def test_the_clvc_full_run_still_serves_exactly_its_three_conditions():
    """Positive control on the real artifact: the defect is latent, so nothing real may change.

    Served `conditions_display` before this change was exactly this triple; it must still be, and
    the declaration must say DECLARED rather than UNKNOWN.
    """
    detail = C.get_composition_run_detail(VARIANT_RUN)
    if detail is None:
        pytest.skip(f"{VARIANT_RUN} is not served on this host")
    assert [item["id"] for item in detail["conditions_display"]] == list(VARIANT), (
        detail["conditions_display"])
    assert detail["conditions_declaration"] == C.CONDITIONS_DECLARED
    assert detail["conditions"] == list(VARIANT)
    page = C.get_composition_run_queries(VARIANT_RUN, subset="full", limit=1,
                                         condition="clvc_image_encoder_only")
    if page.get("total"):
        assert [item["id"] for item in page["conditions"]] == list(VARIANT), page["conditions"]
        assert page["conditions_declaration"] == C.CONDITIONS_DECLARED
