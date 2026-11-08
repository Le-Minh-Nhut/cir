"""Scoped counters must agree with the scope they describe, for every subset.

`measurement.scored_count` used to be whole-run while `sample_count` followed the requested subset,
so `subset=sf` rendered `5,884 / 4,069 (144.6%)`. A coverage above 100% invalidates every other
number on the page, so the invariant is asserted per subset rather than spot-checked:

    scored_count + unscored_query_count == sample_count   (within one scope)
    coverage <= 1.0
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from workbench.backend.composition import (
    _ranks_statistics,
    get_composition_run_detail,
    list_composition_summaries,
    subset_qids,
)

CONDITIONS = ("multimodal", "text_only_black_image", "image_only_empty_text")
SUBSETS = ("full", "sf", "validated")


def _assert_scoped(measurement: dict, statistics: dict, label: str) -> None:
    assert measurement["scored_count"] is not None, label
    assert measurement["unscored_query_count"] is not None, label
    assert measurement["scored_count"] + measurement["unscored_query_count"] \
        == measurement["sample_count"], (
            f"{label}: scored {measurement['scored_count']} + unscored "
            f"{measurement['unscored_query_count']} != sample {measurement['sample_count']}")
    assert measurement["coverage"] is None or measurement["coverage"] <= 1.0, (
        f"{label}: coverage {measurement['coverage']} exceeds 1.0")
    for condition, values in statistics["conditions"].items():
        assert values["scored_count"] + values["unscored_count"] == values["denominator"], (
            f"{label}/{condition}: scored {values['scored_count']} + unscored "
            f"{values['unscored_count']} != denominator {values['denominator']}")
        assert values["coverage"] <= 1.0, f"{label}/{condition}: coverage exceeds 1.0"
    assert statistics["query_count"] == measurement["sample_count"], (
        f"{label}: statistics.query_count {statistics['query_count']} != sample_count "
        f"{measurement['sample_count']}")


def test_every_subset_reports_counters_scoped_to_itself():
    """Uses the real artifacts on this machine, skipping cleanly when none exist."""
    rows = list_composition_summaries(subset=None, include_fixtures=False, include_smoke=False,
                                      include_imported=False)
    if not rows:
        pytest.skip("no real composition summaries on disk")
    run_id = rows[0]["run_id"]
    seen = set()
    for subset in SUBSETS:
        detail = get_composition_run_detail(run_id, subset=subset)
        if detail is None:
            continue
        seen.add(subset)
        measurement = detail["measurement"]
        assert measurement["subset"] == subset
        _assert_scoped(measurement, detail["statistics"], f"{run_id}/{subset}")
        # The whole-run figure is separate and explicitly named, never blended into the scoped one.
        assert detail["statistics_whole_run"]["query_count"] >= detail["statistics"]["query_count"]
    assert "full" in seen, "the full subset must always be reportable"


def test_subset_qids_narrow_the_cohort():
    full = subset_qids("full")
    sf = subset_qids("sf")
    validated = subset_qids("validated")
    if full is None:
        pytest.skip("release dataset not available locally")
    assert validated is not None and sf is not None
    assert len(full) == 6003, f"full subset must be the whole released cohort, got {len(full)}"
    assert validated <= sf <= full
    assert len(sf) < len(full), "sf must be a strict subset"

def test_statistics_helper_is_scope_aware(tmp_path: Path):
    ranks_data = {"ranks": {condition: {"q1": 3, "q2": None, "q3": 9, "q4": None}
                            for condition in CONDITIONS},
                  "query_failures": {condition: {"q2": "query features unavailable",
                                                 "q4": "query reference image is missing"}
                                     for condition in CONDITIONS}}
    whole = _ranks_statistics(ranks_data, None, None)
    scoped = _ranks_statistics(ranks_data, 3, {"q1", "q2", "q3"})
    assert whole["conditions"]["multimodal"]["denominator"] == 4
    assert scoped["conditions"]["multimodal"]["denominator"] == 3
    assert scoped["conditions"]["multimodal"]["scored_count"] == 2
    assert scoped["conditions"]["multimodal"]["unscored_count"] == 1
    assert scoped["conditions"]["multimodal"]["unscored_by_kind"] == {"features_unavailable": 1}
    assert scoped["query_count"] == 3
    assert whole["conditions"]["multimodal"]["unscored_by_kind"] == {
        "features_unavailable": 1, "reference_image_missing": 1}

def test_empty_scope_does_not_divide_by_zero():
    ranks_data = {"ranks": {"multimodal": {"q1": 1}}, "query_failures": {}}
    scoped = _ranks_statistics(ranks_data, 0, set())
    assert scoped["conditions"]["multimodal"]["denominator"] == 0
    assert scoped["conditions"]["multimodal"]["coverage"] == 1.0
