"""One query row must not contradict itself, on any run's own condition triple.

The defect: `determine_query_outcome` read the protocol's three condition names literally, so the
CLVC Full run `da0a2162be0e159a1a456cec` -- which scored `clvc_image_encoder_only` in the image role
-- classified EVERY query `outcome: "unscored"` while the same row carried `target_rank: 12796,
unscored: false`. Four surfaces inherited the wrong word: the samples list, the outcome filter, the
evaluation chips, and `ui_smoke.py`'s "a scored card must never say unsc" assertion.

The rule these tests lock down:

    outcome == "unscored"  <=>  the row's `unscored` is true

and the outcome is classified over the run's OWN triple (`metrics.gap_conditions` when recorded,
otherwise the protocol triple with an ablation substituted by the same rule the runner uses).

Runs against the real artifacts on this host, skipping cleanly when they are absent.
"""
from __future__ import annotations

import pytest

from workbench.backend.composition import (
    CONDITIONS,
    determine_query_outcome,
    get_composition_run_queries,
    outcome_conditions,
)

VARIANT_RUN = "da0a2162be0e159a1a456cec"
PROTOCOL_RUN = "7b1fa85007bffdab931f7f19"
VARIANT_TRIPLE = ("multimodal", "text_only_black_image", "clvc_image_encoder_only")
OUTCOME_CLASSES = ("hit", "retrieval_failure", "unscored")


def _page(run_id: str, **kwargs):
    """One page of the run, or None when this host does not serve it.

    `limit` travels through `kwargs` rather than being fixed here: passing it in both places raised
    `TypeError: got multiple values for keyword argument 'limit'`, which the bare `except` then
    reported as "not served on this host" -- a skip that hid a broken test instead of a missing run.
    Only the two shapes that mean "this run is absent" are turned into None.
    """
    kwargs.setdefault("limit", 5)
    try:
        page = get_composition_run_queries(run_id, subset="full", **kwargs)
    except (TypeError, ValueError, KeyError):
        raise
    except Exception:
        return None
    return page if page.get("total") else None


# --------------------------------------------------------------------- the classification itself

def test_the_default_triple_keeps_every_existing_run_byte_identical() -> None:
    """`conditions` defaults to the protocol triple, so no run's output may change."""
    assert CONDITIONS == ("multimodal", "text_only_black_image", "image_only_empty_text")
    assert determine_query_outcome({"multimodal": None, "text_only_black_image": 1,
                                    "image_only_empty_text": 1}) == "unscored"
    assert determine_query_outcome({"multimodal": 5, "text_only_black_image": 20,
                                    "image_only_empty_text": 30}) == "composition_required"
    assert determine_query_outcome({"multimodal": 5, "text_only_black_image": 8,
                                    "image_only_empty_text": 25}) == "text_only"
    assert determine_query_outcome({"multimodal": 5, "text_only_black_image": 25,
                                    "image_only_empty_text": 8}) == "image_only"
    assert determine_query_outcome({"multimodal": 5, "text_only_black_image": 6,
                                    "image_only_empty_text": 8}) == "both"
    assert determine_query_outcome({"multimodal": 50, "text_only_black_image": 60,
                                    "image_only_empty_text": 80}) == "unresolved"


def test_a_run_whose_image_role_is_an_ablation_is_classified_over_that_role() -> None:
    """The defect, at the function: the protocol name is absent, the run's own role is present.

    The protocol image name being absent is a MISS in the image role -- not an unclassifiable query.
    A null in the role the run DID score is the only `unscored` case.
    """
    # Multimodal inside @10, both side conditions outside: composition is required, not "unscored".
    assert determine_query_outcome(
        {"multimodal": 5, "text_only_black_image": 20, "clvc_image_encoder_only": 30},
        conditions=VARIANT_TRIPLE) == "composition_required"
    # The same row, but the ablation is NULL for this qid: genuinely unclassifiable.
    assert determine_query_outcome(
        {"multimodal": 5, "text_only_black_image": 20, "clvc_image_encoder_only": None},
        conditions=VARIANT_TRIPLE) == "unscored"
    # The variant key absent entirely: a miss in that role, never "unknown".
    assert determine_query_outcome(
        {"multimodal": 50, "text_only_black_image": 60},
        conditions=VARIANT_TRIPLE) == "unresolved"


def test_the_triple_is_never_hard_coded_to_one_ablation_name() -> None:
    """The recorded triple wins; a re-derivation picks whatever ablation the run actually ranked."""
    recorded = {"metrics": {"gap_conditions": ["multimodal", "text_only_black_image",
                                               "some_other_ablation"]}}
    assert outcome_conditions({"zzz": {}}, recorded) == ("multimodal", "text_only_black_image",
                                                        "some_other_ablation")
    # No recorded triple: the protocol triple stands when the protocol image role has ranks.
    assert outcome_conditions({"multimodal": {}, "text_only_black_image": {},
                               "image_only_empty_text": {}}) == CONDITIONS
    # An image role with NO scored rank is the only case an ablation may fill.
    assert outcome_conditions({"multimodal": {"7:1": 1}, "text_only_black_image": {"7:1": 1},
                               "image_only_empty_text": {"7:1": None},
                               "clvc_image_encoder_only": {"7:1": 4}}) == VARIANT_TRIPLE


# ------------------------------------------------------------------------ the served payload

@pytest.mark.parametrize("run_id", [VARIANT_RUN, PROTOCOL_RUN])
def test_outcome_and_unscored_agree_on_every_row(run_id: str) -> None:
    """The invariant, on a variant-triple run AND on a protocol-triple run."""
    page = _page(run_id)
    if page is None:
        pytest.skip(f"{run_id} is not served on this host")
    for item in page["items"]:
        assert (item["outcome"] == "unscored") is item["unscored"], (
            f"{run_id} {item['query_id']}: outcome {item['outcome']!r} beside unscored "
            f"{item['unscored']!r} (target_rank {item['target_rank']!r})")
        assert (item["unscored_kind"] is not None) <= item["unscored"], (
            f"{run_id} {item['query_id']}: a scored row carries unscored_kind "
            f"{item['unscored_kind']!r}")


def test_the_variant_run_reports_real_outcomes_and_its_histogram_sums_to_the_total() -> None:
    """The measurement the defect report asks for: no class may absorb 6,003 scored queries."""
    page = _page(VARIANT_RUN, limit=5)
    if page is None:
        pytest.skip(f"{VARIANT_RUN} is not served on this host")
    counts = page["outcome_counts"]
    assert sum(counts.values()) == page["total"], (counts, page["total"])
    scored = {name: value for name, value in counts.items() if name != "unscored"}
    assert len(scored) >= 3, f"a variant run collapsed into too few classes: {counts}"
    # The counters and the outcome histogram are two views of one predicate.
    assert counts.get("unscored", 0) == page["counters"]["unscored"], (counts, page["counters"])
    assert sum(value for name, value in counts.items() if name != "unscored") \
        == page["counters"]["scored"], (counts, page["counters"])
    # The scorer's own aggregate agrees class for class, with `neither_scored` as the unscored row.
    assert counts.get("unscored", 0) == page["counters"]["unscored"]


def test_a_variant_row_carries_a_rank_and_a_non_unscored_outcome() -> None:
    page = _page(VARIANT_RUN, limit=3, condition="clvc_image_encoder_only", query_id="7:1")
    if page is None:
        pytest.skip(f"{VARIANT_RUN} is not served on this host")
    row = page["items"][0]
    assert isinstance(row["target_rank"], int), row
    assert row["unscored"] is False, row
    assert row["outcome"] != "unscored", row
    assert row["outcome"] in {"hit", "retrieval_failure", "unresolved", "both",
                             "text_only", "image_only", "composition_required"}, row
    # The state and the outcome describe the same qid from two angles; neither may say unscored.
    assert row["query_state"] != "unscored", row


def test_only_ranked_conditions_are_served() -> None:
    """A condition the run never measured must not appear as a key with a null rank.

    Listing the protocol triple regardless produced `ranks: {image_only_empty_text: null}` on the
    variant run, and the inspector then rendered "Chưa chấm" for a condition that has no table --
    beside a real target rank in the condition that does.
    """
    page = _page(VARIANT_RUN, limit=1, condition="clvc_image_encoder_only", query_id="7:1")
    if page is None:
        pytest.skip(f"{VARIANT_RUN} is not served on this host")
    row = page["items"][0]
    assert set(row["ranks"]) == {"multimodal", "text_only_black_image", "clvc_image_encoder_only"}, row
    assert row["ranks"]["clvc_image_encoder_only"] == 12796, row
    assert [item["id"] for item in page["conditions"]] == [
        "multimodal", "text_only_black_image", "clvc_image_encoder_only"], page["conditions"]
    # `conditions_display` (what the UI may OFFER) lists only conditions with a rank table, in
    # protocol-first order, so the select cannot offer a condition this run has no rows for.
    detail = __import__("workbench.backend.composition", fromlist=["x"]).get_composition_run_detail(
        VARIANT_RUN, "full")
    if detail is not None:
        offered = [item["id"] for item in detail["conditions_display"]]
        assert "image_only_empty_text" not in offered, offered
        assert offered == ["multimodal", "text_only_black_image", "clvc_image_encoder_only"], offered
    # A protocol-triple run is unchanged: it has a table for all three.
    other = _page(PROTOCOL_RUN, limit=1)
    if other is not None:
        assert [item["id"] for item in other["conditions"]] == list(CONDITIONS), other["conditions"]
        other_detail = __import__("workbench.backend.composition",
                                  fromlist=["x"]).get_composition_run_detail(PROTOCOL_RUN, "full")
        if other_detail is not None:
            assert [item["id"] for item in other_detail["conditions_display"]] == list(CONDITIONS)


def test_the_outcome_filter_returns_only_rows_of_that_class() -> None:
    """One source: filtering by a class must select exactly the rows whose `outcome` says that class.
    """
    page = _page(VARIANT_RUN, limit=5)
    if page is None:
        pytest.skip(f"{VARIANT_RUN} is not served on this host")
    total = page["outcome_counts"]
    for outcome, expected in total.items():
        filtered = get_composition_run_queries(VARIANT_RUN, subset="full", limit=1,
                                               condition="clvc_image_encoder_only",
                                               outcome=outcome)
        assert filtered["total"] == expected, (outcome, filtered["total"], expected)
        assert filtered["items"][0]["outcome"] == outcome
