"""A derived LIMN subset must state its OWN completeness, without widening `execution_status`.

The problem this file exists for, measured on the published rows: `validated` carries
`execution_status: PARTIAL` (inherited from its parent) while its own coverage is 1.0 — 586/586 scored
in all three conditions. `PARTIAL` means "some queries of THIS artifact are unscored", so a complete
subset labelled `PARTIAL` under-claims, and worse, it makes "derived from a parent with 2 unscored
queries" indistinguishable from "this subset has unscored queries".

The argument for a distinct status is therefore still true — and it is answered by putting the label in
`derived_from`, not in `execution_status`. `execution_status` is an enum four other places read as "did
this run finish" (`workbench/frontend/scope.ts:151`, `OverviewPage.tsx:251`,
`EvaluationPage.tsx:259`, `workbench/backend/composition.py:189`); widening it for one label moves a
contract other code depends on, whereas `derived_from` is already the field that says "this is not a
fresh measurement". So the contract asserted here is:

  * `execution_status` stays inside COMPLETED|PARTIAL|REUSED|FAILED — inherited, the parent's value;
  * `derived_from.own_status_label` names THIS subset's completeness ("DERIVED_PARTIAL" /
    "DERIVED_COMPLETE");
  * `derived_from.own_unscored_count` / `own_scored_count` / `own_coverage` are the subset's own
    numbers, and `parent_unscored_qids` / `parent_unscored_qids_in_this_subset` say what the parent's
    partiality was and which of those qids fall inside this membership.

The facts the two states must keep apart:
  * the parent `9e4918a6…` is PARTIAL with coverage 0.999667 over full (6001/6003);
  * `7:3038` and `7:3790` are the two unscored qids;
  * `sf` membership CONTAINS `7:3790` but not `7:3038`, so sf is genuinely partial (4068/4069);
  * `validated` membership contains NEITHER, so validated is genuinely complete (586/586).

Deliberately NOT asserted: that `execution_status` never repeats the parent's value. It does repeat it,
by decision, and a test that flipped the expected string would be pinning the label in the wrong field.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

SUMMARIES = REPO / "workbench/artifacts/composition/summary"

#: run_id, this subset's own label, own scored count, own unscored count, membership size
DERIVED = {
    "sf": ("6a96c5c199526b61c1470b0e", "DERIVED_PARTIAL", 4068, 1, 4069),
    "validated": ("aee2e8d39214d877a84d7c08", "DERIVED_COMPLETE", 586, 0, 586),
}
UNSCORED = ("7:3038", "7:3790")
PARENT_COVERAGE = 0.9996668332500417
ENUM = frozenset({"COMPLETED", "PARTIAL", "REUSED", "FAILED"})


def _summary(subset: str) -> dict:
    run_id = DERIVED[subset][0]
    path = SUMMARIES / f"limn_base_iter0_all_categories_{run_id}" / f"{subset}.json"
    if not path.is_file():
        pytest.skip(f"derived run {run_id} is not published on this host")
    return json.loads(path.read_text(encoding="utf-8"))


def test_membership_of_the_two_unscored_qids() -> None:
    """The premise: the parent's unscored pair is not evenly distributed across the subsets."""
    from workbench.composition.data import (
        assign_garment_categories, load_release, select_queries)

    # Deliberately NOT `resolve_config()`: the suite's autouse fixture blanks every operator
    # setting (`conftest.py:31`), so the pinned repo-relative roots are the honest source here.
    release_root = REPO / "workbench/artifacts/circus_release"
    fashioniq = REPO / "data/FashionIQ"
    if not (release_root / "mbeir_fashioniq_task7_test.jsonl").is_file():
        pytest.skip("CIRCUS release is not present on this host")
    data = assign_garment_categories(load_release(release_root), fashioniq)
    sf = {query.qid for query in select_queries(data, "sf")}
    validated = {query.qid for query in select_queries(data, "validated")}
    assert "7:3790" in sf and "7:3038" not in sf
    assert not ({"7:3038", "7:3790"} & validated)


def test_execution_status_stays_inside_the_existing_enum() -> None:
    """Option (b): the enum is inherited from the parent, never widened by a derived run."""
    for subset in DERIVED:
        summary = _summary(subset)
        assert summary["execution_status"] in ENUM, (
            f"{subset} states execution_status={summary['execution_status']!r}, which is outside the "
            "enum four consumers read; a derived run reports its own label in derived_from instead")
        # The concrete inherited value for this parent, stated so a change is visible.
        assert summary["execution_status"] == "PARTIAL"


def test_the_subset_own_completeness_is_distinguishable_from_the_parent() -> None:
    """The distinguishing fields: what the READER needs to tell 2 parent-unscored from 0-here."""
    for subset, (run_id, label, scored, own_unscored, membership) in DERIVED.items():
        summary = _summary(subset)
        derived = summary["derived_from"]
        metrics = summary["metrics"]["multimodal"]
        assert summary["run_id"] == run_id
        # 1. this subset's own label, in the field that means "not a fresh measurement"
        assert derived["own_status_label"] == label
        # 2. its own unscored count, which is the fact the parent's PARTIAL would otherwise hide
        assert derived["own_unscored_count"] == own_unscored
        # 3. its own denominators
        assert derived["own_scored_count"] == scored
        assert metrics["scored_count"] == scored and metrics["query_count"] == membership
        # 4. its own coverage, not the parent's
        assert derived["own_coverage"] == pytest.approx(metrics["coverage"])
        assert derived["own_coverage"] == pytest.approx(scored / membership)
        assert derived["own_complete"] is (own_unscored == 0)
        # 5. the membership it was derived over
        assert derived["membership_count"] == membership
        # 6. provenance of the derivation itself
        assert derived["schema"] == "circus-derived-subset/v1"
        assert derived["parent_run_id"] == "9e4918a6b5c243412cbc58c9"
        assert derived["gpu_used"] is False and derived["re_encoded"] is False
        assert derived["rescored"] is False


def test_the_parent_partiality_is_still_reported_separately() -> None:
    """"This subset is complete" must never erase "the parent had two unscored queries"."""
    for subset in DERIVED:
        derived = _summary(subset)["derived_from"]
        assert derived["parent_execution_status"] == "PARTIAL"
        assert derived["parent_coverage"] == pytest.approx(PARENT_COVERAGE)
        assert derived["parent_unscoped_qids"] == list(UNSCORED)
    assert _summary("sf")["derived_from"]["parent_unscoped_qids_in_this_subset"] == ["7:3790"]
    assert _summary("validated")["derived_from"]["parent_unscoped_qids_in_this_subset"] == []


def test_routing_and_scoring_never_contradict_each_other() -> None:
    """The invariant the publisher enforces: still_unscoped == unscored_count, per condition."""
    for subset, (_, label, scored, own_unscored, membership) in DERIVED.items():
        summary = _summary(subset)
        routing = summary["adapter_query_scope_resolution"]
        assert routing["still_unscoped"] == own_unscored, (
            f"{subset} says {routing['still_unscoped']} unscoped qid(s) but "
            f"{own_unscored} unscored query/queries; the row would undercut itself")
        assert len(routing["unscoped_qids"]) == own_unscored
        for condition, row in summary["condition_status"].items():
            assert row["unscored_count"] == own_unscored, (
                f"{subset}/{condition} states unscored_count={row['unscored_count']}, "
                f"but the subset's own arithmetic is {membership}-{scored}={own_unscored}")
            assert row["status"] == ("COMPLETED" if own_unscored == 0 else "PARTIAL")
        # The recomputed scope blocks describe THIS subset, not the parent's query set.
        for count in (summary["category_scope_failures"] or {}).values():
            assert count == own_unscored
        assert summary["unresolved_category_query_count"] == own_unscored
        # ...while the parent's pathology stays visible under its own key.
        assert set((summary.get("parent_category_scope_failures") or {}).values()) == {
            len(UNSCORED)}
