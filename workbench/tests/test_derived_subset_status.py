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
from types import SimpleNamespace

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

def test_the_reason_based_count_still_finds_limns_two_real_failures() -> None:
    """The fix must not launder a real failure into a placement.

    LIMN genuinely routes per category, so its two unscoped qids carry real reasons. Replayed
    through the runner's own record builder over the REAL release and the REAL resolver, the count
    is still 2 — the reason is what makes `still_unscoped` an honest number, in both directions.
    """
    from workbench.backend.composition import load_cached_release
    from workbench.composition import data as composition_data, model_limn, runner

    data = load_cached_release()
    if data is None:
        pytest.skip("CIRCUS release is not present on this host")
    from workbench.backend.operator_config import resolve_config

    fashioniq = getattr(resolve_config(), "FASHIONIQ_ROOT", None)
    if not fashioniq or not (Path(fashioniq) / "captions").is_dir():
        pytest.skip("the FashionIQ annotation root is not present on this host")
    data = composition_data.assign_garment_categories(data, Path(fashioniq))

    # The adapter's own resolver without instantiating the checkpoint (that needs a GPU): routing
    # reads only the caption annotation index, which is built from the release and the pinned
    # annotation files. Same two functions `Adapter.query_scope` itself calls.
    index = model_limn.caption_scope_index(
        Path(fashioniq), composition_data.resolved_annotation_pin(
            data, composition_data.ANNOTATION_PIN))
    adapter = SimpleNamespace(category_scopes=model_limn.Adapter.category_scopes)
    adapter.query_scope = lambda query, names=None: model_limn.resolve_query_scope(
        index, query, names)["scope"]

    queries = tuple(data.queries)
    scopes = adapter.category_scopes
    assignments = [runner._scope_for_query(adapter, query, data.candidate_names) for query in queries]
    block = runner._scope_resolution_record(adapter, queries, scopes, assignments)
    assert block["category_agnostic"] is False
    assert block["resolver"] is not None
    assert block["still_unscoped"] == len(UNSCORED), (
        f"LIMN leaves {len(UNSCORED)} queries genuinely unscoped; the recorded block says "
        f"{block['still_unscoped']}")
    assert sorted(block["unscoped_qids"]) == sorted(UNSCORED)
    # Every one of them carries a REASON, which is what the count is derived from.
    reasons = {query.qid: reason for query, (_scope, reason) in zip(queries, assignments) if reason}
    assert sorted(reasons) == sorted(UNSCORED)
    assert all("no adapter embedding scope" in reason for reason in reasons.values())
    assert block["still_unscoped"] == len(block["unscoped_qids"])
    assert block["adapter_routed"] + block["global_scoped"] + block["still_unscoped"] == len(queries)
    # LIMN has no global matrix, so nothing is "global-scoped" — unlike the TG-CIR shape.
    assert block["global_scoped"] == 0

def test_the_published_derived_rows_predate_the_key_and_keep_their_old_block() -> None:
    """The rows on disk are history: they were written before the key existed and are not rewritten.

    `category_agnostic` is forward-only, so the recorded derived rows must be READ with the parent's
    block beside them (which they carry under `parent_adapter_query_scope_resolution`) rather than
    retrofitted. What matters is that the invariant they already assert still holds.
    """
    for subset in DERIVED:
        routing = _summary(subset)["adapter_query_scope_resolution"]
        assert "category_agnostic" not in routing, (
            f"{subset} was published before the key existed; retrofitting it would falsify the "
            f"record rather than fix the writer")
        # The parent's own block is still reachable, which is what a reader must consult.
        assert routing["parent_adapter_query_scope_resolution"] is not None
        assert routing["still_unscoped"] == len(routing["unscoped_qids"])

def test_a_derived_row_of_a_category_agnostic_parent_says_so(tmp_path: Path, monkeypatch) -> None:
    """The TG-CIR shape: the parent has `category_scopes: [null]`, so the derived row copies True."""
    import importlib.util

    from workbench.composition.data import Gallery, Query, ReleaseData
    from workbench.composition.inputs import CONDITIONS
    from workbench.composition.metrics import compose
    from dataclasses import asdict

    from workbench.tests.test_derived_lineage_r09 import PUBLISHER, _write

    root = tmp_path
    artifacts = root / "workbench/artifacts"
    composition_root = artifacts / "composition"
    protocol, model, checkpoint, parent = "circus_fashioniq_release_v1", "tgcir", "ck", "parent"
    ranks = {condition: {"q1": 1, "q2": 2} for condition in CONDITIONS}
    parent_summary = {
        "execution_status": "COMPLETED", "run_id": parent, "conditions": list(CONDITIONS),
        "model_id": model, "checkpoint_id": checkpoint, "subset": "full", "protocol_id": protocol,
        "category_scopes": [None],
        "adapter_query_scope_resolution": {"loader_scoped": 2, "adapter_routed": 0,
                                           "global_scoped": 2, "still_unscoped": 0,
                                           "unscoped_qids": [], "category_agnostic": True},
        "category_scope_failures": {condition: 0 for condition in CONDITIONS},
        "category_resolution": {"ambiguous_qids": [], "ambiguous_examples": []},
        "metrics": asdict(compose(ranks, query_count=2)),
        "topk_export": {"requested": False, "status": "TOPK_EXPORT_NOT_REQUESTED"},
    }
    _write(composition_root / "runs" / parent / "ranks" / f"{model}_{checkpoint}.json",
           {"ranks": ranks, "query_failures": {}, "identity": {}})
    _write(composition_root / "summary" / f"{model}_{checkpoint}_{parent}" / "full.json",
           parent_summary)
    for condition in CONDITIONS:
        path = (composition_root / "results" / protocol / model / checkpoint / parent / "full" /
                f"{condition}.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps({"qid": qid, "failure": None})
                                  for qid in ("q1", "q2")) + "\n", encoding="utf-8")

    queries = (Query("q1", 0, "c", "a.jpg", ("b.jpg",), None, ("full", "sf")),
               Query("q2", 1, "c", "a.jpg", ("c.jpg",), None, ("full", "sf")))
    spec = importlib.util.spec_from_file_location("publish_derived_tgcir", PUBLISHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.REPO = root
    module.ARTIFACTS = artifacts
    module.COMPOSITION = composition_root
    module.EVIDENCE = root / "evidence"
    module.PARENT, module.MODEL, module.CHECKPOINT, module.PROTOCOL = parent, model, checkpoint, protocol
    module.EXPECTED_QUERY_COUNTS = {"sf": 2, "validated": 0}
    module.resolve_config = lambda: SimpleNamespace(
        WORKBENCH_CIRCUS_RELEASE_ROOT=root, FASHIONIQ_ROOT=root)
    module.load_release = lambda _: ReleaseData(
        Gallery(protocol, ("a.jpg", "b.jpg", "c.jpg")), queries,
        {"a.jpg": "a.jpg", "b.jpg": "b.jpg", "c.jpg": "c.jpg"}, {})
    module.assign_garment_categories = lambda data, _: data
    module.select_queries = lambda data, subset: tuple(
        query for query in data.queries if subset in query.subsets)
    assert module.main() == 0

    summaries = [path for path in (composition_root / "summary").glob("*/sf.json")
                 if parent not in path.parent.name]
    assert summaries, "the publisher must write the derived summary"
    routing = json.loads(summaries[0].read_text(encoding="utf-8"))["adapter_query_scope_resolution"]
    assert routing["category_agnostic"] is True, (
        "a derived row of a category-agnostic parent must not look like a routing failure")
    assert routing["still_unscoped"] == 0
