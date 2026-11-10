"""R5 at the PRODUCER layer: the serializer must not record another query's positive id.

Measured on HEAD `2e2e2d2`: `t2_dual_scope.py:273` wrote `"target_did": query.positive_ids[0]`, where
`query` was left over from the outer `for qid in QIDS` loop at `:241`. By the time the inner
serialization loop at `:257` ran, that variable held the LAST query, so all twelve cells recorded
`7:42304` -- the positive of `7:3790` -- including the six belonging to `7:3038`, whose own positive
is `7:3416`.

The serve layer was already repaired (`limn_diagnostic` takes the id from the query record and
refuses a stale one). That made the endpoint correct while the artifact on disk still lied, so a
regeneration would re-emit the wrong value and look fixed. These tests pin the producer instead:

* the id is looked up **by qid, inside the serialization loop**;
* every cell is asserted against that qid's own release positives before anything is written;
* a report carrying a cross-query id is refused as a whole, not merely not-written.

No model is loaded and nothing is measured here: the helpers are pure functions over the release
records and the report dict. That is deliberate -- a 60 s GPU rerun to test a summary-only field
would be the wrong price.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
PRODUCER = (REPO / "workbench" / "docs" / "reviews" / "20261109-native15-composition-limn"
            / "scripts" / "t2_dual_scope.py")


def _module():
    """Import the producer without running it.

    The module body only defines functions and constants; `main()` is guarded, and none of the heavy
    imports (torch, the LIMN adapter) happen at import time beyond the module-level `from ... import`
    lines the file already had. Import cost measured at ~0.14 s.
    """
    spec = importlib.util.spec_from_file_location("t2_dual_scope_producer", PRODUCER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def producer():
    return _module()


def _query(qid: str, positives: tuple[str, ...]) -> SimpleNamespace:
    return SimpleNamespace(qid=qid, positive_ids=list(positives))


def _two_queries() -> dict:
    """Two synthetic queries with DISTINCT positives, in the order the defect depended on."""
    return {
        "7:3038": _query("7:3038", ("7:3416",)),
        "7:3790": _query("7:3790", ("7:42304",)),
    }


def test_the_serializer_takes_the_did_for_its_own_qid(producer) -> None:
    """The exact shape of the defect: the LAST query in the loop must not be reused for the first."""
    by_qid = _two_queries()
    first = producer.cell_target_did("7:3038", by_qid)
    second = producer.cell_target_did("7:3790", by_qid)
    assert first == "7:3416"
    assert second == "7:42304"
    assert first != second


def test_a_later_query_in_the_mapping_cannot_change_an_earlier_cell(producer) -> None:
    """Order independence: the answer for a qid never depends on which key was visited last."""
    by_qid = _two_queries()
    # Simulate the old failure mode: an outer loop that ended on 7:3790.
    leftover = by_qid["7:3790"]
    assert leftover.positive_ids[0] == "7:42304"
    # Reading by qid still yields each query's own value, for every qid, in any order.
    for qid in ("7:3790", "7:3038", "7:3790", "7:3038"):
        assert producer.cell_target_did(qid, by_qid) == by_qid[qid].positive_ids[0]


def test_a_query_without_a_positive_is_refused(producer) -> None:
    by_qid = {"7:3038": _query("7:3038", ())}
    with pytest.raises(SystemExit, match="no positive id"):
        producer.cell_target_did("7:3038", by_qid)


def test_a_report_with_a_cross_query_did_is_refused_as_a_whole(producer) -> None:
    """The whole-report assertion. This is the state the shipped artifact is in."""
    by_qid = _two_queries()
    leaked = {
        qid: {"scopes": {scope: {condition: {"target_did": "7:42304"}
                                 for condition in producer.CONDITIONS}
                         for scope in producer.SCOPES}}
        for qid in by_qid
    }
    with pytest.raises(SystemExit, match="records target_did '7:42304'"):
        producer.assert_cells_target_own_query(leaked, by_qid)


def test_a_report_with_every_cell_correct_passes(producer) -> None:
    by_qid = _two_queries()
    correct = {
        qid: {"scopes": {scope: {condition: {
            "target_did": by_qid[qid].positive_ids[0]}
            for condition in producer.CONDITIONS} for scope in producer.SCOPES}}
        for qid in by_qid
    }
    producer.assert_cells_target_own_query(correct, by_qid)  # must not raise
    for qid in by_qid:
        for scope in producer.SCOPES:
            for condition in producer.CONDITIONS:
                assert correct[qid]["scopes"][scope][condition]["target_did"] == \
                    by_qid[qid].positive_ids[0]


def test_a_single_bad_cell_is_not_swallowed_by_its_neighbours(producer) -> None:
    """Eleven right and one wrong must still refuse: the assertion is per cell, not per query."""
    by_qid = _two_queries()
    report = {
        qid: {"scopes": {scope: {condition: {
            "target_did": by_qid[qid].positive_ids[0]}
            for condition in producer.CONDITIONS} for scope in producer.SCOPES}}
        for qid in by_qid
    }
    report["7:3038"]["scopes"]["toptee"]["image_only_empty_text"]["target_did"] = "7:42304"
    with pytest.raises(SystemExit, match="7:3038/toptee/image_only_empty_text"):
        producer.assert_cells_target_own_query(report, by_qid)


def test_the_shipped_artifact_now_satisfies_the_producer_contract(producer) -> None:
    """The regenerated summary must pass the very assertion the producer will run next time."""
    import json

    summary_path = (REPO / "workbench" / "docs" / "reviews" / "20261109-native15-composition-limn"
                    / "limn_dual_scope.json")
    if not summary_path.is_file():
        pytest.skip("dual-scope summary is not present on this host (external artifact)")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    by_qid = {qid: _query(qid, (record["release"]["positive_ids"][0],))
              for qid, record in summary["queries"].items()}
    producer.assert_cells_target_own_query(summary["queries"], by_qid,
                                           qids=tuple(summary["queries"]))
    # Every CELL now names its own query's positive; the provenance note records what changed.
    for qid, record in summary["queries"].items():
        own = record["release"]["positive_ids"][0]
        for scope, cells in record["scopes"].items():
            for condition, block in cells.items():
                assert block["target_did"] == own, (qid, scope, condition, block["target_did"], own)
    correction = summary.get("provenance_correction_r5") or {}
    assert correction.get("cells_changed") == 6
    assert correction.get("rescored") is False and correction.get("reranked") is False
    assert len(correction.get("changed") or []) == 6
    assert all(change["qid"] == "7:3038" for change in correction["changed"])


def test_the_source_no_longer_reads_the_outer_loop_variable() -> None:
    """A grep-level guard: the defect's expression must be gone from the producer's source."""
    source = PRODUCER.read_text(encoding="utf-8")
    # The DEFECT's expression, not the word: the docstring must be free to name what was wrong.
    assert '"target_did": query.positive_ids[0]' not in source, (
        "the serializer must not read a query variable left over from another loop")
    assert "cell_target_did(qid, by_qid)" in source
    assert "assert_cells_target_own_query(report[\"queries\"], by_qid)" in source
