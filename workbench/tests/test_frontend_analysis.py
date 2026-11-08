"""Unscored is a pipeline fact, not a retrieval failure; the cohort and its denominator are stated.

Two defects this pins down, both visible in the served data:
  * 119 of 6,003 queries in the CIRCUS release have no exact rank (`neither_scored`); sorting
    "worst first" over a nullable rank floated all of them above a genuine rank-70,016 failure.
  * comparing runs on different query sets publishes a metric whose denominator nobody stated.

`workbench/frontend/analysis.ts` is executed through node, so the rule under test is the shipped one.
"""
from __future__ import annotations

import json
from pathlib import Path

import test_frontend_support as support

MODULE = support.compile_module("analysis.ts", Path("/tmp/cir_frontend_analysis"))
SRC = json.dumps(str(MODULE))


def _run(statements: str) -> dict:
    return support.run_json(f"import * as A from {SRC};\n{statements}")


def _row(query_id: str, rank: int | None, *, category: str | None = "dress",
         outcome: str = "unresolved", ranks: dict | None = None) -> dict:
    return {
        "query_id": query_id, "category": category, "query_txt": "x", "reference_id": "r",
        "positive_ids": ["p"], "ranks": ranks or {"multimodal": rank}, "target_rank": rank,
        "outcome": outcome,
    }


def test_a_null_rank_is_unscored_and_never_the_worst_failure() -> None:
    pages = [
        {"runKey": "a", "runLabel": "A", "items": [
            _row("q-unscored", None), _row("q-huge", 70016), _row("q-good", 3)]},
        {"runKey": "b", "runLabel": "B", "items": [
            _row("q-unscored", None), _row("q-huge", 66000), _row("q-good", 7)]},
    ]
    out = _run(f"""
const pages = {json.dumps(pages)};
const scope = A.withCutoff(A.buildScope(pages), 10);
const worst = A.sortRows(scope.rows, "worst").map((r) => [r.query_id, r.unscored, r.maxRank]);
const all = A.filterRows(scope, {{preset: "all", k: 10, category: "", outcome: "", scoring: ""}})
  .map((r) => r.query_id);
const onlyUnscored = A.filterRows(scope, {{preset: "all", k: 10, category: "", outcome: "",
  scoring: "unscored"}}).map((r) => r.query_id);
const scoredOnly = A.filterRows(scope, {{preset: "allfail", k: 10, category: "", outcome: "",
  scoring: ""}}).map((r) => r.query_id);
console.log('__R__' + JSON.stringify({{worst, all, onlyUnscored, scoredOnly,
  unscoredCount: scope.unscored, cohort: scope.cohort}}));
""")
    order = [row[0] for row in out["worst"]]
    assert order[-1] == "q-unscored", (
        "the unscored query was not last in worst-first order", out["worst"])
    assert order[:2] == ["q-huge", "q-good"], out["worst"]
    assert out["onlyUnscored"] == ["q-unscored"], out
    assert "q-unscored" not in out["scoredOnly"], (
        "an unscored query was recruited into a consensus-failure preset", out["scoredOnly"])
    assert out["unscoredCount"] == 1, out


def test_the_cohort_publishes_its_denominator_and_counts_partial_overlap() -> None:
    """Runs with different universes: the cohort is the intersection, partial is counted, not
    silently dropped and not silently averaged in."""
    pages = [
        {"runKey": "a", "runLabel": "A", "items": [
            _row("q1", 2), _row("q2", 5), _row("q3", 9)]},
        {"runKey": "b", "runLabel": "B", "items": [
            _row("q1", 4), _row("q2", 8), _row("q9", 1)]},
    ]
    out = _run(f"""
const scope = A.buildScope({json.dumps(pages)});
console.log('__R__' + JSON.stringify({{cohort: scope.cohort, partial: scope.partial,
  runs: scope.runs, ids: scope.rows.map((r) => r.query_id).sort()}}));
""")
    assert out["cohort"] == 2, out          # q1, q2 appear in both
    assert out["partial"] == 2, out         # q3 (a only), q9 (b only)
    assert out["runs"] == 2, out
    assert out["ids"] == ["q1", "q2", "q3", "q9"], out


def test_every_rank_is_kept_per_run_and_per_condition() -> None:
    a = _row("q1", 11, ranks={"multimodal": 11, "text_only_black_image": None,
                              "image_only_empty_text": 3})
    b = _row("q1", 40, ranks={"multimodal": 40, "text_only_black_image": 12,
                              "image_only_empty_text": 9})
    out = _run(f"""
const scope = A.buildScope([
  {{runKey: "a", runLabel: "A", items: [{json.dumps(a)}]}},
  {{runKey: "b", runLabel: "B", items: [{json.dumps(b)}]}},
]);
const row = scope.rows[0];
console.log('__R__' + JSON.stringify({{ranks: row.ranks,
  mm: row.conditionRanks.multimodal, text: row.conditionRanks.text_only_black_image,
  image: row.conditionRanks.image_only_empty_text,
  median: row.medianRank, range: row.rankRange}}));
""")
    assert out["ranks"] == {"a": 11, "b": 40}, out
    assert out["text"] == {"a": None, "b": 12}, out
    assert out["median"] == 25.5, out
    assert out["range"] == 29, out


def test_the_consensus_cutoff_is_applied_and_reported() -> None:
    pages = [
        {"runKey": "a", "runLabel": "A", "items": [_row("q1", 30), _row("q2", 2)]},
        {"runKey": "b", "runLabel": "B", "items": [_row("q1", 50), _row("q2", 40)]},
    ]
    out = _run(f"""
const base = A.buildScope({json.dumps(pages)});
const at10 = A.withCutoff(base, 10);
const at100 = A.withCutoff(base, 100);
console.log('__R__' + JSON.stringify({{
  at10: at10.rows.map((r) => [r.query_id, r.failFraction]),
  at100: at100.rows.map((r) => [r.query_id, r.failFraction]),
}}));
""")
    assert dict(map(tuple, out["at10"])) == {"q1": 1, "q2": 0.5}, out["at10"]
    assert dict(map(tuple, out["at100"])) == {"q1": 0, "q2": 0}, out["at100"]


def test_pagination_is_a_window_not_the_whole_set() -> None:
    """`total` must be the full filtered set, or "save filtered set" cannot save more than a page."""
    rows = list(range(57))
    out = _run(f"""
const rows = {json.dumps(rows)}.map((n) => ({{n}}));
const page = A.paginate(rows, 40, 20);
console.log('__R__' + JSON.stringify({{total: page.total, count: page.items.length,
  first: page.items[0].n, offset: page.offset}}));
""")
    assert out["total"] == 57, out
    assert out["count"] == 17, out
    assert out["first"] == 40, out
