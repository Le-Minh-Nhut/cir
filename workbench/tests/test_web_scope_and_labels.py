"""W1/§6: the scope an unnamed URL opens, and the five parts of a result label.

Every assertion executes `workbench/frontend/scope.ts`, so a rename or a re-derivation in the shipping
module shows up here rather than as a page that quietly counts nothing.

The shapes below are the ones `/api/composition/runs` served while the stack was up on 2026-11-10:
`circus_fashioniq_release_v1` has full/sf/validated rows, `fashioniq_original_split` has aggregate
reports only, and every row's `subset_provenance.own.evidence` names whether its per-query artifact
was readable.
"""
from __future__ import annotations

import json
from pathlib import Path

import test_frontend_support as support

MODULE = support.compile_module("scope.ts", Path("/tmp/cir_frontend_scope"))
SRC = json.dumps(str(MODULE))

CIRCUS = "circus_fashioniq_release_v1"
FIQ = "fashioniq_original_split"


def _run(body: str) -> dict:
    return support.run_json(f"import * as S from {SRC};\n{body}")


def _row(model: str, subset: str, **extra) -> dict:
    """One served composition row, with the fields the module reads."""
    return {
        "model_id": model, "protocol_id": CIRCUS, "subset": subset,
        "execution_status": "COMPLETED", "is_failed": False,
        "capabilities": {"aggregate_metrics": True, "per_query_target_ranks": True,
                         "candidate_topk": True, "image_availability": True},
        "metrics": {"coverage": 1.0},
        "subset_provenance": {"own": {"evidence": "PER_QUERY_ARTIFACT"}},
        **extra,
    }


def _served_rows() -> list[dict]:
    """A faithful slice of the served payload: 3 measured models × 3 subsets, one failed model,
    one smoke-limited run and one imported row."""
    rows = []
    for model in ("limn", "conesep", "tme"):
        for subset in ("full", "sf", "validated"):
            rows.append(_row(model, subset))
    rows.append(_row("habit", "full", execution_status="FAILED", is_failed=True,
                     capabilities={"aggregate_metrics": False, "per_query_target_ranks": True,
                                   "candidate_topk": False, "image_availability": True},
                     metrics={"coverage": 0.0}))
    rows.append(_row("encoder", "full", metrics={"coverage": 1.0}))
    rows.append(_row("legacy_import", "full", provenance_source="released_audit_retriever"))
    return rows


def test_an_unnamed_scope_is_chosen_from_real_rows_not_a_constant() -> None:
    out = _run(f"""
const rows = {json.dumps(_served_rows())};
console.log('__R__' + JSON.stringify({{choice: S.chooseInitialScope(rows)}}));
""")
    choice = out["choice"]
    assert choice is not None, "an unnamed URL must still open a real scope"
    # 6 rows per subset (3 models × ... wait: full has 5 research rows) — full is the widest here.
    assert choice["protocol"] == CIRCUS, choice
    assert choice["subset"] == "full", (
        f"the landing subset must be the one with most rows, got {choice['subset']!r}")
    assert choice["rows"] >= 3, choice


def test_a_failed_row_cannot_be_the_reason_a_scope_is_chosen() -> None:
    """The one subset whose only rows failed must not win over a subset with measured rows."""
    rows = [_row("m", "full", execution_status="FAILED", is_failed=True,
                 capabilities={"aggregate_metrics": False, "per_query_target_ranks": False},
                 metrics={"coverage": 0.0}) for _ in range(5)]
    rows += [_row("m", "sf") for _ in range(2)]
    out = _run(f"""
const rows = {json.dumps(rows)};
console.log('__R__' + JSON.stringify({{choice: S.chooseInitialScope(rows)}}));
""")
    assert out["choice"]["subset"] == "sf", out["choice"]


def test_a_deployment_with_no_composition_rows_has_no_invented_scope() -> None:
    out = _run("console.log('__R__' + JSON.stringify({choice: S.chooseInitialScope([])}));")
    assert out["choice"] is None, "no rows must be 'no choice', not a hardcoded protocol"


def test_synthetic_and_imported_rows_do_not_decide_the_landing_scope() -> None:
    rows = [_row("SYNTHETIC_x", "validated", is_synthetic=True, synthetic_fixture=True)
            for _ in range(9)]
    rows.append(_row("real", "sf"))
    out = _run(f"""
const rows = {json.dumps(rows)};
console.log('__R__' + JSON.stringify({{choice: S.chooseInitialScope(rows)}}));
""")
    assert out["choice"]["protocol"] == CIRCUS and out["choice"]["subset"] == "sf", out["choice"]


def test_only_a_protocol_that_declares_subsets_prints_one() -> None:
    out = _run("""
const labels = {full: 'Full', sf: 'Shortcut-Free (SF)', validated: 'Validated'};
console.log('__R__' + JSON.stringify({
  circus: S.protocolHasSubsets("circus_fashioniq_release_v1"),
  fiq: S.protocolHasSubsets("fashioniq_original_split"),
  circusLabel: S.subsetLabelFor("circus_fashioniq_release_v1", "sf", labels),
  fiqLabel: S.subsetLabelFor("fashioniq_original_split", "sf", labels),
}));
""")
    assert out["circus"] is True and out["fiq"] is False, out
    assert out["circusLabel"] == "Shortcut-Free (SF)"
    assert out["fiqLabel"] == "", (
        "a protocol with one cohort must not print a subset label: it describes a cohort that "
        "does not exist")


def test_the_six_counts_are_separate_and_the_verified_one_is_zero() -> None:
    out = _run(f"""
const rows = {json.dumps(_served_rows())};
console.log('__R__' + JSON.stringify({{counts: S.scopeCounts(rows, "{CIRCUS}", "full")}}));
""")
    counts = out["counts"]
    # `full` rows: three measured models, the failed habit row, encoder and the imported row.
    assert counts["served"] == 6, counts
    assert counts["completed"] == 5, counts
    assert counts["artifactPresent"] == 6, counts
    assert counts["membershipDerived"] == 6, counts
    assert counts["failed"] == 1, counts
    assert counts["verified"] == 0, (
        "a composition summary carries no verifier receipt, so a non-zero verified count would be a "
        "claim the API never made")


def test_an_explicit_protocol_keeps_only_its_unstated_half_filled_from_data() -> None:
    """`?protocol=fashioniq_original_split` with no subset must not inherit CIRCUS's `sf`."""
    out = _run(f"""
const rows = {json.dumps(_served_rows())};
console.log('__R__' + JSON.stringify({{
  circus: S.chooseSubsetFor(rows, "{CIRCUS}"),
  fiq: S.chooseSubsetFor(rows, "{FIQ}"),
  nowhere: S.chooseSubsetFor(rows, "made_up"),
}}));
""")
    assert out["circus"] == "full", out
    assert out["fiq"] is None, (
        "a protocol with no composition rows has no subset to choose, and inventing one would put "
        "another cohort's label over an empty table")
    assert out["nowhere"] is None, out


def test_a_scope_count_does_not_include_another_protocol() -> None:
    rows = _served_rows() + [{"model_id": "external", "protocol_id": FIQ, "subset": "full",
                              "execution_status": "COMPLETED", "metrics": {"coverage": 1.0}}]
    out = _run(f"""
const rows = {json.dumps(rows)};
console.log('__R__' + JSON.stringify({{circus: S.scopeCounts(rows, "{CIRCUS}", "full"),
                                        fiq: S.scopeCounts(rows, "{FIQ}", "full")}}));
""")
    assert out["circus"]["served"] == 6, out["circus"]
    assert out["fiq"]["served"] == 1, out["fiq"]


def test_a_model_with_rows_elsewhere_offers_the_scope_it_lives_in() -> None:
    rows = [_row("conesep", "sf"), _row("conesep", "validated"), _row("tme", "full")]
    out = _run(f"""
const rows = {json.dumps(rows)};
console.log('__R__' + JSON.stringify({{conesep: S.otherScopeFor(rows, "conesep"),
                                        tme: S.otherScopeFor(rows, "tme"),
                                        none: S.otherScopeFor(rows, "dcnet")}}));
""")
    assert [entry["subset"] for entry in out["conesep"]] == ["sf", "validated"], out["conesep"]
    assert [entry["subset"] for entry in out["tme"]] == ["full"], out["tme"]
    assert out["none"] == [], "a model with no row anywhere must produce no link, not an empty one"


def test_the_five_label_parts_never_collapse_into_one_claim() -> None:
    out = _run("""
const measuredUnverified = S.resultLabels({execution_status: "COMPLETED", aggregate_metrics: true,
  per_query_target_ranks: true, candidate_topk: false, scored: 4000, total: 4069,
  verified: false, reproduction_status: "BLOCKED_RUNTIME_PREREQUISITES"});
const aggregateOnly = S.resultLabels({execution_status: "COMPLETED", aggregate_metrics: true,
  per_query_target_ranks: false, candidate_topk: false, scored: null, total: null,
  verified: true, verification_state: "certified"});
const failed = S.resultLabels({execution_status: "FAILED", is_failed: true,
  aggregate_metrics: false, per_query_target_ranks: true, candidate_topk: false,
  scored: 0, total: 4069, blocker_reason: "the evaluation did not complete"});
console.log('__R__' + JSON.stringify({measuredUnverified, aggregateOnly, failed}));
""")
    measured = out["measuredUnverified"]
    assert measured["measurement"] == "Đã đo", measured
    assert measured["verification"] == "Chưa xác minh", measured
    assert measured["coverage"] == "Đã chấm 4,000/4,069", measured
    assert measured["samples"] == "Có hạng · chưa xuất ứng viên", measured
    assert "BLOCKED_RUNTIME_PREREQUISITES" in measured["blocker"], measured

    aggregate = out["aggregateOnly"]
    assert aggregate["samples"] == "Không có hạng từng query", aggregate
    assert aggregate["coverage"] == "Chưa có mẫu số", aggregate
    assert "certified" in aggregate["verification"], aggregate

    failed = out["failed"]
    assert "lỗi" in failed["measurement"].lower(), failed
    assert "không hoàn tất" in failed["blocker"] or "did not complete" in failed["blocker"], failed


def test_the_named_labels_the_operator_asked_for_are_reachable() -> None:
    """The review names four sentences that must be expressible; each maps to a distinct state."""
    out = _run("""
const labels = [
  S.resultLabels({execution_status: "COMPLETED", aggregate_metrics: true,
                  per_query_target_ranks: true, candidate_topk: true, scored: 4069, total: 4069}),
  S.resultLabels({execution_status: "COMPLETED", aggregate_metrics: true,
                  per_query_target_ranks: false, candidate_topk: false}),
  S.resultLabels({execution_status: "COMPLETED", aggregate_metrics: true,
                  per_query_target_ranks: true, candidate_topk: false, scored: 4000, total: 4069,
                  is_transfer: true}),
  S.resultLabels({execution_status: "COMPLETED", aggregate_metrics: true,
                  per_query_target_ranks: true, candidate_topk: false, scored: 6001, total: 6003}),
];
console.log('__R__' + JSON.stringify({labels}));
""")
    rendered = {json.dumps(item, ensure_ascii=False) for item in out["labels"]}
    # "Có hạng · đã xuất ứng viên", "Không có hạng từng query", and the coverage pair must all appear.
    assert any("đã xuất ứng viên" in text for text in rendered), rendered
    assert any("Không có hạng từng query" in text for text in rendered), rendered
    assert any("6,001/6,003" in text for text in rendered), rendered
    # The measurement state and the verification state are never the same string.
    for item in out["labels"]:
        assert item["measurement"] != item["verification"], item


def test_the_query_state_chips_are_the_backend_contract() -> None:
    out = _run("console.log('__R__' + JSON.stringify({chips: S.QUERY_STATE_CHIPS}));")
    ids = [chip["id"] for chip in out["chips"]]
    assert ids == ["", "hit", "retrieval_failure", "unscored"], ids
    labels = {chip["id"]: chip["label"] for chip in out["chips"]}
    assert labels["hit"] == "Correct@10" and labels["retrieval_failure"] == "Miss@10", labels
    assert labels[""] == "Tất cả", labels
