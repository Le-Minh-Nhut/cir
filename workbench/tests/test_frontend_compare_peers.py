"""Comparison peers must be reachable, and two runs of one model must not share a label.

Two defects found by the read-only E2E pass (`WEB_WORKFLOW_E2E_20261111.vi.md`, lỗi #1 and #2):

* `ComparePanel` built peers as `served.filter(same protocol+subset).slice(0, 4)` in API order, so
  the four columns were always `limn, conesep, ptha_mtst, intent` and TME — the model the reader was
  comparing against HINT — could never appear. The selection the URL already holds (`runs=`) is the
  comparison the reader asked for, so it must lead; and every other row of the scope must be
  reachable, not sliced away.
* `#run-select` printed `${model_id} / ${checkpoint_id}`, which repeats inside one served scope: two
  `hint / fashioniq` and two `tme / fashioniq_0` rows at subset `full`, and only one of each pair
  exported Top-K. The label now carries a run-id stub and the candidate-export state.

Every assertion executes `workbench/frontend/catalog.ts`, i.e. the code the bundle is built from.
"""
from __future__ import annotations

import json
from pathlib import Path

import test_frontend_support as support

MODULE = support.compile_module("catalog.ts", Path("/tmp/cir_frontend_compare"))
SRC = json.dumps(str(MODULE))

CIRCUS = "circus_fashioniq_release_v1"
HINT_A = "f461f391c6d4696011be8067"
HINT_B = "4bc9893a0e3d4b8c9a1f2d3e4c5b6a70"
TME_A = "ad477312111111111111111111111111"
TME_B = "cd1a696d222222222222222222222222"


def _row(model: str, run_id: str, subset: str = "full", topk: int | None = None) -> dict:
    return {
        "run_id": run_id, "model_id": model, "checkpoint_id": "fashioniq",
        "protocol_id": CIRCUS, "subset": subset,
        "capabilities": {"aggregate_metrics": True, "per_query_target_ranks": True,
                         "candidate_topk": topk is not None, "image_availability": True},
        "capability_detail": {"candidate_topk": {
            "status": "AVAILABLE" if topk is not None else "TOPK_EXPORT_NOT_REQUESTED",
            "depth": topk}},
        "metrics": {"ndcg_multimodal": 0.1498, "coverage": 1.0},
    }


#: `/api/composition/runs?subset=full` order, as served.
SERVED = [
    _row("limn", "9e4918a6b5c243412cbc58c9", topk=100),
    _row("conesep", "aaaa1111", topk=100),
    _row("ptha_mtst", "bbbb2222", topk=100),
    _row("intent", "cccc3333", topk=100),
    _row("hint", HINT_A, topk=100),
    _row("hint", HINT_B, topk=50),
    _row("tme", TME_A, topk=None),
    _row("tme", TME_B, topk=100),
    _row("limn", "dddd4444", subset="sf", topk=100),
]


def _run(body: str) -> dict:
    return support.run_json(f"import * as C from {SRC};\n{body}")


def test_the_run_named_in_the_url_is_the_first_peer() -> None:
    """`runs=` already holds both runs of the comparison the reader asked for."""
    out = _run(f"""
const peers = C.comparePeers({json.dumps(SERVED)}, {{
  currentRunId: {json.dumps(HINT_A)}, protocol: {json.dumps(CIRCUS)}, subset: "full",
  selected: ["c:{HINT_A}", "c:{TME_B}"],
}});
console.log('__R__' + JSON.stringify(peers.map((row) => row.run_id)));
""")
    assert out[0] == TME_B, out
    assert set(out) == {row["run_id"] for row in SERVED if row["subset"] == "full"} - {HINT_A}, out


def test_tme_is_reachable_and_nothing_is_sliced_away() -> None:
    """The old peer list was API-order + slice(0, 4); TME is row 7, so it was never a column."""
    out = _run(f"""
const peers = C.comparePeers({json.dumps(SERVED)}, {{
  currentRunId: {json.dumps(HINT_A)}, protocol: {json.dumps(CIRCUS)}, subset: "full", selected: [],
}});
console.log('__R__' + JSON.stringify(peers.map((row) => row.run_id)));
""")
    assert TME_A in out and TME_B in out, out
    assert "dddd4444" not in out, "a row of another subset is not a peer"
    assert len(out) == len([row for row in SERVED if row["subset"] == "full"]) - 1, out


def test_a_peer_selection_only_changes_the_comparison_cohort() -> None:
    """A selection outside the scope contributes no peer; it never invents one."""
    out = _run(f"""
const peers = C.comparePeers({json.dumps(SERVED)}, {{
  currentRunId: {json.dumps(HINT_A)}, protocol: {json.dumps(CIRCUS)}, subset: "full",
  selected: ["c:not-in-scope", "i:index-key"],
}});
console.log('__R__' + JSON.stringify(peers.map((row) => row.run_id)));
""")
    assert out == [row["run_id"] for row in SERVED
                   if row["subset"] == "full" and row["run_id"] != HINT_A], out


def test_duplicate_model_checkpoint_labels_are_disambiguated() -> None:
    out = _run(f"""
const labels = {json.dumps(SERVED)}.map((row) => C.compositionLabel(row));
console.log('__R__' + JSON.stringify(labels));
""")
    hint = [label for label in out if label.startswith("hint /")]
    tme = [label for label in out if label.startswith("tme /")]
    assert len(hint) == 2 and hint[0] != hint[1], hint
    assert len(tme) == 2 and tme[0] != tme[1], tme
    assert HINT_A[:8] in hint[0] and HINT_B[:8] in hint[1], hint


def test_the_label_states_whether_a_candidate_export_exists() -> None:
    """Choosing the row with no export is what produced 0 candidates and TOPK_EXPORT_NOT_REQUESTED."""
    out = _run(f"""
const rows = {json.dumps(SERVED)};
console.log('__R__' + JSON.stringify({{
  withExport: C.compositionLabel(rows.find((row) => row.run_id === {json.dumps(TME_B)})),
  without: C.compositionLabel(rows.find((row) => row.run_id === {json.dumps(TME_A)})),
}}));
""")
    assert "Top-K 100" in out["withExport"], out
    assert "không có Top-K" in out["without"], out
