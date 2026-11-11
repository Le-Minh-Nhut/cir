"""R09: a derived run must carry the parent's failure reasons and serve the PARENT's candidates.

Both facts are proven through the real publisher script and the real reader: the publisher is
executed against a temporary artifact root (it is the only thing that can write a derived run), and
the reader is then asked for that run. A test that hand-built the derived record would prove
nothing about what the publisher actually writes.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from workbench.backend import composition  # noqa: E402
from workbench.composition import topk  # noqa: E402
from workbench.composition.data import Gallery, Query, ReleaseData  # noqa: E402
from workbench.composition.inputs import CONDITIONS  # noqa: E402
from workbench.composition.metrics import compose  # noqa: E402

PUBLISHER = (REPO / "workbench/docs/reviews/20261111-master-audit/scripts/m03/"
             "publish_derived_runs.py")
PARENT = "parent"
MODEL = "limn"
CHECKPOINT = "base_iter0_all_categories"
PROTOCOL = "circus_fashioniq_release_v1"
UNSCORED_REASON = "query q1: category unresolved has no adapter embedding scope"
GALLERY = ("a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg")


def _release() -> SimpleNamespace:
    """A release whose gallery and query identities are exactly the artifact's own."""
    gallery = SimpleNamespace(protocol_id=PROTOCOL, image_ids=GALLERY)
    queries = (
        Query("q1", 0, "caption", "a.jpg", ("b.jpg",), None, ("full", "sf")),
        Query("q3", 2, "caption", "a.jpg", ("c.jpg",), "shirt", ("full", "sf")),
        Query("q2", 1, "other", "b.jpg", ("d.jpg",), "shirt", ("full", "validated")),
    )
    names = {name: name for name in GALLERY}
    return SimpleNamespace(gallery=gallery, queries=queries, candidate_names=names)


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


@pytest.fixture()
def published(tmp_path: Path, monkeypatch) -> dict:
    """Run the REAL publisher into a temporary root; return the derived records it wrote."""
    root = tmp_path
    artifacts = root / "workbench/artifacts"
    composition_root = artifacts / "composition"
    # q1 is unscored WITH a parent reason; q3 is unscored with NO parent reason (the sentinel
    # case); q2 is scored and belongs to the OTHER subset (it must not appear at all).
    ranks = {condition: {"q1": None, "q3": None, "q2": 2} for condition in CONDITIONS}
    reasons = {condition: {"q1": UNSCORED_REASON} for condition in CONDITIONS}
    parent_summary = {
        "execution_status": "PARTIAL", "run_id": PARENT, "conditions": list(CONDITIONS),
        "model_id": MODEL, "checkpoint_id": CHECKPOINT, "subset": "full", "protocol_id": PROTOCOL,
        "adapter_query_scope_resolution": {"unscoped_qids": ["q1", "q3"], "still_unscoped": 2},
        "category_scope_failures": {condition: 2 for condition in CONDITIONS},
        "category_resolution": {"ambiguous_qids": ["q1", "q3"], "ambiguous_examples": []},
        "metrics": asdict(compose(ranks, query_count=3)),
        "topk_export": {"requested": True, "depth": 100, "conditions": list(CONDITIONS),
                        "partial": False, "status": "TOPK_EXPORT_COMPLETE"},
    }
    _write(composition_root / "runs" / PARENT / "ranks" / f"{MODEL}_{CHECKPOINT}.json",
           {"ranks": ranks, "query_failures": reasons, "identity": {}})
    _write(composition_root / "summary" / f"{MODEL}_{CHECKPOINT}_{PARENT}" / "full.json",
           parent_summary)
    for condition in CONDITIONS:
        rows = [{"qid": qid, "failure": reasons[condition].get(qid)}
                for qid in ("q1", "q3", "q2")]
        path = (composition_root / "results" / PROTOCOL / MODEL / CHECKPOINT / PARENT / "full" /
                f"{condition}.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    spec = importlib.util.spec_from_file_location("publish_derived_runs_r09", PUBLISHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.REPO = root
    module.ARTIFACTS = artifacts
    module.COMPOSITION = composition_root
    module.EVIDENCE = root / "evidence"
    module.PARENT = PARENT
    module.MODEL = MODEL
    module.CHECKPOINT = CHECKPOINT
    module.PROTOCOL = PROTOCOL
    module.EXPECTED_QUERY_COUNTS = {"sf": 2, "validated": 1}
    module.resolve_config = lambda: SimpleNamespace(
        WORKBENCH_CIRCUS_RELEASE_ROOT=root, FASHIONIQ_ROOT=root)
    module.load_release = lambda _: ReleaseData(
        Gallery(PROTOCOL, ("a.jpg", "c.jpg")), tuple(_release().queries),
        {"a.jpg": "a.jpg", "b.jpg": "b.jpg", "c.jpg": "c.jpg", "d.jpg": "d.jpg",
         "e.jpg": "e.jpg"}, {})
    module.assign_garment_categories = lambda data, _: data
    module.select_queries = lambda data, subset: tuple(
        query for query in data.queries if subset in query.subsets)
    assert module.main() == 0

    records = {}
    for path in sorted((composition_root / "runs").glob("*/ranks/*.json")):
        if path.parents[1].name == PARENT:
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        records[record["derived_from"]["subset"]] = record
    return {"root": artifacts, "records": records, "ranks": ranks, "reasons": reasons}


def _page(artifacts: Path, run_id: str, subset: str, monkeypatch) -> dict:
    monkeypatch.setattr(composition, "load_cached_release", lambda *a, **k: _release())
    return composition.get_composition_run_queries(run_id, subset=subset,
                                                   artifacts_root=artifacts)


def _export_parent_candidates(artifacts: Path) -> None:
    writers = topk.open_export(artifacts, PARENT, attempt=PARENT, model_id=MODEL,
                               checkpoint_id=CHECKPOINT, protocol_id=PROTOCOL,
                               gallery_identity=topk.gallery_digest(
                                   PROTOCOL, GALLERY, {name: name for name in GALLERY}),
                               gallery_count=len(GALLERY), depth=3, conditions=["multimodal"])
    writers["multimodal"].write("q1", None, [(4, 0.9), (0, 0.5)])
    writers["multimodal"].write("q3", None, [(2, 0.8)])
    topk.close_export(writers, expected_qids={"multimodal": ["q1", "q3"]})


def test_failure_map_is_carried_by_membership(published, monkeypatch):
    """Every qid IN the subset carries the parent's own reason; a qid outside never appears."""
    record = published["records"]["sf"]
    # q1 carries the parent's OWN text; q3 is unscored with no parent reason and gets the named
    # sentinel; q2 is scored and lives in the other subset, so it must not appear anywhere.
    for condition in CONDITIONS:
        table = record["query_failures"][condition]
        assert table["q1"] == UNSCORED_REASON
        assert table["q3"] == "parent_recorded_no_reason"
        assert "q2" not in table
        assert set(table) <= set(record["ranks"][condition])

    page = _page(published["root"], record["attempt"], "sf", monkeypatch)
    by_id = {item["query_id"]: item for item in page["items"]}
    assert by_id["q1"]["failure"] == UNSCORED_REASON
    assert by_id["q1"]["unscored_kind"] == "category_unresolved"
    assert by_id["q1"]["query_state"] == "unscored"


def test_unscored_qid_without_a_parent_reason_says_so(published, monkeypatch):
    """`parent_recorded_no_reason` is distinguishable from "the reason was not carried"."""
    record = published["records"]["sf"]
    page = _page(published["root"], record["attempt"], "sf", monkeypatch)
    by_id = {item["query_id"]: item for item in page["items"]}
    assert by_id["q3"]["target_rank"] is None
    assert by_id["q3"]["failure"] == "parent_recorded_no_reason"
    # ...and the query that DID have a parent reason is not given the sentinel.
    assert by_id["q1"]["failure"] != "parent_recorded_no_reason"


def test_reader_serves_the_parents_candidates_with_the_parents_status(published, monkeypatch):
    """The window comes from the parent run, labelled with its id and its OWN export status."""
    _export_parent_candidates(published["root"])
    record = published["records"]["sf"]
    page = _page(published["root"], record["attempt"], "sf", monkeypatch)
    assert page["candidates"]["status"] == "COMPLETE"
    assert page["candidates"]["source"] == "parent_run"
    assert page["candidates"]["source_run_id"] == PARENT
    assert page["candidates"]["parent_run_id"] == PARENT
    assert page["candidates"]["depth"] == 3
    assert page["candidates"]["lineage_verified"] is True

    by_id = {item["query_id"]: item for item in page["items"]}
    assert by_id["q1"]["candidate_status"] == "COMPLETE"
    assert by_id["q1"]["candidate_source_run_id"] == PARENT
    assert by_id["q1"]["candidate_source"] == "parent_run"
    assert [row["gallery_position"] for row in by_id["q1"]["top_results"]] == [4, 0]


def test_reader_refuses_when_the_parent_digest_does_not_match(published, monkeypatch):
    """A broken lineage is a NAMED refusal, never an empty window that reads as "not there"."""
    _export_parent_candidates(published["root"])
    record = published["records"]["sf"]
    record["derived_from"] = dict(record["derived_from"],
                                 parent_rank_artifact_sha256="0" * 64)
    broken = published["root"] / "composition/runs/brokenlineage/ranks/limn.json"
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_text(json.dumps(record), encoding="utf-8")

    page = _page(published["root"], "brokenlineage", "sf", monkeypatch)
    assert page["candidates"]["status"] == "parent_topk_unavailable"
    assert page["candidates"]["lineage_verified"] is False
    assert "does not match" in page["candidates"]["reason"]
    by_id = {item["query_id"]: item for item in page["items"]}
    assert by_id["q1"]["candidate_status"] == "parent_topk_unavailable"
    assert by_id["q1"]["top_results"] == []


def test_metrics_and_denominators_do_not_move(published, monkeypatch):
    """Carrying lineage fields may not move a single number in either derived run."""
    summary_dir = published["root"] / "composition/summary"
    for subset, record in published["records"].items():
        run_id = record["attempt"]
        summary = json.loads(next((summary_dir).glob(f"*_{run_id}/{subset}.json"))
                             .read_text(encoding="utf-8"))
        membership = record["derived_from"]["membership_count"]
        page = _page(published["root"], run_id, subset, monkeypatch)
        for label in ("multimodal", "text_only", "image_only"):
            condition = summary["metrics"][label]["condition"]
            statistics = page["statistics"]["conditions"][condition]
            assert statistics["denominator"] == membership
            assert statistics["denominator"] == summary["metrics"][label]["query_count"]
            assert statistics["coverage"] == summary["metrics"][label]["coverage"]
        assert summary["metrics"]["multimodal"]["coverage"] == record["derived_from"]["coverage"]
        assert (summary["metrics"]["multimodal"]["scored_count"] ==
                record["derived_from"]["denominator"])


def test_reader_reports_the_parents_own_export_status(published, monkeypatch):
    """NOT_REQUESTED / NOT_EXPORTED / PARTIAL / COMPLETE are four facts, not one collapse."""
    record = published["records"]["sf"]
    # (a) the parent's summary says the export WAS requested, but no export exists on disk.
    page = _page(published["root"], record["attempt"], "sf", monkeypatch)
    assert page["candidates"]["status"] == "NOT_EXPORTED"

    # (b) the parent's summary says it never requested one.
    summary_path = (published["root"] / "composition/runs" / PARENT / "ranks" /
                    f"{MODEL}_{CHECKPOINT}.json")
    assert summary_path.is_file()
    record["derived_from"] = dict(record["derived_from"], parent_topk_export={
        "requested": False, "status": "TOPK_EXPORT_NOT_REQUESTED", "depth": None,
        "conditions": [], "partial": False})
    unrequested = published["root"] / "composition/runs/unrequested/ranks/limn.json"
    unrequested.parent.mkdir(parents=True, exist_ok=True)
    unrequested.write_text(json.dumps(record), encoding="utf-8")
    page = _page(published["root"], "unrequested", "sf", monkeypatch)
    assert page["candidates"]["status"] == "NOT_REQUESTED"

    # (c) the export exists but is short: PARTIAL, and the refusal says how short.
    writers = topk.open_export(published["root"], PARENT, attempt=PARENT, model_id=MODEL,
                               checkpoint_id=CHECKPOINT, protocol_id=PROTOCOL,
                               gallery_identity=topk.gallery_digest(
                                   PROTOCOL, GALLERY, {name: name for name in GALLERY}),
                               gallery_count=len(GALLERY), depth=3, conditions=["multimodal"])
    writers["multimodal"].write("q1", None, [(4, 0.9)])
    topk.close_export(writers, expected_qids={"multimodal": ["q1", "q3"]})
    page = _page(published["root"], "unrequested", "sf", monkeypatch)
    assert page["candidates"]["status"] == "PARTIAL"
    assert "partial" in page["candidates"]["reason"]

def test_on_disk_derived_runs_carry_the_failure_map(published, monkeypatch):
    """The published runs: sf has a parent reason to carry, validated has none to invent."""
    composition_root = REPO / "workbench/artifacts/composition"
    artifacts = REPO / "workbench/artifacts"
    record = json.loads((composition_root / "runs" / "6a96c5c199526b61c1470b0e" / "ranks" /
                         f"{MODEL}_{CHECKPOINT}.json").read_text(encoding="utf-8"))
    assert record["query_failures"]["multimodal"]["7:3790"].startswith("query 7:3790:")
    assert all(set(table) <= set(record["ranks"][condition])
               for condition, table in record["query_failures"].items())
    assert record["derived_from"]["parent_topk_export"]["requested"] is True

    page = composition.get_composition_run_queries("6a96c5c199526b61c1470b0e", subset="sf",
                                                   artifacts_root=artifacts, query_id="7:3790")
    item = page["items"][0]
    assert item["unscored"] is True
    assert item["unscored_kind"] == "category_unresolved"
    assert item["failure"].startswith("query 7:3790:")
    assert page["candidates"]["source"] == "parent_run"
    assert page["candidates"]["parent_run_id"] == "9e4918a6b5c243412cbc58c9"

def test_on_disk_derived_runs_still_re_aggregate_to_their_published_metrics():
    """The two published derived runs: compose(ranks) equals the summary, unchanged by R09."""
    composition_root = REPO / "workbench/artifacts/composition"
    for run_id, subset in (("6a96c5c199526b61c1470b0e", "sf"),
                           ("aee2e8d39214d877a84d7c08", "validated")):
        record = json.loads((composition_root / "runs" / run_id / "ranks" /
                             f"{MODEL}_{CHECKPOINT}.json").read_text(encoding="utf-8"))
        summary = json.loads((composition_root / "summary" /
                              f"{MODEL}_{CHECKPOINT}_{run_id}" / f"{subset}.json")
                             .read_text(encoding="utf-8"))
        recomposed = asdict(compose(record["ranks"],
                                    query_count=record["derived_from"]["membership_count"]))
        for label in ("multimodal", "text_only", "image_only"):
            assert recomposed[label]["query_count"] == summary["metrics"][label]["query_count"]
            assert recomposed[label]["coverage"] == summary["metrics"][label]["coverage"]
            assert recomposed[label]["recall_at_10"] == summary["metrics"][label]["recall_at_10"]
            assert recomposed[label]["scored_count"] == summary["metrics"][label]["scored_count"]
        assert summary["derived_from"]["denominator"] == recomposed["multimodal"]["scored_count"]
