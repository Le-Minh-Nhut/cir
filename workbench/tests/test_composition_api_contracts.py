"""The three API defects this pass fixes, each asserted against the artifacts it describes.

1. `statistics.query_count` served the release's SUBSET SIZE beside per-condition denominators
   scoped to the queries actually ranked, so a 5-query smoke read `query_count: 6003` next to
   `denominator: 5` (`raw/verify/97-api-surface-tgcir-run.md` §7a).
2. `subset_provenance` appeared on the run-detail route only if the LIST route had already warmed a
   shared, mutable summary cache (`workbench/backend/composition.py` `read_summary_file`), so the same
   request answered 80 104 B or 80 927 B depending on an EARLIER request in the same process
   (§7b). Both call orders must now be byte-identical.
3. R@1 (`recall_at_1`) was added to the scorer after these runs finished; it is served from a derived
   record whose `derived_from.ranks.sha256` is verified on every read, and a stale record is refused.

Plus the fourth display condition: an adapter's named ablation (`clvc_image_encoder_only`) must render
as its own labelled row instead of being dropped, and the gap's own condition triple must be visible.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from workbench.backend import composition as C

CONDITIONS = ("multimodal", "text_only_black_image", "image_only_empty_text")


def _summary(root: Path, run_id: str, **overrides) -> Path:
    payload = {
        "schema_version": 1, "model_id": "m", "checkpoint_id": "c", "run_id": run_id,
        "subset": "full", "protocol_id": "circus_fashioniq_release_v1",
        "conditions": list(CONDITIONS), "query_count": 2, "gallery_count": 4,
        "execution_status": "COMPLETED",
        "metrics": {"coverage": 1.0, "gap_cohort_count": 2,
                    # No `recall_at_1`: the published artifacts predate the metric, which is exactly
                    # the condition the derived record exists for.
                    "multimodal": {"ndcg_full": 0.5, "recall_at_10": 50.0,
                                   "recall_at_50": 50.0, "mrr": 0.5, "query_count": 2,
                                   "scored_count": 2, "coverage": 1.0}},
    }
    payload.update(overrides)
    path = root / "composition" / "summary" / f"m_c_{run_id}" / "full.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _ranks(root: Path, run_id: str, ranks: dict, attempt: str | None = None) -> Path:
    path = root / "composition" / "runs" / run_id / "ranks" / "m_c.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"attempt": attempt or run_id, "ranks": ranks,
                                "query_failures": {}}), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _fresh_caches():
    C._SUMMARY_FILE_CACHE.clear()
    C._RANKS_CACHE.clear()
    C._DERIVED_FILE_CACHE.clear()
    yield
    C._SUMMARY_FILE_CACHE.clear()
    C._RANKS_FILE_CACHE_CLEAR = True


def test_statistics_query_count_describes_the_queries_actually_ranked(tmp_path, monkeypatch):
    """A smoke that ranked 5 of 6003 must say `query_count: 5`, not 6003.

    The release is stubbed with 6003 queries in `full` and only 5 of them present in the run's rank
    table, which is exactly the shape of `1eee4c0adbc93f7708205a5e`: the artifact ranks 5 queries
    while the release's `full` subset holds the whole cohort.
    """
    root = tmp_path
    _summary(root, "smoke1")

    class _Query:
        def __init__(self, qid: str) -> None:
            self.qid = qid
            self.subsets = ("full",)
            self.category = "dress"
            self.query_txt = "clip"
            self.reference_id = 0
            self.positive_ids = [1]

    class _Release:
        queries = [_Query(f"7:{index}") for index in range(1, 6004)]
        candidate_names = {0: "ref.jpg", 1: "target.jpg"}

    ranks = {condition: {f"7:{index}": index for index in range(1, 6)} for condition in CONDITIONS}
    _ranks(root, "smoke1", ranks)
    monkeypatch.setattr(C, "load_cached_release", lambda *a, **k: _Release())
    monkeypatch.setattr("workbench.composition.topk.read_candidates",
                        lambda *a, **k: {"available": False, "status": "TOPK_EXPORT_NOT_REQUESTED",
                                         "reason": None, "rows": {}, "manifest": None})

    page = C.get_composition_run_queries("smoke1", subset="full", artifacts_root=root, limit=1)
    statistics = page["statistics"]
    assert statistics["query_count"] == 5, statistics
    for condition, values in statistics["conditions"].items():
        assert values["denominator"] == 5, (condition, values)
        assert statistics["query_count"] >= values["denominator"], (
            "query_count must not be smaller than a denominator it is meant to describe")
    # The whole-run block agrees, which is the contradiction the defect produced on screen.
    assert page["statistics_whole_run"]["query_count"] == statistics["query_count"]


def test_a_full_run_still_reports_its_whole_cohort(tmp_path, monkeypatch):
    """The same rule on a Full run: `query_count` equals the ranked cohort, not a hard-coded number.
    """
    root = tmp_path
    _summary(root, "full1", query_count=3)

    class _Query:
        def __init__(self, qid: str) -> None:
            self.qid = qid
            self.subsets = ("full",)
            self.category = "dress"
            self.query_txt = "clip"
            self.reference_id = 0
            self.positive_ids = [1]

    class _Release:
        queries = [_Query(f"7:{index}") for index in range(1, 4)]
        candidate_names = {0: "ref.jpg", 1: "target.jpg"}

    ranks = {condition: {f"7:{index}": index for index in range(1, 4)} for condition in CONDITIONS}
    _ranks(root, "full1", ranks)
    monkeypatch.setattr(C, "load_cached_release", lambda *a, **k: _Release())
    monkeypatch.setattr("workbench.composition.topk.read_candidates",
                        lambda *a, **k: {"available": False, "status": "TOPK_EXPORT_NOT_REQUESTED",
                                         "reason": None, "rows": {}, "manifest": None})
    page = C.get_composition_run_queries("full1", subset="full", artifacts_root=root, limit=1)
    assert page["statistics"]["query_count"] == 3
    assert page["statistics_whole_run"]["query_count"] == 3


def _digest(payload: dict) -> tuple[int, str]:
    encoded = json.dumps(payload, sort_keys=True).encode()
    return len(encoded), hashlib.sha256(encoded).hexdigest()


def test_detail_is_byte_identical_in_both_call_orders(tmp_path):
    """`subset_provenance` must not depend on whether the list route ran first.

    Reproduced from the live API: the cold detail response had no `subset_provenance` key and the
    warm one did, because `read_summary_file` handed the SAME dict to every caller while
    `list_composition_summaries` decorated it in place. Both orders are asserted, and the comparison
    is bytes + sha256 rather than the key's presence -- a response that differs in any other field
    would be the same defect.
    """
    root = tmp_path
    _summary(root, "order1")
    _ranks(root, "order1", {condition: {"7:1": 1, "7:2": 2} for condition in CONDITIONS})

    # Order A: cold detail -> list -> detail.
    C._SUMMARY_FILE_CACHE.clear()
    first = C.get_composition_run_detail("order1", "full", root)
    assert first is not None
    C.list_composition_summaries(artifacts_root=root)
    second = C.get_composition_run_detail("order1", "full", root)
    assert second is not None
    assert _digest(first) == _digest(second), (
        "the run-detail response changed after the list route ran: "
        f"{_digest(first)} vs {_digest(second)}")

    # Order B: list first -> detail.
    C._SUMMARY_FILE_CACHE.clear()
    C.list_composition_summaries(artifacts_root=root)
    third = C.get_composition_run_detail("order1", "full", root)
    assert third is not None
    assert _digest(first) == _digest(third), (
        "the run-detail response depends on which route warmed the cache: "
        f"{_digest(first)} vs {_digest(third)}")


def test_a_stale_derived_record_is_refused(tmp_path):
    """A derived number that outlives its source artifact is worse than no number."""
    root = tmp_path
    summary = _summary(root, "derived1")
    # The buckets name the condition that filled each ROLE, exactly as a real summary does. The
    # derived record keys by CONDITION NAME, so a value has to be routed through that name -- the
    # first cut of this overlay looked the record's key up directly and silently served nothing for
    # every bucket whose role name differs from its condition name, which is two of the three.
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["metrics"]["text_only"] = {"condition": "text_only_black_image", "ndcg_full": 0.4}
    payload["metrics"]["image_only"] = {"condition": "image_only_empty_text", "ndcg_full": None}
    summary.write_text(json.dumps(payload), encoding="utf-8")
    ranks_path = _ranks(root, "derived1", {condition: {"7:1": 1, "7:2": 2} for condition in CONDITIONS})
    digest = hashlib.sha256(ranks_path.read_bytes()).hexdigest()

    def write_derived(claimed: str) -> None:
        path = root / "composition" / "derived" / "recall_at_1" / "m_full.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"runs": [{
            "run_id": "derived1", "attempt": "derived1", "subset": "full", "metric": "recall_at_1",
            "conditions": {condition: {"recall_at_1": 50.0, "unavailable_reason": None}
                           for condition in CONDITIONS},
            "derived_from": {"ranks": {"path": str(ranks_path), "sha256": claimed}},
        }]}), encoding="utf-8")
        C._DERIVED_FILE_CACHE.clear()

    write_derived(digest)
    served = C.get_composition_run_detail("derived1", "full", root)
    assert served is not None
    assert served["derived_metrics"]["recall_at_1"]["status"] == "VERIFIED"
    assert served["metrics"]["recall_at_1_multimodal"] == 50.0
    assert served["metrics"]["multimodal"]["recall_at_1"] == 50.0
    # Every role bucket is filled, not just the one whose key happens to equal its role name.
    assert served["metrics"]["text_only"]["recall_at_1"] == 50.0
    assert served["metrics"]["image_only"]["recall_at_1"] == 50.0

    # The artifact is replaced, so the record now describes a table that no longer exists.
    C._RANKS_CACHE.clear()
    ranked = json.loads(ranks_path.read_text(encoding="utf-8"))
    ranked["ranks"]["multimodal"]["7:3"] = 3
    ranks_path.write_text(json.dumps(ranked), encoding="utf-8")
    refreshed = C.get_composition_run_detail("derived1", "full", root)
    assert refreshed is not None
    block = refreshed["derived_metrics"]["recall_at_1"]
    assert block["status"] == "REFUSED", block
    assert "changed since" in (block["reason"] or ""), block
    assert "recall_at_1" not in (refreshed["metrics"]["multimodal"] or {}), (
        "a refused derived record must not leave its number behind")


def test_an_ablation_condition_renders_its_own_labelled_row(tmp_path, monkeypatch):
    """A `clvc_image_encoder_only` run shows that condition instead of silently dropping it.

    `CONDITIONS` keeps its meaning: the served per-query `ranks` map is the protocol triple PLUS the
    ablation the run actually ranked, and the page payload names the label for each.
    """
    root = tmp_path
    summary = _summary(root, "ablate1")
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["condition_status"] = {**{condition: {"status": "COMPLETED"} for condition in CONDITIONS},
                                   "clvc_image_encoder_only": {"status": "COMPLETED"}}
    payload["conditions"] = [*CONDITIONS, "clvc_image_encoder_only"]
    payload["metrics"]["gap_conditions"] = ["multimodal", "text_only_black_image",
                                            "clvc_image_encoder_only"]
    payload["metrics"]["gap_role_reason"] = (
        "image role filled by the adapter's named ablation 'clvc_image_encoder_only'")
    summary.write_text(json.dumps(payload), encoding="utf-8")
    ranks = {condition: {"7:1": 1} for condition in CONDITIONS}
    ranks["clvc_image_encoder_only"] = {"7:1": 2}
    _ranks(root, "ablate1", ranks)

    detail = C.get_composition_run_detail("ablate1", "full", root)
    assert detail is not None
    rendered = [item["id"] for item in detail["conditions_display"]]
    assert "clvc_image_encoder_only" in rendered, rendered
    labels = {item["id"]: item["label"] for item in detail["conditions_display"]}
    assert labels["clvc_image_encoder_only"] == "Image only (image encoder)", labels
    assert detail["gap_conditions"] == ["multimodal", "text_only_black_image",
                                        "clvc_image_encoder_only"]
    assert "named ablation" in (detail["gap_role_reason"] or "")

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
    page = C.get_composition_run_queries("ablate1", subset="full", artifacts_root=root)
    served = {item["id"]: item["label"] for item in page["conditions"]}
    assert served.get("clvc_image_encoder_only") == "Image only (image encoder)", served
    assert page["items"][0]["ranks"].get("clvc_image_encoder_only") == 2


def test_an_artifact_recorded_recall_at_1_is_never_overwritten(tmp_path):
    """The artifact wins. A published `recall_at_1` is what the run actually did."""
    root = tmp_path
    summary = _summary(root, "native1")
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["metrics"]["multimodal"]["recall_at_1"] = 42.0
    summary.write_text(json.dumps(payload), encoding="utf-8")
    _ranks(root, "native1", {condition: {"7:1": 1} for condition in CONDITIONS})
    detail = C.get_composition_run_detail("native1", "full", root)
    assert detail is not None
    assert detail["metrics"]["multimodal"]["recall_at_1"] == 42.0
    assert detail["metrics"]["recall_at_1_multimodal"] == 42.0


def test_a_run_without_a_derived_record_or_ablation_is_unchanged(tmp_path):
    """Absent inputs must behave exactly as before: no key invented, no row added."""
    root = tmp_path
    _summary(root, "plain1")
    _ranks(root, "plain1", {condition: {"7:1": 1} for condition in CONDITIONS})
    detail = C.get_composition_run_detail("plain1", "full", root)
    assert detail is not None
    assert detail["derived_metrics"]["recall_at_1"]["status"] == "MISSING"
    assert [item["id"] for item in detail["conditions_display"]] == list(CONDITIONS)
    assert "gap_conditions" not in detail
    assert detail["metrics"].get("recall_at_1_multimodal") is None
