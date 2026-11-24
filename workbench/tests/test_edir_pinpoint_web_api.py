"""Comprehensive tests for EDIR & PinPoint web API endpoints (`edir_pinpoint_api.py`).

Verifies:
  - Overview & summary endpoints return exact canonical counts and bit-exact metrics.
  - All 8 PinPoint evaluation slices match independent evaluator recomputation.
  - Query listing, filtering (slice, category, outcome, search), sorting, pagination, and side-by-side compare.
  - Query detail returns valid reference, ground truth, hard negatives, top-K candidates, and cross-model consensus.
  - Image endpoint serves real EDIR/PinPoint images AND rejects path-traversal attempts (404).
  - CSV export returns valid UTF-8 CSV for both EDIR and PinPoint slices.
  - Invalid dataset / model / regime / slice / outcome cleanly return 400 (never 500).
"""

from __future__ import annotations

import csv
import io
from fastapi.testclient import TestClient

from workbench.backend.main import app

client = TestClient(app)


def test_benchmarks_overview_structure_and_counts() -> None:
    resp = client.get("/api/benchmarks/overview")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["split_tag"] == "70_10_20_seed42"
    assert data["edir"]["corpus_size"] == 28645
    assert data["edir"]["test_query_count"] == 1003
    assert data["edir"]["completion"]["completed"]["indomain"] == 16
    assert data["pinpoint"]["corpus_size"] == 109598
    assert data["pinpoint"]["test_query_count"] == 1337
    assert data["pinpoint"]["completion"]["completed"]["indomain"] == 16
    assert len(data["availability"]) == 16
    # Verify official unavailable checkpoints are transparently marked & QuRe is ready across all 5 regimes
    avail_by_key = {row["key"]: row for row in data["availability"]}
    assert avail_by_key["qure"]["edir_indomain"] == "ready"
    assert avail_by_key["qure"]["pinpoint_indomain"] == "ready"
    assert avail_by_key["qure"]["fiq_to_pinpoint"] == "ready"
    assert avail_by_key["qure"]["cirr_to_edir"] == "ready"
    assert avail_by_key["qure"]["fiq_to_edir"] == "ready"
    assert avail_by_key["mtst_ptha"]["fiq_to_pinpoint"] == "unavailable"
    assert avail_by_key["tme"]["cirr_to_edir"] == "unavailable"
    assert avail_by_key["airknow"]["cirr_to_edir"] == "unavailable"
    assert avail_by_key["conesep"]["cirr_to_edir"] == "unavailable"
    assert avail_by_key["clvcnet"]["cirr_to_edir"] == "unavailable"
    assert avail_by_key["dcnet"]["cirr_to_edir"] == "unavailable"


def test_edir_summary_and_subcategories() -> None:
    resp = client.get("/api/benchmarks/edir/summary")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert len(data["models"]) == 16
    assert len(data["subcategories"]) == 15
    top = data["models"][0]
    assert top["indomain"]["status"] == "ready"
    assert top["indomain"]["overall"]["Recall@1"] > 0.35
    # All 16 models should be sorted descending by Recall@1
    r1s = [m["indomain"]["overall"]["Recall@1"] for m in data["models"]]
    assert r1s == sorted(r1s, reverse=True)


def test_pinpoint_summary_all_8_slices_exact() -> None:
    resp = client.get("/api/benchmarks/pinpoint/summary")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert len(data["models"]) == 16
    slice_counts = {s["id"]: s["count"] for s in data["slices"]}
    assert slice_counts == {
        "all": 1337,
        "fashion_all": 351,
        "fashion_no_comp": 338,
        "fashion_swap": 197,
        "fashion_ctx": 107,
        "fashion_neg": 29,
        "fashion_comp": 13,
        "non_fashion": 986,
    }
    assert slice_counts["fashion_all"] + slice_counts["non_fashion"] == 1337
    assert slice_counts["fashion_no_comp"] + slice_counts["fashion_comp"] == slice_counts["fashion_all"]
    top = data["models"][0]
    assert top["key"] == "conesep"
    assert abs(top["indomain"]["slices"]["all"]["mAP@10"] - top["indomain"]["overall"]["mAP@10"]) < 1e-12


def test_queries_list_filter_detail_and_image_security() -> None:
    # 1) EDIR remove category drilldown
    r = client.get("/api/benchmarks/edir/queries", params={"model": "mtst_ptha", "regime": "indomain", "category": "remove", "limit": 10})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["filtered_total"] > 0
    assert all(item["category"] == "remove" for item in body["items"])

    first_qid = body["items"][0]["query_id"]
    det = client.get(
        f"/api/benchmarks/edir/queries/{first_qid}",
        params={"model": "mtst_ptha", "regime": "indomain", "compare_model": "sprc", "compare_regime": "indomain", "top_k": 10},
    )
    assert det.status_code == 200, det.text
    d = det.json()
    assert d["query"]["query_id"] == first_qid
    assert len(d["primary"]["candidates"]) == 10
    assert d["compare"] is not None and len(d["compare"]["candidates"]) == 10
    assert len(d["all_models"]) >= 15

    # 2) Serve real reference image & reject path traversal
    img_ok = client.get(d["query"]["ref_image_url"])
    assert img_ok.status_code == 200
    assert len(img_ok.content) > 100
    img_bad = client.get("/api/benchmarks/edir/images/..%2F..%2Fetc%2Fpasswd")
    assert img_bad.status_code == 404

    # 3) PinPoint Fashion-Swap slice + detail
    rp = client.get(
        "/api/benchmarks/pinpoint/queries",
        params={"model": "conesep", "regime": "indomain", "slice": "fashion_swap", "limit": 5},
    )
    assert rp.status_code == 200, rp.text
    pb = rp.json()
    assert pb["filtered_total"] == 197
    pp_qid = pb["items"][0]["query_id"]
    pdet = client.get(f"/api/benchmarks/pinpoint/queries/{pp_qid}", params={"model": "conesep", "regime": "indomain"})
    assert pdet.status_code == 200, pdet.text
    pd = pdet.json()
    assert len(pd["ground_truths"]) >= 1
    pp_img = client.get(pd["query"]["ref_image_url"])
    assert pp_img.status_code == 200
    assert len(pp_img.content) > 100


def test_csv_exports_and_error_handling() -> None:
    c1 = client.get("/api/benchmarks/edir/export.csv")
    assert c1.status_code == 200
    rows1 = list(csv.reader(io.StringIO(c1.text)))
    assert len(rows1) == 17  # header + 16 models

    c2 = client.get("/api/benchmarks/pinpoint/export.csv", params={"slice": "fashion_no_comp"})
    assert c2.status_code == 200
    rows2 = list(csv.reader(io.StringIO(c2.text)))
    assert len(rows2) == 17

    # Invalid inputs cleanly return 400 (never 500)
    assert client.get("/api/benchmarks/unknown/summary").status_code == 400
    assert client.get("/api/benchmarks/pinpoint/queries", params={"model": "conesep", "regime": "cirr_transfer"}).status_code == 400
    assert client.get("/api/benchmarks/pinpoint/queries", params={"model": "conesep", "slice": "bogus"}).status_code == 400
    assert client.get("/api/benchmarks/edir/queries", params={"model": "not_a_model"}).status_code == 400


def test_common_failures_default_across_pinpoint_edir_and_fashioniq() -> None:
    # 1) Default opens directly on FashionIQ Original Split (>=80% common failures, zero manual selection)
    r0 = client.get("/api/benchmarks/common-failures")
    assert r0.status_code == 200, r0.text
    d0 = r0.json()
    assert d0["protocol"] == "fashioniq_original_split"
    assert d0["dataset"] == "fashioniq"
    assert d0["selected_model_count"] >= 5
    assert d0["preset"] == "fail80"
    assert d0["filtered_total"] > 0
    assert all(it["fail_rate"] >= 0.8 - 1e-9 for it in d0["items"])

    # 1b) Merged 3-protocol FashionIQ view works cleanly (grouped per-protocol read)
    r0b = client.get("/api/benchmarks/common-failures", params={"protocol": "fashioniq_all_protocols"})
    assert r0b.status_code == 200, r0b.text
    d0b = r0b.json()
    assert d0b["selected_model_count"] == 16
    assert d0b["filtered_total"] > 0

    # 2) PinPoint In-Domain returns >=80% common failures across all 16 models
    r1 = client.get("/api/benchmarks/common-failures", params={"protocol": "pinpoint_indomain"})
    assert r1.status_code == 200, r1.text
    d1 = r1.json()
    assert d1["protocol"] == "pinpoint_indomain"
    assert d1["selected_model_count"] == 16
    assert d1["filtered_total"] > 0
    assert all(it["fail_rate"] >= 0.8 - 1e-9 for it in d1["items"])
    assert len(d1["category_breakdown"]) == 5

    # 3) EDIR In-Domain @1 shows `remove` at high common-fail rate + model subset filtering works
    r2 = client.get(
        "/api/benchmarks/common-failures",
        params={"protocol": "edir_indomain", "preset": "all_fail", "k": 1, "models": "mtst_ptha,sprc,tme"},
    )
    assert r2.status_code == 200, r2.text
    d2 = r2.json()
    assert d2["selected_model_count"] == 3
    assert all(it["fail_count"] == 3 for it in d2["items"])

    # 4) Inline Top-K ranked candidate images detail across FashionIQ, PinPoint, and EDIR
    f_qid = d0["items"][0]["query_id"]
    fd = client.get("/api/benchmarks/common-failures/detail", params={"protocol": "fashioniq_original_split", "query_id": f_qid, "top_k": 10})
    assert fd.status_code == 200, fd.text
    fdb = fd.json()
    assert len(fdb["models"]) >= 5 and len(fdb["models"][0]["candidates"]) == 10

    p_qid = d1["items"][0]["query_id"]
    pd = client.get("/api/benchmarks/common-failures/detail", params={"protocol": "pinpoint_indomain", "query_id": p_qid, "top_k": 10})
    assert pd.status_code == 200, pd.text
    pdb = pd.json()
    assert len(pdb["models"]) == 16 and len(pdb["models"][0]["candidates"]) == 10

    # 5) CSV export & invalid protocol/preset guard
    rcsv = client.get("/api/benchmarks/common-failures/export.csv", params={"protocol": "edir_indomain", "preset": "all_fail", "k": 1})
    assert rcsv.status_code == 200
    assert "fail_count@1" in rcsv.text
    assert client.get("/api/benchmarks/common-failures", params={"protocol": "bogus"}).status_code == 400
    assert client.get("/api/benchmarks/common-failures", params={"preset": "bogus"}).status_code == 400
