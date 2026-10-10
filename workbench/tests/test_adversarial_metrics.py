"""Adversarial metric validation tests (V01-V15) for LIMN and CSMCIR aggregate extraction."""
from __future__ import annotations

import json

import workbench.backend.metrics_extraction as mx


def _limn(dress=60.0, shirt=55.0, toptee=50.0, d50=80.0, s50=75.0, t50=70.0, declared=None, complete=True, extra=None):
    m10 = (dress + shirt + toptee) / 3
    m50 = (d50 + s50 + t50) / 3
    agg = declared or {"macro_r10": m10, "macro_r50": m50, "macro_mean": (m10 + m50) / 2, "complete": complete}
    cats = [
        {"category": "dress", "r10": dress, "r50": d50},
        {"category": "shirt", "r10": shirt, "r50": s50},
        {"category": "toptee", "r10": toptee, "r50": t50},
    ]
    if extra:
        cats.extend(extra)
    return json.dumps({"model_id": "limn", "aggregate": agg, "categories": cats})


# V01: Correct category macro accepted
def test_v01_limn_correct_macro_accepted():
    r = mx.parse_limn_output(_limn())
    assert r.extraction_status == "EXTRACTED_VERIFIED"
    assert r.observed_metrics == {"r10": 55.0, "r50": 75.0, "mean": 65.0}


# V02: Fabricated declared aggregate rejected
def test_v02_limn_fabricated_aggregate_rejected():
    declared = {"macro_r10": 99.0, "macro_r50": 99.0, "macro_mean": 99.0, "complete": True}
    r = mx.parse_limn_output(_limn(declared=declared))
    assert r.extraction_status == "EVALUATION_COMPLETED_METRICS_UNPARSED"
    assert r.observed_metrics is None


# V03: Missing category rejected
def test_v03_limn_missing_category_rejected():
    payload = json.dumps({"model_id": "limn", "aggregate": {"macro_r10": 50, "macro_r50": 70, "macro_mean": 60, "complete": True},
                          "categories": [{"category": "dress", "r10": 40, "r50": 60}, {"category": "shirt", "r10": 50, "r50": 70}]})
    assert mx.parse_limn_output(payload).observed_metrics is None


# V04: Duplicate category rejected
def test_v04_limn_duplicate_category_rejected():
    payload = _limn(extra=[{"category": "dress", "r10": 40, "r50": 60}])
    assert mx.parse_limn_output(payload).observed_metrics is None


# V05: Missing R@50 rejected
def test_v05_limn_missing_r50_rejected():
    payload = json.dumps({"model_id": "limn", "aggregate": {"macro_r10": 50, "macro_r50": 70, "macro_mean": 60, "complete": True},
                          "categories": [{"category": "dress", "r10": 40, "r50": None},
                                         {"category": "shirt", "r10": 50, "r50": 70},
                                         {"category": "toptee", "r10": 60, "r50": 80}]})
    assert mx.parse_limn_output(payload).observed_metrics is None


# V06: NaN rejected
def test_v06_limn_nan_rejected():
    payload = _limn(dress=float("nan"))
    assert mx.parse_limn_output(payload).observed_metrics is None


# V07: Infinity rejected
def test_v07_limn_infinity_rejected():
    assert mx.parse_limn_output(_limn(dress=float("inf"))).observed_metrics is None
    assert mx.parse_limn_output(_limn(shirt=float("-inf"))).observed_metrics is None


# V08: Negative recall rejected
def test_v08_limn_negative_rejected():
    assert mx.parse_limn_output(_limn(dress=-1.0)).observed_metrics is None


# V09: Recall > 100 rejected
def test_v09_limn_over_100_rejected():
    assert mx.parse_limn_output(_limn(dress=120.0)).observed_metrics is None


# V10: Boolean metrics rejected
def test_v10_limn_boolean_rejected():
    payload = json.dumps({"model_id": "limn", "aggregate": {"macro_r10": 50, "macro_r50": 70, "macro_mean": 60, "complete": True},
                          "categories": [{"category": "dress", "r10": True, "r50": 60},
                                         {"category": "shirt", "r10": 50, "r50": 70},
                                         {"category": "toptee", "r10": 60, "r50": 80}]})
    assert mx.parse_limn_output(payload).observed_metrics is None


# V11: Inconsistent combined mean rejected
def test_v11_limn_inconsistent_mean_rejected():
    declared = {"macro_r10": 55.0, "macro_r50": 75.0, "macro_mean": 71.0, "complete": True}
    assert mx.parse_limn_output(_limn(declared=declared)).observed_metrics is None


# V12: Missing parser never fabricates metrics
def test_v12_missing_parser_no_fabrication():
    r = mx.extract_aggregate_metrics("airknow", "irrelevant output", "", 0)
    assert r.extraction_status == "AGGREGATE_PARSER_UNAVAILABLE"
    assert r.observed_metrics is None


# V13: Paper score never substituted for observed
def test_v13_paper_not_substituted():
    paper = {"r10": 55.0, "r50": 75.0, "mean": 65.0}
    r = mx.extract_aggregate_metrics("airknow", "", "", 0, paper_metrics=paper)
    assert r.observed_metrics is None


# V14: CSMCIR inconsistent category macro rejected
def test_v14_csmcir_inconsistent_macro_rejected():
    payload = json.dumps({
        "dress_recall_at10": 60.0, "dress_recall_at50": 80.0,
        "shirt_recall_at10": 55.0, "shirt_recall_at50": 75.0,
        "toptee_recall_at10": 50.0, "toptee_recall_at50": 70.0,
        "average_recall_at10": 99.0, "average_recall_at50": 75.0, "average_recall": 87.0,
    })
    assert mx.parse_csmcir_output(payload).observed_metrics is None


# V14b: CSMCIR consistent macro accepted
def test_v14b_csmcir_consistent_accepted():
    payload = json.dumps({
        "dress_recall_at10": 60.0, "dress_recall_at50": 80.0,
        "shirt_recall_at10": 55.0, "shirt_recall_at50": 75.0,
        "toptee_recall_at10": 50.0, "toptee_recall_at50": 70.0,
        "average_recall_at10": 55.0, "average_recall_at50": 75.0, "average_recall": 65.0,
    })
    r = mx.parse_csmcir_output(payload)
    assert r.extraction_status == "EXTRACTED_VERIFIED"
    assert r.observed_metrics == {"r10": 55.0, "r50": 75.0, "mean": 65.0}


# V15: Ambiguous duplicate output rejected
def test_v15_ambiguous_duplicate_rejected():
    good = json.dumps({"average_recall_at10": 55.0, "average_recall_at50": 75.0, "average_recall": 65.0,
                       "dress_recall_at10": 60.0, "dress_recall_at50": 80.0,
                       "shirt_recall_at10": 55.0, "shirt_recall_at50": 75.0,
                       "toptee_recall_at10": 50.0, "toptee_recall_at50": 70.0})
    bad = json.dumps({"average_recall_at10": 99.0, "average_recall_at50": 75.0, "average_recall": 87.0,
                      "dress_recall_at10": 60.0, "dress_recall_at50": 80.0,
                      "shirt_recall_at10": 55.0, "shirt_recall_at50": 75.0,
                      "toptee_recall_at10": 50.0, "toptee_recall_at50": 70.0})
    assert mx.parse_csmcir_output(good + "\n" + bad).observed_metrics is None


# Extra: malformed/truncated output never accepted
def test_malformed_output_never_accepted():
    for bad in ("{not json", "", "random log line", '{"model_id": "limn"}'):
        assert mx.parse_limn_output(bad).observed_metrics is None
        assert mx.parse_csmcir_output(bad).observed_metrics is None


# Extra: failed return code yields EVALUATION_FAILED
def test_failed_return_code_marked_failed():
    r = mx.extract_aggregate_metrics("limn", _limn(), "", 3)
    assert r.extraction_status == "EVALUATION_FAILED"
    assert r.observed_metrics is None
