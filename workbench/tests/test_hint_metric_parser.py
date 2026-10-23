"""HINT aggregate metric parsing, for both real output shapes.

The pinned HINT evaluator has no `__main__`, so the workbench's replay wrapper prints the result and
this parser reads it. Before it existed, a valid HINT run with exit code 0 was reported
`AGGREGATE_PARSER_UNAVAILABLE`: the adapter could start a subprocess, but nothing could read what it
produced, and an adapter that can launch is not an integrated model.

Two output shapes are covered because two producers really exist:

* the wrapper's JSON document, and
* the evaluator's own `[('dress_r10', 20.0), ...]` print, which `test.py:103-108` returns and
  `train.py:199` prints verbatim.

The macro is recomputed from the three categories rather than trusted, so a truncated stream, a
misread field, or a producer that contradicts itself is rejected instead of reported as a measurement.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from workbench.backend.metrics_extraction import extract_aggregate_metrics, parse_hint_output  # noqa: E402

CATEGORIES = ("dress", "shirt", "toptee")
PER_CATEGORY_R10 = {"dress": 21.0, "shirt": 22.0, "toptee": 23.0}
PER_CATEGORY_R50 = {"dress": 41.0, "shirt": 42.0, "toptee": 43.0}
MACRO_R10 = sum(PER_CATEGORY_R10.values()) / 3
MACRO_R50 = sum(PER_CATEGORY_R50.values()) / 3
MACRO_MEAN = (MACRO_R10 + MACRO_R50) / 2


def wrapper_output(**overrides) -> str:
    document = {
        "model_id": "hint",
        "protocol": "fashioniq_val_split",
        "complete_benchmark": True,
        "categories": [{"category": c, "r1": 10.0, "r10": PER_CATEGORY_R10[c],
                        "r50": PER_CATEGORY_R50[c]} for c in CATEGORIES],
        "aggregate": {"macro_r10": MACRO_R10, "macro_r50": MACRO_R50, "macro_mean": MACRO_MEAN,
                      "complete": True},
    }
    document.update(overrides)
    return json.dumps(document, indent=2)


def native_output() -> str:
    return str([(f"{c}_r{k}", v) for c in CATEGORIES
                for k, v in ((1, 10.0), (10, PER_CATEGORY_R10[c]), (50, PER_CATEGORY_R50[c]))])


def test_a_valid_wrapper_result_is_parsed() -> None:
    result = extract_aggregate_metrics("hint", wrapper_output(), "", 0)
    assert result.extraction_status == "EXTRACTED_VERIFIED"
    assert result.parser_id == "hint_native_recall"
    assert result.observed_metrics["r10"] == pytest.approx(MACRO_R10, abs=1e-4)
    assert result.observed_metrics["r50"] == pytest.approx(MACRO_R50, abs=1e-4)
    assert result.observed_metrics["mean"] == pytest.approx(MACRO_MEAN, abs=1e-4)


def test_the_native_evaluator_print_is_parsed() -> None:
    """The evaluator's own tuple output must work without the wrapper in the loop."""
    result = extract_aggregate_metrics("hint", native_output(), "", 0)
    assert result.extraction_status == "EXTRACTED_VERIFIED"
    assert result.observed_metrics["mean"] == pytest.approx(MACRO_MEAN, abs=1e-4)


def test_all_three_categories_are_required() -> None:
    for missing in CATEGORIES:
        document = json.loads(wrapper_output())
        document["categories"] = [c for c in document["categories"] if c["category"] != missing]
        result = parse_hint_output(json.dumps(document))
        assert result.extraction_status != "EXTRACTED_VERIFIED", f"{missing} missing was accepted"


def test_recall_values_must_be_numeric_and_in_range() -> None:
    for bad in (150.0, -1.0, "21.0", None, float("nan")):
        document = json.loads(wrapper_output())
        document["categories"][0]["r10"] = bad
        result = parse_hint_output(json.dumps(document))
        assert result.extraction_status != "EXTRACTED_VERIFIED", f"{bad!r} was accepted"


def test_a_declared_macro_that_disagrees_with_its_categories_is_rejected() -> None:
    """The parser recomputes the macro; a producer contradicting itself is not a measurement."""
    document = json.loads(wrapper_output())
    document["aggregate"]["macro_r10"] = MACRO_R10 + 5.0
    result = parse_hint_output(json.dumps(document))
    assert result.extraction_status != "EXTRACTED_VERIFIED"


def test_malformed_and_empty_output_is_rejected() -> None:
    for text in ("", "not json at all", "{", json.dumps({"model_id": "hint"})):
        result = parse_hint_output(text)
        assert result.extraction_status == "EVALUATION_COMPLETED_METRICS_UNPARSED"
        assert result.observed_metrics is None


def test_a_failed_process_yields_no_metrics() -> None:
    result = extract_aggregate_metrics("hint", wrapper_output(), "", 1)
    assert result.extraction_status == "EVALUATION_FAILED"
    assert result.observed_metrics is None


def test_the_macro_is_recomputed_not_copied() -> None:
    """Independence check: change the categories only; the macro must follow them."""
    document = json.loads(wrapper_output())
    for entry in document["categories"]:
        entry["r10"] = 50.0
        entry["r50"] = 70.0
    document["aggregate"] = {"macro_r10": 50.0, "macro_r50": 70.0, "macro_mean": 60.0}
    result = parse_hint_output(json.dumps(document))
    assert result.extraction_status == "EXTRACTED_VERIFIED"
    assert result.observed_metrics == {"r10": 50.0, "r50": 70.0, "mean": 60.0}
    assert result.category_metrics["toptee"] == {"r10": 50.0, "r50": 70.0}


def test_a_result_embedded_in_other_output_is_found() -> None:
    """The wrapper may print progress lines around the document."""
    noisy = "loading processors...\nCompute FashionIQ val metrics\n" + wrapper_output() + "\ndone\n"
    result = parse_hint_output(noisy)
    assert result.extraction_status == "EXTRACTED_VERIFIED"


def test_paper_metrics_produce_a_parity_verdict_not_a_substitution() -> None:
    """Paper numbers are a comparison target; they never become the observed metric."""
    paper = {"r10": MACRO_R10, "r50": MACRO_R50, "mean": MACRO_MEAN}
    result = parse_hint_output(wrapper_output(), paper)
    assert result.extraction_status == "EXTRACTED_VERIFIED"
    assert result.observed_metrics["r10"] == pytest.approx(MACRO_R10, abs=1e-4)
    assert result.parity_status != "PARITY_NOT_EVALUATED"

    far = {"r10": MACRO_R10 + 40, "r50": MACRO_R50 + 40, "mean": MACRO_MEAN + 40}
    mismatched = parse_hint_output(wrapper_output(), far)
    assert mismatched.observed_metrics["r10"] == pytest.approx(MACRO_R10, abs=1e-4), (
        "the observed metric must stay the observed metric")


def test_adding_hint_did_not_disturb_the_existing_parsers() -> None:
    """A new dispatch branch must not change how an already-parsed model is read."""
    from workbench.backend.metrics_extraction import parse_limn_output

    # A realistic LIMN document: its parser requires all three categories with valid recalls AND
    # `aggregate.complete`, and reads it as a ONE-LINE document (what the LIMN wrapper prints).
    limn = json.dumps({
        "model_id": "limn",
        "complete_benchmark": True,
        "categories": [{"category": c, "r1": 10.0, "r10": 21.0, "r50": 41.0}
                       for c in CATEGORIES],
        "aggregate": {"macro_r10": 21.0, "macro_r50": 41.0, "macro_mean": 31.0, "complete": True},
    }, separators=(",", ":"))
    assert parse_limn_output(limn).extraction_status == "EXTRACTED_VERIFIED"
    assert extract_aggregate_metrics("limn", limn, "", 0).observed_metrics["mean"] == 31.0

    # And the LIMN parser does not silently accept HINT's output shape.
    assert parse_limn_output(wrapper_output()).extraction_status != "EXTRACTED_VERIFIED"

    # A model with no parser still reports that honestly rather than borrowing HINT's. The holder of
    # that state keeps moving as wrappers gain parsers (`pair`, then `dcnet`, then `ptha_mtst` when
    # its transfer contract got one), so this uses a model whose mapping is still unresolved: `tgcir`.
    other = extract_aggregate_metrics("tgcir", wrapper_output(), "", 0)
    assert other.extraction_status == "AGGREGATE_PARSER_UNAVAILABLE"
    assert other.parser_id is None
    # And a model that HAS a parser still must not borrow HINT's: unreadable output yields no metric.
    registered = extract_aggregate_metrics("ptha_mtst", wrapper_output(), "", 0)
    assert registered.observed_metrics is None
    assert registered.parser_id != "hint_structured_json"
