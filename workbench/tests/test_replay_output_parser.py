"""Every replay wrapper's output must be readable, and only when it is a complete result.

The eight models whose only runnable entry point is a wrapper — pair, encoder, clvc_net, intent,
habit, conesep, airknow — had an adapter that could start a process and a parser that could read
nothing, so a successful run with exit 0 was reported `AGGREGATE_PARSER_UNAVAILABLE`. A wrapper whose
output nobody can read is not an integrated model.

The tests below pin both directions. Acceptance must work for the shape the wrappers really print,
and rejection must hold for every way a document can claim to be a complete benchmark while not
being one: a missing category, a duplicate category, an incomplete category, a value outside 0..100,
another model's document, and a macro that contradicts the categories it summarises.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from workbench.backend.metrics_extraction import (  # noqa: E402
    extract_aggregate_metrics,
    parse_workbench_replay_output,
)

CATEGORIES = ("dress", "shirt", "toptee")
WRAPPER_MODELS = ("pair", "encoder", "clvc_net", "intent", "habit", "conesep", "airknow")


def replay_document(model_id: str, r10=(21.0, 22.0, 23.0), r50=(41.0, 42.0, 43.0),
                    complete=(True, True, True), **extra) -> str:
    """Exactly the shape `workbench/replay/*.py` prints.

    Built from the same `ReplayResult`/`macro_aggregate` the wrappers use rather than from a
    hand-written literal, so the test cannot pass against a shape no wrapper produces.
    """
    from workbench.replay.common import ReplayResult, macro_aggregate

    results = [ReplayResult(model_id=model_id, checkpoint_id="fixture", category=category,
                            protocol="fashioniq_val_split", r1=10.0, r10=r10[index],
                            r50=r50[index], complete=complete[index])
               for index, category in enumerate(CATEGORIES)]
    payload = {"model_id": model_id, "protocol": "fashioniq_val_split",
               "results": [asdict(record) for record in results],
               "aggregate": macro_aggregate(results)}
    payload.update(extra)
    return json.dumps(payload, indent=2)


@pytest.mark.parametrize("model_id", WRAPPER_MODELS)
def test_every_wrapper_model_can_be_parsed(model_id: str) -> None:
    result = extract_aggregate_metrics(model_id, replay_document(model_id), "", 0)
    assert result.extraction_status == "EXTRACTED_VERIFIED"
    assert result.parser_id == f"{model_id}_workbench_replay_json"
    assert result.observed_metrics == {"r10": 22.0, "r50": 42.0, "mean": 32.0}
    assert set(result.category_metrics) == set(CATEGORIES)


def test_the_macro_is_recomputed_and_a_contradiction_is_rejected() -> None:
    """The wrapper's own arithmetic is not trusted; a document that disagrees with itself fails."""
    honest = json.loads(replay_document("pair"))
    assert parse_workbench_replay_output("pair", json.dumps(honest)).extraction_status == "EXTRACTED_VERIFIED"

    lying = json.loads(replay_document("pair"))
    lying["aggregate"]["macro_r10"] = 90.0
    assert parse_workbench_replay_output("pair", json.dumps(lying)).extraction_status != "EXTRACTED_VERIFIED"

    lying_mean = json.loads(replay_document("pair"))
    lying_mean["aggregate"]["macro_mean"] = 90.0
    assert parse_workbench_replay_output("pair", json.dumps(lying_mean)).extraction_status != "EXTRACTED_VERIFIED"


def test_another_models_document_is_not_accepted() -> None:
    """Dispatch is per model; a document naming a different model must not satisfy it."""
    other = replay_document("encoder")
    assert extract_aggregate_metrics("pair", other, "", 0).extraction_status != "EXTRACTED_VERIFIED"


def test_a_missing_category_is_rejected() -> None:
    document = json.loads(replay_document("pair"))
    document["results"] = [r for r in document["results"] if r["category"] != "toptee"]
    assert parse_workbench_replay_output("pair", json.dumps(document)).extraction_status != "EXTRACTED_VERIFIED"


def test_a_duplicate_category_is_rejected() -> None:
    document = json.loads(replay_document("pair"))
    document["results"].append(dict(document["results"][0]))
    assert parse_workbench_replay_output("pair", json.dumps(document)).extraction_status != "EXTRACTED_VERIFIED"


def test_an_incomplete_category_is_rejected() -> None:
    """A partial run is not a complete benchmark result; the wrapper refuses to claim one either."""
    document = replay_document("pair", complete=(True, True, False))
    assert parse_workbench_replay_output("pair", document).extraction_status != "EXTRACTED_VERIFIED"


def test_out_of_range_and_non_numeric_recalls_are_rejected() -> None:
    for bad in (150.0, -1.0, "21.0", None, float("nan")):
        document = json.loads(replay_document("pair"))
        document["results"][0]["r10"] = bad
        assert parse_workbench_replay_output("pair", json.dumps(document)).extraction_status != "EXTRACTED_VERIFIED", (
            f"{bad!r} was accepted")


def test_a_document_nested_in_progress_output_is_found() -> None:
    stream = "loading checkpoint...\n" + replay_document("pair") + "\n[OK] finished\n"
    assert parse_workbench_replay_output("pair", stream).extraction_status == "EXTRACTED_VERIFIED"


def test_malformed_input_is_rejected() -> None:
    for text in ("", "not json", "{", "[]", json.dumps({"model_id": "pair"})):
        result = parse_workbench_replay_output("pair", text)
        assert result.extraction_status == "EVALUATION_COMPLETED_METRICS_UNPARSED"
        assert result.observed_metrics is None


def test_a_failed_process_still_yields_no_metrics() -> None:
    result = extract_aggregate_metrics("pair", replay_document("pair"), "", 1)
    assert result.extraction_status == "EVALUATION_FAILED"
    assert result.observed_metrics is None


def test_models_without_a_wrapper_keep_reporting_the_honest_gap() -> None:
    """dcnet/tgcir/ptha_mtst print nothing parseable yet; saying so is the correct answer."""
    for model_id in ("dcnet", "tgcir", "ptha_mtst"):
        result = extract_aggregate_metrics(model_id, replay_document(model_id), "", 0)
        assert result.extraction_status == "AGGREGATE_PARSER_UNAVAILABLE", (
            f"{model_id} borrowed a parser it does not have")
        assert result.parser_id is None


def test_the_model_specific_parsers_are_untouched() -> None:
    """The shared replay parser must not answer for a model that prints its own format."""
    assert extract_aggregate_metrics("hint", replay_document("hint"), "", 0).extraction_status != "EXTRACTED_VERIFIED"
    assert extract_aggregate_metrics("limn", "{}", "", 0).parser_id == "limn_structured_json"
    assert extract_aggregate_metrics("csmcir", "{}", "", 0).parser_id == "csmcir_stdout_json"
