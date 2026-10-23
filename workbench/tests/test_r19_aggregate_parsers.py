"""R19-08: parse the native aggregate documents emitted by TME and PTHA wrappers."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from workbench.backend.metrics_extraction import extract_aggregate_metrics  # noqa: E402
from workbench.replay.common import ReplayResult, macro_aggregate  # noqa: E402

CATEGORIES = ("dress", "shirt", "toptee")
TME_PROTOCOL = "fashioniq_val_split_ref_included"
PTHA_PROTOCOL = "fashioniq_original_split"
PTHA_LABEL = "PTHA+MTST | CIRR checkpoint -> FashionIQ | cross-dataset transfer | exploratory"
R10 = {"dress": 50.0, "shirt": 55.0, "toptee": 60.0}
R50 = {"dress": 70.0, "shirt": 75.0, "toptee": 80.0}


def tme_document() -> dict:
    """Use wrapper's `asdict(ReplayResult)` records and `macro_aggregate` output."""
    rows = [ReplayResult(
        model_id="tme", checkpoint_id="fashioniq_0", category=category,
        protocol=TME_PROTOCOL, r1=25.0, r10=R10[category], r50=R50[category],
        complete=True, provenance={"checkpoint": "/fixture/best_model.pth",
                                   "checkpoint_sha256": "a" * 64,
                                   "source_commit": "394aad79df2c104feb32a2c96132582012765062",
                                   "checkpoint_training_noise_pct": 0,
                                   "reference_policy": "reference_included",
                                   "checkpoint_selection_rule": "best_validation_checkpoint_at_sigma_0"})
        for category in CATEGORIES]
    return {"model_id": "tme", "protocol": TME_PROTOCOL,
            "results": [asdict(row) for row in rows],
            "aggregate": macro_aggregate(rows)}


def ptha_document() -> dict:
    r10 = sum(R10.values()) / 3
    r50 = sum(R50.values()) / 3
    return {
        "model_id": "ptha_mtst",
        "protocol_id": PTHA_PROTOCOL,
        "label": PTHA_LABEL,
        "train_dataset": "CIRR",
        "evaluation_dataset": "FashionIQ",
        "checkpoint_id": "cirr_transfer",
        "checkpoint_sha256": "b" * 64,
        "category_metrics": {category: {"r10": R10[category], "r50": R50[category]}
                             for category in CATEGORIES},
        "macro_metrics": {"r10": r10, "r50": r50, "mean": (r10 + r50) / 2},
        "paper_fashioniq_context_only": {
            "training_condition": "FashionIQ_MTST", "r10": 54.99, "r50": 76.28,
            "mean": 65.64,
            "mismatch": "training dataset/checkpoint differs; numerical context only, not reproduction",
            "differences_transfer_minus_paper": {
                "r10": r10 - 54.99, "r50": r50 - 76.28,
                "mean": (r10 + r50) / 2 - 65.64,
            },
        },
    }


def parse(model_id: str, document: dict, **kwargs):
    return extract_aggregate_metrics(model_id, json.dumps(document), "", 0, **kwargs)


@pytest.mark.parametrize(("model_id", "document", "checkpoint", "protocol", "parser"), [
    ("tme", tme_document, "fashioniq_0", TME_PROTOCOL, "tme_replay_json"),
    ("ptha_mtst", ptha_document, "cirr_transfer", PTHA_PROTOCOL, "ptha_mtst_transfer_json"),
])
def test_native_wrapper_documents_parse_and_recompute_macro(
    model_id, document, checkpoint, protocol, parser
):
    result = parse(model_id, document(), expected_checkpoint_id=checkpoint,
                   expected_protocol_id=protocol)
    assert result.extraction_status == "EXTRACTED_VERIFIED"
    assert result.parser_id == parser
    assert set(result.category_metrics) == set(CATEGORIES)
    assert result.observed_metrics == {"r10": 55.0, "r50": 75.0, "mean": 65.0}
    assert result.expected_checkpoint_id == checkpoint
    assert result.expected_protocol_id == protocol
    assert result.identity_status == "IDENTITY_CHECKED_CHECKPOINT_PROTOCOL_RUN_ID_NOT_IN_DOCUMENT"
    assert result.identity_checks["model_id"]["status"] == "MATCHED"
    assert result.identity_checks["checkpoint_id"]["status"] == "MATCHED"
    assert result.identity_checks["protocol_id"]["status"] == "MATCHED"
    assert result.identity_checks["run_id"]["status"] == "NOT_IN_DOCUMENT"
    assert result.refusal_reason is None


def test_tme_parses_the_nested_replay_result_schema():
    result = parse("tme", tme_document(), expected_checkpoint_id="fashioniq_0",
                   expected_protocol_id=TME_PROTOCOL)
    assert result.category_metrics["dress"] == {"r10": 50.0, "r50": 70.0}
    assert result.raw_output["results"][0]["provenance"]["checkpoint_sha256"] == "a" * 64


def test_ptha_label_and_paper_delta_remain_exploratory_context():
    result = parse("ptha_mtst", ptha_document(), expected_checkpoint_id="cirr_transfer",
                   expected_protocol_id=PTHA_PROTOCOL)
    assert result.context_label == PTHA_LABEL
    assert result.parity_status == "PARITY_NOT_EVALUATED"
    assert result.raw_output["paper_fashioniq_context_only"]["mismatch"].endswith(
        "not reproduction")


def test_missing_and_duplicate_categories_are_distinguishable_refusals():
    missing = tme_document()
    missing["results"].pop()
    result = parse("tme", missing)
    assert result.extraction_status == "EVALUATION_COMPLETED_METRICS_UNPARSED"
    assert result.refusal_reason == "category_set_mismatch"

    duplicate = tme_document()
    duplicate["results"].append(dict(duplicate["results"][0]))
    result = parse("tme", duplicate)
    assert result.extraction_status == "EVALUATION_COMPLETED_METRICS_UNPARSED"
    assert result.refusal_reason == "duplicate_category"


def test_invalid_recalls_and_contradictory_macro_are_refused():
    for bad in (float("nan"), float("inf"), -0.01, 100.01):
        document = tme_document()
        document["results"][0]["r10"] = bad
        result = parse("tme", document)
        assert result.refusal_reason == "invalid_recall", repr(bad)

    document = ptha_document()
    document["macro_metrics"]["mean"] += 1
    result = parse("ptha_mtst", document)
    assert result.refusal_reason == "macro_mismatch"


def test_ptha_category_gaps_duplicate_json_keys_and_invalid_values_are_refused():
    missing = ptha_document()
    del missing["category_metrics"]["shirt"]
    assert parse("ptha_mtst", missing).refusal_reason == "category_set_mismatch"

    duplicate_key = '{"model_id":"ptha_mtst","category_metrics":{"dress":{},"dress":{}}}'
    assert extract_aggregate_metrics("ptha_mtst", duplicate_key, "", 0).refusal_reason == (
        "duplicate_json_key")

    for bad in (float("nan"), float("inf"), -0.01, 100.01):
        invalid = ptha_document()
        invalid["category_metrics"]["dress"]["r10"] = bad
        assert parse("ptha_mtst", invalid).refusal_reason == "invalid_recall", repr(bad)

    invalid_macro = ptha_document()
    invalid_macro["macro_metrics"]["r10"] = float("inf")
    assert parse("ptha_mtst", invalid_macro).refusal_reason == "invalid_macro"


def test_checkpoint_and_protocol_request_mismatches_are_distinguishable():
    result = parse("tme", tme_document(), expected_checkpoint_id="cirr_0",
                   expected_protocol_id=TME_PROTOCOL)
    assert result.refusal_reason == "checkpoint_request_mismatch"

    result = parse("ptha_mtst", ptha_document(), expected_checkpoint_id="cirr_transfer",
                   expected_protocol_id="fashioniq_val_split")
    assert result.refusal_reason == "protocol_request_mismatch"


def test_foreign_and_partial_documents_are_refused():
    foreign = tme_document()
    foreign["model_id"] = "pair"
    assert parse("tme", foreign).refusal_reason == "model_identity_mismatch"

    partial = ptha_document()
    del partial["paper_fashioniq_context_only"]
    assert parse("ptha_mtst", partial).refusal_reason == "partial_output"

    incomplete = tme_document()
    incomplete["results"][1]["complete"] = False
    assert parse("tme", incomplete).refusal_reason == "partial_category"


def test_missing_request_identity_is_explicitly_not_checked():
    result = parse("tme", tme_document())
    assert result.extraction_status == "EXTRACTED_VERIFIED"
    assert result.identity_status == "IDENTITY_MODEL_CHECKED_REQUEST_ID_NOT_CHECKED_RUN_ID_NOT_IN_DOCUMENT"
    assert result.identity_checks["checkpoint_id"]["status"] == "NOT_CHECKED"
    assert result.identity_checks["protocol_id"]["status"] == "NOT_CHECKED"
    assert result.expected_checkpoint_id is None
    assert result.expected_protocol_id is None
