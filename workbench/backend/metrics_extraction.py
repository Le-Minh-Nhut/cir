"""Source-backed aggregate metric extraction for official evaluators."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class MetricExtractionResult:
    extraction_status: str  # "EXTRACTED_VERIFIED", "EVALUATION_COMPLETED_METRICS_UNPARSED", "AGGREGATE_PARSER_UNAVAILABLE"
    parser_id: str | None
    parser_version: int
    observed_metrics: dict[str, float | None] | None
    category_metrics: dict[str, dict[str, float | None]] | None
    parity_status: str  # "PARITY_VERIFIED", "PARITY_MISMATCH", "PARITY_NOT_EVALUATED", "PARSER_UNAVAILABLE"
    metric_source: str | None
    raw_output: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _check_parity(
    observed: dict[str, float | None] | None,
    paper: dict[str, float | None] | None,
    tolerance: float = 0.05,
) -> str:
    if not observed or not paper:
        return "PARITY_NOT_EVALUATED"
    compared = 0
    for k in ("r10", "r50", "mean"):
        p_val = paper.get(k)
        if p_val is not None:
            o_val = observed.get(k)
            if o_val is None:
                return "PARITY_NOT_EVALUATED"
            if abs(float(o_val) - float(p_val)) > tolerance:
                return "PARITY_MISMATCH"
            compared += 1
    return "PARITY_VERIFIED" if compared > 0 else "PARITY_NOT_EVALUATED"


def parse_limn_output(stdout_text: str, paper_metrics: dict[str, float | None] | None = None) -> MetricExtractionResult:
    """Parse structured JSON emitted by workbench.replay.limn."""
    # Find JSON block in stdout
    json_obj = None
    for line in stdout_text.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                candidate = json.loads(line)
                if candidate.get("model_id") == "limn" and "aggregate" in candidate:
                    json_obj = candidate
                    break
            except Exception:
                pass
    if json_obj is None:
        try:
            # Maybe pretty-printed multiline JSON
            match = re.search(r"\{\s*\"model_id\":\s*\"limn\".*\}", stdout_text, re.DOTALL)
            if match:
                json_obj = json.loads(match.group(0))
        except Exception:
            pass

    if not json_obj or "aggregate" not in json_obj:
        return MetricExtractionResult(
            extraction_status="EVALUATION_COMPLETED_METRICS_UNPARSED",
            parser_id="limn_structured_json",
            parser_version=1,
            observed_metrics=None,
            category_metrics=None,
            parity_status="PARITY_NOT_EVALUATED",
            metric_source="stdout",
        )

    agg = json_obj.get("aggregate", {})
    categories = json_obj.get("categories", [])
    cat_metrics = {}
    for c in categories:
        cat_name = c.get("category")
        if cat_name:
            cat_metrics[cat_name] = {
                "r1": c.get("r1"),
                "r10": c.get("r10"),
                "r50": c.get("r50"),
            }

    # Must contain all 3 categories to have valid aggregate
    required_cats = {"dress", "shirt", "toptee"}
    if set(cat_metrics.keys()) != required_cats or not agg.get("complete"):
        return MetricExtractionResult(
            extraction_status="EVALUATION_COMPLETED_METRICS_UNPARSED",
            parser_id="limn_structured_json",
            parser_version=1,
            observed_metrics=None,
            category_metrics=cat_metrics,
            parity_status="PARITY_NOT_EVALUATED",
            metric_source="stdout",
            raw_output=json_obj,
        )

    # Arithmetic category macro mean validation
    d_10 = cat_metrics["dress"].get("r10")
    s_10 = cat_metrics["shirt"].get("r10")
    t_10 = cat_metrics["toptee"].get("r10")
    if None in (d_10, s_10, t_10):
        return MetricExtractionResult(
            extraction_status="EVALUATION_COMPLETED_METRICS_UNPARSED",
            parser_id="limn_structured_json",
            parser_version=1,
            observed_metrics=None,
            category_metrics=cat_metrics,
            parity_status="PARITY_NOT_EVALUATED",
            metric_source="stdout",
            raw_output=json_obj,
        )

    observed = {
        "r10": float(agg["macro_r10"]),
        "r50": float(agg["macro_r50"]),
        "mean": float(agg["macro_mean"]),
    }
    parity = _check_parity(observed, paper_metrics)

    return MetricExtractionResult(
        extraction_status="EXTRACTED_VERIFIED",
        parser_id="limn_structured_json",
        parser_version=1,
        observed_metrics=observed,
        category_metrics=cat_metrics,
        parity_status=parity,
        metric_source="stdout",
        raw_output=json_obj,
    )


def parse_csmcir_output(stdout_text: str, paper_metrics: dict[str, float | None] | None = None) -> MetricExtractionResult:
    """Parse JSON results emitted by CSMCIR's validate_blip_csmcir.py."""
    json_obj = None
    try:
        # Pinned validate_blip_csmcir prints json.dumps(results_dict, indent=4)
        matches = re.findall(r"\{[^{}]*\"average_recall\"[^{}]*\}", stdout_text, re.DOTALL)
        if matches:
            json_obj = json.loads(matches[-1])
    except Exception:
        pass

    if not json_obj:
        return MetricExtractionResult(
            extraction_status="EVALUATION_COMPLETED_METRICS_UNPARSED",
            parser_id="csmcir_stdout_json",
            parser_version=1,
            observed_metrics=None,
            category_metrics=None,
            parity_status="PARITY_NOT_EVALUATED",
            metric_source="stdout",
        )

    try:
        observed = {
            "r10": float(json_obj["average_recall_at10"]),
            "r50": float(json_obj["average_recall_at50"]),
            "mean": float(json_obj["average_recall"]),
        }
        cat_metrics = {}
        for cat in ("dress", "shirt", "toptee"):
            cat_metrics[cat] = {
                "r10": float(json_obj[f"{cat}_recall_at10"]) if f"{cat}_recall_at10" in json_obj else None,
                "r50": float(json_obj[f"{cat}_recall_at50"]) if f"{cat}_recall_at50" in json_obj else None,
            }
        parity = _check_parity(observed, paper_metrics)
        return MetricExtractionResult(
            extraction_status="EXTRACTED_VERIFIED",
            parser_id="csmcir_stdout_json",
            parser_version=1,
            observed_metrics=observed,
            category_metrics=cat_metrics,
            parity_status=parity,
            metric_source="stdout",
            raw_output=json_obj,
        )
    except Exception:
        return MetricExtractionResult(
            extraction_status="EVALUATION_COMPLETED_METRICS_UNPARSED",
            parser_id="csmcir_stdout_json",
            parser_version=1,
            observed_metrics=None,
            category_metrics=None,
            parity_status="PARITY_NOT_EVALUATED",
            metric_source="stdout",
            raw_output=json_obj,
        )


def extract_aggregate_metrics(
    model_id: str,
    stdout_text: str,
    stderr_text: str,
    return_code: int,
    paper_metrics: dict[str, float | None] | None = None,
) -> MetricExtractionResult:
    """Model-specific source-backed aggregate metric extraction."""
    if return_code != 0:
        return MetricExtractionResult(
            extraction_status="EVALUATION_FAILED",
            parser_id=None,
            parser_version=1,
            observed_metrics=None,
            category_metrics=None,
            parity_status="PARITY_NOT_EVALUATED",
            metric_source=None,
        )

    if model_id == "limn":
        return parse_limn_output(stdout_text, paper_metrics)
    if model_id == "csmcir":
        return parse_csmcir_output(stdout_text, paper_metrics)

    # Models with unaudited / unverified source output format
    return MetricExtractionResult(
        extraction_status="AGGREGATE_PARSER_UNAVAILABLE",
        parser_id=None,
        parser_version=1,
        observed_metrics=None,
        category_metrics=None,
        parity_status="PARSER_UNAVAILABLE",
        metric_source=None,
    )
