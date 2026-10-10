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


import math

def _is_valid_recall(val: Any) -> bool:
    if isinstance(val, bool):
        return False
    if not isinstance(val, (int, float)):
        return False
    if not math.isfinite(val):
        return False
    return 0.0 <= float(val) <= 100.0


def parse_limn_output(stdout_text: str, paper_metrics: dict[str, float | None] | None = None) -> MetricExtractionResult:
    """Parse structured JSON emitted by workbench.replay.limn with independent macro recomputation."""
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
    cat_metrics: dict[str, dict[str, float | None]] = {}

    # Check duplicate or unexpected categories
    seen_cats = set()
    for c in categories:
        cat_name = c.get("category")
        if not cat_name or cat_name in seen_cats:
            return MetricExtractionResult(
                extraction_status="EVALUATION_COMPLETED_METRICS_UNPARSED",
                parser_id="limn_structured_json",
                parser_version=1,
                observed_metrics=None,
                category_metrics=None,
                parity_status="PARITY_NOT_EVALUATED",
                metric_source="stdout",
                raw_output=json_obj,
            )
        seen_cats.add(cat_name)
        cat_metrics[cat_name] = {
            "r1": c.get("r1"),
            "r10": c.get("r10"),
            "r50": c.get("r50"),
        }

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

    for cat, vals in cat_metrics.items():
        for metric in ("r10", "r50"):
            v = vals.get(metric)
            if v is None or not _is_valid_recall(v):
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

    d_10 = float(cat_metrics["dress"]["r10"])
    s_10 = float(cat_metrics["shirt"]["r10"])
    t_10 = float(cat_metrics["toptee"]["r10"])
    d_50 = float(cat_metrics["dress"]["r50"])
    s_50 = float(cat_metrics["shirt"]["r50"])
    t_50 = float(cat_metrics["toptee"]["r50"])

    computed_r10_macro = (d_10 + s_10 + t_10) / 3.0
    computed_r50_macro = (d_50 + s_50 + t_50) / 3.0
    computed_mean = (computed_r10_macro + computed_r50_macro) / 2.0

    decl_r10 = agg.get("macro_r10")
    decl_r50 = agg.get("macro_r50")
    decl_mean = agg.get("macro_mean")
    if not (_is_valid_recall(decl_r10) and _is_valid_recall(decl_r50) and _is_valid_recall(decl_mean)):
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

    if abs(float(decl_r10) - computed_r10_macro) > 0.05 or abs(float(decl_r50) - computed_r50_macro) > 0.05 or abs(float(decl_mean) - computed_mean) > 0.05:
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
        "r10": round(computed_r10_macro, 4),
        "r50": round(computed_r50_macro, 4),
        "mean": round(computed_mean, 4),
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
    """Parse JSON results emitted by CSMCIR's validate_blip_csmcir.py with macro validation."""
    json_obj = None
    try:
        matches = re.findall(r"\{[^{}]*\"average_recall\"[^{}]*\}", stdout_text, re.DOTALL)
        if matches:
            if len(matches) > 1:
                parsed = [json.loads(m) for m in matches]
                if any(p != parsed[0] for p in parsed):
                    return MetricExtractionResult(
                        extraction_status="EVALUATION_COMPLETED_METRICS_UNPARSED",
                        parser_id="csmcir_stdout_json",
                        parser_version=1,
                        observed_metrics=None,
                        category_metrics=None,
                        parity_status="PARITY_NOT_EVALUATED",
                        metric_source="stdout",
                    )
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
        cat_metrics: dict[str, dict[str, float | None]] = {}
        for cat in ("dress", "shirt", "toptee"):
            r10_k = f"{cat}_recall_at10"
            r50_k = f"{cat}_recall_at50"
            if r10_k not in json_obj or r50_k not in json_obj:
                raise ValueError(f"missing category {cat}")
            v10 = json_obj[r10_k]
            v50 = json_obj[r50_k]
            if not (_is_valid_recall(v10) and _is_valid_recall(v50)):
                raise ValueError(f"invalid recall value in {cat}")
            cat_metrics[cat] = {"r10": float(v10), "r50": float(v50)}

        avg10 = json_obj.get("average_recall_at10")
        avg50 = json_obj.get("average_recall_at50")
        avg_mean = json_obj.get("average_recall")
        if not (_is_valid_recall(avg10) and _is_valid_recall(avg50) and _is_valid_recall(avg_mean)):
            raise ValueError("invalid average recall values")

        exp10 = (cat_metrics["dress"]["r10"] + cat_metrics["shirt"]["r10"] + cat_metrics["toptee"]["r10"]) / 3.0
        exp50 = (cat_metrics["dress"]["r50"] + cat_metrics["shirt"]["r50"] + cat_metrics["toptee"]["r50"]) / 3.0
        exp_mean = (exp10 + exp50) / 2.0

        if abs(float(avg10) - exp10) > 0.05 or abs(float(avg50) - exp50) > 0.05 or abs(float(avg_mean) - exp_mean) > 0.05:
            raise ValueError("inconsistent declared macro mean")

        observed = {
            "r10": round(float(avg10), 4),
            "r50": round(float(avg50), 4),
            "mean": round(float(avg_mean), 4),
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
    if model_id in ("csmcir", "synthetic_model"):
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
