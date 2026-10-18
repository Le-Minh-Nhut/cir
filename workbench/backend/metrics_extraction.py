"""Source-backed aggregate metric extraction for official evaluators."""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any

from workbench.backend import paper_gate


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
    """Compare observed metrics against a paper row, refusing to judge a partial row.

    DEPRECATED as an acceptance decision: this once used an absolute `0.05` tolerance, which is a
    different rule from the canonical five-percent *relative* error the benchmark gate enforces, and
    two contradictory PASS/FAIL meanings for one run is exactly the ambiguity that breaks audits.
    The verdict that matters is the gate's (`benchmark_gate` in the report); this field now applies
    the same canonical relative-error arithmetic so it can never contradict the gate, and it is kept
    only as a display annotation for older consumers.

    A row that publishes R@50 but not R@10 (SPRC's registered row is exactly that) cannot be
    "verified": the missing metric would be silently exempt, so a run whose R@10 is nonsense would
    still report PARITY_VERIFIED. An absent paper value is an unverified reference for that metric,
    not an automatic pass.
    """
    if not observed or not paper:
        return "PARITY_NOT_EVALUATED"
    compared = 0
    for k in ("r10", "r50", "mean"):
        p_val = paper.get(k)
        if p_val is None:
            return "PARITY_NOT_EVALUATED"
        o_val = observed.get(k)
        if o_val is None:
            return "PARITY_NOT_EVALUATED"
        comparison = paper_gate.compare_metric(k, float(p_val), float(o_val))
        if comparison.relative_error_pct is None or not comparison.within_tolerance:
            return "PARITY_MISMATCH"
        compared += 1
    return "PARITY_VERIFIED" if compared > 0 else "PARITY_NOT_EVALUATED"


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
        # Internal consistency only (declared aggregate vs the run's own categories), not a paper
        # comparison; see the same note in parse_csmcir_output.
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
            # Internal consistency only: the evaluator's own average against the categories it just
            # printed, from the same run. This is not a paper comparison and does not use the gate's
            # relative rule; a discrepancy here means the parser misread the output.
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

def parse_hint_output(stdout_text: str, paper_metrics: dict[str, float | None] | None = None) -> MetricExtractionResult:
    """Parse aggregate Recall from the HINT replay wrapper OR from the native evaluator's own print.

    Two real output shapes exist, and both are produced by this repository:

    * the wrapper (`workbench/replay/hint.py`) prints one JSON document containing `aggregate` and
      per-category `r10`/`r50`, because the pinned `test.py` has no `__main__` to print anything;
    * the pinned evaluator itself returns `[('{cat}_r{k}', value), ...]` for k in (1, 10, 50)
      (`test.py:103-108`), which callers such as `train.py:199` print directly.

    Either way the numbers are the native ones — this function reads them, it never derives or
    substitutes them. The macro is recomputed from the categories and compared with whatever the
    producer declared, so a misread or a truncated stream cannot pass as a complete result.
    """
    categories = ("dress", "shirt", "toptee")
    parser_id = "hint_native_recall"
    unparsed = MetricExtractionResult(
        extraction_status="EVALUATION_COMPLETED_METRICS_UNPARSED",
        parser_id=parser_id,
        parser_version=1,
        observed_metrics=None,
        category_metrics=None,
        parity_status="PARITY_NOT_EVALUATED",
        metric_source="stdout",
    )

    category_metrics: dict[str, dict[str, float | None]] = {}

    # Shape 1: the wrapper's JSON document. `json.loads` on the whole stream first (the wrapper
    # prints the document and nothing else), then a brace-matching scan for a document embedded in
    # other output. A regex is not enough here: the document nests `aggregate` inside `categories`,
    # and a pattern that cannot span nesting silently reports a complete result as unparsed.
    json_obj = None
    candidates: list[str] = []
    try:
        parsed = json.loads(stdout_text.strip())
        if isinstance(parsed, dict):
            candidates.append(stdout_text.strip())
    except Exception:
        pass
    for start in (i for i, ch in enumerate(stdout_text) if ch == "{"):
        depth = 0
        for index in range(start, len(stdout_text)):
            char = stdout_text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(stdout_text[start:index + 1])
                    break
    for candidate in reversed(candidates):
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict) and ("aggregate" in parsed or "categories" in parsed):
            json_obj = parsed
            break

    if isinstance(json_obj, dict):
        for category in categories:
            entry = next((item for item in (json_obj.get("categories") or [])
                          if isinstance(item, dict) and item.get("category") == category), None)
            r10 = entry.get("r10") if entry else json_obj.get(f"{category}_r10")
            r50 = entry.get("r50") if entry else json_obj.get(f"{category}_r50")
            if not (_is_valid_recall(r10) and _is_valid_recall(r50)):
                return unparsed
            category_metrics[category] = {"r10": float(r10), "r50": float(r50)}
    else:
        # Shape 2: the evaluator's native `[(label, value), ...]` print.
        pairs = re.findall(r"\(\s*'([a-z]+)_r(1|10|50)'\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*\)", stdout_text)
        if not pairs:
            pairs = re.findall(r"\(\s*\"([a-z]+)_r(1|10|50)\"\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*\)", stdout_text)
        if not pairs:
            return unparsed
        collected: dict[str, dict[str, float]] = {}
        for category, k, value in pairs:
            if category not in categories:
                return unparsed
            collected.setdefault(category, {})[f"r{k}"] = float(value)
        for category in categories:
            entry = collected.get(category)
            if not entry or "r10" not in entry or "r50" not in entry:
                return unparsed
            if not (_is_valid_recall(entry["r10"]) and _is_valid_recall(entry["r50"])):
                return unparsed
            category_metrics[category] = {"r10": entry["r10"], "r50": entry["r50"]}

    if set(category_metrics) != set(categories):
        return unparsed

    # Independent macro recomputation. The evaluator's own convention is the unweighted mean over the
    # three categories, then `(mean_r10 + mean_r50) / 2` (blip_validate.py:90-95) — the same
    # arithmetic the registry's `reported_mean` uses. A declared value that disagrees means the
    # parser misread the stream.
    r10 = sum(category_metrics[c]["r10"] for c in categories) / len(categories)
    r50 = sum(category_metrics[c]["r50"] for c in categories) / len(categories)
    mean = (r10 + r50) / 2.0

    if isinstance(json_obj, dict):
        declared = json_obj.get("aggregate") or {}
        declared_r10 = declared.get("macro_r10")
        declared_r50 = declared.get("macro_r50")
        declared_mean = declared.get("macro_mean")
        if declared_r10 is not None and abs(float(declared_r10) - r10) > 0.05:
            return unparsed
        if declared_r50 is not None and abs(float(declared_r50) - r50) > 0.05:
            return unparsed
        if declared_mean is not None and abs(float(declared_mean) - mean) > 0.05:
            return unparsed

    observed = {"r10": round(r10, 4), "r50": round(r50, 4), "mean": round(mean, 4)}
    return MetricExtractionResult(
        extraction_status="EXTRACTED_VERIFIED",
        parser_id=parser_id,
        parser_version=1,
        observed_metrics=observed,
        category_metrics=category_metrics,
        parity_status=_check_parity(observed, paper_metrics),
        metric_source="stdout",
        raw_output=json_obj,
    )

def parse_combiner_output(stdout_text: str, paper_metrics: dict[str, float | None] | None = None) -> MetricExtractionResult:
    """Parse the `name = value` recall lines CLIP4Cir's src/validate.py prints for FashionIQ.

    The evaluator prints six per-category lines (shirt, dress, toptee x recall@10/50) and two
    unweighted averages; values are percentages. The declared averages are validated against the
    categories the same run printed (internal consistency only — the gate owns the paper
    comparison).
    """
    unparsed = MetricExtractionResult(
        extraction_status="EVALUATION_COMPLETED_METRICS_UNPARSED",
        parser_id="clip4cir_stdout_recalls",
        parser_version=1,
        observed_metrics=None,
        category_metrics=None,
        parity_status="PARITY_NOT_EVALUATED",
        metric_source="stdout",
    )
    values: dict[str, float] = {}
    for match in re.finditer(r"^\s*([a-z_]+_recallat\d+)\s*=\s*([-+0-9.eE]+)\s*$", stdout_text, re.MULTILINE):
        key = match.group(1)
        if key in values:
            # A repeated line that disagrees with the first is ambiguous output; the evaluator
            # prints each recall exactly once, so any repetition is misread or tampered output.
            return unparsed
        try:
            values[key] = float(match.group(2))
        except ValueError:
            return unparsed
    averages: dict[str, float] = {}
    for m in re.finditer(r"^\s*average recall(\d+)\s*=\s*([-+0-9.eE]+)\s*$", stdout_text, re.MULTILINE):
        if m.group(1) in averages:
            return unparsed
        try:
            averages[m.group(1)] = float(m.group(2))
        except ValueError:
            return unparsed
    try:
        cat_metrics: dict[str, dict[str, float | None]] = {}
        for cat in ("dress", "shirt", "toptee"):
            v10 = values[f"{cat}_recallat10"]
            v50 = values[f"{cat}_recallat50"]
            if not (_is_valid_recall(v10) and _is_valid_recall(v50)):
                raise ValueError(f"invalid recall value in {cat}")
            cat_metrics[cat] = {"r10": float(v10), "r50": float(v50)}
        avg10 = averages["10"]
        avg50 = averages["50"]
        if not (_is_valid_recall(avg10) and _is_valid_recall(avg50)):
            raise ValueError("invalid average recall values")
        exp10 = sum(cat_metrics[c]["r10"] for c in ("dress", "shirt", "toptee")) / 3.0
        exp50 = sum(cat_metrics[c]["r50"] for c in ("dress", "shirt", "toptee")) / 3.0
        if abs(avg10 - exp10) > 0.05 or abs(avg50 - exp50) > 0.05:
            # Internal consistency only: the evaluator's own average against the categories it just
            # printed, from the same run. This is not a paper comparison and does not use the gate's
            # relative rule; a discrepancy here means the parser misread the output.
            raise ValueError("inconsistent declared average")
        observed = {
            "r10": round(float(avg10), 4),
            "r50": round(float(avg50), 4),
            "mean": round((float(avg10) + float(avg50)) / 2.0, 4),
        }
        parity = _check_parity(observed, paper_metrics)
        return MetricExtractionResult(
            extraction_status="EXTRACTED_VERIFIED",
            parser_id="clip4cir_stdout_recalls",
            parser_version=1,
            observed_metrics=observed,
            category_metrics=cat_metrics,
            parity_status=parity,
            metric_source="stdout",
            raw_output={"averages": {"recall10": avg10, "recall50": avg50}},
        )
    except (KeyError, ValueError):
        return unparsed

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
    if model_id in ("combiner_rn50x4_noft", "clip4cir_rn50x4_fullft"):
        return parse_combiner_output(stdout_text, paper_metrics)
    if model_id == "hint":
        return parse_hint_output(stdout_text, paper_metrics)
    if model_id == "sprc":
        # blip_validate.py:88-98 prints the same JSON key set as CSMCIR's evaluator; the csmcir
        # parser's macro-validation matches the same keys, so the shape is reused, not assumed.
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
