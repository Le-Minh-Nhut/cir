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


#: Models whose only runnable entry point is a workbench replay wrapper, so the JSON the wrapper
#: prints is the model's real output and gets the shared replay parser. A model is added here only
#: when a wrapper for it exists; the `tb`-style evaluators that print their own format keep their
#: own parsers.
_WORKBENCH_REPLAY_PARSED_MODELS = {
    "pair", "encoder", "clvc_net", "intent", "habit", "conesep", "airknow",
}


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

    # Identity: when the document declares which model produced it, it must be THIS model. The
    # native tuple shape carries no model id, so the check applies only where the producer states
    # one — but it must apply there, or another model's document would be read as HINT's result.
    if isinstance(json_obj, dict) and json_obj.get("model_id") not in (None, "hint"):
        return unparsed

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

def _replay_records(document: dict) -> list | None:
    """The per-category records in a replay document, whichever key the producer used.

    Two real spellings exist in this repository, and both are produced by wrappers that call the same
    native `test()`:

    * `results` — `pair`, `encoder`, `clvc_net` and the LIMN-family observer payloads;
    * `categories` — `airknow`, `habit`, `intent`, `conesep`, and every other `*_replay` main written
      from the same template.

    Accepting only `results` meant four wrappers produced a complete, correct document that the
    parser refused, so a successful run with exit 0 reported
    `EVALUATION_COMPLETED_METRICS_UNPARSED` and the model looked unintegrated. The keys are synonyms
    for the same list, so either is read; if a document carries BOTH and they disagree, that is an
    ambiguous result and is refused rather than resolved by preferring one.
    """
    results = document.get("results")
    categories = document.get("categories")
    if results is not None and categories is not None:
        if results != categories:
            return None
    for candidate in (results, categories):
        if isinstance(candidate, list):
            return candidate
    return None


def parse_workbench_replay_output(model_id: str, stdout_text: str,
                                  paper_metrics: dict[str, float | None] | None = None) -> MetricExtractionResult:
    """Parse the JSON a `workbench/replay/*` wrapper prints.

    Every replay wrapper this repository adds emits one document with a `results` list (one record
    per category, carrying `r1`/`r10`/`r50`) and an `aggregate` block. That document is a workbench
    product, not upstream's, so no model-specific parser reads it: without this one the eight models
    whose only runnable entry point IS a wrapper -- pair, encoder, clvc_net, intent, habit, conesep,
    airknow, hint -- reported AGGREGATE_PARSER_UNAVAILABLE even on a successful run with exit 0.

    The metrics are the native ones (`test()`'s own recall values, recorded by the wrapper). The
    macro is recomputed here from the three categories instead of being copied from `aggregate`, so a
    wrapper whose arithmetic or category set is wrong cannot pass as a complete result.
    """
    categories = ("dress", "shirt", "toptee")
    parser_id = f"{model_id}_workbench_replay_json"
    unparsed = MetricExtractionResult(
        extraction_status="EVALUATION_COMPLETED_METRICS_UNPARSED",
        parser_id=parser_id,
        parser_version=1,
        observed_metrics=None,
        category_metrics=None,
        parity_status="PARITY_NOT_EVALUATED",
        metric_source="stdout",
    )

    # The wrapper prints the document and nothing else, but progress lines are allowed for, so the
    # outermost brace-matched object is used rather than requiring a bare stream.
    document = None
    try:
        candidate = json.loads(stdout_text.strip())
        if isinstance(candidate, dict):
            document = candidate
    except Exception:
        document = None
    if document is None:
        for start in (i for i, ch in enumerate(stdout_text) if ch == "{"):
            depth = 0
            for index in range(start, len(stdout_text)):
                char = stdout_text[index]
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            candidate = json.loads(stdout_text[start:index + 1])
                        except Exception:
                            break
                        if isinstance(candidate, dict) and _replay_records(candidate) is not None:
                            document = candidate
                        break
            if document is not None:
                break

    records = _replay_records(document) if isinstance(document, dict) else None
    if records is None:
        return unparsed
    # A wrapper for a different model must not satisfy this model's dispatch entry.
    if document.get("model_id") != model_id:
        return unparsed
    # Every category must be present exactly once and marked complete: a partial run is not a
    # complete benchmark result, and the wrapper itself refuses to claim one.
    by_category: dict[str, dict] = {}
    for record in records:
        if not isinstance(record, dict):
            return unparsed
        category = record.get("category")
        if category in by_category:
            return unparsed
        by_category[category] = record
    records = by_category
    if set(records) != set(categories):
        return unparsed
    if not all(records[category].get("complete") for category in categories):
        return unparsed

    category_metrics: dict[str, dict[str, float | None]] = {}
    for category in categories:
        record = records[category]
        r10 = record.get("r10")
        r50 = record.get("r50")
        if not (_is_valid_recall(r10) and _is_valid_recall(r50)):
            return unparsed
        category_metrics[category] = {"r10": float(r10), "r50": float(r50)}

    r10 = sum(category_metrics[c]["r10"] for c in categories) / len(categories)
    r50 = sum(category_metrics[c]["r50"] for c in categories) / len(categories)
    mean = (r10 + r50) / 2.0

    declared = document.get("aggregate") or {}
    for key, value in (("macro_r10", r10), ("macro_r50", r50), ("macro_mean", mean)):
        declared_value = declared.get(key)
        if declared_value is not None and abs(float(declared_value) - value) > 0.05:
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
        raw_output=document,
    )

def parse_dcnet_output(stdout_text: str, paper_metrics: dict[str, float | None] | None = None) -> MetricExtractionResult:
    """Parse the metrics DCNet's own evaluator logs.

    `test.py:65-84` prints, per epoch of validation:

        [<category>] comp corr r@10, r@50: <comp10> <comp50> <corr10> <corr50>
        comb: <cat1_10> <cat1_50> <cat2_10> <cat2_50> <cat3_10> <cat3_50>

    and, when a category yields fewer cells, a two-number `[<category>] r@10, r@50:` line instead.

    Two facts decide how to read them:

    * **units.** `model/metric.py:5-21` computes `recall` as a FRACTION (`len(inter)/len(act)`), so
      every printed cell is 0..1 and must be multiplied by 100 EXACTLY ONCE on the way into the
      workbench % schema. Multiplying twice was the obvious way to make the numbers look plausible.
    * **which branch.** Three variants exist (combined, composition, correction). The `comb:` line is
      the one the registry's own paper row describes — `paper_benchmarks/papers.yaml` defines the row
      as "the mean of the six printed cells" — so that is what is read. The composition and correction
      splits are kept in `category_metrics` under distinct keys rather than being silently conflated
      with the combined result, and no branch is chosen by which number looks better.

    The macro is recomputed from the parsed cells and then checked against the evaluator's own
    `comb_avg`, so a misread stream is refused instead of being reported as a measurement.
    """
    parser_id = "dcnet_native_logger"
    unparsed = MetricExtractionResult(
        extraction_status="EVALUATION_COMPLETED_METRICS_UNPARSED",
        parser_id=parser_id,
        parser_version=1,
        observed_metrics=None,
        category_metrics=None,
        parity_status="PARITY_NOT_EVALUATED",
        metric_source="stdout",
    )

    # The category ORDER is the evaluator's, taken from the config it ran with, so each combined pair
    # is mapped by the name in the per-category line rather than by position. Position alone would
    # silently transpose dress/shirt/toptee if a config ordered them differently, which is exactly the
    # class of error the paper row's "mean of the six cells" cannot detect.
    combined_pairs: dict[str, tuple[float, float]] = {}
    for match in re.finditer(
            r"\[(dress|shirt|toptee)\]\s+comp\s+corr\s+r@10,\s*r@50:\s*"
            r"([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)", stdout_text):
        category = match.group(1)
        # `i[0..3]` is comp10, comp50, corr10, corr50 (trainer_joint.py:231 `metric['recall'][i] =`).
        comp10, comp50, corr10, corr50 = (float(match.group(n)) * 100.0 for n in (2, 3, 4, 5))
        combined_pairs[category] = (comp10, comp50)
        combined_pairs[f"{category}__correction"] = (corr10, corr50)
        combined_pairs[f"{category}__composition"] = (comp10, comp50)

    single_pairs: dict[str, tuple[float, float]] = {}
    for match in re.finditer(
            r"\[(dress|shirt|toptee)\]\s+r@10,\s*r@50:\s*([0-9.eE+-]+)\s+([0-9.eE+-]+)",
            stdout_text):
        single_pairs[match.group(1)] = (float(match.group(2)) * 100.0, float(match.group(3)) * 100.0)

    comb_line = re.search(r"(?:^|\n)\s*(?:[^\n]* - INFO - )?comb:\s*"
                          r"([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+"
                          r"([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)", stdout_text)

    if comb_line:
        cells = [float(value) * 100.0 for value in comb_line.groups()]
        # The `comb:` line carries no names, so it is only usable when the named per-category lines
        # are present to establish the order and the count.
        named = [c for c in ("dress", "shirt", "toptee") if c in combined_pairs]
        if len(named) != 3:
            return unparsed
        for index, category in enumerate(named):
            combined_pairs[category] = (cells[index * 2], cells[index * 2 + 1])
    elif single_pairs:
        # The two-cell shape reports the same combined metric; accept it only when all three
        # categories appear, so a truncated stream cannot look complete.
        if set(single_pairs) != {"dress", "shirt", "toptee"}:
            return unparsed
        combined_pairs = dict(single_pairs)
    else:
        return unparsed

    category_metrics: dict[str, dict[str, float | None]] = {}
    for category in ("dress", "shirt", "toptee"):
        pair = combined_pairs.get(category)
        if pair is None:
            return unparsed
        r10, r50 = pair
        if not (_is_valid_recall(r10) and _is_valid_recall(r50)):
            return unparsed
        category_metrics[category] = {"r10": round(r10, 4), "r50": round(r50, 4)}

    # The composition/correction splits are kept beside the combined result, never merged into it.
    for variant in ("__composition", "__correction"):
        for category in ("dress", "shirt", "toptee"):
            pair = combined_pairs.get(f"{category}{variant}")
            if pair and _is_valid_recall(pair[0]) and _is_valid_recall(pair[1]):
                category_metrics[f"{category}{variant}"] = {"r10": round(pair[0], 4),
                                                            "r50": round(pair[1], 4)}

    macro_r10 = sum(category_metrics[c]["r10"] for c in ("dress", "shirt", "toptee")) / 3.0
    macro_r50 = sum(category_metrics[c]["r50"] for c in ("dress", "shirt", "toptee")) / 3.0
    macro_mean = (macro_r10 + macro_r50) / 2.0

    declared = re.search(r"\[comb_avg\]\s*:\s*([0-9.eE+-]+)", stdout_text)
    if declared and abs(float(declared.group(1)) * 100.0 - macro_mean) > 0.5:
        # A cancellation reveals itself here: with every cell multiplied once, the mean of the six is
        # the macro mean, so a mismatch means the stream was misread or the run was partial.
        return unparsed

    observed = {"r10": round(macro_r10, 4), "r50": round(macro_r50, 4), "mean": round(macro_mean, 4)}
    return MetricExtractionResult(
        extraction_status="EXTRACTED_VERIFIED",
        parser_id=parser_id,
        parser_version=1,
        observed_metrics=observed,
        category_metrics=category_metrics,
        parity_status=_check_parity(observed, paper_metrics),
        metric_source="stdout",
        raw_output={"branch": "combined"},
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
    if model_id == "dcnet":
        return parse_dcnet_output(stdout_text, paper_metrics)
    if model_id == "hint":
        return parse_hint_output(stdout_text, paper_metrics)
    if model_id in _WORKBENCH_REPLAY_PARSED_MODELS:
        return parse_workbench_replay_output(model_id, stdout_text, paper_metrics)
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
