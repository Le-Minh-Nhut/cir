from __future__ import annotations

import math
from statistics import mean, median, pstdev
from typing import Any

from workbench.backend.errors import WorkbenchError
from workbench.backend.schemas.results import QueryResult, ResultRun


def require_same_protocol(runs: list[ResultRun]) -> str:
    protocols = {run.run.protocol_id for run in runs}
    if len(protocols) != 1:
        raise WorkbenchError("cross_protocol", "Cross-protocol analysis is invalid. Select runs from one FashionIQ protocol.", {"protocol_ids": sorted(protocols)})
    return protocols.pop()


def require_top_k(runs: list[ResultRun], k: int) -> None:
    if not 1 <= k <= 200:
        raise WorkbenchError("insufficient_top_k_depth", "K must be between 1 and 200.", {"requested_k": k})
    insufficient = [{"run_id": run.run.run_id, "top_k_saved": run.run.top_k_saved} for run in runs if run.run.top_k_saved < k]
    if insufficient:
        raise WorkbenchError("insufficient_top_k_depth", "Selected runs do not store enough top-K retrievals.", {"requested_k": k, "runs": insufficient})


def aligned_queries(runs: list[ResultRun]) -> dict[str, list[QueryResult]]:
    require_same_protocol(runs)
    if not runs:
        return {}
    maps = {run.run.run_id: {query.query_id: query for query in run.queries} for run in runs}
    baseline_id = runs[0].run.run_id
    baseline = maps[baseline_id]
    fields = ("category", "annotation_index", "reference_id", "target_id", "raw_captions")
    for run in runs[1:]:
        candidate = maps[run.run.run_id]
        missing, extra = sorted(set(baseline) - set(candidate)), sorted(set(candidate) - set(baseline))
        if missing or extra:
            raise WorkbenchError("query_alignment_mismatch", "Selected runs do not have an identical query universe.", {"baseline_run_id": baseline_id, "run_id": run.run.run_id, "missing_query_count": len(missing), "missing_query_ids": missing[:20], "extra_query_count": len(extra), "extra_query_ids": extra[:20]})
        for query_id, expected in baseline.items():
            actual = candidate[query_id]
            mismatches = {field: {"expected": getattr(expected, field), "actual": getattr(actual, field)} for field in fields if getattr(expected, field) != getattr(actual, field)}
            if mismatches:
                raise WorkbenchError("canonical_query_mismatch", "Canonical query fields differ for the same query_id.", {"query_id": query_id, "baseline_run_id": baseline_id, "run_id": run.run.run_id, "mismatches": mismatches})
    return {query_id: [maps[run.run.run_id][query_id] for run in runs] for query_id in baseline}


def _rows(run_ids: list[str], maps: dict[str, dict[str, dict[str, Any]]], k: int) -> list[dict[str, Any]]:
    rows = []
    for query_id, baseline in maps[run_ids[0]].items():
        queries = [maps[run_id][query_id] for run_id in run_ids]
        ranks = [query["target_rank"] for query in queries]
        top_sets = [{item["image_id"] for item in query["top_results"][:k]} for query in queries]
        pairwise = [len(left & right) / len(left | right) for index, left in enumerate(top_sets) for right in top_sets[index + 1:] if left | right]
        distractors: dict[str, dict[str, Any]] = {}
        for run_id, query in zip(run_ids, queries, strict=True):
            for item in query["top_results"][:k]:
                if item["image_id"] != query["target_id"]:
                    entry = distractors.setdefault(item["image_id"], {"run_ids": [], "ranks": []})
                    if run_id not in entry["run_ids"]:
                        entry["run_ids"].append(run_id)
                        entry["ranks"].append(item["rank"])
        common = sorted(({"image_id": image_id, "run_count": len(entry["run_ids"]), "run_ids": entry["run_ids"], "best_rank": min(entry["ranks"]), "mean_rank": mean(entry["ranks"])} for image_id, entry in distractors.items()), key=lambda item: (-item["run_count"], item["best_rank"], item["image_id"]))[:10]
        rows.append({"query_id": query_id, "category": baseline["category"], "target_rank": {run_id: query["target_rank"] for run_id, query in zip(run_ids, queries, strict=True)}, "consensus_fail_fraction": sum(rank > k for rank in ranks) / len(ranks), "min_rank": min(ranks), "max_rank": max(ranks), "mean_rank": mean(ranks), "median_rank": median(ranks), "rank_range": max(ranks) - min(ranks), "rank_std": pstdev(ranks) if len(ranks) > 1 else 0.0, "mean_topk_jaccard": mean(pairwise) if pairwise else 1.0, "common_distractors": common})
    return rows


def analysis_rows(runs: list[ResultRun], k: int) -> list[dict[str, Any]]:
    require_top_k(runs, k)
    aligned = aligned_queries(runs)
    maps = {run.run.run_id: {query_id: query.model_dump() for query_id, queries in aligned.items() for query in [queries[index]]} for index, run in enumerate(runs)}
    return _rows([run.run.run_id for run in runs], maps, k)


def analysis_rows_from_maps(run_ids: list[str], maps: dict[str, dict[str, dict[str, Any]]], k: int) -> list[dict[str, Any]]:
    return _rows(run_ids, maps, k)


def failure_jaccard_from_maps(run_ids: list[str], maps: dict[str, dict[str, dict[str, Any]]], k: int) -> list[dict[str, Any]]:
    failures = {run_id: {query_id for query_id, query in maps[run_id].items() if query["target_rank"] > k} for run_id in run_ids}
    return [{"left_run_id": left, "right_run_id": right, "jaccard": len(failures[left] & failures[right]) / len(failures[left] | failures[right]) if failures[left] | failures[right] else 1.0} for left in run_ids for right in run_ids]


def failure_jaccard(runs: list[ResultRun], k: int) -> list[dict[str, Any]]:
    require_top_k(runs, k)
    aligned = aligned_queries(runs)
    run_ids = [run.run.run_id for run in runs]
    maps = {run_id: {query_id: queries[index].model_dump() for query_id, queries in aligned.items()} for index, run_id in enumerate(run_ids)}
    return failure_jaccard_from_maps(run_ids, maps, k)


def research_interest(row: dict, k: int) -> dict:
    severity = mean(math.log1p(rank) for rank in row["target_rank"].values())
    return {"consensus": row["consensus_fail_fraction"], "severity": severity, "disagreement": row["rank_std"] / max(1.0, row["mean_rank"]), "k": k}
