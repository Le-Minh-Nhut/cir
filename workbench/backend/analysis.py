from __future__ import annotations

import math
from statistics import mean, median, pstdev

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
    baseline = {query.query_id: query for query in runs[0].queries}
    baseline_ids = set(baseline)
    for run in runs[1:]:
        candidate = {query.query_id: query for query in run.queries}
        candidate_ids = set(candidate)
        missing, extra = sorted(baseline_ids - candidate_ids), sorted(candidate_ids - baseline_ids)
        if missing or extra:
            raise WorkbenchError("query_alignment_mismatch", "Selected runs do not have an identical query universe.", {"baseline_run_id": runs[0].run.run_id, "run_id": run.run.run_id, "missing_query_count": len(missing), "missing_query_ids": missing[:20], "extra_query_count": len(extra), "extra_query_ids": extra[:20]})
        for query_id, expected in baseline.items():
            actual = candidate[query_id]
            mismatches = {field: {"expected": getattr(expected, field), "actual": getattr(actual, field)} for field in ("category", "annotation_index", "reference_id", "target_id", "raw_captions") if getattr(expected, field) != getattr(actual, field)}
            if mismatches:
                raise WorkbenchError("canonical_query_mismatch", "Canonical query fields differ for the same query_id.", {"query_id": query_id, "baseline_run_id": runs[0].run.run_id, "run_id": run.run.run_id, "mismatches": mismatches})
    return {query_id: [next(query for query in run.queries if query.query_id == query_id) for run in runs] for query_id in baseline}


def analysis_rows(runs: list[ResultRun], k: int) -> list[dict]:
    require_top_k(runs, k)
    rows = []
    for query_id, queries in aligned_queries(runs).items():
        ranks = [query.target_rank for query in queries]
        top_sets = [{result.image_id for result in query.top_results[:k]} for query in queries]
        pairwise = [len(left & right) / len(left | right) for index, left in enumerate(top_sets) for right in top_sets[index + 1:] if left | right]
        distractors: dict[str, list[int]] = {}
        for query in queries:
            for result in query.top_results[:k]:
                if result.image_id != query.target_id:
                    distractors.setdefault(result.image_id, []).append(result.rank)
        common = sorted(({"image_id": image_id, "run_count": len(ranks), "best_rank": min(ranks), "mean_rank": mean(ranks)} for image_id, ranks in distractors.items()), key=lambda item: (-item["run_count"], item["best_rank"], item["image_id"]))[:10]
        rows.append({"query_id": query_id, "category": queries[0].category, "target_rank": {run.run.run_id: query.target_rank for run, query in zip(runs, queries, strict=True)}, "consensus_fail_fraction": sum(rank > k for rank in ranks) / len(ranks), "min_rank": min(ranks), "max_rank": max(ranks), "mean_rank": mean(ranks), "median_rank": median(ranks), "rank_range": max(ranks) - min(ranks), "rank_std": pstdev(ranks) if len(ranks) > 1 else 0.0, "mean_topk_jaccard": mean(pairwise) if pairwise else 1.0, "common_distractors": common})
    return rows


def failure_jaccard(runs: list[ResultRun], k: int) -> list[dict]:
    aligned = aligned_queries(runs)
    failures = {run.run.run_id: {query_id for query_id, queries in aligned.items() if queries[index].target_rank > k} for index, run in enumerate(runs)}
    matrix = []
    for left in runs:
        for right in runs:
            union = failures[left.run.run_id] | failures[right.run.run_id]
            matrix.append({"left_run_id": left.run.run_id, "right_run_id": right.run.run_id, "jaccard": len(failures[left.run.run_id] & failures[right.run.run_id]) / len(union) if union else 1.0})
    return matrix


def cohort_metrics(runs: list[ResultRun], query_ids: list[str]) -> list[dict]:
    selected = set(query_ids)
    metrics = []
    for result in runs:
        ranks = [query.target_rank for query in result.queries if query.query_id in selected]
        metrics.append({"run_id": result.run.run_id, "query_count": len(ranks), **{f"r{k}": 100 * sum(rank <= k for rank in ranks) / len(ranks) if ranks else None for k in (1, 5, 10, 20, 50, 100)}, "mean_target_rank": mean(ranks) if ranks else None, "median_target_rank": median(ranks) if ranks else None})
    return metrics


def research_interest(row: dict, k: int) -> dict:
    severity = mean(math.log1p(rank) for rank in row["target_rank"].values())
    return {"consensus": row["consensus_fail_fraction"], "severity": severity, "disagreement": row["rank_std"] / max(1.0, row["mean_rank"]), "k": k}
