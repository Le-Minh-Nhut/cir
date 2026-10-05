from __future__ import annotations

import math
from collections import Counter
from statistics import mean, median, pstdev

from workbench.backend.schemas.results import ResultRun


def require_same_protocol(runs: list[ResultRun]) -> str:
    protocols = {run.run.protocol_id for run in runs}
    if len(protocols) != 1:
        raise ValueError("Cross-protocol consensus statistics are invalid. Select runs from the same FashionIQ protocol.")
    return protocols.pop()


def _aligned(runs: list[ResultRun]) -> dict[str, list]:
    require_same_protocol(runs)
    rows: dict[str, list] = {}
    for run in runs:
        for query in run.queries:
            rows.setdefault(query.query_id, []).append(query)
    return {query_id: queries for query_id, queries in rows.items() if len(queries) == len(runs)}


def analysis_rows(runs: list[ResultRun], k: int) -> list[dict]:
    if not 1 <= k <= 100:
        raise ValueError("K must be between 1 and 100")
    rows = []
    for query_id, queries in _aligned(runs).items():
        ranks = [query.target_rank for query in queries]
        top_sets = [{result.image_id for result in query.top_results[:k]} for query in queries]
        pairwise = [len(left & right) / len(left | right) for index, left in enumerate(top_sets) for right in top_sets[index + 1:] if left | right]
        distractors = Counter(result.image_id for query in queries for result in query.top_results[:k] if result.image_id != query.target_id)
        rows.append({"query_id": query_id, "category": queries[0].category, "target_rank": {run.run.model_id: query.target_rank for run, query in zip(runs, queries, strict=True)}, "consensus_fail_fraction": sum(rank > k for rank in ranks) / len(ranks), "min_rank": min(ranks), "max_rank": max(ranks), "mean_rank": mean(ranks), "median_rank": median(ranks), "rank_range": max(ranks) - min(ranks), "rank_std": pstdev(ranks) if len(ranks) > 1 else 0.0, "mean_topk_jaccard": mean(pairwise) if pairwise else 1.0, "common_distractors": [{"image_id": image_id, "models": count} for image_id, count in distractors.most_common(10)]})
    return rows


def failure_jaccard(runs: list[ResultRun], k: int) -> list[dict]:
    aligned = _aligned(runs)
    failures = {run.run.model_id: {query_id for query_id, queries in aligned.items() if queries[index].target_rank > k} for index, run in enumerate(runs)}
    matrix = []
    for left in runs:
        for right in runs:
            union = failures[left.run.model_id] | failures[right.run.model_id]
            matrix.append({"left_model_id": left.run.model_id, "right_model_id": right.run.model_id, "jaccard": len(failures[left.run.model_id] & failures[right.run.model_id]) / len(union) if union else 1.0})
    return matrix


def research_interest(row: dict, k: int) -> dict:
    severity = mean(math.log1p(rank) for rank in row["target_rank"].values())
    return {"consensus": row["consensus_fail_fraction"], "severity": severity, "disagreement": row["rank_std"] / max(1.0, row["mean_rank"]), "k": k}
