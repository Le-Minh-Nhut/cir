"""Compute exact-rank composition metrics without coupling them to retrieval backends."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

SHORTCUT_K = 10
_CONDITIONS = ("multimodal", "text_only_black_image", "image_only_empty_text")


def _validate_rank(rank: int) -> None:
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError(f"rank must be a positive 1-based integer, got {rank}")


def hit_at_k(rank: int, k: int) -> int:
    """Return a binary hit for a positive 1-based rank and positive cutoff."""
    _validate_rank(rank)
    if k < 1:
        raise ValueError(f"k must be positive, got {k}")
    return int(rank <= k)


def ndcg_full(rank: int) -> float:
    """Return single-positive full-catalogue nDCG as a fraction in [0, 1]."""
    _validate_rank(rank)
    return 1.0 / math.log2(rank + 1)


def reciprocal_rank(rank: int) -> float:
    """Return reciprocal rank for a positive 1-based exact rank."""
    _validate_rank(rank)
    return 1.0 / rank


@dataclass(frozen=True)
class ConditionMetrics:
    condition: str
    query_count: int
    scored_count: int
    coverage: float
    recall_at_10: float | None
    recall_at_50: float | None
    ndcg_full: float | None
    mrr: float | None
    unavailable_reason: str | None


def compute_condition_metrics(condition: str, ranks: Mapping[str, int | None]) -> ConditionMetrics:
    """Summarize scored queries; Recall is percent, nDCG/MRR are fractions.

    Unscored queries reduce coverage and are excluded from score means. Empty cohorts have
    coverage 1.0, since no query was expected but left unscored. Exact-rank ties are resolved
    by the rank producer; this module consumes ranks and applies no hidden tie-break.
    """
    scored = [rank for rank in ranks.values() if rank is not None]
    for rank in scored:
        _validate_rank(rank)
    query_count = len(ranks)
    scored_count = len(scored)
    if not scored_count:
        reason = "empty query cohort" if not query_count else "no queries have an exact rank"
        return ConditionMetrics(condition, query_count, 0, 1.0 if not query_count else 0.0,
                                None, None, None, None, reason)
    return ConditionMetrics(
        condition,
        query_count,
        scored_count,
        scored_count / query_count,
        100.0 * sum(hit_at_k(rank, 10) for rank in scored) / scored_count,
        100.0 * sum(hit_at_k(rank, 50) for rank in scored) / scored_count,
        sum(ndcg_full(rank) for rank in scored) / scored_count,
        sum(reciprocal_rank(rank) for rank in scored) / scored_count,
        None,
    )


@dataclass(frozen=True)
class CompositionResult:
    multimodal: ConditionMetrics
    text_only: ConditionMetrics
    image_only: ConditionMetrics
    composition_gap_ndcg: float | None
    gap_unavailable_reason: str | None
    mrr_gap: float | None
    shortcut_outcomes: dict[str, int]
    gap_cohort_count: int
    gap_multimodal_ndcg: float | None
    gap_text_only_ndcg: float | None
    gap_image_only_ndcg: float | None


def compose(ranks: Mapping[str, Mapping[str, int | None]], query_count: int | None = None) -> CompositionResult:
    """Aggregate condition metrics and a normalized nDCG gap on the common scored cohort.

    Per-condition metrics retain their own scored cohort and coverage. `gap_*_ndcg` values
    expose the same common cohort used by `composition_gap_ndcg`. Extra known cohort members
    absent from all mappings reduce coverage only; they never enter any mean. Missing condition/qid
    ranks stay unscored, never misses. Ties are resolved by the rank producer.
    """
    qids = set().union(*(set(ranks.get(condition, {})) for condition in _CONDITIONS))
    cohort_count = len(qids) if query_count is None else query_count
    if cohort_count < 0:
        raise ValueError(f"query_count must not be negative, got {cohort_count}")
    if cohort_count < len(qids):
        raise ValueError(f"query_count {cohort_count} is smaller than {len(qids)} distinct qids")

    def cohort_ranks(condition: str) -> dict[str, int | None]:
        condition_ranks = ranks.get(condition, {})
        return {qid: condition_ranks.get(qid) for qid in qids}

    mm_ranks = cohort_ranks(_CONDITIONS[0])
    text_ranks = cohort_ranks(_CONDITIONS[1])
    image_ranks = cohort_ranks(_CONDITIONS[2])
    mm = compute_condition_metrics(_CONDITIONS[0], mm_ranks)
    text = compute_condition_metrics(_CONDITIONS[1], text_ranks)
    image = compute_condition_metrics(_CONDITIONS[2], image_ranks)
    if cohort_count != len(qids):
        mm = _with_query_count(mm, cohort_count)
        text = _with_query_count(text, cohort_count)
        image = _with_query_count(image, cohort_count)

    common = [qid for qid in qids if mm_ranks[qid] is not None and text_ranks[qid] is not None
              and image_ranks[qid] is not None]
    gap_count = len(common)
    gap_mm = gap_text = gap_image = gap = mrr_gap = None
    if not qids and cohort_count == 0:
        reason = "empty query cohort"
    elif not all(condition in ranks for condition in _CONDITIONS):
        missing = [condition for condition in _CONDITIONS if condition not in ranks]
        reason = f"missing condition(s): {', '.join(missing)}"
    elif not common:
        reason = "no queries have exact ranks in all three conditions"
    else:
        gap_mm = sum(ndcg_full(mm_ranks[qid]) for qid in common) / gap_count
        gap_text = sum(ndcg_full(text_ranks[qid]) for qid in common) / gap_count
        gap_image = sum(ndcg_full(image_ranks[qid]) for qid in common) / gap_count
        mm_mrr = sum(reciprocal_rank(mm_ranks[qid]) for qid in common) / gap_count
        text_mrr = sum(reciprocal_rank(text_ranks[qid]) for qid in common) / gap_count
        image_mrr = sum(reciprocal_rank(image_ranks[qid]) for qid in common) / gap_count
        mrr_gap = mm_mrr - max(text_mrr, image_mrr)
        if gap_mm == 0.0:
            reason = "multimodal mean nDCG is zero"
        else:
            gap = 1.0 - max(gap_text, gap_image) / gap_mm
            reason = None

    all_ranks = {condition: ranks.get(condition, {}) for condition in _CONDITIONS}
    outcomes = shortcut_outcomes(all_ranks)
    outcomes["neither_scored"] += cohort_count - len(qids)
    return CompositionResult(mm, text, image, gap, reason, mrr_gap, outcomes,
                             gap_count, gap_mm, gap_text, gap_image)


def _with_query_count(metrics: ConditionMetrics, query_count: int) -> ConditionMetrics:
    from dataclasses import replace

    return replace(metrics, query_count=query_count,
                   coverage=metrics.scored_count / query_count if query_count else 1.0)


def shortcut_outcomes(ranks: Mapping[str, Mapping[str, int | None]], k: int = SHORTCUT_K) -> dict[str, int]:
    """Count mutually exclusive K-hit outcomes; `neither_scored` counts excluded qids.

    A qid missing any condition rank is excluded from outcome bins and counted in
    `neither_scored`; it is not treated as a retrieval miss. Ties are producer-resolved.
    """
    if k < 1:
        raise ValueError(f"k must be positive, got {k}")
    qids = set().union(*(set(ranks.get(condition, {})) for condition in _CONDITIONS))
    outcomes = {name: 0 for name in
                ("text_only", "image_only", "both", "composition_required", "unresolved", "neither_scored")}
    for qid in qids:
        values = [ranks.get(condition, {}).get(qid) for condition in _CONDITIONS]
        if any(rank is None for rank in values):
            outcomes["neither_scored"] += 1
            continue
        mm_hit, text_hit, image_hit = (hit_at_k(rank, k) for rank in values)
        if text_hit and image_hit:
            outcome = "both"
        elif text_hit:
            outcome = "text_only"
        elif image_hit:
            outcome = "image_only"
        elif mm_hit:
            outcome = "composition_required"
        else:
            outcome = "unresolved"
        outcomes[outcome] += 1
    return outcomes


def aggregate_subsets(per_query: Mapping[str, Mapping[str, int | None]],
                      subsets: Mapping[str, Sequence[str]]) -> dict[str, CompositionResult]:
    """Compute query-weighted subset summaries from one already-scored rank table."""
    return {
        name: compose(
            {condition: {qid: condition_ranks.get(qid) for qid in qids}
             for condition, condition_ranks in per_query.items()},
            query_count=len(qids),
        )
        for name, qids in subsets.items()
    }


def paired_bootstrap_gap(deltas: Sequence[float], *, seed: int, iterations: int = 2000,
                         confidence: float = 0.95) -> tuple[float, float]:
    """Return a deterministic percentile interval for paired deltas.

    Convenience interval only; it is not a statistical claim. Values must already be paired
    per query. Percentile interpolation uses the sorted bootstrap means at p * (n - 1).
    """
    if not deltas:
        raise ValueError("deltas must not be empty")
    if iterations < 1:
        raise ValueError(f"iterations must be positive, got {iterations}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be between 0 and 1, got {confidence}")
    rng = random.Random(seed)
    size = len(deltas)
    means = sorted(sum(deltas[rng.randrange(size)] for _ in range(size)) / size
                   for _ in range(iterations))
    lower = (1.0 - confidence) / 2.0
    return _percentile(means, lower), _percentile(means, 1.0 - lower)


def _percentile(values: Sequence[float], probability: float) -> float:
    position = probability * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)
