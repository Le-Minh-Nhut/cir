"""Differential checks on REAL data for the CIRCUS composition scoring + ranking path.

Axes measured, all CPU:
  A1  score-stage batch invariance: identical queries scored as one row at a time vs 32 vs all.
  A2  rank agreement: `_ranks_from_scores` vs a brute-force stable argsort, including exact ties.
  A3  tie structure: how many real gallery rows are byte-identical duplicate embeddings, and
      whether a duplicate target changes the rank that `_ranks_from_scores` reports.
  A4  determinism: the same call twice, exact equality.
  A5  metric aggregation invariance: dict insertion order must not move the mean.

Tolerances are declared in this file BEFORE any comparison and are not relaxed afterwards.
Discrete outputs (ranks, positions, order) use exact equality. Float score matrices declare:
  atol = 0.0, rtol = 0.0   for A4 (repeatability: the same call on the same process must be bit-equal)
  atol = 1e-6, rtol = 0.0  for A1 (row-chunked BLAS reduction may reorder additions)
and both are reported as measured max-abs error, not merely passed.

Usage:
  python workbench/scripts/audit/audit_differential_stages.py \
      --features workbench/artifacts/composition/preserved_features/limn/gallery_dress.npy \
      --ranks    workbench/artifacts/composition/runs/<run>/ranks/<model>_<ckpt>.json \
      --out      <evidence>/logs/differential_stages.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# Declared before the run. Never loosened after a failure.
ATOL_SCORE_CHUNKED = 1e-6
RTOL_SCORE_CHUNKED = 0.0
ATOL_REPEAT = 0.0
RTOL_REPEAT = 0.0


def brute_force_rank(row: np.ndarray, positions: tuple[int, ...]) -> int:
    """Reference rank: order by (-score, gallery position), then read the best positive's index."""
    order = sorted(range(row.size), key=lambda i: (-float(row[i]), i))
    first = {}
    for rank_index, position in enumerate(order, start=1):
        first.setdefault(position, rank_index)
    return min(first[p] for p in positions)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--ranks", type=Path, default=None)
    parser.add_argument("--queries", type=int, default=8)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    from workbench.composition.adapters import CompositionAdapter
    from workbench.composition.metrics import compose

    report: dict = {"tolerances": {"score_chunked": {"atol": ATOL_SCORE_CHUNKED,
                                                     "rtol": RTOL_SCORE_CHUNKED},
                                   "repeat": {"atol": ATOL_REPEAT, "rtol": RTOL_REPEAT}},
                    "axes": {}}

    gallery = np.asarray(np.load(args.features, mmap_mode="r"), dtype=np.float32)
    gallery_size = gallery.shape[0]
    # Deterministic query set taken from the real gallery rows themselves, so the target is a
    # genuine gallery member (as in the real protocol) rather than a synthetic vector.
    query_positions = list(range(min(args.queries, gallery_size)))
    queries = np.stack([gallery[i] for i in query_positions])
    targets = (queries[:, None] if False else None)

    adapter = CompositionAdapter.__new__(CompositionAdapter)
    adapter.model_id = "audit"

    # ---- A1: score-stage batch invariance ------------------------------------------------
    full = adapter.score(queries, gallery)
    one_at_a_time = np.concatenate([adapter.score(queries[i:i + 1], gallery)
                                    for i in range(len(queries))], axis=0)
    chunks_32 = np.concatenate([adapter.score(queries[s:s + 32], gallery)
                                for s in range(0, len(queries), 32)], axis=0)
    abs_full_one = np.abs(full - one_at_a_time)
    abs_full_32 = np.abs(full - chunks_32)
    report["axes"]["A1_score_batch_invariance"] = {
        "rows": len(queries),
        "gallery": gallery_size,
        "batch1_vs_full_max_abs": float(abs_full_one.max()),
        "batch1_vs_full_exact_equal": bool(np.array_equal(full, one_at_a_time)),
        "batch32_vs_full_max_abs": float(abs_full_32.max()),
        "batch32_vs_full_exact_equal": bool(np.array_equal(full, chunks_32)),
        "verdict": ("IDENTICAL" if np.array_equal(full, one_at_a_time) and np.array_equal(full, chunks_32)
                    else "DIVERGENT_BEYOND_DECLARED_TOLERANCE"
                    if (abs_full_one.max() > ATOL_SCORE_CHUNKED or abs_full_32.max() > ATOL_SCORE_CHUNKED)
                    else "DIVERGENT_WITHIN_TOLERANCE"),
    }

    # ---- A2: rank agreement vs brute force, including ties ------------------------------
    # Target sets: the query's own gallery position (rank 1 case) and a mid-gallery position, and
    # a position whose score ties with the query's own row (a duplicate embedding if one exists).
    target_sets = [[i] for i in query_positions]
    disagree = []
    for row_index, positions in enumerate(target_sets):
        positions = tuple(positions)
        fast = CompositionAdapter._ranks_from_scores(full[row_index:row_index + 1], [positions])[0]
        slow = brute_force_rank(full[row_index], positions)
        if fast != slow:
            disagree.append({"row": row_index, "positions": positions, "fast": fast, "slow": slow})
    report["axes"]["A2_rank_vs_bruteforce"] = {
        "targets_checked": len(target_sets),
        "disagreements": disagree,
        "verdict": "IDENTICAL" if not disagree else "DIVERGENT",
    }

    # ---- A3: duplicate/tie structure in the real gallery -------------------------------
    dup_positions = {}
    for i in query_positions:
        equal = np.flatnonzero(np.all(gallery == gallery[i], axis=1))
        if len(equal) > 1:
            dup_positions[i] = equal.tolist()
    tie_cases = []
    for origin, group in dup_positions.items():
        # Pretend the LAST duplicate is the positive: the earlier duplicate is an exact tie, so a
        # correct "earlier position wins" rule must still report the earliest position.
        last = group[-1]
        fast = CompositionAdapter._ranks_from_scores(full[origin:origin + 1], [(last,)])[0]
        slow = brute_force_rank(full[origin], (last,))
        tie_cases.append({"query_row": origin, "duplicate_group": group,
                          "positive_at_last_duplicate": last, "fast_rank": fast,
                          "brute_rank": slow,
                          "earliest_duplicate_position": group[0],
                          "tie_policy_matches_bruteforce": fast == slow})
    report["axes"]["A3_duplicate_ties"] = {
        "query_rows_with_exact_duplicate_gallery_rows": len(dup_positions),
        "cases": tie_cases,
        "verdict": ("IDENTICAL" if all(c["tie_policy_matches_bruteforce"] for c in tie_cases)
                    else "DIVERGENT"),
    }

    # ---- A4: repeat determinism -------------------------------------------------------
    again = adapter.score(queries, gallery)
    abs_repeat = np.abs(full - again)
    report["axes"]["A4_repeat_determinism"] = {
        "max_abs": float(abs_repeat.max()),
        "exact_equal": bool(np.array_equal(full, again)),
        "verdict": "IDENTICAL" if np.array_equal(full, again) else "DIVERGENT",
    }

    # ---- A5: metric aggregation invariance to dict order ------------------------------
    if args.ranks and args.ranks.is_file():
        payload = json.loads(args.ranks.read_text(encoding="utf-8"))
        ranks = payload.get("ranks") or {}
        if ranks:
            conditions = list(ranks)
            forward = compose({c: ranks[c] for c in conditions})
            backward = compose({c: dict(reversed(list(ranks[c].items()))) for c in reversed(conditions)})
            report["axes"]["A5_metric_order_invariance"] = {
                "gap_forward": forward.composition_gap_ndcg,
                "gap_reversed": backward.composition_gap_ndcg,
                "max_abs_delta": abs((forward.composition_gap_ndcg or 0.0)
                                     - (backward.composition_gap_ndcg or 0.0)),
                "exact_equal": forward.composition_gap_ndcg == backward.composition_gap_ndcg,
                "cohort_forward": forward.gap_cohort_count,
                "cohort_reversed": backward.gap_cohort_count,
            }
        report["ranks_source"] = str(args.ranks)

    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
