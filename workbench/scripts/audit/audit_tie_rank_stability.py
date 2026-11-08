"""Do score-stage float differences between batch sizes flip real ranks?

The score matrix is computed in query batches of `batch_size` (`runner.py`, rank loop), and
`CompositionAdapter.score` normalizes and matmuls per call. Row-chunked BLAS can reorder additions,
so a query scored alone and the same query scored in a batch of 32 can differ in the last bits.

A 6e-7 score difference is harmless UNLESS the gallery contains an exact score tie at the decision
boundary -- which it does: the released gallery has byte-identical duplicate images (measured,
851 groups / 2386 rows in the LIMN dress matrix). `_ranks_from_scores` breaks ties by "earlier
gallery position", so an epsilon of float noise is enough to swap two tied candidates and move a
reported rank.

This script measures that on real data:
  * query rows = gallery rows that belong to a duplicate group (worst case, guaranteed ties)
  * positive  = a position inside the same duplicate group
  * compare the rank from batch-1 scoring against the rank from batch-32 scoring

Tolerances declared BEFORE the run: ranks are discrete, so the tolerance is EXACT equality.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def duplicate_groups(matrix: np.ndarray, limit: int) -> list[list[int]]:
    """Position groups whose rows are byte-identical, largest first."""
    import hashlib
    from collections import defaultdict

    buckets: dict[bytes, list[int]] = defaultdict(list)
    for position in range(matrix.shape[0]):
        row = np.ascontiguousarray(matrix[position], dtype=np.float32)
        buckets[hashlib.sha1(row.tobytes()).digest()].append(position)
    groups = [positions for positions in buckets.values() if len(positions) > 1]
    groups.sort(key=len, reverse=True)
    return groups[:limit]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=25)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    from workbench.composition.adapters import CompositionAdapter

    report: dict = {"tolerances": {"ranks": "EXACT equality (discrete output)"}, "axes": {}}
    gallery = np.asarray(np.load(args.features, mmap_mode="r"), dtype=np.float32)
    adapter = CompositionAdapter.__new__(CompositionAdapter)
    adapter.model_id = "audit"

    groups = duplicate_groups(gallery, args.groups)
    query_positions = [group[0] for group in groups]
    positive_positions = [[group[-1]] for group in groups]  # last duplicate: earliest one must win ties
    queries = np.stack([gallery[p] for p in query_positions])

    full = adapter.score(queries, gallery)
    single = np.concatenate([adapter.score(queries[i:i + 1], gallery)
                             for i in range(len(queries))], axis=0)
    rank_full = CompositionAdapter._ranks_from_scores(full, positive_positions)
    rank_single = CompositionAdapter._ranks_from_scores(single, positive_positions)

    score_delta = float(np.abs(full - single).max())
    flips = [{"query_gallery_position": qp, "duplicate_group": groups[i][:8],
              "positive_position": positive_positions[i][0],
              "rank_scored_in_batch": rank_single[i], "rank_scored_alone": rank_full[i],
              "score_max_abs_delta": float(np.abs(full[i] - single[i]).max())}
             for i, qp in enumerate(query_positions) if rank_single[i] != rank_full[i]]

    # How many gallery rows sit exactly at the tie boundary for each query, i.e. how exposed this is.
    tie_exposure = []
    for i, qp in enumerate(query_positions):
        row = full[i]
        group = groups[i]
        tie_exposure.append({
            "query_gallery_position": qp,
            "duplicate_group_size": len(group),
            "identical_scores_in_group": int(np.count_nonzero(row[group] == row[group[0]])),
            "best_rank_alone": rank_full[i],
            "best_rank_batched": rank_single[i],
        })

    report["axes"]["duplicate_tie_rank_stability"] = {
        "query_rows_used": len(query_positions),
        "score_max_abs_delta_between_batch_sizes": score_delta,
        "ranks_that_flipped": len(flips),
        "flips": flips,
        "tie_exposure": tie_exposure,
        "verdict": ("STABLE" if not flips else "RANK_FLIP_UNDER_BATCH_CHANGE"),
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
