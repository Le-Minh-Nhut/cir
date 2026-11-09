"""Per-query rank diff between two runs of one model, from the rank artifacts.

    rank_diff.py <model> <old_run> <new_run> <checkpoint>

Prints per-condition aggregate metrics from each summary, and compares the per-query target
ranks query by query. Aggregate equality can hide reordered ranks; this cannot.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

ART = REPO / "workbench" / "artifacts" / "composition"


def ndcg(rank: int) -> float:
    from workbench.composition.metrics import ndcg_full
    return ndcg_full(rank)


def main() -> int:
    model, old_run, new_run, checkpoint = sys.argv[1:5]
    out: dict = {"model": model, "old_run": old_run, "new_run": new_run, "conditions": {}}
    old_ranks = json.loads((ART / "runs" / old_run / "ranks" / f"{model}_{checkpoint}.json").read_text())
    new_ranks = json.loads((ART / "runs" / new_run / "ranks" / f"{model}_{checkpoint}.json").read_text())
    for condition in sorted(old_ranks["ranks"]):
        a = old_ranks["ranks"][condition]
        b = new_ranks["ranks"][condition]
        shared = sorted(set(a) & set(b))
        changed, deltas = [], []
        for qid in shared:
            ra, rb = a[qid], b[qid]
            if ra != rb:
                changed.append((qid, ra, rb))
                if ra is not None and rb is not None:
                    deltas.append(rb - ra)
        only_a = sorted(set(a) - set(b))
        only_b = sorted(set(b) - set(a))
        ndcg_delta = [ndcg(b[q]) - ndcg(a[q]) for q in shared if a[q] is not None and b[q] is not None]
        out["conditions"][condition] = {
            "queries_compared": len(shared),
            "only_old": len(only_a), "only_new": len(only_b),
            "changed_rank_queries": len(changed),
            "changed_examples": changed[:10],
            "max_abs_delta_rank": max((abs(d) for d in deltas), default=0),
            "mean_delta_rank": (sum(deltas) / len(deltas)) if deltas else 0.0,
            "sum_ndcg_delta": sum(ndcg_delta),
            "per_query_ndcg_identical": all(abs(v) == 0.0 for v in ndcg_delta),
            "bit_identical_ranks": not changed and not only_a and not only_b,
        }
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
