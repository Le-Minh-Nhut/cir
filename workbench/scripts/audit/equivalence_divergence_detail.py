"""Characterize the order-only Top-K divergences and summarize margins, from the saved scan.

Answers two questions the aggregate counters cannot:

  1. For each order-only divergence, are the two swapped gallery positions an EXACT tie under the
     shipped path? If yes, the divergence is purely a tie-policy interaction at score equality, and
     the window is the same SET in a different order -- which the gate still calls a difference.
  2. Across every changed rank, what is the distribution of (shipped tie multiplicity, score margin,
     rank depth), and do the two fast paths agree with each other everywhere?
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/heheboiz/data/cir")
sys.path.insert(0, str(REPO))
RUN = REPO / "workbench/artifacts/composition/runs/497c25c57fbeba885ecfb9d1"
FEATURES = RUN / "work/habit/features"
OUT = REPO / "workbench/docs/reviews/20261108T153045-audit-equivalence"


def score_old(q, t):
    return np.einsum("nd,gqd->ngq", q, t).max(axis=-1)


def score_gemm(q, t, chunk=1024):
    n, _d = q.shape
    g, tokens, _ = t.shape
    scores = np.empty((n, g), dtype=np.float32)
    for start in range(0, g, chunk):
        stop = min(start + chunk, g)
        block = np.asarray(t[start:stop], dtype=np.float32)
        best = np.full((n, stop - start), -np.inf, dtype=np.float32)
        for token in range(tokens):
            np.maximum(best, q @ block[:, token, :].T, out=best)
        scores[:, start:stop] = best
    return scores


def main() -> int:
    scan = json.loads((OUT / "fastpath_vs_stored.json").read_text())
    manifest = json.loads((OUT / "query_manifest.json").read_text())
    qids = list(manifest["query_order"])
    index_of = {qid: index for index, qid in enumerate(qids)}
    gallery = np.load(FEATURES / "gallery_global.npy", mmap_mode="r")
    report: dict = {"order_only_divergences": {}, "changed_rank_margins": {},
                    "fastpath_agreement": {}}
    for condition, payload in scan["conditions"].items():
        features = np.load(FEATURES / f"query_{condition}_global.npy", mmap_mode="r")
        # ---- 1. order-only divergences: are the swapped rows exactly tied under the shipped path?
        rows = []
        for example in payload["paths"]["opt"]["windows"]["order_examples"]:
            index = index_of[example["qid"]]
            query = np.asarray(features[index:index + 1], dtype=np.float32)
            shipped = score_old(query, gallery)[0]
            fast = score_gemm(query, gallery)[0]
            first, second = example["old"][0], example["old"][1]
            rows.append({
                "qid": example["qid"],
                "first_divergent_position": example["first_divergent_position"],
                "shipped_order": [first, second],
                "fast_order": [second, first],
                "shipped_scores": [float(shipped[first]), float(shipped[second])],
                "fast_scores": [float(fast[first]), float(fast[second])],
                "shipped_exactly_equal": bool(shipped[first] == shipped[second]),
                "fast_exactly_equal": bool(fast[first] == fast[second]),
                "shipped_positions_ascending": first < second,
                "kind": ("exact tie broken by earlier gallery position upstream; the fast path's "
                         "1-ULP difference removes the tie and orders by score"
                         if shipped[first] == shipped[second] else
                         "NOT a shipped-path tie: a genuine score reordering"),
            })
        report["order_only_divergences"][condition] = {
            "count": len(rows),
            "all_are_shipped_exact_ties": all(r["shipped_exactly_equal"] for r in rows),
            "all_sets_identical": payload["paths"]["opt"]["windows"]["set_differs"] == 0,
            "rows": rows,
        }
        # ---- 2. margin distribution over every changed rank, and fast-path agreement
        opt_changed = {entry["qid"]: entry for entry in payload["paths"]["opt"]["changed_all"]}
        gemm_changed = {entry["qid"]: entry for entry in payload["paths"]["gemm1024"]["changed_all"]}
        agree = opt_changed.keys() == gemm_changed.keys() and all(
            opt_changed[qid]["new"] == gemm_changed[qid]["new"] for qid in opt_changed)
        report["fastpath_agreement"][condition] = {
            "opt_changed": len(opt_changed), "gemm1024_changed": len(gemm_changed),
            "identical_changed_query_sets_and_new_ranks": agree,
        }
        margins = payload["paths"]["opt"]["margins"]
        tie_interactions = sum(1 for m in margins if m.get("tie_interaction"))
        report["changed_rank_margins"][condition] = {
            "changed": payload["paths"]["opt"]["rank_changed"],
            "margins_captured": len(margins),
            "tie_interactions": tie_interactions,
            "non_tie_margins": len(margins) - tie_interactions,
            "max_abs_score_margin": max((abs(m["score_margin_shipped_minus_fast"]) for m in margins
                                         if m.get("score_margin_shipped_minus_fast") is not None),
                                        default=0.0),
            "rank_delta_histogram": payload["paths"]["opt"]["rank_delta_histogram"],
            "depth_histogram": payload["paths"]["opt"]["depth_histogram"],
            "examples": margins[:12],
        }
        print(f"[{condition}] order_div={len(rows)} all_ties="
              f"{report['order_only_divergences'][condition]['all_are_shipped_exact_ties']} "
              f"agree={agree}", flush=True)
    Path(OUT / "divergence_detail.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps({"out": str(OUT / 'divergence_detail.json')}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
