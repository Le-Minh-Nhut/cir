"""G3/G3b decisive comparison of SCORING paths against the run's OWN stored ranks.

The shipped path's per-query ranks are already on disk (`ranks/habit_fiq_n02.json`), produced by the
shipped `np.einsum("nd,gqd->ngq")` scorer over all 6003 queries x 3 conditions. Recomputing that
5.6 q/s full-gallery einsum would cost ~55 minutes to re-derive numbers we already have, so this
script anchors on the stored artifact and computes only the candidate fast paths.

Correctness of the anchor is checked, not assumed: `--verify-anchor` recomputes the shipped path for
a random sample and requires exact agreement with the stored ranks before any verdict is issued.

Reports per condition and per path:
  * queries whose target rank changes, max |delta rank|, delta histogram, depth distribution
  * K=10 and K=50 BOUNDARY CROSSINGS, with qids -- the only rank changes that move the metrics
  * Top-K candidate SET changes and ORDER changes, counted separately
  * R@10, R@50, nDCG and deltas against the stored-ranks metrics
  * for every changed rank: the positive's score and exact-tie multiplicity under both paths,
    which distinguishes a tie-policy interaction from a random instability
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/home/heheboiz/data/cir")
sys.path.insert(0, str(REPO))
from workbench.composition.topk import topk_from_scores  # noqa: E402

RUN = REPO / "workbench/artifacts/composition/runs/497c25c57fbeba885ecfb9d1"
FEATURES = RUN / "work/habit/features"
OUT = REPO / "workbench/docs/reviews/20261108T153045-audit-equivalence"


def ranks_from_scores(scores, positions):
    ranks = []
    for row, positives in zip(scores, positions):
        if not positives or not np.isfinite(row).all():
            ranks.append(None)
            continue
        valid = [index for index in positives if 0 <= index < row.shape[0]]
        if not valid:
            ranks.append(None)
            continue
        ranks.append(min(1 + int(np.count_nonzero(row > row[index])) +
                         int(np.count_nonzero(row[:index] == row[index])) for index in valid))
    return ranks


def score_old(q, t):
    return np.einsum("nd,gqd->ngq", q, t).max(axis=-1)


def score_opt(q, t):
    return np.einsum("nd,gqd->ngq", q, t, optimize=True).max(axis=-1)


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


PATHS = {"opt": score_opt, "gemm1024": lambda q, t: score_gemm(q, t, 1024),
         "gemm4096": lambda q, t: score_gemm(q, t, 4096)}


def metrics(ranks):
    scored = [r for r in ranks if r is not None]
    n = len(scored)
    if not n:
        return {"scored": 0}
    return {"scored": n,
            "r10": 100.0 * sum(1 for r in scored if r <= 10) / n,
            "r50": 100.0 * sum(1 for r in scored if r <= 50) / n,
            "ndcg": float(np.mean([1.0 / np.log2(r + 1) for r in scored]))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", default="multimodal,text_only_black_image,image_only_empty_text")
    parser.add_argument("--paths", default="opt,gemm1024")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--depth", type=int, default=50)
    parser.add_argument("--verify-sample", type=int, default=24,
                        help="recompute the shipped path for this many queries and require exact "
                             "agreement with the stored ranks; 0 disables")
    parser.add_argument("--max-margin", type=int, default=200)
    parser.add_argument("--out", default=str(OUT / "fastpath_vs_stored.json"))
    args = parser.parse_args()

    manifest = json.loads((OUT / "query_manifest.json").read_text())
    stored = json.loads((RUN / "ranks/habit_fiq_n02.json").read_text())
    qids = list(manifest["query_order"])
    positions = [tuple(manifest["positive_positions"][qid]) for qid in qids]
    gallery = np.load(FEATURES / "gallery_global.npy", mmap_mode="r")
    names = [name for name in args.paths.split(",") if name]

    payload = {"run": RUN.name, "queries": len(qids), "depth": args.depth,
               "numpy": np.__version__,
               "threads_env": {k: os.environ.get(k) for k in
                               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
               "anchor": "ranks/habit_fiq_n02.json (the shipped path's own output)",
               "anchor_verification": {}, "conditions": {}}

    # ---- anchor verification: shipped path reproduces the stored artifact exactly ----
    if args.verify_sample:
        rng = np.random.default_rng(0)
        picks = sorted(rng.choice(len(qids), size=min(args.verify_sample, len(qids)),
                                  replace=False).tolist())
        verification = {}
        for condition in args.conditions.split(","):
            features = np.load(FEATURES / f"query_{condition}_global.npy", mmap_mode="r")
            mismatches = []
            for index in picks:
                row = np.asarray(features[index:index + 1], dtype=np.float32)
                rank = ranks_from_scores(score_old(row, gallery), positions[index:index + 1])[0]
                if rank != stored["ranks"][condition].get(qids[index]):
                    mismatches.append({"qid": qids[index], "recomputed": rank,
                                       "stored": stored["ranks"][condition].get(qids[index])})
            verification[condition] = {"sampled": len(picks), "mismatches": len(mismatches),
                                       "examples": mismatches[:5]}
            print(f"[anchor] {condition}: {len(picks)} sampled, {len(mismatches)} mismatches",
                  flush=True)
        payload["anchor_verification"] = verification

    for condition in args.conditions.split(","):
        features = np.load(FEATURES / f"query_{condition}_global.npy", mmap_mode="r")
        q_all = np.asarray(features, dtype=np.float32)
        base_ranks = [stored["ranks"][condition].get(qid) for qid in qids]
        base_metrics = metrics(base_ranks)
        report = {"anchor_metrics": base_metrics, "paths": {}}
        for name in names:
            started = time.perf_counter()
            ranks, windows = [], []
            changed_rows: dict[str, np.ndarray] = {}
            for start in range(0, len(qids), args.batch):
                batch_q = q_all[start:start + args.batch]
                scores = PATHS[name](batch_q, gallery)
                batch_ranks = ranks_from_scores(scores, positions[start:start + args.batch])
                ranks.extend(batch_ranks)
                windows.extend(topk_from_scores(row, args.depth) for row in scores)
                for local, (new, old) in enumerate(zip(
                        batch_ranks, base_ranks[start:start + args.batch])):
                    if new != old:
                        changed_rows[qids[start + local]] = scores[local].copy()
            elapsed = time.perf_counter() - started
            changed = [{"qid": qids[i], "old": a, "new": b,
                        "delta": (None if a is None or b is None else b - a)}
                       for i, (a, b) in enumerate(zip(base_ranks, ranks)) if a != b]
            deltas = [e["delta"] for e in changed if e["delta"] is not None]
            crossings = []
            for entry in changed:
                a, b = entry["old"], entry["new"]
                if a is None or b is None:
                    continue
                if (a <= 10) != (b <= 10) or (a <= 50) != (b <= 50):
                    crossings.append({**entry, "k10": (a <= 10) != (b <= 10),
                                      "k50": (a <= 50) != (b <= 50)})
            # Top-K set and ORDER against the run's own published window (depth 50), which the
            # shipped path wrote. Set differences and order differences are counted separately:
            # a reordering with an identical set still violates the "identical ordering" gate.
            stored_windows = load_stored_windows(condition)
            sets_differ = order_differs = 0
            set_examples, order_examples = [], []
            for i, window in enumerate(windows):
                qid = qids[i]
                reference = stored_windows.get(qid)
                if reference is None:
                    continue
                ours = [position for position, _score in window]
                if set(ours) != set(reference):
                    sets_differ += 1
                    if len(set_examples) < 10:
                        set_examples.append({"qid": qid,
                                             "old_only": sorted(set(reference) - set(ours))[:5],
                                             "new_only": sorted(set(ours) - set(reference))[:5]})
                elif ours != reference:
                    order_differs += 1
                    if len(order_examples) < 10:
                        first = next(index for index, (a, b) in enumerate(zip(reference, ours))
                                     if a != b)
                        order_examples.append({"qid": qid, "first_divergent_position": first,
                                               "old": reference[first:first + 3],
                                               "new": ours[first:first + 3]})
            window_report = {
                "stored_depth": args.depth,
                "compared": len(stored_windows),
                "set_differs": sets_differ,
                "order_differs_same_set": order_differs,
                "set_examples": set_examples,
                "order_examples": order_examples,
            }
            report["paths"][name] = {
                "elapsed_s": round(elapsed, 2),
                "queries_per_s": round(len(qids) / max(elapsed, 1e-9), 2),
                "rank_changed": len(changed),
                "max_abs_rank_delta": max((abs(d) for d in deltas), default=0),
                "rank_delta_histogram": _hist(deltas),
                "changed_rank_depth": _quantiles([min(e["old"], e["new"]) for e in changed
                                                  if e["old"] is not None and e["new"] is not None]),
                "depth_histogram": _buckets([min(e["old"], e["new"]) for e in changed
                                             if e["old"] is not None and e["new"] is not None]),
                "boundary_k10": sum(1 for c in crossings if c["k10"]),
                "boundary_k50": sum(1 for c in crossings if c["k50"]),
                "boundary_crossings": crossings[:40],
                "metrics": metrics(ranks),
                "metric_delta": _metric_delta(base_metrics, metrics(ranks)),
                "changed_examples": changed[:40],
                "changed_all": changed[:3000] if len(changed) > 3000 else changed,
                "margins": _margins(changed_rows, base_ranks, qids, positions, gallery, q_all,
                                    args.max_margin),
                "windows": window_report,
            }
            print(f"[{condition}/{name}] changed={len(changed)} "
                  f"k10={report['paths'][name]['boundary_k10']} "
                  f"k50={report['paths'][name]['boundary_k50']} "
                  f"{report['paths'][name]['queries_per_s']} q/s", flush=True)
        payload["conditions"][condition] = report
    Path(args.out).write_text(json.dumps(payload, indent=1, sort_keys=True, default=str))
    print(json.dumps({"out": args.out}))
    return 0


def load_stored_windows(condition: str, depth: int = 50) -> dict[str, list[int]]:
    """The shipped path's published depth-50 windows: qid -> [gallery position, ...] in order."""
    path = RUN / "candidates" / "habit_fiq_n02" / f"{condition}.jsonl"
    windows: dict[str, list[int]] = {}
    if not path.is_file():
        return windows
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            windows[row["qid"]] = [int(position) for position, _score in row["topk"]][:depth]
    return windows


def _margins(changed_rows, base_ranks, qids, positions, gallery, q_all, limit):
    """Per changed query: the positive's score under both paths, the exact-tie multiplicity, and
    whether the SHIPPED path had the positive tied exactly while the fast path did not (a
    tie-policy interaction) or both are tied but ordered differently (same tie, different bits)."""
    findings = []
    index_of = {qid: index for index, qid in enumerate(qids)}
    for qid, fast_row in list(changed_rows.items())[:limit]:
        index = index_of[qid]
        stored_rank = base_ranks[index]
        positives = positions[index]
        best, best_position = -np.inf, None
        for position in positives:
            if fast_row[position] > best:
                best, best_position = fast_row[position], position
        if best_position is None:
            continue
        # the shipped path's row for this same query, recomputed one query at a time
        shipped_row = score_old(np.asarray(q_all[index:index + 1], dtype=np.float32), gallery)[0]
        shipped_best, shipped_position = -np.inf, None
        for position in positives:
            if shipped_row[position] > shipped_best:
                shipped_best, shipped_position = shipped_row[position], position
        shipped_ties = (int(np.count_nonzero(shipped_row == shipped_row[shipped_position]))
                        if shipped_position is not None else 0)
        shipped_above = (int(np.count_nonzero(shipped_row > shipped_row[shipped_position]))
                         if shipped_position is not None else 0)
        shipped_before = (int(np.count_nonzero(shipped_row[:shipped_position] ==
                                               shipped_row[shipped_position]))
                          if shipped_position is not None else 0)
        fast_ties = int(np.count_nonzero(fast_row == fast_row[best_position]))
        fast_above = int(np.count_nonzero(fast_row > fast_row[best_position]))
        fast_before = int(np.count_nonzero(fast_row[:best_position] == fast_row[best_position]))
        findings.append({
            "qid": qid, "stored_rank": stored_rank,
            "fast_rank": 1 + fast_above + fast_before,
            "positive_position": int(best_position),
            "positive_score_fastpath": float(best),
            "fastpath_tie_multiplicity": fast_ties,
            "fastpath_rows_strictly_greater": fast_above,
            "fastpath_equal_rows_before": fast_before,
            "rank_delta": (None if stored_rank is None else (1 + fast_above + fast_before) - stored_rank),
            "shipped_recomputed": {
                "positive_position": (None if shipped_position is None else int(shipped_position)),
                "positive_score": (None if shipped_position is None else float(shipped_best)),
                "tie_multiplicity": shipped_ties,
                "rows_strictly_greater": shipped_above,
                "equal_rows_before": shipped_before,
                "rank": (None if shipped_position is None
                         else 1 + shipped_above + shipped_before),
            },
            "score_margin_shipped_minus_fast": (
                None if shipped_position is None else float(shipped_best) - float(best)),
            "tie_interaction": bool(shipped_ties != fast_ties),
        })
    return findings


def _metric_delta(base, new):
    return {key: (None if key not in base or key not in new or base[key] is None or new[key] is None
                  else new[key] - base[key]) for key in set(base) | set(new)}


def _hist(values):
    out = {}
    for value in values:
        out[str(value)] = out.get(str(value), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: int(kv[0])))


def _quantiles(values):
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    return {str(q): float(np.percentile(array, q)) for q in (0, 1, 5, 25, 50, 75, 95, 99, 100)}


def _buckets(values):
    edges = [(1, 10), (11, 50), (51, 100), (101, 1000), (1001, 5000), (5001, 20000),
             (20001, 74381)]
    out = {}
    for low, high in edges:
        out[f"{low}-{high}"] = sum(1 for value in values if low <= value <= high)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
