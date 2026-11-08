"""Audit harness: identical ranks/metrics/candidates under different SCORING paths.

Reads the cached HABIT features and the run's own stored ranks (never writes to them).
Compares, per condition, over ALL 6003 queries:

  old        : np.einsum("nd,gqd->ngq", q, t).max(-1)            <- shipped HABIT/intent/hint/tme
  opt        : same + optimize=True
  gemm512    : chunked BLAS gemm, max over the token axis
  gemm4096   : same with a 4096-row gallery chunk

For each path it derives ranks with the shipped rule (`1 + #greater + #earlier-equal`, min over
positives), the depth-50 window with the shipped `topk_from_scores`, K-hit bits, and the aggregate
metrics. A second pass recomputes the score row for every changed query and reports the positive's
score and its exact-tie multiplicity under each path: that distinguishes a tie-policy interaction
(exact equality in one path, 1-ULP difference in the other) from random instability.

Self-check: `old` ranks must reproduce the run's stored ranks exactly, otherwise the harness is
wrong and every other number here is meaningless.
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
CONDITIONS = ("multimodal", "text_only_black_image", "image_only_empty_text")


def ranks_from_scores(scores: np.ndarray, positions: list[tuple[int, ...]]) -> list[int | None]:
    """Verbatim copy of CompositionAdapter._ranks_from_scores (kept in sync by a test)."""
    ranks: list[int | None] = []
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


def score_old(q: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.einsum("nd,gqd->ngq", q, t).max(axis=-1)


def score_opt(q: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.einsum("nd,gqd->ngq", q, t, optimize=True).max(axis=-1)


def score_gemm_numbered(q: np.ndarray, t: np.ndarray, chunk: int) -> np.ndarray:
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


PATHS = {
    "old": score_old,
    "opt": score_opt,
    "gemm512": lambda q, t: score_gemm_numbered(q, t, 512),
    "gemm4096": lambda q, t: score_gemm_numbered(q, t, 4096),
}


def metrics(ranks: list[int | None]) -> dict:
    scored = [r for r in ranks if r is not None]
    n = len(scored)
    if not n:
        return {"scored": 0}
    return {
        "scored": n,
        "r10": 100.0 * sum(1 for r in scored if r <= 10) / n,
        "r50": 100.0 * sum(1 for r in scored if r <= 50) / n,
        "ndcg": float(np.mean([1.0 / np.log2(r + 1) for r in scored])),
    }


def margin_diagnosis(q_all, gallery, condition: str, changed: list[dict], positions: list,
                     qids: list[str], depth_paths: list[str]) -> list[dict]:
    """For each changed query: the positive's score and exact-tie multiplicity per path."""
    index_of = {qid: index for index, qid in enumerate(qids)}
    findings = []
    for entry in changed:
        index = index_of[entry["qid"]]
        query = np.asarray(q_all[index:index + 1], dtype=np.float32)
        positives = positions[index]
        row: dict = {"qid": entry["qid"], "old": entry["old"], "new": entry["new"]}
        for name in depth_paths:
            scores = PATHS[name](query, gallery)[0]
            best = None
            for positive in positives:
                score = float(scores[positive])
                if best is None or score > best[0]:
                    best = (score, positive)
            score, positive = best
            ties = int(np.count_nonzero(scores == np.float32(score)))
            above = int(np.count_nonzero(scores > np.float32(score)))
            before = int(np.count_nonzero(scores[:positive] == np.float32(score)))
            row[name] = {
                "positive_position": positive,
                "positive_score": score,
                "exact_tie_multiplicity": ties,
                "rows_strictly_greater": above,
                "equal_before_position": before,
                "derived_rank": 1 + above + before,
            }
        old, new = row.get("old", {}), row.get("new", {})
        row["margin_old_minus_new"] = (None if not old or not new
                                       else old["positive_score"] - new["positive_score"])
        row["tie_state"] = (f"old_ties={old.get('exact_tie_multiplicity')},"
                            f"new_ties={new.get('exact_tie_multiplicity')}")
        findings.append(row)
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--paths", default=",".join(PATHS))
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--depth", type=int, default=50)
    parser.add_argument("--out", default=str(OUT / "scoring_paths.json"))
    parser.add_argument("--label", default="")
    parser.add_argument("--no-margins", action="store_true")
    parser.add_argument("--anchor", default="old", choices=["old", "stored"],
                        help="old: recompute the shipped path; stored: compare against the run's "
                             "own ranks artifact (the shipped path's output, already on disk). "
                             "'stored' skips the 5.6 q/s full-gallery einsum.")
    args = parser.parse_args()

    manifest = json.loads((OUT / "query_manifest.json").read_text())
    stored = json.loads((RUN / "ranks/habit_fiq_n02.json").read_text())
    # The feature rows follow the release query order the runner encoded; the stored rank JSON is
    # key-sorted, so it is a lookup by qid, never a positional comparison.
    qids = list(manifest["query_order"])
    if set(qids) != set(stored["ranks"]["multimodal"]):
        raise SystemExit("manifest query ids != stored rank ids; mapping is wrong")
    positions = [tuple(manifest["positive_positions"][qid]) for qid in qids]
    if args.limit:
        qids, positions = qids[:args.limit], positions[:args.limit]
    names = [name for name in args.paths.split(",") if name]

    gallery = np.load(FEATURES / "gallery_global.npy", mmap_mode="r")
    result: dict = {
        "label": args.label, "run": RUN.name, "query_count": len(qids),
        "batch": args.batch, "depth": args.depth, "numpy": np.__version__,
        "threads_env": {k: os.environ.get(k) for k in
                        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "conditions": {},
    }
    selfcheck: dict = {}
    for condition in args.conditions.split(","):
        features = np.load(FEATURES / f"query_{condition}_global.npy", mmap_mode="r")
        q_all = np.asarray(features[:len(qids)], dtype=np.float32)
        ranks: dict[str, list[int | None]] = {name: [] for name in names}
        windows: dict[str, list] = {name: [] for name in names}
        stats: dict[str, dict] = {}
        elapsed: dict[str, float] = {}
        for start in range(0, len(qids), args.batch):
            batch_q = q_all[start:start + args.batch]
            batch_pos = positions[start:start + args.batch]
            local = {}
            for name in names:
                tick = time.perf_counter()
                local[name] = PATHS[name](batch_q, gallery)
                elapsed[name] = elapsed.get(name, 0.0) + (time.perf_counter() - tick)
            base = np.asarray(local.get("old", local[names[0]]), dtype=np.float32)
            for name in names:
                scores = local[name]
                ranks[name].extend(ranks_from_scores(scores, batch_pos))
                windows[name].extend(topk_from_scores(row, args.depth) for row in scores)
                if name == "old":
                    continue
                stat = stats.setdefault(name, {"maxabs": 0.0, "maxrel": 0.0, "elements": 0,
                                               "values_differing": 0, "score_rows_differing": 0})
                other = np.asarray(scores, dtype=np.float32)
                delta = np.abs(other.astype(np.float64) - base.astype(np.float64))
                stat["maxabs"] = max(stat["maxabs"], float(delta.max()) if delta.size else 0.0)
                denom = np.maximum(np.abs(base.astype(np.float64)), 1e-12)
                stat["maxrel"] = max(stat["maxrel"], float((delta / denom).max()) if delta.size else 0.0)
                differing = other != base
                stat["values_differing"] += int(np.count_nonzero(differing))
                stat["score_rows_differing"] += int(np.count_nonzero(differing.any(axis=1)))
                stat["elements"] += int(delta.size)

        per_path: dict[str, dict] = {}
        stored_condition = stored["ranks"][condition]
        for name in names:
            if name == "old":
                continue
            changed = [{"qid": qids[i], "old": a, "new": b,
                        "delta": (None if a is None or b is None else b - a)}
                       for i, (a, b) in enumerate(zip(ranks["old"], ranks[name])) if a != b]
            crossed = []
            for entry in changed:
                a, b = entry["old"], entry["new"]
                if a is None or b is None:
                    continue
                c10 = (a <= 10) != (b <= 10)
                c50 = (a <= 50) != (b <= 50)
                if c10 or c50:
                    crossed.append({**entry, "crossed_k10": c10, "crossed_k50": c50})
            sets_differ = order_differs = 0
            examples = []
            for i, (wa, wb) in enumerate(zip(windows["old"], windows[name])):
                sa = [pos for pos, _ in wa]
                sb = [pos for pos, _ in wb]
                if set(sa) != set(sb):
                    sets_differ += 1
                    if len(examples) < 8:
                        examples.append({"qid": qids[i], "kind": "set",
                                         "old_only": sorted(set(sa) - set(sb))[:3],
                                         "new_only": sorted(set(sb) - set(sa))[:3]})
                elif sa != sb:
                    order_differs += 1
                    if len(examples) < 8:
                        examples.append({"qid": qids[i], "kind": "order"})
            deltas = [e["delta"] for e in changed if e["delta"] is not None]
            entry = {
                "score_delta": stats.get(name, {}),
                "rank_changed": len(changed),
                "changed_within_top50_rank": sum(1 for e in changed
                                                 if (e["old"] or 10**9) <= 50 or (e["new"] or 10**9) <= 50),
                "max_abs_rank_delta": max((abs(d) for d in deltas), default=0),
                "rank_delta_histogram": _hist(deltas),
                "changed_rank_quantiles": _quantiles([min(e["old"], e["new"]) for e in changed
                                                      if e["old"] is not None and e["new"] is not None]),
                "boundary": {"k10_crossings": sum(1 for c in crossed if c["crossed_k10"]),
                             "k50_crossings": sum(1 for c in crossed if c["crossed_k50"]),
                             "total_queries_crossing": len(crossed)},
                "boundary_crossings": crossed[:50],
                "topk_set_differs": sets_differ,
                "topk_order_differs_same_set": order_differs,
                "topk_examples": examples,
                "metrics": metrics(ranks[name]),
                "elapsed_s": round(elapsed.get(name, 0.0), 2),
                "queries_per_s": round(len(qids) / max(elapsed.get(name, 1e-9), 1e-9), 3),
                "changed_examples": changed[:40],
            }
            if not args.no_margins and changed:
                entry["margins"] = margin_diagnosis(q_all, gallery, condition, changed[:60],
                                                    positions, qids, ["old", name])
            per_path[name] = entry
        per_path["old"] = {"metrics": metrics(ranks["old"]),
                           "elapsed_s": round(elapsed.get("old", 0.0), 2),
                           "queries_per_s": round(len(qids) / max(elapsed.get("old", 1e-9), 1e-9), 3)}
        result["conditions"][condition] = per_path
        # shipped-path reproduction of the run's own artifact
        mismatches = [{"qid": q, "recomputed": r, "stored": stored_condition[q]}
                      for q, r in zip(qids, ranks["old"]) if stored_condition.get(q) != r]
        selfcheck[condition] = {"stored_vs_recomputed_mismatches": len(mismatches),
                                "examples": mismatches[:10]}
        print(f"[{condition}] old={elapsed.get('old', 0):.1f}s "
              f"selfcheck_mismatches={len(mismatches)}", flush=True)
    result["selfcheck"] = selfcheck
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=True, default=str))
    print(json.dumps({"out": args.out, "selfcheck": selfcheck}, indent=1))
    return 0


def _hist(values: list[int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[str(value)] = out.get(str(value), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: int(kv[0])))


def _quantiles(values: list[int]) -> dict[str, float]:
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    return {str(q): float(np.percentile(array, q)) for q in (0, 1, 5, 25, 50, 75, 95, 99, 100)}


if __name__ == "__main__":
    raise SystemExit(main())
