"""G3b tail + G4 axes: composition gap under both scoring paths, and batching invariance.

Part 1 -- composition gap. The spec computes
    composition_gap_ndcg = 1 - max(I, T) / MM
AFTER averaging over a common cohort, and emits an explicit unavailable value rather than 0/inf/NaN
when MM is zero or a condition is missing. Both scoring paths are evaluated on the same query set.

Part 2 -- G4 batching axes on the SHIPPED path:
  * batch 1 vs batch 32 over a real query slice (the shipped einsum is 5.6 q/s, so the slice is
    stated rather than implying the full 6003)
  * an incomplete final batch, made explicit by slicing a count that is not a multiple of 32
  * a repeated identical run (determinism)
  * the same batch axis on the fast path over ALL 6003 queries, where it is affordable
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/heheboiz/data/cir")
sys.path.insert(0, str(REPO))
from workbench.composition.metrics import compose  # noqa: E402
from workbench.scripts.audit.equivalence_fastpath import (  # noqa: E402
    score_gemm, score_old, ranks_from_scores,
)

RUN = REPO / "workbench/artifacts/composition/runs/497c25c57fbeba885ecfb9d1"
FEATURES = RUN / "work/habit/features"
OUT = REPO / "workbench/docs/reviews/20261108T153045-audit-equivalence"
CONDITIONS = ("multimodal", "text_only_black_image", "image_only_empty_text")


def run_path(q_all, gallery, positions, batch, path):
    ranks = []
    for start in range(0, len(q_all), batch):
        scores = path(q_all[start:start + batch], gallery)
        ranks.extend(ranks_from_scores(scores, positions[start:start + batch]))
    return ranks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice", type=int, default=320)
    parser.add_argument("--skip-gap", action="store_true",
                        help="reuse the saved gap table and only run the batching axes")
    parser.add_argument("--out", default=str(OUT / "gap_and_axes.json"))
    args = parser.parse_args()

    manifest = json.loads((OUT / "query_manifest.json").read_text())
    stored = json.loads((RUN / "ranks/habit_fiq_n02.json").read_text())
    qids = list(manifest["query_order"])
    positions = [tuple(manifest["positive_positions"][qid]) for qid in qids]
    gallery = np.load(FEATURES / "gallery_global.npy", mmap_mode="r")
    payload: dict = {"gap": {}, "axes": {}}

    # ---------- Part 1: composition gap, stored vs fast path ----------
    gap_tables = {}
    if args.skip_gap and Path(args.out).is_file():
        payload = json.loads(Path(args.out).read_text())
        print("[gap] reused from previous run", flush=True)
        _axes(payload, manifest, qids, positions, gallery, args)
        Path(args.out).write_text(json.dumps(payload, indent=1, sort_keys=True, default=str))
        print(json.dumps({"out": args.out, "axes_only": True}))
        return 0
    for label, source in (("stored", None), ("gemm1024", "gemm"), ("opt", "opt")):
        table = {}
        for condition in CONDITIONS:
            if source is None:
                ranks = [stored["ranks"][condition][qid] for qid in qids]
            else:
                features = np.load(FEATURES / f"query_{condition}_global.npy", mmap_mode="r")
                q_all = np.asarray(features, dtype=np.float32)
                path = (lambda q, t: score_gemm(q, t, 1024)) if source == "gemm" else \
                    (lambda q, t: score_opt_local(q, t))
                ranks = run_path(q_all, gallery, positions, 32, path)
            table[condition] = {qid: rank for qid, rank in zip(qids, ranks)}
        gap_tables[label] = table
        result = compose(table, len(qids))
        payload["gap"][label] = {
            "mm_ndcg": result.multimodal.ndcg_full,
            "text_ndcg": result.text_only.ndcg_full,
            "image_ndcg": result.image_only.ndcg_full,
            "composition_gap_ndcg": result.composition_gap_ndcg,
            "gap_unavailable_reason": result.gap_unavailable_reason,
            "gap_cohort_count": result.gap_cohort_count,
            "gap_mm_ndcg": result.gap_multimodal_ndcg,
            "gap_text_ndcg": result.gap_text_only_ndcg,
            "gap_image_ndcg": result.gap_image_only_ndcg,
            "mm_recall_at_10": result.multimodal.recall_at_10,
            "mm_recall_at_50": result.multimodal.recall_at_50,
            "mm_mrr": result.multimodal.mrr,
            "mrr_gap": result.mrr_gap,
            "scored_counts": {"mm": result.multimodal.scored_count,
                              "text": result.text_only.scored_count,
                              "image": result.image_only.scored_count},
        }
        print(f"[gap/{label}] {json.dumps(payload['gap'][label], default=str)}", flush=True)
    base = payload["gap"]["stored"]
    def delta(a, b):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return a - b
        if isinstance(a, dict) and isinstance(b, dict):
            return {key: delta(a.get(key), b.get(key)) for key in set(a) | set(b)}
        return None

    payload["gap_delta"] = {label: {key: delta(payload["gap"][label].get(key), base.get(key))
                                   for key in base}
                            for label in ("gemm1024", "opt")}

    _axes(payload, manifest, qids, positions, gallery, args)

    Path(args.out).write_text(json.dumps(payload, indent=1, sort_keys=True, default=str))
    print(json.dumps({"out": args.out}))
    return 0



def _axes(payload, manifest, qids, positions, gallery, args):
    from workbench.scripts.audit.equivalence_fastpath import score_old as _old
    # ---------- Part 2: batching axes ----------
    for condition in CONDITIONS:
        features = np.load(FEATURES / f"query_{condition}_global.npy", mmap_mode="r")
        head = args.slice
        q_slice = np.asarray(features[:head], dtype=np.float32)
        pos_slice = positions[:head]
        entry: dict = {"slice": head}
        ref = run_path(q_slice, gallery, pos_slice, 32, _old)
        entry["repeat_32"] = {"rank_mismatches": sum(
            1 for a, b in zip(ref, run_path(q_slice, gallery, pos_slice, 32, _old)) if a != b)}
        entry["batch1_vs_32"] = {"rank_mismatches": sum(
            1 for a, b in zip(ref, run_path(q_slice, gallery, pos_slice, 1, _old)) if a != b)}
        entry["batch7_vs_32"] = {"rank_mismatches": sum(
            1 for a, b in zip(ref, run_path(q_slice, gallery, pos_slice, 7, _old)) if a != b)}
        # incomplete final batch: 300 rows = 9*32 + 12, versus the same rows batched 10*30
        tail_count = 300
        tail_q, tail_pos = q_slice[:tail_count], pos_slice[:tail_count]
        a = run_path(tail_q, gallery, tail_pos, 32, _old)
        b = run_path(tail_q, gallery, tail_pos, 30, _old)
        entry["incomplete_final_batch"] = {
            "queries": tail_count, "mod_32": tail_count % 32,
            "batch32_vs_batch30_mismatches": sum(1 for x, y in zip(a, b) if x != y)}
        # full-set batch axis on the fast path, where the cost is affordable
        q_all = np.asarray(features, dtype=np.float32)
        fast32 = run_path(q_all, gallery, positions, 32, lambda q, t: score_gemm(q, t, 1024))
        fast1 = run_path(q_all, gallery, positions, 1, lambda q, t: score_gemm(q, t, 1024))
        entry["fastpath_full_batch1_vs_32"] = {
            "queries": len(qids),
            "rank_mismatches": sum(1 for x, y in zip(fast32, fast1) if x != y)}
        payload["axes"][condition] = entry
        print(f"[axes/{condition}] {json.dumps(entry)}", flush=True)



def score_opt_local(q, t):
    return np.einsum("nd,gqd->ngq", q, t, optimize=True).max(axis=-1)


if __name__ == "__main__":
    raise SystemExit(main())
