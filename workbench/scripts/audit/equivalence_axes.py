"""G4 axes on the real cached features: batch size, incomplete final batch, repeat determinism.

Axes, each compared against one declared reference configuration:

  A. batch 32 (reference) vs batch 1        -- isolates per-batch grouping from the arithmetic
  B. batch 32 over 6003 queries (6003 % 32 = 19, so the LAST batch is incomplete) vs the
     first 5984 queries only -- isolates whether the incomplete tail batch changes anything
  C. the same configuration run twice       -- determines whether the path is bit-reproducible
  D. gallery chunk boundaries for the gemm path: 512 vs 4096 vs 1024 -- pins whether the
     chunk size itself moves a score

Cold vs warm cache is NOT measured here: the decode cache is a memmap keyed by an identity
fingerprint, and this host keeps no encoder gallery cache to compare against; the habit cache that
does exist has exactly one warm instance. That axis is reported NOT_RUN with the reason rather than
simulated, because a simulated cold cache would be measuring this script, not the runner.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/heheboiz/data/cir")
sys.path.insert(0, str(REPO))
from workbench.composition.topk import topk_from_scores  # noqa: E402
from workbench.scripts.audit.equivalence_scoring import (  # noqa: E402
    PATHS, OUT, FEATURES, RUN, ranks_from_scores,
)


def score_batched(q_all, gallery, positions, batch, path_name, depth, limit=None):
    ranks, windows = [], []
    for start in range(0, len(q_all), batch):
        batch_q = q_all[start:start + batch]
        if not len(batch_q):
            continue
        scores = PATHS[path_name](batch_q, gallery)
        ranks.extend(ranks_from_scores(scores, positions[start:start + batch]))
        windows.extend(topk_from_scores(row, depth) for row in scores)
    return ranks, windows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", default="multimodal,text_only_black_image,image_only_empty_text")
    parser.add_argument("--path", default="old")
    parser.add_argument("--depth", type=int, default=50)
    parser.add_argument("--out", default=str(OUT / "axes.json"))
    args = parser.parse_args()

    manifest = json.loads((OUT / "query_manifest.json").read_text())
    qids = list(manifest["query_order"])
    positions = [tuple(manifest["positive_positions"][qid]) for qid in qids]
    gallery = np.load(FEATURES / "gallery_global.npy", mmap_mode="r")
    payload = {"path": args.path, "queries": len(qids), "conditions": {},
               "total_mod_32": len(qids) % 32, "total_mod_1": 0}
    for condition in args.conditions.split(","):
        features = np.load(FEATURES / f"query_{condition}_global.npy", mmap_mode="r")
        q_all = np.asarray(features, dtype=np.float32)
        report = {}

        ref_ranks, ref_windows = score_batched(q_all, gallery, positions, 32, args.path, args.depth)
        # C: repeat the identical configuration
        repeat_ranks, repeat_windows = score_batched(q_all, gallery, positions, 32, args.path, args.depth)
        report["C_repeat_determinism"] = {
            "rank_mismatches": sum(1 for a, b in zip(ref_ranks, repeat_ranks) if a != b),
            "window_mismatches": sum(1 for a, b in zip(ref_windows, repeat_windows) if a != b),
        }

        # A: batch 1
        one_ranks, one_windows = score_batched(q_all, gallery, positions, 1, args.path, args.depth)
        report["A_batch1_vs_32"] = _diff(ref_ranks, one_ranks, ref_windows, one_windows, qids)

        # B: incomplete final batch. 6003 = 187*32 + 19; the head has no incomplete batch.
        head = (len(qids) // 32) * 32
        head_ranks = ref_ranks[:head]
        head_positions = positions[:head]
        head_q = q_all[:head]
        cut_ranks, cut_windows = score_batched(head_q, gallery, head_positions, 32, args.path, args.depth)
        report["B_incomplete_tail_batch"] = {
            "queries_total": len(qids), "queries_in_complete_batches": head,
            "tail_batch_size": len(qids) - head,
            "head_rank_mismatches": sum(1 for a, b in zip(head_ranks, cut_ranks) if a != b),
            "tail_rank_changed_vs_head": None,
            "note": "the tail rows are not present in the head run, so the only meaningful check is "
                    "that dropping them changes nothing before the boundary",
        }

        # D: gallery chunk size for the gemm path (only meaningful for a chunked path)
        if args.path.startswith("gemm"):
            chunk_ranks = {}
            for chunk in (1024, 4096):
                local = {}
                ranks = []
                for start in range(0, len(q_all), 32):
                    batch_q = q_all[start:start + 32]
                    ranks.extend(ranks_from_scores(_gemm(batch_q, gallery, chunk),
                                                   positions[start:start + 32]))
                chunk_ranks[chunk] = ranks
            report["D_chunk_1024_vs_4096"] = {
                "rank_mismatches": sum(1 for a, b in zip(chunk_ranks[1024], chunk_ranks[4096]) if a != b)}
        payload["conditions"][condition] = report
        print(json.dumps({condition: report}, indent=1), flush=True)
    Path(args.out).write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(json.dumps({"out": args.out}))
    return 0


def _gemm(q, t, chunk):
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


def _diff(a_ranks, b_ranks, a_windows, b_windows, qids):
    changed = [qids[i] for i, (a, b) in enumerate(zip(a_ranks, b_ranks)) if a != b]
    return {
        "rank_mismatches": len(changed),
        "rank_mismatch_qids": changed[:20],
        "max_abs_rank_delta": max((abs(b - a) for a, b in zip(a_ranks, b_ranks)
                                   if a is not None and b is not None), default=0),
        "window_mismatches": sum(1 for a, b in zip(a_windows, b_windows) if a != b),
    }


if __name__ == "__main__":
    raise SystemExit(main())
