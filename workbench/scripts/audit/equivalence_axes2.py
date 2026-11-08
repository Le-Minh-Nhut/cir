"""G4 batching axes, cheap and explicit about what is sampled.

The shipped einsum runs at 5.6 q/s, so a full 6003-query sweep per axis is ~18 min per
configuration and the axes would cost hours. Each axis therefore uses a DECLARED sample, and the
sample size is in every result. Batch invariance is a property of how rows are grouped, not of which
rows they are, so a slice answers the question while the sample size stays visible.

Axes:
  A. batch 1 vs 32 vs 7 vs 256   (shipped path, slice; fast path, FULL 6003)
  B. incomplete final batch      (queries not divisible by the batch)
  C. repeat determinism          (identical configuration twice)
  D. chunk boundaries            (gemm path: 512 / 1024 / 4096 gallery chunks, full set)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/home/heheboiz/data/cir")
sys.path.insert(0, str(REPO))
from workbench.scripts.audit.equivalence_fastpath import (  # noqa: E402
    score_gemm, score_old, ranks_from_scores,
)

RUN = REPO / "workbench/artifacts/composition/runs/497c25c57fbeba885ecfb9d1"
FEATURES = RUN / "work/habit/features"
OUT = REPO / "workbench/docs/reviews/20261108T153045-audit-equivalence"
CONDITIONS = ("multimodal", "text_only_black_image", "image_only_empty_text")


def rank_all(q_all, gallery, positions, batch, path):
    ranks = []
    for start in range(0, len(q_all), batch):
        chunk = q_all[start:start + batch]
        if not len(chunk):
            continue
        ranks.extend(ranks_from_scores(path(chunk, gallery), positions[start:start + batch]))
    return ranks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shipped-slice", type=int, default=96)
    parser.add_argument("--out", default=str(OUT / "axes.json"))
    args = parser.parse_args()

    manifest = json.loads((OUT / "query_manifest.json").read_text())
    qids = list(manifest["query_order"])
    positions = [tuple(manifest["positive_positions"][qid]) for qid in qids]
    gallery = np.load(FEATURES / "gallery_global.npy", mmap_mode="r")
    payload = {"declared_samples": {"shipped_path_queries": args.shipped_slice,
                                    "fast_path_queries": len(qids),
                                    "total_queries": len(qids),
                                    "total_mod_32": len(qids) % 32},
               "conditions": {}}
    for condition in CONDITIONS:
        features = np.load(FEATURES / f"query_{condition}_global.npy", mmap_mode="r")
        q_all = np.asarray(features, dtype=np.float32)
        entry: dict = {}
        # ---- A/B/C on the SHIPPED path over a declared slice ----
        head = args.shipped_slice
        qs, ps = q_all[:head], positions[:head]
        tick = time.perf_counter()
        ref = rank_all(qs, gallery, ps, 32, score_old)
        ref_again = rank_all(qs, gallery, ps, 32, score_old)
        entry["reference_seconds"] = round(time.perf_counter() - tick, 2)
        entry["C_repeat_determinism"] = {
            "queries": head,
            "rank_mismatches": sum(1 for a, b in zip(ref, ref_again) if a != b)}
        for batch in (1, 7, 256):
            other = rank_all(qs, gallery, ps, batch, score_old)
            entry[f"A_batch{batch}_vs_batch32"] = {
                "queries": head,
                "rank_mismatches": sum(1 for a, b in zip(ref, other) if a != b)}
        # ---- B: incomplete final batch, explicitly -------------------------------------------------
        # 32 rows: one FULL batch. 32 + 12 rows: same first batch plus an incomplete tail. If the tail
        # grouping changed anything, the first 32 ranks would differ between the two configurations.
        for count, batch in ((32, 32), (44, 32), (44, 11)):
            part = rank_all(q_all[:count], gallery, positions[:count], batch, score_old)
            entry.setdefault("B_batch_grouping", {})[f"{count}rows_batch{batch}"] = {
                "queries": count, "mod_batch": count % batch,
                "first32_equal_to_reference": sum(
                    1 for a, b in zip(part[:min(32, len(part))],
                                      rank_all(q_all[:min(32, count)], gallery,
                                               positions[:min(32, count)], 32, score_old))
                    if a != b) == 0}
        # ---- A/D on the FAST path over the FULL query set ----
        tick = time.perf_counter()
        fast32 = rank_all(q_all, gallery, positions, 32, lambda q, t: score_gemm(q, t, 1024))
        entry["fastpath_full"] = {"queries": len(qids), "seconds": round(time.perf_counter() - tick, 2)}
        for batch in (1, 256):
            other = rank_all(q_all, gallery, positions, batch, lambda q, t: score_gemm(q, t, 1024))
            entry["fastpath_full"][f"batch{batch}_vs_32_mismatches"] = sum(
                1 for a, b in zip(fast32, other) if a != b)
        chunks = {}
        for chunk in (512, 4096):
            chunks[chunk] = rank_all(q_all, gallery, positions, 32,
                                     lambda q, t, c=chunk: score_gemm(q, t, c))
        entry["D_chunk_size"] = {
            "512_vs_1024_mismatches": sum(1 for a, b in zip(chunks[512], fast32) if a != b),
            "4096_vs_1024_mismatches": sum(1 for a, b in zip(chunks[4096], fast32) if a != b),
        }
        payload["conditions"][condition] = entry
        print(f"[{condition}] {json.dumps(entry)}", flush=True)
    Path(args.out).write_text(json.dumps(payload, indent=1, sort_keys=True, default=str))
    print(json.dumps({"out": args.out}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
