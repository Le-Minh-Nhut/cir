"""Bounded real scan: does BLAS thread count move LIMN's AGGREGATE metrics, not just ranks?

Full 6003x3 needs ~61 min of query encode (measured 4.93 q/s marginal), over budget. This runs a
600-query real subsample per condition through the REAL LIMN adapter: encode once, then score the
same vectors at BLAS=1 and BLAS=10 and compare ranks, boundary crossings, and the production
metrics. Same code path, same checkpoints, real gallery features -- just fewer queries.
"""
import sys, json, time
sys.path.insert(0, "/home/heheboiz/data/cir")
from pathlib import Path
import numpy as np

from workbench.backend.operator_config import resolve_config
from workbench.composition.data import load_release, assign_garment_categories
from workbench.composition.runner import (image_resolver_factory, _positive_positions,
                                          _rank_groups_concurrently, _blas_thread_limit)
from workbench.composition.adapters import adapter_class
from workbench.composition.metrics import compute_condition_metrics

SAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 600
CONDS = ("multimodal", "text_only_black_image", "image_only_empty_text")
PRESERVED = Path("/home/heheboiz/data/cir/workbench/artifacts/composition/preserved_features/limn")

cfg = resolve_config()
data = assign_garment_categories(
    load_release(Path("/home/heheboiz/data/cir/workbench/artifacts/circus_release")),
    Path(cfg.FASHIONIQ_ROOT))
resolver = image_resolver_factory(Path(cfg.FASHIONIQ_ROOT), "fashioniq_standard")
Cls = adapter_class("limn")
adapter = Cls(cfg, "base_iter0_all_categories")
adapter.work_root = Path("/tmp/limn_boundary_work")
adapter.work_root.mkdir(parents=True, exist_ok=True)

galleries = {c: np.load(PRESERVED / f"gallery_{c}.npy", mmap_mode="r")
             for c in ("dress", "shirt", "toptee")}
positions = _positive_positions(data)

selected = [q for q in data.queries if q.category][:SAMPLE]
paths = [resolver(q.reference_id, data.candidate_names) for q in selected]
keep = [i for i, p in enumerate(paths) if p is not None]
selected = [selected[i] for i in keep]
paths = [paths[i] for i in keep]
print(f"sample={len(selected)} queries, real LIMN adapter, real gallery features", flush=True)

report = {}
scores_by_cond = {}
for cond in CONDS:
    print(f"--- encode {cond} ...", flush=True)
    t = time.perf_counter()
    vectors = adapter.encode_queries([str(p) for p in paths],
                                     [q.query_txt for q in selected],
                                     [q.category for q in selected])
    print(f"    encoded in {time.perf_counter()-t:.1f}s", flush=True)
    scores_by_cond[cond] = (np.asarray(vectors, dtype=np.float32),
                            [q.category for q in selected],
                            [q.qid for q in selected])

print("scoring at both BLAS settings ...", flush=True)
for cond in CONDS:
    vectors, cats, qids = scores_by_cond[cond]
    pos = [[p for p in positions.get(q, ()) if p < 74381] for q in qids]
    per_blas = {}
    for blas in (1, 10):
        previous = _blas_thread_limit(blas)
        try:
            ranks = adapter.rank(vectors, galleries, pos, cats)
        finally:
            _blas_thread_limit(previous)
        per_blas[blas] = ranks
    r1, r10 = per_blas[1], per_blas[10]
    changed = [(q, a, b) for q, a, b in zip(qids, r1, r10) if a != b]
    deltas = [abs(a - b) for _, a, b in changed if a is not None and b is not None]
    m1 = compute_condition_metrics(cond, dict(zip(qids, r1)))
    m10 = compute_condition_metrics(cond, dict(zip(qids, r10)))
    row = {
        "queries": len(qids),
        "changed": len(changed), "max_drank": max(deltas) if deltas else 0,
        "cross_k10": sum(1 for _, a, b in changed if a and b and (a <= 10) != (b <= 10)),
        "cross_k50": sum(1 for _, a, b in changed if a and b and (a <= 50) != (b <= 50)),
        "cross_k100": sum(1 for _, a, b in changed if a and b and (a <= 100) != (b <= 100)),
        "r10_blas1": m1.recall_at_10, "r10_blas10": m10.recall_at_10,
        "r50_blas1": m1.recall_at_50, "r50_blas10": m10.recall_at_50,
        "ndcg_blas1": m1.ndcg_full, "ndcg_blas10": m10.ndcg_full,
        "ndcg_delta": abs((m1.ndcg_full or 0) - (m10.ndcg_full or 0)),
        "r10_delta": abs((m1.recall_at_10 or 0) - (m10.recall_at_10 or 0)),
        "r50_delta": abs((m1.recall_at_50 or 0) - (m10.recall_at_50 or 0)),
        "examples": changed[:8],
    }
    report[cond] = row
    print(json.dumps({cond: row}, indent=2), flush=True)

print("\n=== LIMN AGGREGATE SENSITIVITY (real data, "
      f"{len(selected)} queries/condition) ===")
print(f"{'condition':<26}{'R@10 d':>10}{'R@50 d':>10}{'dNDCG':>11}"
      f"{'nchg':>6}{'maxdR':>7}{'xK10':>6}{'xK50':>6}{'xK100':>7}")
any_move = False
for cond in CONDS:
    r = report[cond]
    print(f"{cond:<26}{r['r10_delta']:>10.4f}{r['r50_delta']:>10.4f}"
          f"{r['ndcg_delta']:>11.2e}{r['changed']:>6}{r['max_drank']:>7}"
          f"{r['cross_k10']:>6}{r['cross_k50']:>6}{r['cross_k100']:>7}")
    if r["r10_delta"] > 0 or r["r50_delta"] > 0 or r["ndcg_delta"] > 1e-6:
        any_move = True

verdict = ("AGGREGATES MOVE - BLAS policy changes published metrics"
           if any_move else
           "AGGREGATES STABLE on this sample - changes confined to deep ranks")
print(f"\nVERDICT: {verdict}")
Path("/tmp/limn_boundary_result.json").write_text(json.dumps(report, indent=2) + "\n")
adapter.close()
