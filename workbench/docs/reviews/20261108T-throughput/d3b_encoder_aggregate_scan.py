"""FULL-SCALE: does BLAS thread count move ENCODER's aggregate metrics?

Encoder is the only @-based adapter whose score() uses a BATCH matmul (`queries @ gallery.T` over
8192-row chunks) -- the shape whose kernel depends on the matrix M dimension. LIMN scores one row
at a time and measured zero rank changes; PTHA/CSMCIR use einsum and never reach BLAS.

Encoder has no cached features, so real query vectors are encoded here (measured 13.84 q/s ->
21.7 min for 6003 x 3), then scored against the REAL published gallery geometry at BLAS=1 and
BLAS=10, and compared rank-by-rank including boundary crossings and the production metrics.

The published rank artifact is imported ONLY for its boundary distribution and as the independent
column; the vectors here are freshly encoded by the real adapter.
"""
import sys, json, time
sys.path.insert(0, "/home/heheboiz/data/cir")
from pathlib import Path
import numpy as np

from workbench.backend.operator_config import resolve_config
from workbench.composition.data import load_release, assign_garment_categories
from workbench.composition.runner import (image_resolver_factory, _positive_positions,
                                          _blas_thread_limit)
from workbench.composition.adapters import adapter_class
from workbench.composition.metrics import compute_condition_metrics

CONDS = ("multimodal", "text_only_black_image", "image_only_empty_text")
ART = "workbench/artifacts/composition/runs/6a528e877c328772e2c7bb28/ranks/encoder_fashioniq.json"

cfg = resolve_config()
data = assign_garment_categories(
    load_release(Path("/home/heheboiz/data/cir/workbench/artifacts/circus_release")),
    Path(cfg.FASHIONIQ_ROOT))
resolver = image_resolver_factory(Path(cfg.FASHIONIQ_ROOT), "fashioniq_standard")
Cls = adapter_class("encoder")
adapter = Cls(cfg, "fashioniq")
adapter.work_root = Path("/tmp/enc_full_work")
adapter.work_root.mkdir(parents=True, exist_ok=True)

qids = [q.qid for q in data.queries]
selected = [q for q in data.queries]
paths = [resolver(q.reference_id, data.candidate_names) for q in selected]
usable = [i for i, p in enumerate(paths) if p is not None]
print(f"queries={len(selected)} with reference image={len(usable)}", flush=True)

published = json.load(open(ART))["ranks"]
positions = _positive_positions(data)

# --- Encode the gallery ONCE (real encoder forward), reusing the adapter's own batch path.
print("encoding gallery (74381 images) ...", flush=True)
t0 = time.perf_counter()
gallery_paths = [resolver(identity, data.candidate_names) for identity in data.gallery.image_ids]
gallery_paths = [p for p in gallery_paths if p is not None]
gallery = adapter.encode_targets([str(p) for p in gallery_paths], [None] * len(gallery_paths))
gallery = np.asarray(gallery, dtype=np.float32)
print(f"  gallery done in {(time.perf_counter()-t0)/60:.1f} min shape={gallery.shape}", flush=True)

report = {}
for cond in CONDS:
    t = time.perf_counter()
    vectors = adapter.encode_queries([str(paths[i]) for i in usable],
                                     [selected[i].query_txt for i in usable],
                                     [None] * len(usable))
    vectors = np.asarray(vectors, dtype=np.float32)
    print(f"--- {cond}: encoded {len(vectors)} in {(time.perf_counter()-t)/60:.1f} min", flush=True)

    cond_qids = [qids[i] for i in usable]
    pos = [[p for p in positions.get(q, ()) if p < gallery.shape[0]] for q in cond_qids]

    per_blas = {}
    for blas in (1, 10):
        previous = _blas_thread_limit(blas)
        try:
            per_blas[blas] = adapter.rank(vectors, gallery, pos, [None] * len(vectors))
        finally:
            _blas_thread_limit(previous)

    r1, r10 = per_blas[1], per_blas[10]
    changed = [(q, a, b) for q, a, b in zip(cond_qids, r1, r10) if a != b]
    deltas = [abs(a - b) for _, a, b in changed if a is not None and b is not None]
    m1 = compute_condition_metrics(cond, dict(zip(cond_qids, r1)))
    m10 = compute_condition_metrics(cond, dict(zip(cond_qids, r10)))
    pub = {q: published[cond].get(q) for q in cond_qids if published[cond].get(q) is not None}
    mpub = compute_condition_metrics(cond, pub) if pub else None

    report[cond] = {
        "queries": len(cond_qids),
        "changed": len(changed), "max_drank": max(deltas) if deltas else 0,
        "cross_k10": sum(1 for _, a, b in changed if a and b and (a <= 10) != (b <= 10)),
        "cross_k50": sum(1 for _, a, b in changed if a and b and (a <= 50) != (b <= 50)),
        "cross_k100": sum(1 for _, a, b in changed if a and b and (a <= 100) != (b <= 100)),
        "r10_blas1": m1.recall_at_10, "r10_blas10": m10.recall_at_10,
        "r50_blas1": m1.recall_at_50, "r50_blas10": m10.recall_at_50,
        "ndcg_blas1": m1.ndcg_full, "ndcg_blas10": m10.ndcg_full,
        "r10_delta": abs((m1.recall_at_10 or 0) - (m10.recall_at_10 or 0)),
        "r50_delta": abs((m1.recall_at_50 or 0) - (m10.recall_at_50 or 0)),
        "ndcg_delta": abs((m1.ndcg_full or 0) - (m10.ndcg_full or 0)),
        "published_r10": mpub.recall_at_10 if mpub else None,
        "published_r50": mpub.recall_at_50 if mpub else None,
        "published_ndcg": mpub.ndcg_full if mpub else None,
        "examples": changed[:8],
    }
    print(json.dumps({cond: report[cond]}, indent=2), flush=True)

print("\n=== ENCODER AGGREGATE SENSITIVITY TO BLAS THREADS (full scale, real encode) ===")
print(f"{'condition':<26}{'R@10 d':>10}{'R@50 d':>11}{'dNDCG':>11}"
      f"{'nchg':>6}{'maxdR':>7}{'xK10':>6}{'xK50':>6}{'xK100':>7}")
moved = False
for cond in CONDS:
    r = report[cond]
    print(f"{cond:<26}{r['r10_delta']:>10.4f}{r['r50_delta']:>11.5f}"
          f"{r['ndcg_delta']:>11.2e}{r['changed']:>6}{r['max_drank']:>7}"
          f"{r['cross_k10']:>6}{r['cross_k50']:>6}{r['cross_k100']:>7}")
    if r["r10_delta"] > 0 or r["r50_delta"] > 0 or r["ndcg_delta"] > 1e-6:
        moved = True
print(f"\nVERDICT: {'AGGREGATES MOVE' if moved else 'AGGREGATES STABLE'}")
Path("/tmp/encoder_full_result.json").write_text(json.dumps(report, indent=2) + "\n")
adapter.close()
