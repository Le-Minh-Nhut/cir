"""Corrected BLAS comparison: the REAL Adapter.score of three adapters, not the base class.

The earlier proof instantiated `CompositionAdapter.__new__` and called the BASE score
(`queries @ targets.T`). Both named adapters override it, and the three `optimize=True` einsum
adapters lower to a BLAS gemm too. So the affected set is 5 GEMM-path adapters, not 4 `@`.

Measured here, each through its own `score()`:
  conesep  - np.einsum("nd,mdk->nmk", optimize=True)  -> lowers to gemm. REAL full features exist.
  encoder  - queries @ gallery.T over 8192-row chunks -> batched gemm. Re-encoded (no cache).
  limn     - per-row gemv over a scope->gallery Mapping. Real preserved gallery features.

Only the BLAS thread count differs between the two sides of every comparison.
"""
import sys, json, time, argparse
sys.path.insert(0, "/home/heheboiz/data/cir")
from pathlib import Path
import numpy as np

from workbench.composition.runner import _blas_thread_limit, _positive_positions
from workbench.composition.data import load_release, assign_garment_categories
from workbench.composition.metrics import compute_condition_metrics
from workbench.composition.adapters import CompositionAdapter

CONESEP_FEAT = Path("/home/heheboiz/data/cir/workbench/artifacts/composition/runs/"
                    "30f6de423861cc1d6c78c361/work/conesep/features")
LIMN_GAL = Path("/home/heheboiz/data/cir/workbench/artifacts/composition/preserved_features/limn")
CONDS = ("multimodal", "text_only_black_image", "image_only_empty_text")
BS = 32          # production group shape


def ranks_from_scores(scores, positions):
    return CompositionAdapter._ranks_from_scores(scores, positions)


def metrics_for(ranks, qids):
    return compute_condition_metrics("c", dict(zip(qids, ranks)))


def compare(name, score_fn, vectors, positions, qids, out):
    """Score the SAME vectors at BLAS=1 and BLAS=10, then compare ranks and metrics."""
    per = {}
    for blas in (1, 10):
        prev = _blas_thread_limit(blas)
        try:
            t = time.perf_counter()
            scores = score_fn(vectors)
            elapsed = time.perf_counter() - t
        finally:
            _blas_thread_limit(prev)
        per[blas] = (scores, elapsed)

    (s1, t1), (s10, t10) = per[1], per[10]
    r1 = ranks_from_scores(s1, positions)
    r10 = ranks_from_scores(s10, positions)
    changed = [(q, a, b) for q, a, b in zip(qids, r1, r10) if a != b]
    deltas = [abs(a - b) for _, a, b in changed if a is not None and b is not None]
    m1, m10 = metrics_for(r1, qids), metrics_for(r10, qids)
    row = {
        "adapter": name, "queries": len(qids),
        "score_max_abs_delta": float(np.abs(s1 - s10).max()),
        "changed": len(changed), "max_drank": max(deltas) if deltas else 0,
        "cross_k10": sum(1 for _, a, b in changed if a and b and (a <= 10) != (b <= 10)),
        "cross_k50": sum(1 for _, a, b in changed if a and b and (a <= 50) != (b <= 50)),
        "cross_k100": sum(1 for _, a, b in changed if a and b and (a <= 100) != (b <= 100)),
        "r10_delta": abs((m1.recall_at_10 or 0) - (m10.recall_at_10 or 0)),
        "r50_delta": abs((m1.recall_at_50 or 0) - (m10.recall_at_50 or 0)),
        "ndcg_delta": abs((m1.ndcg_full or 0) - (m10.ndcg_full or 0)),
        "mrr_delta": abs((m1.mrr or 0) - (m10.mrr or 0)),
        "seconds_blas1": round(t1, 2), "seconds_blas10": round(t10, 2),
        "examples": changed[:6],
    }
    out.append(row)
    print(json.dumps(row, indent=2), flush=True)
    return row


def run_conesep(out):
    """REAL full features on disk: gallery 74381x8192, queries 6003x8192 per condition."""
    from workbench.composition.model_conesep import Adapter as ConeSep
    adapter = ConeSep.__new__(ConeSep)
    adapter.model_id = "conesep"
    gallery = np.asarray(np.load(CONESEP_FEAT / "gallery_global.npy", mmap_mode="r"), dtype=np.float32)
    data = load_release(Path("/home/heheboiz/data/cir/workbench/artifacts/circus_release"))
    pos = _positive_positions(data)
    qids_all = [q.qid for q in data.queries]
    positions = [[p for p in pos.get(q, ()) if p < gallery.shape[0]] for q in qids_all]
    for cond in CONDS:
        qpath = CONESEP_FEAT / f"query_{cond}_global.npy"
        if not qpath.exists():
            print(f"conesep/{cond}: NOT_RUN (no query features)", flush=True)
            continue
        queries = np.asarray(np.load(qpath, mmap_mode="r"), dtype=np.float32)
        print(f"--- conesep/{cond}", flush=True)

        def score_fn(v, _g=gallery, _a=adapter):
            pieces = [_a.score(v[s:s + BS], _g) for s in range(0, len(v), BS)]
            return np.concatenate(pieces, axis=0)

        compare(f"conesep:{cond}", score_fn, queries, positions, qids_all, out)


def run_limn(out, sample=600):
    """LIMN needs a scope->full-gallery Mapping; build the real per-category input."""
    from workbench.backend.operator_config import resolve_config
    from workbench.composition.data import load_release as lr
    from workbench.composition.model_limn import Adapter as Limn
    from workbench.composition.runner import image_resolver_factory

    cfg = resolve_config()
    data = assign_garment_categories(lr(Path("/home/heheboiz/data/cir/workbench/artifacts/circus_release")),
                                     Path(cfg.FASHIONIQ_ROOT))
    adapter = Limn(cfg, "base_iter0_all_categories")
    adapter.work_root = Path("/tmp/limn_blas_work")
    adapter.work_root.mkdir(parents=True, exist_ok=True)
    galleries = {c: np.asarray(np.load(LIMN_GAL / f"gallery_{c}.npy", mmap_mode="r"), dtype=np.float32)
                 for c in ("dress", "shirt", "toptee")}
    resolver = image_resolver_factory(Path(cfg.FASHIONIQ_ROOT), "fashioniq_standard")
    pos = _positive_positions(data)
    sel = [q for q in data.queries if q.category][:sample]
    paths = [resolver(q.reference_id, data.candidate_names) for q in sel]
    keep = [i for i, p in enumerate(paths) if p is not None]
    sel = [sel[i] for i in keep]
    paths = [paths[i] for i in keep]
    for cond in CONDS:
        print(f"--- limn/{cond} (encoding {len(sel)} queries)", flush=True)
        vectors = np.asarray(adapter.encode_queries(
            [str(p) for p in paths], [q.query_txt for q in sel], [q.category for q in sel]),
            dtype=np.float32)
        qids = [q.qid for q in sel]
        cats = [q.category for q in sel]
        positions = [[p for p in pos.get(q, ()) if p < 74381] for q in qids]

        def score_fn(v, _g=galleries, _c=cats, _a=adapter):
            out_scores = np.empty((len(v), 74381), dtype=np.float32)
            for s in range(0, len(v), BS):
                e = min(s + BS, len(v))
                out_scores[s:e] = _a.score(v[s:e], _g, _c[s:e])
            return out_scores

        compare(f"limn:{cond}", score_fn, vectors, positions, qids, out)
    adapter.close()


def run_encoder(out, sample=None):
    """ENCODER has no cache: re-encode its gallery and queries with the real adapter."""
    from workbench.backend.operator_config import resolve_config
    from workbench.composition.data import load_release as lr
    from workbench.composition.model_encoder import Adapter as Encoder
    from workbench.composition.runner import image_resolver_factory

    cfg = resolve_config()
    data = lr(Path("/home/heheboiz/data/cir/workbench/artifacts/circus_release"))
    adapter = Encoder(cfg, "fashioniq")
    adapter.work_root = Path("/tmp/encoder_blas_work")
    adapter.work_root.mkdir(parents=True, exist_ok=True)
    resolver = image_resolver_factory(Path(cfg.FASHIONIQ_ROOT), "fashioniq_standard")
    pos = _positive_positions(data)
    print("encoder: encoding gallery ...", flush=True)
    gpaths = [resolver(i, data.candidate_names) for i in data.gallery.image_ids]
    gpaths = [p for p in gpaths if p is not None]
    gallery = np.asarray(adapter.encode_targets([str(p) for p in gpaths], [None] * len(gpaths)),
                         dtype=np.float32)
    print(f"encoder: gallery {gallery.shape}", flush=True)
    sel = [q for q in data.queries]
    paths = [resolver(q.reference_id, data.candidate_names) for q in sel]
    keep = [i for i, p in enumerate(paths) if p is not None]
    if sample:
        keep = keep[:sample]
    sel = [sel[i] for i in keep]
    paths = [paths[i] for i in keep]
    for cond in CONDS:
        print(f"--- encoder/{cond}", flush=True)
        vectors = np.asarray(adapter.encode_queries(
            [str(p) for p in paths], [q.query_txt for q in sel], [None] * len(sel)), dtype=np.float32)
        qids = [q.qid for q in sel]
        positions = [[p for p in pos.get(q, ()) if p < gallery.shape[0]] for q in qids]

        def score_fn(v, _g=gallery, _a=adapter):
            return _a.score(v, _g)

        compare(f"encoder:{cond}", score_fn, vectors, positions, qids, out)
    adapter.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="conesep,limn,encoder")
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--out", default="/tmp/blas_three_adapters.json")
    args = ap.parse_args()
    out = []
    for name in [x.strip() for x in args.only.split(",") if x.strip()]:
        try:
            {"conesep": run_conesep, "limn": lambda o: run_limn(o, args.sample or 600),
             "encoder": lambda o: run_encoder(o, args.sample)}[name](out)
        except Exception as error:
            print(f"NOT_RUN {name}: {type(error).__name__}: {error}", flush=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print("\n=== SUMMARY ===")
    print(f"{'adapter:condition':<34}{'dscore':>10}{'nchg':>6}{'maxdR':>7}"
          f"{'xK10':>6}{'xK50':>6}{'dR10':>8}{'dR50':>8}{'dNDCG':>11}")
    for r in out:
        print(f"{r['adapter']:<34}{r['score_max_abs_delta']:>10.2e}{r['changed']:>6}"
              f"{r['max_drank']:>7}{r['cross_k10']:>6}{r['cross_k50']:>6}"
              f"{r['r10_delta']:>8.4f}{r['r50_delta']:>8.4f}{r['ndcg_delta']:>11.2e}")
