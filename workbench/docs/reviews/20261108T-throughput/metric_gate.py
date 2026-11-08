"""Metric gate for the uniform scoring path, including the path that beats optimize=True.

Tests THREE paths on the real habit features and the real saved ranks:
  saved   - the ranks the finished run actually wrote (production `einsum`, optimize=False)
  opt     - np.einsum(..., optimize=True)
  gemm    - gallery-chunked GEMM (chunk configurable), the fastest path

For each: production metrics via compute_condition_metrics, per-query rank changes, max |drank|,
and -- the gate that actually matters -- how many changed ranks CROSS the K=10 or K=50 boundary.
A rank change can only move R@10/R@50 if the pair straddles the boundary.
"""
import sys, json, time, argparse
sys.path.insert(0, "/home/heheboiz/data/cir")
import numpy as np
from pathlib import Path

from workbench.composition.data import load_release
from workbench.composition.runner import _positive_positions
from workbench.composition.adapters import CompositionAdapter
from workbench.composition.metrics import compute_condition_metrics

ap = argparse.ArgumentParser()
ap.add_argument("--chunk", type=int, default=4096)
ap.add_argument("--qchunk", type=int, default=128, help="query-axis chunk; cannot reorder sums")
ap.add_argument("--conds", default="multimodal,text_only_black_image,image_only_empty_text")
ap.add_argument("--out", default="/tmp/metric_gate.json")
args = ap.parse_args()

RUN = "workbench/artifacts/composition/runs/497c25c57fbeba885ecfb9d1"
FEAT = f"{RUN}/work/habit/features"
G = np.load(f"{FEAT}/gallery_global.npy", mmap_mode="r")
GALLERY = G.shape[0]
Gflat = np.asarray(G, dtype=np.float32).reshape(-1, G.shape[-1])
data = load_release(Path("/home/heheboiz/data/cir/workbench/artifacts/circus_release"))
pos = _positive_positions(data)
qids = [q.qid for q in data.queries]
positions = [[p for p in pos.get(qid, ()) if p < GALLERY] for qid in qids]
saved = json.load(open(f"{RUN}/ranks/habit_fiq_n02.json"))["ranks"]

def score_true(qv):
    return np.einsum("nd,gqd->ngq", qv, G, optimize=True).max(axis=-1)

def score_gemm(qv, chunk):
    out = np.empty((len(qv), GALLERY), dtype=np.float32)
    for s in range(0, GALLERY, chunk):
        e = min(s + chunk, GALLERY)
        block = Gflat[s * G.shape[1]:e * G.shape[1]]
        out[:, s:e] = (qv @ block.T).reshape(len(qv), e - s, G.shape[1]).max(axis=-1)
    return out

def ranks(path, kind, chunk):
    Q = np.load(path, mmap_mode="r")
    n = Q.shape[0]
    got = []
    t = time.perf_counter()
    for s in range(0, n, args.qchunk):
        e = min(s + args.qchunk, n)
        qv = np.asarray(Q[s:e], dtype=np.float32)
        sc = score_true(qv) if kind == "opt" else score_gemm(qv, chunk)
        got.extend(CompositionAdapter._ranks_from_scores(sc, positions[s:e]))
    return got, time.perf_counter() - t

def crossings(changed, k):
    return sum(1 for _, a, b in changed if a is not None and b is not None
               and ((a <= k) != (b <= k)))

report = {}
for cond in [c for c in args.conds.split(",") if c]:
    qpath = f"{FEAT}/query_{cond}_global.npy"
    d_saved = {qid: saved[cond].get(qid) for qid in qids}
    out = {}
    for kind in ("opt", "gemm"):
        got, elapsed = ranks(qpath, kind, args.chunk)
        d_new = {qid: (got[i] if positions[i] else None) for i, qid in enumerate(qids)}
        changed = [(q, d_saved.get(q), d_new.get(q)) for q in qids if d_saved.get(q) != d_new.get(q)]
        deltas = [abs(a - b) for _, a, b in changed if a is not None and b is not None]
        m0 = compute_condition_metrics(cond, d_saved)
        m1 = compute_condition_metrics(cond, d_new)
        out[kind] = {
            "r10_saved": m0.recall_at_10, "r10_new": m1.recall_at_10,
            "r50_saved": m0.recall_at_50, "r50_new": m1.recall_at_50,
            "ndcg_saved": m0.ndcg_full, "ndcg_new": m1.ndcg_full,
            "ndcg_delta": abs((m0.ndcg_full or 0) - (m1.ndcg_full or 0)),
            "mrr_delta": abs((m0.mrr or 0) - (m1.mrr or 0)),
            "changed": len(changed), "max_drank": max(deltas) if deltas else 0,
            "cross10": crossings(changed, 10), "cross50": crossings(changed, 50),
            "deep_rank_changes": sum(1 for _, a, b in changed
                                     if a is not None and b is not None and min(a, b) > 50),
            "seconds": round(elapsed, 1),
            "examples": changed[:6],
        }
    report[cond] = out
    print(json.dumps({cond: out}, indent=2), flush=True)

print(f"\n=== SCORING PATH GATE (chunk={args.chunk}, qchunk={args.qchunk}) ===", flush=True)
hdr = (f"{'condition':<26}{'path':<6}{'R@10':>9}{'R@50':>9}{'|dNDCG|':>11}"
       f"{'nchg':>6}{'maxdR':>7}{'xK10':>6}{'xK50':>6}{'sec':>6}")
print(hdr, flush=True)
ok = True
for cond, out in report.items():
    for kind, r in out.items():
        print(f"{cond:<26}{kind:<6}{r['r10_new']:>9.4f}{r['r50_new']:>9.4f}"
              f"{r['ndcg_delta']:>11.2e}{r['changed']:>6}{r['max_drank']:>7}"
              f"{r['cross10']:>6}{r['cross50']:>6}{r['seconds']:>6.0f}", flush=True)
        if r["r10_saved"] != r["r10_new"] or r["r50_saved"] != r["r50_new"]:
            ok = False
        if r["ndcg_delta"] > 1e-6:
            ok = False
print(f"\nVERDICT: {'ACCEPT' if ok else 'REJECT'} -- R@10 and R@50 identical per condition, "
      f"|dNDCG| <= 1e-6", flush=True)
Path(args.out).write_text(json.dumps({"accept": ok, "report": report}, indent=2) + "\n")
