"""Decide `optimize=True` as a METRIC question, on the real habit features.

The `optimize=False` side is taken from the ranks the finished run ACTUALLY SAVED -- that is the
production result, not a recomputation of it, so it needs no 43-minute re-derivation. Only the
`optimize=True` side is computed. A bounded recomputation of the False path then self-checks that
this harness reproduces the saved ranks on a real sample; if that fails, nothing here is trustworthy.
"""
import sys, json, time
sys.path.insert(0, "/home/heheboiz/data/cir")
import numpy as np
from pathlib import Path

from workbench.composition.data import load_release
from workbench.composition.runner import _positive_positions
from workbench.composition.adapters import CompositionAdapter
from workbench.composition.metrics import compute_condition_metrics

RUN = "workbench/artifacts/composition/runs/497c25c57fbeba885ecfb9d1"
FEAT = f"{RUN}/work/habit/features"
CONDS = ("multimodal", "text_only_black_image", "image_only_empty_text")
CHUNK = 128       # chunks the QUERY axis only: each query's score row is an independent dot
                  # product, so this cannot reorder any summation. Chunking the gallery would.
SELFCHECK = 64

G = np.load(f"{FEAT}/gallery_global.npy", mmap_mode="r")
GALLERY = G.shape[0]
data = load_release(Path("/home/heheboiz/data/cir/workbench/artifacts/circus_release"))
pos = _positive_positions(data)
qids = [q.qid for q in data.queries]
positions = [[p for p in pos.get(qid, ()) if p < GALLERY] for qid in qids]
saved = json.load(open(f"{RUN}/ranks/habit_fiq_n02.json"))["ranks"]
print(f"gallery={G.shape} queries={len(qids)}", flush=True)

def ranks_for(path, mode, limit=None):
    Q = np.load(path, mmap_mode="r")
    n = Q.shape[0] if limit is None else min(limit, Q.shape[0])
    out = []
    t = time.perf_counter()
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        qv = np.asarray(Q[s:e], dtype=np.float32)
        if mode == "false":
            sc = np.einsum("nd,gqd->ngq", qv, G).max(axis=-1)
        else:
            sc = np.einsum("nd,gqd->ngq", qv, G, optimize=True).max(axis=-1)
        out.extend(CompositionAdapter._ranks_from_scores(sc, positions[s:e]))
    return out, time.perf_counter() - t

report = {}
for cond in CONDS:
    qpath = f"{FEAT}/query_{cond}_global.npy"

    # --- self-check: does recomputing the UNITIALISED path reproduce what the run saved?
    recheck, t_check = ranks_for(qpath, "false", limit=SELFCHECK)
    check_bad = [qids[i] for i in range(SELFCHECK)
                 if (recheck[i] if positions[i] else None) != saved[cond].get(qids[i])]

    d_false = {qid: saved[cond].get(qid) for qid in qids}

    t_ranks, elapsed = ranks_for(qpath, "true")
    d_true = {qid: (t_ranks[i] if positions[i] else None) for i, qid in enumerate(qids)}

    changed = [(q, d_false.get(q), d_true.get(q)) for q in qids if d_false.get(q) != d_true.get(q)]
    deltas = [abs(a - b) for _, a, b in changed if a is not None and b is not None]

    m_false = compute_condition_metrics(cond, d_false)
    m_true = compute_condition_metrics(cond, d_true)
    report[cond] = {
        "r10_false": m_false.recall_at_10, "r10_true": m_true.recall_at_10,
        "r50_false": m_false.recall_at_50, "r50_true": m_true.recall_at_50,
        "ndcg_false": m_false.ndcg_full, "ndcg_true": m_true.ndcg_full,
        "ndcg_delta": abs((m_false.ndcg_full or 0) - (m_true.ndcg_full or 0)),
        "mrr_false": m_false.mrr, "mrr_true": m_true.mrr,
        "scored": m_false.scored_count,
        "selfcheck_mismatches": len(check_bad), "selfcheck_n": SELFCHECK,
        "queries_changed_rank": len(changed),
        "max_abs_drank": max(deltas) if deltas else 0,
        "true_path_seconds": round(elapsed, 1),
        "examples": changed[:5],
    }
    print(json.dumps({cond: report[cond]}, indent=2), flush=True)

print("\n=== SUMMARY TABLE ===", flush=True)
hdr = (f"{'condition':<26}{'R@10 F':>9}{'R@10 T':>9}{'R@50 F':>9}{'R@50 T':>9}"
       f"{'nDCG F':>15}{'nDCG T':>15}{'|dNDCG|':>11}{'nchg':>6}{'maxdR':>7}{'chk':>5}")
print(hdr, flush=True)
for cond in CONDS:
    r = report[cond]
    print(f"{cond:<26}{r['r10_false']:>9.4f}{r['r10_true']:>9.4f}"
          f"{r['r50_false']:>9.4f}{r['r50_true']:>9.4f}"
          f"{r['ndcg_false']:>15.10f}{r['ndcg_true']:>15.10f}"
          f"{r['ndcg_delta']:>11.2e}{r['queries_changed_rank']:>6}"
          f"{r['max_abs_drank']:>7}{r['selfcheck_mismatches']:>5}", flush=True)

accept = all(
    report[c]["r10_false"] == report[c]["r10_true"]
    and report[c]["r50_false"] == report[c]["r50_true"]
    and report[c]["ndcg_delta"] <= 1e-6
    and report[c]["selfcheck_mismatches"] == 0
    for c in CONDS)
print(f"\nACCEPT optimize=True = {accept}  (R@10 identical AND R@50 identical AND "
      f"|dNDCG|<=1e-6 AND selfcheck clean)", flush=True)
Path("/tmp/metric_decision.json").write_text(
    json.dumps({"accept": accept, "report": report}, indent=2) + "\n")
