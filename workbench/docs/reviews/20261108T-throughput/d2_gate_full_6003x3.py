"""D2 gate: the SHIPPED ranking path vs the published rank artifact, full 6003 x 3.

The saved artifact was produced by the serial `adapter.rank` path, so it IS the reference and does
not need re-deriving (that recomputation is what made the first attempt a ~3 hour job). This runs
only the shipped concurrent path at the production grouping (chunk_size = batch_size = 32) and
compares every rank against the artifact.

Progress is printed per condition so a stall is visible rather than silent.
"""
import sys, json, time
sys.path.insert(0, "/home/heheboiz/data/cir")
import numpy as np
from pathlib import Path

from workbench.composition.runner import (_rank_groups_concurrently, _positive_positions,
                                         resolved_score_threads)
from workbench.composition.data import load_release
from workbench.composition.adapters import adapter_class

RUN = "workbench/artifacts/composition/runs/497c25c57fbeba885ecfb9d1"
FEAT = f"{RUN}/work/habit/features"
CONDS = ("multimodal", "text_only_black_image", "image_only_empty_text")
BS = 32

G = np.load(f"{FEAT}/gallery_global.npy", mmap_mode="r")
data = load_release(Path("/home/heheboiz/data/cir/workbench/artifacts/circus_release"))
pos = _positive_positions(data)
saved = json.load(open(f"{RUN}/ranks/habit_fiq_n02.json"))["ranks"]
qids = [q.qid for q in data.queries]
Cls = adapter_class("habit")
adapter = Cls.__new__(Cls)          # rank() is pure numpy; no worker needed
adapter.model_id = "habit"
THREADS = resolved_score_threads()
print(f"gallery={G.shape} queries={len(qids)} threads={THREADS} chunk_size={BS}", flush=True)

result = {"threads": THREADS, "chunk_size": BS, "conditions": {}}
overall_ok = True
for cond in CONDS:
    t0 = time.perf_counter()
    Q = np.load(f"{FEAT}/query_{cond}_global.npy", mmap_mode="r")
    qv = np.asarray(Q[:len(qids)], dtype=np.float32)

    groups = []
    for s in range(0, len(qids), BS):
        e = min(s + BS, len(qids))
        group_pos = [[p for p in pos.get(qids[i], ()) if p < G.shape[0]] for i in range(s, e)]
        groups.append((list(range(s, e)), group_pos, qv[s:e], None))

    def rank_group(group):
        group_indices, group_pos, group_vecs, _ = group
        return adapter.rank(group_vecs, G, group_pos, [None] * len(group_indices)), []

    outcomes = _rank_groups_concurrently(rank_group, groups, THREADS)
    ranks = []
    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            raise outcome
        ranks.extend(outcome[0])

    mismatches = [(q, saved[cond].get(q), r) for q, r in zip(qids, ranks)
                  if r != saved[cond].get(q)]
    elapsed = time.perf_counter() - t0
    ok = not mismatches
    overall_ok &= ok
    result["conditions"][cond] = {
        "queries": len(qids), "elapsed_s": round(elapsed, 1),
        "pairs_per_s": round(len(qids) / elapsed, 1),
        "rank_mismatches_vs_artifact": len(mismatches),
        "identical": ok, "examples": mismatches[:5],
    }
    print(f"{cond}: n={len(qids)} {elapsed:6.1f}s {len(qids)/elapsed:5.1f} q/s "
          f"mismatches={len(mismatches)} identical={ok}", flush=True)

result["all_identical"] = overall_ok
print(f"VERDICT: {'PASS' if overall_ok else 'FAIL'} - shipped path reproduces the published "
      f"ranks exactly on all {len(qids) * len(CONDS)} query-condition pairs", flush=True)
Path("/tmp/d2_result.json").write_text(json.dumps(result, indent=2) + "\n")
