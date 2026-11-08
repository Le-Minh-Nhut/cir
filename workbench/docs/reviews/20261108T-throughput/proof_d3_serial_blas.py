"""The unproven link: do serial BLAS=1 and serial BLAS=10 ranks actually agree?

`d3_hoist_proof` compared threads 2/4/8 against the serial-BLAS=1 baseline and printed a dash for
the serial-BLAS=10 row, so it never showed whether the pin is load-bearing or belt-and-braces. This
computes BOTH serial rankings on the same real `@`-path scorer and the same gallery and compares
them rank-by-rank, reporting a boolean plus how many queries change.
"""
import sys, json, os
sys.path.insert(0, "/home/heheboiz/data/cir")
import numpy as np
from pathlib import Path

from workbench.composition.adapters import CompositionAdapter
from workbench.composition.runner import _blas_thread_limit

adapter = CompositionAdapter.__new__(CompositionAdapter)
adapter.model_id = "encoder"          # ENCODER/LIMN inherit this @-based score

rng = np.random.default_rng(17)       # same seed as d3_hoist_proof: same data, comparable row
N, D, G = 512, 512, 74381
queries = rng.random((N, D)).astype(np.float32)
targets = rng.random((G, D)).astype(np.float32)
positions = [[int(rng.integers(0, G))] for _ in range(N)]
BS = 32


def serial_ranks(blas_threads):
    """Exactly the calls the serial branch makes, at a chosen BLAS thread count."""
    previous = _blas_thread_limit(int(blas_threads))
    try:
        out = []
        for s in range(0, N, BS):
            e = min(s + BS, N)
            out.extend(adapter.rank(queries[s:e], targets, positions[s:e], None))
        return out
    finally:
        _blas_thread_limit(previous)


ranks_1 = serial_ranks(1)
ranks_10 = serial_ranks(int(os.environ.get("OPENBLAS_NUM_THREADS", "10")))
identical = ranks_1 == ranks_10
changed = [(i, a, b) for i, (a, b) in enumerate(zip(ranks_1, ranks_10)) if a != b]
deltas = [abs(a - b) for _, a, b in changed if a is not None and b is not None]

print(f"OPENBLAS_NUM_THREADS in env = {os.environ.get('OPENBLAS_NUM_THREADS')}", flush=True)
print(f"serial BLAS=1  vs serial BLAS=10: queries={N}", flush=True)
print(f"  ranks_identical = {identical}", flush=True)
print(f"  queries_changed_rank = {len(changed)}", flush=True)
print(f"  max_abs_drank = {max(deltas) if deltas else 0}", flush=True)
for row in changed[:10]:
    print(f"    {row}", flush=True)

# Also record the raw score delta between the two BLAS settings, for the record.
def serial_scores(blas_threads):
    previous = _blas_thread_limit(int(blas_threads))
    try:
        return adapter.score(queries, targets)
    finally:
        _blas_thread_limit(previous)

s1, s10 = serial_scores(1), serial_scores(10)
print(f"  score_max_abs_delta = {float(np.abs(s1 - s10).max()):.3e}", flush=True)

verdict = ("LOAD-BEARING: the two BLAS settings produce different ranks, so the pin is the only "
           "reason the knob is currently safe" if not identical else
           "BELT-AND-BRACES: BLAS=1 and BLAS=10 produce identical ranks for this scorer, so the pin "
           "changes no number here; it remains correct hygiene against a scorer where it would")
print(f"VERDICT: {verdict}", flush=True)

Path("/tmp/d3_blas_serial_compare.json").write_text(json.dumps({
    "queries": N, "ranks_identical": identical,
    "queries_changed_rank": len(changed),
    "max_abs_drank": max(deltas) if deltas else 0,
    "score_max_abs_delta": float(np.abs(s1 - s10).max()),
    "blas_env": os.environ.get("OPENBLAS_NUM_THREADS"),
    "examples": changed[:10],
}, indent=2) + "\n")
