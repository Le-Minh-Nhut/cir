"""Hoist proof: an @-based scorer must give identical ranks at threads=1 and threads=8.

Before the hoist, threads=1 ran with BLAS at whatever the env set (10 via composition_eval) while
threads=8 ran with BLAS=1. BLAS thread count is not numerically neutral on this contraction, so the
knob could change publishable ranks. With the single pin hoisted above both branches, both settings
compute under BLAS=1 and must agree exactly.

Also reports the throughput trade: forcing BLAS=1 in the serial branch costs speed there, and that
cost is stated rather than hidden.
"""
import sys, time, os
sys.path.insert(0, "/home/heheboiz/data/cir")
import numpy as np

from workbench.composition.adapters import CompositionAdapter
from workbench.composition.runner import _rank_groups_concurrently, _blas_thread_limit

adapter = CompositionAdapter.__new__(CompositionAdapter)
adapter.model_id = "encoder"          # ENCODER/LIMN inherit this @-based score

rng = np.random.default_rng(17)
N, D, G = 512, 512, 74381
queries = rng.random((N, D)).astype(np.float32)
targets = rng.random((G, D)).astype(np.float32)
positions = [[int(rng.integers(0, G))] for _ in range(N)]
BS = 32
groups = [(list(range(s, min(s + BS, N))), positions[s:min(s + BS, N)],
           queries[s:min(s + BS, N)], None) for s in range(0, N, BS)]

def rank_group(group):
    indices, group_positions, vectors, _ = group
    return adapter.rank(vectors, targets, group_positions, None), []

print(f"OPENBLAS_NUM_THREADS in env = {os.environ.get('OPENBLAS_NUM_THREADS')} "
      f"(the value a serial run used to inherit)", flush=True)

def collect(threads):
    out = []
    for outcome in _rank_groups_concurrently(rank_group, groups, threads):
        if isinstance(outcome, BaseException):
            raise outcome
        out.extend(outcome[0])
    return out

# Serial reference as a threads=1 run would now compute it (BLAS pinned by the hoist).
t = time.perf_counter(); base = collect(1); t1 = time.perf_counter() - t
print(f"threads=1 (BLAS pinned to 1): {t1:6.2f}s  {N/t1:6.1f} q/s", flush=True)

# What the serial branch WOULD have measured before the hoist, for the trade report.
previous = _blas_thread_limit(int(os.environ.get("OPENBLAS_NUM_THREADS", "10")))
t = time.perf_counter()
for g in groups:
    rank_group(g)
t_unpinned = time.perf_counter() - t
_blas_thread_limit(previous)
print(f"serial WITHOUT the pin (BLAS={os.environ.get('OPENBLAS_NUM_THREADS')}): "
      f"{t_unpinned:6.2f}s  {N/t_unpinned:6.1f} q/s", flush=True)

result = {"threads_1": {"seconds": round(t1, 2), "q_per_s": round(N / t1, 1)},
          "serial_unpinned": {"seconds": round(t_unpinned, 2),
                              "q_per_s": round(N / t_unpinned, 1),
                              "blas_env": os.environ.get("OPENBLAS_NUM_THREADS")}}
for threads in (2, 4, 8):
    t = time.perf_counter(); got = collect(threads); elapsed = time.perf_counter() - t
    identical = got == base
    print(f"threads={threads} (BLAS pinned to 1): {elapsed:6.2f}s  {N/elapsed:6.1f} q/s  "
          f"ranks_identical_to_threads_1={identical}", flush=True)
    result[f"threads_{threads}"] = {"seconds": round(elapsed, 2),
                                    "q_per_s": round(N / elapsed, 1),
                                    "identical_to_threads_1": identical}
    if not identical:
        diff = [(i, a, b) for i, (a, b) in enumerate(zip(base, got)) if a != b]
        print(f"   MISMATCH {len(diff)}: {diff[:5]}", flush=True)

ok = all(result[f"threads_{t}"]["identical_to_threads_1"] for t in (2, 4, 8))
print(f"VERDICT: {'PASS' if ok else 'FAIL'} - @-based scorer ranks identical across "
      f"WORKBENCH_SCORE_THREADS 1/2/4/8 under the hoisted BLAS pin", flush=True)
from pathlib import Path
Path("/tmp/d3_hoist_result.json").write_text(__import__("json").dumps(result, indent=2) + "\n")
