import sys, time, os
sys.path.insert(0,"/home/heheboiz/data/cir")
import numpy as np
from workbench.composition.adapters import CompositionAdapter
from workbench.composition.runner import _rank_groups_concurrently, _blas_thread_limit

# Real @-based scorer: the base class's, which ENCODER and LIMN inherit.
ad = CompositionAdapter.__new__(CompositionAdapter)
ad.model_id="encoder"
rng=np.random.default_rng(3)
N=512; D=512; G=74381
qv=(rng.random((N,D)).astype(np.float32))
Gm=(rng.random((G,D)).astype(np.float32))   # same shape as encoder's 512-d gallery
pos=[[int(rng.integers(0,G))] for _ in range(N)]
BS=32
groups=[(list(range(s,min(s+BS,N))), pos[s:min(s+BS,N)], qv[s:min(s+BS,N)], None)
        for s in range(0,N,BS)]
def fn(g):
    gi,gp,gv,_=g
    return ad.rank(gv, Gm, gp, None), []

print("OPENBLAS_NUM_THREADS =", os.environ.get("OPENBLAS_NUM_THREADS"), flush=True)
# serial reference at BLAS=1
_blas_thread_limit(1)
base=[]; t=time.perf_counter()
for g in groups: base.extend(fn(g)[0])
t_ser=time.perf_counter()-t
print(f"serial BLAS=1: {t_ser:6.2f}s  {N/t_ser:5.1f} q/s", flush=True)
_blas_thread_limit(int(os.environ.get("OPENBLAS_NUM_THREADS","1")))
t=time.perf_counter()
for g in groups: fn(g)
t_blasN=time.perf_counter()-t
print(f"serial BLAS={os.environ.get('OPENBLAS_NUM_THREADS')}: {t_blasN:6.2f}s {N/t_blasN:5.1f} q/s", flush=True)

for th in (2,4,8):
    res=_rank_groups_concurrently(fn, groups, th)
    got=[]
    for r in res: got.extend(r[0] if not isinstance(r,BaseException) else [])
    # time it
    t=time.perf_counter(); res=_rank_groups_concurrently(fn, groups, th); d=time.perf_counter()-t
    print(f"threads={th} (BLAS forced 1 inside): {d:6.2f}s {N/d:5.1f} q/s speedup_vs_serial={t_ser/d:4.2f}x ranks_identical={got==base}", flush=True)
