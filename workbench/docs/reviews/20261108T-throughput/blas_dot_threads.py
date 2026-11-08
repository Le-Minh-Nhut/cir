import sys, json
sys.path.insert(0,"/home/heheboiz/data/cir")
import numpy as np
from workbench.composition.runner import _blas_thread_limit, _digest

rng=np.random.default_rng(31)
# Real-ish shapes: 512-d encoder-style queries vs a gallery block.
Q=rng.random((256,512)).astype(np.float32)
G=rng.random((8192,512)).astype(np.float32)
pos=[int(rng.integers(0,8192)) for _ in range(256)]

def scores():
    return Q @ G.T

def ranks(S):
    out=[]
    for i,p in enumerate(pos):
        out.append(1+int(np.count_nonzero(S[i]>S[i][p])))
    return out

base_S=None; results={}
for t in (1,2,4,8,20):
    prev=_blas_thread_limit(t)
    try: S=scores()
    finally: _blas_thread_limit(prev)
    if base_S is None: base_S=S; base_r=ranks(S)
    r=ranks(S)
    ch=sum(1 for a,b in zip(base_r,r) if a!=b)
    results[t]={"score_max_abs_delta":float(np.abs(S-base_S).max()),
                "ranks_changed_vs_1":ch,
                "fingerprint_of_scores":_digest([float(S[i,pos[i]]) for i in range(len(pos))])[:16]}
    print(f"threads={t:<3} score_max|delta|={np.abs(S-base_S).max():.3e}  ranks_changed={ch}", flush=True)
json.dump(results, open("/tmp/dot_threads.json","w"), indent=2)
