import sys, numpy as np
sys.path.insert(0,'/home/heheboiz/data/cir')
from workbench.composition.adapters import CompositionAdapter
g=np.asarray(np.load('workbench/artifacts/composition/preserved_features/limn/gallery_dress.npy',mmap_mode='r'),dtype=np.float32)
A=CompositionAdapter.__new__(CompositionAdapter); A.model_id='audit'
rng=np.random.default_rng(0)
pos=rng.choice(len(g),200,replace=False)
Q=np.stack([g[p] for p in pos])
tg=[[int(p)] for p in pos]
f1=A.score(Q,g); f32=np.concatenate([A.score(Q[i:i+32],g) for i in range(0,len(Q),32)])
r1=CompositionAdapter._ranks_from_scores(f1,tg); r32=CompositionAdapter._ranks_from_scores(f32,tg)
d=[abs(a-b) for a,b in zip(r1,r32)]
print('queries',len(Q),'max_abs_score_delta',float(np.abs(f1-f32).max()))
print('ranks differing:',sum(1 for a,b in zip(r1,r32) if a!=b))
print('max rank delta',max(d),'mean rank',float(np.mean(r1)))
print('rank==1 count batch1',sum(1 for r in r1 if r==1),'batch32',sum(1 for r in r32 if r==1))
# also compare to brute force on a subset
import json
print('sample ranks',list(zip(r1[:8],r32[:8])))
