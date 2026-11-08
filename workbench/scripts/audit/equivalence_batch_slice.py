import json,sys
import numpy as np
sys.path.insert(0,"/home/heheboiz/data/cir")
from workbench.scripts.audit.equivalence_fastpath import score_gemm, score_old, ranks_from_scores
RUN="/home/heheboiz/data/cir/workbench/artifacts/composition/runs/497c25c57fbeba885ecfb9d1"
F=RUN+"/work/habit/features"
OUT="/home/heheboiz/data/cir/workbench/docs/reviews/20261108T153045-audit-equivalence"
m=json.load(open(OUT+"/query_manifest.json")); qids=m["query_order"][:600]
pos=[tuple(m["positive_positions"][q]) for q in qids]
g=np.load(F+"/gallery_global.npy",mmap_mode="r")
def opt(q,t): return np.einsum("nd,gqd->ngq",q,t,optimize=True).max(axis=-1)
def run(qa,batch,path):
    r=[]
    for s in range(0,len(qa),batch):
        r.extend(ranks_from_scores(path(qa[s:s+batch],g),pos[s:s+batch]))
    return r
cond="multimodal"
qa=np.asarray(np.load(F+f"/query_{cond}_global.npy",mmap_mode="r")[:600],dtype=np.float32)
res={"queries":600,"condition":cond}
for name,path in (("gemm1024",lambda q,t: score_gemm(q,t,1024)),("opt",opt),("old",score_old)):
    r32=run(qa,32,path); r1=run(qa,1,path)
    ch=[{"qid":qids[i],"rank_b32":r32[i],"rank_b1":r1[i]} for i in range(len(qids)) if r32[i]!=r1[i]]
    res[name]={"batch1_vs_32_mismatches":len(ch),"examples":ch[:10]}
    print(name,json.dumps(res[name]),flush=True)
json.dump(res,open(OUT+"/batch_invariance.json","w"),indent=1)
