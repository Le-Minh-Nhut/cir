import json,sys
import numpy as np
sys.path.insert(0,"/home/heheboiz/data/cir")
from workbench.scripts.audit.equivalence_fastpath import score_gemm, ranks_from_scores
RUN="/home/heheboiz/data/cir/workbench/artifacts/composition/runs/497c25c57fbeba885ecfb9d1"
F=RUN+"/work/habit/features"
OUT="/home/heheboiz/data/cir/workbench/docs/reviews/20261108T153045-audit-equivalence"
m=json.load(open(OUT+"/query_manifest.json")); qids=m["query_order"]
pos=[tuple(m["positive_positions"][q]) for q in qids]
g=np.load(F+"/gallery_global.npy",mmap_mode="r")
qa=np.asarray(np.load(F+"/query_multimodal_global.npy",mmap_mode="r"),dtype=np.float32)
def run(batch):
    r=[]
    for s in range(0,len(qa),batch):
        r.extend(ranks_from_scores(score_gemm(qa[s:s+batch],g,1024),pos[s:s+batch]))
    return r
r32=run(32); r1=run(1)
ch=[{"qid":qids[i],"rank_b32":r32[i],"rank_b1":r1[i]}
    for i in range(len(qids)) if r32[i]!=r1[i]]
res={"condition":"multimodal","queries":len(qids),
     "batch1_vs_32":{"mismatches":len(ch),"all":ch}}
json.dump(res,open(OUT+"/batch_invariance_gemm.json","w"),indent=1)
print(json.dumps(res),flush=True)
