import sys, json, os, hashlib
sys.path.insert(0,"/home/heheboiz/data/cir")
import numpy as np
from pathlib import Path
from workbench.composition.runner import _rank_groups_concurrently
from workbench.composition.data import load_release
from workbench.composition.adapters import adapter_class

RUN="workbench/artifacts/composition/runs/497c25c57fbeba885ecfb9d1"
FEAT=f"{RUN}/work/habit/features"
G=np.load(f"{FEAT}/gallery_global.npy", mmap_mode="r")
data=load_release(Path("/home/heheboiz/data/cir/workbench/artifacts/circus_release"))
from workbench.composition.runner import _positive_positions
pos=_positive_positions(data)
saved=json.load(open(f"{RUN}/ranks/habit_fiq_n02.json"))["ranks"]
Cls=adapter_class("habit")
ad=Cls.__new__(Cls); ad.model_id="habit"
BS=32  # production batch_size
CONDS=("multimodal","text_only_black_image","image_only_empty_text")
out={}
for cond in CONDS:
    Q=np.load(f"{FEAT}/query_{cond}_global.npy", mmap_mode="r")
    qids=[q.qid for q in data.queries]
    N=len(qids)
    qv=np.asarray(Q[:N], dtype=np.float32)
    fdata=[q for q in data.queries]
    base=[]
    # SERIAL reference: exactly the adapter calls the loop makes at group size BS
    for s in range(0,N,BS):
        e=min(s+BS,N)
        ps=[ [p for p in pos.get(qids[i],()) if p<G.shape[0]] for i in range(s,e)]
        base.extend(ad.rank(qv[s:e], G, ps, [None]*(e-s)))
    # SHIPPED path: groups of BS, concurrency across groups
    for th in (1,2,4,8):
        groups=[]
        for s in range(0,N,BS):
            e=min(s+BS,N)
            ps=[ [p for p in pos.get(qids[i],()) if p<G.shape[0]] for i in range(s,e)]
            groups.append((list(range(s,e)), ps, qv[s:e], None))
        def fn(g):
            gi,gp,gv,gc=g
            return ad.rank(gv, G, gp, [None]*len(gi)), []
        res=_rank_groups_concurrently(fn, groups, th)
        got=[]
        for r in res:
            got.extend(r[0] if not isinstance(r,BaseException) else [])
        ident = got==base
        vs_saved = sum(1 for q,r in zip(qids,got) if r!=saved[cond].get(q))
        print(f"{cond} threads={th}: n={N} ranks_identical_to_serial={ident} mismatch_vs_saved={vs_saved}", flush=True)
        if not ident:
            d=[(qids[i],a,b) for i,(a,b) in enumerate(zip(base,got)) if a!=b]
            print(f"   DIFF {len(d)}: {d[:5]}",flush=True)
        out[f"{cond}|{th}"]={"identical":ident,"vs_saved":vs_saved}
    # anchor: serial baseline vs saved artifact
    anchor=sum(1 for q,r in zip(qids,base) if r!=saved[cond].get(q))
    print(f"{cond} SERIAL-vs-SAVED-ANCHOR: mismatch={anchor} (must be 0)", flush=True)
    out[f"{cond}|anchor"]={"serial_vs_saved":anchor}
Path("/tmp/d2_full_result.json").write_text(json.dumps(out,indent=2)+"\n")
