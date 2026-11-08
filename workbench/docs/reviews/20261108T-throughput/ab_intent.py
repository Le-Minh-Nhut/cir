import sys, json, time
from pathlib import Path
sys.path.insert(0,"/home/heheboiz/data/cir")
import torch
from PIL import Image
from workbench.composition.adapters import prefetch_rows
from workbench.composition.runner import image_resolver_factory
from workbench.backend.registry import load_registry
from workbench.replay.habit import _load_replay_modules
m=next(r for r in load_registry()["models"] if r["model_id"]=="intent")
_,_,du,_,_=_load_replay_modules(Path("/home/heheboiz/data/cir/workbench/third_party")/m["source_dir"])
transform=du.targetpad_transform(1.25,224)   # same transform model_intent.py uses
audit=json.loads(Path("/home/heheboiz/data/cir/workbench/artifacts/circus_release/shortcut_audit/retrieval_data_fashioniq_task7_aggregated_best_rank_mm_any.json").read_text())
cand={}
for line in Path("/home/heheboiz/data/cir/workbench/artifacts/circus_release/mbeir_fashioniq_task7_cand_pool.jsonl").open():
    r=json.loads(line); cand[r["did"]]=r["img_path"]
res=image_resolver_factory(Path("/home/heheboiz/data/cir/data/FashionIQ"),"fashioniq_standard")
files=[str(f) for f in (res(x,cand) for x in audit["gallery_paths"][:512]) if f]
BS=64; batches=[files[s:s+BS] for s in range(0,len(files),BS)]
def dec(ch):
    o=[]
    for p in ch:
        with Image.open(p) as im: o.append(transform(im.convert("RGB")))
    return torch.stack(o)
def bench(threads, workers, label):
    torch.set_num_threads(threads)
    ser=[dec(b) for b in batches]
    t=time.perf_counter(); out=list(prefetch_rows(dec,batches,workers)); d=time.perf_counter()-t
    same=all(torch.equal(a,b) for a,b in zip(ser,out))
    delta=max(float((a-b).abs().max()) for a,b in zip(ser,out))
    print(f"{label}: threads={threads} workers={workers} {d:5.2f}s {len(files)/d:7.1f} img/s "
          f"bitwise_equal={same} maxabs_delta={delta} proj_74381={74381/(len(files)/d)/60:5.1f}min",flush=True)
bench(20,1,"INTENT BEFORE")
bench(8,8,"INTENT AFTER ")
