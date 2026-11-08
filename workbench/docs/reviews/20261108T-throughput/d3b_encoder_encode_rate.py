import sys, time
sys.path.insert(0,"/home/heheboiz/data/cir")
from pathlib import Path
from workbench.backend.operator_config import resolve_config
from workbench.composition.data import load_release, assign_garment_categories
from workbench.composition.runner import image_resolver_factory
from workbench.composition.adapters import adapter_class
cfg=resolve_config()
data=assign_garment_categories(load_release(Path("/home/heheboiz/data/cir/workbench/artifacts/circus_release")), Path(cfg.FASHIONIQ_ROOT))
res=image_resolver_factory(Path(cfg.FASHIONIQ_ROOT),"fashioniq_standard")
Cls=adapter_class("encoder"); ad=Cls(cfg,"fashioniq")
ad.work_root=Path("/tmp/enc_probe"); ad.work_root.mkdir(exist_ok=True)
qs=[q for q in data.queries][:256]
paths=[res(q.reference_id, data.candidate_names) for q in qs]
keep=[i for i,p in enumerate(paths) if p is not None]
qs=[qs[i] for i in keep]; paths=[paths[i] for i in keep]
N=len(paths); print("queries",N,flush=True)
t=time.perf_counter()
v=ad.encode_queries([str(p) for p in paths],[q.query_txt for q in qs],[None]*N)
d=time.perf_counter()-t
print(f"encode {N} in {d:.1f}s = {N/d:.2f} q/s -> 6003x3 = {3*6003*d/N/60:.1f} min",flush=True)
print("dim",v.shape,flush=True)
ad.close()
