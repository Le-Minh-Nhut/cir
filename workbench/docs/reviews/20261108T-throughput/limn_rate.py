import sys, json, time
sys.path.insert(0,"/home/heheboiz/data/cir")
from pathlib import Path
from workbench.composition.data import load_release, assign_garment_categories
from workbench.composition.runner import image_resolver_factory
from workbench.composition.adapters import adapter_class
from workbench.backend.operator_config import resolve_config

cfg=resolve_config()
data=load_release(Path("/home/heheboiz/data/cir/workbench/artifacts/circus_release"))
data=assign_garment_categories(data, Path(cfg.FASHIONIQ_ROOT))
res=image_resolver_factory(Path(cfg.FASHIONIQ_ROOT),"fashioniq_standard")
Cls=adapter_class("limn")
ad=Cls(cfg, "base_iter0_all_categories")
ad.work_root=Path("/tmp/limn_probe_work"); ad.work_root.mkdir(exist_ok=True)
qs=[q for q in data.queries][:32]
paths=[res(q.reference_id, data.candidate_names) for q in qs]
keep=[i for i,p in enumerate(paths) if p is not None]
print("sample", len(keep), "of", len(qs), flush=True)
qs=[qs[i] for i in keep]; paths=[paths[i] for i in keep]
t=time.perf_counter()
out=ad.encode_queries([str(p) for p in paths],[q.query_txt for q in qs],[q.category for q in qs])
d=time.perf_counter()-t
print(f"encode {len(paths)} queries in {d:.1f}s = {len(paths)/d:.1f} q/s", flush=True)
print(f"=> 6003 queries: {6003*d/len(paths)/60:.1f} min per condition, x3 = {3*6003*d/len(paths)/60:.1f} min", flush=True)
ad.close()
