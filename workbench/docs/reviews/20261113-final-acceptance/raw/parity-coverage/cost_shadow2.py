import importlib.util, sys, time, json
sys.path.insert(0,'/home/heheboiz/data/cir')
from workbench.backend import parity_gate as real
ROOT=real.evidence_root()
def load(name, path):
    spec=importlib.util.spec_from_file_location(name, path)
    m=importlib.util.module_from_spec(spec); sys.modules[name]=m; spec.loader.exec_module(m); return m
accum = load("gate_accum","/tmp/costfix/gate_accum.py")
assert accum.component_digest([0.5,0.25])==real.component_digest([0.5,0.25])
out=[]
for label, mod in (("real", real), ("accum+bulk", accum), ("real#2", real), ("accum+bulk#2", accum)):
    ts=[]
    for _ in range(2):
        t=time.perf_counter(); v=mod.evaluate_model("encoder", root=ROOT); ts.append(time.perf_counter()-t)
    out.append({"variant":label,"runs":[round(x,3) for x in ts],"best":round(min(ts),3),
                "failures":len(v.failures),"status":v.status,
                "md5_of_verdict":__import__("hashlib").md5(json.dumps(v.to_dict(),sort_keys=True).encode()).hexdigest()})
print(json.dumps(out, indent=1))
