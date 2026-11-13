"""Pre-8214175 tree: the old preview reader, against the artifacts of that same commit."""
import importlib.util, sys, time, json
from pathlib import Path
spec=importlib.util.spec_from_file_location("gate_prev","/tmp/prevtree/parity_gate_prev.py")
m=importlib.util.module_from_spec(spec); sys.modules["gate_prev"]=m; spec.loader.exec_module(m)
ROOT=Path("/tmp/prevtree")
print("gate from:", m.__file__, "| COVERAGE:", m.COMPONENT_COVERAGE)
ts=[]
for _ in range(3):
    t=time.perf_counter(); res=m.evaluate_all(root=ROOT); ts.append(time.perf_counter()-t)
import resource
print(json.dumps({"runs":[round(x,3) for x in ts],"best_s":round(min(ts),3),
 "peak_rss_mib":round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1),
 "models":{k:v["status"] for k,v in res["models"].items()}}, indent=1))
