import sys, time, json
sys.path.insert(0,'/home/heheboiz/data/cir')
from workbench.backend import parity_gate as gate
t0=time.perf_counter(); gate.evaluate_model("limn"); pre=time.perf_counter()-t0  # warm imports
out=[]; total0=time.perf_counter()
for m in ("conesep","encoder","habit","hint","intent","limn","tgcir"):
    t=time.perf_counter(); v=gate.evaluate_model(m); out.append((m, round(time.perf_counter()-t,3), v.status))
print(json.dumps({"per_model":out,"sum_s":round(time.perf_counter()-total0,3)}, indent=0))
