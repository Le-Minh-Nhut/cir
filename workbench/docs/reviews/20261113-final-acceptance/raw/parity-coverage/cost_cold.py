import resource, sys, time, json
sys.path.insert(0,'/home/heheboiz/data/cir')
from workbench.backend import parity_gate as gate

def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0  # MiB

r0=rss(); t0=time.perf_counter()
res = gate.evaluate_all()
t1=time.perf_counter()
r1=rss()
print(json.dumps({"total_s": round(t1-t0,3), "rss_before_mib": round(r0,1),
                  "rss_peak_after_mib": round(r1,1), "rss_delta_mib": round(r1-r0,1),
                  "models": sorted(res["models"])}))
