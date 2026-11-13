import sys, time, json, resource
sys.path.insert(0,'/home/heheboiz/data/cir')
from workbench.backend import parity_gate as gate
def hwm():
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM"): return int(line.split()[1])/1024.0
rss0=open("/proc/self/status").read()
base=[l for l in rss0.splitlines() if l.startswith("VmRSS")][0]
t=time.perf_counter(); gate.evaluate_all(); el=time.perf_counter()-t
print(json.dumps({"elapsed_s":round(el,3),"rss_before":base.strip(),"VmHWM_after_mib":round(hwm(),1)}))
