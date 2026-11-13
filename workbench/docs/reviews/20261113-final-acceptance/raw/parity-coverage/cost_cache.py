"""What would a (mtime,size) memo save? Cost of the stamp scan itself vs full evaluation."""
import sys, time, json, hashlib
sys.path.insert(0,'/home/heheboiz/data/cir')
from workbench.backend import parity_gate as gate
ROOT=gate.evidence_root()
t=time.perf_counter()
stamp=[]
for p in sorted(ROOT.glob("g1_g2.*.json")):
    st=p.stat(); stamp.append((p.name, st.st_mtime_ns, st.st_size))
scan=time.perf_counter()-t
t=time.perf_counter(); v=gate.evaluate_all(); full=time.perf_counter()-t
t=time.perf_counter(); size=len(json.dumps(v)); ser=time.perf_counter()-t
print(json.dumps({"files_stamped":len(stamp),"stamp_scan_s":round(scan,4),
  "full_evaluate_all_s":round(full,3),"serialize_s":round(ser,4),
  "payload_bytes":size,"saving_per_warm_request_s":round(full-scan,3),
  "speedup":round(full/scan,1)}))
