import sys, json
sys.path.insert(0,'/home/heheboiz/data/cir')
from workbench.backend.registry import model_by_id
from workbench.backend.runtime import verify_environment
for mid in ("clvc_net","tgcir"):
    m = model_by_id(mid)
    r = verify_environment(m, probe=True)
    print("###", mid)
    print("  tier:", r.tier, "status:", r.status)
    print("  reason:", r.reason)
    print("  interpreter:", r.interpreter, "python:", r.python_version)
    print("  packages:", json.dumps(r.packages, sort_keys=True))
    print("  deps_probe_executed:", getattr(r,"dependency_probe_executed",None))
    print("  imports:", json.dumps(getattr(r,"imports",{}), sort_keys=True, default=str)[:600])
