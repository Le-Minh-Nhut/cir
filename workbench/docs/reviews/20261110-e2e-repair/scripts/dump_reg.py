import json
from workbench.backend.registry import load_registry
r=load_registry()
out=[]
for m in r["models"]:
    out.append(m["model_id"]+" "+str(m.get("native_protocol"))+" "+str(sorted(m.keys()))[:200])
print("\n".join(out))
