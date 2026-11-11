"""Does a registry-declared adapted_runtime (in memory only; no file edited) clear the env gate?

Nothing is written. `WORKBENCH_RUNTIME_ADAPTATION` is exercised the way the registry would use it.
"""
import sys, json
sys.path.insert(0,'/home/heheboiz/data/cir')
from workbench.backend.registry import model_by_id
from workbench.backend.runtime import verify_environment
BLOCK = {"target_compute_capability": "sm_120", "python": "3.9", "pytorch": "==2.7.1",
         "torchvision": "==0.22.1", "cuda": "12.8",
         "packages": ["numpy<2", "Pillow", "tqdm"],
         "pip_index_url": "https://download.pytorch.org/whl/cu128"}
BLOCK_TG = dict(BLOCK, packages=["numpy==1.22.3", "pandas==1.4.2", "Pillow", "tqdm", "ftfy", "regex"])
import copy
for mid, block in (("clvc_net", BLOCK), ("tgcir", BLOCK_TG)):
    m = copy.deepcopy(model_by_id(mid))
    m["environment"]["adapted_runtime"] = block
    r = verify_environment(m, probe=True)
    print("###", mid, "with an in-memory adapted_runtime")
    print("  tier:", r.tier, "status:", r.status)
    print("  reason:", r.reason)
