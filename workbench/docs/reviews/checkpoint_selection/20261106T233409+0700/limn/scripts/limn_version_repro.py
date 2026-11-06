"""LIMN: reproduce load success/failure by open_clip version on the real checkpoint."""
import sys, json, pickle
from pathlib import Path
CIR=Path("/home/heheboiz/data/cir"); SR=CIR/"workbench/third_party/LIMN"
CK=Path(sys.argv[1])
sys.path.insert(0,str(CIR)); sys.path.insert(0,str(SR)); sys.path.insert(0,str(SR/"LIMN"))
import torch, open_clip
import importlib.metadata as md
print("open_clip_torch:", md.version("open_clip_torch"))
print("checkpoint:", CK.name, CK.stat().st_size)
try:
    obj = torch.load(CK, map_location="cpu", weights_only=False)
    print("LOAD OK ->", type(obj).__module__+"."+type(obj).__name__)
    vt = None
    for n,m in obj.named_modules():
        if type(m).__name__=="VisionTransformer":
            vt=m; break
    if vt is not None:
        print("VisionTransformer has global_average_pool:", hasattr(vt,"global_average_pool"))
        print("VisionTransformer has pool_type:", hasattr(vt,"pool_type"))
except Exception as exc:
    print("LOAD FAILED:", type(exc).__name__, ":", exc)
