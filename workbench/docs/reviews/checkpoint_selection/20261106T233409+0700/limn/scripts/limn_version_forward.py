"""LIMN: exercise the pinned native path under each open_clip version.

The pickled module has no `forward`; the pinned test.py drives it through named methods. We call the
same method the author's evaluator uses, on a real image, so the open_clip code path that matters
(`VisionTransformer` pooling) is actually reached.
"""
import sys, importlib.metadata as md
from pathlib import Path
CIR=Path("/home/heheboiz/data/cir"); SR=CIR/"workbench/third_party/LIMN"; CK=Path(sys.argv[1])
sys.path.insert(0,str(CIR)); sys.path.insert(0,str(SR)); sys.path.insert(0,str(SR/"LIMN"))
import torch
print("open_clip_torch:", md.version("open_clip_torch"))
obj = torch.load(CK, map_location="cpu", weights_only=False)
dev="cuda" if torch.cuda.is_available() else "cpu"
m=obj.to(dev).eval()
print("methods:", [n for n in ("encode_image","encode_text","forward","get_features") if hasattr(m,n)])
import open_clip
vt = m.clip.visual if hasattr(m,"clip") else None
print("has .clip:", vt is not None, "| pool_type attr:", hasattr(vt,"pool_type") if vt is not None else "n/a")
x = torch.randn(2,3,224,224, device=dev)
try:
    with torch.no_grad():
        if hasattr(m,"encode_image"):
            f = m.encode_image(x)
        elif vt is not None:
            f = vt(x)
        else:
            f = None
    print("ENCODE OK:", tuple(f.shape) if hasattr(f,"shape") else f)
except Exception as exc:
    print("ENCODE FAILED:", type(exc).__name__, ":", str(exc)[:300])
