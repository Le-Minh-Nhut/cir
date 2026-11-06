"""TME: measure the dtype/shape/storage that the native `vit_encode` path actually returns.

Questions:
 1. what dtype does `blip_model.vit_encode()` return under the pinned autocast?
 2. is the difference native, runtime-specific, or introduced by the bounded precompute runner?
 3. what storage does the resulting cache require?
"""
import sys, json
from pathlib import Path
CIR=Path("/home/heheboiz/data/cir"); SR=CIR/"workbench/third_party/TME"
sys.path.insert(0,str(CIR)); sys.path.insert(0,str(SR/"src"))
import torch
rec={}
print("torch:", torch.__version__, "| cuda:", torch.version.cuda)
print("autocast default enabled:", torch.is_autocast_enabled())

from lavis.models import load_model_and_preprocess
model, vis, txt = load_model_and_preprocess(name="blip2_feature_extractor",
    model_type="pretrain_vitL", is_eval=True, device="cuda")
ve = model.visual_encoder
print("visual_encoder param dtypes:", sorted({str(p.dtype) for p in ve.parameters()}))
print("model precision attr:", getattr(ve, "precision", "<none>"))
x = torch.randn(2,3,224,224, device="cuda")
with torch.no_grad():
    out = model.vit_encode(x, mode="image")
if isinstance(out, tuple):
    feats = out[0]
else:
    feats = out
print("vit_encode output:", tuple(feats.shape), feats.dtype, "| element_size:", feats.element_size())
rec["vit_encode"]={"shape":list(feats.shape),"dtype":str(feats.dtype),"element_size":feats.element_size()}
b = feats.element_size()*feats[0].numel()
print("bytes per image:", b, "| MiB:", round(b/2**20,4))
rec["bytes_per_image"]=b
for n in (2017,2628,3817,6346,5373,14392):
    print(f"  {n} images -> {b*n/2**30:.2f} GiB")
print(json.dumps(rec, indent=1))
