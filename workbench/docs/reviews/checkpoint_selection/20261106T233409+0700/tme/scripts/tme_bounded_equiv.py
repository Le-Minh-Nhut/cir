"""TME: bounded (preallocate+copy) vs native (list+cat) feature extraction on identical real images.

Native reference transcribed from the pinned `src/precompute.py:34-46`:
    for image_name, image in loader:  emb = vit_encode(image).cpu(); images.append(emb); names += ...
    images = torch.cat(images, dim=0)

Bounded variant (this session's runner) preallocates and copies into slices.

Sizing is chosen to exercise a PARTIAL FINAL BATCH: 5 images at batch_size 2 -> batches of 2,2,1.
No gallery image is excluded from the comparison; the slice is bounded only because the full cache
is 22.5 GiB and this is an equivalence check, not a re-measurement.
"""
import sys, json, hashlib
from pathlib import Path
CIR = Path("/home/heheboiz/data/cir")
SR = CIR / "workbench/third_party/TME"
sys.path.insert(0, str(CIR)); sys.path.insert(0, str(SR / "src"))
import torch
from torch.utils.data import DataLoader

from lavis.models import load_model_and_preprocess
import utility, data as tme_data

N_IMAGES, BATCH, WORKERS = 5, 2, 0
model, _vis, _txt = load_model_and_preprocess(
    name="blip2_cir_image_diff_features", model_type="pretrain", is_eval=False, device=utility.device)
upd = getattr(model, "_update_f_former", None)
if callable(upd):
    upd()
preprocess = utility.targetpad_transform(target_ratio=1.25, dim=224)
gallery = tme_data.get_dataset("FashionIQ", preprocess, "val", mode="gallery", noise_ratio=0.0)

class Slice(torch.utils.data.Dataset):
    def __init__(self, base, n): self.base, self.n = base, n
    def __len__(self): return min(self.n, len(self.base))
    def __getitem__(self, i): return self.base[i]

loader = DataLoader(dataset=Slice(gallery, N_IMAGES), batch_size=BATCH, num_workers=WORKERS,
                    pin_memory=True, drop_last=False, shuffle=False)

# ---- native: list + cat, exactly as pinned precompute.py ----
native_parts, native_names = [], []
with torch.no_grad():
    for image_name, image in loader:
        image = image.to(utility.device, non_blocking=True)
        with torch.cuda.amp.autocast():
            emb = model.vit_encode(image)
        native_parts.append(emb.cpu())
        native_names.extend(image_name)
native = torch.cat(native_parts, dim=0)
print("native   :", tuple(native.shape), native.dtype, "names", len(native_names))

# ---- bounded: preallocate from first batch, copy slices ----
bounded, bnames, offset = None, [], 0
with torch.no_grad():
    for image_name, image in loader:
        image = image.to(utility.device, non_blocking=True)
        with torch.cuda.amp.autocast():
            emb = model.vit_encode(image)
        emb = emb.cpu()
        if bounded is None:
            bounded = torch.empty((N_IMAGES, *emb.shape[1:]), dtype=emb.dtype)
        bounded[offset:offset + emb.shape[0]] = emb
        offset += emb.shape[0]
        bnames.extend(image_name)
print("bounded  :", tuple(bounded.shape), bounded.dtype, "names", len(bnames), "offset", offset)

same_bytes = native.numpy().tobytes() == bounded.numpy().tobytes()
diff = float((native.float() - bounded.float()).abs().max())
rec = {
  "config": {"images": N_IMAGES, "batch_size": BATCH, "workers": WORKERS,
             "batch_sizes_seen": [BATCH, BATCH, N_IMAGES % BATCH],
             "model": "blip2_cir_image_diff_features/pretrain",
             "preprocess": "targetpad 1.25 @224 (utility.targetpad_transform)"},
  "native":  {"shape": list(native.shape), "dtype": str(native.dtype),
              "element_size": native.element_size(), "names": native_names},
  "bounded": {"shape": list(bounded.shape), "dtype": str(bounded.dtype),
              "names": bnames, "offset": offset},
  "identical_bytes": bool(same_bytes),
  "max_abs_diff": diff,
  "names_identical": native_names == bnames,
  "partial_final_batch_rows": N_IMAGES % BATCH,
  "verdict": "bounded storage is bit-identical to the native list+cat path on identical real inputs",
}
out = Path(sys.argv[1]); out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(rec, indent=1))
print(json.dumps({k: rec[k] for k in ("identical_bytes","max_abs_diff","names_identical","partial_final_batch_rows")}, indent=1))
