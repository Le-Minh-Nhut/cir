"""PTHA: identify the checkpoint key, class, and projection tensor shapes."""
import sys, json
from pathlib import Path
CIR = Path("/home/heheboiz/data/cir")
CKPT = Path(sys.argv[1])
sys.path.insert(0, str(CIR))
import torch
from workbench.replay.common import sha256_file

print("path:", CKPT, "bytes:", CKPT.stat().st_size)
print("sha256:", sha256_file(CKPT))
payload = torch.load(CKPT, map_location="cpu", weights_only=True)
print("top-level type:", type(payload).__name__, "keys:", list(payload) if isinstance(payload, dict) else None)
if isinstance(payload, dict):
    for k, v in payload.items():
        if isinstance(v, dict):
            print(f"  {k}: {len(v)} tensors")
            keys = [t for t in v if any(s in t for s in ("vision_proj", "text_proj", "query_tokens", "temp", "itm_head"))]
            for t in sorted(keys):
                print("     ", t, tuple(v[t].shape), v[t].dtype)
