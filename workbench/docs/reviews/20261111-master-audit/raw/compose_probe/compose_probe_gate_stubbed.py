"""Prove where the run stops once the ENVIRONMENT GATE is satisfied.

`require_verified_model_interpreter` is replaced in-memory with the model's own resolved interpreter
(the exact value the gate returns on success, `runtime.py:2308`). No registry file, no source file and
no config file is modified. No forward pass, no CUDA kernel, nothing loaded.
"""
import os, sys, tempfile, traceback
from pathlib import Path
sys.path.insert(0, '/home/heheboiz/data/cir')
import workbench.backend.runtime as rt
rt.require_verified_model_interpreter = lambda model, **kw: os.environ[
    (model.get("environment") or {})["python_env_var"]]
from workbench.composition import model_clvc_net, model_tgcir

IMG = "/home/heheboiz/data/cir/data/FashionIQ/resized_image/dress/B005X4PL1G.jpg"
TXT = "is shiny and silver with shorter sleeves"
for mod, cat in ((model_clvc_net, "dress"), (model_tgcir, None)):
    print("###", mod.__name__)
    a = mod.Adapter(); a.work_root = Path(tempfile.mkdtemp())
    print("  interpreter:", a._interpreter)
    print("  checkpoint_path:", a.checkpoint_path)
    try:
        v = a.encode_queries([IMG], [TXT], [cat]); print("  RETURNED", v.shape, v.dtype)
    except Exception as e:
        print("  RAISED", type(e).__name__)
        for line in traceback.format_exc().strip().splitlines()[-6:]: print("   ", line)
