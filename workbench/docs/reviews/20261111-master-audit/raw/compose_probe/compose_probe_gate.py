import sys, traceback, tempfile
from pathlib import Path
sys.path.insert(0, '/home/heheboiz/data/cir')
from workbench.composition import model_clvc_net, model_tgcir

IMG = "/home/heheboiz/data/cir/data/FashionIQ/resized_image/dress/B005X4PL1G.jpg"
TXT = "is shiny and silver with shorter sleeves"

for mod, cat in ((model_clvc_net, "dress"), (model_tgcir, None)):
    print("###", mod.__name__)
    try:
        a = mod.Adapter()
    except Exception:
        print("CONSTRUCT FAILED"); traceback.print_exc(); continue
    print("  interpreter attr:", a._interpreter)
    print("  checkpoint_path:", a.checkpoint_path)
    a.work_root = Path(tempfile.mkdtemp(prefix="compose_probe_"))
    try:
        v = a.encode_queries([IMG], [TXT], [cat])
        print("  encode_queries RETURNED", v.shape, v.dtype)
    except Exception as e:
        print("  encode_queries RAISED", type(e).__name__)
        print("  msg:", e)
        traceback.print_exc()
    print("  mode_supported:", a.mode_supported("multimodal"))
