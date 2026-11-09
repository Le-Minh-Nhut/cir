"""Measure clip4cir fullft bundle key coverage and PAIR state-dict structure.

Both run on CPU with a 2-thread budget. CLIP fullft comparison uses max-abs-diff per key so a
value-level difference is not confused with a shape/identity difference.
"""
import sys, torch, warnings
warnings.filterwarnings("ignore")

def clip_fullft(base):
    sys.path.insert(0, "workbench/third_party/CLIP4Cir/src")
    import clip
    from combiner import Combiner
    cs = torch.load(base + "/fiq_comb_RN50x4_fullft.pt", map_location="cpu")
    comb = Combiner(640, 2560, 5120)
    r = comb.load_state_dict(cs["Combiner"], strict=False)
    print("COMB epoch", cs.get("epoch"), "keys", len(cs["Combiner"]),
          "missing", len(r.missing_keys), "unexpected", len(r.unexpected_keys))
    clip_model, _ = clip.load("RN50x4", device="cpu", jit=False)
    stock = clip_model.state_dict()
    ck = torch.load(base + "/fiq_clip_RN50x4_fullft.pt", map_location="cpu")
    csd = ck["CLIP"]
    sk = set(stock); ckk = set(csd)
    diff = [k for k in sk & ckk if (stock[k].float() - csd[k].float()).abs().max().item() > 0]
    shape = [k for k in sk & ckk if tuple(stock[k].shape) != tuple(csd[k].shape)]
    print("CLIP epoch", ck.get("epoch"), "keys", len(csd),
          "missing_in_model", len(sk - ckk), "unexpected_in_ckpt", len(ckk - sk),
          "shape_mismatch", len(shape), "value_diff>0", len(diff), "/", len(sk & ckk))

if __name__ == "__main__":
    clip_fullft("workbench/artifacts/checkpoint_selection/20261106T002617+0700/"
                "checkpoints/clip4cir/FashionIQ/RN50x4_fullft")
