"""R4: does the relation tensor differ between constructions, and can a seed pin it?

Compares, per construction: the drawn tensor, the checkpoint's 2 unexpected keys, and the
difference a second clean process produces. Seeding is applied BEFORE construction, which is the
only placement that can pin an nn.Parameter drawn in __init__.
"""
import json, os, sys
from pathlib import Path
import torch
ROOT = Path("/home/heheboiz/data/cir"); SRC = ROOT / "workbench/third_party/ENCODER"
sys.path.insert(0, str(SRC)); os.chdir(SRC)
import model_try2

def build(seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    m = model_try2.Encoder(hidden_dim=512, dropout=0.5, local_token_num=3, t=0.1, wc=2, N_p=2, weighted=True)
    return m.backbone.binding_decoder.learnable_relation.detach().clone()

ckpt = torch.load(ROOT/"workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads/encoder/fashioniq.pt", map_location="cpu")
m0 = model_try2.Encoder(hidden_dim=512, dropout=0.5, local_token_num=3, t=0.1, wc=2, N_p=2, weighted=True)
sd = m0.state_dict()
unexpected = sorted(set(ckpt) - set(sd))
missing = sorted(set(sd) - set(ckpt))
print("R4DET:" + json.dumps({
    "checkpoint_keys": len(ckpt), "state_dict_entries": len(sd),
    "unexpected_keys": unexpected, "missing_keys": missing,
    "unseeded_a_vs_b_max_abs": float((build() - build()).abs().max()),
    "seeded_same_seed_max_abs": float((build(0) - build(0)).abs().max()),
    "seeded_diff_seed_max_abs": float((build(0) - build(1)).abs().max()),
    "relation_requires_grad": bool(m0.backbone.binding_decoder.learnable_relation.requires_grad),
    "relation_device": str(m0.backbone.binding_decoder.learnable_relation.device),
    "registered_parameters": [n for n, _ in m0.named_parameters() if "relation" in n],
    "is_parameter": type(m0.backbone.binding_decoder.learnable_relation).__name__,
}))
