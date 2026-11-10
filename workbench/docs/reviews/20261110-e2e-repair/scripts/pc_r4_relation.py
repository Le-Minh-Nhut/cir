"""R4: is `learnable_relation` ever registered, and does a fresh construction reproduce?

Two independent facts, measured:
  1. the checkpoint's key set vs the instance's state_dict key set;
  2. two clean processes (seeded identically, then unseeded) building the same model.

Run once per process; `--repeat` runs the construction twice inside one process to separate
"per-construction RNG" from "cross-process RNG".
"""
import argparse, hashlib, json, sys
from pathlib import Path

ROOT = Path("/home/heheboiz/data/cir")
SRC = ROOT / "workbench/third_party/ENCODER"
sys.path.insert(0, str(SRC))

def tensor_digest(t) -> str:
    import numpy as np
    a = t.detach().cpu().numpy()
    return hashlib.sha256(a.tobytes()).hexdigest()

def main() -> int:
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()

    import os
    os.chdir(SRC)
    import model_try2

    ckpt_path = (ROOT / "workbench/artifacts/checkpoint_selection/20261106T002617+0700/"
                        "cohort/downloads/encoder/fashioniq.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_keys = list(ckpt.keys())

    report = {"checkpoint": {"path": str(ckpt_path), "keys": len(ckpt_keys),
                             "relation_keys": [k for k in ckpt_keys if "relation" in k]},
              "cuda_available": torch.cuda.is_available(), "constructions": []}

    if args.seed is not None:
        torch.manual_seed(args.seed)

    for attempt in range(args.repeat):
        model = model_try2.Encoder(hidden_dim=512, dropout=0.5, local_token_num=3, t=0.1,
                                   wc=2, N_p=2, weighted=True)
        sd = model.state_dict()
        relation_attr = getattr(model.backbone.binding_decoder, "learnable_relation", None)
        entry = {
            "attempt": attempt,
            "state_dict_entries": len(sd),
            "state_dict_has_relation": [k for k in sd if "relation" in k],
            "attribute_type": type(relation_attr).__name__,
            "in_state_dict": any(k.endswith("learnable_relation") for k in sd),
            "attribute_digest": tensor_digest(relation_attr) if relation_attr is not None else None,
        }
        # the value the forward pass actually uses
        used = model.backbone.binding_decoder.learnable_relation
        entry["used_in_forward_digest"] = tensor_digest(used)
        entry["first_values"] = [round(float(v), 6) for v in used.flatten()[:4].tolist()]
        # what a strict=False load reports for the missing parameter
        load_report = model.load_state_dict(ckpt, strict=False)
        entry["load_missing_relation_keys"] = [k for k in load_report.missing_keys if "relation" in k]
        entry["load_missing_count"] = len(load_report.missing_keys)
        entry["load_unexpected_count"] = len(load_report.unexpected_keys)
        entry["after_load_digest"] = tensor_digest(model.backbone.binding_decoder.learnable_relation)
        report["constructions"].append(entry)

    print("R4JSON:" + json.dumps(report))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
