"""Category-vocabulary probe for the CLVC-Net per-category whole-model pickles.

`train.py:load_dataset` builds a `SimpleVocab` per FashionIQ category from that category's
captions, so each saved model carries its own vocabulary. If the six staged files are really the
three per-category pairs, each category's vocabulary must be disjoint-ish and contain words drawn
from that category's annotations.
"""
import json
import sys

import torch

sys.path.insert(0, 'workbench/third_party/CLVCNet')
BASE = "workbench/artifacts/checkpoint_selection/20261106T002617+0700/checkpoints/clvc"
FILES = [f"{cat}_{member}_model.pt" for cat in ("dress", "shirt", "toptee")
         for member in ("local", "global")]

out = {}
vocabularies = {}
for name in FILES:
    model = torch.load(f"{BASE}/{name}", map_location="cpu", weights_only=False)
    text_model = model.text_model
    vocab = getattr(text_model, "vocab", None)
    word2id = getattr(vocab, "word2id", None) or {}
    words = list(word2id)
    vocabularies.setdefault(name.split("_")[0], []).append(len(words))
    entry = {
        "class": type(model).__name__,
        "vocab_size": len(words),
        "vocab_sha1_order": __import__("hashlib").sha1("|".join(words).encode()).hexdigest(),
        "has_probe_words": {probe: probe in word2id for probe in
                            ("dress", "shirt", "toptee", "sleeve", "neckline", "collar",
                             "skirt", "button", "floral", "plaid", "pattern", "length")},
        "text_model_state": {key: list(value.shape)
                             for key, value in model.state_dict().items()
                             if key.startswith("text_model")},
        "categories_attr": getattr(model, "categories", None),
        "number_of_models": getattr(model, "number_of_models", None),
    }
    out[name] = entry
    print(name, entry["class"], "vocab", entry["vocab_size"], entry["vocab_sha1_order"][:12],
          "probes", {k: v for k, v in entry["has_probe_words"].items() if v})

# Are the local and global members of one category built from the SAME vocabulary?
for cat in ("dress", "shirt", "toptee"):
    local = out[f"{cat}_local_model.pt"]["vocab_sha1_order"]
    glob = out[f"{cat}_global_model.pt"]["vocab_sha1_order"]
    out[f"_same_vocab_{cat}"] = local == glob

keys = [out[f"{cat}_{m}_model.pt"]["vocab_sha1_order"] for cat in ("dress", "shirt", "toptee")
        for m in ("local", "global")]
out["_all_six_share_one_vocab"] = len(set(keys)) == 1
print(json.dumps({k: v for k, v in out.items() if k.startswith("_")}, indent=1))
json.dump(out, open(sys.argv[1], "w"), indent=1)
