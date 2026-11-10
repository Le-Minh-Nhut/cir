"""R2: native `preprocess_train` vs the adapter's `preprocess_val`, on identical pixels.

Three arms, kept separate because they answer different questions:
  A. same image, both transforms  -> is the ADAPTER reproducing the native EVAL transform?
  B. preprocess_val vs preprocess_val -> the same-input control (must be bitwise equal)
  C. native preprocess_train vs adapter preprocess_val -> the real native-vs-adapter gap
Only arm C can accuse the adapter; arms A/B can only prove or disprove transform identity.
"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch

ROOT = Path("/home/heheboiz/data/cir")
SRC = ROOT / "workbench/third_party/ENCODER"
sys.path.insert(0, str(SRC))

import open_clip
_, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained=str(SRC / "open_clip_pytorch_model.bin"))

img_path = ROOT / "data/FashionIQ/resized_image/dress" 
first = sorted(img_path.glob("*.jpg"))[0] if img_path.is_dir() else None
if first is None:
    raise SystemExit("no resized image found")
image = Image.open(first).convert("RGB")

def tensors_of(tf, repeats=3):
    return [tf(image.copy()) for _ in range(repeats)]

train_arm = tensors_of(preprocess_train)
val_arm = tensors_of(preprocess_val)
train2 = tensors_of(preprocess_train)

def cmp(a, b):
    d = (np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    return {"max_abs": float(np.abs(d).max()), "mean_abs": float(np.abs(d).mean()),
            "equal": bool(np.array_equal(np.asarray(a), np.asarray(b)))}

print("R2JSON:" + json.dumps({
    "image": str(first.relative_to(ROOT)),
    "image_size": list(image.size),
    "preprocess_train_repeatable": cmp(train_arm[0], train_arm[1]),
    "preprocess_val_repeatable": cmp(val_arm[0], val_arm[1]),
    "control_val_vs_val": cmp(val_arm[0], val_arm[2]),
    "train_vs_val_same_image": cmp(train_arm[0], val_arm[0]),
    "train_vs_train_second_draw": cmp(train_arm[0], train2[0]),
    "train_transform_repr": str(preprocess_train)[:300],
    "val_transform_repr": str(preprocess_val)[:300],
}, indent=1))
