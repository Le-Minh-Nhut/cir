"""Pure native-vs-native control: encode one category's population in THIS process, unpinned relation.

Writes /tmp/relnat.<category>.<tag>.npz only. No repo file is touched.
"""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

REPO = "/home/heheboiz/data/cir"
SRC = f"{REPO}/workbench/third_party/ENCODER"
CKPT = (f"{REPO}/workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads/"
        "encoder/fashioniq.pt")
DATA = f"{REPO}/data/FashionIQ"
category, tag = sys.argv[1], sys.argv[2]

sys.path.insert(0, REPO)
sys.path.insert(0, SRC)
import open_clip  # noqa: E402
import model_try2  # noqa: E402

from workbench.replay.common import enable_deterministic_inference  # noqa: E402

_, _, preprocess_val = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained=f"{SRC}/open_clip_pytorch_model.bin")

cwd = os.getcwd()
os.chdir(SRC)
model = model_try2.Encoder(hidden_dim=512, dropout=0.5, local_token_num=3, t=0.1,
                           wc=2, N_p=2, weighted=True)
os.chdir(cwd)
model.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=False)
model = model.to("cuda").eval()
enable_deterministic_inference(torch)
relation = model.backbone.binding_decoder.learnable_relation.detach().cpu().numpy().copy()

inputs = json.load(open(f"{REPO}/workbench/docs/reviews/20261109-native15-composition-limn/"
                        f"raw/encoder_parity/inputs.{category}.json"))
# Texts/ids are pinned in the input file; recompose the native query order from the annotations.
captions = json.load(open(f"{DATA}/captions/cap.{category}.val.json"))
correction = json.load(open(f"{DATA}/captions/correction_dict_{category}.json"))
images = json.load(open(f"{DATA}/image_splits/split.{category}.val.json"))


def correct(text):
    import string

    translation = str.maketrans({key: " " for key in string.punctuation})
    tokens = str(text).lower().translate(translation).strip().split()
    return " ".join(correction.get(word, word) for word in tokens)


texts = [f"{correct(row['captions'][0])} and {correct(row['captions'][1])}" for row in captions]
reference_ids = [row["candidate"] for row in captions]
target_ids = [row["target"] for row in captions]
gallery_positions = []
for row in captions:
    for key in ("candidate", "target"):
        if images.index(row[key]) not in gallery_positions:
            gallery_positions.append(images.index(row[key]))
gallery_ids = [images[p] for p in gallery_positions]

BATCH = 64


def encode(names, kind, texts=None):
    rows = []
    base = f"{DATA}/resized_image/{category}"
    for start in range(0, len(names), BATCH):
        chunk_names = names[start:start + BATCH]
        decoded = []
        for name in chunk_names:
            with open(f"{base}/{name}.jpg", "rb") as handle:
                image = Image.open(handle).convert("RGB")
            decoded.append(preprocess_val(image))
        chunk = torch.stack(decoded).float().cuda()
        with torch.no_grad():
            out = (model.extract_retrieval_compose(chunk, texts[start:start + BATCH])
                   if kind == "queries" else model.extract_retrieval_target(chunk))
        rows.append(out.data.cpu().numpy())
    return np.concatenate(rows)


queries = encode(reference_ids, "queries", texts)
gallery = encode(gallery_ids, "targets")
np.savez(f"/tmp/relnat.{category}.{tag}.npz", queries=queries, gallery=gallery, relation=relation)
print("WROTE", category, tag, "relation_sum", float(relation.sum()),
      "q", queries.shape, "g", gallery.shape)
