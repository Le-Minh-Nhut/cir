"""Does the unseeded `learnable_relation` move a measured metric? CPU-only, two clean processes.

The vendor's `BindingDecoder.__init__` calls `.cuda()` (`model_try2.py:93`). Neutralising that one
device move keeps the *draw* identical and lets the whole forward run on CPU, so the only variable is
the relation tensor itself. Each process encodes the same N gallery images and the same M queries
through the model's own entrypoints (`extract_retrieval_target`, `extract_retrieval_compose`) and
reports: the relation digest, the feature digests, every target rank and recall@10.

Run it with `--seed S` for a pinned draw, without `--seed` for the unseeded draw HEAD actually
performs. Two processes of each are enough to see whether a "reproduction" reproduces.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path("/home/heheboiz/data/cir")
SRC = REPO / "workbench/third_party/ENCODER"
CKPT = (REPO / "workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads/"
        "encoder/fashioniq.pt")
FASHIONIQ = REPO / "data/FashionIQ"
GALLERY_LIMIT = 60
QUERY_LIMIT = 8

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

torch.Tensor.cuda = lambda self, *args, **kwargs: self          # type: ignore[method-assign]
torch.nn.Module.cuda = lambda self, *args, **kwargs: self       # type: ignore[method-assign]

import open_clip  # noqa: E402
import model_try2  # noqa: E402


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float32).tobytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dump", type=Path, default=None,
                        help="write gallery/query features here for a pairwise delta")
    args = parser.parse_args()

    _, _, preprocess_val = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained=str(SRC / "open_clip_pytorch_model.bin"))
    # The pin goes HERE, immediately before construction, because everything above it (the CLIP
    # tower) also consumes the generator: seeding earlier pins a different tensor. This is the same
    # placement the composition worker uses.
    if args.seed is not None:
        torch.manual_seed(args.seed)
    cwd = Path.cwd()
    os.chdir(SRC)
    model = model_try2.Encoder(hidden_dim=512, dropout=0.5, local_token_num=3, t=0.1,
                               wc=2, N_p=2, weighted=True)
    os.chdir(cwd)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=False)
    model.eval()
    relation = model.backbone.binding_decoder.learnable_relation.detach().float().cpu().contiguous()

    captions = json.loads((FASHIONIQ / "captions/cap.shirt.val.json").read_text(encoding="utf-8"))
    gallery_stems = sorted({Path(row["candidate"]).stem for row in captions})[:GALLERY_LIMIT]

    def image(stem: str) -> torch.Tensor:
        with Image.open(FASHIONIQ / "resized_image/shirt" / f"{stem}.jpg") as handle:
            return preprocess_val(handle.convert("RGB"))

    gallery = torch.stack([image(stem) for stem in gallery_stems])
    queries = []
    for row in captions:
        if len(queries) >= QUERY_LIMIT:
            break
        reference = Path(row["candidate"]).stem
        target = Path(row["target"]).stem if row.get("target") else None
        if target in gallery_stems and reference in gallery_stems:
            queries.append((reference, " and ".join(row["captions"]), target))
    if not queries:
        raise SystemExit("no fixture query has both reference and target in the gallery subset")

    with torch.no_grad():
        gallery_features = model.extract_retrieval_target(gallery).float().cpu().numpy()
        ranks, scores, composed_rows = [], [], []
        for reference, text, target in queries:
            ref = image(reference)
            # The vendor indexes the tokenizer output against batch size 1, so encode one query
            # at a time -- exactly what the composition worker's batch-of-1 path does.
            composed = model.extract_retrieval_compose(ref.unsqueeze(0),
                                                       [text]).float().cpu().numpy()[0]
            sims = gallery_features @ composed
            order = np.argsort(-sims, kind="stable")
            rank = int(np.flatnonzero(order == gallery_stems.index(target))[0]) + 1
            ranks.append(rank)
            scores.append(float(sims.max() - sims[gallery_stems.index(target)]))
            composed_rows.append(composed)

    if args.dump is not None:
        import numpy as _np
        _np.savez(args.dump, gallery=gallery_features, composed=_np.stack(composed_rows),
                  relation=relation.numpy())
    print("M01DELTA:" + json.dumps({
        "pid": os.getpid(), "seed": args.seed,
        "relation_sha256": digest(relation.numpy()),
        "relation_sum": float(relation.double().sum()),
        "gallery_sha256": digest(gallery_features),
        "ranks": ranks,
        "recall_at_10": 100.0 * sum(rank <= 10 for rank in ranks) / len(ranks),
        "margin_to_rank1": scores,
        "gallery_size": len(gallery_stems), "query_count": len(queries),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
