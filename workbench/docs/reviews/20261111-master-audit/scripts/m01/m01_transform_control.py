"""ENCODER transform policy as a DECLARED CONTROL, not a patch: val-for-both vs train-query/val-gallery.

Native ENCODER uses `transform[0]` (preprocess_train) for the query and `transform[1]`
(preprocess_val) for the gallery (`third_party/ENCODER/datasets.py:177` vs `:195,202`). CIRCUS
composition uses `preprocess_val` for both (`workbench/composition/model_encoder.py`). Switching the
production path would silently move every published number, so this measures the sensitivity instead:
identical pixels, identical weights, identical pinned relation, ONLY the query transform differs.

`preprocess_train` is `RandomResizedCrop`, so it is seeded; a repeat run of the same seed is reported
so the delta is not confused with its own sampling noise.

CPU-only: the vendor's two `.cuda()` calls are neutralised, which leaves every arithmetic op and the
draw order untouched.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path("/home/heheboiz/data/cir")
SRC = REPO / "workbench/third_party/ENCODER"
CKPT = (REPO / "workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads/"
        "encoder/fashioniq.pt")
FASHIONIQ = REPO / "data/FashionIQ"
CATEGORY = "shirt"
GALLERY_LIMIT = 1000
QUERY_LIMIT = 60
RELATION_SEED = 20261111

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

torch.Tensor.cuda = lambda self, *a, **k: self            # type: ignore[method-assign]
torch.nn.Module.cuda = lambda self, *a, **k: self         # type: ignore[method-assign]

import open_clip  # noqa: E402
import model_try2  # noqa: E402


def ranks_for(model, preprocess_query, preprocess_gallery, gallery, queries, stems):
    """`gallery` and `queries` carry PIL images: only the transform differs between the two arms."""
    with torch.no_grad():
        gallery_features = model.extract_retrieval_target(
            torch.stack([preprocess_gallery(p) for p in gallery])).float().cpu().numpy()
        ranks, hit = [], 0
        for reference, text, target in queries:
            composed = model.extract_retrieval_compose(
                preprocess_query(reference).unsqueeze(0), [text]).float().cpu().numpy()[0]
            order = np.argsort(-(gallery_features @ composed), kind="stable")
            rank = int(np.flatnonzero(order == stems.index(target))[0]) + 1
            ranks.append(rank)
            hit += rank <= 10
    return ranks, 100.0 * hit / len(ranks)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transform-seed", type=int, default=7)
    args = parser.parse_args()

    # create_model_and_transforms returns (model, preprocess_train, preprocess_val).
    _, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained=str(SRC / "open_clip_pytorch_model.bin"))

    torch.manual_seed(RELATION_SEED)
    cwd = Path.cwd()
    os.chdir(SRC)
    model = model_try2.Encoder(hidden_dim=512, dropout=0.5, local_token_num=3, t=0.1,
                               wc=2, N_p=2, weighted=True)
    os.chdir(cwd)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=False)
    model.eval()

    def image(stem: str):
        with Image.open(FASHIONIQ / f"resized_image/{CATEGORY}" / f"{stem}.jpg") as handle:
            return handle.convert("RGB")

    rows = json.loads((FASHIONIQ / f"captions/cap.{CATEGORY}.val.json").read_text(encoding="utf-8"))
    stems = sorted({Path(row["candidate"]).stem for row in rows})[:GALLERY_LIMIT]
    queries = []
    for row in rows:
        target = Path(row["target"]).stem if row.get("target") else None
        reference = Path(row["candidate"]).stem
        if target in stems and reference in stems and len(queries) < QUERY_LIMIT:
            queries.append((reference, " and ".join(row["captions"]), target))
    gallery = [image(stem) for stem in stems]
    queries = [(image(reference), text, target) for reference, text, target in queries]

    # Two identical val-for-both runs first: the control for "there is no noise in this path".
    val_a, val_a_r10 = ranks_for(model, preprocess_val, preprocess_val, gallery, queries, stems)
    val_b, val_b_r10 = ranks_for(model, preprocess_val, preprocess_val, gallery, queries, stems)

    torch.manual_seed(args.transform_seed)
    train_q, train_q_r10 = ranks_for(model, preprocess_train, preprocess_val, gallery,
                                    queries, stems)
    torch.manual_seed(args.transform_seed)
    train_q_again, train_q_again_r10 = ranks_for(model, preprocess_train, preprocess_val,
                                                gallery, queries, stems)

    # Proof the two arms actually receive different pixels on the same image; a 0.0 delta with an
    # identical input would be a broken control, not a finding.
    probe_stem = stems[0]
    with Image.open(FASHIONIQ / f"resized_image/{CATEGORY}" / f"{probe_stem}.jpg") as handle:
        raw = handle.convert("RGB")
    torch.manual_seed(args.transform_seed)
    a = preprocess_val(raw)
    torch.manual_seed(args.transform_seed)
    b = preprocess_train(raw)

    print("M01TRANSFORM:" + json.dumps({
        "pixel_max_abs_val_vs_train": float((a - b).abs().max()),
        "pixel_identical_val_vs_train": bool(torch.equal(a, b)),
        "category": CATEGORY, "gallery_size": len(stems), "query_count": len(queries),
        "relation_seed": RELATION_SEED, "transform_seed": args.transform_seed,
        "val_both_ranks": val_a, "val_both_recall_at_10": val_a_r10,
        "val_both_repeat_identical": val_a == val_b,
        "train_query_val_gallery_ranks": train_q,
        "train_query_val_gallery_recall_at_10": train_q_r10,
        "train_repeat_same_seed_identical": train_q == train_q_again,
        "recall_at_10_delta_train_minus_val": train_q_r10 - val_a_r10,
        "ranks_changed": sum(1 for a, b in zip(val_a, train_q) if a != b),
        "max_abs_rank_delta": max((abs(a - b) for a, b in zip(val_a, train_q)), default=0),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
