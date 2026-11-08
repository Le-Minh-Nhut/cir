"""ENCODER reference-transform rank impact on REAL (reference, positive) pairs.

Second, sharper probe than `equivalence_encoder_transform.py`. No ENCODER gallery feature cache
survives on this host, so the full 74,381-gallery rank cannot be recomputed. Instead this builds a
mini-gallery from the union of real reference images and their real positive targets -- which is what
the target rank is computed over, just smaller -- and reports:

  * target rank of each query's real positive under reference-transform val vs train
  * hit@10 / hit@50 within the mini-gallery, both ways
  * the exact score margin at the positive, and the number of other rows inside that margin
  * whether torch seeding makes preprocess_train reproducible

The mini-gallery hit rates are NOT the run's R@10/R@50 and are labelled PROXY throughout. What
transfers is the per-query rank delta and its margin structure.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/heheboiz/data/cir")
sys.path.insert(0, str(REPO))
OUT = REPO / "workbench/docs/reviews/20261108T153045-audit-equivalence"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=64)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=str(OUT / "encoder_rank_impact.json"))
    args = parser.parse_args()

    from PIL import Image
    from workbench.backend.operator_config import resolve_config
    from workbench.composition import data as composition_data
    import open_clip

    config = resolve_config()
    source = Path(config.WORKBENCH_THIRD_PARTY_ROOT) / "ENCODER"
    root = Path(config.FASHIONIQ_ROOT) / "images"
    model, _, preprocess_val = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained=str(source / "open_clip_pytorch_model.bin"))
    _, preprocess_train, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained=str(source / "open_clip_pytorch_model.bin"))
    model = model.to(args.device).eval()
    release = composition_data.load_release(Path(config.WORKBENCH_CIRCUS_RELEASE_ROOT),
                                            verify_hashes=False)

    def resolve(candidate_id):
        name = release.candidate_names.get(candidate_id)
        if not name:
            return None
        stem = Path(name).stem
        for extension in (".png", ".jpg", ".jpeg"):
            candidate = root / f"{stem}{extension}"
            if candidate.is_file():
                return str(candidate)
        return None

    pairs = []
    for query in release.queries:
        reference = resolve(query.reference_id)
        positives = [position for position in (resolve(value) for value in query.positive_ids)
                     if position]
        if reference and positives:
            pairs.append((query.qid, reference, positives[0]))
        if len(pairs) >= args.queries:
            break
    if not pairs:
        raise SystemExit("no (reference, positive) pairs resolved")

    # mini-gallery: every reference and every positive, deduplicated, order preserved
    gallery_paths = list(dict.fromkeys([path for _q, ref, pos in pairs for path in (ref, pos)]))
    index = {path: position for position, path in enumerate(gallery_paths)}

    def encode(paths, transform):
        features = []
        for start in range(0, len(paths), args.batch):
            chunk = paths[start:start + args.batch]
            pixels = []
            for path in chunk:
                with Image.open(path) as image:
                    pixels.append(transform(image.convert("RGB")))
            with torch.no_grad():
                features.append(model.encode_image(torch.stack(pixels).to(args.device))
                                .float().cpu().numpy())
        return np.concatenate(features, axis=0)

    def normalize(matrix):
        return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)

    gallery_val = normalize(encode(gallery_paths, preprocess_val))
    refs_val = normalize(encode([ref for _q, ref, _p in pairs], preprocess_val))
    refs_train = normalize(encode([ref for _q, ref, _p in pairs], preprocess_train))

    def ranks(queries):
        scores = queries @ gallery_val.T
        out = []
        for row, (_qid, _ref, positive) in zip(scores, pairs):
            position = index[positive]
            score = row[position]
            # shipped rule: 1 + #greater + #equal-before
            rank = 1 + int(np.count_nonzero(row > score)) + \
                int(np.count_nonzero(row[:position] == score))
            out.append((rank, float(score), int(np.count_nonzero(row > score)),
                        int(np.count_nonzero(np.abs(row - score) < 0.01))))
        return out

    val_ranks = ranks(refs_val)
    train_ranks = ranks(refs_train)
    changed = [(qid, a[0], b[0], a[1] - b[1]) for (qid, _r, _p), a, b in
               zip(pairs, val_ranks, train_ranks) if a[0] != b[0]]

    # Stochasticity under an explicit seed, twice.
    torch.manual_seed(0)
    first = encode([pairs[0][1]], preprocess_train)
    torch.manual_seed(0)
    second = encode([pairs[0][1]], preprocess_train)
    torch.manual_seed(1)
    third = encode([pairs[0][1]], preprocess_train)

    payload = {
        "kind": "PROXY over a mini-gallery of real references and their real positive targets; "
                "not the run's 74,381-gallery metric",
        "queries": len(pairs),
        "mini_gallery": len(gallery_paths),
        "seeded_stochasticity": {
            "same_seed_max_abs_delta": float(np.abs(first - second).max()),
            "different_seed_max_abs_delta": float(np.abs(first - third).max()),
            "same_seed_identical": bool(np.array_equal(first, second)),
        },
        "rank_metrics": {
            "val": _summary(val_ranks),
            "train": _summary(train_ranks),
        },
        "rank_changed": len(changed),
        "max_abs_rank_delta": max((abs(b - a) for _q, a, b, _m in changed), default=0),
        "changed": [{"qid": q, "rank_val": a, "rank_train": b,
                     "score_val_minus_train": m} for q, a, b, m in changed],
        "unchanged_queries": len(pairs) - len(changed),
    }
    Path(args.out).write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(json.dumps(payload, indent=1, sort_keys=True))
    return 0


def _summary(rows):
    ranks = [row[0] for row in rows]
    return {
        "hit_at_10": sum(1 for rank in ranks if rank <= 10),
        "hit_at_50": sum(1 for rank in ranks if rank <= 50),
        "hit_at_10_fraction": sum(1 for rank in ranks if rank <= 10) / len(ranks),
        "hit_at_50_fraction": sum(1 for rank in ranks if rank <= 50) / len(ranks),
        "median_rank": float(np.median(ranks)),
        "mean_ndcg": float(np.mean([1.0 / np.log2(rank + 1) for rank in ranks])),
        "rank_histogram_head": {str(k): ranks.count(k) for k in sorted(set(ranks))[:12]},
    }


if __name__ == "__main__":
    raise SystemExit(main())
