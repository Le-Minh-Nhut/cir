"""ENCODER reference-transform probe: `preprocess_train` vs `preprocess_val`.

Upstream `third_party/ENCODER/datasets.py::FashionIQ.get_test_data` encodes the REFERENCE
(`source_img_data`) with `get_img(..., stage=0)` and the GALLERY (`target_img_data`) with
`stage=1`, and `evaluate_model.py` builds `transform=[preprocess_train, preprocess_val]`.
So upstream reference = preprocess_train, gallery = preprocess_val.

The workbench adapter uses `preprocess_val` for both. This script measures the consequence on real
reference images:

  * is `preprocess_train` stochastic (same image twice -> different tensor)?
  * max abs delta between the two transforms' pixel tensors
  * the same delta after the model's own image encoder (ViT-B-32 vision tower): feature-level
  * a rank-movement PROXY: because no ENCODER gallery feature cache survives on this host, the
    effect is measured on a self-consistent mini-gallery -- the first N real gallery images encoded
    with preprocess_val -- with queries being the same images encoded under each transform. The
    real quantity we need is "how often does the reference's cosine ordering change", and that is
    what this proxy answers; it is labelled PROXY in the report, never presented as the run's own
    gallery.

Runs under the ENCODER interpreter. Writes one JSON into the audit evidence dir.
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


def encode(model, batches, device):
    outputs = []
    with torch.no_grad():
        for pixels in batches:
            feature = model.encode_image(pixels.to(device))
            outputs.append(feature.float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--references", type=int, default=64)
    parser.add_argument("--gallery", type=int, default=512)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--out", default=str(OUT / "encoder_transform.json"))
    parser.add_argument("--device", default="cpu")
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
    # CPU on purpose: the cohort owns the GPU, and this probe is a pure measurement of the
    # transform, not a run. ViT-B-32 on CPU is slow but the comparison is exact either way.
    device = args.device
    model = model.to(device).eval()

    release = composition_data.load_release(Path(config.WORKBENCH_CIRCUS_RELEASE_ROOT),
                                            verify_hashes=False)
    def resolve(identity):
        stem = Path(identity).stem
        for extension in (".png", ".jpg", ".jpeg"):
            candidate = root / f"{stem}{extension}"
            if candidate.is_file():
                return str(candidate)
        return None

    # Real reference images: the released queries' own reference ids, in query order. A query's
    # `reference_id` is a CANDIDATE id; `candidate_names` maps it to the released image name.
    reference_ids = []
    for query in release.queries:
        name = release.candidate_names.get(query.reference_id) or \
            release.candidate_names.get(query.reference_id.split(":")[-1])
        path = resolve(name) if name else None
        if path and path not in reference_ids:
            reference_ids.append(path)
        if len(reference_ids) >= args.references:
            break
    if not reference_ids:
        raise SystemExit("no reference images resolved; candidate mapping is wrong")
    gallery_paths = []
    for identity in release.gallery.image_ids:
        path = resolve(identity)
        if path and path not in gallery_paths:
            gallery_paths.append(path)
        if len(gallery_paths) >= args.gallery:
            break

    # 1) stochasticity of preprocess_train on ONE image, three applications
    sample = Image.open(reference_ids[0]).convert("RGB")
    repeats = [preprocess_train(sample.copy()).numpy() for _ in range(3)]
    stochastic = {
        "transform": "open_clip preprocess_train (ViT-B-32)",
        "same_image_repeats": 3,
        "pairwise_max_abs_delta": [
            float(np.abs(repeats[i].astype(np.float64) - repeats[j].astype(np.float64)).max())
            for i, j in ((0, 1), (0, 2), (1, 2))],
        "pairwise_identical": bool(np.array_equal(repeats[0], repeats[1]) and
                                   np.array_equal(repeats[0], repeats[2])),
    }
    sample_val = [preprocess_val(Image.open(reference_ids[0]).convert("RGB").copy()).numpy()
                  for _ in range(2)]
    stochastic["preprocess_val_pairwise_identical"] = bool(np.array_equal(sample_val[0], sample_val[1]))

    # 2) pixel-level delta over real reference images
    def pixels(paths, transform):
        out = []
        for path in paths:
            with Image.open(path) as image:
                out.append(transform(image.convert("RGB")).numpy())
        return np.stack(out)

    val_pixels = pixels(reference_ids, preprocess_val)
    train_pixels = pixels(reference_ids, preprocess_train)
    pixel_delta = np.abs(train_pixels.astype(np.float64) - val_pixels.astype(np.float64))
    pixel_report = {
        "images": len(reference_ids),
        "max_abs_delta": float(pixel_delta.max()),
        "mean_abs_delta": float(pixel_delta.mean()),
        "fraction_elements_differing": float(np.count_nonzero(pixel_delta) / pixel_delta.size),
        "images_with_any_difference": int(np.count_nonzero(pixel_delta.reshape(len(reference_ids), -1).any(axis=1))),
    }

    def batches_of(paths, transform):
        chunks = []
        for start in range(0, len(paths), args.batch):
            chunk = paths[start:start + args.batch]
            images = []
            for path in chunk:
                with Image.open(path) as image:
                    images.append(transform(image.convert("RGB")))
            chunks.append(torch.stack(images))
        return chunks

    val_features = encode(model, batches_of(reference_ids, preprocess_val), device)
    train_features = encode(model, batches_of(reference_ids, preprocess_train), device)
    feature_delta = np.abs(train_features.astype(np.float64) - val_features.astype(np.float64))

    def normalize(matrix):
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)

    cos = (normalize(train_features) * normalize(val_features)).sum(axis=1)
    feature_report = {
        "images": len(reference_ids),
        "max_abs_delta": float(feature_delta.max()),
        "mean_abs_delta": float(feature_delta.mean()),
        "cosine_train_vs_val": {"min": float(cos.min()), "mean": float(cos.mean()),
                                "max": float(cos.max())},
        "relative_delta_max": float((feature_delta / np.maximum(np.abs(val_features), 1e-12)).max()),
    }

    # 3) rank-movement PROXY on a self-consistent mini-gallery of real gallery images
    gallery_features = encode(model, batches_of(gallery_paths, preprocess_val), device)
    gallery_norm = normalize(gallery_features)
    proxy = {}
    for label, queries in (("reference_val", val_features), ("reference_train", train_features)):
        scores = normalize(queries) @ gallery_norm.T
        proxy[label] = np.argsort(-scores, axis=1, kind="stable")
    shared = np.asarray(gallery_features.shape[0])
    # A reference is its own best match in both paths; the interesting number is whether the
    # ORDERING of the mini-gallery changes, which is the ordering the target rank is computed over.
    rank_changes, max_shift, order_differs = 0, 0, 0
    for index in range(len(reference_ids)):
        a, b = proxy["reference_val"][index], proxy["reference_train"][index]
        if not np.array_equal(a, b):
            order_differs += 1
            positions = {int(position): order for order, position in enumerate(a)}
            shift = max(abs(positions[int(position)] - order) for order, position in enumerate(b))
            max_shift = max(max_shift, shift)
        if a[0] != b[0]:
            rank_changes += 1
    proxy_report = {
        "kind": "PROXY (self-consistent mini-gallery of real gallery images, gallery encoded with "
                "preprocess_val because no ENCODER gallery feature cache survives on this host)",
        "gallery_images": int(gallery_features.shape[0]),
        "queries": len(reference_ids),
        "queries_whose_top1_changes": rank_changes,
        "queries_whose_full_order_changes": order_differs,
        "max_top1_shift_positions": max_shift,
        "fraction_top1_changed": rank_changes / len(reference_ids),
    }
    payload = {
        "device": device, "interpreter": sys.executable, "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "open_clip": getattr(open_clip, "__version__", "unknown"),
        "stochasticity": stochastic, "pixels": pixel_report, "features": feature_report,
        "rank_proxy": proxy_report,
        "upstream_wiring": {
            "file": "workbench/third_party/ENCODER/datasets.py",
            "FashionIQ.get_test_data": "source_img_data = get_img(..., stage=0); "
                                       "target_img_data = get_img(..., stage=1)",
            "transform_list": "evaluate_model.py: transform=[preprocess_train, preprocess_val]",
            "conclusion": "upstream reference = preprocess_train, gallery = preprocess_val; "
                          "the workbench adapter applies preprocess_val to both",
        },
    }
    Path(args.out).write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(json.dumps(payload, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
