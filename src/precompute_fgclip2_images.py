from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from backbones.fgclip2 import (
    FGCLIP2Backbone,
    FGCLIP2_LARGE_DIM,
    FGCLIP2_LARGE_MODEL_ID,
)
from datasets.common import DirectoryImageStore
from datasets.fashioniq import load_fashioniq_split_ids


CATEGORIES = ("dress", "shirt", "toptee")
VALID_SPLITS = ("train", "val")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute frozen FG-CLIP2-Large global FashionIQ image features."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/fashionIQ_dataset"),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("features/fashioniq/fgclip2-large"),
    )
    parser.add_argument(
        "--model-id",
        default=FGCLIP2_LARGE_MODEL_ID,
    )
    parser.add_argument("--splits", nargs="+", choices=VALID_SPLITS, default=list(VALID_SPLITS))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-num-patches", type=int, default=784)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--parity-samples", type=int, default=3)
    return parser.parse_args()


def load_all_image_ids(split_root: Path, split: str) -> list[str]:
    image_ids: list[str] = []
    seen: set[str] = set()
    for category in CATEGORIES:
        for image_id in load_fashioniq_split_ids(split_root, split, category):
            if image_id not in seen:
                seen.add(image_id)
                image_ids.append(image_id)
    return image_ids


@torch.inference_mode()
def precompute_split(
    *,
    split: str,
    image_ids: list[str],
    image_store: DirectoryImageStore,
    backbone: FGCLIP2Backbone,
    output_dir: Path,
    batch_size: int,
    parity_samples: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "images.npy"
    features = np.lib.format.open_memmap(
        feature_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(image_ids), 1, FGCLIP2_LARGE_DIM),
    )

    parity_ids: list[str] = []
    parity_direct: torch.Tensor | None = None
    for start in tqdm(
        range(0, len(image_ids), batch_size),
        desc=f"FG-CLIP2 images [{split}]",
        dynamic_ncols=True,
    ):
        batch_ids = image_ids[start : start + batch_size]
        images = [image_store.load(image_id) for image_id in batch_ids]
        encoded = backbone.encode_image_global(images)
        if encoded.requires_grad:
            raise RuntimeError("Image precompute unexpectedly recorded gradients")
        end = start + len(batch_ids)
        features[start:end, 0, :] = encoded.cpu().numpy()

        if parity_direct is None and parity_samples > 0:
            keep = min(parity_samples, len(batch_ids))
            parity_ids = batch_ids[:keep]
            parity_direct = encoded[:keep].cpu().clone()

    features.flush()
    name_to_idx = {image_id: index for index, image_id in enumerate(image_ids)}
    with (output_dir / "name_to_idx.json").open("w", encoding="utf-8") as file:
        json.dump(name_to_idx, file, indent=2)

    reloaded = np.load(feature_path, mmap_mode="r")
    if reloaded.shape != (len(image_ids), 1, FGCLIP2_LARGE_DIM):
        raise RuntimeError(f"Reloaded image cache has unexpected shape {reloaded.shape}")
    if not np.isfinite(reloaded).all():
        raise FloatingPointError("Reloaded image cache contains NaN or Inf")

    parity_max_abs_error = 0.0
    if parity_direct is not None:
        parity_indices = [name_to_idx[image_id] for image_id in parity_ids]
        parity_cached = torch.from_numpy(reloaded[parity_indices, 0, :].copy())
        parity_max_abs_error = float((parity_direct - parity_cached).abs().max().item())
        if not torch.allclose(parity_direct, parity_cached, rtol=0.0, atol=1e-6):
            raise RuntimeError(
                "FG-CLIP2 image direct/cache parity failed: "
                f"max_abs_error={parity_max_abs_error}"
            )

    manifest = {
        "dataset": "FashionIQ",
        "split": split,
        "feature_kind": "fgclip2_global_image",
        "model_id": backbone.model_id,
        "normalized": True,
        "max_num_patches": backbone.max_num_patches,
        "num_samples": len(image_ids),
        "images_shape": list(reloaded.shape),
        "images_dtype": str(reloaded.dtype),
        "requires_grad": False,
        "parity_samples": len(parity_ids),
        "parity_max_abs_error": parity_max_abs_error,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    print(f"[{split}] images: {tuple(reloaded.shape)} -> {output_dir}")
    print(f"[{split}] cache parity max abs error: {parity_max_abs_error:.3e}")


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.parity_samples < 0:
        raise ValueError("--parity-samples must be >= 0")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    split_root = args.dataset_root / "image_splits"
    image_root = args.dataset_root / "images"
    if not split_root.is_dir() or not image_root.is_dir():
        raise FileNotFoundError(
            f"FashionIQ split/images directories not found under {args.dataset_root}"
        )

    backbone = FGCLIP2Backbone(
        model_id=args.model_id,
        max_num_patches=args.max_num_patches,
    ).to(device)
    backbone.eval()
    if any(parameter.requires_grad for parameter in backbone.model.parameters()):
        raise RuntimeError("FG-CLIP2-Large is not fully frozen")

    image_store = DirectoryImageStore(image_root=image_root)
    for split in args.splits:
        image_ids = load_all_image_ids(split_root, split)
        precompute_split(
            split=split,
            image_ids=image_ids,
            image_store=image_store,
            backbone=backbone,
            output_dir=args.cache_root / split / "images",
            batch_size=args.batch_size,
            parity_samples=args.parity_samples,
        )


if __name__ == "__main__":
    main()
