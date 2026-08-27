from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from backbones.fgclip2 import (
    FGCLIP2_DYNAMIC_PATCH_BUDGETS,
    FGCLIP2_LARGE_DIM,
    FGCLIP2_LARGE_MODEL_ID,
    FGCLIP2_LARGE_REVISION,
    FGCLIP2_PATCH_POLICY_NAME,
    FGCLIP2_PATCH_SIZE,
    FGCLIP2Backbone,
    determine_max_num_patches,
    validate_fgclip2_revision,
)
from datasets.common import DirectoryImageStore
from datasets.fashioniq import load_fashioniq_split_ids

CATEGORIES = ("dress", "shirt", "toptee")
VALID_SPLITS = ("train", "val")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute ragged frozen FG-CLIP2-Large dense image tokens."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data/fashionIQ_dataset"))
    parser.add_argument(
        "--cache-root", type=Path, default=Path("features/fashioniq/fgclip2-large")
    )
    parser.add_argument("--model-id", default=FGCLIP2_LARGE_MODEL_ID)
    parser.add_argument("--revision", default=FGCLIP2_LARGE_REVISION)
    parser.add_argument("--splits", nargs="+", choices=VALID_SPLITS, default=list(VALID_SPLITS))
    parser.add_argument("--batch-size", type=int, default=90)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--storage-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--parity-samples", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_all_image_ids(split_root: Path, split: str) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for category in CATEGORIES:
        for image_id in load_fashioniq_split_ids(split_root, split, category):
            if image_id not in seen:
                seen.add(image_id)
                result.append(image_id)
    return result


def _manifest(
    *,
    split: str,
    backbone: FGCLIP2Backbone,
    image_ids: list[str],
    storage_dtype: np.dtype,
    token_counts: np.ndarray,
    patch_budget_counts: Counter[int],
    parity_count: int,
    parity_max_abs_error: float,
) -> dict:
    return {
        "dataset": "FashionIQ",
        "split": split,
        "feature_kind": "fgclip2_dense_image_tokens_ragged",
        "model_id": backbone.model_id,
        "revision": backbone.revision,
        "feature_dim": FGCLIP2_LARGE_DIM,
        "storage_dtype": np.dtype(storage_dtype).name,
        "requires_grad": False,
        "num_images": len(image_ids),
        "total_token_count": int(token_counts.sum()),
        "min_token_count": int(token_counts.min()),
        "mean_token_count": float(token_counts.mean()),
        "max_token_count": int(token_counts.max()),
        "preprocessing": {
            "patch_policy": FGCLIP2_PATCH_POLICY_NAME,
            "patch_size": FGCLIP2_PATCH_SIZE,
            "possible_patch_budgets": list(FGCLIP2_DYNAMIC_PATCH_BUDGETS),
            "real_token_rule": "spatial_shapes[0] * spatial_shapes[1]",
            "source": (
                "https://huggingface.co/qihoo360/fg-clip2-large/blob/"
                f"{backbone.revision}/README.md#dense-feature-effect-display"
            ),
        },
        "patch_budget_counts": {
            str(value): int(patch_budget_counts.get(value, 0))
            for value in FGCLIP2_DYNAMIC_PATCH_BUDGETS
        },
        "parity_samples": parity_count,
        "parity_max_abs_error": parity_max_abs_error,
    }


@torch.inference_mode()
def precompute_split(
    *,
    split: str,
    image_ids: list[str],
    image_store: DirectoryImageStore,
    backbone: FGCLIP2Backbone,
    output_dir: Path,
    batch_size: int,
    storage_dtype: np.dtype,
    parity_samples: int,
    overwrite: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    final_files = (
        "values.npy", "offsets.npy", "spatial_shapes.npy", "name_to_idx.json", "manifest.json"
    )
    if not overwrite and any((output_dir / name).exists() for name in final_files):
        raise FileExistsError(
            f"Dense cache already exists under {output_dir}; pass --overwrite explicitly"
        )

    chunk_paths: list[Path] = []
    all_shapes: list[np.ndarray] = []
    all_counts: list[int] = []
    patch_budget_counts: Counter[int] = Counter()
    parity_direct: list[torch.Tensor] = []
    parity_ids: list[str] = []

    with tempfile.TemporaryDirectory(prefix="dense_chunks_", dir=output_dir) as temp_name:
        temp_dir = Path(temp_name)
        for start in tqdm(
            range(0, len(image_ids), batch_size),
            desc=f"FG-CLIP2 dense [{split}]",
            dynamic_ncols=True,
        ):
            batch_ids = image_ids[start : start + batch_size]
            images = [image_store.load(image_id) for image_id in batch_ids]
            patch_budget_counts.update(determine_max_num_patches(image) for image in images)
            batch_tokens, batch_shapes = backbone.encode_image_dense(images)
            arrays: list[np.ndarray] = []
            for image_id, tokens, shape in zip(
                batch_ids, batch_tokens, batch_shapes, strict=True
            ):
                if len(parity_direct) < parity_samples:
                    parity_ids.append(image_id)
                    parity_direct.append(tokens.cpu().clone())
                array = tokens.cpu().numpy().astype(storage_dtype, copy=False)
                arrays.append(array)
                all_shapes.append(shape.cpu().numpy().astype(np.int32, copy=False))
                all_counts.append(len(array))
            chunk = np.concatenate(arrays, axis=0)
            chunk_path = temp_dir / f"{start:08d}.npy"
            np.save(chunk_path, chunk)
            chunk_paths.append(chunk_path)

        counts = np.asarray(all_counts, dtype=np.int64)
        offsets = np.empty(len(image_ids) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])
        values = np.lib.format.open_memmap(
            output_dir / "values.npy",
            mode="w+",
            dtype=storage_dtype,
            shape=(int(offsets[-1]), FGCLIP2_LARGE_DIM),
        )
        cursor = 0
        for chunk_path in chunk_paths:
            chunk = np.load(chunk_path, mmap_mode="r")
            values[cursor : cursor + len(chunk)] = chunk
            cursor += len(chunk)
        values.flush()

    np.save(output_dir / "offsets.npy", offsets)
    np.save(output_dir / "spatial_shapes.npy", np.stack(all_shapes).astype(np.int32))
    name_to_idx = {image_id: index for index, image_id in enumerate(image_ids)}
    with (output_dir / "name_to_idx.json").open("w", encoding="utf-8") as file:
        json.dump(name_to_idx, file, indent=2)

    reloaded = np.load(output_dir / "values.npy", mmap_mode="r")
    if reloaded.shape != (int(offsets[-1]), FGCLIP2_LARGE_DIM):
        raise RuntimeError("Reloaded dense values have an unexpected shape")
    if not np.isfinite(reloaded).all():
        raise FloatingPointError("Reloaded dense cache contains NaN or Inf")

    parity_max_abs_error = 0.0
    for image_id, direct in zip(parity_ids, parity_direct, strict=True):
        index = name_to_idx[image_id]
        cached = torch.from_numpy(reloaded[offsets[index] : offsets[index + 1]].copy()).float()
        if cached.shape != direct.shape:
            raise RuntimeError("FG-CLIP2 dense direct/cache token-count parity failed")
        error = float((direct.float() - cached).abs().max().item())
        parity_max_abs_error = max(parity_max_abs_error, error)
    tolerance = 5.0e-3 if np.dtype(storage_dtype) == np.dtype(np.float16) else 1.0e-6
    if parity_max_abs_error > tolerance:
        raise RuntimeError(
            "FG-CLIP2 dense direct/cache value parity failed: "
            f"max_abs_error={parity_max_abs_error}, tolerance={tolerance}"
        )

    manifest = _manifest(
        split=split,
        backbone=backbone,
        image_ids=image_ids,
        storage_dtype=storage_dtype,
        token_counts=counts,
        patch_budget_counts=patch_budget_counts,
        parity_count=len(parity_ids),
        parity_max_abs_error=parity_max_abs_error,
    )
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    print(
        f"[{split}] dense: {len(image_ids)} images, {int(offsets[-1])} tokens "
        f"({np.dtype(storage_dtype).name}) -> {output_dir}"
    )
    print(f"[{split}] dense parity max abs error: {parity_max_abs_error:.3e}")


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.parity_samples < 0:
        raise ValueError("Invalid batch-size/parity-samples")
    if args.model_id != FGCLIP2_LARGE_MODEL_ID:
        raise ValueError(f"A8.0 requires exactly {FGCLIP2_LARGE_MODEL_ID}")
    revision = validate_fgclip2_revision(args.revision)
    if revision != FGCLIP2_LARGE_REVISION:
        raise ValueError(f"A8.0 requires revision={FGCLIP2_LARGE_REVISION}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    split_root = args.dataset_root / "image_splits"
    image_root = args.dataset_root / "images"
    if not split_root.is_dir() or not image_root.is_dir():
        raise FileNotFoundError("FashionIQ image_splits/images are missing")

    backbone = FGCLIP2Backbone(model_id=args.model_id, revision=revision).to(device)
    backbone.eval()
    image_store = DirectoryImageStore(image_root=image_root)
    storage_dtype = np.dtype(args.storage_dtype)
    for split in args.splits:
        precompute_split(
            split=split,
            image_ids=load_all_image_ids(split_root, split),
            image_store=image_store,
            backbone=backbone,
            output_dir=args.cache_root / split / "dense_images",
            batch_size=args.batch_size,
            storage_dtype=storage_dtype,
            parity_samples=args.parity_samples,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
