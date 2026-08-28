from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from backbones.fgclip2_base import (
    FGCLIP2_BASE_REVISION,
    FGCLIP2BaseBackbone,
    TextTuningConfig,
    VisionTuningConfig,
    determine_max_num_patches,
)
from cache.taper_mag import (
    DENSE_DIRECTORY,
    GLOBAL_DIRECTORY,
    ImageCacheManifest,
    stable_json_hash,
)
from datasets.common import DirectoryImageStore
from datasets.fashioniq import load_fashioniq_annotations, load_fashioniq_split_ids


CATEGORIES = ("dress", "shirt", "toptee")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute split global and reference-only dense FG-CLIP2 caches"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=Path("features"))
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES, default=list(CATEGORIES))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument(
        "--max-images",
        type=int,
        help="Debug-only truncation of both scopes; manifests are rejected by full training",
    )
    return parser.parse_args()


def deduplicate_ordered(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def fashioniq_cache_scopes(
    dataset_root: Path,
    split: str,
    categories: Sequence[str],
) -> tuple[list[str], list[str]]:
    """Return complete split IDs and annotation-derived reference IDs."""
    global_ids = deduplicate_ordered(
        [
            image_id
            for category in categories
            for image_id in load_fashioniq_split_ids(
                dataset_root / "image_splits", split, category
            )
        ]
    )
    annotations = [
        annotation
        for category in categories
        for annotation in load_fashioniq_annotations(
            dataset_root / "captions", split, category
        )
    ]
    reference_ids = deduplicate_ordered(
        [annotation.reference_id for annotation in annotations]
    )
    required_global_ids = {
        image_id
        for annotation in annotations
        for image_id in (annotation.reference_id, annotation.target_id)
        if image_id is not None
    }
    missing = sorted(required_global_ids.difference(global_ids))
    if missing:
        raise RuntimeError(
            f"Official {split} image split omits {len(missing)} annotated reference/target IDs; "
            f"first missing ID: {missing[0]}"
        )
    return global_ids, reference_ids


def _processor_spatial_shapes(
    backbone: FGCLIP2BaseBackbone,
    images: Sequence[Image.Image],
) -> torch.Tensor:
    grouped: dict[int, list[int]] = {}
    for index, image in enumerate(images):
        grouped.setdefault(determine_max_num_patches(image), []).append(index)
    shapes = torch.empty(len(images), 2, dtype=torch.long)
    for budget, indices in grouped.items():
        processor_batch = backbone.image_processor(
            images=[images[index] for index in indices],
            max_num_patches=budget,
            return_tensors="pt",
        )
        group_shapes = processor_batch["spatial_shapes"].long().cpu()
        if group_shapes.shape != (len(indices), 2):
            raise RuntimeError("FG-CLIP2 processor returned invalid spatial_shapes")
        for local_index, original_index in enumerate(indices):
            shapes[original_index] = group_shapes[local_index]
    return shapes


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError("--max-images must be positive")
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    model_dtype = (
        torch.bfloat16
        if args.dtype == "bfloat16" and device.type == "cuda"
        else torch.float32
    )
    storage_dtype = np.float16 if args.dtype == "bfloat16" else np.float32
    output = (
        args.cache_root
        / "fashioniq"
        / "fgclip2-base"
        / FGCLIP2_BASE_REVISION
        / args.split
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing cache namespace: {output}")
    global_ids, reference_ids = fashioniq_cache_scopes(
        args.dataset_root, args.split, tuple(args.categories)
    )
    complete_scope = args.max_images is None
    if args.max_images is not None:
        global_ids = global_ids[: args.max_images]
        reference_ids = reference_ids[: args.max_images]

    global_dir = output / GLOBAL_DIRECTORY
    dense_dir = output / DENSE_DIRECTORY
    global_dir.mkdir(parents=True, exist_ok=False)
    dense_dir.mkdir(parents=True, exist_ok=False)
    global_mapping = {image_id: index for index, image_id in enumerate(global_ids)}
    reference_mapping = {image_id: index for index, image_id in enumerate(reference_ids)}
    image_store = DirectoryImageStore(args.dataset_root / "images")
    backbone = FGCLIP2BaseBackbone(
        dtype=model_dtype,
        text_tuning=TextTuningConfig(
            mode="frozen",
            num_unfrozen_blocks=0,
            train_final_norm=False,
            train_projection=False,
        ),
        vision_tuning=VisionTuningConfig(),
    ).to(device).eval()

    global_mmap = np.lib.format.open_memmap(
        global_dir / "global.npy",
        mode="w+",
        dtype=storage_dtype,
        shape=(len(global_ids), backbone.contract.retrieval_dim),
    )
    for start in tqdm(
        range(0, len(global_ids), args.batch_size), desc="FG-CLIP2 global"
    ):
        batch_ids = global_ids[start : start + args.batch_size]
        images = [image_store.load(image_id) for image_id in batch_ids]
        values = backbone.encode_image_global(images).cpu().numpy().astype(storage_dtype)
        global_mmap[start : start + len(batch_ids)] = values
    global_mmap.flush()

    spatial_shapes = np.empty((len(reference_ids), 2), dtype=np.int64)
    for start in tqdm(
        range(0, len(reference_ids), args.batch_size),
        desc="FG-CLIP2 dense shape scan",
    ):
        batch_ids = reference_ids[start : start + args.batch_size]
        images = [image_store.load(image_id) for image_id in batch_ids]
        spatial_shapes[start : start + len(batch_ids)] = _processor_spatial_shapes(
            backbone, images
        ).numpy()
    lengths = spatial_shapes.prod(axis=1, dtype=np.int64)
    offsets = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(lengths, dtype=np.int64)]
    )
    dense_mmap = np.lib.format.open_memmap(
        dense_dir / "dense_values.npy",
        mode="w+",
        dtype=storage_dtype,
        shape=(int(offsets[-1]), backbone.contract.vision_dim),
    )
    for start in tqdm(
        range(0, len(reference_ids), args.batch_size), desc="FG-CLIP2 reference dense"
    ):
        batch_ids = reference_ids[start : start + args.batch_size]
        images = [image_store.load(image_id) for image_id in batch_ids]
        dense = backbone.encode_image_dense(images)
        for local_index in range(len(batch_ids)):
            absolute_index = start + local_index
            row = dense.tokens[local_index, dense.mask[local_index]].cpu().numpy()
            if tuple(dense.spatial_shapes[local_index].cpu().tolist()) != tuple(
                spatial_shapes[absolute_index].tolist()
            ) or row.shape[0] != lengths[absolute_index]:
                raise RuntimeError("Dense extraction disagrees with processor shape scan")
            dense_mmap[offsets[absolute_index] : offsets[absolute_index + 1]] = row.astype(
                storage_dtype
            )
    dense_mmap.flush()
    np.save(dense_dir / "dense_offsets.npy", offsets, allow_pickle=False)
    np.save(dense_dir / "spatial_shapes.npy", spatial_shapes, allow_pickle=False)
    _write_json(global_dir / "name_to_idx.json", global_mapping)
    _write_json(dense_dir / "reference_name_to_idx.json", reference_mapping)

    backbone_manifest = backbone.manifest()
    processor_hash = stable_json_hash(backbone_manifest.image_processor_config)
    common = dict(
        schema_version=2,
        model_id=backbone.model_id,
        revision=backbone.revision,
        processor_config_hash=processor_hash,
        dtype=np.dtype(storage_dtype).name,
        patch_policy=backbone_manifest.vision_patch_policy,
        split=args.split,
        complete_split=complete_scope,
    )
    global_manifest = ImageCacheManifest(
        cache_kind="global",
        image_scope="complete_split",
        extraction_method="official get_image_features + L2 normalize",
        normalization="L2",
        image_id_mapping_hash=stable_json_hash(global_mapping),
        feature_dim=backbone.contract.retrieval_dim,
        spatial_shapes_present=False,
        image_count=len(global_ids),
        **common,
    )
    dense_manifest = ImageCacheManifest(
        cache_kind="dense_reference",
        image_scope="reference_only",
        extraction_method="official get_image_dense_feature; real prefix from spatial_shapes",
        normalization="none",
        image_id_mapping_hash=stable_json_hash(reference_mapping),
        feature_dim=backbone.contract.vision_dim,
        spatial_shapes_present=True,
        image_count=len(reference_ids),
        **common,
    )
    global_manifest.write(global_dir / "manifest.json")
    dense_manifest.write(dense_dir / "manifest.json")
    _write_json(output / "backbone_manifest.json", asdict(backbone_manifest))
    print(
        f"Wrote mmap cache to {output}: {len(global_ids)} global images, "
        f"{len(reference_ids)} reference-only dense images, {int(offsets[-1])} dense tokens"
    )


if __name__ == "__main__":
    main()
