from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from tqdm import tqdm

from backbones.fgclip2_base import (
    FGCLIP2_BASE_REVISION,
    FGCLIP2BaseBackbone,
    TextTuningConfig,
    VisionTuningConfig,
)
from cache.taper_mag import ImageCacheManifest, stable_json_hash
from datasets.common import DirectoryImageStore
from datasets.fashioniq import load_fashioniq_split_ids


CATEGORIES = ("dress", "shirt", "toptee")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute frozen FG-CLIP2-Base FashionIQ vision features")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=Path("features"))
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument(
        "--max-images",
        type=int,
        help="Debug-only partial split; manifests mark complete_split=false",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model_dtype = torch.bfloat16 if args.dtype == "bfloat16" and device.type == "cuda" else torch.float32
    output = (
        args.cache_root
        / "fashioniq"
        / "fgclip2-base"
        / FGCLIP2_BASE_REVISION
        / args.split
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing cache namespace: {output}")
    output.mkdir(parents=True, exist_ok=False)
    split_root = args.dataset_root / "image_splits"
    image_ids: list[str] = []
    seen: set[str] = set()
    for category in CATEGORIES:
        for image_id in load_fashioniq_split_ids(split_root, args.split, category):
            if image_id not in seen:
                seen.add(image_id)
                image_ids.append(image_id)
    complete_split = args.max_images is None
    if args.max_images is not None:
        if args.max_images <= 0:
            raise ValueError("--max-images must be positive")
        image_ids = image_ids[: args.max_images]
    mapping = {image_id: index for index, image_id in enumerate(image_ids)}
    image_store = DirectoryImageStore(args.dataset_root / "images")
    backbone = FGCLIP2BaseBackbone(
        dtype=model_dtype,
        text_tuning=TextTuningConfig(
            mode="frozen", num_unfrozen_blocks=0, train_final_norm=False, train_projection=False
        ),
        vision_tuning=VisionTuningConfig(),
    ).to(device).eval()
    globals_: list[torch.Tensor] = []
    dense_rows: list[torch.Tensor] = []
    shapes: list[torch.Tensor] = []
    for start in tqdm(range(0, len(image_ids), args.batch_size), desc="FG-CLIP2 vision"):
        batch_ids = image_ids[start : start + args.batch_size]
        images = [image_store.load(image_id) for image_id in batch_ids]
        globals_.append(backbone.encode_image_global(images).cpu().to(model_dtype))
        dense = backbone.encode_image_dense(images)
        for index in range(len(images)):
            dense_rows.append(dense.tokens[index, dense.mask[index]].cpu().to(model_dtype))
            shapes.append(dense.spatial_shapes[index].cpu())
    global_tensor = torch.cat(globals_)
    dense_flat = torch.cat(dense_rows)
    offsets = torch.tensor(
        [0] + list(torch.tensor([row.shape[0] for row in dense_rows]).cumsum(0).tolist()),
        dtype=torch.long,
    )
    spatial_shapes = torch.stack(shapes)
    torch.save(global_tensor, output / "global.pt")
    torch.save(dense_flat, output / "dense_flat.pt")
    torch.save(offsets, output / "dense_offsets.pt")
    torch.save(spatial_shapes, output / "spatial_shapes.pt")
    (output / "name_to_idx.json").write_text(
        json.dumps(mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    backbone_manifest = backbone.manifest()
    mapping_hash = stable_json_hash(mapping)
    processor_hash = stable_json_hash(backbone_manifest.image_processor_config)
    common = dict(
        schema_version=1,
        model_id=backbone.model_id,
        revision=backbone.revision,
        processor_config_hash=processor_hash,
        dtype=str(global_tensor.dtype).removeprefix("torch."),
        image_id_mapping_hash=mapping_hash,
        patch_policy=backbone_manifest.vision_patch_policy,
        split=args.split,
        image_count=len(image_ids),
        complete_split=complete_split,
    )
    global_manifest = ImageCacheManifest(
        extraction_method="official get_image_features + L2 normalize",
        normalization="L2",
        feature_dim=backbone.contract.retrieval_dim,
        spatial_shapes_present=False,
        **common,
    )
    dense_manifest = ImageCacheManifest(
        extraction_method="official get_image_dense_feature; real prefix from spatial_shapes",
        normalization="none",
        feature_dim=backbone.contract.vision_dim,
        spatial_shapes_present=True,
        **common,
    )
    global_manifest.write(output / "manifest_global.json")
    dense_manifest.write(output / "manifest_dense.json")
    (output / "backbone_manifest.json").write_text(
        json.dumps(asdict(backbone_manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(image_ids)} images to {output}")


if __name__ == "__main__":
    main()
