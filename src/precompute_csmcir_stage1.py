from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
import numpy as np

from datasets.fashioniq import load_fashioniq_annotations, load_fashioniq_split_ids
from teachers.csmcir import CSMCIRStage1Teacher

CATEGORIES = ("dress", "shirt", "toptee")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-root", type=Path, default=Path("data/FashionIQ"))
    parser.add_argument("--csmcir-root", type=Path, default=Path("teacher/repos/CSMCIR"))
    parser.add_argument("--checkpoint", type=Path, default=Path("teacher/checkpoints/csmcir/fashioniq_tuned_clip_best.pt"))
    parser.add_argument("--output-root", type=Path, default=Path("features/fashioniq/csmcir"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()

@torch.inference_mode()
def precompute_to_disk(
    teacher,
    entries,
    image_root,
    batch_size,
    device,
    output_dir,
    kind,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    mmap = None

    for start in tqdm(range(0, len(entries), batch_size), desc=f"CSMCIR {kind}"):
        batch_entries = entries[start:start + batch_size]
        images = torch.stack([
            load_image(image_root=image_root, image_id=image_id, category=category, preprocess=teacher.preprocess)
            for image_id, category in batch_entries
        ]).to(device)

        if kind == "retrieval":
            features, _ = teacher.encode_image_tokens(images)
        elif kind == "native":
            features = teacher.encode_reference(images)
        else:
            raise ValueError(f"Unsupported feature kind: {kind}")

        features = features.float().cpu()
        if not torch.isfinite(features).all():
            raise FloatingPointError(
                f"Non-finite {kind} features at rows "
                f"{start}:{start + len(batch_entries)}"
            )
        if mmap is None:
            shape = (len(entries), *features.shape[1:])
            mmap = np.lib.format.open_memmap(output_dir / "images.npy", mode="w+", dtype=np.float32, shape=shape)

        end = start + len(batch_entries)
        mmap[start:end] = features.numpy()
        mmap.flush()

        del images, features

    if mmap is None:
        raise RuntimeError("No features produced")

    image_ids = [image_id for image_id, _ in entries]

    with (output_dir / "name_to_idx.json").open("w", encoding="utf-8") as file:
        json.dump(
            {image_id: i for i, image_id in enumerate(image_ids)},
            file,
            indent=2,
        )

    del mmap

def resolve_image_path(image_root: Path, image_id: str, category: str) -> Path:
    candidates = []

    for ext in (".jpg", ".png", ".jpeg"):
        candidates.append(image_root / category / f"{image_id}{ext}")

    for ext in (".jpg", ".png", ".jpeg"):
        candidates.append(image_root / f"{image_id}{ext}")

    for path in candidates:
        if path.is_file():
            return path

    raise FileNotFoundError(f"Image not found: {category}/{image_id}")


def load_image(*, image_root: Path, image_id: str, category: str, preprocess) -> torch.Tensor:
    path = resolve_image_path(
        image_root=image_root,
        image_id=image_id,
        category=category,
    )

    with Image.open(path) as image:
        image = image.convert("RGB")
        return preprocess(image)


def unique_reference_entries(annotation_root: Path, split: str) -> list[tuple[str, str]]:
    entries = []
    seen = set()

    for category in CATEGORIES:
        annotations = load_fashioniq_annotations(annotation_root=annotation_root, split=split, category=category)

        for annotation in annotations:
            image_id = annotation.reference_id
            if image_id in seen:
                continue

            seen.add(image_id)
            entries.append((image_id, category))

    return entries

def split_entries(split_root: Path, split: str) -> list[tuple[str, str]]:
    entries = []
    seen = set()

    for category in CATEGORIES:
        image_ids = load_fashioniq_split_ids(split_root=split_root, split=split, category=category)

        for image_id in image_ids:
            if image_id in seen:
                continue

            seen.add(image_id)
            entries.append((image_id, category))

    return entries

def main():
    args = parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    dataset_root = args.dataset_root.resolve()
    split_root = dataset_root / "image_splits"
    image_root = dataset_root / "images"
    device = torch.device(args.device)

    teacher = CSMCIRStage1Teacher(
        csmcir_root=args.csmcir_root,
        checkpoint_path=args.checkpoint,
        device=args.device,
    ).to(device).eval()

    annotation_root = dataset_root / "captions"

    for split in ("train", "val"):
        retrieval_entries = split_entries(split_root, split)
        native_entries = unique_reference_entries(annotation_root, split)

        print(f"{split} retrieval images:", len(retrieval_entries))
        print(f"{split} reference images:", len(native_entries))

        precompute_to_disk(
            teacher=teacher,
            entries=retrieval_entries,
            image_root=image_root,
            batch_size=args.batch_size,
            device=device,
            output_dir=args.output_root / split / "retrieval",
            kind="retrieval",
        )

        precompute_to_disk(
            teacher=teacher,
            entries=native_entries,
            image_root=image_root,
            batch_size=args.batch_size,
            device=device,
            output_dir=args.output_root / split / "native",
            kind="native",
        )

    print()
    print("DONE")
    print("Saved under:", args.output_root)


if __name__ == "__main__":
    main()