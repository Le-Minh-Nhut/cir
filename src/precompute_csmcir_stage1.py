from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

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


def validation_gallery_entries_for_category(split_root: Path, category: str) -> list[tuple[str, str]]:
    image_ids = load_fashioniq_split_ids(split_root=split_root, split="val", category=category)
    if len(set(image_ids)) != len(image_ids):
        raise RuntimeError(f"Duplicate image IDs inside FashionIQ category={category}")

    return [
        (image_id, category)
        for image_id in image_ids
    ]


def load_csmcir_target_captions(csmcir_root: Path) -> dict:
    root = csmcir_root.resolve() / "COT_ours2" / "fashioniq"
    result = {}

    for category in CATEGORIES:
        path = root / f"{category}_cot_val.json"

        if not path.is_file():
            raise FileNotFoundError(f"Missing CSMCIR target caption file: {path}")

        with path.open("r", encoding="utf-8",) as file:
            result.update(json.load(file))

    return result

def get_target_caption(caption_dict: dict, *, image_id: str) -> str:
    if image_id not in caption_dict:
        raise KeyError(f"Missing CSMCIR target caption: {image_id}")

    entry = caption_dict[image_id]
    if isinstance(entry, str):
        return entry

    if isinstance(entry, dict) and "Final_Caption" in entry:
        return entry["Final_Caption"]

    raise ValueError(f"Unsupported CSMCIR target-caption entry for {image_id}: {entry!r}")


def save_features(features: torch.Tensor, entries: list[tuple[str, str]], output_dir: Path) -> None:
    image_ids = [
        image_id
        for image_id, _ in entries
    ]

    if features.ndim < 2:
        raise ValueError(f"Expected [N,...,D], got {tuple(features.shape)}")

    if features.shape[0] != len(image_ids):
        raise ValueError("Feature count != image count")

    if len(set(image_ids))!= len(image_ids):
        raise ValueError("Duplicate image IDs")

    assert_finite_chunked(features)

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(features, output_dir / "images.pt")
    name_to_idx = {
        image_id: index
        for index, image_id
        in enumerate(image_ids)
    }
    with (output_dir / "name_to_idx.json").open("w", encoding="utf-8") as file:
        json.dump(name_to_idx, file, indent=2)

def assert_finite_chunked(features: torch.Tensor, chunk_rows: int = 32) -> None:
    for start in range(0, features.shape[0], chunk_rows):
        end = min(start + chunk_rows, features.shape[0],)

        if not torch.isfinite(features[start:end]).all().item():
            raise FloatingPointError(f"Feature cache contains NaN/Inf in rows {start}:{end}")

@torch.inference_mode()
def precompute_references(
    *,
    teacher: CSMCIRStage1Teacher,
    entries: list[tuple[str, str]],
    image_root: Path,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    output = None

    for start in tqdm(range(0, len(entries), batch_size),desc="CSMCIR references"):
        batch_entries = entries[start:start + batch_size]

        images = torch.stack(
            [
                load_image(
                    image_root=image_root,
                    image_id=image_id,
                    category=category,
                    preprocess=teacher.preprocess,
                )
                for image_id, category in batch_entries
            ]
        ).to(device)

        features = teacher.encode_reference(images)
        features_cpu = features.cpu()

        if output is None:
            output = torch.empty((len(entries), *features_cpu.shape[1:]), dtype=features_cpu.dtype)

        end = start + len(batch_entries)
        output[start:end].copy_(features_cpu)
        del images
        del features
        del features_cpu

    if output is None:
        raise RuntimeError("No reference features produced")

    return output

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

@torch.inference_mode()
def precompute_images(teacher, entries, image_root, batch_size, device):
    retrieval_output = None
    native_output = None

    for start in tqdm(range(0, len(entries), batch_size), desc="CSMCIR images"):
        batch_entries = entries[start:start + batch_size]

        images = torch.stack([
            load_image(
                image_root=image_root,
                image_id=image_id,
                category=category,
                preprocess=teacher.preprocess,
            )
            for image_id, category in batch_entries
        ]).to(device)

        retrieval, native = teacher.encode_image_tokens(images)
        retrieval = retrieval.cpu()
        native = native.cpu()

        if retrieval_output is None:
            retrieval_output = torch.empty((len(entries), *retrieval.shape[1:]), dtype=retrieval.dtype)
            native_output = torch.empty((len(entries), *native.shape[1:]), dtype=native.dtype)

        end = start + len(batch_entries)
        retrieval_output[start:end].copy_(retrieval)
        native_output[start:end].copy_(native)

        del images, retrieval, native

    if retrieval_output is None or native_output is None:
        raise RuntimeError("No image features produced")

    return retrieval_output, native_output

@torch.inference_mode()
def precompute_gallery(
    *,
    teacher: CSMCIRStage1Teacher,
    entries: list[tuple[str, str]],
    caption_dicts: dict[str, dict],
    image_root: Path,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    output = None

    for start in tqdm(range(0, len(entries), batch_size), desc="CSMCIR gallery"):
        batch_entries = entries[start:start + batch_size]
        images = torch.stack(
            [
                load_image(
                    image_root=image_root,
                    image_id=image_id,
                    category=category,
                    preprocess=teacher.preprocess,
                )
                for image_id, category in batch_entries
            ]
        ).to(device)

        captions = [
            get_target_caption(
                caption_dicts,
                image_id=image_id,
            )
            for image_id, category in batch_entries
        ]

        features = teacher.encode_gallery(images, captions)
        features_cpu = features.cpu()
        if output is None:
            output = torch.empty((len(entries), *features_cpu.shape[1:]), dtype=features_cpu.dtype)

        end = start + len(batch_entries)
        output[start:end].copy_(features_cpu)

        del images
        del features
        del features_cpu

    if output is None:
        raise RuntimeError("No gallery features produced")

    return output


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

    for split in ("train", "val"):
        entries = split_entries(split_root, split)

        print(f"{split} images:", len(entries))

        retrieval, native = precompute_images(
            teacher=teacher,
            entries=entries,
            image_root=image_root,
            batch_size=args.batch_size,
            device=device,
        )

        print(f"{split} retrieval:", tuple(retrieval.shape))
        print(f"{split} native:", tuple(native.shape))

        save_features(retrieval, entries, args.output_root / split / "retrieval")
        save_features(native, entries, args.output_root / split / "native")

        del retrieval, native

    print()
    print("DONE")
    print("Saved under:", args.output_root)


if __name__ == "__main__":
    main()