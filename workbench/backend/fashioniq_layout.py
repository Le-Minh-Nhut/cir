from __future__ import annotations

import json
from pathlib import Path

CATEGORIES = ("dress", "shirt", "toptee")


def standard_paths(root: Path) -> tuple[Path, ...]:
    return (
        root / "captions",
        root / "image_splits",
        root / "images",
        *(root / "captions" / f"cap.{category}.val.json" for category in CATEGORIES),
        *(root / "image_splits" / f"split.{category}.val.json" for category in CATEGORIES),
    )


def ilearn_resized_paths(root: Path) -> tuple[Path, ...]:
    return (
        root / "captions",
        root / "image_splits",
        root / "resized_image",
        *(root / "resized_image" / category for category in CATEGORIES),
        *(root / "captions" / f"cap.{category}.val.json" for category in CATEGORIES),
        *(root / "captions" / f"correction_dict_{category}.json" for category in CATEGORIES),
        *(root / "image_splits" / f"split.{category}.val.json" for category in CATEGORIES),
    )


def ilearn_resized_training_paths(root: Path) -> tuple[Path, ...]:
    return (
        ilearn_resized_paths(root)
        + tuple(root / "captions" / f"cap.{category}.train.json" for category in CATEGORIES)
    )


def clvc_resized_paths(root: Path) -> tuple[Path, ...]:
    return (
        root / "captions",
        root / "image_splits",
        root / "resized_image",
        *(root / "resized_image" / category for category in CATEGORIES),
        *(root / "captions" / f"cap.{category}.{split}.json" for category in CATEGORIES for split in ("train", "val")),
        *(root / "image_splits" / f"split.{category}.val.json" for category in CATEGORIES),
    )


def dcnet_paths(root: Path) -> tuple[Path, ...]:
    return (
        root / "captions",
        root / "image_splits",
        root / "resized_images",
        *(root / "captions" / f"cap.{category}.glove.val.pkl" for category in CATEGORIES),
        *(root / "image_splits" / f"split.{category}.val.json" for category in CATEGORIES),
    )


def paths_for_layout(root: Path, layout: str) -> tuple[Path, ...]:
    if layout == "fashioniq_standard":
        return standard_paths(root)
    if layout == "fashioniq_ilearn_resized":
        return ilearn_resized_paths(root)
    if layout == "fashioniq_ilearn_resized_training":
        return ilearn_resized_training_paths(root)
    if layout == "fashioniq_clvc_resized":
        return clvc_resized_paths(root)
    if layout == "fashioniq_dcnet":
        return dcnet_paths(root)
    raise ValueError(f"unknown FashionIQ layout: {layout}")


def image_extension_problems(root: Path, extension: str) -> list[str]:
    """Every gallery image referenced by the validation splits must have *extension*.

    An official evaluator's image path is hard-coded in its source, so the split files and the
    files on disk have to agree. CSMCIR reads `fashionIQ_dataset/images/{id}.png` while the
    official FashionIQ release ships `.jpg`, which its own README shows; without this check the
    mismatch only appears when a GPU run fails on the first missing image.
    """
    problems: list[str] = []
    images = root / "images"
    if not images.is_dir():
        return [f"image directory missing: {images}"]
    for category in CATEGORIES:
        split = root / "image_splits" / f"split.{category}.val.json"
        if not split.is_file():
            problems.append(f"split file missing: {split}")
            continue
        try:
            image_ids = json.loads(split.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            problems.append(f"split file unreadable: {split} ({error})")
            continue
        missing = [image_id for image_id in image_ids
                   if not (images / f"{image_id}{extension}").is_file()]
        if missing:
            problems.append(
                f"{len(missing)} of {len(image_ids)} {category} validation images lack "
                f"{extension} in {images} (first: {missing[0]}{extension})")
    return problems


def missing_paths(paths: tuple[Path, ...]) -> list[Path]:
    return [path for path in paths if not (path.is_dir() if path.suffix == "" else path.is_file())]
