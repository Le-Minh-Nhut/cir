from __future__ import annotations

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


def missing_paths(paths: tuple[Path, ...]) -> list[Path]:
    return [path for path in paths if not (path.is_dir() if path.suffix == "" else path.is_file())]
