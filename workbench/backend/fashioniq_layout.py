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


def paths_for_layout(root: Path, layout: str) -> tuple[Path, ...]:
    if layout == "fashioniq_standard":
        return standard_paths(root)
    if layout == "fashioniq_ilearn_resized":
        return ilearn_resized_paths(root)
    raise ValueError(f"unknown FashionIQ layout: {layout}")


def missing_paths(paths: tuple[Path, ...]) -> list[Path]:
    return [path for path in paths if not (path.is_dir() if path.suffix == "" else path.is_file())]
