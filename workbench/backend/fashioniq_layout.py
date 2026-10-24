from __future__ import annotations

import json
from pathlib import Path

CATEGORIES = ("dress", "shirt", "toptee")


def canonical_dataset_link_path(model: dict, source_root: Path) -> Path | None:
    """Return model's declared in-checkout FashionIQ link, if any."""
    source_dir = model.get("source_dir")
    link_name = model.get("canonical_dataset_link")
    return source_root / source_dir / link_name if source_dir and link_name else None

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


#: The image directory each layout actually opens, plus whether it is flat or per-category.
#: An evaluator's image path is hard-coded in its own source, so the check has to look where that
#: source looks — `images/{id}.png` for the standard layout, `resized_image/{category}/{id}.jpg`
#: for the iLearn family, `resized_images/{id}.jpg` for DCNet.
_LAYOUT_IMAGE_DIRECTORY = {
    "fashioniq_standard": ("images", False),
    "fashioniq_ilearn_resized": ("resized_image", True),
    "fashioniq_ilearn_resized_training": ("resized_image", True),
    "fashioniq_clvc_resized": ("resized_image", True),
    "fashioniq_dcnet": ("resized_images", False),
}


_LAYOUT_IMAGE_EXTENSION = {
    "fashioniq_standard": ".png",
    "fashioniq_ilearn_resized": ".jpg",
    "fashioniq_ilearn_resized_training": ".jpg",
    "fashioniq_clvc_resized": ".jpg",
    "fashioniq_dcnet": ".jpg",
}


def image_extension_for_layout(layout: str, declared_extensions: list[str] | None = None) -> str:
    """Return layout's image suffix, preferring an explicit model declaration."""
    if declared_extensions:
        return declared_extensions[0]
    try:
        return _LAYOUT_IMAGE_EXTENSION[layout]
    except KeyError as error:
        raise ValueError(f"unknown FashionIQ layout: {layout}") from error


def image_extension_problems(root: Path, extension: str,
                             layout: str | None = None) -> list[str]:
    """Every gallery image referenced by the validation splits must exist with *extension*.

    An official evaluator hard-codes both the directory and the extension it opens, so the split
    files and the files on disk have to agree. CSMCIR reads `images/{id}.png` while the official
    FashionIQ release ships `.jpg`; the iLearn family reads `resized_image/<category>/{id}.jpg`.
    Without this check the mismatch appears only when a run fails on the first missing image.
    """
    problems: list[str] = []
    if layout is None:
        layout = "fashioniq_standard"
    try:
        directory_name, per_category = _LAYOUT_IMAGE_DIRECTORY[layout]
    except KeyError:
        return [f"unknown FashionIQ layout for an image-extension check: {layout}"]

    images = root / directory_name
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
        if per_category:
            folder = images / category
            if not folder.is_dir():
                problems.append(f"image directory missing: {folder}")
                continue
        else:
            folder = images
        missing = [image_id for image_id in image_ids
                   if not (folder / f"{image_id}{extension}").is_file()]
        if missing:
            problems.append(
                f"{len(missing)} of {len(image_ids)} {category} validation images lack "
                f"{extension} in {folder} (first: {missing[0]}{extension})")
    return problems


def missing_paths(paths: tuple[Path, ...]) -> list[Path]:
    return [path for path in paths if not (path.is_dir() if path.suffix == "" else path.is_file())]
