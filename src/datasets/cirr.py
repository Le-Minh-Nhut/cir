from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

from datasets.common import CIRSample, ImageStore


VALID_SPLITS = ("train", "val", "test1")


@dataclass(frozen=True, slots=True)
class CIRRAnnotation:
    pair_id: str
    reference_id: str
    target_id: str | None
    caption: str
    group_id: str | None
    group_members: tuple[str, ...]
    reference_rank: int | None
    target_rank: int | None
    target_soft: Any = None


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"CIRR JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _required_string(record: Mapping[str, Any], field: str, index: int) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"CIRR record {index} has invalid {field!r}")
    return value


def _validate_split(split: str) -> None:
    if split not in VALID_SPLITS:
        raise ValueError(f"Unsupported CIRR split {split!r}; expected one of {VALID_SPLITS}")


def load_cirr_annotations(
    caption_root: str | Path,
    split: str,
    *,
    version: str = "rc2",
) -> list[CIRRAnnotation]:
    _validate_split(split)
    path = Path(caption_root) / f"cap.{version}.{split}.json"
    records = _read_json(path)
    if not isinstance(records, list):
        raise ValueError(f"CIRR caption file must contain a JSON list: {path}")

    annotations: list[CIRRAnnotation] = []
    seen_pair_ids: set[str] = set()

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"CIRR record {index} must be a JSON object")

        pair_value = record.get("pairid")
        if not isinstance(pair_value, (str, int)) or isinstance(pair_value, bool):
            raise ValueError(f"CIRR record {index} has invalid 'pairid'")
        pair_id = str(pair_value)
        if pair_id in seen_pair_ids:
            raise ValueError(f"Duplicate CIRR pairid: {pair_id}")
        seen_pair_ids.add(pair_id)

        target_id = record.get("target_hard")
        if target_id is not None and (not isinstance(target_id, str) or not target_id):
            raise ValueError(f"CIRR record {index} has invalid 'target_hard'")
        if split in {"train", "val"} and target_id is None:
            raise ValueError(f"CIRR {split} record {index} is missing 'target_hard'")

        image_set = record.get("img_set") or {}
        if not isinstance(image_set, dict):
            raise ValueError(f"CIRR record {index} has invalid 'img_set'")
        members_value = image_set.get("members", [])
        if not isinstance(members_value, list) or not all(
            isinstance(member, str) and member for member in members_value
        ):
            raise ValueError(f"CIRR record {index} has invalid 'img_set.members'")
        group_members = tuple(members_value)
        if len(set(group_members)) != len(group_members):
            raise ValueError(f"CIRR record {index} has duplicate group members")

        reference_id = _required_string(record, "reference", index)
        if split == "val" and (
            reference_id not in group_members or target_id not in group_members
        ):
            raise ValueError(
                f"CIRR val record {index} must contain its reference and target in the group"
            )

        group_id = image_set.get("id")
        annotations.append(
            CIRRAnnotation(
                pair_id=pair_id,
                reference_id=reference_id,
                target_id=target_id,
                caption=_required_string(record, "caption", index),
                group_id=str(group_id) if group_id is not None else None,
                group_members=group_members,
                reference_rank=image_set.get("reference_rank"),
                target_rank=image_set.get("target_rank"),
                target_soft=record.get("target_soft"),
            )
        )

    return annotations


def load_cirr_image_mapping(
    split_root: str | Path,
    split: str,
    *,
    version: str = "rc2",
) -> dict[str, str]:
    _validate_split(split)
    path = Path(split_root) / f"split.{version}.{split}.json"
    value = _read_json(path)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"CIRR image split must be a non-empty JSON object: {path}")
    if not all(
        isinstance(image_id, str)
        and image_id
        and isinstance(relative_path, str)
        and relative_path
        for image_id, relative_path in value.items()
    ):
        raise ValueError(f"CIRR image split must map image IDs to relative paths: {path}")
    return dict(value)


def resolve_cirr_image_root(
    dataset_root: str | Path,
    split: str,
    *,
    version: str = "rc2",
) -> Path:
    """Resolve both common CIRR layouts from paths stored in the split JSON.

    Official split files contain paths such as ``./dev/...`` or ``./test1/...``.
    Some installations put those directories directly below the CIRR root, while
    older local setups place them below ``img_raw``.
    """
    root = Path(dataset_root)
    mapping = load_cirr_image_mapping(
        root / "image_splits", split, version=version
    )
    candidates = (root / "img_raw", root)
    missing_examples: dict[Path, str] = {}

    for candidate in candidates:
        missing = next(
            (
                relative_path
                for relative_path in mapping.values()
                if not (candidate / relative_path).is_file()
            ),
            None,
        )
        if missing is None:
            return candidate
        missing_examples[candidate] = missing

    details = "; ".join(
        f"{candidate} (missing {relative_path!r})"
        for candidate, relative_path in missing_examples.items()
    )
    raise FileNotFoundError(
        "Could not resolve CIRR image root from split paths. Tried: " + details
    )


@dataclass(frozen=True, slots=True)
class CIRRImageStore(ImageStore):
    image_root: Path
    image_paths: Mapping[str, str]

    def path_for(self, image_id: str) -> Path:
        try:
            relative_path = self.image_paths[image_id]
        except KeyError as error:
            raise KeyError(f"Image ID {image_id!r} is absent from the CIRR image split") from error

        path = self.image_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"CIRR image not found for {image_id!r}: {path}")
        return path

    def load(self, image_id: str) -> Image.Image:
        path = self.path_for(image_id)
        with Image.open(path) as image:
            return image.convert("RGB")


def build_cirr_image_store(
    image_root: str | Path,
    split_root: str | Path,
    split: str,
    *,
    version: str = "rc2",
) -> CIRRImageStore:
    return CIRRImageStore(
        image_root=Path(image_root),
        image_paths=load_cirr_image_mapping(split_root, split, version=version),
    )


class CIRRDataset(Dataset[CIRSample]):
    def __init__(
        self,
        caption_root: str | Path,
        split: str,
        *,
        version: str = "rc2",
    ) -> None:
        self.split = split
        self.version = version
        self.annotations = load_cirr_annotations(caption_root, split, version=version)

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> CIRSample:
        annotation = self.annotations[index]
        ground_truth_ids = (
            (annotation.target_id,) if annotation.target_id is not None else ()
        )
        return CIRSample(
            sample_id=f"cirr:{self.version}:{self.split}:{annotation.pair_id}",
            benchmark_id=annotation.pair_id,
            reference_id=annotation.reference_id,
            target_id=annotation.target_id,
            modification_text=annotation.caption,
            group_members=annotation.group_members,
            ground_truth_ids=ground_truth_ids,
            metadata={
                "group_id": annotation.group_id,
                "reference_rank": annotation.reference_rank,
                "target_rank": annotation.target_rank,
                "target_soft": annotation.target_soft,
            },
        )
