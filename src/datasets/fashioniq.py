from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from datasets.common import (
    CIRSample,
    compose_fashioniq_caption,
    expand_fashioniq_captions_bidirectionally,
)


VALID_CATEGORIES = {"dress", "shirt", "toptee"}
VALID_SPLITS = {"train", "val", "test"}
VALID_CAPTION_POLICIES = {
    "ordered_and",
    "normalized_ordered_and",
    "randomized_four_way",
}
VALID_SAMPLE_EXPANSIONS = {"none", "bidirectional"}


@dataclass(frozen=True)
class FashionIQAnnotation:
    """One original FashionIQ annotation before caption processing."""

    reference_id: str
    target_id: str | None
    captions: tuple[str, str]
    category: str
    index: int # vị trí sample trong file json 


def _read_json(path: str | Path) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_correction_dict(path: str | Path) -> dict[str, str]:
    """Load one FashionIQ token-correction dictionary."""

    data = _read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Correction dictionary must be a JSON object: {path}")

    return {
        str(key).lower(): str(value).lower()
        for key, value in data.items()
    }


def load_fashioniq_annotations(annotation_root: str | Path, split: str, category: str) -> list[FashionIQAnnotation]:
    """Load one ``cap.<category>.<split>.json`` annotation file."""

    if split not in VALID_SPLITS:
        raise ValueError(f"Invalid split: {split}")

    if category not in VALID_CATEGORIES:
        raise ValueError(f"Invalid category: {category}")

    path = Path(annotation_root) / f"cap.{category}.{split}.json"
    records = _read_json(path)

    if not isinstance(records, list):
        raise ValueError(f"Annotation file must contain a list: {path}")

    annotations: list[FashionIQAnnotation] = []

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Invalid record at index {index} in {path}")

        reference_id = record.get("candidate")
        target_id = record.get("target")
        captions = record.get("captions")

        if not isinstance(reference_id, str) or not reference_id:
            raise ValueError(f"Missing candidate at index {index} in {path}")

        if split in {"train", "val"}:
            if not isinstance(target_id, str) or not target_id:
                raise ValueError(f"Missing target at index {index} in {path}")
        elif target_id is not None and not isinstance(target_id, str):
            raise ValueError(f"Invalid target at index {index} in {path}")

        if (
            not isinstance(captions, list)
            or len(captions) != 2
            or not all(isinstance(caption, str) for caption in captions)
        ):
            raise ValueError(
                f"Expected exactly two captions at index {index} in {path}"
            )

        annotations.append(
            FashionIQAnnotation(
                reference_id=reference_id,
                target_id=target_id,
                captions=(captions[0], captions[1]),
                category=category,
                index=index,
            )
        )

    return annotations


def load_fashioniq_split_ids(
    split_root: str | Path,
    split: str,
    category: str,
) -> list[str]:
    """Load image IDs from ``split.<category>.<split>.json``."""

    if split not in VALID_SPLITS:
        raise ValueError(f"Invalid split: {split}")

    if category not in VALID_CATEGORIES:
        raise ValueError(f"Invalid category: {category}")

    path = Path(split_root) / f"split.{category}.{split}.json"
    image_ids = _read_json(path)

    if not isinstance(image_ids, list):
        raise ValueError(f"Split file must contain a list: {path}")

    if not all(isinstance(image_id, str) for image_id in image_ids):
        raise ValueError(f"Split file contains invalid image IDs: {path}")

    return image_ids


def build_pair_union_gallery(annotations: Sequence[FashionIQAnnotation]) -> list[str]:
    """Build the reduced VAL-style gallery.

    The gallery is the ordered unique union of all reference and target IDs
    appearing in the selected annotations.
    """

    gallery_ids = []

    for annotation in annotations:
        if annotation.reference_id not in gallery_ids:
            gallery_ids.append(annotation.reference_id)

        if (annotation.target_id is not None and annotation.target_id not in gallery_ids):
            gallery_ids.append(annotation.target_id)

    return gallery_ids


class FashionIQDataset(Dataset):
    """FashionIQ query dataset returning CIRSample objects.

    Caption policy:
        - ordered_and
        - normalized_ordered_and
        - randomized_four_way

    Sample expansion:
        - none
        - bidirectional

    ``set_epoch`` is used only to change deterministic random caption choices
    for ``randomized_four_way``.
    """

    def __init__(
        self,
        annotation_root: str | Path,
        split: str,
        categories: Sequence[str],
        caption_policy: str = "ordered_and",
        sample_expansion: str = "none",
        correction_dicts: Mapping[str, Mapping[str, str]] | None = None,
        seed: int = 42,
    ) -> None:
        if split not in VALID_SPLITS:
            raise ValueError(f"Invalid split: {split}")

        if not categories:
            raise ValueError("categories must not be empty")

        invalid_categories = [
            category
            for category in categories
            if category not in VALID_CATEGORIES
        ]
        if invalid_categories:
            raise ValueError(f"Invalid categories: {invalid_categories}")

        if caption_policy not in VALID_CAPTION_POLICIES:
            raise ValueError(f"Invalid caption policy: {caption_policy}")

        if sample_expansion not in VALID_SAMPLE_EXPANSIONS:
            raise ValueError(f"Invalid sample expansion: {sample_expansion}")

        if (
            sample_expansion == "bidirectional"
            and caption_policy == "randomized_four_way"
        ):
            raise ValueError(
                "bidirectional cannot be combined with randomized_four_way"
            )

        if split != "train" and caption_policy == "randomized_four_way":
            raise ValueError(
                "randomized_four_way should only be used for training"
            )

        if split != "train" and sample_expansion == "bidirectional":
            raise ValueError(
                "bidirectional expansion changes the evaluation query set; "
                "use it only for training or create a separate protocol"
            )

        self.annotation_root = Path(annotation_root)
        self.split = split
        self.categories = list(categories)
        self.caption_policy = caption_policy
        self.sample_expansion = sample_expansion
        self.seed = seed
        self.epoch = 0

        self.correction_dicts: dict[str, Mapping[str, str]] = {}
        if correction_dicts is not None:
            self.correction_dicts = {
                category: {
                    str(key).lower(): str(value).lower()
                    for key, value in correction_dict.items()
                }
                for category, correction_dict in correction_dicts.items()
            }

        if caption_policy == "normalized_ordered_and":
            missing = [
                category
                for category in self.categories
                if category not in self.correction_dicts
            ]
            if missing:
                raise ValueError(
                    f"Missing correction dictionaries for: {missing}"
                )

        self.annotations: list[FashionIQAnnotation] = []

        for category in self.categories:
            self.annotations.extend(
                load_fashioniq_annotations(
                    annotation_root=self.annotation_root,
                    split=self.split,
                    category=category,
                )
            )

    def set_epoch(self, epoch: int) -> None:
        """Change caption randomization for the next epoch."""

        self.epoch = epoch

    def __len__(self) -> int:
        if self.sample_expansion == "bidirectional":
            return 2 * len(self.annotations)

        return len(self.annotations)

    def _get_annotation(
        self,
        index: int,
    ) -> tuple[FashionIQAnnotation, str]:
        if self.sample_expansion == "bidirectional":
            annotation = self.annotations[index // 2]
            order = "forward" if index % 2 == 0 else "reverse"
            return annotation, order

        return self.annotations[index], "single"

    def _make_sample_id(
        self,
        annotation: FashionIQAnnotation,
        order: str,
    ) -> str:
        sample_id = (
            f"fashioniq:{self.split}:{annotation.category}:{annotation.index}"
        )

        if order != "single":
            sample_id += f":{order}"

        return sample_id

    def _make_rng(self, sample_id: str) -> random.Random:
        """Create a deterministic RNG for one sample and epoch."""

        text = f"{self.seed}:{self.epoch}:{sample_id}"
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        sample_seed = int.from_bytes(digest[:8], byteorder="big")

        return random.Random(sample_seed)

    def _compose_text(
        self,
        annotation: FashionIQAnnotation,
        order: str,
        sample_id: str,
    ) -> str:
        correction_dict = self.correction_dicts.get(annotation.category)

        if order in {"forward", "reverse"}:
            forward_text, reverse_text = (
                expand_fashioniq_captions_bidirectionally(
                    annotation.captions,
                    policy=self.caption_policy,
                    correction_dict=correction_dict,
                )
            )

            if order == "forward":
                return forward_text

            return reverse_text

        rng = None
        if self.caption_policy == "randomized_four_way":
            rng = self._make_rng(sample_id)

        return compose_fashioniq_caption(
            annotation.captions,
            policy=self.caption_policy,
            correction_dict=correction_dict,
            rng=rng,
        )

    def __getitem__(self, index: int) -> CIRSample:
        annotation, order = self._get_annotation(index)
        sample_id = self._make_sample_id(annotation, order)

        modification_text = self._compose_text(
            annotation=annotation,
            order=order,
            sample_id=sample_id,
        )

        metadata: dict[str, Any] = {
            "original_captions": annotation.captions,
            "caption_policy": self.caption_policy,
            "caption_order": order,
        }

        return CIRSample(
            sample_id=sample_id,
            benchmark_id=None,
            reference_id=annotation.reference_id,
            target_id=annotation.target_id,
            modification_text=modification_text,
            category=annotation.category,
            group_members=(),
            ground_truth_ids=(),
            metadata=metadata,
        )
