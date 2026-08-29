from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from datasets.common import CIRSample, DirectoryImageStore


@dataclass(slots=True)
class ImageBatch:
    sample_ids: list[str]
    reference_ids: list[str]
    target_ids: list[str | None]
    modification_texts: list[str]
    categories: list[str | None]
    reference_pixels: Tensor
    target_pixels: Tensor | None
    input_ids: Tensor
    attention_mask: Tensor
    content_mask: Tensor

    def to(self, device: torch.device) -> "ImageBatch":
        return ImageBatch(
            self.sample_ids,
            self.reference_ids,
            self.target_ids,
            self.modification_texts,
            self.categories,
            self.reference_pixels.to(device, non_blocking=True),
            None
            if self.target_pixels is None
            else self.target_pixels.to(device, non_blocking=True),
            self.input_ids.to(device, non_blocking=True),
            self.attention_mask.to(device, non_blocking=True),
            self.content_mask.to(device, non_blocking=True),
        )


class FashionIQImageCollator:
    """Resolve IDs to raw images and apply the exact current FG-CLIP preprocessing."""

    def __init__(
        self,
        image_store: DirectoryImageStore,
        tokenizer: Any,
        image_processor: Any,
        max_text_length: int = 77,
        include_targets: bool = True,
    ) -> None:
        self.image_store = image_store
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_text_length = max_text_length
        self.include_targets = include_targets

    def __call__(self, samples: list[CIRSample]) -> ImageBatch:
        if not samples:
            raise ValueError("cannot collate an empty batch")
        reference_images = [self.image_store.load(sample.reference_id) for sample in samples]
        reference_pixels = self.image_processor.preprocess(reference_images, return_tensors="pt")[
            "pixel_values"
        ]
        target_pixels = None
        if self.include_targets:
            if any(sample.target_id is None for sample in samples):
                raise ValueError("target image is required for this collator")
            target_images = [self.image_store.load(str(sample.target_id)) for sample in samples]
            target_pixels = self.image_processor.preprocess(target_images, return_tensors="pt")[
                "pixel_values"
            ]
        tokenized = self.tokenizer(
            [sample.modification_text for sample in samples],
            max_length=self.max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"].to(torch.long)
        attention_mask = tokenized["attention_mask"].to(torch.bool)
        content_mask = attention_mask.clone()
        content_mask[:, 0] = False
        final_positions = attention_mask.sum(dim=1).sub(1).clamp_min(0)
        content_mask.scatter_(1, final_positions[:, None], False)
        if not content_mask.any(dim=1).all():
            raise ValueError("modification must contain at least one content token")
        return ImageBatch(
            sample_ids=[sample.sample_id for sample in samples],
            reference_ids=[sample.reference_id for sample in samples],
            target_ids=[sample.target_id for sample in samples],
            modification_texts=[sample.modification_text for sample in samples],
            categories=[sample.category for sample in samples],
            reference_pixels=reference_pixels,
            target_pixels=target_pixels,
            input_ids=input_ids,
            attention_mask=attention_mask,
            content_mask=content_mask,
        )


class ImageIdDataset(Dataset[str]):
    def __init__(self, image_ids: list[str]) -> None:
        self.image_ids = image_ids

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> str:
        return self.image_ids[index]


def collate_image_ids(
    image_ids: list[str], image_store: DirectoryImageStore, image_processor: Any
) -> tuple[list[str], Tensor]:
    images = [image_store.load(image_id) for image_id in image_ids]
    pixels = image_processor.preprocess(images, return_tensors="pt")["pixel_values"]
    return image_ids, pixels
