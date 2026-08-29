from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

@dataclass(frozen=True, slots=True)
class CIRSample:
    sample_id: str # id của sample do codebase quản lý 
    benchmark_id: str | None
    reference_id: str # id của ref image 
    target_id: str | None # id của target image
    modification_text: str # câu mô tả
    category: str | None = None # dành cho FashionIQ
    group_members: tuple[str, ...] = () # dành cho đánh giá CIRR 
    ground_truth_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict) # thông tin phụ


# 1 nhóm nhiều CIRSample
@dataclass(slots=True)
class CIRBatch:
    sample_ids: list[str]
    benchmark_ids: list[str | None]

    reference_ids: list[str]
    target_ids: list[str | None]
    modification_texts: list[str]

    categories: list[str | None]
    group_members: list[tuple[str, ...]] # chứa nhiều group của nhiều query
    ground_truth_ids: list[tuple[str, ...]]


def collate_cir_samples(samples: list[CIRSample]) -> CIRBatch:
    if not samples:
        raise ValueError("Cannot collate an empty list of CIR samples.")

    return CIRBatch(
        sample_ids=[sample.sample_id for sample in samples],
        benchmark_ids=[sample.benchmark_id for sample in samples],
        reference_ids=[sample.reference_id for sample in samples],
        target_ids=[sample.target_id for sample in samples],
        modification_texts=[
            sample.modification_text for sample in samples
        ],
        categories=[sample.category for sample in samples],
        group_members=[sample.group_members for sample in samples],
        ground_truth_ids=[
            sample.ground_truth_ids for sample in samples
        ],
    )


class ImageStore(ABC):
    @abstractmethod
    def path_for(self, image_id: str) -> Path:
        """Return the path corresponding to an image ID."""
        raise NotImplementedError

    @abstractmethod
    def load(self, image_id: str) -> Image.Image:
        """Load an image as an RGB PIL image."""
        raise NotImplementedError

@dataclass(frozen=True, slots=True)
class DirectoryImageStore(ImageStore):
    image_root: Path
    extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg")

    def path_for(self, image_id: str) -> Path:
        if not image_id:
            raise ValueError("image_id must not be empty.")

        for extension in self.extensions:
            image_path = self.image_root / f"{image_id}{extension}"

            if image_path.is_file():
                return image_path

        raise FileNotFoundError(
            f"Could not find image '{image_id}' inside "
            f"'{self.image_root}'. Tried extensions: {self.extensions}"
        )

    def load(self, image_id: str) -> Image.Image:
        image_path = self.path_for(image_id)

        with Image.open(image_path) as image:
            return image.convert("RGB")
