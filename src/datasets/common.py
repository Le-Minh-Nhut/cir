from dataclasses import dataclass, field
#Protocol ở đây chỉ định rằng bất kỳ class nào có đủ hai hàm -> thì đều có thể được xem như một ImageStore
from typing import Any, Protocol 
from pathlib import Path
from abc import ABC, abstractmethod
import random
import string
from collections.abc import Mapping, Sequence
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
class DirectoryImageStore (ImageStore):
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


def normalize_fashioniq_caption(caption: str, correction_dict: Mapping[str, str]) -> str:

    # Tạo bảng thay toàn bộ dấu câu thành khoảng trắng.
    punctuation_table = str.maketrans(
        {character: " " for character in string.punctuation}
    )

    # Lowercase, bỏ dấu câu rồi tách thành từng từ.
    tokens = (
        caption
        .lower()
        .translate(punctuation_table)
        .strip()
        .split()
    )

    # Sửa từng từ nếu nó xuất hiện trong correction_dict.
    corrected_tokens = [
        correction_dict.get(token, token)
        for token in tokens
    ]

    return " ".join(corrected_tokens)

def compose_fashioniq_caption(captions: Sequence[str], policy: str = "ordered_and", correction_dict: Mapping[str, str] | None = None, rng: random.Random | None = None) -> str:
    """
    Convert the two FashionIQ captions into one modification text.

    Policies:
        ordered_and:
            QuRe-style deterministic composition.

        normalized_ordered_and:
            ConeSep/Air-Know/HABIT/INTENT-style normalization.

        randomized_four_way:
            TME-style training augmentation.
    """

    if len(captions) != 2:
        raise ValueError(
            "FashionIQ samples must contain exactly two captions."
        )

    first_caption = captions[0].strip(".?, ")
    second_caption = captions[1].strip(".?, ")

    if not first_caption or not second_caption:
        raise ValueError(
            "FashionIQ captions must not be empty."
        )

    if policy == "ordered_and":
        return (
            f"{first_caption.capitalize()} and "
            f"{second_caption}"
        )

    if policy == "normalized_ordered_and":
        if correction_dict is None:
            raise ValueError(
                "normalized_ordered_and requires a correction_dict."
            )

        normalized_first = normalize_fashioniq_caption(
            first_caption,
            correction_dict,
        )
        normalized_second = normalize_fashioniq_caption(
            second_caption,
            correction_dict,
        )

        return (
            f"{normalized_first} and "
            f"{normalized_second}"
        )

    if policy == "randomized_four_way":
        random_generator = rng if rng is not None else random
        random_number = random_generator.random()

        if random_number < 0.25:
            return (
                f"{first_caption.capitalize()} and "
                f"{second_caption}"
            )

        if random_number < 0.5:
            return (
                f"{second_caption.capitalize()} and "
                f"{first_caption}"
            )

        if random_number < 0.75:
            return first_caption.capitalize()

        return second_caption.capitalize()

    raise ValueError(
        f"Unsupported FashionIQ caption policy: '{policy}'."
    )


def expand_fashioniq_captions_bidirectionally(captions: Sequence[str], policy: str = "ordered_and", correction_dict: Mapping[str, str] | None = None) -> tuple[str, str]:
    """
    Generate both caption orders for one FashionIQ annotation.

    Example:
        [cap1, cap2]
        -> (
            "cap1 and cap2",
            "cap2 and cap1",
        )

    This function only generates the two texts. The dataset is
    responsible for exposing them as two separate samples.
    """

    if len(captions) != 2:
        raise ValueError(
            "FashionIQ samples must contain exactly two captions."
        )

    if policy not in {
        "ordered_and",
        "normalized_ordered_and",
    }:
        raise ValueError(
            "Bidirectional expansion only supports "
            "'ordered_and' and 'normalized_ordered_and'."
        )

    forward = compose_fashioniq_caption(
        captions=captions,
        policy=policy,
        correction_dict=correction_dict,
    )

    reverse = compose_fashioniq_caption(
        captions=(captions[1], captions[0]),
        policy=policy,
        correction_dict=correction_dict,
    )

    return forward, reverse