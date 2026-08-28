from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import random
import string
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from torch.utils.data import Dataset
from datasets.common import CIRSample


VALID_CATEGORIES = {"dress", "shirt", "toptee"}
VALID_SPLITS = {"train", "val", "test"}
VALID_CAPTION_POLICIES = {"ordered_and", "normalized_ordered_and", "randomized_four_way"}
CORRECTION_POLICIES = {"fashioniq", "none"}
REQUIRED_CORRECTION_DICTIONARIES = tuple(
    f"correction_dict_{category}.json" for category in ("dress", "shirt", "toptee")
)


def validate_correction_policy(policy: str) -> str:
    if policy not in CORRECTION_POLICIES:
        raise ValueError(
            f"Unsupported FashionIQ correction policy {policy!r}; "
            f"expected one of {sorted(CORRECTION_POLICIES)}"
        )
    return policy


def resolve_fashioniq_correction_dicts(
    annotation_root: str | Path,
    policy: str,
) -> dict[str, dict[str, str]] | None:
    """Resolve the audited dictionaries, never silently changing text protocol."""
    validated = validate_correction_policy(policy)
    if validated == "none":
        return None
    root = Path(annotation_root).resolve()
    missing = [name for name in REQUIRED_CORRECTION_DICTIONARIES if not (root / name).is_file()]
    if missing:
        required = "\n".join(f"  {name}" for name in REQUIRED_CORRECTION_DICTIONARIES)
        raise FileNotFoundError(
            "FashionIQ correction_policy=fashioniq requires:\n"
            f"{required}\n\n"
            f"Expected under: {root}\n\n"
            "Either supply the audited dictionaries or explicitly run a control\n"
            "with correction_policy=none."
        )
    return {
        category: load_correction_dict(root / f"correction_dict_{category}.json")
        for category in ("dress", "shirt", "toptee")
    }


@dataclass(frozen=True, slots=True)
class FashionIQAnnotation:
    reference_id: str
    target_id: str | None
    captions: tuple[str, str]
    category: str
    index: int


def _read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)

# đọc dictionary sửa lỗi từ ENCODER và giữ nguyên nội dung
def load_correction_dict(path: str | Path) -> dict[str, str]:
    """
    Load an ENCODER-style FashionIQ correction dictionary unchanged.

    Important:
    ENCODER lowercases the caption text before dictionary lookup,
    but loads the correction dictionary itself directly from JSON.
    We therefore do not lowercase or otherwise mutate the dictionary here.
    """
    data = _read_json(path)
    correction_dict: dict[str, str] = {}

    for key, value in data.items():
        correction_dict[key] = value

    return correction_dict


def load_fashioniq_annotations(annotation_root: str | Path, split: str, category: str) -> list[FashionIQAnnotation]:
    assert split in VALID_SPLITS
    assert category in VALID_CATEGORIES

    path = Path(annotation_root) / f"cap.{category}.{split}.json"
    records = _read_json(path)
    annotations = []
    for index, record in enumerate(records):
        reference_id = record["candidate"]
        target_id = record.get("target")
        captions = record["captions"]

        # FashionIQ composition assumes exactly two relative captions.
        assert len(captions) == 2

        # Train/validation retrieval needs a known positive target.
        if split in {"train", "val"}: 
            assert target_id is not None

        annotation = FashionIQAnnotation(
            reference_id=reference_id,
            target_id=target_id,
            captions=(captions[0], captions[1]),
            category=category,
            index=index,
        )

        annotations.append(annotation)

    return annotations


def load_fashioniq_split_ids(split_root: str | Path, split: str, category: str) -> list[str]:
    assert split in VALID_SPLITS
    assert category in VALID_CATEGORIES
    path = Path(split_root)/ f"split.{category}.{split}.json"
    image_ids = _read_json(path)

    return image_ids # -> trả về danh sách ids của các ảnh 


def normalize_fashioniq_caption(caption: str, correction_dict: Mapping[str, str] | None = None) -> str:
    """
    Reproduce ENCODER's FashionIQ text normalization:

        lowercase
        -> replace every string.punctuation character with a space
        -> strip
        -> split
        -> token-wise correction dictionary lookup
        -> join with a single space
    """
    punctuation_map = {}
    # string.punctuation là 1 chuỗi chứa các dấu: "!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
    for character in string.punctuation:
        punctuation_map[character] = " "
    punctuation_to_space = str.maketrans(punctuation_map) # -> biến dict đó thành như 1 bản dịch 

    tokens = (str(caption).lower().translate(punctuation_to_space).strip().split()) # -> lower -> xóa dấu -> xóa khoảng trắng 2 đầu -> tách thành list
    corrected_tokens: list[str] = []
    for token in tokens:
        if correction_dict is not None and token in correction_dict:
            corrected_token = correction_dict[token] # -> sửa chữ sai chính tả 
        else:
            corrected_token = token 
        corrected_tokens.append(corrected_token)
    return " ".join(corrected_tokens)

def compose_fashioniq_caption(captions: tuple[str, str], policy: str, correction_dict: Mapping[str, str] | None = None, rng: random.Random | None = None) -> str:
    """
    Compose the two original FashionIQ captions according to one
    audited research policy.

    Policies
    --------
    ordered_and
        QuRe-style deterministic composition.
        This also matches the deterministic caption construction
        used by TME during FashionIQ evaluation.

    normalized_ordered_and
        ENCODER-style normalization followed by:
            normalized_cap1 + " and " + normalized_cap2

    randomized_four_way
        TME-style training augmentation:
            cap1 and cap2
            cap2 and cap1
            cap1
            cap2

        The string formatting matches the TME source. The RNG schedule
        in this project is intentionally more deterministic than TME:
        a local RNG is derived from (seed, epoch, sample_id).
    """
    assert policy in VALID_CAPTION_POLICIES

    caption_1 = captions[0]
    caption_2 = captions[1]

    if policy == "ordered_and":
        # Exact QuRe / TME-evaluation string construction.
        caption_1 = caption_1.strip(".?, ").capitalize()
        caption_2 = caption_2.strip(".?, ")

        return f"{caption_1} and {caption_2}"

    if policy == "normalized_ordered_and":
        caption_1 = normalize_fashioniq_caption(caption_1, correction_dict)
        caption_2 = normalize_fashioniq_caption(caption_2, correction_dict)

        return f"{caption_1} and {caption_2}"

    # randomized_four_way
    assert rng is not None
    random_number = rng.random()

    # These branch boundaries intentionally mirror the TME source.
    if random_number < 0.25:
        caption_1 = caption_1.strip(".?, ").capitalize()
        caption_2 = caption_2.strip(".?, ")

        return f"{caption_1} and {caption_2}"

    if 0.25 < random_number < 0.50:
        caption_2 = caption_2.strip(".?, ").capitalize()
        caption_1 = caption_1.strip(".?, ")

        return f"{caption_2} and {caption_1}"

    if 0.50 < random_number < 0.75:
        return caption_1.strip(".?, ").capitalize()
    return caption_2.strip(".?, ").capitalize()


def build_pair_union_gallery(annotations: Sequence[FashionIQAnnotation]) -> list[str]:
    """
    Build the reduced FashionIQ validation gallery as the ordered,
    duplicate-free union of reference and target IDs.

    This mirrors the logical gallery construction used by ENCODER's
    val-split evaluation, while keeping protocol selection outside
    the Dataset itself.
    """
    gallery_ids: list[str] = []
    seen: set[str] = set()

    for annotation in annotations:
        assert annotation.target_id is not None

        if annotation.reference_id not in seen:
            seen.add(annotation.reference_id)
            gallery_ids.append(annotation.reference_id)

        if annotation.target_id not in seen:
            seen.add(annotation.target_id)
            gallery_ids.append(annotation.target_id)

    return gallery_ids


class FashionIQDataset(Dataset):
    """
    FashionIQ annotation/query Dataset.

    The Dataset returns stable IDs + composed modification text through
    CIRSample. Raw image loading, model-specific preprocessing, and
    FeatureCache resolution intentionally live outside this Dataset.
    """
    def __init__(
        self,
        annotation_root: str | Path, # -> folder chứa cap.dress.train.json, cap.shirt.val.json,...
        split: str, # -> train / val / test
        categories: Sequence[str], # -> category muốn load
        caption_policy: str = "ordered_and", # -> policy hình thành caption
        correction_dicts: ( # -> mảng đúng chính tả 
            Mapping[
                str,
                Mapping[str, str],
            ]
            | None
        ) = None,
        seed: int = 42, #-> dùng cho randomized_four_way
    ) -> None:
        super().__init__()

        assert split in VALID_SPLITS
        assert categories
        assert caption_policy in VALID_CAPTION_POLICIES

        for category in categories:
            assert category in VALID_CATEGORIES

        # TME's four-way caption randomization is a training behavior.
        # Standard FashionIQ evaluation uses deterministic ordered captions.
        if (caption_policy == "randomized_four_way"):
            assert split == "train"

        self.annotation_root = Path(annotation_root)
        self.split = split
        self.categories = list(categories)
        self.caption_policy = caption_policy
        self.seed = seed
        self._epoch = mp.Value("q", 0, lock=True,) # -> tạo một biến epoch dùng chung giữa các process
        self.correction_dicts: dict[str, dict[str, str]] = {}

        if correction_dicts is not None:
            for category, correction_dict in correction_dicts.items():
                self.correction_dicts[category] = dict(correction_dict)

        if self.caption_policy == "normalized_ordered_and" and correction_dicts is not None:
            for category in self.categories:
                assert category in self.correction_dicts

        self.annotations: list[FashionIQAnnotation] = []

        for category in self.categories:
            category_annotations = (
                load_fashioniq_annotations(
                    annotation_root=(self.annotation_root),
                    split=self.split,
                    category=category,
                )
            )
            self.annotations.extend(category_annotations)

    @property # -> cho phép gọi epoch như một biến dù thực chất bên trong nó là một function
    def epoch(self) -> int:
        return int(self._epoch.value)

    def set_epoch(self, epoch: int,) -> None:
        self._epoch.value = epoch

    def __len__(self) -> int:
        return len(self.annotations)

    def _make_rng(self, sample_id: str) -> random.Random:
        """
        Build a deterministic local RNG from:
            global seed + epoch + stable sample ID.

        This intentionally differs from TME's single global RNG stream.
        It preserves the same four-way augmentation distribution while
        making caption choice independent of DataLoader worker scheduling.
        """
        key = f"{self.seed}:{self.epoch}:{sample_id}"
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        sample_seed = int.from_bytes(digest[:8], byteorder="big")

        return random.Random(sample_seed)

    def _compose_text(self, annotation: FashionIQAnnotation, sample_id: str) -> str:
        correction_dict = self.correction_dicts.get(annotation.category)
        rng = None
        if (self.caption_policy == "randomized_four_way"):
            rng = self._make_rng(sample_id)

        return compose_fashioniq_caption(
            captions=annotation.captions,
            policy=self.caption_policy,
            correction_dict=correction_dict,
            rng=rng,
        )

    def __getitem__(self, index: int) -> CIRSample:
        annotation = self.annotations[index]

        # Keep sample_id as a stable codebase-owned query identity.
        # The old _make_sample_id() helper is unnecessary now that
        # bidirectional forward/reverse expansion has been removed.
        sample_id = f"fashioniq:{self.split}:{annotation.category}:{annotation.index}"
        modification_text = self._compose_text(annotation=annotation, sample_id=sample_id)
        return CIRSample(
            sample_id=sample_id,
            benchmark_id=None,
            reference_id=(annotation.reference_id),
            target_id=(annotation.target_id),
            modification_text=(modification_text),
            category=(annotation.category),
            group_members=(),
            ground_truth_ids=(),
            metadata={
                "original_captions": annotation.captions,
                "caption_policy": self.caption_policy,
            },
        )
