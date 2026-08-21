from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cache.features import get_features_by_ids, load_features
from datasets.common import collate_cir_samples
from datasets.fashioniq import FashionIQDataset, load_correction_dict
from teachers.csmcir import CSMCIRStage1Teacher


CATEGORIES = ("dress", "shirt", "toptee")
CAPTION_POLICY = "normalized_ordered_and"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/FashionIQ"),
    )

    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("features/fashioniq/csmcir"),
    )

    parser.add_argument(
        "--csmcir-root",
        type=Path,
        default=Path("teacher/repos/CSMCIR"),
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "teacher/checkpoints/csmcir/fashioniq_tuned_clip_best.pt"
        ),
    )

    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="train",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )

    return parser.parse_args()


def load_correction_dicts(annotation_root: Path):
    correction_dicts = {}

    for category in CATEGORIES:
        path = annotation_root / f"correction_dict_{category}.json"

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing correction dictionary: {path}"
            )

        correction_dicts[category] = load_correction_dict(path)

    return correction_dicts


def load_text_cache(cache_dir: Path):
    states = np.load(
        cache_dir / "states.npy",
        mmap_mode="r",
    )

    teacher_states = np.load(
        cache_dir / "teacher_states.npy",
        mmap_mode="r",
    )

    attention_mask = np.load(
        cache_dir / "attention_mask.npy",
        mmap_mode="r",
    )

    content_mask = np.load(
        cache_dir / "content_mask.npy",
        mmap_mode="r",
    )

    with (
        cache_dir / "sample_to_idx.json"
    ).open("r", encoding="utf-8") as file:
        sample_to_idx = json.load(file)

    with (
        cache_dir / "captions.json"
    ).open("r", encoding="utf-8") as file:
        captions = json.load(file)

    return {
        "states": states,
        "teacher_states": teacher_states,
        "attention_mask": attention_mask,
        "content_mask": content_mask,
        "sample_to_idx": sample_to_idx,
        "captions": captions,
    }


def max_abs_diff(
    a: torch.Tensor,
    b: torch.Tensor,
) -> float:
    return (
        (a.float() - b.float())
        .abs()
        .max()
        .item()
    )


@torch.inference_mode()
def main():
    args = parse_args()

    if args.num_samples < 1:
        raise ValueError(
            "--num-samples must be >= 1"
        )

    if args.batch_size < 1:
        raise ValueError(
            "--batch-size must be >= 1"
        )

    device = torch.device(args.device)

    dataset_root = args.dataset_root.resolve()
    annotation_root = dataset_root / "captions"

    correction_dicts = load_correction_dicts(
        annotation_root
    )

    dataset = FashionIQDataset(
        annotation_root=annotation_root,
        split=args.split,
        categories=CATEGORIES,
        caption_policy=CAPTION_POLICY,
        correction_dicts=correction_dicts,
        seed=42,
    )

    num_samples = min(
        args.num_samples,
        len(dataset),
    )

    # Important:
    # take the first N samples deterministically.
    subset = torch.utils.data.Subset(
        dataset,
        list(range(num_samples)),
    )

    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_cir_samples,
    )

    cache_root = args.cache_root.resolve()

    native_features, native_name_to_idx = (
        load_features(
            cache_root
            / args.split
            / "native"
        )
    )

    text_cache = load_text_cache(
        cache_root
        / args.split
        / "text"
    )

    teacher = CSMCIRStage1Teacher(
        csmcir_root=args.csmcir_root,
        checkpoint_path=args.checkpoint,
        device=str(device),
    ).to(device).eval()

    contextual_max_diff = 0.0
    teacher_max_diff = 0.0

    checked = 0

    for batch in loader:
        # --------------------------------------------------
        # 1. Recompute LIVE features
        # --------------------------------------------------

        reference_native = get_features_by_ids(
            batch.reference_ids,
            native_features,
            native_name_to_idx,
        ).to(
            device=device,
            dtype=torch.float32,
        )

        (
            live_teacher_states,
            live_attention_mask,
            live_content_mask,
        ) = teacher.encode_text_tokens(
            batch.modification_texts
        )

        live_states = (
            teacher.encode_contextual_text_tokens(
                reference_native,
                live_teacher_states,
                live_attention_mask,
            )
        )

        # --------------------------------------------------
        # 2. Resolve CACHE rows
        # --------------------------------------------------

        rows = []

        for sample_id, runtime_caption in zip(
            batch.sample_ids,
            batch.modification_texts,
            strict=True,
        ):
            if sample_id not in text_cache["sample_to_idx"]:
                raise KeyError(
                    f"Missing sample_id from text cache: "
                    f"{sample_id}"
                )

            cached_caption = (
                text_cache["captions"][sample_id]
            )

            if runtime_caption != cached_caption:
                raise RuntimeError(
                    "Caption mismatch for "
                    f"{sample_id}\n"
                    f"runtime: {runtime_caption!r}\n"
                    f"cache:   {cached_caption!r}"
                )

            rows.append(
                text_cache["sample_to_idx"][sample_id]
            )

        cached_states = torch.from_numpy(
            np.asarray(
                text_cache["states"][rows]
            ).copy()
        ).to(
            device=device,
            dtype=torch.float32,
        )

        cached_teacher_states = torch.from_numpy(
            np.asarray(
                text_cache["teacher_states"][rows]
            ).copy()
        ).to(
            device=device,
            dtype=torch.float32,
        )

        cached_attention_mask = torch.from_numpy(
            np.asarray(
                text_cache["attention_mask"][rows]
            ).copy()
        ).to(
            device=device,
            dtype=torch.bool,
        )

        cached_content_mask = torch.from_numpy(
            np.asarray(
                text_cache["content_mask"][rows]
            ).copy()
        ).to(
            device=device,
            dtype=torch.bool,
        )

        # --------------------------------------------------
        # 3. Shape checks
        # --------------------------------------------------

        if live_states.shape != cached_states.shape:
            raise AssertionError(
                "Contextual states shape mismatch: "
                f"{tuple(live_states.shape)} vs "
                f"{tuple(cached_states.shape)}"
            )

        if (
            live_teacher_states.shape
            != cached_teacher_states.shape
        ):
            raise AssertionError(
                "Teacher states shape mismatch"
            )

        # --------------------------------------------------
        # 4. Mask equality — MUST be exact
        # --------------------------------------------------

        if not torch.equal(
            live_attention_mask,
            cached_attention_mask,
        ):
            raise AssertionError(
                "attention_mask LIVE != CACHE"
            )

        if not torch.equal(
            live_content_mask,
            cached_content_mask,
        ):
            raise AssertionError(
                "content_mask LIVE != CACHE"
            )

        # --------------------------------------------------
        # 5. Numerical feature parity
        # --------------------------------------------------

        contextual_diff = max_abs_diff(
            live_states,
            cached_states,
        )

        teacher_diff = max_abs_diff(
            live_teacher_states,
            cached_teacher_states,
        )

        contextual_max_diff = max(
            contextual_max_diff,
            contextual_diff,
        )

        teacher_max_diff = max(
            teacher_max_diff,
            teacher_diff,
        )

        checked += len(batch.sample_ids)

        print(
            f"checked={checked:4d} | "
            f"contextual_max_diff={contextual_diff:.8e} | "
            f"teacher_max_diff={teacher_diff:.8e}"
        )

    print()
    print("=" * 80)
    print("TEXT CACHE PARITY RESULT")
    print("=" * 80)

    print("split:", args.split)
    print("samples checked:", checked)

    print(
        "contextual global max abs diff:",
        f"{contextual_max_diff:.8e}",
    )

    print(
        "teacher global max abs diff:",
        f"{teacher_max_diff:.8e}",
    )

    # Cache was written as float32.
    # With the same frozen model/checkpoint/input contract,
    # this should normally be extremely close.
    tolerance = 1e-5

    if contextual_max_diff > tolerance:
        raise AssertionError(
            "Contextual text cache parity FAILED: "
            f"{contextual_max_diff} > {tolerance}"
        )

    if teacher_max_diff > tolerance:
        raise AssertionError(
            "Teacher text cache parity FAILED: "
            f"{teacher_max_diff} > {tolerance}"
        )

    print()
    print("PASS")
    print("LIVE text features == cached text features")
    print("Masks are exactly identical.")


if __name__ == "__main__":
    main()