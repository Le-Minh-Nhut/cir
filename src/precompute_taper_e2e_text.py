from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cache.features import get_features_by_ids, load_features
from datasets.common import collate_cir_samples
from datasets.fashioniq import FashionIQDataset, load_correction_dict
from teachers.csmcir import CSMCIRStage1Teacher


CATEGORIES = ("dress", "shirt", "toptee")
CAPTION_POLICY = "normalized_ordered_and"


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute frozen CSMCIR text features for TAPER Competitive-NULL E2E training.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/FashionIQ"),)
    parser.add_argument("--cache-root", type=Path, default=Path("features/fashioniq/csmcir"),)
    parser.add_argument("--csmcir-root", type=Path, default=Path("teacher/repos/CSMCIR"))
    parser.add_argument("--checkpoint", type=Path,default=Path("teacher/checkpoints/csmcir/fashioniq_tuned_clip_best.pt"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def load_correction_dicts(annotation_root: Path) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    correction_dicts: dict[str, dict[str, str]] = {}
    correction_hashes: dict[str, str] = {}

    for category in CATEGORIES:
        path = annotation_root / f"correction_dict_{category}.json"

        if not path.is_file():
            raise FileNotFoundError(f"Missing FashionIQ correction dictionary: {path}")

        correction_dicts[category] = load_correction_dict(path)
        correction_hashes[category] = sha256_file(path)

    return correction_dicts, correction_hashes


def build_dataset(
    *,
    annotation_root: Path,
    split: str,
    correction_dicts: dict[str, dict[str, str]],
) -> FashionIQDataset:
    return FashionIQDataset(
        annotation_root=annotation_root,
        split=split,
        categories=CATEGORIES,
        caption_policy=CAPTION_POLICY,
        correction_dicts=correction_dicts,
        seed=42,
    )


def _validate_batch_outputs(
    *,
    text_states: torch.Tensor,
    teacher_text_states: torch.Tensor,
    attention_mask: torch.Tensor,
    content_mask: torch.Tensor,
) -> None:
    if teacher_text_states.ndim != 3:
        raise ValueError(f"teacher_text_states must be [B,N,D], got {tuple(teacher_text_states.shape)}")

    if text_states.ndim != 3:
        raise ValueError(f"text_states must be [B,N,D], got {tuple(text_states.shape)}")

    if text_states.shape[:2] != teacher_text_states.shape[:2]:
        raise ValueError(
            "Contextual and teacher-native text states "
            "must have the same [B,N] dimensions: "
            f"{tuple(text_states.shape)} vs "
            f"{tuple(teacher_text_states.shape)}"
        )

    expected_mask_shape = teacher_text_states.shape[:2]

    if attention_mask.shape != expected_mask_shape:
        raise ValueError(
            "attention_mask shape mismatch: "
            f"expected {tuple(expected_mask_shape)}, "
            f"got {tuple(attention_mask.shape)}"
        )

    if content_mask.shape != expected_mask_shape:
        raise ValueError(
            "content_mask shape mismatch: "
            f"expected {tuple(expected_mask_shape)}, "
            f"got {tuple(content_mask.shape)}"
        )

    if not torch.isfinite(text_states).all():
        raise FloatingPointError("Contextual text states contain NaN/Inf")

    if not torch.isfinite(teacher_text_states).all():
        raise FloatingPointError("Teacher-native text states contain NaN/Inf")

    if (content_mask & ~attention_mask).any():
        raise ValueError("content_mask contains positions outside attention_mask")


@torch.inference_mode()
def precompute_split(
    *,
    split: str,
    teacher: CSMCIRStage1Teacher,
    dataset: FashionIQDataset,
    native_features: torch.Tensor,
    native_name_to_idx: dict[str, int],
    batch_size: int,
    device: torch.device,
    output_dir: Path,
    checkpoint_hash: str,
    correction_hashes: dict[str, str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_cir_samples,
        pin_memory=device.type == "cuda",
    )

    states_mmap = None
    teacher_states_mmap = None
    attention_mmap = None
    content_mmap = None

    sample_to_idx: dict[str, int] = {}
    captions: dict[str, str] = {}

    row = 0

    for batch in tqdm(loader, desc=f"CSMCIR text {split}", dynamic_ncols=True,):
        current_batch_size = len(batch.sample_ids)

        if current_batch_size == 0:
            raise RuntimeError("Encountered empty batch")

        reference_native = get_features_by_ids(batch.reference_ids, native_features, native_name_to_idx,).to(device=device, dtype=torch.float32, non_blocking=True,)
        if reference_native.ndim != 3:
            raise ValueError(f"Native reference features must be [B,K,D], got {tuple(reference_native.shape)}")

        if not torch.isfinite(reference_native).all():
            raise FloatingPointError(f"Non-finite native reference features in {split}")

        (
            teacher_text_states,
            attention_mask,
            content_mask,
        ) = teacher.encode_text_tokens(
            batch.modification_texts
        )

        text_states = teacher.encode_contextual_text_tokens(
            reference_native,
            teacher_text_states,
            attention_mask,
        )

        _validate_batch_outputs(
            text_states=text_states,
            teacher_text_states=teacher_text_states,
            attention_mask=attention_mask,
            content_mask=content_mask,
        )

        if text_states.shape[0] != current_batch_size:
            raise ValueError("Contextual text batch size mismatch")

        if teacher_text_states.shape[0] != current_batch_size:
            raise ValueError("Teacher text batch size mismatch")

        if states_mmap is None:
            total_samples = len(dataset)

            num_tokens = text_states.shape[1]
            text_dim = text_states.shape[2]
            teacher_text_dim = teacher_text_states.shape[2]

            states_mmap = np.lib.format.open_memmap(
                output_dir / "states.npy",
                mode="w+",
                dtype=np.float32,
                shape=(total_samples, num_tokens, text_dim),
            )

            teacher_states_mmap = np.lib.format.open_memmap(
                output_dir / "teacher_states.npy",
                mode="w+",
                dtype=np.float32,
                shape=(total_samples, num_tokens, teacher_text_dim),
            )

            attention_mmap = np.lib.format.open_memmap(
                output_dir / "attention_mask.npy",
                mode="w+",
                dtype=np.bool_,
                shape=(total_samples, num_tokens),
            )

            content_mmap = np.lib.format.open_memmap(
                output_dir / "content_mask.npy",
                mode="w+",
                dtype=np.bool_,
                shape=(total_samples, num_tokens),
            )

        assert states_mmap is not None
        assert teacher_states_mmap is not None
        assert attention_mmap is not None
        assert content_mmap is not None

        end = row + current_batch_size

        if end > len(dataset):
            raise RuntimeError("Attempted to write beyond dataset size")

        states_cpu = text_states.detach().float().cpu().numpy()
        teacher_states_cpu = teacher_text_states.detach().float().cpu().numpy()
        attention_cpu = attention_mask.detach().bool().cpu().numpy()
        content_cpu = content_mask.detach().bool().cpu().numpy()
        states_mmap[row:end] = states_cpu
        teacher_states_mmap[row:end] = teacher_states_cpu
        attention_mmap[row:end] = attention_cpu
        content_mmap[row:end] = content_cpu

        for offset, (sample_id, caption) in enumerate(zip(batch.sample_ids, batch.modification_texts, strict=True)):
            if sample_id in sample_to_idx:
                raise ValueError(f"Duplicate sample_id encountered: {sample_id}")

            cache_row = row + offset
            sample_to_idx[sample_id] = cache_row
            captions[sample_id] = caption

        row = end

        del reference_native
        del teacher_text_states
        del text_states
        del attention_mask
        del content_mask

    if states_mmap is None:
        raise RuntimeError(f"No text features produced for split={split}")

    if row != len(dataset):
        raise RuntimeError(f"Expected {len(dataset)} cached rows, wrote {row}")

    if len(sample_to_idx) != len(dataset):
        raise RuntimeError(f"sample_to_idx size does not match dataset size: {len(sample_to_idx)} != {len(dataset)}")

    if len(captions) != len(dataset):
        raise RuntimeError("caption index size does not match dataset size")

    states_mmap.flush()
    teacher_states_mmap.flush()
    attention_mmap.flush()
    content_mmap.flush()

    states_shape = list(states_mmap.shape)
    teacher_states_shape = list(teacher_states_mmap.shape)
    attention_shape = list(attention_mmap.shape)
    content_shape = list(content_mmap.shape)

    with (output_dir / "sample_to_idx.json").open("w", encoding="utf-8") as file:
        json.dump(sample_to_idx, file, indent=2, ensure_ascii=False)

    with (output_dir / "captions.json").open("w", encoding="utf-8",) as file:
        json.dump(captions, file, indent=2, ensure_ascii=False)

    manifest = {
        "dataset": "FashionIQ",
        "split": split,
        "feature_kind": "taper_e2e_text",
        "caption_policy": CAPTION_POLICY,
        "num_samples": len(dataset),
        "states_shape": states_shape,
        "teacher_states_shape": teacher_states_shape,
        "attention_mask_shape": attention_shape,
        "content_mask_shape": content_shape,
        "states_dtype": "float32",
        "teacher_states_dtype": "float32",
        "mask_dtype": "bool",
        "checkpoint_sha256": checkpoint_hash,
        "correction_dict_sha256": correction_hashes,
        "categories": list(CATEGORIES),
    }

    with (output_dir / "manifest.json").open("w", encoding="utf-8",) as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)

    del states_mmap
    del teacher_states_mmap
    del attention_mmap
    del content_mmap

    print()
    print(f"[{split}] DONE")
    print("Samples:", len(dataset))
    print("Saved:", output_dir)
    print("states:", states_shape)
    print("teacher_states:", teacher_states_shape)
    print("attention_mask:", attention_shape)
    print("content_mask:", content_shape)


def main():
    args = parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    dataset_root = args.dataset_root.resolve()
    cache_root = args.cache_root.resolve()
    csmcir_root = args.csmcir_root.resolve()
    checkpoint = args.checkpoint.resolve()

    annotation_root = dataset_root / "captions"

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"FashionIQ dataset root not found: {dataset_root}")

    if not annotation_root.is_dir():
        raise FileNotFoundError(f"FashionIQ captions directory not found: {annotation_root}")

    if not checkpoint.is_file():
        raise FileNotFoundError(f"CSMCIR checkpoint not found: {checkpoint}")

    correction_dicts, correction_hashes = load_correction_dicts(annotation_root)
    checkpoint_hash = sha256_file(checkpoint)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    print("Device:", device)
    print("Dataset:", dataset_root)
    print("Cache root:", cache_root)
    print("Caption policy:", CAPTION_POLICY)
    print("Checkpoint SHA256:", checkpoint_hash)

    teacher = CSMCIRStage1Teacher(csmcir_root=csmcir_root, checkpoint_path=checkpoint, device=str(device)).to(device).eval()

    for parameter in teacher.parameters():
        if parameter.requires_grad:
            raise RuntimeError("Teacher contains trainable parameters during frozen text precompute")

    for split in ("train", "val"):
        print()
        print("=" * 80)
        print(f"PRECOMPUTE SPLIT: {split}")
        print("=" * 80)

        dataset = build_dataset(
            annotation_root=annotation_root,
            split=split,
            correction_dicts=correction_dicts,
        )

        native_dir = cache_root / split / "native"
        native_features, native_name_to_idx = load_features(native_dir)
        if native_features.ndim != 3:
            raise ValueError(f"Native cache must contain [N,K,D] features, got {tuple(native_features.shape)}")
        missing_reference_ids = sorted(
            {
                sample.reference_id
                for sample in dataset
                if sample.reference_id
                not in native_name_to_idx
            }
        )

        if missing_reference_ids:
            preview = missing_reference_ids[:10]
            raise KeyError(
                "Native reference cache is missing "
                f"{len(missing_reference_ids)} image IDs. "
                f"First entries: {preview}"
            )

        output_dir = cache_root / split / "text"
        precompute_split(
            split=split,
            teacher=teacher,
            dataset=dataset,
            native_features=native_features,
            native_name_to_idx=native_name_to_idx,
            batch_size=args.batch_size,
            device=device,
            output_dir=output_dir,
            checkpoint_hash=checkpoint_hash,
            correction_hashes=correction_hashes,
        )

    print()
    print("=" * 80)
    print("ALL TEXT PRECOMPUTE COMPLETE")
    print("=" * 80)
    print("Saved under:")
    print(cache_root)


if __name__ == "__main__":
    main()