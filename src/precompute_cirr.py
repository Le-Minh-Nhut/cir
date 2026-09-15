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
from datasets.cirr import (
    CIRRDataset,
    build_cirr_image_store,
    load_cirr_image_mapping,
)
from datasets.common import collate_cir_samples
from teachers.csmcir import CSMCIRStage1Teacher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute CSMCIR image and text caches for CIRR."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data/CIRR"))
    parser.add_argument("--cache-root", type=Path, default=Path("features/cirr/csmcir"))
    parser.add_argument("--version", default="rc2")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test1"),
        default=("train", "val"),
    )
    parser.add_argument("--csmcir-root", type=Path, default=Path("teacher/repos/CSMCIR"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("teacher/checkpoints/csmcir/fashioniq_tuned_clip_best.pt"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_name_index(output_dir: Path, image_ids: list[str]) -> None:
    with (output_dir / "name_to_idx.json").open("w", encoding="utf-8") as file:
        json.dump({image_id: index for index, image_id in enumerate(image_ids)}, file, indent=2)


@torch.inference_mode()
def precompute_image_features(
    *,
    teacher: CSMCIRStage1Teacher,
    dataset_root: Path,
    split: str,
    version: str,
    image_ids: list[str],
    batch_size: int,
    device: torch.device,
    output_root: Path,
) -> None:
    image_store = build_cirr_image_store(
        dataset_root / "img_raw",
        dataset_root / "image_splits",
        split,
        version=version,
    )
    retrieval_dir = output_root / split / "retrieval"
    native_dir = output_root / split / "native"
    retrieval_dir.mkdir(parents=True, exist_ok=True)
    native_dir.mkdir(parents=True, exist_ok=True)

    retrieval_mmap = None
    native_mmap = None

    for start in tqdm(
        range(0, len(image_ids), batch_size),
        desc=f"CIRR CSMCIR images {split}",
        dynamic_ncols=True,
    ):
        batch_ids = image_ids[start : start + batch_size]
        images = torch.stack(
            [teacher.preprocess(image_store.load(image_id)) for image_id in batch_ids]
        ).to(device)
        retrieval_features, native_features = teacher.encode_image_tokens(images)
        retrieval_cpu = retrieval_features.float().cpu()
        native_cpu = native_features.float().cpu()

        if not torch.isfinite(retrieval_cpu).all() or not torch.isfinite(native_cpu).all():
            raise FloatingPointError(f"Non-finite CIRR image features in split={split}")

        if retrieval_mmap is None:
            retrieval_mmap = np.lib.format.open_memmap(
                retrieval_dir / "images.npy",
                mode="w+",
                dtype=np.float32,
                shape=(len(image_ids), *retrieval_cpu.shape[1:]),
            )
            native_mmap = np.lib.format.open_memmap(
                native_dir / "images.npy",
                mode="w+",
                dtype=np.float32,
                shape=(len(image_ids), *native_cpu.shape[1:]),
            )

        end = start + len(batch_ids)
        retrieval_mmap[start:end] = retrieval_cpu.numpy()
        native_mmap[start:end] = native_cpu.numpy()

    if retrieval_mmap is None or native_mmap is None:
        raise RuntimeError(f"No CIRR image features produced for split={split}")

    retrieval_mmap.flush()
    native_mmap.flush()
    _write_name_index(retrieval_dir, image_ids)
    _write_name_index(native_dir, image_ids)


@torch.inference_mode()
def precompute_text_features(
    *,
    teacher: CSMCIRStage1Teacher,
    dataset: CIRRDataset,
    split: str,
    native_features: torch.Tensor,
    native_name_to_idx: dict[str, int],
    batch_size: int,
    device: torch.device,
    output_dir: Path,
    checkpoint_hash: str,
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

    for batch in tqdm(loader, desc=f"CIRR CSMCIR text {split}", dynamic_ncols=True):
        reference_native = get_features_by_ids(
            batch.reference_ids, native_features, native_name_to_idx
        ).to(device=device, dtype=torch.float32)
        teacher_text_states, attention_mask, content_mask = teacher.encode_text_tokens(
            batch.modification_texts
        )
        text_states = teacher.encode_contextual_text_tokens(
            reference_native, teacher_text_states, attention_mask
        )

        if text_states.shape != teacher_text_states.shape:
            raise ValueError("Contextual and teacher-native CIRR text states must align")
        if attention_mask.shape != text_states.shape[:2]:
            raise ValueError("CIRR text attention mask shape mismatch")
        if content_mask.shape != text_states.shape[:2]:
            raise ValueError("CIRR text content mask shape mismatch")
        if (content_mask & ~attention_mask).any():
            raise ValueError("CIRR content mask contains positions outside the attention mask")
        if not torch.isfinite(text_states).all() or not torch.isfinite(
            teacher_text_states
        ).all():
            raise FloatingPointError(f"Non-finite CIRR text features in split={split}")

        current_batch_size = len(batch.sample_ids)
        if states_mmap is None:
            total_samples = len(dataset)
            states_mmap = np.lib.format.open_memmap(
                output_dir / "states.npy",
                mode="w+",
                dtype=np.float32,
                shape=(total_samples, *text_states.shape[1:]),
            )
            teacher_states_mmap = np.lib.format.open_memmap(
                output_dir / "teacher_states.npy",
                mode="w+",
                dtype=np.float32,
                shape=(total_samples, *teacher_text_states.shape[1:]),
            )
            attention_mmap = np.lib.format.open_memmap(
                output_dir / "attention_mask.npy",
                mode="w+",
                dtype=np.bool_,
                shape=(total_samples, *attention_mask.shape[1:]),
            )
            content_mmap = np.lib.format.open_memmap(
                output_dir / "content_mask.npy",
                mode="w+",
                dtype=np.bool_,
                shape=(total_samples, *content_mask.shape[1:]),
            )

        end = row + current_batch_size
        states_mmap[row:end] = text_states.float().cpu().numpy()
        teacher_states_mmap[row:end] = teacher_text_states.float().cpu().numpy()
        attention_mmap[row:end] = attention_mask.bool().cpu().numpy()
        content_mmap[row:end] = content_mask.bool().cpu().numpy()

        for offset, (sample_id, caption) in enumerate(
            zip(batch.sample_ids, batch.modification_texts, strict=True)
        ):
            if sample_id in sample_to_idx:
                raise ValueError(f"Duplicate CIRR sample ID: {sample_id}")
            sample_to_idx[sample_id] = row + offset
            captions[sample_id] = caption
        row = end

    if any(
        value is None
        for value in (states_mmap, teacher_states_mmap, attention_mmap, content_mmap)
    ):
        raise RuntimeError(f"No CIRR text features produced for split={split}")
    if row != len(dataset):
        raise RuntimeError(f"Expected {len(dataset)} CIRR text rows, wrote {row}")

    states_mmap.flush()
    teacher_states_mmap.flush()
    attention_mmap.flush()
    content_mmap.flush()

    with (output_dir / "sample_to_idx.json").open("w", encoding="utf-8") as file:
        json.dump(sample_to_idx, file, indent=2, ensure_ascii=False)
    with (output_dir / "captions.json").open("w", encoding="utf-8") as file:
        json.dump(captions, file, indent=2, ensure_ascii=False)
    manifest = {
        "dataset": "CIRR",
        "version": dataset.version,
        "split": split,
        "feature_kind": "taper_e2e_text",
        "num_samples": len(dataset),
        "states_shape": list(states_mmap.shape),
        "teacher_states_shape": list(teacher_states_mmap.shape),
        "attention_mask_shape": list(attention_mmap.shape),
        "content_mask_shape": list(content_mmap.shape),
        "states_dtype": "float32",
        "teacher_states_dtype": "float32",
        "mask_dtype": "bool",
        "checkpoint_sha256": checkpoint_hash,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)


def _validate_dataset_image_ids(
    dataset: CIRRDataset,
    image_mapping: dict[str, str],
) -> None:
    required: set[str] = set()
    for annotation in dataset.annotations:
        required.add(annotation.reference_id)
        required.update(annotation.group_members)
        if annotation.target_id is not None:
            required.add(annotation.target_id)
    missing = sorted(required - image_mapping.keys())
    if missing:
        raise KeyError(
            f"CIRR split mapping is missing {len(missing)} annotation image IDs: {missing[:10]}"
        )


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    dataset_root = args.dataset_root.resolve()
    cache_root = args.cache_root.resolve()
    checkpoint = args.checkpoint.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"CIRR dataset root not found: {dataset_root}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"CSMCIR checkpoint not found: {checkpoint}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    teacher = CSMCIRStage1Teacher(
        csmcir_root=args.csmcir_root,
        checkpoint_path=checkpoint,
        device=str(device),
    ).to(device).eval()
    checkpoint_hash = sha256_file(checkpoint)

    for split in args.splits:
        dataset = CIRRDataset(
            dataset_root / "captions", split, version=args.version
        )
        image_mapping = load_cirr_image_mapping(
            dataset_root / "image_splits", split, version=args.version
        )
        _validate_dataset_image_ids(dataset, image_mapping)
        image_ids = list(image_mapping)

        precompute_image_features(
            teacher=teacher,
            dataset_root=dataset_root,
            split=split,
            version=args.version,
            image_ids=image_ids,
            batch_size=args.batch_size,
            device=device,
            output_root=cache_root,
        )
        native_features, native_name_to_idx = load_features(
            cache_root / split / "native"
        )
        precompute_text_features(
            teacher=teacher,
            dataset=dataset,
            split=split,
            native_features=native_features,
            native_name_to_idx=native_name_to_idx,
            batch_size=args.batch_size,
            device=device,
            output_dir=cache_root / split / "text",
            checkpoint_hash=checkpoint_hash,
        )

    print(f"CIRR caches saved under: {cache_root}")


if __name__ == "__main__":
    main()
