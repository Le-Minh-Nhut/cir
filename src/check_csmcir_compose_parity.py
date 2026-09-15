from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cache.features import load_features
from teachers.csmcir import CSMCIRStage1Teacher
from teachers.csmcir_compose import CSMCIRComposeTeacher


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=Path("features/fashioniq/csmcir"))
    parser.add_argument("--csmcir-root", type=Path, default=Path("teacher/repos/CSMCIR"),)
    parser.add_argument("--checkpoint", type=Path, default=Path("teacher/checkpoints/csmcir/fashioniq_tuned_clip_best.pt"))
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--batch-size", type=int, default=2)

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cpu")
    cache_root = args.cache_root.resolve() / args.split
    native_features, _ = load_features(cache_root / "native")
    text_root = cache_root / "text"
    teacher_states = torch.from_numpy(np.load(text_root / "teacher_states.npy", mmap_mode="c"))
    attention_mask = torch.from_numpy(np.load(text_root / "attention_mask.npy", mmap_mode="c"))

    with (text_root / "sample_to_idx.json").open("r", encoding="utf-8") as file:
        sample_to_idx = json.load(file)

    sample_ids = list(sample_to_idx)[: args.batch_size]
    rows = [
        sample_to_idx[sample_id]
        for sample_id in sample_ids
    ]

    from datasets.fashioniq import FashionIQDataset, load_correction_dict
    dataset_root = Path("data/FashionIQ")
    annotation_root = dataset_root / "captions"
    categories = ("dress", "shirt", "toptee")

    corrections = {
        category: load_correction_dict(annotation_root / f"correction_dict_{category}.json")
        for category in categories
    }

    dataset = FashionIQDataset(
        annotation_root=annotation_root,
        split=args.split,
        categories=categories,
        caption_policy="normalized_ordered_and",
        correction_dicts=corrections,
        seed=42,
    )

    samples_by_id = {
        sample.sample_id: sample
        for sample in dataset
    }

    native_features, native_idx = load_features(cache_root / "native")
    reference_ids = [
        samples_by_id[sample_id].reference_id
        for sample_id in sample_ids
    ]

    reference_rows = [
        native_idx[reference_id]
        for reference_id in reference_ids
    ]

    reference = native_features[reference_rows].float().to(device)
    text = teacher_states[rows].float().to(device)
    mask = attention_mask[rows].bool().to(device)
    print("Loading full teacher on CPU...")

    full_teacher = CSMCIRStage1Teacher(
        csmcir_root=args.csmcir_root,
        checkpoint_path=args.checkpoint,
        device="cpu",
    ).eval()

    print("Loading compose-only teacher on CPU...")

    compose_teacher = CSMCIRComposeTeacher(csmcir_root=args.csmcir_root, checkpoint_path=args.checkpoint).eval()

    with torch.no_grad():
        q_full = full_teacher.compose(reference, text, mask, normalize=False)
        q_compose = compose_teacher.compose(reference, text, mask, normalize=False)

    output_diff = (q_full - q_compose).abs().max().item()

    print()
    print("output max abs diff:", f"{output_diff:.12e}")

    if output_diff > 1e-6:
        raise AssertionError(f"Compose output parity failed: {output_diff}")

    full_input = text.detach().clone().requires_grad_(True)
    compose_input = text.detach().clone().requires_grad_(True)
    q1 = full_teacher.compose(reference, full_input, mask, normalize=False)
    q2 = compose_teacher.compose(reference, compose_input, mask, normalize=False)
    loss1 = q1.square().mean()
    loss2 = q2.square().mean()

    loss1.backward()
    loss2.backward()

    if full_input.grad is None:
        raise AssertionError("Full teacher produced no input gradient")

    if compose_input.grad is None:
        raise AssertionError("Compose-only teacher produced no input gradient")

    grad_diff = (full_input.grad - compose_input.grad).abs().max().item()
    print("input-grad max abs diff:", f"{grad_diff:.12e}")
    print("full grad norm:", f"{full_input.grad.norm().item():.12e}")
    print("compose grad norm:", f"{compose_input.grad.norm().item():.12e}")

    if not torch.isfinite(compose_input.grad).all():
        raise AssertionError("Compose-only input gradient contains NaN/Inf")

    if compose_input.grad.abs().sum().item() == 0:
        raise AssertionError("Compose-only input gradient is exactly zero")

    if grad_diff > 1e-6:
        raise AssertionError(
            "Compose gradient parity failed: "
            f"{grad_diff}"
        )

    trainable = [
        name
        for name, parameter
        in compose_teacher.named_parameters()
        if parameter.requires_grad
    ]

    if trainable:
        raise AssertionError(
            "Compose teacher has trainable params: "
            f"{trainable}"
        )

    print()
    print("=" * 80)
    print("COMPOSE-ONLY TEACHER PARITY: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()