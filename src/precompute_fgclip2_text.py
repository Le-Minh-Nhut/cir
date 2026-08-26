from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from backbones.fgclip2 import (
    FGCLIP2Backbone,
    FGCLIP2_LARGE_DIM,
    FGCLIP2_LARGE_MODEL_ID,
    FGCLIP2_SHORT_TEXT_LENGTH,
)
from datasets.common import collate_cir_samples
from datasets.fashioniq import FashionIQDataset, load_correction_dict


CATEGORIES = ("dress", "shirt", "toptee")
CAPTION_POLICY = "normalized_ordered_and"
VALID_SPLITS = ("train", "val")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and precompute frozen FG-CLIP2-Large FashionIQ text-token states."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/fashionIQ_dataset"),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("features/fashioniq/fgclip2-large"),
    )
    parser.add_argument("--model-id", default=FGCLIP2_LARGE_MODEL_ID)
    parser.add_argument("--splits", nargs="+", choices=VALID_SPLITS, default=list(VALID_SPLITS))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--parity-samples", type=int, default=3)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Run the exact-tokenizer length preflight without loading model weights.",
    )
    return parser.parse_args()


def load_correction_dicts(annotation_root: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for category in CATEGORIES:
        path = annotation_root / f"correction_dict_{category}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing FashionIQ correction dictionary: {path}")
        result[category] = load_correction_dict(path)
    return result


def build_dataset(
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


def audit_token_lengths(tokenizer, captions: list[str]) -> dict[str, float | int]:
    lengths: list[int] = []
    audit_batch_size = 1024
    for start in range(0, len(captions), audit_batch_size):
        encoded = tokenizer(
            captions[start : start + audit_batch_size],
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )
        lengths.extend(len(input_ids) for input_ids in encoded["input_ids"])

    values = np.asarray(lengths, dtype=np.int64)
    over_limit = int((values > FGCLIP2_SHORT_TEXT_LENGTH).sum())
    return {
        "num_samples": int(values.size),
        "maximum_token_length": int(values.max()),
        "mean_token_length": float(values.mean()),
        "p95_token_length": float(np.percentile(values, 95)),
        "p99_token_length": float(np.percentile(values, 99)),
        "num_samples_over_64": over_limit,
        "fraction_samples_over_64": float(over_limit / values.size),
    }


def print_token_audit(audit: dict[str, float | int]) -> None:
    print("FashionIQ FG-CLIP2-Large token-length preflight")
    for key, value in audit.items():
        print(f"  {key}: {value}")


@torch.inference_mode()
def precompute_split(
    *,
    split: str,
    dataset: FashionIQDataset,
    backbone: FGCLIP2Backbone,
    output_dir: Path,
    batch_size: int,
    num_workers: int,
    parity_samples: int,
    token_audit: dict[str, float | int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_cir_samples,
        pin_memory=backbone.device.type == "cuda",
    )

    states = np.lib.format.open_memmap(
        output_dir / "states.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(dataset), FGCLIP2_SHORT_TEXT_LENGTH, FGCLIP2_LARGE_DIM),
    )
    attention = np.lib.format.open_memmap(
        output_dir / "attention_mask.npy",
        mode="w+",
        dtype=np.bool_,
        shape=(len(dataset), FGCLIP2_SHORT_TEXT_LENGTH),
    )
    content = np.lib.format.open_memmap(
        output_dir / "content_mask.npy",
        mode="w+",
        dtype=np.bool_,
        shape=(len(dataset), FGCLIP2_SHORT_TEXT_LENGTH),
    )

    sample_to_idx: dict[str, int] = {}
    captions: dict[str, str] = {}
    parity_direct: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    parity_rows: list[int] = []
    row = 0

    for batch in tqdm(loader, desc=f"FG-CLIP2 text [{split}]", dynamic_ncols=True):
        batch_states, batch_attention, batch_content = backbone.encode_text_tokens(
            batch.modification_texts
        )
        if batch_states.requires_grad:
            raise RuntimeError("Text precompute unexpectedly recorded gradients")
        current_batch_size = len(batch.sample_ids)
        end = row + current_batch_size
        states[row:end] = batch_states.cpu().numpy()
        attention[row:end] = batch_attention.cpu().numpy()
        content[row:end] = batch_content.cpu().numpy()

        if parity_direct is None and parity_samples > 0:
            keep = min(parity_samples, current_batch_size)
            parity_direct = (
                batch_states[:keep].cpu().clone(),
                batch_attention[:keep].cpu().clone(),
                batch_content[:keep].cpu().clone(),
            )
            parity_rows = list(range(row, row + keep))

        for offset, (sample_id, caption) in enumerate(
            zip(batch.sample_ids, batch.modification_texts, strict=True)
        ):
            if sample_id in sample_to_idx:
                raise ValueError(f"Duplicate sample_id: {sample_id}")
            sample_to_idx[sample_id] = row + offset
            captions[sample_id] = caption
        row = end

    if row != len(dataset):
        raise RuntimeError(f"Expected {len(dataset)} text rows, wrote {row}")
    states.flush()
    attention.flush()
    content.flush()

    with (output_dir / "sample_to_idx.json").open("w", encoding="utf-8") as file:
        json.dump(sample_to_idx, file, indent=2)
    with (output_dir / "captions.json").open("w", encoding="utf-8") as file:
        json.dump(captions, file, indent=2, ensure_ascii=False)

    cached_states = np.load(output_dir / "states.npy", mmap_mode="r")
    cached_attention = np.load(output_dir / "attention_mask.npy", mmap_mode="r")
    cached_content = np.load(output_dir / "content_mask.npy", mmap_mode="r")
    if not np.isfinite(cached_states).all():
        raise FloatingPointError("Reloaded text cache contains NaN or Inf")

    parity_max_abs_error = 0.0
    if parity_direct is not None:
        direct_states, direct_attention, direct_content = parity_direct
        saved_states = torch.from_numpy(cached_states[parity_rows].copy())
        saved_attention = torch.from_numpy(cached_attention[parity_rows].copy())
        saved_content = torch.from_numpy(cached_content[parity_rows].copy())
        parity_max_abs_error = float((direct_states - saved_states).abs().max().item())
        if not torch.allclose(direct_states, saved_states, rtol=0.0, atol=1e-6):
            raise RuntimeError(
                "FG-CLIP2 text direct/cache parity failed: "
                f"max_abs_error={parity_max_abs_error}"
            )
        if not torch.equal(direct_attention, saved_attention):
            raise RuntimeError("FG-CLIP2 attention-mask cache parity failed")
        if not torch.equal(direct_content, saved_content):
            raise RuntimeError("FG-CLIP2 content-mask cache parity failed")

    manifest = {
        "dataset": "FashionIQ",
        "split": split,
        "feature_kind": "fgclip2_contextual_text_tokens",
        "model_id": backbone.model_id,
        "caption_policy": CAPTION_POLICY,
        "max_text_length": backbone.max_text_length,
        "num_samples": len(dataset),
        "states_shape": list(cached_states.shape),
        "attention_mask_shape": list(cached_attention.shape),
        "content_mask_shape": list(cached_content.shape),
        "states_dtype": str(cached_states.dtype),
        "mask_dtype": str(cached_attention.dtype),
        "requires_grad": False,
        "token_length_preflight": token_audit,
        "parity_samples": len(parity_rows),
        "parity_max_abs_error": parity_max_abs_error,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    print(f"[{split}] states: {tuple(cached_states.shape)} -> {output_dir}")
    print(f"[{split}] cache parity max abs error: {parity_max_abs_error:.3e}")


def main() -> None:
    args = parse_args()
    if args.model_id != FGCLIP2_LARGE_MODEL_ID:
        raise ValueError(
            f"This experiment requires exactly {FGCLIP2_LARGE_MODEL_ID!r}"
        )
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("Invalid batch-size/num-workers")
    if args.parity_samples < 0:
        raise ValueError("--parity-samples must be >= 0")

    annotation_root = args.dataset_root / "captions"
    correction_dicts = load_correction_dicts(annotation_root)
    datasets = {
        split: build_dataset(annotation_root, split, correction_dicts)
        for split in args.splits
    }
    all_captions = [
        datasets[split][index].modification_text
        for split in args.splits
        for index in range(len(datasets[split]))
    ]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
    )
    token_audit = audit_token_lengths(tokenizer, all_captions)
    print_token_audit(token_audit)
    if args.audit_only:
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    backbone = FGCLIP2Backbone(
        model_id=args.model_id,
        max_text_length=FGCLIP2_SHORT_TEXT_LENGTH,
    ).to(device)
    backbone.eval()
    if any(parameter.requires_grad for parameter in backbone.model.parameters()):
        raise RuntimeError("FG-CLIP2-Large is not fully frozen")

    for split, dataset in datasets.items():
        precompute_split(
            split=split,
            dataset=dataset,
            backbone=backbone,
            output_dir=args.cache_root / split / "text",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            parity_samples=args.parity_samples,
            token_audit=token_audit,
        )


if __name__ == "__main__":
    main()
