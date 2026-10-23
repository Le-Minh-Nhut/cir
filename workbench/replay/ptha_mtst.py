"""PTHA+MTST preflight and explicitly labelled CIRR-to-FashionIQ transfer replay."""
from __future__ import annotations

import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend.fashioniq_layout import CATEGORIES
from workbench.backend.registry import model_by_id
from workbench.replay.common import source_pin_status, sha256_file

MODEL = model_by_id("ptha_mtst")
RAW_INPUTS = tuple(
    Path(path)
    for category in CATEGORIES
    for path in (
        f"captions/cap.{category}.val.json",
        f"image_splits/split.{category}.val.json",
    )
)
HF_REPO = "yinanzhou1/mtst_ptha"
PUBLISHED_CIRR_WEIGHT = "tuned_mtst_cirr.pt"
TRANSFER_LABEL = "PTHA+MTST | CIRR checkpoint -> FashionIQ | cross-dataset transfer | exploratory"


def source_tree(source_root: Path) -> Path:
    """The directory that actually holds the upstream tree's `src/`.

    The vendored checkout keeps the repository's own top-level directory inside the checkout, so the
    code lives at `<source_root>/MTST_PTHA/MTST_PTHA/src` while `requirements.txt` and `provenance.json`
    sit at `<source_root>/MTST_PTHA`. Resolving it in ONE place means the evaluator path and the
    import path cannot disagree, which is what they did: both named `<source_root>/MTST_PTHA/src`, a
    directory that does not exist, so every preflight reported the evaluator missing and the import
    would have failed on `data_utils`.
    """
    outer = source_root / "MTST_PTHA"
    nested = outer / "MTST_PTHA"
    return nested if (nested / "src").is_dir() else outer


def evaluator_path(source_root: Path) -> Path:
    return source_tree(source_root) / "src" / "validate_blip.py"


def preflight(
    source_root: Path,
    dataset_root: Path,
    checkpoint_root: Path,
    *,
    transfer_cirr_to_fashioniq: bool = False,
) -> list[str]:
    """Return blockers without importing upstream code or mutating inputs."""
    blockers: list[str] = []
    ok, reason = source_pin_status(source_root, MODEL["upstream_commit_sha"], MODEL)
    if not ok:
        blockers.append(f"pinned PTHA+MTST source unavailable: {reason}")

    evaluator = evaluator_path(source_root)
    if not evaluator.is_file():
        blockers.append(f"pinned evaluator missing at {evaluator}")

    blockers.extend(
        f"FashionIQ raw input missing: {dataset_root / path}"
        for path in RAW_INPUTS
        if not (dataset_root / path).is_file()
    )

    if transfer_cirr_to_fashioniq:
        checkpoint = checkpoint_root / MODEL["model_id"] / PUBLISHED_CIRR_WEIGHT
        if not checkpoint.is_file():
            blockers.append(f"PTHA+MTST CIRR checkpoint missing: {checkpoint}")
        elif checkpoint.stat().st_size == 0:
            blockers.append(f"PTHA+MTST CIRR checkpoint is an empty placeholder: {checkpoint}")
    else:
        blockers.append(
            f"PTHA+MTST FashionIQ fine-tuned checkpoint missing: author repository "
            f"https://huggingface.co/{HF_REPO} publishes only CIRR weights "
            f"('{PUBLISHED_CIRR_WEIGHT}'); CIRR weights require explicit transfer opt-in"
        )
    return blockers


def _state_dict_report(model_state: dict[str, Any], checkpoint_state: dict[str, Any]) -> dict[str, Any]:
    expected = set(model_state)
    supplied = set(checkpoint_state)
    missing = sorted(expected - supplied)
    unexpected = sorted(supplied - expected)
    mismatched_shapes = sorted(
        (key, tuple(checkpoint_state[key].shape), tuple(model_state[key].shape))
        for key in expected & supplied
        if tuple(checkpoint_state[key].shape) != tuple(model_state[key].shape)
    )
    return {
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatches": [
            {"key": key, "checkpoint": source, "model": target}
            for key, source, target in mismatched_shapes
        ],
        "strict_false_would_hide": {"missing_keys": missing, "unexpected_keys": unexpected},
        "strict_false_shape_errors": [key for key, _, _ in mismatched_shapes],
    }


def evaluate_transfer(source_root: Path, dataset_root: Path, checkpoint: Path) -> dict[str, Any]:
    """Safely inspect and compare checkpoint, then run the pinned FashionIQ metrics."""
    import torch

    src = source_tree(source_root) / "src"
    sys.path.insert(0, str(src))
    os.environ["MTST_DATA_ROOT"] = str(dataset_root.parent)

    checkpoint_sha256 = sha256_file(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not torch.cuda.is_available():
        raise RuntimeError("PTHA native inference uses CUDA-only `.cuda()` calls; CUDA is unavailable")

    from omegaconf import OmegaConf
    from lavis.common.registry import registry
    from lavis.models import load_preprocess
    from data_utils import FashionIQDataset, targetpad_transform
    from utils import extract_index_blip_features
    from validate_blip import compute_fiq_val_metrics

    checkpoint_models = {
        "Blip2QformerCirCat": ("blip2_cir_cat", "pretrain"),
        "Blip2Qformer4TGIRCESIM": ("blip2_tgir_ce_sim", "pretrain_mtst"),
    }
    checkpoint_key = next((key for key in checkpoint_models if key in payload), None) if isinstance(payload, dict) else None
    if checkpoint_key is None:
        keys = sorted(map(str, payload)) if isinstance(payload, dict) else [type(payload).__name__]
        raise ValueError(
            f"checkpoint sha256={checkpoint_sha256}; expected one of {list(checkpoint_models)}; "
            f"observed keys={keys}"
        )
    state = payload[checkpoint_key]
    if not isinstance(state, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()
    ):
        raise ValueError(
            f"checkpoint sha256={checkpoint_sha256}; {checkpoint_key!r} is not a tensor-only state dictionary"
        )

    model_name, model_type = checkpoint_models[checkpoint_key]
    model_class = registry.get_model_class(model_name)
    config = OmegaConf.load(model_class.default_config_path(model_type))
    # The state dict must not cause the constructor to fetch an implicit BLIP-2 base checkpoint.
    config.model.load_pretrained = False
    model = model_class.from_config(config.model)
    report = _state_dict_report(model.state_dict(), state)
    if report["missing_keys"] or report["unexpected_keys"] or report["shape_mismatches"]:
        raise ValueError(
            f"checkpoint sha256={checkpoint_sha256} incompatible with "
            f"{model_name}/{model_type}: {json.dumps(report)}"
        )
    model.load_state_dict(state, strict=True)
    report["strict_false_hid_anything"] = False
    model = model.cuda().eval()

    _vis_processors, text_processors = load_preprocess(config.preprocess)
    transform = targetpad_transform(1.25, 224)
    category_metrics: dict[str, dict[str, float]] = {}
    for category in CATEGORIES:
        relative = FashionIQDataset("val", [category], "relative", transform)
        classic = FashionIQDataset("val", [category], "classic", transform)
        index_features, index_names = extract_index_blip_features(classic, model)
        r10, r50 = compute_fiq_val_metrics(relative, model, index_features, index_names, text_processors)
        category_metrics[category] = {"r10": r10, "r50": r50}
        torch.cuda.empty_cache()

    r10 = sum(values["r10"] for values in category_metrics.values()) / len(category_metrics)
    r50 = sum(values["r50"] for values in category_metrics.values()) / len(category_metrics)
    return {
        "label": TRANSFER_LABEL,
        "train_dataset": "CIRR",
        "evaluation_dataset": "FashionIQ",
        "checkpoint_training_noise": "unknown",
        "checkpoint_id": "ptha_mtst_cirr_transfer_exploratory",
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_top_level_model_key": checkpoint_key,
        "serialization_inspection": "torch.load(weights_only=True)",
        "model_name": model_name,
        "backbone_config": f"{model_type} (EVA-CLIP ViT-g; 224 px, 32 Q-Former tokens)",
        "load_report": {**report, "strict_load": True},
        "category_metrics": category_metrics,
        "macro_metrics": {"r10": r10, "r50": r50, "mean": (r10 + r50) / 2},
        "paper_fashioniq_context_only": {
            "training_condition": "FashionIQ_MTST",
            "r10": 54.99,
            "r50": 76.28,
            "mean": 65.64,
            "mismatch": "training dataset/checkpoint differs; numerical context only, not reproduction",
            "differences_transfer_minus_paper": {
                "r10": r10 - 54.99,
                "r50": r50 - 76.28,
                "mean": (r10 + r50) / 2 - 65.64,
            },
        },
    }



def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument(
        "--transfer-cirr-to-fashioniq",
        action="store_true",
        help="explicit exploratory transfer: CIRR-trained checkpoint evaluated on FashionIQ",
    )
    args = parser.parse_args(argv)

    blockers = preflight(
        args.source_root,
        args.dataset_root,
        args.checkpoint_root,
        transfer_cirr_to_fashioniq=args.transfer_cirr_to_fashioniq,
    )
    if blockers:
        for blocker in blockers:
            print(f"[BLOCKED] {blocker}", file=sys.stderr)
        return 1
    if not args.transfer_cirr_to_fashioniq:
        return 0

    checkpoint = args.checkpoint_root / MODEL["model_id"] / PUBLISHED_CIRR_WEIGHT
    try:
        result = evaluate_transfer(args.source_root, args.dataset_root, checkpoint)
    except Exception as error:
        print(f"[BLOCKED] {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
