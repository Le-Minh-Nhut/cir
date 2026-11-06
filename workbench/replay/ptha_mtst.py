"""PTHA+MTST preflight and explicitly labelled CIRR-to-FashionIQ transfer replay."""
from __future__ import annotations

import importlib
import json
import os
import sys
from argparse import ArgumentParser
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend.fashioniq_layout import CATEGORIES
from workbench.backend.registry import model_by_id
from workbench.replay.common import isolated_cache_root, source_pin_status, sha256_file

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
CHECKPOINT_IDS = tuple(checkpoint["checkpoint_id"] for checkpoint in MODEL["checkpoint_variants"])
DEFAULT_CHECKPOINT_ID = CHECKPOINT_IDS[0]
PUBLISHED_CIRR_WEIGHT = next(
    checkpoint["filename"] for checkpoint in MODEL["checkpoint_variants"]
    if checkpoint.get("transfer_condition") == "cross_dataset_transfer"
)
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
    checkpoint_id: str = DEFAULT_CHECKPOINT_ID,
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

    if not (dataset_root / "images").is_dir():
        blockers.append(f"FashionIQ raw input missing: {dataset_root / 'images'}")
    blockers.extend(
        f"FashionIQ raw input missing: {dataset_root / path}"
        for path in RAW_INPUTS
        if not (dataset_root / path).is_file()
    )
    from workbench.replay.common import fashioniq_image_binding_blockers
    blockers.extend(fashioniq_image_binding_blockers(dataset_root, "val", "images", ".png"))

    checkpoint = next(
        (item for item in MODEL["checkpoint_variants"] if item["checkpoint_id"] == checkpoint_id),
        None,
    )
    if checkpoint is None:
        blockers.append(
            f"unknown checkpoint id {checkpoint_id!r}; valid checkpoint ids: {', '.join(CHECKPOINT_IDS)}"
        )
        return blockers

    is_cirr_transfer = (
        checkpoint.get("transfer_condition") == "cross_dataset_transfer"
        and checkpoint.get("checkpoint_training_dataset") == "CIRR"
        and checkpoint.get("checkpoint_forwards_dataset") == "FashionIQ"
    )
    if transfer_cirr_to_fashioniq and not is_cirr_transfer:
        blockers.append(
            f"checkpoint {checkpoint_id!r} is not registered as a CIRR-to-FashionIQ transfer"
        )
    elif is_cirr_transfer and transfer_cirr_to_fashioniq:
        path = checkpoint_root / checkpoint["filename"]
        if not path.is_file():
            blockers.append(f"PTHA+MTST CIRR checkpoint missing: {path}")
        elif path.stat().st_size == 0:
            blockers.append(f"PTHA+MTST CIRR checkpoint is an empty placeholder: {path}")
        from workbench.backend.registry import external_asset_blockers
        blockers.extend(external_asset_blockers(MODEL))
    elif is_cirr_transfer:
        blockers.append(
            f"PTHA+MTST FashionIQ fine-tuned checkpoint missing: author repository "
            f"https://huggingface.co/{HF_REPO} publishes only CIRR weights "
            f"('{checkpoint['filename']}'); CIRR weights require explicit transfer opt-in"
        )
    else:
        blockers.append(
            f"PTHA+MTST FashionIQ fine-tuned checkpoint missing: author repository "
            f"https://huggingface.co/{HF_REPO} publishes only CIRR weights "
            f"('{PUBLISHED_CIRR_WEIGHT}'); CIRR weights require explicit transfer opt-in"
        )
    return blockers


@contextmanager
def _fashioniq_dataset_binding(dataset_root: Path) -> Iterator[Any]:
    """Bind MTST's literal dataset path to this requested root for one evaluation."""
    previous_env = os.environ.get("MTST_DATA_ROOT")
    previous_module = sys.modules.get("data_utils")
    previous_base_path = getattr(previous_module, "base_path", None)
    with ExitStack() as stack:
        if dataset_root.name == "fashionIQ_dataset":
            data_root = dataset_root.parent
        else:
            data_root = stack.enter_context(isolated_cache_root(dataset_root, directory_names=()))
            (data_root / "fashionIQ_dataset").symlink_to(
                dataset_root.resolve(), target_is_directory=True
            )
        os.environ["MTST_DATA_ROOT"] = str(data_root)
        try:
            data_utils = importlib.import_module("data_utils")
            restore_base_path = (
                previous_base_path if data_utils is previous_module
                else Path(previous_env) if previous_env else Path("path_to_data_dir")
            )
            data_utils.base_path = data_root
            try:
                yield data_utils
            finally:
                data_utils.base_path = restore_base_path
        finally:
            if previous_env is None:
                os.environ.pop("MTST_DATA_ROOT", None)
            else:
                os.environ["MTST_DATA_ROOT"] = previous_env


#: Which side of the boundary owns the `vision_proj` step, decided by the SELECTED AND LOADED class.
#:
#: Both supported classes call `extract_target_features(image, mode="mean")` and both keep the
#: projection commented out, so both hand the boundary raw Q-Former states `[N, 32, 768]`. They
#: differ in where the projection belongs:
#:
#: ``cir_cat``  -- `blip2_qformer_cir_cat.py` defines `extract_target_features` TWICE (lines 453 and
#:   483), Python keeps the second one, and its `inference()` (line 287) matmuls the 256-wide query
#:   side against the target tensor it is handed. It never calls `vision_proj` on targets itself, so
#:   the boundary MUST: the width cannot otherwise meet, and the checkpoint carries the trained
#:   parameters (`vision_proj` 256x768 = 196 864).
#:
#: ``tgir_ce_sim`` -- `blip2_qformer_tgir_concatembedding_sim.py:429` runs
#:   `F.normalize(self.vision_proj(target_embeds), dim=-1)` INSIDE `inference()`, and its
#:   `extract_target_features` (line 501) also returns the raw states. The boundary must therefore
#:   pass raw states through and leave the projection to the model.
#:
#: Choosing this by class rather than by tensor width or filename is what the earlier revision got
#: wrong: applying the projection unconditionally to both classes double-projected the TGIR branch.
#: A shape check would not have caught that either, because projecting raw tgir states first yields a
#: tensor that this same `inference` then projects again -- a wrong-but-shaped-correct result.
TARGET_FEATURE_CONTRACTS: dict[str, str] = {
    "Blip2QformerCirCat": "project_at_boundary",
    "Blip2Qformer4TGIRCESIM": "project_inside_inference",
}

def _target_feature_contract(model: Any, index_features: Any, torch: Any,
                             contract: str | None = None) -> tuple[Any, dict[str, Any]]:
    """Give `inference()` the target representation THIS loaded class expects, exactly once.

    `index_features` is the native `extract_index_blip_features` output: a 2-tuple whose members are
    the gallery target features and the frozen vision embeddings. Both members are preserved
    unchanged in type and order, because `compute_fiq_val_metrics` reaches for `index_features[0]`
    to score against and the reference side is taken from `index_features[-1]`.

    Raises instead of guessing when the contract is not one this workbench declares, or when the
    declared contract does not match the loaded class's geometry.
    """
    contract = contract if contract is not None else TARGET_FEATURE_CONTRACTS.get(type(model).__name__)
    class_name = type(model).__name__
    if contract is None:
        raise ValueError(
            f"PTHA+MTST target-feature contract is undeclared for loaded class {class_name!r}; "
            f"declared classes are {sorted(TARGET_FEATURE_CONTRACTS)}. Refusing to guess whether "
            f"this class projects target embeddings inside `inference()`, because guessing wrong "
            f"either double-projects or fails to project at all.")
    if contract not in TARGET_FEATURE_CONTRACTS.values():
        raise ValueError(
            f"unknown PTHA+MTST target-feature contract {contract!r} for class {class_name!r}; "
            f"expected one of {sorted(set(TARGET_FEATURE_CONTRACTS.values()))}")

    if not (isinstance(index_features, tuple) and len(index_features) == 2):
        raise ValueError(
            "PTHA+MTST index features are not the native (target_features, raw_vision_embeds) pair: "
            f"got {type(index_features).__name__} of length "
            f"{len(index_features) if hasattr(index_features, '__len__') else 'n/a'}")
    raw_targets, frozen_vision = index_features
    report: dict[str, Any] = {
        "contract": contract,
        "loaded_class": class_name,
        "extracted_shape": list(raw_targets.shape),
        "new_parameters_introduced": 0,
        "frozen_vision_preserved": True,
    }

    if contract == "project_inside_inference":
        # The model projects these itself at `..._concatembedding_sim.py:429`. Any projection here
        # would be applied a second time inside `inference()`, which no shape check would flag.
        report.update(
            action="passed_through_unprojected",
            reason="this class applies vision_proj to target embeddings inside inference(); "
                   "projecting at the boundary would double-project",
            output_shape=list(raw_targets.shape),
        )
        return (raw_targets, frozen_vision), report

    hidden = getattr(model.vision_proj, "in_features", None)
    embed = getattr(model.vision_proj, "out_features", None)
    # Width is validated BEFORE projecting: otherwise a changed extraction path surfaces as an
    # opaque `mat1 and mat2 shapes cannot be multiplied` from inside `nn.Linear` instead of as the
    # declared-contract diagnostic below.
    if hidden is not None and raw_targets.shape[-1] != hidden:
        raise ValueError(
            f"PTHA+MTST contract {contract!r} for class {class_name!r} expects unprojected targets of "
            f"width {hidden}, but the extracted features are width {raw_targets.shape[-1]}. "
            f"Either the extraction path changed or the wrong contract was dispatched.")
    projected = model.inference_target(raw_targets)
    if embed is not None and projected.shape[-1] != embed:
        raise ValueError(
            f"PTHA+MTST boundary projection produced width {projected.shape[-1]}, but this class's "
            f"inference() consumes width {embed}")
    report.update(
        action="projected_at_boundary_via_inference_target",
        reason="effective extract_target_features returns unprojected Q-Former states while "
               "inference() does not project targets itself",
        projection_parameters_reused=sum(p.numel() for p in model.vision_proj.parameters()),
        projection_in_features=hidden,
        projection_out_features=embed,
        output_shape=list(projected.shape),
    )
    # Equivalence is checked against the author's own operation with the SAME weights, on the SAME
    # device, so this stays meaningful on CPU-only test hosts too.
    device = model.vision_proj.weight.device
    with torch.no_grad():
        direct = torch.nn.functional.normalize(model.vision_proj(raw_targets.to(device)), dim=-1)
    report["equivalence_vs_direct_projection_max_abs_diff"] = float(
        (direct.to(projected.device) - projected).abs().max())
    return (projected, frozen_vision), report

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


def evaluate_transfer(
    source_root: Path, dataset_root: Path, checkpoint: Path, checkpoint_id: str
) -> dict[str, Any]:
    """Safely inspect and compare checkpoint, then run the pinned FashionIQ metrics."""
    import torch

    src = source_tree(source_root) / "src"
    sys.path.insert(0, str(src))

    checkpoint_sha256 = sha256_file(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not torch.cuda.is_available():
        raise RuntimeError("PTHA native inference uses CUDA-only `.cuda()` calls; CUDA is unavailable")

    from omegaconf import OmegaConf
    from lavis.common.registry import registry
    from lavis.models import load_preprocess

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
    # Resolved from the INSTANCE that was actually constructed and loaded, not from the checkpoint
    # key and not from a tensor width: the same payload key could in principle build a different
    # class, and the contract belongs to the class whose `inference()` will run.
    target_contract = TARGET_FEATURE_CONTRACTS.get(type(model).__name__)
    if target_contract is None:
        raise ValueError(
            f"checkpoint sha256={checkpoint_sha256}; checkpoint key {checkpoint_key!r} built class "
            f"{type(model).__name__!r}, which has no declared target-feature contract "
            f"({sorted(TARGET_FEATURE_CONTRACTS)}); refusing to guess a projection")

    _vis_processors, text_processors = load_preprocess(config.preprocess)
    with _fashioniq_dataset_binding(dataset_root) as data_utils:
        from utils import extract_index_blip_features
        from validate_blip import compute_fiq_val_metrics

        FashionIQDataset = data_utils.FashionIQDataset
        targetpad_transform = data_utils.targetpad_transform
        transform = targetpad_transform(1.25, 224)
        category_metrics: dict[str, dict[str, float]] = {}
        contract_reports: dict[str, dict[str, Any]] = {}
        for category in CATEGORIES:
            relative = FashionIQDataset("val", [category], "relative", transform)
            classic = FashionIQDataset("val", [category], "classic", transform)
            index_features, index_names = extract_index_blip_features(classic, model)
            index_features, contract_report = _target_feature_contract(
                model, index_features, torch, target_contract)
            contract_reports[category] = contract_report
            r10, r50 = compute_fiq_val_metrics(relative, model, index_features, index_names, text_processors)
            category_metrics[category] = {"r10": r10, "r50": r50}
            torch.cuda.empty_cache()


    r10 = sum(values["r10"] for values in category_metrics.values()) / len(category_metrics)
    r50 = sum(values["r50"] for values in category_metrics.values()) / len(category_metrics)
    return {
        "model_id": MODEL["model_id"],
        "protocol_id": MODEL["native_protocol"],
        "label": TRANSFER_LABEL,
        "train_dataset": "CIRR",
        "evaluation_dataset": "FashionIQ",
        "checkpoint_training_noise": "unknown",
        "checkpoint_id": checkpoint_id,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_top_level_model_key": checkpoint_key,
        "serialization_inspection": "torch.load(weights_only=True)",
        "model_name": model_name,
        "backbone_config": f"{model_type} (EVA-CLIP ViT-g; 224 px, 32 Q-Former tokens)",
        "load_report": {**report, "strict_load": True},
        "target_feature_contract": contract_reports,
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
    parser.add_argument("--checkpoint-id", choices=CHECKPOINT_IDS, default=DEFAULT_CHECKPOINT_ID)
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
        checkpoint_id=args.checkpoint_id,
        transfer_cirr_to_fashioniq=args.transfer_cirr_to_fashioniq,
    )
    if blockers:
        for blocker in blockers:
            print(f"[BLOCKED] {blocker}", file=sys.stderr)
        return 1
    checkpoint_variant = next(
        item for item in MODEL["checkpoint_variants"] if item["checkpoint_id"] == args.checkpoint_id
    )
    checkpoint = args.checkpoint_root / checkpoint_variant["filename"]
    try:
        result = evaluate_transfer(
            args.source_root, args.dataset_root, checkpoint, args.checkpoint_id
        )
    except Exception as error:
        print(f"[BLOCKED] {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
