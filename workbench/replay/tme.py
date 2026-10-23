"""Source-faithful evaluation runner for CVPR25-TME.

The pinned `precompute_test.py` has a `main()` that reads a local checkpoint path and prints the
native all-category metrics, but it takes no `--checkpoint-root`. This module resolves the checkpoint
and normalizes the native per-category results for the workbench.

from __future__ import annotations
rather than assumed:

* The gallery is the complete per-category validation split in file order, and the reference image
  is eligible. `precompute_evaluation.py` masks the reference only for CIRR, so TME's FashionIQ
  cohort is `fashioniq_original_split`, not a separate reference-included protocol.
* FashionIQ's two annotations are joined as `"<first> and <second>"` after the pinned
  `strip('.?, ').capitalize()` / `strip('.?, ')` treatment, and the ranking is `argsort` over
  `1 - similarity`, i.e. descending similarity. Both are performed by the pinned code called below.

The native `load_state_dict(..., strict=False)` report is NOT discarded: it is checked, and a load with
missing or unexpected keys fails rather than printing plausible Recall from a partially initialised
model.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import struct
import sys
from argparse import ArgumentParser
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from workbench.backend.fashioniq_layout import CATEGORIES
from workbench.backend.registry import (
    external_asset_statuses,
    model_by_id,
    preparation_contract_for_model,
    preparation_paths,
    required_runtime_assets,
)
from workbench.replay.common import (
    ReplayResult,
    isolated_cache_root,
    macro_aggregate,
    sha256_file,
    source_pin_status,
)

MODEL = model_by_id("tme")
CHECKPOINTS = {variant["checkpoint_id"]: variant for variant in MODEL["checkpoint_variants"]}
DEFAULT_CHECKPOINT = "fashioniq_0"
PROTOCOL = "fashioniq_original_split"
DATASET_LABEL = "FashionIQ"
#: The pinned model name and type the author's own `precompute_test.py` selects for evaluation.
MODEL_NAME = "blip2_cir_image_diff_features"
MODEL_TYPE = "pretrain"

#: Inputs the pinned constructor opens. `data.py` reads `cap.<cat>.<split>.json` and
#: `split.<cat>.<split>.json` from the dataset root, and the gallery path needs the images.
RAW_INPUTS = tuple(
    path
    for category in CATEGORIES
    for path in (
        f"captions/cap.{category}.val.json",
        f"image_splits/split.{category}.val.json",
    )
)


def _runtime_asset_check() -> tuple[list[str], list[dict[str, Any]]]:
    blockers: list[str] = []
    evidence: list[dict[str, Any]] = []
    assets = required_runtime_assets(MODEL)
    try:
        statuses = {status["name"]: status for status in external_asset_statuses(MODEL)}
    except OSError as error:
        return [f"TME runtime asset inspection failed: {error}"], evidence
    for asset in assets:
        status = statuses[asset["name"]]
        detail = f" ({status['path']})" if status["path"] else ""
        source = f"; source: {asset['source']}" if asset.get("source") else ""
        state = status["status"]
        if state in {"EXISTS_UNVERIFIED_CHECKSUM", "VERIFIED_LOCAL_ASSET", "CHECKSUM_MISMATCH"}:
            path = Path(status["path"])
            if not path.is_file():
                blockers.append(f"TME runtime asset {asset['name']} unreadable or not a file{detail}")
                continue
            try:
                actual_size = path.stat().st_size
            except OSError as error:
                blockers.append(f"TME runtime asset {asset['name']} unreadable{detail}: {error}")
                continue
            expected_size = asset.get("expected_size_bytes")
            if expected_size is not None and actual_size != expected_size:
                blockers.append(
                    f"TME runtime asset {asset['name']} size mismatch{detail}: "
                    f"expected {expected_size} bytes, got {actual_size}")
                continue
        expected_digest = asset.get("expected_sha256") or asset.get("expected_sha256_prefix")
        if state == "EXISTS_UNVERIFIED_CHECKSUM" and not expected_digest:
            verification = (
                "author digest unavailable; local digest recorded, not verified against a published value")
            print(f"[WARNING] TME runtime asset {asset['name']} {state}{detail}{source}; "
                  f"{verification}", file=sys.stderr)
        elif state != "VERIFIED_LOCAL_ASSET":
            blockers.append(f"TME runtime asset {asset['name']} {state}{detail}{source}")
            continue
        evidence.append({
            "name": asset["name"],
            "status": state,
            "path": status["path"],
            "expected_sha256": asset.get("expected_sha256"),
            "expected_sha256_prefix": asset.get("expected_sha256_prefix"),
            "local_sha256": status["actual_sha256"],
            "verification": verification if state == "EXISTS_UNVERIFIED_CHECKSUM"
            and not expected_digest else "verified against published digest",
        })
    return blockers, evidence


def source_root_for(checkout: Path) -> Path:
    """The directory that holds `src/`, for the vendored layout as delivered."""
    return checkout if (checkout / "src").is_dir() else checkout


def evaluator_path(checkout: Path) -> Path:
    return source_root_for(checkout) / "src" / "precompute_test.py"


def preflight(checkout: Path, dataset_root: Path, checkpoint_root: Path,
              checkpoint_id: str | None = None,
              runtime_assets: tuple[list[str], list[dict[str, Any]]] | None = None) -> list[str]:
    """Return blockers without importing upstream code or mutating inputs."""
    blockers: list[str] = []
    ok, reason = source_pin_status(checkout, MODEL["upstream_commit_sha"], MODEL)
    if not ok:
        blockers.append(f"pinned TME source unavailable: {reason}")

    evaluator = evaluator_path(checkout)
    if not evaluator.is_file():
        blockers.append(f"pinned evaluator missing at {evaluator}")

    blockers.extend(
        f"TME dataset input missing: {dataset_root / path}"
        for path in RAW_INPUTS
        if not (dataset_root / path).is_file()
    )

    variant = CHECKPOINTS.get(checkpoint_id or DEFAULT_CHECKPOINT)
    if variant is None:
        blockers.append(f"unsupported TME checkpoint: {checkpoint_id}; valid: {sorted(CHECKPOINTS)}")
        return blockers
    checkpoint = checkpoint_root / variant["filename"]
    if not checkpoint.is_file():
        blockers.append(f"TME checkpoint missing: {checkpoint}")
    elif checkpoint.stat().st_size == 0:
        blockers.append(f"TME checkpoint is an empty placeholder: {checkpoint}")

    # Native `load_image_features()` resolves `./features/FashionIQ` from evaluator cwd; the
    # launcher uses the checkout as cwd, so preflight and evaluation share this root.
    contract = preparation_contract_for_model(MODEL)
    feature_parameters = next(
        artifact["parameters"] for artifact in contract["generated_artifacts"]
        if artifact["generator"] == "tme_precompute_features")
    feature_root = preparation_paths(
        contract, source_root_for(checkout), "generated_artifacts")[0]
    feature_files = (feature_root / "val_images.pt", feature_root / "val_name_to_idx.json")
    try:
        dtype_bytes = struct.calcsize(
            {"float16": "e", "float32": "f", "float64": "d"}
            [feature_parameters["feature_dtype"]])
        image_count = sum(
            len(json.loads((dataset_root / f"image_splits/split.{category}.val.json")
                           .read_text(encoding="utf-8")))
            for category in CATEGORIES)
        required_bytes = (image_count * feature_parameters["tokens_per_image"]
                          * feature_parameters["feature_width"] * dtype_bytes)
    except (OSError, ValueError, TypeError, KeyError, struct.error) as error:
        blockers.append(f"TME feature cache size estimate unavailable: {error}")
    else:
        destination = feature_root
        while not destination.exists() and destination != destination.parent:
            destination = destination.parent
        try:
            free_text = f"{shutil.disk_usage(destination).free} bytes"
        except OSError:
            free_text = "unavailable"
        shape_text = (f"{image_count} gallery images x {feature_parameters['tokens_per_image']} tokens "
                      f"x {feature_parameters['feature_width']} features x {dtype_bytes} bytes")
        for path in feature_files:
            if not path.is_file():
                blockers.append(
                    f"TME feature cache file missing: {path} (required feature tensor lower bound: "
                    f"{required_bytes} bytes; {shape_text}; free space at destination: {free_text})")
    asset_blockers, _ = runtime_assets if runtime_assets is not None else _runtime_asset_check()
    blockers.extend(asset_blockers)
    return blockers

def _prior_feature_provenance() -> dict[str, Any] | None:
    logs = Path(__file__).resolve().parents[1] / "artifacts" / "logs"
    for path in sorted(logs.glob("tme__*/stdout.log"), reverse=True):
        try:
            output = path.read_text(encoding="utf-8")
            start = output.rfind('{"model_id":')
            if start < 0:
                continue
            payload, _ = json.JSONDecoder().raw_decode(output[start:])
        except (OSError, ValueError):
            continue
        for row in payload.get("results", []):
            provenance = row.get("provenance")
            if isinstance(provenance, dict) and "feature_cache_sha256" in provenance:
                return provenance
    return None
def _evaluate_categories(checkout: Path, dataset_root: Path, checkpoint_root: Path,
                         categories: tuple[str, ...], checkpoint_id: str | None,
                         prior_provenance: dict[str, Any] | None,
                         runtime_asset_evidence: list[dict[str, Any]]) -> tuple[list[ReplayResult], dict[str, Any]]:
    import torch

    variant = CHECKPOINTS.get(checkpoint_id or DEFAULT_CHECKPOINT)
    if variant is None:
        raise ValueError(f"unsupported TME checkpoint: {checkpoint_id}")
    checkpoint_path = checkpoint_root / variant["filename"]
    if len(categories) != len(CATEGORIES) or set(categories) != set(CATEGORIES):
        # The native evaluator always computes all three categories and has no single-category API.
        # Refuse subsets rather than report metrics for categories the operator did not request.
        raise ValueError("TME native evaluator only supports the full FashionIQ category set")
    source = source_root_for(checkout)
    sys.path.insert(0, str(source / "src"))
    os.environ["MTST_DATA_ROOT"] = str(dataset_root.parent)

    from lavis.models import load_model_and_preprocess
    import data as tme_data
    from precompute_evaluation import evaluate_features
    from utility import targetpad_transform

    tme_data.FashioniqPath = str(dataset_root) + "/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blip_model, _, txt_processors = load_model_and_preprocess(
        name=MODEL_NAME, model_type=MODEL_TYPE, is_eval=False, device=device)
    checkpoint_state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    report = blip_model.load_state_dict(checkpoint_state, strict=False)
    missing = [key for key in getattr(report, "missing_keys", [])]
    unexpected = [key for key in getattr(report, "unexpected_keys", [])]
    if missing or unexpected:
        raise RuntimeError(
            f"TME checkpoint did not load completely: {len(missing)} missing, {len(unexpected)} "
            f"unexpected (missing: {missing[:3]}, unexpected: {unexpected[:3]}); the native "
            "`strict=False` would have hidden this")
    blip_model.eval()

    contract = preparation_contract_for_model(MODEL)
    artifact = next(item for item in contract["generated_artifacts"]
                    if item["generator"] == "tme_precompute_features")
    parameters = artifact["parameters"]
    feature_root = preparation_paths(contract, source, "generated_artifacts")[0]
    feature_path = feature_root / f"{parameters['split']}_images.pt"
    mapping_path = feature_root / f"{parameters['split']}_name_to_idx.json"
    if not feature_path.is_file() or not mapping_path.is_file():
        raise FileNotFoundError(
            f"TME native feature cache missing: {feature_path} and/or {mapping_path}")

    preprocess = targetpad_transform(target_ratio=1.25, dim=224)
    with isolated_cache_root(dataset_root, directory_names=("captions", "image_splits")) as cache_root:
        tme_data.FashioniqPath = str(cache_root).rstrip("/") + "/"
        dataset = tme_data.get_dataset(DATASET_LABEL, preprocess, "val")
        if hasattr(dataset, "load_image_features"):
            dataset.load_image_features()
        names = dataset.current_image_names
        expected_index = {name: index for index, name in enumerate(names)}
        if list(dataset.name_to_idx.items()) != list(expected_index.items()):
            raise RuntimeError(
                "TME feature cache name-to-index mapping does not exactly match the ordered "
                "FashionIQ validation gallery")
        features = dataset.images
        expected_shape = (len(names), parameters["tokens_per_image"], parameters["feature_width"])
        expected_dtype = {"float16": torch.float16, "float32": torch.float32,
                          "float64": torch.float64}[parameters["feature_dtype"]]
        if not isinstance(features, torch.Tensor) or tuple(features.shape) != expected_shape:
            raise RuntimeError(
                f"TME feature cache shape mismatch: expected {expected_shape}, "
                f"got {getattr(features, 'shape', None)}")
        if features.dtype != expected_dtype:
            raise RuntimeError(
                f"TME feature cache dtype mismatch: expected {expected_dtype}, got {features.dtype}")

        provenance = {
            "dataset_root": str(dataset_root.resolve()),
            "feature_split": parameters["split"],
            "feature_backbone": f"{MODEL_NAME}:{MODEL_TYPE}@{MODEL['upstream_commit_sha']}",
            "source_commit": MODEL["upstream_commit_sha"],
            "feature_cache_sha256": sha256_file(feature_path),
            "feature_name_to_idx_sha256": sha256_file(mapping_path),
            "feature_gallery_sha256": {
                category: sha256_file(dataset_root / f"image_splits/split.{category}.val.json")
                for category in CATEGORIES
            },
            "feature_shape": list(features.shape),
            "feature_dtype": str(features.dtype),
        }
        differences = ([key for key in sorted(provenance)
                       if prior_provenance.get(key) != provenance[key]]
                       if prior_provenance is not None else ["no prior run provenance"])
        cache_provenance = ("VERIFIED_PRIOR_RUN"
                            if prior_provenance is not None and not differences else "UNVERIFIED")
        if cache_provenance == "UNVERIFIED":
            print("[WARNING] TME feature cache provenance UNVERIFIED for this run: "
                  + ", ".join(differences), file=sys.stderr)

        # This large feature cache is a legitimate reusable artifact. Matching provenance reuses it;
        # differing or unknown inputs warn and never regenerate over existing files.
        metrics = evaluate_features(blip_model, dataset, txt_processors["eval"])
        if not isinstance(metrics, dict):
            raise RuntimeError("TME native evaluator returned a non-dictionary result")

        results: list[ReplayResult] = []
        for category in CATEGORIES:
            category_metrics = metrics.get(category)
            if not isinstance(category_metrics, dict):
                raise RuntimeError(f"TME native evaluator omitted category metrics: {category}")
            recalls = {}
            for key in ("recall_at1", "recall_at10", "recall_at50"):
                try:
                    value = category_metrics[key]
                    finite = math.isfinite(value)
                except (KeyError, TypeError, ValueError, OverflowError):
                    finite = False
                if not finite:
                    raise RuntimeError(f"TME native evaluator returned invalid {key} for {category}")
                recalls[key] = float(value)
            results.append(ReplayResult(
                model_id="tme",
                checkpoint_id=variant["checkpoint_id"],
                category=category,
                protocol=PROTOCOL,
                r1=recalls["recall_at1"],
                r10=recalls["recall_at10"],
                r50=recalls["recall_at50"],
                complete=True,
                provenance={
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "checkpoint_training_noise_pct": variant.get("checkpoint_training_noise_pct"),
                    "reference_policy": "reference_included",
                    "checkpoint_selection_rule": "best_validation_checkpoint_at_sigma_0",
                    "feature_cache": str(feature_path),
                    **provenance,
                    "feature_cache_provenance": cache_provenance,
                    "feature_cache_unverified_inputs": differences if cache_provenance == "UNVERIFIED" else [],
                    "feature_gallery_order": "FashionIQ val split files, dress then shirt then toptee",
                    "runtime_assets": runtime_asset_evidence,
                },
            ))
    return results, macro_aggregate(results)


def evaluate_categories(checkout: Path, dataset_root: Path, checkpoint_root: Path,
                        categories: tuple[str, ...] = CATEGORIES,
                        checkpoint_id: str | None = None,
                        runtime_asset_evidence: list[dict[str, Any]] | None = None) -> tuple[list[ReplayResult], dict[str, Any]]:
    source = source_root_for(checkout)
    original_cwd = Path.cwd()
    os.chdir(source)
    try:
        return _evaluate_categories(checkout, dataset_root, checkpoint_root, categories,
                                    checkpoint_id, _prior_feature_provenance(),
                                    runtime_asset_evidence or [])
    finally:
        os.chdir(original_cwd)
def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint-id", type=str, default=None)
    parser.add_argument("--category", choices=CATEGORIES, action="append", dest="categories")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)

    selected = tuple(args.categories) if args.categories else CATEGORIES
    runtime_assets = _runtime_asset_check()
    blockers = preflight(args.source_root, args.dataset_root, args.checkpoint_root,
                         args.checkpoint_id, runtime_assets)
    if blockers:
        for blocker in blockers:
            print(f"[BLOCKED] {blocker}", file=sys.stderr)
        return 1
    if args.preflight_only:
        return 0

    results, aggregate = evaluate_categories(
        args.source_root, args.dataset_root, args.checkpoint_root,
        categories=selected, checkpoint_id=args.checkpoint_id,
        runtime_asset_evidence=runtime_assets[1])
    payload = {
        "model_id": "tme",
        "protocol": PROTOCOL,
        "results": [asdict(r) for r in results],
        "aggregate": aggregate,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
