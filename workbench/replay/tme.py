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
import hashlib
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
    generated_artifact_parameters,
    model_by_id,
    preparation_contract_for_model,
    preparation_paths,
    required_runtime_assets,
)
from workbench.replay.common import (
    ReplayResult,
    fashioniq_gallery_digests,
    fashioniq_image_binding_blockers,
    isolated_cache_root,
    macro_aggregate,
    sha256_file,
    source_pin_status,
)

MODEL = model_by_id("tme")
CHECKPOINTS = {variant["checkpoint_id"]: variant for variant in MODEL["checkpoint_variants"]}
DEFAULT_CHECKPOINT = "fashioniq_0"
PROTOCOL = "fashioniq_original_split"


def _tme_source_id(category: str, triplet: dict[str, Any]) -> str:
    """Derive stable source identity from native annotation content, not row position."""
    content = json.dumps({key: triplet[key] for key in ("candidate", "target", "captions")},
                         sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return f"tme:{category}:{digest}"


def _install_tme_observer(model: Any, dataset: Any, capture_root: Path,
                          selected_queries: dict[str, set[str]] | set[str],
                          selected_gallery: dict[str, set[str]] | set[str],
                          invocation_id: str, started_at: str) -> Any:
    """Capture native tensors and exact score rows while streaming target ranks."""
    import hashlib
    import numpy as np
    import precompute_evaluation
    import torch

    original_metric = precompute_evaluation.compute_metric_features
    original_encode_image = model.encode_image
    original_inference = model.inference
    original_encode_fusion = model.encode_fusion
    projection_hook = None
    state: dict[str, Any] = {}
    import torch.nn.functional as functional

    def capture_projected_gallery(module: Any, inputs: Any, output: Any) -> None:
        if state.get("projected_written"):
            return
        names = state["gallery_names"]
        path = state["feature_root"] / f"{state['category']}.gallery-scoring.npy"
        projected = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float32,
            shape=(len(names), *tuple(output.shape[1:])))
        for start in range(0, len(output), 128):
            stop = min(start + 128, len(output))
            normalized = functional.normalize(output[start:stop].float(), dim=-1)
            projected[start:stop] = normalized.detach().cpu().numpy()
        projected.flush()
        state["gallery_scoring_features"] = projected
        state["projected_written"] = True

    projection_hook = model.vision_proj.register_forward_hook(capture_projected_gallery)

    original_argsort = torch.argsort
    def capture_argsort(input: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_argsort(input, *args, **kwargs)
        if (input.ndim == 2 and state.get("native_order") is not None and
                tuple(input.shape) == tuple(state["native_order"].shape)):
            state["native_order"][:] = result.detach().cpu().numpy()
            state["native_argsort_device"] = str(input.device)
        return result

    torch.argsort = capture_argsort
    def append_selected(key: str, ids: list[str], value: Any) -> None:
        rows = value.detach().cpu().numpy()
        if rows.ndim == 1 and len(ids) == 1:
            state["arrays"][key].append(rows)
            return
        if rows.ndim == 0 or rows.shape[0] != len(ids):
            raise ValueError(
                f"TME selected capture {state['category']}/{key} batch mismatch: "
                f"query_batch={len(ids)} shape={rows.shape} selected_ids={ids}")
        state["arrays"][key].extend(rows[index] for index in range(len(ids)))

    def metric(current_model: Any, current_dataset: Any, preprocessor: Any, category: str) -> Any:
        names = list(current_dataset.current_image_names)
        category_query_ids = {_tme_source_id(category, triplet)
                              for triplet in current_dataset.current_triplets}
        query_selection = selected_queries.get(category, set()) if isinstance(selected_queries, dict) else selected_queries
        gallery_selection = selected_gallery.get(category, set()) if isinstance(selected_gallery, dict) else selected_gallery
        category_selected_queries = query_selection & category_query_ids
        category_selected_gallery = gallery_selection & set(names)
        feature_root = capture_root / ".features"
        feature_root.mkdir(parents=True, exist_ok=True)
        state.update(category=category, gallery_names=names, gallery_cursor=0, query_cursor=0,
                     feature_root=feature_root, query_features=None, gallery_features=None,
                     gallery_scoring_features=None, projected_written=False,
                     selected_queries=category_selected_queries,
                     selected_gallery=category_selected_gallery, active_ids=[], query_rows=[],
                     arrays={key: [] for key in ("query_ids", "reference_ids", "target_ids",
                             "gallery_features", "target_ranks", "gallery_ids", "selected_query_ids",
                             "selected_scores", "selected_reference_features", "selected_fusion_features",
                             "selected_input_ids", "selected_attention_mask")},
                     selected_score_batches=[])

        state["native_order"] = np.lib.format.open_memmap(
            feature_root / f"{category}.native-order.npy", mode="w+", dtype=np.int64,
            shape=(len(current_dataset.current_triplets), len(names)))
        state["scores"] = np.lib.format.open_memmap(
            feature_root / f"{category}.scores.npy", mode="w+", dtype=np.float32,
            shape=(len(current_dataset.current_triplets), len(names)))
        result = original_metric(current_model, current_dataset, preprocessor, category)
        if state["gallery_cursor"] != len(names):
            raise RuntimeError(f"TME native observer gallery coverage mismatch: {category}")
        if state["query_cursor"] != len(current_dataset.current_triplets):
            raise RuntimeError(f"TME native observer query coverage mismatch: {category}")
        arrays = state["arrays"]
        category_queries = {query_id for query_id in state["selected_queries"]
                            if query_id.startswith(f"tme:{category}:")}
        if not category_queries <= set(arrays["query_ids"]):
            missing = sorted(category_queries - set(arrays["query_ids"]))
            raise RuntimeError(f"TME selected query IDs absent from native run: {missing}")
        if category_queries and not state["selected_gallery"] <= set(arrays["gallery_ids"]):
            missing = sorted(state["selected_gallery"] - set(arrays["gallery_ids"]))
            raise RuntimeError(f"TME selected gallery IDs absent from native run: {missing}")
        target_positions = {name: index for index, name in enumerate(names)}
        source_ranks = [int(np.flatnonzero(
            state["native_order"][index] == target_positions[target])[0]) + 1
            for index, target in enumerate(arrays["target_ids"])]
        arrays["target_ranks"][:] = source_ranks
        for row, rank in zip(state["query_rows"], source_ranks, strict=True):
            row["native_target_rank"] = rank
        state["native_order"].flush()
        state["scores"].flush()
        state["query_features"].flush()
        state["gallery_features"].flush()
        state["gallery_scoring_features"].flush()
        payload: dict[str, Any] = {
            "query_ids": np.asarray(arrays["query_ids"]),
            "gallery_ids": np.asarray(names),
            "query_features": state["query_features"],
            "gallery_features": state["gallery_scoring_features"],
            "scores": state["scores"],
            "native_order": state["native_order"],
            "rank_qids": np.asarray(arrays["query_ids"]),
            "ranks": np.asarray(arrays["target_ranks"], dtype=np.int64),
            "reference_ids": np.asarray(arrays["reference_ids"]),
            "metadata": np.asarray(json.dumps({"schema": "cir-tme-native-capture-v1",
                "invocation_id": invocation_id, "started_at": started_at, "category": category,
                "metrics": result, "query_rows": state["query_rows"],
                "selected_score_batches": state["selected_score_batches"],
                "gallery_order_sha256": hashlib.sha256(
                    json.dumps(names, separators=(",", ":")).encode()).hexdigest()}, sort_keys=True)),
        }
        for key in ("selected_query_ids", "selected_scores", "selected_reference_features",
                    "selected_fusion_features", "selected_input_ids", "selected_attention_mask"):
            values = arrays[key]
            if values:
                if len(values) != len(arrays["selected_query_ids"]):
                    raise ValueError(
                        f"TME selected capture {category}/{key} row count {len(values)} differs "
                        f"from selected query count {len(arrays['selected_query_ids'])}")
                payload[key] = np.stack(values) if key != "selected_query_ids" else np.asarray(values)
                if key != "selected_query_ids":
                    payload[f"{key}_row_ids"] = np.asarray(arrays["selected_query_ids"])
        from workbench.replay.clvc_tme_parity import _write_streaming_npz
        _write_streaming_npz(capture_root / f"{category}.npz", payload)
        return result

    def encode_image(images: Any) -> Any:
        first = state["gallery_cursor"]
        output = original_encode_image(images)
        names = state["gallery_names"][first:first + len(images)]
        if len(names) != len(images):
            raise RuntimeError("TME gallery feature batch exceeds native order")
        if state["gallery_features"] is None:
            shape = (len(state["gallery_names"]), *tuple(output.shape[1:]))
            path = state["feature_root"] / f"{state['category']}.gallery.npy"
            state["gallery_features"] = np.lib.format.open_memmap(
                path, mode="w+", dtype=np.float32, shape=shape)
        state["gallery_features"][first:first + len(output)] = output.detach().float().cpu().numpy()
        keep = [idx for idx, name in enumerate(names) if name in state["selected_gallery"]]
        if keep:
            state["arrays"]["gallery_ids"].extend(names[idx] for idx in keep)
            state["arrays"]["gallery_features"].append(output[keep].detach().cpu().numpy())
        state["gallery_cursor"] += len(images)
        return output

    def encode_fusion(reference: Any, tokens: Any) -> Any:
        ids = state["active_ids"]
        first = state["query_cursor"]
        output = original_encode_fusion(reference, tokens)
        if state["query_features"] is None:
            path = state["feature_root"] / f"{state['category']}.queries.npy"
            state["query_features"] = np.lib.format.open_memmap(
                path, mode="w+", dtype=np.float32,
                shape=(len(dataset.current_triplets), int(output.shape[-1])))
        state["query_features"][first:first + len(output)] = output.detach().float().cpu().numpy()
        keep = [idx for idx, sample_id in enumerate(ids) if sample_id in state["selected_queries"]]
        if keep:
            selected_ids = [ids[idx] for idx in keep]
            append_selected("selected_input_ids", selected_ids, tokens.input_ids[keep])
            append_selected("selected_attention_mask", selected_ids, tokens.attention_mask[keep])
            append_selected("selected_fusion_features", selected_ids, output[keep])
        return output
    def inference(references: Any, gallery: Any, captions: Any) -> Any:
        first = state["query_cursor"]
        triplets = dataset.current_triplets[first:first + len(captions)]
        category = state["category"]
        ids = [_tme_source_id(category, triplet) for triplet in triplets]
        if len(ids) != len(captions) or len(ids) != len(set(ids)):
            raise ValueError(f"TME native triplet identities are ambiguous: {category}")
        state["active_ids"] = ids
        scores = original_inference(references, gallery, captions)
        state["scores"][first:first + len(ids)] = scores.detach().float().cpu().numpy()
        for idx, triplet in enumerate(triplets):
            target, reference = triplet["target"], triplet["candidate"]
            sample_id = ids[idx]
            state["arrays"]["query_ids"].append(sample_id)
            state["arrays"]["reference_ids"].append(reference)
            state["arrays"]["target_ids"].append(target)
            state["arrays"]["target_ranks"].append(0)
            state["query_rows"].append({"sample_id": sample_id, "reference_id": reference,
                                        "native_processed_text": captions[idx],
                                        "captions": triplet["captions"]})
        keep = [idx for idx, sample_id in enumerate(ids) if sample_id in state["selected_queries"]]
        state["selected_score_batches"].append({"query_ids": ids,
            "selected_query_ids": [ids[idx] for idx in keep],
            "input_shape": list(scores.shape), "selected_shape": list(
                (scores.unsqueeze(0) if scores.ndim == 1 else scores)[keep].shape)})
        keep = [idx for idx, sample_id in enumerate(ids) if sample_id in state["selected_queries"]]
        if keep:
            selected_ids = [ids[idx] for idx in keep]
            state["arrays"]["selected_query_ids"].extend(selected_ids)
            score_rows = scores.unsqueeze(0) if scores.ndim == 1 else scores
            append_selected("selected_scores", selected_ids, score_rows[keep])
            append_selected("selected_reference_features", selected_ids,
                            references[keep])
        state["query_cursor"] += len(triplets)
        return scores

    precompute_evaluation.compute_metric_features = metric
    model.encode_image = encode_image
    model.inference = inference
    model.encode_fusion = encode_fusion

    def restore() -> None:
        precompute_evaluation.compute_metric_features = original_metric
        model.encode_image = original_encode_image
        model.inference = original_inference
        model.encode_fusion = original_encode_fusion
        torch.argsort = original_argsort
        if projection_hook is not None:
            projection_hook.remove()
    return restore
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
            # `location_type: directory` marks an asset that IS a directory (a Hugging Face hub repo
            # cache). Testing `is_file()` on it reported a correctly populated cache as "not a file",
            # which is the same class of mistake as pointing the check at the wrong root.
            if asset.get("location_type") == "directory":
                if not path.is_dir() or not any(path.iterdir()):
                    blockers.append(
                        f"TME runtime asset {asset['name']} is not a populated directory{detail}")
                    continue
                # A populated directory is not a usable cache. The shared cache check is the same
                # decision readiness makes, so the wrapper cannot call a cache ready that the
                # preflight refuses -- or vice versa, which is what happened when this branch
                # admitted any non-empty directory while the shared check required a loadable
                # revision snapshot.
                from workbench.backend.registry import cache_asset_blocker

                cache_reason = cache_asset_blocker(asset, status)
                if cache_reason:
                    blockers.append(
                        f"TME runtime asset {asset['name']} cache unusable{detail}: {cache_reason}")
                    continue
                evidence.append({"name": asset["name"], "status": state, "path": str(path),
                                 "verification": "cache layout verified; loaded by its consumer"})
                continue
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
            # Same admission decision readiness makes, so a file with the right size but no
            # plausible content cannot be treated as a usable weight by the run either.
            from workbench.backend.registry import asset_admission_status

            admission = asset_admission_status(asset, status)
            if admission == "SUSPECT_PAYLOAD":
                blockers.append(
                    f"TME runtime asset {asset['name']} content is not a plausible payload{detail}")
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
    # Runtime-aware: an adapted runtime may legitimately produce a different dtype than the
    # historical contract, and preflight's size lower bound must agree with what replay accepts.
    feature_parameters = generated_artifact_parameters(
        contract, "tme_precompute_features", MODEL)
    from workbench.backend.fashioniq_layout import image_extension_for_layout
    image_extension = image_extension_for_layout(MODEL["fashioniq_layout"], MODEL.get("fashioniq_image_extensions"))
    blockers.extend(f"TME native gallery input missing: {problem}"
                    for problem in fashioniq_image_binding_blockers(
                        dataset_root, feature_parameters["split"], image_source="images",
                        image_extension=image_extension))
    # The scope decides the root: this artifact is `evaluator_cwd`, so it resolves against the
    # CHECKOUT, not the dataset root. Passing the dataset root here is what made the shared preflight
    # and this check disagree about which path they were talking about.
    feature_root = preparation_paths(
        contract, dataset_root, "generated_artifacts",
        evaluator_root=source_root_for(checkout))[0]
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


def _prior_feature_provenance(config=None) -> dict[str, Any] | None:
    from workbench.backend.operator_config import resolve_config

    config = config if config is not None else resolve_config()
    contract = preparation_contract_for_model(MODEL)
    artifact = next(item for item in contract["generated_artifacts"]
                    if item.get("generator") == "tme_precompute_features")
    parameters = generated_artifact_parameters(contract, "tme_precompute_features", MODEL)
    receipt = (config.WORKBENCH_ARTIFACTS_ROOT / artifact["paths"][0]
               / parameters["receipt_filename"])
    try:
        declared = json.loads(receipt.read_text(encoding="utf-8"))
        if (declared.get("kind") != "workbench_tme_precompute_receipt"
                or declared.get("model_id") != MODEL["model_id"]
                or declared.get("generator") != artifact["generator"]
                or declared.get("source_commit") != MODEL["upstream_commit_sha"]
                or declared.get("dataset") != parameters["eval_dataset"]
                or declared.get("split") != parameters["split"]):
            raise ValueError("receipt does not match TME source or feature contract")
        dataset_root = config.FASHIONIQ_ROOT.resolve()
        inputs = declared["inputs"]
        from workbench.scripts.prepare_model import _tme_inputs
        current_inputs = _tme_inputs(config, contract, MODEL)
        if current_inputs is None or inputs != current_inputs:
            raise ValueError("TME generation inputs changed or receipt lacks image identity")
        gallery = inputs["gallery_images"]
        # The receipt records two different gallery identities, so the prior-run half carries both
        # under the names the replay half computes them with. `split_files` is the same quantity as
        # the replay side's `feature_split_files_sha256` (sha256 of each split JSON), and a receipt
        # that predates either key refuses below rather than silently comparing against `None`.
        split_files = inputs["split_files"]
        checkout = config.WORKBENCH_THIRD_PARTY_ROOT / MODEL["source_dir"]
        feature_root = preparation_paths(contract, dataset_root, "generated_artifacts",
                                         evaluator_root=checkout)[0]
        feature = feature_root / f"{parameters['split']}_images.pt"
        mapping = feature_root / f"{parameters['split']}_name_to_idx.json"
        recorded_files = declared["files"]
        if (sha256_file(feature) != recorded_files["features"]["sha256"]
                or sha256_file(mapping) != recorded_files["name_to_idx"]["sha256"]):
            raise ValueError("feature cache bytes changed")
        return {
            "dataset_root": str(dataset_root),
            "feature_split": parameters["split"],
            "feature_backbone": f"{MODEL_NAME}:{MODEL_TYPE}@{MODEL['upstream_commit_sha']}",
            "source_commit": MODEL["upstream_commit_sha"],
            "feature_cache_sha256": recorded_files["features"]["sha256"],
            "feature_name_to_idx_sha256": recorded_files["name_to_idx"]["sha256"],
            "feature_gallery_sha256": gallery,
            "feature_split_files_sha256": split_files,
            "feature_shape": declared["feature_shape"],
            "feature_dtype": declared["feature_dtype"],
        }
    except (KeyError, OSError, ValueError, TypeError):
        if receipt.exists():
            print("[WARNING] TME feature cache preparation receipt does not bind current inputs; "
                  "cache origin remains UNVERIFIED", file=sys.stderr)
        return None
def feature_cache_provenance(dataset_root: Path, image_extension: str, parameters: dict,
                             feature_path: Path, mapping_path: Path,
                             features, prior_provenance: dict[str, Any] | None,
                             ) -> tuple[dict[str, Any], str, list[str]]:
    """The replay half of the feature-cache provenance, and its verdict against the receipt half.

    Kept as one named function because the contract is pairwise: every key here has to mean the SAME
    quantity as the key of that name in `_prior_feature_provenance`, or the comparison below reports
    a difference that does not exist. Two identities of the gallery are therefore named for what they
    hash — the ordered IMAGE BYTES the cache was generated from, and the split JSON files that say
    which images are in the gallery — rather than sharing one name between two different digests.
    """
    provenance = {
        "dataset_root": str(dataset_root.resolve()),
        "feature_split": parameters["split"],
        "feature_backbone": f"{MODEL_NAME}:{MODEL_TYPE}@{MODEL['upstream_commit_sha']}",
        "source_commit": MODEL["upstream_commit_sha"],
        "feature_cache_sha256": sha256_file(feature_path),
        "feature_name_to_idx_sha256": sha256_file(mapping_path),
        # Both computed HERE, from the dataset, rather than copied from the receipt: copying one side
        # into the other would make every comparison a tautology.
        "feature_gallery_sha256": fashioniq_gallery_digests(dataset_root, parameters["split"], image_extension),
        "feature_split_files_sha256": {
            category: sha256_file(dataset_root / f"image_splits/split.{category}.val.json")
            for category in CATEGORIES
        },
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
    }
    differences = ([key for key in sorted(provenance) if prior_provenance.get(key) != provenance[key]]
                   if prior_provenance is not None else ["no prior run provenance"])
    verdict = ("VERIFIED_PRIOR_RUN"
               if prior_provenance is not None and not differences else "UNVERIFIED")
    return provenance, verdict, differences


def _load_feature_cache(feature_path: Path, mapping_path: Path):
    import torch

    features = torch.load(feature_path, map_location="cpu", weights_only=True, mmap=True)
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict):
        raise RuntimeError("TME feature cache name-to-index mapping is not a JSON object")
    return features, mapping

def _evaluate_categories(checkout: Path, dataset_root: Path, checkpoint_root: Path,
                         categories: tuple[str, ...], checkpoint_id: str | None,
                         prior_feature_provenance: dict[str, Any],
                         runtime_asset_evidence: list[dict[str, Any]], capture_root: Path | None,
                         selected_queries: dict[str, set[str]] | None,
                         selected_gallery: dict[str, set[str]] | None,
                         sample_queries: dict[str, set[str]] | None,
                         invocation_id: str | None) -> tuple[list[ReplayResult], dict[str, Any]]:
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
    parameters = generated_artifact_parameters(contract, "tme_precompute_features", MODEL)
    feature_root = preparation_paths(contract, dataset_root, "generated_artifacts",
                                     evaluator_root=source)[0]
    feature_path = feature_root / f"{parameters['split']}_images.pt"
    mapping_path = feature_root / f"{parameters['split']}_name_to_idx.json"
    if not feature_path.is_file() or not mapping_path.is_file():
        raise FileNotFoundError(
            f"TME native feature cache missing: {feature_path} and/or {mapping_path}")
    from workbench.backend.fashioniq_layout import image_extension_for_layout

    preprocess = targetpad_transform(target_ratio=1.25, dim=224)
    image_extension = image_extension_for_layout(MODEL["fashioniq_layout"], MODEL.get("fashioniq_image_extensions"))
    with isolated_cache_root(dataset_root, directory_names=("captions", "image_splits"),
                             image_source="images", image_extension=image_extension) as cache_root:
        tme_data.FashioniqPath = str(cache_root).rstrip("/") + "/"
        dataset = tme_data.get_dataset(DATASET_LABEL, preprocess, parameters["split"])
        features, mapping = _load_feature_cache(feature_path, mapping_path)
        dataset.load_image_features_tensor_and_dict(features, mapping)
        global_names = [
            name for category in CATEGORIES
            for name in json.loads((dataset_root / "image_splits" /
                                    f"split.{category}.val.json").read_text(encoding="utf-8"))
        ]
        expected_index = {name: index for index, name in enumerate(global_names)}
        if list(dataset.name_to_idx.items()) != list(expected_index.items()):
            raise RuntimeError(
                "TME feature cache name-to-index mapping does not exactly match the ordered "
                "FashionIQ global feature-cache input")
        features = dataset.images
        expected_shape = (len(global_names), parameters["tokens_per_image"], parameters["feature_width"])
        expected_dtype = {"float16": torch.float16, "float32": torch.float32,
                          "float64": torch.float64}[parameters["feature_dtype"]]
        if not isinstance(features, torch.Tensor) or tuple(features.shape) != expected_shape:
            raise RuntimeError(
                f"TME feature cache shape mismatch: expected {expected_shape}, "
                f"got {getattr(features, 'shape', None)}")
        if features.dtype != expected_dtype:
            raise RuntimeError(
                f"TME feature cache dtype mismatch: expected {expected_dtype}, got {features.dtype}")

        provenance, cache_provenance, differences = feature_cache_provenance(
            dataset_root, image_extension, parameters, feature_path, mapping_path,
            features, prior_feature_provenance)
        provenance["feature_cache_loading"] = "torch.load_mmap_cpu_weights_only_v1"
        if cache_provenance == "UNVERIFIED":
            print("[WARNING] TME feature cache provenance UNVERIFIED for this run: "
                  + ", ".join(differences), file=sys.stderr)

        # This large feature cache is a legitimate reusable artifact. Matching provenance reuses it;
        # differing or unknown inputs warn and never regenerate over existing files.
        if sample_queries is not None:
            original_set_category = dataset.set_specific_dress_type

            def set_sample_category(category: str) -> None:
                original_set_category(category)
                selected_category = (sample_queries.get(category)
                                     if isinstance(sample_queries, dict) else sample_queries)
                if selected_category is not None:
                    source_rows = list(dataset.current_triplets)
                    selected_rows = [triplet for triplet in source_rows
                                     if _tme_source_id(category, triplet) in selected_category]
                    if len(selected_rows) != len(selected_category):
                        raise ValueError(f"TME sample IDs differ from source queries for {category}")
                    dataset.current_triplets = selected_rows
                    if not dataset.current_triplets:
                        raise ValueError(f"TME sample has no queries for category {category}")

            dataset.set_specific_dress_type = set_sample_category
        restore_observer = None
        if capture_root is not None:
            if selected_queries is None or selected_gallery is None or not invocation_id:
                raise ValueError("TME capture requires selections and pre-run invocation ID")
            started_at = datetime.now(timezone.utc).isoformat()
            restore_observer = _install_tme_observer(
                blip_model, dataset, capture_root, selected_queries, selected_gallery,
                invocation_id, started_at)
        try:
            metrics = evaluate_features(blip_model, dataset, txt_processors["eval"])
        finally:
            if restore_observer is not None:
                restore_observer()
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
                        runtime_asset_evidence: list[dict[str, Any]] | None = None,
                        config=None, capture_root: Path | None = None,
                        selected_queries: dict[str, set[str]] | None = None,
                        selected_gallery: dict[str, set[str]] | None = None,
                        sample_queries: dict[str, set[str]] | None = None,
                        invocation_id: str | None = None) -> tuple[list[ReplayResult], dict[str, Any]]:
    from workbench.backend.operator_config import resolve_config

    capture_root = Path(capture_root).resolve() if capture_root is not None else None
    original_cwd = os.getcwd()
    source = source_root_for(checkout)
    os.chdir(source)
    try:
        return _evaluate_categories(checkout, dataset_root, checkpoint_root, categories,
                                    checkpoint_id, _prior_feature_provenance(config),
                                    runtime_asset_evidence or [], capture_root,
                                    selected_queries, selected_gallery, sample_queries,
                                    invocation_id)
    finally:
        os.chdir(original_cwd)
def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint-id", type=str, default=None)
    parser.add_argument("--capture-root", type=Path, default=None)
    parser.add_argument("--capture-selection", type=Path, default=None,
                        help="JSON category map of selected native query and gallery IDs")
    parser.add_argument("--invocation-id", type=str, default=None,
                        help="UUID generated before model execution for capture receipt binding")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--sample-queries", type=Path, default=None,
                        help="JSON array of native TME query IDs to evaluate over each full category gallery")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    capture_selection = None
    if (args.capture_root is None) != (args.capture_selection is None):
        parser.error("--capture-root and --capture-selection must be supplied together")
    if args.capture_root is not None and not args.invocation_id:
        parser.error("capture requires pre-run --invocation-id")
    if args.capture_selection is not None:
        try:
            capture_selection = json.loads(args.capture_selection.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            parser.error(f"invalid capture selection: {error}")
        if not isinstance(capture_selection, dict) or set(capture_selection) != set(CATEGORIES) or any(
                not isinstance(value, dict) or set(value) != {"queries", "gallery"}
                for value in capture_selection.values()):
            parser.error("capture selection must map every category to queries and gallery arrays")
        if any(not isinstance(capture_selection[category][key], list) or
               len(capture_selection[category][key]) > 32 or
               len(capture_selection[category][key]) != len(set(capture_selection[category][key])) or
               any(not isinstance(value, str) or not value for value in capture_selection[category][key])
               for category in CATEGORIES for key in ("queries", "gallery")):
            parser.error("capture selection arrays must contain at most 32 non-empty unique IDs")
    sample_queries = None
    if args.sample_queries is not None:
        try:
            sample_queries = json.loads(args.sample_queries.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            parser.error(f"invalid sample query selection: {error}")
        if isinstance(sample_queries, dict):
            if set(sample_queries) != set(CATEGORIES) or any(
                    not isinstance(values, list) or not values or
                    any(not isinstance(value, str) or not value for value in values) or
                    len(values) != len(set(values)) for values in sample_queries.values()):
                parser.error("sample query map must contain all categories with non-empty unique IDs")
            sample_queries = {category: set(values) for category, values in sample_queries.items()}
        else:
            parser.error("sample query selection must map all three categories to non-empty unique IDs")

    selected = CATEGORIES
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
        runtime_asset_evidence=runtime_assets[1], capture_root=args.capture_root,
        selected_queries=({category: set(capture_selection[category]["queries"])
                           for category in CATEGORIES} if capture_selection is not None else None),
        selected_gallery=({category: set(capture_selection[category]["gallery"])
                           for category in CATEGORIES} if capture_selection is not None else None),
        sample_queries=sample_queries, invocation_id=args.invocation_id)
    payload = {
        "model_id": "tme",
        "protocol": PROTOCOL,
        "results": [asdict(r) for r in results],
        "aggregate": aggregate,
        "invocation_id": args.invocation_id,
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
