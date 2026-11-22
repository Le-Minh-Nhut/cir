"""Convert same-invocation native LIMN rankings into a plain JSON observation payload."""
from __future__ import annotations

import os
import math
import json
import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend.fashioniq_layout import CATEGORIES
from workbench.backend.registry import model_by_id
from workbench.replay.limn_observer import ObservedCategory, ObservedQuery, cross_check_with_native, _image_ids, _raw_captions, assert_alignment

DEFAULT_TOP_K = 200
MODEL = model_by_id("limn")
LIMN_CHECKPOINTS = {item["checkpoint_id"].removeprefix("base_iter0_"): item
                    for item in MODEL["checkpoint_variants"]}


def _load_native_category(category: str, rankings_root: Path, dataset_root: Path,
                          top_k: int) -> ObservedCategory:
    receipt = json.loads((rankings_root / f"{category}.json").read_text())
    if not isinstance(receipt, dict):
        raise ValueError(f"{category}: native ranking receipt is not a JSON object")
    if receipt.get("capture_error"):
        raise ValueError(f"{category}: native ranking capture failed: {receipt['capture_error']}")
    expected_run_id = os.environ.get("WORKBENCH_LIMN_RUN_ID")
    if expected_run_id and receipt.get("run_id") != expected_run_id:
        raise ValueError(f"{category}: native ranking receipt run identity mismatch")
    if (receipt.get("schema_version") != 1
            or receipt.get("provenance") != "native_argsort_same_invocation"
            or receipt.get("category") != category):
        raise ValueError(f"{category}: invalid native ranking receipt identity")
    images = _image_ids(dataset_root, category)
    annotations = _raw_captions(dataset_root, category)
    rows = receipt["queries"]
    queries = [{"source_img_id": row["source_position"],
                "target_img_id": row["target_position"],
                "mod": {"str": row["model_input_text"]}} for row in rows]
    assert_alignment(category, queries, images, annotations, dataset_root)
    gallery_positions = receipt["gallery_positions"]
    gallery_ids = receipt["gallery_ids"]
    image_positions = {image: index for index, image in enumerate(images)}
    expected_gallery_ids = list(dict.fromkeys(
        image for record in annotations for image in (record["candidate"], record["target"])))
    expected_gallery = [image_positions[image] for image in expected_gallery_ids]
    if (gallery_positions != expected_gallery
            or gallery_ids != [images[position] for position in expected_gallery]
            or len(rows) != len(annotations)
            or receipt["gallery_size"] != len(gallery_ids)
            or receipt["eligible_count"] != len(gallery_ids) - 1
            or receipt["source_mask_value"] != 10e10):
        raise ValueError(f"{category}: invalid native gallery metadata")
    gallery_set = set(gallery_ids)
    observed = []
    for index, row in enumerate(rows):
        if row["annotation_index"] != index or row["reference_id"] != images[row["source_position"]] \
                or row["target_id"] != images[row["target_position"]] \
                or row["raw_captions"] != annotations[index]["captions"]:
            raise ValueError(f"{category}: native ranking query identity mismatch at {index}")
        top = row["top_results"]
        rank = row["target_rank"]
        source_rank = row["source_rank"]
        if (not isinstance(rank, int) or isinstance(rank, bool) or not 1 <= rank <= len(gallery_ids) - 1
                or not isinstance(source_rank, int) or isinstance(source_rank, bool)
                or source_rank != len(gallery_ids)
                or len(top) != min(top_k, len(gallery_ids) - 1)):
            raise ValueError(f"{category}: invalid native rank depth at {index}")
        top_ids = [item["image_id"] for item in top]
        top_scores = [item["score"] for item in top]
        top_set = set(top_ids)
        if (len(top_ids) != len(top_set) or row["reference_id"] in top_set
                or any(item["rank"] != position or item["image_id"] not in gallery_set
                       or not isinstance(item["score"], (int, float))
                       or not math.isfinite(item["score"])
                       for position, item in enumerate(top, 1))
                or any(left < right for left, right in zip(top_scores, top_scores[1:]))
                or (rank <= len(top) and top[rank - 1]["image_id"] != row["target_id"])
                or (rank > len(top) and row["target_id"] in top_ids)):
            raise ValueError(f"{category}: invalid native Top-K or target rank at {index}")
        observed.append(ObservedQuery(
            annotation_index=index, reference_id=row["reference_id"], target_id=row["target_id"],
            raw_captions=tuple(row["raw_captions"]), model_input_text=row["model_input_text"],
            target_rank=row["target_rank"],
            top_results=tuple((item["rank"], item["image_id"], item["score"]) for item in top),
            category=category))
    return ObservedCategory(category, len(gallery_ids), len(gallery_ids) - 1, tuple(observed))


def observation_payload(observed: list[ObservedCategory], mismatches: dict[str, tuple[str, ...]],
                        *, run_id: str, top_k: int) -> dict:
    """The interchange between the model interpreter and the publishing interpreter.

    Everything here is plain JSON-serialisable data. It exists so the observer never has to import
    the canonical schema: this side runs in the model's environment, which has torch/numpy but not
    pydantic, and importing pydantic there made every export fail on a clean LIMN environment.
    """
    categories = []
    for observation in observed:
        categories.append({
            "category": observation.category,
            "gallery_size": observation.gallery_size,
            "eligible_count": observation.eligible_count,
            "queries": [
                {
                    "annotation_index": query.annotation_index,
                    "reference_id": query.reference_id,
                    "target_id": query.target_id,
                    "raw_captions": list(query.raw_captions),
                    "model_input_text": query.model_input_text,
                    "target_rank": query.target_rank,
                    "top_results": [{"rank": rank, "image_id": image_id, "score": score}
                                    for rank, image_id, score in query.top_results],
                }
                for query in observation.queries
            ],
        })
    return {
        "artifact_type": "limn_observation_payload",
        "schema_version": 1,
        "model_id": "limn",
        "method_name": "LIMN",
        "checkpoint_id": "base_iter0_all_categories",
        "protocol_id": "fashioniq_val_split",
        "run_id": run_id,
        "top_k": top_k,
        "ranking_provenance": "native_argsort_same_invocation",
        "checkpoint_members": {
            category: {"filename": LIMN_CHECKPOINTS[category]["filename"],
                       "checkpoint_id": LIMN_CHECKPOINTS[category]["checkpoint_id"],
                       "sha256": LIMN_CHECKPOINTS[category]["expected_sha256"]}
            for category in CATEGORIES
        },
        "instrumentation_parity": ("INSTRUMENTATION_PARITY_MISMATCH" if mismatches
                                   else "INSTRUMENTATION_PARITY_VERIFIED"),
        "instrumentation_mismatches": {name: list(items) for name, items in mismatches.items()},
        "categories": categories,
    }


def export(rankings_root: Path, dataset_root: Path, results_root: Path, native_metrics: dict,
           top_k: int = DEFAULT_TOP_K, run_id: str | None = None,
           command_digest: str | None = None, environment_digest: str | None = None,
           planned_bundle_manifest_digest: str | None = None,
           runtime_adaptation: str | None = None) -> int:
    """Validate same-pass native receipts and stage a CPU-only observation payload."""
    observed: list[ObservedCategory] = []
    mismatches: dict[str, tuple[str, ...]] = {}
    for category in CATEGORIES:
        observation = _load_native_category(category, rankings_root, dataset_root, top_k)
        category_metrics = native_metrics.get(category)
        if not isinstance(category_metrics, dict):
            raise ValueError(f"{category}: native aggregate metrics missing")
        native = {f"{category}_{name}": value for name, value in category_metrics.items()}
        category_mismatches = tuple(cross_check_with_native(observation, native))
        observed.append(observation)
        if category_mismatches:
            mismatches[category] = category_mismatches
    identity = run_id or "dev-limn-base_iter0_all_categories-fashioniq_val_split"
    payload = observation_payload(observed, mismatches, run_id=identity, top_k=top_k)
    payload["ranking_provenance"] = "native_argsort_same_invocation"
    payload["command_digest"] = command_digest
    payload["environment_digest"] = environment_digest
    payload["planned_bundle_manifest_digest"] = planned_bundle_manifest_digest
    payload["runtime_adaptation"] = runtime_adaptation
    results_root.mkdir(parents=True, exist_ok=True)
    staged = results_root / f".observation-{identity}.payload.json"
    staged.write_text(json.dumps(payload), encoding="utf-8")
    print(json.dumps({"model_id": "limn", "run_id": identity,
                      "protocol": "fashioniq_val_split", "top_k": top_k,
                      "instrumentation_parity": payload["instrumentation_parity"],
                      "instrumentation_mismatches": payload["instrumentation_mismatches"],
                      "payload": str(staged),
                      "queries": sum(len(item["queries"]) for item in payload["categories"])},
                     separators=(",", ":")))
    return 0 if not mismatches else 1


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--rankings-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--native-metrics", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--command-digest", default=None)
    parser.add_argument("--environment-digest", default=None)
    parser.add_argument("--runtime-adaptation", default=None)
    parser.add_argument("--planned-bundle-manifest-digest", default=None)
    args = parser.parse_args(argv)
    try:
        native_metrics = json.loads(args.native_metrics.read_text())
        return export(args.rankings_root, args.dataset_root, args.results_root, native_metrics,
                      top_k=args.top_k, run_id=args.run_id,
                      command_digest=args.command_digest,
                      environment_digest=args.environment_digest,
                      runtime_adaptation=args.runtime_adaptation,
                      planned_bundle_manifest_digest=args.planned_bundle_manifest_digest)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"[BLOCKED] {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
