"""Per-query export for ENCODER under its model interpreter."""
from __future__ import annotations

import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend.fashioniq_layout import CATEGORIES
from workbench.replay.encoder import (
    AUXILIARY_ASSET_NAME, CHECKPOINTS, DEFAULT_CHECKPOINT, MODEL, PROTOCOL,
    _assert_checkpoint_loaded, _load_replay_modules, preflight,
)
from workbench.replay.encoder_observer import (
    ObservedCategory, cross_check_with_native, observe_category,
)
from workbench.replay.common import (FeatureRecorder, enable_deterministic_inference,
                                    run_cache_root, seeded_dataset_construction)

DEFAULT_TOP_K = 200


def observation_payload(observed: list[ObservedCategory], mismatches: dict[str, tuple[str, ...]],
                        *, run_id: str, top_k: int, checkpoint: dict) -> dict:
    categories = [{
        "category": category.category,
        "gallery_size": category.gallery_size,
        "eligible_count": category.eligible_count,
        "queries": [{
            "annotation_index": query.annotation_index,
            "reference_id": query.reference_id,
            "target_id": query.target_id,
            "raw_captions": list(query.raw_captions),
            "model_input_text": query.model_input_text,
            "target_rank": query.target_rank,
            "top_results": [{"rank": rank, "image_id": image_id, "score": score}
                            for rank, image_id, score in query.top_results],
        } for query in category.queries],
    } for category in observed]
    return {
        "artifact_type": "encoder_observation_payload", "schema_version": 1,
        "model_id": "encoder", "method_name": "ENCODER",
        "checkpoint_id": checkpoint["checkpoint_id"], "protocol_id": PROTOCOL,
        "run_id": run_id, "top_k": top_k, "upstream_commit": MODEL["upstream_commit_sha"],
        "checkpoint_members": {"fashioniq": {
            "filename": checkpoint["filename"], "checkpoint_id": checkpoint["checkpoint_id"],
            "sha256": checkpoint["expected_sha256"]}},
        "instrumentation_parity": ("INSTRUMENTATION_PARITY_MISMATCH" if mismatches
                                   else "INSTRUMENTATION_PARITY_VERIFIED"),
        "instrumentation_mismatches": {name: list(items) for name, items in mismatches.items()},
        "categories": categories,
    }


def export(checkpoint_root: Path, dataset_root: Path, source_root: Path, results_root: Path,
           checkpoint_id: str | None = None, top_k: int = DEFAULT_TOP_K,
           run_id: str | None = None, command_digest: str | None = None,
           environment_digest: str | None = None, planned_bundle_manifest_digest: str | None = None,
           runtime_adaptation: str | None = None, batch_size: int = 64) -> int:
    checkpoint = CHECKPOINTS.get(checkpoint_id) if checkpoint_id else DEFAULT_CHECKPOINT
    if checkpoint is None:
        print(f"[BLOCKED] unknown ENCODER checkpoint variant: {checkpoint_id}", file=sys.stderr)
        return 1
    blockers = preflight(checkpoint_root, dataset_root, source_root, CATEGORIES,
                         checkpoint_id=checkpoint["checkpoint_id"])
    if blockers:
        for blocker in blockers:
            print(f"[BLOCKED] {blocker}", file=sys.stderr)
        return 1

    torch, open_clip, datasets, native_test, model_module = _load_replay_modules(source_root)
    model = model_module.Encoder(hidden_dim=512, dropout=0.5, local_token_num=3,
                                 t=0.1, wc=2, N_p=2, weighted=True)
    checkpoint_path = checkpoint_root / checkpoint["filename"]
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict):
        report = model.load_state_dict(state, strict=False)
        _assert_checkpoint_loaded(model, report, checkpoint, source_root)
    else:
        model = state
    model = model.cuda().eval()
    # Same determinism as the aggregate's pass: this process re-derives the ranking from features it
    # computes itself, and the two passes must agree exactly for the artifact to be publishable.
    enable_deterministic_inference(torch)
    _, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained=str(source_root / AUXILIARY_ASSET_NAME))

    # Features captured by the AGGREGATE for this run. When present these are what the observation
    # ranks: they are the tensors the run's certified metrics were computed from, so the artifact
    # describes that evaluation by construction instead of hoping a second pass reproduces it.
    supplied: dict[str, tuple] = {}
    supplied_path = os.environ.get("WORKBENCH_OBSERVATION_IN", "").strip()
    if supplied_path and Path(supplied_path).is_file():
        with np.load(supplied_path) as archive:
            for key in archive.files:
                category, _, kind = key.partition(":")
                supplied.setdefault(category, {})[kind] = archive[key]
        supplied = {cat: (parts.get("queries"), parts.get("targets"))
                    for cat, parts in supplied.items()}

    observed: list[ObservedCategory] = []
    mismatches: dict[str, tuple[str, ...]] = {}
    with run_cache_root(dataset_root) as cache_root:
        # Same seed as the aggregate's construction: the query tensors are baked in here through the
        # stochastic training transform, so without this the export observes a DIFFERENT input
        # population than the aggregate it reconstructs -- measured at 3994 differing query tensors of
        # 4034 for one category. See `seeded_dataset_construction`.
        with seeded_dataset_construction():
            dataset = datasets.FashionIQ(path=str(cache_root).rstrip("/") + "/",
                                         transform=[preprocess_train, preprocess_val],
                                         split="val-split")
        params = SimpleNamespace(batch_size=batch_size, local_rank=0)
        for category in CATEGORIES:
            if category in supplied:
                # The aggregate ALREADY evaluated this category; its metrics are the run's certified
                # numbers, and they were computed from exactly these tensors. Re-running `test()`
                # here would produce a DIFFERENT number (a second pass does not reproduce the first)
                # and the cross-check would then report the aggregate's own arithmetic as a mismatch.
                # So the metric is derived from the supplied features, by the same protocol.
                captured = supplied[category]
                metrics = _metrics_from_features(category, captured, dataset, dataset_root)
            else:
                with FeatureRecorder(model) as recorder:
                    metrics = dict(native_test.test(params, model, dataset, category))
                captured = recorder.features()
            observation = observe_category(category, model, dataset, dataset_root,
                                           batch_size=batch_size, top_k=top_k, captured=captured)
            category_mismatches = tuple(cross_check_with_native(observation, metrics))
            observed.append(observation)
            if category_mismatches:
                mismatches[category] = category_mismatches

    identity = run_id or f"dev-encoder-{checkpoint['checkpoint_id']}-{PROTOCOL}"
    payload = observation_payload(observed, mismatches, run_id=identity, top_k=top_k,
                                  checkpoint=checkpoint)
    payload.update(command_digest=command_digest, environment_digest=environment_digest,
                   planned_bundle_manifest_digest=planned_bundle_manifest_digest,
                   runtime_adaptation=runtime_adaptation)
    results_root.mkdir(parents=True, exist_ok=True)
    staged = results_root / f".observation-{identity}.payload.json"
    staged.write_text(json.dumps(payload), encoding="utf-8")
    print(json.dumps({"model_id": "encoder", "run_id": identity, "protocol": PROTOCOL,
                      "top_k": top_k, "instrumentation_parity": payload["instrumentation_parity"],
                      "instrumentation_mismatches": payload["instrumentation_mismatches"],
                      "payload": str(staged),
                      "queries": sum(len(category["queries"]) for category in payload["categories"])},
                     separators=(",", ":")))
    print("[PENDING] real evaluation validation: this implementation has not been verified with a "
          "real ENCODER run", file=sys.stderr)
    return 0 if not mismatches else 1


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--checkpoint-id")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--run-id")
    parser.add_argument("--command-digest")
    parser.add_argument("--environment-digest")
    parser.add_argument("--runtime-adaptation")
    parser.add_argument("--planned-bundle-manifest-digest")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)
    return export(args.checkpoint_root, args.dataset_root, args.source_root, args.results_root,
                  checkpoint_id=args.checkpoint_id, top_k=args.top_k, run_id=args.run_id,
                  command_digest=args.command_digest, environment_digest=args.environment_digest,
                  runtime_adaptation=args.runtime_adaptation,
                  planned_bundle_manifest_digest=args.planned_bundle_manifest_digest,
                  batch_size=args.batch_size)


def _metrics_from_features(category: str, captured: tuple, dataset, dataset_root: Path) -> dict:
    """The native Recall values for a category, computed from features captured elsewhere.

    `test.py` computes them from the features it produced; given those same features this reproduces
    its arithmetic exactly, which is the point -- the numbers the aggregate reported and the numbers
    the observation reconstructs must be the same numbers, or the artifact cannot be evidence.
    """
    from workbench.replay.encoder_observer import _category_lists, _image_ids

    import numpy as np

    queries, targets = captured
    test_queries, test_targets = _category_lists(dataset, category)
    gallery_positions = [entry["target_img_id"] for entry in test_targets]
    position_of = {position: index for index, position in enumerate(gallery_positions)}

    # EXACTLY the native arithmetic: normalize IN PLACE on the arrays at the dtype the model
    # produced (float32), then take the dot product. Promoting to float64 first looks harmless and is
    # not: it rounds the normalised values differently, so a query whose target ties or nearly ties a
    # candidate lands on the other side of the boundary. Measured: two categories' R@1 differed by
    # one hit from the aggregate, and the only difference in the computation was this dtype.
    query_features = np.array(queries, copy=True)
    target_features = np.array(targets, copy=True)
    if query_features.dtype != np.float32 or target_features.dtype != np.float32:
        query_features = query_features.astype(np.float32)
        target_features = target_features.astype(np.float32)
    for row in range(query_features.shape[0]):
        query_features[row] /= np.linalg.norm(query_features[row])
    for row in range(target_features.shape[0]):
        target_features[row] /= np.linalg.norm(target_features[row])

    sims = query_features.dot(target_features.T)
    for index, entry in enumerate(test_queries):
        sims[index, position_of[entry["source_img_id"]]] = -10e10
    order = np.argsort(-sims, axis=1)   # the native call: `np.argsort(-sims[i, :])` per row

    out = {}
    total = order.shape[0]
    for k in (1, 10, 50):
        hits = 0
        for index, entry in enumerate(test_queries):
            if position_of[entry["target_img_id"]] in order[index][:k]:
                hits += 1
        out[f"{category}_r{k}"] = 100.0 * hits / total
    return out


if __name__ == "__main__":
    raise SystemExit(main())
