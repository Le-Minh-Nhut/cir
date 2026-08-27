from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from backbones.fgclip2 import FGCLIP2_LARGE_MODEL_ID, FGCLIP2_LARGE_REVISION
from cache.features import (
    get_dense_features_by_ids,
    get_features_by_ids,
    get_text_features_with_global_by_sample_ids,
    load_dense_image_features,
    load_features,
    load_text_features,
    validate_feature_manifest,
)
from datasets.common import collate_cir_samples
from datasets.fashioniq import FashionIQDataset, load_correction_dict
from evaluation.entity_action_binding import evaluate_entity_action_binding
from models.entity_action_binding import EntityActionBindingCIR
from runtime import resolve_device, seed_everything
from training.entity_action_binding import load_entity_action_checkpoint

CATEGORIES = ("dress", "shirt", "toptee")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A8.0 relation-pair functional forensics")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/fashionIQ_dataset"))
    parser.add_argument("--cache-root", type=Path, default=Path("features/fashioniq/fgclip2-large"))
    parser.add_argument("--text-cache-subdir", default="text")
    parser.add_argument("--protocol", choices=("fashioniq_original", "fashioniq_val"), default="fashioniq_original")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--json-output", type=Path, default=Path("reports/a8_0_binding_diagnostics.json"))
    return parser.parse_args()


def _model_from_checkpoint(path: Path, device: torch.device) -> EntityActionBindingCIR:
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict) or not isinstance(raw.get("experiment_provenance"), dict):
        raise TypeError("A8.0 checkpoint is missing experiment_provenance")
    provenance = raw["experiment_provenance"]
    if provenance.get("architecture") != "fgclip2_shared_entity_action_binding":
        raise RuntimeError("Checkpoint is not an A8.0 Entity–Action Binding checkpoint")
    model = EntityActionBindingCIR(
        dim=int(provenance["dim"]),
        num_relations=int(provenance["num_relations"]),
        fusion_hidden_dim=int(provenance["fusion_hidden_dim"]),
        retrieval_temperature=float(provenance["retrieval_temperature"]),
        entity_action_temperature=float(provenance["entity_action_temperature"]),
    ).to(device)
    load_entity_action_checkpoint(model, path, map_location=device)
    model.eval()
    return model


@torch.no_grad()
def aggregate_representation_diagnostics(
    model: EntityActionBindingCIR,
    loaders: dict[str, DataLoader],
    *,
    global_features: torch.Tensor,
    global_index: dict[str, int],
    dense_cache,
    text_cache,
    device: torch.device,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    sample_count = 0
    for loader in loaders.values():
        for batch in loader:
            global_ref = get_features_by_ids(batch.reference_ids, global_features, global_index)
            dense, dense_mask = get_dense_features_by_ids(batch.reference_ids, dense_cache)
            states, _, content, text_global = get_text_features_with_global_by_sample_ids(
                batch.sample_ids, batch.modification_texts, text_cache
            )
            output = model(
                reference_global=global_ref[:, 0].to(device=device, dtype=torch.float32),
                reference_dense=dense.to(device),
                reference_dense_mask=dense_mask.to(device),
                text_global=text_global.to(device=device, dtype=torch.float32),
                text_states=states.to(device=device, dtype=torch.float32),
                text_content_mask=content.to(device=device, dtype=torch.bool),
            )
            diagnostics = model.diagnostics(
                output, dense_mask.to(device), content.to(device=device, dtype=torch.bool)
            )
            count = len(batch.sample_ids)
            for name, value in diagnostics.items():
                totals[name] = totals.get(name, 0.0) + float(value) * count
            sample_count += count
    return {name: value / sample_count for name, value in totals.items()}


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("Invalid batch-size/num-workers")
    seed_everything(42, deterministic=True)
    device = resolve_device(args.device)
    model = _model_from_checkpoint(args.checkpoint, device)

    annotation_root = args.dataset_root / "captions"
    correction_dicts = {
        category: load_correction_dict(annotation_root / f"correction_dict_{category}.json")
        for category in CATEGORIES
    }
    loaders: dict[str, DataLoader] = {}
    annotations: dict[str, list] = {}
    for category in CATEGORIES:
        dataset = FashionIQDataset(
            annotation_root,
            "val",
            [category],
            caption_policy="normalized_ordered_and",
            correction_dicts=correction_dicts,
            seed=42,
        )
        loaders[category] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_cir_samples,
        )
        annotations[category] = dataset.annotations

    global_dir = args.cache_root / "val" / "images"
    dense_dir = args.cache_root / "val" / "dense_images"
    text_dir = args.cache_root / "val" / args.text_cache_subdir
    for path, name in ((global_dir, "val/images"), (dense_dir, "val/dense_images"), (text_dir, "val/text")):
        from cache.features import load_feature_manifest
        manifest = load_feature_manifest(path)
        validate_feature_manifest(
            manifest,
            model_id=FGCLIP2_LARGE_MODEL_ID,
            revision=FGCLIP2_LARGE_REVISION,
            cache_name=name,
            correction_policy="fashioniq" if name == "val/text" else None,
        )
    globals_, global_index = load_features(global_dir)
    dense = load_dense_image_features(dense_dir)
    text = load_text_features(text_dir)
    if text.global_features is None:
        raise FileNotFoundError("A8.0 diagnostic requires text global.npy")

    common = {
        "protocol": args.protocol,
        "split_root": args.dataset_root / "image_splits",
        "split": "val",
        "global_features": globals_,
        "global_name_to_idx": global_index,
        "dense_cache": dense,
        "text_cache": text,
        "device": device,
        "gallery_batch_size": args.batch_size,
    }
    results: dict[str, dict[str, float]] = {}
    results["FULL"] = evaluate_entity_action_binding(model, loaders, annotations, **common)
    results["GLOBAL_ONLY"] = evaluate_entity_action_binding(
        model, loaders, annotations, variant="global_only", **common
    )
    for relation in range(model.num_relations):
        for variant, label in (("drop", "DROP"), ("single", "SINGLE"), ("repeat", "REPEAT")):
            results[f"{label}-{relation}"] = evaluate_entity_action_binding(
                model,
                loaders,
                annotations,
                variant=variant,
                relation_index=relation,
                **common,
            )
    full_mean = results["FULL"]["mean_recall"]
    ratios = {
        "best_single_full_ratio": max(
            results[f"SINGLE-{index}"]["mean_recall"] for index in range(model.num_relations)
        ) / max(full_mean, 1e-12),
        "best_repeat_full_ratio": max(
            results[f"REPEAT-{index}"]["mean_recall"] for index in range(model.num_relations)
        ) / max(full_mean, 1e-12),
    }
    representation = aggregate_representation_diagnostics(
        model,
        loaders,
        global_features=globals_,
        global_index=global_index,
        dense_cache=dense,
        text_cache=text,
        device=device,
    )
    report = {
        "checkpoint": str(args.checkpoint),
        "model_id": FGCLIP2_LARGE_MODEL_ID,
        "revision": FGCLIP2_LARGE_REVISION,
        "results": results,
        "ratios": ratios,
        "representation_diagnostics": representation,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    with args.json_output.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    for name, metrics in results.items():
        print(
            f"{name:12s} R@10={metrics['recall_at_10']:.2f} "
            f"R@50={metrics['recall_at_50']:.2f} mean={metrics['mean_recall']:.2f}"
        )
    print("best SINGLE/FULL:", ratios["best_single_full_ratio"])
    print("best REPEAT/FULL:", ratios["best_repeat_full_ratio"])
    print("Saved:", args.json_output)


if __name__ == "__main__":
    main()
