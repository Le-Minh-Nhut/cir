import argparse
import json
import string
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F


def load_cases(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def normalize_edit_label(text: str) -> str:
    punctuation_to_space = str.maketrans({character: " " for character in string.punctuation})
    return " ".join(text.lower().translate(punctuation_to_space).split())


def summarize(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().cpu()
    if values.numel() == 0:
        raise ValueError("Cannot summarize empty tensor")
    return {
        "mean": values.mean().item(),
        "median": values.median().item(),
        "std": values.std(unbiased=False).item(),
        "min": values.min().item(),
        "max": values.max().item(),
    }


def validate_artifact(artifact: dict) -> None:
    required_keys = (
        "sample_ids", "reference_ids", "target_ids", "categories", "q_full_pre_norm",
        "q_minus_1_pre_norm", "q_minus_2_pre_norm", "q_full", "q_minus_1", "q_minus_2",
    )
    for key in required_keys:
        if key not in artifact:
            raise KeyError(f"Missing artifact key: {key}")
    num_samples = len(artifact["sample_ids"])
    if num_samples == 0:
        raise ValueError("Artifact contains no samples")
    for key in ("reference_ids", "target_ids", "categories"):
        if len(artifact[key]) != num_samples:
            raise ValueError(f"{key} length mismatch")
    for key in (
        "q_full_pre_norm", "q_minus_1_pre_norm", "q_minus_2_pre_norm", "q_full", "q_minus_1",
        "q_minus_2",
    ):
        tensor = artifact[key]
        if tensor.ndim != 2:
            raise ValueError(f"{key} must be [N,D], got {tuple(tensor.shape)}")
        if tensor.shape[0] != num_samples:
            raise ValueError(f"{key} sample count mismatch")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{key} contains NaN or Inf")


def align_cases(artifact: dict, cases: list[dict]) -> list[dict]:
    case_by_id = {case["sample_id"]: case for case in cases}
    aligned = []
    for sample_id in artifact["sample_ids"]:
        if sample_id not in case_by_id:
            raise KeyError(f"Cannot find audit case for {sample_id}")
        aligned.append(case_by_id[sample_id])
    return aligned


def effect_metrics(
    q_full_pre: torch.Tensor,
    q_minus_pre: torch.Tensor,
    q_full: torch.Tensor,
    q_minus: torch.Tensor,
) -> dict:
    delta = q_full_pre - q_minus_pre
    delta_norm = delta.norm(dim=-1)
    full_norm = q_full_pre.norm(dim=-1)
    relative_effect_norm = delta_norm / full_norm.clamp_min(1e-8)
    full_minus_cosine = F.cosine_similarity(q_full, q_minus, dim=-1)
    cosine_drop = 1.0 - full_minus_cosine
    return {
        "delta_norm": summarize(delta_norm),
        "relative_effect_norm": summarize(relative_effect_norm),
        "full_minus_cosine": summarize(full_minus_cosine),
        "cosine_drop": summarize(cosine_drop),
        "near_zero_absolute_fraction": (delta_norm < 1e-6).float().mean().item(),
        "near_zero_relative_fraction": (relative_effect_norm < 1e-3).float().mean().item(),
    }


def same_edit_directional_consistency(effects: torch.Tensor, labels: list[str], min_group_count: int = 2) -> dict:
    if effects.ndim != 2:
        raise ValueError("effects must be [N,D]")
    if effects.shape[0] != len(labels):
        raise ValueError("effects/labels mismatch")
    norms = effects.norm(dim=-1)
    valid_mask = norms > 1e-8
    effects = effects[valid_mask]
    labels = [label for label, valid in zip(labels, valid_mask.tolist()) if valid]
    if len(labels) < 2:
        return {"status": "insufficient_data", "num_effects": len(labels)}
    directions = F.normalize(effects, dim=-1)
    counts = Counter(labels)
    allowed_labels = {
        label for label, count in counts.items() if label and count >= min_group_count
    }
    keep_indices = [index for index, label in enumerate(labels) if label in allowed_labels]
    if len(keep_indices) < 2:
        return {
            "status": "insufficient_repeated_edits",
            "num_effects": len(labels),
            "num_repeated_groups": len(allowed_labels),
        }
    directions = directions[keep_indices]
    kept_labels = [labels[index] for index in keep_indices]
    num_effects = len(kept_labels)
    total_pair_count = num_effects * (num_effects - 1) // 2
    summed_direction = directions.sum(dim=0)
    total_pair_sum = (summed_direction.pow(2).sum().item() - num_effects) / 2.0
    same_pair_count = 0
    same_pair_sum = 0.0
    group_sizes = {}
    for label in sorted(allowed_labels):
        indices = [
            index for index, current_label in enumerate(kept_labels) if current_label == label
        ]
        group = directions[indices]
        group_size = len(indices)
        group_sizes[label] = group_size
        pair_count = group_size * (group_size - 1) // 2
        if pair_count == 0:
            continue
        group_sum = group.sum(dim=0)
        pair_sum = (group_sum.pow(2).sum().item() - group_size) / 2.0
        same_pair_count += pair_count
        same_pair_sum += pair_sum
    different_pair_count = total_pair_count - same_pair_count
    different_pair_sum = total_pair_sum - same_pair_sum
    same_mean = None if same_pair_count == 0 else same_pair_sum / same_pair_count
    different_mean = (
        None if different_pair_count == 0 else different_pair_sum / different_pair_count
    )
    gap = None if same_mean is None or different_mean is None else same_mean - different_mean
    return {
        "status": "ok",
        "num_effects": num_effects,
        "num_repeated_groups": len(allowed_labels),
        "same_pair_count": same_pair_count,
        "different_pair_count": different_pair_count,
        "same_mean_cosine": same_mean,
        "different_mean_cosine": different_mean,
        "same_vs_different_gap": gap,
        "group_sizes": group_sizes,
    }


def build_report(
    artifact: dict, cases: list[dict], teacher_name: str, min_group_count: int
) -> dict:
    validate_artifact(artifact)
    aligned_cases = align_cases(artifact=artifact, cases=cases)
    q_full_pre = artifact["q_full_pre_norm"].float()
    q_minus_1_pre = artifact["q_minus_1_pre_norm"].float()
    q_minus_2_pre = artifact["q_minus_2_pre_norm"].float()
    q_full = artifact["q_full"].float()
    q_minus_1 = artifact["q_minus_1"].float()
    q_minus_2 = artifact["q_minus_2"].float()
    delta_1 = q_full_pre - q_minus_1_pre
    delta_2 = q_full_pre - q_minus_2_pre
    caption_1_labels = [normalize_edit_label(case["caption_1"]) for case in aligned_cases]
    caption_2_labels = [normalize_edit_label(case["caption_2"]) for case in aligned_cases]
    all_effects = torch.cat([delta_1, delta_2], dim=0)
    all_labels = caption_1_labels + caption_2_labels
    within_sample_effect_cosine = F.cosine_similarity(delta_1, delta_2, dim=-1)
    return {
        "teacher": teacher_name,
        "num_queries": len(artifact["sample_ids"]),
        "query_dimension": int(q_full.shape[-1]),
        "effect_1": effect_metrics(q_full_pre, q_minus_1_pre, q_full, q_minus_1),
        "effect_2": effect_metrics(q_full_pre, q_minus_2_pre, q_full, q_minus_2),
        "within_sample_effect_cosine": summarize(within_sample_effect_cosine),
        "same_edit_directional_consistency": same_edit_directional_consistency(
            effects=all_effects, labels=all_labels, min_group_count=min_group_count
        ),
        "retrieval_necessity": {
            "status": "not_available",
            "reason": "Current smoke artifact does not contain native target scores or ranks.",
        },
        "compound_compositionality": {
            "status": "not_scored",
            "reason": "Compound metric requires a scientifically defined reference/null baseline.",
        },
        "counterfactual_stability": {
            "status": "not_available",
            "reason": "Current audit only uses caption deletion as the counterfactual "
            "intervention.",
        },
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="QuRe/TME smoke.pt")
    parser.add_argument(
        "--cases", type=Path, default=Path("teacher/audit/fashioniq_val_cases.json")
    )
    parser.add_argument("--teacher-name", type=str, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-group-count", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    artifact = torch.load(args.input, map_location="cpu")
    cases = load_cases(args.cases)
    report = build_report(
        artifact=artifact,
        cases=cases,
        teacher_name=args.teacher_name,
        min_group_count=args.min_group_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    print()
    print(f"=== {args.teacher_name} geometry audit ===")
    effect_1 = report["effect_1"]
    effect_2 = report["effect_2"]
    same_edit = report["same_edit_directional_consistency"]
    print("queries:", report["num_queries"])
    print("mean ||delta_1||:", effect_1["delta_norm"]["mean"])
    print("mean ||delta_2||:", effect_2["delta_norm"]["mean"])
    print("mean cosine drop 1:", effect_1["cosine_drop"]["mean"])
    print("mean cosine drop 2:", effect_2["cosine_drop"]["mean"])
    print("same-edit status:", same_edit["status"])
    if same_edit["status"] == "ok":
        print("same cosine:", same_edit["same_mean_cosine"])
        print("different cosine:", same_edit["different_mean_cosine"])
        print("same-different gap:", same_edit["same_vs_different_gap"])
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
