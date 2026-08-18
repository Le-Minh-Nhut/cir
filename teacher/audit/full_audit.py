from __future__ import annotations

import argparse
import gc
import json
import math
import random
import string
import subprocess
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from PIL import Image

CATEGORIES = ("dress", "shirt", "toptee")
QUERY_VARIANTS = ("full", "minus_1", "minus_2", "swap", "null")
EPS = 1e-8

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

def _ensure_repo_on_path() -> None:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

def load_cases(path: Path, limit: int | None = None) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        cases = json.load(file)
    if limit is None or limit <= 0:
        return cases
    limited = []
    for category in CATEGORIES:
        category_cases = [case for case in cases if case["category"] == category]
        limited.extend(category_cases[:limit])
    return limited

def load_split_ids(split_root: Path, category: str) -> list[str]:
    path = split_root / f"split.{category}.val.json"
    with path.open("r", encoding="utf-8") as file:
        image_ids = json.load(file)
    if not image_ids:
        raise ValueError(f"Empty FashionIQ gallery: {path}")
    return image_ids

def build_pair_union_gallery_ids(category_cases: list[dict]) -> list[str]:
    """Reproduce ENCODER's upstream FashionIQ val-split gallery."""
    gallery_ids: list[str] = []
    seen: set[str] = set()
    for case in category_cases:
        for image_id in (case["reference_id"], case["target_id"]):
            if image_id not in seen:
                seen.add(image_id)
                gallery_ids.append(image_id)
    if not gallery_ids:
        raise ValueError("Cannot build an empty pair-union gallery")
    return gallery_ids

def resolve_image_path(image_root: Path, image_id: str, category: str) -> Path:
    candidates: list[Path] = []
    for ext in (".jpg", ".png", ".jpeg"):
        candidates.append(image_root / category / f"{image_id}{ext}")
    for ext in (".jpg", ".png", ".jpeg"):
        candidates.append(image_root / f"{image_id}{ext}")
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find FashionIQ image id={image_id!r}, category={category!r} under {image_root}")

def load_image_batch(
    image_ids: list[str],
    category: str,
    image_root: Path,
    preprocess,
) -> torch.Tensor:
    tensors = []
    for image_id in image_ids:
        path = resolve_image_path(image_root, image_id, category)
        with Image.open(path) as image:
            tensors.append(preprocess(image.convert("RGB")))
    return torch.stack(tensors)

def load_case_image_batch(
    cases: list[dict],
    image_root: Path,
    preprocess,
) -> torch.Tensor:
    tensors = []
    for case in cases:
        path = resolve_image_path(
            image_root=image_root,
            image_id=case["reference_id"],
            category=case["category"],
        )
        with Image.open(path) as image:
            tensors.append(preprocess(image.convert("RGB")))
    return torch.stack(tensors)

def compose_single_caption(caption: str) -> str:
    return caption.strip(".?, ").capitalize()

def compose_ordered(caption_1: str, caption_2: str) -> str:
    return f"{compose_single_caption(caption_1)} and {caption_2.strip('.?, ')}"

def compose_swapped(caption_1: str, caption_2: str) -> str:
    return f"{compose_single_caption(caption_2)} and {caption_1.strip('.?, ')}"

def normalize_edit_label(text: str) -> str:
    punctuation_to_space = str.maketrans({character: " " for character in string.punctuation})
    return " ".join(text.lower().translate(punctuation_to_space).split())

def validate_cases(cases: list[dict]) -> None:
    required = {
        "sample_id",
        "category",
        "reference_id",
        "target_id",
        "caption_1",
        "caption_2",
        "full_text",
        "minus_1_text",
        "minus_2_text",
    }
    if not cases:
        raise ValueError("Audit case list is empty")
    seen = set()
    category_counts = Counter()
    for index, case in enumerate(cases):
        missing = required.difference(case)
        if missing:
            raise KeyError(f"Audit case {index} missing keys: {sorted(missing)}")
        sample_id = case["sample_id"]
        if sample_id in seen:
            raise ValueError(f"Duplicate audit sample_id: {sample_id}")
        seen.add(sample_id)
        category = case["category"]
        if category not in CATEGORIES:
            raise ValueError(f"Invalid FashionIQ category in {sample_id}: {category}")
        category_counts[category] += 1
        if not case["reference_id"] or not case["target_id"]:
            raise ValueError(f"Missing reference/target ID in {sample_id}")
        expected_full = compose_ordered(case["caption_1"], case["caption_2"])
        expected_minus_1 = compose_single_caption(case["caption_2"])
        expected_minus_2 = compose_single_caption(case["caption_1"])
        if case["full_text"] != expected_full:
            raise ValueError(f"{sample_id}: full_text violates locked ordered_and protocol")
        if case["minus_1_text"] != expected_minus_1:
            raise ValueError(f"{sample_id}: minus_1_text mismatch")
        if case["minus_2_text"] != expected_minus_2:
            raise ValueError(f"{sample_id}: minus_2_text mismatch")
    missing_categories = [category for category in CATEGORIES if category_counts[category] == 0]
    if missing_categories:
        raise ValueError(f"Audit cases missing categories: {missing_categories}")

def summarize(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().cpu().reshape(-1)
    if values.numel() == 0:
        raise ValueError("Cannot summarize an empty tensor")
    return {
        "mean": values.mean().item(),
        "median": values.median().item(),
        "std": values.std(unbiased=False).item(),
        "min": values.min().item(),
        "max": values.max().item(),
    }

def effect_metrics(
    q_full_pre: torch.Tensor,
    q_minus_pre: torch.Tensor,
    q_full: torch.Tensor,
    q_minus: torch.Tensor,
) -> dict:
    delta = q_full_pre - q_minus_pre
    delta_norm = delta.norm(dim=-1)
    full_norm = q_full_pre.norm(dim=-1)
    relative_effect_norm = delta_norm / full_norm.clamp_min(EPS)
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

def pairwise_mean_cosine(directions: torch.Tensor) -> float | None:
    n = directions.shape[0]
    if n < 2:
        return None
    summed = directions.sum(dim=0)
    pair_sum = (summed.pow(2).sum().item() - n) / 2.0
    pair_count = n * (n - 1) // 2
    return pair_sum / pair_count

def pair_weighted_same_edit_consistency(
    effects: torch.Tensor,
    labels: list[str],
    min_group_count: int,
) -> dict:
    if effects.ndim != 2 or effects.shape[0] != len(labels):
        raise ValueError("effects/labels shape mismatch")
    norms = effects.norm(dim=-1)
    valid = norms > EPS
    effects = effects[valid]
    labels = [label for label, keep in zip(labels, valid.tolist()) if keep]
    counts = Counter(labels)
    allowed = {label for label, count in counts.items() if label and count >= min_group_count}
    keep_indices = [i for i, label in enumerate(labels) if label in allowed]
    if len(keep_indices) < 2:
        return {
            "status": "insufficient_repeated_edits",
            "num_effects": len(keep_indices),
            "num_repeated_groups": len(allowed),
        }
    directions = F.normalize(effects[keep_indices], dim=-1)
    kept_labels = [labels[i] for i in keep_indices]
    n = len(kept_labels)
    total_pairs = n * (n - 1) // 2
    total_sum = (directions.sum(0).pow(2).sum().item() - n) / 2.0
    same_pairs = 0
    same_sum = 0.0
    group_sizes = {}
    for label in sorted(allowed):
        idx = [i for i, current in enumerate(kept_labels) if current == label]
        group = directions[idx]
        group_sizes[label] = len(idx)
        pair_count = len(idx) * (len(idx) - 1) // 2
        if pair_count == 0:
            continue
        pair_sum = (group.sum(0).pow(2).sum().item() - len(idx)) / 2.0
        same_pairs += pair_count
        same_sum += pair_sum
    diff_pairs = total_pairs - same_pairs
    diff_sum = total_sum - same_sum
    same_mean = same_sum / same_pairs if same_pairs else None
    diff_mean = diff_sum / diff_pairs if diff_pairs else None
    gap = None if same_mean is None or diff_mean is None else same_mean - diff_mean
    return {
        "status": "ok",
        "num_effects": n,
        "num_repeated_groups": len(allowed),
        "same_pair_count": same_pairs,
        "different_pair_count": diff_pairs,
        "same_mean_cosine": same_mean,
        "different_mean_cosine": diff_mean,
        "same_vs_different_gap": gap,
        "group_sizes": group_sizes,
    }

def balanced_same_edit_consistency(
    effects: torch.Tensor,
    labels: list[str],
    min_group_count: int,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    """
    Edit-balanced directional consistency with explicit coverage accounting.

    The statistic asks whether repeated *exact caption phrases* induce more
    similar effect directions than other repeated phrases. It does NOT assume
    one universal edit vector: reference-conditioned effects are allowed.

    same_g and different_g are both averages of pairwise cosine similarities
    between unit effect directions. Cross-group means are computed exactly by
    dot(mean_direction_g, mean_direction_h) WITHOUT re-normalizing means.

    Numerically zero effects cannot define a direction, so they are excluded
    from cosine estimation. Their exclusion is reported explicitly; otherwise
    a teacher could look artificially clean by producing zero effects on hard
    cases and having those cases silently disappear.
    """
    if effects.ndim != 2 or effects.shape[0] != len(labels):
        raise ValueError("effects/labels shape mismatch")
    raw_labels = list(labels)
    raw_counts = Counter(raw_labels)
    raw_allowed = {label for label, count in raw_counts.items() if label and count >= min_group_count}
    raw_eligible_effects = sum(raw_counts[label] for label in raw_allowed)
    norms = effects.norm(dim=-1)
    valid = norms > EPS
    valid_fraction = valid.float().mean().item()
    effects = effects[valid]
    labels = [label for label, keep in zip(labels, valid.tolist()) if keep]
    directions = F.normalize(effects, dim=-1)
    counts = Counter(labels)
    allowed = sorted(label for label, count in counts.items() if label and count >= min_group_count)
    common_meta = {
        "min_group_count": int(min_group_count),
        "num_effects_raw": len(raw_labels),
        "num_effects_direction_valid": int(valid.sum().item()),
        "direction_valid_fraction": valid_fraction,
        "num_repeated_groups_before_zero_filter": len(raw_allowed),
        "num_repeated_groups_after_zero_filter": len(allowed),
        "repeated_group_effect_coverage_before_zero_filter": (raw_eligible_effects / max(len(raw_labels), 1)),
    }
    if len(allowed) < 2:
        return {
            "status": "insufficient_repeated_edits",
            "num_repeated_groups": len(allowed),
            **common_meta,
        }
    groups = {}
    means = {}
    intra = {}
    for label in allowed:
        idx = [i for i, current in enumerate(labels) if current == label]
        group = directions[idx]
        value = pairwise_mean_cosine(group)
        if value is None:
            continue
        groups[label] = group
        means[label] = group.mean(dim=0)
        intra[label] = value
    usable = sorted(intra)
    if len(usable) < 2:
        return {
            "status": "insufficient_repeated_edits",
            "num_repeated_groups": len(usable),
            **common_meta,
        }
    rows = []
    gaps = []
    used_effects = 0
    for label in usable:
        cross = torch.stack([torch.dot(means[label], means[other]) for other in usable if other != label])
        different = cross.mean().item()
        gap = intra[label] - different
        gaps.append(gap)
        valid_count = int(groups[label].shape[0])
        raw_count = int(raw_counts[label])
        used_effects += valid_count
        rows.append(
            {
                "label": label,
                "raw_count": raw_count,
                "valid_direction_count": valid_count,
                "valid_direction_fraction": valid_count / raw_count,
                "same_pairwise_cosine": intra[label],
                "different_pairwise_cosine_macro_other_groups": different,
                "gap": gap,
            }
        )
    gap_tensor = torch.tensor(gaps, dtype=torch.float32)
    same_tensor = torch.tensor(
        [row["same_pairwise_cosine"] for row in rows],
        dtype=torch.float32,
    )
    diff_tensor = torch.tensor(
        [row["different_pairwise_cosine_macro_other_groups"] for row in rows],
        dtype=torch.float32,
    )
    generator = torch.Generator().manual_seed(seed)
    boot = []
    n_groups = len(rows)
    for _ in range(bootstrap_samples):
        sample_idx = torch.randint(0, n_groups, (n_groups,), generator=generator)
        boot.append(gap_tensor[sample_idx].mean())
    boot_tensor = torch.stack(boot)
    return {
        "status": "ok",
        "label_definition": ("normalized exact FashionIQ caption string; compound captions remain distinct exact-phrase groups"),
        **common_meta,
        "num_repeated_groups": n_groups,
        "used_effect_fraction": used_effects / max(len(raw_labels), 1),
        "macro_same_cosine": same_tensor.mean().item(),
        "macro_different_cosine": diff_tensor.mean().item(),
        "macro_same_vs_different_gap": gap_tensor.mean().item(),
        "median_group_gap": gap_tensor.median().item(),
        "positive_gap_fraction": (gap_tensor > 0).float().mean().item(),
        "bootstrap_95_interval_approx": [
            torch.quantile(boot_tensor, 0.025).item(),
            torch.quantile(boot_tensor, 0.975).item(),
        ],
        "bootstrap_unit": "edit_group",
        "bootstrap_caveat": ("Robustness interval only: per-group gaps share the other-group baseline, so edit groups are not fully independent."),
        "groups": rows,
    }

def compound_compositionality(
    q_null_pre: torch.Tensor,
    q_single_1_pre: torch.Tensor,
    q_single_2_pre: torch.Tensor,
    q_full_pre: torch.Tensor,
) -> dict:
    """
    Operational additive-composition diagnostic.

    q_null is the same reference image with an empty modification string.
    This is a standardized operational baseline, not a claim that empty text
    is an in-distribution semantic identity for every teacher.
    """
    e1 = q_single_1_pre - q_null_pre
    e2 = q_single_2_pre - q_null_pre
    e12 = q_full_pre - q_null_pre
    additive = e1 + e2
    residual = e12 - additive
    cosine = F.cosine_similarity(e12, additive, dim=-1)
    relative_residual = residual.norm(dim=-1) / e12.norm(dim=-1).clamp_min(EPS)
    interaction_ratio = residual.norm(dim=-1) / (e1.norm(dim=-1) + e2.norm(dim=-1)).clamp_min(EPS)
    return {
        "status": "ok_operational_null",
        "baseline": "empty_modification_string",
        "additivity_cosine": summarize(cosine),
        "relative_additivity_residual": summarize(relative_residual),
        "interaction_ratio": summarize(interaction_ratio),
    }

def caption_order_geometry_robustness(
    q_full_pre: torch.Tensor,
    q_swap_pre: torch.Tensor,
    q_single_1_pre: torch.Tensor,
    q_single_2_pre: torch.Tensor,
    q_full: torch.Tensor,
    q_swap: torch.Tensor,
) -> dict:
    """
    Semantically equivalent order counterfactual:
      cap1 and cap2  <->  cap2 and cap1

    For caption 1, compare the effect of adding cap1 when it appears first vs
    second. Likewise for caption 2.
    """
    effect_1_original = q_full_pre - q_single_2_pre
    effect_1_swapped = q_swap_pre - q_single_2_pre
    effect_2_original = q_full_pre - q_single_1_pre
    effect_2_swapped = q_swap_pre - q_single_1_pre
    cos_1 = F.cosine_similarity(effect_1_original, effect_1_swapped, dim=-1)
    cos_2 = F.cosine_similarity(effect_2_original, effect_2_swapped, dim=-1)
    rel_1 = (effect_1_original - effect_1_swapped).norm(dim=-1) / (effect_1_original.norm(dim=-1).clamp_min(EPS))
    rel_2 = (effect_2_original - effect_2_swapped).norm(dim=-1) / (effect_2_original.norm(dim=-1).clamp_min(EPS))
    return {
        "status": "ok",
        "counterfactual": "caption_order_swap",
        "full_vs_swapped_query_cosine": summarize(F.cosine_similarity(q_full, q_swap, dim=-1)),
        "effect_1_direction_cosine": summarize(cos_1),
        "effect_2_direction_cosine": summarize(cos_2),
        "effect_1_relative_change": summarize(rel_1),
        "effect_2_relative_change": summarize(rel_2),
    }

def _rank_statistics(
    scores: torch.Tensor,
    target_idx: torch.Tensor,
    reference_idx: torch.Tensor,
    exclude_reference: bool,
) -> dict[str, torch.Tensor]:
    if scores.ndim != 2:
        raise ValueError(f"scores must be [B,G], got {tuple(scores.shape)}")
    scores = scores.float()
    target_idx = target_idx.to(scores.device)
    reference_idx = reference_idx.to(scores.device)
    batch_idx = torch.arange(scores.shape[0], device=scores.device)
    working = scores.clone()
    if exclude_reference:
        valid_reference = reference_idx != target_idx
        if valid_reference.any():
            working[
                batch_idx[valid_reference],
                reference_idx[valid_reference],
            ] = -torch.inf
    target_score = working[batch_idx, target_idx]
    target_masked = working.clone()
    target_masked[batch_idx, target_idx] = -torch.inf
    best_negative = target_masked.max(dim=-1).values
    rank = 1 + (working > target_score.unsqueeze(-1)).sum(dim=-1)
    return {
        "target_score": target_score.detach().cpu(),
        "best_negative_score": best_negative.detach().cpu(),
        "rank": rank.detach().cpu().long(),
    }

class RetrievalCollector:
    def __init__(self) -> None:
        self.categories: list[str] = []
        self.data: dict[str, dict[str, dict[str, list[torch.Tensor]]]] = {}
        for policy in ("include_reference", "exclude_reference"):
            self.data[policy] = {}
            for variant in QUERY_VARIANTS:
                self.data[policy][variant] = {
                    "target_score": [],
                    "best_negative_score": [],
                    "rank": [],
                }
    def add_batch(
        self,
        category: str,
        score_by_variant: dict[str, torch.Tensor],
        target_idx: torch.Tensor,
        reference_idx: torch.Tensor,
    ) -> None:
        batch_size = target_idx.numel()
        self.categories.extend([category] * batch_size)
        for policy, exclude_reference in (
            ("include_reference", False),
            ("exclude_reference", True),
        ):
            for variant, scores in score_by_variant.items():
                stats = _rank_statistics(
                    scores=scores,
                    target_idx=target_idx,
                    reference_idx=reference_idx,
                    exclude_reference=exclude_reference,
                )
                for key, value in stats.items():
                    self.data[policy][variant][key].append(value)
    def finalize(self) -> dict:
        out = {"categories": self.categories, "policies": {}}
        for policy, policy_data in self.data.items():
            out["policies"][policy] = {}
            for variant, variant_data in policy_data.items():
                out["policies"][policy][variant] = {key: torch.cat(parts, dim=0) for key, parts in variant_data.items()}
        return out

def recall_summary(ranks: torch.Tensor, categories: list[str]) -> dict:
    ranks = ranks.long()
    if len(categories) != ranks.numel():
        raise ValueError("rank/category length mismatch")
    def metrics(mask: torch.Tensor) -> dict[str, float]:
        selected = ranks[mask]
        return {
            "num_queries": int(selected.numel()),
            "r1": (selected <= 1).float().mean().item() * 100,
            "r5": (selected <= 5).float().mean().item() * 100,
            "r10": (selected <= 10).float().mean().item() * 100,
            "r50": (selected <= 50).float().mean().item() * 100,
            "mean_r10_r50": ((selected <= 10).float().mean().item() * 100 + (selected <= 50).float().mean().item() * 100) / 2.0,
        }
    all_mask = torch.ones(ranks.numel(), dtype=torch.bool)
    result = {"overall": metrics(all_mask), "per_category": {}}
    for category in CATEGORIES:
        mask = torch.tensor([x == category for x in categories], dtype=torch.bool)
        result["per_category"][category] = metrics(mask)
    macro_r10 = sum(result["per_category"][x]["r10"] for x in CATEGORIES) / len(CATEGORIES)
    macro_r50 = sum(result["per_category"][x]["r50"] for x in CATEGORIES) / len(CATEGORIES)
    result["macro"] = {
        "r10": macro_r10,
        "r50": macro_r50,
        "mean_r10_r50": (macro_r10 + macro_r50) / 2.0,
    }
    return result

def native_retrieval_quality(retrieval: dict) -> dict:
    result = {}
    categories = retrieval["categories"]
    for policy, policy_data in retrieval["policies"].items():
        result[policy] = {
            "full": recall_summary(policy_data["full"]["rank"], categories),
            "swap": recall_summary(policy_data["swap"]["rank"], categories),
        }
    return result

def _conditional_fraction(
    event: torch.Tensor,
    condition: torch.Tensor,
) -> float | None:
    condition = condition.bool()
    if not condition.any():
        return None
    return event[condition].float().mean().item()

def _conditional_summary(
    values: torch.Tensor,
    condition: torch.Tensor,
) -> dict[str, float] | None:
    condition = condition.bool()
    if not condition.any():
        return None
    return summarize(values[condition])

def _necessity_tensors(
    full: dict[str, torch.Tensor],
    minus: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    full_margin = full["target_score"] - full["best_negative_score"]
    minus_margin = minus["target_score"] - minus["best_negative_score"]
    full_rank = full["rank"].float()
    minus_rank = minus["rank"].float()
    return {
        "score_drop": full["target_score"] - minus["target_score"],
        "rank_degradation": minus_rank - full_rank,
        "log_rank_ratio": torch.log(minus_rank / full_rank),
        "margin_drop": full_margin - minus_margin,
        "full_r10": full_rank <= 10,
        "full_r50": full_rank <= 50,
        "minus_r10": minus_rank <= 10,
        "minus_r50": minus_rank <= 50,
    }

def _necessity_one(
    full: dict[str, torch.Tensor],
    minus: dict[str, torch.Tensor],
) -> dict:
    values = _necessity_tensors(full, minus)
    return {
        "target_score_drop": summarize(values["score_drop"]),
        "rank_degradation": summarize(values["rank_degradation"]),
        "log_rank_ratio": summarize(values["log_rank_ratio"]),
        "margin_drop": summarize(values["margin_drop"]),
        "rank_worse_fraction": (values["rank_degradation"] > 0).float().mean().item(),
        "target_score_lower_fraction": (values["score_drop"] > 0).float().mean().item(),
        "margin_worse_fraction": (values["margin_drop"] > 0).float().mean().item(),
        "rank_worse_fraction_given_full_r50": _conditional_fraction(values["rank_degradation"] > 0, values["full_r50"]),
        "margin_worse_fraction_given_full_r50": _conditional_fraction(values["margin_drop"] > 0, values["full_r50"]),
        "num_full_r10": int(values["full_r10"].sum().item()),
        "num_full_r50": int(values["full_r50"].sum().item()),
        "escape_r10_fraction_given_full_r10": _conditional_fraction(~values["minus_r10"], values["full_r10"]),
        "escape_r50_fraction_given_full_r50": _conditional_fraction(~values["minus_r50"], values["full_r50"]),
        "rank_degradation_given_full_r10": _conditional_summary(values["rank_degradation"], values["full_r10"]),
        "rank_degradation_given_full_r50": _conditional_summary(values["rank_degradation"], values["full_r50"]),
        "log_rank_ratio_given_full_r10": _conditional_summary(values["log_rank_ratio"], values["full_r10"]),
        "log_rank_ratio_given_full_r50": _conditional_summary(values["log_rank_ratio"], values["full_r50"]),
        "margin_drop_given_full_r50": _conditional_summary(values["margin_drop"], values["full_r50"]),
    }

def teacher_edit_retrieval_sensitivity_metrics(retrieval: dict) -> dict:
    result = {}
    for policy, policy_data in retrieval["policies"].items():
        effect_1 = _necessity_one(policy_data["full"], policy_data["minus_1"])
        effect_2 = _necessity_one(policy_data["full"], policy_data["minus_2"])
        all_text = _necessity_one(policy_data["full"], policy_data["null"])
        t1 = _necessity_tensors(policy_data["full"], policy_data["minus_1"])
        t2 = _necessity_tensors(policy_data["full"], policy_data["minus_2"])
        combined = {key: torch.cat([t1[key], t2[key]], dim=0) for key in t1}
        result[policy] = {
            "interpretation": (
                "Single-caption removals probe caption-level decomposition; "
                "FashionIQ captions can be redundant/paraphrastic. "
                "all_text_removal separately tests whether modification text "
                "as a whole affects retrieval."
            ),
            "effect_1_remove_caption_1": effect_1,
            "effect_2_remove_caption_2": effect_2,
            "all_text_removal": all_text,
            "combined_single_caption_removals": {
                "target_score_drop": summarize(combined["score_drop"]),
                "rank_degradation": summarize(combined["rank_degradation"]),
                "log_rank_ratio": summarize(combined["log_rank_ratio"]),
                "margin_drop": summarize(combined["margin_drop"]),
                "rank_worse_fraction": (combined["rank_degradation"] > 0).float().mean().item(),
                "score_lower_fraction": (combined["score_drop"] > 0).float().mean().item(),
                "margin_worse_fraction": (combined["margin_drop"] > 0).float().mean().item(),
                "rank_worse_fraction_given_full_r50": _conditional_fraction(combined["rank_degradation"] > 0, combined["full_r50"]),
                "margin_worse_fraction_given_full_r50": _conditional_fraction(combined["margin_drop"] > 0, combined["full_r50"]),
                "num_full_r50_effects": int(combined["full_r50"].sum().item()),
                "escape_r50_fraction_given_full_r50": _conditional_fraction(~combined["minus_r50"], combined["full_r50"]),
                "rank_degradation_given_full_r50": _conditional_summary(combined["rank_degradation"], combined["full_r50"]),
                "log_rank_ratio_given_full_r50": _conditional_summary(combined["log_rank_ratio"], combined["full_r50"]),
                "margin_drop_given_full_r50": _conditional_summary(combined["margin_drop"], combined["full_r50"]),
            },
        }
    return result

def retrieval_order_robustness(retrieval: dict) -> dict:
    result = {}
    for policy, policy_data in retrieval["policies"].items():
        full = policy_data["full"]
        swap = policy_data["swap"]
        result[policy] = {
            "target_score_absolute_change": summarize((full["target_score"] - swap["target_score"]).abs()),
            "target_rank_absolute_change": summarize((full["rank"].float() - swap["rank"].float()).abs()),
            "same_target_rank_fraction": (full["rank"] == swap["rank"]).float().mean().item(),
        }
    return result

def score_vector_gallery(
    query: torch.Tensor,
    target_features: torch.Tensor,
) -> torch.Tensor:
    return query @ target_features.T

def score_token_gallery(
    query: torch.Tensor,
    target_features: torch.Tensor,
) -> torch.Tensor:

    return torch.einsum("bd,gtd->bgt", query, target_features).max(dim=-1).values

def build_encoder_query_texts(cases: list[dict], adapter, correction_dicts) -> dict[str, list[str]]:
    full = adapter.prepare_texts(cases, "full_text", correction_dicts)
    minus_1 = adapter.prepare_texts(cases, "minus_1_text", correction_dicts)
    minus_2 = adapter.prepare_texts(cases, "minus_2_text", correction_dicts)
    swapped = []
    for case in cases:
        correction = correction_dicts[case["category"]]
        cap1 = adapter.correct_encoder_text(case["caption_1"], correction)
        cap2 = adapter.correct_encoder_text(case["caption_2"], correction)
        swapped.append(f"{cap2} and {cap1}")
    return {
        "full": full,
        "minus_1": minus_1,
        "minus_2": minus_2,
        "swap": swapped,
        "null": [""] * len(cases),
    }

def build_hint_query_texts(cases: list[dict], adapter, correction_dicts) -> dict[str, list[str]]:
    full = adapter.prepare_texts(
        cases,
        "full_text",
        correction_dicts,
    )
    minus_1 = adapter.prepare_texts(cases, "minus_1_text", correction_dicts)
    minus_2 = adapter.prepare_texts(cases, "minus_2_text", correction_dicts)
    swapped = []
    for case in cases:
        correction = correction_dicts[case["category"]]
        cap1 = adapter.correct_hint_text(case["caption_1"], correction)
        cap2 = adapter.correct_hint_text(case["caption_2"], correction)
        swapped.append(f"{cap2} and {cap1}")
    return {
        "full": full,
        "minus_1": minus_1,
        "minus_2": minus_2,
        "swap": swapped,
        "null": [""] * len(cases),
    }

def build_standard_query_texts(cases: list[dict]) -> dict[str, list[str]]:
    return {
        "full": [case["full_text"] for case in cases],
        "minus_1": [case["minus_1_text"] for case in cases],
        "minus_2": [case["minus_2_text"] for case in cases],
        "swap": [compose_swapped(case["caption_1"], case["caption_2"]) for case in cases],
        "null": [""] * len(cases),
    }

def _append_query_output(
    store: dict[str, list[torch.Tensor]],
    variant: str,
    pre: torch.Tensor,
    normalized: torch.Tensor,
) -> None:
    store[f"q_{variant}_pre_norm"].append(pre.detach().cpu())
    store[f"q_{variant}"].append(normalized.detach().cpu())

def _finalize_query_store(
    store: dict[str, list[torch.Tensor]],
) -> dict[str, torch.Tensor]:
    return {key: torch.cat(value, dim=0) for key, value in store.items()}

def collect_encoder_queries(
    model,
    preprocess,
    correction_dicts,
    cases: list[dict],
    image_root: Path,
    device: torch.device,
    batch_size: int,
    adapter,
) -> dict[str, torch.Tensor]:
    store: dict[str, list[torch.Tensor]] = {}
    for variant in ("full", "minus_1", "minus_2", "swap", "null"):
        store[f"q_{variant}_pre_norm"] = []
        store[f"q_{variant}"] = []
    for start in range(0, len(cases), batch_size):
        batch = cases[start : start + batch_size]
        images = load_case_image_batch(
            cases=batch,
            image_root=image_root,
            preprocess=preprocess,
        ).to(device)
        texts = build_encoder_query_texts(batch, adapter, correction_dicts)
        for variant in ("full", "minus_1", "minus_2", "swap", "null"):
            pre, normalized = adapter.compose_query(model, images, texts[variant])
            _append_query_output(store, variant, pre, normalized)
        del images
    return _finalize_query_store(store)

def collect_qformer_queries(
    model,
    preprocess,
    txt_processor,
    cases: list[dict],
    image_root: Path,
    device: torch.device,
    batch_size: int,
    teacher: str,
    adapter,
) -> dict[str, torch.Tensor]:
    store: dict[str, list[torch.Tensor]] = {}
    for variant in ("full", "minus_1", "minus_2", "swap", "null"):
        store[f"q_{variant}_pre_norm"] = []
        store[f"q_{variant}"] = []
    for start in range(0, len(cases), batch_size):
        batch = cases[start : start + batch_size]
        images = load_case_image_batch(
            cases=batch,
            image_root=image_root,
            preprocess=preprocess,
        ).to(device)
        texts = build_standard_query_texts(batch)
        if teacher == "tme":
            vit_states = adapter.encode_vit_states(model, images)
            reference_repr = adapter.encode_reference(model, vit_states)
        elif teacher in {"sprc", "tgcir", "csmcir"}:
            reference_repr = adapter.encode_reference(model, images)
        else:
            raise ValueError(teacher)
        for variant in ("full", "minus_1", "minus_2", "swap", "null"):
            pre, normalized = adapter.compose_query(
                model,
                reference_repr,
                texts[variant],
                txt_processor,
            )
            _append_query_output(store, variant, pre, normalized)
        del images, reference_repr
    return _finalize_query_store(store)

def collect_hint_queries(
    model,
    preprocess,
    txt_processor,
    correction_dicts,
    cases: list[dict],
    image_root: Path,
    device: torch.device,
    batch_size: int,
    adapter,
) -> dict[str, torch.Tensor]:
    store: dict[str, list[torch.Tensor]] = {}
    for variant in QUERY_VARIANTS:
        store[f"q_{variant}_pre_norm"] = []
        store[f"q_{variant}"] = []
    for start in range(0, len(cases), batch_size):
        batch = cases[start : start + batch_size]
        images = load_case_image_batch(
            cases=batch,
            image_root=image_root,
            preprocess=preprocess,
        ).to(device)
        reference_embeds = adapter.encode_reference(
            model=model,
            reference_images=images,
        )
        texts = build_hint_query_texts(batch, adapter, correction_dicts)
        for variant in QUERY_VARIANTS:
            pre, normalized = adapter.compose_query_from_reference(
                model=model,
                reference_embeds=reference_embeds,
                texts=texts[variant],
                txt_processor=txt_processor,
            )
            _append_query_output(store, variant, pre, normalized)
        del images, reference_embeds
    return _finalize_query_store(store)

def build_encoder_gallery(
    model,
    preprocess,
    image_ids: list[str],
    category: str,
    image_root: Path,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    parts = []
    for start in range(0, len(image_ids), batch_size):
        ids = image_ids[start : start + batch_size]
        images = load_image_batch(ids, category, image_root, preprocess).to(device)
        with torch.no_grad():
            features = model.extract_retrieval_target(images)
        parts.append(features.detach().float().cpu())
        del images, features
    return torch.cat(parts, dim=0)

def build_tme_gallery(
    model,
    preprocess,
    image_ids: list[str],
    category: str,
    image_root: Path,
    device: torch.device,
    batch_size: int,
    adapter,
) -> torch.Tensor:
    parts = []
    for start in range(0, len(image_ids), batch_size):
        ids = image_ids[start : start + batch_size]
        images = load_image_batch(ids, category, image_root, preprocess).to(device)
        vit_states = adapter.encode_vit_states(model, images)
        with torch.no_grad():
            f_target = model.encode_image(vit_states)
            z_target = F.normalize(model.vision_proj(f_target), dim=-1)
        parts.append(z_target.detach().float().cpu())
        del images, vit_states, f_target, z_target
    return torch.cat(parts, dim=0)

def build_sprc_gallery(
    model,
    preprocess,
    image_ids: list[str],
    category: str,
    image_root: Path,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    parts = []
    for start in range(0, len(image_ids), batch_size):
        ids = image_ids[start : start + batch_size]
        images = load_image_batch(ids, category, image_root, preprocess).to(device)
        with torch.no_grad():
            target_features, _ = model.extract_target_features(images)
        parts.append(target_features.detach().float().cpu())
        del images, target_features
    return torch.cat(parts, dim=0)


def build_tgcir_gallery(
    model,
    preprocess,
    image_ids,
    category,
    image_root,
    device,
    batch_size,
    adapter,
):
    parts = []

    for start in range(0, len(image_ids), batch_size):
        ids = image_ids[start : start + batch_size]

        images = load_image_batch(
            ids,
            category,
            image_root,
            preprocess,
        ).to(device)

        features = adapter.encode_target(
            model,
            images,
        )

        parts.append(
            features.detach().float().cpu()
        )

        del images, features

    return torch.cat(parts, dim=0)


def build_csmcir_gallery(
    model,
    preprocess,
    image_ids,
    category,
    image_root,
    device,
    batch_size,
    adapter,
    caption_dicts,
):
    parts = []

    for start in range(0, len(image_ids), batch_size):
        ids = image_ids[start : start + batch_size]

        images = load_image_batch(
            ids,
            category,
            image_root,
            preprocess,
        ).to(device)

        captions = [
            adapter.target_caption(
                caption_dicts,
                category,
                image_id,
            )
            for image_id in ids
        ]

        features = adapter.encode_target(
            model,
            images,
            captions,
        )

        parts.append(
            features.detach().float().cpu()
        )

        del images, features

    return torch.cat(parts, dim=0)


def build_qure_gallery(
    model,
    preprocess,
    image_ids: list[str],
    category: str,
    image_root: Path,
    device: torch.device,
    batch_size: int,
    adapter,
) -> torch.Tensor:
    parts = []
    for start in range(
        0,
        len(image_ids),
        batch_size,
    ):
        ids = image_ids[start : start + batch_size]
        images = load_image_batch(
            ids,
            category,
            image_root,
            preprocess,
        ).to(device)
        target_features = adapter.encode_target(
            model=model,
            images=images,
        )
        if target_features.ndim != 3:
            raise ValueError(f"QuRe native target representation must be [B,T,D], got {tuple(target_features.shape)}")
        parts.append(target_features.detach().float().cpu())
        del images, target_features
    return torch.cat(
        parts,
        dim=0,
    )

def build_hint_gallery(
    model,
    preprocess,
    image_ids: list[str],
    category: str,
    image_root: Path,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    parts = []
    for start in range(0, len(image_ids), batch_size):
        ids = image_ids[start : start + batch_size]
        images = load_image_batch(ids, category, image_root, preprocess).to(device)
        with torch.no_grad():
            native_features = model.extract_retrieval_target(images)
        if native_features.ndim != 3:
            raise ValueError(f"HINT target features must be rank-3, got {tuple(native_features.shape)}")
        target_tokens = native_features.permute(0, 2, 1).contiguous()
        parts.append(target_tokens.detach().float().cpu())
        del images, native_features, target_tokens
    return torch.cat(parts, dim=0)

def score_all_queries(
    query_store: dict[str, torch.Tensor],
    cases: list[dict],
    gallery_builder: Callable[[list[str], str], torch.Tensor],
    score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    score_batch_size: int,
    gallery_id_provider: Callable[[str, list[dict]], list[str]],
    protocol_name: str,
) -> dict:
    collector = RetrievalCollector()
    gallery_sizes = {}
    for category in CATEGORIES:
        category_indices = [i for i, case in enumerate(cases) if case["category"] == category]
        category_cases = [cases[i] for i in category_indices]
        if not category_cases:
            raise ValueError(f"No audit cases for category={category}")
        gallery_ids = gallery_id_provider(category, category_cases)
        gallery_sizes[category] = len(gallery_ids)
        name_to_idx = {name: i for i, name in enumerate(gallery_ids)}
        if len(name_to_idx) != len(gallery_ids):
            raise ValueError(f"{category}: duplicate gallery image IDs")
        missing_targets = [case["target_id"] for case in category_cases if case["target_id"] not in name_to_idx]
        missing_refs = [case["reference_id"] for case in category_cases if case["reference_id"] not in name_to_idx]
        if missing_targets or missing_refs:
            raise KeyError(f"{category}: gallery mismatch. missing_targets={len(missing_targets)} missing_refs={len(missing_refs)}")
        target_features = gallery_builder(gallery_ids, category).to(device)
        for start in range(0, len(category_indices), score_batch_size):
            idx = category_indices[start : start + score_batch_size]
            batch_cases = [cases[i] for i in idx]
            target_idx = torch.tensor(
                [name_to_idx[c["target_id"]] for c in batch_cases],
                dtype=torch.long,
            )
            reference_idx = torch.tensor(
                [name_to_idx[c["reference_id"]] for c in batch_cases],
                dtype=torch.long,
            )
            score_by_variant = {}
            for variant in QUERY_VARIANTS:
                query = query_store[f"q_{variant}"][idx].to(device)
                with torch.no_grad():
                    scores = score_fn(query, target_features)
                expected = (len(idx), len(gallery_ids))
                if scores.shape != expected:
                    raise ValueError(f"{category}/{variant}: scorer shape {tuple(scores.shape)} != {expected}")
                if not torch.isfinite(scores).all():
                    raise ValueError(f"{category}/{variant}: scores contain NaN/Inf")
                score_by_variant[variant] = scores.detach()
            collector.add_batch(
                category=category,
                score_by_variant=score_by_variant,
                target_idx=target_idx,
                reference_idx=reference_idx,
            )
        del target_features
        if device.type == "cuda":
            torch.cuda.empty_cache()
    output = collector.finalize()
    output["protocol_name"] = protocol_name
    output["gallery_sizes"] = gallery_sizes
    return output

def full_gallery_provider(split_root: Path):
    def provider(category: str, _category_cases: list[dict]) -> list[str]:
        return load_split_ids(split_root, category)
    return provider

def pair_union_gallery_provider(
    _category: str,
    category_cases: list[dict],
) -> list[str]:
    return build_pair_union_gallery_ids(category_cases)

def _autograd_connectivity_probe(
    query: torch.Tensor,
    alpha: torch.Tensor,
    boundary: str,
) -> dict:
    if not query.requires_grad:
        return {
            "status": "fail",
            "boundary": boundary,
            "reason": "Query tensor does not require gradients.",
        }
    d = query.numel()
    device, dtype = query.device, query.dtype
    probes = [
        torch.ones_like(query),
        torch.linspace(-1.0, 1.0, d, device=device, dtype=dtype).reshape_as(query),
        torch.where(
            torch.arange(d, device=device).reshape_as(query) % 2 == 0,
            torch.ones((), device=device, dtype=dtype),
            -torch.ones((), device=device, dtype=dtype),
        ),
    ]
    values = []
    for index, grad_output in enumerate(probes):
        grad = torch.autograd.grad(
            outputs=query,
            inputs=alpha,
            grad_outputs=grad_output,
            retain_graph=index < len(probes) - 1,
            allow_unused=True,
        )[0]
        values.append(None if grad is None else float(grad.detach().abs().item()))
    finite_nonzero = [v for v in values if v is not None and math.isfinite(v) and v > 1e-12]
    return {
        "status": "pass" if finite_nonzero else "fail",
        "boundary": boundary,
        "vjp_gradient_abs": values,
        "num_nonzero_finite_probes": len(finite_nonzero),
        "interpretation": (
            "Pass proves autograd connectivity for a continuous intervention "
            "at this boundary; it is necessary but not sufficient for the "
            "final TAPER slot-mask parameterization."
        ),
    }

def probe_encoder_differentiability(
    model,
    preprocess,
    correction_dicts,
    case: dict,
    image_root: Path,
    device: torch.device,
    adapter,
) -> dict:
    """
    Continuous perturbation is injected at ENCODER's text representation
    boundary (output of backbone.text_out) while all teacher weights remain
    frozen. This tests whether a soft intervention can backpropagate from
    composed query output into an external continuous control variable.
    """
    image = load_image_batch(
        [case["reference_id"]],
        case["category"],
        image_root,
        preprocess,
    ).to(device)
    text = build_encoder_query_texts([case], adapter, correction_dicts)["full"]
    alpha = torch.tensor(0.0, device=device, requires_grad=True)
    original_text_out = model.backbone.text_out
    def patched_text_out(tokens):
        pooled, local = original_text_out(tokens)
        def pattern_like(x: torch.Tensor) -> torch.Tensor:
            axis = torch.linspace(
                -1.0,
                1.0,
                x.shape[-1],
                dtype=x.dtype,
                device=x.device,
            )
            shape = [1] * x.ndim
            shape[-1] = x.shape[-1]
            return axis.view(*shape).expand_as(x).detach()
        return (
            pooled + 1e-2 * alpha * pattern_like(pooled),
            local + 1e-2 * alpha * pattern_like(local),
        )
    model.backbone.text_out = patched_text_out
    try:
        fuse_local, _, _, _, _ = model.compose_feature(image, text)
        query = fuse_local.mean(dim=1)
        return _autograd_connectivity_probe(
            query=query,
            alpha=alpha,
            boundary="ENCODER backbone.text_out continuous perturbation",
        )
    except Exception as exc:
        return {
            "status": "error",
            "boundary": "ENCODER backbone.text_out",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    finally:
        model.backbone.text_out = original_text_out
        model.zero_grad(set_to_none=True)

def _qformer_word_embedding_module(model):
    candidates = [
        getattr(getattr(model.Qformer, "bert", None), "embeddings", None),
        getattr(model.Qformer, "embeddings", None),
    ]
    for embeddings in candidates:
        if embeddings is not None and hasattr(embeddings, "word_embeddings"):
            return embeddings.word_embeddings
    raise AttributeError("Could not locate Q-Former word_embeddings module")

def probe_tme_differentiability(
    model,
    preprocess,
    txt_processor,
    case: dict,
    image_root: Path,
    device: torch.device,
    adapter,
) -> dict:
    image = load_image_batch(
        [case["reference_id"]],
        case["category"],
        image_root,
        preprocess,
    ).to(device)
    with torch.no_grad():
        vit_states = adapter.encode_vit_states(model, image)
        reference_tokens = adapter.encode_reference(model, vit_states).detach()
    text = txt_processor(case["full_text"])
    text_tokens = model.tokenizer(
        [text],
        padding="max_length",
        truncation=True,
        max_length=model.max_txt_len,
        return_tensors="pt",
    ).to(device)
    reference_attention_mask = torch.ones(
        reference_tokens.shape[:-1],
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.cat(
        [reference_attention_mask, text_tokens.attention_mask],
        dim=1,
    )
    alpha = torch.tensor(0.0, device=device, requires_grad=True)
    word_embeddings = _qformer_word_embedding_module(model)
    def hook(_module, _inputs, output):
        axis = torch.linspace(
            -1.0,
            1.0,
            output.shape[-1],
            dtype=output.dtype,
            device=output.device,
        )
        shape = [1] * output.ndim
        shape[-1] = output.shape[-1]
        perturb = axis.view(*shape).expand_as(output).detach()
        return output + 1e-2 * alpha * perturb
    handle = word_embeddings.register_forward_hook(hook)
    try:
        query = model.encode_fusion(reference_tokens, text_tokens)
        return _autograd_connectivity_probe(
            query=query,
            alpha=alpha,
            boundary=("TME encode_fusion / Q-Former word-embedding perturbation"),
        )
    except Exception as exc:
        return {
            "status": "error",
            "boundary": "TME Q-Former word embeddings",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)

def probe_sprc_differentiability(
    model,
    preprocess,
    txt_processor,
    case: dict,
    image_root: Path,
    device: torch.device,
    adapter,
) -> dict:
    image = load_image_batch(
        [case["reference_id"]],
        case["category"],
        image_root,
        preprocess,
    ).to(device)
    with torch.no_grad():
        reference_embeds = adapter.encode_reference(model, image).detach()
    text = txt_processor(case["full_text"])
    text_tokens = model.tokenizer(
        [text],
        padding="max_length",
        truncation=True,
        max_length=model.max_txt_len,
        return_tensors="pt",
    ).to(device)
    image_atts = torch.ones(
        reference_embeds.size()[:-1],
        dtype=torch.long,
        device=device,
    )
    query_tokens = model.query_tokens.expand(1, -1, -1).detach()
    query_atts = torch.ones(
        query_tokens.size()[:-1],
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.cat(
        [query_atts, text_tokens.attention_mask],
        dim=1,
    )
    alpha = torch.tensor(0.0, device=device, requires_grad=True)
    word_embeddings = _qformer_word_embedding_module(model)
    def hook(_module, _inputs, output):
        axis = torch.linspace(
            -1.0,
            1.0,
            output.shape[-1],
            dtype=output.dtype,
            device=output.device,
        )
        shape = [1] * output.ndim
        shape[-1] = output.shape[-1]
        perturb = axis.view(*shape).expand_as(output).detach()
        return output + 1e-2 * alpha * perturb
    handle = word_embeddings.register_forward_hook(hook)
    try:
        fusion_output = model.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=reference_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        second_stage_query = fusion_output.last_hidden_state[:, : query_tokens.size(1), :]
        text_output = model.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=second_stage_query,
            attention_mask=attention_mask,
            return_dict=True,
        )
        query_pre = model.text_proj(text_output.last_hidden_state[:, query_tokens.size(1), :])
        return _autograd_connectivity_probe(
            query=query_pre,
            alpha=alpha,
            boundary=("SPRC two-stage Q-Former word-embedding perturbation"),
        )
    except Exception as exc:
        return {
            "status": "error",
            "boundary": "SPRC Q-Former word embeddings",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)

def probe_hint_differentiability(
    model,
    preprocess,
    txt_processor,
    correction_dicts,
    case: dict,
    image_root: Path,
    device: torch.device,
    adapter,
) -> dict:
    image = load_image_batch(
        [case["reference_id"]],
        case["category"],
        image_root,
        preprocess,
    ).to(device)
    with torch.no_grad():
        reference_embeds = adapter.encode_reference(
            model=model,
            reference_images=image,
        ).detach()
    raw_text = build_hint_query_texts(
        [case],
        adapter,
        correction_dicts,
    )["full"][0]
    processed_text = txt_processor(raw_text)
    text_tokens = model.tokenizer(
        [processed_text],
        padding="max_length",
        truncation=True,
        max_length=model.max_txt_len,
        return_tensors="pt",
    ).to(device)
    image_atts = torch.ones(
        reference_embeds.size()[:-1],
        dtype=torch.long,
        device=device,
    )
    query_tokens = model.query_tokens.expand(
        1,
        -1,
        -1,
    ).detach()
    if query_tokens.size(1) != 32:
        return {
            "status": "fail",
            "boundary": "HINT Q-Former",
            "reason": (f"HINT native retrieval expects 32 query tokens, got {query_tokens.size(1)}"),
        }
    query_atts = torch.ones(
        query_tokens.size()[:-1],
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.cat(
        [
            query_atts,
            text_tokens.attention_mask,
        ],
        dim=1,
    )
    alpha = torch.tensor(
        0.0,
        device=device,
        requires_grad=True,
    )
    word_embeddings = _qformer_word_embedding_module(model)
    def hook(_module, _inputs, output):
        axis = torch.linspace(
            -1.0,
            1.0,
            output.shape[-1],
            dtype=output.dtype,
            device=output.device,
        )
        shape = [1] * output.ndim
        shape[-1] = output.shape[-1]
        perturb = axis.view(*shape).expand_as(output).detach()
        return output + 1e-2 * alpha * perturb
    handle = word_embeddings.register_forward_hook(hook)
    try:
        fusion_output = model.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=reference_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        query_pre = model.text_proj(fusion_output.last_hidden_state[:, query_tokens.size(1), :])
        return _autograd_connectivity_probe(
            query=query_pre,
            alpha=alpha,
            boundary=("HINT Q-Former word-embedding continuous perturbation"),
        )
    except Exception as exc:
        return {
            "status": "error",
            "boundary": "HINT Q-Former word embeddings",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)

def probe_qure_differentiability(
    model,
    preprocess,
    txt_processor,
    case: dict,
    image_root: Path,
    device: torch.device,
    adapter,
) -> dict:
    image = load_image_batch(
        [case["reference_id"]],
        case["category"],
        image_root,
        preprocess,
    ).to(device)
    with torch.no_grad():
        reference_states = adapter.encode_reference(
            model=model,
            images=image,
        ).detach()
    processed_text = txt_processor(case["full_text"])
    text_tokens = model.tokenizer(
        [processed_text],
        padding="max_length",
        truncation=True,
        max_length=model.max_txt_len,
        return_tensors="pt",
    ).to(device)
    reference_attention_mask = torch.ones(
        reference_states.size()[:-1],
        dtype=torch.long,
        device=device,
    )
    query_tokens = model.query_tokens.expand(
        1,
        -1,
        -1,
    ).detach()
    query_attention_mask = torch.ones(
        query_tokens.size()[:-1],
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.cat(
        [
            query_attention_mask,
            text_tokens.attention_mask,
        ],
        dim=1,
    )
    alpha = torch.tensor(
        0.0,
        device=device,
        requires_grad=True,
    )
    word_embeddings = _qformer_word_embedding_module(model)
    def hook(
        _module,
        _inputs,
        output,
    ):
        axis = torch.linspace(
            -1.0,
            1.0,
            output.shape[-1],
            dtype=output.dtype,
            device=output.device,
        )
        shape = [1] * output.ndim
        shape[-1] = output.shape[-1]
        perturb = axis.view(*shape).expand_as(output).detach()
        return output + 1e-2 * alpha * perturb
    handle = word_embeddings.register_forward_hook(hook)
    try:
        query_output = model.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=reference_states,
            encoder_attention_mask=(reference_attention_mask),
            return_dict=True,
        )
        query_token_states = query_output.last_hidden_state[:, : query_tokens.size(1), :]
        query_pre = model.text_proj(query_token_states).mean(dim=1)
        return _autograd_connectivity_probe(
            query=query_pre,
            alpha=alpha,
            boundary=("QuRe Q-Former word-embedding continuous perturbation"),
        )
    except Exception as exc:
        return {
            "status": "error",
            "boundary": ("QuRe Q-Former word embeddings"),
            "reason": (f"{type(exc).__name__}: {exc}"),
        }
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


def probe_tgcir_differentiability(
    model,
    preprocess,
    txt_processor,
    case,
    image_root,
    device,
    adapter,
):
    image = load_image_batch(
        [case["reference_id"]],
        case["category"],
        image_root,
        preprocess,
    ).to(device)

    with torch.no_grad():
        reference_tokens = adapter.encode_reference(
            model,
            image,
        ).detach()

    alpha = torch.tensor(
        0.0,
        device=device,
        requires_grad=True,
    )

    embedding = model.backbone.clip.token_embedding

    def hook(_module, _inputs, output):
        axis = torch.linspace(
            -1.0,
            1.0,
            output.shape[-1],
            device=output.device,
            dtype=output.dtype,
        )

        perturb = axis.view(
            *([1] * (output.ndim - 1)),
            output.shape[-1],
        ).expand_as(output).detach()

        return output + 1e-2 * alpha * perturb

    handle = embedding.register_forward_hook(hook)

    try:
        texts = [txt_processor(case["full_text"])]

        mod_tokens = model.backbone.extract_text_fea(
            texts
        )

        remain_mask = model.s_remain_map(
            torch.cat(
                [reference_tokens, mod_tokens],
                dim=-1,
            )
        )

        fused = (
            remain_mask * reference_tokens
            + (1.0 - remain_mask) * mod_tokens
        )

        query_pre = fused.mean(dim=1)

        return _autograd_connectivity_probe(
            query=query_pre,
            alpha=alpha,
            boundary=(
                "TG-CIR CLIP text-token embedding "
                "continuous perturbation"
            ),
        )

    except Exception as exc:
        return {
            "status": "error",
            "boundary": "TG-CIR CLIP text embeddings",
            "reason": f"{type(exc).__name__}: {exc}",
        }

    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


def probe_csmcir_differentiability(
    model,
    preprocess,
    txt_processor,
    case,
    image_root,
    device,
    adapter,
):
    image = load_image_batch(
        [case["reference_id"]],
        case["category"],
        image_root,
        preprocess,
    ).to(device)

    with torch.no_grad():
        reference_embeds = adapter.encode_reference(
            model,
            image,
        ).detach()

    processed = txt_processor(
        case["full_text"]
    )

    text_tokens = model.tokenizer(
        [processed],
        padding="max_length",
        truncation=True,
        max_length=model.max_txt_len,
        return_tensors="pt",
    ).to(device)

    query_tokens = model.query_tokens.expand(
        1,
        -1,
        -1,
    ).detach()

    query_atts = torch.ones(
        query_tokens.size()[:-1],
        dtype=torch.long,
        device=device,
    )

    image_atts = torch.ones(
        reference_embeds.size()[:-1],
        dtype=torch.long,
        device=device,
    )

    attention_mask = torch.cat(
        [
            query_atts,
            text_tokens.attention_mask,
        ],
        dim=1,
    )

    alpha = torch.tensor(
        0.0,
        device=device,
        requires_grad=True,
    )

    word_embeddings = (
        _qformer_word_embedding_module(model)
    )

    def hook(_module, _inputs, output):
        axis = torch.linspace(
            -1.0,
            1.0,
            output.shape[-1],
            dtype=output.dtype,
            device=output.device,
        )

        perturb = axis.view(
            *([1] * (output.ndim - 1)),
            output.shape[-1],
        ).expand_as(output).detach()

        return output + 1e-2 * alpha * perturb

    handle = word_embeddings.register_forward_hook(
        hook
    )

    try:
        fusion_output = model.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=reference_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        query_pre = model.text_proj(
            fusion_output.last_hidden_state[
                :,
                query_tokens.size(1),
                :
            ]
        )

        return _autograd_connectivity_probe(
            query=query_pre,
            alpha=alpha,
            boundary=(
                "CSMCIR Q-Former word-embedding "
                "continuous perturbation"
            ),
        )

    except Exception as exc:
        return {
            "status": "error",
            "boundary": "CSMCIR Q-Former embeddings",
            "reason": f"{type(exc).__name__}: {exc}",
        }

    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


def balanced_geometry_by_category(
    cases: list[dict],
    delta_1: torch.Tensor,
    delta_2: torch.Tensor,
    min_group_count: int,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    results = {}
    valid_gaps = []
    for category_index, category in enumerate(CATEGORIES):
        indices = [i for i, case in enumerate(cases) if case["category"] == category]
        effects = torch.cat([delta_1[indices], delta_2[indices]], dim=0)
        labels = [normalize_edit_label(cases[i]["caption_1"]) for i in indices] + [normalize_edit_label(cases[i]["caption_2"]) for i in indices]
        result = balanced_same_edit_consistency(
            effects,
            labels,
            min_group_count,
            bootstrap_samples,
            seed + category_index + 1,
        )
        results[category] = result
        if result.get("status") == "ok" and result.get("macro_same_vs_different_gap") is not None:
            valid_gaps.append(result["macro_same_vs_different_gap"])
    return {
        "per_category": results,
        "macro_category_gap": (None if not valid_gaps else sum(valid_gaps) / len(valid_gaps)),
        "min_category_gap": (None if not valid_gaps else min(valid_gaps)),
        "num_categories_with_valid_gap": len(valid_gaps),
    }

def geometry_group_count_sensitivity(
    cases: list[dict],
    delta_1: torch.Tensor,
    delta_2: torch.Tensor,
    bootstrap_samples: int,
    seed: int,
    thresholds: tuple[int, ...] = (2, 3, 5),
) -> dict:
    """
    Recompute category-balanced geometry under several minimum-repeat cutoffs.

    This is not an extra vote. It exposes whether the teacher ranking depends
    on giving extremely small repeated-edit groups (especially n=2) the same
    macro weight as well-supported groups.
    """
    out = {}
    for offset, threshold in enumerate(thresholds):
        result = balanced_geometry_by_category(
            cases=cases,
            delta_1=delta_1,
            delta_2=delta_2,
            min_group_count=threshold,
            bootstrap_samples=bootstrap_samples,
            seed=seed + 1000 + offset * 100,
        )
        out[str(threshold)] = {
            "macro_category_gap": result.get("macro_category_gap"),
            "min_category_gap": result.get("min_category_gap"),
            "num_categories_with_valid_gap": result.get("num_categories_with_valid_gap"),
            "per_category_num_groups": {
                category: category_result.get("num_repeated_groups") for category, category_result in result["per_category"].items()
            },
        }
    return out

def select_published_native_policy(
    teacher_name: str,
    retrieval_native: dict,
) -> dict:
    policy = "exclude_reference" if teacher_name == "ENCODER" else "include_reference"
    return {
        "policy": policy,
        "quality": native_retrieval_quality(retrieval_native)[policy]["full"],
        "gallery_sizes": retrieval_native.get("gallery_sizes"),
        "protocol_name": retrieval_native.get("protocol_name"),
    }

def build_report(
    teacher_name: str,
    cases: list[dict],
    query_store: dict[str, torch.Tensor],
    retrieval_common: dict,
    retrieval_native: dict,
    differentiability: dict,
    integrity: dict,
    min_group_count: int,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    q_full_pre = query_store["q_full_pre_norm"].float()
    q_m1_pre = query_store["q_minus_1_pre_norm"].float()
    q_m2_pre = query_store["q_minus_2_pre_norm"].float()
    q_swap_pre = query_store["q_swap_pre_norm"].float()
    q_null_pre = query_store["q_null_pre_norm"].float()
    q_full = query_store["q_full"].float()
    q_m1 = query_store["q_minus_1"].float()
    q_m2 = query_store["q_minus_2"].float()
    q_swap = query_store["q_swap"].float()
    delta_1 = q_full_pre - q_m1_pre
    delta_2 = q_full_pre - q_m2_pre
    delta_1_unit = q_full - q_m1
    delta_2_unit = q_full - q_m2
    labels_1 = [normalize_edit_label(c["caption_1"]) for c in cases]
    labels_2 = [normalize_edit_label(c["caption_2"]) for c in cases]
    all_effects = torch.cat([delta_1, delta_2], dim=0)
    all_labels = labels_1 + labels_2
    balanced_overall = balanced_same_edit_consistency(
        all_effects,
        all_labels,
        min_group_count,
        bootstrap_samples,
        seed,
    )
    balanced_categories = balanced_geometry_by_category(
        cases,
        delta_1,
        delta_2,
        min_group_count,
        bootstrap_samples,
        seed,
    )
    balanced_unit = balanced_same_edit_consistency(
        torch.cat([delta_1_unit, delta_2_unit], dim=0),
        all_labels,
        min_group_count,
        bootstrap_samples,
        seed + 100,
    )
    balanced_unit_categories = balanced_geometry_by_category(
        cases,
        delta_1_unit,
        delta_2_unit,
        min_group_count,
        bootstrap_samples,
        seed + 100,
    )
    pre_norm_group_count_sensitivity = geometry_group_count_sensitivity(cases, delta_1, delta_2, bootstrap_samples, seed)
    normalized_group_count_sensitivity = geometry_group_count_sensitivity(cases, delta_1_unit, delta_2_unit, bootstrap_samples, seed + 500)
    common_policy_data = retrieval_common["policies"]["include_reference"]
    comparison_payload = {
        "categories": list(retrieval_common["categories"]),
        "common_full_gallery_include_reference_full_ranks": (common_policy_data["full"]["rank"].tolist()),
    }
    return {
        "teacher": teacher_name,
        "audit_version": 6,
        "integrity": integrity,
        "num_queries": len(cases),
        "query_dimension": int(q_full.shape[-1]),
        "protocol": {
            "dataset": "FashionIQ val",
            "common_comparison": ("official full split.<category>.val.json gallery; both include-reference and exclude-reference reported"),
            "published_native": ("teacher-specific upstream evaluator reproduced separately"),
            "native_scorer": True,
            "geometry_query_preprocessing": "deterministic teacher-valid path",
            "compound_null": ("empty modification string; operational diagnostic only"),
            "counterfactual": ("caption conjunction order swap only; not a comprehensive counterfactual suite"),
            "natural_geometry_semantics": (
                "Exact repeated FashionIQ captions are a high-precision proxy "
                "for repeated modification instructions, NOT guaranteed atomic "
                "edit labels. Natural-caption geometry is a screening signal; "
                "final atomic-edit claims require a controlled atomic audit."
            ),
        },
        "effect_1": effect_metrics(q_full_pre, q_m1_pre, q_full, q_m1),
        "effect_2": effect_metrics(q_full_pre, q_m2_pre, q_full, q_m2),
        "within_sample_effect_cosine": summarize(F.cosine_similarity(delta_1, delta_2, dim=-1)),
        "same_edit_directional_consistency_pair_weighted": (pair_weighted_same_edit_consistency(all_effects, all_labels, min_group_count)),
        "same_edit_directional_consistency_balanced": {
            "space": "pre_norm_query_space",
            **balanced_overall,
        },
        "same_edit_directional_consistency_balanced_by_category": (balanced_categories),
        "same_edit_directional_consistency_unit_query_space": {
            "space": "l2_normalized_query_space",
            **balanced_unit,
        },
        "same_edit_directional_consistency_unit_query_by_category": (balanced_unit_categories),
        "geometry_group_count_sensitivity": {
            "selection_role": (
                "robustness_only_not_an_independent_vote; reports whether "
                "the result depends strongly on the arbitrary minimum repeat "
                "count used for exact-caption groups"
            ),
            "pre_norm": pre_norm_group_count_sensitivity,
            "l2_normalized_query": normalized_group_count_sensitivity,
        },
        "retrieval_quality": {
            "common_full_gallery": native_retrieval_quality(retrieval_common),
            "published_native": select_published_native_policy(teacher_name, retrieval_native),
        },
        "teacher_edit_retrieval_sensitivity": {
            "selection_role": (
                "required functional-validity evidence but not a magnitude "
                "race. Large deletion damage is not automatically better. "
                "Single-caption effects can be attenuated by redundant human "
                "captions; all-text removal is only a coarse text-use guard."
            ),
            "scope": (
                "common full-gallery teacher-native scoring. Report target "
                "margin/rank change under caption removal and all-text removal; "
                "do not call this true TAPER slot necessity."
            ),
            "metrics": teacher_edit_retrieval_sensitivity_metrics(retrieval_common),
        },
        "compound_compositionality": {
            "selection_role": (
                "exploratory_only_not_for_teacher_selection: natural FashionIQ caption pairs are not guaranteed controlled atomic A/B factors"
            ),
            **compound_compositionality(
                q_null_pre=q_null_pre,
                q_single_1_pre=q_m2_pre,
                q_single_2_pre=q_m1_pre,
                q_full_pre=q_full_pre,
            ),
        },
        "caption_order_robustness": {
            "selection_role": (
                "exploratory_only_not_intervention_stability: swapping caption "
                "order tests wording/order robustness, not robustness of delta "
                "to different neutralization/intervention mechanisms"
            ),
            "geometry": caption_order_geometry_robustness(
                q_full_pre=q_full_pre,
                q_swap_pre=q_swap_pre,
                q_single_1_pre=q_m2_pre,
                q_single_2_pre=q_m1_pre,
                q_full=q_full,
                q_swap=q_swap,
            ),
            "retrieval": retrieval_order_robustness(retrieval_common),
        },
        "differentiable_intervention_probe": {
            "selection_role": (
                "gradient-access gate only: failure is fatal, pass is necessary but does NOT validate the exact TAPER erasure operator"
            ),
            **differentiability,
        },
        "controlled_atomic_geometry_gate": {
            "status": "not_run_in_natural_fashioniq_screen",
            "required_before_final_teacher_lock": True,
            "required_evidence": [
                "high-confidence controlled atomic edit/spans, independent of learned TAPER slots",
                "same atomic edit is more directionally consistent than different atomic edits",
                "both PRE-NORM functional space and retrieval-visible normalized-query space are reported",
                "effect coverage / near-zero rate is reported rather than silently dropping zero effects",
            ],
            "note": (
                "This gate validates atomic functional structure. Controlled "
                "A+B additivity is useful supporting evidence but is not a "
                "fatal requirement because TAPER execution is nonlinear."
            ),
        },
        "exact_intervention_credibility_gate": {
            "status": "not_run_in_generic_teacher_audit",
            "required_before_final_teacher_lock": True,
            "required_evidence": [
                "zero-erasure identity / no-op check",
                "hard-omission vs differentiable-effect direction agreement",
                "target-score or target-margin change agreement",
                "candidate score-change correlation or equivalent ranking agreement",
                "finite non-degenerate gradients through the exact intervention",
                "audit that selected text information cannot leak through an uncontrolled bypass",
            ],
        },
        "counterfactual_training_feasibility_gate": {
            "status": "not_profiled_in_generic_teacher_audit",
            "required_before_final_teacher_lock": True,
            "required_evidence": [
                "peak VRAM for representative B x L counterfactual batch",
                "wall-clock / throughput for L counterfactuals",
                "fits target hardware with teacher frozen but input gradients enabled",
            ],
        },
        "teacher_information_path_review": {
            "status": "requires_static_review_per_candidate",
            "required_before_final_teacher_lock": True,
            "questions": [
                "Can the exact selected text information be controlled before contextual leakage makes erasure meaningless?",
                "Does the composer contain an uncontrolled text bypass around the proposed intervention point?",
                "Does the teacher impose an explicit decomposition ontology that could confound TAPER mechanism claims?",
            ],
            "role": ("qualitative architecture/information-flow review; do not convert this into an arbitrary numeric quality score"),
        },
        "dual_encoder_bridge_gate": {
            "status": "conditional_not_applicable_until_encoder_choice",
            "required_only_if_taper_semantic_encoder_differs_from_teacher": True,
            "required_evidence": [
                "deterministic raw-text span/token mapping coverage",
                "unmapped-content rate and many-to-many mapping statistics",
                "hard-vs-trainable counterfactual agreement after the bridge",
            ],
        },
        "comparison_payload": comparison_payload,
    }

def git_repo_provenance(
    repo_root: Path,
    expected_audited_commit: str,
) -> dict:
    repo_root = repo_root.resolve()
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    try:
        head = git("rev-parse", "HEAD")
        tracked_changes = [line for line in git("status", "--porcelain", "--untracked-files=no").splitlines() if line.strip()]
        return {
            "status": "ok",
            "repo_root": str(repo_root),
            "head": head,
            "expected_audited_commit": expected_audited_commit,
            "matches_audited_snapshot": head == expected_audited_commit,
            "tracked_worktree_clean": len(tracked_changes) == 0,
            "tracked_changes": tracked_changes,
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "repo_root": str(repo_root),
            "expected_audited_commit": expected_audited_commit,
            "reason": f"{type(exc).__name__}: {exc}",
        }

@contextmanager
def capture_load_state_dict_results():
    original = torch.nn.Module.load_state_dict
    records = []
    def wrapped(module, *args, **kwargs):
        result = original(module, *args, **kwargs)
        records.append(
            {
                "module_class": module.__class__.__name__,
                "missing_keys": list(result.missing_keys),
                "unexpected_keys": list(result.unexpected_keys),
            }
        )
        return result
    torch.nn.Module.load_state_dict = wrapped
    try:
        yield records
    finally:
        torch.nn.Module.load_state_dict = original

def checkpoint_load_audit(
    records: list[dict],
    checkpoint_path: Path,
) -> dict:
    checkpoint_path = checkpoint_path.resolve()
    if not records:
        return {
            "status": "unavailable",
            "checkpoint": str(checkpoint_path),
            "checkpoint_size_bytes": checkpoint_path.stat().st_size,
            "reason": "No load_state_dict call was captured.",
        }
    last = records[-1]
    missing = last["missing_keys"]
    unexpected = last["unexpected_keys"]
    return {
        "status": "clean" if not missing and not unexpected else "warning",
        "checkpoint": str(checkpoint_path),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "loaded_module_class": last["module_class"],
        "missing_key_count": len(missing),
        "unexpected_key_count": len(unexpected),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "interpretation": ("warning can be intentional for fine-tuning checkpoints, but must be reviewed before final teacher lock"),
    }

def checkpoint_load_audit_allow_full_object(
    records: list[dict],
    checkpoint_path: Path,
    model,
) -> dict:
    if records:
        return checkpoint_load_audit(
            records,
            checkpoint_path,
        )
    checkpoint_path = checkpoint_path.resolve()
    if not isinstance(model, torch.nn.Module):
        return {
            "status": "unavailable",
            "checkpoint": str(checkpoint_path),
            "checkpoint_size_bytes": checkpoint_path.stat().st_size,
            "reason": ("No load_state_dict call captured and loaded object is not an nn.Module."),
        }
    return {
        "status": "clean",
        "load_mode": "full_serialized_module",
        "checkpoint": str(checkpoint_path),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "loaded_module_class": model.__class__.__name__,
        "missing_key_count": 0,
        "unexpected_key_count": 0,
        "missing_keys": [],
        "unexpected_keys": [],
        "interpretation": (
            "Checkpoint was loaded successfully as a complete "
            "serialized nn.Module; no load_state_dict call is "
            "expected for this upstream checkpoint format."
        ),
    }


def tgcir_origin_checkpoint_audit(
    records: list[dict],
    checkpoint_path: Path,
    model,
) -> dict:
    """
    Audit TG-CIR first-stage/origin checkpoint loading.

    Upstream TG-CIR intentionally reconstructs the text-side TokenLearner
    and text masks from their image-side counterparts when
    load_ckpt(..., is_origin=True) is used.
    """

    raw = checkpoint_load_audit(
        records,
        checkpoint_path,
    )

    expected_missing = {
        *{
            f"backbone.tokenlearn_text.tokenizers.{i}.conv.0.weight"
            for i in range(8)
        },
        *{
            f"backbone.tokenlearn_text.tokenizers.{i}.conv.0.bias"
            for i in range(8)
        },
        "backbone.masks_text.weight",
    }

    actual_missing = set(
        raw.get("missing_keys", [])
    )
    actual_unexpected = set(
        raw.get("unexpected_keys", [])
    )

    missing_pattern_ok = (
        actual_missing == expected_missing
    )

    unexpected_pattern_ok = (
        actual_unexpected == {"loss_T"}
    )

    recovery_checks = {}

    recovery_ok = True

    for i, (image_tokenizer, text_tokenizer) in enumerate(
        zip(
            model.backbone.tokenlearn.tokenizers,
            model.backbone.tokenlearn_text.tokenizers,
        )
    ):
        weight_equal = torch.equal(
            image_tokenizer.conv[0].weight,
            text_tokenizer.conv[0].weight,
        )

        bias_equal = torch.equal(
            image_tokenizer.conv[0].bias,
            text_tokenizer.conv[0].bias,
        )

        recovery_checks[f"tokenizer_{i}"] = {
            "weight_equal": weight_equal,
            "bias_equal": bias_equal,
        }

        recovery_ok &= (
            weight_equal
            and bias_equal
        )

    masks_equal = torch.equal(
        model.backbone.masks.weight,
        model.backbone.masks_text.weight,
    )

    recovery_checks["masks_equal"] = masks_equal
    recovery_ok &= masks_equal

    clean = (
        missing_pattern_ok
        and unexpected_pattern_ok
        and recovery_ok
    )

    return {
        "status": "clean" if clean else "warning",
        "load_mode": "tgcir_upstream_origin_reconstruction",
        "checkpoint": raw["checkpoint"],
        "checkpoint_size_bytes": raw[
            "checkpoint_size_bytes"
        ],

        # Preserve the raw load_state_dict evidence.
        "raw_missing_key_count": len(actual_missing),
        "raw_unexpected_key_count": len(actual_unexpected),
        "raw_missing_keys": sorted(actual_missing),
        "raw_unexpected_keys": sorted(actual_unexpected),

        "expected_origin_missing_pattern": (
            missing_pattern_ok
        ),
        "expected_origin_unexpected_pattern": (
            unexpected_pattern_ok
        ),
        "post_load_recovery_pass": recovery_ok,
        "post_load_recovery_checks": recovery_checks,

        # Existing tournament code expects these fields.
        # Semantically there are no unresolved missing/unexpected
        # inference weights after successful upstream recovery.
        "missing_key_count": 0 if clean else len(actual_missing),
        "unexpected_key_count": 0 if clean else len(actual_unexpected),
        "missing_keys": [] if clean else sorted(actual_missing),
        "unexpected_keys": [] if clean else sorted(actual_unexpected),

        "interpretation": (
            "TG-CIR first-stage checkpoint is intentionally incomplete "
            "for tokenlearn_text/masks_text. Upstream is_origin=True "
            "reconstructs those weights from the corresponding image-side "
            "modules. loss_T is an extra checkpoint entry not required by "
            "the loaded CIRPlus inference model."
            if clean
            else
            "TG-CIR checkpoint does not match the expected upstream "
            "first-stage/origin reconstruction pattern."
        ),
    }



def encoder_checkpoint_audit(
    records: list[dict],
    checkpoint_path: Path,
    model,
) -> dict:
    """
    ENCODER checkpoint audit.

    The audited FashionIQ checkpoint contains two obsolete
    binding_decoder.fc keys which are absent from the current audited
    upstream model definition. They are accepted only when:
      - no current model weights are missing,
      - the unexpected-key set is exactly the known two-key set,
      - the loaded model does not expose binding_decoder.fc,
      - those obsolete keys are absent from the current model state_dict.
    """
    raw = checkpoint_load_audit(
        records,
        checkpoint_path,
    )

    if raw.get("status") == "clean":
        raw["load_mode"] = "exact_state_dict"
        return raw

    expected_unexpected = {
        "backbone.binding_decoder.fc.0.weight",
        "backbone.binding_decoder.fc.0.bias",
    }

    missing = set(raw.get("missing_keys", []))
    unexpected = set(raw.get("unexpected_keys", []))

    backbone = getattr(model, "backbone", None)
    binding_decoder = getattr(
        backbone,
        "binding_decoder",
        None,
    )

    obsolete_fc_absent = (
        binding_decoder is not None
        and not hasattr(binding_decoder, "fc")
    )

    current_state_keys = set(model.state_dict().keys())
    obsolete_keys_absent_from_model = (
        expected_unexpected.isdisjoint(
            current_state_keys
        )
    )

    clean = (
        not missing
        and unexpected == expected_unexpected
        and obsolete_fc_absent
        and obsolete_keys_absent_from_model
    )

    return {
        "status": "clean" if clean else "warning",
        "load_mode": (
            "encoder_obsolete_checkpoint_keys_review"
        ),
        "checkpoint": raw["checkpoint"],
        "checkpoint_size_bytes": raw[
            "checkpoint_size_bytes"
        ],
        "loaded_module_class": raw.get(
            "loaded_module_class"
        ),

        # Preserve raw evidence.
        "raw_missing_key_count": len(missing),
        "raw_unexpected_key_count": len(unexpected),
        "raw_missing_keys": sorted(missing),
        "raw_unexpected_keys": sorted(unexpected),

        "expected_obsolete_key_pattern": (
            unexpected == expected_unexpected
        ),
        "binding_decoder_fc_absent": (
            obsolete_fc_absent
        ),
        "obsolete_keys_absent_from_current_state_dict": (
            obsolete_keys_absent_from_model
        ),

        # Tournament consumes these unresolved counts.
        "missing_key_count": (
            0 if clean else len(missing)
        ),
        "unexpected_key_count": (
            0 if clean else len(unexpected)
        ),
        "missing_keys": (
            [] if clean else sorted(missing)
        ),
        "unexpected_keys": (
            [] if clean else sorted(unexpected)
        ),

        "interpretation": (
            "The two raw unexpected keys belong to an obsolete "
            "binding_decoder.fc checkpoint submodule that is absent "
            "from the audited current ENCODER model. No current model "
            "weights are missing. Raw evidence is retained."
            if clean
            else
            "ENCODER checkpoint does not match the narrowly approved "
            "obsolete-key pattern and requires manual review."
        ),
    }


def csmcir_checkpoint_audit(
    records: list[dict],
    checkpoint_path: Path,
    model,
) -> dict:
    """
    CSMCIR retrieval checkpoint audit.

    token_importance is accepted only when:
      - no model keys are missing,
      - it is the sole unexpected checkpoint key,
      - the runtime model exposes token_importance,
      - token_importance is not registered in the current state_dict,
      - native retrieval inference methods do not reference it.

    Native-interface parity remains a separate mandatory gate.
    """
    import inspect

    raw = checkpoint_load_audit(
        records,
        checkpoint_path,
    )

    if raw.get("status") == "clean":
        raw["load_mode"] = "exact_state_dict"
        return raw

    missing = set(raw.get("missing_keys", []))
    unexpected = set(
        raw.get("unexpected_keys", [])
    )

    expected_unexpected = {"token_importance"}

    attribute_exists = hasattr(
        model,
        "token_importance",
    )

    registered_in_state_dict = (
        "token_importance"
        in model.state_dict()
    )

    method_review = {}
    inference_dependency_free = True

    for method_name in (
        "inference",
        "inference_tsen",
    ):
        if not hasattr(type(model), method_name):
            method_review[method_name] = {
                "source_available": False,
                "mentions_token_importance": None,
            }
            inference_dependency_free = False
            continue

        try:
            source = inspect.getsource(
                getattr(type(model), method_name)
            )
            mentions = (
                "token_importance" in source
            )
            method_review[method_name] = {
                "source_available": True,
                "mentions_token_importance": (
                    mentions
                ),
            }
            if mentions:
                inference_dependency_free = False
        except Exception as exc:
            method_review[method_name] = {
                "source_available": False,
                "mentions_token_importance": None,
                "reason": (
                    f"{type(exc).__name__}: {exc}"
                ),
            }
            inference_dependency_free = False

    clean = (
        not missing
        and unexpected == expected_unexpected
        and attribute_exists
        and not registered_in_state_dict
        and inference_dependency_free
    )

    return {
        "status": "clean" if clean else "warning",
        "load_mode": (
            "csmcir_native_retrieval_checkpoint_review"
        ),
        "checkpoint": raw["checkpoint"],
        "checkpoint_size_bytes": raw[
            "checkpoint_size_bytes"
        ],
        "loaded_module_class": raw.get(
            "loaded_module_class"
        ),

        # Preserve raw evidence.
        "raw_missing_key_count": len(missing),
        "raw_unexpected_key_count": len(unexpected),
        "raw_missing_keys": sorted(missing),
        "raw_unexpected_keys": sorted(unexpected),

        "expected_token_importance_pattern": (
            unexpected == expected_unexpected
        ),
        "token_importance_attribute_exists": (
            attribute_exists
        ),
        "token_importance_registered_in_state_dict": (
            registered_in_state_dict
        ),
        "native_retrieval_dependency_free": (
            inference_dependency_free
        ),
        "native_retrieval_method_review": (
            method_review
        ),

        "missing_key_count": (
            0 if clean else len(missing)
        ),
        "unexpected_key_count": (
            0 if clean else len(unexpected)
        ),
        "missing_keys": (
            [] if clean else sorted(missing)
        ),
        "unexpected_keys": (
            [] if clean else sorted(unexpected)
        ),

        "interpretation": (
            "The raw checkpoint contains token_importance, but the "
            "audited runtime model does not register it in state_dict "
            "and the native retrieval inference methods do not "
            "reference it. Native-interface parity remains an "
            "independent required fidelity gate."
            if clean
            else
            "CSMCIR checkpoint does not satisfy the narrowly approved "
            "native-retrieval exception and requires manual review."
        ),
    }


def git_repo_provenance_with_compat_patch(
    repo_root: Path,
    expected_audited_commit: str,
    allowed_file: str,
    expected_diff_sha256: str,
    reason: str,
) -> dict:
    """
    Preserve tracked_worktree_clean truthfully, but allow one exact,
    fingerprinted compatibility patch as reproducible provenance.
    """
    import hashlib

    base = git_repo_provenance(
        repo_root,
        expected_audited_commit,
    )

    if base.get("status") != "ok":
        base["provenance_pass"] = False
        return base

    # A fully clean matching snapshot always passes.
    if (
        base.get("matches_audited_snapshot") is True
        and base.get("tracked_worktree_clean") is True
    ):
        base["approved_compatibility_patch"] = False
        base["provenance_pass"] = True
        return base

    repo_root = repo_root.resolve()

    tracked_changes = base.get(
        "tracked_changes",
        [],
    )

    # Do not parse paths from `git status --porcelain` here.
    # git_repo_provenance() currently strips stdout, which can remove
    # the leading status-column space from entries such as:
    #   " M path/to/file"
    # Use git diff --name-only instead for exact tracked file scope.
    try:
        changed_paths_result = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "diff",
                "--name-only",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        changed_paths = [
            line.strip()
            for line in changed_paths_result.stdout.splitlines()
            if line.strip()
        ]

        result = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "diff",
                "--no-ext-diff",
                "--no-color",
                "HEAD",
                "--",
                allowed_file,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        diff_text = result.stdout
        actual_sha256 = hashlib.sha256(
            diff_text.encode("utf-8")
        ).hexdigest()
    except Exception as exc:
        base["approved_compatibility_patch"] = False
        base["provenance_pass"] = False
        base["compatibility_patch_error"] = (
            f"{type(exc).__name__}: {exc}"
        )
        return base

    exact_file_scope = (
        changed_paths == [allowed_file]
    )

    exact_diff_match = (
        actual_sha256
        == expected_diff_sha256
    )

    approved = (
        base.get("matches_audited_snapshot") is True
        and exact_file_scope
        and exact_diff_match
    )

    base.update(
        {
            # Keep this FALSE when the repo is patched.
            "tracked_worktree_clean": (
                base.get(
                    "tracked_worktree_clean"
                )
            ),
            "approved_compatibility_patch": (
                approved
            ),
            "compatibility_patch": {
                "file": allowed_file,
                "reason": reason,
                "expected_diff_sha256": (
                    expected_diff_sha256
                ),
                "actual_diff_sha256": (
                    actual_sha256
                ),
                "exact_file_scope": (
                    exact_file_scope
                ),
                "exact_diff_match": (
                    exact_diff_match
                ),
            },
            "provenance_pass": approved,
        }
    )

    return base


def tensor_parity(
    ours: torch.Tensor,
    native: torch.Tensor,
    name: str,
    atol: float = 2e-5,
    rtol: float = 2e-4,
) -> dict:
    ours = ours.detach().float().cpu()
    native = native.detach().float().cpu()
    if ours.shape != native.shape:
        return {
            "status": "fail",
            "name": name,
            "reason": (f"shape mismatch ours={tuple(ours.shape)} native={tuple(native.shape)}"),
        }
    diff = (ours - native).abs()
    cosine = (
        F.cosine_similarity(
            ours.reshape(ours.shape[0], -1),
            native.reshape(native.shape[0], -1),
            dim=-1,
        )
        .mean()
        .item()
    )
    return {
        "status": ("pass" if torch.allclose(ours, native, atol=atol, rtol=rtol) else "fail"),
        "name": name,
        "shape": list(ours.shape),
        "max_abs_error": diff.max().item(),
        "mean_abs_error": diff.mean().item(),
        "mean_cosine": cosine,
        "atol": atol,
        "rtol": rtol,
    }

def assert_native_parity(parity: dict, teacher_name: str) -> None:
    if parity.get("status") != "pass":
        raise RuntimeError(f"{teacher_name} native-interface parity failed: {parity}")

def encoder_native_parity_probe(
    model,
    preprocess,
    correction_dicts,
    cases,
    image_root,
    device,
    adapter,
) -> dict:
    batch = cases[: min(2, len(cases))]
    images = load_case_image_batch(batch, image_root, preprocess).to(device)
    texts = build_encoder_query_texts(batch, adapter, correction_dicts)["full"]
    with torch.no_grad():
        _, ours = adapter.compose_query(model, images, texts)
        native = model.extract_retrieval_compose(images, texts)
        target_cases = [{**case, "reference_id": case["target_id"]} for case in batch]
        target_images = load_case_image_batch(target_cases, image_root, preprocess).to(device)
        target_features = model.extract_retrieval_target(target_images)
        ours_scores = score_vector_gallery(ours, target_features)
        native_scores = native @ target_features.T
    qcheck = tensor_parity(ours, native, "ENCODER adapter query vs extract_retrieval_compose")
    scheck = tensor_parity(ours_scores, native_scores, "ENCODER scorer vs native dot product")
    return {
        "status": ("pass" if qcheck["status"] == "pass" and scheck["status"] == "pass" else "fail"),
        "query": qcheck,
        "scorer": scheck,
    }

def tme_native_parity_probe(
    model,
    preprocess,
    txt_processor,
    cases,
    image_root,
    device,
    adapter,
) -> dict:
    batch = cases[: min(2, len(cases))]
    images = load_case_image_batch(batch, image_root, preprocess).to(device)
    vit_states = adapter.encode_vit_states(model, images)
    reference_tokens = adapter.encode_reference(model, vit_states)
    raw_texts = [case["full_text"] for case in batch]
    with torch.no_grad():
        _, ours = adapter.compose_query(model, reference_tokens, raw_texts, txt_processor)
        processed = [txt_processor(text) for text in raw_texts]
        text_tokens = model.tokenizer(
            processed,
            padding="max_length",
            truncation=True,
            max_length=model.max_txt_len,
            return_tensors="pt",
        ).to(device)
        native_query = model.encode_fusion(reference_tokens, text_tokens)
        target_cases = [{**case, "reference_id": case["target_id"]} for case in batch]
        target_images = load_case_image_batch(target_cases, image_root, preprocess).to(device)
        target_vit = adapter.encode_vit_states(model, target_images)
        f_target = model.encode_image(target_vit)
        z_target = F.normalize(model.vision_proj(f_target), dim=-1)
        ours_scores = score_token_gallery(ours, z_target)
        native_scores = model.inference(reference_tokens, f_target, processed)
        if native_scores.ndim == 1:
            native_scores = native_scores.unsqueeze(0)
    qcheck = tensor_parity(ours, native_query, "TME adapter query vs encode_fusion")
    scheck = tensor_parity(ours_scores, native_scores, "TME scorer vs model.inference")
    return {
        "status": ("pass" if qcheck["status"] == "pass" and scheck["status"] == "pass" else "fail"),
        "query": qcheck,
        "scorer": scheck,
    }

def sprc_native_parity_probe(
    model,
    preprocess,
    txt_processor,
    cases,
    image_root,
    device,
    adapter,
) -> dict:
    batch = cases[: min(2, len(cases))]
    images = load_case_image_batch(batch, image_root, preprocess).to(device)
    reference_embeds = adapter.encode_reference(model, images)
    raw_texts = [case["full_text"] for case in batch]
    with torch.no_grad():
        _, ours = adapter.compose_query(model, reference_embeds, raw_texts, txt_processor)
        target_cases = [{**case, "reference_id": case["target_id"]} for case in batch]
        target_images = load_case_image_batch(target_cases, image_root, preprocess).to(device)
        target_features, _ = model.extract_target_features(target_images)
        processed = [txt_processor(text) for text in raw_texts]
        ours_scores = score_token_gallery(ours, target_features)
        native_scores = model.inference(reference_embeds, target_features, processed)
        if native_scores.ndim == 1:
            native_scores = native_scores.unsqueeze(0)
    scheck = tensor_parity(ours_scores, native_scores, "SPRC adapter/scorer vs model.inference")
    return {
        "status": scheck["status"],
        "scorer": scheck,
        "note": ("SPRC native inference does not expose the intermediate query; parity is tested end-to-end at the score boundary."),
    }

def hint_native_parity_probe(
    model,
    preprocess,
    txt_processor,
    correction_dicts,
    cases,
    image_root,
    device,
    adapter,
) -> dict:
    batch = cases[: min(2, len(cases))]
    images = load_case_image_batch(
        batch,
        image_root,
        preprocess,
    ).to(device)
    reference_embeds = adapter.encode_reference(
        model=model,
        reference_images=images,
    )
    raw_texts = build_hint_query_texts(
        batch,
        adapter,
        correction_dicts,
    )["full"]
    with torch.no_grad():
        _, ours = adapter.compose_query_from_reference(
            model=model,
            reference_embeds=reference_embeds,
            texts=raw_texts,
            txt_processor=txt_processor,
        )
        processed = [txt_processor(text) for text in raw_texts]
        native_query_4d = model.extract_retrieval_compose(
            images,
            processed,
        )
        native_query = native_query_4d.squeeze(1).squeeze(1)
        target_cases = [
            {
                **case,
                "reference_id": case["target_id"],
            }
            for case in batch
        ]
        target_images = load_case_image_batch(
            target_cases,
            image_root,
            preprocess,
        ).to(device)
        native_target = model.extract_retrieval_target(target_images)
        target_tokens = native_target.permute(
            0,
            2,
            1,
        ).contiguous()
        ours_scores = score_token_gallery(
            ours,
            target_tokens,
        )
        native_scores = (
            torch.matmul(
                native_query_4d,
                native_target,
            )
            .squeeze(-2)
            .max(dim=-1)
            .values
        )
    qcheck = tensor_parity(
        ours,
        native_query,
        "HINT adapter query vs extract_retrieval_compose",
    )
    scheck = tensor_parity(
        ours_scores,
        native_scores,
        "HINT scorer vs upstream matmul-max scorer",
    )
    return {
        "status": ("pass" if qcheck["status"] == "pass" and scheck["status"] == "pass" else "fail"),
        "query": qcheck,
        "scorer": scheck,
    }

def qure_native_parity_probe(
    model,
    preprocess,
    txt_processor,
    cases,
    image_root,
    device,
    adapter,
) -> dict:
    batch = cases[: min(2, len(cases))]
    images = load_case_image_batch(
        batch,
        image_root,
        preprocess,
    ).to(device)
    reference_states = adapter.encode_reference(
        model=model,
        images=images,
    )
    raw_texts = [case["full_text"] for case in batch]
    with torch.no_grad():
        _, ours_query = adapter.compose_query(
            model=model,
            reference_states=reference_states,
            texts=raw_texts,
            txt_processor=txt_processor,
        )
        fake_query_loader = [
            (
                images,
                raw_texts,
                [case["target_id"] for case in batch],
            )
        ]
        native_query, native_target_names = model.extract_query_features_fiq(
            fake_query_loader,
            True,
            {"eval": txt_processor},
            device,
        )
        target_cases = [
            {
                **case,
                "reference_id": case["target_id"],
            }
            for case in batch
        ]
        target_images = load_case_image_batch(
            target_cases,
            image_root,
            preprocess,
        ).to(device)
        ours_target = adapter.encode_target(
            model=model,
            images=target_images,
        )
        fake_target_loader = [
            (
                target_images,
                [case["target_id"] for case in batch],
            )
        ]
        native_target, native_target_names_2 = model.extract_target_features(
            fake_target_loader,
            True,
            device,
        )
        ours_scores = score_token_gallery(
            ours_query,
            ours_target,
        )
        native_scores = model.score(
            native_query,
            native_target,
        )
    qcheck = tensor_parity(
        ours_query,
        native_query,
        "QuRe adapter query vs extract_query_features_fiq",
    )
    tcheck = tensor_parity(
        ours_target,
        native_target,
        "QuRe target tokens vs extract_target_features",
    )
    scheck = tensor_parity(
        ours_scores,
        native_scores,
        "QuRe score_token_gallery vs model.score",
    )
    return {
        "status": ("pass" if (qcheck["status"] == "pass" and tcheck["status"] == "pass" and scheck["status"] == "pass") else "fail"),
        "query": qcheck,
        "target": tcheck,
        "scorer": scheck,
    }


def tgcir_native_parity_probe(
    model,
    preprocess,
    txt_processor,
    cases,
    image_root,
    device,
    adapter,
):
    batch = cases[: min(2, len(cases))]

    images = load_case_image_batch(
        batch,
        image_root,
        preprocess,
    ).to(device)

    reference_tokens = adapter.encode_reference(
        model,
        images,
    )

    texts = [
        case["full_text"]
        for case in batch
    ]

    with torch.no_grad():
        _, ours_query = adapter.compose_query(
            model,
            reference_tokens,
            texts,
            txt_processor,
        )

        native_query = model.img_txt_fusion(
            reference_tokens,
            texts,
        )

        target_cases = [
            {
                **case,
                "reference_id": case["target_id"],
            }
            for case in batch
        ]

        target_images = load_case_image_batch(
            target_cases,
            image_root,
            preprocess,
        ).to(device)

        target_features = adapter.encode_target(
            model,
            target_images,
        )

        ours_scores = score_vector_gallery(
            ours_query,
            target_features,
        )

        native_scores = (
            native_query
            @ target_features.T
        )

    qcheck = tensor_parity(
        ours_query,
        native_query,
        "TG-CIR adapter query vs img_txt_fusion",
    )

    scheck = tensor_parity(
        ours_scores,
        native_scores,
        "TG-CIR scorer vs native dot product",
    )

    return {
        "status": (
            "pass"
            if (
                qcheck["status"] == "pass"
                and scheck["status"] == "pass"
            )
            else "fail"
        ),
        "query": qcheck,
        "scorer": scheck,
    }


def csmcir_native_parity_probe(
    model,
    preprocess,
    txt_processor,
    cases,
    image_root,
    device,
    adapter,
    caption_dicts,
):
    batch = cases[: min(2, len(cases))]

    images = load_case_image_batch(
        batch,
        image_root,
        preprocess,
    ).to(device)

    reference_embeds = adapter.encode_reference(
        model,
        images,
    )

    raw_texts = [
        case["full_text"]
        for case in batch
    ]

    processed = [
        txt_processor(text)
        for text in raw_texts
    ]

    target_cases = [
        {
            **case,
            "reference_id": case["target_id"],
        }
        for case in batch
    ]

    target_images = load_case_image_batch(
        target_cases,
        image_root,
        preprocess,
    ).to(device)

    target_captions = [
        adapter.target_caption(
            caption_dicts,
            case["category"],
            case["target_id"],
        )
        for case in batch
    ]

    with torch.no_grad():
        _, ours_query = adapter.compose_query(
            model,
            reference_embeds,
            raw_texts,
            txt_processor,
        )

        target_features = adapter.encode_target(
            model,
            target_images,
            target_captions,
        )

        ours_scores = score_token_gallery(
            ours_query,
            target_features,
        )

        (
            native_token_scores,
            native_query,
        ) = model.inference_tsen(
            reference_embeds,
            target_features,
            processed,
            processed,
        )

        native_scores = (
            native_token_scores
            .max(dim=-1)
            .values
        )

    qcheck = tensor_parity(
        ours_query,
        native_query,
        "CSMCIR adapter query vs inference_tsen",
    )

    scheck = tensor_parity(
        ours_scores,
        native_scores,
        "CSMCIR scorer vs native inference",
    )

    return {
        "status": (
            "pass"
            if (
                qcheck["status"] == "pass"
                and scheck["status"] == "pass"
            )
            else "fail"
        ),
        "query": qcheck,
        "scorer": scheck,
    }


def run_encoder(args, cases, device):
    _ensure_repo_on_path()
    from teacher.adapters import encoder as adapter
    with capture_load_state_dict_results() as load_records:
        model, preprocess_train, preprocess_val = adapter.build_encoder(
            args.encoder_root.resolve(),
            args.checkpoint.resolve(),
            device,
        )
    integrity = {
        "checkpoint_load": encoder_checkpoint_audit(
            load_records,
            args.checkpoint,
            model,
        ),
        "upstream_repo": git_repo_provenance(
            args.encoder_root,
            "29a2a31d6a56f677bf450c3be7cdaef423fb7018",
        ),
    }
    correction_dicts = adapter.load_correction_dicts(args.correction_root.resolve())
    parity = encoder_native_parity_probe(
        model,
        preprocess_val,
        correction_dicts,
        cases,
        args.image_root,
        device,
        adapter,
    )
    integrity["native_interface_parity"] = parity
    assert_native_parity(parity, "ENCODER")
    query_store = collect_encoder_queries(
        model=model,
        preprocess=preprocess_val,
        correction_dicts=correction_dicts,
        cases=cases,
        image_root=args.image_root,
        device=device,
        batch_size=args.batch_size,
        adapter=adapter,
    )
    query_store_native = collect_encoder_queries(
        model=model,
        preprocess=preprocess_train,
        correction_dicts=correction_dicts,
        cases=cases,
        image_root=args.image_root,
        device=device,
        batch_size=args.batch_size,
        adapter=adapter,
    )
    gallery_cache = {}
    def gallery_builder(ids, category):
        if category not in gallery_cache:
            full_ids = load_split_ids(args.split_root, category)
            full_features = build_encoder_gallery(
                model=model,
                preprocess=preprocess_val,
                image_ids=full_ids,
                category=category,
                image_root=args.image_root,
                device=device,
                batch_size=args.gallery_batch_size,
            )
            gallery_cache[category] = (
                {name: i for i, name in enumerate(full_ids)},
                full_features,
            )
        name_to_idx, full_features = gallery_cache[category]
        indices = [name_to_idx[name] for name in ids]
        return full_features[indices]
    retrieval_common = score_all_queries(
        query_store=query_store,
        cases=cases,
        gallery_builder=gallery_builder,
        score_fn=score_vector_gallery,
        device=device,
        score_batch_size=args.score_batch_size,
        gallery_id_provider=full_gallery_provider(args.split_root),
        protocol_name=("common_full_gallery_deterministic_query_preprocess_val"),
    )
    retrieval_native = score_all_queries(
        query_store=query_store_native,
        cases=cases,
        gallery_builder=gallery_builder,
        score_fn=score_vector_gallery,
        device=device,
        score_batch_size=args.score_batch_size,
        gallery_id_provider=pair_union_gallery_provider,
        protocol_name=("ENCODER_upstream_val-split_pair-union_gallery_query_preprocess_train_target_preprocess_val"),
    )
    diff = probe_encoder_differentiability(
        model=model,
        preprocess=preprocess_val,
        correction_dicts=correction_dicts,
        case=cases[0],
        image_root=args.image_root,
        device=device,
        adapter=adapter,
    )
    return query_store, retrieval_common, retrieval_native, diff, integrity

def run_tme(args, cases, device):
    _ensure_repo_on_path()
    from teacher.adapters import tme as adapter
    with capture_load_state_dict_results() as load_records:
        model, txt_processor, preprocess = adapter.build_tme(
            args.tme_root.resolve(),
            args.checkpoint.resolve(),
            device,
        )
    integrity = {
        "checkpoint_load": checkpoint_load_audit(load_records, args.checkpoint),
        "upstream_repo": git_repo_provenance(
            args.tme_root,
            "7fe811992820d30828e24065eb0c7a7ba099dd3b",
        ),
    }
    parity = tme_native_parity_probe(
        model,
        preprocess,
        txt_processor,
        cases,
        args.image_root,
        device,
        adapter,
    )
    integrity["native_interface_parity"] = parity
    assert_native_parity(parity, "TME")
    query_store = collect_qformer_queries(
        model=model,
        preprocess=preprocess,
        txt_processor=txt_processor,
        cases=cases,
        image_root=args.image_root,
        device=device,
        batch_size=args.batch_size,
        teacher="tme",
        adapter=adapter,
    )
    def gallery_builder(ids, category):
        return build_tme_gallery(
            model=model,
            preprocess=preprocess,
            image_ids=ids,
            category=category,
            image_root=args.image_root,
            device=device,
            batch_size=args.gallery_batch_size,
            adapter=adapter,
        )
    retrieval_common = score_all_queries(
        query_store=query_store,
        cases=cases,
        gallery_builder=gallery_builder,
        score_fn=score_token_gallery,
        device=device,
        score_batch_size=args.score_batch_size,
        gallery_id_provider=full_gallery_provider(args.split_root),
        protocol_name="TME_upstream_full_gallery",
    )
    retrieval_native = retrieval_common
    diff = probe_tme_differentiability(
        model=model,
        preprocess=preprocess,
        txt_processor=txt_processor,
        case=cases[0],
        image_root=args.image_root,
        device=device,
        adapter=adapter,
    )
    return query_store, retrieval_common, retrieval_native, diff, integrity

def run_sprc(args, cases, device):
    _ensure_repo_on_path()
    from teacher.adapters import sprc as adapter
    with capture_load_state_dict_results() as load_records:
        model, txt_processor, preprocess = adapter.build_sprc(
            args.sprc_root.resolve(),
            args.checkpoint.resolve(),
            args.backbone,
            device,
        )
    integrity = {
        "checkpoint_load": checkpoint_load_audit(load_records, args.checkpoint),
        "upstream_repo": git_repo_provenance(
            args.sprc_root,
            "2935a5397732260d1db6fa577e5926f963e36f0f",
        ),
    }
    parity = sprc_native_parity_probe(
        model,
        preprocess,
        txt_processor,
        cases,
        args.image_root,
        device,
        adapter,
    )
    integrity["native_interface_parity"] = parity
    assert_native_parity(parity, "SPRC")
    query_store = collect_qformer_queries(
        model=model,
        preprocess=preprocess,
        txt_processor=txt_processor,
        cases=cases,
        image_root=args.image_root,
        device=device,
        batch_size=args.batch_size,
        teacher="sprc",
        adapter=adapter,
    )
    def gallery_builder(ids, category):
        return build_sprc_gallery(
            model=model,
            preprocess=preprocess,
            image_ids=ids,
            category=category,
            image_root=args.image_root,
            device=device,
            batch_size=args.gallery_batch_size,
        )
    retrieval_common = score_all_queries(
        query_store=query_store,
        cases=cases,
        gallery_builder=gallery_builder,
        score_fn=score_token_gallery,
        device=device,
        score_batch_size=args.score_batch_size,
        gallery_id_provider=full_gallery_provider(args.split_root),
        protocol_name="SPRC_upstream_full_gallery",
    )
    retrieval_native = retrieval_common
    diff = probe_sprc_differentiability(
        model=model,
        preprocess=preprocess,
        txt_processor=txt_processor,
        case=cases[0],
        image_root=args.image_root,
        device=device,
        adapter=adapter,
    )
    return query_store, retrieval_common, retrieval_native, diff, integrity


def run_tgcir(args, cases, device):
    _ensure_repo_on_path()
    from teacher.adapters import tgcir as adapter

    tgcir_root = args.tgcir_root.resolve()
    checkpoint_path = args.checkpoint.resolve()

    with capture_load_state_dict_results() as load_records:
        (
            model,
            txt_processor,
            preprocess,
        ) = adapter.build_tgcir(
            tgcir_root,
            checkpoint_path,
            device,
        )

    integrity = {
        "checkpoint_load": tgcir_origin_checkpoint_audit(
            load_records,
            checkpoint_path,
            model,
        ),
        "upstream_repo": git_repo_provenance(
            tgcir_root,
            "84005f643ccaacf999982694ad5631df92cef098",
        ),
    }

    parity = tgcir_native_parity_probe(
        model,
        preprocess,
        txt_processor,
        cases,
        args.image_root,
        device,
        adapter,
    )

    integrity["native_interface_parity"] = parity
    assert_native_parity(parity, "TG-CIR")

    query_store = collect_qformer_queries(
        model=model,
        preprocess=preprocess,
        txt_processor=txt_processor,
        cases=cases,
        image_root=args.image_root,
        device=device,
        batch_size=args.batch_size,
        teacher="tgcir",
        adapter=adapter,
    )

    def gallery_builder(ids, category):
        return build_tgcir_gallery(
            model=model,
            preprocess=preprocess,
            image_ids=ids,
            category=category,
            image_root=args.image_root,
            device=device,
            batch_size=args.gallery_batch_size,
            adapter=adapter,
        )

    retrieval_common = score_all_queries(
        query_store=query_store,
        cases=cases,
        gallery_builder=gallery_builder,
        score_fn=score_vector_gallery,
        device=device,
        score_batch_size=args.score_batch_size,
        gallery_id_provider=full_gallery_provider(
            args.split_root
        ),
        protocol_name=(
            "TG-CIR_native_full_gallery_dot_product"
        ),
    )

    retrieval_native = retrieval_common

    diff = probe_tgcir_differentiability(
        model=model,
        preprocess=preprocess,
        txt_processor=txt_processor,
        case=cases[0],
        image_root=args.image_root,
        device=device,
        adapter=adapter,
    )

    return (
        query_store,
        retrieval_common,
        retrieval_native,
        diff,
        integrity,
    )


def run_csmcir(args, cases, device):
    _ensure_repo_on_path()
    from teacher.adapters import csmcir as adapter

    with capture_load_state_dict_results() as load_records:
        (
            model,
            txt_processor,
            preprocess,
        ) = adapter.build_csmcir(
            args.csmcir_root.resolve(),
            args.checkpoint.resolve(),
            device,
        )

    caption_dicts = adapter.load_target_captions(
        args.csmcir_root.resolve()
    )

    integrity = {
        "checkpoint_load": csmcir_checkpoint_audit(
            load_records,
            args.checkpoint,
            model,
        ),
        "upstream_repo": git_repo_provenance_with_compat_patch(
            args.csmcir_root.resolve(),
            "774f94e2076ff17ea91703a6239d2a08f0e1a44e",
            "src/lavis/models/__init__.py",
            "87fb74265ee2e2fb712be2a4e75cc064d63e5a500ae2a02fcfd9863b77a633a8",
            (
                "Disable imports of modules absent from the audited "
                "upstream repository snapshot; retrieval model "
                "implementation itself is not modified."
            ),
        ),
    }

    parity = csmcir_native_parity_probe(
        model,
        preprocess,
        txt_processor,
        cases,
        args.image_root,
        device,
        adapter,
        caption_dicts,
    )

    integrity["native_interface_parity"] = parity
    assert_native_parity(parity, "CSMCIR")

    query_store = collect_qformer_queries(
        model=model,
        preprocess=preprocess,
        txt_processor=txt_processor,
        cases=cases,
        image_root=args.image_root,
        device=device,
        batch_size=args.batch_size,
        teacher="csmcir",
        adapter=adapter,
    )

    def gallery_builder(ids, category):
        return build_csmcir_gallery(
            model=model,
            preprocess=preprocess,
            image_ids=ids,
            category=category,
            image_root=args.image_root,
            device=device,
            batch_size=args.gallery_batch_size,
            adapter=adapter,
            caption_dicts=caption_dicts,
        )

    retrieval_common = score_all_queries(
        query_store=query_store,
        cases=cases,
        gallery_builder=gallery_builder,
        score_fn=score_token_gallery,
        device=device,
        score_batch_size=args.score_batch_size,
        gallery_id_provider=full_gallery_provider(
            args.split_root
        ),
        protocol_name=(
            "CSMCIR_native_target_caption_gallery_"
            "QFormer_token_max"
        ),
    )

    retrieval_native = retrieval_common

    diff = probe_csmcir_differentiability(
        model=model,
        preprocess=preprocess,
        txt_processor=txt_processor,
        case=cases[0],
        image_root=args.image_root,
        device=device,
        adapter=adapter,
    )

    return (
        query_store,
        retrieval_common,
        retrieval_native,
        diff,
        integrity,
    )


def run_hint(args, cases, device):
    _ensure_repo_on_path()
    from teacher.adapters import hint as adapter
    hint_root = args.hint_root.resolve()
    checkpoint_path = args.checkpoint.resolve()
    correction_root = args.correction_root.resolve()
    with capture_load_state_dict_results() as load_records:
        model, txt_processor, preprocess = adapter.build_hint(
            hint_root=hint_root,
            checkpoint_path=checkpoint_path,
            device=device,
        )
    integrity = {
        "checkpoint_load": checkpoint_load_audit_allow_full_object(
            load_records,
            checkpoint_path,
            model,
        ),
        "upstream_repo": git_repo_provenance(
            hint_root,
            "bec50b6c8c19111893b617979502b948d1cea5b2",
        ),
    }
    correction_dicts = adapter.load_correction_dicts(correction_root)
    parity = hint_native_parity_probe(
        model=model,
        preprocess=preprocess,
        txt_processor=txt_processor,
        correction_dicts=correction_dicts,
        cases=cases,
        image_root=args.image_root,
        device=device,
        adapter=adapter,
    )
    integrity["native_interface_parity"] = parity
    assert_native_parity(
        parity,
        "HINT",
    )
    query_store = collect_hint_queries(
        model=model,
        preprocess=preprocess,
        txt_processor=txt_processor,
        correction_dicts=correction_dicts,
        cases=cases,
        image_root=args.image_root,
        device=device,
        batch_size=args.batch_size,
        adapter=adapter,
    )
    def gallery_builder(ids, category):
        return build_hint_gallery(
            model=model,
            preprocess=preprocess,
            image_ids=ids,
            category=category,
            image_root=args.image_root,
            device=device,
            batch_size=args.gallery_batch_size,
        )
    retrieval_common = score_all_queries(
        query_store=query_store,
        cases=cases,
        gallery_builder=gallery_builder,
        score_fn=score_token_gallery,
        device=device,
        score_batch_size=args.score_batch_size,
        gallery_id_provider=full_gallery_provider(args.split_root),
        protocol_name=("HINT_common_full_gallery_native_matmul_max"),
    )
    retrieval_native = retrieval_common
    diff = probe_hint_differentiability(
        model=model,
        preprocess=preprocess,
        txt_processor=txt_processor,
        correction_dicts=correction_dicts,
        case=cases[0],
        image_root=args.image_root,
        device=device,
        adapter=adapter,
    )
    return (
        query_store,
        retrieval_common,
        retrieval_native,
        diff,
        integrity,
    )

def run_qure(
    args,
    cases,
    device,
):
    _ensure_repo_on_path()
    from teacher.adapters import qure as adapter
    qure_root = args.qure_root.resolve()
    qure_config = args.qure_config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    with capture_load_state_dict_results() as load_records:
        (
            model,
            txt_processor,
            preprocess,
        ) = adapter.build_qure(
            qure_root=qure_root,
            config_path=qure_config,
            checkpoint_path=checkpoint_path,
            device=device,
        )
    integrity = {
        "checkpoint_load": (
            checkpoint_load_audit(
                load_records,
                checkpoint_path,
            )
        ),
        "upstream_repo": (
            git_repo_provenance(
                qure_root,
                "6a50a27c307e151b95533d05ffdfa126fbe5550a",
            )
        ),
    }
    parity = qure_native_parity_probe(
        model=model,
        preprocess=preprocess,
        txt_processor=txt_processor,
        cases=cases,
        image_root=args.image_root,
        device=device,
        adapter=adapter,
    )
    integrity["native_interface_parity"] = parity
    assert_native_parity(
        parity,
        "QuRe",
    )
    query_store = collect_qformer_queries(
        model=model,
        preprocess=preprocess,
        txt_processor=txt_processor,
        cases=cases,
        image_root=args.image_root,
        device=device,
        batch_size=args.batch_size,
        teacher="qure",
        adapter=adapter,
    )
    def gallery_builder(
        ids,
        category,
    ):
        return build_qure_gallery(
            model=model,
            preprocess=preprocess,
            image_ids=ids,
            category=category,
            image_root=args.image_root,
            device=device,
            batch_size=(args.gallery_batch_size),
            adapter=adapter,
        )
    retrieval_common = score_all_queries(
        query_store=query_store,
        cases=cases,
        gallery_builder=gallery_builder,
        score_fn=score_token_gallery,
        device=device,
        score_batch_size=(args.score_batch_size),
        gallery_id_provider=(full_gallery_provider(args.split_root)),
        protocol_name=("QuRe_common_full_gallery_native_query_token_max_scorer"),
    )
    retrieval_native = retrieval_common
    diff = probe_qure_differentiability(
        model=model,
        preprocess=preprocess,
        txt_processor=txt_processor,
        case=cases[0],
        image_root=args.image_root,
        device=device,
        adapter=adapter,
    )
    return (
        query_store,
        retrieval_common,
        retrieval_native,
        diff,
        integrity,
    )

def discover_correction_root(*roots: Path) -> Path:
    required = {
        "correction_dict_dress.json",
        "correction_dict_shirt.json",
        "correction_dict_toptee.json",
    }
    for root in roots:
        if root is None or not root.exists():
            continue
        direct = {path.name for path in root.glob("correction_dict_*.json")}
        if required.issubset(direct):
            return root
        for dress_path in root.rglob("correction_dict_dress.json"):
            parent = dress_path.parent
            names = {path.name for path in parent.glob("correction_dict_*.json")}
            if required.issubset(names):
                return parent
    raise FileNotFoundError("Could not auto-discover all three ENCODER FashionIQ correction dictionaries. Pass --correction-root explicitly.")

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Full TAPER teacher audit: geometry + balanced edit consistency + "
            "native FashionIQ retrieval + retrieval necessity + compound "
            "additivity + order counterfactual stability + differentiability."
        )
    )
    parser.add_argument(
        "--teacher",
        required=True,
        choices=("encoder", "tme", "sprc", "tgcir", "csmcir"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("teacher/audit/fashioniq_val_cases.json"),
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("data/FashionIQ/images"),
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=Path("data/FashionIQ/image_splits"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-output", type=Path, default=None)
    parser.add_argument(
        "--encoder-root",
        type=Path,
        default=Path("teacher/repos/AAAI25-ENCODER"),
    )
    parser.add_argument(
        "--correction-root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--tme-root",
        type=Path,
        default=Path("teacher/repos/TME"),
    )
    parser.add_argument(
        "--sprc-root",
        type=Path,
        default=Path("teacher/repos/SPRC"),
    )
    parser.add_argument(
        "--hint-root",
        type=Path,
        default=Path("teacher/repos/ICASSP26-HINT"),
    )
    parser.add_argument(
        "--qure-root",
        type=Path,
        default=Path("teacher/repos/QuRe"),
    )
    parser.add_argument(
        "--qure-config",
        type=Path,
        default=Path("teacher/repos/QuRe/configs/fashionIQ/eval.json"),
    )
    parser.add_argument(
        "--tgcir-root",
        type=Path,
        default=Path("teacher/repos/SPN4CIR/tgcir"),
    )
    parser.add_argument(
        "--csmcir-root",
        type=Path,
        default=Path("teacher/repos/CSMCIR"),
    )
    parser.add_argument(
        "--backbone",
        choices=("pretrain", "pretrain_vitL"),
        default="pretrain_vitL",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gallery-batch-size", type=int, default=16)
    parser.add_argument("--score-batch-size", type=int, default=32)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=("Debug only: maximum validation queries PER CATEGORY. Omit for the full FashionIQ validation set."),
    )
    parser.add_argument("--min-group-count", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()

def main():
    args = parse_args()
    args.cases = args.cases.resolve()
    args.image_root = args.image_root.resolve()
    args.split_root = args.split_root.resolve()
    args.output = args.output.resolve()
    if args.artifact_output is not None:
        args.artifact_output = args.artifact_output.resolve()
    if args.teacher == "encoder" and args.correction_root is None:
        args.correction_root = discover_correction_root(
            args.encoder_root.resolve(),
            (_repo_root() / "teacher/checkpoints/encoder").resolve(),
            (_repo_root() / "data/FashionIQ").resolve(),
        )
    if args.correction_root is not None:
        args.correction_root = args.correction_root.resolve()
    cases = load_cases(args.cases, args.limit)
    validate_cases(cases)
    if not torch.cuda.is_available():
        raise RuntimeError("Finalist teacher adapters are GPU-oriented; run on CUDA.")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.teacher == "encoder":
        (
            query_store,
            retrieval_common,
            retrieval_native,
            differentiability,
            integrity,
        ) = run_encoder(
            args,
            cases,
            device,
        )
        teacher_name = "ENCODER"
    elif args.teacher == "tme":
        (
            query_store,
            retrieval_common,
            retrieval_native,
            differentiability,
            integrity,
        ) = run_tme(
            args,
            cases,
            device,
        )
        teacher_name = "TME"
    elif args.teacher == "sprc":
        (
            query_store,
            retrieval_common,
            retrieval_native,
            differentiability,
            integrity,
        ) = run_sprc(
            args,
            cases,
            device,
        )
        teacher_name = "SPRC"
    elif args.teacher == "tgcir":
        (
            query_store,
            retrieval_common,
            retrieval_native,
            differentiability,
            integrity,
        ) = run_tgcir(
            args,
            cases,
            device,
        )
        teacher_name = "TG-CIR"
    elif args.teacher == "csmcir":
        (
            query_store,
            retrieval_common,
            retrieval_native,
            differentiability,
            integrity,
        ) = run_csmcir(
            args,
            cases,
            device,
        )
        teacher_name = "CSMCIR"
    else:
        raise ValueError(f"Unsupported teacher: {args.teacher}")
    report = build_report(
        teacher_name=teacher_name,
        cases=cases,
        query_store=query_store,
        retrieval_common=retrieval_common,
        retrieval_native=retrieval_native,
        differentiability=differentiability,
        integrity=integrity,
        min_group_count=args.min_group_count,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    if args.artifact_output is not None:
        args.artifact_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "sample_ids": [case["sample_id"] for case in cases],
                "reference_ids": [case["reference_id"] for case in cases],
                "target_ids": [case["target_id"] for case in cases],
                "categories": [case["category"] for case in cases],
                **query_store,
                "retrieval_common": retrieval_common,
                "retrieval_native": retrieval_native,
            },
            args.artifact_output,
        )
    common = report["retrieval_quality"]["common_full_gallery"]["include_reference"]["full"]
    balanced = report["same_edit_directional_consistency_balanced"]
    sensitivity = report["teacher_edit_retrieval_sensitivity"]["metrics"]["include_reference"]["combined_single_caption_removals"]
    print()
    print(f"=== {teacher_name} TEACHER SHORTLIST AUDIT V6 ===")
    print("queries:", report["num_queries"])
    print(
        "common macro mean(R@10,R@50):",
        common["macro"]["mean_r10_r50"],
    )
    print(
        "balanced gap:",
        balanced.get("macro_same_vs_different_gap"),
    )
    print(
        "balanced bootstrap interval:",
        balanced.get("bootstrap_95_interval_approx"),
    )
    conditional = sensitivity["log_rank_ratio_given_full_r50"]
    print(
        "teacher sensitivity mean log-rank ratio | full R@50:",
        None if conditional is None else conditional["mean"],
    )
    print(
        "gradient access probe:",
        report["differentiable_intervention_probe"]["status"],
    )
    print("Saved:", args.output)

if __name__ == "__main__":
    main()
