from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

_EPS = 1e-8
_DIRECTION_NORM_THRESHOLD = 1e-6
_GENERIC_STAGES = ("proposals", "entities", "actions", "delta", "delta_q")


def _flatten_candidates(values: Tensor) -> Tensor:
    if values.ndim < 3:
        raise ValueError("candidate values must have shape [B,K,...]")
    return values.detach().flatten(start_dim=2)


def _pairwise_mean(values: Tensor, fn) -> Tensor:
    candidates = values.shape[1]
    if candidates < 2:
        return values.new_zeros(())
    return torch.stack(
        [fn(values[:, left], values[:, right]) for left in range(candidates) for right in range(left + 1, candidates)]
    ).mean()


def _pairwise_cosine(values: Tensor) -> tuple[Tensor, Tensor]:
    candidates = values.shape[1]
    if candidates < 2:
        return values.new_zeros(()), values.new_zeros(())
    flat = values.float()
    norms = flat.norm(dim=-1)
    valid = norms > _DIRECTION_NORM_THRESHOLD
    cosine_sum = flat.new_zeros(())
    valid_pairs = flat.new_zeros(())
    total_pairs = flat.new_zeros(())
    for left in range(candidates):
        for right in range(left + 1, candidates):
            pair_valid = valid[:, left] & valid[:, right]
            cosine_sum = cosine_sum + (
                (flat[:, left] * flat[:, right]).sum(dim=-1)
                / (norms[:, left] * norms[:, right]).clamp_min(_EPS)
                * pair_valid
            ).sum()
            valid_pairs = valid_pairs + pair_valid.sum()
            total_pairs = total_pairs + pair_valid.numel()
    return cosine_sum / valid_pairs.clamp_min(1), valid_pairs / total_pairs


def _rank_statistics(values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    flat = values.double()
    energy = flat.norm(dim=(-2, -1))
    valid = energy > _DIRECTION_NORM_THRESHOLD
    normalized = torch.where(
        valid[:, None, None], flat / energy[:, None, None].clamp_min(_EPS), torch.zeros_like(flat)
    )
    gram = normalized @ normalized.transpose(-1, -2)
    singular = torch.linalg.eigvalsh(gram).clamp_min(0).sqrt()
    singular_sum = singular.sum(dim=-1)
    ranks = torch.where(
        valid,
        singular_sum.square() / singular.square().sum(dim=-1).clamp_min(_EPS),
        torch.zeros_like(singular_sum),
    )
    spectrum = torch.where(
        valid[:, None], singular / singular_sum[:, None].clamp_min(_EPS), torch.zeros_like(singular)
    )
    mean_spectrum = spectrum[valid].mean(dim=0) if valid.any() else spectrum.new_zeros(spectrum.shape[-1])
    return ranks, mean_spectrum, valid


def _effective_rank(values: Tensor) -> Tensor:
    ranks, _, valid = _rank_statistics(values)
    result = ranks[valid].mean() if valid.any() else ranks.new_zeros(())
    return result.to(values.dtype)

def generic_diversity(values: Tensor) -> dict[str, float | list[float]]:
    """Detached sibling metrics with explicit low-energy directional coverage."""
    flat = _flatten_candidates(values)
    candidates = flat.shape[1]
    values32 = flat.float()
    norms = values32.norm(dim=-1)
    centered = values32 - values32.mean(dim=1, keepdim=True)
    centered_norms = centered.norm(dim=-1)
    spread = centered_norms.mean(dim=-1)
    mean_norm = norms.mean(dim=-1)
    effect_valid = mean_norm > _DIRECTION_NORM_THRESHOLD
    cosine, pair_coverage = _pairwise_cosine(flat)
    ranks, spectrum, rank_valid = _rank_statistics(values32)
    centered_ranks, centered_spectrum, centered_valid = _rank_statistics(centered)
    relative_spread = torch.where(
        effect_valid, spread / mean_norm.clamp_min(_EPS), torch.zeros_like(spread)
    )
    quantiles = torch.quantile(
        norms.flatten(), torch.tensor([0.0, 0.5, 0.9, 0.99], device=norms.device)
    )
    return {
        "pairwise_cosine": float(cosine),
        "pairwise_cosine_valid_pair_fraction": float(pair_coverage),
        "effective_rank": float(ranks[rank_valid].mean()) if rank_valid.any() else 0.0,
        "centered_effective_rank": float(centered_ranks[centered_valid].mean()) if centered_valid.any() else 0.0,
        "singular_value_spectrum": [float(value) for value in spectrum],
        "centered_singular_value_spectrum": [float(value) for value in centered_spectrum],
        "spread": float(spread.mean()),
        "relative_spread": float(relative_spread[effect_valid].mean()) if effect_valid.any() else 0.0,
        "mean_norm": float(norms.mean()),
        "norm_quantile_p00": float(quantiles[0]),
        "norm_quantile_p50": float(quantiles[1]),
        "norm_quantile_p90": float(quantiles[2]),
        "norm_quantile_p99": float(quantiles[3]),
        "valid_candidate_fraction": float((norms > _DIRECTION_NORM_THRESHOLD).float().mean()),
        "input_dtype_bits": float(torch.finfo(values.dtype).bits) if values.is_floating_point() else 0.0,
        "valid_sibling_effect_fraction": float(effect_valid.float().mean()),
        "low_energy_fraction": float((~effect_valid).float().mean()),
        "direction_norm_threshold": _DIRECTION_NORM_THRESHOLD,
        **{f"mean_norm_c{slot}": float(norms[:, slot].mean()) for slot in range(candidates)},
    }


def alpha_read_diversity(values: Tensor) -> dict[str, float]:
    """Detached distribution metrics for normalized grounding [B,K,N]."""
    alpha = values.detach().float().clamp_min(0)
    alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(_EPS)

    def js(left: Tensor, right: Tensor) -> Tensor:
        midpoint = 0.5 * (left + right)
        return 0.5 * (
            (left * (left.clamp_min(_EPS).log() - midpoint.clamp_min(_EPS).log())).sum(dim=-1)
            + (right * (right.clamp_min(_EPS).log() - midpoint.clamp_min(_EPS).log())).sum(dim=-1)
        ).mean()

    return {
        "pairwise_cosine": float(_pairwise_cosine(alpha)[0]),
        "pairwise_js_divergence": float(_pairwise_mean(alpha, js)),
        "entropy": float(-(alpha * alpha.clamp_min(_EPS).log()).sum(dim=-1).mean()),
        "argmax_agreement": float(
            _pairwise_mean(alpha, lambda left, right: left.argmax(dim=-1).eq(right.argmax(dim=-1)).float().mean())
        ),
    }


def exec_mask_diversity(values: Tensor) -> dict[str, float]:
    """Detached soft-mask metrics for executor support [B,K,N]."""
    masks = values.detach().float()
    flat = _flatten_candidates(masks)
    centered = flat - flat.mean(dim=1, keepdim=True)
    norms = flat.norm(dim=-1)
    spread = centered.norm(dim=-1).mean(dim=-1)
    return {
        "pairwise_cosine": float(_pairwise_cosine(flat)[0]),
        "soft_iou": float(
            _pairwise_mean(
                flat,
                lambda left, right: torch.minimum(left, right).sum(dim=-1).div(
                    torch.maximum(left, right).sum(dim=-1) + _EPS
                ).mean(),
            )
        ),
        "spread": float(spread.mean()),
        "relative_spread": float((spread / (norms.mean(dim=-1) + _EPS)).mean()),
        "mean_activation": float(masks.mean()),
    }


def _step_audit(step: Mapping[str, object]) -> dict[str, object]:
    tensors = {name: step[name] for name in (*_GENERIC_STAGES, "alpha_read", "exec_mask")}
    if not all(isinstance(value, Tensor) for value in tensors.values()):
        raise TypeError("functional collapse audit requires tensor trajectory stages")
    generic = {name: generic_diversity(tensors[name]) for name in _GENERIC_STAGES}
    return {
        "live_samples": int(tensors["delta_q"].shape[0]),
        **generic,
        "alpha_read": alpha_read_diversity(tensors["alpha_read"]),
        "exec_mask": exec_mask_diversity(tensors["exec_mask"]),
    }


def _aggregate(steps: list[dict[str, object]]) -> dict[str, object]:
    if not steps:
        return {}
    total = sum(int(step["live_samples"]) for step in steps)

    def combine(values: list[dict[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key in values[0]:
            if isinstance(values[0][key], list):
                result[key] = [
                    sum(float(value[key][slot]) * int(step["live_samples"]) for value, step in zip(values, steps, strict=True)) / total
                    for slot in range(len(values[0][key]))
                ]
            else:
                result[key] = sum(float(value[key]) * int(step["live_samples"]) for value, step in zip(values, steps, strict=True)) / total
        return result

    return {
        "live_samples": total,
        **{name: combine([step[name] for step in steps]) for name in _GENERIC_STAGES},
        "alpha_read": combine([step["alpha_read"] for step in steps]),
        "exec_mask": combine([step["exec_mask"] for step in steps]),
    }


@torch.no_grad()
def functional_collapse_audit(output: Mapping[str, object]) -> dict[str, object]:
    """Small detached, sample-weighted trajectory audit with per-step summaries."""

    trajectory = output.get("steps")
    if not isinstance(trajectory, list):
        raise TypeError("model output must contain a trajectory list")
    by_step = {f"t{step['timestep']}": _step_audit(step) for step in trajectory}
    return {"overall": _aggregate(list(by_step.values())), "by_step": by_step}

_PROPOSAL_STAGES = (
    "base_query",
    "expanded_query",
    "conditioned_residual",
    "query_pre_norm",
    "query_post_norm",
    "raw_attention_output",
    "proposal_output",
)


def _proposal_step_audit(step: Mapping[str, object]) -> dict[str, object]:
    internal = step.get("proposal_internal")
    if not isinstance(internal, Mapping):
        raise TypeError("proposal internal audit requires proposal diagnostic stages")
    tensors = {name: internal[name] for name in _PROPOSAL_STAGES}
    if not all(isinstance(value, Tensor) for value in tensors.values()):
        raise TypeError("proposal internal audit requires tensor stages")
    base = tensors["expanded_query"]
    conditioned = tensors["conditioned_residual"]
    pre = tensors["query_pre_norm"]
    base_norm = base.float().norm(dim=-1)
    displacement = (pre.float() - base.float()).norm(dim=-1) / (base_norm + _EPS)
    base_pre_cosine = (base.float() * pre.float()).sum(dim=-1) / (
        base_norm * pre.float().norm(dim=-1) + _EPS
    )
    post = generic_diversity(tensors["query_post_norm"])
    output = generic_diversity(tensors["proposal_output"])
    return {
        "live_samples": int(base.shape[0]),
        **{name: generic_diversity(tensors[name]) for name in _PROPOSAL_STAGES},
        "conditioner_to_base_norm_ratio": float(
            (conditioned.float().norm(dim=-1) / (base_norm + _EPS)).mean()
        ),
        "base_pre_cosine": float(base_pre_cosine.mean()),
        "mean_relative_displacement": float(displacement.mean()),
        **{
            f"mean_relative_displacement_c{slot}": float(displacement[:, slot].mean())
            for slot in range(base.shape[1])
        },
        "attention_diversity_retention": float(
            output["relative_spread"] / (post["relative_spread"] + _EPS)
        ),
        "attention_rank_retention": float(
            output["effective_rank"] / (post["effective_rank"] + _EPS)
        ),
    }


def _selection_step_audit(
    step: Mapping[str, object],
    targets: Tensor,
    target_ids: list[str],
    temperature: float,
    epsilon_stop: float = 0.0,
) -> dict[str, float]:
    from models.iag_srme.utils.retrieval import build_teacher_masks, marginal_teacher_utilities

    scores, selected, live, current, candidate = (
        step["scores"], step["selected_idx"], step["live_indices"], step["current_query"], step["candidate_queries"]
    )
    if not all(isinstance(value, Tensor) for value in (scores, selected, live, current, candidate)):
        raise TypeError("selection audit requires tensor trajectory stages")
    positive, negative, _ = build_teacher_masks(target_ids, targets.device)
    teacher, valid = marginal_teacher_utilities(
        current, candidate, targets.detach(), positive.index_select(0, live), negative.index_select(0, live), temperature
    )
    candidates = scores.shape[1]
    raw_best_score, raw_best_idx = scores.max(dim=-1)
    top_two = scores.topk(min(2, candidates), dim=-1).values
    score_margin = (
        top_two[:, 0] - top_two[:, 1]
        if candidates > 1
        else torch.full_like(raw_best_score, torch.inf)
    )
    stop = selected.eq(candidates)
    selected_slot = selected.clamp_max(candidates - 1)
    selected_utility = teacher.gather(1, selected_slot[:, None]).squeeze(1)
    selected_utility = torch.where(stop, torch.zeros_like(selected_utility), selected_utility)
    utility_with_keep = torch.cat([teacher, teacher.new_zeros(teacher.shape[0], 1)], dim=-1)
    oracle_utility, oracle_idx = utility_with_keep.max(dim=-1)
    oracle_top_two = utility_with_keep.topk(min(2, candidates + 1), dim=-1).values
    utility_margin = oracle_top_two[:, 0] - oracle_top_two[:, 1]
    valid_values = valid.float()
    valid_count = valid_values.sum().clamp_min(1)
    return {
        "live_samples": float(scores.shape[0]),
        "teacher_valid_fraction": float(valid_values.mean()),
        "score_mean": float(scores.float().mean()),
        "score_std": float(scores.float().std(unbiased=False)),
        "raw_best_score": float(raw_best_score.float().mean()),
        "score_margin_mean": float(score_margin.float().mean()),
        "score_margin_near_tie_fraction": float((score_margin.abs() <= 1e-6).float().mean()),
        "score_margin_to_stop_mean": float((raw_best_score - epsilon_stop).float().mean()),
        "executed_fraction": float((~stop).float().mean()),
        "stop_fraction": float(stop.float().mean()),
        "oracle_keep_fraction": float(oracle_idx.eq(candidates).float().mean()),
        "useful_candidate_count": float(((teacher > 0) & valid[:, None]).sum() / valid_count),
        "selected_utility": float((selected_utility * valid_values).sum() / valid_count),
        "oracle_utility": float((oracle_utility * valid_values).sum() / valid_count),
        "one_step_regret": float(((oracle_utility - selected_utility) * valid_values).sum() / valid_count),
        "teacher_oracle_utility_margin": float((utility_margin * valid_values).sum() / valid_count),
        "teacher_utility_std": float(teacher[valid].float().std(unbiased=False)) if valid.any() else 0.0,
        **{f"score_c{slot}": float(scores[:, slot].float().mean()) for slot in range(candidates)},
        **{
            f"teacher_utility_c{slot}": float((teacher[:, slot] * valid_values).sum() / valid_count)
            for slot in range(candidates)
        },
        **{f"selected_c{slot}_fraction": float((~stop & selected_slot.eq(slot)).float().mean()) for slot in range(candidates)},
        **{f"raw_best_c{slot}_fraction": float(raw_best_idx.eq(slot).float().mean()) for slot in range(candidates)},
        **{f"oracle_c{slot}_fraction": float(oracle_idx.eq(slot).float().mean()) for slot in range(candidates)},
    }


def _scalar_aggregate(steps: list[dict[str, float]]) -> dict[str, float]:
    if not steps:
        return {}
    total = sum(step["live_samples"] for step in steps)
    return {key: sum(step[key] * step["live_samples"] for step in steps) / total for key in steps[0]}


@torch.no_grad()
def selection_quality_audit(
    output: Mapping[str, object],
    targets: Tensor,
    target_ids: list[str],
    temperature: float,
    epsilon_stop: float = 0.0,
) -> dict[str, object]:
    """Target-derived training diagnostics; never used by model, loss, or routing."""
    trajectory = output.get("steps")
    if not isinstance(trajectory, list):
        raise TypeError("model output must contain a trajectory list")
    by_step = {
        f"t{step['timestep']}": _selection_step_audit(
            step, targets, target_ids, temperature, epsilon_stop
        )
        for step in trajectory
    }
    return {"overall": _scalar_aggregate(list(by_step.values())), "by_step": by_step}


def _proposal_aggregate(steps: list[dict[str, object]]) -> dict[str, object]:
    if not steps:
        return {}
    total = sum(int(step["live_samples"]) for step in steps)

    def mean(name: str) -> object:
        value = steps[0][name]
        if isinstance(value, Mapping):
            return {
                key: mean_nested([step[name][key] for step in steps], total, steps)
                for key in value
            }
        return mean_nested([step[name] for step in steps], total, steps)

    return {"live_samples": total, **{name: mean(name) for name in steps[0] if name != "live_samples"}}


def mean_nested(values: list[object], total: int, steps: list[dict[str, object]]) -> object:
    if isinstance(values[0], list):
        return [
            sum(float(value[slot]) * int(step["live_samples"]) for value, step in zip(values, steps, strict=True)) / total
            for slot in range(len(values[0]))
        ]
    return sum(float(value) * int(step["live_samples"]) for value, step in zip(values, steps, strict=True)) / total


@torch.no_grad()
def proposal_internal_audit(output: Mapping[str, object]) -> dict[str, object]:
    """Detached, sample-weighted ProposalNet stage audit."""
    trajectory = output.get("steps")
    if not isinstance(trajectory, list):
        raise TypeError("model output must contain a trajectory list")
    by_step = {f"t{step['timestep']}": _proposal_step_audit(step) for step in trajectory}
    return {"overall": _proposal_aggregate(list(by_step.values())), "by_step": by_step}
