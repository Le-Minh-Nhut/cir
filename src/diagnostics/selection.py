from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from models.iag_srme.utils.retrieval import teacher_retrieval_loss


def transition_retrieval(
    current_query: Tensor,
    candidate_queries: Tensor,
    target_bank: Tensor,
    positive_mask: Tensor,
    negative_mask: Tensor,
    temperature: float,
) -> dict[str, Tensor]:
    """Teacher-aligned parent/sibling retrieval diagnostics using one shared pool."""

    parent_loss = teacher_retrieval_loss(
        current_query, target_bank.detach(), positive_mask, negative_mask, temperature
    )
    batch, candidates, dim = candidate_queries.shape
    flat_candidates = candidate_queries.reshape(batch * candidates, dim)
    flat_positive = (
        positive_mask[:, None].expand(-1, candidates, -1).reshape(batch * candidates, -1)
    )
    flat_negative = (
        negative_mask[:, None].expand(-1, candidates, -1).reshape(batch * candidates, -1)
    )
    candidate_loss = teacher_retrieval_loss(
        flat_candidates,
        target_bank.detach(),
        flat_positive,
        flat_negative,
        temperature,
    )
    candidate_loss = candidate_loss.reshape(batch, candidates)
    valid = positive_mask.any(dim=-1) & negative_mask.any(dim=-1)
    utility = parent_loss[:, None] - candidate_loss

    query = F.normalize(current_query.detach().float(), dim=-1)
    candidate = F.normalize(candidate_queries.detach().float(), dim=-1)
    targets = F.normalize(target_bank.detach().float(), dim=-1)
    parent_similarity = query @ targets.T
    candidate_similarity = torch.einsum("bkd,gd->bkg", candidate, targets)
    positive_similarity = parent_similarity.masked_fill(~positive_mask, -torch.inf).max(-1).values
    hardest_negative = parent_similarity.masked_fill(~negative_mask, -torch.inf).max(-1).values
    candidate_positive = (
        candidate_similarity.masked_fill(~positive_mask[:, None], -torch.inf).max(-1).values
    )
    candidate_hardest_negative = (
        candidate_similarity.masked_fill(~negative_mask[:, None], -torch.inf).max(-1).values
    )
    return {
        "valid": valid,
        "parent_loss": parent_loss,
        "candidate_loss": candidate_loss,
        "utility": utility,
        "parent_positive_similarity": positive_similarity,
        "parent_hardest_negative_similarity": hardest_negative,
        "parent_margin": positive_similarity - hardest_negative,
        "candidate_positive_similarity": candidate_positive,
        "candidate_hardest_negative_similarity": candidate_hardest_negative,
        "candidate_margin": candidate_positive - candidate_hardest_negative,
    }


def caption_utility_comparison(
    correct_utility: Tensor,
    shuffled_utility: Tensor,
    *,
    stop_threshold: float,
) -> dict[str, Tensor]:
    """Separate raw best-candidate gain from STOP-aware oracle-policy gain."""

    if correct_utility.ndim != 2 or shuffled_utility.shape != correct_utility.shape:
        raise ValueError("caption utilities must have matching [B,K] shapes")
    correct_best = correct_utility.max(dim=-1).values
    shuffled_best = shuffled_utility.max(dim=-1).values
    correct_oracle = torch.where(
        correct_best > stop_threshold,
        correct_best,
        torch.zeros_like(correct_best),
    )
    shuffled_oracle = torch.where(
        shuffled_best > stop_threshold,
        shuffled_best,
        torch.zeros_like(shuffled_best),
    )
    return {
        "best_candidate_utility_correct": correct_best,
        "best_candidate_utility_shuffled": shuffled_best,
        "best_candidate_utility_correct_minus_shuffled": correct_best - shuffled_best,
        "oracle_policy_utility_correct": correct_oracle,
        "oracle_policy_utility_shuffled": shuffled_oracle,
        "oracle_policy_utility_correct_minus_shuffled": correct_oracle - shuffled_oracle,
    }


def score_utility_calibration(
    predicted: Tensor, teacher: Tensor, *, num_bins: int = 10
) -> dict[str, Any]:
    """Simple detached calibration summary, with equal-count teacher-utility bins."""

    if predicted.shape != teacher.shape or predicted.numel() == 0:
        raise ValueError("predicted and teacher must have the same non-empty shape")
    score = predicted.detach().float().cpu().flatten()
    target = teacher.detach().float().cpu().flatten()
    error = score - target
    centered_score = score - score.mean()
    centered_target = target - target.mean()
    denominator = centered_score.norm() * centered_target.norm()
    pearson = (
        float(centered_score.dot(centered_target) / denominator)
        if float(denominator) > 1e-12
        else float("nan")
    )
    order = target.argsort()
    bins = []
    for indices in torch.tensor_split(order, min(num_bins, order.numel())):
        if indices.numel() == 0:
            continue
        bins.append(
            {
                "count": int(indices.numel()),
                "teacher_utility_mean": float(target.index_select(0, indices).mean()),
                "predicted_score_mean": float(score.index_select(0, indices).mean()),
                "bias": float(error.index_select(0, indices).mean()),
            }
        )
    return {
        "count": int(score.numel()),
        "pearson": pearson,
        "bias": float(error.mean()),
        "mae": float(error.abs().mean()),
        "rmse": float(error.square().mean().sqrt()),
        "sign_agreement_at_zero": float(score.gt(0).eq(target.gt(0)).float().mean()),
        "bins_by_teacher_utility": bins,
    }


def selection_metrics(
    utility: Tensor,
    selected_idx: Tensor,
    *,
    stop_threshold: float,
    delta_q_norm: Tensor | None = None,
) -> dict[str, Any]:
    """Candidate quality, selector regret and STOP confusion for valid rows."""

    if utility.ndim != 2:
        raise ValueError("utility must be [B,K]")
    batch, candidates = utility.shape
    if selected_idx.shape != (batch,):
        raise ValueError("selected_idx must be [B]")
    values = utility.detach().float().cpu()
    selected = selected_idx.detach().long().cpu()
    stop = selected >= candidates
    selected_candidate = selected.clamp_max(candidates - 1)
    selected_utility = values.gather(1, selected_candidate[:, None]).squeeze(1)
    selected_utility = torch.where(stop, torch.zeros_like(selected_utility), selected_utility)
    best_utility, best_slot = values.max(dim=-1)
    oracle_stop = best_utility <= stop_threshold
    oracle_action = torch.where(oracle_stop, torch.full_like(best_slot, candidates), best_slot)
    oracle_utility = torch.where(oracle_stop, torch.zeros_like(best_utility), best_utility)
    execute = ~stop
    oracle_execute = ~oracle_stop
    harmful = execute & (selected_utility < 0)

    tp = int((stop & oracle_stop).sum())
    fp = int((stop & ~oracle_stop).sum())
    tn = int((~stop & ~oracle_stop).sum())
    fn = int((~stop & oracle_stop).sum())

    def ratio(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator else 0.0

    slot_metrics = []
    norms = delta_q_norm.detach().float().cpu() if delta_q_norm is not None else None
    executed_count = int(execute.sum())
    oracle_executed_count = int(oracle_execute.sum())
    for slot in range(candidates):
        chosen = execute & selected_candidate.eq(slot)
        oracle = (~oracle_stop) & best_slot.eq(slot)
        slot_metrics.append(
            {
                "slot": slot,
                "selected_count": int(chosen.sum()),
                "selected_fraction_of_all_decisions": ratio(int(chosen.sum()), batch),
                "selected_fraction_given_execute": ratio(int(chosen.sum()), executed_count),
                "oracle_best_count": int(oracle.sum()),
                "oracle_best_fraction_of_all_decisions": ratio(int(oracle.sum()), batch),
                "oracle_best_fraction_given_oracle_execute": ratio(
                    int(oracle.sum()), oracle_executed_count
                ),
                "mean_teacher_utility": float(values[:, slot].mean()),
                "median_teacher_utility": float(values[:, slot].median()),
                "positive_utility_fraction": float((values[:, slot] > 0).float().mean()),
                "harmful_fraction": float((values[:, slot] < 0).float().mean()),
                "mean_delta_q_norm": (
                    float(norms[:, slot].mean()) if norms is not None else float("nan")
                ),
            }
        )

    agreement = selected.eq(oracle_action)
    return {
        "decision_count": batch,
        "execute_count": executed_count,
        "execute_rate": ratio(executed_count, batch),
        "oracle_execute_count": oracle_executed_count,
        "oracle_execute_rate": ratio(oracle_executed_count, batch),
        "slot_metrics": slot_metrics,
        "selected_equals_oracle_count": int(agreement.sum()),
        "selected_equals_oracle_fraction": float(agreement.float().mean()),
        "mean_selected_utility": float(selected_utility.mean()),
        "mean_oracle_utility": float(oracle_utility.mean()),
        "mean_regret": float((oracle_utility - selected_utility).mean()),
        "positive_utility_candidate_count_mean": float((values > 0).sum(-1).float().mean()),
        "harmful_candidate_count_mean": float((values < 0).sum(-1).float().mean()),
        "stop": {
            "reference_threshold": stop_threshold,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": ratio(tp, tp + fp),
            "recall": ratio(tp, tp + fn),
            "f1": ratio(2 * tp, 2 * tp + fp + fn),
            "stop_count": int(stop.sum()),
            "stop_rate": float(stop.float().mean()),
            "oracle_stop_count": int(oracle_stop.sum()),
            "oracle_stop_rate": float(oracle_stop.float().mean()),
            "harmful_execution_count": int(harmful.sum()),
            "executed_action_count": executed_count,
            "harmful_execution_fraction_of_executions": ratio(int(harmful.sum()), executed_count),
            "harmful_execution_fraction_of_decisions": ratio(int(harmful.sum()), batch),
            "premature_stop_count": fp,
        },
    }


def slot_monopoly(
    metrics: dict[str, Any], *, oracle: bool = False, threshold: float = 0.60
) -> dict[str, int | float | bool]:
    """Detect slot concentration using execution-conditional denominators."""

    key = (
        "oracle_best_fraction_given_oracle_execute" if oracle else "selected_fraction_given_execute"
    )
    values = [float(slot[key]) for slot in metrics["slot_metrics"]]
    slot = max(range(len(values)), key=values.__getitem__)
    fraction = values[slot]
    return {"detected": fraction > threshold, "slot": slot, "fraction": fraction}
