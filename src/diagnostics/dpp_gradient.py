from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn

from losses.objective import functional_dpp_loss, update_executed_history
from models.iag_srme.utils.retrieval import build_teacher_masks, marginal_teacher_utilities

_QUANTILES = (0.0, 0.1, 0.5, 0.9, 0.99, 1.0)
_QUANTILE_NAMES = ("p00", "p10", "p50", "p90", "p99", "p100")


def _quantiles(values: list[float], device: torch.device) -> dict[str, float]:
    if not values:
        return {name: 0.0 for name in _QUANTILE_NAMES}
    result = torch.quantile(torch.tensor(values, device=device), torch.tensor(_QUANTILES, device=device))
    return {name: float(value) for name, value in zip(_QUANTILE_NAMES, result, strict=True)}


def _empty() -> dict[str, object]:
    return {
        "dpp_raw_averages_valid_rows": True,
        "dpp_valid_row_count": 0,
        "dpp_total_eligible_row_count": 0,
        "dpp_valid_rate": 0.0,
        "dpp_useful_count": 0.0,
        "effect_norm_quantiles": _quantiles([], torch.device("cpu")),
        "live_grad_norm_quantiles": _quantiles([], torch.device("cpu")),
        "fp32_reference_grad_norm_quantiles": _quantiles([], torch.device("cpu")),
        "finite_in_fp32_but_nonfinite_live_count": 0,
        "by_step": {},
        "producer_gradients": {},
    }


def _trainable(module: nn.Module | None) -> list[nn.Parameter]:
    return [parameter for parameter in module.parameters() if parameter.requires_grad] if module else []


def _producer_groups(model: nn.Module) -> dict[str, list[nn.Parameter]]:
    backbone = getattr(model, "backbone", None)
    checkpoint = getattr(backbone, "model", None)
    readout_modules = (
        getattr(checkpoint, "vision_model", None),
        getattr(checkpoint, "visual_projection", None),
    )
    readout_parameters = [
        parameter
        for module in readout_modules
        for parameter in _trainable(module)
    ]
    if isinstance(getattr(backbone, "q_G", None), nn.Parameter) and backbone.q_G.requires_grad:
        readout_parameters.append(backbone.q_G)
    return {
        "text_encoder": _trainable(getattr(checkpoint, "text_model", None)),
        "text_adapter": _trainable(getattr(backbone, "text_adapter", None)),
        "backbone_readout": readout_parameters,
        "proposal": _trainable(getattr(model, "proposal", None)),
        "grounder": _trainable(getattr(model, "grounder", None)),
        "action_fusion": _trainable(getattr(model, "action_fusion", None)),
        "executor": _trainable(getattr(model, "executor", None)),
        "score_net": _trainable(getattr(model, "score_net", None)),
    }


def _producer_report(
    groups: Mapping[str, list[nn.Parameter]], gradients: Mapping[int, Tensor | None]
) -> dict[str, dict[str, float | int]]:
    report = {}
    for name, parameters in groups.items():
        sum_squares = 0.0
        nonfinite = elements = receiving = 0
        for parameter in parameters:
            gradient = gradients.get(id(parameter))
            if gradient is None:
                continue
            receiving += 1
            value = gradient.detach().float()
            sum_squares += float(value.square().sum())
            nonfinite += int((~torch.isfinite(value)).sum())
            elements += value.numel()
        report[name] = {
            "global_l2_norm": sum_squares**0.5,
            "parameters_with_gradient": receiving,
            "parameter_count": len(parameters),
            "nonfinite_element_count": nonfinite,
            "nonfinite_element_fraction": nonfinite / elements if elements else 0.0,
        }
    return report


def _fp32_reference(
    effect: Tensor,
    history: Tensor | None,
    teacher: Tensor,
    config: object,
    valid_rows: Tensor,
) -> tuple[Tensor | None, int]:
    """Production DPP loss on a detached FP32 clone; this graph never reaches training."""
    reference = effect.detach().float().clone().requires_grad_()
    result = functional_dpp_loss(
        reference,
        history,
        teacher,
        kappa=config.kappa_dpp,
        sigma=config.sigma_dpp,
        tau=config.tau_dpp,
        useful_threshold=config.useful_threshold,
        jitter=config.dpp_jitter,
    )
    if not bool(result["valid"]):
        return None, int(result["useful_count"])
    gradient = torch.autograd.grad(result["loss"] / valid_rows, reference)[0]
    return gradient, int(result["useful_count"])


def dpp_gradient_audit(
    model: nn.Module,
    objective: nn.Module,
    output: Mapping[str, object],
    targets: Tensor,
    target_ids: Sequence[str],
    dpp_raw: Tensor,
    dpp_valid_row_count: Tensor,
) -> dict[str, object]:
    """Measure DPP-only gradients before total backward without touching ``.grad``."""
    steps = output.get("steps")
    config = getattr(objective, "config", None)
    if not isinstance(steps, list) or config is None or not bool(config.dpp_enabled):
        return _empty()
    effect_steps = [step for step in steps if isinstance(step.get("delta_q"), Tensor)]
    if not effect_steps or not dpp_raw.requires_grad or not int(dpp_valid_row_count):
        return _empty()

    effects = [step["delta_q"] for step in effect_steps]
    candidates = [step["candidate_queries"] for step in effect_steps]
    currents = [step["current_query"] for step in effect_steps]
    assert all(isinstance(value, Tensor) for value in [*effects, *candidates, *currents])
    groups = _producer_groups(model)
    parameters = [parameter for values in groups.values() for parameter in values]
    query_targets = [
        (name, index, value)
        for name, values in (("candidate", candidates), ("current", currents))
        for index, value in enumerate(values)
        if value.requires_grad
    ]
    targets_to_grad = [*effects, *parameters, *(value for _, _, value in query_targets)]
    gradients = torch.autograd.grad(
        dpp_raw, targets_to_grad, retain_graph=True, allow_unused=True
    )
    effect_gradients = gradients[: len(effects)]
    parameter_gradients = {
        id(parameter): gradient
        for parameter, gradient in zip(parameters, gradients[len(effects) : len(effects) + len(parameters)], strict=True)
    }
    query_gradients = {
        (name, index): gradient
        for (name, index, _), gradient in zip(
            query_targets, gradients[len(effects) + len(parameters) :], strict=True
        )
    }

    positive, negative, _ = build_teacher_masks(target_ids, targets.device)
    histories: list[list[Tensor]] = [[] for _ in target_ids]
    valid_rows = dpp_valid_row_count.detach().clamp_min(1)
    eligible_rows = valid_row_count = 0
    useful_counts: list[float] = []
    all_effect_norms: list[float] = []
    all_live_norms: list[float] = []
    all_reference_norms: list[float] = []
    finite_in_fp32_but_nonfinite_live = 0
    by_step: dict[str, object] = {}
    for index, (step, effect_gradient) in enumerate(zip(effect_steps, effect_gradients, strict=True)):
        candidate_gradient = query_gradients.get(("candidate", index))
        current_gradient = query_gradients.get(("current", index))
        current = step["current_query"]
        live = step["live_indices"]
        candidate = step["candidate_queries"]
        selected = step["selected_idx"]
        effect = step["delta_q"]
        assert all(isinstance(value, Tensor) for value in (live, current, candidate, selected, effect))
        teacher, teacher_valid = marginal_teacher_utilities(
            current,
            candidate,
            targets.detach(),
            positive.index_select(0, live),
            negative.index_select(0, live),
            config.retrieval_temperature,
        )
        rows: list[dict[str, object]] = []
        for local_row, sample_index in enumerate(live.tolist()):
            if not bool(teacher_valid[local_row]):
                continue
            eligible_rows += 1
            history_values = histories[sample_index]
            history = torch.stack(history_values) if history_values else None
            reference_gradient, useful_count = _fp32_reference(
                effect[local_row], history, teacher[local_row], config, valid_rows
            )
            useful_counts.append(float(useful_count))
            if reference_gradient is None:
                continue
            valid_row_count += 1
            live_gradient = (
                effect_gradient[local_row].detach().float()
                if effect_gradient is not None
                else torch.zeros_like(reference_gradient)
            )
            effect_norm = effect[local_row].detach().float().norm(dim=-1)
            live_norm = live_gradient.norm(dim=-1)
            reference_norm = reference_gradient.detach().float().norm(dim=-1)
            all_effect_norms.extend(float(value) for value in effect_norm)
            all_live_norms.extend(float(value) for value in live_norm)
            all_reference_norms.extend(float(value) for value in reference_norm)
            finite_in_fp32_but_nonfinite_live += int(
                (torch.isfinite(reference_gradient) & ~torch.isfinite(live_gradient)).sum()
            )
            rows.append(
                {
                    "min_effect_norm": float(effect_norm.min()),
                    "mean_effect_norm": float(effect_norm.mean()),
                    "max_effect_norm": float(effect_norm.max()),
                    "effect_norms": [float(value) for value in effect_norm],
                    "live_effect_grad_norms": [float(value) for value in live_norm],
                    "fp32_reference_grad_norms": [float(value) for value in reference_norm],
                    "live_to_fp32_grad_norms": [
                        float(value) for value in live_norm / reference_norm.clamp_min(1e-30)
                    ],
                    "source_effect_dtype": str(effect.dtype).removeprefix("torch."),
                    "live_grad_nonfinite_fraction": float((~torch.isfinite(live_gradient)).float().mean()),
                    "fp32_grad_nonfinite_fraction": float(
                        (~torch.isfinite(reference_gradient)).float().mean()
                    ),
                    "finite_in_fp32_but_nonfinite_live_count": int(
                        (torch.isfinite(reference_gradient) & ~torch.isfinite(live_gradient)).sum()
                    ),
                }
            )
        update_executed_history(histories, live, selected, effect, effect.shape[1])
        if rows:
            candidate_live = candidate_gradient.detach().float() if candidate_gradient is not None else None
            current_live = current_gradient.detach().float() if current_gradient is not None else None
            by_step[f"t{step['timestep']}"] = {
                "valid_row_count": len(rows),
                "candidate_query_grad_norm": float(candidate_live.norm()) if candidate_live is not None else 0.0,
                "candidate_query_grad_norm_per_slot": [
                    float(candidate_live[:, slot].norm()) if candidate_live is not None else 0.0
                    for slot in range(effect.shape[1])
                ],
                "candidate_query_grad_max": float(candidate_live.abs().max()) if candidate_live is not None else 0.0,
                "candidate_query_grad_nonfinite_fraction": float(
                    (~torch.isfinite(candidate_live)).float().mean()
                ) if candidate_live is not None else 0.0,
                "current_query_grad_norm": float(current_live.norm()) if current_live is not None else 0.0,
                "current_query_grad_nonfinite_fraction": float(
                    (~torch.isfinite(current_live)).float().mean()
                ) if current_live is not None else 0.0,
                "rows": rows,
            }
    return {
        "dpp_raw_averages_valid_rows": True,
        "dpp_valid_row_count": valid_row_count,
        "dpp_total_eligible_row_count": eligible_rows,
        "dpp_valid_rate": valid_row_count / eligible_rows if eligible_rows else 0.0,
        "dpp_useful_count": sum(useful_counts) / len(useful_counts) if useful_counts else 0.0,
        "effect_norm_quantiles": _quantiles(all_effect_norms, targets.device),
        "live_grad_norm_quantiles": _quantiles(all_live_norms, targets.device),
        "fp32_reference_grad_norm_quantiles": _quantiles(all_reference_norms, targets.device),
        "finite_in_fp32_but_nonfinite_live_count": finite_in_fp32_but_nonfinite_live,
        "by_step": by_step,
        "producer_gradients": _producer_report(groups, parameter_gradients),
    }
