"""
Exhaustive diagnostic for CIR IAG-SRME V2 R0.

Put this file at:
    src/diagnose_iag_srme.py

Typical use:
    TOKENIZERS_PARALLELISM=false python src/diagnose_iag_srme.py \
      backbone=fgclip_base_text_native_cls \
      +checkpoint=outputs/r0_ncls_text_strong_aux/best.pt \
      +diagnostic_batches=6 \
      +diagnostic_batch_size=8 \
      hydra.run.dir=outputs/diagnose_r0_ncls_text

The script never optimizer.step()s and never modifies the checkpoint.
It inspects checkpoint/config consistency, loss decomposition, retrieval improvement,
proposal/action/effect collapse, grounding, masks, STOP, ScoreNet/teacher agreement,
concept/relation auxiliaries, DPP activation, gradient routing, total gradients,
parameter movement, NaN/Inf and peak CUDA memory.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.nn.functional as F
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.utils.data import DataLoader

from data.images import FashionIQImageCollator
from datasets.common import DirectoryImageStore
from datasets.fashioniq import FashionIQDataset
from diagnostics.cohort import (
    load_or_create_manifest,
    sample_ids_fingerprint,
    validate_processed_manifest,
)
from diagnostics.selection import selection_metrics
from evaluate import validate_checkpoint_backbone_metadata
from losses.objective import IAGSRMEObjective, ObjectiveConfig
from models.iag_srme.utils.retrieval import build_teacher_masks, marginal_teacher_utilities
from runtime import configure_torch_runtime, resolve_device, seed_everything
from train import CATEGORIES, build_concept_vocabulary, build_model, encode_concept_prototypes
from training.engine import resolve_precision


def scalar(x: Any) -> float:
    if isinstance(x, Tensor):
        if x.numel() != 1:
            raise ValueError(f"expected scalar tensor, got {tuple(x.shape)}")
        return float(x.detach().float().cpu())
    return float(x)


def finite_tensor(x: Tensor) -> bool:
    return bool(torch.isfinite(x.detach()).all())


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def pearson(x: Tensor, y: Tensor) -> float:
    x = x.detach().float().flatten()
    y = y.detach().float().flatten()
    if x.numel() < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if float(denom) <= 1e-12:
        return float("nan")
    return float((x @ y / denom).cpu())


def normalized_entropy(probability: Tensor) -> Tensor:
    p = probability.detach().float().clamp_min(1e-12)
    h = -(p * p.log()).sum(dim=-1)
    n = probability.shape[-1]
    return h / max(math.log(n), 1e-12)


def pairwise_cosine_mean(x: Tensor) -> float:
    if x.ndim != 3 or x.shape[1] < 2:
        return float("nan")
    z = F.normalize(x.detach().float(), dim=-1)
    cosine = z @ z.transpose(-1, -2)
    k = x.shape[1]
    off = ~torch.eye(k, dtype=torch.bool, device=x.device)
    return float(cosine[:, off].mean().cpu())


def pairwise_cosine_max(x: Tensor) -> float:
    if x.ndim != 3 or x.shape[1] < 2:
        return float("nan")
    z = F.normalize(x.detach().float(), dim=-1)
    cosine = z @ z.transpose(-1, -2)
    k = x.shape[1]
    off = ~torch.eye(k, dtype=torch.bool, device=x.device)
    return float(cosine[:, off].max().cpu())


def pairwise_l2_mean(x: Tensor) -> float:
    if x.ndim < 3 or x.shape[1] < 2:
        return float("nan")
    flat = x.detach().float().flatten(2)
    values = []
    for i in range(flat.shape[1]):
        for j in range(i + 1, flat.shape[1]):
            values.append((flat[:, i] - flat[:, j]).norm(dim=-1))
    return float(torch.stack(values).mean().cpu())


def effective_rank(x: Tensor) -> float:
    s = torch.linalg.svdvals(x.detach().float())
    rank = s.sum(dim=-1).square() / s.square().sum(dim=-1).clamp_min(1e-12)
    return float(rank.mean().cpu())


def binary_mask_jaccard(mask: Tensor, threshold: float = 0.5) -> float:
    if mask.ndim != 3 or mask.shape[1] < 2:
        return float("nan")
    hard = mask.detach() > threshold
    values = []
    for i in range(hard.shape[1]):
        for j in range(i + 1, hard.shape[1]):
            inter = (hard[:, i] & hard[:, j]).sum(dim=-1).float()
            union = (hard[:, i] | hard[:, j]).sum(dim=-1).float()
            values.append(inter / union.clamp_min(1))
    return float(torch.stack(values).mean().cpu())


def positive_score_and_rank(query: Tensor, targets: Tensor, positive: Tensor, negative: Tensor):
    # In-batch diagnostic only, NOT official FashionIQ Recall.
    scores = query.detach().float() @ targets.detach().float().T
    positive_score = scores.masked_fill(~positive, float("-inf")).max(dim=-1).values
    negative_higher = (scores > positive_score[:, None]) & negative
    rank = 1 + negative_higher.sum(dim=-1)
    return positive_score, rank.float()


def aggregate_records(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for record in records:
        for key, value in record.items():
            if isinstance(value, bool):
                value = float(value)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                buckets[key].append(float(value))
    return {key: mean(values) for key, values in sorted(buckets.items())}


def model_parameter_group(name: str) -> str:
    if name == "backbone.q_G":
        return "backbone_q_g"
    if name.startswith("backbone.model.vision_model"):
        return "backbone_vision"
    if "visual_projection" in name and name.startswith("backbone"):
        return "backbone_visual_projection"
    if name.startswith("backbone.model.text_model"):
        return "backbone_text"
    if "text_projection" in name and name.startswith("backbone"):
        return "backbone_text_projection"
    if name.startswith("backbone.text_adapter"):
        return "backbone_text_adapter"
    for group in ("proposal", "grounder", "action_fusion", "executor", "score_net"):
        if name.startswith(group):
            return group
    return "other_model"


def objective_parameter_group(name: str) -> str:
    if name.startswith("concept"):
        return "objective_concept"
    if name.startswith("relation"):
        return "objective_relation"
    return "other_objective"


def parameter_movement_from_fresh(model: nn.Module, checkpoint_state: Mapping[str, Tensor]):
    acc = defaultdict(
        lambda: {"diff_sq": 0.0, "base_sq": 0.0, "abs_sum": 0.0, "max_abs": 0.0, "numel": 0.0}
    )
    for name, parameter in model.named_parameters():
        if name not in checkpoint_state:
            continue
        current = parameter.detach().float().cpu()
        trained = checkpoint_state[name].detach().float().cpu()
        if current.shape != trained.shape:
            continue
        diff = trained - current
        slot = acc[model_parameter_group(name)]
        slot["diff_sq"] += float(diff.square().sum())
        slot["base_sq"] += float(current.square().sum())
        slot["abs_sum"] += float(diff.abs().sum())
        slot["max_abs"] = max(slot["max_abs"], float(diff.abs().max()))
        slot["numel"] += float(diff.numel())
    result = {}
    for group, slot in acc.items():
        result[group] = {
            "relative_l2_change": math.sqrt(slot["diff_sq"])
            / max(math.sqrt(slot["base_sq"]), 1e-12),
            "mean_abs_change": slot["abs_sum"] / max(slot["numel"], 1.0),
            "max_abs_change": slot["max_abs"],
            "numel": slot["numel"],
        }
    return result


def gradient_group_stats(model: nn.Module, objective: nn.Module):
    acc = defaultdict(
        lambda: {
            "grad_sq": 0.0,
            "grad_max": 0.0,
            "numel": 0.0,
            "grad_numel": 0.0,
            "nonzero_numel": 0.0,
            "trainable_tensors": 0.0,
            "no_grad_tensors": 0.0,
        }
    )

    def consume(module: nn.Module, is_model: bool):
        for name, p in module.named_parameters():
            if not p.requires_grad:
                continue
            group = model_parameter_group(name) if is_model else objective_parameter_group(name)
            slot = acc[group]
            slot["numel"] += float(p.numel())
            slot["trainable_tensors"] += 1
            if p.grad is None:
                slot["no_grad_tensors"] += 1
                continue
            g = p.grad.detach().float()
            slot["grad_sq"] += float(g.square().sum())
            slot["grad_max"] = max(slot["grad_max"], float(g.abs().max()))
            slot["grad_numel"] += float(g.numel())
            slot["nonzero_numel"] += float((g != 0).sum())

    consume(model, True)
    consume(objective, False)
    result = {}
    for group, slot in acc.items():
        result[group] = {
            "grad_l2": math.sqrt(slot["grad_sq"]),
            "grad_max_abs": slot["grad_max"],
            "trainable_numel": slot["numel"],
            "trainable_tensors": slot["trainable_tensors"],
            "no_grad_tensor_fraction": slot["no_grad_tensors"]
            / max(slot["trainable_tensors"], 1.0),
            "nonzero_grad_element_fraction": slot["nonzero_numel"] / max(slot["grad_numel"], 1.0),
        }
    return result


def first_trainable(module: nn.Module | None):
    if module is None:
        return None
    for p in module.parameters():
        if p.requires_grad:
            return p
    return None


def build_gradient_probes(model: nn.Module, objective: IAGSRMEObjective):
    probes: dict[str, nn.Parameter] = {}
    candidates = [
        ("proposal", first_trainable(getattr(model, "proposal", None))),
        ("grounder", first_trainable(getattr(model, "grounder", None))),
        ("action_fusion", first_trainable(getattr(model, "action_fusion", None))),
        ("executor", first_trainable(getattr(model, "executor", None))),
        ("score_net", first_trainable(getattr(model, "score_net", None))),
        ("objective_concept", first_trainable(getattr(objective, "concept", None))),
        ("objective_relation", first_trainable(getattr(objective, "relation", None))),
    ]
    backbone = getattr(model, "backbone", None)
    if backbone is not None:
        qg = getattr(backbone, "q_G", None)
        if isinstance(qg, nn.Parameter) and qg.requires_grad:
            candidates.append(("backbone_q_g", qg))
        inner = getattr(backbone, "model", None)
        if inner is not None:
            candidates += [
                ("backbone_text", first_trainable(getattr(inner, "text_model", None))),
                ("backbone_vision", first_trainable(getattr(inner, "vision_model", None))),
                (
                    "backbone_visual_projection",
                    first_trainable(getattr(inner, "visual_projection", None)),
                ),
            ]
        candidates.append(
            ("backbone_text_adapter", first_trainable(getattr(backbone, "text_adapter", None)))
        )
    seen = set()
    for name, p in candidates:
        if p is not None and id(p) not in seen:
            probes[name] = p
            seen.add(id(p))
    return probes


def checkpoint_summary(checkpoint: Mapping[str, Any]):
    metadata = checkpoint.get("metadata") if isinstance(checkpoint.get("metadata"), dict) else {}
    optimizer = checkpoint.get("optimizer")
    optimizer_summary = {}
    if isinstance(optimizer, dict):
        states = optimizer.get("state", {})
        groups = optimizer.get("param_groups", [])
        steps = []
        if isinstance(states, dict):
            for value in states.values():
                if isinstance(value, dict) and "step" in value:
                    try:
                        steps.append(scalar(value["step"]))
                    except (TypeError, ValueError):
                        continue
        optimizer_summary = {
            "state_entries": len(states) if isinstance(states, dict) else None,
            "param_groups": len(groups) if isinstance(groups, list) else None,
            "step_min": min(steps) if steps else None,
            "step_max": max(steps) if steps else None,
            "learning_rates": [float(g["lr"]) for g in groups if isinstance(g, dict) and "lr" in g],
            "weight_decays": [
                float(g["weight_decay"])
                for g in groups
                if isinstance(g, dict) and "weight_decay" in g
            ],
        }
    return {
        "epoch": checkpoint.get("epoch"),
        "metric": checkpoint.get("metric"),
        "architecture": metadata.get("architecture"),
        "experiment_identity": metadata.get("experiment_identity"),
        "global_readout_mode": metadata.get("global_readout_mode"),
        "finetune_policy": metadata.get("finetune_policy"),
        "train_vision": metadata.get("train_vision"),
        "train_text": metadata.get("train_text"),
        "train_text_projection": metadata.get("train_text_projection"),
        "precision": metadata.get("precision"),
        "objective_config": metadata.get("objective_config"),
        "model_config": metadata.get("model_config"),
        "optimizer": optimizer_summary,
    }


def compare_objective_configs(current: ObjectiveConfig, stored: Mapping[str, Any] | None):
    if not isinstance(stored, Mapping):
        return {"stored_available": False, "differences": {}}
    current_dict = asdict(current)
    differences = {}
    for key in sorted(set(current_dict) | set(stored)):
        if current_dict.get(key) != stored.get(key):
            differences[key] = {"current": current_dict.get(key), "checkpoint": stored.get(key)}
    return {"stored_available": True, "differences": differences}


def build_objective(
    config: ObjectiveConfig,
    model: nn.Module,
    tokenizer: object,
    train_dataset: FashionIQDataset,
    max_text_length: int,
):
    vocabulary = None
    prototypes = None
    if config.concept_enabled:
        vocabulary = build_concept_vocabulary(train_dataset, config)
        prototypes = encode_concept_prototypes(model, tokenizer, vocabulary, max_text_length)
    return IAGSRMEObjective(
        config,
        state_dim=model.backbone.state_dim,
        concept_vocabulary=vocabulary,
        concept_prototypes=prototypes,
    )


def analyze_batch(
    model: nn.Module,
    objective: IAGSRMEObjective,
    output: Mapping[str, Any],
    losses: Mapping[str, Tensor],
    targets: Tensor,
    target_ids: Sequence[str | None],
):
    record: dict[str, float] = {}
    selection = Counter()
    cfg = objective.config

    raw = {
        "terminal": scalar(losses["terminal"]),
        "pair": scalar(losses["pair"]),
        "gain": scalar(losses["gain"]),
        "concept": scalar(losses["concept_loss"]),
        "bind": scalar(losses["bind_loss"]),
        "rel_ortho": scalar(losses["rel_ortho_loss"]),
        "dpp": scalar(losses["dpp_raw"]),
    }
    weighted = {
        "terminal": cfg.terminal_weight * raw["terminal"],
        "pair": cfg.lambda_pair * raw["pair"],
        "gain": cfg.lambda_gain * raw["gain"],
        "concept": cfg.lambda_c * raw["concept"],
        "bind": cfg.lambda_bind * raw["bind"],
        "rel_ortho": cfg.lambda_rel * raw["rel_ortho"],
        "dpp": cfg.lambda_dpp * raw["dpp"],
    }
    abs_total = sum(abs(v) for v in weighted.values())
    for name, value in raw.items():
        record[f"loss_raw/{name}"] = value
        record[f"loss_weighted/{name}"] = weighted[name]
        record[f"loss_abs_fraction/{name}"] = abs(weighted[name]) / max(abs_total, 1e-12)
    record["loss/total"] = scalar(losses["total"])

    for name in [
        "concept_positive_recall",
        "concept_negative_false_positive_rate",
        "instruction_concept_coverage",
        "prototype_occupancy",
        "prototype_entropy",
        "prototype_pairwise_cosine",
        "functional_pairwise_cosine",
        "functional_rank",
        "mean_delta_q_norm",
        "stop_rate",
        "mean_rollout_length",
        "useful_candidate_count",
        "dpp_valid_timestep_count",
        "dpp_valid_rate",
        "teacher_invalid_rows",
    ]:
        if name in losses:
            record[f"objective/{name}"] = scalar(losses[name])

    query = output["query"]
    initial_state = output["initial_state"]
    final_state = output["state"]
    initial_query = output["steps"][0]["current_query"] if output["steps"] else query
    positive, negative, _ = build_teacher_masks(target_ids, targets.device)
    initial_terminal = objective.terminal(initial_query, targets, positive)
    record["retrieval/initial_terminal_loss"] = scalar(initial_terminal)
    record["retrieval/final_terminal_loss"] = scalar(losses["terminal"])
    record["retrieval/terminal_loss_improvement"] = scalar(initial_terminal) - scalar(
        losses["terminal"]
    )
    init_pos, init_rank = positive_score_and_rank(initial_query, targets, positive, negative)
    final_pos, final_rank = positive_score_and_rank(query, targets, positive, negative)
    record["retrieval/positive_similarity_gain"] = float((final_pos - init_pos).mean().cpu())
    record["retrieval/inbatch_rank_improvement"] = float((init_rank - final_rank).mean().cpu())
    record["retrieval/final_inbatch_recall1"] = float((final_rank <= 1).float().mean().cpu())
    state_diff = (final_state.detach().float() - initial_state.detach().float()).flatten(1)
    state_base = initial_state.detach().float().flatten(1)
    record["rollout/final_state_relative_change"] = float(
        (state_diff.norm(dim=-1) / state_base.norm(dim=-1).clamp_min(1e-8)).mean().cpu()
    )
    record["rollout/final_initial_query_cosine"] = float(
        F.cosine_similarity(query.detach().float(), initial_query.detach().float(), dim=-1)
        .mean()
        .cpu()
    )

    buckets = defaultdict(list)
    teacher_all, predicted_all = [], []
    for step in output["steps"]:
        tensors = {
            k: step[k]
            for k in [
                "proposals",
                "entities",
                "actions",
                "grounding",
                "alpha_read",
                "exec_mask",
                "delta",
                "candidate_states",
                "candidate_queries",
                "delta_q",
                "scores",
                "fuse_gamma",
                "fuse_beta",
            ]
        }
        for name, tensor in tensors.items():
            buckets[f"finite/{name}"].append(float(finite_tensor(tensor)))

        for prefix, tensor in [
            ("proposal", step["proposals"]),
            ("entity", step["entities"]),
            ("action", step["actions"]),
            ("delta_q", step["delta_q"]),
            ("candidate_query", step["candidate_queries"]),
        ]:
            buckets[f"{prefix}/pairwise_cosine_mean"].append(pairwise_cosine_mean(tensor))
            buckets[f"{prefix}/pairwise_cosine_max"].append(pairwise_cosine_max(tensor))

        for prefix, tensor in [
            ("proposal", step["proposals"]),
            ("entity", step["entities"]),
            ("action", step["actions"]),
        ]:
            buckets[f"{prefix}/norm"].append(
                float(tensor.detach().float().norm(dim=-1).mean().cpu())
            )

        grounding = step["grounding"].detach().float()
        alpha = step["alpha_read"].detach().float()
        mask = step["exec_mask"].detach().float()
        buckets["grounding/raw_mean"].append(float(grounding.mean().cpu()))
        buckets["grounding/raw_std"].append(float(grounding.std(unbiased=False).cpu()))
        buckets["grounding/alpha_entropy"].append(float(normalized_entropy(alpha).mean().cpu()))
        buckets["grounding/alpha_peak"].append(float(alpha.max(dim=-1).values.mean().cpu()))
        buckets["grounding/exec_mask_mean"].append(float(mask.mean().cpu()))
        buckets["grounding/exec_mask_std"].append(float(mask.std(unbiased=False).cpu()))
        buckets["grounding/exec_mask_fraction_gt_0_5"].append(
            float((mask > 0.5).float().mean().cpu())
        )
        buckets["grounding/exec_mask_fraction_lt_0_05"].append(
            float((mask < 0.05).float().mean().cpu())
        )
        buckets["grounding/exec_mask_fraction_gt_0_95"].append(
            float((mask > 0.95).float().mean().cpu())
        )
        buckets["grounding/exec_mask_jaccard"].append(binary_mask_jaccard(step["exec_mask"]))

        gamma, beta = step["fuse_gamma"].detach().float(), step["fuse_beta"].detach().float()
        buckets["fusion/gamma_mean"].append(float(gamma.mean().cpu()))
        buckets["fusion/beta_mean"].append(float(beta.mean().cpu()))
        buckets["fusion/gamma_std"].append(float(gamma.std(unbiased=False).cpu()))
        buckets["fusion/beta_std"].append(float(beta.std(unbiased=False).cpu()))
        saturation = (
            ((gamma < 0.05) | (gamma > 0.95)).float().mean()
            + ((beta < 0.05) | (beta > 0.95)).float().mean()
        ) / 2
        buckets["fusion/gate_saturation"].append(float(saturation.cpu()))

        delta = step["delta"].detach().float().flatten(2)
        parent = step["parent_state"].detach().float().flatten(1)
        delta_norm = delta.norm(dim=-1)
        buckets["executor/delta_l2"].append(float(delta_norm.mean().cpu()))
        buckets["executor/delta_relative_parent"].append(
            float((delta_norm / parent.norm(dim=-1)[:, None].clamp_min(1e-8)).mean().cpu())
        )
        buckets["executor/candidate_state_pairwise_l2"].append(
            pairwise_l2_mean(step["candidate_states"])
        )

        delta_q = step["delta_q"].detach().float()
        dq_norm = delta_q.norm(dim=-1)
        buckets["effect/delta_q_norm"].append(float(dq_norm.mean().cpu()))
        buckets["effect/delta_q_sibling_singular_effective_rank"].append(effective_rank(delta_q))
        buckets["effect/delta_q_near_zero_fraction"].append(
            float((dq_norm < 1e-5).float().mean().cpu())
        )

        scores = step["scores"].detach().float()
        buckets["score/mean"].append(float(scores.mean().cpu()))
        buckets["score/std"].append(float(scores.std(unbiased=False).cpu()))
        buckets["score/range"].append(
            float((scores.max(dim=-1).values - scores.min(dim=-1).values).mean().cpu())
        )
        for idx in step["selected_idx"].detach().cpu().tolist():
            selection[int(idx)] += 1

        live = step["live_indices"]
        pos_live, neg_live = positive.index_select(0, live), negative.index_select(0, live)
        teacher, valid = marginal_teacher_utilities(
            step["current_query"],
            step["candidate_queries"],
            targets.detach(),
            pos_live,
            neg_live,
            cfg.retrieval_temperature,
        )
        if valid.any():
            teacher = teacher[valid].detach().float()
            predicted = step["scores"][valid].detach().float()
            teacher_all.append(teacher.flatten())
            predicted_all.append(predicted.flatten())
            buckets["teacher/utility_mean"].append(float(teacher.mean().cpu()))
            buckets["teacher/utility_std"].append(float(teacher.std(unbiased=False).cpu()))
            buckets["teacher/positive_candidate_fraction"].append(
                float((teacher > 0).float().mean().cpu())
            )
            quality = torch.sigmoid(teacher / cfg.tau_dpp)
            buckets["teacher/dpp_useful_candidate_count"].append(
                float((quality > cfg.useful_threshold).sum(dim=-1).float().mean().cpu())
            )

            left, right = torch.triu_indices(
                teacher.shape[1], teacher.shape[1], offset=1, device=teacher.device
            )
            td, pd = teacher[:, left] - teacher[:, right], predicted[:, left] - predicted[:, right]
            confident = td.abs() >= cfg.epsilon_pair
            if confident.any():
                correct = (td[confident] * pd[confident]) > 0
                ties = pd[confident].abs() < 1e-12
                buckets["score/pairwise_teacher_accuracy"].append(
                    float((correct.float() + 0.5 * ties.float()).mean().cpu())
                )

            selected = step["selected_idx"][valid]
            decision = selection_metrics(
                teacher,
                selected,
                stop_threshold=float(model.config.epsilon_stop),
            )
            stop_metrics = decision["stop"]
            buckets["decision/selected_teacher_utility"].append(decision["mean_selected_utility"])
            buckets["decision/oracle_teacher_utility"].append(decision["mean_oracle_utility"])
            buckets["decision/oracle_regret"].append(decision["mean_regret"])
            buckets["decision/harmful_execution_fraction_of_executions"].append(
                stop_metrics["harmful_execution_fraction_of_executions"]
            )
            buckets["decision/harmful_execution_fraction_of_decisions"].append(
                stop_metrics["harmful_execution_fraction_of_decisions"]
            )
            buckets["decision/missed_opportunity_stop_rate"].append(
                stop_metrics["premature_stop_count"] / decision["decision_count"]
            )
            buckets["decision/stop_execute_accuracy"].append(
                (stop_metrics["tp"] + stop_metrics["tn"]) / decision["decision_count"]
            )
            buckets["decision/exact_oracle_action_accuracy"].append(
                decision["selected_equals_oracle_fraction"]
            )

    for key, values in buckets.items():
        values = [v for v in values if math.isfinite(v)]
        if values:
            record[f"step/{key}"] = mean(values)
    if teacher_all:
        record["score/global_teacher_pearson"] = pearson(
            torch.cat(predicted_all), torch.cat(teacher_all)
        )
    return record, selection


def component_gradient_routing(
    model: nn.Module, objective: IAGSRMEObjective, losses: Mapping[str, Tensor]
):
    probes = build_gradient_probes(model, objective)
    names, params = list(probes), list(probes.values())
    components = {
        "terminal": losses["terminal"],
        "pair": losses["pair"],
        "gain": losses["gain"],
        "concept": losses["concept_loss"],
        "bind": losses["bind_loss"],
        "rel_ortho": losses["rel_ortho_loss"],
        "dpp": losses["dpp_weighted"],
    }
    routing = {}
    for loss_name, loss in components.items():
        route = {name: 0.0 for name in names}
        if loss.requires_grad:
            try:
                grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                for name, grad in zip(names, grads, strict=True):
                    if grad is not None:
                        route[name] = float(grad.detach().float().norm().cpu())
            except RuntimeError as exc:
                route["_error"] = str(exc)
        routing[loss_name] = route
    return routing


def run_gradient_diagnostic(
    model: nn.Module, objective: IAGSRMEObjective, batch: Any, precision: Any, device: torch.device
):
    model.train()
    objective.train()
    model.zero_grad(set_to_none=True)
    objective.zero_grad(set_to_none=True)
    batch = batch.to(device)
    assert batch.target_pixels is not None
    with torch.autocast(
        device_type=device.type, enabled=precision.autocast_enabled, dtype=precision.autocast_dtype
    ):
        output = model(
            batch.reference_pixels, batch.input_ids, batch.attention_mask, batch.content_mask
        )
        targets = model.encode_global_images(batch.target_pixels)
        losses = objective(output, targets, batch.target_ids, batch.modification_texts)
    routing = component_gradient_routing(model, objective, losses)
    losses["total"].backward()
    groups = gradient_group_stats(model, objective)
    all_finite = all(
        torch.isfinite(p.grad).all()
        for module in (model, objective)
        for p in module.parameters()
        if p.grad is not None
    )
    model.zero_grad(set_to_none=True)
    objective.zero_grad(set_to_none=True)
    model.eval()
    objective.eval()
    return {
        "all_gradients_finite": bool(all_finite),
        "component_routing_probe_grad_l2": routing,
        "total_gradient_groups": groups,
    }


def flag(flags, level, code, message):
    flags.append({"level": level, "code": code, "message": message})


def build_health_flags(
    agg: Mapping[str, float], gradient: Mapping[str, Any], config_diff: Mapping[str, Any]
):
    flags = []
    diffs = config_diff.get("differences", {})
    if diffs:
        flag(
            flags,
            "WARN",
            "OBJECTIVE_CONFIG_MISMATCH",
            f"Current config differs from checkpoint: {', '.join(sorted(diffs))}.",
        )

    bad_finite = [k for k, v in agg.items() if k.startswith("step/finite/") and v < 0.999]
    if bad_finite:
        flag(flags, "FAIL", "NONFINITE_FORWARD", f"NaN/Inf in: {', '.join(bad_finite)}")

    dq = agg.get("objective/mean_delta_q_norm", agg.get("step/effect/delta_q_norm"))
    if dq is not None and dq < 1e-5:
        flag(
            flags,
            "FAIL",
            "RETRIEVAL_EFFECT_NEAR_ZERO",
            f"mean delta_q norm={dq:.3e}; Executor is nearly invisible to retrieval.",
        )
    elif dq is not None and dq < 1e-4:
        flag(flags, "WARN", "RETRIEVAL_EFFECT_TINY", f"mean delta_q norm={dq:.3e}.")

    cos, rank = (
        agg.get("objective/functional_pairwise_cosine"),
        agg.get("objective/functional_rank"),
    )
    if cos is not None and rank is not None and cos > 0.98 and rank < 1.25:
        flag(
            flags,
            "FAIL",
            "FUNCTIONAL_CANDIDATE_COLLAPSE",
            f"delta_q cosine={cos:.4f}, effective_rank={rank:.3f}.",
        )
    elif cos is not None and cos > 0.90:
        flag(
            flags, "WARN", "FUNCTIONAL_CANDIDATES_TOO_SIMILAR", f"delta_q sibling cosine={cos:.4f}."
        )

    useful, dpp_rate = (
        agg.get("objective/useful_candidate_count"),
        agg.get("objective/dpp_valid_rate"),
    )
    if useful is not None and useful < 2:
        flag(
            flags,
            "WARN",
            "DPP_MOSTLY_GATED_OFF",
            f"useful candidates={useful:.2f}; DPP needs >=2, so lambda_dpp may not matter.",
        )
    if dpp_rate is not None and dpp_rate < 0.10:
        flag(flags, "WARN", "DPP_LOW_ACTIVATION", f"DPP valid rate={dpp_rate:.3f}.")

    for key, code, label in [
        ("step/proposal/pairwise_cosine_mean", "PROPOSAL_COLLAPSE", "Proposal"),
        ("step/action/pairwise_cosine_mean", "ACTION_COLLAPSE", "Action"),
    ]:
        value = agg.get(key)
        if value is not None and value > 0.97:
            flag(flags, "WARN", code, f"{label} sibling cosine={value:.4f}.")

    alpha = agg.get("step/grounding/alpha_entropy")
    if alpha is not None and alpha > 0.97:
        flag(flags, "WARN", "GROUNDING_TOO_DIFFUSE", f"normalized read entropy={alpha:.3f}.")
    elif alpha is not None and alpha < 0.08:
        flag(flags, "WARN", "GROUNDING_TOO_PEAKY", f"normalized read entropy={alpha:.3f}.")

    mask = agg.get("step/grounding/exec_mask_fraction_gt_0_5")
    if mask is not None and mask < 0.01:
        flag(flags, "WARN", "WRITE_MASK_ALMOST_EMPTY", f"Only {mask:.2%} positions >0.5.")
    elif mask is not None and mask > 0.95:
        flag(flags, "WARN", "WRITE_MASK_ALMOST_GLOBAL", f"{mask:.2%} positions >0.5.")
    jaccard = agg.get("step/grounding/exec_mask_jaccard")
    if jaccard is not None and jaccard > 0.90:
        flag(flags, "WARN", "WRITE_MASKS_IDENTICAL", f"Sibling mask Jaccard={jaccard:.3f}.")

    acc = agg.get("step/score/pairwise_teacher_accuracy")
    if acc is not None and acc < 0.55:
        flag(
            flags,
            "WARN",
            "SCORENET_WEAK_RANKING",
            f"ScoreNet pairwise teacher accuracy={acc:.3f}, near chance.",
        )
    corr = agg.get("score/global_teacher_pearson")
    if corr is not None and math.isfinite(corr) and corr < 0.20:
        flag(flags, "WARN", "SCORENET_LOW_CORRELATION", f"ScoreNet/teacher Pearson={corr:.3f}.")
    harmful = agg.get("step/decision/harmful_execution_fraction_of_executions")
    if harmful is not None and harmful > 0.20:
        flag(
            flags,
            "WARN",
            "HARMFUL_EXECUTIONS",
            f"{harmful:.1%} selected executions have negative teacher utility.",
        )

    retrieval_gain = agg.get("retrieval/terminal_loss_improvement")
    if retrieval_gain is not None and retrieval_gain <= 0:
        flag(
            flags,
            "FAIL",
            "ROLLOUT_DOES_NOT_IMPROVE_RETRIEVAL",
            f"Initial->final terminal improvement={retrieval_gain:.5f}.",
        )

    concept_recall = agg.get("objective/concept_positive_recall")
    concept_fpr = agg.get("objective/concept_negative_false_positive_rate")
    if concept_recall is not None and concept_recall < 0.50:
        flag(flags, "WARN", "LOW_CONCEPT_RECALL", f"Concept positive recall={concept_recall:.3f}.")
    if concept_fpr is not None and concept_fpr > 0.50:
        flag(
            flags,
            "WARN",
            "HIGH_CONCEPT_FALSE_POSITIVE_RATE",
            f"Concept negative FPR={concept_fpr:.3f}.",
        )
    occupancy = agg.get("objective/prototype_occupancy")
    if occupancy is not None and occupancy < 0.50:
        flag(flags, "WARN", "RELATION_PROTOTYPE_COLLAPSE", f"Prototype occupancy={occupancy:.1%}.")

    if gradient:
        if not gradient.get("all_gradients_finite", True):
            flag(flags, "FAIL", "NONFINITE_GRADIENT", "At least one gradient has NaN/Inf.")
        groups = gradient.get("total_gradient_groups", {})
        for name in ("proposal", "grounder", "action_fusion", "executor", "score_net"):
            info = groups.get(name)
            if (
                isinstance(info, Mapping)
                and info.get("trainable_numel", 0) > 0
                and info.get("grad_l2", 0) == 0
            ):
                flag(
                    flags,
                    "FAIL",
                    f"ZERO_GRAD_{name.upper()}",
                    f"{name} is trainable but gets zero total gradient.",
                )
        dpp_route = gradient.get("component_routing_probe_grad_l2", {}).get("dpp", {})
        if isinstance(dpp_route, Mapping) and dpp_route.get("executor", 0.0) == 0.0:
            flag(
                flags,
                "WARN",
                "DPP_NOT_REACHING_EXECUTOR_PROBE",
                "DPP gradient to representative Executor parameter is zero on this batch; usually the DPP guard is inactive.",
            )

    if not flags:
        flag(
            flags,
            "OK",
            "NO_MAJOR_AUTOMATIC_FLAGS",
            "No major issue crossed conservative thresholds; inspect full report anyway.",
        )
    return flags


def table(rows, headers):
    def fmt(v):
        if isinstance(v, float):
            if math.isnan(v):
                return "NaN"
            if 0 < abs(v) < 1e-3:
                return f"{v:.3e}"
            return f"{v:.6f}"
        return str(v)

    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    out += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in rows]
    return "\n".join(out)


def render_markdown(report: Mapping[str, Any]):
    agg, ckpt = report["aggregate"], report["checkpoint"]
    lines = [
        "# IAG-SRME V2 R0 Diagnostic Report",
        "",
        "## Checkpoint",
        "",
        f"- epoch: `{ckpt.get('epoch')}`; stored metric: `{ckpt.get('metric')}`",
        f"- identity: `{ckpt.get('experiment_identity')}`; readout: `{ckpt.get('global_readout_mode')}`; fine-tune: `{ckpt.get('finetune_policy')}`",
        "",
        "## Automatic health flags",
        "",
    ]
    for f in report["health_flags"]:
        lines.append(f"- **{f['level']} — {f['code']}**: {f['message']}")
    lines += ["", "## Loss decomposition", ""]
    rows = [
        [
            name,
            agg.get(f"loss_raw/{name}", float("nan")),
            agg.get(f"loss_weighted/{name}", float("nan")),
            agg.get(f"loss_abs_fraction/{name}", float("nan")),
        ]
        for name in ("terminal", "pair", "gain", "concept", "bind", "rel_ortho", "dpp")
    ]
    lines += [table(rows, ["Loss", "Raw", "Weighted", "|Contribution| fraction"]), ""]

    sections = {
        "Retrieval / rollout": [
            "retrieval/initial_terminal_loss",
            "retrieval/final_terminal_loss",
            "retrieval/terminal_loss_improvement",
            "retrieval/positive_similarity_gain",
            "retrieval/inbatch_rank_improvement",
            "rollout/final_state_relative_change",
            "rollout/final_initial_query_cosine",
            "objective/mean_rollout_length",
            "objective/stop_rate",
        ],
        "Collapse / diversity": [
            "step/proposal/pairwise_cosine_mean",
            "step/entity/pairwise_cosine_mean",
            "step/action/pairwise_cosine_mean",
            "objective/functional_pairwise_cosine",
            "objective/functional_rank",
            "objective/mean_delta_q_norm",
            "step/effect/delta_q_sibling_singular_effective_rank",
            "step/effect/delta_q_near_zero_fraction",
            "step/executor/candidate_state_pairwise_l2",
        ],
        "Grounding / write masks": [
            "step/grounding/alpha_entropy",
            "step/grounding/alpha_peak",
            "step/grounding/exec_mask_mean",
            "step/grounding/exec_mask_fraction_gt_0_5",
            "step/grounding/exec_mask_fraction_lt_0_05",
            "step/grounding/exec_mask_fraction_gt_0_95",
            "step/grounding/exec_mask_jaccard",
            "step/fusion/gamma_mean",
            "step/fusion/beta_mean",
            "step/fusion/gate_saturation",
        ],
        "ScoreNet / teacher": [
            "score/global_teacher_pearson",
            "step/score/pairwise_teacher_accuracy",
            "step/decision/selected_teacher_utility",
            "step/decision/oracle_teacher_utility",
            "step/decision/oracle_regret",
            "step/decision/harmful_execution_fraction_of_executions",
            "step/decision/harmful_execution_fraction_of_decisions",
            "step/decision/missed_opportunity_stop_rate",
            "step/decision/stop_execute_accuracy",
            "step/decision/exact_oracle_action_accuracy",
        ],
        "Semantic auxiliaries / DPP": [
            "objective/concept_positive_recall",
            "objective/concept_negative_false_positive_rate",
            "objective/instruction_concept_coverage",
            "objective/prototype_occupancy",
            "objective/prototype_entropy",
            "objective/prototype_pairwise_cosine",
            "objective/useful_candidate_count",
            "objective/dpp_valid_rate",
        ],
    }
    for title, keys in sections.items():
        lines += [
            f"## {title}",
            "",
            table([[k, agg.get(k, float("nan"))] for k in keys], ["Metric", "Value"]),
            "",
        ]

    gradient = report.get("gradient", {})
    if gradient:
        lines += ["## Total gradient by module", ""]
        rows = [
            [
                g,
                v.get("grad_l2", float("nan")),
                v.get("grad_max_abs", float("nan")),
                v.get("nonzero_grad_element_fraction", float("nan")),
                v.get("no_grad_tensor_fraction", float("nan")),
            ]
            for g, v in sorted(gradient.get("total_gradient_groups", {}).items())
        ]
        lines += [
            table(
                rows,
                ["Module", "grad L2", "max |grad|", "nonzero elem frac", "no-grad tensor frac"],
            ),
            "",
        ]
        routing = gradient.get("component_routing_probe_grad_l2", {})
        probes = sorted(
            {
                p
                for r in routing.values()
                if isinstance(r, Mapping)
                for p in r
                if not p.startswith("_")
            }
        )
        rows = [
            [loss] + [float(route.get(p, 0.0)) for p in probes]
            for loss, route in routing.items()
            if isinstance(route, Mapping)
        ]
        lines += ["## Per-loss gradient routing probes", "", table(rows, ["Loss"] + probes), ""]

    movement = report.get("parameter_movement_from_fresh", {})
    if movement:
        rows = [
            [g, v["relative_l2_change"], v["mean_abs_change"], v["max_abs_change"], int(v["numel"])]
            for g, v in sorted(movement.items())
        ]
        lines += [
            "## Approximate checkpoint movement from fresh initialization",
            "",
            table(
                rows, ["Group", "relative L2 change", "mean abs change", "max abs change", "numel"]
            ),
            "",
            "> Interpret this only when seed/code/config reproduce the same fresh initialization.",
            "",
        ]

    diffs = report["objective_config_comparison"].get("differences", {})
    if diffs:
        lines += [
            "## Current vs checkpoint objective config",
            "",
            table(
                [[k, v.get("current"), v.get("checkpoint")] for k, v in diffs.items()],
                ["Field", "Current", "Checkpoint"],
            ),
            "",
        ]
    lines += [
        "## Selection histogram",
        "",
        f"`{json.dumps(report['selection_histogram'], sort_keys=True)}`",
        "",
        "## All aggregate metrics",
        "",
        table([[k, v] for k, v in sorted(agg.items())], ["Metric", "Mean"]),
        "",
        "> In-batch retrieval and target-privileged teacher statistics are diagnostics only, not official FashionIQ validation metrics.",
    ]
    return "\n".join(lines)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    checkpoint_value = cfg.get("checkpoint")
    if checkpoint_value is None:
        raise ValueError("Pass +checkpoint=path/to/best.pt")
    checkpoint_path = Path(str(checkpoint_value))
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    diagnostic_batches = int(cfg.get("diagnostic_batches", 6))
    diagnostic_batch_size = int(cfg.get("diagnostic_batch_size", 8))
    manifest_value = cfg.get("diagnostic_manifest")
    if manifest_value is None:
        raise ValueError(
            "pass +diagnostic_manifest=path/to/shared_manifest.json; "
            "all checkpoint diagnostics must replay one cohort"
        )
    diagnostic_split = str(cfg.get("diagnostic_split", "train"))
    deep_grad = bool(cfg.get("diagnostic_deep_grad", True))
    movement_enabled = bool(cfg.get("diagnostic_parameter_movement", True))
    use_checkpoint_objective = bool(cfg.get("diagnostic_use_checkpoint_objective", True))

    seed_everything(int(cfg.seed), bool(cfg.runtime.deterministic))
    configure_torch_runtime(
        deterministic=bool(cfg.runtime.deterministic), benchmark=bool(cfg.runtime.benchmark)
    )
    device = resolve_device(str(cfg.runtime.device), int(cfg.runtime.accelerator_index))
    precision = resolve_precision(str(cfg.runtime.precision), device)
    print(f"[diagnostic] device={device} checkpoint={checkpoint_path}")

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("checkpoint does not contain model state")
    validate_checkpoint_backbone_metadata(
        checkpoint.get("metadata"),
        str(cfg.backbone.checkpoint),
        str(cfg.backbone.revision),
        str(cfg.backbone.global_readout_mode),
        expected_readout_experiment=str(cfg.backbone.readout_experiment),
        expected_finetune_policy=str(cfg.backbone.finetune_policy),
        expected_train_vision=bool(cfg.backbone.train_vision),
        expected_train_text=bool(cfg.backbone.train_text),
        expected_train_text_projection=bool(cfg.backbone.train_text_projection),
        expected_model_config={
            "num_candidates": int(cfg.model.num_candidates),
            "max_steps": int(cfg.model.max_steps),
            "stop_enabled": bool(cfg.model.stop_enabled),
            "epsilon_stop": float(cfg.model.epsilon_stop),
        },
        allow_counterfactual=bool(cfg.get("allow_counterfactual_eval", False)),
    )

    model, tokenizer, processor = build_model(cfg)
    movement = parameter_movement_from_fresh(model, checkpoint["model"]) if movement_enabled else {}
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    dataset_root = Path(cfg.dataset.root)
    annotation_root = dataset_root / str(cfg.dataset.annotation_dir)
    image_store = DirectoryImageStore(dataset_root / str(cfg.dataset.image_dir))
    train_dataset = FashionIQDataset(
        annotation_root,
        "train",
        CATEGORIES,
        caption_policy=str(cfg.experiment.train_caption_policy),
        seed=int(cfg.seed),
    )
    caption_policy = (
        str(cfg.experiment.train_caption_policy)
        if diagnostic_split == "train"
        else str(cfg.experiment.val_caption_policy)
    )
    diagnostic_dataset = FashionIQDataset(
        annotation_root,
        diagnostic_split,
        CATEGORIES,
        caption_policy=caption_policy,
        seed=int(cfg.seed),
    )

    current_obj = ObjectiveConfig(**{k: v for k, v in cfg.objective.items() if k != "name"})
    metadata = checkpoint.get("metadata") if isinstance(checkpoint.get("metadata"), dict) else {}
    stored_obj = metadata.get("objective_config") if isinstance(metadata, dict) else None
    comparison = compare_objective_configs(current_obj, stored_obj)
    active_obj = (
        ObjectiveConfig(**dict(stored_obj))
        if use_checkpoint_objective and isinstance(stored_obj, Mapping)
        else current_obj
    )
    print(
        "[diagnostic] objective source:",
        "checkpoint" if active_obj is not current_obj else "current config",
    )

    objective = build_objective(
        active_obj, model, tokenizer, train_dataset, int(cfg.backbone.max_text_length)
    ).to(device)
    if "objective" in checkpoint:
        missing, unexpected = objective.load_state_dict(checkpoint["objective"], strict=False)
        if missing or unexpected:
            print(
                "[diagnostic] objective state mismatch",
                {"missing": missing, "unexpected": unexpected},
            )

    collator = FashionIQImageCollator(
        image_store, tokenizer, processor, int(cfg.backbone.max_text_length), include_targets=True
    )
    cohort, manifest = load_or_create_manifest(
        diagnostic_dataset,
        str(manifest_value),
        sample_count=diagnostic_batches * diagnostic_batch_size,
        batch_size=diagnostic_batch_size,
        seed=int(cfg.seed),
        split=diagnostic_split,
        caption_policy=caption_policy,
    )
    loader = DataLoader(
        cohort,
        batch_size=diagnostic_batch_size,
        shuffle=False,
        num_workers=int(cfg.experiment.num_workers),
        pin_memory=True,
        collate_fn=collator,
        drop_last=False,
    )

    model.eval()
    objective.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    records, selections = [], Counter()
    gradient_batch = None
    processed_sample_ids: list[str] = []
    live_sample_ids_by_t: dict[int, list[str]] = defaultdict(list)
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            processed_sample_ids.extend(batch.sample_ids)
            if gradient_batch is None:
                gradient_batch = batch
            batch = batch.to(device)
            assert batch.target_pixels is not None
            with torch.autocast(
                device_type=device.type,
                enabled=precision.autocast_enabled,
                dtype=precision.autocast_dtype,
            ):
                output = model(
                    batch.reference_pixels,
                    batch.input_ids,
                    batch.attention_mask,
                    batch.content_mask,
                )
                targets = model.encode_global_images(batch.target_pixels)
                losses = objective(output, targets, batch.target_ids, batch.modification_texts)
            for step in output["steps"]:
                live_sample_ids_by_t[int(step["timestep"])].extend(
                    [
                        batch.sample_ids[index]
                        for index in step["live_indices"].detach().cpu().tolist()
                    ]
                )
            record, selected = analyze_batch(
                model, objective, output, losses, targets, batch.target_ids
            )
            records.append(record)
            selections.update(selected)
            print(
                f"[diagnostic] batch={batch_index + 1}/{len(loader)} "
                f"total={record['loss/total']:.4f} "
                f"dq={record.get('objective/mean_delta_q_norm', float('nan')):.3e} "
                f"cos={record.get('objective/functional_pairwise_cosine', float('nan')):.4f} "
                f"rank={record.get('objective/functional_rank', float('nan')):.3f} "
                f"dpp_valid={record.get('objective/dpp_valid_rate', float('nan')):.3f}"
            )

    cohort_metadata = validate_processed_manifest(
        manifest, processed_sample_ids, batch_size=diagnostic_batch_size
    )
    if not records:
        raise RuntimeError("no complete diagnostic batch")
    aggregate = aggregate_records(records)
    gradient = (
        run_gradient_diagnostic(model, objective, gradient_batch, precision, device)
        if deep_grad and gradient_batch is not None
        else {}
    )
    memory = (
        {
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
        }
        if device.type == "cuda"
        else None
    )
    health = build_health_flags(aggregate, gradient, comparison)

    report = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint": checkpoint_summary(checkpoint),
        "active_objective_config": asdict(active_obj),
        "current_objective_config": asdict(current_obj),
        "objective_config_comparison": comparison,
        "diagnostic": {
            "batches": len(records),
            "batch_size": diagnostic_batch_size,
            "precision": str(cfg.runtime.precision),
            "device": str(device),
            "deep_grad": deep_grad,
            "use_checkpoint_objective": use_checkpoint_objective,
            "manifest_path": str(manifest_value),
            "split": diagnostic_split,
            "caption_policy": caption_policy,
            **cohort_metadata,
        },
        "metric_definitions": {
            "delta_q_sibling_singular_effective_rank": (
                "Per-input effective rank of the K-by-D delta_q sibling singular values; "
                "not cross-input covariance participation ratio."
            ),
            "cross_input_covariance_rank_authority": (
                "diagnose_latent_geometry.py reports per-slot and slot-centered "
                "cross-input covariance participation ratio."
            ),
            "positive_retrieval_utility": "candidate utility > 0",
            "oracle_execute": (
                "maximum teacher utility > model epsilon_stop; this may differ from "
                "positive utility when epsilon_stop > 0"
            ),
            "harmful_execution_fraction_of_executions": (
                "harmful executed actions / all executed actions"
            ),
            "harmful_execution_fraction_of_decisions": (
                "harmful executed actions / all live teacher-valid decisions"
            ),
        },
        "timestep_cohorts": {
            str(timestep): {
                "live_sample_ids": sample_ids,
                "live_sample_count": len(sample_ids),
                "live_sample_ids_fingerprint": sample_ids_fingerprint(sample_ids),
            }
            for timestep, sample_ids in sorted(live_sample_ids_by_t.items())
        },
        "aggregate": aggregate,
        "per_batch": records,
        "selection_histogram": {str(k): int(v) for k, v in sorted(selections.items())},
        "gradient": gradient,
        "parameter_movement_from_fresh": movement,
        "peak_cuda_memory": memory,
        "health_flags": health,
    }

    out = Path(HydraConfig.get().runtime.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out / "diagnostic_report.json", out / "diagnostic_report.md"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print("\n================ HEALTH FLAGS ================")
    for f in health:
        print(f"[{f['level']}] {f['code']}: {f['message']}")
    print("==============================================")
    print(f"[diagnostic] JSON: {json_path}")
    print(f"[diagnostic] Markdown: {md_path}")


if __name__ == "__main__":
    main()
