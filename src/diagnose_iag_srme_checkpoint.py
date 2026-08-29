from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from data.images import FashionIQImageCollator, ImageBatch
from datasets.common import DirectoryImageStore
from diagnostics.iag_srme import functional_effective_rank
from evaluation.fashioniq import (
    build_fashioniq_gallery,
    build_validation_datasets,
    encode_gallery,
    evaluate_fashioniq_recall,
)
from models.iag_srme import (
    BackboneOutput,
    FGCLIPBackbone,
    FGCLIPRegime,
    IAGSRME,
    IAGSRMEConfig,
    IAGSRMECore,
    IAGSRMEOutput,
)
from runtime import configure_torch_runtime, resolve_device, seed_everything


CATEGORIES = ("dress", "shirt", "toptee")
PROTOCOL = "fashioniq_original"
SPLIT = "val"
CAPTION_POLICY = "ordered_and"
REQUIRED_REPORT_KEYS = {
    "checkpoint",
    "checkpoint_epoch",
    "checkpoint_metric",
    "backbone_metadata",
    "protocol",
    "global_metrics",
    "per_category_metrics",
    "selection_diagnostics",
    "grounding_diagnostics",
    "functional_diagnostics",
    "dynamic_diagnostics",
    "control_retrieval_metrics",
    "specialization_matrices",
    "failure_flags",
}


def _cosine_matrix(values: Tensor) -> Tensor:
    normalized = F.normalize(values.float(), dim=-1)
    return normalized @ normalized.transpose(-1, -2)


def _off_diagonal_mean(matrix: Tensor) -> float:
    size = matrix.shape[-1]
    mask = ~torch.eye(size, dtype=torch.bool)
    return float(matrix[mask].mean())


@dataclass
class _MeanTensor:
    total: Tensor | None = None
    count: int = 0

    def update(self, values: Tensor) -> None:
        values = values.detach().float().cpu()
        if values.shape[0] == 0:
            return
        batch_total = values.sum(dim=0, dtype=torch.float64)
        self.total = batch_total if self.total is None else self.total + batch_total
        self.count += values.shape[0]

    def mean(self) -> Tensor:
        if self.total is None or self.count == 0:
            raise RuntimeError("diagnostic accumulator has no observations")
        return (self.total / self.count).float()


@dataclass
class _ScalarStats:
    total: float = 0.0
    count: int = 0
    maximum: float = 0.0

    def update(self, values: Tensor) -> None:
        values = values.detach().float()
        if values.numel() == 0:
            return
        self.total += float(values.sum())
        self.count += values.numel()
        self.maximum = max(self.maximum, float(values.max()))

    def summary(self) -> dict[str, float | int]:
        return {
            "mean_absolute_change": self.total / max(self.count, 1),
            "max_absolute_change": self.maximum,
            "element_count": self.count,
        }


@dataclass
class SingleControlResult:
    query: Tensor
    state: Tensor
    executed_edit_count: int = 1


def single_candidate_control(output: IAGSRMEOutput, candidate: int) -> SingleControlResult:
    """Execute candidate k exactly once from Z0, then conceptually STOP."""

    if not 0 <= candidate < output.intents.shape[1]:
        raise ValueError("candidate index outside the candidate bank")
    first = output.trace[0]
    if not first.live_before.all():
        raise AssertionError("all validation samples must be live at the root state")
    return SingleControlResult(
        query=first.candidate_queries[:, candidate],
        state=first.candidate_states[:, candidate],
    )


def repeat_candidate_control(
    core: IAGSRMECore, encoded: BackboneOutput, candidate: int
) -> IAGSRMEOutput:
    if not 0 <= candidate < core.config.num_candidates:
        raise ValueError("candidate index outside the candidate bank")
    return core(encoded, control=f"repeat_candidate_{candidate + 1}")


def same_parent_candidate_queries(
    output: IAGSRMEOutput, timestep: int
) -> tuple[Tensor, Tensor]:
    """Return qhat candidates only for valid/live parent states at timestep t."""

    step = output.trace[timestep]
    expected = step.current_state[:, None] + step.delta_z
    if not torch.equal(step.candidate_states, expected):
        raise AssertionError("counterfactual queries are not from the same parent state")
    return step.candidate_queries[step.live_before], step.live_before


def _selection_batch_counts(output: IAGSRMEOutput) -> dict[str, Tensor]:
    candidates = output.intents.shape[1]
    actions = torch.stack([step.selected_index for step in output.trace], dim=1)
    live = torch.stack([step.live_before for step in output.trace], dim=1)
    stopped_now = torch.stack([step.stopped_now for step in output.trace], dim=1)
    occupancy = actions.eq(candidates)
    executed = live & actions.lt(candidates)
    repeated = torch.zeros(actions.shape[0], dtype=torch.bool, device=actions.device)
    for row in range(actions.shape[0]):
        sequence = actions[row][executed[row]].tolist()
        repeated[row] = len(sequence) != len(set(sequence))
    return {
        "actions": actions,
        "live": live,
        "stopped_now": stopped_now,
        "stop_occupancy": occupancy,
        "executed": executed,
        "repeated": repeated,
    }


class ValidationDiagnosticAccumulator:
    def __init__(self, candidates: int = 4, timesteps: int = 3) -> None:
        self.candidates = candidates
        self.timesteps = timesteps
        self.queries = 0
        self.candidate_counts = torch.zeros(candidates, dtype=torch.long)
        self.live_action_counts = torch.zeros(candidates + 1, dtype=torch.long)
        self.stop_occupancy = torch.zeros(timesteps, dtype=torch.long)
        self.new_stop_counts = torch.zeros(timesteps, dtype=torch.long)
        self.live_counts = torch.zeros(timesteps, dtype=torch.long)
        self.executed_edits = 0
        self.repeated_queries = 0
        self.visual_tokens: int | None = None

        self.support_fraction = _MeanTensor()
        self.support_entropy = _MeanTensor()
        self.support_effective_size = _MeanTensor()
        self.support_cosine = _MeanTensor()
        self.support_overlap = _MeanTensor()
        self.dominant_grounding_share = _MeanTensor()
        self.intent_cosine = _MeanTensor()

        self.delta_z_norm = [_MeanTensor() for _ in range(timesteps)]
        self.delta_q_norm = [_MeanTensor() for _ in range(timesteps)]
        self.effect_rank = [_MeanTensor() for _ in range(timesteps)]
        self.context_cosine = [_MeanTensor() for _ in range(timesteps)]
        self.delta_z_cosine = [_MeanTensor() for _ in range(timesteps)]
        self.delta_q_cosine = [_MeanTensor() for _ in range(timesteps)]
        self.context_cosine_all = _MeanTensor()
        self.delta_z_cosine_all = _MeanTensor()
        self.delta_q_cosine_all = _MeanTensor()

        names = ("g_t", "d_t", "context", "delta_z", "candidate_query", "scores")
        self.dynamic = {
            name: [_ScalarStats() for _ in range(timesteps - 1)] for name in names
        }

    def update(self, output: IAGSRMEOutput) -> None:
        batch_size, candidates, tokens = output.supports.shape
        if candidates != self.candidates or len(output.trace) != self.timesteps:
            raise ValueError("diagnostic accumulator/model K or Tmax mismatch")
        self.queries += batch_size
        self.visual_tokens = tokens

        selection = _selection_batch_counts(output)
        actions = selection["actions"]
        live = selection["live"]
        executed = selection["executed"]
        for candidate in range(candidates):
            count = (executed & actions.eq(candidate)).sum().cpu()
            self.candidate_counts[candidate] += count
            self.live_action_counts[candidate] += count
        self.live_action_counts[candidates] += selection["stopped_now"].sum().cpu()
        self.stop_occupancy += selection["stop_occupancy"].sum(dim=0).cpu()
        self.new_stop_counts += selection["stopped_now"].sum(dim=0).cpu()
        self.live_counts += live.sum(dim=0).cpu()
        self.executed_edits += int(executed.sum())
        self.repeated_queries += int(selection["repeated"].sum())

        supports = output.supports.float()
        entropy = -(supports * supports.clamp_min(1e-8).log()).sum(dim=-1)
        self.support_fraction.update((supports > 0).float().mean(dim=-1))
        self.support_entropy.update(entropy)
        self.support_effective_size.update(entropy.exp())
        self.support_cosine.update(_cosine_matrix(supports))
        overlap = torch.minimum(supports[:, :, None], supports[:, None, :]).sum(dim=-1)
        self.support_overlap.update(overlap)
        dominant_share = supports.max(dim=1).values.sum(dim=-1) / supports.sum(
            dim=(1, 2)
        ).clamp_min(1e-8)
        self.dominant_grounding_share.update(dominant_share[:, None])
        self.intent_cosine.update(_cosine_matrix(output.intents))

        for timestep, step in enumerate(output.trace):
            valid = step.live_before
            if not valid.any():
                continue
            delta_z = step.delta_z[valid].float()
            delta_q = step.delta_q[valid].float()
            contexts = step.contexts[valid].float()
            self.delta_z_norm[timestep].update(delta_z.flatten(2).norm(dim=-1))
            self.delta_q_norm[timestep].update(delta_q.norm(dim=-1))
            self.effect_rank[timestep].update(
                functional_effective_rank(delta_q)[:, None]
            )
            context_matrix = _cosine_matrix(contexts)
            delta_z_matrix = _cosine_matrix(delta_z.flatten(2))
            delta_q_matrix = _cosine_matrix(delta_q)
            self.context_cosine[timestep].update(context_matrix)
            self.delta_z_cosine[timestep].update(delta_z_matrix)
            self.delta_q_cosine[timestep].update(delta_q_matrix)
            self.context_cosine_all.update(context_matrix)
            self.delta_z_cosine_all.update(delta_z_matrix)
            self.delta_q_cosine_all.update(delta_q_matrix)

        for transition, (previous, current) in enumerate(
            zip(output.trace[:-1], output.trace[1:], strict=True)
        ):
            valid = previous.live_before & previous.selected_index.lt(candidates)
            if not valid.any():
                continue
            pairs = {
                "g_t": (previous.current_evidence, current.current_evidence),
                "d_t": (
                    previous.accumulated_local_change,
                    current.accumulated_local_change,
                ),
                "context": (previous.contexts, current.contexts),
                "delta_z": (previous.delta_z, current.delta_z),
                "candidate_query": (
                    previous.candidate_queries,
                    current.candidate_queries,
                ),
                "scores": (previous.scores, current.scores),
            }
            for name, (before, after) in pairs.items():
                self.dynamic[name][transition].update((after[valid] - before[valid]).abs())

    def selection_summary(self) -> dict[str, Any]:
        edit_total = int(self.candidate_counts.sum())
        live_total = int(self.live_action_counts.sum())
        return {
            "candidate_distribution_conditional_on_edit": (
                self.candidate_counts.float() / max(edit_total, 1)
            ).tolist(),
            "live_action_distribution_candidates_plus_stop": (
                self.live_action_counts.float() / max(live_total, 1)
            ).tolist(),
            "stop_distribution_among_live_decisions": float(
                self.live_action_counts[-1] / max(live_total, 1)
            ),
            "absorbed_stop_occupancy_by_timestep": (
                self.stop_occupancy.float() / max(self.queries, 1)
            ).tolist(),
            "new_stop_hazard_by_timestep": (
                self.new_stop_counts.float() / self.live_counts.clamp_min(1)
            ).tolist(),
            "mean_executed_edit_count": self.executed_edits / max(self.queries, 1),
            "fraction_queries_with_repeated_candidate_selections": (
                self.repeated_queries / max(self.queries, 1)
            ),
            "counts": {
                "queries": self.queries,
                "candidate_selections": self.candidate_counts.tolist(),
                "new_stops": self.new_stop_counts.tolist(),
                "live_parents": self.live_counts.tolist(),
            },
        }

    def grounding_summary(self) -> dict[str, Any]:
        fraction = self.support_fraction.mean()
        entropy = self.support_entropy.mean()
        effective_size = self.support_effective_size.mean()
        cosine = self.support_cosine.mean()
        overlap = self.support_overlap.mean()
        return {
            "stable_over_recurrence": True,
            "visual_token_count": self.visual_tokens,
            "support_fraction": float(fraction.mean()),
            "support_entropy": float(entropy.mean()),
            "support_effective_size": float(effective_size.mean()),
            "per_candidate_support_fraction": fraction.tolist(),
            "per_candidate_support_entropy": entropy.tolist(),
            "per_candidate_support_effective_size": effective_size.tolist(),
            "pairwise_support_cosine_mean_off_diagonal": _off_diagonal_mean(cosine),
            "pairwise_support_overlap_mean_off_diagonal": _off_diagonal_mean(overlap),
            "dominant_tokenwise_grounding_mass_share": float(
                self.dominant_grounding_share.mean().mean()
            ),
            "dominant_share_definition": (
                "sum_n max_k P[k,n] divided by total candidate support mass; "
                "diagnostic only, not semantic ownership"
            ),
        }

    def functional_summary(self) -> dict[str, Any]:
        per_timestep = []
        for timestep in range(self.timesteps):
            if self.delta_z_norm[timestep].count == 0:
                per_timestep.append({"timestep": timestep, "live_parent_count": 0})
                continue
            delta_z_norm = self.delta_z_norm[timestep].mean()
            delta_q_norm = self.delta_q_norm[timestep].mean()
            delta_q_cosine = self.delta_q_cosine[timestep].mean()
            per_timestep.append(
                {
                    "timestep": timestep,
                    "live_parent_count": self.delta_z_norm[timestep].count,
                    "mean_delta_z_norm": float(delta_z_norm.mean()),
                    "mean_delta_q_norm": float(delta_q_norm.mean()),
                    "candidate_wise_delta_z_norm": delta_z_norm.tolist(),
                    "candidate_wise_effect_norm": delta_q_norm.tolist(),
                    "pairwise_delta_q_cosine_mean_off_diagonal": _off_diagonal_mean(
                        delta_q_cosine
                    ),
                    "functional_effective_rank": float(
                        self.effect_rank[timestep].mean().mean()
                    ),
                }
            )
        return {"per_timestep": per_timestep}

    def dynamic_summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, transitions in self.dynamic.items():
            summaries = [stats.summary() for stats in transitions]
            total_count = sum(stats.count for stats in transitions)
            total = sum(stats.total for stats in transitions)
            result[name] = {
                "per_transition": summaries,
                "overall_mean_absolute_change": total / max(total_count, 1),
                "overall_max_absolute_change": max(
                    (stats.maximum for stats in transitions), default=0.0
                ),
            }
        return result

    def specialization_summary(self) -> dict[str, Any]:
        return {
            "pairwise_intent_cosine": self.intent_cosine.mean().tolist(),
            "pairwise_support_cosine": self.support_cosine.mean().tolist(),
            "pairwise_context_cosine": self.context_cosine_all.mean().tolist(),
            "pairwise_delta_z_cosine": self.delta_z_cosine_all.mean().tolist(),
            "pairwise_delta_q_cosine": self.delta_q_cosine_all.mean().tolist(),
        }


def _retrieval_metrics(
    queries: Tensor,
    gallery: Tensor,
    target_ids: Sequence[str],
    gallery_ids: Sequence[str],
    reference_ids: Sequence[str],
) -> dict[str, float]:
    queries = queries.to(gallery.device)
    scores = F.normalize(queries.float(), dim=-1) @ F.normalize(gallery.float(), dim=-1).T
    result = evaluate_fashioniq_recall(scores, target_ids, gallery_ids, reference_ids)
    result["mean_recall"] = 0.5 * (result["recall_at_10"] + result["recall_at_50"])
    return result


def _oracle_candidate_metrics(
    candidate_queries: Tensor,
    gallery: Tensor,
    target_ids: Sequence[str],
    gallery_ids: Sequence[str],
    reference_ids: Sequence[str],
) -> dict[str, float]:
    candidate_queries = candidate_queries.to(gallery.device)
    scores = torch.einsum(
        "qkd,gd->qkg",
        F.normalize(candidate_queries.float(), dim=-1),
        F.normalize(gallery.float(), dim=-1),
    )
    gallery_index = {image_id: index for index, image_id in enumerate(gallery_ids)}
    for row, reference_id in enumerate(reference_ids):
        if reference_id in gallery_index and reference_id != target_ids[row]:
            scores[row, :, gallery_index[reference_id]] = -torch.inf
    targets = torch.tensor(
        [gallery_index[target_id] for target_id in target_ids], device=scores.device
    )[:, None, None]
    result: dict[str, float] = {}
    for k in (10, 50):
        hits = scores.topk(k, dim=-1).indices.eq(targets).any(dim=-1).any(dim=-1)
        result[f"recall_at_{k}"] = float(hits.float().mean() * 100.0)
    result["mean_recall"] = 0.5 * (result["recall_at_10"] + result["recall_at_50"])
    return result


def _counterfactual_retrieval_metrics(
    candidate_queries: Tensor,
    gallery: Tensor,
    target_ids: Sequence[str],
    gallery_ids: Sequence[str],
    reference_ids: Sequence[str],
) -> dict[str, Any]:
    candidates = candidate_queries.shape[1]
    result: dict[str, Any] = {
        f"candidate_{candidate}": _retrieval_metrics(
            candidate_queries[:, candidate],
            gallery,
            target_ids,
            gallery_ids,
            reference_ids,
        )
        for candidate in range(candidates)
    }
    result["best_single_candidate_oracle"] = _oracle_candidate_metrics(
        candidate_queries, gallery, target_ids, gallery_ids, reference_ids
    )
    result["mean_candidate_query"] = _retrieval_metrics(
        candidate_queries.mean(dim=1),
        gallery,
        target_ids,
        gallery_ids,
        reference_ids,
    )
    return result


def _macro_numeric_tree(items: Sequence[Any]) -> Any:
    if not items:
        raise ValueError("cannot macro-average an empty result list")
    first = items[0]
    if isinstance(first, Mapping):
        return {key: _macro_numeric_tree([item[key] for item in items]) for key in first}
    if isinstance(first, (float, int)) and not isinstance(first, bool):
        return sum(float(item) for item in items) / len(items)
    raise TypeError(f"non-numeric retrieval result cannot be macro-averaged: {type(first)}")


def _macro_control_results(category_controls: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    control_names = (
        "full",
        "reference_only",
        "single_0",
        "single_1",
        "single_2",
        "single_3",
        "repeat_0",
        "repeat_1",
        "repeat_2",
        "repeat_3",
        "mean_candidate",
    )
    result = {
        name: _macro_numeric_tree([controls[name] for controls in category_controls.values()])
        for name in control_names
    }
    result["counterfactual_same_parent_by_timestep"] = {}
    for timestep in range(3):
        key = f"t{timestep}"
        category_items = [
            controls["counterfactual_same_parent_by_timestep"][key]
            for controls in category_controls.values()
        ]
        valid = [item for item in category_items if "candidate_0" in item]
        if not valid:
            result["counterfactual_same_parent_by_timestep"][key] = {
                "live_parent_count": 0
            }
            continue
        without_counts = [
            {name: value for name, value in item.items() if name != "live_parent_count"}
            for item in valid
        ]
        averaged = _macro_numeric_tree(without_counts)
        averaged["live_parent_count"] = sum(int(item["live_parent_count"]) for item in valid)
        result["counterfactual_same_parent_by_timestep"][key] = averaged
    return result


def _usefulness_ratios(controls: Mapping[str, Any]) -> dict[str, float]:
    full = float(controls["full"]["mean_recall"])
    denominator = max(full, 1e-8)
    best_single = max(float(controls[f"single_{index}"]["mean_recall"]) for index in range(4))
    best_repeat = max(float(controls[f"repeat_{index}"]["mean_recall"]) for index in range(4))
    return {
        "best_single_over_full": best_single / denominator,
        "best_repeat_over_full": best_repeat / denominator,
        "mean_candidate_over_full": float(controls["mean_candidate"]["mean_recall"])
        / denominator,
        "reference_only_over_full": float(controls["reference_only"]["mean_recall"])
        / denominator,
    }


def _load_checkpoint_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[IAGSRME, object, object, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint has no reproducible backbone metadata")
    backbone_checkpoint = metadata.get("backbone_checkpoint")
    backbone_revision = metadata.get("backbone_revision")
    if not isinstance(backbone_checkpoint, str) or not isinstance(backbone_revision, str):
        raise ValueError("checkpoint backbone checkpoint/revision metadata is incomplete")
    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise ValueError("checkpoint has no model state")
    query_bank = state.get("core.intent_encoder.query_bank")
    if not isinstance(query_bank, Tensor):
        raise ValueError("checkpoint is not an IAG-SRME checkpoint")
    candidates, width = query_bank.shape
    if candidates != 4 or width != 256:
        raise ValueError("diagnostic runner supports canonical K=4, d=256 checkpoints")

    regime = FGCLIPRegime(
        checkpoint=backbone_checkpoint,
        revision=backbone_revision,
        train_vision=False,
        train_text=False,
        train_text_projection=False,
    )
    backbone = FGCLIPBackbone.from_pretrained(regime, internal_width=width)
    tokenizer, processor = FGCLIPBackbone.load_processor(
        regime.checkpoint, regime.revision, regime.trust_remote_code
    )
    enable_claim = any(key.startswith("core.claim_head.") for key in state)
    enable_factor = any(key.startswith("core.factor_fuser.") for key in state)
    core = IAGSRMECore(
        IAGSRMEConfig(
            width=width,
            num_candidates=candidates,
            max_steps=3,
            num_heads=8,
            retrieval_dim=backbone.retrieval_dim,
            lambda_z=0.10,
            query_cap=0.50,
            selector_temperature=1.0,
            selector_gumbel_noise=False,
            enable_claim_head=enable_claim,
            enable_factor_head=enable_factor,
        )
    )
    model = IAGSRME(backbone, core)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, tokenizer, processor, checkpoint


@torch.no_grad()
def _diagnose_category(
    model: IAGSRME,
    loader: DataLoader[ImageBatch],
    gallery: Tensor,
    gallery_ids: list[str],
    device: torch.device,
    category_accumulator: ValidationDiagnosticAccumulator,
    global_accumulator: ValidationDiagnosticAccumulator,
) -> dict[str, Any]:
    query_lists: defaultdict[str, list[Tensor]] = defaultdict(list)
    target_ids: list[str] = []
    reference_ids: list[str] = []
    counterfactual_queries: list[list[Tensor]] = [[] for _ in range(3)]
    counterfactual_targets: list[list[str]] = [[] for _ in range(3)]
    counterfactual_references: list[list[str]] = [[] for _ in range(3)]

    for cpu_batch in loader:
        batch = cpu_batch.to(device)
        if any(target is None for target in batch.target_ids):
            raise ValueError("validation target ID is missing")
        encoded = model.backbone(
            batch.reference_pixels,
            batch.input_ids,
            batch.attention_mask,
            batch.content_mask,
        )
        full = model.core(encoded, control="full")
        category_accumulator.update(full)
        global_accumulator.update(full)
        query_lists["full"].append(full.final_query.cpu())
        query_lists["reference_only"].append(encoded.reference_global.cpu())
        for candidate in range(4):
            single = single_candidate_control(full, candidate)
            query_lists[f"single_{candidate}"].append(single.query.cpu())
            repeated = repeat_candidate_control(model.core, encoded, candidate)
            query_lists[f"repeat_{candidate}"].append(repeated.final_query.cpu())
        mean_output = model.core(encoded, control="mean_candidate")
        query_lists["mean_candidate"].append(mean_output.final_query.cpu())

        batch_targets = [str(target) for target in batch.target_ids]
        target_ids.extend(batch_targets)
        reference_ids.extend(batch.reference_ids)
        for timestep in range(3):
            candidates, valid = same_parent_candidate_queries(full, timestep)
            counterfactual_queries[timestep].append(candidates.cpu())
            valid_cpu = valid.cpu().tolist()
            counterfactual_targets[timestep].extend(
                target for target, keep in zip(batch_targets, valid_cpu, strict=True) if keep
            )
            counterfactual_references[timestep].extend(
                reference
                for reference, keep in zip(batch.reference_ids, valid_cpu, strict=True)
                if keep
            )

    controls = {
        name: _retrieval_metrics(
            torch.cat(queries), gallery, target_ids, gallery_ids, reference_ids
        )
        for name, queries in query_lists.items()
    }
    controls["usefulness_ratios"] = _usefulness_ratios(controls)
    controls["counterfactual_same_parent_by_timestep"] = {}
    for timestep in range(3):
        if not counterfactual_queries[timestep]:
            continue
        queries = torch.cat(counterfactual_queries[timestep])
        if queries.shape[0] == 0:
            controls["counterfactual_same_parent_by_timestep"][f"t{timestep}"] = {
                "live_parent_count": 0
            }
            continue
        result = _counterfactual_retrieval_metrics(
            queries,
            gallery,
            counterfactual_targets[timestep],
            gallery_ids,
            counterfactual_references[timestep],
        )
        result["live_parent_count"] = queries.shape[0]
        controls["counterfactual_same_parent_by_timestep"][f"t{timestep}"] = result
    return controls


def _failure_flags(
    selection: Mapping[str, Any],
    grounding: Mapping[str, Any],
    functional: Mapping[str, Any],
    controls: Mapping[str, Any],
    specialization: Mapping[str, Any],
) -> dict[str, Any]:
    full = float(controls["full"]["mean_recall"])
    best_single = max(float(controls[f"single_{index}"]["mean_recall"]) for index in range(4))
    best_repeat = max(float(controls[f"repeat_{index}"]["mean_recall"]) for index in range(4))
    reference = float(controls["reference_only"]["mean_recall"])
    oracle_t0 = float(
        controls["counterfactual_same_parent_by_timestep"]["t0"]
        ["best_single_candidate_oracle"]["mean_recall"]
    )
    candidate_distribution = selection["candidate_distribution_conditional_on_edit"]
    delta_q_matrix = torch.tensor(specialization["pairwise_delta_q_cosine"])
    support_matrix = torch.tensor(specialization["pairwise_support_cosine"])
    ranks = [
        item.get("functional_effective_rank", 0.0)
        for item in functional["per_timestep"]
        if item.get("live_parent_count", 0) > 0
    ]
    mean_rank = sum(ranks) / max(len(ranks), 1)
    support_fraction = float(grounding["support_fraction"])
    margin = 2.0
    thresholds = {
        "stop_or_monopoly_fraction": 0.95,
        "clone_cosine": 0.95,
        "grounding_sparse_fraction": 0.02,
        "grounding_diffuse_fraction": 0.80,
        "functional_rank": 1.50,
        "retrieval_margin_points": margin,
    }
    supporting = {
        "stop_t0_occupancy": selection["absorbed_stop_occupancy_by_timestep"][0],
        "total_new_stops": sum(selection["counts"]["new_stops"]),
        "maximum_candidate_selection_share": max(candidate_distribution),
        "mean_delta_q_off_diagonal_cosine": _off_diagonal_mean(delta_q_matrix),
        "mean_support_off_diagonal_cosine": _off_diagonal_mean(support_matrix),
        "support_fraction": support_fraction,
        "mean_functional_effective_rank": mean_rank,
        "full_mean_recall": full,
        "best_single_mean_recall": best_single,
        "best_repeat_mean_recall": best_repeat,
        "reference_only_mean_recall": reference,
        "t0_candidate_oracle_mean_recall": oracle_t0,
    }
    flags = {
        "all_stop_t0": supporting["stop_t0_occupancy"] >= thresholds["stop_or_monopoly_fraction"],
        "never_stop": supporting["total_new_stops"] == 0,
        "single_candidate_monopoly": (
            supporting["maximum_candidate_selection_share"]
            >= thresholds["stop_or_monopoly_fraction"]
        ),
        "candidate_clone_effects": (
            supporting["mean_delta_q_off_diagonal_cosine"] >= thresholds["clone_cosine"]
        ),
        "grounding_clone": (
            supporting["mean_support_off_diagonal_cosine"] >= thresholds["clone_cosine"]
        ),
        "grounding_over_sparse": support_fraction <= thresholds["grounding_sparse_fraction"],
        "grounding_over_diffuse": support_fraction >= thresholds["grounding_diffuse_fraction"],
        "functional_rank_collapse": mean_rank <= thresholds["functional_rank"],
        "repeat_beats_full": best_repeat >= full + margin,
        "single_beats_full": best_single >= full + margin,
        "reference_dominates": reference >= full + margin,
        "selected_policy_underperforms_candidate_oracle": oracle_t0 >= full + margin,
    }
    return {"flags": flags, "supporting_numbers": supporting, "thresholds": thresholds}


def _validate_report_schema(report: Mapping[str, Any]) -> None:
    missing = REQUIRED_REPORT_KEYS - report.keys()
    if missing:
        raise AssertionError(f"diagnostic report is missing top-level keys: {sorted(missing)}")
    json.dumps(report, allow_nan=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose a trained IAG-SRME checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gallery-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--accelerator-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1 or args.gallery_batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch sizes must be positive and num-workers non-negative")
    return args


def main() -> None:
    args = _parse_args()
    seed_everything(args.seed, deterministic=True)
    configure_torch_runtime(deterministic=True, benchmark=False)
    device = resolve_device(args.device, args.accelerator_index)
    model, tokenizer, processor, checkpoint = _load_checkpoint_model(args.checkpoint, device)
    model.eval()

    annotation_root = args.dataset_root / "captions"
    split_root = args.dataset_root / "image_splits"
    image_store = DirectoryImageStore(args.dataset_root / "images")
    datasets = build_validation_datasets(
        annotation_root,
        CATEGORIES,
        CAPTION_POLICY,
        seed=args.seed,
    )
    collator = FashionIQImageCollator(
        image_store,
        tokenizer,
        processor,
        max_text_length=77,
        include_targets=False,
    )
    global_accumulator = ValidationDiagnosticAccumulator()
    category_results: dict[str, Any] = {}
    category_controls: dict[str, Any] = {}
    for category in CATEGORIES:
        dataset = datasets[category]
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=collator,
        )
        gallery_ids = build_fashioniq_gallery(
            PROTOCOL, split_root, category, dataset.annotations, SPLIT
        )
        gallery = encode_gallery(
            model,
            gallery_ids,
            image_store,
            processor,
            device,
            args.gallery_batch_size,
            args.num_workers,
        ).to(device)
        category_accumulator = ValidationDiagnosticAccumulator()
        controls = _diagnose_category(
            model,
            loader,
            gallery,
            gallery_ids,
            device,
            category_accumulator,
            global_accumulator,
        )
        category_controls[category] = controls
        category_selection = category_accumulator.selection_summary()
        category_grounding = category_accumulator.grounding_summary()
        category_functional = category_accumulator.functional_summary()
        category_dynamic = category_accumulator.dynamic_summary()
        category_specialization = category_accumulator.specialization_summary()
        category_results[category] = {
            "global_metrics": controls["full"],
            "selection_diagnostics": category_selection,
            "grounding_diagnostics": category_grounding,
            "functional_diagnostics": category_functional,
            "dynamic_diagnostics": category_dynamic,
            "control_retrieval_metrics": controls,
            "specialization_matrices": category_specialization,
            "failure_flags": _failure_flags(
                category_selection,
                category_grounding,
                category_functional,
                controls,
                category_specialization,
            ),
        }

    global_controls = _macro_control_results(category_controls)
    global_controls["usefulness_ratios"] = _usefulness_ratios(global_controls)
    selection = global_accumulator.selection_summary()
    grounding = global_accumulator.grounding_summary()
    functional = global_accumulator.functional_summary()
    dynamic = global_accumulator.dynamic_summary()
    specialization = global_accumulator.specialization_summary()
    metadata = checkpoint["metadata"]
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_metric": checkpoint.get("metric"),
        "backbone_metadata": {
            "checkpoint": metadata["backbone_checkpoint"],
            "revision": metadata["backbone_revision"],
            "training_precision": metadata.get("precision"),
        },
        "protocol": {
            "dataset": "FashionIQ",
            "split": SPLIT,
            "caption_policy": CAPTION_POLICY,
            "gallery_protocol": PROTOCOL,
            "reference_filtering": (
                "remove the query reference image from its gallery row unless it is the target"
            ),
            "deterministic": True,
            "seed": args.seed,
            "selector": "eval-mode hard argmax; no Gumbel noise",
            "architecture_assumptions": "canonical K=4, Tmax=3, d=256",
            "control_definitions": {
                "single_k": "execute same-parent candidate k once from Z0, then STOP",
                "repeat_k": "execute candidate k through the real recurrence for Tmax=3",
                "mean_candidate": (
                    "mean same-parent candidate token effect before readout using the "
                    "existing matched-compute control"
                ),
                "candidate_oracle": "offline target-ranked diagnostic only; never executed",
            },
        },
        "global_metrics": global_controls["full"],
        "per_category_metrics": category_results,
        "selection_diagnostics": selection,
        "grounding_diagnostics": grounding,
        "functional_diagnostics": functional,
        "dynamic_diagnostics": dynamic,
        "control_retrieval_metrics": global_controls,
        "specialization_matrices": specialization,
        "failure_flags": _failure_flags(
            selection, grounding, functional, global_controls, specialization
        ),
        "omitted_controls": {
            "all_candidate_sequential_0_1_2_3": (
                "omitted: canonical Tmax=3 cannot execute four sequential candidates "
                "without changing the model horizon"
            )
        },
    }
    _validate_report_schema(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps({"output": str(args.output), "global_metrics": report["global_metrics"]}))


if __name__ == "__main__":
    main()
