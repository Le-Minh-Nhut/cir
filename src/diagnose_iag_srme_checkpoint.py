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
from diagnostics.iag_srme import (
    FUNCTIONAL_ACTIVITY_EPSILON,
    flatten_delta_z,
    functional_effect_activity,
    functional_effective_rank,
    masked_pairwise_cosine,
    off_diagonal_values,
    pairwise_cosine_matrix,
    verify_same_parent_counterfactuals,
)
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
PROTOCOLS = ("fashioniq_original", "fashioniq_val")
SPLIT = "val"
CAPTION_POLICY = "ordered_and"
REQUIRED_REPORT_KEYS = {
    "checkpoint",
    "checkpoint_epoch",
    "checkpoint_metric",
    "checkpoint_model_config_provenance",
    "checkpoint_replay_guard",
    "backbone_metadata",
    "protocol",
    "global_metrics",
    "per_category_metrics",
    "intent_diagnostics",
    "selection_diagnostics",
    "grounding_diagnostics",
    "temporal_grounding_diagnostics",
    "visual_null_diagnostics",
    "functional_diagnostics",
    "dynamic_diagnostics",
    "control_retrieval_metrics",
    "same_parent_counterfactual_diagnostics",
    "selected_path_marginal_diagnostics",
    "specialization_matrices",
    "failure_flags",
    "diagnostic_definitions",
    "trusted_r1a_baseline",
}

TEMPORAL_SUPPORT_TOP_M = 10
TRUSTED_R1A_BASELINE = {
    "full_mean_recall": 38.764146,
    "mean_delta_q_norm_by_timestep": [0.336634, 0.272417, 0.197111],
    "delta_q_retention": {
        "t1_over_t0": 0.809238,
        "t2_over_t0": 0.585534,
        "t2_over_t1": 0.723562,
    },
    "selected_target_relative_gain_by_timestep": [0.07424, 0.02488, -0.00261],
    "pairwise_support_cosine": 0.999842,
    "pairwise_support_overlap": 0.995108,
    "repeated_candidate_trajectory_fraction": 0.957447,
    "mean_executed_edits": 2.86586,
}


def _off_diagonal_mean(matrix: Tensor) -> float:
    return float(off_diagonal_values(matrix).mean())


def _off_diagonal_summary(matrix: Tensor) -> dict[str, float]:
    values = off_diagonal_values(matrix)
    return {
        "mean": float(values.mean()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


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
class _MaskedMatrixStats:
    candidates: int
    total: Tensor | None = None
    valid_counts: Tensor | None = None
    observations: int = 0

    def update(self, values: Tensor, valid: Tensor) -> None:
        values = values.detach().float().cpu()
        valid = valid.detach().bool().cpu()
        expected = (values.shape[0], self.candidates, self.candidates)
        if values.shape != expected or valid.shape != expected:
            raise ValueError("masked pairwise matrices must be [B,K,K]")
        if values.shape[0] == 0:
            return
        batch_total = torch.where(valid, values, 0.0).sum(dim=0, dtype=torch.float64)
        batch_counts = valid.sum(dim=0, dtype=torch.long)
        self.total = batch_total if self.total is None else self.total + batch_total
        self.valid_counts = (
            batch_counts
            if self.valid_counts is None
            else self.valid_counts + batch_counts
        )
        self.observations += values.shape[0]

    def nullable_matrix(self) -> list[list[float | None]]:
        if self.total is None or self.valid_counts is None:
            return [[None] * self.candidates for _ in range(self.candidates)]
        means = self.total / self.valid_counts.clamp_min(1)
        return [
            [
                float(means[row, column])
                if int(self.valid_counts[row, column]) > 0
                else None
                for column in range(self.candidates)
            ]
            for row in range(self.candidates)
        ]

    def off_diagonal_summary(self) -> dict[str, float | int | None]:
        if self.total is None or self.valid_counts is None:
            valid_values = torch.empty(0, dtype=torch.float64)
            valid_pair_count = 0
            valid_pair_total = 0.0
        else:
            off_diagonal = ~torch.eye(self.candidates, dtype=torch.bool)
            valid_cells = self.valid_counts > 0
            selected = off_diagonal & valid_cells
            valid_values = (self.total / self.valid_counts.clamp_min(1))[selected]
            valid_pair_count = int(self.valid_counts[off_diagonal].sum())
            valid_pair_total = float(self.total[off_diagonal].sum())
        possible_pair_count = self.observations * self.candidates * (self.candidates - 1)
        return {
            "mean": (
                valid_pair_total / valid_pair_count if valid_pair_count > 0 else None
            ),
            "minimum": float(valid_values.min()) if valid_values.numel() else None,
            "maximum": float(valid_values.max()) if valid_values.numel() else None,
            "valid_pair_count": valid_pair_count,
            "possible_pair_count": possible_pair_count,
            "valid_pair_fraction": (
                valid_pair_count / possible_pair_count
                if possible_pair_count > 0
                else None
            ),
        }


@dataclass
class _EffectActivityStats:
    candidates: int
    active_candidates: int = 0
    total_candidates: int = 0
    dead_parents: int = 0
    parents: int = 0

    def update(self, active: Tensor) -> None:
        active = active.detach().bool().cpu()
        if active.ndim != 2 or active.shape[1] != self.candidates:
            raise ValueError("functional activity mask must be [B,K]")
        self.active_candidates += int(active.sum())
        self.total_candidates += active.numel()
        self.dead_parents += int((~active.any(dim=-1)).sum())
        self.parents += active.shape[0]

    def summary(self) -> dict[str, float | int | None]:
        if self.parents == 0:
            return {
                "active_candidate_fraction": None,
                "dead_candidate_fraction": None,
                "dead_parent_fraction": None,
                "active_candidate_count": 0,
                "total_candidate_count": 0,
                "dead_parent_count": 0,
                "parent_count": 0,
            }
        return {
            "active_candidate_fraction": self.active_candidates
            / self.total_candidates,
            "dead_candidate_fraction": 1.0
            - self.active_candidates / self.total_candidates,
            "dead_parent_fraction": self.dead_parents / self.parents,
            "active_candidate_count": self.active_candidates,
            "total_candidate_count": self.total_candidates,
            "dead_parent_count": self.dead_parents,
            "parent_count": self.parents,
        }


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

    def summary(
        self, *, parent_count: int, population: str
    ) -> dict[str, float | int | str]:
        return {
            "mean_absolute_change": self.total / max(self.count, 1),
            "max_absolute_change": self.maximum,
            "element_count": self.count,
            "live_executed_parent_count": parent_count,
            "metric_population": population,
        }


@dataclass
class _DistributionStats:
    chunks: list[Tensor]

    def __init__(self) -> None:
        self.chunks = []

    def update(self, values: Tensor) -> None:
        values = values.detach().float().flatten().cpu()
        if values.numel() > 0:
            self.chunks.append(values)

    def summary(self) -> dict[str, float | int | None]:
        if not self.chunks:
            return {
                "count": 0,
                "mean": None,
                "median": None,
                "standard_deviation": None,
                "p10": None,
                "p25": None,
                "p75": None,
                "p90": None,
                "p95": None,
                "minimum": None,
                "maximum": None,
            }
        values = torch.cat(self.chunks)
        return {
            "count": values.numel(),
            "mean": float(values.mean()),
            "median": float(values.median()),
            "standard_deviation": float(values.std(unbiased=False)),
            "p10": float(torch.quantile(values, 0.10)),
            "p25": float(torch.quantile(values, 0.25)),
            "p75": float(torch.quantile(values, 0.75)),
            "p90": float(torch.quantile(values, 0.90)),
            "p95": float(torch.quantile(values, 0.95)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }

    def values(self) -> Tensor:
        return torch.cat(self.chunks) if self.chunks else torch.empty(0)


NULL_EFFECT_BIN_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.000001)


class _NullEffectBinAccumulator:
    def __init__(self) -> None:
        bins = len(NULL_EFFECT_BIN_EDGES) - 1
        self.count = torch.zeros(bins, dtype=torch.long)
        self.delta_z_total = torch.zeros(bins, dtype=torch.float64)
        self.delta_q_total = torch.zeros(bins, dtype=torch.float64)
        self.selected = torch.zeros(bins, dtype=torch.long)

    def update(
        self,
        null_probabilities: Tensor,
        delta_z_norms: Tensor,
        delta_q_norms: Tensor,
        selected: Tensor,
    ) -> None:
        values = null_probabilities.detach().float().cpu().flatten()
        delta_z = delta_z_norms.detach().float().cpu().flatten()
        delta_q = delta_q_norms.detach().float().cpu().flatten()
        selected = selected.detach().bool().cpu().flatten()
        if not (values.shape == delta_z.shape == delta_q.shape == selected.shape):
            raise ValueError("NULL/effect/selection bin tensors must have equal shape")
        for index, (lower, upper) in enumerate(
            zip(NULL_EFFECT_BIN_EDGES[:-1], NULL_EFFECT_BIN_EDGES[1:], strict=True)
        ):
            mask = values.ge(lower) & values.lt(upper)
            self.count[index] += int(mask.sum())
            self.delta_z_total[index] += float(delta_z[mask].sum())
            self.delta_q_total[index] += float(delta_q[mask].sum())
            self.selected[index] += int(selected[mask].sum())

    def summary(self) -> list[dict[str, float | int | None | list[float]]]:
        result = []
        for index, (lower, upper) in enumerate(
            zip(NULL_EFFECT_BIN_EDGES[:-1], NULL_EFFECT_BIN_EDGES[1:], strict=True)
        ):
            count = int(self.count[index])
            result.append(
                {
                    "range": [lower, min(upper, 1.0)],
                    "count": count,
                    "mean_delta_z_norm": (
                        float(self.delta_z_total[index] / count) if count else None
                    ),
                    "mean_delta_q_norm": (
                        float(self.delta_q_total[index] / count) if count else None
                    ),
                    "candidate_selection_probability": (
                        float(self.selected[index] / count) if count else None
                    ),
                    "selected_count": int(self.selected[index]),
                }
            )
        return result


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
    verify_same_parent_counterfactuals(step)
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


class TemporalGroundingAccumulator:
    """Lineage-safe dynamic-WHERE measurements; never feeds model execution."""

    def __init__(self, candidates: int = 4, timesteps: int = 3) -> None:
        self.candidates = candidates
        self.timesteps = timesteps
        self.dynamic_regrounding: bool | None = None
        self.live_counts = torch.zeros(timesteps, dtype=torch.long)
        self.mass = [_DistributionStats() for _ in range(timesteps)]
        self.entropy = [_DistributionStats() for _ in range(timesteps)]
        self.effective_size = [_DistributionStats() for _ in range(timesteps)]
        self.support_fraction = [_DistributionStats() for _ in range(timesteps)]
        self.between_candidate_cosine = [
            _MaskedMatrixStats(candidates) for _ in range(timesteps)
        ]
        self.between_candidate_overlap = [
            _MaskedMatrixStats(candidates) for _ in range(timesteps)
        ]
        transitions = timesteps - 1
        names = (
            "cosine",
            "overlap",
            "top_m_jaccard",
            "entropy_change",
            "effective_size_change",
            "l1_change",
            "l2_change",
        )
        self.transition = {
            name: [_DistributionStats() for _ in range(transitions)] for name in names
        }
        self.conditioned = {
            condition: {
                name: [_DistributionStats() for _ in range(transitions)]
                for name in ("cosine", "overlap", "l1_change")
            }
            for condition in ("same_candidate_executed", "other_candidate_executed", "stop")
        }
        self.argmax_changed = torch.zeros(transitions, dtype=torch.long)
        self.argmax_total = torch.zeros(transitions, dtype=torch.long)
        self.transition_live_parent_counts = torch.zeros(transitions, dtype=torch.long)
        self.displacement_alignment = [
            _MaskedMatrixStats(candidates) for _ in range(transitions)
        ]

    @staticmethod
    def _entropy(supports: Tensor) -> Tensor:
        values = supports.float()
        return -(values * values.clamp_min(1e-8).log()).sum(dim=-1)

    @staticmethod
    def _masked_update(stats: _DistributionStats, values: Tensor, mask: Tensor) -> None:
        stats.update(values[mask])

    def update(self, output: IAGSRMEOutput) -> None:
        if output.temporal_supports is None:
            supports = torch.stack(
                [step.spatial_supports for step in output.trace], dim=1
            )
        else:
            supports = output.temporal_supports
        if supports.ndim != 4:
            raise ValueError("temporal supports must be [B,T,K,N]")
        batch, timesteps, candidates, tokens = supports.shape
        if (timesteps, candidates) != (self.timesteps, self.candidates):
            raise ValueError("temporal support K/T mismatch")
        if self.dynamic_regrounding is None:
            self.dynamic_regrounding = output.dynamic_regrounding
        elif self.dynamic_regrounding != output.dynamic_regrounding:
            raise ValueError("mixed static/dynamic grounding outputs in one accumulator")

        supports = supports.float()
        for timestep, step in enumerate(output.trace):
            if step.spatial_supports is None:
                raise AssertionError("trace is missing the support used by this step")
            torch.testing.assert_close(
                supports[:, timestep], step.spatial_supports.float(), atol=0.0, rtol=0.0
            )
            valid = step.live_before
            self.live_counts[timestep] += int(valid.sum())
            if not valid.any():
                continue
            current = supports[valid, timestep]
            mass = current.sum(dim=-1)
            entropy = self._entropy(current)
            self.mass[timestep].update(mass)
            self.entropy[timestep].update(entropy)
            self.effective_size[timestep].update(entropy.exp())
            self.support_fraction[timestep].update((current > 0).float().mean(dim=-1))
            valid_pairs = torch.ones(
                current.shape[:2] + (candidates,), dtype=torch.bool, device=current.device
            )
            self.between_candidate_cosine[timestep].update(
                pairwise_cosine_matrix(current), valid_pairs
            )
            overlap = torch.minimum(
                current[:, :, None], current[:, None, :]
            ).sum(dim=-1)
            self.between_candidate_overlap[timestep].update(overlap, valid_pairs)

        for transition in range(self.timesteps - 1):
            previous_step = output.trace[transition]
            current_step = output.trace[transition + 1]
            before = supports[:, transition]
            after = supports[:, transition + 1]
            valid_live = current_step.live_before
            self.transition_live_parent_counts[transition] += int(valid_live.sum())
            cosine = F.cosine_similarity(before, after, dim=-1)
            overlap = torch.minimum(before, after).sum(dim=-1)
            l1 = (after - before).abs().sum(dim=-1)
            l2 = (after - before).norm(dim=-1)
            entropy_before = self._entropy(before)
            entropy_after = self._entropy(after)
            effective_before = entropy_before.exp()
            effective_after = entropy_after.exp()
            top_m = min(TEMPORAL_SUPPORT_TOP_M, tokens)
            top_before = before.topk(top_m, dim=-1).indices
            top_after = after.topk(top_m, dim=-1).indices
            intersection = (
                top_before[..., :, None] == top_after[..., None, :]
            ).any(dim=-1).sum(dim=-1)
            jaccard = intersection.float() / (2 * top_m - intersection).clamp_min(1)
            transition_values = {
                "cosine": cosine,
                "overlap": overlap,
                "top_m_jaccard": jaccard,
                "entropy_change": entropy_after - entropy_before,
                "effective_size_change": effective_after - effective_before,
                "l1_change": l1,
                "l2_change": l2,
            }
            for name, values in transition_values.items():
                self._masked_update(
                    self.transition[name][transition],
                    values,
                    valid_live[:, None].expand_as(values),
                )
            self.argmax_changed[transition] += int(
                (before.argmax(dim=-1)[valid_live] != after.argmax(dim=-1)[valid_live]).sum()
            )
            self.argmax_total[transition] += int(valid_live.sum()) * candidates

            displacement = after[valid_live] - before[valid_live]
            if displacement.shape[0]:
                matrix, pair_valid = masked_pairwise_cosine(displacement)
                self.displacement_alignment[transition].update(matrix, pair_valid)

            action = previous_step.selected_index
            candidate_ids = torch.arange(candidates, device=action.device)[None]
            same = previous_step.live_before[:, None] & action[:, None].eq(candidate_ids)
            other = (
                previous_step.live_before[:, None]
                & action[:, None].lt(candidates)
                & ~action[:, None].eq(candidate_ids)
            )
            stopped = previous_step.live_before[:, None] & action[:, None].eq(candidates)
            for condition, mask in (
                ("same_candidate_executed", same),
                ("other_candidate_executed", other),
                ("stop", stopped.expand(batch, candidates)),
            ):
                for name, values in (
                    ("cosine", cosine),
                    ("overlap", overlap),
                    ("l1_change", l1),
                ):
                    self._masked_update(self.conditioned[condition][name][transition], values, mask)

    def summary(self) -> dict[str, Any]:
        per_timestep = []
        for timestep in range(self.timesteps):
            per_timestep.append(
                {
                    "timestep": timestep,
                    "live_parent_count": int(self.live_counts[timestep]),
                    "metric_population": f"samples live before decision at t={timestep}",
                    "support_mass": self.mass[timestep].summary(),
                    "support_entropy": self.entropy[timestep].summary(),
                    "support_effective_size": self.effective_size[timestep].summary(),
                    "support_fraction": self.support_fraction[timestep].summary(),
                    "between_candidate_support_cosine_matrix": (
                        self.between_candidate_cosine[timestep].nullable_matrix()
                    ),
                    "between_candidate_support_cosine_off_diagonal": (
                        self.between_candidate_cosine[timestep].off_diagonal_summary()
                    ),
                    "between_candidate_support_overlap_matrix": (
                        self.between_candidate_overlap[timestep].nullable_matrix()
                    ),
                    "between_candidate_support_overlap_off_diagonal": (
                        self.between_candidate_overlap[timestep].off_diagonal_summary()
                    ),
                }
            )
        transitions = []
        for transition in range(self.timesteps - 1):
            transitions.append(
                {
                    "transition": f"t{transition}_to_t{transition + 1}",
                    "live_parent_count": int(self.transition_live_parent_counts[transition]),
                    "metric_population": (
                        f"same-candidate supports for samples live at t={transition + 1}"
                    ),
                    "same_candidate_temporal_cosine": self.transition["cosine"][transition].summary(),
                    "same_candidate_temporal_overlap": self.transition["overlap"][transition].summary(),
                    "top_m_jaccard": {
                        "m": TEMPORAL_SUPPORT_TOP_M,
                        **self.transition["top_m_jaccard"][transition].summary(),
                    },
                    "entropy_change": self.transition["entropy_change"][transition].summary(),
                    "effective_size_change": self.transition["effective_size_change"][transition].summary(),
                    "support_l1_change": self.transition["l1_change"][transition].summary(),
                    "support_l2_change": self.transition["l2_change"][transition].summary(),
                    "argmax_token_changed_fraction": (
                        int(self.argmax_changed[transition])
                        / int(self.argmax_total[transition])
                        if int(self.argmax_total[transition]) > 0
                        else None
                    ),
                    "candidate_displacement_cosine_matrix": (
                        self.displacement_alignment[transition].nullable_matrix()
                    ),
                    "candidate_displacement_cosine_off_diagonal": (
                        self.displacement_alignment[transition].off_diagonal_summary()
                    ),
                    "conditioned_on_previous_decision": {
                        condition: {
                            name: stats[transition].summary()
                            for name, stats in metrics.items()
                        }
                        for condition, metrics in self.conditioned.items()
                    },
                }
            )
        return {
            "enable_dynamic_regrounding": bool(self.dynamic_regrounding),
            "support_source": (
                "Ground(I_k, Z_t) recomputed before every recurrent decision"
                if self.dynamic_regrounding
                else "Ground(I_k, A) computed once and reused"
            ),
            "top_m_jaccard_m": TEMPORAL_SUPPORT_TOP_M,
            "per_timestep": per_timestep,
            "per_transition": transitions,
            "interpretation_limit": (
                "support motion alone does not establish semantic residual behavior; "
                "inspect displacement alignment and functional/retrieval consequences"
            ),
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
        self.temporal_grounding = TemporalGroundingAccumulator(candidates, timesteps)

        self.support_fraction = [_DistributionStats() for _ in range(candidates)]
        self.support_entropy = [_DistributionStats() for _ in range(candidates)]
        self.support_effective_size = [
            _DistributionStats() for _ in range(candidates)
        ]
        self.support_cosine = _MaskedMatrixStats(candidates)
        self.support_overlap = _MaskedMatrixStats(candidates)
        self.real_visual_mass = _MeanTensor()
        self.real_visual_mass_distribution = _DistributionStats()
        self.dominant_grounding_share = _MeanTensor()
        self.intent_cosine = _MeanTensor()
        self.intent_norm = _MeanTensor()
        self.visual_null_seen = False
        self.null_probability = _DistributionStats()
        self.visual_confidence_distribution = _DistributionStats()
        self.null_probability_by_candidate = [
            _DistributionStats() for _ in range(candidates)
        ]
        self.null_probability_by_timestep = [
            _DistributionStats() for _ in range(timesteps)
        ]
        self.confidence_by_timestep = [
            _DistributionStats() for _ in range(timesteps)
        ]
        self.null_probability_by_candidate_timestep = [
            [_DistributionStats() for _ in range(candidates)]
            for _ in range(timesteps)
        ]
        self.selected_null_probability = _DistributionStats()
        self.non_selected_null_probability = _DistributionStats()
        self.selected_null_by_timestep = [
            _DistributionStats() for _ in range(timesteps)
        ]
        self.non_selected_null_by_timestep = [
            _DistributionStats() for _ in range(timesteps)
        ]
        self.temporal_confidence_change = [
            _DistributionStats() for _ in range(timesteps - 1)
        ]
        self.temporal_confidence_absolute_change = [
            _DistributionStats() for _ in range(timesteps - 1)
        ]
        self.executed_confidence_before = _DistributionStats()
        self.executed_confidence_after = _DistributionStats()
        self.executed_confidence_change = _DistributionStats()
        self.repeated_confidence_change = _DistributionStats()
        self.null_effect_bins = _NullEffectBinAccumulator()
        self.null_effect_bins_by_timestep = [
            _NullEffectBinAccumulator() for _ in range(timesteps)
        ]
        self.stop_mean_confidence = _DistributionStats()
        self.edit_mean_confidence = _DistributionStats()
        self.stop_max_confidence = _DistributionStats()
        self.edit_max_confidence = _DistributionStats()
        self.all_candidates_high_null_count = 0
        self.all_candidates_high_null_stop_count = 0

        self.delta_z_norm = [_MeanTensor() for _ in range(timesteps)]
        self.delta_q_norm = [_MeanTensor() for _ in range(timesteps)]
        self.delta_q_norm_distribution = [
            _DistributionStats() for _ in range(timesteps)
        ]
        self.delta_z_rank = [_MeanTensor() for _ in range(timesteps)]
        self.delta_q_rank = [_MeanTensor() for _ in range(timesteps)]
        self.delta_z_activity = [
            _EffectActivityStats(candidates) for _ in range(timesteps)
        ]
        self.delta_q_activity = [
            _EffectActivityStats(candidates) for _ in range(timesteps)
        ]
        self.context_cosine = [_MeanTensor() for _ in range(timesteps)]
        self.delta_z_cosine = [
            _MaskedMatrixStats(candidates) for _ in range(timesteps)
        ]
        self.delta_q_cosine = [
            _MaskedMatrixStats(candidates) for _ in range(timesteps)
        ]
        self.context_cosine_all = _MeanTensor()
        self.delta_z_cosine_all = _MaskedMatrixStats(candidates)
        self.delta_q_cosine_all = _MaskedMatrixStats(candidates)

        names = ("g_t", "d_t", "context", "delta_z", "candidate_query", "scores")
        self.dynamic = {
            name: [_ScalarStats() for _ in range(timesteps - 1)] for name in names
        }
        self.dynamic_parent_counts = torch.zeros(timesteps - 1, dtype=torch.long)

    def update(self, output: IAGSRMEOutput) -> None:
        batch_size, candidates, tokens = output.supports.shape
        if candidates != self.candidates or len(output.trace) != self.timesteps:
            raise ValueError("diagnostic accumulator/model K or Tmax mismatch")
        self.queries += batch_size
        self.visual_tokens = tokens
        self.temporal_grounding.update(output)

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
        support_mass = supports.sum(dim=-1)
        self.real_visual_mass.update(support_mass)
        self.real_visual_mass_distribution.update(support_mass)
        conditional_supports = (
            output.conditional_supports.float()
            if output.conditional_supports is not None
            else supports / support_mass[..., None].clamp_min(1e-8)
        )
        entropy = -(
            conditional_supports * conditional_supports.clamp_min(1e-8).log()
        ).sum(dim=-1)
        valid_shape = support_mass > 1e-8
        fraction = (conditional_supports > 0).float().mean(dim=-1)
        for candidate in range(candidates):
            valid_candidate = valid_shape[:, candidate]
            self.support_fraction[candidate].update(
                fraction[valid_candidate, candidate]
            )
            self.support_entropy[candidate].update(
                entropy[valid_candidate, candidate]
            )
            self.support_effective_size[candidate].update(
                entropy[valid_candidate, candidate].exp()
            )
        shape_pair_valid = valid_shape[:, :, None] & valid_shape[:, None, :]
        self.support_cosine.update(
            pairwise_cosine_matrix(conditional_supports), shape_pair_valid
        )
        overlap = torch.minimum(
            conditional_supports[:, :, None], conditional_supports[:, None, :]
        ).sum(dim=-1)
        self.support_overlap.update(overlap, shape_pair_valid)
        dominant_share = conditional_supports.max(dim=1).values.sum(
            dim=-1
        ) / conditional_supports.sum(
            dim=(1, 2)
        ).clamp_min(1e-8)
        self.dominant_grounding_share.update(dominant_share[:, None])
        self.intent_cosine.update(pairwise_cosine_matrix(output.intents))
        self.intent_norm.update(output.intents.float().norm(dim=-1))

        for timestep, step in enumerate(output.trace):
            if step.visual_null_probability is None or step.visual_confidence is None:
                continue
            self.visual_null_seen = True
            valid = step.live_before
            if not valid.any():
                continue
            live_null = step.visual_null_probability[valid].float()
            live_confidence = step.visual_confidence[valid].float()
            torch.testing.assert_close(
                live_null + live_confidence,
                torch.ones_like(live_null),
                atol=1e-6,
                rtol=1e-6,
            )
            self.null_probability.update(live_null)
            self.visual_confidence_distribution.update(live_confidence)
            self.null_probability_by_timestep[timestep].update(live_null)
            self.confidence_by_timestep[timestep].update(live_confidence)
            for candidate in range(candidates):
                self.null_probability_by_candidate[candidate].update(
                    live_null[:, candidate]
                )
                self.null_probability_by_candidate_timestep[timestep][candidate].update(
                    live_null[:, candidate]
                )

            selected_indices = step.selected_index[valid]
            selected_mask = torch.zeros_like(live_null, dtype=torch.bool)
            edit_selected = selected_indices.lt(candidates)
            if edit_selected.any():
                rows = torch.arange(live_null.shape[0], device=live_null.device)[
                    edit_selected
                ]
                columns = selected_indices[edit_selected]
                selected_mask[rows, columns] = True
                selected_values = live_null[rows, columns]
                self.selected_null_probability.update(selected_values)
                self.selected_null_by_timestep[timestep].update(selected_values)
            non_selected_values = live_null[~selected_mask]
            self.non_selected_null_probability.update(non_selected_values)
            self.non_selected_null_by_timestep[timestep].update(non_selected_values)
            delta_z_norm = flatten_delta_z(step.delta_z[valid]).norm(dim=-1)
            delta_q_norm = step.delta_q[valid].float().norm(dim=-1)
            self.null_effect_bins.update(
                live_null, delta_z_norm, delta_q_norm, selected_mask
            )
            self.null_effect_bins_by_timestep[timestep].update(
                live_null, delta_z_norm, delta_q_norm, selected_mask
            )

            stopped = selected_indices.eq(candidates)
            mean_confidence = live_confidence.mean(dim=-1)
            max_confidence = live_confidence.max(dim=-1).values
            self.stop_mean_confidence.update(mean_confidence[stopped])
            self.edit_mean_confidence.update(mean_confidence[~stopped])
            self.stop_max_confidence.update(max_confidence[stopped])
            self.edit_max_confidence.update(max_confidence[~stopped])
            all_high_null = live_null.gt(0.8).all(dim=-1)
            self.all_candidates_high_null_count += int(all_high_null.sum())
            self.all_candidates_high_null_stop_count += int(
                (all_high_null & stopped).sum()
            )

        for transition, (previous, current_step) in enumerate(
            zip(output.trace[:-1], output.trace[1:], strict=True)
        ):
            if (
                previous.visual_confidence is None
                or current_step.visual_confidence is None
            ):
                continue
            valid = current_step.live_before
            if not valid.any():
                continue
            confidence_change = (
                current_step.visual_confidence[valid].float()
                - previous.visual_confidence[valid].float()
            )
            self.temporal_confidence_change[transition].update(confidence_change)
            self.temporal_confidence_absolute_change[transition].update(
                confidence_change.abs()
            )
            previous_selected = previous.selected_index[valid]
            executed = previous_selected.lt(candidates)
            if executed.any():
                rows = torch.arange(int(valid.sum()), device=valid.device)[executed]
                columns = previous_selected[executed]
                before = previous.visual_confidence[valid][rows, columns].float()
                after = current_step.visual_confidence[valid][rows, columns].float()
                self.executed_confidence_before.update(before)
                self.executed_confidence_after.update(after)
                self.executed_confidence_change.update(after - before)
                repeated = current_step.selected_index[valid][executed].eq(columns)
                self.repeated_confidence_change.update((after - before)[repeated])

        for timestep, step in enumerate(output.trace):
            verify_same_parent_counterfactuals(step)
            valid = step.live_before
            if not valid.any():
                continue
            delta_z = step.delta_z[valid].float()
            delta_q = step.delta_q[valid].float()
            contexts = step.contexts[valid].float()
            flat_delta_z = flatten_delta_z(delta_z)
            delta_q_norm = delta_q.norm(dim=-1)
            self.delta_z_norm[timestep].update(flat_delta_z.norm(dim=-1))
            self.delta_q_norm[timestep].update(delta_q_norm)
            self.delta_q_norm_distribution[timestep].update(delta_q_norm)
            self.delta_z_rank[timestep].update(
                functional_effective_rank(flat_delta_z)[:, None]
            )
            self.delta_q_rank[timestep].update(
                functional_effective_rank(delta_q)[:, None]
            )
            _, delta_z_active = functional_effect_activity(flat_delta_z)
            _, delta_q_active = functional_effect_activity(delta_q)
            self.delta_z_activity[timestep].update(delta_z_active)
            self.delta_q_activity[timestep].update(delta_q_active)
            context_matrix = pairwise_cosine_matrix(contexts)
            delta_z_matrix, delta_z_valid = masked_pairwise_cosine(flat_delta_z)
            delta_q_matrix, delta_q_valid = masked_pairwise_cosine(delta_q)
            self.context_cosine[timestep].update(context_matrix)
            self.delta_z_cosine[timestep].update(delta_z_matrix, delta_z_valid)
            self.delta_q_cosine[timestep].update(delta_q_matrix, delta_q_valid)
            self.context_cosine_all.update(context_matrix)
            self.delta_z_cosine_all.update(delta_z_matrix, delta_z_valid)
            self.delta_q_cosine_all.update(delta_q_matrix, delta_q_valid)

        for transition, (previous, current) in enumerate(
            zip(output.trace[:-1], output.trace[1:], strict=True)
        ):
            expected_live = previous.live_before & ~previous.stopped_now
            if not torch.equal(current.live_before, expected_live):
                raise AssertionError("recurrent live lineage is inconsistent after STOP")
            valid = previous.live_before & previous.selected_index.lt(candidates)
            if not valid.any():
                continue
            self.dynamic_parent_counts[transition] += int(valid.sum())
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
            "metric_populations": {
                "candidate_distribution_conditional_on_edit": (
                    "selected non-STOP decisions across all live timesteps"
                ),
                "live_action_distribution_candidates_plus_stop": (
                    "all decisions among live parents before each timestep"
                ),
                "absorbed_stop_occupancy_by_timestep": (
                    "all validation trajectories, including already absorbed STOP"
                ),
                "new_stop_hazard_by_timestep": (
                    "live parents before the decision at each timestep"
                ),
            },
            "counts": {
                "queries": self.queries,
                "candidate_selections": self.candidate_counts.tolist(),
                "live_actions_candidates_plus_stop": self.live_action_counts.tolist(),
                "absorbed_stop_occupancy": self.stop_occupancy.tolist(),
                "new_stops": self.new_stop_counts.tolist(),
                "live_parents": self.live_counts.tolist(),
                "executed_edits": self.executed_edits,
                "queries_with_repeated_candidate_selections": self.repeated_queries,
            },
        }

    def intent_summary(self) -> dict[str, Any]:
        cosine = self.intent_cosine.mean()
        norms = self.intent_norm.mean()
        return {
            "pairwise_intent_cosine_matrix": cosine.tolist(),
            "pairwise_intent_cosine_off_diagonal": _off_diagonal_summary(cosine),
            "per_candidate_intent_norm": norms.tolist(),
            "metric_population": "all validation queries; intents are computed once per query",
            "tensor_shape_before_reduction": "[B,K,d]",
        }

    def grounding_summary(self) -> dict[str, Any]:
        def combined_values(items: list[_DistributionStats]) -> Tensor:
            chunks = [stats.values() for stats in items]
            nonempty = [values for values in chunks if values.numel()]
            return torch.cat(nonempty) if nonempty else torch.empty(0)

        fraction_summaries = [stats.summary() for stats in self.support_fraction]
        entropy_summaries = [stats.summary() for stats in self.support_entropy]
        effective_summaries = [
            stats.summary() for stats in self.support_effective_size
        ]
        all_fraction = combined_values(self.support_fraction)
        all_entropy = combined_values(self.support_entropy)
        all_effective = combined_values(self.support_effective_size)
        cosine = self.support_cosine.nullable_matrix()
        overlap = self.support_overlap.nullable_matrix()
        cosine_summary = self.support_cosine.off_diagonal_summary()
        overlap_summary = self.support_overlap.off_diagonal_summary()
        visual_mass = self.real_visual_mass.mean()
        return {
            "support_recomputed_each_timestep": bool(output_dynamic := self.temporal_grounding.dynamic_regrounding),
            "support_static_by_current_architecture": not bool(output_dynamic),
            "empirical_recurrence_stability_measured": bool(output_dynamic),
            "visual_token_count": self.visual_tokens,
            "spatial_support_mass": float(visual_mass.mean()),
            "per_candidate_spatial_support_mass": visual_mass.tolist(),
            "real_visual_support_mass": float(visual_mass.mean()),
            "per_candidate_real_visual_support_mass": visual_mass.tolist(),
            "real_visual_support_mass_distribution": (
                self.real_visual_mass_distribution.summary()
            ),
            "conditional_shape_valid_candidate_fraction": (
                all_fraction.numel()
                / max(self.queries * self.candidates, 1)
            ),
            "support_fraction": (
                float(all_fraction.mean()) if all_fraction.numel() else None
            ),
            "support_entropy": (
                float(all_entropy.mean()) if all_entropy.numel() else None
            ),
            "conditional_support_entropy": (
                float(all_entropy.mean()) if all_entropy.numel() else None
            ),
            "support_effective_size": (
                float(all_effective.mean()) if all_effective.numel() else None
            ),
            "conditional_support_effective_size": (
                float(all_effective.mean()) if all_effective.numel() else None
            ),
            "per_candidate_support_fraction": [
                item["mean"] for item in fraction_summaries
            ],
            "per_candidate_support_entropy": [
                item["mean"] for item in entropy_summaries
            ],
            "per_candidate_support_effective_size": [
                item["mean"] for item in effective_summaries
            ],
            "pairwise_support_cosine_mean_off_diagonal": cosine_summary["mean"],
            "pairwise_support_overlap_mean_off_diagonal": overlap_summary["mean"],
            "pairwise_support_cosine_off_diagonal": cosine_summary,
            "pairwise_support_probability_overlap_off_diagonal": overlap_summary,
            "pairwise_support_cosine_matrix": cosine,
            "pairwise_support_probability_overlap_matrix": overlap,
            "support_shape_normalization": (
                "corrected R1b spatial Entmax support already has unit mass over real "
                "tokens; this diagnostic normalization is an identity up to epsilon"
            ),
            "dominant_tokenwise_grounding_mass_share": float(
                self.dominant_grounding_share.mean().mean()
            ),
            "dominant_share_definition": (
                "sum_n max_k P[k,n] divided by total candidate support mass; "
                "diagnostic only, not semantic ownership"
            ),
            "metric_population": (
                "all validation queries; backward-compatible t0 support map per query/candidate; "
                "use temporal_grounding_diagnostics for R1c1 timestep populations"
            ),
        }

    def temporal_grounding_summary(self) -> dict[str, Any]:
        return self.temporal_grounding.summary()

    def visual_null_summary(self) -> dict[str, Any]:
        if not self.visual_null_seen:
            return {
                "enabled": False,
                "architecture_generation": (
                    "r1c1_dynamic_current_state_reground_v1"
                    if self.temporal_grounding.dynamic_regrounding
                    else "legacy_r0_or_r1a"
                ),
                "metric_population": "not applicable: checkpoint has no Visual NULL",
            }
        null_values = self.null_probability.values()
        thresholds = (0.10, 0.25, 0.50, 0.80)
        overall_bins = self.null_effect_bins.summary()
        nonempty_bins = [item for item in overall_bins if item["count"]]
        low_bin_delta = (
            nonempty_bins[0]["mean_delta_z_norm"] if nonempty_bins else None
        )
        high_bin_delta = (
            nonempty_bins[-1]["mean_delta_z_norm"] if nonempty_bins else None
        )
        mean_null = float(null_values.mean())
        candidate_means = [
            stats.summary()["mean"] for stats in self.null_probability_by_candidate
        ]
        return {
            "enabled": True,
            "architecture_generation": "r1b_dynamic_applicability_gate_v2",
            "static_by_architecture": False,
            "dynamic_whether_by_current_context": True,
            "support_recomputed_each_timestep": False,
            "null_probability": self.null_probability.summary(),
            "fraction_above_threshold": {
                f"p_null_gt_{threshold:.2f}": float(
                    null_values.gt(threshold).float().mean()
                )
                for threshold in thresholds
            },
            "per_candidate_null_probability": [
                stats.summary() for stats in self.null_probability_by_candidate
            ],
            "null_probability_by_timestep": [
                {"timestep": timestep, **stats.summary()}
                for timestep, stats in enumerate(self.null_probability_by_timestep)
            ],
            "confidence_by_timestep": [
                {"timestep": timestep, **stats.summary()}
                for timestep, stats in enumerate(self.confidence_by_timestep)
            ],
            "null_probability_by_candidate_and_timestep": [
                {
                    "timestep": timestep,
                    "candidates": [stats.summary() for stats in candidate_stats],
                }
                for timestep, candidate_stats in enumerate(
                    self.null_probability_by_candidate_timestep
                )
            ],
            "visual_confidence": self.visual_confidence_distribution.summary(),
            "selected_candidate_null_probability": (
                self.selected_null_probability.summary()
            ),
            "non_selected_candidate_null_probability": (
                self.non_selected_null_probability.summary()
            ),
            "selected_candidate_null_probability_by_timestep": [
                {
                    "timestep": timestep,
                    **stats.summary(),
                    "population": (
                        "selected non-STOP candidates among live decisions at this dynamic "
                        "current-state context"
                    ),
                }
                for timestep, stats in enumerate(self.selected_null_by_timestep)
            ],
            "non_selected_candidate_null_probability_by_timestep": [
                {
                    "timestep": timestep,
                    **stats.summary(),
                    "population": "non-selected candidates of live parents",
                }
                for timestep, stats in enumerate(self.non_selected_null_by_timestep)
            ],
            "temporal_applicability": {
                "confidence_change_by_transition": [
                    {
                        "transition": f"t{transition}_to_t{transition + 1}",
                        "signed_change": self.temporal_confidence_change[
                            transition
                        ].summary(),
                        "absolute_change": self.temporal_confidence_absolute_change[
                            transition
                        ].summary(),
                        "population": (
                            "all candidates of parents still live at the later timestep"
                        ),
                    }
                    for transition in range(self.timesteps - 1)
                ],
                "selected_action_confidence_before_execution": (
                    self.executed_confidence_before.summary()
                ),
                "same_action_confidence_after_execution": (
                    self.executed_confidence_after.summary()
                ),
                "same_action_confidence_change_after_execution": (
                    self.executed_confidence_change.summary()
                ),
                "repeated_selected_action_confidence_change": (
                    self.repeated_confidence_change.summary()
                ),
                "interpretation_limitation": (
                    "confidence changes are observational; reduced confidence after an "
                    "execution is not assumed unless measured"
                ),
            },
            "null_vs_effect_magnitude_bins": overall_bins,
            "null_vs_effect_magnitude_bins_by_timestep": [
                {
                    "timestep": timestep,
                    "population": f"all K candidates of live parents at t={timestep}",
                    "bins": bins.summary(),
                }
                for timestep, bins in enumerate(self.null_effect_bins_by_timestep)
            ],
            "stop_interaction": {
                "mean_candidate_confidence_when_stop": (
                    self.stop_mean_confidence.summary()
                ),
                "mean_candidate_confidence_when_edit": (
                    self.edit_mean_confidence.summary()
                ),
                "max_candidate_confidence_when_stop": (
                    self.stop_max_confidence.summary()
                ),
                "max_candidate_confidence_when_edit": (
                    self.edit_max_confidence.summary()
                ),
                "all_candidates_p_null_gt_0.8_count": (
                    self.all_candidates_high_null_count
                ),
                "all_candidates_p_null_gt_0.8_stop_count": (
                    self.all_candidates_high_null_stop_count
                ),
                "stop_rate_when_all_candidates_p_null_gt_0.8": (
                    self.all_candidates_high_null_stop_count
                    / self.all_candidates_high_null_count
                    if self.all_candidates_high_null_count
                    else None
                ),
                "interpretation_limitation": (
                    "STOP architecture is unchanged; these are observational associations"
                ),
            },
            "metric_population": (
                "dynamic p_null for all candidates of live parents at each timestep"
            ),
            "shortcut_observation_flags": {
                "null_effectively_ignored": mean_null <= 0.01,
                "null_globally_dominant": mean_null >= 0.99,
                "higher_null_bin_fails_to_reduce_delta_z": (
                    None
                    if low_bin_delta is None or high_bin_delta is None
                    else high_bin_delta >= low_bin_delta
                ),
                "candidate_mean_null_range": (
                    max(float(value) for value in candidate_means if value is not None)
                    - min(float(value) for value in candidate_means if value is not None)
                ),
                "threshold_note": (
                    "descriptive conventions only; flags do not establish semantic "
                    "applicability or causal failure"
                ),
            },
        }

    def functional_summary(self) -> dict[str, Any]:
        per_timestep = []
        mean_delta_q_by_timestep: list[float | None] = []
        for timestep in range(self.timesteps):
            if self.delta_z_norm[timestep].count == 0:
                mean_delta_q_by_timestep.append(None)
                per_timestep.append(
                    {
                        "timestep": timestep,
                        "live_parent_count": 0,
                        "metric_population": (
                            f"live parents before decision at t={timestep}"
                        ),
                    }
                )
                continue
            delta_z_norm = self.delta_z_norm[timestep].mean()
            delta_q_norm = self.delta_q_norm[timestep].mean()
            mean_delta_q = float(delta_q_norm.mean())
            mean_delta_q_by_timestep.append(mean_delta_q)
            context_cosine = self.context_cosine[timestep].mean()
            delta_z_cosine = self.delta_z_cosine[timestep].nullable_matrix()
            delta_q_cosine = self.delta_q_cosine[timestep].nullable_matrix()
            delta_z_cosine_summary = self.delta_z_cosine[
                timestep
            ].off_diagonal_summary()
            delta_q_cosine_summary = self.delta_q_cosine[
                timestep
            ].off_diagonal_summary()
            delta_z_activity = self.delta_z_activity[timestep].summary()
            delta_q_activity = self.delta_q_activity[timestep].summary()
            distribution = self.delta_q_norm_distribution[timestep].summary()
            per_timestep.append(
                {
                    "timestep": timestep,
                    "live_parent_count": self.delta_z_norm[timestep].count,
                    "metric_population": f"live parents before decision at t={timestep}",
                    "mean_delta_z_norm": float(delta_z_norm.mean()),
                    "mean_delta_q_norm": mean_delta_q,
                    "median_delta_q_norm": distribution["median"],
                    "candidate_wise_delta_z_norm": delta_z_norm.tolist(),
                    "candidate_wise_delta_q_norm": delta_q_norm.tolist(),
                    "context_pairwise_cosine_matrix": context_cosine.tolist(),
                    "context_pairwise_cosine_off_diagonal": _off_diagonal_summary(
                        context_cosine
                    ),
                    "delta_z_pairwise_cosine_matrix": delta_z_cosine,
                    "delta_z_pairwise_cosine_matrix_among_valid_effects": (
                        delta_z_cosine
                    ),
                    "delta_z_pairwise_cosine_off_diagonal": delta_z_cosine_summary,
                    "delta_z_pairwise_cosine_valid_pair_fraction": (
                        delta_z_cosine_summary["valid_pair_fraction"]
                    ),
                    "delta_z_pairwise_cosine_valid_pair_count": (
                        delta_z_cosine_summary["valid_pair_count"]
                    ),
                    "delta_q_pairwise_cosine_matrix": delta_q_cosine,
                    "pairwise_delta_q_cosine_matrix_among_valid_effects": (
                        delta_q_cosine
                    ),
                    "delta_q_pairwise_cosine_off_diagonal": delta_q_cosine_summary,
                    "pairwise_delta_q_cosine_mean_off_diagonal": (
                        delta_q_cosine_summary["mean"]
                    ),
                    "pairwise_delta_q_cosine_valid_pair_fraction": (
                        delta_q_cosine_summary["valid_pair_fraction"]
                    ),
                    "pairwise_delta_q_cosine_valid_pair_count": (
                        delta_q_cosine_summary["valid_pair_count"]
                    ),
                    "delta_z_activity": delta_z_activity,
                    "active_delta_z_candidate_fraction": delta_z_activity[
                        "active_candidate_fraction"
                    ],
                    "dead_delta_z_candidate_fraction": delta_z_activity[
                        "dead_candidate_fraction"
                    ],
                    "dead_delta_z_parent_fraction": delta_z_activity[
                        "dead_parent_fraction"
                    ],
                    "delta_q_activity": delta_q_activity,
                    "active_candidate_fraction": delta_q_activity[
                        "active_candidate_fraction"
                    ],
                    "dead_candidate_fraction": delta_q_activity[
                        "dead_candidate_fraction"
                    ],
                    "dead_parent_fraction": delta_q_activity[
                        "dead_parent_fraction"
                    ],
                    "delta_z_effective_rank": float(
                        self.delta_z_rank[timestep].mean().mean()
                    ),
                    "functional_effective_rank": float(
                        self.delta_q_rank[timestep].mean().mean()
                    ),
                }
            )
        epsilon = 1e-8

        def ratio(numerator: int, denominator: int) -> float | None:
            top = mean_delta_q_by_timestep[numerator]
            bottom = mean_delta_q_by_timestep[denominator]
            if top is None or bottom is None:
                return None
            return top / (bottom + epsilon)

        return {
            "functional_effect_activity_epsilon": FUNCTIONAL_ACTIVITY_EPSILON,
            "activity_definition": (
                "active(Delta)=1 when its L2 norm is strictly greater than the "
                "diagnostic activity epsilon"
            ),
            "per_timestep": per_timestep,
            "late_step_effect_retention": {
                "mean_delta_q_norm_t1_over_t0": ratio(1, 0),
                "mean_delta_q_norm_t2_over_t0": ratio(2, 0),
                "mean_delta_q_norm_t2_over_t1": ratio(2, 1),
                "epsilon": epsilon,
                "interpretation_limitation": (
                    "descriptive effect retention only; no universal collapse threshold"
                ),
            },
        }

    def dynamic_summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, transitions in self.dynamic.items():
            summaries = [
                stats.summary(
                    parent_count=int(self.dynamic_parent_counts[transition]),
                    population=(
                        f"live parents at t={transition} that executed a non-STOP edit; "
                        f"change measured from t={transition} to t={transition + 1}"
                    ),
                )
                for transition, stats in enumerate(transitions)
            ]
            total_count = sum(stats.count for stats in transitions)
            total = sum(stats.total for stats in transitions)
            result[name] = {
                "per_transition": summaries,
                "overall_mean_absolute_change": total / max(total_count, 1),
                "overall_max_absolute_change": max(
                    (stats.maximum for stats in transitions), default=0.0
                ),
                "overall_is_secondary_aggregate": True,
            }
        return result

    def specialization_summary(self) -> dict[str, Any]:
        return {
            "pairwise_intent_cosine": self.intent_cosine.mean().tolist(),
            "pairwise_support_cosine": self.support_cosine.nullable_matrix(),
            "pairwise_support_cosine_off_diagonal_valid_shapes": (
                self.support_cosine.off_diagonal_summary()
            ),
            "pairwise_context_cosine": self.context_cosine_all.mean().tolist(),
            "pairwise_delta_z_cosine": self.delta_z_cosine_all.nullable_matrix(),
            "pairwise_delta_q_cosine": self.delta_q_cosine_all.nullable_matrix(),
            "pairwise_delta_z_cosine_off_diagonal_among_valid_effects": (
                self.delta_z_cosine_all.off_diagonal_summary()
            ),
            "pairwise_delta_q_cosine_off_diagonal_among_valid_effects": (
                self.delta_q_cosine_all.off_diagonal_summary()
            ),
            "aggregation_note": (
                "context/delta-Z/delta-q matrices are secondary aggregates weighted by "
                "live-parent observations across timesteps; use functional per_timestep "
                "matrices for lineage-safe conclusions"
            ),
        }


class SelectedPathMarginalAccumulator:
    """Offline target-relative measurements of transitions already selected by the model."""

    def __init__(self, timesteps: int = 3) -> None:
        self.timesteps = timesteps
        self.similarity_improvements = [
            _DistributionStats() for _ in range(timesteps)
        ]

    def update(self, output: IAGSRMEOutput, target_features: Tensor) -> None:
        if target_features.ndim != 2 or target_features.shape[0] != output.intents.shape[0]:
            raise ValueError("target features must be [B,D] for offline diagnostics")
        normalized_targets = F.normalize(target_features.float(), dim=-1)
        candidates = output.intents.shape[1]
        for timestep, step in enumerate(output.trace):
            verify_same_parent_counterfactuals(step)
            valid = step.live_before & step.selected_index.lt(candidates)
            if not valid.any():
                continue
            selected_queries = step.candidate_queries[
                valid, step.selected_index[valid]
            ]
            torch.testing.assert_close(
                step.next_query[valid], selected_queries, atol=1e-6, rtol=1e-6
            )
            target = normalized_targets[valid]
            similarity_before = (
                F.normalize(step.current_query[valid].float(), dim=-1) * target
            ).sum(dim=-1)
            similarity_after = (
                F.normalize(step.next_query[valid].float(), dim=-1) * target
            ).sum(dim=-1)
            self.similarity_improvements[timestep].update(
                similarity_after - similarity_before
            )

    def summary(self) -> dict[str, Any]:
        return {
            "target_similarity_improvement_by_timestep": [
                {
                    "timestep": timestep,
                    "selected_non_stop_transition_count": stats.summary()["count"],
                    "metric_population": (
                        f"live parents at t={timestep} whose selected action was non-STOP"
                    ),
                    "delta_target_cosine_similarity": {
                        key: value
                        for key, value in stats.summary().items()
                        if key != "count"
                    },
                }
                for timestep, stats in enumerate(self.similarity_improvements)
            ],
            "target_firewall": (
                "target gallery features are consumed only after the complete target-free "
                "rollout has been constructed"
            ),
            "interpretation_limitation": (
                "target-similarity change describes the executed transition; it does not "
                "identify why the policy selected it"
            ),
        }


class NullTargetUtilityAccumulator:
    """Offline NULL/utility association after target-free candidates already exist."""

    def __init__(self, timesteps: int = 3) -> None:
        self.null_probabilities = _DistributionStats()
        self.utilities = _DistributionStats()
        bins = len(NULL_EFFECT_BIN_EDGES) - 1
        self.bin_count = torch.zeros(bins, dtype=torch.long)
        self.bin_utility_total = torch.zeros(bins, dtype=torch.float64)
        self.bin_positive = torch.zeros(bins, dtype=torch.long)
        self.bin_negative = torch.zeros(bins, dtype=torch.long)
        self.timestep_bin_count = torch.zeros(timesteps, bins, dtype=torch.long)
        self.timestep_bin_utility_total = torch.zeros(
            timesteps, bins, dtype=torch.float64
        )
        self.timestep_bin_positive = torch.zeros(timesteps, bins, dtype=torch.long)
        self.timestep_bin_negative = torch.zeros(timesteps, bins, dtype=torch.long)

    def update(self, output: IAGSRMEOutput, target_features: Tensor) -> None:
        if output.visual_null_probabilities is None:
            return
        normalized_targets = F.normalize(target_features.float(), dim=-1)
        for timestep, step in enumerate(output.trace):
            if step.visual_null_probability is None:
                continue
            valid = step.live_before
            if not valid.any():
                continue
            targets = normalized_targets[valid]
            current_similarity = (
                F.normalize(step.current_query[valid].float(), dim=-1) * targets
            ).sum(dim=-1)
            candidate_similarity = torch.einsum(
                "bkd,bd->bk",
                F.normalize(step.candidate_queries[valid].float(), dim=-1),
                targets,
            )
            utility = candidate_similarity - current_similarity[:, None]
            null_values = step.visual_null_probability[valid].float()
            self.null_probabilities.update(null_values)
            self.utilities.update(utility)
            flat_null = null_values.detach().cpu().flatten()
            flat_utility = utility.detach().cpu().flatten()
            for index, (lower, upper) in enumerate(
                zip(NULL_EFFECT_BIN_EDGES[:-1], NULL_EFFECT_BIN_EDGES[1:], strict=True)
            ):
                mask = flat_null.ge(lower) & flat_null.lt(upper)
                self.bin_count[index] += int(mask.sum())
                self.bin_utility_total[index] += float(flat_utility[mask].sum())
                self.bin_positive[index] += int(flat_utility[mask].gt(0).sum())
                self.bin_negative[index] += int(flat_utility[mask].lt(0).sum())
                self.timestep_bin_count[timestep, index] += int(mask.sum())
                self.timestep_bin_utility_total[timestep, index] += float(
                    flat_utility[mask].sum()
                )
                self.timestep_bin_positive[timestep, index] += int(
                    flat_utility[mask].gt(0).sum()
                )
                self.timestep_bin_negative[timestep, index] += int(
                    flat_utility[mask].lt(0).sum()
                )

    def summary(self) -> dict[str, Any]:
        null_values = self.null_probabilities.values()
        utilities = self.utilities.values()
        if null_values.numel() > 1:
            centered_null = null_values - null_values.mean()
            centered_utility = utilities - utilities.mean()
            denominator = centered_null.norm() * centered_utility.norm()
            pearson = (
                float((centered_null * centered_utility).sum() / denominator)
                if float(denominator) > 0
                else None
            )
        else:
            pearson = None
        bins = []
        for index, (lower, upper) in enumerate(
            zip(NULL_EFFECT_BIN_EDGES[:-1], NULL_EFFECT_BIN_EDGES[1:], strict=True)
        ):
            count = int(self.bin_count[index])
            bins.append(
                {
                    "range": [lower, min(upper, 1.0)],
                    "count": count,
                    "mean_candidate_target_similarity_gain": (
                        float(self.bin_utility_total[index] / count) if count else None
                    ),
                    "positive_utility_rate": (
                        float(self.bin_positive[index] / count) if count else None
                    ),
                    "negative_utility_rate": (
                        float(self.bin_negative[index] / count) if count else None
                    ),
                }
            )
        timestep_bins = []
        for timestep in range(self.timestep_bin_count.shape[0]):
            items = []
            for index, (lower, upper) in enumerate(
                zip(NULL_EFFECT_BIN_EDGES[:-1], NULL_EFFECT_BIN_EDGES[1:], strict=True)
            ):
                count = int(self.timestep_bin_count[timestep, index])
                items.append(
                    {
                        "range": [lower, min(upper, 1.0)],
                        "count": count,
                        "mean_candidate_target_similarity_gain": (
                            float(self.timestep_bin_utility_total[timestep, index] / count)
                            if count
                            else None
                        ),
                        "positive_utility_rate": (
                            float(self.timestep_bin_positive[timestep, index] / count)
                            if count
                            else None
                        ),
                        "negative_utility_rate": (
                            float(self.timestep_bin_negative[timestep, index] / count)
                            if count
                            else None
                        ),
                    }
                )
            timestep_bins.append(
                {
                    "timestep": timestep,
                    "population": f"all candidates of live parents at t={timestep}",
                    "bins": items,
                }
            )
        return {
            "pearson_p_null_vs_candidate_utility": pearson,
            "utility_definition": "cos(qhat_t+1,k, target)-cos(q_t, target)",
            "utility_by_null_bin": bins,
            "utility_by_null_bin_and_timestep": timestep_bins,
            "candidate_observation_count": null_values.numel(),
            "target_firewall": (
                "targets are accessed only after target-free rollout/candidates are complete"
            ),
            "interpretation_limitation": (
                "offline association is diagnostic and never supervises forward execution"
            ),
        }


def diagnostic_definitions() -> dict[str, dict[str, Any]]:
    return {
        "temporal_support_change": {
            "definition": (
                "for fixed candidate k, compare pi[t,k]=Ground(I_k,Z_t) with "
                "pi[t+1,k]=Ground(I_k,Z_t+1) using cosine, probability overlap, "
                "top-M Jaccard, L1/L2 displacement, entropy/effective-size change, "
                "and argmax-token movement"
            ),
            "population": (
                "samples live before the later timestep; decision-conditioned summaries "
                "also retain same-candidate, other-candidate, and STOP populations"
            ),
            "tensor_shape_before_reduction": "temporal supports [B,T,K,N]",
            "reduction_axes": "token axis N, then distribution over lineage-valid B,K",
            "interpretation_limitation": (
                "motion can be self-induced key drift or shared jitter; it is not semantic "
                "success without functional/retrieval evidence"
            ),
        },
        "candidate_support_displacement_alignment": {
            "definition": "cos(pi[t+1,i]-pi[t,i], pi[t+1,j]-pi[t,j])",
            "population": "candidate pairs of samples live before the later timestep",
            "tensor_shape_before_reduction": "support displacement [B_live,K,N]",
            "reduction_axes": "cosine over N; B mean with KxK matrix retained",
            "interpretation_limitation": (
                "high alignment indicates co-motion, not necessarily identical semantics"
            ),
        },
        "visual_null_probability": {
            "definition": (
                "p_null[t,k]=1-sigmoid(G_app(context[t,k])); it is a dynamic Bernoulli "
                "applicability complement, not a spatial Entmax coordinate"
            ),
            "population": "all four candidates of live parents at each timestep",
            "tensor_shape_before_reduction": "[B_live,T,K]",
            "reduction_axes": "distribution by timestep and candidate identity",
            "interpretation_limitation": (
                "target-free numerical applicability; not semantic NULL and not STOP"
            ),
        },
        "real_visual_support_mass": {
            "definition": "sum_n pi[k,n]=1 for fixed legacy/R1a Entmax WHERE",
            "population": "all validation queries/candidates",
            "tensor_shape_before_reduction": "pi [B,K,N]",
            "reduction_axes": "sum over N; distribution over B,K",
            "interpretation_limitation": (
                "spatial support mass is separate from dynamic execution confidence"
            ),
        },
        "conditional_support_shape": {
            "definition": "pi is already unit-mass Entmax support over real tokens",
            "population": "all candidate supports",
            "tensor_shape_before_reduction": "[B,K,N]",
            "reduction_axes": "shape entropy/overlap/cosine over N; zero-mass shapes excluded",
            "interpretation_limitation": (
                "diagnostic normalization does not enter or alter execution"
            ),
        },
        "null_vs_effect_magnitude": {
            "definition": (
                "bin same-parent candidates by dynamic p_null[t,k] and summarize ||DeltaZ||, "
                "||Deltaq||, and selection rate"
            ),
            "population": "all K candidates of live parents at each timestep",
            "tensor_shape_before_reduction": "p_null [B_live,K], effects [B_live,K,*]",
            "reduction_axes": "effect norm then aggregate within fixed p_null bins",
            "interpretation_limitation": (
                "descriptive confidence response; selection association is not a new policy loss"
            ),
        },
        "null_vs_offline_candidate_utility": {
            "definition": "utility=cos(qhat_t+1,k,y)-cos(q_t,y)",
            "population": "all same-parent candidates of live parents",
            "tensor_shape_before_reduction": "candidate queries [B_live,K,D]",
            "reduction_axes": "target cosine over D; association/binning over observations",
            "interpretation_limitation": (
                "target is offline-only and never enters candidate construction or selection"
            ),
        },
        "pairwise_intent_cosine": {
            "definition": "cos(I_i, I_j) for every ordered candidate pair i,j",
            "population": "all validation queries",
            "tensor_shape_before_reduction": "[B,K,d]",
            "reduction_axes": "mean over B only; KxK matrix preserved",
            "interpretation_limitation": "representational similarity is not semantic correctness",
        },
        "support_probability_overlap": {
            "definition": "sum_n min(P_i[n], P_j[n])",
            "population": "all validation queries; static supports computed once",
            "tensor_shape_before_reduction": "[B,K,K,N]",
            "reduction_axes": "sum over N, then mean over B; KxK preserved",
            "interpretation_limitation": "overlap does not establish semantic ownership",
        },
        "dominant_tokenwise_grounding_mass_share": {
            "definition": "sum_n max_k P[k,n] / sum_{k,n} P[k,n]",
            "population": "all validation queries",
            "tensor_shape_before_reduction": "[B,K,N]",
            "reduction_axes": "max over K per token, sum over N, then mean over B",
            "interpretation_limitation": "heuristic concentration statistic, not ownership",
        },
        "context_pairwise_cosine": {
            "definition": "cos(C_t,i, C_t,j)",
            "population": "live parents before the decision at each reported timestep",
            "tensor_shape_before_reduction": "[B_live,K,d]",
            "reduction_axes": "mean over B_live only; reported separately by timestep",
            "interpretation_limitation": "context similarity alone does not imply equal edits",
        },
        "delta_z_pairwise_cosine": {
            "definition": (
                "cos(vec(DeltaZ_t,i), vec(DeltaZ_t,j)) only when both effects have "
                f"L2 norm > {FUNCTIONAL_ACTIVITY_EPSILON}"
            ),
            "population": "live parents before the decision at each reported timestep",
            "tensor_shape_before_reduction": "[B_live,K,N,d] -> [B_live,K,N*d]",
            "reduction_axes": "flatten N,d; mean over B_live; KxK preserved",
            "interpretation_limitation": (
                "token-effect similarity is not retrieval utility; null means no valid "
                "active-pair observations"
            ),
        },
        "delta_q_pairwise_cosine": {
            "definition": (
                "Deltaq_t,k=qhat_t+1,k-q_t; pairwise cosine is defined only when both "
                f"effect norms are > {FUNCTIONAL_ACTIVITY_EPSILON}"
            ),
            "population": "live same-parent counterfactuals at each timestep",
            "tensor_shape_before_reduction": "[B_live,K,D]",
            "reduction_axes": "mean over B_live only; KxK preserved",
            "interpretation_limitation": (
                "functional similarity does not identify its cause; inactive pairs are "
                "excluded rather than assigned cosine zero"
            ),
        },
        "functional_effect_activity": {
            "definition": (
                f"active(Delta_t,k)=[||Delta_t,k||_2>{FUNCTIONAL_ACTIVITY_EPSILON}]"
            ),
            "population": "live same-parent counterfactuals at each timestep",
            "tensor_shape_before_reduction": (
                "Deltaq [B_live,K,D] or DeltaZ [B_live,K,N*d]"
            ),
            "reduction_axes": (
                "candidate fraction over B_live,K; dead-parent fraction over B_live"
            ),
            "interpretation_limitation": (
                "numerical inactivity is not semantic edit failure and has no causal meaning"
            ),
        },
        "functional_effective_rank": {
            "definition": (
                "exp(entropy(normalized singular values of the KxD effect matrix)); "
                "diagnostic convention r_eff=0 when singular-value mass <= 1e-8"
            ),
            "population": "live parents before the decision at each timestep",
            "tensor_shape_before_reduction": "[B_live,K,D]",
            "reduction_axes": "SVD over K,D; mean scalar rank over B_live",
            "interpretation_limitation": (
                "uncentered numerical rank is not semantic factor count; zero rank denotes "
                "only a numerically zero effect matrix"
            ),
        },
        "late_step_effect_retention": {
            "definition": "E||Deltaq_t|| / (E||Deltaq_reference|| + 1e-8)",
            "population": "each timestep's own live-parent population",
            "tensor_shape_before_reduction": "[B_live,K,D]",
            "reduction_axes": "norm over D, mean over B_live,K",
            "interpretation_limitation": "different live populations make this descriptive, not causal",
        },
        "absorbed_stop_occupancy": {
            "definition": "count(action_t=STOP) / count(all validation trajectories)",
            "population": "all trajectories, including trajectories stopped earlier",
            "tensor_shape_before_reduction": "[B,T]",
            "reduction_axes": "mean over B separately by timestep",
            "interpretation_limitation": "must not be interpreted as new STOP hazard",
        },
        "new_stop_hazard": {
            "definition": "new STOP selections at t / live parents before decision t",
            "population": "live parents before the decision at each timestep",
            "tensor_shape_before_reduction": "live_before[B,T], stopped_now[B,T]",
            "reduction_axes": "sum over B separately by timestep",
            "interpretation_limitation": "policy behavior, not semantic factor inactivity",
        },
        "selected_path_target_similarity_improvement": {
            "definition": "cos(q_t+1,y)-cos(q_t,y) for selected non-STOP transitions",
            "population": "live parents whose actual selected action was non-STOP",
            "tensor_shape_before_reduction": "q_t,q_t+1,y: [B_selected,D]",
            "reduction_axes": "cosine over D; distribution summarized per timestep",
            "interpretation_limitation": "offline target-relative observation; no target enters forward",
        },
        "same_parent_counterfactual_retrieval": {
            "definition": "retrieval of every qhat_t+1,k constructed from the same q_t/Z_t",
            "population": "live parents before the decision at each timestep",
            "tensor_shape_before_reduction": "candidate queries [B_live,K,D]",
            "reduction_axes": "retrieval per K plus offline best-candidate oracle and mean query",
            "interpretation_limitation": "oracle uses targets offline and is never a model action",
        },
        "single_and_repeat_controls": {
            "definition": (
                "SINGLE_k executes k once from Z0 then stops; REPEAT_k executes k through "
                "the real updated recurrent state for Tmax steps"
            ),
            "population": "all validation queries",
            "tensor_shape_before_reduction": "final controlled query [B,D]",
            "reduction_axes": "FashionIQ R@10/R@50 then category macro-average",
            "interpretation_limitation": "control superiority is an observation, not a causal mechanism",
        },
        "retrieval_control_ratios": {
            "definition": "control Mean Recall / FULL Mean Recall",
            "population": "the same FashionIQ validation queries and protocol gallery",
            "tensor_shape_before_reduction": "scalar macro Mean Recall values",
            "reduction_axes": "per-category recalls macro-averaged before ratio",
            "interpretation_limitation": "relative performance alone does not establish causality",
        },
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
                "live_parent_count": 0,
                "metric_population": (
                    f"live parents before decision at t={timestep}; no observations"
                ),
            }
            continue
        without_counts = [
            {
                name: value
                for name, value in item.items()
                if name not in {"live_parent_count", "metric_population"}
            }
            for item in valid
        ]
        averaged = _macro_numeric_tree(without_counts)
        averaged["live_parent_count"] = sum(int(item["live_parent_count"]) for item in valid)
        averaged["metric_population"] = (
            f"live parents before decision at t={timestep}; retrieval metrics are "
            "category-macro averages and live_parent_count is the raw pooled count"
        )
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


_LEGACY_CANONICAL_MODEL_CONFIG = {
    "max_steps": 3,
    "num_heads": 8,
    "lambda_z": 0.10,
    "query_cap": 0.50,
    "selector_temperature": 1.0,
    "enable_visual_null": False,
    "visual_null_initial_logit": 0.0,
    "enable_dynamic_applicability": False,
    "initial_applicability": 0.98,
    "grounding_normalization": "entmax15",
    "enable_dynamic_regrounding": False,
}
_SERIALIZED_MODEL_CONFIG_KEYS = (
    "model_config",
    "iag_srme_model_config",
    "iag_srme_config",
)
_SELF_DESCRIBING_MODEL_CONFIG_FIELDS = {
    "width",
    "num_candidates",
    "max_steps",
    "num_heads",
    "retrieval_dim",
    "lambda_z",
    "query_cap",
    "selector_temperature",
    "selector_gumbel_noise",
    "enable_claim_head",
    "enable_factor_head",
}


def _serialized_model_config(checkpoint: Mapping[str, Any]) -> Mapping[str, Any] | None:
    metadata = checkpoint.get("metadata")
    containers = (checkpoint, metadata if isinstance(metadata, Mapping) else {})
    for container in containers:
        for key in _SERIALIZED_MODEL_CONFIG_KEYS:
            value = container.get(key)
            if isinstance(value, Mapping):
                return value
    return None


def _resolve_checkpoint_model_config(
    checkpoint: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    retrieval_dim: int,
) -> tuple[IAGSRMEConfig, dict[str, Any]]:
    """Resolve replay-critical config, preferring serialized checkpoint provenance."""
    query_bank = state.get("core.intent_encoder.query_bank")
    if not isinstance(query_bank, Tensor) or query_bank.ndim != 2:
        raise ValueError("checkpoint is not an IAG-SRME checkpoint")
    candidates, width = query_bank.shape
    enable_claim = any(key.startswith("core.claim_head.") for key in state)
    enable_factor = any(key.startswith("core.factor_fuser.") for key in state)
    inferred_entmax_null_v1 = "core.grounder.visual_null_key" in state
    inferred_dynamic_applicability = any(
        key.startswith("core.applicability_head.") for key in state
    )
    if inferred_entmax_null_v1:
        raise ValueError(
            "checkpoint uses superseded r1b_visual_null_entmax_v1; it cannot be "
            "replayed as r1b_dynamic_applicability_gate_v2"
        )
    factor_output = state.get("core.factor_fuser.network.2.weight")
    inferred_factor_dim = (
        int(factor_output.shape[0]) if isinstance(factor_output, Tensor) else None
    )
    serialized = _serialized_model_config(checkpoint)
    metadata = checkpoint.get("metadata")
    stored_generation = (
        metadata.get("architecture_generation")
        if isinstance(metadata, Mapping)
        else None
    )
    serialized_dynamic_regrounding = False

    if serialized is None:
        if inferred_dynamic_applicability:
            raise ValueError(
                "dynamic-applicability checkpoint is not self-describing; exact R1b "
                "replay is unsafe"
            )
        replay_values = dict(_LEGACY_CANONICAL_MODEL_CONFIG)
        source = "legacy_checkpoint_plus_canonical_assumption"
        fully_self_describing = False
        warning = (
            "checkpoint does not encode every non-state-dict model hyperparameter"
        )
    else:
        required_fields = set(_SELF_DESCRIBING_MODEL_CONFIG_FIELDS)
        serialized_dynamic_regrounding = bool(
            serialized.get("enable_dynamic_regrounding", False)
        )
        if stored_generation == "r1c1_dynamic_current_state_reground_v1":
            required_fields.add("enable_dynamic_regrounding")
        if serialized_dynamic_regrounding:
            required_fields.update(
                {"enable_dynamic_regrounding", "grounding_normalization"}
            )
        if inferred_dynamic_applicability:
            required_fields.update(
                {
                    "enable_dynamic_applicability",
                    "initial_applicability",
                    "grounding_normalization",
                }
            )
        if bool(serialized.get("enable_factor_head")):
            required_fields.add("factor_dim")
        missing = sorted(required_fields - serialized.keys())
        if missing:
            raise ValueError(
                "serialized checkpoint model config is incomplete: "
                f"missing {sorted(missing)}"
            )
        replay_values = {
            key: serialized.get(key, default)
            for key, default in _LEGACY_CANONICAL_MODEL_CONFIG.items()
        }
        source = "checkpoint"
        fully_self_describing = True
        warning = None
        inferable_expected = {
            "width": width,
            "num_candidates": candidates,
            "retrieval_dim": retrieval_dim,
            "enable_claim_head": enable_claim,
            "enable_factor_head": enable_factor,
            "enable_visual_null": False,
            "enable_dynamic_applicability": inferred_dynamic_applicability,
        }
        for key, expected in inferable_expected.items():
            if key in serialized and serialized[key] != expected:
                raise ValueError(
                    f"serialized model config {key}={serialized[key]!r} conflicts "
                    f"with state-dict/backbone inferred value {expected!r}"
                )
        if "factor_dim" in serialized and serialized["factor_dim"] not in {
            None,
            inferred_factor_dim,
        }:
            raise ValueError("serialized factor_dim conflicts with state dict")
        if serialized_dynamic_regrounding and inferred_dynamic_applicability:
            raise ValueError(
                "R1c1 checkpoint cannot combine dynamic regrounding with R1b applicability"
            )
        if serialized_dynamic_regrounding and stored_generation != (
            "r1c1_dynamic_current_state_reground_v1"
        ):
            raise ValueError(
                "dynamic-regrounding checkpoint has missing or conflicting architecture generation"
            )
        if stored_generation == "r1c1_dynamic_current_state_reground_v1" and not (
            serialized_dynamic_regrounding
        ):
            raise ValueError(
                "R1c1 architecture metadata requires enable_dynamic_regrounding=true"
            )

    if int(replay_values["max_steps"]) != 3:
        raise ValueError("R0 diagnostic runner supports canonical Tmax=3 checkpoints")
    resolved = {
        "width": int(width),
        "num_candidates": int(candidates),
        "max_steps": int(replay_values["max_steps"]),
        "num_heads": int(replay_values["num_heads"]),
        "retrieval_dim": int(retrieval_dim),
        "lambda_z": float(replay_values["lambda_z"]),
        "query_cap": float(replay_values["query_cap"]),
        "selector_temperature": float(replay_values["selector_temperature"]),
        # Diagnostics always use deterministic hard argmax regardless of training noise.
        "selector_gumbel_noise": False,
        "enable_claim_head": enable_claim,
        "enable_factor_head": enable_factor,
        "factor_dim": inferred_factor_dim,
        "enable_visual_null": bool(replay_values["enable_visual_null"]),
        "visual_null_initial_logit": float(
            replay_values["visual_null_initial_logit"]
        ),
        "enable_dynamic_applicability": bool(
            replay_values["enable_dynamic_applicability"]
        ),
        "initial_applicability": float(replay_values["initial_applicability"]),
        "grounding_normalization": str(replay_values["grounding_normalization"]),
        "enable_dynamic_regrounding": bool(
            replay_values["enable_dynamic_regrounding"]
        ),
    }
    provenance: dict[str, Any] = {
        "source": source,
        "fully_self_describing": fully_self_describing,
        "state_dict_inferred": {
            "width": width,
            "num_candidates": candidates,
            "enable_claim_head": enable_claim,
            "enable_factor_head": enable_factor,
            "factor_dim": inferred_factor_dim,
            "enable_visual_null": False,
            "enable_dynamic_applicability": inferred_dynamic_applicability,
            "enable_dynamic_regrounding": (
                "not inferable from state dict; resolved from serialized model config"
            ),
        },
        "resolved_diagnostic_config": resolved,
        "diagnostic_inference_override": {"selector_gumbel_noise": False},
        "warning": warning,
        "architecture_generation": (
            "r1c1_dynamic_current_state_reground_v1"
            if resolved["enable_dynamic_regrounding"]
            else (
                "r1b_dynamic_applicability_gate_v2"
                if inferred_dynamic_applicability
                else "legacy_r0_or_r1a"
            )
        ),
    }
    if not fully_self_describing:
        provenance["assumed_config"] = {
            "width": width,
            "num_candidates": candidates,
            **_LEGACY_CANONICAL_MODEL_CONFIG,
        }
    else:
        provenance["serialized_training_config"] = dict(serialized)
    return IAGSRMEConfig(**resolved), provenance


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
    model_config, provenance = _resolve_checkpoint_model_config(
        checkpoint, state, retrieval_dim=backbone.retrieval_dim
    )
    core = IAGSRMECore(model_config)
    model = IAGSRME(backbone, core)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    checkpoint["checkpoint_model_config_provenance"] = provenance
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
    category_selected_path: SelectedPathMarginalAccumulator,
    global_selected_path: SelectedPathMarginalAccumulator,
    category_null_utility: NullTargetUtilityAccumulator,
    global_null_utility: NullTargetUtilityAccumulator,
) -> dict[str, Any]:
    query_lists: defaultdict[str, list[Tensor]] = defaultdict(list)
    target_ids: list[str] = []
    reference_ids: list[str] = []
    counterfactual_queries: list[list[Tensor]] = [[] for _ in range(3)]
    counterfactual_targets: list[list[str]] = [[] for _ in range(3)]
    counterfactual_references: list[list[str]] = [[] for _ in range(3)]
    gallery_index = {image_id: index for index, image_id in enumerate(gallery_ids)}

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
        batch_targets = [str(target) for target in batch.target_ids]
        if any(target not in gallery_index for target in batch_targets):
            raise ValueError("diagnostic target is missing from the evaluation gallery")
        target_indices = torch.tensor(
            [gallery_index[target] for target in batch_targets],
            dtype=torch.long,
            device=gallery.device,
        )
        target_features = gallery.index_select(0, target_indices)
        # Target access begins only here, after the complete target-free rollout exists.
        category_selected_path.update(full, target_features)
        global_selected_path.update(full, target_features)
        category_null_utility.update(full, target_features)
        global_null_utility.update(full, target_features)
        query_lists["full"].append(full.final_query.cpu())
        query_lists["reference_only"].append(encoded.reference_global.cpu())
        for candidate in range(4):
            single = single_candidate_control(full, candidate)
            query_lists[f"single_{candidate}"].append(single.query.cpu())
            repeated = repeat_candidate_control(model.core, encoded, candidate)
            query_lists[f"repeat_{candidate}"].append(repeated.final_query.cpu())
        mean_output = model.core(encoded, control="mean_candidate")
        query_lists["mean_candidate"].append(mean_output.final_query.cpu())

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
                "live_parent_count": 0,
                "metric_population": (
                    f"live parents before decision at t={timestep}; no observations"
                ),
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
        result["metric_population"] = (
            f"live parents before decision at t={timestep}; every candidate query branches "
            "from the same current parent query/state"
        )
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
    ranks = [
        item.get("functional_effective_rank", 0.0)
        for item in functional["per_timestep"]
        if item.get("live_parent_count", 0) > 0
    ]
    mean_rank = sum(ranks) / max(len(ranks), 1)
    support_fraction_value = grounding["support_fraction"]
    support_fraction = (
        float(support_fraction_value) if support_fraction_value is not None else None
    )
    margin = 2.0
    thresholds = {
        "stop_or_monopoly_fraction": 0.95,
        "clone_cosine": 0.95,
        "grounding_sparse_fraction": 0.02,
        "grounding_diffuse_fraction": 0.80,
        "functional_rank": 1.50,
        "dead_delta_q_fraction": 0.80,
        "retrieval_margin_points": margin,
    }
    aggregate_delta_q_cosine = specialization[
        "pairwise_delta_q_cosine_off_diagonal_among_valid_effects"
    ]["mean"]
    timestep_functional: dict[int, dict[str, float | int | None]] = {}
    for timestep in range(3):
        item = functional["per_timestep"][timestep]
        live_parent_count = int(item.get("live_parent_count", 0))
        timestep_functional[timestep] = {
            "live_parent_count": live_parent_count,
            "mean_delta_q_off_diagonal_cosine_among_valid_effects": (
                item.get("pairwise_delta_q_cosine_mean_off_diagonal")
                if live_parent_count > 0
                else None
            ),
            "functional_effective_rank": (
                item.get("functional_effective_rank")
                if live_parent_count > 0
                else None
            ),
            "dead_delta_q_candidate_fraction": (
                item.get("dead_candidate_fraction")
                if live_parent_count > 0
                else None
            ),
        }
    supporting = {
        "stop_t0_occupancy": selection["absorbed_stop_occupancy_by_timestep"][0],
        "total_new_stops": sum(selection["counts"]["new_stops"]),
        "maximum_candidate_selection_share": max(candidate_distribution),
        "mean_delta_q_off_diagonal_cosine": aggregate_delta_q_cosine,
        "mean_support_off_diagonal_cosine": specialization[
            "pairwise_support_cosine_off_diagonal_valid_shapes"
        ]["mean"],
        "support_fraction": support_fraction,
        "mean_functional_effective_rank": mean_rank,
        "full_mean_recall": full,
        "best_single_mean_recall": best_single,
        "best_repeat_mean_recall": best_repeat,
        "reference_only_mean_recall": reference,
        "t0_candidate_oracle_mean_recall": oracle_t0,
    }
    flags = {
        "high_stop_t0_occupancy": (
            supporting["stop_t0_occupancy"] >= thresholds["stop_or_monopoly_fraction"]
        ),
        "never_stop": supporting["total_new_stops"] == 0,
        "single_candidate_monopoly": (
            supporting["maximum_candidate_selection_share"]
            >= thresholds["stop_or_monopoly_fraction"]
        ),
        "high_delta_q_similarity": (
            None
            if supporting["mean_delta_q_off_diagonal_cosine"] is None
            else supporting["mean_delta_q_off_diagonal_cosine"]
            >= thresholds["clone_cosine"]
        ),
        "high_support_similarity": (
            None
            if supporting["mean_support_off_diagonal_cosine"] is None
            else supporting["mean_support_off_diagonal_cosine"]
            >= thresholds["clone_cosine"]
        ),
        "grounding_over_sparse": (
            None
            if support_fraction is None
            else support_fraction <= thresholds["grounding_sparse_fraction"]
        ),
        "grounding_over_diffuse": (
            None
            if support_fraction is None
            else support_fraction >= thresholds["grounding_diffuse_fraction"]
        ),
        "low_functional_effective_rank": mean_rank <= thresholds["functional_rank"],
        "repeat_beats_full": best_repeat >= full + margin,
        "single_beats_full": best_single >= full + margin,
        "reference_dominates": reference >= full + margin,
        "selected_policy_underperforms_candidate_oracle": oracle_t0 >= full + margin,
    }
    for timestep, observations in timestep_functional.items():
        cosine = observations[
            "mean_delta_q_off_diagonal_cosine_among_valid_effects"
        ]
        rank = observations["functional_effective_rank"]
        dead_fraction = observations["dead_delta_q_candidate_fraction"]
        flags[f"high_delta_q_similarity_t{timestep}"] = (
            None if cosine is None else cosine >= thresholds["clone_cosine"]
        )
        flags[f"low_functional_effective_rank_t{timestep}"] = (
            None if rank is None else rank <= thresholds["functional_rank"]
        )
        flags[f"high_dead_delta_q_fraction_t{timestep}"] = (
            None
            if dead_fraction is None
            else dead_fraction >= thresholds["dead_delta_q_fraction"]
        )
    definitions = {
        "high_stop_t0_occupancy": (
            "observes STOP occupancy at t0 >= threshold; does not explain why STOP was chosen"
        ),
        "never_stop": "observes zero new STOP decisions; not a claim about required edit count",
        "single_candidate_monopoly": (
            "observes maximum conditional edit-selection share >= threshold"
        ),
        "high_delta_q_similarity": (
            "observes high mean off-diagonal same-parent Deltaq cosine; potential functional "
            "candidate collapse, without causal localization"
        ),
        "high_support_similarity": (
            "observes high support-map cosine; does not establish semantic grounding failure"
        ),
        "grounding_over_sparse": "observes support fraction <= configured audit threshold",
        "grounding_over_diffuse": "observes support fraction >= configured audit threshold",
        "low_functional_effective_rank": (
            "observes low uncentered Deltaq effect rank; not semantic factor count"
        ),
        "repeat_beats_full": "observes best REPEAT Mean Recall exceeds FULL by margin",
        "single_beats_full": "observes best SINGLE Mean Recall exceeds FULL by margin",
        "reference_dominates": "observes REFERENCE_ONLY exceeds FULL by margin",
        "selected_policy_underperforms_candidate_oracle": (
            "observes offline t0 oracle exceeds FULL by margin; oracle never enters policy"
        ),
    }
    for timestep in range(3):
        definitions[f"high_delta_q_similarity_t{timestep}"] = (
            f"observes high same-parent Deltaq cosine among active pairs at t{timestep}; "
            "null means no live parents or no valid active pair"
        )
        definitions[f"low_functional_effective_rank_t{timestep}"] = (
            f"observes low numerical Deltaq effective rank at t{timestep}; rank is not "
            "semantic factor count and all-zero effects have diagnostic rank zero"
        )
        definitions[f"high_dead_delta_q_fraction_t{timestep}"] = (
            f"observes many numerically inactive candidate effects at t{timestep}; this "
            "does not establish semantic edit failure or a causal mechanism"
        )
    audit_contracts = {
        "high_stop_t0_occupancy": (
            "stop_t0_occupancy >= stop_or_monopoly_fraction",
            ("stop_t0_occupancy",),
            ("stop_or_monopoly_fraction",),
        ),
        "never_stop": ("total_new_stops == 0", ("total_new_stops",), ()),
        "single_candidate_monopoly": (
            "maximum_candidate_selection_share >= stop_or_monopoly_fraction",
            ("maximum_candidate_selection_share",),
            ("stop_or_monopoly_fraction",),
        ),
        "high_delta_q_similarity": (
            "mean_delta_q_off_diagonal_cosine >= clone_cosine",
            ("mean_delta_q_off_diagonal_cosine",),
            ("clone_cosine",),
        ),
        "high_support_similarity": (
            "mean_support_off_diagonal_cosine >= clone_cosine",
            ("mean_support_off_diagonal_cosine",),
            ("clone_cosine",),
        ),
        "grounding_over_sparse": (
            "support_fraction <= grounding_sparse_fraction",
            ("support_fraction",),
            ("grounding_sparse_fraction",),
        ),
        "grounding_over_diffuse": (
            "support_fraction >= grounding_diffuse_fraction",
            ("support_fraction",),
            ("grounding_diffuse_fraction",),
        ),
        "low_functional_effective_rank": (
            "mean_functional_effective_rank <= functional_rank",
            ("mean_functional_effective_rank",),
            ("functional_rank",),
        ),
        "repeat_beats_full": (
            "best_repeat_mean_recall >= full_mean_recall + retrieval_margin_points",
            ("best_repeat_mean_recall", "full_mean_recall"),
            ("retrieval_margin_points",),
        ),
        "single_beats_full": (
            "best_single_mean_recall >= full_mean_recall + retrieval_margin_points",
            ("best_single_mean_recall", "full_mean_recall"),
            ("retrieval_margin_points",),
        ),
        "reference_dominates": (
            "reference_only_mean_recall >= full_mean_recall + retrieval_margin_points",
            ("reference_only_mean_recall", "full_mean_recall"),
            ("retrieval_margin_points",),
        ),
        "selected_policy_underperforms_candidate_oracle": (
            "t0_candidate_oracle_mean_recall >= full_mean_recall + retrieval_margin_points",
            ("t0_candidate_oracle_mean_recall", "full_mean_recall"),
            ("retrieval_margin_points",),
        ),
    }
    for timestep in range(3):
        prefix = f"t{timestep}"
        supporting.update(
            {
                f"{prefix}_live_parent_count": timestep_functional[timestep][
                    "live_parent_count"
                ],
                f"{prefix}_mean_delta_q_off_diagonal_cosine_among_valid_effects": (
                    timestep_functional[timestep][
                        "mean_delta_q_off_diagonal_cosine_among_valid_effects"
                    ]
                ),
                f"{prefix}_functional_effective_rank": timestep_functional[timestep][
                    "functional_effective_rank"
                ],
                f"{prefix}_dead_delta_q_candidate_fraction": timestep_functional[
                    timestep
                ]["dead_delta_q_candidate_fraction"],
            }
        )
        audit_contracts[f"high_delta_q_similarity_t{timestep}"] = (
            f"{prefix}_mean_delta_q_off_diagonal_cosine_among_valid_effects >= clone_cosine",
            (
                f"{prefix}_live_parent_count",
                f"{prefix}_mean_delta_q_off_diagonal_cosine_among_valid_effects",
            ),
            ("clone_cosine",),
        )
        audit_contracts[f"low_functional_effective_rank_t{timestep}"] = (
            f"{prefix}_functional_effective_rank <= functional_rank",
            (f"{prefix}_live_parent_count", f"{prefix}_functional_effective_rank"),
            ("functional_rank",),
        )
        audit_contracts[f"high_dead_delta_q_fraction_t{timestep}"] = (
            f"{prefix}_dead_delta_q_candidate_fraction >= dead_delta_q_fraction",
            (
                f"{prefix}_live_parent_count",
                f"{prefix}_dead_delta_q_candidate_fraction",
            ),
            ("dead_delta_q_fraction",),
        )
    flag_audit = {
        name: {
            "condition": condition,
            "supporting_numbers": {key: supporting[key] for key in supporting_keys},
            "thresholds": {key: thresholds[key] for key in threshold_keys},
            "interpretation_limitation": definitions[name],
        }
        for name, (condition, supporting_keys, threshold_keys) in audit_contracts.items()
    }
    return {
        "flags": flags,
        "supporting_numbers": supporting,
        "thresholds": thresholds,
        "definitions_and_limits": definitions,
        "per_flag_audit_contract": flag_audit,
        "status": "OBSERVATION flags only; causal INTERPRETATION requires follow-up experiments",
        "aggregate_functional_flags_are_secondary": True,
    }


def _validate_report_schema(report: Mapping[str, Any]) -> None:
    missing = REQUIRED_REPORT_KEYS - report.keys()
    if missing:
        raise AssertionError(f"diagnostic report is missing top-level keys: {sorted(missing)}")
    json.dumps(report, allow_nan=False)


def _checkpoint_replay_guard(
    checkpoint: Mapping[str, Any],
    provenance: Mapping[str, Any],
    replayed_mean_recall: float,
    *,
    tolerance: float = 1e-4,
) -> dict[str, Any]:
    generation = provenance.get("architecture_generation")
    supported_generations = {
        "r1b_dynamic_applicability_gate_v2",
        "r1c1_dynamic_current_state_reground_v1",
    }
    if generation not in supported_generations:
        return {
            "applicable": False,
            "architecture_generation": generation,
            "trusted_r1b_replay": None,
            "trusted_r1c1_replay": None,
        }
    resolved = provenance.get("resolved_diagnostic_config")
    if not isinstance(resolved, Mapping):
        raise ValueError("R1b replay has no resolved model config")
    checks = {
        "query_cap_is_1000": float(resolved.get("query_cap", -1.0)) == 1000.0,
        "checkpoint_fully_self_describing": (
            provenance.get("fully_self_describing") is True
        ),
    }
    if generation == "r1b_dynamic_applicability_gate_v2":
        checks.update(
            {
                "dynamic_applicability_enabled": (
                    resolved.get("enable_dynamic_applicability") is True
                ),
                "initial_applicability_is_0.98": (
                    float(resolved.get("initial_applicability", -1.0)) == 0.98
                ),
                "dynamic_regrounding_disabled": (
                    resolved.get("enable_dynamic_regrounding", False) is False
                ),
            }
        )
    else:
        checks.update(
            {
                "dynamic_regrounding_enabled": (
                    resolved.get("enable_dynamic_regrounding") is True
                ),
                "dynamic_applicability_disabled": (
                    resolved.get("enable_dynamic_applicability") is False
                ),
                "grounding_is_entmax15": (
                    resolved.get("grounding_normalization") == "entmax15"
                ),
            }
        )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"untrusted {generation} checkpoint replay: failed {failed}")
    saved_metric = checkpoint.get("metric")
    if not isinstance(saved_metric, (int, float)):
        raise ValueError(f"{generation} checkpoint has no saved selection metric")
    metric_error = abs(float(saved_metric) - replayed_mean_recall)
    if metric_error > tolerance:
        raise ValueError(
            f"{generation} replayed FULL Mean Recall does not match saved checkpoint metric: "
            f"saved={saved_metric}, replayed={replayed_mean_recall}, error={metric_error}"
        )
    return {
        "applicable": True,
        "architecture_generation": generation,
        "trusted_r1b_replay": generation == "r1b_dynamic_applicability_gate_v2",
        "trusted_r1c1_replay": generation == "r1c1_dynamic_current_state_reground_v1",
        "checks": checks,
        "saved_checkpoint_metric": float(saved_metric),
        "replayed_full_mean_recall": replayed_mean_recall,
        "absolute_error": metric_error,
        "tolerance": tolerance,
    }


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
    parser.add_argument("--protocol", choices=PROTOCOLS, default=PROTOCOL)
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
    global_selected_path = SelectedPathMarginalAccumulator()
    global_null_utility = NullTargetUtilityAccumulator()
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
            args.protocol, split_root, category, dataset.annotations, SPLIT
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
        category_selected_path = SelectedPathMarginalAccumulator()
        category_null_utility = NullTargetUtilityAccumulator()
        controls = _diagnose_category(
            model,
            loader,
            gallery,
            gallery_ids,
            device,
            category_accumulator,
            global_accumulator,
            category_selected_path,
            global_selected_path,
            category_null_utility,
            global_null_utility,
        )
        category_controls[category] = controls
        category_selection = category_accumulator.selection_summary()
        category_intent = category_accumulator.intent_summary()
        category_grounding = category_accumulator.grounding_summary()
        category_temporal_grounding = (
            category_accumulator.temporal_grounding_summary()
        )
        category_functional = category_accumulator.functional_summary()
        category_dynamic = category_accumulator.dynamic_summary()
        category_specialization = category_accumulator.specialization_summary()
        category_results[category] = {
            "global_metrics": controls["full"],
            "selection_diagnostics": category_selection,
            "intent_diagnostics": category_intent,
            "grounding_diagnostics": category_grounding,
            "temporal_grounding_diagnostics": category_temporal_grounding,
            "visual_null_diagnostics": {
                **category_accumulator.visual_null_summary(),
                "offline_target_relative_utility": category_null_utility.summary(),
            },
            "functional_diagnostics": category_functional,
            "dynamic_diagnostics": category_dynamic,
            "control_retrieval_metrics": controls,
            "same_parent_counterfactual_diagnostics": controls[
                "counterfactual_same_parent_by_timestep"
            ],
            "selected_path_marginal_diagnostics": category_selected_path.summary(),
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
    intent = global_accumulator.intent_summary()
    grounding = global_accumulator.grounding_summary()
    temporal_grounding = global_accumulator.temporal_grounding_summary()
    functional = global_accumulator.functional_summary()
    dynamic = global_accumulator.dynamic_summary()
    specialization = global_accumulator.specialization_summary()
    selected_path = global_selected_path.summary()
    current_delta_q = [
        item.get("mean_delta_q_norm") for item in functional["per_timestep"]
    ]
    current_selected_gain = [
        item["delta_target_cosine_similarity"]["mean"]
        for item in selected_path["target_similarity_improvement_by_timestep"]
    ]
    current_support_cosine = [
        item["between_candidate_support_cosine_off_diagonal"]["mean"]
        for item in temporal_grounding["per_timestep"]
    ]
    r1a_comparison = {
        "full_mean_recall": {
            "r1c1": float(global_controls["full"]["mean_recall"]),
            "r1a": TRUSTED_R1A_BASELINE["full_mean_recall"],
            "difference": float(global_controls["full"]["mean_recall"])
            - TRUSTED_R1A_BASELINE["full_mean_recall"],
        },
        "mean_delta_q_norm_by_timestep": {
            "r1c1": current_delta_q,
            "r1a": TRUSTED_R1A_BASELINE["mean_delta_q_norm_by_timestep"],
            "difference": [
                None if current is None else current - baseline
                for current, baseline in zip(
                    current_delta_q,
                    TRUSTED_R1A_BASELINE["mean_delta_q_norm_by_timestep"],
                    strict=True,
                )
            ],
        },
        "selected_target_relative_gain_by_timestep": {
            "r1c1": current_selected_gain,
            "r1a": TRUSTED_R1A_BASELINE[
                "selected_target_relative_gain_by_timestep"
            ],
            "difference": [
                None if current is None else current - baseline
                for current, baseline in zip(
                    current_selected_gain,
                    TRUSTED_R1A_BASELINE[
                        "selected_target_relative_gain_by_timestep"
                    ],
                    strict=True,
                )
            ],
        },
        "between_candidate_support_cosine_by_timestep": {
            "r1c1": current_support_cosine,
            "r1a_static_reference": TRUSTED_R1A_BASELINE[
                "pairwise_support_cosine"
            ],
        },
        "mean_executed_edits": {
            "r1c1": selection["mean_executed_edit_count"],
            "r1a": TRUSTED_R1A_BASELINE["mean_executed_edits"],
            "difference": selection["mean_executed_edit_count"]
            - TRUSTED_R1A_BASELINE["mean_executed_edits"],
        },
        "repeated_candidate_trajectory_fraction": {
            "r1c1": selection[
                "fraction_queries_with_repeated_candidate_selections"
            ],
            "r1a": TRUSTED_R1A_BASELINE[
                "repeated_candidate_trajectory_fraction"
            ],
            "difference": selection[
                "fraction_queries_with_repeated_candidate_selections"
            ]
            - TRUSTED_R1A_BASELINE["repeated_candidate_trajectory_fraction"],
        },
        "interpretation_limit": (
            "historical comparison only; no value is used by training or selection"
        ),
    }
    metadata = checkpoint["metadata"]
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_metric": checkpoint.get("metric"),
        "checkpoint_model_config_provenance": checkpoint[
            "checkpoint_model_config_provenance"
        ],
        "checkpoint_replay_guard": _checkpoint_replay_guard(
            checkpoint,
            checkpoint["checkpoint_model_config_provenance"],
            float(global_controls["full"]["mean_recall"]),
        ),
        "backbone_metadata": {
            "checkpoint": metadata["backbone_checkpoint"],
            "revision": metadata["backbone_revision"],
            "training_precision": metadata.get("precision"),
        },
        "protocol": {
            "dataset": "FashionIQ",
            "split": SPLIT,
            "caption_policy": CAPTION_POLICY,
            "gallery_protocol": args.protocol,
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
        "intent_diagnostics": intent,
        "selection_diagnostics": selection,
        "grounding_diagnostics": grounding,
        "temporal_grounding_diagnostics": temporal_grounding,
        "visual_null_diagnostics": {
            **global_accumulator.visual_null_summary(),
            "offline_target_relative_utility": global_null_utility.summary(),
        },
        "functional_diagnostics": functional,
        "dynamic_diagnostics": dynamic,
        "control_retrieval_metrics": global_controls,
        "same_parent_counterfactual_diagnostics": global_controls[
            "counterfactual_same_parent_by_timestep"
        ],
        "selected_path_marginal_diagnostics": selected_path,
        "specialization_matrices": specialization,
        "failure_flags": _failure_flags(
            selection, grounding, functional, global_controls, specialization
        ),
        "diagnostic_definitions": diagnostic_definitions(),
        "trusted_r1a_baseline": {
            **TRUSTED_R1A_BASELINE,
            "status": "fixed historical comparison values; not a training objective",
            "automatic_comparison": r1a_comparison,
        },
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
