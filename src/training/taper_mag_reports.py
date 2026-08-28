from __future__ import annotations

import inspect
import hashlib
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from models.taper_mag.rollout import TaperOutput
from training.taper_mag_diagnostics import summarize_training_diagnostics
from training.taper_mag_health import QueryGradientTracker, recursively_finite, utility_health_metrics


@dataclass(slots=True)
class GradientRuntimeTracker:
    num_queries: int
    query: QueryGradientTracker = field(init=False)
    updates: int = 0
    clipped_updates: int = 0
    module_norm_sums: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.query = QueryGradientTracker(self.num_queries)

    def update(
        self,
        model: nn.Module,
        backbone: nn.Module,
        *,
        pre_clip_global_norm: float,
        clip_threshold: float,
    ) -> None:
        self.query.update(model.operator_generator.text_reader.queries)
        modules = {
            "text_reader": model.operator_generator.text_reader,
            "grounding": model.operator_generator.grounding,
            "operator_fusion": model.operator_generator.fusion,
            "executor": model.executor,
            "readout": model.readout,
            "utility": model.utility,
            "trainable_text": backbone,
        }
        for name, module in modules.items():
            squared = sum(
                float(parameter.grad.detach().float().square().sum())
                for parameter in module.parameters()
                if parameter.requires_grad and parameter.grad is not None
            )
            self.module_norm_sums[name] = self.module_norm_sums.get(name, 0.0) + squared**0.5
        self.updates += 1
        self.clipped_updates += int(pre_clip_global_norm > clip_threshold)

    def report(self) -> dict[str, Any]:
        denominator = max(self.updates, 1)
        return {
            **self.query.report(),
            "major_module_gradient_norm_mean": {
                name: value / denominator for name, value in sorted(self.module_norm_sums.items())
            },
            "fraction_updates_clipped": self.clipped_updates / denominator,
            "clip_observation_count": self.updates,
        }


@dataclass(slots=True)
class EpochHealthAccumulator:
    near_tie_band: float
    step_cost: float
    calibration_bins: int = 5
    scalar_sums: dict[str, float] = field(default_factory=dict)
    batches: int = 0
    predicted: list[Tensor] = field(default_factory=list)
    teacher: list[Tensor] = field(default_factory=list)
    active: list[Tensor] = field(default_factory=list)

    def update(self, output: TaperOutput, teacher_gain: Tensor) -> None:
        scalars = summarize_training_diagnostics(
            output,
            teacher_gain,
            near_tie_band=self.near_tie_band,
            step_cost=self.step_cost,
            calibration_bins=self.calibration_bins,
        )
        for name, value in scalars.items():
            self.scalar_sums[name] = self.scalar_sums.get(name, 0.0) + float(value)
        self.batches += 1
        self.predicted.append(output.trace.predicted_gain.detach().float().cpu())
        self.teacher.append(teacher_gain.detach().float().cpu())
        self.active.append(output.trace.active.detach().cpu())

    def report(self) -> tuple[dict[str, float], dict[str, Any]]:
        if self.batches == 0:
            raise RuntimeError("Epoch health report requires at least one batch")
        actor = {
            name: value / self.batches for name, value in sorted(self.scalar_sums.items())
        }
        utility = utility_health_metrics(
            torch.cat(self.predicted),
            torch.cat(self.teacher),
            active=torch.cat(self.active),
            near_tie_band=self.near_tie_band,
            step_cost=self.step_cost,
            calibration_bins=self.calibration_bins,
        )
        return actor, utility


def static_firewall_report(model: nn.Module) -> dict[str, Any]:
    parameters = inspect.signature(model.forward).parameters
    state_keys = tuple(model.state_dict())
    forbidden = ("teacher", "target", "negative_bank", "oracle")
    return {
        "pass": "target" not in parameters
        and not any(token in key for key in state_keys for token in forbidden),
        "forward_parameters": list(parameters),
        "target_argument_absent": "target" not in parameters,
        "teacher_or_target_state_absent": not any(
            token in key for key in state_keys for token in forbidden
        ),
        "behavioral_target_shuffle": "covered_by_teacher_shadow_and_unit_test",
        "supervision_in_policy_history": False,
    }


def build_functional_health_report(
    *,
    epoch: int,
    phase: str,
    retrieval: dict[str, float],
    actor: dict[str, Any],
    critic: dict[str, Any],
    gradients: dict[str, Any],
    firewall: dict[str, Any],
    dynamic_policy: dict[str, Any] | None = None,
    repeat: dict[str, Any] | None = None,
    clone_controls: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidate_variance = float(actor.get("candidate_outcome_variance", 0.0))
    report: dict[str, Any] = {
        "schema_version": 1,
        "epoch": epoch,
        "phase": phase,
        "retrieval": retrieval,
        "actor": {**actor, "gradient_health": gradients},
        "candidate_space": {
            "candidate_outcome_variance": candidate_variance,
            "oracle_positive_gain_rate": actor.get("oracle_positive_gain_rate"),
            "oracle_best_gain": actor.get("oracle_best_gain"),
            "oracle_best_second_gap": actor.get("oracle_best_second_gap"),
            "catastrophic_exact_collapse": candidate_variance == 0.0,
        },
        "critic": critic,
        "stop": {
            key: value for key, value in actor.items() if key.startswith("stop_fraction_")
        }
        | {
            "false_stop_rate": critic.get("false_stop_rate"),
            "false_continue_rate": critic.get("false_continue_rate"),
        },
        "dynamic_policy": dynamic_policy or {"valid": True, "status": "not_applicable_or_not_audited"},
        "repeat": repeat or {
            "valid": True,
            "repeat_frequency": actor.get("repeat_action_frequency"),
            "status": "recomputed_staleness_requires_multistep_audit",
        },
        "response_rank": {
            "mean_effective_rank": actor.get("response_effective_rank", 0.0)
        },
        "clone_controls": clone_controls or {"status": "teacher_shadow_audit_required"},
        "firewall": firewall,
    }
    report["numerical_health"] = {
        "pass": recursively_finite(report),
        "nan_or_inf_count": 0 if recursively_finite(report) else 1,
        "state_norm": actor.get("state_norm"),
        "delta_norm": actor.get("delta_norm"),
        "support_mass": actor.get("support_mass"),
        "support_saturation": actor.get("support_saturation"),
        "edit_gate_saturation": actor.get("edit_gate_saturation"),
        "fraction_updates_clipped": gradients.get("fraction_updates_clipped"),
    }
    return report


def sampled_policy_trace_records(
    *,
    sample_ids: tuple[str, ...],
    reference_ids: tuple[str, ...],
    target_ids: tuple[str, ...],
    modification_texts: tuple[str, ...],
    output: TaperOutput,
    teacher_gain: Tensor,
    negative_ids: tuple[tuple[tuple[str, ...], ...], ...],
    limit: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    count = min(limit, len(sample_ids))
    for sample in range(count):
        trajectory = []
        supervision_audit = []
        for step in range(output.trace.actions.shape[1]):
            trajectory.append(
                {
                    "step": step,
                    "alive": bool(output.trace.active[sample, step]),
                    "predicted_gains": output.trace.predicted_gain[sample, step]
                    .detach()
                    .float()
                    .cpu()
                    .tolist(),
                    "action_values": [
                        (float(value) if torch.isfinite(value) else None)
                        for value in output.trace.action_values[sample, step]
                        .detach()
                        .float()
                        .cpu()
                    ],
                    "selected_action": int(output.trace.actions[sample, step]),
                    "stop_allowed": step >= 1,
                    "support_mass": output.trace.support_mass[sample, step]
                    .detach()
                    .float()
                    .cpu()
                    .tolist(),
                    "delta_norm": output.trace.delta_norm[sample, step]
                    .detach()
                    .float()
                    .cpu()
                    .tolist(),
                }
            )
            supervision_audit.append(
                {
                    "step": step,
                    "teacher_gains": teacher_gain[sample, step]
                    .detach()
                    .float()
                    .cpu()
                    .tolist(),
                    "negative_ids": list(negative_ids[step][sample]),
                }
            )
        records.append(
            {
                "policy": {
                    "sample_id": sample_ids[sample],
                    "reference_id": reference_ids[sample],
                    "modification_text_sha256": hashlib.sha256(
                        modification_texts[sample].encode("utf-8")
                    ).hexdigest(),
                    "trajectory": trajectory,
                },
                "supervision_audit": {
                    "target_id": target_ids[sample],
                    "steps": supervision_audit,
                },
            }
        )
    return records
