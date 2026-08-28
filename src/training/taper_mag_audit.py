from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from models.taper_mag.contracts import EncodedPolicyBatch, SupervisionBatch
from models.taper_mag.operator_generator import OperatorSet
from models.taper_mag.rollout import RolloutConfig, TaperOutput
from models.taper_mag.state import LocalState
from models.taper_mag.utility import HistoryState, append_stop
from training.marginal_gain_teacher import MarginalGainTeacher
from training.negative_bank import NegativeBank
from training.taper_mag_health import (
    dynamic_frozen_metrics,
    recursively_finite,
    repeat_staleness_metrics,
    response_effective_rank,
    utility_health_metrics,
)


def clone_operator_sets(
    operators: OperatorSet,
    best_indices: Tensor,
) -> dict[str, OperatorSet]:
    """Construct audit-only causal operator interventions before execution."""
    values = operators.operators
    if best_indices.shape != values.shape[:1]:
        raise ValueError("best_indices must be [B]")
    batch, count, width = values.shape
    selected = values.gather(
        1, best_indices[:, None, None].expand(batch, 1, width)
    )
    mean = values.mean(dim=1, keepdim=True)
    return {
        "clone_all_best": replace(
            operators, operators=selected.expand(-1, count, -1).clone()
        ),
        "clone_all_mean": replace(
            operators, operators=mean.expand(-1, count, -1).clone()
        ),
    }


def execute_operator_once(
    model: nn.Module,
    state: LocalState,
    operator: Tensor,
) -> LocalState:
    """Execute one [B,D] operator through the real state-dependent executor."""
    if operator.ndim != 2 or operator.shape[0] != state.local.shape[0]:
        raise ValueError("operator must be [B,D]")
    current = model.readout(state)
    features = model.executor.encode_state(state, current.context)
    candidate = model.executor.enumerate(state, features, operator[:, None])
    return state.with_local(candidate.local[:, 0])


def rollout_with_operator_set(
    model: nn.Module,
    state: LocalState,
    operators: OperatorSet,
    *,
    max_steps: int,
    step_cost: float,
) -> Tensor:
    """Target-free learned rollout under a causal replacement of operator anchors."""
    if not 1 <= max_steps <= 4:
        raise ValueError("audit rollout horizon must be in [1,4]")
    batch, tokens = state.local.shape[:2]
    history = HistoryState.initialize(
        batch, operators.operators.shape[1], tokens, state.local.device
    )
    stop_index = operators.operators.shape[1]
    for step in range(max_steps):
        active = state.alive
        current, candidates, _, predicted = model.preview(
            state,
            operators,
            history,
            step=step,
            max_steps=max_steps,
            detach_utility_inputs=True,
        )
        values = append_stop(predicted, step_cost, stop_allowed=step >= 1)
        actions = values.argmax(dim=-1)
        actions = torch.where(active, actions, torch.full_like(actions, stop_index))
        execute_mask = active & actions.ne(stop_index)
        features = model.executor.encode_state(state, current.context)
        selected, _ = model.executor.recompute_selected(
            state,
            features,
            operators.operators,
            actions,
            execute_mask,
        )
        state = selected.with_local(
            selected.local,
            alive=active & actions.ne(stop_index),
        )
        history = history.update(
            actions=actions,
            execute_mask=execute_mask,
            predicted_gain=predicted,
            candidates=candidates,
            step=step,
        )
    return model.readout(state).query


@torch.inference_mode()
def causal_operator_interventions(
    model: nn.Module,
    encoded: EncodedPolicyBatch,
    best_indices: Tensor,
    *,
    max_steps: int,
    step_cost: float = 0.0,
) -> dict[str, Tensor]:
    """Execute V4 repeat/clone/zero/mean controls in local-state space."""
    if model.training:
        raise RuntimeError("Causal operator interventions are audit-only and require eval()")
    _, initial, operators = model.prepare(encoded)
    values = operators.operators
    batch, _, width = values.shape
    selected = values.gather(
        1, best_indices[:, None, None].expand(batch, 1, width)
    ).squeeze(1)
    mean = values.mean(dim=1)
    zero = torch.zeros_like(mean)

    repeat_best_state = execute_operator_once(model, initial, selected)
    repeat_best_state = execute_operator_once(model, repeat_best_state, selected)
    mean_repeat_state = execute_operator_once(model, initial, mean)
    mean_repeat_state = execute_operator_once(model, mean_repeat_state, mean)
    operator_zero_state = execute_operator_once(model, initial, zero)
    operator_mean_state = execute_operator_once(model, initial, mean)
    cloned = clone_operator_sets(operators, best_indices)
    return {
        "repeat_best": model.readout(repeat_best_state).query,
        "mean_repeat": model.readout(mean_repeat_state).query,
        "clone_all_best": rollout_with_operator_set(
            model,
            initial,
            cloned["clone_all_best"],
            max_steps=max_steps,
            step_cost=step_cost,
        ),
        "clone_all_mean": rollout_with_operator_set(
            model,
            initial,
            cloned["clone_all_mean"],
            max_steps=max_steps,
            step_cost=step_cost,
        ),
        "operator_zero": model.readout(operator_zero_state).query,
        "operator_mean": model.readout(operator_mean_state).query,
    }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def module_fingerprint(*modules: nn.Module) -> str:
    digest = hashlib.sha256()
    for module_index, module in enumerate(modules):
        for name, value in sorted(module.state_dict().items()):
            digest.update(f"{module_index}:{name}".encode())
            tensor = value.detach().cpu().contiguous()
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def teacher_shadow_firewall_passes(firewall: dict[str, Any]) -> bool:
    return (
        firewall.get("policy_forward_target_argument_absent") is True
        and firewall.get("inference_without_supervision_succeeded") is True
        and firewall.get("target_shuffle_changed_teacher") is True
        and firewall.get("target_entered_policy_or_history") is False
    )


def validate_teacher_shadow_report(path: str | Path) -> str:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"Approved actor health gate requires teacher-shadow report: {source}"
        )
    report = json.loads(source.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "audit_kind",
        "sample_count",
        "parameter_updates",
        "numerical_health",
        "firewall",
        "cache_manifest_hashes",
    }
    missing = required - set(report)
    if missing:
        raise RuntimeError(f"Teacher-shadow report is missing fields: {sorted(missing)}")
    if report["schema_version"] != 1 or report["audit_kind"] != "teacher_shadow":
        raise RuntimeError("Invalid teacher-shadow report schema/kind")
    if int(report["sample_count"]) <= 0:
        raise RuntimeError("Teacher-shadow report contains no audited samples")
    if report["parameter_updates"].get("changed") or not report["numerical_health"].get("finite"):
        raise RuntimeError("Teacher-shadow report failed parameter/numerical invariants")
    if not teacher_shadow_firewall_passes(report["firewall"]):
        raise RuntimeError("Teacher-shadow report failed the target-firewall audit")
    return file_sha256(source)


def validate_teacher_shadow_provenance(
    report_path: str | Path,
    checkpoint_path: str | Path,
    cache_manifest_hashes: dict[str, str],
) -> None:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if report.get("model_checkpoint_sha256") != file_sha256(checkpoint_path):
        raise RuntimeError("Teacher-shadow report was not generated from the resumed checkpoint")
    if report.get("cache_manifest_hashes") != dict(sorted(cache_manifest_hashes.items())):
        raise RuntimeError("Teacher-shadow report cache manifests differ from the current run")


@dataclass(slots=True)
class TeacherShadowAuditor:
    model: nn.Module
    backbone_model: nn.Module
    negative_bank: NegativeBank
    teacher: MarginalGainTeacher
    seed: int
    near_tie_band: float = 0.0
    sample_count: int = 0
    _candidate_variance: list[Tensor] = field(default_factory=list)
    _teacher_gain: list[Tensor] = field(default_factory=list)
    _predicted_gain: list[Tensor] = field(default_factory=list)
    _random_gain: list[Tensor] = field(default_factory=list)
    _uniform_gain: list[Tensor] = field(default_factory=list)
    _learned_gain: list[Tensor] = field(default_factory=list)
    _response_rank: list[float] = field(default_factory=list)
    _clone_controls: list[dict[str, Any]] = field(default_factory=list)
    traces: list[dict[str, Any]] = field(default_factory=list)
    audited_sample_ids: list[str] = field(default_factory=list)
    _before: str = field(init=False)
    _target_shuffle_changed_teacher: bool = False

    def __post_init__(self) -> None:
        self._before = module_fingerprint(self.model, self.backbone_model)

    @staticmethod
    def _random_actions(sample_ids: tuple[str, ...], seed: int, count: int, device: torch.device) -> Tensor:
        values = []
        for sample_id in sample_ids:
            digest = hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()
            values.append(int.from_bytes(digest[:8], "big") % count)
        return torch.tensor(values, device=device, dtype=torch.long)

    @torch.inference_mode()
    def update(
        self,
        encoded: EncodedPolicyBatch,
        supervision: SupervisionBatch,
        *,
        sample_ids: tuple[str, ...],
        reference_ids: tuple[str, ...],
        modification_texts: tuple[str, ...],
        max_trace_samples: int = 8,
    ) -> None:
        _, state, operators = self.model.prepare(encoded)
        batch, tokens = state.local.shape[:2]
        history = HistoryState.initialize(
            batch, operators.operators.shape[1], tokens, state.local.device
        )
        current, candidates, candidate_readout, predicted = self.model.preview(
            state,
            operators,
            history,
            step=0,
            max_steps=1,
            detach_utility_inputs=True,
        )
        negatives = self.negative_bank.mine_once(current.query, supervision)
        labels = self.teacher.score(
            current.query, candidate_readout.query, supervision, negatives
        )
        if batch > 1:
            shuffled = SupervisionBatch(
                target_embedding=supervision.target_embedding.flip(0),
                target_ids=supervision.target_ids,
                positive_ids=supervision.positive_ids,
            )
            shuffled_labels = self.teacher.score(
                current.query, candidate_readout.query, shuffled, negatives
            )
            self._target_shuffle_changed_teacher |= not torch.allclose(
                labels.raw_gain, shuffled_labels.raw_gain
            )
        uniform_state = state.with_local(candidates.local.mean(dim=1))
        uniform_query = self.model.readout(uniform_state).query[:, None]
        uniform_gain = self.teacher.score(
            current.query, uniform_query, supervision, negatives
        ).raw_gain[:, 0]
        random_actions = self._random_actions(
            sample_ids, self.seed, candidates.local.shape[1], current.query.device
        )
        random_gain = labels.raw_gain.gather(1, random_actions[:, None]).squeeze(1)
        learned_gain = labels.raw_gain.gather(
            1, predicted.argmax(dim=-1, keepdim=True)
        ).squeeze(1)
        rank = response_effective_rank(current.query, candidate_readout.query)
        best = labels.raw_gain.argmax(dim=-1)
        causal_queries = causal_operator_interventions(
            self.model,
            encoded,
            best,
            max_steps=1,
        )
        control_names = tuple(causal_queries)
        intervention_gain = self.teacher.score(
            current.query,
            torch.stack([causal_queries[name] for name in control_names], dim=1),
            supervision,
            negatives,
        ).raw_gain.mean(dim=0)
        clones = {
            f"{name}_causal_teacher_gain": float(value)
            for name, value in zip(control_names, intervention_gain, strict=True)
        }
        self.sample_count += batch
        self.audited_sample_ids.extend(sample_ids)
        self._candidate_variance.append(
            candidate_readout.query.float().var(dim=1, unbiased=False).mean(dim=-1).cpu()
        )
        self._teacher_gain.append(labels.raw_gain.cpu())
        self._predicted_gain.append(predicted.cpu())
        self._random_gain.append(random_gain.cpu())
        self._uniform_gain.append(uniform_gain.cpu())
        self._learned_gain.append(learned_gain.cpu())
        self._response_rank.append(float(rank["mean_effective_rank"]))
        self._clone_controls.append(clones)
        remaining = max(0, max_trace_samples - len(self.traces))
        for index in range(min(batch, remaining)):
            self.traces.append(
                {
                    "policy": {
                        "sample_id": sample_ids[index],
                        "reference_id": reference_ids[index],
                        "modification_text_sha256": hashlib.sha256(
                            modification_texts[index].encode("utf-8")
                        ).hexdigest(),
                        "step": 0,
                        "alive": True,
                        "predicted_gains": predicted[index].float().cpu().tolist(),
                        "action_values": predicted[index].float().cpu().tolist() + [None],
                        "selected_action": int(predicted[index].argmax()),
                        "stop_allowed": False,
                        "support_mass": candidates.support[index].mean(dim=-1).float().cpu().tolist(),
                        "delta_norm": candidates.delta_norm[index].float().cpu().tolist(),
                    },
                    "supervision_audit": {
                        "target_id": supervision.target_ids[index],
                        "teacher_gains": labels.raw_gain[index].cpu().tolist(),
                        "negative_ids": list(labels.negative_ids[index]),
                    },
                }
            )

    def finalize(
        self,
        *,
        checkpoint_path: str | Path,
        cache_manifest_hashes: dict[str, str],
    ) -> dict[str, Any]:
        if self.sample_count == 0:
            raise RuntimeError("Teacher-shadow audit received no samples")
        teacher = torch.cat(self._teacher_gain)
        predicted = torch.cat(self._predicted_gain)
        random_gain = torch.cat(self._random_gain)
        uniform_gain = torch.cat(self._uniform_gain)
        learned_gain = torch.cat(self._learned_gain)
        sorted_gain = teacher.sort(dim=-1, descending=True).values
        utility = utility_health_metrics(
            predicted,
            teacher,
            near_tie_band=self.near_tie_band,
        )
        after = module_fingerprint(self.model, self.backbone_model)
        clone_keys = self._clone_controls[0]
        report: dict[str, Any] = {
            "schema_version": 1,
            "audit_kind": "teacher_shadow",
            "sample_count": self.sample_count,
            "seed": self.seed,
            "audit_subset": {
                "selection": "first_N_official_triplets_in_stable_dataset_order",
                "sample_ids_sha256": hashlib.sha256(
                    "\n".join(self.audited_sample_ids).encode("utf-8")
                ).hexdigest(),
            },
            "model_checkpoint_sha256": file_sha256(checkpoint_path),
            "cache_manifest_hashes": dict(sorted(cache_manifest_hashes.items())),
            "candidate_space": {
                "candidate_outcome_variance": float(torch.cat(self._candidate_variance).mean()),
                "oracle_positive_gain_rate": float((teacher.max(dim=-1).values > 0).float().mean()),
                "oracle_best_gain": float(teacher.max(dim=-1).values.mean()),
                "oracle_best_second_gap": float((sorted_gain[:, 0] - sorted_gain[:, 1]).mean()),
                "random_action_realized_gain": float(random_gain.mean()),
                "uniform_mean_realized_gain": float(uniform_gain.mean()),
                "learned_policy_realized_gain": float(learned_gain.mean()),
                "oracle_action_realized_gain": float(teacher.max(dim=-1).values.mean()),
                "oracle_vs_random_gap": float((teacher.max(dim=-1).values - random_gain).mean()),
                "oracle_vs_uniform_gap": float((teacher.max(dim=-1).values - uniform_gain).mean()),
            },
            "critic_shadow": utility,
            "response_rank": {
                "mean_effective_rank": sum(self._response_rank) / len(self._response_rank)
            },
            "clone_controls": {
                "execution_contract": "operator_to_executor_to_state_to_readout",
                "query_delta_arithmetic_used": False,
                **{
                    key: sum(float(item[key]) for item in self._clone_controls)
                    / len(self._clone_controls)
                    for key in clone_keys
                },
            },
            "parameter_updates": {
                "before_sha256": self._before,
                "after_sha256": after,
                "changed": self._before != after,
            },
            "firewall": {
                "policy_forward_target_argument_absent": True,
                "inference_without_supervision_succeeded": True,
                "target_shuffle_changed_teacher": self._target_shuffle_changed_teacher,
                "target_entered_policy_or_history": False,
            },
        }
        report["numerical_health"] = {
            "finite": recursively_finite(report),
            "nan_or_inf_count": 0 if recursively_finite(report) else 1,
        }
        return report


@torch.inference_mode()
def dynamic_frozen_audit(
    model: nn.Module,
    encoded: EncodedPolicyBatch,
    supervision: SupervisionBatch,
    negative_bank: NegativeBank,
    teacher: MarginalGainTeacher,
    *,
    max_steps: int = 4,
    step_cost: float = 0.0,
) -> dict[str, Any]:
    dynamic: TaperOutput = model(
        encoded,
        RolloutConfig(max_steps=max_steps, selection_mode="learned", step_cost=step_cost),
        detach_utility_inputs=True,
    )
    frozen: TaperOutput = model(
        encoded,
        RolloutConfig(max_steps=max_steps, selection_mode="frozen_order", step_cost=step_cost),
        detach_utility_inputs=True,
    )

    def labels(output: TaperOutput) -> Tensor:
        values = []
        for step in range(max_steps):
            negatives = negative_bank.mine_once(
                output.trace.current_queries[:, step], supervision
            )
            values.append(
                teacher.score(
                    output.trace.current_queries[:, step],
                    output.trace.candidate_queries[:, step],
                    supervision,
                    negatives,
                    step_cost=step_cost,
                ).raw_gain
            )
        return torch.stack(values, dim=1)

    dynamic_teacher = labels(dynamic)
    frozen_teacher = labels(frozen)

    def realized(output: TaperOutput, gains: Tensor) -> Tensor:
        selected = gains.gather(
            -1, output.trace.actions.clamp_max(gains.shape[-1] - 1).unsqueeze(-1)
        ).squeeze(-1)
        selected = torch.where(
            output.trace.actions.eq(gains.shape[-1]), torch.zeros_like(selected), selected
        )
        return (selected * output.trace.active).sum(dim=-1)

    metrics = dynamic_frozen_metrics(
        dynamic.trace.actions,
        frozen.trace.actions,
        dynamic.trace.action_values,
        frozen.trace.action_values,
        dynamic_retrieval=(dynamic.final_query * supervision.target_embedding).sum(dim=-1),
        frozen_retrieval=(frozen.final_query * supervision.target_embedding).sum(dim=-1),
        dynamic_realized_gain=realized(dynamic, dynamic_teacher),
        frozen_realized_gain=realized(frozen, frozen_teacher),
        stop_index=dynamic_teacher.shape[-1],
    )
    return {
        **metrics,
        "retrieval_metric": "normalized_target_cosine_audit_only",
        "critic": utility_health_metrics(
            dynamic.trace.predicted_gain,
            dynamic_teacher,
            active=dynamic.trace.active,
            step_cost=step_cost,
        ),
        "repeat": repeat_staleness_metrics(
            dynamic.trace.actions,
            dynamic_teacher,
            dynamic.trace.active,
            stop_index=dynamic_teacher.shape[-1],
        ),
    }


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_policy_traces(path: str | Path, traces: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as file:
        for trace in traces:
            file.write(json.dumps(trace, sort_keys=True, allow_nan=False) + "\n")


def mean_audit_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        return {"status": "not_run"}
    result: dict[str, Any] = {}
    for key in reports[0]:
        values = [report[key] for report in reports]
        if all(isinstance(value, dict) for value in values):
            result[key] = mean_audit_reports(values)
        elif all(isinstance(value, (int, float)) for value in values):
            result[key] = sum(float(value) for value in values) / len(values)
        else:
            result[key] = values[0]
    return result
