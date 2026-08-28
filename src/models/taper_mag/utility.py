from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from models.taper_mag.executor import CandidateBatch
from models.taper_mag.operator_generator import OperatorSet
from models.taper_mag.readout import CandidateReadoutOutput, ReadoutOutput


@dataclass(frozen=True, slots=True)
class HistoryState:
    use_count: Tensor
    last_used_step: Tensor
    last_predicted_gain: Tensor
    last_delta_norm: Tensor
    last_support_mass: Tensor
    last_support: Tensor
    previous_action: Tensor
    valid: Tensor

    @classmethod
    def initialize(
        cls, batch_size: int, num_queries: int, num_tokens: int, device: torch.device
    ) -> HistoryState:
        zeros = torch.zeros(batch_size, num_queries, device=device)
        return cls(
            use_count=zeros.clone(),
            last_used_step=torch.full_like(zeros, -1.0),
            last_predicted_gain=zeros.clone(),
            last_delta_norm=zeros.clone(),
            last_support_mass=zeros.clone(),
            last_support=torch.zeros(batch_size, num_queries, num_tokens, device=device),
            previous_action=torch.full(
                (batch_size,), num_queries, device=device, dtype=torch.long
            ),
            valid=torch.zeros_like(zeros, dtype=torch.bool),
        )

    def features(self, support: Tensor, *, step: int, max_steps: int) -> Tensor:
        previous = torch.nn.functional.one_hot(
            self.previous_action.clamp_max(self.use_count.shape[1] - 1),
            num_classes=self.use_count.shape[1],
        ).to(self.use_count.dtype)
        previous = previous * (self.previous_action[:, None] < self.use_count.shape[1])
        steps_since = torch.where(
            self.last_used_step >= 0,
            (float(step) - self.last_used_step).clamp_min(0) / max_steps,
            torch.zeros_like(self.last_used_step),
        )
        overlap = (support * self.last_support).sum(dim=-1) / (
            support.sum(dim=-1).clamp_min(1e-8)
        )
        timestep = torch.full_like(self.use_count, float(step) / max_steps)
        return torch.stack(
            [
                self.use_count / max_steps,
                previous,
                steps_since,
                self.last_predicted_gain,
                self.last_delta_norm,
                self.last_support_mass,
                overlap,
                timestep,
                self.valid.to(self.use_count.dtype),
            ],
            dim=-1,
        )

    def update(
        self,
        *,
        actions: Tensor,
        execute_mask: Tensor,
        predicted_gain: Tensor,
        candidates: CandidateBatch,
        step: int,
    ) -> HistoryState:
        num_queries = self.use_count.shape[1]
        selected = torch.nn.functional.one_hot(
            actions.clamp_max(num_queries - 1), num_classes=num_queries
        ).bool()
        selected = selected & execute_mask[:, None]
        use_count = self.use_count + selected.to(self.use_count.dtype)
        step_value = torch.full_like(self.last_used_step, float(step))
        last_used = torch.where(selected, step_value, self.last_used_step)
        support_mass = candidates.support.mean(dim=-1)
        alive = execute_mask[:, None]
        return HistoryState(
            use_count=use_count,
            last_used_step=last_used,
            last_predicted_gain=torch.where(alive, predicted_gain.detach(), self.last_predicted_gain),
            last_delta_norm=torch.where(alive, candidates.delta_norm.detach(), self.last_delta_norm),
            last_support_mass=torch.where(alive, support_mass.detach(), self.last_support_mass),
            last_support=torch.where(
                alive.unsqueeze(-1), candidates.support.detach(), self.last_support
            ),
            previous_action=torch.where(execute_mask, actions, self.previous_action),
            valid=self.valid | alive,
        )


class TargetFreeUtilityCritic(nn.Module):
    def __init__(self, d_model: int = 256, history_dim: int = 9) -> None:
        super().__init__()
        self.history_projection = nn.Sequential(
            nn.LayerNorm(history_dim),
            nn.Linear(history_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.input_norm = nn.LayerNorm(10 * d_model)
        self.mlp = nn.Sequential(
            nn.Linear(10 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def build_features(
        self,
        current: ReadoutOutput,
        candidates: CandidateBatch,
        candidate_readout: CandidateReadoutOutput,
        operators: OperatorSet,
        history: HistoryState,
        *,
        step: int,
        max_steps: int,
    ) -> Tensor:
        num_queries = operators.operators.shape[1]
        current_internal = current.internal[:, None, :].expand(-1, num_queries, -1)
        mean_text = operators.text_reads.mean(dim=1, keepdim=True).expand(-1, num_queries, -1)
        global_delta = candidate_readout.internal - current_internal
        history_embedding = self.history_projection(
            history.features(candidates.support, step=step, max_steps=max_steps)
        )
        return torch.cat(
            [
                current_internal,
                operators.operators,
                operators.text_reads,
                operators.visual_reads,
                mean_text,
                candidates.support_context,
                global_delta,
                operators.operators * candidates.support_context,
                operators.text_reads * operators.visual_reads,
                history_embedding,
            ],
            dim=-1,
        )

    def forward(self, features: Tensor, *, detach_inputs: bool = False) -> Tensor:
        if detach_inputs:
            features = features.detach()
        return self.mlp(self.input_norm(features)).squeeze(-1)


def append_stop(values: Tensor, step_cost: float, *, stop_allowed: bool) -> Tensor:
    action_values = values - step_cost
    stop = torch.zeros((*values.shape[:-1], 1), device=values.device, dtype=values.dtype)
    result = torch.cat([action_values, stop], dim=-1)
    if not stop_allowed:
        result = result.clone()
        result[..., -1] = torch.finfo(result.dtype).min
    return result
