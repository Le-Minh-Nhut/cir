from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn

from models.taper_mag.contracts import EncodedPolicyBatch
from models.taper_mag.executor import CandidateBatch, SharedLocalExecutor
from models.taper_mag.operator_generator import CandidateOperatorGenerator, OperatorSet
from models.taper_mag.readout import CandidateReadoutOutput, ChangeAwareReadout, ReadoutOutput
from models.taper_mag.rollout import RolloutConfig, RolloutTrace, TaperOutput
from models.taper_mag.state import InputAdapters, LocalState, ProjectedInputs
from models.taper_mag.utility import HistoryState, TargetFreeUtilityCritic, append_stop


def candidate_mixture(
    candidate_local: Tensor,
    action_values: Tensor,
    *,
    mode: str,
    temperature: float,
) -> tuple[Tensor, Tensor]:
    """Mix K real actions; STOP is intentionally absent from one-step mixtures."""
    if candidate_local.ndim != 4 or action_values.shape != candidate_local.shape[:2]:
        raise ValueError("candidate mixture expects [B,K,N,D] states and [B,K] values")
    if mode == "uniform":
        weights = torch.full_like(action_values, 1.0 / action_values.shape[-1])
    elif mode == "soft":
        if temperature <= 0:
            raise ValueError("soft candidate-mixture temperature must be positive")
        weights = torch.softmax(action_values / temperature, dim=-1)
    else:
        raise ValueError(f"candidate mixture does not support mode={mode}")
    return torch.einsum("bk,bknd->bnd", weights, candidate_local), weights


@dataclass(frozen=True, slots=True)
class TaperMAGConfig:
    text_dim: int = 768
    vision_dim: int = 768
    retrieval_dim: int = 768
    d_model: int = 256
    num_queries: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    max_steps: int = 4

    def validate(self) -> None:
        if self.d_model != 256:
            raise ValueError("This experiment fixes TAPER internal width d_model=256")
        if self.num_queries != 4:
            raise ValueError("This experiment fixes num_queries=4")
        if self.d_model // self.num_heads != 32:
            raise ValueError("Canonical attention requires 8 heads with head_dim=32")
        if not 1 <= self.max_steps <= 4:
            raise ValueError("max_steps must be in [1,4]")


class TaperMAG(nn.Module):
    """Target-free inference graph. Teachers/targets are intentionally absent."""

    def __init__(self, config: TaperMAGConfig | None = None) -> None:
        super().__init__()
        self.config = config or TaperMAGConfig()
        self.config.validate()
        cfg = self.config
        self.adapters = InputAdapters(
            text_dim=cfg.text_dim,
            vision_dim=cfg.vision_dim,
            retrieval_dim=cfg.retrieval_dim,
            d_model=cfg.d_model,
        )
        self.operator_generator = CandidateOperatorGenerator(
            d_model=cfg.d_model,
            num_queries=cfg.num_queries,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
        )
        self.readout = ChangeAwareReadout(cfg.d_model, cfg.retrieval_dim)
        self.executor = SharedLocalExecutor(cfg.d_model)
        self.utility = TargetFreeUtilityCritic(cfg.d_model)

    def prepare(
        self, batch: EncodedPolicyBatch
    ) -> tuple[ProjectedInputs, LocalState, OperatorSet]:
        batch.validate(
            self.config.text_dim, self.config.vision_dim, self.config.retrieval_dim
        )
        projected = self.adapters(batch)
        state = self.adapters.initialize_state(projected)
        operators = self.operator_generator(
            projected.text,
            batch.text_content_mask.bool(),
            projected.initial_local,
            projected.local_mask,
        )
        return projected, state, operators

    def preview(
        self,
        state: LocalState,
        operators: OperatorSet,
        history: HistoryState,
        *,
        step: int,
        max_steps: int,
        detach_utility_inputs: bool = False,
    ) -> tuple[ReadoutOutput, CandidateBatch, CandidateReadoutOutput, Tensor]:
        current = self.readout(state)
        state_features = self.executor.encode_state(state, current.context)
        candidates = self.executor.enumerate(state, state_features, operators.operators)
        candidate_readout = self.readout.forward_candidates(state, candidates)
        utility_features = self.utility.build_features(
            current,
            candidates,
            candidate_readout,
            operators,
            history,
            step=step,
            max_steps=max_steps,
            detach_inputs=detach_utility_inputs,
        )
        predicted_gain = self.utility(utility_features, detach_inputs=False)
        return current, candidates, candidate_readout, predicted_gain

    def preview_detached_actor(
        self,
        state: LocalState,
        operators: OperatorSet,
        history: HistoryState,
        *,
        step: int,
        max_steps: int,
        detach_utility_inputs: bool = False,
    ) -> tuple[ReadoutOutput, CandidateBatch, CandidateReadoutOutput, Tensor]:
        """Preview all K transitions without retaining actor activation graphs."""
        with torch.no_grad():
            current = self.readout(state)
            state_features = self.executor.encode_state(state, current.context)
            candidates = self.executor.enumerate(state, state_features, operators.operators)
            candidate_readout = self.readout.forward_candidates(state, candidates)
        utility_features = self.utility.build_features(
            current,
            candidates,
            candidate_readout,
            operators,
            history,
            step=step,
            max_steps=max_steps,
            detach_inputs=detach_utility_inputs,
        )
        predicted_gain = self.utility(utility_features, detach_inputs=False)
        return current, candidates, candidate_readout, predicted_gain

    def forward(
        self,
        batch: EncodedPolicyBatch,
        rollout: RolloutConfig | None = None,
        *,
        detach_utility_inputs: bool = False,
    ) -> TaperOutput:
        return self._rollout(
            batch,
            rollout,
            detach_utility_inputs=detach_utility_inputs,
            training_action_selector=None,
        )

    def rollout_training(
        self,
        batch: EncodedPolicyBatch,
        rollout: RolloutConfig,
        *,
        action_selector: Callable[
            [int, Tensor, Tensor, Tensor, Tensor], Tensor
        ]
        | None = None,
        detach_utility_inputs: bool = True,
    ) -> TaperOutput:
        """Training-only rollout hook for external DAgger/oracle roll-in.

        The callback receives target-free tensors. The training engine may judge those tensors
        with supervision, but targets never enter this module or its state/critic features.
        """
        return self._rollout(
            batch,
            rollout,
            detach_utility_inputs=detach_utility_inputs,
            training_action_selector=action_selector,
        )

    def _rollout(
        self,
        batch: EncodedPolicyBatch,
        rollout: RolloutConfig | None,
        *,
        detach_utility_inputs: bool,
        training_action_selector: Callable[
            [int, Tensor, Tensor, Tensor, Tensor], Tensor
        ]
        | None,
    ) -> TaperOutput:
        rollout = rollout or RolloutConfig(max_steps=self.config.max_steps)
        rollout.validate()
        _, state, operators = self.prepare(batch)
        batch_size, num_tokens = state.local.shape[:2]
        history = HistoryState.initialize(
            batch_size,
            self.config.num_queries,
            num_tokens,
            state.local.device,
        )
        actions_trace: list[Tensor] = []
        active_trace: list[Tensor] = []
        predicted_trace: list[Tensor] = []
        values_trace: list[Tensor] = []
        current_query_trace: list[Tensor] = []
        candidate_query_trace: list[Tensor] = []
        support_trace: list[Tensor] = []
        delta_trace: list[Tensor] = []
        frozen_values: Tensor | None = None
        stop_index = self.config.num_queries

        for step in range(rollout.max_steps):
            active_before = state.alive
            hard_two_pass = (
                rollout.selection_mode not in {"uniform", "soft"}
                and not rollout.straight_through
            )
            preview_fn = self.preview_detached_actor if hard_two_pass else self.preview
            current, candidates, candidate_readout, predicted_gain = preview_fn(
                state,
                operators,
                history,
                step=step,
                max_steps=rollout.max_steps,
                detach_utility_inputs=detach_utility_inputs,
            )
            stop_allowed = step >= rollout.min_steps
            dynamic_values = append_stop(
                predicted_gain, rollout.step_cost, stop_allowed=stop_allowed
            )
            if rollout.selection_mode == "frozen_order":
                if frozen_values is None:
                    frozen_values = dynamic_values
                selection_values = frozen_values
            else:
                selection_values = dynamic_values

            if training_action_selector is not None and rollout.selection_mode not in {
                "uniform",
                "soft",
            }:
                selection_values = training_action_selector(
                    step,
                    current.query,
                    candidate_readout.query,
                    predicted_gain,
                    selection_values,
                )
                if selection_values.shape != dynamic_values.shape:
                    raise ValueError("Training action selector must return [B,K+1] values")

            if rollout.selection_mode in {"uniform", "soft"}:
                real_action_values = predicted_gain - rollout.step_cost
                mixed_local, mixture_weights = candidate_mixture(
                    candidates.local,
                    real_action_values,
                    mode=rollout.selection_mode,
                    temperature=rollout.selection_temperature,
                )
                actions = mixture_weights.argmax(dim=-1)
                state = state.with_local(
                    torch.where(active_before[:, None, None], mixed_local, state.local),
                    alive=active_before,
                )
                execute_mask = active_before
            else:
                actions = selection_values.argmax(dim=-1)
                if self.training and rollout.exploration_probability > 0:
                    top_two = selection_values.topk(k=2, dim=-1).indices
                    explore = torch.rand(
                        batch_size, device=state.local.device
                    ) < rollout.exploration_probability
                    actions = torch.where(explore & active_before, top_two[:, 1], actions)
                actions = torch.where(
                    active_before,
                    actions,
                    torch.full_like(actions, stop_index),
                )
                execute_mask = active_before & actions.ne(stop_index)
                if rollout.straight_through:
                    scaled_values = selection_values.detach() + rollout.rho_gate * (
                        selection_values - selection_values.detach()
                    )
                    probabilities = torch.softmax(
                        scaled_values / rollout.selection_temperature, dim=-1
                    )
                    hard = torch.nn.functional.one_hot(
                        actions, num_classes=stop_index + 1
                    ).to(probabilities.dtype)
                    weights = probabilities + (hard - probabilities).detach()
                    options = torch.cat([candidates.local, state.local[:, None]], dim=1)
                    selected_local = torch.einsum("bq,bqnd->bnd", weights, options)
                    state = state.with_local(
                        torch.where(
                            active_before[:, None, None], selected_local, state.local
                        ),
                        alive=active_before & actions.ne(stop_index),
                    )
                else:
                    # Recompute only k* with gradients from the same, still-immutable parent.
                    recompute_current = self.readout(state)
                    recompute_features = self.executor.encode_state(
                        state, recompute_current.context
                    )
                    selected_state, _ = self.executor.recompute_selected(
                        state,
                        recompute_features,
                        operators.operators,
                        actions,
                        execute_mask,
                    )
                    state = selected_state.with_local(
                        selected_state.local,
                        alive=active_before & actions.ne(stop_index),
                    )

            history = history.update(
                actions=actions,
                execute_mask=execute_mask,
                predicted_gain=predicted_gain,
                candidates=candidates,
                step=step,
            )
            actions_trace.append(actions)
            active_trace.append(active_before)
            predicted_trace.append(predicted_gain)
            values_trace.append(dynamic_values)
            current_query_trace.append(current.query)
            candidate_query_trace.append(candidate_readout.query)
            support_trace.append(candidates.support.mean(dim=-1))
            delta_trace.append(candidates.delta_norm)

        final = self.readout(state)
        trace = RolloutTrace(
            actions=torch.stack(actions_trace, dim=1),
            active=torch.stack(active_trace, dim=1),
            predicted_gain=torch.stack(predicted_trace, dim=1),
            action_values=torch.stack(values_trace, dim=1),
            current_queries=torch.stack(current_query_trace, dim=1),
            candidate_queries=torch.stack(candidate_query_trace, dim=1),
            support_mass=torch.stack(support_trace, dim=1),
            delta_norm=torch.stack(delta_trace, dim=1),
        )
        diagnostics = {
            "text_attention": operators.text_attention,
            "visual_attention": operators.visual_attention,
            "edit_gate_mean": operators.edit_gates.mean(),
            "edit_gate_std": operators.edit_gates.std(),
            "edit_gate_saturation": (
                (operators.edit_gates < 0.01) | (operators.edit_gates > 0.99)
            ).float().mean(),
            "query_cosine": torch.nn.functional.normalize(
                operators.text_reads, dim=-1
            )
            @ torch.nn.functional.normalize(operators.text_reads, dim=-1).transpose(-1, -2),
            "operator_cosine": torch.nn.functional.normalize(
                operators.operators, dim=-1
            )
            @ torch.nn.functional.normalize(operators.operators, dim=-1).transpose(-1, -2),
        }
        return TaperOutput(final_query=final.query, trace=trace, diagnostics=diagnostics)
