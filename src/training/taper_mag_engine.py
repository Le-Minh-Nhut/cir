from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor

from backbones.fgclip2_base import FGCLIP2BaseBackbone, TokenizedTextBatch
from models.taper_mag.contracts import EncodedPolicyBatch, PolicyBatch, SupervisionBatch
from models.taper_mag.model import TaperMAG
from models.taper_mag.rollout import RolloutConfig, TaperOutput
from training.marginal_gain_teacher import MarginalGainTeacher
from training.negative_bank import NegativeBank
from training.taper_mag_losses import (
    stop_anchored_listwise_utility_loss,
    terminal_bidirectional_infonce,
)


class CurriculumStage(str, Enum):
    ACTOR_WARMUP = "actor_warmup"
    UTILITY_SHADOW = "utility_shadow"
    CRITIC_WARMUP = "critic_warmup"
    DAGGER_T2 = "dagger_t2"
    ST_BRIDGE = "st_bridge"
    HARDEN = "harden"


@dataclass(frozen=True, slots=True)
class EngineConfig:
    stage: CurriculumStage = CurriculumStage.ACTOR_WARMUP
    horizon: int = 1
    step_cost: float = 0.0
    retrieval_temperature: float = 0.07
    utility_weight: float = 1.0
    oracle_mix: float = 0.0

    def rollout(self) -> RolloutConfig:
        if self.stage in {CurriculumStage.ACTOR_WARMUP, CurriculumStage.UTILITY_SHADOW}:
            return RolloutConfig(max_steps=1, selection_mode="uniform", step_cost=self.step_cost)
        if self.stage == CurriculumStage.CRITIC_WARMUP:
            return RolloutConfig(max_steps=1, selection_mode="uniform", step_cost=self.step_cost)
        return RolloutConfig(
            max_steps=self.horizon,
            selection_mode="learned",
            straight_through=self.stage == CurriculumStage.ST_BRIDGE,
            step_cost=self.step_cost,
        )


@dataclass(frozen=True, slots=True)
class TrainingStepOutput:
    loss: Tensor
    retrieval_loss: Tensor
    utility_loss: Tensor
    model_output: TaperOutput
    teacher_gain: Tensor


class TaperMAGTrainingEngine:
    """Training-only coordinator that owns targets, negatives, and detached teacher."""

    def __init__(
        self,
        backbone: FGCLIP2BaseBackbone,
        model: TaperMAG,
        negative_bank: NegativeBank,
        teacher: MarginalGainTeacher | None = None,
    ) -> None:
        self.backbone = backbone
        self.model = model
        self.negative_bank = negative_bank
        self.teacher = teacher or MarginalGainTeacher()

    def encode_policy(self, batch: PolicyBatch) -> EncodedPolicyBatch:
        tokenized = TokenizedTextBatch(
            batch.text_input_ids,
            batch.text_attention_mask,
            batch.text_content_mask,
        )
        text_tokens = self.backbone.encode_text_tokens(tokenized)
        return EncodedPolicyBatch(
            reference_local=batch.reference_local,
            reference_local_mask=batch.reference_local_mask,
            reference_global=batch.reference_global,
            text_tokens=text_tokens,
            text_attention_mask=batch.text_attention_mask,
            text_content_mask=batch.text_content_mask,
            spatial_shapes=batch.spatial_shapes,
        )

    def step(
        self,
        policy: PolicyBatch,
        supervision: SupervisionBatch,
        config: EngineConfig,
    ) -> TrainingStepOutput:
        encoded = self.encode_policy(policy)
        action_selector = None
        if config.stage == CurriculumStage.DAGGER_T2 and config.oracle_mix > 0:
            if not 0 <= config.oracle_mix <= 1:
                raise ValueError("oracle_mix must be in [0,1]")

            def dagger_selector(
                step: int,
                current_query: Tensor,
                candidate_queries: Tensor,
                predicted_gain: Tensor,
                predicted_values: Tensor,
            ) -> Tensor:
                del predicted_gain
                negatives = self.negative_bank.mine_once(current_query.detach(), supervision)
                labels = self.teacher.score(
                    current_query,
                    candidate_queries,
                    supervision,
                    negatives,
                    step_cost=config.step_cost,
                ).net_values
                if step == 0:
                    labels = labels.clone()
                    labels[:, -1] = torch.finfo(labels.dtype).min
                use_oracle = torch.rand(
                    labels.shape[0], device=labels.device
                ) < config.oracle_mix
                return torch.where(use_oracle[:, None], labels, predicted_values)

            action_selector = dagger_selector
        output = self.model.rollout_training(
            encoded,
            config.rollout(),
            action_selector=action_selector,
            detach_utility_inputs=True,
        )
        retrieval_loss = terminal_bidirectional_infonce(
            output.final_query,
            supervision.target_embedding,
            supervision.target_ids,
            supervision.positive_ids,
            temperature=config.retrieval_temperature,
        )
        teacher_steps: list[Tensor] = []
        utility_terms: list[Tensor] = []
        for step in range(output.trace.current_queries.shape[1]):
            negatives = self.negative_bank.mine_once(
                output.trace.current_queries[:, step].detach(), supervision
            )
            labels = self.teacher.score(
                output.trace.current_queries[:, step],
                output.trace.candidate_queries[:, step],
                supervision,
                negatives,
                step_cost=config.step_cost,
            )
            teacher_steps.append(labels.raw_gain)
            utility_terms.append(
                stop_anchored_listwise_utility_loss(
                    output.trace.predicted_gain[:, step],
                    labels.raw_gain,
                    step_cost=config.step_cost,
                )
            )
        utility_loss = torch.stack(utility_terms).mean()
        if config.stage in {CurriculumStage.ACTOR_WARMUP, CurriculumStage.UTILITY_SHADOW}:
            utility_coefficient = 0.0
        else:
            utility_coefficient = config.utility_weight
        loss = retrieval_loss + utility_coefficient * utility_loss
        return TrainingStepOutput(
            loss, retrieval_loss, utility_loss, output, torch.stack(teacher_steps, dim=1)
        )
