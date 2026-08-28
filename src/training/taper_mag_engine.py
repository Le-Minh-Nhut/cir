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
    PREDICTED_T4 = "predicted_t4"
    HARDEN = "harden"


@dataclass(frozen=True, slots=True)
class EngineConfig:
    stage: CurriculumStage = CurriculumStage.ACTOR_WARMUP
    horizon: int = 1
    step_cost: float = 0.0
    retrieval_temperature: float = 0.07
    utility_weight: float = 1.0
    oracle_mix: float = 0.0
    straight_through: bool | None = None
    selection_temperature: float = 1.0
    rho_gate: float = 0.0
    exploration_probability: float = 0.0

    def validate(self) -> None:
        if not 1 <= self.horizon <= 4:
            raise ValueError("Curriculum horizon must be in [1,4]")
        if self.retrieval_temperature <= 0:
            raise ValueError("retrieval_temperature must be positive")
        if self.utility_weight < 0:
            raise ValueError("utility_weight must be non-negative")
        if not 0 <= self.oracle_mix <= 1:
            raise ValueError("oracle_mix must be in [0,1]")
        if self.selection_temperature <= 0:
            raise ValueError("selection_temperature must be positive")
        if not 0 <= self.rho_gate <= 1:
            raise ValueError("rho_gate must be in [0,1]")
        if not 0 <= self.exploration_probability <= 1:
            raise ValueError("exploration_probability must be in [0,1]")
        one_step = {
            CurriculumStage.ACTOR_WARMUP,
            CurriculumStage.UTILITY_SHADOW,
            CurriculumStage.CRITIC_WARMUP,
        }
        if self.stage in one_step and self.horizon != 1:
            raise ValueError(f"{self.stage.value} requires horizon=1")
        if self.stage == CurriculumStage.DAGGER_T2 and self.horizon != 2:
            raise ValueError("dagger_t2 requires horizon=2")
        if self.stage == CurriculumStage.ST_BRIDGE and self.horizon < 2:
            raise ValueError("st_bridge requires horizon>=2")
        if self.stage == CurriculumStage.PREDICTED_T4 and self.horizon != 4:
            raise ValueError("predicted_t4 requires horizon=4")
        if self.stage not in {CurriculumStage.DAGGER_T2, CurriculumStage.ST_BRIDGE} and self.oracle_mix != 0:
            raise ValueError(f"{self.stage.value} requires oracle_mix=0")
        expected_st = self.stage == CurriculumStage.ST_BRIDGE
        if self.straight_through is not None and self.straight_through != expected_st:
            raise ValueError(
                f"{self.stage.value} requires straight_through={str(expected_st).lower()}"
            )
        if self.exploration_probability > 0 and self.stage != CurriculumStage.PREDICTED_T4:
            raise ValueError("top-2 exploration is allowed only in predicted_t4")

    def rollout(self) -> RolloutConfig:
        self.validate()
        if self.stage in {CurriculumStage.ACTOR_WARMUP, CurriculumStage.UTILITY_SHADOW}:
            result = RolloutConfig(max_steps=1, selection_mode="uniform", step_cost=self.step_cost)
        elif self.stage == CurriculumStage.CRITIC_WARMUP:
            result = RolloutConfig(
                max_steps=1,
                selection_mode="soft",
                step_cost=self.step_cost,
                selection_temperature=self.selection_temperature,
            )
        else:
            result = RolloutConfig(
                max_steps=self.horizon,
                selection_mode="learned",
                straight_through=self.stage == CurriculumStage.ST_BRIDGE,
                step_cost=self.step_cost,
                selection_temperature=self.selection_temperature,
                rho_gate=self.rho_gate,
                exploration_probability=self.exploration_probability,
            )
        if self.stage == CurriculumStage.HARDEN:
            if (
                result.selection_mode != "learned"
                or result.straight_through
                or self.oracle_mix != 0
                or result.exploration_probability != 0
            ):
                raise AssertionError(
                    "HARDEN must be learned hard rollout without ST/oracle/exploration"
                )
        return result


@dataclass(frozen=True, slots=True)
class TrainingStepOutput:
    loss: Tensor
    retrieval_loss: Tensor
    utility_loss: Tensor
    model_output: TaperOutput
    teacher_gain: Tensor


def encode_policy_batch(
    backbone: FGCLIP2BaseBackbone, batch: PolicyBatch
) -> EncodedPolicyBatch:
    """Online target-free text encoding usable without a training engine/supervision."""
    tokenized = TokenizedTextBatch(
        batch.text_input_ids,
        batch.text_attention_mask,
        batch.text_content_mask,
    )
    text_tokens = backbone.encode_text_tokens(tokenized)
    return EncodedPolicyBatch(
        reference_local=batch.reference_local,
        reference_local_mask=batch.reference_local_mask,
        reference_global=batch.reference_global,
        text_tokens=text_tokens,
        text_attention_mask=batch.text_attention_mask,
        text_content_mask=batch.text_content_mask,
        spatial_shapes=batch.spatial_shapes,
    )


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
        return encode_policy_batch(self.backbone, batch)

    def step(
        self,
        policy: PolicyBatch,
        supervision: SupervisionBatch,
        config: EngineConfig,
    ) -> TrainingStepOutput:
        config.validate()
        encoded = self.encode_policy(policy)
        action_selector = None
        if config.stage in {CurriculumStage.DAGGER_T2, CurriculumStage.ST_BRIDGE} and config.oracle_mix > 0:
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
