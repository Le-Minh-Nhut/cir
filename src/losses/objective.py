from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from torch import Tensor, nn

from models.iag_srme.outputs import IAGSRMEOutput

from .action_claim_binding import ActionClaimBindingLoss
from .complementary_claim import ComplementaryClaimLoss, claim_weighted_text_pool
from .factor import FactorCompletenessLoss
from .marginal import MarginalActionLoss
from .retrieval import TerminalRetrievalLoss
from .unique import UniqueContributionLoss


@dataclass(frozen=True, slots=True)
class ObjectiveConfig:
    terminal_weight: float = 1.0
    marginal_weight: float = 0.5
    complementary_claim_weight: float = 0.0
    binding_weight: float = 0.0
    factor_weight: float = 0.0
    unique_weight: float = 0.0
    retrieval_temperature: float = 0.07
    utility_temperature: float = 1.0
    score_temperature: float = 1.0
    binding_temperature: float = 0.1
    factor_anchor_temperature: float = 0.1
    factor_temperature: float = 0.1
    unique_margin: float = 0.05
    relational_self_masked: bool = False
    require_activity_weights_for_unique: bool = True


class IAGSRMEObjective(nn.Module):
    def __init__(self, config: ObjectiveConfig, width: int = 256) -> None:
        super().__init__()
        self.config = config
        self.terminal = TerminalRetrievalLoss(config.retrieval_temperature)
        self.marginal = MarginalActionLoss(
            config.retrieval_temperature,
            config.utility_temperature,
            config.score_temperature,
        )
        self.complementary_claim = ComplementaryClaimLoss()
        self.binding = ActionClaimBindingLoss(
            width=width, temperature=config.binding_temperature
        )
        self.factor = FactorCompletenessLoss(
            config.factor_anchor_temperature,
            config.factor_temperature,
            config.relational_self_masked,
        )
        self.unique = UniqueContributionLoss(config.unique_margin)

    def forward(
        self,
        output: IAGSRMEOutput,
        target_embeddings: Tensor,
        positive_mask: Tensor,
        *,
        active_weights: Tensor | None = None,
    ) -> Mapping[str, Tensor]:
        components: dict[str, Tensor] = {}
        terminal = self.terminal(output.final_query, target_embeddings, positive_mask)
        marginal = self.marginal(output.trace, target_embeddings, positive_mask)
        components["terminal"] = terminal
        components["marginal"] = marginal
        total = self.config.terminal_weight * terminal + self.config.marginal_weight * marginal

        if self.config.complementary_claim_weight > 0 or self.config.binding_weight > 0:
            if output.claims is None:
                raise ValueError("claim losses enabled but model claim head is disabled")
            claim_result = self.complementary_claim(output.claims, output.text_content_mask)
            components["complementary_claim"] = claim_result.loss
            components["diagnostic/claim_mass"] = claim_result.raw_claim_mass.mean().detach()
            total = total + self.config.complementary_claim_weight * claim_result.loss
            if self.config.binding_weight > 0:
                claimed_semantics = claim_weighted_text_pool(
                    output.claims, output.text_tokens, output.text_content_mask
                )
                binding = self.binding(output.intents, claimed_semantics)
                components["binding"] = binding
                total = total + self.config.binding_weight * binding

        if self.config.factor_weight > 0 or self.config.unique_weight > 0:
            if output.factors is None or output.auxiliary_anchor is None:
                raise ValueError("factor losses enabled but model factor head is disabled")
            factor_loss, geometry = self.factor(
                output.factors, output.auxiliary_anchor, active_weights=active_weights
            )
            components["factor"] = factor_loss
            total = total + self.config.factor_weight * factor_loss
            if self.config.unique_weight > 0:
                if self.config.require_activity_weights_for_unique and active_weights is None:
                    raise ValueError(
                        "L_unique requires externally justified activity weights; STOP is not "
                        "factor inactivity. Set the guard false only for an explicit all-active ablation."
                    )
                unique_loss, contribution = self.unique(geometry, active_weights=active_weights)
                components["unique"] = unique_loss
                components["diagnostic/unique_contribution"] = contribution.mean().detach()
                total = total + self.config.unique_weight * unique_loss

        components["total"] = total
        return components
