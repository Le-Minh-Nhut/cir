from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.iag_srme.utils.retrieval import (
    build_teacher_masks,
    marginal_teacher_utilities,
)

from .retrieval import TerminalRetrievalLoss


@dataclass(frozen=True, slots=True)
class ObjectiveConfig:
    terminal_weight: float = 1.0
    lambda_pair: float = 0.5
    lambda_gain: float = 0.5
    retrieval_temperature: float = 0.07
    pair_temperature: float = 1.0
    pair_weight_temperature: float = 1.0
    epsilon_pair: float = 0.01
    huber_delta: float = 1.0


def pairwise_ranking_loss(
    predicted: Tensor,
    teacher: Tensor,
    valid_rows: Tensor,
    *,
    epsilon: float,
    temperature: float,
    weight_temperature: float,
) -> tuple[Tensor, Tensor]:
    """Confidence-weighted ranking over every unordered sibling pair."""

    candidates = predicted.shape[-1]
    left, right = torch.triu_indices(candidates, candidates, offset=1, device=predicted.device)
    teacher_delta = teacher[:, left] - teacher[:, right]
    confident = teacher_delta.abs() >= epsilon
    pair_mask = valid_rows[:, None] & confident
    weight = 1.0 - torch.exp(-teacher_delta.abs() / weight_temperature)
    signed_margin = teacher_delta.sign() * (predicted[:, left] - predicted[:, right])
    values = -weight * F.logsigmoid(signed_margin / temperature)
    normalizer = (weight * pair_mask).sum()
    loss = (values * pair_mask).sum() / normalizer.clamp_min(1e-8)
    return loss, normalizer


def absolute_gain_loss(
    predicted: Tensor, teacher: Tensor, valid_rows: Tensor, delta: float
) -> tuple[Tensor, Tensor]:
    values = F.huber_loss(predicted, teacher, reduction="none", delta=delta)
    mask = valid_rows[:, None].expand_as(values)
    count = mask.sum()
    return (values * mask).sum() / count.clamp_min(1), count


class IAGSRMEObjective(nn.Module):
    """V2 core: terminal retrieval plus calibrated ScoreNet supervision."""

    def __init__(self, config: ObjectiveConfig, width: int = 256) -> None:
        super().__init__()
        del width  # kept so the existing optimizer/training builder stays minimal
        self.config = config
        self.terminal = TerminalRetrievalLoss(config.retrieval_temperature)

    def forward(
        self,
        output: Mapping[str, object],
        target_embeddings: Tensor,
        target_ids: Sequence[str | None],
    ) -> Mapping[str, Tensor]:
        positive, negative, _ = build_teacher_masks(target_ids, target_embeddings.device)
        query = output["query"]
        assert isinstance(query, Tensor)
        terminal = self.terminal(query, target_embeddings, positive)

        steps = output["steps"]
        assert isinstance(steps, list)
        score_zero = query.sum() * 0.0
        pair_numerator = score_zero
        gain_numerator = score_zero
        pair_count = query.new_zeros(())
        gain_count = query.new_zeros(())
        invalid_rows = query.new_zeros(())

        for step in steps:
            live_indices = step["live_indices"]
            current_query = step["current_query"]
            candidate_queries = step["candidate_queries"]
            predicted = step["scores"]
            pos_live = positive.index_select(0, live_indices)
            neg_live = negative.index_select(0, live_indices)
            teacher, valid_rows = marginal_teacher_utilities(
                current_query,
                candidate_queries,
                target_embeddings.detach(),
                pos_live,
                neg_live,
                self.config.retrieval_temperature,
            )
            pair, pair_weight = pairwise_ranking_loss(
                predicted,
                teacher,
                valid_rows,
                epsilon=self.config.epsilon_pair,
                temperature=self.config.pair_temperature,
                weight_temperature=self.config.pair_weight_temperature,
            )
            gain, gains = absolute_gain_loss(
                predicted, teacher, valid_rows, self.config.huber_delta
            )
            pair_numerator = pair_numerator + pair * pair_weight
            gain_numerator = gain_numerator + gain * gains.clamp_min(1)
            pair_count = pair_count + pair_weight
            gain_count = gain_count + gains
            invalid_rows = invalid_rows + (~valid_rows).sum()

        pair = pair_numerator / pair_count.clamp_min(1e-8)
        gain = gain_numerator / gain_count.clamp_min(1)
        total = (
            self.config.terminal_weight * terminal
            + self.config.lambda_pair * pair
            + self.config.lambda_gain * gain
        )
        return {
            "terminal": terminal,
            "pair": pair,
            "gain": gain,
            "teacher_invalid_rows": invalid_rows.detach(),
            "total": total,
        }
