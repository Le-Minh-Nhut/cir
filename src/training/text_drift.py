from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from backbones.fgclip2_base import FGCLIP2BaseBackbone, TokenizedTextBatch


@dataclass(frozen=True, slots=True)
class TextReferenceSnapshot:
    input_ids: Tensor
    attention_mask: Tensor
    content_mask: Tensor
    token_states: Tensor
    pooled_states: Tensor
    trainable_parameters: dict[str, Tensor]


class TextDriftMonitor:
    @staticmethod
    @torch.no_grad()
    def capture(
        backbone: FGCLIP2BaseBackbone, batch: TokenizedTextBatch
    ) -> TextReferenceSnapshot:
        device_states = backbone.encode_text_tokens(batch)
        device_pooled = backbone.pool_short_text_states(device_states)
        states = device_states.detach().float().cpu()
        pooled = device_pooled.detach().float().cpu()
        parameters = {
            name: parameter.detach().float().cpu().clone()
            for name, parameter in backbone.model.named_parameters()
            if parameter.requires_grad
        }
        return TextReferenceSnapshot(
            batch.input_ids.cpu(),
            batch.attention_mask.cpu(),
            batch.content_mask.cpu(),
            states,
            pooled,
            parameters,
        )

    @staticmethod
    @torch.no_grad()
    def measure(
        backbone: FGCLIP2BaseBackbone, snapshot: TextReferenceSnapshot
    ) -> dict[str, float]:
        batch = TokenizedTextBatch(
            snapshot.input_ids.to(backbone.device),
            snapshot.attention_mask.to(backbone.device),
            snapshot.content_mask.to(backbone.device),
        )
        device_states = backbone.encode_text_tokens(batch)
        device_pooled = backbone.pool_short_text_states(device_states)
        states = device_states.detach().float().cpu()
        pooled = device_pooled.detach().float().cpu()
        mask = snapshot.content_mask.unsqueeze(-1)
        token_cosine = F.cosine_similarity(states, snapshot.token_states, dim=-1)
        token_cosine = token_cosine[snapshot.content_mask].mean().item()
        pooled_cosine = F.cosine_similarity(pooled, snapshot.pooled_states, dim=-1).mean().item()
        del mask  # mask is kept explicit above to make special-token exclusion auditable.
        squared_change = 0.0
        squared_reference = 0.0
        block_squared_change: dict[int, float] = {}
        current_parameters = dict(backbone.model.named_parameters())
        for name, reference in snapshot.trainable_parameters.items():
            current = current_parameters[name].detach().float().cpu()
            squared_change += float((current - reference).square().sum())
            squared_reference += float(reference.square().sum())
            prefix = "text_model.encoder.layers."
            if name.startswith(prefix):
                block_index = int(name[len(prefix) :].split(".", 1)[0])
                block_squared_change[block_index] = block_squared_change.get(block_index, 0.0) + float(
                    (current - reference).square().sum()
                )
        result = {
            "text_token_cosine": token_cosine,
            "text_token_cosine_drift": 1.0 - token_cosine,
            "text_pooled_cosine": pooled_cosine,
            "text_pooled_cosine_drift": 1.0 - pooled_cosine,
            "text_parameter_relative_change": (
                squared_change**0.5 / max(squared_reference**0.5, 1e-12)
            ),
        }
        result.update(
            {
                f"text_block_{index}_parameter_update_norm": squared**0.5
                for index, squared in sorted(block_squared_change.items())
            }
        )
        return result


def text_block_gradient_norms(backbone: FGCLIP2BaseBackbone) -> dict[str, float]:
    result: dict[str, float] = {}
    for index, block in enumerate(backbone.model.text_model.encoder.layers):
        squared = 0.0
        for parameter in block.parameters():
            if parameter.grad is not None:
                squared += float(parameter.grad.detach().float().square().sum())
        result[f"text_block_{index}_grad_norm"] = squared**0.5
    return result
