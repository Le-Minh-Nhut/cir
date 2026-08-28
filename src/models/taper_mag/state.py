from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from models.taper_mag.contracts import EncodedPolicyBatch


@dataclass(frozen=True, slots=True)
class LocalState:
    local: Tensor
    initial_local: Tensor
    local_mask: Tensor
    reference_global: Tensor
    reference_anchor: Tensor
    alive: Tensor

    def with_local(self, local: Tensor, alive: Tensor | None = None) -> LocalState:
        return LocalState(
            local=local,
            initial_local=self.initial_local,
            local_mask=self.local_mask,
            reference_global=self.reference_global,
            reference_anchor=self.reference_anchor,
            alive=self.alive if alive is None else alive,
        )


@dataclass(frozen=True, slots=True)
class ProjectedInputs:
    text: Tensor
    initial_local: Tensor
    local_mask: Tensor
    reference_global: Tensor
    reference_anchor: Tensor


class InputAdapters(nn.Module):
    def __init__(
        self,
        *,
        text_dim: int,
        vision_dim: int,
        retrieval_dim: int,
        d_model: int = 256,
    ) -> None:
        super().__init__()
        self.text_projection = nn.Linear(text_dim, d_model)
        self.visual_projection = nn.Linear(vision_dim, d_model)
        self.reference_projection = nn.Linear(retrieval_dim, d_model)
        self.text_norm = nn.LayerNorm(d_model)
        self.visual_norm = nn.LayerNorm(d_model)
        self.reference_norm = nn.LayerNorm(d_model)

    def forward(self, batch: EncodedPolicyBatch) -> ProjectedInputs:
        text = self.text_norm(self.text_projection(batch.text_tokens))
        local = self.visual_norm(self.visual_projection(batch.reference_local))
        mask = batch.reference_local_mask.bool()
        local = local.masked_fill(~mask.unsqueeze(-1), 0.0)
        reference_global = torch.nn.functional.normalize(
            batch.reference_global.float(), dim=-1
        ).to(local.dtype)
        anchor = self.reference_norm(self.reference_projection(reference_global))
        return ProjectedInputs(text, local, mask, reference_global, anchor)

    @staticmethod
    def initialize_state(inputs: ProjectedInputs) -> LocalState:
        batch = inputs.initial_local.shape[0]
        return LocalState(
            local=inputs.initial_local,
            initial_local=inputs.initial_local,
            local_mask=inputs.local_mask,
            reference_global=inputs.reference_global,
            reference_anchor=inputs.reference_anchor,
            alive=torch.ones(batch, dtype=torch.bool, device=inputs.initial_local.device),
        )
