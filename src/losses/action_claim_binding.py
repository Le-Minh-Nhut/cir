from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ActionClaimBindingLoss(nn.Module):
    def __init__(
        self,
        width: int = 256,
        projection_dim: int = 256,
        temperature: float = 0.1,
        stop_gradient_claim: bool = False,
        symmetric: bool = False,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.stop_gradient_claim = stop_gradient_claim
        self.symmetric = symmetric
        self.action_projection = nn.Linear(width, projection_dim, bias=False)
        self.claim_projection = nn.Linear(width, projection_dim, bias=False)

    def forward(self, intents: Tensor, claimed_semantics: Tensor) -> Tensor:
        if intents.shape != claimed_semantics.shape:
            raise ValueError("intents and claimed_semantics must share [B,K,d]")
        if self.stop_gradient_claim:
            claimed_semantics = claimed_semantics.detach()
        actions = F.normalize(self.action_projection(intents), dim=-1)
        claims = F.normalize(self.claim_projection(claimed_semantics), dim=-1)
        similarities = torch.einsum("bkd,bjd->bkj", actions, claims) / self.temperature
        labels = torch.arange(intents.shape[1], device=intents.device).expand(intents.shape[0], -1)
        forward = F.cross_entropy(similarities.flatten(0, 1), labels.flatten())
        if not self.symmetric:
            return forward
        reverse = F.cross_entropy(similarities.transpose(1, 2).flatten(0, 1), labels.flatten())
        return 0.5 * (forward + reverse)

