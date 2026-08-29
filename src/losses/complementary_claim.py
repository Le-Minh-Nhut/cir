from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(slots=True)
class ComplementaryClaimResult:
    loss: Tensor
    raw_claim_mass: Tensor  # [B,K]
    normalized_claims: Tensor  # [B,K,L]
    normalized_peer_complement: Tensor  # [B,K,L]


class ComplementaryClaimLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, claims: Tensor, content_mask: Tensor) -> ComplementaryClaimResult:
        if claims.ndim != 3 or content_mask.shape != (claims.shape[0], claims.shape[2]):
            raise ValueError("claims=[B,K,L], content_mask=[B,L] required")
        if claims.shape[1] < 2:
            raise ValueError("complementary claims require at least two candidates")
        mask = content_mask[:, None, :]
        valid_claims = claims.clamp(0.0, 1.0) * mask.to(claims.dtype)
        raw_mass = valid_claims.sum(dim=-1)
        own = valid_claims / raw_mass[..., None].clamp_min(self.epsilon)

        log_complement = torch.log1p(-valid_claims.clamp_max(1.0 - self.epsilon))
        total_log_complement = log_complement.sum(dim=1, keepdim=True)
        peer_log = (total_log_complement - log_complement) / (claims.shape[1] - 1)
        peer_raw = peer_log.exp() * mask.to(claims.dtype)
        peer = peer_raw / peer_raw.sum(dim=-1, keepdim=True).clamp_min(self.epsilon)

        midpoint = 0.5 * (own + peer)
        own_kl = (
            own * (own.clamp_min(self.epsilon).log() - midpoint.clamp_min(self.epsilon).log())
        ).sum(dim=-1)
        peer_kl = (
            peer * (peer.clamp_min(self.epsilon).log() - midpoint.clamp_min(self.epsilon).log())
        ).sum(dim=-1)
        loss = (0.5 * (own_kl + peer_kl)).mean()
        return ComplementaryClaimResult(loss, raw_mass, own, peer)


def claim_weighted_text_pool(
    claims: Tensor, text_tokens: Tensor, content_mask: Tensor, epsilon: float = 1e-8
) -> Tensor:
    valid = claims * content_mask[:, None].to(claims.dtype)
    return torch.einsum("bkl,bld->bkd", valid, text_tokens) / valid.sum(
        dim=-1, keepdim=True
    ).clamp_min(epsilon)
