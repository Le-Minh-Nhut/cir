from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def initialize_semantic_residual(content_mask: Tensor) -> Tensor:
    """Create rho_0=1 on content and exactly zero on padding in FP32."""

    if content_mask.ndim != 2 or content_mask.dtype != torch.bool:
        raise ValueError("content_mask must be boolean [B,L]")
    if not content_mask.any(dim=1).all():
        raise ValueError("every sample must contain at least one content token")
    return content_mask.float()


def claimed_text_content(
    text_tokens: Tensor,
    claims: Tensor,
    residual: Tensor,
    epsilon: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    """Return claim-gated token weights and their normalized semantic content."""

    if text_tokens.ndim != 3 or claims.ndim != 3 or residual.ndim != 2:
        raise ValueError("text/claims/residual must be [B,L,d]/[B,K,L]/[B,L]")
    if claims.shape[0] != text_tokens.shape[0] or claims.shape[2] != text_tokens.shape[1]:
        raise ValueError("claim/text axes mismatch")
    if residual.shape != text_tokens.shape[:2]:
        raise ValueError("residual/text axes mismatch")
    with torch.autocast(device_type=text_tokens.device.type, enabled=False):
        weights = claims.float() * residual.float()[:, None, :]
        content = torch.einsum("bkl,bld->bkd", weights, text_tokens.float())
        content = content / weights.sum(dim=-1, keepdim=True).clamp_min(epsilon)
    return weights, content


def candidate_residual_previews(residual: Tensor, claims: Tensor) -> Tensor:
    """Same-parent rho previews; no candidate is committed by this function."""

    if residual.ndim != 2 or claims.ndim != 3:
        raise ValueError("residual/claims must be [B,L]/[B,K,L]")
    if claims.shape[0] != residual.shape[0] or claims.shape[2] != residual.shape[1]:
        raise ValueError("residual/claim axes mismatch")
    with torch.autocast(device_type=residual.device.type, enabled=False):
        previews = residual.float()[:, None, :] * (1.0 - claims.float())
    return previews.clamp_(0.0, 1.0)


def select_next_residual(
    candidate_residuals: Tensor,
    current_residual: Tensor,
    action: Tensor,
    *,
    freeze: bool = False,
) -> Tensor:
    """Commit only the selected candidate claim; STOP/freeze preserves rho exactly."""

    if candidate_residuals.ndim != 3 or current_residual.ndim != 2:
        raise ValueError("candidate/current residual must be [B,K,L]/[B,L]")
    if candidate_residuals.shape[0] != current_residual.shape[0]:
        raise ValueError("candidate/current residual batch mismatch")
    if action.shape != (
        current_residual.shape[0],
        candidate_residuals.shape[1] + 1,
    ):
        raise ValueError("action must contain K candidates plus STOP")
    if freeze:
        return current_residual
    all_residuals = torch.cat(
        [candidate_residuals, current_residual[:, None, :]], dim=1
    )
    with torch.autocast(device_type=current_residual.device.type, enabled=False):
        next_residual = torch.einsum(
            "bk,bkl->bl", action.float(), all_residuals.float()
        )
    return next_residual.clamp_(0.0, 1.0)


class SemanticClaimModule(nn.Module):
    """Shared target-free claims over currently remaining semantic evidence."""

    def __init__(
        self,
        width: int = 256,
        initial_claim_probability: float = 0.99,
    ) -> None:
        super().__init__()
        if not 0.0 < initial_claim_probability < 1.0:
            raise ValueError("initial_claim_probability must be strictly between 0 and 1")
        self.width = width
        self.initial_claim_probability = initial_claim_probability
        self.query_projection = nn.Linear(width, width, bias=False)
        self.token_projection = nn.Linear(width, width, bias=False)
        self.state_projection = nn.Linear(width, width, bias=False)
        self.compatibility = nn.Sequential(
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, 1),
        )
        final = self.compatibility[-1]
        nn.init.zeros_(final.weight)
        nn.init.constant_(
            final.bias,
            math.log(initial_claim_probability / (1.0 - initial_claim_probability)),
        )

    def forward(
        self,
        candidate_queries: Tensor,
        text_tokens: Tensor,
        content_mask: Tensor,
        residual: Tensor,
        current_state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if candidate_queries.ndim != 2 or candidate_queries.shape[-1] != self.width:
            raise ValueError("candidate_queries must be [K,d]")
        if text_tokens.ndim != 3 or text_tokens.shape[-1] != self.width:
            raise ValueError("text_tokens must be [B,L,d]")
        if content_mask.shape != text_tokens.shape[:2] or content_mask.dtype != torch.bool:
            raise ValueError("content_mask must be boolean [B,L]")
        if residual.shape != text_tokens.shape[:2]:
            raise ValueError("residual must be [B,L]")
        if current_state.ndim != 3 or current_state.shape[0] != text_tokens.shape[0]:
            raise ValueError("current_state must be [B,N,d]")
        with torch.autocast(device_type=text_tokens.device.type, enabled=False):
            remaining_tokens = text_tokens.float() * residual.float().unsqueeze(-1)
            query = self.query_projection(candidate_queries.float())[None, :, None, :]
            token = self.token_projection(remaining_tokens)[:, None, :, :]
            state = self.state_projection(current_state.float().mean(dim=1))[
                :, None, None, :
            ]
            logits = self.compatibility(query + token + state).squeeze(-1)
            claims = torch.sigmoid(logits)
            valid = content_mask[:, None, :]
            logits = logits.masked_fill(~valid, 0.0)
            claims = claims.masked_fill(~valid, 0.0)
        return logits, claims
