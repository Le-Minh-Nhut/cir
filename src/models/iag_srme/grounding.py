from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from entmax import entmax15
from torch import Tensor, nn


@dataclass(slots=True)
class GroundingOutput:
    visual_supports: Tensor  # [B,K,N], total mass is visual confidence
    conditional_supports: Tensor  # [B,K,N], conditional WHERE shape
    null_probabilities: Tensor  # [B,K]
    visual_confidence: Tensor  # [B,K] = 1 - p_null
    full_probabilities: Tensor  # [B,K,N] legacy or [B,K,N+1] Visual NULL


class AnchorGrounder(nn.Module):
    """Ground stable intents independently over immutable anchor tokens."""

    def __init__(
        self,
        width: int = 256,
        grounding_width: int | None = None,
        *,
        enable_visual_null: bool = False,
        visual_null_initial_logit: float = 0.0,
        normalization: str = "entmax15",
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        if normalization != "entmax15":
            raise ValueError("canonical grounding normalization must be entmax15")
        grounding_width = grounding_width or width
        self.enable_visual_null = enable_visual_null
        self.normalization = normalization
        self.epsilon = epsilon
        self.intent_projection = nn.Linear(width, grounding_width, bias=False)
        self.anchor_projection = nn.Linear(width, grounding_width, bias=False)
        self.scale = math.sqrt(grounding_width)
        if enable_visual_null:
            # Zero-key initialization gives a neutral, reachable logit on the same scale as
            # centered projected-dot-product logits without initially dominating real tokens.
            self.visual_null_key = nn.Parameter(torch.zeros(grounding_width))
            self.visual_null_bias = nn.Parameter(
                torch.tensor(float(visual_null_initial_logit))
            )
        else:
            self.register_parameter("visual_null_key", None)
            self.register_parameter("visual_null_bias", None)

    def forward(self, intents: Tensor, anchor: Tensor) -> GroundingOutput:
        if intents.ndim != 3 or anchor.ndim != 3:
            raise ValueError("intents and anchor must be rank 3")
        if intents.shape[0] != anchor.shape[0] or intents.shape[-1] != anchor.shape[-1]:
            raise ValueError("intent/anchor batch and width must match")
        projected_intents = self.intent_projection(intents)
        visual_logits = (
            torch.einsum("bkd,bnd->bkn", projected_intents, self.anchor_projection(anchor))
            / self.scale
        )
        logits = visual_logits
        if self.enable_visual_null:
            if self.visual_null_key is None or self.visual_null_bias is None:
                raise AssertionError("Visual NULL parameters are missing")
            null_logits = (
                torch.einsum("bkd,d->bk", projected_intents, self.visual_null_key)
                / self.scale
                + self.visual_null_bias
            )
            logits = torch.cat([visual_logits, null_logits[..., None]], dim=-1)
        # Entmax's threshold/root arithmetic is an explicit AMP FP32 island.
        probabilities_fp32 = entmax15(logits.float(), dim=-1)
        expected_positions = anchor.shape[1] + int(self.enable_visual_null)
        if probabilities_fp32.shape != (
            intents.shape[0],
            intents.shape[1],
            expected_positions,
        ):
            raise AssertionError("grounding probability shape invariant failed")
        if not torch.allclose(
            probabilities_fp32.sum(dim=-1),
            torch.ones_like(probabilities_fp32[..., 0]),
            atol=1e-5,
            rtol=1e-5,
        ):
            raise AssertionError(
                "each candidate must normalize over visual tokens plus optional NULL"
            )

        if self.enable_visual_null:
            visual_fp32 = probabilities_fp32[..., :-1]
            null_fp32 = probabilities_fp32[..., -1]
            confidence_fp32 = 1.0 - null_fp32
            if not torch.allclose(
                visual_fp32.sum(dim=-1), confidence_fp32, atol=1e-5, rtol=1e-5
            ):
                raise AssertionError("real visual mass must equal one minus NULL probability")
            conditional_fp32 = visual_fp32 / confidence_fp32[..., None].clamp_min(
                self.epsilon
            )
        else:
            visual_fp32 = probabilities_fp32
            conditional_fp32 = probabilities_fp32
            null_fp32 = torch.zeros_like(probabilities_fp32[..., 0])
            confidence_fp32 = torch.ones_like(null_fp32)

        return GroundingOutput(
            visual_supports=visual_fp32.to(logits.dtype),
            conditional_supports=conditional_fp32.to(logits.dtype),
            null_probabilities=null_fp32.to(logits.dtype),
            visual_confidence=confidence_fp32.to(logits.dtype),
            full_probabilities=probabilities_fp32.to(logits.dtype),
        )
