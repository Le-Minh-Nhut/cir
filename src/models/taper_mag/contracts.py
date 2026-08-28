from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True, slots=True)
class PolicyBatch:
    """Target-free inputs accepted by the online backbone/TAPER boundary."""

    reference_ids: tuple[str, ...]
    modification_texts: tuple[str, ...]
    reference_local: Tensor
    reference_local_mask: Tensor
    reference_global: Tensor
    text_input_ids: Tensor
    text_attention_mask: Tensor
    text_content_mask: Tensor
    spatial_shapes: Tensor | None = None


@dataclass(frozen=True, slots=True)
class EncodedPolicyBatch:
    """Target-free backbone outputs consumed by TAPER-MAG."""

    reference_local: Tensor
    reference_local_mask: Tensor
    reference_global: Tensor
    text_tokens: Tensor
    text_attention_mask: Tensor
    text_content_mask: Tensor
    spatial_shapes: Tensor | None = None

    def validate(self, text_dim: int, vision_dim: int, retrieval_dim: int) -> None:
        batch = self.reference_local.shape[0]
        if self.reference_local.ndim != 3 or self.reference_local.shape[-1] != vision_dim:
            raise ValueError("reference_local must be [B,N,vision_dim]")
        if self.reference_local_mask.shape != self.reference_local.shape[:2]:
            raise ValueError("reference_local_mask must be [B,N]")
        if self.text_tokens.ndim != 3 or self.text_tokens.shape != (
            batch,
            self.text_tokens.shape[1],
            text_dim,
        ):
            raise ValueError("text_tokens must be [B,M,text_dim]")
        if self.text_attention_mask.shape != self.text_tokens.shape[:2]:
            raise ValueError("text_attention_mask must be [B,M]")
        if self.text_content_mask.shape != self.text_tokens.shape[:2]:
            raise ValueError("text_content_mask must be [B,M]")
        if self.reference_global.shape != (batch, retrieval_dim):
            raise ValueError("reference_global must be [B,retrieval_dim]")
        if not self.reference_local_mask.bool().any(dim=1).all():
            raise ValueError("Every reference requires at least one valid local token")
        if not self.text_content_mask.bool().any(dim=1).all():
            raise ValueError("Every modification requires at least one content token")


@dataclass(frozen=True, slots=True)
class SupervisionBatch:
    """Training-only target data. Never accepted by TaperMAG.forward."""

    target_embedding: Tensor
    target_ids: tuple[str, ...]
    positive_ids: tuple[tuple[str, ...], ...]

    def validate(self, batch_size: int, retrieval_dim: int) -> None:
        if self.target_embedding.shape != (batch_size, retrieval_dim):
            raise ValueError("target_embedding must be [B,retrieval_dim]")
        if len(self.target_ids) != batch_size or len(self.positive_ids) != batch_size:
            raise ValueError("Supervision ID metadata must align with batch size")
