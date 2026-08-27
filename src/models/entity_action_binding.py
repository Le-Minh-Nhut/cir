from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from backbones.fgclip2 import FGCLIP2_LARGE_MODEL_ID, FGCLIP2_LARGE_REVISION
from models.retrieval import multi_positive_retrieval_loss

RelationVariant = Literal["full", "global_only", "drop", "single", "repeat"]


class SharedRelationBinder(nn.Module):
    """One relation-query bank cross-attending to either frozen modality."""

    def __init__(self, *, dim: int = 1024, num_relations: int = 4) -> None:
        super().__init__()
        if dim < 1 or num_relations < 1:
            raise ValueError("dim and num_relations must be positive")
        self.dim = dim
        self.num_relations = num_relations
        self.relation_queries = nn.Parameter(torch.empty(num_relations, dim))
        self.query_projection = nn.Linear(dim, dim, bias=False)
        self.key_projection = nn.Linear(dim, dim, bias=False)
        self.value_projection = nn.Linear(dim, dim, bias=False)
        self.output_projection = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim)
        )
        nn.init.normal_(self.relation_queries, std=dim**-0.5)

    def forward(self, tokens: Tensor, valid_mask: Tensor) -> tuple[Tensor, Tensor]:
        """Bind [B,N,D] tokens; mask True means real/allowed."""

        if tokens.ndim != 3 or tokens.shape[-1] != self.dim:
            raise ValueError(f"tokens must be [B,N,{self.dim}]")
        if valid_mask.shape != tokens.shape[:2] or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be bool [B,N]")
        batch_size, token_count, _ = tokens.shape
        queries = self.query_projection(self.relation_queries)  # [K,D]
        keys = self.key_projection(tokens)  # [B,N,D]
        values = self.value_projection(tokens)  # [B,N,D]
        scores = torch.einsum("kd,bnd->bkn", queries.float(), keys.float())
        scores = scores / math.sqrt(self.dim)
        allowed = valid_mask[:, None, :]
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attention = attention * allowed.to(attention.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        bound = torch.einsum("bkn,bnd->bkd", attention.to(values.dtype), values)
        bound = self.output_projection(bound)
        bound = self.norm(bound + self.mlp(bound))
        any_valid = valid_mask.any(dim=-1)[:, None, None]
        bound = bound * any_valid.to(bound.dtype)
        if tuple(attention.shape) != (batch_size, self.num_relations, token_count):
            raise RuntimeError("Unexpected relation attention shape")
        if not torch.isfinite(attention).all() or not torch.isfinite(bound).all():
            raise FloatingPointError("Relation binding produced NaN or Inf")
        return bound, attention


class FeatureWiseAffineFusion(nn.Module):
    """ENCODER-inspired paired-token affine fusion."""

    def __init__(self, *, dim: int = 1024, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.dim = dim
        self.network = nn.Sequential(
            nn.LayerNorm(2 * dim),
            nn.Linear(2 * dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * dim),
        )
        last = self.network[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        with torch.no_grad():
            last.bias[:dim].fill_(1.0)
            last.bias[dim:].zero_()

    def forward(self, reference_tokens: Tensor, text_tokens: Tensor) -> Tensor:
        if reference_tokens.shape != text_tokens.shape or reference_tokens.shape[-1] != self.dim:
            raise ValueError("paired reference/text tokens must share shape [B,K+1,D]")
        gamma, beta = self.network(
            torch.cat((reference_tokens, text_tokens), dim=-1)
        ).chunk(2, dim=-1)
        return gamma * reference_tokens + beta * text_tokens


def symmetric_set_similarity(action: Tensor, entity: Tensor) -> Tensor:
    """Symmetric unweighted Chamfer-style set similarity [B,B]."""

    if action.ndim != 3 or entity.ndim != 3 or action.shape[1:] != entity.shape[1:]:
        raise ValueError("action and entity must be [B,K,D] with matching K,D")
    action = F.normalize(action, dim=-1)
    entity = F.normalize(entity, dim=-1)
    pairwise = torch.einsum("ikd,jld->ijkl", action, entity)
    action_to_entity = pairwise.amax(dim=-1).mean(dim=-1)
    entity_to_action = pairwise.amax(dim=-2).mean(dim=-1)
    return 0.5 * (action_to_entity + entity_to_action)


class EntityActionBindingCIR(nn.Module):
    """Frozen-feature CIR head with shared image/text relation binding."""

    def __init__(
        self,
        *,
        dim: int = 1024,
        num_relations: int = 4,
        fusion_hidden_dim: int | None = None,
        retrieval_temperature: float = 0.07,
        entity_action_temperature: float = 0.07,
    ) -> None:
        super().__init__()
        if retrieval_temperature <= 0 or entity_action_temperature <= 0:
            raise ValueError("loss temperatures must be positive")
        self.dim = dim
        self.num_relations = num_relations
        self.fusion_hidden_dim = fusion_hidden_dim or dim
        self.retrieval_temperature = retrieval_temperature
        self.entity_action_temperature = entity_action_temperature
        self.binder = SharedRelationBinder(dim=dim, num_relations=num_relations)
        self.fusion = FeatureWiseAffineFusion(dim=dim, hidden_dim=self.fusion_hidden_dim)

    def experiment_provenance(self) -> dict[str, object]:
        return {
            "architecture": "fgclip2_shared_entity_action_binding",
            "backbone_model_id": FGCLIP2_LARGE_MODEL_ID,
            "backbone_revision": FGCLIP2_LARGE_REVISION,
            "dim": self.dim,
            "num_relations": self.num_relations,
            "fusion_hidden_dim": self.fusion_hidden_dim,
            "retrieval_temperature": self.retrieval_temperature,
            "entity_action_temperature": self.entity_action_temperature,
        }

    def encode_image(
        self, image_global: Tensor, image_dense: Tensor, image_dense_mask: Tensor
    ) -> dict[str, Tensor]:
        self._validate_global(image_global, "image_global")
        entity, attention = self.binder(image_dense, image_dense_mask)
        tokens = torch.cat((image_global[:, None, :], entity), dim=1)
        embedding = F.normalize(tokens.mean(dim=1), dim=-1)
        return {"entity": entity, "attention": attention, "tokens": tokens, "embedding": embedding}

    def encode_text(
        self, text_global: Tensor, text_states: Tensor, text_content_mask: Tensor
    ) -> dict[str, Tensor]:
        self._validate_global(text_global, "text_global")
        action, attention = self.binder(text_states, text_content_mask)
        tokens = torch.cat((text_global[:, None, :], action), dim=1)
        return {"action": action, "attention": attention, "tokens": tokens}

    def _validate_global(self, value: Tensor, name: str) -> None:
        if value.ndim != 2 or value.shape[-1] != self.dim:
            raise ValueError(f"{name} must be [B,{self.dim}]")
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"{name} contains NaN or Inf")

    def compose_bound(
        self,
        reference_tokens: Tensor,
        text_tokens: Tensor,
        *,
        variant: RelationVariant = "full",
        relation_index: int | None = None,
    ) -> Tensor:
        fused = self.fusion(reference_tokens, text_tokens)
        if variant == "full":
            selected = fused
        elif variant == "global_only":
            selected = fused[:, :1]
        else:
            if relation_index is None or not 0 <= relation_index < self.num_relations:
                raise ValueError("relation_index is required and must identify a real relation")
            relation = fused[:, relation_index + 1 : relation_index + 2]
            if variant == "drop":
                keep = [0] + [index + 1 for index in range(self.num_relations) if index != relation_index]
                selected = fused[:, keep]
            elif variant == "single":
                selected = torch.cat((fused[:, :1], relation), dim=1)
            elif variant == "repeat":
                selected = torch.cat(
                    (fused[:, :1], relation.expand(-1, self.num_relations, -1)), dim=1
                )
            else:
                raise ValueError(f"Unsupported relation variant: {variant}")
        return F.normalize(selected.mean(dim=1), dim=-1)

    def forward(
        self,
        *,
        reference_global: Tensor,
        reference_dense: Tensor,
        reference_dense_mask: Tensor,
        text_global: Tensor,
        text_states: Tensor,
        text_content_mask: Tensor,
        target_global: Tensor | None = None,
        target_dense: Tensor | None = None,
        target_dense_mask: Tensor | None = None,
        variant: RelationVariant = "full",
        relation_index: int | None = None,
    ) -> dict[str, Tensor]:
        reference = self.encode_image(reference_global, reference_dense, reference_dense_mask)
        text = self.encode_text(text_global, text_states, text_content_mask)
        query = self.compose_bound(
            reference["tokens"], text["tokens"], variant=variant, relation_index=relation_index
        )
        output = {
            "query": query,
            "entity": reference["entity"],
            "action": text["action"],
            "vision_attention": reference["attention"],
            "text_attention": text["attention"],
            "reference_image_tokens": reference["tokens"],
            "text_tokens": text["tokens"],
        }
        provided = (target_global is not None, target_dense is not None, target_dense_mask is not None)
        if any(provided) and not all(provided):
            raise ValueError("target_global, target_dense and target_dense_mask are all-or-none")
        if all(provided):
            assert target_global is not None and target_dense is not None and target_dense_mask is not None
            target = self.encode_image(target_global, target_dense, target_dense_mask)
            output.update(
                {
                    "target_embedding": target["embedding"],
                    "target_entity": target["entity"],
                    "target_vision_attention": target["attention"],
                }
            )
        return output

    def relation_ortho_loss(self) -> Tensor:
        relations = F.normalize(self.binder.relation_queries, dim=-1)
        gram = relations @ relations.T
        return F.mse_loss(gram, torch.eye(self.num_relations, device=gram.device, dtype=gram.dtype))

    def compute_loss(self, batch: dict[str, object]) -> dict[str, Tensor]:
        output = self.forward(
            reference_global=batch["reference_global"],  # type: ignore[arg-type]
            reference_dense=batch["reference_dense"],  # type: ignore[arg-type]
            reference_dense_mask=batch["reference_dense_mask"],  # type: ignore[arg-type]
            text_global=batch["text_global"],  # type: ignore[arg-type]
            text_states=batch["text_states"],  # type: ignore[arg-type]
            text_content_mask=batch["text_content_mask"],  # type: ignore[arg-type]
            target_global=batch["target_global"],  # type: ignore[arg-type]
            target_dense=batch["target_dense"],  # type: ignore[arg-type]
            target_dense_mask=batch["target_dense_mask"],  # type: ignore[arg-type]
        )
        target_ids = batch["target_ids"]
        retrieval = multi_positive_retrieval_loss(
            output["query"], output["target_embedding"], target_ids,  # type: ignore[arg-type]
            temperature=self.retrieval_temperature,
        )
        set_logits = symmetric_set_similarity(output["action"], output["entity"])
        labels = torch.arange(set_logits.shape[0], device=set_logits.device)
        entity_action = 0.5 * (
            F.cross_entropy(set_logits / self.entity_action_temperature, labels)
            + F.cross_entropy(set_logits.T / self.entity_action_temperature, labels)
        )
        losses = {
            "retrieval_loss": retrieval,
            "entity_action_loss": entity_action,
            "relation_ortho_loss": self.relation_ortho_loss(),
        }
        losses.update(self.diagnostics(output, batch["reference_dense_mask"], batch["text_content_mask"]))  # type: ignore[arg-type]
        return losses

    @torch.no_grad()
    def diagnostics(
        self, output: dict[str, Tensor], vision_mask: Tensor, text_mask: Tensor
    ) -> dict[str, Tensor]:
        relations = F.normalize(self.binder.relation_queries, dim=-1)
        relation_gram = relations @ relations.T
        off = ~torch.eye(self.num_relations, dtype=torch.bool, device=relations.device)
        result: dict[str, Tensor] = {
            "diagnostic/relation_offdiag_cosine_mean": relation_gram[off].mean(),
            "diagnostic/relation_offdiag_cosine_max": relation_gram[off].max(),
            "diagnostic/entity_slot_pairwise_cosine": self._slot_pair_cosine(output["entity"]),
            "diagnostic/action_slot_pairwise_cosine": self._slot_pair_cosine(output["action"]),
        }
        entity = F.normalize(output["entity"], dim=-1)
        action = F.normalize(output["action"], dim=-1)
        pairing = torch.einsum("bkd,bld->bkl", entity, action)
        diagonal = torch.eye(self.num_relations, dtype=torch.bool, device=pairing.device)
        same = pairing[:, diagonal].mean()
        different = pairing[:, ~diagonal].mean()
        result.update(
            {
                "diagnostic/entity_action_same_index_cosine": same,
                "diagnostic/entity_action_off_index_cosine": different,
                "diagnostic/entity_action_pairing_gap": same - different,
            }
        )
        result.update(self._attention_diagnostics("vision", output["vision_attention"], vision_mask))
        result.update(self._attention_diagnostics("text", output["text_attention"], text_mask))
        return result

    def _slot_pair_cosine(self, slots: Tensor) -> Tensor:
        normalized = F.normalize(slots, dim=-1)
        gram = torch.einsum("bkd,bld->bkl", normalized, normalized)
        off = ~torch.eye(self.num_relations, dtype=torch.bool, device=slots.device)
        return gram[:, off].mean()

    def _attention_diagnostics(
        self, prefix: str, attention: Tensor, valid_mask: Tensor
    ) -> dict[str, Tensor]:
        safe = attention.clamp_min(1e-12)
        entropy = -(attention * safe.log()).sum(dim=-1)
        valid_count = valid_mask.sum(dim=-1).clamp_min(1).float()
        denominator = valid_count.log().clamp_min(1e-12)[:, None]
        normalized_entropy = torch.where(
            valid_count[:, None] > 1, entropy / denominator, torch.zeros_like(entropy)
        )
        result: dict[str, Tensor] = {}
        for relation in range(self.num_relations):
            result[f"diagnostic/{prefix}_relation_{relation}_attention_entropy"] = entropy[:, relation].mean()
            result[f"diagnostic/{prefix}_relation_{relation}_normalized_entropy"] = normalized_entropy[:, relation].mean()
            result[f"diagnostic/{prefix}_relation_{relation}_max_probability"] = attention[:, relation].amax(dim=-1).mean()
            result[f"diagnostic/{prefix}_relation_{relation}_effective_support"] = entropy[:, relation].exp().mean()
        return result
