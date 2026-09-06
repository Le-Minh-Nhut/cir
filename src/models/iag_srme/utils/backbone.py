from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class FGCLIPRegime:
    checkpoint: str
    revision: str
    train_vision: bool
    train_text: bool = True
    train_text_projection: bool = False
    trust_remote_code: bool = True
    global_readout_mode: str = "learned_qg"


class FGCLIPBackbone(nn.Module):
    """Small Track-B adapter around the official FG-CLIP v1 checkpoint.

    The recurrent state is always the penultimate patch tensor. ``learned_qg``
    reads it with the original shared learned query; ``native_cls`` carries the
    image-specific penultimate CLS as immutable readout context and computes
    the exact CLS row of the checkpoint's final block.
    """

    def __init__(
        self,
        model: nn.Module,
        text_width: int = 256,
        train_vision: bool = True,
        train_text: bool = True,
        train_text_projection: bool = False,
        checkpoint: str | None = None,
        revision: str | None = None,
        global_readout_mode: str = "learned_qg",
    ) -> None:
        super().__init__()
        if global_readout_mode not in {"learned_qg", "native_cls"}:
            raise ValueError(f"unknown Track-B global readout: {global_readout_mode}")
        self.model = model
        self.internal_width = text_width  # retained for train/eval infrastructure
        self.text_dim = text_width
        self.train_vision = train_vision
        self.train_text = train_text
        self.train_text_projection = train_text_projection
        self.checkpoint = checkpoint
        self.revision = revision
        self.global_readout_mode = global_readout_mode

        config = model.config
        self.state_dim = int(config.vision_config.hidden_size)
        self.retrieval_dim = int(config.projection_dim)
        self.dense_dim = self.retrieval_dim
        image_size = int(config.vision_config.image_size)
        patch_size = int(config.vision_config.patch_size)
        side = image_size // patch_size
        self.patch_grid = (side, side)

        text_hidden = int(config.text_config.hidden_size)
        self.text_adapter = nn.Sequential(
            nn.Linear(text_hidden, text_width),
            nn.LayerNorm(text_width),
        )

        # A checkpoint-derived initialization is deterministic and gives q_G the
        # same scale as the native visual CLS role. It remains a learned parameter.
        class_token = model.vision_model.embeddings.class_embedding.detach().clone()
        self.q_G = nn.Parameter(class_token)
        self._apply_freeze_policy()

    @classmethod
    def from_pretrained(
        cls, regime: FGCLIPRegime, text_width: int = 256
    ) -> "FGCLIPBackbone":
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            regime.checkpoint,
            revision=regime.revision,
            trust_remote_code=regime.trust_remote_code,
        )
        return cls(
            model,
            text_width=text_width,
            train_vision=regime.train_vision,
            train_text=regime.train_text,
            train_text_projection=regime.train_text_projection,
            checkpoint=regime.checkpoint,
            revision=regime.revision,
            global_readout_mode=regime.global_readout_mode,
        )

    @staticmethod
    def load_processor(
        checkpoint: str, revision: str, trust_remote_code: bool = True
    ) -> tuple[Any, Any]:
        from transformers import AutoImageProcessor, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint,
            revision=revision,
            trust_remote_code=trust_remote_code,
            tokenizer_type="clip",
        )
        processor = AutoImageProcessor.from_pretrained(
            checkpoint, revision=revision, trust_remote_code=trust_remote_code
        )
        return tokenizer, processor

    def _apply_freeze_policy(self) -> None:
        for parameter in self.model.vision_model.parameters():
            parameter.requires_grad_(self.train_vision)
        for parameter in self.model.visual_projection.parameters():
            parameter.requires_grad_(self.train_vision)
        for parameter in self.model.text_model.parameters():
            parameter.requires_grad_(self.train_text)
        for parameter in self.model.text_projection.parameters():
            parameter.requires_grad_(self.train_text_projection)
        if hasattr(self.model, "text_filip_projection"):
            for parameter in self.model.text_filip_projection.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "FGCLIPBackbone":
        super().train(mode)
        if not self.train_vision:
            self.model.vision_model.eval()
            self.model.visual_projection.eval()
        if not self.train_text:
            self.model.text_model.eval()
        if not self.train_text_projection:
            self.model.text_projection.eval()
        if hasattr(self.model, "text_filip_projection"):
            self.model.text_filip_projection.eval()
        return self

    def initial_state(self, reference_images: Tensor) -> Tensor:
        """Return patch-only H_{L-1}; used by the learned-q_G mode."""

        patches, _ = self.initial_state_with_anchor(reference_images)
        return patches

    def initial_state_with_anchor(self, reference_images: Tensor) -> tuple[Tensor, Tensor]:
        """Return mutable patches and the image-specific immutable H_{L-1} CLS."""

        context = nullcontext() if self.train_vision else torch.no_grad()
        with context:
            outputs = self.model.vision_model(
                pixel_values=reference_images,
                output_hidden_states=True,
                return_dict=True,
            )
        penultimate = outputs.hidden_states[-2]
        return penultimate[:, 1:], penultimate[:, 0]

    def _flatten_state(self, state: Tensor) -> tuple[Tensor, torch.Size]:
        leading = state.shape[:-2]
        return state.reshape(-1, state.shape[-2], state.shape[-1]), leading

    @staticmethod
    def _expand_cls_anchor(cls_anchor: Tensor, leading: torch.Size) -> Tensor:
        anchor_leading = cls_anchor.shape[:-1]
        if tuple(leading[: len(anchor_leading)]) != tuple(anchor_leading):
            raise ValueError("CLS anchor batch dimensions do not match the patch state")
        extra = len(leading) - len(anchor_leading)
        view = cls_anchor.reshape(*anchor_leading, *([1] * extra), cls_anchor.shape[-1])
        return view.expand(*leading, cls_anchor.shape[-1]).reshape(-1, cls_anchor.shape[-1])

    def _learned_qg_readout(self, patches: Tensor) -> Tensor:
        """Preserved learned-query readout used by the original Track-B R0 code."""

        block = self.model.vision_model.encoder.layers[-1]
        batch, tokens, width = patches.shape
        query = self.q_G.to(patches.dtype).view(1, 1, width).expand(batch, 1, width)

        # Final-block-style cross attention: q_G is Q; current patches are K/V.
        q = block.self_attn.q_proj(block.layer_norm1(query)) * block.self_attn.scale
        kv = block.layer_norm1(patches)
        k = block.self_attn.k_proj(kv)
        v = block.self_attn.v_proj(kv)
        heads = block.self_attn.num_heads
        head_dim = block.self_attn.head_dim
        q = q.view(batch, 1, heads, head_dim).transpose(1, 2)
        k = k.view(batch, tokens, heads, head_dim).transpose(1, 2)
        v = v.view(batch, tokens, heads, head_dim).transpose(1, 2)
        weights = torch.softmax(q @ k.transpose(-2, -1), dim=-1)
        weights = F.dropout(
            weights,
            p=float(block.self_attn.dropout),
            training=block.training,
        )
        attended = (weights @ v).transpose(1, 2).reshape(batch, 1, width)
        query = query + block.self_attn.out_proj(attended)
        query = query + block.mlp(block.layer_norm2(query))
        return self.model.vision_model.post_layernorm(query[:, 0])

    def _native_cls_readout_full(self, patches: Tensor, cls_anchor: Tensor) -> Tensor:
        """Exact full-token final block, retained as the native parity oracle."""

        full_state = torch.cat([cls_anchor[:, None], patches], dim=1)
        block = self.model.vision_model.encoder.layers[-1]
        final_state = block(
            hidden_states=full_state,
            attention_mask=None,
            causal_attention_mask=None,
            output_attentions=False,
        )[0]
        return self.model.vision_model.post_layernorm(final_state[:, 0])

    def _native_cls_readout_cls_only(
        self, patches: Tensor, cls_anchor: Tensor
    ) -> Tensor:
        """Compute only the final block's CLS output, with checkpoint weights.

        In a pre-LN block, the final CLS attention row needs Q from CLS and K/V
        from all penultimate tokens. Newly updated patch outputs from the same
        block cannot affect CLS, so materializing them is unnecessary.
        """

        block = self.model.vision_model.encoder.layers[-1]
        attention = block.self_attn
        full_state = torch.cat([cls_anchor[:, None], patches], dim=1)
        normalized = block.layer_norm1(full_state)
        batch, tokens, width = normalized.shape
        heads = attention.num_heads
        head_dim = attention.head_dim

        query = attention.q_proj(normalized[:, :1]) * attention.scale
        key = attention.k_proj(normalized)
        value = attention.v_proj(normalized)
        query = (
            query.view(batch, 1, heads, head_dim)
            .transpose(1, 2)
            .contiguous()
            .view(batch * heads, 1, head_dim)
        )
        key = (
            key.view(batch, tokens, heads, head_dim)
            .transpose(1, 2)
            .contiguous()
            .view(batch * heads, tokens, head_dim)
        )
        value = (
            value.view(batch, tokens, heads, head_dim)
            .transpose(1, 2)
            .contiguous()
            .view(batch * heads, tokens, head_dim)
        )
        weights = torch.bmm(query, key.transpose(1, 2))
        weights = F.softmax(weights, dim=-1)
        weights = F.dropout(
            weights,
            p=float(attention.dropout),
            training=block.training,
        )
        attended = torch.bmm(weights, value)
        attended = (
            attended.view(batch, heads, 1, head_dim)
            .transpose(1, 2)
            .reshape(batch, 1, width)
        )

        cls = cls_anchor[:, None] + attention.out_proj(attended)
        cls = cls + block.mlp(block.layer_norm2(cls))
        return self.model.vision_model.post_layernorm(cls[:, 0])

    def global_readout(self, state: Tensor, cls_anchor: Tensor | None = None) -> Tensor:
        """Read current patches using the configured Track-B global token."""

        patches, leading = self._flatten_state(state)
        width = patches.shape[-1]
        if self.global_readout_mode == "learned_qg":
            global_state = self._learned_qg_readout(patches)
        else:
            if cls_anchor is None:
                raise ValueError("native_cls readout requires the image-specific CLS anchor")
            anchor = self._expand_cls_anchor(cls_anchor, leading).to(patches.dtype)
            global_state = self._native_cls_readout_cls_only(patches, anchor)
        return global_state.reshape(*leading, width)

    def dense_readout(self, state: Tensor) -> Tensor:
        """Official v1 local path: forward_without_attn -> post-LN -> projection."""

        patches, leading = self._flatten_state(state)
        dense = self.model.forward_without_attn(patches)
        dense = self.model.vision_model.post_layernorm(dense)
        dense = self.model.visual_projection(dense)
        return dense.reshape(*leading, state.shape[-2], self.dense_dim)

    def retrieval_from_global(self, global_state: Tensor) -> Tensor:
        """Project a computed global state once and normalize in FP32."""

        query = self.model.visual_projection(global_state)
        return F.normalize(query.float(), dim=-1).to(query.dtype)

    def retrieval_readout(self, state: Tensor, cls_anchor: Tensor | None = None) -> Tensor:
        """Compatibility wrapper for callers without an existing global state."""

        return self.retrieval_from_global(self.global_readout(state, cls_anchor))

    def encode_text(
        self, input_ids: Tensor, attention_mask: Tensor, content_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        outputs = self.model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            walk_short_pos=True,
        )
        tokens = self.text_adapter(outputs.last_hidden_state)
        mask = content_mask.to(tokens.dtype).unsqueeze(-1)
        global_text = (tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return tokens, global_text

    def encode_global_images(self, pixel_values: Tensor) -> Tensor:
        context = nullcontext() if self.train_vision else torch.no_grad()
        with context:
            features = self.model.get_image_features(pixel_values=pixel_values)
        return F.normalize(features.float(), dim=-1).to(features.dtype)


def assert_cache_legal(train_vision: bool, image_cache_path: str | None) -> None:
    if train_vision and image_cache_path is not None:
        raise ValueError("persistent image-feature caches are illegal when FG-CLIP is trainable")
