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


class FGCLIPBackbone(nn.Module):
    """Small Track-B adapter around the official FG-CLIP v1 checkpoint.

    The recurrent state is exactly the penultimate patch tensor. The learned
    ``q_G`` replaces the discarded CLS token and reads every later state through
    the checkpoint's final-block projections, norms, MLP and visual projection.
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
    ) -> None:
        super().__init__()
        self.model = model
        self.internal_width = text_width  # retained for train/eval infrastructure
        self.text_dim = text_width
        self.train_vision = train_vision
        self.train_text = train_text
        self.train_text_projection = train_text_projection
        self.checkpoint = checkpoint
        self.revision = revision

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
        """Return H_{L-1}^{patch}; CLS is deliberately discarded."""

        context = nullcontext() if self.train_vision else torch.no_grad()
        with context:
            outputs = self.model.vision_model(
                pixel_values=reference_images,
                output_hidden_states=True,
                return_dict=True,
            )
        return outputs.hidden_states[-2][:, 1:]

    def _flatten_state(self, state: Tensor) -> tuple[Tensor, torch.Size]:
        leading = state.shape[:-2]
        return state.reshape(-1, state.shape[-2], state.shape[-1]), leading

    def global_readout(self, state: Tensor) -> Tensor:
        """Read current patches with q_G through the native final-block machinery."""

        patches, leading = self._flatten_state(state)
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
        global_state = self.model.vision_model.post_layernorm(query[:, 0])
        return global_state.reshape(*leading, width)

    def dense_readout(self, state: Tensor) -> Tensor:
        """Official v1 local path: forward_without_attn -> post-LN -> projection."""

        patches, leading = self._flatten_state(state)
        dense = self.model.forward_without_attn(patches)
        dense = self.model.vision_model.post_layernorm(dense)
        dense = self.model.visual_projection(dense)
        return dense.reshape(*leading, state.shape[-2], self.dense_dim)

    def retrieval_readout(self, state: Tensor) -> Tensor:
        global_state = self.global_readout(state)
        query = self.model.visual_projection(global_state)
        return F.normalize(query.float(), dim=-1).to(query.dtype)

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
