from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .intent import masked_text_mean
from .outputs import BackboneOutput


@dataclass(frozen=True, slots=True)
class FGCLIPRegime:
    checkpoint: str
    train_vision: bool
    train_text: bool = True
    trust_remote_code: bool = True


class FGCLIPBackbone(nn.Module):
    """Thin adapter around the official FG-CLIP v1 checkpoint API.

    The adapter intentionally calls ``get_image_dense_features`` for patch tokens and
    ``text_model`` for contextual states. It never substitutes FG-CLIP2.
    """

    def __init__(
        self, model: nn.Module, internal_width: int = 256, train_vision: bool = True
    ) -> None:
        super().__init__()
        self.model = model
        self.internal_width = internal_width
        self.train_vision = train_vision
        config = getattr(model, "config", None)
        if config is None:
            raise ValueError("FG-CLIP model must expose config")
        vision_input_dim = int(config.projection_dim)
        text_input_dim = int(config.text_config.hidden_size)
        self.retrieval_dim = int(config.projection_dim)
        self.anchor_projection = nn.Sequential(
            nn.Linear(vision_input_dim, internal_width), nn.LayerNorm(internal_width)
        )
        self.text_projection = nn.Sequential(
            nn.Linear(text_input_dim, internal_width), nn.LayerNorm(internal_width)
        )
        self._apply_freeze_policy()

    @classmethod
    def from_pretrained(cls, regime: FGCLIPRegime, internal_width: int = 256) -> "FGCLIPBackbone":
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            regime.checkpoint, trust_remote_code=regime.trust_remote_code
        )
        backbone = cls(model, internal_width=internal_width, train_vision=regime.train_vision)
        for parameter in backbone.model.text_model.parameters():
            parameter.requires_grad_(regime.train_text)
        for parameter in backbone.model.text_projection.parameters():
            parameter.requires_grad_(regime.train_text)
        if hasattr(backbone.model, "text_filip_projection"):
            for parameter in backbone.model.text_filip_projection.parameters():
                parameter.requires_grad_(regime.train_text)
        return backbone

    @staticmethod
    def load_processor(checkpoint: str, trust_remote_code: bool = True) -> tuple[Any, Any]:
        from transformers import AutoImageProcessor, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=trust_remote_code)
        image_processor = AutoImageProcessor.from_pretrained(
            checkpoint, trust_remote_code=trust_remote_code
        )
        return tokenizer, image_processor

    def _apply_freeze_policy(self) -> None:
        for parameter in self.model.vision_model.parameters():
            parameter.requires_grad_(self.train_vision)
        for parameter in self.model.visual_projection.parameters():
            parameter.requires_grad_(self.train_vision)
        if not self.train_vision:
            self.model.vision_model.eval()
            self.model.visual_projection.eval()

    def train(self, mode: bool = True) -> "FGCLIPBackbone":
        super().train(mode)
        if not self.train_vision:
            self.model.vision_model.eval()
            self.model.visual_projection.eval()
        return self

    def encode_images(self, pixel_values: Tensor) -> tuple[Tensor, Tensor]:
        if pixel_values.ndim != 4:
            raise ValueError("pixel_values must be [B,C,H,W]")
        context = nullcontext() if self.train_vision else torch.no_grad()
        with context:
            dense = self.model.get_image_dense_features(pixel_values=pixel_values)
            global_features = self.model.get_image_features(pixel_values=pixel_values)
        anchor = self.anchor_projection(dense)
        reference_global = F.normalize(global_features, dim=-1)
        return anchor, reference_global

    def encode_text(
        self, input_ids: Tensor, attention_mask: Tensor, content_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        outputs = self.model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            walk_short_pos=True,
        )
        text_tokens = self.text_projection(outputs.last_hidden_state)
        text_global = masked_text_mean(text_tokens, content_mask)
        return text_tokens, text_global

    def forward(
        self,
        reference_pixels: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        content_mask: Tensor,
    ) -> BackboneOutput:
        anchor, reference_global = self.encode_images(reference_pixels)
        text_tokens, text_global = self.encode_text(input_ids, attention_mask, content_mask)
        return BackboneOutput(
            anchor=anchor,
            reference_global=reference_global,
            text_tokens=text_tokens,
            text_global=text_global,
            text_content_mask=content_mask,
        )


def assert_cache_legal(train_vision: bool, image_cache_path: str | None) -> None:
    if train_vision and image_cache_path is not None:
        raise ValueError(
            "persistent image-feature caches are illegal when FG-CLIP vision is trainable"
        )
