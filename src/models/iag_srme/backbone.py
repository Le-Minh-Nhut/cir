from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from numerics import normalize_fp32

from .intent import masked_text_mean
from .outputs import BackboneOutput


@dataclass(frozen=True, slots=True)
class FGCLIPRegime:
    checkpoint: str
    revision: str
    train_vision: bool
    train_text: bool = True
    train_text_projection: bool = False
    trust_remote_code: bool = True


class FGCLIPBackbone(nn.Module):
    """Thin adapter around the official FG-CLIP v1 checkpoint API.

    Reference encoding derives dense and global features from one official vision-model
    output. Target/gallery encoding uses only the checkpoint's global-image API. The
    adapter never substitutes FG-CLIP2.
    """

    def __init__(
        self,
        model: nn.Module,
        internal_width: int = 256,
        train_vision: bool = True,
        train_text: bool = True,
        train_text_projection: bool = False,
        checkpoint: str | None = None,
        revision: str | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.internal_width = internal_width
        self.train_vision = train_vision
        self.train_text = train_text
        self.train_text_projection = train_text_projection
        self.checkpoint = checkpoint
        self.revision = revision
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
            regime.checkpoint,
            revision=regime.revision,
            trust_remote_code=regime.trust_remote_code,
        )
        backbone = cls(
            model,
            internal_width=internal_width,
            train_vision=regime.train_vision,
            train_text=regime.train_text,
            train_text_projection=regime.train_text_projection,
            checkpoint=regime.checkpoint,
            revision=regime.revision,
        )
        return backbone

    @staticmethod
    def load_processor(
        checkpoint: str, revision: str, trust_remote_code: bool = True
    ) -> tuple[Any, Any]:
        from transformers import AutoImageProcessor, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint,
            revision=revision,
            trust_remote_code=trust_remote_code,
            # The pinned repository declares a custom FGCLIPConfig but ships the
            # standard CLIP tokenizer files. This explicit hint avoids asking the
            # Auto registry for a nonexistent custom tokenizer class.
            tokenizer_type="clip",
        )
        image_processor = AutoImageProcessor.from_pretrained(
            checkpoint, revision=revision, trust_remote_code=trust_remote_code
        )
        return tokenizer, image_processor

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
        if not self.train_vision:
            self.model.vision_model.eval()
            self.model.visual_projection.eval()
        if not self.train_text:
            self.model.text_model.eval()
        if not self.train_text_projection:
            self.model.text_projection.eval()
        if hasattr(self.model, "text_filip_projection"):
            self.model.text_filip_projection.eval()

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

    def reference_features_from_vision_outputs(self, vision_outputs: Any) -> tuple[Tensor, Tensor]:
        """Reconstruct official dense/global FG-CLIP features from one vision output."""

        if vision_outputs.hidden_states is None:
            raise RuntimeError("FG-CLIP vision_model did not return hidden states")
        dense = self.model.forward_without_attn(vision_outputs.hidden_states[-2])[:, 1:]
        dense = self.model.vision_model.post_layernorm(dense)
        dense = self.model.visual_projection(dense)
        global_features = self.model.visual_projection(vision_outputs.pooler_output)
        return dense, global_features

    def encode_reference_images(self, pixel_values: Tensor) -> tuple[Tensor, Tensor]:
        """Return projected patch anchor and global embedding from one vision pass.

        This mirrors the official FG-CLIP v1 dense/global helpers: the dense branch
        uses the penultimate hidden state, ``forward_without_attn``, drops CLS, then
        applies post-layernorm and ``visual_projection``; the global branch projects
        the vision pooler output.
        """

        if pixel_values.ndim != 4:
            raise ValueError("pixel_values must be [B,C,H,W]")
        context = nullcontext() if self.train_vision else torch.no_grad()
        with context:
            vision_outputs = self.model.vision_model(
                pixel_values=pixel_values, output_hidden_states=True, return_dict=True
            )
            dense, global_features = self.reference_features_from_vision_outputs(vision_outputs)
        anchor = self.anchor_projection(dense)
        reference_global = normalize_fp32(global_features, dim=-1)
        return anchor, reference_global

    def encode_global_images(self, pixel_values: Tensor) -> Tensor:
        """Return retrieval embeddings without constructing dense patch features."""

        if pixel_values.ndim != 4:
            raise ValueError("pixel_values must be [B,C,H,W]")
        context = nullcontext() if self.train_vision else torch.no_grad()
        with context:
            global_features = self.model.get_image_features(pixel_values=pixel_values)
        return normalize_fp32(global_features, dim=-1)

    def encode_text(
        self, input_ids: Tensor, attention_mask: Tensor, content_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        outputs = self.model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            walk_short_pos=True,
        )
        text_tokens = self.text_projection(outputs.last_hidden_state)
        text_global = masked_text_mean(text_tokens, content_mask)
        # This is the checkpoint's trained retrieval-space text representation, not
        # a detached random auxiliary head. It remains auxiliary-only in IAG-SRME.
        text_semantic_global = normalize_fp32(
            self.model.text_projection(outputs.pooler_output), dim=-1
        )
        return text_tokens, text_global, text_semantic_global

    def forward(
        self,
        reference_pixels: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        content_mask: Tensor,
    ) -> BackboneOutput:
        anchor, reference_global = self.encode_reference_images(reference_pixels)
        text_tokens, text_global, text_semantic_global = self.encode_text(
            input_ids, attention_mask, content_mask
        )
        return BackboneOutput(
            anchor=anchor,
            reference_global=reference_global,
            text_tokens=text_tokens,
            text_global=text_global,
            text_semantic_global=text_semantic_global,
            text_content_mask=content_mask,
        )


def assert_cache_legal(train_vision: bool, image_cache_path: str | None) -> None:
    if train_vision and image_cache_path is not None:
        raise ValueError(
            "persistent image-feature caches are illegal when FG-CLIP vision is trainable"
        )
