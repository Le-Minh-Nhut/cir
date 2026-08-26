from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn
from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer


FGCLIP2_LARGE_MODEL_ID = "qihoo360/fg-clip2-large"
FGCLIP2_LARGE_DIM = 1024
FGCLIP2_SHORT_TEXT_LENGTH = 64


class FGCLIP2Backbone(nn.Module):
    """Frozen adapter for the official FG-CLIP2-Large Hugging Face model."""

    def __init__(
        self,
        *,
        model_id: str = FGCLIP2_LARGE_MODEL_ID,
        max_text_length: int = FGCLIP2_SHORT_TEXT_LENGTH,
        max_num_patches: int = 784,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if model_id != FGCLIP2_LARGE_MODEL_ID:
            raise ValueError(
                "This experiment requires exactly "
                f"{FGCLIP2_LARGE_MODEL_ID!r}; got {model_id!r}"
            )
        if max_text_length != FGCLIP2_SHORT_TEXT_LENGTH:
            raise ValueError(
                "FG-CLIP2 short-text mode requires max_text_length=64; "
                "audit truncation before changing text strategy"
            )
        if max_num_patches < 1:
            raise ValueError("max_num_patches must be >= 1")

        load_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if dtype is not None:
            load_kwargs["dtype"] = dtype

        self.model_id = model_id
        self.max_text_length = max_text_length
        self.max_num_patches = max_num_patches
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        self.image_processor = AutoImageProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

        text_dim = int(self.model.config.text_config.hidden_size)
        image_dim = int(self.model.config.vision_config.hidden_size)
        if text_dim != FGCLIP2_LARGE_DIM or image_dim != FGCLIP2_LARGE_DIM:
            raise RuntimeError(
                "Unexpected FG-CLIP2-Large dimensions: "
                f"text={text_dim}, image={image_dim}"
            )

        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def train(self, mode: bool = True) -> "FGCLIP2Backbone":
        # This adapter is intentionally frozen even when a parent module is trained.
        super().train(False)
        self.model.eval()
        return self

    def _assert_frozen(self) -> None:
        if self.model.training:
            raise RuntimeError("FG-CLIP2-Large must remain in eval mode")
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("FG-CLIP2-Large contains trainable parameters")

    @torch.inference_mode()
    def encode_text_tokens(
        self,
        captions: Sequence[str],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return contextual token states plus attention/content masks."""

        self._assert_frozen()
        if not captions:
            raise ValueError("captions must not be empty")

        tokenized = self.tokenizer(
            list(captions),
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_attention_mask=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        special_tokens_mask = tokenized.pop("special_tokens_mask").to(
            device=self.device,
            dtype=torch.bool,
        )
        input_ids = tokenized["input_ids"].to(device=self.device)
        attention_mask = tokenized["attention_mask"].to(
            device=self.device,
            dtype=torch.bool,
        )
        # The official remote implementation registers ``position_ids`` twice.
        # Supplying the canonical short-text positions avoids relying on that
        # non-persistent buffer when Transformers initializes through meta tensors.
        position_ids = torch.arange(
            self.max_text_length,
            device=self.device,
            dtype=torch.long,
        ).unsqueeze(0)

        outputs = self.model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            walk_type="short",
        )
        states = outputs.last_hidden_state.float()
        content_mask = attention_mask & ~special_tokens_mask

        expected_shape = (len(captions), self.max_text_length, FGCLIP2_LARGE_DIM)
        if tuple(states.shape) != expected_shape:
            raise RuntimeError(
                f"Expected FG-CLIP2-Large text states {expected_shape}, "
                f"got {tuple(states.shape)}"
            )
        if content_mask.shape != attention_mask.shape:
            raise RuntimeError("FG-CLIP2 text mask shape mismatch")
        if (content_mask & ~attention_mask).any():
            raise RuntimeError("content_mask includes padding tokens")
        if not torch.isfinite(states).all():
            raise FloatingPointError("FG-CLIP2 text states contain NaN or Inf")
        if states.requires_grad:
            raise RuntimeError("FG-CLIP2 text precompute recorded gradients")

        return states, attention_mask, content_mask

    @torch.inference_mode()
    def encode_image_global(self, images: Sequence[Image.Image]) -> Tensor:
        """Return normalized global image embeddings with shape [B,1024]."""

        self._assert_frozen()
        if not images:
            raise ValueError("images must not be empty")

        inputs = self.image_processor(
            images=list(images),
            max_num_patches=self.max_num_patches,
            return_tensors="pt",
        ).to(self.device)
        features = self.model.get_image_features(**inputs)
        features = F.normalize(features.float(), dim=-1)

        expected_shape = (len(images), FGCLIP2_LARGE_DIM)
        if tuple(features.shape) != expected_shape:
            raise RuntimeError(
                f"Expected FG-CLIP2-Large image features {expected_shape}, "
                f"got {tuple(features.shape)}"
            )
        if not torch.isfinite(features).all():
            raise FloatingPointError("FG-CLIP2 image features contain NaN or Inf")
        if features.requires_grad:
            raise RuntimeError("FG-CLIP2 image precompute recorded gradients")

        return features
