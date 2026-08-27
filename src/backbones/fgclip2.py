from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn
from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer

FGCLIP2_LARGE_MODEL_ID = "qihoo360/fg-clip2-large"
FGCLIP2_LARGE_REVISION = "4d1d5dc35c716902f07c172dbfc23b82a7bc6bf3"
FGCLIP2_LARGE_DIM = 1024
FGCLIP2_SHORT_TEXT_LENGTH = 64
FGCLIP2_PATCH_SIZE = 16
FGCLIP2_DYNAMIC_PATCH_BUDGETS = (128, 256, 576, 784, 1024)
FGCLIP2_PATCH_POLICY_NAME = "official_fgclip2_dynamic"


def validate_fgclip2_revision(revision: str) -> str:
    if not revision or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("FG-CLIP2 revision must be a full immutable 40-character git SHA")
    return revision


def determine_max_num_patches(image: Image.Image) -> int:
    """Mirror ``determine_max_value`` from the pinned official README."""

    width, height = image.size
    patch_count = (width // FGCLIP2_PATCH_SIZE) * (height // FGCLIP2_PATCH_SIZE)
    if patch_count > 784:
        return 1024
    if patch_count > 576:
        return 784
    if patch_count > 256:
        return 576
    if patch_count > 128:
        return 256
    return 128


class FGCLIP2Backbone(nn.Module):
    """Frozen adapter for the official FG-CLIP2-Large Hugging Face model."""

    def __init__(
        self,
        *,
        model_id: str = FGCLIP2_LARGE_MODEL_ID,
        revision: str = FGCLIP2_LARGE_REVISION,
        max_text_length: int = FGCLIP2_SHORT_TEXT_LENGTH,
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
        revision = validate_fgclip2_revision(revision)

        load_kwargs: dict[str, Any] = {
            "revision": revision,
            "trust_remote_code": True,
        }
        if dtype is not None:
            load_kwargs["dtype"] = dtype

        self.model_id = model_id
        self.revision = revision
        self.max_text_length = max_text_length
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=True,
        )
        self.image_processor = AutoImageProcessor.from_pretrained(
            model_id,
            revision=revision,
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

    def train(self, mode: bool = True) -> FGCLIP2Backbone:
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
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        self._assert_frozen()

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

        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(
            device=self.device,
            dtype=torch.bool,
        )

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

        global_features = F.normalize(
            outputs.pooler_output.float(),
            dim=-1,
        )

        content_mask = attention_mask & ~special_tokens_mask

        return (
            states,
            attention_mask,
            content_mask,
            global_features,
        )

    @torch.inference_mode()
    def encode_text_global(self, captions: Sequence[str]) -> Tensor:
        """Return normalized official short-walk text embeddings [B,1024]."""

        self._assert_frozen()
        if not captions:
            raise ValueError("captions must not be empty")
        tokenized = self.tokenizer(
            list(captions),
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_attention_mask=True,
            return_tensors="pt",
        ).to(self.device)
        features = self.model.get_text_features(
            **tokenized,
            walk_type="short",
        )
        features = F.normalize(features.float(), dim=-1)
        expected_shape = (len(captions), FGCLIP2_LARGE_DIM)
        if tuple(features.shape) != expected_shape:
            raise RuntimeError(
                f"Expected FG-CLIP2 text globals {expected_shape}, got {tuple(features.shape)}"
            )
        if not torch.isfinite(features).all() or features.requires_grad:
            raise FloatingPointError("Invalid frozen FG-CLIP2 text global features")
        return features

    @torch.inference_mode()
    def encode_image_global(self, images: Sequence[Image.Image]) -> Tensor:
        """Return normalized global image embeddings with shape [B,1024]."""

        self._assert_frozen()
        if not images:
            raise ValueError("images must not be empty")

        grouped_indices: dict[int, list[int]] = {}
        for index, image in enumerate(images):
            budget = determine_max_num_patches(image)
            grouped_indices.setdefault(budget, []).append(index)

        ordered_features: Tensor | None = None
        for budget, indices in grouped_indices.items():
            grouped_images = [images[index] for index in indices]
            inputs = self.image_processor(
                images=grouped_images,
                max_num_patches=budget,
                return_tensors="pt",
            ).to(self.device)
            grouped_features = self.model.get_image_features(**inputs)
            grouped_features = F.normalize(grouped_features.float(), dim=-1)
            if ordered_features is None:
                ordered_features = torch.empty(
                    len(images),
                    grouped_features.shape[-1],
                    dtype=grouped_features.dtype,
                    device=grouped_features.device,
                )
            ordered_features[indices] = grouped_features

        assert ordered_features is not None
        features = ordered_features

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

    @torch.inference_mode()
    def encode_image_dense(
        self,
        images: Sequence[Image.Image],
    ) -> tuple[list[Tensor], Tensor]:
        """Return real (unpadded) dense tokens and their [height,width] grids.

        The pinned model card defines the real token prefix as
        ``spatial_shapes[b, 0] * spatial_shapes[b, 1]``.  We deliberately
        preserve ragged outputs instead of padding them inside the backbone.
        """

        self._assert_frozen()
        if not images:
            raise ValueError("images must not be empty")

        grouped_indices: dict[int, list[int]] = {}
        for index, image in enumerate(images):
            grouped_indices.setdefault(determine_max_num_patches(image), []).append(index)

        ordered_tokens: list[Tensor | None] = [None] * len(images)
        ordered_shapes = torch.empty(
            len(images), 2, dtype=torch.long, device=self.device
        )
        for budget, indices in grouped_indices.items():
            inputs = self.image_processor(
                images=[images[index] for index in indices],
                max_num_patches=budget,
                return_tensors="pt",
            ).to(self.device)
            if "spatial_shapes" not in inputs:
                raise RuntimeError("Pinned FG-CLIP2 processor omitted spatial_shapes")
            spatial_shapes = inputs["spatial_shapes"].to(dtype=torch.long)
            if tuple(spatial_shapes.shape) != (len(indices), 2):
                raise RuntimeError(
                    "Expected spatial_shapes [B,2], got "
                    f"{tuple(spatial_shapes.shape)}"
                )
            dense = self.model.get_image_dense_feature(**inputs)
            if dense.ndim != 3 or dense.shape[0] != len(indices):
                raise RuntimeError(
                    "FG-CLIP2 dense output must be [B,N,D], got "
                    f"{tuple(dense.shape)}"
                )
            if dense.shape[-1] != FGCLIP2_LARGE_DIM:
                raise RuntimeError(
                    f"FG-CLIP2 dense feature dim must be {FGCLIP2_LARGE_DIM}"
                )
            pixel_attention_mask = inputs.get("pixel_attention_mask")
            for local_index, original_index in enumerate(indices):
                height, width = spatial_shapes[local_index].tolist()
                real_count = int(height * width)
                if height <= 0 or width <= 0 or real_count > dense.shape[1]:
                    raise RuntimeError(
                        "Invalid FG-CLIP2 dense spatial shape: "
                        f"shape=({height},{width}), available={dense.shape[1]}"
                    )
                if pixel_attention_mask is not None:
                    flattened_mask = pixel_attention_mask[local_index].reshape(-1).bool()
                    # Some processor versions expose a patch-level mask while
                    # others expose a pixel-level mask. Only the former is
                    # directly comparable with dense-token positions.
                    if flattened_mask.numel() == dense.shape[1] and (
                            int(flattened_mask.sum().item()) != real_count
                            or not flattened_mask[:real_count].all()
                            or flattened_mask[real_count:].any()
                    ):
                        raise RuntimeError(
                            "pixel_attention_mask disagrees with spatial_shapes"
                        )
                tokens = dense[local_index, :real_count].float()
                if tuple(tokens.shape) != (real_count, FGCLIP2_LARGE_DIM):
                    raise RuntimeError("Unexpected sliced FG-CLIP2 dense-token shape")
                if not torch.isfinite(tokens).all() or tokens.requires_grad:
                    raise FloatingPointError("Invalid frozen FG-CLIP2 dense features")
                ordered_tokens[original_index] = tokens
                ordered_shapes[original_index] = spatial_shapes[local_index]

        if any(tokens is None for tokens in ordered_tokens):
            raise RuntimeError("FG-CLIP2 dense batching failed to restore input order")
        return [tokens for tokens in ordered_tokens if tokens is not None], ordered_shapes
