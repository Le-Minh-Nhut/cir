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
class OpenCLIPRegime:
    model_name: str
    pretrained: str
    library_version: str
    weights_repository: str
    weights_revision: str
    train_vision: bool
    train_text: bool = True
    train_text_projection: bool = False


class OpenCLIPTokenizerAdapter:
    """Expose OpenCLIP tokenization through the collator's tokenizer contract."""

    def __init__(self, tokenizer: Any, context_length: int) -> None:
        self.tokenizer = tokenizer
        self.context_length = context_length

    def __call__(
        self,
        texts: list[str],
        *,
        max_length: int,
        padding: str,
        truncation: bool,
        return_tensors: str,
    ) -> dict[str, Tensor]:
        if max_length != self.context_length:
            raise ValueError("OpenCLIP tokenizer length must match its pretrained context length")
        if padding != "max_length" or not truncation or return_tensors != "pt":
            raise ValueError("unsupported OpenCLIP tokenizer collation policy")
        input_ids = self.tokenizer(texts, context_length=max_length)
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(0),
        }


class OpenCLIPImageProcessorAdapter:
    """Expose the official deterministic OpenCLIP validation transform."""

    def __init__(self, transform: Any) -> None:
        self.transform = transform

    def preprocess(self, images: list[Any], return_tensors: str) -> dict[str, Tensor]:
        if return_tensors != "pt":
            raise ValueError("OpenCLIP image processor only returns PyTorch tensors")
        return {"pixel_values": torch.stack([self.transform(image) for image in images])}


class OpenCLIPBackbone(nn.Module):
    """Controlled OpenCLIP ViT adapter for the unchanged IAG-SRME core contract."""

    backbone_type = "openclip"
    library = "open_clip_torch"

    def __init__(
        self,
        model: nn.Module,
        internal_width: int = 256,
        train_vision: bool = True,
        train_text: bool = True,
        train_text_projection: bool = False,
        model_name: str = "ViT-B-16",
        pretrained: str = "laion2b_s34b_b88k",
        library_version: str = "3.3.0",
        weights_repository: str = "laion/CLIP-ViT-B-16-laion2B-s34B-b88K",
        weights_revision: str = "7288da5a0d6f0b51c4a2b27c624837a9236d0112",
    ) -> None:
        super().__init__()
        if model_name != "ViT-B-16":
            raise ValueError("controlled ablation requires OpenCLIP ViT-B-16")
        visual = model.visual
        if tuple(visual.image_size) != (224, 224) or tuple(visual.patch_size) != (16, 16):
            raise ValueError("controlled ablation requires native 224px ViT-B/16 preprocessing")
        if tuple(visual.grid_size) != (14, 14):
            raise ValueError("OpenCLIP visual grid must be 14x14")
        if visual.proj is None:
            raise ValueError("OpenCLIP visual tower must expose its pretrained projection")

        self.model = model
        self.internal_width = internal_width
        self.train_vision = train_vision
        self.train_text = train_text
        self.train_text_projection = train_text_projection
        self.checkpoint = model_name
        self.revision = pretrained
        self.library_version = library_version
        self.weights_repository = weights_repository
        self.weights_revision = weights_revision
        self.image_size = 224
        self.patch_tokens = 196
        self.retrieval_dim = int(visual.output_dim)
        visual_projection_width = int(visual.proj.shape[-1])
        text_width = int(model.transformer.width)
        if visual_projection_width != self.retrieval_dim:
            raise ValueError("OpenCLIP visual projection/retrieval dimensions differ")

        self.anchor_projection = nn.Sequential(
            nn.Linear(self.retrieval_dim, internal_width), nn.LayerNorm(internal_width)
        )
        self.text_projection = nn.Sequential(
            nn.Linear(text_width, internal_width), nn.LayerNorm(internal_width)
        )
        self._apply_freeze_policy()

    @classmethod
    def from_pretrained(
        cls, regime: OpenCLIPRegime, internal_width: int = 256
    ) -> tuple[OpenCLIPBackbone, OpenCLIPTokenizerAdapter, OpenCLIPImageProcessorAdapter]:
        import open_clip
        from huggingface_hub import hf_hub_download

        if open_clip.__version__ != regime.library_version:
            raise RuntimeError(
                "OpenCLIP version mismatch: "
                f"configured={regime.library_version}, installed={open_clip.__version__}"
            )
        weights_path = hf_hub_download(
            repo_id=regime.weights_repository,
            filename="open_clip_model.safetensors",
            revision=regime.weights_revision,
        )
        preprocessing = open_clip.get_pretrained_cfg(
            regime.model_name, regime.pretrained
        )
        model, _, validation_transform = open_clip.create_model_and_transforms(
            regime.model_name,
            pretrained=weights_path,
            force_image_size=224,
            image_mean=tuple(preprocessing["mean"]),
            image_std=tuple(preprocessing["std"]),
            image_interpolation=str(preprocessing["interpolation"]),
            image_resize_mode=str(preprocessing["resize_mode"]),
        )
        tokenizer = open_clip.get_tokenizer(
            regime.model_name, context_length=int(model.context_length)
        )
        backbone = cls(
            model,
            internal_width=internal_width,
            train_vision=regime.train_vision,
            train_text=regime.train_text,
            train_text_projection=regime.train_text_projection,
            model_name=regime.model_name,
            pretrained=regime.pretrained,
            library_version=regime.library_version,
            weights_repository=regime.weights_repository,
            weights_revision=regime.weights_revision,
        )
        return (
            backbone,
            OpenCLIPTokenizerAdapter(tokenizer, int(model.context_length)),
            OpenCLIPImageProcessorAdapter(validation_transform),
        )

    def vision_encoder_parameters(self):
        return self.model.visual.parameters()

    def text_encoder_parameters(self):
        modules = (self.model.token_embedding, self.model.transformer, self.model.ln_final)
        yield self.model.positional_embedding
        for module in modules:
            yield from module.parameters()

    @property
    def vision_encoder_module(self) -> nn.Module:
        return self.model.visual

    @property
    def vision_call_module(self) -> nn.Module:
        # Both the intermediate reference path and global-only path invoke the
        # patch convolution exactly once per underlying visual-tower pass.
        return self.model.visual.conv1

    @property
    def text_encoder_module(self) -> nn.Module:
        return self.model.transformer

    def _apply_freeze_policy(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        for parameter in self.model.visual.parameters():
            parameter.requires_grad_(self.train_vision)
        for parameter in self.text_encoder_parameters():
            parameter.requires_grad_(self.train_text)
        self.model.text_projection.requires_grad_(self.train_text_projection)
        if not self.train_vision:
            self.model.visual.eval()
        if not self.train_text:
            self.model.transformer.eval()
            self.model.token_embedding.eval()
            self.model.ln_final.eval()

    def train(self, mode: bool = True) -> OpenCLIPBackbone:
        super().train(mode)
        if not self.train_vision:
            self.model.visual.eval()
        if not self.train_text:
            self.model.transformer.eval()
            self.model.token_embedding.eval()
            self.model.ln_final.eval()
        return self

    def encode_reference_images(self, pixel_values: Tensor) -> tuple[Tensor, Tensor]:
        if pixel_values.ndim != 4 or tuple(pixel_values.shape[-2:]) != (224, 224):
            raise ValueError("OpenCLIP reference pixels must be [B,3,224,224]")
        context = nullcontext() if self.train_vision else torch.no_grad()
        with context:
            outputs = self.model.forward_intermediates(
                image=pixel_values,
                image_indices=1,
                normalize=True,
                normalize_intermediates=True,
                image_output_fmt="NLC",
            )
            contextual_tokens = outputs["image_intermediates"][-1]
            if contextual_tokens.shape[1] != self.patch_tokens:
                raise AssertionError("OpenCLIP anchor must contain exactly 196 patch tokens")
            dense_retrieval_tokens = contextual_tokens @ self.model.visual.proj
            reference_global = outputs["image_features"]
        anchor = self.anchor_projection(dense_retrieval_tokens)
        return anchor, normalize_fp32(reference_global, dim=-1)

    def encode_global_images(self, pixel_values: Tensor) -> Tensor:
        if pixel_values.ndim != 4 or tuple(pixel_values.shape[-2:]) != (224, 224):
            raise ValueError("OpenCLIP global image pixels must be [B,3,224,224]")
        context = nullcontext() if self.train_vision else torch.no_grad()
        with context:
            global_features = self.model.encode_image(pixel_values, normalize=True)
        return normalize_fp32(global_features, dim=-1)

    def encode_text(
        self, input_ids: Tensor, attention_mask: Tensor, content_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        del attention_mask
        outputs = self.model.forward_intermediates(
            text=input_ids,
            text_indices=1,
            normalize=True,
            normalize_intermediates=True,
            text_output_fmt="NLC",
        )
        contextual_tokens = outputs["text_intermediates"][-1]
        text_tokens = self.text_projection(contextual_tokens)
        text_global = masked_text_mean(text_tokens, content_mask)
        text_semantic_global = normalize_fp32(outputs["text_features"], dim=-1)
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
