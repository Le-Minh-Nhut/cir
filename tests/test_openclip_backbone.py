from __future__ import annotations

import inspect
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn

from losses.objective import IAGSRMEObjective, ObjectiveConfig
from models.iag_srme import IAGSRME, IAGSRMEConfig, IAGSRMECore
from models.iag_srme.openclip_backbone import OpenCLIPBackbone


class TinyTransformer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.projection = nn.Linear(width, width)

    def forward(self, values: Tensor) -> Tensor:
        return self.projection(values)


class TinyVisual(nn.Module):
    def __init__(self, width: int = 8, retrieval_dim: int = 6) -> None:
        super().__init__()
        self.image_size = (224, 224)
        self.patch_size = (16, 16)
        self.grid_size = (14, 14)
        self.output_dim = retrieval_dim
        self.proj = nn.Parameter(torch.randn(width, retrieval_dim) / width**0.5)
        self.patch_embedding = nn.Conv2d(3, width, kernel_size=16, stride=16)
        self.transformer = TinyTransformer(width)
        self.norm = nn.LayerNorm(width)

    def contextual_tokens(self, pixels: Tensor) -> Tensor:
        patches = self.patch_embedding(pixels).flatten(2).transpose(1, 2)
        return self.norm(self.transformer(patches))


class TinyOpenCLIP(nn.Module):
    def __init__(self, vision_width: int = 8, text_width: int = 7, retrieval_dim: int = 6) -> None:
        super().__init__()
        self.visual = TinyVisual(vision_width, retrieval_dim)
        self.context_length = 9
        self.token_embedding = nn.Embedding(32, text_width)
        self.positional_embedding = nn.Parameter(torch.randn(self.context_length, text_width))
        self.transformer = TinyTransformer(text_width)
        self.ln_final = nn.LayerNorm(text_width)
        self.text_projection = nn.Parameter(torch.randn(text_width, retrieval_dim))
        self.logit_scale = nn.Parameter(torch.ones(()))
        self.image_intermediate_calls = 0
        self.global_image_calls = 0

    def forward_intermediates(
        self,
        *,
        image: Tensor | None = None,
        text: Tensor | None = None,
        **kwargs,
    ) -> dict[str, Tensor | list[Tensor]]:
        del kwargs
        result: dict[str, Tensor | list[Tensor]] = {}
        if image is not None:
            self.image_intermediate_calls += 1
            tokens = self.visual.contextual_tokens(image)
            global_features = tokens.mean(dim=1) @ self.visual.proj
            result["image_intermediates"] = [tokens]
            result["image_features"] = F.normalize(global_features, dim=-1)
        if text is not None:
            tokens = self.token_embedding(text) + self.positional_embedding
            tokens = self.ln_final(self.transformer(tokens))
            eos_positions = text.argmax(dim=-1)
            pooled = tokens[torch.arange(text.shape[0]), eos_positions]
            result["text_intermediates"] = [tokens]
            result["text_features"] = F.normalize(pooled @ self.text_projection, dim=-1)
        return result

    def encode_image(self, pixels: Tensor, normalize: bool = True) -> Tensor:
        self.global_image_calls += 1
        tokens = self.visual.contextual_tokens(pixels)
        features = tokens.mean(dim=1) @ self.visual.proj
        return F.normalize(features, dim=-1) if normalize else features


def _inputs(batch_size: int = 2) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    pixels = torch.randn(batch_size, 3, 224, 224)
    input_ids = torch.tensor([[31, 4, 5, 30, 0, 0, 0, 0, 0]]).expand(batch_size, -1)
    attention_mask = input_ids.ne(0)
    content_mask = attention_mask.clone()
    content_mask[:, 0] = False
    content_mask[:, 3] = False
    return pixels, input_ids, attention_mask, content_mask


def _build_model() -> IAGSRME:
    backbone = OpenCLIPBackbone(
        TinyOpenCLIP(),
        internal_width=8,
        train_vision=True,
        train_text=True,
        train_text_projection=False,
    )
    core = IAGSRMECore(
        IAGSRMEConfig(
            width=8,
            num_candidates=4,
            max_steps=3,
            num_heads=2,
            retrieval_dim=6,
            selector_gumbel_noise=False,
        )
    )
    with torch.no_grad():
        core.scorer.score_head[-1].weight.zero_()
        core.scorer.score_head[-1].bias.fill_(0.5)
    return IAGSRME(backbone, core)


def test_openclip_backbone_visual_and_text_contracts() -> None:
    model = _build_model().eval()
    pixels, input_ids, attention_mask, content_mask = _inputs()
    encoded = model.backbone(pixels, input_ids, attention_mask, content_mask)

    assert encoded.anchor.shape == (2, 196, 8)
    assert encoded.reference_global.shape == (2, 6)
    assert encoded.text_tokens.shape == (2, 9, 8)
    assert encoded.text_global.shape == (2, 8)
    assert encoded.text_semantic_global.shape == (2, 6)
    assert torch.isfinite(encoded.anchor).all()


def test_openclip_reference_is_single_pass_and_gallery_is_global_only() -> None:
    model = _build_model().eval()
    pixels, _, _, _ = _inputs()

    anchor, reference_global = model.backbone.encode_reference_images(pixels)
    gallery_global = model.encode_global_images(pixels)

    assert anchor.shape == (2, 196, 8)
    assert reference_global.shape == gallery_global.shape == (2, 6)
    assert model.backbone.model.image_intermediate_calls == 1
    assert model.backbone.model.global_image_calls == 1


def test_openclip_full_forward_and_core_gradients() -> None:
    torch.manual_seed(301)
    model = _build_model().train()
    objective = IAGSRMEObjective(ObjectiveConfig(), width=8)
    reference, input_ids, attention_mask, content_mask = _inputs(batch_size=3)
    target = torch.randn_like(reference)
    output = model(reference, input_ids, attention_mask, content_mask)
    target_embeddings = model.encode_global_images(target)
    loss = objective(output, target_embeddings, torch.eye(3, dtype=torch.bool))["total"]
    loss.backward()

    families = {
        "vision": next(model.backbone.vision_encoder_parameters()),
        "text": next(model.backbone.text_encoder_parameters()),
        "intent": model.core.intent_encoder.query_bank,
        "grounder": model.core.grounder.intent_projection.weight,
        "editor": model.core.editor.direction.weight,
        "readout": model.core.readout.output_projection.weight,
        "scorer": model.core.scorer.score_head[-1].weight,
    }
    assert torch.isfinite(output.final_query).all()
    assert torch.isfinite(loss)
    for name, parameter in families.items():
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name


def test_openclip_trainability_matches_base_full_policy() -> None:
    model = _build_model()
    assert all(parameter.requires_grad for parameter in model.backbone.model.visual.parameters())
    assert all(parameter.requires_grad for parameter in model.backbone.text_encoder_parameters())
    assert not model.backbone.model.text_projection.requires_grad
    assert not model.backbone.model.logit_scale.requires_grad


def test_openclip_forward_contract_has_no_target_argument() -> None:
    parameters = inspect.signature(IAGSRME.forward).parameters
    assert "target_pixels" not in parameters
    assert "target_features" not in parameters


def test_openclip_experiment_matches_baseline_hyperparameters() -> None:
    root = Path(__file__).parents[1]
    baseline = yaml.safe_load(
        (root / "conf/experiment/iag_srme_base_full.yaml").read_text(encoding="utf-8")
    )
    ablation = yaml.safe_load(
        (root / "conf/experiment/iag_srme_openclip_b16_valsplit.yaml").read_text(
            encoding="utf-8"
        )
    )
    matched = {
        "epochs",
        "batch_size",
        "eval_batch_size",
        "gallery_batch_size",
        "num_workers",
        "learning_rate",
        "weight_decay",
        "train_caption_policy",
        "val_caption_policy",
    }
    assert {key: ablation[key] for key in matched} == {
        key: baseline[key] for key in matched
    }
