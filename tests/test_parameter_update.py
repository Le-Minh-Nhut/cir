from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn
from torch.optim import AdamW

from losses.objective import IAGSRMEObjective, ObjectiveConfig
from models.iag_srme import FGCLIPBackbone, IAGSRME, IAGSRMEConfig, IAGSRMECore
from training.engine import assert_training_setup, trainable_parameters


class TinyVisionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.channel_projection = nn.Parameter(torch.randn(3, 6))
        self.post_layernorm = nn.LayerNorm(6)

    def forward(self, pixel_values, output_hidden_states=False, return_dict=True):
        del output_hidden_states, return_dict
        pooled = pixel_values.mean(dim=(-2, -1)) @ self.channel_projection
        offsets = torch.linspace(-0.2, 0.2, 5, device=pooled.device)[None, :, None]
        tokens = pooled[:, None, :] + offsets
        return SimpleNamespace(hidden_states=(tokens * 0.7, tokens), pooler_output=pooled)


class TinyTextModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, 7)

    def forward(self, input_ids, attention_mask, return_dict=True, walk_short_pos=True):
        del return_dict, walk_short_pos
        hidden = self.embedding(input_ids)
        weights = attention_mask.to(hidden.dtype)
        pooled = (hidden * weights[..., None]).sum(dim=1) / weights.sum(
            dim=1, keepdim=True
        ).clamp_min(1)
        return SimpleNamespace(last_hidden_state=hidden, pooler_output=pooled)


class TinyFGCLIP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            projection_dim=8, text_config=SimpleNamespace(hidden_size=7)
        )
        self.vision_model = TinyVisionModel()
        self.visual_projection = nn.Linear(6, 8, bias=False)
        self.text_model = TinyTextModel()
        self.text_projection = nn.Linear(7, 8, bias=False)

    def forward_without_attn(self, hidden_state):
        return hidden_state

    def get_image_features(self, pixel_values):
        output = self.vision_model(pixel_values=pixel_values, return_dict=True)
        return self.visual_projection(output.pooler_output)


def _build(train_vision: bool) -> tuple[IAGSRME, IAGSRMEObjective]:
    torch.manual_seed(101)
    backbone = FGCLIPBackbone(TinyFGCLIP(), internal_width=8, train_vision=train_vision)
    core = IAGSRMECore(
        IAGSRMEConfig(
            width=8,
            num_candidates=4,
            max_steps=2,
            num_heads=2,
            retrieval_dim=8,
            selector_gumbel_noise=False,
        )
    )
    with torch.no_grad():
        core.scorer.score_head[-1].weight.zero_()
        core.scorer.score_head[-1].bias.fill_(0.5)
    return IAGSRME(backbone, core), IAGSRMEObjective(ObjectiveConfig(), width=8)


def _one_update(model: IAGSRME, objective: IAGSRMEObjective) -> None:
    device = torch.device("cpu")
    model.to(device).train()
    objective.to(device).train()
    optimizer = AdamW(trainable_parameters(model, objective), lr=1e-2, weight_decay=0.0)
    assert_training_setup(model, objective, optimizer, device)
    reference = torch.randn(4, 3, 4, 4)
    target = torch.randn(4, 3, 4, 4)
    input_ids = torch.randint(0, 32, (4, 6))
    mask = torch.ones(4, 6, dtype=torch.long)
    content_mask = mask.bool()

    output = model(reference, input_ids, mask, content_mask)
    target_embeddings = model.encode_global_images(target)
    positives = torch.eye(4, dtype=torch.bool)
    loss = objective(output, target_embeddings, positives)["total"]
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()


def test_base_full_optimizer_step_changes_all_expected_parameter_families() -> None:
    model, objective = _build(train_vision=True)
    selected = {
        "intent": model.core.intent_encoder.query_bank,
        "grounding": model.core.grounder.intent_projection.weight,
        "editor": model.core.editor.direction.weight,
        "readout": model.core.readout.output_projection.weight,
        "scorer": model.core.scorer.score_head[-1].weight,
        "text": model.backbone.model.text_model.embedding.weight,
        "vision": model.backbone.model.vision_model.channel_projection,
    }
    before = {name: parameter.detach().clone() for name, parameter in selected.items()}

    _one_update(model, objective)

    for name, parameter in selected.items():
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name
        assert not torch.equal(before[name], parameter.detach()), name


def test_large_style_frozen_vision_does_not_change_but_text_does() -> None:
    model, objective = _build(train_vision=False)
    vision = model.backbone.model.vision_model.channel_projection
    projection = model.backbone.model.visual_projection.weight
    text = model.backbone.model.text_model.embedding.weight
    before = {
        "vision": vision.detach().clone(),
        "projection": projection.detach().clone(),
        "text": text.detach().clone(),
    }

    _one_update(model, objective)

    assert vision.grad is None and torch.equal(before["vision"], vision.detach())
    assert projection.grad is None and torch.equal(before["projection"], projection.detach())
    assert text.grad is not None and text.grad.abs().sum() > 0
    assert not torch.equal(before["text"], text.detach())
