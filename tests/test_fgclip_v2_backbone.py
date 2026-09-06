from __future__ import annotations

import inspect
from types import SimpleNamespace

import torch
from torch import Tensor, nn

from models.iag_srme.utils.backbone import FGCLIPBackbone


class MockAttention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.num_heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.dropout = 0.0
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, state: Tensor) -> Tensor:
        batch, tokens, dim = state.shape
        q = self.q_proj(state).view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(state).view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(state).view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        attention = torch.softmax((q * self.scale) @ k.transpose(-2, -1), dim=-1)
        value = (attention @ v).transpose(1, 2).reshape(batch, tokens, dim)
        return self.out_proj(value)


class MockBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(dim)
        self.self_attn = MockAttention(dim, 2)
        self.layer_norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, state: Tensor) -> Tensor:
        state = state + self.self_attn(self.layer_norm1(state))
        return state + self.mlp(self.layer_norm2(state))


class MockVision(nn.Module):
    def __init__(self, dim: int = 6) -> None:
        super().__init__()
        self.input = nn.Linear(3, dim)
        self.embeddings = SimpleNamespace(class_embedding=nn.Parameter(torch.randn(dim)))
        self.encoder = SimpleNamespace(layers=nn.ModuleList([MockBlock(dim)]))
        self.post_layernorm = nn.LayerNorm(dim)
        self.calls = 0

    def forward(self, pixel_values, output_hidden_states=False, return_dict=False):
        del output_hidden_states, return_dict
        self.calls += 1
        patches = pixel_values.permute(0, 2, 3, 1).reshape(pixel_values.shape[0], 4, 3)
        patches = self.input(patches)
        cls = self.embeddings.class_embedding[None, None].expand(patches.shape[0], 1, -1)
        penultimate = torch.cat([cls, patches], dim=1)
        final = self.encoder.layers[-1](penultimate)
        return SimpleNamespace(
            hidden_states=(penultimate, final),
            pooler_output=self.post_layernorm(final[:, 0]),
        )


class MockFGCLIP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            projection_dim=5,
            vision_config=SimpleNamespace(hidden_size=6, image_size=2, patch_size=1),
            text_config=SimpleNamespace(hidden_size=7),
        )
        self.vision_model = MockVision()
        self.visual_projection = nn.Linear(6, 5, bias=False)
        self.text_model = nn.Linear(7, 7)
        self.text_projection = nn.Linear(7, 5, bias=False)

    def forward_without_attn(self, state: Tensor) -> Tensor:
        block = self.vision_model.encoder.layers[-1]
        residual = state
        state = block.layer_norm1(state)
        state = block.self_attn.v_proj(state)
        state = block.self_attn.out_proj(state)
        state = residual + state
        return state + block.mlp(block.layer_norm2(state))

    def get_image_dense_features(self, pixel_values: Tensor) -> Tensor:
        output = self.vision_model(pixel_values, output_hidden_states=True, return_dict=True)
        dense = self.forward_without_attn(output.hidden_states[-2])[:, 1:]
        return self.visual_projection(self.vision_model.post_layernorm(dense))

    def get_image_features(self, pixel_values: Tensor) -> Tensor:
        output = self.vision_model(pixel_values, return_dict=True)
        return self.visual_projection(output.pooler_output)


def test_penultimate_patch_state_and_native_dense_parity_without_replay() -> None:
    torch.manual_seed(6)
    checkpoint = MockFGCLIP()
    backbone = FGCLIPBackbone(checkpoint, text_width=4).eval()
    pixels = torch.randn(2, 3, 2, 2)
    state = backbone.initial_state(pixels)
    calls = checkpoint.vision_model.calls
    actual = backbone.dense_readout(state)
    expected = checkpoint.get_image_dense_features(pixels)

    assert state.shape == (2, 4, 6)  # patch-only; no recurrent CLS
    assert torch.allclose(actual, expected, atol=1e-6)
    assert calls + 1 == checkpoint.vision_model.calls
    assert backbone.global_readout(state).shape == (2, 6)
    assert checkpoint.vision_model.calls == calls + 1  # readouts did not replay vision


def test_current_state_readouts_change_and_candidate_vectorization_matches_loop() -> None:
    torch.manual_seed(8)
    backbone = FGCLIPBackbone(MockFGCLIP(), text_width=4).eval()
    state = backbone.initial_state(torch.randn(2, 3, 2, 2))
    changed = state.clone()
    changed[:, 0, 0] += 2.0
    assert not torch.allclose(backbone.global_readout(state), backbone.global_readout(changed))
    assert not torch.allclose(backbone.dense_readout(state), backbone.dense_readout(changed))
    assert not torch.allclose(
        backbone.retrieval_readout(state), backbone.retrieval_readout(changed)
    )

    candidates = torch.stack([state, changed, state * 0.5], dim=1)
    vectorized = backbone.retrieval_readout(candidates)
    loop = torch.stack(
        [backbone.retrieval_readout(candidates[:, index]) for index in range(3)], dim=1
    )
    assert torch.allclose(vectorized, loop, atol=1e-6)
    assert tuple(inspect.signature(backbone.retrieval_readout).parameters) == ("state",)
