from __future__ import annotations

import inspect
from dataclasses import replace
from types import SimpleNamespace

import torch
from torch import Tensor, nn

from models.iag_srme import Executor, IAGSRME, IAGSRMEConfig
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

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask=None,
        causal_attention_mask=None,
        output_attentions=False,
    ) -> tuple[Tensor]:
        del attention_mask, causal_attention_mask, output_attentions
        hidden_states = hidden_states + self.self_attn(self.layer_norm1(hidden_states))
        hidden_states = hidden_states + self.mlp(self.layer_norm2(hidden_states))
        return (hidden_states,)


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
        # Stand in for the image-conditioned CLS produced by preceding vision blocks.
        cls = self.embeddings.class_embedding[None, None] + patches.mean(dim=1, keepdim=True)
        penultimate = torch.cat([cls, patches], dim=1)
        final = self.encoder.layers[-1](penultimate)[0]
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
    assert tuple(inspect.signature(backbone.retrieval_readout).parameters) == (
        "state",
        "cls_anchor",
    )


@torch.no_grad()
def test_retrieval_from_global_matches_compatibility_wrapper_in_both_modes() -> None:
    torch.manual_seed(9)
    pixels = torch.randn(2, 3, 2, 2)
    for mode in ("learned_qg", "native_cls"):
        backbone = FGCLIPBackbone(
            MockFGCLIP(), text_width=4, global_readout_mode=mode
        ).eval()
        patches, cls_anchor = backbone.initial_state_with_anchor(pixels)
        anchor = cls_anchor if mode == "native_cls" else None
        global_state = backbone.global_readout(patches, anchor)
        expected = backbone.retrieval_from_global(global_state)
        actual = backbone.retrieval_readout(patches, anchor)
        assert torch.equal(actual, expected)

        candidates = torch.stack([patches, patches + 0.1, patches - 0.2], dim=1)
        candidate_global = backbone.global_readout(candidates, anchor)
        candidate_expected = backbone.retrieval_from_global(candidate_global)
        candidate_actual = backbone.retrieval_readout(candidates, anchor)
        assert torch.equal(candidate_actual, candidate_expected)


def _legacy_learned_qg_readout(backbone: FGCLIPBackbone, state: Tensor) -> Tensor:
    patches, leading = backbone._flatten_state(state)
    block = backbone.model.vision_model.encoder.layers[-1]
    batch, tokens, width = patches.shape
    query = backbone.q_G.to(patches.dtype).view(1, 1, width).expand(batch, 1, width)
    q = block.self_attn.q_proj(block.layer_norm1(query)) * block.self_attn.scale
    key_value = block.layer_norm1(patches)
    key = block.self_attn.k_proj(key_value)
    value = block.self_attn.v_proj(key_value)
    heads = block.self_attn.num_heads
    head_dim = block.self_attn.head_dim
    q = q.view(batch, 1, heads, head_dim).transpose(1, 2)
    key = key.view(batch, tokens, heads, head_dim).transpose(1, 2)
    value = value.view(batch, tokens, heads, head_dim).transpose(1, 2)
    weights = torch.softmax(q @ key.transpose(-2, -1), dim=-1)
    attended = (weights @ value).transpose(1, 2).reshape(batch, 1, width)
    query = query + block.self_attn.out_proj(attended)
    query = query + block.mlp(block.layer_norm2(query))
    result = backbone.model.vision_model.post_layernorm(query[:, 0])
    return result.reshape(*leading, width)


def test_learned_qg_is_trainable_initialized_and_regression_compatible() -> None:
    torch.manual_seed(10)
    checkpoint = MockFGCLIP()
    expected_initialization = checkpoint.vision_model.embeddings.class_embedding.detach().clone()
    backbone = FGCLIPBackbone(checkpoint, text_width=4, global_readout_mode="learned_qg").eval()
    state = backbone.initial_state(torch.randn(2, 3, 2, 2))

    assert backbone.q_G.requires_grad
    assert torch.equal(backbone.q_G.detach(), expected_initialization)
    assert torch.allclose(
        backbone.global_readout(state), _legacy_learned_qg_readout(backbone, state)
    )
    assert torch.equal(
        backbone.global_readout(state), backbone.global_readout(state, torch.randn(2, 6))
    )


def test_native_cls_initial_parity_and_dynamic_current_state() -> None:
    torch.manual_seed(12)
    checkpoint = MockFGCLIP()
    backbone = FGCLIPBackbone(checkpoint, text_width=4, global_readout_mode="native_cls").eval()
    pixels = torch.randn(2, 3, 2, 2)
    patches, cls_anchor = backbone.initial_state_with_anchor(pixels)
    official = checkpoint.vision_model(pixels, output_hidden_states=True, return_dict=True)
    expected_global = official.pooler_output
    expected_query = torch.nn.functional.normalize(
        checkpoint.visual_projection(expected_global).float(), dim=-1
    )

    assert cls_anchor.shape == (2, 6)
    assert not torch.equal(cls_anchor[0], cls_anchor[1])
    assert torch.allclose(cls_anchor, official.hidden_states[-2][:, 0])
    assert torch.allclose(patches, official.hidden_states[-2][:, 1:])
    full_global = backbone._native_cls_readout_full(patches, cls_anchor)
    cls_only_global = backbone._native_cls_readout_cls_only(patches, cls_anchor)
    assert torch.allclose(cls_only_global, full_global, atol=1e-6)
    assert torch.allclose(backbone.global_readout(patches, cls_anchor), expected_global)
    assert torch.allclose(backbone.retrieval_readout(patches, cls_anchor), expected_query)

    changed = patches.clone()
    changed[:, 0, 0] += 1.0
    assert not torch.allclose(
        backbone.global_readout(changed, cls_anchor),
        backbone.global_readout(patches, cls_anchor),
    )
    assert not torch.allclose(
        backbone.retrieval_readout(changed, cls_anchor),
        backbone.retrieval_readout(patches, cls_anchor),
    )


def test_native_cls_candidate_vectorization_and_sample_anchor_association() -> None:
    torch.manual_seed(13)
    backbone = FGCLIPBackbone(
        MockFGCLIP(), text_width=4, global_readout_mode="native_cls"
    ).eval()
    patches, cls_anchor = backbone.initial_state_with_anchor(torch.randn(2, 3, 2, 2))
    candidates = torch.stack([patches, patches + 0.1, patches - 0.2], dim=1)
    vectorized = backbone.retrieval_readout(candidates, cls_anchor)
    loop = torch.stack(
        [backbone.retrieval_readout(candidates[:, index], cls_anchor) for index in range(3)],
        dim=1,
    )
    permutation = torch.tensor([2, 0, 1])

    assert vectorized.shape == (2, 3, 5)
    assert torch.allclose(vectorized, loop, atol=1e-6)
    assert torch.allclose(
        backbone.retrieval_readout(candidates[:, permutation], cls_anchor),
        vectorized[:, permutation],
        atol=1e-6,
    )


def test_native_cls_rollout_keeps_anchor_outside_mutable_executor_state() -> None:
    torch.manual_seed(14)
    backbone = FGCLIPBackbone(
        MockFGCLIP(), text_width=4, global_readout_mode="native_cls"
    ).eval()
    model = IAGSRME(
        backbone,
        IAGSRMEConfig(
            width=4,
            num_candidates=3,
            max_steps=2,
            num_heads=2,
            exec_dim=4,
            stop_enabled=False,
            score_dropout=0.0,
        ),
    ).eval()
    patches, cls_anchor = backbone.initial_state_with_anchor(torch.randn(2, 3, 2, 2))
    before = cls_anchor.clone()
    output = model.forward_from_features(
        patches,
        torch.randn(2, 5, 4),
        torch.randn(2, 4),
        torch.ones(2, 5, dtype=torch.bool),
        cls_anchor,
    )

    assert torch.equal(output["cls_anchor"], before)
    assert torch.equal(cls_anchor, before)
    assert tuple(inspect.signature(Executor.forward).parameters) == (
        "self",
        "parent",
        "actions",
        "exec_mask",
        "patch_grid",
    )
    for step in output["steps"]:
        assert torch.equal(step["cls_anchor"], before.index_select(0, step["live_indices"]))
        assert step["candidate_queries"].shape == (2, 3, 5)
        assert torch.allclose(
            step["delta_q"], step["candidate_queries"] - step["current_query"][:, None]
        )

    model.config = replace(model.config, stop_enabled=True)
    with torch.no_grad():
        model.score_net.net[-1].weight.zero_()
        model.score_net.net[-1].bias.fill_(-1.0)
    stopped = model.forward_from_features(
        patches,
        torch.randn(2, 5, 4),
        torch.randn(2, 4),
        torch.ones(2, 5, dtype=torch.bool),
        cls_anchor,
    )
    assert stopped["stopped"].all()
    assert torch.equal(stopped["state"], patches)
    assert torch.equal(stopped["cls_anchor"], before)


def test_rollout_computes_parent_and_candidate_global_once_per_timestep(
    monkeypatch,
) -> None:
    torch.manual_seed(141)
    backbone = FGCLIPBackbone(MockFGCLIP(), text_width=4).eval()
    model = IAGSRME(
        backbone,
        IAGSRMEConfig(
            width=4,
            num_candidates=3,
            max_steps=2,
            num_heads=2,
            exec_dim=4,
            stop_enabled=False,
            score_dropout=0.0,
        ),
    ).eval()
    patches = backbone.initial_state(torch.randn(2, 3, 2, 2))
    original = backbone.global_readout
    calls = 0

    def counted(state: Tensor, cls_anchor: Tensor | None = None) -> Tensor:
        nonlocal calls
        calls += 1
        return original(state, cls_anchor)

    monkeypatch.setattr(backbone, "global_readout", counted)
    output = model.forward_from_features(
        patches,
        torch.randn(2, 5, 4),
        torch.randn(2, 4),
        torch.ones(2, 5, dtype=torch.bool),
    )

    # Parent + vectorized siblings once per step, plus one terminal readout.
    assert calls == 2 * len(output["steps"]) + 1


def test_both_readout_modes_have_the_same_public_shapes() -> None:
    torch.manual_seed(15)
    checkpoint = MockFGCLIP()
    pixels = torch.randn(2, 3, 2, 2)
    native = FGCLIPBackbone(
        checkpoint, text_width=4, global_readout_mode="native_cls"
    ).eval()
    learned = FGCLIPBackbone(
        checkpoint, text_width=4, global_readout_mode="learned_qg"
    ).eval()
    patches, cls_anchor = native.initial_state_with_anchor(pixels)

    assert native.global_readout(patches, cls_anchor).shape == learned.global_readout(patches).shape
    assert native.retrieval_readout(patches, cls_anchor).shape == learned.retrieval_readout(
        patches
    ).shape
