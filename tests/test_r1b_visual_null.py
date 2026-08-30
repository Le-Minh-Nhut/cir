from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
import torch
from torch import nn
from torch.optim import AdamW

from diagnose_iag_srme_checkpoint import _resolve_checkpoint_model_config
from diagnose_iag_srme_checkpoint import (
    NullTargetUtilityAccumulator,
    ValidationDiagnosticAccumulator,
    _checkpoint_replay_guard,
)
from losses.objective import IAGSRMEObjective, ObjectiveConfig
from models.iag_srme import IAGSRMEConfig, IAGSRMECore
from models.iag_srme.editor import SharedTokenEditor
from models.iag_srme.grounding import AnchorGrounder
from training.engine import PrecisionPolicy, save_checkpoint


def _identity_grounder(*, null_logit: float) -> AnchorGrounder:
    grounder = AnchorGrounder(
        width=4,
        grounding_width=4,
        enable_visual_null=True,
        visual_null_initial_logit=null_logit,
    )
    with torch.no_grad():
        grounder.intent_projection.weight.copy_(torch.eye(4))
        grounder.anchor_projection.weight.copy_(torch.eye(4))
        grounder.visual_null_key.zero_()
    return grounder


def test_visual_null_dominates_and_real_mass_vanishes() -> None:
    grounder = _identity_grounder(null_logit=100.0)
    intents = torch.zeros(2, 4, 4)
    anchor = torch.zeros(2, 6, 4)
    grounding = grounder(intents, anchor)

    assert grounding.null_probabilities.gt(0.999).all()
    assert grounding.visual_supports.sum(dim=-1).lt(1e-6).all()
    torch.testing.assert_close(
        grounding.visual_supports.sum(dim=-1),
        grounding.visual_confidence,
        atol=1e-6,
        rtol=1e-6,
    )

    editor = SharedTokenEditor(width=4, lambda_z=0.1)
    delta_z, _ = editor(
        torch.randn(2, 4, 4),
        grounding.conditional_supports,
        anchor,
        anchor,
        execution_confidence=grounding.visual_confidence,
    )
    assert delta_z.norm() < 1e-6


def test_real_visual_token_dominates_null_and_edit_remains_active() -> None:
    grounder = _identity_grounder(null_logit=-100.0)
    intents = torch.tensor([[[10.0, 0.0, 0.0, 0.0]]]).expand(1, 4, 4)
    anchor = torch.tensor(
        [[[10.0, 0.0, 0.0, 0.0], [-10.0, 0.0, 0.0, 0.0]]]
    )
    grounding = grounder(intents, anchor)

    assert grounding.null_probabilities.lt(1e-6).all()
    torch.testing.assert_close(
        grounding.visual_supports.sum(dim=-1),
        torch.ones(1, 4),
        atol=1e-6,
        rtol=1e-6,
    )
    editor = SharedTokenEditor(width=4, lambda_z=0.1)
    delta_z, _ = editor(
        torch.randn(1, 4, 4),
        grounding.conditional_supports,
        anchor,
        anchor,
        execution_confidence=grounding.visual_confidence,
    )
    assert delta_z.norm() > 0


def test_same_spatial_shape_different_confidence_scales_edit_once() -> None:
    torch.manual_seed(811)
    editor = SharedTokenEditor(width=8, lambda_z=0.1).eval()
    contexts = torch.randn(1, 4, 8)
    anchor = torch.randn(1, 5, 8)
    state = anchor + 0.1 * torch.randn_like(anchor)
    spatial = torch.tensor([[[0.7, 0.3, 0.0, 0.0, 0.0]]]).expand(1, 4, 5)
    high = torch.full((1, 4), 0.9)
    low = torch.full((1, 4), 0.1)

    delta_high, _ = editor(
        contexts, spatial, anchor, state, execution_confidence=high
    )
    delta_low, _ = editor(
        contexts, spatial, anchor, state, execution_confidence=low
    )

    torch.testing.assert_close(delta_low, delta_high / 9.0, atol=1e-7, rtol=1e-6)
    assert delta_low.norm() < delta_high.norm()


def test_unit_confidence_preserves_legacy_editor_exactly() -> None:
    torch.manual_seed(817)
    editor = SharedTokenEditor(width=8, lambda_z=0.1).eval()
    contexts = torch.randn(2, 4, 8)
    anchor = torch.randn(2, 5, 8)
    state = anchor + 0.1 * torch.randn_like(anchor)
    spatial = torch.softmax(torch.randn(2, 4, 5), dim=-1)

    legacy_delta, legacy_states = editor(contexts, spatial, anchor, state)
    r1b_delta, r1b_states = editor(
        contexts,
        spatial,
        anchor,
        state,
        execution_confidence=torch.ones(2, 4),
    )

    torch.testing.assert_close(r1b_delta, legacy_delta, atol=0.0, rtol=0.0)
    torch.testing.assert_close(r1b_states, legacy_states, atol=0.0, rtol=0.0)


def test_edit_norm_decreases_monotonically_as_null_probability_increases() -> None:
    torch.manual_seed(821)
    editor = SharedTokenEditor(width=8, lambda_z=0.1).eval()
    contexts = torch.randn(1, 4, 8)
    anchor = torch.randn(1, 5, 8)
    spatial = torch.softmax(torch.randn(1, 4, 5), dim=-1)
    norms = []
    for null_probability in (0.0, 0.25, 0.5, 0.75, 1.0):
        confidence = torch.full((1, 4), 1.0 - null_probability)
        delta_z, _ = editor(
            contexts,
            spatial,
            anchor,
            anchor,
            execution_confidence=confidence,
        )
        norms.append(float(delta_z.detach().norm()))

    assert all(left > right for left, right in zip(norms[:-1], norms[1:], strict=True))
    assert norms[-1] == 0.0


def test_visual_null_parameters_receive_finite_nonzero_gradient() -> None:
    torch.manual_seed(829)
    grounder = AnchorGrounder(width=8, enable_visual_null=True)
    intents = torch.randn(3, 4, 8, requires_grad=True)
    anchor = torch.randn(3, 11, 8, requires_grad=True)
    grounding = grounder(intents, anchor)
    loss = grounding.visual_supports.square().sum()
    loss.backward()

    for parameter in (grounder.visual_null_key, grounder.visual_null_bias):
        assert parameter is not None
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.norm() > 0


def test_r1b_same_parent_and_static_grounding_invariants(synthetic_encoded) -> None:
    core = IAGSRMECore(
        IAGSRMEConfig(
            width=32,
            num_candidates=4,
            max_steps=3,
            num_heads=4,
            retrieval_dim=24,
            query_cap=1000.0,
            selector_gumbel_noise=False,
            enable_visual_null=True,
        )
    ).eval()
    grounding_calls = 0

    def count_call(_module, _inputs, _output) -> None:
        nonlocal grounding_calls
        grounding_calls += 1

    handle = core.grounder.register_forward_hook(count_call)
    output = core(synthetic_encoded)
    handle.remove()

    assert grounding_calls == 1
    assert output.visual_null_probabilities is not None
    assert output.visual_confidence is not None
    for step in output.trace:
        torch.testing.assert_close(
            step.candidate_states,
            step.current_state[:, None] + step.delta_z,
            atol=0.0,
            rtol=0.0,
        )


def test_r1b_hydra_config_carries_forward_r1a_query_cap() -> None:
    config_dir = str(Path(__file__).resolve().parents[1] / "conf")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        config = compose(
            config_name="config",
            overrides=[
                "model=iag_srme_r1b_visual_null",
                "experiment=iag_srme_r1b_visual_null",
            ],
        )

    assert config.model.query_cap == 1000.0
    assert config.model.enable_visual_null is True
    assert config.model.grounding_normalization == "entmax15"


def test_r1b_checkpoint_config_replays_deterministic_core(synthetic_encoded) -> None:
    config = IAGSRMEConfig(
        width=32,
        num_candidates=4,
        max_steps=3,
        num_heads=4,
        retrieval_dim=24,
        query_cap=1000.0,
        selector_gumbel_noise=True,
        enable_visual_null=True,
        visual_null_initial_logit=0.0,
        grounding_normalization="entmax15",
    )
    original = IAGSRMECore(config).eval()
    checkpoint = {
        "metadata": {
            "model_config": asdict(config),
            "architecture_generation": "r1b_visual_null_confidence_gate",
        },
        "model": {f"core.{key}": value for key, value in original.state_dict().items()},
    }
    replay_config, provenance = _resolve_checkpoint_model_config(
        checkpoint, checkpoint["model"], retrieval_dim=24
    )
    replay = IAGSRMECore(replay_config).eval()
    replay.load_state_dict(original.state_dict(), strict=True)

    before = original(synthetic_encoded)
    after = replay(synthetic_encoded)
    assert replay.config.query_cap == 1000.0
    assert replay.config.enable_visual_null is True
    assert provenance["fully_self_describing"] is True
    assert provenance["architecture_generation"] == "r1b_visual_null_confidence_gate"
    for name in ("final_query", "final_state", "supports", "visual_null_probabilities"):
        torch.testing.assert_close(
            getattr(after, name), getattr(before, name), atol=0.0, rtol=0.0
        )


def test_checkpoint_writer_serializes_all_replay_critical_r1b_config(tmp_path) -> None:
    config = IAGSRMEConfig(
        width=8,
        num_candidates=4,
        max_steps=3,
        num_heads=2,
        retrieval_dim=8,
        lambda_z=0.1,
        query_cap=1000.0,
        selector_temperature=1.0,
        selector_gumbel_noise=True,
        enable_visual_null=True,
        visual_null_initial_logit=0.0,
        grounding_normalization="entmax15",
    )

    class CheckpointModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.core = IAGSRMECore(config)
            self.backbone = SimpleNamespace(
                checkpoint="qihoo360/fg-clip-base", revision="test-revision"
            )

    model = CheckpointModel()
    objective = IAGSRMEObjective(ObjectiveConfig(), width=8)
    optimizer = AdamW([*model.parameters(), *objective.parameters()], lr=1e-5)
    path = tmp_path / "r1b.pt"
    save_checkpoint(
        path,
        model,
        objective,
        optimizer,
        epoch=1,
        metric=12.5,
        precision=PrecisionPolicy("fp16", True, torch.float16, False),
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    saved = checkpoint["metadata"]["model_config"]

    assert saved == asdict(config)
    assert saved["query_cap"] == 1000.0
    assert saved["enable_visual_null"] is True
    assert saved["grounding_normalization"] == "entmax15"
    assert checkpoint["metadata"]["architecture_generation"] == (
        "r1b_visual_null_confidence_gate"
    )


def test_r1b_diagnostics_separate_visual_mass_from_conditional_shape(
    synthetic_encoded,
) -> None:
    core = IAGSRMECore(
        IAGSRMEConfig(
            width=32,
            num_candidates=4,
            max_steps=3,
            num_heads=4,
            retrieval_dim=24,
            query_cap=1000.0,
            selector_gumbel_noise=False,
            enable_visual_null=True,
        )
    ).eval()
    output = core(synthetic_encoded)
    accumulator = ValidationDiagnosticAccumulator()
    accumulator.update(output)
    grounding = accumulator.grounding_summary()
    null = accumulator.visual_null_summary()

    assert null["enabled"] is True
    assert null["static_by_architecture"] is True
    assert null["null_probability"]["count"] == 12
    assert len(null["per_candidate_null_probability"]) == 4
    assert len(null["null_vs_effect_magnitude_bins_by_timestep"]) == 3
    assert grounding["conditional_shape_valid_candidate_fraction"] > 0
    torch.testing.assert_close(
        output.supports.sum(dim=-1),
        1.0 - output.visual_null_probabilities,
        atol=1e-5,
        rtol=1e-5,
    )


def test_null_dominant_diagnostics_mark_conditional_shape_undefined(
    synthetic_encoded,
) -> None:
    core = IAGSRMECore(
        IAGSRMEConfig(
            width=32,
            num_candidates=4,
            max_steps=3,
            num_heads=4,
            retrieval_dim=24,
            query_cap=1000.0,
            selector_gumbel_noise=False,
            enable_visual_null=True,
            visual_null_initial_logit=100.0,
        )
    ).eval()
    output = core(synthetic_encoded)
    accumulator = ValidationDiagnosticAccumulator()
    accumulator.update(output)
    grounding = accumulator.grounding_summary()
    null = accumulator.visual_null_summary()

    assert grounding["conditional_shape_valid_candidate_fraction"] == 0.0
    assert grounding["conditional_support_entropy"] is None
    assert all(
        value is None for row in grounding["pairwise_support_cosine_matrix"] for value in row
    )
    assert null["shortcut_observation_flags"]["null_globally_dominant"] is True


def test_offline_null_utility_diagnostic_cannot_change_forward(
    synthetic_encoded,
) -> None:
    core = IAGSRMECore(
        IAGSRMEConfig(
            width=32,
            num_candidates=4,
            max_steps=3,
            num_heads=4,
            retrieval_dim=24,
            query_cap=1000.0,
            selector_gumbel_noise=False,
            enable_visual_null=True,
        )
    ).eval()
    output = core(synthetic_encoded)
    final_before = output.final_query.detach().clone()
    accumulator = NullTargetUtilityAccumulator()
    accumulator.update(output, torch.randn_like(output.final_query))
    summary = accumulator.summary()

    torch.testing.assert_close(output.final_query, final_before, atol=0.0, rtol=0.0)
    assert summary["candidate_observation_count"] > 0
    assert len(summary["utility_by_null_bin"]) == 5
    assert "after target-free rollout" in summary["target_firewall"]


def test_r1b_replay_guard_requires_exact_cap_null_and_metric() -> None:
    provenance = {
        "architecture_generation": "r1b_visual_null_confidence_gate",
        "fully_self_describing": True,
        "resolved_diagnostic_config": {
            "query_cap": 1000.0,
            "enable_visual_null": True,
        },
    }
    guard = _checkpoint_replay_guard(
        {"metric": 38.75}, provenance, replayed_mean_recall=38.75001
    )

    assert guard["trusted_r1b_replay"] is True
    assert guard["checks"] == {
        "query_cap_is_1000": True,
        "visual_null_enabled": True,
        "checkpoint_fully_self_describing": True,
    }


def test_legacy_checkpoint_replay_guard_remains_non_applicable() -> None:
    guard = _checkpoint_replay_guard(
        {"metric": 10.0},
        {"architecture_generation": "legacy_r0_or_r1a"},
        replayed_mean_recall=9.0,
    )

    assert guard["applicable"] is False
    assert guard["trusted_r1b_replay"] is None
