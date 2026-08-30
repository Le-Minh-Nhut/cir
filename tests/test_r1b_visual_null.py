from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
import pytest
import torch
from torch import nn
from torch.optim import AdamW

from diagnose_iag_srme_checkpoint import (
    NullTargetUtilityAccumulator,
    ValidationDiagnosticAccumulator,
    _checkpoint_replay_guard,
    _resolve_checkpoint_model_config,
)
from losses.objective import IAGSRMEObjective, ObjectiveConfig
from models.iag_srme import IAGSRMEConfig, IAGSRMECore
from models.iag_srme.applicability import DynamicApplicabilityGate
from models.iag_srme.editor import SharedTokenEditor
from models.iag_srme.grounding import AnchorGrounder
from training.engine import PrecisionPolicy, save_checkpoint


def _r1b_config(**overrides) -> IAGSRMEConfig:
    values = {
        "width": 32,
        "num_candidates": 4,
        "max_steps": 3,
        "num_heads": 4,
        "retrieval_dim": 24,
        "query_cap": 1000.0,
        "selector_gumbel_noise": False,
        "enable_dynamic_applicability": True,
        "initial_applicability": 0.98,
    }
    values.update(overrides)
    return IAGSRMEConfig(**values)


def test_corrected_r1b_spatial_grounding_exactly_matches_legacy() -> None:
    torch.manual_seed(901)
    legacy = AnchorGrounder(width=8).eval()
    corrected = IAGSRMECore(
        IAGSRMEConfig(
            width=8,
            num_heads=2,
            retrieval_dim=8,
            max_steps=1,
            enable_dynamic_applicability=True,
        )
    ).grounder.eval()
    corrected.load_state_dict(legacy.state_dict(), strict=True)
    intents = torch.randn(2, 4, 8)
    anchor = torch.randn(2, 13, 8)
    torch.testing.assert_close(
        corrected(intents, anchor), legacy(intents, anchor), atol=0.0, rtol=0.0
    )


def test_corrected_r1b_spatial_support_has_unit_mass_and_sparse_zeros() -> None:
    grounder = AnchorGrounder(width=4, grounding_width=4).eval()
    with torch.no_grad():
        grounder.intent_projection.weight.copy_(torch.eye(4))
        grounder.anchor_projection.weight.copy_(torch.eye(4))
    intents = torch.tensor([[[10.0, 0.0, 0.0, 0.0]]]).expand(1, 4, 4)
    anchor = torch.tensor(
        [[[10.0, 0.0, 0.0, 0.0], [-10.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]
    )
    support = grounder(intents, anchor)
    torch.testing.assert_close(support.sum(dim=-1), torch.ones(1, 4))
    assert (support == 0).any()
    assert support.shape[-1] == anchor.shape[1]


def test_initial_applicability_is_exactly_configured_near_r1a() -> None:
    gate = DynamicApplicabilityGate(width=8, initial_applicability=0.98).eval()
    logits, confidence, null = gate(torch.randn(3, 4, 8))
    torch.testing.assert_close(confidence, torch.full_like(confidence, 0.98))
    torch.testing.assert_close(null, torch.full_like(null, 0.02))
    torch.testing.assert_close(
        logits, torch.full_like(logits, math.log(0.98 / 0.02))
    )
    assert torch.count_nonzero(gate.projection.weight) == 0


@pytest.mark.parametrize("initial", [0.0, 1.0, -0.1, 1.1])
def test_initial_applicability_must_be_strictly_between_zero_and_one(
    initial: float,
) -> None:
    with pytest.raises(ValueError, match="strictly between"):
        DynamicApplicabilityGate(width=8, initial_applicability=initial)


def test_initial_r1b_delta_is_point_98_times_r1a() -> None:
    torch.manual_seed(907)
    editor = SharedTokenEditor(width=8, lambda_z=0.1).eval()
    contexts = torch.randn(2, 4, 8)
    anchor = torch.randn(2, 7, 8)
    state = anchor + 0.1 * torch.randn_like(anchor)
    support = torch.softmax(torch.randn(2, 4, 7), dim=-1)
    legacy, _ = editor(contexts, support, anchor, state)
    corrected, _ = editor(
        contexts,
        support,
        anchor,
        state,
        execution_confidence=torch.full((2, 4), 0.98),
    )
    torch.testing.assert_close(corrected, 0.98 * legacy, atol=1e-7, rtol=1e-6)


def test_applicability_monotonically_scales_edit_and_zero_is_exact() -> None:
    torch.manual_seed(911)
    editor = SharedTokenEditor(width=8, lambda_z=0.1).eval()
    contexts = torch.randn(1, 4, 8)
    anchor = torch.randn(1, 7, 8)
    support = torch.softmax(torch.randn(1, 4, 7), dim=-1)
    norms = []
    for confidence_value in (1.0, 0.75, 0.5, 0.25, 0.0):
        delta, _ = editor(
            contexts,
            support,
            anchor,
            anchor,
            execution_confidence=torch.full((1, 4), confidence_value),
        )
        norms.append(float(delta.detach().norm()))
        if confidence_value == 0.0:
            assert torch.count_nonzero(delta) == 0
    assert all(left > right for left, right in zip(norms[:-1], norms[1:], strict=True))


def test_full_confidence_exactly_preserves_legacy_editor() -> None:
    torch.manual_seed(919)
    editor = SharedTokenEditor(width=8, lambda_z=0.1).eval()
    contexts = torch.randn(2, 4, 8)
    anchor = torch.randn(2, 7, 8)
    state = anchor + 0.1 * torch.randn_like(anchor)
    support = torch.softmax(torch.randn(2, 4, 7), dim=-1)
    legacy_delta, legacy_states = editor(contexts, support, anchor, state)
    gated_delta, gated_states = editor(
        contexts,
        support,
        anchor,
        state,
        execution_confidence=torch.ones(2, 4),
    )
    torch.testing.assert_close(gated_delta, legacy_delta, atol=0.0, rtol=0.0)
    torch.testing.assert_close(gated_states, legacy_states, atol=0.0, rtol=0.0)


def test_applicability_head_has_finite_nonzero_gradient_even_at_low_null() -> None:
    torch.manual_seed(929)
    gate = DynamicApplicabilityGate(width=8, initial_applicability=0.999)
    with torch.no_grad():
        gate.projection.weight.normal_(std=0.01)
    contexts = torch.randn(3, 4, 8, requires_grad=True)
    _, confidence, _ = gate(contexts)
    weights = torch.linspace(-1.0, 1.0, confidence.numel()).reshape_as(confidence)
    (confidence * weights).sum().backward()
    for parameter in (gate.projection.weight, gate.projection.bias):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.norm() > 0


def test_dynamic_applicability_changes_with_state_but_where_is_static(
    synthetic_encoded,
) -> None:
    core = IAGSRMECore(_r1b_config()).eval()
    with torch.no_grad():
        core.applicability_head.projection.weight.normal_(std=0.1)
        core.scorer.score_head[-1].weight.zero_()
        core.scorer.score_head[-1].bias.fill_(1.0)
    output = core(synthetic_encoded, control="repeat_candidate_1")
    torch.testing.assert_close(output.supports.sum(dim=-1), torch.ones(3, 4))
    assert not torch.equal(output.trace[0].contexts, output.trace[1].contexts)
    assert not torch.equal(
        output.trace[0].applicability_logits, output.trace[1].applicability_logits
    )
    torch.testing.assert_close(output.supports, output.conditional_supports)


def test_grounder_once_applicability_each_timestep_and_same_parent(
    synthetic_encoded,
) -> None:
    core = IAGSRMECore(_r1b_config()).eval()
    calls = {"grounder": 0, "applicability": 0}

    def count(name):
        def hook(_module, _inputs, _output) -> None:
            calls[name] += 1

        return hook

    handles = [
        core.grounder.register_forward_hook(count("grounder")),
        core.applicability_head.register_forward_hook(count("applicability")),
    ]
    output = core(synthetic_encoded)
    for handle in handles:
        handle.remove()
    assert calls == {"grounder": 1, "applicability": 3}
    assert output.visual_null_probabilities.shape == (3, 3, 4)
    for step in output.trace:
        assert step.applicability_logits.shape == (3, 4)
        assert step.visual_confidence.shape == (3, 4)
        assert step.visual_null_probability.shape == (3, 4)
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
    assert config.model.enable_visual_null is False
    assert config.model.enable_dynamic_applicability is True
    assert config.model.initial_applicability == 0.98
    assert config.model.grounding_normalization == "entmax15"


def test_corrected_r1b_checkpoint_replays_deterministic_core(
    synthetic_encoded,
) -> None:
    config = _r1b_config(selector_gumbel_noise=True)
    original = IAGSRMECore(config).eval()
    checkpoint = {
        "metadata": {
            "model_config": asdict(config),
            "architecture_generation": "r1b_dynamic_applicability_gate_v2",
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
    assert replay.config.enable_dynamic_applicability is True
    assert replay.config.initial_applicability == 0.98
    assert provenance["architecture_generation"] == "r1b_dynamic_applicability_gate_v2"
    for name in ("final_query", "final_state", "supports", "visual_null_probabilities"):
        torch.testing.assert_close(
            getattr(after, name), getattr(before, name), atol=0.0, rtol=0.0
        )


def test_checkpoint_writer_serializes_corrected_generation(tmp_path) -> None:
    config = _r1b_config(width=8, num_heads=2, retrieval_dim=8)

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
    path = tmp_path / "r1b-v2.pt"
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
    assert saved["enable_dynamic_applicability"] is True
    assert saved["initial_applicability"] == 0.98
    assert checkpoint["metadata"]["architecture_generation"] == (
        "r1b_dynamic_applicability_gate_v2"
    )


def test_entmax_null_v1_checkpoint_is_explicitly_rejected() -> None:
    checkpoint = {
        "metadata": {"model_config": {}},
        "model": {
            "core.intent_encoder.query_bank": torch.randn(4, 256),
            "core.grounder.visual_null_key": torch.randn(256),
        },
    }
    with pytest.raises(ValueError, match="superseded r1b_visual_null_entmax_v1"):
        _resolve_checkpoint_model_config(checkpoint, checkpoint["model"], retrieval_dim=512)


def test_dynamic_null_diagnostics_are_timestep_specific(synthetic_encoded) -> None:
    core = IAGSRMECore(_r1b_config()).eval()
    with torch.no_grad():
        core.applicability_head.projection.weight.normal_(std=0.1)
        core.scorer.score_head[-1].weight.zero_()
        core.scorer.score_head[-1].bias.fill_(1.0)
    output = core(synthetic_encoded, control="repeat_candidate_1")
    accumulator = ValidationDiagnosticAccumulator()
    accumulator.update(output)
    grounding = accumulator.grounding_summary()
    null = accumulator.visual_null_summary()
    assert null["architecture_generation"] == "r1b_dynamic_applicability_gate_v2"
    assert null["static_by_architecture"] is False
    assert len(null["null_probability_by_timestep"]) == 3
    assert len(null["null_probability_by_candidate_and_timestep"]) == 3
    assert len(null["temporal_applicability"]["confidence_change_by_transition"]) == 2
    assert grounding["spatial_support_mass"] == pytest.approx(1.0, abs=1e-6)


def test_offline_null_utility_diagnostic_cannot_change_forward(
    synthetic_encoded,
) -> None:
    core = IAGSRMECore(_r1b_config()).eval()
    output = core(synthetic_encoded)
    final_before = output.final_query.detach().clone()
    accumulator = NullTargetUtilityAccumulator()
    accumulator.update(output, torch.randn_like(output.final_query))
    summary = accumulator.summary()
    torch.testing.assert_close(output.final_query, final_before, atol=0.0, rtol=0.0)
    assert summary["candidate_observation_count"] > 0
    assert len(summary["utility_by_null_bin"]) == 5
    assert "after target-free rollout" in summary["target_firewall"]


def test_corrected_r1b_replay_guard_requires_exact_config_and_metric() -> None:
    provenance = {
        "architecture_generation": "r1b_dynamic_applicability_gate_v2",
        "fully_self_describing": True,
        "resolved_diagnostic_config": {
            "query_cap": 1000.0,
            "enable_dynamic_applicability": True,
            "initial_applicability": 0.98,
        },
    }
    guard = _checkpoint_replay_guard(
        {"metric": 38.75}, provenance, replayed_mean_recall=38.75001
    )
    assert guard["trusted_r1b_replay"] is True
    assert all(guard["checks"].values())


def test_legacy_checkpoint_replay_guard_remains_non_applicable() -> None:
    guard = _checkpoint_replay_guard(
        {"metric": 10.0},
        {"architecture_generation": "legacy_r0_or_r1a"},
        replayed_mean_recall=9.0,
    )
    assert guard["applicable"] is False
    assert guard["trusted_r1b_replay"] is None
