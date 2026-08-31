from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
import pytest
import torch
from torch import nn
from torch.optim import AdamW

from diagnose_iag_srme_checkpoint import (
    TemporalGroundingAccumulator,
    _checkpoint_replay_guard,
    _resolve_checkpoint_model_config,
)
from losses.objective import IAGSRMEObjective, ObjectiveConfig
from models.iag_srme import IAGSRMEConfig, IAGSRMECore
from models.iag_srme.grounding import AnchorGrounder
from training.engine import PrecisionPolicy, save_checkpoint


def _r1c1_config(**overrides) -> IAGSRMEConfig:
    values = {
        "width": 32,
        "num_candidates": 4,
        "max_steps": 3,
        "num_heads": 4,
        "retrieval_dim": 24,
        "lambda_z": 0.1,
        "query_cap": 1000.0,
        "selector_gumbel_noise": False,
        "enable_dynamic_applicability": False,
        "enable_dynamic_regrounding": True,
        "grounding_normalization": "entmax15",
    }
    values.update(overrides)
    return IAGSRMEConfig(**values)


def _force_non_stop(core: IAGSRMECore) -> None:
    with torch.no_grad():
        core.scorer.score_head[-1].weight.zero_()
        core.scorer.score_head[-1].bias.fill_(1.0)


def test_intent_once_and_grounder_once_per_timestep(synthetic_encoded) -> None:
    core = IAGSRMECore(_r1c1_config()).eval()
    _force_non_stop(core)
    calls = {"intent": 0, "grounder": 0}

    def count(name):
        def hook(_module, _inputs):
            calls[name] += 1

        return hook

    handles = [
        core.intent_encoder.register_forward_pre_hook(count("intent")),
        core.grounder.register_forward_pre_hook(count("grounder")),
    ]
    try:
        core(synthetic_encoded)
    finally:
        for handle in handles:
            handle.remove()
    assert calls == {"intent": 1, "grounder": 3}


def test_t0_grounding_exactly_matches_static_r1a(synthetic_encoded) -> None:
    dynamic = IAGSRMECore(_r1c1_config()).eval()
    static = IAGSRMECore(
        _r1c1_config(enable_dynamic_regrounding=False)
    ).eval()
    static.load_state_dict(dynamic.state_dict(), strict=True)
    _force_non_stop(dynamic)
    _force_non_stop(static)
    dynamic_output = dynamic(synthetic_encoded)
    static_output = static(synthetic_encoded)
    torch.testing.assert_close(
        dynamic_output.trace[0].spatial_supports,
        static_output.trace[0].spatial_supports,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        dynamic_output.trace[0].delta_z,
        static_output.trace[0].delta_z,
        atol=0.0,
        rtol=0.0,
    )


def test_controlled_current_state_perturbation_changes_where() -> None:
    grounder = AnchorGrounder(width=2)
    with torch.no_grad():
        grounder.intent_projection.weight.copy_(torch.eye(2))
        grounder.anchor_projection.weight.copy_(torch.eye(2))
    intents = torch.tensor([[[1.0, 0.0]]]).expand(1, 4, 2)
    initial = torch.tensor([[[1.0, 0.0], [0.0, 0.0], [-1.0, 0.0]]])
    changed = initial.clone()
    changed[:, 1, 0] = 3.0
    before = grounder(intents, initial)
    after = grounder(intents, changed)
    assert not torch.allclose(before, after)
    assert after[..., 1].gt(before[..., 1]).all()


def test_anchor_same_parent_trace_and_no_r1b_gate(synthetic_encoded) -> None:
    core = IAGSRMECore(_r1c1_config()).eval()
    _force_non_stop(core)
    anchor_before = synthetic_encoded.anchor.clone()
    output = core(synthetic_encoded)
    assert torch.equal(output.anchor, anchor_before)
    assert torch.equal(synthetic_encoded.anchor, anchor_before)
    assert core.applicability_head is None
    assert output.visual_confidence is None
    assert output.visual_null_probabilities is None
    for step in output.trace:
        assert step.visual_confidence is None
        assert step.spatial_supports is not None
        assert torch.equal(step.delta_z, step.ungated_delta_z)
        assert torch.equal(
            step.candidate_states,
            step.current_state[:, None] + step.delta_z,
        )


def test_temporal_support_contract_and_target_firewall(synthetic_encoded) -> None:
    core = IAGSRMECore(_r1c1_config()).eval()
    _force_non_stop(core)
    first = core(synthetic_encoded)
    arbitrary_targets = torch.randn(3, 24)[torch.tensor([2, 0, 1])]
    second = core(synthetic_encoded)
    assert arbitrary_targets.shape == first.final_query.shape
    assert first.temporal_supports.shape == (3, 3, 4, 13)
    assert torch.equal(first.supports, first.initial_supports)
    assert torch.equal(first.supports, first.temporal_supports[:, 0])
    torch.testing.assert_close(
        first.temporal_supports.sum(dim=-1),
        torch.ones(3, 3, 4),
        atol=1e-5,
        rtol=1e-5,
    )
    for name in ("intents", "temporal_supports", "final_state", "final_query"):
        assert torch.equal(getattr(first, name), getattr(second, name))


def test_dynamic_where_controls_reground_actual_parent_state(synthetic_encoded) -> None:
    core = IAGSRMECore(_r1c1_config()).eval()
    captured_states: list[torch.Tensor] = []

    def capture(_module, inputs):
        captured_states.append(inputs[1].detach().clone())

    handle = core.grounder.register_forward_pre_hook(capture)
    try:
        repeated = core(synthetic_encoded, control="repeat_candidate_2")
    finally:
        handle.remove()
    assert len(captured_states) == 3
    for timestep, state in enumerate(captured_states):
        assert torch.equal(state, repeated.trace[timestep].current_state)
    assert all(step.selected_index.eq(1).all() for step in repeated.trace)
    assert any(
        not torch.equal(repeated.temporal_supports[:, 0], repeated.temporal_supports[:, timestep])
        for timestep in (1, 2)
    )


def test_temporal_grounding_diagnostics_preserve_decision_conditioning(
    synthetic_encoded,
) -> None:
    core = IAGSRMECore(_r1c1_config()).eval()
    repeated = core(synthetic_encoded, control="repeat_candidate_2")
    accumulator = TemporalGroundingAccumulator()
    accumulator.update(repeated)
    summary = accumulator.summary()
    assert summary["enable_dynamic_regrounding"] is True
    assert len(summary["per_timestep"]) == 3
    assert len(summary["per_transition"]) == 2
    first = summary["per_transition"][0]
    assert first["support_l1_change"]["mean"] > 0
    assert first["same_candidate_temporal_cosine"]["count"] == 12
    conditioned = first["conditioned_on_previous_decision"]
    assert conditioned["same_candidate_executed"]["l1_change"]["count"] == 3
    assert conditioned["other_candidate_executed"]["l1_change"]["count"] == 9
    assert first["candidate_displacement_cosine_matrix"] is not None

    stopped = core(synthetic_encoded, control="zero_edit")
    stop_accumulator = TemporalGroundingAccumulator()
    stop_accumulator.update(stopped)
    stop_summary = stop_accumulator.summary()["per_transition"][0]
    stop_metrics = stop_summary["conditioned_on_previous_decision"]["stop"]
    assert stop_metrics["l1_change"]["count"] == 12
    assert stop_metrics["l1_change"]["maximum"] == 0.0


@pytest.mark.parametrize("control", ["clone_candidate_1", "mean_candidate"])
def test_clone_and_mean_controls_use_current_timestep_supports(
    synthetic_encoded, control: str
) -> None:
    core = IAGSRMECore(_r1c1_config()).eval()
    output = core(synthetic_encoded, control=control)
    for step in output.trace:
        expected = step.spatial_supports[:, :1].expand_as(step.spatial_supports)
        assert torch.equal(step.spatial_supports, expected)


def test_entmax_remains_fp32_island_under_autocast(synthetic_encoded) -> None:
    core = IAGSRMECore(_r1c1_config()).eval()
    encoded = SimpleNamespace(
        **{
            name: (
                getattr(synthetic_encoded, name).bfloat16()
                if getattr(synthetic_encoded, name).is_floating_point()
                else getattr(synthetic_encoded, name)
            )
            for name in (
                "anchor",
                "reference_global",
                "text_tokens",
                "text_global",
                "text_semantic_global",
                "text_content_mask",
            )
        }
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = core(encoded)
    assert output.temporal_supports.dtype == torch.bfloat16
    torch.testing.assert_close(
        output.temporal_supports.float().sum(dim=-1),
        torch.ones(3, 3, 4),
        atol=1e-3,
        rtol=1e-3,
    )


def test_late_dynamic_support_retains_grounder_gradient(synthetic_encoded) -> None:
    core = IAGSRMECore(_r1c1_config()).train()
    _force_non_stop(core)
    output = core(synthetic_encoded, control="repeat_candidate_1")
    token_weights = torch.linspace(
        -1.0,
        1.0,
        output.trace[-1].spatial_supports.shape[-1],
    )
    loss = (output.trace[-1].spatial_supports * token_weights).sum()
    loss.backward()
    for parameter in (
        core.grounder.intent_projection.weight,
        core.grounder.anchor_projection.weight,
        core.intent_encoder.query_bank,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_r1c1_config_and_checkpoint_replay(synthetic_encoded, tmp_path) -> None:
    config_dir = str(Path(__file__).resolve().parents[1] / "conf")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        hydra_config = compose(
            config_name="config",
            overrides=[
                "model=iag_srme_r1c1_dynamic_reground",
                "experiment=iag_srme_r1c1_dynamic_reground",
            ],
        )
    assert hydra_config.model.query_cap == 1000.0
    assert hydra_config.model.enable_dynamic_regrounding is True
    assert hydra_config.model.enable_dynamic_applicability is False

    config = _r1c1_config(width=32, retrieval_dim=24)
    original = IAGSRMECore(config).eval()
    checkpoint = {
        "metadata": {
            "model_config": asdict(config),
            "architecture_generation": "r1c1_dynamic_current_state_reground_v1",
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
    assert replay_config.enable_dynamic_regrounding is True
    assert replay_config.enable_dynamic_applicability is False
    assert provenance["architecture_generation"] == (
        "r1c1_dynamic_current_state_reground_v1"
    )
    for name in ("final_query", "final_state", "temporal_supports"):
        torch.testing.assert_close(
            getattr(before, name), getattr(after, name), atol=0.0, rtol=0.0
        )

    class CheckpointModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.core = IAGSRMECore(config)
            self.backbone = SimpleNamespace(checkpoint="fgclip", revision="revision")

    model = CheckpointModel()
    objective = IAGSRMEObjective(ObjectiveConfig(), width=32)
    optimizer = AdamW([*model.parameters(), *objective.parameters()], lr=1e-5)
    path = tmp_path / "r1c1.pt"
    save_checkpoint(
        path,
        model,
        objective,
        optimizer,
        epoch=1,
        metric=12.0,
        precision=PrecisionPolicy("fp16", True, torch.float16, False),
    )
    saved = torch.load(path, map_location="cpu", weights_only=True)
    assert saved["metadata"]["model_config"]["enable_dynamic_regrounding"] is True
    assert saved["metadata"]["architecture_generation"] == (
        "r1c1_dynamic_current_state_reground_v1"
    )


def test_r1c1_rejects_applicability_stack() -> None:
    with pytest.raises(ValueError, match="cannot enable R1b applicability"):
        IAGSRMECore(
            _r1c1_config(enable_dynamic_applicability=True)
        )


def test_r1c1_replay_guard_requires_exact_causal_config_and_metric() -> None:
    provenance = {
        "architecture_generation": "r1c1_dynamic_current_state_reground_v1",
        "fully_self_describing": True,
        "resolved_diagnostic_config": {
            "query_cap": 1000.0,
            "enable_dynamic_regrounding": True,
            "enable_dynamic_applicability": False,
            "grounding_normalization": "entmax15",
        },
    }
    guard = _checkpoint_replay_guard(
        {"metric": 38.0}, provenance, replayed_mean_recall=38.00001
    )
    assert guard["trusted_r1c1_replay"] is True
    assert guard["trusted_r1b_replay"] is False
    broken = {
        **provenance,
        "resolved_diagnostic_config": {
            **provenance["resolved_diagnostic_config"],
            "enable_dynamic_applicability": True,
        },
    }
    with pytest.raises(ValueError, match="dynamic_applicability_disabled"):
        _checkpoint_replay_guard(
            {"metric": 38.0}, broken, replayed_mean_recall=38.0
        )
