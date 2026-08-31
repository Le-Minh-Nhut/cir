from __future__ import annotations

from dataclasses import asdict
import inspect
from pathlib import Path

from hydra import compose, initialize_config_dir
import pytest
import torch
from torch.optim import SGD

from diagnose_iag_srme_checkpoint import (
    SelectedPathMarginalAccumulator,
    TemporalGroundingAccumulator,
    TemporalIntentAccumulator,
    _checkpoint_replay_guard,
    _resolve_checkpoint_model_config,
)
from canary_train_iag_srme import _reproposal_audit_groups
from models.iag_srme import IAGSRMEConfig, IAGSRMECore
from models.iag_srme.reproposal import DynamicIntentReproposal


def _config(**overrides) -> IAGSRMEConfig:
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
        "enable_dynamic_reproposal": True,
        "grounding_normalization": "entmax15",
    }
    values.update(overrides)
    return IAGSRMEConfig(**values)


def _force_non_stop(core: IAGSRMECore) -> None:
    with torch.no_grad():
        core.scorer.score_head[-1].weight.zero_()
        core.scorer.score_head[-1].bias.fill_(1.0)


def _activate_reproposal(core: IAGSRMECore) -> None:
    assert core.reproposal is not None
    with torch.no_grad():
        core.reproposal.output_projection.weight.copy_(torch.eye(core.config.width))
        core.reproposal.output_projection.bias.zero_()


def test_r1c2_config_rejects_scientific_stacking() -> None:
    with pytest.raises(ValueError, match="requires dynamic regrounding"):
        IAGSRMECore(_config(enable_dynamic_regrounding=False))
    with pytest.raises(ValueError, match="cannot enable R1b applicability"):
        IAGSRMECore(_config(enable_dynamic_applicability=True))


def test_exact_t0_parity_with_r1c1(synthetic_encoded) -> None:
    r1c1 = IAGSRMECore(
        _config(enable_dynamic_reproposal=False)
    ).eval()
    r1c2 = IAGSRMECore(_config()).eval()
    missing, unexpected = r1c2.load_state_dict(r1c1.state_dict(), strict=False)
    assert missing and all(key.startswith("reproposal.") for key in missing)
    assert not unexpected
    _force_non_stop(r1c1)
    _force_non_stop(r1c2)

    baseline = r1c1(synthetic_encoded)
    dynamic = r1c2(synthetic_encoded)
    assert torch.equal(dynamic.initial_intents, baseline.intents)
    assert torch.equal(dynamic.temporal_intents[:, 0], baseline.intents)
    for name in (
        "raw_spatial_supports",
        "original_evidence",
        "current_evidence",
        "accumulated_local_change",
        "delta_z",
        "candidate_queries",
    ):
        assert torch.equal(
            getattr(dynamic.trace[0], name), getattr(baseline.trace[0], name)
        )


def test_base_intent_once_reproposal_twice_and_grounder_three_times(
    synthetic_encoded,
) -> None:
    core = IAGSRMECore(_config()).eval()
    calls = {"intent": 0, "reproposal": 0, "grounder": 0, "applicability": 0}

    def count(name):
        def hook(_module, _inputs) -> None:
            calls[name] += 1

        return hook

    handles = [
        core.intent_encoder.register_forward_pre_hook(count("intent")),
        core.reproposal.register_forward_pre_hook(count("reproposal")),
        core.grounder.register_forward_pre_hook(count("grounder")),
    ]
    try:
        core(synthetic_encoded)
    finally:
        for handle in handles:
            handle.remove()
    assert core.applicability_head is None
    assert calls == {"intent": 1, "reproposal": 2, "grounder": 3, "applicability": 0}


def test_zero_initialized_reproposal_is_exact_r1c1_identity(
    synthetic_encoded,
) -> None:
    core = IAGSRMECore(_config()).eval()
    _force_non_stop(core)
    output = core(synthetic_encoded)
    assert core.reproposal is not None
    assert torch.count_nonzero(core.reproposal.output_projection.weight) == 0
    assert torch.count_nonzero(core.reproposal.output_projection.bias) == 0
    expected = output.initial_intents[:, None].expand_as(output.temporal_intents)
    assert torch.equal(output.temporal_intents, expected)
    for step in output.trace:
        assert torch.count_nonzero(step.intent_residual) == 0


def test_reproposal_is_state_conditioned_and_rereads_token_text() -> None:
    module = DynamicIntentReproposal(width=8, num_heads=2).eval()
    with torch.no_grad():
        module.output_projection.weight.copy_(torch.eye(8))
    base = torch.randn(2, 4, 8)
    anchor = torch.randn(2, 6, 8)
    changed = anchor.clone()
    changed[:, 2, :] += torch.linspace(-1.0, 1.0, 8)
    text = torch.randn(2, 5, 8)
    mask = torch.ones(2, 5, dtype=torch.bool)
    first, _ = module(base, anchor, anchor, text, mask)
    second, _ = module(base, changed, anchor, text, mask)
    assert not torch.allclose(first, second)

    changed_text = text.clone()
    changed_text[:, 1] += torch.linspace(1.0, -1.0, 8)
    third, _ = module(base, changed, anchor, changed_text, mask)
    assert not torch.allclose(second, third)
    assert "target" not in inspect.signature(module.forward).parameters


def test_dynamic_what_changes_where_for_fixed_state(synthetic_encoded) -> None:
    core = IAGSRMECore(_config()).eval()
    _activate_reproposal(core)
    base = core.intent_encoder(
        synthetic_encoded.text_tokens, synthetic_encoded.text_content_mask
    )
    changed_intent, _ = core.reproposal(
        base,
        synthetic_encoded.anchor + 0.2 * torch.randn_like(synthetic_encoded.anchor),
        synthetic_encoded.anchor,
        synthetic_encoded.text_tokens,
        synthetic_encoded.text_content_mask,
    )
    base_support = core.grounder(base, synthetic_encoded.anchor)
    changed_support = core.grounder(changed_intent, synthetic_encoded.anchor)
    assert not torch.allclose(base, changed_intent)
    assert not torch.allclose(base_support, changed_support)


def test_repeat_uses_actual_parent_for_reproposal_and_grounding(
    synthetic_encoded,
) -> None:
    core = IAGSRMECore(_config()).eval()
    _activate_reproposal(core)
    captured_reproposal_states: list[torch.Tensor] = []
    captured_grounder: list[tuple[torch.Tensor, torch.Tensor]] = []

    def capture_reproposal(_module, inputs) -> None:
        captured_reproposal_states.append(inputs[1].detach().clone())

    def capture_grounder(_module, inputs) -> None:
        captured_grounder.append(
            (inputs[0].detach().clone(), inputs[1].detach().clone())
        )

    handles = [
        core.reproposal.register_forward_pre_hook(capture_reproposal),
        core.grounder.register_forward_pre_hook(capture_grounder),
    ]
    try:
        output = core(synthetic_encoded, control="repeat_candidate_2")
    finally:
        for handle in handles:
            handle.remove()

    assert len(captured_reproposal_states) == 2
    assert len(captured_grounder) == 3
    assert torch.equal(captured_reproposal_states[0], output.trace[1].current_state)
    assert torch.equal(captured_reproposal_states[1], output.trace[2].current_state)
    for timestep, (intents, state) in enumerate(captured_grounder):
        assert torch.equal(state, output.trace[timestep].current_state)
        assert torch.equal(intents, output.trace[timestep].current_intents)
    assert all(step.selected_index.eq(1).all() for step in output.trace)
    assert not torch.equal(
        output.temporal_intents[:, 0, 1], output.temporal_intents[:, 1, 1]
    )


def test_frozen_t0_what_disables_only_reproposal(synthetic_encoded) -> None:
    core = IAGSRMECore(_config()).eval()
    _activate_reproposal(core)
    _force_non_stop(core)
    calls = {"reproposal": 0, "grounder": 0}

    def count(name):
        def hook(_module, _inputs) -> None:
            calls[name] += 1

        return hook

    handles = [
        core.reproposal.register_forward_pre_hook(count("reproposal")),
        core.grounder.register_forward_pre_hook(count("grounder")),
    ]
    try:
        output = core(synthetic_encoded, control="frozen_t0_what")
    finally:
        for handle in handles:
            handle.remove()
    expected = output.initial_intents[:, None].expand_as(output.temporal_intents)
    assert torch.equal(output.temporal_intents, expected)
    assert calls == {"reproposal": 0, "grounder": 3}
    for step in output.trace:
        expected_support = core.grounder(output.initial_intents, step.current_state)
        torch.testing.assert_close(
            step.raw_spatial_supports, expected_support, atol=0.0, rtol=0.0
        )


@pytest.mark.parametrize("control", ["full", "clone_candidate_1", "mean_candidate"])
def test_temporal_intent_trace_is_raw_under_controls(
    synthetic_encoded, control: str
) -> None:
    core = IAGSRMECore(_config()).eval()
    _activate_reproposal(core)
    _force_non_stop(core)
    output = core(synthetic_encoded, control=control)
    assert output.temporal_intents.shape == (3, 3, 4, 32)
    assert torch.equal(output.temporal_intents[:, 0], output.initial_intents)
    for timestep, step in enumerate(output.trace):
        assert torch.equal(
            output.temporal_intents[:, timestep], step.current_intents
        )
        expected, _ = core.reproposal(
            output.initial_intents,
            step.current_state,
            output.anchor,
            output.text_tokens,
            output.text_content_mask,
        ) if timestep > 0 else (output.initial_intents, None)
        torch.testing.assert_close(
            step.current_intents, expected, atol=0.0, rtol=0.0
        )


def test_target_permutation_changes_only_offline_diagnostics(
    synthetic_encoded,
) -> None:
    core = IAGSRMECore(_config()).eval()
    _activate_reproposal(core)
    _force_non_stop(core)
    assert "target" not in inspect.signature(core.forward).parameters
    output = core(synthetic_encoded)
    snapshots = {
        "base": output.initial_intents.clone(),
        "intent": output.temporal_intents.clone(),
        "support": output.temporal_supports.clone(),
        "state": output.final_state.clone(),
        "query": output.final_query.clone(),
        "candidate_states": [step.candidate_states.clone() for step in output.trace],
        "candidate_queries": [step.candidate_queries.clone() for step in output.trace],
        "selected": [step.selected_index.clone() for step in output.trace],
    }
    targets = torch.nn.functional.normalize(
        output.trace[0].candidate_queries[:, 0].detach(), dim=-1
    )
    first = SelectedPathMarginalAccumulator()
    second = SelectedPathMarginalAccumulator()
    first.update(output, targets)
    second.update(output, targets.roll(1, 0))
    assert first.summary() != second.summary()
    assert torch.equal(output.initial_intents, snapshots["base"])
    assert torch.equal(output.temporal_intents, snapshots["intent"])
    assert torch.equal(output.temporal_supports, snapshots["support"])
    assert torch.equal(output.final_state, snapshots["state"])
    assert torch.equal(output.final_query, snapshots["query"])
    for timestep, step in enumerate(output.trace):
        assert torch.equal(step.candidate_states, snapshots["candidate_states"][timestep])
        assert torch.equal(step.candidate_queries, snapshots["candidate_queries"][timestep])
        assert torch.equal(step.selected_index, snapshots["selected"][timestep])


def test_zero_init_then_upstream_gradient_and_parameter_movement(
    synthetic_encoded,
) -> None:
    core = IAGSRMECore(_config()).train()
    _force_non_stop(core)
    assert core.reproposal is not None
    optimizer = SGD(core.parameters(), lr=1e-2)
    families, representatives = _reproposal_audit_groups(core.reproposal)
    initial_parameters = {
        name: parameter.detach().clone()
        for name, parameter in representatives.items()
    }
    intent_weights = torch.linspace(-1.0, 1.0, 32)
    support_weights = torch.linspace(-1.0, 1.0, 13)

    first = core(synthetic_encoded)
    first_loss = (
        first.temporal_intents[:, 1] * intent_weights
    ).sum() + (first.temporal_supports[:, 1] * support_weights).sum()
    first_loss.backward()
    output_gradient = sum(
        float(parameter.grad.abs().sum())
        for parameter in families["reproposal_output"]
        if parameter.grad is not None
    )
    assert output_gradient > 0
    for name, parameters in families.items():
        if name == "reproposal_output":
            continue
        assert all(
            parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
            for parameter in parameters
        )
    optimizer.step()
    assert not torch.equal(
        representatives["reproposal_output"].detach(),
        initial_parameters["reproposal_output"],
    )

    optimizer.zero_grad(set_to_none=True)
    second = core(synthetic_encoded)
    second_loss = (
        second.temporal_intents[:, 1] * intent_weights
    ).sum() + (second.temporal_supports[:, 1] * support_weights).sum()
    second_loss.backward()
    for parameters in families.values():
        gradients = [
            parameter.grad for parameter in parameters if parameter.grad is not None
        ]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0
    for gradient in (
        core.intent_encoder.query_bank.grad,
        core.grounder.intent_projection.weight.grad,
    ):
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
    optimizer.step()
    for name, parameter in representatives.items():
        assert not torch.equal(parameter.detach(), initial_parameters[name]), name


def test_temporal_intent_diagnostics_and_checkpoint_replay(
    synthetic_encoded,
) -> None:
    core = IAGSRMECore(_config()).eval()
    _activate_reproposal(core)
    _force_non_stop(core)
    output = core(synthetic_encoded)
    accumulator = TemporalIntentAccumulator()
    accumulator.update(output)
    summary = accumulator.summary()
    assert summary["enable_dynamic_reproposal"] is True
    assert len(summary["per_timestep"]) == 3
    assert len(summary["per_transition"]) == 2
    assert summary["per_transition"][0][
        "candidate_intent_displacement_cosine_matrix"
    ] is not None
    conditioned = summary["per_transition"][0]["conditioned_on_previous_decision"]
    assert conditioned["same_candidate_executed"]["l2_change"]["count"] == 3
    assert conditioned["other_candidate_executed"]["l2_change"]["count"] == 9
    assert "stop" not in conditioned

    checkpoint = {
        "metadata": {
            "model_config": asdict(core.config),
            "architecture_generation": (
                "r1c2_dynamic_current_state_reproposal_v1"
            ),
        },
        "model": {f"core.{key}": value for key, value in core.state_dict().items()},
    }
    config, provenance = _resolve_checkpoint_model_config(
        checkpoint, checkpoint["model"], retrieval_dim=24
    )
    assert config.enable_dynamic_reproposal is True
    assert config.enable_dynamic_regrounding is True
    assert config.enable_dynamic_applicability is False
    assert provenance["architecture_generation"] == (
        "r1c2_dynamic_current_state_reproposal_v1"
    )
    replay = IAGSRMECore(config).eval()
    replay.load_state_dict(core.state_dict(), strict=True)
    replay_output = replay(synthetic_encoded)
    for name in (
        "initial_intents",
        "temporal_intents",
        "temporal_supports",
        "final_state",
        "final_query",
    ):
        torch.testing.assert_close(
            getattr(output, name), getattr(replay_output, name), atol=0.0, rtol=0.0
        )
    guard = _checkpoint_replay_guard(
        {"metric": 38.0}, provenance, replayed_mean_recall=38.00001
    )
    assert guard["trusted_r1c2_replay"] is True
    assert guard["trusted_r1c1_replay"] is False


def test_temporal_intent_diagnostics_exclude_absorbed_stop_lineage(
    synthetic_encoded,
) -> None:
    core = IAGSRMECore(_config()).eval()
    _activate_reproposal(core)
    output = core(synthetic_encoded, control="zero_edit")
    assert output.trace[0].live_before.all()
    assert not output.trace[1].live_before.any()
    assert not output.trace[2].live_before.any()

    accumulator = TemporalIntentAccumulator()
    accumulator.update(output)
    intent_summary = accumulator.summary()
    grounding_accumulator = TemporalGroundingAccumulator()
    grounding_accumulator.update(output)
    grounding_summary = grounding_accumulator.summary()
    assert [
        item["live_parent_count"] for item in intent_summary["per_timestep"]
    ] == [
        synthetic_encoded.anchor.shape[0],
        0,
        0,
    ]
    for transition in intent_summary["per_transition"]:
        assert transition["intent_l2_change"]["count"] == 0
        assert transition[
            "candidate_intent_displacement_alignment_off_diagonal"
        ]["valid_pair_count"] == 0
        assert "stop" not in transition["conditioned_on_previous_decision"]
    for transition in grounding_summary["per_transition"]:
        assert transition["support_l1_change"]["count"] == 0
        assert transition[
            "candidate_displacement_cosine_off_diagonal"
        ]["valid_pair_count"] == 0
        assert "stop" not in transition["conditioned_on_previous_decision"]

    expected_parents = synthetic_encoded.anchor.shape[0]
    intent_hypothetical = intent_summary[
        "hypothetical_post_stop_recomputation"
    ]
    grounding_hypothetical = grounding_summary[
        "hypothetical_post_stop_recomputation"
    ]
    for hypothetical in (intent_hypothetical, grounding_hypothetical):
        assert hypothetical["excluded_from_primary_temporal_metrics"] is True
        assert "never enter execution" in hypothetical["execution_semantics"]
        assert hypothetical["per_transition"][0][
            "newly_stopped_parent_count"
        ] == expected_parents
        assert hypothetical["per_transition"][1][
            "newly_stopped_parent_count"
        ] == 0
    assert intent_hypothetical["per_transition"][0]["intent_l2_change"][
        "count"
    ] == expected_parents * 4
    assert grounding_hypothetical["per_transition"][0]["support_l1_change"][
        "count"
    ] == expected_parents * 4


def test_canonical_hydra_config_is_matched_to_r1c1() -> None:
    config_dir = str(Path(__file__).resolve().parents[1] / "conf")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        config = compose(
            config_name="config",
            overrides=[
                "model=iag_srme_r1c2_dynamic_reproposal",
                "experiment=iag_srme_r1c2_dynamic_reproposal",
                "protocol=fashioniq_original",
            ],
        )
    assert config.model.query_cap == 1000.0
    assert config.model.enable_dynamic_regrounding is True
    assert config.model.enable_dynamic_reproposal is True
    assert config.model.enable_dynamic_applicability is False
    assert config.model.enable_visual_null is False
    assert config.model.grounding_normalization == "entmax15"
    assert config.experiment.epochs == 20
    assert config.experiment.batch_size == 32
    assert config.experiment.eval_batch_size == 32
    assert config.experiment.gallery_batch_size == 128
    assert config.experiment.num_workers == 8
    assert config.experiment.learning_rate == 1e-5
    assert config.experiment.weight_decay == 0.01
    assert config.experiment.train_caption_policy == "ordered_and"
    assert config.experiment.val_caption_policy == "ordered_and"
