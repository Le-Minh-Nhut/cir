from __future__ import annotations

from dataclasses import asdict, replace
import inspect
from pathlib import Path
from unittest.mock import patch

from hydra import compose, initialize_config_dir
import pytest
import torch
from torch.optim import AdamW

from canary_train_iag_srme import _semantic_claim_audit_groups
from diagnose_iag_srme_checkpoint import (
    ValidationDiagnosticAccumulator,
    _checkpoint_replay_guard,
    _resolve_checkpoint_model_config,
)
from models.iag_srme import IAGSRMEConfig, IAGSRMECore
from models.iag_srme.semantic_residual import (
    SemanticClaimModule,
    candidate_residual_previews,
    claimed_text_content,
    initialize_semantic_residual,
    select_next_residual,
)


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
        "enable_dynamic_regrounding": True,
        "enable_dynamic_reproposal": False,
        "enable_dynamic_applicability": False,
        "enable_semantic_residual": True,
    }
    values.update(overrides)
    return IAGSRMEConfig(**values)


def _force_non_stop(core: IAGSRMECore) -> None:
    with torch.no_grad():
        core.scorer.score_head[-1].weight.zero_()
        core.scorer.score_head[-1].bias.fill_(1.0)


def test_r2_config_rejects_scientific_stacking() -> None:
    with pytest.raises(ValueError, match="requires dynamic regrounding"):
        IAGSRMECore(_config(enable_dynamic_regrounding=False))
    with pytest.raises(ValueError, match="cannot retain unrestricted"):
        IAGSRMECore(_config(enable_dynamic_reproposal=True))
    with pytest.raises(ValueError, match="cannot stack R1b"):
        IAGSRMECore(_config(enable_dynamic_applicability=True))


def test_residual_initialization_and_padding_never_claimed(synthetic_encoded) -> None:
    mask = synthetic_encoded.text_content_mask.clone()
    mask[:, -2:] = False
    encoded = replace(synthetic_encoded, text_content_mask=mask)
    core = IAGSRMECore(_config()).eval()
    output = core(encoded, control="zero_edit")
    expected = mask.float()
    assert output.initial_semantic_residual.dtype == torch.float32
    assert torch.equal(output.initial_semantic_residual, expected)
    assert torch.equal(output.final_semantic_residual, expected)
    assert torch.count_nonzero(output.temporal_semantic_claims[..., -2:]) == 0
    assert output.temporal_semantic_claims.dtype == torch.float32


def test_residual_monotonicity_selected_only_and_same_parent(
    synthetic_encoded,
) -> None:
    core = IAGSRMECore(_config()).eval()
    output = core(synthetic_encoded, control="repeat_candidate_2")
    residuals = output.temporal_semantic_residuals
    assert residuals.shape == (3, 4, 8)
    assert torch.all(residuals[:, 1:] <= residuals[:, :-1] + 1e-7)
    for step in output.trace:
        expected = candidate_residual_previews(
            step.parent_semantic_residual, step.effective_semantic_claims
        )
        torch.testing.assert_close(
            step.candidate_semantic_residuals, expected, atol=0.0, rtol=0.0
        )
        selected = step.action_hard[:, :4]
        expected_next = torch.einsum(
            "bk,bkl->bl", selected, expected
        )
        torch.testing.assert_close(
            step.next_semantic_residual, expected_next, atol=0.0, rtol=0.0
        )
        assert torch.equal(
            step.parent_semantic_residual,
            residuals[:, step.timestep],
        )


def test_frozen_residual_control_preserves_rho0(synthetic_encoded) -> None:
    core = IAGSRMECore(_config()).eval()
    _force_non_stop(core)
    output = core(synthetic_encoded, control="frozen_residual")
    expected = output.initial_semantic_residual[:, None].expand_as(
        output.temporal_semantic_residuals
    )
    assert torch.equal(output.temporal_semantic_residuals, expected)
    assert output.temporal_semantic_claims is not None


def test_claim_firewall_blocks_unclaimed_token_content(synthetic_encoded) -> None:
    tokens = synthetic_encoded.text_tokens.clone()
    residual = synthetic_encoded.text_content_mask.float()
    claims = torch.zeros(3, 4, 8)
    claims[:, 0, 1] = 1.0
    weights, content = claimed_text_content(tokens, claims, residual)
    changed = tokens.clone()
    changed[:, 6] += 1000.0
    changed_weights, changed_content = claimed_text_content(
        changed, claims, residual
    )
    assert torch.equal(weights, changed_weights)
    assert torch.equal(content[:, 0], changed_content[:, 0])

    core = IAGSRMECore(_config()).eval()
    first = core.intent_encoder.forward_weighted(
        tokens, synthetic_encoded.text_content_mask, weights
    )
    second = core.intent_encoder.forward_weighted(
        changed,
        synthetic_encoded.text_content_mask,
        changed_weights,
    )
    assert torch.equal(first[:, 0], second[:, 0])


def test_claim_swap_changes_executable_semantics(synthetic_encoded) -> None:
    core = IAGSRMECore(_config()).eval()
    _force_non_stop(core)
    controlled = torch.zeros(3, 4, 8)
    for candidate in range(4):
        controlled[:, candidate, candidate + 1] = 0.8
    logits = torch.zeros_like(controlled)

    def claims(*_args, **_kwargs):
        return logits, controlled

    with patch.object(core.semantic_claim, "forward", side_effect=claims):
        full = core(synthetic_encoded)
        swapped = core(synthetic_encoded, control="claim_swap")
    assert torch.equal(full.trace[0].raw_semantic_claims, controlled)
    assert torch.equal(
        swapped.trace[0].effective_semantic_claims,
        controlled.roll(1, dims=1),
    )
    assert not torch.equal(
        full.trace[0].claimed_text_content,
        swapped.trace[0].claimed_text_content,
    )
    assert not torch.equal(full.trace[0].delta_z, swapped.trace[0].delta_z)


def test_r2_t0_is_close_to_r1c1_parent(synthetic_encoded) -> None:
    parent = IAGSRMECore(
        _config(enable_semantic_residual=False)
    ).eval()
    r2 = IAGSRMECore(_config()).eval()
    missing, unexpected = r2.load_state_dict(parent.state_dict(), strict=False)
    assert missing and all(key.startswith("semantic_claim.") for key in missing)
    assert not unexpected
    _force_non_stop(parent)
    _force_non_stop(r2)
    baseline = parent(synthetic_encoded)
    output = r2(synthetic_encoded)
    assert float((output.intents - baseline.intents).detach().abs().max()) < 0.05
    assert float(
        (
            output.trace[0].raw_spatial_supports
            - baseline.trace[0].raw_spatial_supports
        ).detach().abs().max()
    ) < 0.02
    assert float(
        (output.trace[0].delta_z - baseline.trace[0].delta_z).detach().abs().max()
    ) < 0.01


def test_target_firewall_and_r2_trace_shapes(synthetic_encoded) -> None:
    core = IAGSRMECore(_config()).eval()
    assert "target" not in inspect.signature(core.forward).parameters
    assert "target" not in inspect.signature(core.semantic_claim.forward).parameters
    output = core(synthetic_encoded)
    assert output.temporal_semantic_residuals.shape == (3, 4, 8)
    assert output.temporal_semantic_claims.shape == (3, 3, 4, 8)
    assert output.temporal_intents.shape == (3, 3, 4, 32)
    assert output.temporal_supports.shape == (3, 3, 4, 13)


def test_amp_claim_and_residual_arithmetic_stays_fp32() -> None:
    module = SemanticClaimModule(width=8, initial_claim_probability=0.99)
    queries = torch.randn(4, 8).bfloat16()
    text = torch.randn(2, 5, 8).bfloat16()
    mask = torch.ones(2, 5, dtype=torch.bool)
    residual = initialize_semantic_residual(mask)
    state = torch.randn(2, 7, 8).bfloat16()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits, claims = module(queries, text, mask, residual, state)
        previews = candidate_residual_previews(residual, claims)
        action = torch.nn.functional.one_hot(
            torch.zeros(2, dtype=torch.long), 5
        ).float()
        updated = select_next_residual(previews, residual, action)
    assert logits.dtype == claims.dtype == updated.dtype == torch.float32
    assert torch.all(updated <= residual)
    assert torch.isfinite(claims).all()


def test_canonical_r2_hydra_config() -> None:
    config_dir = str(Path(__file__).resolve().parents[1] / "conf")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        config = compose(
            config_name="config",
            overrides=[
                "model=iag_srme_r2_semantic_residual",
                "experiment=iag_srme_r2_semantic_residual",
                "protocol=fashioniq_original",
            ],
        )
    assert config.model.query_cap == 1000.0
    assert config.model.enable_dynamic_regrounding is True
    assert config.model.enable_dynamic_reproposal is False
    assert config.model.enable_dynamic_applicability is False
    assert config.model.enable_semantic_residual is True
    assert config.model.claim_activation == "sigmoid"
    assert config.model.semantic_residual_fp32 is True
    assert config.experiment.epochs == 20
    assert config.experiment.learning_rate == 1e-5


def test_r2_diagnostics_and_control_traces_are_json_safe(synthetic_encoded) -> None:
    core = IAGSRMECore(_config()).eval()
    _force_non_stop(core)
    output = core(synthetic_encoded, control="repeat_candidate_1")
    accumulator = ValidationDiagnosticAccumulator(candidates=4, timesteps=3)
    accumulator.update(output)
    summary = accumulator.semantic_residual_summary()
    assert summary["enabled"] is True
    assert len(summary["residual_states"]) == 4
    assert len(summary["per_timestep"]) == 3
    assert summary["per_timestep"][0]["selected_transition_count"] == 3
    assert summary["per_timestep"][0]["claim_pairwise_cosine"]["mean"] is not None


def test_r2_checkpoint_replay_requires_exact_architecture(synthetic_encoded) -> None:
    core = IAGSRMECore(_config())
    state = {f"core.{key}": value for key, value in core.state_dict().items()}
    checkpoint = {
        "model_config": asdict(core.config),
        "metadata": {
            "architecture_generation": "r2_semantic_residual_claim_firewall_v1"
        },
        "metric": 12.5,
    }
    resolved, provenance = _resolve_checkpoint_model_config(
        checkpoint, state, retrieval_dim=24
    )
    assert resolved.enable_semantic_residual is True
    assert resolved.enable_dynamic_regrounding is True
    assert resolved.enable_dynamic_reproposal is False
    guard = _checkpoint_replay_guard(checkpoint, provenance, 12.5)
    assert guard["trusted_r2_replay"] is True
    assert all(guard["checks"].values())

    ambiguous = {"metadata": checkpoint["metadata"]}
    with pytest.raises(ValueError, match="exact replay is unsafe"):
        _resolve_checkpoint_model_config(ambiguous, state, retrieval_dim=24)


def test_claim_zero_init_then_upstream_learns(synthetic_encoded) -> None:
    core = IAGSRMECore(_config()).train()
    _force_non_stop(core)
    families, representatives = _semantic_claim_audit_groups(core.semantic_claim)
    initial = {name: parameter.detach().clone() for name, parameter in representatives.items()}
    optimizer = AdamW(core.parameters(), lr=1e-3)

    optimizer.zero_grad(set_to_none=True)
    first = core(synthetic_encoded, control="repeat_candidate_1")
    first.final_query.square().mean().backward()
    assert any(parameter.grad is not None for parameter in families["semantic_claim_output"])
    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        for name, parameters in families.items()
        if name != "semantic_claim_output"
        for parameter in parameters
    )
    optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    second = core(synthetic_encoded, control="repeat_candidate_1")
    second.final_query.square().mean().backward()
    for name, parameters in families.items():
        assert any(
            parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
            for parameter in parameters
        ), name
    optimizer.step()
    for name, parameter in representatives.items():
        assert not torch.equal(parameter, initial[name]), name
