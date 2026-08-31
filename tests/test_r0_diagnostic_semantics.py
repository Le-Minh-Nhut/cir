from __future__ import annotations

from dataclasses import replace

import torch

from diagnose_iag_srme_checkpoint import (
    SelectedPathMarginalAccumulator,
    ValidationDiagnosticAccumulator,
    _EffectActivityStats,
    _MaskedMatrixStats,
    _failure_flags,
    _resolve_checkpoint_model_config,
)
from diagnostics.iag_srme import (
    FUNCTIONAL_ACTIVITY_EPSILON,
    flatten_delta_z,
    functional_effect_activity,
    functional_effective_rank,
    masked_pairwise_cosine,
    off_diagonal_values,
    pairwise_cosine_matrix,
    verify_same_parent_counterfactuals,
)


def _functional_effect_statistics(values: torch.Tensor) -> dict[str, object]:
    _, active = functional_effect_activity(values)
    cosine, valid = masked_pairwise_cosine(values)
    activity = _EffectActivityStats(values.shape[-2])
    activity.update(active)
    pairwise = _MaskedMatrixStats(values.shape[-2])
    pairwise.update(cosine, valid)
    return {
        "rank": functional_effective_rank(values),
        "activity": activity.summary(),
        "matrix": pairwise.nullable_matrix(),
        "pairwise": pairwise.off_diagonal_summary(),
    }


def test_zero_functional_effects_are_dead_not_diverse() -> None:
    effects = torch.zeros(2, 4, 8)
    summary = _functional_effect_statistics(effects)

    torch.testing.assert_close(summary["rank"], torch.zeros(2))
    assert summary["activity"]["dead_candidate_fraction"] == 1.0
    assert summary["activity"]["dead_parent_fraction"] == 1.0
    assert summary["activity"]["active_candidate_fraction"] == 0.0
    assert summary["pairwise"]["valid_pair_fraction"] == 0.0
    assert summary["pairwise"]["valid_pair_count"] == 0
    assert all(value is None for row in summary["matrix"] for value in row)


def test_cloned_active_functional_effects_have_rank_one_and_cosine_one() -> None:
    vector = torch.arange(1.0, 9.0)
    effects = vector.view(1, 1, -1).expand(2, 4, -1)
    summary = _functional_effect_statistics(effects)

    torch.testing.assert_close(summary["rank"], torch.ones(2), atol=1e-5, rtol=1e-5)
    assert summary["activity"]["dead_candidate_fraction"] == 0.0
    assert summary["activity"]["dead_parent_fraction"] == 0.0
    assert summary["pairwise"]["valid_pair_fraction"] == 1.0
    assert abs(summary["pairwise"]["mean"] - 1.0) < 1e-6


def test_orthogonal_active_functional_effects_have_rank_four() -> None:
    effects = torch.eye(4).unsqueeze(0).expand(2, -1, -1)
    summary = _functional_effect_statistics(effects)

    torch.testing.assert_close(summary["rank"], torch.full((2,), 4.0), atol=1e-5, rtol=1e-5)
    assert summary["activity"]["dead_candidate_fraction"] == 0.0
    assert summary["pairwise"]["valid_pair_fraction"] == 1.0
    assert abs(summary["pairwise"]["mean"]) < 1e-6


def test_mixed_active_dead_effects_only_count_active_pairs() -> None:
    effects = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]]
    )
    summary = _functional_effect_statistics(effects)

    assert summary["activity"]["dead_candidate_fraction"] == 0.5
    assert summary["activity"]["dead_parent_fraction"] == 0.0
    assert summary["pairwise"]["valid_pair_count"] == 2
    assert summary["pairwise"]["possible_pair_count"] == 12
    assert summary["pairwise"]["valid_pair_fraction"] == 1 / 6
    assert summary["matrix"][0][1] == 0.0
    assert summary["matrix"][0][2] is None
    assert summary["matrix"][2][3] is None


def _minimal_checkpoint_state() -> dict[str, torch.Tensor]:
    return {"core.intent_encoder.query_bank": torch.zeros(4, 256)}


def test_legacy_checkpoint_model_config_provenance_is_explicit() -> None:
    config, provenance = _resolve_checkpoint_model_config(
        {"metadata": {}}, _minimal_checkpoint_state(), retrieval_dim=512
    )

    assert config.max_steps == 3
    assert config.num_heads == 8
    assert config.lambda_z == 0.10
    assert config.query_cap == 0.50
    assert config.selector_temperature == 1.0
    assert provenance["source"] == "legacy_checkpoint_plus_canonical_assumption"
    assert provenance["fully_self_describing"] is False
    assert provenance["fully_self_describing_model_config"] is False
    assert "model architecture/configuration replay only" in provenance[
        "provenance_scope"
    ]
    assert provenance["warning"]
    assert provenance["assumed_config"]["width"] == 256


def test_self_describing_checkpoint_model_config_is_preferred() -> None:
    serialized = {
        "width": 256,
        "num_candidates": 4,
        "max_steps": 3,
        "num_heads": 4,
        "retrieval_dim": 512,
        "lambda_z": 0.07,
        "query_cap": 0.40,
        "selector_temperature": 0.75,
        "selector_gumbel_noise": True,
        "enable_claim_head": False,
        "enable_factor_head": False,
    }
    config, provenance = _resolve_checkpoint_model_config(
        {"metadata": {"model_config": serialized}},
        _minimal_checkpoint_state(),
        retrieval_dim=512,
    )

    assert config.num_heads == 4
    assert config.lambda_z == 0.07
    assert config.query_cap == 0.40
    assert config.selector_temperature == 0.75
    assert config.selector_gumbel_noise is False
    assert provenance["source"] == "checkpoint"
    assert provenance["fully_self_describing"] is True
    assert provenance["fully_self_describing_model_config"] is True
    assert provenance["warning"] is None


def test_same_parent_state_and_true_delta_q_invariants(core, synthetic_encoded) -> None:
    core.eval()
    output = core(synthetic_encoded)

    for step in output.trace:
        verify_same_parent_counterfactuals(step)
        torch.testing.assert_close(
            step.candidate_states,
            step.current_state[:, None] + step.delta_z,
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            step.delta_q,
            step.candidate_queries - step.current_query[:, None],
            atol=0.0,
            rtol=0.0,
        )


def test_pairwise_diagnostics_preserve_candidate_dimension() -> None:
    candidates = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]]]
    )
    matrix = pairwise_cosine_matrix(candidates)

    assert matrix.shape == (1, 4, 4)
    assert matrix[0, 0, 1] == 0.0
    assert matrix[0, 0, 3] == -1.0
    assert 0.70 < float(matrix[0, 1, 2]) < 0.71
    assert off_diagonal_values(matrix).shape == (1, 12)


def test_cloned_candidate_sanity_for_intent_support_delta_z_and_delta_q() -> None:
    torch.manual_seed(601)
    intent = torch.randn(3, 1, 8).expand(-1, 4, -1)
    support = torch.rand(3, 1, 12).expand(-1, 4, -1)
    delta_z = torch.randn(3, 1, 5, 8).expand(-1, 4, -1, -1)
    delta_q = torch.randn(3, 1, 16).expand(-1, 4, -1)

    for values in (intent, support, flatten_delta_z(delta_z), delta_q):
        off_diagonal = off_diagonal_values(pairwise_cosine_matrix(values))
        torch.testing.assert_close(off_diagonal, torch.ones_like(off_diagonal), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        functional_effective_rank(delta_q),
        torch.ones(3),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        functional_effective_rank(flatten_delta_z(delta_z)),
        torch.ones(3),
        atol=1e-5,
        rtol=1e-5,
    )


def test_orthogonal_candidate_effects_have_full_effective_rank() -> None:
    orthogonal = torch.eye(4).unsqueeze(0).expand(2, -1, -1)
    rank = functional_effective_rank(orthogonal)
    torch.testing.assert_close(rank, torch.full((2,), 4.0), atol=1e-5, rtol=1e-5)


def test_live_parent_denominator_excludes_absorbed_trajectories(
    core, synthetic_encoded
) -> None:
    core.eval()
    output = core(synthetic_encoded)
    assert output.intents.shape[0] == 3
    device = output.intents.device
    live_by_timestep = (
        torch.tensor([True, True, True], device=device),
        torch.tensor([False, True, True], device=device),
        torch.tensor([False, False, True], device=device),
    )
    selected_by_timestep = (
        torch.tensor([4, 0, 1], device=device),
        torch.tensor([4, 4, 2], device=device),
        torch.tensor([4, 4, 3], device=device),
    )
    stopped_by_timestep = (
        torch.tensor([True, False, False], device=device),
        torch.tensor([False, True, False], device=device),
        torch.tensor([False, False, False], device=device),
    )
    trace = tuple(
        replace(
            step,
            live_before=live_by_timestep[timestep],
            selected_index=selected_by_timestep[timestep],
            stopped_now=stopped_by_timestep[timestep],
        )
        for timestep, step in enumerate(output.trace)
    )
    audited_output = replace(output, trace=trace)
    accumulator = ValidationDiagnosticAccumulator()
    accumulator.update(audited_output)
    summary = accumulator.functional_summary()["per_timestep"]

    assert [item["live_parent_count"] for item in summary] == [3, 2, 1]
    dynamic = accumulator.dynamic_summary()["context"]["per_transition"]
    assert [item["live_executed_parent_count"] for item in dynamic] == [2, 1]
    assert all("metric_population" in item for item in dynamic)


def test_selected_path_target_diagnostics_are_offline_and_do_not_mutate_forward(
    core, synthetic_encoded
) -> None:
    core.eval()
    with torch.no_grad():
        core.scorer.score_head[-1].weight.zero_()
        core.scorer.score_head[-1].bias.fill_(1.0)
    output = core(synthetic_encoded)
    before = {
        "final_query": output.final_query.detach().clone(),
        "final_state": output.final_state.detach().clone(),
        "intents": output.intents.detach().clone(),
        "supports": output.supports.detach().clone(),
        "scores": torch.stack([step.scores for step in output.trace]).detach().clone(),
    }
    target_a = torch.randn_like(output.final_query)
    target_b = -target_a
    audit_a = SelectedPathMarginalAccumulator()
    audit_b = SelectedPathMarginalAccumulator()
    audit_a.update(output, target_a)
    audit_b.update(output, target_b)

    for name, expected in before.items():
        actual = (
            torch.stack([step.scores for step in output.trace])
            if name == "scores"
            else getattr(output, name)
        )
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    summary_a = audit_a.summary()["target_similarity_improvement_by_timestep"]
    summary_b = audit_b.summary()["target_similarity_improvement_by_timestep"]
    assert summary_a[0]["selected_non_stop_transition_count"] == 3
    assert summary_a[0]["delta_target_cosine_similarity"]["mean"] != summary_b[0][
        "delta_target_cosine_similarity"
    ]["mean"]


def test_r0_instrumentation_freezes_numerical_forward_behavior(
    core, synthetic_encoded
) -> None:
    core.eval()
    before = core(synthetic_encoded)
    accumulator = ValidationDiagnosticAccumulator()
    accumulator.update(before)
    _ = accumulator.intent_summary()
    _ = accumulator.grounding_summary()
    _ = accumulator.functional_summary()
    _ = accumulator.dynamic_summary()
    after = core(synthetic_encoded)

    for name in ("final_query", "final_state", "anchor", "intents", "supports"):
        torch.testing.assert_close(
            getattr(after, name), getattr(before, name), atol=0.0, rtol=0.0
        )
    for before_step, after_step in zip(before.trace, after.trace, strict=True):
        for name in (
            "current_state",
            "current_query",
            "contexts",
            "delta_z",
            "candidate_states",
            "candidate_queries",
            "delta_q",
            "scores",
            "selected_index",
            "next_state",
            "next_query",
        ):
            torch.testing.assert_close(
                getattr(after_step, name),
                getattr(before_step, name),
                atol=0.0,
                rtol=0.0,
            )


def test_functional_report_keeps_timestep_matrices_and_retention(
    core, synthetic_encoded
) -> None:
    core.eval()
    output = core(synthetic_encoded)
    accumulator = ValidationDiagnosticAccumulator()
    accumulator.update(output)
    summary = accumulator.functional_summary()

    assert len(summary["per_timestep"]) == 3
    for timestep in summary["per_timestep"]:
        assert torch.tensor(timestep["context_pairwise_cosine_matrix"]).shape == (4, 4)
        assert torch.tensor(timestep["delta_z_pairwise_cosine_matrix"]).shape == (4, 4)
        assert torch.tensor(timestep["delta_q_pairwise_cosine_matrix"]).shape == (4, 4)
        assert timestep["median_delta_q_norm"] is not None
    assert set(summary["late_step_effect_retention"]) >= {
        "mean_delta_q_norm_t1_over_t0",
        "mean_delta_q_norm_t2_over_t0",
        "mean_delta_q_norm_t2_over_t1",
    }


def test_every_failure_flag_has_an_auditable_noncausal_contract(
    core, synthetic_encoded
) -> None:
    core.eval()
    output = core(synthetic_encoded)
    accumulator = ValidationDiagnosticAccumulator()
    accumulator.update(output)
    selection = accumulator.selection_summary()
    grounding = accumulator.grounding_summary()
    functional = accumulator.functional_summary()
    specialization = accumulator.specialization_summary()
    retrieval = {
        "full": {"mean_recall": 10.0},
        "reference_only": {"mean_recall": 9.0},
        "counterfactual_same_parent_by_timestep": {
            "t0": {"best_single_candidate_oracle": {"mean_recall": 11.0}}
        },
    }
    retrieval.update(
        {f"single_{index}": {"mean_recall": 10.0 + index} for index in range(4)}
    )
    retrieval.update(
        {f"repeat_{index}": {"mean_recall": 9.0 + index} for index in range(4)}
    )
    audit = _failure_flags(
        selection, grounding, functional, retrieval, specialization
    )

    assert set(audit["flags"]) == set(audit["per_flag_audit_contract"])
    for contract in audit["per_flag_audit_contract"].values():
        assert contract["condition"]
        assert isinstance(contract["supporting_numbers"], dict)
        assert isinstance(contract["thresholds"], dict)
        assert contract["interpretation_limitation"]


def test_functional_failure_flags_are_timestep_specific(
    core, synthetic_encoded
) -> None:
    core.eval()
    output = core(synthetic_encoded)
    accumulator = ValidationDiagnosticAccumulator()
    accumulator.update(output)
    selection = accumulator.selection_summary()
    grounding = accumulator.grounding_summary()
    functional = accumulator.functional_summary()
    specialization = accumulator.specialization_summary()
    synthetic = (
        (0.0, 4.0, 0.0),
        (0.99, 1.0, 0.0),
        (None, 0.0, 1.0),
    )
    for timestep, (cosine, rank, dead_fraction) in enumerate(synthetic):
        functional["per_timestep"][timestep].update(
            {
                "live_parent_count": 3,
                "pairwise_delta_q_cosine_mean_off_diagonal": cosine,
                "functional_effective_rank": rank,
                "dead_candidate_fraction": dead_fraction,
            }
        )
    retrieval = {
        "full": {"mean_recall": 10.0},
        "reference_only": {"mean_recall": 9.0},
        "counterfactual_same_parent_by_timestep": {
            "t0": {"best_single_candidate_oracle": {"mean_recall": 11.0}}
        },
        **{
            f"single_{index}": {"mean_recall": 10.0}
            for index in range(4)
        },
        **{
            f"repeat_{index}": {"mean_recall": 10.0}
            for index in range(4)
        },
    }
    flags = _failure_flags(
        selection, grounding, functional, retrieval, specialization
    )["flags"]

    assert flags["high_delta_q_similarity_t0"] is False
    assert flags["low_functional_effective_rank_t0"] is False
    assert flags["high_dead_delta_q_fraction_t0"] is False
    assert flags["high_delta_q_similarity_t1"] is True
    assert flags["low_functional_effective_rank_t1"] is True
    assert flags["high_dead_delta_q_fraction_t1"] is False
    assert flags["high_delta_q_similarity_t2"] is None
    assert flags["low_functional_effective_rank_t2"] is True
    assert flags["high_dead_delta_q_fraction_t2"] is True


def test_functional_failure_flags_are_null_without_live_parents(
    core, synthetic_encoded
) -> None:
    core.eval()
    output = core(synthetic_encoded)
    accumulator = ValidationDiagnosticAccumulator()
    accumulator.update(output)
    selection = accumulator.selection_summary()
    grounding = accumulator.grounding_summary()
    functional = accumulator.functional_summary()
    specialization = accumulator.specialization_summary()
    functional["per_timestep"][2] = {
        "timestep": 2,
        "live_parent_count": 0,
        "metric_population": "no live parents",
    }
    retrieval = {
        "full": {"mean_recall": 10.0},
        "reference_only": {"mean_recall": 9.0},
        "counterfactual_same_parent_by_timestep": {
            "t0": {"best_single_candidate_oracle": {"mean_recall": 11.0}}
        },
        **{
            f"single_{index}": {"mean_recall": 10.0}
            for index in range(4)
        },
        **{
            f"repeat_{index}": {"mean_recall": 10.0}
            for index in range(4)
        },
    }
    flags = _failure_flags(
        selection, grounding, functional, retrieval, specialization
    )["flags"]

    assert flags["high_delta_q_similarity_t2"] is None
    assert flags["low_functional_effective_rank_t2"] is None
    assert flags["high_dead_delta_q_fraction_t2"] is None


def test_functional_activity_epsilon_is_exposed_in_report(
    core, synthetic_encoded
) -> None:
    core.eval()
    accumulator = ValidationDiagnosticAccumulator()
    accumulator.update(core(synthetic_encoded))

    assert (
        accumulator.functional_summary()["functional_effect_activity_epsilon"]
        == FUNCTIONAL_ACTIVITY_EPSILON
    )
