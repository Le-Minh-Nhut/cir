from __future__ import annotations

from dataclasses import replace

import torch

from diagnose_iag_srme_checkpoint import (
    SelectedPathMarginalAccumulator,
    ValidationDiagnosticAccumulator,
    _failure_flags,
)
from diagnostics.iag_srme import (
    flatten_delta_z,
    functional_effective_rank,
    off_diagonal_values,
    pairwise_cosine_matrix,
    verify_same_parent_counterfactuals,
)


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
