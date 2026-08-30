from __future__ import annotations

from dataclasses import replace

import torch

from diagnose_iag_srme_checkpoint import ValidationDiagnosticAccumulator
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
