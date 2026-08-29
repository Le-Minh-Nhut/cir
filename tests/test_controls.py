from __future__ import annotations

import torch


def _assert_candidate_axis_identical(tensor: torch.Tensor) -> None:
    expected = tensor[:, :1].expand_as(tensor)
    assert torch.equal(tensor, expected)


def test_clone_and_mean_controls_make_all_scorer_consequences_identical(
    core, synthetic_encoded
) -> None:
    core.eval()
    for control in ("clone_candidate_1", "mean_candidate"):
        output = core(synthetic_encoded, control=control)
        for step in output.trace:
            for tensor in (
                step.original_evidence,
                step.current_evidence,
                step.accumulated_local_change,
                step.contexts,
                step.delta_z,
                step.candidate_states,
                step.candidate_queries,
                step.delta_q,
                step.scores,
            ):
                _assert_candidate_axis_identical(tensor)


def test_repeat_and_single_candidate_controls_execute_declared_actions(
    core, synthetic_encoded
) -> None:
    core.eval()
    repeated = core(synthetic_encoded, control="repeat_candidate_3")
    for step in repeated.trace:
        assert step.selected_index.eq(2).all()

    single = core(synthetic_encoded, control="single_candidate")
    assert single.trace[0].selected_index.lt(4).all()
    assert single.trace[1].selected_index.eq(4).all()
    assert not single.trace[2].live_before.any()


def test_frozen_t0_order_is_reused_without_rescoring_order(core, synthetic_encoded) -> None:
    core.eval()
    output = core(synthetic_encoded, control="frozen_t0_order")
    order = output.trace[0].scores.argsort(dim=-1, descending=True)
    for timestep, step in enumerate(output.trace):
        assert torch.equal(step.selected_index, order[:, timestep])
