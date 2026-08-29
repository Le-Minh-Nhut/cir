import torch

from diagnostics.iag_srme import MATCHED_COMPUTE_CONTROLS, summarize_trajectory


def test_full_shape_contract_and_finite_values(core, synthetic_encoded) -> None:
    output = core.eval()(synthetic_encoded)
    batch, tokens, width = synthetic_encoded.anchor.shape
    assert output.intents.shape == (batch, 4, width)
    assert output.supports.shape == (batch, 4, tokens)
    assert output.final_state.shape == (batch, tokens, width)
    assert output.final_query.shape == (batch, 24)
    assert len(output.trace) == 3
    for step in output.trace:
        assert step.contexts.shape == (batch, 4, width)
        assert step.delta_z.shape == (batch, 4, tokens, width)
        assert step.candidate_queries.shape == (batch, 4, 24)
        assert step.scores.shape == (batch, 4)
        assert step.logits_with_stop.shape == (batch, 5)
    tensors = [output.final_query, output.final_state, output.intents, output.supports]
    tensors.extend(step.delta_z for step in output.trace)
    assert all(torch.isfinite(tensor).all() for tensor in tensors)
    diagnostics = summarize_trajectory(output)
    assert all(torch.isfinite(value).all() for value in diagnostics.values())


def test_all_matched_compute_controls_execute(core, synthetic_encoded) -> None:
    core.eval()
    for control in MATCHED_COMPUTE_CONTROLS:
        output = core(synthetic_encoded, control=control)
        assert output.final_query.shape == (3, 24)
        assert torch.isfinite(output.final_query).all()

