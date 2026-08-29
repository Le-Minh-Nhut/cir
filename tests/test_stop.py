import torch


def test_stop_identity_and_absorbing(core, synthetic_encoded) -> None:
    with torch.no_grad():
        final = core.scorer.score_head[-1]
        final.weight.zero_()
        final.bias.fill_(-1.0)
    output = core.eval()(synthetic_encoded)
    for step in output.trace:
        assert torch.equal(step.selected_index, torch.full_like(step.selected_index, 4))
        assert torch.equal(step.next_state, step.current_state)
        assert torch.equal(step.next_query, step.current_query)
    assert torch.equal(output.final_state, synthetic_encoded.anchor)
    assert not output.trace[1].live_before.any()
    assert not output.trace[2].live_before.any()
