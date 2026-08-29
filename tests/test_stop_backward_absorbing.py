import copy

import torch

from losses.retrieval import TerminalRetrievalLoss
from models.iag_srme import BackboneOutput, IAGSRMEConfig, IAGSRMECore


def _build_core(max_steps: int) -> IAGSRMECore:
    return IAGSRMECore(
        IAGSRMEConfig(
            width=16,
            num_candidates=4,
            max_steps=max_steps,
            num_heads=4,
            retrieval_dim=12,
            selector_gumbel_noise=False,
        )
    )


def _encoded() -> BackboneOutput:
    return BackboneOutput(
        anchor=torch.randn(3, 9, 16),
        reference_global=torch.nn.functional.normalize(torch.randn(3, 12), dim=-1),
        text_tokens=torch.randn(3, 7, 16),
        text_global=torch.randn(3, 16),
        text_content_mask=torch.ones(3, 7, dtype=torch.bool),
    )


def _gradients(core: IAGSRMECore, encoded: BackboneOutput, targets: torch.Tensor):
    core.train()
    output = core(encoded)
    assert output.trace[0].selected_index.eq(4).all()
    loss = TerminalRetrievalLoss()(output.final_query, targets)
    loss.backward()
    return {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in core.named_parameters()
    }


def test_immediate_stop_has_identical_tmax1_and_tmax3_terminal_gradients() -> None:
    torch.manual_seed(123)
    one_step = _build_core(max_steps=1)
    three_steps = _build_core(max_steps=3)
    three_steps.load_state_dict(copy.deepcopy(one_step.state_dict()))
    for core in (one_step, three_steps):
        with torch.no_grad():
            core.scorer.score_head[-1].weight.zero_()
            core.scorer.score_head[-1].bias.fill_(-1.0)
    encoded = _encoded()
    targets = torch.nn.functional.normalize(torch.randn(3, 12), dim=-1)

    gradients_one = _gradients(one_step, encoded, targets)
    gradients_three = _gradients(three_steps, encoded, targets)

    assert not three_steps(encoded).trace[1].live_before.any()
    assert gradients_one.keys() == gradients_three.keys()
    for name in gradients_one:
        left, right = gradients_one[name], gradients_three[name]
        assert (left is None) == (right is None), name
        if left is not None and right is not None:
            assert torch.allclose(left, right, atol=1e-7, rtol=1e-6), name


def test_dead_selector_action_has_zero_logit_gradient() -> None:
    core = _build_core(max_steps=1).train()
    logits = torch.randn(2, 5, requires_grad=True)
    live = torch.zeros(2, dtype=torch.bool)
    action, hard = core.selector(logits, live)
    assert torch.equal(action, hard)
    # Attach a zero-valued dependency so autograd can explicitly report the masked gradient.
    (action.sum() + logits.sum() * 0).backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))
