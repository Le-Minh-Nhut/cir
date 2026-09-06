from __future__ import annotations

import torch

from losses.objective import IAGSRMEObjective, ObjectiveConfig


def _gradient_sum(module) -> float:
    gradients = [
        float(parameter.grad.detach().abs().sum())
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    return sum(gradients)


def test_score_losses_only_update_shared_scorenet(model, features) -> None:
    initial, tokens, text, mask = features
    output = model.forward_from_features(initial, tokens, text, mask)
    targets = torch.randn(3, 12, requires_grad=True)
    objective = IAGSRMEObjective(
        ObjectiveConfig(terminal_weight=0.0, lambda_pair=1.0, lambda_gain=1.0)
    )
    objective(output, targets, ["a", "b", "c"])["total"].backward()

    for name in (
        "context_global",
        "context_text",
        "context_norm",
        "action_projection",
        "local_projection",
        "global_projection",
        "net",
    ):
        assert _gradient_sum(getattr(model.score_net, name)) > 0, name
    for name in ("backbone", "proposal", "grounder", "action_fusion", "executor"):
        assert _gradient_sum(getattr(model, name)) == 0, name
    assert targets.grad is None or targets.grad.count_nonzero() == 0


def test_terminal_gradient_uses_only_hard_committed_sibling(model, features) -> None:
    initial, tokens, text, mask = features
    model.config = type(model.config)(
        width=model.config.width,
        num_candidates=model.config.num_candidates,
        max_steps=1,
        num_heads=model.config.num_heads,
        exec_dim=model.config.exec_dim,
        epsilon_stop=model.config.epsilon_stop,
        stop_enabled=False,
        read_scale_init=model.config.read_scale_init,
        exec_scale_init=model.config.exec_scale_init,
        exec_bias_init=model.config.exec_bias_init,
        scale_min=model.config.scale_min,
        scale_max=model.config.scale_max,
        score_dropout=model.config.score_dropout,
    )
    torch.nn.init.normal_(model.executor.state_up.weight, std=0.03)
    with torch.no_grad():
        model.score_net.net[-1].weight.zero_()
        model.score_net.net[-1].bias.fill_(1.0)
    output = model.forward_from_features(initial, tokens, text, mask)
    candidates = output["steps"][0]["candidate_states"]
    candidates.retain_grad()
    targets = torch.randn(3, 12)
    objective = IAGSRMEObjective(
        ObjectiveConfig(terminal_weight=1.0, lambda_pair=0.0, lambda_gain=0.0)
    )
    objective(output, targets, ["a", "b", "c"])["total"].backward()

    assert candidates.grad[:, 0].abs().sum() > 0
    assert torch.equal(candidates.grad[:, 1:], torch.zeros_like(candidates.grad[:, 1:]))


def test_target_shuffle_changes_teacher_loss_not_live_forward(model, features) -> None:
    initial, tokens, text, mask = features
    model.eval()
    first = model.forward_from_features(initial, tokens, text, mask)
    second = model.forward_from_features(initial, tokens, text, mask)
    for left, right in zip(first["steps"], second["steps"], strict=True):
        for key in ("proposals", "grounding", "actions", "candidate_states", "scores"):
            assert torch.equal(left[key], right[key])

    targets = torch.randn(3, 12)
    objective = IAGSRMEObjective(ObjectiveConfig())
    normal = objective(first, targets, ["a", "b", "c"])["terminal"]
    shuffled = objective(first, targets.flip(0), ["c", "b", "a"])["terminal"]
    assert not torch.equal(normal, shuffled)
