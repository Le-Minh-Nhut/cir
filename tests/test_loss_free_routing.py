from __future__ import annotations

from dataclasses import replace

import torch

from losses.objective import IAGSRMEObjective, ObjectiveConfig
from models.iag_srme import IAGSRME
from training.engine import trainable_parameters


def _balanced_model(model, **changes):
    model.config = replace(model.config, loss_free_balance_enabled=True, **changes)
    return model


def _output(raw_scores: torch.Tensor, selected: torch.Tensor | None = None) -> dict[str, object]:
    candidates = raw_scores.shape[-1]
    raw_best = raw_scores.argmax(dim=-1)
    selected = raw_best if selected is None else selected
    return {
        "steps": [
            {
                "scores": raw_scores,
                "raw_best_idx": raw_best,
                "routed_idx": selected.clamp_max(candidates - 1),
                "selected_idx": selected,
            }
        ]
    }


def test_disabled_balance_preserves_raw_routing_and_zero_bias(model) -> None:
    raw = torch.tensor([[0.1, 0.4, 0.2, -0.1]])
    model.routing_bias.fill_(10.0)

    route = model.route_scores(raw)

    assert model.routing_bias.eq(10).all()
    assert route["routed_idx"].equal(raw.argmax(dim=-1))
    assert route["selection_scores"].equal(raw)


def test_bias_is_zero_buffer_not_optimizer_parameter(model) -> None:
    assert model.routing_bias.eq(0).all()
    assert not model.routing_bias.requires_grad
    assert "routing_bias" in dict(model.named_buffers())
    assert "routing_bias" not in dict(model.named_parameters())
    assert all(id(model.routing_bias) != id(parameter) for parameter in trainable_parameters(model))


def test_zero_bias_routes_raw_argmax_for_eligible_candidates(model) -> None:
    _balanced_model(model, stop_enabled=True, epsilon_stop=0.0)
    raw = torch.tensor([[-1.0, 0.2, 0.4, 0.1]])

    route = model.route_scores(raw)

    assert route["routed_idx"].equal(raw.argmax(dim=-1))
    assert route["selected_idx"].equal(raw.argmax(dim=-1))


def test_bias_changes_route_without_changing_raw_scores(model) -> None:
    _balanced_model(model, stop_enabled=False)
    raw = torch.tensor([[0.4, 0.3, 0.2, 0.1]])
    model.routing_bias.copy_(torch.tensor([0.0, 0.2, 0.0, 0.0]))

    route = model.route_scores(raw)

    assert route["raw_best_idx"].item() == 0
    assert route["routed_idx"].item() == 1
    assert route["selected_raw_score"].item() == raw[0, 1].item()
    assert raw.equal(torch.tensor([[0.4, 0.3, 0.2, 0.1]]))


def test_bias_update_uses_sign_of_committed_load(model) -> None:
    _balanced_model(model, loss_free_bias_update_rate=0.1)
    model.train()
    output = _output(torch.ones(4, 4), torch.tensor([0, 0, 0, 1]))

    diagnostics = model.update_routing_bias(output)

    torch.testing.assert_close(model.routing_bias, torch.tensor([-0.1, 0.0, 0.1, 0.1]))
    assert diagnostics["committed_selection_count"] == [3, 1, 0, 0]


def test_equal_load_and_stop_rows_do_not_change_bias(model) -> None:
    _balanced_model(model, loss_free_bias_update_rate=0.1)
    model.train()
    output = _output(torch.ones(5, 4), torch.tensor([0, 1, 2, 3, 4]))

    diagnostics = model.update_routing_bias(output)

    assert model.routing_bias.eq(0).all()
    assert diagnostics["committed_selection_count"] == [1, 1, 1, 1]
    assert diagnostics["stop_count"] == 1


def test_no_executions_and_evaluation_do_not_update_bias(model) -> None:
    _balanced_model(model, loss_free_bias_update_rate=0.1)
    stopped = _output(torch.ones(2, 4), torch.full((2,), 4))
    model.train()
    model.update_routing_bias(stopped)
    assert model.routing_bias.eq(0).all()

    model.eval()
    model.update_routing_bias(_output(torch.ones(2, 4), torch.tensor([0, 0])))
    assert model.routing_bias.eq(0).all()


def test_stop_uses_raw_eligibility_not_positive_bias(model) -> None:
    _balanced_model(model, stop_enabled=True, epsilon_stop=0.0)
    model.routing_bias.copy_(torch.tensor([100.0, 0.0, 0.0, 0.0]))

    route = model.route_scores(torch.tensor([[-0.1, -0.2, -0.3, -0.4]]))

    assert route["selected_idx"].item() == 4
    assert route["stop_now"].item()


def test_routing_bias_persists_and_old_state_dict_loads(model) -> None:
    _balanced_model(model)
    model.routing_bias.copy_(torch.tensor([0.1, -0.2, 0.3, -0.4]))
    restored = IAGSRME(model.backbone, model.config)
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored.routing_bias, model.routing_bias)

    old_state = model.state_dict()
    old_state.pop("routing_bias")
    restored.routing_bias.fill_(1.0)
    restored.load_state_dict(old_state)
    assert restored.routing_bias.eq(0).all()


def test_objective_uses_raw_scores_not_selection_scores(model, features) -> None:
    _balanced_model(model, stop_enabled=False)
    model.routing_bias.copy_(torch.tensor([0.0, 10.0, 0.0, 0.0]))
    initial, tokens, text, mask = features
    output = model.forward_from_features(initial, tokens, text, mask)
    raw = [step["scores"].clone() for step in output["steps"]]
    objective = IAGSRMEObjective(ObjectiveConfig(terminal_weight=0.0, lambda_pair=1.0, lambda_gain=1.0))
    targets = torch.randn(3, 12)
    before = objective(output, targets, ["a", "b", "c"])
    for step in output["steps"]:
        step["selection_scores"].add_(1000.0)
    after = objective(output, targets, ["a", "b", "c"])

    for score, step in zip(raw, output["steps"], strict=True):
        torch.testing.assert_close(step["scores"], score)
    torch.testing.assert_close(before["pair"], after["pair"])
    torch.testing.assert_close(before["gain"], after["gain"])


def test_repeated_raw_monopoly_is_counteracted(model) -> None:
    _balanced_model(model, stop_enabled=False, loss_free_bias_update_rate=0.001)
    model.train()
    raw = torch.tensor([[1.0, 0.9995, 0.0, 0.0]])
    routed = model.route_scores(raw)["routed_idx"]
    model.update_routing_bias(_output(raw, routed))

    assert model.route_scores(raw)["routed_idx"].item() == 1
