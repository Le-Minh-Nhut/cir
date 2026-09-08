from __future__ import annotations

from dataclasses import replace

import torch

from losses.objective import IAGSRMEObjective, ObjectiveConfig
from models.iag_srme.utils.retrieval import candidate_safety_loss


def _gradient_sum(module) -> float:
    return sum(
        float(parameter.grad.detach().abs().sum())
        for parameter in module.parameters()
        if parameter.grad is not None
    )


def _one_row_safety_inputs():
    current = torch.tensor([[1.0, 0.0]], requires_grad=True)
    # Slot 0 is the selected, safe sibling. Slot 1 is harmful and unselected.
    candidates = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True
    )
    targets = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]], requires_grad=True
    )
    positive = torch.tensor([[True, False]])
    negative = torch.tensor([[False, True]])
    return current, candidates, targets, positive, negative


def test_safe_loss_trains_harmful_unselected_candidate_only() -> None:
    current, candidates, targets, positive, negative = _one_row_safety_inputs()
    selected_idx = torch.tensor([0])
    result = candidate_safety_loss(
        current, candidates, targets, positive, negative, temperature=0.1
    )
    result["loss"].backward()

    assert selected_idx.item() == 0
    assert candidates.grad is not None
    assert candidates.grad[0, 1].abs().sum() > 0  # harmful, unselected sibling
    assert torch.equal(candidates.grad[0, 0], torch.zeros_like(candidates.grad[0, 0]))
    assert current.grad is None  # parent threshold is stop-gradient
    assert targets.grad is None  # target bank is supervision-only
    assert result["candidate_count"] == 2
    assert result["harmful_candidate_count"] == 1
    assert result["positive_candidate_count"] == 0


def test_safe_loss_filters_invalid_rows_before_retrieval_math() -> None:
    current = torch.randn(2, 3, requires_grad=True)
    candidates = torch.randn(2, 4, 3, requires_grad=True)
    targets = torch.randn(2, 3, requires_grad=True)
    # Row 0 has no negative; row 1 has no positive. Neither may enter logsumexp.
    positive = torch.tensor([[True, True], [False, False]])
    negative = torch.tensor([[False, False], [True, True]])
    result = candidate_safety_loss(
        current, candidates, targets, positive, negative, temperature=0.1
    )

    assert result["candidate_count"] == 0
    assert result["valid_parent_count"] == 0
    assert result["loss"].requires_grad
    assert torch.isfinite(result["loss"])
    assert result["loss"] == 0
    result["loss"].backward()
    assert candidates.grad is not None
    assert torch.equal(candidates.grad, torch.zeros_like(candidates.grad))
    assert current.grad is None
    assert targets.grad is None


def test_safe_loss_normalizes_only_valid_rows_times_all_candidates() -> None:
    torch.manual_seed(41)
    current = torch.randn(2, 5, requires_grad=True)
    candidates = torch.randn(2, 4, 5, requires_grad=True)
    targets = torch.randn(2, 5, requires_grad=True)
    positive = torch.tensor([[True, False], [False, False]])
    negative = torch.tensor([[False, True], [True, True]])
    result = candidate_safety_loss(
        current, candidates, targets, positive, negative, temperature=0.2
    )

    assert result["valid_parent_count"] == 1
    assert result["candidate_count"] == 4
    assert torch.allclose(result["loss"], result["numerator"] / 4)
    result["loss"].backward()
    assert candidates.grad is not None
    assert torch.equal(candidates.grad[1], torch.zeros_like(candidates.grad[1]))
    assert targets.grad is None


def test_safe_objective_routes_to_all_candidate_modules_not_scorenet(
    model, features
) -> None:
    model.config = replace(model.config, max_steps=1, stop_enabled=False)
    torch.nn.init.normal_(model.executor.action_mlp[-1].weight, std=0.03)
    torch.nn.init.normal_(model.executor.state_up.weight, std=0.03)
    output = model.forward_from_features(*features)
    # Each parent query is its own positive target. Candidate degradations therefore
    # provide a direct task gradient without routing through hard selection.
    targets = output["steps"][0]["current_query"].detach().clone()
    objective = IAGSRMEObjective(
        ObjectiveConfig(
            terminal_weight=0.0,
            lambda_pair=0.0,
            lambda_gain=0.0,
            lambda_safe=1.0,
        )
    )
    components = objective(output, targets, ["a", "b", "c"])
    assert components["safe_candidate_count"] == 12
    assert components["safe_raw"] > 0
    components["total"].backward()

    for module in (model.proposal, model.grounder, model.action_fusion, model.executor):
        assert _gradient_sum(module) > 0
    assert _gradient_sum(model.score_net) == 0


def test_safe_loss_reaches_trainable_text_through_frozen_visual_readouts(model) -> None:
    model.config = replace(model.config, max_steps=1, stop_enabled=False)
    torch.nn.init.normal_(model.executor.action_mlp[-1].weight, std=0.03)
    torch.nn.init.normal_(model.executor.state_up.weight, std=0.03)
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(False)
    model.backbone.text_embedding.weight.requires_grad_(True)

    images = torch.randn(3, 3, 3, 3)
    input_ids = torch.randint(0, 32, (3, 6))
    attention_mask = torch.ones(3, 6, dtype=torch.long)
    content_mask = torch.ones(3, 6, dtype=torch.bool)
    output = model(images, input_ids, attention_mask, content_mask)
    targets = output["steps"][0]["current_query"].detach().clone()
    objective = IAGSRMEObjective(
        ObjectiveConfig(
            terminal_weight=0.0,
            lambda_pair=0.0,
            lambda_gain=0.0,
            lambda_safe=1.0,
        )
    )
    components = objective(output, targets, ["a", "b", "c"])
    assert components["safe_raw"] > 0
    components["total"].backward()

    assert model.backbone.text_embedding.weight.grad is not None
    assert model.backbone.text_embedding.weight.grad.abs().sum() > 0
    for name in (
        "image_projection",
        "global_projection",
        "dense_projection",
        "retrieval_projection",
    ):
        assert _gradient_sum(getattr(model.backbone, name)) == 0
    for module in (model.proposal, model.grounder, model.action_fusion, model.executor):
        assert _gradient_sum(module) > 0


def test_safe_loss_includes_rows_where_live_policy_stops(model, features) -> None:
    model.config = replace(model.config, max_steps=3, stop_enabled=True, epsilon_stop=0.0)
    with torch.no_grad():
        model.score_net.net[-1].weight.zero_()
        model.score_net.net[-1].bias.fill_(-1.0)
    output = model.forward_from_features(*features)
    assert len(output["steps"]) == 1
    assert output["steps"][0]["stopped_now"].all()

    targets = output["steps"][0]["current_query"].detach().clone()
    objective = IAGSRMEObjective(
        ObjectiveConfig(
            terminal_weight=0.0,
            lambda_pair=0.0,
            lambda_gain=0.0,
            lambda_safe=1.0,
        )
    )
    components = objective(output, targets, ["a", "b", "c"])
    assert components["safe_valid_parent_count"] == 3
    assert components["safe_candidate_count"] == 12


def test_lambda_safe_zero_preserves_existing_total(model, features, monkeypatch) -> None:
    output = model.forward_from_features(*features)
    targets = torch.randn(3, 12)

    def unexpected_safe_call(*args, **kwargs):
        raise AssertionError("disabled L_safe must not add retrieval computation")

    monkeypatch.setattr("losses.objective.candidate_safety_loss", unexpected_safe_call)
    config = ObjectiveConfig(lambda_safe=0.0)
    components = IAGSRMEObjective(config)(output, targets, ["a", "b", "c"])
    previous_total = (
        config.terminal_weight * components["terminal"]
        + config.lambda_pair * components["pair"]
        + config.lambda_gain * components["gain"]
        + config.lambda_c * components["concept_loss"]
        + config.lambda_bind * components["bind_loss"]
        + config.lambda_rel * components["rel_ortho_loss"]
        + components["dpp_weighted"]
    )

    assert torch.equal(components["total"], previous_total)
    assert components["safe_raw"] == 0
    assert components["safe_weighted"] == 0
    assert components["safe_candidate_count"] == 0
