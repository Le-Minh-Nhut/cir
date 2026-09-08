from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from hydra import compose, initialize_config_dir
import torch

from losses.objective import IAGSRMEObjective, ObjectiveConfig
from models.iag_srme.utils.retrieval import candidate_safety_loss


ROOT = Path(__file__).resolve().parents[1]


def _gradient_sum(module) -> float:
    return sum(
        float(parameter.grad.detach().abs().sum())
        for parameter in module.parameters()
        if parameter.grad is not None
    )


def test_q1_hydra_override_preserves_text_only_strong_contract() -> None:
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "conf")):
        cfg = compose(
            config_name="config",
            overrides=[
                "backbone=fgclip_base_text_native_cls",
                "objective.lambda_safe=0.1",
            ],
        )

    assert cfg.backbone.train_vision is False
    assert cfg.backbone.train_text is True
    assert cfg.backbone.train_text_projection is False
    assert cfg.backbone.finetune_policy == "text_only"
    assert cfg.backbone.experiment_identity == "R0-NCLS-TEXT"
    assert cfg.objective.terminal_weight == 1.0
    assert cfg.objective.lambda_pair == 0.5
    assert cfg.objective.lambda_gain == 0.5
    assert cfg.objective.lambda_c == 0.6
    assert cfg.objective.lambda_bind == 1.0
    assert cfg.objective.lambda_rel == 0.6
    assert cfg.objective.lambda_dpp == 0.6
    assert cfg.objective.lambda_safe == 0.1
    objective_config = ObjectiveConfig(
        **{key: value for key, value in cfg.objective.items() if key != "name"}
    )
    assert objective_config.lambda_safe == 0.1


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


def test_safe_loss_reports_exact_detached_per_slot_quality_counts() -> None:
    current = torch.tensor([[0.8, 0.2]], requires_grad=True)
    candidates = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.9, 0.1]]],
        requires_grad=True,
    )
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    positive = torch.tensor([[True, False]])
    negative = torch.tensor([[False, True]])

    result = candidate_safety_loss(
        current, candidates, targets, positive, negative, temperature=0.1
    )

    assert result["candidate_count_per_slot"].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert result["harmful_candidate_count_per_slot"].tolist() == [0.0, 1.0, 1.0, 0.0]
    assert result["positive_candidate_count_per_slot"].tolist() == [1.0, 0.0, 0.0, 1.0]
    for name in (
        "candidate_count_per_slot",
        "harmful_candidate_count_per_slot",
        "positive_candidate_count_per_slot",
    ):
        assert not result[name].requires_grad


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
    assert torch.equal(result["candidate_count_per_slot"], torch.zeros(4))
    assert torch.equal(result["harmful_candidate_count_per_slot"], torch.zeros(4))
    assert torch.equal(result["positive_candidate_count_per_slot"], torch.zeros(4))
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


def _two_step_safety_output() -> tuple[dict[str, object], list[torch.Tensor]]:
    current_t0 = torch.tensor([[0.8, 0.2], [0.2, 0.8]])
    candidates_t0 = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.9, 0.1]],
            [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [0.1, 0.9]],
        ],
        requires_grad=True,
    )
    current_t1 = torch.tensor([[0.8, 0.2]])
    candidates_t1 = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.9, 0.1]]],
        requires_grad=True,
    )

    def step(
        live_indices: torch.Tensor,
        current: torch.Tensor,
        candidates: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        delta = candidates - current[:, None]
        return {
            "live_indices": live_indices,
            "current_query": current,
            "candidate_queries": candidates,
            "scores": torch.zeros(candidates.shape[:2]),
            "delta_q": delta,
            "stopped_now": torch.zeros(candidates.shape[0], dtype=torch.bool),
            "selected_idx": torch.zeros(candidates.shape[0], dtype=torch.long),
        }

    return (
        {
            "query": current_t0,
            "steps": [
                step(torch.tensor([0, 1]), current_t0, candidates_t0),
                step(torch.tensor([0]), current_t1, candidates_t1),
            ],
        },
        [candidates_t0, candidates_t1],
    )


def test_objective_aggregates_detached_per_slot_metrics_across_live_timesteps() -> None:
    output, candidate_tensors = _two_step_safety_output()
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    objective = IAGSRMEObjective(
        ObjectiveConfig(
            terminal_weight=0.0,
            lambda_pair=0.0,
            lambda_gain=0.0,
            lambda_safe=1.0,
        )
    )
    components = objective(output, targets, ["a", "b"])

    expected_harmful = [0.0, 1.0, 1.0, 0.0]
    expected_positive = [1.0, 0.0, 0.0, 1.0]
    all_delta = torch.cat(
        [step["delta_q"].detach().norm(dim=-1) for step in output["steps"]]
    )
    for slot in range(4):
        assert components[f"safe_candidate_count_c{slot}"] == 3
        assert components[f"safe_harmful_candidate_fraction_c{slot}"] == expected_harmful[slot]
        assert components[f"safe_positive_candidate_fraction_c{slot}"] == expected_positive[slot]
        assert torch.allclose(
            components[f"mean_delta_q_norm_c{slot}"], all_delta[:, slot].mean()
        )
        for prefix in (
            "safe_candidate_count",
            "safe_harmful_candidate_count",
            "safe_positive_candidate_count",
            "safe_harmful_candidate_fraction",
            "safe_positive_candidate_fraction",
            "mean_delta_q_norm",
        ):
            metric = components[f"{prefix}_c{slot}"]
            assert metric.ndim == 0
            assert not metric.requires_grad

    components["total"].backward()
    assert all(candidate.grad is not None for candidate in candidate_tensors)
    assert targets.grad is None or targets.grad.count_nonzero() == 0


def test_inspecting_detached_per_slot_metrics_does_not_change_safety_gradients() -> None:
    objective = IAGSRMEObjective(
        ObjectiveConfig(
            terminal_weight=0.0,
            lambda_pair=0.0,
            lambda_gain=0.0,
            lambda_safe=1.0,
        )
    )
    first_output, first_candidates = _two_step_safety_output()
    first = objective(first_output, torch.eye(2), ["a", "b"])
    inspected = {
        name: float(value.detach())
        for name, value in first.items()
        if name.startswith("safe_") or name.startswith("mean_delta_q_norm_c")
    }
    assert inspected
    first_gradients = torch.autograd.grad(first["total"], first_candidates)

    second_output, second_candidates = _two_step_safety_output()
    second = objective(second_output, torch.eye(2), ["a", "b"])
    second_gradients = torch.autograd.grad(second["total"], second_candidates)

    for inspected_gradient, ignored_gradient in zip(
        first_gradients, second_gradients, strict=True
    ):
        assert torch.equal(inspected_gradient, ignored_gradient)


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
    all_delta = torch.cat(
        [step["delta_q"].detach().float().norm(dim=-1) for step in output["steps"]]
    )
    for slot in range(model.config.num_candidates):
        assert components[f"safe_candidate_count_c{slot}"] == 0
        assert components[f"safe_harmful_candidate_fraction_c{slot}"] == 0
        assert components[f"safe_positive_candidate_fraction_c{slot}"] == 0
        assert torch.allclose(
            components[f"mean_delta_q_norm_c{slot}"], all_delta[:, slot].mean()
        )
        assert not components[f"mean_delta_q_norm_c{slot}"].requires_grad
