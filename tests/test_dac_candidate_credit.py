from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest
import torch

import losses.objective as objective_module
from losses.dac import (
    dac_groups,
    dac_responsibility_weights,
    dac_stage,
    validate_dac_candidate_count,
)
from losses.objective import IAGSRMEObjective, ObjectiveConfig
from diagnose_iag_srme import build_health_flags


@pytest.mark.parametrize(
    ("stage", "expected"),
    (
        (0, ((0, 1, 2, 3, 4, 5, 6, 7),)),
        (1, ((0, 1, 2, 3), (4, 5, 6, 7))),
        (2, ((0, 1), (2, 3), (4, 5), (6, 7))),
        (3, ((0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,))),
    ),
)
def test_k8_hierarchy_matches_dac_binary_partition(
    stage: int, expected: tuple[tuple[int, ...], ...]
) -> None:
    assert dac_groups(8, stage) == expected


@pytest.mark.parametrize(
    ("global_step", "expected_stage"),
    (
        (0, 0),
        (1999, 0),
        (2000, 1),
        (3999, 1),
        (4000, 2),
        (5999, 2),
        (6000, 3),
        (12345, 3),
    ),
)
def test_dac_schedule_uses_exact_optimizer_step_boundaries(
    global_step: int, expected_stage: int
) -> None:
    assert dac_stage(global_step, split_interval_steps=2000, num_candidates=8) == expected_stage


def test_dac_rejects_non_power_of_two_candidate_count() -> None:
    with pytest.raises(ValueError, match="power of two"):
        validate_dac_candidate_count(6)


def test_dac_rejects_invalid_schedule_inputs() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        dac_stage(-1, split_interval_steps=2000, num_candidates=8)
    with pytest.raises(ValueError, match="positive"):
        dac_stage(0, split_interval_steps=0, num_candidates=8)


@pytest.mark.parametrize(
    ("stage", "losses", "expected_gradient", "expected_group"),
    (
        (
            0,
            [4.0, 3.0, 2.0, 1.0, 8.0, 7.0, 6.0, 5.0],
            [0.125] * 8,
            0,
        ),
        (
            1,
            [4.0, 3.0, 2.0, 1.0, 8.0, 7.0, 6.0, 5.0],
            [0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0],
            0,
        ),
        (
            2,
            [4.0, 3.0, 2.0, 1.0, 8.0, 7.0, 6.0, 5.0],
            [0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
            1,
        ),
        (
            3,
            [4.0, 3.0, 2.0, 1.0, 8.0, 7.0, 6.0, 5.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            3,
        ),
    ),
)
def test_dac_routes_exact_gradient_to_every_winning_group_member(
    stage: int,
    losses: list[float],
    expected_gradient: list[float],
    expected_group: int,
) -> None:
    candidate_losses = torch.tensor([losses], requires_grad=True)

    weights, winning_groups = dac_responsibility_weights(candidate_losses, stage)
    row_loss = (weights * candidate_losses).sum()
    row_loss.backward()

    assert winning_groups.tolist() == [expected_group]
    assert torch.equal(candidate_losses.grad, torch.tensor([expected_gradient]))


def test_dac_winning_group_loss_is_mean_not_argmin_loss() -> None:
    losses = torch.tensor(
        [[4.0, 3.0, 2.0, 1.0, 8.0, 7.0, 6.0, 5.0]], requires_grad=True
    )
    weights, _ = dac_responsibility_weights(losses, stage=1)

    value = (weights * losses).sum()

    assert value == torch.tensor(2.5)
    assert value != losses[0, 3]


def test_dac_selects_right_hand_group_for_candidate_six_winner() -> None:
    losses = torch.tensor([[9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 1.0, 3.0]])

    weights, winning_groups = dac_responsibility_weights(losses, stage=1)

    assert winning_groups.tolist() == [1]
    assert torch.equal(weights, torch.tensor([[0.0] * 4 + [0.25] * 4]))


def test_dac_routes_different_batch_rows_to_different_groups() -> None:
    losses = torch.tensor(
        [
            [4.0, 3.0, 2.0, 1.0, 8.0, 7.0, 6.0, 5.0],
            [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 1.0, 3.0],
        ]
    )

    weights, winning_groups = dac_responsibility_weights(losses, stage=1)

    assert winning_groups.tolist() == [0, 1]
    assert torch.equal(
        weights,
        torch.tensor([[0.25] * 4 + [0.0] * 4, [0.0] * 4 + [0.25] * 4]),
    )


def test_dac_assignment_is_stop_gradient() -> None:
    losses = torch.tensor(
        [[4.0, 3.0, 2.0, 1.0, 8.0, 7.0, 6.0, 5.0]], requires_grad=True
    )

    weights, winning_groups = dac_responsibility_weights(losses, stage=1)
    (weights * losses).sum().backward()

    assert not weights.requires_grad
    assert not winning_groups.requires_grad
    assert torch.equal(
        losses.grad, torch.tensor([[0.25] * 4 + [0.0] * 4])
    )


def test_dac_leaf_matches_hard_wta_numerically_and_in_gradient() -> None:
    dac_losses = torch.tensor(
        [[3.0, 1.0, 4.0, 2.0, 8.0, 7.0, 6.0, 5.0]], requires_grad=True
    )
    hard_losses = dac_losses.detach().clone().requires_grad_(True)

    dac_weights, _ = dac_responsibility_weights(dac_losses, stage=3)
    hard_weights = torch.nn.functional.one_hot(
        hard_losses.detach().argmin(dim=-1), num_classes=8
    ).float()
    dac_value = (dac_weights * dac_losses).sum()
    hard_value = (hard_weights * hard_losses).sum()
    dac_value.backward()
    hard_value.backward()

    assert torch.equal(dac_value, hard_value)
    assert torch.equal(dac_weights, hard_weights)
    assert torch.equal(dac_losses.grad, hard_losses.grad)


def _candidate_credit_output(candidate_queries: torch.Tensor) -> dict[str, object]:
    batch, candidates, dimension = candidate_queries.shape
    query = torch.randn(batch, dimension, requires_grad=True)
    return {
        "query": query,
        "steps": [
            {
                "live_indices": torch.arange(batch),
                "current_query": torch.randn(batch, dimension),
                "candidate_queries": candidate_queries,
                "scores": torch.randn(batch, candidates, requires_grad=True),
                "delta_q": torch.randn(batch, candidates, dimension),
                "stopped_now": torch.zeros(batch, dtype=torch.bool),
                "selected_idx": torch.zeros(batch, dtype=torch.long),
            }
        ],
    }


def _patch_candidate_losses_from_first_coordinate(monkeypatch: pytest.MonkeyPatch) -> None:
    def valid_teacher(current, candidates, targets, positive, negative, temperature):
        del current, targets, positive, negative, temperature
        return (
            candidates.new_zeros(candidates.shape[:-1]),
            torch.ones(candidates.shape[0], dtype=torch.bool),
        )

    def first_coordinate_losses(candidates, targets, positive, negative, temperature):
        del targets, positive, negative, temperature
        return candidates[..., 0]

    monkeypatch.setattr(objective_module, "marginal_teacher_utilities", valid_teacher)
    monkeypatch.setattr(
        objective_module, "teacher_retrieval_loss", first_coordinate_losses
    )


def test_dac_objective_uses_global_step_and_routes_rows_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_candidate_losses_from_first_coordinate(monkeypatch)
    prescribed = torch.tensor(
        [
            [[4.0], [3.0], [2.0], [1.0], [8.0], [7.0], [6.0], [5.0]],
            [[9.0], [8.0], [7.0], [6.0], [5.0], [4.0], [1.0], [3.0]],
        ],
        requires_grad=True,
    )
    objective = IAGSRMEObjective(
        ObjectiveConfig(
            terminal_weight=0.0,
            lambda_pair=0.0,
            lambda_gain=0.0,
            candidate_credit_mode="dac",
            lambda_candidate_credit=1.0,
            dac_split_interval_steps=2000,
        )
    )

    components = objective(
        _candidate_credit_output(prescribed),
        torch.randn(2, 1),
        ["left", "right"],
        global_step=2000,
    )
    components["candidate_credit_loss"].backward()

    assert torch.equal(components["dac_stage"], torch.tensor(1.0))
    assert torch.equal(components["dac_num_groups"], torch.tensor(2.0))
    assert torch.equal(components["dac_group_size"], torch.tensor(4.0))
    assert torch.equal(components["dac_split_interval_steps"], torch.tensor(2000.0))
    assert torch.equal(components["dac_gradient_candidate_fraction"], torch.tensor(0.5))
    assert torch.equal(components["dac_group_winning_frequency_0"], torch.tensor(0.5))
    assert torch.equal(components["dac_group_winning_frequency_1"], torch.tensor(0.5))
    for candidate in range(8):
        assert torch.equal(
            components[f"dac_responsibility_frequency_{candidate}"],
            torch.tensor(0.5),
        )
    assert torch.equal(
        prescribed.grad[..., 0],
        torch.tensor(
            [[0.125] * 4 + [0.0] * 4, [0.0] * 4 + [0.125] * 4]
        ),
    )


def test_dac_objective_skips_invalid_rows_with_finite_zero_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def invalid_teacher(current, candidates, targets, positive, negative, temperature):
        del current, targets, positive, negative, temperature
        return (
            candidates.new_zeros(candidates.shape[:-1]),
            torch.zeros(candidates.shape[0], dtype=torch.bool),
        )

    monkeypatch.setattr(objective_module, "marginal_teacher_utilities", invalid_teacher)
    candidates = torch.randn(2, 8, 3, requires_grad=True)
    objective = IAGSRMEObjective(
        ObjectiveConfig(
            terminal_weight=0.0,
            lambda_pair=0.0,
            lambda_gain=0.0,
            candidate_credit_mode="dac",
            lambda_candidate_credit=1.0,
        )
    )

    components = objective(
        _candidate_credit_output(candidates),
        torch.randn(2, 3),
        ["same", "same"],
        global_step=4000,
    )

    for name in (
        "candidate_credit_loss",
        "candidate_credit_weighted",
        "dac_responsibility_concentration",
        "dac_group_winning_frequency_0",
        "dac_responsibility_frequency_0",
        "total",
    ):
        assert torch.isfinite(components[name])
        assert components[name] == 0
    assert components["teacher_invalid_rows"] == 2
    assert components["dac_stage"] == 2


def test_dac_credit_detaches_targets_and_reaches_four_candidate_queries(
    model, features
) -> None:
    model = model.__class__(
        model.backbone,
        replace(model.config, num_candidates=8, max_steps=1),
    )
    torch.nn.init.normal_(model.executor.action_mlp[-1].weight, std=0.03)
    torch.nn.init.normal_(model.executor.state_up.weight, std=0.03)
    output = model.forward_from_features(*features)
    candidate_queries = output["steps"][0]["candidate_queries"]
    candidate_queries.retain_grad()
    targets = torch.randn(3, 12, requires_grad=True)
    objective = IAGSRMEObjective(
        ObjectiveConfig(
            terminal_weight=0.0,
            lambda_pair=0.0,
            lambda_gain=0.0,
            candidate_credit_mode="dac",
            lambda_candidate_credit=1.0,
        )
    )

    components = objective(
        output,
        targets,
        ["a", "b", "c"],
        global_step=2000,
    )
    components["total"].backward()

    nonzero_by_candidate = candidate_queries.grad.abs().sum(dim=-1) > 0
    assert torch.equal(
        nonzero_by_candidate.sum(dim=-1), torch.full((3,), 4, dtype=torch.long)
    )
    assert targets.grad is None or targets.grad.count_nonzero() == 0


def test_k8_model_components_and_hard_rollout_shapes(model, features) -> None:
    model = model.__class__(
        model.backbone,
        replace(model.config, num_candidates=8, max_steps=1),
    )
    torch.nn.init.normal_(model.executor.state_up.weight, std=0.03)

    output = model.forward_from_features(*features)
    step = output["steps"][0]

    assert step["proposals"].shape == (3, 8, 16)
    assert step["grounding"].shape == (3, 8, 9)
    assert step["alpha_read"].shape == (3, 8, 9)
    assert step["exec_mask"].shape == (3, 8, 9)
    assert step["candidate_states"].shape == (3, 8, 9, 16)
    assert step["candidate_queries"].shape == (3, 8, 12)
    assert step["scores"].shape == (3, 8)
    assert step["selected_idx"].shape == (3,)
    assert torch.all(step["selected_idx"] < 8)
    assert "target" not in inspect.signature(model.forward).parameters


def _compose_k8(objective: str):
    config_dir = str(Path(__file__).parents[1] / "conf")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(
            config_name="config",
            overrides=["model=iag_srme_k8", f"objective={objective}"],
        )


@pytest.mark.parametrize(
    ("objective_name", "expected_mode"),
    (
        ("core_dac", "dac"),
        ("core_awta", "awta"),
        ("core_hard_wta", "hard_wta"),
    ),
)
def test_matched_k8_candidate_credit_configs_compose(
    objective_name: str, expected_mode: str
) -> None:
    cfg = _compose_k8(objective_name)

    assert cfg.model.num_candidates == 8
    assert cfg.objective.candidate_credit_mode == expected_mode
    assert cfg.objective.lambda_candidate_credit == 1.0
    if expected_mode == "dac":
        assert cfg.objective.dac_split_interval_steps == 2000


def test_dac_no_dpp_config_only_disables_dpp_terms() -> None:
    regular = _compose_k8("core_dac")
    isolated = _compose_k8("core_dac_no_dpp")
    regular_objective = OmegaConf.to_container(regular.objective, resolve=True)
    isolated_objective = OmegaConf.to_container(isolated.objective, resolve=True)

    assert regular_objective.pop("dpp_enabled") is True
    assert isolated_objective.pop("dpp_enabled") is False
    assert regular_objective.pop("lambda_dpp") == 0.6
    assert isolated_objective.pop("lambda_dpp") == 0.0
    regular_objective.pop("name")
    isolated_objective.pop("name")
    assert regular_objective == isolated_objective


def test_stage_zero_shared_responsibility_suppresses_similarity_collapse_flags() -> None:
    metrics = {
        "objective/dac_num_groups": 1.0,
        "objective/dac_stage": 0.0,
        "objective/functional_pairwise_cosine": 0.999,
        "objective/functional_rank": 1.0,
        "step/proposal/pairwise_cosine_mean": 0.999,
        "step/action/pairwise_cosine_mean": 0.999,
    }

    flags = build_health_flags(metrics, {}, {"differences": {}})

    codes = {flag["code"] for flag in flags}
    assert "FUNCTIONAL_CANDIDATE_COLLAPSE" not in codes
    assert "FUNCTIONAL_CANDIDATES_TOO_SIMILAR" not in codes
    assert "PROPOSAL_COLLAPSE" not in codes
    assert "ACTION_COLLAPSE" not in codes


def test_post_split_similarity_still_triggers_collapse_diagnostic() -> None:
    metrics = {
        "objective/dac_num_groups": 2.0,
        "objective/dac_stage": 1.0,
        "objective/functional_pairwise_cosine": 0.999,
        "objective/functional_rank": 1.0,
    }

    flags = build_health_flags(metrics, {}, {"differences": {}})

    assert "FUNCTIONAL_CANDIDATE_COLLAPSE" in {flag["code"] for flag in flags}
