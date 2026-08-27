from __future__ import annotations

from pathlib import Path
import sys
import types

import pytest
import torch
from torch import Tensor, nn

from models.functional_ownership import (
    block_residual_credit,
    build_conditional_residual_plan,
    coalition_index,
    coalition_masks,
    conditional_phi,
    functional_credit_loss,
    functional_mode_assignment,
    pair_synergy_credit,
    pairwise_error_modes,
    slot_pairs,
)
from models.taper import TAPER


class DummyTeacher(nn.Module):
    def __init__(self, query_dim: int) -> None:
        super().__init__()
        self.query_dim = query_dim
        self.scale = nn.Parameter(torch.ones(query_dim))

    def compose(
        self,
        reference_features: Tensor,
        text_states: Tensor,
        attention_mask: Tensor,
        *,
        normalize: bool = False,
    ) -> Tensor:
        mask = attention_mask.to(text_states.dtype).unsqueeze(-1)
        pooled = (text_states * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        reference_scale = 1.0 + 0.01 * reference_features.mean(
            dim=(1, 2),
            keepdim=False,
        ).unsqueeze(1)
        query = pooled[:, : self.query_dim] * reference_scale * self.scale
        if normalize:
            query = torch.nn.functional.normalize(query, dim=-1)
        return query


def make_model(
    *,
    enabled: bool,
    pair_lookahead: bool = True,
    rank_threshold: float = 0.25,
    credit_schedule: str = "first_round",
) -> TAPER:
    torch.manual_seed(17)
    return TAPER(
        DummyTeacher(query_dim=4),
        text_dim=5,
        reference_dim=7,
        teacher_text_dim=5,
        teacher_query_dim=4,
        query_dim=4,
        slot_dim=4,
        state_dim=4,
        num_slots=3,
        num_primitives=2,
        retrieval_temperature=0.07,
        counterfactual_chunk_size=3,
        slot_value_source="contextual",
        slot_effect_in_value=False,
        slot_value_assignment="soft_shared",
        functional_ownership_enabled=enabled,
        functional_num_hard_negatives=2,
        functional_lambda=0.1,
        functional_margin=0.0,
        functional_temperature=0.07,
        functional_pair_lookahead=pair_lookahead,
        functional_rank_threshold=rank_threshold,
        functional_credit_schedule=credit_schedule,
    )


def make_batch(batch_size: int = 4) -> dict[str, object]:
    torch.manual_seed(23)
    num_tokens = 5
    attention = torch.ones(batch_size, num_tokens, dtype=torch.bool)
    content = attention.clone()
    content[:, 0] = False
    content[:, -1] = False
    return {
        "reference_features": torch.randn(batch_size, 7),
        "teacher_reference_features": torch.randn(batch_size, 2, 7),
        "text_states": torch.randn(batch_size, num_tokens, 5),
        "teacher_text_states": torch.randn(batch_size, num_tokens, 5),
        "text_attention_mask": attention,
        "text_content_mask": content,
        "target_features": torch.randn(batch_size, 1, 4),
        "target_ids": [f"target-{index}" for index in range(batch_size)],
    }


def test_exact_clone_loses_block_residual_credit() -> None:
    effects = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    result = block_residual_credit(
        effects,
        torch.ones(1, 3, dtype=torch.bool),
    )
    assert torch.equal(
        result["credited_mask"],
        torch.tensor([[True, False, True]]),
    )
    assert torch.equal(result["credit"][:, 1], torch.zeros(1, 2))
    assert result["credit"][:, 2].sum() > 0


def test_independent_two_mode_slots_receive_unique_credit() -> None:
    effects = torch.tensor([[[2.0, 0.0], [0.0, 3.0]]])
    result = block_residual_credit(
        effects,
        torch.ones(1, 2, dtype=torch.bool),
    )
    assert result["credited_mask"].all()
    torch.testing.assert_close(result["unique_mode_coverage"], torch.ones(1))
    torch.testing.assert_close(
        result["redundant_credit_fraction"],
        torch.zeros(1),
    )


def test_v1_residual_detects_giant_but_is_not_assignment_success() -> None:
    effects = torch.tensor([[[2.0, 2.0], [0.2, 0.2]]])
    result = block_residual_credit(
        effects,
        torch.ones(1, 2, dtype=torch.bool),
    )
    assert result["credited_mask"].sum().item() == 1
    assert result["credited_mask"][0, 0]
    assert result["unique_mode_coverage"].item() == 1.0


def assign_modes(effects: Tensor, rank: float) -> dict[str, Tensor]:
    return functional_mode_assignment(
        effects,
        torch.ones(effects.shape[:2], dtype=torch.bool),
        torch.tensor([rank], dtype=torch.float32),
        rank_gate_enabled=True,
        rank_threshold=0.25,
        mode_capacity=None,
        allow_unassigned_modes=True,
    )


def test_mode_assignment_exact_clone_and_independent_specialist() -> None:
    effects = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    result = assign_modes(effects, rank=2.0)
    assignment = result["assignment"][0]
    assert assignment[:, 0].sum().item() == 1
    assert assignment[:, 1].sum().item() == 1
    assert not (assignment[0, 0] and assignment[1, 0])
    assert assignment[2, 1]
    assert result["credited_mask"].sum().item() == 2
    assert result["unresolved_multimode"].item() is False


def test_mode_assignment_independent_two_mode_world() -> None:
    effects = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
    result = assign_modes(effects, rank=2.0)
    assert torch.equal(
        result["assignment"],
        torch.tensor([[[True, False], [False, True]]]),
    )
    assert result["inferred_k_eff"].item() == 2
    assert result["owned_mode_count"].item() == 2
    assert result["credited_mask"].sum().item() == 2
    assert result["unique_mode_coverage"].item() == 1.0


def test_mode_assignment_true_rank_one_allows_one_owner() -> None:
    effects = torch.tensor([[[1.0, 2.0], [2.0, 4.0], [0.5, 1.0]]])
    result = functional_mode_assignment(
        effects,
        torch.ones(1, 3, dtype=torch.bool),
        torch.tensor([1.1]),
        rank_gate_enabled=True,
        rank_threshold=0.25,
        mode_capacity=1,
        allow_unassigned_modes=True,
    )
    assert result["inferred_k_eff"].item() == 1
    assert result["credited_mask"].sum().item() == 1
    assert result["owned_mode_count"].item() == 2
    assert not result["unresolved_multimode"].item()


def test_multimode_giant_cannot_own_all_when_specialists_exist() -> None:
    effects = torch.tensor(
        [[[2.0, 2.0], [1.5, 0.0], [0.0, 1.5]]]
    )
    result = assign_modes(effects, rank=2.0)
    assignment = result["assignment"][0]
    assert assignment[0].sum().item() == 1
    assert assignment.any(dim=1).sum().item() == 2
    assert assignment[:, 0].sum().item() == 1
    assert assignment[:, 1].sum().item() == 1
    assert not result["giant_owner"].item()
    assert not result["unresolved_multimode"].item()


def test_multimode_giant_only_is_flagged_unresolved_without_fake_owner() -> None:
    effects = torch.tensor(
        [[[2.0, 2.0], [-1.0, 0.0], [0.0, -1.0]]]
    )
    result = assign_modes(effects, rank=2.0)
    assert result["credited_mask"].sum().item() == 1
    assert result["owned_mode_count"].item() == 1
    assert result["unowned_positive_mode_count"].item() == 1
    assert result["giant_owner"].item()
    assert result["unresolved_multimode"].item()


def test_identical_phi_rows_are_unresolved_not_fake_split() -> None:
    effects = torch.tensor([[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]])
    result = assign_modes(effects, rank=2.0)
    assert result["proposal_assignment"].any(dim=1).sum().item() == 2
    assert result["assignment"].any(dim=1).sum().item() == 1
    assert result["owned_mode_count"].item() == 2
    assert result["unique_mode_coverage"].item() == pytest.approx(0.5)
    assert result["credited_mask"].sum().item() == 1
    assert result["unowned_positive_mode_count"].item() == 0
    assert result["training_credit"][0, 1, 1].item() > 0.0
    assert result["credit"][0, 1].sum().item() == 0.0
    assert result["ownership_row_similarity"].item() == pytest.approx(1.0)
    assert result["unresolved_multimode"].item()

    objectives = torch.zeros_like(effects, requires_grad=True)
    functional_credit_loss(objectives, result["training_credit"]).backward()
    assert objectives.grad[0, 0, 0].item() > 0.0
    assert objectives.grad[0, 1, 1].item() > 0.0


def test_pair_lookahead_keeps_xor_synergy() -> None:
    empty = torch.tensor([[1.0, 1.0]])
    singletons = torch.tensor([[[1.0, 1.0], [1.0, 1.0]]])
    pair = torch.tensor([[[0.0, 0.0]]])
    pairs = torch.tensor([[0, 1]])
    result = pair_synergy_credit(
        empty,
        singletons,
        pair,
        pairs,
        torch.ones(1, 2, dtype=torch.bool),
    )
    assert result["credited_mask"].item()
    assert result["credit"].sum().item() == pytest.approx(2.0)
    assert result["synergy_fraction"].item() == 1.0


def test_pair_synergy_only_receives_currently_unsolved_modes() -> None:
    empty = torch.tensor([[1.0, 1.0]])
    singletons = torch.tensor([[[1.0, 1.0], [1.0, 1.0]]])
    pair = torch.tensor([[[0.0, 0.0]]])
    result = pair_synergy_credit(
        empty,
        singletons,
        pair,
        torch.tensor([[0, 1]]),
        torch.ones(1, 2, dtype=torch.bool),
        available_mode_mask=torch.tensor([[False, True]]),
    )
    assert result["credit"][0, 0, 0].item() == 0.0
    assert result["credit"][0, 0, 1].item() > 0.0


def test_positive_interaction_without_pair_improvement_gets_no_credit() -> None:
    empty = torch.tensor([[1.0]])
    singletons = torch.tensor([[[10.0], [10.0]]])
    pair = torch.tensor([[[15.0]]])
    result = pair_synergy_credit(
        empty,
        singletons,
        pair,
        torch.tensor([[0, 1]]),
        torch.ones(1, 2, dtype=torch.bool),
    )
    assert not result["credited_mask"].item()
    assert result["credit"].sum().item() == 0.0


def test_rank_one_task_allows_one_effective_credit_block() -> None:
    effects = torch.tensor(
        [[[1.0, 2.0], [2.0, 4.0], [0.5, 1.0]]]
    )
    result = block_residual_credit(
        effects,
        torch.ones(1, 3, dtype=torch.bool),
    )
    assert result["credited_mask"].sum().item() == 1


def test_orthogonal_junk_without_positive_improvement_gets_no_credit() -> None:
    effects = torch.tensor([[[1.0, 0.0], [-1.0, -3.0]]])
    result = assign_modes(effects, rank=2.0)
    assert result["credited_mask"][0, 0]
    assert not result["credited_mask"][0, 1]
    assert not result["assignment"][0, 1].any()


def test_functional_gradient_isolated_to_credited_blocks() -> None:
    losses = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]],
        requires_grad=True,
    )
    credit = torch.tensor(
        [[[2.0, 0.0], [0.0, 0.0], [0.0, 4.0]]]
    )
    loss = functional_credit_loss(losses, credit)
    loss.backward()
    assert torch.equal(losses.grad[:, 1], torch.zeros(1, 2))
    assert losses.grad[0, 0, 0] > 0
    assert losses.grad[0, 0, 1] == 0
    assert losses.grad[0, 2, 0] == 0
    assert losses.grad[0, 2, 1] > 0


def test_parameter_gradient_isolation_detaches_competitor_slot_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = make_model(enabled=True, pair_lookahead=False).train()
    batch = make_batch()

    def fixed_owner(
        effects: Tensor,
        candidate_mask: Tensor,
        task_rank: Tensor,
        **_kwargs,
    ) -> dict[str, Tensor]:
        del task_rank
        batch_size, num_slots, num_modes = effects.shape
        assignment = torch.zeros_like(effects, dtype=torch.bool)
        assignment[:, 0, 0] = True
        credit = torch.zeros_like(effects)
        credit[:, 0, 0] = 1.0
        credited = torch.zeros_like(candidate_mask)
        credited[:, 0] = True
        zeros = effects.new_zeros(batch_size)
        return {
            "assignment": assignment,
            "training_assignment": assignment.clone(),
            "proposal_assignment": assignment.clone(),
            "credit": credit,
            "training_credit": credit.clone(),
            "credited_mask": credited,
            "credit_order": torch.zeros(
                batch_size,
                num_slots,
                dtype=torch.long,
            ),
            "inferred_k_eff": torch.ones(batch_size, dtype=torch.long),
            "owned_mode_count": torch.ones(batch_size, dtype=torch.long),
            "unowned_positive_mode_count": torch.zeros(
                batch_size,
                dtype=torch.long,
            ),
            "max_modes_per_owner": torch.ones(batch_size, dtype=torch.long),
            "giant_owner": torch.zeros(batch_size, dtype=torch.bool),
            "ownership_row_similarity": zeros,
            "unresolved_multimode": torch.zeros(batch_size, dtype=torch.bool),
            "residual_active_modes": zeros,
            "unique_mode_coverage": zeros,
            "redundant_credit_fraction": zeros,
        }

    monkeypatch.setattr("models.taper.functional_mode_assignment", fixed_owner)
    output = model.forward(
        batch["reference_features"],
        batch["text_states"],
        batch["text_attention_mask"],
        text_content_mask=batch["text_content_mask"],
        teacher_reference_features=batch["teacher_reference_features"],
        teacher_text_states=batch["teacher_text_states"],
    )
    isolated_slots = model._functional_credit_isolated_edit_slots(
        output,
        batch["text_states"],
    )
    assert torch.equal(
        isolated_slots.detach(),
        output["edit_slots"].detach(),
    )

    losses = model.compute_loss(batch)
    model.zero_grad(set_to_none=True)
    losses["functional_loss"].backward()
    assert model.slot_queries.grad is not None
    assert model.slot_queries.grad[0].abs().sum().item() > 0.0
    torch.testing.assert_close(
        model.slot_queries.grad[1:],
        torch.zeros_like(model.slot_queries.grad[1:]),
        rtol=0.0,
        atol=1e-10,
    )
    for shared_parameter in (
        model.slot_query_projection.weight,
        model.text_key_projection.weight,
        model.slot_mlp[0].weight,
        model.router[0].weight,
        model.transition_delta[0].weight,
        model.query_head[0].weight,
    ):
        assert shared_parameter.grad is not None
        assert torch.isfinite(shared_parameter.grad).all()
        assert shared_parameter.grad.abs().sum().item() > 0.0


def test_pairwise_modes_remain_unaggregated() -> None:
    queries = torch.randn(2, 5, 4)
    candidates = torch.randn(2, 4, 1, 4)
    loss = pairwise_error_modes(
        queries,
        candidates,
        margin=0.0,
        temperature=0.07,
    )
    assert loss.shape == (2, 5, 3)
    assert torch.isfinite(loss).all()


def _plan(
    losses: Tensor,
    *,
    rank: float = 2.0,
    pair_lookahead: bool = True,
) -> dict[str, Tensor]:
    num_slots = (losses.shape[1]).bit_length() - 1
    return build_conditional_residual_plan(
        losses,
        torch.ones(losses.shape[0], num_slots, dtype=torch.bool),
        torch.full((losses.shape[0],), rank),
        pair_lookahead=pair_lookahead,
    )


def test_exact_conditional_clone_is_rejected_after_first_owner() -> None:
    losses = torch.tensor(
        [[
            [4.0, 4.0],  # EMPTY
            [2.0, 4.0],  # S0
            [2.0, 4.0],  # S1: E1 clone
            [2.0, 4.0],  # S0+S1: no conditional E1 gain
            [4.0, 2.0],  # S2
            [2.0, 2.0],
            [2.0, 2.0],
            [2.0, 2.0],
        ]]
    )
    plan = _plan(losses)
    assert torch.equal(
        plan["new_block_mask"][0, :2],
        torch.tensor([[True, False, False], [False, False, True]]),
    )
    assert plan["mode_credit"][0, 0, 0] > 0
    assert plan["mode_credit"][0, 1, 1] > 0
    assert not plan["new_block_mask"][0, :, 1].any()


def test_initial_assignment_can_lie_but_conditional_plan_rejects_job() -> None:
    losses = torch.tensor(
        [[
            [5.0, 5.0],
            [1.0, 5.0],  # S0 solves E1.
            [5.0, 2.0],  # S1 appears to solve E2 from EMPTY.
            [1.0, 5.0],  # After S0, S1 adds no E2 improvement.
            [5.0, 5.0],
            [1.0, 5.0],
            [5.0, 2.0],
            [1.0, 5.0],
        ]]
    )
    empty_effect = torch.stack(
        [losses[:, 0] - losses[:, 1 << slot] for slot in range(3)],
        dim=1,
    )
    initial = functional_mode_assignment(
        empty_effect,
        torch.ones(1, 3, dtype=torch.bool),
        torch.tensor([2.0]),
    )
    assert initial["training_assignment"][0, 1, 1]

    plan = _plan(losses, pair_lookahead=False)
    assert plan["accepted_mask"].sum() == 1
    assert plan["new_block_mask"][0, 0, 0]
    assert not plan["new_block_mask"][0, :, 1].any()
    assert plan["stop_no_gain"][0]


def test_conditional_plan_keeps_independent_specialists() -> None:
    losses = torch.tensor(
        [[
            [4.0, 4.0],
            [2.0, 4.0],
            [4.0, 2.0],
            [2.0, 2.0],
        ]]
    )
    plan = _plan(losses)
    assert plan["accepted_mask"].sum() == 2
    assert torch.equal(
        plan["new_block_mask"][0, :2],
        torch.tensor([[True, False], [False, True]]),
    )


def test_conditional_plan_allows_true_rank_one() -> None:
    losses = torch.tensor(
        [[
            [4.0, 4.0],
            [2.0, 2.0],
            [2.0, 2.0],
            [2.0, 2.0],
        ]]
    )
    plan = _plan(losses, rank=1.0)
    assert plan["accepted_mask"].sum() == 1


def test_conditional_plan_reports_giant_only_residual_without_fake_split() -> None:
    losses = torch.tensor(
        [[
            [5.0, 5.0],
            [1.0, 1.0],
            [5.0, 5.0],
            [1.0, 1.0],
            [5.0, 5.0],
            [1.0, 1.0],
            [5.0, 5.0],
            [1.0, 1.0],
        ]]
    )
    plan = _plan(losses, pair_lookahead=False)
    assert plan["accepted_mask"].sum() == 1
    assert plan["new_block_mask"][0, 0, 0]
    assert plan["unresolved_modes"][0].any()
    assert plan["stop_no_gain"][0]


def test_conditional_pair_lookahead_preserves_xor_block() -> None:
    losses = torch.tensor(
        [[
            [5.0, 5.0],
            [5.0, 5.0],
            [5.0, 5.0],
            [1.0, 1.0],
        ]]
    )
    plan = _plan(losses)
    assert plan["accepted_mask"].sum() == 1
    assert torch.equal(
        plan["new_block_mask"][0, 0],
        torch.tensor([True, True]),
    )
    assert plan["mode_credit"][0, 0].gt(0).all()


def test_coalition_helpers_are_exact_and_deterministic() -> None:
    masks = coalition_masks(3, torch.device("cpu"))
    assert masks.shape == (8, 3)
    assert torch.equal(coalition_index(masks), torch.arange(8))
    losses = torch.arange(16, dtype=torch.float32).reshape(1, 8, 2)
    phi = conditional_phi(losses, torch.tensor([1]), slot_id=2)
    torch.testing.assert_close(phi, losses[:, 1] - losses[:, 5])


def test_later_conditional_step_gradient_reaches_only_new_slot_query() -> None:
    model = make_model(
        enabled=True,
        credit_schedule="conditional_residual",
    ).train()
    batch = make_batch()
    output = model.forward(
        batch["reference_features"],
        batch["text_states"],
        batch["text_attention_mask"],
        text_content_mask=batch["text_content_mask"],
        teacher_reference_features=batch["teacher_reference_features"],
        teacher_text_states=batch["teacher_text_states"],
    )
    isolated = model._functional_credit_isolated_edit_slots(
        output,
        batch["text_states"],
    )
    new_block = torch.zeros(4, 1, 3, dtype=torch.bool)
    new_block[:, 0, 1] = True
    queries = model._functional_conditional_step_queries(
        deployed_edit_slots=output["edit_slots"],
        isolated_edit_slots=isolated,
        new_block_mask=new_block,
        next_coalition_index=torch.full((4, 1), 3, dtype=torch.long),
        candidate_slots=torch.ones(4, 3, dtype=torch.bool),
        z0=output["z0"],
        reference_state=output["reference_state"],
    )
    model.zero_grad(set_to_none=True)
    queries.sum().backward()
    assert model.slot_queries.grad is not None
    assert model.slot_queries.grad[1].abs().sum() > 0
    torch.testing.assert_close(
        model.slot_queries.grad[[0, 2]],
        torch.zeros_like(model.slot_queries.grad[[0, 2]]),
        rtol=0.0,
        atol=1e-10,
    )
    assert model.slot_mlp[0].weight.grad is not None
    assert model.query_head[0].weight.grad is not None


def test_conditional_step_forward_matches_ordinary_forced_coalition() -> None:
    model = make_model(
        enabled=True,
        credit_schedule="conditional_residual",
    ).eval()
    batch = make_batch()
    output = model.forward(
        batch["reference_features"],
        batch["text_states"],
        batch["text_attention_mask"],
        text_content_mask=batch["text_content_mask"],
        teacher_reference_features=batch["teacher_reference_features"],
        teacher_text_states=batch["teacher_text_states"],
    )
    isolated = model._functional_credit_isolated_edit_slots(
        output,
        batch["text_states"],
    )
    candidates = torch.ones(4, 3, dtype=torch.bool)
    ordinary, _ = model._functional_coalition_queries(
        edit_slots=output["edit_slots"],
        candidate_slots=candidates,
        z0=output["z0"],
        reference_state=output["reference_state"],
    )
    new_block = torch.zeros(4, 1, 3, dtype=torch.bool)
    new_block[:, 0, 1] = True
    isolated_query = model._functional_conditional_step_queries(
        deployed_edit_slots=output["edit_slots"],
        isolated_edit_slots=isolated,
        new_block_mask=new_block,
        next_coalition_index=torch.full((4, 1), 3, dtype=torch.long),
        candidate_slots=candidates,
        z0=output["z0"],
        reference_state=output["reference_state"],
    )
    assert torch.equal(isolated_query[:, 0], ordinary[:, 3])


def test_disabled_conditional_schedule_preserves_first_round_forward() -> None:
    first = make_model(enabled=False, credit_schedule="first_round").eval()
    conditional = make_model(
        enabled=False,
        credit_schedule="conditional_residual",
    ).eval()
    conditional.load_state_dict(first.state_dict())
    batch = make_batch()
    with torch.no_grad():
        first_output = first.forward(
            batch["reference_features"],
            batch["text_states"],
            batch["text_attention_mask"],
            text_content_mask=batch["text_content_mask"],
            teacher_reference_features=batch["teacher_reference_features"],
            teacher_text_states=batch["teacher_text_states"],
        )
        conditional_output = conditional.forward(
            batch["reference_features"],
            batch["text_states"],
            batch["text_attention_mask"],
            text_content_mask=batch["text_content_mask"],
            teacher_reference_features=batch["teacher_reference_features"],
            teacher_text_states=batch["teacher_text_states"],
        )
    assert torch.equal(first_output["edit_slots"], conditional_output["edit_slots"])
    assert torch.equal(first_output["q0"], conditional_output["q0"])


def test_full_functional_compute_loss_backward_and_fixed_executor_steps() -> None:
    model = make_model(enabled=True).train()
    batch = make_batch()
    losses = model.compute_loss(batch)
    assert torch.isfinite(losses["retrieval_loss"])
    assert torch.isfinite(losses["functional_loss"])
    assert losses["functional_loss"].requires_grad
    for key in (
        "functional/error_mode_rank",
        "functional/residual_active_modes",
        "functional/credited_slots",
        "functional/unique_mode_coverage",
        "functional/redundant_credit_fraction",
        "functional/pair_synergy_fraction",
        "functional/heldout_validation_available",
        "functional/inferred_k_eff",
        "functional/owned_mode_count",
        "functional/unowned_positive_mode_count",
        "functional/max_modes_per_owner",
        "functional/giant_owner_fraction",
        "functional/ownership_row_similarity",
        "functional/unresolved_multimode_fraction",
    ):
        assert key in losses
        assert torch.isfinite(losses[key])
    total = losses["retrieval_loss"] + 0.1 * losses["functional_loss"]
    total.backward()
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
    assert all(parameter.grad is None for parameter in model.teacher.parameters())

    output = model.forward(
        batch["reference_features"],
        batch["text_states"],
        batch["text_attention_mask"],
        text_content_mask=batch["text_content_mask"],
        teacher_reference_features=batch["teacher_reference_features"],
        teacher_text_states=batch["teacher_text_states"],
    )
    assert output["trace_valid_mask"].shape[1] == model.num_slots


def test_full_conditional_compute_loss_backward() -> None:
    model = make_model(
        enabled=True,
        credit_schedule="conditional_residual",
    ).train()
    losses = model.compute_loss(make_batch())
    assert torch.isfinite(losses["functional_loss"])
    assert losses["functional_loss"].requires_grad
    for key in (
        "functional/conditional_steps",
        "functional/conditional_credited_slots",
        "functional/conditional_credited_modes",
        "functional/conditional_residual_gain",
        "functional/conditional_stop_no_gain_fraction",
        "functional/conditional_clone_rejection_fraction",
        "functional/conditional_pair_fraction",
    ):
        assert key in losses
        assert torch.isfinite(losses[key])
    (losses["retrieval_loss"] + 0.1 * losses["functional_loss"]).backward()
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
    assert all(parameter.grad is None for parameter in model.teacher.parameters())


def test_disabled_baseline_preserves_run_c_forward_and_zero_auxiliary() -> None:
    model_off = make_model(enabled=False).eval()
    model_on = make_model(enabled=True).eval()
    model_on.load_state_dict(model_off.state_dict())
    batch = make_batch()
    with torch.no_grad():
        off = model_off.compute_loss(batch)
        on_output = model_on.forward(
            batch["reference_features"],
            batch["text_states"],
            batch["text_attention_mask"],
            text_content_mask=batch["text_content_mask"],
            teacher_reference_features=batch["teacher_reference_features"],
            teacher_text_states=batch["teacher_text_states"],
        )
        off_output = model_off.forward(
            batch["reference_features"],
            batch["text_states"],
            batch["text_attention_mask"],
            text_content_mask=batch["text_content_mask"],
            teacher_reference_features=batch["teacher_reference_features"],
            teacher_text_states=batch["teacher_text_states"],
        )
    assert off["functional_loss"].item() == 0.0
    assert torch.equal(off_output["edit_slots"], on_output["edit_slots"])
    assert torch.equal(off_output["q0"], on_output["q0"])
    assert model_off.slot_value_source == "contextual"
    assert not model_off.slot_effect_in_value
    assert model_off.slot_value_assignment == "soft_shared"


def test_checkpoint_provenance_rejects_enabled_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    omegaconf_stub = types.ModuleType("omegaconf")
    omegaconf_stub.OmegaConf = object
    monkeypatch.setitem(sys.modules, "omegaconf", omegaconf_stub)
    from evaluate_qasa_inference import load_checkpoint

    source = make_model(enabled=False)
    destination = make_model(enabled=True)
    state = {
        key: value
        for key, value in source.state_dict().items()
        if not key.startswith("teacher.")
    }
    path = tmp_path / "baseline.pt"
    torch.save(
        {
            "model_state_dict": state,
            "experiment_provenance": source.experiment_provenance(),
        },
        path,
    )
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        load_checkpoint(destination, path)


def test_checkpoint_provenance_rejects_rank_gate_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    omegaconf_stub = types.ModuleType("omegaconf")
    omegaconf_stub.OmegaConf = object
    monkeypatch.setitem(sys.modules, "omegaconf", omegaconf_stub)
    from evaluate_qasa_inference import load_checkpoint

    source = make_model(enabled=True, rank_threshold=0.25)
    destination = make_model(enabled=True, rank_threshold=0.5)
    path = tmp_path / "rank-threshold.pt"
    torch.save(
        {
            "model_state_dict": {
                key: value
                for key, value in source.state_dict().items()
                if not key.startswith("teacher.")
            },
            "experiment_provenance": source.experiment_provenance(),
        },
        path,
    )
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        load_checkpoint(destination, path)


def test_checkpoint_provenance_rejects_credit_schedule_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    omegaconf_stub = types.ModuleType("omegaconf")
    omegaconf_stub.OmegaConf = object
    monkeypatch.setitem(sys.modules, "omegaconf", omegaconf_stub)
    from evaluate_qasa_inference import load_checkpoint

    source = make_model(enabled=True, credit_schedule="first_round")
    destination = make_model(
        enabled=True,
        credit_schedule="conditional_residual",
    )
    path = tmp_path / "first-round.pt"
    torch.save(
        {
            "model_state_dict": {
                key: value
                for key, value in source.state_dict().items()
                if not key.startswith("teacher.")
            },
            "experiment_provenance": source.experiment_provenance(),
        },
        path,
    )
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        load_checkpoint(destination, path)


def test_credit_schedule_validation() -> None:
    with pytest.raises(ValueError, match="functional_credit_schedule"):
        make_model(enabled=True, credit_schedule="not-a-schedule")


def test_functional_mode_rejects_non_run_c_architecture() -> None:
    with pytest.raises(ValueError, match="Run C contract"):
        TAPER(
            DummyTeacher(query_dim=4),
            text_dim=5,
            reference_dim=7,
            teacher_text_dim=5,
            teacher_query_dim=4,
            query_dim=4,
            slot_value_source="teacher_raw",
            slot_effect_in_value=False,
            slot_value_assignment="soft_shared",
            functional_ownership_enabled=True,
        )


def test_slot_pairs_are_deterministic() -> None:
    assert torch.equal(
        slot_pairs(3, torch.device("cpu")),
        torch.tensor([[0, 1], [0, 2], [1, 2]]),
    )
