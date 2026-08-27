from __future__ import annotations

from pathlib import Path
import sys
import types

import pytest
import torch
from torch import Tensor, nn

from models.functional_ownership import (
    block_residual_credit,
    functional_credit_loss,
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


def make_model(*, enabled: bool) -> TAPER:
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
        functional_pair_lookahead=True,
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


def test_global_giant_is_reported_without_artificial_split() -> None:
    effects = torch.tensor([[[2.0, 2.0], [0.2, 0.2]]])
    result = block_residual_credit(
        effects,
        torch.ones(1, 2, dtype=torch.bool),
    )
    assert result["credited_mask"].sum().item() == 1
    assert result["credited_mask"][0, 0]
    assert result["unique_mode_coverage"].item() == 1.0


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
    result = block_residual_credit(
        effects,
        torch.ones(1, 2, dtype=torch.bool),
    )
    assert result["credited_mask"][0, 0]
    assert not result["credited_mask"][0, 1]


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
    torch.testing.assert_close(off_output["q0"], on_output["q0"])
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


def test_slot_pairs_are_deterministic() -> None:
    assert torch.equal(
        slot_pairs(3, torch.device("cpu")),
        torch.tensor([[0, 1], [0, 2], [1, 2]]),
    )
