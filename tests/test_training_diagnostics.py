from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from torch import nn

from training.diagnostics import (
    append_step_jsonl,
    build_step_log_record,
    per_slot_gradient_l2,
    proposal_query_gradient_diagnostics,
    slot_gradient_metrics,
    validate_checkpoint_candidate_count,
)


def test_step_logger_writes_global_step_and_discrete_dac_state(tmp_path: Path) -> None:
    components = {
        "total": torch.tensor(3.0),
        "terminal": torch.tensor(1.0),
        "candidate_credit_loss": torch.tensor(0.5),
        "candidate_credit_weighted": torch.tensor(0.5),
        "dpp_raw": torch.tensor(0.25),
        "dpp_weighted": torch.tensor(0.0025),
        "dac_stage": torch.tensor(2.0),
        "dac_num_groups": torch.tensor(4.0),
        "dac_group_size": torch.tensor(2.0),
        "dac_gradient_candidate_fraction": torch.tensor(0.25),
        "dac_responsibility_concentration": torch.tensor(0.375),
        "dac_responsibility_frequency_0": torch.tensor(0.5),
        "dac_responsibility_frequency_1": torch.tensor(0.5),
        "dac_group_winning_frequency_0": torch.tensor(0.75),
        "dac_group_winning_frequency_1": torch.tensor(0.25),
        "useful_candidate_count": torch.tensor(2.5),
        "functional_pairwise_cosine": torch.tensor(0.2),
        "functional_rank": torch.tensor(3.5),
        "stop_rate": torch.tensor(0.1),
    }
    record = build_step_log_record(
        global_step=4001,
        objective_global_step=4000,
        epoch=3,
        components=components,
        gradient_metrics={"proposal_query_grad_l2_0": 1.25},
    )
    path = tmp_path / "train_steps.jsonl"
    append_step_jsonl(path, record)

    stored = json.loads(path.read_text(encoding="utf-8").strip())
    assert stored["global_step"] == 4001
    assert stored["objective_global_step"] == 4000
    assert stored["epoch"] == 3
    assert stored["dac_stage"] == 2
    assert stored["dac_num_groups"] == 4
    assert stored["dac_group_size"] == 2
    assert stored["dac_responsibility_frequency"] == pytest.approx([0.5, 0.5])
    assert stored["dac_group_winning_frequency"] == pytest.approx([0.75, 0.25])
    assert stored["proposal_query_grad_l2_0"] == pytest.approx(1.25)


def test_per_slot_query_gradient_norms_measure_zero_and_nonzero_slots() -> None:
    queries = nn.Parameter(torch.ones(8, 2))
    loss = queries[0].sum() + 2.0 * queries[3].sum()

    norms = per_slot_gradient_l2(loss, queries)
    metrics = slot_gradient_metrics(norms)

    assert norms.shape == (8,)
    assert norms.tolist() == pytest.approx(
        [math.sqrt(2.0), 0.0, 0.0, 2.0 * math.sqrt(2.0), 0.0, 0.0, 0.0, 0.0]
    )
    assert metrics["proposal_query_grad_l2_nonzero_count"] == 2
    assert metrics["proposal_query_grad_l2_nonzero_fraction"] == pytest.approx(0.25)
    assert [metrics[f"proposal_query_grad_l2_{index}"] for index in range(8)] == pytest.approx(
        norms.tolist()
    )


def test_proposal_query_diagnostics_expose_eight_slots_for_each_loss_route() -> None:
    model = nn.Module()
    model.proposal = nn.Module()
    model.proposal.queries = nn.Parameter(torch.ones(8, 2))
    queries = model.proposal.queries
    losses = {
        "total": queries.sum(),
        "terminal": queries[0].sum(),
        "candidate_credit_weighted": queries[4:].sum(),
    }

    metrics = proposal_query_gradient_diagnostics(model, losses)

    for prefix in (
        "proposal_query_grad_l2",
        "terminal_proposal_query_grad_l2",
        "candidate_credit_proposal_query_grad_l2",
    ):
        assert len([f"{prefix}_{index}" for index in range(8) if f"{prefix}_{index}" in metrics]) == 8
    assert metrics["terminal_proposal_query_grad_l2_0"] > 0
    assert metrics["terminal_proposal_query_grad_l2_1"] == 0
    assert metrics["candidate_credit_proposal_query_grad_l2_3"] == 0
    assert metrics["candidate_credit_proposal_query_grad_l2_4"] > 0
    assert queries.grad is None


def test_checkpoint_candidate_count_mismatch_has_k8_specific_remedy() -> None:
    checkpoint = {"metadata": {"model_config": {"num_candidates": 8}}}

    with pytest.raises(ValueError, match=r"stored=8.*configured=4.*model=iag_srme_k8"):
        validate_checkpoint_candidate_count(checkpoint, configured_num_candidates=4)

    validate_checkpoint_candidate_count(checkpoint, configured_num_candidates=8)
