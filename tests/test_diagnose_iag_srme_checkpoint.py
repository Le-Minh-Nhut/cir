from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from diagnose_iag_srme_checkpoint import (
    REQUIRED_REPORT_KEYS,
    _retrieval_metrics,
    _selection_batch_counts,
    _validate_report_schema,
    repeat_candidate_control,
    same_parent_candidate_queries,
    single_candidate_control,
)
from evaluation.fashioniq import evaluate_fashioniq_recall


def test_new_stop_hazard_differs_from_absorbed_stop_occupancy() -> None:
    output = SimpleNamespace(
        intents=torch.empty(3, 4, 8),
        trace=(
            SimpleNamespace(
                selected_index=torch.tensor([4, 0, 1]),
                live_before=torch.tensor([True, True, True]),
                stopped_now=torch.tensor([True, False, False]),
            ),
            SimpleNamespace(
                selected_index=torch.tensor([4, 4, 2]),
                live_before=torch.tensor([False, True, True]),
                stopped_now=torch.tensor([False, True, False]),
            ),
            SimpleNamespace(
                selected_index=torch.tensor([4, 4, 4]),
                live_before=torch.tensor([False, False, True]),
                stopped_now=torch.tensor([False, False, True]),
            ),
        ),
    )
    counts = _selection_batch_counts(output)
    occupancy = counts["stop_occupancy"].float().mean(dim=0)
    hazard = counts["stopped_now"].sum(dim=0) / counts["live"].sum(dim=0)

    assert torch.allclose(occupancy, torch.tensor([1 / 3, 2 / 3, 1.0]))
    assert torch.allclose(hazard, torch.tensor([1 / 3, 1 / 2, 1.0]))
    assert occupancy[1] != hazard[1]


def test_single_control_executes_exactly_one_same_parent_candidate(
    core, synthetic_encoded
) -> None:
    core.eval()
    output = core(synthetic_encoded)
    single = single_candidate_control(output, candidate=2)

    assert single.executed_edit_count == 1
    assert torch.equal(single.query, output.trace[0].candidate_queries[:, 2])
    assert torch.equal(single.state, output.trace[0].candidate_states[:, 2])


def test_repeat_control_updates_recurrent_state_at_every_step(core, synthetic_encoded) -> None:
    core.eval()
    output = repeat_candidate_control(core, synthetic_encoded, candidate=1)

    for timestep, step in enumerate(output.trace):
        assert step.selected_index.eq(1).all()
        assert not torch.equal(step.next_state, step.current_state)
        if timestep > 0:
            assert torch.equal(step.current_state, output.trace[timestep - 1].next_state)


def test_counterfactual_controls_use_same_parent_candidates(core, synthetic_encoded) -> None:
    core.eval()
    output = core(synthetic_encoded)
    candidates, valid = same_parent_candidate_queries(output, timestep=0)

    step = output.trace[0]
    assert valid.all()
    assert torch.equal(step.candidate_states, step.current_state[:, None] + step.delta_z)
    assert torch.equal(candidates, step.candidate_queries)


def test_diagnostic_retrieval_preserves_reference_filtering() -> None:
    gallery = torch.eye(60)
    query = torch.zeros(1, 60)
    query[0, 0] = 100.0
    query[0, 1:10] = torch.arange(99.0, 90.0, -1.0)
    query[0, 10] = 90.0
    gallery_ids = [f"image-{index}" for index in range(60)]
    target_ids = ["image-10"]
    reference_ids = ["image-0"]
    raw_scores = F.normalize(query, dim=-1) @ gallery.T

    unfiltered = evaluate_fashioniq_recall(raw_scores, target_ids, gallery_ids)
    filtered = _retrieval_metrics(
        query, gallery, target_ids, gallery_ids, reference_ids
    )

    assert unfiltered["recall_at_10"] == 0.0
    assert filtered["recall_at_10"] == 100.0


def test_report_schema_has_required_keys_and_is_json_serializable() -> None:
    report = {key: {} for key in REQUIRED_REPORT_KEYS}
    report["checkpoint"] = "outputs/run/best.pt"
    report["checkpoint_epoch"] = 3
    report["checkpoint_metric"] = 42.0

    _validate_report_schema(report)
    assert json.loads(json.dumps(report))["checkpoint_epoch"] == 3

    with pytest.raises(AssertionError, match="missing top-level keys"):
        _validate_report_schema({"checkpoint": "best.pt"})
