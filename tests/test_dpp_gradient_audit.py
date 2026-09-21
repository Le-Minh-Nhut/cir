from __future__ import annotations

import json
import torch
from torch import nn

from analyze_dpp_gradient import main as analyze_dpp_gradient
from diagnostics.dpp_gradient import dpp_gradient_audit
from losses.objective import IAGSRMEObjective, ObjectiveConfig, functional_dpp_loss
from models.iag_srme.utils.retrieval import build_teacher_masks, marginal_teacher_utilities


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.executor = nn.Linear(1, 3, bias=False)


def _output(model: _Model, *, zero_effect: bool = False) -> tuple[dict[str, object], torch.Tensor, list[str]]:
    targets = torch.eye(3)
    ids = ["a", "b", "c"]
    current = torch.full((3, 3), 1 / 3**0.5)
    desired = targets[:, None].expand(-1, 3, -1).clone()
    desired = desired + 0.05 * torch.eye(3)[None]
    if zero_effect:
        desired[:, 0] = current
    perturbation = model.executor(torch.ones(3, 1))[:, None] * 1e-3
    candidate = desired + perturbation
    effect = candidate - current[:, None]
    return (
        {
            "query": current,
            "steps": [
                {
                    "timestep": 0,
                    "live_indices": torch.arange(3),
                    "current_query": current,
                    "candidate_queries": candidate,
                    "delta_q": effect,
                    "selected_idx": torch.tensor([1, 1, 1]),
                    "scores": torch.zeros(3, 3),
                    "stopped_now": torch.zeros(3, dtype=torch.bool),
                }
            ],
        },
        targets,
        ids,
    )


def _dpp_raw(output: dict[str, object], targets: torch.Tensor, ids: list[str], config: ObjectiveConfig):
    step = output["steps"][0]
    positive, negative, _ = build_teacher_masks(ids, targets.device)
    teacher, valid = marginal_teacher_utilities(
        step["current_query"],
        step["candidate_queries"],
        targets,
        positive,
        negative,
        config.retrieval_temperature,
    )
    values = [
        functional_dpp_loss(
            step["delta_q"][row],
            None,
            teacher[row],
            kappa=config.kappa_dpp,
            sigma=config.sigma_dpp,
            tau=config.tau_dpp,
            useful_threshold=config.useful_threshold,
            jitter=config.dpp_jitter,
        )
        for row in valid.nonzero(as_tuple=False).flatten().tolist()
    ]
    valid_values = [value for value in values if bool(value["valid"])]
    return sum((value["loss"] for value in valid_values)) / len(valid_values), torch.tensor(len(valid_values))


def test_dpp_gradient_audit_reports_exact_dpp_gradients_without_grad_contamination() -> None:
    model = _Model()
    output, targets, ids = _output(model)
    config = ObjectiveConfig(dpp_enabled=True, useful_threshold=0.55)
    raw, count = _dpp_raw(output, targets, ids, config)
    objective = nn.Module()
    objective.config = config

    report = dpp_gradient_audit(model, objective, output, targets, ids, raw, count, ids)

    assert report["objective_dpp_valid_row_count"] == 3
    assert report["audit_dpp_valid_row_count"] == 3
    assert report["dpp_valid_count_matches_objective"] is True
    assert report["dpp_valid_row_count"] == 3
    assert report["dpp_total_eligible_row_count"] == 3
    assert report["dpp_valid_rate"] == 1.0
    assert report["effect_norm_quantiles"]["p00"] > 0
    assert report["live_grad_norm_quantiles"]["p50"] > 0
    assert report["fp32_reference_grad_norm_quantiles"]["p50"] > 0
    assert [row["batch_sample_index"] for row in report["by_step"]["t0"]["rows"]] == [0, 1, 2]
    assert [row["sample_id"] for row in report["by_step"]["t0"]["rows"]] == ids
    assert all(parameter.grad is None for parameter in model.parameters())
    raw.backward()
    assert model.executor.weight.grad is not None


def test_dpp_audit_count_agrees_with_live_objective() -> None:
    model = _Model()
    output, targets, ids = _output(model)
    config = ObjectiveConfig(
        terminal_weight=0.0,
        lambda_pair=0.0,
        lambda_gain=0.0,
        dpp_enabled=True,
        lambda_dpp=1.0,
        useful_threshold=0.55,
    )
    objective = IAGSRMEObjective(config, width=3)
    components = objective(output, targets, ids)

    report = dpp_gradient_audit(
        model,
        objective,
        output,
        targets,
        ids,
        components["dpp_raw"],
        components["dpp_valid_timestep_count"],
        ids,
    )

    assert report["objective_dpp_valid_row_count"] == int(components["dpp_valid_timestep_count"])
    assert report["audit_dpp_valid_row_count"] == int(components["dpp_valid_timestep_count"])
    assert report["dpp_valid_count_matches_objective"] is True


def test_dpp_gradient_audit_returns_empty_report_without_valid_rows() -> None:
    model = _Model()
    output, targets, ids = _output(model)
    config = ObjectiveConfig(dpp_enabled=True, useful_threshold=1.0)
    raw = output["steps"][0]["delta_q"].sum() * 0
    objective = nn.Module()
    objective.config = config

    report = dpp_gradient_audit(model, objective, output, targets, ids, raw, torch.tensor(0))

    assert report["dpp_valid_row_count"] == 0
    assert report["effect_norm_quantiles"]["p50"] == 0
    assert all(parameter.grad is None for parameter in model.parameters())


def test_dpp_gradient_audit_exposes_zero_effect_with_open_quality_gate() -> None:
    model = _Model()
    output, targets, ids = _output(model, zero_effect=True)
    config = ObjectiveConfig(dpp_enabled=True, useful_threshold=0.49)
    raw, count = _dpp_raw(output, targets, ids, config)
    objective = nn.Module()
    objective.config = config

    report = dpp_gradient_audit(model, objective, output, targets, ids, raw, count)

    rows = report["by_step"]["t0"]["rows"]
    assert report["dpp_valid_row_count"] == 3
    assert min(row["min_effect_norm"] for row in rows) < 0.01
    assert all(row["fp32_grad_nonfinite_fraction"] == 0.0 for row in rows)
    assert max(value for row in rows for value in row["fp32_reference_grad_norms"]) > 1.0


def test_analyzer_reports_row_sample_identifier(tmp_path, capsys, monkeypatch) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        json.dumps(
            {
                "dpp_gradient_audit": {
                    "objective_dpp_valid_row_count": 1,
                    "audit_dpp_valid_row_count": 1,
                    "dpp_valid_count_matches_objective": True,
                    "dpp_total_eligible_row_count": 1,
                    "finite_in_fp32_but_nonfinite_live_count": 0,
                    "by_step": {
                        "t0": {
                            "rows": [
                                {
                                    "sample_id": "trace-me",
                                    "batch_sample_index": 2,
                                    "min_effect_norm": 0.01,
                                    "effect_norms": [0.01],
                                    "live_effect_grad_norms": [1.0],
                                    "fp32_reference_grad_norms": [1.0],
                                }
                            ]
                        }
                    },
                }
            }
        )
    )
    monkeypatch.setattr("sys.argv", ["analyze_dpp_gradient.py", str(path)])

    analyze_dpp_gradient()

    assert "sample_id=trace-me batch_sample_index=2" in capsys.readouterr().out
