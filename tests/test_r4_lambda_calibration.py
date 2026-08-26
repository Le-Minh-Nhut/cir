from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from calibrate_r4_lambda import (
    DEFAULT_LAMBDAS,
    DEFAULT_THETAS,
    calibrate_model,
    force_r4a,
    parse_args,
    print_tables,
    write_report,
)
from models.taper import TAPER


def tiny_model(
    *,
    routing_mode: str = "qisca",
    capacity_enabled: bool = False,
) -> TAPER:
    torch.manual_seed(71)
    return TAPER(
        text_dim=4,
        slot_dim=4,
        reference_dim=4,
        query_dim=4,
        state_dim=4,
        num_slots=4,
        num_primitives=2,
        routing_mode=routing_mode,
        r4_theta=0.10,
        r4_lambda=1.0,
        r4_capacity_enabled=capacity_enabled,
        r4_slot_capacity=0.5,
    )


def synthetic_batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(73)
    return {
        "text_states": torch.randn(2, 6, 4),
        "text_attention_mask": torch.ones(2, 6, dtype=torch.bool),
        "text_content_mask": torch.tensor(
            [
                [False, True, True, True, True, False],
                [False, True, True, True, False, False],
            ]
        ),
    }


def test_parse_theta_lambda_lists_and_defaults() -> None:
    parsed = parse_args(
        [
            "--checkpoint",
            "checkpoint.pt",
            "--thetas",
            "0.25",
            "0.10",
            "--lambdas",
            "1.0",
            "0.5",
            "0.15",
        ]
    )
    assert parsed.thetas == [0.25, 0.10]
    assert parsed.lambdas == [1.0, 0.5, 0.15]

    defaults = parse_args(["--checkpoint", "checkpoint.pt"])
    assert defaults.thetas == list(DEFAULT_THETAS)
    assert defaults.lambdas == list(DEFAULT_LAMBDAS)

    with pytest.raises(SystemExit):
        parse_args(["--checkpoint", "checkpoint.pt", "--lambdas", "0"])


@pytest.mark.parametrize("theta", ["-0.01", "1.0", "nan", "inf", "-inf"])
def test_invalid_theta_is_rejected(theta: str) -> None:
    with pytest.raises(SystemExit):
        parse_args(["--checkpoint", "checkpoint.pt", "--thetas", theta])


def test_force_r4a_disables_capacity_and_selects_qisca() -> None:
    model = tiny_model(routing_mode="entmax15", capacity_enabled=True)
    force_r4a(model)
    assert model.routing_mode == "qisca"
    assert model.r4_capacity_enabled is False
    assert model.r4_candidate_mode == "qasa_selected"


def test_one_model_is_reused_and_hyperparameters_are_restored() -> None:
    model = tiny_model(routing_mode="entmax15", capacity_enabled=True)
    parameter_ids = tuple(id(parameter) for parameter in model.parameters())
    parameter_values = tuple(
        parameter.detach().clone() for parameter in model.parameters()
    )
    original_theta = model.r4_theta
    original_lambda = model.r4_lambda
    batch = synthetic_batch()
    seen_model_ids: list[int] = []
    original_build = model.build_edit_slots

    def recording_build(*args, **kwargs):
        seen_model_ids.append(id(model))
        return original_build(*args, **kwargs)

    model.build_edit_slots = recording_build  # type: ignore[method-assign]
    results, num_queries = calibrate_model(
        model,
        [0.25, 0.10],
        [1.0, 0.5, 0.25],
        lambda: [batch],
    )

    assert num_queries == 2
    assert len(results) == 6
    assert set(seen_model_ids) == {id(model)}
    assert tuple(id(parameter) for parameter in model.parameters()) == parameter_ids
    for parameter, expected in zip(model.parameters(), parameter_values, strict=True):
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
    assert model.routing_mode == "qisca"
    assert model.r4_capacity_enabled is False
    assert model.r4_candidate_mode == "qasa_selected"
    assert model.r4_theta == original_theta
    assert model.r4_lambda == original_lambda


def test_qasa_is_invariant_across_theta_and_lambda() -> None:
    model = tiny_model()
    batch = synthetic_batch()
    model.r4_theta = 0.25
    model.r4_lambda = 1.0
    first = model.build_edit_slots(
        batch["text_states"],
        batch["text_attention_mask"],
        text_content_mask=batch["text_content_mask"],
    )
    model.r4_theta = 0.05
    model.r4_lambda = 0.15
    second = model.build_edit_slots(
        batch["text_states"],
        batch["text_attention_mask"],
        text_content_mask=batch["text_content_mask"],
    )

    for name in (
        "ownership_logits",
        "slot_masks",
        "qasa_attention",
        "qasa_quality",
        "qasa_selected_mask",
    ):
        torch.testing.assert_close(first[name], second[name])


def test_theta_controls_support_and_lambda_controls_row_pressure() -> None:
    model = tiny_model()
    competition = torch.tensor([[[0.40], [0.30], [0.20], [0.10]]])
    valid = torch.ones(1, 1, dtype=torch.bool)
    selected = torch.ones(1, 4, dtype=torch.bool)

    model.r4_theta = 0.25
    model.r4_lambda = 1.0
    high_theta = model._qisca_routing(competition, valid, selected)
    assert torch.equal(
        high_theta[:, :, 0] > 0,
        torch.tensor([[True, True, False, False]]),
    )

    model.r4_theta = 0.15
    model.r4_lambda = 1.0
    low_theta = model._qisca_routing(competition, valid, selected)
    assert torch.equal(
        low_theta[:, :, 0] > 0,
        torch.tensor([[True, True, True, False]]),
    )
    torch.testing.assert_close(low_theta.sum(dim=1), torch.tensor([[0.45]]))

    model.r4_lambda = 0.10
    pressured = model._qisca_routing(competition, valid, selected)
    low_pressure_eligibility = (competition - 0.15) / 1.0 > 0
    high_pressure_eligibility = (competition - 0.15) / 0.10 > 0
    assert torch.equal(low_pressure_eligibility, high_pressure_eligibility)
    torch.testing.assert_close(pressured.sum(dim=1), torch.ones(1, 1))


def test_grid_rejects_different_query_counts() -> None:
    model = tiny_model()
    original_theta = model.r4_theta
    original_lambda = model.r4_lambda
    batch = synthetic_batch()
    calls = 0

    def inconsistent_batches():
        nonlocal calls
        calls += 1
        return [batch] if calls == 1 else [batch, batch]

    with pytest.raises(RuntimeError, match="different query counts"):
        calibrate_model(model, [0.25], [1.0, 0.5], inconsistent_batches)
    assert model.r4_theta == original_theta
    assert model.r4_lambda == original_lambda


def test_calibration_json_schema_is_stable(tmp_path: Path) -> None:
    model = tiny_model()
    results, num_queries = calibrate_model(
        model,
        [0.25, 0.10],
        [1.0, 0.5],
        lambda: [synthetic_batch()],
    )
    report = {
        "checkpoint": "checkpoint.pt",
        "capacity_enabled": model.r4_capacity_enabled,
        "candidate_mode": model.r4_candidate_mode,
        "routing_mode": model.routing_mode,
        "num_queries": num_queries,
        "thetas": [0.25, 0.10],
        "lambdas": [1.0, 0.5],
        "results": results,
    }
    path = tmp_path / "report.json"
    write_report(path, report)
    restored = json.loads(path.read_text(encoding="utf-8"))

    assert set(restored) == {
        "checkpoint",
        "capacity_enabled",
        "candidate_mode",
        "routing_mode",
        "num_queries",
        "thetas",
        "lambdas",
        "results",
    }
    assert restored["routing_mode"] == "qisca"
    assert restored["capacity_enabled"] is False
    assert restored["candidate_mode"] == "qasa_selected"
    assert len(restored["results"]) == 4
    for result in restored["results"]:
        assert set(result) == {
            "theta",
            "lambda",
            "metrics",
            "slot_active_frequency",
            "soft_dominant_slot_frequency",
        }
        assert "r4_preprojection_token_mass_mean" in result["metrics"]
        assert "routing_support_overlap_mean" in result["metrics"]
        assert set(result["slot_active_frequency"]) == {"0", "1", "2", "3"}


def test_output_has_theta_only_summary_and_full_grid(capsys) -> None:
    model = tiny_model()
    results, _ = calibrate_model(
        model,
        [0.25, 0.10],
        [1.0, 0.5],
        lambda: [synthetic_batch()],
    )
    print_tables(results)
    output = capsys.readouterr().out
    assert "THETA SUPPORT CALIBRATION (lambda=1.0)" in output
    assert "theta  lambda  pre_mass" in output
    assert "theta=0.10 lambda=0.50" in output
