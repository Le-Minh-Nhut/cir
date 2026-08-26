from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from calibrate_r4_lambda import (
    DEFAULT_LAMBDAS,
    calibrate_model,
    force_r4a,
    parse_args,
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


def test_parse_lambda_list_and_defaults() -> None:
    parsed = parse_args(
        [
            "--checkpoint",
            "checkpoint.pt",
            "--lambdas",
            "1.0",
            "0.5",
            "0.15",
        ]
    )
    assert parsed.lambdas == [1.0, 0.5, 0.15]

    defaults = parse_args(["--checkpoint", "checkpoint.pt"])
    assert defaults.lambdas == list(DEFAULT_LAMBDAS)

    with pytest.raises(SystemExit):
        parse_args(["--checkpoint", "checkpoint.pt", "--lambdas", "0"])


def test_force_r4a_disables_capacity_and_selects_qisca() -> None:
    model = tiny_model(routing_mode="entmax15", capacity_enabled=True)
    force_r4a(model)
    assert model.routing_mode == "qisca"
    assert model.r4_capacity_enabled is False


def test_one_model_and_unchanged_parameters_are_reused_across_lambdas() -> None:
    model = tiny_model(routing_mode="entmax15", capacity_enabled=True)
    parameter_ids = tuple(id(parameter) for parameter in model.parameters())
    parameter_values = tuple(
        parameter.detach().clone() for parameter in model.parameters()
    )
    batch = synthetic_batch()
    seen_model_ids: list[int] = []
    original_build = model.build_edit_slots

    def recording_build(*args, **kwargs):
        seen_model_ids.append(id(model))
        return original_build(*args, **kwargs)

    model.build_edit_slots = recording_build  # type: ignore[method-assign]
    results, num_queries = calibrate_model(
        model,
        [1.0, 0.5, 0.25],
        lambda: [batch],
    )

    assert num_queries == 2
    assert len(results) == 3
    assert set(seen_model_ids) == {id(model)}
    assert tuple(id(parameter) for parameter in model.parameters()) == parameter_ids
    for parameter, expected in zip(model.parameters(), parameter_values, strict=True):
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
    assert model.routing_mode == "qisca"
    assert model.r4_capacity_enabled is False


def test_qasa_is_invariant_but_routing_changes_with_lambda() -> None:
    model = tiny_model()
    batch = synthetic_batch()
    model.r4_lambda = 1.0
    first = model.build_edit_slots(
        batch["text_states"],
        batch["text_attention_mask"],
        text_content_mask=batch["text_content_mask"],
    )
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
    assert not torch.allclose(first["routing_masks"], second["routing_masks"])


def test_calibration_json_schema_is_stable(tmp_path: Path) -> None:
    model = tiny_model()
    results, num_queries = calibrate_model(
        model,
        [1.0, 0.5],
        lambda: [synthetic_batch()],
    )
    report = {
        "checkpoint": "checkpoint.pt",
        "theta": model.r4_theta,
        "capacity_enabled": model.r4_capacity_enabled,
        "routing_mode": model.routing_mode,
        "num_queries": num_queries,
        "lambdas": [1.0, 0.5],
        "results": results,
    }
    path = tmp_path / "report.json"
    write_report(path, report)
    restored = json.loads(path.read_text(encoding="utf-8"))

    assert set(restored) == {
        "checkpoint",
        "theta",
        "capacity_enabled",
        "routing_mode",
        "num_queries",
        "lambdas",
        "results",
    }
    assert restored["routing_mode"] == "qisca"
    assert restored["capacity_enabled"] is False
    assert len(restored["results"]) == 2
    for result in restored["results"]:
        assert set(result) == {
            "lambda",
            "metrics",
            "slot_active_frequency",
            "soft_dominant_slot_frequency",
        }
        assert "r4_preprojection_token_mass_mean" in result["metrics"]
        assert "routing_support_overlap_mean" in result["metrics"]
        assert set(result["slot_active_frequency"]) == {"0", "1", "2", "3"}
