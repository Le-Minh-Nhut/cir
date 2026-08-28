from __future__ import annotations

import torch

from models.taper_mag.model import TaperMAG, TaperMAGConfig
from training.taper_mag_audit import (
    causal_operator_interventions,
    clone_operator_sets,
    execute_operator_once,
)
from test_taper_mag_actor import encoded_batch


def test_causal_repeat_executes_nonlinear_state_transition_twice() -> None:
    torch.manual_seed(23)
    model = TaperMAG(
        TaperMAGConfig(
            text_dim=20,
            vision_dim=24,
            retrieval_dim=32,
            dropout=0,
            max_steps=4,
        )
    ).eval()
    encoded = encoded_batch()
    best = torch.tensor([0, 1])
    calls = 0
    original_enumerate = model.executor.enumerate

    def counted_enumerate(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_enumerate(*args, **kwargs)

    model.executor.enumerate = counted_enumerate  # type: ignore[method-assign]
    outputs = causal_operator_interventions(
        model, encoded, best, max_steps=4, step_cost=0.0
    )
    model.executor.enumerate = original_enumerate  # type: ignore[method-assign]
    assert calls >= 10

    _, initial, operators = model.prepare(encoded)
    selected = operators.operators.gather(
        1, best[:, None, None].expand(-1, 1, operators.operators.shape[-1])
    ).squeeze(1)
    first_state = execute_operator_once(model, initial, selected)
    second_state = execute_operator_once(model, first_state, selected)
    real_repeat = model.readout(second_state).query
    torch.testing.assert_close(outputs["repeat_best"], real_repeat)

    q0 = model.readout(initial).query
    q1 = model.readout(first_state).query
    synthetic_query_repeat = torch.nn.functional.normalize(q0 + 2.0 * (q1 - q0), dim=-1)
    assert not torch.allclose(real_repeat, synthetic_query_repeat, atol=1e-8, rtol=1e-6)


def test_clone_zero_and_mean_controls_replace_operators_before_executor() -> None:
    torch.manual_seed(29)
    model = TaperMAG(
        TaperMAGConfig(
            text_dim=20,
            vision_dim=24,
            retrieval_dim=32,
            dropout=0,
            max_steps=2,
        )
    ).eval()
    encoded = encoded_batch()
    best = torch.tensor([2, 0])
    _, initial, operators = model.prepare(encoded)
    cloned = clone_operator_sets(operators, best)
    expected_best = operators.operators.gather(
        1, best[:, None, None].expand(-1, 1, operators.operators.shape[-1])
    )
    expected_mean = operators.operators.mean(dim=1, keepdim=True)
    torch.testing.assert_close(
        cloned["clone_all_best"].operators,
        expected_best.expand_as(operators.operators),
    )
    torch.testing.assert_close(
        cloned["clone_all_mean"].operators,
        expected_mean.expand_as(operators.operators),
    )

    outputs = causal_operator_interventions(model, encoded, best, max_steps=2)
    zero_state = execute_operator_once(
        model, initial, torch.zeros_like(expected_mean[:, 0])
    )
    mean_state = execute_operator_once(model, initial, expected_mean[:, 0])
    torch.testing.assert_close(outputs["operator_zero"], model.readout(zero_state).query)
    torch.testing.assert_close(outputs["operator_mean"], model.readout(mean_state).query)
    assert set(outputs) == {
        "repeat_best",
        "mean_repeat",
        "clone_all_best",
        "clone_all_mean",
        "operator_zero",
        "operator_mean",
    }
