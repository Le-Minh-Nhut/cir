from __future__ import annotations

import copy

import torch

from training.taper_mag_engine import CurriculumStage, EngineConfig
from training.taper_mag_profiler import profile_taper_runtime
from test_taper_mag_training_contract import _end_to_end_fixture


def test_profiler_smoke_is_finite_and_does_not_change_weights_or_optimizer() -> None:
    fg, taper, policy, supervision, engine = _end_to_end_fixture()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in taper.parameters() if parameter.requires_grad], lr=1e-3
    )
    sum(parameter.square().sum() for parameter in taper.parameters()).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    before_model = {name: value.detach().clone() for name, value in taper.state_dict().items()}
    before_text = {name: value.detach().clone() for name, value in fg.model.state_dict().items()}
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    report = profile_taper_runtime(
        engine,
        policy,
        supervision,
        EngineConfig(stage=CurriculumStage.ACTOR_WARMUP, horizon=1),
        optimizer=optimizer,
        repeats=2,
    )
    assert report["numerical"]["finite"]
    assert report["parameter_counts"]["trainable_taper_params"] > 0
    assert report["timing_ms"]["backbone_text"] >= 0
    assert report["timing_ms"]["validation_query_p95"] >= 0
    assert report["timing_ms"]["train_optimizer_step"] >= 0
    assert report["throughput"]["candidate_previews_per_second"] > 0
    assert report["flops"]["preview"] is None
    for name, value in taper.state_dict().items():
        torch.testing.assert_close(value, before_model[name])
    for name, value in fg.model.state_dict().items():
        torch.testing.assert_close(value, before_text[name])
    after_optimizer = optimizer.state_dict()
    assert after_optimizer["param_groups"] == before_optimizer["param_groups"]
    assert after_optimizer["state"].keys() == before_optimizer["state"].keys()
    for parameter_id, before_state in before_optimizer["state"].items():
        after_state = after_optimizer["state"][parameter_id]
        assert after_state.keys() == before_state.keys()
        for key, before_value in before_state.items():
            if torch.is_tensor(before_value):
                torch.testing.assert_close(after_state[key], before_value)
            else:
                assert after_state[key] == before_value
