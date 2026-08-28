from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import MethodType

import pytest
import torch
import yaml

from models.taper_mag.model import TaperMAG, TaperMAGConfig
from models.taper_mag.rollout import RolloutConfig
from test_taper_mag_actor import encoded_batch
from train_taper_mag import verify_resume_schedule_config
from training.taper_mag_curriculum import CanonicalV4Curriculum, CurriculumScheduler
from training.taper_mag_engine import CurriculumStage, EngineConfig


def test_exact_canonical_v4_epoch_boundaries() -> None:
    schedule = CanonicalV4Curriculum()
    expected = {
        1: (CurriculumStage.ACTOR_WARMUP, 1),
        8: (CurriculumStage.ACTOR_WARMUP, 1),
        9: (CurriculumStage.CRITIC_WARMUP, 1),
        14: (CurriculumStage.CRITIC_WARMUP, 1),
        15: (CurriculumStage.DAGGER_T2, 2),
        26: (CurriculumStage.DAGGER_T2, 2),
        27: (CurriculumStage.ST_BRIDGE, 3),
        40: (CurriculumStage.ST_BRIDGE, 3),
        41: (CurriculumStage.PREDICTED_T4, 4),
        52: (CurriculumStage.PREDICTED_T4, 4),
        53: (CurriculumStage.HARDEN, 4),
        60: (CurriculumStage.HARDEN, 4),
    }
    for epoch, (phase, horizon) in expected.items():
        state = schedule.state_for_epoch(epoch)
        assert (state.phase, state.horizon) == (phase, horizon)
    assert schedule.state_for_epoch(15).oracle_mix == pytest.approx(0.8)
    assert schedule.state_for_epoch(26).oracle_mix == pytest.approx(0.3)
    assert schedule.state_for_epoch(27).oracle_mix == pytest.approx(0.3)
    assert schedule.state_for_epoch(40).oracle_mix == pytest.approx(0.0)
    assert schedule.state_for_epoch(27).selection_temperature == pytest.approx(1.0)
    assert schedule.state_for_epoch(40).selection_temperature == pytest.approx(0.5)
    assert schedule.state_for_epoch(27).rho_gate == pytest.approx(0.0)
    assert schedule.state_for_epoch(40).rho_gate == pytest.approx(0.25)
    assert schedule.state_for_epoch(41).exploration_probability == pytest.approx(0.05)
    assert schedule.state_for_epoch(47).exploration_probability == pytest.approx(0.05)
    assert schedule.state_for_epoch(52).exploration_probability == pytest.approx(0.0)
    harden = schedule.state_for_epoch(53)
    assert not harden.straight_through
    assert harden.oracle_mix == 0
    assert harden.exploration_probability == 0


def test_curriculum_checkpoint_consistency_and_manual_mode() -> None:
    scheduler = CurriculumScheduler.from_config(
        {"curriculum_mode": "canonical_v4"}, step_cost=0.1
    )
    state = scheduler.state_for_epoch(27)
    scheduler.verify_checkpoint(27, state.checkpoint_dict())
    with pytest.raises(RuntimeError, match="curriculum state mismatch"):
        scheduler.verify_checkpoint(
            27, replace(state, selection_temperature=0.7).checkpoint_dict()
        )
    manual = CurriculumScheduler.from_config(
        {
            "curriculum_mode": "manual",
            "stage": "harden",
            "horizon": 4,
            "oracle_mix": 0.0,
            "straight_through": False,
        },
        step_cost=0.0,
    ).state_for_epoch(999)
    assert manual.phase == CurriculumStage.HARDEN and manual.horizon == 4


def _model() -> TaperMAG:
    return TaperMAG(
        TaperMAGConfig(text_dim=20, vision_dim=24, retrieval_dim=32, dropout=0)
    )


def test_top2_exploration_is_training_only_target_free_and_seeded() -> None:
    taper = _model().train()

    def controlled(self, features, *, detach_inputs=False):
        del self, detach_inputs
        values = torch.zeros(features.shape[:2], device=features.device)
        values[:, 0] = 2.0
        values[:, 1] = 1.0
        return values

    taper.utility.forward = MethodType(controlled, taper.utility)
    rollout = RolloutConfig(
        max_steps=1,
        selection_mode="learned",
        exploration_probability=1.0,
    )
    torch.manual_seed(123)
    first = taper(encoded_batch(), rollout).trace.actions
    torch.manual_seed(123)
    second = taper(encoded_batch(), rollout).trace.actions
    assert torch.equal(first, second)
    assert first[:, 0].tolist() == [1, 1]
    taper.eval()
    deterministic = taper(encoded_batch(), rollout).trace.actions
    assert deterministic[:, 0].tolist() == [0, 0]


def test_rho_gate_controls_only_st_selection_gradient() -> None:
    batch = encoded_batch()
    closed = _model().train()
    closed_output = closed(
        batch,
        RolloutConfig(
            max_steps=2,
            selection_mode="learned",
            straight_through=True,
            rho_gate=0.0,
        ),
    )
    closed_output.final_query[:, 0].sum().backward()
    closed_utility = sum(
        float(parameter.grad.abs().sum())
        for parameter in closed.utility.parameters()
        if parameter.grad is not None
    )
    opened = _model().train()
    opened_output = opened(
        batch,
        RolloutConfig(
            max_steps=2,
            selection_mode="learned",
            straight_through=True,
            rho_gate=0.25,
        ),
    )
    opened_output.final_query[:, 0].sum().backward()
    opened_utility = sum(
        float(parameter.grad.abs().sum())
        for parameter in opened.utility.parameters()
        if parameter.grad is not None
    )
    assert closed_utility == 0.0
    assert opened_utility > 0.0
    assert opened.executor.film.weight.grad is not None


def test_stage_rollout_contracts_match_schedule_semantics() -> None:
    schedule = CanonicalV4Curriculum()
    for epoch in (1, 9, 15, 27, 41, 47, 53):
        state = schedule.state_for_epoch(epoch)
        config = EngineConfig(
            stage=state.phase,
            horizon=state.horizon,
            oracle_mix=state.oracle_mix,
            straight_through=state.straight_through,
            selection_temperature=state.selection_temperature,
            rho_gate=state.rho_gate,
            exploration_probability=state.exploration_probability,
        )
        rollout = config.rollout()
        assert rollout.max_steps == state.horizon
        assert rollout.straight_through == state.straight_through
        assert rollout.exploration_probability == state.exploration_probability


def test_main_config_enables_full_canonical_schedule_and_matched_budget() -> None:
    path = Path(__file__).parents[1] / "conf" / "taper_mag_v4_base.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["training"]["curriculum_mode"] == "canonical_v4"
    assert config["training"]["epochs"] == 60
    assert config["training"]["max_optimizer_updates"] == 4260


def test_resume_rejects_changed_curriculum_or_update_budget() -> None:
    saved = {
        "config": {
            "training": {
                "curriculum_mode": "canonical_v4",
                "epochs": 60,
                "max_optimizer_updates": 4260,
            }
        }
    }
    current = {
        "training": {
            "curriculum_mode": "canonical_v4",
            "epochs": 60,
            "max_optimizer_updates": 4260,
        }
    }
    verify_resume_schedule_config(saved, current)
    changed = {"training": {**current["training"], "max_optimizer_updates": 4000}}
    with pytest.raises(RuntimeError, match="schedule config mismatch"):
        verify_resume_schedule_config(saved, changed)
