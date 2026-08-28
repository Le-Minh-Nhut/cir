from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import MethodType

import pytest
import torch
import yaml

from models.taper_mag.model import TaperMAG, TaperMAGConfig, candidate_mixture
from models.taper_mag.rollout import RolloutConfig
from test_taper_mag_actor import encoded_batch
from train_taper_mag import verify_resume_schedule_config
from training.taper_mag_curriculum import (
    CanonicalV4Curriculum,
    CurriculumGateState,
    CurriculumScheduler,
)
from training.taper_mag_engine import CurriculumStage, EngineConfig


def test_exact_canonical_v4_epoch_boundaries() -> None:
    schedule = CanonicalV4Curriculum(
        gate_state=CurriculumGateState(
            approved_transitions=CanonicalV4Curriculum.valid_gate_names
        )
    )
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
    assert schedule.state_for_epoch(1).selection_mode == "uniform"
    assert schedule.state_for_epoch(9).selection_mode == "soft"


def test_health_gate_boundary_requires_explicit_approval() -> None:
    default = CanonicalV4Curriculum()
    assert default.state_for_epoch(8).phase == CurriculumStage.ACTOR_WARMUP
    with pytest.raises(RuntimeError, match="actor_warmup_passed"):
        default.state_for_epoch(9)

    approved = CanonicalV4Curriculum(
        gate_state=CurriculumGateState(
            approved_transitions=frozenset({"actor_warmup_passed"})
        )
    )
    assert approved.state_for_epoch(9).phase == CurriculumStage.CRITIC_WARMUP


def test_health_gate_smoke_bypass_is_explicit_and_default_is_enforced() -> None:
    default = CurriculumScheduler.from_config(
        {"curriculum_mode": "canonical_v4"}, step_cost=0.0
    )
    assert not default.gate_state.bypass_for_smoke
    with pytest.raises(RuntimeError, match="non-scientific smoke run"):
        default.state_for_epoch(60)
    smoke = CurriculumScheduler.from_config(
        {
            "curriculum_mode": "canonical_v4",
            "bypass_health_gates_for_smoke": True,
        },
        step_cost=0.0,
    )
    assert smoke.state_for_epoch(60).phase == CurriculumStage.HARDEN


def test_curriculum_checkpoint_consistency_and_manual_mode() -> None:
    scheduler = CurriculumScheduler.from_config(
        {
            "curriculum_mode": "canonical_v4",
            "approved_health_gates": sorted(CanonicalV4Curriculum.valid_gate_names),
        },
        step_cost=0.1,
    )
    state = scheduler.state_for_epoch(27)
    saved = scheduler.checkpoint_state(27)
    scheduler.verify_checkpoint(27, saved)
    with pytest.raises(RuntimeError, match="curriculum state mismatch"):
        changed = dict(saved)
        changed["schedule"] = replace(state, selection_temperature=0.7).checkpoint_dict()
        scheduler.verify_checkpoint(27, changed)
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


def test_gate_checkpoint_preserves_approvals_and_allows_explicit_new_approval() -> None:
    actor = CurriculumScheduler.from_config(
        {"curriculum_mode": "canonical_v4"}, step_cost=0.0
    )
    saved = actor.checkpoint_state(8)
    assert saved["current_phase"] == "actor_warmup"
    assert saved["next_allowed_phase"] == "actor_warmup"
    resumed = CurriculumScheduler.from_config(
        {
            "curriculum_mode": "canonical_v4",
            "approved_health_gates": ["actor_warmup_passed"],
        },
        step_cost=0.0,
    )
    resumed.verify_checkpoint(8, saved)
    assert resumed.state_for_epoch(9).phase == CurriculumStage.CRITIC_WARMUP
    approved_saved = resumed.checkpoint_state(9)
    with pytest.raises(RuntimeError, match="removed a health-gate approval"):
        actor.verify_checkpoint(9, approved_saved)


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
    schedule = CanonicalV4Curriculum(
        gate_state=CurriculumGateState(
            approved_transitions=CanonicalV4Curriculum.valid_gate_names
        )
    )
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
    assert EngineConfig(
        stage=CurriculumStage.ACTOR_WARMUP, horizon=1
    ).rollout().selection_mode == "uniform"
    assert EngineConfig(
        stage=CurriculumStage.CRITIC_WARMUP, horizon=1
    ).rollout().selection_mode == "soft"


def test_soft_candidate_mixture_uses_utility_forward_but_detaches_score_gradient() -> None:
    candidates = torch.tensor([[[[0.0]], [[1.0]], [[0.0]], [[0.0]]]], requires_grad=True)
    values = torch.tensor([[0.0, 6.0, -1.0, -2.0]], requires_grad=True)
    uniform, uniform_weights = candidate_mixture(
        candidates,
        values,
        mode="uniform",
        temperature=1.0,
        detach_action_values=False,
    )
    soft, soft_weights = candidate_mixture(
        candidates,
        values,
        mode="soft",
        temperature=1.0,
        detach_action_values=True,
    )
    torch.testing.assert_close(uniform_weights, torch.full_like(values, 0.25))
    torch.testing.assert_close(soft_weights.sum(dim=-1), torch.ones(1))
    assert torch.isfinite(soft_weights).all()
    assert soft_weights.argmax(dim=-1).item() == 1
    assert not torch.allclose(soft, uniform)
    assert abs(float(soft.item()) - 1.0) < abs(float(uniform.item()) - 1.0)
    soft.sum().backward()
    assert values.grad is None or values.grad.abs().sum() == 0
    assert candidates.grad is not None and candidates.grad.abs().sum() > 0


def test_main_config_enables_full_canonical_schedule_and_matched_budget() -> None:
    path = Path(__file__).parents[1] / "conf" / "taper_mag_v4_base.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["training"]["curriculum_mode"] == "canonical_v4"
    assert config["training"]["epochs"] == 60
    assert config["training"]["max_optimizer_updates"] == 4260
    assert config["training"]["health_gate_mode"] == "manual_approval"
    assert config["training"]["approved_health_gates"] == []
    assert config["training"]["bypass_health_gates_for_smoke"] is False
    assert config["optimizer"]["ema_decay"] == pytest.approx(0.999)


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
