from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from training.checkpointing import load_checkpoint, save_checkpoint
from training.ema import ModelEMA, ema_required_for_phase


def _modules() -> tuple[nn.Linear, nn.Sequential]:
    model = nn.Linear(1, 1, bias=False)
    backbone = nn.Sequential(nn.Linear(1, 1, bias=False), nn.Linear(1, 1, bias=False))
    next(backbone[0].parameters()).requires_grad_(False)
    return model, backbone


def test_ema_activation_update_formula_and_no_grad_shadow() -> None:
    model, backbone = _modules()
    with torch.no_grad():
        model.weight.fill_(1.0)
        backbone[1].weight.fill_(1.0)
    ema = ModelEMA(0.999)
    assert not ema.active
    ema.activate(model, backbone)
    assert ema.num_updates == 0
    assert all(not value.requires_grad for value in ema.model_parameters.values())
    assert all(not value.requires_grad for value in ema.backbone_parameters.values())
    with torch.no_grad():
        model.weight.fill_(2.0)
        backbone[1].weight.fill_(2.0)
    ema.update(model, backbone)
    torch.testing.assert_close(ema.model_parameters["weight"], torch.tensor([[1.001]]))
    torch.testing.assert_close(
        ema.backbone_parameters["1.weight"], torch.tensor([[1.001]])
    )
    assert ema.num_updates == 1


def test_ema_validation_swap_restores_live_parameters() -> None:
    model, backbone = _modules()
    with torch.no_grad():
        model.weight.fill_(1.0)
        backbone[1].weight.fill_(1.0)
    ema = ModelEMA(0.999)
    ema.activate(model, backbone)
    with torch.no_grad():
        model.weight.fill_(3.0)
        backbone[1].weight.fill_(4.0)
    with ema.average_parameters(model, backbone):
        torch.testing.assert_close(model.weight, torch.tensor([[1.0]]))
        torch.testing.assert_close(backbone[1].weight, torch.tensor([[1.0]]))
    torch.testing.assert_close(model.weight, torch.tensor([[3.0]]))
    torch.testing.assert_close(backbone[1].weight, torch.tensor([[4.0]]))


def test_ema_phase_activation_contract() -> None:
    assert not ema_required_for_phase("actor_warmup")
    assert ema_required_for_phase("critic_warmup")
    assert ema_required_for_phase("dagger_t2")
    assert ema_required_for_phase("harden")


def test_ema_checkpoint_roundtrip_and_decay_validation(tmp_path) -> None:
    model, backbone_model = _modules()
    backbone = SimpleNamespace(model=backbone_model)
    optimizer = torch.optim.SGD(
        [model.weight, backbone_model[1].weight], lr=0.1
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    ema = ModelEMA(0.999)
    ema.activate(model, backbone_model)
    with torch.no_grad():
        model.weight.add_(1.0)
    ema.update(model, backbone_model)
    checkpoint = tmp_path / "ema.ckpt"
    save_checkpoint(
        checkpoint,
        model=model,
        backbone=backbone,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=8,
        global_step=7,
        stage="critic_warmup",
        curriculum_state={"epoch": 9},
        resolved_config={"optimizer": {"ema_decay": 0.999}},
        manifest_hashes={},
        best_metrics={},
        ema_state=ema.state_dict(),
    )
    payload = load_checkpoint(
        checkpoint,
        model=model,
        backbone=backbone,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    restored = ModelEMA(0.999)
    restored.load_state_dict(
        payload["ema"], model, backbone_model, expected_active=True
    )
    assert restored.active and restored.num_updates == 1
    for name in ema.model_parameters:
        torch.testing.assert_close(
            restored.model_parameters[name], ema.model_parameters[name]
        )
    with pytest.raises(RuntimeError, match="EMA decay"):
        ModelEMA(0.9).load_state_dict(
            payload["ema"], model, backbone_model, expected_active=True
        )
