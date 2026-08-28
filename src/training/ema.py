from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch
from torch import Tensor, nn


class ModelEMA:
    """Non-trainable EMA shadow for TAPER and its trainable text backbone."""

    def __init__(self, decay: float = 0.999) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0,1)")
        self.decay = float(decay)
        self.active = False
        self.num_updates = 0
        self.model_parameters: dict[str, Tensor] = {}
        self.model_buffers: dict[str, Tensor] = {}
        self.backbone_parameters: dict[str, Tensor] = {}

    @staticmethod
    def _trainable_backbone(backbone: nn.Module) -> dict[str, nn.Parameter]:
        return {
            name: parameter
            for name, parameter in backbone.named_parameters()
            if parameter.requires_grad
        }

    def activate(self, model: nn.Module, backbone: nn.Module) -> None:
        if self.active:
            raise RuntimeError("EMA is already active")
        self.model_parameters = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        self.model_buffers = {
            name: buffer.detach().clone() for name, buffer in model.named_buffers()
        }
        self.backbone_parameters = {
            name: parameter.detach().clone()
            for name, parameter in self._trainable_backbone(backbone).items()
        }
        self.active = True
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module, backbone: nn.Module) -> None:
        if not self.active:
            raise RuntimeError("EMA must be activated before update")
        live_model = dict(model.named_parameters())
        live_buffers = dict(model.named_buffers())
        live_backbone = self._trainable_backbone(backbone)
        self._require_same_keys("model parameters", self.model_parameters, live_model)
        self._require_same_keys("model buffers", self.model_buffers, live_buffers)
        self._require_same_keys("backbone parameters", self.backbone_parameters, live_backbone)
        one_minus_decay = 1.0 - self.decay
        for name, shadow in self.model_parameters.items():
            shadow.mul_(self.decay).add_(live_model[name].detach(), alpha=one_minus_decay)
        for name, shadow in self.backbone_parameters.items():
            shadow.mul_(self.decay).add_(live_backbone[name].detach(), alpha=one_minus_decay)
        for name, shadow in self.model_buffers.items():
            shadow.copy_(live_buffers[name].detach())
        self.num_updates += 1

    @staticmethod
    def _require_same_keys(
        label: str, shadow: dict[str, Tensor], live: dict[str, Tensor]
    ) -> None:
        if set(shadow) != set(live):
            raise RuntimeError(f"EMA {label} disagree with live model")

    @contextmanager
    def average_parameters(
        self, model: nn.Module, backbone: nn.Module
    ) -> Iterator[None]:
        if not self.active:
            yield
            return
        live_model = dict(model.named_parameters())
        live_buffers = dict(model.named_buffers())
        live_backbone = self._trainable_backbone(backbone)
        self._require_same_keys("model parameters", self.model_parameters, live_model)
        self._require_same_keys("model buffers", self.model_buffers, live_buffers)
        self._require_same_keys("backbone parameters", self.backbone_parameters, live_backbone)
        backups = {
            "model_parameters": {
                name: value.detach().clone() for name, value in live_model.items()
            },
            "model_buffers": {
                name: value.detach().clone() for name, value in live_buffers.items()
            },
            "backbone_parameters": {
                name: value.detach().clone() for name, value in live_backbone.items()
            },
        }
        try:
            with torch.no_grad():
                for name, shadow in self.model_parameters.items():
                    live_model[name].copy_(shadow)
                for name, shadow in self.model_buffers.items():
                    live_buffers[name].copy_(shadow)
                for name, shadow in self.backbone_parameters.items():
                    live_backbone[name].copy_(shadow)
            yield
        finally:
            with torch.no_grad():
                for name, value in backups["model_parameters"].items():
                    live_model[name].copy_(value)
                for name, value in backups["model_buffers"].items():
                    live_buffers[name].copy_(value)
                for name, value in backups["backbone_parameters"].items():
                    live_backbone[name].copy_(value)

    def state_dict(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "decay": self.decay,
            "num_updates": self.num_updates,
            "model_parameters": {
                name: value.detach().cpu() for name, value in self.model_parameters.items()
            },
            "model_buffers": {
                name: value.detach().cpu() for name, value in self.model_buffers.items()
            },
            "backbone_parameters": {
                name: value.detach().cpu() for name, value in self.backbone_parameters.items()
            },
        }

    def load_state_dict(
        self,
        state: dict[str, Any] | None,
        model: nn.Module,
        backbone: nn.Module,
        *,
        expected_active: bool,
    ) -> None:
        if state is None:
            raise RuntimeError("Checkpoint is missing EMA state")
        if float(state.get("decay", -1.0)) != self.decay:
            raise RuntimeError(
                f"Checkpoint EMA decay {state.get('decay')} != configured {self.decay}"
            )
        if bool(state.get("active")) != expected_active:
            raise RuntimeError("Checkpoint EMA active state disagrees with curriculum phase")
        if not expected_active:
            if any(
                state.get(key)
                for key in (
                    "model_parameters",
                    "model_buffers",
                    "backbone_parameters",
                )
            ) or int(state.get("num_updates", 0)) != 0:
                raise RuntimeError("Inactive checkpoint EMA contains unexpected shadow state")
            self.active = False
            self.num_updates = int(state.get("num_updates", 0))
            self.model_parameters = {}
            self.model_buffers = {}
            self.backbone_parameters = {}
            return
        live_model = dict(model.named_parameters())
        live_buffers = dict(model.named_buffers())
        live_backbone = self._trainable_backbone(backbone)
        stored_model = state.get("model_parameters", {})
        stored_buffers = state.get("model_buffers", {})
        stored_backbone = state.get("backbone_parameters", {})
        self._require_same_keys("model parameters", stored_model, live_model)
        self._require_same_keys("model buffers", stored_buffers, live_buffers)
        self._require_same_keys("backbone parameters", stored_backbone, live_backbone)
        self.model_parameters = {
            name: value.detach().to(live_model[name].device).clone()
            for name, value in stored_model.items()
        }
        self.model_buffers = {
            name: value.detach().to(live_buffers[name].device).clone()
            for name, value in stored_buffers.items()
        }
        self.backbone_parameters = {
            name: value.detach().to(live_backbone[name].device).clone()
            for name, value in stored_backbone.items()
        }
        self.active = True
        self.num_updates = int(state["num_updates"])


def ema_required_for_phase(phase: str) -> bool:
    return phase not in {"actor_warmup"}
