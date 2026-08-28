from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from torch import Tensor


SelectionMode = Literal["learned", "uniform", "soft", "frozen_order"]


@dataclass(frozen=True, slots=True)
class RolloutConfig:
    min_steps: int = 1
    max_steps: int = 4
    step_cost: float = 0.0
    selection_mode: SelectionMode = "learned"
    straight_through: bool = False
    selection_temperature: float = 1.0
    rho_gate: float = 0.0
    exploration_probability: float = 0.0

    def validate(self) -> None:
        if self.min_steps != 1:
            raise ValueError("Canonical TAPER-MAG requires min_steps=1")
        if not self.min_steps <= self.max_steps <= 4:
            raise ValueError("Canonical horizon must satisfy 1 <= max_steps <= 4")
        if self.selection_mode not in {"learned", "uniform", "soft", "frozen_order"}:
            raise ValueError(f"Unsupported selection_mode: {self.selection_mode}")
        if self.straight_through and self.selection_mode != "learned":
            raise ValueError("Straight-through selection is defined only for learned rollout")
        if self.selection_temperature <= 0:
            raise ValueError("selection_temperature must be positive")
        if not 0 <= self.rho_gate <= 1:
            raise ValueError("rho_gate must be in [0,1]")
        if not 0 <= self.exploration_probability <= 1:
            raise ValueError("exploration_probability must be in [0,1]")
        if self.exploration_probability > 0 and (
            self.selection_mode != "learned" or self.straight_through
        ):
            raise ValueError("Exploration requires learned hard non-ST rollout")


@dataclass(frozen=True, slots=True)
class RolloutTrace:
    actions: Tensor
    active: Tensor
    predicted_gain: Tensor
    action_values: Tensor
    current_queries: Tensor
    candidate_queries: Tensor
    support_mass: Tensor
    delta_norm: Tensor


@dataclass(frozen=True, slots=True)
class TaperOutput:
    final_query: Tensor
    trace: RolloutTrace
    diagnostics: dict[str, Tensor]
