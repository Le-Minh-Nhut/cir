from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from training.taper_mag_engine import CurriculumStage


CurriculumMode = Literal["canonical_v4", "manual"]


def _linear(epoch: int, start_epoch: int, end_epoch: int, start: float, end: float) -> float:
    if start_epoch == end_epoch:
        return end
    progress = (epoch - start_epoch) / (end_epoch - start_epoch)
    return start + progress * (end - start)


@dataclass(frozen=True, slots=True)
class CurriculumState:
    phase: CurriculumStage
    horizon: int
    oracle_mix: float
    straight_through: bool
    selection_temperature: float
    rho_gate: float
    exploration_probability: float
    selection_mode: str
    step_cost: float

    def checkpoint_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["phase"] = self.phase.value
        return result


class CanonicalV4Curriculum:
    """Central deterministic 1-indexed epoch schedule from V4 section 15."""

    first_epoch = 1
    last_epoch = 60

    def __init__(self, *, step_cost: float = 0.0) -> None:
        self.step_cost = float(step_cost)

    def state_for_epoch(self, epoch: int) -> CurriculumState:
        if not self.first_epoch <= epoch <= self.last_epoch:
            raise ValueError("canonical_v4 epoch must be in [1,60]")
        if epoch <= 8:
            return CurriculumState(
                CurriculumStage.ACTOR_WARMUP, 1, 0.0, False, 1.0, 0.0, 0.0,
                "uniform", self.step_cost,
            )
        if epoch <= 14:
            return CurriculumState(
                CurriculumStage.CRITIC_WARMUP, 1, 0.0, False, 1.0, 0.0, 0.0,
                "uniform", self.step_cost,
            )
        if epoch <= 26:
            return CurriculumState(
                CurriculumStage.DAGGER_T2,
                2,
                _linear(epoch, 15, 26, 0.8, 0.3),
                False,
                1.0,
                0.0,
                0.0,
                "learned",
                self.step_cost,
            )
        if epoch <= 40:
            return CurriculumState(
                CurriculumStage.ST_BRIDGE,
                3,
                _linear(epoch, 27, 40, 0.3, 0.0),
                True,
                _linear(epoch, 27, 40, 1.0, 0.5),
                _linear(epoch, 27, 40, 0.0, 0.25),
                0.0,
                "learned",
                self.step_cost,
            )
        if epoch <= 46:
            return CurriculumState(
                CurriculumStage.PREDICTED_T4, 4, 0.0, False, 0.5, 0.25, 0.05,
                "learned", self.step_cost,
            )
        if epoch <= 52:
            return CurriculumState(
                CurriculumStage.PREDICTED_T4,
                4,
                0.0,
                False,
                _linear(epoch, 47, 52, 0.5, 0.25),
                0.25,
                _linear(epoch, 47, 52, 0.05, 0.0),
                "learned",
                self.step_cost,
            )
        return CurriculumState(
            CurriculumStage.HARDEN, 4, 0.0, False, 0.25, 0.25, 0.0,
            "learned", self.step_cost,
        )


class CurriculumScheduler:
    def __init__(
        self,
        mode: CurriculumMode,
        *,
        step_cost: float,
        manual_state: CurriculumState | None = None,
    ) -> None:
        if mode not in {"canonical_v4", "manual"}:
            raise ValueError(f"Unsupported curriculum_mode: {mode}")
        if mode == "manual" and manual_state is None:
            raise ValueError("manual curriculum requires an explicit state")
        self.mode = mode
        self.canonical = CanonicalV4Curriculum(step_cost=step_cost)
        self.manual_state = manual_state

    @classmethod
    def from_config(
        cls,
        training: dict[str, Any],
        *,
        step_cost: float,
    ) -> CurriculumScheduler:
        mode = str(training.get("curriculum_mode", "manual"))
        if mode == "canonical_v4":
            return cls("canonical_v4", step_cost=step_cost)
        if mode != "manual":
            raise ValueError(f"Unsupported curriculum_mode: {mode}")
        stage = CurriculumStage(str(training["stage"]))
        straight_through = bool(
            training.get("straight_through", stage == CurriculumStage.ST_BRIDGE)
        )
        selection_mode = (
            "uniform"
            if stage in {
                CurriculumStage.ACTOR_WARMUP,
                CurriculumStage.UTILITY_SHADOW,
                CurriculumStage.CRITIC_WARMUP,
            }
            else "learned"
        )
        manual = CurriculumState(
            phase=stage,
            horizon=int(training["horizon"]),
            oracle_mix=float(training.get("oracle_mix", 0.0)),
            straight_through=straight_through,
            selection_temperature=float(training.get("selection_temperature", 1.0)),
            rho_gate=float(training.get("rho_gate", 0.0)),
            exploration_probability=float(training.get("exploration_probability", 0.0)),
            selection_mode=selection_mode,
            step_cost=step_cost,
        )
        return cls("manual", step_cost=step_cost, manual_state=manual)

    def state_for_epoch(self, epoch: int) -> CurriculumState:
        if self.mode == "canonical_v4":
            return self.canonical.state_for_epoch(epoch)
        assert self.manual_state is not None
        if epoch <= 0:
            raise ValueError("epoch must be 1-indexed and positive")
        return self.manual_state

    def verify_checkpoint(self, epoch: int, saved: dict[str, Any]) -> None:
        expected = self.state_for_epoch(epoch).checkpoint_dict()
        if saved != expected:
            raise RuntimeError(
                f"Checkpoint curriculum state mismatch: expected={expected}, saved={saved}"
            )
