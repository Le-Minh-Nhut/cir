from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from training.taper_mag_engine import CurriculumStage
from training.taper_mag_audit import validate_teacher_shadow_report


CurriculumMode = Literal["canonical_v4", "manual"]
HealthGateMode = Literal["manual_approval"]


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


@dataclass(frozen=True, slots=True)
class CurriculumGateState:
    """Explicit, reviewed approvals for qualitative V4 health gates."""

    approved_transitions: frozenset[str] = frozenset()
    bypass_for_smoke: bool = False
    mode: HealthGateMode = "manual_approval"
    teacher_shadow_report_sha256: str | None = None

    def checkpoint_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "approved_transitions": sorted(self.approved_transitions),
            "bypass_for_smoke": self.bypass_for_smoke,
            "teacher_shadow_report_sha256": self.teacher_shadow_report_sha256,
        }


class CanonicalV4Curriculum:
    """Central V4 schedule whose phase boundaries require explicit health approval."""

    first_epoch = 1
    last_epoch = 60
    transitions = (
        (9, "actor_warmup_passed", CurriculumStage.ACTOR_WARMUP, CurriculumStage.CRITIC_WARMUP),
        (15, "critic_warmup_passed", CurriculumStage.CRITIC_WARMUP, CurriculumStage.DAGGER_T2),
        (27, "dagger_t2_passed", CurriculumStage.DAGGER_T2, CurriculumStage.ST_BRIDGE),
        (41, "st_bridge_passed", CurriculumStage.ST_BRIDGE, CurriculumStage.PREDICTED_T4),
        (53, "predicted_t4_passed", CurriculumStage.PREDICTED_T4, CurriculumStage.HARDEN),
    )
    valid_gate_names = frozenset(item[1] for item in transitions)

    def __init__(
        self,
        *,
        step_cost: float = 0.0,
        gate_state: CurriculumGateState | None = None,
    ) -> None:
        self.step_cost = float(step_cost)
        self.gate_state = gate_state or CurriculumGateState()
        unknown = self.gate_state.approved_transitions - self.valid_gate_names
        if unknown:
            raise ValueError(f"Unknown canonical V4 health-gate approvals: {sorted(unknown)}")

    def _require_health_gates(self, epoch: int) -> None:
        if self.gate_state.bypass_for_smoke:
            return
        for boundary, gate, previous, following in self.transitions:
            if epoch >= boundary and gate not in self.gate_state.approved_transitions:
                raise RuntimeError(
                    f"{previous.value} reached canonical boundary at epoch {boundary - 1}, "
                    f"but health gate '{gate}' is not approved. Refusing to advance to "
                    f"{following.value}. Add it to training.approved_health_gates only "
                    "after explicit health review, or set "
                    "training.bypass_health_gates_for_smoke=true for a non-scientific smoke run."
                )

    def state_for_epoch(self, epoch: int) -> CurriculumState:
        if not self.first_epoch <= epoch <= self.last_epoch:
            raise ValueError("canonical_v4 epoch must be in [1,60]")
        self._require_health_gates(epoch)
        return self.reference_state_for_epoch(epoch)

    def reference_state_for_epoch(self, epoch: int) -> CurriculumState:
        """Return numeric schedule values without granting phase promotion."""
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
                "soft", self.step_cost,
            )
        if epoch <= 26:
            return CurriculumState(
                CurriculumStage.DAGGER_T2, 2, _linear(epoch, 15, 26, 0.8, 0.3),
                False, 1.0, 0.0, 0.0, "learned", self.step_cost,
            )
        if epoch <= 40:
            return CurriculumState(
                CurriculumStage.ST_BRIDGE, 3, _linear(epoch, 27, 40, 0.3, 0.0),
                True, _linear(epoch, 27, 40, 1.0, 0.5),
                _linear(epoch, 27, 40, 0.0, 0.25), 0.0, "learned", self.step_cost,
            )
        if epoch <= 46:
            return CurriculumState(
                CurriculumStage.PREDICTED_T4, 4, 0.0, False, 0.5, 0.25, 0.05,
                "learned", self.step_cost,
            )
        if epoch <= 52:
            return CurriculumState(
                CurriculumStage.PREDICTED_T4, 4, 0.0, False,
                _linear(epoch, 47, 52, 0.5, 0.25), 0.25,
                _linear(epoch, 47, 52, 0.05, 0.0), "learned", self.step_cost,
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
        gate_state: CurriculumGateState | None = None,
    ) -> None:
        if mode not in {"canonical_v4", "manual"}:
            raise ValueError(f"Unsupported curriculum_mode: {mode}")
        if mode == "manual" and manual_state is None:
            raise ValueError("manual curriculum requires an explicit state")
        self.mode = mode
        self.gate_state = gate_state or CurriculumGateState()
        self.canonical = CanonicalV4Curriculum(
            step_cost=step_cost, gate_state=self.gate_state
        )
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
            gate_mode = str(training.get("health_gate_mode", "manual_approval"))
            if gate_mode != "manual_approval":
                raise ValueError(f"Unsupported health_gate_mode: {gate_mode}")
            approvals = frozenset(
                str(item) for item in training.get("approved_health_gates", [])
            )
            bypass = bool(training.get("bypass_health_gates_for_smoke", False))
            shadow_hash = None
            if "actor_warmup_passed" in approvals and not bypass:
                report_path = training.get("teacher_shadow_report")
                if not report_path:
                    raise ValueError(
                        "actor_warmup_passed requires training.teacher_shadow_report"
                    )
                shadow_hash = validate_teacher_shadow_report(report_path)
            gate_state = CurriculumGateState(
                approved_transitions=approvals,
                bypass_for_smoke=bypass,
                teacher_shadow_report_sha256=shadow_hash,
            )
            return cls("canonical_v4", step_cost=step_cost, gate_state=gate_state)
        if mode != "manual":
            raise ValueError(f"Unsupported curriculum_mode: {mode}")
        stage = CurriculumStage(str(training["stage"]))
        straight_through = bool(
            training.get("straight_through", stage == CurriculumStage.ST_BRIDGE)
        )
        if stage in {CurriculumStage.ACTOR_WARMUP, CurriculumStage.UTILITY_SHADOW}:
            selection_mode = "uniform"
        elif stage == CurriculumStage.CRITIC_WARMUP:
            selection_mode = "soft"
        else:
            selection_mode = "learned"
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

    def checkpoint_state(self, epoch: int) -> dict[str, Any]:
        state = self.state_for_epoch(epoch)
        next_phase = state.phase
        if self.mode == "canonical_v4" and epoch < self.canonical.last_epoch:
            try:
                next_phase = self.state_for_epoch(epoch + 1).phase
            except RuntimeError:
                next_phase = state.phase
        return {
            "epoch": epoch,
            "current_phase": state.phase.value,
            "next_allowed_phase": next_phase.value,
            "schedule": state.checkpoint_dict(),
            "health_gates": self.gate_state.checkpoint_dict(),
        }

    def verify_checkpoint(self, epoch: int, saved: dict[str, Any]) -> None:
        if int(saved.get("epoch", -1)) != epoch:
            raise RuntimeError("Checkpoint curriculum epoch mismatch")
        saved_gates = saved.get("health_gates")
        if not isinstance(saved_gates, dict):
            raise RuntimeError("Checkpoint is missing curriculum health-gate state")
        if saved_gates.get("mode") != self.gate_state.mode:
            raise RuntimeError("Checkpoint health-gate mode mismatch")
        if bool(saved_gates.get("bypass_for_smoke")) != self.gate_state.bypass_for_smoke:
            raise RuntimeError("Checkpoint health-gate smoke-bypass mismatch")
        approved_at_save = frozenset(saved_gates.get("approved_transitions", []))
        if not approved_at_save.issubset(self.gate_state.approved_transitions):
            raise RuntimeError("Current config removed a health-gate approval saved in checkpoint")
        saved_shadow = saved_gates.get("teacher_shadow_report_sha256")
        if saved_shadow is not None and saved_shadow != self.gate_state.teacher_shadow_report_sha256:
            raise RuntimeError("Teacher-shadow report hash changed after health-gate approval")
        expected = self.state_for_epoch(epoch).checkpoint_dict()
        if saved.get("schedule") != expected or saved.get("current_phase") != expected["phase"]:
            raise RuntimeError(
                f"Checkpoint curriculum state mismatch: expected={expected}, saved={saved}"
            )
