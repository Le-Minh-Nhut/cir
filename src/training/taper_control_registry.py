from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ControlRegistryEntry:
    experiment_id: str
    name: str
    purpose: str
    available_in_this_pass: bool


TAPER_EXPERIMENT_MATRIX = (
    ControlRegistryEntry("M0", "reference-only", "image shortcut", True),
    ControlRegistryEntry("M1", "text-only", "text shortcut", True),
    ControlRegistryEntry("M2", "normalized scalar-gated sum", "simple composition floor", True),
    ControlRegistryEntry("M3", "one-shot gated MLP combiner", "principal benchmark control", True),
    ControlRegistryEntry("M4", "TAPER T=1", "architecture before multi-step", True),
    ControlRegistryEntry("M5", "matched recurrent global editor", "mechanism versus compute", False),
    ControlRegistryEntry("M6", "uniform/all execution", "learned utility necessity", False),
    ControlRegistryEntry("M7", "learned utility T=1", "selection without recurrence", False),
    ControlRegistryEntry("M8", "frozen t=0 ordering", "state dependence control", True),
    ControlRegistryEntry("M9", "dynamic fixed-T without STOP", "STOP isolation", False),
    ControlRegistryEntry("M10", "full dynamic TAPER", "canonical Phase-1 system", True),
)


def control_entry(experiment_id: str) -> ControlRegistryEntry:
    for entry in TAPER_EXPERIMENT_MATRIX:
        if entry.experiment_id == experiment_id.upper():
            return entry
    raise ValueError(f"Unknown TAPER experiment ID: {experiment_id}")
