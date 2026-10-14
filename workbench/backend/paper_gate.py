"""The paper-metric gate: does a reproduced run agree with the published numbers?

The gate is deliberately separate from any evaluator. It takes a reproduced metric set and a paper
metric set, decides whether they describe the same experiment, and only then compares numbers. A
number that comes from a different protocol, a different checkpoint condition, or a mock run is not
evidence about reproduction no matter how close it is.

Metric scales are normalised before comparison: papers publish `57.07` and result files carry
`0.5707` for the same Recall@K depending on the author's convention, and comparing those directly
would fail every run. The normaliser is explicit rather than automatic so a genuinely wrong value
cannot be absorbed by it.

The tolerance is the user's accepted rule: 5 percent RELATIVE error, evaluated per metric. A run
passes only when every comparable metric passes; the mean cannot carry a broken R@10.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

#: The user's acceptance rule. Relative percentage error, on every comparable metric.
TOLERANCE_PCT = 5.0

METRIC_KEYS = ("r10", "r50", "mean")

#: Result states. `BLOCKED_*` means the comparison was refused, not that it failed a tolerance.
PASS_WITHIN_5_PERCENT = "PASS_WITHIN_5_PERCENT"
PASS_WITHIN_5_PERCENT_VARIANT = "PASS_WITHIN_5_PERCENT_VARIANT"
ALERT_METRIC_DEVIATION = "ALERT_METRIC_DEVIATION"
BLOCKED_PROTOCOL_MISMATCH = "BLOCKED_PROTOCOL_MISMATCH"
BLOCKED_CHECKPOINT_MISMATCH = "BLOCKED_CHECKPOINT_MISMATCH"
BLOCKED_DATA_PROVENANCE = "BLOCKED_DATA_PROVENANCE"
BLOCKED_SOURCE = "BLOCKED_SOURCE"
BLOCKED_ENVIRONMENT = "BLOCKED_ENVIRONMENT"
NOT_RUN = "NOT_RUN"
METRIC_REFERENCE_UNVERIFIED = "METRIC_REFERENCE_UNVERIFIED"

#: Provenance keys that must agree before any number is compared. Each is the name of a field in the
#: run metadata; a missing value is treated as unknown, and unknown never matches.
COMPARABILITY_FIELDS = (
    "protocol_id",
    "checkpoint_id",
    "checkpoint_training_noise_pct",
    "evaluation_noise_pct",
    "split",
    "dataset",
)

#: Fields that label a run as a declared variant rather than strict parity. Staged aliases are the
#: current case: the bytes are source-compatible but their identity with the author's file is unknown.
VARIANT_FIELDS = ("caption_alias_staged", "instrumentation_patch_sha256")
VARIANT_KEYS = ("caption_alias_staged",)


@dataclass(frozen=True)
class MetricComparison:
    metric: str
    paper: float
    reproduced: float | None
    absolute_error_pp: float | None
    relative_error_pct: float | None
    within_tolerance: bool | None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GateVerdict:
    status: str
    passed: bool
    variant: bool
    comparisons: list[MetricComparison] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "passed": self.passed,
            "variant": self.variant,
            "blockers": list(self.blockers),
            "detail": self.detail,
            "comparisons": [c.to_dict() for c in self.comparisons],
        }


def normalize_scale(value: float | None) -> float | None:
    """Put a Recall value on the percentage scale the papers publish.

    Papers print Recall@K as `57.07`; result files store the same quantity as `0.5707`. A value in
    [0, 1) is a fraction and is scaled by 100; anything else is already a percentage. Exactly `1.0`
    is not scaled: a result file carrying `1.0` under the fraction convention means 1 percent, and
    under the percentage convention means 1 percent too, so scaling it would manufacture a 100
    percent score out of a near-zero one. Either way the run is then compared honestly.
    """
    if value is None:
        return None
    number = float(value)
    if 0.0 <= number < 1.0:
        return number * 100.0
    return number


def _is_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return float(value) == float(value) and abs(float(value)) != float("inf")


def compare_metric(metric: str, paper: float | None, reproduced: float | None,
                   tolerance_pct: float = TOLERANCE_PCT) -> MetricComparison:
    """Compare one metric under the relative-error rule, with an explicit zero-paper policy."""
    if paper is None:
        return MetricComparison(metric, float("nan"), reproduced, None, None, None, "no verified paper value")
    paper_value = normalize_scale(paper)
    if not _is_number(reproduced):
        return MetricComparison(metric, paper_value, None, None, None, False, "no reproduced value")
    reproduced_value = normalize_scale(reproduced)

    if paper_value == 0.0:
        # A zero paper value has no relative error to speak of, and dividing by it is meaningless.
        # Equality is the only defensible test; any nonzero result is a real deviation.
        ok = reproduced_value == 0.0
        return MetricComparison(metric, 0.0, reproduced_value, abs(reproduced_value), None, ok,
                                "zero paper value: exact equality required")
    absolute = abs(reproduced_value - paper_value)
    relative = absolute / abs(paper_value) * 100.0
    # The rule is `<= 5.0`, so a value that is exactly at the boundary must pass. Binary floating
    # point makes `57.07 * 0.95` land a few ulps below the paper-derived 5.0, which would fail a run
    # that is mathematically exactly at the tolerance. A relative epsilon far below any measurement
    # precision absorbs that without letting a real 5.01 percent deviation through.
    return MetricComparison(metric, paper_value, reproduced_value, absolute, relative,
                            relative <= tolerance_pct * (1 + 1e-12))


def _comparability_blockers(paper_meta: dict[str, Any], run_meta: dict[str, Any]) -> list[str]:
    blockers = []
    for key in COMPARABILITY_FIELDS:
        expected = paper_meta.get(key)
        actual = run_meta.get(key)
        if expected is None or actual is None:
            blockers.append(f"{key} unknown: paper={expected!r} run={actual!r}")
        elif expected != actual:
            blockers.append(f"{key} mismatch: paper={expected!r} run={actual!r}")
    return blockers


def _failure_note(comparison: MetricComparison) -> str:
    if comparison.relative_error_pct is None:
        return f"{comparison.metric}: {comparison.note or 'not comparable'}"
    return f"{comparison.metric} differs by {comparison.relative_error_pct:.4f}%"


def evaluate_gate(
    paper_metrics: dict[str, float | None] | None,
    reproduced_metrics: dict[str, float | None] | None,
    *,
    paper_meta: dict[str, Any] | None = None,
    run_meta: dict[str, Any] | None = None,
    data_kind: str | None = None,
    real_execution: bool = False,
    tolerance_pct: float = TOLERANCE_PCT,
) -> GateVerdict:
    """Decide a run's reproduction status, refusing to compare what is not comparable.

    Order matters. A mocked run, an unverified paper reference, or a protocol/checkpoint mismatch is
    rejected before any arithmetic, and each returns a distinct state so the report can say why.
    """
    if data_kind is not None and data_kind != "experiment":
        return GateVerdict(NOT_RUN, False, False, [], [f"data_kind={data_kind!r} is not a real experiment"],
                           "a mock or placeholder result is never evidence about reproduction")
    if not real_execution:
        return GateVerdict(NOT_RUN, False, False, [], ["no genuine execution recorded"],
                           "the gate compares real runs only")
    if not paper_metrics or all(paper_metrics.get(k) is None for k in METRIC_KEYS):
        return GateVerdict(METRIC_REFERENCE_UNVERIFIED, False, False, [],
                           ["no verified paper metric for this experiment"],
                           "without a comparable published number there is nothing to reproduce")
    if not reproduced_metrics:
        return GateVerdict(NOT_RUN, False, False, [], ["no reproduced metrics"], "the evaluator produced no metrics")

    blockers = _comparability_blockers(paper_meta or {}, run_meta or {})
    if blockers:
        # Distinguish a protocol/checkpoint problem from missing metadata: both refuse comparison,
        # but they send an operator to different places.
        joined = " ".join(blockers)
        if "protocol_id" in joined or "split" in joined or "dataset" in joined:
            status = BLOCKED_PROTOCOL_MISMATCH
        elif "checkpoint" in joined or "noise" in joined:
            status = BLOCKED_CHECKPOINT_MISMATCH
        else:
            status = BLOCKED_DATA_PROVENANCE
        return GateVerdict(status, False, False, [], blockers, "paper and run metadata do not describe one experiment")

    comparisons = [compare_metric(k, paper_metrics.get(k), reproduced_metrics.get(k), tolerance_pct)
                   for k in METRIC_KEYS]
    comparable = [c for c in comparisons if c.within_tolerance is not None]
    if not comparable:
        return GateVerdict(METRIC_REFERENCE_UNVERIFIED, False, False, comparisons,
                           ["no metric had both a paper value and a reproduced value"],
                           "nothing was actually compared")
    failed = [c for c in comparable if not c.within_tolerance]
    variant = any((run_meta or {}).get(key) for key in VARIANT_KEYS)
    if failed:
        return GateVerdict(ALERT_METRIC_DEVIATION, False, variant, comparisons,
                           [_failure_note(c) for c in failed],
                           "at least one comparable metric exceeds the tolerance")
    status = PASS_WITHIN_5_PERCENT_VARIANT if variant else PASS_WITHIN_5_PERCENT
    return GateVerdict(status, True, variant, comparisons, [],
                       "every comparable metric is within 5 percent relative error")
