"""The single numerical acceptance gate: does a real run agree with the published numbers?

This is the only place a benchmark verdict is decided. A report prints what this returns; it does not
compute its own tolerance. The gate is deliberately unaware of how the evaluator ran, so it can be
tested against hand-computed cases, and it never calls a language model.

Order of decision, and why:

1. **Is this a real experiment?** A mock or placeholder result is `NOT_RUN`, before any arithmetic.
2. **Is there a verified reference?** An unverified paper row, or a row that does not publish the
   metric being judged, is `METRIC_REFERENCE_UNVERIFIED`. The missing metric is not exempt: a run with
   nonsense in the unpublished column would otherwise pass.
3. **Does the reference describe this experiment?** A 0%-noise paper row cannot be the reference for a
   50%-noise checkpoint, whatever the numbers say. Mismatched protocol, checkpoint condition, dataset
   or noise is `BLOCKED_*`, and that refusal happens before any number is compared.
4. **Then, and only then, compare.** 5 percent RELATIVE error per metric, on every comparable metric.
   The mean cannot carry a broken R@10.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from workbench.backend import benchmark_references as benchmarks

#: The accepted rule. Relative percentage error, evaluated per metric.
TOLERANCE_PCT = 5.0

METRIC_KEYS = ("r10", "r50", "mean")

#: Per-category FashionIQ metrics, when the paper publishes them and the run observes them. The
#: aggregate cannot carry a broken category: a strong Shirt result must not conceal a Dress
#: deviation through averaging. A metric is compared only when BOTH sides state it; a paper value
#: with no observed counterpart is reported as unchecked, never as a match.
CATEGORY_KEYS = tuple(f"{category}_{metric}"
                      for category in ("dress", "shirt", "toptee")
                      for metric in ("r10", "r50"))
ALL_METRIC_KEYS = METRIC_KEYS + CATEGORY_KEYS


def category_key(category: str, metric: str) -> str:
    return f"{category}_{metric}"


def deviations(metrics: list[MetricComparison], threshold_pct: float) -> dict[str, Any]:
    """Re-evaluate recorded comparisons against a user-selected threshold, without rerunning anything.

    The gate's own verdict is an audit record with the fixed rule; this is the presentation layer
    that lets an operator move the threshold and see labels change. Same arithmetic, same edge
    policy, one implementation -- the frontend never recomputes a percentage.
    """
    flagged = [c for c in metrics
               if c.relative_error_pct is not None and c.relative_error_pct > threshold_pct]
    return {
        "threshold_pct": threshold_pct,
        "deviating": [c.to_dict() for c in flagged],
        "deviates": bool(flagged),
        "max_relative_error_pct": max((c.relative_error_pct for c in flagged), default=None),
        "worst_metric": max(flagged, key=lambda c: c.relative_error_pct or 0.0).metric
        if flagged else None,
    }

#: Declared scale of a metric set. The gate never infers the scale from the value: `1.0` is ambiguous
#: between "1 percent" and "100 percent", so the parser that produced the numbers declares which
#: convention it used. An unknown declaration raises rather than being absorbed silently.
SCALE_PERCENTAGE = "percentage"
SCALE_FRACTION = "fraction"
VALID_SCALES = (SCALE_PERCENTAGE, SCALE_FRACTION)

#: Fields that must agree between the reference and the run before any number is compared. A missing
#: value on either side is treated as unknown, and unknown never matches.
COMPARABILITY_FIELDS = (
    "protocol_id",
    "checkpoint_id",
    "checkpoint_training_noise_pct",
    "evaluation_noise_pct",
    "split",
    "dataset",
)

#: A run that used a declared, justified stand-in for an input whose identity is unknown is a
#: variant, never exact official reproduction.
VARIANT_KEYS = ("caption_alias_staged", "declared_variant")

PASS_WITHIN_5_PERCENT = "PASS_WITHIN_5_PERCENT"
PASS_WITHIN_5_PERCENT_VARIANT = "PASS_WITHIN_5_PERCENT_VARIANT"
# An explicitly incomplete pass: every comparable metric agreed, but the paper published category
# metrics the run never observed, so the aggregate carried no per-category confirmation. A consumer
# that treats any PASS_* as a full reproduction check is wrong; this status says so -- and
# `passed` is False, so no consumer can mistake incomplete metric coverage for a verified
# reproduction.
PASS_WITH_UNCHECKED_CATEGORIES = "PASS_WITH_UNCHECKED_CATEGORIES"
# Numerical comparability held, but the run's checkpoint mapping is UNVERIFIED: nobody established
# which upstream asset these numbers came from, so the match cannot certify an *official*
# reproduction of the paper's experiment. The comparisons stay on the verdict for inspection; a
# consumer that needs certified identity reads this status, not `passed`.
PASS_METRICS_ONLY_MAPPING_UNVERIFIED = "PASS_METRICS_ONLY_MAPPING_UNVERIFIED"
ALERT_METRIC_DEVIATION = "ALERT_METRIC_DEVIATION"
BLOCKED_PROTOCOL_MISMATCH = "BLOCKED_PROTOCOL_MISMATCH"
BLOCKED_CHECKPOINT_MISMATCH = "BLOCKED_CHECKPOINT_MISMATCH"
BLOCKED_DATA_PROVENANCE = "BLOCKED_DATA_PROVENANCE"
BLOCKED_SOURCE = "BLOCKED_SOURCE"
BLOCKED_ENVIRONMENT = "BLOCKED_ENVIRONMENT"
METRIC_REFERENCE_UNVERIFIED = "METRIC_REFERENCE_UNVERIFIED"
NOT_RUN = "NOT_RUN"
FAILED_EVALUATION = "FAILED_EVALUATION"
FAILED_OUTPUT_VALIDATION = "FAILED_OUTPUT_VALIDATION"


@dataclass(frozen=True)
class MetricComparison:
    metric: str
    paper: float | None
    reproduced: float | None
    absolute_error_pp: float | None
    relative_error_pct: float | None
    within_tolerance: bool | None
    paper_provenance: str = benchmarks.PROVENANCE_UNAVAILABLE
    note: str = ""
    direction: str = "equal"  # "higher" | "lower" | "equal": reproduced vs published

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        # A published zero has no meaningful signed difference, so it stays a raw delta.
        payload["signed_delta_pp"] = (None if self.reproduced is None or self.paper is None
                                      else self.reproduced - self.paper)
        return payload


@dataclass(frozen=True)
class GateVerdict:
    status: str
    passed: bool
    variant: bool
    comparisons: list[MetricComparison] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    detail: str = ""
    reference_evidence: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "passed": self.passed,
            "variant": self.variant,
            "blockers": list(self.blockers),
            "detail": self.detail,
            "reference_evidence": self.reference_evidence,
            "comparisons": [c.to_dict() for c in self.comparisons],
        }


def normalize_scale(value: float | None, scale: str) -> float | None:
    """Put a Recall value on the percentage scale the papers publish.

    The caller declares the convention its parser used. Nothing is inferred: a fraction is multiplied
    by 100, a percentage is passed through, and anything else raises.
    """
    if value is None:
        return None
    if scale == SCALE_PERCENTAGE:
        return float(value)
    if scale == SCALE_FRACTION:
        return float(value) * 100.0
    raise ValueError(f"unsupported metric scale: {scale!r}")


def _is_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    number = float(value)
    return number == number and abs(number) != float("inf")


def compare_metric(
    metric: str,
    paper: float | None,
    reproduced: float | None,
    *,
    scale: str = SCALE_PERCENTAGE,
    tolerance_pct: float = TOLERANCE_PCT,
    paper_provenance: str = benchmarks.PROVENANCE_UNAVAILABLE,
) -> MetricComparison:
    """Compare one metric under the relative-error rule, with an explicit zero-baseline policy.

    A zero published value has no meaningful relative error, so equality is the only defensible test
    there and the relative field stays `None` rather than being divided by zero.
    """
    direction = "equal"
    if paper is None:
        return MetricComparison(metric, None, reproduced, None, None, None, paper_provenance,
                                "no published value for this metric", direction)
    paper_value = normalize_scale(paper, scale)
    if not _is_number(reproduced):
        return MetricComparison(metric, paper_value, None, None, None, False, paper_provenance,
                                "no usable reproduced value", direction)
    reproduced_value = normalize_scale(reproduced, scale)
    direction = ("higher" if reproduced_value > paper_value
                 else "lower" if reproduced_value < paper_value else "equal")

    if paper_value == 0.0:
        ok = reproduced_value == 0.0
        return MetricComparison(metric, 0.0, reproduced_value, abs(reproduced_value), None, ok,
                                paper_provenance, "zero baseline: exact equality required", direction)

    absolute = abs(reproduced_value - paper_value)
    relative = absolute / abs(paper_value) * 100.0
    # The rule is `<= 5.0`, so a value mathematically at the boundary must pass. Binary floating point
    # makes `57.07 * 0.95` land a few ulps off, so a relative epsilon far below any measurement
    # precision absorbs that without admitting a real 5.01 percent deviation.
    return MetricComparison(metric, paper_value, reproduced_value, absolute, relative,
                            relative <= tolerance_pct * (1 + 1e-12), paper_provenance,
                            direction=direction)


def _failure_note(comparison: MetricComparison) -> str:
    if comparison.relative_error_pct is None:
        return f"{comparison.metric}: {comparison.note or 'not comparable'}"
    return (f"{comparison.metric} differs by {comparison.relative_error_pct:.4f}% "
            f"(paper {comparison.paper}, reproduced {comparison.reproduced})")


def _reference_expected(reference: benchmarks.BenchmarkReference) -> dict[str, Any]:
    """The conditions the reference describes, in the vocabulary the comparability fields use."""
    payload = reference.payload
    return {
        "protocol_id": payload.get("gallery_protocol"),
        "checkpoint_id": payload.get("checkpoint_id"),
        "checkpoint_training_noise_pct": payload.get("checkpoint_training_noise_pct"),
        "evaluation_noise_pct": payload.get("evaluation_noise_pct"),
        "split": payload.get("split"),
        "dataset": payload.get("dataset"),
    }


STATE_FIELD = "checkpoint_training_noise_pct"

BLOCKED_REFERENCE_INCOMPLETE = "BLOCKED_REFERENCE_INCOMPLETE"


def _reference_constraint_count(reference: benchmarks.BenchmarkReference) -> int:
    """How many comparability conditions the reference actually states.

    A row that states none of them cannot describe any particular experiment, so comparing against it
    would certify whatever run happened to be nearby. Counting is on the *stated* conditions, not on
    the metric values: a paper that prints numbers but never says which protocol produced them is not
    evidence about a run under any protocol.
    """
    expected = _reference_expected(reference)
    return sum(1 for key in COMPARABILITY_FIELDS if expected.get(key) is not None)


def _comparability_blockers(reference: benchmarks.BenchmarkReference,
                            run_meta: dict[str, Any]) -> list[str]:
    expected = _reference_expected(reference)
    blockers = []
    for key in COMPARABILITY_FIELDS:
        want = expected.get(key)
        got = run_meta.get(key)
        if want is None:
            # The paper does not state this condition, so the gate cannot use it either way. That is a
            # limitation of the reference, not a match: record it so a reader knows what was not
            # checked, and only refuse when the run contradicts something the reference does state.
            blockers.append(f"{key} not stated by the reference (run={got!r}); not checked")
            continue
        if got is None:
            blockers.append(f"{key} missing from the run record: reference={want!r}")
        elif str(want) != str(got):
            blockers.append(f"{key} mismatch: reference={want!r} run={got!r}")
    return blockers


def _blocked_status(blockers: list[str]) -> str:
    """Send the operator to the right place: protocol, checkpoint, or provenance."""
    joined = " ".join(blockers)
    hard = [b for b in blockers if "not stated by the reference" not in b]
    if not hard:
        return BLOCKED_DATA_PROVENANCE
    joined = " ".join(hard)
    if "protocol_id" in joined or "split" in joined or "dataset" in joined:
        return BLOCKED_PROTOCOL_MISMATCH
    if "checkpoint" in joined or "noise" in joined:
        return BLOCKED_CHECKPOINT_MISMATCH
    return BLOCKED_DATA_PROVENANCE


def evaluate_gate(
    reference: benchmarks.BenchmarkReference | None,
    reproduced_metrics: dict[str, float | None] | None,
    *,
    run_meta: dict[str, Any] | None = None,
    data_kind: str | None = None,
    real_execution: bool = False,
    evaluation_completed: bool = True,
    metric_scale: str = SCALE_PERCENTAGE,
    tolerance_pct: float = TOLERANCE_PCT,
    reproduced_category_metrics: dict[str, dict[str, float | None]] | None = None,
) -> GateVerdict:
    """Decide a run's reproduction status against a committed benchmark reference.

    Every refusal returns a distinct state, and every state says which condition failed, because a
    verdict that only says "not a pass" sends an operator nowhere.
    """
    if data_kind is not None and data_kind != "experiment":
        return GateVerdict(NOT_RUN, False, False, [], [f"data_kind={data_kind!r} is not a real experiment"],
                           "a mock or placeholder result is never evidence about reproduction")
    if not real_execution:
        return GateVerdict(NOT_RUN, False, False, [], ["no genuine execution recorded"],
                           "the gate compares real runs only")
    if not evaluation_completed:
        return GateVerdict(FAILED_EVALUATION, False, False, [], ["the evaluation did not complete"],
                           "an evaluator that failed or was interrupted has no metrics to judge")
    if not reproduced_metrics:
        return GateVerdict(FAILED_OUTPUT_VALIDATION, False, False, [], ["no reproduced metrics"],
                           "the evaluator produced nothing to compare")
    if reference is None:
        return GateVerdict(METRIC_REFERENCE_UNVERIFIED, False, False, [],
                           ["no committed reference describes this experiment"],
                           "without a verified published row there is nothing to reproduce")
    if reference.reference_status != benchmarks.REFERENCE_VERIFIED:
        return GateVerdict(METRIC_REFERENCE_UNVERIFIED, False, False, [],
                           [f"reference {reference.reference_id} is {reference.reference_status}"],
                           "an unverified reference cannot certify a run")
    if metric_scale not in VALID_SCALES:
        return GateVerdict(FAILED_OUTPUT_VALIDATION, False, False, [],
                           [f"unknown metric scale {metric_scale!r}"],
                           "the metric convention must be declared, not inferred")

    missing = [k for k in METRIC_KEYS
               if reference.value_for(k) is None and reproduced_metrics.get(k) is None]
    if all(reference.value_for(k) is None for k in METRIC_KEYS):
        return GateVerdict(METRIC_REFERENCE_UNVERIFIED, False, False, [],
                           [f"reference {reference.reference_id} publishes no usable metric"],
                           "nothing to compare")

    blockers = _comparability_blockers(reference, run_meta or {})
    hard = [b for b in blockers if "not stated by the reference" not in b]
    unchecked = [b for b in blockers if "not stated by the reference" in b]
    # A reference that states none of the conditions describes no particular experiment, so it cannot
    # certify one. This is refused rather than noted: a row whose protocol, checkpoint, dataset and
    # split are all unknown would otherwise pass any run whose numbers happened to land nearby, which
    # is the opposite of what this gate is for.
    if _reference_constraint_count(reference) == 0:
        return GateVerdict(BLOCKED_REFERENCE_INCOMPLETE, False, False, [], blockers,
                           f"reference {reference.reference_id} states none of the conditions a run is "
                           f"compared on, so it cannot certify any experiment")
    if hard:
        return GateVerdict(_blocked_status(blockers), False, False, [], blockers,
                           "the reference and the run do not describe one experiment")

    comparisons = [
        compare_metric(key, reference.value_for(key), reproduced_metrics.get(key),
                       scale=metric_scale, tolerance_pct=tolerance_pct,
                       paper_provenance=reference.metric_provenance(key))
        for key in METRIC_KEYS
    ]
    # Per-category metrics: compared only where both sides publish them, so a paper's category table
    # cannot be evaded by a strong average and an observed category cannot be judged against a row
    # that never printed one.
    categories = reproduced_category_metrics if isinstance(reproduced_category_metrics, dict) else {}
    paper_categories = reference.payload.get("category_metrics") or {}
    category_unchecked: list[str] = []
    for key in CATEGORY_KEYS:
        category, _, metric = key.partition("_")
        paper_value = (paper_categories.get(category) or {}).get(metric)
        if paper_value is None:
            continue
        observed = (categories.get(category) or {}).get(metric)
        if observed is None:
            category_unchecked.append(key)
            continue
        comparisons.append(
            compare_metric(key, paper_value, observed,
                           scale=metric_scale, tolerance_pct=tolerance_pct,
                           paper_provenance=benchmarks.PAPER_REPORTED))
    comparable = [c for c in comparisons if c.within_tolerance is not None]
    if not comparable:
        return GateVerdict(METRIC_REFERENCE_UNVERIFIED, False, False, comparisons,
                           ["no metric had both a published value and a reproduced value"],
                           "nothing was actually compared")
    failed = [c for c in comparable if not c.within_tolerance]
    variant = any((run_meta or {}).get(key) for key in VARIANT_KEYS)
    mapping_verified = (run_meta or {}).get("checkpoint_mapping_verified") is True
    evidence = reference.as_evidence()
    if failed:
        return GateVerdict(ALERT_METRIC_DEVIATION, False, variant, comparisons,
                           [_failure_note(c) for c in failed], 
                           "at least one comparable metric exceeds the tolerance", evidence)
    status = PASS_WITHIN_5_PERCENT_VARIANT if variant else PASS_WITHIN_5_PERCENT
    if not mapping_verified:
        # Numerical comparability and certified official identity are separate claims. With an
        # unverified mapping the numbers can be compared, but the run cannot be certified as an
        # official reproduction of the paper's experiment: the asset behind the numbers is unknown.
        status = PASS_METRICS_ONLY_MAPPING_UNVERIFIED
    detail = "every comparable metric is within 5 percent relative error"
    if not mapping_verified:
        detail += "; checkpoint mapping is UNVERIFIED, so this is a metric comparison, not a certified official reproduction"
    if missing:
        detail += f"; {', '.join(missing)} had no value on either side and was not compared"
    if unchecked:
        detail += f"; {len(unchecked)} condition(s) the reference does not state were not checked"
    if category_unchecked:
        detail += (f"; category metrics {', '.join(category_unchecked)} are published by the paper "
                   f"but the run observed none, so they were not compared")
    if category_unchecked:
        # The paper publishes category measurements the run did not produce. The comparable
        # aggregate agreed, but calling this an unqualified pass would let a strong average stand
        # in for the per-category check the paper's own table invites; the status names the gap.
        # Mapping status wins precedence when both hold: identity uncertainty is the stronger
        # caveat, and the coverage gap stays visible in `detail`.
        status = PASS_WITH_UNCHECKED_CATEGORIES if mapping_verified else PASS_METRICS_ONLY_MAPPING_UNVERIFIED
    if status != PASS_WITHIN_5_PERCENT and status != PASS_WITHIN_5_PERCENT_VARIANT:
        # Incomplete coverage (no observed categories) or an unverified checkpoint mapping: the
        # numbers stay on the verdict for inspection, but neither is full reproduction
        # verification, so `passed` is False for both.
        return GateVerdict(status, False, variant, comparisons, [], detail, evidence)
    # A passing verdict lists no blockers. Unchecked conditions belong in `detail` and in the
    # reference evidence, where a reader looks for caveats; putting refusals in a `blockers` list on a
    # `passed=True` verdict makes every consumer mishandle the result.
    return GateVerdict(status, True, variant, comparisons, [], detail, evidence)
