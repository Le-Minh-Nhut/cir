"""G1/G2 parity gate: a mandatory subcheck that failed can never surface as PASS.

Why this module exists
----------------------
Before it, the G1/G2 verdict lived only in hand-written surfaces (`GATES.csv`, `ENCODER_PARITY.vi.md`,
`FINAL_REPORT.vi.md`). The measurement artifacts those surfaces cite carry their own booleans --
`within_tolerance: false`, `non_tie_count: 1`, `max_rel: 1.165e-02` against a declared `1e-3` -- while
the surface above them reads "G2 PASS". Nothing in the codebase joined the two, so nothing could
contradict a PASS with the evidence underneath it.

On HEAD, for `encoder`, the artifacts record these mandatory subchecks failing:

* `batch_size_1_vs_n.max_rel` = 3.131e-03 / 7.795e-03 / 1.165e-02 (shirt/dress/toptee) against the
  declared `batch_max_rel` of `1e-3` -- `within_tolerance: false` in all three files;
* `rank_differences_vs_native_val_control.non_tie_count` = 1 in dress and 1 in shirt, against the
  declared rule "a rank change is allowed ONLY when the two candidates' scores are equal within
  `tie_epsilon`"; the artifact itself classifies both as `explained_by_tie_epsilon: false`;
* `population.*.max_rel` failing its declared `forward_max_rel` (`1e-4`) on the near-zero-denominator
  effect -- `within_tolerance: false`.

Every one of them was declared, measured and reported honestly in prose. The defect is that no
surface *aggregated* them, so "G2 PASS" could be printed beside them.

Four things that are easy to conflate, kept apart
------------------------------------------------
Each check carries a `dimension`, and a status is never transferred between dimensions:

* `input_parity`    -- G1: ids, gallery order, model text, tokens, masks, decode expression.
* `forward_parity`  -- G2: the same batch composition, cross-process.
* `batch_parity`    -- G2: batch size 1 vs N. Legitimately different reduction order; its own
                       tolerances, and the subcheck whose relative term fails here.
* `scoring_parity`  -- G2: scores and ranks on the same features (tie epsilon, non-tie rule).
* `metrics_parity`  -- G2: Recall hit counts.
* `observer_parity` -- the observation pipeline reproducing the macro it was built from. It is NOT
                       forward parity and is never averaged with it.

`observer matches macro` is therefore its own dimension: a run whose Recall reproduces its own
aggregate exactly (`native_metrics_match_feature_rerank`) says nothing about whether the adapter's
forward pass matches the native one, and this module never lets one stand in for the other.

Two relative-error standards, versioned, never merged
-----------------------------------------------------
`MAX_REL` on a unit-norm embedding divides an absolute deviation by a component that can be ~1e-8, so
a `1e-7` absolute error reads as `7.8`. That is a property of the denominator, not of the model.

The declared standard in every existing artifact is `parity-relative/v1`:

    fail if  max_abs > atol   OR   max_rel > rtol

`parity-absolute-relative/v2` is the combined form, which does not collapse when the denominator
vanishes:

    fail if  abs(a-b) > atol + rtol * abs(b)   for any component

v2 is **not** applied to the v1 measurements. Doing so would move a declared tolerance after seeing
the number, which is exactly what this campaign forbids. An artifact is evaluated under v2 only when
the artifact ITSELF declares `standard: "parity-absolute-relative/v2"`, i.e. when it was produced
after the declaration was written down. Applied to an existing v1 artifact, v2 is reported as
`ADVISORY_ONLY` with `declared_before_measurement: False`, and it cannot change `passed`.

The v1 result is always kept alongside (`checks[].standard`, `checks[].measured`, and the artifact's
own recorded `within_tolerance`), so the historical record stays readable exactly as it was.

The gate never edits an artifact, a CSV or a document. It derives a verdict and names its inputs.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

#: The directory holding the G1/G2 measurement artifacts. Fixed, never request-derived.
PARITY_EVIDENCE_RELATIVE = ("workbench", "docs", "reviews", "20261109-native15-composition-limn")

#: Models known to have G1/G2 artifacts. A SEED for discovery, never the set the gate will judge:
#: `_discover_models` adds every model id whose artifacts are on disk, so a third model is gated on
#: its own evidence instead of being reported NOT_RUN by construction.
EVIDENCED_MODELS = ("limn", "encoder")

#: Percent-free status vocabulary. `PASS` is the only value that claims every mandatory subcheck was
#: MEASURED and held. Four distinct verdicts, and `None` is never a pass:
#:
#: * `PASS`             -- every mandatory dimension was measured and held;
#: * `INCOMPLETE`       -- nothing failed, but a mandatory dimension was never measured (an empty
#:                         adapter body, a missing batch comparison). Not a pass, not a failure;
#: * `MANDATORY_SUBCHECK_FAILED` -- a measured mandatory subcheck held a value outside its declared
#:                         tolerance;
#: * `NOT_RUN`          -- no artifact exists for this model at all.
PASS = "PARITY_PASS"
INCOMPLETE = "PARITY_INCOMPLETE"
MANDATORY_FAILED = "PARITY_MANDATORY_SUBCHECK_FAILED"
NOT_RUN = "PARITY_NOT_RUN"
EVIDENCE_MISSING = "PARITY_EVIDENCE_MISSING"
MALFORMED = "PARITY_EVIDENCE_MALFORMED"

#: Dimensions a verdict must rest on to be called a pass at all. `observer_parity` is deliberately
#: absent: it is the observation pipeline reproducing its own macro, which is not a G1/G2 subcheck
#: and cannot be mandatory for one.
REQUIRED_DIMENSIONS = ("input_parity", "forward_parity", "batch_parity", "scoring_parity",
                       "metrics_parity")

STANDARD_RELATIVE_V1 = "parity-relative/v1"
STANDARD_ABS_REL_V2 = "parity-absolute-relative/v2"
STANDARDS = (STANDARD_RELATIVE_V1, STANDARD_ABS_REL_V2)

#: Dimensions. A status is per dimension; there is no cross-dimension average anywhere.
DIMENSIONS = ("input_parity", "forward_parity", "batch_parity", "scoring_parity",
              "metrics_parity", "observer_parity", "relation_parity")


@dataclass(frozen=True)
class ParityCheck:
    """One declared subcheck and what the artifact measured for it."""

    name: str
    dimension: str
    gate: str  # "G1" | "G2"
    declared: Any
    measured: Any
    ok: bool
    mandatory: bool
    standard: str = STANDARD_RELATIVE_V1
    note: str = ""
    #: The boolean the HARNESS itself recorded for this comparison, when it recorded one. Kept beside
    #: this gate's verdict so the old label is readable and any disagreement is visible.
    harness_recorded: Any = None
    #: True when the check reports ABSENT evidence rather than a measured value that failed. Such a
    #: check belongs to `missing`, not to `failures`: "no claim was recorded" is `INCOMPLETE`, and it
    #: is only promoted to a failure when the artifact DID make claims for that category.
    absence: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "dimension": self.dimension, "gate": self.gate,
                "declared": self.declared, "measured": self.measured, "ok": self.ok,
                "mandatory": self.mandatory, "standard": self.standard, "note": self.note,
                "harness_recorded_within_tolerance": self.harness_recorded,
                "reports_absent_evidence": self.absence}


@dataclass
class ParityVerdict:
    """The derived verdict for one model. `passed` is True only when no mandatory check failed."""

    model_id: str
    status: str = NOT_RUN
    passed: bool = False
    complete: bool = False
    checks: list[ParityCheck] = field(default_factory=list)
    failures: list[ParityCheck] = field(default_factory=list)
    dimensions: dict[str, bool] = field(default_factory=dict)
    advisory: list[dict[str, Any]] = field(default_factory=list)
    coverage: dict[str, list[str]] = field(default_factory=dict)
    #: Mandatory dimensions that were never measured. Non-empty => `INCOMPLETE`, never `PASS`.
    #: A dimension is listed when ANY judged category carries no mandatory check for it: "measured
    #: globally, once" is not the same claim as "measured on every category that has evidence".
    missing: list[str] = field(default_factory=list)
    #: Which mandatory dimensions each category actually rested on, and which it did not. The pair
    #: `missing` + this is what makes a per-category gap nameable instead of averaged away.
    missing_by_category: dict[str, list[str]] = field(default_factory=dict)
    mandatory_by_category: dict[str, list[str]] = field(default_factory=dict)
    #: The categories this verdict judged, in order. A pass asserts every required dimension had a
    #: mandatory check in EACH of them.
    categories: tuple[str, ...] = ()
    #: The standard the verdict was ACTUALLY computed under -- read from the artifacts, which is the
    #: only defensible answer: the declared tolerance and its form are part of the measurement. A
    #: caller may REQUEST another standard; that request cannot relabel a v1 result.
    applied_standard: str = STANDARD_RELATIVE_V1
    requested_standard: str = STANDARD_RELATIVE_V1
    #: Comparisons where the harness's own `within_tolerance` boolean disagrees with the declared
    #: tolerance table. The declared rule wins; the old label is preserved as the v1 record.
    label_disagreements: list[dict[str, Any]] = field(default_factory=list)
    historical: dict[str, Any] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        _check_invariants(self)
        return {"model_id": self.model_id, "status": self.status, "passed": self.passed,
                "complete": self.complete,
                "applied_standard": self.applied_standard,
                "requested_standard": self.requested_standard,
                "dimensions": dict(self.dimensions),
                # Which measurements each dimension ACTUALLY rests on, and what it never saw. A
                # dimension True with a gap is "every check that exists passed", not "fully checked",
                # and a reader must be able to tell the two apart.
                "coverage": {key: list(value) for key, value in self.coverage.items()},
                # The mandatory dimensions nothing measured. Named here so `INCOMPLETE` is
                # actionable rather than a bare status. Non-empty when ANY judged category lacks the
                # dimension, so one well-measured category cannot carry a bare one.
                "missing": list(self.missing),
                "missing_by_category": {key: list(value)
                                        for key, value in self.missing_by_category.items()},
                "mandatory_by_category": {key: list(value)
                                          for key, value in self.mandatory_by_category.items()},
                "label_disagreements": list(self.label_disagreements),
                "failures": [check.to_dict() for check in self.failures],
                "failure_count": len(self.failures),
                "checks": [check.to_dict() for check in self.checks],
                "advisory": list(self.advisory),
                # The verdict a human surface printed before this module existed, verbatim, so the
                # record stays readable and nothing here rewrites history.
                "historical": dict(self.historical),
                "evidence": list(self.evidence), "detail": self.detail}


def evidence_root(repo_root: Path | None = None) -> Path:
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    return root.joinpath(*PARITY_EVIDENCE_RELATIVE)


class _Malformed(Exception):
    """Evidence that cannot be judged: wrong shape, non-finite, or two sides that contradict.

    Raised rather than returned so no consumer can continue past it: a comparison the gate cannot
    understand must become `PARITY_EVIDENCE_MALFORMED`, never an empty `[]` of checks -- which is
    what let an empty adapter body report `PARITY_PASS`.
    """


def _read(path: Path) -> dict[str, Any]:
    """One artifact document. Anything that is not a JSON OBJECT is malformed evidence."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise _Malformed(f"{path.name}: not valid JSON ({error})") from None
    if not isinstance(document, dict):
        raise _Malformed(f"{path.name}: the document is a JSON {type(document).__name__}, not an "
                         "object, so there are no fields to judge")
    return document


def _number(container: dict[str, Any], key: str, *, where: str) -> tuple[bool, float | None]:
    """`(present, value)` for one numeric field.

    `present` says the KEY exists; `value` is None when the key exists holding an explicit `null`.
    Present-but-null is "the harness recorded no value": not a pass, and not a crash.

    Rejects three things the old gate accepted as numbers: a JSON boolean (`False == 0` makes a
    failing delta look like an exact match), a string, and NaN/Inf (every comparison against NaN is
    False, so a failing measurement took the branch that meant "not worse").
    """
    if key not in container:
        return False, None
    value = container[key]
    if value is None:
        return True, None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _Malformed(f"{where}.{key} is {value!r}, not a number")
    if not math.isfinite(float(value)):
        raise _Malformed(f"{where}.{key} is {value!r}: a non-finite value cannot be judged "
                         "against a tolerance")
    return True, float(value)


#: Models whose G1/G2 artifacts were written under the bare `g1_g2.<category>.<side>.json` names
#: instead of a model namespace. LIMN's three category files predate the namespaced form; the bare
#: names are a legacy convention for that one model, NOT a default for every model.
LEGACY_BARE_MODELS = ("limn",)

#: Models whose construction DRAWS a relation tensor from the global RNG (`ENCODER` draws
#: `learnable_relation`; it is not in `state_dict()` and no checkpoint restores it). For these, an
#: unrecorded `relation_state` digest means two runs may have compared features computed with
#: different relations -- a mandatory failure, never an exemption. A model not listed here is still
#: treated as relation-drawing when EITHER artifact carries the `relation_state` key at all.
RELATION_DRAWING_MODELS = ("encoder",)

def _artifact_names(model_id: str, category: str) -> tuple[str, str]:
    """The namespaced filenames for a model's evidence.

    Namespacing used to be reserved for `encoder`, so no other model could ever have its own files:
    its artifacts were invisible to the gate by construction, not by absence of evidence.
    """
    return (f"g1_g2.{model_id}.{category}.native.json",
            f"g1_g2.{model_id}.{category}.adapter.json")

def _artifact_paths(base: Path, model_id: str, category: str) -> tuple[Path, Path] | None:
    """This model's two artifact paths, or None when the evidence directory holds no pair for it.

    Tries the namespaced names first, then the legacy bare names for the one model that used them.
    Never falls back to another model's files.
    """
    candidates = [_artifact_names(model_id, category)]
    if model_id in LEGACY_BARE_MODELS:
        candidates.append((f"g1_g2.{category}.native.json", f"g1_g2.{category}.adapter.json"))
    for native_name, adapter_name in candidates:
        native_path, adapter_path = base / native_name, base / adapter_name
        if native_path.is_file() and adapter_path.is_file():
            return native_path, adapter_path
    return None


def _tolerances(native: dict[str, Any], adapter: dict[str, Any]) -> dict[str, Any]:
    """The tolerances BOTH sides declared, or a refusal when they disagree.

    The old merge kept the native value and dropped the adapter's silently, so two artifacts that
    declared different tolerances were judged against whichever side was written first -- and the
    looser number could win with nothing recording that it had. Nothing here picks a winner: a
    numeric disagreement is malformed evidence, reported with both values, because the declared
    tolerance is part of the measurement and cannot be reconciled after the fact.

    A tolerance declared on ONE side only is not a disagreement (nothing contradicts it); it stands.
    Numeric tolerances must be finite and non-negative: an infinite tolerance accepts any
    measurement, which is the same as having no gate at all.
    """
    left, right = native.get("tolerances"), adapter.get("tolerances")
    for side, block in (("native", left), ("adapter", right)):
        if block is not None and not isinstance(block, dict):
            raise _Malformed(f"the {side} artifact's `tolerances` is a "
                             f"{type(block).__name__}, not an object")
    left, right = left or {}, right or {}
    conflicts = []
    for key in sorted(set(left) & set(right)):
        a, b = left[key], right[key]
        if a == b:
            continue
        numeric = all(isinstance(value, (int, float)) and not isinstance(value, bool)
                      for value in (a, b))
        if not numeric:
            raise _Malformed(f"tolerances.{key} differs between the two artifacts: "
                             f"native {a!r}, adapter {b!r}")
        if not (math.isfinite(float(a)) and math.isfinite(float(b))):
            raise _Malformed(f"tolerances.{key} is non-finite: native {a!r}, adapter {b!r}")
        conflicts.append(f"{key}: native {a!r}, adapter {b!r}")
    if conflicts:
        raise _Malformed("the two artifacts declare different tolerances for "
                         + "; ".join(conflicts)
                         + " -- refusing to pick one, because the declared tolerance is part of "
                           "the measurement and a looser side must never win silently")
    merged = {**left, **{key: value for key, value in right.items() if key not in left}}
    for key, value in merged.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if not math.isfinite(float(value)):
            raise _Malformed(f"tolerances.{key} is {value!r}: a non-finite tolerance accepts any "
                             "measurement, so it is not a tolerance")
        if float(value) < 0.0:
            raise _Malformed(f"tolerances.{key} is {value!r}: a negative tolerance can never be met")
    return merged


def evaluate_abs_rel(actual: float, reference: float, *, atol: float, rtol: float) -> bool:
    """`abs(a-b) <= atol + rtol*abs(b)` for one component, the v2 rule."""
    return abs(actual - reference) <= atol + rtol * abs(reference)


def _relative_checks(measured: dict[str, Any], declared: dict[str, Any], *,
                     name: str, dimension: str, gate: str,
                     atol_key: str, rtol_key: str) -> list[ParityCheck]:
    """The three-part absolute / relative / cosine rule for one forward comparison.

    Every comparison is recomputed from the numbers. `within_tolerance` is kept only as
    `harness_recorded`, so an artifact that declares `true` beside a delta of 0.1 against `1e-6`
    fails and the disagreement is visible. A block that RECORDS a measurement the gate cannot judge
    (a null, a string, a NaN) is malformed, never an empty check list: silently returning `[]` is
    how an unmeasured comparison is mistaken for a satisfied one.
    """
    recorded = measured.get("within_tolerance")
    present_abs, max_abs = _number(measured, "max_abs", where=name)
    present_rel, max_rel = _number(measured, "max_rel", where=name)
    present_cos, cosine = _number(measured, "min_cosine", where=name)
    if not (present_abs or present_rel or present_cos):
        raise _Malformed(f"{name} records no max_abs, max_rel or min_cosine to compare")
    checks: list[ParityCheck] = []
    # The declared v1 rule is always evaluated, whatever standard was REQUESTED: a post-hoc standard
    # may be reported beside the record, never used to replace it.
    if present_abs:
        if max_abs is None:
            raise _Malformed(f"{name}.max_abs is null: a null measurement is not a passing one")
        atol = declared.get(atol_key)
        if atol is None:
            raise _Malformed(f"{name}.max_abs was measured but the artifact declares no "
                             f"`{atol_key}` to judge it against")
        checks.append(ParityCheck(
            name=f"{name}.max_abs", dimension=dimension, gate=gate,
            declared=atol, measured=max_abs, ok=max_abs <= float(atol),
            mandatory=True, standard=STANDARD_RELATIVE_V1, harness_recorded=recorded))
    if present_rel:
        rtol = declared.get(rtol_key)
        if rtol is None:
            raise _Malformed(f"{name}.max_rel was measured but the artifact declares no "
                             f"`{rtol_key}` to judge it against")
        if max_rel is None:
            # The declared table applies the relative term to layer 4 of every comparison of this
            # kind, so a null relative error cannot be read as "no relative error".
            raise _Malformed(f"{name}.max_rel is null: the declared `{rtol_key}` cannot be met by "
                             "an unrecorded value")
        checks.append(ParityCheck(
            name=f"{name}.max_rel", dimension=dimension, gate=gate,
            declared=rtol, measured=max_rel, ok=max_rel <= float(rtol),
            mandatory=True, standard=STANDARD_RELATIVE_V1, harness_recorded=recorded,
            note=("relative error on a component near zero: the denominator, not the model, drives "
                  "this. The declared tolerance is not moved; see parity-absolute-relative/v2")))
    if present_cos:
        min_cosine = declared.get("forward_min_cosine")
        if cosine is None:
            raise _Malformed(f"{name}.min_cosine is null: an unrecorded cosine is not agreement")
        if min_cosine is not None:
            checks.append(ParityCheck(
                name=f"{name}.min_cosine", dimension=dimension, gate=gate,
                declared=min_cosine, measured=cosine, ok=cosine >= float(min_cosine),
                mandatory=True, standard=STANDARD_RELATIVE_V1, harness_recorded=recorded))
    return checks


def _identity_checks(native: dict[str, Any], declared: dict[str, Any],
                     category: str) -> list[ParityCheck]:
    """G1: the discrete, zero-tolerance inputs. Missing evidence is not a pass."""
    layer = native.get("layer1_identity")
    checks: list[ParityCheck] = []
    if not isinstance(layer, dict):
        return [ParityCheck(f"{category}.layer1_identity", "input_parity", "G1", "present", None,
                            ok=False, mandatory=True,
                            note="the native artifact records no layer1 identity block")]
    required = ("gallery_order_identical", "reference_ids_identical", "target_ids_identical",
                "model_text_identical_to_author_composition")
    for key in required:
        checks.append(ParityCheck(
            name=f"{category}.identity.{key}", dimension="input_parity", gate="G1",
            declared="exact equality, zero tolerance", measured=layer.get(key),
            ok=layer.get(key) is True, mandatory=True))
    # Two PAIRS, not four checks. The native side and the annotation side are two independent
    # counts of the same population; comparing a field to itself across the "native"/"annotations"
    # naming would make two mandatory G1 checks unable to fail. `encoder_parity.py:216-219` and
    # `t4_parity.py:98-101` write them as `len(test_queries)`/`len(captions)` and
    # `len(test_targets)`/`len(rebuilt)`.
    for pair, native_key, declared_key, side in (
            ("query_count", "query_count_native", "query_count_annotations", "annotations"),
            ("gallery_count", "gallery_count_native",
             "gallery_count_rebuilt_from_annotations", "rebuilt from the annotations")):
        left, right = layer.get(native_key), layer.get(declared_key)
        # A count the artifact does not record is not agreement: absent evidence is a failure whose
        # note names the missing side, so a dropped field can never read as a pass.
        absent = [name for name, value, ok_type in
                  ((native_key, left, int), (declared_key, right, int))
                  if not (isinstance(value, ok_type) and not isinstance(value, bool))]
        ok = not absent and left == right
        checks.append(ParityCheck(
            name=f"{category}.identity.{pair}_agreement", dimension="input_parity", gate="G1",
            declared=right, measured={"native": left, "annotations": right},
            ok=ok, mandatory=True,
            note="" if ok else (
                f"no count recorded on the native side (`{native_key}`)" if native_key in absent
                else f"no count recorded on the {side} side (`{declared_key}`)" if declared_key in absent
                else f"the native evaluator scored {left} but the {side} declare {right}")))
    if layer.get("gallery_has_duplicates") is True:
        checks.append(ParityCheck(f"{category}.identity.gallery_has_duplicates", "input_parity", "G1",
                                  "no duplicates", True, ok=False, mandatory=True))
    return checks


#: The identity-bearing fields an artifact MAY carry at its TOP level. Read from both documents and
#: compared value-to-value: the `*_identical` booleans inside a payload are written by the harness
#: that predicts them, so they cannot certify that the two sides agree.
IDENTITY_FIELDS = ("model_id", "checkpoint_id", "checkpoint_identity", "protocol_id", "protocol",
                   "query_order_sha256", "gallery_order_sha256")

_ABSENT = object()

def _identity_cross_checks(native: dict[str, Any], adapter: dict[str, Any],
                           category: str) -> list[ParityCheck]:
    """G1: the two artifacts must declare the SAME identity, compared field by field.

    `_identity_checks` reads the native side only, so two artifacts that measured different
    checkpoints, protocols or query orders reported `input_parity: True`. Here both documents are
    read and the values compared. A field carried by ONE side and not the other is a failure too:
    "the adapter never declared which checkpoint it loaded" is not agreement.

    A field both sides omit is skipped -- most artifacts carry only a subset of these (LIMN carries
    none), and `input_parity` still has its mandatory layer1 check per category.
    """
    checks: list[ParityCheck] = []
    for field in IDENTITY_FIELDS:
        left = native[field] if field in native else _ABSENT
        right = adapter[field] if field in adapter else _ABSENT
        if left is _ABSENT and right is _ABSENT:
            continue
        ok = left is not _ABSENT and right is not _ABSENT and left == right
        checks.append(ParityCheck(
            name=f"{category}.input.identity_agreement.{field}", dimension="input_parity",
            gate="G1", declared="the same value recorded on both sides",
            measured={"native": None if left is _ABSENT else left,
                      "adapter": None if right is _ABSENT else right},
            ok=ok, mandatory=True,
            note="" if ok else ("the two artifacts disagree on this identity field (or only one "
                                "records it); the payload's own booleans are not consulted")))
    return checks

def _pixel_check(adapter: dict[str, Any], category: str) -> list[ParityCheck]:
    layer = adapter.get("layer2_pixels") or {}
    decode = layer.get("adapter_decode_vs_native_val_decode")
    if not isinstance(decode, dict):
        return []
    return [ParityCheck(
        name=f"{category}.pixels.adapter_decode_vs_native_val_decode", dimension="input_parity", gate="G1",
        declared="bitwise equality (max_abs == 0)",
        measured={"max_abs": decode.get("max_abs"), "bitwise_equal": decode.get("bitwise_equal")},
        ok=decode.get("bitwise_equal") is True and decode.get("max_abs") == 0.0, mandatory=True)]


def _rank_checks(adapter: dict[str, Any], declared: dict[str, Any],
                 category: str) -> list[ParityCheck]:
    """A rank change is permitted ONLY inside the tie epsilon; anything else is a mandatory FAIL.

    The two models record this differently and both shapes are read: ENCODER splits differences into
    tie-explained and non-tie (`rank_differences_vs_native_val_control.non_tie_count`), LIMN counts
    the differences it could not explain (`unexplained_rank_differences`).
    """
    scoring = adapter.get("scoring_ranking") or {}
    if not isinstance(scoring, dict):
        raise _Malformed(f"{category}: `scoring_ranking` is a {type(scoring).__name__}, not an object")
    if "unexplained_rank_differences" in scoring:
        _present, unexplained = _number(scoring, "unexplained_rank_differences",
                                       where=f"{category}.scoring_ranking")
        if unexplained is None:
            raise _Malformed(f"{category}.scoring_ranking.unexplained_rank_differences is null: an "
                             "unrecorded count is not zero unexplained differences")
        checks = [ParityCheck(
            name=f"{category}.ranks.unexplained_differences", dimension="scoring_parity",
            gate="G2", declared="0 rank differences the tie rule cannot explain",
            measured=unexplained, ok=unexplained == 0, mandatory=True,
            note=f"{scoring.get('rank_differences')!r} rank differences recorded; every one must be "
                 "a declared tie")]
        if "score_max_abs_delta_vs_dot" in scoring and declared.get("score_max_abs") is not None:
            _delta_present, delta = _number(scoring, "score_max_abs_delta_vs_dot",
                                            where=f"{category}.scoring_ranking")
            if delta is None:
                raise _Malformed(f"{category}.scoring_ranking.score_max_abs_delta_vs_dot is null")
            checks.append(ParityCheck(
                name=f"{category}.scores.max_abs_vs_dot", dimension="scoring_parity", gate="G2",
                declared=declared["score_max_abs"], measured=delta,
                ok=delta <= float(declared["score_max_abs"]), mandatory=True,
                harness_recorded=scoring.get("score_within_tolerance")))
        return checks
    rank_block = scoring.get("rank_differences_vs_native_val_control")
    if not isinstance(rank_block, dict):
        return []
    _count_present, non_tie = _number(rank_block, "non_tie_count",
                                      where=f"{category}.rank_differences_vs_native_val_control")
    if non_tie is None:
        # `non_tie_count: null` is an UNMEASURED comparison. The old code compared it with `== 0`,
        # so null became "no non-tie changes" -- the exact value a clean artifact records.
        raise _Malformed(f"{category}.rank_differences_vs_native_val_control.non_tie_count is null: "
                         "an unmeasured rank comparison is not a passing one")
    checks = [ParityCheck(
        name=f"{category}.ranks.non_tie_changes", dimension="scoring_parity", gate="G2",
        declared="0 non-tie rank changes (tie_epsilon %r)" % (declared.get("tie_epsilon"),),
        measured=non_tie, ok=non_tie == 0, mandatory=True,
        note="the artifact classifies each difference against the tie epsilon itself")]
    _gap_present, min_gap = _number(rank_block, "min_gap_abs",
                                    where=f"{category}.rank_differences_vs_native_val_control")
    if min_gap is not None and declared.get("tie_epsilon") is not None:
        checks.append(ParityCheck(
            name=f"{category}.ranks.min_gap_vs_tie_epsilon", dimension="scoring_parity", gate="G2",
            declared=declared.get("tie_epsilon"), measured=min_gap,
            ok=(non_tie == 0) or min_gap <= float(declared["tie_epsilon"]),
            mandatory=True))
    convention = scoring.get("adapter_vs_argsort_unmasked") or {}
    if not isinstance(convention, dict):
        raise _Malformed(f"{category}.scoring_ranking.adapter_vs_argsort_unmasked is a "
                         f"{type(convention).__name__}, not an object")
    if "non_tie_count" in convention:
        _conv_present, convention_non_tie = _number(
            convention, "non_tie_count", where=f"{category}.adapter_vs_argsort_unmasked")
        if convention_non_tie is None:
            raise _Malformed(f"{category}.adapter_vs_argsort_unmasked.non_tie_count is null: an "
                             "unmeasured convention comparison is not a match")
        checks.append(ParityCheck(
            name=f"{category}.ranks.tie_convention_non_tie", dimension="scoring_parity", gate="G2",
            declared="0 non-tie differences between rank conventions",
            measured=convention_non_tie, ok=convention_non_tie == 0,
            mandatory=True))
    return checks


def _tolerance_value(value: Any, *, where: str) -> float:
    """One declared tolerance, or a refusal. An infinite/negative value is not a tolerance."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _Malformed(f"{where} is {value!r}, not a number")
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise _Malformed(f"{where} is {value!r}: a tolerance must be a finite non-negative number")
    return float(value)

def _score_checks(adapter: dict[str, Any], declared: dict[str, Any],
                  category: str) -> list[ParityCheck]:
    """The score agreement, RECOMPUTED from the delta and the declared tolerance.

    `score_within_tolerance` is the harness's own boolean and is kept as `harness_recorded` only.
    The old code returned that boolean AS the verdict, so a block declaring
    `score_max_abs_delta_vs_dot: 0.1` beside `score_within_tolerance: true` reported a pass.

    ONE tolerance source. The body-level `score_max_abs_declared` used to win over the top-level
    `score_max_abs` unconditionally, so an adapter body could restate the tolerance as `1.0` beside a
    top-level `1e-6` and have a `0.1` delta pass. Now:

    * both declared and the body is LOOSER -> `_Malformed`, naming both numbers: a body cannot
      override a stricter declared tolerance;
    * both declared and the body is STRICTER -> the body wins, because it is the narrower claim;
    * only one declared -> that one is used.

    The artifact's recorded `within_tolerance` is never rewritten; it stays in `harness_recorded`.
    """
    scoring = adapter.get("scoring_ranking")
    if scoring is None:
        return []
    if not isinstance(scoring, dict):
        raise _Malformed(f"{category}: `scoring_ranking` is a {type(scoring).__name__}, not an object")
    recorded = scoring.get("score_within_tolerance")
    _delta_present, delta = _number(scoring, "score_max_abs_delta_vs_dot",
                                   where=f"{category}.scoring_ranking")
    if delta is None:
        if "score_within_tolerance" in scoring:
            raise _Malformed(f"{category}.scoring_ranking declares `score_within_tolerance` but "
                             "records no `score_max_abs_delta_vs_dot` to check it against")
        return []
    body_declared = scoring.get("score_max_abs_declared")
    top_declared = declared.get("score_max_abs")
    body_value = (None if body_declared is None else
                  _tolerance_value(body_declared,
                                   where=f"{category}.scoring_ranking.score_max_abs_declared"))
    top_value = None if top_declared is None else _tolerance_value(
        top_declared, where=f"{category}.tolerances.score_max_abs")
    if body_value is not None and top_value is not None and body_value > top_value:
        raise _Malformed(
            f"{category}: the body declares `score_max_abs_declared` {body_value!r} while the "
            f"artifact declares `score_max_abs` {top_value!r}; a looser body tolerance cannot "
            "override a stricter declared one")
    tolerance = body_value if body_value is not None else top_value
    if tolerance is None:
        raise _Malformed(f"{category}: a score delta of {delta!r} is recorded but the artifact "
                         "declares no `score_max_abs_declared`/`score_max_abs` tolerance")
    return [ParityCheck(
        name=f"{category}.scores.max_abs_vs_dot", dimension="scoring_parity", gate="G2",
        declared=tolerance, measured=delta,
        ok=delta <= tolerance, mandatory=True, harness_recorded=recorded)]


def _scoring_evidence(checks: list[ParityCheck],
                      category: str) -> tuple[list[ParityCheck], bool]:
    """`scoring_parity` needs BOTH a score comparison and a rank comparison.

    A score delta alone says the values agree at the compared pairs; it says nothing about whether
    the ORDER the scoring produces still matches, which is what Recall is computed from. So a
    category that records scores but no rank comparison gets a mandatory FAILURE naming the gap
    instead of a silent pass. The reverse is returned as `score_missing`: rank evidence alone must
    not satisfy the dimension either, and rule 1 turns that into `missing` (hence `INCOMPLETE`).
    """
    names = [check.name for check in checks if check.dimension == "scoring_parity"]
    has_score = any(".scores." in name for name in names)
    has_rank = any(".ranks." in name for name in names)
    if has_score and not has_rank:
        return [ParityCheck(
            name=f"{category}.ranks.evidence_present", dimension="scoring_parity", gate="G2",
            declared="a rank comparison beside the score comparison",
            measured="no rank comparison recorded", ok=False, mandatory=True, absence=True,
            note=("the artifact records score deltas but no rank comparison, so whether the "
                  "scoring still produces the same ORDER is unmeasured") )], False
    return [], has_rank and not has_score

def _unclaimable(check: ParityCheck) -> ParityCheck:
    """An absence-reporting check for a category whose adapter claims nothing.

    Non-mandatory, so it never lands in `failures`; `ok` stays False so the dimension is still
    reported False rather than absent. The dimension is separately reported `missing` by rule 1.
    """
    return ParityCheck(
        name=check.name, dimension=check.dimension, gate=check.gate, declared=check.declared,
        measured=check.measured, ok=check.ok, mandatory=False, standard=check.standard,
        harness_recorded=check.harness_recorded, absence=True,
        note=(check.note + "; " if check.note else "") +
             "the artifact claims NOTHING for this category, so this is absent evidence, not a "
             "failed claim")

def _check_invariants(verdict: ParityVerdict) -> None:
    """A pass is impossible on an empty or partly-measured mandatory set.

    Every early return above is covered because `to_dict` calls this. The check is deliberately
    loud: a future edit that reintroduces a false pass must break a test, not a report reader.
    """
    if not verdict.passed:
        return
    assert verdict.status == PASS, (verdict.model_id, verdict.status)
    assert not verdict.failures, [check.name for check in verdict.failures]
    assert not verdict.missing, verdict.missing
    assert any(check.mandatory for check in verdict.checks), verdict.model_id
    for category in verdict.categories:
        for dimension in REQUIRED_DIMENSIONS:
            assert dimension in verdict.mandatory_by_category.get(category, []), (
                f"{verdict.model_id}/{category}: {dimension} is required but carries no mandatory "
                "check, so this verdict cannot be a pass")

def _metrics_checks(native: dict[str, Any], category: str) -> list[ParityCheck]:
    """Hit-count equality. A non-boolean flag is malformed, never "not applicable"."""
    if "native_metrics_match_feature_rerank" not in native:
        return []
    value = native.get("native_metrics_match_feature_rerank")
    if not isinstance(value, bool):
        raise _Malformed(f"{category}.native_metrics_match_feature_rerank is {value!r}: only a "
                         "recorded boolean can certify or refuse hit-count equality")
    return [ParityCheck(
        name=f"{category}.metrics.hit_count_equality", dimension="metrics_parity", gate="G2",
        declared="exact hit-count equality", measured=value,
        ok=value is True, mandatory=True)]


def _observer_checks(native: dict[str, Any], category: str) -> list[ParityCheck]:
    """The observation pipeline reproducing its own macro. Its own dimension, never merged."""
    recheck = native.get("native_metrics_recheck")
    derived = native.get("native_rank_recall_from_features")
    if not isinstance(recheck, dict) or not isinstance(derived, dict):
        return []
    checks: list[ParityCheck] = []
    for key, value in recheck.items():
        metric = key.rsplit("_", 1)[-1]
        other = derived.get(metric)
        if not isinstance(value, (int, float)) or not isinstance(other, (int, float)):
            continue
        checks.append(ParityCheck(
            name=f"{category}.observer.{key}", dimension="observer_parity", gate="G2",
            declared="exact equality (same features, same hit counts)", measured=other,
            ok=value == other, mandatory=False,
            note=f"reported macro {value!r} vs rerank {other!r}; observer/macro agreement is not "
                 "forward parity"))
    return checks


def _declared_standard(native: dict[str, Any], adapter: dict[str, Any]) -> str:
    """The standard the ARTIFACTS declare, or `parity-relative/v1` when they declare none.

    This is what the verdict is computed under, and it is read from the evidence rather than from
    the caller. A caller may request v2; a v1 artifact stays a v1 result, because re-evaluating it
    under a standard it was not measured with moves a declared tolerance after seeing the number.

    An artifact that declares v2 must also carry the per-component comparison v2 is defined over
    (`abs(a-b) <= atol + rtol*abs(b)` per element). These artifacts record only `max_abs`/`max_rel`,
    which are v1 summary statistics: a v2 label over v1 measurements is malformed, not a v2 result.

    Two artifacts declaring different standards is malformed too -- the same rule as tolerances.
    """
    left, right = native.get("standard"), adapter.get("standard")
    for side, value in (("native", left), ("adapter", right)):
        if value is not None and value not in STANDARDS:
            raise _Malformed(f"the {side} artifact declares an unknown standard {value!r}")
    if left is not None and right is not None and left != right:
        raise _Malformed(f"the two artifacts declare different standards: native {left!r}, "
                         f"adapter {right!r}")
    declared = left or right or STANDARD_RELATIVE_V1
    if declared == STANDARD_ABS_REL_V2:
        raise _Malformed(
            "the artifact declares parity-absolute-relative/v2 but records only `max_abs`/`max_rel` "
            "aggregates; that standard is defined per component (abs(a-b) <= atol + rtol*abs(b)) and "
            "cannot be evaluated from the recorded statistics, so it must never be applied")
    return declared


def _body(artifact: dict[str, Any], *, side: str) -> dict[str, Any]:
    """The measurement block of one artifact, or a refusal when the shape is unusable."""
    if side == "native":
        inner = artifact.get("native")
        if inner is None:
            # LIMN's native artifact stores its layers at the top level; ENCODER's nests them.
            return artifact
    else:
        inner = artifact.get("adapter")
        if inner is None:
            inner = artifact.get("layer2_4_forward")
        if inner is None:
            raise _Malformed("the adapter artifact carries neither an `adapter` nor a "
                             "`layer2_4_forward` block, so it records no measurement")
    if not isinstance(inner, dict):
        raise _Malformed(f"the {side} artifact's measurement block is a "
                         f"{type(inner).__name__}, not an object")
    return inner


def _relation_checks(native: dict[str, Any], adapter: dict[str, Any],
                     category: str, model_id: str) -> list[ParityCheck]:
    """M-01: did BOTH sides compute their features with the SAME relation tensor?

    A model can draw its relation tensor at construction (`ENCODER` draws
    `learnable_relation` from the global RNG; it is not in `state_dict()` and no checkpoint restores
    it). Two sides that drew different bytes compared features computed with different relations, so
    a numeric agreement between them means nothing -- and `forward_parity` cannot see it, because it
    only reads layer outputs.

    Recorded per artifact as `relation_state: {sha256, expected_sha256, matches_pinned_digest}`. The
    three outcomes are kept apart on purpose, because collapsing them is how a silent mismatch would
    become a pass:

    * both record a digest and they AGREE  -> one check, mandatory, ok;
    * both record a digest and they DIFFER -> one check, mandatory, NOT ok;
    * either side records no digest        -> one check, NOT ok with `measured: "not_recorded"`.
      Whether it is MANDATORY depends on the model: `RELATION_DRAWING_MODELS` (and any artifact
      pair where either side carries the key at all) draws a relation, so an unrecorded digest is a
      mandatory failure there. A model that genuinely has no relation keeps the check non-mandatory
      -- `relation_parity` is reported False instead of vanishing -- so an old artifact is not
      retro-failed for a tensor its construction never had.
    """
    def record(document: dict[str, Any]) -> Any:
        """The `relation_state` key itself, or `_ABSENT` when the artifact predates the field."""
        return document["relation_state"] if "relation_state" in document else _ABSENT

    def digest(block: Any) -> Any:
        return block.get("sha256") if isinstance(block, dict) else None

    native_block, adapter_block = record(native), record(adapter)
    native_digest, adapter_digest = digest(native_block), digest(adapter_block)
    # A model whose CONSTRUCTION draws a relation tensor cannot pass without recording that both
    # sides drew the same one. `ENCODER` draws `learnable_relation` from the global RNG, so an
    # artifact that predates `relation_state` is `not_recorded`, which is a mandatory failure for it,
    # not an exemption. Any artifact that carries the key at all is treated the same way: recording
    # `not_recorded` states that a relation exists and was not pinned.
    draws_relation = (model_id in RELATION_DRAWING_MODELS or native_block is not _ABSENT
                      or adapter_block is not _ABSENT)
    name = f"relation.{category}.digest"
    measured = (f"native={'not_recorded' if native_digest is None else 'recorded'}, "
                f"adapter={'not_recorded' if adapter_digest is None else 'recorded'}")
    declared = ("native and adapter constructed the identical relation tensor"
                if draws_relation else "both sides record the relation tensor they constructed")
    if native_digest is None or adapter_digest is None:
        return [ParityCheck(
            name=name, dimension="relation_parity", gate="G1", mandatory=draws_relation, ok=False,
            declared=declared, measured=measured, harness_recorded=False, absence=True,
            note=("this model draws a relation tensor at construction, so an unrecorded digest is a "
                  "mandatory failure: whether the two sides computed features with the same relation "
                  "is unknown, and unknown is not a pass") if draws_relation else
                 ("at least one side does not record its relation digest, so whether the two computed "
                  "features with the same relation is unknown. Unknown is not a pass"))]
    return [ParityCheck(
        name=name, dimension="relation_parity", gate="G1", mandatory=True,
        ok=native_digest == adapter_digest,
        declared=declared,
        measured=f"native={str(native_digest)[:16]} adapter={str(adapter_digest)[:16]}",
        harness_recorded=True,
        note=None if native_digest == adapter_digest else
             "the two sides drew DIFFERENT relation tensors, so their features are not comparable")]

def evaluate_model(model_id: str, *, categories: Iterable[str] = ("dress", "shirt", "toptee"),
                   root: Path | None = None,
                   standard: str = STANDARD_RELATIVE_V1) -> ParityVerdict:
    """Derive one model's G1/G2 verdict from the artifacts, never from a prose surface.

    `standard` is a REQUEST. The verdict is computed under the standard the artifacts declare
    (`verdict.applied_standard`); a request for another standard is recorded beside it as advisory
    and cannot change `passed`.
    """
    if standard not in STANDARDS:
        raise ValueError(f"unsupported parity standard: {standard!r}")
    verdict = ParityVerdict(model_id=model_id, requested_standard=standard)
    base = root if root is not None else evidence_root()
    categories = tuple(categories)
    verdict.categories = categories
    missing_artifacts = []
    for category in categories:
        if _artifact_paths(base, model_id, category) is None:
            native_name, adapter_name = _artifact_names(model_id, category)
            missing_artifacts.extend(name for name in (native_name, adapter_name)
                                     if not (base / name).is_file())
    if len(missing_artifacts) == 2 * len(categories):
        # Nothing at all for this model. `NOT_RUN`, whatever its name: the gate judges evidence, and
        # there is none. A model whose artifacts ARE present is gated even if it is neither of the
        # two the campaign happened to measure first.
        verdict.status = NOT_RUN
        verdict.detail = (f"no G1/G2 artifact exists for {model_id}; absent evidence is never a "
                          "pass, so the gate stays NOT_RUN")
        return verdict
    if missing_artifacts:
        verdict.status = EVIDENCE_MISSING
        verdict.detail = f"missing parity artifact(s): {missing_artifacts}"
        return verdict

    historical: dict[str, Any] = {}
    standards: set[str] = set()
    for category in categories:
        resolved = _artifact_paths(base, model_id, category)
        if resolved is None:
            missing_artifacts = list(_artifact_names(model_id, category))
            verdict.status = EVIDENCE_MISSING
            verdict.detail = f"missing parity artifact(s): {missing_artifacts}"
            return verdict
        native_path, adapter_path = resolved
        try:
            native = _read(native_path)
            adapter = _read(adapter_path)
            verdict.evidence.extend([str(native_path), str(adapter_path)])
            declared = _tolerances(native, adapter)
            standards.add(_declared_standard(native, adapter))
            native_body = _body(native, side="native")
            adapter_body = _body(adapter, side="adapter")

            checks: list[ParityCheck] = []
            checks.extend(_identity_checks(native_body, declared, category))
            # G1 cross-side: both documents' identity fields, compared value to value. The native
            # side alone cannot tell whether the adapter measured the same checkpoint. An adapter
            # that measured NOTHING has no identity claim to contradict -- it is `INCOMPLETE` on the
            # rules below, not "measured and failed". Once it has a body, a field present on one side
            # only IS a failure.
            if adapter_body:
                checks.extend(_identity_cross_checks(native, adapter, category))
            checks.extend(_pixel_check(adapter_body, category))
            # M-01: read from the WHOLE document, not the `body`: the relation record sits beside the
            # measured payload, exactly as `policy_observed` does.
            checks.extend(_relation_checks(native, adapter, category, model_id))

            sample = (adapter_body.get("layer4_forward_sample")
                      or adapter_body.get("sample_forward"))
            if sample is None:
                sample = {}
            if not isinstance(sample, dict):
                raise _Malformed(f"{category}: the forward sample block is a "
                                 f"{type(sample).__name__}, not an object")
            for side in ("query_val_pixels", "target", "query"):
                block = sample.get(side)
                if block is None:
                    continue
                if not isinstance(block, dict):
                    raise _Malformed(f"{category}.forward.{side} is a {type(block).__name__}, "
                                     "not an object")
                checks.extend(_relative_checks(
                    block, declared, name=f"{category}.forward.{side}",
                    dimension="forward_parity", gate="G2",
                    atol_key="forward_max_abs", rtol_key="forward_max_rel"))
            population = adapter_body.get("population") or {}
            if not isinstance(population, dict):
                raise _Malformed(f"{category}: the population block is a "
                                 f"{type(population).__name__}, not an object")
            for side, block in population.items():
                if not isinstance(block, dict) or "max_abs" not in block:
                    continue
                if "train_query" in side:
                    # The comparison against the native TRAIN-transform query tensors measures the
                    # declared protocol divergence (preprocess_train vs preprocess_val), not parity.
                    # The artifact records it; it is never counted as a mandatory failure here.
                    continue
                checks.extend(_relative_checks(
                    block, declared, name=f"{category}.population.{side}",
                    dimension="forward_parity", gate="G2",
                    atol_key="forward_max_abs", rtol_key="forward_max_rel"))
            batch = adapter_body.get("batch_size_1_vs_n")
            if batch is not None:
                if not isinstance(batch, dict):
                    raise _Malformed(f"{category}.batch_size_1_vs_n is a "
                                     f"{type(batch).__name__}, not an object")
                checks.extend(_relative_checks(
                    batch, declared, name=f"{category}.batch_1_vs_n", dimension="batch_parity",
                    gate="G2", atol_key="batch_max_abs", rtol_key="batch_max_rel"))
            checks.extend(_rank_checks(adapter_body, declared, category))
            checks.extend(_score_checks(adapter_body, declared, category))
            # A score comparison without a rank comparison is half of `scoring_parity`. Do not let
            # the score half certify the dimension.
            evidence_gap, score_missing = _scoring_evidence(checks, category)
            checks.extend(evidence_gap)
            # ABSENT EVIDENCE IS NOT A FAILED CLAIM. A check that reports "nothing was recorded"
            # (`absence=True`) measures nothing, so an artifact pair that made NO claim for this
            # category must not be reported as "1 mandatory subcheck failed" -- that says a claim was
            # measured and did not hold, which is `MANDATORY_FAILED`'s meaning, not `INCOMPLETE`'s.
            # The adapter body is where the claims live: an empty `{}` claims nothing in this
            # category, so its absence checks become non-mandatory (the dimension then reports
            # `missing`, and `relation_parity` reports False). When the artifact DOES claim anything
            # here -- forward deltas, score deltas -- a missing relation digest stays a mandatory
            # failure beside those real claims.
            claims_present = bool(adapter_body)
            if not claims_present:
                checks = [check if not check.absence else _unclaimable(check) for check in checks]
            checks.extend(_metrics_checks(native_body, category))
            checks.extend(_observer_checks(native_body, category))

            # Keep the artifact's own verdicts verbatim: history is not rewritten, only aggregated.
            historical[category] = {
                "declared_tolerances": declared,
                "recorded_within_tolerance": {
                    key: value for key, value in (
                        ("batch_size_1_vs_n", (batch or {}).get("within_tolerance")),
                        ("sample_query_val_pixels",
                         ((sample.get("query_val_pixels") or {}).get("within_tolerance"))),
                    ) if value is not None},
                "recorded_non_tie_rank_changes":
                    ((adapter_body.get("scoring_ranking") or {})
                     .get("rank_differences_vs_native_val_control") or {}).get("non_tie_count"),
            }
            verdict.checks.extend(checks)
            # Per-category mandatory coverage. A required dimension measured in `shirt` is not
            # measured in `dress`: an adapter body of `{}` in one category used to be carried by the
            # other categories' checks, and the verdict reported PASS.
            have = {check.dimension for check in checks if check.mandatory}
            if score_missing:
                # Rank evidence alone: keep the rank checks (they are real), but the dimension was
                # never fully measured here, so it is `missing` rather than satisfied.
                have.discard("scoring_parity")
            missing_here = [dimension for dimension in REQUIRED_DIMENSIONS
                            if dimension not in have]
            verdict.mandatory_by_category[category] = [d for d in REQUIRED_DIMENSIONS if d in have]
            verdict.missing_by_category[category] = missing_here
        except _Malformed as error:
            verdict.status = MALFORMED
            verdict.detail = f"malformed parity artifact: {error}"
            return verdict

    if len(standards) > 1:
        verdict.status = MALFORMED
        verdict.detail = (f"the artifacts disagree on the declared standard: {sorted(standards)}; "
                          "one verdict cannot be reported under two standards")
        return verdict
    verdict.applied_standard = standards.pop()
    verdict.historical = historical
    # Coverage is derived from the checks that exist, so an unevaluated area cannot be mistaken for a
    # passing one. `population.*` forward comparisons and the batch check are the two that vary by
    # artifact: the LIMN artifact records no population forward delta at all.
    seen = {}
    for check in verdict.checks:
        seen.setdefault(check.dimension, set()).add(check.name.split(".")[-1])
    names = sorted({check.name for check in verdict.checks})
    for dimension in DIMENSIONS:
        have = sorted(seen.get(dimension, set()))
        gaps = []
        if dimension == "forward_parity":
            if not any(".population." in name for name in names):
                gaps.append("no population-level forward comparison recorded; only the fixed "
                            "sample was compared")
            if not any(name.endswith("min_cosine") for name in names):
                gaps.append("no cosine similarity recorded")
        if dimension == "batch_parity" and not any("batch_1_vs_n" in name for name in names):
            gaps.append("no batch 1-vs-N comparison recorded")
        if dimension == "metrics_parity" and "hit_count_equality" not in have:
            gaps.append("no hit-count equality recorded")
        if dimension == "observer_parity" and not have:
            gaps.append("the artifact records no observer-vs-macro recheck")
        verdict.coverage[dimension] = {"measured": have, "not_measured": gaps}

    # Where the harness's recorded boolean says True but the declared tolerance table says the
    # comparison failed, the difference is an INCONSISTENCY in the harness, not a pass. Naming it
    # keeps the old label readable without letting it certify the comparison.
    seen_pairs = set()
    for check in verdict.checks:
        if check.harness_recorded is None or check.dimension not in (
                "forward_parity", "batch_parity"):
            continue
        key = (check.name.rsplit(".", 1)[0], check.harness_recorded, check.ok)
        if check.harness_recorded is True and check.ok is False and key not in seen_pairs:
            seen_pairs.add(key)
            verdict.label_disagreements.append({
                "comparison": check.name.rsplit(".", 1)[0],
                "dimension": check.dimension,
                "subcheck": check.name.rsplit(".", 1)[-1],
                "declared": check.declared,
                "measured": check.measured,
                "harness_recorded": True,
                "gate_verdict": False,
                "why": ("the harness declared `forward_max_rel` but tested only `max_abs` for this "
                        "key; the declared table applies to every layered forward comparison, so the "
                        "gate applies it. The old boolean stays as the v1 label"),
            })

    mandatory = [check for check in verdict.checks if check.mandatory]
    verdict.failures = [check for check in mandatory if not check.ok]
    for dimension in DIMENSIONS:
        scoped = [check for check in mandatory if check.dimension == dimension]
        if scoped:
            verdict.dimensions[dimension] = all(check.ok for check in scoped)
        elif any(check.dimension == dimension and not check.ok for check in verdict.checks):
            # Measured-and-UNKNOWN (e.g. an artifact that predates `relation_state`). A mandatory
            # failure it is not -- an old artifact must keep the verdict it earned -- but it is not a
            # pass either, so the dimension is reported False with the check's own `note` beside it.
            verdict.dimensions[dimension] = False

    if not mandatory:
        verdict.status = INCOMPLETE
        verdict.missing = list(REQUIRED_DIMENSIONS)
        verdict.detail = ("the artifacts carry no mandatory subcheck this gate can evaluate; an "
                          f"unmeasured comparison is never {PASS}")
        return verdict

    # A mandatory dimension that NO category measured was never measured. Distinguishing that from
    # "measured and held" is the point of `INCOMPLETE`: an adapter artifact of `{}` reported PASS and
    # `complete: True`, because an empty check list was indistinguishable from a satisfied one.
    #
    # PER CATEGORY, not globally. `other_categories_empty` failed because the shirt checks were
    # aggregated over the whole model: dress and toptee carried nothing and the verdict still read
    # PASS. A dimension now counts only when it has a mandatory check in EVERY judged category.
    verdict.missing = [dimension for dimension in REQUIRED_DIMENSIONS
                       if any(dimension in verdict.missing_by_category.get(category, [])
                              for category in categories)]

    # VERDICT PRECEDENCE between `INCOMPLETE` and `MANDATORY_FAILED` is decided PER CATEGORY, in the
    # artifact loop above -- not here. Those two statuses answer different questions, and the more
    # accurate one wins:
    #   * a failure means a CLAIM was measured for that category and did not hold;
    #   * missing means the dimension was never measured for some category at all.
    # So an absence-reporting check (`absence=True`) is demoted to non-mandatory when its own
    # category's adapter body claims nothing; the dimension then reports `missing`. When the same
    # artifact DOES claim results in that category, the absence is a real failure beside real claims
    # and stays mandatory.
    #
    # A GLOBAL version of this rule would be wrong, and `other_categories_empty` is the fixture that
    # proves it: shirt claims forward and scoring results and records no relation digest (a genuine
    # mandatory failure), while dress and toptee claim nothing (missing). A global "failures are all
    # about absence => INCOMPLETE" test would throw away shirt's real failure and report a
    # partly-measured pass as merely incomplete.

    if standard != verdict.applied_standard:
        # v2 has not been measured on these artifacts, and applying it after the fact would move a
        # declared tolerance post hoc. It is reported for the record and cannot certify anything.
        verdict.advisory = [{
            "standard": standard,
            "declared_before_measurement": False,
            "applies": False,
            "applied_instead": verdict.applied_standard,
            "why": (f"the artifacts declare {verdict.applied_standard} tolerances and were measured "
                    f"under them; evaluating them under {standard} now would be a post-hoc tolerance "
                    "change and cannot contribute to a pass"),
            "form": ("fail if abs(a-b) > atol + rtol*abs(b) for any component"
                     if standard == STANDARD_ABS_REL_V2 else None),
        }]

    if verdict.failures:
        verdict.status = MANDATORY_FAILED
        verdict.passed = False
        verdict.complete = False
        names = ", ".join(sorted({check.name for check in verdict.failures}))
        verdict.detail = (
            f"{len(verdict.failures)} mandatory subcheck(s) failed: {names}. A mandatory failure is "
            "never reported as a pass, whatever a prose surface above it says")
        return verdict

    if verdict.missing:
        verdict.status = INCOMPLETE
        verdict.passed = False
        verdict.complete = False
        verdict.detail = (
            f"nothing failed, but {len(verdict.missing)} mandatory dimension(s) were never measured: "
            f"{', '.join(verdict.missing)}. Every check that EXISTS held, which is not the same claim "
            f"as a pass, so the gate reports {INCOMPLETE}")
        return verdict

    verdict.status = PASS
    verdict.passed = True
    verdict.complete = True
    verdict.detail = ("every mandatory subcheck of every mandatory dimension was measured and held "
                      "under its own declared tolerance")
    return verdict


def _discover_models(base: Path) -> list[str]:
    """Every model id with G1/G2 artifacts on disk, plus the known seed.

    `EVIDENCED_MODELS` used to be the SET of models the gate would judge, so a third model's
    artifacts could sit in the evidence directory and still be reported NOT_RUN -- the gate refused
    to look. Discovery reads the filenames instead: `g1_g2.<model>.<category>.<side>.json` for
    ENCODER's namespaced files, and the spec-model form `g1_g2.<category>.<side>.json` for LIMN's.
    Anything else under the root is left alone.
    """
    models = set(EVIDENCED_MODELS)
    try:
        names = sorted(path.name for path in base.iterdir())
    except OSError:
        return sorted(models)
    for name in names:
        if not (name.startswith("g1_g2.") and name.endswith(".json")):
            continue
        parts = name[len("g1_g2."):-len(".json")].split(".")
        if len(parts) < 3 or parts[-1] not in ("native", "adapter"):
            continue
        head = parts[:-1]
        if len(head) < 2 or head[-1] not in ("dress", "shirt", "toptee"):
            continue
        models.add(".".join(head[:-1]))
    return sorted(models)

def evaluate_all(*, root: Path | None = None,
                 standard: str = STANDARD_RELATIVE_V1) -> dict[str, Any]:
    """Every model with G1/G2 evidence on disk, each judged on its own artifacts.

    `standard` is reported as the standard actually APPLIED (the artifacts' own). A request that
    differs is reported beside it, so a caller cannot mistake a v2 request for a v2 result.
    """
    base = root if root is not None else evidence_root()
    verdicts = {model_id: evaluate_model(model_id, root=base, standard=standard).to_dict()
                for model_id in _discover_models(base)}
    applied = sorted({verdict["applied_standard"] for verdict in verdicts.values()
                      if verdict["status"] != NOT_RUN})
    return {
        "standard": applied[0] if len(applied) == 1 else (applied or STANDARD_RELATIVE_V1),
        "requested_standard": standard,
        "advisory_standards": sorted(set(STANDARDS) - set(applied)) if applied else [],
        "passed_models": sorted(model_id for model_id, verdict in verdicts.items() if verdict["passed"]),
        "models": verdicts,
        "note": ("`passed` is True only when every mandatory dimension was measured and no mandatory "
                 f"subcheck failed. No artifact is {NOT_RUN}; nothing measured held is {INCOMPLETE}; "
                 f"a measured subcheck outside its tolerance is {MANDATORY_FAILED}. Only {PASS} "
                 "claims a pass, and `standard` is the standard the artifacts declare."),
    }
