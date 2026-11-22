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

import base64
import copy
import binascii
import hashlib
import json
import math
import struct
import zlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable


def _fp32(value: float) -> float:
    return struct.unpack("=f", struct.pack("=f", value))[0]


def _score_order_matches(scores: list[float], order: list[int], convention: str) -> bool:
    if convention == "fp32_descending_score":
        keys, ascending = scores, False
    elif convention == "fp32_one_minus_score_ascending":
        keys, ascending = [_fp32(1.0 - _fp32(score)) for score in scores], True
    else:
        return False
    return all((keys[left] <= keys[right] if ascending else keys[left] >= keys[right])
               for left, right in zip(order, order[1:]))

#: The directory holding the G1/G2 measurement artifacts. Fixed, never request-derived.
PARITY_EVIDENCE_RELATIVE = ("workbench", "docs", "reviews", "20261109-native15-composition-limn")
PARITY_V3_PUBLISHED_RELATIVE = ("workbench", "artifacts", "parity", "v3")

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
STANDARD_ABS_REL_V3 = "parity-absolute-relative/v3"
STANDARDS = (STANDARD_RELATIVE_V1, STANDARD_ABS_REL_V2, STANDARD_ABS_REL_V3)

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
    category: str | None = None

    @property
    def check_type(self) -> str:
        """Stable classifier field; dimensions already distinguish input from component checks."""
        return self.dimension

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "dimension": self.dimension, "gate": self.gate,
                "declared": self.declared, "measured": self.measured, "ok": self.ok,
                "mandatory": self.mandatory, "standard": self.standard, "note": self.note,
                "reports_absent_evidence": self.absence,
                "harness_recorded": self.harness_recorded,
                "harness_recorded_within_tolerance": self.harness_recorded,
                "check_type": self.dimension,
                "category": self.category}


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
                "categories": list(self.categories),
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


def v3_published_root(repo_root: Path | None = None) -> Path:
    """Resolve the trusted atomic V3 publication pointer; never request-derived."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    publication = root.joinpath(*PARITY_V3_PUBLISHED_RELATIVE)
    pointer = publication / "CURRENT.json"
    try:
        current = json.loads(pointer.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return publication / "generations" / "unpublished"
    except (OSError, json.JSONDecodeError) as error:
        raise _Malformed(f"V3 publication pointer is unreadable: {error}") from error
    generation = current.get("generation") if isinstance(current, dict) else None
    if not isinstance(generation, str) or not generation or Path(generation).name != generation:
        raise _Malformed("V3 publication pointer has invalid generation ID")
    target = (publication / "generations" / generation).resolve()
    if not target.is_relative_to((publication / "generations").resolve()):
        raise _Malformed("V3 publication generation escapes trusted root")
    return target


class _Malformed(Exception):
    """Evidence that cannot be judged: wrong shape, non-finite, or two sides that contradict.

    Raised rather than returned so no consumer can continue past it: a comparison the gate cannot
    understand must become `PARITY_EVIDENCE_MALFORMED`, never an empty `[]` of checks -- which is
    what let an empty adapter body report `PARITY_PASS`.
    """


def _artifact_stamp(base: Path) -> tuple[tuple[str, int, int], ...]:
    """Stamp evidence stats plus external V3 proof bytes to invalidate exact-content cache hits.

    The gate reads artifacts in exactly three places, and all three resolve names under `base`:

    * `_discover_models` — `base.iterdir()`, keeping `g1_g2.*.json` with a `<category>.<side>` tail;
    * `_artifact_paths` — `base / name` for names built by `_artifact_names`, and the legacy bare
      `g1_g2.<category>.<side>.json` form;
    * `_read` — called only on the two paths `_artifact_paths` returned.

    Every one of those names matches `g1_g2.*.json`, so this glob covers the whole read set: a file
    APPEARING, DISAPPEARING, or being REWRITTEN (which changes mtime_ns and usually size) changes the
    stamp. `mtime_ns` is used rather than `st_mtime` so two writes inside the same second cannot hash
    to the same stamp. A directory that cannot be listed raises, and the caller then declines to cache
    -- an unreadable evidence root must never be cached as "no artifacts".

    Plan dependencies remain stat-stamped. Raw scorer proof JSONL is hashed in chunks because its
    bytes directly determine V3 verdicts; it is small compared with the component evidence cache.
    """
    if not base.is_dir():
        # `glob` yields nothing for a directory that is absent OR unlistable, which would make "the
        # evidence root is not there" indistinguishable from "it holds no artifacts" -- and the empty
        # stamp would then be CACHED. Raise instead, so the caller derives without caching.
        raise FileNotFoundError(f"{base} is not a readable directory")
    artifacts = tuple(base.glob("g1_g2.*.json"))
    stamps = {path.name: (path.stat().st_mtime_ns, path.stat().st_size) for path in artifacts}
    # V3 preregistration uses deterministic sibling naming, so cache dependencies can be statted
    # without parsing artifacts or reading component arrays. Missing plans are stamped as absent.
    for path in artifacts:
        if path.name.endswith((".native.json", ".adapter.json")):
            prefix = path.name.rsplit(".", 2)[0]
            dependency = base / f"{prefix}.v3-plan.json"
            try:
                stat = dependency.stat()
                stamps[dependency.name] = (stat.st_mtime_ns, stat.st_size)
            except FileNotFoundError:
                stamps[dependency.name] = (-1, -1)
            for side in ("native", "adapter"):
                proof = base / f"{prefix}.{side}.score-ranks.jsonl"
                try:
                    stat = proof.stat()
                    digest = hashlib.sha256()
                    with proof.open("rb") as stream:
                        while chunk := stream.read(1024 * 1024):
                            digest.update(chunk)
                    stamps[f"{proof.name}:{digest.hexdigest()}"] = (stat.st_mtime_ns, stat.st_size)
                except FileNotFoundError:
                    stamps[proof.name] = (-1, -1)
    return tuple(sorted((name, mtime, size) for name, (mtime, size) in stamps.items()))

#: The last `evaluate_all` answer, with the stamp it was derived from. ONE entry: a service that has
#: seen a hundred artifact revisions must not accumulate a hundred verdicts. Keyed on the stamp, the
#: requested standard, and the resolved root, because each of those changes the answer.
_EVALUATE_ALL_CACHE: list[tuple[tuple, dict[str, Any]]] = []


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
    """`abs(a-b) <= atol + rtol*abs(b)` for one component, the v2 rule.

    The production caller is `_component_scope_check`: v2 is defined per component, so it can only be
    evaluated over recorded components, never over a `max_abs`/`max_rel` aggregate.
    """
    return abs(actual - reference) <= atol + rtol * abs(reference)

#: The v2 component block: the per-element record `parity-absolute-relative/v2` is DEFINED over.
#: Additive by construction -- a v1 reader ignores an unknown key, and an artifact without the block
#: is still a v1 result. Shape and every refusal rule are written down first in
#: `workbench/docs/reviews/20261112-parity-repair/V2_STANDARD.md`.
COMPONENT_KEY = "component_parity"
COMPONENT_RULE = "abs(adapter - native) <= atol + rtol * abs(native)"
#: Which DECLARED tolerance pair judges a scope, chosen by the dimension the scope feeds. A block
#: cannot name its own tolerance: it is judged against the table this artifact already declares, so
#: a component dump can never loosen the rule it is measured by.
#:
#: Only `forward_parity` appears, and that is a statement about the reference side: the rule names
#: NATIVE as the reference, so only a comparison whose reference array came from the native side can
#: be recorded here. `batch_parity` (`batch 1 vs N`) and `cache_off_restart` compare the adapter
#: against its OWN output; their v1 aggregate checks judge them exactly as before, and calling their
#: left-hand array "native" would misdescribe the measurement.
COMPONENT_TOLERANCES = {"forward_parity": ("forward_max_abs", "forward_max_rel")}
V3_COMPONENT_TOLERANCES = {
    "forward_parity": ("forward_max_abs", "forward_max_rel"),
    "batch_parity": ("batch_max_abs", "batch_max_rel"),
}

#: How much of a scope the record actually carries. A CLOSED set: any other string is malformed.
#:
#: * `full`     -- every component of the array, in one list. Complete coverage.
#: * `chunks`   -- every component of the array, split into ordered, non-overlapping chunks that
#:                 tile `[0, components_total)` exactly. Complete coverage, each chunk bound to its
#:                 own bytes by a digest the reader RECOMPUTES.
#: * `chunked`  -- a PREVIEW: a head plus the offenders it happened to find. Partial coverage.
#:
#: The governing rule, and the reason `chunked` exists as a separate value rather than as a budget:
#: **a preview sample plus a full-array hash is partial coverage and can never be a PASS.** The hash
#: is an anchor for an external reviewer holding the arrays; it is not evidence ABOUT the components
#: nobody looked at. So a clean preview is `INCOMPLETE` -- never ok -- and a preview that DOES contain
#: a violating component is a genuine failure. Only `full` and `chunks` can earn a verdict.
COMPONENT_FULL = "full"
COMPONENT_CHUNKS = "chunks"
COMPONENT_PREVIEW = "chunked"
COMPONENT_COVERAGE = (COMPONENT_FULL, COMPONENT_CHUNKS, COMPONENT_PREVIEW)

#: The ONE serialization every component digest is taken over: canonical little-endian float64, in
#: flat order. Named once here so the reader's recomputation and the producers' `_component_digest`
#: cannot drift into two different definitions of "the same bytes". `struct` spells that dtype `<d`
#: and numpy spells it `<f8`: same eight bytes, little-endian IEEE-754 double.
COMPONENT_DIGEST_DTYPE = "<f8"
COMPONENT_DIGEST_FORMAT = "<d"
COMPONENT_DIGEST_KEYS = ("reference", "actual", "delta")
_COMPONENT_HEX = frozenset("0123456789abcdef")


def component_digest(values: Iterable[float]) -> str:
    """`sha256` over one component array, canonical little-endian float64, flat order.

    Stdlib only (`struct` + `hashlib`): the gate stays numpy-free, and the producers' numpy
    `ascontiguousarray(a, dtype="<f8").tobytes()` produces the same bytes, so a digest written by a
    producer can be recomputed here. This is the reader-side half of that one definition.
    """
    packed = struct.Struct(COMPONENT_DIGEST_FORMAT)
    digest = hashlib.sha256()
    for value in values:
        digest.update(packed.pack(float(value)))
    return digest.hexdigest()

_COMPONENT_PACK = struct.Struct(COMPONENT_DIGEST_FORMAT)


def _decode_components(value: Any, key: str, *, where: str) -> list[float]:
    """One component array, from either transport: a list of numbers, or base64 canonical bytes.

    Two transports because the scopes differ by four orders of magnitude. A 2048-component sample is
    readable as numbers and a reviewer should be able to read it; a 1.5M-component population scope is
    20 bytes of JSON per component as numbers, which turns a review artifact into a binary blob with
    JSON punctuation. Base64 of zlib-compressed canonical `<f8` bytes is a third of that and decompresses
    to the exact bytes the digest is taken over -- nothing about the evidence changes, only its
    transport, and the compression is lossless by construction.

    Both are decoded to the same `list[float]`, so every check downstream is written once.
    """
    if isinstance(value, str):
        try:
            raw = zlib.decompress(base64.b64decode(value, validate=True))
        except (binascii.Error, ValueError, zlib.error):
            raise _Malformed(f"{where}.{key} is a string that is not valid base64(zlib(...))") from None
        if len(raw) % _COMPONENT_PACK.size:
            raise _Malformed(f"{where}.{key} decodes to {len(raw)} bytes, not a whole number of "
                             f"{_COMPONENT_PACK.size}-byte little-endian float64 values")
        return list(struct.unpack(f"{len(raw) // _COMPONENT_PACK.size}d", raw))
    if not isinstance(value, list):
        raise _Malformed(f"{where}.{key} is a {type(value).__name__}, not a list of components or a "
                         "base64 string of canonical float64 components")
    numbers: list[float] = []
    for position, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise _Malformed(f"{where}.{key}[{position}] is {item!r}, not a number")
        if not math.isfinite(float(item)):
            raise _Malformed(f"{where}.{key}[{position}] is {item!r}: a non-finite component cannot "
                             "be judged against a tolerance, so it is malformed evidence, not a pass")
        numbers.append(float(item))
    return numbers


def _component_numbers(record: dict[str, Any], key: str, *, where: str) -> list[float]:
    """One recorded component array. A non-number or a non-finite value is malformed, never a pass."""
    numbers = _decode_components(record.get(key), key, where=where)
    for position, value in enumerate(numbers):
        if not math.isfinite(value):
            raise _Malformed(f"{where}.{key}[{position}] is {value!r}: a non-finite component cannot "
                             "be judged against a tolerance, so it is malformed evidence, not a pass")
    return numbers

def _component_indices(record: dict[str, Any], total: int, *, where: str) -> list[int]:
    """The flat indices of the emitted components: ascending, unique, inside `[0, total)`."""
    values = record.get("indices")
    if not isinstance(values, list):
        raise _Malformed(f"{where}.indices is a {type(values).__name__}, not a list of indices")
    indices: list[int] = []
    for position, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int):
            raise _Malformed(f"{where}.indices[{position}] is {value!r}, not an integer index")
        if not 0 <= value < total:
            raise _Malformed(f"{where}.indices[{position}] is {value} but the scope holds "
                             f"{total} component(s)")
        indices.append(value)
    if indices != sorted(set(indices)):
        raise _Malformed(f"{where}.indices must be ascending and unique, got {indices[:10]}")
    return indices

def _component_digests(record: dict[str, Any], *, where: str) -> dict[str, str]:
    """The three recorded digests, each REQUIRED to be 64 hex characters.

    Presence and shape only: what the digests are FOR differs by coverage form, and the caller
    decides. `full`/`chunks` recompute them from the record's own values (see `_verify_digests`);
    a `chunked` preview has no complete array to recompute against, so its digests are an unverified
    anchor and cannot contribute to a verdict either way.
    """
    digests = record.get("sha256")
    if not isinstance(digests, dict):
        raise _Malformed(f"{where}.sha256 is a {type(digests).__name__}, not an object of digests")
    checked: dict[str, str] = {}
    for key in COMPONENT_DIGEST_KEYS:
        value = digests.get(key)
        if (not isinstance(value, str) or len(value) != 64
                or not set(value.lower()) <= _COMPONENT_HEX):
            raise _Malformed(f"{where}.sha256.{key} is {value!r}, not a sha256 hex digest: without "
                             "the digest the emitted components are an unbound claim")
        checked[key] = value.lower()
    return checked


def _verify_digests(record: dict[str, Any], reference: list[float], actual: list[float], *,
                    where: str) -> None:
    """Recompute all three digests from the values the record CARRIES, and refuse a mismatch.

    This is what turns a digest from an assertion into a binding: the record cannot present one set
    of components and another set's digest. `delta` is `abs(actual - reference)` component-wise, the
    same quantity the producers hash.
    """
    recorded = _component_digests(record, where=where)
    delta = [abs(a - b) for a, b in zip(actual, reference)]
    for key, values in (("reference", reference), ("actual", actual), ("delta", delta)):
        derived = component_digest(values)
        if derived != recorded[key]:
            raise _Malformed(f"{where}.sha256.{key} is {recorded[key]!r} but the components it "
                             f"carries hash to {derived!r}: the digest does not bind these values")

class _Chunk:
    """One emitted chunk, already shape-checked: its flat indices and the two aligned value lists."""

    __slots__ = ("where", "indices", "reference", "actual")

    def __init__(self, where: str, indices: list[int], reference: list[float],
                 actual: list[float]) -> None:
        self.where = where
        self.indices = indices
        self.reference = reference
        self.actual = actual


def _chunk_payload(record: Any, total: int, *, where: str) -> _Chunk:
    """One `{start, reference, actual, sha256}` chunk, shape-checked against `total`.

    A chunk IS a contiguous run, so its flat indices are `range(start, start + len(reference))` and
    spelling them out would cost ~40% of the payload to say nothing. `indices` is still accepted when
    present and must agree with the run, so a record written either way reads the same.

    Every rule the reader applies to a whole scope applies here to a chunk: integers in range,
    ascending and unique, the value lists the same length, values finite. A chunk is not a lesser
    claim about its own bytes.
    """
    if not isinstance(record, dict):
        raise _Malformed(f"{where} is a {type(record).__name__}, not an object")
    reference = _component_numbers(record, "reference", where=where)
    actual = _component_numbers(record, "actual", where=where)
    if not (len(reference) == len(actual)):
        raise _Malformed(f"{where} carries {len(reference)} reference value(s) and {len(actual)} "
                         "actual value(s); a chunk must be aligned")
    if not reference:
        raise _Malformed(f"{where} carries no components: an empty chunk measures nothing")
    start = record.get("start")
    if isinstance(start, bool) or not isinstance(start, int):
        raise _Malformed(f"{where}.start is {start!r}, not an integer flat index")
    if not 0 <= start < total:
        raise _Malformed(f"{where}.start is {start} but the scope holds {total} component(s)")
    indices = list(range(start, start + len(reference)))
    if indices[-1] >= total:
        raise _Malformed(f"{where} runs to flat index {indices[-1]} but the scope holds {total} "
                         "component(s)")
    if "indices" in record:
        declared = _component_indices(record, total, where=where)
        if declared != indices:
            raise _Malformed(f"{where}.indices is not the contiguous run its `start` implies "
                             f"({declared[:4]}... vs {indices[:4]}...)")
    _verify_digests(record, reference, actual, where=where)
    return _Chunk(where, indices, reference, actual)


def _chunks_tile(scope: dict[str, Any], total: int, *, where: str):
    """Yield validated chunks while proving exact tiling with bounded memory."""
    records = scope.get("chunks")
    if not isinstance(records, list) or not records:
        raise _Malformed(f"{where}.chunks is {records!r}: the `{COMPONENT_CHUNKS}` form carries its "
                         "components as a non-empty list of chunks")
    expected = 0
    for position, item in enumerate(records):
        chunk = _chunk_payload(item, total, where=f"{where}.chunks[{position}]")
        if chunk.indices[0] != expected:
            raise _Malformed(f"{where}.chunks[{position}] starts at flat index {chunk.indices[0]} but "
                             f"the components before it cover [0, {expected}); the chunks must tile "
                             "[0, components_total) in order with no gap and no overlap")
        expected += len(chunk.indices)
        if expected > total:
            raise _Malformed(f"{where}.chunks[{position}] ends at {expected}, past the "
                             f"{total} component(s) the scope declares")
        yield chunk
    if expected != total:
        raise _Malformed(f"{where}.chunks cover [0, {expected}) but the scope declares {total} "
                         f"component(s): the tail was never emitted, so this record is not complete "
                         "coverage")



def _component_scope_check(name: str, record: Any, declared: dict[str, Any],
                           category: str, *, standard: str = STANDARD_ABS_REL_V2,
                           tolerances: dict[str, tuple[str, str]] = COMPONENT_TOLERANCES) -> ParityCheck:
    """One v2 component scope: judged per component, across EVERY component it covers.

    Three coverage forms, and the verdict differs because the EVIDENCE differs:

    * `full` / `chunks` -- complete coverage. Every component is present, every digest is recomputed
      from the values carried, and the rule is applied to all of them. `ok` needs zero derived
      offenders. This is the only form that can be `ok`.
    * `chunked` -- a preview (head plus whatever offenders it found). It is judged on what it
      carries: a violating component among the emitted ones FAILS the scope, because a real violation
      is a real violation however it was sampled. A clean preview is NOT ok -- it is a partial
      measurement reported as absent, so the dimension reads `INCOMPLETE`. A full-array digest over
      components nobody looked at cannot upgrade a sample into a proof, which is the defect this
      form used to have.

    In every form the recorded summary must be consistent with the values actually carried, and a
    record whose own `worst_normalized` exceeds the rule's unit threshold while naming no offender is
    self-contradictory: the writer emits every violating component, so the two cannot both be true.
    """
    where = f"{category}.{COMPONENT_KEY}.{name}"
    if not isinstance(record, dict):
        raise _Malformed(f"{where} is a {type(record).__name__}, not an object")
    dimension = record.get("dimension")
    if dimension not in tolerances:
        raise _Malformed(f"{where}.dimension is {dimension!r}: the componentwise rule is recorded "
                         f"only for {sorted(tolerances)}, the dimensions the producers compare "
                         "aligned arrays for")
    atol_key, rtol_key = tolerances[dimension]
    for key in (atol_key, rtol_key):
        if key not in declared:
            raise _Malformed(f"{where} is judged against the declared `{atol_key}`/`{rtol_key}` and "
                             f"the artifact declares no `{key}`, so this comparison cannot be "
                             "evaluated")
    atol, rtol = float(declared[atol_key]), float(declared[rtol_key])
    total = record.get("components_total")
    if isinstance(total, bool) or not isinstance(total, int) or total <= 0:
        raise _Malformed(f"{where}.components_total is {total!r}: a scope with no components has "
                         "nothing to judge, and an empty comparison is never ok")
    shape = record.get("shape")
    if (not isinstance(shape, list) or not shape
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                   for value in shape)):
        raise _Malformed(f"{where}.shape is {shape!r}, not a list of non-negative integers")
    if math.prod(shape) != total:
        raise _Malformed(f"{where}.shape {shape} holds {math.prod(shape)} components but "
                         f"`components_total` says {total}")
    coverage = record.get("coverage")
    if coverage not in COMPONENT_COVERAGE:
        raise _Malformed(f"{where}.coverage is {coverage!r}, not one of {list(COMPONENT_COVERAGE)}")

    if coverage == COMPONENT_CHUNKS:
        chunks = _chunks_tile(record, total, where=where)
    else:
        indices = _component_indices(record, total, where=where)
        reference = _component_numbers(record, "reference", where=where)
        actual = _component_numbers(record, "actual", where=where)
        emitted = record.get("components_emitted")
        if isinstance(emitted, bool) or not isinstance(emitted, int):
            raise _Malformed(f"{where}.components_emitted is {emitted!r}, not a count")
        if not (emitted == len(indices) == len(reference) == len(actual)):
            raise _Malformed(f"{where} claims {emitted} emitted component(s) but carries "
                             f"{len(indices)} index/ies, {len(reference)} reference value(s) and "
                             f"{len(actual)} actual value(s)")
        if emitted > total:
            raise _Malformed(f"{where} emits {emitted} of {total} component(s)")
        if (coverage == COMPONENT_FULL) != (emitted == total):
            raise _Malformed(f"{where}.coverage is {coverage!r} while {emitted} of {total} "
                             "component(s) were emitted: `full` describes the whole array and "
                             "nothing else")
        if coverage == COMPONENT_FULL:
            _verify_digests(record, reference, actual, where=where)
        else:
            _component_digests(record, where=where)
        chunks = iter((_Chunk(where, indices, reference, actual),))

    offenders: list[int] = []
    worst = 0.0
    unbounded = False
    compared = 0
    derived_max_abs = 0.0
    #: Values for the first few offenders only, because that is all `offending_values` reads. The
    #: earlier form built one dict entry per component -- ~195 MiB and ~0.55 s of the run on a
    #: 1.5M-component scope -- to build a map consulted at most five times, and then walked it again
    #: to fold `derived_max_abs`. Both folds happen here, inside the pass that already computes
    #: `delta`, so each component is visited once.
    shown_values: dict[int, tuple[float, float]] = {}
    root_digests = {key: hashlib.sha256() for key in COMPONENT_DIGEST_KEYS}
    chunk_count = 0
    for chunk in chunks:
        chunk_count += 1
        compared += len(chunk.indices)
        for index, actual_value, reference_value in zip(chunk.indices, chunk.actual, chunk.reference):
            root_digests["reference"].update(_COMPONENT_PACK.pack(reference_value))
            root_digests["actual"].update(_COMPONENT_PACK.pack(actual_value))
            root_digests["delta"].update(_COMPONENT_PACK.pack(abs(actual_value - reference_value)))
            if not evaluate_abs_rel(actual_value, reference_value, atol=atol, rtol=rtol):
                offenders.append(index)
                if len(shown_values) < 5:
                    shown_values[index] = (reference_value, actual_value)
            denominator = atol + rtol * abs(reference_value)
            delta = abs(actual_value - reference_value)
            if delta > derived_max_abs:
                derived_max_abs = delta
            if denominator > 0.0:
                if delta / denominator > worst:
                    worst = delta / denominator
            elif delta > 0.0:
                unbounded = True
    if coverage == COMPONENT_CHUNKS:
        recorded = _component_digests(record, where=where)
        for key, digest in root_digests.items():
            if digest.hexdigest() != recorded[key]:
                raise _Malformed(f"{where}.sha256.{key} is {recorded[key]!r} but the complete "
                                 f"chunk payload hashes to {digest.hexdigest()!r}: root digest does "
                                 "not bind these components")

    complete = coverage != COMPONENT_PREVIEW
    if complete and compared != total:
        # Belt and braces: `_chunks_tile`/the `full` check already prove this, and a verdict must not
        # depend on one guard being right.
        raise _Malformed(f"{where} claims complete coverage ({coverage!r}) but carries {compared} of "
                         f"{total} component(s)")

    recorded_mismatches = record.get("mismatch_count")
    if isinstance(recorded_mismatches, bool) or not isinstance(recorded_mismatches, int):
        raise _Malformed(f"{where}.mismatch_count is {recorded_mismatches!r}, not a count")
    if recorded_mismatches != len(offenders):
        raise _Malformed(f"{where}.mismatch_count is {recorded_mismatches} but the components it "
                         f"carries hold {len(offenders)} violation(s) at {offenders[:5]}: the record "
                         "contradicts the values it carries")
    recorded_offenders = record.get("offenders")
    if not isinstance(recorded_offenders, list):
        raise _Malformed(f"{where}.offenders is a {type(recorded_offenders).__name__}, not a list")
    offender_indices: list[int] = []
    for position, item in enumerate(recorded_offenders):
        index = item.get("index") if isinstance(item, dict) else None
        if isinstance(index, bool) or not isinstance(index, int):
            raise _Malformed(f"{where}.offenders[{position}] carries no integer `index`")
        offender_indices.append(index)
    if sorted(offender_indices) != offenders:
        raise _Malformed(f"{where}.offenders names {sorted(offender_indices)[:10]} but the "
                         f"components it carries violate the rule at {offenders[:10]}")

    present_abs, recorded_max_abs = _number(record, "max_abs", where=where)
    if present_abs and recorded_max_abs is not None:
        if complete and not math.isclose(recorded_max_abs, derived_max_abs, rel_tol=1e-9):
            raise _Malformed(f"{where}.max_abs is {recorded_max_abs!r} but the components it "
                             f"carries hold {derived_max_abs!r}")
        if recorded_max_abs < derived_max_abs:
            raise _Malformed(f"{where}.max_abs is {recorded_max_abs!r}, below {derived_max_abs!r}: a "
                             "summary cannot be smaller than the values it summarises")
    present_worst, recorded_worst = _number(record, "worst_normalized", where=where)
    if present_worst and recorded_worst is not None and not unbounded:
        if complete and not math.isclose(recorded_worst, worst, rel_tol=1e-9):
            raise _Malformed(f"{where}.worst_normalized is {recorded_worst!r} but the components it "
                             f"carries hold {worst!r}")
        if recorded_worst < worst:
            raise _Malformed(f"{where}.worst_normalized is {recorded_worst!r}, below {worst!r}")
        if recorded_worst > 1.0 and not offenders:
            # The writer emits every component whose normalized violation exceeds 1. A record that
            # says one exists while naming none is contradictory in EVERY coverage form -- and in a
            # preview this is exactly how a removed offender used to disappear.
            raise _Malformed(f"{where}.worst_normalized is {recorded_worst!r} (> 1) but no offending "
                             "component is recorded: a violation above the rule's own unit "
                             "threshold must name its coordinate, whatever the coverage")

    if offenders:
        ok = False
        note = (f"{len(offenders)} component(s) violate the rule; the verdict is derived from the "
                f"recorded components, not from the aggregate `max_abs`/`max_rel`")
    elif complete:
        ok = True
        note = (f"every one of the {compared} component(s) satisfies the rule under the declared "
                f"`{atol_key}`/`{rtol_key}`; coverage {coverage!r} means the whole array was judged")
    else:
        ok = False
        note = (f"PREVIEW ONLY: {compared} of {total} component(s) were inspected and every one of "
                f"them is inside the rule, which is not evidence about the other {total - compared}. "
                f"A sample plus a full-array digest is partial coverage and can never be a pass")
    return ParityCheck(
        name=f"{category}.components.{name}", dimension=dimension, gate="G2",
        declared={"rule": COMPONENT_RULE, "atol": atol, "rtol": rtol,
                  "atol_key": atol_key, "rtol_key": rtol_key},
        measured={
            "scope": name, "coverage": coverage,
            "components_compared": compared, "components_total": total,
            "complete_coverage": complete,
            "chunks": chunk_count if coverage == COMPONENT_CHUNKS else None,
            "mismatch_count": len(offenders),
            "worst_normalized": None if unbounded else worst,
            "offending_coordinates": offenders[:64],
            "offending_values": [
                {"index": index, "reference": shown_values[index][0],
                 "actual": shown_values[index][1]}
                for index in offenders[:5]],
        },
        ok=ok, mandatory=complete or bool(offenders), absence=(not complete and not offenders),
        standard=standard, note=note)


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

#: The identifiers a hit-count comparison must declare before it can certify anything. Keyed by the
#: top-level field, with the human name used in the refusal note. Reported as ONE mandatory check so
#: the note can list every missing identifier at once instead of one per run.
METRICS_PROVENANCE = (
    ("protocol_id", "protocol"),
    ("protocol", "protocol"),
    ("checkpoint_id", "checkpoint"),
    ("query_order_sha256", "query order digest"),
    ("gallery_order_sha256", "gallery order digest"),
)


def _identity_value(document: dict[str, Any], field: str) -> Any:
    """One provenance field from either document, or `_ABSENT`."""
    return document[field] if field in document else _ABSENT


def _recorded_flag(native: dict[str, Any], adapter: dict[str, Any]) -> Any:
    """The recorded `native_metrics_match_feature_rerank`, wherever the producer put it.

    LIMN and HINT write it at the TOP level of the native document (beside the block it summarises);
    ENCODER writes it inside its nested `native` body, beside the metric recheck it derives from. Both
    are the same claim, so both are read -- and two different values on one side is malformed.
    """
    found = []
    for label, document in (("native", native), ("adapter", adapter)):
        for place_name, place in ((label, document),
                                  (f"{label}.native", document.get("native")),
                                  (f"{label}.adapter", document.get("adapter"))):
            if not isinstance(place, dict):
                continue
            if "native_metrics_match_feature_rerank" in place:
                found.append((place_name, place["native_metrics_match_feature_rerank"]))
    if not found:
        return None
    distinct = {repr(value) for _place, value in found}
    if len(distinct) > 1:
        raise _Malformed("the artifacts record `native_metrics_match_feature_rerank` more than once "
                         f"with different values: {found}; one claim cannot have two answers")
    return found[0][1]


def _metrics_detail(native: dict[str, Any], adapter: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """The per-side hit-count block, whichever side carries it, and which side that was.

    Both producers write it into the NATIVE artifact (the phase that measured both vectors), and the
    key is deliberately accepted on the adapter side too: `limn_hits.py` mirrors the tolerance into
    both documents, and a future producer may record the counts where it ranked. The block is
    compared, never trusted -- what it says about itself is only an assertion to cross-check.
    """
    for side, document in (("native", native), ("adapter", adapter)):
        for key in ("native_metrics_parity_detail", "adapter_metrics_parity_detail"):
            block = document.get(key)
            if block is None:
                continue
            if not isinstance(block, dict):
                raise _Malformed(f"the {side} artifact's `{key}` is a "
                                 f"{type(block).__name__}, not an object")
            return block, f"{side}.{key}"
    return None, ""


def _hit_counts(block: dict[str, Any], side: str, *, where: str) -> dict[str, int]:
    """One side's integer hit counts, keyed by cutoff. A missing or non-integer count is malformed."""
    entry = block.get(side)
    if not isinstance(entry, dict):
        raise _Malformed(f"{where}.{side} is a {type(entry).__name__}, not an object with `hits`")
    hits = entry.get("hits")
    if not isinstance(hits, dict) or not hits:
        raise _Malformed(f"{where}.{side}.hits is {hits!r}: without per-cutoff counts there is "
                         "nothing to derive hit-count equality from")
    counts: dict[str, int] = {}
    for key, value in hits.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise _Malformed(f"{where}.{side}.hits.{key} is {value!r}, not an integer hit count")
        if value < 0:
            raise _Malformed(f"{where}.{side}.hits.{key} is {value!r}, not a count")
        counts[key] = value
    return counts


def _metrics_checks(native: dict[str, Any], adapter: dict[str, Any],
                    category: str) -> list[ParityCheck]:
    """Hit-count equality DERIVED from the per-side counts, with the recorded flag only a cross-check.

    The defect this replaces: the gate read `native_metrics_match_feature_rerank` and used the boolean
    AS the verdict, so an artifact could record `native r1 = 375`, `adapter r1 = 376` and still be
    certified -- the mismatch was written beside the flag and never read. Now:

    * `equality` is derived from the two recorded count vectors, cutoff by cutoff;
    * the recorded flag must AGREE with that derivation -- a disagreement is `_Malformed`, because the
      document cannot be both right and wrong, and a `true` flag beside differing counts is exactly
      the artifact that used to pass;
    * a flag with no counts behind it is not evidence: the check is absent (hence `INCOMPLETE`), never
      a pass;
    * a certifiable comparison also needs provenance -- both denominators, a protocol id, a checkpoint
      identity and both order digests. Absent provenance beside a recorded comparison is a mandatory
      FAILURE naming what is missing, because the artifact claimed a comparison and cannot show that
      the two sides compared the same queries over the same gallery.
    """
    detail, detail_where = _metrics_detail(native, adapter)
    recorded = _recorded_flag(native, adapter)
    if recorded is not None and not isinstance(recorded, bool):
        raise _Malformed(f"{category}.native_metrics_match_feature_rerank is {recorded!r}: only a "
                         "recorded boolean can certify or refuse hit-count equality")

    if detail is None:
        if recorded is None:
            return []
        # A bare boolean with nothing behind it. Non-mandatory + absence so the dimension reads
        # `missing` (INCOMPLETE) instead of passing on a self-assertion.
        return [ParityCheck(
            name=f"{category}.metrics.hit_count_equality", dimension="metrics_parity", gate="G2",
            declared="exact hit-count equality derived from per-side counts",
            measured={"recorded_flag": recorded, "counts": None},
            ok=False, mandatory=False, absence=True,
            note=("the artifact records `native_metrics_match_feature_rerank` but no per-side hit "
                  "counts (`native_metrics_parity_detail` / `adapter_metrics_parity_detail`), so "
                  "equality is unmeasurable: a boolean asserting its own conclusion is not evidence"))]

    cutoffs = detail.get("cutoffs")
    if not isinstance(cutoffs, list) or not cutoffs:
        raise _Malformed(f"{detail_where}.cutoffs is {cutoffs!r}: the declared cutoffs are what the "
                         "counts are compared at")
    for value in cutoffs:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise _Malformed(f"{detail_where}.cutoffs holds {value!r}, not a positive integer cutoff")
    native_hits = _hit_counts(detail, "native", where=detail_where)
    adapter_hits = _hit_counts(detail, "adapter", where=detail_where)
    keys = [f"r{k}" for k in cutoffs]
    for side, counts in (("native", native_hits), ("adapter", adapter_hits)):
        missing = [key for key in keys if key not in counts]
        if missing:
            raise _Malformed(f"{detail_where}.{side}.hits records {sorted(counts)} but the declared "
                             f"cutoffs need {missing}: a comparison missing a cutoff cannot be "
                             "certified at that cutoff")

    derived = {key: native_hits[key] == adapter_hits[key] for key in keys}
    derived_equal = all(derived.values())

    # The counts must also be counts OF THE SAME POPULATION, or equal hits mean nothing.
    native_entry = detail.get("native") if isinstance(detail.get("native"), dict) else {}
    adapter_entry = detail.get("adapter") if isinstance(detail.get("adapter"), dict) else {}
    denominators = {"native": native_entry.get("query_count"),
                    "adapter": adapter_entry.get("query_count")}
    counts_agree = (isinstance(denominators["native"], int) and not isinstance(
                        denominators["native"], bool)
                    and denominators["native"] == denominators["adapter"])

    checks: list[ParityCheck] = []

    recorded_equality = detail.get("equality")
    if isinstance(recorded_equality, dict):
        contradicted = {key: recorded_equality[key] for key in keys
                        if key in recorded_equality and recorded_equality[key] is not derived[key]}
        if contradicted:
            raise _Malformed(
                f"{detail_where}.equality says {contradicted} but the recorded counts derive "
                f"{ {key: derived[key] for key in contradicted} } (native "
                f"{ {key: native_hits[key] for key in contradicted} }, adapter "
                f"{ {key: adapter_hits[key] for key in contradicted} }): the block contradicts the "
                "counts it carries")

    if recorded is not None and recorded is not derived_equal:
        raise _Malformed(
            f"{category}.native_metrics_match_feature_rerank is {recorded} but the recorded hit "
            f"counts derive {derived_equal} (native { {k: native_hits[k] for k in keys} } vs adapter "
            f"{ {k: adapter_hits[k] for k in keys} }): the flag is an assertion to cross-check, and "
            "an artifact that contradicts its own counts cannot certify anything")

    checks.append(ParityCheck(
        name=f"{category}.metrics.hit_count_equality", dimension="metrics_parity", gate="G2",
        declared="exact hit-count equality at every declared cutoff, derived from the two count vectors",
        measured={"derived_equal": derived_equal, "per_cutoff": derived,
                  "native": {key: native_hits[key] for key in keys},
                  "adapter": {key: adapter_hits[key] for key in keys},
                  "recorded_flag": recorded},
        ok=derived_equal, mandatory=True, harness_recorded=recorded,
        note=("derived from the per-side counts; the recorded flag is kept as a cross-check and a "
              "disagreement is malformed evidence, never a pass" if recorded is not None else
              "derived from the per-side counts; the artifact records no flag")))

    # THREE outcomes per check, never two. The gate's own rule is that absent evidence is
    # `INCOMPLETE` and a contradicted claim is a FAILURE, and a check whose `measured` is two nulls
    # must not blur them:
    #
    #   * recorded on both sides and equal        -> ok, mandatory;
    #   * recorded on both sides and different    -> a MEASURED DISAGREEMENT: mandatory failure;
    #   * absent (one or both sides)              -> an ABSENT-EVIDENCE refusal: `absence=True`,
    #                                                non-mandatory, so the dimension reads `missing`
    #                                                and the verdict is `INCOMPLETE`.
    #
    # The middle case is the one the campaign cares about (equal hits over different denominators are
    # not equality). The third case is what a producer that has not recorded denominators yet looks
    # like, and calling that "failed" would claim it measured something it never measured.
    denominators_present = {side: isinstance(value, int) and not isinstance(value, bool)
                            for side, value in denominators.items()}
    if all(denominators_present.values()):
        denominators_missing: list[str] = []
    else:
        denominators_missing = [f"`{side}.query_count`" for side, present
                                in denominators_present.items() if not present]
    checks.append(ParityCheck(
        name=f"{category}.metrics.eligible_denominators", dimension="metrics_parity", gate="G2",
        declared="the same eligible query denominator on both sides",
        measured={"native": denominators["native"], "adapter": denominators["adapter"]},
        ok=counts_agree, mandatory=not denominators_missing, absence=bool(denominators_missing),
        note="" if counts_agree else (
            "ABSENT DENOMINATOR: the block records no " + ", ".join(denominators_missing)
            + ", so the two count vectors cannot be shown to describe the same eligible population. "
              "This is an unmeasured input, not a measured disagreement"
            if denominators_missing else
            "MEASURED DISAGREEMENT: equal hit counts over different denominators are not equality; "
            "one side dropped queries the other scored")))

    membership = detail.get("eligible_query_set_identical")
    if membership is None:
        # Absent, same rule: an unrecorded membership assertion measures nothing.
        checks.append(ParityCheck(
            name=f"{category}.metrics.eligible_query_set_identical", dimension="metrics_parity",
            gate="G2", declared="the two sides ranked the same eligible query set",
            measured={"eligible_query_set_identical": None}, ok=False, mandatory=False, absence=True,
            note=("ABSENT MEMBERSHIP: the block does not record `eligible_query_set_identical`, so "
                  "whether both sides ranked the same eligible query set was never measured")))
    else:
        checks.append(ParityCheck(
            name=f"{category}.metrics.eligible_query_set_identical", dimension="metrics_parity",
            gate="G2", declared="the two sides ranked the same eligible query set",
            measured={"eligible_query_set_identical": membership}, ok=membership is True,
            mandatory=True,
            note="" if membership is True else
                 "MEASURED DISAGREEMENT: both sides were recorded, and the block does not assert they "
                 "ranked the same eligible query set"))

    absent_fields = []
    values: dict[str, Any] = {}
    for field, label in METRICS_PROVENANCE:
        left = _identity_value(native, field)
        right = _identity_value(adapter, field)
        if left is _ABSENT and right is _ABSENT:
            absent_fields.append(f"`{field}` ({label})")
            continue
        values[field] = {"native": None if left is _ABSENT else left,
                         "adapter": None if right is _ABSENT else right}
        if left is _ABSENT or right is _ABSENT:
            absent_fields.append(f"`{field}` ({label}) on one side only")
    # `protocol_id` and `protocol` name the same thing in the two producers; either satisfies it.
    protocol_present = (not any(item.startswith("`protocol_id`") for item in absent_fields)
                        or not any(item.startswith("`protocol`") for item in absent_fields))
    if protocol_present:
        absent_fields = [item for item in absent_fields
                         if not (item.startswith("`protocol_id`") or item.startswith("`protocol`"))]
    provenance_ok = not absent_fields
    checks.append(ParityCheck(
        name=f"{category}.metrics.provenance", dimension="metrics_parity", gate="G2",
        declared=("a protocol id, a checkpoint identity, and both order digests on BOTH sides, because "
                  "a hit-count comparison across different queries, galleries or checkpoints "
                  "measures nothing"),
        measured={"fields": values, "absent": absent_fields},
        ok=provenance_ok, mandatory=True, absence=not provenance_ok,
        note="" if provenance_ok else
             ("the artifacts record a hit-count comparison without the identifiers that make it one: "
              + ", ".join(absent_fields)
              + ". Two counts are only comparable when both sides ranked the same queries over the "
                "same gallery under the same protocol and checkpoint")))
    return checks


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
    (`abs(a-b) <= atol + rtol*abs(b)` per element), in `component_parity`. `max_abs`/`max_rel` are
    v1 summary statistics: a v2 label over them alone is malformed, not a v2 result. The block is
    written by the phase that holds BOTH arrays (the adapter), so it is required on at least one
    side; a block on both sides is accepted only when every shared scope is identical.

    Two artifacts declaring different standards is malformed too -- the same rule as tolerances.
    An artifact with NO `standard` key is a v1 result, so ONE side v1-by-default and the other v2 is
    exactly that mixed case: the two sides were measured under different rules and one verdict cannot
    be reported under both. (Both sides absent is not a disagreement -- it is the v1 record.)
    """
    left, right = native.get("standard"), adapter.get("standard")
    for side, value in (("native", left), ("adapter", right)):
        if value is not None and value not in STANDARDS:
            raise _Malformed(f"the {side} artifact declares an unknown standard {value!r}")
    if (left or STANDARD_RELATIVE_V1) != (right or STANDARD_RELATIVE_V1):
        raise _Malformed(f"the two artifacts declare different standards: native {left!r}, "
                         f"adapter {right!r} -- an unstated standard is "
                         f"{STANDARD_RELATIVE_V1!r}, and one verdict cannot be reported under two")
    declared = left or right or STANDARD_RELATIVE_V1
    if declared == STANDARD_ABS_REL_V2 and not _component_scopes(native, adapter):
        raise _Malformed(_V2_WITHOUT_COMPONENTS)
    if declared == STANDARD_ABS_REL_V3:
        _v3_manifest(native, adapter)
    return declared

#: One refusal, used by both places that can spot a bare v2 label, so the reason never drifts.
_V2_WITHOUT_COMPONENTS = (
    "the artifact declares parity-absolute-relative/v2 but records only `max_abs`/`max_rel` "
    "aggregates; that standard is defined per component (abs(a-b) <= atol + rtol*abs(b)) and cannot "
    f"be evaluated from the recorded statistics, so it must never be applied. A v2 artifact carries "
    f"`{COMPONENT_KEY}` (see workbench/docs/reviews/20261112-parity-repair/V2_STANDARD.md)")

V3_CONTROLS = ("native_repeat", "adapter_repeat", "native_adapter",
               "native_batch_1_vs_n", "adapter_batch_1_vs_n")
V3_KEY = "parity_v3"
V3_RULE = "abs(actual - reference) <= atol + rtol * abs(reference)"


def _v3_manifest(native: dict[str, Any], adapter: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    """Validate immutable preregistration plus post-run receipts embedded in final evidence."""
    blocks = [(side, item.get(V3_KEY)) for side, item in (("native", native), ("adapter", adapter))
              if item.get(V3_KEY) is not None]
    if len(blocks) != 1:
        raise _Malformed(f"v3 requires exactly one `{V3_KEY}` block, found {len(blocks)}")
    side, block = blocks[0]
    if not isinstance(block, dict) or block.get("standard") != STANDARD_ABS_REL_V3:
        raise _Malformed(f"the {side} artifact `{V3_KEY}` block must declare {STANDARD_ABS_REL_V3}")
    plan_rel, digest = block.get("plan_path"), block.get("plan_sha256")
    if not isinstance(plan_rel, str) or not plan_rel or Path(plan_rel).is_absolute():
        raise _Malformed("v3 plan_path must be a non-empty relative path")
    evidence = Path((native if side == "native" else adapter).get("_artifact_path", "."))
    plan_path = evidence.parent / plan_rel
    expected_plan_name = f"{evidence.name.rsplit('.', 2)[0]}.v3-plan.json"
    if Path(plan_rel).name != expected_plan_name or Path(plan_rel).parent != Path("."):
        raise _Malformed(f"v3 plan_path must be deterministic sibling {expected_plan_name!r}")
    if not plan_path.resolve().is_relative_to(evidence.parent.resolve()):
        raise _Malformed("v3 plan_path escapes the evidence directory")
    try:
        raw = plan_path.read_bytes()
    except OSError as error:
        raise _Malformed(f"v3 preregistered plan is unavailable: {plan_path}: {error}") from error
    if not isinstance(digest, str) or hashlib.sha256(raw).hexdigest() != digest:
        raise _Malformed("v3 plan_sha256 does not match persisted plan bytes")
    if plan_path.stat().st_mtime_ns >= evidence.stat().st_mtime_ns:
        raise _Malformed("v3 plan was not persisted before final run evidence")
    try:
        plan = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _Malformed(f"v3 plan is not valid JSON: {error}") from error
    artifact = native if side == "native" else adapter
    if not isinstance(plan, dict) or plan.get("standard") != STANDARD_ABS_REL_V3:
        raise _Malformed("v3 plan must declare the exact v3 standard")
    if plan.get("model_id") != artifact.get("model_id") or plan.get("category") != artifact.get("category"):
        raise _Malformed("v3 plan model/category do not match artifact identity")
    identity_fields = ("checkpoint", "configuration", "environment", "protocol", "category",
                       "cohort_id", "precision", "seed", "query_order_sha256",
                       "gallery_order_sha256", "reference_exclusions_sha256", "query_count",
                       "gallery_count")
    identity = plan.get("identity")
    if not isinstance(identity, dict) or set(identity) != set(identity_fields):
        raise _Malformed("v3 plan.identity must contain the exact preregistered identity fields")
    for field in identity_fields:
        value = identity[field]
        if field in ("query_count", "gallery_count"):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise _Malformed(f"v3 identity.{field} must be a positive integer")
        elif field == "seed":
            if isinstance(value, bool) or not isinstance(value, (int, str)):
                raise _Malformed("v3 identity.seed must be an integer or explicit seed string")
        elif not isinstance(value, str) or not value:
            raise _Malformed(f"v3 identity.{field} must be a non-empty string")
    if len(identity["checkpoint"]) != 64 or any(
            char not in _COMPONENT_HEX for char in identity["checkpoint"]):
        raise _Malformed("v3 identity.checkpoint must be a SHA256 digest of verified checkpoint bytes")
    for field in ("query_order_sha256", "gallery_order_sha256", "reference_exclusions_sha256"):
        value = identity[field]
        if len(value) != 64 or any(char not in _COMPONENT_HEX for char in value):
            raise _Malformed(f"v3 identity.{field} must be a SHA256 hex digest")
    if identity["category"] != artifact.get("category"):
        raise _Malformed("v3 plan identity category differs from artifact")
    for side_artifact in (native, adapter):
        if side_artifact.get("identity") != identity:
            raise _Malformed("v3 artifact identity differs from immutable preregistered identity")
    for field in ("supported_batch_policy", "population_scope"):
        value = plan.get(field)
        if value is None or value == "" or value == [] or value == {}:
            raise _Malformed(f"v3 preregistered plan is missing {field}")
    batch_policy = plan["supported_batch_policy"]
    if (not isinstance(batch_policy, dict) or batch_policy.get("supported") is not True
            or not isinstance(batch_policy.get("batch_sizes"), list)
            or len(batch_policy["batch_sizes"]) < 2
            or any(isinstance(size, bool) or not isinstance(size, int) or size <= 0
                   for size in batch_policy["batch_sizes"])):
        raise _Malformed("v3 supported_batch_policy must predeclare supported batch sizes")
    if "batch_invariant" in batch_policy and not isinstance(batch_policy["batch_invariant"], bool):
        raise _Malformed("v3 supported_batch_policy.batch_invariant must be boolean")
    population = plan["population_scope"]
    if (not isinstance(population, dict) or population.get("coverage") not in ("full", "sample")
            or any(isinstance(population.get(key), bool) or not isinstance(population.get(key), int)
                   or population[key] <= 0 for key in ("query_count", "gallery_count"))):
        raise _Malformed("v3 population_scope must declare full/sample coverage and positive query/gallery counts")
    if any(population[key] != identity[key] for key in ("query_count", "gallery_count")):
        raise _Malformed("v3 population denominators differ from actual artifact identity")
    identity_aliases = {"protocol": "protocol",
                        "query_order_sha256": "query_order_sha256",
                        "gallery_order_sha256": "gallery_order_sha256"}
    for side_artifact in (native, adapter):
        for identity_key, artifact_key in identity_aliases.items():
            if side_artifact.get(artifact_key) != identity[identity_key]:
                raise _Malformed(f"v3 artifact {artifact_key} does not bind identity.{identity_key}")
        if side_artifact.get("checkpoint_sha256") != identity["checkpoint"]:
            raise _Malformed("v3 artifact checkpoint_sha256 does not bind verified checkpoint bytes")
    native_body = native.get("native")
    layer1 = native_body.get("layer1_identity") if isinstance(native_body, dict) else None
    metrics = native.get("native_metrics_parity_detail")
    if not isinstance(layer1, dict) or not isinstance(metrics, dict):
        raise _Malformed("v3 native artifact lacks measured layer1 identity or metrics denominator")
    if (layer1.get("query_count_native") != identity["query_count"]
            or layer1.get("query_count_annotations") != identity["query_count"]
            or layer1.get("gallery_count_native") != identity["gallery_count"]
            or layer1.get("gallery_count_rebuilt_from_annotations") != identity["gallery_count"]):
        raise _Malformed("v3 identity population differs from actual native layer1 inputs")
    for side in ("native", "adapter"):
        side_metrics = metrics.get(side)
        if not isinstance(side_metrics, dict) or side_metrics.get("query_count") != identity["query_count"]:
            raise _Malformed(f"v3 identity query_count differs from actual {side} metric denominator")
    stable_identity = set(identity_fields) - {"cohort_id", "query_order_sha256", "gallery_order_sha256",
                                              "reference_exclusions_sha256", "query_count", "gallery_count"}
    runs = plan.get("runs")
    if not isinstance(runs, dict):
        raise _Malformed("v3 plan.runs must map predeclared run roles to run records")
    required_roles = ("native_repeat_1", "native_repeat_2", "adapter_repeat_1",
                      "adapter_repeat_2", "native_adapter_native", "native_adapter_adapter",
                      "native_batch_1", "native_batch_n", "adapter_batch_1", "adapter_batch_n")
    ids = []
    for role in required_roles:
        run = runs.get(role)
        if not isinstance(run, dict) or not isinstance(run.get("run_id"), str) or not run["run_id"]:
            raise _Malformed(f"v3 plan is missing predeclared run ID for {role}")
        execution = run.get("execution")
        if not isinstance(execution, dict):
            raise _Malformed(f"v3 plan is missing predeclared execution partition for {role}")
        for key in ("partition_id", "query_partition_sha256", "gallery_partition_sha256"):
            if not isinstance(execution.get(key), str) or not execution[key]:
                raise _Malformed(f"v3 run {role} execution lacks {key}")
        for key in ("query_batch_size", "gallery_batch_size"):
            value = execution.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise _Malformed(f"v3 run {role} execution has invalid {key}")
        source_id, run_identity = run.get("source_id"), run.get("identity")
        if not isinstance(source_id, str) or not source_id or not isinstance(run_identity, dict):
            raise _Malformed(f"v3 run {role} must bind actual source_id and identity")
        if set(run_identity) != set(identity_fields):
            raise _Malformed(f"v3 run {role} identity has incorrect fields")
        if any(run_identity[key] != identity[key] for key in stable_identity):
            raise _Malformed(f"v3 run {role} changed model, protocol, or evaluation policy identity")
        for count_key in ("query_count", "gallery_count"):
            value = run_identity[count_key]
            if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= identity[count_key]:
                raise _Malformed(f"v3 run {role} has invalid subset {count_key}")
        if not isinstance(run_identity["cohort_id"], str) or not run_identity["cohort_id"]:
            raise _Malformed(f"v3 run {role} has empty cohort_id identity")
        for digest_key in ("query_order_sha256", "gallery_order_sha256", "reference_exclusions_sha256"):
            digest_value = run_identity[digest_key]
            if (not isinstance(digest_value, str) or len(digest_value) != 64
                    or any(char not in _COMPONENT_HEX for char in digest_value)):
                raise _Malformed(f"v3 run {role} identity {digest_key} must be a SHA256 hex digest")
        for key in ("query_partition_sha256", "gallery_partition_sha256"):
            digest_value = execution[key]
            if len(digest_value) != 64 or any(char not in _COMPONENT_HEX for char in digest_value):
                raise _Malformed(f"v3 run {role} execution {key} must be a SHA256 hex digest")
        ids.append(run["run_id"])
    if len(set(ids)) != len(ids):
        raise _Malformed("v3 plan run IDs must identify distinct executions")
    matched_roles = (("native_repeat_1", "native_repeat_2"),
                    ("adapter_repeat_1", "adapter_repeat_2"),
                    ("native_adapter_native", "native_adapter_adapter"),
                    ("native_batch_1", "native_batch_n"),
                    ("adapter_batch_1", "adapter_batch_n"))
    for reference_role, actual_role in matched_roles:
        if (runs[reference_role]["identity"]["reference_exclusions_sha256"]
                != runs[actual_role]["identity"]["reference_exclusions_sha256"]):
            raise _Malformed(f"v3 matched run roles {reference_role}/{actual_role} have different actual reference-exclusion masks")
    batch_sizes = batch_policy["batch_sizes"]
    if len(set(batch_sizes)) != len(batch_sizes) or 1 not in batch_sizes:
        raise _Malformed("v3 batch policy must declare distinct supported sizes including B1")
    non_singleton_sizes = {size for size in batch_sizes if size > 1}
    if not non_singleton_sizes:
        raise _Malformed("v3 batch policy must declare a supported BN size greater than one")
    for role in required_roles:
        execution = runs[role]["execution"]
        expected_size = 1 if role.endswith("batch_1") else None
        if role.endswith("batch_n") and execution["query_batch_size"] not in non_singleton_sizes:
            raise _Malformed(f"v3 run {role} uses an unsupported BN query batch size")
        if expected_size is not None and (execution["query_batch_size"] != expected_size
                                           or execution["gallery_batch_size"] != expected_size):
            raise _Malformed(f"v3 run {role} must execute with query/gallery batch size 1")
        if role.endswith("batch_n") and execution["gallery_batch_size"] not in non_singleton_sizes:
            raise _Malformed(f"v3 run {role} uses an unsupported BN gallery batch size")
    aligned_roles = (("native_repeat_1", "native_repeat_2"),
                     ("adapter_repeat_1", "adapter_repeat_2"),
                     ("native_adapter_native", "native_adapter_adapter"),
                     ("native_batch_1", "native_batch_n"),
                     ("adapter_batch_1", "adapter_batch_n"),
                     ("native_batch_1", "adapter_batch_1"),
                     ("native_batch_n", "adapter_batch_n"))
    for left_role, right_role in aligned_roles:
        left, right = runs[left_role]["execution"], runs[right_role]["execution"]
        if any(left.get(key) != right.get(key) for key in
               ("partition_id", "query_partition_sha256", "gallery_partition_sha256")):
            raise _Malformed(f"v3 matched runs {left_role}/{right_role} do not share ordered query/gallery partition")
        if left_role == "native_adapter_native" and any(
                left.get(key) != right.get(key) for key in ("query_batch_size", "gallery_batch_size")):
            raise _Malformed("v3 native/adapter cross-path runs must use matched query/gallery batch sizes")
        if left_role in {"native_repeat_1", "adapter_repeat_1"} and any(
                left.get(key) != right.get(key) for key in ("query_batch_size", "gallery_batch_size")):
            raise _Malformed(f"v3 repeated runs {left_role}/{right_role} must use matched query/gallery batch sizes")
    controls = plan.get("controls")
    receipts = block.get("receipts")
    scopes_by_control = block.get("scopes")
    if not isinstance(controls, dict) or not isinstance(receipts, dict) or not isinstance(scopes_by_control, dict):
        raise _Malformed("v3 plan.controls and final artifact receipts/scopes must be objects")
    expected = {"native_repeat": ("native_repeat_1", "native_repeat_2", "native"),
                "adapter_repeat": ("adapter_repeat_1", "adapter_repeat_2", "adapter"),
                "native_adapter": ("native_adapter_native", "native_adapter_adapter", "native"),
                "native_batch_1_vs_n": ("native_batch_1", "native_batch_n", "native"),
                "adapter_batch_1_vs_n": ("adapter_batch_1", "adapter_batch_n", "adapter")}
    for name, (reference_role, actual_role, reference_side) in expected.items():
        control, emitted_scopes = controls.get(name), scopes_by_control.get(name)
        if not isinstance(control, dict) or tuple(control.get(key) for key in ("reference_run_id", "actual_run_id")) != (runs[reference_role]["run_id"], runs[actual_role]["run_id"]):
            raise _Malformed(f"v3 plan control {name!r} must reference its predeclared runs")
        scopes = control.get("scopes")
        if not isinstance(emitted_scopes, dict) or set(emitted_scopes) != {"query", "gallery"}:
            raise _Malformed(f"v3 artifact scope {name!r} must provide query and gallery evidence")
        if control.get("reference_side") != reference_side:
            raise _Malformed(f"v3 plan control {name!r} must use {reference_side} as reference")
        dimension = "batch_parity" if "batch" in name else "forward_parity"
        if control.get("dimension") != dimension:
            raise _Malformed(f"v3 plan control {name!r} has incorrect dimension")
        if not isinstance(control.get("dtype"), str) or not control["dtype"]:
            raise _Malformed(f"v3 plan control {name!r} must declare captured dtype")
        tolerance = control.get("tolerance")
        if not isinstance(tolerance, dict) or any(not isinstance(tolerance.get(k), (int, float)) or isinstance(tolerance.get(k), bool) or tolerance[k] < 0 for k in ("atol", "rtol")):
            raise _Malformed(f"v3 plan control {name!r} needs non-negative atol and rtol")
        if not isinstance(scopes, dict) or set(scopes) != {"query", "gallery"}:
            raise _Malformed(f"v3 control {name!r} must capture separate query and gallery tensors")
        for scope_name, scope in scopes.items():
            if not isinstance(scope, dict) or scope.get("coverage") not in ("full", "sample"):
                raise _Malformed(f"v3 {name}.{scope_name} must declare full/sample population scope")
            count_key = "query_count" if scope_name == "query" else "gallery_count"
            if scope.get("population_count") != population[count_key]:
                raise _Malformed(f"v3 {name}.{scope_name} population denominator differs from preregistration")
            sample_count = scope.get("sample_count")
            if (scope.get("coverage") == "sample"
                    and (isinstance(sample_count, bool) or not isinstance(sample_count, int)
                         or not 0 < sample_count < population[count_key])):
                raise _Malformed(f"v3 {name}.{scope_name} sample_count must be less than declared population")
            tensor_shape, component_shape = scope.get("tensor_shape"), scope.get("component_shape")
            expected_count = sample_count if scope.get("coverage") == "sample" else population[count_key]
            ordered_ids = scope.get("ordered_ids_sha256")
            if (not isinstance(ordered_ids, str) or len(ordered_ids) != 64
                    or any(char not in _COMPONENT_HEX for char in ordered_ids)):
                raise _Malformed(f"v3 {name}.{scope_name} needs an ordered ID digest")
            if (not isinstance(tensor_shape, list) or len(tensor_shape) < 2
                    or any(isinstance(part, bool) or not isinstance(part, int) or part <= 0 for part in tensor_shape)
                    or tensor_shape[0] != expected_count):
                raise _Malformed(f"v3 {name}.{scope_name} needs tensor_shape bound to declared coverage")
            if (not isinstance(component_shape, list) or not component_shape
                    or any(isinstance(part, bool) or not isinstance(part, int) or part <= 0 for part in component_shape)
                    or __import__("math").prod(tensor_shape) != __import__("math").prod(component_shape)):
                raise _Malformed(f"v3 {name}.{scope_name} component shape does not match tensor shape")
            for role in (reference_role, actual_role):
                run_identity = runs[role]["identity"]
                order_key = "query_order_sha256" if scope_name == "query" else "gallery_order_sha256"
                if (run_identity[count_key] != expected_count
                        or run_identity[order_key] != ordered_ids
                        or run_identity["cohort_id"] != scope.get("cohort_id")):
                    raise _Malformed(f"v3 {name}.{scope_name} scope differs from actual run identity {role}")
            if name == "native_adapter":
                for role in (reference_role, actual_role):
                    if runs[role]["identity"] != identity:
                        raise _Malformed("v3 native_adapter control must match complete primary population identity")
    role_sides = {role: ("adapter" if role.startswith("adapter_") or role == "native_adapter_adapter" else "native")
                  for role in required_roles}
    from datetime import datetime, timezone
    artifact_mtime = datetime.fromtimestamp(evidence.stat().st_mtime, timezone.utc)
    plan_mtime = datetime.fromtimestamp(plan_path.stat().st_mtime, timezone.utc)
    for role in required_roles:
        receipt = receipts.get(role)
        if not isinstance(receipt, dict) or receipt.get("run_id") != runs[role]["run_id"]:
            raise _Malformed(f"v3 final artifact lacks receipt for actual run {role}")
        if receipt.get("plan_sha256") != digest:
            raise _Malformed(f"v3 run receipt {role} does not bind the immutable preregistration plan")
        if receipt.get("source_id") != runs[role]["source_id"]:
            raise _Malformed(f"v3 run receipt {role} source ID differs from preregistered run")
        if receipt.get("identity") != runs[role]["identity"]:
            raise _Malformed(f"v3 run receipt {role} identity differs from preregistered actual run")
        if receipt.get("side") != role_sides[role]:
            raise _Malformed(f"v3 run receipt {role} has incorrect native/adapter side")
        try:
            started = datetime.fromisoformat(receipt["started_at"].replace("Z", "+00:00"))
            captured = datetime.fromisoformat(receipt["captured_at"].replace("Z", "+00:00"))
            if started.tzinfo is None or captured.tzinfo is None:
                raise ValueError("timestamps must include a timezone")
            if not plan_mtime < started <= captured <= artifact_mtime:
                raise ValueError("expected plan persisted < actual start <= capture <= artifact publication")
        except (TypeError, ValueError, OverflowError, AttributeError, KeyError) as error:
            raise _Malformed(f"v3 run receipt {role} has invalid actual capture chronology: {error}") from error
        if receipt.get("execution") != runs[role]["execution"]:
            raise _Malformed(f"v3 run receipt {role} execution differs from preregistered partition")
    for name, (reference_role, actual_role, _) in expected.items():
        emitted_control_scopes = scopes_by_control[name]
        for tensor in ("query", "gallery"):
            emitted_scope = emitted_control_scopes[tensor]
            expected_ids = {reference_role, actual_role}
            for role in expected_ids:
                captured = receipts[role].get("arrays", {}).get(f"{name}.{tensor}")
                if not isinstance(captured, dict):
                    raise _Malformed(f"v3 run receipt {role} lacks captured {name}.{tensor} array metadata")
                expected_digest = emitted_scope["sha256"]["reference" if role == reference_role else "actual"]
                component_digest = captured.get("component_sha256")
                if (captured.get("dtype") != controls[name]["dtype"]
                        or captured.get("shape") != controls[name]["scopes"][tensor].get("tensor_shape")
                        or captured.get("ordered_ids_sha256") != controls[name]["scopes"][tensor].get("ordered_ids_sha256")
                        or component_digest != expected_digest
                        or not isinstance(component_digest, str) or len(component_digest) != 64
                        or any(char not in _COMPONENT_HEX for char in component_digest)):
                    raise _Malformed(f"v3 run receipt {role} does not bind actual {name}.{tensor} tensor")
    return plan, plan_path


def _inverted_pairs_within_epsilon(native_scores: list[Any], adapter_scores: list[Any],
                                   native_order: list[int], adapter_order: list[int],
                                   epsilon: float) -> bool:
    """Check every inverted pair in O(G log G) time and O(G) memory."""
    size = len(native_order)
    base = 1 << (size - 1).bit_length()
    tree = [[math.inf, -math.inf, math.inf, -math.inf] for _ in range(2 * base)]
    adapter_position = [0] * size
    for position, candidate in enumerate(adapter_order):
        adapter_position[candidate] = position

    def update(position: int, native_score: float, adapter_score: float) -> None:
        node = base + position
        tree[node] = [native_score, native_score, adapter_score, adapter_score]
        node //= 2
        while node:
            left, right = tree[2 * node], tree[2 * node + 1]
            tree[node] = [min(left[0], right[0]), max(left[1], right[1]),
                          min(left[2], right[2]), max(left[3], right[3])]
            node //= 2

    def query(start: int, end: int) -> list[float]:
        result = [math.inf, -math.inf, math.inf, -math.inf]
        left, right = base + start, base + end
        while left < right:
            if left & 1:
                node = tree[left]
                result = [min(result[0], node[0]), max(result[1], node[1]),
                          min(result[2], node[2]), max(result[3], node[3])]
                left += 1
            if right & 1:
                right -= 1
                node = tree[right]
                result = [min(result[0], node[0]), max(result[1], node[1]),
                          min(result[2], node[2]), max(result[3], node[3])]
            left //= 2
            right //= 2
        return result

    for candidate in native_order:
        position = adapter_position[candidate]
        if position + 1 < size:
            lo_native, hi_native, lo_adapter, hi_adapter = query(position + 1, size)
            if lo_native != math.inf and (
                    native_scores[candidate] - lo_native > epsilon
                    or hi_native - native_scores[candidate] > epsilon
                    or adapter_scores[candidate] - lo_adapter > epsilon
                    or hi_adapter - adapter_scores[candidate] > epsilon):
                return False
        update(position, float(native_scores[candidate]), float(adapter_scores[candidate]))
    return True


def _v3_score_rank_checks(native: dict[str, Any], adapter: dict[str, Any],
                          declared: dict[str, Any], plan: dict[str, Any],
                          category: str) -> list[ParityCheck]:
    """Derive V3 score/rank and integer outcomes from bounded per-query proof rows."""
    artifact = native if V3_KEY in native else adapter
    block = artifact[V3_KEY]
    evidence = Path(artifact.get("_artifact_path", "."))
    expected_name = f"{evidence.name.rsplit('.', 1)[0]}.score-ranks.jsonl"
    if plan.get("scoring_proof_path") != expected_name or block.get("scoring_proof_path") != expected_name:
        raise _Malformed("v3 scoring_proof_path must be preregistered deterministic sibling "
                         f"{expected_name!r}")
    proof_path = evidence.parent / expected_name
    if not proof_path.resolve().is_relative_to(evidence.parent.resolve()):
        raise _Malformed("v3 scoring proof escapes the evidence directory")
    try:
        with proof_path.open("rb") as stream:
            proof_hash = hashlib.sha256()
            while chunk := stream.read(1024 * 1024):
                proof_hash.update(chunk)
            if block.get("scoring_proof_sha256") != proof_hash.hexdigest():
                raise _Malformed("v3 scoring_proof_sha256 does not match persisted raw bytes")
            stream.seek(0)
            header_line = stream.readline()
            header = json.loads(header_line)
            if (not isinstance(header, dict) or header.get("schema") != "cir.score-ranks/v1"
                    or header.get("identity") != plan["identity"]):
                raise _Malformed("v3 scoring proof schema/identity differs from preregistration")
            query_ids, gallery_ids = header.get("query_ids"), header.get("gallery_ids")
            if (not isinstance(query_ids, list) or len(query_ids) != plan["identity"]["query_count"]
                    or not all(isinstance(value, str) and value for value in query_ids)
                    or len(set(query_ids)) != len(query_ids)
                    or not isinstance(gallery_ids, list)
                    or len(gallery_ids) != plan["identity"]["gallery_count"]
                    or not all(isinstance(value, str) and value for value in gallery_ids)
                    or len(set(gallery_ids)) != len(gallery_ids)):
                raise _Malformed("v3 scoring proof ordered query/gallery IDs are malformed")
            id_bytes = lambda values: json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            if (hashlib.sha256(id_bytes(query_ids)).hexdigest() != plan["identity"]["query_order_sha256"]
                    or hashlib.sha256(id_bytes(gallery_ids)).hexdigest() != plan["identity"]["gallery_order_sha256"]):
                raise _Malformed("v3 scoring proof query/gallery order digests differ from preregistration")
            run_receipts = block.get("scoring_run_receipts")
            if not isinstance(run_receipts, dict):
                raise _Malformed("v3 scoring proof lacks actual native/adapter run receipts")
            expected_roles = {"native_adapter_native": "native", "native_adapter_adapter": "adapter"}
            for role, side in expected_roles.items():
                run, receipt = plan["runs"][role], run_receipts.get(role)
                if (not isinstance(receipt, dict) or receipt.get("run_id") != run["run_id"]
                        or receipt.get("source_id") != run["source_id"] or receipt.get("side") != side
                        or receipt.get("identity") != run["identity"]
                        or not isinstance(receipt.get("score_order_sha256"), str)):
                    raise _Malformed(f"v3 scoring proof does not bind actual {role} capture")
            labels = plan.get("scoring_labels")
            if (not isinstance(labels, dict) or set(labels) != {"query_targets", "reference_exclusions"}
                    or labels.get("query_targets") != header.get("query_targets")
                    or labels.get("reference_exclusions") != header.get("reference_exclusions")):
                raise _Malformed("v3 score proof labels/exclusions differ from preregistered metadata")
            label_bytes = json.dumps(labels, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            if hashlib.sha256(label_bytes).hexdigest() != plan["identity"]["reference_exclusions_sha256"]:
                raise _Malformed("v3 preregistered labels do not bind reference exclusion digest")
            targets, exclusions = labels["query_targets"], labels["reference_exclusions"]
            query_count, gallery_count = plan["identity"]["query_count"], plan["identity"]["gallery_count"]
            if (not isinstance(targets, list) or len(targets) != query_count
                    or not isinstance(exclusions, list) or len(exclusions) != query_count):
                raise _Malformed("v3 score proof target/exclusion labels must cover every query")
            excluded_by_query = {}
            for index, (target_entry, exclusion_entry) in enumerate(zip(targets, exclusions)):
                if (not isinstance(target_entry, list) or len(target_entry) != 2
                        or target_entry[0] != query_ids[index] or target_entry[1] not in gallery_ids
                        or not isinstance(exclusion_entry, list) or len(exclusion_entry) != 2
                        or exclusion_entry[0] != query_ids[index]
                        or not isinstance(exclusion_entry[1], list)
                        or any(value not in gallery_ids for value in exclusion_entry[1])
                        or len(set(exclusion_entry[1])) != len(exclusion_entry[1])
                        or target_entry[1] in exclusion_entry[1]):
                    raise _Malformed(f"v3 score proof has invalid target/exclusion labels for query {query_ids[index]}")
                excluded_by_query[query_ids[index]] = set(exclusion_entry[1])
            score_max = 0.0
            score_hashes = {role: hashlib.sha256() for role in expected_roles}
            query_count, gallery_count = len(query_ids), len(gallery_ids)
            rows_seen = 0
            target_rank_differences = sorted_rows_differ = inversion_gap_violations = 0
            topk_membership_differences = topk_order_differences = 0
            metric_record, metric_where = _metrics_detail(native, adapter)
            if metric_record is None:
                raise _Malformed(f"{category}: V3 raw scorer proof requires per-side integer hit counts")
            cutoffs = metric_record.get("cutoffs")
            if (not isinstance(cutoffs, list) or not cutoffs
                    or any(isinstance(k, bool) or not isinstance(k, int) or k <= 0 for k in cutoffs)
                    or cutoffs != sorted(set(cutoffs))):
                raise _Malformed(f"v3 {metric_where}.cutoffs must be sorted unique positive integers")
            declared_cutoffs = plan.get("metric_cutoffs")
            if declared_cutoffs is not None and declared_cutoffs != cutoffs:
                raise _Malformed("v3 plan metric_cutoffs differ from bound metric detail cutoffs")
            native_hits = {k: 0 for k in cutoffs}
            adapter_hits = {k: 0 for k in cutoffs}
            failure_set_differences = {k: 0 for k in cutoffs}
            failure_overlap_equal = {k: True for k in cutoffs}
            flipped_queries = 0
            epsilon = _tolerance_value(declared.get("tie_epsilon"), where=f"{category}.tolerances.tie_epsilon")
            for encoded in stream:
                query_index = rows_seen
                if query_index >= query_count:
                    raise _Malformed("v3 scoring proof has extra query rows")
                row = json.loads(encoded)
                if not isinstance(row, dict) or row.get("query_id") != query_ids[query_index]:
                    raise _Malformed(f"v3 scoring proof row {query_index} is not bound to its ordered query ID")
                target_id = row.get("target_id")
                if target_id != labels["query_targets"][query_index][1]:
                    raise _Malformed(f"v3 score row {query_index} target differs from preregistered label")
                try:
                    target = gallery_ids.index(target_id)
                except (ValueError, TypeError):
                    raise _Malformed(f"v3 scoring proof row {query_index} target ID is outside gallery") from None
                if (row.get("target_position") != target
                        or row.get("excluded_gallery_ids") != sorted(excluded_by_query[query_ids[query_index]])):
                    raise _Malformed(f"v3 score row {query_index} target/exclusions differ from preregistration")
                vectors, orders = [], []
                for side in ("native", "adapter"):
                    scores, order = row.get(f"{side}_scores"), row.get(f"{side}_order")
                    if (not isinstance(scores, list) or len(scores) != gallery_count
                            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                                   for v in scores)):
                        raise _Malformed(f"v3 scoring proof row {query_index} {side} scores are malformed")
                    if (not isinstance(order, list) or len(order) != gallery_count
                            or any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < gallery_count
                                   for v in order) or len(set(order)) != gallery_count):
                        raise _Malformed(f"v3 scoring proof row {query_index} {side} order is not a gallery permutation")
                    convention = plan.get("score_order_conventions", {}).get(side, "fp32_descending_score")
                    if not _score_order_matches(scores, order, convention):
                        raise _Malformed(f"v3 scoring proof row {query_index} {side} order contradicts {convention}")
                    score_payload = json.dumps({"query_id": query_ids[query_index],
                                                "scores": scores, "order": order},
                                               ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                    score_hashes[f"native_adapter_{side}"].update(score_payload + b"\n")
                    vectors.append(scores)
                    orders.append(order)
                native_scores, adapter_scores = vectors
                native_order, adapter_order = orders
                score_max = max(score_max, max((abs(float(a) - float(b))
                                                for a, b in zip(native_scores, adapter_scores)), default=0.0))
                excluded_positions = {gallery_index for gallery_index, gallery_id in enumerate(gallery_ids)
                                      if gallery_id in excluded_by_query[query_ids[query_index]]}
                native_eligible = [candidate for candidate in native_order if candidate not in excluded_positions]
                adapter_eligible = [candidate for candidate in adapter_order if candidate not in excluded_positions]
                native_rank, adapter_rank = native_eligible.index(target) + 1, adapter_eligible.index(target) + 1
                rank_changed = native_rank != adapter_rank
                target_rank_differences += rank_changed
                if not _inverted_pairs_within_epsilon(
                        native_scores, adapter_scores, native_order, adapter_order, epsilon):
                    inversion_gap_violations += 1
                sorted_rows_differ += native_order != adapter_order
                for k in native_hits:
                    native_top, adapter_top = native_eligible[:k], adapter_eligible[:k]
                    native_hits[k] += target in native_top
                    adapter_hits[k] += target in adapter_top
                    topk_membership_differences += set(native_top) != set(adapter_top)
                    topk_order_differences += native_top != adapter_top
                    failed_native, failed_adapter = target not in native_top, target not in adapter_top
                    failure_set_differences[k] += failed_native != failed_adapter
                    failure_overlap_equal[k] &= failed_native == failed_adapter
                flipped_queries += any((target in native_eligible[:k]) != (target in adapter_eligible[:k])
                                       for k in native_hits)
                rows_seen += 1
            if rows_seen != query_count:
                raise _Malformed("v3 scoring proof does not contain every preregistered query row")
    except _Malformed:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _Malformed(f"v3 scoring proof is unreadable or invalid: {error}") from error
    for role in expected_roles:
        if run_receipts[role].get("score_order_sha256") != score_hashes[role].hexdigest():
            raise _Malformed(f"v3 scoring proof {role} score/order digest differs from its actual rows")
    metrics, metrics_where = _metrics_detail(native, adapter)
    if metrics is None:
        raise _Malformed(f"{category}: V3 raw scorer proof requires per-side integer hit counts")
    for side, derived in (("native", native_hits), ("adapter", adapter_hits)):
        recorded_hits = _hit_counts(metrics, side, where=metrics_where)
        if any(recorded_hits.get(f"r{k}") != value for k, value in derived.items()):
            raise _Malformed(f"v3 {side} integer metric summary contradicts raw score/rank proof")
    score_tol = _tolerance_value(declared.get("score_max_abs"), where=f"{category}.tolerances.score_max_abs")
    diagnostic = {"queries": len(query_ids), "score_max_abs": score_max,
                  "target_rank_differences": target_rank_differences,
                  "all_inverted_pairs_within_epsilon_queries": query_count - inversion_gap_violations,
                  "inverted_pair_gap_violation_queries": inversion_gap_violations,
                  "sorted_rows_differ": sorted_rows_differ,
                  "topk_membership_differences": topk_membership_differences,
                  "topk_order_differences": topk_order_differences,
                  "per_query_target_hit_flips": flipped_queries,
                  "failure_set_differences": {f"r{k}": v for k, v in failure_set_differences.items()},
                  "failure_overlap_equal": {f"r{k}": v for k, v in failure_overlap_equal.items()},
                  "native_hits": {f"r{k}": value for k, value in native_hits.items()},
                  "adapter_hits": {f"r{k}": value for k, value in adapter_hits.items()},
                  "aggregate_hit_counts_equal": native_hits == adapter_hits,
                  "failure_overlap_equivalence": failure_overlap_equal}

    return [ParityCheck(f"{category}.scores.raw_rows", "scoring_parity", "G2",
                        score_tol, score_max, score_max <= score_tol, mandatory=True,
                        standard=STANDARD_ABS_REL_V3),
            ParityCheck(f"{category}.ranks.raw_rows", "scoring_parity", "G2",
                        "all inverted score-pair gaps within preregistered tie_epsilon",
                        diagnostic, inversion_gap_violations == 0, mandatory=True, standard=STANDARD_ABS_REL_V3)]


def _v3_checks(native: dict[str, Any], adapter: dict[str, Any], category: str) -> list[ParityCheck]:
    block_side = "native" if V3_KEY in native else "adapter"
    artifact = native if block_side == "native" else adapter
    block = artifact[V3_KEY]
    plan, _ = _v3_manifest(native, adapter)
    declared_controls = plan["controls"]
    records = block["scopes"]
    artifact_checks = []
    for control_name in V3_CONTROLS:
        control = declared_controls[control_name]
        for scope_name in ("query", "gallery"):
            record = records[control_name][scope_name]
            scope_plan = control["scopes"][scope_name]
            if record.get("reference_run_id") != control["reference_run_id"] or record.get("actual_run_id") != control["actual_run_id"]:
                raise _Malformed(f"v3 scope {control_name}.{scope_name} run IDs differ from preregistered plan")
            if record.get("dtype") != control["dtype"] or record.get("dimension") != control["dimension"]:
                raise _Malformed(f"v3 scope {control_name}.{scope_name} dtype/dimension differ from plan")
            if record.get("shape") != scope_plan.get("component_shape"):
                raise _Malformed(f"v3 scope {control_name}.{scope_name} shape differs from preregistered population")
            if record.get("coverage") not in COMPONENT_COVERAGE:
                raise _Malformed(f"v3 scope {control_name}.{scope_name} has unsupported coverage evidence")
            ref_role = next(role for role, value in plan["runs"].items() if value["run_id"] == control["reference_run_id"])
            act_role = next(role for role, value in plan["runs"].items() if value["run_id"] == control["actual_run_id"])
            for role, digest_key in ((ref_role, "reference_digest"), (act_role, "actual_digest")):
                receipt = block["receipts"][role]
                arr = receipt.get("arrays", {}).get(f"{control_name}.{scope_name}")
                if (not isinstance(arr, dict) or arr.get("dtype") != control["dtype"]
                        or arr.get("shape") != scope_plan.get("tensor_shape")
                        or arr.get("component_sha256") != record.get("sha256", {}).get(digest_key.removesuffix("_digest"))
                        or arr.get("ordered_ids_sha256") != scope_plan.get("ordered_ids_sha256")):
                    raise _Malformed(f"v3 receipt {role} does not bind {control_name}.{scope_name} captured data/order")
            tol = control["tolerance"]
            declared = {"forward_max_abs": tol["atol"], "forward_max_rel": tol["rtol"],
                        "batch_max_abs": tol["atol"], "batch_max_rel": tol["rtol"]}
            checked = _component_scope_check(f"{control_name}.{scope_name}", record, declared, category,
                                             standard=STANDARD_ABS_REL_V3,
                                             tolerances=V3_COMPONENT_TOLERANCES)
            complete = checked.measured.get("complete_coverage") is True
            mismatch = bool(checked.measured["mismatch_count"])
            batch_sensitivity = "batch_1_vs_n" in control_name
            if not complete and not batch_sensitivity:
                if mismatch:
                    checked = replace(checked, mandatory=True, ok=False, absence=False,
                                      note="Measured preview violation; preregistered scope remains incomplete")
                else:
                    checked = replace(checked, mandatory=False, ok=False, absence=True,
                                      note="PREVIEW ONLY: captured evidence incomplete")
            reference_execution = next(run["execution"] for run in plan["runs"].values()
                                       if run["run_id"] == control["reference_run_id"])
            actual_execution = next(run["execution"] for run in plan["runs"].values()
                                    if run["run_id"] == control["actual_run_id"])
            checked = replace(checked, declared={**checked.declared,
                                "coverage": scope_plan["coverage"],
                                "population_count": scope_plan["population_count"],
                                "ordered_ids_sha256": scope_plan["ordered_ids_sha256"]},
                              measured={**checked.measured,
                                "coverage_scope": scope_plan["coverage"],
                                "population_count": scope_plan["population_count"],
                                "ordered_ids_sha256": scope_plan["ordered_ids_sha256"],
                                "sample_count": scope_plan.get("sample_count"),
                                "tensor_shape": scope_plan["tensor_shape"],
                                "dtype": control["dtype"], "control": control_name,
                                "batch_policy": ({**plan["supported_batch_policy"],
                                                  "reference_execution": reference_execution,
                                                  "actual_execution": actual_execution}
                                                 if batch_sensitivity else None)})
            if batch_sensitivity:
                policy = plan["supported_batch_policy"]
                if not isinstance(policy, dict) or policy.get("supported") is not True or not policy.get("batch_sizes"):
                    raise _Malformed("batch sensitivity requires an explicit supported batch policy")
                invariance_required = policy.get("batch_invariant", False)
                checked = replace(checked, mandatory=invariance_required,
                                  ok=checked.ok if invariance_required else False,
                                  absence=not complete and not mismatch,
                                  note=(checked.note or "") + (
                                      "; preregistered contract requires batch invariance"
                                      if invariance_required else
                                      "; observed batch sensitivity is diagnostic, not a parity PASS"
                                      if mismatch else
                                      "; batch sensitivity is diagnostic, not a parity veto"))
                artifact_checks.append(ParityCheck(
                    name=f"{category}.components.{control_name}.{scope_name}.coverage",
                    dimension="batch_parity", gate="G2",
                    declared={"coverage": scope_plan["coverage"], "population_count": scope_plan["population_count"]},
                    measured={"complete_coverage": complete, "coverage_scope": scope_plan["coverage"],
                              "population_count": scope_plan["population_count"],
                              "sample_count": scope_plan.get("sample_count"),
                              "tensor_shape": scope_plan["tensor_shape"], "dtype": control["dtype"],
                              "control": control_name, "batch_policy": policy},
                    ok=complete, mandatory=True, absence=not complete,
                    standard=STANDARD_ABS_REL_V3,
                    note="batch control coverage is mandatory; sensitivity is diagnostic unless invariance was predeclared"))
            artifact_checks.append(checked)
    return artifact_checks + _v3_score_rank_checks(native, adapter, _tolerances(native, adapter), plan, category)

def _component_scopes(native: dict[str, Any], adapter: dict[str, Any]) -> list[tuple[str, Any]]:
    """Every `(scope name, record)` the pair records, adapter first, and a refusal on a contradiction.

    The block is written where both arrays were held, so it normally sits on ONE side. Two sides may
    each carry one (a future producer that compares on the native side too); then a scope named on
    both must record the SAME bytes of comparison, because two records of one comparison that
    disagree are an artifact contradicting itself, not a second measurement.
    """
    scopes: dict[str, Any] = {}
    for side, artifact in (("adapter", adapter), ("native", native)):
        block = artifact.get(COMPONENT_KEY)
        if block is None:
            continue
        where = f"the {side} artifact's `{COMPONENT_KEY}`"
        if not isinstance(block, dict):
            raise _Malformed(f"{where} is a {type(block).__name__}, not an object")
        declared = block.get("standard", STANDARD_ABS_REL_V2)
        if declared != STANDARD_ABS_REL_V2:
            raise _Malformed(f"{where}.standard is {declared!r}: the block records components of "
                             f"{STANDARD_ABS_REL_V2} and nothing else")
        rule = block.get("declared_rule")
        if rule is not None and rule != COMPONENT_RULE:
            raise _Malformed(f"{where}.declared_rule is {rule!r}, not the declared v2 rule "
                             f"{COMPONENT_RULE!r}: a block cannot restate the rule it is judged by")
        reference = block.get("reference_side")
        if reference is not None and reference != "native":
            raise _Malformed(f"{where}.reference_side is {reference!r}: v2 makes the NATIVE side the "
                             "reference, and a block cannot swap them after the fact")
        records = block.get("scopes")
        if not isinstance(records, dict):
            raise _Malformed(f"{where}.scopes is a {type(records).__name__}, not an object of scopes")
        for name, record in records.items():
            if name in scopes and scopes[name] != record:
                raise _Malformed(f"the two artifacts record different components for scope {name!r}; "
                                 "one comparison cannot have two records")
            scopes[name] = record
    return list(scopes.items())

def _component_checks(native: dict[str, Any], adapter: dict[str, Any],
                      declared: dict[str, Any], category: str) -> list[ParityCheck]:
    """The v2 componentwise subchecks: one mandatory check per recorded scope, rule applied per element.

    Called only for a pair that DECLARED v2. A pair that declares v2 and records no scope is
    `_component_scopes`' refusal, so this never returns `[]` for a declared-v2 pair.
    """
    scopes = _component_scopes(native, adapter)
    if not scopes:
        raise _Malformed(_V2_WITHOUT_COMPONENTS)
    return [_component_scope_check(name, record, declared, category) for name, record in scopes]


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
            native["_artifact_path"] = str(native_path)
            adapter["_artifact_path"] = str(adapter_path)
            verdict.evidence.extend([str(native_path), str(adapter_path)])
            declared = _tolerances(native, adapter)
            applied = _declared_standard(native, adapter)
            standards.add(applied)
            native_body = _body(native, side="native")
            adapter_body = _body(adapter, side="adapter")

            checks: list[ParityCheck] = []
            # The v2 componentwise subchecks, only for a pair that DECLARED v2 (a v1 pair carries no
            # block and earns no v2 check, whatever standard the caller requested).
            if applied == STANDARD_ABS_REL_V2:
                checks.extend(_component_checks(native, adapter, declared, category))
            elif applied == STANDARD_ABS_REL_V3:
                checks.extend(_v3_checks(native, adapter, category))
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
                if applied != STANDARD_ABS_REL_V3:
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
                if applied != STANDARD_ABS_REL_V3:
                    checks.extend(_relative_checks(
                        block, declared, name=f"{category}.population.{side}",
                        dimension="forward_parity", gate="G2",
                        atol_key="forward_max_abs", rtol_key="forward_max_rel"))
            batch = adapter_body.get("batch_size_1_vs_n")
            if batch is not None:
                if not isinstance(batch, dict):
                    raise _Malformed(f"{category}.batch_size_1_vs_n is a "
                                     f"{type(batch).__name__}, not an object")
                if applied != STANDARD_ABS_REL_V3:
                    checks.extend(_relative_checks(
                        batch, declared, name=f"{category}.batch_1_vs_n", dimension="batch_parity",
                        gate="G2", atol_key="batch_max_abs", rtol_key="batch_max_rel"))
            if applied != STANDARD_ABS_REL_V3:
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
            # The metric checks are collected BEFORE the demotion below, so an absence-reporting
            # metric check on an adapter that claims nothing is demoted like every other absence
            # rather than reported as a failed claim. (A `true` flag with no counts behind it, and a
            # comparison whose provenance is unstated, are both absences: the artifact did not measure
            # the thing, it failed to record the inputs.)
            checks.extend(_metrics_checks(native, adapter, category))
            checks.extend(_observer_checks(native_body, category))
            claims_present = bool(adapter_body)
            if not claims_present:
                checks = [check if not check.absence else _unclaimable(check) for check in checks]

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
            checks = [replace(check, category=category) for check in checks]
            verdict.checks.extend(checks)
            # Per-category mandatory coverage. A required dimension measured in `shirt` is not
            # measured in `dress`: an adapter body of `{}` in one category used to be carried by the
            # other categories' checks, and the verdict reported PASS.
            have = {check.dimension for check in checks if check.mandatory}
            if score_missing:
                # Rank evidence alone: keep the rank checks (they are real), but the dimension was
                # never fully measured here, so it is `missing` rather than satisfied.
                have.discard("scoring_parity")
            # A dimension carrying an absence-reporting check that is not ok was NOT measured, even
            # when other checks of the same dimension exist. Without this a clean 512-of-4096
            # component preview was masked by the aggregate `max_abs`/`max_rel` checks that also feed
            # `forward_parity`, and the verdict read PASS on evidence about 12% of the array.
            unmeasured = {check.dimension for check in checks if check.absence and not check.ok}
            failed = {check.dimension for check in checks if check.mandatory and not check.ok}
            missing_here = [dimension for dimension in REQUIRED_DIMENSIONS
                            if dimension not in have
                            or (dimension in unmeasured and dimension not in failed)]
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
                     if standard in (STANDARD_ABS_REL_V2, STANDARD_ABS_REL_V3) else None),
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
    """Every model's verdict, MEMOISED on the artifacts' own `(mtime_ns, size)` stamp.

    Deriving this costs ~2.9 s and ~270 MiB because complete component coverage means reading and
    checking 7.77 M components; `/api/runs/parity-gate` is fetched on every mount of the parity page,
    so deriving it per request blocked the UI for seconds and allocated the same 270 MiB per
    concurrent request. The derivation is a pure function of the artifacts, the requested standard and
    the root, so the answer for a given stamp is the SAME ANSWER -- recomputing it is pure cost.

    Three properties this keeps, in the order they matter:

    1. **It cannot go stale.** The key is a stat of every artifact the gate reads (see
       `_artifact_stamp`), so any file that appears, disappears or is rewritten produces a different
       key and a fresh derivation. There is deliberately NO TTL: a clock cannot tell you whether a
       file changed, so a time-based expiry would introduce staleness that the stamp rules out
       exactly.
    2. **It cannot be poisoned.** The cached object is never returned; every call gets a
       `copy.deepcopy`, so a caller mutating the payload cannot affect the next request. The copy
       costs 0.5 ms against the 2.9 s derivation.
    3. **It degrades to correct, not to fast.** If the evidence root cannot be listed, the stamp is
       unavailable, and the call derives without caching -- an unreadable directory must never be
       remembered as "no artifacts".

    `evaluate_all_uncached` remains the plain derivation, and the tests compare against it directly.
    """
    publication_root = None if root is not None else Path(__file__).resolve().parents[2].joinpath(*PARITY_V3_PUBLISHED_RELATIVE)
    pointer = publication_root / "CURRENT.json" if publication_root is not None else None
    publication_error = None
    if root is not None:
        bases = (Path(root),)
    else:
        legacy_base = evidence_root()
        try:
            v3_base = v3_published_root()
        except _Malformed as error:
            v3_base = Path(__file__).resolve().parents[2].joinpath(*PARITY_V3_PUBLISHED_RELATIVE, "generations", "unpublished")
            publication_error = str(error)
        bases = (legacy_base, v3_base)
    try:
        if any(not base.is_dir() for base in bases):
            return evaluate_all_uncached(root=root, standard=standard)
        pointer_stamp = ((pointer.stat().st_mtime_ns, pointer.stat().st_size)
                         if pointer is not None and pointer.is_file() else (-1, -1))
        stamp: tuple | None = (tuple(str(base) for base in bases), standard, pointer_stamp,
                               tuple(_artifact_stamp(base) for base in bases))
    except OSError:
        return evaluate_all_uncached(root=root, standard=standard)
    if _EVALUATE_ALL_CACHE and _EVALUATE_ALL_CACHE[0][0] == stamp:
        return copy.deepcopy(_EVALUATE_ALL_CACHE[0][1])
    result = evaluate_all_uncached(root=root, standard=standard)
    if publication_error:
        result["publication_error"] = publication_error
    _EVALUATE_ALL_CACHE.clear()
    _EVALUATE_ALL_CACHE.append((stamp, result))
    return copy.deepcopy(result)


def evaluate_all_uncached(*, root: Path | None = None,
                          standard: str = STANDARD_RELATIVE_V1) -> dict[str, Any]:
    """Every model with G1/G2 evidence on disk, each judged on its own artifacts.

    `standard` is reported as the standard actually APPLIED (the artifacts' own). A request that
    differs is reported beside it, so a caller cannot mistake a v2 request for a v2 result.

    `standard` is ALWAYS a string, and `standards` is the breakdown. These answer different
    questions, and conflating them broke both a render and a test:

    * `standard` -- ONE label, for a surface that has room for one string
      (`Chuẩn ${gate.standard}`). Chosen DETERMINISTICALLY: the standard of the FIRST model in
      sorted model-id order whose verdict is not `NOT_RUN`; `parity-relative/v1` when no model
      yielded a verdict at all. It must not depend on which artifacts happen to be on disk, so it is
      not "the majority standard" (a majority flips when one pair lands) and not a list.
    * `standards` -- `{standard: [model_id, ...]}`, every model grouped under the standard its OWN
      artifacts declared. This is the honest aggregate: a mixed evidence directory is normal while a
      campaign migrates, and a per-model answer already exists in `models[*].applied_standard`.

    A list in `standard` used to appear the moment a second standard existed among the verdicts,
    because the slot held `applied[0] if len(applied) == 1 else applied`. The frontend types this
    field as `string` and interpolates it, so that was a live defect, not a cosmetic one.
    """
    publication_error = None
    if root is None:
        legacy_base = evidence_root()
        try:
            v3_base = v3_published_root()
        except _Malformed as error:
            v3_base = Path(__file__).resolve().parents[2].joinpath(*PARITY_V3_PUBLISHED_RELATIVE, "generations", "unpublished")
            publication_error = str(error)
        models = set(_discover_models(legacy_base)) | set(_discover_models(v3_base))
        verdicts = {}
        for model_id in sorted(models):
            # Versioned V3 publication overrides only same-model evidence; legacy files remain
            # immutable and directly reproducible via evaluate_all(root=evidence_root()).
            base = v3_base if _artifact_paths(v3_base, model_id, "dress") else legacy_base
            verdicts[model_id] = evaluate_model(model_id, root=base, standard=standard).to_dict()
    else:
        base = Path(root)
        verdicts = {model_id: evaluate_model(model_id, root=base, standard=standard).to_dict()
                    for model_id in _discover_models(base)}
    models = sorted(verdicts)
    standards: dict[str, list[str]] = {}
    for model_id in models:
        verdict = verdicts[model_id]
        if verdict["status"] == NOT_RUN:
            continue
        standards.setdefault(verdict["applied_standard"], []).append(model_id)
    applied = sorted(standards)
    # Sorted model order, not artifact arrival order: a new pair appearing on disk cannot move the
    # label unless it sorts before the current first model.
    label = next((verdicts[model_id]["applied_standard"] for model_id in models
                  if verdicts[model_id]["status"] != NOT_RUN), STANDARD_RELATIVE_V1)
    result = {
        "standard": label,
        "standards": {key: standards[key] for key in applied},
        "requested_standard": standard,
        "advisory_standards": ([STANDARD_ABS_REL_V2] if STANDARD_ABS_REL_V2 not in applied else []),
        "passed_models": sorted(model_id for model_id, verdict in verdicts.items() if verdict["passed"]),
        "models": verdicts,
        "note": ("`passed` is True only when every mandatory dimension was measured and no mandatory "
                 f"subcheck failed. No artifact is {NOT_RUN}; nothing measured held is {INCOMPLETE}; "
                 f"a measured subcheck outside its tolerance is {MANDATORY_FAILED}. Only {PASS} "
                 "claims a pass. `standard` is ONE label -- the standard of the first model in "
                 "sorted order with a verdict -- and `standards` groups every model under the "
                 "standard its own artifacts declared."),
    }
    if publication_error:
        result["publication_error"] = publication_error
    return result
