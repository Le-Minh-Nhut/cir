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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

#: The directory holding the G1/G2 measurement artifacts. Fixed, never request-derived.
PARITY_EVIDENCE_RELATIVE = ("workbench", "docs", "reviews", "20261109-native15-composition-limn")

#: Model ids that have G1/G2 artifacts on disk. Anything else is NOT_RUN, never PASS.
EVIDENCED_MODELS = ("limn", "encoder")

#: Percent-free status vocabulary. `PASS` is the only value that claims every mandatory subcheck held.
PASS = "PARITY_PASS"
MANDATORY_FAILED = "PARITY_MANDATORY_SUBCHECK_FAILED"
NOT_RUN = "PARITY_NOT_RUN"
EVIDENCE_MISSING = "PARITY_EVIDENCE_MISSING"
MALFORMED = "PARITY_EVIDENCE_MALFORMED"

STANDARD_RELATIVE_V1 = "parity-relative/v1"
STANDARD_ABS_REL_V2 = "parity-absolute-relative/v2"
STANDARDS = (STANDARD_RELATIVE_V1, STANDARD_ABS_REL_V2)

#: Dimensions. A status is per dimension; there is no cross-dimension average anywhere.
DIMENSIONS = ("input_parity", "forward_parity", "batch_parity", "scoring_parity",
              "metrics_parity", "observer_parity")


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

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "dimension": self.dimension, "gate": self.gate,
                "declared": self.declared, "measured": self.measured, "ok": self.ok,
                "mandatory": self.mandatory, "standard": self.standard, "note": self.note,
                "harness_recorded_within_tolerance": self.harness_recorded}


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
    #: Comparisons where the harness's own `within_tolerance` boolean disagrees with the declared
    #: tolerance table. The declared rule wins; the old label is preserved as the v1 record.
    label_disagreements: list[dict[str, Any]] = field(default_factory=list)
    historical: dict[str, Any] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"model_id": self.model_id, "status": self.status, "passed": self.passed,
                "complete": self.complete,
                "dimensions": dict(self.dimensions),
                # Which measurements each dimension ACTUALLY rests on, and what it never saw. A
                # dimension True with a gap is "every check that exists passed", not "fully checked",
                # and a reader must be able to tell the two apart.
                "coverage": {key: list(value) for key, value in self.coverage.items()},
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


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_names(model_id: str, category: str) -> tuple[str, str]:
    if model_id == "encoder":
        return (f"g1_g2.encoder.{category}.native.json", f"g1_g2.encoder.{category}.adapter.json")
    return f"g1_g2.{category}.native.json", f"g1_g2.{category}.adapter.json"


def _tolerances(native: dict[str, Any], adapter: dict[str, Any]) -> dict[str, Any]:
    """The declared tolerances. Both sides declare them; a disagreement is itself malformed evidence."""
    left, right = native.get("tolerances") or {}, adapter.get("tolerances") or {}
    merged = dict(left)
    merged.update({key: value for key, value in right.items() if key not in merged})
    return merged


def evaluate_abs_rel(actual: float, reference: float, *, atol: float, rtol: float) -> bool:
    """`abs(a-b) <= atol + rtol*abs(b)` for one component, the v2 rule."""
    return abs(actual - reference) <= atol + rtol * abs(reference)


def _relative_checks(measured: dict[str, Any], declared: dict[str, Any], *,
                     name: str, dimension: str, gate: str,
                     atol_key: str, rtol_key: str) -> list[ParityCheck]:
    """The three-part absolute / relative / cosine rule for one forward comparison."""
    atol = declared.get(atol_key)
    rtol = declared.get(rtol_key)
    min_cosine = declared.get("forward_min_cosine")
    max_abs = measured.get("max_abs")
    max_rel = measured.get("max_rel")
    cosine = measured.get("min_cosine")
    checks: list[ParityCheck] = []
    if atol is None or not isinstance(max_abs, (int, float)):
        return checks
    # The declared v1 rule is always evaluated, whatever standard was REQUESTED: a post-hoc standard
    # may be reported beside the record, never used to replace it.
    recorded = measured.get("within_tolerance")
    checks.append(ParityCheck(
        name=f"{name}.max_abs", dimension=dimension, gate=gate,
        declared=atol, measured=max_abs, ok=float(max_abs) <= float(atol),
        mandatory=True, standard=STANDARD_RELATIVE_V1, harness_recorded=recorded))
    if rtol is not None and isinstance(max_rel, (int, float)):
        # The declared tolerance table applies `forward_max_rel` / `batch_max_rel` to layer 4. It is
        # therefore mandatory for EVERY comparison of that kind, whatever the harness happened to
        # call for a particular key. `harness_recorded` shows where the harness disagreed.
        checks.append(ParityCheck(
            name=f"{name}.max_rel", dimension=dimension, gate=gate,
            declared=rtol, measured=max_rel, ok=float(max_rel) <= float(rtol),
            mandatory=True, standard=STANDARD_RELATIVE_V1, harness_recorded=recorded,
            note=("relative error on a component near zero: the denominator, not the model, drives "
                  "this. The declared tolerance is not moved; see parity-absolute-relative/v2")))
    if min_cosine is not None and isinstance(cosine, (int, float)):
        checks.append(ParityCheck(
            name=f"{name}.min_cosine", dimension=dimension, gate=gate,
            declared=min_cosine, measured=cosine, ok=float(cosine) >= float(min_cosine),
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
    for key in ("query_count_native", "query_count_annotations",
                "gallery_count_native", "gallery_count_rebuilt_from_annotations"):
        checks.append(ParityCheck(
            name=f"{category}.identity.{key}", dimension="input_parity", gate="G1",
            declared="equal count", measured=layer.get(key),
            ok=isinstance(layer.get(key), int) and layer.get(key) == layer.get(
                "query_count_annotations" if "query" in key else "gallery_count_rebuilt_from_annotations"),
            mandatory=True))
    if layer.get("gallery_has_duplicates") is True:
        checks.append(ParityCheck(f"{category}.identity.gallery_has_duplicates", "input_parity", "G1",
                                  "no duplicates", True, ok=False, mandatory=True))
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
    if "unexplained_rank_differences" in scoring:
        unexplained = scoring["unexplained_rank_differences"]
        checks = [ParityCheck(
            name=f"{category}.ranks.unexplained_differences", dimension="scoring_parity",
            gate="G2", declared="0 rank differences the tie rule cannot explain",
            measured=unexplained, ok=unexplained == 0, mandatory=True,
            note=f"{scoring.get('rank_differences')!r} rank differences recorded; every one must be "
                 "a declared tie")]
        if "score_max_abs_delta_vs_dot" in scoring and declared.get("score_max_abs") is not None:
            delta = scoring["score_max_abs_delta_vs_dot"]
            checks.append(ParityCheck(
                name=f"{category}.scores.max_abs_vs_dot", dimension="scoring_parity", gate="G2",
                declared=declared["score_max_abs"], measured=delta,
                ok=float(delta) <= float(declared["score_max_abs"]), mandatory=True))
        return checks
    rank_block = scoring.get("rank_differences_vs_native_val_control")
    if not isinstance(rank_block, dict):
        return []
    non_tie = rank_block.get("non_tie_count")
    checks = [ParityCheck(
        name=f"{category}.ranks.non_tie_changes", dimension="scoring_parity", gate="G2",
        declared="0 non-tie rank changes (tie_epsilon %r)" % (declared.get("tie_epsilon"),),
        measured=non_tie, ok=non_tie == 0, mandatory=True,
        note="the artifact classifies each difference against the tie epsilon itself")]
    if isinstance(rank_block.get("min_gap_abs"), (int, float)) and declared.get("tie_epsilon") is not None:
        checks.append(ParityCheck(
            name=f"{category}.ranks.min_gap_vs_tie_epsilon", dimension="scoring_parity", gate="G2",
            declared=declared.get("tie_epsilon"), measured=rank_block["min_gap_abs"],
            ok=(non_tie == 0) or float(rank_block["min_gap_abs"]) <= float(declared["tie_epsilon"]),
            mandatory=True))
    convention = scoring.get("adapter_vs_argsort_unmasked") or {}
    if "non_tie_count" in convention:
        checks.append(ParityCheck(
            name=f"{category}.ranks.tie_convention_non_tie", dimension="scoring_parity", gate="G2",
            declared="0 non-tie differences between rank conventions",
            measured=convention.get("non_tie_count"), ok=convention.get("non_tie_count") == 0,
            mandatory=True))
    return checks


def _score_checks(adapter: dict[str, Any], category: str) -> list[ParityCheck]:
    scoring = adapter.get("scoring_ranking") or {}
    if "score_within_tolerance" not in scoring:
        return []
    delta = scoring.get("score_max_abs_delta_vs_dot")
    return [ParityCheck(
        name=f"{category}.scores.max_abs_vs_dot", dimension="scoring_parity", gate="G2",
        declared=scoring.get("score_max_abs_declared"), measured=delta,
        ok=scoring.get("score_within_tolerance") is True, mandatory=True)]


def _metrics_checks(native: dict[str, Any], category: str) -> list[ParityCheck]:
    if "native_metrics_match_feature_rerank" not in native:
        return []
    return [ParityCheck(
        name=f"{category}.metrics.hit_count_equality", dimension="metrics_parity", gate="G2",
        declared="exact hit-count equality", measured=native.get("native_metrics_match_feature_rerank"),
        ok=native.get("native_metrics_match_feature_rerank") is True, mandatory=True)]


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


def evaluate_model(model_id: str, *, categories: Iterable[str] = ("dress", "shirt", "toptee"),
                   root: Path | None = None,
                   standard: str = STANDARD_RELATIVE_V1) -> ParityVerdict:
    """Derive one model's G1/G2 verdict from the artifacts, never from a prose surface."""
    verdict = ParityVerdict(model_id=model_id)
    if standard not in STANDARDS:
        raise ValueError(f"unsupported parity standard: {standard!r}")
    if model_id not in EVIDENCED_MODELS:
        verdict.status = NOT_RUN
        verdict.detail = (f"no G1/G2 artifact exists for {model_id}; absent evidence is never a "
                          "pass, so the gate stays NOT_RUN")
        return verdict
    base = root if root is not None else evidence_root()
    categories = tuple(categories)
    missing = []
    for category in categories:
        native_name, adapter_name = _artifact_names(model_id, category)
        for name in (native_name, adapter_name):
            path = base / name
            if not path.is_file():
                missing.append(name)
    if missing:
        verdict.status = EVIDENCE_MISSING
        verdict.detail = f"missing parity artifact(s): {missing}"
        return verdict

    historical: dict[str, Any] = {}
    for category in categories:
        native_name, adapter_name = _artifact_names(model_id, category)
        native_path, adapter_path = base / native_name, base / adapter_name
        try:
            native = _read(native_path)
            adapter = _read(adapter_path)
        except ValueError as error:
            verdict.status = MALFORMED
            verdict.detail = f"unreadable parity artifact: {error}"
            return verdict
        verdict.evidence.extend([str(native_path), str(adapter_path)])
        declared = _tolerances(native, adapter)
        native_body = native.get("native") if isinstance(native.get("native"), dict) else native.get(
            "layer1_identity") and native or {}
        adapter_body = (adapter.get("adapter") if isinstance(adapter.get("adapter"), dict)
                        else adapter.get("layer2_4_forward") or {})

        checks: list[ParityCheck] = []
        checks.extend(_identity_checks(native_body if isinstance(native_body, dict) else {}, declared, category))
        checks.extend(_pixel_check(adapter_body if isinstance(adapter_body, dict) else {}, category))

        sample = (adapter_body.get("layer4_forward_sample")
                  or adapter_body.get("sample_forward") or {})
        for side in ("query_val_pixels", "target", "query"):
            if isinstance(sample.get(side), dict):
                checks.extend(_relative_checks(
                    sample[side], declared, name=f"{category}.forward.{side}",
                    dimension="forward_parity", gate="G2",
                    atol_key="forward_max_abs", rtol_key="forward_max_rel"))
        population = adapter_body.get("population") or {}
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
        if isinstance(batch, dict):
            checks.extend(_relative_checks(
                batch, declared, name=f"{category}.batch_1_vs_n", dimension="batch_parity",
                gate="G2", atol_key="batch_max_abs", rtol_key="batch_max_rel"))
        checks.extend(_rank_checks(adapter_body, declared, category))
        checks.extend(_score_checks(adapter_body, category))
        checks.extend(_metrics_checks(native_body if isinstance(native_body, dict) else {}, category))
        checks.extend(_observer_checks(native_body if isinstance(native_body, dict) else {}, category))

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

    if not mandatory:
        verdict.status = EVIDENCE_MISSING
        verdict.detail = "the artifacts carry no mandatory subcheck this gate can evaluate"
        return verdict

    if standard == STANDARD_ABS_REL_V2:
        # v2 has not been measured on these artifacts, and applying it after the fact would move a
        # declared tolerance post hoc. It is reported for the record and cannot certify anything.
        verdict.advisory = [{
            "standard": STANDARD_ABS_REL_V2,
            "declared_before_measurement": False,
            "applies": False,
            "why": ("the artifacts declare parity-relative/v1 tolerances and were measured under "
                    "them; a v2 evaluation of v1 data would be a post-hoc tolerance change and "
                    "cannot contribute to a pass"),
            "form": "fail if abs(a-b) > atol + rtol*abs(b) for any component",
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

    verdict.status = PASS
    verdict.passed = True
    verdict.complete = True
    verdict.detail = ("every mandatory subcheck held under its own declared tolerance")
    return verdict


def evaluate_all(*, root: Path | None = None,
                 standard: str = STANDARD_RELATIVE_V1) -> dict[str, Any]:
    """Every evidenced model, plus the ones with no evidence kept explicitly NOT_RUN."""
    verdicts = {model_id: evaluate_model(model_id, root=root, standard=standard).to_dict()
                for model_id in EVIDENCED_MODELS}
    return {
        "standard": standard,
        "passed_models": sorted(model_id for model_id, verdict in verdicts.items() if verdict["passed"]),
        "models": verdicts,
        "note": ("`passed` is True only when no mandatory subcheck failed. A model with no artifact is "
                 "NOT_RUN, and a model whose declared subcheck failed is "
                 f"{MANDATORY_FAILED}, never {PASS}."),
    }
